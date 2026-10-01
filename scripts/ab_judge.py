"""J4 — blind A/B judging against a live dashboard + judge endpoint.

Reads stored eval results for a suite, groups them per item (model A/B),
randomizes presentation order, sends an anonymized pairwise prompt to a
judge endpoint, parses the verdict, and stores a judgment via the API.

Usage:
    venv/bin/python scripts/ab_judge.py [--api http://localhost:24816]
        --suite-id 1 --judge-url http://your-judge-host:8000
        [--judge-model <name>] [--template blind-pair-v1]

--judge-url may also come from arcturos.local.toml ([demo] judge_url) or
the ARCTUROS_DEMO__JUDGE_URL environment variable.

CAVEAT printed at runtime: judge must not be a contestant for trustworthy
results. Enforced with --allow-contestant-judge to override.
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from urllib import request as urlreq
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arcturos import config

JUDGE_PROMPT_TEMPLATE = """You are a blind preference judge. Two anonymous responses to the
same prompt follow. Judge ONLY response quality: correctness, instruction
following, clarity. Do not guess model identities.

[PROMPT]
{prompt}

[RESPONSE 1]
{resp1}

[RESPONSE 2]
{resp2}

Which response is better? Reply with EXACTLY one word: "1", "2", or "tie".
"""


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("ARCTUROS_JUDGE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urlreq.Request(url, data=json.dumps(payload).encode(),
                         headers=headers)
    with urlreq.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_json(url: str) -> dict:
    with urlreq.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def serialize_multiturn(item: dict, outputs_a: list[str], outputs_b: list[str]) -> str:
    """Render a multi-turn item + both models' assistant turns as judge text."""
    turn_no = 0
    lines = []
    for t in item.get("turns", []):
        if t["role"] == "user":
            lines.append(f"USER: {t.get('content', '')}")
        elif t["role"] == "assistant":
            turn_no += 1
            idx = turn_no - 1
            a = outputs_a[idx] if idx < len(outputs_a) else "(missing)"
            b = outputs_b[idx] if idx < len(outputs_b) else "(missing)"
            lines.append(f"ASSISTANT A (response A): {a}")
            lines.append(f"ASSISTANT B (response B): {b}")
    return "\n".join(lines)


def _normalize_base(url: str) -> str:
    """Accept both http://host:port and http://host:port/v1 bases."""
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def normalize_fingerprint(fp: str) -> str:
    """Collapse server-suffix variants: 'Model (host)' / 'Model' -> 'Model'.

    Model identity is the model; host is metadata. Without this, re-seeding
    under a slightly different label creates phantom third contestants.
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", fp.strip()).strip()


def _extract_verdict(raw: str) -> str | None:
    """Find 1/2/tie verdict, tolerating reasoning-model chatter."""
    tokens = re.findall(r"\b(1|2|tie)\b", raw.lower())
    if not tokens:
        return None
    return tokens[-1]  # final verdict is the last one mentioned


def judge_pair(judge_url: str, judge_model: str, prompt: str,
               resp1: str, resp2: str, template_version: str) -> dict:
    """Send blind pair to judge; returns parsed verdict dict."""
    payload_text = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, resp1=resp1, resp2=resp2)
    resp = post_json(_normalize_base(judge_url) + "/v1/chat/completions", {
        "model": judge_model,
        "messages": [{"role": "user", "content": payload_text}],
        "max_tokens": 512,  # reasoning models burn tokens before answering
        "temperature": 0.0,
        "stream": False,
    })
    msg = resp["choices"][0]["message"]
    raw = (msg.get("content") or "").strip()
    if not raw:
        # reasoning models: final answer may land in reasoning_content tail
        raw = (msg.get("reasoning_content") or "").strip()
    winner = _extract_verdict(raw)
    if winner is None:
        return {"valid": False, "raw": raw[:200]}
    return {"valid": True, "raw": raw[-120:], "winner": winner,
            "template_version": template_version}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:24816")
    ap.add_argument("--suite-id", type=int, required=True)
    ap.add_argument("--judge-url", default=None,
                    help="falls back to [demo] judge_url in arcturos.local.toml / arcturos.toml")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--template", default="blind-pair-v1")
    ap.add_argument("--allow-contestant-judge", action="store_true")
    args = ap.parse_args()

    # --judge-url falls back to the config layer (local overlay > template)
    if not args.judge_url:
        args.judge_url = config.get("demo", "judge_url", None)
    if not args.judge_url:
        ap.error("--judge-url is required (or set [demo] judge_url in arcturos.local.toml)")

    # load suite items for category + prompts
    suite_file = ROOT / "qa" / "suites" / "smoke-reasoning-v1.json"
    suite = json.loads(suite_file.read_text())
    item_by_id = {i["id"]: i for i in suite["items"]}

    results = get_json(f"{args.api}/api/eval-suites/{args.suite_id}/results")
    by_item: dict[str, list] = {}
    latest_by_model: dict[str, dict] = {}
    for r in sorted(results, key=lambda x: x["id"]):
        # append-only store: the LAST result per (item, model) supersedes;
        # model identity is the normalized fingerprint (server suffix stripped)
        latest_by_model[(r["item_id"], normalize_fingerprint(r["model_fingerprint"]))] = r
    for (_item_id, _model), r in latest_by_model.items():
        by_item.setdefault(r["item_id"], []).append(r)

    model_names = set()
    for r in results:
        model_names.add(r["model_fingerprint"])
    if args.judge_model:
        judge_is_contestant = any(args.judge_model in n for n in model_names)
        if judge_is_contestant and not args.allow_contestant_judge:
            print("!! judge is also a contestant:", args.judge_model)
            print("   results are NOT trustworthy preference signal.")
            print("   re-run with --allow-contestant-judge to force.")
            sys.exit(2)

    print(f"judging {len(by_item)} items with judge {args.judge_model or '(server default)'}"
          f" @ {args.judge_url}")
    stored = 0
    judged_base_ids: set = set()
    for item_id, _ in sorted(by_item.items()):
        base_id = item_id.split("#")[0]  # multi-turn turn records share base id
        if base_id in judged_base_ids:
            continue  # judge each base item once, not once per turn record
        if base_id not in item_by_id:
            print(f"  {item_id}: skipped (not in suite definition)")
            continue
        item = item_by_id[base_id]
        is_multiturn = item.get("type") == "multi-turn"
        # group all turn records of this base item by normalized model
        group: dict[str, list] = {}
        for iid, rs in by_item.items():
            if iid.split("#")[0] != base_id:
                continue
            for r in rs:
                group.setdefault(normalize_fingerprint(r["model_fingerprint"]),
                                 []).append(r)
        if len(group) != 2:
            print(f"  {base_id}: skipped (models: {len(group)}, need 2)")
            continue
        m_keys = sorted(group)
        model_a_fp, model_b_fp = m_keys
        if is_multiturn:
            o_a = [r["output"] for r in sorted(group[model_a_fp], key=lambda x: x["id"])]
            o_b = [r["output"] for r in sorted(group[model_b_fp], key=lambda x: x["id"])]
            # judge each turn pair separately: clearer verdicts than a whole-
            # conversation blob, and each judgment links its turn's result ids
            turn_records_a = sorted(group[model_a_fp], key=lambda x: x["id"])
            turn_records_b = sorted(group[model_b_fp], key=lambda x: x["id"])
            for t_idx, (ra_t, rb_t) in enumerate(zip(turn_records_a, turn_records_b)):
                context = serialize_multiturn(item, [ra_t["output"]], [rb_t["output"]])
                order = random.random() < 0.5
                first_was_a = order
                resp1, resp2 = (ra_t["output"], rb_t["output"]) if order else \
                               (rb_t["output"], ra_t["output"])
                verdict = judge_pair(args.judge_url, args.judge_model or "test",
                                     context, resp1, resp2, args.template)
                if not verdict["valid"]:
                    print(f"  {base_id} t{t_idx+1}: INVALID: {verdict['raw']!r}")
                    continue
                winner_raw = verdict["winner"]
                if winner_raw == "1":
                    winner = "a" if first_was_a else "b"
                elif winner_raw == "2":
                    winner = "a" if not first_was_a else "b"
                else:
                    winner = "tie"
                # judgment links the turn-specific result ids; item_id of the
                # stored eval_result already carries the turn suffix
                post_json(f"{args.api}/api/judgments", {
                    "eval_result_a": ra_t["id"] if first_was_a else rb_t["id"],
                    "eval_result_b": rb_t["id"] if first_was_a else ra_t["id"],
                    "judge_model": args.judge_model or "(default)",
                    "judge_template_version": verdict["template_version"],
                    "winner": winner,
                    "confidence": None,
                    "rationale": None,
                })
                print(f"  {base_id} t{t_idx+1}: {verdict['winner']} (winner={winner})")
                stored += 1
            judged_base_ids.add(base_id)
        else:
            ra, rb = group[model_a_fp][0], group[model_b_fp][0]
            order = random.random() < 0.5
            first_was_a = order
            resp1, resp2 = (ra["output"], rb["output"]) if order else \
                           (rb["output"], ra["output"])
            verdict = judge_pair(args.judge_url, args.judge_model or "test",
                                 item.get("prompt", ""), resp1, resp2,
                                 args.template)
            if not verdict["valid"]:
                print(f"  {base_id}: INVALID: {verdict['raw']!r}")
                continue
            winner_raw = verdict["winner"]
            if winner_raw == "1":
                winner = "a" if first_was_a else "b"
            elif winner_raw == "2":
                winner = "a" if not first_was_a else "b"
            else:
                winner = "tie"
            post_json(f"{args.api}/api/judgments", {
                "eval_result_a": ra["id"] if first_was_a else rb["id"],
                "eval_result_b": rb["id"] if first_was_a else ra["id"],
                "judge_model": args.judge_model or "(default)",
                "judge_template_version": verdict["template_version"],
                "winner": winner,
                "confidence": None,
                "rationale": None,
            })
            print(f"  {base_id}: {verdict['winner']} (winner={winner})")
            stored += 1
            judged_base_ids.add(base_id)
    print(f"stored {stored} judgments")


if __name__ == "__main__":
    main()
