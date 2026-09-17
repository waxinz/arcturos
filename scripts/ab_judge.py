"""J4 — blind A/B judging against a live dashboard + judge endpoint.

Reads stored eval results for a suite, groups them per item (model A/B),
randomizes presentation order, sends an anonymized pairwise prompt to a
judge endpoint, parses the verdict, and stores a judgment via the API.

Usage:
    venv/bin/python scripts/ab_judge.py [--api http://localhost:24816]
        --suite-id 1 --judge-url http://10.10.10.222:8000
        [--judge-model GLM-5.3-Flash] [--template blind-pair-v1]

CAVEAT printed at runtime: judge must not be a contestant for trustworthy
results. Enforced with --allow-contestant-judge to override.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from urllib import request as urlreq
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent

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
    req = urlreq.Request(url, data=json.dumps(payload).encode(),
                         headers={"Content-Type": "application/json"})
    with urlreq.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_json(url: str) -> dict:
    with urlreq.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def judge_pair(judge_url: str, judge_model: str, prompt: str,
               resp1: str, resp2: str, template_version: str) -> dict:
    """Send blind pair to judge; returns parsed verdict dict."""
    payload_text = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, resp1=resp1, resp2=resp2)
    resp = post_json(judge_url.rstrip("/") + "/v1/chat/completions", {
        "model": judge_model,
        "messages": [{"role": "user", "content": payload_text}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
    })
    raw = resp["choices"][0]["message"]["content"].strip().lower()
    if raw.startswith("1"):
        winner = "1"
    elif raw.startswith("2"):
        winner = "2"
    elif "tie" in raw:
        winner = "tie"
    else:
        return {"valid": False, "raw": raw}
    return {"valid": True, "raw": raw, "winner": winner,
            "template_version": template_version}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:24816")
    ap.add_argument("--suite-id", type=int, required=True)
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--template", default="blind-pair-v1")
    ap.add_argument("--allow-contestant-judge", action="store_true")
    args = ap.parse_args()

    # load suite items for category + prompts
    suite_file = ROOT / "qa" / "suites" / "smoke-reasoning-v1.json"
    suite = json.loads(suite_file.read_text())
    item_by_id = {i["id"]: i for i in suite["items"]}

    results = get_json(f"{args.api}/api/eval-suites/{args.suite_id}/results")
    by_item: dict[str, list] = {}
    for r in results:
        by_item.setdefault(r["item_id"], []).append(r)

    contested = set()
    for item_id, pair in by_item.items():
        if len(pair) != 2:
            continue
        contested.add(pair[0]["model_fingerprint"])
        contested.add(pair[1]["model_fingerprint"])

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
    for item_id, pair in sorted(by_item.items()):
        if len(pair) != 2:
            print(f"  {item_id}: skipped (have {len(pair)} results, need 2)")
            continue
        ra, rb = pair[0], pair[1]
        prompt_text = item_by_id.get(item_id, {}).get("prompt", "")
        # randomize presentation order per item (true blind A/B)
        order = random.random() < 0.5
        r1, r2 = (ra, rb) if order else (rb, ra)
        verdict = judge_pair(args.judge_url, args.judge_model or "test",
                             prompt_text, r1["output"], r2["output"],
                             args.template)
        if not verdict["valid"]:
            print(f"  {item_id}: INVALID judge output: {verdict['raw']!r}")
            continue
        # map back: winner 1 -> whichever result was presented first
        winner_side = {"1": r1["id"], "2": r2["id"], "tie": None}[verdict["winner"]]
        if winner_side == ra["id"]:
            winner = "a"
        elif winner_side == rb["id"]:
            winner = "b"
        else:
            winner = "tie"
        post_json(f"{args.api}/api/judgments", {
            "eval_result_a": ra["id"], "eval_result_b": rb["id"],
            "judge_model": args.judge_model or "(default)",
            "judge_template_version": verdict["template_version"],
            "winner": winner,
            "confidence": None,
            "rationale": None,
        })
        print(f"  {item_id}: {verdict['winner']} (stored winner={winner})")
        stored += 1
    print(f"stored {stored} judgments")


if __name__ == "__main__":
    main()
