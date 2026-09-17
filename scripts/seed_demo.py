"""Seed demo data into a running Arcturos dashboard (or a DB via the API).

Usage:
    venv/bin/python scripts/seed_demo.py [--api http://localhost:24816]

Behavior:
  1. POSTs demo benchmark runs from qa/suites/demo-seed.json.
  2. Creates eval suites from qa/suites/*.json (skips demo-seed.json).
  3. If an eval target is reachable, replays suite items; otherwise seeds a
     small synthetic set of eval_results + judgments so J5 report views have
     data in demo mode.

Idempotent-ish: creates new rows each call (append-only by design). Safe to
re-run; every invocation is a fresh, traceable dataset.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib import request as urlreq
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
SUITES_DIR = ROOT / "qa" / "suites"


def post(api: str, path: str, payload: dict) -> dict:
    req = urlreq.Request(
        api.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlreq.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def get(api: str, path: str) -> dict | list:
    with urlreq.urlopen(api.rstrip("/") + path, timeout=10) as resp:
        return json.loads(resp.read())


def seed_benchmarks(api: str, demo: dict) -> int:
    count = 0
    for run in demo["runs"]:
        created = post(api, "/api/runs", {
            "server_url": run["server_url"],
            "model_fingerprint": run["model_fingerprint"],
            "engine": run["engine"],
            "context_size": run["context_size"],
        })
        run_id = created["id"]
        for bench in run["benchmarks"]:
            post(api, f"/api/runs/{run_id}/benchmarks", bench)
            count += 1
    return count


def seed_eval_data(api: str, suites: list[dict], live_target: str | None) -> dict:
    seeded = {}
    for suite in suites:
        created = post(api, "/api/eval-suites", {
            "name": suite["suite"], "version": suite["version"],
        })
        suite_id = created["id"]
        seeded[suite["suite"]] = suite_id

        if live_target:
            results = replay_live(api, suite_id, suite["items"], live_target)
            if results:
                continue  # live replay already produced real results

        results = seed_synthetic(api, suite_id, suite["items"])
    return seeded


def replay_live(api: str, suite_id: int, items: list[dict], target: str) -> int:
    """Replay suite items against a live OpenAI-compatible target (J3)."""
    n = 0
    for item in items:
        if item["type"] != "single-turn":
            continue  # multi-turn replay lands with the J3 runner implementation
        t0 = time.monotonic()
        try:
            resp = post_openai(target, [{"role": "system", "content": item.get("system") or ""},
                                        {"role": "user", "content": item["prompt"]}])
        except (URLError, OSError, HTTPError) as e:
            print(f"  target unreachable for {item['id']}: {e}", file=sys.stderr)
            return 0
        latency_ms = (time.monotonic() - t0) * 1000
        post(api, f"/api/eval-suites/{suite_id}/results", {
            "model_fingerprint": resp.get("model", "unknown"),
            "item_id": item["id"],
            "output": resp["choices"][0]["message"]["content"],
            "prompt_tokens": resp.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": resp.get("usage", {}).get("completion_tokens", 0),
            "latency_ms": round(latency_ms, 1),
        })
        n += 1
    return n


def post_openai(target: str, messages: list[dict]) -> dict:
    req = urlreq.Request(
        target.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({"model": "test", "messages": messages,
                         "max_tokens": 128, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlreq.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def seed_synthetic(api: str, suite_id: int, items: list[dict]) -> int:
    """Demo-mode fallback: two models, canned outputs, judgments across pairs."""
    models = ["model-alpha (demo)", "model-beta (demo)"]
    answers = {
        "sr-001": ("The ball costs $0.05.", "The ball costs $0.05; the bat $1.05."),
        "sr-002": ("R appears 3 times.", "There are 3 Rs in strawberry."),
        "sr-003": ("Hamlet procrastinates...", "Prince Hamlet ..."),
        "sr-004": ("def is_palindrome(s): return ...", "def is_palindrome(s): ..."),
    }
    result_ids: dict[str, list[int]] = {}
    n = 0
    for model_idx, model in enumerate(models):
        for item in items:
            if item["type"] != "single-turn":
                continue
            out = answers.get(item["id"], (f"{model} answer to {item['id']}",))[0]
            created = post(api, f"/api/eval-suites/{suite_id}/results", {
                "model_fingerprint": model,
                "item_id": item["id"],
                "output": out,
                "prompt_tokens": 20 + 7 * model_idx,
                "completion_tokens": 30 + 5 * model_idx,
                "latency_ms": 840.0 + 130 * model_idx,
            })
            result_ids.setdefault(item["id"], []).append(created["id"])
            n += 1
    # judgments: alpha preferred on logic, beta on writing, tie on counting
    winners = {"sr-001": "a", "sr-002": "tie", "sr-003": "b", "sr-004": "b"}
    for item_id, (ra, rb) in result_ids.items():
        w = winners.get(item_id, "tie")
        post(api, "/api/judgments", {
            "eval_result_a": ra, "eval_result_b": rb,
            "judge_model": "external-judge (demo)",
            "judge_template_version": "blind-pair-v1",
            "winner": w,
            "confidence": 0.72,
            "rationale": "Seeded demo judgment.",
        })
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:24816")
    args = ap.parse_args()

    demo = json.loads((SUITES_DIR / "demo-seed.json").read_text())
    suites = [json.loads(p.read_text()) for p in sorted(SUITES_DIR.glob("*.json"))
              if p.name != "demo-seed.json"]

    get(args.api, "/health")  # raises if dashboard is down
    bench_rows = seed_benchmarks(args.api, demo)
    print(f"seeded {bench_rows} benchmark rows")
    live_target = None
    for candidate in ("http://10.10.10.122:8000", "http://10.10.10.222:8000"):
        try:
            urlreq.urlopen(candidate + "/health", timeout=4)
            live_target = candidate
            break
        except (URLError, OSError):
            continue
    seeded = seed_eval_data(args.api, suites, live_target)
    print(f"seeded suites: {seeded} (live target: {live_target or 'none — synthetic'}")


if __name__ == "__main__":
    main()
