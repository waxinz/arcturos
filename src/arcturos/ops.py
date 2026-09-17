"""Dispatch layer — kick off benchmark / eval runs against a live server.

Called from API endpoints (thin wrappers here), so the dashboard UI can
CREATE entities and START real work, not only view stored rows. Every
dispatch is synchronous one-shot work: a bench point, a full bench sweep,
or one eval suite replay. Long jobs are still bounded by timeouts; the UI
gets progress via on_point callbacks -> stored rows appear as they land.

Append-only is preserved: dispatch only ever INSERTs (runs, benchmarks,
eval_results). There is no UPDATE/DELETE path anywhere.
"""

from __future__ import annotations

import time

from typing import Any

from arcturos import bench
from arcturos import multiturn as mt
from arcturos import preflight


class DispatchError(ValueError):
    status_code = 400


def dispatch_bench(
    db_path,
    server_url: str,
    targets: list[int],
    n_predict: int,
    timeout: float = 3600.0,
    transport=None,
    on_point=None,
) -> dict:
    """Run a cold-cache bench sweep; store run + points; return run payload."""
    if not targets:
        raise DispatchError("targets list is empty")
    if any(t < 1 for t in targets):
        raise DispatchError("every target must be >= 1 token")
    if n_predict < 1:
        raise DispatchError("n_predict must be >= 1")
    pf = preflight.preflight(server_url, kind="bench", transport=transport)
    if not pf.ok():
        raise DispatchError(
            "preflight failed: " + "; ".join(
                c["detail"] for c in pf["checks"] if not c["ok"]))
    try:
        points = bench.run_benchmark(
            server_url, "default", targets, n_predict,
            timeout=timeout, transport=transport, on_point=on_point,
        )
    except (RuntimeError, ValueError) as exc:
        raise DispatchError(str(exc)) from exc
    engine_meta = bench.capture_engine_metadata(server_url, transport=transport)
    run_id = bench.store_benchmark_run(db_path, server_url, engine_meta, points)
    return {
        "run_id": run_id,
        "engine_metadata": engine_meta,
        "points": [
            {
                "context_tokens": p.target_tokens,
                "prefill_tps": p.prefill_tps,
                "decode_tps": p.decode_tps,
                "wall_s": p.wall_s,
                "output_tokens": p.output_tokens,
                "mtp_draft_n": p.mtp_draft_n,
                "mtp_accepted": p.mtp_accepted,
                "prompt_tokens": p.prompt_tokens,
                "stop_reason": p.stop_reason,
            }
            for p in points
        ],
    }


def dispatch_eval(
    db_path,
    suite_id: int,
    target: str,
    model_fingerprint: str,
    suite: dict[str, Any],
    api_key: str | None = None,
) -> dict:
    """Replay one eval suite against a target; store eval_results per item.

    Single-turn items replay as one POST; multi-turn items replay via
    multiturn.replay_multiturn (one stored row per assistant turn).
    ``model_fingerprint`` is passed to the target as the chat model name —
    proxy-fronted endpoints (litellm/vLLM) need the real name; native
    llama.cpp servers ignore it.
    """
    items = suite.get("items")
    if not isinstance(items, list) or not items:
        raise DispatchError("suite has no items to replay")
    # cheap shape validation BEFORE preflight: bad input needs no network
    for item in items:
        if item.get("type") == "multi-turn":
            mt.validate_multiturn_item(item)
        else:
            item_id = item.get("id") or item.get("item_id")
            if not item_id:
                raise DispatchError(f"item missing id: {item!r}")
            if not (item.get("prompt") or item.get("content")):
                raise DispatchError(f"item {item_id!r} missing prompt")

    from arcturos import db as dbmod
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    if conn.execute("SELECT 1 FROM eval_suites WHERE id = ?", (suite_id,)).fetchone() is None:
        conn.close()
        raise DispatchError(f"eval suite {suite_id} not found (create it first)")
    conn.close()
    pf = preflight.preflight(target, kind="eval", model=model_fingerprint)
    if not pf.ok():
        raise DispatchError(
            "preflight failed: " + "; ".join(
                c["detail"] for c in pf["checks"] if not c["ok"]))
    conn = dbmod.connect(db_path)
    dbmod.init_db(conn)
    stored: list[dict] = []
    try:
        for item in items:
            if item.get("type") == "multi-turn":
                records = mt.replay_multiturn(target, item, api_key=api_key,
                                              model=model_fingerprint)
                for rec in records:
                    row = _store_eval_result(
                        conn, suite_id, model_fingerprint, rec["item_id"],
                        rec["output"], rec["prompt_tokens"],
                        rec["completion_tokens"], rec["latency_ms"])
                    stored.append(row)
            else:
                item_id = item.get("id") or item.get("item_id")
                if not item_id:
                    raise DispatchError(f"item missing id: {item!r}")
                prompt = item.get("prompt") or item.get("content")
                if not prompt:
                    raise DispatchError(f"item {item_id!r} missing prompt")
                t0 = time.monotonic()
                resp = mt.post_openai_chat(
                    target, [{"role": "user", "content": prompt}],
                    api_key=api_key, model=model_fingerprint)
                latency_ms = round((time.monotonic() - t0) * 1000, 1)
                content, ptoks, ctoks = mt.extract_output(resp)
                stored.append(_store_eval_result(
                    conn, suite_id, model_fingerprint, item_id,
                    content, ptoks, ctoks, latency_ms))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"suite_id": suite_id, "stored": stored}


def _store_eval_result(conn, suite_id: int, model_fingerprint: str,
                       item_id: str, output: str, prompt_tokens: int,
                       completion_tokens: int, latency_ms: float) -> dict:
    cur = conn.execute(
        "INSERT INTO eval_results (suite_id, model_fingerprint, item_id, output,"
        " prompt_tokens, completion_tokens, latency_ms, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (suite_id, model_fingerprint, item_id, output,
         prompt_tokens, completion_tokens, latency_ms,
         bench._utcnow()))
    row = dict(conn.execute(
        "SELECT * FROM eval_results WHERE id = ?", (cur.lastrowid,)).fetchone())
    return row
