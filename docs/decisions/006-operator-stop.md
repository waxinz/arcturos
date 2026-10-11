# ADR 006 — Operator stop for running bench jobs: cooperative abort flags

Date: 2026-10-11
Status: accepted

## Context

A faulty bench job (dead target, runaway context sweep, wrong form
values) can only be stopped today by restarting the whole service —
which loses the in-process job record, orphans the half-finished sweep
thread (it keeps measuring into a dead store handle), and gives the
operator no record of what completed. Runs like #126 (Strata, 32k-256k
sweep on a misconfigured target) waste an hour measuring garbage that
then has to be hidden manually.

## Decision

Cooperative abort, not thread killing:

1. **AbortFlag** (`ops.py`): a `threading.Event` wrapper.
   `request()` sets it (from the API thread), `check()` raises
   `SweepAborted` (a `RuntimeError`) when set (from sweep threads).
2. **Check points.** The sweep checks the flag:
   - between points, in both transports' point loops
     (`bench.run_benchmark`, `ops._run_openai_bench_points`);
   - inside the openai streaming loop (`bench_openai.run_openai_stream_point`),
     so a 256k-token decode that would run for minutes stops at the
     next SSE chunk instead of running to completion.
   A mid-stream abort abandons that point's partial counts — only whole
   points ever reach the store.
3. **Failure path, not a special path.** `SweepAborted` is a
   `RuntimeError`, so it flows into the existing partial-run nets in
   `dispatch_bench`: completed points are stored as a partial run, and
   the return payload carries `status: "aborted"` (the job record
   surfaces the same). Zero completed points → `DispatchFailure`, the
   no-empty-run-rows contract holds.
4. **Registry.** `run_bench_job` stores each job's flag in
   `_ABORT_FLAGS` (outside the JSON job record — an Event must not leak
   into the status snapshot). `abort_bench_job(job_id)` pops the flag,
   requests it, and stamps `abort_requested: true` on the record;
   `POST /api/ops/bench/jobs/{id}/abort` exposes it (404 unknown, 200
   no-op on a finished job). Flags are dropped with their job records
   by the TTL pruner.
5. **UI.** The Kick off page shows a red Stop button only while a job
   is in flight (preflight/running); confirm dialog names the
   partial-run semantics; the poller renders the `aborted` terminal
   state with a link to the stored partial run.

## Rejected alternatives

- *Kill the worker thread.* Rejected: `ThreadPoolExecutor`/daemon
  threads cannot be cancelled safely mid-HTTP-call; killing leaks the
  open SSE socket and any half-written store transaction.
- *Kill the service process.* Rejected: loses the job record for every
  job, not just the faulty one, and the thread may still outlive the
  restart briefly.
- *Synchronous abort via a second request to the target server.* Rejected:
  tabbyAPI/vLLM expose no portable cancellation handle, and the operator
  should not need target credentials to stop a measurement.

## Consequences

- The current point is abandoned mid-decode (its token counts never
  stored); the previous points are kept. "Stop" therefore means
  "keep what finished, stop measuring" — never a torn row.
- The abort flag is in-process state: a service restart still loses the
  ability to stop a job (the job is gone with it — same trade-off the
  2026-09-21 async-jobs ADR already accepted).
- Historical note: the button rename on the same card ("✔ Check target
  - just one ping, Vasily", owner call) is a label-only change; the
  preflight behaviour is unchanged.
