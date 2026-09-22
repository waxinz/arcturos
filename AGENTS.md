# AGENTS.md — Arcturos

**Project:** Arcturos — local-first LLM benchmarking & evaluation suite
**Interface:** local web dashboard, `0.0.0.0:24816` (owner decision 2026-09-17)
**Repo host:** taupo (`alexei@192.168.122.1`), path `~/arcturos`
**Owner:** Alexei
**Created:** 2026-09-17
**Status:** Phase 1 (foundation) — see [Current Status](#current-status)

> **Read this file first.** It is the single source of truth for project
> progress, conventions, and how any agent (human or AI) picks up work at any
> point. Update it **in the same commit** as any meaningful change.

---

## 1. Mission

Arcturos benchmarks and evaluates LLM inference servers and stores every result
for future comparison. It answers two questions, repeatably:

1. **How fast is a model?** — decode tok/s, prefill tok/s, TTFT, wall time,
   MTP/speculative acceptance, power draw — at multiple context lengths.
2. **How good is a model?** — prompt-based evals (single- and multi-turn)
   scored by external LLM judges, including blind pairwise A/B preference.

All data is local-first (SQLite), reproducible, and exportable.

## 2. Key user journeys (from the PRD)

| # | Journey | PRD section |
|---|---------|-------------|
| 1 | Benchmark a model at multiple context lengths, store metrics | PRD §J1 |
| 2 | Visual comparison across models and runs | PRD §J2 |
| 3 | Administer/run prompt evals (single & multi-turn) | PRD §J3 |
| 4 | Blind A/B scoring of two models' outputs by an external judge | PRD §J4 |
| 5 | Reports on eval suite performance (win rates, coverage) | PRD §J5 |
| 6 | Supporting features (model registry, env snapshots, export) | PRD §J6 |

## 3. Repository layout

```
arcturos/
├── AGENTS.md            ← this file (progress + conventions)
├── PRD.md               ← product requirements (journeys, acceptance criteria)
├── docs/
│   ├── ux-design.md     ← UX design (Phase 3)
│   └── decisions/       ← architecture decision records (ADR-style, one file per decision)
├── qa/
│   ├── test-plan.md     ← QA strategy (Phase 4)
│   └── suites/          ← concrete test suites / fixtures
├── src/arcturos/        ← implementation (Phase 5+; language: Python 3.10+)
├── scripts/             ← operational helpers (bench runners, seed data)
└── data/                ← NOT committed. Local SQLite + result artifacts only.
```

## 4. Conventions

- **Commits:** imperative mood, reference journey/phase, e.g.
  `bench: capture MTP acceptance per draft step (J1)`.
- **Every phase ends with:** tests passing, AGENTS.md status updated, PRD
  checkboxes ticked for completed acceptance criteria.
- **Data model changes** require an entry in `docs/decisions/` and a schema
  migration note in AGENTS.md §Current Status.
- **Metrics are append-only.** Never mutate stored run data; supersede runs.
- **Bench methodology:** cold-cache measurement (fresh UUID prompt prefix,
  server restart between context-size tests where applicable). See
  `docs/decisions/` for the full methodology record.
- Language: Python 3.10+ (taupo has 3.12.3). SQLite stdlib. Dashboard: local
  web app bound `0.0.0.0:24816` (LAN-accessible; no auth beyond LAN boundary
  for v1). Charts rendered client-side (Chart.js or similar); API = JSON/REST.

## 5. Environment facts (taupo)

- Ubuntu 24.04, git 2.43.0, Python 3.12.3, `/` has 400+ GB free.
- GPU inference servers available for benchmarking targets (see
  home-server inventory skill): `ruapehu` 10.10.10.122 and
  `pakuranga-inf` 10.10.10.222 — 6× RTX 3090 each, llama.cpp `llama serve`
  on port 8000 (DS4-Flash, 262k ctx). `aotea` 10.10.10.14:8080 is another
  OpenAI-compatible target. `langfuse` on taupo (:3001) is an alternative
  eval-tracing backend but Arcturos is local-first by design.
- SSH as `alexei` is passwordless to all boxes.

## 6. Current Status

**Phase:** 2 of 6 — PRD signed off; UX design + implementation underway.
**Next:** dashboard skeleton live on taupo :24816, then J1 bench core.

- [x] Repo initialized on taupo (`~/arcturos`, branch `main`)
- [x] AGENTS.md created (this file)
- [x] PRD.md drafted from key user journeys
- [x] PRD reviewed and signed off by Alexei
- [x] UX design doc (journey walkthroughs, wireframe-level flows)
- [ ] QA test plan + suites
- [x] J1 bench core: bench.py (tokenize sizing, cold UUID prompts, timings parser, storage, export) — commit f7c2fee
- [x] J1 validated end-to-end: real bench of ruapehu DeepSeek-V4-Flash (512+4096 tok) stored via API, visible at :24816/api/runs/1/benchmarks
- [x] M3 QA: test plan + fixture eval suites + demo seed script + J3/J4 contract tests — commit 4e0cb3b
- [x] API key auth end-to-end (J6): optional `api_key` on every outbound request — bench (tokenize/completion/props), bench_stream, eval dispatch, preflight probes, `/api/ops/bench` + `/api/ops/eval` body field, `/create` UI password fields (bench + eval); keys never persisted to the store. Authed-mock tests added (98 green).
- [x] J2 comparison UI: compare/diff endpoints + chart views at /compare /diff (ef887c5)
- [x] J4 blind A/B judge runner: scripts/ab_judge.py (e958d32+fixes 69d6c21, 785bc1f, 2708523, b73ea6f)
- [x] J5 report aggregation: src/arcturos/reports.py (5ee2768)
- [x] J3 multi-turn replay: src/arcturos/multiturn.py + live validation (807f24d)
- [x] J4 multi-turn judging: per-turn blind judgments (turn-suffixed item ids)
- [x] Eval/judgment/reports views: /evals /judgments /reports + /api/reports/suite/{id} (42218eb + tests 5c08e31)
- [x] TTFT streaming variant: bench_stream.py (llama.cpp) + bench_openai.py (litellm/vLLM) with reasoning-model TTFT semantics
- [x] Power draw: power.py validated (134W on ruapehu during bench) — dedicated hosts only per ADR 002
- [x] Power provenance wired into store path: benchmarks.power_host + power_gpu_index columns, db migration for pre-existing DBs, API enforces watts-require-host (422)
- [x] J7 CRUD + dispatch: ops.py (dispatch_bench/dispatch_eval), /api/ops/bench + /api/ops/eval, /create view (6 form cards: bench dispatch, suite create, suite item editor, eval replay, manual bench point, judgment record); PRD J7 + ux-design §J7 + test-plan J7 sections written; 86 tests green
- [x] M9 hardening remainder: multi-host compare metadata (server_url in compare payload), health preflight before dispatch (J6), seed power into seed_demo path
- [x] First real benchmark captured via /create dispatch (live smoke, quiet window)

### Log (newest first)

- 2026-09-21 — OpenAI-compatible bench transport (26474d8): /api/ops/bench
  gains transport=native|openai + model; openai runs eval-kind preflight
  (no /tokenize on tabbyAPI/vLLM), cold UUID prompts sized client-side,
  streamed points with client-side TTFT/decode, engine='openai' stored,
  power fields stay None per ADR 002. First real benchmark captured via
  dispatch: run 9 vs pakuranga tabbyAPI GLM-5.3-Flash (512 ctx, 64 tok,
  prefill 386.3 tok/s, TTFT 1258.0 ms, wall 2.847 s). 140 tests green.

- 2026-09-21 — Suite definition snapshots + per-category report breakdown
  (1d0b37e + 34fa9c4): the DB stores item_id only, so /reports showed
  everything as "uncategorized" despite the ux-design category-breakdown
  spec. New append-only suite_definitions table (snapshot per PUT, newest
  wins) + PUT/GET /api/eval-suites/{id}/definition; reports.category_map_for()
  rebuilds the item->category map at report time and suite_report joins it.
  PUT allowlist contract test updated (one allowed PUT surface: definition
  snapshots). Live on taupo: /api/reports/suite/1 now reports 6 categories
  (logic 0/4/0, writing 0/4/0, counting 0/1/3, coding 1/2/1, reasoning 1/0/0,
  instruction-following 1/0/0 — a/b/tie/total). 136 tests green.

- 2026-09-21 — M9 hardening remainder closed (288b1ed + 1b847ee):
  seed demo benches carry power_host/gpu_index per ADR 002
  (demo-seed.json); tabbyAPI returns usage:null — resp.get('usage',
  {}) default never fires on explicit None, so extract_output
  crashed mid-dispatch with AttributeError; fixed with 'or {}' in
  multiturn.py extract_output + scripts/seed_demo.py; regression
  test test_extract_output_null_usage added; live re-verified: eval
  dispatch vs pakuranga tabbyAPI GLM-5.3-Flash stored eval_result
  id 38, latency 794.1ms, 133 tests green.

- 2026-09-17 — §4.6 backlog finished (bcd1cb8): compare baseline overlay
  (dashed 📌 reference dataset + per-run delta tables with direction-aware
  good/bad colouring; deltas only where BOTH sides have the metric at the
  same ctx length — honest gaps, never fabricated zeros; pinned and ad-hoc
  baselines both selectable); CSV export everywhere — compare payload
  (schema_version arcturos-compare-v1, fingerprint + host columns on every
  row, optional baseline reference row), /runs, per-run benchmarks, eval
  results, judgments (arcturos-v1 rows; missing metric = empty cell);
  export buttons on /runs /evals /judgments /compare and run detail;
  eval drill-down keyboard parity (Tab focus, Enter/Space toggle,
  aria-expanded). 129 tests green; live-verified on taupo (real delta
  rows + both CSVs download with attachment headers). §4.6 is now fully
  implemented; UX review findings all closed.

- 2026-09-17 — Model registry + baseline tracking (§4.6 J6, commits
  192dea7 + 7774f0d): /models view (cards per fingerprint: alias-only
  edit trigger-enforced, engine, first/last seen, run count) and
  /baselines view (pin/re-pin/unpin per model × metric-family, wrong-
  model pin rejected 422). Auto-registration on first sighting in bench
  dispatch, eval replay, and manual run create; insert-only registry
  backfill in init_db so pre-existing taupo runs appear without touching
  stored rows. Run-detail gained "Pin as baseline" (confirm dialog);
  compare legends show registry aliases as 'alias (short-fp)'. API
  contract test amended: PATCH /api/models/{fp} + DELETE
  /api/baselines/{id} are the two allowed non-data surfaces (PUT still
  forbidden). 116 tests green; live-verified on taupo (pin, re-pin,
  alias edit, unpin, backfill of 4 historical fingerprints).

- 2026-09-17 — UX review round (both reviewer agents finally completed
  after gateway 429/timeout retries; 11 findings deduped) + fixes
  (6233984): DispatchFailure/DispatchError split (runtime target failures
  now 502 with 'check the server' guidance, input/preflight stay 422);
  preflight moved to POST — api_key travels in the body, never a URL (key
  leak into access logs/history flagged by both reviewers); actionable
  preflight failure detail (names checks + target + next step); UI renders
  Pydantic 422 detail arrays as 'field: msg' instead of '[object Object]';
  bench form warns on dropped non-numeric targets; honest nulls in
  run-detail ('—' + AA-contrast + tooltip, no falsy-zero '?'); run-detail
  header actions (compare/diff/copy-JSON) + engine-env <details> block;
  compare view: busy-disable, 'select ≥2 runs' hint, decode_tps default
  only, accessible select-column header; eval drill-down keyboard note
  deferred. BONUS: fixed pre-existing test bug — _pass_preflight()
  permanently replaced the preflight module attribute, leaking ok:stub
  checks into later tests; now a restoring fixture. 103 tests green,
  live-verified on taupo.

- 2026-09-17 — UX backlog landed (1b2dcaa): confirm dialogs on
  judgment-record + manual-bench-point append posts (ux principle 5);
  eval output click-to-expand drill-down in /evals. 99 tests green,
  live on taupo. Backlog now: multi-host compare metadata (M9).
- 2026-09-17 — UX review and fixes (862ef4e): run drill-down view
  /runs/{id} (meta + per-point metrics, honest nulls) + clickable run ids
  on /runs; interactive preflight 8s per-check timeout (dead target
  answers <=16s, was 40s). Eval dispatch verified to consume api_key
  end-to-end (single + multi-turn + UI). 99 tests green, live-verified
  on taupo. Remaining UX backlog: confirm dialogs for re-judge,
  eval-results drill-down view, multi-host compare metadata (M9).
- 2026-09-17 — API key auth end-to-end (14edfb2 + 5733999): optional
  `api_key` threaded through the whole outbound path — bench.py
  (_client headers), bench_stream.py, ops.dispatch_bench/dispatch_eval,
  preflight module + `/api/ops/preflight?api_key=`, `/api/ops/bench` +
  `/api/ops/eval` body fields, `/create` UI password fields (bench card +
  eval card + preflight "Check target" buttons pass them). Keys are
  never persisted (store/log/export). Authed-mock tests (llama.cpp
  `--api-key` behavior) added; 98 tests green. Docs updated: PRD J6/J7,
  ux-design §1/§3, test-plan J7.
- 2026-09-17 — UX design doc written: `docs/ux-design.md` — journey
  walkthroughs J1–J6, CLI surface spec, dashboard view specs (M2).
- 2026-09-17 — J7 CRUD + dispatch landed (8cfd214 + f979b12): ops.py dispatch layer, /api/ops/bench + /api/ops/eval endpoints, /create view (6 form cards), power provenance (benchmarks.power_host + power_gpu_index with db migration; API rejects watts without host per ADR 002), bench.py nounlets→nounits fix (second instance of the same typo bug). PRD J7 + ux-design §6/§8 + qa/test-plan J7 written to match. 86 tests green. Dispatch validation contract: bad shapes → 422, server failure → 502, unknown suite checked BEFORE calling the target, eval batches transactional (rollback on failure). Live-validated on taupo: suite create 201 → eval replay vs qwen rotation 200, result stored. Follow-up fixes: model_fingerprint passed as chat model name (litellm 400 root cause — hardcoded "test" was rejected, d77ff9d); single-turn wall latency real (was stub 0.0, 5820b6d).
- 2026-09-17 — M9 surfaces complete: all 6 dashboard views live, 71 tests green on taupo. ADR 002: power sampling only on dedicated inference hosts (litellm is a proxy; qwen serving hosts elsewhere); never bench shared servers with unacknowledged queueing (ruapehu TTFT artifacts 32s-254s were self-inflicted; true cold TTFT 1.5-9s). Reasoning-model bench semantics: TTFT = first chunk of ANY payload; decode = content chunks only. qwen rotation validated: TTFT ~1.9-2.2s, prefill est ~373-450 tok/s, decode ~82-101 tok/s content.
- 2026-09-17 — J3 multi-turn complete: sr-101/sr-102 replayed vs both models, per-turn records stored (item_id #t-suffix), per-turn blind judging working. Final smoke-reasoning v1 report (n=18 judgments, 2 judges): overall a/b/tie 3/11/4 — DeepSeek 16.7% / GLM 61.1% / tie 22.2%; per-judge: qwen 0/6/2, laguna 3/5/2. All 6 journeys J1-J6 now have working implementations with real data.
- 2026-09-17 — J2/J4/J5 landed and validated with real data: smoke-reasoning suite replayed live vs DeepSeek-V4-Flash (ruapehu) + GLM-5.3-Flash (pakuranga-inf); blind judging via qwen rotation + laguna2.1-s judges on litellm (localhost:4000/v1, key sk-1234); report: overall 1/8/3 (a/b/tie, n=12), GLM 66.7% wins, per-judge + per-category breakdowns working. Judge is NOT a contestant. Lessons: reasoning models (DeepSeek-V4-Flash, qwen3.8) return empty content — read reasoning_content, max_tokens>=512; model fingerprints vary by server suffix — normalize identity in ab_judge; litellm /v1 suffix must be stripped.
- 2026-09-17 — J1 bench core landed (f7c2fee) + validated live; QA suite landed (4e0cb3b). 33 tests green on taupo. Bench targets: ruapehu .122:8000 (DeepSeek-V4-Flash) + pakuranga-inf .222:8000 (GLM-5.3-Flash) confirmed live; aotea .14:8080 down.
- 2026-09-17 — Dashboard skeleton live: FastAPI+SQLite append-only schema, 14 tests passing on taupo, systemd user service at 0.0.0.0:24816 (commit 789eaf3). UX design doc committed (f03ef37).
- 2026-09-17 — PRD signed off; dashboard decided: web UI on 0.0.0.0:24816.
- 2026-09-17 — Repo created on taupo; AGENTS.md + PRD.md committed (Phase 1).

## 7. How to pick up work (any agent)

1. `git -C ~/arcturos pull` (or just read if local) and read this file.
2. Check §6 Current Status: what phase, what's next.
3. Read `PRD.md` for the journey you're implementing; check `docs/decisions/`
   for constraints already decided.
4. Work, test, commit, then **update §6 (status + log)** in the same commit.
5. Never commit `data/` contents or credentials.
