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
- [x] J2 comparison UI: compare/diff endpoints + chart views at /compare /diff (ef887c5)
- [x] J4 blind A/B judge runner: scripts/ab_judge.py (e958d32+fixes 69d6c21, 785bc1f, 2708523, b73ea6f)
- [x] J5 report aggregation: src/arcturos/reports.py (5ee2768)
- [x] J3 multi-turn replay: src/arcturos/multiturn.py + live validation (807f24d)
- [x] J4 multi-turn judging: per-turn blind judgments (turn-suffixed item ids)
- [x] Eval/judgment/reports views: /evals /judgments /reports + /api/reports/suite/{id} (42218eb + tests 5c08e31)
- [x] TTFT streaming variant: bench_stream.py (llama.cpp) + bench_openai.py (litellm/vLLM) with reasoning-model TTFT semantics
- [x] Power draw: power.py validated (134W on ruapehu during bench) — dedicated hosts only per ADR 002
- [ ] M9 hardening remainder: seed power into store path, per-host power metadata, multi-host compare
- [ ] M9 hardening: TTFT streaming variant, power draw wiring, multi-turn report rows
- [ ] First real benchmark captured against a live server (validation)

### Log (newest first)

- 2026-09-17 — UX design doc written: `docs/ux-design.md` — journey
  walkthroughs J1–J6, CLI surface spec, dashboard view specs (M2).
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
