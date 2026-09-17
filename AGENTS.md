# AGENTS.md — Arcturos

**Project:** Arcturos — local-first LLM benchmarking & evaluation suite
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
- Language: Python 3.10+, SQLite via stdlib or `sqlite3`/SQLAlchemy. CLI-first;
  web UI is a later phase.

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

**Phase:** 1 of 6 — Foundation & PRD
**Next phase:** 2 — PRD review/refinement with owner, then Phase 3 UX design.

- [x] Repo initialized on taupo (`~/arcturos`, branch `main`)
- [x] AGENTS.md created (this file)
- [x] PRD.md drafted from key user journeys
- [ ] PRD reviewed and signed off by Alexei
- [ ] UX design doc (journey walkthroughs, wireframe-level flows)
- [ ] QA test plan + suites
- [ ] Implementation (bench runner → storage → comparison UI → evals → judge → reports)
- [ ] First real benchmark captured against a live server (validation)

### Log (newest first)

- 2026-09-17 — Repo created on taupo; AGENTS.md + PRD.md committed (Phase 1).

## 7. How to pick up work (any agent)

1. `git -C ~/arcturos pull` (or just read if local) and read this file.
2. Check §6 Current Status: what phase, what's next.
3. Read `PRD.md` for the journey you're implementing; check `docs/decisions/`
   for constraints already decided.
4. Work, test, commit, then **update §6 (status + log)** in the same commit.
5. Never commit `data/` contents or credentials.
