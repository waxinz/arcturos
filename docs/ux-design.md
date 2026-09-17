# Arcturos — UX Design (Phase 2 / M2)

Local-first LLM benchmarking & evaluation suite.
Source of truth: `PRD.md` (signed off 2026-09-17). This document walks each
key user journey (J1–J6) end-to-end, specifies every CLI surface, and gives
view-level specs for the web dashboard at `0.0.0.0:24816`.

Owner: Alexei · Status: draft for implementation.
Companion docs: `PRD.md` (requirements), `docs/decisions/` (methodology/ADR).

---

## 1. Design principles

1. **One command per journey.** J1, J3, and J4 must each be launchable from a
   single CLI invocation with sensible defaults; every flag is overridable but
   nothing is mandatory beyond a target + model.
2. **Append-only, visibly.** The UI never offers destructive edit of stored
   metrics. Re-runs supersede; the UI shows superseded history behind an
   explicit "history" affordance, not buried.
3. **Fingerprints over labels.** Any comparison UI shows the model fingerprint
   (quant, engine build, flags) alongside the human label, so two "same"
   models are never silently compared.
4. **Nulls are honest.** A metric the engine couldn't report shows as
   `null` with a reason string, never as `0` or a dash implying zero.
5. **Local-first, LAN-trust.** No auth screens; the dashboard is a LAN tool.
   All destructive-adjacent actions (re-run, re-judge) get confirm dialogs,
   not credentials.
6. **Keyboard-first CLI, glance-first UI.** The operator (P1) lives in the
   terminal; the dashboard is for reading and comparing, not data entry.

## 2. Personas and surfaces

| Persona | Hat | Primary surface | Secondary |
|---|---|---|---|
| P1 — benchmark operator | "how fast" | CLI (`arcturos bench …`) | Dashboard: compare views |
| P2 — eval operator | "how good" | CLI (`arcturos eval …`) | Dashboard: reports, item drill-down |

Both personas share one SQLite store and one dashboard.

## 3. Information architecture

```
CLI                          Dashboard (0.0.0.0:24816)
────────────                 ─────────────────────────
arcturos bench               ├─ Runs          (list, filters, run detail)
arcturos eval                ├─ Compare       (models × metrics; run diff)
arcturos judge               ├─ Evals         (suites, results, drill-down)
arcturos report              ├─ Judge         (A/B setup, blind runs, re-judge)
arcturos models              ├─ Reports       (win-rate matrix, exports)
arcturos export              ├─ Models        (registry, fingerprints)
arcturos health              └─ Baselines     (pin reference runs)
arcturos serve
```

Dashboard nav is a persistent left rail (7 items above), each opening a
section with its own filter bar. URL-routed sections (`/runs`, `/compare`,
`/evals`, `/judge`, `/reports`, `/models`, `/baselines`) so a view can be
bookmarked and re-shared over the LAN.

## 4. Journey walkthroughs

### 4.1 J1 — Benchmark at multiple context lengths (P1)

**Flow**

1. Operator picks target + model. Either via flags or interactively:
   `arcturos bench http://ruapehu:8000/vllm --model ds4-flash --ctx 4096 32768 131072 250000`
2. **Health preflight (J6)** runs automatically before launch: server reachability,
   `/tokenize` availability, model loaded, context size discovered, GPU visible
   for power sampling. Failures abort with an actionable message
   ("server reports 200k ctx max; drop 250k point or pass `--force-ctx`").
3. Harness tokenizes each context point via `/tokenize`, prepends a fresh UUID
   prefix, and runs. Live progress prints per context point:
   `[32k] prefill 8,412 tok/s · decode 3,120 tok/s · ttft 214ms · 12% MTP accept · 412W`
4. On completion the command prints the **run ID** (UUID) and a one-line
   metric table. Nothing overwrites: a repeated run at the same point
   creates a new record; the CLI notes "run X supersedes run Y for
   model M @ 32k".
5. Raw request/response JSON artifacts stored when `--save-artifacts` passed
   (default: off, to keep the data dir lean).

**States & errors**

| State | UI/CLI behavior |
|---|---|
| Server unreachable | Abort before any run; show endpoint tried, curl-style error |
| Metric missing | Stored `null` + reason (e.g. `mtp_accepted: null — engine did not report counters`); table shows `—` with hover/reason column |
| Power sampling unavailable | `power_w: null — nvidia-smi not reachable from host`; run continues |
| Cold-cache not provable | CLI prints warning naming the point(s) and the restart requirement (per `docs/decisions/`) |
| Interrupted mid-run | Partial context points marked `incomplete`; run ID still issued; resume = new run |

**Post-run hand-off:** the printed run ID is the token the operator pastes
into compare/baseline commands. CLI suggests: `arcturos report compare <run-id>`.

### 4.2 J2 — Visual comparison (P1)

**View: `/compare`**

- **Selector bar (top):** model multi-select (registry-backed, fingerprint
  shown under label), metric multi-select (decode_tps, prefill_tps, ttft,
  wall_time, acceptance, power_w, mtp_accepted), context-point range slider
  (log-scale awareness for 4k→250k).
- **Chart area (center):** Chart.js line chart, x = context length,
  y = selected metric, one series per model. Metric switch = tabs above chart
  (not page reload). Bar mode available for wall_time.
- **Run diff (below, collapsible):** pick two runs of the same model →
  side-by-side table: flags/cmdline/quant on the left column pair, metrics
  with deltas on the right. Deltas colored: green ≥ +2% improvement,
  red ≤ −2%, gray otherwise. Absolute and % delta shown.
- **Export:** `PNG`, `SVG`, `Shareable JSON` buttons top-right. JSON export
  carries series data + run IDs + fingerprints so a chart can be rebuilt.
- **Baseline mode:** when a run is pinned (J6), compare view shows a dashed
  reference series and delta column vs baseline.

**States:** empty (no runs) → onboarding card with the exact `bench` command
template. Filtered-empty → "No runs match: model X, metric Y, ctx ≥ 128k"
with a "clear filters" link. Superseded runs are toggleable ("show history").

### 4.3 J3 — Run prompt evals (P2)

**Flow**

1. Suites live in versioned JSON/YAML. `arcturos eval suites/qa-basic.yaml
   http://aotea:8080 --model m3 --concurrency 4 --retries 2`
2. Runner prints a live table: item ID, turn shape (1-turn/mt), status
   (pending → running → ok / error / retrying), latency so far. Failures are
   printed inline, not swallowed: `[item 17] error 500 after 2 retries — stored`.
3. Every output stored with: suite version hash, model fingerprint, token
   counts, latency, engine timings, concurrency/retry policy in effect.
4. Completion summary: N items, N ok / N failed, wall time, result-set ID.

**Suite authoring UX (admin side):** a suite file template + JSON Schema is
shipped (`docs` + `arcturos eval validate <file>`), with validation errors
pointing at line/item. Multi-turn items use ordered messages with
`assistant:` placeholders; the validator flags empty placeholder chains.

**States:** suite file missing/unparseable → abort with the offending path
and parse error. Server/model mismatch vs suite expectations → warn, allow
`--force`. Re-run always creates a **new result set**, tagged with suite
version; old sets remain queryable.

### 4.4 J4 — Blind A/B judging (P2)

**Flow**

1. `arcturos judge <suite> --models A,B --judge-model <api-model> --template templates/pairwise-v2.yaml`
2. The UI/CLI shows the setup panel: judge model, template version (pinned
   and stored with results), order randomization policy, retry policy for
   invalid judge output. Anonymization is **structural**: payloads sent to
   the judge contain only `Response 1` / `Response 2`; model identities are
   held by the tool and mapped only at storage time.
3. Live progress: per-item judgment status. Invalid judge output (unparseable
   preference) is retried per policy; after policy exhaustion it is stored as
   `invalid` with the raw response kept for inspection — never silently
   dropped.
4. Re-judging with a different template/judge model appends new judgment
   records; nothing is edited in place. The report view lets the operator
   pick which judgment generation to aggregate.

**Leakage guard (acceptance-critical):** stored judgment payload contains
eval item ID + both result IDs + template version + judge model, but the
sent-to-judge payload is what must be leakage-free. The design keeps a
"what the judge saw" artifact per item (optional, `--save-artifacts`) so
leakage can be audited. UI labels the mapping column "internal only".

### 4.5 J5 — Eval performance reports (P2)

**View: `/reports`**

- **Suite + result-set selector** (top). Defaults to newest set per suite.
- **Win-rate matrix (center):** rows = model pairs, columns = suite-level
  A / B / tie counts and percentages, e.g. `A 62% · B 34% · tie 4% (n=100)`.
  n is always shown; small-n cells (n < 20) get a "low confidence" tint.
- **Category breakdown (below):** horizontal bar per category tag × model,
  win-rate per category.
- **Item drill-down (click-through):** table of items → row expands to the
  raw outputs of both models side by side + the judgment(s) with template
  version and rationale. From drill-down the operator can jump to the
  single-item judgment history (all generations).
- **Exports:** CSV, JSON, HTML buttons; export mirrors the filtered view.

### 4.6 J6 — Supporting features UX

- **Model registry (`/models`):** cards per fingerprint: repo/file hash,
  quant, engine build, flags, first/last seen, run count. The label is
  editable (alias) but the fingerprint is immutable and shown prominently;
  compare/eval selectors only ever offer registry entries.
- **Environment snapshot:** shown in run detail as a collapsible "engine
  environment" block (cmdline, flags, driver, power limit) — copyable as
  JSON. Every comparison surface links back to it.
- **Data export:** `arcturos export <run-id|result-set> --format jsonl|csv`
  and per-view export buttons. Exports include schema version + fingerprint
  fields so external analysis can self-describe.
- **Baseline tracking (`/baselines`):** pin/unpin a run as reference per
  (model, metric-family). Compare views render the dashed baseline and
  delta columns; the pin action lives in run detail ("Pin as baseline").
- **Health preflight:** `arcturos health <endpoint>` standalone command +
  automatic pre-run step. Output: checklist (reachable ✓, tokenize ✓,
  model loaded ✓, ctx discovered: 262k, GPU power: visible) with ✗ reasons.

## 5. CLI surface spec

All commands share: `--json` (machine-readable output), `--db <path>`
(default `data/arcturos.sqlite`), quiet/verbose.

```
arcturos bench <endpoint> --model M [--ctx N…] [--runs-per-point N]
    [--save-artifacts] [--power-sample SECONDS] [--force-ctx]
    → prints run ID + per-point metric table

arcturos eval <suite-file> <endpoint> --model M
    [--concurrency N] [--retries N] [--force]
    → result-set ID + summary

arcturos judge <suite> --models A,B --judge-model J
    [--template T] [--retries N] [--save-artifacts]
    → judgment-set ID + summary

arcturos report compare <run-id…> [--metric M] [--baseline RUN]
arcturos report winrate <suite> [--set ID] [--format csv|json|html]
arcturos export <run-id | result-set> [--format jsonl|csv] [--out PATH]
arcturos models [list | show FINGERPRINT | alias FINGERPRINT NAME]
arcturos health <endpoint>
arcturos serve [--host 0.0.0.0] [--port 24816]
```

Exit codes: 0 ok, 1 operational failure (server unreachable, suite invalid),
2 preflight failure, 3 partial (some items failed but stored). `--json`
emits the same payload the API stores, so CLI and dashboard see identical
data.

## 6. Dashboard view specs (wireframes)

Common chrome: left rail nav (icon + label), top bar = section filter bar
+ export controls, content area, bottom status strip (DB path, server
version, last refresh). No auth. All views URL-addressable.

```
┌──────────┬──────────────────────────────────────────────┐
│ ▤ Runs   │ FILTERS  [model ▾][ctx ▾][date ▾][history ☐] │
│ ▦ Compare│ ──────────────────────────────────────────── │
│ ☰ Evals  │  run-id        model        ctx   decode tps │
│ ⚖ Judge  │  9f2c…  ds4-flash  32k       3,120  → detail │
│ ▧ Reports│  1a3b…  m3-8b      4k        5,400  → detail │
│ ◪ Models │  …                                              │
│ ☆ Bases  │                              [export JSONL/CSV] │
└──────────┴──────────────────────────────────────────────┘
```

- **Run detail:** header = run ID, model fingerprint chip, timestamps;
  metric table (nulls show `—` + reason tooltip); engine environment
  collapsible; actions: compare-from-here, pin as baseline, export, view
  artifacts.
- **Compare:** as §4.2. Chart tabs = metrics; legend = model labels with
  fingerprint tooltip.
- **Evals:** suite list → result sets → item table → drill-down (J3/J5
  share this hierarchy to avoid two navigations for the same data).
- **Judge:** setup panel (models, judge, template) + live run status +
  judgment history list with generation selector.
- **Reports:** as §4.5.
- **Models:** registry cards; alias edit inline; fingerprint copy button.

## 7. Cross-cutting UX rules

- **Nulls:** `—` in tables, never `0`; reason string always one hover or
  column away.
- **Deltas:** always paired absolute + %; colored; a delta without its base
  run ID in the tooltip is a bug.
- **IDs everywhere:** run IDs, result-set IDs, judgment-set IDs are shown,
  copyable, and accepted by every cross-view link.
- **Append-only feedback:** any action that creates a new record (re-run,
  re-judge, supersede) says so in the confirmation and names what it
  supersedes.
- **Empty states:** every view has a first-run card with the exact CLI
  command that would populate it.
- **Latency:** dashboard polls (or SSE) only the active section; long bench
  runs surface progress via the run detail page, not blocking the nav.

## 8. Open items for implementation

- Chart library confirmed as Chart.js (PRD §5); SSE vs polling for live
  progress — implementer's choice, both fit this spec.
- Multi-turn drill-down layout (message-thread view) is J3-implementation
  detail; the spec here fixes the requirement (ordered, per-turn outputs
  visible) not the pixel layout.
- Dashboard serves LAN without auth (PRD §5 / AGENTS.md §4); any future
  hardening is out of scope for this doc.
