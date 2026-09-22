# ADR 003 — Run naming: labels vs. measurements, and the eval-run entity

**Date:** 2026-09-22
**Status:** Accepted
**Deciders:** Alexei (owner), Mantis (implementation)

## Context

Runs and eval replays were identified only by server URL + model
fingerprint + row id. Comparing "run 26 vs run 30" required remembering
what each sweep was. The owner asked for optional human names on every
benchmark run and eval run, defaulting to `server · model · context
targets`, overridable before kick-off or after the fact, and surfaced as
the /runs second column, chart legends, and diff headers.

Two structural questions fell out:

1. **How can a stored row be renamed in an append-only store?** The
   schema enforces immutability with triggers; a blanket "renames are
   fine" would erode the guarantee that measurements never change.
2. **What is "an eval run"?** Eval replays stored loose
   `eval_results` rows keyed by suite + model fingerprint — there was no
   batch entity to hang a name on, and grouping by (suite, fingerprint)
   conflates two separate replays of the same suite against the same
   model.

## Decision

### 1. Labels are mutable; measurements are not

A **label** is presentation metadata a human attaches to a record. A
**measurement** is anything a server or sampler produced. The
append-only guarantee now reads: *measurements are immutable; labels are
the owner's to set.*

Concretely:

- `runs.name` (TEXT NULL) is a label column. The `trg_runs_no_update`
  trigger allows UPDATEs **only** when every other column is unchanged
  (`IS NOT` comparison, NULL-safe); any other mutation aborts.
- `eval_runs.name` follows the same pattern via `trg_eval_runs_no_update`.
- This is the same precedent as `models.alias` (ADR 002 era): the
  registry alias was the first label column; naming generalizes it.
- Rename history is NOT kept. The column holds the current label. If a
  history is ever wanted, it can be added as an append-only
  `run_names` log without disturbing this contract.

Default names are computed at dispatch time and written into
`runs.name` at store, so /runs shows the humanized default instead of a
sea of `—`. NULL remains possible (runs created before this feature,
manual rows), and the UI falls back to `—` / fingerprint rendering for
those.

### 2. Eval runs get an entity

New table `eval_runs`:

    id INTEGER PK AUTOINCREMENT
    suite_id  INTEGER NOT NULL REFERENCES eval_suites(id)
    model_fingerprint TEXT NOT NULL
    name      TEXT                      -- label, trigger-guarded
    created_at TEXT NOT NULL

`eval_results.eval_run_id` (INTEGER NULL, idempotent migration) stamps
which replay batch produced each row. One `dispatch_eval` call = one
`eval_runs` row; every result it stores carries the id. Historical rows
grandfather NULL (same migration pattern as `runs.status`).

This also fixes a latent reporting ambiguity: two replays of suite 1
against the same model were previously indistinguishable at the row
level; now each batch is addressable.

### 3. Where the name appears

- `/runs` — second column (after ID), `—` when unset.
- Run detail — meta row + rename control.
- `/compare` — chart legends and the 📌 baseline overlay label prefer
  the name, falling back to the model label (`alias (short-fp)`).
- `/diff` — "Metrics — A vs B" headers use names.
- CSV exports — `name` column on the runs list and per-run benchmark
  exports; compare payload rows carry it too.
- Eval reports grouping by eval-run name was considered and deferred —
  suite-level reports remain the primary eval surface.

## Consequences

- **Trigger discipline:** every append-only table now carries at most
  one label-column exception, each guarded by an explicit trigger that
  whitelists the label and blacklists everything else. Adding a new
  mutable column to any of these tables requires touching its trigger —
  deliberately annoying.
- **Migration:** three idempotent tuples (runs.name, eval_runs table,
  eval_results.eval_run_id); taupo's live DB migrates on next boot
  without touching stored rows.
- **API surface:** `PATCH /api/runs/{id}/name` joins the short list of
  non-POST/GET routes; the editable-routes contract test pins it.
- **Display units:** unchanged — names are strings, never rounded,
  never parsed.

## Alternatives considered

- **Append-only rename log (newest-wins)** like run_visibility: more
  faithful to "append-only everything", but a name is not a fact about
  the world — it is a pointer to a human intention, and newest-wins
  tables for every label would triple the join cost of every listing
  for no provenance benefit. The trigger-whitelist gives the same
  integrity guarantee (nothing else can change) at a fraction of the
  complexity. Revisit if rename auditing is ever actually needed.
- **Skip eval naming** (bench runs only): rejected — the owner
  explicitly asked for eval runs, and the missing batch entity was the
  root cause of not having names there.
- **Store default names as computed views** (render-time fallback):
  rejected — /runs would show `—` for every unnamed run, which is
  exactly the noise the feature is meant to remove.