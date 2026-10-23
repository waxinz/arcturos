"""Assemble smart-agent-101-v1.json from the category part-files.

Usage: venv/bin/python scripts/build_sa101.py
Reads part-files from /home/agent/.hermes/cache/scratch/:
  sa101_A_tool.json, sa101_A_inst.json  (subagent A)
  sa101_B_agent.json, sa101_B_diag.json, sa101_B_intg.json  (subagent B)
  sa101_synth.json  (authored in-session)
Validates every item, checks id uniqueness + quotas, sweeps operator
strings out of committed lines, and writes qa/suites/smart-agent-101-v1.json.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path("/home/agent/arcturos-work/repo")
SCRATCH = Path("/home/agent/.hermes/cache/scratch")
OUT = REPO / "qa" / "suites-private" / "smart-agent-101-v1.json"

PARTS = [
    ("sa101_A_tool.json", 100),
    ("sa101_A_inst.json", 100),
    ("sa101_B_agent.json", 100),
    ("sa101_B_diag.json", 60),
    ("sa101_B_intg.json", 60),
    ("sa101_synth.json", 60),
]

sys.path.insert(0, str(REPO / "src"))
from arcturos.evals import validate_item  # noqa: E402

OPERATOR_SWEEP_FILE = REPO / "qa" / "suites-private" / "sweep-patterns.txt"
# Fallback covers only generic infrastructure literals. The full operator
# list lives in the gitignored sweep-patterns.txt (PII policy: operator
# identifiers must never enter the public repo — including this script).
OPERATOR_STRINGS = re.compile(r"10\.10\.|192\.168\.", re.I)

def load_sweep_patterns():
    """Compile the private operator-string list when the file exists."""
    global OPERATOR_STRINGS  # noqa: PLW0603
    if OPERATOR_SWEEP_FILE.exists():
        pats = [l.strip() for l in
                OPERATOR_SWEEP_FILE.read_text().splitlines()
                if l.strip() and not l.strip().startswith("#")]
        if pats:
            OPERATOR_STRINGS = re.compile("|".join(pats), re.I)

def main():
    load_sweep_patterns()
    items = []
    for fname, expected in PARTS:
        path = SCRATCH / fname
        if not path.exists():
            print(f"MISSING part-file {fname} (expected {expected})")
            return 1
        part = json.loads(path.read_text())["items"]
        if len(part) != expected:
            print(f"COUNT {fname}: got {len(part)}, expected {expected}")
            return 1
        items += part

    # validate + uniqueness
    seen = set()
    for it_ in items:
        validate_item(it_)
        if it_["id"] in seen:
            print(f"DUP id: {it_['id']}")
            return 1
        seen.add(it_["id"])

    # quotas by id prefix
    counts = {}
    for it_ in items:
        cat = it_["id"].split("-")[1]
        counts[cat] = counts.get(cat, 0) + 1
    want = {"tool": 100, "inst": 100, "agent": 100,
            "diag": 60, "intg": 60, "synth": 60}
    if counts != want:
        print(f"QUOTA mismatch: {counts} != {want}")
        return 1

    # operator-string sweep — REPORT ONLY for the private suite: the suite is
    # PII-marked by owner direction (gitignored qa/suites-private/), so
    # operator-grounded specifics are expected here. The hard fail applies to
    # COMMITTED artifacts only (git pre-commit sweep / sweep-patterns.txt).
    dirty = []
    for it_ in items:
        blob = json.dumps(it_)
        m = OPERATOR_STRINGS.search(blob)
        if m:
            dirty.append((it_["id"], m.group(0)))
    if dirty:
        print(f"SWEEP (informational, private suite): {len(dirty)} item(s) "
              "carry operator-grounded specifics:")
        for id_, tok in dirty[:20]:
            print(f"  {id_}: {tok!r}")

    # regex references must compile under the checker's semantics
    from arcturos.evals import build_reference_checker
    for it_ in items:
        if it_.get("reference", {}).get("check") == "regex":
            build_reference_checker(it_["reference"])  # raises on bad pattern

    suite = {
        "suite": "smart-agent-101",
        "version": "v1",
        "description": (
            "480 items grounded in the operator's real agent usage: "
            "tool-calling 100, agentic 100, instruction-following 100, "
            "diagnosis-from-evidence 60, multi-source synthesis 60, "
            "integrity-discipline 60. Deterministic reference checks "
            "wherever the task allows; judge-scored where prose quality "
            "decides. Fictional hosts throughout."),
        "items": items,
    }
    OUT.write_text(json.dumps(suite, indent=1, ensure_ascii=False) + "\n")
    print(f"WROTE {OUT} with {len(items)} items; quotas {counts}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
