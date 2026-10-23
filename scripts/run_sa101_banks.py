"""Run sa101 item banks -> JSON part-files, with _good/_bad probes.

Each bank module defines ITEMS_* (list of item dicts) and may carry
_good/_bad keys per item: outputs that MUST pass / MUST fail the item's
reference check. The builder pops them, runs both probes, and reports.
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/agent/arcturos-work/repo/src")
from arcturos.evals import validate_item, score_output  # noqa: E402

BANKS = [
    ("sa101/tools_a.py", "ITEMS_A", "sa101_A_tool.json"),
    ("sa101/tools_b.py", "ITEMS_B", "sa101_A_tool.json"),
    ("sa101/inst_a.py", "ITEMS_IA", "sa101_A_inst.json"),
    ("sa101/inst_b.py", "ITEMS_IB", "sa101_A_inst.json"),
    ("sa101_build/agent_a.py", "ITEMS", "sa101_B_agent.json"),
    ("sa101_build/agent_b.py", "ITEMS", "sa101_B_agent.json"),
    ("sa101_build/agent_c.py", "ITEMS", "sa101_B_agent.json"),
    ("sa101_build/agent_d.py", "ITEMS", "sa101_B_agent.json"),
    ("sa101_build/agent_e.py", "AGENT_E", "sa101_B_agent.json"),
    ("sa101_build/diag_a.py", "DIAG_A", "sa101_B_diag.json"),
    ("sa101_build/diag_b.py", "DIAG_B", "sa101_B_diag.json"),
    ("sa101_build/intg_a.py", "INTG_A", "sa101_B_intg.json"),
    ("sa101_build/intg_b.py", "INTG_B", "sa101_B_intg.json"),
]
SCRATCH = Path("/home/agent/.hermes/cache/scratch")


def load_bank(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(
        path.stem.replace("-", "_") + str(abs(hash(str(path))) % 1000), path)
    mod = importlib.util.module_from_spec(spec)
    # banks import helpers from their own directory; drop stale caches so a
    # second bank's helpers.py is not shadowed by the first bank's module
    for stale in ("helpers",):
        sys.modules.pop(stale, None)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("helpers", None)
        sys.path.remove(str(path.parent))
    return mod


def main():
    parts = {}
    probe_fail = []
    total = 0
    for bank_path, var, out_name in BANKS:
        mod = load_bank(SCRATCH / bank_path)
        items = getattr(mod, var, None)
        if items is None:
            for name in dir(mod):
                if name.startswith(("ITEMS", "AGENT_", "DIAG_", "INTG_")):
                    items = getattr(mod, name)
                    break
        if not items:
            print(f"NO ITEMS in {bank_path}")
            continue
        # mini-dict banks (sa101_build) need their builder applied first
        if items and isinstance(items[0], dict) and "kind" in items[0]:
            probes, built = {}, []
            for d in items:
                d = dict(d)
                g, b = d.pop("_good", []), d.pop("_bad", [])
                if "turns" in d:
                    out = (mod.build_mt_other(d)
                           if "dimension" in d and d["dimension"] != "agentic"
                           else mod.build_agent(d))
                else:
                    out = mod.build_single(d)
                probes[out["id"]] = (g, b)
                built.append(out)
            items = built
        else:
            probes = {}
        clean = []
        for it_ in items:
            it_ = dict(it_)
            good = it_.pop("_good", None)
            bad = it_.pop("_bad", None)
            if good is None:  # mini-dict: probes were carried through build
                good, bad = probes.get(it_["id"], ([], []))
            try:
                validate_item(it_)
            except Exception as e:
                probe_fail.append((bank_path, it_.get("id"), f"validate: {e}"))
                continue
            ref = it_.get("reference")
            if ref:
                for g in good:
                    ok, _ = score_output(g, ref)
                    if not ok:
                        probe_fail.append((bank_path, it_["id"],
                                           f"_good failed: {g[:60]!r}"))
                for b in bad:
                    ok, _ = score_output(b, ref)
                    if ok:
                        probe_fail.append((bank_path, it_["id"],
                                           f"_bad PASSED: {b[:60]!r}"))
            clean.append(it_)
        parts.setdefault(out_name, []).extend(clean)
        total += len(clean)
        print(f"{bank_path}: {len(clean)} items ok")

    for out_name, items in parts.items():
        (SCRATCH / out_name).write_text(json.dumps({"items": items}, indent=1))
        print(f"WROTE {out_name}: {len(items)}")

    if probe_fail:
        print(f"\nPROBE FAILURES: {len(probe_fail)}")
        for p in probe_fail[:40]:
            print("  ", *p)
        return 1
    print(f"\nALL PROBES PASS; total {total} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
