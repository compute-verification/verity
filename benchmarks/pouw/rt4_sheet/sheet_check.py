"""RT4-SHEET job 4: a family's data-sheet headline in Lean against `machine-model headline` on the same tree.

`lean_headline.instance()` is what the exporter writes into `Headline/SmNN.lean`; `headline.record()` is the machine
model's sheet reading. This lines them up: π* against the cheater's binding rows, ε_skip, ♭'s bound B, and the Lean's
binding branch (the larger of folded and unfolded) against the machine model's core ε and the family's literal.

    uv run python benchmarks/pouw/rt4_sheet/sheet_check.py [--tree DIR]   # DIR: a checkout (default: this one)

Exit 0 when π*, ε_skip and the identity at π* agree and the binding branch is at most the literal; 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def check(tree: Path) -> dict:
    sys.path.insert(0, str(tree / "benchmarks" / "pouw" / "machine_model"))
    lh = _load(tree / "benchmarks" / "pouw" / "lean_headline.py", "rt4_lean_headline")
    from verity_machine_model import headline as H

    out: dict = {"families": {}}
    rec = H.record(("sheet",))
    rows = {r["family"]: r for r in rec["readings"]["sheet"]["rows"]}
    for fam in lh.FAMILIES:
        i = lh.instance(fam)
        binding = "unfolded" if i.unfolded >= i.folded else "folded"
        r = rows.get(fam.key)
        res = {
            "lean": {"pistar": i.pistar, "B": str(i.B), "B_float": float(i.B), "eps_skip": float(i.eps_skip),
                     "folded": float(i.folded), "unfolded": float(i.unfolded), "bound": float(i.bound),
                     "binding": binding, "literal": float(fam.eps), "Nw": i.Nw,
                     "w_star": str(min([lh.coef(i, i.pistar, k) / om for k, om in i.writers])),
                     "phi_star": str(min(lh.load(i, i.pistar, f) for f in i.folds)),
                     "tight_per_unit": float(i.B * (1 - fam.eps)),
                     "margin": float(fam.eps - i.bound)},
            "machine_model": r and {k: r[k] for k in ("design", "vector", "q", "skip", "mand", "machine", "luck",
                                                      "core", "binding", "floor")},
        }
        agree = []
        if r:
            agree.append(("pistar_in_binding", i.pistar in r["binding"]))
            # headline.toml records ε_skip to 6 decimals; the exporter reads eps_budget's full value
            agree.append(("eps_skip", abs(float(i.eps_skip) - r["skip"]) < 1e-6))
            agree.append(("design", r["design"] == fam.design))
        agree.append(("bound_le_literal", i.bound <= fam.eps))
        res["agree"] = dict(agree)
        res["ok"] = all(v for _, v in agree) and r is not None
        out["families"][fam.key] = res
    out["ok"] = all(f["ok"] for f in out["families"].values())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", type=Path, default=Path(__file__).resolve().parents[3])
    a = ap.parse_args(argv)
    out = check(a.tree.resolve())
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
