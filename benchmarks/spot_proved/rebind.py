"""Bind one staged type of a stage.py values directory to another Definition with the same signature, through the match
gate again: each staged instance's inputs are read back from its rows, both Definitions are evaluated on them, and the
instance is kept only if the new one reproduces the staged outputs bit for bit (the old one must too, or the read-back is
wrong). Writes a values directory holding that one type.

    python rebind.py --values DIR --label OLD --block NEW_DEFINITION_ID --as NEW_LABEL --out DIR [--why TEXT]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

TREE = Path.cwd()
for _p in ("benchmarks/one_stage", "tools/verity/circuit_check/src", "benchmarks/numerical/python", "adapters/vllm",
           "verity/protocols/flock/prover/python", "."):
    sys.path.insert(0, str(TREE / _p))

import numpy as np  # noqa: E402


def leaves_of(unit, rows_spec, rows: dict, i: int) -> list[np.ndarray]:
    """Instance i's parameter words from its rows, `a11.unit_rows` inverted."""
    params, _ = unit.signature()
    widths = {name: t.leaf_widths() for name, t in params}
    vals: dict[str, list[int]] = {}
    for r, _s, _b, ps in rows_spec:
        bits = np.unpackbits(np.asarray(rows[r][i], dtype="<u2").view(np.uint8), bitorder="little")
        at = 0
        for p in ps:
            out = []
            for w in widths[p]:
                v = bits[at:at + w]
                out.append(int(np.dot(v.astype(np.uint64), (np.uint64(1) << np.arange(w, dtype=np.uint64)))) if w <= 64
                           else sum(int(b) << k for k, b in enumerate(v)))
                at += w
            vals[p] = out
    return [np.asarray(vals[name], dtype=np.uint64) for name, _ in params]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--as", dest="new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--why", default="")
    ap.add_argument("--full-gates", type=int, default=4_000_000)
    a = ap.parse_args()
    import a11
    import a11_kinds as K
    from verity.circuits.codec import canonical_json, encode_program
    from verity.protocols.spot.verifier.one_stage import partition as P
    from verity_vllm.pipeline import serving_rows as SRW

    from verity.circuits.codec import decode_program
    from verity.definitions.registry.catalog import load as load_registry

    src, out = Path(a.values), Path(a.out)
    doc = json.loads((src / "values.json").read_text())
    rec = doc["staged"][a.label]
    registry = load_registry()[0]
    defs = {x.id: x for x in decode_program(json.loads((src / "units.json").read_text()), registry).defs.values()}
    old = a11.live(defs[rec["unit"]], registry)
    new = SRW.template(a.block).fn
    rows_spec = K.rows_of(old)
    if K.rows_of(new) != rows_spec or old.signature() != new.signature():
        raise SystemExit(f"{new.id} does not have {old.id}'s signature and rows")
    z = np.load(src / f"{a.label}.npz")
    rows = {k[2:]: z[k] for k in z.files if k.startswith("i.")}
    outs = {k[2:]: z[k] for k in z.files if k.startswith("o.")}
    n = next(iter(rows.values())).shape[0]
    ins, louts, gate = [], [], []
    for i in range(n):
        args = leaves_of(old, rows_spec, rows, i)
        want = {o: v[i] for o, v in outs.items()}
        res = {}
        for tag, spec in (("old", old), ("new", new)):
            ev = a11.evaluate_units(spec, spec, args, 1, a.full_gates)
            _, ro = a11.unit_rows(spec, rows_spec, ev["in"], ev["out"])
            res[tag] = (all(np.array_equal(ro[o][0], want[o]) for o in want), ev)
        gate.append({"instance": i, "old": res["old"][0], "new": res["new"][0]})
        if not res["old"][0]:
            raise SystemExit(f"instance {i}: {old.id} does not reproduce its own staged outputs: the read-back is wrong")
        if res["new"][0]:
            ins += res["new"][1]["in"]
            louts += res["new"][1]["out"]
    if not ins:
        print(json.dumps(gate))
        raise SystemExit(f"no instance passes {new.id}'s match gate")
    out.mkdir(parents=True, exist_ok=True)
    nrows, nouts = a11.unit_rows(new, rows_spec, ins, louts)
    np.savez(out / f"{a.new}.npz", **{f"i.{r}": np.asarray(v, dtype=np.uint16) for r, v in nrows.items()},
             **{f"o.{o}": np.asarray(v, dtype=np.uint16) for o, v in nouts.items()})
    (out / "units.json").write_bytes(canonical_json(encode_program(P.mixed_population_program([(new, 1)]))))
    kept = [g["instance"] for g in gate if g["new"]]
    staged = {**rec, "unit": new.id, "name": a.new, "n": len(kept), "calls": [rec["calls"][i] for i in kept],
              "event": rec["calls"][kept[0]], "rebound": {"from": a.label, "unit": rec["unit"], "why": a.why, "gate": gate}}
    doc["staged"] = {a.new: staged}
    (out / "values.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    (out / "kinds.json").write_text(json.dumps({"kinds": {a.new: {"gates": new.gates, "outcome": "lowers"}}}, indent=1, sort_keys=True))
    for f in ("window.json", "stage.json"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    print(f"rebind {a.label} -> {a.new}: {len(kept)}/{n} pass {new.id}'s match gate (and {old.id}'s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
