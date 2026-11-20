"""The Lean partition object's `blocks` (`verity-verify-proof partition-object`, `Flock.HmRow`) against the Python's,
`verity.circuits.partition_object`: each object's form against `validate`, and with a program its blocks against `verify`'s
cover (`_blocks_cover`) at the population the Python derived for that program, the vectors' `population`.

    python partition_object_agree.py --lean BIN --vectors template_instance_vectors.json --out DIR

Accepted on both sides or refused on both, a refusal with the Python's first code (its `keys` is Lean's keys refusal), and
an accepted object's blocks the same. Writes DIR/result.json; exits 1 unless every case agrees.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lean_run as LR

from verity.circuits import partition_object as PO

MISSING = object()


def code(error: str) -> str:
    c = error.removeprefix("partition: ").split(": ")[0]
    return "keys" if c.startswith("the object's keys are not") else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--vectors", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    v = json.loads(a.vectors.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    (_, three, q3, e3), (_, seven, q7, e7) = v["cases"][0], v["cases"][1]
    assert (e3["population"], e7["population"]) == (3, 7), (e3["population"], e7["population"])

    def obj(d, q, blocks=MISSING, **extra):
        o = PO.build(json.loads(v["descriptors"][d]), q) | extra
        return o if blocks is MISSING else o | {"blocks": blocks}

    form = [obj(three, q3, [[[0, 1], [1, 2], [2, 3]]]), obj(three, q3, []), obj(three, q3, [[]]), obj(three, q3, "0-3"),
            obj(three, q3, [[0, 1], [1, 2], [2, 3]]), obj(three, q3, [[[0, 2], [2, 2]]]), obj(three, q3, [[[-1, 2]]]),
            obj(three, q3, [[[0, 2, 4]]]), obj(three, q3, [[[0, True]]]), obj(three, q3, [[[0, 2.5]]]), obj(three, q3, [[[0, 2.0]]]),
            obj(three, q3, [[[0, 1], None]]), obj(three, q3, {}), obj(three, q3, [[[1, 2], [2, 3]]]),
            obj(three, q3, [[[0, 1], [2, 3]]]), obj(three, q3, [[[0, 2], [1, 3]]]), obj(three, q3, [[[0, 1], [1, 2], [0, 3]]]),
            obj(three, q3, [[[0, 3]], [[0, 1], [2, 3]]]), obj(three, q3, [[[0, 2], [2, 3]], [[0, 1], [1, 3]]]),
            obj(three, q3, [[[0, 3]], [[0, 2]]]), obj(three, q3, [[[0, 1], [1, 3]], [[0, 1], [1, 3]]]),
            obj(three, PO.query(), [[[0, 1], [1, 3]]]), obj(three, q3, [[[0, 3]]], grouping=[[0, 3]]), obj(three, q3)]
    held = [(three, obj(three, q3, [[[0, 3]]])), (three, obj(three, q3, [[[0, 1], [1, 3]]])), (three, obj(three, q3, [[[0, 4]]])),
            (three, obj(three, q3, [[[0, 2]]])), (three, obj(three, q3)), (seven, obj(seven, q7, [[[0, 3], [3, 7]]])),
            (seven, obj(seven, q7, [[[0, 8]]])), (seven, obj(seven, q7, [[[0, 6]]])),
            (seven, obj(seven, q7, [[[0, 7]], [[0, 3], [3, 7]]])), (seven, obj(seven, q7, [[[0, 7]], [[0, 3], [3, 8]]]))]
    population = {three: e3["population"], seven: e7["population"]}

    def verified(o, d):
        bad: dict[str, list] = {}
        v0 = PO.validate(o)
        if not v0.ok or PO.levels(o) is None:
            return v0
        PO._blocks_cover(o["blocks"], population[d], bad)
        return PO.CutVerdict(not bad, tuple(bad), {})
    files, want = [], {}
    for i, o in enumerate(form):
        f = a.out / f"form{i}.json"
        f.write_text(json.dumps({"object": o}))
        files.append(str(f))
        want[str(f)] = (PO.validate(o), None)
    for i, (d, o) in enumerate(held):
        f = a.out / f"held{i}.json"
        f.write_text(json.dumps({"object": o, "descriptor": v["descriptors"][d]}))
        files.append(str(f))
        want[str(f)] = (verified(o, d), PO.levels(o))
    p = LR.run([a.lean, "partition-object", *files])
    got = {}
    for line in p.stdout.splitlines():
        if line.startswith("PARTITIONOBJECT "):
            j = json.loads(line[len("PARTITIONOBJECT "):])
            got[j["file"]] = j
    bad = []
    for f in files:
        g, (verdict, blocks) = got.get(f) or {}, want[f]
        if verdict.ok:
            ok = g.get("ok") is True and ("descriptor" not in json.loads(Path(f).read_text())
                                          or g.get("blocks") == (None if blocks is None else [[list(b) for b in lv] for lv in blocks]))
        else:
            ok = g.get("ok") is False and code(g.get("error", "")) == verdict.codes[0]
        if not ok:
            bad.append({"file": f, "lean": g, "python": [verdict.ok, list(verdict.codes)]})
    result = {"vectors": str(a.vectors), "form": len(form), "held": len(held), "agree": len(files) - len(bad), "disagree": bad}
    (a.out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
