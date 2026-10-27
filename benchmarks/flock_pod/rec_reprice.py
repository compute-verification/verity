"""85-rec-reprice's inner statement program (``85-rec-reprice.sh``): ``N`` instances of ``GemmCoordinate_v2{K, DOT}``, the
population program of that template (`one_stage.partition.population_program`, instance i is unit i), cut by
``Q_template_instance`` v0, as `rec_inner.stage` binds pod 80's RoPE d64 statement: the law's strata are the template's
instances and every unit is an instance of the circuit's own template, which Lean's verifier derives from the program.
``K`` and ``N`` come from the environment.

    python -m verity_flock.class_statement --program-module rec_reprice.py:program --partition q-template-instance ...
                                         the inner statement, staged over that program
    python rec_reprice.py partition DIR  the verifier's copies beside DIR's circuit.txt: the partition object META binds
                                         (partition.json), its program (program.json) and the full draw over them
                                         (rec_live.full_draw, draw.json); prints the bindings
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DOT = '{"fn":"HopperBF16WgmmaDot16_v1"}'


def template():
    """``GemmCoordinate_v2{K, DOT}``: the Definition each unit of the inner statement is one instance of."""
    from circuit_check import targets as TG

    TG.load_registries()
    return TG.definition(f"GemmCoordinate_v2{{K={int(os.environ['K'])},DOT={DOT}}}")


def program():
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    return P.population_program(template(), int(os.environ["N"]))


def partition(d: Path) -> dict:
    """The partition object, program and full draw the inner statement in ``d`` binds, written beside it; refused unless META
    binds exactly that partition and program and the population is the statement's ``N`` units."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity_flock import rec_live as RL

    t, n = template(), int(os.environ["N"])
    prog = P.population_program(t, n)
    part = P.partition(prog, P.template_instance_query(t))
    with open(d / "circuit.txt") as f:
        f.readline()
        meta = json.loads(f.readline()[5:])
    if (meta["partition"]["digest"], meta["program_sha512"]) != (P.digest(part), part["program"]):
        raise SystemExit(f"{d}: META binds partition {meta['partition']['digest']} over {meta['program_sha512']}, not the template's")
    draw = RL.full_draw(prog, part["query"])
    if draw["population"] != n:
        raise SystemExit(f"the partition's population is {draw['population']}, not the statement's {n} units")
    (d / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (d / "program.json").write_bytes(canonical_json(encode_program(prog)))
    (d / "draw.json").write_text(json.dumps(draw))
    return {"partition": P.digest(part), "program": part["program"], "template": list(meta["program_digests"]),
            "population": draw["population"]}


if __name__ == "__main__":
    if sys.argv[1:2] != ["partition"] or len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(partition(Path(sys.argv[2]))))
