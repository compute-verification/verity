"""85-rec-reprice's inner statement program (``85-rec-reprice.sh``): ``N`` instances of ``GemmCoordinate_v2{K, DOT}``, the
population program of that template (`one_stage.partition.population_program`, instance i is unit i), cut by
``Q_template_instance`` v0, as `rec_inner.stage` binds pod 80's RoPE d64 statement: the law's strata are the template's
instances and every unit is an instance of the circuit's own template, which Lean's verifier derives from the program.
``K`` and ``N`` come from the environment.

With REC_LINKS=1 the inner population is wired (`chain_program`): the ``N`` instances in a chain, unit u's first ``x`` word
unit u - 1's output, so ``N - 1`` cut wires, each a link port (`class_statement.Wiring`, META ``links``) the inner statement
commits under its writer's output salt, and each a row of the link table V*'s link part holds (`rec_links`,
`rec_vstage.stage_links`).

    python -m verity_flock.class_statement --program-module rec_reprice.py:program --partition q-template-instance ...
                                         the inner statement, staged over that program
    python rec_reprice.py partition DIR  the verifier's copies beside DIR's circuit.txt: the partition object META binds
                                         (partition.json), its program (program.json) and the full draw over them
                                         (rec_live.full_draw, draw.json), over `chain_program` with REC_LINKS=1;
                                         prints the bindings
    python rec_reprice.py stage-chain DIR
                                         the wired inner statement (`chain_program`) staged into DIR, every instance's own
                                         rows (FLOCK_PLAIN_LEAVES as class_statement reads it): circuit.txt, inst-N.bin,
                                         pub-N.bin, the verifier's links file (links.json, `class_statement.links_file`)
                                         and the partition's files (`partition`); prints the staging's record
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

DOT = '{"fn":"HopperBF16WgmmaDot16_v1"}'
SPEC = "q-template-instance"


def template():
    """``GemmCoordinate_v2{K, DOT}``: the Definition each unit of the inner statement is one instance of."""
    from circuit_check import targets as TG

    TG.load_registries()
    return TG.definition(f"GemmCoordinate_v2{{K={int(os.environ['K'])},DOT={DOT}}}")


def program():
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    return P.population_program(template(), int(os.environ["N"]))


def chain_program():
    """``N`` instances of the template in a chain over ``x0`` (one word), ``xs`` (``N`` rows of ``K - 1``) and ``w`` (``N``
    rows of ``K``): unit 0's ``x`` is ``x0`` then ``xs[0]``, unit u's unit u - 1's output then ``xs[u]``, its ``wrow``
    ``w[u]``; the root returns the last unit's output."""
    from verity.ml.circuits.defs import CompositeDefinition
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import coll_concat
    from verity.ml.circuits.types import Array

    t, n = template(), int(os.environ["N"])
    (_, xt), _ = t.signature()[0]
    bf, k = xt.elem, xt.n
    sig = (("x0", bf), ("xs", Array(n, Array(k - 1, bf))), ("w", Array(n, Array(k, bf))))

    def body(B, S, x0, xs, w):
        y = x0
        for u in range(n):
            y = B.call(t, coll_concat(xt, [y, xs[u]]), w[u])
        return y

    return Program(CompositeDefinition(f"RecLinkChain_N{n}", 1, (), lambda S: (sig, bf), body, register=False))


def partition(d: Path, prog=None) -> dict:
    """The partition object, program and full draw the inner statement in ``d`` binds, written beside it; refused unless META
    binds exactly that partition and program and the population is the statement's ``N`` units. ``prog``: the program
    (default `chain_program` under REC_LINKS=1, else `program`)."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity_flock import rec_live as RL

    t, n = template(), int(os.environ["N"])
    if prog is None:
        prog = chain_program() if os.environ.get("REC_LINKS") == "1" else P.population_program(t, n)
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


def chain_lanes(units, cl, tp, seed: str) -> tuple[list, list, list]:
    """Every unit's lanes in population order, each reading its writer's output as the program wires it, over root inputs
    drawn once from ``seed`` (BF16 words of normal samples, so no product overflows); every row identity the instance's own."""
    import numpy as np
    from verity_flock import class_statement as CS
    from verity_flock import partition_units as PU

    act, uc = units.activation(cl.definition), cl.representative
    groups = CS.sources(act, uc, CS.owners_of(units, cl.definition.id))
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little"))
    roots = sorted({g for u in range(tp.population) for g in tp.inputs(u)} - {g for u in range(tp.population) for g in tp.outputs(u)})
    draws = rng.standard_normal(len(roots)).astype(np.float32).view(np.uint32) >> 16
    value = dict(zip(roots, (int(x) for x in draws)))
    ins, want, rows = [[] for _ in uc.inputs], [[] for _ in uc.outputs], [[] for _ in groups]
    for u in range(tp.population):
        run = PU.evaluate(act, [value[g] for g in tp.inputs(u)])
        for port, x in zip(ins, uc.inputs):
            port.append(run[x])
        for port, j in zip(want, uc.outputs):
            port.append(run[j])
        for g, j in zip(tp.outputs(u), uc.outputs):
            value[g] = run[j]
        for i, ids in enumerate(rows):
            ids.append(("row", i, u))
    return ins, want, rows


def stage_chain(d: Path) -> dict:
    """The wired inner statement (`chain_program`) staged into ``d`` with every instance's own rows, its links file and its
    partition's files; the staging's record."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.ml.circuits import partition_object as PO
    from verity_flock import class_statement as CS
    from verity_flock import partition_units as PU

    prog = chain_program()
    query = P.template_instance_query(template())
    units = PU.PartitionUnits(prog, query=query)
    [(shape, cl)] = units.classes().items()
    tp = PO.TemplatePartition(prog, query)
    CS.cache_slot_circuits()
    CS._CTX["max_statement_bits"] = 1 << 35
    wiring = CS.Wiring(prog, query, list(range(tp.population)))
    key = hashlib.sha256(b"verity/flock/rec-reprice/chain-salts\0" + str(tp.population).encode()).digest()
    d.mkdir(parents=True, exist_ok=True)
    rec = CS.stage(units, SPEC, shape, cl, chain_lanes(units, cl, tp, f"rec-reprice/chain/{tp.population}"), d, key,
                   wiring=wiring, share_rows=False)
    text = (d / "circuit.txt").read_text()
    links = CS.links_file(prog, query, text, wiring.indices)
    (d / "links.json").write_text(json.dumps(links))
    return {"shape": shape, "stage": rec, "links": len(links["links"]), "bindings": partition(d, prog)}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("partition", "stage-chain"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "partition":
        print(json.dumps(partition(Path(sys.argv[2]))))
    else:
        print(json.dumps(stage_chain(Path(sys.argv[2]))))
