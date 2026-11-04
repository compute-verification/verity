"""A served tree registered as a port (`verity.core.protocols.sampled_proofs.one_stage.served`), on a statement Flock
proves: the end-to-end fixture of `test_lean_served_registered.py`, beside it in Flock's tests (which may import core).

The program (`program`) is `N` RoPE d64 template instances, instance `i` reading row `POS[i]` of the served tree `cs_table`
(`ROWS` rows of 64 u16 words) through its `cs` parameter, under `Q_template_instance` v0. `stage` writes, under `out`:

* the hidden-output circuit bound to that program and its partition (`circuit.txt`, `program.json`, `partition.json`) and
  the MUFU tables if the lowering cuts any (`tables/`);
* the tree, committed as a registered value (`Registered.commit`): each instance's `cs` row of a generated RoPE set at its
  position, the other rows random, so every instance's output is the set's own;
* the prover's `inst-N.bin` and the verifier's `pub-N.bin`, a shared-row file whose `cs` table rows carry the registered
  salts and paths (header `registered`);
* `served.json`, the tree as the service holds it (`ServedTree`: value, root, leaves, words, word_bits, domain,
  row_schema), and `registered.json`, the verifier's own `--registered` file from it (`register_served`).

`--schema` is the tree's row format (`FORMATS`): `v1` (`hm96-sha512/row/v1`), or `v2` or `seg`, for which the circuit's
input rows are bit rows of their words (`compose`'s `port_bits`), `cs` a `row/v2` or a segmented `row-seg/v1` row, and
the tree is committed in that format.

    python verity/ml/flock/tests/served_registered.py --out DIR [--seed S] [--schema v1|v2|seg]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.circuits.codec import canonical_json, encode_program
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G
from verity.core.protocols.sampled_proofs.one_stage.served import ServedTree, register_served

N = 32
ROWS = 40
#: the row of `cs_table` instance i reads: distinct, since 7 is prime to ROWS
POS = tuple((7 * i + 3) % ROWS for i in range(N))
VALUE = "cs_table"
WORDS = 64
SEED = 20261020
#: `--schema`'s names for the row formats a served tree may be
FORMATS = {"v1": G.SCHEMA, "v2": G.SCHEMA_V2, "seg": G.SCHEMA_SEG}


def program() -> tuple[Program, dict]:
    """The verifier's own program and query."""
    from verity_numerical.bench import templates as TM

    template = TM.rope_head(64).definition()
    (_, xt), (_, ct) = template.signature()[0]
    ret = template.signature()[1]

    def sig(S):
        return (("x", Array(N, xt)), (VALUE, Array(ROWS, ct))), Array(N, ret)

    def body(B, S, x, cs):
        return coll_concat(sig(S)[1], [B.call(template, x[i], cs[p]) for i, p in enumerate(POS)])

    return (Program(bind(CompositeDefinition("ServedRegisteredRoPE", 0, (), sig, body, register=False))),
            P.template_instance_query(template))


def stage(out: Path, seed: int = SEED, schema: str = G.SCHEMA) -> dict:
    from verity_flock import circuit as FC
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    out.mkdir(parents=True, exist_ok=True)
    prog, query = program()
    part = P.partition(prog, query)
    s = InputSet.open(generate(TM.rope_head(64), N, seed, out / "set")["path"])
    low = FC.lowering_for_set(s)
    x = np.asarray(s.port("x", 0, N), np.uint16).reshape(N, -1)
    bits = {} if schema == G.SCHEMA else {"port_bits": [16 * x.shape[1], 16 * WORDS],
                                          "seg_ports": frozenset({"cs"} if schema == G.SCHEMA_SEG else ())}
    text = FC.compose(low, s.subcircuit, partition=P.digest(part), program=P.program_digest(prog), statics=FC.set_statics(s),
                      **bits)
    (out / "circuit.txt").write_text(text)
    (out / "program.json").write_bytes(canonical_json(encode_program(prog)))
    (out / "partition.json").write_text(json.dumps(part))
    (out / "tables").mkdir(exist_ok=True)
    if low.units.cut:
        from verity_flock import ir_lower as IL

        IL.write_tables(out / "tables")
    cs = np.asarray(s.port("cs", 0, N), np.uint16).reshape(N, WORDS)
    table = np.random.default_rng(seed).integers(0, 1 << 16, (ROWS, WORDS))
    for i, p in enumerate(POS):
        table[p] = cs[i]
    reg = G.Registered.commit(P.program_digest(prog), VALUE, table.tolist(), schema=schema)
    positions, refs = G.table(POS)
    rows, salts, entry = reg.statement(positions)
    FC.write(text, s, 0, N, out / f"inst-{N}.bin", out / f"pub-{N}.bin", None, list(range(N)), share_rows=True,
             tables={"x": x, "cs": np.asarray(rows)}, refs={"x": list(range(N)), "cs": refs},
             salts={"x": [hm96.SHA512.fresh_salt() for _ in range(N)], "cs": salts,
                    "out": [hm96.SHA512.fresh_salt() for _ in range(N)]},
             registered={"cs": entry})
    tree = ServedTree.of(reg)
    (out / "served.json").write_text(json.dumps(dataclasses.asdict(tree), sort_keys=True))
    register_served(prog, query, [tree]).write(out / "registered.json")
    return {"n": N, "rows": ROWS, "value": VALUE, "root": tree.root, "row_schema": schema, "program": P.program_digest(prog),
            "partition": P.digest(part), "pin": FC.digest(text)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--schema", choices=sorted(FORMATS), default="v1")
    a = ap.parse_args(argv)
    print(json.dumps(stage(Path(a.out), a.seed, FORMATS[a.schema])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
