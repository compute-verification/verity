"""A served tree registered as a port (`verity.core.protocols.sampled_proofs.one_stage.served`), on a statement C-Flock
proves: the end-to-end fixture of `test_lean_served_registered.py`, beside it in C-Flock's tests (which may import core).

The program (`program`) is `N` RoPE d64 template instances, instance `i` reading row `POS[i]` of the served tree `cs_table`
(`ROWS` rows of 64 u16 words) through its `cs` parameter, under `Q_template_instance` v0. `stage` writes, under `out`:

* the hidden-output circuit bound to that program and its partition (`circuit.txt`, `program.json`, `partition.json`) and
  the MUFU tables if the lowering cuts any (`tables/`);
* the tree, committed as a registered value (`Registered.commit`): each instance's `cs` row of a generated RoPE set at its
  position, the other rows random, so every instance's output is the set's own;
* `x`, each instance's input row, committed as a registered value too (row i instance i's), and the instances' outputs
  as the registered value `rope_out` (row i instance i's output), so every port of the circuit is registered
  (`Flock.Registered.checkComplete`, which `--registered` runs);
* the prover's `inst-N.bin` and the verifier's `pub-N.bin`, a shared-row file whose `x` and `cs` table rows and output
  rows carry the registered salts and paths (header `registered`);
* `served.json` and `served-x.json`, the trees as the service holds them (`ServedTree`: value, root, leaves, words,
  word_bits, domain, row_schema), `out-port.json`, the output rows' port, and `registered.json`, the verifier's own
  `--registered` file from them (`registered_file`: `register_served`'s ports and the output port).

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
#: the registered value whose rows are the instances' outputs, row i instance i's
OUT_VALUE = "rope_out"
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
    bit_rows = G.SCHEMA if schema == G.SCHEMA else G.SCHEMA_V2
    reg_x = G.Registered.commit(P.program_digest(prog), "x", x.tolist(), schema=bit_rows)
    x_rows, x_salts, x_entry = reg_x.statement(range(N))
    outs = np.asarray(s.port("out", 0, N), np.uint16).reshape(N, -1)
    reg_out = G.Registered.commit(P.program_digest(prog), OUT_VALUE, outs.tolist(), schema=bit_rows)
    out_rows, out_salts, out_entry = reg_out.statement(range(N))
    FC.write(text, s, 0, N, out / f"inst-{N}.bin", out / f"pub-{N}.bin", None, list(range(N)), share_rows=True,
             outs={"out": np.asarray(out_rows, np.uint16)},
             tables={"x": np.asarray(x_rows), "cs": np.asarray(rows)}, refs={"x": list(range(N)), "cs": refs},
             salts={"x": x_salts, "cs": salts, "out": out_salts},
             registered={"cs": entry, "x": x_entry, "out": out_entry})
    tree, x_tree = ServedTree.of(reg), ServedTree.of(reg_x)
    out_port = {**reg_out.entry(), **G.width_entry(G.WORD_BITS), **G.schema_entry(bit_rows), "reads": list(range(N))}
    (out / "served.json").write_text(json.dumps(dataclasses.asdict(tree), sort_keys=True))
    (out / "served-x.json").write_text(json.dumps(dataclasses.asdict(x_tree), sort_keys=True))
    (out / "out-port.json").write_text(json.dumps(out_port, sort_keys=True))
    write_registered(registered_file(prog, query, tree, x_tree, out_port), out / "registered.json")
    return {"n": N, "rows": ROWS, "value": VALUE, "root": tree.root, "x_root": x_tree.root, "out_root": out_port["root"],
            "row_schema": schema, "program": P.program_digest(prog), "partition": P.digest(part), "pin": FC.digest(text)}


def load(stage_dir: Path) -> tuple[ServedTree, ServedTree, dict]:
    """A stage's served trees `cs_table` and `x`, and its output rows' port, as `stage` wrote them."""
    return (ServedTree(**json.loads((stage_dir / "served.json").read_text())),
            ServedTree(**json.loads((stage_dir / "served-x.json").read_text())),
            json.loads((stage_dir / "out-port.json").read_text()))


def registered_file(prog: Program, query: dict, tree: ServedTree, x_tree: ServedTree, out_port: dict) -> dict:
    """The verifier's own `--registered` file: `register_served`'s ports for the served trees `cs_table` and `x`, and the
    output rows' port, so every port of the circuit is registered (`Flock.Registered.checkComplete`)."""
    sr = register_served(prog, query, [tree, x_tree])
    return {**sr.file, "ports": {**sr.ports, "out": out_port}}


def write_registered(doc: dict, path: Path) -> Path:
    path.write_bytes(canonical_json(doc))
    return path


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
