"""One output tree for a wired program, on statements Flock proves: the tree of (T, out), T the RoPE template, holds every
T unit's output row once, each writer member opening its own rows in it by path, and a reader member whose `x` is a link
row (META `links`) reading its writer's row by path in that tree, its reads the verifier's program's (a tree port).

The window's program (`program`) is 2n RoPE d64 instances: writer unit i (i < n) reads `x[i]` and `cs[i]`; reader unit
n + i reads writer unit n - 1 - i's output as its `x`, and `cs[n + i]`. Leaf u of the tree is unit u's output row. `stage`
writes, under `out`:

* the hidden-output circuits bound to that program and its `Q_template_instance` v0 partition: the writers'
  (`circuit.txt`) and the reader's, whose `x` is a link row (`circuit-r.txt`), with `program.json`, `partition.json`,
  `tables/`;
* the tree `RoPEHead_v1{D=64}/out` (fresh salts), the writers' `x` (`x`) and every unit's `cs` (`cs`), each a registered
  value under the program's digest;
* each statement's prover file and public file (`<name>.inst`, `<name>.pub`), the verifier's own `--registered` file for
  it (`<name>.reads.json`) and, for a reader, its `--links` file (`<name>.links.json`, no link inside the file): `m1`
  writer units [0, n/2), `m2` [n/2, n), `r` reader units [n, 2n), and the refusals `ra` (reader unit n reads row n - 2
  with that row's own path), `rb` (one bit of its row flipped under the same salt), `mc` (`m2` with its output rows under
  fresh salts) and `rd` (the reader's first two paths swapped); `r4` reads 4 rows (reader units [n, n + 4)), staged for
  its public file's size only. A reader that reads other rows registers the tree of the outputs it computes from them;
* `stage.json`: each statement's units, circuit, rows read and public file's bytes.

    python tests/verity/protocols/proofs/flock/prover/shared_tree.py --out DIR --n N [--seed S]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verity.circuits.codec import canonical_json, encode_program
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import coll_concat
from verity.circuits.types import Array
from verity.commitments import hm96
from verity.protocols.proofs.spot.verifier.one_stage import partition as P, registered as G

SEED = 20261118
FIELD = "out"
LINKS_FORMAT = "verity/flock-links/v1"


def program(n: int) -> tuple[Program, dict]:
    from verity_flock import subcircuit_templates as TM

    t = TM.rope_head(64).definition()
    (_, xt), (_, ct) = t.signature()[0]
    ret = t.signature()[1]

    def sig(S):
        return (("x", Array(n, xt)), ("cs", Array(2 * n, ct))), Array(2 * n, ret)

    def body(B, S, x, cs):
        o = [B.call(t, x[i], cs[i]) for i in range(n)]
        return coll_concat(sig(S)[1], o + [B.call(t, o[n - 1 - i], cs[n + i]) for i in range(n)])

    return Program(bind(CompositeDefinition("SharedTreeRoPE", 0, (), sig, body, register=False))), P.template_instance_query(t)


def port(reg: G.Registered, reads: dict[int, int], units: int) -> dict:
    """The verifier's own entry for a port reading `reg`: unit u reads row `reads[u]`, other units none."""
    return {**reg.entry(), **G.width_entry(G.WORD_BITS), **G.schema_entry(G.SCHEMA),
            "reads": [reads.get(u) for u in range(units)]}


def tree_port(reg: G.Registered, template: str, field: str) -> dict:
    """The verifier's own entry for a link row reading the tree of (`template`, `field`): its reads the program's."""
    return {**reg.entry(), **G.width_entry(G.WORD_BITS), **G.schema_entry(G.SCHEMA),
            "tree": {"template": template, "field": field}, "reads": []}


def stage(out: Path, n: int, seed: int = SEED) -> dict:
    from verity_flock import circuit as FC
    from verity_flock import subcircuit_templates as TM
    from verity_flock import synthetic as SY
    from verity_flock.input_sets import InputSet

    out.mkdir(parents=True, exist_ok=True)
    prog, query = program(n)
    part = P.partition(prog, query)
    pd = P.program_digest(prog)
    sub = TM.rope_head(64)
    T = sub.definition().id
    tree = f"{T}/{FIELD}"
    s = InputSet.open(SY.generate(sub, 2 * n, seed, out / "set")["path"])
    low = FC.lowering_for_set(s)
    xw = len(np.asarray(s.port("x", 0, 1)).reshape(-1))
    texts = {"circuit.txt": FC.compose(low, s.subcircuit, partition=P.digest(part), program=pd, statics=FC.set_statics(s)),
             "circuit-r.txt": FC.compose(low, s.subcircuit, partition=P.digest(part), program=pd, statics=FC.set_statics(s),
                                         links=[{"port": "x", "leaves": [0, xw]}])}
    for k, v in texts.items():
        (out / k).write_text(v)
    (out / "program.json").write_bytes(canonical_json(encode_program(prog)))
    (out / "partition.json").write_text(json.dumps(part))
    (out / "tables").mkdir(exist_ok=True)
    if low.units.cut:
        from verity_flock import ir_lower as IL

        IL.write_tables(out / "tables")

    def rope(x: np.ndarray, cs: np.ndarray) -> np.ndarray:
        return np.asarray(sub.evaluate({"x": list(x), "cs": list(cs)}, [{}] * len(x))["out"], np.uint16).reshape(len(x), -1)

    x = np.asarray(s.port("x", 0, n), np.uint16).reshape(n, -1)
    cs = np.asarray(s.port("cs", 0, 2 * n), np.uint16).reshape(2 * n, -1)
    w = rope(x, cs[:n])
    reg_x = G.Registered.commit(pd, "x", x.tolist())
    reg_cs = G.Registered.commit(pd, "cs", cs.tolist())
    shared = G.Registered.commit(pd, tree, np.concatenate([w, rope(w[::-1], cs[n:])]).tolist())
    units = 2 * n
    readers = {n + i: n - 1 - i for i in range(n)}

    def write(name: str, lo: int, hi: int, *, x_from: dict[int, int] | None = None, flip: bool = False,
              fresh_out: bool = False, swap: bool = False) -> dict:
        """Units [lo, hi): writers' `x` from `x`, readers' from the tree at `x_from` (default the wiring's)."""
        us = list(range(lo, hi))
        reader = lo >= n
        cs_pos, cs_refs = G.table(us)
        cs_rows, cs_salts, cs_e = reg_cs.statement(cs_pos)
        circuit = "circuit-r.txt" if reader else "circuit.txt"
        if reader:
            at = {**readers, **(x_from or {})}
            xin = np.asarray([w[at[u]] for u in range(n, units)], np.uint16)
            if flip:
                xin[0, 5] ^= 1
            t = G.Registered.commit(pd, tree, np.concatenate([w, rope(xin, cs[n:])]).tolist()) if flip or x_from else shared
            x_pos, x_refs = G.table([at[u] for u in us])
            _, x_salts, x_e = t.statement(x_pos)
            x_rows = np.asarray([w[p] for p in x_pos], np.uint16)
            if flip:
                x_rows[x_refs[0]] = xin[0]
            if swap:
                x_e = {**x_e, "paths": [x_e["paths"][1], x_e["paths"][0], *x_e["paths"][2:]]}
            o_rows, o_salts, o_e = t.statement(us)
            own = {"x": tree_port(t, T, FIELD), "cs": port(reg_cs, {u: u for u in range(units)}, units),
                   "out": port(t, {u: u for u in range(n, units)}, units)}
            (out / f"{name}.links.json").write_text(json.dumps(
                {"format": LINKS_FORMAT, "circuit_sha512": FC.digest(texts[circuit]), "partition": P.digest(part),
                 "ports": ["x"], "links": []}))
        else:
            x_pos, x_refs = G.table(us)
            x_rows, x_salts, x_e = reg_x.statement(x_pos)
            o_rows, o_salts, o_e = shared.statement(us)
            if fresh_out:
                o_salts = [hm96.SHA512.fresh_salt() for _ in us]
            own = {"x": port(reg_x, {u: u for u in range(n)}, units), "cs": port(reg_cs, {u: u for u in range(units)}, units),
                   "out": port(shared, {u: u for u in range(n)}, units)}
        inst, pub = out / f"{name}.inst", out / f"{name}.pub"
        FC.write(texts[circuit], s, lo, hi, inst, pub, us, share_rows=True, outs={"out": np.asarray(o_rows, np.uint16)},
                 tables={"x": np.asarray(x_rows), "cs": np.asarray(cs_rows)}, refs={"x": x_refs, "cs": cs_refs},
                 salts={"x": x_salts, "cs": cs_salts, "out": o_salts}, registered={"x": x_e, "cs": cs_e, "out": o_e})
        (out / f"{name}.reads.json").write_bytes(canonical_json({"format": G.READS, "ports": own}))
        body = pub.read_bytes().split(b"\n", 1)[1]
        return {"units": [lo, hi], "circuit": circuit, "links": reader, "x_rows_read": len(x_pos),
                "public_bytes": pub.stat().st_size, "shared_strings": [p for p in range(n) if shared.bcs[p] in body],
                "public_header_bytes": len(pub.read_bytes().split(b"\n", 1)[0]), "reads_bytes": (out / f"{name}.reads.json").stat().st_size}

    h = n // 2
    st = {"m1": write("m1", 0, h), "m2": write("m2", h, n), "r": write("r", n, 2 * n),
          "ra": write("ra", n, 2 * n, x_from={n: n - 2}), "rb": write("rb", n, 2 * n, flip=True),
          "mc": write("mc", h, n, fresh_out=True), "rd": write("rd", n, 2 * n, swap=True), "r4": write("r4", n, n + 4)}
    return {"n": n, "units": units, "tree": {"value": tree, "root": shared.root.hex(), "leaves": shared.leaves},
            "program": pd, "partition": P.digest(part), "pins": {k: FC.digest(v) for k, v in texts.items()}, "statements": st}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    print(json.dumps(stage(Path(a.out), a.n, a.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
