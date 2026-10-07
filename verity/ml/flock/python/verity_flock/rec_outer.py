"""V*'s outer statements (hiding model architectures, lane rec-v0): the inner session's Ligerito openings, one C-Flock
statement per level, each instance a `RecOpen` of one opening, its output registered by the verifier.

A level's openings share one shape (its rows' lanes, ``H = depth - c`` siblings below its cap of ``2^c`` nodes), so each level
is one template, ``RecOpen{LANES, H}``, and one statement over that level's openings of both reps, in rep and query order
(M0 proves one template per statement). Its program is `population_program` of the template over the level's openings, its
partition that program's ``Q_template_instance`` v0. Every instance reads two fresh private rows, the opened row and its
``aux`` (salt, siblings, the position's low bits: `rec_open.pack`), and returns one hidden output row (§16.13).

The verifier knows what each output must be: the cap node at the opening's position and the position's low bits
(`rec_open.expected`), both from the inner session's public record (its caps and the query coins it drew). It registers
those rows as the value ``rec-L<level>`` under salts it derives itself (`expected_salts`: the rows are public, the salts only
keep each row's ``b || c`` its own), and its reads file puts instance ``i``'s output at row ``i`` (PROTOCOL.md §16.13
"Registered reads", `verity.experimental.protocols.sampled_proofs.one_stage.registered`). An instance is satisfiable only if its private rows open
to that registered row, so the statement holds only if every opening climbs to the cap node its coin names.

With ``link_rows``, the opened rows are not fresh: the prover registers them first, as the value ``rec-rows-L<level>`` under
its own salts (what it would send before the outer coins), and the statement's ``row`` port reads row ``i`` of that value in
instance ``i``. The verifier then checks every opened row against that root, which is how another statement (the inner
verifier's algebra) can read the same rows, and how V* reads rows the prover committed earlier (the inner messages under
their c_k, com_P).

`stage` writes one level's ``circuit.txt``, the prover's ``inst.bin``, the verifier's ``pub.bin``, ``program.json``,
``partition.json`` and the reads file ``registered.json``; ``python -m verity_flock.rec_outer INNER OUT`` stages every level
of an inner session (`verity_flock.rec_vstar.load`'s directory).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from verity.ml.commitments import hm96
from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.program import Program

from . import circuit as FC
from . import ir_lower as IL
from . import rec_open as R
from . import rec_vstar as V
from .class_statement import ClassDefinition

EXPECTED_SALT = b"verity/rec-v0/expected-salt/v0"


def lowering(LANES: int, H: int) -> IL.Lowering:
    """`rec_open.unit_circuit` as M0's unit: ports ``row`` and ``aux`` (u16 words, whole SHA-512 blocks) and ``out`` (one row)."""
    name = f"rec-open/l{LANES}-h{H}"
    C, ins, outs = R.unit_circuit(LANES, H)
    w_in, w_out = len(ins), len(outs)
    units = IL.Units(target=None, in_widths=[16] * w_in, out_widths=[16] * w_out, in_src=[[("leaf", p) for p in range(w_in)]],
                     out_src=[[("ret", p) for p in range(w_out)]], signature=(name,), gates=[], arg_gates=[], ret_gates=[])
    return IL.Lowering(name, ClassDefinition(name, (R.row_words(LANES), R.aux_words(H)), w_out), units, C, IL.layout(C, ins, outs))


def subcircuit(LANES: int, H: int) -> SimpleNamespace:
    """What `circuit.compose` and `circuit.write` read of a subcircuit: its ports and its Definition, ``RecOpen{LANES, H}``."""
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")
    defn = R.definition(LANES, H)
    digest = program_digest(Program(defn))
    return SimpleNamespace(id=f"rec-open/l{LANES}-h{H}", template=SimpleNamespace(id="rec-open"), inputs=[port("row"), port("aux")],
                           outputs=[port("out")], definition=lambda **_: defn, program_digests=lambda statics: {defn.id: digest})


def levels(queries: list[V.Query]) -> list[dict]:
    """The openings by level, both reps in rep and query order, each level's shape checked to be one template's."""
    by: dict[int, list[V.Query]] = {}
    for q in queries:
        by.setdefault(q.level, []).append(q)
    out = []
    for lv, qs in sorted(by.items()):
        shapes = {(len(q.row), q.depth - q.c) for q in qs}
        if len(shapes) != 1:
            raise ValueError(f"level {lv}: openings of {sorted(shapes)} (lanes, siblings); a statement is one template")
        ((lanes, h),) = shapes
        out.append({"level": lv, "LANES": lanes, "H": h, "queries": sorted(qs, key=lambda q: (q.rep, q.j))})
    return out


def expected_salts(value: str, n: int) -> list[bytes]:
    return [hashlib.shake_256(EXPECTED_SALT + value.encode() + b"\0" + i.to_bytes(4, "little")).digest(hm96.SHA512.salt_bytes)
            for i in range(n)]


def stage(level: dict, out_dir: Path, link_rows: bool = False) -> dict:
    """One level's statement, the verifier's own files beside the prover's (``out_dir``), and what it costs."""
    from verity.experimental.protocols.sampled_proofs.one_stage import partition as P
    from verity.experimental.protocols.sampled_proofs.one_stage import registered as G

    LANES, H, qs = level["LANES"], level["H"], level["queries"]
    n = len(qs)
    value = f"rec-L{level['level']}"
    defn = R.definition(LANES, H)
    program = P.population_program(defn, n)
    query = P.template_instance_query(defn)
    part = P.partition(program, query)
    prog = P.program_sha512(program)
    packed = [R.pack(q.row, q.salt, q.sibs, q.pos) for q in qs]
    rows = np.array([r for r, _ in packed], np.uint16)
    aux = np.array([a for _, a in packed], np.uint16)
    want = [R.expected(q.cap_node, q.pos, H) for q in qs]
    low = lowering(LANES, H)
    got, _cut, ok = low.evaluate(np.concatenate([rows, aux], axis=1).astype(np.uint64))
    opened = [bool((int(ok) >> i) & 1) and got[i].astype(np.uint16).tolist() == want[i] for i in range(n)]
    sub = subcircuit(LANES, H)
    text = FC.compose(low, sub, partition=P.digest(part), program=prog)
    reg = G.Registered.commit(prog, value, want, salts=expected_salts(value, n))
    out_rows, out_salts, entry = reg.statement(range(n))
    own = {"format": G.READS, "ports": {"out": {**reg.entry(), "word_bits": 16, "reads": list(range(n))}}}
    fresh = lambda: [hm96.SHA512.fresh_salt() for _ in range(n)]
    salts, registered = {"row": fresh(), "aux": fresh(), "out": out_salts}, {"out": entry}
    linked = None
    if link_rows:
        linked = G.Registered.commit(prog, f"rec-rows-L{level['level']}", rows.tolist())
        _rows, salts["row"], registered["row"] = linked.statement(range(n))
        own["ports"]["row"] = {**linked.entry(), "word_bits": 16, "reads": list(range(n))}
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(program)))
    (d / "partition.json").write_text(json.dumps(part))
    (d / "registered.json").write_text(json.dumps(own))
    FC.write(text, SimpleNamespace(name=value, subcircuit=sub), 0, n, d / "inst.bin", d / "pub.bin", None, list(range(n)),
             share_rows=True, outs={"out": np.asarray(out_rows, np.uint16)}, tables={"row": rows, "aux": aux},
             refs={"row": list(range(n)), "aux": list(range(n))}, salts=salts, registered=registered)
    meta = json.loads(text.split("\n", 2)[1][5:])
    return {"level": level["level"], "value": value, "template": defn.id, "LANES": LANES, "H": H, "instances": n,
            "opened": sum(opened), "not_opened": [(q.rep, q.j) for q, o in zip(qs, opened) if not o],
            "unit_rows": low.layout.useful, "unit_log": meta["unit_log"], "k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"],
            "per_vu": meta["per_vu"], "compressions": LANES // 8 + 5 + 2 * H, "circuit_sha512": FC.digest(text),
            "program_sha512": prog, "partition_digest": P.digest(part), "registered_root": reg.root.hex(),
            "rows_root": linked.root.hex() if linked else None}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("inner", help="an inner session directory (public.json, private.json, session/)")
    ap.add_argument("out", help="one directory per level, L<level>/")
    ap.add_argument("--levels", default=None, help="comma-separated levels (default: every level)")
    ap.add_argument("--link-rows", action="store_true", help="the opened rows read from the prover's registration of them")
    a = ap.parse_args(argv)
    public, _private, proofs = V.load(Path(a.inner))
    want = None if a.levels is None else {int(x) for x in a.levels.split(",")}
    res = []
    for lv in levels(V.queries(public, proofs)):
        if want is None or lv["level"] in want:
            res.append(stage(lv, Path(a.out) / f"L{lv['level']}", a.link_rows))
            print(json.dumps(res[-1], sort_keys=True), flush=True)
    return 0 if all(not r["not_opened"] for r in res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
