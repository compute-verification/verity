"""V*'s hold of each Ligerito level's committed tops (`RecCap`): at level 0 one statement over the ``2^cap₀``
nodes of ``root_B``, instance ``i`` reading node ``i`` of ``rec-top-L0`` (`rec_outer.TopValue` at level 0) as its row
``top``, whatever the coins draw; at each later level ``l`` one over both reps' ``2 · 2^cap_l`` nodes of ``rec-top-L<l>``
(rep 0's level-``l`` top, then rep 1's), instance ``i`` reading row ``i``.

A level's openings (`rec_outer.stage`) read ``rec-top-L<l>`` only at the nodes their query positions name, so the nodes no
query reaches were held by no V* statement. RecCap holds every one at a fixed row: its unit reads the node's registered row
(the node's 32 words, then 32 zero words, `rec_live.top_rows`) through the row's hm96 slot, and outputs the row's words past
its node, then zeros, which V* registers as all-zero rows ``rec-cap-zero`` under public salts (`rec_residuals.zero_salts`),
as it registers every unit's outputs. So the statement holds only if every held row is a node's, and its ``b || c`` is the
firewall's: the verifier holds each table row of ``top`` to its node of ``root_B``, or of the reps' level-``l`` tops, in
the coin server's record (``Flock/RecVStar.lean``'s ``checkTopRows``) and ``top`` to the registration the level's ``out``
reads (``checkShared``, `rec_vstage.shared_rows`). The unit has no AND.

The Lean statement is ``VStar.capUnit`` (``VStar/Cap.lean``): the later levels' caps, then level 0's, last in V*'s
statements (``RecVStar.statements``, ``VStar.levelCapStatements``), and ``RecCoins.checkCap`` holds the reads file to node
``i`` in instance ``i``.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ZERO

from . import circuit as FC
from . import ir_lower as IL
from . import rec_budget as RB
from . import rec_residuals as RR
from .class_statement import ClassDefinition

TOP_WORDS = 64
OUT_WORDS = 64
NODE_WORDS = 32
TOP_VALUE = "rec-top-L0"
TOP_PREFIX = "rec-top-L"
ZERO_VALUE = "rec-cap-zero"
TEMPLATE = "rec-cap"

BIT = BL.BIT
U16 = Array(16, BIT)


def unit_name(N: int, level: int = 0) -> str:
    """The cap of ``N`` nodes at Ligerito level ``level``: ``rec-cap/n<N>`` at level 0, ``rec-cap/n<N>-l<l>`` at a later
    level (``VStar.capName``)."""
    return f"rec-cap/n{N}" + (f"-l{level}" if level else "")


def dir_name(level: int) -> str:
    """The statement's directory under V*'s (``VStar.Stmt.dir``): ``cap`` at level 0, ``cap/L<l>`` at a later level."""
    return "cap" if level == 0 else f"cap/L{level}"


def level_of(value: str) -> int:
    """``l`` of the tops value ``rec-top-L<l>`` (`rec_outer.top_value_name`)."""
    if not value.startswith(TOP_PREFIX) or not value[len(TOP_PREFIX):].isdigit():
        raise ValueError(f"RecCap holds a level's tops {TOP_PREFIX}<l>, not {value}")
    return int(value[len(TOP_PREFIX):])


def ref(top: list[int]) -> list[int]:
    """RecCap's output on a held row ``top``: its words past the node, then zero words; all zero iff the row is a node's."""
    if len(top) != TOP_WORDS:
        raise ValueError(f"a held row of {len(top)} words, not {TOP_WORDS}")
    return list(top[NODE_WORDS:]) + [0] * (OUT_WORDS - (TOP_WORDS - NODE_WORDS))


def _signature(S):
    return (("top", Array(TOP_WORDS, U16)),), Array(OUT_WORDS, U16)


def _body(B, S, top):
    bits = [top.refs[i] for i in range(len(top.refs))]
    zero = B.call(BL.constant(0)).refs[0]
    out = bits[16 * NODE_WORDS:] + [zero] * (16 * (OUT_WORDS - (TOP_WORDS - NODE_WORDS)))
    return Coll(Array(OUT_WORDS, U16), TR._refs(out))


RecCap = CompositeDefinition(
    "RecCap", 1, (), _signature, _body, conformance="exact-model-tested",
    doc="One node of the circuit table's level-0 cap, held as its registered row: the row's words past the node's 32, then "
        "32 zero words, all zero iff the row is the node's 32 words then zeros")


def definition():
    return bind(RecCap)


def unit_circuit():
    """The unit over `forms`: inputs ``in0 .. in63`` (``top``'s words, 16 bits each), outputs ``out0 .. out63``."""
    from verity.ml.circuits.boolean.forms import Circuit

    C = Circuit()
    ins = [C.input(f"in{j}", 16) for j in range(TOP_WORDS)]
    flat = [b for w in ins for b in w]
    bits = flat[16 * NODE_WORDS:] + [ZERO] * (16 * (OUT_WORDS - (TOP_WORDS - NODE_WORDS)))
    outs = [f"out{j}" for j in range(OUT_WORDS)]
    for j, nm in enumerate(outs):
        C.output(nm, bits[16 * j:16 * j + 16])
    return C, [f"in{j}" for j in range(TOP_WORDS)], outs


def lowering(N: int, level: int = 0) -> IL.Lowering:
    """`unit_circuit` as M0's unit named `unit_name`: one u16 row ``top``, one output row ``out``."""
    name = unit_name(N, level)
    C, ins, outs = unit_circuit()
    units = IL.Units(target=None, in_widths=[16] * len(ins), out_widths=[16] * len(outs),
                     in_src=[[("leaf", p) for p in range(len(ins))]], out_src=[[("ret", p) for p in range(len(outs))]],
                     signature=(name,), gates=[], arg_gates=[], ret_gates=[])
    return IL.Lowering(name, ClassDefinition(name, (TOP_WORDS,), OUT_WORDS), units, C, IL.layout(C, ins, outs))


def subcircuit(N: int, level: int = 0) -> SimpleNamespace:
    """What `circuit.compose` and `circuit.write` read of a subcircuit: its ports and its Definition, ``RecCap_v1``."""
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
    defn = definition()
    digest = program_digest(Program(defn))
    return SimpleNamespace(id=unit_name(N, level), template=SimpleNamespace(id=TEMPLATE), inputs=[port("top")],
                           outputs=[port("out")], definition=lambda **_: defn, program_digests=lambda statics: {defn.id: digest})


def stage(tops, out_dir: Path, evaluate: bool = True) -> dict:
    """RecCap over a level's tops ``tops`` (`rec_outer.TopValue` of ``rec-top-L<l>``: at level 0 both reps' one ``root_B``,
    at a later level rep 0's top then rep 1's), the verifier's own files beside the prover's (``out_dir``): ``circuit.txt``,
    the prover's ``inst.bin``, the verifier's ``pub.bin``, ``program.json``, ``partition.json`` and the reads file
    ``registered.json``. Instance ``i`` reads row ``i``."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.core.protocols.sampled_proofs.one_stage import registered as G

    level = level_of(tops.own["value"])
    N = tops.own["leaves"]
    if level == 0 and set(tops.first.values()) != {0}:
        raise ValueError(f"RecCap at level 0 holds the reps' one cap {TOP_VALUE}, not reps' rows {tops.first}")
    if level and (sorted(tops.first) != [0, 1] or tops.first[0] != 0 or not 0 < tops.first[1] < N):
        raise ValueError(f"RecCap at level {level} holds rep 0's top then rep 1's, not reps' rows {tops.first} of {N}")
    defn = definition()
    program_ = P.population_program(defn, N)
    part = P.partition(program_, P.template_instance_query(defn))
    prog = P.program_sha512(program_)
    rows, top_salts, top_entry = tops.reg.statement(range(N))
    rows = np.array(rows, np.uint16)
    low = lowering(N, level)
    want = [ref(r.tolist()) for r in rows]
    if any(any(w) for w in want):
        raise ValueError(f"a row of {tops.own['value']} is not a node's 32 words then zeros")
    agrees = None
    if evaluate:
        got, _cut, ok = low.evaluate(rows.astype(np.uint64))
        agrees = [bool((int(ok) >> i) & 1) and got[i].astype(np.uint16).tolist() == want[i] for i in range(N)]
    zero = G.Registered.commit(prog, ZERO_VALUE, [[0] * OUT_WORDS for _ in range(N)], salts=RR.zero_salts(ZERO_VALUE, N))
    out_rows, out_salts, out_entry = zero.statement(range(N))
    sub = subcircuit(N, level)
    text = FC.compose(low, sub, partition=P.digest(part), program=prog)
    read = lambda e: {**e, "word_bits": 16, "reads": list(range(N))}  # noqa: E731
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(program_)))
    (d / "partition.json").write_text(json.dumps(part))
    own = {"format": G.READS, "ports": {"top": read(tops.own), "out": read(zero.entry())}}
    RB.check(own, TEMPLATE, N)
    (d / "registered.json").write_text(json.dumps(own))
    FC.write(text, SimpleNamespace(name=TEMPLATE, subcircuit=sub), 0, N, d / "inst.bin", d / "pub.bin", None, list(range(N)),
             share_rows=True, outs={"out": np.asarray(out_rows, np.uint16)}, tables={"top": rows}, refs={"top": list(range(N))},
             salts={"top": top_salts, "out": out_salts}, registered={"top": top_entry, "out": out_entry})
    meta = json.loads(text.split("\n", 2)[1][5:])
    return {"template": defn.id, "level": level, "name": unit_name(N, level), "instances": N, "unit_agrees": agrees, "unit_rows": low.layout.useful,
            "unit_log": meta["unit_log"], "k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"],
            "per_vu": meta["per_vu"], "circuit_sha512": FC.digest(text), "program_sha512": prog,
            "partition_digest": P.digest(part), "registered_root": tops.own["root"], "zero_root": zero.root.hex()}
