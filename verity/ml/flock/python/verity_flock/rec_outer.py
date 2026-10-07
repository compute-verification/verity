"""V*'s outer statements (hiding model architectures, lane rec-v0): the inner session's Ligerito openings, one C-Flock
statement per level, each instance a `RecOpen` of one opening, its node registered by the verifier and its running sum read
from the prover's registration of them.

A level's openings share one shape (its rows' lanes, ``H = depth - c`` siblings below its cap of ``2^c`` nodes), so each level
is one template, ``RecOpen{LANES, H}``, and one statement over that level's openings of both reps, in rep and query order
(M0 proves one template per statement). Its program is `population_program` of the template over the level's openings, its
partition that program's ``Q_template_instance`` v0. Every instance reads two fresh private rows, the opened row and its
``aux`` (salt, siblings, the position's low bits: `rec_open.pack`), one public input, its coefficients ``coef``, and two
registered rows, ``acc`` in and ``acc_out`` out (§16.13).

The verifier knows what each ``out`` must be: the cap node at the opening's position and the position's low bits
(`rec_open.expected`), both from the inner session's public record (its caps and the query coins it drew). It registers
those rows as the value ``rec-L<level>`` under salts it derives itself (`expected_salts`: the rows are public, the salts only
keep each row's ``b || c`` its own), and its reads file puts instance ``i``'s output at row ``i`` (PROTOCOL.md §16.13
"Registered reads", `verity.experimental.protocols.sampled_proofs.one_stage.registered`). An instance is satisfiable only if its private rows open
to that registered row, so the statement holds only if every opening climbs to the cap node its coin names.

**The running sums** (`Chain`). The openings of both reps, rep-major, then by level and query (`chain_order`), are positions
``0 .. N - 1`` of one chain; the prover registers its ``N + 1`` rows as the value ``rec-acc`` (`ACC_VALUE`) under the V*
session's program string (`session_program`, which every statement reading the value shares), row 0 zero and row ``p + 1``
opening ``p``'s ``acc_out`` on row ``p`` (`rec_open.acc_ref`). Opening ``p``'s instance reads ``acc`` at row ``p`` and
``acc_out`` at row ``p + 1``, so every level's statement holds only if each row is the one before it plus its opening's
consistency terms, and V*'s algebra (`verity_flock.rec_residuals`) reads a rep's first and last rows. The coefficients are
the verifier's (`coefficients`: the final check's κ at the rep's coins times the level's eq(α)[j],
`verity_flock.rec_algebra.verifier_rows`).

With ``link_rows``, the opened rows are not fresh: the prover registers them first, as the value ``rec-rows-L<level>`` under
its own salts (what it would send before the outer coins), and the statement's ``row`` port reads row ``i`` of that value in
instance ``i``. The verifier then checks every opened row against that root, which is how another statement can read the
same rows.

`stage` writes one level's ``circuit.txt``, the prover's ``inst.bin``, the verifier's ``pub.bin`` and ``public-inputs.bin``,
``program.json``, ``partition.json`` and the reads file ``registered.json``; `verity_flock.rec_vstage` stages every level
and the algebra of an inner session.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from verity.ml.commitments import hm96
from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.program import Program
from verity.ml.circuits.ops.boolean.gf2k import mul

from . import circuit as FC
from . import ir_lower as IL
from . import rec_open as R
from . import rec_vstar as V
from .class_statement import ClassDefinition

EXPECTED_SALT = b"verity/rec-v0/expected-salt/v0"
SESSION_TAG = b"verity/rec-v0/v-star/v0\0"
ACC_VALUE = "rec-acc"
COEF_PUBLIC = {"seed": "the inner session's public record (rec/inner-transcript/v0), the opening = instance",
               "derivation": "verity/rec-v0/opening-coefficients/v0"}


def session_program(record: bytes) -> str:
    """The program string of the registrations V*'s statements share (the running sums, the inner messages): SHA-512 of the
    tag and the inner session's public record."""
    return hashlib.sha512(SESSION_TAG + hashlib.sha512(record).digest()).hexdigest()


def lowering(LANES: int, H: int) -> IL.Lowering:
    """`rec_open.unit_circuit` as M0's unit: ports ``row``, ``aux``, ``acc`` and ``coef`` (u16 words, whole SHA-512 blocks), and
    ``out`` and ``acc_out`` (one row each)."""
    name = f"rec-open/l{LANES}-h{H}"
    C, ins, outs = R.unit_circuit(LANES, H)
    w_in, w_out = len(ins), len(outs)
    units = IL.Units(target=None, in_widths=[16] * w_in, out_widths=[16] * w_out, in_src=[[("leaf", p) for p in range(w_in)]],
                     out_src=[[("ret", p) for p in range(w_out)]], signature=(name,), gates=[], arg_gates=[], ret_gates=[])
    defn = ClassDefinition(name, (R.row_words(LANES), R.aux_words(H), R.ACC_WORDS, R.coef_words(LANES)), w_out,
                           (R.OUT_WORDS, R.ACC_WORDS))
    return IL.Lowering(name, defn, units, C, IL.layout(C, ins, outs))


def subcircuit(LANES: int, H: int) -> SimpleNamespace:
    """What `circuit.compose` and `circuit.write` read of a subcircuit: its ports and its Definition, ``RecOpen{LANES, H}``."""
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")
    defn = R.definition(LANES, H)
    digest = program_digest(Program(defn))
    return SimpleNamespace(id=f"rec-open/l{LANES}-h{H}", template=SimpleNamespace(id="rec-open"),
                           inputs=[port(nm) for nm in ("row", "aux", "acc", "coef")], outputs=[port("out"), port("acc_out")],
                           definition=lambda **_: defn, program_digests=lambda statics: {defn.id: digest})


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


def chain_order(queries: list[V.Query]) -> list[tuple[int, int, int]]:
    """The chain's positions: every opening's (rep, level, j), rep-major, then by level and query."""
    return sorted((q.rep, q.level, q.j) for q in queries)


def coefficients(kappa: list, ea: list[int], level: int, j: int) -> list[int]:
    """Opening j of a level's ``coef`` words: ``c_r[l] = κ[level][r][l] · eq(α)[j]`` (`rec_open.cons_structure`'s order)."""
    return R.words([mul(k, ea[j]) for r in range(2) for k in kappa[level][r]])


@dataclass
class Chain:
    """The prover's running sums: the registered value (`ACC_VALUE`, rows ``0 .. N``) and each opening's position."""

    reg: object
    at: dict[tuple[int, int, int], int]

    @classmethod
    def build(cls, program: str, queries: list[V.Query], coefs: dict[tuple[int, int, int], list[int]],
              salts: list[bytes] | None = None) -> "Chain":
        from verity.experimental.protocols.sampled_proofs.one_stage import registered as G

        by = {(q.rep, q.level, q.j): q for q in queries}
        order = chain_order(queries)
        rows = [[0] * R.ACC_WORDS]
        for key in order:
            rows.append(R.acc_ref(rows[-1], coefs[key], R.words(by[key].row)))
        return cls(G.Registered.commit(program, ACC_VALUE, rows, salts), {k: p for p, k in enumerate(order)})

    def sums(self, position: int) -> list[int]:
        """Row ``position``'s two running sums."""
        return R.elements(self.reg.rows[position][:16])

    def rep_ends(self, rep: int) -> tuple[int, int]:
        """The rows before a rep's first opening and after its last."""
        ps = [p for (r, _l, _j), p in self.at.items() if r == rep]
        return min(ps), max(ps) + 1


def stage(level: dict, out_dir: Path, chain: Chain, coefs: dict[tuple[int, int, int], list[int]], link_rows: bool = False) -> dict:
    """One level's statement, the verifier's own files beside the prover's (``out_dir``), and what it costs. ``chain``: the
    prover's running sums; ``coefs``: each opening's ``coef`` words (the verifier's)."""
    from verity.experimental.protocols.sampled_proofs.one_stage import partition as P
    from verity.experimental.protocols.sampled_proofs.one_stage import registered as G

    LANES, H, qs = level["LANES"], level["H"], level["queries"]
    n = len(qs)
    value = f"rec-L{level['level']}"
    defn = R.definition(LANES, H)
    program_ = P.population_program(defn, n)
    query = P.template_instance_query(defn)
    part = P.partition(program_, query)
    prog = P.program_sha512(program_)
    keys = [(q.rep, q.level, q.j) for q in qs]
    packed = [R.pack(q.row, q.salt, q.sibs, q.pos) for q in qs]
    rows = np.array([r for r, _ in packed], np.uint16)
    aux = np.array([a for _, a in packed], np.uint16)
    coef = np.array([coefs[k] for k in keys], np.uint16)
    at = [chain.at[k] for k in keys]
    acc_rows, acc_salts, acc_entry = chain.reg.statement(at)
    out_acc, out_acc_salts, out_acc_entry = chain.reg.statement([p + 1 for p in at])
    want = [R.expected(q.cap_node, q.pos, H) for q in qs]
    low = lowering(LANES, H)
    flat = np.concatenate([rows, aux, np.array(acc_rows, np.uint16), coef], axis=1).astype(np.uint64)
    got, _cut, ok = low.evaluate(flat)
    opened = [bool((int(ok) >> i) & 1) and got[i, :R.OUT_WORDS].astype(np.uint16).tolist() == want[i] for i in range(n)]
    summed = [bool((int(ok) >> i) & 1) and got[i, R.OUT_WORDS:].astype(np.uint16).tolist() == list(out_acc[i]) for i in range(n)]
    sub = subcircuit(LANES, H)
    text = FC.compose(low, sub, partition=P.digest(part), program=prog, public_inputs={"coef": COEF_PUBLIC})
    reg = G.Registered.commit(prog, value, want, salts=expected_salts(value, n))
    out_rows, out_salts, entry = reg.statement(range(n))
    read = lambda r, reads: {**r.entry(), "word_bits": 16, "reads": list(reads)}
    own = {"format": G.READS, "ports": {"out": read(reg, range(n)), "acc": read(chain.reg, at),
                                       "acc_out": read(chain.reg, [p + 1 for p in at])}}
    fresh = lambda: [hm96.SHA512.fresh_salt() for _ in range(n)]
    salts = {"row": fresh(), "aux": fresh(), "acc": acc_salts, "out": out_salts, "acc_out": out_acc_salts}
    registered = {"out": entry, "acc": acc_entry, "acc_out": out_acc_entry}
    linked = None
    if link_rows:
        linked = G.Registered.commit(prog, f"rec-rows-L{level['level']}", rows.tolist())
        _rows, salts["row"], registered["row"] = linked.statement(range(n))
        own["ports"]["row"] = read(linked, range(n))
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(program_)))
    (d / "partition.json").write_text(json.dumps(part))
    (d / "registered.json").write_text(json.dumps(own))
    FC.write_public_inputs(text, {"coef": coef}, n, d / "public-inputs.bin")
    tables = {"row": rows, "aux": aux, "acc": np.array(acc_rows, np.uint16)}
    outs = {"out": np.asarray(out_rows, np.uint16), "acc_out": np.asarray(out_acc, np.uint16)}
    FC.write(text, SimpleNamespace(name=value, subcircuit=sub), 0, n, d / "inst.bin", d / "pub.bin", None, list(range(n)),
             share_rows=True, outs=outs, tables=tables, refs={nm: list(range(n)) for nm in tables}, salts=salts,
             registered=registered, public_words={"coef": coef})
    meta = json.loads(text.split("\n", 2)[1][5:])
    return {"level": level["level"], "value": value, "template": defn.id, "LANES": LANES, "H": H, "instances": n,
            "opened": sum(opened), "not_opened": [(q.rep, q.j) for q, o in zip(qs, opened) if not o],
            "summed": sum(summed), "not_summed": [(q.rep, q.j) for q, s in zip(qs, summed) if not s],
            "unit_rows": low.layout.useful, "unit_log": meta["unit_log"], "k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"],
            "per_vu": meta["per_vu"], "compressions": LANES // 8 + 5 + 2 * H, "circuit_sha512": FC.digest(text),
            "program_sha512": prog, "partition_digest": P.digest(part), "registered_root": reg.root.hex(),
            "acc_root": chain.reg.root.hex(), "rows_root": linked.root.hex() if linked else None}
