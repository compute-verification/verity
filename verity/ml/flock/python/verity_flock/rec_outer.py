"""V*'s outer statements (hiding model architectures, lane rec-v0): the inner session's Ligerito openings, one C-Flock
statement per level, each instance a `RecOpen` of one opening, the top node it climbs to and its running sum read from the
prover's registrations of them.

A level's openings share one shape (its rows' lanes, ``H = depth - c`` siblings below its top of ``2^c`` nodes), so each level
is one template, ``RecOpen{LANES, H}``, and one statement over that level's openings of both reps, in rep and query order
(M0 proves one template per statement). Its program is `population_program` of the template over the level's openings, its
partition that program's ``Q_template_instance`` v0. Every instance reads two fresh private rows, the opened row and its
``aux`` (its siblings: `rec_open.pack`), and five registered rows: its coefficients ``coef`` and the position's low bits
``dirs``, which the verifier registers itself, ``out``, ``acc`` in and ``acc_out`` out (§16.13). The statement has no public
input.

**The verifier's values** (`COEF_VALUE`, `DIRS_VALUE`). The verifier computes each opening's ``coef`` (`coefficients`) and
``dirs`` (`rec_open.dirs_words` of the position its own query coins give) and registers each as a value of the level's
statement program, row i instance i's, under public salts (`rec_residuals.verifier_value`): their roots are functions of the
inner session's public record, and the prover's rows and salts are the same registration's.

**The tops** (`TopValue`). Neither side sees a tree's top in the clear: the firewall committed each node as a registered row
(`verity_flock.rec_live.top_rows`) under a fresh salt of its own, and the coin server recorded only the nodes' ``b || c``
(`verity_flock.rec_vstar.tops`). The prover registers the tops a level's openings climb to (level 0's ``root_B``, both reps'
one tree, else rep 0's top then rep 1's) as the value ``rec-top-L<level>`` under the V* session's program and those very salts;
the verifier builds its own entry of that value from the recorded ``b || c`` alone (`rec_residuals.commits_entry`), and its
reads file puts instance ``i``'s output at the row of the top node its position names (rep ``r``'s first row plus
``pos >> H``; PROTOCOL.md §16.13 "Registered reads", `verity.core.protocols.sampled_proofs.one_stage.registered`). An
instance is satisfiable only if its private rows climb, by the verifier's ``dirs``, to that registered row, whose ``b || c``
the firewall committed: the statement holds only if every opening climbs to the committed top node its coin names.

**The running sums** (`Chain`). The openings of both reps, rep-major, then by level and query (`chain_order`), are positions
``0 .. N - 1`` of one chain; the prover registers its ``N + 1`` rows as the value ``rec-acc`` (`ACC_VALUE`) under the V*
session's program string (`session_program`, which every statement reading the value shares), row 0 zero and row ``p + 1``
opening ``p``'s ``acc_out`` on row ``p`` (`rec_open.acc_ref`). Opening ``p``'s instance reads ``acc`` at row ``p`` and
``acc_out`` at row ``p + 1``, so every level's statement holds only if each row is the one before it plus its opening's
consistency terms, and V*'s algebra (`verity_flock.rec_residuals`) reads a rep's first and last rows. The coefficients are
the verifier's (`coefficients`: the final check's κ at the rep's coins times the level's eq(α)[j],
`verity_flock.rec_algebra.verifier_rows`).

**The claims** (`note:proofs/20261010T1820Z-draft-c3-claim-layout`). Beside ``rec-acc``, under the same program string and its
own salts, the prover registers each rep's `rec_sparse.RecSparse` claim as the value ``rec-claim`` (`CLAIM_VALUE`), row r rep
r's (one element, 64 words): RecSparse's instance r reads row r, and the registered opening (`rec_residuals.stage_opening`)
reads row 0 as ``claim0`` and row 1 as ``claim1``, so the two statements open one root's cells.

With ``link_rows``, the opened rows are not fresh: the prover registers them first, as the value ``rec-rows-L<level>`` under
its own salts (what it would send before the outer coins), and the statement's ``row`` port reads row ``i`` of that value in
instance ``i``. The verifier then checks every opened row against that root, which is how another statement can read the
same rows.

`stage` writes one level's ``circuit.txt``, the prover's ``inst.bin``, the verifier's ``pub.bin``, ``program.json``,
``partition.json`` and the reads file ``registered.json``; `verity_flock.rec_vstage` stages every level
and the algebra of an inner session. The openings are of plain SHA-512 leaves (`rec_open`); a level of salted ones is refused.
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
from . import rec_live as RL
from . import rec_open as R
from . import rec_residuals as RR
from . import rec_vstar as V
from .class_statement import ClassDefinition

SESSION_TAG = b"verity/rec-v0/v-star/v0\0"
ACC_VALUE = "rec-acc"
CLAIM_VALUE = "rec-claim"
COEF_VALUE = "rec-coef"
DIRS_VALUE = "rec-dirs"


def session_program(record: bytes) -> str:
    """The program string of the registrations V*'s statements share (the running sums, the inner messages): SHA-512 of the
    tag and the inner session's public record."""
    return hashlib.sha512(SESSION_TAG + hashlib.sha512(record).digest()).hexdigest()


def unit_name(LANES: int, H: int) -> str:
    """A level of the circuit table's openings: ``rec-open/l<LANES>-h<H>``."""
    return f"rec-open/l{LANES}-h{H}"


def stacked_name(LANES: int, H: int) -> str:
    """A level of the stacked table R's openings (`rec_vstage.stage_stacked`): ``rec-open-stacked/l<LANES>-h<H>``, never a
    name of the circuit table's levels, which can share R's shape over another instance count (``Flock/RecVStar.lean``'s
    ``recOpenStackedName``)."""
    return f"rec-open-stacked/l{LANES}-h{H}"


def lowering(LANES: int, H: int, name: str | None = None) -> IL.Lowering:
    """`rec_open.unit_circuit` as M0's unit named ``name`` (default `unit_name`): ports ``row``, ``aux``, ``acc``, ``coef``
    and ``dirs`` (u16 words), and ``out`` and ``acc_out`` (one row each)."""
    name = name or unit_name(LANES, H)
    C, ins, outs = R.unit_circuit(LANES, H)
    w_in, w_out = len(ins), len(outs)
    units = IL.Units(target=None, in_widths=[16] * w_in, out_widths=[16] * w_out, in_src=[[("leaf", p) for p in range(w_in)]],
                     out_src=[[("ret", p) for p in range(w_out)]], signature=(name,), gates=[], arg_gates=[], ret_gates=[])
    defn = ClassDefinition(name, R.port_words(LANES, H), w_out, (R.OUT_WORDS, R.ACC_WORDS))
    return IL.Lowering(name, defn, units, C, IL.layout(C, ins, outs))


def subcircuit(LANES: int, H: int, name: str | None = None) -> SimpleNamespace:
    """What `circuit.compose` and `circuit.write` read of a subcircuit: its ports and its Definition, ``RecOpen{LANES, H}``,
    under the unit's name ``name`` (default `unit_name`)."""
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")
    defn = R.definition(LANES, H)
    digest = program_digest(Program(defn))
    return SimpleNamespace(id=name or unit_name(LANES, H), template=SimpleNamespace(id="rec-open"),
                           inputs=[port(nm) for nm in R.PORTS], outputs=[port("out"), port("acc_out")],
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


def top_value_name(level: int) -> str:
    return f"rec-top-L{level}"


@dataclass
class TopValue:
    """A level's tops as one registered value (`top_value_name`): the prover's registration (``reg``, the top rows under the
    firewall's salts), the verifier's own entry from the coin server's ``b || c`` alone (``own``), and each rep's first row."""

    reg: object
    own: dict
    first: dict[int, int]

    @classmethod
    def build(cls, program: str, level: int, tops: dict[int, V.Top], value: str | None = None) -> "TopValue":
        """``tops``: each rep's top at the level (`rec_vstar.tops`); a top two reps share (level 0's ``root_B``) is one run of
        rows. Refused when the prover's registration is not the value the recorded ``b || c`` commit. ``value``: another
        table's name for it (default `top_value_name`)."""
        from verity.core.protocols.sampled_proofs.one_stage import registered as G

        runs: list[V.Top] = []
        first = {}
        for rep, t in sorted(tops.items()):
            if t not in runs:
                runs.append(t)
            first[rep] = sum(len(u.nodes) for u in runs[:runs.index(t)])
        value = value or top_value_name(level)
        rows = [r for t in runs for r in RL.top_rows(b"".join(t.nodes))]
        reg = G.Registered.commit(program, value, rows, [s for t in runs for s in t.salts])
        own = RR.commits_entry(program, value, R.OUT_WORDS, [b for t in runs for b in t.bcs])
        if own["root"] != reg.entry()["root"]:
            raise ValueError(f"level {level}: the prover's tops are not the ones the firewall committed")
        return cls(reg, own, first)

    def position(self, q: V.Query) -> int:
        return self.first[q.rep] + (q.pos >> (q.depth - q.c))


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
        from verity.core.protocols.sampled_proofs.one_stage import registered as G

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


def stage(level: dict, out_dir: Path, chain: Chain, coefs: dict[tuple[int, int, int], list[int]], tops: TopValue,
          link_rows: bool = False, name: str | None = None) -> dict:
    """One level's statement, the verifier's own files beside the prover's (``out_dir``), and what it costs. ``chain``: the
    prover's running sums; ``coefs``: each opening's ``coef`` words (the verifier's); ``tops``: the level's committed tops;
    ``name``: its unit's name (default `unit_name`; R's levels `stacked_name`). The openings' positions are the ones the
    verifier's query coins give (`rec_vstage.coefficients` checks them), so ``dirs`` is the verifier's too."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.core.protocols.sampled_proofs.one_stage import registered as G

    LANES, H, qs = level["LANES"], level["H"], level["queries"]
    n = len(qs)
    if any(q.salt is not None for q in qs):
        raise ValueError(f"level {level['level']}: openings of hm96-sha512/v1 leaves; RecOpen opens plain SHA-512 leaves "
                         f"(flock-leaf/sha512-unsalted)")
    defn = R.definition(LANES, H)
    program_ = P.population_program(defn, n)
    query = P.template_instance_query(defn)
    part = P.partition(program_, query)
    prog = P.program_sha512(program_)
    keys = [(q.rep, q.level, q.j) for q in qs]
    packed = [R.pack(q.row, q.sibs) for q in qs]
    rows = np.array([r for r, _ in packed], np.uint16)
    aux = np.array([a for _, a in packed], np.uint16)
    coef_reg = RR.verifier_value(prog, COEF_VALUE, [coefs[k] for k in keys])
    dirs_reg = RR.verifier_value(prog, DIRS_VALUE, [R.dirs_words(q.pos, H) for q in qs])
    coef_rows, coef_salts, coef_entry = coef_reg.statement(range(n))
    dirs_rows, dirs_salts, dirs_entry = dirs_reg.statement(range(n))
    coef, dirs = np.array(coef_rows, np.uint16), np.array(dirs_rows, np.uint16)
    at = [chain.at[k] for k in keys]
    acc_rows, acc_salts, acc_entry = chain.reg.statement(at)
    out_acc, out_acc_salts, out_acc_entry = chain.reg.statement([p + 1 for p in at])
    tops_at = [tops.position(q) for q in qs]
    out_rows, out_salts, entry = tops.reg.statement(tops_at)
    if [list(r) for r in out_rows] != [R.expected(q.cap_node) for q in qs]:
        raise ValueError(f"level {level['level']}: an opening's top node is not its row of the level's tops")
    low = lowering(LANES, H, name)
    flat = np.concatenate([rows, aux, np.array(acc_rows, np.uint16), coef, dirs], axis=1).astype(np.uint64)
    got, _cut, ok = low.evaluate(flat)
    opened = [bool((int(ok) >> i) & 1) and got[i, :R.OUT_WORDS].astype(np.uint16).tolist() == list(out_rows[i]) for i in range(n)]
    summed = [bool((int(ok) >> i) & 1) and got[i, R.OUT_WORDS:].astype(np.uint16).tolist() == list(out_acc[i]) for i in range(n)]
    sub = subcircuit(LANES, H, name)
    text = FC.compose(low, sub, partition=P.digest(part), program=prog)
    read = lambda e, reads: {**e, "word_bits": 16, "reads": list(reads)}
    own = {"format": G.READS, "ports": {"out": read(tops.own, tops_at), "acc": read(chain.reg.entry(), at),
                                       "coef": read(coef_reg.entry(), range(n)), "dirs": read(dirs_reg.entry(), range(n)),
                                       "acc_out": read(chain.reg.entry(), [p + 1 for p in at])}}
    fresh = lambda: [hm96.SHA512.fresh_salt() for _ in range(n)]
    salts = {"row": fresh(), "aux": fresh(), "acc": acc_salts, "coef": coef_salts, "dirs": dirs_salts, "out": out_salts,
             "acc_out": out_acc_salts}
    registered = {"out": entry, "acc": acc_entry, "coef": coef_entry, "dirs": dirs_entry, "acc_out": out_acc_entry}
    linked = None
    if link_rows:
        linked = G.Registered.commit(prog, f"rec-rows-L{level['level']}", rows.tolist())
        _rows, salts["row"], registered["row"] = linked.statement(range(n))
        own["ports"]["row"] = read(linked.entry(), range(n))
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(program_)))
    (d / "partition.json").write_text(json.dumps(part))
    (d / "registered.json").write_text(json.dumps(own))
    tables = {"row": rows, "aux": aux, "acc": np.array(acc_rows, np.uint16), "coef": coef, "dirs": dirs}
    outs = {"out": np.asarray(out_rows, np.uint16), "acc_out": np.asarray(out_acc, np.uint16)}
    FC.write(text, SimpleNamespace(name=f"rec-L{level['level']}", subcircuit=sub), 0, n, d / "inst.bin", d / "pub.bin", None,
             list(range(n)), share_rows=True, outs=outs, tables=tables, refs={nm: list(range(n)) for nm in tables}, salts=salts,
             registered=registered)
    meta = json.loads(text.split("\n", 2)[1][5:])
    return {"level": level["level"], "value": tops.own["value"], "template": defn.id, "LANES": LANES, "H": H, "instances": n,
            "opened": sum(opened), "not_opened": [(q.rep, q.j) for q, o in zip(qs, opened) if not o],
            "summed": sum(summed), "not_summed": [(q.rep, q.j) for q, s in zip(qs, summed) if not s],
            "unit_rows": low.layout.useful, "unit_log": meta["unit_log"], "k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"],
            "per_vu": meta["per_vu"], "compressions": R.compressions(LANES, H), "circuit_sha512": FC.digest(text),
            "program_sha512": prog, "partition_digest": P.digest(part), "registered_root": tops.own["root"],
            "top_rows": tops.own["leaves"], "acc_root": chain.reg.root.hex(), "rows_root": linked.root.hex() if linked else None,
            "coef_root": coef_reg.root.hex(), "dirs_root": dirs_reg.root.hex()}
