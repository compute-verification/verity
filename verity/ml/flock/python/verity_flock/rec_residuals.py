"""V*'s algebra statement (hiding model architectures, lane rec-v0): the inner verifier's algebraic checks on each rep of the
inner session (`verity_flock.rec_algebra`) as ``InnerRepCheck_v1{S}``, one instance a rep, one C-Flock statement.

    U16 = Array<16, Value<1>>
    InnerRepCheck_v1{S}(m0: Array<P(n_0), U16>, .., acc0: Array<64, U16>, acc1: Array<64, U16>, v: Array<8 nv, U16>)
        -> Array<128, U16>

``S = {"gf": GfResiduals' S, "ports": [[name, n], ...]}`` (`spec`: the row ports in order, each with the F elements it holds).
An F element is 8 words ``lo || hi``, so bit k of a port's element i is bit ``k % 16`` of word ``8 i + k // 16``, and a row
port of n elements is ``P(n) = 64 ceil(n / 8)`` words (whole M0 rows), its words after the n elements unread. The body reads
the row ports' elements in order as ``GfResiduals_v1{S.gf}``'s w and ``v`` as its v, and returns the residuals (8 words each)
then zero words.

* ``m<i>``: the rep's i-th round that absorbs message elements (`rec_algebra.rounds`), its elements as the round's ``c_k``
  commits them (`rec_live.message_rows`). The verifier registers the round's two ``c_k`` (rep 0's, rep 1's) as the value
  ``rec-c<i>`` from the inner session's public record alone (`commits_entry`: a registered value's tree over the given
  ``b || c``), and instance r reads row r: the prover opens the row it committed in the inner session under its salt, so
  each message is the one the coins after it were drawn against.
* ``acc0``, ``acc1``: the rep's running sums before its first opening and after its last (two F elements, then the 48 words
  `rec_open` carries): rows of the prover's ``rec-acc`` registration, which the opening statements read too
  (`verity_flock.rec_outer.Chain`), so the final check sees the consistency sums the openings added.
* ``v``: the coefficients and constants at the rep's coins, then ``z_partial``'s comb, the inner circuit's fold at the rep's
  point (§10): a public input (PROTOCOL.md §16.10), which the verifier computes from the inner session's public record and the
  inner statement (`rec_algebra.verifier_rows`; the extra claims and the comb `verity_flock.rec_vstage` computes).
* ``out``: one output row, registered by the verifier as zero, so an instance is satisfiable only if every inner check of its
  rep holds.

The M0 unit (`unit_circuit`) computes S.gf's residuals with each residual's products summed leaf by leaf (`residual_forms`).
Karatsuba's recombination and the reduction are linear, so ``Σ_p reduce(K(a_p, b_p)) = reduce(recombine(Σ_p leaves_p))``: each
of the 2,187 leaf sums (the XOR of the residual's products' i-th ANDs) is committed (a copy row, `sha512.commit`) and
recombined once. A residual's output forms are then one product's, about 75K nonzeros, not its products'.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import time
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from verity.ml.commitments import hm96
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ZERO, Circuit, L
from verity.ml.commitments.gates.sha512 import commit
from verity.ml.circuits.ops.boolean import gf2k as G

from . import circuit as FC
from . import ir_lower as IL
from .class_statement import ClassDefinition

__all__ = ["InnerRepCheck", "MAX_REGIONS", "OUT_WORDS", "commits_entry", "definition", "parts", "port_words", "regions", "residual_forms",
           "residual_words", "spec",
           "statement", "unit_circuit", "write"]

U16 = Array(16, BL.BIT)
OUT_WORDS = 128
F_WORDS = 8
ROW_BLOCK = 64
LEAVES = 3 ** 7
ZERO_SALT = b"verity/rec-v0/zero-residual-salt/v0"
DERIVATION = "verity/rec-v0/inner-rep-check/coefficients/v1"
ZERO_VALUE = "rec-residuals-zero"
# a part's regions, at most: each region is two claims, each claim a round of the rep's coin stream, and a rep's other rounds
# (about 107 to 122 at m = 29 to 32) must fit beside them in --zk's committed schedule of 256 (`coin_spec_of`)
MAX_REGIONS = 64


def port_words(n: int) -> int:
    return -(-F_WORDS * n // ROW_BLOCK) * ROW_BLOCK


def v_words(S: dict) -> int:
    return F_WORDS * S["gf"]["nv"]


def spec(sh) -> dict:
    """``InnerRepCheck``'s S for a `rec_algebra.Shape`: the structure, a port per round that absorbs elements, and the running
    sums' two ports (w's layout, `rec_algebra.structure`)."""
    from . import rec_algebra as A

    gf, lay = A.structure(sh)
    ports = [[f"m{i}", len(r)] for i, r in enumerate(A.rounds(sh))] + [["acc0", 2], ["acc1", 2]]
    S = {"gf": gf, "ports": ports}
    _check(S)
    if lay.w[-4:] != A.ACC_NAMES:
        raise AssertionError("w does not end with the running sums")
    return S


def regions(ports: list, nv: int) -> int:
    """The statement's public regions (live `circuit.rs`, `Stmt::regions`): a digest region per row port and one for ``out``,
    and v's words, which follow the row ports' words, in aligned power-of-two blocks."""
    a = sum(port_words(n) for _, n in ports)
    b, blocks = a + F_WORDS * nv, 0
    while a < b:
        e = (b - a).bit_length() - 1
        while a % (1 << e):
            e -= 1
        a, blocks = a + (1 << e), blocks + 1
    return len(ports) + 1 + blocks


def parts(S: dict, max_regions: int = MAX_REGIONS) -> list[SimpleNamespace]:
    """S's residuals cut into InnerRepCheck S's of at most ``max_regions`` public regions each (`regions`), each residual in
    the first part it fits, in S's order. A part reads the row ports its residuals touch (in S's order, under their names)
    and the v elements they touch (in v's order), w and v renumbered; ``res`` and ``v`` are their indices in S. Every
    residual of S is one of a part's, so the algebra holds exactly when every part does."""
    _check(S)
    gf = S["gf"]
    starts = [0]
    for _, n in S["ports"]:
        starts.append(starts[-1] + n)

    def touched(r):
        w, v = set(), set()
        for t in list(r["add"]) + [t for pair in r["mul"] for k in pair for t in gf["ops"][k]]:
            if t[0] == "w":
                w.add(t[1])
            elif t[0] == "wt":
                w.update(range(t[1], t[1] + 128))
            elif t[0] == "v":
                v.add(t[1])
            else:
                raise ValueError(f"unknown term {t!r}")
        return {bisect.bisect_right(starts, i) - 1 for i in w}, v

    groups: list[tuple[list, set, set]] = []
    for i, (P, V) in enumerate(map(touched, gf["res"])):
        fit = lambda g: regions([S["ports"][p] for p in g[1] | P], len(g[2] | V)) <= max_regions
        g = next((g for g in groups if fit(g)), None)
        if g is None:
            g = ([], set(), set())
            if not fit(g):
                raise ValueError(f"residual {i} alone reads more than {max_regions} regions")
            groups.append(g)
        g[0].append(i)
        g[1].update(P)
        g[2].update(V)
    out = []
    for R, P, V in groups:
        wmap = {i: j for j, i in enumerate(i for p in sorted(P) for i in range(starts[p], starts[p + 1]))}
        vmap = {i: j for j, i in enumerate(sorted(V))}

        def term(t):
            if t[0] == "w":
                return ["w", wmap[t[1]]]
            if t[0] == "v":
                return ["v", vmap[t[1]]]
            if any(wmap.get(t[1] + j) != wmap[t[1]] + j for j in range(128)):
                raise ValueError(f"{t!r}: a transposed block over row ports the part does not read in a run")
            return ["wt", wmap[t[1]], t[2]]

        used = sorted({k for r in R for pair in gf["res"][r]["mul"] for k in pair})
        op = {k: j for j, k in enumerate(used)}
        g = {"nw": len(wmap), "nv": len(vmap), "ops": [[term(t) for t in gf["ops"][k]] for k in used],
             "res": [{"mul": [[op[a], op[b]] for a, b in gf["res"][r]["mul"]], "add": [term(t) for t in gf["res"][r]["add"]]}
                     for r in R]}
        part = {"gf": g, "ports": [list(S["ports"][p]) for p in sorted(P)]}
        _check(part)
        out.append(SimpleNamespace(S=part, res=R, v=sorted(V)))
    return out


def _check(S: dict) -> None:
    if not isinstance(S, dict) or set(S) != {"gf", "ports"}:
        raise TypeError("InnerRepCheck's S is {'gf': GfResiduals' S, 'ports': [[name, elements], ...]}")
    names = [nm for nm, _ in S["ports"]]
    if len(set(names)) != len(names) or "v" in names or any(not isinstance(n, int) or n < 1 for _, n in S["ports"]):
        raise ValueError("InnerRepCheck's row ports are distinct names, not v, each of at least one element")
    if sum(n for _, n in S["ports"]) != S["gf"]["nw"]:
        raise ValueError(f"row ports of {sum(n for _, n in S['ports'])} elements for a w of {S['gf']['nw']}")
    if F_WORDS * len(S["gf"]["res"]) > OUT_WORDS:
        raise ValueError(f"{len(S['gf']['res'])} residuals: more than {OUT_WORDS // F_WORDS}")


def words(elems, n_words: int) -> list[int]:
    """F elements as u16 words (``lo || hi``, 8 words each), zero words up to ``n_words``."""
    out = [(x >> (16 * j)) & 0xFFFF for x in elems for j in range(F_WORDS)]
    return out + [0] * (n_words - len(out))


def elements(ws, n: int) -> list[int]:
    return [sum(int(ws[F_WORDS * i + j]) << (16 * j) for j in range(F_WORDS)) for i in range(n)]


def _gather(S: dict, rows: list) -> list:
    """w's element bits from the row ports' bits (each port's first ``16 P(n)``), in port order."""
    out = []
    for (_, n), bits in zip(S["ports"], rows, strict=True):
        out += bits[:128 * n]
    return out


# --- the reference ----------------------------------------------------------------------------------------------------------

def residual_words(S: dict, rows: list[list[int]], v: list[int]) -> list[int]:
    """The reference: the output words on the row ports' words and v's elements (`gf2k.residuals`)."""
    w = [x for (_, n), r in zip(S["ports"], rows, strict=True) for x in elements(r, n)]
    return words(G.residuals(S["gf"], w, v), OUT_WORDS)


# --- the Definition ---------------------------------------------------------------------------------------------------------

def _signature(S):
    s = S.S
    _check(s)
    params = tuple((nm, Array(port_words(n), U16)) for nm, n in s["ports"]) + (("v", Array(v_words(s), U16)),)
    return params, Array(OUT_WORDS, U16)


def _body(B, S, *ports):
    s = S.S
    flat = lambda c: [c.refs[i] for i in range(len(c.refs))]
    w = _gather(s, [flat(p) for p in ports[:-1]])
    gf = s["gf"]
    res = B.call(bind(G.GfResiduals, S=gf), Coll(Array(gf["nw"], G.F), TR._refs(w)),
                 Coll(Array(gf["nv"], G.F), TR._refs(flat(ports[-1]))))
    bits = flat(res)
    zero = B.call(BL.constant(0)).refs[0]
    bits += [zero] * (16 * OUT_WORDS - len(bits))
    return Coll(Array(OUT_WORDS, U16), TR._refs(bits))


InnerRepCheck = CompositeDefinition(
    "InnerRepCheck", 1, ("S",), _signature, _body, conformance="exact-model-tested",
    doc="One rep's inner algebraic checks: GfResiduals{S.gf} on the row ports' first elements (w, in port order) and v, its "
        "residuals as u16 words then zero words")


def definition(S: dict):
    return bind(InnerRepCheck, S=S)


# --- the M0 unit ------------------------------------------------------------------------------------------------------------

class _Pairs:
    """A `forms.Circuit` stand-in for `gf2k.karatsuba`: records each AND's operands, returns leaf k as the variable ``-1 - k``."""

    def __init__(self):
        self.pairs = []

    def AND(self, a, b):
        self.pairs.append((a, b))
        return L(frozenset((-len(self.pairs),)), 0)


@cache
def _recombine() -> tuple[tuple[int, ...], ...]:
    """For each of the 128 reduced product bits, the Karatsuba leaves it sums."""
    rec = _Pairs()
    out = G.reduce_forms(G.karatsuba(rec, [L(frozenset((i,)), 0) for i in range(128)], [L(frozenset((128 + i,)), 0) for i in range(128)]))
    if len(rec.pairs) != LEAVES or any(o.c or any(x >= 0 for x in o.s) for o in out):
        raise AssertionError("Karatsuba's product is not linear in its leaves")
    return tuple(tuple(sorted(-1 - x for x in o.s)) for o in out)


def _leaves(a: list, b: list) -> list[tuple]:
    rec = _Pairs()
    G.karatsuba(rec, a, b)
    return rec.pairs


def _port_words(S: dict) -> tuple[int, ...]:
    return tuple(port_words(n) for _, n in S["ports"]) + (v_words(S),)


def unit_circuit(S: dict):
    """The unit over `forms`: inputs ``in0 ..`` (each row port's words in order, then v's, 16 bits each), outputs ``out0 ..
    out{OUT_WORDS - 1}``."""
    _check(S)
    C = Circuit()
    pw = _port_words(S)
    ins = [C.input(f"in{j}", 16) for j in range(sum(pw))]
    flat = [b for wd in ins for b in wd]
    cuts = [16 * sum(pw[:i]) for i in range(len(pw) + 1)]
    parts = [flat[a:b] for a, b in zip(cuts, cuts[1:])]
    bits = residual_forms(C, S["gf"], _gather(S, parts[:-1]), parts[-1])
    bits += [ZERO] * (16 * OUT_WORDS - len(bits))
    outs = [f"out{j}" for j in range(OUT_WORDS)]
    for j, nm in enumerate(outs):
        C.output(nm, bits[16 * j:16 * j + 16])
    return C, [f"in{j}" for j in range(len(ins))], outs


def residual_forms(C, S: dict, W: list, V: list) -> list:
    """``GfResiduals{S}``'s residuals over `forms` (128 bits each, in order) from w's and v's element bits (element ``i`` is
    bits ``128 i ..``), each residual's products summed leaf by leaf."""
    G.check_structure(S)

    def term(t):
        if t[0] == "w":
            return W[128 * t[1]:128 * t[1] + 128]
        if t[0] == "v":
            return V[128 * t[1]:128 * t[1] + 128]
        if t[0] == "wt":
            return [W[128 * (t[1] + j) + t[2]] for j in range(128)]
        raise ValueError(f"unknown term {t!r}")

    def xor(xs):
        acc = [ZERO] * 128
        for x in xs:
            acc = [p ^ q for p, q in zip(acc, x)]
        return acc

    ops: dict[int, list] = {}

    def op(k):
        if k not in ops:
            ops[k] = xor(term(t) for t in S["ops"][k])
        return ops[k]

    rec = _recombine()
    bits = []
    for r in S["res"]:
        acc = xor(term(t) for t in r["add"])
        if r["mul"]:
            sums, consts = [set() for _ in range(LEAVES)], [0] * LEAVES
            for a, b in r["mul"]:
                for i, (x, y) in enumerate(_leaves(op(a), op(b))):
                    z = C.AND(x, y)
                    sums[i].symmetric_difference_update(z.s)
                    consts[i] ^= z.c
            leaf = commit(C, [L(frozenset(s), c) for s, c in zip(sums, consts)])
            for k in range(128):
                s, c = set(acc[k].s), acc[k].c
                for i in rec[k]:
                    s.symmetric_difference_update(leaf[i].s)
                    c ^= leaf[i].c
                acc[k] = L(frozenset(s), c)
        bits += acc
    return bits


def lowering(S: dict, name: str) -> IL.Lowering:
    C, ins, outs = unit_circuit(S)
    w_in, w_out = len(ins), len(outs)
    units = IL.Units(target=None, in_widths=[16] * w_in, out_widths=[16] * w_out, in_src=[[("leaf", p) for p in range(w_in)]],
                     out_src=[[("ret", p) for p in range(w_out)]], signature=(name,), gates=[], arg_gates=[], ret_gates=[])
    return IL.Lowering(name, ClassDefinition(name, _port_words(S), w_out), units, C, IL.layout(C, ins, outs))


def subcircuit(S: dict, name: str) -> SimpleNamespace:
    """What `circuit.compose` and `circuit.write` read of a subcircuit: its ports and its Definition, ``InnerRepCheck_v1{S}``."""
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")
    defn = definition(S)
    digest = program_digest(Program(defn))
    return SimpleNamespace(id=name, template=SimpleNamespace(id="rec-residuals"),
                           inputs=[port(nm) for nm, _ in S["ports"]] + [port("v")], outputs=[port("out")],
                           definition=lambda **_: defn, program_digests=lambda statics: {defn.id: digest})


# --- the statement ----------------------------------------------------------------------------------------------------------

def zero_salts(value: str, n: int) -> list[bytes]:
    return [hashlib.shake_256(ZERO_SALT + value.encode() + b"\0" + i.to_bytes(4, "little")).digest(hm96.SHA512.salt_bytes)
            for i in range(n)]


def commits_entry(program: str, value: str, words_: int, bcs: list[bytes]) -> dict:
    """The verifier's own entry of a registered value it knows only by its rows' ``b || c`` (`registered.Registered.entry`'s
    fields): its domain and the root of its tree over them."""
    from verity.ml.commitments.merkle import tree_levels
    from verity.experimental.protocols.sampled_proofs.one_stage import registered as RG

    dom = RG.domain(program, value, words_, len(bcs))
    root = tree_levels(dom, [RG.leaf(dom, i, bc) for i, bc in enumerate(bcs)])[-1][0]
    return {"value": value, "root": root.hex(), "leaves": len(bcs), "words": int(words_), "domain": dom.domain_id.hex()}


def statement(S: dict, name: str, n: int) -> SimpleNamespace:
    """``InnerRepCheck_v1{S}`` over ``n`` instances: its Program, partition, unit and circuit text, which only ``n`` and ``S``
    decide."""
    from verity.experimental.protocols.sampled_proofs.one_stage import partition as P

    t0 = time.perf_counter()
    defn = definition(S)
    program = P.population_program(defn, n)
    part = P.partition(program, P.template_instance_query(defn))
    prog = P.program_sha512(program)
    low = lowering(S, name)
    sub = subcircuit(S, name)
    public = {"v": {"seed": "the inner session's public record (rec/inner-transcript/v0) and the inner statement, rep = instance",
                    "derivation": DERIVATION}}
    text = FC.compose(low, sub, partition=P.digest(part), program=prog, public_inputs=public)
    return SimpleNamespace(S=S, n=n, defn=defn, program=program, part=part, prog=prog, low=low, sub=sub, text=text,
                           build_s=round(time.perf_counter() - t0, 1))


def write(st: SimpleNamespace, reads: dict, vs: list[list[int]], out_dir, evaluate: bool = True, own: dict | None = None) -> dict:
    """The statement (`statement`) on its instances, written to ``out_dir``: the circuit, the prover's ``inst.bin``, the
    verifier's ``pub.bin``, ``public-inputs.bin`` (v), ``program.json``, ``partition.json`` and the reads file
    ``registered.json``.

    ``reads``: per row port, ``(registration, positions)``, the prover's registered value (`registered.Registered`) and each
    instance's row of it; ``vs``: each instance's v (F elements). ``own``: per row port the verifier's own entry (default the
    registration's), which its reads file names; the outputs are the verifier's zero rows, registered under its own salts."""
    from verity.experimental.protocols.sampled_proofs.one_stage import partition as P
    from verity.experimental.protocols.sampled_proofs.one_stage import registered as RG

    S, n, low, sub, text = st.S, st.n, st.low, st.sub, st.text
    names = [nm for nm, _ in S["ports"]]
    if set(reads) != set(names) or len(vs) != n or any(len(pos) != n for _, pos in reads.values()):
        raise ValueError(f"reads of {sorted(reads)} and {len(vs)} v rows for the row ports {names} of {n} instances")
    tables, salts, registered, entries = {}, {}, {}, {}
    for nm, (reg, pos) in reads.items():
        rows, ss, e = reg.statement(pos)
        tables[nm], salts[nm], registered[nm] = np.array(rows, np.uint16), ss, e
        entries[nm] = {**((own or {}).get(nm) or reg.entry()), "word_bits": 16, "reads": [int(p) for p in pos]}
    vrows = np.array([words(v, v_words(S)) for v in vs], np.uint16)
    ref = [residual_words(S, [tables[nm][i].tolist() for nm in names], vs[i]) for i in range(n)]
    agrees = None
    if evaluate:
        got, _cut, ok = low.evaluate(np.concatenate([tables[nm] for nm in names] + [vrows], axis=1).astype(np.uint64))
        agrees = [bool((int(ok) >> i) & 1) and got[i].astype(np.uint16).tolist() == ref[i] for i in range(n)]
    zero = RG.Registered.commit(st.prog, ZERO_VALUE, [[0] * OUT_WORDS for _ in range(n)], salts=zero_salts(ZERO_VALUE, n))
    out_rows, salts["out"], registered["out"] = zero.statement(range(n))
    entries["out"] = {**zero.entry(), "word_bits": 16, "reads": list(range(n))}
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(st.program)))
    (d / "partition.json").write_text(json.dumps(st.part))
    (d / "registered.json").write_text(json.dumps({"format": RG.READS, "ports": entries}))
    FC.write_public_inputs(text, {"v": vrows}, n, d / "public-inputs.bin")
    FC.write(text, SimpleNamespace(name=ZERO_VALUE, subcircuit=sub), 0, n, d / "inst.bin", d / "pub.bin", None, list(range(n)),
             share_rows=True, outs={"out": np.asarray(out_rows, np.uint16)}, tables=tables, refs={nm: list(range(n)) for nm in names},
             salts=salts, registered=registered, public_words={"v": vrows})
    gf = S["gf"]
    meta = json.loads(text.split("\n", 2)[1][5:])
    return {"template": st.defn.id, "instances": n, "ports": len(names), "nw": gf["nw"], "nv": gf["nv"],
            "products": sum(len(r["mul"]) for r in gf["res"]), "residuals": len(gf["res"]), "holds": [not any(x) for x in ref],
            "nonzero_residuals": [[i for i in range(len(gf["res"])) if any(x[8 * i:8 * i + 8])] for x in ref],
            "unit_agrees": agrees, "unit_rows": low.layout.useful, "unit_log": meta["unit_log"], "k_log": meta["k_log"],
            "vus_per_block": meta["vus_per_block"], "per_vu": meta["per_vu"], "circuit_sha512": FC.digest(text),
            "circuit_bytes": len(text), "program_sha512": st.prog, "partition_digest": P.digest(st.part),
            "zero_root": zero.root.hex(), "build_s": st.build_s}
