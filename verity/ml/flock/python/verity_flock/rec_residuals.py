"""V*'s algebra statement (hiding model architectures, lane rec-v0): the inner verifier's algebraic checks on each rep of the
inner session (`verity_flock.rec_algebra`) as ``InnerRepCheck_v2{S}``, one instance a rep, one C-Flock statement.

    U16 = Array<16, Value<1>>
    InnerRepCheck_v2{S}(m0: Array<P(n_0), U16>, .., acc0: Array<64, U16>, acc1: Array<64, U16>, v: Array<P(nv), U16>)
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
  each message is the one the coins after it were drawn against. A holo shape's h round (`rec_algebra.h_round`) is one of
  them, absorbed after z_partial and before r′ and ζ_c are drawn: the one registered value both this statement (L4′, h an XOR
  term of the comb's operand, no product of its own) and `rec_sparse.RecSparse`'s ``h`` (H1) read at row r.
* ``acc0``, ``acc1``: the rep's running sums before its first opening and after its last (two F elements, then the 48 words
  `rec_open` carries): rows of the prover's ``rec-acc`` registration, which the opening statements read too
  (`verity_flock.rec_outer.Chain`), so the final check sees the consistency sums the openings added.
* ``claim<k>`` (an opening-only shape, the registered opening, `stage_opening`): row k of the prover's ``rec-claim``
  (`verity_flock.rec_outer.CLAIM_VALUE`), RecSparse's rep k's claim, one element read as a w element (an add term of claim
  k's ring switch). The opening is one instance, ``rec-opening/m<m>-c<claims>-p<j>``, its rounds' messages one-row
  registrations.
* ``v``: the coefficients and constants at the rep's coins, then ``z_partial``'s comb, the inner circuit's fold at the rep's
  point (§10), which the verifier computes from the inner session's public record and the inner statement
  (`rec_algebra.verifier_rows`; the extra claims and the comb `verity_flock.rec_vstage` computes), as a row port of
  ``P(nv)`` words, its words after the nv elements zero and unread. The verifier registers it itself (`verifier_value`): the
  value ``rec-v`` (`V_VALUE`), row r rep r's, under public salts (`verifier_salts`), so its root is a function of the public
  record and the statement has no public input.
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
from . import rec_budget as RB
from .class_statement import ClassDefinition

__all__ = ["HOLO_KEY", "InnerRepCheck", "MAX_LONE_SLOTS", "MAX_REGIONS", "MAX_SLOTS", "OUT_WORDS", "commits_entry", "definition", "parts", "port_words", "regions",
           "opening_rows", "residual_forms", "residual_words", "slots", "spec", "stage_opening",
           "statement", "unit_circuit", "write", "write_rows"]

U16 = Array(16, BL.BIT)
OUT_WORDS = 128
F_WORDS = 8
ROW_BLOCK = 64
LEAVES = 3 ** 7
ZERO_SALT = b"verity/rec-v0/zero-residual-salt/v0"
VERIFIER_SALT = b"verity/rec-v0/verifier-salt/v0"
ZERO_VALUE = "rec-residuals-zero"
V_VALUE = "rec-v"
# the META key that says the statement is over a holo shape's parts (`rec_algebra.Shape.holo`), written only when true: the
# unit's name does not say, and the verifier (`Flock.RecResiduals.HOLO_KEY`) builds the one net it selects, absent meaning plain
HOLO_KEY = "holo"
# a part's regions, at most: each region is two claims, each claim a round of the rep's coin stream, and a rep's other rounds
# (about 107 to 122 at m = 29 to 32) must fit beside them in --zk's committed schedule of 512 (`coin_spec_of`)
MAX_REGIONS = 64
# a part's sha512x3 slots per instance, at most (`slots`): 160 slots of 2^18 bits (2^25.32) beside its hm96 slots' 2^24 (at
# most MAX_REGIONS) leave room for a 2^22 unit in a 2^26 block (for a 2^23 unit, as at 12 claims, 159 do), below
# circuit.K_MAX's 2^27 (`circuit.compose` refuses a larger one); K = 4096's units are 2^22, and without the bound v's
# compressions (about nv / 8) put its first part at 2^27. A shape's widest residual reads about 10.75 slots a claim, so 160
# holds up to 14 claims (12 claims: 130)
MAX_SLOTS = 160
# a residual that fits no part alone under MAX_SLOTS stands alone up to this many: no other residual joins it (slots only
# grow), so the parts of a shape whose residuals each fit MAX_SLOTS are unchanged, and the bound reaches only residuals
# MAX_SLOTS refuses. The widest, the batched T, reads 172 slots at 16 claims (llama32-1b/b1/i256's 2^29 bucket), 194 at 18
# and 215 at 20, beside hm96's 2^23 (32 slots of 2^18), a 2^23 unit (32) and the mask (one): 191 fill a 2^26 block, so past
# it a lone part's block is 2^27 = circuit.K_MAX, which holds 447 (172 at
# k_log 26, 194 to 215 at 27). 21 claims' 226 refuse
MAX_LONE_SLOTS = 223


def port_words(n: int) -> int:
    return -(-F_WORDS * n // ROW_BLOCK) * ROW_BLOCK


def v_words(S: dict) -> int:
    return port_words(S["gf"]["nv"])


def spec(sh) -> dict:
    """``InnerRepCheck``'s S for a `rec_algebra.Shape`: the structure, a port per round that absorbs elements, and the running
    sums' two ports (w's layout, `rec_algebra.structure`); an opening-only shape's then a port ``claim<k>`` per registered
    claim (`rec_algebra.claim_names`)."""
    from . import rec_algebra as A

    gf, lay = A.structure(sh)
    claims = A.claim_names(sh)
    ports = [[f"m{i}", len(r)] for i, r in enumerate(A.rounds(sh))] + [["acc0", 2], ["acc1", 2]]
    ports += [[f"claim{k}", 1] for _, k in claims]
    S = {"gf": gf, "ports": ports}
    _check(S, whole=True)
    if lay.w[len(lay.w) - len(claims) - 4:] != A.ACC_NAMES + claims:
        raise AssertionError("w does not end with the running sums and the registered claims")
    return S


def regions(ports: list) -> int:
    """The statement's public regions (live `circuit.rs`, `Stmt::regions`): a digest region per row port, ``v``'s included,
    and one for ``out``. The statement has no public input, so no region of public words."""
    return len(ports) + 2


def slots(ports: list, nv: int) -> int:
    """The statement's sha512x3 slots per instance (`circuit._row_slots`): the compressions of its row ports, ``v``'s and
    ``out``'s before their last blocks, three a slot."""
    rows = [FC.Port(nm, port_words(n)) for nm, n in ports] + [FC.Port("v", port_words(nv)), FC.Port("out", OUT_WORDS)]
    return -(-sum(p.comps for p in rows) // FC.PER_SLOT)


def parts(S: dict, max_regions: int = MAX_REGIONS, max_slots: int = MAX_SLOTS,
          max_lone_slots: int = MAX_LONE_SLOTS) -> list[SimpleNamespace]:
    """S's residuals cut into InnerRepCheck S's of at most ``max_regions`` public regions (`regions`), ``max_slots`` sha512x3
    slots (`slots`) and ``OUT_WORDS / 8`` residuals each, each residual in the first part it fits, in S's order; a residual
    that fits no part, not even alone, is a part of its own if it reads at most ``max_lone_slots`` slots. A part
    reads the row ports its residuals touch (in S's order, under their names) and the v elements they touch (in v's order),
    w and v renumbered; ``res`` and ``v`` are their indices in S. Every residual of S is one of a part's, so the algebra
    holds exactly when every part does."""
    _check(S, whole=True)
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
        def fit(g, n=max_slots):
            ports = [S["ports"][p] for p in g[1] | P]
            return (regions(ports) <= max_regions and slots(ports, len(g[2] | V)) <= n
                    and F_WORDS * (len(g[0]) + 1) <= OUT_WORDS)

        g = next((g for g in groups if fit(g)), None)
        if g is None:
            g = ([], set(), set())
            if not fit(g, max(max_slots, max_lone_slots)):
                raise ValueError(f"residual {i} alone reads more than {max_regions} regions or "
                                 f"{max(max_slots, max_lone_slots)} sha512x3 slots")
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


def _check(S: dict, whole: bool = False) -> None:
    """S is an InnerRepCheck S; one statement's (``whole`` False) also holds at most ``OUT_WORDS / 8`` residuals, which a
    whole algebra's S (``whole``, `parts`' input) need not."""
    if not isinstance(S, dict) or set(S) != {"gf", "ports"}:
        raise TypeError("InnerRepCheck's S is {'gf': GfResiduals' S, 'ports': [[name, elements], ...]}")
    names = [nm for nm, _ in S["ports"]]
    if len(set(names)) != len(names) or "v" in names or any(not isinstance(n, int) or n < 1 for _, n in S["ports"]):
        raise ValueError("InnerRepCheck's row ports are distinct names, not v, each of at least one element")
    if sum(n for _, n in S["ports"]) != S["gf"]["nw"]:
        raise ValueError(f"row ports of {sum(n for _, n in S['ports'])} elements for a w of {S['gf']['nw']}")
    if not whole and F_WORDS * len(S["gf"]["res"]) > OUT_WORDS:
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
                 Coll(Array(gf["nv"], G.F), TR._refs(flat(ports[-1])[:128 * gf["nv"]])))
    bits = flat(res)
    zero = B.call(BL.constant(0)).refs[0]
    bits += [zero] * (16 * OUT_WORDS - len(bits))
    return Coll(Array(OUT_WORDS, U16), TR._refs(bits))


InnerRepCheck = CompositeDefinition(
    "InnerRepCheck", 2, ("S",), _signature, _body, conformance="exact-model-tested",
    doc="One rep's inner algebraic checks: GfResiduals{S.gf} on the row ports' first elements (w, in port order) and v's first "
        "S.gf.nv, its residuals as u16 words then zero words")


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


def _leaf_operands(a: list) -> list:
    """One side's operands of `gf2k.karatsuba`'s ANDs, in its order: its leaf pairs are this on a zipped with this on b."""
    n = len(a)
    if n == 1:
        return [a[0]]
    h = n // 2
    return _leaf_operands(a[:h]) + _leaf_operands(a[h:]) + _leaf_operands([x ^ y for x, y in zip(a[:h], a[h:])])


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
    bits = residual_forms(C, S["gf"], _gather(S, parts[:-1]), parts[-1][:128 * S["gf"]["nv"]])
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
            ops[k] = _leaf_operands(xor(term(t) for t in S["ops"][k]))
        return ops[k]

    rec = _recombine()
    bits = []
    for r in S["res"]:
        acc = xor(term(t) for t in r["add"])
        if r["mul"]:
            sums, consts = [set() for _ in range(LEAVES)], [0] * LEAVES
            for a, b in r["mul"]:
                for i, (x, y) in enumerate(zip(op(a), op(b))):
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
    """What `circuit.compose` and `circuit.write` read of a subcircuit: its ports and its Definition, ``InnerRepCheck_v2{S}``."""
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


def verifier_salts(program: str, value: str, n: int) -> list[bytes]:
    """The salts of a value the verifier registers itself (`verifier_value`), row i's from the program, the value's name and
    i alone. They are public, as the rows are: the commitment's binding (cr/sha-512) is what the reads need, and it hides
    nothing the verifier does not publish."""
    head = VERIFIER_SALT + b"\0" + program.encode() + b"\0" + value.encode() + b"\0"
    return [hashlib.shake_256(head + i.to_bytes(4, "little")).digest(hm96.SHA512.salt_bytes) for i in range(n)]


def verifier_value(program: str, value: str, rows: list[list[int]]):
    """A value the verifier registers itself (`registered.Registered`): rows it computes from the inner session's public record,
    the inner statement and its own coins, under `verifier_salts`. Its entry is then a function of those rows alone, which the
    verifier's reads file names, and the prover's table rows and salts are the same registration's."""
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG

    return RG.Registered.commit(program, value, rows, verifier_salts(program, value, len(rows)))


def commits_entry(program: str, value: str, words_: int, bcs: list[bytes]) -> dict:
    """The verifier's own entry of a registered value it knows only by its rows' ``b || c`` (`registered.Registered.entry`'s
    fields): its domain and the root of its tree over them."""
    from verity.ml.commitments.merkle import tree_levels
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG

    dom = RG.domain(program, value, words_, len(bcs))
    root = tree_levels(dom, [RG.leaf(dom, i, bc) for i, bc in enumerate(bcs)])[-1][0]
    return {"value": value, "root": root.hex(), "leaves": len(bcs), "words": int(words_), "domain": dom.domain_id.hex()}


def statement(S: dict, name: str, n: int, holo: bool = False) -> SimpleNamespace:
    """``InnerRepCheck_v2{S}`` over ``n`` instances: its Program, partition, unit and circuit text, which only ``n``, ``S`` and
    ``holo`` decide. ``holo``: S is a part of a holo shape's spec, which META says (`HOLO_KEY`). With `rec_circuits`' cache
    the text is built once per (S, name, n, holo), and ``low`` is None when it was read back."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    from . import rec_circuits as RC

    t0 = time.perf_counter()
    defn = definition(S)
    program = P.population_program(defn, n)
    part = P.partition(program, P.template_instance_query(defn))
    prog = P.program_sha512(program)
    sub = subcircuit(S, name)
    key = {"name": name, "n": n, "holo": holo, "definition": sub.program_digests(None)[defn.id]}
    if kept := RC.load("rec-residuals", key):
        low, (text, info) = None, kept
    else:
        low = lowering(S, name)
        text = FC.compose(low, sub, partition=P.digest(part), program=prog, extra_meta={HOLO_KEY: True} if holo else None)
        info = {"unit_rows": low.layout.useful, "digests": [FC.digest(text), FC.class_digest(text)]}
        RC.store("rec-residuals", key, text, info)
    names = [nm for nm, _ in S["ports"]]
    ref = lambda rows, vw: residual_words(S, [rows[nm] for nm in names], elements(vw, S["gf"]["nv"]))
    return SimpleNamespace(S=S, n=n, defn=defn, program=program, part=part, prog=prog, low=low, sub=sub, text=text,
                           names=names, v_words=v_words(S), out_words=OUT_WORDS, v_value=V_VALUE, zero_value=ZERO_VALUE,
                           ref=ref, unit_rows=info["unit_rows"], digests=tuple(info["digests"]), circuit_kept=low is None,
                           build_s=round(time.perf_counter() - t0, 1))


def write(st: SimpleNamespace, reads: dict, vs: list[list[int]], out_dir, evaluate: bool = True, own: dict | None = None) -> dict:
    """The statement (`statement`) on its instances, written to ``out_dir``: the circuit, the prover's ``inst.bin``, the
    verifier's ``pub.bin``, ``program.json``, ``partition.json`` and the reads file ``registered.json``.

    ``reads``: per row port, ``(registration, positions)``, the prover's registered value (`registered.Registered`) and each
    instance's row of it; ``vs``: each instance's v (F elements), which the verifier registers as `V_VALUE` (`verifier_value`),
    instance i reading row i. ``own``: per row port the verifier's own entry (default the registration's), which its reads file
    names; the outputs are the verifier's zero rows, registered under its own salts."""
    gf = st.S["gf"]
    s = write_rows(st, reads, [words(v, st.v_words) for v in vs], out_dir, evaluate, own)
    nonzero = [[i for i in range(len(gf["res"])) if any(x[8 * i:8 * i + 8])] for x in s.pop("ref")]
    return {"template": st.defn.id, "instances": st.n, "ports": len(st.names), "nw": gf["nw"], "nv": gf["nv"],
            "products": sum(len(r["mul"]) for r in gf["res"]), "residuals": len(gf["res"]), "holds": s.pop("holds"),
            "nonzero_residuals": nonzero, **s}


def write_rows(st: SimpleNamespace, reads: dict, vrows: list[list[int]], out_dir, evaluate: bool = True,
               own: dict | None = None) -> dict:
    """`write` for any statement of its form: ``st`` names its row ports (``names``, each read through ``reads``), the words
    of its ``v`` row (``v_words``; ``vrows``, which the verifier registers as ``st.v_value``) and of its one output row
    (``out_words``, the verifier's zero rows under ``st.zero_value``), and its reference ``st.ref(rows by name, v's words)``.
    Returns what every such statement costs and holds, with each instance's reference output (``ref``)."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG

    n, low, sub, text, names = st.n, st.low, st.sub, st.text, st.names
    if set(reads) != set(names) or len(vrows) != n or any(len(pos) != n for _, pos in reads.values()):
        raise ValueError(f"reads of {sorted(reads)} and {len(vrows)} v rows for the row ports {names} of {n} instances")
    tables, salts, registered, entries = {}, {}, {}, {}
    for nm in names:
        reg, pos = reads[nm]
        rows, ss, e = reg.statement(pos)
        tables[nm], salts[nm], registered[nm] = np.array(rows, np.uint16), ss, e
        entries[nm] = {**((own or {}).get(nm) or reg.entry()), "word_bits": 16, "reads": [int(p) for p in pos]}
    vreg = verifier_value(st.prog, st.v_value, [list(r) for r in vrows])
    vrows_, salts["v"], registered["v"] = vreg.statement(range(n))
    tables["v"] = vt = np.array(vrows_, np.uint16)
    entries["v"] = {**vreg.entry(), "word_bits": 16, "reads": list(range(n))}
    ref = [st.ref({nm: tables[nm][i].tolist() for nm in names}, vt[i].tolist()) for i in range(n)]
    agrees = None
    if evaluate and low is not None:
        got, _cut, ok = low.evaluate(np.concatenate([tables[nm] for nm in names] + [vt], axis=1).astype(np.uint64))
        agrees = [bool((int(ok) >> i) & 1) and got[i].astype(np.uint16).tolist() == ref[i] for i in range(n)]
    zero = RG.Registered.commit(st.prog, st.zero_value, [[0] * st.out_words for _ in range(n)],
                                salts=zero_salts(st.zero_value, n))
    out_rows, salts["out"], registered["out"] = zero.statement(range(n))
    entries["out"] = {**zero.entry(), "word_bits": 16, "reads": list(range(n))}
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(st.program)))
    (d / "partition.json").write_text(json.dumps(st.part))
    RB.check({"format": RG.READS, "ports": entries}, st.defn.id, n)
    (d / "registered.json").write_text(json.dumps({"format": RG.READS, "ports": entries}))
    ds = getattr(st, "digests", None) or (FC.digest(text), FC.class_digest(text))
    FC.write(text, SimpleNamespace(name=st.zero_value, subcircuit=sub), 0, n, d / "inst.bin", d / "pub.bin", None, list(range(n)),
             share_rows=True, outs={"out": np.asarray(out_rows, np.uint16)}, tables=tables, refs={nm: list(range(n)) for nm in tables},
             salts=salts, registered=registered, digests=ds)
    meta = FC.meta_of(text)
    return {"holds": [not any(x) for x in ref], "ref": ref, "unit_agrees": agrees,
            "unit_rows": low.layout.useful if low is not None else st.unit_rows, "unit_log": meta["unit_log"],
            "k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"], "per_vu": meta["per_vu"], "circuit_sha512": ds[0],
            "circuit_bytes": len(text), "circuit_kept": getattr(st, "circuit_kept", False), "program_sha512": st.prog,
            "partition_digest": P.digest(st.part), "zero_root": zero.root.hex(), "build_s": st.build_s}


# --- the registered opening -------------------------------------------------------------------------------------------------

def opening_rows(sh, w: list[int]) -> dict[str, list[int]]:
    """An opening-only shape's row of each port (`spec`'s, by name) from its w (`rec_algebra.witness`): the port's elements as
    words, zero words up to ``P(n)``."""
    S = spec(sh)
    if not sh.opening_only or len(w) != S["gf"]["nw"]:
        raise ValueError(f"{sh}: rows of a registered opening's w, {S['gf']['nw']} elements")
    out, at = {}, 0
    for nm, n in S["ports"]:
        out[nm] = words(w[at:at + n], port_words(n))
        at += n
    return out


def stage_opening(sh, regs: dict, acc: tuple, v: list[int], out_dir, evaluate: bool = True, own: dict | None = None) -> list[dict]:
    """The registered opening (an opening-only `rec_algebra.Shape`) as ``InnerRepCheck_v2`` statements of one instance, one per
    part (`parts`), each unit ``rec-opening/m<m>-c<claims>-p<j>``, written to ``out_dir/p<j>`` (`write`).

    ``regs``: per round port ``m<i>``, the prover's one-row registration of the round's message, and under ``claim`` the
    prover's ``rec-claim`` (`rec_outer.CLAIM_VALUE`): ``claim<k>`` reads its row k. ``acc``: the ``rec-acc`` registration and
    the rows ``acc0`` and ``acc1`` read. ``v``: the opening's v (`rec_algebra.verifier_rows`), its claims' points from the
    coins. ``own``: per row port the verifier's own entry its reads file names."""
    from .rec_outer import CLAIM_VALUE

    if not sh.opening_only:
        raise ValueError(f"{sh} is not a registered opening")
    if regs["claim"].value != CLAIM_VALUE or len(regs["claim"].rows) != sh.claims:
        raise ValueError(f"the claims are {sh.claims} rows of {CLAIM_VALUE}, not {len(regs['claim'].rows)} of "
                         f"{regs['claim'].value}")
    reads = {nm: (regs[nm], [0]) for nm in regs if nm != "claim"}
    reads["acc0"], reads["acc1"] = (acc[0], [acc[1]]), (acc[0], [acc[2]])
    reads.update({f"claim{k}": (regs["claim"], [k]) for k in range(sh.claims)})
    out = []
    for j, p in enumerate(parts(spec(sh))):
        st = statement(p.S, f"rec-opening/m{sh.m}-c{sh.claims}-p{j}", 1)
        names = [nm for nm, _ in p.S["ports"]]
        s = write(st, {nm: reads[nm] for nm in names}, [[v[i] for i in p.v]], Path(out_dir) / f"p{j}", evaluate=evaluate,
                  own={nm: (own or {})[nm] for nm in names if nm in (own or {})})
        out.append({**s, "part": j, "of_residuals": p.res, "regions": regions(p.S["ports"])})
    return out
