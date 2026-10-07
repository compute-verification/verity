"""V*'s algebra statement (hiding model architectures, lane rec-v0): rec-algebra's `InnerRepCheck` = ``GfResiduals_v1{S}``
(`verity_flock.rec_algebra`) on each rep of the inner session, one instance a rep, as one C-Flock statement.

Ports (u16 words; an F element is 8 words ``lo || hi``, so bit k of element i is bit ``k % 16`` of word ``8 i + k // 16``):

    w    S's ``nw`` witness elements (`rec_algebra.witness`: the rep's free messages, its consistency sums, the comb's
         hidden part), zero words up to a whole 64-word row: a private row
    v    S's ``nv`` coefficients and constants at the rep's coins (`rec_algebra.verifier_rows`): a public input port
         (PROTOCOL.md §16.10), the verifier's own recomputation from the inner session's public record, not a row
    out  the residuals (8 words each), zero words up to `OUT_WORDS`: one output row, registered by the verifier as zero, so an
         instance is satisfiable only if every one of its rep's inner checks holds

The M0 unit (`unit_circuit`) computes S's residuals with each residual's products summed leaf by leaf. Karatsuba's
recombination and the reduction are linear, so ``Σ_p reduce(K(a_p, b_p)) = reduce(recombine(Σ_p leaves_p))``: each of the
2,187 leaf sums (the XOR of the residual's products' i-th ANDs) is committed (a copy row, `sha512.commit`) and recombined once.
A residual's output forms are then one product's, about 75K nonzeros, not its products' (306M for a rep at m = 27).

**Stubs.** The comb: w's ``comb_hid`` slots hold the whole circuit fold of §10 at the rep's coins and v's ``comb_pub`` slots
are zero, so nothing ties the comb to M until rec-amat's `HiddenComb` writes ``comb_hid`` (and the class's public part
``comb_pub``). The messages and the consistency sums are fresh private words, not yet read from c_k's registration and the
openings' `GfScale` outputs.
"""
from __future__ import annotations

import hashlib
import json
import time
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from verity.ml.commitments import hm96
from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.defs import bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.ops.boolean import gf2k as G
from verity.ml.circuits.boolean.forms import ZERO, Circuit, L
from verity.ml.commitments.gates.sha512 import commit

from . import circuit as FC
from . import ir_lower as IL
from . import rec_algebra as A
from .class_statement import ClassDefinition

OUT_WORDS = 128
LEAVES = 3 ** 7
ZERO_SALT = b"verity/rec-v0/zero-residual-salt/v0"
DERIVATION = "verity/rec-v0/inner-rep-check/coefficients/v0"


def w_words(S: dict) -> int:
    return -(-8 * S["nw"] // 64) * 64


def v_words(S: dict) -> int:
    return 8 * S["nv"]


def words(elems, n_words: int) -> list[int]:
    """F elements as u16 words (``lo || hi``, 8 words each), zero words up to ``n_words``."""
    out = [(x >> (16 * j)) & 0xFFFF for x in elems for j in range(8)]
    return out + [0] * (n_words - len(out))


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


def unit_circuit(S: dict):
    """The unit over `forms`: inputs ``in0 ..`` (w's words, then v's, 16 bits each), outputs ``out0 .. out{OUT_WORDS - 1}``."""
    if 8 * len(S["res"]) > OUT_WORDS:
        raise ValueError(f"{len(S['res'])} residuals: more than {OUT_WORDS // 8}")
    C = Circuit()
    nw = w_words(S)
    ins = [C.input(f"in{j}", 16) for j in range(nw + v_words(S))]
    flat = [b for wd in ins for b in wd]
    W, V = flat[:16 * nw], flat[16 * nw:]

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
    bits += [ZERO] * (16 * OUT_WORDS - len(bits))
    outs = [f"out{j}" for j in range(OUT_WORDS)]
    for j, nm in enumerate(outs):
        C.output(nm, bits[16 * j:16 * j + 16])
    return C, [f"in{j}" for j in range(len(ins))], outs


def residual_words(S: dict, w: list[int], v: list[int]) -> list[int]:
    """The reference: the unit's output words on the ports' elements (`gf2k.residuals`)."""
    return words(G.residuals(S, w, v), OUT_WORDS)


def lowering(S: dict, name: str) -> IL.Lowering:
    C, ins, outs = unit_circuit(S)
    w_in, w_out = len(ins), len(outs)
    units = IL.Units(target=None, in_widths=[16] * w_in, out_widths=[16] * w_out, in_src=[[("leaf", p) for p in range(w_in)]],
                     out_src=[[("ret", p) for p in range(w_out)]], signature=(name,), gates=[], arg_gates=[], ret_gates=[])
    return IL.Lowering(name, ClassDefinition(name, (w_words(S), v_words(S)), w_out), units, C, IL.layout(C, ins, outs))


def subcircuit(defn, name: str) -> SimpleNamespace:
    """What `circuit.compose` and `circuit.write` read of a subcircuit: its ports and its Definition, ``GfResiduals_v1{S}``."""
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")
    digest = program_digest(Program(defn))
    return SimpleNamespace(id=name, template=SimpleNamespace(id="rec-residuals"), inputs=[port("w"), port("v")],
                           outputs=[port("out")], definition=lambda **_: defn, program_digests=lambda statics: {defn.id: digest})


def zero_salts(value: str, n: int) -> list[bytes]:
    return [hashlib.shake_256(ZERO_SALT + value.encode() + b"\0" + i.to_bytes(4, "little")).digest(hm96.SHA512.salt_bytes)
            for i in range(n)]


def rep_inputs(inner: Path, claims: dict, rep: int, shape: A.Shape) -> tuple[list[int], list[int]]:
    """One rep's ports as F elements: the prover's w (its messages, consistency sums and whole comb, the stub) and the verifier's
    v (its coins' coefficients, the comb's public part zero)."""
    record = json.loads((inner / "session" / "session.json").read_text())
    rounds = A.rep_rounds(record, f"circuit/rep{rep}")
    c = claims["reps"][rep]
    extras = [A.Claim(G.of_hex(e["value"]), tuple(map(G.of_hex, e["skip"])), tuple(map(G.of_hex, e["x"]))) for e in c["extras"]]
    v, _info = A.verifier_rows(shape, rounds, extras, [0] * 64)
    p = A.decode(shape, (inner / "session" / f"circuit.rep{rep}.bin").read_bytes())
    w = A.witness(shape, p, rounds, [G.of_hex(x) for x in c["comb"]], extras)
    return w, v


def stage(inner: Path, claims: dict, out_dir: Path, forged: Path | None = None, evaluate: bool = True) -> dict:
    """The statement over both reps (instance r = rep r), the verifier's own files beside the prover's (``out_dir/honest``),
    and what it costs. ``forged``: a session whose coins differ from ``inner``'s (a forged transcript), staged too
    (``out_dir/forged``, the same circuit): the honest prover's w beside the verifier's v at the forged coins."""
    shape = A.Shape(int(claims["m"]), int(claims["k_log"]), 2 + len(claims["reps"][0]["extras"]))
    S, _layout = A.structure(shape)
    reps = [rep_inputs(inner, claims, r, shape) for r in range(2)]
    st = statement(S, f"rec-residuals/m{shape.m}-k{shape.k_log}-c{shape.claims}", len(reps))
    out = {"shape": [shape.m, shape.k_log, shape.claims], "build_s": st.build_s,
           "honest": write_statement(st, [w for w, _ in reps], [v for _, v in reps], Path(out_dir) / "honest", evaluate)}
    if forged is not None:
        vs = [rep_inputs(forged, claims, r, shape)[1] for r in range(2)]
        out["forged"] = write_statement(st, [w for w, _ in reps], vs, Path(out_dir) / "forged", evaluate)
    return out


def statement(S: dict, name: str, n: int) -> SimpleNamespace:
    """``GfResiduals_v1{S}`` over ``n`` instances: its Program, partition, unit and circuit text, which only ``n`` and ``S``
    decide."""
    from verity.experimental.protocols.sampled_proofs.one_stage import partition as P

    t0 = time.perf_counter()
    defn = bind(G.GfResiduals, S=S)
    program = P.population_program(defn, n)
    part = P.partition(program, P.template_instance_query(defn))
    prog = P.program_sha512(program)
    low = lowering(S, name)
    sub = subcircuit(defn, name)
    public = {"v": {"seed": "the inner session's public record (rec/inner-transcript/v0), rep = instance", "derivation": DERIVATION}}
    text = FC.compose(low, sub, partition=P.digest(part), program=prog, public_inputs=public)
    return SimpleNamespace(S=S, n=n, defn=defn, program=program, part=part, prog=prog, low=low, sub=sub, text=text,
                           build_s=round(time.perf_counter() - t0, 1))


def write_statement(st: SimpleNamespace, ws: list[list[int]], vs: list[list[int]], out_dir: Path, evaluate: bool = True) -> dict:
    """The statement (:func:`statement`) on one instance per (w, v), its outputs registered as zero, written to ``out_dir``: the
    circuit, the prover's ``inst.bin``, the verifier's ``pub.bin``, ``public-inputs.bin`` (v), ``program.json``,
    ``partition.json`` and the reads file ``registered.json``."""
    from verity.experimental.protocols.sampled_proofs.one_stage import partition as P
    from verity.experimental.protocols.sampled_proofs.one_stage import registered as RG

    S, n, low, sub, text, prog, program, part = st.S, st.n, st.low, st.sub, st.text, st.prog, st.program, st.part
    if len(ws) != n or len(vs) != n:
        raise ValueError(f"{len(ws)} witnesses and {len(vs)} coefficient rows for a statement of {n} instances")
    wrows = np.array([words(w, w_words(S)) for w in ws], np.uint16)
    vrows = np.array([words(v, v_words(S)) for v in vs], np.uint16)
    ref = [residual_words(S, w, v) for w, v in zip(ws, vs, strict=True)]
    agrees = None
    if evaluate:
        got, _cut, ok = low.evaluate(np.concatenate([wrows, vrows], axis=1).astype(np.uint64))
        agrees = [bool((int(ok) >> i) & 1) and got[i].astype(np.uint16).tolist() == ref[i] for i in range(n)]
    value = "rec-residuals-zero"
    reg = RG.Registered.commit(prog, value, [[0] * OUT_WORDS for _ in range(n)], salts=zero_salts(value, n))
    out_rows, out_salts, entry = reg.statement(range(n))
    own = {"format": RG.READS, "ports": {"out": {**reg.entry(), "word_bits": 16, "reads": list(range(n))}}}
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text(text)
    (d / "program.json").write_bytes(canonical_json(encode_program(program)))
    (d / "partition.json").write_text(json.dumps(part))
    (d / "registered.json").write_text(json.dumps(own))
    FC.write_public_inputs(text, {"v": vrows}, n, d / "public-inputs.bin")
    salts = {"w": [hm96.SHA512.fresh_salt() for _ in range(n)], "out": out_salts}
    FC.write(text, SimpleNamespace(name=value, subcircuit=sub), 0, n, d / "inst.bin", d / "pub.bin", None, list(range(n)),
             share_rows=True, outs={"out": np.asarray(out_rows, np.uint16)}, tables={"w": wrows}, refs={"w": list(range(n))},
             salts=salts, registered={"out": entry}, public_words={"v": vrows})
    meta = json.loads(text.split("\n", 2)[1][5:])
    return {"template": st.defn.id, "instances": n, "nw": S["nw"], "nv": S["nv"], "products": sum(len(r["mul"]) for r in S["res"]),
            "residuals": len(S["res"]), "holds": [not any(x) for x in ref],
            "nonzero_residuals": [[i for i in range(len(S["res"])) if any(x[8 * i:8 * i + 8])] for x in ref],
            "unit_agrees": agrees, "unit_rows": low.layout.useful, "unit_log": meta["unit_log"], "k_log": meta["k_log"],
            "vus_per_block": meta["vus_per_block"], "per_vu": meta["per_vu"], "circuit_sha512": FC.digest(text),
            "circuit_bytes": len(text), "program_sha512": prog, "partition_digest": P.digest(part), "registered_root": reg.root.hex()}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("inner", help="the inner session (session/, claims.json: 80-rec-inner.sh's out/inner)")
    ap.add_argument("out", help="honest/ and, with --forged, forged/")
    ap.add_argument("--forged", default=None, help="a forged copy of the inner session (rec_inner.py forge): its coins for v")
    ap.add_argument("--no-evaluate", action="store_true", help="skip evaluating the unit against the reference")
    a = ap.parse_args(argv)
    inner = Path(a.inner)
    claims = json.loads((inner / "claims.json").read_text())
    s = stage(inner, claims, Path(a.out), Path(a.forged) if a.forged else None, not a.no_evaluate)
    print(json.dumps(s, sort_keys=True), flush=True)
    agrees = lambda x: x["unit_agrees"] in (None, [True] * x["instances"])
    ok = all(s["honest"]["holds"]) and agrees(s["honest"])
    return 0 if ok and ("forged" not in s or (not all(s["forged"]["holds"]) and agrees(s["forged"]))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
