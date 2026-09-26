"""V[B]'s inputs from a recorded inner session: the prototype's hidden-message mode, applied after the fact.

A recorded session carries each round's framed bytes (or their digest) and the coins that followed. The private track's
coin server would record `c_j = HM(m_j; r_j)` instead (Project store `docs/circuit-privacy.md` I.2). `commitments` builds
those from what the record says was sent: the retained bytes, else the proof's bytes when they hash to the recorded digest,
else the digest itself, which no witness opens. So a record the public verifier rejects on R2 gives commitments the
witness does not open. The witness is the prover's side: its round bytes rendered from the proofs, the query answers,
Enc(C), the data openings and the lookups' hints.

`convert` refuses, as a decode step, what has no counterpart in V[B]'s inputs: a proof that does not decode to the
schedule's shape, PCS parameters other than the pinned ones, and the two proof fields that are neither messages nor query
answers when they are not what the protocol fixes: a nonzero ring-switch grinding nonce (declared divergence D-rs) and a
`final_c_eval` other than P_c(z) (D-fc; V[B] computes P_c from round1_c and never reads the field).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from . import gf, hm, memcheck
from .session import Session, decode_proof, render_rounds
from .statement import PrivateCircuit, Public, Statement
from .vb import Frame, R, const_bits


class Refused(Exception):
    pass


def _fbits(a: np.ndarray) -> np.ndarray:
    """F arrays (..., 2) uint64 -> bits (..., 128), bit i = coefficient of x^i."""
    return np.unpackbits(np.ascontiguousarray(a).view(np.uint8).reshape(a.shape[:-1] + (16,)), axis=-1, bitorder="little")


def enc_fields(fr: Frame, C: PrivateCircuit) -> dict[str, np.ndarray]:
    B = fr.B
    if len(C.a) > B.qa_max or len(C.b) > B.qb_max or C.useful > (1 << B.unit_log) - 1:
        raise Refused("C does not fit B")
    pad = 1 << B.unit_log

    def entries(es, q):
        r = np.full(q, pad, dtype=np.int64)
        c = np.zeros(q, dtype=np.int64)
        if es:
            a = np.array(es, dtype=np.int64)
            r[:len(es)], c[:len(es)] = a[:, 0], a[:, 1]
        return const_bits(r, 14), const_bits(c, 13)

    ra, ca = entries(C.a, B.qa_max)
    rb, cb = entries(C.b, B.qb_max)
    return {"enc.cp": const_bits(C.const_pos, 13), "enc.ra": ra, "enc.ca": ca, "enc.rb": rb, "enc.cb": cb,
            "enc.lin": const_bits(np.array(C.leaves_in), 7), "enc.lout": const_bits(np.array(C.leaves_out), 6),
            "enc.dummy": const_bits(np.array(C.dummy_outputs), 16)}


def enc_bytes(fr: Frame, fields: dict[str, np.ndarray]) -> bytes:
    flat = np.concatenate([fields[n].reshape(-1) for n, _ in fr.enc_fields()])
    flat = np.concatenate([flat, np.zeros((-len(flat)) % 8, np.uint8)])
    return np.packbits(flat, bitorder="little").tobytes()


def decode(fr: Frame, sess: Session) -> list:
    proofs = []
    for rep in range(R):
        if rep not in sess.proofs:
            raise Refused(f"rep {rep}: no proof")
        try:
            pr = decode_proof(sess.proofs[rep], fr.B.hash_len)
        except (ValueError, IndexError) as e:
            raise Refused(f"rep {rep}: {e}")
        sch = fr.shape.sched
        want = (fr.B.m, 1, sch.k0, 1, None, 0 if fr.B.hash_len == 32 else 2)
        if pr.params != want:
            raise Refused("R7/F2: the proof's PCS parameters are not the pinned fast100 parameters")
        if len(pr.rs) != fr.shape.claims or any(len(s) != 128 for s, _ in pr.rs):
            raise Refused("opening: ring_switches is not one per claim")
        if any(nonce != 0 for _, nonce in pr.rs):
            raise Refused("D-rs: ring switch grinding_nonce ≠ 0 (a proof-format constant; no counterpart in V[B])")
        proofs.append(pr)
    return proofs


def commitments(fr: Frame, sess: Session, rounds: list[list[bytes]], salts: list[bytes]) -> np.ndarray:
    """c_j for root_B and every round, from what the record says the prover sent."""
    out = [hm.commit("inner-round", sess.root_b, salts[0])]
    for j, (rep, k) in enumerate(fr.rounds):
        rec = sess.streams[f"circuit/rep{rep}"].rounds[k]
        mine = rounds[rep][k]
        if rec.msg is not None:
            sent = rec.msg
        elif hashlib.sha256(mine).digest() == rec.sha:
            sent = mine
        else:
            sent = rec.sha
        out.append(hm.commit("inner-round", sent, salts[j + 1]))
    return np.frombuffer(b"".join(out), dtype=np.uint8).reshape(-1, 64).copy()


def convert(fr: Frame, st: Statement, C: PrivateCircuit, data: Public, sess: Session, tabs: dict,
            c_of: PrivateCircuit | None = None, rng: np.random.Generator | None = None) -> tuple[dict, dict]:
    """(public inputs c, D, cj; witness) for session `sess` proved with circuit C. `c_of` names the circuit the registry
    holds (default C)."""
    B, sch = fr.B, fr.shape.sched
    rand = (lambda n: rng.bytes(n)) if rng is not None else os.urandom
    proofs = decode(fr, sess)
    for rep, pr in enumerate(proofs):
        wl = gf.from_arr(tabs["zc.wL"][rep])
        pc = 0
        for a, b in zip(wl, pr.zc["c"]):
            pc ^= gf.mul(a, b)
        if pc != pr.zc["fc"]:
            raise Refused(f"D-fc: rep {rep}: final_c_eval is not P_c(z) (a proof field V[B] computes, never reads)")
    rounds = []
    for rep in range(R):
        st_ = sess.streams.get(f"circuit/rep{rep}")
        if st_ is None or len(st_.rounds) != len(fr.layouts[rep].rounds) \
                or [x.n for x in st_.rounds] != fr.layouts[rep].ns:
            raise Refused(f"S11/S15: circuit/rep{rep}: the record's rounds are not the schedule's")
        rr = render_rounds(fr.shape, sess, rep, proofs[rep])
        if [len(b) for b in rr] != [fr.layouts[rep].round_bytes(k) for k in range(len(rr))]:
            raise Refused(f"rep {rep}: the proof's messages are not the schedule's shape")
        rounds.append(rr)
    J = len(fr.msg_lens)
    salts = [rand(hm.SALT_BYTES) for _ in range(J)]
    m_commit = b"".join(proofs[0].cap)
    m = m_commit + b"".join(rounds[r][k] for r, k in fr.rounds)
    fields = enc_fields(fr, C)
    r_c = rand(hm.SALT_BYTES)
    reg_fields = enc_fields(fr, c_of) if c_of is not None else fields
    fv = data.header["frame_v3"]
    pub = {"c": np.frombuffer(hm.commit("circuit", enc_bytes(fr, reg_fields), r_c), np.uint8).copy(),
           "D": np.stack([np.frombuffer(bytes.fromhex(fv["roots"][n]), np.uint8) for n in fr.ports + [fr.out_port]]),
           "cj": commitments(fr, sess, rounds, salts)}
    w = {"m": np.frombuffer(m, np.uint8).copy(), "r": np.frombuffer(b"".join(salts), np.uint8).reshape(J, -1).copy(),
         "r_c": np.frombuffer(r_c, np.uint8).copy(), **fields}
    # query answers
    hl = B.hash_len
    for lv, L in enumerate(sch.levels):
        rows_r, sibs_r = [], []
        for pr in proofs:
            lig = pr.lig
            if lv == 0:
                rows, sibs = lig["iproof"]
            elif lv < len(sch.levels) - 1:
                rows, sibs = lig["rproofs"][lv - 1]
            else:
                rows, sibs = lig["frows"], lig["fsibs"]
            if len(rows) != L.q or any(len(x) != 2 ** L.lanes for x in rows) or len(sibs) != L.q * (L.d - L.cap):
                raise Refused(f"merkle: level {lv}: shape")
            rows_r.append(gf.to_arr([v for row in rows for v in row]).reshape(L.q, 2 ** L.lanes, 2))
            sibs_r.append(np.frombuffer(b"".join(sibs), np.uint8).reshape(L.q, L.d - L.cap, hl))
        w[f"lig.rows.{lv}"] = np.stack(rows_r)
        w[f"lig.sibs.{lv}"] = np.stack(sibs_r)
    # data openings
    w["data.dig"] = np.array([[np.frombuffer(d, np.uint8) for d in inst] for inst in data.digests])
    w["data.out"] = const_bits(np.array(data.outputs), 16)
    # lookups
    zp = [gf.to_arr(pr.lc["zp"]) for pr in proofs]
    U = 1 << B.unit_log
    cols = np.arange(U)
    e_vals = _fbits(np.transpose(tabs["lc.e0"], (1, 0, 2))).reshape(U + 1, R * 128)
    ra = np.r_[[r for r, _ in C.a], np.full(B.qa_max - len(C.a), U)]
    rb = np.r_[[r for r, _ in C.b], np.full(B.qb_max - len(C.b), U)]
    w.update(memcheck.lookup_witness("mc.E", U + 1, np.r_[ra, rb, C.const_pos].astype(np.int64), e_vals))
    vloc = np.stack([gf.vmul(tabs["lc.eqTlo"][r][cols >> 6], zp[r][cols & 63]) for r in range(R)], axis=1)
    ca = np.r_[[c for _, c in C.a], np.zeros(B.qa_max - len(C.a))]
    cb = np.r_[[c for _, c in C.b], np.zeros(B.qb_max - len(C.b))]
    w.update(memcheck.lookup_witness("mc.V", U, np.r_[ca, cb].astype(np.int64), _fbits(vloc).reshape(U, R * 128)))
    src = fr.leaf_src.reshape(-1)
    wmsg = np.stack([gf.vmul(tabs["lc.eqTmsg"][r], zp[r][src & 63]) for r in range(R)], axis=1)
    q = [t + 16 * C.leaves_in[u][j] + 2048 * g for g in range(B.g) for u in range(B.upv)
         for j in range(B.leaves_per_unit_in) for t in range(16)]
    w.update(memcheck.lookup_witness("mc.M", src.size, np.array(q), _fbits(wmsg).reshape(src.size, R * 128)))
    table = const_bits(np.r_[np.array(data.outputs).reshape(-1), C.dummy_outputs], 16)
    qo = [C.leaves_out[u][j] + B.n_out_words * fr.inst_slots[b, g] for b in range(1 << B.nbl) for g in range(B.g)
          for u in range(B.upv) for j in range(B.leaves_per_unit_out)]
    w.update(memcheck.lookup_witness("mc.O", table.shape[0], np.array(qo), table))
    lout = np.array(C.leaves_out).reshape(-1)
    if sorted(lout.tolist()) == list(range(len(lout))):
        w.update(memcheck.permutation_witness("phi.lout", lout))
    else:
        w["phi.lout.sw"] = np.zeros((2 * (len(lout).bit_length() - 1) - 1, len(lout) // 2), np.uint8)
    return pub, w
