"""V[B]: the fixed public verifier circuit of the private track (Project store `docs/circuit-privacy.md` I.3), hand-built.

Its relation, checked for both reps of the inner session at once:
1. `HM(Enc(C); r_c) = c`;
2. Φ(C): the unit's entries are in circuit form, its constant row follows its input rows, `leaves_out` is a permutation,
   and C fits B (fixed-size fields; padding entries canonical);
3. `HM(m_j; r_j) = c_j` for the commitment of root_B and every round of both reps;
4. the inner verifier accepts (PROTOCOL.md §9–§14): the round framing, zerocheck, lincheck with the matrix fold computed
   from Enc(C), the ring switch, and Ligerito with its Merkle openings; both reps bind one root (R1);
5. the data the region claims read (row digests, outputs) hashes to the frame-v3 roots D.

Every inner check is an affine form in the inner messages whose coefficients depend only on the inner coins; those
coefficients are public inputs (`coins`). The products that remain private × private are the zerocheck's final product
and the fold's per-entry products, whose factors come from lookups at private positions (`memcheck`).

The builder reads only the public frame, never C or a session, so the program (and `Program.digest`) is a function of B.
Two statements with equal frames give identical V[B]s: that is the swap test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import gadgets, gf, hm, memcheck
from .program import Program, Reg
from .session import Layout, Shape, layout
from .statement import Bounds, Statement

R = 2
FV3 = b"veritor/protocol/merkle/frame/v3\0"


def _uint(x: int) -> bytes:
    b = x.to_bytes(8, "big")
    return b.lstrip(b"\0") or b"\0"


def fv3_head(tag: bytes, parts: list[bytes]) -> bytes:
    """The frame-v3 hash input up to the parts given (each with its be64 length)."""
    out = FV3 + len(tag).to_bytes(4, "big") + tag
    for part in parts:
        out += len(part).to_bytes(8, "big") + part
    return out


@dataclass
class Frame:
    """The public frame: everything V[B] may depend on."""
    name: str
    B: Bounds
    shape: Shape
    domains: tuple[bytes, bytes]
    regions: list
    pin: int
    unit_pos0: int
    leaf_src: np.ndarray            # (g, leaf, t) -> message column
    inst_slots: np.ndarray          # (block, g) -> instance, or n for a padding instance
    fv3_domains: dict[str, bytes]
    fv3_schemas: dict[str, bytes]
    ports: list[str]
    out_port: str
    layouts: list[Layout] = field(default_factory=list)

    @classmethod
    def of(cls, st: Statement, name: str = "rope-head/d64 prototype") -> "Frame":
        B = st.B
        sh = Shape(B.m, B.k_log, 2 + 2 * len(st.regions), B.hash_len)
        dom = (b"flock-circuit/fast100x2/rep0", b"flock-circuit/fast100x2/rep1")
        src = np.array([[[st.leaf_src(g, leaf, t) for t in range(16)] for leaf in range(B.n_in_words)] for g in range(B.g)])
        slots = np.array([[st.inst_of(b, g) if st.inst_of(b, g) is not None else B.n for g in range(B.g)]
                          for b in range(1 << B.nbl)])
        fv = st.pub.header["frame_v3"]
        fr = cls(name, B, sh, dom, st.regions, st.pin, st.circ.unit_range.pos0, src, slots,
                 {k: bytes.fromhex(v) for k, v in fv["domain_ids"].items()},
                 {k: v.encode() for k, v in fv["schemas"].items()}, [pt.name for pt in st.circ.ports],
                 st.circ.meta["out_ports"][0][0])
        fr.layouts = [layout(sh, d) for d in dom]
        return fr

    @property
    def rounds(self) -> list[tuple[int, int]]:
        """The inner rounds in V[B]'s order: rep 0's, then rep 1's (the record's interleaving does not matter)."""
        return [(r, k) for r in range(R) for k in range(len(self.layouts[r].rounds))]

    @property
    def msg_lens(self) -> list[int]:
        """Committed messages: root_B (the cap), then every round's framed bytes."""
        cap = 2 ** self.shape.sched.levels[0].cap * self.B.hash_len
        return [cap] + [self.layouts[r].round_bytes(k) for r, k in self.rounds]

    @property
    def msg_offsets(self) -> list[int]:
        return [int(x) for x in np.cumsum([0] + self.msg_lens)[:-1]]

    def enc_fields(self) -> list[tuple[str, tuple[int, ...]]]:
        """Enc(C)'s fields in order, as bit arrays (LSB first); the encoding concatenates them and pads to bytes.
        Rows are 14 bits: 8192 marks a padding entry."""
        B = self.B
        return [("enc.cp", (13,)), ("enc.ra", (B.qa_max, 14)), ("enc.ca", (B.qa_max, 13)), ("enc.rb", (B.qb_max, 14)),
                ("enc.cb", (B.qb_max, 13)), ("enc.lin", (B.upv, B.leaves_per_unit_in, 7)),
                ("enc.lout", (B.upv, B.leaves_per_unit_out, 6)), ("enc.dummy", (B.n_out_words, 16))]

    @property
    def algo(self) -> str:
        return "sha256" if self.B.hash_len == 32 else "sha512"


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------

def lin(p: Program, coef: Reg, x: Reg, axes) -> Reg:
    """Σ coef·x over the given axes (coef public, x private): one F product per term."""
    y = p.fmul(coef, x)
    for ax in sorted(axes, reverse=True):
        y = p.xorsum(y, ax)
    return y


def transpose(p: Program, a: Reg, axes) -> Reg:
    return p.gather(a, np.arange(a.size).reshape(a.shape).transpose(axes))


def kmul(p: Program, a: Reg, b: Reg) -> Reg:
    """K products of (..., 2) registers (c0, c1) (PROTOCOL.md §2.4): three F products."""
    a0, a1, b0, b1 = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
    p0, p1, p2 = p.fmul(a0, b0), p.fmul(a1, b1), p.fmul(a0 ^ a1, b0 ^ b1)
    c0, c1 = p0 ^ p.fmulc(p1, gf.XINV), p2 ^ p0
    k = len(c0.shape)
    return p.concat([c0.reshape(*c0.shape, 1), c1.reshape(*c1.shape, 1)], axis=k)


def kfmul(p: Program, a: Reg, f: Reg) -> Reg:
    """K · F coordinate-wise: two F products."""
    return p.fmul(a, f.reshape(*f.shape, 1))


def bit_or(p: Program, x: Reg, y: Reg) -> Reg:
    return x ^ y ^ p.band(x, y)


def bits_lt(p: Program, a: Reg, b: Reg) -> Reg:
    """a < b for (..., k) LSB-first bit registers: the borrow out of a - b, one AND per bit."""
    borrow = None
    for i in range(a.shape[-1]):
        x = p.bnot(a[..., i])
        y = b[..., i]
        borrow = p.band(x, y) if borrow is None else x ^ p.band(x ^ y, x ^ borrow)
    return borrow


def bits_eq(p: Program, a: Reg, b: Reg) -> Reg:
    d = p.bnot(a ^ b)
    acc = d[..., 0]
    for i in range(1, d.shape[-1]):
        acc = p.band(acc, d[..., i])
    return acc


def const_bits(values, width: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.int64)
    return ((v[..., None] >> np.arange(width)) & 1).astype(np.uint8)


def md_blocks(p: Program, algo: str, parts: list[Reg], total_prefix: int = 0) -> Reg:
    """Concatenate byte registers (..., L_i) and FIPS padding into (..., nb, block) blocks."""
    lead = parts[0].shape[:-1]
    L = sum(x.shape[-1] for x in parts)
    pad = np.frombuffer(gadgets.md_pad(algo, total_prefix + L), dtype=np.uint8)
    blk = gadgets.SHA_BLOCK[algo]
    msg = p.concat(parts + [p.const("byte", np.broadcast_to(pad, lead + pad.shape).copy())], axis=len(lead))
    return msg.reshape(*lead, -1, blk)


def sha_iv(p: Program, algo: str) -> Reg:
    return p.const("byte", np.frombuffer(gadgets.iv_bytes(algo), dtype=np.uint8).copy())


def _consts_of(fr: Frame):
    offs, vals = [], []
    base = fr.msg_offsets
    for j, (r, k) in enumerate(fr.rounds):
        for s in fr.layouts[r].rounds[k]:
            if s.const is not None:
                offs.append(np.arange(s.size) + base[j + 1] + s.offset)
                vals.append(np.frombuffer(s.const, dtype=np.uint8))
    return np.concatenate(offs), np.concatenate(vals)


def field_offsets(fr: Frame) -> dict:
    """Flat offsets in m of each named message field, for both reps: {key: (R, size)}."""
    base = fr.msg_offsets
    out: dict = {}
    for j, (r, k) in enumerate(fr.rounds):
        for s in fr.layouts[r].rounds[k]:
            if s.key is not None:
                out.setdefault(s.key, [None, None])[r] = np.arange(s.size) + base[j + 1] + s.offset
    return {k: np.stack(v) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------------------------------------------------

def build(fr: Frame) -> Program:
    B, sh = fr.B, fr.shape
    sch = sh.sched
    N = sh.claims
    nlev = len(sch.levels)
    Rl = nlev - 1
    nmsg = sch.k0 + 8 + 7 * (Rl - 1)
    nood = 1 + 2 * Rl
    hl = B.hash_len
    p = Program(f"V[B] {fr.name}")
    J = len(fr.msg_lens)

    c_pub = p.public("c", "byte", (64,))
    D = p.public("D", "byte", (len(fr.ports) + 1, 32))
    cj = p.public("cj", "byte", (J, 64))

    def pub(name, shape, kind="F"):
        return p.public(name, kind, shape)

    m = p.witness("m", "byte", (sum(fr.msg_lens),))
    salts = p.witness("r", "byte", (J, hm.SALT_BYTES))
    fo = field_offsets(fr)

    def fbytes(key) -> Reg:
        return p.gather(m, fo[key])

    def ffield(key) -> Reg:
        return p.field(fbytes(key).reshape(R, -1, 16))

    def ffields(prefix, count) -> Reg:
        return p.field(p.gather(m, np.stack([fo[(prefix, i)] for i in range(count)], axis=1)).reshape(R, count, 16))

    with p.component("framing"):
        off, val = _consts_of(fr)
        p.assert_eq(p.gather(m, off), p.const("byte", val), "R2: a round's framing is not the schedule's")
        cap = fbytes(("cap",)).reshape(R, -1, hl)
        p.assert_eq(fbytes(("icap",)).reshape(R, -1, hl), cap, "ligerito: initial_cap differs from the commitment's cap")
        p.assert_eq(cap[0], cap[1], "S18/R1: the reps' caps differ")
        m_commit = p.gather(m, np.arange(fr.msg_lens[0])).reshape(-1, hl)
        p.assert_eq(m_commit, cap[0], "S16/R1: rep 0's cap is not the committed root_B")

    # 3. every message opens its commitment
    with p.component("commit-open/rounds"):
        by_len: dict[int, list[int]] = {}
        for j, L in enumerate(fr.msg_lens):
            by_len.setdefault(L, []).append(j)
        offs = fr.msg_offsets
        for L, js in sorted(by_len.items()):
            msgs = p.gather(m, np.stack([np.arange(L) + offs[j] for j in js]))
            leaf = hm.commit_circuit(p, "inner-round", msgs, salts[np.array(js)])
            p.assert_eq(leaf, cj[np.array(js)], "R2: a message does not open its commitment")

    # 1. C opens c
    enc = {name: p.witness(name, "bit", shape) for name, shape in fr.enc_fields()}
    r_c = p.witness("r_c", "byte", (hm.SALT_BYTES,))
    with p.component("commit-open/circuit"):
        flat = p.concat([enc[n].reshape(-1) for n, _ in fr.enc_fields()])
        pad = (-flat.shape[0]) % 8
        if pad:
            flat = p.concat([flat, p.const("bit", np.zeros(pad, np.uint8))])
        p.assert_eq(hm.commit_circuit(p, "circuit", p.bytes_(flat), r_c), c_pub, "C does not open c")

    # 2. Φ
    with p.component("phi"):
        cp = enc["enc.cp"]
        in_rows = p.const("bit", const_bits(B.in_rows, 13))
        p.assert_zero(bits_lt(p, cp, in_rows), "Φ: the constant row is an input row")
        p.assert_zero(bits_lt(p, p.const("bit", const_bits((1 << B.unit_log) - 2, 13)), cp), "Φ: the unit fills its slot")
        for rr, cc in (("enc.ra", "enc.ca"), ("enc.rb", "enc.cb")):
            r, c = enc[rr], enc[cc]
            padding, r13 = r[:, 13], r[:, :13]
            p.assert_zero(p.band(padding.reshape(-1, 1), r13), "Φ: a padding entry is not canonical")
            p.assert_zero(p.band(padding.reshape(-1, 1), c), "Φ: a padding entry is not canonical")
            ok = p.band(p.band(p.bnot(bits_lt(p, r13, in_rows)), bits_lt(p, r13, cp)),
                        bit_or(p, bits_lt(p, c, r13), bits_eq(p, c, cp)))
            p.assert_zero(p.band(p.bnot(padding), p.bnot(ok)), "Φ: an entry is not in circuit form")
        memcheck.permutation_check(p, "phi.lout", enc["enc.lout"].reshape(-1, 6))

    # 4. the inner verifier
    with p.component("inner/zerocheck"):
        ab, cz = ffield(("zc_ab",)), ffield(("zc_c",))
        g1, ginf = ffields("zc_g1", sh.m - 6), ffields("zc_ginf", sh.m - 6)
        fa, fb = ffield(("zc_fa",))[:, 0], ffield(("zc_fb",))[:, 0]
        claim = (lin(p, pub("zc.kab", (R, 64)), ab, [1]) ^ lin(p, pub("zc.kc", (R, 64)), cz, [1])
                 ^ lin(p, pub("zc.kg1", (R, sh.m - 6)), g1, [1]) ^ lin(p, pub("zc.kginf", (R, sh.m - 6)), ginf, [1]))
        p.assert_eq(claim, p.fmul(fa, fb), "zerocheck: SumcheckFinalFailed")
        v_c = lin(p, pub("zc.wL", (R, 64)), cz, [1])

    with p.component("inner/lincheck"):
        kl6 = sh.k_log - 6
        q1, qinf = ffields("lc_q1", kl6), ffields("lc_qinf", kl6)
        zp = ffield(("lc_zp",))
        claim = (p.fmul(pub("lc.kfa", (R,)), fa) ^ p.fmul(pub("lc.kfb", (R,)), fb) ^ pub("lc.kconst", (R,))
                 ^ lin(p, pub("lc.kq1", (R, kl6)), q1, [1]) ^ lin(p, pub("lc.kqinf", (R, kl6)), qinf, [1]))
        lhs = lin(p, pub("lc.ypub", (R, 64)), zp, [1])
        with p.component("fold"):
            lhs = lhs ^ _private_fold(p, fr, enc, zp, pub)
        p.assert_eq(lhs, claim, "lincheck: ConsistencyFailed (sumcheck-final)")
        w = lin(p, pub("lc.wrp", (R, 64)), zp, [1])

    with p.component("data"):
        vals = _region_values(p, fr, enc, pub, D)

    with p.component("inner/ring-switch"):
        s = p.field(p.gather(m, np.stack([fo[("rs", k)] for k in range(N)], axis=1)).reshape(R, N, 128, 16))
        claim_vals = p.concat([w.reshape(R, 1), v_c.reshape(R, 1), vals], axis=1)
        p.assert_eq(lin(p, pub("rs.w", (R, N, 128)), s, [2]), claim_vals, "opening: ring-switch claim mismatch")
        t = p.field(transpose(p, p.bits(s), (0, 1, 3, 2)))
        T = lin(p, pub("rs.kT", (R, N, 128)), t, [1, 2])
        p.assert_eq(ffield(("T",))[:, 0], T, "R2: the batched target T is not the verifier's")

    with p.component("inner/ligerito"):
        es = []
        for lv in range(nlev):
            L = sch.levels[lv]
            lanes = 2 ** L.lanes
            rows = p.witness(f"lig.rows.{lv}", "F", (R, L.q, lanes))
            sibs = p.witness(f"lig.sibs.{lv}", "byte", (R, L.q, L.d - L.cap, hl))
            against = cap if lv == 0 else fbytes(("rcap", lv - 1)).reshape(R, -1, hl)
            with p.component("merkle"):
                acc = p.sha(fr.algo, sha_iv(p, fr.algo), md_blocks(p, fr.algo, [p.bytes_(rows).reshape(R, L.q, -1)]))
                dirs = pub(f"lig.dir.{lv}", (R, L.q, L.d - L.cap), "bit")
                for tt in range(L.d - L.cap):
                    a_bits, s_bits = p.bits(acc), p.bits(sibs[:, :, tt])
                    dl = p.band(dirs[:, :, tt].reshape(R, L.q, 1), a_bits ^ s_bits)
                    left, right = p.bytes_(a_bits ^ dl), p.bytes_(s_bits ^ dl)
                    acc = p.sha(fr.algo, sha_iv(p, fr.algo), md_blocks(p, fr.algo, [left, right]))
                sel = pub(f"lig.sel.{lv}", (R, L.q, 2 ** L.cap), "bit")
                chosen = p.xorsum(p.band(sel.reshape(R, L.q, 2 ** L.cap, 1), p.bits(against).reshape(R, 1, 2 ** L.cap, 8 * hl)), 2)
                p.assert_eq(chosen, p.bits(acc), f"merkle: a level-{lv} query does not open to its cap")
            x = lin(p, pub(f"lig.ea.{lv}", (R, L.q, 1)), rows, [1])
            es.append(p.xorsum(kfmul(p, pub(f"lig.ef.{lv}", (R, lanes, 2)), x), 1).reshape(R, 1, 2))
        e = p.concat(es, axis=1)
        msg_idx = np.stack([np.stack([fo[("lig_msg", i, h)] for h in (0, 1)], axis=1) for i in range(nmsg)], axis=1)
        msgs = p.field(p.gather(m, msg_idx).reshape(R, nmsg, 2, 2, 16))
        ood = ffields("lig_ood", nood)
        yr = ffields("lig_yr", 2 ** (sch.ext + 1)).reshape(R, 2 ** sch.ext, 2)
        s_final = (kfmul(p, pub("lig.kT", (R, 2)), T)
                   ^ p.xorsum(p.xorsum(kmul(p, pub("lig.kmsg", (R, nmsg, 2, 2)), msgs), 2), 1)
                   ^ p.xorsum(kfmul(p, pub("lig.kood", (R, nood, 2)), ood), 1)
                   ^ p.xorsum(kmul(p, pub("lig.ke", (R, nlev, 2)), e), 1))
        lhs = p.xorsum(kmul(p, pub("lig.res", (R, 2 ** sch.ext, 2)), yr), 1)
        p.assert_eq(lhs, s_final, "ligerito: final check failed")
    return p


def _private_fold(p: Program, fr: Frame, enc: dict, zp: Reg, pub) -> Reg:
    """C's part of Σ_c comb[c]·eqT(c>>6)·zp[c&63] (PROTOCOL.md §10): the unit's entries, its constant and its leaf wiring."""
    B = fr.B
    U = 1 << B.unit_log
    with p.component("tables"):
        cols = np.arange(U)
        vloc = p.fmul(pub("lc.eqTlo", (R, U >> 6))[:, cols >> 6], zp[:, cols & 63])
        v_tab = p.bits(transpose(p, vloc, (1, 0))).reshape(U, R * 128)
        e_tab = p.bits(transpose(p, pub("lc.e0", (R, U + 1)), (1, 0))).reshape(U + 1, R * 128)
    cp = enc["enc.cp"]
    qa, qb = B.qa_max, B.qb_max
    with p.component("lookup-e"):
        cp14 = p.concat([cp, p.const("bit", np.zeros(1, np.uint8))]).reshape(1, 14)
        he = memcheck.lookup(p, "mc.E", e_tab, p.concat([enc["enc.ra"], enc["enc.rb"], cp14]))
    with p.component("lookup-v"):
        hv = memcheck.lookup(p, "mc.V", v_tab, p.concat([enc["enc.ca"], enc["enc.cb"]]))
    with p.component("products"):
        he = p.field(he.reshape(qa + qb + 1, R, 128))
        hv = p.field(hv.reshape(qa + qb, R, 128))
        prod = p.fmul(he[:qa + qb], hv)
        fold = p.fmul(pub("lc.alpha", (R,)), p.xorsum(prod[:qa], 0)) ^ p.xorsum(prod[qa:], 0)
        term = p.fmul(pub("lc.su", (R,)), fold)
        term = term ^ p.fmul(p.fmul(pub("lc.kcp", (R,)), he[qa + qb]), zp[:, fr.pin & 63])
    with p.component("lookup-leaves"):
        G, nin = B.g, B.n_in_words
        src = fr.leaf_src.reshape(-1)
        wmsg = p.fmul(pub("lc.eqTmsg", (R, src.size)), zp[:, src & 63])
        m_tab = p.bits(transpose(p, wmsg, (1, 0))).reshape(src.size, R * 128)
        lin_bits = enc["enc.lin"]                                               # (upv, lpu, 7)
        lpu = B.leaves_per_unit_in
        q_t = p.const("bit", np.broadcast_to(const_bits(np.arange(16), 4), (G, B.upv, lpu, 16, 4)).copy())
        q_leaf = p.take(lin_bits, (np.zeros((G, 1, 1, 16), dtype=np.int64) + np.arange(B.upv)[None, :, None, None],
                                    np.arange(lpu)[None, None, :, None] + np.zeros((G, B.upv, 1, 16), dtype=np.int64)))
        q_g = p.const("bit", np.broadcast_to(const_bits(np.arange(G), max(1, (G - 1).bit_length()))[:, None, None, None, :],
                                             (G, B.upv, lpu, 16, max(1, (G - 1).bit_length()))).copy())
        q = p.concat([q_t, q_leaf, q_g], axis=4).reshape(G * B.upv * lpu * 16, -1)
        hm_ = memcheck.lookup(p, "mc.M", m_tab, q)
        hm_ = transpose(p, p.field(hm_.reshape(-1, R, 128)), (1, 0))
        term = term ^ lin(p, pub("lc.kleaf", (R, hm_.shape[1])), hm_, [1])
    return term


def _fv3_tree(p: Program, fr: Frame, dom: bytes, leaves: Reg) -> Reg:
    """A frame-v3 root over leaf digests (n, 32): pad leaves are constants, nodes are hashed in V[B]."""
    n = leaves.shape[0]
    width = 1 << max(0, (n - 1).bit_length())
    if width > n:
        pads = [gadgets.sha_native("sha256", np.frombuffer(gadgets.iv_bytes("sha256"), np.uint8),
                                   np.frombuffer(_padded(fv3_head(b"pad", [dom, _uint(r)])), np.uint8).reshape(1, -1, 64))
                for r in range(n, width)]
        leaves = p.concat([leaves, p.const("byte", np.stack(pads))])
    level, depth = leaves, 0
    while level.shape[0] > 1:
        k = level.shape[0] // 2
        heads = [fv3_head(b"node", [dom, _uint(depth), _uint(i)]) for i in range(k)]
        groups: dict[int, list[int]] = {}
        for i, h in enumerate(heads):
            groups.setdefault(len(h), []).append(i)
        out = [None] * k
        for hl_, idx in groups.items():
            idx = np.array(idx)
            head = p.const("byte", np.stack([np.frombuffer(heads[i], np.uint8) for i in idx]))
            l8 = p.const("byte", np.broadcast_to(np.frombuffer((32).to_bytes(8, "big"), np.uint8), (len(idx), 8)).copy())
            blocks = md_blocks(p, "sha256", [head, l8, level[2 * idx], l8, level[2 * idx + 1]])
            d = p.sha("sha256", sha_iv(p, "sha256"), blocks)
            for t, i in enumerate(idx):
                out[i] = (d, t)
        level = p.concat([d[t].reshape(1, 32) for d, t in out]) if len({id(d) for d, _ in out}) > 1 else out[0][0]
        depth += 1
    return level.reshape(32)


def _padded(msg: bytes) -> bytes:
    return msg + gadgets.md_pad("sha256", len(msg))


def _region_values(p: Program, fr: Frame, enc: dict, pub, D: Reg) -> Reg:
    """The region claims' values (R, N - 2), from the private data, and the data bound to the roots D."""
    B = fr.B
    dig = p.witness("data.dig", "byte", (B.n, len(fr.ports), 32))
    out = p.witness("data.out", "bit", (B.n, B.n_out_words, 16))
    with p.component("binding"):
        for pi, name in enumerate(fr.ports):
            dom, schema = fr.fv3_domains[name], fr.fv3_schemas[name]
            heads = np.stack([np.frombuffer(fv3_head(b"leaf", [dom, _uint(i), _uint(i), schema]) + (32).to_bytes(8, "big"),
                                            np.uint8) for i in range(B.n)])
            leaves = p.sha("sha256", sha_iv(p, "sha256"), md_blocks(p, "sha256", [p.const("byte", heads), dig[:, pi]]))
            p.assert_eq(_fv3_tree(p, fr, dom, leaves), D[pi], f"the row digests of {name} do not hash to its root")
        dom = fr.fv3_domains[fr.out_port]
        be16 = p.bytes_(p.take(out.reshape(-1, 16), (slice(None), np.r_[8:16, 0:8])))   # (n*words, 2)
        nleaf = B.n * B.n_out_words
        leaf_regs = []
        for lo, hi in ((0, min(nleaf, 256)), (256, nleaf)):
            if lo >= hi:
                continue
            heads = np.stack([np.frombuffer(fv3_head(b"leaf", [dom, _uint(r), _uint(r), fr.fv3_schemas[fr.out_port]])
                                            + (2).to_bytes(8, "big"), np.uint8) for r in range(lo, hi)])
            leaf_regs.append(p.sha("sha256", sha_iv(p, "sha256"),
                                   md_blocks(p, "sha256", [p.const("byte", heads), be16[lo:hi]])))
        leaves = p.concat(leaf_regs) if len(leaf_regs) > 1 else leaf_regs[0]
        p.assert_eq(_fv3_tree(p, fr, dom, leaves), D[len(fr.ports)], "the outputs do not hash to their root")
    with p.component("regions"):
        const = pub("reg.const", (R, 2 * len(fr.regions)))
        cols = []
        dbits = p.bits(dig)                                                       # (n, P, 256)
        with p.component("lookup-out"):
            nslots = B.n + 1
            table = p.concat([out.reshape(-1, 16), enc["enc.dummy"]])            # (nslots * words, 16)
            G, lpo = B.g, B.leaves_per_unit_out
            blocks = 1 << B.nbl
            slot_bits = max(1, (nslots - 1).bit_length())
            wbits = max(1, (B.n_out_words - 1).bit_length())
            q_word = p.take(enc["enc.lout"], (np.zeros((blocks, G, 1, 1), dtype=np.int64) + np.arange(B.upv)[None, None, :, None],
                                              np.arange(lpo)[None, None, None, :] + np.zeros((blocks, G, B.upv, 1), dtype=np.int64)))
            q_slot = p.const("bit", np.broadcast_to(const_bits(fr.inst_slots, slot_bits)[:, :, None, None, :],
                                                    (blocks, G, B.upv, lpo, slot_bits)).copy())
            q = p.concat([q_word, q_slot], axis=4).reshape(-1, wbits + slot_bits)
            outv = memcheck.lookup(p, "mc.O", table, q)                          # (blocks*G*upv*lpo, 16)
        for ri, reg in enumerate(fr.regions):
            for pt in range(2):
                col = const[:, 2 * ri + pt]
                if reg.kind == "digest":
                    coef = pub(f"reg.dcoef.{ri}.{pt}", (R, B.n, 256))
                    col = col ^ p.xorsum(p.xorsum(p.fscale(dbits[:, reg.port], coef), 2), 1)
                elif reg.kind == "out":
                    coef = pub(f"reg.ocoef.{pt}", (R, outv.shape[0], 16))
                    col = col ^ p.xorsum(p.xorsum(p.fscale(outv, coef), 2), 1)
                cols.append(col.reshape(R, 1))
        return p.concat(cols, axis=1)
