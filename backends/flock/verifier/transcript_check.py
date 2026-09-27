"""Check PROTOCOL.md's transcript against recorded Flock sessions.

Rebuilds every round of each rep's transcript from the spec alone (framing §6.1; order §9.1, §9.2, §10, §12.2, §13.3;
the batched target T from §12.3, with its own GF(2^128) arithmetic) and compares each round's SHA-256 and coin count
with the session record. It checks only the transcript and the proof layout (§8), not the verifier's equations.

    python transcript_check.py session.json rep0.proof rep1.proof

Vectors: vectors.json (fetch a session with `research data fetch <record> --path <path>`).
"""
import hashlib, json, math, struct, sys

class R:
    def __init__(s, b, hlen=32): s.b, s.i, s.hlen = b, 0, hlen
    def take(s, n):
        if s.i + n > len(s.b): raise ValueError("short")
        v = s.b[s.i:s.i + n]; s.i += n; return v
    def u8(s): return s.take(1)[0]
    def u32(s): return struct.unpack("<I", s.take(4))[0]
    def u64(s): return struct.unpack("<Q", s.take(8))[0]
    def f128(s): lo, hi = struct.unpack("<QQ", s.take(16)); return (hi << 64) | lo
    def vec(s, f, cap=1 << 26):
        n = s.u64()
        if n > cap: raise ValueError(f"length {n}")
        return [f() for _ in range(n)]
    def h32(s): return s.take(s.hlen)

MOD = (1 << 128) | 0x87


def gmul(a, b):
    r = 0
    while b:
        if b & 1: r ^= a
        b >>= 1; a <<= 1
        if a >> 128: a ^= MOD
    return r


def fbytes(v): return struct.pack("<QQ", v & ((1 << 64) - 1), v >> 64)


def coin(h): return (int(h[:16], 16) << 64) | int(h[16:], 16)


class T:
    def __init__(s, rounds, domain):
        s.rounds, s.k, s.fail = rounds, 0, None
        s.pending = s.hdr(1, 0, len(domain)) + s.pad(domain)
    @staticmethod
    def hdr(op, kind, n): return bytes([op, kind]) + b"\0" * 6 + struct.pack("<Q", n)
    @staticmethod
    def pad(b): return b + b"\0" * ((-len(b)) % 16)
    def label(s, b): s.pending += s.hdr(2, 0, len(b)) + s.pad(b)
    def obs(s, v): s.pending += s.hdr(3, 1, 1) + fbytes(v)
    def obs_slice(s, vs): s.pending += s.hdr(3, 2, len(vs)) + b"".join(fbytes(v) for v in vs)
    def obs_bytes(s, b): s.pending += s.hdr(5, 0, len(b)) + s.pad(b)
    def pow(s, nonce): s.obs_bytes(struct.pack("<Q", nonce))
    def squeeze(s, kind, n):
        s.pending += s.hdr(4, kind, n)
        if n == 0: return []
        c, s.pending = s.pending, b""
        rec = s.rounds[s.k] if s.k < len(s.rounds) else None
        if rec is None or rec["n"] != n or rec["msg_sha256"] != hashlib.sha256(c).hexdigest():
            if s.fail is None:
                s.fail = f"round {s.k}: n={n} len={len(c)} rec={None if rec is None else (rec['n'], rec['msg_len'])}"
        s.k += 1
        return [coin(h) for h in rec["coins"]] if rec else [0] * n
    def scalar(s): return s.squeeze(1, 1)[0]
    def vec(s, n): return s.squeeze(2, n)


HASH_KIND_WIDTH = {0: 32, 2: 64, 3: 64}  # PcsParams.merkle_hash: Sha256 = 0, Sha512 = 2, Hm96Sha512 = 3 (Blake3 = 1 is not a Merkle kind here)
SALT_BYTES = {3: 192}  # an hm96-sha512 tree's openings carry each opened row's salt (`opened_salts`)


def merkle_kind(b):
    """`PcsParams.merkle_hash` and the Merkle digest width, read off the commitment."""
    for h in (32, 64):
        try:
            r = R(b, h)
            r.vec(r.h32); r.u64(); r.u64(); r.u64(); r.u32(); t = r.u8(); r.u64() if t else None
            k = r.u32()
            if HASH_KIND_WIDTH.get(k) == h:
                return k, h
        except ValueError:
            pass
    raise ValueError("no Merkle digest width fits the commitment")


def digest_width(b):
    """The Merkle digest width, read off the commitment: the one whose `PcsParams.merkle_hash` names it."""
    return merkle_kind(b)[1]


def salt_bytes(b):
    """Bytes of each opened row's salt (0: unsalted leaves, no `opened_salts` field)."""
    return SALT_BYTES.get(merkle_kind(b)[0], 0)


def decode(b):
    r = R(b, digest_width(b))
    sb = salt_bytes(b)
    cap = r.vec(r.h32); m = r.u64(); r.u64(); k0 = r.u64(); r.u32(); t = r.u8(); r.u64() if t else None; r.u32()
    pair = lambda: (r.f128(), r.f128())
    zc = dict(ab=r.vec(r.f128), c=r.vec(r.f128), rounds=r.vec(pair), fa=r.f128(), fb=r.f128(), fc=r.f128(), nonces=r.vec(r.u64))
    lc = dict(rounds=r.vec(pair), zp=r.vec(r.f128), me=r.vec(pair), nonces=r.vec(r.u64))
    rs = r.vec(lambda: (r.vec(r.f128), r.u64())); bn = r.vec(r.u64)
    rec = lambda: (r.vec(lambda: r.vec(r.f128)), r.vec(r.h32), r.vec(lambda: r.take(sb)) if sb else None)
    lig = dict(icap=r.vec(r.h32), iproof=rec(), rcaps=r.vec(lambda: r.vec(r.h32)), rproofs=r.vec(rec))
    lig["yr"] = r.vec(r.f128); r.vec(lambda: r.vec(r.f128)); r.vec(r.h32)
    if sb:
        r.vec(lambda: r.take(sb))
    lig["sc"] = r.vec(pair)
    lig["sc256"] = r.vec(lambda: ((r.f128(), r.f128()), (r.f128(), r.f128())))
    for k in ("qn", "ood", "fold", "cbn", "consn"):
        lig[k] = r.vec(r.f128 if k == "ood" else r.u64)
    assert r.i == len(b)
    return dict(cap=cap, m=m, k0=k0, zc=zc, lc=lc, rs=rs, bn=bn, lig=lig)


Q = [218, 106, 71, 53, 43, 36, 32]


def replay(rounds, domain, digest, p):
    t = T(rounds, domain); m, k0 = p["m"], p["k0"]; zc, lc, lig = p["zc"], p["lc"], p["lig"]
    k_log = len(lc["rounds"]) + 6
    # §9.1 binding, §9.2 zerocheck
    t.label(b"flock-r1cs-v0"); t.obs_bytes(digest); t.obs_bytes(b"".join(p["cap"]))
    t.label(b"flock-zerocheck-v0"); t.vec(6); t.vec(m - 13)
    t.obs_slice(zc["ab"]); t.obs_slice(zc["c"]); t.scalar()
    for g1, ginf in zc["rounds"]:
        t.obs(g1); t.obs(ginf); t.scalar()
    t.obs(zc["fa"]); t.obs(zc["fb"])
    # §10 lincheck
    t.label(b"flock-lincheck-v0"); t.scalar(); t.scalar()
    for q1, qinf in lc["rounds"]:
        t.obs(q1); t.obs(qinf); t.scalar()
    t.obs_slice(lc["zp"]); t.scalar()
    # §12.2 opening
    t.label(b"flock-pcs-open-batch-v0"); betas = []
    for s_k, _nonce in p["rs"]:
        t.label(b"flock-ring-switch-v0"); t.obs_slice(s_k); rpp = t.vec(7)
        tk = [sum(((s_k[v] >> u) & 1) << v for v in range(128)) for u in range(128)]   # §12.3 column repack
        beta = 0
        for u in range(128):
            w = 1
            for i in range(7):
                w = gmul(w, rpp[i] if (u >> i) & 1 else rpp[i] ^ 1)
            beta ^= gmul(w, tk[u])
        betas.append(beta)
    gam = t.vec(len(p["rs"]))
    T_ = 0
    for g, b in zip(gam, betas): T_ ^= gmul(g, b)
    # §13.3 Ligerito
    n = m - 7; msgs = iter(lig["sc256"]); ood = iter(lig["ood"]); qn = iter(lig["qn"]); cbn = iter(lig["cbn"]); con = iter(lig["consn"])
    def msg():
        (a0, a1), (b0, b1) = next(msgs); t.obs_slice([a0, a1]); t.obs_slice([b0, b1])
    def ood_intro(dim):
        t.vec(dim); t.obs(next(ood)); msg(); t.pow(next(cbn)); t.scalar()
    def queries(level):
        t.pow(next(qn)); t.vec(Q[level]); t.pow(next(con)); t.vec(math.ceil(math.log2(Q[level])))
    t.label(b"flock-ligerito-basis-f256-split-v0"); t.obs(T_); t.obs_bytes(b"".join(lig["icap"]))
    t.vec(n); t.obs(next(ood)); t.pow(next(cbn)); t.scalar()
    msg()
    for _ in range(k0):
        t.vec(2); msg()
    split = n - k0 + 1; R_ = len(lig["rcaps"])
    t.obs_bytes(b"".join(lig["rcaps"][0]))
    ood_intro(split); ood_intro(split)
    queries(0); msg(); t.pow(next(cbn)); t.scalar()
    for i in range(R_):
        k = 4
        for _ in range(k):
            t.vec(2); msg()
        ext = split - k
        if i + 1 < R_:
            split = ext + 1
            t.obs_bytes(b"".join(lig["rcaps"][i + 1]))
            ood_intro(split); ood_intro(split)
            queries(i + 1); msg(); t.pow(next(cbn)); t.scalar()
        else:
            for w in lig["yr"]: t.obs(w)
            queries(R_); t.pow(next(cbn)); t.scalar()
    left = [x for x in (msgs, ood, qn, cbn, con) if next(x, None) is not None]
    return t.k, len(rounds), t.fail, len(left)


if __name__ == "__main__":
    sess, proofs = sys.argv[1], sys.argv[2:]
    rec = json.load(open(sess))
    binding = rec["config"]["tables"][0]["binding"]
    digest = bytes(int(x) for x in binding.split("[", 1)[1].split("]")[0].split(","))
    for rep, pf in enumerate(proofs):
        st = [s for s in rec["streams"] if s["stream"].endswith(f"/rep{rep}")][0]
        p = decode(open(pf, "rb").read())
        k, total, fail, left = replay(st["rounds"], st["label"].encode(), digest, p)
        print(f"{st['stream']}: m={p['m']} rounds rebuilt {k}/{total}, first mismatch: {fail}, unconsumed proof lists: {left}")
