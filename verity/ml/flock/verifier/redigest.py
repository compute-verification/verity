"""Proof mutants with a consistent transcript: the negatives that reach the verifier's equations.

A recorded session binds every proof message to the coins that followed it (R2), so an edited proof is rejected by the
transcript before any equation is read. This tool edits one value of one rep's proof, then rebuilds that rep's round
digests (`msg_sha256`, `msg_len`) and its `proof_sha256` from the edited proof with `transcript_check.py`'s spec
transcript, keeping every coin. The record is then what a prover that sent the edited proof would have produced
against the same coins, and only the check the edit breaks can reject it.

    python redigest.py SESSION_DIR OUT_DIR       # SESSION_DIR: session.json, <t>.rep0.bin, <t>.rep1.bin
    python redigest.py --retain SESSION_DIR OUT_DIR [--upstream-checks retained_bytes]

Writes OUT_DIR/<name>/ per mutant and OUT_DIR/mutants.json: name, rep, edit, and the verdict both verifiers must reach
(`expect`: accept or reject, and the Lean verifier's reason prefix). `--retain` writes the session with every round's
framed bytes retained (`rounds[k].msg`, hex; the challenger's retained round bytes) and two negatives: rep 0's first
round bytes edited with its digest rebuilt (the replay's exact comparison rejects), and edited without it (S12 rejects;
upstream accepts it, divergence D4, unless `--upstream-checks retained_bytes`: PR #83 from 4bceebef checks the bytes
against the digest).
"""
import hashlib, json, shutil, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcript_check as tc  # noqa: E402


def layout(b, zk=False):
    """Byte offsets of every field element, hash and nonce of a proof (PROTOCOL.md §8.2), by field; `zk`: a `--zk` proof's
    (live PROTOCOL.md), whose Ligerito proof ends with level 0's hiding messages and is followed by the inner proof."""
    r, off, sb = tc.R(b, tc.digest_width(b)), {}, tc.salt_bytes(b)

    def at(key, n):
        off.setdefault(key, []).append(r.i)
        r.take(n)

    def f(key): at(key, 16)
    def h(key): at(key, r.hlen)
    def u(key): at(key, 8)

    def vec(fn):
        for _ in range(r.u64()):
            fn()

    vec(lambda: h("cap"))
    r.u64(), r.u64(), r.u64(), r.u32()
    if r.u8():
        r.u64()
    r.u32()
    vec(lambda: f("zc.ab")); vec(lambda: f("zc.c"))
    vec(lambda: (f("zc.g1"), f("zc.ginf")))
    f("zc.fa"); f("zc.fb"); f("zc.fc"); vec(lambda: u("zc.nonce"))
    vec(lambda: (f("lc.q1"), f("lc.qinf"))); vec(lambda: f("lc.zp")); vec(lambda: (f("lc.me"), f("lc.me")))
    vec(lambda: u("lc.nonce"))
    for k in range(r.u64()):
        vec(lambda: f(f"rs.{k}")); u(f"rs.{k}.nonce")
    vec(lambda: u("batching.nonce"))
    vec(lambda: h("lig.icap"))

    def level(name):
        vec(lambda: vec(lambda: f(f"{name}.rows")))
        vec(lambda: h(f"{name}.sib"))
        if sb:
            vec(lambda: at(f"{name}.salt", sb))

    level("lig.l0")
    for i in range(r.u64()):
        vec(lambda: h(f"lig.rcap{i}"))
    for i in range(r.u64()):
        level(f"lig.l{i + 1}")
    vec(lambda: f("lig.yr"))
    level("lig.final")
    vec(lambda: (f("lig.sc"), f("lig.sc")))
    vec(lambda: [f("lig.msg") for _ in range(4)])
    for k in ("lig.qn", "lig.ood", "lig.fold", "lig.cbn", "lig.consn"):
        vec(lambda: (f if k == "lig.ood" else u)(k))
    if zk:
        at("zk.tag", 1)
        for k in ("zk.e0", "zk.e1", "zk.tp"):
            at(k, 32)
        vec(lambda: at("zk.ybar", 32))
        at("in.root", r.u64())
        f("in.rho")
        f("in.sigma")
        at("in.tauc", r.u64())
        at("in.taus", r.u64())
        f("in.tau")
        vec(lambda: f("in.y"))
        vec(lambda: (f("in.open"), f("in.open")))
        vec(lambda: at("in.salt", 192))
        vec(lambda: vec(lambda: at("in.path", 64)))
    assert r.i == len(b), (r.i, len(b))
    return off


def get128(b, o):
    lo, hi = struct.unpack_from("<QQ", b, o)
    return (hi << 64) | lo


def put128(b, o, v):
    struct.pack_into("<QQ", b, o, v & ((1 << 64) - 1), v >> 64)


def eq_table(r):
    t = [1]
    for x in r:
        t = [tc.gmul(v, x ^ 1) for v in t] + [tc.gmul(v, x) for v in t]
    return t


def orthogonal(b, off, rec, k):
    """Change claim k's `s_hat` along a direction its claim weights annihilate (claim k a region claim whose point starts
    with the link point's first seven coordinates: region 0 at point 0), so only the opening can notice."""
    r0 = [tc.coin(h) for h in rec["link"]["points"][0]]
    skip, x0 = eq_table(r0[:6]), r0[6]
    w = [tc.gmul(skip[v % 64], x0 if v >= 64 else x0 ^ 1) for v in range(128)]
    o = off[f"rs.{k}"]
    put128(b, o[0], get128(b, o[0]) ^ w[1])
    put128(b, o[1], get128(b, o[1]) ^ w[0])


class Retain(tc.T):
    def squeeze(s, kind, n):
        s.pending += s.hdr(4, kind, n)
        if n == 0:
            return []
        c, s.pending = s.pending, b""
        rec = s.rounds[s.k]
        assert rec["msg_sha256"] == hashlib.sha256(c).hexdigest(), (s.k, "the spec transcript is not the record's")
        rec["msg"] = c.hex()
        s.k += 1
        return [tc.coin(h) for h in rec["coins"]]


def retain(src, out, checks=()):
    src, out = Path(src), Path(out)
    rec = json.loads((src / "session.json").read_text())
    table = rec["config"]["tables"][0]["name"]
    tc.T = Retain
    for rep in (0, 1):
        st = next(s for s in rec["streams"] if s["stream"] == f"{table}/rep{rep}")
        tc.replay(st["rounds"], st["label"].encode(), digest_of(rec), tc.decode((src / f"{table}.rep{rep}.bin").read_bytes()))

    def write(name, r):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "session.json").write_text(json.dumps(r))
        for rep in (0, 1):
            shutil.copy(src / f"{table}.rep{rep}.bin", d / f"{table}.rep{rep}.bin")

    write("retained", rec)
    first = lambda r: next(s for s in r["streams"] if s["stream"] == f"{table}/rep0")["rounds"][0]
    bad = json.loads(json.dumps(rec))
    b = bytearray.fromhex(first(bad)["msg"])
    b[-17] ^= 1
    first(bad)["msg"], first(bad)["msg_sha256"] = b.hex(), hashlib.sha256(b).hexdigest()
    write("retained-edited", bad)
    bad2 = json.loads(json.dumps(rec))
    first(bad2)["msg"] = b.hex()
    write("retained-not-digest", bad2)
    manifest = [{"name": "retained", "rep": 0, "expect": ACCEPT},
                {"name": "retained-edited", "rep": 0, "expect": reject("R2: round 0: the proof's messages are not the bytes")},
                {"name": "retained-not-digest", "rep": 0,
                 "expect": reject("S12:") if "retained_bytes" in checks else {**reject("S12:"), "upstream_accepted": True}}]
    (out / "retained.json").write_text(json.dumps(manifest, indent=1))
    print(f"retained-bytes sessions in {out}")


class Redigest(tc.T):
    def squeeze(s, kind, n):
        s.pending += s.hdr(4, kind, n)
        if n == 0:
            return []
        c, s.pending = s.pending, b""
        rec = s.rounds[s.k]
        rec["msg_sha256"], rec["msg_len"] = hashlib.sha256(c).hexdigest(), len(c)
        if "msg" in rec:
            rec["msg"] = c.hex()
        s.k += 1
        return [tc.coin(h) for h in rec["coins"]]


def flip(key, i=0, byte=0):
    return lambda b, off, rec: b.__setitem__(off[key][i] + byte, b[off[key][i] + byte] ^ 1)


def reject(why):
    return {"accepted": False, "why": why}


ACCEPT = {"accepted": True, "why": ""}

MUTANTS = [
    ("zc-final-a", 0, flip("zc.fa"), reject("zerocheck: SumcheckFinalFailed")),
    ("zc-final-c", 0, flip("zc.fc"), reject("zerocheck: CEvalMismatch")),
    ("zc-round1-c", 0, flip("zc.c", 3), reject("zerocheck: CEvalMismatch")),
    ("zc-round1-ab", 0, flip("zc.ab", 3), reject("zerocheck: SumcheckFinalFailed")),
    ("zc-residual-round", 0, flip("zc.g1", 2), reject("zerocheck: SumcheckFinalFailed")),
    ("lc-round", 0, flip("lc.q1", 1), reject("lincheck: ConsistencyFailed")),
    ("lc-z-partial", 0, flip("lc.zp", 3), reject("lincheck: ConsistencyFailed")),
    ("rs-ab-claim", 0, flip("rs.0", 5), reject("opening: ring-switch claim 0 mismatch")),
    ("rs-c-claim", 0, flip("rs.1", 5), reject("opening: ring-switch claim 1 mismatch")),
    ("rs-region-claim", 0, flip("rs.2", 5), reject("opening: ring-switch claim 2 mismatch")),
    ("rs-region-orthogonal", 0, lambda b, off, rec: orthogonal(b, off, rec, 2), reject("ligerito: final check failed")),
    ("rs-nonce", 0, flip("rs.3.nonce"), reject("ring switch grinding_nonce")),
    ("lig-yr", 0, flip("lig.yr", 1), reject("ligerito: final check failed")),
    ("lig-ood", 0, flip("lig.ood", 0), reject("ligerito: final check failed")),
    ("lig-message", 0, flip("lig.msg", 5), reject("ligerito: final check failed")),
    ("lig-row-l0", 0, flip("lig.l0.rows", 7), reject("merkle:")),
    ("lig-sibling-l0", 0, flip("lig.l0.sib", 3, 5), reject("merkle:")),
    ("lig-row-l1", 0, flip("lig.l1.rows", 7), reject("merkle:")),
    ("lig-row-final", 0, flip("lig.final.rows", 7), reject("merkle:")),
    ("lig-query-nonce", 0, flip("lig.qn", 0), ACCEPT),
    ("lig-claim-nonce", 0, flip("lig.cbn", 1), ACCEPT),
    ("rep1-zc-final-a", 1, flip("zc.fa"), reject("zerocheck: SumcheckFinalFailed")),
    ("rep1-lig-yr", 1, flip("lig.yr", 1), reject("ligerito: final check failed")),
]


def digest_of(rec):
    binding = rec["config"]["tables"][0]["binding"]
    return bytes(int(x) for x in binding.split("[", 1)[1].split("]")[0].split(","))


def main(src, out):
    src, out = Path(src), Path(out)
    base = json.loads((src / "session.json").read_text())
    table = base["config"]["tables"][0]["name"]
    tc.T = Redigest
    manifest = []
    for name, rep, edit, expect in MUTANTS:
        rec = json.loads((src / "session.json").read_text())
        b = bytearray((src / f"{table}.rep{rep}.bin").read_bytes())
        edit(b, layout(bytes(b)), rec)
        st = next(s for s in rec["streams"] if s["stream"] == f"{table}/rep{rep}")
        k, total, _, left = tc.replay(st["rounds"], st["label"].encode(), digest_of(rec), tc.decode(bytes(b)))
        assert (k, left) == (total, 0), (name, k, total, left)
        for p in rec["proof_sha256"]:
            if p["stream"] == st["stream"]:
                p["sha256"], p["bytes"] = hashlib.sha256(b).hexdigest(), len(b)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "session.json").write_text(json.dumps(rec))
        (d / f"{table}.rep{rep}.bin").write_bytes(b)
        shutil.copy(src / f"{table}.rep{1 - rep}.bin", d / f"{table}.rep{1 - rep}.bin")
        manifest.append({"name": name, "rep": rep, "expect": expect})
    (out / "mutants.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} mutants in {out}")


if __name__ == "__main__":
    if sys.argv[1] == "--retain":
        retain(*sys.argv[2:4], checks=sys.argv[5].split(",") if sys.argv[4:5] == ["--upstream-checks"] else [])
    else:
        main(*sys.argv[1:3])
