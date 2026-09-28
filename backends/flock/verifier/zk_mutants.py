"""Tampered `flock-circuit --zk` sessions for the agreement check (`agree.py --zk`): one edit of an honest session each.

    python zk_mutants.py SESSION_DIR OUT_DIR [--table T]    # SESSION_DIR: session.json and <t>.rep<r>.bin for every table

A proof edit reaches the record as a prover that sent the edited proof would have made it: every retained round message
(`rounds[k].msg`) of that rep holding the old bytes gets the new ones, with its `msg_sha512` rebuilt, and the rep's
`proof_sha512` follows the file. The coins stay: the coin tree (live PROTOCOL.md §9) commits them at `Hello`, before any
message, so only the check the edit breaks can reject it. A value no round absorbs (`τ`'s opening, the opened columns,
their salts and paths, which follow the last coin) changes the file alone. Record edits change one field of
`session.json`. The edits are table `T`'s (the first by default); a session of several tables adds its first two tables'
rep-0 proofs swapped.

Writes OUT_DIR/<name>/ per case and OUT_DIR/zk_mutants.json, the `--expect` form of `agree.py`: the verdict both verifiers
must reach, the Lean verifier's reason prefix, and `upstream_accepted` where upstream's offline replay, which takes the
record's coins without re-deriving them from the coin tree (verifier PROTOCOL.md §7.4), accepts what the Lean verifier
refuses.
"""
import hashlib, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redigest as rd  # noqa: E402

ACCEPT = {"accepted": True, "why": ""}


def reject(why, upstream_accepted=None):
    e = {"accepted": False, "why": why}
    if upstream_accepted is not None:
        e["upstream_accepted"] = upstream_accepted
    return e


def flip(key, i=0, byte=0):
    """One bit of the `i`-th `key` value of the proof."""
    def edit(b, off):
        b[off[key][i] + byte] ^= 1
    return edit


def row_lane(lane, q=0, byte=0):
    """One bit of level 0's opened row `q` at lane `lane` (each row is `2^lanes + 4` lanes wide)."""
    def edit(b, off):
        width = len(off["lig.l0.rows"]) // len(off["lig.l0.salt"])
        b[off["lig.l0.rows"][q * width + lane] + byte] ^= 1
    return edit


def without_level0(b, off):
    """`zk_level0: None`: the hiding messages dropped."""
    o = off["zk.tag"][0]
    b[o:off["in.root"][0] - 8] = b"\x00"


def ybar_short(b, off):
    """One element of `ybar` dropped, its length one less."""
    o, first = off["zk.ybar"][-1], off["zk.ybar"][0]
    n = int.from_bytes(b[first - 8:first], "little")
    b[first - 8:first] = (n - 1).to_bytes(8, "little")
    del b[o:o + 32]


def trailing(b, off):
    """Bytes after the inner proof (ignored, as after §8's pair)."""
    b += b"\x00" * 16


TAU = reject("zk inner: τ does not open its commitment")

#: (name, rep, edit, expect): proof edits, the retained rounds rebuilt. An edit of claim 0's `s_hat_v` changes a value the
#: opening absorbs later, which no single replacement carries, so the replay rejects it (R2).
PROOF = [
    ("zk-zc-final-a", 0, flip("zc.fa"), reject("zk inner: the batched constraint fails")),
    ("zk-zc-round1-c", 0, flip("zc.c", 3), reject("zk inner: the batched constraint fails")),
    ("zk-zc-round", 0, flip("zc.g1", 2), reject("zk inner: the batched constraint fails")),
    ("zk-lc-round", 0, flip("lc.q1", 1), reject("zk inner: the batched constraint fails")),
    ("zk-lc-z-partial", 0, flip("lc.zp", 3), reject("zk inner: the batched constraint fails")),
    ("zk-rs-ab-claim", 0, flip("rs.0", 5), reject("R2: round ")),
    ("zk-rs-region-claim", 0, flip("rs.2", 5), reject("opening: ring-switch claim 2 mismatch")),
    ("zk-lig-lane-message", 0, flip("lig.msg", 1), reject("zk inner: the batched constraint fails")),
    ("zk-lig-message-after-level0", 0, flip("lig.msg", 4 * 20 + 1), reject("ligerito: final check failed")),
    ("zk-lig-ood-level0", 0, flip("lig.ood", 0), reject("zk inner: the batched constraint fails")),
    ("zk-lig-ood-level1", 0, flip("lig.ood", 1), reject("ligerito: final check failed")),
    ("zk-lig-yr", 0, flip("lig.yr", 1), reject("ligerito: final check failed")),
    ("zk-lig-row-lane", 0, row_lane(7), reject("merkle: query")),
    ("zk-lig-row-extra-lane", 0, row_lane(65), reject("merkle: query")),
    ("zk-lig-row-last-extra-lane", 0, row_lane(67, q=3), reject("merkle: query")),
    ("zk-l0-e0", 0, flip("zk.e0"), reject("zk inner: the batched constraint fails")),
    ("zk-l0-e1", 0, flip("zk.e1", byte=17), reject("zk inner: the batched constraint fails")),
    ("zk-l0-blinded-claim", 0, flip("zk.tp"), reject("ligerito: final check failed")),
    ("zk-l0-ybar", 0, flip("zk.ybar", 5), reject("ligerito: final check failed")),
    ("zk-l0-missing", 0, without_level0, reject("zk: level 0's hiding messages are missing")),
    ("zk-l0-ybar-short", 0, ybar_short, reject("ligerito: level 0's hiding messages have the wrong shape")),
    ("zk-pads-root", 0, flip("in.root", byte=3), reject("zk inner: column")),
    ("zk-triple-rho", 0, flip("in.rho"), reject("zk inner: the batched constraint fails")),
    ("zk-triple-sigma", 0, flip("in.sigma"), reject("zk inner: the batched constraint fails")),
    ("zk-tau-commitment", 0, flip("in.tauc", byte=9), TAU),
    ("zk-tau-salt", 0, flip("in.taus", byte=100), TAU),
    ("zk-tau", 0, flip("in.tau"), TAU),
    ("zk-inner-y", 0, flip("in.y", 11), reject("zk inner: the batched constraint fails")),
    ("zk-inner-opened", 0, flip("in.open", 7), reject("zk inner: column")),
    ("zk-inner-salt", 0, flip("in.salt", 2, 40), reject("zk inner: column")),
    ("zk-inner-path", 0, flip("in.path", 5, 1), reject("zk inner: column")),
    ("zk-trailing-bytes", 0, trailing, ACCEPT),
    ("zk-rep1-tau", 1, flip("in.tau"), TAU),
    ("zk-rep1-l0-e0", 1, flip("zk.e0"), reject("zk inner: the batched constraint fails")),
    ("zk-rep1-lane-message", 1, flip("lig.msg", 2), reject("zk inner: the batched constraint fails")),
]


def rep_stream(rec, table, rep):
    return next(s for s in rec["streams"] if s["stream"] == f"{table}/rep{rep}")


def carry(rec, table, rep, old, new):
    """Every retained round of `table`'s rep `rep` holding `old` gets `new`, its digest rebuilt; the rounds changed."""
    changed = []
    if old == new:
        return changed
    for r in rep_stream(rec, table, rep)["rounds"]:
        m = bytes.fromhex(r["msg"])
        if old in m:
            m = m.replace(old, new)
            r["msg"], r["msg_sha512"], r["msg_len"] = m.hex(), hashlib.sha512(m).hexdigest(), len(m)
            changed.append(r["k"])
    return changed


def width(key, b, o):
    """Bytes of the `key` value at offset `o` of proof `b` (`redigest.layout`'s keys, SHA-512 digests and hm96 salts)."""
    if key in ("in.root", "in.tauc", "in.taus"):
        return int.from_bytes(b[o - 8:o], "little")
    if key.startswith("zk."):
        return 1 if key == "zk.tag" else 32
    if key.endswith(".salt"):
        return 192
    if key in ("cap", "lig.icap", "in.path") or key.endswith(".sib") or key.startswith("lig.rcap"):
        return 64
    if key.endswith(("nonce", ".qn", ".fold", ".cbn", ".consn")):
        return 8
    return 16


def edited_values(before, after, off):
    """(key, old, new) bytes of every proof value an edit that kept the file's length changed."""
    if len(before) != len(after):
        return []
    out = []
    for key, offs in off.items():
        for o in offs:
            n = width(key, before, o)
            if before[o:o + n] != after[o:o + n]:
                out.append((key, bytes(before[o:o + n]), bytes(after[o:o + n])))
    return out


def write(out, name, rec, src, files):
    """Case `name`: `rec`, every proof file of `src`, and `files` (`{file name: bytes}`) in their place."""
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.json").write_text(json.dumps(rec))
    for f in src.glob("*.rep*.bin"):
        if f.name not in files:
            shutil.copy(f, d / f.name)
    for f, b in files.items():
        (d / f).write_bytes(b)


def set_proof_digest(rec, stream, b):
    for p in rec["proof_sha512"]:
        if p["stream"] == stream:
            p["sha512"], p["bytes"] = hashlib.sha512(b).hexdigest(), len(b)


def flip_hex(h, nibble=0):
    return h[:nibble] + "%x" % (int(h[nibble], 16) ^ 1) + h[nibble + 1:]


def main(src, out, table=None):
    src, out = Path(src), Path(out)
    base = json.loads((src / "session.json").read_text())
    tables = [t["name"] for t in base["config"]["tables"]]
    table = table or tables[0]
    proof = lambda rep: f"{table}.rep{rep}.bin"
    manifest = []
    for name, rep, edit, expect in PROOF:
        rec = json.loads(json.dumps(base))
        before = (src / proof(rep)).read_bytes()
        off = rd.layout(before, zk=True)
        b = bytearray(before)
        edit(b, off)
        rounds = []
        for _key, old, new in edited_values(before, b, off):
            rounds += carry(rec, table, rep, old, new)
        set_proof_digest(rec, f"{table}/rep{rep}", b)
        write(out, name, rec, src, {proof(rep): bytes(b)})
        manifest.append({"name": name, "table": table, "rep": rep, "rounds": sorted(set(rounds)), "expect": expect})

    def record_case(name, mutate, expect, files=None):
        rec = json.loads(json.dumps(base))
        mutate(rec)
        write(out, name, rec, src, files or {})
        manifest.append({"name": name, "table": table, "rep": None, "expect": expect})

    r0 = lambda rec: rep_stream(rec, table, 0)["rounds"]

    def coin(k):
        return lambda rec: r0(rec)[k]["coins"].__setitem__(0, flip_hex(r0(rec)[k]["coins"][0], 31))

    def coin_tree(field, value):
        return lambda rec: rec["coin_tree"].__setitem__(field, value(rec["coin_tree"][field]))

    # round 0's coins are r_skip, drawn and never read (§19 A1)
    record_case("zk-coin-unused", coin(0), ACCEPT)
    record_case("zk-coin-round1", coin(1), reject("ligerito: final check failed"))
    record_case("zk-coin-last-round", coin(-1), reject("zk inner: column"))
    # the verifier checks the record's coin tree, not its openings: the commitment protects the prover (verifier PROTOCOL.md
    # §7.4); upstream's offline replay ignores `coin_tree` (divergence D6)
    record_case("zk-coin-tree-key", coin_tree("key", lambda h: flip_hex(h, 5)), ACCEPT)
    record_case("zk-coin-tree-root", coin_tree("root", lambda h: flip_hex(h, 5)), ACCEPT)
    record_case("zk-coin-tree-key-bit-2047", coin_tree("key", lambda h: h[:-2] + "%02x" % (int(h[-2:], 16) | 0x80)),
                reject("S2: coin_tree.key", upstream_accepted=True))
    record_case("zk-coin-tree-nonce", coin_tree("nonce", lambda h: flip_hex(h, 5)),
                reject("S2: coin_tree.nonce is not Hello's nonce", upstream_accepted=True))
    record_case("zk-coin-tree-missing", lambda rec: rec.__setitem__("coin_tree", None),
                reject("S2: coin_tree is not {key, nonce, root, spec}", upstream_accepted=True))

    def retained(digest):
        def edit(rec):
            r = r0(rec)[0]
            m = bytearray.fromhex(r["msg"])
            m[-17] ^= 1
            r["msg"] = m.hex()
            if digest:
                r["msg_sha512"] = hashlib.sha512(m).hexdigest()
        return edit
    record_case("zk-retained-not-digest", retained(False), reject("S12:"))
    record_case("zk-retained-edited", retained(True), reject("R2: round 0"))

    def swap(a, b):
        """Proof files `a` and `b` (`(table, rep)`) exchanged, with their digests."""
        fa, fb = (src / f"{a[0]}.rep{a[1]}.bin").read_bytes(), (src / f"{b[0]}.rep{b[1]}.bin").read_bytes()

        def edit(rec):
            set_proof_digest(rec, f"{a[0]}/rep{a[1]}", fb)
            set_proof_digest(rec, f"{b[0]}/rep{b[1]}", fa)
        return edit, {f"{a[0]}.rep{a[1]}.bin": fb, f"{b[0]}.rep{b[1]}.bin": fa}

    edit, files = swap((table, 0), (table, 1))
    record_case("zk-reps-swapped", edit, reject("R2: round 0"), files)
    if len(tables) > 1:
        other = next(t for t in tables if t != table)
        edit, files = swap((table, 0), (other, 0))
        record_case("zk-tables-swapped", edit, reject("R2: round 0"), files)
    record_case("zk-link-point", lambda rec: rec["link"]["points"][0].__setitem__(3, flip_hex(rec["link"]["points"][0][3], 31)),
                reject("S14: link_sha512"))
    (out / "zk_mutants.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"mutants": len(manifest), "out": str(out)}))


if __name__ == "__main__":
    args = sys.argv[1:]
    table = args[args.index("--table") + 1] if "--table" in args else None
    main(args[0], args[1], table)
