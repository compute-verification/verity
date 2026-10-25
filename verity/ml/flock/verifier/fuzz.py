"""Fuzzing for the agreement check (PROTOCOL.md §17.1 item 2): random single mutations of an honest session.

    python fuzz.py SESSION_DIR OUT_DIR --seed S --n N [--upstream-checks link_sigma]

Each case edits one thing, chosen with `random.Random(seed)`:
- `proof-byte`: a bit of one rep's proof file, with its `proof_sha256` updated (so S17 passes) and nothing else. The
  transcript (R2) or the decoder must reject (past the pair in its statement's pinned form, §8.1 refuses every byte: D17).
- `proof-value`: one field element or hash of one rep's proof, with that rep's round digests rebuilt (`redigest.py`). It must
  be rejected; a proof-of-work nonce by the Lean verifier's decoder only, which refuses every nonce but `0` (R4), divergence
  D21.
- `coin`: one bit of one recorded coin. No expectation: an unused coin (§19 A1) or the high bits of a query coin (§13.5)
  leave the verdict unchanged; the verifiers must only agree.
- `digest`: one bit of one round's `msg_sha256`. Must be rejected (R2).
- `record`: one bit of `root_b`, `sigma`, a link point, `y`, `link_sha256` or a proof's `sha256`. Must be rejected (a
  flipped `sigma` by the Lean verifier only, divergence D3, unless `--upstream-checks link_sigma`: PR #83 from b3baabd8).

Writes OUT_DIR/<case>/ and OUT_DIR/fuzz.json (the `--expect` form of `agree.py`; cases without an expectation are omitted).
Use a fixed seed per CI run and a rolling seed for nightly runs.
"""
import argparse, hashlib, json, random, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redigest as rd  # noqa: E402
import transcript_check as tc  # noqa: E402

NONCES = ("zc.nonce", "lc.nonce", "batching.nonce", "lig.qn", "lig.fold", "lig.cbn", "lig.consn")


def flip_hex(h, bit):
    b = bytearray.fromhex(h)
    b[(bit // 8) % len(b)] ^= 1 << (bit % 8)
    return b.hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--upstream-checks", default="", help="comma-separated checks upstream's from_record makes: link_sigma")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    base = json.loads((a.src / "session.json").read_text())
    table = base["config"]["tables"][0]["name"]
    proofs = [(a.src / f"{table}.rep{r}.bin").read_bytes() for r in (0, 1)]
    layouts = [rd.layout(p) for p in proofs]
    expect = []
    for k in range(a.n):
        kind = rng.choice(["proof-byte", "proof-value", "coin", "digest", "record"])
        rec = json.loads(json.dumps(base))
        pf = [bytearray(p) for p in proofs]
        rep = rng.randrange(2)
        st = next(s for s in rec["streams"] if s["stream"] == f"{table}/rep{rep}")
        want, what = {"accepted": False, "why": ""}, ""
        if kind == "proof-byte":
            i = rng.randrange(len(pf[rep]))
            pf[rep][i] ^= 1 << rng.randrange(8)
            what = f"rep{rep} byte {i}"
        elif kind == "proof-value":
            field = rng.choice(sorted(layouts[rep]))
            o = rng.choice(layouts[rep][field])
            pf[rep][o + rng.randrange(8)] ^= 1 << rng.randrange(8)
            tc.T = rd.Redigest
            tc.replay(st["rounds"], st["label"].encode(), rd.digest_of(rec), tc.decode(bytes(pf[rep])))
            what = f"rep{rep} {field} @ {o}"
            if field in NONCES:
                want["upstream_accepted"] = True
        elif kind == "coin":
            r = rng.randrange(len(st["rounds"]))
            c = rng.randrange(len(st["rounds"][r]["coins"]))
            bit = rng.randrange(128)
            st["rounds"][r]["coins"][c] = flip_hex(st["rounds"][r]["coins"][c], bit)
            what, want = f"rep{rep} round {r} coin {c} bit {bit}", None
        elif kind == "digest":
            r = rng.randrange(len(st["rounds"]))
            st["rounds"][r]["msg_sha256"] = flip_hex(st["rounds"][r]["msg_sha256"], rng.randrange(256))
            what = f"rep{rep} round {r} msg_sha256"
        else:
            l = rec["link"]
            which = rng.choice(["root_b", "sigma", "point", "y", "link_sha256", "proof_sha256"])
            if which == "root_b":
                l["root_b"][table] = flip_hex(l["root_b"][table], rng.randrange(256))
            elif which == "sigma":
                l["sigma"] = flip_hex(l["sigma"], rng.randrange(256))
                # D3 before PR #83 b3baabd8: upstream's from_record ignored link.sigma (Σ is bound by Hello and root_F)
                want = {"accepted": False, "why": "S4"}
                if "link_sigma" not in a.upstream_checks.split(","):
                    want["upstream_accepted"] = True
            elif which == "point":
                p = rng.randrange(len(l["points"]))
                j = rng.randrange(len(l["points"][p]))
                l["points"][p][j] = flip_hex(l["points"][p][j], rng.randrange(128))
            elif which == "y":
                l["y"][0] = flip_hex(l["y"][0], rng.randrange(128))
            elif which == "link_sha256":
                rec["link_sha256"] = flip_hex(rec["link_sha256"], rng.randrange(256))
            else:
                rec["proof_sha256"][rep]["sha256"] = flip_hex(rec["proof_sha256"][rep]["sha256"], rng.randrange(256))
            what = f"record {which}"
        if kind in ("proof-byte", "proof-value"):
            for p in rec["proof_sha256"]:
                if p["stream"] == f"{table}/rep{rep}":
                    p["sha256"], p["bytes"] = hashlib.sha256(pf[rep]).hexdigest(), len(pf[rep])
        name = f"fuzz-{a.seed}-{k:03d}-{kind}"
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "session.json").write_text(rd.session_text(rec))
        for r in (0, 1):
            (d / f"{table}.rep{r}.bin").write_bytes(pf[r])
        (d / "what.txt").write_text(what + "\n")
        if want is not None:
            expect.append({"name": name, "expect": want, "what": what})
    (a.out / "fuzz.json").write_text(json.dumps(expect, indent=1))
    print(f"{a.n} cases in {a.out}, {len(expect)} with an expectation")


if __name__ == "__main__":
    main()
