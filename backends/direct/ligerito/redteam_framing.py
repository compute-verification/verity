"""red-team-ligerito-3: LGTO0001 framing-JSON malleability fixtures.

The framing blob (``{"sib_len": [...], "final_len": int[, "t_pad": int]}``) is not absorbed and both readers parse it with a
lenient JSON parser (``proof.py`` ``json.loads``; ``lgto.rs`` ``json::parse`` without the params blob's canonical check), so
re-encoding it yields a different, still-valid proof (Fiat-Shamir included).  Writes, from a dump's ``honest_fiat-shamir``
proof, four re-encodings (space after a colon, an unknown key, reversed key order, indented) and, for a non-ZK proof, an
explicit ``"t_pad":0`` as ``expect: reject`` entries: each parses to the honest proof, so each must be rejected (the framing
must be the writer's exact bytes).
Run: ``python -m backends.direct.ligerito.redteam_framing DUMP OUT [--bin ligerito-verify]``."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess


def blobs(raw: bytes) -> list:
    o, out = 8, []
    for _ in range(3):
        n = struct.unpack_from("<I", raw, o)[0]
        out.append((o + 4, n))
        o += 4 + n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--bin", default=None)
    a = ap.parse_args()
    raw = open(os.path.join(a.dump, "honest_fiat-shamir.lgto"), "rb").read()
    stmt = open(os.path.join(a.dump, "honest_fiat-shamir.stmt"), "rb").read()
    (_, _), (f0, fn), _ = blobs(raw)
    fr = raw[f0:f0 + fn].decode()
    fj = json.loads(fr)
    variants = {"space": fr.replace(":", ": ", 1), "extra_key": json.dumps({**fj, "zz": 1}, separators=(",", ":")),
                "reordered": json.dumps(dict(reversed(list(fj.items()))), separators=(",", ":")), "indented": json.dumps(fj, indent=2)}
    if "t_pad" not in fj:
        # compact, known keys, writer's order, yet not the writer's bytes: the writer emits t_pad only with ZK padding, and
        # t_pad 0 parses to the same proof (no ybar); relation-2 32e9bd59's canonical check rebuilds it, verify-rs-3 2dfbb90a
        # rejects it
        variants["t_pad_zero"] = json.dumps({**fj, "t_pad": 0}, separators=(",", ":"))
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(os.path.join(a.dump, "key.bin"), os.path.join(a.out, "key.bin"))
    files = []
    for name, txt in variants.items():
        b = txt.encode()
        pr = raw[:f0 - 4] + struct.pack("<I", len(b)) + b + raw[f0 + fn:]
        stem = f"framing_{name}"
        open(os.path.join(a.out, f"{stem}.lgto.neg"), "wb").write(pr)
        open(os.path.join(a.out, f"{stem}.stmt"), "wb").write(stmt)
        ent = {"proof": f"{stem}.lgto.neg", "stmt": f"{stem}.stmt", "expect": "reject", "coins": "fiat-shamir",
               "name": f"honest FS proof, framing re-encoded ({name}): {txt[:60]!r}", "proof_sha256": hashlib.sha256(pr).hexdigest()}
        if a.bin:
            shutil.copy(os.path.join(a.out, f"{stem}.lgto.neg"), "/tmp/_rtl3_f.lgto")
            j = json.loads(subprocess.run([a.bin, "verify", "--proof", "/tmp/_rtl3_f.lgto", "--stmt", os.path.join(a.out, f"{stem}.stmt"),
                                           "--key", os.path.join(a.out, "key.bin")], capture_output=True, text=True).stdout)
            ent["rust_verdict_today"] = {"accepted": j["accepted"], "reason": j["reason"]}
            print(f"framing {name}: accepted={j['accepted']} ({j['reason'][:80]})")
        files.append(ent)
    src = json.load(open(os.path.join(a.dump, "manifest.json")))
    json.dump({"format": src["format"], "relation": src["relation"], "l": src["l"], "commit": src.get("commit"),
               "kind": "red-team-ligerito-3 framing re-encodings (accepted today; must reject once framing is canonical)",
               "files": files}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
