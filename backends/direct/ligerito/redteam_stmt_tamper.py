"""red-team-ligerito-3: statement-tamper must-reject fixtures for LGTO0001 dumps (no prover needed).

From a gate dump's ``honest_local.{lgto,stmt}`` + ``key.bin``, write copies of the honest proof next to LGTS0001
statements edited in one field each: a non-end claimed word, an out-of-range claimed word, the first pad unit's operand
a / b words and claimed word, the first end word +1, one real operand word, the same words re-encoded at twice the
relation's word_bytes (Python verify does not compare word_bytes / y_bytes with the relation; Rust does, lgto.rs "statement
shape").  Every entry must be rejected: the public rows
enter ``z(r_i, r_c)`` directly, so an honest proof never survives a statement edit, even under local coins.  This does NOT
show the edited field is constrained: the end constraint is masked off the chain ends, so a proof MADE for a statement with a
nonzero off-end word verifies unless the statement check rejects it (relation-2 0db857a9 added that check).  The output directory
is a ``ligerito-verify batch --dir`` dump (``manifest.json`` files with ``expect: reject``; no Python verdicts: the laptop has
no torch).  Run: ``python -m backends.direct.ligerito.redteam_stmt_tamper DUMP OUT [--bin ligerito-verify]``."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess


def parse(raw: bytes) -> dict:
    o = 8
    n = struct.unpack_from("<I", raw, o)[0]
    rel = raw[o + 4:o + 4 + n].decode()
    o += 4 + n + 64
    l, steps, K, S = struct.unpack_from("<4I", raw, o)
    o += 16
    nv = list(struct.unpack_from(f"<{S}I", raw, o))
    o += 4 * S
    wb, yb = raw[o], raw[o + 1]
    o += 2
    subs = []
    for s in range(S):
        if nv[s]:
            subs.append({"s": s, "a": o, "b": o + l * K * wb, "y": o + 2 * l * K * wb})
            o += 2 * l * K * wb + l * yb
    assert o == len(raw), "LGTS0001 length"
    return {"relation": rel, "l": l, "steps": steps, "K": K, "n_vus": nv, "wb": wb, "yb": yb, "subs": subs}


def tampers(raw: bytes, st: dict) -> list:
    l, steps, K, wb, yb = st["l"], st["steps"], st["K"], st["wb"], st["yb"]
    sub = st["subs"][0]
    nv = st["n_vus"][sub["s"]]
    ends = [v * steps + steps - 1 for v in range(nv)]
    rd = lambda off, w: int.from_bytes(raw[off:off + w], "little")
    nonend = next(j for j in range(l) if j not in ends)
    out = [("y_nonend_1", "claimed word 1 at a non-end column (honest proof made for 0 there)", [(sub["y"] + nonend * yb, yb, 1)]),
           ("y_out_of_range", "claimed word 2^(8 y_bytes) - 1 at a non-end column", [(sub["y"] + nonend * yb, yb, (1 << (8 * yb)) - 1)]),
           ("y_end_plus1", "first end column's claimed word + 1", [(sub["y"] + ends[0] * yb, yb, (rd(sub["y"] + ends[0] * yb, yb) + 1) % (1 << (8 * yb)))]),
           ("a_real_plus1", "operand a word of unit 0, operand 0, + 1", [(sub["a"], wb, (rd(sub["a"], wb) + 1) % (1 << (8 * wb)))])]
    pad0 = nv * steps
    if pad0 < l:
        out += [("pad_a", f"first pad unit {pad0}: operand a[0] = 1", [(sub["a"] + pad0 * K * wb, wb, 1)]),
                ("pad_b", f"first pad unit {pad0}: operand b[1] = 1", [(sub["b"] + (pad0 * K + 1) * wb, wb, 1)]),
                ("pad_y", f"first pad unit {pad0}: claimed word 7", [(sub["y"] + pad0 * yb, yb, 7)])]
    res = []
    for name, what, edits in out:
        b = bytearray(raw)
        for off, w, val in edits:
            b[off:off + w] = val.to_bytes(w, "little")
        res.append((name, what, bytes(b)))
    if wb < 4:
        res.append(("words_widened", f"same words re-encoded at word_bytes {2 * wb} (the relation's is {wb})",
                    widen(raw, st, 2 * wb, yb)))
    return res


def widen(raw: bytes, st: dict, wb2: int, yb2: int) -> bytes:
    """The same statement (every word unchanged) with word_bytes / y_bytes = wb2 / yb2."""
    l, K, wb, yb = st["l"], st["K"], st["wb"], st["yb"]
    head = st["subs"][0]["a"] - 2
    out = bytearray(raw[:head]) + bytes([wb2, yb2])
    word = lambda off, w: int.from_bytes(raw[off:off + w], "little")
    for sub in st["subs"]:
        for side, n, w, w2 in (("a", l * K, wb, wb2), ("b", l * K, wb, wb2), ("y", l, yb, yb2)):
            for i in range(n):
                out += word(sub[side] + i * w, w).to_bytes(w2, "little")
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--bin", default=None, help="ligerito-verify release binary: run each entry, require reject")
    a = ap.parse_args()
    raw = open(os.path.join(a.dump, "honest_local.stmt"), "rb").read()
    proof = open(os.path.join(a.dump, "honest_local.lgto"), "rb").read()
    src = json.load(open(os.path.join(a.dump, "manifest.json")))
    seed = next(f.get("coins_seed") for f in src["files"] if f["proof"] == "honest_local.lgto")
    st = parse(raw)
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(os.path.join(a.dump, "key.bin"), os.path.join(a.out, "key.bin"))
    files, bad = [], 0
    for name, what, b in tampers(raw, st):
        stem = f"tamper_{name}"
        open(os.path.join(a.out, f"{stem}.stmt"), "wb").write(b)
        open(os.path.join(a.out, f"{stem}.lgto.neg"), "wb").write(proof)
        ent = {"proof": f"{stem}.lgto.neg", "stmt": f"{stem}.stmt", "expect": "reject", "coins": "local", "coins_seed": seed,
               "name": what, "proof_sha256": hashlib.sha256(proof).hexdigest(), "stmt_sha256": hashlib.sha256(bytes(b)).hexdigest()}
        if a.bin:
            shutil.copy(os.path.join(a.out, f"{stem}.lgto.neg"), "/tmp/_rtl3_t.lgto")
            shutil.copy(os.path.join(a.out, f"{stem}.stmt"), "/tmp/_rtl3_t.stmt")
            shutil.copy(os.path.join(a.out, "key.bin"), "/tmp/key.bin")
            j = json.loads(subprocess.run([a.bin, "verify", "--proof", "/tmp/_rtl3_t.lgto", "--stmt", "/tmp/_rtl3_t.stmt",
                                           "--key", "/tmp/key.bin"], capture_output=True, text=True).stdout)
            ent["rust_verdict"] = {"accepted": j["accepted"], "reason": j["reason"]}
            bad += bool(j["accepted"])
            print(f"{st['relation']} {name}: accepted={j['accepted']} ({j['reason'][:90]})")
        files.append(ent)
    json.dump({"format": src["format"], "relation": st["relation"], "l": st["l"], "commit": src.get("commit"),
               "kind": "red-team-ligerito-3 statement tampers (honest local proof, edited statement)", "files": files},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
