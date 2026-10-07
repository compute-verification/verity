"""Statement negatives for a Flock cell's instance files, run with the cell's own binary against a loopback verifier.

    python -m verity_flock.negatives --bin flock-pure-gpu --relation R --vus N --leaf sha256|blake3 --out DIR [--prove-arg --gpu]

Each case writes the verifier's instance file from the recipe, changes it the way a cheating prover would need the
verifier to accept, and checks that an honest prover's session over the original data is refused:
- ``valid_chain_over_wrong_row``: the verifier's committed x row 1 differs by one word (its row digest and roots follow),
  so a valid hash chain over the prover's row is a chain over the wrong row;
- ``forged_output_word``: the verifier's committed output word of VU 0 differs by one bit (its y leaf and root follow).
An honest session is run first as the control. Lines: ``NEG\\t{json}``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import instances, lowering


def _rewrite(src: Path, dst: Path, relation: str, scheme: str, change: str) -> None:
    head, body = src.read_bytes().split(b"\n", 1)
    h = json.loads(head)
    n, rb, units = h["vus"], h["row_bytes"], h["units"]
    dt = "<u2" if h["word_bits"] == 16 else "u1"
    x = np.frombuffer(body[:n * rb], dtype=dt).reshape(n, -1).copy()
    w = np.frombuffer(body[n * rb:2 * n * rb], dtype=dt).reshape(n, -1).copy()
    accs = np.frombuffer(body[2 * n * rb:2 * n * rb + 4 * n * units], dtype="<u4").reshape(n, units).copy()
    out = np.frombuffer(body[2 * n * rb + 4 * n * units:], dtype="<u4")[:n].copy()
    if change == "row":
        x[1, 5] ^= 1
    st = instances.statement(relation, x, w, accs, n, 0, scheme)
    y = np.asarray(st["y"], dtype="<u4")
    if change == "output":
        out[0] ^= 1 << 12
        y[0] = out[0] if h["epilogue"] else out[0] >> 10
    h.update({k: v for k, v in st.items() if k != "y"})
    if change == "output":
        from verity.ml.commitments.frame_v3 import FrameV3, Port, WordLeaf
        from verity.ml.commitments.indexed import RangeIndexedDomain
        from verity.ml.commitments.merkle import CommitmentDomain
        dom = CommitmentDomain(bytes.fromhex(st["bindings"]["y"]), st["owners"]["y"], RangeIndexedDomain(0, n))
        fv = FrameV3({"y": Port(dom, WordLeaf(st["schemas"]["y"], st["y_bytes"]))})
        leaves = [fv.tree_leaf("y", r, fv.leaf_layout("y", r).digest(int(v).to_bytes(st["y_bytes"], "big"))) for r, v in enumerate(y)]
        h["roots"]["y"] = fv.root("y", leaves).hex()
    with open(dst, "wb") as f:
        f.write(json.dumps(h, sort_keys=True).encode() + b"\n")
        for arr in (x, w, accs, out, y):
            f.write(np.ascontiguousarray(arr).tobytes())


def session(a, prover_file: Path, verifier_file: Path, d: Path, port: int) -> dict:
    net = d / "net.txt"
    pin = lowering.PINS[a.relation]
    log = d / f"serve-{port}.log"
    srv = subprocess.Popen([a.bin, "serve", "--listen", f"127.0.0.1:{port}", "--out", str(d / f"s{port}"), "--pin", pin,
                            "--instances", str(verifier_file), "--netlist", str(net), "--sessions", "1"],
                           stdout=open(log, "w"), stderr=subprocess.STDOUT)
    t = time.time()
    while "SERVING" not in (log.read_text() if log.exists() else "") and time.time() - t < 600:
        time.sleep(0.5)
    out = subprocess.run([a.bin, "prove", "--verifier", f"127.0.0.1:{port}", "--instances", str(prover_file), "--netlist", str(net),
                          *a.prove_arg], capture_output=True, text=True)
    srv.wait(timeout=300)
    live = [json.loads(line.split("\t", 1)[1], strict=False) for line in out.stdout.splitlines() if line.startswith("LIVE\t")]
    r = live[0] if live else {}
    return {"accepted": r.get("accepted") is True, "why": str(r.get("why") or r.get("prover_stopped") or out.stderr[-300:])[:300]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verity_flock.negatives")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--relation", required=True)
    ap.add_argument("--vus", type=int, default=64)
    ap.add_argument("--leaf", default="blake3", choices=sorted(instances.SCHEMES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prove-arg", action="append", default=[])
    a = ap.parse_args(argv)
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "net.txt").write_text(lowering.netlist(a.relation))
    honest = d / "honest.bin"
    instances.write(honest, a.relation, a.vus, leaf=a.leaf)
    ok = True
    for i, (case, change, want) in enumerate([("honest_control", None, True), ("valid_chain_over_wrong_row", "row", False),
                                              ("forged_output_word", "output", False)]):
        vf = honest
        if change:
            vf = d / f"verifier-{case}.bin"
            _rewrite(honest, vf, a.relation, a.leaf, change)
        r = session(a, honest, vf, d, 7600 + i)
        passed = r["accepted"] == want
        ok &= passed
        print("NEG\t" + json.dumps({"case": case, "leaf": a.leaf, "relation": a.relation, "expect": "accept" if want else "reject",
                                    "accepted": r["accepted"], "pass": passed, "why": r["why"]}), flush=True)
    print("NEGATIVES\t" + json.dumps({"all_pass": ok}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
