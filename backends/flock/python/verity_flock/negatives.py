"""Statement negatives for a Flock cell's instance files, run with the cell's own binary against a loopback verifier.

    python -m verity_flock.negatives --bin flock-pure-gpu --relation R --vus N --leaf sha256|blake3 --out DIR [--prove-arg --gpu]

Each case writes the verifier's instance file from the recipe, changes it the way a cheating prover would need the
verifier to accept, and checks that an honest prover's session over the original data is refused:
- ``valid_chain_over_wrong_row``: the verifier's committed x row 1 differs by one word (its row digest and roots follow),
  so a valid hash chain over the prover's row is a chain over the wrong row;
- ``forged_output_word``: the verifier's committed output word of VU 0 differs by one bit (its y leaf and root follow).
A total relation (``bf16-ampere-total``) runs on ``instances.write_probe`` (NaN, +-inf, signed zeros, subnormals) and adds,
each with the output and y words moved together so admission passes, run twice: an honest prover (refused at the session
parameters, Sigma differs) and a cheating prover holding the same forged file (Sigma agrees; only the proof can refuse):
- ``nan_output_claimed_finite``: a VU whose output is the NaN word 0x7FFF (a NaN operand) is committed as 1.0 (0x3F80);
- ``inf_times_zero_claimed_finite``: a VU whose NaN comes from an infinity times a zero is committed as +0;
- ``nan_payload_changed``: a NaN output committed as another NaN payload (0x7FC0, the software conversion's);
- ``inf_output_sign_flipped``: a VU whose output is an infinity is committed as the other infinity;
- ``finite_output_claimed_nan``: a VU with a finite output is committed as 0x7FFF.
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
        from verity.commitments.frame_v3 import FrameV3, Port, WordLeaf
        from verity.commitments.indexed import RangeIndexedDomain
        from verity.commitments.merkle import CommitmentDomain
        dom = CommitmentDomain(bytes.fromhex(st["bindings"]["y"]), st["owners"]["y"], RangeIndexedDomain(0, n))
        fv = FrameV3({"y": Port(dom, WordLeaf(st["schemas"]["y"], st["y_bytes"]))})
        leaves = [fv.tree_leaf("y", r, fv.leaf_layout("y", r).digest(int(v).to_bytes(st["y_bytes"], "big"))) for r, v in enumerate(y)]
        h["roots"]["y"] = fv.root("y", leaves).hex()
    with open(dst, "wb") as f:
        f.write(json.dumps(h, sort_keys=True).encode() + b"\n")
        for arr in (x, w, accs, out, y):
            f.write(np.ascontiguousarray(arr).tobytes())


def _rewrite_output(src: Path, dst: Path, pick, word: int) -> int:
    """``src`` with the output and y words of the first VU whose output satisfies ``pick`` set to ``word`` (an epilogue
    relation: y = out), the y root recomputed under the file's own binding; returns that VU, or -1."""
    from verity.commitments.frame_v3 import FrameV3, Port, WordLeaf
    from verity.commitments.indexed import RangeIndexedDomain
    from verity.commitments.merkle import CommitmentDomain

    head, body = src.read_bytes().split(b"\n", 1)
    h = json.loads(head)
    n, rb, units = h["vus"], h["row_bytes"], h["units"]
    off = 2 * n * rb + 4 * n * units
    out = np.frombuffer(body[off:off + 4 * n], dtype="<u4").copy()
    y = np.frombuffer(body[off + 4 * n:off + 8 * n], dtype="<u4").copy()
    v = next((i for i in range(n) if pick(i, int(out[i]))), -1)
    if v < 0:
        return v
    out[v] = y[v] = word
    dom = CommitmentDomain(bytes.fromhex(h["bindings"]["y"]), h["owners"]["y"], RangeIndexedDomain(0, n))
    fv = FrameV3({"y": Port(dom, WordLeaf(h["schemas"]["y"], h["y_bytes"]))})
    leaves = [fv.tree_leaf("y", r, fv.leaf_layout("y", r).digest(int(u).to_bytes(h["y_bytes"], "big"))) for r, u in enumerate(y)]
    h["roots"]["y"] = fv.root("y", leaves).hex()
    with open(dst, "wb") as f:
        f.write(json.dumps(h, sort_keys=True).encode() + b"\n")
        f.write(body[:off])
        f.write(out.tobytes())
        f.write(y.tobytes())
    return v


TOTAL_CASES = [("nan_output_claimed_finite", lambda v, o: o == 0x7FFF and v % 8 == 2, 0x3F80),
               ("inf_times_zero_claimed_finite", lambda v, o: o == 0x7FFF and v % 8 == 6, 0x0000),
               ("nan_payload_changed", lambda v, o: o == 0x7FFF, 0x7FC0),
               ("inf_output_sign_flipped", lambda v, o: o in (0x7F80, 0xFF80), None),
               ("finite_output_claimed_nan", lambda v, o: (o >> 7) & 0xFF != 0xFF, 0x7FFF)]


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
    total = lowering.PIPES[a.relation].total
    if total:
        instances.write_probe(honest, a.relation, a.vus, leaf=a.leaf)
    else:
        instances.write(honest, a.relation, a.vus, leaf=a.leaf)
    ok = True
    if total:
        for i, (case, pick, word) in enumerate(TOTAL_CASES):
            vf = d / f"verifier-{case}.bin"
            head, body = honest.read_bytes().split(b"\n", 1)
            v = _rewrite_output(honest, vf, pick, word if word is not None else 0)
            if case == "inf_output_sign_flipped" and v >= 0:
                h = json.loads(head)
                off = 2 * h["vus"] * h["row_bytes"] + 4 * h["vus"] * h["units"]
                o = int(np.frombuffer(body[off:off + 4 * h["vus"]], dtype="<u4")[v])
                v = _rewrite_output(honest, vf, lambda i, u, v=v: i == v, o ^ 0x8000)
            if v < 0:
                print("NEG\t" + json.dumps({"case": case, "relation": a.relation, "pass": False, "why": "the probe has no such VU"}), flush=True)
                ok = False
                continue
            for j, (who, pf) in enumerate((("honest_prover", honest), ("cheating_prover", vf))):
                r = session(a, pf, vf, d, 7700 + 2 * i + j)
                passed = not r["accepted"]
                ok &= passed
                print("NEG\t" + json.dumps({"case": case, "prover": who, "vu": v, "leaf": a.leaf, "relation": a.relation,
                                            "expect": "reject", "accepted": r["accepted"], "pass": passed, "why": r["why"]}), flush=True)
    base = [("honest_control", None, True)] + ([] if total else [("valid_chain_over_wrong_row", "row", False),
                                                                ("forged_output_word", "output", False)])
    for i, (case, change, want) in enumerate(base):
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
