"""Route (a) cell: A-GKR (this backend, the prime side with the sigma link, ``gpu/link.py``) and Flock (the binary side,
``backends/flock/live``: ``flock-link cell-prove``) in ONE live session per batch, on the operands-committed statement
``bf16-ampere+blake3`` (row leaves ``blake3-keyed/row/v2``, the leaf Flock proves).

    python tools/cell.py circuits --out DIR        the linked statement's circuit files (torch-free; pins.txt's generator)
    python tools/cell.py statement --cell CELL --instances DIR [--vus N]
        CELL/statement/ (circuit files, public.bin, commitment.txt, sigma.txt), CELL/leaf_digests.bin (x rows then W
        columns, 32 bytes each: what the Flock verifier is configured with), CELL/operands.bin (the prover's rows)
    python tools/cell.py prove --cell CELL --flock BIN --verifier HOST:PORT [--sessions 3] [--warmup 1] [--rust VB]
        per session: the prime prover runs to its link, hands root_F to ``flock-link cell-prove`` (which commits it with
        root_B, gets the verifier's two points, sends y and proves both Flock reps), takes the session's context and
        finishes; CELL/s<i>/ holds proof.bin, link.txt, flock.jsonl, times.json; the Python verifier (and the Rust one
        with --rust) checks proof.bin against link.txt

A session's evidence is the verifier's record, not these files: ``tools/cell_gate.py`` admits it (E1) and writes the
link.txt the prime verifier checks the proof against.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

REL, LEAF, MODEL, BITS = "bf16-ampere", "blake3", "ampere_bf16_m16n8k16", 16


def circuits(out: Path) -> None:
    import link_stub as LS
    from gpu import commit as CM

    tmp = out / ".base"
    subprocess.run([sys.executable, "-m", "gpu.v2.export", "circuits", "--model", MODEL, "--out", str(tmp)], check=True, cwd=HERE,
                   stdout=subprocess.DEVNULL)
    base = (tmp / "circuit.txt").read_text()
    names = [ln.split()[2] for ln in base.splitlines() if ln.startswith("col ")]
    k = (names.index("y.s") - 5) // 2
    (out / "circuit.txt").write_text(LS.extend_unit(base, list(range(5, 5 + 2 * k)), BITS))
    (out / "epilogue.txt").write_text(CM.extend_epilogue((tmp / "epilogue.txt").read_text()))
    (out / "chain.txt").write_text(CM.extend_chain((tmp / "chain.txt").read_text()))
    shutil.copyfile(tmp / "manifest.json", out / "manifest.json")
    shutil.rmtree(tmp)


def statement(cell: Path, instances: Path, n: int, flock: str) -> None:
    import numpy as np

    import bench_result as br
    from gpu import commit as CM
    from gpu.bb_export import write_rows

    sd = cell / "statement"
    sd.mkdir(parents=True, exist_ok=True)
    circuits(sd)
    src = HERE.parents[1]
    frozen = src / "fixtures" / "bench-instances" / "v1" / "manifest.json"
    x, W, y0, _ = br.load_frozen(instances, frozen, br.TIER, 0, n)
    X = np.array(x, dtype=np.uint16)
    Wn = np.array(W, dtype=np.uint16).reshape(n, -1)
    y = np.asarray(y0).reshape(n, 1)
    C = CM.commit(LEAF, REL, CM.instance_set(REL, 0, n, src, frozen), 0, X, Wn, y, 2)
    (sd / "commitment.txt").write_text(C.text())
    write_rows(C.public(np.asarray(y, dtype=np.int64)).tolist(), sd / "public.bin")
    (cell / "leaf_digests.bin").write_bytes(b"".join(C.digests["a"]) + b"".join(C.digests["b"]))
    (cell / "operands.bin").write_bytes(X.astype("<u2").tobytes() + Wn.astype("<u2").tobytes())
    np.save(cell / "y.npy", np.asarray(y0))
    pc = hashlib.sha256((sd / "commitment.txt").read_bytes()).hexdigest()
    sig = subprocess.run([flock, "sigma", "--vus", str(n), "--digests", str(cell / "leaf_digests.bin"), "--prime-commitment", pc],
                         check=True, capture_output=True, text=True).stdout
    (sd / "sigma.txt").write_text(sig)
    (cell / "cell.json").write_text(json.dumps({"relation": C.relation, "vus": n, "prime_commitment": pc,
                                                "sigma": hashlib.sha256(sig.encode()).hexdigest()}, indent=1))
    print(f"statement {C.relation} vus {n} prime_commitment {pc} sigma {hashlib.sha256(sig.encode()).hexdigest()}", flush=True)


def prove(a) -> int:
    import numpy as np
    import torch

    from gpu import link as LK
    from gpu import prover
    from gpu.circuit import layers, parse_circuit
    from gpu.run import read_chain
    from gpu.v2.witness import Generator, Ops
    from verity_numerical.checker import REAL

    cell = Path(a.cell)
    sd = cell / "statement"
    meta = json.loads((cell / "cell.json").read_text())
    n = meta["vus"]
    dev = torch.device("cuda")
    man = json.loads((sd / "manifest.json").read_text())
    steps = man["steps"]
    uc, ec = parse_circuit((sd / "circuit.txt").read_text()), parse_circuit((sd / "epilogue.txt").read_text())
    ul, el = layers(uc), layers(ec)
    ops_b = np.frombuffer((cell / "operands.bin").read_bytes(), dtype="<u2").reshape(2 * n, -1)
    X, Wn = ops_b[:n].copy(), ops_b[n:].copy()
    K = X.shape[1]
    y0 = np.load(cell / "y.npy")
    ops = Ops("cuda")
    gen = Generator(ops, REAL)
    rows = gen.run(ops.asarray(X), ops.asarray(Wn), ops.asarray(y0))
    assert int(rows.bad.sum()) == 0
    k = (uc.col_names.index("y.s") - 5) // 2
    ci = torch.tensor(list(range(5, 5 + 2 * k)), device=dev)
    sh = torch.arange(BITS, device=dev)
    v = rows.units.index_select(1, ci)
    units = torch.cat([rows.units, ((v[:, :, None] >> sh) & 1).reshape(rows.units.shape[0], -1)], 1)
    pub = np.fromfile(sd / "public.bin", dtype="<u4")[4:].astype(np.int64)
    lim = pub.reshape(n, -1)[:, 1:]
    epi = torch.cat([rows.epilogue, torch.from_numpy(lim).to(dev)], 1)
    lay = LK.derive(uc, n, steps, K, BITS)
    cd = hashlib.sha256((sd / "commitment.txt").read_bytes()).digest()
    pc = meta["prime_commitment"]
    out = []
    for i in range(a.warmup + a.sessions):
        s = cell / f"s{i}"
        shutil.rmtree(s, ignore_errors=True)
        s.mkdir()
        fl = open(s / "flock.jsonl", "w")
        fe = open(s / "flock.stderr", "w")
        fp = subprocess.Popen([a.flock, "cell-prove", "--verifier", a.verifier, "--vus", str(n), "--digests", str(cell / "leaf_digests.bin"),
                               "--prime-commitment", pc, "--operands", str(cell / "operands.bin"), "--root-f", str(s / "root_f.hex"),
                               "--link-out", str(s / "link.json"), "--start", str(s / "start")], stdout=fl, stderr=fe)
        while "READY" not in (s / "flock.stderr").read_text():   # the Flock statement is built (preprocessing, untimed)
            if fp.poll() is not None:
                raise SystemExit(f"flock-link cell-prove exited {fp.returncode}: {(s / 'flock.stderr').read_text()[-400:]}")
            time.sleep(0.01)
        t0 = time.perf_counter()
        (s / "start").touch()
        tw = {}

        def exchange(root_f: bytes, s=s, tw=tw) -> LK.Context:
            tx = time.perf_counter()
            (s / "root_f.tmp").write_text(root_f.hex())
            os.replace(s / "root_f.tmp", s / "root_f.hex")
            while not (s / "link.json").exists():
                if fp.poll() is not None:
                    raise SystemExit(f"flock-link cell-prove exited {fp.returncode} before the link context: {(s / 'flock.stderr').read_text()[-400:]}")
                time.sleep(0.001)
            d = json.loads((s / "link.json").read_text())
            tw["exchange_wait_s"] = time.perf_counter() - tx
            return LK.Context(bytes.fromhex(d["sigma"]), bytes.fromhex(d["root_f_sha256"]), bytes.fromhex(d["root_b_sha256"]),
                              tuple(tuple(int(c, 16) for c in p) for p in d["points"]), tuple(int(c, 16) for c in d["y"]))

        ch = read_chain(sd / "chain.txt", uc, ec, steps, [int(v) for v in pub])
        ch.commit_digest = cd
        inst = prover.Instance([prover.Segment("unit", uc, ul, units, uc.hash), prover.Segment("epilogue", ec, el, epi, ec.hash)], ch,
                               link=LK.Link(lay, exchange=exchange, x=X, w=Wn))
        torch.cuda.synchronize(dev)
        tp = time.perf_counter()
        proof, st = prover.prove(inst, True)
        torch.cuda.synchronize(dev)
        t_prime = time.perf_counter() - tp
        rc = fp.wait()
        t_cell = time.perf_counter() - t0
        fl.close()
        fe.close()
        ctx = inst.link.context
        data = proof.to_bytes()
        (s / "proof.bin").write_bytes(data)
        (s / "link.txt").write_text(ctx.text())
        live = [json.loads(ln.split("\t", 1)[1]) for ln in (s / "flock.jsonl").read_text().splitlines() if ln.startswith("LIVE\t")]
        vinst = prover.Instance([prover.Segment("unit", uc, ul, units, uc.hash), prover.Segment("epilogue", ec, el, epi, ec.hash)], ch,
                                link=LK.Link(lay, context=LK.parse_context((s / "link.txt").read_text())))
        tv = time.perf_counter()
        try:
            prover.verify(vinst, proof, True)
            py = "accept"
        except prover.VerifyError as e:
            py = f"reject ({str(e)[:160]})"
        t_py = time.perf_counter() - tv
        row = {"session": i, "warmup": i < a.warmup, "t_cell_s": t_cell, "t_prime_prove_s": t_prime, "t_link_s": st.t_link,
               "flock_rc": rc, "flock": live[0] if live else None, "python_verify": py, "python_verify_s": t_py,
               "proof_bytes": len(data), "proof_sha256": hashlib.sha256(data).hexdigest(), **tw}
        if a.rust:
            vd = s / "verify"
            shutil.copytree(sd, vd, dirs_exist_ok=True)
            shutil.copyfile(s / "link.txt", vd / "link.txt")
            r = subprocess.run([a.rust, "verify", "--dir", str(vd), "--proof", str(s / "proof.bin"), "--vus", str(n), "--threads", str(a.threads),
                                "--relation", meta["relation"], "--json", str(s / "rust.json")]
                               + (["--allow-unpinned-commitment"] if n != 4096 else []), capture_output=True, text=True)
            doc = json.loads((s / "rust.json").read_text()) if (s / "rust.json").is_file() else {"error": r.stderr[-300:]}
            row["rust"] = {k: doc.get(k) for k in ("accepted", "error", "verify_seconds", "t_link", "circuit_pinned", "operands_committed")}
        (s / "times.json").write_text(json.dumps(row, indent=1))
        fl_ok = bool(live and live[0].get("accepted"))
        print(f"session {i}{' (warm-up)' if i < a.warmup else ''}: cell {t_cell:.3f} s, prime prove {t_prime:.3f} s (link {st.t_link:.3f}, "
              f"wait {tw.get('exchange_wait_s', 0):.3f}), flock {live[0]['prove_total_s'] if live else None} s accepted {fl_ok}, "
              f"python {py}" + (f", rust {row['rust']}" if a.rust else ""), flush=True)
        out.append(row)
        del inst, vinst, proof
    (cell / "sessions.json").write_text(json.dumps(out, indent=1))
    ok = all(r["python_verify"] == "accept" and r["flock_rc"] == 0 and r["flock"] and r["flock"].get("accepted")
             and (not a.rust or r["rust"].get("accepted")) for r in out)
    print("CELL-PROVE OK" if ok else "CELL-PROVE FAILED", flush=True)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools/cell.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("circuits")
    c.add_argument("--out", required=True)
    s = sub.add_parser("statement")
    s.add_argument("--cell", required=True)
    s.add_argument("--instances", required=True)
    s.add_argument("--vus", type=int, default=4096)
    s.add_argument("--flock", required=True)
    p = sub.add_parser("prove")
    p.add_argument("--cell", required=True)
    p.add_argument("--flock", required=True)
    p.add_argument("--verifier", required=True)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--rust", default=None)
    p.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    if a.cmd == "circuits":
        Path(a.out).mkdir(parents=True, exist_ok=True)
        circuits(Path(a.out))
        return 0
    if a.cmd == "statement":
        statement(Path(a.cell), Path(a.instances), a.vus, a.flock)
        return 0
    return prove(a)


if __name__ == "__main__":
    sys.exit(main())
