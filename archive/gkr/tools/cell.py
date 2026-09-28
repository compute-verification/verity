"""Route (a) cell: A-GKR (this backend, the prime side with the sigma link, ``gpu/link.py``) and Flock (the binary side,
``verity/ml/flock/live``: ``flock-link cell-prove``) in ONE live session per batch, on the operands-committed statement
``bf16-ampere+blake3`` (row leaves ``blake3-keyed/row/v2``, the leaf Flock proves) at K = 1536, and ``bf16-ampere-k<K>+blake3``
at vLLM's reduction lengths (the same circuit files, ``steps`` = K / 16; Flock's row of 2K bytes is K / 512 BLAKE3 chunks).

    python tools/cell.py circuits --out DIR [--k K]   the linked statement's circuit files (torch-free; pins.txt's generator)
    python tools/cell.py statement --cell CELL (--instances DIR | --input-set SET) [--vus N] [--verifier-side]
        CELL/statement/ (circuit files, public.bin, commitment.txt, sigma.txt), CELL/leaf_digests.bin (x rows then W
        columns, 32 bytes each: what the Flock verifier is configured with), CELL/operands.bin (the prover's rows).
        --instances: the frozen bench-instances/v1 (K = 1536, tiled past its 4,096 VUs); --input-set: VUs [0, N) of an
        input set (``verity_numerical.bench.input_sets``, never recycled), its K, its identity bound as B-Ligero binds it.
        --verifier-side: only leaf_digests.bin and cell.json (the verifier's own statement from the set: no torch, no sigma)
    python tools/cell.py prove --cell CELL --flock BIN --verifier HOST:PORT [--sessions 3] [--warmup 1] [--rust VB]
                               [--prime-coins live|fs]
        per session: the prime prover runs to its link, hands root_F to ``flock-link cell-prove`` (which commits it with
        root_B, gets the verifier's two points, sends y and proves both Flock reps), takes the session's context and
        finishes; CELL/s<i>/ holds proof.bin, link.txt, prime_coins.txt, flock.jsonl, times.json; the Python verifier
        (and the Rust one with --rust) checks proof.bin against link.txt and the prime rounds.  With live prime coins
        (the default, PROTOCOL 17.6) every prime challenge is a round with the same session's verifier, forwarded by
        cell-prove from CELL/s<i>/prime.sock; ``--prime-coins fs`` is the earlier Fiat-Shamir prime half (diagnostic)

A session's evidence is the verifier's record, not these files: ``tools/cell_gate.py`` admits it (E1) and writes the
link.txt the prime verifier checks the proof against.

The bench.cell driver (``verity_numerical.bench.drivers.a_route_a``, pod script ``archive/gkr/cell.sh``) adds:

    python tools/cell.py finish --points DIR... --verifier HOST:PORT [--operator LANE] --out result.json
        the plateau of a sweep (each DIR holds one size's result.json, times of its loopback warm-up and its guard verdict):
        the plateau point's envelope with the sweep, protocol, loopback and commit blocks and the verifier's placement
    python tools/cell.py register --prover-run DIR [--verifier-run DIR] --lane LANE --out META
        the producing lane and the verifier run's session verdicts stamped on the prover run's result.json
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
SEMANTICS = "sm80.mma.m16n8k16.bf16"
K_FROZEN = 1536


def base_relation(K: int) -> str:
    """The relation the circuit files are pinned under at reduction length ``K`` (``pins.txt``); ``+blake3`` is the committed one."""
    return REL if K == K_FROZEN else f"{REL}-k{K}"


def circuits(out: Path, K: int = K_FROZEN) -> None:
    import link_stub as LS
    from gpu import commit as CM

    tmp = out / ".base"
    subprocess.run([sys.executable, "-m", "gpu.v2.export", "circuits", "--model", MODEL, "--k", str(K), "--out", str(tmp)], check=True,
                   cwd=HERE, stdout=subprocess.DEVNULL)
    base = (tmp / "circuit.txt").read_text()
    names = [ln.split()[2] for ln in base.splitlines() if ln.startswith("col ")]
    k = (names.index("y.s") - 5) // 2
    (out / "circuit.txt").write_text(LS.extend_unit(base, list(range(5, 5 + 2 * k)), BITS))
    (out / "epilogue.txt").write_text(CM.extend_epilogue((tmp / "epilogue.txt").read_text()))
    (out / "chain.txt").write_text(CM.extend_chain((tmp / "chain.txt").read_text()))
    shutil.copyfile(tmp / "manifest.json", out / "manifest.json")
    shutil.rmtree(tmp)


def operands(n: int, instances: Path | None = None, input_set: Path | None = None):
    """``(X, W, y, iset, ref, tiled)``: VUs ``[0, n)`` as ``(n, K)`` uint16 rows and ``(n,)`` words, the instance identity the
    commitment binds, the fingerprint's ``instances`` entry, and how a frozen set was tiled (``None``: not tiled)."""
    import numpy as np

    from gpu import commit as CM

    if input_set is not None:
        from verity_numerical.bench.input_sets import InputSet

        sem = InputSet.open(input_set).subcircuit.params.get("semantics")
        if sem != SEMANTICS:
            raise SystemExit(f"cell.py: input set {input_set} is {sem}; route (a)'s prime relation is {SEMANTICS}")
        X, W, y, iset, ref = CM.input_set(input_set, n, BITS)
        return X, W, y.astype(np.uint16), iset, ref, None
    from verity_numerical.bench import contract

    import bench_result as br

    src = HERE.parents[1]
    frozen = src / "fixtures" / "bench-instances" / "v1" / "manifest.json"
    # past the frozen set's 4,096 VUs the sweep tiles it: VU i is frozen VU i mod 4096 (a re-packing; Rule I wants an
    # instance-equiv/v1 document for it before such a point can head a cell)
    n0 = min(n, 4096)
    x, W, y0, _ = br.load_frozen(instances, frozen, br.TIER, 0, n0)
    reps = -(-n // n0)
    X = np.tile(np.array(x, dtype=np.uint16), (reps, 1))[:n]
    Wn = np.tile(np.array(W, dtype=np.uint16).reshape(n0, -1), (reps, 1))[:n]
    y0 = np.tile(np.asarray(y0), reps)[:n]
    ref = None if n > 4096 else contract.instances_ref(br.TIER, 0, n, hashlib.sha256(frozen.read_bytes()).hexdigest())
    return X, Wn, y0, CM.instance_set(REL, 0, n, src, frozen), ref, None if n <= 4096 else f"bench-instances/v1 vu-k1536 [0, 4096) x {reps}"


def statement(cell: Path, instances: Path | None, n: int, flock: str | None, input_set: Path | None = None,
              verifier_side: bool = False) -> None:
    import numpy as np

    from gpu import commit as CM
    from gpu.bb_export import write_rows

    X, Wn, y0, iset, ref, tiled = operands(n, instances, input_set)
    K = int(X.shape[1])
    y = np.asarray(y0).reshape(n, 1)
    tc = time.perf_counter()
    C = CM.commit(LEAF, base_relation(K), iset, 0, X, Wn, y, 2)
    t_commit = time.perf_counter() - tc
    sd = cell / "statement"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "commitment.txt").write_text(C.text())
    (cell / "leaf_digests.bin").write_bytes(b"".join(C.digests["a"]) + b"".join(C.digests["b"]))
    pc = hashlib.sha256((sd / "commitment.txt").read_bytes()).hexdigest()
    meta = {"relation": C.relation, "vus": n, "K": K, "prime_commitment": pc, "serving_commit_seconds": t_commit,
            "manifest_sha256": iset.manifest_sha256, "instances": ref, "tiled_from": tiled}
    if not verifier_side:
        circuits(sd, K)
        write_rows(C.public(np.asarray(y, dtype=np.int64)).tolist(), sd / "public.bin")
        (cell / "operands.bin").write_bytes(X.astype("<u2").tobytes() + Wn.astype("<u2").tobytes())
        np.save(cell / "y.npy", np.asarray(y0))
        sig = subprocess.run([flock, "sigma", "--vus", str(n), "--k", str(K), "--digests", str(cell / "leaf_digests.bin"),
                              "--prime-commitment", pc], check=True, capture_output=True, text=True).stdout
        (sd / "sigma.txt").write_text(sig)
        meta["sigma"] = hashlib.sha256(sig.encode()).hexdigest()
    (cell / "cell.json").write_text(json.dumps(meta, indent=1))
    print(f"statement {C.relation} vus {n} K {K} prime_commitment {pc} sigma {meta.get('sigma')} commit {t_commit:.2f} s", flush=True)


class PrimeLink:
    """The prime prover's end of the live session: ``Prime`` frames (``verity/ml/flock/live``: tag 8, the state and
    the count; the answer tag 5, the words) on cell-prove's Unix socket, which forwards them on the session."""

    def __init__(self, path: Path) -> None:
        self.path, self.sock, self.rounds, self.bytes_up, self.bytes_down = path, None, 0, 0, 0

    def _connect(self):
        import socket
        t = time.perf_counter()
        while True:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(str(self.path))
                return s
            except (FileNotFoundError, ConnectionRefusedError):
                s.close()
                if time.perf_counter() - t > 120:
                    raise
                time.sleep(0.001)

    def _read(self, n: int) -> bytes:
        b = b""
        while len(b) < n:
            c = self.sock.recv(n - len(b))
            if not c:
                raise ConnectionError("cell-prove closed the prime socket")
            b += c
        return b

    def __call__(self, state: bytes, n: int) -> list[int]:
        import struct

        from gpu.transcript import CoinError
        if self.sock is None:
            self.sock = self._connect()
        req = bytes([8]) + struct.pack("<I", len(state)) + state + struct.pack("<I", n)
        self.sock.sendall(struct.pack("<I", len(req)) + req)
        (ln,) = struct.unpack("<I", self._read(4))
        resp = self._read(ln)
        self.rounds += 1
        self.bytes_up += 4 + len(req)
        self.bytes_down += 4 + ln
        if resp[0] == 5:
            (k,) = struct.unpack_from("<I", resp, 1)
            return list(struct.unpack_from(f"<{k}I", resp, 5))
        if resp[0] == 3:
            (k,) = struct.unpack_from("<I", resp, 1)
            raise CoinError("live prime round refused: " + resp[5:5 + k].decode(errors="replace"))
        raise CoinError(f"live prime round: unexpected response tag {resp[0]}")

    def close(self) -> None:
        if self.sock is None:
            # the session waits for the prime prover to close its end before Finish
            try:
                self.sock = self._connect()
            except OSError:
                return
        self.sock.close()


def stale_live(call, k: int):
    """Negative (a coin before its commitment): at round ``k`` the prime prover sends the verifier the state from before
    its last message, then keeps that message in its proof, so the message follows the coin.  Its own view of the rounds
    (``tr.rounds``) is the honest chain; the verifier's record holds the stale state, and a replay of the record rejects."""
    from gpu.transcript import LiveTranscript

    class Stale(LiveTranscript):
        prev = None

        def absorb(self, data: bytes) -> None:
            self.prev = self.state
            super().absorb(data)

        def _round(self) -> None:
            if len(self.rounds) == k and self.prev is not None:
                real = self.call
                self.call = lambda state, n: real(self.prev, n)
                try:
                    super()._round()
                finally:
                    self.call = real
            else:
                super()._round()
    return Stale(call)


def prove(a) -> int:
    import numpy as np
    import torch

    from gpu import link as LK
    from gpu import prover
    from gpu.transcript import LiveTranscript, ReplayTranscript, coins_text
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
    k = (uc.col_names.index("y.s") - 5) // 2
    ci = torch.tensor(list(range(5, 5 + 2 * k)), device=dev)
    sh = torch.arange(BITS, device=dev)
    pub = np.fromfile(sd / "public.bin", dtype="<u4")[4:].astype(np.int64)
    lim_d = torch.from_numpy(pub.reshape(n, -1)[:, 1:]).to(dev)
    xd, wd, yd = ops.asarray(X), ops.asarray(Wn), ops.asarray(y0)

    def witness():
        """The prime witness from the operands on the device: the unit rows with every operand bit, the epilogue with
        the digest limbs (timed as t.witness)."""
        rows = gen.run(xd, wd, yd)
        if int(rows.bad.sum()):
            bad = torch.nonzero(rows.bad).flatten().tolist()
            raise SystemExit(f"cell.py: {len(bad)} VUs rejected at witness generation (first ids {bad[:8]})")
        v = rows.units.index_select(1, ci)
        units = torch.cat([rows.units, ((v[:, :, None] >> sh) & 1).reshape(rows.units.shape[0], -1)], 1)
        return units, torch.cat([rows.epilogue, lim_d], 1)
    lay = LK.derive(uc, n, steps, K, BITS)
    cd = hashlib.sha256((sd / "commitment.txt").read_bytes()).digest()
    pc = meta["prime_commitment"]
    out = []
    guard = None
    for i in range(a.warmup + a.sessions):
        if i == a.warmup:
            from archive.direct.ligero.timing_guard import Guard
            guard = Guard().start()
        s = cell / f"s{i}"
        shutil.rmtree(s, ignore_errors=True)
        s.mkdir()
        fl = open(s / "flock.jsonl", "w")
        fe = open(s / "flock.stderr", "w")
        addr = a.warmup_verifier if (i < a.warmup and a.warmup_verifier) else a.verifier
        live = a.prime_coins == "live"
        plink = PrimeLink(s / "prime.sock") if live else None
        fp = subprocess.Popen([a.flock, "cell-prove", "--verifier", addr, "--vus", str(n), "--k", str(K), "--digests", str(cell / "leaf_digests.bin"),
                               "--prime-commitment", pc, "--operands", str(cell / "operands.bin"), "--root-f", str(s / "root_f.hex"),
                               "--link-out", str(s / "link.json"), "--start", str(s / "start")]
                              + (["--prime-sock", str(s / "prime.sock")] if live else []), stdout=fl, stderr=fe)
        while "READY" not in (s / "flock.stderr").read_text():   # the Flock statement is built (preprocessing, untimed)
            if fp.poll() is not None:
                raise SystemExit(f"flock-link cell-prove exited {fp.returncode}: {(s / 'flock.stderr').read_text()[-400:]}")
            time.sleep(0.01)
        t0 = time.perf_counter()
        (s / "start").touch()
        tw = {}

        tr = None if not live else stale_live(plink, a.neg_stale_round) if a.neg_stale_round is not None else LiveTranscript(plink)

        def exchange(root_f: bytes, s=s, tw=tw, tr=tr) -> LK.Context:
            tx = time.perf_counter()
            if tr is not None:
                tw["prime_rounds_before_root_f"] = len(tr.rounds)
                tw["prime_wait_before_root_f_s"] = tr.wait_s
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
        torch.cuda.synchronize(dev)
        tp = time.perf_counter()
        units, epi = witness()
        torch.cuda.synchronize(dev)
        t_wit = time.perf_counter() - tp
        inst = prover.Instance([prover.Segment("unit", uc, ul, units, uc.hash), prover.Segment("epilogue", ec, el, epi, ec.hash)], ch,
                               link=LK.Link(lay, exchange=exchange, x=X, w=Wn))
        try:
            proof, st = prover.prove(inst, True, transcript=tr)
        except BaseException:
            # cell-prove would wait for root_F (or for the prime socket) for an hour
            fp.kill()
            raise
        finally:
            if plink is not None:
                plink.close()
        tb = time.perf_counter()
        data = proof.to_bytes()
        t_bytes = time.perf_counter() - tb
        torch.cuda.synchronize(dev)
        t_prime = time.perf_counter() - tp
        rc = fp.wait()
        t_cell = time.perf_counter() - t0
        fl.close()
        fe.close()
        ctx = inst.link.context
        (s / "proof.bin").write_bytes(data)
        (s / "link.txt").write_text(ctx.text())
        before = tw.get("prime_rounds_before_root_f", 0)
        if tr is not None:
            # the prover's view of the rounds (the evidence is the verifier's record: tools/cell_gate.py writes it from there)
            (s / "prime_coins.txt").write_text(coins_text(tr.rounds, before))
        flive = [json.loads(ln.split("\t", 1)[1]) for ln in (s / "flock.jsonl").read_text().splitlines() if ln.startswith("LIVE\t")]
        vinst = prover.Instance([prover.Segment("unit", uc, ul, units, uc.hash), prover.Segment("epilogue", ec, el, epi, ec.hash)], ch,
                                link=LK.Link(lay, context=LK.parse_context((s / "link.txt").read_text())))
        tv = time.perf_counter()
        try:
            prover.verify(vinst, proof, True, transcript=ReplayTranscript(tr.rounds, before) if tr is not None else None)
            py = "accept"
        except prover.VerifyError as e:
            py = f"reject ({str(e)[:160]})"
        t_py = time.perf_counter() - tv
        fls = flive[0] if flive else {}
        # the prover's time: both provers run at once from the start signal; the Flock prover's includes its live-coin
        # round trips (wait_s), the prime prover's the wait for the session's points (exchange_wait_s)
        t_flock = float(fls.get("prove_total_s", 0.0)) + float(fls.get("y_s", 0.0))
        # one connection carries both provers' rounds: its counters are the session's; the proxy's are the prime part
        calls, wait = int(fls.get("calls", 0) or 0), float(fls.get("wait_s", 0.0) or 0.0)
        px = fls.get("prime") or {}
        p_rounds, p_wait_px = int(px.get("rounds") or 0), float(px.get("wait_s") or 0.0)
        vv = (fls.get("verdict") or {})
        net = {"rounds": calls, "bytes_up": fls.get("wire_sent"), "bytes_down": fls.get("wire_recv"),
               "flock_rounds": calls - p_rounds, "flock_wait_s": wait - p_wait_px, "in_session_rtt_ms": 1e3 * wait / calls if calls else None,
               "rtt_probe_ms": a.rtt_ms, "rtt_probe_method": a.rtt_method,
               "flock_prover_compute_s": float(fls.get("prove_total_s", 0.0)) - (wait - p_wait_px),
               "verifier_compute_s": vv.get("verify_s"),
               "session_wall_s": fls.get("session_wall_s"),
               "prime_coins": a.prime_coins, "prime_rounds": p_rounds, "prime_challenges": px.get("challenges"),
               "prime_rounds_before_root_f": tw.get("prime_rounds_before_root_f"),
               "prime_bytes_up": px.get("bytes_up"), "prime_bytes_down": px.get("bytes_down"),
               # the prime prover's time blocked on its rounds (socket + forward + RTT), and the part before root_F: that
               # part delays root_F, so Flock's Commit and every Flock round after it (the critical path)
               "prime_wait_s": tr.wait_s if tr is not None else 0.0, "prime_wait_before_root_f_s": tw.get("prime_wait_before_root_f_s", 0.0),
               "prime_forward_wait_s": p_wait_px, "prime_refused": px.get("refused"),
               "recorded_prime_rounds": vv.get("prime_rounds"), "recorded_prime_before_commit": vv.get("prime_before_commit")}
        row = {"session": i, "warmup": i < a.warmup, "verifier": addr, "t_total": max(t_prime, t_flock), "t_cell_s": t_cell, "t_prime_prove_s": t_prime,
               "t_flock_prove_s": t_flock, "t_link_s": st.t_link, "net": net,
               "buckets": {"t.witness": t_wit + st.t_witness_wires, "t.encoding_commitment": st.t_commit,
                           "t.arithmetic": st.t_arith + st.t_open_acc + st.t_open_wq, "t.lookup": st.t_mults + st.t_lookup,
                           "t.zk_additional": 0.0, "t.serialization": st.t_open_cols + t_bytes},
               "prime": {"ligero_rows": st.ligero_rows, "linkage_constraints": st.linkage_constraints, "committed_elements": st.committed_elements,
                         # live coins serialise the whole transcript: every round is sequential (the parallel depth is the FS figure)
                         "sequential_depth": st.live_rounds if tr is not None else st.sequential_depth(True), "parallel_depth": st.sequential_depth(True),
                         "live_rounds": st.live_rounds, "slots": st.slots,
                         "peak_mem_bytes": st.peak_mem},
               "flock_rc": rc, "flock": flive[0] if flive else None, "python_verify": py, "python_verify_s": t_py,
               "proof_bytes": len(data), "proof_sha256": hashlib.sha256(data).hexdigest(), **tw}
        if a.rust:
            vd = s / "verify"
            shutil.copytree(sd, vd, dirs_exist_ok=True)
            shutil.copyfile(s / "link.txt", vd / "link.txt")
            (vd / "prime_coins.txt").unlink(missing_ok=True)
            if tr is not None:
                shutil.copyfile(s / "prime_coins.txt", vd / "prime_coins.txt")
            r = subprocess.run([a.rust, "verify", "--dir", str(vd), "--proof", str(s / "proof.bin"), "--vus", str(n), "--threads", str(a.threads),
                                "--relation", meta["relation"], "--json", str(s / "rust.json")]
                               + (["--require-live-coins"] if tr is not None else [])
                               # a pinned commitment is checked either way (a pin that does not match refuses); the flag only
                               # waives a missing pin, and the verdict records whether it was pinned
                               + (["--allow-unpinned-commitment"] if n != 4096 or meta.get("K", K_FROZEN) != K_FROZEN else []),
                               capture_output=True, text=True)
            doc = json.loads((s / "rust.json").read_text()) if (s / "rust.json").is_file() else {"error": r.stderr[-300:]}
            row["rust"] = {k: doc.get(k) for k in ("accepted", "error", "verify_seconds", "t_link", "circuit_pinned", "operands_committed",
                                                   "prime_coins", "prime_rounds", "prime_before_commit")}
            row["rust"]["commitment_pinned"] = (doc.get("commitment") or {}).get("pinned")
        (s / "times.json").write_text(json.dumps(row, indent=1))
        fl_ok = bool(flive and flive[0].get("accepted"))
        print(f"session {i}{' (warm-up)' if i < a.warmup else ''}: cell {t_cell:.3f} s, prime prove {t_prime:.3f} s (link {st.t_link:.3f}, "
              f"wait {tw.get('exchange_wait_s', 0):.3f}), flock {flive[0]['prove_total_s'] if flive else None} s accepted {fl_ok}, "
              f"python {py}" + (f", rust {row['rust']}" if a.rust else ""), flush=True)
        if i == a.warmup:
            from gpu.circuit import Circuit  # noqa: F401
            import soundness
            budget = soundness.summary(soundness.budget(inst, st.ligero_rows, st.linkage_constraints, K=K), live=tr is not None)
            (cell / "prime_soundness.json").write_text(json.dumps(budget, indent=1, default=str))
        out.append(row)
        del inst, vinst, proof
    if guard is not None:
        (cell / "contention.json").write_text(json.dumps(guard.stop(), indent=1))
    (cell / "sessions.json").write_text(json.dumps(out, indent=1))
    if a.result:
        result(a, cell, meta, out)
    ok = all(r["python_verify"] == "accept" and r["flock_rc"] == 0 and r["flock"] and r["flock"].get("accepted")
             and (not a.rust or r["rust"].get("accepted")) for r in out)
    print("CELL-PROVE OK" if ok else "CELL-PROVE FAILED", flush=True)
    return 0 if ok else 1


# The Flock side's terms of the composed bound (red-team-flock third audit, note:red-team-flock/20260925T1107Z, "Composed
# bound"): flock-128-r2 at dense m = 33 (two Fast100 reps, live coins) and the link claims' reduction in both reps.
FLOCK_TERMS_LOG2 = {"flock-128-r2 (two reps, live coins)": -195.5, "link claims' reduction, both reps": -243.9}


def result(a, cell: Path, meta: dict, rows: list[dict]) -> None:
    import math
    import platform
    import statistics

    import numpy as np
    import torch
    import triton

    from verity_numerical.bench import contract

    import bench_result as br

    timed = [r for r in rows if not r["warmup"]]
    n, K = meta["vus"], int(meta.get("K", K_FROZEN))
    prime = json.loads((cell / "prime_soundness.json").read_text())
    terms = {"A-GKR prime proof (incl. the link's sigma batching and Schwartz-Zippel over two GF(2^128) points)": prime["achieved_log2"],
             **FLOCK_TERMS_LOG2}
    ach = math.log2(sum(2.0 ** v for v in terms.values()))
    ok = all(r["python_verify"] == "accept" and r["flock"] and r["flock"].get("accepted") and (r.get("rust") or {}).get("accepted")
             for r in timed)
    rust = [r["rust"]["verify_seconds"] for r in timed if (r.get("rust") or {}).get("verify_seconds") is not None]
    # NON_ZK_PROOF only when every timed session's prime coins were the live verifier's and were replayed as such
    live = all((r.get("net") or {}).get("prime_coins") == "live" and (r.get("rust") or {}).get("prime_coins") == "live" for r in timed)
    mode = ("interactive (one live session: every A-GKR challenge -- the GKR and LogUp coins, the sigma batching rho, the "
            "functional and Ligero row coefficients and the Ligero query indices -- is the live verifier's, issued after the prime "
            "transcript state that commits the messages before it; Flock: live verifier coins; the link points: the live verifier's OS "
            "coins after root_F and root_B)") if live else (
            "prime half Fiat-Shamir (the prover's own SHA-256 transcript: runner-derived coins, not transferable); Flock: live verifier "
            "coins; the link points: the live verifier's OS coins after root_F and root_B")
    security = {"target": -128, "achieved_log2": ach, "achieved_bits": -ach, "field": "BabyBear^6 (prime side); GF(2^128) (Flock, the link)",
                "mode": mode, "prime_coins": "live" if live else "fiat-shamir",
                "composed_terms_log2": terms, "prime_by_component_log2": prime.get("by_component_log2"),
                "hash_assumption": "collision resistance of SHA-512 (A-GKR Merkle), SHA-256 (Flock, transcripts), BLAKE3 (row leaves)",
                "union_over": 1, "source": "archive/gkr/soundness.py (prime) + red-team-flock third audit (Flock terms)"}
    sw = {"backend": {"name": br.BACKEND + " + Flock flock-128-r2 (verity/ml/flock/live flock-link, CPU) + the sigma link (route (a))",
                      "commit": br.source_commit(), "relation": meta["relation"], "scheme": "frame-v3 blake3-keyed/row/v2",
                      "version": "route (a): gpu/link.py session-2xgf2_128/v2, flock-link Σ verity/link-L/v2"},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "triton": triton.__version__, "numpy": np.__version__}}
    rel = meta["relation"]
    if "instances" in meta:
        inst = meta["instances"]
    else:                   # a statement written before cell.json carried its instances entry: the frozen set, untiled
        inst = None if n > 4096 else contract.instances_ref(br.TIER, 0, n, meta.get("manifest_sha256") or
            hashlib.sha256((HERE.parents[1] / "fixtures" / "bench-instances" / "v1" / "manifest.json").read_bytes()).hexdigest())
    fp = contract.fingerprint(
        profile=contract.PROFILE, domain=contract.DOMAIN, K=K, B=n, N_subbatches=1,
        proof_class="NON_ZK_PROOF" if live else "NON_ZK_PROOF_DIAGNOSTIC",
        authentication="included-hash", zk_mode=contract.ZK_NONE, security=security, hardware=contract.probe_hardware(), software=sw,
        preprocessing=("circuit files, GKR layers, Lambda_F derivation, the Flock statement shape (untimed; the Flock prover starts on a "
                       "signal once built); Triton compilation in the untimed warm-up session"),
        retention="retained", instances=inst, instances_tiled=meta.get("tiled_from"), relation=rel,
        scope="vu", candidate="A-GKR", checker="v2", field=br.FIELD, variant="A-GKR route (a): A-GKR + Flock + link",
        leaf="blake3", sharing=None, sessions=len(timed), warmup=a.warmup,
        verifier={"flock": "flock-link cell-serve (live coins; the session record is the evidence, admitted by tools/cell_gate.py)",
                  "prime": f"verity-gkr-verify --relation {rel} --require-live-coins against the record's link.txt and prime "
                           "rounds (pinned circuit set, pinned commitment)" if live else
                           f"verity-gkr-verify --relation {rel} against the record's link.txt (pinned circuit set, pinned commitment)"},
        note=(f"one A-GKR proof of all {n} VUs ({base_relation(K)}, operands committed under frame-v3 blake3-keyed/row/v2) linked to one Flock "
              f"flock-128-r2 session proving the keyed-BLAKE3 row leaves; both provers run at once on the prover pod (A-GKR on the GPU, "
              f"Flock on the host CPU), t.total = max of the two from the start signal; the prime buckets are the median session's, "
              f"the Flock prover's time is in Other when it is the longer; serving commit {meta.get('serving_commit_seconds', 0):.2f} s "
              f"(CPU reference committer) outside t.total"),
    )
    meas, med, tt, ix = session_measurements(contract, timed, n, live, a.rtt_ms, a.rtt_method, -ach, meta.get("serving_commit_seconds", 0.0),
                                             rust, K)
    fp["interaction"] = ix
    res = {"schema": contract.RESULT_SCHEMA, "run_id": os.environ.get("RESEARCH_RUN_ID"), "kind": "bench-result/v1", "workload_fingerprint": fp,
           "validation": {"status": "passed" if ok else "failed",
                          "checker": "per session: the live Flock verifier's verdict, gpu.prover.verify and verity-gkr-verify against the session's link",
                          "detail": f"{len(timed)} timed sessions; the independent evidence is the non-producer verifier's records (tools/cell_gate.py)",
                          "evidence": {"sessions": timed}},
           "measurements": meas, "artifacts": [f"out/cell-{n}/s{r['session']}/proof.bin" for r in timed]}   # where 10-cell.sh keeps them in the run
    res["contract_problems"] = contract.validate(res)
    Path(a.result).write_text(json.dumps(res, indent=1, default=str))
    print(f"result {a.result}: t.total {tt:.3f} s, bound 2^{ach:.1f}, problems {res['contract_problems']}", flush=True)


def critical_wait(r: dict) -> float:
    """The network wait on a session's critical path.  Flock is the longer prover: its Commit waits for root_F, so the
    prime rounds before root_F are on its path, then its own rounds; else the prime prover's own rounds."""
    n = r.get("net") or {}
    if r["t_flock_prove_s"] >= r["t_prime_prove_s"]:
        return float(n.get("prime_wait_before_root_f_s") or 0.0) + float(n.get("flock_wait_s") or 0.0)
    return float(n.get("prime_wait_s") or 0.0)


def session_measurements(contract, timed: list[dict], n: int, live: bool, rtt_ms, rtt_method, soundness_bits: float,
                         serving_s: float, rust: list[float], K: int = K_FROZEN):
    """The envelope's measurements from the timed sessions.  The contract's ``t.total`` is the prover without its network
    waits and ``t.total_live`` the wall with them; the median session is the median ``t.total``.  ``rounds.sequential_depth``
    is every round of the session (one connection; the live prime transcript is sequential, and Flock's rounds follow
    root_F), ``net.*`` the session's.  Returns (measurements, the median session, its t.total, the interaction block)."""
    import statistics

    import bench_result as br

    comp = lambda r: r["t_total"] - critical_wait(r)   # noqa: E731
    med = sorted(timed, key=comp)[len(timed) // 2]
    tt, nm = comp(med), med.get("net") or {}
    flock_bytes = sum((med["flock"] or {}).get("proof_bytes", []))
    meas = [contract.measurement(k, v) for k, v in med["buckets"].items()]
    meas += [contract.measurement("t.total", tt), contract.measurement("t.total_live", med["t_total"]),
             contract.measurement("transcript.bytes", med["proof_bytes"] + flock_bytes),
             contract.measurement("vu.count", n), contract.measurement("vu.seconds_per_vu", tt / n), contract.measurement("vu.per_second", n / tt),
             *contract.rate_measurements(tt, n, K, contract.PROFILE),
             br._m("t_total_min", min(comp(r) for r in timed), "s"), br._m("t_total_max", max(comp(r) for r in timed), "s"),
             br._m("t_total_live_min", min(r["t_total"] for r in timed), "s"), br._m("t_total_live_max", max(r["t_total"] for r in timed), "s"),
             br._m("t_prime_seconds", med["t_prime_prove_s"], "s"), br._m("t_flock_seconds", med["t_flock_prove_s"], "s"),
             br._m("t_session_wall_seconds", med["t_cell_s"], "s"), br._m("soundness_bits", soundness_bits, "bits"),
             br._m("commit.serving_seconds", serving_s, "s")]
    ix = None
    if nm:
        meas += [contract.measurement("rounds.sequential_depth", nm["rounds"]), contract.measurement("rounds.protocol", nm["rounds"]),
                 contract.measurement("net.messages", nm["rounds"]), contract.measurement("net.wait_seconds", critical_wait(med)),
                 contract.measurement("net.bytes_out", nm["bytes_up"]), contract.measurement("net.bytes_in", nm["bytes_down"]),
                 br._m("session.flock_rounds", nm["flock_rounds"], "count"), br._m("net.rtt_in_session_ms", nm["in_session_rtt_ms"], "ms"),
                 br._m("flock.prover_compute_seconds", nm["flock_prover_compute_s"], "s"), br._m("flock.network_wait_seconds", nm["flock_wait_s"], "s"),
                 br._m("flock.verifier_compute_seconds", nm["verifier_compute_s"], "s")]
        if live:
            meas += [br._m("session.prime_rounds", nm["prime_rounds"], "count"),
                     br._m("session.prime_rounds_before_root_f", nm["prime_rounds_before_root_f"], "count"),
                     br._m("session.prime_bytes_up", nm["prime_bytes_up"], "bytes"), br._m("session.prime_bytes_down", nm["prime_bytes_down"], "bytes"),
                     br._m("prime.network_wait_seconds", nm["prime_wait_s"], "s"), br._m("prime.wait_before_root_f_seconds", nm["prime_wait_before_root_f_s"], "s")]
        if rtt_ms is not None:
            meas.append(contract.measurement("net.rtt_ms", rtt_ms))
        ix = {"rtt_method": rtt_method, "rounds": (f"{nm['rounds']} = {nm.get('prime_rounds', 0)} prime ({nm.get('prime_rounds_before_root_f', 0)} before "
                                                   f"root_F) + {nm['flock_rounds']} Flock, one connection, all sequential"),
              "wait": "prime rounds before root_F + Flock's rounds (Flock is the longer prover and its Commit waits for root_F)",
              "t_total": "t.total_live (wall from the start signal) - net.wait_seconds"}
    if rust:
        meas.append(contract.measurement("verifier.seconds", statistics.median(rust)))
    return meas, med, tt, ix


def rederive(src: Path, out: Path, supersedes: str | None, protocol: dict | None = None, sweep: dict | None = None,
             commit: str | None = None, loopback: tuple[dict, str] | None = None) -> dict:
    """A new envelope from an old one's embedded sessions (no pod): the measurements and the interaction block again, from
    the same runs; the fingerprint, hardware and evidence are kept.  ``protocol`` / ``sweep`` (JSON documents) become the
    fingerprint's blocks of those names; ``commit`` names the committer whose time (the envelope's ``commit.serving_seconds``)
    is reported as ``commit.seconds``, the commitment P counts.  ``loopback`` = (the times.json of a session against a verifier
    on 127.0.0.1 in the same run, its description): ``live.loopback_round_seconds`` = its critical-path coin wait / its rounds."""
    for p in (HERE.parents[1], HERE.parents[1] / "benchmarks" / "numerical" / "python"):
        sys.path.insert(0, str(p))
    from verity_numerical.bench import contract

    old = json.loads(src.read_text())
    timed = old["validation"]["evidence"]["sessions"]
    for r in timed:
        pr = r.get("prime") or {}
        if (r.get("net") or {}).get("prime_coins") == "live" and pr.get("live_rounds"):
            pr.setdefault("parallel_depth", pr.get("sequential_depth"))
            pr["sequential_depth"] = pr["live_rounds"]
    om = {m["name"]: m["value"] for m in old["measurements"]}
    fp = old["workload_fingerprint"]
    live = fp.get("security", {}).get("prime_coins") == "live"
    rust = [r["rust"]["verify_seconds"] for r in timed if (r.get("rust") or {}).get("verify_seconds") is not None]
    rtt = om.get("net.rtt_ms", om.get("net.rtt_probe_ms"))
    method = next((r["net"].get("rtt_probe_method") for r in timed if (r.get("net") or {}).get("rtt_probe_method")), None)
    meas, med, tt, ix = session_measurements(contract, timed, fp["B"], live, rtt, method, om["soundness_bits"], om.get("commit.serving_seconds", 0.0),
                                             rust, int(fp.get("K", K_FROZEN)))
    fp = dict(fp, interaction=ix)
    if protocol is not None:
        fp["protocol"] = protocol
    if sweep is not None:
        fp["sweep"] = sweep
    if loopback is not None:
        row, where = loopback
        if not str(row.get("verifier", "")).startswith("127.0.0.1") or not (row.get("net") or {}).get("rounds"):
            raise SystemExit("rederive --loopback-times: not a loopback session with rounds")
        per = critical_wait(row) / row["net"]["rounds"]
        meas.append(contract.measurement("live.loopback_round_seconds", per) if "live.loopback_round_seconds" in contract.UNITS
                    else {"name": "live.loopback_round_seconds", "value": per, "unit": "s"})
        fp["interaction"] = dict(fp["interaction"] or {}, t_total_includes_wait=False, loopback_method=(
            f"loopback probe in this run: {where} (verifier on 127.0.0.1, same pod, binary and batch), its critical-path coin wait "
            f"(prime rounds before root_F + Flock's rounds, {critical_wait(row):.3f} s) / its {row['net']['rounds']} rounds; one session, a mean"))
    if commit is not None:
        cs = om.get("commit.serving_seconds")
        if not isinstance(cs, (int, float)) or cs <= 0:
            raise SystemExit("rederive --commit: the envelope has no commit.serving_seconds")
        import bench_result as br
        meas.append(br._m("commit.seconds", cs, "s"))
        fp["commit"] = {"seconds": cs, "committer": commit, "in_t_total": False}
    new = dict(old, workload_fingerprint=fp, measurements=meas,
               derived_from={"artifact": supersedes, "run_id": old.get("run_id"), "tool": "archive/gkr/tools/cell.py rederive",
                             "why": "t.total without the network waits (t.total_live the wall), rounds.sequential_depth = the session's "
                                    "rounds (the prime half's live transcript is sequential: 328 was its parallel Fiat-Shamir depth), net.*"})
    new["contract_problems"] = contract.validate(new)
    out.write_text(json.dumps(new, indent=1, default=str))
    print(f"rederived {out}: t.total {tt:.3f} s (t.total_live {med['t_total']:.3f} s), rounds {(med.get('net') or {}).get('rounds')}, "
          f"problems {new['contract_problems']}", flush=True)
    return new


SWEEP_RULE = ("double the batch from the first size until two successive doublings together raise the throughput (VUs / t.total) "
              "by < 2 %, a size fails (the prime prover runs out of GPU memory) or the input set is exhausted (a set is not "
              "recycled); plateau = the size with the highest throughput (kb/TABLES.md)")
COMMITTER = ("CPU reference committer (archive/gkr/gpu/commit.py: frame-v3 + blake3-keyed/row/v2 over the batch's operands and "
             "words), one pass before the sessions; a GPU committer is not measured")


def _meas(doc: dict) -> dict:
    return {m["name"]: m.get("value") for m in doc.get("measurements") or [] if isinstance(m, dict) and "name" in m}


def _set(doc: dict, name: str, value, unit: str) -> None:
    ms = doc.setdefault("measurements", [])
    for m in ms:
        if m.get("name") == name:
            m["value"] = value
            return
    ms.append({"name": name, "value": value, "unit": unit})


def finish(points: list[Path], verifier: str, operator: str | None, sweep_id: str) -> tuple[dict, list[dict]]:
    """``(the cell's result, the other sizes' results)`` from a sweep's point directories (``cell.sh``): each holds
    ``status.json`` (``vus``, ``rc``, ``why``, ``verifier``: the remote session address that size used) and, for a size that ran,
    ``result.json`` (``prove --result``), ``loopback-times.json`` (its warm-up session, against a verifier on 127.0.0.1 in the
    same pod) and ``contention.json`` (the timing guard over its timed sessions).  The plateau size's result gains the sweep,
    protocol, loopback and commit blocks and the remote verifier's placement."""
    from verity_numerical.bench import contract

    rows = []
    for d in points:
        st = json.loads((d / "status.json").read_text())
        doc = json.loads((d / "result.json").read_text()) if (d / "result.json").is_file() else None
        good = st.get("rc") == 0 and doc is not None and (doc.get("validation") or {}).get("status") == "passed"
        rows.append({"vus": int(st["vus"]), "rc": st.get("rc"), "why": st.get("why"), "dir": d, "doc": doc, "verifier": st.get("verifier") or verifier,
                     "vu_per_second": int(st["vus"]) / float(_meas(doc)["t.total"]) if good else None})
    rows.sort(key=lambda r: r["vus"])
    ok = [r for r in rows if r["vu_per_second"] is not None]
    if not ok:
        raise SystemExit(f"finish: no size ran ({[(r['vus'], r['rc'], r['why']) for r in rows]})")
    best = max(ok, key=lambda r: (r["vu_per_second"], -r["vus"]))
    tps = [r["vu_per_second"] for r in ok]
    last = rows[-1]
    stopped = (f"{last['vus']} VUs failed (rc {last['rc']}: {last['why']})" if last["vu_per_second"] is None else
               "< 2 % over two doublings" if len(tps) >= 3 and tps[-1] < 1.02 * tps[-3] else
               f"the input set or the sweep's last size ({last['vus']} VUs)")
    sweep = {"id": sweep_id, "point": ok.index(best), "plateau": True, "total_vus": best["vus"], "rule": SWEEP_RULE, "stopped": stopped,
             "points": [[r["vus"], r["vu_per_second"]] for r in rows],
             "failures": [{"vus": r["vus"], "rc": r["rc"], "why": r["why"]} for r in rows if r["vu_per_second"] is None]}
    others = []
    for r in ok:
        doc, fp = r["doc"], r["doc"]["workload_fingerprint"]
        guard = json.loads((r["dir"] / "contention.json").read_text()) if (r["dir"] / "contention.json").is_file() else {}
        external = [x for x in guard.get("reasons") or [] if not x.startswith("CPU throttled")]
        timed = [s for s in doc["validation"]["evidence"]["sessions"] if not s["warmup"]]
        doc["protocol"] = {
            "warm": True, "warmup": "1 local warm-up session per size in the same process (loopback verifier; Triton kernels compiled)",
            "runs": len(timed), "statistic": "median", "contended": bool(external) if guard else None,
            "contention": {**guard, "rule": (
                "the timing guard (archive/direct/ligero/timing_guard.py) over the timed sessions: contended on another GPU compute "
                "process or a load above the host's cores; its CPU-throttling rule is recorded, not applied: the cell's own Flock "
                "prover runs on every core of the pod's quota beside the GPU prover by design (route-a-live, art:77411c93)")}}
        doc["sweep"] = dict(sweep, point=ok.index(r), plateau=r is best, total_vus=r["vus"])
        cs = _meas(doc).get("commit.serving_seconds")
        if isinstance(cs, (int, float)) and cs > 0:
            _set(doc, "commit.seconds", cs, "s")
            fp["commit"] = {"seconds": cs, "committer": COMMITTER, "in_t_total": False}
        inter = dict(fp.get("interaction") or {})
        lt = r["dir"] / "loopback-times.json"
        if lt.is_file():
            row = json.loads(lt.read_text())
            if str(row.get("verifier", "")).startswith("127.0.0.1") and (row.get("net") or {}).get("rounds"):
                per = critical_wait(row) / row["net"]["rounds"]
                _set(doc, "live.loopback_round_seconds", per, "s")
                inter.update(t_total_includes_wait=False, loopback_method=(
                    f"loopback probe in this run: the size's warm-up session (verifier on 127.0.0.1, same pod, binary and batch), its "
                    f"critical-path coin wait (prime rounds before root_F + Flock's rounds, {critical_wait(row):.3f} s) / its "
                    f"{row['net']['rounds']} rounds; one session, a mean"))
        fp["verifier"] = {**(fp.get("verifier") or {}), "placement": r["verifier"], "operator": operator,
                          "process": "flock-link cell-serve on a separate pod (serves the Flock and the prime coins, one statement per size)"}
        fp["interaction"] = doc["interaction"] = inter
        doc["contract_problems"] = contract.validate(doc)
        if r is not best:
            others.append(doc)
    return best["doc"], others


def _tree_sha(run: Path) -> str | None:
    try:
        return (json.loads((run / "job.json").read_text()).get("computation") or {}).get("tree_sha")
    except (OSError, ValueError):
        return None


def register(prover_run: Path, verifier_run: Path | None, lane: str) -> dict:
    """The prover run's cell result with the producing lane, the shipped tree's commit, and the verifier run's session verdicts."""
    doc = json.loads((prover_run / "result.json").read_text())
    doc["lane"] = doc["registered_by"] = lane
    fp, sha = doc.get("workload_fingerprint") or {}, _tree_sha(prover_run)
    be = (fp.get("software") or {}).get("backend")
    if sha and isinstance(be, dict) and not be.get("commit"):
        be["commit"] = sha
    if verifier_run is not None:
        sessions = []
        for f in sorted(verifier_run.rglob("index.jsonl")):
            sessions += [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        doc.setdefault("validation", {}).setdefault("evidence", {})["verifier_run"] = {
            "run": verifier_run.name, "tree_sha": _tree_sha(verifier_run), "sessions": len(sessions),
            "accepted": sum(1 for x in sessions if x.get("accepted")),
            "note": "the verifier pod's session indexes (flock-link cell-serve --out), recorded by its research run"}
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools/cell.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("circuits")
    c.add_argument("--out", required=True)
    c.add_argument("--k", type=int, default=K_FROZEN)
    s = sub.add_parser("statement")
    s.add_argument("--cell", required=True)
    s.add_argument("--instances", default=None, help="the frozen bench-instances/v1 data directory (K = 1536)")
    s.add_argument("--input-set", default=None, help="an input set directory (VUs [0, --vus), its K)")
    s.add_argument("--vus", type=int, default=4096)
    s.add_argument("--flock", default=None)
    s.add_argument("--verifier-side", action="store_true", help="only the leaf digests and cell.json (the verifier's statement)")
    f = sub.add_parser("finish", help="the cell's result from a sweep's point directories (cell.sh)")
    f.add_argument("--points", nargs="+", required=True)
    f.add_argument("--verifier", required=True, help="HOST:PORT of the remote verifier (a size's status.json may name its own)")
    f.add_argument("--operator", default=None)
    f.add_argument("--sweep-id", default=None)
    f.add_argument("--out", required=True)
    f.add_argument("--others", default=None, help="write the other sizes' results here (outputs.json, bench-result/v1)")
    g = sub.add_parser("register", help="the prover run's cell result stamped for research data put")
    g.add_argument("--prover-run", required=True)
    g.add_argument("--verifier-run", default=None)
    g.add_argument("--lane", required=True)
    g.add_argument("--out", required=True)
    p = sub.add_parser("prove")
    p.add_argument("--cell", required=True)
    p.add_argument("--flock", required=True)
    p.add_argument("--verifier", required=True)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--rust", default=None)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--warmup-verifier", default=None, help="the warm-up sessions' verifier (a loopback one: warm-ups are never evidence)")
    p.add_argument("--result", default=None, help="write the bench-result/v1 envelope here")
    p.add_argument("--rtt-ms", type=float, default=None, help="the prover-to-verifier RTT measured before the sessions")
    p.add_argument("--rtt-method", default=None)
    p.add_argument("--neg-stale-round", type=int, default=None,
                   help="negative: commit a stale prime state at this round (the session record then fails replay)")
    p.add_argument("--prime-coins", choices=("live", "fs"), default="live",
                   help="live: the session's verifier issues every prime challenge (NON_ZK_PROOF); fs: the prime half's own Fiat-Shamir (diagnostic)")
    d = sub.add_parser("rederive", help="a new envelope from an old one's embedded sessions (no pod)")
    d.add_argument("--from", dest="src", required=True)
    d.add_argument("--result", required=True)
    d.add_argument("--supersedes", default=None, help="the old envelope's art: id")
    d.add_argument("--protocol", default=None, help="JSON file: the fingerprint's protocol block (warm, runs, statistic, contended, contention)")
    d.add_argument("--sweep", default=None, help="JSON file: the fingerprint's sweep block (id, point, plateau, points)")
    d.add_argument("--loopback-times", default=None, help="times.json of the same run's loopback session (live.loopback_round_seconds)")
    d.add_argument("--loopback-where", default="the warm-up session", help="which session --loopback-times is, for loopback_method")
    d.add_argument("--commit", default=None, help="the committer whose time (commit.serving_seconds) is reported as commit.seconds")
    r = sub.add_parser("result", help="the bench-result/v1 envelope from CELL/sessions.json (on the prover pod: it probes the hardware)")
    r.add_argument("--cell", required=True)
    r.add_argument("--sessions-json", action="append", default=[], help="more sessions.json files (other runs of the same cell)")
    r.add_argument("--warmup", type=int, default=1)
    r.add_argument("--result", required=True)
    r.add_argument("--rtt-ms", type=float, default=None)
    r.add_argument("--rtt-method", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "rederive":
        rd = lambda f: json.loads(Path(f).read_text()) if f else None   # noqa: E731
        lb = (rd(a.loopback_times), a.loopback_where) if a.loopback_times else None
        rederive(Path(a.src), Path(a.result), a.supersedes, rd(a.protocol), rd(a.sweep), a.commit, lb)
        return 0
    if a.cmd == "result":
        cell = Path(a.cell)
        rows = [r for f in [cell / "sessions.json", *map(Path, a.sessions_json)] for r in json.loads(Path(f).read_text())]
        result(a, cell, json.loads((cell / "cell.json").read_text()), rows)
        return 0
    if a.cmd == "circuits":
        Path(a.out).mkdir(parents=True, exist_ok=True)
        circuits(Path(a.out), a.k)
        return 0
    if a.cmd == "statement":
        if (a.instances is None) == (a.input_set is None):
            ap.error("statement: --instances (the frozen set) or --input-set")
        if a.flock is None and not a.verifier_side:
            ap.error("statement: --flock (sigma.txt), unless --verifier-side")
        statement(Path(a.cell), a.instances and Path(a.instances), a.vus, a.flock, a.input_set and Path(a.input_set), a.verifier_side)
        return 0
    if a.cmd == "finish":
        import datetime
        sid = a.sweep_id or f"agkr-route-a sweep {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%MZ} ({os.environ.get('RESEARCH_RUN_ID')})"
        best, others = finish([Path(p) for p in a.points], a.verifier, a.operator, sid)
        Path(a.out).write_text(json.dumps(best, indent=1, default=str))
        if a.others:
            outs = [{"name": f"sweep-{d['workload_fingerprint']['B']}", "kind": "bench-result/v1",
                     "meta": dict(d, label=f"{a.operator} {d['workload_fingerprint'].get('relation')} sweep point {d['workload_fingerprint']['B']} VUs")}
                    for d in others]
            Path(a.others).write_text(json.dumps({"schema": "research/outputs/v0.1", "outputs": outs}, indent=1, default=str))
        from verity_numerical.bench import views
        ix = views.interaction(best)
        print(json.dumps({"out": a.out, "sweep": best["sweep"], "protocol": {k: v for k, v in best["protocol"].items() if k != "contention"},
                          "interaction_problem": views.interaction_problem(ix), "contract_problems": best.get("contract_problems")},
                         indent=1, default=str))
        return 0
    if a.cmd == "register":
        doc = register(Path(a.prover_run), a.verifier_run and Path(a.verifier_run), a.lane)
        Path(a.out).write_text(json.dumps(doc, indent=1, default=str))
        return 0
    return prove(a)


if __name__ == "__main__":
    sys.exit(main())
