"""prover.py: the slice's prover. One o_proj served by the GPU engine (slice.cu), CALLS calls a round, through the challenger's
round protocol (challenger.py) over the loopback link:

    setup  B on the device; B's tree (commit.py), its root in hello, before any coin
    (a)    each call's A on the device and to a file; A's tree a call; the roots out
    (b)    the coin sigma in
    (c)    each call through the edited GEMM under call_key(sigma, c) and receipt_seed(sigma, c) (plain cuBLASLt
           interleaved for the baseline); a receipt tree a call; the roots out
    (d)    the draw in; the drawn tile's served product and receipts kept, its 16 xa and wb paths taken, the round's trees
           dropped
    proof  (a worker, a round at a time, overlapping the next rounds) the drawn tile recomputed from A, B and the coin alone
           (derive.tile_check, the honest prover's own check of its GPU), the block's statement staged (statement.Stager),
           the public file and the paths out, then `flock-circuit prove --zk` against the challenger's server

At most --max-pending rounds wait on their proofs before the next round's (a) goes out (a delay before (a) is free).
Writes rounds.jsonl (timings per round) and prover.json under --out.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import commit as CM  # noqa: E402
import derive  # noqa: E402
import statement as ST  # noqa: E402
from engine import Engine  # noqa: E402
from wire import connect  # noqa: E402


class Prover:
    def __init__(self, a):
        self.a = a
        self.scratch = Path(a.scratch)
        self.out = Path(a.out)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.pool = CM.pool(a.workers)
        self.session = os.urandom(8).hex()
        self.salt_key = os.urandom(32)
        self.jobs: queue.Queue = queue.Queue()
        self.pending = threading.Semaphore(a.max_pending)
        self.log = open(self.out / "rounds.jsonl", "w")
        self.log_lock = threading.Lock()
        self.records: dict[int, dict] = {}
        self.errors: list[str] = []

    def rec(self, w: int, **f) -> None:
        with self.log_lock:
            self.records.setdefault(w, {"round": w}).update(f)

    def setup(self) -> dict:
        a = self.a
        self.E = Engine(a.engine, open(self.out / "engine.err", "w"))
        t = time.time()
        self.setup_info = self.E.cmd("setup", a.m, a.n, a.k, a.bseed, self.scratch / "B.bin")
        self.B = np.fromfile(self.scratch / "B.bin", np.int8).reshape(a.n, a.k)
        t1 = time.time()
        self.tb = CM.commit_operand(self.pool, str(self.scratch / "B.bin"), (a.n, a.k), CM.label_b(self.session), self.salt_key)
        t2 = time.time()
        self.st = ST.Stager(self.scratch / "circuit.txt")
        t3 = time.time()
        self.wire = connect(a.port)
        self.wire.start_reader()
        self.wire.send("hello", m=a.m, n=a.n, k=a.k, calls=a.calls, session=self.session, b_root=self.tb.root.hex(),
                       b_count=self.tb.count, pin=self.st.pin)
        return {"engine": self.setup_info, "setup_s": round(t1 - t, 3), "b_commit_s": round(t2 - t1, 3),
                "b_commit_cpu_s": round(self.tb.cpu_s, 3), "b_leaves": self.tb.count, "stager_s": round(t3 - t2, 3),
                "circuit": self.st.stats()}

    def round(self, w: int) -> None:
        a, per = self.a, self.a.k // 32
        self.pending.acquire()
        t0 = time.time()
        apaths = [self.scratch / f"A-{w}-{c}.bin" for c in range(a.calls)]
        for c in range(a.calls):
            self.E.cmd("opnds", c, (w * a.calls + c) & 0xFFFFFFFF, apaths[c])
        t1 = time.time()
        trees = [CM.commit_operand(self.pool, str(apaths[c]), (a.m, a.k), CM.label_a(self.session, w, c), self.salt_key)
                 for c in range(a.calls)]
        t2 = time.time()
        self.wire.send("operands", round=w, roots=[t.root.hex() for t in trees])
        sigma = bytes.fromhex(self.wire.take("coin", w)["sigma"])
        t3 = time.time()
        gpu, rpaths = [], [self.scratch / f"R-{w}-{c}.bin" for c in range(a.calls)]
        for c in range(a.calls):
            gpu.append(self.E.cmd("call", c, *derive.sig_words(derive.call_key(sigma, c)), derive.receipt_seed(sigma, c),
                                  rpaths[c], a.fused))
        t4 = time.time()
        Rs = [np.fromfile(p, np.uint32).reshape(-1, derive.NCT, 8) for p in rpaths]
        tc = time.process_time()
        roots = [CM.receipt_root(R, self.session, w, c, self.salt_key).hex() for c, R in enumerate(Rs)]
        t5, receipt_cpu = time.time(), time.process_time() - tc
        self.wire.send("products", round=w, receipt_roots=roots, gpu=gpu)
        d = self.wire.take("draw", w)
        t6 = time.time()
        c, (tm, tn), j = d["call"], d["tile"], d["step"]
        ypath = self.scratch / f"Y-{w}.bin"
        self.E.cmd("tile", c, tm, tn, ypath)
        Y = np.fromfile(ypath, np.int32).reshape(derive.TM, derive.BN)
        ypath.unlink()
        R_tile = Rs[c][tm * (a.n // derive.BN) + tn].copy()
        coords = [tuple(x) for x in d["coords"]]
        open_ = {"xa": [[h.hex() for h in trees[c].path(i * per + j)] for i, _l in coords],
                 "wb": [[h.hex() for h in self.tb.path(l * per + j)] for _i, l in coords],
                 "salts": {"xa": [CM.leaf_salt(self.salt_key, trees[c].label, i * per + j) for i, _l in coords],
                           "wb": [CM.leaf_salt(self.salt_key, self.tb.label, l * per + j) for _i, l in coords]}}
        for p in rpaths:
            p.unlink()
        for cc, p in enumerate(apaths):
            if cc != c:
                p.unlink()
        commit_cpu = sum(t.cpu_s for t in trees)
        del trees, Rs
        t7 = time.time()
        self.rec(w, gpu=gpu, a_gen_s=round(t1 - t0, 3), a_commit_s=round(t2 - t1, 3), a_commit_cpu_s=round(commit_cpu, 3),
                 coin_wait_s=round(t3 - t2, 3), calls_s=round(t4 - t3, 3), receipt_commit_s=round(t5 - t4, 3),
                 receipt_cpu_s=round(receipt_cpu, 3), draw_wait_s=round(t6 - t5, 3), open_s=round(t7 - t6, 3), draw=d)
        self.jobs.put({"round": w, "sigma": sigma, "draw": d, "coords": coords, "apath": apaths[c], "Y": Y, "R": R_tile, **open_})

    def proof_worker(self) -> None:
        while (job := self.jobs.get()) is not None:
            try:
                self.prove(job)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"round {job['round']}: {type(e).__name__}: {e}")
                self.rec(job["round"], error=str(e)[:500])
            finally:
                self.pending.release()

    def prove(self, job: dict) -> None:
        a, w, d, sigma = self.a, job["round"], job["draw"], job["sigma"]
        c, (tm, tn) = d["call"], d["tile"]
        t0 = time.time()
        A = np.fromfile(job["apath"], np.int8).reshape(a.m, a.k)
        want_r, want_y, _U, _F = derive.tile_check(A, self.B, sigma, c, tm, tn, a.k)
        self_check = {"receipts": bool((job["R"] == want_r).all()), "product": bool((job["Y"] == want_y).all())}
        t1 = time.time()
        key = derive.call_key(sigma, c)
        U = derive.u_noise64(key, a.m)
        F = {l: derive.f_rows(key, l - l % 4, 4, a.k)[l % 4] for _i, l in job["coords"]}
        rd = self.scratch / f"r{w:04d}"
        draw = {"call": c, "step": d["step"], "coords": job["coords"]}
        self.st.stage(rd, A, self.B, U, F, draw, sigma, job["salts"], self.salt_key)
        job["apath"].unlink()
        t2 = time.time()
        self.wire.send("opening", round=w, pub=(rd / "pub.bin").read_bytes().hex(), xa=job["xa"], wb=job["wb"],
                       self_check=self_check)
        addr = self.wire.take("serve", w)["addr"]
        t3 = time.time()
        rc, prove_cpu, rss = None, None, None
        if addr is not None:
            cmd = [a.flock, "prove", "--zk", "--verifier", addr, "--instances", str(rd / "inst.bin"), "--circuit",
                   str(self.st.circuit), "--runs", "1", "--warm", "0"]
            env = {**os.environ, "RAYON_NUM_THREADS": str(a.prove_threads)}
            with open(rd / "prove.out", "w") as o, open(rd / "prove.err", "w") as e:
                p = subprocess.Popen(cmd, stdout=o, stderr=e, env=env)
                _pid, status, ru = os.wait4(p.pid, 0)
            rc, prove_cpu, rss = os.waitstatus_to_exitcode(status), ru.ru_utime + ru.ru_stime, ru.ru_maxrss // 1024
        t4 = time.time()
        self.wire.send("proved", round=w, rc=rc, prove_s=round(t4 - t3, 3), prove_cpu_s=prove_cpu)
        v = self.wire.take("verdict", w)
        t5 = time.time()
        tail = (rd / "prove.out").read_text()[-2000:] if (rd / "prove.out").exists() else ""
        self.rec(w, self_check=self_check, tile_check_s=round(t1 - t0, 3), stage_s=round(t2 - t1, 3),
                 serve_wait_s=round(t3 - t2, 3), prove_s=round(t4 - t3, 3), prove_rc=rc, prove_cpu_s=prove_cpu,
                 prove_maxrss_mb=rss, verdict_wait_s=round(t5 - t4, 3), verdict=v["status"], why=v.get("why"),
                 prove_tail=tail[-600:])
        with self.log_lock:
            self.log.write(json.dumps(self.records[w]) + "\n")
            self.log.flush()
        for f in ("inst.bin",):
            (rd / f).unlink(missing_ok=True)
        if v["status"] == "passed" and not a.keep:
            shutil.rmtree(rd, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--flock", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--bseed", type=int, default=0x5EED)
    ap.add_argument("--fused", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--prove-threads", type=int, default=64)
    ap.add_argument("--max-pending", type=int, default=2)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    if a.calls > 8:
        raise SystemExit("--calls: at most the engine's 8 slots")
    P = Prover(a)
    info = P.setup()
    print(json.dumps({"setup": info}), flush=True)
    worker = threading.Thread(target=P.proof_worker)
    worker.start()
    t = time.time()
    try:
        for w in range(a.rounds):
            P.round(w)
    finally:
        P.jobs.put(None)
        worker.join()
        P.E.close()
        wall = time.time() - t
        P.wire.send("bye", rounds=a.rounds, errors=P.errors)
        try:
            closed = P.wire.take("closed")
        except EOFError:
            closed = None
        info.update(rounds=a.rounds, calls=a.calls, wall_s=round(wall, 2), errors=P.errors,
                    window=closed and closed.get("window"))
        (P.out / "prover.json").write_text(json.dumps(info, indent=1))
        print(json.dumps({k: info[k] for k in ("rounds", "wall_s", "errors", "window")}), flush=True)
    return 0 if not P.errors else 1


if __name__ == "__main__":
    sys.exit(main())
