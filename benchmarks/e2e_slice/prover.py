"""prover.py: the slice's prover. One o_proj served by the GPU engine (slice.cu), CALLS calls a round, and with --stages 2 a
second GEMM on its own B reading the first's served product through the public map derive.phi, through the challenger's
round protocol (challenger.py) over the loopback link:

    setup  each stage's B on the device; its tree (commit.py), the roots in hello, before any coin
    (a)    each call's stage-1 A on the device and to a file; A's tree a call; the roots out
    (b)    the coin sigma_0 in
    (c)    stage 1: each call through the edited GEMM under call_key(sigma_0, c) and receipt_seed(sigma_0, c) (DUOS receipts,
           plain cuBLASLt interleaved for the baseline); a receipt tree a call. With --stages 2: stage 2's A = phi(Y) a call,
           its trees, then with --coins one-draw nothing more is drawn (sigma_1 = sigma_0), with --coins per-stage the roots
           go out and sigma_1 comes in; stage 2's calls under call_key(sigma_1, c, 1), their receipt trees, the roots out
    (d)    the draw in: a stage, a call, a tile, a thread and a segment j; the tile's receipts and served product kept, the
           thread's xa, wb and rc paths taken, the round's trees dropped
    proof  (a worker, a round at a time, overlapping the next rounds) the drawn tile recomputed from A, B and the coin alone
           (derive.tile_noised, the prover's own check of its GPU), the thread's statement staged (statement.Stager), the
           public file and the paths out, then `flock-circuit prove --zk` against the challenger's server

--attack cancel is RT2's cancellation (decision 36's Proposition 1.2): stage 2's A is committed as -U_2 W^T (A_ik = -U_{i, k
mod 32}) under the noise the prover expects, call_key(sigma_0, c, 1), so A' = A + U_2 W^T is zero and so is every prefix word
of the noised product: its receipts are the chains' keys alone, no MAC of it is needed. Under one draw a round that is the
true noise; under per-stage draws it is a guess made before sigma_1.

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
        self.ks = a.k // 32
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
        if a.engine == "fake":
            from fake_engine import FakeEngine
            self.E = FakeEngine()
        else:
            from engine import Engine
            self.E = Engine(a.engine, open(self.out / "engine.err", "w"))
        t = time.time()
        self.setup_info = [self.E.cmd("setup", a.m, a.n, a.k, a.bseed + s, self.scratch / f"B{s}.bin", "duos", s)
                           for s in range(a.stages)]
        self.B = [np.fromfile(self.scratch / f"B{s}.bin", np.int8).reshape(a.n, a.k) for s in range(a.stages)]
        t1 = time.time()
        self.tb = [CM.commit_operand(self.pool, str(self.scratch / f"B{s}.bin"), (a.n, a.k), CM.label_b(self.session, s),
                                     self.salt_key) for s in range(a.stages)]
        t2 = time.time()
        self.st = ST.Stager(self.scratch / "circuit.txt", self.ks)
        t3 = time.time()
        self.wire = connect(a.port)
        self.wire.start_reader()
        self.wire.send("hello", m=a.m, n=a.n, k=a.k, calls=a.calls, stages=a.stages, coins=a.coins, session=self.session,
                       b_roots=[t.root.hex() for t in self.tb], b_count=self.tb[0].count, pin=self.st.pin)
        return {"engine": self.setup_info, "setup_s": round(t1 - t, 3), "b_commit_s": round(t2 - t1, 3),
                "b_commit_cpu_s": round(sum(t.cpu_s for t in self.tb), 3), "b_leaves": self.tb[0].count,
                "stager_s": round(t3 - t2, 3), "circuit": self.st.stats(), "stages": a.stages, "coins": a.coins,
                "attack": a.attack}

    def receipts(self, w: int, s: int, sigmas: list, keys: list) -> tuple[list, list, list, float]:
        """Stage s's calls on the engine (A already in each slot), their receipt trees."""
        a, gpu = self.a, []
        rpaths = [self.scratch / f"R-{w}-{s}-{c}.bin" for c in range(a.calls)]
        for c in range(a.calls):
            gpu.append(self.E.cmd("call", c, *derive.sig_words(keys[c]), derive.receipt_seed(sigmas[s], c, s), rpaths[c],
                                  a.fused, s))
        shape = (a.m // derive.TM * (a.n // derive.BN), self.ks, derive.NCT)
        trees = [CM.commit_receipts(self.pool, str(rpaths[c]), shape, CM.label_r(self.session, w, c, s), self.salt_key)
                 for c in range(a.calls)]
        return gpu, rpaths, trees, sum(t.cpu_s for t in trees)

    def round(self, w: int) -> None:
        a = self.a
        self.pending.acquire()
        t0 = time.time()
        apaths = [[self.scratch / f"A-{w}-{s}-{c}.bin" for c in range(a.calls)] for s in range(a.stages)]
        for c in range(a.calls):
            self.E.cmd("opnds", c, (w * a.calls + c) & 0xFFFFFFFF, apaths[0][c], 0)
        t1 = time.time()
        atrees = [[CM.commit_operand(self.pool, str(apaths[0][c]), (a.m, a.k), CM.label_a(self.session, w, c, 0), self.salt_key)
                   for c in range(a.calls)]]
        t2 = time.time()
        self.wire.send("operands", round=w, stage=0, roots=[t.root.hex() for t in atrees[0]])
        sigmas = [bytes.fromhex(self.wire.take("coin", w)["sigma"])]
        t3 = time.time()
        gpu0, rp0, rt0, rcpu = self.receipts(w, 0, sigmas, [derive.call_key(sigmas[0], c, 0) for c in range(a.calls)])
        rpaths, rtrees, gpu, tm_ = [rp0], [rt0], [gpu0], {"stage0_s": round(time.time() - t3, 3)}
        if a.stages == 2:
            t4 = time.time()
            guess = [derive.call_key(sigmas[0], c, 1) for c in range(a.calls)]
            for c in range(a.calls):
                if a.attack == "cancel":
                    U2 = derive.u_noise64(guess[c], a.m)
                    np.tile(-U2, (1, a.k // 32)).astype(np.int8).tofile(apaths[1][c])
                else:
                    yp = self.scratch / f"Y-{w}-{c}.bin"
                    self.E.cmd("ydump", c, yp, 0)
                    derive.phi(np.fromfile(yp, np.int32).reshape(a.m, a.n)).tofile(apaths[1][c])
                    yp.unlink()
                self.E.cmd("aload", c, apaths[1][c], 1)
            atrees.append([CM.commit_operand(self.pool, str(apaths[1][c]), (a.m, a.k), CM.label_a(self.session, w, c, 1),
                                             self.salt_key) for c in range(a.calls)])
            t5 = time.time()
            self.wire.send("products", round=w, stage=0, receipt_roots=[t.root.hex() for t in rt0], gpu=gpu0,
                           next_roots=[t.root.hex() for t in atrees[1]])
            sigmas.append(bytes.fromhex(self.wire.take("coin", w)["sigma"]) if a.coins == "per-stage" else sigmas[0])
            t6 = time.time()
            keys = guess if a.attack == "cancel" else [derive.call_key(sigmas[1], c, 1) for c in range(a.calls)]
            gpu1, rp1, rt1, rcpu1 = self.receipts(w, 1, sigmas, keys)
            rpaths.append(rp1), rtrees.append(rt1), gpu.append(gpu1)
            rcpu += rcpu1
            tm_.update(a2_s=round(t5 - t4, 3), coin1_wait_s=round(t6 - t5, 3), stage1_s=round(time.time() - t6, 3))
        t7 = time.time()
        last = a.stages - 1
        self.wire.send("products", round=w, stage=last, receipt_roots=[t.root.hex() for t in rtrees[last]], gpu=gpu[last])
        d = self.wire.take("draw", w)
        t8 = time.time()
        s, c, (tm, tn), th, j = d["stage"], d["call"], d["tile"], d["thread"], d["seg"]
        tile = tm * (a.n // derive.BN) + tn
        ypath = self.scratch / f"Y-{w}.bin"
        self.E.cmd("tile", c, tm, tn, ypath, s)
        Y = np.fromfile(ypath, np.int32).reshape(derive.TM, derive.BN)
        ypath.unlink()
        R = np.fromfile(rpaths[s][c], np.uint32).reshape(-1, self.ks, derive.NCT)[tile].copy()
        rows, cols = derive.thread_slots(th)
        per = self.ks
        at, bt, rt = atrees[s][c], self.tb[s], rtrees[s][c]
        pos_a = [(tm * derive.TM + int(r)) * per + j for r in rows]
        pos_b = [(tn * derive.BN + int(cc)) * per + j for cc in cols]
        pos_r = tile * derive.NCT + th
        open_ = {"xa": [[h.hex() for h in at.path(p)] for p in pos_a], "wb": [[h.hex() for h in bt.path(p)] for p in pos_b],
                 "rc": [h.hex() for h in rt.path(pos_r)],
                 "salts": {**{f"xa{r}": [CM.leaf_salt(self.salt_key, at.label, p)] for r, p in enumerate(pos_a)},
                           **{f"wb{q}": [CM.leaf_salt(self.salt_key, bt.label, p)] for q, p in enumerate(pos_b)},
                           "rc": [CM.leaf_salt(self.salt_key, rt.label, pos_r)]}}
        for ps in rpaths:
            for p in ps:
                p.unlink()
        for ss, ps in enumerate(apaths):
            for cc, p in enumerate(ps):
                if (ss, cc) != (s, c):
                    p.unlink()
        commit_cpu = sum(t.cpu_s for ts in atrees for t in ts)
        del atrees, rtrees
        t9 = time.time()
        self.rec(w, gpu=gpu, a_gen_s=round(t1 - t0, 3), a_commit_s=round(t2 - t1, 3), a_commit_cpu_s=round(commit_cpu, 3),
                 coin_wait_s=round(t3 - t2, 3), **tm_, receipt_cpu_s=round(rcpu, 3), draw_wait_s=round(t8 - t7, 3),
                 open_s=round(t9 - t8, 3), draw=d)
        self.jobs.put({"round": w, "sigma": sigmas[s], "draw": d, "apath": apaths[s][c], "Y": Y, "R": R, **open_})

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
        s, c, (tm, tn), th, j = d["stage"], d["call"], d["tile"], d["thread"], d["seg"]
        t0 = time.time()
        A = np.fromfile(job["apath"], np.int8).reshape(a.m, a.k)
        key, rseed = derive.call_key(sigma, c, s), derive.receipt_seed(sigma, c, s)
        Ar, U, Bp, F, Ts = derive.tile_noised(A, self.B[s], key, tm, tn, a.k)
        tile = tm * (a.n // derive.BN) + tn
        want_r = derive.seg_receipts(rseed, Ts, tile)
        self_check = {"receipts": bool((job["R"] == want_r).all()), "product": bool((job["Y"] == Ar @ Bp.T).all()),
                      "noised_product_zero": bool((Ts[-1] == 0).all())}
        t1 = time.time()
        rows, cols = derive.thread_slots(th)
        ent_r, ent_c = derive.thread_entries()
        prev = (Ts[j - 1][ent_r[th], ent_c[th]] if j else np.zeros(32, np.uint32)).astype("<u4")
        jj = slice(32 * j, 32 * j + 32)
        inst = {**{f"xa{r}": ST._u16(A[tm * derive.TM + int(rw), jj].view(np.uint8)) for r, rw in enumerate(rows)},
                **{f"wb{q}": ST._u16(self.B[s][tn * derive.BN + int(cc), jj].view(np.uint8)) for q, cc in enumerate(cols)},
                "prev": ST._u16(prev.view(np.uint8)), "rc": ST._u16(job["R"][:, th].astype("<u4").view(np.uint8)),
                **ST.public_rows(U[rows], F[cols, jj], rseed, {**d, "tile_index": tile}, self.ks)}
        rd = self.scratch / f"r{w:04d}"
        ok = self.st.stage(rd, inst, job["salts"], sigma, s, c, self.salt_key)
        self_check["ok"] = ok
        job["apath"].unlink()
        t2 = time.time()
        self.wire.send("opening", round=w, pub=(rd / "pub.bin").read_bytes().hex(), xa=job["xa"], wb=job["wb"], rc=job["rc"],
                       self_check=self_check)
        addr = self.wire.take("serve", w)["addr"]
        t3 = time.time()
        rc, prove_cpu, rss = None, None, None
        if addr is not None and a.flock != "none":
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
        (rd / "inst.bin").unlink(missing_ok=True)
        if v["status"] == "passed" and not a.keep:
            shutil.rmtree(rd, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="slice.cu's binary, or fake (fake_engine.py, numpy, small shapes)")
    ap.add_argument("--flock", required=True, help="flock-circuit, or none (no proof: the challenger's checks only)")
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
    ap.add_argument("--stages", type=int, choices=(1, 2), default=1)
    ap.add_argument("--coins", choices=("one", "per-stage"), default="one")
    ap.add_argument("--attack", choices=("none", "cancel"), default="none")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--prove-threads", type=int, default=64)
    ap.add_argument("--max-pending", type=int, default=2)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    if a.calls > 8:
        raise SystemExit("--calls: at most the engine's 8 slots")
    if a.attack != "none" and a.stages != 2:
        raise SystemExit("--attack: on stage 2, so --stages 2")
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
