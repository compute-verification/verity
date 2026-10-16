"""challenger.py: the slice's challenger, a process of its own on the loopback link (wire.py), holding the round protocol of the
frozen definition v2.2 to the prover:

    (a) the prover commits round w's operands A (one tree a call, commit.py; each stage's B tree once, at hello)
    (b) the coin sigma_w, 64 bytes from the OS generator, only after (a), at least CADENCE seconds after the last coin
    (c) the prover commits the products (one DUOS receipt tree a call). With two stages (decision 36), stage 1's receipt
        roots come with stage 2's operand roots; --coins per-stage then draws stage 2's own coin sigma_w,1 right after
        them (protocol (a): each stage commits its operands, then draws, then commits its outputs), --coins one draws
        nothing more (sigma_w,1 = sigma_w), and stage 2's receipt roots follow
    (d) after the last of (c), one block drawn with fresh OS coins: a stage, a call, a 128 x 128 tile, one of its 512
        threads and a 32-step segment j
    (e) the block's statement (statement.py, `PouwSegCheck_v1{KS}`, or with --served dither `PouwSegCheckDither_v1{KS}`, the
        activations on B under decision 29's dither and the weights on A) checked against the commitments and the coins: the
        thread's 4 A leaves and 8 B leaves at j and its receipt leaf by their paths, the noise rows, chain keys and one-hot
        j recomputed, the output row held to 1; then one C-Flock --zk session of the Rust verifier (`flock-circuit serve
        --zk`, its own circuit and pin), one session at a time

A round fails after its coin if it is abandoned before (c), its session started and didn't finish, or its check or proof
fails; nothing is retried or redrawn. A round whose (a) came but not its coin is skipped. At the end: rounds.jsonl and the
window's floor (window.py) under --out.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
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
import window as WD  # noqa: E402
from wire import Wire, listen  # noqa: E402

COORDS = 1


class Challenger:
    def __init__(self, a, wire: Wire, stager: ST.Stager):
        self.a, self.wire, self.st = a, wire, stager
        self.names = [p.name for p in stager.ports]
        self.rounds: dict[int, dict] = {}
        self.lock = threading.Lock()
        self.last_coin = 0.0
        self.serve_q: list[int] = []
        self.serve_cv = threading.Condition()
        self.serving: subprocess.Popen | None = None
        self.done = False
        self.t0 = time.time()

    def now(self) -> float:
        return round(time.time() - self.t0, 4)

    def rec(self, w: int, **f) -> dict:
        with self.lock:
            r = self.rounds.setdefault(w, {"round": w, "status": "committed"})
            r.update(f)
            return r

    def final(self, w: int, status: str, why: str | None = None) -> None:
        with self.lock:
            r = self.rounds[w]
            if r["status"] in ("passed", "failed"):
                return
            r.update(status=status, why=why, t_verdict=self.now())
        try:
            self.wire.send("verdict", round=w, status=status, why=why)
        except OSError:
            pass

    # (a) and (b)
    def on_hello(self, m: dict) -> None:
        self.m, self.n, self.k, self.calls, self.sid = m["m"], m["n"], m["k"], m["calls"], m["session"]
        self.stages = m["stages"]
        self.b_roots, self.b_count = [bytes.fromhex(h) for h in m["b_roots"]], m["b_count"]
        fixed_rows = self.m if self.st.dither else self.n
        if (self.b_count != fixed_rows * self.k // 32 or m["pin"] != self.st.pin or self.k // 32 != self.st.ks
                or len(self.b_roots) != self.stages or m["coins"] != self.a.coins or m.get("served") != self.a.served):
            raise SystemExit(f"hello: the fixed trees {self.b_count} leaves, pin {m['pin'][:16]} (own {self.st.pin[:16]}), k "
                             f"{self.k}, {len(self.b_roots)} fixed roots for {self.stages} stages, coins {m['coins']} (own "
                             f"{self.a.coins}), served {m.get('served')} (own {self.a.served})")
        self.F = 2 * self.m * self.n * self.k * self.calls * self.stages

    def on_operands(self, m: dict) -> None:
        w = m["round"]
        if w in self.rounds or len(m["roots"]) != self.calls or m.get("stage") != 0:
            raise SystemExit(f"operands {w}: a round once, stage 0's, one root a call")
        self.rec(w, roots=[m["roots"]], receipt_roots=[], sigmas=[], t_a=self.now())
        with self.lock:
            due = max(time.time(), self.last_coin + self.a.cadence)
            self.last_coin = due
        threading.Timer(due - time.time(), self.coin, (w,)).start()

    def coin(self, w: int, stage: int = 0) -> None:
        sigma = os.urandom(64)
        with self.lock:
            self.rounds[w]["sigmas"].append(sigma.hex())
        self.rec(w, status="live", **{f"t_b{stage}" if stage else "t_b": self.now()})
        self.wire.send("coin", round=w, stage=stage, sigma=sigma.hex())

    # (c) and (d)
    def on_products(self, m: dict) -> None:
        w, s = m["round"], m["stage"]
        r = self.rounds.get(w)
        if (r is None or r["status"] != "live" or len(m["receipt_roots"]) != self.calls or s != len(r["receipt_roots"])
                or len(r["sigmas"]) != (s + 1 if self.a.coins == "per-stage" else 1)):
            raise SystemExit(f"products {w} stage {s}: before its coin, out of order, or not one receipt root a call")
        r["receipt_roots"].append(m["receipt_roots"])
        self.rec(w, **{f"t_c{s}": self.now()}, **{f"gpu{s}": m["gpu"]})
        if s + 1 < self.stages:
            if len(m.get("next_roots", [])) != self.calls:
                raise SystemExit(f"products {w} stage {s}: stage {s + 1}'s operand roots, one a call")
            r["roots"].append(m["next_roots"])
            if self.a.coins == "per-stage":
                self.coin(w, s + 1)
            return
        tm, tn = secrets.randbelow(self.m // derive.TM), secrets.randbelow(self.n // derive.BN)
        draw = {"stage": secrets.randbelow(self.stages), "call": secrets.randbelow(self.calls), "tile": [tm, tn],
                "thread": secrets.randbelow(derive.NCT), "seg": secrets.randbelow(self.k // 32)}
        self.rec(w, draw=draw, t_d=self.now(), t_c=self.now())
        self.wire.send("draw", round=w, **draw)

    # (e)
    def check(self, w: int, m: dict, pub: bytes) -> str | None:
        r = self.rounds[w]
        d = r["draw"]
        st, c, (tm, tn), th, j, ks = d["stage"], d["call"], d["tile"], d["thread"], d["seg"], self.k // 32
        sigma = bytes.fromhex(r["sigmas"][st if self.a.coins == "per-stage" else 0])
        path = Path(self.a.scratch) / f"r{w:04d}" / "pub.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pub)
        try:
            head, bc = ST.read_public(path, self.names)
        except (ValueError, json.JSONDecodeError) as e:
            return f"public file: {e}"
        if head.get("circuit_sha512") != self.st.pin or head.get("instances") != COORDS or head.get("range") != [0, COORDS]:
            return "public file: not this circuit's, or not one instance"
        if len(m["xa"]) != 4 or len(m["wb"]) != 8:
            return "opening: 4 xa paths and 8 wb paths"
        key, rseed = derive.call_key(sigma, c, st), derive.receipt_seed(sigma, c, st)
        tile = tm * (self.n // derive.BN) + tn
        rows, cols = derive.thread_slots(th)
        U = derive.u_noise64(key, self.m)[tm * derive.TM + rows]
        F = np.stack([derive.f_rows(key, (l := tn * derive.BN + int(cc)) - l % 4, 4, self.k)[l % 4] for cc in cols])
        want = ST.public_rows(U, F[:, 32 * j:32 * j + 32], rseed, {**d, "tile_index": tile}, ks)
        for port, row in [*want.items(), ("o0", ST.ok_row())]:
            salt = np.frombuffer(ST.public_salt(sigma, st, c, port), np.uint8)[None]
            if CM.commit_strings(row, salt)[0].tobytes() != bc[port][0]:
                return f"{port}: not the coin's row" if port != "o0" else "o0: the statement's output is not 1"
        # the call's tree (committed in (a), before the coin) and the stage's fixed tree (at hello): A and B, or under the
        # dither (the activations served) B and A
        call_tree = (CM.label_a(self.sid, w, c, st), bytes.fromhex(r["roots"][st][c]))
        fixed_tree = (CM.label_b(self.sid, st), self.b_roots[st])
        (la, ra), (lb, rb) = (fixed_tree, call_tree) if self.st.dither else (call_tree, fixed_tree)
        for q, rw in enumerate(rows):
            if not CM.verify_path(la, self.m * ks, ra, (tm * derive.TM + int(rw)) * ks + j, bc[f"xa{q}"][0],
                                  [bytes.fromhex(h) for h in m["xa"][q]]):
                return f"xa{q}: not stage {st} A's committed leaf ({tm * derive.TM + int(rw)}, {j})"
        for q, cc in enumerate(cols):
            if not CM.verify_path(lb, self.n * ks, rb, (tn * derive.BN + int(cc)) * ks + j, bc[f"wb{q}"][0],
                                  [bytes.fromhex(h) for h in m["wb"][q]]):
                return f"wb{q}: not stage {st} B's committed leaf ({tn * derive.BN + int(cc)}, {j})"
        nr = self.m // derive.TM * (self.n // derive.BN) * derive.NCT
        if not CM.verify_path(CM.label_r(self.sid, w, c, st), nr, bytes.fromhex(r["receipt_roots"][st][c]),
                              tile * derive.NCT + th, bc["rc"][0], [bytes.fromhex(h) for h in m["rc"]]):
            return f"rc: not the committed receipt leaf (tile {tile}, thread {th})"
        return None

    def on_opening(self, m: dict) -> None:
        w = m["round"]
        if self.rounds.get(w, {}).get("draw") is None:
            raise SystemExit(f"opening {w} before its draw")
        t = time.time()
        why = self.check(w, m, bytes.fromhex(m["pub"]))
        self.rec(w, t_open=self.now(), check_s=round(time.time() - t, 3), self_check=m.get("self_check"))
        if why:
            self.wire.send("serve", round=w, addr=None)
            self.final(w, "failed", f"check: {why}")
            return
        with self.serve_cv:
            self.serve_q.append(w)
            self.serve_cv.notify()

    def serve_worker(self) -> None:
        while True:
            with self.serve_cv:
                while not self.serve_q and not self.done:
                    self.serve_cv.wait()
                if not self.serve_q:
                    return
                w = self.serve_q.pop(0)
            self.serve_one(w)

    def serve_one(self, w: int) -> None:
        d = Path(self.a.scratch) / f"r{w:04d}"
        if self.a.flock == "none":
            self.wire.send("serve", round=w, addr=None)
            self.rec(w, serve_s=0.0, session=None)
            self.final(w, "passed", "checks only (no proof: --flock none)")
            return
        cmd = [self.a.flock, "serve", "--zk", "--listen", "127.0.0.1:0", "--out", str(d / "sessions"), "--instances",
               str(d / "pub.bin"), "--circuit", str(self.st.circuit), "--pin", self.st.pin, "--operator", "loopback",
               "--draw", f"subset:{COORDS}", "--sessions", "1"]
        env = {**os.environ, "RAYON_NUM_THREADS": str(self.a.verify_threads)}
        t = time.time()
        with open(d / "serve.err", "w") as err:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, env=env)
            self.serving = p
            addr = session = None
            for line in p.stdout:
                if line.startswith("SERVING "):
                    addr = line.split()[1]
                    self.rec(w, t_serving=self.now(), serve_setup_s=round(time.time() - t, 3))
                    self.wire.send("serve", round=w, addr=addr)
                elif line.startswith("SESSION\t"):
                    session = json.loads(line.split("\t", 1)[1])
            _pid, status, ru = os.wait4(p.pid, 0)
            p.returncode = os.waitstatus_to_exitcode(status)
            self.serving = None
        kept = Path(self.a.out) / "sessions"
        kept.mkdir(exist_ok=True)
        proof_bytes = 0
        for sd in (d / "sessions").glob("*/"):
            proof_bytes += sum(f.stat().st_size for f in sd.glob("*.bin"))
            if (sd / "session.json").exists():
                shutil.copy(sd / "session.json", kept / f"r{w:04d}-{sd.name}.json")
        shutil.rmtree(d / "sessions", ignore_errors=True)
        self.rec(w, serve_s=round(time.time() - t, 3), serve_rc=p.returncode, verify_cpu_s=round(ru.ru_utime + ru.ru_stime, 3),
                 serve_maxrss_mb=round(ru.ru_maxrss / 1024), session=session, proof_bytes=proof_bytes)
        if addr is None:
            self.wire.send("serve", round=w, addr=None)
            self.final(w, "failed", f"serve exited before SERVING (rc {p.returncode}): {(d / 'serve.err').read_text()[-300:]}")
        elif session is None:
            self.final(w, "failed", "its session started and didn't finish")
        elif session.get("accepted") is True:
            self.final(w, "passed")
        else:
            self.final(w, "failed", f"proof: {session.get('why')}")

    def close(self) -> None:
        with self.serve_cv:
            self.done = True
            self.serve_cv.notify()
        for w, r in sorted(self.rounds.items()):
            if r["status"] == "live":
                self.final(w, "failed", "abandoned after its draw, no proof" if "t_d" in r else "abandoned before (c)")
            elif r["status"] == "committed":
                self.rec(w, status="skipped")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flock", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--port-file", required=True)
    ap.add_argument("--cadence", type=float, default=10.0)
    ap.add_argument("--verify-threads", type=int, default=16)
    ap.add_argument("--p", type=float, default=0.75)
    ap.add_argument("--k", type=int, default=4096, help="the layer's k (the statement's KS = k / 32), held to the prover's")
    ap.add_argument("--coins", choices=("one", "per-stage"), default="one")
    ap.add_argument("--served", choices=("pm1", "dither"), default="dither", help="held to the prover's (prover.py --served)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    st = ST.Stager(Path(a.scratch) / "circuit.txt", a.k // 32, a.served == "dither")
    stager_s = time.time() - t
    srv, port = listen()
    Path(a.port_file).write_text(str(port))
    conn, _ = srv.accept()
    wire = Wire(conn)
    ch = Challenger(a, wire, st)
    worker = threading.Thread(target=ch.serve_worker)
    worker.start()
    handlers = {"hello": ch.on_hello, "operands": ch.on_operands, "products": ch.on_products, "opening": ch.on_opening}
    bye = None
    try:
        while (m := wire.recv()) is not None:
            if m["kind"] == "bye":
                bye = m
                break
            if m["kind"] == "proved":
                ch.rec(m["round"], prover=m)
                continue
            handlers[m["kind"]](m)
    finally:
        if bye is None:
            with ch.serve_cv:
                ch.serve_q.clear()
            if ch.serving is not None:
                ch.serving.kill()
        ch.close()
        worker.join()
        rounds = [dict(r, F=ch.F if r["status"] != "skipped" else 0) for _w, r in sorted(ch.rounds.items())]
        with open(out / "rounds.jsonl", "w") as f:
            for r in rounds:
                f.write(json.dumps(r) + "\n")
        coins = [r["t_b"] for r in rounds if "t_b" in r]
        summary = {"window": WD.floor(rounds, a.p), "circuit": st.stats(), "stager_s": round(stager_s, 2), "bye": bye,
                   "coins": a.coins, "served": a.served, "stages": getattr(ch, "stages", None),
                   "coin_intervals_s": [round(b - x, 3) for x, b in zip(coins, coins[1:])],
                   "why_failed": {r["round"]: r.get("why") for r in rounds if r["status"] == "failed"}}
        (out / "window.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary["window"]))
        try:
            wire.send("closed", window=summary["window"])
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
