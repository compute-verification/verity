"""PoUS's off-region fallback at fixture scale (MEM-71): batched rounds through the memory challenger's party process
(`challenger_party`, the deployed LMS key), on one GPU, with the uk-south2 ↔ us-central1 round trip simulated as a wait
the prover adds before each answer (Daniel, 23:40Z: the round trip "really can be simulated").

    research run --on vy-dc-h200-2 --gpus 1 ... -- python benchmarks/pous/batch/fixture.py --lib DIR [--lean EXE]

Two stores of random W, encoded on the GPU (`p2_v1.gpu`, the library `pod.sh IMPL=p2-scale` builds) and set up as
`period.setup_store` does (the Python public-encoder check of all of `C*`): 256 blocks (`period.certified_point`, P2's
proved family) and 4,096 blocks. A batch of b blocks has the deadline `sizing.deadline_s`: RTT_max + b·2,056/BW +
margin, at two links, `ssh` (2 MiB channel window over the 97.39 ms round trip) and `10g` (10 Gbit/s). Its prover waits
until RTT + the batch's transfer time after the indices reach it. The cases, each one session:

- `seq`: the 256-block store at its certified point, honest, no added round trip: accepted, S certified.
- `seq-far`: the same with the round trip added to every round: refused, late (854 µs is no deadline off-region).
- `seq-relaxed`: the certified point with its limit raised to RTT_max + margin: accepted, but the certificate refuses
  the deadline, so S = 0.
- `honest` / `zeros` / `gpu` / `floor`, at each batched point: the HBM prover; a prover that dropped a fixed 10% of the
  blocks and answers them with zeros; one that re-encodes the dropped blocks a batch asks for on this GPU from W
  (`Kernels`, two roots a block, all of them in one launch) while the rest cross the link; and one that recomputes them
  at P2's timing red team's floor, 2 × 5,960.54 µs a block on each of the card's 16 threads, counted from the moment the
  batch is sent (as if its compute sat beside the challenger) and so simulated, not run.

Every case's answer comes from `period.audit`; with `--lean`, `pous-run-protocol` answers it too (it must agree on the
sequential cases and refuse the batched ones, whose header it doesn't know). `$OUT/summary.json` has every session's
verdict, reason, S, and per batch the dropped blocks asked for and when the answer landed against T.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import secrets
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sizing as Z  # noqa: E402

from verity.core.protocols.pous import challenger_party, period, setup  # noqa: E402
from verity.core.protocols.pous import memory_challenger as mc  # noqa: E402
from verity.core.protocols.pous import run as R  # noqa: E402
from verity.core.protocols.pous.schemes import p2 as P2  # noqa: E402

RTT_NS = 97_390_000
"""p50 of 20 TCP connects vy-dc-h200-2 → vy-auditor-1 (r20261025-233837-25b4); cpu-1 → vy-dc-h200-2 97.44 ms."""
RTT_MAX_S, MARGIN_S = 0.10018, 0.010
"""The largest of 20 connects vy-dc-h200 → vy-auditor-1 (r20261025-233837-b76c), and the margin T adds."""
LINKS = {"ssh": 2 * Z.MIB / (RTT_NS / 1e9), "10g": 1.25e9}
FLOOR_BLOCK_NS, FLOOR_THREADS = 2 * 5_960_540, 16
"""Two roots a block at the timing red team's single-core floor (`schemes.p2.ROOT_CALL_NS`'s derivation), on each of the
card's 16 hardware threads."""
DROP = 0.10


def wait_until(t_ns: int) -> None:
    while time.perf_counter_ns() < t_ns:
        pass


def transfer_ns(blocks: int, bw: float) -> int:
    return int(blocks * Z.BLOCK_BYTES / bw * 1e9)


class Gpu:
    """`libp2v1.so` for one setup's keys: a whole segment encoded (and decoded back to W), or any blocks at once."""

    def __init__(self, lib: str, q: P2.P2Params, salt: bytes) -> None:
        import torch
        from p2_v1 import reference as REF
        from p2_v1.gpu import Kernels

        self.torch, self.q = torch, q
        self.K = Kernels(lib, "", REF.Params(q.w, q.c0, q.payload_bytes, q.blocks_per_segment, q.segments),
                         P2.expand_prefix(q, salt))

    def _rows(self, payloads: bytes, n: int):
        q = self.q
        m = self.torch.zeros(n, q.block_bytes, dtype=self.torch.uint8)
        m[:, q.block_bytes - q.payload_bytes:] = self.torch.frombuffer(bytearray(payloads), dtype=self.torch.uint8).view(
            n, q.payload_bytes)
        return m.to(self.K.device)

    def segment(self, s: int, w_seg: bytes) -> list[bytes]:
        n = self.q.blocks_per_segment
        m = self._rows(w_seg, n)
        c = self.K.encode(m, s * n)
        if self.K.raised() or not self.torch.equal(self.K.decode(c, s * n), m) or self.K.raised():
            raise RuntimeError(f"segment {s}: the device's encoding is flagged or doesn't decode to W")
        data = c.cpu().numpy().tobytes()
        return [data[i:i + self.q.block_bytes] for i in range(0, len(data), self.q.block_bytes)]

    def blocks(self, indices: list[int], w: bytes) -> list[bytes]:
        """Blocks `indices` of `C*` re-encoded from W in one launch, their keys expanded for each."""
        torch, q, K = self.torch, self.q, self.K
        keys = [K.keys(i, 1) for i in indices]
        k, t = torch.cat([x[0] for x in keys]), torch.cat([x[1] for x in keys])
        m = self._rows(b"".join(w[i * q.payload_bytes:(i + 1) * q.payload_bytes] for i in indices), len(indices))
        c = torch.empty_like(m)
        a, n, r = K.split
        rc = K.lib.p2v1_encode(m.data_ptr(), c.data_ptr(), k.data_ptr(), t.data_ptr(), m.shape[0], q.c0, a, n, r,
                               K.flags.data_ptr(), K._stream())
        torch.cuda.synchronize()
        if rc or K.raised():
            raise RuntimeError(f"p2v1_encode: rc {rc} or flags")
        data = c.cpu().numpy().tobytes()
        return [data[j * q.block_bytes:(j + 1) * q.block_bytes] for j in range(len(indices))]


def setup_gpu(lib: str, point: R.Point, store: bytes, coins: bytes) -> tuple[period.Store, Gpu, bytes]:
    """`period.setup_store` with `C*` encoded on the GPU: the commitment, the salt drawn after it, and the Python
    public-encoder check of every block (`setup.verifier_key_from_encoding`), which refuses a wrong encoding."""
    q = point.params
    w = bytes(store) + bytes(q.w_bytes - len(store))
    commitment = setup.commit(q, w)
    salt = setup.draw_salt(coins, commitment)
    gpu = Gpu(lib, q, salt)
    encoded = [gpu.segment(s, w[s * q.segment_bytes:(s + 1) * q.segment_bytes]) for s in range(q.segments)]
    root = setup.verifier_key_from_encoding(q, None, commitment, salt, encoded).root
    blocks = tuple(b for seg in encoded for b in seg)
    held = period.Store(point, R.Setup(q.blocks, commitment, root, salt), blocks,
                        b"".join(mc.block_digest(b) for b in blocks), period.setup_case(q, commitment, salt, encoded))
    return held, gpu, w


def batched_point(held: period.Store, b: int, bw: float, rounds: int = 1) -> R.Point:
    t = Z.deadline_s(b, RTT_MAX_S, bw, MARGIN_S)
    return R.Point(held.point.params, b * rounds, math.ceil(t * 1e9), b)


class Provers:
    """The batch provers of one store, one link: each waits until its answer would reach the challenger, and logs per
    batch the dropped blocks asked for (`missing`) and when its answer was ready, in ns after the batch arrived."""

    def __init__(self, held: period.Store, gpu: Gpu, w: bytes, bw: float, dropped: frozenset[int]) -> None:
        import torch

        self.held, self.gpu, self.w, self.bw, self.dropped, self.torch = held, gpu, w, bw, dropped, torch
        n = held.point.params.block_bytes
        self.n = n
        self.hbm = torch.frombuffer(bytearray(b"".join(held.blocks)), dtype=torch.uint8).to("cuda").view(-1, n)
        self.log: list[dict] = []

    def _read(self, idx) -> list[bytes]:
        torch = self.torch
        rows = self.hbm.index_select(0, torch.tensor(list(idx), device=self.hbm.device)).cpu().numpy().tobytes()
        return [rows[j * self.n:(j + 1) * self.n] for j in range(len(idx))]

    def _done(self, t0: int, ready: int, missing: int, extra: dict | None = None) -> None:
        self.log.append({"missing": missing, "ready_ns": ready - t0, **(extra or {})})

    def honest(self, _blocks):
        def respond(idx):
            t0 = time.perf_counter_ns()
            out = self._read(idx)
            ready = t0 + RTT_NS + transfer_ns(len(idx), self.bw)
            self._done(t0, ready, 0)
            wait_until(ready)
            return out
        return respond

    def zeros(self, _blocks):
        def respond(idx):
            t0 = time.perf_counter_ns()
            out = [bytes(self.n) if i in self.dropped else b for i, b in zip(idx, self._read(idx), strict=True)]
            ready = t0 + RTT_NS + transfer_ns(len(idx), self.bw)
            self._done(t0, ready, len({i for i in idx if i in self.dropped}))
            wait_until(ready)
            return out
        return respond

    def gpu_recompute(self, _blocks):
        def respond(idx):
            t0 = time.perf_counter_ns()
            missing = sorted({i for i in idx if i in self.dropped})
            got = dict(zip(missing, self.gpu.blocks(missing, self.w), strict=True)) if missing else {}
            computed = time.perf_counter_ns() - t0
            wrong = sum(got[i] != self.held.blocks[i] for i in missing)
            out = [got[i] if i in got else b for i, b in zip(idx, self._read(idx), strict=True)]
            ready = t0 + RTT_NS + max(transfer_ns(len(idx), self.bw), computed + transfer_ns(len(missing), self.bw))
            self._done(t0, ready, len(missing), {"compute_ns": computed, "recomputed_wrong": wrong})
            wait_until(ready)
            return out
        return respond

    def floor(self, _blocks):
        def respond(idx):
            t0 = time.perf_counter_ns()
            missing = len({i for i in idx if i in self.dropped})
            compute = math.ceil(missing / FLOOR_THREADS) * FLOOR_BLOCK_NS
            out = self._read(idx)
            ready = t0 + max(RTT_NS + transfer_ns(len(idx) - missing, self.bw), compute)
            self._done(t0, ready, missing, {"compute_ns": compute})
            wait_until(ready)
            return out
        return respond


def far(blocks):
    def respond(i: int) -> bytes:
        wait_until(time.perf_counter_ns() + RTT_NS)
        return blocks[i]
    return respond


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", required=True, help="the directory of libp2v1.so")
    ap.add_argument("--lean", help="pous-run-protocol, to answer every case beside the Python")
    ap.add_argument("--reps", type=int, default=2, help="sessions of each batched case")
    ap.add_argument("--big", type=int, default=4096, help="the second store's blocks")
    a = ap.parse_args(argv)
    out, node = Path(os.environ.get("OUT", ".")), socket.gethostname()
    scratch = Path(tempfile.mkdtemp(prefix="pous-batch-"))
    summary: dict = {"node": node, "rtt_ns": RTT_NS, "rtt_max_s": RTT_MAX_S, "margin_s": MARGIN_S,
                     "links": LINKS, "floor": {"block_ns": FLOOR_BLOCK_NS, "threads": FLOOR_THREADS}, "drop": DROP,
                     "stores": {}, "sessions": []}
    try:
        w, c0, _ = P2.POINT
        payload = (w - 16) // 8
        small = period.certified_point(0)
        big = R.Point(P2.P2Params(w, c0, a.big, 1), period.K, period.LIMIT_NS)
        party = challenger_party.Party.local(node, scratch / "challenger")
        stores = {}
        for name, point in (("s256", small), (f"s{a.big}", big)):
            t = time.monotonic()
            held, gpu, wbytes = setup_gpu(a.lib, point, os.urandom(point.params.blocks * payload), secrets.token_bytes(64))
            dropped = frozenset(random.Random(f"verity/mem71/drop/{name}").sample(range(point.params.blocks),
                                                                                round(DROP * point.params.blocks)))
            stores[name] = (held, gpu, wbytes, dropped)
            summary["stores"][name] = {"blocks": point.params.blocks, "dropped": len(dropped),
                                       "setup_s": round(time.monotonic() - t, 2), "root": held.published.root.hex()}
            print("store", name, json.dumps(summary["stores"][name]), flush=True)

        def record(name: str, point: R.Point, p: period.Period, log: list[dict] | None, wall: float) -> None:
            s = R.Session.from_json(json.loads(bytes.fromhex(p.case["evidence"]))["session"])
            el = sorted({r.elapsed_ns for r in s.signed.rounds})
            rec = {"case": name, "blocks": point.params.blocks, "batch": point.batch, "k": point.k,
                   "limit_ns": point.limit_ns, "verdict": p.verdict, "reason": p.answer.get("reason"),
                   "public": p.answer.get("public"), "gaps": (p.answer.get("profile") or {}).get("gaps"),
                   "ended": s.signed.ended, "rounds": len(s.signed.rounds), "elapsed_ns": el[:4] + el[-2:],
                   "batches": log, "wall_s": round(wall, 2)}
            if a.lean:
                try:
                    lean = period.serve(a.lean, [p.case])[0]
                    rec["lean"] = {"verdict": lean["verdict"], "reason": lean.get("reason"),
                                   "agrees": {k: v for k, v in lean.items() if k != "reason"}
                                   == {k: v for k, v in p.answer.items() if k != "reason"}}
                except Exception as e:  # noqa: BLE001 - a Lean that exits on a form it doesn't know refuses it
                    rec["lean"] = {"verdict": None, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            summary["sessions"].append(rec)
            print(name, json.dumps({k: v for k, v in rec.items() if k != "batches"}), flush=True)

        def session(name: str, held: period.Store, point: R.Point, *, prover=period.honest, batch_prover=None,
                    log: list[dict] | None = None) -> None:
            t = time.monotonic()
            n0 = len(log) if log is not None else 0
            p = period.audit(node, dataclasses.replace(held, point=point), secrets.token_bytes(64), party=party,
                             prover=prover, batch_prover=batch_prover)
            record(name, point, p, None if log is None else log[n0:], time.monotonic() - t)

        s256 = stores["s256"][0]
        session("seq", s256, small)
        session("seq-far", s256, small, prover=far)
        relaxed = dataclasses.replace(small, limit_ns=math.ceil((RTT_MAX_S + MARGIN_S) * 1e9))
        session("seq-relaxed", s256, relaxed, prover=far)

        for name, (held, gpu, wbytes, dropped) in stores.items():
            b = held.point.params.blocks
            for link, bw in LINKS.items():
                point = batched_point(held, b, bw)
                pv = Provers(held, gpu, wbytes, bw, dropped)
                for case in ("honest", "zeros", "gpu", "floor"):
                    make = {"honest": pv.honest, "zeros": pv.zeros, "gpu": pv.gpu_recompute, "floor": pv.floor}[case]
                    for _ in range(a.reps):
                        session(f"{name}/{link}/{case}", held, point, batch_prover=make, log=pv.log)

        summary["sizing"] = {
            "measured_rate": {"h200_blocks_per_s": 647.32, "h200_latency_s": 1.0931, "cpu_gmpy2_blocks_per_s": 18.50,
                              "cpu_gmpy2_block_s": 0.4602, "run": "r20261026-000854-1494"},
            "conservative_rate": 647.32 + FLOOR_THREADS / (FLOOR_BLOCK_NS / 1e9),
        }
        summary["sizing"]["table"] = Z.table(summary["sizing"]["conservative_rate"], RTT_MAX_S, MARGIN_S,
                                             {k: v for k, v in LINKS.items()},
                                             {"0.10": 0.10, "1/19": 1 / 19, "obligation": Z.f_obligation()}, 0.01)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        checks = check(summary, a.reps, a.big)
        for k, v in checks.items():
            print(("PASS " if v else "FAIL ") + k, flush=True)
        (out / "result.json").write_text(json.dumps({"ok": all(checks.values()), "checks": checks}, indent=1))
        return 0 if all(checks.values()) else 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def check(summary: dict, reps: int, big: int) -> dict[str, bool]:
    S = {}
    for r in summary["sessions"]:
        S.setdefault(r["case"], []).append(r)
    every = lambda case, f: case in S and all(f(r) for r in S[case])  # noqa: E731
    acc0 = lambda r: r["verdict"] == "accept" and r["public"]["s_bits"] == 0 and any(  # noqa: E731
        "indices a round under one deadline" in g for g in r["gaps"])
    out = {
        "seq: the certified point accepts the honest store with S > 0": every(
            "seq", lambda r: r["verdict"] == "accept" and r["public"]["certified"] and r["public"]["s_bits"] > 0),
        "seq-far: an off-region round trip at 854 µs is refused, late": every(
            "seq-far", lambda r: r["verdict"] == "reject" and r["ended"] == "late"),
        "seq-relaxed: a deadline past RTT_max is accepted with S = 0 (the certificate refuses it)": every(
            "seq-relaxed", lambda r: r["verdict"] == "accept" and r["public"]["s_bits"] == 0 and r["gaps"]),
    }
    for name in ("s256", f"s{big}"):
        for link in LINKS:
            p = f"{name}/{link}"
            out[f"{p}/honest: accepted, S = 0 with the batch named a gap"] = every(f"{p}/honest", acc0)
            out[f"{p}/zeros: refused for wrong answers"] = every(
                f"{p}/zeros", lambda r: r["verdict"] == "reject" and "answers are not their blocks of C" in r["reason"])
            out[f"{p}/gpu: refused, late (the H200 finishes no block inside T)"] = every(
                f"{p}/gpu", lambda r: r["verdict"] == "reject" and r["ended"] == "late"
                and all(x["recomputed_wrong"] == 0 for x in r["batches"]))
    return out


if __name__ == "__main__":
    sys.exit(main())
