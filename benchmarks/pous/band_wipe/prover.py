"""The prover's side of the band-fill wipe, on the GPU node (`wire.py` has the control channel).

    python -m band_wipe.prover --out DIR --build-dir DIR [--listen 127.0.0.1:47610] [--resp-port 47611] [--segments N]
        [--reserve-mib 512] [--threads 32] [--responder-cpu C]

It allocates `C`, every whole 32 MiB segment the device's free memory holds after `--reserve-mib` and the fill's scratch
(`SCRATCH_PER_SEGMENT`), or `--segments`, and waits for the memory challenger.  Each wipe:

1. **Fill.** Under the memory challenger's fresh salt, `W = 0`, so `C_v = key_v`: 512 launches of `pous_native.cu`'s
   `band_chain` (block `v` of every segment per launch, its `min(v, 12) + 1` `Π₂` calls reading its parents in place in
   `C`), each writing the key to `C`'s block `v`.  The fill's device time over `C`'s GiB is the fill time per GiB; the
   device memory left free after it (its scratch freed) is memory outside the fill, which the guarantee adds to what a
   prover can keep.
2. **Gate.** Blocks 0 .. 15 of the first and last segments against the reference encoder (`wire.reference_blocks`); a
   mismatch refuses the wipe and serves nothing.
3. **Serve.** Honest: the native responder (`pous_native.cu`) answers each challenge from device memory.  Cheat: random
   foreign bytes (`os.urandom`) are written over every block `i` with `i % 19 == 18` (`wire.DROP_EVERY`, about `|C| / 19`),
   hashed as they are written and again after the sessions, and a Python `ControlResponder` answers a kept block from
   device memory and a dropped one by recomputing it from its kept parents on the device (one `band_chain` row, 13 calls,
   its key XORed out of the foreign bytes), reporting the recompute's time.

`prover.json` in `--out` records every wipe: the fill's timings, the gate, the cheat's setup and the foreign bytes' digests
before and after, the responder's counts and each recomputed answer's time.  None of it is trusted by the verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from band_wipe.wire import DROP_EVERY, SCHEME, Channel, dropped, reference_blocks  # noqa: E402

GATE_BLOCKS = 16
SCRATCH_PER_SEGMENT = 2 * 483 * 144 + 65_536 + 64
"""The fill's device scratch per segment: one chunk-major `Π₂` state (`dense._Chunked`, two halves of 483 slots of 144 B)
and one output row of 64 KiB, plus its indices."""


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    if not s:
        return {}
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p50": at(0.5), "p99": at(0.99), "max": s[-1], "sum": sum(s)}


class GpuProver:
    """`C` on the current CUDA device, filled by `band_chain` and served by the native responder (module docstring)."""

    def __init__(self, build_dir: str, *, segments: int = 0, reserve_mib: int = 512, threads: int = 32,
                 base: Any = None) -> None:
        import torch
        from band_gpu import native
        from verity.core.protocols.pous import band

        self.torch, self.ext, self.threads = torch, native.load(build_dir), threads
        self.dev = torch.device("cuda", torch.cuda.current_device())
        base = base or band.SECURE_FIRST
        if base.block_bytes != 65_536:
            raise ValueError("band_chain is the 64 KiB kernel")
        free, total = torch.cuda.mem_get_info(self.dev)
        per = base.segment_bytes + SCRATCH_PER_SEGMENT
        fit = (free - reserve_mib * 2**20) // per
        if fit < 1 or segments > fit:
            raise ValueError(f"{segments or 'any'} segments do not fit: {free} bytes free, {reserve_mib} MiB reserved, "
                             f"{per} bytes a segment with its scratch")
        self.params = base.with_segments(segments or fit)
        self.c = torch.zeros(self.params.blocks, base.block_bytes, dtype=torch.uint8, device=self.dev)
        self.memory = {"total_bytes": total, "free_before_bytes": free, "reserve_bytes": reserve_mib * 2**20,
                       "c_bytes": self.c.numel(), "segments_that_fit": int(fit)}
        self.salt = self.tags = self.drops = None
        self.foreign_sha512: str | None = None

    def facts(self) -> dict[str, Any]:
        from band_gpu import harness

        props = self.torch.cuda.get_device_properties(self.dev)
        return {**harness.gpu_facts(self.dev.index), "name": props.name, "sms": props.multi_processor_count,
                "torch": self.torch.__version__, "cuda": self.torch.version.cuda, "nvcc": harness._nvcc_version()}

    def fill(self, salt: bytes) -> dict[str, Any]:
        torch, ext = self.torch, self.ext
        from band_gpu import codec, dense

        p = self.params
        n, b, w, d = p.segments, p.blocks_per_segment, p.block_bytes, p.d
        t0 = time.perf_counter_ns()
        self.c.zero_()
        tags = codec.segment_tags(salt, list(range(n)), self.dev)
        st = dense._Chunked(n, w, self.dev)
        out = torch.empty(n, w, dtype=torch.uint8, device=self.dev)
        rows = torch.arange(n, dtype=torch.int64, device=self.dev)
        base, blk, ln = rows * b, torch.empty_like(rows), torch.empty_like(rows)
        dom = torch.zeros(n, 32, dtype=torch.uint8, device=self.dev)
        cv = self.c.view(n, b, w)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(b + 1)]
        ev[0].record()
        for v in range(b):
            st.words.zero_()
            st.put(1, w - 32, slice(None), tags)
            dom[:, 0], dom[:, 1] = v & 0xFF, v >> 8
            st.put(1, w, slice(None), dom)
            blk.fill_(v)
            ln.fill_(min(v, d))
            for start, count, _ in st.launches(0, allow_split=False):
                ext.band_chain(st.words, start, count, codec.PI_DOMAIN, self.threads, self.c, base, blk, ln, rows, out)
            cv[:, v].copy_(out)
            ev[v + 1].record()
        ev[-1].synchronize()
        wall_ns = time.perf_counter_ns() - t0
        steps = [ev[v].elapsed_time(ev[v + 1]) / 1e3 for v in range(b)]
        device_s = ev[0].elapsed_time(ev[-1]) / 1e3
        del st, out, rows, base, blk, ln, dom, cv
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(self.dev)
        self.salt, self.tags, self.drops, self.foreign_sha512 = salt, tags, None, None
        gib = self.c.numel() / 2**30
        return {"kernel": "pous_native.cu band_chain", "threads": self.threads, "segments": n, "c_bytes": self.c.numel(),
                "c_gib": gib, "device_s": device_s, "wall_s": wall_ns / 1e9, "s_per_gib": device_s / gib,
                "wall_s_per_gib": wall_ns / 1e9 / gib, "step_s": quantiles(steps), "steps_s": steps,
                "free_after_fill_bytes": free, "torch_reserved_bytes": torch.cuda.memory_reserved(self.dev),
                "torch_allocated_bytes": torch.cuda.memory_allocated(self.dev)}

    def blocks(self, idx: list[int]) -> list[bytes]:
        t = self.torch.tensor(idx, dtype=self.torch.int64, device=self.dev)
        return [bytes(row) for row in self.c.index_select(0, t).cpu().numpy()]

    def drop(self) -> dict[str, Any]:
        torch = self.torch
        t0 = time.perf_counter_ns()
        w = self.params.block_bytes
        self.drops = torch.arange(DROP_EVERY - 1, self.c.shape[0], DROP_EVERY, dtype=torch.int64, device=self.dev)
        h = hashlib.sha512()
        for j in range(0, len(self.drops), FOREIGN_CHUNK):
            part = self.drops[j:j + FOREIGN_CHUNK]
            data = bytearray(os.urandom(len(part) * w))
            h.update(data)
            self.c.index_copy_(0, part, torch.frombuffer(data, dtype=torch.uint8).view(len(part), w).to(self.dev))
        torch.cuda.synchronize(self.dev)
        self.foreign_sha512 = h.hexdigest()
        return {"every": DROP_EVERY, "dropped_blocks": len(self.drops), "foreign_bytes": len(self.drops) * w,
                "foreign_source": "os.urandom", "foreign_sha512": self.foreign_sha512,
                "write_s": (time.perf_counter_ns() - t0) / 1e9}

    def foreign_digest(self) -> str:
        h = hashlib.sha512()
        for j in range(0, len(self.drops), FOREIGN_CHUNK):
            h.update(self.c.index_select(0, self.drops[j:j + FOREIGN_CHUNK]).cpu().numpy())
        return h.hexdigest()

    def honest_responder(self, host: str, port: int, cpu: int | None) -> Any:
        r = self.ext.Responder(self.c.data_ptr(), self.c.shape[0], self.params.block_bytes, self.dev.index, host, port,
                               -1 if cpu is None else cpu)
        r.start()
        return r

    def cheat_responder(self, host: str, port: int) -> tuple[Any, list[tuple[int, int]]]:
        torch, ext = self.torch, self.ext
        from band_gpu import codec, dense
        from band_gpu.harness import ControlResponder

        p = self.params
        b, w, d, rows = p.blocks_per_segment, p.block_bytes, p.d, self.c.shape[0]
        st = dense._Chunked(1, w, self.dev)
        base, blk, ln, dest = (torch.zeros(1, dtype=torch.int64, device=self.dev) for _ in range(4))
        out = torch.empty(1, w, dtype=torch.uint8, device=self.dev)
        dom = torch.zeros(1, 32, dtype=torch.uint8, device=self.dev)
        buf = torch.empty(w, dtype=torch.uint8).pin_memory()
        recomputed: list[tuple[int, int]] = []

        def answer(i: int) -> tuple[bytes, int, int, int]:
            t0 = time.perf_counter_ns()
            if not 0 <= i < rows:
                return bytes(w), 0, 0, 0
            if not dropped(i):
                buf.copy_(self.c[i])
                return bytes(buf.numpy()), 0, time.perf_counter_ns() - t0, 0
            s, v = divmod(i, b)
            st.words.zero_()
            st.put(1, w - 32, slice(None), self.tags[s:s + 1])
            dom[0, 0], dom[0, 1] = v & 0xFF, v >> 8
            st.put(1, w, slice(None), dom)
            base.fill_(s * b)
            blk.fill_(v)
            ln.fill_(min(v, d))
            ext.band_chain(st.words, 0, 1, codec.PI_DOMAIN, 32, self.c, base, blk, ln, dest, out)
            buf.copy_(torch.bitwise_xor(out[0], self.c[i]))
            ns = time.perf_counter_ns() - t0
            recomputed.append((i, ns))
            return bytes(buf.numpy()), ns, ns, 0

        return ControlResponder(w, answer, host=host, port=port), recomputed


FOREIGN_CHUNK = 1024
"""Blocks of foreign bytes written (and read back) at a time: 64 MiB."""


class CpuProver:
    """The CPU tests' stand-in for `GpuProver`: the reference encoder fills `C` in host memory and `ControlResponder`s serve
    it.  A recomputed answer reports `(d + 1)` calls at the certificate's call time as its server time, so the memory
    challenger's logical clock times it with no machine in it."""

    def __init__(self, params: Any) -> None:
        self.params = params
        self.memory = {"c_bytes": params.blocks * params.block_bytes}
        self.c = bytearray()
        self.salt: bytes | None = None
        self.foreign_sha512: str | None = None

    def facts(self) -> dict[str, Any]:
        return {"name": "CPU stand-in (tests)"}

    def fill(self, salt: bytes) -> dict[str, Any]:
        p = self.params
        self.salt = salt
        self.c = bytearray(b"".join(b"".join(reference_blocks(p, salt, s, p.blocks_per_segment)) for s in range(p.segments)))
        return {"kernel": "reference encoder (CPU stand-in)", "segments": p.segments, "c_bytes": len(self.c)}

    def blocks(self, idx: list[int]) -> list[bytes]:
        w = self.params.block_bytes
        return [bytes(self.c[i * w:(i + 1) * w]) for i in idx]

    def drop(self) -> dict[str, Any]:
        w, h = self.params.block_bytes, hashlib.sha512()
        drops = [i for i in range(self.params.blocks) if dropped(i)]
        for i in drops:
            self.c[i * w:(i + 1) * w] = data = os.urandom(w)
            h.update(data)
        self.foreign_sha512 = h.hexdigest()
        self.drops = drops
        return {"every": DROP_EVERY, "dropped_blocks": len(drops), "foreign_bytes": len(drops) * w,
                "foreign_source": "os.urandom", "foreign_sha512": self.foreign_sha512}

    def foreign_digest(self) -> str:
        w = self.params.block_bytes
        return hashlib.sha512(b"".join(self.c[i * w:(i + 1) * w] for i in self.drops)).hexdigest()

    def _responder(self, host: str, port: int, recompute: bool) -> tuple[Any, list[tuple[int, int]]]:
        from band_gpu.harness import ControlResponder

        p, recomputed = self.params, []
        w, b, ns = p.block_bytes, p.blocks_per_segment, (p.d + 1) * p.call_ns

        def answer(i: int) -> tuple[bytes, int, int, int]:
            if not 0 <= i < p.blocks:
                return bytes(w), 0, 0, 0
            if recompute and dropped(i):
                s, v = divmod(i, b)
                recomputed.append((i, ns))
                return reference_blocks(p, self.salt, s, v + 1)[v], ns, ns, 0
            return self.blocks([i])[0], 0, 1_000, 0

        return ControlResponder(w, answer, host=host, port=port), recomputed

    def honest_responder(self, host: str, port: int, cpu: int | None) -> Any:
        return self._responder(host, port, False)[0]

    def cheat_responder(self, host: str, port: int) -> tuple[Any, list[tuple[int, int]]]:
        return self._responder(host, port, True)


def gate(prover: Any, salt: bytes) -> dict[str, Any]:
    """Blocks `0 .. GATE_BLOCKS - 1` of the first and last segments of the fill against the reference encoder."""
    p = prover.params
    t0 = time.perf_counter_ns()
    n = min(GATE_BLOCKS, p.blocks_per_segment)
    checked = {}
    for s in sorted({0, p.segments - 1}):
        got = prover.blocks([s * p.blocks_per_segment + v for v in range(n)])
        checked[s] = got == reference_blocks(p, salt, s, n)
    return {"passed": all(checked.values()), "segments": {str(s): ok for s, ok in checked.items()}, "blocks_per_segment": n,
            "s": (time.perf_counter_ns() - t0) / 1e9}


def serve(prover: Any, listener: socket.socket, out: Path, *, host: str = "127.0.0.1", resp_port: int = 0,
          responder_cpu: int | None = None) -> dict[str, Any]:
    """One memory challenger's wipes over the first connection to `listener` (`wire.py`'s channel), recorded in
    `out/prover.json` when it says bye or the channel closes."""
    p = prover.params
    conn, _ = listener.accept()
    ch = Channel(conn)
    doc: dict[str, Any] = {"scheme": SCHEME, "params": p.manifest(), "memory": prover.memory, "device": prover.facts(),
                           "wipes": []}
    ch.send({"op": "hello", "scheme": SCHEME, "segments": p.segments, "params": p.manifest(), "memory": prover.memory,
             "device": doc["device"]})
    responder, rec, recomputed = None, None, []
    try:
        while True:
            msg = ch.recv()
            if msg["op"] == "fill":
                salt = bytes.fromhex(msg["salt"])
                rec = {"wipe": msg["wipe"], "cheat": bool(msg["cheat"]), "salt": msg["salt"], "fill": prover.fill(salt)}
                rec["gate"] = gate(prover, salt)
                if not rec["gate"]["passed"]:
                    doc["wipes"].append(rec)
                    ch.send({"op": "refused", "reason": "the fill's blocks are not the reference encoder's", "gate": rec["gate"]})
                    continue
                if rec["cheat"]:
                    rec["cheat_setup"] = prover.drop()
                    responder, recomputed = prover.cheat_responder(host, resp_port)
                else:
                    responder, recomputed = prover.honest_responder(host, resp_port, responder_cpu), []
                rec["responder"] = "ControlResponder (cheat)" if rec["cheat"] else "pous_native.cu Responder"
                ch.send({"op": "filled", "port": int(responder.port()), "fill": rec["fill"], "gate": rec["gate"],
                         "cheat": rec.get("cheat_setup")})
            elif msg["op"] == "done":
                rec["counts"] = {"answered": int(responder.answered()), "pings": int(responder.pings())}
                responder.stop()
                responder = None
                if rec["cheat"]:
                    after = prover.foreign_digest()
                    rec["foreign_after_sha512"] = after
                    rec["foreign_kept"] = after == rec["cheat_setup"]["foreign_sha512"]
                    rec["recomputed"] = [{"index": i, "ns": ns} for i, ns in recomputed]
                doc["wipes"].append(rec)
                ch.send({"op": "report", **{k: v for k, v in rec.items() if k not in ("fill", "gate")}})
            elif msg["op"] == "bye":
                break
    finally:
        if responder is not None:
            responder.stop()
        ch.close()
        out.mkdir(parents=True, exist_ok=True)
        (out / "prover.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--build-dir", required=True, help="pous_native's JIT build directory")
    ap.add_argument("--listen", default="127.0.0.1:47610", help="the control channel's address")
    ap.add_argument("--resp-port", type=int, default=47611, help="the responder's port, on --listen's host")
    ap.add_argument("--segments", type=int, default=0, help="segments of C (0: every whole one the device holds)")
    ap.add_argument("--reserve-mib", type=int, default=512, help="device memory left out of C beyond the fill's scratch")
    ap.add_argument("--threads", type=int, default=32, help="band_chain's threads per block")
    ap.add_argument("--responder-cpu", type=int, default=None)
    a = ap.parse_args(argv)
    host, _, port = a.listen.rpartition(":")
    prover = GpuProver(a.build_dir, segments=a.segments, reserve_mib=a.reserve_mib, threads=a.threads)
    listener = socket.create_server((host, int(port)))
    print(f"listening {a.listen}: {prover.params.segments} segments, {prover.memory['c_bytes']} bytes of C", flush=True)
    doc = serve(prover, listener, a.out, host=host, resp_port=a.resp_port, responder_cpu=a.responder_cpu)
    listener.close()
    return 0 if doc["wipes"] and all(w["gate"]["passed"] for w in doc["wipes"]) else 1


if __name__ == "__main__":
    sys.exit(main())
