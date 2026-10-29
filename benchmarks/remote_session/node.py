"""The prover of a serving node (run.sh's SERVE=node): the registered store is every engine's weights arena
(`engine.py`, one vLLM engine a GPU) where it is served, read in place by `device.py`'s whole-node pass, beside an
optional host tier of zero filler: the store `remote_bandwidth.store` lays out from the checkpoint, which `--model-dir`
checks.

~~~text
arenas     each engine exports its arena over CUDA IPC (`engine.<g>.json` in --engines-dir); once every engine has run
           its plain baseline (status baseline), the prover opens each in its GPU's context and takes it as that GPU's
           HBM shard (`device.DeviceStore`'s `hbm`): the store's chunks are the arenas' in GPU order, then the host
           tier's, and the hello's layout labels each arena's weights chunks weights and the rest filler. With
           --model-dir each arena is first held to the checkpoint's (`held_to_checkpoint`)
register   the payload tree over the plain weights as served, salted under the verifier's payload key (its first
           message); at the registration the engines pause (ctl.json's
           pause), the device gates run, the arenas are encoded in place at the key, and the engines get the key and
           their arena's first block, check that decoding gives back their plain bytes and probe (`engine.py`), and
           serve again, decoding each weight as it is used; their checks join the gates
phases     ctl.json names what the node is doing, and each engine tags its steps with it: decode-only (--decode-s of
           serving with no pass), calibrate (`GpuBackend.calibrate`, under serving), session-K-MODE (each session, from
           its arming to its end), after; then stop, and each engine's summary goes to node.json beside the prover's
~~~

    python node.py --connect HOST:PORT --out OUT --plan honest,skip:0.0625 --engines-dir B --engines 8 \
        --lib session.so --kv kv.so [--model-dir DIR] [device options, `device.py`]
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import store as rstore  # noqa: E402

import device  # noqa: E402
import store_layout  # noqa: E402
import wire  # noqa: E402
from engine import IpcHandle, write_json  # noqa: E402
from prover import ProverSide, parse_plan  # noqa: E402

BLOCK = device.BLOCK


def held_to_checkpoint(model_dir: Path, exports: list[dict], statuses: list[dict], workers: int) -> dict:
    """Each engine's arena held to the one `remote_bandwidth.store.arena` lays out from the checkpoint in `model_dir`: its
    weights' chunks, its entries (name, offset, bytes) and its plain SHA-256s a GiB, else RuntimeError with why. So a
    store a verifier with `--model` would refuse is refused here, before the hello."""
    arena = rstore.arena_of(model_dir)
    want = [[e.name, e.offset, e.nbytes] for e in arena.entries]
    plain = {s["gpu"]: s.get("plain_sha256") for s in statuses}
    sha = {}
    for e in exports:
        if e["weights_blocks"] != arena.weights_blocks or e.get("entries") != want:
            got = e.get("entries") or []
            first = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
            raise RuntimeError(f"engine {e['gpu']}'s arena is not {model_dir}'s: {e['weights_blocks']} weights blocks "
                               f"and {len(got)} entries against {arena.weights_blocks} and {len(want)}, first "
                               f"differing at entry {first} ({got[first] if first < len(got) else None} against "
                               f"{want[first] if first < len(want) else None})")
        if e["bytes"] not in sha:
            sha[e["bytes"]] = store_layout.arena_sha256(arena, e["bytes"], workers)
        if plain[e["gpu"]] != sha[e["bytes"]]:
            bad = [i for i, (x, y) in enumerate(zip(plain[e["gpu"]] or [], sha[e["bytes"]])) if x != y]
            raise RuntimeError(f"engine {e['gpu']}'s arena bytes are not {model_dir}'s laid out: its plain SHA-256s "
                               f"differ at GiB {bad[:8] or 'count'}")
    return {"model_dir": str(model_dir), "weights_blocks": arena.weights_blocks, "entries": len(want),
            "weights_bytes": arena.weights_bytes}


class Engines:
    """The engines' handshake directory: their exports and states (`status.<g>.json`), and the prover's ctl.json."""

    def __init__(self, d: Path, count: int, log=print) -> None:
        self.d, self.count, self.log, self.ctl_state = d, count, log, {"seq": 0}

    def ctl(self, **kw) -> None:
        self.ctl_state = {**self.ctl_state, **kw, "seq": self.ctl_state["seq"] + 1}
        write_json(self.d / "ctl.json", self.ctl_state)

    def statuses(self) -> list[dict | None]:
        out = []
        for g in range(self.count):
            try:
                out.append(json.loads((self.d / f"status.{g}.json").read_text()))
            except (FileNotFoundError, json.JSONDecodeError):
                out.append(None)
        return out

    def alive(self, g: int) -> bool:
        try:
            pid = int((self.d / f"engine.{g}.pid").read_text())
        except (FileNotFoundError, ValueError):
            return True
        return Path(f"/proc/{pid}").exists()

    def wait(self, state: str, timeout_s: float) -> list[dict]:
        """Every engine's status once each says `state`; an engine that died first, or the timeout, raises."""
        t0 = time.monotonic()
        while True:
            got = self.statuses()
            if all(s is not None and s["state"] == state for s in got):
                return got
            dead = [g for g in range(self.count) if not self.alive(g)]
            if dead:
                raise RuntimeError(f"engines {dead} ended while the node waited for {state}")
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"engines not {state} after {timeout_s} s: {[s and s['state'] for s in got]}")
            time.sleep(0.1)


class _Pointer:
    """A device range as `torch.as_tensor` takes it (`__cuda_array_interface__`)."""

    def __init__(self, ptr: int, n: int) -> None:
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "strides": None,
                                         "version": 2}


def open_arena(torch, export: dict, index: int):
    """An engine's arena (`engine.export`'s), opened in GPU `index`'s context, as a uint8 tensor on it."""
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuIpcOpenMemHandle_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), IpcHandle, ctypes.c_uint]
    with torch.cuda.device(index):
        torch.empty(1, device=f"cuda:{index}")
        ptr = ctypes.c_uint64()
        handle = IpcHandle.from_buffer_copy(bytes.fromhex(export["handle"]))
        if rc := cu.cuIpcOpenMemHandle_v2(ctypes.byref(ptr), handle, 1):
            raise RuntimeError(f"cuIpcOpenMemHandle of GPU {index}'s arena: {rc}")
        t = torch.as_tensor(_Pointer(ptr.value + export["offset"], export["bytes"]), device=f"cuda:{index}")
    if t.data_ptr() != ptr.value + export["offset"] or t.numel() != export["bytes"]:
        raise RuntimeError("the arena's tensor is not the mapped range")
    return t


def key_json(key) -> dict:
    """A codec key as `engine.Decoder.set_key` takes it."""
    f = key.codec.field
    s = f.word_bytes
    word = lambda x: x.to_bytes(s, "little").hex()
    return {"bits": f.n, "word_bytes": s, "rounds": key.rounds, "masked": key.codec.masked,
            "c": "".join(word(x) for x in key.c), "tau": word(key.tau),
            "mask": word(key.mask) if key.codec.masked else None}


class ServingNode(device.GpuBackend):
    """`device.GpuBackend` over the engines' arenas (`open_arena`, one a GPU in GPU order), pausing them for the
    encode and handing them the key (`encode`), and naming each phase to them."""

    def __init__(self, torch, engines: Engines, exports: list[dict], arenas: list, *, decode_s: float,
                 wait_s: float, **kw) -> None:
        self.engines, self.exports, self.decode_s, self.wait_s = engines, exports, decode_s, wait_s
        super().__init__(torch, blocks=sum(t.numel() for t in arenas) // BLOCK, gpus=len(arenas), hbm=arenas, **kw)
        self.bases = {e["uuid"]: self.store.base[g] for g, e in enumerate(exports)}

    def layout(self) -> list[dict]:
        s, out = self.store, []
        for q in range(len(s.n)):
            if s.on_host[q]:
                out.append({"kind": "filler", "tier": "host", "first": s.base[q], "blocks": s.n[q]})
                continue
            w = self.exports[q]["weights_blocks"]
            out.append({"kind": "weights", "tier": "hbm", "first": s.base[q], "blocks": w})
            if s.n[q] > w:
                out.append({"kind": "filler", "tier": "hbm", "first": s.base[q] + w, "blocks": s.n[q] - w})
        return out

    def encode(self, registration) -> dict:
        self.engines.ctl(phase="encode", pause=True)
        self.engines.wait("paused", self.wait_s)
        out = super().encode(registration)
        self.engines.ctl(phase="resume", pause=False, key=key_json(registration.key), bases=self.bases)
        got = self.engines.wait("decoding", self.wait_s)
        self.gates["engines"] = [{k: s[k] for k in ("gpu", "arena_decodes", "probe_tokens_equal", "first_block",
                                                     "hooked_modules")} for s in got]
        self.log(f"node: engines decode {self.gates['engines']}")
        if not all(s["arena_decodes"] and s["probe_tokens_equal"] for s in got):
            raise RuntimeError(f"an engine's decode is not its plain weights: {self.gates['engines']}")
        self.engines.ctl(phase="decode-only")
        time.sleep(self.decode_s)
        return out

    def calibrate(self) -> dict:
        self.engines.ctl(phase="calibrate")
        return super().calibrate()


def phases(summaries: list[dict]) -> dict:
    """The node's serving by phase: tokens a second summed over the engines, and their step times' medians."""
    out = {}
    for e in summaries:
        for phase, p in e.get("phases", {}).items():
            x = out.setdefault(phase, {"engines": 0, "tokens": 0, "tokens_per_s": 0.0, "steps_per_s": [],
                                       "step_ms_p50": [], "step_ms_p99": []})
            x["engines"] += 1
            x["tokens"] += p["tokens"]
            x["tokens_per_s"] += p["tokens_per_s"] or 0.0
            x["steps_per_s"].append(p["steps_per_s"])
            x["step_ms_p50"].append(p["step_ms"]["p50"])
            x["step_ms_p99"].append(p["step_ms"]["p99"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connect", default="127.0.0.1:7781")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", default="honest")
    ap.add_argument("--node", default="vy-nebius/rtx-pro-6000/serving")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--warmup-ms", type=float, default=2500.0)
    ap.add_argument("--engines-dir", type=Path, required=True)
    ap.add_argument("--engines", type=int, required=True)
    ap.add_argument("--decode-s", type=float, default=60.0, help="serving decoded with no pass, before calibration")
    ap.add_argument("--after-s", type=float, default=20.0, help="serving after the last session, before stop")
    ap.add_argument("--wait-s", type=float, default=900.0, help="the longest wait for every engine's next state")
    ap.add_argument("--overlap-openings", action="store_true",
                    help="open each window as its draws arrive, beside the later passes (`prover.py "
                         "--overlap-openings`), so the verifier checks while the session runs")
    ap.add_argument("--model-dir", type=Path, metavar="DIR",
                    help="the checkpoint the engines serve: before the hello each engine's arena is held to the one "
                         "`remote_bandwidth.store` lays out from it (`held_to_checkpoint`), the store a verifier with --model "
                         "binds the registration to")
    ap.add_argument("--layout-workers", type=int, default=16, help="--model-dir's hashing processes")
    ap.add_argument("--seed-payload-key", action="store_true",
                    help="tests only: salt the payload tree with the device's own key, from --seed, not the "
                         "verifier's, which a verifier with --model refuses")
    device.add_arguments(ap)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.setswitchinterval(2e-4)
    log = lambda m: print(m, flush=True)
    engines = Engines(a.engines_dir, a.engines, log)
    out, code = {"schema": "verity/remote-session/node/v0"}, 0
    try:
        got = engines.wait("baseline", a.wait_s)
        log(f"node: engines at baseline {[(s['gpu'], s['uuid']) for s in got]}")
        engines.ctl(phase="setup")
        import torch

        index = {str(torch.cuda.get_device_properties(i).uuid): i for i in range(torch.cuda.device_count())}
        exports = sorted((json.loads((a.engines_dir / f"engine.{s['gpu']}.json").read_text()) for s in got),
                         key=lambda e: index[e["uuid"]])
        if [index[e["uuid"]] for e in exports] != list(range(len(index))):
            raise RuntimeError(f"an engine a visible GPU: {[e['uuid'] for e in exports]} against {index}")
        if a.model_dir is not None:
            out["checkpoint"] = held_to_checkpoint(a.model_dir, exports, got, a.layout_workers)
            log(f"node: every arena is {a.model_dir}'s laid out: {json.dumps(out['checkpoint'])}")
        arenas = [open_arena(torch, e, g) for g, e in enumerate(exports)]
        host = int(a.host_gib * 2 ** 30) // BLOCK // a.chunk_blocks * a.chunk_blocks
        backend = ServingNode(torch, engines, exports, arenas, decode_s=a.decode_s, wait_s=a.wait_s,
                              lib_path=a.lib, kv_path=a.kv, seed=a.seed, log=log, gate_chunk=a.gate_chunk,
                              gate_samples=a.gate_samples, calibration_passes=a.calibration_passes,
                              warm_openings=a.warm_openings, chunk=a.chunk_blocks, kept_from=a.kept_from,
                              sketch_ctas=a.sketch_ctas, payload_kept_from=a.payload_kept_from,
                              trace_allocator=a.trace_allocator, host_blocks=host, host_ctas=a.host_ctas,
                              host_consumers=a.host_consumers)
        out["engines"] = exports
        out["layout"] = backend.layout()
        host_, port = a.connect.rsplit(":", 1)
        conn = wire.connect(host_, int(port))
        p = ProverSide(conn, backend, a.node.encode(), parse_plan(a.plan), seed=a.seed,
                       warmup_ns=round(a.warmup_ms * 1e6), defer_openings=not a.overlap_openings,
                       seed_payload_key=a.seed_payload_key, log=log)
        out["overlap_openings"] = a.overlap_openings
        cal = p.register()
        log(f"node: registered, calibration {json.dumps(cal)}")
        out.update({"calibration": cal, "gates": backend.gates, "calibration_trace": backend.calibration_trace,
                    "registration": wire.registration_json(p.registration), "sessions": []})
        (a.out / "prover.json").write_text(json.dumps(out, indent=1))
        for mode, fraction in p.plan:
            try:
                head, _, _ = p.expect("session")
            except wire.Closed:
                break
            engines.ctl(phase=f"session-{head['k']}-{mode}" + (f":{fraction:g}" if fraction else ""))
            gc.collect()
            gc.freeze()
            gc.disable()
            try:
                r = p.session(head, mode, fraction)
            finally:
                gc.enable()
                gc.unfreeze()
            engines.ctl(phase="between")
            out["sessions"].append(r)
            log(f"node: session {r['k']} ({mode}:{fraction}): accepted={r['accepted']} passes={r['passes']} "
                f"failure={r['failure']} on_dropped={r['failure_on_dropped']}")
            out["wire"] = conn.totals()
            (a.out / "prover.json").write_text(json.dumps(out, indent=1))
        engines.ctl(phase="after")
        time.sleep(a.after_s)
    except BaseException as e:
        out["error"], code = repr(e), 1
        traceback.print_exc()
    finally:
        engines.ctl(stop=True)
        try:
            done = engines.wait("stopped", 300)
            out["serving"] = phases([json.loads((a.out / f"engine.{s['gpu']}.json").read_text()) for s in done])
        except (RuntimeError, TimeoutError, FileNotFoundError) as e:
            out["serving_error"] = str(e)
        (a.out / "prover.json").write_text(json.dumps(out, indent=1))
        log(f"node: serving {json.dumps(out.get('serving'))}")
        sys.stdout.flush()
        sys.stderr.flush()
        # A whole-node store's teardown can outlive the lease (prover.py's main).
        os._exit(code)


if __name__ == "__main__":
    sys.exit(main())
