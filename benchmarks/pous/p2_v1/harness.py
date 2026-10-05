"""P2 v1's arm of the POUS-only harness (`bench.py --impl p2-v1-gpu`): the frozen v1 scheme (`reference.py`) with its GPU
kernels (`p2v1.cu`, `gpu.py`) on one device, the same steps, controls and metrics as the band's arm (`band_gpu/harness.py`), so
the two compare side by side.

1. **Kernel gates.** The pinned vector of the width (`vectors.json`, the P2 kit's): its keys expanded on the device, its `m`
   encoded and its `c` decoded, all bit-exact. Then the run's own blocks: the encoded ones decode back to their payloads (the
   round trip), and sampled blocks of `C` (the first and last encoded ones, the first and last of the rest, and
   pseudo-random ones) decode on the device to the reference's decode under the reference's keys, with the device's keys
   equal to the reference's; and no kernel flag raised. **A failed kernel gate stops the run; it times nothing.**
2. **Setup and encode.** `W` is the weight set's bytes, packed as 2,054-byte payloads (`p2-2054B`), in `B` blocks, whole
   segments of 65,536. The commitment to `W` comes first, then the setup salt from it (§4). Encoding is two roots per block,
   so the device encodes the first `--p2-encode-blocks` blocks (encode GB/s is their payload bytes over the device time) and
   the rest of `C` is uniformly random canonical blocks: every `C` is the encoding of its decode, the decode's cost does not
   depend on the bytes, and the audit is over `C`'s leaves.
3. **The synthetic loop.** As the band's: a POUS forward expands every block's keys, decodes all of `C`, takes the payload
   bytes as `W` and runs the weight set's GEMMs on them; plain reads `W`. Both are timed from a CUDA graph where the path is
   capturable, every rep kept. Decode overhead is POUS over plain, minus one.
4. **The limit and the audits.** The limit is fixed before any load (`Δ = 0.5 ms` plus the loopback allowance, which spec v1.1
   caps at 354 µs); `p2_v1/live.py` audits at k = 105 against it: the honest audit under POUS forwards, then the value control
   (one block in ten zeroed), the late control (one answer in ten held past the limit; the only answers held) and the
   recompute control (one block in ten of all of `C` dropped, each challenged dropped block re-encoded on the device from its
   decoded row `Dec(C_i)`, which the game gives the prover free: two roots, and the margin is the limit minus each
   recomputed answer's recorded device time).
5. **SeqRoot.** One block's encode on one warp is two roots in sequence: half of the fastest of `--reps` encodes of a
   pseudo-random block against twice the audit's limit, which SeqRoot says a root outlasts. The kernel's root is an addition
   chain on CGBN's generic squaring, not tuned, so this is an upper bound on the fastest GPU root, not a floor. The SM clock is
   sampled beside it.
6. **NVMe and clocks**, as the band's.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from band_gpu import harness as H

from . import reference as R

HERE = Path(__file__).resolve().parent
LIVE = HERE / "live.py"
SETUP_SOURCE = hashlib.sha256(b"verity-pous/bench/p2-v1/setup-source").digest()
"""The bench's setup source: in a deployment the verifier's secret or a beacon round drawn after the commitment to `W`."""
PINS = ("P2MeetsM1pB19Uncond", "P2MeetsM1pFamilyUncond", "P2MeetsM1pW8192Uncond", "P2SlackB19Uncond", "P2SlackFamilyUncond",
        "P2ErrorSplitB19Uncond")
ASSUMPTIONS = ("P2-EXP-IO", "SHAKE256/P2-EXP-IO storage-game idealization", "P2–M1-SGI (not accepted)", "SeqRoot",
               "perfect isolation", "cr/sha-256")
RECOMPUTE_EVERY = 10
"""The recompute control drops one block in this many; P2's blocks are independent, so any pattern keeps what a block needs."""


@dataclass
class Backend:
    """What P2 v1's arm computes with: `bind(params, salt, model)` gives the kernels of one setup (`keys(first, count)`,
    `encode(m_rows, first)`, `decode(c_rows, first)`, `raised()`) on `[n, W/8]` big-endian rows; `responder`, `timer`,
    `graphed`, `copy_bw`, `clock_mhz` and `facts` as the band's `Backend`."""

    device: Any
    bind: Callable[[R.Params, bytes, bytes], Any]
    responder: Callable[[Any, int | None], Any]
    timer: Callable[[Callable[[], Any]], float]
    graphed: Callable[[Callable[[], Any]], Callable[[], Any]] | None = None
    copy_bw: Callable[[], float] | None = None
    clock_mhz: Callable[[], float] | None = None
    facts: Callable[[], dict[str, Any]] = dict


def gpu(lib_dir: str, build_dir: str) -> Backend:
    import torch

    from band_gpu import native

    from .gpu import Kernels

    device = torch.device("cuda", torch.cuda.current_device())

    def responder(c: Any, cpu: int | None) -> Any:
        r = native.ext().Responder(c.data_ptr(), c.shape[0], c.shape[1], device.index, "127.0.0.1", 0, -1 if cpu is None else cpu)
        r.start()
        return r

    def timer(fn: Callable[[], Any]) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 1e3

    def graphed(fn: Callable[[], Any]) -> Callable[[], Any]:
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g.replay

    def copy_bw() -> float:
        n = 1 << 28
        src, dst = torch.empty(n, dtype=torch.uint8, device=device), torch.empty(n, dtype=torch.uint8, device=device)
        return 2 * n / statistics.median(H._reps_raw(timer, lambda: dst.copy_(src), 5)) / 1e9

    def facts() -> dict[str, Any]:
        props = torch.cuda.get_device_properties(device)
        return {**H.gpu_facts(device.index), "sms": props.multi_processor_count, "torch": torch.__version__,
                "cuda": torch.version.cuda, "nvcc": H._nvcc_version()}

    smi_id = H.device_uuid(device.index)
    return Backend(device, lambda params, salt, model: Kernels(lib_dir, build_dir, params, salt, model), responder, timer,
                   graphed, copy_bw, lambda: H._sm_clock_mhz(smi_id), facts)


def rows(values: list[int], width: int, device: Any) -> Any:
    import torch
    return torch.tensor([list(v.to_bytes(width, "big")) for v in values], dtype=torch.uint8, device=device)


def vector_gate(backend: Backend, params: R.Params, vectors: dict[str, Any] | None = None) -> dict[str, Any]:
    """The pinned vector at `params.w`, through the device: keys, encode and decode, bit-exact."""
    doc = vectors or json.loads(R.VECTORS.read_text())
    v = next((x for x in doc["vectors"] if int(x["w"]) == params.w and int(x["c0"]) == params.c0), None)
    if v is None:
        return {"passed": False, "reason": f"no pinned vector at w = {params.w}, c0 = {params.c0}"}
    salt, model = bytes.fromhex(v["setup_salt"]), bytes.fromhex(v["model_commitment"])
    bound = backend.bind(params, salt, model)
    first = int(v["segment"]) * params.blocks_per_segment + int(v["block"])
    k, t = bound.keys(first, 1)
    nb = params.width_bytes
    m, c = rows([int(v["m"], 16)], nb, backend.device), rows([int(v["c"], 16)], nb, backend.device)
    got = {"K": k[0].cpu().numpy().tobytes().hex() == v["K"], "t": t[0].cpu().numpy().tobytes().hex() == v["t"],
           "encode": bytes(bound.encode(m, first)[0].cpu().numpy()) == bytes.fromhex(v["c"]),
           "decode": bytes(bound.decode(c, first)[0].cpu().numpy()) == bytes.fromhex(v["m"]), "flags": bound.raised() == 0}
    return {"passed": all(got.values()), "vector": v["id"], **got}


def picks(encoded: int, blocks: int, extra: int = 8) -> list[int]:
    """The blocks the reference decodes: the first and last encoded ones, the first and last of the rest, and `extra` more."""
    out = {0, encoded - 1, encoded % blocks, blocks - 1}
    for j in range(extra):
        out.add(int.from_bytes(hashlib.sha256(f"verity-pous/bench/p2-v1/pick/{j}".encode()).digest()[:8], "big") % blocks)
    return sorted(i for i in out if 0 <= i < blocks)


def reference_gate(bound: Any, params: R.Params, salt: bytes, model: bytes, c_host: bytes, m_dev: Any,
                   chosen: list[int]) -> dict[str, Any]:
    """`chosen` blocks of `C`, decoded on the device, equal to the reference's decode under the reference's keys."""
    nb, bps = params.width_bytes, params.blocks_per_segment
    equal, keys_equal = [], []
    for i in chosen:
        k, t, _ = R.keys(salt, params, model, i // bps, i % bps)
        want = R.decode(params, int.from_bytes(c_host[i * nb:(i + 1) * nb], "big"), k, t).to_bytes(nb, "big")
        equal.append(bytes(m_dev[i].cpu().numpy()) == want)
        dk, dt = bound.keys(i, 1)
        keys_equal.append(bytes(dk[0].cpu().numpy()) == k.to_bytes(nb, "big") and bytes(dt[0].cpu().numpy()) == t.to_bytes(nb, "big"))
    return {"passed": all(equal) and all(keys_equal), "blocks": chosen, "equal": equal, "keys_equal": keys_equal}


def payload_bytes(m: Any, params: R.Params, used: int) -> Any:
    """`W` from decoded rows: each row's payload bytes, back to back, the first `used` of them."""
    return m[:, params.width_bytes - params.payload_bytes:].reshape(-1)[:used]


def profile(k: int, limit_ns: int, gaps: tuple[str, ...]) -> dict[str, Any]:
    return {"scheme": R.PROTOCOL, "k": k, "limit_ns": limit_ns, "rho": "18/19", "error": "1% + 2^-128", "pins": list(PINS),
            "assumptions": list(ASSUMPTIONS), "gaps": list(gaps), "certified": False,
            "experimental": "P2 remains experimental: the P2–M1-SGI instantiation assumption is not accepted (spec §10.8)"}


def plaintext_rows(bound: Any, c: Any, dropped: list[int]) -> Any:
    """The dropped blocks' decoded rows `Dec(C_i)`, on the device: the plaintext the game gives the adversary free.  Every block
    of `C` is the encoding of its decode, so re-encoding these rows gives back `C_i` for the encoded head (where `Dec(C_i)` is
    `W_i`'s payload row) and for the random canonical rest alike."""
    import torch

    return bound.decode(c, 0)[torch.tensor(dropped, dtype=torch.long, device=c.device)].contiguous()


def _recomputer(backend: Backend, bound: Any, rows_: Any, dropped: list[int]) -> Callable[[int], tuple[bytes, int]]:
    """The recompute control's adversary: it holds `C` without the dropped blocks and their plaintext rows (`plaintext_rows`,
    free), and answers a challenged dropped block by encoding its row again on the device, two roots."""
    row = {i: j for j, i in enumerate(dropped)}

    def rc(i: int) -> tuple[bytes, int]:
        m, held = rows_[row[i]:row[i] + 1], []
        seconds = backend.timer(lambda: held.append(bound.encode(m, i)))
        return bytes(held[0][0].cpu().numpy()), round(seconds * 1e9)

    return rc


def _seqroot(backend: Backend, bound: Any, plain_rows: Any, dropped: list[int], reps: int, limit: int | None) -> dict[str, Any]:
    """SeqRoot's block (`seqroot_block`) encoded again from its plaintext row, `max(3, reps)` times: its two roots, against
    twice the limit when there is one (`None` in an untimed run)."""
    clocks = []
    j = seqroot_block(dropped)
    m_j = plain_rows[dropped.index(j):dropped.index(j) + 1]
    root_reps = H._reps_raw(backend.timer, lambda: bound.encode(m_j, j), max(3, reps))
    one = min(root_reps)
    if backend.clock_mhz is not None and backend.clock_mhz():
        clocks.append(backend.clock_mhz())
    return {"block": j, "encode_one_block_s": one, "encode_one_block_reps_s": root_reps, "root_s": one / 2,
            "twice_limit_s": 2 * limit / 1e9 if limit is not None else None,
            "root_over_twice_limit": (one / 2) / (2 * limit / 1e9) if limit else None, "sampled_sm_mhz": clocks or None,
            "label": "untuned root (an addition chain on CGBN's generic squaring): an upper bound on the fastest GPU root, "
                     "not a SeqRoot floor"}


def seqroot_block(dropped: list[int]) -> int:
    """SeqRoot's block: a pseudo-random one of the dropped blocks, whose plaintext row the recompute control holds."""
    return dropped[int.from_bytes(hashlib.sha256(b"verity-pous/bench/p2-v1/seqroot").digest()[:8], "big") % len(dropped)]


def _failed(gates: dict[str, Any], facts: dict[str, Any], P: R.Params, segments: int, total: int, used: int,
            commit: str | None) -> dict[str, Any]:
    return {"implementation": {"name": "p2-v1-gpu", "commit": commit, "harness": "benchmarks/pous/p2_v1/harness.py"},
            "device": facts, "segments": segments, "segments_of_weight_set": total, "blocks": P.blocks, "weight_bytes": used,
            "gates": gates, "metrics": None, "profile": None, "measurements": [], "stopped": H.stopped_reason(gates)}


def run(a: Any, ws: dict[str, Any], out: Path, *, w_segment: Callable[[str, int, int, int], bytes],
        backend: Backend | None = None, params: R.Params | None = None, vectors: dict[str, Any] | None = None) -> dict[str, Any]:
    """One run of P2 v1's arm.  `backend`, `params` and `vectors` replace the GPU kernels, the deployment point and the pinned
    vectors (the CPU tests)."""
    import torch

    base = params or R.DEPLOYMENT
    payload, nb, bps = base.payload_bytes, base.width_bytes, base.blocks_per_segment
    total = -(-(-(-ws["bytes"] // payload)) // bps)
    segments = min(total, a.max_segments) if a.max_segments else total
    P = base.with_segments(segments)
    blocks = P.blocks
    used = min(ws["bytes"], blocks * payload)
    backend = backend or gpu(a.p2_lib, a.build_dir)
    facts: dict[str, Any] = {"before": backend.facts()}
    gates: dict[str, Any] = {"vector": vector_gate(backend, base, vectors)}

    w_host = b"".join(w_segment(ws["id"], s, bps * payload, used) for s in range(segments))
    model = R.w_commitment(P, w_host)
    salt = R.setup_salt(SETUP_SOURCE, P, model)
    bound = backend.bind(P, salt, model)
    pad = nb - payload
    m_dev = torch.zeros(blocks, nb, dtype=torch.uint8, device=backend.device)
    m_dev[:, pad:] = torch.frombuffer(bytearray(w_host), dtype=torch.uint8).to(backend.device).view(blocks, payload)
    encoded = min(a.p2_encode_blocks or blocks, blocks)
    encode_s, head = _encode_head(backend, bound, m_dev, encoded)
    c = torch.empty(blocks, nb, dtype=torch.uint8, device=backend.device)
    c[:encoded] = head
    if encoded < blocks:
        g = torch.Generator().manual_seed(20260930)
        tail = torch.randint(0, 256, (blocks - encoded, nb), dtype=torch.uint8, generator=g)
        tail[:, 0] &= 0x7F
        c[encoded:] = tail.to(backend.device)
    gates["round_trip"] = {"passed": bool(torch.equal(bound.decode(c[:encoded].contiguous(), 0), m_dev[:encoded])),
                           "blocks": encoded}
    c_host = c.cpu().numpy().tobytes()
    gates["reference_decode"] = reference_gate(bound, P, salt, model, c_host, bound.decode(c, 0), picks(encoded, blocks))
    gates["kernel_flags"] = {"passed": bound.raised() == 0}
    if not all(g["passed"] for g in gates.values()) or getattr(a, "gates_only", False):
        facts["after"] = backend.facts()
        return _failed(gates, facts, P, segments, total, used, a.impl_commit)

    tokens = [int(t) for t in a.tokens.split(",") if t]
    pl = H.placements(ws["tensors"], used)
    xs = H.inputs(pl, set(tokens) | ({a.load_tokens} if a.load_tokens else set()), backend.device)
    copy_bw = backend.copy_bw() if backend.copy_bw is not None else None
    plain_reps, plain_how = H._plain_arm(backend, torch.frombuffer(bytearray(w_host), dtype=torch.uint8).to(backend.device),
                                         pl, xs, tokens, a.reps)
    del m_dev

    def decode_pass() -> Any:
        return payload_bytes(bound.decode(c, 0), P, used)

    def pous_forward(t: int) -> None:
        flat = decode_pass()
        H.forward(flat, pl, xs[t])
        flat.zero_()

    decode_reps = H._reps(backend, decode_pass, a.reps)
    pous_reps, pous_how = {}, {}
    for t in tokens:
        pous_reps[t], pous_how[t] = H._timed_arm(backend, lambda t=t: pous_forward(t), a.reps)
    plain = {t: statistics.median(plain_reps[t]) for t in tokens}
    pous = {t: statistics.median(pous_reps[t]) for t in tokens}
    decode_s = statistics.median(decode_reps)
    launch_bound = {t: (copy_bw is not None and (used / plain[t]) / 1e9 < H.LAUNCH_BOUND_FRACTION * copy_bw) for t in tokens}
    loop = {"tokens": tokens, "tensors": len(pl), "tensor_bytes": sum(size for *_, size in pl), "plain_s": plain, "pous_s": pous,
            "decode_s": decode_s, "overhead": {t: pous[t] / plain[t] - 1 for t in tokens},
            "slowdown": {t: pous[t] / plain[t] for t in tokens}, "plain_graphed": plain_how, "pous_graphed": pous_how,
            "plain_reps_s": plain_reps, "pous_reps_s": pous_reps, "device_copy_GBps": copy_bw, "launch_bound": launch_bound}
    dropped = list(range(0, blocks, RECOMPUTE_EVERY))

    def result(metrics: dict[str, Any], rows_: list[tuple[str, Any, str]], prof: dict[str, Any] | None) -> dict[str, Any]:
        return {"implementation": {"name": "p2-v1-gpu", "commit": a.impl_commit, "harness": "benchmarks/pous/p2_v1/harness.py",
                                   "kernels": "p2v1.cu: p2dec.cu's tuned decode with v1 keys expanded by pous_native.shake256_rows; "
                                              "encode by an addition-chain root on CGBN's generic squaring"},
                "device": facts, "segments": segments, "segments_of_weight_set": total, "blocks": blocks, "weight_bytes": used,
                "setup": {"manifest": P.manifest(), "w_commitment": model.hex(), "salt": salt.hex()},
                **({} if a.audits else H.UNTIMED), "gates": gates, "metrics": metrics, "profile": prof,
                "measurements": [{"name": n, "value": v, "unit": u} for n, v, u in rows_ if v is not None]}

    if not a.audits:
        seqroot = _seqroot(backend, bound, plaintext_rows(bound, c, dropped), dropped, a.reps, None)
        nvme = H.nvme_probe(a.nvme_dir, c_host, nb, a.nvme_samples) if a.nvme_dir else None
        facts["after"] = backend.facts()
        metrics = {**H.UNTIMED, "encode_s": encode_s, "encoded_blocks": encoded, "encode_GBps": encoded * payload / encode_s / 1e9,
                   "encode_stored_GBps": encoded * nb / encode_s / 1e9, "decode_pass_s": decode_s, "ewb": used / decode_s,
                   "stored_GBps": blocks * nb / decode_s / 1e9, "decode_reps_s": decode_reps, "loop": loop, "seqroot": seqroot,
                   "nvme": nvme}
        rows_ = [("encode_GBps", metrics["encode_GBps"], "GB/s"), ("stored_GBps", metrics["stored_GBps"], "GB/s")]
        rows_ += [(f"decode_overhead_T{t}", loop["overhead"][t], "x") for t in tokens]
        rows_ += [("seqroot_root_s", seqroot["root_s"], "s")]
        rows_ += [("nvme_block_read_p99.9_s", nvme["block_read_s"].get("p99.9"), "s")] if nvme else []
        return result(metrics, rows_, None)

    leaves = out / "leaves-p2.bin"
    leaves.write_bytes(b"".join(R.leaf(salt, i // bps, i % bps, c_host[i * nb:(i + 1) * nb]) for i in range(blocks)))
    fields = {"salt": salt.hex(), "model": model.hex(), "manifest": P.manifest(), "payload_bytes": payload}
    native_start = lambda: backend.responder(c, a.responder_cpu)  # noqa: E731
    sync = torch.cuda.synchronize if backend.device != "cpu" and torch.cuda.is_available() else None
    cal = H.calibrate_limit(native_start, fields, LIVE, out, a, leaves, "p2-calibrate")
    limit = int(cal.get("limit_ns") or 0)
    honest = H.audit_round(native_start, fields, LIVE, leaves, out, "p2-honest", a,
                           (lambda: pous_forward(a.load_tokens)) if a.load_tokens else None, limit, sync)

    def control(tag: str, answer: Callable[[int], tuple[bytes, int, int, int]]) -> dict[str, Any]:
        start = lambda: H.ControlResponder(nb, answer, real_delay=a.clock == "wall")  # noqa: E731
        return H.audit_round(start, fields, LIVE, leaves, out, tag, a, None, limit)

    value = control("p2-value", H.control_answers(c_host, nb, zero=set(range(0, blocks, 10))))
    late = control("p2-late", H.control_answers(c_host, nb, late_every=10, late_ns=limit + H.LATE_EXTRA_NS))
    plain_rows = plaintext_rows(bound, c, dropped)
    rec = control("p2-recompute", H.control_answers(c_host, nb, recompute=_recomputer(backend, bound, plain_rows, dropped),
                                                    dropped=set(dropped)))
    recompute = {"dropped_fraction": len(dropped) / blocks, "dropped_blocks": len(dropped),
                 "pattern": f"one block in {RECOMPUTE_EVERY} of all of C, each re-encoded on demand from its decoded row Dec(C_i)",
                 **rec, **H.recompute_margins(rec, set(dropped), limit)}

    seqroot = _seqroot(backend, bound, plain_rows, dropped, a.reps, limit)
    del plain_rows
    nvme = H.nvme_probe(a.nvme_dir, c_host, nb, a.nvme_samples) if a.nvme_dir else None
    facts["after"] = backend.facts()

    k = int(honest.get("k") or 0)
    gates["timed_audit"] = {"passed": honest.get("audits") == honest.get("audits_accepted") and bool(honest.get("audits")),
                            "rc": honest["rc"]}
    gates["value_control"] = {"passed": value.get("audits_accepted") == 0 and value.get("wrong", 0) > 0,
                              "accepted": value.get("audits_accepted"), "wrong": value.get("wrong")}
    gates["late_control"] = {"passed": late.get("audits_accepted") == 0 and late.get("late", 0) > 0 and late.get("wrong", 1) == 0,
                             "accepted": late.get("audits_accepted"), "late": late.get("late"), "wrong": late.get("wrong")}
    gates["recompute_control"] = H.recompute_gate(recompute)
    point = {"k": k == R.K_AUDIT, "rtt_allowance": bool(honest.get("rtt_allowance_within_spec")),
             "deadline": honest.get("deadline_ns") == R.DELTA_NS, "blocks": 1 << 8 <= blocks <= 1 << 23}
    gates["operating_point"] = {"passed": all(point.values()), **point}
    gaps = tuple(f"operating point: {name} is not the frozen value" for name, ok in point.items() if not ok)
    deadline = {"limit_ns": limit, "limit_fixed_before_load": bool(honest.get("limit_fixed_before_load")),
                "allowance_kind": honest.get("allowance_kind"), "rtt_allowance_ns": honest.get("rtt_allowance_ns"),
                "honest_margin_min_s": (honest.get("margin_s") or {}).get("min"),
                "spec_rtt_allowance_max_ns": R.MAX_RTT_ALLOWANCE_NS}
    metrics = {"encode_s": encode_s, "encoded_blocks": encoded, "encode_GBps": encoded * payload / encode_s / 1e9,
               "encode_stored_GBps": encoded * nb / encode_s / 1e9, "decode_pass_s": decode_s, "ewb": used / decode_s,
               "stored_GBps": blocks * nb / decode_s / 1e9, "decode_reps_s": decode_reps, "loop": loop, "deadline": deadline,
               "calibration": cal, "audit": honest, "value_control": value, "late_control": late, "recompute_control": recompute,
               "seqroot": seqroot, "nvme": nvme}
    rows_ = [("encode_GBps", metrics["encode_GBps"], "GB/s"), ("stored_GBps", metrics["stored_GBps"], "GB/s")]
    rows_ += [(f"decode_overhead_T{t}", loop["overhead"][t], "x") for t in tokens]
    rows_ += [(f"audit_answer_{q}_s", (honest.get("elapsed_s") or {}).get(q), "s") for q in ("p50", "p99.9", "max")]
    rows_ += [("audit_margin_min_s", deadline["honest_margin_min_s"], "s"), ("audit_limit_s", limit / 1e9, "s"),
              ("recompute_answer_p99.9_s", (recompute.get("elapsed_s") or {}).get("p99.9"), "s"),
              ("recompute_margin_min_s", recompute["device_margin_s"].get("min"), "s"),
              ("recompute_margin_max_s", recompute["device_margin_s"].get("max"), "s"),
              ("seqroot_root_s", seqroot["root_s"], "s")]
    rows_ += [("nvme_block_read_p99.9_s", nvme["block_read_s"].get("p99.9"), "s")] if nvme else []
    return result(metrics, rows_, profile(k, limit, gaps) if gates["timed_audit"]["passed"] else None)


def _encode_head(backend: Backend, bound: Any, m_dev: Any, encoded: int) -> tuple[float, Any]:
    held: list[Any] = []
    head = m_dev[:encoded].contiguous()
    seconds = backend.timer(lambda: held.append(bound.encode(head, 0)))
    return seconds, held[0]
