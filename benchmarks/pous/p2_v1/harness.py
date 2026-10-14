"""P2's arm of the POUS-only harness (`bench.py --impl p2-v1-gpu | p2-v3-gpu | p2-v3-mc-gpu`): P2 with its GPU kernels (`p2v1.cu`, `gpu.py`)
on one device, the same steps, controls and metrics as the band's arm (`band_gpu/harness.py`), so the two compare side by side.
The kernels take a setup's key prefix, so one arm runs either framing, both on v4's counter-mode SHA-512 keys:

- `V1` (`p2-v1-gpu`): the v4 spec as `reference.py` states it, audited by `memory_challenger.py`'s own loop.
- `V3` (`p2-16448/v4`, `p2-v3-gpu`): `verity`'s registered scheme, whose commitment, 24-byte salt, key prefix and leaves are
  the protocol's (`schemes.p2`, `setup`, `verifier`), audited by `protocol.TimedAudit` through `band_gpu/memory_challenger.py --strict`, which
  refuses any `k` or `Δ + RTT` the certificate does not cover.  Its extra gate (`codec`) decodes segment 0 of the device's `C`
  with the scheme's own codec back to `W`'s, when that segment is wholly encoded.  A joint run's `--bandwidth-salt GPU` sets
  `C` up under the bandwidth audit's salt for that GPU instead (`V3.bind_device`), so the bandwidth verifier audits the same `C`.
- `MC` (`p2-v3-mc-gpu`): `V3`'s blocks audited by the memory challenger (`memory_challenger/live.py`) at `Δ` plus the RTT
  cap, with two more controls: a tampered block (the top bit of its last byte flipped, one block in ten), which only a hash
  of the whole answer catches, and the honest sessions with a signature bit flipped or a receipt time edited after signing.
  Its sessions are signed with S1's LMS key in `--mc-lms-key DIR` (made there when absent), else with the PoUS tests'
  test signer, whose sessions the verdict rejects (its `fail_closed` gate).

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
   caps at 354 µs); `p2_v1/memory_challenger.py` audits at k = 105 against it: the honest audit under POUS forwards, then the value control
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
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from band_gpu import harness as H

from . import reference as R

HERE = Path(__file__).resolve().parent
MEMORY_CHALLENGER = HERE / "memory_challenger.py"
SETUP_SOURCE = hashlib.sha256(b"verity-pous/bench/p2-v1/setup-source").digest()
"""The bench's setup source: in a deployment the verifier's secret or a beacon round drawn after the commitment to `W`."""
PINS = ("P2MeetsM1pB19Uncond", "P2MeetsM1pFamilyUncond", "P2MeetsM1pW8192Uncond", "P2SlackB19Uncond", "P2SlackFamilyUncond",
        "P2ErrorSplitB19Uncond")
ASSUMPTIONS = ("P2-EXP-IO", "SHA-512-CTR/P2-EXP-IO", "P2–M1-SGI (not accepted)", "SeqRoot", "perfect isolation", "cr/sha-512",
               "trusted-nebius-host")
RECOMPUTE_EVERY = 10
"""The recompute control drops one block in this many; P2's blocks are independent, so any pattern keeps what a block needs."""


class V1:
    """The frozen spec's framing (`reference.py`): its commitment to `W`, 32-byte setup salt, key prefix (which frames the
    commitment and that salt) and leaves, audited by `p2_v1/memory_challenger.py`'s own loop (mode 0's challenge derivation)."""

    impl, scheme, live = "p2-v1-gpu", R.PROTOCOL, MEMORY_CHALLENGER
    memory_challenger_args: tuple[str, ...] = ()
    accepted = "audits_accepted"
    """The audit document's count of accepted audits that the gates read."""
    device: bytes | None = None
    """The GPU whose bandwidth salt the store is set up under (`V3.bind_device`), else None: PoUS's own salt."""
    setup_source = SETUP_SOURCE
    """What the setup salt is drawn from, which a bandwidth verifier checks the store's salt against."""

    @property
    def c_of(self) -> str:
        """The arm whose `C` this framing serves, which keys `CCache`."""
        return self.impl

    def __init__(self, P: R.Params, w_host: bytes, research: bool = False) -> None:
        self.P = P
        self.model = R.w_commitment(P, w_host)
        self.salt = R.setup_salt(SETUP_SOURCE, P, self.model)
        self.prefix = R.expand_prefix(self.salt, P, self.model)

    def manifest(self) -> dict[str, Any]:
        return self.P.manifest()

    def keys(self, i: int) -> tuple[int, int]:
        k, t, _ = R.keys_after(self.prefix, self.P, *divmod(i, self.P.blocks_per_segment))
        return k, t

    def leaf(self, i: int, block: bytes) -> bytes:
        return R.leaf(self.salt, *divmod(i, self.P.blocks_per_segment), block)

    def ready(self) -> dict[str, Any]:
        return {"salt": self.salt.hex(), "model": self.model.hex(), "manifest": self.P.manifest(),
                "payload_bytes": self.P.payload_bytes}

    def codec_gate(self, c_host: bytes, w_host: bytes, encoded: int) -> dict[str, Any] | None:
        return None


class V3(V1):
    """`p2-16448/v4` (`verity.core.protocols.pous.schemes.p2`): `protocol.commit`'s commitment to `W` under the
    scheme's manifest, `protocol.draw_salt`'s 24-byte salt, its key prefix (which frames that salt and no commitment) and `vk`'s
    leaves, audited by `protocol.TimedAudit` (`band_gpu/memory_challenger.py --strict`), which refuses any `k` or `Δ + RTT` the
    certificate does not cover.  `research` (the CPU tests' widths, where P2 has no certificate) audits anyway and lists
    the gaps."""

    impl, live = "p2-v3-gpu", HERE.parent / "band_gpu" / "memory_challenger.py"

    def __init__(self, P: R.Params, w_host: bytes, research: bool = False) -> None:
        from verity.core.protocols.pous import protocol, schemes
        from verity.core.protocols.pous.schemes import p2

        self.P, self.p2 = P, p2
        self.params = schemes.P2Params(P.w, P.c0, P.blocks_per_segment, P.segments)
        if self.params.payload_bytes != P.payload_bytes:
            raise ValueError(f"the registered P2 carries {self.params.payload_bytes} payload bytes a block, not {P.payload_bytes}")
        self.scheme_obj = schemes.P2(self.params)
        self.scheme = self.scheme_obj.name
        self.model = protocol.commit(self.scheme_obj, w_host)
        self.salt = protocol.draw_salt(SETUP_SOURCE, self.model)
        self.prefix = p2.expand_prefix(self.params, self.salt)
        self.memory_challenger_args = () if research else ("--strict",)

    def bind_device(self, device: bytes) -> None:
        """Set `C` up under the bandwidth audit's salt for `device` (`bandwidth.audit.draw_salt` over the commitment to `W`
        from the same setup source, the GPU's 16-byte id bound in) in place of `protocol.draw_salt`'s, for a joint run
        (`--bandwidth-salt`): the bandwidth verifier then accepts `C` as this device's store, and P2's audit runs over the
        same `C`, its key prefix and leaves under that salt."""
        from verity.core.protocols.bandwidth import audit

        self.salt = audit.draw_salt(SETUP_SOURCE, self.model, device)
        self.prefix = self.p2.expand_prefix(self.params, self.salt)
        self.device = device

    def manifest(self) -> dict[str, Any]:
        return self.params.manifest()

    def keys(self, i: int) -> tuple[int, int]:
        return self.p2.block_keys(self.params, self.salt, *divmod(i, self.P.blocks_per_segment))

    def leaf(self, i: int, block: bytes) -> bytes:
        from verity.core.protocols.pous.verifier import leaf_hash
        return leaf_hash(self.params, self.salt, i, block)

    def ready(self) -> dict[str, Any]:
        return {"salt": self.salt.hex(), "scheme": self.scheme, "params": self.params.manifest()}

    def codec_gate(self, c_host: bytes, w_host: bytes, encoded: int) -> dict[str, Any] | None:
        """Segment 0 of `C`, as the device encoded it, decoded by the scheme's own codec (`P2.decode_segment`) to segment 0 of `W`:
        decoding is a bijection, so `C`'s first segment is exactly the scheme's encoding of `W`'s.  None (no gate) when segment 0 is
        not wholly encoded."""
        bps, nb, seg = self.P.blocks_per_segment, self.P.width_bytes, self.params.segment_bytes
        if encoded < bps:
            return None
        blocks = [c_host[i * nb:(i + 1) * nb] for i in range(bps)]
        try:
            ok = self.scheme_obj.decode_segment(self.salt, 0, blocks) == w_host[:seg]
        except ValueError as e:
            return {"passed": False, "segment": 0, "reason": str(e)}
        return {"passed": ok, "segment": 0, "codec": self.params.codec.name}


class MC(V3):
    """`p2-16448/v4` audited by the memory challenger (`p2-v3-mc-gpu`, `memory_challenger/live.py`): sessions over a coin
    tree committed before them, each round's whole answer hashed after its receipt time, one signature a session.  Its
    leaves are each block's `sha512/row/v2` digest (`memory_challenger.block_digest`), what the verdict checks a round's
    digest against.  `sign_with` picks the
    signer: S1's LMS key in a directory, whose sessions the verdict accepts, or the PoUS tests' test signer, which
    `memory_challenger.SCHEMES` never accepts (fail closed), so its gates read the verdict under the signer's own mapping
    (`accepted`) and, under the test signer, check the real verdict rejected them all."""

    impl, live = "p2-v3-mc-gpu", HERE.parent / "memory_challenger" / "live.py"
    accepted = "audits_accepted_under_signer"
    c_of = V3.impl
    lms_key = ""

    def sign_with(self, lms_key: str) -> None:
        self.lms_key = lms_key
        self.memory_challenger_args = (*self.memory_challenger_args, *(("--signer", "lms", "--lms-key", lms_key) if lms_key else ("--signer", "test")))

    def leaf(self, i: int, block: bytes) -> bytes:
        from verity.core.protocols.pous.memory_challenger import block_digest
        return block_digest(block)


FRAMINGS = {f.impl: f for f in (V1, V3, MC)}
TAMPER_EVERY = 10
"""The tampered control flips the top bit of the last byte of one block in this many."""


@dataclass
class Backend:
    """What P2's arm computes with: `bind(params, prefix)` gives the kernels of one setup's key prefix (`keys(first,
    count)`, `encode(m_rows, first)`, `decode(c_rows, first)`, `raised()`) on `[n, W/8]` big-endian rows; `responder`,
    `timer`, `graphed`, `copy_bw`, `clock_mhz`, `facts` and `warmup_s` as the band's `Backend`."""

    device: Any
    bind: Callable[[R.Params, bytes], Any]
    responder: Callable[[Any, int | None], Any]
    timer: Callable[[Callable[[], Any]], float]
    graphed: Callable[[Callable[[], Any]], Callable[[], Any]] | None = None
    copy_bw: Callable[[], float] | None = None
    clock_mhz: Callable[[], float] | None = None
    facts: Callable[[], dict[str, Any]] = dict
    warmup_s: float = 0.0


def gpu(lib_dir: str, build_dir: str) -> Backend:
    import torch

    from band_gpu import native

    from .gpu import Kernels

    device = torch.device("cuda", torch.cuda.current_device())
    ext = native.load(build_dir)

    def responder(c: Any, cpu: int | None) -> Any:
        r = ext.Responder(c.data_ptr(), c.shape[0], c.shape[1], device.index, "127.0.0.1", 0, -1 if cpu is None else cpu)
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
    return Backend(device, lambda params, prefix: Kernels(lib_dir, build_dir, params, prefix), responder, timer,
                   graphed, copy_bw, lambda: H._sm_clock_mhz(smi_id), facts, warmup_s=H.PLAIN_WARMUP_S)


def rows(values: list[int], width: int, device: Any) -> Any:
    import torch
    return torch.tensor([list(v.to_bytes(width, "big")) for v in values], dtype=torch.uint8, device=device)


KEYS = Path(__file__).resolve().parents[3] / "verity/core/protocols/pous/tests/fixtures/p2-v4-keys.json"
"""The pinned v4 keys (`p2_v1.vectors` writes them from `vectors.json`), which the vector gate holds the device's keys to."""


def vector_gate(backend: Backend, params: R.Params, vectors: dict[str, Any] | None = None,
                keys: dict[str, Any] | None = None) -> dict[str, Any]:
    """The pinned vector at `params.w`, through the device: its `K` and `t` against `KEYS`, encode and decode against
    `vectors.json`, bit-exact, from a library that reports v4's keys."""
    doc = vectors or json.loads(R.VECTORS.read_text())
    v = next((x for x in doc["vectors"] if int(x["w"]) == params.w and int(x["c0"]) == params.c0), None)
    if v is None:
        return {"passed": False, "reason": f"no pinned vector at w = {params.w}, c0 = {params.c0}"}
    kv = next((x for x in (keys or json.loads(KEYS.read_text()))["vectors"] if x["id"] == v["id"]), None)
    if kv is None or (kv["setup_salt"], kv["model_commitment"]) != (v["setup_salt"], v["model_commitment"]):
        return {"passed": False, "reason": f"no pinned keys for {v['id']}"}
    salt, model = bytes.fromhex(v["setup_salt"]), bytes.fromhex(v["model_commitment"])
    bound = backend.bind(params, R.expand_prefix(salt, params, model))
    first = int(v["segment"]) * params.blocks_per_segment + int(v["block"])
    k, t = bound.keys(first, 1)
    nb = params.width_bytes
    m, c = rows([int(v["m"], 16)], nb, backend.device), rows([int(v["c"], 16)], nb, backend.device)
    version = keys_version(bound)
    got = {"keys_v4": version == 4, "K": k[0].cpu().numpy().tobytes().hex() == kv["K"],
           "t": t[0].cpu().numpy().tobytes().hex() == kv["t"],
           "encode": bytes(bound.encode(m, first)[0].cpu().numpy()) == bytes.fromhex(v["c"]),
           "decode": bytes(bound.decode(c, first)[0].cpu().numpy()) == bytes.fromhex(v["m"]), "flags": bound.raised() == 0}
    return {"passed": all(got.values()), "vector": v["id"], "keys_version": version, **got}


def keys_version(bound: Any) -> int:
    """The key framing a bound library reports expanding: 4 is `reference.keys`; anything else is retired."""
    return getattr(bound, "keys_version", 1)


def picks(encoded: int, blocks: int, extra: int = 8) -> list[int]:
    """The blocks the reference decodes: the first and last encoded ones, the first and last of the rest, and `extra` more."""
    out = {0, encoded - 1, encoded % blocks, blocks - 1}
    for j in range(extra):
        out.add(int.from_bytes(hashlib.sha256(f"verity-pous/bench/p2-v1/pick/{j}".encode()).digest()[:8], "big") % blocks)
    return sorted(i for i in out if 0 <= i < blocks)


def reference_gate(bound: Any, params: R.Params, keys: Callable[[int], tuple[int, int]], c_host: bytes, m_dev: Any,
                   chosen: list[int]) -> dict[str, Any]:
    """`chosen` blocks of `C`, decoded on the device, equal to the reference's decode under the framing's reference keys
    (`keys(i)`)."""
    nb = params.width_bytes
    equal, keys_equal = [], []
    for i in chosen:
        k, t = keys(i)
        want = R.decode(params, int.from_bytes(c_host[i * nb:(i + 1) * nb], "big"), k, t).to_bytes(nb, "big")
        equal.append(bytes(m_dev[i].cpu().numpy()) == want)
        dk, dt = bound.keys(i, 1)
        keys_equal.append(bytes(dk[0].cpu().numpy()) == k.to_bytes(nb, "big") and bytes(dt[0].cpu().numpy()) == t.to_bytes(nb, "big"))
    return {"passed": all(equal) and all(keys_equal), "blocks": chosen, "equal": equal, "keys_equal": keys_equal}


def payload_bytes(m: Any, params: R.Params, used: int) -> Any:
    """`W` from decoded rows: each row's payload bytes, back to back, the first `used` of them."""
    return m[:, params.width_bytes - params.payload_bytes:].reshape(-1)[:used]


def profile(k: int, limit_ns: int, gaps: tuple[str, ...], scheme: str = R.PROTOCOL,
            storage: dict[str, Any] | None = None) -> dict[str, Any]:
    """The run's profile; `storage` is `protocol.TimedAudit`'s `StorageProfile` of the audit of record (the registered scheme), whose
    `certified` it reports alongside the arm's own gaps."""
    return {"scheme": scheme, "k": k, "limit_ns": limit_ns, "rho": "18/19", "error": "1% + 2^-128", "pins": list(PINS),
            "assumptions": list(ASSUMPTIONS), "gaps": list(gaps),
            "certified": bool(storage and storage.get("certified")) and not gaps,
            **({"storage_profile": storage} if storage is not None else {}),
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
            commit: str | None, impl: str) -> dict[str, Any]:
    return {"implementation": {"name": impl, "commit": commit, "harness": "benchmarks/pous/p2_v1/harness.py"},
            "device": facts, "segments": segments, "segments_of_weight_set": total, "blocks": P.blocks, "weight_bytes": used,
            "gates": gates, "metrics": None, "profile": None, "measurements": [], "stopped": H.stopped_reason(gates)}


def run(a: Any, ws: dict[str, Any], out: Path, *, w_segment: Callable[[str, int, int, int], bytes], framing: type[V1] = V1,
        backend: Backend | None = None, params: R.Params | None = None, vectors: dict[str, Any] | None = None,
        research: bool = False, load: Callable[[Any, V1], Callable[[], Any] | None] | None = None) -> dict[str, Any]:
    """One run of P2's arm under `framing` (`V1`, the v4 spec's, or `V3`, `p2-16448/v4`'s).  `backend`, `params` and
    `vectors` replace the GPU kernels, the deployment point and the pinned vectors, and `research` lets `V3`'s memory
    challenger audit where P2 has no certificate (the CPU tests).  `load(c, framing)`, from a caller hosting this harness in
    its own process, is called once `C` is on the device and before the limit is calibrated, and gives the honest audit's load
    in place of `--load-tokens`' POUS forwards (None: no load)."""
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
    fr = framing(P, w_host, research)
    if getattr(a, "bandwidth_salt", ""):
        if not isinstance(fr, V3):
            raise SystemExit(f"--bandwidth-salt sets up a registered P2 store (p2-v3-gpu, p2-v3-mc-gpu), not {fr.impl}'s")
        from verity.core.protocols.bandwidth import audit as bandwidth

        fr.bind_device(bandwidth.read_device(a.bandwidth_salt))
    if isinstance(fr, MC):
        fr.sign_with(getattr(a, "mc_lms_key", "") or "")
    bound = backend.bind(P, fr.prefix)
    pad = nb - payload
    m_dev = torch.zeros(blocks, nb, dtype=torch.uint8, device=backend.device)
    m_dev[:, pad:] = torch.frombuffer(bytearray(w_host), dtype=torch.uint8).to(backend.device).view(blocks, payload)
    encoded = min(a.p2_encode_blocks or blocks, blocks)
    cache = CCache(Path(a.p2_c_cache), fr) if getattr(a, "p2_c_cache", "") and encoded == blocks else None
    hit = cache.load(blocks * nb) if cache is not None else None
    if hit is not None:
        c_host, encode_s = hit
        c = torch.frombuffer(bytearray(c_host), dtype=torch.uint8).to(backend.device).view(blocks, nb)
    else:
        encode_s, head = _encode_head(backend, bound, m_dev, encoded)
        c = torch.empty(blocks, nb, dtype=torch.uint8, device=backend.device)
        c[:encoded] = head
        del head
        if encoded < blocks:
            g = torch.Generator().manual_seed(20260930)
            tail = torch.randint(0, 256, (blocks - encoded, nb), dtype=torch.uint8, generator=g)
            tail[:, 0] &= 0x7F
            c[encoded:] = tail.to(backend.device)
        c_host = c.cpu().numpy().tobytes()
        if cache is not None:
            cache.store(c_host, encode_s)
    gates["round_trip"] = {"passed": bool(torch.equal(bound.decode(c[:encoded].contiguous(), 0), m_dev[:encoded])),
                           "blocks": encoded}
    gates["reference_decode"] = reference_gate(bound, P, fr.keys, c_host, bound.decode(c, 0), picks(encoded, blocks))
    gates["kernel_flags"] = {"passed": bound.raised() == 0}
    codec = fr.codec_gate(c_host, w_host, encoded)
    if codec is not None:
        gates["codec"] = codec
    if not all(g["passed"] for g in gates.values()) or getattr(a, "gates_only", False):
        facts["after"] = backend.facts()
        return _failed(gates, facts, P, segments, total, used, a.impl_commit, fr.impl)

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
        return {"implementation": {"name": fr.impl, "commit": a.impl_commit, "harness": "benchmarks/pous/p2_v1/harness.py",
                                   "kernels": f"p2v1.cu: p2dec.cu's tuned decode with {fr.scheme} keys expanded by p2keys4.cuh (counter-mode "
                                              "SHA-512 from the key prefix's midstate); encode by an addition-chain root on CGBN's generic squaring",
                                   "memory_challenger": str(fr.live.relative_to(HERE.parent)), "memory_challenger_args": list(fr.memory_challenger_args)},
                "device": facts, "segments": segments, "segments_of_weight_set": total, "blocks": blocks, "weight_bytes": used,
                "setup": {"scheme": fr.scheme, "manifest": fr.manifest(), "w_commitment": fr.model.hex(), "salt": fr.salt.hex(),
                          "salt_from": "pous" if fr.device is None else "bandwidth",
                          "device": None if fr.device is None else fr.device.hex(), "keys_version": keys_version(bound)},
                **({} if a.audits else H.UNTIMED), "gates": gates, "metrics": metrics, "profile": prof,
                "measurements": [{"name": n, "value": v, "unit": u} for n, v, u in rows_ if v is not None]}

    if not a.audits:
        seqroot = _seqroot(backend, bound, plaintext_rows(bound, c, dropped), dropped, a.reps, None)
        nvme = H.nvme_probe(a.nvme_dir, c_host, nb, a.nvme_samples) if a.nvme_dir else None
        facts["after"] = backend.facts()
        metrics = {**H.UNTIMED, "encode_s": encode_s, "encode_cache": cache.record if cache else None,
                   "encoded_blocks": encoded, "encode_GBps": encoded * payload / encode_s / 1e9,
                   "encode_stored_GBps": encoded * nb / encode_s / 1e9, "decode_pass_s": decode_s, "ewb": used / decode_s,
                   "stored_GBps": blocks * nb / decode_s / 1e9, "decode_reps_s": decode_reps, "loop": loop, "seqroot": seqroot,
                   "nvme": nvme}
        rows_ = [("encode_GBps", metrics["encode_GBps"], "GB/s"), ("stored_GBps", metrics["stored_GBps"], "GB/s")]
        rows_ += [(f"decode_overhead_T{t}", loop["overhead"][t], "x") for t in tokens]
        rows_ += [("seqroot_root_s", seqroot["root_s"], "s")]
        rows_ += [("nvme_block_read_p99.9_s", nvme["block_read_s"].get("p99.9"), "s")] if nvme else []
        return result(metrics, rows_, None)

    leaves = out / "leaves-p2.bin"
    leaves.write_bytes(b"".join(fr.leaf(i, c_host[i * nb:(i + 1) * nb]) for i in range(blocks)))
    fields, live, extra = fr.ready(), fr.live, list(fr.memory_challenger_args)
    H.leave_reserved(a)
    native_start = lambda: backend.responder(c, a.responder_cpu)  # noqa: E731
    sync = torch.cuda.synchronize if backend.device != "cpu" and torch.cuda.is_available() else None
    honest_load = load(c, fr) if load is not None else (lambda: pous_forward(a.load_tokens)) if a.load_tokens else None
    cal = H.calibrate_limit(native_start, fields, live, out, a, leaves, "p2-calibrate")
    limit = int(cal.get("limit_ns") or 0)
    honest = H.audit_round(native_start, fields, live, leaves, out, "p2-honest", a, honest_load, limit, sync, extra)

    def control(tag: str, answer: Callable[[int], tuple[bytes, int, int, int]]) -> dict[str, Any]:
        start = lambda: H.ControlResponder(nb, answer, real_delay=a.clock == "wall")  # noqa: E731
        return H.audit_round(start, fields, live, leaves, out, tag, a, None, limit, None, extra)

    value = control("p2-value", H.control_answers(c_host, nb, zero=set(range(0, blocks, 10))))
    late = control("p2-late", H.control_answers(c_host, nb, late_every=10, late_ns=limit + H.LATE_EXTRA_NS))
    plain_rows = plaintext_rows(bound, c, dropped)
    rec = control("p2-recompute", H.control_answers(c_host, nb, recompute=_recomputer(backend, bound, plain_rows, dropped),
                                                    dropped=set(dropped)))
    recompute = {"dropped_fraction": len(dropped) / blocks, "dropped_blocks": len(dropped),
                 "pattern": f"one block in {RECOMPUTE_EVERY} of all of C, each re-encoded on demand from its decoded row Dec(C_i)",
                 **rec, **H.recompute_margins(rec, set(dropped), limit)}
    mc = isinstance(fr, MC)
    if mc:
        from memory_challenger.live import signature_control

        tampered = control("p2-tampered", H.control_answers(c_host, nb, tamper=set(range(0, blocks, TAMPER_EVERY))))
        signature = signature_control(honest, lambda i: fr.leaf(i, c_host[i * nb:(i + 1) * nb]))

    seqroot = _seqroot(backend, bound, plain_rows, dropped, a.reps, limit)
    del plain_rows
    nvme = H.nvme_probe(a.nvme_dir, c_host, nb, a.nvme_samples) if a.nvme_dir else None
    facts["after"] = backend.facts()

    k = int(honest.get("k") or 0)
    acc = fr.accepted
    gates["timed_audit"] = {"passed": honest.get("audits") == honest.get(acc) and bool(honest.get("audits")),
                            "rc": honest["rc"], "accepted_by": acc}
    gates["value_control"] = {"passed": value.get(acc) == 0 and value.get("wrong", 0) > 0,
                              "accepted": value.get(acc), "wrong": value.get("wrong")}
    gates["late_control"] = {"passed": late.get(acc) == 0 and late.get("late", 0) > 0 and late.get("wrong", 1) == 0,
                             "accepted": late.get(acc), "late": late.get("late"), "wrong": late.get("wrong")}
    gates["recompute_control"] = H.recompute_gate(recompute)
    if mc:
        gates["tampered_control"] = {"passed": tampered.get(acc) == 0 and tampered.get("wrong", 0) > 0,
                                     "accepted": tampered.get(acc), "wrong": tampered.get("wrong")}
        gates["signature_control"] = signature
        rounds = (honest, value, late, rec, tampered)
        if fr.lms_key:
            gates["signer"] = {"passed": all((d.get("signer") or {}).get("test") is False
                                             and d.get("audits_accepted") == d.get(acc) for d in rounds),
                               "schemes": sorted({(d.get("signer") or {}).get("scheme") for d in rounds}, key=str),
                               "note": "S1's LMS at lms.DEPLOYED: the verdict is the one under memory_challenger.SCHEMES"}
        else:
            gates["fail_closed"] = {"passed": all(d.get("audits_accepted") == 0 for d in rounds),
                                    "accepted": [d.get("audits_accepted") for d in rounds],
                                    "note": "memory_challenger.SCHEMES accepts no test-signed session: no session is "
                                            "accepted, honest or not, and no storage profile is issued"}
    allowance = honest.get("rtt_allowance_ns")
    point = {"k": k == R.K_AUDIT, "rtt_allowance": allowance is not None and allowance <= R.MAX_RTT_ALLOWANCE_NS,
             "deadline": honest.get("deadline_ns") == R.DELTA_NS, "blocks": 1 << 8 <= blocks <= 1 << 23}
    gates["operating_point"] = {"passed": all(point.values()), **point}
    gaps = tuple(f"operating point: {name} is not the frozen value" for name, ok in point.items() if not ok)
    if "--strict" in fr.memory_challenger_args:
        gates["certificate"] = {"passed": not honest.get("refused") and bool(honest.get("audits")),
                                "refused": honest.get("refused"), "memory_challenger_gaps": honest.get("gaps"),
                                "note": "protocol.TimedAudit refuses any k or Δ + RTT the certificate does not cover; the "
                                        "gaps it lists (such as W's segment count) are reported, not refused"}
        gaps += tuple(honest.get("gaps") or ())
    deadline = {"limit_ns": limit, "limit_fixed_before_load": bool(honest.get("limit_fixed_before_load")),
                "allowance_kind": honest.get("allowance_kind"), "rtt_allowance_ns": honest.get("rtt_allowance_ns"),
                "honest_margin_min_s": (honest.get("margin_s") or {}).get("min"),
                "spec_rtt_allowance_max_ns": R.MAX_RTT_ALLOWANCE_NS}
    metrics = {"encode_s": encode_s, "encode_cache": cache.record if cache else None, "encoded_blocks": encoded,
               "encode_GBps": encoded * payload / encode_s / 1e9,
               "encode_stored_GBps": encoded * nb / encode_s / 1e9, "decode_pass_s": decode_s, "ewb": used / decode_s,
               "stored_GBps": blocks * nb / decode_s / 1e9, "decode_reps_s": decode_reps, "loop": loop, "deadline": deadline,
               "calibration": cal, "audit": honest, "value_control": value, "late_control": late, "recompute_control": recompute,
               "seqroot": seqroot, "nvme": nvme,
               **({"tampered_control": tampered, "signature_control": signature} if mc else {})}
    rows_ = [("encode_GBps", metrics["encode_GBps"], "GB/s"), ("stored_GBps", metrics["stored_GBps"], "GB/s")]
    rows_ += [(f"decode_overhead_T{t}", loop["overhead"][t], "x") for t in tokens]
    rows_ += [(f"audit_answer_{q}_s", (honest.get("elapsed_s") or {}).get(q), "s") for q in ("p50", "p99.9", "max")]
    rows_ += [("audit_margin_min_s", deadline["honest_margin_min_s"], "s"), ("audit_limit_s", limit / 1e9, "s"),
              ("recompute_answer_p99.9_s", (recompute.get("elapsed_s") or {}).get("p99.9"), "s"),
              ("recompute_margin_min_s", recompute["device_margin_s"].get("min"), "s"),
              ("recompute_margin_max_s", recompute["device_margin_s"].get("max"), "s"),
              ("seqroot_root_s", seqroot["root_s"], "s")]
    rows_ += [("nvme_block_read_p99.9_s", nvme["block_read_s"].get("p99.9"), "s")] if nvme else []
    issued = gates["timed_audit"]["passed"] and honest.get("audits_accepted") == honest.get("audits")
    return result(metrics, rows_, profile(k, limit, gaps, fr.scheme, honest.get("profile")) if issued else None)


class CCache:
    """`C = Enc(W)` kept on disk between runs (`--p2-c-cache DIR`), so a timed run of a large `W` skips the encode, which
    an untimed run of the same setup measured (P2's roots take minutes a GB).  Keyed by the framing, its manifest, key
    prefix and commitment to `W`; a cached `C` must match the sha256 it was stored with, and goes through every gate a
    fresh one does (the round trip decodes all of it back to `W`).  `record` says which run encoded it and in what time."""

    def __init__(self, root: Path, fr: V1) -> None:
        key = {"impl": fr.c_of, "manifest": fr.manifest(), "prefix": fr.prefix.hex(), "model": fr.model.hex()}
        self.key = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        self.data, self.meta = root / f"{self.key}.c", root / f"{self.key}.json"
        self.record: dict[str, Any] = {"key": self.key, "hit": False}

    def load(self, size: int) -> tuple[bytes, float] | None:
        if not (self.data.is_file() and self.meta.is_file()):
            return None
        meta = json.loads(self.meta.read_text())
        data = self.data.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != meta["sha256"]:
            return None
        self.record = {**meta, "key": self.key, "hit": True}
        return data, float(meta["encode_s"])

    def store(self, data: bytes, encode_s: float) -> None:
        self.data.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.data.with_suffix(f".tmp{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, self.data)
        run = os.environ.get("RESEARCH_RUN_DIR", "")
        meta = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "encode_s": encode_s,
                "encoded_by_run": Path(run).name if run else None}
        self.meta.write_text(json.dumps(meta, indent=1) + "\n")
        self.record = {**meta, "key": self.key, "hit": False}


def _encode_head(backend: Backend, bound: Any, m_dev: Any, encoded: int) -> tuple[float, Any]:
    held: list[Any] = []
    head = m_dev[:encoded].contiguous()
    seconds = backend.timer(lambda: held.append(bound.encode(head, 0)))
    return seconds, held[0]
