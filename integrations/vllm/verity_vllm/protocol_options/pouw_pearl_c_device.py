"""Pearl-C on the RTX PRO 6000 (sm_120) at the linears' ``quant_method.apply``: every call computed and committed on the device
by GPU 1's kernel pipeline, one epoch per install over every linear weight, one transcript per pass over every call.

The kernel is a directory (setting ``kernel``): the run tree's ``benchmarks/pouw/pearl_c_sm120``, whose ``run.py`` has the
`Pipeline` (its kernels in order, each a named step of a phase, over buffers the caller may own) and whose ``fixture.py`` has
the keys (`keys_of`, `unit_block`).  The cubin is build.sh's (setting ``cubin``, default the directory's
``pearl_c_sm120.cubin``).  This module owns the buffers and runs the pipeline's steps on torch's current stream, in the
device's primary context:

* **The weight side**, once per weight and epoch (`_Weight`).  B's FP32 words (the BF16 weight widened, exact) are hashed
  into B's frame-b3 row tree (phase ``epoch``), whose root must be the committed one.  Then, with the salt's seed_B, come E_B's
  and the F lines, B's statistics and forming, B̃·F_A, G and P_B (phase ``job``).  B's FP32 copy is dropped after, so the
  weight side keeps about a byte per weight (B̃) beside the model's own weight.
* **Each call** (`gemm`):
  - x's rows are widened into A's FP32 words, with zero filler rows up to m₁₆ (the domain's 16 | m; filler is committed as
    zero rows) and zero padding up to the kernels' tile;
  - ``commit``: A's pads and level keys under its domain, A's row tree, seed_A and E_A's line key;
  - ``forming``: E_A's lines, A's statistics and forming;
  - ``cleanup``: A′·F_B, P_A, and the chain with its peel and y (``gemm_pearlc`` writes y into the tensor this returns);
  - ``hashing``: the message digests and the tile leaves, into the pass's leaf buffer at the call's offset.  No per-call
    tile tree is built, since the pass has one.
* **The records** (`pearl_c_work.UnitRecord`): each call's real rows and width, and the rows that fail the per-row rules
  (`pearl_c_work.row_passes`).  The noise floor comes from the kernel's own ρ and α, whose product is exact as a double.
  The liveness is screened in FP32 with a 2⁻¹⁰ band (`screen`), and a row inside the band is decided exactly by
  `pearl_c.live_row` from its retained words.
* **Each pass** (`begin`, `end`): the transcript is every call's A root and one tile tree over every leaf, in call order,
  under `audit.transcript_domain` (the audit `Prover`'s layout).  With a retain directory, the pass also writes what the
  prover opens from: A's rows (BF16, exact) and tree levels per call, B's rows and tree levels per weight, the leaves, the
  tile tree and the manifest (`write_manifest`) that the reference verifier reads
  (``benchmarks/pouw/pearl_c_vllm/verify_run.py``).  A retained pass starts from poisoned outputs (`poison`): a replay of an
  earlier pass's calls finds that pass's bytes still in the buffers, and they would be right for it, so a skipped write
  must not be able to pass as a repeat.  `omit` drops named steps, for the negative control.

The per-call host work is kept small (the decode profile, research store ``internal/pouw/rtx-pro/pouw-mvp-decode-profile.md``):
the keys by native BLAKE3 (`_Keys`, byte for byte `call_keys`), their upload through a pinned ring (`_Uploads`), the records'
screen as one fused kernel (`screen_into`, Triton; `screen` where Triton can't run), A's roots copied into a pass buffer,
and the stream read from torch's raw accessor.

With ``defer`` (the decode plan's step 2, bc-b139c29c's schedule) each call's tile hashing (`SIDE_STEPS`: the message
digests and the tile leaves) runs on a low-priority side stream behind an event after the GEMM, while the next calls go on:
A's commitment stays on the call's stream, since forming reads seed_A.  Each shape then rotates `DEFER_SETS` sets of C̃, U
and the digests, and a call reuses a set only once the side stream has finished with it; the pass's end waits for the side
stream.  The same kernels read and write the same bytes, so the commitments are the serial schedule's.

With ``hashing`` off (the bench's ``pearlc-nohash``) the call skips A's tree, E_A's lines, the digests, the leaves and the
records.  What is left is the same forming and GEMM without hashing, for the time split.  ``hashing`` may change between
passes, never within one.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.commitments import RangeIndexedDomain
from verity.commitments.merkle import _FRAME, _uint
from verity_pouw.audit import Epoch, Workload, activation_binding, rows_domain, transcript_domain
from verity_pouw.protocol import Matmul, Shape
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_c_work as PW
from verity_pouw.schemes import pearl_kw as P
from verity_pouw.serving import SchemeId, hash_format

#: the Pearl-C schemes this device path runs, by name (`verity_pouw.serving.SchemeId`: forming, device record, hashing format).
#: The format is a switch: another joins as one name here once GPU 1's pipeline commits with it (which is Daniel's call); its
#: keys and tree hash come from `verity_pouw.serving.HASH_FORMATS`, the one table, through the scheme.
SCHEMES = ("pearl-c-sm120-v1-h1",)
#: the domain's row granularity: m₁₆ = m rounded up to it, the filler rows zero
ROW_ALIGN = 16
#: tiles one pass's leaf buffer holds (256 MiB)
LEAF_CAPACITY = 1 << 23
#: the liveness screen's relative band, far wider than its FP32 error (2⁻²³)
BAND = 2.0 ** -10
#: calls one pass's A-root buffer holds, and rows its screen-flag buffer holds
CALL_CAPACITY = 1 << 16
FLAG_ROWS = 1 << 22
#: the pinned ring's key slots (each call's dom_a ‖ unit, 136 bytes)
UPLOAD_SLOTS = 1024
MANIFEST_SCHEMA = "verity-vllm/pouw-pearl-c-run/v1"
WEIGHT_BUFFERS = ("eb_codes", "fa_codes", "faT32", "fb_codes", "fbT32", "fbT128", "g", "alpha_b", "beta_b", "ia_b", "aux_b",
                  "epb_codes", "bt", "bf", "pb_t", "tree_b", "keys_b")
CALL_BUFFERS = ("alpha_a", "beta_a", "ia_a", "aux_a", "epa_codes", "ea_codes", "at", "af", "pa_t", "c", "u", "digests",
                "tree_a", "keys_a", "line_a", "seed_a")
EPOCH_STEPS = ("pad_b", "node_keys_b", "hash_rows_b", "tree_b")
JOB_STEPS = ("lines_eb", "lines_fa", "lines_fb", "stats_b", "eprime_b", "form_b", "bf", "g", "pb")
CALL_STEPS = ("pad_a", "node_keys_a", "hash_rows_a", "tree_a", "seed_a", "lines_ea", "stats_a", "eprime_a", "form_a", "af", "pa",
              "gemm_pearlc", "hash_msg", "hash_leaf")
HASH_STEPS = frozenset({"pad_a", "node_keys_a", "hash_rows_a", "tree_a", "seed_a", "lines_ea", "hash_msg", "hash_leaf"})
#: the deferred schedule's side-stream steps, and the buffers they read that each shape rotates
SIDE_STEPS = ("hash_msg", "hash_leaf")
ROTATED = ("c", "u", "digests")
DEFER_SETS = 2


def scheme_of(name: str, **instance: Any) -> C.PearlC:
    """The scheme `name` names (`SCHEMES`); `instance` (``k_min``, ``tile``) makes a test instance."""
    from verity_pouw.schemes.pearl_c_device import DEVICES
    if name not in SCHEMES:
        raise KeyError(f"{name!r}: this device path runs {', '.join(SCHEMES)}")
    sid = SchemeId.parse(name)
    s = C.PearlC(forming=sid.forming, device=DEVICES[sid.device], hashing=sid.hashing, **instance)
    if s.name != name:
        raise ValueError(f"the scheme built for {name!r} names itself {s.name!r}")
    return s


def tree_hash(name: str) -> str:
    """The scheme's tree hash (`CommitmentDomain.hash` of A's, B's and the transcript's trees), from its hashing format."""
    return scheme_of(name).commitment_hash


def _format_keys(scheme: C.PearlC) -> bytes:
    """The message and leaf keys of the scheme's hashing format, as the kernels read them (``k_msg`` ‖ ``k_leaf``)."""
    fmt = hash_format(scheme)
    if fmt.message_key() is None:
        raise ValueError(f"{scheme.name}: hashing {fmt.name} has no keyed-BLAKE3 message and leaf, which the kernels hash with")
    return fmt.message_key() + fmt.leaf_key()


def refusals(name: str, shapes: dict[int, tuple[int, int]]) -> list[str]:
    """Every linear outside the scheme's domain or the kernels' (k a multiple of 1,024, n of 128), by weight id."""
    s, out = scheme_of(name), []
    for w, (n, k) in sorted(shapes.items()):
        if k % 1024 or n % 128:
            out.append(f"linear {w}: (n, k) = ({n}, {k}), and the kernels take k a multiple of 1,024 and n of 128")
            continue
        try:
            s.check(Matmul(0, Shape(ROW_ALIGN, k, n), w))
        except Exception as e:  # noqa: BLE001 - DomainError, named per layer
            out.append(f"linear {w}: {e}")
    return out


def _load(name: str, path: Path) -> tuple[Any, str]:
    """`path` as a module and its bytes' SHA-256.  The module is named by the digest, so two kernel directories in one process
    never share a module, and the same bytes loaded twice are one module."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    key = f"{name}_{digest[:16]}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        mod = sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key], digest


def source_digest(run_sha256: str, fixture_sha256: str) -> str:
    """The kernel sources' digest: `run.py`'s and `fixture.py`'s, in that order."""
    return hashlib.sha256(bytes.fromhex(run_sha256) + bytes.fromhex(fixture_sha256)).hexdigest()


@dataclass
class Kernel:
    """GPU 1's pipeline, loaded once per process per (sources, cubin): ``run.py`` (as `run`), ``fixture.py`` (as `fixture`), the
    device (`dev`), and the digests that name what ran (the manifest records both)."""

    run: Any
    fixture: Any
    dev: Any
    cubin_sha256: str
    source_sha256: str | None = None

    @classmethod
    def load(cls, directory: str, cubin: str | None = None) -> Kernel:
        d = Path(directory)
        run, run_sha = _load("pearl_c_sm120_run", d / "run.py")
        fixture, fixture_sha = _load("pearl_c_sm120_fixture", d / "fixture.py")
        raw = Path(cubin or d / "pearl_c_sm120.cubin").read_bytes()
        key = (source_digest(run_sha, fixture_sha), hashlib.sha256(raw).hexdigest())
        if key not in _KERNELS:
            dev = run.Dev(raw)
            dev.open()
            _KERNELS[key] = cls(run, fixture, dev, key[1], key[0])
        return _KERNELS[key]


#: (source_sha256, cubin_sha256) -> the loaded kernel
_KERNELS: dict[tuple[str, str], Kernel] = {}


def _zeros(n: int, device: Any) -> Any:
    import torch
    return torch.zeros(max(n, 16), dtype=torch.uint8, device=device)


def _put(t: Any, data: bytes) -> None:
    """`data` into device bytes `t`, on the current stream (the pinned staging copy is torch's to keep alive)."""
    import torch
    host = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    t[:len(data)].copy_(host.pin_memory() if t.is_cuda else host, non_blocking=t.is_cuda)


class _TorchStream:
    """torch's Stream object for the current stream, looked up again only when the raw handle changes: `Event.record()`
    without a stream calls `torch.cuda.current_stream`, about 15 µs."""

    def __init__(self):
        self.raw, self.stream = None, None

    def __call__(self, raw: int) -> Any:
        if raw != self.raw:
            import torch
            self.raw, self.stream = raw, torch.cuda.current_stream()
        return self.stream


def _stream(device: Any) -> int:
    """The current stream's handle: torch's raw accessor, not `torch.cuda.current_stream`, which costs about 15 µs a call."""
    import torch
    if device.type != "cuda":
        return 0
    return torch._C._cuda_getCurrentRawStream(device.index if device.index is not None else torch.cuda.current_device())


def _bytes(t: Any) -> bytes:
    return t.cpu().numpy().tobytes()


def screen(x: Any, alpha: Any, aux: Any, chunk: int = 1 << 26) -> tuple[Any, Any]:
    """The per-row rules on FP32 rows x (rows x k, on the device) with their `stats_s5` α and (s, ρ): (fails, unsure), two
    bool vectors.  A row fails on the noise floor (ρ·α < σ_min, exact as a double product) or when its dead count, screened
    in FP32 against 64·Σ_{p ≡ 0 (8)} x_p² / ⌈k/8⌉ with a relative band of 2⁻¹⁰, is surely over k/64; `unsure` rows are those
    the band leaves open, for `pearl_c.live_row` to decide exactly."""
    import torch
    rows, k = x.shape
    rho = aux.view(torch.float32)[:2 * rows].view(rows, 2)[:, 1].double()
    floor = rho * alpha.view(torch.float32)[:rows].double() < P.SIGMA_MIN
    k8 = len(range(0, k, C.S5_STRIDE))
    fails, unsure = torch.empty_like(floor), torch.empty_like(floor)
    step = max(1, chunk // k)
    for r0 in range(0, rows, step):
        sq = x[r0:r0 + step].square()
        thr = (64.0 * sq[:, ::C.S5_STRIDE].double().sum(1) / k8)[:, None]
        hi, lo = (thr * (1 + BAND)).float(), (thr * (1 - BAND)).float()
        dead = (sq > hi).sum(1)
        band = ((sq >= lo) & (sq <= hi)).sum(1)
        f = floor[r0:r0 + step]
        sure_fail, sure_pass = 64 * dead > k, 64 * (dead + band) <= k
        fails[r0:r0 + step] = f | sure_fail
        unsure[r0:r0 + step] = ~f & ~sure_fail & ~sure_pass
    return fails, unsure


_SCREEN_KERNEL: list = []


def _screen_kernel():
    """`screen` as one Triton program a row: FP64 sum of the squares at positions ≡ 0 (mod 8), the band's bounds rounded to
    FP32 as `screen` rounds them, the dead and in-band counts, and the noise floor from the row's α and ρ.  Its sum's order is
    not torch's, so a row can come out unsure in one and decided in the other; both leave only rows within the band undecided,
    and `excluded` decides those exactly, so the records are the same."""
    if not _SCREEN_KERNEL:
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(x, alpha, aux, out, k, k8, sigma, band, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            base = x + row.to(tl.int64) * k
            acc = tl.zeros([BLOCK], dtype=tl.float64)
            for s in range(0, k, BLOCK):
                offs = s + tl.arange(0, BLOCK)
                v = tl.load(base + offs, mask=offs < k, other=0.0)
                acc += tl.where(offs % 8 == 0, (v * v).to(tl.float64), 0.0)
            thr = 64.0 * tl.sum(acc, axis=0) / k8
            hi = (thr * (1.0 + band)).to(tl.float32)
            lo = (thr * (1.0 - band)).to(tl.float32)
            dead = tl.zeros([BLOCK], dtype=tl.int32)
            near = tl.zeros([BLOCK], dtype=tl.int32)
            for s in range(0, k, BLOCK):
                offs = s + tl.arange(0, BLOCK)
                live = offs < k
                v = tl.load(base + offs, mask=live, other=0.0)
                sq = v * v
                dead += tl.where(live & (sq > hi), 1, 0)
                near += tl.where(live & (sq >= lo) & (sq <= hi), 1, 0)
            d = tl.sum(dead, axis=0)
            b = tl.sum(near, axis=0)
            floor = tl.where(tl.load(aux + 2 * row + 1).to(tl.float64) * tl.load(alpha + row).to(tl.float64) < sigma, 1, 0)
            sure_fail = tl.where(64 * d > k, 1, 0)
            sure_pass = tl.where(64 * (d + b) <= k, 1, 0)
            tl.store(out + 2 * row, tl.maximum(floor, sure_fail).to(tl.uint8))
            tl.store(out + 2 * row + 1, ((1 - floor) * (1 - sure_fail) * (1 - sure_pass)).to(tl.uint8))
        _SCREEN_KERNEL.append(kernel)
    return _SCREEN_KERNEL[0]


def fused_screen_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def screen_into(out: Any, x: Any, alpha: Any, aux: Any, fused: bool) -> None:
    """`screen`'s two flags per row of x (fails, unsure) into `out` (uint8, rows x 2): one fused kernel with `fused`, else
    `screen`'s torch operations."""
    import torch
    rows, k = x.shape
    if fused:
        _screen_kernel()[(rows,)](x, alpha.view(torch.float32), aux.view(torch.float32), out, k, len(range(0, k, C.S5_STRIDE)),
                                  P.SIGMA_MIN, BAND, BLOCK=1024)
    else:
        fails, unsure = screen(x, alpha, aux)
        out[:, 0].copy_(fails)
        out[:, 1].copy_(unsure)


def excluded(fails: Any, unsure: Any, x: Any) -> tuple[int, ...]:
    """The rows of x (FP32, on the device) that fail the rules: `screen`'s fails, and its unsure rows that
    `pearl_c.live_row` rejects on their words."""
    import torch
    rows = unsure.nonzero().flatten()
    out = {int(i) for i in fails.nonzero().flatten().tolist()}
    if len(rows):
        for r, row in zip(rows.tolist(), x[rows].view(torch.int32).cpu().numpy().astype("<u4")):
            if not C.live_row([int(v) for v in row]):
                out.add(int(r))
    return tuple(sorted(out))


@dataclass
class _Weight:
    n: int
    k: int
    b: dict[str, Any]
    keys: Any                      # line_b ‖ dom_b
    root: bytes
    excluded: tuple[int, ...]
    tensor: Any                    # the layer's own weight, for the retained transcript


@dataclass
class _Shape:
    m16: int
    mp: int
    n: int
    k: int
    b: dict[str, Any]
    xa: Any                        # FP32 (mp x k): rows m16 … mp − 1 stay zero
    keys: Any                      # dom_a ‖ unit
    sets: list = field(default_factory=list)   # deferred: DEFER_SETS sets of C̃, U and the digests, set 0 b's own
    done: list = field(default_factory=list)   # deferred: each set's last side-stream use (a CUDA event)
    turn: int = 0


@dataclass
class _Pass:
    retain: Path | None
    run: str | None = None          # the run id a retained pass's manifest names
    matmuls: list[Matmul] = field(default_factory=list)
    flag_rows: int = 0               # rows of the pass's screen flags so far
    records: list[PW.UnitRecord] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    tiles: int = 0
    poison: int | None = None       # the byte every call buffer starts the pass as, those created during it included


def weight_roots(kernel: Kernel, weights: dict[int, Any], hash: str = "blake3") -> dict[int, bytes]:
    """Each weight's B-row tree root (phase ``epoch`` alone), before any salt: the composition's weight phase.  `hash` is
    the scheme's tree hash (`tree_hash`)."""
    import torch
    R, out = kernel.run, {}
    for w, weight in weights.items():
        n, k = weight.shape
        size = R.Pipeline.sizes(R.BM, n, k, R.BM)
        b = {name: _zeros(size[name], weight.device) for name in ("tree_b", "keys_b", "faT32", "fbT32", "fbT128")}
        keys = _zeros(64, weight.device)
        _put(keys[32:], rows_domain(_weight_binding(w), -2, n, hash).domain_id)
        xb = weight.to(torch.float32).contiguous()
        pipe = _pipeline(kernel, R.BM, n, k, {**{x: t.data_ptr() for x, t in b.items()}, "line_a": 0}, xb.data_ptr(),
                         {"line_b": keys.data_ptr(), "dom_b": keys.data_ptr() + 32})
        _run(kernel, pipe, EPOCH_STEPS, _stream(weight.device))
        out[w] = _bytes(b["tree_b"][size["tree_b"] - 32:size["tree_b"]])
    return out


def call_keys(fixture: Any, salt: bytes, root_b: bytes, index: int, m16: int, hash: str = "blake3") -> bytes:
    """A call's dom_a ‖ unit, as `fixture.keys_of` derives them for matmul `index` of m₁₆ rows, under tree hash `hash`."""
    return rows_domain(activation_binding(salt, index), index, m16, hash).domain_id + fixture.unit_block(salt, root_b, index)


class _Keys:
    """`call_keys` by native BLAKE3 (the `blake3` package): dom_a is `CommitmentDomain.domain_id`'s framed message hashed
    natively, and unit's label hash is a constant.  Under another tree hash than ``blake3`` (another format's framing), or
    without the package, `call_keys` itself."""

    def __init__(self, fixture: Any, salt: bytes, hash: str = "blake3"):
        try:
            import blake3 as native
        except ImportError:
            native = None
        self.native, self.fixture, self.salt, self.hash = native if hash == "blake3" else None, fixture, salt, hash
        self.label = fixture.unit_block(salt, bytes(32), 0)[-32:]
        self.positions: dict[int, bytes] = {}

    def __call__(self, root_b: bytes, index: int, m16: int) -> bytes:
        if self.native is None:
            return call_keys(self.fixture, self.salt, root_b, index, m16, self.hash)
        pos = self.positions.get(m16)
        if pos is None:
            pos = self.positions[m16] = bytes.fromhex(RangeIndexedDomain(m16).identity_digest)
        parts = (activation_binding(self.salt, index), _uint(index + 2), pos, _uint(m16))
        msg = _FRAME + (6).to_bytes(4, "big") + b"domain" + b"".join(len(p).to_bytes(8, "big") + p for p in parts)
        return self.native.blake3(msg).digest() + self.salt + root_b + index.to_bytes(8, "little") + self.label


class _Side:
    """The deferred schedule's side stream, at the least priority, and its handle (a stand-in handle off the GPU, for the
    recording device's log).  vLLM's stream keeps its own priority, which is the default and so also the least: the side
    work fills what the call's stream leaves free."""

    def __init__(self, device: Any):
        import torch
        if device.type == "cuda":
            self.stream = torch.cuda.Stream(device=device, priority=0)
            self.raw = self.stream.cuda_stream
        else:
            self.stream, self.raw = None, 1


class _Uploads:
    """Small host-to-device copies through a pinned ring allocated once: each is one copy on the current stream, and a slot is
    written again only once its last copy is done (its event)."""

    def __init__(self, device: Any, slots: int = UPLOAD_SLOTS, width: int = 144):
        import torch
        self.cuda, self.slots, self.i = device.type == "cuda", slots, 0
        self.host = torch.zeros((slots, width), dtype=torch.uint8, pin_memory=self.cuda)
        self.view = self.host.numpy()
        self.events: list[Any] = [None] * slots

    def put(self, dst: Any, data: bytes, stream: Any = None) -> None:
        """`data` into `dst` on `stream` (torch's current stream when None)."""
        import numpy as np
        import torch
        i, n = self.i, len(data)
        self.i = (i + 1) % self.slots
        if self.events[i] is not None:
            self.events[i].synchronize()
        self.view[i, :n] = np.frombuffer(data, np.uint8)
        dst[:n].copy_(self.host[i, :n], non_blocking=self.cuda)
        if self.cuda:
            if self.events[i] is None:
                self.events[i] = torch.cuda.Event()
            self.events[i].record(stream)


def _weight_binding(w: int) -> bytes:
    from verity_pouw.audit import weight_binding
    return weight_binding(w)


def _pipeline(kernel: Kernel, m: int, n: int, k: int, bufs: dict[str, int], xb: int, keys: dict[str, int], xa: int = 0,
              m_real: int | None = None) -> Any:
    full = {"dom_a": 0, "dom_b": 0, "dom_t": 0, "line_b": 0, "k_msg": 0, "k_leaf": 0, "unit": 0, **keys}
    return kernel.run.Pipeline(kernel.dev, m, n, k, "sm120", full, xa, xb, bufs=bufs, m_real=m_real)


def _steps(pipe: Any, names: tuple[str, ...]) -> list[Any]:
    """The named steps in order.  A tree's pads and level keys exist only where the tree needs them, and on the fused path
    (``pipe.fused``: 128 x 128 tiles over every row) ``gemm_pearlc`` writes the message digests itself, so ``hash_msg`` isn't
    a step."""
    by = {name: f for name, _, f in pipe.steps()}
    optional = {"hash_msg"} if getattr(pipe, "fused", False) else set()
    missing = [x for x in names if x not in by and not x.startswith(("pad_", "node_keys_")) and x not in optional]
    if missing:
        raise RuntimeError(f"the kernel's pipeline has no steps {missing}")
    return [by[x] for x in names if x in by]


def _run(kernel: Kernel, pipe: Any, names: tuple[str, ...], stream: int) -> None:
    kernel.dev.stream = stream
    try:
        for f in _steps(pipe, names):
            f()
    finally:
        kernel.dev.stream = None


class PearlCDeviceRun:
    """One install's Pearl-C workload on the device, with `pouw._Run`'s interface (`gemm`, `finish`, `commits`, `details`) and
    the passes the bench and the verifier need (`begin`, `end`)."""

    def __init__(self, scheme: C.PearlC, entry: str, kernel: Kernel, weights: dict[int, Any], roots: dict[int, bytes],
                 salt: bytes, beacon: bytes, hashing: bool = True, capacity: int = LEAF_CAPACITY, time_gemm: bool = False,
                 defer: bool = False):
        self.scheme, self.entry, self.kernel, self.hashing, self.capacity = scheme, entry, kernel, hashing, capacity
        self.time_gemm, self.events, self.omit = time_gemm, [], frozenset()
        self.torch_stream = _TorchStream()
        self.defer, self.side = defer, _Side(next(iter(weights.values())).device) if defer else None
        self.salt, self.beacon, self.roots = salt, beacon, dict(roots)
        self.device = next(iter(weights.values())).device
        self.epoch = Epoch(Workload(tuple(Matmul(w, Shape(ROW_ALIGN, t.shape[1], t.shape[0]), w) for w, t in weights.items())),
                           self.roots, salt)
        self.const = _zeros(64, self.device)
        _put(self.const, _format_keys(scheme))
        self.leaves = _zeros(32 * capacity, self.device)
        self.roots_buf = _zeros(32 * CALL_CAPACITY, self.device)
        self.flag_buf = _zeros(2 * FLAG_ROWS, self.device).view(FLAG_ROWS, 2)
        self.tree_hash = scheme.commitment_hash
        self.keys, self.uploads = _Keys(kernel.fixture, salt, self.tree_hash), _Uploads(self.device)
        self.fused_screen = self.device.type == "cuda" and fused_screen_available()
        self.weights = {w: self._weight_side(w, t) for w, t in weights.items()}
        self.shapes: dict[tuple[int, int, int], _Shape] = {}
        self.pipes: dict[tuple[int, int, frozenset, bool], tuple[Any, list[Any]]] = {}
        self.stream = lambda: _stream(self.device)
        self.commits: dict[str, Any] = {"scheme": scheme.name, "salt": salt.hex(), "weights_root": self.epoch.weights_root.hex(),
                                        "transcript_root": None, "units": 0, "tiles": 0}
        self.details: dict[str, Any] = {
            "scheme": scheme.name, "committer": f"pouw_pearl_c_device (GPU 1's pipeline, cubin {kernel.cubin_sha256[:16]}, "
                                                f"sources {(kernel.source_sha256 or 'unrecorded')[:16]})",
            "hashing": hashing, "passes": [], "excluded_b": {str(w): len(x.excluded) for w, x in self.weights.items()},
            "host": {"keys": "native blake3" if self.keys.native else "verity.commitments.blake3 (pure Python)",
                     "screen": "fused (Triton)" if self.fused_screen else "torch"},
            "schedule": "deferred (tile hashing on a side stream)" if defer else "serial",
            "uncovered": "every GEMM outside LinearBase (lm_head, attention); served y's glue (bias) is not bound (red team F4)"}
        self.cur: _Pass | None = None
        self.begin()

    # ---- the weight side --------------------------------------------------------------------------------------------

    def _weight_side(self, w: int, weight: Any) -> _Weight:
        import torch
        K, R = self.kernel, self.kernel.run
        n, k = weight.shape
        size = R.Pipeline.sizes(R.BM, n, k, R.BM)
        b = {name: _zeros(size[name], weight.device) for name in WEIGHT_BUFFERS}
        keys = _zeros(64, weight.device)
        mm = Matmul(0, Shape(ROW_ALIGN, k, n), w)
        seed_b = C._labelled(self.salt + self.roots[w], "seed-B")
        kk = K.fixture.keys_of(self.scheme, self.epoch, mm, seed_b)
        _put(keys, kk["line_b"] + kk["dom_b"])
        xb = weight.to(torch.float32).contiguous()
        pipe = _pipeline(K, R.BM, n, k, {**{x: t.data_ptr() for x, t in b.items()}, "line_a": 0}, xb.data_ptr(),
                         {"line_b": keys.data_ptr(), "dom_b": keys.data_ptr() + 32})
        stream = _stream(weight.device)
        _run(K, pipe, EPOCH_STEPS, stream)
        root = _bytes(b["tree_b"][size["tree_b"] - 32:size["tree_b"]])
        if root != self.roots[w]:
            raise ValueError(f"{self.entry}: weight {w}'s rows on the device root at {root.hex()[:16]}, not the committed "
                             f"{self.roots[w].hex()[:16]}")
        _run(K, pipe, JOB_STEPS, stream)
        return _Weight(n, k, b, keys, root, excluded(*screen(xb, b["alpha_b"], b["aux_b"]), xb), weight)

    # ---- calls ------------------------------------------------------------------------------------------------------

    def _shape(self, m16: int, n: int, k: int) -> _Shape:
        import torch
        key = (m16, n, k)
        if key not in self.shapes:
            R = self.kernel.run
            mp = R.cdiv(m16, R.BM_D) * R.BM_D
            size = R.Pipeline.sizes(mp, n, k, m16)
            sh = _Shape(m16, mp, n, k, {x: _zeros(size[x], self.device) for x in CALL_BUFFERS},
                        torch.zeros((mp, k), dtype=torch.float32, device=self.device), _zeros(32 + 104, self.device))
            sh.sets = [{x: sh.b[x] for x in ROTATED}]
            if self.defer:
                sh.sets += [{x: _zeros(size[x], self.device) for x in ROTATED} for _ in range(DEFER_SETS - 1)]
            sh.done = [None] * len(sh.sets)
            if self.cur is not None and self.cur.poison is not None:
                for t in self._buffers(sh):
                    t.fill_(self.cur.poison)
            self.shapes[key] = sh
        return self.shapes[key]

    @staticmethod
    def _buffers(sh: _Shape) -> list[Any]:
        """Every buffer a call of `sh` writes: its own, and the deferred schedule's other sets."""
        return [*sh.b.values(), *(t for st in sh.sets[1:] for t in st.values())]

    def _pipe(self, w: int, sh: _Shape) -> tuple[Any, list[Any], list[Any]]:
        """The call's pipeline and its steps: those on the call's stream, and (deferred, with hashing) the side stream's."""
        key = (w, sh.m16, self.omit, self.hashing, self.defer)
        if key not in self.pipes:
            wt, K = self.weights[w], self.kernel
            bufs = {**{x: t.data_ptr() for x, t in wt.b.items()}, **{x: t.data_ptr() for x, t in sh.b.items()}, "y": 0,
                    "leaves": 0}
            keys = {"line_b": wt.keys.data_ptr(), "dom_b": wt.keys.data_ptr() + 32, "dom_a": sh.keys.data_ptr(),
                    "unit": sh.keys.data_ptr() + 32, "k_msg": self.const.data_ptr(), "k_leaf": self.const.data_ptr() + 32}
            pipe = _pipeline(K, sh.mp, sh.n, sh.k, bufs, 0, keys, sh.xa.data_ptr(), sh.m16)
            names = tuple(x for x in CALL_STEPS if (self.hashing or x not in HASH_STEPS) and x not in self.omit)
            present = [x for x in names if x in {n for n, _, _ in pipe.steps()}]
            steps = _steps(pipe, names)
            if self.time_gemm and self.device.type == "cuda":
                at = present.index("gemm_pearlc")
                steps[at] = self._timed(steps[at])
            side = self.defer and self.hashing
            main = [f for x, f in zip(present, steps, strict=True) if not (side and x in SIDE_STEPS)]
            aside = [f for x, f in zip(present, steps, strict=True) if side and x in SIDE_STEPS]
            self.pipes[key] = (pipe, main, aside)
        return self.pipes[key]

    def _timed(self, f):
        import torch

        def timed():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            stream = self.torch_stream(_stream(self.device))
            a.record(stream)
            f()
            b.record(stream)
            self.events.append((a, b))
        return timed

    def gemm_ms(self) -> tuple[float, int]:
        """The GEMM kernel's summed time (``gemm_pearlc``, CUDA events on its stream) and its call count since the last read."""
        import torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        out = (sum(a.elapsed_time(b) for a, b in self.events), len(self.events))
        self.events.clear()
        return out

    def gemm(self, w: int, x: Any, weight: Any, bias: Any) -> Any:
        """One call, under inference mode as vLLM runs it: a call's buffers may be inference tensors made during a forward,
        which torch writes only there, and a caller outside a forward (the e2e's gate) runs the same code."""
        import torch
        with torch.inference_mode():
            return self._gemm(w, x, weight, bias)

    def _gemm(self, w: int, x: Any, weight: Any, bias: Any) -> Any:
        import torch
        cur, wt = self.cur, self.weights[w]
        lead, k = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, k)
        m = x2.shape[0]
        m16 = -(-m // ROW_ALIGN) * ROW_ALIGN
        sh = self._shape(m16, wt.n, k)
        index = len(cur.matmuls)
        tiles = self.kernel.run.cdiv(m16, self.kernel.run.TILE) * (wt.n // self.kernel.run.TILE)
        if self.hashing and (cur.tiles + tiles > self.capacity or index >= CALL_CAPACITY or cur.flag_rows + m > FLAG_ROWS):
            raise RuntimeError(f"{self.entry}: the pass is over its capacity ({self.capacity} tiles, {CALL_CAPACITY} calls, "
                               f"{FLAG_ROWS} screened rows); end it and begin another")
        sh.xa[:m].copy_(x2)
        if m16 > m:
            sh.xa[m:m16].zero_()
        y = torch.empty((sh.mp, wt.n), dtype=torch.bfloat16, device=x.device)
        pipe, steps, side = self._pipe(w, sh)
        pipe.b["y"], pipe.b["leaves"] = y.data_ptr(), self.leaves.data_ptr() + 32 * cur.tiles
        raw = self.stream()
        cuda = self.device.type == "cuda"
        if side:
            j = sh.turn % len(sh.sets)
            sh.turn += 1
            pipe.b.update({x: t.data_ptr() for x, t in sh.sets[j].items()})
            if cuda and sh.done[j] is not None:
                self.torch_stream(raw).wait_event(sh.done[j])
        if self.hashing:
            self.uploads.put(sh.keys, self.keys(self.roots[w], index, m16), self.torch_stream(raw) if self.uploads.cuda else None)
        self.kernel.dev.stream = raw
        try:
            for f in steps:
                f()
            if side:
                if cuda:
                    made = torch.cuda.Event()
                    made.record(self.torch_stream(raw))
                    self.side.stream.wait_event(made)
                self.kernel.dev.stream = self.side.raw
                for f in side:
                    f()
                if cuda:
                    sh.done[j] = sh.done[j] or torch.cuda.Event()
                    sh.done[j].record(self.side.stream)
        finally:
            self.kernel.dev.stream = None
        cur.matmuls.append(Matmul(index, Shape(m16, k, wt.n), w))
        if self.hashing:
            size = pipe.size["tree_a"]
            self.roots_buf[32 * index:32 * index + 32].copy_(sh.b["tree_a"][size - 32:size])
            flags = self.flag_buf[cur.flag_rows:cur.flag_rows + m]
            screen_into(flags, sh.xa[:m], sh.b["alpha_a"], sh.b["aux_a"], self.fused_screen)
            cur.flag_rows += m
            if cur.retain is not None:
                cur.records.append(PW.UnitRecord(cur.matmuls[-1], m, k, excluded(flags[:, 0].bool(), flags[:, 1].bool(), sh.xa[:m]),
                                                 wt.excluded))
                cur.files.append(self._retain_call(cur, index, sh, size))
            cur.tiles += tiles
        out = y[:m] if bias is None else y[:m] + bias
        return out.reshape(*lead, wt.n)

    def _retain_call(self, cur: _Pass, index: int, sh: _Shape, tree_bytes: int) -> dict:
        import torch
        d = cur.retain / "calls"
        d.mkdir(parents=True, exist_ok=True)
        rows = _bytes(sh.xa[:sh.m16].to(torch.bfloat16).view(torch.uint8))
        tree = _bytes(sh.b["tree_a"][:tree_bytes])
        return {"rows": _write(d / f"{index}.rows.bf16", rows, digest=False), "tree": _write(d / f"{index}.tree", tree)}

    # ---- passes -----------------------------------------------------------------------------------------------------

    def begin(self, retain: str | None = None, run: str | None = None) -> None:
        """A new pass: its own transcript under the install's epoch, the call indices from 0.  With `retain`, the pass writes
        what the prover opens from into that directory, from poisoned outputs, and its manifest names `run`."""
        self.cur = _Pass(Path(retain) if retain else None, run)
        if retain and self.hashing:
            self.poison()

    def poison(self, value: int = 0xA5) -> None:
        """Every buffer a call of this pass writes (each shape's, those created later in the pass too, and the leaves) filled
        with `value`.  Under inference mode: buffers made during a vLLM forward are inference tensors, which torch writes only
        there."""
        import torch
        self.cur.poison = value
        with torch.inference_mode():
            self.leaves.fill_(value)
            for sh in self.shapes.values():
                for t in self._buffers(sh):
                    t.fill_(value)

    def end(self) -> dict[str, Any]:
        """Close the pass: its tile tree on the device, every A root, and with `retain` the manifest.  Returns the pass's
        commitment (also in `commits` and `details["passes"]`)."""
        import torch
        cur, K = self.cur, self.kernel
        out: dict[str, Any] = {"units": len(cur.matmuls), "tiles": cur.tiles, "hashing": self.hashing}
        if self.hashing and cur.matmuls:
            if self.side is not None and self.side.stream is not None:
                self.torch_stream(self.stream()).wait_stream(self.side.stream)
            epoch = Epoch(Workload(tuple(cur.matmuls)), self.roots, self.salt)
            R, count = K.run, cur.tiles
            tree = _zeros(32 * (2 * R.pow2(count) - 1), self.device)
            lk, dom = _zeros(32 * max(R.depth(count), 1), self.device), _zeros(32, self.device)
            _put(dom, transcript_domain(epoch, count, self.tree_hash).domain_id)
            ns = types.SimpleNamespace(dev=K.dev, b={"tree_t": tree.data_ptr(), "keys_t": lk.data_ptr()},
                                       keys={"dom_t": dom.data_ptr()})
            K.dev.stream = self.stream()
            try:
                for _, f in R.Pipeline.tree_setup(ns, "t", count, "dom_t"):
                    f()
                K.dev.launch("hash_tree_leaves", R.cdiv(count, 128), 128, p=(self.leaves.data_ptr(), tree.data_ptr(), dom.data_ptr()),
                             i=(count,))
                R.Pipeline.levels(ns, "t", count)
            finally:
                K.dev.stream = None
            roots_a = _bytes(self.roots_buf[:32 * len(cur.matmuls)])
            flags = self.flag_buf[:cur.flag_rows].to(torch.int64).sum(0).tolist() if cur.flag_rows else [0, 0]
            out.update(transcript_root=_bytes(tree[-32:]).hex(),
                       activation_roots=hashlib.sha256(roots_a).hexdigest(),
                       screened_rows=cur.flag_rows, failing_rows=int(flags[0]), unsure_rows=int(flags[1]),
                       excluded_rows=sum(len(r.excluded_a) for r in cur.records) if cur.retain is not None else None)
            if cur.retain is not None:
                out["manifest"] = str(self._write_retained(cur, epoch, roots_a, tree, count))
            self.commits.update(transcript_root=out["transcript_root"], units=len(cur.matmuls), tiles=count)
        self.details["passes"].append(out)
        return out

    def _write_retained(self, cur: _Pass, epoch: Epoch, roots_a: bytes, tree: Any, count: int) -> Path:
        import torch
        d = cur.retain
        wd = d / "weights"
        wd.mkdir(parents=True, exist_ok=True)
        weights = {}
        for w in sorted(self.weights):
            wt = self.weights[w]
            size = self.kernel.run.Pipeline.sizes(self.kernel.run.BM, wt.n, wt.k, self.kernel.run.BM)["tree_b"]
            weights[str(w)] = {"n": wt.n, "k": wt.k, "root": wt.root.hex(), "excluded_b": list(wt.excluded),
                               "rows": _write(wd / f"{w}.rows.bf16", _bytes(wt.tensor.to(torch.bfloat16).contiguous().view(torch.uint8)),
                                              digest=False),
                               "tree": _write(wd / f"{w}.tree", _bytes(wt.b["tree_b"][:size]))}
        matmuls = [{"index": u.index, "m": u.shape.m, "k": u.shape.k, "n": u.shape.n, "weight": u.weight,
                    "real_rows": r.real_rows, "k_kept": r.k_kept, "excluded_a": list(r.excluded_a),
                    "root_a": roots_a[32 * u.index:32 * u.index + 32].hex(), **f}
                   for u, r, f in zip(cur.matmuls, cur.records, cur.files)]
        head = {"schema": MANIFEST_SCHEMA, "run": cur.run, "entry": self.entry, "scheme": self.scheme.name,
                "scheme_args": {k: getattr(SchemeId.parse(self.scheme.name), k) for k in ("forming", "device", "hashing")},
                "beacon": self.beacon.hex(), "salt": self.salt.hex(), "cubin_sha256": self.kernel.cubin_sha256,
                "kernel_source_sha256": self.kernel.source_sha256,
                "records_root": PW.records_root(self.salt, epoch.weights_root,
                                                {w: PW.Split(self.weights[w].k) for w in epoch.workload.weights()}, cur.records).hex(),
                "commitment": {"root": _bytes(tree[-32:]).hex(), "count": count},
                "leaves": _write(d / "leaves", _bytes(self.leaves[:32 * count])),
                "tree_t": _write(d / "tree_t", _bytes(tree))}
        return write_manifest(d, head, weights, matmuls)

    def finish(self) -> None:
        if self.cur is not None and self.cur.matmuls:
            self.end()
            self.cur = _Pass(None)


def _write(path: Path, data: bytes, digest: bool = True) -> dict:
    """A transcript file.  Row files carry no digest: the committed roots bind them, and the verifier checks every opened row
    against them."""
    path.write_bytes(data)
    nested = path.parent.name in ("calls", "weights")
    out = {"path": f"{path.parent.name}/{path.name}" if nested else path.name, "bytes": len(data)}
    if digest:
        out["sha256"] = hashlib.sha256(data).hexdigest()
    return out


def write_manifest(d: Path, head: dict, weights: dict[str, dict], matmuls: list[dict]) -> Path:
    """The pass's manifest, which `verify_run.py` reads: `head` (the scheme and its arguments, the beacon, the salt, the
    records root, the commitment, the leaves and the tile tree), every weight (n, k, root, excluded B rows, its rows and tree
    levels) and every matmul in order (index, m, k, n, weight, real rows, |K|, excluded A rows, A's root, rows and tree
    levels).  Trees are the device's layout: level 0 padded to a power of two, then each level above it, 32 bytes a node."""
    path = d / "manifest.json"
    path.write_text(json.dumps({**head, "weights": weights, "matmuls": matmuls}, indent=1) + "\n")
    return path
