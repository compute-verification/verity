"""PoUW as a protocol option: every linear layer's GEMM computed as a `verity.core.protocols.pouw` scheme's honest work.

The executor at each linear layer's ``quant_method.apply`` (``EXECUTES``; innermost, `protocol_options.SITES`).  In the
composition's weight phase, with the model untouched and no coin opened, `commit_weights` finds the linear layers by
type (vLLM's ``LinearBase``, `linear_types`) and commits every layer's int7 weight rows (`verity.core.protocols.pouw.audit.commit_rows`).
`install` then asks the epoch's challenger (the setting ``coins``, or under ``ncp-v2`` and Pearl-C the auditor host's
session, ``challenger``: `pouw_h1_window.salt_session`) for its salt coin, opened after those roots' weights root, and
starts the epoch from it (`verity.core.protocols.pouw.audit.Epoch.start`, which refuses any other coin; Pearl-C's
`pouw_pearl_c_host.epoch_of`, a live session's under the same rules), and replaces each ``apply`` through the
composition's tagged `Hooks`.  There is no beacon and no salt without a coin.  Each call of a replaced
``apply(layer, x, bias)`` is one unit of the run's workload:

1. quantize x's rows to int7 per row (`quantize`), and prepare the weight operands (`weight_operands`: the int7 weight per
   output channel with the epoch's noise, Y = [B + F₁ ; B + F₂ ; B + F₃], and c₀) from the weight of the layer the call
   is handed, so no weight-derived copy outlives the call and under POUS the GEMM reads POUS's decoded weight;
2. commit the unit's activation rows, derive its noise and compute every 16 x 16 tile's route-U checked values and
   digest (`ncp_tiles`, gated bit for bit against the scheme's reference in the tests); the digests join the transcript;
3. return the useful output, the final checkpoint A·Bᵀ, dequantized by the row and column scales, plus the bias.

Its commitment to the weights of record is made first, in the composition's weight phase (`commit_weights`: every
layer's weight root); `commitment` is the scheme, the salt, the weights root the salt is derived from, the epoch's coin
commitment and its salt coin's opening and, once closed,
the transcript root with its unit and tile counts.  Pearl-C's also says what it claims of the weights (``registration``,
`weights_claim`): the executor applies no keyed rotation and reads no curated list, so it claims nothing about which
weights were served.  Pearl-C commits one pass per engine step, a step held to what one pass takes (`pouw_pass`), each
retained in its own directory under ``transcript_dir`` when given.  The audit (drawing tiles, opening
them) is `verity.core.protocols.pouw.audit`'s, after the run.  Beside sampled proofs the composition asks
`verity.core.protocols.pouw.circuit` (`protocol_options.CIRCUITS`), and only ``ncp-v2`` has a circuit.  Under ``ncp-v2`` each weight's side (its Y strips and c₀) is instead bound to its own
`NcpWeight_v1` Program call, once per run, and each call to its own `NcpLinear_v1` Program call reading it, their units
computed natively and committed a leaf per unit (`pouw_native`, `pouw_circuit`); the replay draws each call's units under
its window's key, regenerates them and checks each against its template (`check.replay.pouw_circuit`).
Under H1 + fx1 (``eps1-sm120-h1fx1``, `pouw_h1`) on the eager int8_exact row nothing is executed in the linear's place
(`executes`): the arm serves its own y, and each call is recorded as its own Program, compiled to H1 tile units
(`verity.core.protocols.pouw.circuit.compile`), with its seed group's roots and its positions.
What is a placeholder is listed in `info` (``placeholder``).
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from verity.ml.commitments import MerkleTree
from verity.core.protocols.pouw import circuit as pouw_circuit
from verity.core.protocols.pouw import schemes as pouw_schemes
from verity.core.protocols.pouw.audit import (
    TILE_SCHEMA,
    Epoch,
    Workload,
    activation_binding,
    commit_rows,
    transcript_domain,
    weight_binding,
    weights_root,
)
from verity.core.protocols.pouw import verifier
from verity.core.protocols.pouw.coins import Challenger, CoinCommitment
from verity.core.protocols.pouw.protocol import Context as UnitContext
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape
from verity.core.protocols.pouw.schemes import ncp
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs.challenger import LiveRecord

from verity_vllm.protocols.pouw import circuit as PC
from verity_vllm.protocols.pouw import device as PD
from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import hooks as PHK
from verity_vllm.protocols.pouw.h1 import settings as PHS
from verity_vllm.protocols.pouw.h1.settings import fills, from_engine  # noqa: F401  (the adapter's hooks)
from verity_vllm.protocols.pouw.h1 import window as PHW
from verity_vllm.protocols.pouw.h1 import requests as RQ
from verity_vllm.protocols.pouw import native as PN
from verity_vllm.protocols.pouw.pearl_c import passes as PP
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.interface import Context, Option, Target

NAME = "pouw"
#: PoUW replaces the computation here; POUS wraps it from outside (`protocol_options.SITES`).
EXECUTES = ("LinearBase.quant_method.apply",)
#: The schemes this adapter's executor computes. ``ncp-v2`` is NCP-INT as a circuit (`verity.core.protocols.pouw.circuit`): each call is its
#: own `NcpLinear_v1` Program call, reading its weight's `NcpWeight_v1` call (`pouw_circuit`).  Pearl-C on sm_120 and sm_90a
#: (`pouw_pearl_c_device.SCHEMES`) runs on the device only, through GPU 1's kernel pipeline (settings ``kernel``, ``cubin``).
#: H1 + fx1 (`pouw_h1.SCHEME`) runs beside the int8_exact arm, which serves y itself.
EXECUTED = ("ncp-v1", "ncp-v1-shift24", "ncp-v2", *PPD.SCHEMES, PH.SCHEME)
CHALLENGED = PHS.CHALLENGED
#: The weight dtypes the executor reads (it quantizes them itself); any other, an FP8 checkpoint's float8 above all, is refused
UNQUANTIZED = ("bfloat16", "float16", "float32")
#: A quantized linear's parameters beside (or instead of) its weight: a layer carrying one is refused whatever its weight's
#: dtype, and a packed one (compressed-tensors' ``weight_packed``) has no ``weight`` to read at all.
QUANT_PARAMS = ("weight_scale", "weight_scale_inv", "input_scale", "weight_zero_point", "weight_packed")
#: The pinned int7 quantizer of the FP8 lane's reference (docs/pouw-fp8/reference.md), applied without a basis rotation.
QUANTIZER = "verity.pouw.symmetric-int7-rowmax-rne-f32/v1"
KEYS = ("coins", "retain_steps", "transcript_dir", "draws", "tamper", "device", "kernel", "cubin", "schedule", "start_forward",
        "rows", "steps", "runner", "rounds", "challenger", "session", "admission", "gate", "served", "graphs", "period")
#: ``ncp-v2``'s negative controls (`Settings.tamper`): a prover that commits and opens one value wrong
TAMPERS = ("tile", "bias")
#: where the Commit computes `ncp-v2`'s values: the host (`pouw_native`, the default) or ``cuda``, the serving kernel
#: (`pouw_device.commit_call(on="fused")`, held to `pouw_native` leaf for leaf by its gate)
DEVICES = ("cuda",)
PLACEHOLDER = (
    "a forward's weight is not re-checked against the root committed at install",
    "not auditable yet: one epoch per install (no rotation), activation rows kept only in memory for the last "
    "`retain_steps` units (0 by default) and never persisted, no opener, and the transcript's leaves grow without bound "
    "for the life of the install",
    "the executor runs on the host in numpy, not a native route-U kernel",
)
#: The transforms Pearl-C's executor applies to a checkpoint's weights before committing them: none.  The keyed 8-block
#: rotation (`registration.ROTATION_8`) is not built on the device, so served FP8 registers only a curated unrotated
#: checkpoint, and no curated list is wired here.
SERVED_TRANSFORMS: tuple[str, ...] = ()
#: What Pearl-C's commitment says of weights that are not registered (`weights_claim`)
UNREGISTERED = "nothing about which weights were served: the checkpoint is not registered"
#: The host bytes `weight_operands` works in at a time (its copy of the weight's rows and their int64 noise and sums).
FORM_BYTES = 1 << 28
_INV63 = np.float32(1.0) / np.float32(63.0)
_M32 = 0xFFFFFFFF


def schemes() -> tuple[str, ...]:
    return tuple(sorted({*pouw_schemes.SCHEMES, *pouw_circuit.SCHEMES, *PPD.SCHEMES, PH.SCHEME}))


def executes(scheme: str) -> tuple[str, ...]:
    """The sites where `scheme`'s executor replaces the computation: none under H1, whose arm serves its own y."""
    return () if scheme == PH.SCHEME else EXECUTES


@dataclass(frozen=True)
class Settings:
    #: the epoch's challenger, in-process: its coin commitment (``commitment``, published before the epoch) and
    #: ``open(label, after)``, each coin once (`verity.core.protocols.pouw.coins.Challenger`, or a proxy of one)
    coins: Any = None
    retain_steps: int = 0
    transcript_dir: str | None = None
    #: ``ncp-v2``'s replay: the window's work-law budget K, each call drawing its share (`plan.window_laws`); None is
    #: `RECORD_K`
    draws: int | None = None
    #: ``ncp-v2``'s negative control, None for an honest run: ``tile`` commits and opens word 7 of every tile of the first
    #: call with its low bit flipped, so whatever the window's fresh draw, a drawn tile is wrong
    #: (`check.replay.pouw_circuit.tamper_tiles`); ``bias`` commits and serves the first biased call with bias word 0's low
    #: bit flipped
    tamper: str | None = None
    device: str | None = None
    #: Pearl-C on sm_120: the kernel directory (the run tree's ``benchmarks/pouw/pearl_c_sm120``) and build.sh's cubin
    kernel: str | None = None
    cubin: str | None = None
    #: Pearl-C on sm_120: the schedule each pass runs under (`verity.core.protocols.pouw.serving.SCHEDULES`), serial by default
    schedule: str = "serial"
    #: ``ncp-v2``: the anchored forward index the install's first pass takes.  A salt's first window starts at 0; a second
    #: served window under the same salt starts at the verifier's `anchors.SaltWindows.next_forward`, so no call index
    #: repeats across the salt's windows (a repeat would reuse a noise key, and `SaltWindows.open` refuses it)
    start_forward: int = 0
    #: H1's static layout (`pouw_h1.H1Run`): T, the rows of every call (vLLM's ``max_num_batched_tokens``, a multiple of 32),
    #: and S, the steps of the install's window, public parameters that fix every call's rows and positions
    rows: int | None = None
    steps: int | None = None
    #: H1's admissions and rounds (`pouw_h1.H1Run.admit`): the engine's model runner (Model Runner V2, whose
    #: ``decode_query_len`` orders a step's rows, `pouw_h1_requests.batch_order`), whose ``execute_model`` gets each step's
    #: scheduler output, the requests it admits and schedules among it; and the window's round coins, a `LiveRecord` over a
    #: live challenger's round family (`verity.core.protocols.pouw.window.live_challenger`), which opens round s's coin
    #: after round s's root
    runner: Any = None
    rounds: Any = None
    #: H1's salt and round coins (``ncp-v2``'s and Pearl-C's salt) from the auditor host's challenger, ``HOST:PORT``
    #: (`pouw_h1_challenger`); ``coins`` and ``rounds`` held in this process in its place are for tests, and credit
    #: nothing; ``session`` is the challenger gate's reply for the window: ncp-v2's its open answer for the window's
    #: Commit (`pouw_h1_challenger.session_of`), H1's and Pearl-C's its reserve answer (`pouw_h1_window.reserved_of`),
    #: which the install opens itself, on H1's window-open root or Pearl-C's weights' root (`pouw_h1_settings.OPENS`)
    challenger: str | None = None
    session: Mapping[str, Any] | None = None
    #: H1's admissions (`pouw_h1_requests.ADMISSIONS`): None or ``round-0``, admit in round 0 only (the live tiles
    #: are fixed before the first coin), or ``per-round``
    admission: str | None = None
    #: H1's and Pearl-C's ``gate(session, root)``: the challenger gate's open of the reserved session on that root
    #: (`pouw_h1_window.Gate`; `pouw_h1_window.Window`, `pouw_h1_window.salt_session`)
    gate: Any = None
    #: H1's served tables (`pouw_h1_served.Served`): each call's and the weights' registered values, under the round roots;
    #: ``graphs`` the engine's CUDA-graph capture sizes, none when eager (`pouw_h1_static.graph_pad`)
    served: Any = None
    graphs: tuple[int, ...] = ()
    #: Pearl-C's: the lane's open period before ``session``'s reserve, or None (`pouw_h1_window.reserve_salt`, `period_reason`)
    period: Mapping[str, Any] | None = None

    @classmethod
    def of(cls, raw: Mapping[str, Any]) -> Settings:
        """PoUW's keys of the protocols config; ValueError names the first bad one."""
        unknown = sorted(set(raw) - set(KEYS))
        if unknown:
            raise ValueError(f"unknown pouw settings {unknown} (known: {', '.join(KEYS)})")
        coins = raw.get("coins")
        if coins is not None and not (isinstance(getattr(coins, "commitment", None), CoinCommitment)
                                      and callable(getattr(coins, "open", None))):
            raise ValueError("pouw setting 'coins' is the epoch's challenger: its coin commitment and open(label, after)")
        retain = raw.get("retain_steps", 0)
        if not isinstance(retain, int) or isinstance(retain, bool) or retain < 0:
            raise ValueError("pouw setting 'retain_steps' is a non-negative integer")
        tdir = raw.get("transcript_dir")
        if tdir is not None and not isinstance(tdir, str):
            raise ValueError("pouw setting 'transcript_dir' is a path string")
        draws = raw.get("draws")
        if draws is not None and (not isinstance(draws, int) or isinstance(draws, bool) or draws < 1):
            raise ValueError("pouw setting 'draws' is a positive integer")
        tamper = raw.get("tamper")
        if tamper is not None and tamper not in TAMPERS:
            raise ValueError(f"pouw setting 'tamper' is one of {', '.join(TAMPERS)}")
        device = raw.get("device")
        if device is not None and device not in DEVICES:
            raise ValueError(f"pouw setting 'device' is one of {', '.join(DEVICES)}")
        paths = [raw.get(key) for key in ("kernel", "cubin")]
        if any(p is not None and not isinstance(p, str) for p in paths):
            raise ValueError("pouw settings 'kernel' and 'cubin' are path strings")
        from verity.core.protocols.pouw.serving import SCHEDULES
        schedule = raw.get("schedule", "serial")
        if schedule not in SCHEDULES:
            raise ValueError(f"pouw setting 'schedule' is one of {', '.join(SCHEDULES)}")
        start = raw.get("start_forward", 0)
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError("pouw setting 'start_forward' is a non-negative integer (the salt's SaltWindows.next_forward)")
        return cls(coins, retain, tdir, draws, tamper, device, *paths, schedule, start, *PHS.of(raw))


def refusals(scheme: str, ctx: Context) -> tuple[str, ...]:
    out = []
    if scheme not in EXECUTED:
        out.append(f"{scheme}: this adapter's executor runs {', '.join(EXECUTED)} only")
    if not ctx.enforce_eager:
        out.append("compiled execution: the wrapper around each linear layer's apply is not graph-capturable")
    if ctx.world > 1:
        out.append(f"TP {ctx.world}: PoUW's units and transcript are per rank, which is not wired")
    try:
        settings = Settings.of(ctx.settings_of(NAME))
    except ValueError as e:
        out.append(str(e))
    else:
        if scheme in PPD.SCHEMES and (settings.device != "cuda" or not settings.kernel):
            out.append(f"{scheme}: runs on the device only, so it needs the pouw settings 'device': 'cuda' and 'kernel'")
        if scheme in PPD.SCHEMES:
            try:
                PPD.schedule_of(settings.schedule, PPD.scheme_of(scheme))
            except ValueError as e:
                out.append(str(e))
        elif settings.schedule != "serial":
            out.append(f"{scheme}: the pouw setting 'schedule' is Pearl-C's; this executor runs serially")
        out += PHS.refusals(scheme, settings, ctx.stage, ctx.filled_of(NAME))
        if settings.start_forward and scheme not in pouw_circuit.SCHEMES:
            out.append(f"{scheme}: the pouw setting 'start_forward' anchors the circuit-backed executor's forward indices "
                       f"({', '.join(pouw_circuit.SCHEMES)}); this executor counts its own from 0")
    return tuple(out)


def requires_build(scheme: str, ctx: Context) -> bool:
    """The circuit-backed executor is lowered into the row's Build."""
    return scheme in pouw_circuit.SCHEMES


def keeps_weight_copy(scheme: str, ctx: Context) -> bool:
    """True where the executor keeps a weight-derived copy across forwards, so its GEMMs would not read what POUS decodes:
    Pearl-C's B̃, formed once per install (`pouw_pearl_c_device.PearlCDeviceRun`), and ncp-v2's weight sides, committed at
    each weight's first call and reused (`_CircuitRun.sides`).  ncp-v1 forms its operands per call from the weight it is
    handed."""
    return scheme in PPD.SCHEMES or scheme in pouw_circuit.SCHEMES


def linear_types() -> tuple[type, ...]:
    """The hook-site types: vLLM's ``LinearBase`` (qkv, o, gate_up, down).  lm_head is a ``VocabParallelEmbedding``, not
    covered."""
    from vllm.model_executor.layers.linear import LinearBase
    return (LinearBase,)


# --- operands --------------------------------------------------------------------------------------------------------------

def quantize(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``QUANTIZER`` per row: scale = RN_f32(rowmax · RN_f32(1/63)) (1 for a zero row), q = clamp(RNE(f32(v / scale)), −63, 63).
    Returns int8 rows and their FP32 scales; non-finite input is refused."""
    v = np.asarray(v, dtype=np.float32)
    if not np.isfinite(v).all():
        raise ValueError("pouw: the quantizer refuses NaN and infinity")
    scale = (np.abs(v).max(axis=1) * _INV63).astype(np.float32)
    scale[scale == 0] = np.float32(1.0)
    return np.clip(np.rint(v / scale[:, None]), -63, 63).astype(np.int8), scale


def _noise(tag: str, count: int, k: int, parts) -> np.ndarray:
    """The scheme's noise rows (`ncp.noise_row`: one XOF byte's low six bits minus 31 per entry), as int64."""
    return np.stack([np.frombuffer(ncp.xof(tag, k, *parts(i)), dtype=np.uint8).astype(np.int64) % 64 - 31 for i in range(count)])


@dataclass(frozen=True)
class Operands:
    """One weight's operands for an epoch: its column scales, Y = [B + F₁ ; B + F₂ ; B + F₃] as n rows of 3k, and
    c₀ = −β·Σ_l Y[l][j], the route-U accumulator's initial value (Lean `initU`)."""

    scales: np.ndarray
    y: np.ndarray
    c0: np.ndarray


def stack_y(scheme: Any, salt: bytes, w: int, b: np.ndarray, noise: tuple, first: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Y (as int8 rows: every entry lies in [−128, 127], Lean `rangeU`) and c₀ for the int7 weight rows ``b`` (n x k, the
    weight's rows from ``first``), with F₁ from the epoch salt (per epoch, weight and row, as `ncp.stack` draws it),
    F₂ = −P·F₁ and F₃ = F₁ + F₂ (`ncp.assemble`)."""
    n, k = b.shape
    pi, sigma = np.asarray(noise[0]), np.asarray(noise[1], dtype=np.int64)
    f1 = _noise("verity/pouw/ncp/F1/v1", n, k, lambda j: (salt, w, first + j))
    f2 = np.empty_like(f1)
    f2[:, pi] = -(f1 * sigma)
    b = b.astype(np.int64)
    y = np.concatenate([b + f1, b + f2, b + f1 + f2], axis=1)
    return y.astype(np.int8), (-scheme.beta * y.sum(axis=1)).astype(np.float64)


def weight_operands(scheme: Any, salt: bytes, w: int, weight: Any, noise: tuple) -> Operands:
    """``weight`` (n x k, a tensor or an array) quantized per output channel (`quantize`) and stacked (`stack_y`), a block
    of rows at a time: its host copy and int64 work stay under about `FORM_BYTES` whatever the layer, beside Y itself."""
    n, k = weight.shape
    step = max(1, FORM_BYTES // (128 * k))
    scales, y, c0 = np.empty(n, np.float32), np.empty((n, 3 * k), np.int8), np.empty(n, np.float64)
    for j in range(0, n, step):
        b, scales[j:j + step] = quantize(_host(weight[j:j + step]))
        y[j:j + step], c0[j:j + step] = stack_y(scheme, salt, w, b, noise, j)
    return Operands(scales, y, c0)


def ncp_tiles(scheme: Any, ctx: UnitContext, noise: tuple, a: np.ndarray, ops: Operands,
              chunk: int = 256) -> dict[tuple[int, int], tuple[bytes, np.ndarray]]:
    """One unit's tiles as the scheme's Prover kernel contract gives them: (first row, first column) -> (digest, int32
    output tile), for int7 ``a`` (m x k) and the weight's `Operands`.  X + β is formed as `ncp.assemble` forms X, the
    accumulator starts at c₀ (Lean `mmWU β`), and a tile's digest covers every depth-16 checkpoint, step-major and
    row-major within a step (`NCP.digest`).  Float64 products are exact here: every partial sum is an integer under 2^53."""
    u = ctx.matmul
    (m, k), n = a.shape, ops.y.shape[0]
    pi, sigma = np.asarray(noise[0]), np.asarray(noise[1], dtype=np.int64)
    a = a.astype(np.int64)
    e1 = _noise("verity/pouw/ncp/E1/v1", m, k, lambda i: (ctx.salt, u.index, ctx.root_a, u.weight, i))
    e2 = e1[:, pi] * sigma
    x = np.concatenate([a + e1, a + e2, -(a + e1 + e2)], axis=1) + scheme.beta
    y, c0 = ops.y, ops.c0
    steps = 3 * k // ncp.D
    out: dict[tuple[int, int], tuple[bytes, np.ndarray]] = {}
    for r in range(0, m, 16):
        xs = x[r:r + 16].reshape(16, steps, ncp.D).transpose(1, 0, 2).astype(np.float64)       # [T, 16, d]
        for c in range(0, n, chunk):
            ys = y[c:c + chunk].reshape(-1, steps, ncp.D).transpose(1, 2, 0).astype(np.float64)  # [T, d, w]
            run = np.cumsum(np.matmul(xs, ys), axis=0) + c0[None, None, c:c + chunk]
            words = (run.astype(np.int64) & _M32).astype("<u4")                                  # [T, 16, w]
            for j in range(0, words.shape[2], 16):
                block = np.ascontiguousarray(words[:, :, j:j + 16])
                digest = hashlib.shake_256(b"verity/pouw/ncp/tile/v1\0" + block.tobytes()).digest(32)
                out[(r, c + j)] = (digest, block[-1].astype(np.int64) - ((block[-1].astype(np.int64) >> 31) << 32))
    return out


def _host(t: Any) -> np.ndarray:
    return t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t, dtype=np.float32)


def _like(y: np.ndarray, x: Any) -> Any:
    if hasattr(x, "detach"):
        import torch
        return torch.from_numpy(y).to(device=x.device, dtype=x.dtype)
    return y.astype(np.asarray(x).dtype)


# --- the run -----------------------------------------------------------------------------------------------------------------

class _Run:
    """One install's workload: the units served so far, their activation roots and the transcript's leaves."""

    def __init__(self, scheme: Any, entry: str, settings: Settings, weight_roots: dict[int, bytes], salt: bytes) -> None:
        self.scheme, self.entry, self.settings = scheme, entry, settings
        self.weight_roots, self.salt = weight_roots, salt
        self.units: list[Matmul] = []
        self.roots_a: list[bytes] = []
        self.leaves: list[bytes] = []
        self.retained: deque = deque(maxlen=settings.retain_steps)
        cert = scheme.certificate
        self.commits: dict[str, Any] = {
            "scheme": scheme.name, "salt": salt.hex(), "weights_root": weights_root(weight_roots).hex(),
            "quantizer": QUANTIZER, "transcript_root": None, "units": 0, "tiles": 0}
        self.details: dict[str, Any] = {
            "scheme": scheme.name, "quantizer": QUANTIZER, "basis": "none", "layers": len(weight_roots),
            "certificate": {"conjecture": cert.conjecture, "gamma": None if cert.gamma is None else str(cert.gamma),
                            "parameters": dict(cert.parameters)},
            "units": 0, "tiles": 0, "padded_rows": 0, "uncovered": "every GEMM outside LinearBase (lm_head, attention)",
            "placeholder": list(PLACEHOLDER)}

    def noise_of(self, w: int, n: int, k: int) -> tuple:
        """The scheme's permutation at depth k (its noise before the salt-derived rows)."""
        return self.scheme.noise(UnitContext(self.salt, Matmul(0, Shape(16, k, n), w), b"", self.weight_roots[w]))

    def gemm(self, w: int, x: Any, weight: Any, bias: Any) -> Any:
        ops = weight_operands(self.scheme, self.salt, w, weight, self.noise_of(w, *weight.shape))
        n, k = ops.y.shape[0], ops.y.shape[1] // 3
        xs = _host(x)
        lead = xs.shape[:-1]
        a, s = quantize(xs.reshape(-1, k))
        t = ops.scales
        m = a.shape[0]
        mp = max(16, -(-m // 16) * 16)
        if mp > m:
            a = np.concatenate([a, np.zeros((mp - m, k), np.int8)])
        rows = [a[i].tobytes() for i in range(mp)]
        u = Matmul(len(self.units), Shape(mp, k, n), w)
        self.scheme.check(u)
        root_a = commit_rows(activation_binding(self.salt, u.index), u.index, rows).commitment.root
        ctx = UnitContext(self.salt, u, root_a, self.weight_roots[w])
        tiles = ncp_tiles(self.scheme, ctx, self.scheme.noise(ctx), a, ops)
        z = np.empty((mp, n), np.int64)
        for tile in self.scheme.tiles(u.shape):
            digest, block = tiles[(tile.rows[0], tile.cols[0])]
            self.leaves.append(digest)
            z[tile.rows[0]:tile.rows[-1] + 1, tile.cols[0]:tile.cols[-1] + 1] = block
        self.units.append(u)
        self.roots_a.append(root_a)
        self.retained.append((u.index, rows))
        self.details.update(units=len(self.units), tiles=len(self.leaves), padded_rows=self.details["padded_rows"] + mp - m)
        y = (s[:, None] * t[None, :]).astype(np.float32) * z[:m].astype(np.float32)
        out = _like(y.reshape(*lead, n), x)
        return out if bias is None else out + bias

    def finish(self) -> None:
        """Commit the transcript: the Merkle root over every tile digest, in unit and tile order (`Prover.prove`'s tree)."""
        if self.leaves:
            epoch = Epoch(Workload(tuple(self.units)), dict(self.weight_roots), self.salt)
            tree = MerkleTree(transcript_domain(epoch, len(self.leaves)), dict(enumerate(self.leaves)), lambda _: TILE_SCHEMA)
            self.commits["transcript_root"] = tree.commitment.root.hex()
        self.commits.update(units=len(self.units), tiles=len(self.leaves))
        if self.settings.transcript_dir:
            d = Path(self.settings.transcript_dir)
            d.mkdir(parents=True, exist_ok=True)
            units = [{"index": u.index, "shape": [u.shape.m, u.shape.k, u.shape.n], "weight": u.weight, "root_a": r.hex()}
                     for u, r in zip(self.units, self.roots_a)]
            (d / "pouw-transcript.json").write_text(json.dumps(
                {"entry": self.entry, "commitment": self.commits, "units": units, "leaves": [x.hex() for x in self.leaves],
                 "retained": [i for i, _ in self.retained]}, indent=1) + "\n")


def _f32_bits(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float32).view(np.uint32)


def _bf16_bits(v: np.ndarray) -> np.ndarray:
    """BF16 encodings of finite FP32 values, rounded to nearest even."""
    b = _f32_bits(v).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


class _LayerRows:
    """A layer's weight rows, quantized per output channel (`quantize`) as they are read, from the layer's own tensor:
    no copy of the weight outlives the read (`pouw_circuit.Rows`)."""

    def __init__(self, weight: Any) -> None:
        self.weight = weight
        self.n, self.k = weight.shape

    def rows(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        b, t = quantize(_host(self.weight[lo:hi]))
        return b, _f32_bits(t)


class _CircuitRun:
    """One install's `ncp-v2` workload: every call bound to its `NcpLinear_v1` Program's inputs, its units computed
    natively and committed (`pouw_native`). Without ``commit`` (a Match capture) each call serves the same y
    (`pouw_native.served`) and commits nothing."""

    def __init__(self, scheme: str, entry: str, settings: Settings, weight_roots: dict[int, bytes], salt: bytes,
                 commit: bool = True) -> None:
        self.scheme, self.entry, self.settings, self.commit = scheme, entry, settings, commit
        self.weight_roots, self.salt = weight_roots, salt
        self.tampered: dict[str, Any] | None = None
        self.records: list[PC.CallRecord] = []
        self.sides: dict[int, PC.WeightRecord] = {}     # each weight's side, committed at the run's first call on it
        self.seconds: list[float] = []                  # each call's executor wall time: quantize, commit, dequantize
        self.forward, self.row_base, self.rows_seen = settings.start_forward - 1, 0, 0
        self.commits: dict[str, Any] = {
            "scheme": scheme, "salt": salt.hex(), "weights_root": weights_root(weight_roots).hex(), "quantizer": QUANTIZER,
            "calls_root": None, "calls": 0, "start_forward": settings.start_forward}
        self.details: dict[str, Any] = {
            "scheme": scheme, "quantizer": QUANTIZER, "basis": "none", "layers": len(weight_roots),
            "circuit": "verity.core.protocols.pouw.circuit: each weight's side is its own NcpWeight_v1 Program call, once per run, and each "
                       "call its own NcpLinear_v1 Program call reading it, their units computed natively and committed a leaf "
                       "per unit; y served, credited words committed",
            "permutation": PC.kind_of(scheme),
            "calls": 0, "padded_rows": 0, "commit_seconds": 0.0, "uncovered": "every GEMM outside LinearBase (lm_head, attention)",
            "committer": "pouw_device fused (ncp2_fused.cu, cuda)" if settings.device == "cuda" else "pouw_native (host)",
            "placeholder": list(PLACEHOLDER)}

    def gemm(self, w: int, x: Any, weight: Any, bias: Any) -> Any:
        """The call's anchored fields: the forward index counts the passes from ``start_forward`` (a pass starts at linear
        0; a salt's second window starts where `anchors.SaltWindows.next_forward` says), the position is
        ``w << 32``, the weight id is ``w``, a pass's rows get global ids from the rows served before it, and the weight's
        rows are its columns 0 … n − 1."""
        t0 = time.monotonic()
        if w == 0:
            self.forward += 1
            self.row_base = self.rows_seen
        xs = _host(x)
        n, k = weight.shape
        lead = xs.shape[:-1]
        a, s = quantize(xs.reshape(-1, k))
        m = a.shape[0]
        mp = max(16, -(-m // 16) * 16)
        if mp > m:
            a = np.concatenate([a, np.zeros((mp - m, k), np.int8)])
            s = np.concatenate([s, np.ones(mp - m, np.float32)])
        if w == 0:
            self.rows_seen += mp
        bias_bits = np.full(n, 0x8000, np.uint16) if bias is None else _bf16_bits(_host(bias).reshape(-1))   # −0: the bf16 add's identity
        if self.settings.tamper == "bias" and bias is not None and self.tampered is None and self.commit:
            bias_bits = bias_bits.copy()
            bias_bits[0] ^= 1
            self.tampered = self.details["tampered"] = {"what": "bias", "call": len(self.records), "word": 0}
        call = PC.Call(a, _f32_bits(s), _LayerRows(weight), bias_bits, self.salt, forward=self.forward, layer=w, site=0, weight=w,
                       rowids=np.arange(self.row_base, self.row_base + mp, dtype=np.uint32), colids=np.arange(n, dtype=np.uint32))
        if self.commit:
            side = self.sides.get(w)
            if self.settings.device == "cuda":
                rec = PD.commit_call(len(self.records), self.scheme, call, on="fused", side=side)
            else:
                rec = PN.commit_call(len(self.records), self.scheme, call, side=side or PN.commit_weight(self.scheme, call))
            self.sides[w] = rec.side
            self.records.append(rec)
            served = rec.y
        else:
            served = PN.served(call)
        y = _like(((np.asarray(served[:m], dtype=np.uint32) << 16).view(np.float32)).reshape(*lead, n), x)
        self.seconds.append(time.monotonic() - t0)
        self.details.update(calls=len(self.seconds), padded_rows=self.details["padded_rows"] + mp - m,
                            commit_seconds=round(sum(self.seconds), 3))
        return y

    def finish(self) -> None:
        """Commit the calls: the root over every call's leaf (`pouw_circuit.calls_root`)."""
        if self.records:
            self.commits["calls_root"] = PC.calls_root(self.salt, bytes.fromhex(self.commits["weights_root"]), self.records).hex()
        self.commits["calls"] = len(self.records)
        if self.settings.transcript_dir:
            d = Path(self.settings.transcript_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / PC.CALLS_FILE).write_text(json.dumps({"entry": self.entry, "commitment": self.commits,
                                                        "weights": [PC.weight_summary(s) for s in PC.sides(self.records)],
                                                        "calls": [PC.summary(r) for r in self.records]}) + "\n")
            PC.save_calls(d / PC.CALLS_DATA_FILE, self.records)


#: The live `ncp-v2` runs by entry, with their sites (the collector's path of each linear -> its ordinal), for the Commit's
#: replay (`check.replay.pouw_circuit.around`) before the composition closes.
_CIRCUIT_RUNS: dict[str, tuple[_CircuitRun, dict[str, int]]] = {}
# The Pearl-C prover's hm96 salt key per entry: its weight trees' salts, drawn afresh by each `commit_weights` (so each install's W,
# and its salt session, is its own) and read by `install`, never in the record (published commitments without it fail its root check).
_SALT_KEYS: dict[str, bytes] = {}


def circuit_run(entry: str) -> tuple[_CircuitRun, dict[str, int]] | None:
    return _CIRCUIT_RUNS.get(entry)


def _sites(model: Any, layers: list[Any]) -> dict[str, int]:
    named = getattr(model, "named_modules", None)
    if named is None:
        return {}
    ids = {id(m): w for w, m in enumerate(layers)}
    return {name: ids[id(m)] for name, m in named() if id(m) in ids}


@cache
def ncp_linear_op() -> Any:
    """``verity_pouw::ncp_linear(x, weight, bias, entry, w) -> y``: one `ncp-v2` call as one dispatcher op, whose body is the
    live run's `gemm`. It is the executor's own record of the call: an observer records the op with its operands and its
    served y, and nothing of the host work inside it (`observe.fold.patterns.pouw`)."""
    import torch

    @torch.library.custom_op("verity_pouw::ncp_linear", mutates_args=(), schema="(Tensor x, Tensor weight, Tensor? bias, str entry, int w) -> Tensor")
    def ncp_linear(x, weight, bias, entry, w):
        return _CIRCUIT_RUNS[entry][0].gemm(w, x, weight, bias)

    @ncp_linear.register_fake
    def _(x, weight, bias, entry, w):
        return x.new_empty((*x.shape[:-1], weight.shape[0]))

    return torch.ops.verity_pouw.ncp_linear


def _executor(run: _Run | _CircuitRun | PPD.PearlCDeviceRun, w: int):
    if isinstance(run, _CircuitRun):
        op = ncp_linear_op()

        def call(layer: Any, x: Any, bias: Any = None) -> Any:
            return op(x, layer.weight, bias, run.entry, w)
        return call

    def apply(layer: Any, x: Any, bias: Any = None) -> Any:
        return run.gemm(w, x, layer.weight, bias)
    return apply


def _layers(scheme: str, target: Target) -> tuple[Any, list[Any], dict[int, tuple[int, int]]]:
    """The scheme and the executed sites, the linear layers by type in module order (a layer's ordinal is its weight id),
    with their (n, k); ValueError names every layer outside the scheme's domain, sharing a quant method or quantized."""
    entry = f"{NAME}:{scheme}"
    s = (None if scheme in pouw_circuit.SCHEMES else PPD.scheme_of(scheme) if scheme in PPD.SCHEMES
         else pouw_schemes.get(scheme))
    types = linear_types()
    layers = [m for m in target.model.modules() if isinstance(m, types)]
    if not layers:
        raise ValueError(f"{entry}: the model has no linear layer to execute")
    bad, seen, shapes = [], set(), {}
    for w, layer in enumerate(layers):
        params = [p for p in QUANT_PARAMS if getattr(layer, p, None) is not None]
        if params:
            bad.append(f"linear {w}: it carries {', '.join(params)}, a quantized layer's parameters: a quantized (e.g. FP8) "
                       "layer is refused, not misread")
            continue
        dtype = str(getattr(layer.weight, "dtype", None)).removeprefix("torch.")
        if dtype not in UNQUANTIZED:
            bad.append(f"linear {w}: its weight is {dtype}, and the executor reads an unquantized weight: a quantized (e.g. FP8) "
                       "layer is refused, not misread")
            continue
        n, k = layer.weight.shape
        shapes[w] = (n, k)
        if scheme in PPD.SCHEMES:
            bad += PPD.refusals(scheme, {w: (n, k)})
        try:
            if s is None:
                pouw_circuit.traced_as(scheme, 16, k, n)
            else:
                s.check(Matmul(0, Shape(16, k, n), w))
        except DomainError as e:
            bad.append(f"linear {w}: {e}")
        if id(layer.quant_method) in seen:
            bad.append(f"linear {w}: its quant method is shared with another layer")
        seen.add(id(layer.quant_method))
    if bad:
        raise ValueError(f"{entry} refused at install: " + "; ".join(bad))
    return s, layers, shapes


def commit_weights(scheme: str, target: Target) -> dict[str, Any]:
    """The composition's first phase, with the model untouched and no coin opened: every linear layer's int7 weight rows
    committed (`verity.core.protocols.pouw.audit.commit_rows`), as ``{"weight_roots": {id: hex}, "shapes": {id: [n, k]}}``.
    Under H1, the int8_exact arm's int8 weight rows as they are served, once a process (`pouw_h1_hooks.weight_roots`)."""
    if scheme == PH.SCHEME:
        layers = PHK.layers(target.model, linear_types())
        return {"weight_roots": {str(w): r.hex() for w, r in PHK.weight_roots(layers).items()},
                "shapes": {str(w): list(m.weight.shape) for w, m in enumerate(layers)}}
    _, layers, shapes = _layers(scheme, target)
    if scheme in PPD.SCHEMES:
        settings = Settings.of(target.context.settings_of(NAME))
        key = _SALT_KEYS[f"{NAME}:{scheme}"] = PPD.new_salt_key(scheme)
        roots = PPD.weight_roots(PPD.Kernel.load(settings.kernel, settings.cubin), {w: m.weight for w, m in enumerate(layers)}, scheme,
                                 salt_key=key)
        return {"weight_roots": {str(w): r.hex() for w, r in roots.items()}, "shapes": {str(w): list(v) for w, v in shapes.items()}}
    roots = {}
    for w, layer in enumerate(layers):
        b, _ = quantize(_host(layer.weight))
        roots[str(w)] = commit_rows(weight_binding(w), -2, [b[j].tobytes() for j in range(b.shape[0])]).commitment.root.hex()
    return {"weight_roots": roots, "shapes": {str(w): list(v) for w, v in shapes.items()}}


def weights_claim(scheme: Any, curated: bool = False, needs_gamma_fold: bool = False) -> dict[str, Any]:
    """What served FP8 (Pearl-C) claims of the weights it serves, the commitment's ``registration``.  It claims registered
    weights only when `SERVED_TRANSFORMS` pass the registered-weights rule (`registration.check_registration`, Pearl-C4's
    too); a keyed transform would take its key from the epoch's salt coin (`registration.rotation_key`).  Otherwise the
    claim says it is about nothing (`UNREGISTERED`) and names the refusal, and serving goes on: the salt is still the
    challenger's live coin (`Epoch.start`), so what the draw establishes of the work is unchanged."""
    out: dict[str, Any] = {"transforms": list(SERVED_TRANSFORMS), "curated": curated}
    try:
        scheme.check_registration(SERVED_TRANSFORMS, curated, needs_gamma_fold)
    except DomainError as e:
        return {**out, "registered": False, "claims": UNREGISTERED, "refused": str(e)}
    return {**out, "registered": True}


def challenged(settings: Mapping[str, Mapping[str, Any]], enabled: Any, epoch: int = 0) -> dict[str, dict[str, Any]]:
    """`settings` (the loaded protocols config) with PoUW's ``coins`` a challenger of `epoch` held in this process
    (`verity.core.protocols.pouw.coins.Challenger`, its coins the Lean verifier of record's, `pouw.verifier.coins`) when
    `enabled` (the protocol set) has PoUW, for a pipeline that is its run's own auditor (it replays and judges what it
    served); a config's own ``coins``, or its auditor host's ``challenger`` (the coins' one source), is kept alone."""
    out = {k: dict(v) for k, v in settings.items()}
    if any(str(e).partition(":")[0] == NAME for e in (enabled or ())):
        if not {"coins", "challenger"} & set(out.get(NAME, {})):
            out.setdefault(NAME, {})["coins"] = Challenger(epoch, verifier.coins)
    return out


def install(scheme: str, target: Target) -> Option:
    entry = f"{NAME}:{scheme}"
    settings = Settings.of(target.context.settings_of(NAME))
    if settings.coins is None and not (scheme in CHALLENGED and settings.challenger is not None):
        raise ValueError(f"{entry}: no 'coins' setting: the epoch's salt is its challenger's salt coin, opened after the "
                         "weights' root, and there is no beacon")
    if scheme == PH.SCHEME:
        return _install_h1(entry, settings, target)
    s, layers, shapes = _layers(scheme, target)
    committed = target.weight_commitments.get(entry) or commit_weights(scheme, target)
    roots = {int(w): bytes.fromhex(r) for w, r in committed["weight_roots"].items()}
    declared = Workload(tuple(Matmul(w, Shape(16, k, n), w) for w, (n, k) in shapes.items()))
    session, device, k = None, scheme in PPD.SCHEMES, None
    if settings.challenger is not None:
        session, commitment, opening, salt, carried, k = PHW.salt_session(entry, settings, weights_root(roots), device)
    else:
        commitment, opening = settings.coins.commitment, settings.coins.open("salt", weights_root(roots))
        salt = Epoch.start(declared, roots, commitment, opening).salt
    with PHW.closed_on_error(session):
        if device:
            run = PPD.PearlCDeviceRun(s, entry, PPD.Kernel.load(settings.kernel, settings.cubin),
                                      {w: m.weight for w, m in enumerate(layers)}, roots, commitment, opening,
                                      schedule=settings.schedule, salt_key=_SALT_KEYS.get(entry), pass_index=k)
            run.commits["registration"] = weights_claim(s)
        else:
            run = (_CircuitRun(scheme, entry, settings, roots, salt, commit=target.context.stage != "capture") if s is None
                   else _Run(s, entry, settings, roots, salt))
            run.commits.update(coins=commitment.to_json(), salt_opening=opening.to_json())
        if session is not None:
            run.commits.update(coin_source=PH.CHALLENGER, session=carried)
        if isinstance(run, _CircuitRun):
            _CIRCUIT_RUNS[entry] = (run, _sites(target.model, layers))
        h = target.hooks(entry)
        for w, layer in enumerate(layers):
            h.wrap(layer.quant_method, "apply", lambda orig, w=w: _executor(run, w))
        if device:
            PP.hook_steps(h, run, PPD.CALL_CAPACITY, settings.transcript_dir)
    return Option(name=entry, hooks=h, commits=run.commits, details=run.details, end_pass=run.end_pass if device else None,
                  release=run.finish if session is None else lambda: PHW.finish(run, session))


#: The live H1 runs by entry, for the Commit and its checks before the composition closes.
_H1_RUNS: dict[str, PH.H1Run] = {}


def h1_run(entry: str) -> PH.H1Run | None:
    return _H1_RUNS.get(entry)


def _install_h1(entry: str, settings: Settings, target: Target) -> Option:
    """H1 on the int8_exact arm: the epoch's salt `audit.salt_of` over the weights' root, as every scheme's, from the
    challenger's salt coin (the auditor host's opened after the window's open root, once round 0 is admitted:
    `pouw_h1.H1Run.open_window`), and each layer's ``observe`` the arm's own with its call recorded and handed to the run's listeners
    (`pouw_h1_hooks.observer`), in the static layout of the settings' ``rows`` and ``steps``; each layer's ``apply`` runs
    under the credit gate's trace (`pouw_h1_hooks.traced`), and the ``runner``'s ``execute_model`` hands each step's admitted
    requests and its batch order to the run (`pouw_h1_hooks.admitting`), each entry and each served row's request under its
    round's root before the ``rounds`` coin, and the window's draw coin opens at release after its manifest. With the
    setting ``challenger`` the salt, round and draw coins are the auditor host's, from the one session infra's gate
    reserved for the window (``session``, the gate's reserve answer) and ``gate`` opened on its window-open root (the
    open answer, which the commitment carries without its secret; `pouw_h1_window.Window`), closed once the window has
    finished; without it the coins are this process's and the window credits nothing (`pouw_h1.IN_PROCESS`). The
    window's weights' root is its period's (`pouw_h1_served.Served.weights_root`), else `audit.weights_root`, which
    registers no ``b/<K>`` and so credits nothing. ``admission`` is the run's (`pouw_h1_requests.ADMISSIONS`, round 0
    only by default). On an arm that serves z/y records (``zy_rows``, `engine.int8_vllm.Int8PlainLinearMethod`) each
    committed layer whose N is whole 32-column units serves through ``scaled_mm_zy`` over T slots, the records its z. The
    served values are the arm's, untouched."""
    layers = PHK.layers(target.model, linear_types())
    committed = target.weight_commitments.get(entry) or commit_weights(PH.SCHEME, target)
    roots = {int(w): bytes.fromhex(r) for w, r in committed["weight_roots"].items()}
    declared = Workload(tuple(Matmul(int(w), Shape(16, k, n), int(w)) for w, (n, k) in committed["shapes"].items()))
    if settings.rows is None or settings.steps is None:
        raise ValueError(f"{entry}: the static layout needs the pouw settings 'rows' (T) and 'steps' (S)")
    session, rounds, draw, salt, opener = None, settings.rounds, None, None, None
    shapes = [tuple(int(d) for d in m.weight.shape) for m in layers]
    modules = {w: f"linear {w}" for w in roots} | _named(target.model, layers)
    wroot = (settings.served and settings.served.weights_root(shapes, modules, settings.rows, settings.steps, layers,
                                                              target.context.stage != "capture")) or weights_root(roots)
    if settings.challenger is not None:
        session = PHW.Window(entry, settings.challenger, settings.steps, wroot, settings.session, settings.gate)

        def opener(requests_root: bytes) -> tuple[bytes, Any, Any, Mapping[str, Any]]:
            epoch_salt, *opened = session.open(requests_root)
            return (PH.program_salt(epoch_salt), *opened)
    else:
        commitment, opening = settings.coins.commitment, settings.coins.open("salt", weights_root(roots))
        salt = PH.program_salt(Epoch.start(declared, roots, commitment, opening).salt)
        draw = LiveRecord(rounds.challenger, Wd.DRAW) if rounds is not None else None
    source = PH.CHALLENGER if session is not None else "in-process"
    run = PH.H1Run(entry, salt, modules, shapes, settings.rows, settings.steps, retain=settings.retain_steps,
                   transcript_dir=settings.transcript_dir, commit=target.context.stage != "capture", rounds=rounds, draw=draw,
                   admission=settings.admission or RQ.ROUND0, coin_source=source, opener=opener, served=settings.served,
                   session=session.reserved["session"] if session is not None else None, graphs=settings.graphs)
    run.commits.update(weights_root=wroot.hex(), coin_source=source)
    if session is None:
        run.commits.update(coins=commitment.to_json(), salt_opening=opening.to_json())
        if rounds is not None:
            run.commits["round_coins"] = rounds.challenger.commitment.to_json()
    _H1_RUNS[entry] = run
    h = target.hooks(entry)
    PHK.wire(run, h, layers, settings.runner, settings.rows)
    return Option(name=entry, hooks=h, commits=run.commits, details=run.details, release=lambda: PHW.finish(run, session))


def _named(model: Any, layers: list[Any]) -> dict[int, str]:
    return {w: name for name, w in _sites(model, layers).items()} | PHK.expert_names(layers)
