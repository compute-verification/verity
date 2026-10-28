"""Pearl-C on the RTX PRO 6000 (sm_120) at the linears' ``quant_method.apply``: every call computed and committed on the device
by GPU 1's kernel pipeline, one epoch per install over every linear weight, one transcript per pass over every call.

The kernel is a directory (setting ``kernel``): the run tree's ``benchmarks/pouw/pearl_c_sm120``, or ``pearl_c_sm90`` for the
``h100`` record (an H100 or H200, sm_90a), whose ``run.py`` has the `Pipeline` (its kernels in order, each a named step of a
phase, over buffers the caller may own) and whose ``fixture.py`` has the keys (`keys_of`, `unit_block`).  The cubin is
build.sh's (setting ``cubin``, default the directory's cubin).  This module owns the buffers and runs the pipeline's steps on
torch's current stream, in the device's primary context:

* **The trees** are -h3's (`SCHEMES`): `audit.HM96`, frame-v3-sha512 over hm96-sha512 leaves (64-byte nodes), each leaf
  over a row's SHA-512 row digest, salted by the prover's hm96 salt key (the install's, `weight_roots`' and the run's
  ``salt_key``, which a retained manifest records for the reference verifier).
* **The weight side**, once per weight and epoch (`_Weight`).  B's FP32 words (the BF16 weight widened, exact) are hashed
  into B's row tree (phase ``epoch``), whose root must be the committed one.  Then, with the salt's seed_B, come E_B's
  and the F lines, B's statistics and forming, B̃·F_A, G and P_B (phase ``job``).  B's FP32 copy is dropped after, so the
  weight side keeps about a byte per weight (B̃) beside the model's own weight.
* **Each call** (`gemm`):
  - x's rows are widened into A's FP32 words, with zero filler rows up to m₁₆ (the domain's 16 | m; filler is committed as
    zero rows) and zero padding up to the kernels' tile;
  - ``commit``: A's pads under its domain, each row's SHA-512 row digest (``rowdig_a``: s512_rows, `pearl_c.row_digest`, the
    inner digest of A's hm96 leaves), A's row tree, seed_A and E_A's line key, and each real row's seed over its row digest
    with its line key (``seeds_a``: s512_seeds, `pearl_c.row_seed`);
  - ``forming``: E_A's lines (each real row's line 0 from its own seed), A's statistics and forming;
  - ``cleanup``: A′·F_B, P_A, and the chain with its peel and y (``gemm_pearlc`` writes y into the tensor this returns);
  - ``hashing``: the tile leaves (``sha512/row-seg/v1`` over C̃ and U), into the pass's leaf buffer at the call's offset.
    No per-call tile tree is built, since the pass has one.
  The row digests and the tile leaves run as hash_fast.cuh's segment and fold launches, over each shape's segment scratch
  (``segdig_a``, ``segdig_t``) and, for B's tree at install, a scratch dropped after it (``segdig_b``).  At decode (the
  shapes `run.pd_variant` takes) hash_decode.cuh's launches write the same bytes: ``commit_a`` (pd_commit) does
  ``commit`` with E_A's lines in one launch, over each shape's counter (`_Shape.count`), and ``hash_leaf`` is pd_tiles.
* **The records** (`pearl_c_work.UnitRecord`): each call's real rows and width, and the rows that fail the per-row rules
  (`clear_work.row_passes`).  The noise floor comes from the kernel's own ρ and α, whose product is exact as a double.
  The liveness is screened in FP32 with a 2⁻¹⁰ band (`screen`), and a row inside the band is decided exactly by
  `pearl_c.live_row` from its retained words.
* **Each pass** (`begin`, `end`; `end_pass` ends one and begins the next, which the PoUW option does at each engine step's
  end, `pouw_pass`, holding a step to what one pass takes): the transcript is every call's A root and one tile tree over
  every leaf, in call order, under `audit.transcript_domain` (the audit `Prover`'s layout).  With a retain directory, the
  pass also writes what the prover opens from: A's rows (BF16, exact) and tree levels per call, B's rows and tree levels per
  weight (linked to the first retained pass's), the leaves, the tile tree and the manifest (`write_manifest`) that the reference verifier reads
  (``benchmarks/pouw/pearl_c_vllm/verify_run.py``).  A retained pass starts from poisoned outputs (`poison`): a replay of an
  earlier pass's calls finds that pass's bytes still in the buffers, and they would be right for it, so a skipped write
  must not be able to pass as a repeat.  `omit` drops named steps, for the negative control.  A pass's state and its end
  are `ends`'s (`PassEnds`).

The per-call host work is kept small:
the keys by native SHA-512 (`_Keys`, byte for byte `call_keys`, the index numbered across the epoch: `pouw_pass.call_index`)
and their upload through a pinned ring (`_Uploads`); the records' screen as one fused kernel (`screen_into`, Triton, its
compiled kernel launched directly after the first launch; `screen` where Triton can't run); A's roots copied into a pass
buffer by one driver copy; filler rows zeroed only where an earlier call wrote them; the stream from torch's raw accessor.
The filler bookkeeping holds while no call has run under graphs (`eager_only`).

The schedule (`verity.core.protocols.pouw.serving.Schedule`) places each call's step groups on lanes (`pouw_pearl_c_lanes`):
``deferred`` hashes tiles on a side lane beside the next calls, ``deferred-a`` A's tree too where the kernel splits A's
commitment; the commitments are the serial schedule's.  With ``hashing`` off (``pearlc-nohash``) a call skips A's tree, E_A's
lines, the leaves and the records, for the time split.  ``hashing`` may change between passes, never within.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.core.protocols.pouw.audit import Workload, rows_domain
from verity.core.protocols.pouw.protocol import Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_c_work as PW  # noqa: F401 - kept importable here, as the tests read it
from verity.core.protocols.pouw.schemes.pearl_c_device import H100, SM120, SM120_UNPROMOTED
from verity.core.protocols.pouw.serving import (
    SERIAL,
    Schedule,
    SchemeId,
    hash_format,
)

from verity_vllm.protocols.pouw.pearl_c.passes import EpochCalls, call_index, pass_dir, room
from verity_vllm.protocols.pouw.pearl_c.graphs import GRAPH_ROWS, SLOT_BYTES, GraphCalls, _Graph
from verity_vllm.protocols.pouw.pearl_c.lanes import (  # noqa: F401 - the schedule's names, kept importable here, as callers read them
    SCHEDULES,
    SIDE_GROUPS,
    SIDE_IO,
    STEP_GROUPS,
    LaneCalls,
    admitted,
    rotated,
    schedule_of,
    splits,
    whole_options,
)
from verity_vllm.protocols.pouw.pearl_c.host import (
    SaltCoins,
    SaltOpening,
    _bytes,
    _constant_keys,
    _constants,
    _Driver,
    _Keys,
    _Lane,
    _put,
    _stream,
    _TorchStream,
    _Uploads,
    _write,  # noqa: F401 - kept importable here, as the tests read it
    _zeros,
    call_keys,  # noqa: F401 - kept importable here, as the tests read it
    epoch_of,
    memmove,
    write_manifest,  # noqa: F401 - kept importable here, as the tests read it
)
from verity_vllm.protocols.pouw.pearl_c.kernel import (
    Kernel,
    source_digest,  # noqa: F401 - kept importable here, as callers read it
)
from verity_vllm.protocols.pouw.pearl_c.manifest import (  # noqa: F401 - the manifest's readers, kept importable here, as callers read them
    manifest_epoch,
    manifest_schema,
    session_head,
)
from verity_vllm.protocols.pouw.pearl_c.ends import PassEnds, _Pass
from verity_vllm.protocols.pouw.pearl_c.screen import (
    excluded,
    screen,
    screen_into,
)

#: the Pearl-C schemes this device path runs, by name (`verity.core.protocols.pouw.serving.SchemeId`: forming, device record, hashing format);
#: the device record is GPU 1's pipeline variant too.  v1 is ``sm120`` (G = 4, the panel's ``pearl-c-sm120`` v1); v2 is
#: ``sm120-unpromoted`` (no promotion, the panel's v2, counted once the theory and Lean lanes restate it); ``h100`` (G = 4,
#: GLM-5.3's FP8 form on the H200) is pearl_c_sm90's.  The format is -h3, SHA-512 throughout (its seeds, lines,
#: tile leaves and HM96 trees: the one hash), each row's E_A line from its own seed (the work law capping each
#: credited row too); its tree hash comes from `verity.core.protocols.pouw.serving.HASH_FORMATS` through the scheme.
SCHEMES = tuple(C.PearlC(forming="v1", device=d, hashing="h3").name for d in (SM120, SM120_UNPROMOTED, H100))
#: each record's kernels' (k, n) grain past its domain: pearl_c_sm120's (1,024, 128); pearl_c_sm90's (256, 64), 1 KiB row segments, 64-wide tiles
KERNEL_GRAIN = {SM120.name: (1024, 128), SM120_UNPROMOTED.name: (1024, 128), H100.name: (256, 64)}
#: the domain's row granularity: m₁₆ = m rounded up to it, the filler rows zero
ROW_ALIGN = 16
#: tiles one pass's leaf buffer holds (256 MiB)
LEAF_CAPACITY = 1 << 23
#: calls one pass's A-root buffer holds, and rows its screen-flag buffer holds
CALL_CAPACITY = 1 << 16
FLAG_ROWS = 1 << 22
WEIGHT_BUFFERS = ("eb_codes", "fa_codes", "faT32", "fb_codes", "fbT32", "fbT128", "g", "alpha_b", "beta_b", "ia_b", "aux_b",
                  "epb_codes", "bt", "bf", "pb_t", "tree_b", "rowdig_b")
CALL_BUFFERS = ("alpha_a", "beta_a", "ia_a", "aux_a", "epa_codes", "ea_codes", "at", "af", "pa_t", "c", "u", "tree_a", "line_a",
                "seed_a", "rowdig_a", "seeds_a", "lkeys_a", "segdig_a", "segdig_t")
#: per-call buffers only some kernels' pipelines size (`Pipeline.sizes`), each made where its pipeline has it: pearl_c_sm90's
#: A'F_B windows
CALL_BUFFERS_OWN = ("af_win",)
EPOCH_STEPS = ("pad_b", "rowdig_b", "hash_rows_b", "tree_b")
JOB_STEPS = ("lines_eb", "lines_fa", "lines_fb", "stats_b", "eprime_b", "form_b", "bf", "g", "pb")
CALL_STEPS = ("pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "lines_ea", "stats_a", "eprime_a", "form_a",
              "af", "pa", "gemm_pearlc", "hash_leaf")
HASH_STEPS = frozenset({"pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "lines_ea", "hash_leaf", "commit_a",
                        "rows_a", "tree_seed_a"})


def scheme_of(name: str, **instance: Any) -> C.PearlC:
    """The scheme `name` names (`SCHEMES`); `instance` (``k_min``, ``tile``) makes a test instance."""
    from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES
    if name not in SCHEMES:
        raise KeyError(f"{name!r}: this device path runs {', '.join(SCHEMES)}")
    sid = SchemeId.parse(name)
    s = C.PearlC(forming=sid.forming, device=DEVICES[sid.device], hashing=sid.hashing, **instance)
    if s.name != name:
        raise ValueError(f"the scheme built for {name!r} names itself {s.name!r}")
    return s


def refusals(name: str, shapes: dict[int, tuple[int, int]]) -> list[str]:
    """Every linear outside the scheme's domain or its kernels' (`KERNEL_GRAIN`), by weight id."""
    s, out = scheme_of(name), []
    gk, gn = KERNEL_GRAIN[s.device.name]
    for w, (n, k) in sorted(shapes.items()):
        if k % gk or n % gn:
            out.append(f"linear {w}: (n, k) = ({n}, {k}), and the kernels take k a multiple of {gk:,} and n of {gn}")
            continue
        try:
            s.check(Matmul(0, Shape(ROW_ALIGN, k, n), w))
        except Exception as e:  # noqa: BLE001 - DomainError, named per layer
            out.append(f"linear {w}: {e}")
    return out


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
    sets: list = field(default_factory=list)   # the ring: set 0 b's own, then the schedule's further sets of `rotated` buffers
    done: list = field(default_factory=list)   # each set's last side-lane use (a CUDA event)
    turn: int = 0
    size: dict = field(default_factory=dict)   # the pipeline's buffer sizes at this shape
    count: Any = None              # pd_commit's counter (`run.Pipeline`'s pd_count): 16 zero bytes, never poisoned
    dirty: int = 0                 # xa's rows an eager call may have left nonzero (`PearlCDeviceRun.eager_only`)
    j: int = 0                     # the set the last call wrote, and the buffers it took from it (its schedule's `rotated`)
    rot: tuple = ()

    def buf(self, name: str) -> Any:
        """The last call's buffer `name` (``keys`` among them): its set's where it took it from one (set 0's keys are the
        shape's own), else the shape's own."""
        if name in self.rot and name in self.sets[self.j]:
            return self.sets[self.j][name]
        return self.keys if name == "keys" else self.b[name]


def weight_roots(kernel: Kernel, weights: dict[int, Any], scheme: str = SCHEMES[0], salt_key: bytes | None = None) -> dict[int, bytes]:
    """Each weight's B-row tree root (phase ``epoch`` alone), before any salt: the composition's weight phase, under the scheme
    `scheme` names: its device record is the pipeline variant, and its hashing format gives the tree hash.  The trees' hm96
    salts are the prover's, from `salt_key` (the install's, which its run is given too)."""
    import torch
    s = scheme_of(scheme)
    variant, hash = s.device.name, hash_format(s).tree_hash
    R, out = kernel.run, {}
    nb = R.NB
    for w, weight in weights.items():
        n, k = weight.shape
        size = R.Pipeline.sizes(R.BM, n, k, R.BM)
        b = {name: _zeros(size[name], weight.device) for name in ("tree_b", "faT32", "fbT32", "fbT128", "rowdig_b", "segdig_b")}
        keys, const = _zeros(128, weight.device), _constants(s, weight.device, salt_key)
        _put(keys[64:], rows_domain(_weight_binding(w), -2, n, hash).domain_id)
        xb = weight.to(torch.float32).contiguous()
        pipe = _pipeline(kernel, R.BM, n, k, {**{x: t.data_ptr() for x, t in b.items()}, "line_a": 0}, xb.data_ptr(),
                         {"line_b": keys.data_ptr(), "dom_b": keys.data_ptr() + 64, **_constant_keys(const)},
                         variant=variant, weight=w)
        _run(kernel, pipe, EPOCH_STEPS, _stream(weight.device))
        out[w] = _bytes(b["tree_b"][size["tree_b"] - nb:size["tree_b"]])
    return out


def new_salt_key(scheme: str) -> bytes:
    """A fresh prover hm96 salt key for `scheme`'s trees (their salts, kept by the prover and never in a record)."""
    import os
    scheme_of(scheme)
    return os.urandom(32)


def _weight_binding(w: int) -> bytes:
    from verity.core.protocols.pouw.audit import weight_binding
    return weight_binding(w)


def _pipeline(kernel: Kernel, m: int, n: int, k: int, bufs: dict[str, int], xb: int, keys: dict[str, int], xa: int = 0,
              m_real: int | None = None, *, variant: str, weight: int = 0, split_a: bool = False) -> Any:
    """The kernel's pipeline over the caller's buffers; with `split_a` (A's tree on a side lane) A's commitment split and
    the padding rows' E_A lines left unwritten (`run.Pipeline`'s ``fillers``)."""
    full = {"dom_a": 0, "dom_b": 0, "dom_t": 0, "line_b": 0, "unit": 0, "salt_key": 0, "hm": 0, **keys}
    return kernel.run.Pipeline(kernel.dev, m, n, k, variant, full, xa, xb, bufs=bufs, m_real=m_real, weight=weight,
                               **({"split_a": True, "fillers": False} if split_a else {}))


def _steps(pipe: Any, names: tuple[str, ...]) -> list[Any]:
    """The named steps in order.  A tree's pads exist only where the tree needs them."""
    by = {name: f for name, _, f in pipe.steps()}
    missing = [x for x in names if x not in by and not x.startswith("pad_")]
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


class PearlCDeviceRun(LaneCalls, GraphCalls, EpochCalls, PassEnds):
    """One install's Pearl-C workload on the device, with `pouw._Run`'s interface (`gemm`, `finish`, `commits`, `details`) and
    the passes the bench and the verifier need (`begin`, `end`, `end_pass`)."""

    def __init__(self, scheme: C.PearlC, entry: str, kernel: Kernel, weights: dict[int, Any], roots: dict[int, bytes],
                 coins: SaltCoins, opening: SaltOpening, hashing: bool = True, capacity: int = LEAF_CAPACITY,
                 time_gemm: bool = False, schedule: str | Schedule = SERIAL, graphs: bool = False, whole_defer: bool = False,
                 flag_rows: int = FLAG_ROWS, salt_key: bytes | None = None, whole_defer_a: bool = False,
                 whole_fork_stats: bool = False, lag_ends: bool = False, whole_tree_sm: bool = False, aside_ends: bool = False,
                 pass_index: int | None = None, whole_prefetch_b: bool = False, whole_eager: str | Schedule | None = None,
                 whole_shape_joins: bool = False):
        """`coins` is the epoch's coin commitment, held in this process or a live session's on the auditor host, and `opening`
        its salt coin, opened after the weights' root, or a live session's after salt epoch `pass_index`'s root (`epoch_of`,
        which raises `DomainError` otherwise): the salt is the coin's.
        The whole capture's options: `whole_options`.  `lag_ends`: an engine step's pass end (`_ExecutorPass.end`, `end_pass`
        with ``step``) leaves its commitment's readback in flight (`end` with ``lag``, `LaneCalls.settle`); with `aside_ends`
        too, its tile tree runs on a stream of its own over a copy of the leaves (`LaneCalls._end_aside`).  `whole_eager`
        is the schedule a forward a whole-step arm runs eagerly takes (`eager_forward`)."""
        whole_options(entry, kernel, whole_defer, whole_defer_a, whole_fork_stats, whole_tree_sm, lag_ends, aside_ends,
                      whole_prefetch_b, whole_shape_joins)
        self.scheme, self.entry, self.kernel, self.hashing, self.capacity = scheme, entry, kernel, hashing, capacity
        self.nb, self.salt_key, self.retain_dir, self.retained, self._weight_files = kernel.run.NB, salt_key, None, 0, {}
        self.graphs = graphs and next(iter(weights.values())).device.type == "cuda"
        self.whole_defer, self.whole_defer_a, self.whole_fork_stats = whole_defer, whole_defer_a, whole_fork_stats
        self.whole_tree_sm, self.whole_prefetch_b, self.whole_shape_joins = whole_tree_sm, whole_prefetch_b, whole_shape_joins
        self.lag_ends, self.aside_ends, self._lagged, self._end_side = lag_ends, aside_ends, None, None
        self._graphs: dict[tuple, _Graph] = {}
        self.variant = scheme.device.name
        self.time_gemm, self.events, self.omit = time_gemm, [], frozenset()
        self.torch_stream = _TorchStream()
        self.roots, self.coins, self.opening = dict(roots), coins, opening
        self.epoch = epoch_of(Workload(tuple(Matmul(w, Shape(ROW_ALIGN, t.shape[1], t.shape[0]), w) for w, t in weights.items())),
                              self.roots, coins, opening, pass_index)
        self.salt = salt = self.epoch.salt
        self.device = next(iter(weights.values())).device
        self.schedule = schedule_of(schedule, scheme, kernel)
        self.whole_eager = schedule_of(whole_eager, scheme, kernel) if whole_eager else None
        self._eager_from: Schedule | None = None   # the open pass's own schedule, while an eager forward runs it under whole_eager
        self.main: _Lane | None = None      # the main lane, when the schedule runs it above the caller's priority
        self.sides: list[_Lane] = []         # the side lanes, as many as the schedule's ring, which calls take in turn
        self.fork: _Lane | None = None      # the lane graphs' fork beside the main lane's per-row keying, made at first use
        self.driver = _Driver(kernel.dev.cu) if self.graphs else None   # a lane graph's replay's driver calls
        self._side_turn = 0
        self._side_cache: dict[tuple, frozenset[str]] = {}
        self._launching: Any = None          # the torch stream the current step group launches on (the GEMM's timing events)
        self.const = _constants(scheme, self.device, salt_key)
        self.const_keys = _constant_keys(self.const)
        if hasattr(kernel.run, "pd_setup"):   # it syncs, so here rather than at a call's first pipeline, which may be captured
            kernel.run.pd_setup(kernel.dev)
        self.leaves = _zeros(self.nb * capacity, self.device)
        self.roots_buf = _zeros(self.nb * CALL_CAPACITY, self.device)
        self.flag_buf = _zeros(2 * flag_rows, self.device).view(flag_rows, 2)
        self.tree_hash = hash_format(scheme).tree_hash
        self.keys, self.uploads = _Keys(kernel.fixture, salt, self.tree_hash), _Uploads(self.device)
        self.copy = kernel.dev.copy_async if self.device.type == "cuda" else memmove
        self.eager_only = True              # no call has run under graphs, whose replays write rows and keys unseen
        self.fused_screen = self.device.type == "cuda"
        self.weights = {w: self._weight_side(w, t) for w, t in weights.items()}
        if self.graphs:
            import torch
            self.order = {w: i for i, w in enumerate(sorted(self.weights))}
            self.key_slots = _zeros(len(self.order) * SLOT_BYTES, self.device)[:len(self.order) * SLOT_BYTES].view(-1, SLOT_BYTES)
            self.offs = torch.zeros((len(self.order), 3), dtype=torch.int64, device=self.device)
            self._host_keys = torch.zeros((len(self.order), SLOT_BYTES), dtype=torch.uint8, pin_memory=True)
            self._host_offs = torch.zeros((len(self.order), 3), dtype=torch.int64, pin_memory=True)
            self._slots_copied: Any = None     # the event after the last whole-forward slot upload
            self._predicted: dict[int, tuple] = {}
        self.shapes: dict[tuple[int, int, int], _Shape] = {}
        self.pipes: dict[tuple[int, int, frozenset, bool], tuple[Any, list[Any]]] = {}
        self.stream = lambda: _stream(self.device)
        self.commits: dict[str, Any] = {"scheme": scheme.name, "salt": salt.hex(), "weights_root": self.epoch.weights_root.hex(),
                                        "coins": coins.to_json(), "salt_opening": opening.to_json(),
                                        "transcript_root": None, "units": 0, "tiles": 0,
                                        **({} if pass_index is None else {"pass_index": pass_index})}
        self.details: dict[str, Any] = {
            "scheme": scheme.name, "committer": f"pouw_pearl_c_device (GPU 1's pipeline, cubin {kernel.cubin_sha256[:16]}, "
                                                f"sources {(kernel.source_sha256 or 'unrecorded')[:16]})",
            "hashing": hashing, "passes": [], "excluded_b": {str(w): len(x.excluded) for w, x in self.weights.items()},
            "host": {"keys": "native sha512", "screen": "fused (Triton)" if self.fused_screen else "torch"},
            "schedule": None,
            "graphs": (f"calls of at most {GRAPH_ROWS} rows: the serial schedule's one graph a call, a side lane's schedule's "
                       f"one a lane and ring set (A's tree its own where it is aside), replayed by the driver"
                       + (", a whole capture's tile hashing beside the next call" if whole_defer else "")
                       + (" and its A tree beside its forming and GEMM" if whole_defer_a else "")
                       + (" (the GEMM leaving the tree its SM)" if whole_tree_sm else "")
                       + (", its A scales beside its A commitment" if whole_fork_stats else "")
                       + (", its B̃ prefetched into L2 beside its A commitment" if whole_prefetch_b else "")
                       + (", its side steps joined before its shape's next call" if whole_shape_joins else "")
                       + (f", an eager forward's pass under {self.whole_eager.name}" if self.whole_eager else ""))
            if self.graphs else None,
            "pass_end": ("an engine step's commitment read back at the next end" + (", its tile tree on a stream of its own"
                                                                                     if aside_ends else "")) if lag_ends else "read back at its end",
            "uncovered": "every GEMM outside LinearBase (lm_head, attention); served y's glue (bias) is not bound"}
        self.cur: _Pass | None = None
        self.begin()

    # ---- the schedule -----------------------------------------------------------------------------------------------

    @property
    def defer(self) -> bool:
        """Whether the schedule runs a group off the main lane."""
        return bool(self.schedule.side)

    def _use(self, schedule: Schedule) -> None:
        """Run the next passes under `schedule`: its lanes made on first use, and what they run at recorded."""
        self.schedule = schedule
        if schedule.main_priority == "greatest" and self.main is None:
            self.main = _Lane(self.device, "greatest", 2)
        while schedule.side and len(self.sides) < schedule.ring:    # stand-ins 1, then 3, 4, … past the main lane's 2
            self.sides.append(_Lane(self.device, schedule.side_priority, 1 if not self.sides else 2 + len(self.sides)))
        self.details["schedule"] = {
            "name": schedule.name, "side": sorted(schedule.side), "ring": schedule.ring,
            "main": self.main.priority if schedule.main_priority == "greatest" else "the caller's",
            "side_lane": self.sides[0].priority if schedule.side else None}

    def join_sides(self, stream: Any = None) -> None:
        """`stream` (the current stream's torch stream when None) waits for every side lane's work so far."""
        if self.device.type != "cuda":
            return
        stream = self.torch_stream(self.stream()) if stream is None else stream
        for lane in self.sides:
            stream.wait_stream(lane.stream)

    def eager_forward(self) -> None:
        """The open pass's forward runs eagerly under a whole-step arm (`pouw_pearl_c_whole.WholeStep`: a batch its graphs
        lack, a prefill's or a mixed step's): with `whole_eager`, and before the pass's first call, the pass runs under that
        schedule, so its calls' tile hashing and A trees go on side lanes beside their GEMMs, and the next `begin` puts the
        pass's own schedule back.  The commitments are the pass's schedule's."""
        if self.whole_eager is None or self.cur is None or self.cur.matmuls or self._eager_from is not None:
            return
        self._eager_from = self.schedule
        self._use(self.whole_eager)

    # ---- the weight side --------------------------------------------------------------------------------------------

    def _weight_side(self, w: int, weight: Any) -> _Weight:
        import torch
        K, R = self.kernel, self.kernel.run
        n, k = weight.shape
        size = R.Pipeline.sizes(R.BM, n, k, R.BM)
        b = {name: _zeros(size[name], weight.device) for name in WEIGHT_BUFFERS}
        keys = _zeros(128, weight.device)
        mm = Matmul(0, Shape(ROW_ALIGN, k, n), w)
        seed_b = C.seed_b(self.salt, self.roots[w], self.scheme.hashing)
        kk = K.fixture.keys_of(self.scheme, self.epoch, mm, seed_b)
        _put(keys, kk["line_b"].ljust(64, b"\0") + kk["dom_b"])
        xb = weight.to(torch.float32).contiguous()
        seg = _zeros(size["segdig_b"], weight.device)
        pipe = _pipeline(K, R.BM, n, k, {**{x: t.data_ptr() for x, t in b.items()}, "line_a": 0, "segdig_b": seg.data_ptr()},
                         xb.data_ptr(), {"line_b": keys.data_ptr(), "dom_b": keys.data_ptr() + 64, **self.const_keys},
                         variant=self.variant, weight=w)
        stream = _stream(weight.device)
        _run(K, pipe, EPOCH_STEPS, stream)
        root = _bytes(b["tree_b"][size["tree_b"] - self.nb:size["tree_b"]])
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
            tile = R.BM if m16 > getattr(R, "MAINLOOP_ROWS", m16) else R.BM_D
            mp = R.cdiv(m16, tile) * tile
            size = R.Pipeline.sizes(mp, n, k, m16)
            own = tuple(x for x in CALL_BUFFERS_OWN if x in size)
            sh = _Shape(m16, mp, n, k, {x: _zeros(size[x], self.device) for x in (*CALL_BUFFERS, *own)},
                        torch.zeros((mp, k), dtype=torch.float32, device=self.device), _zeros(self.nb + 104, self.device))
            sh.sets = [dict(sh.b)]
            sh.size, sh.count = size, _zeros(16, self.device)
            if self.cur is not None and self.cur.poison is not None:
                for t in self._buffers(sh):
                    t.fill_(self.cur.poison)
            self.shapes[key] = sh
        return self._ring(self.shapes[key])

    def _ring(self, sh: _Shape) -> _Shape:
        """`sh` with the schedule's ring of per-call buffers: set 0 is its own (its keys too); each further set holds the
        buffers the side lane touches (`rotated`; ``keys``, dom_a ‖ unit, among them when A's tree moves), each made when a
        schedule first needs it, from poisoned bytes inside a poisoned pass."""
        names = rotated(self.schedule)
        while len(sh.sets) < self.schedule.ring:
            sh.sets.append({})
        for st in sh.sets[1:]:
            for x in names:
                if x not in st:
                    st[x] = _zeros(len(sh.keys) if x == "keys" else sh.size[x], self.device)
                    if self.cur is not None and self.cur.poison is not None:
                        st[x].fill_(self.cur.poison)
        sh.done += [None] * (len(sh.sets) - len(sh.done))
        return sh

    @staticmethod
    def _buffers(sh: _Shape) -> list[Any]:
        """Every buffer a call of `sh` writes: its own, and the ring's other sets."""
        return [*sh.b.values(), *(t for st in sh.sets[1:] for t in st.values())]

    def _call_names(self, pipe: Any) -> tuple[str, ...]:
        """`pipe`'s call step names: every call step but the hashing ones when hashing is off, and but the omitted ones, with
        each of the pipeline's fused steps (`run.Pipeline.FUSED`) in place of its parts, and dropped if any part is omitted."""
        names = [x for x in CALL_STEPS if (self.hashing or x not in HASH_STEPS) and x not in self.omit]
        have = {name for name, _, _ in pipe.steps()}
        for fused, parts in getattr(pipe, "FUSED", {}).items():
            if fused in have:
                at = next((i for i, x in enumerate(names) if x in parts), None)
                names = [x for x in names if x not in parts]
                if at is not None and not self.omit & set(parts):
                    names.insert(at, fused)
        return tuple(names)

    def _call_steps(self, pipe: Any) -> list[Any]:
        """The call's steps in order (`_call_names`)."""
        return _steps(pipe, self._call_names(pipe))

    def _pipe(self, w: int, sh: _Shape) -> tuple[Any, list[Any], list[Any]]:
        """The call's pipeline and its steps: the main lane's, and (with hashing) the side lane's, by the schedule's groups
        (`_side_groups`; A's commitment split where A's tree goes there, or where a deferred-a whole capture forks it,
        `_split_whole`).  The pipeline's ``lane_cuts`` are how many of the main lane's steps are its per-row keying, which A's
        tree reads, and how many of the side lane's are A's tree; its ``main_names`` name the main lane's steps."""
        side_groups = self._side_groups(sh)
        aside = "a_tree" in side_groups
        split = aside or self._split_whole(sh)
        key = (w, sh.m16, self.omit, self.hashing, self.schedule.name, self._graphed(sh.m16), split)
        if key not in self.pipes:
            wt, K = self.weights[w], self.kernel
            bufs = {**{x: t.data_ptr() for x, t in wt.b.items()}, **{x: t.data_ptr() for x, t in sh.b.items()}, "y": 0,
                    "leaves": 0, "pd_count": sh.count.data_ptr()}
            kp = self.key_slots[self.order[w]].data_ptr() if self._graphed(sh.m16) else sh.keys.data_ptr()
            keys = {"line_b": wt.keys.data_ptr(), "dom_b": wt.keys.data_ptr() + 64, "dom_a": kp, "unit": kp + self.nb,
                    **self.const_keys}
            pipe = _pipeline(K, sh.mp, sh.n, sh.k, bufs, 0, keys, sh.xa.data_ptr(), sh.m16, variant=self.variant,
                             split_a=split)
            names = self._call_names(pipe)
            present = [x for x in names if x in {n for n, _, _ in pipe.steps()}]
            steps = _steps(pipe, names)
            if self.time_gemm and self.device.type == "cuda":
                at = present.index("gemm_pearlc")
                steps[at] = self._timed(steps[at])
            lane = {x: "side" if STEP_GROUPS[x] in side_groups else "main" for x in present}
            main = [f for x, f in zip(present, steps, strict=True) if lane[x] == "main"]
            aside = [f for x, f in zip(present, steps, strict=True) if lane[x] == "side"]
            pipe.lane_cuts = tuple(len(list(itertools.takewhile(lambda x, g=g: STEP_GROUPS[x] == g, [x for x in present
                                                                                                     if lane[x] == ln])))
                                   for ln, g in (("main", "row_seeds"), ("side", "a_tree")))
            pipe.main_names = tuple(x for x in present if lane[x] == "main")
            self.pipes[key] = (pipe, main, aside)
        return self.pipes[key]

    def _timed(self, f):
        import torch

        def timed():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            stream = self._launching
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
        if torch.is_inference_mode_enabled():            # a vLLM forward: entering the mode again costs a call's microseconds
            return self._gemm(w, x, weight, bias)
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
        tiles = self.kernel.run.cdiv(m16, self.kernel.run.TILE) * (wt.n // self.kernel.run.TILE)
        cur = room(self, tiles, m, CALL_CAPACITY)
        index = call_index(self)
        self.join_before(sh)
        self.eager_only = self.eager_only and not self.graphs
        self._stage_a(sh, x2, m, sh.dirty if self.eager_only else m16)
        sh.dirty = m
        pipe, steps, side = self._pipe(w, sh)
        raw = self.stream()
        caller = self.torch_stream(raw) if self.device.type == "cuda" else None
        if self._graphed(m16):
            sh.j, sh.rot = 0, ()
            y, screened = self._graph_call(w, sh, pipe, raw, tiles, m, index, caller)
            return self._after(cur, w, wt, sh, pipe, index, m, k, m16, tiles, y, lead, bias, screened, scattered=self.hashing)
        aside = bool(pipe.lane_cuts[1])                 # A's tree is among the side steps (omits can drop it)
        if self._lane_graphed(sh, side):
            g, screened = self._lanes(w, sh, pipe, steps, side, raw, caller, index, aside, graph=(m, tiles))
            return self._after(cur, w, wt, sh, pipe, index, m, k, m16, tiles, g.y.clone(), lead, bias, screened, scattered=True)
        y = torch.empty((sh.mp, wt.n), dtype=torch.bfloat16, device=x.device)
        pipe.b["y"], pipe.b["leaves"] = y.data_ptr(), self.leaves.data_ptr() + self.nb * cur.tiles
        self._lanes(w, sh, pipe, steps, side, raw, caller, index, aside)
        return self._after(cur, w, wt, sh, pipe, index, m, k, m16, tiles, y, lead, bias,
                           rows=x2 if x2.is_contiguous() and x2.dtype == torch.bfloat16 else None, scattered=aside)

    def _after(self, cur: _Pass, w: int, wt: _Weight, sh: _Shape, pipe: Any, index: int, m: int, k: int, m16: int, tiles: int,
               y: Any, lead: Any, bias: Any, screened: bool = False, rows: Any = None, scattered: bool = False) -> Any:
        """A call's records after its steps: its matmul, A's root, the screen (unless a graph `screened` it), the retained
        files, and y.  The screen reads `rows` (the call's BF16 x, whose widening to A's FP32 words is exact) when given, at
        half the bytes of A's words.  A graph that `scattered` wrote A's root, the leaves and (when it screened) the flags."""
        cur.matmuls.append(Matmul(index, Shape(m16, k, wt.n), w))
        if self.hashing:
            size = pipe.size["tree_a"]
            if not scattered:
                self.copy(self.roots_buf.data_ptr() + self.nb * (index - cur.base), sh.buf("tree_a").data_ptr() + size - self.nb,
                          self.nb, self.stream())
            flags = self.flag_buf[cur.flag_rows:cur.flag_rows + m]
            if not screened:
                screen_into(flags, sh.xa[:m] if rows is None else rows, sh.b["alpha_a"], sh.b["aux_a"], self.fused_screen)
            cur.flag_rows += m
            if cur.retain is not None:
                if "a_tree" in self._side_groups(sh) and sh.done[sh.j] is not None:   # A's tree is the call's side lane's
                    self.torch_stream(self.stream()).wait_event(sh.done[sh.j])
                self._retained(cur, cur.matmuls[-1], m, flags, *self._kept(sh, pipe))
            cur.tiles += tiles
        out = y[:m] if bias is None else y[:m] + bias
        return out.reshape(*lead, wt.n)

    # ---- passes -----------------------------------------------------------------------------------------------------

    def begin(self, retain: str | None = None, run: str | None = None, schedule: str | Schedule | None = None,
              stepped: bool = False, at: int | None = None) -> None:
        """A new pass: its own transcript under the install's epoch, its indices from `pass_base` (``at``), under `schedule` (the
        last pass's own when None; a schedule changes only between passes, and before a pass's first call by `eager_forward`).
        With `retain`, the pass writes what the prover opens from into that directory, from poisoned outputs, and its manifest
        names `run`.  `stepped`: a step's end begins it."""
        self.join_sides()                   # the last pass's side lanes are done before anything new
        retain = retain or pass_dir(self)
        last, self._eager_from = self._eager_from or self.schedule, None
        self._use(last if schedule is None else schedule_of(schedule, self.scheme, self.kernel))
        self.cur = _Pass(Path(retain) if retain else None, run, stepped=stepped, base=self.pass_base(at))
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
            for t in self._graph_buffers():
                t.fill_(value)


class PearlCExecutor:
    """A `PearlCDeviceRun` behind `verity_vllm.protocols.linear.api.LinearExecutor`, the one object a `Stack` computes a site's y with.
    Its weight state is B̃, formed once per install, so it can't sit behind a source that forbids weight state (POUS); it reads
    BF16 weights; its controls are the poison byte and omitted step groups (by step name, as `PearlCDeviceRun.omit` takes
    them); its schedules are those `schedule_of` admits for its scheme."""

    keeps_weight_state = True
    accepts = frozenset({"bf16"})
    controls = frozenset({"poison", "omit"})

    def __init__(self, run: PearlCDeviceRun) -> None:
        self.run, self.name, self.schedules = run, run.entry, admitted(run.scheme, run.kernel)

    def refusals(self, sites: Any) -> list[str]:
        return refusals(self.run.scheme.name, {s.weight_id: (s.n, s.k) for s in sites})

    def begin(self, spec: Any) -> _ExecutorPass:
        return _ExecutorPass(self.run, spec)

    def close(self) -> None:
        self.run.finish()


class _ExecutorPass:
    """One `PassSpec` on the run: its hashing, schedule and retain directory, a poison byte and omitted steps for its length."""

    def __init__(self, run: PearlCDeviceRun, spec: Any) -> None:
        self.run, self.spec = run, spec
        run.hashing = spec.hashing
        run.begin(spec.retain, spec.run_id, schedule=spec.schedule)
        run.omit = frozenset(x for c in spec.controls if c.kind == "omit" for x in c.arg)
        for c in spec.controls:
            if c.kind == "poison":
                run.poison(0xA5 if c.arg is None else c.arg)

    def call(self, site: Any, x: Any, weight: Any, bias: Any) -> Any:
        return self.run.gemm(site.weight_id, x, weight, bias)

    def end(self) -> dict[str, Any]:
        try:
            return self.run.end(lag=self.run.lag_ends)
        finally:
            self.run.omit = frozenset()
