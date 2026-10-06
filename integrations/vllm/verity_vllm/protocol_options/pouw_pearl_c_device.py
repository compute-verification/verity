"""Pearl-C on the RTX PRO 6000 (sm_120) at the linears' ``quant_method.apply``: every call computed and committed on the device
by GPU 1's kernel pipeline, one epoch per install over every linear weight, one transcript per pass over every call.

The kernel is a directory (setting ``kernel``): the run tree's ``benchmarks/pouw/pearl_c_sm120``, whose ``run.py`` has the
`Pipeline` (its kernels in order, each a named step of a phase, over buffers the caller may own) and whose ``fixture.py`` has
the keys (`keys_of`, `unit_block`).  The cubin is build.sh's (setting ``cubin``, default the directory's
cubin).  This module owns the buffers and runs the pipeline's steps on torch's current stream, in the
device's primary context:

* **The trees** are the scheme's hashing format's (`SCHEMES`, the pipeline's ``hashing``): frame-b3 under -h1, frame-b3s
  under -h2, whose row leaves are trees over 256-byte segments (the segment trees' level keys derived in the leaf kernel).
* **The weight side**, once per weight and epoch (`_Weight`).  B's FP32 words (the BF16 weight widened, exact) are hashed
  into B's row tree (phase ``epoch``), whose root must be the committed one.  Then, with the salt's seed_B, come E_B's
  and the F lines, B's statistics and forming, B̃·F_A, G and P_B (phase ``job``).  B's FP32 copy is dropped after, so the
  weight side keeps about a byte per weight (B̃) beside the model's own weight.
* **Each call** (`gemm`):
  - x's rows are widened into A's FP32 words, with zero filler rows up to m₁₆ (the domain's 16 | m; filler is committed as
    zero rows) and zero padding up to the kernels' tile;
  - ``commit``: A's pads and level keys (and -h2's segment level keys) under its domain, A's row tree, seed_A and E_A's
    line key;
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

The schedule (`verity.protocols.accounting.work.pouw.serving.Schedule`, by name from `SCHEDULES`; the decode plan's step 2, bc-b139c29c's) places each
call's step groups (`STEP_GROUPS`) on lanes.  Under ``deferred``, the measured one, the call's tile hashing (`SIDE_IO`: the
message digests and the tile leaves) runs on a side lane at the least priority behind an event after the GEMM, while the next
calls go on, and the rest of the call runs on a main lane of the run's own at the greatest priority, joined to the caller's
stream by events on both sides (``deferred-caller`` leaves it on the caller's stream, #564's first cut).  A's commitment stays
on the main lane, since forming reads seed_A.  Each shape rotates `Schedule.ring` sets of the buffers the side lane touches
(`rotated`), and a call reuses a set only once the side lane has finished with it; the pass's end, and the next pass's begin,
wait for the side lane.  The same kernels read and write the same bytes, so the commitments are the serial schedule's.  A
schedule changes only between passes (`begin`), and one whose placement this path or the scheme's format can't run is refused
(`schedule_of`).

With ``hashing`` off (the bench's ``pearlc-nohash``) the call skips A's tree, E_A's lines, the digests, the leaves and the
records.  What is left is the same forming and GEMM without hashing, for the time split.  ``hashing`` may change between
passes, never within one.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.protocols.accounting.work.pouw.audit import Epoch, Workload, rows_domain, transcript_domain
from verity.protocols.accounting.work.pouw.coins import CoinCommitment, CoinOpening
from verity.protocols.accounting.work.pouw.protocol import DomainError, Matmul, Shape
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C
from verity.protocols.accounting.work.pouw.schemes import pearl_c_work as PW
from verity.protocols.accounting.work.pouw.schemes.pearl_c_device import SM120, SM120_UNPROMOTED
from verity.protocols.accounting.work.pouw.serving import (
    SCHEDULES,
    SERIAL,
    Schedule,
    SchemeId,
    StepIO,
    hash_format,
    schedule_refusals,
)

from .pouw_pearl_c_graphs import GRAPH_ROWS, SLOT_BYTES, GraphCalls, _Graph
from .pouw_pearl_c_host import (
    _bytes,
    _Keys,
    _Lane,
    _put,
    _stream,
    _TorchStream,
    _Uploads,
    _write,
    _zeros,
    call_keys,  # noqa: F401 - kept importable here, as the tests read it
    write_manifest,
)
from .pouw_pearl_c_screen import (
    excluded,
    screen,
    screen_into,
)

#: the Pearl-C schemes this device path runs, by name (`verity.protocols.accounting.work.pouw.serving.SchemeId`: forming, device record, hashing format);
#: the device record is GPU 1's pipeline variant too.  v1 is ``sm120`` (G = 4, the panel's ``pearl-c-sm120`` v1); v2 is
#: ``sm120-unpromoted`` (no promotion, the panel's v2, counted once the theory and Lean lanes restate it).  The format is a
#: switch: -h1 commits with frame-b3 trees, -h2 with frame-b3s (GPU 1's pipeline, `run.Pipeline`'s ``hashing``); which one
#: serves is Daniel's call.  Its keys and tree hash come from `verity.protocols.accounting.work.pouw.serving.HASH_FORMATS`, the one table, through the
#: scheme.
SCHEMES = tuple(C.PearlC(forming="v1", device=d, hashing=h).name for d, h in ((SM120, "h1"), (SM120_UNPROMOTED, "h1"), (SM120, "h2")))
#: the domain's row granularity: m₁₆ = m rounded up to it, the filler rows zero
ROW_ALIGN = 16
#: tiles one pass's leaf buffer holds (256 MiB)
LEAF_CAPACITY = 1 << 23
#: calls one pass's A-root buffer holds, and rows its screen-flag buffer holds
CALL_CAPACITY = 1 << 16
FLAG_ROWS = 1 << 22
MANIFEST_SCHEMA = "verity-vllm/pouw-pearl-c-run/v2"
#: a pass recorded before epoch coins, salted from a beacon: only a replay of a recorded pass reads it (`manifest_epoch`)
MANIFEST_SCHEMA_RECORDED = "verity-vllm/pouw-pearl-c-run/v1"
MANIFEST_SCHEMAS = (MANIFEST_SCHEMA, MANIFEST_SCHEMA_RECORDED)
WEIGHT_BUFFERS = ("eb_codes", "fa_codes", "faT32", "fb_codes", "fbT32", "fbT128", "g", "alpha_b", "beta_b", "ia_b", "aux_b",
                  "epb_codes", "bt", "bf", "pb_t", "tree_b", "keys_b", "skeys_b")
CALL_BUFFERS = ("alpha_a", "beta_a", "ia_a", "aux_a", "epa_codes", "ea_codes", "at", "af", "pa_t", "c", "u", "digests",
                "tree_a", "keys_a", "skeys_a", "line_a", "seed_a")
EPOCH_STEPS = ("pad_b", "node_keys_b", "hash_rows_b", "tree_b")
JOB_STEPS = ("lines_eb", "lines_fa", "lines_fb", "stats_b", "eprime_b", "form_b", "bf", "g", "pb")
CALL_STEPS = ("pad_a", "node_keys_a", "hash_rows_a", "tree_a", "seed_a", "lines_ea", "stats_a", "eprime_a", "form_a", "af", "pa",
              "gemm_pearlc", "hash_msg", "hash_leaf")
HASH_STEPS = frozenset({"pad_a", "node_keys_a", "hash_rows_a", "tree_a", "seed_a", "lines_ea", "hash_msg", "hash_leaf"})
#: each call step's group (`verity.protocols.accounting.work.pouw.serving.GROUPS`), which a schedule places on a lane
STEP_GROUPS = {"pad_a": "a_tree", "node_keys_a": "a_tree", "hash_rows_a": "a_tree", "tree_a": "a_tree", "seed_a": "a_tree",
               "lines_ea": "forming", "stats_a": "forming", "eprime_a": "forming", "form_a": "forming",
               "af": "cleanup", "pa": "cleanup", "gemm_pearlc": "cleanup", "hash_msg": "tile_hash", "hash_leaf": "tile_hash"}
#: the steps this path can run on a side lane, with the per-call buffers they read and write (the constant keys and the pass's
#: leaf buffer aside, which no later call rewrites): each shape rotates these, so a call never overwrites what the side lane
#: still reads.  Only the tile hashing moves: A's tree stays on the main lane under -h1, and the screen runs on the caller.
SIDE_IO = {"hash_msg": StepIO("tile_hash", frozenset({"c", "u"}), frozenset({"digests"})),
           "hash_leaf": StepIO("tile_hash", frozenset({"digests"}), frozenset())}
SIDE_GROUPS = frozenset(io.group for io in SIDE_IO.values())


def rotated(schedule: Schedule) -> tuple[str, ...]:
    """The per-call buffers each shape rotates under `schedule`: those its side-lane steps read or write."""
    return tuple(sorted({b for io in SIDE_IO.values() if io.group in schedule.side for b in io.reads | io.writes}))


def schedule_of(name: str | Schedule, scheme: C.PearlC) -> Schedule:
    """`name`'s schedule (`verity.protocols.accounting.work.pouw.serving.SCHEDULES`), refused where the scheme's format or this path can't run it."""
    s = SCHEDULES[name] if isinstance(name, str) else name
    bad = schedule_refusals(hash_format(scheme), s)
    if set(s.side) - SIDE_GROUPS:
        bad.append(f"{s.name}: this path runs only {', '.join(sorted(SIDE_GROUPS))} on a side lane, not "
                   f"{', '.join(sorted(set(s.side) - SIDE_GROUPS))}")
    if bad:
        raise ValueError(f"{scheme.name}: " + "; ".join(bad))
    return s


def scheme_of(name: str, **instance: Any) -> C.PearlC:
    """The scheme `name` names (`SCHEMES`); `instance` (``k_min``, ``tile``) makes a test instance."""
    from verity.protocols.accounting.work.pouw.schemes.pearl_c_device import DEVICES
    if name not in SCHEMES:
        raise KeyError(f"{name!r}: this device path runs {', '.join(SCHEMES)}")
    sid = SchemeId.parse(name)
    s = C.PearlC(forming=sid.forming, device=DEVICES[sid.device], hashing=sid.hashing, **instance)
    if s.name != name:
        raise ValueError(f"the scheme built for {name!r} names itself {s.name!r}")
    return s


def manifest_schema(man: dict, recorded: bool = False) -> str:
    """`man`'s schema, if a verifier reads it: `MANIFEST_SCHEMA`, or `MANIFEST_SCHEMA_RECORDED` only as a replay of a
    recorded pass (`recorded`).  Raises `DomainError` otherwise."""
    schema = man.get("schema")
    if schema == MANIFEST_SCHEMA_RECORDED and not recorded:
        raise DomainError(f"the pass's manifest is {schema}, recorded before epoch coins: only a replay of a recorded pass "
                          "reads it")
    if schema not in MANIFEST_SCHEMAS:
        raise DomainError(f"schema {schema!r}, not {MANIFEST_SCHEMA}" + (f" or {MANIFEST_SCHEMA_RECORDED}" if recorded else ""))
    return schema


def manifest_epoch(man: dict, coins: CoinCommitment | None, recorded: bool = False) -> tuple[Epoch, str]:
    """A retained pass's epoch, as its verifier re-starts it, and how its salt was had.  A `MANIFEST_SCHEMA` pass carries
    the epoch's coin commitment and its salt coin's opening: the commitment must be `coins`, the one the challenger published
    (the verifier's copy, not the pass's), and the opening the salt coin's against it, opened after every weight's root
    (`Epoch.start`).  A pass recorded before epoch coins (`MANIFEST_SCHEMA_RECORDED`) carries a beacon instead, which only a
    replay of a recorded pass reads (`recorded`: `Epoch.unverified`, a salt that claims nothing).  Either way the pass's salt
    must be the epoch's.  Raises `DomainError` naming what doesn't hold."""
    schema = manifest_schema(man, recorded)
    matmuls = tuple(Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
    roots = {int(w): bytes.fromhex(v["root"]) for w, v in man["weights"].items()}
    if schema == MANIFEST_SCHEMA:
        if "salt_opening" not in man:
            raise DomainError("the pass has no salt coin")
        if coins is None:
            raise DomainError("the pass's salt coin opens against the epoch's published coin commitment, which the verifier holds")
        try:
            opening = CoinOpening.from_json(man["salt_opening"])
            claimed = CoinCommitment.from_json(man["coins"])
        except (KeyError, TypeError, ValueError) as e:
            raise DomainError(f"the pass's coins are malformed: {e}") from None
        if claimed != coins:
            raise DomainError("the pass's coin commitment is not the epoch's")
        epoch, how = Epoch.start(Workload(matmuls), roots, coins, opening), "coin"
    elif "beacon" in man:
        epoch, how = Epoch.unverified(Workload(matmuls), roots, bytes.fromhex(man["beacon"])), "recorded beacon"
    else:
        raise DomainError("the recorded pass has no beacon")
    if epoch.salt.hex() != man.get("salt"):
        raise DomainError(f"the pass's salt is not its {how}'s and the weights'")
    return epoch, how


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


def _the_cubin(directory: Path) -> Path:
    """A kernel directory's one cubin, when the caller names none."""
    found = sorted(directory.glob("*.cubin"))
    if len(found) != 1:
        raise ValueError(f"{directory}: {len(found)} cubins, not one: name the cubin")
    return found[0]


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
        run, run_sha = _load("pearl_c_run", d / "run.py")
        fixture, fixture_sha = _load("pearl_c_fixture", d / "fixture.py")
        raw = Path(cubin or _the_cubin(d)).read_bytes()
        key = (source_digest(run_sha, fixture_sha), hashlib.sha256(raw).hexdigest())
        if key not in _KERNELS:
            dev = run.Dev(raw)
            dev.open()
            _KERNELS[key] = cls(run, fixture, dev, key[1], key[0])
        return _KERNELS[key]


#: (source_sha256, cubin_sha256) -> the loaded kernel
_KERNELS: dict[tuple[str, str], Kernel] = {}


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


def weight_roots(kernel: Kernel, weights: dict[int, Any], scheme: str = SCHEMES[0]) -> dict[int, bytes]:
    """Each weight's B-row tree root (phase ``epoch`` alone), before any salt: the composition's weight phase, under the scheme
    `scheme` names: its device record is the pipeline variant, and its hashing format gives the tree hash and the trees'
    frame."""
    import torch
    s = scheme_of(scheme)
    variant, hash = s.device.name, hash_format(s).tree_hash
    R, out = kernel.run, {}
    for w, weight in weights.items():
        n, k = weight.shape
        size = R.Pipeline.sizes(R.BM, n, k, R.BM)
        b = {name: _zeros(size[name], weight.device) for name in ("tree_b", "keys_b", "skeys_b", "faT32", "fbT32", "fbT128")}
        keys, const = _zeros(64, weight.device), _constants(kernel, s, weight.device)
        _put(keys[32:], rows_domain(_weight_binding(w), -2, n, hash).domain_id)
        xb = weight.to(torch.float32).contiguous()
        pipe = _pipeline(kernel, R.BM, n, k, {**{x: t.data_ptr() for x, t in b.items()}, "line_a": 0}, xb.data_ptr(),
                         {"line_b": keys.data_ptr(), "dom_b": keys.data_ptr() + 32, **_constant_keys(const)}, variant=variant,
                         hashing=s.hashing)
        _run(kernel, pipe, EPOCH_STEPS, _stream(weight.device))
        out[w] = _bytes(b["tree_b"][size["tree_b"] - 32:size["tree_b"]])
    return out


def _weight_binding(w: int) -> bytes:
    from verity.protocols.accounting.work.pouw.audit import weight_binding
    return weight_binding(w)


def _constants(kernel: Kernel, scheme: C.PearlC, device: Any) -> Any:
    """The pipeline's constant keys on the device: the scheme's digest keys (message, leaf) and frame-b3s's segment-key
    templates (row, tile)."""
    keys = scheme.digest_keys
    if keys is None:
        raise ValueError(f"{scheme.name}: the device path's digests are keyed BLAKE3, and hashing {scheme.hashing} has no keys")
    R, t = kernel.run, _zeros(192, device)
    _put(t, keys["message"] + keys["leaf"] + R.b3s_template(R.ROW_SCHEMA) + R.b3s_template(R.TILE_SCHEMA))
    return t


def _constant_keys(t: Any) -> dict[str, int]:
    p = t.data_ptr()
    return {"k_msg": p, "k_leaf": p + 32, "t_row": p + 64, "t_tile": p + 128}


def _pipeline(kernel: Kernel, m: int, n: int, k: int, bufs: dict[str, int], xb: int, keys: dict[str, int], xa: int = 0,
              m_real: int | None = None, *, variant: str, hashing: str, rows_form: str = "w") -> Any:
    full = {"dom_a": 0, "dom_b": 0, "dom_t": 0, "line_b": 0, "k_msg": 0, "k_leaf": 0, "t_row": 0, "t_tile": 0, "unit": 0, **keys}
    return kernel.run.Pipeline(kernel.dev, m, n, k, variant, full, xa, xb, bufs=bufs, m_real=m_real, hashing=hashing,
                               rows_form=rows_form)


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


class PearlCDeviceRun(GraphCalls):
    """One install's Pearl-C workload on the device, with `pouw._Run`'s interface (`gemm`, `finish`, `commits`, `details`) and
    the passes the bench and the verifier need (`begin`, `end`)."""

    def __init__(self, scheme: C.PearlC, entry: str, kernel: Kernel, weights: dict[int, Any], roots: dict[int, bytes],
                 coins: CoinCommitment, opening: CoinOpening, hashing: bool = True, capacity: int = LEAF_CAPACITY,
                 time_gemm: bool = False, schedule: str | Schedule = SERIAL, graphs: bool = False, rows_form: str = "w",
                 whole_defer: bool = False):
        """`coins` is the challenger's coin commitment of the epoch and `opening` its salt coin, opened after the weights'
        root (`Epoch.start`, which raises `DomainError` for any other coin): the epoch's salt is the coin's."""
        self.scheme, self.entry, self.kernel, self.hashing, self.capacity = scheme, entry, kernel, hashing, capacity
        self.rows_form = rows_form
        self.graphs = graphs and next(iter(weights.values())).device.type == "cuda"
        self.whole_defer = whole_defer
        self._graphs: dict[tuple, _Graph] = {}
        self.variant = scheme.device.name
        self.time_gemm, self.events, self.omit = time_gemm, [], frozenset()
        self.torch_stream = _TorchStream()
        self.roots, self.coins, self.opening = dict(roots), coins, opening
        self.epoch = Epoch.start(Workload(tuple(Matmul(w, Shape(ROW_ALIGN, t.shape[1], t.shape[0]), w) for w, t in weights.items())),
                                 self.roots, coins, opening)
        self.salt = salt = self.epoch.salt
        self.device = next(iter(weights.values())).device
        self.schedule = schedule_of(schedule, scheme)
        self.main: _Lane | None = None      # the main lane, when the schedule runs it above the caller's priority
        self.side: _Lane | None = None      # the side lane, when the schedule has one
        self._launching: Any = None          # the torch stream the current step group launches on (the GEMM's timing events)
        self.const = _constants(kernel, scheme, self.device)
        self.leaves = _zeros(32 * capacity, self.device)
        self.roots_buf = _zeros(32 * CALL_CAPACITY, self.device)
        self.flag_buf = _zeros(2 * FLAG_ROWS, self.device).view(FLAG_ROWS, 2)
        self.tree_hash = hash_format(scheme).tree_hash
        self.keys, self.uploads = _Keys(kernel.fixture, salt, self.tree_hash), _Uploads(self.device)
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
                                        "transcript_root": None, "units": 0, "tiles": 0}
        self.details: dict[str, Any] = {
            "scheme": scheme.name, "committer": f"pouw_pearl_c_device (GPU 1's pipeline, cubin {kernel.cubin_sha256[:16]}, "
                                                f"sources {(kernel.source_sha256 or 'unrecorded')[:16]})",
            "hashing": hashing, "passes": [], "excluded_b": {str(w): len(x.excluded) for w, x in self.weights.items()},
            "host": {"keys": "native blake3",
                     "screen": "fused (Triton)" if self.fused_screen else "torch"},
            "schedule": None,
            "graphs": (f"calls of at most {GRAPH_ROWS} rows, serial schedule"
                       + (", a whole capture's tile hashing beside the next call" if whole_defer else "")) if self.graphs else None,
            "uncovered": "every GEMM outside LinearBase (lm_head, attention); served y's glue (bias) is not bound (red team F4)"}
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
        if schedule.side and self.side is None:
            self.side = _Lane(self.device, schedule.side_priority, 1)
        self.details["schedule"] = {
            "name": schedule.name, "side": sorted(schedule.side), "ring": schedule.ring,
            "main": self.main.priority if schedule.main_priority == "greatest" else "the caller's",
            "side_lane": self.side.priority if schedule.side else None}

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
                         {"line_b": keys.data_ptr(), "dom_b": keys.data_ptr() + 32, **_constant_keys(self.const)},
                         variant=self.variant, hashing=self.scheme.hashing)
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
            sh.sets = [dict(sh.b)]
            sh.size = size
            if self.cur is not None and self.cur.poison is not None:
                for t in self._buffers(sh):
                    t.fill_(self.cur.poison)
            self.shapes[key] = sh
        return self._ring(self.shapes[key])

    def _ring(self, sh: _Shape) -> _Shape:
        """`sh` with the schedule's ring of per-call buffers: set 0 is its own; each further set holds the buffers the side lane
        touches (`rotated`), made when a schedule first needs them, from poisoned bytes inside a poisoned pass."""
        if len(sh.sets) >= self.schedule.ring and len(sh.done) == len(sh.sets):
            return sh
        names = rotated(self.schedule)
        while len(sh.sets) < self.schedule.ring:
            extra = {x: _zeros(sh.size[x], self.device) for x in names}
            if self.cur is not None and self.cur.poison is not None:
                for t in extra.values():
                    t.fill_(self.cur.poison)
            sh.sets.append(extra)
        sh.done += [None] * (len(sh.sets) - len(sh.done))
        return sh

    @staticmethod
    def _buffers(sh: _Shape) -> list[Any]:
        """Every buffer a call of `sh` writes: its own, and the ring's other sets."""
        return [*sh.b.values(), *(t for st in sh.sets[1:] for t in st.values())]


    def _rows(self) -> str:
        """A's rows form: `rows_form`, but `w` without hashing, since `s`'s stats ride on A's row leaves."""
        return self.rows_form if self.hashing else "w"

    def _call_names(self) -> tuple[str, ...]:
        """The call's step names: every call step but the hashing ones when hashing is off, but the omitted ones, and but
        stats_a under rows form `s` (hash_rows_a writes it)."""
        return tuple(x for x in CALL_STEPS if (self.hashing or x not in HASH_STEPS) and x not in self.omit
                     and not (x == "stats_a" and self._rows() == "s"))

    def _call_steps(self, pipe: Any) -> list[Any]:
        """The call's steps in order (`_call_names`)."""
        return _steps(pipe, self._call_names())

    def _call_cuts(self, pipe: Any) -> tuple[int, int]:
        """Where a deferred whole capture (`whole_defer`) splits `_call_steps`: after A's tree, beside which the call before's
        tile hashing runs, and before the call's own tile hashing, which it forks."""
        have = {name for name, _, _ in pipe.steps()}
        groups = [STEP_GROUPS[x] for x in self._call_names() if x in have]
        a = next((i for i, g in enumerate(groups) if g != "a_tree"), len(groups))
        b = next((i for i, g in enumerate(groups) if g == "tile_hash"), len(groups))
        if "a_tree" in groups[a:] or any(g != "tile_hash" for g in groups[b:]):
            raise RuntimeError(f"{self.entry}: the call's steps {groups} don't start with A's tree and end with the tile hashing")
        return a, b

    def _pipe(self, w: int, sh: _Shape) -> tuple[Any, list[Any], list[Any]]:
        """The call's pipeline and its steps: the main lane's, and (with hashing) the side lane's, by the schedule's groups."""
        key = (w, sh.m16, self.omit, self.hashing, self.schedule.name, self._graphed(sh.m16))
        if key not in self.pipes:
            wt, K = self.weights[w], self.kernel
            bufs = {**{x: t.data_ptr() for x, t in wt.b.items()}, **{x: t.data_ptr() for x, t in sh.b.items()}, "y": 0,
                    "leaves": 0}
            kp = self.key_slots[self.order[w]].data_ptr() if self._graphed(sh.m16) else sh.keys.data_ptr()
            keys = {"line_b": wt.keys.data_ptr(), "dom_b": wt.keys.data_ptr() + 32, "dom_a": kp, "unit": kp + 32,
                    **_constant_keys(self.const)}
            pipe = _pipeline(K, sh.mp, sh.n, sh.k, bufs, 0, keys, sh.xa.data_ptr(), sh.m16, variant=self.variant,
                             hashing=self.scheme.hashing, rows_form=self._rows())
            names = self._call_names()
            present = [x for x in names if x in {n for n, _, _ in pipe.steps()}]
            steps = _steps(pipe, names)
            if self.time_gemm and self.device.type == "cuda":
                at = present.index("gemm_pearlc")
                steps[at] = self._timed(steps[at])
            lane = {x: "side" if self.hashing and STEP_GROUPS[x] in self.schedule.side else "main" for x in present}
            main = [f for x, f in zip(present, steps, strict=True) if lane[x] == "main"]
            aside = [f for x, f in zip(present, steps, strict=True) if lane[x] == "side"]
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
        index = len(cur.matmuls)
        tiles = self.kernel.run.cdiv(m16, self.kernel.run.TILE) * (wt.n // self.kernel.run.TILE)
        if self.hashing and (cur.tiles + tiles > self.capacity or index >= CALL_CAPACITY or cur.flag_rows + m > FLAG_ROWS):
            raise RuntimeError(f"{self.entry}: the pass is over its capacity ({self.capacity} tiles, {CALL_CAPACITY} calls, "
                               f"{FLAG_ROWS} screened rows); end it and begin another")
        self.join_before(sh)
        sh.xa[:m].copy_(x2)
        if m16 > m:
            sh.xa[m:m16].zero_()
        pipe, steps, side = self._pipe(w, sh)
        raw = self.stream()
        cuda = self.device.type == "cuda"
        caller = self.torch_stream(raw) if cuda else None
        main = self.main if self.schedule.main_priority == "greatest" else None
        if self._graphed(m16):
            y, screened = self._graph_call(w, sh, pipe, raw, tiles, m, index, caller)
            return self._after(cur, w, wt, sh, pipe, index, m, k, m16, tiles, y, lead, bias, screened, scattered=self.hashing)
        y = torch.empty((sh.mp, wt.n), dtype=torch.bfloat16, device=x.device)
        pipe.b["y"], pipe.b["leaves"] = y.data_ptr(), self.leaves.data_ptr() + 32 * cur.tiles
        lane_raw, lane = (main.raw, main.stream) if main is not None else (raw, caller)
        if side:
            j = sh.turn % len(sh.sets)
            sh.turn += 1
            pipe.b.update({x: t.data_ptr() for x, t in sh.sets[j].items()})
        if self.hashing:
            self.uploads.put(sh.keys, self.keys(self.roots[w], index, m16), caller if self.uploads.cuda else None)
        if cuda and main is not None:               # the main lane starts after the caller's x, y and keys
            main.event.record(caller)
            lane.wait_event(main.event)
        if cuda and side and sh.done[j] is not None:  # and rewrites set j only once the side lane is done with it
            lane.wait_event(sh.done[j])
        self._launching = lane
        self.kernel.dev.stream = lane_raw
        try:
            for f in steps:
                f()
            if side:
                if cuda:
                    self.side.event.record(lane)
                    self.side.stream.wait_event(self.side.event)
                self.kernel.dev.stream = self.side.raw
                for f in side:
                    f()
                if cuda:
                    sh.done[j] = sh.done[j] or torch.cuda.Event()
                    sh.done[j].record(self.side.stream)
            if cuda and main is not None:           # the caller reads A's root and y only after the main lane
                main.event.record(lane)
                caller.wait_event(main.event)
        finally:
            self.kernel.dev.stream = None
            self._launching = None
        return self._after(cur, w, wt, sh, pipe, index, m, k, m16, tiles, y, lead, bias,
                           rows=x2 if x2.is_contiguous() and x2.dtype == torch.bfloat16 else None)


    def _after(self, cur: _Pass, w: int, wt: _Weight, sh: _Shape, pipe: Any, index: int, m: int, k: int, m16: int, tiles: int,
               y: Any, lead: Any, bias: Any, screened: bool = False, rows: Any = None, scattered: bool = False) -> Any:
        """A call's records after its steps: its matmul, A's root, the screen (unless a graph `screened` it), the retained
        files, and y.  The screen reads `rows` (the call's BF16 x, whose widening to A's FP32 words is exact) when given, at
        half the bytes of A's words.  A graph that `scattered` wrote A's root, the leaves and (when it screened) the flags."""
        cur.matmuls.append(Matmul(index, Shape(m16, k, wt.n), w))
        if self.hashing:
            size = pipe.size["tree_a"]
            if not scattered:
                self.roots_buf[32 * index:32 * index + 32].copy_(sh.b["tree_a"][size - 32:size])
            flags = self.flag_buf[cur.flag_rows:cur.flag_rows + m]
            if not screened:
                screen_into(flags, sh.xa[:m] if rows is None else rows, sh.b["alpha_a"], sh.b["aux_a"], self.fused_screen)
            cur.flag_rows += m
            if cur.retain is not None:
                self._retained(cur, cur.matmuls[-1], m, flags, *self._kept(sh, pipe))
            cur.tiles += tiles
        out = y[:m] if bias is None else y[:m] + bias
        return out.reshape(*lead, wt.n)

    # ---- passes -----------------------------------------------------------------------------------------------------

    def begin(self, retain: str | None = None, run: str | None = None, schedule: str | Schedule | None = None) -> None:
        """A new pass: its own transcript under the install's epoch, the call indices from 0, under `schedule` (the last one's
        when None; a schedule changes only between passes).  With `retain`, the pass writes what the prover opens from into
        that directory, from poisoned outputs, and its manifest names `run`."""
        if self.side is not None and self.side.stream is not None:     # the last pass's side lane is done before anything new
            self.torch_stream(self.stream()).wait_stream(self.side.stream)
        self._use(self.schedule if schedule is None else schedule_of(schedule, self.scheme))
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
            for t in self._graph_buffers():
                t.fill_(value)

    def end(self) -> dict[str, Any]:
        """Close the pass: its tile tree on the device, every A root, and with `retain` the manifest.  Returns the pass's
        commitment (also in `commits` and `details["passes"]`)."""
        import torch
        cur, K = self.cur, self.kernel
        out: dict[str, Any] = {"units": len(cur.matmuls), "tiles": cur.tiles, "hashing": self.hashing, "schedule": self.schedule.name}
        if self.hashing and cur.matmuls:
            if self.side is not None and self.side.stream is not None:
                self.torch_stream(self.stream()).wait_stream(self.side.stream)
            epoch = Epoch(Workload(tuple(cur.matmuls)), self.roots, self.salt)
            R, count = K.run, cur.tiles
            tree = _zeros(32 * (2 * R.pow2(count) - 1), self.device)
            lk, dom = _zeros(32 * max(R.depth(count), 1), self.device), _zeros(32, self.device)
            _put(dom, transcript_domain(epoch, count, self.tree_hash).domain_id)
            ns = types.SimpleNamespace(dev=K.dev, b={"tree_t": tree.data_ptr(), "keys_t": lk.data_ptr()},
                                       keys={"dom_t": dom.data_ptr(), **_constant_keys(self.const)}, hashing=self.scheme.hashing)
            K.dev.stream = self.stream()
            try:
                for _, f in R.Pipeline.tree_setup(ns, "t", count, "dom_t"):
                    f()
                R.Pipeline.tree_leaves(ns, "t", self.leaves.data_ptr(), count, "dom_t")
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
                "coins": self.coins.to_json(), "salt_opening": self.opening.to_json(), "salt": self.salt.hex(),
                "cubin_sha256": self.kernel.cubin_sha256,
                "kernel_source_sha256": self.kernel.source_sha256, "schedule": self.details["schedule"],
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


class PearlCExecutor:
    """A `PearlCDeviceRun` behind `verity_vllm.linear.api.LinearExecutor`, the one object a `Stack` computes a site's y with.
    Its weight state is B̃, formed once per install, so it can't sit behind a source that forbids weight state (POUS); it reads
    BF16 weights; its controls are the poison byte and omitted step groups (by step name, as `PearlCDeviceRun.omit` takes
    them); its schedules are those `schedule_of` admits for its scheme."""

    keeps_weight_state = True
    accepts = frozenset({"bf16"})
    controls = frozenset({"poison", "omit"})

    def __init__(self, run: PearlCDeviceRun) -> None:
        self.run, self.name = run, run.entry
        ok = set()
        for name in SCHEDULES:
            try:
                schedule_of(name, run.scheme)
            except ValueError:
                continue
            ok.add(name)
        self.schedules = frozenset(ok)

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
            return self.run.end()
        finally:
            self.run.omit = frozenset()
