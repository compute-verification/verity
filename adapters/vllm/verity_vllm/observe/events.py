"""Raw observation log: the event vocabulary (schema `verity-capture/log/v1`).

The observer emits *facts* about one eager vLLM execution.  It never decides whether an event is a view, a copy,
a computation, which Verity definition it corresponds to, or whether it is supported: those are resolver
judgements made later, offline, by pure code (`resolver.py`, `fold.py`).  Everything in this module is plain
Python (no torch import) so that `fold(read_log(path), profile)` runs with no GPU and no vLLM process.

Physical tensor views
---------------------
Every tensor argument or result is described by a `ViewDesc`: the allocation it lives in (an `AllocationId`
assigned by `storage.py`, never a raw device pointer), its storage offset in elements, shape, strides, dtype and
device.  Accurately describing the physical views involved in an event is the observer's one nontrivial
responsibility.

Event ids
---------
`DispatchEnter`/`DispatchExit` and `TritonLaunch` carry a run-unique integer `id`; nesting is expressed by
`parent` (the id of the enclosing dispatcher call, or None).  A dispatcher call's children (re-entered custom-op
bodies, Triton launches issued from Python inside it) appear between its Enter and Exit records.  Nesting is
information for the resolver's subtree patterns; it does not by itself decide Verity boundaries.

Encoding of arguments
---------------------
`encode_arg` (runtime side, in `observer.py`) produces the nested plain structure stored in `args`/`kwargs`/
`outs`: `ViewDesc` for tensors, `None`, `bool`/`int`/`float`/`str`, lists (tuples and tensor lists), and
`{"$repr": ...}` for anything else (dtypes, devices, memory formats, generators are rendered as their `str`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Iterable

SCHEMA = "verity-capture/log/v1"

# torch dtype name (without the "torch." prefix) -> bytes per element.  The pure side never imports torch.
DTYPE_BYTES: dict[str, int] = {
    "bfloat16": 2, "float16": 2, "float32": 4, "float64": 8,
    "int8": 1, "uint8": 1, "int16": 2, "uint16": 2, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8,
    "bool": 1, "float8_e4m3fn": 1, "float8_e5m2": 1, "float8_e8m0fnu": 1, "complex64": 8,
}

# dtype -> Verity value width in bits (for gate identities; a bf16 element is one Value<16>)
DTYPE_BITS: dict[str, int] = {k: 8 * v for k, v in DTYPE_BYTES.items()}
DTYPE_BITS["bool"] = 8


@dataclass(frozen=True)
class ViewDesc:
    """A physical tensor view.  `offset` is the storage offset in *elements of `dtype`* (torch `storage_offset()`).
    Logical element `(i_0, ..., i_{r-1})` lives at storage element `offset + sum(i_k * strides[k])`."""

    alloc: int
    offset: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "strides", tuple(int(s) for s in self.strides))
        if len(self.shape) != len(self.strides):
            raise ValueError(f"shape/strides rank mismatch: {self.shape} vs {self.strides}")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unknown dtype {self.dtype!r}")

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def itemsize(self) -> int:
        return DTYPE_BYTES[self.dtype]

    @property
    def bits(self) -> int:
        return DTYPE_BITS[self.dtype]

    @property
    def byte_offset(self) -> int:
        return self.offset * self.itemsize

    @property
    def rank(self) -> int:
        return len(self.shape)

    def to_json(self) -> dict:
        return {"$v": [self.alloc, self.offset, list(self.shape), list(self.strides), self.dtype, self.device]}

    @staticmethod
    def from_json(d: dict) -> "ViewDesc":
        a, o, sh, st, dt, dev = d["$v"]
        return ViewDesc(int(a), int(o), tuple(sh), tuple(st), dt, dev)

    def __repr__(self) -> str:
        return f"View(a{self.alloc}+{self.offset} {self.dtype}{list(self.shape)} s={list(self.strides)})"


# ---------------------------------------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunHeader:
    """First record of a log."""

    schema: str
    created_utc: str
    profile_id: str            # `verity_vllm.engine.engine_profile.profile_id_of(manifest)` of the pinned profile
    profile: dict              # the manifest itself (versions, flags, gpu, model, sampling)
    model: dict                # {"repo", "revision", "config": {...hf config subset...}}
    workload: dict             # {"requests": [{"request_id", "prompt_token_ids", "max_tokens"}], "sampling": {...}}
    notes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunEnd:
    stats: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StorageAlloc:
    """A device allocation as identified by the storage adapter.  `generation` increments each time the same
    device pointer range is observed to be reused for a new allocation (an implied free of the previous one)."""

    alloc: int
    ptr: int
    nbytes: int
    device: str
    generation: int
    origin: str                # "param" | "buffer" | "op" | "external" | "unknown"


@dataclass(frozen=True)
class StorageFree:
    alloc: int
    implied: bool = True       # True: inferred from pointer reuse; False: observed directly (allocator hook)


@dataclass(frozen=True)
class Root:
    """A value with no producer inside the observed computation: a model parameter or persistent buffer, a
    prompt token (prescribed input), a position, or a constant.  `values` carries the public integer contents
    for token/position roots (they are prescribed metadata, not secrets)."""

    view: ViewDesc
    role: str                  # "weight" | "buffer" | "prompt_token" | "position" | "const"
    name: str                  # parameter/buffer qualified name, or "prompt[<request>]" etc.
    values: list[int] | None = None


@dataclass(frozen=True)
class StepStart:
    """Engine step context, read from the model runner *after* vLLM's input preparation / request reorder."""

    step: int
    rows: list[tuple[str, int]]            # per token row: (request_id, absolute position)
    input_token_ids: list[int]             # per token row
    positions: list[int]                   # per token row
    seq_lens: dict[str, int]               # request -> tokens in the KV cache after this step
    num_computed: dict[str, int]           # request -> tokens already in the KV cache before this step
    block_tables: dict[str, list[int]]     # request -> physical block ids (positions 0..seq_len-1)
    block_size: int
    slot_mapping: list[int]                # per token row: KV slot = block * block_size + offset
    logits_rows: list[tuple[str, int]]     # per logits row: (request_id, position whose logits are sampled)
    logits_indices: list[int]              # per logits row: the token row it gathers hidden states from
    num_reqs: int
    num_tokens: int
    num_tokens_padded: int
    input_ids_view: ViewDesc | None = None
    positions_view: ViewDesc | None = None
    phase_hint: str = ""                   # "prefill" | "decode" | "mixed" (diagnostic only)
    # vLLM's hybrid KV-cache manager gives layers of different attention kinds
    # (Gemma-2: sliding-window even layers / global odd layers) separate KV-cache GROUPS, each with its own block tables and
    # slot mapping, that index the SAME physical tensors (log Note: kv_cache[layers.0] and kv_cache[layers.1] at one offset).
    # `block_tables`/`slot_mapping` above are group 0's; with several groups the storage identity of a layer's KV rows is
    # decided by ITS group -- recorded here, one entry per group, in the runner's group order:
    #   {"layers": [layer names], "block_size": int, "block_tables": {request -> block ids}, "slot_mapping": [per token row]}
    # None (old logs) or a single entry: one group, the legacy fields are complete.
    kv_groups: list[dict] | None = None


@dataclass(frozen=True)
class SampleEnd:
    """Sampler result of a step: sampled token per logits row (greedy in the MVP profile)."""

    step: int
    logits_view: ViewDesc | None
    sampled_token_ids: list[int]           # per logits row
    sampled_view: ViewDesc | None = None


@dataclass(frozen=True)
class StepEnd:
    step: int
    forward_s: float = 0.0
    sample_s: float = 0.0


@dataclass(frozen=True)
class StepLink:
    """Host-mediated linkage: token row `row` of step `step` carries the token sampled for `request_id` at
    step `src_step` (D2H -> scheduler -> H2D).  `relation == "eq"`: Input_{step,row} = TokenSelect_{src_step,q}.
    `value` is the public token id for a consistency check."""

    step: int
    row: int
    request_id: str
    src_step: int
    relation: str = "eq"
    value: int | None = None


@dataclass(frozen=True)
class DispatchEnter:
    id: int
    parent: int | None
    step: int
    phase: str                 # "forward" | "sample" | "other"
    op: str                    # e.g. "aten.mm.default", "_C.fused_add_rms_norm.default", "vllm.unified_attention_with_output.default"
    schema: str
    args: list
    kwargs: dict
    stream: int
    module: str | None = None  # module-path hint from forward hooks (diagnostic; not part of any identity)


@dataclass(frozen=True)
class DispatchExit:
    id: int
    outs: Any                  # encoded like args (ViewDesc / list / scalar / None)


@dataclass(frozen=True)
class TritonLaunch:
    """A Triton kernel launch observed at `JITFunction.run` (bound arguments by name)."""

    id: int
    parent: int | None
    step: int
    phase: str
    fn: str                    # "<module>.<qualname>" of the JITFunction
    src_sha: str               # sha256 of the kernel source text
    args: dict[str, Any]       # bound argument name -> encoded value (tensors as ViewDesc), constexprs included
    constexprs: dict[str, Any]
    grid: list[int]
    num_warps: int | None
    num_stages: int | None
    stream: int
    module: str | None = None


@dataclass(frozen=True)
class Note:
    text: str
    data: dict = field(default_factory=dict)


Event = (RunHeader | RunEnd | StorageAlloc | StorageFree | Root | StepStart | SampleEnd | StepEnd | StepLink
         | DispatchEnter | DispatchExit | TritonLaunch | Note)

EVENT_TYPES: dict[str, type] = {c.__name__: c for c in (
    RunHeader, RunEnd, StorageAlloc, StorageFree, Root, StepStart, SampleEnd, StepEnd, StepLink,
    DispatchEnter, DispatchExit, TritonLaunch, Note)}


# ---------------------------------------------------------------------------------------------------------
# JSON (de)serialisation
# ---------------------------------------------------------------------------------------------------------


def _enc(x: Any) -> Any:
    if isinstance(x, ViewDesc):
        return x.to_json()
    if isinstance(x, (list, tuple)):
        return [_enc(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _enc(v) for k, v in x.items()}
    return x


def _dec(x: Any) -> Any:
    if isinstance(x, dict):
        if "$v" in x and len(x) == 1:
            return ViewDesc.from_json(x)
        return {k: _dec(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_dec(v) for v in x]
    return x


def to_json(ev: Event) -> dict:
    d = {"$e": type(ev).__name__}
    for f in dataclasses.fields(ev):
        d[f.name] = _enc(getattr(ev, f.name))
    return d


_TUPLE_LIST_FIELDS = {("StepStart", "rows"), ("StepStart", "logits_rows")}


def from_json(d: dict) -> Event:
    tag = d["$e"]
    cls = EVENT_TYPES[tag]
    kw = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = _dec(d[f.name])
        if (tag, f.name) in _TUPLE_LIST_FIELDS:
            v = [(r, int(p)) for r, p in v]
        elif tag == "TritonLaunch" and f.name == "grid":
            v = [int(g) for g in v]
        kw[f.name] = v
    return cls(**kw)


def views_in(x: Any) -> Iterable[ViewDesc]:
    """Every `ViewDesc` inside an encoded argument structure (depth-first, argument order)."""
    if isinstance(x, ViewDesc):
        yield x
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from views_in(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from views_in(v)
