"""Correspondence between capture-format-v2 identities and the canonical B1 circuit.

Two namespaces, kept separate in every record this tool emits:

  * `value_identity`  -- the *logical* namespace: which `FunctionCall` of the canonical circuit produced the
    value(s), i.e. the gate indices of `call.returns` (or, for prescribed inputs, the `Input` gates).
  * `authentication_location` -- the *physical* namespace: commitment tree, leaf hash, byte offset / length
    of the words that authenticate those values (a replay-unit leaf, a boundary leaf or a weights leaf).

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values, not semantic components
    of gate outputs.

Checks performed per located call: (i) shape/type consistency between the captured tensor and the call's
return type, (ii) bit-for-bit comparison of the captured words with the circuit's own gate-by-gate evaluation of
that call, where the call's *argument* values are read from other captured leaves (production values) and
from the registered checkpoint (the values of the weight `Input` gates). A prescribed-input record states
that the committed weight leaf words *are* the outputs of the `Input` gates (one scalar per gate) and gives
their row-block authentication position; nothing is treated as "given".
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import MissingValue, evaluate_region
from verity.ml.circuits.layout import Circuit, FunctionCall, resolve
from verity.ml.circuits.program import Program
from verity_vllm.config import UsageError, option
from verity.ml.definitions.registry import b1

_ID_RE = re.compile(r"^p=(?P<p>[^/]+)/r=(?P<r>[^/]+)/t=(?P<t>\d+)/ph=(?P<ph>[^/]+)/l=(?P<l>[^/]+)/op=(?P<op>[^/]+)/port=(?P<output>\d+)$")


@dataclass(frozen=True)
class CaptureIdentity:
    profile: str
    request: str
    t: int
    phase: str
    layer: str
    op: str
    output: int  # the capture format's `port=<k>` field: k-th output tensor of the production op

    @classmethod
    def parse(cls, canonical_id: str) -> "CaptureIdentity":
        m = _ID_RE.match(canonical_id)
        if not m:
            raise ValueError(f"not a capture-v2 identity: {canonical_id}")
        return cls(m["p"], m["r"], int(m["t"]), m["ph"], m["l"], m["op"], int(m["output"]))


# ---------------------------------------------------------------------------------------------------------
# the map: capture identity -> (named root call, child selector, returns slice)
# ---------------------------------------------------------------------------------------------------------

# (op suffix after "layers.L.", output index) -> (root call role, child function selector, which return component)
_LAYER_OPS: dict[tuple[str, int], tuple[str, str | None, int | None]] = {
    ("input_layernorm.rmsnorm", 0): ("pre", "RMSNorm*", 0),
    ("input_layernorm.rmsnorm", 1): ("pre", "RMSNormFusedCuda", 1),
    ("self_attn.qkv_proj.gemm", 0): ("pre", "Gemm", 0),  # B1: pre-bias GEMM output; B0 (no bias): the QKV values
    ("self_attn.qkv_proj.add", 0): ("pre", "BiasAdd", 0),
    ("self_attn.rotary_emb.rope", 0): ("pre", "RoPE#0", 0),
    ("self_attn.rotary_emb.rope", 1): ("pre", "RoPE#1", 0),
    ("self_attn.attn.attn", 0): ("attn", None, 0),
    ("self_attn.o_proj.gemm", 0): ("post", "Gemm#0", 0),
    ("post_attention_layernorm.rmsnorm", 0): ("post", "RMSNormFusedCuda", 0),
    ("post_attention_layernorm.rmsnorm", 1): ("post", "RMSNormFusedCuda", 1),
    ("mlp.gate_up_proj.gemm", 0): ("post", "Gemm#1", 0),
    ("mlp.act_fn.act_mul", 0): ("post", "SiluMul", 0),
    ("mlp.down_proj.gemm", 0): ("post", "Gemm#2", 0),
}


def _select_child(parent: FunctionCall, sel: str, names: dict[str, str] | None = None) -> FunctionCall:
    """`Name` (unique), `Name#k` (k-th child with that name), `Name*` (prefix). `names` maps the mapping's (TA1)
    function names to a profile port's bound names (e.g. `Gemm -> GemmHopper`); the mapping itself is unchanged."""
    k = 0
    if "#" in sel:
        sel, ks = sel.split("#")
        k = int(ks)
    if names:
        star = sel.endswith("*")
        base = sel[:-1] if star else sel
        sel = names.get(base, base) + ("*" if star else "")
    kids = [c for c in parent.children() if (c.name.startswith(sel[:-1]) if sel.endswith("*") else c.name == sel)]
    if k >= len(kids):
        raise KeyError(f"{parent.node.name}: no child {sel!r}#{k} among {[c.name for c in parent.children()]}")
    return kids[k]


def _returns_component(call: FunctionCall, which: int) -> tuple[int, ...]:
    """Gate indices of the `which`-th top-level component of `call.returns` (a Tuple return) or all of them."""
    rets = call.returns
    rt = call.return_type()
    if type(rt).__name__ != "Tuple":
        return rets
    off = 0
    for i, t in enumerate(rt.fields):
        n = t.leaves
        if i == which:
            return rets[off:off + n]
        off += n
    raise IndexError(which)


class NotInProgram(KeyError):
    """A captured production value with no counterpart in the canonical program (recorded, never silently dropped)."""


class CircuitMap:
    """Capture-v2 identity -> value identity (gates of a `FunctionCall`'s returns) for the B1 `Serve` program."""

    def __init__(self, program: Program, names: dict[str, str] | None = None):
        self.program = program
        self.circuit: Circuit = program.circuit
        self.names = names or {}
        self.named: dict[str, FunctionCall] = {}
        for c in self.circuit.root_calls():
            if c.node.name:
                self.named[c.node.name] = c

    def locate(self, ident: CaptureIdentity) -> tuple[FunctionCall, tuple[int, ...]]:
        """The producing call and the root-scoped gate indices of the captured words, in tensor row-major order."""
        t = ident.t
        if ident.layer == "embed":
            c = self.named[f"t={t}/embed"]
            return c, c.returns
        if ident.layer in ("norm", "lm_head", "sampler") and f"t={t}/final" not in self.named:
            raise NotInProgram(f"t={t}: production computes {ident.op} at every prefill position; the canonical program "
                               f"has a Final call only at sampled positions (this value has no consumer)")
        if ident.layer == "norm":
            fin = self.named[f"t={t}/final"]
            c = _select_child(fin, "RMSNormFusedCuda", self.names)
            return c, _returns_component(c, ident.output)
        if ident.layer == "lm_head":
            c = _select_child(self.named[f"t={t}/final"], "Gemm", self.names)
            return c, c.returns
        if ident.layer == "sampler":
            c = _select_child(self.named[f"t={t}/final"], "TokenSelect", self.names)
            return c, c.returns
        L = int(ident.layer)
        suffix = ident.op[len(f"layers.{L}."):]
        role, sel, which = _LAYER_OPS[(suffix, ident.output)]
        root = self.named[f"t={t}/layers.{L}/{role}"]
        c = root if sel is None else _select_child(root, sel, self.names)
        return c, _returns_component(c, which or 0)


# ---------------------------------------------------------------------------------------------------------
# value sources: segments of root gate indices backed by uint16/uint32 words
# ---------------------------------------------------------------------------------------------------------


class Segments:
    """Sorted disjoint [start, end) gate ranges, each backed by a flat words array and a location record.
    Additions are buffered and sorted on first lookup (Gemm returns are one gate per coordinate)."""

    def __init__(self) -> None:
        self._pending: list[tuple[int, np.ndarray, dict]] = []
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.words: list[np.ndarray] = []
        self.locs: list[dict] = []

    def add(self, start: int, words: np.ndarray, loc: dict) -> None:
        self._pending.append((start, words, loc))

    def _finalize(self) -> None:
        if not self._pending:
            return
        items = sorted(zip(self.starts, self.words, self.locs), key=lambda x: x[0]) + sorted(self._pending, key=lambda x: x[0])
        items.sort(key=lambda x: x[0])
        self._pending = []
        self.starts, self.ends, self.words, self.locs = [], [], [], []
        for start, words, loc in items:
            if self.ends and self.ends[-1] > start:
                raise ValueError(f"overlapping segment at gate {start} ({loc.get('leaf_id')} vs {self.locs[-1].get('leaf_id')})")
            self.starts.append(start)
            self.ends.append(start + len(words))
            self.words.append(words)
            self.locs.append(loc)

    def __len__(self) -> int:
        return len(self.starts) + len(self._pending)

    def find(self, g: int) -> tuple[int, int] | None:
        if self._pending:
            self._finalize()
        i = bisect.bisect_right(self.starts, g) - 1
        if i >= 0 and g < self.ends[i]:
            return i, g - self.starts[i]
        return None

    def value(self, g: int) -> int | None:
        f = self.find(g)
        if f is None:
            return None
        i, k = f
        return int(self.words[i][k])

    def location(self, g: int) -> dict | None:
        f = self.find(g)
        if f is None:
            return None
        i, k = f
        return {**self.locs[i], "word": k}


class LazyTranscript(dict):
    """A `Trace` whose missing entries are read from value segments (weights + captured leaves).
    Values computed by `evaluate_region` are stored in the dict and take precedence."""

    def __init__(self, *sources: Segments) -> None:
        super().__init__()
        self.sources = sources

    def __missing__(self, g: int) -> int:
        for s in self.sources:
            v = s.value(g)
            if v is not None:
                return v
        raise KeyError(g)

    def __contains__(self, g: object) -> bool:  # type: ignore[override]
        return dict.__contains__(self, g) or any(s.find(g) is not None for s in self.sources)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------------
# weights: the checkpoint gives the values of the weight Input gates; the weights leaf authenticates them
# ---------------------------------------------------------------------------------------------------------

_WEIGHT_FIELDS = {  # circuit weights record field -> (capture canonical id, checkpoint tensors concatenated along dim 0)
    "embed": ("w=embed_tokens.weight", ["model.embed_tokens.weight"]),
    "norm": ("w=norm.weight", ["model.norm.weight"]),
    "cos_sin": ("w=layers.0.self_attn.rotary_emb.cos_sin_cache", None),
}
_LAYER_FIELDS = {
    "ln1": ("w=layers.{L}.input_layernorm.weight", ["model.layers.{L}.input_layernorm.weight"]),
    "qkv_w": ("w=layers.{L}.self_attn.qkv_proj.weight", [f"model.layers.{{L}}.self_attn.{p}_proj.weight" for p in "qkv"]),
    "qkv_b": ("w=layers.{L}.self_attn.qkv_proj.bias", [f"model.layers.{{L}}.self_attn.{p}_proj.bias" for p in "qkv"]),
    "o_w": ("w=layers.{L}.self_attn.o_proj.weight", ["model.layers.{L}.self_attn.o_proj.weight"]),
    "ln2": ("w=layers.{L}.post_attention_layernorm.weight", ["model.layers.{L}.post_attention_layernorm.weight"]),
    "gate_up_w": ("w=layers.{L}.mlp.gate_up_proj.weight", ["model.layers.{L}.mlp.gate_proj.weight", "model.layers.{L}.mlp.up_proj.weight"]),
    "down_w": ("w=layers.{L}.mlp.down_proj.weight", ["model.layers.{L}.mlp.down_proj.weight"]),
}


def _coll_range(circuit: Circuit, coll) -> tuple[int, int]:
    """Root gate range [start, end) of a contiguous Input collection."""
    n = coll.refs.count if hasattr(coll.refs, "count") else coll.type.leaves
    first = resolve(circuit.root_scope, coll.refs[0])
    last = resolve(circuit.root_scope, coll.refs[n - 1])
    if last - first != n - 1:
        raise ValueError("input collection is not contiguous")
    return first, first + n


def load_safetensors(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    """Memory-mapped raw uint16 views of every bf16 tensor across the shards."""
    out: dict[str, np.ndarray] = {}
    for p in paths:
        with open(p, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        base = 8 + n
        mm = np.memmap(p, dtype=np.uint8, mode="r")
        for k, v in header.items():
            if k == "__metadata__":
                continue
            assert v["dtype"] == "BF16", (k, v["dtype"])
            a, b = v["data_offsets"]
            out[k] = mm[base + a:base + b].view(np.uint16).reshape(v["shape"])
    return out


def cos_sin_cache_words(max_pos: int, D: int, theta: float, device: str = "cpu") -> np.ndarray:
    """vLLM `RotaryEmbedding._compute_cos_sin_cache` (neox style, fp32 cos/sin, bf16 cast at the end) as raw
    uint16 words. vLLM builds this table *on the GPU*; fp32 `cos`/`sin` differ between CUDA libdevice and the
    CPU libm in the last ulp for some arguments, so only `device="cuda"` reproduces the committed leaf."""
    import torch  # deferred import: only the weights binding needs torch

    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float, device=device) / D))
    t = torch.arange(max_pos, dtype=torch.float, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)
    return cache.cpu().contiguous().view(torch.int16).numpy().view(np.uint16)


@dataclass
class WeightsBinding:
    segments: Segments
    records: list[dict] = field(default_factory=list)


# The `cos_sin_cache` Input gates are a *derived* table (typed-b1.md §5.2): the committed leaf reproduces only from
# vLLM's `_compute_cos_sin_cache` run on the GPU. The record says so in a fixed, machine-readable form.
COS_SIN_LEAF = _WEIGHT_FIELDS["cos_sin"][0]
COS_SIN_REBUILT_FROM = "vllm._compute_cos_sin_cache (CUDA cosf/sinf), GPU only"
COS_SIN_REBUILT_FROM_CPU = ("vllm._compute_cos_sin_cache (CPU libm cosf/sinf) — does NOT reproduce the committed leaf; "
                            "only the CUDA computation does")


def cos_sin_qualification(total_words: int, cpu_rebuild: dict | None, device: str) -> dict:
    """The qualification attached to the `cos_sin_cache` Input-gate record: the registered source of these
    Input gates is a GPU computation, and the CPU re-computation disagrees in `differing_words` of `total_words`."""
    q: dict = {"kind": "derived-table", "source": "vllm.model_executor.layers.rotary_embedding RotaryEmbedding._compute_cos_sin_cache",
               "reproduces_committed_leaf_on": ["cuda"], "rebuilt_on": device, "total_words": total_words}
    if cpu_rebuild is not None:
        q["cpu_rebuild_differing_words"] = cpu_rebuild["differing_words"]
        q["cpu_rebuild_matches_leaf"] = cpu_rebuild["matches_leaf"]
    q["statement"] = ("the Input gates of this table are 'rebuilt from the registered source' only if that source is the CUDA "
                      "computation" + (f"; the CPU rebuild differs in {cpu_rebuild['differing_words']:,} of {total_words:,} words"
                                       if cpu_rebuild is not None else ""))
    return q


def summary_qualifications(weights_records: list[dict], records: list[dict], summary: dict) -> list[str]:
    """Human-readable qualifications that every summary line of a correspondence record must carry (GUIDANCE §18.7):
    a record is not 'universally exact' while any of these holds."""
    quals: list[str] = []
    for r in weights_records:
        if r.get("leaf_id") == COS_SIN_LEAF:
            q = r.get("qualification") or {}
            cpu = r.get("cpu_rebuild")
            s = f"cos_sin_cache Input gates ({q.get('total_words', r['value_identity']['scalars']):,} words): rebuilt from {r['rebuilt_from']}"
            if cpu is not None:
                s += f"; CPU rebuild differs in {cpu['differing_words']:,} words"
            quals.append(s)
    nip = [r for r in weights_records if r.get("not_in_program")]
    if nip:
        quals.append(f"{len(nip)} committed weight leaves are not read by any gate of the program (recorded not_in_program, e.g. {nip[0]['leaf_id']})")
    mism = [r for r in records if r.get("check", {}).get("status") == "mismatch"]
    for r in mism:
        fn = r.get("value_identity", {}).get("function", "?")
        m = r["check"].get("mismatches", "?")
        quals.append(f"mismatch: {r['capture_id'].split('/', 4)[-1]} ({fn}; {m} of {r['check'].get('words_compared', '?')} word(s))")
    if summary.get("sampled"):
        quals.append(f"{summary['sampled']} Gemm leaves are coordinate-sampled, not compared in full")
    return quals


def requalify(path: Path, write: bool = True) -> dict:
    """Bring an existing `verity-ir/correspondence/v0` record to the explicit qualification form: the
    `cos_sin_cache` Input-gate record's `rebuilt_from` becomes the fixed GPU-only string, gets a `qualification`
    object (counts from its own `cpu_rebuild`), and `summary.qualifications` lists everything that keeps the
    record from reading as 'universally exact'. Nothing else in the record changes; the edit is noted in
    `requalified`."""
    d = json.loads(path.read_text())
    if d.get("schema") != "verity-ir/correspondence/v0":
        raise ValueError(f"{path}: not a correspondence/v0 record")
    for r in d["input_gate_records"]["weights"]:
        if r.get("leaf_id") == COS_SIN_LEAF and "value_identity" in r and r["value_identity"]:
            old = (r.get("qualification") or {}).get("previous_rebuilt_from", r["rebuilt_from"])  # idempotent
            dev = "cuda" if "cuda" in old.lower() or r.get("sha256_matches_leaf") else "cpu"
            r["rebuilt_from"] = COS_SIN_REBUILT_FROM if dev == "cuda" else COS_SIN_REBUILT_FROM_CPU
            r["qualification"] = cos_sin_qualification(r["value_identity"]["scalars"], r.get("cpu_rebuild"), dev)
            r["qualification"]["previous_rebuilt_from"] = old
    d["summary"]["qualifications"] = summary_qualifications(d["input_gate_records"]["weights"], d["records"], d["summary"])
    d["requalified"] = {"by": "verity_vllm.correspondence.capture_identities.requalify", "changed": ["input_gate_records.weights[cos_sin_cache].rebuilt_from",
                                                                              "input_gate_records.weights[cos_sin_cache].qualification",
                                                                              "summary.qualifications"]}
    if write:
        path.write_text(json.dumps(d, indent=1))
    return d


def bind_weights(program: Program, circuit: Circuit, tensors: dict[str, np.ndarray], weights_index: dict,
                 C: dict, cos_sin: tuple[np.ndarray, str] | None = None, cos_sin_cpu: np.ndarray | None = None) -> WeightsBinding:
    """Bind every weight `Input` gate to its checkpoint word and record, per committed weights leaf, the
    Input-gate range (value identity) and the leaf (authentication location) with a sha256 cross-check."""
    idx = {e["canonical_id"]: e for e in weights_index["entries"]}
    W = program.input_coll(1)
    seg = Segments()
    records: list[dict] = []

    def one(coll, cid: str, words: np.ndarray, shape: list[int], source: str = "checkpoint tensor(s), raw bf16 words"):
        start, end = _coll_range(circuit, coll)
        flat = np.ascontiguousarray(words).reshape(-1)
        assert end - start == flat.size, (cid, end - start, flat.size)
        e = idx[cid]
        assert list(e["shape"]) == shape, (cid, e["shape"], shape)
        digest = hashlib.sha256(flat.tobytes()).hexdigest()
        seg.add(start, flat, {"namespace": "authentication", "commitment": "weights", "leaf_id": cid, "leaf": e["leaf"],
                              "sha256": e["sha256"]})
        records.append({
            "leaf_id": cid,
            "value_identity": {"input_gates": [start, end], "gate_class": "Input/weights", "scalars": end - start,
                               "field": coll_path},
            "authentication_location": {"commitment": "weights", "leaf": e["leaf"], "sha256": e["sha256"], "dtype": e["dtype"],
                                        "shape": e["shape"], "row_block": "row-major words of the leaf; word j <-> Input gate start+j"},
            "rebuilt_from": source,
            "rebuilt_sha256": digest,
            "sha256_matches_leaf": digest == e["sha256"],
        })

    coll_path = "weights.embed"
    one(W["embed"], *_WEIGHT_FIELDS["embed"][:1], tensors["model.embed_tokens.weight"], [C["V"], C["H"]])
    for L in range(C["NL"]):
        w = W["layers"][L]
        for fname, (cid_t, srcs) in _LAYER_FIELDS.items():
            coll_path = f"weights.layers[{L}].{fname}"
            if fname == "qkv_b" and not C.get("BIAS", True):
                continue
            parts = [tensors[s.format(L=L)] for s in srcs]
            arr = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
            one(w[fname], cid_t.format(L=L), arr, list(arr.shape))
    coll_path = "weights.norm"
    one(W["norm"], "w=norm.weight", tensors["model.norm.weight"], [C["H"]])
    coll_path = "weights.cos_sin"
    if cos_sin is None:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cos_sin = (cos_sin_cache_words(C["MAX_POS"], C["D"], C["ROPE_THETA"], dev), dev)
        if dev == "cuda" and cos_sin_cpu is None:
            cos_sin_cpu = cos_sin_cache_words(C["MAX_POS"], C["D"], C["ROPE_THETA"], "cpu")
    cs, dev = cos_sin
    one(W["cos_sin"], _WEIGHT_FIELDS["cos_sin"][0], cs, [C["MAX_POS"], C["D"]],
        source=COS_SIN_REBUILT_FROM if dev == "cuda" else COS_SIN_REBUILT_FROM_CPU)
    cpu_rebuild = None
    if cos_sin_cpu is not None:  # record the CPU divergence explicitly
        h = hashlib.sha256(cos_sin_cpu.tobytes()).hexdigest()
        cpu_rebuild = {"sha256": h, "differing_words": int((cos_sin_cpu != cs).sum()),
                       "matches_leaf": h == idx[_WEIGHT_FIELDS["cos_sin"][0]]["sha256"]}
        records[-1]["cpu_rebuild"] = cpu_rebuild
    records[-1]["qualification"] = cos_sin_qualification(int(np.asarray(cs).size), cpu_rebuild, dev)
    bound = {r["leaf_id"] for r in records}
    for cid in idx:
        if cid not in bound:  # e.g. fp8 `_k_scale`/`_v_scale` scalars: committed, but no gate of the bf16 program reads them
            records.append({"leaf_id": cid, "value_identity": None, "not_in_program": "no gate of the canonical program consumes this leaf",
                            "authentication_location": {"commitment": "weights", "leaf": idx[cid]["leaf"], "shape": idx[cid]["shape"]}})
    return WeightsBinding(seg, records)


# ---------------------------------------------------------------------------------------------------------
# captured replay-unit leaves
# ---------------------------------------------------------------------------------------------------------


class CommitStore:
    def __init__(self, root: Path):
        self.root = root
        self.commitments = json.loads((root / "commitments.json").read_text())
        self.weights_index = json.loads((root / "weights" / "index.json").read_text())
        self._ru: dict[tuple[str, int], tuple[dict, np.ndarray]] = {}

    def ru(self, request: str, step: int) -> tuple[dict, np.ndarray]:
        key = (request, step)
        if key not in self._ru:
            d = self.root / "ru" / request / f"{step:06d}"
            index = json.loads((d / "index.json").read_text())
            values = np.memmap(d / "values.bin", dtype=np.uint8, mode="r")
            self._ru[key] = (index, values)
        return self._ru[key]

    def boundary(self, request: str) -> dict:
        return json.loads((self.root / "ru" / request / "boundary.json").read_text())

    def steps(self, request: str) -> list[int]:
        return sorted(int(p.name) for p in (self.root / "ru" / request).iterdir() if p.is_dir())

    def step_of_position(self, request: str, t: int) -> int:
        for k, v in self.commitments["trees"].items():
            if v.get("kind") == "ru" and v["request_id"] == request and t in v["positions"]:  # commitments.json tree record
                return int(v["step"])
        raise KeyError((request, t))

    def leaf_words(self, request: str, step: int, entry: dict) -> np.ndarray:
        _, values = self.ru(request, step)
        raw = values[entry["offset"]:entry["offset"] + entry["nbytes"]]
        dt = {"bfloat16": np.uint16, "int32": np.uint32, "float32": np.uint32,
              "float8_e4m3fn": np.uint8}[entry["dtype"]]  # [QZ, additive] e4m3 leaves of the fp8 ports
        return np.frombuffer(raw.tobytes(), dtype=dt)

    def entries_at(self, request: str, t: int) -> tuple[int, list[dict]]:
        step = self.step_of_position(request, t)
        index, _ = self.ru(request, step)
        return step, [e for e in index["entries"] if f"/t={t}/" in e["canonical_id"]]

    def tree_root(self, request: str, step: int) -> str:
        return self.commitments["trees"][f"ru/{request}/{step:06d}"]["root"]


# ---------------------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------------------


def _shape_ok(call: FunctionCall, gates: tuple[int, ...], entry: dict) -> tuple[bool, str]:
    exp = int(np.prod(entry["shape"])) if entry["shape"] else 1
    if exp != len(gates):
        return False, f"captured {entry['shape']} has {exp} words, IR returns {len(gates)} gates"
    w = {call.circuit.gate(g).prim.ret.w for g in (gates[0], gates[-1])} if gates else set()
    want = {"bfloat16": 16, "int32": 32, "float32": 32, "float8_e4m3fn": 8}[entry["dtype"]]  # [QZ, additive] e4m3
    if w != {want}:
        return False, f"captured dtype {entry['dtype']} ({want} bits) vs gate widths {sorted(w)}"
    return True, "ok"


_MISSING_OPERAND_RE = re.compile(r"operand (\d+) is not in the transcript")


class Recomputer:
    """Exact recomputation of a value the capture never materialised (CF2 / W1cF: the compiled graph keeps
    B1's mid-layer residual in registers).  When evaluating a call raises `MissingValue` for operand gate `g`,
    find the *deepest composite call* whose region contains `g`, evaluate that region into the same overlay from
    committed values (recursively, bounded), and retry.  Nothing is invented: every recomputed word is a function
    of committed leaves under the registered definitions, and the record lists the regions recomputed."""

    def __init__(self, circuit: Circuit, *, max_depth: int = 6, max_gates: int = 20_000_000):
        self.circuit = circuit
        self.roots = sorted(circuit.root_calls(), key=lambda c: c.lo)
        self.los = [c.lo for c in self.roots]
        self.max_depth, self.max_gates = max_depth, max_gates
        self.stats = {"regions": 0, "gates": 0}

    def producing_call(self, g: int) -> FunctionCall | None:
        import bisect
        i = bisect.bisect_right(self.los, g) - 1
        if i < 0 or not (self.roots[i].lo <= g < self.roots[i].hi):
            return None
        c = self.roots[i]
        while True:
            nxt = next((k for k in c.children() if k.lo <= g < k.hi), None)
            if nxt is None:
                return c
            c = nxt

    def fill(self, overlay: "LazyTranscript", missing_gate: int, log: list[dict], depth: int = 0) -> None:
        c = self.producing_call(missing_gate)
        if c is None:
            raise MissingValue(f"gate {missing_gate} has no producing composite call")
        if depth >= self.max_depth or (c.hi - c.lo) > self.max_gates:
            raise MissingValue(f"refusing to recompute {c.id} ({c.hi - c.lo:,} gates, depth {depth}) for gate {missing_gate}")
        for _ in range(64):
            try:
                evaluate_region(self.circuit, c, overlay)
                break
            except MissingValue as e:
                m = _MISSING_OPERAND_RE.search(str(e))
                if not m:
                    raise
                self.fill(overlay, int(m.group(1)), log, depth + 1)
        else:
            raise MissingValue(f"recomputation of {c.id} did not converge")
        self.stats["regions"] += 1
        self.stats["gates"] += c.hi - c.lo
        log.append({"function": c.id, "call": value_identity(c, c.returns)["call"], "region": [c.lo, c.hi], "for_gate": missing_gate, "depth": depth})


def check_call(call: FunctionCall, gates: tuple[int, ...], transcript: LazyTranscript, captured: np.ndarray,
               *, sub=None, recompute: Recomputer | None = None) -> dict:
    """Evaluate `sub` (default: the whole call) gate by gate, reading outside operands from the transcript's
    sources, and compare the words at `gates` that lie inside `sub` with `captured`.  With `recompute`, an operand
    the capture never materialised is recomputed exactly from committed values (see `Recomputer`)."""
    region = call if sub is None else sub
    t0 = time.time()
    # evaluate into a fresh overlay so captured words of *this* call's own returns are never read as operands
    overlay = LazyTranscript(*transcript.sources)
    recomputed: list[dict] = []
    for _ in range(64):
        try:
            evaluate_region(call.circuit, region, overlay)
            break
        except MissingValue as e:
            m = _MISSING_OPERAND_RE.search(str(e)) if recompute is not None else None
            if m is None:
                return {"status": "missing-operand", "detail": str(e), "seconds": round(time.time() - t0, 3), "recomputed": recomputed or None}
            try:
                recompute.fill(overlay, int(m.group(1)), recomputed)
            except MissingValue as e2:
                return {"status": "missing-operand", "detail": f"{e}; recomputation failed: {e2}", "seconds": round(time.time() - t0, 3),
                        "recomputed": recomputed or None}
    else:
        return {"status": "missing-operand", "detail": "recomputation did not converge", "seconds": round(time.time() - t0, 3), "recomputed": recomputed}
    lo, hi = region.lo, region.hi
    compared = 0
    mism: list[dict] = []
    for j, g in enumerate(gates):
        if not (lo <= g < hi):
            continue
        compared += 1
        v = dict.__getitem__(overlay, g)
        if v != int(captured[j]):
            if len(mism) < 16:
                mism.append({"word": j, "gate": g, "ir": int(v), "captured": int(captured[j])})
            else:
                mism.append({})
    n_mism = len([m for m in mism if m]) + len([m for m in mism if not m])
    out = {"status": "exact" if n_mism == 0 else "mismatch", "gates_evaluated": hi - lo, "words_compared": compared,
           "mismatches": n_mism, "first_mismatches": [m for m in mism if m][:16], "seconds": round(time.time() - t0, 3)}
    if recomputed:
        out["recomputed"] = recomputed
    return out


def value_identity(call: FunctionCall, gates: tuple[int, ...]) -> dict:
    root_name = call.node.name
    if root_name is None:  # a child of a named root call: "<root call name>/<child function name>"
        sc = call.parent
        while sc.parent is not None and sc.parent.parent is not None:
            sc = sc.parent
        root_name = f"{sc.parent.spec.body.nodes[sc.node_idx].name}/{call.name}"
    contiguous = all(gates[i + 1] == gates[i] + 1 for i in range(len(gates) - 1))
    return {"namespace": "value", "function": call.id, "call": root_name, "region": [call.lo, call.hi],
            "returns_gates": [gates[0], gates[-1] + 1] if contiguous and gates else None,
            "returns_gate_count": len(gates), "contiguous": contiguous}


# ---------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Port:
    """A profile port's static configuration (coverage-wave contract, `vllm-poc/profiles/<port>/static_config.py`):
    the `Serve` definition to specialise, its config record, the port's names for TA1's function names, and which
    call names are Gemm-shaped."""
    serve: Any
    config: dict
    names: dict[str, str] = field(default_factory=dict)
    gemm_names: frozenset = frozenset({"Gemm"})
    id: str = "b1"
    # [QZ, additive] optional port seams for programs whose weights record / capture identities are not TA1's bf16 B1's
    # (port b1-fp8: e4m3 weights + f32 scales bound from the commit's stored words, `fp8_quant` capture ports, the V
    # rows in `qkv_proj.gemm` because the bias is fused): `BIND_WEIGHTS(program, circuit, tensors, weights_index, C,
    # cos_sin, cos_sin_cpu, store=) -> WeightsBinding`, `CIRCUIT_MAP(program) -> CircuitMap-like`, `V_SOURCE` (op suffix).
    bind_weights: Any = None
    v_source: str | None = None
    # optional seams a port may declare (CF2's b1-hopper-compiled; all default to TA1's eager B1 behaviour):
    program_factory: Any = None        # `program(LP=, STEPS=) -> Program` (a Serve under a static binding record)
    circuit_map: Any = None            # `CIRCUIT_MAP(program)` (QZ) or `CIRCUIT_MAP(program, names)` (CF2) -> CircuitMap-like
    kv_sources: frozenset | None = None            # {(op suffix, port)} leaves holding every position's K and V rows
    prev_layer_sources: tuple[str, ...] | None = None  # layer L-1 op suffixes needed to evaluate layer L's first call
    recompute_missing: bool = False    # recompute unmaterialised operands exactly from committed values

    @classmethod
    def load(cls, path: Path) -> "Port":
        import importlib.util
        path = Path(path)
        file = path if path.is_file() else path / "static_config.py"
        pid = path.stem if path.is_file() else path.name
        spec = importlib.util.spec_from_file_location(f"port_{pid}", file)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        kv = getattr(mod, "KV_SOURCES", None)
        prev = getattr(mod, "PREV_LAYER_SOURCES", None)
        return cls(serve=getattr(mod, "SERVE", None), config=dict(mod.CONFIG), names=dict(getattr(mod, "CALL_NAMES", {})),
                   gemm_names=frozenset(getattr(mod, "GEMM_NAMES", {"Gemm"})), id=getattr(mod, "PORT_ID", pid),
                   bind_weights=getattr(mod, "BIND_WEIGHTS", None), v_source=getattr(mod, "V_SOURCE", None),
                   program_factory=getattr(mod, "program", None) if hasattr(mod, "program") else None,
                   circuit_map=getattr(mod, "CIRCUIT_MAP", None),
                   kv_sources=frozenset(tuple(x) for x in kv) if kv is not None else None,
                   prev_layer_sources=tuple(prev) if prev is not None else None,
                   recompute_missing=bool(getattr(mod, "RECOMPUTE_MISSING", False)))


def run(store: CommitStore, checkpoint: list[Path], request: str, positions: list[int], layers: list[int],
        coord_samples: int, full: set[tuple[int, int]], out_path: Path, seed: int, LP: int | None = None,
        STEPS: int | None = None, cos_sin_npy: Path | None = None, cos_sin_cpu_npy: Path | None = None,
        config: str = "B1", program_factory=None, circuit_map=None, static_config: dict | None = None,
        port: Port | None = None) -> dict:
    """Two ways a profile port supplies its static configuration: `port` (a `Port`: the `Serve` definition, its config
    record, TA1-name -> port-name table and Gemm-shaped call names; `--port <dir>` on the command line), or CH's
    seams `program_factory(LP=, STEPS=) -> Program`, `circuit_map(program) -> CircuitMap-like` and `static_config`
    (the C dict).  Without either the registered B1/B0 configurations are used."""
    if port is None:
        C0 = dict(static_config) if static_config is not None else {"B1": b1.B1_CONFIG, "B0": b1.B0_CONFIG}[config]
        port = Port(serve=b1.Serve, config=C0, id=config)
    C = dict(port.config)  # MAX_POS = max_position_embeddings: the committed cos_sin leaf
    gemm_names = port.gemm_names
    bnd = store.boundary(request)
    prompt = bnd["prompt_token_ids"]
    lp = LP or len(prompt)
    steps = STEPS if STEPS is not None else (len(store.steps(request)) - 1)
    t0 = time.time()
    if program_factory is None and port.serve is None and port.program_factory is not None:
        program_factory = port.program_factory  # a port whose Serve is specialised under a binding record (CF2)
    if circuit_map is None and port.circuit_map is not None:
        # a port-supplied mapping: QZ's `CIRCUIT_MAP(program)` or CF2's `CIRCUIT_MAP(program, names)`
        import inspect
        _cm, _names = port.circuit_map, port.names
        try:
            _two = len([q for q in inspect.signature(_cm).parameters.values()
                        if q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]) >= 2
        except (TypeError, ValueError):
            _two = False
        circuit_map = (lambda program: _cm(program, _names)) if _two else _cm
    program = program_factory(LP=lp, STEPS=steps) if program_factory is not None else Program(bind(port.serve, C=C, LP=lp, STEPS=steps))
    circuit = program.circuit
    cmap = circuit_map(program) if circuit_map is not None else CircuitMap(program, port.names)
    if getattr(cmap, "names", None):  # a port mapping carries its own name table: Gemm-shaped calls go by port names
        gemm_names = gemm_names | frozenset(cmap.names.get(g, g) for g in gemm_names)
    recomputer = Recomputer(circuit) if port.recompute_missing else None
    build_s = time.time() - t0
    print(f"[corr] program built: {program.gates:,} gates, {program.input_gates:,} Input gates, {len(cmap.named)} named calls, {build_s:.1f}s", flush=True)

    tensors = load_safetensors(checkpoint)
    t0 = time.time()
    cs = (np.load(cos_sin_npy), "cuda") if cos_sin_npy else None
    cs_cpu = np.load(cos_sin_cpu_npy) if cos_sin_cpu_npy else None
    if port.bind_weights is not None:  # [QZ, additive] port-supplied weights binder (see Port)
        wb = port.bind_weights(program, circuit, tensors, store.weights_index, C, cs, cs_cpu, store=store)
    else:
        wb = bind_weights(program, circuit, tensors, store.weights_index, C, cs, cs_cpu)
    print(f"[corr] weights bound: {len(wb.records)} leaves, sha256 ok={sum(r.get('sha256_matches_leaf', False) for r in wb.records)}, "
          f"not-in-program={sum('not_in_program' in r for r in wb.records)}, {time.time() - t0:.1f}s", flush=True)

    # prompt tokens: Input gates of parameter 0, authenticated by the boundary leaf
    pcoll = program.input_coll(0)
    pstart, pend = _coll_range(circuit, pcoll)
    prompt_seg = Segments()
    prompt_seg.add(pstart, np.asarray(prompt, dtype=np.uint32), {"namespace": "authentication", "commitment": f"boundary/{request}",
                                                                  "leaf_id": "prompt_token_ids"})
    prompt_record = {"value_identity": {"input_gates": [pstart, pend], "gate_class": "Input/prompt", "scalars": pend - pstart},
                     "authentication_location": {"commitment": f"boundary/{request}", "leaf_id": "prompt_token_ids",
                                                 "root": store.commitments["trees"][f"boundary/{request}"]["root"]}}

    cap = Segments()
    located: dict[str, tuple[FunctionCall, tuple[int, ...], dict]] = {}
    not_in_program: list[dict] = []
    layer_set = set(layers)
    # positions whose K/V rows are needed: everything up to max(positions) for the selected layers
    t0 = time.time()
    kv_positions = list(range(0, max(positions) + 1))
    v_source = "qkv_proj.add" if C.get("BIAS", True) else "qkv_proj.gemm"  # the leaf holding the V rows of every position
    if port.v_source:  # [QZ, additive] e.g. b1-fp8: the bias is fused into the scaled GEMM, no `.add` leaf exists
        v_source = port.v_source
    # which leaves hold every position's K and V rows, and which layer L-1 leaves feed layer L's first call: TA1's eager
    # partition by default; a port with a different capture partition (compiled) declares its own
    kv_sources = port.kv_sources if port.kv_sources is not None else frozenset({("rotary_emb.rope", 1), (v_source, 0)})
    prev_sources = port.prev_layer_sources if port.prev_layer_sources is not None else ("down_proj.gemm", "post_attention_layernorm.rmsnorm")
    for t in kv_positions:
        step, entries = store.entries_at(request, t)
        root = store.tree_root(request, step)
        for e in entries:
            ident = CaptureIdentity.parse(e["canonical_id"])
            keep = t in positions or (ident.layer.isdigit() and int(ident.layer) in layer_set and
                                      any(ident.op.endswith(s) and ident.output == p for s, p in kv_sources))
            keep = keep or (ident.layer == "sampler")  # decode tokens feed the next position's Embedding
            if not keep:
                continue
            if ident.layer.isdigit() and int(ident.layer) not in layer_set and not (
                    int(ident.layer) + 1 in layer_set and any(ident.op.endswith(s) for s in prev_sources)):
                continue
            try:
                call, gates = cmap.locate(ident)
            except NotInProgram as ex:
                not_in_program.append({"capture_id": e["canonical_id"], "reason": str(ex),
                                       "authentication_location": {"commitment": f"ru/{request}/{step:06d}", "leaf": e["leaf"]}})
                continue
            words = store.leaf_words(request, step, e)
            if len(gates) != words.size:
                raise ValueError(f"{e['canonical_id']}: {len(gates)} gates vs {words.size} words")
            loc = {"namespace": "authentication", "commitment": f"ru/{request}/{step:06d}", "root": root,
                   "leaf_id": e["canonical_id"], "leaf": e["leaf"], "offset": e["offset"], "nbytes": e["nbytes"]}
            located[e["canonical_id"]] = (call, gates, {**e, "step": step, "root": root})
            run_start = 0
            for i in range(1, len(gates) + 1):
                if i == len(gates) or gates[i] != gates[i - 1] + 1:
                    cap.add(gates[run_start], words[run_start:i], {**loc, "word_base": run_start})
                    run_start = i
    print(f"[corr] captured leaves located: {len(located)} ({len(cap)} segments), {time.time() - t0:.1f}s", flush=True)

    transcript = LazyTranscript(wb.segments, prompt_seg, cap)
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    summary = {"exact": 0, "mismatch": 0, "missing-operand": 0, "sampled": 0, "shape_mismatch": 0}
    for cid, (call, gates, e) in sorted(located.items(), key=lambda kv: kv[1][1][0]):
        ident = CaptureIdentity.parse(cid)
        if ident.t not in positions:
            continue
        if ident.layer.isdigit() and int(ident.layer) not in layer_set:
            continue
        ok, why = _shape_ok(call, gates, e)
        rec = {"capture_id": cid, "value_identity": value_identity(call, gates),
               "authentication_location": {"namespace": "authentication", "commitment": f"ru/{request}/{e['step']:06d}", "root": e["root"],
                                           "leaf": e["leaf"], "offset": e["offset"], "nbytes": e["nbytes"], "dtype": e["dtype"], "shape": e["shape"]},
               "shape_type_consistent": ok, "shape_type_detail": why}
        captured = store.leaf_words(request, e["step"], e)
        if not ok:
            summary["shape_mismatch"] += 1
            rec["check"] = {"status": "shape-mismatch"}
        elif call.name in gemm_names and (ident.t, int(ident.layer) if ident.layer.isdigit() else -1) not in full:
            # coordinate-sampled: evaluate `coord_samples` GemmCoordinate members (plus, for lm_head, the argmax coordinate)
            kids = list(call.children())
            n = len(kids)
            pick = sorted(set(int(x) for x in rng.choice(n, size=min(coord_samples, n), replace=False)))
            if ident.layer == "lm_head":
                argmax = int(np.frombuffer(store.leaf_words(request, e["step"], next(
                    x for x in store.ru(request, e["step"])[0]["entries"] if x["canonical_id"] == cid.replace("l=lm_head/op=lm_head.gemm", "l=sampler/op=sampler.select"))).tobytes(), dtype=np.int32)[0])
                pick = sorted(set(pick) | {argmax})
            checks = []
            t1 = time.time()
            for i in pick:
                sub = kids[i]
                r = check_call(call, gates, transcript, captured, sub=sub, recompute=recomputer)
                checks.append({"coordinate": i, **{k: r[k] for k in ("status", "mismatches")}})
            statuses = {c["status"] for c in checks}
            rec["check"] = {"status": "exact" if statuses == {"exact"} else ("mismatch" if "mismatch" in statuses else "missing-operand"),
                            "mode": "coordinate-sampled", "coordinates": len(pick), "of": n,
                            "words_compared": sum(1 for c in checks if c["status"] == "exact"),
                            "failed_coordinates": [c for c in checks if c["status"] != "exact"][:16], "seconds": round(time.time() - t1, 3)}
            summary["sampled"] += 1
        else:
            rec["check"] = check_call(call, gates, transcript, captured, recompute=recomputer)
        summary[rec["check"]["status"]] = summary.get(rec["check"]["status"], 0) + 1
        records.append(rec)
        print(f"[corr] {rec['check']['status']:16s} {cid.split('/ph=')[0].split('/r=')[1][-8:]}/{cid.split('/', 4)[-1]:70s} {call.id[:48]:48s} {rec['check'].get('seconds', 0):8.2f}s", flush=True)
        out = {"schema": "verity-ir/correspondence/v0", "config": port.id, "model": C["model"], "request": request, "positions": positions, "layers": layers,
               "program": {"function": program.fn.id, "gates": program.gates,
                           "input_gates": program.input_gates, "build_seconds": round(build_s, 1),
                           "call_names": port.names or None, "recompute_missing": recomputer.stats if recomputer else None},
               "input_gate_records": {"prompt": prompt_record, "weights": wb.records},
               "coordinate_sample": {"seed": seed, "per_gemm": coord_samples}, "not_in_program": not_in_program, "records": records,
               "summary": {**summary, "qualifications": summary_qualifications(wb.records, records, summary)}}
        out_path.write_text(json.dumps(out, indent=1))
    return out


@dataclass(frozen=True, kw_only=True)
class Options:
    requalify: list[Path] | None = option("--requalify", type=Path, nargs="+", default=None,
                                          help="rewrite existing corr-*.json records in place with the explicit cos_sin_cache qualification and summary.qualifications")
    commit: Path | None = option("--commit", default=None, type=Path)
    checkpoint: list[Path] | None = option("--checkpoint", default=None, type=Path, nargs="+", help="safetensors shard(s)")
    request: str | None = option("--request", default=None)
    positions: list[int] | None = option("--positions", default=None, type=int, nargs="+")
    layers: list[int] = option("--layers", type=int, nargs="+", default=[0, 1, 13, 27])
    coord_samples: int = option("--coord-samples", type=int, default=32)
    full: list[str] = option("--full", nargs="*", default=[],
                             help="t,L pairs whose Gemms are evaluated in full (L=-1: lm_head)")
    out: Path | None = option("--out", default=None, type=Path)
    seed: int | None = option("--seed", type=int, default=None, help="the coordinate sample's seed (required)")
    cos_sin_npy: Path | None = option("--cos-sin-npy", type=Path, default=None,
                                      help="precomputed CUDA cos_sin_cache words (uint16 [MAX_POS,128])")
    cos_sin_cpu_npy: Path | None = option("--cos-sin-cpu-npy", type=Path, default=None)
    config: str = option("--config", choices=["B1", "B0"], default="B1")
    port: Path | None = option("--port", type=Path, default=None,
                               help="profile-port directory (vllm-poc/profiles/<port>/static_config.py: SERVE, CONFIG, CALL_NAMES, GEMM_NAMES); overrides --config")


def main(a: Options) -> int:
    if a.requalify:
        for p in a.requalify:
            d = requalify(p)
            print(f"{p}: summary {d['summary']}")
        return 0
    if not (a.commit and a.checkpoint and a.request and a.positions and a.out) or a.seed is None:
        raise UsageError("--commit, --checkpoint, --request, --positions, --out and --seed are required (or use --requalify)")
    full = {tuple(int(x) for x in s.split(",")) for s in a.full}
    port = Port.load(a.port) if a.port else None
    run(CommitStore(a.commit), a.checkpoint, a.request, a.positions, a.layers, a.coord_samples, full, a.out, a.seed,  # type: ignore[arg-type]
        cos_sin_npy=a.cos_sin_npy, cos_sin_cpu_npy=a.cos_sin_cpu_npy, config=a.config, port=port)
    return 0
