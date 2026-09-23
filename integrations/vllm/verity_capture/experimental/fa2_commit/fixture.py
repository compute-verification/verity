"""Replay fixtures: attention inputs (q / k / v as bf16 bit patterns) + the static context the layouts rank against.

Two sources:
  * :func:`synthetic_fixture` -- seeded random bf16 inputs for a chosen geometry (mechanics tests);
  * :func:`archived_geometry_fixture` -- the REAL geometry of one (layer, step) of the accepted record
    `B0-cov_b0_b1_c256` read from its compact `instances.jsonl` (every `Attention_v3{T=...}` row: node id, query position,
    visible keys T, NH/KVH/D/BN), with SYNTHETIC values: the compact record stores runtime tensor VIEWS
    (`operand_views`: alloc id / offset / shape / strides), not the tensor bytes, so q/k/v of the run are not recoverable
    from it (stated in the fixture as ``values="synthetic"``).

Context (:class:`Context`) = Program digest, request id, step, prompt length, layer, phase -- the runtime data an opener
needs (C0 §C0.3: request id / prompt length are per-record metadata, not arithmetic).  The row set of a (layer, step) is
derived from it exactly as C0's `RUContext.positions()`: prefill -> qrows [0, P); decode step s -> qrow P + s - 1.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .encoding import Geometry


@dataclass(frozen=True, slots=True)
class Context:
    program_digest: bytes      # 32 B (the Program's fold digest)
    request_id: str
    step: int
    prompt_len: int
    layer: int
    # prefill chunk: the step's query rows are [chunk_lo, prompt_len) (vLLM chunked prefill / the kernel's bottom-right
    # aligned causal mask with seqlen_q < seqlen_k; 0 = the whole prompt, as in the accepted record)
    chunk_lo: int = 0

    @property
    def phase(self) -> str:
        return "prefill" if self.step == 0 else "decode"

    def qrows(self) -> range:
        if self.step == 0:
            return range(self.chunk_lo, self.prompt_len)
        return range(self.prompt_len + self.step - 1, self.prompt_len + self.step)

    def digest(self) -> bytes:
        doc = {"program": self.program_digest.hex(), "request": self.request_id, "step": self.step,
               "prompt_len": self.prompt_len, "layer": self.layer, "chunk_lo": self.chunk_lo}
        return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).digest()


@dataclass
class AttentionFixture:
    """Inputs of every Attention Instance of one (layer, step): q per (qrow, head), k/v per (position, kv head)."""

    ctx: Context
    geo: Geometry
    q16: np.ndarray            # (n_rows, NH, D) uint16, indexed by qrow - qrows[0]
    k16: np.ndarray            # (T_max, KVH, D) uint16   (KV cache contents, positions 0..T_max-1)
    v16: np.ndarray            # (T_max, KVH, D) uint16
    values: str = "synthetic"  # "synthetic" | "recorded"
    source: dict = field(default_factory=dict)
    # the deterministic reference that regenerates one row's committed stream: `reference.replay_row` (fast numpy
    # approximation, mechanics) or `oracle.oracle_row` (the registry Definition evaluated by the IR oracle: Verity semantics)
    replay: Callable[..., tuple[np.ndarray, np.ndarray | None]] = field(default=None, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.replay is None:
            from .reference import replay_row

            object.__setattr__(self, "replay", replay_row)

    def with_replay(self, fn: Callable[..., tuple[np.ndarray, np.ndarray | None]]) -> "AttentionFixture":
        return dataclasses.replace(self, replay=fn)

    def row_inputs(self, head: int, qrow: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self.ctx.qrows()
        g = head // (self.geo.NH // self.geo.KVH)
        T = qrow + 1
        return self.q16[qrow - rows[0], head], self.k16[:T, g], self.v16[:T, g]


def _random_bf16(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    from .reference import f32_to_bf16_rne

    return f32_to_bf16_rne((rng.standard_normal(shape) * scale).astype(np.float32))


def synthetic_fixture(*, seed: int, geo: Geometry, layer: int, step: int, prompt_len: int,
                      request_id: str = "req-synth", program_digest: bytes | None = None, scale: float = 1.0) -> AttentionFixture:
    ctx = Context(program_digest or hashlib.sha256(b"synthetic-program").digest(), request_id, step, prompt_len, layer)
    rng = np.random.default_rng(seed)
    rows = ctx.qrows()
    T_max = rows[-1] + 1
    q16 = _random_bf16(rng, (len(rows), geo.NH, geo.D), scale)
    k16 = _random_bf16(rng, (T_max, geo.KVH, geo.D), scale)
    v16 = _random_bf16(rng, (T_max, geo.KVH, geo.D), scale)
    return AttentionFixture(ctx, geo, q16, k16, v16, "synthetic", {"seed": seed})


_SPEC_KEYS = ("T", "NH", "KVH", "D", "BN")


def _parse_spec(spec: str) -> dict[str, int]:
    inner = spec[spec.index("{") + 1: spec.rindex("}")]
    return {k: int(v) for k, v in (kv.split("=") for kv in inner.split(","))}


def archived_geometry(instances_jsonl: str | Path, *, layer: int, step: int) -> dict:
    """Every `attention.fa2_varlen_fwd` Instance of (layer, step) in a compact record: geometry only, plus provenance."""
    path = Path(instances_jsonl)
    h = hashlib.sha256()
    rows: list[dict] = []
    header: dict | None = None
    with path.open("rb") as f:
        for raw in f:
            h.update(raw)
            if header is None:
                header = json.loads(raw)
                continue
            if b'"pattern":"attention.fa2_varlen_fwd"' not in raw:
                continue
            rec = json.loads(raw)
            if rec.get("step") != step or rec.get("hint", "").split("/")[0] != f"model.layers.{layer}.self_attn.attn":
                continue
            spec = _parse_spec(rec["spec"])
            t = int(rec["name"].split("/")[0].split("=")[1])
            rows.append({"node": rec["node"], "qrow": t, **spec, "q_view": rec["operand_views"][0],
                         "kv_view_shape": rec["operand_views"][1]["shape"]})
    rows.sort(key=lambda r: r["qrow"])
    return {"file": str(path), "sha256": h.hexdigest(), "program_digest": (header or {}).get("digest"),
            "profile": (header or {}).get("profile"), "layer": layer, "step": step, "rows": rows}


GEOMETRY_JSON = Path(__file__).with_name("fixtures") / "b0_c256_attention_geometry.json"


def archived_geometry_fixture(instances_jsonl: str | Path | None, *, layer: int, step: int, seed: int = 0,
                              request_id: str = "B0-cov_b0_b1_c256") -> AttentionFixture:
    """Real (layer, step) geometry of the accepted record with synthetic bf16 values (the record has no tensor bytes).
    With ``instances_jsonl=None`` the geometry comes from the committed extract :data:`GEOMETRY_JSON`."""
    if instances_jsonl is None:
        doc = json.loads(GEOMETRY_JSON.read_text())
        g = doc[f"layer{layer}_step{step}"]
    else:
        g = archived_geometry(instances_jsonl, layer=layer, step=step)
    if not g["rows"]:
        raise ValueError(f"no attention instances for layer {layer} step {step} in {instances_jsonl}")
    r0 = g["rows"][0]
    geo = Geometry(D=r0["D"], BN=r0["BN"], QB=64, NH=r0["NH"], KVH=r0["KVH"])
    for r in g["rows"]:
        if r["T"] != r["qrow"] + 1:
            raise ValueError(f"record row {r}: T != qrow + 1 (the fixture assumes one request, no prefix cache)")
    qrows = [r["qrow"] for r in g["rows"]]
    if step == 0:
        prompt_len = len(qrows)
        if qrows != list(range(prompt_len)):
            raise ValueError("prefill rows are not 0..P-1")
    else:
        if len(qrows) != 1:
            raise ValueError("a decode step has one attention row per layer")
        prompt_len = qrows[0] + 1 - step
    fx = synthetic_fixture(seed=seed, geo=geo, layer=layer, step=step, prompt_len=prompt_len, request_id=request_id,
                           program_digest=bytes.fromhex(g["program_digest"]))
    fx.values = "synthetic"
    fx.source = {"kind": "archived-geometry", "instances_jsonl": g["file"], "instances_sha256": g["sha256"],
                 "profile": g["profile"], "n_rows": len(qrows), "nodes": [r["node"] for r in g["rows"]][:4] + ["..."],
                 "note": "compact record carries runtime tensor views (alloc/offset), not values; q/k/v are seeded synthetic"}
    return fx
