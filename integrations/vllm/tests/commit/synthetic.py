"""A numpy toy "model" whose capture proves the harness end to end without vLLM.

Per position ``t`` with input token ``x_t``::

    h0 = E[x_t]                                   embed_tokens        (bfloat16 [H])
    for l in layers:  g = bf16(h @ W_l)           layers.l.mlp.gemm   (bfloat16 [H])
                      h = bf16(h + g)             layers.l.residual   (bfloat16 [H])
    logits = bf16(h @ W_lm)                       lm_head.gemm        (bfloat16 [V], sampled t)
    token  = argmax(logits) (lowest index)        sampler.select      (int32 [],  sampled t)

The GEMM semantics are deliberately trivial and exact: every BF16 product is exact in float64,
the K-term sum is exact in float64 for the value ranges used, then one rounding to float32 and
one RNE to BF16.  The relation ``gemm.toy.f32acc.v1`` registered here implements exactly that and
is labelled TOY ONLY -- it is not a model of any production kernel; the production ``gemm``
slots stay unimplemented until W4 fills them.

:func:`build_toy_run` writes a complete capture (manifest, weights, boundaries, replay units,
description.json, required.json, commitments.json) and returns a :class:`ToyRun`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.commit import merkle as commit, identity
from verity_vllm.program import dtypes
from verity_vllm.observe.capture_v1 import Boundary, CaptureReader, Entry, RunWriter, StepRecord
from verity_vllm.check.poc_description import Description, NodeInstance
from verity_vllm.program.kernels.relations import INDEPENDENT, Inputs, RelationError, need, register
from verity_vllm.check.poc_required_interface import freeze_required

TOY_MANIFEST_BASE: dict[str, Any] = {
    "schema": "verity-vllm/profile/v1",
    "model": {"repo": "verity-vllm/toy", "revision": "toy-v1", "weights_sha256": {}},
    "vllm": {"version": "none", "wheel_sha256": None, "commit": None},
    "python": {"version": "3", "torch": None, "cuda": None, "triton": None, "flashinfer": None},
    "gpu": {"name": "cpu", "count": 0, "driver": None, "compute_capability": None, "pod": None},
    "dtype": "bfloat16",
    "quantization": None,
    "backends": {"attention": None, "linear": "numpy-toy", "moe": None, "sampler": "argmax"},
    "parallelism": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
    "compilation": {"mode": "none", "cudagraph_mode": "none", "enforce_eager": True,
                    "model_runner": "toy"},
    "engine_flags": {},
    "sampling": {"greedy": True, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "seeds": []},
    "adapter": {"revision": "synthetic", "interception": "numpy"},
}  # fmt: skip

TOY_PARTITION = {"partition_id": "toy-row", "decomposition": {"default": "row"}}


def toy_gemm_f32(x_bf16: np.ndarray, w_bf16: np.ndarray) -> np.ndarray:
    """float32 result of the toy GEMM: exact float64 products and sums, one rounding to f32."""
    x = dtypes.bf16_bits_to_f32(x_bf16).astype(np.float64)
    w = dtypes.bf16_bits_to_f32(w_bf16).astype(np.float64)
    acc = np.zeros(w.shape[1], dtype=np.float64)
    for k in range(w.shape[0]):
        acc += x[k] * w[k]
    return acc.astype(np.float32)


def toy_gemm_bf16(x_bf16: np.ndarray, w_bf16: np.ndarray) -> np.ndarray:
    return dtypes.f32_to_bf16_bits(toy_gemm_f32(x_bf16, w_bf16))


def toy_add_bf16(a_bf16: np.ndarray, b_bf16: np.ndarray) -> np.ndarray:
    return dtypes.f32_to_bf16_bits(
        dtypes.bf16_bits_to_f32(a_bf16) + dtypes.bf16_bits_to_f32(b_bf16)
    )


@register(
    "gemm.toy.f32acc.v1",
    klass=INDEPENDENT,
    op_type="gemm",
    includes_full_op=False,
    doc="TOY ONLY: x @ W with exact float64 products/sums, one rounding to float32, RNE to BF16. "
    "Not a model of any production kernel.",
)
def gemm_toy(inst: NodeInstance, inputs: Inputs) -> bytes:
    x = need(inputs, "x")
    w = need(inputs, "w")
    if x.dtype != "bfloat16" or w.dtype != "bfloat16":
        raise RelationError("gemm.toy.f32acc.v1 needs bfloat16 inputs")
    if len(w.shape) != 2 or x.shape != (w.shape[0],):
        raise RelationError(f"shape mismatch x{list(x.shape)} @ w{list(w.shape)}")
    return dtypes.array_to_bytes(toy_gemm_bf16(x.array(), w.array()))


TOY_RELATIONS = ("gemm.toy.f32acc.v1",)


def install() -> list[str]:
    """Plugin entry point for the fresh verifier: ``--plugin tests.commit.synthetic:install``.

    The toy relation is registered when this module is imported; the verifier process imports
    nothing but the bundle, so it must be told to load this module to check a toy bundle.
    """
    return list(TOY_RELATIONS)


def _small_bf16(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    """Random BF16 bit patterns of modest magnitude (keeps every toy sum exact in float64)."""
    values = rng.uniform(-scale, scale, size=shape).astype(np.float32)
    return dtypes.f32_to_bf16_bits(values)


def toy_description(profile_id: str, layers: int, hidden: int, vocab: int) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
        {
            "op": "embed_tokens",
            "layer": "embed",
            "op_type": "embedding",
            "dtype": "bfloat16",
            "shape": ["H"],
            "args": [
                {"name": "token", "source": "token", "which": "input"},
                {"name": "table", "source": "weight", "param": "embed_tokens.weight"},
            ],
            "relation": "embedding.row.v1",
            "origin": "captured",
            "decompositions": [
                {"id": "row", "interface": {"input_bytes": 4, "output_bits": 16 * hidden}}
            ],
            "runtime_impl": "numpy-toy",
        }
    ]  # fmt: skip
    prev = "embed_tokens"
    for layer in range(layers):
        gemm = f"layers.{layer}.mlp.gemm"
        residual = f"layers.{layer}.residual"
        nodes.append(
            {
                "op": gemm,
                "layer": str(layer),
                "op_type": "gemm",
                "dtype": "bfloat16",
                "shape": ["H"],
                "args": [
                    {"name": "x", "source": "node", "op": prev, "port": 0},
                    {"name": "w", "source": "weight", "param": f"layers.{layer}.mlp.weight"},
                ],
                "relation": "gemm.toy.f32acc.v1",
                "origin": "captured",
                "decompositions": [
                    {
                        "id": "row",
                        "interface": {
                            "input_bytes": 2 * hidden * (hidden + 1),
                            "output_bits": 16 * hidden,
                        },
                    },
                    {
                        "id": "coordinate",
                        "interface": {"input_bytes": 4 * hidden, "output_bits": 16},
                    },
                ],
                "runtime_impl": "numpy-toy",
            }  # fmt: skip
        )
        nodes.append(
            {
                "op": residual,
                "layer": str(layer),
                "op_type": "residual_add",
                "dtype": "bfloat16",
                "shape": ["H"],
                "args": [
                    {"name": "a", "source": "node", "op": prev, "port": 0},
                    {"name": "b", "source": "node", "op": gemm, "port": 0},
                ],
                "relation": "residual_add.bf16.v1",
                "origin": "captured",
                "decompositions": [
                    {
                        "id": "row",
                        "interface": {"input_bytes": 4 * hidden, "output_bits": 16 * hidden},
                    }
                ],
                "runtime_impl": "numpy-toy",
            }  # fmt: skip
        )
        prev = residual
    nodes.append(
        {
            "op": "lm_head.gemm",
            "layer": "lm_head",
            "op_type": "lm_head",
            "dtype": "bfloat16",
            "shape": ["V"],
            "positions": "sampled",
            "args": [
                {"name": "x", "source": "node", "op": prev, "port": 0},
                {"name": "w", "source": "weight", "param": "lm_head.weight"},
            ],
            "relation": "gemm.toy.f32acc.v1",
            "origin": "captured",
            "decompositions": [
                {
                    "id": "row",
                    "interface": {
                        "input_bytes": 2 * hidden * (vocab + 1),
                        "output_bits": 16 * vocab,
                    },
                }
            ],
            "runtime_impl": "numpy-toy",
        }  # fmt: skip
    )
    nodes.append(
        {
            "op": "sampler.select",
            "layer": "sampler",
            "op_type": "sampler",
            "dtype": "int32",
            "shape": [],
            "positions": "sampled",
            "args": [
                {"name": "logits", "source": "node", "op": "lm_head.gemm", "port": 0},
                {"name": "params", "source": "boundary", "field": "sampling_params"},
            ],
            "relation": "argmax.lowest_index.v1",
            "origin": "captured",
            "decompositions": [
                {"id": "row", "interface": {"input_bytes": 2 * vocab, "output_bits": 32}}
            ],
            "runtime_impl": "numpy-argmax",
            # the pinned §31.2/§32.5 token-output convention: 16-bit declared domain inside the
            # int32 container; the verifier's linkage check enforces high word == 0 on every audit
            "declared_domain": 16,
        }  # fmt: skip
    )
    return {
        "schema": "verity-vllm/description/v1",
        "profile_id": profile_id,
        "model": "verity-vllm/toy",
        "symbols": {"H": hidden, "V": vocab, "L": layers},
        "nodes": nodes,
    }


@dataclass(slots=True)
class ToyRun:
    run_dir: Path
    profile_id: str
    description: Description
    required: dict[str, Any]
    commitments: dict[str, Any]
    weights: dict[str, np.ndarray]
    boundaries: dict[str, Boundary]

    def reader(self) -> CaptureReader:
        return CaptureReader(self.run_dir)

    def value_id(self, request_id: str, t: int, op: str, port: int = 0) -> str:
        boundary = self.boundaries[request_id]
        step = boundary.step_of(t)
        if step is None:
            raise KeyError(f"position {t} was never processed for {request_id}")
        node = self.description.node(op)
        return str(
            identity.value_id(self.profile_id, request_id, t, step.phase, node.layer, node.op, port)
        )


def build_toy_run(
    run_dir: str | os.PathLike[str],
    *,
    seed: int = 0,
    requests: int = 2,
    prompt_len: int = 4,
    gen: int = 3,
    layers: int = 2,
    hidden: int = 8,
    vocab: int = 32,
    store_weights: bool = True,
    chunked_prefill: bool = False,
) -> ToyRun:
    """Write a complete toy capture and its commitments; deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    weights = {
        "embed_tokens.weight": _small_bf16(rng, (vocab, hidden), 1.0),
        **{
            f"layers.{layer}.mlp.weight": _small_bf16(rng, (hidden, hidden), 0.5)
            for layer in range(layers)
        },
        "lm_head.weight": _small_bf16(rng, (hidden, vocab), 1.0),
    }  # fmt: skip
    weights_blob = b"".join(dtypes.array_to_bytes(w) for w in weights.values())
    manifest = dict(TOY_MANIFEST_BASE)
    manifest["model"] = {
        "repo": "verity-vllm/toy",
        "revision": f"toy-v1-seed{seed}",
        "weights_sha256": {"weights/values.bin": commit.hashing.sha256_hex(weights_blob)},
    }
    manifest["toy"] = {"layers": layers, "hidden": hidden, "vocab": vocab, "seed": seed}
    run = RunWriter(run_dir, manifest)
    pid = run.profile_id

    ww = run.weights(store_values=store_weights)
    for name, array in weights.items():
        ww.add(name, "bfloat16", list(array.shape), array)
    ww.finish()

    description = Description(toy_description(pid, layers, hidden, vocab))
    boundaries: dict[str, Boundary] = {}

    def forward(token: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        values: dict[str, np.ndarray] = {}
        h = weights["embed_tokens.weight"][token].copy()
        values["embed_tokens"] = h
        for layer in range(layers):
            g = toy_gemm_bf16(h, weights[f"layers.{layer}.mlp.weight"])
            values[f"layers.{layer}.mlp.gemm"] = g
            h = toy_add_bf16(h, g)
            values[f"layers.{layer}.residual"] = h
        logits = toy_gemm_bf16(h, weights["lm_head.weight"])
        return values, logits

    for r in range(requests):
        request_id = f"req-{r}"
        prompt = [int(x) for x in rng.integers(1, vocab, size=prompt_len)]
        # plan the steps: one prefill (or two chunks) then one decode per further output
        steps: list[StepRecord] = []
        if chunked_prefill and prompt_len >= 2:
            half = prompt_len // 2
            steps.append(StepRecord(0, "prefill", tuple(range(half))))
            steps.append(StepRecord(1, "prefill", tuple(range(half, prompt_len))))
        else:
            steps.append(StepRecord(0, "prefill", tuple(range(prompt_len))))
        outputs: list[int] = []
        # request 0 stops early on a "stop token" so its record list has blank slots (N14)
        max_tokens = gen + 2 if r == 0 else gen
        per_position: dict[int, tuple[dict[str, np.ndarray], np.ndarray | None]] = {}
        tokens = list(prompt)
        for t in range(prompt_len):
            values, logits = forward(tokens[t])
            per_position[t] = (values, logits if t == prompt_len - 1 else None)
        while len(outputs) < gen:
            t = prompt_len - 1 + len(outputs)
            logits = per_position[t][1]
            assert logits is not None
            token = int(np.argmax(dtypes.bf16_bits_to_f32(logits)))
            outputs.append(token)
            if len(outputs) == gen:
                break
            next_t = t + 1
            steps.append(StepRecord(len(steps), "decode", (next_t,)))
            values, logits = forward(token)
            per_position[next_t] = (values, logits)
        if r == 0:
            finish_reason = "stop"
            sampling = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": max_tokens,
                        "seed": None, "stop_token_ids": [outputs[-1]]}  # fmt: skip
        else:
            finish_reason = "length"
            sampling = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": max_tokens,
                        "seed": None}  # fmt: skip
        boundary = Boundary.build(
            request_id, prompt, outputs, sampling, seed=None, finish_reason=finish_reason,
            steps=steps, max_tokens=max_tokens,
        )  # fmt: skip
        boundaries[request_id] = boundary
        run.write_boundary(boundary)
        sampled = set(boundary.sampled_positions())
        for step in steps:
            writer = run.ru(request_id, step.step, phase=step.phase, positions=step.positions)
            for t in step.positions:
                values, logits = per_position[t]
                for op, array in values.items():
                    node = description.node(op)
                    cid = str(identity.value_id(pid, request_id, t, step.phase, node.layer, op, 0))
                    writer.add(cid, "bfloat16", list(array.shape), array)
                if t in sampled:
                    assert logits is not None
                    cid = str(
                        identity.value_id(
                            pid, request_id, t, step.phase, "lm_head", "lm_head.gemm", 0
                        )
                    )
                    writer.add(cid, "bfloat16", [vocab], logits)
                    cid = str(
                        identity.value_id(
                            pid, request_id, t, step.phase, "sampler", "sampler.select", 0
                        )
                    )
                    token = boundary.sampled_token_at(t)
                    assert token is not None
                    writer.add(cid, "int32", [], np.array(token, np.int32))
            writer.finish()

    run.write_description(description.to_json())
    required = freeze_required(Path(run_dir) / "required.json", description, TOY_PARTITION)
    run.finish()
    commitments = commit.build_commitments(run_dir)
    return ToyRun(Path(run_dir), pid, description, required, commitments, weights, boundaries)


def corrupt_committed_value(
    run_dir: str | os.PathLike[str],
    canonical_id: str,
    mutate: Any = None,
) -> tuple[bytes, bytes]:
    """Make an on-disk *wrong-result bundle*: rewrite one committed value and rebuild everything
    consistently (values.bin, the index leaf hash, commitments.json), so authentication succeeds
    and only the relation can reject it (N02 on disk; what the fresh verifier must turn red).

    ``mutate(bytes) -> bytes`` defaults to flipping the lowest bit of the first byte.  Returns
    ``(old_bytes, new_bytes)``.
    """
    from verity_vllm.observe.capture_v1 import INDEX_FILE, VALUES_FILE, _write_json
    from verity_vllm.commit.hashing import leaf_hash

    run_dir = Path(run_dir)
    reader = CaptureReader(run_dir)
    where = reader.locate(canonical_id)
    if where is None:
        raise KeyError(f"{canonical_id} is not committed in {run_dir}")
    request_id, step = where
    ru = reader.ru(request_id, step)
    entry = ru.entry(canonical_id)
    old = ru.read(canonical_id)
    if mutate is None:
        buf = bytearray(old)
        buf[0] ^= 0x01
        new = bytes(buf)
    else:
        new = bytes(mutate(old))
    if len(new) != len(old):
        raise ValueError("mutation must keep the byte length")
    values_path = ru.dir / VALUES_FILE
    blob = bytearray(values_path.read_bytes())
    blob[entry.offset : entry.offset + entry.nbytes] = new
    values_path.write_bytes(bytes(blob))
    index = ru.index
    index.entries = [
        Entry(
            e.canonical_id,
            e.dtype,
            e.shape,
            e.offset,
            e.nbytes,
            e.origin,
            leaf_hash(e.canonical_id, e.dtype, e.shape, new).hex(),
        )
        if e.canonical_id == canonical_id
        else e
        for e in index.entries
    ]
    _write_json(ru.dir / INDEX_FILE, index.to_json())
    commit.build_commitments(run_dir)
    return old, new
