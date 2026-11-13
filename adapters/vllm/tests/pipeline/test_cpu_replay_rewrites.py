"""`cpu_replay.OnRead` and `open_weights` over a checkpoint vLLM rewrites at load: a block-FP8 linear, served as the ue8m0
requant of the checkpoint's words, opens against its committed tensor root only when composed by the Build's roles, whole or as
a TP rank's shard, and a tensor read on demand is the tensor `open_weights` opens."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import verity_vllm.pipeline.cpu_replay as CPU
from tests.pipeline.test_cpu_replay import weights_tree
from verity_vllm.check import fp8_ue8m0 as FU
from verity_vllm.check import weights_of_record as WR

G, ROWS, COLS = 128, 256, 256
LINEAR = "model.layers.0.self_attn.q_b_proj"
W, S, NORM = f"{LINEAR}.weight", f"{LINEAR}.weight_scale_inv", "model.norm.weight"
PREFIX = "inner.model."
ROLES = {W: {"role": "fp8_w_requant_ue8m0"}, S: {"role": "fp8_ws_requant_ue8m0"}, NORM: {"role": "rmsnorm.weight"}}
WHOLE, RANK_1 = None, {"world": 2, "rank": 1}
Served = dict[str, tuple[str, np.ndarray]]


def safetensors(path: Path, tensors: Served) -> None:
    """A safetensors file of ``tensors`` (name -> (safetensors dtype, array))."""
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    for name, (dtype, a) in tensors.items():
        at = sum(map(len, blobs))
        blobs.append(np.ascontiguousarray(a).tobytes())
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [at, at + len(blobs[-1])]}
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


@pytest.fixture
def world(tmp_path: Path) -> SimpleNamespace:
    """A checkpoint of one block-FP8 linear with its scales and a norm, and the Build's roles for them by registered name."""
    rng = np.random.default_rng(11)
    w = rng.integers(0, 256, (ROWS, COLS), dtype=np.uint8)
    w = np.where((w & 0x7F) == 0x7F, w & 0xF0, w).astype(np.uint8)            # no NaN words
    s = rng.uniform(1e-4, 3e-3, (ROWS // G, COLS // G)).astype(np.float32)
    norm = rng.integers(0, 1 << 16, (COLS,), dtype=np.uint16)
    ckd, build = tmp_path / "ck", tmp_path / "build"
    ckd.mkdir()
    build.mkdir()
    (ckd / "config.json").write_text(json.dumps({"quantization_config": {"weight_block_size": [G, G]}}))
    safetensors(ckd / "model.safetensors", {W: ("F8_E4M3", w), S: ("F32", s), NORM: ("BF16", norm)})
    (build / "derive-report.json").write_text(json.dumps({"provenance": {"weight_roles": {PREFIX + n: r for n, r in ROLES.items()}}}))
    return SimpleNamespace(ck=WR.CheckpointIndex(str(ckd), verify_shards=False), roles=CPU.registered_roles([str(build)], PREFIX),
                           w=w, s=s, norm=norm)


def served(world: SimpleNamespace, tp: dict[str, int] | None) -> Served:
    """What vLLM serves from the checkpoint: the requant of the linear (of a rank's rows of it under ``tp``), and the norm as it is."""
    def shard(a: np.ndarray) -> np.ndarray:
        if tp is None:
            return a
        n = a.shape[0] // tp["world"]
        return a[tp["rank"] * n:(tp["rank"] + 1) * n]

    qw, qs = FU.requant(shard(world.w), shard(world.s).view(np.uint32), G)
    return {W: ("float8_e4m3fn", qw), S: ("float32", qs), NORM: ("bfloat16", world.norm)}


def raw(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).tobytes()


def test_the_roles_are_read_by_registered_name(world: SimpleNamespace) -> None:
    assert set(world.roles) == {W, S, NORM}, "the Program's fields less the wrapper prefix"


@pytest.mark.parametrize("tp", [WHOLE, RANK_1])
def test_a_tensor_read_on_demand_is_the_one_open_weights_opens_by_the_builds_roles(tmp_path: Path, world: SimpleNamespace, tp: dict[str, int] | None) -> None:
    tensors = served(world, tp)
    tree = weights_tree(tensors) | {"extra": {}}
    words, wc = CPU.open_weights(tmp_path, tree, tree["root"], world.ck, threads=1, tp=tp, roles=world.roles)
    assert wc["equal"] is True and wc["opened"] == 3 and wc["mismatched_n"] == 0 and wc["unopened_n"] == 0, wc
    lazy = CPU.OnRead(tmp_path, tree, tree["root"], world.ck, tp=tp, roles=world.roles)
    assert lazy.words == {} and set(lazy) == set(tensors) and len(lazy) == 3, "nothing opens before its first read"
    for name, (_dtype, a) in tensors.items():
        assert raw(lazy[name]) == raw(words[name]) == raw(a), name
    assert set(lazy.words) == set(tensors)


@pytest.mark.parametrize("tp", [WHOLE, RANK_1])
def test_without_the_roles_the_requantised_linear_does_not_open(tmp_path: Path, world: SimpleNamespace, tp: dict[str, int] | None) -> None:
    tree = weights_tree(served(world, tp)) | {"extra": {}}
    _, bare = CPU.open_weights(tmp_path, tree, tree["root"], world.ck, threads=1, tp=tp)
    assert bare["equal"] is False and sorted(m["name"] for m in bare["mismatched"]) == [W, S], bare
    lazy = CPU.OnRead(tmp_path, tree, tree["root"], world.ck, tp=tp)
    assert raw(lazy[NORM]) == raw(world.norm), "a tensor vLLM copies opens without its role"
    for name in (W, S):
        with pytest.raises(LookupError, match=f"registered tensor {name} .checkpoint. does not open against the committed weights root: "
                                              "tensor root != the committed tensor root"):
            lazy[name]


def test_another_ranks_shard_does_not_open_by_the_roles(tmp_path: Path, world: SimpleNamespace) -> None:
    tree = weights_tree(served(world, RANK_1)) | {"extra": {}}
    rank_0 = {"world": 2, "rank": 0}
    _, wc = CPU.open_weights(tmp_path, tree, tree["root"], world.ck, threads=1, tp=rank_0, roles=world.roles)
    assert wc["equal"] is False and sorted(m["name"] for m in wc["mismatched"]) == [W, S], "the roles compose the rank's own rows"
    _, wc = CPU.open_weights(tmp_path, tree, tree["root"], world.ck, threads=1, tp=WHOLE, roles=world.roles)
    assert wc["opened"] == 1 and sorted(u["name"] for u in wc["unopened"]) == [W, S], "the whole linear is not a rank's shard of it"
    for other in (rank_0, WHOLE):
        lazy = CPU.OnRead(tmp_path, tree, tree["root"], world.ck, tp=other, roles=world.roles)
        assert raw(lazy[NORM]) == raw(world.norm), "the norm is every rank's"
        with pytest.raises(LookupError, match=f"registered tensor {W} .checkpoint. does not open against the committed weights root"):
            lazy[W]
