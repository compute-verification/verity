"""Differential-test adapter (`verity_vllm.properties.admission`) for `RoPE_v1{NHEADS,D}` against vLLM d9105ea80's CUDA
`_C.rotary_embedding` (NeoX, bf16 query / key / cos-sin cache), one token per case:

    # pod, torch venv:  verity-vllm properties-admission produce --adapter verity_vllm.properties.difftests.rope_difftest --n 48 --seed 11 --out DIR
    # verity venv:      verity-vllm properties-admission check   --adapter verity_vllm.properties.difftests.rope_difftest --dir DIR --out RUN/rope.json

The cache has one row, `cs = [cos[0:D/2] | sin[0:D/2]]`, at position 0; the kernel rotates the query (NH heads) and the key (KVH heads)
in place with it, and each is one `RoPE_v1` call on that row.  Head dims 64 / 96 / 128 / 256 and the served models' head counts;
the query, key and cache words cycle through the edge kinds."""
from __future__ import annotations

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry.prims import BF16

SEMANTIC_ID = "b1/RoPE@v1"

SHAPES = ((9, 3, 64), (32, 8, 64), (12, 2, 128), (32, 8, 128), (32, 32, 96), (8, 4, 256), (16, 8, 128))
KINDS = ("random", "zeros", "max", "subnormal", "alternating")


def cases(rng, i: int) -> dict:
    NH, KVH, D = SHAPES[i % len(SHAPES)]
    kq, kk, kc = KINDS[i % len(KINDS)], KINDS[(i // 2) % len(KINDS)], "random" if i % 3 else KINDS[(i // 3) % len(KINDS)]
    return {"statics": {"NH": NH, "KVH": KVH, "D": D},
            "inputs": [{"shape": [NH * D], "dtype": "bfloat16", "kind": kq},
                       {"shape": [KVH * D], "dtype": "bfloat16", "kind": kk},
                       {"shape": [D], "dtype": "bfloat16", "kind": kc}],
            "edge": "random" not in (kq, kk)}


def production(statics: dict, inputs: list) -> list:
    import torch
    from vllm import _custom_ops as ops

    q, k, cs = inputs
    D = int(statics["D"])
    query, key = q.reshape(1, -1).clone(), k.reshape(1, -1).clone()
    positions = torch.zeros(1, dtype=torch.int64, device=q.device)
    ops.rotary_embedding(positions, query, key, D, cs.reshape(1, D).contiguous(), True)
    statics["cc"] = list(torch.cuda.get_device_capability(q.device))
    return [query.reshape(-1), key.reshape(-1)]


def _difftest_body(B, S, q, k, cs):
    """The query's and the key's `RoPE_v1` on the one cache row: the two outputs `_C.rotary_embedding` writes in place."""
    return tuple_of(B.call(bind(b1.RoPE, NHEADS=S.NH, D=S.D), q, cs), B.call(bind(b1.RoPE, NHEADS=S.KVH, D=S.D), k, cs))


RoPEDifftest = CompositeDefinition(
    "RoPEDifftest", 1, ("NH", "KVH", "D"),
    lambda S: ((("q", Array(S.NH, Array(S.D, BF16))), ("k", Array(S.KVH, Array(S.D, BF16))), ("cs", Array(S.D, BF16))),
               Tuple(Array(S.NH, Array(S.D, BF16)), Array(S.KVH, Array(S.D, BF16)))),
    _difftest_body, doc=_difftest_body.__doc__ or "", conformance="derived: difftest", register=False)


def spec(statics: dict):
    return bind(RoPEDifftest, NH=int(statics["NH"]), KVH=int(statics["KVH"]), D=int(statics["D"]))
