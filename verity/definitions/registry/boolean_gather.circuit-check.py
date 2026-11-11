"""circuit-check's bindings of `verity.definitions.registry.boolean_gather`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_gather_roots() -> list[Any]:
    """The Boolean embedding (`verity.definitions.registry.boolean_gather`) over gathers traced whole (V = 64), and a gather
    that is a tree of Calls with a partial last chunk (V = 1100), bf16, i32 and f32, and the bit selection of a 32-bit word; and
    the two levels of the f32 gather over the training step's vocabulary (V = 50304) that V = 1100 doesn't reach, its partial
    chunk of 128 words and its 50 chunk words. That gather's whole chunks are V = 1100's; the gather itself, 4.8M gates, is too heavy a target for every
    `circuit-check --all`, and its tests pin it and hold it to its word (`adapters/vllm/tests/program/test_boolean_gather.py`).
    The bf16 widening (`verity.definitions.ops.boolean.cast`) is static-free.

    The chunked gather at V = 100 in V-chunks of at most 40 (three, 34, 33 and 33 words: its offsets, both widths and the
    three-word XOR), and the levels of the training vocabulary's at `W_MAX` that it doesn't reach: its offsets, the 576-word
    partial chunk and 25 chunk words of each 25,152-word V-chunk, and the two-word XOR with the fill. The V-chunks' whole
    chunks are V = 1100's, and the gathers themselves, 2.4M gates a V-chunk, are held to their word by the same tests."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_gather as BGA

    return [
        bind(BGA.Embedding, V=64, H=16),
        bind(BGA.gather_bf16(1100)),
        bind(BGA.gather_i32(1100)),
        bind(BGA.gather_f32(1100)),
        bind(BGA.gather_index(128, 10, "F32")),
        bind(BGA.gather_index(50, 22, "F32")),
        bind(BGA.GatherF32Chunked, V=100, W=40),
        bind(BGA.GatherChunkOffsets, V=50304, W=BGA.W_MAX),
        bind(BGA.gather_index(576, 10, "F32")),
        bind(BGA.gather_index(25, 22, "F32")),
        bind(BGA.GatherChunkXor, C=2),
        bind(BGA.bit_at(32)),
    ]
