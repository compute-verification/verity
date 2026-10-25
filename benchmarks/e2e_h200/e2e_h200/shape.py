"""The frozen MQA decoder (CMP-92, compute-accounting 23:00Z) and the small shapes the skeleton runs on fixtures.

A pre-norm decoder: RMSNorm, W8A8 QKV with one KV head, attention over the slot's KV cache, W8A8 O, the residual add,
RMSNorm, W8A8 up, ReLU folded into the down projection's quantizer, W8A8 down, the residual add; then the final RMSNorm, the
W8A8 head over the byte vocabulary and argmax. The weights' s8 codes live in W, the encoded store's plaintext, tensor after
tensor in `tensors()` order; their f32 channel scales, the norm gains and the embeddings are small and registered beside it.
"""
from __future__ import annotations

from dataclasses import dataclass

#: RMSNorm's chunk (the `RmsSumSqPart_v1{C}` sum) and how many chunk sums one `F32SumPart_v1{G}` adds (circuits' plan, 1b)
RMS_CHUNK, RMS_GROUP = 128, 64
#: the row absmax's chunk (`RowAbsMax_v1{128}`); a max composes under any grouping
AMAX_CHUNK = 128


@dataclass(frozen=True)
class Shape:
    name: str
    d: int
    layers: int
    heads: int
    head_dim: int
    mlp: int
    vocab: int = 256
    ctx: int = 32
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.heads * self.head_dim != self.d:
            raise ValueError(f"{self.name}: {self.heads} heads of {self.head_dim} are not width {self.d}")
        for k in (self.d, self.mlp):
            if k % 32 or k % RMS_CHUNK:
                raise ValueError(f"{self.name}: every K is a multiple of the s8 step (32) and of {RMS_CHUNK}, not {k}")
        if self.ctx > 32:
            raise ValueError("PV pads the keys to K = 32 (`GemmCoordinatePad_v1{T, K=32}`), so ctx <= 32")

    @property
    def qkv(self) -> int:
        """QKV's rows: every query head, then the one KV head's key and value."""
        return self.d + 2 * self.head_dim

    def tensors(self) -> list[tuple[str, int, int]]:
        """(name, N, K) of every s8 weight tensor, in W's order: per layer QKV, O, up, down; then the head."""
        out = []
        for layer in range(self.layers):
            out += [(f"l{layer}.qkv", self.qkv, self.d), (f"l{layer}.o", self.d, self.d),
                    (f"l{layer}.up", self.mlp, self.d), (f"l{layer}.down", self.d, self.mlp)]
        return out + [("head", self.vocab, self.d)]

    @property
    def w_bytes(self) -> int:
        return sum(n * k for _, n, k in self.tensors())


FROZEN = Shape("mqa16k-l52", d=16384, layers=52, heads=256, head_dim=64, mlp=65536)
#: the frozen shape at 1/128 of its width and 2 layers: the CPU fixture (same ops, same templates up to K)
TINY = Shape("mqa128-l2", d=128, layers=2, heads=2, head_dim=64, mlp=512)
#: a GPU slice: 4 layers at width 2048 (32 heads), for a one-GPU smoke before the store is on the card
SLICE = Shape("mqa2k-l4", d=2048, layers=4, heads=32, head_dim=64, mlp=8192)
SHAPES = {s.name: s for s in (FROZEN, TINY, SLICE)}
