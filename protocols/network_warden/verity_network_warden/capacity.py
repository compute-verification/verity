"""The counts the Lean package pins, and the capacity bounds built on them (timing-channel doc §2 and §3).

All logarithms are base 2, and all bounds hold on accepted runs unless a docstring says otherwise.
"""

from __future__ import annotations

import math

K_BITS = 9.12e7  # the inference-only policy's budget per IU: α·Γ = 9.6e−11 × 9.5e17 bits, about 11 MB


def free_patterns(buckets: int, allocation: int) -> int:
    """Lean `card_free`: occupancies if the sender could pick slots within a bucket, `2^(T·r)`."""
    return 2 ** (buckets * allocation)


def left_packed_patterns(buckets: int, allocation: int) -> int:
    """Lean `card_leftPacked`: left-packed occupancies, `(r + 1)^T`."""
    return (allocation + 1) ** buckets


def all_released_patterns(buckets: int, frames: int) -> int:
    """Lean `card_counts_eq_le`: at most `binom(T + M − 1, M)` count vectors release exactly M frames."""
    return math.comb(buckets + frames - 1, frames) if buckets else int(frames == 0)


def available_patterns(buckets: int, frames: int) -> int:
    """Lean `encard_fifoGrids_le_of_available`: at most `binom(T + M, M)` grids when at most M frames are enqueued."""
    return math.comb(buckets + frames, frames)


def envelope_bits(buckets: int, allocation: int) -> float:
    """What occupancy carries per link-window when nothing pins it, `T·log2(r + 1)` (Proposition 2, tight)."""
    return buckets * math.log2(allocation + 1)


def unbound_bits(buckets: int, allocation: int, frame_bits: int) -> float:
    """What a link-window carries if nothing checks its content: `T·log2 Σ_{n≤r} 2^(n·B)`, about `T·r·B`. This is what a
    rejected window may already have carried (every run, Theorem 1(a))."""
    return buckets * (allocation * frame_bits + math.log2(sum(2.0 ** (-(allocation - n) * frame_bits)
                                                          for n in range(allocation + 1))))


def accepted_bound_bits(content_bits: float, sync_bits: float, windows: int, jitter_classes: int = 1,
                        steps: int = 0) -> float:
    """Lean `encard_accSeqs_le_sync`: accepted sequences number at most `#Ac · #Sync^W · D^n`, which is
    `A_c + W·log2 #Sync + n·log2 D` bits. It is 0 for a fixed clock and J (`encard_accSeqs_le_fixedClock`)."""
    return content_bits + windows * sync_bits + steps * math.log2(jitter_classes)


def ingress_rate(allocation: int, bucket_ms: int = 100) -> float:
    """An ingress link's covert rate in bit/s, `log2(r_in + 1)/τ_in`. No clock pins ingress timing, because the outside
    sender chooses when requests arrive. The 100 ms default is decision D's."""
    return math.log2(allocation + 1) / (bucket_ms / 1000)


def k_share(bits: float) -> float:
    """A charge in bits, as a share of K."""
    return bits / K_BITS
