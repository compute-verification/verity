"""GPU Merkle commitments for the proof-internal commitment layer (hash-gpu lane, 2026-09-22).

Layout committed: a matrix of ``R`` rows x ``N`` columns of BabyBear elements (Ligero's codeword layout: ``N``
codeword columns, one leaf per column, leaf = hash of the column), stored row-major as ``uint32`` on the device.
Three hashes, each with a bit-exact CPU reference (``reference.py``):

* SHA-256 (CUDA, ``sha256_cuda.py``): leaf = SHA-256 of the column's bytes (elements as little-endian u32),
  node = SHA-256(left || right) over the 32-byte digests.  ``chunk`` mode splits a column into 1 KiB sub-chunks
  hashed in parallel and combines them with a binary SHA-256 tree (a different, labelled leaf function).
* Blake3 (CUDA, ``blake3_cuda.py``): leaf = standard Blake3 of the column's bytes, computed as the spec's chunk
  tree (1 KiB chunks in parallel, parent nodes level by level), so it equals the ``blake3`` PyPI package exactly;
  node = Blake3(left || right).
* Poseidon2 over BabyBear (``poseidon2_torch.py``, ``poseidon2_cuda.py``; Plonky3's permutation, width 16 or 24):
  leaf = binary compression tree over the column's 8-element blocks (``TruncatedPermutation``-style 2:1
  compression), node = compression of the two 8-element digests.

``merkle.py`` is the tree builder; ``bench.py`` the recorded measurement.  The Verity *external* commitments
(``verity.ml.commitments.merkle``: domain-bound SHA-256) are unchanged and not what this module measures.
"""

from __future__ import annotations

__all__ = ["WORDS_PER_DIGEST", "P"]

P = 0x78000001                 # BabyBear, 2^31 - 2^27 + 1
WORDS_PER_DIGEST = 8           # every hash here yields 8 x 32-bit words (32 bytes; Poseidon2: 8 field elements)
