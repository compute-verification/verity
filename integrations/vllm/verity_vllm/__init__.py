"""verity_vllm: the Verity x vLLM replay proof-of-concept harness (worker W2).

Pure Python + numpy.  Nothing here imports torch at module level; the
``same_runtime`` relation checkers that need torch live behind optional
imports inside :mod:`verity_vllm.program.kernels.relations`.

Module map (see ``docs/vllm-poc/capture-format.md`` for the on-disk format):

- :mod:`verity_vllm.commit.hashing`     ``H(tag, *parts)`` and the leaf/node/lift/ids tags
- :mod:`verity_vllm.program.dtypes`      torch-style dtype strings, itemsizes, BF16 <-> float32
- :mod:`verity_vllm.commit.identity`    canonical ids (contract section 5), weight ids, boundary ids
- :mod:`verity_vllm.observe.capture_v1`     capture format v1 writer and lazy reader
- :mod:`verity_vllm.commit.merkle`      Merkle trees, openings, ``commitments.json``
- :mod:`verity_vllm.check.poc_description` ``description.json`` schema and validation
- :mod:`verity_vllm.check.poc_required_interface`    ``required.json`` derivation and freeze
- :mod:`verity_vllm.program.kernels.relations`   relation registry, the ``check`` runner, adjacent-VU helpers
- :mod:`verity_vllm.relations_torch` optional torch ``same_runtime`` checkers (``install()``)
- :mod:`verity_vllm.negatives`   targeted negative tests N01-N07, N13, N14
- :mod:`verity_vllm.render`      ``docs/fixtures/results.md`` renderer (``python -m``)
- :mod:`verity_vllm.verify`      fresh-process bundle verifier (``python -m``, contract 15.1)
"""

from __future__ import annotations

__version__ = "0.1.0"

CAPTURE_FORMAT_VERSION = 1
