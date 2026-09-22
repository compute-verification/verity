"""`verity_capture.bench.vu_census` with the padding primitives registered (the census imports `b1` only; a padded
Program also names `serve3` / `pad_prims` primitives).  Same CLI."""

from __future__ import annotations

import sys

import verity_vllm.registry.b1  # noqa: F401
import verity_vllm.registry.serve3  # noqa: F401

from verity_capture.bench import vu_census

if __name__ == "__main__":
    sys.exit(vu_census.main())
