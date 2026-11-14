"""h1r.cu's front end on torch: verity's `h1r.H1R` (verity/protocols/experimental/dither/prover/h1r.py) with the table's
upload, the plan's buffers and the launches' stream, which verity leaves to its caller."""
from __future__ import annotations

import torch

from verity.protocols.experimental.dither.prover import h1r as _h1r


class H1R(_h1r.H1R):
    def upload(self, data: bytes, like: torch.Tensor) -> torch.Tensor:
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(like.device)

    def empty(self, n: int, dtype: str, like: torch.Tensor) -> torch.Tensor:
        return torch.empty(n, dtype=getattr(torch, dtype), device=like.device)

    def stream(self) -> int:
        return torch.cuda.current_stream().cuda_stream
