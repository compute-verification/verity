"""The flip negative of `verity_vllm.acquire.sources.partial_source` (`TPPartialSource(flip_site=...)`, a rank worker's
`tp_partials_flip_site`): at the one site it names, the rank commits its genuine partial and feeds the collective another
tensor, element 0 moved by 2x+1024 (`<site>`) or by one bf16 ulp (`<site>:ulp`). The source flips only a CUDA tensor, so the
tensors here are CPU tensors whose `device` answers `cuda`; the committer, modules and model are the stand-ins of
`test_tp_partial_source_moe`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from tests.acquire.test_tp_partial_source_moe import _Committer, _Dev, _make, _model
from verity_vllm.acquire.sources import partial_source as tps

OPROJ = "model.layers.0.self_attn.o_proj"
SITE = (OPROJ, "RowParallelLinear")


class _OnCuda(torch.Tensor):
    """A CPU tensor whose `device` answers `cuda`, as the source reads it."""

    device = _Dev()


def _on_cuda(*values: float, dtype: Any = torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype).as_subclass(_OnCuda)


def _wrapped(flip: str | None) -> tuple[Any, _Committer, Any, list[torch.Tensor]]:
    """The source on one o_proj site with flip spec `flip`, its committer, its all-reduce wrapped over a recorder, and what the
    recorder was handed."""
    c = _Committer()
    src = tps.TPPartialSource(c, _model({OPROJ: _make("RowParallelLinear")}), rank=0, world=2, flip_site=flip)
    handed: list[torch.Tensor] = []

    def all_reduce(t: torch.Tensor, *a: Any, **k: Any) -> torch.Tensor:
        handed.append(t)
        return t * 2

    return src, c, src._wrap("all_reduce", all_reduce), handed


@pytest.mark.parametrize("flip, dtype", [("self_attn.o_proj", torch.float32), ("self_attn.o_proj", torch.bfloat16),
                                         ("self_attn.o_proj:ulp", torch.float32)],
                         ids=["gross", "gross in bf16", "ulp of a tensor not in bf16 is gross"])
def test_the_collective_at_the_flipped_site_is_handed_element_0_moved_by_2x_plus_1024_and_the_genuine_partial_is_committed(flip, dtype):
    src, c, w, handed = _wrapped(flip)
    src.cur = SITE
    x = _on_cuda(3.0, 1.0, -2.0, dtype=dtype)
    y = w(x)
    assert [m for m, _x in c.acquired] == [f"{OPROJ}/part_rank0"] and c.acquired[0][1] is x
    assert torch.equal(x, torch.tensor([3.0, 1.0, -2.0], dtype=dtype))
    assert handed[0] is not x and torch.equal(handed[0], torch.tensor([1030.0, 1.0, -2.0], dtype=dtype))
    assert torch.equal(y, handed[0] * 2) and src.flipped == 1
    w(x)
    assert src.flipped == 2 and torch.equal(handed[1], handed[0])


def test_the_ulp_flip_moves_element_0_of_a_bf16_partial_by_one_ulp():
    src, c, w, handed = _wrapped("o_proj:ulp")
    assert (src.flip_site, src.flip_mode) == ("o_proj", "ulp")
    src.cur = SITE
    x = _on_cuda(1.0, -1.0, dtype=torch.bfloat16)
    w(x)
    assert torch.equal(x, torch.tensor([1.0, -1.0], dtype=torch.bfloat16)) and c.acquired[0][1] is x
    assert handed[0].view(torch.int16).tolist() == [x.view(torch.int16)[0].item() + 1, x.view(torch.int16)[1].item()]
    assert handed[0].tolist() == [1.0078125, -1.0] and src.flipped == 1


@pytest.mark.parametrize("flip, cur, on_cuda", [
    (None, SITE, True), ("mlp.down_proj", SITE, True), ("self_attn.o_proj", None, True), ("self_attn.o_proj", SITE, False),
], ids=["no flip", "another site", "outside a hooked module", "a tensor not on CUDA"])
def test_the_collective_is_handed_the_partial_itself_everywhere_else(flip, cur, on_cuda):
    src, c, w, handed = _wrapped(flip)
    src.cur = cur
    x = _on_cuda(3.0, 1.0) if on_cuda else torch.tensor([3.0, 1.0])
    assert torch.equal(w(x), x * 2)
    assert len(handed) == 1 and handed[0] is x and src.flipped == 0
    assert len(c.acquired) == int(on_cuda and cur is not None)
