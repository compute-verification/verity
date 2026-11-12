"""`served_compiled_source.linkage` takes the committer alone.

The two committed values it compares, a step's sampled token ids and the next step's input ids of the first piece, are named in its
own text, so a call that names either one is refused.  The committer here is a stand-in that holds only what `linkage` reads: two
requests that decode at steps 0 and 1 in swapped row order, and one committed value a step.  CPU only."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from verity_vllm.acquire.sources import served_compiled_source as S

SAMPLED, IDS = "runner.sampler/sampled_token_ids", "c00.input/ids"


def _committer(sampled: list[int], ids: list[int]) -> SimpleNamespace:
    """A committer whose step 0 committed `sampled` as the sample and whose step 1 committed `ids` as the first piece's input."""
    blocks, layouts = {}, {}
    for step, name, values in ((0, SAMPLED, sampled), (1, IDS, ids)):
        raw = torch.tensor(values, dtype=torch.int64).view(torch.uint8)
        blocks[step] = (raw, raw.numel())
        layouts[step] = [SimpleNamespace(name=name, dev_off=0, nbytes=raw.numel(), dtype="torch.int64", shape=(len(values),))]
    return SimpleNamespace(step_rows={0: [("a", 0, 1), ("b", 1, 2)], 1: [("b", 0, 1), ("a", 1, 2)]}, _gpu_blocks=blocks, _layouts=layouts)


def test_linkage_takes_the_committer_alone():
    link: Any = S.linkage
    com = _committer([11, 22], [22, 11])
    for option, name in {"ids_name": IDS, "sampled_name": SAMPLED}.items():
        with pytest.raises(TypeError, match=f"got an unexpected keyword argument '{option}'"):
            link(com, **{option: name})
    for given in ([IDS], [IDS, SAMPLED]):
        with pytest.raises(TypeError, match=f"takes 1 positional argument but {1 + len(given)} were given"):
            link(com, *given)
    assert link(com) == {"checked": 2, "mismatches": 0, "first": [], "passed": True}
