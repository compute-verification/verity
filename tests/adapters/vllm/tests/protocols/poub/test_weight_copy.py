"""Storage beside PoUB's two schemes, by the adapter's weight-copy rule (`messages.weight_copy_refusals`) on what each
declares (`keeps_weight_copy`): the remote session keeps a weight-derived copy, the bandwidth audit keeps none."""
from __future__ import annotations

from verity_vllm import messages as M


def test_storage_beside_the_remote_session_is_refused_for_the_copy_it_declares() -> None:
    pous, poub = M.named("pous"), M.named("poub")
    assert M.storage(pous) and not M.storage(poub) and poub.keeps_weight_copy is True
    for listed in ([pous, poub], [poub, pous]):
        (refusal,) = M.weight_copy_refusals(listed)
        assert "pous" in refusal and "poub keeps a weight-derived copy" in refusal and "keeps_weight_copy is undeclared" not in refusal


def test_storage_beside_the_bandwidth_audit_is_not_refused_for_a_weight_copy() -> None:
    pous, bandwidth = M.named("pous"), M.named("bandwidth")
    assert M.storage(pous) and bandwidth.keeps_weight_copy is False
    assert M.weight_copy_refusals([pous, bandwidth]) == M.weight_copy_refusals([bandwidth, pous]) == []
