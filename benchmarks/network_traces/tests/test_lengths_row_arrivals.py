"""The network certifier's lengths row against the vLLM adapter's replay chain: its request row is the chain's K arrival
slots without a cap bit."""
from __future__ import annotations

from verity.protocols.resource_accounting.network.prover import lengths_row as LR
from verity_vllm.compiler import replay_chain as RC


def test_the_request_row_is_the_arrival_slots_layout():
    """The request row's fields are `replay_chain`'s K arrival slots without a cap bit (valid, id, P, O as a struct of
    arrays, LW the bit length of the longest request)."""
    sh = LR.Shape(3, 512, 7, 10)
    chain = RC.Shape(R=1, W=1, A=1, L=1023, NB=1, IW=7, Q=1, D=1, P=1, K=3)
    assert RC.widths(chain).LW == sh.LW
    assert [(name.removeprefix("a_"), n, w) for name, n, w in RC.arrival_fields(chain)] == LR.request_fields(sh)
