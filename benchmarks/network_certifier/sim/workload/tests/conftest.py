import pytest
from datacenter import Assumption, Cadence
from datacenter.examples import leaf_spine


def declared(blocks: int = 2):
    """`datacenter.examples.leaf_spine` on one leaf at a tick of 1 µs and a bucket of 10 ticks: per block a node of 8
    GPUs whose NIC goes to the block's certifier 1 µs away, each certifier and the outside host `ext0` 2 µs from the
    leaf, inside a node no latency."""
    return leaf_spine(blocks, cadence=Cadence(bucket_ticks=10, tick=Assumption("tick", 1e-6, "s", "declared")))


@pytest.fixture
def datacenter():
    return declared
