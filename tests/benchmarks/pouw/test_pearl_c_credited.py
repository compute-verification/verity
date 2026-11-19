"""`serving.pearl_c.credited.credited_inputs`: the ports it fills from a proved Pearl+ pass's directory are ones Pearl+'s
tile core passes, evaluated on words by the reference evaluator, with the pass's transcript leaf as its digest, and a
served y word other than the committed one fails."""
from __future__ import annotations

import pytest
from verity.circuits.evaluation import evaluate
from verity.protocols.resource_accounting.compute.pouw.prover import schemes as pouw_schemes
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as T
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.credited import credited_inputs
from verity.protocols.resource_accounting.compute.pouw.verifier.audit import tile_refs

import test_pearl_c_vllm as V  # the served passes' test: its proved passes, written in the device run's layout

SCHEME = "pearl-c-h100-v1-h3"
#: 2 x 2 tiles: the core on words over a 2 x 2 tile is seconds, a 16 x 16 one minutes
TILE = (2, 2)
#: the core's cap circuit needs two whole promotion windows of 128 (`inert_bound._promoted`)
K = 256
#: the core takes its tile's real rows as a static (TR), so a tile part filler has y on its real rows alone
TAKES_TR = "TR" in T.PearlPlusTileCore.statics


def _flat(x) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, int):
        return [x]
    return [w for v in x for w in _flat(v)]


def _on_words(c) -> tuple[bytes, int]:
    out = list(evaluate(c.core(), *(_flat(a) for a in c.core_args())))
    return bytes(out[:64]), out[64]


def _tile_id(epoch, call: int, row: int) -> int:
    """The id of the tile of call ``call`` whose rows begin at ``row``, at column tile 0."""
    return next(r.id for r in tile_refs(pouw_schemes.get(SCHEME, k_min=32, tile=TILE), epoch.workload)
                if r.matmul == call and r.tile.rows[0] == row and r.tile.cols[0] == 0)


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    return V.proved_pass(tmp_path_factory.mktemp("full"), SCHEME, tile=TILE, k=K)


@pytest.mark.slow
def test_a_full_tiles_ports_pass_the_core_with_its_leaf_as_digest(full):
    """Call 1's first tile (every row real): the core on the filled ports is ok 1, on words as in `reference`, its digest is
    the pass's transcript leaf (the prover's ``digest``) and opens at the tile's position under the transcript root. A
    flipped word of the tile's served y gives ok 0, on words too. Slow (about a minute: the pass at k = 256 and two
    evaluations of the core on words): the only check that the ports a pass's files fill are the circuit's, not only
    `reference`'s."""
    path, _s, epoch, prover, tr, _splits, _records = full
    tid = _tile_id(epoch, 1, 0)
    c = credited_inputs(path.parent, tid)
    assert c.real == c.tm == 2 and len(c.y) == 2 and c.unit == epoch.workload.matmuls[1].index
    leaf = prover.digest(tid)
    assert c.reference() == (leaf, 1)
    assert _on_words(c) == (leaf, 1)
    assert c.leaf_reference(leaf) == 1 and c.troot == tr.commitment.root and c.root == epoch.weight_roots[1]
    c.y[1][c.col * c.tn] ^= 1
    assert c.reference() == (leaf, 0)
    assert _on_words(c) == (leaf, 0)


@pytest.mark.slow
@pytest.mark.skipif(not TAKES_TR, reason="the core takes no real-row count yet (pearl_plus_tile's TR)")
def test_a_partial_tiles_ports_pass_with_y_on_its_real_rows(tmp_path):
    """A round of a class's fixed load (13 real rows of 16): call 1's tile at rows 12-13 has one real row, so y holds it
    alone and row 13 is zero filler; its ports pass the core (TR = 1) with the pass's leaf as digest, and the tile at
    rows 14-15, filler throughout, is refused. Slow, as the full tile's test, and the only check of a partial tile's
    ports against the circuit."""
    path, _s, epoch, prover, _tr, _splits, _records = V.proved_pass(tmp_path, SCHEME, tile=TILE, load=(13, 16), k=K)
    tid = _tile_id(epoch, 1, 12)
    c = credited_inputs(path.parent, tid)
    assert (c.real, c.tm, len(c.y)) == (1, 2, 1) and not any(c.a[1])
    assert c.statics()["TR"] == 1
    leaf = prover.digest(tid)
    assert c.reference() == (leaf, 1)
    assert _on_words(c) == (leaf, 1)
    assert c.leaf_reference(leaf) == 1
    with pytest.raises(ValueError, match="is filler throughout"):
        credited_inputs(path.parent, _tile_id(epoch, 1, 14))
