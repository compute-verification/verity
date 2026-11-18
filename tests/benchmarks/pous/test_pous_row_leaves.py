"""PoUS's one leaf digest, `verifier.row_leaves`, held to the bytes of the three computations it replaced: the verifier's
`row_leaf`, the P2 store's `segment_root` and PoUB's `_segment_paths` (`poub/prover/p2_stage/bandwidth.py`), each
recorded from the code before it called `row_leaves`, on one fixed segment (24-byte salt `01..18`, segment 5 of 16 row
digests `sha512(b"row %d" % v)`)."""
from __future__ import annotations

import hashlib
import types

from verity.protocols.resource_accounting.memory.poub.prover.p2_stage import bandwidth as BW
from verity.protocols.resource_accounting.memory.pous.prover.p2.p2_store import store as ST
from verity.protocols.resource_accounting.memory.pous.verifier import verifier as V

SALT, S, N = bytes(range(1, 25)), 5, 16
ROWS = b"".join(hashlib.sha512(b"row %d" % v).digest() for v in range(N))
GEOMETRY = types.SimpleNamespace(blocks_per_segment=N)

LEAVES = {
    0: "4d986cf95c8e0b60beb2b27d476a338cadb52571a80bf0cf6e6267b491c5116c3e2519bb4799d26237aed5143d9600dd78c8fe46d99a1c838e528cf9c28c92d9",
    7: "5cdaff9e603b608e3fed29baa3a5b56b25b58b4fa593bb7e726b56538b108beda973cd2c2f5b504cf1915f77133e18669b474ed3551cc828c5bdd2a7545b333f",
    9: "872fffddc8be40e534398e5d224eb48443da6d5fd57b6c26d13d70992a4286a8bd0abec154aa798798d908ffeb6068c30418ff22a8b2eb27e9536a1b416c2ab6",
    15: "37c3a5c0230690beaf31e444e60d55836b4c1fade8b94480d1ab6989f91c2020c10fb2f1b1f882927cc85a33e75332cdfcb6dda236f67331cccc01704e1a9024",
}
ROOT = "624a9becea6ef65cdec8e65b180c31d63bcac2b829603bd7d7ebf8ae47849ad19638bb8faf9e011820fafa7832c6476b682dc0e9f8092034026854cc3d6bc81f"
# sha512 of each leaf's joined authentication path, as `_segment_paths` returned it
PATHS = {
    0: "cd18d574a6d52db7cf1820fa49b3fccbe57b633cf7655a1d569b867b3edc5cd9d497fddc6563d73a1be9dfe50f7ef152f0151bf9af4ab66a3b9be61408e87668",
    7: "2af180dac94308d64dcc87ffb0c8b3876f89eaa6d098737390beec27fe629b6e5771793fc40e103f7b6ddde06c75ba462968a0995808fde46d4601e599bd494d",
    15: "97bab740b34df480443992d2b4811275130ad1ebddf56912d96452570531795bfc53191ad503157a58611956ebbc85241e01cebaa3715496dbd7bd61fbbd7319",
}


def test_row_leaves_are_the_old_leaves():
    leaves = V.row_leaves(SALT, S, ROWS)
    for v, want in LEAVES.items():
        assert leaves[v].hex() == want
        assert V.row_leaf(GEOMETRY, SALT, S * N + v, ROWS[64 * v:64 * v + 64]).hex() == want
    assert [x.hex() for x in V.row_leaves(SALT, S, ROWS[64 * 7:64 * 10], v0=7)] == [LEAVES[7], leaves[8].hex(), LEAVES[9]]


def test_the_provers_trees_are_the_old_trees(tmp_path):
    assert ST.segment_root(GEOMETRY, SALT, S, ROWS).hex() == ROOT
    rows = tmp_path / "rows.bin"
    rows.write_bytes(ROWS)
    paths = BW._segment_paths((str(rows), SALT, S, N, list(PATHS), bytes.fromhex(ROOT)))
    assert {v: hashlib.sha512(b"".join(p)).hexdigest() for v, p in paths.items()} == PATHS
