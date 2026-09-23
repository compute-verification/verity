"""The hashed relation (``hashchain.py`` / ``hashauth.py`` / statement v5): census, gadget digest == committer digest,
file round trips, the extended chain tables, and the committed-operand negatives -- on the CPU, two VUs."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.commitments import poseidon2_babybear as h2  # noqa: E402

from . import chain, hashauth, hashchain  # noqa: E402
from .compile import P  # noqa: E402
from .relations import relation  # noqa: E402
from .relchain import HashedRelationRunner, hashed_negatives, instances, tile_instances  # noqa: E402
from .serialize import (  # noqa: E402
    STMT_MAGIC_V5,
    SYS_MAGIC_V3,
    _statement_of,
    proof_bytes,
    read_proof,
    read_statement,
    read_system,
    statement_bytes,
    system_bytes,
    verify_files,
)

CENSUS = {"bf16-hopper": (3292, 608, 1720), "fp8-ada": (3769, 704, 1720), "fp8-hopper": (3396, 704, 1720)}


@pytest.mark.parametrize("name", sorted(CENSUS))
def test_compose_census(name: str) -> None:
    hr = hashchain.compose(relation(name))
    base, private, hashed = CENSUS[name]
    c = hr.census
    assert (c["rows_base"], c["rows_private_operands"], c["rows_hash"], c["rows_digest_pins"]) == (base, private, hashed, 17)
    assert c["sbox_rows_per_permutation"] == 852 and c["permutations_per_unit"] == 2 and c["linked_rows"] == 19
    assert [n for _, n in hr.sys.pins] == [hashchain.HASH_PIN_END] + [hashchain.hash_pin(x) for x in range(16)]
    assert not any(r.kind == "pin" and not r.name.startswith("hash.") for r in hr.sys.rows)
    assert len(hr.sys.chain["c"]) == len(hr.sys.chain["y"]) == 19


@pytest.fixture(scope="module")
def runner():
    rel = relation("fp8-ada")
    R = HashedRelationRunner(rel, "cpu", -128.0, zk=False, mode="fiat-shamir")
    data = instances(rel, 4)
    R.commit_vus(data)
    return R, data


def test_prove_verify_and_committer_digest(runner) -> None:
    R, data = runner
    proof, pubs, lay = R.prove_vus(data[:2], check=True, vu_ids=[0, 1])
    table, idx, ypub, auth, sibs = pubs
    ends = np.arange(2) * lay.steps + lay.steps - 1
    assert np.array_equal(table[ends, 1:9], R.digests["a"][:2]) and np.array_equal(table[ends, 9:], R.digests["b"][:2])
    assert table[:, 0].sum() == 2 and not table[np.setdiff1d(np.arange(lay.l), ends)].any()
    assert R.verify_vus(proof, pubs, lay) == (True, "")
    # the committer on the words the prover used, independently
    a0 = np.asarray(data[0][0]); b1 = np.asarray(data[1][1])
    assert h2.hash_row(a0.tolist(), 8, h2.ROLE_X) == table[ends[0], 1:9].tolist()
    assert h2.hash_row(b1.tolist(), 8, h2.ROLE_W) == table[ends[1], 9:].tolist()


def test_statement_v5_and_system_v3_round_trip(runner, tmp_path) -> None:
    R, data = runner
    proof, pubs, lay = R.prove_vus([data[1], data[3]], check=False, vu_ids=[1, 3])
    st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
    raw = statement_bytes(st)
    assert raw[:8] == STMT_MAGIC_V5 and len(raw) < 2000            # no operand words: ~1.5 kB for two VUs
    st2 = read_statement(raw)
    assert st2.v5 and st2.authentication == "included-hash" and st2.relation == "fp8-ada"
    assert np.array_equal(st2.a, st.a) and np.array_equal(st2.b, st.b) and np.array_equal(st2.y16, st.y16)
    assert st2.hash_auth == st.hash_auth and st2.hash_sibs == st.hash_sibs
    sysb = system_bytes(R.sys)
    assert sysb[:8] == SYS_MAGIC_V3
    rd = read_system(sysb)
    assert rd["m"] == R.sys.m and len(rd["chain"]["c"]) == 19 and rd["chain"]["y16"] is not None
    (tmp_path / "s.stmt").write_bytes(raw)
    (tmp_path / "s.proof").write_bytes(proof_bytes(proof))
    assert read_proof((tmp_path / "s.proof").read_bytes()).root == proof.root
    ok, why, _ = verify_files(tmp_path / "s.stmt", tmp_path / "s.proof", "cpu", runner=R)
    assert ok, why
    # a v5 statement against the bare runner, and vice versa, is refused
    from .relchain import RelationChainRunner
    bare = RelationChainRunner(relation("fp8-ada"), "cpu", -128.0, zk=False, mode="fiat-shamir")
    ok, why, _ = verify_files(tmp_path / "s.stmt", tmp_path / "s.proof", "cpu", runner=bare)
    assert not ok and "authentication" in why


def test_negatives_rejected(runner) -> None:
    R, data = runner
    negs = hashed_negatives(R, data, row_negatives=3)
    accepted = [name for name, (ok, _) in negs if ok]
    assert len(negs) >= 20 and accepted == []
    names = " | ".join(n for n, _ in negs)
    for must in ("digest lane", "swapped", "multiproof sibling", "root replaced", "computed on the flipped row", "W column",
                 "hashed word", "capacity link", "out of range"):
        assert must in names


def test_tile_sharing_layout(runner) -> None:
    rel = relation("fp8-ada")
    data, x, w = tile_instances(rel, 2, 3, seed=7)
    assert len(data) == 6 and x.shape == (2, 1536) and w.shape == (3, 1536)
    assert data[4][0] == x[1].tolist() and data[4][1] == w[1].tolist()
    R = HashedRelationRunner(rel, "cpu", -128.0, zk=False, mode="fiat-shamir")
    R.commit_vus(data, tile=(2, 3), x_rows=x, w_cols=w)
    assert R.trees["a"].count == 2 and R.trees["b"].count == 3 and R.trees["y"].count == 6
    proof, pubs, lay = R.prove_vus(data[3:6], check=False, vu_ids=[3, 4, 5])
    auth = pubs[3]
    assert auth.x_index == (1, 1, 1) and auth.w_index == (0, 1, 2)
    assert R.verify_vus(proof, pubs, lay) == (True, "")
    # a VU claiming the wrong column of the tile
    bad = dataclasses.replace(auth, w_index=(0, 2, 2))
    idx = bad.indices()
    assert R.verify_vus(proof, (pubs[0], idx, pubs[2], bad, pubs[4]), lay)[0] is False


def test_chain_tables_extend_the_three_link_case() -> None:
    """``chain_coefs`` for a system with 19 linked rows: the first three families are the accumulator's, the extras are
    monomials in the seven base families; a 3-link system's tables are exactly the pre-extension ones."""
    torch.manual_seed(0)
    lay = chain.Layout(16, 2, 8)
    rc = torch.randint(0, P, (2, chain.N_FAMILIES, lay.l), dtype=torch.int64)
    e0, e1, e6 = (chain.extra_coef(rc, e) for e in (0, 1, 6))
    assert torch.equal(e0, rc[:, 0] * rc[:, 6] % P) and torch.equal(e1, rc[:, 1] * rc[:, 6] % P)
    assert torch.equal(e6, rc[:, 0] * rc[:, 6] % P * rc[:, 6] % P)
    # the 3-link tables of a bare relation are the pre-extension ones (no extra virtual families)
    from .witness import tables
    sys3 = relation("fp8-ada").compile(relation("fp8-ada").params, True)
    tabs = chain._chain_tables(sys3, lay, "cpu")
    assert tabs["n_extra"] == 0 and tabs["famsel"].tolist() == [0, 1, 2, 3, 4, 5, 6] and tabs["link_fams"].tolist() == [0, 1, 2]
    hr = hashchain.compose(relation("fp8-ada"))
    tabs19 = chain._chain_tables(hr.sys, lay, "cpu")
    assert tabs19["n_extra"] == 32 and tabs19["link_fams"].tolist() == [0, 1, 2] + list(range(7, 23))
