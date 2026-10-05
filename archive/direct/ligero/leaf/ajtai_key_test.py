"""Red-team-leaf-2 G1 (must reject): a hashed system whose REAL Ajtai chain is keyed by ``B = 0`` (every digest 0: the trees
commit to nothing about the operands) plus 2n decoy linear rows carrying the derived key over always-zero hint rows.  The
old key checks scanned for the coefficient pattern and passed it; :func:`leaf.ajtai.chain_key` follows the digest lanes to
the rows the chain uses.  Torch-free: the honest systems pass, the decoy / naive ``B = 0`` / moved-bit systems are refused;
with torch: the Python verifier refuses the decoy proof end to end (the Rust side: ``leaf.rs`` unit tests and
``tests/relations.rs::ajtai_g1_decoy_fixture_is_refused``)."""
from __future__ import annotations

import numpy as np
import pytest

from .. import hashchain
from ..compile import P, Linear
from ..relations import relation
from . import ajtai as A

DECOYS = [f"decoy.z{i}" for i in range(A.M_COL + 2)]


def _patch_zero_key(monkeypatch, decoys: bool) -> None:
    """The red-team construction (``redteam/leaf2_ajtai_decoy.py`` on lane/red-team-leaf-2): the circuit and the committer
    use B = 0; with ``decoys``, every gadget call also appends n rows with the derived key over zero hint rows."""
    orig_init, orig_gadget, orig_cw = A.AjtaiLeaf.__init__, A.AjtaiLeaf.gadget, A.AjtaiLeaf.chain_witness

    def init(self, *a, **k):
        orig_init(self, *a, **k)                    # params_sha256 = the honest scheme's (the statement names it)
        self._B = np.zeros_like(A.derive_matrix(self.n, 2 * self.m_col))

    monkeypatch.setattr(A.AjtaiLeaf, "__init__", init)

    def gadget(self, ctx, words, *, word_bits, role, row_words, carry_in, first=False, last=True):
        out = orig_gadget(self, ctx, words, word_bits=word_bits, role=role, row_words=row_words, carry_in=carry_in,
                          first=first, last=last)
        if decoys:
            z = getattr(ctx, "_decoy_rows", None)
            if z is None:
                z = ctx._decoy_rows = [next(iter(ctx.hint(nm).lin)) for nm in DECOYS]
            b = A.derive_matrix(self.n)
            off = A.key_offset(role, self.m_col)
            for x in range(self.n):
                lin = {z[i]: (P - int(b[x, off + i])) % P for i in range(self.m_col) if int(b[x, off + i])}
                lin[z[self.m_col]] = P - 1
                lin[z[self.m_col + 1]] = 1
                ctx.sys.linear.append(Linear(f"hash.decoy.r{role}.{x}", lin, 0, None))
        return out

    def chain_witness(self, words, *, word_bits, role, row_words, n_vus, steps, l):
        import torch
        carry, dg, extras = orig_cw(self, words, word_bits=word_bits, role=role, row_words=row_words, n_vus=n_vus, steps=steps, l=l)
        if decoys:
            extras = {**extras, **{nm: torch.zeros(l, dtype=torch.int64, device=words.device) for nm in DECOYS}}
        return carry, dg, extras

    monkeypatch.setattr(A.AjtaiLeaf, "gadget", gadget)
    monkeypatch.setattr(A.AjtaiLeaf, "chain_witness", chain_witness)


@pytest.mark.parametrize("rel,leaf,n", [("fp8-ada", "ajtai-n64", 64), ("bf16-hopper", "ajtai-n128", 128)])
def test_honest_systems_pass_and_name_their_bit_rows(rel, leaf, n) -> None:
    hr = hashchain.compose(relation(rel), leaf)
    bx, bw = A.chain_key(hr.sys, n)
    assert len(bx) == len(bw) == A.M_COL and not set(bx) & set(bw)
    names = {r.idx if hasattr(r, "idx") else i: r.name for i, r in enumerate(hr.sys.rows)}
    assert names[bx[0]] == "hash.a[0].b0" and names[bw[-1]].startswith("hash.b[")
    hr.leaf.check_system(hr.sys)
    with pytest.raises(ValueError):                                  # the other degree's lanes
        A.chain_key(hr.sys, 128 if n == 64 else 64)


def test_g1_zero_key_chain_with_decoy_key_rows_is_refused(monkeypatch) -> None:
    _patch_zero_key(monkeypatch, decoys=True)
    hr = hashchain.compose(relation("fp8-ada"), "ajtai-n64")
    assert sum(1 for L in hr.sys.linear if L.name.startswith("hash.decoy.")) == 2 * 64
    with pytest.raises(ValueError, match="the key row is not out - "):
        A.chain_key(hr.sys, 64)


def test_naive_zero_key_chain_is_refused(monkeypatch) -> None:
    _patch_zero_key(monkeypatch, decoys=False)
    hr = hashchain.compose(relation("fp8-ada"), "ajtai-n64")
    with pytest.raises(ValueError, match="the key row is not out - "):
        A.chain_key(hr.sys, 64)


def test_a_key_row_reading_a_bit_off_another_row_is_refused() -> None:
    hr = hashchain.compose(relation("fp8-ada"), "ajtai-n64")
    bx, bw = A.chain_key(hr.sys, 64)
    L = next(L for L in hr.sys.linear if L.name == "hash.ajtai.r1.out[3]")
    L.lin[bw[5]] = L.lin.pop(bx[5])
    with pytest.raises(ValueError, match="bit position 5 is row"):
        A.chain_key(hr.sys, 64)


def test_python_verifier_refuses_the_g1_decoy_proof(monkeypatch) -> None:
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("blake3", exc_type=ImportError)
    _patch_zero_key(monkeypatch, decoys=True)
    from ..relchain import HashedRelationRunner, instances
    rel = relation("fp8-ada")
    R = HashedRelationRunner(rel, "cpu", -128.0, zk=False, mode="fiat-shamir", leaf="ajtai-n64")
    data = instances(rel, 4)
    R.commit_vus(data)
    assert not np.asarray(R.digests["a"]).any() and not np.asarray(R.digests["b"]).any()      # the trees commit to zeros
    proof, pubs, lay = R.prove_vus(data[:2], check=True, vu_ids=[0, 1])                        # the constraints hold ...
    ok, why = R.verify_vus(proof, pubs, lay)
    assert not ok and "does not use the derived key" in why, why                              # ... the key check refuses
