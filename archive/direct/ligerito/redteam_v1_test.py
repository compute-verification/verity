"""Red-team V1 (virtual rows unconstrained): ``run.v1_forgeries`` -- honest sumchecks on the prover's ``z``, the commitment
non-zero on the virtual rows.  With the zero claims the verifier rejects each at its stage; without them it accepts every
forgery but the ``next:*`` one (pinned by the shift).  GPU."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from archive.direct.ligero.relchain import instances  # noqa: E402

from . import prove as prove_mod  # noqa: E402
from .prove import Prover, make_coins  # noqa: E402
from .run import v1_forgeries  # noqa: E402


@pytest.fixture(scope="module")
def pv():
    return Prover("fp8-ada", "cuda", 256)


@pytest.fixture(scope="module")
def vus(pv):
    return instances(pv.rel, 12)


@pytest.mark.parametrize("which", range(5))
@pytest.mark.parametrize("zero_claims", [True, False])
def test_forgery(pv, vus, which, zero_claims, monkeypatch):
    if not zero_claims:
        monkeypatch.setattr(prove_mod, "zero_blocks", lambda virt: [])
    bt = pv.batch(vus, "fiat-shamir")
    assert (bt.params["n_claims"] == 1 + bt.lay.n_links + 6) == zero_claims
    name, kw, stage = v1_forgeries(pv, bt)[which]
    proof = pv.prove(bt, make_coins("fiat-shamir", 1), **kw).proof
    ok, why, _ = pv.verify(kw.get("forge_stmt", bt.stmt), proof, make_coins("fiat-shamir", 1))
    if zero_claims or stage != "PCS":
        assert not ok and stage in why, (name, why)
    else:
        assert ok, (name, why)


def test_honest_accepts_with_zero_claims(pv, vus):
    bt = pv.batch(vus, "fiat-shamir")
    assert bt.params["zero_blocks"] == [[0, 3577], [1, 1789], [2, 895], [7, 28], [6, 58], [0, 3776]]
    ok, why, info = pv.verify(bt.stmt, pv.prove(bt, make_coins("fiat-shamir", 1)).proof, make_coins("fiat-shamir", 1))
    assert ok, why
