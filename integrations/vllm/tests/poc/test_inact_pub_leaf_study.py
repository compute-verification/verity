"""INACT-PUB validation: the leaf-study facts hold (new file; no store code touched).

Run:  cd vllm-poc && ../.venv/bin/python -m pytest tests/test_inact_pub_leaf_study.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_POC))

from verity_vllm import hashing  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "inact_pub_leaf_study", _POC / "scripts" / "inact_pub_leaf_study.py")
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


def test_f1_id_first_breaks_pad_leaf_identity():
    """Byte-identical pad rows at different positions hash to distinct v2 leaves."""
    val = b"\x00" * 1152
    id_a = "p=d7afb8da14c4/r=req-0/t=40/ph=decode/l=0/op=layers.0.mlp.down_proj.gemm/port=0"
    id_b = "p=d7afb8da14c4/r=req-0/t=41/ph=decode/l=0/op=layers.0.mlp.down_proj.gemm/port=0"
    la = hashing.leaf_hash(id_a, "bfloat16", [576], val)
    lb = hashing.leaf_hash(id_b, "bfloat16", [576], val)
    assert la != lb


@pytest.mark.parametrize("m", [1, 2, 3, 5, 7, 8, 15, 16, 17, 63, 64, 65, 100, 945, 1024, 4096])
def test_uniform_fold_matches_naive_fold(m):
    """O(log m) dedup fold reproduces the exact pair/lift root, all widths."""
    leaf = hashing.H("verity-vllm/leaf/v3-idfree-CANDIDATE", b"bfloat16", b"[576]", b"\x00" * 1152)
    root_naive, calls_naive = study._fold_naive([leaf] * m)
    root_dedup, calls_dedup = study._fold_uniform(leaf, m)
    assert root_dedup == root_naive
    assert calls_dedup <= max(1, 4 * m.bit_length())  # O(log m)
    if m > 1:
        assert calls_naive >= m - 1


def test_id_free_leaf_sequence_is_step_invariant():
    """Retargeting a step to another position leaves the id-free leaf sequence unchanged
    (the basis of per-step root reuse), while the v2 leaf sequence differs everywhere."""
    entries, _ = study.load_step_entries(None)  # embedded census, no archive needed
    a = study.retarget_step(entries, 100)
    b = study.retarget_step(entries, 2000)

    def idfree(step):
        return [hashing.H("verity-vllm/leaf/v3-idfree-CANDIDATE", e["dtype"].encode(),
                          hashing.shape_text(e["shape"]).encode(), b"\x00" * e["nbytes"])
                for e in sorted(step, key=lambda e: e["canonical_id"])]

    def v2(step):
        return [hashing.leaf_hash(e["canonical_id"], e["dtype"], e["shape"], b"\x00" * e["nbytes"])
                for e in sorted(step, key=lambda e: e["canonical_id"])]

    assert idfree(a) == idfree(b)
    assert all(x != y for x, y in zip(v2(a), v2(b), strict=True))


def test_open_and_compare_uses_real_verifier():
    """verify_opening authenticates a pad-row opening built with the real tree code, and
    the canonical-constant comparison rejects a one-byte deviation."""
    entries, _ = study.load_step_entries(None)
    res = study.bench_open_and_compare(entries)
    assert res["verified"] is True
    # a deviating committed value is caught by the byte compare
    from verity_vllm.capture import Value
    from verity_vllm.commit import MerkleTree
    e = entries[0]
    bad = b"\x01" + b"\x00" * (e["nbytes"] - 1)
    leaves = {e["canonical_id"]: hashing.leaf_hash(e["canonical_id"], e["dtype"], e["shape"], bad)}
    tree = MerkleTree(leaves, "ru/req-synth/000041")
    opening = tree.open(Value(e["canonical_id"], e["dtype"], tuple(e["shape"]), bad, "captured"))
    from verity_vllm.commit import verify_opening
    assert verify_opening(tree.root, tree.leaf_count, opening)  # authentic commitment...
    assert opening.value != study.pad_value(e["nbytes"])  # ...but not the canonical fill
