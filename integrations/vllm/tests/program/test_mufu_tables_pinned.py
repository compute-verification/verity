"""The measured tables the MUFU Definitions evaluate through are in-tree files pinned by digest: no environment variable
selects another table or declares the arch, and a table that is not the pinned one does not load."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from verity.ml.circuits.ops import mufu
from verity_vllm.program.kernels import fa2_model as FA
from verity_vllm.program.kernels import fa2_relation, rms_relation
from verity_vllm.program.kernels import rms_triton_model as RM
from verity_vllm.program.registry import prims as P

TANH_TABLE_WORD = 0x3F000000  # tanh(0.5): between the identity and the saturation rule, so answered from the table


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_the_environment_selects_no_table(monkeypatch, tmp_path):
    for name in ("VERITY_MUFU_TABLES", "VERITY_RMS_TABLES", "VERITY_MUFU_TANH_TABLES"):
        monkeypatch.setenv(name, str(tmp_path))
    fa2 = {n: _sha256(os.path.join(fa2_relation.tables_dir(), f"mufu_{n}_delta_int8.xz")) for n in fa2_relation.TABLE_SHA256}
    assert fa2 == fa2_relation.TABLE_SHA256
    for kernel, pinned in rms_relation.TABLE_SHA256.items():
        assert {n: _sha256(os.path.join(rms_relation.tables_dir(kernel), f"mufu_{n}_delta_int8.xz")) for n in pinned} == pinned
    mufu.mufu_tanh_shards.cache_clear()
    assert isinstance(P.MufuTanh.evaluate(TANH_TABLE_WORD), int)
    table = os.path.join(P.MUFU_TANH_TABLE_DIR, P.mufu_tanh_table_manifest()["table"]["file"])
    assert [s["path"] for s in P.mufu_tanh_coverage()["shards"]] == [table]


def test_a_table_that_is_not_the_pinned_one_does_not_load(monkeypatch):
    z, other = np.zeros(1, np.uint32), {"sha256": "0" * 64}
    monkeypatch.setitem(fa2_relation._TABLES, fa2_relation.tables_dir(), FA.MufuTables(z, z, {"ex2": other, "rcp": other}))
    with pytest.raises(ValueError, match="pinned"):
        fa2_relation.tables()
    monkeypatch.setitem(rms_relation._TABLES, rms_relation.tables_dir("cuda"), RM.RmsTables(z, z, {"sqrt": other, "rcp": other, "rsq": other}, z))
    with pytest.raises(ValueError, match="pinned"):
        rms_relation.tables("cuda")
    monkeypatch.setattr(mufu, "MUFU_TANH_TABLE_MANIFEST_SHA512", "0" * 128)
    mufu.mufu_tanh_shards.cache_clear()
    with pytest.raises(P.MufuTanhUnmeasured, match="pinned"):
        P.MufuTanh.evaluate(TANH_TABLE_WORD)


def test_the_decoded_tables_are_core_library_tables():
    """Each measured MUFU table, decoded as raw little-endian u32 words, has the SHA-256 and SHA-512 core's library lists."""
    from verity.ml.circuits import library as L

    t, fa2 = rms_relation.tables("cuda"), fa2_relation.tables()
    for name in ("rsq", "sqrt", "rcp", "ex2"):
        raw = np.ascontiguousarray(getattr(fa2 if name == "ex2" else t, name), dtype="<u4").tobytes()
        entry = L.table(name)
        assert len(raw) == 4 * entry.words
        assert (hashlib.sha256(raw).hexdigest(), hashlib.sha512(raw).hexdigest()) == (entry.sha256, entry.sha512)


def test_the_arch_is_declared_never_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("VERITY_ARCH", "sm_89")
    assert FA.arch_of({}) is None and FA.arch_of({"arch": "sm_89"}) == "sm_89"
    monkeypatch.setenv("VERITY_ARCH", "sm_90")
    row = {"x": np.zeros(16, np.uint16), "w": np.zeros(16, np.uint16)}
    assert rms_relation.coverage({"extra": {"consts": {"eps": 1e-6}}}, row)["covered"]
    assert not rms_relation.coverage({"extra": {"consts": {"eps": 1e-6, "arch": "sm_90"}}}, row)["covered"]
