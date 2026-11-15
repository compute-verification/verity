"""The ids two of the engine's records carry: `verity_vllm.engine.meta_inventory.live_document`'s document (the ranks'
inventory records in a meta build's layout) and `verity_vllm.engine.kernel_identity.collect`'s record of a scan.  No vLLM and no
GPU: the ranks' records are written here and the scan is of an empty directory."""
from __future__ import annotations

from verity_vllm.engine import kernel_identity as KI
from verity_vllm.engine import meta_inventory as MI

INVENTORY = "verity-vllm/meta-inventory/v1"
KERNELS = "verity-vllm/kernel-identity/v1"


def test_a_live_document_is_the_ranks_by_number_under_the_inventorys_id_with_the_callers_fields():
    one, zero = {"rank": 1, "modules": {"b": {}}}, {"rank": 0, "modules": {"a": {}}}
    doc = MI.live_document([one, zero], repo="org/model", revision="abc")
    assert doc == {"schema": INVENTORY, "repo": "org/model", "revision": "abc", "ranks": {"0": zero, "1": one}}
    assert list(doc["ranks"]) == ["0", "1"]
    assert MI.live_document([]) == {"schema": INVENTORY, "ranks": {}}
    assert MI.SCHEMA == INVENTORY


def test_a_scan_of_no_cache_is_a_kernel_identity_record_under_its_id(tmp_path):
    doc = KI.collect(str(tmp_path / "no-inductor"), None)
    assert doc["schema"] == KERNELS
    assert doc["wrappers"] == [] and doc["kernels"] == {} and doc["cubins"] == []
    assert KI.collect(None)["schema"] == KERNELS
