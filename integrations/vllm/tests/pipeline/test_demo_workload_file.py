"""The vLLM demo's workload (`examples/vllm/workloads/`) is a fixture as the integration's are (`test_workload_file`): its directory's
SHA256SUMS lists it and fixtures/artifacts.json registers it."""
from __future__ import annotations

from tests.pipeline.test_workload_file import registered_and_listed
from verity_vllm import config as C

DEMO = C.REPO / "examples" / "vllm" / "workloads"


def test_the_demos_one_workload_is_a_fixture_its_directory_lists_and_the_registry_holds():
    registered, listed = registered_and_listed(DEMO)
    assert registered == listed, (f"in fixtures/artifacts.json, not in {DEMO / C.DIGESTS}: {sorted(registered - listed)}; "
                                  f"in it, not registered: {sorted(listed - registered)}")
    assert len(C.stored_workloads(DEMO)) == 1
