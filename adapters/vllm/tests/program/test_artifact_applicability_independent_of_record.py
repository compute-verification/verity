"""`test_artifact_applicability_independent`'s static check on the reference construction hosts' recorded manifests (`out/gen`,
written on GPU hosts, which no tracked file holds). Skipped where they are absent.
"""
from __future__ import annotations

import json

import pytest

from tests.program.test_artifact_applicability_independent import B0_REQUEST_8_4_DIGEST, R9_REFERENCE_MANIFESTS, REPO


def test_static_reference_hosts_share_the_program_digest():
    """(iii)/(v) on the recorded artefacts (laptop-runnable): every construction host that built B0 request 8/4 for the sm_89
    profile — a 4090 with its GPU visible, a host without a GPU, an L40S with it hidden — recorded the same Program digest."""
    seen = []
    for p in R9_REFERENCE_MANIFESTS:
        if p.exists():
            m = json.load(open(p))
            seen.append((str(p.relative_to(REPO)), (m.get("program") or {}).get("digest"), (m.get("construction") or {}).get("cuda_available")))
    if not seen:
        pytest.skip("reference manifests not in this checkout (sparse tree without out/gen)")
    assert all(d == B0_REQUEST_8_4_DIGEST for _p, d, _c in seen), seen
    print(f"\n[cb-test] reference hosts: {seen}")
