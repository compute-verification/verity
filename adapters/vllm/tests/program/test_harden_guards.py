"""Export guard / applicability preservation of the Bet A artifact contract (`TargetProfile`, `ArtifactIdentity`, `check_reuse`;
`verity_vllm.pipeline.build` manifest v3).

Contract under test: (a) the specialisation guards `torch.export` records — input shapes / dtypes, the (empty) range
constraints, the graph-signature facts — are captured into the manifest/artifact; (b) a build for a different declared target has
a different artifact identity even with an identical Program digest; (c) an artifact built under assumptions A is REFUSED
(fail-closed) when applied under B: device class, dtype, batch, sequence bound; (d) manifest fields are not duplicated literals —
one source of truth.

Evidence sources: the pure-Python contract module `verity_vllm.program.frontend.target_profile`, the builder source
`verity_vllm/pipeline/build.py`, and (opt-in, `HARDEN_LIVE=1`, pod venv with vLLM) two live constructions differing only in a
structural input. The checks of (a), (b), (c) and G4a/G4b read recorded constructions on a GPU host, which no tracked file holds:
they are `test_harden_guards_of_record`.

Verdict convention: assertions state the CONTRACT. Where the contract is known not to hold today the test is marked
`xfail(strict=True)` with the finding as the reason — fixing the gap turns the xfail into an XPASS failure that forces the marker
to be removed, so the finding stays load-bearing. None of these gaps is a semantic failure of a generated Program (the Program
digest DOES carry shapes/dtypes — G6); they are applicability-contract gaps (open item A4: "guards not preserved as
artefacts").

Run: laptop `python -m pytest -q tests/program/test_harden_guards.py -k "not live and not G6"` (pure Python); pod
`CUDA_VISIBLE_DEVICES="" PYTHONPATH=. HARDEN_LIVE=1 nice -n 10 python -m pytest -q -p no:cacheprovider
tests/program/test_harden_guards.py`; `HARDEN_OUT=<dir>` writes `guards_results.json`.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

import pytest

import verity.circuits
from verity_vllm.program.frontend.target_profile import ArtifactIdentity, TargetProfile, reuse_mismatches

INTEGRATION = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # adapters/vllm
CORE_SRC = os.path.dirname(os.path.dirname(os.path.abspath(verity.circuits.__file__)))  # the repository root

RESULTS: list[dict] = []


def _record(case, expected, observed, verdict, cause="", **extra):
    RESULTS.append({"case": case, "expected": expected, "observed": observed, "verdict": verdict, "cause": cause, **extra})


@pytest.fixture(scope="module", autouse=True)
def _dump_results():
    yield
    out = os.environ.get("HARDEN_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(RESULTS, open(os.path.join(out, "guards_results.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------------------------
# (d) one source of truth
# ---------------------------------------------------------------------------------------------------------


def test_G4c_no_capability_or_sm_literal_outside_target_profile():
    """The invariant, re-checked on this tree: the declared capability / SM count have one authoritative site."""
    files = ["verity_vllm/program/frontend/export_compat.py", "verity_vllm/program/frontend/vllm_meta.py", "verity_vllm/pipeline/build.py",
             *sorted(os.path.relpath(p, INTEGRATION) for p in glob.glob(os.path.join(INTEGRATION, "verity_vllm/program/frontend/rules/vllm_bindings/*.py")))]
    hits = []
    for rel in files:
        p = os.path.join(INTEGRATION, rel)
        assert os.path.exists(p), f"{rel} moved: point G4c at the file that now holds it"
        for i, line in enumerate(open(p).read().splitlines(), 1):
            # applied / declared VALUES only (the hard-coded literals): not prose that names the architecture in a docstring
            if re.search(r"DeviceCapability\(major=\d", line) or '"8.9 (sm_89' in line or re.search(r'"NUM_SMS":\s*\d', line) \
                    or re.search(r"=\s*\(8,\s*9\)", line):
                hits.append(f"{rel}:{i}: {line.strip()[:80]}")
    _record("G4c capability / NUM_SMS literals outside target_profile.py", "none", f"{hits}", "PASS" if not hits else "FAIL")
    assert not hits, hits


# ---------------------------------------------------------------------------------------------------------
# G6: the shape / dtype guard lives in the Program digest (derive level; torch needed)
# ---------------------------------------------------------------------------------------------------------


def test_G6_program_digest_carries_input_shape_and_dtype():
    torch = pytest.importorskip("torch")
    from verity_vllm.program.frontend.derive import derive_report
    from verity_vllm.program.frontend import derive_examples as ex
    torch.manual_seed(0)
    m = ex.MLP2(8, 16, 8)
    r3 = derive_report(m, (torch.empty(3, 8),), mode="reference", name="MLP2")
    r4 = derive_report(m, (torch.empty(4, 8),), mode="reference", name="MLP2")
    assert r3.ok and r4.ok, (r3.unsupported, r4.unsupported)
    assert r3.digest != r4.digest
    assert r3.provenance["source"]["example_inputs"] == [("torch.float32", [3, 8])]
    rb = derive_report(m, (torch.empty(3, 8, dtype=torch.bfloat16),), mode="reference", name="MLP2")
    assert (not rb.ok) or rb.digest != r3.digest  # another dtype: refused or another circuit, never the same digest
    _record("G6 digest vs T=3 / T=4 / bf16 input", "distinct digests (or refusal for the dtype)", f"{r3.digest[:10]} {r4.digest[:10]} bf16 ok={rb.ok}",
            "PASS", "the specialisation IS pinned in the digest; the gap is only that the reuse API cannot see the request")


# ---------------------------------------------------------------------------------------------------------
# G5 (live, opt-in): two constructions differing only in a structural input
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("HARDEN_LIVE") != "1", reason="live constructions need the pod venv (vLLM); set HARDEN_LIVE=1")
def test_G5_live_constructions_differing_in_M_are_distinct_but_not_distinguishable_at_reuse():
    out = os.environ.get("HARDEN_OUT", "/tmp/harden_out")
    arts = {}
    for M in (1, 2):
        d = os.path.join(out, f"g5_step_M{M}")
        if not os.path.exists(os.path.join(d, "artifact.json")):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.pathsep.join([INTEGRATION, CORE_SRC]), "HF_HUB_OFFLINE": "1"}
            env.setdefault("HF_HOME", "/workspace/hf" if os.path.isdir("/workspace/hf") else os.path.expanduser("~/.cache/huggingface"))
            r = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "build", "--out", d, "--wrapper", "step", "--M", str(M), "--pos0", "8"],
                               cwd=INTEGRATION, env=env, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        arts[M] = json.load(open(os.path.join(d, "artifact.json")))
    a1, a2 = arts[1], arts[2]
    assert a1["program_digest"] != a2["program_digest"] and a1["identity"] != a2["identity"]
    assert a1["target_profile"] == a2["target_profile"] and a1["applicability"] == a2["applicability"]
    # the M=2 requester is admitted to the M=1 artifact: nothing in the applicability distinguishes them
    admitted = not reuse_mismatches(ArtifactIdentity.from_json(a1), TargetProfile.from_json(a2["target_profile"]), a2["applicability"])
    _record("G5 live: step M=1 vs M=2", "distinct digest+identity (PASS); reuse of M=1 by an M=2 requester refused (GAP)",
            f"digests {a1['program_digest'][:10]}≠{a2['program_digest'][:10]}; identities differ; applicability equal; M=2 requester admitted to M=1 artifact={admitted}",
            "PASS + GAP", "structural inputs not in applicability")
    assert admitted, "if this fails the gap was closed: update G1c/G3c"
