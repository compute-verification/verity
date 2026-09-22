"""[CAPCORR-2] Regression test for GUIDANCE §50: every sampled VU in the certificate's challenge must be a relation of
the PINNED program.

On the certified record INTPAD2-B0-20260909T013519Z (program pin `d4f4b1f1…` = Serve_v1 / DERIVE's `GreedyServe`), the
challenge samples 3 `norm.rmsnorm` VUs at non-final prefill positions that the pinned program does not compute (vLLM
applies the final RMSNorm to every prefill row; the pinned program only to the sampled row). The first test therefore
FAILS today and is marked `xfail(strict=True)`: it turns green — and the marker must be removed — once either fix of §50
lands (A: challenge population drawn from the pinned program's relations; B: pin the all-rows program `a25fc156…`).

The second test is the same statement against `a25fc156…` (DERIVE's `--rows vllm`), located through DERIVE's verified
Serve_v1 bijection + the 31 extra final-norm nodes (`scripts/capcorr_a25_rows.py`); it needs the map file
(`derive_cli serve --rows vllm --map-to-family` output `serve_v1-map.json`) via `CAPCORR_A25_MAP` and is skipped otherwise.

Heavy (derives the program: ~10-20 s CPU, torch on the meta device); run on the pod, not on the laptop.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (REPO, REPO / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RECORD = ROOT / "results" / "INT-PAD-2" / "INTPAD2-B0-20260909T013519Z"
PINNED_DIGEST = "d4f4b1f1"
ALL_ROWS_DIGEST = "a25fc156997e362c355ec076d8d998ccbb3ed55dcdece5c2a56a4312099f75ce"
GUIDANCE_50 = ("GUIDANCE-2026-09-07.md §50: the certificate's challenge samples 3 prefill `norm.rmsnorm` VUs that are not "
               "relations of the pinned program d4f4b1f1… (vLLM final-norms every prefill row; the program only the sampled row); "
               "fix A (population from the program) or fix B (pin a25fc156…) is a user gate — remove this marker with the fix")

pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not (RECORD / "verify" / "challenge.json").exists(), reason="certified record not present")


def _sampled_vus() -> list[str]:
    ch = json.loads((RECORD / "verify" / "challenge.json").read_text())
    return sorted(set(ch["sample"]) | set(ch.get("adjacent_vus", [])))


def _unlocated(cmap, vus: list[str]) -> list[str]:
    from verity_vllm.correspondence.core import CaptureIdentity, NotInProgram

    out = []
    for v in vus:
        try:
            cmap.locate(CaptureIdentity.parse(v))
        except (NotInProgram, KeyError):
            out.append(v)
    return out


@pytest.mark.xfail(strict=True, reason=GUIDANCE_50)
def test_every_sampled_vu_is_a_relation_of_the_pinned_program():
    from capcorr_derived_program import program as derived_program
    from verity_vllm.correspondence.program import StructuralCircuitMap

    pins = json.loads((RECORD / "certificate.json").read_text())
    prog = derived_program(32, 15)
    digest = prog.annotations["capcorr_derive_summary"]["digest"]
    assert digest.startswith(PINNED_DIGEST), digest
    vus = _sampled_vus()
    assert len(vus) == 1022, len(vus)
    unlocated = _unlocated(StructuralCircuitMap(prog), vus)
    assert unlocated == [], f"{len(unlocated)} sampled VU(s) are not relations of the pinned program {digest[:8]}…: {unlocated}"


def test_every_sampled_vu_is_a_relation_of_the_all_rows_program():
    mp = os.environ.get("CAPCORR_A25_MAP")
    if not mp or not Path(mp).exists():
        pytest.skip("CAPCORR_A25_MAP (DERIVE's serve_v1-map.json for a25fc156…) not set")
    from capcorr_a25_rows import MappedCircuitMap, derive_a25, serve_v1

    rep = derive_a25(32, 15)
    assert rep.digest == ALL_ROWS_DIGEST, rep.digest
    m = json.loads(Path(mp).read_text())
    assert m["derived_digest"] == ALL_ROWS_DIGEST
    cmap = MappedCircuitMap(rep.program, serve_v1(32, 15), m)
    vus = _sampled_vus()
    unlocated = _unlocated(cmap, vus)
    assert unlocated == [], unlocated
    assert cmap.stats["via_extra"] == 3, cmap.stats
