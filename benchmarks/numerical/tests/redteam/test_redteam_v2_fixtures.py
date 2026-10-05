"""red-team-v2 fixtures: the gkr chain bundles load and match their manifest, and the zero_swap vacuity holds.

These are cheap structural / reference checks (they do NOT invoke the Rust verifier; the ACCEPTED/REJECTED verdicts
against the real `verity-gkr` binary are recorded in backends/numerical/reports/redteam_v2.{md,json}).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

from verity_numerical.redteam.v2_forge import zero_swap_scan

REPO = Path(__file__).resolve().parents[4]  # repo root (…/tests/redteam -> numerical -> backends -> repo)
GKR = REPO / "fixtures" / "redteam-v2" / "gkr"


def _rows(path: Path) -> tuple[int, int, list[list[int]]]:
    b = path.read_bytes()
    n, m = struct.unpack("<QQ", b[:16])
    rows, off = [], 16
    for _ in range(n):
        rows.append(list(struct.unpack(f"<{m}Q", b[off:off + 8 * m])))
        off += 8 * m
    return n, m, rows


def test_manifest_and_bundles_load() -> None:
    manifest = json.loads((GKR / "manifest.json").read_text())
    ncols_u, ncols_e = manifest["unit_columns"], manifest["epilogue_columns"]
    assert ncols_u == 264 and ncols_e == 16, (ncols_u, ncols_e)
    for case in manifest["cases"]:
        d = GKR / "neg" / case
        assert (d / "circuit.txt").exists() and (d / "epilogue.txt").exists() and (d / "chain.txt").exists(), case
        nu, mu, urows = _rows(d / "units.bin")
        ne, me, erows = _rows(d / "epilogue.bin")
        npb, mpb, prows = _rows(d / "public.bin")
        assert mu == ncols_u and me == ncols_e and mpb == 1, case
        assert nu % ne == 0 and npb == ne, case          # units a multiple of the VU count; one public word per VU


def test_chain_spec_stripped_has_empty_chain() -> None:
    # the accepted forgery: an empty chain.txt drops the state/ovf/sgn links
    assert (GKR / "neg" / "chain_spec_stripped" / "chain.txt").read_text() == ""


def test_zero_swap_is_vacuous() -> None:
    res = zero_swap_scan(n=1500)
    assert res["claim_vacuous"] and not res["divergences"], res["divergences"][:2]
    assert res["all_zero_corner"]["word_pos0"] == res["all_zero_corner"]["word_neg0"]
