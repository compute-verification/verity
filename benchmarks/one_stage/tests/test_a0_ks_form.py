"""`a0.lean_verify`, the Lean verify every one-stage audit's session goes through (a5's and records_audit's, prove_bundle's),
adds `--ks-form` to a `--zk` session whose circuit's ports are all row/v2 rows, and always when the audit cites
`SampledKSLiveV3` (``ks_form=True``); a plain session or a row/v1 one gets none of itself. The verdict says whether it ran.
Read off a stubbed run, so no Lean is built."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a0  # noqa: E402

V2 = {"ports": [["x", 64, 0]], "row_schemas": [["hm96-sha512/row/v2", 1024]]}
V1 = {"ports": [["x", 64, 0]], "row_schema": "hm96-sha512/row/v1"}


def _stage(tmp: Path, meta: dict) -> tuple[dict, Path]:
    st = tmp / "stage"
    st.mkdir()
    (st / "circuit.txt").write_text(f"flock-circuit toy\nMETA {json.dumps(meta)}\n")
    (st / "pub-4.bin").write_bytes(b"{}\n")
    out = tmp / "out"
    (out / "sessions" / "s0").mkdir(parents=True)
    (out / "sessions" / "s0" / "session.json").write_text("{}")
    return {"dir": st}, out


@pytest.mark.parametrize("meta, zk, ks_form, want", [
    (V2, True, None, True), (V1, True, None, False), (V2, False, None, False),
    (V1, True, True, True), (V2, True, True, True),
], ids=["row/v2 zk", "row/v1 zk", "row/v2 plain", "row/v1 cites V3", "row/v2 cites V3"])
def test_ks_form_goes_to_a_row_v2_zk_session_and_whenever_the_audit_cites_v3(tmp_path, monkeypatch, meta, zk, ks_form, want):
    st, out = _stage(tmp_path, meta)
    ran = []

    def run(cmd, log, env=None, timeout=3600):
        ran.append(cmd)
        if cmd[1] == "archive-put":
            return subprocess.CompletedProcess(cmd, 0, "".join(f"k{i} {f}\n" for i, f in enumerate(cmd[3:])), "")
        return subprocess.CompletedProcess(cmd, 0, 'VERDICT {"accepted": true}\n', "")

    monkeypatch.setattr(a0, "run", run)
    v = a0.lean_verify("flock-verify", st, 4, out, zk=zk, ks_form=ks_form)
    verify = ran[-1]
    assert verify[1] == "verify" and ("--zk" in verify) is zk and ("--ks-form" in verify) is want, verify
    assert v["accepted"] is True and v["ks_form"] is want
