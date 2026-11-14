"""`compiled verify`, which a served Commit's script runs after its treatment: the Build of record held to the Build the treatment
rederived from its own compile artifact, torch-free.  It passes when the Build of record's digest is the digest of its rows and the
two Builds agree on rows, Program and compile artifact, with the rederived one complete; otherwise it exits 3 and its record says
why."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.query.manifest.format import manifest_digest
from verity_vllm.records.schema_ids import MANIFEST_VERIFY_SCHEMA

PD = "2" * 64
ARTIFACT = {"digest": "a" * 128}


def _rows(n: int = 3) -> list[dict]:
    return [{"family": "served", "program_digest": PD, "request_id": "r0", "step": s, "invocation": 0, "op_path": "c00.mul",
             "output_member": "out", "element_range": [0, 4], "rank": 0} for s in range(n)]


def _builds() -> tuple[dict, dict]:
    """(the Build of record, the Build the treatment rederived), alike."""
    rows = _rows()
    used = {"identities": rows, "manifest_digest": manifest_digest(rows), "program_digest": PD, "artifact": dict(ARTIFACT)}
    rebuilt = {"manifest_digest": used["manifest_digest"], "program_digest": PD, "complete": True, "populations": {"identities": len(rows)},
               "artifact": dict(ARTIFACT)}
    return used, rebuilt


def _verify(tmp_path, capsys, used: dict, rebuilt: dict) -> tuple[int, str, dict]:
    paths = [tmp_path / name for name in ("build.json", "build_rebuilt.json", "manifest_verify.json")]
    paths[0].write_text(json.dumps(used))
    paths[1].write_text(json.dumps(rebuilt))
    rc = cli.main(["compiled", "verify", "--manifest", str(paths[0]), "--rebuilt", str(paths[1]), "--out", str(paths[2])])
    return rc, capsys.readouterr().out.strip().splitlines()[-1], json.loads(paths[2].read_text())


def test_a_build_the_treatment_rederived_alike_passes(tmp_path, capsys):
    used, rebuilt = _builds()
    rc, line, rec = _verify(tmp_path, capsys, used, rebuilt)
    assert rc == 0 and rec["ok"] is True and "why" not in rec and rec["schema"] == MANIFEST_VERIFY_SCHEMA
    assert rec["used_digest"] == rec["used_rows_digest"] == rec["rebuilt_digest"] == used["manifest_digest"]
    assert (rec["used_identities"], rec["rebuilt_identities"], rec["rebuilt_complete"]) == (3, 3, True)
    assert rec["used_program_digest"] == rec["rebuilt_program_digest"] == PD
    assert rec["used_artifact"] == rec["rebuilt_artifact"] == ARTIFACT["digest"]
    assert (rec["manifest"], rec["rebuilt_from"]) == (str(tmp_path / "build.json"), str(tmp_path / "build_rebuilt.json"))
    assert line.startswith(f"[compiled_manifest verify] ok=True used {used['manifest_digest'][:16]} rows {used['manifest_digest'][:16]} ")
    assert line.endswith(f"-> {tmp_path / 'manifest_verify.json'}") and " -- " not in line


@pytest.mark.parametrize("side,change,why", [
    ("used", {"identities": _rows(2)}, "(edited after digesting); rows digest"),
    ("rebuilt", {"manifest_digest": "0" * 64, "populations": {"identities": 2}}, "!= the treatment-derived manifest 0000000000000000 (3 vs 2 identities)"),
    ("rebuilt", {"program_digest": "3" * 64}, f"program_digest used {PD[:16]} != derived from the treatment's output code {'3' * 16}"),
    ("rebuilt", {"complete": False, "unmodelled": "a launch"}, "treatment-derived manifest not complete: unmodelled a launch"),
    ("rebuilt", {"artifact": {"digest": "b" * 128}}, f"compile artifact used {'a' * 16} != the treatment's {'b' * 16}"),
], ids=["rows-edited", "other-rows", "other-program", "incomplete", "other-artifact"])
def test_a_build_that_differs_is_refused_and_says_why(tmp_path, capsys, side, change, why):
    used, rebuilt = _builds()
    (used if side == "used" else rebuilt).update(change)
    rc, line, rec = _verify(tmp_path, capsys, used, rebuilt)
    assert rc == 3 and rec["ok"] is False and why in rec["why"]
    assert line.startswith("[compiled_manifest verify] ok=False ") and f" -- {rec['why']} -> " in line


@pytest.mark.parametrize("side", ["--manifest", "--rebuilt"])
@pytest.mark.parametrize(("form", "refusal"), [("{}/", NotADirectoryError), ("", FileNotFoundError)], ids=["slash-after", "empty"])
def test_a_build_is_opened_at_the_path_as_given(tmp_path, monkeypatch, side, form, refusal):
    """A Build's path is opened as it is given, nothing normalised on the way: a file's name with a slash after it names no
    directory, the empty path names nothing, and no record is written."""
    used, rebuilt = _builds()
    args = {"--manifest": str(tmp_path / "build.json"), "--rebuilt": str(tmp_path / "build_rebuilt.json")}
    (tmp_path / "build.json").write_text(json.dumps(used))
    (tmp_path / "build_rebuilt.json").write_text(json.dumps(rebuilt))
    args[side] = form.format(args[side])
    out = tmp_path / "manifest_verify.json"
    monkeypatch.chdir(tmp_path)     # the suite's read guard takes an open of "" for a read of the directory the test runs in
    with pytest.raises(refusal):
        cli.main(["compiled", "verify", "--manifest", args["--manifest"], "--rebuilt", args["--rebuilt"], "--out", str(out)])
    assert not out.exists()


LEFT_OUT = "left out"


@pytest.mark.parametrize("held", [LEFT_OUT, None, [], {}, "", 0, False], ids=repr)
def test_a_field_a_build_leaves_out_is_read_as_empty(tmp_path, capsys, held):
    """A Build with no rows, no compile artifact and no populations, each field left out, null or holding an empty value: the
    rows are none, their digest is the digest of no row, and neither side has an artifact digest."""
    none = manifest_digest([])
    used: dict = {"manifest_digest": none, "program_digest": PD}
    rebuilt: dict = {"manifest_digest": none, "program_digest": PD, "complete": True}
    if held != LEFT_OUT:
        used.update(identities=held, artifact=held)
        rebuilt.update(populations=held, artifact=held)
    rc, _line, rec = _verify(tmp_path, capsys, used, rebuilt)
    assert rc == 0 and rec["ok"] is True
    assert (rec["used_rows_digest"], rec["used_identities"], rec["rebuilt_identities"]) == (none, 0, None)
    assert rec["used_artifact"] is None and rec["rebuilt_artifact"] is None


def test_importing_the_commands_module_loads_neither_numpy_nor_torch_nor_the_core():
    """`cli` imports `verity_vllm.acquire.compiled` when `compiled` is named.  Only `assign_wrappers`, which the served Build calls,
    imports the Build's refusal from `acquire.inductor_templates`, whose module loads numpy and the Definitions."""
    code = ("import sys; before = set(sys.modules); import verity_vllm.acquire.compiled; "
            "print(sorted({m.split('.')[0] for m in set(sys.modules) - before} & {'numpy', 'torch', 'verity'}))")
    child = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)},
                           capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "[]"
