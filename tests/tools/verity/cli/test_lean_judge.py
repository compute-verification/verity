"""`verity lean judge`: the source judge on a fixture package with one orphan module and one import outside its layer, and
on the same package with both mended."""
from __future__ import annotations

import shutil
from pathlib import Path

from verity_cli import judge

FIXTURE = Path(__file__).resolve().parents[4] / "tests" / "tools" / "verity" / "lean" / "fixtures" / "judge"


def test_each_failure_with_its_fix(capsys):
    assert judge.main([str(FIXTURE)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("judge: 2 failure(s)")
    assert out[1].startswith("  orphan.file Judged/Orphan.lean: orphan: Judged/Orphan.lean is not imported by the roots")
    assert out[1].endswith("; fix: add Judged.Orphan to this lock's `roots`, or import it from a root")
    assert out[2].startswith("  layers.import Judged.Defs: layers: Judged.Defs imports Judged.Uses, outside")
    assert out[2].endswith("; fix: add Judged.Uses to Judged.Defs's `imports` in this lock, or move the import")
    assert len(out) == 3


def test_mended_package_passes(tmp_path, capsys):
    pkg = tmp_path / "judge"
    shutil.copytree(FIXTURE, pkg)
    (pkg / "Judged" / "Orphan.lean").unlink()
    (pkg / "Judged" / "Defs.lean").write_text("module\n\npublic def Judged.two : Nat := 2\n")
    (pkg / "Judged.lean").write_text("module\n\npublic import Judged.Defs\npublic import Judged.Uses\n")
    assert judge.main([str(pkg)]) == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith("judge: 0 failure(s)")
