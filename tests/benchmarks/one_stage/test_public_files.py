"""A launcher's copy into its run's output directory takes no private file, and every such copy goes through `public_files.py`."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "one_stage"))

import public_files as PF  # noqa: E402

#: one path of each private kind, relative to a launcher's work directory
PRIVATE = ("a/private/h1/key", "a/x.salts/s0", "a/y.salts", "a/saved/k.npz", "saved-a9.tar", "saved-a11.tar", "a/index-salt.bin",
           "a/salts.bin", "a/weights.key", "a/keys.bin", "a/k-private.bin", "a/m0-private.bin", "inst-0.bin", "windows/7/rows",
           "a/values/v.json", "a/prover/rows.bin", "a/requests.jsonl", "a/w.safetensors", "a/t.pt", "a/t.npy", "a/h1/x.key")
PUBLIC = ("a/summary.json", "a/audit.json", "a11.log", "a/session.json")


def copied(src: Path, dest: Path) -> set[str]:
    """What the loops' pipeline copies of ``src``: `find`, the filter, `cp --parents`."""
    cmd = (f'cd "{src}" && find . -type f -print0 | python3 "{PF.__file__}" filter | xargs -0 -r -I{{}} cp --parents {{}} "{dest}/"')
    subprocess.run(["bash", "-c", cmd], check=True)
    return {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}


def test_a_directory_holding_each_private_name_copies_none_of_them(tmp_path: Path) -> None:
    src, dest = tmp_path / "work", tmp_path / "out"
    dest.mkdir()
    for rel in (*PRIVATE, *PUBLIC):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text("x")
    assert copied(src, dest) == set(PUBLIC)


@pytest.mark.parametrize("rel", PRIVATE)
def test_each_private_name_is_refused_with_a_reason(rel: str) -> None:
    assert PF.private_why(rel)


def test_the_three_witness_tables_are_the_sender_s() -> None:
    import served_verdicts as SV
    assert PF.witness is SV.witness


def test_every_copy_of_ours_into_an_output_directory_goes_through_the_filter() -> None:
    """Each `xargs ... cp` loop of a launcher under `benchmarks/one_stage` and `benchmarks/prove_path` takes its
    paths from `public_files.py filter`, and no launcher names a private file in its own `find`."""
    bad, own = [], []
    for f in sorted([*(ROOT / "benchmarks" / "one_stage").glob("*.sh"), *(ROOT / "benchmarks" / "prove_path").glob("*.sh")]):
        text = f.read_text()
        for m in re.finditer(r"xargs -0[^\n]* cp ", text):
            if not text[max(0, m.start() - 22):m.start()].endswith('" filter | ') or "\nPUBF=" not in text:
                bad.append(f"{f.name}:{text.count(chr(10), 0, m.start()) + 1}")
        own += [f.name for pat in ("*-private.bin", "index-salt.bin", "salts.bin", "*private*") if f"! -name '{pat}'" in text
                or f"! -name {pat}" in text]
    assert not bad, f"copies not through public_files.py: {bad}"
    assert not own, f"launchers that keep a list of private names of their own: {sorted(set(own))}"


def test_no_launcher_writes_staged_values_under_its_output() -> None:
    """The a9 launcher keeps the staged units' values outside `$OUT`."""
    for name in ("a9.sh",):
        text = (ROOT / "benchmarks" / "one_stage" / name).read_text()
        assert not re.search(r'\$OUT/(a9/saved|saved-)', text), name
        assert "tar -cf" not in text, name
