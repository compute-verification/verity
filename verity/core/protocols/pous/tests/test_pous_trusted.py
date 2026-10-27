"""`lean/TRUSTED.sha256` is what the regeneration command in `lean/README.md` writes.  `grade.sh` refuses every submission
unless each listed file matches its hash, and trusts a file nobody listed without hashing it; `check` never runs `grade.sh`
(it needs `unshare --user`), so this suite, which reruns on every change under `lean/`, holds the file to the command."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import pytest

LEAN = Path(__file__).parents[1] / "lean"
#: the trusted layer's directory as TRUSTED.sha256 names it, relative to `lean/` (the grader and its driver are here)
SECURITY = "../../../../../Security/"
COMMAND = re.compile(r"^(sha256sum ([^$\n]*)\$\(find ([^)\n]*) -name '\*\.lean' \| LC_ALL=C sort -u\) > TRUSTED\.sha256)$", re.M)
LINE = re.compile(r"([0-9a-f]{64})  (\S.*)")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def command(pkg: Path) -> tuple[str, list[str], list[str]]:
    """The README's regeneration command on one line: (the command, the files it names, the trees whose `.lean` files follow)."""
    m = COMMAND.search(re.sub(r"\\\n\s*", "", (pkg / "README.md").read_text()))
    assert m, (f"{pkg / 'README.md'} has no command `sha256sum FILES $(find TREES -name '*.lean' | LC_ALL=C sort -u) > "
               f"TRUSTED.sha256`; if the command changed shape, change `COMMAND` here with it")
    return m.group(1), m.group(2).split(), m.group(3).split()


def trusted(pkg: Path) -> list[str]:
    """The files the command hashes, in its order: those it names, then the trees' `.lean` files as `LC_ALL=C sort -u`
    orders them, once each however many trees hold them."""
    _, named, trees = command(pkg)
    found = {p.relative_to(pkg).as_posix() for t in trees for p in (pkg / t).rglob("*.lean") if p.is_file()}
    return named + sorted(found, key=str.encode)


def regenerated(pkg: Path, files: list[str]) -> str:
    return "".join(f"{sha256(pkg / f)}  {f}\n" for f in files)


def stale(pkg: Path) -> list[str]:
    """Why ``pkg``'s `TRUSTED.sha256` is not what the command writes there, one line per file at fault; empty when it is."""
    want = trusted(pkg)
    if missing := [f for f in want if not (pkg / f).is_file()]:
        return [f"{f} is named by the command but missing" for f in missing]
    have = (pkg / "TRUSTED.sha256").read_text()
    if have == regenerated(pkg, want):
        return []
    lines = have.splitlines()
    listed = {m.group(2): m.group(1) for line in lines if (m := LINE.fullmatch(line))}
    problems = [f"line {i} is not `<sha256>  <path>`: {line!r}" for i, line in enumerate(lines, 1) if not LINE.fullmatch(line)]
    problems += [f"{f} is trusted but not listed" if f not in listed else f"{f} does not match its hash in TRUSTED.sha256"
                 for f in want if listed.get(f) != sha256(pkg / f)]
    problems += [f"{f} is listed but the command does not hash it" for f in listed if f not in want]
    return problems or ["the files are listed in another order than the command's, or more than once"]


def test_trusted_hashes_are_current():
    cmd, _, _ = command(LEAN)
    problems = stale(LEAN)
    assert not problems, ("lean/TRUSTED.sha256 is not what its regeneration command writes (grade.sh refuses every "
                          "submission while a listed file differs, and never hashes an unlisted one):\n  " +
                          "\n  ".join(problems) + "\nRegenerate it, then run the audit (`bash check.sh`):\n"
                          f"  cd verity/core/protocols/pous/lean && {cmd}")


def copy(tmp: Path) -> Path:
    """The files `stale` reads, copied from `lean/` and the trusted layer into ``tmp`` at their places under `verity/` (so
    `SECURITY` names the copy), with `TRUSTED.sha256` regenerated there: the regressions start clean even when `lean/` is
    stale."""
    base = LEAN.parents[4]
    pkg = tmp / LEAN.relative_to(base)
    _, named, trees = command(LEAN)
    for rel in ["README.md", *named, *trees]:
        dest = tmp / (LEAN / rel).resolve().relative_to(base.resolve())
        dest.parent.mkdir(parents=True, exist_ok=True)
        if (LEAN / rel).is_dir():
            shutil.copytree(LEAN / rel, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(LEAN / rel, dest)
    (pkg / "TRUSTED.sha256").write_text(regenerated(pkg, trusted(pkg)))
    return pkg


def relist(pkg: Path, change) -> None:
    """Rewrite ``pkg``'s `TRUSTED.sha256` as ``change`` ({path: hash} -> lines) makes it."""
    f = pkg / "TRUSTED.sha256"
    listed = {m.group(2): m.group(1) for m in map(LINE.fullmatch, f.read_text().splitlines())}
    f.write_text("".join(f"{line}\n" for line in change(listed)))


def edited(pkg: Path) -> None:
    (pkg / "Grader/Registry.lean").write_text((pkg / "Grader/Registry.lean").read_text() + "\n")


def corrupted(pkg: Path) -> None:
    relist(pkg, lambda d: [f"{'0' * 64 if f == SECURITY + 'Definitions/Pous/Guarantees.lean' else h}  {f}" for f, h in d.items()])


def dropped(pkg: Path) -> None:
    relist(pkg, lambda d: [f"{h}  {f}" for f, h in d.items() if f != "lake-manifest.json"])


def added(pkg: Path) -> None:
    (pkg / SECURITY / "Definitions/Pous/Guarantees/Added.lean").write_text("import Definitions.Pous\n")


def extra(pkg: Path) -> None:
    relist(pkg, lambda d: [*(f"{h}  {f}" for f, h in d.items()), f"{sha256(pkg / 'README.md')}  README.md"])


def reordered(pkg: Path) -> None:
    relist(pkg, lambda d: [f"{h}  {f}" for f, h in reversed(d.items())])


@pytest.mark.parametrize("mutate, problems", [
    (edited, ["Grader/Registry.lean does not match its hash in TRUSTED.sha256"]),
    (corrupted, [f"{SECURITY}Definitions/Pous/Guarantees.lean does not match its hash in TRUSTED.sha256"]),
    (dropped, ["lake-manifest.json is trusted but not listed"]),
    (added, [f"{SECURITY}Definitions/Pous/Guarantees/Added.lean is trusted but not listed"]),
    (extra, ["README.md is listed but the command does not hash it"]),
    (reordered, ["the files are listed in another order than the command's, or more than once"]),
], ids=lambda x: getattr(x, "__name__", None))
def test_a_stale_trusted_file_is_named(tmp_path, mutate, problems):
    pkg = copy(tmp_path)
    assert stale(pkg) == []
    mutate(pkg)
    assert stale(pkg) == problems
