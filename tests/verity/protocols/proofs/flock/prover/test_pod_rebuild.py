"""A second build in one FLOCK_WORK applies `pod/60-circuit.sh`'s patches again, on the checkout `pod/20-gpu-link.sh` reset
with `git checkout -- .`, which restores tracked files only: every file such a patch adds is removed before it is applied,
by an `rm` of 20-gpu-link.sh or one of 60-circuit.sh's ahead of that `git apply`, else the rebuild stops there."""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

PROVER = (Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proofs" / "flock" / "prover")
CIRCUIT = (PROVER / "pod" / "60-circuit.sh").read_text()
RESET = (PROVER / "pod" / "20-gpu-link.sh").read_text()
APPLY = re.compile(r"git apply \$REPO/verity/protocols/proofs/flock/prover/(\S+\.patch)")


def removed(script: str) -> list[str]:
    """The paths (globs as written) the `rm` commands of `script` name."""
    return [w for args in re.findall(r"\brm\s+([^&;|)\n]+)", script.replace("\\\n", " "))
            for w in args.split() if not w.startswith("-")]


def added(patch: str) -> list[str]:
    """The files `patch` creates."""
    out, path = [], None
    for line in (PROVER / patch).read_text().splitlines():
        if m := re.match(r"diff --git a/\S+ b/(\S+)$", line):
            path = m.group(1)
        elif line.startswith("new file mode") and path:
            out.append(path)
    return out


def cleared(path: str, names: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, n) or path.startswith(n.rstrip("/") + "/") for n in names)


@pytest.mark.parametrize("at,patch", [(m.start(), m.group(1)) for m in APPLY.finditer(CIRCUIT)])
def test_each_file_a_patch_adds_is_removed_before_it_is_applied(at: int, patch: str):
    names = removed(RESET) + removed(CIRCUIT[:at])
    assert [p for p in added(patch) if not cleared(p, names)] == [], (
        f"{patch} adds these files, which a rebuild in the same FLOCK_WORK finds already there: `rm -f` them ahead of its "
        f"`git apply` in pod/60-circuit.sh")


def test_the_wide_patch_is_one_that_adds_files():
    assert APPLY.search(CIRCUIT) and len(added("flock-wide-b684b12.patch")) == 4
