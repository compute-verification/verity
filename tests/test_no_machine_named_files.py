"""No tracked file is named for a machine.

A machine is a row of data: put what this machine needs in its row (`infra/cluster/descriptions/`, `deploy.toml`), and write the
code, script or unit once for every machine that has such a row.  A file named for one machine (`n1_*`, `n2_*`, a unit
`vy-n1-*`, `vy-node1-*`, `pool_n1`, `nebius1`, `nebius2`, `nebius-after-n1`) or kept under a directory named for one (`MACHINE_DIRS`)
is code that outlives its machine: when the machine goes, nothing says the file may go with it.

``ALLOWED`` holds the files that are still so named; it only shrinks.  A file named in it that is gone, or that no pattern
matches, fails too: delete its line.
"""
from __future__ import annotations

import posixpath
import re

from test_repository import _tracked

PATTERN = re.compile(r"(?:^|[-_.])(?:n1_|n2_|vy-n1-|vy-n2-|vy-node1-|pool_n1|nebius1|nebius2|nebius-after-n1)")
MACHINE_DIRS = ("infra/vcf/",)

ALLOWED = {
    "infra/cluster/src/cluster/nebius2.py",
    "infra/vcf/payload.sh",
    "infra/vcf/vcf-bench.sh",
    "infra/vcf/vcf-job-run.sh",
    "infra/vcf/vcf-publish.sh",
    "tests/infra/cluster/test_nebius2.py",
    "tools/research/src/research/pods/nebius/n2_custody.sh",
}


def named() -> set[str]:
    return {f for f in _tracked() if PATTERN.search(posixpath.basename(f)) or f.startswith(MACHINE_DIRS)}


def test_no_new_file_is_named_for_a_machine() -> None:
    new = sorted(named() - ALLOWED)
    assert not new, "files named for a machine; a machine is a row of data: put what this machine needs in its row:\n" + "\n".join(new)


def test_allowed_only_shrinks() -> None:
    stale = sorted(ALLOWED - named())
    assert not stale, "no longer named for a machine, or gone; delete these lines:\n" + "\n".join(stale)
