"""The challenger's and the verifier's pin scripts and gate wrappers (`pods/nebius/challenger/`, `pods/nebius/verifier/`)
against the tree they pin: every file a pin requires in the archive is the tree's, every `lake build` names an executable
its package's lakefile declares, the pin logs through the entry the wrapper execs, and the wrapper re-execs itself in
the form its setup.sh's sudoers line admits."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PODS = REPO / "tools/research/src/research/pods/nebius"


def required(pin: str) -> list[str]:
    files = re.findall(r'\[ -f "\$tmp/([^"$]+)" \]', pin)
    for block in re.findall(r"^\s*for f in (.*?); do$", pin, re.M | re.S):
        files += block.replace("\\\n", " ").split()
    return files


def lake_builds(pin: str) -> list[tuple[str, str]]:
    dirs = dict(re.findall(r"^(\w+)=\$T/(\S+)$", pin, re.M))
    found = []
    for where, names in re.findall(r"cd '([^']+)' && lake build ([a-z0-9 -]+?)\s*(?:\"|&&)", pin):
        d = dirs[where[1:]] if where.startswith("$") and where[1:] in dirs else where.removeprefix("$T/")
        found += [(d, n) for n in names.split()]
    return found


@pytest.mark.parametrize("service, gate, entry", [
    ("challenger", "vy-challenger-gate", "verity/protocols/compose/verifier/live_challenger/auditor.py"),
    ("verifier", "vy-verifier-gate", "verity/protocols/resource_accounting/memory/poub/attester/live_verifier/host.py"),
])
def test_a_pin_names_the_trees_files_and_executables_and_the_wrapper_execs_its_entry(service, gate, entry):
    pin = (PODS / service / "pin.sh").read_text()
    wrapper = (PODS / service / gate).read_text()
    files = required(pin)
    assert entry in files and [f for f in files if not (REPO / f).is_file()] == []
    builds = lake_builds(pin)
    assert builds
    for pkg, exe in builds:
        lakefile = tomllib.loads((REPO / pkg / "lakefile.toml").read_text())
        assert exe in [e["name"] for e in lakefile.get("lean_exe", [])], f"{pkg}/lakefile.toml declares no {exe}"
    assert f'"$T/{entry}" pinned "$c"' in pin
    assert f'exec /usr/bin/python3 -I /opt/vy-{service}/current/{entry} gate "$@"\n' in wrapper
    sudo = f"/usr/local/sbin/{gate} --ssh"
    assert f'exec sudo -n {sudo} "${{SSH_ORIGINAL_COMMAND:-}}" --key "${{1:--}}"' in wrapper
    assert f"'research ALL=(root) NOPASSWD: {sudo} *'" in (PODS / service / "setup.sh").read_text()
