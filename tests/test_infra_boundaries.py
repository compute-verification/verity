"""Nothing outside `infra/` imports `infra/` (the hub layout: the machines, the cluster, deploys and messaging depend on the
tools, never the other way). `research doctor|deploy|keeper|msg|slack` reach `ops` by exec, with no import. ``KNOWN`` lists every
site outside `infra/` that does import an infra package, with its reason, and may only shrink: a new site fails, and so does a
listed site that is gone from a file that exists, so the list stays exact.

An import is an import statement, a string handed to ``importlib.import_module``, ``__import__`` or ``pytest.importorskip``, or
a ``"module:attr"`` string naming what to load at run time.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: infra's import packages, from each distribution's wheel configuration.
INFRA = frozenset(Path(p).name for pyproject in sorted((ROOT / "infra").glob("*/pyproject.toml"))
                  for p in tomllib.loads(pyproject.read_text())["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
SKIP_DIRS = {"target", "node_modules", "__pycache__"}
MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}

#: Every file outside infra/ that imports an infra package, with each module it imports. Move the file into infra/ (or drop
#: the use) and delete its line; a new line needs a reason the dependency can't be an exec of `ops`.
KNOWN = {
    # The spec-change alert posts to Slack from `research data`'s hook (comms').
    ("tools/research/src/research/spec_alert.py", "verity_ops.slack"),
    ("tools/research/tests/test_spec_alert.py", "verity_ops.slack"),
    # The monitors' relay posts their alerts to Slack from a VM (on node 1 they only write files); monitors.py moves with
    # pods/nebius. test_monitors checks the relay, test_nebius_lean_sandbox the lean sandbox's deploy entries
    # and test_nebius_write_probe the probe's relay.
    ("tools/research/src/research/monitors.py", "verity_ops"),
    ("tools/research/tests/test_nebius_lean_sandbox.py", "verity_ops"),
    ("tools/research/tests/test_monitors.py", "verity_ops"),
    ("tools/research/tests/test_nebius_write_probe.py", "verity_ops"),
    # Node 1's alert loop and the probes post to Slack from pods/nebius, which moves to infra/nebius after the layout move:
    # node 1 runs research.monitors from its installed package paths, and node 2's scripts read pods/nebius/sky in job trees.
    ("tools/research/src/research/pods/nebius/n1_alerts.py", "verity_ops"),
    ("tools/research/src/research/pods/nebius/preflight_probe.py", "verity_ops"),
    ("tools/research/src/research/pods/nebius/write_probe.py", "verity_ops"),
    # The control pod's disk alert, which the layout move takes to infra/control.
    ("tools/research/control/control_disk_alert.py", "verity_ops"),
}


def _modules(path: Path, text: str) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(text, filename=str(path))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
              and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) in IMPORT_CALLS):
            out.add(node.args[0].value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_ATTR.fullmatch(node.value):
            out.add(node.value.partition(":")[0])
    return {m for m in out if m.split(".")[0] in INFRA}


def _files() -> list[Path]:
    out = []
    for top in sorted(ROOT.iterdir()):
        if top.name.startswith(".") or top.name == "infra" or top.name in SKIP_DIRS:
            continue
        out += [top] if top.suffix == ".py" else sorted(top.rglob("*.py")) if top.is_dir() else []
    return out


@functools.cache
def _sites() -> frozenset[tuple[str, str]]:
    sites = set()
    for path in _files():
        rel = path.relative_to(ROOT)
        if any(part.startswith(".") or part in SKIP_DIRS for part in rel.parts):
            continue
        text = path.read_text(encoding="utf-8")
        if any(name in text for name in INFRA):
            sites.update((rel.as_posix(), m) for m in _modules(path, text))
    return frozenset(sites)


def test_infra_packages_are_found_under_infra():
    """The packages come from infra/*/pyproject.toml: one that resolved outside infra/ would leave its importers unchecked."""
    assert "verity_ops" in INFRA
    for name in INFRA:
        origin = Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(ROOT / "infra"), f"{name} is at {origin}, outside infra/"


def test_nothing_outside_infra_imports_infra_outside_known():
    new = sorted(_sites() - KNOWN)
    assert not new, ("a file outside infra/ imports an infra package; exec `ops` instead (research/cli.py's hand_to_ops), move "
                     "the file into infra/, or add it with a reason to KNOWN in tests/test_infra_boundaries.py:\n"
                     + "\n".join(f'    ("{f}", "{m}"),' for f, m in new))


def test_every_known_site_still_exists():
    # a listed file that does not exist is a site the layout move took into infra/, or an open PR's
    gone = sorted((f, m) for f, m in KNOWN - _sites() if (ROOT / f).exists())
    assert not gone, ("these imports of infra are gone; delete them from KNOWN in tests/test_infra_boundaries.py:\n"
                      + "\n".join(f'    ("{f}", "{m}"),' for f, m in gone))
