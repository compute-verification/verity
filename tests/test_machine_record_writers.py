"""Every function under tools/research and infra/ that writes under machines.d is one known to write a machine record only
through `research.pods.registry.write_key` (one top-level key) or `write_entry` (a whole record), or to write no record:
a writer that appended a mark put it inside a record's [properties] table, where placement never read it."""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: every function under tools/research and infra/ that writes a file and names machines.d or a record writer, and why it
#: may: each writes a record only through `write_key` (one key) or `write_entry` (a whole record), or writes no record
KNOWN = {
    "tools/research/src/research/pods/registry.py::write_entry": "the whole-record writer",
    "tools/research/src/research/pods/registry.py::write_key": "the one-key writer",
    "tools/research/src/research/pods/registry.py::_write_checked": "register: write_entry",
    "tools/research/src/research/pods/registry.py::main": "import: write_entry; the families file is no record",
    "tools/research/src/research/pods/registry.py::stopped": "write_key's text (with_key) through _replace",
    "tools/research/src/research/pods/nebius/fleet.py::set_key": "write_key",
    "tools/research/src/research/pods/nebius/fleet.py::watch_unreachable": "set_key; its own write is fleet's state file",
    "infra/control/vm_stop.py::one_pass": "writes its own state file, no record",
}
_NAMES = ("MACHINES_D", "machines.d", "write_dir", "registry_dir", "REGISTRY_ENV", "write_entry", "write_key", "with_key")
_WRITES = {"write_text", "write_bytes", "replace", "rename", "dump", "write_entry", "write_key", "_replace"}


def _writers() -> set[str]:
    found = set()
    for root in ("tools/research", "infra"):
        for f in sorted((REPO / root).rglob("*.py")):
            text = f.read_text()
            if not any(n in text for n in _NAMES):
                continue
            for fn in ast.walk(ast.parse(text)):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name in ("with_key", "_replace"):
                    continue
                if not any(n in (ast.get_source_segment(text, fn) or "") for n in _NAMES):
                    continue
                for c in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                    called = c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")
                    mode = [a.value for a in [*c.args[1:2], *(k.value for k in c.keywords if k.arg == "mode")]
                            if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                    if called in _WRITES or (called == "open" and any(m[:1] in "wax" for m in mode)):
                        found.add(f"{f.relative_to(REPO)}::{fn.name}")
                        break
    return found


def test_every_writer_under_machines_d_is_a_known_one():
    """Rule: a new writer of a machine record is seen; weakening: a third writer that appends a mark into a table, unseen."""
    found = _writers()
    assert found - KNOWN.keys() == set(), ("a new function writes under machines.d: write a record's key through "
                                           "research.pods.registry.write_key (or a whole record through write_entry), then "
                                           "add it to KNOWN with why")
    assert KNOWN.keys() - found == set(), "KNOWN names a writer that is gone: drop it"
