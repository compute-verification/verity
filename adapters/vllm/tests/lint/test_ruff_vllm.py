"""Ruff's findings under vLLM's rules, held to ``allowlists/ruff_vllm.json``: ``ruff check`` under ``ruff_vllm.toml`` (vLLM's
selection less what ``adapters/vllm/ruff.toml`` already selects, and less line length; ``print`` flagged in the library),
run from ``adapters/vllm`` on the whole integration with the ruff the root ``dev`` group pins.

Findings are keyed as ``test_ruff.py`` keys them.  ``ruff check --config tests/lint/ruff_vllm.toml --fix --extend-ignore I001
PATH`` fixes some; the import sort (I001) touches most files, so it runs once, in a change of its own.  A library ``print`` is
a diagnostic for the logger, or what a command prints, which ``pipeline/cli.py`` writes.  Delete the entry each fix makes
stale.

The allowlist is a ratchet the ratchet gate holds to no growth, and the gate judges a change to this module, to
``ruff_vllm.toml`` or to the ruff pin by rerunning ``test_no_new_findings`` as the change's base has it on the change's tree.
In the suite, a finding the list doesn't hold and an entry that matches nothing warn (``Bookkeeping``) and a later PR fixes
them, as ruff's style findings do across the repository: these rules are style, and a PR that lands with one is never failed
or reverted for it.  Once both ruff lists are small, one config with both selections replaces the two.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
import warnings
from functools import cache
from pathlib import Path

import pytest

from . import _ratchet as R
from . import test_ruff

NAME = "ruff_vllm"
RULE = "ruff with vLLM's rules (adapters/vllm/tests/lint/ruff_vllm.toml)"
CONFIG = R.HERE / "ruff_vllm.toml"
#: a ruff selector: its linter's letters, then the leading digits of its codes
SELECTOR = re.compile(r"([A-Z]+)(\d*)")


class Bookkeeping(UserWarning):
    """What the list doesn't match: a later PR fixes it."""


@cache
def scan() -> tuple[R.Violation, ...]:
    return test_ruff.findings(CONFIG, ".")


def test_no_new_findings() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Fix the code (`ruff check --config tests/lint/ruff_vllm.toml --fix PATH` from adapters/vllm fixes "
                    "some; a library `print` becomes a logger call, or goes through pipeline/cli.py when "
                    "it is what a command prints).")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


test_no_new_findings.__test__ = test_allowlist_has_no_stale_entries.__test__ = False


@pytest.mark.parametrize("check", [test_no_new_findings, test_allowlist_has_no_stale_entries], ids=["new", "stale"])
def test_the_list_matches_or_warns(check) -> None:
    try:
        check()
    except AssertionError as e:
        warnings.warn(Bookkeeping(str(e)), stacklevel=1)


def _selected(config: dict) -> set[str]:
    return set(config.get("lint", {}).get("select", []))


def overlap(a: str, b: str) -> bool:
    """Whether ruff selectors ``a`` and ``b`` share a rule: one linter (its letters), and one's digits begin the other's
    (``B`` holds ``B006`` but not ``BLE001``)."""
    (la, da), (lb, db) = (SELECTOR.fullmatch(s).groups() for s in (a, b))
    return la == lb and (da.startswith(db) or db.startswith(da))


def test_no_rule_is_counted_by_both_ruff_lints() -> None:
    """Every rule ``ruff.toml`` (and the root config it extends) selects is ignored here, or outside this selection, so a
    finding sits in one list."""
    ours = tomllib.loads(test_ruff.CONFIG.read_text(encoding="utf-8"))
    root = tomllib.loads((test_ruff.CONFIG.parent / ours["extend"]).read_text(encoding="utf-8"))
    vllm = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    assert "extend" not in vllm
    ignored = set(vllm["lint"]["ignore"])
    both = sorted(c for c in _selected(root) | _selected(ours) if c not in ignored
                  and any(overlap(c, p) for p in _selected(vllm)))
    assert not both, f"{CONFIG.name} selects {both}, which ruff.toml selects too: add them to its `ignore`"
    assert overlap("B", "B006") and overlap("E402", "E") and not overlap("B", "BLE001") and not overlap("E4", "E501")


def test_print_is_flagged_in_the_library_alone(tmp_path) -> None:
    for path in ("verity_vllm/a.py", "verity_vllm/pipeline/cli.py", "tests/test_a.py", "tools/a.py"):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("print(1)\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "ruff", "check", "--config", str(CONFIG), "--no-cache", "--exit-zero",
                        "--output-format", "json", "--select", "T201", "."], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert [Path(f["filename"]).relative_to(tmp_path).as_posix() for f in json.loads(r.stdout)] == ["verity_vllm/a.py"]
