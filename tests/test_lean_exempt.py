"""A Lean lock's `exempt` entry that names an owner, an issue and a due date ({why, owner, issue, due}) is a file owed
back under a root: it is red until then, and waits only under an allowance of the landing policy (`tools/check/policy.toml`)
with the same owner, which expires at the end of its due day."""
from __future__ import annotations

import datetime
import json
import os
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".lake", ".git", ".venv", "node_modules", "__pycache__"}
FIXTURES = ROOT / "tests/tools/verity/lean/fixtures"
OWED = {"why", "owner", "issue", "due"}
#: the allowance of the case `PACKAGE:ENTRY` of the test below: a test of the root suite, by its node id
ALLOWANCE = "suite repository::test_lean_exempt.py::test_an_exempt_file_with_an_owner_is_owed[{case}]"


def owed() -> dict[str, dict]:
    """{`PACKAGE:ENTRY`: the entry} of every lock's `exempt` entries in the four-field form: the lock's directory, then the
    entry as the lock writes it."""
    out = {}
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        if "lean-audit.json" in files and not Path(d).is_relative_to(FIXTURES):
            exempt = json.loads((Path(d) / "lean-audit.json").read_text()).get("exempt") or {}
            out |= {f"{Path(d).relative_to(ROOT)}:{p}": e for p, e in exempt.items() if isinstance(e, dict) and set(e) == OWED}
    return out


def expiry(due: str) -> str:
    """The end of the day ``due`` (YYYY-MM-DD) at UTC-8, as the policy writes a time."""
    return f"{datetime.date.fromisoformat(due) + datetime.timedelta(days=1)}T08:00:00Z"


@pytest.mark.parametrize("case", sorted(owed()))
def test_an_exempt_file_with_an_owner_is_owed(case):
    """Fails for each such entry, by design: the file is out of the build until its issue ends. Its allowance lets a
    candidate land meanwhile, and at the allowance's expiry this is red."""
    e = owed()[case]
    pytest.fail(f"{case}: left out of the roots ({e['why']}), owed by {e['owner']} on {e['issue']}, due {e['due']}: put the "
                f"file back under a root and remove the `exempt` entry; until {expiry(e['due'])} it waits under the "
                f"allowance `{ALLOWANCE.format(case=case)}` of tools/check/policy.toml")


def test_each_owed_exempt_entry_has_its_allowance_and_each_such_allowance_its_entry():
    entries = {ALLOWANCE.format(case=c): e for c, e in owed().items()}
    allow = {a["id"]: a for a in tomllib.loads((ROOT / "tools/check/policy.toml").read_text()).get("allow", [])}
    prefix = ALLOWANCE.split("[")[0] + "["
    missing = sorted(set(entries) - set(allow))
    assert not missing, (f"`exempt` entries with an owner and no allowance: add to tools/check/policy.toml, for each, an "
                         f"allowance with this id, the entry's owner and `expires` the end of its due day: {missing}")
    stale = sorted(i for i in allow if i.startswith(prefix) and i not in entries)
    assert not stale, f"allowances of `exempt` entries no lock has: remove them from tools/check/policy.toml: {stale}"
    wrong = {i: (allow[i]["owner"], allow[i]["expires"]) for i, e in entries.items()
             if (allow[i]["owner"], allow[i]["expires"]) != (e["owner"], expiry(e["due"]))}
    assert not wrong, (f"allowances whose owner or expiry is not their `exempt` entry's (its owner; the end of its due day, "
                       f"{{due + 1 day}}T08:00:00Z): {wrong}")
