"""No new superseded Definition version: the registries' set of them only shrinks.

A Definition's id is ``Name_vN``, and a version is superseded when the registries register a higher version of its name and
kind.  A name's versions are of two kinds: its Boolean versions, each a Boolean Definition registered with the word Definition
it computes (`boolean.definition`'s ``word``), and the rest.  A Boolean version is its word on bits, a new version of the same
name by the catalog's convention and not a successor: ``Name_v2`` computing ``Name_v1`` on bits supersedes nothing, while a
second word ``Name_v3`` supersedes ``Name_v1``, and its Boolean version ``Name_v4`` supersedes ``Name_v2``.  The registries are
the ones circuit-check loads (`targets.load_registries`), read in an interpreter that ran nothing before (`registered`): the
registry is the process's and grows as code runs, so read in this process it would hold whatever a test before it made.

Every registered Definition counts, wherever its module is: a protocol's circuits and an adapter's as the catalog's.  What the
registries don't hold doesn't: a Definition made with ``register=False``, which only the code that makes it can reach, and a
module or a Program named ``_vN``, whose name is no Definition's version.

``tests/lint/allowlists/superseded_versions.json`` lists every superseded version, one entry each: ``file``, the module that
defines it (`sidecars.owner`); ``kind``, ``superseded``; ``symbol``, its name; ``detail``, its id.  A superseded version with no
entry fails, so a change that adds ``Name_vN+1`` does one of three things:

* deletes ``Name_vN`` with it;
* deletes another superseded version and gives ``Name_vN`` its entry;
* says which workload of record still names ``Name_vN``: ``"Name_vN": "<workload>"`` in the list's ``held`` table, the
  workload's id as ``adapters/vllm/manifests/records.json`` has it.  A workload's pinned Programs name Definitions by id,
  so the old version stays until the workload is pinned again.  The line is a statement nothing here checks: the index holds
  the Programs only by digest.

The list is exact: an entry or a ``held`` line for a version that is gone or no longer superseded fails too, and so does an
entry whose module no longer defines its version.  ``--prune`` drops the first and rewrites the second, and adds nothing.  The
ratchet gate (``adapters/vllm/tools/ratchet_gate.py``) reports a change that leaves ``entries`` longer than it found them;
a ``held`` line is a declaration, which it doesn't count.

    uv run python tools/verity/circuit_check/tests/test_superseded_versions.py --prune   # drop what is gone, follow a moved module
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
LIST = REPO / "tests/lint/allowlists/superseded_versions.json"
RULE = "superseded Definition versions (tools/verity/circuit_check/tests/test_superseded_versions.py)"
KIND = "superseded"
ID = re.compile(r"(.+)_v(\d+)")
FIX = ("A change that adds Name_vN+1 deletes Name_vN with it; or deletes another superseded version and gives this one its "
       "entry (the line below; `--prune` drops the deleted one's); or, where a workload of record still names the old version, "
       'says which: add "Name_vN": "<workload>" to the list\'s `held`, the workload\'s id as '
       "adapters/vllm/manifests/records.json has it. A Boolean version registered with the word Definition it computes "
       "(`word=`) supersedes nothing.")
PRUNE = f"uv run python {Path(__file__).resolve().relative_to(REPO)} --prune"

#: run in a fresh interpreter: each id the registries register with the file of the module that defines it and whether it is a
#: Boolean version, as JSON on its last line
_LOAD = """
import json, sys
from circuit_check import sidecars, targets
from verity.circuits.defs import REGISTRY
targets.load_registries()
print(json.dumps({i: [sidecars.rel(sys.modules[sidecars.owner(d)].__file__), getattr(d, "word", None) is not None]
                  for i in targets.REGISTERED for d in [REGISTRY.defs[i]]}))
"""


@cache
def registered() -> dict[str, tuple[str, bool]]:
    """{id: (the file of the module that defines it, whether it is a Boolean version)} for every Definition the registries
    register."""
    p = subprocess.run([sys.executable, "-c", _LOAD], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-4000:]
    return {i: (file, boolean) for i, (file, boolean) in json.loads(p.stdout.splitlines()[-1]).items()}


def superseded(ids: dict[str, tuple[str, bool]]) -> dict[str, tuple[str, str]]:
    """{each superseded id of ``ids``: (its file, the newest id of its name and kind)}."""
    versions = {i: ((m[1], boolean), int(m[2])) for i, (_, boolean) in ids.items() if (m := ID.fullmatch(i))}
    newest: dict[tuple[str, bool], int] = {}
    for kind, n in versions.values():
        newest[kind] = max(n, newest.get(kind, n))
    return {i: (ids[i][0], f"{kind[0]}_v{newest[kind]}") for i, (kind, n) in versions.items() if n < newest[kind]}


def entry(file: str, fid: str) -> dict[str, str]:
    return {"file": file, "kind": KIND, "symbol": ID.fullmatch(fid)[1], "detail": fid}


def parse(doc: object) -> tuple[dict[str, str], dict[str, str]]:
    """({listed id: its file}, {held id: its workload}) from the list's document."""
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list) and isinstance(doc.get("held"), dict), \
        f"{LIST.name}: {{'held': {{...}}, 'entries': [...]}} expected"
    entries: dict[str, str] = {}
    for e in doc["entries"]:
        fid = e.get("detail") if isinstance(e, dict) else None
        assert isinstance(fid, str) and ID.fullmatch(fid) and isinstance(e.get("file"), str) and e == entry(e["file"], fid), \
            f"{LIST.name}: an entry is {{file, kind: {KIND!r}, symbol: Name, detail: Name_vN}}: {e}"
        assert fid not in entries, f"{LIST.name}: duplicate entry {e}"
        entries[fid] = e["file"]
    held = doc["held"]
    assert all(isinstance(w, str) and w.strip() for w in held.values()), f"{LIST.name}: `held` is {{Name_vN: workload}}"
    return entries, held


def findings(now: dict[str, tuple[str, str]], entries: dict[str, str], held: dict[str, str]) -> dict[str, list[str]]:
    """What the list gets wrong about ``now``, the superseded versions: ``new``, the ones with neither an entry nor a ``held``
    line; ``gone``, the entries and ``held`` lines for none of them; ``moved``, the entries naming another file than the one
    that defines their version; ``twice``, the versions both listed and held."""
    return {
        "new": [f"{now[i][1]} supersedes {i}: {json.dumps(entry(now[i][0], i))}"
                for i in sorted(now) if i not in entries and i not in held],
        "gone": sorted(i for i in {*entries, *held} if i not in now),
        "moved": [f"{i}: {f} -> {now[i][0]}" for i, f in sorted(entries.items()) if i in now and now[i][0] != f],
        "twice": sorted(set(entries) & set(held)),
    }


def listed() -> tuple[dict[str, str], dict[str, str]]:
    return parse(json.loads(LIST.read_text(encoding="utf-8")))


@cache
def found() -> dict[str, list[str]]:
    return findings(superseded(registered()), *listed())


def _lines(items: list[str]) -> str:
    return "\n".join(f"  {x}" for x in items)


def test_no_new_superseded_versions() -> None:
    new = found()["new"]
    assert not new, (f"superseded Definition versions: {len(new)} version(s) are superseded and not in "
                     f"{LIST.relative_to(REPO)}, whose entries only shrink. {FIX}\n{_lines(new)}")


def test_the_list_holds_only_superseded_versions_where_they_are_defined() -> None:
    gone, moved, twice = (found()[k] for k in ("gone", "moved", "twice"))
    assert not gone and not moved and not twice, (
        f"superseded Definition versions: {LIST.relative_to(REPO)} is exact, and {len(gone)} of its versions are gone or no "
        f"longer superseded, {len(moved)} are defined in another module than their entry names, and {len(twice)} have both an "
        f"entry and a `held` line. Drop the first and rewrite the second with `{PRUNE}`, and keep one line for each of the "
        f"third:\n{_lines(gone + moved + twice)}")


def test_what_is_superseded_and_what_the_list_gets_wrong() -> None:
    rest, boolean = False, True
    ids = {"A_v1": ("a.py", rest), "A_v2": ("a.py", rest), "A_v4": ("b.py", rest), "B_v1": ("b.py", rest),
           "C[A_v1]_v2": ("c.py", rest), "D_v3": ("d.py", rest), "D_v7": ("d.py", rest), "E_v10": ("e.py", rest),
           "E_v9": ("e.py", rest), "plain": ("p.py", rest),
           "F_v1": ("f.py", rest), "F_v2": ("f.py", boolean), "G_v1": ("g.py", rest), "G_v2": ("g.py", boolean),
           "G_v3": ("g.py", rest), "G_v4": ("h.py", boolean), "H_v1": ("h.py", boolean), "H_v2": ("h.py", rest)}
    now = superseded(ids)
    assert {i: now.pop(i) for i in ("G_v1", "G_v2")} == {"G_v1": ("g.py", "G_v3"), "G_v2": ("g.py", "G_v4")}
    assert now == {"A_v1": ("a.py", "A_v4"), "A_v2": ("a.py", "A_v4"), "D_v3": ("d.py", "D_v7"), "E_v9": ("e.py", "E_v10")}
    assert entry("a.py", "C[A_v1]_v2") == {"file": "a.py", "kind": KIND, "symbol": "C[A_v1]", "detail": "C[A_v1]_v2"}
    assert not any(findings(now, {"A_v1": "a.py", "A_v2": "a.py", "D_v3": "d.py"}, {"E_v9": "a workload"}).values())
    listed_wrong = {"A_v1": "a.py", "A_v2": "z.py", "B_v1": "b.py", "D_v3": "d.py"}, {"D_v3": "a workload", "A_v4": "another"}
    assert findings(now, *listed_wrong) == {
        "new": ['E_v10 supersedes E_v9: {"file": "e.py", "kind": "superseded", "symbol": "E", "detail": "E_v9"}'],
        "gone": ["A_v4", "B_v1"], "moved": ["A_v2: z.py -> a.py"], "twice": ["D_v3"]}
    entries, held = pruned(now, *listed_wrong)
    assert (entries, held) == ({"A_v1": "a.py", "A_v2": "a.py", "D_v3": "d.py"}, {"D_v3": "a workload"})
    assert parse(json.loads(text(entries, held))) == (entries, held)
    for bad in ({"file": "a.py", "kind": KIND, "symbol": "B", "detail": "A_v1"}, {"file": "a.py", "detail": "A_v1"},
                {"file": "a.py", "kind": KIND, "symbol": "A", "detail": "A_v1", "count": 2}, "A_v1"):
        try:
            parse({"held": {}, "entries": [bad]})
        except AssertionError as e:
            assert str(e).startswith(f"{LIST.name}: "), e
        else:
            raise AssertionError(f"{bad} parsed")


def pruned(now: dict[str, tuple[str, str]], entries: dict[str, str], held: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """``entries`` and ``held`` less the versions ``now`` doesn't hold, each entry at the file that defines its version."""
    return {i: now[i][0] for i in entries if i in now}, {i: w for i, w in held.items() if i in now}


def text(entries: dict[str, str], held: dict[str, str]) -> str:
    """The list's file: one entry a line, as the lists' merge driver writes it."""
    def order(i: str) -> tuple[str, str, int]:
        m = ID.fullmatch(i)
        return entries[i], m[1], int(m[2])

    rows = [f"    {json.dumps(i)}: {json.dumps(held[i])}" for i in sorted(held)]
    table = "{\n" + ",\n".join(rows) + "\n  }" if rows else "{}"
    lines = [f"    {json.dumps(entry(entries[i], i), ensure_ascii=False)}" for i in sorted(entries, key=order)]
    return f'{{\n  "rule": {json.dumps(RULE)},\n  "held": {table},\n  "entries": [\n' + ",\n".join(lines) + "\n  ]\n}\n"


if __name__ == "__main__":
    if sys.argv[1:] == ["--prune"]:
        before = listed()
        after = pruned(superseded(registered()), *before)
        LIST.write_text(text(*after), encoding="utf-8")
        print(f"{LIST.relative_to(REPO)}: {len(before[0])} -> {len(after[0])} entries, {len(before[1])} -> {len(after[1])} held")
    else:
        sys.exit(__doc__.rsplit("\n\n", 1)[-1])
