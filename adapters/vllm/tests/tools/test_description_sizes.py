"""Each template GLM-5.3's served partitions name (`description_sizes.served`) has a description, as `program.json` holds
it, of at most `description_sizes.SIZE` bytes, or an entry in ``allowlists/description_sizes.json`` saying why it is past and
what removes it, a list that only shrinks (`tools/ratchet_gate.py` holds it).  A norm row counts as a partition names it:
by its parts where the serving cut names them, whole only where a partition names the whole row.

The first part reads only: the record (`tools/description_sizes.json`, which the tool's ``update`` writes) has an entry
for exactly the templates the partitions name; every one recorded past SIZE is listed; and every listed entry still
matches a named template, and only ones recorded past SIZE.  The second rebuilds each template recorded at most SIZE and
holds its bytes to the record, so a change that grows one past SIZE, or moves its bytes, fails here.  A template recorded
past SIZE is never built in a suite (a traced one takes far more time and memory than a test may): its entry stands on the
run it cites until a change rewrites the template and its record.  The last part holds ``update``'s rule for a template
it measures (`description_sizes.entry`), on entries written here, building nothing: an entry whose description has not
moved stands as it is too."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from functools import cache
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "description_sizes.py"
LIST = Path(__file__).resolve().parent / "allowlists" / "description_sizes.json"
_spec = importlib.util.spec_from_file_location("description_sizes", TOOL)
DS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DS)

UPDATE = ("`uv run python adapters/vllm/tools/description_sizes.py update -j 8 --run RUN` in a node job with --mem-gb 16 "
          "(`research run --on auto --cwd clone`, the writing-runs skill) that copies the record it writes to $OUT, then "
          "committing that record")
FIX = ("write it as Calls to its parts: one Call to what the items share, then a batch of the per-item Definition, as "
       "DsaMqaLogits_v4 and SparseMlaHeadBlock_v3 are (`EW.shared_specialized`, `EW.shared_operand`)")
#: a ``*`` in a listed template stands for the whole value of one static, a position
STAR = re.compile(r"=\*(?=[,}])")


def _entries() -> list[dict]:
    return json.loads(LIST.read_text())["entries"]


def _pattern(template: str) -> re.Pattern:
    return re.compile("".join("[^,{}]+" if p == "*" else re.escape(p) for p in re.split(r"(\*)", template)))


@cache
def _served() -> dict[str, list[str]]:
    return DS.served()


def _past(t: str, e: dict) -> str:
    fid, n = e["largest"]
    return (f"{t}: its description is {e['bytes']:,} bytes, past {DS.SIZE:,} (description_sizes.SIZE); its largest "
            f"Definition is {fid} at {n:,} bytes.  Fix: {FIX}")


def test_the_record_has_exactly_the_served_templates() -> None:
    named, rec = _served(), DS.load()
    missing = sorted(t for t in named if t not in rec)
    gone = sorted(t for t in rec if t not in named)
    assert not missing, (f"{len(missing)} served templates have no entry in {DS.RECORD.name}; measure them with {UPDATE}:\n"
                         + "\n".join(f"{t}  (named by {', '.join(named[t])})" for t in missing))
    assert not gone, (f"{len(gone)} entries of {DS.RECORD.name} name a template no served partition names; {UPDATE} drops "
                      "them:\n" + "\n".join(gone))


def test_the_list_is_well_formed() -> None:
    bad = [e for e in _entries() if set(e) != {"template", "why", "removed_by"} or not all(e.values())
           or "*" in STAR.sub("=", e["template"])]
    assert not bad, ("each entry is {template, why, removed_by}, all set, and a `*` in its template stands for one "
                     "static's whole value:\n" + "\n".join(map(json.dumps, bad)))


def test_a_served_template_past_the_size_is_listed() -> None:
    patterns = [_pattern(e["template"]) for e in _entries()]
    rec = DS.load()
    unlisted = [t for t in sorted(_served()) if t in rec and rec[t]["bytes"] > DS.SIZE
                and not any(p.fullmatch(t) for p in patterns)]
    assert not unlisted, "\n\n".join(_past(t, rec[t]) for t in unlisted)


def test_each_listed_entry_is_still_served_and_past_the_size() -> None:
    rec, named = DS.load(), _served()
    stale = []
    for e in _entries():
        hits = [t for t in named if _pattern(e["template"]).fullmatch(t)]
        under = [t for t in hits if t in rec and rec[t]["bytes"] <= DS.SIZE]
        if not hits:
            stale.append(f"{e['template']}: no served partition names it any more; remove its entry")
        elif under:
            stale.append(f"{e['template']}: {', '.join(under)} now at most {DS.SIZE:,} bytes; remove the entry, or narrow "
                         "it to what is still past")
    assert not stale, f"{LIST.name}: entries the list no longer needs:\n" + "\n".join(stale)


def _under() -> list:
    rec = DS.load()
    out = []
    for t, e in sorted(rec.items()):
        if e["bytes"] <= DS.SIZE:
            name = f"{t.split('{')[0]}-{hashlib.sha256(t.encode()).hexdigest()[:10]}"
            out.append(pytest.param(t, e["bytes"], id=name, marks=[pytest.mark.slow] if e["seconds"] > 4 else []))
    return out


@pytest.mark.parametrize(("template", "recorded"), _under())
def test_a_template_recorded_under_the_size_rebuilds_to_its_bytes(template: str, recorded: int) -> None:
    if template not in _served():
        pytest.skip("no served partition names it: the record test says to drop its entry")
    desc = DS.description(DS.definition(template))
    got = DS.size(desc)
    if got > DS.SIZE:
        pytest.fail(_past(template, {"bytes": got, "largest": DS.largest(desc)}))
    assert got == recorded, (f"{template}: its description is {got:,} bytes, its record {recorded:,}; when the change "
                             f"means to move it, record it with {UPDATE}")


#: an entry of the record, and what a later ``update`` measures of the same description: its seconds and gb differ, as
#: two runs' do, and it cites no run until ``update`` writes it
_OLD = {"bytes": 1000, "gb": 0.1, "largest": ["Dot_v1", 600], "run": "an-earlier-run", "seconds": 1.5}
_SAME = {"bytes": 1000, "gb": 0.7, "largest": ["Dot_v1", 600], "seconds": 9.9}
_BYTES = {**_SAME, "bytes": 1001}
_LARGEST = {**_SAME, "largest": ["Dot_v2", 600]}


@pytest.mark.parametrize(("old", "new", "remeasure", "want"), [
    pytest.param(_OLD, _SAME, False, _OLD, id="same-bytes-and-largest"),
    pytest.param(_OLD, _BYTES, False, {**_BYTES, "run": "this-run"}, id="bytes-moved"),
    pytest.param(_OLD, _LARGEST, False, {**_LARGEST, "run": "this-run"}, id="largest-moved"),
    pytest.param(_OLD, _SAME, True, {**_SAME, "run": "this-run"}, id="remeasure-names-it"),
    pytest.param(None, _SAME, False, {**_SAME, "run": "this-run"}, id="no-entry"),
])
def test_update_keeps_a_measured_entry_whose_description_has_not_moved(old: dict | None, new: dict, remeasure: bool,
                                                                       want: dict) -> None:
    """An entry whose template measures to the same bytes and largest Definition stays whole, with the run, seconds and
    gb it has, so ``update`` writes its line back as it was; a moved description, a template ``--remeasure`` names and
    one the record has no entry for get the measurement, citing the run."""
    assert DS.entry(copy.deepcopy(old), copy.deepcopy(new), "this-run", remeasure) == want
