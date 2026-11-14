"""Each benchmark or loop declares its numbers in a `metrics.toml` beside the script its command runs
(`benchmarks/README.md`'s "Metrics"): every metric once across all those files, with an id no other row has, under its
owner's prefix, an owner the handle registry knows and whose `owns` there covers the declaring file, the dashboard's
words (a headline `title`, a short `unit` and a one-sentence `axis`), a direction, an `every` in minutes, hours or days,
and a command whose first script sits under the declaring file's directory and whose
script's directory records the metric under its id. A whole-tree invariant, since it reads the handle registry and the
benchmarks and loops the commands run."""
from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path

from test_repository import ROOT

NAME = "metrics.toml"
#: where a benchmark's metrics.toml may sit: the trees a command's script is under (`_SCRIPT`)
TREES = ("benchmarks", "demos", "infra", "tools")
#: the one file every row once sat in, which no row may return to
RETIRED = "benchmarks/metrics.toml"
SKIP = {".git", ".lake", ".venv", "node_modules", "__pycache__", "target", "build", "dist"}
HANDLES = ".agents/skills/using-slack/registry.json"
REQUIRED = frozenset({"id", "title", "owner", "unit", "axis", "better", "command", "every"})
OPTIONAL = frozenset({"floor", "goal", "tags"})
#: a title's least and most words, and a unit's most, a rate's two sides (either side of " a ") counted apart
TITLE_WORDS, UNIT_WORDS = (3, 6), 3
_ID = re.compile(r"^[a-z][a-z0-9-]*/[a-z0-9][a-z0-9.-]*$")
_EVERY = re.compile(r"^[1-9][0-9]*[mhd]$")
_SCRIPT = re.compile(rf"(?:^|\s)((?:{'|'.join(TREES)})/\S+\.(?:sh|py))(?:\s|$)")
_UNDER = ", ".join(f"{t}/" for t in TREES[:-1]) + f" or {TREES[-1]}/"


def _files() -> list[str]:
    out = []
    for tree in TREES:
        for d, dirs, files in os.walk(ROOT / tree):
            dirs[:] = sorted(x for x in dirs if x not in SKIP)
            if NAME in files:
                out.append((Path(d) / NAME).relative_to(ROOT).as_posix())
    return sorted(out)


def _metrics(files: list[str]) -> tuple[list[tuple[str, dict]], list[str]]:
    rows, broken = [], []
    for f in files:
        try:
            rows += [(f, m) for m in tomllib.loads((ROOT / f).read_text(encoding="utf-8")).get("metric", [])]
        except tomllib.TOMLDecodeError as e:
            broken.append(f"  {f}: not TOML ({e}); fix its syntax")
    return rows, broken


def _words(m: dict) -> list[str]:
    """What breaks a rule of the dashboard's words in row `m`: its `title`, `unit` and `axis`, each with its fix."""
    out = []
    title, unit, axis = (m.get(k) for k in ("title", "unit", "axis"))
    lo, hi = TITLE_WORDS
    if isinstance(title, str) and not lo <= len(title.split()) <= hi:
        out.append(f"title {title!r} is {len(title.split())} words, not {lo} to {hi}: make it a plain headline and say "
                   "the rest in axis")
    if isinstance(title, str) and re.search(r"[0-9`]", title):
        out.append(f"title {title!r} holds a digit or a backtick: spell it in plain words and put figures and names "
                   "of code in axis")
    sides = unit.split(" a ", 1) if isinstance(unit, str) else []
    if "unit" in m and not (sides and all(0 < len(s.split()) <= UNIT_WORDS for s in sides)):
        out.append(f"unit {unit!r} is not {UNIT_WORDS} words at most (a rate's two sides, either side of \" a \", "
                   "counted apart): shorten it, put a span in the measurement's period or over, and say the rest in "
                   "axis")
    if "axis" in m and not (isinstance(axis, str) and axis.strip() and axis.endswith(".") and "\n" not in axis
                            and not re.search(r"[.!?]\s", axis)):
        out.append(f"axis {axis!r} is not one sentence: write one line saying what the y axis is, ending with a "
                   "period")
    return out


def _problems(m: dict, handles: set[str], where: str) -> list[str]:
    """What breaks a rule in row `m` of the metrics.toml in directory `where`, each with its fix."""
    keys = set(m)
    out = [f"lacks {k}: add it, as benchmarks/README.md's \"Metrics\" says" for k in sorted(REQUIRED - keys)]
    out += [f"has unknown key {k}" for k in sorted(keys - REQUIRED - OPTIONAL)] + _words(m)
    i, owner = str(m.get("id", "")), m.get("owner")
    if not _ID.match(i):
        out.append(f"id {i!r} is not OWNER/NAME in lower case")
    if owner not in handles:
        out.append(f"owner {owner!r} is no handle in {HANDLES}")
    elif not i.startswith(f"{owner}/"):
        out.append(f"id {i!r} is not under its owner's prefix {owner}/")
    if m.get("better") not in ("lower", "higher"):
        out.append(f"better is {m.get('better')!r}, not lower or higher")
    if not _EVERY.match(str(m.get("every", ""))):
        out.append(f"every is {m.get('every')!r}, not a count of minutes, hours or days such as 5m, 12h or 7d")
    if "floor" in m and (isinstance(m["floor"], bool) or not isinstance(m["floor"], int | float)):
        out.append(f"floor is {m['floor']!r}, not a number")
    if "tags" in m and not (isinstance(m["tags"], list) and all(isinstance(t, str) for t in m["tags"])):
        out.append("tags is not a list of words")
    scripts = _SCRIPT.findall(str(m.get("command", "")))
    if not scripts:
        out.append(f"its command names no script under {_UNDER}")
    elif not scripts[0].startswith(f"{where}/"):
        out.append(f"its command's script {scripts[0]} is not under {where}/: move the row to "
                   f"{scripts[0].rsplit('/', 1)[0]}/{NAME}, beside its script")
    for s in scripts:
        path = ROOT / s
        if not path.is_file():
            out.append(f"its command's script {s} doesn't exist")
        elif not any(i in p.read_text(encoding="utf-8", errors="replace") for p in path.parent.rglob("*")
                     if p.is_file() and p.suffix in (".py", ".sh") and "tests" not in p.relative_to(path.parent).parts):
            out.append(f"nothing under {s.rsplit('/', 1)[0]}/ records a measurement named {i!r}")
    return out


def _unowned(f: str, m: dict, reg: dict) -> list[str]:
    """Whether row `m`'s owner owns its file `f`, the one the longest `owns` entry covering it names, with the fix."""
    from research.slack import owner

    holder, want = owner(reg, f), m.get("owner")
    if holder == want or want not in {h["handle"] for h in reg["handles"]}:
        return []
    by = f"@{holder}'s" if holder else "no handle's"
    return [f"its file is {by} by {HANDLES}'s longest `owns` entry covering it: add {f!r} to @{want}'s `owns` there"]


def test_every_metric_is_declared_once_by_a_known_owner_beside_its_producing_script() -> None:
    reg = json.loads((ROOT / HANDLES).read_text(encoding="utf-8"))
    handles = {h["handle"] for h in reg["handles"]}
    files = _files()
    assert files, f"no {NAME} under {', '.join(TREES)}: the Metrics tab would show nothing"
    rows, broken = _metrics(files)
    where = {f: f.rsplit("/", 1)[0] for f in files}
    declared: dict[str, list[str]] = {}
    for f, m in rows:
        declared.setdefault(str(m.get("id", "?")), []).append(f)
    repeated = [f"  {i}: declared in {', '.join(fs)}; keep one row\n" for i, fs in sorted(declared.items()) if len(fs) > 1]
    off = [f"  {f}: {m.get('id', '?')}: {p}" for f, m in rows for p in _problems(m, handles, where[f]) + _unowned(f, m, reg)]
    assert not (broken or repeated or off), (
        "rows of the metrics.toml files break benchmarks/README.md's \"Metrics\" (fix the row):\n"
        + "\n".join(broken) + ("\n" if broken else "") + "".join(repeated) + "\n".join(off))


def test_no_row_sits_in_the_retired_shared_file() -> None:
    assert not (ROOT / RETIRED).exists(), (
        f"{RETIRED} is retired: move each of its rows, unchanged, into the {NAME} beside the first script its command "
        f"names (benchmarks/README.md's \"Metrics\"), then delete {RETIRED}")


def test_a_row_that_breaks_a_rule_is_refused() -> None:
    good = {"id": "vllm/x", "title": "Serving cost over stock", "owner": "vllm", "unit": "s",
            "axis": "Seconds to serve one request.", "better": "lower",
            "command": "bash benchmarks/serve_tax/run.sh", "every": "7d"}
    here = "benchmarks/serve_tax"
    assert _problems(good, {"vllm"}, here) == ["nothing under benchmarks/serve_tax/ records a measurement named 'vllm/x'"]
    assert _problems({**good, "id": "vllm/serving-cost-glm53"}, {"vllm"}, here) == []
    bad = {**good, "id": "infra/serving-cost-glm53", "better": "smaller", "every": "weekly", "floor": "3", "colour": 1}
    assert _problems(bad, {"vllm", "infra"}, here) == [
        "has unknown key colour", "id 'infra/serving-cost-glm53' is not under its owner's prefix vllm/",
        "better is 'smaller', not lower or higher",
        "every is 'weekly', not a count of minutes, hours or days such as 5m, 12h or 7d",
        "floor is '3', not a number", "nothing under benchmarks/serve_tax/ records a measurement named "
        "'infra/serving-cost-glm53'"]
    assert _problems({**good, "owner": "nobody", "command": "echo"}, {"vllm"}, here) == [
        f"owner 'nobody' is no handle in {HANDLES}",
        "its command names no script under benchmarks/, demos/, infra/ or tools/"]
    assert _problems({**good, "id": "vllm/serving-cost-glm53"}, {"vllm"}, "benchmarks/landing") == [
        "its command's script benchmarks/serve_tax/run.sh is not under benchmarks/landing/: move the row to "
        "benchmarks/serve_tax/metrics.toml, beside its script"]


WORDS = {"title": "Prover GPUs per serving GPU", "unit": "prover GPUs a serving GPU",
         "axis": "GPUs proving for each GPU serving GLM-5.3, under SPOT at its chosen sampling plan."}


def test_a_row_without_an_axis_is_refused() -> None:
    row = {"id": "vllm/x", "owner": "vllm", "better": "lower", "command": "bash benchmarks/serve_tax/run.sh",
           "every": "7d", **WORDS}
    assert _words(row) == []
    del row["axis"]
    assert _problems(row, {"vllm"}, "benchmarks/serve_tax")[0] == (
        "lacks axis: add it, as benchmarks/README.md's \"Metrics\" says")
    assert _words(row) == []


def test_a_title_is_three_to_six_plain_words() -> None:
    assert _words({**WORDS, "title": "Time to merge"}) == []
    assert _words({**WORDS, "title": "Prover GPUs per serving GPU, modeled"}) == []
    for title in ("Serving cost", "Exfiltration capacity over the world's year at ten percent"):
        assert _words({**WORDS, "title": title}) == [
            f"title {title!r} is {len(title.split())} words, not 3 to 6: make it a plain headline and say the rest in "
            "axis"]
    for title in ("Llama 1B compute share", "Time to `main`, Lean"):
        assert _words({**WORDS, "title": title}) == [
            f"title {title!r} holds a digit or a backtick: spell it in plain words and put figures and names of code "
            "in axis"]


def test_a_unit_is_three_words_at_most_a_rate_s_sides_apart() -> None:
    for unit in ("hours", "GB", "percent of tokens", "Lean verifier seconds a served token"):
        assert _words({**WORDS, "unit": unit}) == [], unit
    for unit in ("percent of stock's serving time", "GB a world year at 10% of compute", "", "seconds a "):
        assert _words({**WORDS, "unit": unit}) == [
            f"unit {unit!r} is not 3 words at most (a rate's two sides, either side of \" a \", counted apart): "
            "shorten it, put a span in the measurement's period or over, and say the rest in axis"], unit


def test_an_axis_is_one_sentence() -> None:
    assert _words({**WORDS, "axis": "Share of Llama 3.2 1B's multiply-adds that the verifier credits."}) == []
    for axis in ("", "Hours from the ready mark to main", "Hours to merge.\nOver a week.",
                 "Hours to merge. Over a week.", 7):
        assert _words({**WORDS, "axis": axis}) == [
            f"axis {axis!r} is not one sentence: write one line saying what the y axis is, ending with a period"], axis


def test_a_row_whose_script_is_under_demos_sits_beside_it() -> None:
    row = {"id": "vllm/remote-demo-minutes", "title": "Remote demo run time", "owner": "vllm", "unit": "minutes",
           "axis": "Minutes from the first command to the verdict.", "better": "lower",
           "command": "research run --cwd source -- python3 demos/vllm/minutes.py FIRST_RUN VERDICT_RUN", "every": "7d"}
    assert _problems(row, {"vllm"}, "demos/vllm") == []
    assert _problems(row, {"vllm"}, "benchmarks/serve_tax") == [
        "its command's script demos/vllm/minutes.py is not under benchmarks/serve_tax/: move the row to "
        "demos/vllm/metrics.toml, beside its script"]


def test_a_row_a_loop_sends_sits_beside_the_loop_and_may_be_due_in_minutes() -> None:
    row = {"id": "infra/no-such-loop-number", "title": "No such loop number", "owner": "infra", "unit": "s",
           "axis": "Seconds a loop takes.", "better": "lower",
           "command": "sh infra/control/store_index.sh", "every": "5m"}
    assert _problems(row, {"infra"}, "infra/control") == [
        "nothing under infra/control/ records a measurement named 'infra/no-such-loop-number'"]
    assert _problems(row, {"infra"}, "benchmarks/landing")[0] == (
        "its command's script infra/control/store_index.sh is not under benchmarks/landing/: move the row to "
        "infra/control/metrics.toml, beside its script")
    assert _problems({**row, "every": "30s"}, {"infra"}, "infra/control")[0] == (
        "every is '30s', not a count of minutes, hours or days such as 5m, 12h or 7d")


def test_a_row_in_a_file_another_handle_owns_is_refused() -> None:
    f = "benchmarks/serve_tax/metrics.toml"
    reg = {"handles": [{"handle": "proofs", "owns": ["benchmarks/serve_tax/"]},
                       {"handle": "vllm", "owns": ["demos/vllm/"]}]}
    fix = f"add {f!r} to @vllm's `owns` there"
    assert _unowned(f, {"owner": "proofs"}, reg) == []
    assert _unowned(f, {"owner": "vllm"}, reg) == [
        f"its file is @proofs's by {HANDLES}'s longest `owns` entry covering it: {fix}"]
    reg["handles"][1]["owns"].append(f)
    assert _unowned(f, {"owner": "vllm"}, reg) == []
    assert _unowned("benchmarks/tree/metrics.toml", {"owner": "vllm"}, reg) == [
        f"its file is no handle's by {HANDLES}'s longest `owns` entry covering it: "
        "add 'benchmarks/tree/metrics.toml' to @vllm's `owns` there"]
    assert _unowned(f, {"owner": "nobody"}, reg) == []
