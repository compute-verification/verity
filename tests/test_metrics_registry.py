"""Each benchmark declares its numbers in a `metrics.toml` beside the script its command runs (`benchmarks/README.md`'s
"Metrics"): every metric once across all those files, with an id no other row has, under its owner's prefix, an owner
the handle registry knows, a direction, and a command whose first script sits under the declaring file's directory and
whose script's directory records the metric under its id. A whole-tree invariant, since it reads the handle registry
and the benchmarks the commands run."""
from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path

from test_repository import ROOT

NAME = "metrics.toml"
#: where a benchmark's metrics.toml may sit: the trees a command's script is under (`_SCRIPT`)
TREES = ("benchmarks", "demos", "tools")
#: the one file every row once sat in, which no row may return to
RETIRED = "benchmarks/metrics.toml"
SKIP = {".git", ".lake", ".venv", "node_modules", "__pycache__", "target", "build", "dist"}
HANDLES = ".agents/skills/using-slack/registry.json"
REQUIRED = frozenset({"id", "title", "owner", "unit", "better", "command", "every"})
OPTIONAL = frozenset({"floor", "goal", "tags"})
_ID = re.compile(r"^[a-z][a-z0-9-]*/[a-z0-9][a-z0-9.-]*$")
_EVERY = re.compile(r"^[1-9][0-9]*[hd]$")
_SCRIPT = re.compile(r"(?:^|\s)((?:benchmarks|tools|demos)/\S+\.(?:sh|py))(?:\s|$)")


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


def _problems(m: dict, handles: set[str], where: str) -> list[str]:
    """What breaks a rule in row `m` of the metrics.toml in directory `where`, each with its fix."""
    keys = set(m)
    out = [f"lacks {k}" for k in sorted(REQUIRED - keys)] + [f"has unknown key {k}" for k in sorted(keys - REQUIRED - OPTIONAL)]
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
        out.append(f"every is {m.get('every')!r}, not a count of hours or days such as 12h or 7d")
    if "floor" in m and (isinstance(m["floor"], bool) or not isinstance(m["floor"], int | float)):
        out.append(f"floor is {m['floor']!r}, not a number")
    if "tags" in m and not (isinstance(m["tags"], list) and all(isinstance(t, str) for t in m["tags"])):
        out.append("tags is not a list of words")
    scripts = _SCRIPT.findall(str(m.get("command", "")))
    if not scripts:
        out.append("its command names no script under benchmarks/, tools/ or demos/")
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


def test_every_metric_is_declared_once_by_a_known_owner_beside_its_producing_script() -> None:
    handles = {h["handle"] for h in json.loads((ROOT / HANDLES).read_text(encoding="utf-8"))["handles"]}
    files = _files()
    assert files, f"no {NAME} under {', '.join(TREES)}: the Metrics tab would show nothing"
    rows, broken = _metrics(files)
    where = {f: f.rsplit("/", 1)[0] for f in files}
    declared: dict[str, list[str]] = {}
    for f, m in rows:
        declared.setdefault(str(m.get("id", "?")), []).append(f)
    repeated = [f"  {i}: declared in {', '.join(fs)}; keep one row\n" for i, fs in sorted(declared.items()) if len(fs) > 1]
    off = [f"  {f}: {m.get('id', '?')}: {p}" for f, m in rows for p in _problems(m, handles, where[f])]
    assert not (broken or repeated or off), (
        "rows of the metrics.toml files break benchmarks/README.md's \"Metrics\" (fix the row):\n"
        + "\n".join(broken) + ("\n" if broken else "") + "".join(repeated) + "\n".join(off))


def test_no_row_sits_in_the_retired_shared_file() -> None:
    assert not (ROOT / RETIRED).exists(), (
        f"{RETIRED} is retired: move each of its rows, unchanged, into the {NAME} beside the first script its command "
        f"names (benchmarks/README.md's \"Metrics\"), then delete {RETIRED}")


def test_a_row_that_breaks_a_rule_is_refused() -> None:
    good = {"id": "vllm/x", "title": "t", "owner": "vllm", "unit": "s", "better": "lower",
            "command": "bash benchmarks/serve_tax/run.sh", "every": "7d"}
    here = "benchmarks/serve_tax"
    assert _problems(good, {"vllm"}, here) == ["nothing under benchmarks/serve_tax/ records a measurement named 'vllm/x'"]
    assert _problems({**good, "id": "vllm/serving-cost-glm53"}, {"vllm"}, here) == []
    bad = {**good, "id": "infra/serving-cost-glm53", "better": "smaller", "every": "weekly", "floor": "3", "colour": 1}
    assert _problems(bad, {"vllm", "infra"}, here) == [
        "has unknown key colour", "id 'infra/serving-cost-glm53' is not under its owner's prefix vllm/",
        "better is 'smaller', not lower or higher", "every is 'weekly', not a count of hours or days such as 12h or 7d",
        "floor is '3', not a number", "nothing under benchmarks/serve_tax/ records a measurement named "
        "'infra/serving-cost-glm53'"]
    assert _problems({**good, "owner": "nobody", "command": "echo"}, {"vllm"}, here) == [
        f"owner 'nobody' is no handle in {HANDLES}",
        "its command names no script under benchmarks/, tools/ or demos/"]
    assert _problems({**good, "id": "vllm/serving-cost-glm53"}, {"vllm"}, "benchmarks/landing") == [
        "its command's script benchmarks/serve_tax/run.sh is not under benchmarks/landing/: move the row to "
        "benchmarks/serve_tax/metrics.toml, beside its script"]


def test_a_row_whose_script_is_under_demos_sits_beside_it() -> None:
    row = {"id": "vllm/remote-demo-minutes", "title": "t", "owner": "vllm", "unit": "minutes", "better": "lower",
           "command": "research run --cwd source -- python3 demos/vllm/minutes.py FIRST_RUN VERDICT_RUN", "every": "7d"}
    assert _problems(row, {"vllm"}, "demos/vllm") == []
    assert _problems(row, {"vllm"}, "benchmarks/serve_tax") == [
        "its command's script demos/vllm/minutes.py is not under benchmarks/serve_tax/: move the row to "
        "demos/vllm/metrics.toml, beside its script"]
