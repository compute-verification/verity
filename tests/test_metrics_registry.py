"""`benchmarks/metrics.toml` declares each metric once (`benchmarks/README.md`'s "Metrics"): an id no other row has,
under its owner's prefix, an owner the handle registry knows, a direction, and a command whose script's directory
records the metric under its id. A whole-tree invariant, since it reads the handle registry and the benchmarks the
commands run."""
from __future__ import annotations

import json
import re
import tomllib

from test_repository import ROOT

REGISTRY = "benchmarks/metrics.toml"
HANDLES = ".agents/skills/using-slack/registry.json"
REQUIRED = frozenset({"id", "title", "owner", "unit", "better", "command", "every"})
OPTIONAL = frozenset({"floor", "goal", "tags"})
_ID = re.compile(r"^[a-z][a-z0-9-]*/[a-z0-9][a-z0-9.-]*$")
_EVERY = re.compile(r"^[1-9][0-9]*[hd]$")
_SCRIPT = re.compile(r"(?:^|\s)((?:benchmarks|tools|demos)/\S+\.(?:sh|py))(?:\s|$)")


def _metrics() -> list[dict]:
    return tomllib.loads((ROOT / REGISTRY).read_text(encoding="utf-8")).get("metric", [])


def _problems(m: dict, handles: set[str]) -> list[str]:
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
    for s in scripts:
        path = ROOT / s
        if not path.is_file():
            out.append(f"its command's script {s} doesn't exist")
        elif not any(i in p.read_text(encoding="utf-8", errors="replace") for p in path.parent.rglob("*")
                     if p.is_file() and p.suffix in (".py", ".sh") and "tests" not in p.relative_to(path.parent).parts):
            out.append(f"nothing under {s.rsplit('/', 1)[0]}/ records a measurement named {i!r}")
    return out


def test_every_metric_is_declared_once_by_a_known_owner_with_a_producing_command() -> None:
    handles = {h["handle"] for h in json.loads((ROOT / HANDLES).read_text(encoding="utf-8"))["handles"]}
    metrics = _metrics()
    ids = [m.get("id") for m in metrics]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    off = [f"  {m.get('id', '?')}: {p}" for m in metrics for p in _problems(m, handles)]
    assert not (repeated or off), (
        f"{REGISTRY}'s rows break benchmarks/README.md's \"Metrics\" (fix the row):\n"
        + "".join(f"  {i}: declared more than once; keep one row\n" for i in repeated) + "\n".join(off))


def test_a_row_that_breaks_a_rule_is_refused() -> None:
    good = {"id": "vllm/x", "title": "t", "owner": "vllm", "unit": "s", "better": "lower",
            "command": "bash benchmarks/serve_tax/run.sh", "every": "7d"}
    assert _problems(good, {"vllm"}) == ["nothing under benchmarks/serve_tax/ records a measurement named 'vllm/x'"]
    assert _problems({**good, "id": "vllm/serving-cost-glm53"}, {"vllm"}) == []
    bad = {**good, "id": "infra/serving-cost-glm53", "better": "smaller", "every": "weekly", "floor": "3", "colour": 1}
    assert _problems(bad, {"vllm", "infra"}) == [
        "has unknown key colour", "id 'infra/serving-cost-glm53' is not under its owner's prefix vllm/",
        "better is 'smaller', not lower or higher", "every is 'weekly', not a count of hours or days such as 12h or 7d",
        "floor is '3', not a number", "nothing under benchmarks/serve_tax/ records a measurement named "
        "'infra/serving-cost-glm53'"]
    assert _problems({**good, "owner": "nobody", "command": "echo"}, {"vllm"}) == [
        f"owner 'nobody' is no handle in {HANDLES}",
        "its command names no script under benchmarks/, tools/ or demos/"]
