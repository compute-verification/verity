"""Each benchmark or loop declares its numbers in a `metrics.toml` beside the code that records them
(`benchmarks/README.md`'s "Metrics"): every metric once across all those files, with an id no other row has, under its
owner's prefix, an owner the handle registry knows and whose `owns` there covers the declaring file, the dashboard's
words (a headline `title`, a short `unit` and a one-sentence `axis`), a direction, an `every` in minutes, hours or days,
and a command naming scripts that exist, with the declaring file's directory recording the metric under its id. Each
row's `version` says what changed above 1, its unit is pinned to its version (`PINS`), the kit's `VERSIONS` is the rows
above 1, a `runs` is `control`, and no row carries a version key or `runs` until the site's reader takes them. A whole-tree invariant, since it reads the
handle registry and the benchmarks and loops the commands run."""
from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest
from research.metrics import REGISTRY_FILE, REGISTRY_ROOTS, registry_files
from test_repository import ROOT

NAME = REGISTRY_FILE
#: the one file every row once sat in, which no row may return to
RETIRED = "benchmarks/metrics.toml"
HANDLES = ".agents/skills/using-slack/registry.json"
REQUIRED = frozenset({"id", "title", "owner", "unit", "axis", "better", "command", "every"})
OPTIONAL = frozenset({"floor", "goal", "tags", "version", "changed", "was", "runs"})
#: `runs`'s one value: the row runs on the control pod itself (infra/control/metrics_rerun.py's CONTROL)
RUNS = "control"
#: each row's version and unit as last pinned, so a unit changes only with its version
PINS = "tests/metric_pins.json"
PIN = "uv run python tests/test_metrics_registry.py --pin"
VERSIONS_AT = "benchmarks/harness/verity_bench/kit_report.py"
VERSION_KEYS = ("version", "changed", "was")
#: whether the site's reader (infra's) takes `VERSION_KEYS`: the site refuses a metric with a key its reader doesn't
#: take, so until it does no row may carry one; set True once it takes them
SITE_TAKES_VERSIONS = True
#: whether the site's reader takes `runs` on a row: until it does, the site refuses a row that carries it and the door doesn't
#: list it, so the rerun loop never runs it; set True once it takes it
SITE_TAKES_RUNS = True
#: a title's least and most words, and a unit's most, a rate's two sides (either side of " a ") counted apart
TITLE_WORDS, UNIT_WORDS = (3, 6), 3
_ID = re.compile(r"^[a-z][a-z0-9-]*/[a-z0-9][a-z0-9.-]*$")
_EVERY = re.compile(r"^[1-9][0-9]*[mhd]$")
_SCRIPT = re.compile(rf"(?:^|\s)((?:{'|'.join(REGISTRY_ROOTS)})/\S+\.(?:sh|py))(?:\s|$)")
_UNDER = ", ".join(f"{t}/" for t in REGISTRY_ROOTS[:-1]) + f" or {REGISTRY_ROOTS[-1]}/"


def _files() -> list[str]:
    return registry_files(ROOT)


def _metrics(files: list[str]) -> tuple[list[tuple[str, dict]], list[str]]:
    rows, broken = [], []
    for f in files:
        try:
            rows += [(f, m) for m in tomllib.loads((ROOT / f).read_text(encoding="utf-8")).get("metric", [])]
        except tomllib.TOMLDecodeError as e:
            broken.append(f"  {f}: not TOML ({e}); fix its syntax")
    return rows, broken


def _sentence(s: object) -> bool:
    return isinstance(s, str) and bool(s.strip()) and s.endswith(".") and "\n" not in s and not re.search(r"[.!?]\s", s)


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
    if "axis" in m and not _sentence(axis):
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
    if "runs" in m and m["runs"] != RUNS:
        out.append(f"runs is {m['runs']!r}, not {RUNS!r}: give runs = {RUNS!r} for a row only the control pod can run, and "
                   "none for one a node runs")
    out += _versioned(m)
    scripts, home = _SCRIPT.findall(str(m.get("command", ""))), ROOT / where
    missing = [f"its command's script {s} doesn't exist" for s in scripts if not (ROOT / s).is_file()]
    if not scripts:
        out.append(f"its command names no script under {_UNDER}")
    elif missing:
        out += missing
    elif not any(i in p.read_text(encoding="utf-8", errors="replace") for p in home.rglob("*")
                 if p.is_file() and p.suffix in (".py", ".sh") and "tests" not in p.relative_to(home).parts):
        out.append(f"nothing under {where}/ records a measurement named {i!r}")
    return out


def _versioned(m: dict) -> list[str]:
    out, v = [], m.get("version", 1)
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        out.append(f"version is {v!r}, not a whole number 1 or above: give the version of what the number means")
    elif v > 1 and "changed" not in m:
        out.append(f"version is {v} and the row has no changed: say in one sentence what changed in what the number "
                   f"means from version {v - 1}")
    elif v == 1 and "changed" in m:
        out.append("changed is given at version 1: drop it, or raise version where what the number means changed")
    if "changed" in m and not _sentence(m["changed"]):
        out.append(f"changed {m['changed']!r} is not one sentence: write one line saying what changed, ending with a "
                   "period")
    was = m.get("was", [])
    if not (isinstance(was, list) and all(isinstance(w, str) and _ID.match(w) for w in was)):
        out.append(f"was is {was!r}, not a list of former ids, each OWNER/NAME in lower case")
    elif len(set(was)) < len(was) or m.get("id") in was:
        out.append(f"was {was!r} repeats an id or names the row's own: list each former id once")
    return out


def _formers(rows: list[tuple[str, dict]]) -> list[str]:
    ids = {m.get("id") for _, m in rows}
    seen: dict[str, str] = {}
    out = []
    for f, m in rows:
        for w in m.get("was", []) if isinstance(m.get("was"), list) else []:
            if w in ids:
                out.append(f"  {f}: {m.get('id')}: was names {w}, which is another row's id: a former id names no row")
            elif w in seen and seen[w] != m.get("id"):
                out.append(f"  {f}: {m.get('id')}: was names {w}, which {seen[w]} names too: one row takes a former id")
            seen.setdefault(w, str(m.get("id")))
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
    assert files, f"no {NAME} under {', '.join(REGISTRY_ROOTS)}: the Metrics tab would show nothing"
    rows, broken = _metrics(files)
    where = {f: f.rsplit("/", 1)[0] for f in files}
    declared: dict[str, list[str]] = {}
    for f, m in rows:
        declared.setdefault(str(m.get("id", "?")), []).append(f)
    repeated = [f"  {i}: declared in {', '.join(fs)}; keep one row\n" for i, fs in sorted(declared.items()) if len(fs) > 1]
    off = [f"  {f}: {m.get('id', '?')}: {p}" for f, m in rows for p in _problems(m, handles, where[f]) + _unowned(f, m, reg)]
    off += _formers(rows)
    assert not (broken or repeated or off), (
        "rows of the metrics.toml files break benchmarks/README.md's \"Metrics\" (fix the row):\n"
        + "\n".join(broken) + ("\n" if broken else "") + "".join(repeated) + "\n".join(off))


def test_no_row_sits_in_the_retired_shared_file() -> None:
    assert not (ROOT / RETIRED).exists(), (
        f"{RETIRED} is retired: move each of its rows, unchanged, into the {NAME} beside the code that records it "
        f"(benchmarks/README.md's \"Metrics\"), then delete {RETIRED}")


def test_a_row_that_breaks_a_rule_is_refused() -> None:
    good = {"id": "vllm/x", "title": "Serving cost over stock", "owner": "vllm", "unit": "s",
            "axis": "Seconds to serve one request.", "better": "lower",
            "command": "bash benchmarks/serve_tax/run.sh", "every": "7d"}
    here = "benchmarks/serve_tax"
    assert _problems(good, {"vllm"}, here) == ["nothing under benchmarks/serve_tax/ records a measurement named 'vllm/x'"]
    assert _problems({**good, "id": "vllm/serving-cost-glm53"}, {"vllm"}, here) == []
    bad = {**good, "id": "infra/serving-cost-glm53", "better": "smaller", "every": "weekly", "floor": "3", "colour": 1,
           "runs": "node"}
    assert _problems(bad, {"vllm", "infra"}, here) == [
        "has unknown key colour", "id 'infra/serving-cost-glm53' is not under its owner's prefix vllm/",
        "better is 'smaller', not lower or higher",
        "every is 'weekly', not a count of minutes, hours or days such as 5m, 12h or 7d",
        "floor is '3', not a number",
        "runs is 'node', not 'control': give runs = 'control' for a row only the control pod can run, and none for one a "
        "node runs",
        "nothing under benchmarks/serve_tax/ records a measurement named 'infra/serving-cost-glm53'"]
    assert _problems({**good, "id": "vllm/serving-cost-glm53", "runs": "control"}, {"vllm"}, here) == []
    assert _problems({**good, "owner": "nobody", "command": "echo"}, {"vllm"}, here) == [
        f"owner 'nobody' is no handle in {HANDLES}",
        "its command names no script under benchmarks/, calibration/, applications/, infra/ or tools/"]
    assert _problems({**good, "id": "vllm/serving-cost-glm53"}, {"vllm"}, "benchmarks/landing") == [
        "nothing under benchmarks/landing/ records a measurement named 'vllm/serving-cost-glm53'"]


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


def test_a_row_sits_beside_the_code_that_records_it() -> None:
    row = {"id": "vllm/application-minutes", "title": "Application run time", "owner": "vllm", "unit": "minutes",
           "axis": "Minutes from the start of one application run to its verdicts.", "better": "lower",
           "command": "research run --cwd clone -- uv run python benchmarks/application_runs/run.py CONFIG", "every": "1d"}
    assert _problems(row, {"vllm"}, "benchmarks/application_runs") == []
    assert _problems(row, {"vllm"}, "benchmarks/serve_tax") == [
        "nothing under benchmarks/serve_tax/ records a measurement named 'vllm/application-minutes'"]


def test_a_row_a_loop_sends_sits_beside_the_loop_and_may_be_due_in_minutes() -> None:
    row = {"id": "infra/no-such-loop-number", "title": "No such loop number", "owner": "infra", "unit": "s",
           "axis": "Seconds a loop takes.", "better": "lower",
           "command": "sh infra/control/store_index.sh", "every": "5m"}
    assert _problems(row, {"infra"}, "infra/control") == [
        "nothing under infra/control/ records a measurement named 'infra/no-such-loop-number'"]
    assert _problems(row, {"infra"}, "benchmarks/landing") == [
        "nothing under benchmarks/landing/ records a measurement named 'infra/no-such-loop-number'"]
    assert _problems({**row, "every": "30s"}, {"infra"}, "infra/control")[0] == (
        "every is '30s', not a count of minutes, hours or days such as 5m, 12h or 7d")


def test_a_row_under_demos_is_found_and_held_to_the_same_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "applications" / "remote").mkdir(parents=True)
    (tmp_path / "applications" / "remote" / NAME).write_text("", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert _files() == ["applications/remote/metrics.toml"]
    row = {"id": "vllm/remote-demo-minutes", "title": "Remote demo run time", "owner": "vllm", "unit": "minutes",
           "axis": "Minutes from the first command to the verdict.", "better": "lower",
           "command": "bash applications/remote/no-such-run.sh", "every": "7d"}
    assert _problems(row, {"vllm"}, "applications/remote") == ["its command's script applications/remote/no-such-run.sh doesn't exist"]
    assert _problems({**row, "title": "Remote demo", "every": "weekly"}, {"vllm"}, "benchmarks/serve_tax") == [
        "title 'Remote demo' is 2 words, not 3 to 6: make it a plain headline and say the rest in axis",
        "every is 'weekly', not a count of minutes, hours or days such as 5m, 12h or 7d",
        "its command's script applications/remote/no-such-run.sh doesn't exist"]


def test_a_row_in_a_file_another_handle_owns_is_refused() -> None:
    f = "benchmarks/serve_tax/metrics.toml"
    reg = {"handles": [{"handle": "proofs", "owns": ["benchmarks/serve_tax/"]},
                       {"handle": "vllm", "owns": ["benchmarks/demo_minutes/"]}]}
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


#: our machines are all named vy-*; the register `research run --on` resolves them by is not in the tree, and
#: verity_bench.jobs.DIRECT_NODES lists only the GPU nodes
_MACHINE = re.compile(r"(?<![\w/.-])vy-[a-z0-9]+(?:-[a-z0-9]+)*")
_LEASE = re.compile(r"(?<![\w-])gpu-lease\b(.*?)(?=\s--\s|$)", re.S)
_CARD = re.compile(r"--on [A-Z][A-Z0-9_]*\b")
PINNED_FIX = ("take the node, the card and the cores as parameters (`research run --on NODE`, the runner's `--on CARD "
              "--cores CORES`) and take the lease through the runner, so the command is typed for any node and card")
#: {metrics.toml: its commands that name a machine or take a bare gpu-lease}; an entry only falls
ALLOWED = {
    "benchmarks/attester_load/metrics.toml": 3,  # network-accounting
    "benchmarks/dsa_topk/metrics.toml": 1,  # proofs
    "benchmarks/flock_s8_block/metrics.toml": 1,  # proofs
    "infra/dashboard/metrics.toml": 1,  # console
}


def _pinned(command: str) -> list[str]:
    out = [f"names the machine {m}" for m in dict.fromkeys(_MACHINE.findall(command))]
    return out + ["takes gpu-lease with no card parameter" for a in _LEASE.findall(command) if not _CARD.search(a)]


def _pinned_rows() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for f, m in _metrics(_files())[0]:
        if why := _pinned(str(m.get("command", ""))):
            found.setdefault(f, []).append(f"{m.get('id')}: {', '.join(why)}")
    return found


def test_no_command_names_a_machine_or_a_bare_lease() -> None:
    assert _pinned("research run --on vy-spot-glm53 -- env GPU_LEASE_WHO=proofs gpu-lease 1 --wait --cores 16-31 -- "
                   "bash x.sh") == ["names the machine vy-spot-glm53", "takes gpu-lease with no card parameter"]
    assert _pinned("research run --on NODE -- env GPU_LEASE_WHO=kernels gpu-lease 1 --wait --on CARD --cores CORES "
                   "-- bash x.sh") == []
    assert _pinned("research run --on NODE -- bash benchmarks/kernels/wide_leaf/run.sh --on CARD --cores CORES") == []
    over = [f"  {f} ({len(rows)}, {ALLOWED.get(f, 0)} allowed):\n" + "\n".join(f"    {r}" for r in rows)
            for f, rows in sorted(_pinned_rows().items()) if len(rows) > ALLOWED.get(f, 0)]
    assert not over, f"registered commands that name a machine of ours or a bare gpu-lease: {PINNED_FIX}:\n" + "\n".join(over)


def test_the_pinned_command_allowance_only_falls() -> None:
    found = {f: len(rows) for f, rows in _pinned_rows().items()}
    stale = [f"  {f}: {n} allowed, {found.get(f, 0)} found" for f, n in ALLOWED.items() if found.get(f, 0) < n]
    assert not stale, "lower these ALLOWED entries in tests/test_metrics_registry.py, deleting one at 0:\n" + "\n".join(stale)


def _tree_pins(rows: list[tuple[str, dict]]) -> dict[str, dict]:
    return {str(m.get("id")): {"version": m.get("version", 1), "unit": m.get("unit")} for _, m in rows}


def _read_pins() -> dict[str, dict]:
    return json.loads((ROOT / PINS).read_text(encoding="utf-8"))


def _pin_of(i: str, rows: list[tuple[str, dict]], pins: dict[str, dict]) -> dict | None:
    m = next(m for _, m in rows if m.get("id") == i)
    return pins.get(i) or next((pins[w] for w in m.get("was", []) if w in pins), None)


def _unit_changes(rows: list[tuple[str, dict]], pins: dict[str, dict]) -> list[str]:
    out = []
    for i, now in _tree_pins(rows).items():
        was = _pin_of(i, rows, pins)
        if was and now["version"] == was["version"] and now["unit"] != was["unit"]:
            out.append(f"  {i}: its unit is {now['unit']!r}, pinned {was['unit']!r} at version {now['version']}: a new "
                       f"unit is a new meaning: raise version to {now['version'] + 1}, add changed saying in one "
                       f"sentence what changed, then run {PIN}")
        elif was and now["version"] < was["version"]:
            out.append(f"  {i}: its version is {now['version']}, pinned {was['version']}: a version never falls")
    return out


def _unpinned(rows: list[tuple[str, dict]], pins: dict[str, dict]) -> list[str]:
    tree = _tree_pins(rows)
    out = [f"  {i}: no pin" for i in tree if _pin_of(i, rows, pins) is None]
    out += [f"  {i}: pinned {pins[i]}, the row {now}" for i, now in tree.items() if i in pins and pins[i] != now]
    return out + [f"  {i}: pinned under a former id" for i in tree if i not in pins and _pin_of(i, rows, pins)]


def _outlived(rows: list[tuple[str, dict]], pins: dict[str, dict]) -> list[str]:
    known = set(_tree_pins(rows)) | {w for _, m in rows for w in m.get("was", [])}
    return [f"  {i}: pinned, and no row has it" for i in sorted(set(pins) - known)]


def test_a_unit_changes_only_with_its_version() -> None:
    rows = _metrics(_files())[0]
    changed = _unit_changes(rows, _read_pins())
    assert not changed, f"rows whose unit changed at the version {PINS} pins:\n" + "\n".join(changed)


def test_every_row_is_pinned_at_its_version_and_unit() -> None:
    rows = _metrics(_files())[0]
    stale = _unpinned(rows, _read_pins())
    assert not stale, f"{PINS} is not the rows' versions and units: run {PIN}\n" + "\n".join(stale)


def test_no_pin_outlives_its_row() -> None:
    gone = _outlived(_metrics(_files())[0], _read_pins())
    assert not gone, f"{PINS} pins ids no row has: run {PIN}\n" + "\n".join(gone)


def test_a_unit_pin_holds_a_row_to_its_version() -> None:
    row = {"id": "vllm/x", "unit": "s"}
    pins = {"vllm/x": {"version": 1, "unit": "s"}}
    assert _unit_changes([("f", row)], pins) == _unpinned([("f", row)], pins) == []
    assert _unit_changes([("f", {**row, "unit": "ms"})], pins) == [
        f"  vllm/x: its unit is 'ms', pinned 's' at version 1: a new unit is a new meaning: raise version to 2, add "
        f"changed saying in one sentence what changed, then run {PIN}"]
    raised = {**row, "unit": "ms", "version": 2, "changed": "Milliseconds, not seconds."}
    assert _unit_changes([("f", raised)], pins) == []
    assert _unpinned([("f", raised)], pins) == ["  vllm/x: pinned {'version': 1, 'unit': 's'}, the row {'version': 2, 'unit': 'ms'}"]
    assert _unpinned([("f", raised)], _tree_pins([("f", raised)])) == []
    assert _unit_changes([("f", row)], {"vllm/x": {"version": 2, "unit": "s"}}) == [
        "  vllm/x: its version is 1, pinned 2: a version never falls"]
    two = [("f", row), ("g", {"id": "vllm/y", "unit": "s"})]
    assert _unpinned(two, {**pins, "vllm/gone": pins["vllm/x"]}) == ["  vllm/y: no pin"]
    assert _outlived(two, {**pins, "vllm/gone": pins["vllm/x"]}) == ["  vllm/gone: pinned, and no row has it"]
    renamed = {"id": "vllm/z", "unit": "ms", "was": ["vllm/x"]}
    assert _unit_changes([("f", renamed)], pins)[0].startswith("  vllm/z: its unit is 'ms', pinned 's' at version 1")
    assert _unpinned([("f", {**renamed, "unit": "s"})], pins) == ["  vllm/z: pinned under a former id"]
    assert _outlived([("f", renamed)], pins) == []


def test_a_version_is_whole_and_above_one_says_what_changed() -> None:
    assert _versioned({"id": "vllm/x"}) == _versioned({"id": "vllm/x", "version": 1}) == []
    assert _versioned({"id": "vllm/x", "version": 2, "changed": "The denominator is the locked clock's rate."}) == []
    assert _versioned({"id": "vllm/x", "version": 2}) == [
        "version is 2 and the row has no changed: say in one sentence what changed in what the number means from "
        "version 1"]
    for v in (0, "2", True, 1.5):
        assert _versioned({"id": "vllm/x", "version": v}) == [
            f"version is {v!r}, not a whole number 1 or above: give the version of what the number means"], v
    assert _versioned({"id": "vllm/x", "changed": "Nothing."}) == [
        "changed is given at version 1: drop it, or raise version where what the number means changed"]
    assert _versioned({"id": "vllm/x", "version": 3, "changed": "A new clock. A new run."}) == [
        "changed 'A new clock. A new run.' is not one sentence: write one line saying what changed, ending with a period"]


def test_a_former_id_names_no_row_and_is_listed_once() -> None:
    row = {"id": "vllm/new", "was": ["vllm/old"]}
    assert _versioned(row) == _formers([("f", row), ("g", {"id": "vllm/other"})]) == []
    for was in (["vllm/old", "vllm/old"], ["vllm/new"]):
        assert _versioned({**row, "was": was}) == [
            f"was {was!r} repeats an id or names the row's own: list each former id once"], was
    assert _versioned({**row, "was": "vllm/old"}) == [
        "was is 'vllm/old', not a list of former ids, each OWNER/NAME in lower case"]
    assert _formers([("f", row), ("g", {"id": "vllm/old"})]) == [
        "  f: vllm/new: was names vllm/old, which is another row's id: a former id names no row"]
    assert _formers([("f", row), ("g", {"id": "vllm/two", "was": ["vllm/old"]})]) == [
        "  g: vllm/two: was names vllm/old, which vllm/new names too: one row takes a former id"]


def _kit_versions() -> dict[str, int]:
    tree = ast.parse((ROOT / VERSIONS_AT).read_text(encoding="utf-8"))
    return next(ast.literal_eval(s.value) for s in tree.body if isinstance(s, ast.AnnAssign) and s.value is not None
                and isinstance(s.target, ast.Name) and s.target.id == "VERSIONS")


def _versions_off(rows: list[tuple[str, dict]], versions: dict[str, int]) -> list[str]:
    tree = {str(m.get("id")): m.get("version", 1) for _, m in rows}
    out = [f"  add the line {i!r}: {v}," for i, v in tree.items() if v != 1 and versions.get(i) != v]
    return out + [f"  delete the line {i!r}: {v}, (its row's version is {tree.get(i, 'gone')})"
                  for i, v in sorted(versions.items()) if tree.get(i) != v or v == 1]


def test_the_kits_versions_are_the_rows_above_version_one() -> None:
    off = _versions_off(_metrics(_files())[0], _kit_versions())
    assert not off, f"{VERSIONS_AT}'s VERSIONS, which the kit stamps each measurement's version from, holds exactly the " \
                    "rows whose version is above 1:\n" + "\n".join(off)


def test_versions_hold_a_raised_row_and_nothing_else() -> None:
    row, raised = {"id": "vllm/x"}, {"id": "vllm/y", "version": 2}
    assert _versions_off([("f", row), ("g", raised)], {"vllm/y": 2}) == []
    assert _versions_off([("f", row), ("g", raised)], {}) == ["  add the line 'vllm/y': 2,"]
    assert _versions_off([("f", row), ("g", raised)], {"vllm/y": 2, "vllm/x": 1, "vllm/gone": 3}) == [
        "  delete the line 'vllm/gone': 3, (its row's version is gone)",
        "  delete the line 'vllm/x': 1, (its row's version is 1)"]
    assert _versions_off([("g", {**raised, "version": 3})], {"vllm/y": 2}) == [
        "  add the line 'vllm/y': 3,", "  delete the line 'vllm/y': 2, (its row's version is 3)"]


def _site_refused(rows: list[tuple[str, dict]], takes: bool = SITE_TAKES_VERSIONS,
                  keys: tuple[str, ...] = VERSION_KEYS) -> list[str]:
    return [] if takes else [f"  {f}: {m.get('id')}: {k}" for f, m in rows for k in keys if k in m]


def test_no_row_carries_a_version_key_until_the_site_takes_them() -> None:
    held = _site_refused(_metrics(_files())[0])
    assert not held, ("the site refuses a metric with a key its reader doesn't take, and its reader (infra's) doesn't "
                      "take version, changed or was yet: drop them from these rows; once it takes them, set "
                      "SITE_TAKES_VERSIONS = True in tests/test_metrics_registry.py:\n" + "\n".join(held))


def test_the_site_check_holds_each_version_key_until_flipped() -> None:
    rows = [("f", {"id": "vllm/x", "version": 2, "changed": "A new clock.", "was": ["vllm/old"]}), ("g", {"id": "vllm/y"})]
    assert _site_refused(rows, takes=False) == ["  f: vllm/x: version", "  f: vllm/x: changed", "  f: vllm/x: was"]
    assert _site_refused(rows, takes=True) == []
    rows = [("f", {"id": "ci/x", "runs": "control"}), ("g", {"id": "ci/y"})]
    assert _site_refused(rows, False, ("runs",)) == ["  f: ci/x: runs"] and _site_refused(rows, True, ("runs",)) == []


def test_no_row_carries_runs_until_the_site_takes_it() -> None:
    held = _site_refused(_metrics(_files())[0], SITE_TAKES_RUNS, ("runs",))
    assert not held, ("the site refuses a metric with a key its reader doesn't take, so the door wouldn't list these rows and "
                      "the rerun loop would never run them, and its reader (infra's) doesn't take runs yet: once it takes "
                      "runs = \"control\", set SITE_TAKES_RUNS = True in tests/test_metrics_registry.py, and until then drop "
                      "it from these rows:\n" + "\n".join(held))


def pin() -> int:
    rows, broken = _metrics(_files())
    if broken or (changed := _unit_changes(rows, _read_pins() if (ROOT / PINS).is_file() else {})):
        print(f"{PINS} not refreshed:\n" + "\n".join(broken or changed))
        return 1
    pins = _tree_pins(rows)
    (ROOT / PINS).write_text("{\n" + ",\n".join(f" {json.dumps(i)}: {json.dumps(pins[i])}" for i in sorted(pins))
                             + "\n}\n", encoding="utf-8")
    print(f"{PINS}: {len(pins)} rows pinned")
    return 0


if __name__ == "__main__":
    sys.exit(pin() if sys.argv[1:] == ["--pin"] else f"usage: {PIN}")
