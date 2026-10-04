"""The cache's per-target keys (`circuit_check.cache`, `circuit-check-cache/v2`), on a synthetic tree: a registry of small
Definitions, a catalog (`targets.py`) and a pins file of its own, checked through `cli.run`'s `--all` path with the catalog's
functions standing in for the real ones (the real catalog takes a minute to build).  A target's entry holds what its own check
read or resolved, and nothing another target or the task list read."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

FILES = {
    "syn/__init__.py": "",
    "syn/prims.py": """
from verity.ir import Value, primitive

V = Value(16)


@primitive("SynAdd", 1, [("a", V), ("b", V)], V)
def SynAdd(a, b):
    return (a + b) & 0xFFFF
""",
    # the sub-Definition SynTop resolves: imported only while SynTop's body is built, so no import statement names it
    "syn/sub.py": """
from verity.ir import composite

from syn.prims import SynAdd, V


@composite("SynSub", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynSub(B, S, a, b):
    return B.call(SynAdd, a, b)
""",
    "syn/top.py": """
from verity.ir import composite

from syn.prims import V


@composite("SynTop", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynTop(B, S, a, b):
    from syn.shape import operands
    from syn.sub import SynSub
    return B.call(SynSub, *operands(a, b))
""",
    # a helper SynTop's body builder runs: no Definition, and imported only during the build
    "syn/shape.py": "def operands(a, b):\n    return a, b\n",
    "syn/other.py": """
from verity.ir import composite

from syn.prims import SynAdd, V


@composite("SynOther", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynOther(B, S, a, b):
    return B.call(SynAdd, b, a)
""",
    # registered with the rest, resolved by no target
    "syn/idle.py": """
from verity.ir import composite

from syn.prims import SynAdd, V


@composite("SynIdle", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynIdle(B, S, a, b):
    return B.call(SynAdd, a, a)
""",
    "syn/h.py": "def compute():\n    return 7\n",
    # the checks: a Definition's gate count, its pin, and a value memoized across a run's targets (computed by `syn.h`)
    "syn/check.py": """
def check_definition(fn, opts, *, memo, pins, as_call=None, **kw):
    if "h" not in memo:
        from syn import h
        memo["h"] = h.compute()
    pin = pins.get("pieces", {}).get(fn.id)
    return {"failures": [], "warnings": [], "gates": fn.gates, "h": memo["h"], "pin": pin}
""",
    "targets.py": """
def _core_roots():
    return ["SynTop_v1"]


def _model_roots():
    return [
        ("SynOther_v1", lambda: "SynOther_v1"),
        ("Spare_v1", lambda: "Spare_v1"),
    ]


TEMPLATE_PARAMS = {"t": [{"K": 1}]}
""",
    "pins.json": json.dumps({"pieces": {"SynTop_v1": {"and": 3}, "SynOther_v1": {"and": 5}}}),
    ".gitignore": "cache/\n",
}

#: one cached `--all` run over the synthetic catalog; prints its cache stats and each target's result
HARNESS = """
import json, sys
from pathlib import Path

root, repo = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root))
from circuit_check import cache as CC, checks as CK, cli, targets as TG

CC.TARGETS, CC.PINS, CC._TARGETS_FILE = root / "targets.py", root / "pins.json", str(root / "targets.py")
CK.PINS = root / "pins.json"
cli.REPO = root
inner = CC._interpreter_dirs
CC._interpreter_dirs = lambda: (*inner(), str(repo) + "/")      # the checker's own modules come from the repository
CC.trace(root)

import syn.idle, syn.other, syn.top  # noqa: E401  (the registries, imported before any check, as `load_registries` does)
from syn import check
from verity.ir.defs import bind

suite = {"SynOther_v1": bind(syn.other.SynOther), "SynTop_v1": bind(syn.top.SynTop)}
for fn in suite.values():
    fn.body                                                     # built in the catalog, before every check
TG.load_registries = lambda: None
TG.suite = lambda: dict(suite)
TG.templates = lambda: []
TG.call_families = lambda: set()
TG.BINDINGS.update({"SynTop_v1": {"_core_roots"}, "SynOther_v1": {"_model_roots:SynOther_v1"}})
CK.check_definition = check.check_definition
rep = cli.run([], CK.Options(), everything=True, log=None, cache=root / "cache")
hits = sorted(t["target"] for t in rep["targets"] if "cache" in t)
print(json.dumps({"stats": {k: v for k, v in rep["cache"].items()}, "hits": hits}))
"""


def _tree(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for rel, text in FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text.lstrip("\n"))
    return tmp_path


def _run(root: Path) -> dict:
    r = subprocess.run([sys.executable, "-c", HARNESS, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert not out["stats"]["untraced"] and not out["stats"]["errors"] and not out["stats"]["changed"], out
    return out


def _edit(root: Path, rel: str, old: str, new: str) -> None:
    p = root / rel
    text = p.read_text()
    assert old in text
    p.write_text(text.replace(old, new, 1))


def _entry(root: Path, target: str) -> dict:
    (e,) = [json.loads(p.read_text()) for p in (root / "cache").glob("*/*/*.json")
            if json.loads(p.read_text())["key"]["request"]["target"][1] == target]
    return e


def test_a_target_holds_its_own_reads_and_resolutions(tmp_path):
    """Each target holds the module of the Definitions its check reached, the sub-Definition SynTop's build imported (no import
    statement names it), its own binding unit and pin, and none of the registry's other modules."""
    root = _tree(tmp_path)
    assert _run(root)["hits"] == []
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]
    top, other = _entry(root, "SynTop_v1")["reads"], _entry(root, "SynOther_v1")["reads"]
    assert {"syn/top.py", "syn/sub.py", "syn/prims.py", "syn/check.py", "syn/h.py"} <= set(top["files"])
    assert {"syn/other.py", "syn/prims.py", "syn/h.py"} <= set(other["files"])
    assert not {"syn/idle.py", "syn/top.py", "syn/sub.py"} & set(other["files"]) and "syn/idle.py" not in top["files"]
    assert set(top["units"]) == {"targets.py#_core_roots", "targets.py#common"}
    assert set(other["units"]) == {"targets.py#_model_roots:SynOther_v1", "targets.py#common"}
    assert set(top["pins"]) == {"pins.json#pieces#SynTop_v1"} and set(other["pins"]) == {"pins.json#pieces#SynOther_v1"}
    assert "targets.py" not in top["files"] and "pins.json" not in top["files"]


def test_editing_a_registry_module_no_target_resolves_reuses_every_target(tmp_path):
    root = _tree(tmp_path)
    _run(root)
    _edit(root, "syn/idle.py", "B.call(SynAdd, a, a)", "B.call(SynAdd, b, b)")
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]


def test_editing_a_sub_definition_a_target_resolves_reruns_that_target(tmp_path):
    """SynSub is reached only through SynTop's body, built before the check, and `syn.shape` only ran while it was built: the
    build's recording (and SynSub's body builder) carries `syn/sub.py`, the recording alone `syn/shape.py`."""
    root = _tree(tmp_path)
    _run(root)
    _edit(root, "syn/sub.py", "B.call(SynAdd, a, b)", "B.call(SynAdd, b, a)")
    out = _run(root)
    assert out["hits"] == ["SynOther_v1"] and out["stats"]["stored"] == 1
    _edit(root, "syn/shape.py", "return a, b", "return b, a")      # what only the build ran: its recording holds it
    out = _run(root)
    assert out["hits"] == ["SynOther_v1"] and out["stats"]["stored"] == 1


def test_another_binding_or_pin_reuses_every_target_and_its_own_reruns_it(tmp_path):
    """A new `_model_roots` entry, a new `TEMPLATE_PARAMS` entry, a new pin and a comment change no target's key; a target's
    own binding unit or its own pin does."""
    root = _tree(tmp_path)
    _run(root)
    _edit(root, "targets.py", '("Spare_v1", lambda: "Spare_v1"),', '("Spare_v1", lambda: "Spare_v1"),\n        ("New_v1", lambda: 0),  # new')
    _edit(root, "targets.py", '"t": [{"K": 1}]', '"t": [{"K": 1}], "u": [{"K": 2}]')
    pins = json.loads((root / "pins.json").read_text())
    pins["pieces"]["New_v1"] = {"and": 1}
    (root / "pins.json").write_text(json.dumps(pins))
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]
    _edit(root, "targets.py", 'return ["SynTop_v1"]', 'return ["SynTop_v1", "SynSub_v1"]')
    assert _run(root)["hits"] == ["SynOther_v1"]
    _edit(root, "targets.py", '("SynOther_v1", lambda: "SynOther_v1")', '("SynOther_v1", lambda: "SynOther_v1".lower())')
    assert _run(root)["hits"] == ["SynTop_v1"]
    pins["pieces"]["SynTop_v1"] = {"and": 4}
    (root / "pins.json").write_text(json.dumps(pins))
    assert _run(root)["hits"] == ["SynOther_v1"]


def test_a_target_that_would_reuse_a_memoized_result_reruns_when_its_dependency_changes(tmp_path):
    """`syn.h` is run by the first target whose check misses the memo; under a shared memo the second would reuse its value and
    never read `syn/h.py`.  Each traced check has a memo of its own, so both hold it."""
    root = _tree(tmp_path)
    _run(root)
    assert "syn/h.py" in _entry(root, "SynTop_v1")["reads"]["files"]
    _edit(root, "syn/h.py", "return 7", "return 8")
    assert _run(root)["hits"] == []


def test_the_catalogs_units_are_its_root_functions_and_entries():
    """The real `targets.py` splits into the units the cache keys on: each root function, each `_model_roots` entry, each
    `TEMPLATE_PARAMS` entry, and ``common``; a comment changes none, an entry's edit only its own."""
    import re

    from circuit_check import cache as CC
    from circuit_check import targets as TG

    src = CC.TARGETS.read_text()
    units, _spans = CC._units_of(src)
    labels = re.findall(r'^ +\("([^"]+)", lambda', src, re.M)
    assert {f"_model_roots:{label}" for label in labels} <= set(units) and len(labels) > 50
    assert {f"TEMPLATE_PARAMS:{tid}" for tid in TG.TEMPLATE_PARAMS} <= set(units)
    assert {"_core_roots", "_boolean_roots", "_pouw_roots", "_kernel_roots", "_vocabulary_bindings", "common"} <= set(units)
    assert CC._units_of(src.replace("\ndef catalog(", "\n# a comment\ndef catalog(", 1))[0] == units
    edited = CC._units_of(src.replace('("ResidualAdd_v1", lambda: bind(b1.ResidualAdd, N=16))',
                                      '("ResidualAdd_v1", lambda: bind(b1.ResidualAdd, N=32))', 1))[0]
    assert {u for u in units if edited.get(u) != units[u]} == {"_model_roots:ResidualAdd_v1"}
