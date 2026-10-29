"""The cache's per-target keys (`circuit_check.cache`, `circuit-check-cache/v2`), on a synthetic tree: a registry of small
Definitions, a catalog (`targets.py`) and shared pins of its own (``pins/``, a file per pin), checked through `cli.run`'s `--all` path with the catalog's
functions standing in for the real ones (the real catalog takes a minute to build).  A target's entry holds what its own check
read or resolved, and nothing another target or the task list read."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from circuit_check import entries as E

REPO = Path(__file__).resolve().parents[4]

FILES = {
    "syn/__init__.py": "",
    "syn/prims.py": """
from verity.ml.circuits import Value, primitive

V = Value(16)


@primitive("SynAdd", 1, [("a", V), ("b", V)], V)
def SynAdd(a, b):
    return (a + b) & 0xFFFF
""",
    # the sub-Definition SynTop resolves: imported only while SynTop's body is built, so no import statement names it
    "syn/sub.py": """
from verity.ml.circuits import composite

from syn.prims import SynAdd, V


@composite("SynSub", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynSub(B, S, a, b):
    return B.call(SynAdd, a, b)
""",
    "syn/top.py": """
from verity.ml.circuits import composite

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
from verity.ml.circuits import composite

from syn.prims import SynAdd, V


@composite("SynOther", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynOther(B, S, a, b):
    return B.call(SynAdd, b, a)
""",
    # registered with the rest, resolved by no target
    "syn/idle.py": """
from verity.ml.circuits import composite

from syn.prims import SynAdd, V


@composite("SynIdle", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynIdle(B, S, a, b):
    return B.call(SynAdd, a, a)
""",
    "syn/h.py": "def compute():\n    return 7\n",
    # what a check looks up in a module, which runs none of its code: imported by the first target's check, already loaded for
    # the next one's
    "syn/reg.py": 'KINDS = {"SynTop_v1": "top"}\n',
    # the checks: a Definition's gate count, its value on one input (its evaluators run), its pin, and a value memoized across a
    # run's targets (computed by `syn.h`)
    "syn/check.py": """
def check_definition(fn, opts, *, memo, pins, as_call=None, **kw):
    import inspect
    import sys

    from verity.ml.circuits.evaluation.reference import evaluate

    inspect.getmodule(sys._getframe())                 # as torch's import does: it looks for every loaded module's source
    if "h" not in memo:
        from syn import h
        memo["h"] = h.compute()
    from syn import reg
    pin = pins.get("pieces", {}).get(fn.id)
    return {"failures": [], "warnings": [], "gates": fn.gates, "value": evaluate(fn, 3, 0x0105), "h": memo["h"], "pin": pin,
            "kind": reg.KINDS.get(fn.id)}
""",
    # the bindings, as the harness runs them (`targets._produce`), and the families the suite keeps of what their roots reach
    "syn/catalog.py": """
from verity.ml.circuits.defs import bind


def core():
    from syn import extra, top
    return [bind(top.SynTop)]


def model():
    from syn import other
    return [bind(other.SynOther)]


def made():
    from syn import factory, ops
    return [bind(factory.make(ops.SWAP))]


def sized():
    from syn import ksize, kdef
    return [bind(kdef.SynK, K=ksize.K + 1)]


def xor():
    from syn import xdef
    return [bind(xdef.SynX)]


def setknob():
    from syn import knob, ksize
    knob.N[0] = ksize.K + 1
    return []


def knobbed():
    from syn import kn
    return [bind(kn.SynKn)]


BINDINGS = [(("_core_roots",), core), (("_model_roots:SynOther_v1",), model)]
FAMILIES = ["SynOther_v1", "SynTop_v1"]
""",
    # a registry module the core binding imports and uses nothing of
    "syn/extra.py": "X = 1\n",
    # a Definition a binding makes, whose body closes over a value the binding read from `syn.ops`
    "syn/factory.py": """
from verity.ml.circuits import composite

from syn.prims import SynAdd, V


def make(swap):
    @composite("SynMade", 1, [], lambda S: ((("a", V), ("b", V)), V))
    def SynMade(B, S, a, b):
        return B.call(SynAdd, b, a) if swap else B.call(SynAdd, a, b)
    return SynMade
""",
    "syn/ops.py": "SWAP = False\n",
    # a Definition bound at a static the binding computed from `syn.ksize`
    "syn/kdef.py": """
from verity.ml.circuits import composite

from syn.prims import SynAdd, V


@composite("SynK", 1, ["K"], lambda S: ((("a", V), ("b", V)), V))
def SynK(B, S, a, b):
    out = a
    for _ in range(S.K):
        out = B.call(SynAdd, out, b)
    return out
""",
    "syn/ksize.py": "K = 1\n",
    # a primitive whose evaluator reads a constant of another module, and the Definition that calls it
    "syn/consts.py": "MASK = 0xFFFF\n",
    "syn/xkern.py": """
from verity.ml.circuits import primitive

from syn.consts import MASK
from syn.prims import V


@primitive("SynXor", 1, [("a", V), ("b", V)], V)
def SynXor(a, b):
    return (a ^ b) & MASK
""",
    "syn/xdef.py": """
from verity.ml.circuits import composite

from syn.xkern import SynXor
from syn.prims import V


@composite("SynX", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynX(B, S, a, b):
    return B.call(SynXor, a, b)
""",
    # a Definition whose body reads module state another binding sets from a module only that binding read
    "syn/knob.py": "N = [1]\n",
    "syn/kn.py": """
from verity.ml.circuits import composite

from syn.prims import SynAdd, V


@composite("SynKn", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynKn(B, S, a, b):
    from syn import knob
    out = a
    for _ in range(knob.N[0]):
        out = B.call(SynAdd, out, b)
    return out
""",
    "targets.py": """
import importlib
import pkgutil


def load_registries():
    import syn
    from circuit_check import cache, sidecars
    for mod in sidecars.discover():
        importlib.import_module(mod)
    for m in pkgutil.iter_modules(syn.__path__):
        importlib.import_module(f"syn.{m.name}")
    cache.registries(sidecars.discover())


def _core_roots():
    return ["SynTop_v1"]


def _model_roots():
    return [
        ("SynOther_v1", lambda: "SynOther_v1"),
        ("Spare_v1", lambda: "Spare_v1"),
    ]


TEMPLATE_PARAMS = {"t": [{"K": 1}]}
""",
    "pins/pieces/SynTop_v1.json": json.dumps({"pieces": {"SynTop_v1": {"and": 3}}}),
    "pins/pieces/SynOther_v1.json": json.dumps({"pieces": {"SynOther_v1": {"and": 5}}}),
    ".gitignore": "cache/\n",
}

#: one cached `--all` run over the synthetic catalog; prints its cache stats and each target's result
HARNESS = """
import json, sys
from pathlib import Path

root, repo = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root))
from circuit_check import cache as CC, checks as CK, cli, sidecars as SC, targets as TG

CC.TARGETS, CC.PINS, CC._TARGETS_FILE = root / "targets.py", root / "pins", str(root / "targets.py")
CK.PINS = root / "pins"
SC.REPO, SC.PACKAGES = root, ("syn",)
cli.REPO = root
inner = CC._interpreter_dirs
CC._interpreter_dirs = lambda: (*inner(), str(repo) + "/")      # the checker's own modules come from the repository
CC.trace(root)

# the registries, imported before any check: enumerated by the catalog's own `load_registries`, or by name without one
import runpy
load = runpy.run_path(str(root / "targets.py")).get("load_registries")
if load is not None:
    load()
else:
    import syn.extra, syn.idle, syn.kdef, syn.other, syn.top
from syn import catalog, check

suite = {}
for units, make in catalog.BINDINGS:
    roots = TG._produce(units, make)
    for fid, fn in TG.reachable(roots).items():                 # built in the catalog, before every check
        if TG.family(fn) in catalog.FAMILIES:
            suite[fid] = fn
            TG.BINDINGS.setdefault(fid, set()).update(units)
for _mod, m in SC.binding_modules():                            # each sidecar's `_model_roots` entries, as `catalog` runs them
    for label, make in m._model_roots():
        units = (f"{m.__name__}#_model_roots:{label}",)
        for fid, fn in TG.reachable(TG._produce(units, lambda make=make: [make()])).items():
            if TG.family(fn) in catalog.FAMILIES:
                suite[fid] = fn
                TG.BINDINGS.setdefault(fid, set()).update(units)
TG.load_registries = lambda: None
TG.suite = lambda: dict(sorted(suite.items()))
TG.templates = lambda: []
TG.call_families = lambda found=None: set()
CK.check_definition = check.check_definition
rep = cli.run([], CK.Options(), everything=True, log=None, cache=root / "cache")
hits = sorted(t["target"] for t in rep["targets"] if "cache" in t)
print(json.dumps({"stats": {k: v for k, v in rep["cache"].items()}, "hits": hits}))
"""


def _pin(key: str) -> str:
    """Where the synthetic tree's shared pin ``key`` (a piece) is, or goes."""
    return E.path(Path("pins/pieces"), key, ".json").as_posix()


def _pins(root: Path, key: str) -> dict:
    return json.loads((root / _pin(key)).read_text())["pieces"][key]


def _set_pin(root: Path, key: str, counts: dict | None) -> None:
    """Pin ``key`` at ``counts`` in its shared file, or remove the file (None)."""
    p = root / _pin(key)
    if counts is None:
        p.unlink()
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"pieces": {key: counts}}))


def _tree(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for rel, text in FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text.lstrip("\n"))
    return tmp_path


def _run(root: Path, harness: str = HARNESS) -> dict:
    # no bytecode: an edit that keeps a file's size within the second its bytecode was written would run the old code
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", harness, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert not out["stats"]["untraced"] and not out["stats"]["errors"] and not out["stats"]["changed"], out
    return out


def _edit(root: Path, rel: str, old: str, new: str) -> None:
    p = root / rel
    text = p.read_text()
    assert old in text
    p.write_text(text.replace(old, new, 1))


def _values(root: Path, target: str) -> list:
    """The values every stored entry of ``target`` reported."""
    return sorted(e["report"]["value"] for p in (root / "cache").glob("*/*/*.json")
                  if (e := json.loads(p.read_text()))["key"]["request"]["target"][1] == target)


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
    assert {"syn.top", "syn.sub", "syn.prims", "syn.check", "syn.h"} <= set(top["modules"])
    assert {"syn.other", "syn.prims", "syn.h"} <= set(other["modules"])
    assert not {"syn.idle", "syn.top", "syn.sub"} & set(other["modules"]) and "syn.idle" not in top["modules"]
    assert not [f for f in [*top["files"], *other["files"]] if f.endswith(".py")]   # modules are held by name, not path
    assert set(top["units"]) == {"targets.py#_core_roots", "targets.py#common"}
    assert set(other["units"]) == {"targets.py#_model_roots:SynOther_v1", "targets.py#common"}
    assert set(top["pins"]) == {f"{_pin('SynTop_v1')}#pieces#SynTop_v1"}
    assert set(other["pins"]) == {f"{_pin('SynOther_v1')}#pieces#SynOther_v1"}
    assert "targets.py" not in top["files"] and not [f for f in top["files"] if f.startswith("pins/")]
    assert set(top["enums"]) == {"syn"} and {"top", "other", "idle"} <= set(top["enums"]["syn"])   # `load_registries` listed it


#: the harness with `syn` served as hatch's `dev-mode-exact` serves core's `verity`: by `editables`' finder, after the path-based
#: one, with the tree's root on no `sys.path` entry
EXACT = HARNESS.replace("sys.path.insert(0, str(root))",
                        "from editables.redirector import RedirectingFinder\n"
                        "RedirectingFinder.install()\n"
                        "RedirectingFinder.map_module('syn', str(root / 'syn' / '__init__.py'))")


def test_a_package_an_exact_editable_install_serves_is_held_like_one_on_the_path(tmp_path):
    """Its modules are the tree's, held by name and resolved through the redirect at lookup, and an edit to one a target
    resolved reruns that target (held by none, as before this, every target would be reused after any edit to core)."""
    assert EXACT != HARNESS
    root = _tree(tmp_path)
    assert _run(root, EXACT)["hits"] == []
    assert _run(root, EXACT)["hits"] == ["SynOther_v1", "SynTop_v1"]
    top = _entry(root, "SynTop_v1")
    assert {"syn.top", "syn.sub", "syn.prims", "syn.check", "syn.h"} <= set(top["reads"]["modules"])
    assert "path" not in top["key"] and "distributions" not in top["key"]
    _edit(root, "syn/sub.py", "B.call(SynAdd, a, b)", "B.call(SynAdd, b, a)")
    out = _run(root, EXACT)
    assert out["hits"] == ["SynOther_v1"] and out["stats"]["stored"] == 1


#: the harness with one more `sys.path` entry, holding a package no check imports: a new workspace member
EXTRA_PATH = HARNESS.replace("sys.path.insert(0, str(root))", "sys.path.insert(0, str(root))\nsys.path.append(str(root / 'extra'))")
#: the harness with every installed distribution at another version
BUMPED = HARNESS.replace("from circuit_check import cache", "import importlib.metadata as _md\n_md.version = lambda d: '0.0.0-bumped'\nfrom circuit_check import cache")


def test_a_new_path_entry_or_package_no_check_imports_reuses_every_target(tmp_path):
    """A new workspace member puts a distribution and a `sys.path` entry into the environment, and a key that held both whole
    would run every target again. A target holds only what it loaded."""
    root = _tree(tmp_path)
    (root / "extra" / "unrelated_pkg").mkdir(parents=True)
    (root / "extra" / "unrelated_pkg" / "__init__.py").write_text("X = 1\n")
    _run(root)
    assert _run(root, EXTRA_PATH)["hits"] == ["SynOther_v1", "SynTop_v1"]


def test_another_version_of_a_distribution_a_check_loaded_reruns_it(tmp_path):
    """Modules from outside the tree aren't followed file by file: each top-level one the process loaded is held by the
    versions of the distributions that provide it."""
    root = _tree(tmp_path)
    _run(root)
    assert _entry(root, "SynTop_v1")["reads"]["dists"], "the checks load numpy at least"
    assert _run(root, BUMPED)["hits"] == []


def test_a_package_in_the_tree_that_shadows_a_loaded_distribution_reruns_what_holds_it(tmp_path):
    """A name a check loaded from an installed distribution counts as ``tree`` once a file in the tree resolves it, so the
    entry no longer matches."""
    root = _tree(tmp_path)
    _run(root)
    held = _entry(root, "SynTop_v1")["reads"]["dists"]
    top = next(n for n in sorted(held) if n != "verity")       # the harness's checker and `verity` come from the repository
    assert held[top] not in (None, "tree")
    (root / top).mkdir()
    (root / top / "__init__.py").write_text("")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    import sys

    from circuit_check import cache as CC

    sys.path.insert(0, str(root))
    try:
        assert CC._dist(root, top) == "tree"
    finally:
        sys.path.remove(str(root))


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
    _set_pin(root, "New_v1", {"and": 1})
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]
    _edit(root, "targets.py", 'return ["SynTop_v1"]', 'return ["SynTop_v1", "SynSub_v1"]')
    assert _run(root)["hits"] == ["SynOther_v1"]
    _edit(root, "targets.py", '("SynOther_v1", lambda: "SynOther_v1")', '("SynOther_v1", lambda: "SynOther_v1".lower())')
    assert _run(root)["hits"] == ["SynTop_v1"]
    _set_pin(root, "SynTop_v1", {"and": 4})
    assert _run(root)["hits"] == ["SynOther_v1"]


#: SynOther's binding and pin moved out of `targets.py` and ``pins/`` into its module's sidecar
SIDECAR = """
def _model_roots():
    from verity.ml.circuits.defs import bind
    from syn import idle, other
    return [
        ("SynOther_v1", lambda: bind(other.SynOther)),
        ("SynIdle_v1", lambda: bind(idle.SynIdle)),
    ]
"""


def _sidecar(root: Path) -> None:
    text = (root / "syn/catalog.py").read_text()
    (root / "syn/catalog.py").write_text(text.replace(', (("_model_roots:SynOther_v1",), model)', "", 1))
    _edit(root, "targets.py", '("SynOther_v1", lambda: "SynOther_v1"),\n', "")
    (root / "syn/other.circuit-check.py").write_text(SIDECAR.lstrip("\n"))
    (root / "syn/other.circuit-check.json").write_text(json.dumps({"pieces": {"SynOther_v1": _pins(root, "SynOther_v1")}}))
    _set_pin(root, "SynOther_v1", None)


def test_a_sidecars_binding_and_pin_are_held_as_targets_pys_are(tmp_path):
    """A binding in a Definition module's sidecar is a unit of that file, named by its module (``<module>.circuit-check#<unit>``)
    with the file's ``common``, and a pin in its pins file is held there: editing another entry of the sidecar or another
    file's pin reuses the target, its own entry or pin reruns it and nothing else."""
    root = _tree(tmp_path)
    _sidecar(root)
    _run(root)
    other, top = _entry(root, "SynOther_v1")["reads"], _entry(root, "SynTop_v1")["reads"]
    side = "syn/other.circuit-check.py"
    assert set(other["units"]) == {"syn.other.circuit-check#_model_roots:SynOther_v1", "syn.other.circuit-check#common",
                                   "targets.py#common"}
    assert set(other["pins"]) == {"syn.other.circuit-check#pieces#SynOther_v1"}
    assert set(top["pins"]) == {f"{_pin('SynTop_v1')}#pieces#SynTop_v1"}
    assert not {side, "syn/other.circuit-check.json"} & set(other["files"])
    _edit(root, side, '("SynIdle_v1", lambda: bind(idle.SynIdle))', '("SynIdle_v1", lambda: bind(idle.SynIdle, **{}))')
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]
    _edit(root, side, '("SynOther_v1", lambda: bind(other.SynOther))', '("SynOther_v1", lambda: bind(other.SynOther, **{}))')
    assert _run(root)["hits"] == ["SynTop_v1"]
    pins = json.loads((root / "syn/other.circuit-check.json").read_text())
    pins["pieces"]["SynOther_v1"] = {"and": 6}
    (root / "syn/other.circuit-check.json").write_text(json.dumps(pins))
    assert _run(root)["hits"] == ["SynTop_v1"]


def test_an_unpinned_target_reruns_when_its_pin_is_added_where_it_goes_or_to_its_shared_file(tmp_path):
    """SynK has no pin: its check read the absence in the pins file a new pin of it goes to (its module's sidecar) and in its
    file of ``pins/``, so adding it to either reruns SynK alone.  (A new file in `syn` would rerun every target, whose imports
    listed the directory, so the sidecar is there from the start.)"""
    root = _tree(tmp_path)
    _bindings(root, '(("_sized_roots",), sized)', families=("SynK_v1",))
    _edit(root, "syn/catalog.py", '(("_model_roots:SynOther_v1",), model), ', "")
    _sidecar(root)
    (root / "syn/kdef.circuit-check.json").write_text("{}")
    _run(root)
    assert set(_entry(root, "SynK_v1{K=2}")["reads"]["pins"]) == {"syn.kdef.circuit-check#pieces#SynK_v1{K=2}",
                                                                  f"{_pin('SynK_v1{K=2}')}#pieces#SynK_v1{{K=2}}"}
    (root / "syn/kdef.circuit-check.json").write_text(json.dumps({"pieces": {"SynK_v1{K=2}": {"and": 2}}}))
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]
    (root / "syn/kdef.circuit-check.json").write_text("{}")
    _run(root)
    _set_pin(root, "SynK_v1{K=2}", {"and": 2})
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]


def test_a_new_pin_goes_to_the_sidecar_of_its_definitions_module(tmp_path):
    """`sidecars.pin_file`: a pin is written where it is, and a new one to the pins file of the module that defines its
    Definition, a module without a sidecar getting one beside it."""
    root = _tree(tmp_path)
    _sidecar(root)
    code = HARNESS.split("CC.trace(root)")[0] + """
import syn.kdef, syn.other, syn.top  # noqa: E401
CK.load_pins()
SC.write_pins({("pieces", "SynOther_v1"): {"and": 7}, ("pieces", "SynTop_v1"): {"and": 8}, ("pieces", "SynK_v1"): {"and": 9},
               ("pieces", "Nobody_v1"): {"and": 1}})
"""
    r = subprocess.run([sys.executable, "-c", code, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    read = lambda rel: json.loads((root / rel).read_text())["pieces"]  # noqa: E731
    assert read("syn/other.circuit-check.json") == {"SynOther_v1": {"and": 7}}
    assert read("syn/kdef.circuit-check.json") == {"SynK_v1": {"and": 9}}
    assert read(_pin("SynTop_v1")) == {"SynTop_v1": {"and": 8}} and read(_pin("Nobody_v1")) == {"Nobody_v1": {"and": 1}}


def test_a_new_sidecar_reruns_only_the_targets_that_reached_what_its_module_registers(tmp_path):
    """A sidecar makes the registries import its module (`targets.load_registries`), so the task list's key holds the set of
    them; a target's holds none of it, only whether the module that registered each Definition its check reached has one
    (``registrars``).  A pins file beside `syn.idle` (SynIdle, which no target reaches) and a binding file beside `syn.extra`
    (no Definition) reuse every target, though the task list is built again; one beside `syn.sub` reruns SynTop, whose body
    calls SynSub, and not SynOther."""
    root = _tree(tmp_path)
    _run(root)
    assert _run(root)["stats"].get("task_list")
    (root / "syn/idle.circuit-check.json").write_text("{}")
    (root / "syn/extra.circuit-check.py").write_text("def _model_roots():\n    return []\n")
    out = _run(root)
    assert out["hits"] == ["SynOther_v1", "SynTop_v1"] and "task_list" not in out["stats"]
    assert {"SynTop_v1", "SynSub_v1", "SynAdd_v1"} <= set(_entry(root, "SynTop_v1")["reads"]["registrars"])
    (root / "syn/sub.circuit-check.json").write_text("{}")
    out = _run(root)
    assert out["hits"] == ["SynOther_v1"] and out["stats"]["stored"] == 1


def test_removing_a_sidecar_reruns_the_targets_it_bound_and_those_that_reached_what_its_module_registers(tmp_path):
    """SynOther is bound by the catalog and in its module's sidecar (`SIDECAR`), whose pins file holds its pin, and `syn.sub` has
    a pins file: removing `syn.other`'s sidecar (its pin back in ``pins/``) reruns SynOther, whose units, pin and registration
    changed, and removing `syn.sub`'s reruns SynTop, which reached SynSub, each and nothing else.  (The first check a process
    runs holds every loaded module's source, `syn.check`'s `inspect` lookup: SynOther's, checked before SynTop.)"""
    root = _tree(tmp_path)
    pin = _pins(root, "SynOther_v1")
    (root / "syn/other.circuit-check.py").write_text(SIDECAR.lstrip("\n"))
    (root / "syn/other.circuit-check.json").write_text(json.dumps({"pieces": {"SynOther_v1": pin}}))
    _set_pin(root, "SynOther_v1", None)
    (root / "syn/sub.circuit-check.json").write_text("{}")
    _run(root)
    assert "syn.other.circuit-check#_model_roots:SynOther_v1" in _entry(root, "SynOther_v1")["reads"]["units"]
    for ext in (".py", ".json"):
        (root / f"syn/other.circuit-check{ext}").unlink()
    _set_pin(root, "SynOther_v1", pin)
    out = _run(root)
    assert out["hits"] == ["SynTop_v1"] and out["stats"]["stored"] == 1
    (root / "syn/sub.circuit-check.json").unlink()
    out = _run(root)
    assert out["hits"] == ["SynOther_v1"] and out["stats"]["stored"] == 1


#: a check that looks SynLate up in the registry (absent until a sidecar makes the registries import `syn.late.mod`), finds the
#: Boolean Definition of SynAdd's word as `verity_flock.ir_lower._boolean_build` does (`cache._WORD_ENUMERATORS`), and, for
#: SynOther, counts every registered Definition
LOOKUPS = {
    "syn/lookup.py": """
def boolean_of(key):
    from verity.ml.circuits.defs import REGISTRY
    for d in list(REGISTRY.defs.values()):
        if callable(getattr(d, "build", None)) and getattr(getattr(d, "word", None), "id", None) == key:
            return d.id
    return None


def check_definition(fn, opts, *, memo, pins, as_call=None, **kw):
    from verity.ml.circuits.defs import REGISTRY
    got = {"failures": [], "warnings": [], "gates": fn.gates, "late": "SynLate_v1" in REGISTRY, "bits": boolean_of("SynAdd_v1")}
    if fn.id == "SynOther_v1":
        got["n"] = len(REGISTRY.defs)
    return got
""",
    "syn/late/__init__.py": "",
    "syn/late/mod.py": """
from verity.ml.circuits import composite

from syn.prims import SynAdd, V


@composite("SynLate", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynLate(B, S, a, b):
    return B.call(SynAdd, a, b)
""",
    "syn/late/bits.py": """
from types import SimpleNamespace

from verity.ml.circuits.defs import REGISTRY

from syn.prims import SynAdd

REGISTRY.add(SimpleNamespace(id="SynAddBits_v1", word=SynAdd, build=lambda *a: None))
""",
}
LOOKUP = HARNESS.replace("from syn import catalog, check\n",
                         "from syn import catalog\nfrom syn import lookup as check\n"
                         'CC._WORD_ENUMERATORS[("syn.lookup", "boolean_of")] = "key"\n')


def _reports(root: Path, target: str) -> list[dict]:
    return [e["report"] for p in (root / "cache").glob("*/*/*.json")
            if (e := json.loads(p.read_text()))["key"]["request"]["target"][1] == target]


def test_a_sidecar_whose_module_registers_what_a_check_looked_up_absent_reruns_it(tmp_path):
    """`syn.late`'s modules are imported by nothing until a sidecar makes the registries import them: SynLate, which each check
    looks up, is absent (``absent:SynLate_v1``), and so is the Boolean Definition of SynAdd's word (``word:SynAdd_v1``); a
    sidecar beside either module reruns both targets, whose verdict it changes.  SynOther's check enumerated the whole registry
    (``*``), which holds every sidecar as the key did: the pins file beside `syn.idle` reruns it, and not SynTop."""
    assert LOOKUP != HARNESS
    root = _tree(tmp_path)
    for rel, text in LOOKUPS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text.lstrip("\n"))
    _run(root, LOOKUP)
    top, other = _entry(root, "SynTop_v1")["reads"]["registrars"], _entry(root, "SynOther_v1")["reads"]["registrars"]
    assert {"absent:SynLate_v1", "word:SynAdd_v1"} <= set(top) and "*" not in top and "*" in other
    (root / "syn/idle.circuit-check.json").write_text("{}")
    assert _run(root, LOOKUP)["hits"] == ["SynTop_v1"]
    (root / "syn/late/mod.circuit-check.json").write_text("{}")
    assert _run(root, LOOKUP)["hits"] == []
    assert sorted(r["late"] for r in _reports(root, "SynTop_v1")) == [False, True]
    (root / "syn/late/bits.circuit-check.json").write_text("{}")
    assert _run(root, LOOKUP)["hits"] == []
    assert sorted(str(r["bits"]) for r in _reports(root, "SynTop_v1")) == ["None", "None", "SynAddBits_v1"]


def test_a_sidecar_whose_module_is_gone_is_an_error(tmp_path, monkeypatch):
    import pytest

    from circuit_check import sidecars as SC

    (tmp_path / "synorphan").mkdir()
    (tmp_path / "synorphan/__init__.py").write_text("")
    (tmp_path / "synorphan/kept.py").write_text("")
    (tmp_path / "synorphan/kept.circuit-check.json").write_text("{}")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(SC, "PACKAGES", ("synorphan",))
    SC.discover.cache_clear()
    try:
        assert list(SC.discover()) == ["synorphan.kept"]
        (tmp_path / "synorphan/gone.circuit-check.py").write_text("")
        SC.discover.cache_clear()
        with pytest.raises(FileNotFoundError, match="synorphan.gone, which has no module"):
            SC.discover()
    finally:
        SC.discover.cache_clear()


def test_a_sidecar_under_two_packages_roots_is_named_once_from_the_deeper(tmp_path, monkeypatch):
    from circuit_check import sidecars as SC

    (tmp_path / "synouter/src/syninner").mkdir(parents=True)
    (tmp_path / "synouter/__init__.py").write_text("")
    (tmp_path / "synouter/src/syninner/__init__.py").write_text("")
    (tmp_path / "synouter/src/syninner/rec.py").write_text("")
    (tmp_path / "synouter/src/syninner/rec.circuit-check.json").write_text('{"definitions": {"SynRec_v1": {}}}')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path / "synouter/src"))
    monkeypatch.setattr(SC, "PACKAGES", ("synouter", "syninner"))
    SC.discover.cache_clear()
    try:
        assert list(SC.discover()) == ["syninner.rec"]
        assert SC.pin_files()[len(SC._shared_files()):] == [(tmp_path / "synouter/src/syninner/rec.circuit-check.json").resolve()]
    finally:
        SC.discover.cache_clear()


def test_a_sidecar_behind_a_namespace_directory_is_named_only_from_its_own_package(tmp_path, monkeypatch):
    """Two roots side by side, the outer holding a copy of the inner's tree behind a directory without ``__init__.py`` (as
    `verity_flock`'s stage-cache test copies the sources): the copy is no module of the outer package, so importing it under
    that name would register its Definitions a second time."""
    from circuit_check import sidecars as SC

    for base in (tmp_path / "synout2/src/synin2", tmp_path / "synin2"):
        base.mkdir(parents=True)
        (base / "__init__.py").write_text("")
        (base / "rec.py").write_text("")
        (base / "rec.circuit-check.json").write_text('{"definitions": {"SynRec_v1": {}}}')
    (tmp_path / "synout2/__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(SC, "PACKAGES", ("synout2", "synin2"))
    SC.discover.cache_clear()
    try:
        assert list(SC.discover()) == ["synin2.rec"]
    finally:
        SC.discover.cache_clear()


def test_every_package_that_defines_a_definition_is_one_sidecars_are_found_in():
    """`sidecars.PACKAGES` holds the top-level package of every module that defines a registered Definition: a sidecar in any
    other package is never read, so its bindings would leave the task list without an error."""
    from circuit_check import sidecars as SC
    from circuit_check import targets as TG
    from verity.ml.circuits.defs import REGISTRY

    TG.load_registries()
    tops = {m.partition(".")[0] for d in REGISTRY.defs.values() if TG._authored(d) and (m := SC.owner(d))}
    assert tops <= set(SC.PACKAGES), sorted(tops - set(SC.PACKAGES))


def test_a_target_that_would_reuse_a_memoized_result_reruns_when_its_dependency_changes(tmp_path):
    """`syn.h` is run by the first target whose check misses the memo; under a shared memo the second would reuse its value and
    never read `syn/h.py`.  Each traced check has a memo of its own, so both hold it."""
    root = _tree(tmp_path)
    _run(root)
    assert "syn.h" in _entry(root, "SynTop_v1")["reads"]["modules"]
    _edit(root, "syn/h.py", "return 7", "return 8")
    assert _run(root)["hits"] == []


def test_a_module_an_earlier_target_imported_is_held_by_every_check_that_imports_it(tmp_path):
    """`syn.reg` runs only when the first target's check imports it; the second's import finds it loaded and runs none of it,
    yet what it looks up there is its verdict."""
    root = _tree(tmp_path)
    _run(root)
    assert all("syn.reg" in _entry(root, t)["reads"]["modules"] for t in ("SynOther_v1", "SynTop_v1"))
    _edit(root, "syn/reg.py", '"top"', '"tip"')
    assert _run(root)["hits"] == []


#: body reads under nested recordings: which of SynTop and the SynSub its body calls each recording noted
BODY_READS = """
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from circuit_check import cache as CC
from verity.ml.circuits.defs import bind

CC.trace(root)
from syn import top

top_, sub = (sp := bind(top.SynTop)), sp.body.nodes[0].fn
names = lambda fr: sorted(n for n, d in (("top", top_), ("sub", sub)) if id(d) in fr.defs)
got = {}
with CC.recording() as a:
    for _ in range(3):
        top_.body
    with CC.recording(merge=False) as b:
        sub.body
        sub.body
    got["b"], got["a then"] = names(b), names(a)
    sub.body
    got["a"] = names(a)
with CC.recording() as c:
    top_.body
    with CC.recording() as d:
        top_.body
    got["c"], got["d"] = names(c), names(d)
print(json.dumps(got))
"""


def test_a_body_read_is_noted_in_every_recording_that_reads_it(tmp_path):
    """A check's layout walks read a specialization's body millions of times, so the tracer notes a read once in the recording
    it happens in rather than on every read; a read in any other recording (a nested one, its encloser once it ends, a later
    one) is noted there again, however often the body was read before."""
    root = _tree(tmp_path)
    r = subprocess.run([sys.executable, "-c", BODY_READS, str(root)], capture_output=True, text=True, cwd=REPO, check=True)
    assert json.loads(r.stdout.strip().splitlines()[-1]) == {"b": ["sub"], "a then": ["top"], "a": ["sub", "top"],
                                                             "c": ["top"], "d": ["top"]}


def _bindings(root: Path, *extra: str, families: tuple[str, ...] = ()) -> None:
    """Run the catalog's bindings ``extra`` beside the core and model ones, and keep ``families`` too."""
    _edit(root, "syn/catalog.py", '(("_model_roots:SynOther_v1",), model)]',
          '(("_model_roots:SynOther_v1",), model)' + "".join(f", {e}" for e in extra) + "]")
    _edit(root, "syn/catalog.py", 'FAMILIES = [', "FAMILIES = [" + "".join(f'"{f}", ' for f in families))


def test_a_registry_module_a_binding_imports_reruns_only_what_that_binding_made(tmp_path):
    """The core binding imports `syn.extra` and makes SynTop_v1; SynSub_v1, which its root reaches, was made by SynTop's body
    build, after the binding ran, so it holds the binding's unit but not what the binding imported."""
    root = _tree(tmp_path)
    _bindings(root, families=("SynSub_v1",))
    _run(root)
    top, sub = _entry(root, "SynTop_v1")["reads"], _entry(root, "SynSub_v1")["reads"]
    assert "syn.extra" in top["modules"] and "syn.extra" not in sub["modules"]
    assert "targets.py#_core_roots" in sub["units"]
    _edit(root, "syn/extra.py", "X = 1", "X = 2")
    assert _run(root)["hits"] == ["SynOther_v1", "SynSub_v1"]


def test_a_definition_a_binding_made_reruns_when_a_module_the_binding_read_changes(tmp_path):
    """SynMade's body closes over `syn.ops.SWAP`, which only the binding read: the binding's text is unchanged, the
    Definition's id too, and its body is not."""
    root = _tree(tmp_path)
    _bindings(root, '(("_made_roots",), made)', families=("SynMade_v1",))
    _run(root)
    assert "syn.ops" in _entry(root, "SynMade_v1")["reads"]["modules"]
    _edit(root, "syn/ops.py", "SWAP = False", "SWAP = True")
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]


def test_a_static_a_binding_computed_from_a_module_reruns_its_definition(tmp_path):
    """SynK is bound at ``K = ksize.K + 1``: a change to `syn.ksize` changes the specialization's id and the module the binding
    read, so it reruns either way, and the other targets do not; a comment is no change of the module."""
    root = _tree(tmp_path)
    _bindings(root, '(("_sized_roots",), sized)', families=("SynK_v1",))
    out = _run(root)
    assert out["stats"]["stored"] == 3
    assert "syn.ksize" in _entry(root, "SynK_v1{K=2}")["reads"]["modules"]
    _edit(root, "syn/ksize.py", "K = 1", "K = 1  # unchanged value")
    assert _run(root)["hits"] == ["SynK_v1{K=2}", "SynOther_v1", "SynTop_v1"]
    _edit(root, "syn/ksize.py", "K = 1  # unchanged value", "K = 2")
    out = _run(root)
    assert out["hits"] == ["SynOther_v1", "SynTop_v1"] and out["stats"]["stored"] == 1


def _move(root: Path, src: str, dst: str, *rewrites: tuple[str, str, str]) -> None:
    """Move ``src`` to ``dst`` (a new package's ``__init__.py`` a docstring alone) and rewrite the import lines ``rewrites``."""
    (root / dst).parent.mkdir(parents=True, exist_ok=True)
    if not (init := (root / dst).parent / "__init__.py").exists():
        init.write_text('"""Families, moved here."""\n')
    (root / src).rename(root / dst)
    for rel, old, new in rewrites:
        _edit(root, rel, old, new)


def test_a_pure_move_of_a_registry_module_reuses_every_target(tmp_path):
    """`syn/top.py` (SynTop's module, which `load_registries` enumerates) moves to a new package `syn/fam/`, and the one import
    that names it is rewritten, split from the import it shared and put first (as an import sorter orders them): no module's
    content changes, nor any binding unit, and the
    moved module is imported anyway (by the core binding), so every target is reused, SynTop's included.  The checks' `inspect`
    lookups probed `syn/top.py`, which is held as the module, not the path.  Each reused entry is stored again where its modules
    are now, so a later edit of the moved module reruns SynTop alone, not every target that held the enumeration."""
    root = _tree(tmp_path)
    _run(root)
    assert any("syn.top" in _entry(root, t)["reads"]["present"] for t in ("SynOther_v1", "SynTop_v1"))
    _move(root, "syn/top.py", "syn/fam/top.py",
          ("syn/catalog.py", "from syn import extra, top", "from syn.fam import top\n    from syn import extra"))
    out = _run(root)
    assert out["hits"] == ["SynOther_v1", "SynTop_v1"] and out["stats"]["stored"] == 0
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]
    _edit(root, "syn/fam/top.py", "*operands(a, b)", "*operands(b, a)")
    assert _run(root)["hits"] == ["SynOther_v1"]


def test_a_pure_move_of_a_definition_module_with_a_sidecar_reuses_its_targets(tmp_path):
    """SynOther's binding and pin are in its module's sidecar (`_sidecar`).  `syn/other.py` moves to `syn/fam/` with both
    sidecar files, and every import line that names it is rewritten, the sidecar's own split and put first: a sidecar's units
    and pins are named by its module and read where its module is now, so both targets are reused.  Each is stored again under
    the module's new name, so afterwards an edit of the moved sidecar's other entry reruns nothing, and one of its binding of
    SynOther, or of SynOther's pin, reruns SynOther alone."""
    root = _tree(tmp_path)
    _sidecar(root)
    _run(root)
    assert "syn.other" in _entry(root, "SynOther_v1")["reads"]["modules"]
    side = _move_other(root)
    out = _run(root)
    assert out["hits"] == ["SynOther_v1", "SynTop_v1"] and out["stats"]["stored"] == 0
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]
    _edit(root, side, '("SynIdle_v1", lambda: bind(idle.SynIdle))', '("SynIdle_v1", lambda: bind(idle.SynIdle, **{}))')
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]
    _edit(root, side, '("SynOther_v1", lambda: bind(other.SynOther))', '("SynOther_v1", lambda: bind(other.SynOther, **{}))')
    assert _run(root)["hits"] == ["SynTop_v1"]
    (root / "syn/fam/other.circuit-check.json").write_text(json.dumps({"pieces": {"SynOther_v1": {"and": 6}}}))
    assert _run(root)["hits"] == ["SynTop_v1"]


def _move_other(root: Path) -> str:
    """`syn/other.py` moved to `syn/fam/` with both its sidecar files, and every import line that names it rewritten (the
    sidecar's own split and put first); the moved binding file's path."""
    _move(root, "syn/other.py", "syn/fam/other.py", ("syn/catalog.py", "from syn import other", "from syn.fam import other"))
    for ext in (".py", ".json"):
        (root / f"syn/other.circuit-check{ext}").rename(root / f"syn/fam/other.circuit-check{ext}")
    side = "syn/fam/other.circuit-check.py"
    _edit(root, side, "from syn import idle, other", "from syn.fam import other\n    from syn import idle")
    return side


def test_a_sidecars_units_move_with_its_module_where_the_target_does_not_hold_the_module(tmp_path):
    """SynAdd, which both of the sidecar's entries reach, holds their units but not `syn.other`, which only SynOther's check
    reads (as every gate a model reaches holds the bindings of that model's Definitions).  The move finds the module where the
    units' import sites of it lead now, the one imported module with the content they recorded, so SynAdd is reused too; it is
    stored again there, so a later edit of an entry it holds reruns it.  The module starts in a package with code of its own,
    `syn.models.attn`, which SynAdd does not hold either (no enumeration lists it) and which the module is imported through
    before the move as after."""
    root = _tree(tmp_path)
    _sidecar(root)
    _edit(root, "syn/catalog.py", 'FAMILIES = ["SynOther_v1"', 'FAMILIES = ["SynAdd_v1", "SynOther_v1"')
    was, pkg = "syn/models/attn", "syn.models.attn"
    _move(root, "syn/other.py", f"{was}/other.py", ("syn/catalog.py", "from syn import other", f"from {pkg} import other"))
    (root / "syn/models/__init__.py").write_text('"""Models."""\n')
    (root / f"{was}/__init__.py").write_text("NAME = 'attn'\n")
    for ext in (".py", ".json"):
        (root / f"syn/other.circuit-check{ext}").rename(root / f"{was}/other.circuit-check{ext}")
    _edit(root, f"{was}/other.circuit-check.py", "from syn import idle, other", f"from {pkg} import other\n    from syn import idle")
    _run(root)
    add = _entry(root, "SynAdd_v1")["reads"]
    assert f"{pkg}.other.circuit-check#_model_roots:SynIdle_v1" in add["units"]
    assert not {pkg, f"{pkg}.other"} & set(add["modules"])
    _move(root, f"{was}/other.py", f"{was}/fam/other.py", ("syn/catalog.py", f"from {pkg} import other", f"from {pkg}.fam import other"))
    for ext in (".py", ".json"):
        (root / f"{was}/other.circuit-check{ext}").rename(root / f"{was}/fam/other.circuit-check{ext}")
    side = f"{was}/fam/other.circuit-check.py"
    _edit(root, side, f"from {pkg} import other", f"from {pkg}.fam import other")
    out = _run(root)
    assert out["hits"] == ["SynAdd_v1", "SynOther_v1", "SynTop_v1"] and out["stats"]["stored"] == 0
    _edit(root, side, '("SynIdle_v1", lambda: bind(idle.SynIdle))', '("SynIdle_v1", lambda: bind(idle.SynIdle, **{}))')
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]


def test_a_move_that_is_not_pure_reruns_what_it_changes(tmp_path):
    """A moved module that reads its own path (`__file__`) reruns what holds it; a registry module moved where nothing
    imports it is no longer registered by the enumeration, which every target holds, so every target reruns."""
    root = _tree(tmp_path)
    _edit(root, "syn/shape.py", "def operands", "HERE = __file__\n\n\ndef operands")
    _run(root)
    _move(root, "syn/shape.py", "syn/fam/shape.py", ("syn/top.py", "from syn.shape import", "from syn.fam.shape import"))
    assert _run(root)["hits"] == ["SynOther_v1"]
    _move(root, "syn/idle.py", "syn/fam/idle.py")
    assert _run(root)["hits"] == []


def test_a_descriptor_change_reruns_exactly_its_target_when_no_module_it_holds_changed(tmp_path):
    """SynKn's body reads `syn.knob.N`, which another binding sets from `syn.ksize` and which makes nothing SynKn reaches, so
    no module SynKn holds changes when `syn.ksize` does; its body, and so its descriptor, does, and SynKn alone reruns."""
    root = _tree(tmp_path)
    _bindings(root, '(("_set_roots",), setknob)', '(("_knobbed_roots",), knobbed)', families=("SynKn_v1",))
    assert _run(root)["hits"] == []
    assert "syn.ksize" not in _entry(root, "SynKn_v1")["reads"]["modules"]
    _edit(root, "syn/ksize.py", "K = 1", "K = 2")
    out = _run(root)
    assert out["hits"] == ["SynOther_v1", "SynTop_v1"] and out["stats"]["stored"] == 1
    assert _values(root, "SynKn_v1") == [[3 + 2 * 0x0105], [3 + 3 * 0x0105]]


def test_an_evaluator_or_a_module_constant_it_reads_reruns_its_target(tmp_path):
    """SynX calls the primitive SynXor, whose evaluator (`syn.xkern`) masks with `syn.consts.MASK`: neither is in SynX's
    descriptor (a primitive's entry is its signature), and each is held as a module SynX's check ran or imported."""
    root = _tree(tmp_path)
    _bindings(root, '(("_xor_roots",), xor)', families=("SynX_v1",))
    assert _run(root)["hits"] == []
    assert _entry(root, "SynX_v1")["report"]["value"] == [3 ^ 0x0105]
    _edit(root, "syn/xkern.py", "return (a ^ b) & MASK", "return (b ^ a) & MASK")
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]
    _edit(root, "syn/consts.py", "MASK = 0xFFFF", "MASK = 0x00FF")
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]
    assert [(3 ^ 0x0105) & 0xFF] in _values(root, "SynX_v1")


def test_what_runs_of_a_module_is_its_source_not_bytecode_of_another_version(tmp_path):
    """Python takes a module's cached bytecode when its header matches the source's mtime, in whole seconds, and its size: an
    edit of the same size within the second the bytecode was written (here the source's mtime set back to the old one) runs the
    old code, while the key digests the new source.  A module the cache imports while tracing is compiled from its source, so
    SynX's verdict after such an edit of its evaluator is the new one; one imported before tracing from such bytecode makes the
    process uncacheable, and stale bytecode Python did not take (its header no longer matches) does not."""
    import importlib.util
    import py_compile

    root = _tree(tmp_path)
    _bindings(root, '(("_xor_roots",), xor)', families=("SynX_v1",))
    assert _run(root)["hits"] == []
    p = root / "syn/xkern.py"
    py_compile.compile(str(p), doraise=True)
    before = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(root)!r})
import syn.xkern
from circuit_check import cache as CC
CC.trace(Path({str(root)!r}))
print(CC.untraced())
"""
    pre = subprocess.run([sys.executable, "-c", before], capture_output=True, text=True, cwd=REPO, check=True)
    assert pre.stdout.strip() == "[]"
    st = p.stat()
    _edit(root, "syn/xkern.py", "return (a ^ b) & MASK", "return (a | b) & MASK")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    head = Path(importlib.util.cache_from_source(str(p))).read_bytes()[:16]
    assert int.from_bytes(head[8:12], "little") == int(p.stat().st_mtime) and int.from_bytes(head[12:16], "little") == st.st_size
    assert _run(root)["hits"] == ["SynOther_v1", "SynTop_v1"]
    assert [3 | 0x0105] in _values(root, "SynX_v1")
    post = subprocess.run([sys.executable, "-c", before], capture_output=True, text=True, cwd=REPO, check=True)
    assert "syn/xkern.py: imported before tracing from bytecode that is not its source's" in post.stdout
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    later = subprocess.run([sys.executable, "-c", before], capture_output=True, text=True, cwd=REPO, check=True, env=env)
    assert Path(importlib.util.cache_from_source(str(p))).read_bytes()[:16] == head
    assert later.stdout.strip() == "[]"                # the old bytecode is still there, but its header no longer matches


#: a kernel module imported, as `verity.ml.definitions.numerics` imports its rows, by a registry module the catalog enumerates
KERNEL_FILES = {
    "syn/rows.py": """
import numpy as np

from verity.ml.circuits.evaluation import register_kernel

register_kernel("SynTop_v1", lambda a, b: ((a.astype(np.uint64) + b) & 0xFFFF, None))
""",
    "syn/kernels.py": "from syn import rows\n",
    "syn/corr.py": """
def check_definition(fn, opts, *, memo, pins, as_call=None, **kw):
    from circuit_check import checks as CK
    from verity.ml.circuits.evaluation.reference import resolve

    F = CK.Findings()
    CK.correspondence(resolve(fn), opts, F)
    return {"failures": F.failures, "warnings": F.warnings}
""",
}


def test_a_pure_move_of_a_kernel_module_reuses_its_target_and_a_kernel_edit_reruns_it(tmp_path):
    """SynTop's check looks up its kernel, registered in `syn/rows.py`: the lookup is held by where the kernel is defined in
    the module's content, not by its path, so moving `syn/rows.py` to `syn/fam/rows.py` (its one import rewritten) reuses
    SynTop's entry, and a wrong kernel there afterwards reruns it, which fails `correspondence/kernel-mismatch`."""
    root = _tree(tmp_path)
    for rel, text in KERNEL_FILES.items():
        (root / rel).write_text(text.lstrip("\n"))
    harness = HARNESS.replace("from syn import catalog, check\n", "from syn import catalog\nfrom syn import corr as check\n")
    assert _run(root, harness)["hits"] == []
    assert list(_entry(root, "SynTop_v1")["reads"]["kernels"]) == ["SynTop_v1"]
    _move(root, "syn/rows.py", "syn/fam/rows.py", ("syn/kernels.py", "from syn import rows", "from syn.fam import rows"))
    assert _run(root, harness)["hits"] == ["SynOther_v1", "SynTop_v1"]
    _edit(root, "syn/fam/rows.py", "+ b) & 0xFFFF", "- b) & 0xFFFF")
    assert _run(root, harness)["hits"] == ["SynOther_v1"]
    codes = {tuple(sorted(f"{f['check']}/{f['code']}" for f in e["report"]["failures"]))
             for p in (root / "cache").glob("*/*/*.json")
             if (e := json.loads(p.read_text()))["key"]["request"]["target"][1] == "SynTop_v1"}
    assert codes == {(), ("correspondence/kernel-mismatch",)}


def test_the_catalogs_units_are_its_root_functions_and_entries():
    """The real binding files (`targets.py` and each sidecar) split into the units the cache keys on: each root function, each
    `_model_roots` entry, each `TEMPLATE_PARAMS` entry, and each file's ``common``; a comment changes none, an entry's edit
    only its own."""
    import re

    from circuit_check import cache as CC
    from circuit_check import sidecars as SC
    from circuit_check import targets as TG

    srcs = {p: p.read_text() for p in (CC.TARGETS, *(f["py"] for f in SC.discover().values() if "py" in f))}
    units = {p: CC._units_of(src)[0] for p, src in srcs.items()}
    every = {u for us in units.values() for u in us}
    labels = [label for src in srcs.values() for label in re.findall(r'^ +\("([^"]+)", lambda', src, re.M)]
    assert {f"_model_roots:{label}" for label in labels} <= every and len(labels) > 50
    assert {f"TEMPLATE_PARAMS:{tid}" for tid in TG.template_params()} <= every
    assert {"_core_roots", "_boolean_roots", "_pouw_roots", "_kernel_roots", "_vocabulary_bindings", "common"} <= every
    assert {"_kernel_roots", "_vocabulary_bindings", "common"} <= set(units[CC.TARGETS])
    src = srcs[CC.TARGETS]
    assert CC._units_of(src.replace("\ndef catalog(", "\n# a comment\ndef catalog(", 1))[0] == units[CC.TARGETS]
    b1 = SC.path_of("verity.ml.definitions.registry.b1", "py")
    edited = CC._units_of(srcs[b1].replace('("ResidualAdd_v1", lambda: bind(b1.ResidualAdd, N=16))',
                                           '("ResidualAdd_v1", lambda: bind(b1.ResidualAdd, N=32))', 1))[0]
    assert {u for u in units[b1] if edited.get(u) != units[b1][u]} == {"_model_roots:ResidualAdd_v1"}


#: `targets.py`'s own binding units: what no one module owns (every kernel's and vocabulary kind's self-check bindings), its
#: module body, and `_root_bindings`, a helper whose name reads as a binding's
TARGETS_UNITS = {"_kernel_roots", "_vocabulary_bindings", "_vocabulary_roots", "_root_bindings", "common"}


def test_every_binding_and_pin_is_in_the_module_it_binds():
    """`targets.py` binds only what no one module owns, and every pin is in the file of the module that owns it, so a binding
    or pin added in the wrong place fails here: a Definition's goes in its module's sidecar (`circuit_check.sidecars`)."""
    from circuit_check import cache as CC
    from circuit_check import checks as CK
    from circuit_check import sidecars as SC
    from circuit_check import targets as TG

    stray = sorted(set(CC._units_of(CC.TARGETS.read_text())[0]) - TARGETS_UNITS)
    assert not stray, f"targets.py binds {stray}: move each to the sidecar of the module that defines what it binds"
    TG.load_registries()
    misplaced = []
    for kind, pins in CK.load_pins().items():
        for key in pins:
            mod = SC.pin_owner(kind, key)
            want = (SC.path_of(mod, "json") if mod else None) or SC.shared_pin(kind, key)
            if (have := SC.pin_file(kind, key)) != want:
                misplaced.append(f"{kind} {key}: in {SC.rel(have)}, owned by {SC.rel(want)}")
    assert not misplaced, misplaced


OWNERS = """
import json
from circuit_check import checks as CK, sidecars as SC
from verity.ml.circuits.defs import REGISTRY, bind

ON_DEMAND = [("definitions", "GatherBf16x1024u10_v1"), ("definitions", "GatherBf16x64_v2"), ("pieces", "GatherBf16x1_v1"),
             ("pieces", "BitAtx32_v1")]
keys = sorted({*((kind, key) for kind, pins in CK.load_pins().items() for key in pins), *ON_DEMAND})
owners = lambda: {f"{kind} {key}": SC.pin_owner(kind, key) for kind, key in keys}
before = owners()
from verity.ml.definitions.registry import boolean_gather, prims, sampling
_ = bind(boolean_gather.gather_bf16(1100)).body
boolean_gather.gather_bf16(64), prims.gather_bf16(1), sampling.bit_at(32)
print(json.dumps({"before": before, "after": owners(), "built": [key for _, key in ON_DEMAND if key in REGISTRY.defs]}))
"""


def test_a_pins_owner_is_what_its_key_names_not_what_the_process_built():
    """`sidecars.pin_owner` asked in a fresh process, then again once the on-demand Definitions are built (an earlier test in the
    process, or another target's check, builds them): the same owner for every pin, so a pin's file is a function of the tree.
    An on-demand primitive is its factory's (`prims`, `sampling`), not its caller's; a level of a gather's tree, which no key
    makes alone, has none."""
    r = subprocess.run([sys.executable, "-c", OWNERS], capture_output=True, text=True, cwd=REPO, check=True)
    got = json.loads(r.stdout)
    assert len(got["built"]) == 4, got["built"]
    assert got["before"] == got["after"], {k: (v, got["after"][k]) for k, v in got["before"].items() if got["after"][k] != v}
    assert got["before"]["pieces GatherBf16x1_v1"] == "verity.ml.definitions.registry.prims"
    assert got["before"]["pieces BitAtx32_v1"] == "verity.ml.definitions.registry.sampling"
    assert got["before"]["definitions GatherBf16x1024u10_v1"] is None


def test_a_run_of_imports_is_held_by_the_names_it_binds_not_their_order():
    """`cache._commuted`: a run of consecutive imports ordered otherwise digests alike, with its sites; two that bind one name,
    a statement between two, or an ``import *`` among them keep their order."""
    import ast

    from circuit_check import cache as CC

    def norm(src: str) -> tuple:
        return CC._normalized(ast.parse(src))[:2]

    assert norm("from a import x, y\nimport b.c\nfrom d import z as w\n") == \
        norm("from d import z as w\nimport b.c\nfrom a import y\nfrom a import x\n")
    assert norm("from a import f\nfrom b import f\n") != norm("from b import f\nfrom a import f\n")
    assert norm("from a import x\nX = 1\nfrom b import y\n") != norm("from b import y\nX = 1\nfrom a import x\n")
    assert norm("from a import *\nfrom b import y\n") != norm("from b import y\nfrom a import *\n")


def test_each_enumerator_only_imports_what_it_lists():
    """`cache._ENUMERATORS` (and `targets.load_registries`) are held by the modules a package lists rather than by its
    listing, on a claim about their bodies: each name a ``pkgutil`` listing yields is only compared, imported
    (``importlib.import_module``) or written into an error."""
    import ast
    import importlib
    import inspect
    import textwrap

    from circuit_check import cache as CC

    for mod, qual in sorted(CC._ENUMERATORS | {("circuit_check.targets", "load_registries")}):
        fn = importlib.import_module(mod)
        for part in qual.split("."):
            fn = getattr(fn, part)
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and "iter_modules" in ast.unparse(n.iter)]
        assert loops, (mod, qual)
        for loop in loops:
            allowed: set[int] = set()
            for n in ast.walk(loop):
                if isinstance(n, ast.Call) and ast.unparse(n.func) in ("importlib.import_module", "errors.append"):
                    allowed |= {id(x) for a in n.args for x in ast.walk(a)}
                elif isinstance(n, ast.Compare):
                    allowed |= {id(x) for x in ast.walk(n)}
            uses = [n for s in loop.body for n in ast.walk(s) if isinstance(n, ast.Name) and n.id == loop.target.id]
            assert uses and all(id(n) in allowed for n in uses), (mod, qual, ast.unparse(loop))


def test_the_task_list_is_reused_only_while_nothing_its_build_could_depend_on_changed(tmp_path):
    """A circuit-check that reuses every target would still spend most of its time building the catalog to find them: the
    task list is reused whole when every target is as recorded.  An edit no target read (a registry module none resolves, a new
    binding entry) still rebuilds the list, which it may change, and every target is reused as before."""
    root = _tree(tmp_path)
    assert "task_list" not in _run(root)["stats"]
    again = _run(root)
    assert again["hits"] == ["SynOther_v1", "SynTop_v1"] and again["stats"]["task_list"]
    _edit(root, "syn/idle.py", "B.call(SynAdd, a, a)", "B.call(SynAdd, b, b)")
    out = _run(root)
    assert out["hits"] == ["SynOther_v1", "SynTop_v1"] and "task_list" not in out["stats"]
    assert _run(root)["stats"]["task_list"]
    _edit(root, "targets.py", '("Spare_v1", lambda: "Spare_v1"),', '("Spare_v1", lambda: "Spare_v1"),\n        ("New_v1", lambda: 0),')
    out = _run(root)
    assert out["hits"] == ["SynOther_v1", "SynTop_v1"] and "task_list" not in out["stats"]
    _edit(root, "syn/sub.py", "B.call(SynAdd, a, b)", "B.call(SynAdd, b, a)")
    out = _run(root)
    assert out["hits"] == ["SynOther_v1"] and "task_list" not in out["stats"] and out["stats"]["stored"] == 1
    assert _run(root)["stats"]["task_list"]


def _named(target: str) -> str:
    """The harness with ``target`` named on the command line (`cli._tasks`' last branch, resolved by `targets.definition`), not
    found by ``--all``."""
    return (HARNESS.replace("TG.load_registries = lambda: None",
                            "TG.load_registries = lambda: None\n"
                            "from verity.ml.circuits.defs import REGISTRY\n"
                            "TG.definition = lambda spec: REGISTRY[spec].bind()")
            .replace("cli.run([], CK.Options(), everything=True,", f"cli.run([{target!r}], CK.Options(),"))


def test_a_target_named_on_the_command_line_is_keyed_by_its_descriptor_and_holds_its_modules(tmp_path):
    """A Definition named on the command line is not built inside its check, where its key would hold no descriptor and its
    entry only what that check's recording noted, so it would be reused after a module it reaches changed.  Built with the
    task list, it is a root as a sidecar's target is: its key holds its descriptor, its
    entry the modules it reaches, and an edit to one of them checks it again."""
    root = _tree(tmp_path)
    assert _run(root, _named("SynTop_v1"))["hits"] == []
    assert _run(root, _named("SynTop_v1"))["hits"] == ["SynTop_v1"]
    e = _entry(root, "SynTop_v1")
    assert "descriptor" in e["key"]["request"] and {"syn.top", "syn.sub", "syn.prims"} <= set(e["reads"]["modules"])
    _edit(root, "syn/top.py", "operands(a, b)", "operands(a, a)")
    out = _run(root, _named("SynTop_v1"))
    assert out["hits"] == [] and out["stats"]["stored"] == 1
    assert _values(root, "SynTop_v1") == [[6], [3 + 0x0105]]


def test_a_target_named_on_the_command_line_reruns_when_its_descriptor_changes_with_no_module_it_holds(tmp_path):
    """SynKn's body reads `syn.knob.N`, which another binding sets from `syn.ksize`: no module SynKn holds changes when
    `syn.ksize` does, and named on the command line it reruns because its key holds its descriptor."""
    root = _tree(tmp_path)
    _bindings(root, '(("_set_roots",), setknob)', '(("_knobbed_roots",), knobbed)', families=("SynKn_v1",))
    assert _run(root, _named("SynKn_v1"))["hits"] == []
    assert "syn.ksize" not in _entry(root, "SynKn_v1")["reads"]["modules"]
    _edit(root, "syn/ksize.py", "K = 1", "K = 2")
    out = _run(root, _named("SynKn_v1"))
    assert out["hits"] == [] and out["stats"]["stored"] == 1
    assert _values(root, "SynKn_v1") == [[3 + 2 * 0x0105], [3 + 3 * 0x0105]]


def _shimmed(harness: str) -> str:
    """``harness`` with the synthetic package imported through a link outside the tree (``shim/syn -> tree/syn``) first on the
    import path, as a remote run puts its shim (``pyshim/<key>/verity -> <tree>/verity``)."""
    return harness.replace("sys.path.insert(0, str(root))",
                           "sys.path.insert(0, str(root))\nsys.path.insert(0, str(root.parent / 'shim'))")


@pytest.mark.parametrize("named", [False, True], ids=["all", "named"])
def test_a_module_imported_through_a_link_into_the_tree_is_held(tmp_path, named):
    """In a remote run every module of `verity` has its file in the shim, outside the root, so a file through a link into
    the tree is the tree's (`cache._LINKS`): else no entry would hold one, and an edit to a primitive's evaluator, which
    changes no descriptor, would be reused."""
    root, shim = _tree(tmp_path / "tree"), tmp_path / "shim"
    shim.mkdir()
    (shim / "syn").symlink_to(root / "syn", target_is_directory=True)
    harness = _shimmed(_named("SynTop_v1") if named else HARNESS)
    targets = ["SynTop_v1"] if named else ["SynOther_v1", "SynTop_v1"]
    assert _run(root, harness)["hits"] == []
    assert _run(root, harness)["hits"] == targets
    assert {"syn.top", "syn.sub", "syn.prims"} <= set(_entry(root, "SynTop_v1")["reads"]["modules"])
    _edit(root, "syn/prims.py", "(a + b) & 0xFFFF", "(a + b + 1) & 0xFFFF")
    out = _run(root, harness)
    assert out["hits"] == [] and out["stats"]["stored"] == len(targets)
    assert _values(root, "SynTop_v1") == [[3 + 0x0105], [3 + 0x0105 + 1]]


#: the harness with each target checked in a process of its own, forked from a zygote that is itself forked from the harness's
#: process once the task list is built (`cli.Zygote` ``fork``, as `circuit-check` runs), so it holds this process's catalog
FORKED = HARNESS.replace("log=None, cache=", "log=None, jobs=2, isolate=True, fork=True, cache=")


def test_a_zygote_forked_once_the_task_list_is_built_records_what_a_check_here_does(tmp_path):
    """A forkserver zygote would build the catalog a second time, after circuit-check's process built it for the keys.  Forked
    from the process that built it, the zygote builds nothing:
    its children check the harness's own task list, which a forkserver zygote would build again from the real catalog.  A
    target's process records what checking it first in this process records (less the probes this process's lookups made of
    the cache's git-ignored directories, which are always absent); its entry is reused, and an edit reruns only what it reaches."""
    here, forked = _tree(tmp_path / "here"), _tree(tmp_path / "forked")
    assert _run(here)["hits"] == [] and _run(forked, FORKED)["hits"] == []

    def held(root: Path, target: str) -> dict:
        reads = _entry(root, target)["reads"]
        return {**reads, "probes": {p: v for p, v in reads["probes"].items() if not p.startswith("cache/")}}

    assert held(forked, "SynOther_v1") == held(here, "SynOther_v1")
    for target in ("SynOther_v1", "SynTop_v1"):
        assert _values(forked, target) == _values(here, target) and _entry(forked, target)["report"]["peak_gb"] > 0
    assert _run(forked, FORKED)["hits"] == ["SynOther_v1", "SynTop_v1"]
    _edit(forked, "syn/sub.py", "B.call(SynAdd, a, b)", "B.call(SynAdd, b, a)")
    out = _run(forked, FORKED)
    assert out["hits"] == ["SynOther_v1"] and out["stats"]["stored"] == 1


#: the harness with the checker a copy inside the tree (``checker/circuit_check``), as `src/circuit_check/` is inside the
#: repository, whose scheduler's peaks the process has read before the checks (`cli.heavy`, as a run of the parallel path does)
CHECKER = (HARNESS.replace("sys.path.insert(0, str(root))", "sys.path.insert(0, str(root))\nsys.path.insert(0, str(root / 'checker'))")
           .replace("rep = cli.run(", "cli.heavy()\nrep = cli.run(")
           .replace('"hits": hits}', '"hits": hits, "checker": cli.__file__}'))


@pytest.mark.slow
def test_the_scheduler_s_peaks_are_in_no_key(tmp_path):
    """`heavy.json` (`cli.HEAVY`) says only which targets may run beside which (`cli.run`'s scheduler), never what a check
    computes: an edit to it reuses every target and the task list (so adding a peak never makes
    `circuit-check --all` run cold), and so does an edit to the known failures (`cache.KNOWN`,
    which decide only the summary) or to the checker's docstrings and comments (`cache.version`), while an edit to the
    checker's code still reruns them all."""
    import shutil

    root = _tree(tmp_path)
    checker = root / "checker" / "circuit_check"
    shutil.copytree(REPO / "tools/verity/circuit_check/src/circuit_check", checker,
                    ignore=shutil.ignore_patterns("__pycache__", "heavy.json"))
    (checker / "heavy.json").write_text(json.dumps({"SynOther_v1": 20.0}))
    assert _run(root, CHECKER)["checker"] == str(checker / "cli.py")
    (checker / "heavy.json").write_text(json.dumps({"SynOther_v1": 20.0, "SynTop_v1": 161.8}))
    out = _run(root, CHECKER)
    assert out["hits"] == ["SynOther_v1", "SynTop_v1"] and out["stats"]["task_list"]
    for p in (checker / "known.py", *sorted((checker / "known").rglob("*.toml"))):
        p.write_text(p.read_text() + "\n# another known failure\n")
    out = _run(root, CHECKER)
    assert out["hits"] == ["SynOther_v1", "SynTop_v1"] and out["stats"]["task_list"]
    checks = checker / "checks.py"
    checks.write_text(checks.read_text().replace('"""The checks,', '"""Reworded: the checks,', 1) + "\n# a comment\n")
    out = _run(root, CHECKER)
    assert out["hits"] == ["SynOther_v1", "SynTop_v1"] and out["stats"]["task_list"]
    checks.write_text(checks.read_text() + "\n_ANOTHER_VERSION_OF_THE_CHECKER = 1\n")
    assert _run(root, CHECKER)["hits"] == []


def test_the_checker_s_version_is_its_code_not_its_prose(tmp_path, monkeypatch):
    """`cache.version`: an edit to a docstring, a comment or the layout of the checker's modules keeps its version, so it
    reruns no target, while an edit to their code, or to a file of it that is no module, changes it."""
    from circuit_check import cache as CC

    monkeypatch.setattr(CC, "_HERE", tmp_path)
    mod, data = tmp_path / "checks.py", tmp_path / "data.json"
    src = '"""The checks."""\n\n\ndef f(x):\n    """Twice ``x``."""\n    return 2 * x  # doubled\n'
    mod.write_text(src)
    data.write_text("{}")
    here = CC.version()
    mod.write_text('"""The checks, reworded."""\n# a new comment\n\ndef f(x):\n    """``x`` doubled."""\n\n    return 2 * x\n')
    assert CC.version() == here
    mod.write_text(src.replace("2 * x", "3 * x"))
    assert CC.version() != here
    mod.write_text(src)
    assert CC.version() == here
    data.write_text('{"a": 1}')
    assert CC.version() != here
