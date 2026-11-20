"""The cache's per-target keys (`circuit_check.cache`, `circuit-check-cache/v4`), on a synthetic tree: a registry of small
Definitions, a catalog (`targets.py`) and shared pins of its own (``pins/``, a file per pin), checked through `cli.run`'s `--all` path with the catalog's
functions standing in for the real ones (the real catalog takes a minute to build).  A target's entry holds what its own check
read or resolved, and nothing another target or the task list read."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from circuit_check import cache as CC
from circuit_check import entries as E

REPO = Path(__file__).resolve().parents[4]

FILES = {
    "syn/__init__.py": "",
    "syn/prims.py": """
from verity.circuits import Value, primitive

V = Value(16)


@primitive("SynAdd", 1, [("a", V), ("b", V)], V)
def SynAdd(a, b):
    return (a + b) & 0xFFFF
""",
    # the sub-Definition SynTop resolves: imported only while SynTop's body is built, so no import statement names it
    "syn/sub.py": """
from verity.circuits import composite

from syn.prims import SynAdd, V


@composite("SynSub", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynSub(B, S, a, b):
    return B.call(SynAdd, a, b)
""",
    "syn/top.py": """
from verity.circuits import composite

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
from verity.circuits import composite

from syn.prims import SynAdd, V


@composite("SynOther", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynOther(B, S, a, b):
    return B.call(SynAdd, b, a)
""",
    # registered with the rest, resolved by no target
    "syn/idle.py": """
from verity.circuits import composite

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

    from verity.circuits.evaluation.reference import evaluate

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
from verity.circuits.defs import bind


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
from verity.circuits import composite

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
from verity.circuits import composite

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
from verity.circuits import primitive

from syn.consts import MASK
from syn.prims import V


@primitive("SynXor", 1, [("a", V), ("b", V)], V)
def SynXor(a, b):
    return (a ^ b) & MASK
""",
    "syn/xdef.py": """
from verity.circuits import composite

from syn.xkern import SynXor
from syn.prims import V


@composite("SynX", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynX(B, S, a, b):
    return B.call(SynXor, a, b)
""",
    # a Definition whose body reads module state another binding sets from a module only that binding read
    "syn/knob.py": "N = [1]\n",
    "syn/kn.py": """
from verity.circuits import composite

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


def _tree(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    """A git repository at ``tmp_path`` holding `FILES` and ``extra`` (by path; ``extra`` wins)."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for rel, text in {**FILES, **(extra or {})}.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text.lstrip("\n"))
    return tmp_path


def _run(root: Path, harness: str = HARNESS, cwd: Path = REPO, env: dict[str, str] | None = None) -> dict:
    # no bytecode: an edit that keeps a file's size within the second its bytecode was written would run the old code
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
    r = subprocess.run([sys.executable, "-c", harness, str(root), str(REPO)], capture_output=True, text=True, cwd=cwd, env=env)
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
    """The one stored entry of ``target``, its reads as `cache.store` was handed them (`cache.read`)."""
    (e,) = [CC.read(p) for p in (root / "cache").glob("*/*/*.json")
            if json.loads(p.read_text())["key"]["request"]["target"][1] == target]
    return e


def test_a_probe_under_a_path_that_is_no_directory_is_declared_as_that_path(tmp_path):
    """`cache.probed`: nothing is under a path that is no directory, so a probe under one is declared as the first such path
    (its kind holds theirs), and one whose every ancestor is a directory as itself.  In a checkout the directories are those
    git has (`visible`), in a tree with no git the filesystem's; once a directory is there, the probes under it are their own."""
    from circuit_check import cache as CC

    probes = ["pins/digests/X_v1.json", "pins/digests/Y_v1.json", "pins/definitions/X_v1.json", "a.py/b", "ast/__init__.py",
              "ast.py", "top"]
    want = {"pins/digests", "pins/definitions/X_v1.json", "a.py", "ast", "ast.py", "top"}
    plain = tmp_path / "plain"
    (plain / "pins" / "definitions").mkdir(parents=True)
    (plain / "a.py").write_text("")
    assert CC.probed(plain, probes) == want
    assert CC.probed(_tree(tmp_path / "git", {"pins/definitions/Z_v1.json": "{}"}), probes) == want
    (plain / "pins" / "digests").mkdir()
    assert CC.probed(plain, probes) == want - {"pins/digests"} | {"pins/digests/X_v1.json", "pins/digests/Y_v1.json"}


#: SynOther's binding and pin moved out of `targets.py` and ``pins/`` into its module's sidecar
SIDECAR = """
def _model_roots():
    from verity.circuits.defs import bind
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


def test_a_new_pin_goes_to_the_sidecar_of_its_definitions_module(tmp_path):
    """`sidecars.pin_file`: a pin is written where it is, and a new one to the pins file of the module that defines its
    Definition, a module without a sidecar getting one beside it.  The owners are found by `targets.load_registries`, which reads
    the tree's catalog from the sidecar that hands it (`targets.HOOKS`): a stand-in that loads nothing more."""
    root = _tree(tmp_path)
    _sidecar(root)
    (root / "syn/catalog.circuit-check.py").write_text("def enumerators():\n    return set()\n\n\ndef load_catalog():\n    pass\n")
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


#: a check that looks SynLate up in the registry (absent until a sidecar makes the registries import `syn.late.mod`), finds the
#: Boolean Definition of SynAdd's word as `verity_flock.ir_lower._boolean_build` does (`cache._WORD_ENUMERATORS`), and, for
#: SynOther, counts every registered Definition
LOOKUPS = {
    "syn/lookup.py": """
def boolean_of(key):
    from verity.circuits.defs import REGISTRY
    for d in list(REGISTRY.defs.values()):
        if callable(getattr(d, "build", None)) and getattr(getattr(d, "word", None), "id", None) == key:
            return d.id
    return None


def check_definition(fn, opts, *, memo, pins, as_call=None, **kw):
    from verity.circuits.defs import REGISTRY
    got = {"failures": [], "warnings": [], "gates": fn.gates, "late": "SynLate_v1" in REGISTRY, "bits": boolean_of("SynAdd_v1")}
    if fn.id == "SynOther_v1":
        got["n"] = len(REGISTRY.defs)
    return got
""",
    "syn/late/__init__.py": "",
    "syn/late/mod.py": """
from verity.circuits import composite

from syn.prims import SynAdd, V


@composite("SynLate", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynLate(B, S, a, b):
    return B.call(SynAdd, a, b)
""",
    "syn/late/bits.py": """
from types import SimpleNamespace

from verity.circuits.defs import REGISTRY

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
    (``*``), which holds every sidecar as the key did: the pins file beside `syn.extra` (no Definition, and it imports none)
    reruns it, and not SynTop."""
    assert LOOKUP != HARNESS
    root = _tree(tmp_path)
    for rel, text in LOOKUPS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text.lstrip("\n"))
    _run(root, LOOKUP)
    top, other = _entry(root, "SynTop_v1")["reads"]["registrars"], _entry(root, "SynOther_v1")["reads"]["registrars"]
    assert {"absent:SynLate_v1", "word:SynAdd_v1"} <= set(top) and "*" not in top and "*" in other
    (root / "syn/extra.circuit-check.json").write_text("{}")
    assert _run(root, LOOKUP)["hits"] == ["SynTop_v1"]
    (root / "syn/late/mod.circuit-check.json").write_text("{}")
    assert _run(root, LOOKUP)["hits"] == []
    assert sorted(r["late"] for r in _reports(root, "SynTop_v1")) == [False, True]
    (root / "syn/late/bits.circuit-check.json").write_text("{}")
    assert _run(root, LOOKUP)["hits"] == []
    assert sorted(str(r["bits"]) for r in _reports(root, "SynTop_v1")) == ["None", "None", "SynAddBits_v1"]


#: a module whose import imports `syn.late.mod` (`LOOKUPS`): `syn.late.wrap` imports it, and `syn.late.deep.leaf`'s package does,
#: before any code of `syn.late.deep.leaf` runs
WRAPPERS = {
    "syn.late.wrap": {"syn/late/wrap.py": "from syn.late import mod  # noqa: F401\n"},
    "syn.late.deep.leaf": {"syn/late/deep/__init__.py": "from syn.late import mod  # noqa: F401\n", "syn/late/deep/leaf.py": ""},
}


@pytest.mark.parametrize("wrapper", sorted(WRAPPERS))
def test_removing_the_sidecar_of_a_module_whose_import_registered_what_a_check_found_reruns_it(tmp_path, wrapper):
    """The registries register SynLate only for a sidecar beside a module whose import imports `syn.late.mod` (`WRAPPERS`), not
    beside `syn.late.mod` itself.  SynTop's check finds SynLate; removing the sidecar reruns SynTop, whose check now looks it up
    absent, though no module SynTop holds changed.  (`syn.idle`'s pins file, there throughout, makes the registries import
    `syn.prims` first for it, so SynAdd, which SynTop also reached, is registered for the same sidecar either way.)"""
    root = _tree(tmp_path)
    for rel, text in {**LOOKUPS, **WRAPPERS[wrapper]}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text.lstrip("\n"))
    (root / "syn/idle.circuit-check.json").write_text("{}")
    side = root / f"{wrapper.replace('.', '/')}.circuit-check.json"
    side.write_text("{}")
    _run(root, LOOKUP)
    assert _run(root, LOOKUP)["hits"] == ["SynOther_v1", "SynTop_v1"]
    assert "SynLate_v1" in _entry(root, "SynTop_v1")["reads"]["registrars"]
    side.unlink()
    out = _run(root, LOOKUP)
    assert out["hits"] == [] and out["stats"]["stored"] == 2
    assert sorted(r["late"] for r in _reports(root, "SynTop_v1")) == [False, True]


def test_a_sidecar_whose_module_is_gone_is_an_error(tmp_path, monkeypatch):
    import pytest

    from circuit_check import sidecars as SC

    (tmp_path / "synorphan").mkdir()
    (tmp_path / "synorphan/__init__.py").write_text("")
    (tmp_path / "synorphan/kept.py").write_text("")
    (tmp_path / "synorphan/kept.circuit-check.json").write_text("{}")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(SC, "PACKAGES", ("synorphan",))
    monkeypatch.setattr(SC, "TREES", ())
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
    monkeypatch.setattr(SC, "TREES", ())
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
    monkeypatch.setattr(SC, "TREES", ())
    SC.discover.cache_clear()
    try:
        assert list(SC.discover()) == ["synin2.rec"]
    finally:
        SC.discover.cache_clear()


def test_a_package_of_another_distribution_registers_itself_by_a_sidecar_of_its_own_module(tmp_path, monkeypatch):
    """A package in one of `sidecars.TREES` (``<tree>/<stack>/<package>``) is searched once it has a sidecar of its own module,
    and its sidecars are named from its own top; one without is not searched, and `sidecars.PACKAGES` names neither."""
    from circuit_check import sidecars as SC

    for pkg in ("synreg", "synunreg"):
        sub = tmp_path / "stacks/s" / pkg / "sub"
        sub.mkdir(parents=True)
        for d in (sub.parent, sub):
            (d / "__init__.py").write_text("")
        (sub / "rec.py").write_text("")
        (sub / "rec.circuit-check.py").write_text("")
    own = tmp_path / "stacks/s/synreg/__init__.circuit-check.py"
    own.write_text("")
    monkeypatch.syspath_prepend(str(tmp_path / "stacks/s"))
    monkeypatch.setattr(SC, "REPO", tmp_path)
    monkeypatch.setattr(SC, "PACKAGES", ())
    monkeypatch.setattr(SC, "TREES", ("stacks",))
    SC.discover.cache_clear()
    try:
        assert list(SC.discover()) == ["synreg", "synreg.sub.rec"] and SC.packages() == {"synreg"}
        assert SC.path_of("synreg", "py") == own.resolve()
        assert SC.path_of("synreg.sub.rec", "json") == (own.parent / "sub/rec.circuit-check.json").resolve()
        assert SC.path_of("synunreg.sub.rec", "py") is None
    finally:
        SC.discover.cache_clear()


def test_every_package_that_defines_a_definition_is_one_sidecars_are_found_in():
    """`sidecars.packages` holds the top-level package of every module that defines a registered Definition: a sidecar in any
    other package is never read, so its bindings would leave the task list without an error."""
    from circuit_check import sidecars as SC
    from circuit_check import targets as TG
    from verity.circuits.defs import REGISTRY

    TG.load_registries()
    tops = {m.partition(".")[0] for d in REGISTRY.defs.values() if TG._authored(d) and (m := SC.owner(d))}
    assert tops <= SC.packages(), (f"{sorted(tops - SC.packages())} define Definitions where circuit-check finds no sidecar: "
                                   f"register each by a sidecar of its own module (`<package>/__init__{SC.SUFFIX}.py`, "
                                   f"`sidecars.TREES`)")


#: body reads under nested recordings: which of SynTop and the SynSub its body calls each recording noted
BODY_READS = """
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from circuit_check import cache as CC
from verity.circuits.defs import bind

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
    assert {"_kernel_roots", "common"} <= set(units[CC.TARGETS])
    src = srcs[CC.TARGETS]
    assert CC._units_of(src.replace("\ndef catalog(", "\n# a comment\ndef catalog(", 1))[0] == units[CC.TARGETS]
    b1 = SC.path_of("verity_vllm.definitions.registry.b1", "py")
    edited = CC._units_of(srcs[b1].replace('("ResidualAdd_v1", lambda: bind(b1.ResidualAdd, N=16))',
                                           '("ResidualAdd_v1", lambda: bind(b1.ResidualAdd, N=32))', 1))[0]
    assert {u for u in units[b1] if edited.get(u) != units[b1][u]} == {"_model_roots:ResidualAdd_v1"}


#: `targets.py`'s own binding units: what no one module owns (every kernel's self-check bindings), its module body, and
#: `_root_bindings`, a helper whose name reads as a binding's
TARGETS_UNITS = {"_kernel_roots", "_root_bindings", "common"}


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
from verity.circuits.defs import REGISTRY, bind

ON_DEMAND = [("definitions", "GatherBf16x1024u10_v1"), ("definitions", "GatherBf16x64_v2"), ("pieces", "GatherBf16x1_v1"),
             ("pieces", "BitAtx32_v1")]
keys = sorted({*((kind, key) for kind, pins in CK.load_pins().items() for key in pins), *ON_DEMAND})
owners = lambda: {f"{kind} {key}": SC.pin_owner(kind, key) for kind, key in keys}
before = owners()
from verity_vllm.definitions.registry import boolean_gather, prims, sampling
_ = bind(boolean_gather.gather_bf16(1100)).body
boolean_gather.gather_bf16(64), prims.gather_bf16(1), sampling.bit_at(32)
print(json.dumps({"before": before, "after": owners(), "built": [key for _, key in ON_DEMAND if key in REGISTRY.defs]}))
"""


def test_a_pins_owner_is_what_its_key_names_not_what_the_process_built():
    """`sidecars.pin_owner` asked in a fresh process, then again once the on-demand Definitions are built (an earlier test in the
    process, or another target's check, builds them): the same owner for every pin, so a pin's file is a function of the tree.
    An on-demand primitive is its factory's (`prims`, `sampling`), not its caller's; a level of a gather's tree, which no key
    makes alone, is the module's whose binding file holds its family (`sidecars.body_family`)."""
    r = subprocess.run([sys.executable, "-c", OWNERS], capture_output=True, text=True, cwd=REPO, check=True)
    got = json.loads(r.stdout)
    assert len(got["built"]) == 4, got["built"]
    assert got["before"] == got["after"], {k: (v, got["after"][k]) for k, v in got["before"].items() if got["after"][k] != v}
    assert got["before"]["pieces GatherBf16x1_v1"] == "verity_vllm.definitions.registry.prims"
    assert got["before"]["pieces BitAtx32_v1"] == "verity_vllm.definitions.registry.sampling"
    assert got["before"]["definitions GatherBf16x1024u10_v1"] == "verity_vllm.definitions.registry.boolean_gather"


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


def _builds_a_name(v) -> bool:
    """`v` only builds a string from names: an f-string, `+`, a literal's `join`/`format`, or a conditional of those."""
    import ast

    if isinstance(v, (ast.Constant, ast.Name)):
        return True
    if isinstance(v, ast.Attribute):
        return isinstance(v.value, ast.Name)
    if isinstance(v, ast.JoinedStr):
        return all(isinstance(x, ast.Constant) or (isinstance(x, ast.FormattedValue) and _builds_a_name(x.value))
                   for x in v.values)
    if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add):
        return _builds_a_name(v.left) and _builds_a_name(v.right)
    if isinstance(v, ast.IfExp):
        return isinstance(v.test, ast.Compare) and _builds_a_name(v.body) and _builds_a_name(v.orelse)
    if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr in ("join", "format"):
        return isinstance(v.func.value, ast.Constant) and not v.keywords and all(_builds_a_name(a) for a in v.args)
    return False


def _enumerator_misuses(tree) -> list[str]:
    """Each `pkgutil` listing loop in `tree` whose names (or names only built from them) are used other than compared,
    imported (``importlib.import_module``), written into an error, or queued on a work list (``todo``) whose entries are
    popped and only imported; and the listing loops' count, so a body with none is caught."""
    import ast

    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and "iter_modules" in ast.unparse(n.iter)]
    bad = [] if loops else ["no pkgutil listing loop"]
    for loop in loops:
        allowed: set[int] = set()
        tracked = {loop.target.id}
        for n in ast.walk(loop):
            if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                    and any(isinstance(x, ast.Name) and x.id in tracked for x in ast.walk(n.value))
                    and _builds_a_name(n.value)):
                tracked.add(n.targets[0].id)
                allowed |= {id(x) for x in ast.walk(n.value)}
        for n in ast.walk(loop):
            if isinstance(n, ast.Call) and ast.unparse(n.func) in ("importlib.import_module", "errors.append", "todo.append"):
                allowed |= {id(x) for a in n.args for x in ast.walk(a)}
            elif isinstance(n, ast.Compare):
                allowed |= {id(x) for x in ast.walk(n)}
        uses = [n for s in loop.body for n in ast.walk(s)
                if isinstance(n, ast.Name) and n.id in tracked and isinstance(n.ctx, ast.Load)]
        if not uses or not all(id(n) in allowed for n in uses):
            bad.append(ast.unparse(loop))
    if "todo.append" in ast.unparse(tree):
        pops = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and ast.unparse(n.value) == "todo.pop()"]
        popped = {ast.unparse(t.elts[0] if isinstance(t, ast.Tuple) else t) for p in pops for t in p.targets}
        imported = {ast.unparse(c.args[0]) for c in ast.walk(tree)
                    if isinstance(c, ast.Call) and ast.unparse(c.func) == "importlib.import_module"}
        if not pops or not popped <= imported:
            bad.append(f"work list: popped {sorted(popped)}, imported {sorted(imported)}")
    return bad


def test_each_enumerator_only_imports_what_it_lists():
    """The enumerators the catalog's sidecar hands (`targets.HOOKS`, `cache.enumerators`), its ``load_catalog`` among them,
    are held by the modules a package lists rather than by its listing, on a claim about their bodies: each name a ``pkgutil``
    listing yields, and each name only built from one, is only compared, imported (``importlib.import_module``), written into
    an error, or queued on a work list (``todo``) whose entries are popped and only imported."""
    import ast
    import importlib
    import inspect
    import textwrap

    from circuit_check import targets as TG

    for mod, qual in sorted(TG._handed("enumerators")()):
        fn = importlib.import_module(mod)
        for part in qual.split("."):
            fn = getattr(fn, part)
        assert not _enumerator_misuses(ast.parse(textwrap.dedent(inspect.getsource(fn)))), (mod, qual)


def test_the_enumerator_rule_refuses_a_listed_name_used_for_anything_else():
    """The rule `test_each_enumerator_only_imports_what_it_lists` holds bodies to: a work list of names built from the
    listing passes, and a read keyed by a listed name, directly or through a name derived from it, fails."""
    import ast

    ok = """
def load(pkg):
    todo = [(pkg, True)]
    while todo:
        name, is_package = todo.pop()
        importlib.import_module(name)
        for m in pkgutil.iter_modules([]):
            child = f"{name}.{m.name}" if name == "x" else name + "." + m.name
            if m.name != "tests" and child != __name__:
                todo.append((child, m.ispkg))
"""
    assert not _enumerator_misuses(ast.parse(ok))
    for body in ("data = open(path_for(m.name)).read()",
                 "path = f'{m.name}.json'\n        data = open(path).read()",
                 "path = path_for(m.name)\n        importlib.import_module(path)"):
        src = f"""
def load(pkg):
    for m in pkgutil.iter_modules([]):
        {body}
        importlib.import_module(m.name)
"""
        assert _enumerator_misuses(ast.parse(src)), body
    stray = """
def load(pkg):
    todo = []
    for m in pkgutil.iter_modules([]):
        todo.append(m.name)
    for name in todo:
        open(name).read()
"""
    assert _enumerator_misuses(ast.parse(stray))


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


def test_the_checker_s_version_is_read_once_until_a_file_of_it_changes(tmp_path, monkeypatch):
    """`cache.version`, which `key` asks for once for each target, reads the checker's files once a process: again only when one
    of them or a directory of them has another stat (an edit, a new file), and on every call while one of them changed within
    `cache.RACY_NS` of the reading, when another write in the same timestamp tick could leave its stat as it was."""
    from circuit_check import cache as CC

    mod, code, read = tmp_path / "checks.py", CC._code, []
    mod.write_text("def f(x):\n    return 2 * x\n")
    now = [mod.stat().st_ctime_ns + 86400 * 10**9]
    monkeypatch.setattr(CC, "_HERE", tmp_path)
    monkeypatch.setattr(CC, "_VERSION", [])
    monkeypatch.setattr(CC, "_clock", lambda: now[0])
    monkeypatch.setattr(CC, "_code", lambda p: read.append(p.name) or code(p))
    here = CC.version()
    assert CC.version() == here and read == ["checks.py"]
    mod.write_text("def f(x):\n    return 2 * x + 1\n")
    edited = CC.version()
    assert edited != here and CC.version() == edited and read == ["checks.py"] * 2
    (tmp_path / "data.json").write_text("{}")
    # a tick of its own for the new entry: in the first write's tick the directory's stat could stay as it was (the race
    # RACY_NS covers, which this test's clock, a day on, leaves out)
    os.utime(tmp_path, ns=(0, 0))
    added = CC.version()
    assert added != edited and CC.version() == added and read == ["checks.py"] * 3 + ["data.json"]
    mod.write_text("def f(x):\n    return 3 * x\n")
    now[0] = mod.stat().st_ctime_ns
    racy = CC.version()
    assert CC.version() == racy != added and read[4:] == ["checks.py", "data.json"] * 2
    now[0] += CC.RACY_NS + 1
    assert CC.version() == CC.version() == racy and read[8:] == ["checks.py", "data.json"]


def _walked(fn) -> str:
    """`cache.descriptor` as a walk of every body node and static, with no memo: what its memo must give."""
    import hashlib

    from verity.circuits import codec as C
    from verity.circuits.defs import FunctionDefinition, PrimitiveDefinition, SpecializedDefinition

    def statics(v):
        if isinstance(v, FunctionDefinition):
            yield v
        elif isinstance(v, (list, tuple, dict)):
            for x in v.values() if isinstance(v, dict) else v:
                yield from statics(x)

    parts, seen, stack = [], set(), [fn]
    while stack:
        f = stack.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        if isinstance(f, (PrimitiveDefinition, SpecializedDefinition)):
            parts.append([C.descriptor_id(f), hashlib.sha512(C.canonical_json(C.definition_entry(f))).hexdigest()])
        if isinstance(f, SpecializedDefinition):
            stack.extend(n.fn for n in f.body.nodes)
            stack.extend(statics(f.bindings))
    return hashlib.sha512(json.dumps(sorted(parts)).encode()).hexdigest()


def test_a_descriptor_is_its_walk_of_every_body_node_however_often_a_body_calls_one_definition():
    """`cache.descriptor` takes each specialization's distinct callees and statics once (`cache._NEXT`): every digest, first
    and again, is the one a walk of every node gives, for a body that calls one Definition from many nodes, a function
    static, and two specializations that share a callee."""
    from verity.circuits import Array, bind
    from verity.circuits.defs import CompositeDefinition
    from verity.circuits.ops.scalar import BIT, F32, F32Eq
    from verity.circuits.refs import tuple_of
    from verity.circuits.types import Tuple

    from circuit_check import cache as CC

    same = CompositeDefinition("DescSame", 1, (), lambda S: ((("a", F32), ("b", F32)), BIT),
                               lambda B, S, a, b: B.call(F32Eq, a, b), register=False)
    many = CompositeDefinition("DescMany", 1, ("N",), lambda S: ((("x", Array(S.N, F32)), ("y", Array(S.N, F32))),
                                                                  Tuple(*[BIT] * S.N)),
                               lambda B, S, x, y: tuple_of(*(B.call(bind(same), x[i], y[i]) for i in range(S.N))),
                               register=False)
    via = CompositeDefinition("DescVia", 1, ("OP",), lambda S: ((("a", F32), ("b", F32)), BIT),
                              lambda B, S, a, b: B.call(S.OP, a, b), register=False)
    roots = [bind(many, N=5), bind(many, N=2), bind(via, OP=bind(same)), bind(same), F32Eq]
    first = [CC.descriptor(fn) for fn in roots]
    assert first == [_walked(fn) for fn in roots] == [CC.descriptor(fn) for fn in roots]
    assert len(set(first)) == len(roots) and not any(d.startswith("error:") for d in first)
