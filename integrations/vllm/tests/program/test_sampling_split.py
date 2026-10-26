"""The sampling family's split into Definitions, twins and vocabulary changes no Definition.

* Every Definition id and canonical descriptor digest (`query.word.definition_digest`: the descriptor entry and its callees'
  digests) is the one `data/sampling_split_digests.json` pins, at every statics a circuit-check target or pin, a kept record or
  manifest, or a served vocabulary uses, and each of the family's Definitions is still constructed by the module that
  constructed it.  The pins were written on the unsplit tree; a change to a Definition of the
  family rewrites them (`python tests/program/test_sampling_split.py --write`, from `integrations/vllm`).
* The family's Definition modules import only the stdlib, numpy, `verity`, one another, and the two other families' modules
  they take Definitions from (`OTHER_DEFINITION_MODULES`), less the one listed exception: so they can move to `verity/ml/circuits/`, which
  imports only `verity`, with the Definitions they call.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

VLLM = Path(__file__).resolve().parents[2]
REPO = VLLM.parents[1]
REG = Path(importlib.util.find_spec("verity.ml.definitions.registry").origin).parent
FIXTURE = Path(__file__).resolve().parent / "data" / "sampling_split_digests.json"
TASKS = REPO / "tools" / "verity" / "circuit_check" / "tests" / "data" / "sampling_split_tasks.json"
PKG = "verity.ml.definitions.registry"
#: the catalog, where the layout move put core's Definitions and silicon models
CATALOG = "verity.ml.circuits"

DEFINITION_MODULES = ("sampling", "sampling_boolean", "gumbel_boolean", "topp_word_gates", "topp_keep_builder", "topp_words", "topp_geometry")
TWIN_MODULES = ("sampling_rows", "sampling_topp_difftest", "topp_split", "sampling_twins")
VOCAB_MODULES = ("sampling_vocab",)
#: the modules of the unsplit tree too, so the writer and the test find the same Definitions on either
FAMILY_MODULES = DEFINITION_MODULES + TWIN_MODULES + VOCAB_MODULES
OTHER_DEFINITION_MODULES = {
    "ref_prims": "F32GtStrict_v1 and F32Sub_v1, the v1 fp32 compare and subtract the selects and the top-p gates call",
    "b1": "ArgmaxStep_v1 and TokenSelect_v1, the word Definitions sampling_boolean's Boolean ones compute (word=)",
}
#: (module, enclosing function, imported module) -> why the import stays
EXCEPTIONS = {
    ("sampling", "topp_keep", f"{PKG}.topp_split"): "the evaluator of TopPMaskWordx{V}_v1 is the exact model of the split top-p "
                                                    "pipeline (topp_split.topp_keep_row, numpy and the C-JIT MUFU EX2 and div.full "
                                                    "tables): the keep word is that model's output",
}
#: the vocabulary sizes of the served rows, and the split counts at the largest one's single-request geometries
SERVED_V = (32000, 49152, 128256, 151936, 152064, 154880, 256000)
SERVED_VS = ((128256, 16), (128256, 32))
#: Boolean Definitions whose served sizes take seconds each to trace: pinned at one served V
ONE_SERVED_V = ("GumbelSelectF32_v2",)
#: the pins that take over 5 s to trace at circuit-check's statics: the lane and its rounds 8-9, the uniform and the logs, the noise
#: group, NvLogf and the select at a served V; every pin at a served (V, S) pair is slow too (`_params`)
SLOW_TRACE = ("GumbelLane_v1", "GumbelLaneTail_v1", "GumbelNoiseGroup_v1{G=1}", "NvLogf_v3", "GumbelSelectF32_v2{V=128256}")


# --- the family's Definitions ------------------------------------------------------------------------------------------------

def defining_module(fn) -> str | None:
    """The module whose code a Definition runs: a primitive's evaluator's, a composite's body's, a traced Boolean one's build's."""
    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition

    d = fn.defn if isinstance(fn, SpecializedDefinition) else fn
    f = d.evaluate if isinstance(d, PrimitiveDefinition) else getattr(d, "build", d.body) if isinstance(d, CompositeDefinition) else None
    return getattr(getattr(f, "func", f), "__module__", None)


def family_definitions() -> dict[str, object]:
    """Every Definition a family module constructs and binds to a global, by id (a family maker's, such as `BitAtx{W}_v1`, is
    registered on first use, so whether the registry holds one depends on what ran before)."""
    import importlib

    from verity.ml.circuits.defs import FunctionDefinition
    from verity.ml.definitions.registry.catalog import load

    load()
    mods = {f"{PKG}.{m}" for m in FAMILY_MODULES}
    out = {}
    for m in sorted(mods):
        if (REG / f"{m.rsplit('.', 1)[1]}.py").exists():
            for v in vars(importlib.import_module(m)).values():
                if isinstance(v, FunctionDefinition) and defining_module(v) in mods:
                    out.setdefault(v.id, v)
    return out


def _statics(body: str) -> dict:
    out = {}
    for part in filter(None, body.split(",")):
        k, v = part.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def resolve(key: str, defs: dict[str, object]):
    """The Definition a spec id names: the registry's binding (`query.call_scope.specialization`), else a family module's own."""
    from verity.ml.definitions.call_scope import specialization

    m = re.fullmatch(r"([^{]+)(?:\{(.*)\})?", key)
    fam, statics = m[1], _statics(m[2] or "")
    try:
        fn = specialization(fam, statics)
    except KeyError:
        fn = defs[fam]
        fn = fn.bind(**statics) if hasattr(fn, "bind") else fn
    if hasattr(fn, "complete") and not fn.complete:
        raise KeyError(f"{key}: statics unbound")
    return fn


def pin(key: str, defs: dict[str, object]) -> dict:
    from verity.ml.definitions.word import definition_digest

    fn = resolve(key, defs)
    return {"id": fn.id, "digest": definition_digest(fn)}


# --- the writer --------------------------------------------------------------------------------------------------------------

def spec_keys(defs: dict[str, object]) -> list[str]:
    """circuit-check's family targets and pins, the family ids a tracked JSON file writes with all their statics, and the
    family's V- and (V, S)-indexed Definitions at the served sizes."""
    tasks = json.loads(TASKS.read_text())
    keys = {t["id"] for t in tasks["tasks"] if t["kind"] == "definition"} | {k for p in tasks["pins"].values() for k in p}
    bases = sorted({k.split("{")[0] for k in keys} | set(defs), key=len, reverse=True)
    pat = re.compile(r"\b((?:" + "|".join(map(re.escape, bases)) + r")\{[^{}\"]*\})")
    files = subprocess.run(["git", "ls-files", "-z", "*.json"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split("\0")
    for f in filter(None, files):
        p = REPO / f
        if p.is_file() and p != FIXTURE:
            keys |= set(pat.findall(p.read_text(errors="replace")))
    for k, d in defs.items():
        statics = tuple(getattr(d, "statics", ()))
        if statics == ("V",):
            keys |= {f"{k}{{V={V}}}" for V in (SERVED_V[2:3] if k in ONE_SERVED_V else SERVED_V)}
        elif statics == ("V", "S"):
            keys |= {f"{k}{{V={V},S={S}}}" for V, S in SERVED_VS}
    out = []
    for k in sorted(keys):
        try:
            resolve(k, defs)
        except (KeyError, TypeError, ValueError):
            continue
        out.append(k)
    return out


def _load() -> dict:
    return json.loads(FIXTURE.read_text())


def _params():
    if not FIXTURE.exists():
        return []
    served = tuple(f"{{V={v},S={s}}}" for v, s in SERVED_VS)
    return [pytest.param(k, marks=pytest.mark.slow) if k in SLOW_TRACE or k.endswith(served) else k for k in _load()["pins"]]


@pytest.fixture(scope="module")
def defs():
    return family_definitions()


def test_each_definition_is_constructed_where_it_was(defs):
    assert {k: defining_module(d) for k, d in sorted(defs.items())} == _load()["modules"]


@pytest.mark.parametrize("key", _params())
def test_the_definition_id_and_digest_are_unchanged(key, defs):
    """The keys in `SLOW_TRACE` and those at a served (V, S) pair trace a Boolean Definition's body for over 5 s, so they are slow;
    the rest, served vocabularies included, stay quick."""
    assert pin(key, defs) == _load()["pins"][key]


def test_the_pins_cover_every_circuit_check_target_of_the_family():
    have = set(_load()["pins"])
    assert len(have) > 100 and {"GumbelTopPTokenSelect_v1{V=128256}", "TopPKeepWord_v1{V=16,S=4}", "GumbelSelectF32_v2{V=2}"} <= have


# --- the boundary --------------------------------------------------------------------------------------------------------------

def _is_stdlib(mod: str) -> bool:
    return mod.split(".")[0] in sys.stdlib_module_names or mod == "__future__"


def _imports(path: Path) -> list[tuple[str, int, str | None]]:
    """(module, line, enclosing top-level function) for every import statement and `importlib.import_module` string."""
    tree = ast.parse(path.read_text())
    within = {}
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(top):
                within[id(n)] = top.name
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(a.name, n.lineno, within.get(id(n))) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                base = ".".join(PKG.split(".")[: len(PKG.split(".")) - n.level + 1] + ([n.module] if n.module else []))
            out += [(f"{base}.{a.name}" if base == PKG else base, n.lineno, within.get(id(n))) for a in n.names]
        elif (isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) in ("import_module", "__import__")
              and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
            out.append((n.args[0].value, n.lineno, within.get(id(n))))
    return out


@pytest.mark.parametrize("module", DEFINITION_MODULES)
def test_a_definition_module_imports_only_verity_and_definition_modules(module):
    allowed = {f"{PKG}.{m}" for m in DEFINITION_MODULES + tuple(OTHER_DEFINITION_MODULES)}
    path = REG / f"{module}.py"
    assert path.exists(), f"{module}: no such module (moved? update DEFINITION_MODULES)"
    bad = sorted({f"{module}.py:{line}: imports {mod}" for mod, line, fn in _imports(path)
                  if not (_is_stdlib(mod) or mod.split(".")[0] in ("numpy", "verity") or mod in allowed or mod == CATALOG
                          or mod.startswith(CATALOG + ".") or (module, fn, mod) in EXCEPTIONS)})
    assert not bad, "\n".join(bad)


def test_the_exceptions_are_current():
    used = {(m, fn, mod) for m in DEFINITION_MODULES if (REG / f"{m}.py").exists() for mod, _, fn in _imports(REG / f"{m}.py")}
    assert set(EXCEPTIONS) <= used, sorted(set(EXCEPTIONS) - used)


def test_twins_and_vocabulary_are_not_definition_modules(defs):
    """The twin and vocabulary modules construct no Definition the registry serves (the difftest's own is unregistered)."""
    served = {k for k, d in defs.items() if defining_module(d) in {f"{PKG}.{m}" for m in TWIN_MODULES + VOCAB_MODULES} and d.register}
    assert not served, sorted(served)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: test_sampling_split.py --write")
    defs = family_definitions()
    keys = spec_keys(defs)
    data = {"modules": {k: defining_module(d) for k, d in sorted(defs.items())}, "pins": {k: pin(k, defs) for k in keys}}
    lines = ["{", '  "modules": {', *[f"    {json.dumps(k)}: {json.dumps(v)}," for k, v in data["modules"].items()]]
    lines[-1] = lines[-1].rstrip(",")
    lines += ["  },", '  "pins": {', *[f"    {json.dumps(k)}: {json.dumps(v)}," for k, v in data["pins"].items()]]
    lines[-1] = lines[-1].rstrip(",")
    lines += ["  }", "}"]
    FIXTURE.write_text("\n".join(lines) + "\n")
    print(f"{FIXTURE}: {len(data['modules'])} Definitions, {len(data['pins'])} pins")
