"""The dense, activation, norm and RoPE family's registry split (`integrations/vllm/tools/migrate/split_dense.py`) changes no
circuit: every Definition id and canonical descriptor digest the family has at the statics a circuit-check binding, a pin, a
tracked fixture or a difftest uses, and the family's slice of circuit-check's task list (target id, statics, pins), equal the
ones `fixtures/split_dense.json` recorded on the tree before the split.

    uv run --directory tools/circuit_check python tests/test_split_dense.py --update    # rerecord (on the tree before a rerun)

A family Definition added later is a new id here, and `--update` records it (review the diff: only additions).
"""
from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).parent / "fixtures" / "split_dense.json"
REGISTRY = "verity_vllm.program.registry"
#: the family's Definition modules
DEFINITION_MODULES = ("act", "boolean_activation", "boolean_dense", "boolean_dense_norm", "boolean_norms", "boolean_rope",
                      "boolean_silu", "dense", "gemm_bias")
#: the family's difftest adapters (twins), each with a ``register=False`` spec Definition at its ``SHAPES``: the old and the new
#: package, so the fixture is recorded before the split and checked after it
DIFFTESTS = ("rope_difftest", "rope_gptj_difftest")
TWIN_PACKAGES = ("verity_vllm.program.kernels", REGISTRY)


def _twin(name: str):
    for pkg in TWIN_PACKAGES:
        try:
            return importlib.import_module(f"{pkg}.{name}")
        except ModuleNotFoundError as e:
            if e.name not in (f"{pkg}.{name}",):
                raise
    raise ModuleNotFoundError(f"{name} is in none of {TWIN_PACKAGES}")


def family_names() -> set[str]:
    """``Name_vN`` of every Definition the family's modules define (not the ones they import from other families or core)."""
    from verity.ir.defs import CompositeDefinition, PrimitiveDefinition
    import verity.ml.prims as core_prims
    from verity_vllm.program.registry import b1, prims, sampling

    foreign = {id(v) for m in (b1, prims, sampling, core_prims) for v in vars(m).values()}
    out = set()
    for m in DEFINITION_MODULES:
        for v in vars(importlib.import_module(f"{REGISTRY}.{m}")).values():
            if isinstance(v, (CompositeDefinition, PrimitiveDefinition)) and id(v) not in foreign:
                out.add(f"{v.name}_v{v.version}")
    return out


def _stem(fid: str) -> str:
    return fid.split("{", 1)[0]


def digest(fn) -> dict:
    """The descriptor id and the canonical digest: a composite's whole Program's (`program_digest`, every callee included), a
    primitive's descriptor entry's."""
    from verity.ir.codec import canonical_json, definition_entry, descriptor_id, encode_program, program_digest
    from verity.ir.defs import PrimitiveDefinition
    from verity.ir.program import Program

    if isinstance(fn, PrimitiveDefinition):
        d = hashlib.sha256(canonical_json(definition_entry(fn))).hexdigest()
    else:
        d = program_digest(encode_program(Program(fn)))
    return {"descriptor_id": descriptor_id(fn), "digest": d}


def resolve(spec: str):
    """The family specialization ``spec`` names, through circuit-check's resolver, never its suite fallback (`TG.suite()` is
    not idempotent in one process: a second call adds the lazily registered constant families)."""
    from circuit_check import targets as TG
    from verity_vllm.query import word as W

    TG.load_registries()
    m = re.fullmatch(r"([^{]+)(?:\{(.*)\})?", spec)
    statics = {k: TG._value(v) for k, v in TG._split(m[2]).items()} if m[2] else {}
    fn = W.specialization(m[1], statics)
    return fn.bind() if hasattr(fn, "bind") and not statics and getattr(fn, "statics", None) == () else fn


def difftest_specs() -> dict[str, object]:
    out = {}
    for name in DIFFTESTS:
        mod = _twin(name)
        for nh, kvh, d in mod.SHAPES:
            fn = mod.spec({"NH": nh, "KVH": kvh, "D": d})
            out[fn.id] = fn
    return out


_TASKS = """
import json, sys
from circuit_check import checks as CK, cli
names = set(json.loads(sys.argv[1]))
tasks = cli._tasks([], CK.Options(), True, None)
fam = [[k, l] for k, l, _ in tasks if k == "definition" and l.split("{", 1)[0] in names]
full = [[k, l] for k, l, _ in tasks]
print(json.dumps({"family": fam, "count": len(full), "sha256": __import__("hashlib").sha256(json.dumps(full).encode()).hexdigest()}))
"""


def task_list(names: set[str]) -> dict:
    """``circuit-check --all``'s task list, built as the CLI builds it (a fresh process: its first `TG.suite()`): the family's
    definition targets in target order, and the whole list's length and SHA-256."""
    out = subprocess.run([sys.executable, "-c", _TASKS, json.dumps(sorted(names))], check=True, capture_output=True, text=True,
                         cwd=REPO / "tools/circuit_check")
    return json.loads(out.stdout.strip().splitlines()[-1])


def family_pins(names: set[str]) -> dict:
    from circuit_check import checks as CK

    return {sec: {k: v for k, v in sorted(entries.items()) if _stem(k) in names} for sec, entries in CK.load_pins().items()}


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_family_definitions_are_the_recorded_ones():
    assert sorted(family_names()) == _fixture()["names"]


def test_family_pins_are_the_recorded_ones():
    assert family_pins(set(_fixture()["names"])) == _fixture()["pins"]


@pytest.mark.slow
def test_every_family_id_and_digest_is_unchanged():
    want = _fixture()["digests"]
    specs = difftest_specs()
    got = {fid: digest(specs[fid] if fid in specs else resolve(fid)) for fid in want}
    changed = sorted(f for f in want if {k: want[f][k] for k in ("descriptor_id", "digest")} != got[f])
    assert not changed, "ids or digests changed:\n" + "\n".join(f"{f}: {want[f]} -> {got[f]}" for f in changed)


@pytest.mark.slow
def test_the_family_slice_of_the_task_list_is_unchanged():
    """About 70 s (a fresh process builds the whole `--all` list). It catches what the digests can't: a target the split adds
    or drops (a moved module that registered a Definition, a binding that no longer resolves), and a cheaper in-process check
    can't build the list the CLI builds, as a second `TG.suite()` in one process returns more targets."""
    assert task_list(set(_fixture()["names"]))["family"] == _fixture()["tasks"]


# --- recording ------------------------------------------------------------------------------------------------------------

def _tracked_ids(names: set[str]) -> dict[str, list[str]]:
    """{id with concrete statics: tracked JSON / JSONL files naming it} over the repository (pins.json aside)."""
    alt = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
    pat = re.compile(rf"(?<![\w\[])({alt})\{{")
    files = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], check=True, capture_output=True).stdout.decode().split("\0")
    found: dict[str, set[str]] = {}
    for rel in files:
        if not rel.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz")) or rel.endswith("circuit_check/pins.json"):
            continue
        if rel == str(FIXTURE.relative_to(REPO)):
            continue
        raw = (REPO / rel).read_bytes()
        text = (gzip.decompress(raw) if rel.endswith(".gz") else raw).decode("utf-8", "replace")
        for m in pat.finditer(text):
            depth, j = 0, m.end() - 1
            while j < len(text):
                depth += text[j] in "{[("
                depth -= text[j] in "}])"
                if depth == 0:
                    break
                j += 1
            fid = text[m.start():j + 1]
            if re.fullmatch(r"[^{]+\{(\w+=[^,{}]+(,\w+=[^,{}]+)*)?\}", fid):
                found.setdefault(fid, set()).add(rel)
    return {k: sorted(v) for k, v in sorted(found.items())}


def record() -> dict:
    names = family_names()
    tasks = task_list(names)
    pins = family_pins(names)
    sources: dict[str, set[str]] = {}
    for _, fid in tasks["family"]:
        sources.setdefault(fid, set()).add("circuit-check target")
    for sec, entries in pins.items():
        for fid in entries:
            sources.setdefault(fid, set()).add(f"pins.json {sec}")
    for fid, files in _tracked_ids(names).items():
        sources.setdefault(fid, set()).update(files)
    specs = difftest_specs()
    for fid in specs:
        sources.setdefault(fid, set()).add("difftest spec")
    digests = {fid: {**digest(specs[fid] if fid in specs else resolve(fid)), "used_by": sorted(src)}
               for fid, src in sorted(sources.items())}
    whole = {"count": tasks["count"], "sha256": tasks["sha256"]}
    return {"names": sorted(names), "tasks": tasks["family"], "pins": pins, "digests": digests}, whole


if __name__ == "__main__":
    if sys.argv[1:] != ["--update"]:
        sys.exit(__doc__)
    FIXTURE.parent.mkdir(exist_ok=True)
    rec, whole = record()
    FIXTURE.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(f"{FIXTURE.relative_to(REPO)}: {len(rec['names'])} names, {len(rec['tasks'])} targets, {len(rec['digests'])} ids; "
          f"the whole task list (not recorded: every new Definition changes it): {whole}")
