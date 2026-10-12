"""circuit-check's task list for the sampling family is the one found on the unsplit tree.

`integrations/vllm/tools/migrate/split_sampling.py` splits the family's registry modules into Definitions, twins and vocabulary
without changing a Definition, so every circuit-check target of the family keeps its kind, label, id, statics and pins.
`data/sampling_split_tasks.json` holds them as `split_sampling.py snapshot` wrote them; a change to the family that adds, drops or
rebinds a target, or repins one, rewrites it (`python tools/verity/circuit_check/tests/test_sampling_split_tasks.py --write`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent / "data" / "sampling_split_tasks.json"
PKG = "verity_vllm.program.registry."
#: every module of the family, before and after the split; a Definition belongs to the family when one of them constructs it
FAMILY_MODULES = ("sampling", "sampling_boolean", "gumbel_boolean", "topp_word_gates", "topp_keep_builder", "topp_words", "topp_geometry",
                  "sampling_rows", "sampling_topp_difftest", "topp_split", "sampling_twins", "sampling_vocab")


def defining_module(fn) -> str | None:
    """The module whose code a Definition runs: a primitive's evaluator's, a composite's body's, a traced Boolean one's build's
    (`verity.ml.circuits.boolean.trace.definition`)."""
    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition

    d = fn.defn if isinstance(fn, SpecializedDefinition) else fn
    f = d.evaluate if isinstance(d, PrimitiveDefinition) else getattr(d, "build", d.body) if isinstance(d, CompositeDefinition) else None
    return getattr(getattr(f, "func", f), "__module__", None)


def in_family(fn) -> bool:
    return (defining_module(fn) or "").removeprefix(PKG) in FAMILY_MODULES


def _static(v):
    return v if isinstance(v, (bool, int, float, str)) or v is None else getattr(v, "id", repr(v))


def family_tasks(suite: dict | None = None) -> dict:
    """The family's entries of `circuit-check --all`'s task list (`cli._tasks`), each with its pin, and the pins of every family
    Definition those targets call; with ``suite``, the task list is built on that `targets.suite()`."""
    from verity.ml.circuits.defs import REGISTRY, PrimitiveDefinition

    from circuit_check import checks as CK
    from circuit_check import cli
    from circuit_check import targets as TG

    real = TG.suite
    if suite is not None:
        TG.suite = lambda: suite
    try:
        tasks = cli._tasks([], CK.Options(), everything=True, as_call=None)
    finally:
        TG.suite = real
    pins = CK.load_pins()
    family = {k for k, d in REGISTRY.defs.items() if in_family(d)}
    out = []
    for kind, label, thunk, *_ in tasks:
        if kind == "definition":
            fn = thunk.__defaults__[0]
            if not in_family(fn):
                continue
            out.append({"kind": kind, "label": label, "id": fn.id, "statics": {k: _static(v) for k, v in getattr(fn, "bindings", {}).items()},
                        "pin": pins["definitions"].get(fn.id)})
        else:
            sub, sts = thunk.__defaults__[:2]
            rel = (getattr(sub, "relation", None) or {}).get("definition")
            if rel is None or rel.split("{")[0] not in family:
                continue
            out.append({"kind": kind, "label": label, "id": sub.id, "template": sub.template.id,
                        "params": {k: _static(v) for k, v in dict(sub.params).items()}, "instance_statics": sts,
                        "pin": pins["units"].get(sub.id)})
    reached, todo = {}, [thunk.__defaults__[0] for kind, _, thunk, *_ in tasks if kind == "definition" and in_family(thunk.__defaults__[0])]
    while todo:
        fn = todo.pop()
        if fn.id in reached:
            continue
        reached[fn.id] = fn
        if not isinstance(fn, PrimitiveDefinition):
            todo += [n.fn for n in fn.body.nodes]
    ids = {i for i, fn in reached.items() if in_family(fn)} | {k for k, d in REGISTRY.defs.items() if in_family(d)}
    return {"tasks": out, "pins": {sec: {k: v for k, v in sorted(pins[sec].items()) if k in ids} for sec in ("definitions", "pieces")}}


def _suite() -> dict:
    """The process's first `targets.suite()`, the one circuit-check builds: `test_circuit_check.py`'s, built at import, when it is
    loaded. `targets.suite()` grows as tracing registers Definitions, so a later call in this process lists more targets."""
    from circuit_check import targets as TG

    other = sys.modules.get("test_circuit_check")
    return other.SUITE if other is not None and hasattr(other, "SUITE") else TG.suite()


@pytest.mark.slow
def test_the_family_s_circuit_check_tasks_are_unchanged():
    """Builds circuit-check's whole task list on the suite `test_circuit_check.py` built (about a minute alone, as
    `test_split_prims_serve.py` does): only the list itself shows that the split left every target of the family where
    circuit-check finds it, at the same statics and pins."""
    want = json.loads(FIXTURE.read_text())
    got = family_tasks(_suite())
    assert len(got["tasks"]) == len(want["tasks"]) and got["tasks"] == want["tasks"], (
        sorted({t["label"] for t in want["tasks"]} ^ {t["label"] for t in got["tasks"]}) or "same labels, other entries")
    assert got["pins"] == want["pins"]


if __name__ == "__main__":
    data = family_tasks()
    if sys.argv[1:] == ["--print"]:
        print(json.dumps(data))
    elif sys.argv[1:] == ["--write"]:
        FIXTURE.parent.mkdir(exist_ok=True)
        def rows(items: list[str], indent: str) -> list[str]:
            return [f"{indent}{s}{',' if i < len(items) - 1 else ''}" for i, s in enumerate(items)]

        lines = ["{", '  "tasks": [', *rows([json.dumps(t) for t in data["tasks"]], "    "), "  ],", '  "pins": {']
        for j, (sec, p) in enumerate(data["pins"].items()):
            lines += [f"    {json.dumps(sec)}: {{", *rows([f"{json.dumps(k)}: {json.dumps(v)}" for k, v in p.items()], "      "),
                      "    }" + ("," if j < len(data["pins"]) - 1 else "")]
        lines += ["  }", "}"]
        FIXTURE.write_text("\n".join(lines) + "\n")
        print(f"{FIXTURE}: {len(data['tasks'])} tasks, pins " + ", ".join(f"{len(p)} {sec}" for sec, p in data["pins"].items()))
    else:
        sys.exit("usage: test_sampling_split_tasks.py --print | --write")
