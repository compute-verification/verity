"""The MoE / FP8 registry split (`integrations/vllm/tools/migrate/split_moe_fp8.py`) keeps every Definition and every circuit-check
target as it was.

`fixtures/moe_fp8_split.json` was written by ``python tests/test_moe_fp8_split.py --write`` on the tree before the split; only the
SHA-512 re-baseline of the circuit identities has rewritten it since.  It holds

* ``digests``: descriptor id -> SHA-512 of the canonical descriptor entry (`verity.ml.circuits.codec.definition_entry`, callees by id) of every
  Definition defined in, or named by, one of the family's registry modules (`FAMILY`) that circuit-check's catalog reaches, and of
  every other binding of those Definitions that a tracked pin, fixture, record or capture map names (``extra``);
* ``tasks``: circuit-check's task list restricted to the family (kind, target id with its statics, and the pins of that id).

Both tests trace the whole catalog (about a minute alone, nothing extra after `test_circuit_check.py` in one session): only the
traced bodies carry the ids and digests, which is what the split must not move.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "moe_fp8_split.json"
FAMILY = ("moe", "moe_pad", "moe_difftest", "boolean_moe", "fp8", "fp8_moe", "boolean_fp8", "boolean_fp8_moe", "nvfp4", "boolean_gather")
MODULES = tuple(f"verity_vllm.program.registry.{m}" for m in FAMILY)


def _home(d) -> str | None:
    f = getattr(d, "evaluate", None) or getattr(d, "body", None)
    return getattr(f, "__module__", None)


@lru_cache(maxsize=1)
def _named_bases() -> frozenset[str]:
    """The base ids of the Definitions the family's modules hold as module attributes (their own and those they import)."""
    out = set()
    for m in MODULES:
        for v in vars(importlib.import_module(m)).values():
            for o in (v if isinstance(v, (list, tuple)) else (v,)):
                if not isinstance(o, type) and hasattr(o, "id") and (hasattr(o, "body") or hasattr(o, "evaluate")):
                    out.add(getattr(o, "base_id", None) or o.id)
    return frozenset(out)


def _owned(fn) -> bool:
    return _home(getattr(fn, "defn", fn)) in MODULES or (getattr(fn, "base_id", None) or fn.id) in _named_bases()


def _digest(fn) -> str:
    from verity.ml.circuits.codec import canonical_json, definition_entry
    return hashlib.sha512(canonical_json(definition_entry(fn))).hexdigest()


@lru_cache(maxsize=1)
def _found() -> dict:
    """Every Definition circuit-check's catalog reaches, by descriptor id (scalar references have no descriptor)."""
    from circuit_check import targets as TG
    from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
    return {fn_id(fn): fn for fn in TG.reachable(TG.catalog()).values() if isinstance(fn, (PrimitiveDefinition, SpecializedDefinition))}


def fn_id(fn) -> str:
    from verity.ml.circuits.codec import descriptor_id
    return descriptor_id(fn)


def family_digests() -> dict[str, str]:
    return {i: _digest(fn) for i, fn in sorted(_found().items()) if _owned(fn)}


def _resolve(i: str):
    from circuit_check import targets as TG
    return _found().get(i) or TG.definition(i)


def family_tasks() -> list[list]:
    """(kind, target id, pins) of every circuit-check task on a family Definition or template, in task order."""
    from circuit_check import checks as CK
    from circuit_check import cli
    from circuit_check import targets as TG

    pins = CK.load_pins()
    suite = TG.suite()
    subs = {label: (sub, sts) for label, sub, sts in TG.templates()}
    out = []
    for kind, label, *_ in cli._tasks([], CK.Options(), True, None):
        if kind == "definition":
            owned = _owned(suite[label])
        else:
            sub, sts = subs[label]
            owned = any(_owned(sub.definition(**st)) for st in sts)
        if owned:
            pinned = {t: pins[t][label] for t in ("definitions", "pieces", "units") if label in pins.get(t, {})}
            out.append([kind, label, pinned])
    return out


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.mark.slow
def test_every_family_definition_keeps_its_id_and_digest():
    fx = _fixture()
    now = family_digests()
    lost = sorted(set(fx["digests"]) - set(now))
    assert not lost, f"family Definitions circuit-check no longer reaches at these ids: {lost[:20]}"
    changed = sorted(i for i, d in fx["digests"].items() if now[i] != d)
    assert not changed, f"descriptor digests that changed: {changed[:20]}"
    extra = {i: _digest(_resolve(i)) for i in fx["extra"]}
    assert extra == fx["extra"], "a pinned or recorded binding resolves to another descriptor"


@pytest.mark.slow
def test_the_family_keeps_its_circuit_check_tasks():
    want = [tuple(json.dumps(x, sort_keys=True) for x in t) for t in _fixture()["tasks"]]
    have = {tuple(json.dumps(x, sort_keys=True) for x in t) for t in family_tasks()}
    missing = [t for t in want if t not in have]
    assert not missing, f"circuit-check tasks (kind, target, pins) of the family that are gone or changed: {missing[:10]}"


def _extra(digests: dict[str, str]) -> dict[str, str]:
    """Bindings of the family's Definitions at concrete statics that tracked JSON/TOML names and the catalog does not reach."""
    import re
    import subprocess

    repo = Path(__file__).resolve().parents[4]
    bases = sorted({i.split("{", 1)[0] for i in digests} | _named_bases(), key=len, reverse=True)
    names = sorted({b.rsplit("_v", 1)[0] for b in bases}, key=len, reverse=True)
    pat = re.compile(r"(?<![A-Za-z0-9_\[])(" + "|".join(map(re.escape, names)) + r")_v(\d+)\{")
    files = subprocess.run(["git", "ls-files", "*.json", "*.toml", "*.jsonl"], cwd=repo, capture_output=True, text=True,
                           check=True).stdout.split()
    found = set()
    for f in files:
        s = (repo / f).read_text(errors="replace")
        for m in pat.finditer(s):
            if f"{m[1]}_v{m[2]}" not in bases:
                continue
            depth, j = 0, m.end() - 1
            while j < len(s):
                depth += {"{": 1, "}": -1}.get(s[j], 0)
                if depth == 0:
                    break
                j += 1
            found.add(s[m.start():j + 1].replace('\\"', '"'))
    out = {}
    for i in sorted(found - set(digests)):
        try:
            fn = _resolve(i)
            fid, d = fn_id(fn), _digest(fn)
        except Exception:  # noqa: BLE001  (symbolic or partial statics such as `{E, K}`, or an id that does not parse back)
            continue
        if fid not in digests:
            out[i] = d
    return out


def write() -> None:
    digests = family_digests()
    extra = _extra(digests)
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps({"digests": digests, "extra": extra, "tasks": family_tasks()}, indent=1, sort_keys=True) + "\n")
    print(f"{FIXTURE}: {len(digests)} digests, {len(extra)} extra, tasks written")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: python tests/test_moe_fp8_split.py --write   (only on the tree before the split)")
    write()
