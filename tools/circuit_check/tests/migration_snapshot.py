"""What a module move must leave unchanged, as data: run in a fresh process, it prints (or with ``--write DIR`` writes) the
snapshot `test_migration_invariants.py` compares with the one taken before a split.

    uv run --extra torch-cpu python tools/circuit_check/tests/migration_snapshot.py --write tools/circuit_check/tests/fixtures/migration

- ``tasks``: `circuit-check --all`'s task list as `cli._tasks` builds it in a fresh process (`load_registries`, `suite()`, then
  `templates()`), each with the pin `pins.json` holds for it;
- ``definitions``: the canonical descriptor digest (sha256 of `canonical_json(definition_entry(fn))`, 32 hex digits) of every
  Definition the catalog's roots reach, and of every Definition `pins.json` names, by descriptor id;
- ``vocabularies``: each frontend `Vocabulary`'s `version`, the digest an instance record pins;
- ``ref_vocab_kinds``: the digest of `ref_prims.ref_vocab_manifest()` without ``module_sha256``.  The ``reference-ref-prims``
  vocabulary's version pins the whole manifest, which hashes `ref_prims.py`'s bytes, so any edit to that file moves it; this
  digest is what stays when no reference kind's evaluator, body or family changed.

The order is fixed because it matters: tracing the roots registers lazy families' members, so a second `catalog()` reaches
more than the first.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

DIGITS = 32


def snapshot() -> dict:
    from circuit_check import checks as CK
    from circuit_check import targets as TG
    from verity.evaluation import ScalarReference
    from verity.ir.codec import canonical_json, definition_entry, descriptor_id

    def digest(fn) -> str:
        return hashlib.sha256(canonical_json(definition_entry(fn))).hexdigest()[:DIGITS]

    TG.load_registries()
    pins = CK.load_pins()
    tasks = [["definition", fid, pins.get("definitions", {}).get(fid)] for fid in TG.suite()]
    tasks += [["template", label, sts] for label, _sub, sts in TG.templates()]
    definitions, references = {}, []
    for fid, fn in TG.reachable(TG.catalog()).items():
        if isinstance(fn, ScalarReference):
            references.append(fid)
        else:
            definitions[descriptor_id(fn)] = digest(fn)
    unresolved = []
    for fid in sorted(set(pins.get("definitions", {})) | set(pins.get("pieces", {}))):
        try:
            fn = TG.definition(fid)
        except KeyError:
            if fid in pins.get("definitions", {}):
                raise
            unresolved.append(fid)   # a Boolean lowering's piece that no registry module defines by that id
            continue
        definitions.setdefault(descriptor_id(fn), digest(fn))

    from verity_vllm.program.frontend.rules import vocab
    from verity_vllm.program.registry import ref_prims
    vocabularies = {v.name: v.version for _n, v in inspect.getmembers(vocab) if isinstance(v, vocab.Vocabulary)}
    kinds = {k: v for k, v in ref_prims.ref_vocab_manifest().items() if k != "module_sha256"}
    return {"tasks": tasks, "definitions": dict(sorted(definitions.items())), "references": sorted(references),
            "registered": sorted(TG.REGISTERED), "unresolved_pieces": unresolved, "vocabularies": dict(sorted(vocabularies.items())),
            "ref_vocab_kinds": hashlib.sha256(json.dumps(kinds, sort_keys=True).encode()).hexdigest()}


def write(snap: dict, out: Path) -> None:
    """Two files, each under the repository's 256 KiB cap: the task list (with the vocabularies) and the digests."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_text(json.dumps({"tasks": snap["tasks"], "references": snap["references"],
                                                "unresolved_pieces": snap["unresolved_pieces"], "vocabularies": snap["vocabularies"],
                                                "ref_vocab_kinds": snap["ref_vocab_kinds"]}, indent=0, sort_keys=True) + "\n")
    (out / "definitions.json").write_text(json.dumps({"definitions": snap["definitions"], "registered": snap["registered"]},
                                                     indent=0, sort_keys=True) + "\n")


def read(out: Path) -> dict:
    return json.loads((out / "tasks.json").read_text()) | json.loads((out / "definitions.json").read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", type=Path, help="write tasks.json and definitions.json there instead of printing")
    a = ap.parse_args(argv)
    snap = snapshot()
    if a.write:
        write(snap, a.write)
    else:
        json.dump(snap, sys.stdout, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
