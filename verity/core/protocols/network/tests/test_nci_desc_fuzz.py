"""The desc fuzzer's run (`NciCheck.DescFuzzMain`): the Python half.

`verity/core/protocols/network/nci/lean/scripts/desc_fuzz_programs.py` writes the 1,856 Programs `NciCheck/DescFuzzMain.lean`
decides the statements behind `DescAgrees` on (`runs` in that package's `lean-audit.json`), which the Lean audit runs and
caches on the Lean package's files alone, against the data's SHA-256 in the record. So the first test is what says the
data is still what the generator writes. The second, that it holds the shape #4066 fixed: one gate reading, through one
reference to one root node, whole batch members returning leaves of arguments they share, each another root node's
output, which `NciCheck.Desc` once attributed to the copy's first piece's node alone.
"""

from __future__ import annotations

import bisect
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

LEAN = Path(__file__).resolve().parents[1] / "nci" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fuzz = _load("desc_fuzz_programs")
reference = _load("nci_reference")
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NciCheck.DescFuzzMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in fuzz.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{desc_fuzz/programs.jsonl}", str(fuzz.count())]


def _shared_reads(text: str) -> list[tuple[str, str, int]]:
    """Each root node all of whose arguments name one root node, and one of whose gates reads gates of two or more
    other root nodes: the named node's form, the reader's, and the most root nodes one of its gates reads."""
    _, _, spans = reference.root_nodes(text)
    _, _, _, reads = reference.circuit(text)
    offs = [lo for lo, _ in spans]
    desc = json.loads(text)
    nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
    out = []
    for k, node in enumerate(nodes):
        named = {a["refs"]["affine"][1] if "affine" in a["refs"] and a["refs"]["affine"][0] == "n" else None
                 for a in node["args"]}
        if len(named) != 1 or None in named:
            continue
        most = max((len({bisect.bisect_right(offs, h) - 1 for h in reads[g]} - {k}) for g in range(*spans[k])),
                   default=0)
        if most >= 2:
            out.append((nodes[named.pop()]["form"], node["form"], most))
    return out


def test_the_shared_family_reads_several_nodes_through_one(tmp_path):
    """`desc_difftest_vectors.shared()`'s six declarations come right after `hand`, each read so; of the 600 `shared`
    seeds, every one decodes and most are read so, through a batch at the root or a call of one, by a primitive, a
    batch mapped over the members or a call, and some gate reads four root nodes through the one."""
    data = fuzz.write(tmp_path)["desc_fuzz/programs.jsonl"]
    lines = [json.loads(x) for x in data.read_text().splitlines()]
    families = Counter(x["family"] for x in lines)
    assert families["shared-vector"] == 6 and families["shared"] == 600 and families["shared-mutant"] > 0
    vectors = lines[families["hand"]:families["hand"] + 6]
    assert [x["family"] for x in vectors] == ["shared-vector"] * 6
    assert [x["program"] for x in vectors] == fuzz.shared_vectors()
    assert all(_shared_reads(x["program"]) for x in vectors)
    seeds = [x["program"] for x in lines if x["family"] == "shared"]
    assert len(set(seeds)) == 600
    found = [_shared_reads(text) for text in seeds]
    assert sum(map(bool, found)) >= 300
    assert {(named, reader) for f in found for named, reader, _ in f} == \
        {(n, r) for n in ("batch", "call") for r in ("primitive", "batch", "call")}
    assert max(most for f in found for *_, most in f) >= 4
