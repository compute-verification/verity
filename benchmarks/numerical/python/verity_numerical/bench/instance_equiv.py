"""``instance-equiv/v1`` (campaign morning-tables BRIEF §5): are the instances a re-packed relation's results carry the SAME
numbers as the frozen set of their target?

    python -m verity_numerical.bench.instance_equiv --relation fp8-ada-v3x4 --out-dir DIR     # one file per relation
    python -m verity_numerical.bench.instance_equiv --check DIR/*.json                        # re-derive files, compare

The derived B-Ligero relations (``-v2``, ``-v3``, ``-v2x4``, ``-v3x4``, ``-x4`` of fp8-ada, bf16-hopper, fp8-hopper) draw their
VUs like their v1 relation -- same seed, operand recipe and model, the folds keep every ``m``-th recorded accumulator public --
but ``relchain.instances_digest`` names the RELATION in the manifest string, so their results carry another
``manifest_sha256`` than the target's frozen ref (``tables.FROZEN_INSTANCES``).  This tool loads both sides with their own
loaders -- the frozen set with the registered relation of the target whose runner ref IS the frozen ref (dataset, tier,
range, manifest_sha256), the candidate with the candidate relation's (``relchain.instances``, what ``bench-vu`` proves) --
decodes each to canonical x, W, y arrays and writes the §5 JSON; ``equal`` iff every array's two sha256 agree.  The
candidate ref is ``relchain.instances_ref`` (field for field what ``bench-vu`` writes).  It never labels anything.

The loaders are the runner's (``archive.direct.ligero.relchain`` imports torch): run it where the prover runs (a pod).
``--check`` re-derives each file from the relation its candidate ref names and reports whether every field reproduces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["KEYS", "SCHEMA", "arrays", "document", "main", "tool_id"]

SCHEMA = "instance-equiv/v1"
#: the instances fields the tables compare (``tables._candidate_problems``)
KEYS = ("dataset", "tier", "range", "manifest_sha256")
REPO = Path(__file__).resolve().parents[5]


def tool_id() -> str:
    """``verity_numerical.bench.instance_equiv@<commit>``: ``$RESEARCH_GIT_COMMIT``, else the ``.research-source.json`` that
    ``research pods sync`` writes (``-dirty`` when the shipped tree was), else ``git rev-parse HEAD``."""
    commit = os.environ.get("RESEARCH_GIT_COMMIT")
    src = REPO / ".research-source.json"
    if not commit and src.is_file():
        try:
            doc = json.loads(src.read_text())
            commit = doc.get("commit")
            if commit and doc.get("dirty"):
                commit += "-dirty"
        except (OSError, ValueError):
            commit = None
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=10,
                                    check=False).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            commit = None
    return f"verity_numerical.bench.instance_equiv@{commit or 'unknown'}"


def _canon(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


def same_ref(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return all(_canon(a.get(k)) == _canon(b.get(k)) for k in KEYS)


def arrays(vus: Sequence[tuple], word_dtype: str) -> dict[str, np.ndarray]:
    """Canonical arrays of a loader's VUs ``(a, b, accs, y_final)``: x = the ``a`` operand words, W = the ``b`` operand
    words (the datatype's bit words as ``word_dtype``, little-endian), y = the chain-end FP32 accumulator word (``<u4``);
    rows VU-major, then K."""
    wd = np.dtype(word_dtype).newbyteorder("<")
    out = {}
    for name, col in (("x", 0), ("W", 1)):
        v = np.asarray([np.asarray(u[col], dtype=np.int64).reshape(-1) for u in vus], dtype=np.int64)
        if v.size and (v.min() < 0 or v.max() >= 1 << (8 * wd.itemsize)):
            raise ValueError(f"{name}: operand words outside {wd.str}")
        out[name] = v.astype(wd)
    y = np.asarray([int(u[3]) for u in vus], dtype=np.int64)
    if y.size and (y.min() < 0 or y.max() >= 1 << 32):
        raise ValueError("y: chain-end words outside <u4")
    out["y"] = y.astype(np.dtype("<u4"))
    return out


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def document(target: str, frozen: Mapping[str, Any], candidate: Mapping[str, Any], fa: Mapping[str, np.ndarray],
             ca: Mapping[str, np.ndarray], canonical: str, tool: str) -> dict[str, Any]:
    """The §5 JSON for two decoded sides; ``equal`` iff every array's digests (and dtype and shape) agree."""
    arr, equal = {}, True
    for name in ("x", "W", "y"):
        f, c = fa[name], ca[name]
        fs, cs = _sha(f), _sha(c)
        equal &= fs == cs and f.dtype == c.dtype and f.shape == c.shape
        arr[name] = {"frozen_sha256": fs, "candidate_sha256": cs, "dtype": f.dtype.str if f.dtype == c.dtype else
                     f"{f.dtype.str} vs {c.dtype.str}", "shape": list(f.shape) if f.shape == c.shape else [list(f.shape), list(c.shape)]}
    return {"schema": SCHEMA, "target": target, "frozen": dict(frozen), "candidate": dict(candidate), "canonical": canonical,
            "arrays": arr, "equal": bool(equal), "tool": tool}


# --- the runner's loaders (pod) ----------------------------------------------------------------------------------------


def _runner():
    from archive.direct.ligero import relchain
    from archive.direct.ligero.relations import RELATIONS
    from verity_numerical.bench.tables import FROZEN_INSTANCES
    return relchain, RELATIONS, FROZEN_INSTANCES


def declared(target: str, ref: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Whether a runner ref over [0, n) is ``target``'s declared set -- the frozen ref, a prefix of it, or a synthetic stream by
    generator, seed and n (``views.instance_range``) -- with why not when it names the declared set but repeats it."""
    from verity.ml.circuits.targets.target import TARGETS
    from verity_numerical.bench.tables import FROZEN_INSTANCES
    from verity_numerical.bench.views import instance_range
    if same_ref(ref, FROZEN_INSTANCES[target]):
        return True, None
    how, why = instance_range(TARGETS[target], ref)
    return how is not None, why


def frozen_relation(target: str, n: int, root=None):
    """The registered relation of ``target`` whose runner ref over ``n`` VUs is the declared set (the set's own loader), and
    that ref: the unfolded, shortest-named one when several are (the bf16-ampere family all read the frozen ``vu-k1536``).
    Past the frozen 4,096 the ref is the synthetic stream (generator, seed, n); a fixture with no stream (A100) refuses."""
    relchain, RELATIONS, FROZEN = _runner()
    if target not in FROZEN:
        raise SystemExit(f"{target}: no frozen input set (tables.FROZEN_INSTANCES)")
    hits, whys = [], set()
    for r in RELATIONS.values():
        if r.target.name != target:
            continue
        ok, why = declared(target, relchain.instances_ref(r, n, root=root))
        if ok:
            hits.append(r)
        elif why:
            whys.add(why)
    if not hits:
        raise SystemExit(f"{target}: " + ("; ".join(sorted(whys)) or f"no registered relation carries the declared set at n = {n}"))
    base = min(hits, key=lambda r: (r.fold, len(r.name), r.name))
    return base, relchain.instances_ref(base, n, root=root)


class _Sets:
    """Each relation's VUs drawn once per process (the frozen side is shared by every candidate of a target)."""

    def __init__(self, n: int, procs: int, root=None):
        self.n, self.procs, self.root, self._memo = n, procs, root, {}

    def load(self, rel):
        if rel.name not in self._memo:
            relchain, _, _ = _runner()
            vus = relchain.instances(rel, self.n, procs=self.procs, root=self.root)
            if len(vus) != self.n:
                raise ValueError(f"{rel.name}: the loader returned {len(vus)} VUs, not {self.n}")
            self._memo[rel.name] = arrays(vus, rel.word_dtype)
        return self._memo[rel.name]


def derive(name: str, sets: _Sets) -> dict[str, Any]:
    relchain, RELATIONS, _ = _runner()
    cand = RELATIONS.get(name)
    if cand is None:
        raise SystemExit(f"unknown relation {name!r}; known: {sorted(RELATIONS)}")
    base, frozen = frozen_relation(cand.target.name, sets.n, sets.root)
    cref = relchain.instances_ref(cand, sets.n, root=sets.root)
    if cand.word_dtype != base.word_dtype:
        raise SystemExit(f"{name}: operand words {cand.word_dtype}, the frozen set's are {base.word_dtype}")
    canonical = (f"each side decoded by its own loader: frozen = relchain.instances({base.name!r}, {sets.n}) (the registered relation "
                 f"whose runner ref is the target's declared set over [0, {sets.n}): FROZEN_INSTANCES[target], or its synthetic stream by "
                 f"generator, seed and n), candidate = relchain.instances({name!r}, {sets.n}); x = the a "
                 f"operand words, W = the b operand words, as {np.dtype(base.word_dtype).str} bit words of the datatype "
                 f"({base.model_name}), little-endian; y = the chain-end FP32 accumulator word (the final recorded accumulator "
                 f"of the VU; the public claim is the relation's y_public of it), <u4 little-endian; index order VU-major, then K")
    return document(cand.target.name, frozen, cref, sets.load(base), sets.load(cand), canonical, tool_id())


def _relation_of(doc: Mapping[str, Any], n: int, root=None) -> str:
    relchain, RELATIONS, _ = _runner()
    hits = [r.name for r in RELATIONS.values()
            if r.target.name == doc.get("target") and relchain.instances_ref(r, n, root=root) == doc.get("candidate")]
    if len(hits) != 1:
        raise SystemExit(f"candidate ref {doc.get('candidate')} names {len(hits)} registered relations: {hits}")
    return hits[0]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verity_numerical.bench.instance_equiv", description=__doc__.split("\n\n")[0])
    ap.add_argument("--relation", action="append", default=[], help="a candidate relation (repeatable)")
    ap.add_argument("--vus", type=int, default=4096, help="the range [0, VUS) of both refs (4096, or a sweep point on a synthetic stream)")
    ap.add_argument("--root", default=None, help="the frozen bench-instances/v1 root (frozen-tier relations; default the fixture)")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("VY_CPU_THREADS", os.cpu_count() or 1)))
    ap.add_argument("--out", default=None, help="the JSON file (one --relation)")
    ap.add_argument("--out-dir", default=None, help="write <dir>/instance-equiv-<relation>-<vus>.json per relation")
    ap.add_argument("--check", nargs="+", default=[], metavar="FILE", help="re-derive each file and compare every field")
    args = ap.parse_args(argv)
    sets = _Sets(args.vus, args.procs, args.root)
    bad = 0
    if args.check:
        for path in args.check:
            doc = json.loads(Path(path).read_text())
            name = _relation_of(doc, args.vus, args.root)
            again = derive(name, sets)
            diff = [k for k in doc if k != "tool" and doc[k] != again.get(k)] + [k for k in again if k not in doc]
            ok = not diff
            bad += (not ok) or (not again["equal"])
            print(f"{path}: {name} {'reproduces' if ok else 'DIFFERS on ' + ', '.join(diff)}; equal={again['equal']}; "
                  f"file by {doc.get('tool')}, re-derived by {again['tool']}")
        return 1 if bad else 0
    if not args.relation or (args.out and len(args.relation) != 1) or not (args.out or args.out_dir):
        ap.error("give --relation (repeatable) with --out-dir, or one --relation with --out")
    for name in args.relation:
        doc = derive(name, sets)
        path = Path(args.out) if args.out else Path(args.out_dir) / f"instance-equiv-{name}-{args.vus}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2) + "\n")
        bad += not doc["equal"]
        print(f"{name}: equal={doc['equal']} " + " ".join(f"{k}={v['candidate_sha256'][:12]}/{v['frozen_sha256'][:12]}"
                                                          for k, v in doc["arrays"].items()) + f" -> {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
