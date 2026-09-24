"""One row per bench result: the per-lane ``summ.py`` / ``table.py`` scripts, once.

    python -m verity_numerical.bench.summary PATH... [--cols t.total_live,net.rtt_ms,...] [--md | --json] [--sort COL] [--best]
                                                     [--store ROOT | --no-refs]

PATH is a result file (``result.json``, a flat ``<tag>.json``, or a store manifest of kind ``bench-result/v1``), a run dir
holding a ``result.json``, or a directory of those, walked three levels down (``proofs/`` and dot-dirs skipped; a ``*.json``
that is not a result is skipped).  A row: the relation and key config (VUs, sub-batches, RS ``l``, pipeline depth, zk,
mode, authentication), ``t.total`` (the emitter's median over reps) and ``t.total_live``, the phase buckets as
:func:`tables.phases` defines them (a joint prints in the first bucket it covers, ``tables.JOINT_CONTINUATION`` in the others)
and ``other``, peak device / host memory, proof bytes, the result's ``validation.status``, its contract conformance
(:func:`contract.validate`), and the Rust verifier's batch verdict and union bound when ``<stem>_rust.json`` sits beside the
file or, for a ``result.json``, ``proofs/rust_batch.json`` / ``rust_batch.json`` in its dir.  ``--cols`` adds measurement
names, or dotted paths into the fingerprint or the result (``security.achieved_log2``, ``run_id``).  Columns empty in
every row are not printed; ``--json`` gives every field of every row, with the full phases and the contract problems.

``contended`` is the bench timing guard's verdict (``backends/direct/ligero/timing_guard.py``: another GPU process, CPU
throttling while the GPU was busy, or a saturated host during the run; ``contention.reasons`` has the why).  ``--best`` keeps
one row per like-for-like config (:data:`KEY`): the fastest ``t.total`` whose validation did not fail and that was not
contended; contended rows are refused (named on stderr) and counted in ``refused``.

``vs_ref`` is ``t.total`` over the best preserved reference of the same config: the fastest ``bench-result/v1`` in the
evidence store (``--store``, default ``$RESEARCH_STORE`` or ``~/.research/store``, read only when its catalog exists) that
has a remote replica, did not fail validation and was not contended; ``ref`` is its art id.  ``flag`` says ``slow-vs-ref``
when a row is more than :data:`SLOW` slower.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verity_numerical.bench import contract, tables

SHORT = dict(zip(contract.BUCKETS, ("wit", "enc", "arith", "lookup", "zk_add", "ser")))
SECONDS = (contract.T_TOTAL, contract.T_TOTAL_LIVE, *SHORT.values(), "other")
COLUMNS = ("label", "relation", "gpu", "VUs", "N_sub", "l", "pipe", "zk", "mode", "auth", *SECONDS, "dev_GiB", "host_GiB", "proof_MB",
           "validation", "contract", "rust", "bound", "contended", "refused", "vs_ref", "ref", "flag")
FORMATS = {**dict.fromkeys(SECONDS, "{:.4f}"), "dev_GiB": "{:.2f}", "host_GiB": "{:.2f}", "proof_MB": "{:.1f}", "bound": "2^{:.2f}",
           "vs_ref": "{:.2f}x"}
DEPTH = 3
#: A like-for-like config: rows with equal values here are the same measurement (``--best`` groups, ``vs_ref`` references).
KEY = ("relation", "gpu", "auth", "l", "pipe", "VUs", "zk", "mode")
#: ``slow-vs-ref`` beyond this fraction over the reference's ``t.total``.
SLOW = 0.15
#: The store's preserved (remotely replicated) artifacts of a kind, with their manifests.
REFS_SQL = "SELECT a.id, a.manifest_json FROM artifacts a WHERE a.kind = ? AND EXISTS (SELECT 1 FROM replicas r WHERE r.id = a.id)"


def _at(doc: Any, path: str) -> Any:
    v = contract._lookup(doc, path) if isinstance(doc, Mapping) else contract.MISSING
    return None if v is contract.MISSING else v


def load(path: Path) -> dict[str, Any] | None:
    """The bench result in ``path`` (a ``bench-result/v1`` store manifest gives its ``meta``), ``None`` when it holds none."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(doc, dict) and doc.get("kind") == tables.RESULT_KIND and isinstance(doc.get("meta"), dict):
        doc = doc["meta"]
    ok = isinstance(doc, dict) and isinstance(doc.get("workload_fingerprint"), Mapping) and isinstance(doc.get("measurements"), list)
    return doc if ok else None


def discover(path: Path, depth: int = DEPTH) -> list[Path]:
    """``path`` itself unless it is a directory; else its ``*.json`` but the Rust verdicts, and its subdirectories' to ``depth``."""
    if not path.is_dir():
        return [path]
    out: list[Path] = []
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix == ".json" and not (p.name.endswith("_rust.json") or p.name.startswith("rust_batch")):
            out.append(p)
        elif p.is_dir() and depth > 0 and p.name != "proofs" and not p.name.startswith("."):
            out += discover(p, depth - 1)
    return out


def rust_batch(path: Path) -> tuple[Path, Any] | None:
    """``(file, parsed or None)`` of the Rust verifier's batch verdict on the result at ``path``, ``None`` when there is none."""
    names = [path.stem + "_rust.json", *(("proofs/rust_batch.json", "rust_batch.json") if path.name == "result.json" else ())]
    c = next((path.parent / n for n in names if (path.parent / n).is_file()), None)
    try:
        return (c, json.loads(c.read_text())) if c else None
    except (OSError, ValueError):
        return c, None


def verdict(rb: Any) -> str:
    if not isinstance(rb, Mapping):
        return "unreadable"
    v = f"{'ACCEPT' if rb.get('batch_accepted') else 'REJECT'} {rb.get('accepted')}/{rb.get('n')}"
    if rb.get("system_pinned") is False:
        v += " UNPINNED"
    return v + (f" py-disagree={rb['python_disagree']}" if rb.get("python_disagree") else "")


def label(base: Path, path: Path) -> str:
    """``path`` relative to ``base``, without ``result.json`` / ``.json``: the run dir or file stem that names the row."""
    p = path.resolve()
    rel = p.relative_to(base) if p.is_relative_to(base) else Path(p.name)
    parts = rel.parts[:-1] if rel.name == "result.json" else (*rel.parts[:-1], rel.stem)
    return "/".join(parts) or p.parent.name


def config(fp: Mapping[str, Any], m: Mapping[str, Any]) -> dict[str, Any]:
    """The key config of a result (fingerprint ``fp``, measurements ``m``): the :data:`KEY` fields and ``N_sub``."""
    backend = str(_at(fp, "software.backend.name") or "")
    rel = re.search(r",\s*(\S+) relation", backend)
    return {"relation": rel.group(1) if rel else backend.split(" (")[0] or None, "gpu": tables.normalise_sku(_at(fp, "hardware.gpu.name")),
            "VUs": m.get("vu.count", fp.get("B")), "N_sub": fp.get("N_subbatches"), "l": _at(fp, "security.rs_l"),
            "pipe": _at(fp, "software.backend.pipeline"), "zk": _at(fp, "security.zk"), "mode": tables._mode_of(fp), "auth": fp.get("authentication")}


def key(r: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(json.dumps(r.get(k), sort_keys=True, default=str) for k in KEY)


def row(name: str, path: Path, result: Mapping[str, Any], cols: Sequence[str] = ()) -> dict[str, Any]:
    """The summary row of ``result`` (read from ``path``): raw values under the :data:`COLUMNS` keys and ``cols``, plus
    ``path``, ``run_id``, ``phases``, ``contract_problems``, ``rust_file`` and the timing guard's ``contention`` record."""
    fp, m, ph = result["workload_fingerprint"], tables._measurements(result), tables.phases(result)
    num = lambda *keys, s=1: next((m[k] / s for k in keys if tables._num(m.get(k))), None)  # noqa: E731
    problems, rb = contract.validate(result), rust_batch(path)
    r: dict[str, Any] = {
        "label": name, **config(fp, m),
        contract.T_TOTAL: ph["total"], contract.T_TOTAL_LIVE: m.get(contract.T_TOTAL_LIVE), **{SHORT[b]: v for b, v in ph["buckets"].items()},
    }
    for j in ph["joints"].values():
        first, *rest = j["covers"]
        r[SHORT[first]] = j["seconds"]
        r.update(dict.fromkeys((SHORT[b] for b in rest), tables.JOINT_CONTINUATION))
    r.update({"other": ph["other"], "dev_GiB": num("mem.peak_device_bytes", "peak_device_bytes", s=2**30),
              "host_GiB": num("mem.peak_host_bytes", s=2**30), "proof_MB": num("transcript.bytes", "proof_bytes", "proof.bytes_total", s=1e6),
              "validation": _at(result, "validation.status"), "contract": f"{len(problems)} problem{'s' * (len(problems) > 1)}" if problems else "ok",
              "rust": verdict(rb[1]) if rb else None, "bound": _at(rb[1], "batch_log2") if rb else None,
              "contended": _at(result, "contention.contended")})
    for c in cols:
        r.setdefault(c, m[c] if c in m else next((v for v in (_at(fp, c), _at(result, c)) if v is not None), None))
    r.update({"path": str(path), "run_id": result.get("run_id"), "phases": ph, "contract_problems": problems, "rust_file": str(rb[0]) if rb else None,
              "contention": result.get("contention")})
    return r


def references(index: Any, kind: str = tables.RESULT_KIND) -> dict[tuple[Any, ...], tuple[float, str]]:
    """``{key: (t.total, art id)}``: per config, the fastest preserved ``kind`` artifact in the store ``index`` (anything with
    ``.sql``) whose validation did not fail and that was not contended."""
    best: dict[tuple[Any, ...], tuple[float, str]] = {}
    for art, text in index.sql(REFS_SQL, (kind,)):
        try:
            meta = json.loads(text)["meta"]
            fp, m = meta["workload_fingerprint"], tables._measurements(meta)
        except (ValueError, TypeError, KeyError):
            continue
        t = m.get(contract.T_TOTAL)
        if not isinstance(fp, Mapping) or not tables._num(t) or t <= 0 or _at(meta, "validation.status") == "failed" or _at(meta, "contention.contended"):
            continue
        k = key(config(fp, m))
        if k not in best or t < best[k][0]:
            best[k] = (t, art)
    return best


def compare(rows: Sequence[dict[str, Any]], refs: Mapping[tuple[Any, ...], tuple[float, str]], slow: float = SLOW) -> None:
    """Set ``vs_ref`` / ``ref`` on every row with a reference, and ``flag = "slow-vs-ref"`` beyond ``slow``."""
    for r in rows:
        ref, t = refs.get(key(r)), r.get(contract.T_TOTAL)
        if ref is not None and tables._num(t):
            r.update({"vs_ref": t / ref[0], "ref": ref[1]})
            if t > (1 + slow) * ref[0]:
                r["flag"] = "slow-vs-ref"


def best(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(chosen, refused)``: per config the fastest row whose validation did not fail and that was not contended (``refused`` on
    it counts the group's contended rows), and every contended row."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    chosen, refused = [], [r for r in rows if r.get("contended")]
    for rs in groups.values():
        ok = [r for r in rs if r.get("validation") != "failed" and not r.get("contended") and tables._num(r.get(contract.T_TOTAL))]
        if ok:
            chosen.append({**min(ok, key=lambda r: r[contract.T_TOTAL]), "refused": sum(1 for r in rs if r.get("contended"))})
    return chosen, refused


def open_index(root: str | None) -> Any:
    """The evidence store's catalog, or ``None`` without ``research.store`` or a catalog (a pod): nothing is created."""
    try:
        from research.store.index import Index
        from research.store.local import store_root
    except ImportError:
        return None
    cat = store_root(root) / "catalog.sqlite"
    return Index(cat) if cat.is_file() else None


def fmt(key: str, v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if tables._num(v):
        return FORMATS.get(key, "{:.4g}" if isinstance(v, float) else "{}").format(v)
    return str(v)[:16] if key == "ref" else str(v)


def render(rows: Sequence[Mapping[str, Any]], cols: Sequence[str] = (), md: bool = False) -> str:
    """The rows as an aligned plain-text table (``md``: a markdown table), without the columns empty in every row."""
    keys = [k for k in dict.fromkeys((*COLUMNS, *cols)) if k == "label" or any(r.get(k) is not None for r in rows)]
    cells = [[fmt(k, r.get(k)) for k in keys] for r in rows]
    if md:
        out = tables._md_table(keys, cells)
    else:
        w = [max([len(k)] + [len(c[i]) for c in cells]) for i, k in enumerate(keys)]
        out = "\n".join("  ".join(s.ljust(w[i]) if i == 0 else s.rjust(w[i]) for i, s in enumerate(line)) for line in (keys, *cells))
    if any(tables.JOINT_CONTINUATION in c for c in cells):
        out += f"\n\n{tables.JOINT_CONTINUATION} = fused into the bucket on its left (a t.joint.* stage)"
    return out


def _sort_key(v: Any) -> tuple[int, Any, str]:
    return (0, v, "") if tables._num(v) else (1, 0, str(v)) if v is not None else (2, 0, "")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    ap.add_argument("--cols", default="", help="extra columns, comma-separated: measurement names or dotted fingerprint / result paths")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--md", action="store_true", help="a markdown table")
    out.add_argument("--json", action="store_true", help="every field of every row, as JSON")
    ap.add_argument("--sort", metavar="COL", help="sort by COL: numbers ascending, then text, then missing")
    ap.add_argument("--best", action="store_true", help="one row per config: the fastest that passed and was not contended")
    refs = ap.add_mutually_exclusive_group()
    refs.add_argument("--store", default=None, metavar="ROOT", help="evidence store for vs_ref (default $RESEARCH_STORE or ~/.research/store)")
    refs.add_argument("--no-refs", action="store_true", help="no vs_ref / slow-vs-ref (no store lookup)")
    ns = ap.parse_args(argv)
    cols = [c for c in ns.cols.split(",") if c] + ([ns.sort] if ns.sort and ns.sort not in COLUMNS else [])
    base = Path(os.path.commonpath([str((p if p.is_dir() else p.parent).resolve()) for p in ns.paths]))
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for arg in ns.paths:
        for p in discover(arg):
            if p.resolve() in seen:
                continue
            result = load(p)
            if result is None:
                if p == arg:
                    print(f"summary: {arg}: unreadable or not a bench result", file=sys.stderr)
                continue
            seen.add(p.resolve())
            try:
                rows.append(row(label(base, p), p, result, cols))
            except (ValueError, TypeError) as e:
                print(f"summary: {p}: {e}", file=sys.stderr)
    if not rows:
        print("summary: no bench results found", file=sys.stderr)
        return 1
    index = None if ns.no_refs else open_index(ns.store)
    if index is not None:
        compare(rows, references(index))
    if ns.best:
        rows, refused = best(rows)
        for r in refused:
            print(f"summary: --best refuses {r['label']}: contended ({'; '.join(_at(r, 'contention.reasons') or [])})", file=sys.stderr)
    if ns.sort:
        rows.sort(key=lambda r: _sort_key(r.get(ns.sort)))
    print(json.dumps(rows, indent=1, default=str) if ns.json else render(rows, cols, md=ns.md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
