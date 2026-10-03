"""A TP Build's derives with one template rank (``--build-relabel on``): the template's request derives write the other ranks'
(``build --emit-ranks``, `pipeline.tp.emit_ranks`), behind two checks that fail closed.

* **Step guard.** Every rank's step is derived as without the relabel, and one more step derive of the template writes every other
  rank's step from its derivation into ``build/relabel_check/rank<r>/build_step``.  Each is compared with the rank's real step
  (`compare_dirs`), and the template's extra derive with its real one.  A rank that differs gets its requests derived for real;
  the first difference (file, JSON path or byte offset) is in ``relabel_check/guard.json`` and the ``build.relabel_guard`` mark.
* **Requests.** The template's request derives write the ranks that passed into ``relabel_check/staged/rank<r>/``; a rank whose
  relabel failed is derived for real.  The highest relabeled rank (W - 1 unless its step failed the guard) of the first relabeled
  shape is also derived in full, in place, and compared with its relabel: the canary.  Only after it passes do the staged
  directories move into ``build/rank<r>/``.  A canary difference fails the Build, naming the file and the path, and moves nothing;
  a canary that could not compare (either side failed) moves nothing either, and every rank is derived for real.

A request directory reaches ``build/rank<r>/`` derived, or relabeled and checked, so a Build stopped half-way resumes by deriving
what is missing.  The template is rank 1 for W > 2 (rank 0's embedding shard starts at 0, which merges constants only rank 0
has), else rank 0.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
from typing import Any

from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.pipeline.tp import emit_ranks

SCHEMA = "verity-vllm/build-relabel-guard/v1"
#: what two derives of one rank may differ in (wall, memory, host, when), and a relabel's provenance
TELEMETRY = {
    "result.json": ("utc", "wall_secs_total", "max_rss_mb", "derive_secs", "export/secs", "target/host", "relabeled_from"),
    "construction_manifest.json": ("utc", "construction/export/secs", "construction/derive_secs", "construction/wall_secs_total",
                                   "construction/max_rss_mb", "target/host", "inputs_trace", "relabeled_from"),
}
#: a relabeled rank's `inputs.json` is the template's trace: what the process that wrote its bytes read
SKIPPED = ("inputs.json",)
#: a differing JSON file is parsed for its first differing path up to this size (larger: the byte offset only)
PARSE_MAX_BYTES = 64 << 20
_BLOCK = 1 << 20


def template_rank(world: int) -> int:
    return 1 if world > 2 else 0


# ------------------------------------------------------------------------------------------------ comparing two rank directories
def compare_dirs(real: str, relabel: str) -> dict | None:
    """None when `relabel` (a rank's directory written from the template) equals `real` (the rank derived), else the first
    difference: ``{file, path}`` (a JSON path), ``{file, offset, real, relabel}`` (a byte offset and the bytes around it), or
    ``{file, only_in}``.  Every file is compared byte for byte (``.gz`` decompressed) except `TELEMETRY`'s, compared as JSON
    without those keys, and `SKIPPED`."""
    names = sorted((set(os.listdir(real)) | set(os.listdir(relabel))) - set(SKIPPED))
    for name in names:
        a, b = os.path.join(real, name), os.path.join(relabel, name)
        if os.path.isdir(a) and os.path.isdir(b):
            d = compare_dirs(a, b)
            if d is not None:
                return {**d, "file": f"{name}/{d['file']}"}
            continue
        if not (os.path.isfile(a) and os.path.isfile(b)):
            return {"file": name, "only_in": "real" if os.path.exists(a) else "relabel"}
        d = _compare_file(name, a, b)
        if d is not None:
            return {"file": name, **d}
    return None


def _compare_file(name: str, a: str, b: str) -> dict | None:
    if name in TELEMETRY:
        x, y = json.load(open(a)), json.load(open(b))
        for p in TELEMETRY[name]:
            _drop(x, p)
            _drop(y, p)
        return None if x == y else _path_diff(x, y)
    d = _first_byte(a, b)
    if d is None:
        return None
    if name.endswith((".json", ".json.gz")) and max(os.path.getsize(a), os.path.getsize(b)) <= PARSE_MAX_BYTES:
        x, y = (json.load(_open(p)) for p in (a, b))
        return _path_diff(x, y) if x != y else {**d, "path": "(same values, other bytes: key order or spacing)"}
    return d


def _open(path: str):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _first_byte(a: str, b: str) -> dict | None:
    with _open(a) as fa, _open(b) as fb:
        off = 0
        while True:
            x, y = fa.read(_BLOCK), fb.read(_BLOCK)
            if x != y:
                i = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
                return {"offset": off + i, "real": x[max(0, i - 60):i + 60].decode(errors="replace"),
                        "relabel": y[max(0, i - 60):i + 60].decode(errors="replace")}
            if not x:
                return None
            off += len(x)


def _drop(doc: Any, path: str) -> None:
    *head, last = path.split("/")
    for k in head:
        doc = doc.get(k) if isinstance(doc, dict) else None
    if isinstance(doc, dict):
        doc.pop(last, None)


def _path_diff(x: Any, y: Any) -> dict:
    p = _first_path(x, y, "") or "/"
    return {"path": p, "real": _short(_at(x, p)), "relabel": _short(_at(y, p))}


def _first_path(a: Any, b: Any, at: str) -> str | None:
    """The first JSON path at which `a` and `b` differ (None: equal)."""
    if type(a) is not type(b):
        return at or "/"
    if isinstance(a, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if k not in a or k not in b:
                return f"{at}/{k}"
            p = _first_path(a[k], b[k], f"{at}/{k}")
            if p is not None:
                return p
        return None
    if isinstance(a, list):
        for i, (u, v) in enumerate(zip(a, b)):
            p = _first_path(u, v, f"{at}/{i}")
            if p is not None:
                return p
        return None if len(a) == len(b) else f"{at}/{min(len(a), len(b))}"
    return None if a == b else (at or "/")


def _at(doc: Any, path: str) -> Any:
    for k in [k for k in path.split("/") if k]:
        if isinstance(doc, dict):
            doc = doc.get(k, "<absent>")
        elif isinstance(doc, list) and k.isdigit():
            doc = doc[int(k)] if int(k) < len(doc) else "<absent>"
        else:
            return doc
    return doc


def _short(v: Any) -> str:
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= 300 else s[:300] + "..."


# ------------------------------------------------------------------------------------------------ the Build's derives
def derive(row, todo: list[tuple[Any, ...]], jobs: int) -> None:
    """`TpRow.build`'s derives of `todo` (``(rank, out, *args)``) with the relabel on (this module's docstring), `jobs` wide;
    what it did -> ``relabel_check/guard.json``.  Raises `Stop` (10) when the canary differs from its relabel."""
    W, t, chk = row.W, template_rank(row.W), f"{row.bdir}/relabel_check"
    steps = [j for j in todo if os.path.basename(j[1]) == "build_step"]
    by_shape: dict[str, dict[int, tuple[Any, ...]]] = {}
    for j in todo:
        if j not in steps:
            by_shape.setdefault(os.path.basename(j[1]), {})[j[0]] = j
    if not any(t in js and len(js) > 1 for js in by_shape.values()):
        row.derive_all(todo, jobs)       # no request of another rank can come from the template's
        return
    shutil.rmtree(chk, ignore_errors=True)
    os.makedirs(f"{chk}/rank{t}")
    others = [r for r in range(W) if r != t]
    row.derive_all(steps + [row.step(t, f"{chk}/rank{t}/build_step") + _emit_args(others, chk)], jobs)
    guard = step_guard(row.bdir, chk, W, t)
    for r, g in sorted(guard.items()):
        d = g.get("difference") or {}
        row.tl("build.relabel_guard", "mark", ("rank", r), ("ok", _tlv(g["ok"])), ("file", _tlv(d.get("file"))),
               ("path", _tlv(d.get("path", d.get("offset")))))
    passed = [r for r in others if guard[r]["ok"]] if guard[t]["ok"] else []
    doc: dict[str, Any] = {"schema": SCHEMA, "world": W, "template": t, "steps": {str(r): g for r, g in sorted(guard.items())},
                           "requests": {}, "canary": None}
    first: list[tuple[Any, ...]] = []
    for name, js in by_shape.items():
        rel = [r for r in sorted(js) if r != t and r in passed] if t in js else []
        doc["requests"][name] = {"relabel": rel, "derived": [r for r in sorted(js) if r not in rel], "emit_failed": {}}
        first += [js[t] + _emit_args(rel, f"{chk}/staged") if r == t and rel else js[r] for r in sorted(js) if r not in rel]
        if rel and doc["canary"] is None:
            doc["canary"] = {"rank": rel[-1], "request": name, "ok": None}
            first.append(js[rel[-1]])
    row.derive_all(first, jobs)
    for name, js in by_shape.items():
        rep = {x["rank"]: x for x in emit_ranks.read_report(js[t][1])["ranks"]} if t in js else {}
        for r in doc["requests"][name]["relabel"]:
            if not (rep.get(r) or {}).get("ok"):
                doc["requests"][name]["emit_failed"][str(r)] = (rep.get(r) or {}).get("error") or "no emit report"
    can = doc["canary"]
    if can is not None:
        failed = doc["requests"][can["request"]]["emit_failed"]
        if str(can["rank"]) in failed or not R.derived_ok(by_shape[can["request"]][can["rank"]][1]):
            can["ok"], can["why"] = None, "not compared: " + ("its relabel failed" if str(can["rank"]) in failed else "its derive failed")
        else:
            d = compare_dirs(by_shape[can["request"]][can["rank"]][1], f"{chk}/staged/rank{can['rank']}/{can['request']}")
            can["ok"], can["difference"] = d is None, d
        row.tl("build.relabel_canary", "mark", ("rank", can["rank"]), ("ok", _tlv(can["ok"])),
               ("file", _tlv((can.get("difference") or {}).get("file"))))
    _write(chk, doc)
    if can is not None and can["ok"] is False:
        d = can["difference"]
        row.verdict("build", "FAIL", f"relabel canary: rank {can['rank']} {can['request']} derived in full differs from its relabel at "
                                     f"{d['file']} {d.get('path', d.get('offset'))} (see {chk}/guard.json); no relabeled Program used")
        row.tl("stage.build", "end", ("ok", "FAIL"))
        row.tl("row", "end", ("ok", "false"), ("stopped_at", "build"))
        raise Stop(10)
    _place(row, chk, by_shape, doc, t, jobs)


def _emit_args(ranks: list[int], root: str) -> tuple[str, ...]:
    return ("--emit-ranks", ",".join(str(r) for r in ranks), "--emit-root", root)


def _tlv(v: Any) -> Any:
    return {None: "null", True: "true", False: "false"}[v] if v is None or isinstance(v, bool) else v


def step_guard(bdir: str, chk: str, world: int, t: int) -> dict[int, dict]:
    """Per rank: its real step against the template's relabel of it (the template: its real step against its extra derive)."""
    rep = {x["rank"]: x for x in emit_ranks.read_report(f"{chk}/rank{t}/build_step")["ranks"]}
    out: dict[int, dict] = {}
    for r in range(world):
        real, rel = f"{bdir}/rank{r}/build_step", f"{chk}/rank{r}/build_step"
        if not R.derived_ok(real):
            out[r] = {"ok": False, "why": "its step derive failed"}
        elif not R.derived_ok(rel):
            out[r] = {"ok": False, "why": "the template's step derive failed" if r == t else
                      f"its relabel failed: {(rep.get(r) or {}).get('error') or 'no emit report'}"}
        else:
            d = compare_dirs(real, rel)
            out[r] = {"ok": d is None, "difference": d} if d else {"ok": True}
    return out


def _place(row, chk: str, by_shape: dict, doc: dict, t: int, jobs: int) -> None:
    """The staged relabels into ``build/rank<r>/`` once the canary passed; every other rank of a relabeled shape derived for real."""
    can, todo = doc["canary"], []
    for name, js in by_shape.items():
        q = doc["requests"][name]
        for r in q["relabel"]:
            dst, src = js[r][1], f"{chk}/staged/rank{r}/{name}"
            if (r, name) == (can["rank"], can["request"]):
                continue                                   # the canary: derived in full, in place
            if can["ok"] and str(r) not in q["emit_failed"]:
                shutil.rmtree(dst, ignore_errors=True)
                os.replace(src, dst)
                row.log(f"relabel {name} rank{r} from rank{t}: placed")
            else:
                todo.append(js[r])
                q.setdefault("derived_after", []).append(r)
    _write(chk, doc)
    if todo:
        row.log(f"relabel: {len(todo)} request derive(s) for real ({'the canary did not compare' if not can['ok'] else 'their relabel failed'})")
        row.derive_all(todo, jobs)


def _write(chk: str, doc: dict) -> None:
    json.dump(doc, open(f"{chk}/guard.json", "w"), indent=1, default=str)


__all__ = ["PARSE_MAX_BYTES", "SCHEMA", "SKIPPED", "TELEMETRY", "compare_dirs", "derive", "step_guard", "template_rank"]
