"""The subcircuit catalog: one build is one artifact of kind ``verity/subcircuit-catalog/v1``, every census subcircuit in it.

    python -m verity_numerical.bench.subcircuit_catalog --records circuit-check.json --out DIR [--put [--preserve]] [ID ...]

The census (``census.SUBCIRCUITS``) is the list, and every entry of it is public: a model's pinned menu of unit types is
public, and which type a slot holds never enters the catalog. Developer circuits never enter it (an id the census lacks is
refused).

**Payload** (a tree): ``subcircuits/<census id>.boolean.json``, the subcircuit's Boolean export
(``verity_flock.boolean_export.subcircuit``, the website visualizer's dataset) at every variant, so an unchanged circuit is
the same blob in every build; and ``program-digests/<census id>.json``, an entry's full variant map where it has more than
:data:`INLINE_VARIANTS` variants (attention's key counts), which the store's 256 KiB cap on meta can't hold inline.

**Meta**: ``{schema, census_sha256, entries}``, one entry per census subcircuit in census order:

- ``id``, ``display``, ``template``, ``params``: the census's;
- ``descriptor_id``: the key ``verity_flock.circuit.program_digests`` gives its Definition at ``default_statics``
  (``verity.ml.circuits.codec``'s id; a silicon reference no core Definition wraps keeps its own key);
- ``default_statics``: ``{}``, or for statics that vary per instance (the census's ``variable_ranges``) the largest value of
  each; ``variants``: ``{count, varying: {name: [lo, hi]}}``, every value in each range a variant;
- ``program_digests``: ``{statics: program_digest}`` for every variant (statics as ``boolean_export.statics_key``), as META's
  ``program_digests`` would name each, or ``program_digests_file: {path, sha256}`` for a map held in the payload;
- ``export``: ``{path, sha256, bytes}`` of its Boolean export;
- ``counts``: ``{record: {report, target, sha512}, variants: {statics: {gates, lowered, units_per_instance, ands, rows,
  reason?}}}``, copied from its circuit-check record and never recomputed: the target's entry in the circuit-check report
  ``report`` (a ``circuit-check-report/v1`` artifact, also ``refs.circuit_check``), whose canonical JSON's SHA-512 is
  ``sha512``; per statics circuit-check checked, the Definition's word gates, and its C-Flock lowering's units per instance
  with each unit's ANDs and rows;
- ``missing``: ``{program_digests?, export?, counts?}``, the reason for each that the entry lacks (a model-only reference has
  no program, so no ``descriptor_id``, digests or export); an entry is never dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from . import census

KIND = "verity/subcircuit-catalog/v1"
RECORDS_KIND = "circuit-check-report/v1"
#: an entry with more variants than this holds its program digests in the payload
INLINE_VARIANTS = 64
#: the store's cap on an artifact's meta (`research.store.model`)
META_LIMIT = 256 * 1024
LOWERING_FIELDS = ("lowered", "units_per_instance", "ands", "rows", "reason")


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def statics_key(statics: dict) -> str:
    from verity_flock.boolean_export import statics_key as key

    return key(statics)


def census_ids() -> list[str]:
    return list(census.SUBCIRCUITS)


def _ranges(entry: Any) -> dict[str, tuple[int, int]]:
    vr = getattr(entry, "variable_ranges", None)
    return {k: (int(v.min), int(v.max)) for k, v in sorted(vars(vr).items())} if vr is not None else {}


def variants(entry: Any) -> list[dict[str, int]]:
    """Every instance statics of a census subcircuit: ``[{}]``, or each combination of its variable ranges' values."""
    r = _ranges(entry)
    return [dict(zip(r, vals)) for vals in itertools.product(*(range(lo, hi + 1) for lo, hi in r.values()))]


def default_statics(entry: Any) -> dict[str, int]:
    return {k: hi for k, (_lo, hi) in _ranges(entry).items()}


def record_digest(rec: dict) -> str:
    return hashlib.sha512(canonical(rec)).hexdigest()


def counts(report_art: str, rec: dict) -> dict:
    """An entry's counts: its circuit-check record cited by report, target and digest, and per statics checked the fields that
    record holds (its ``definitions`` and ``lowering`` are one per statics, in order)."""
    out = {}
    for d, low in zip(rec["definitions"], rec["lowering"]):
        out[statics_key(low["instance_statics"])] = {"gates": d.get("gates"),
                                                     **{k: low[k] for k in LOWERING_FIELDS if low.get(k) is not None}}
    return {"record": {"report": report_art, "target": rec["target"], "sha512": record_digest(rec)}, "variants": out}


def _records(report: dict) -> dict[str, dict]:
    return {t["target"]: t for t in report["targets"] if t.get("kind") == "template"}


def _counts_missing(rec: dict | None) -> str | None:
    if rec is None:
        return "no circuit-check record: the report has no target of this id"
    errors = [f["name"] for f in rec.get("failures", []) if f.get("code") == "error"]
    if errors or not rec.get("definitions"):
        return "circuit-check raised on it, so its record holds no counts: " + "; ".join(errors)[:500]
    return None


def export(sub: Any, statics: list[dict]) -> bytes:
    """The Boolean export of ``sub`` at every statics, as canonical JSON."""
    from verity_flock import boolean_export as BX

    return canonical(BX.subcircuit(sub, statics))


def build(out: Path, report: dict, report_art: str, ids: Iterable[str] | None = None,
          export: Callable[[Any, list[dict]], bytes] = export, log: Callable[[str], None] | None = None) -> dict:
    """Write the payload tree under ``out`` and return the meta, citing ``report`` (a circuit-check report, stored as
    ``report_art``) for the counts."""
    from verity.ml.circuits.codec import descriptor_id
    from verity_flock import circuit as FC

    from . import templates as TM

    ids = census_ids() if ids is None else list(ids)
    if unknown := [i for i in ids if i not in census.SUBCIRCUITS]:
        raise KeyError(f"not census subcircuits (developer circuits never enter the catalog): {unknown}")
    recs = _records(report)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for sid in ids:
        c = census.SUBCIRCUITS[sid]
        sts, dflt = variants(c), default_statics(c)
        e: dict[str, Any] = {"id": sid, "display": c.display, "template": c.template,
                             "params": json.loads(json.dumps(c.parameters, default=vars)), "default_statics": dflt,
                             "variants": {"count": len(sts), "varying": {k: list(v) for k, v in _ranges(c).items()}}}
        missing: dict[str, str] = {}
        sub = TM.parse_subcircuit_id(sid)
        blob = None
        try:
            (e["descriptor_id"], _), = FC.program_digests(sub, dflt).items()
            digests = {}
            for st in sts:
                (_k, v), = FC.program_digests(sub, st).items()
                digests[statics_key(st)] = v
        except Exception as x:  # noqa: BLE001  (a model-only reference has no program: recorded as missing)
            e.pop("descriptor_id", None)
            why = f"{type(x).__name__}: {x}"[:500]
            missing["program_digests"] = f"no program digest: {why}"
            missing["export"] = f"no Boolean export: no program to export: {why}"
        else:
            if len(sts) <= INLINE_VARIANTS:
                e["program_digests"] = digests
            else:
                path = f"program-digests/{sid}.json"
                data = canonical(digests)
                (out / path).parent.mkdir(parents=True, exist_ok=True)
                (out / path).write_bytes(data)
                e["program_digests_file"] = {"path": path, "sha256": hashlib.sha256(data).hexdigest()}
            try:
                descriptor_id(sub.definition(**dflt))
            except Exception as x:  # noqa: BLE001  (a silicon reference: no core Definition to export)
                missing["export"] = (f"no Boolean export: no core Definition wraps it (a silicon reference, "
                                     f"{e['descriptor_id']}): {type(x).__name__}: {x}")[:500]
        if "export" not in missing:
            try:
                blob = export(sub, sts)
            except Exception as x:  # noqa: BLE001  (no Boolean export: recorded as missing, never dropped)
                missing["export"] = f"no Boolean export: {type(x).__name__}: {x}"[:500]
        if blob is not None:
            path = f"subcircuits/{sid}.boolean.json"
            (out / path).parent.mkdir(parents=True, exist_ok=True)
            (out / path).write_bytes(blob)
            e["export"] = {"path": path, "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
        rec = recs.get(sid)
        if why := _counts_missing(rec):
            missing["counts"] = why
        else:
            e["counts"] = counts(report_art, rec)
        if missing:
            e["missing"] = missing
        entries.append(e)
        if log:
            log(f"{sid}: {len(sts)} variant(s)" + "".join(f"; missing {k}: {v[:120]}" for k, v in missing.items()))
    meta = {"schema": KIND, "census_sha256": hashlib.sha256((census.ROOT / "subcircuits.json").read_bytes()).hexdigest(),
            "entries": entries}
    if len(canonical(meta)) >= META_LIMIT:
        raise ValueError(f"the catalog's meta is {len(canonical(meta))} bytes, over the store's {META_LIMIT}: lower INLINE_VARIANTS")
    return meta


def put(store: Any, tree: Path, meta: dict, report_art: str) -> str:
    return store.put_artifact(KIND, meta, {"circuit_check": report_art}, tree=tree)


def put_report(store: Any, path: Path) -> str:
    report = json.loads(path.read_text())
    return store.put_artifact(RECORDS_KIND, {"targets": len(report["targets"]), "failures": len(report["summary"]["new_failures"])},
                              file=path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verity_numerical.bench.subcircuit_catalog", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="census subcircuit ids (default: the whole census)")
    ap.add_argument("--records", type=Path, required=True, help="the circuit-check report the counts cite (`circuit-check ID... --out`)")
    ap.add_argument("--out", type=Path, required=True, help="writes tree/ (the payload) and meta.json here")
    ap.add_argument("--put", action="store_true", help="put the report and the catalog in the store (catalog.json names both)")
    ap.add_argument("--preserve", action="store_true", help="with --put: push both to the remote and verify")
    a = ap.parse_args(argv)
    from research.store.local import LocalStore

    st = LocalStore() if a.put else None
    report_art = put_report(st, a.records) if st else "sha256:" + hashlib.sha256(a.records.read_bytes()).hexdigest()
    meta = build(a.out / "tree", json.loads(a.records.read_text()), report_art, a.ids or None, log=lambda s: print(s, flush=True))
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    missing = [e["id"] for e in meta["entries"] if "missing" in e]
    print(f"{len(meta['entries'])} entries, {len(missing)} with something missing")
    if st:
        art = put(st, a.out / "tree", meta, report_art)
        done = {"catalog": art, "circuit_check": report_art}
        (a.out / "catalog.json").write_text(json.dumps(done, indent=1) + "\n")
        if a.preserve:
            for x in (report_art, art):
                rep = st.push(x, preserve=True)
                if not rep["preserved"]:
                    print(f"{x}: NOT preserved: {'; '.join(rep['errors'])}", file=sys.stderr)
                    return 1
            done["preserved"] = True
            (a.out / "catalog.json").write_text(json.dumps(done, indent=1) + "\n")
        print(json.dumps(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
