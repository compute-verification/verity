"""[A9 contract 2026-09-10] The sampled population WITHOUT input gates -- ``veritor/a9/population-v5/v1``, construction
``legacy-minus-inputs`` -- until A1's population (input gates absent by construction) lands.

Inputs are not VUs (docs/PROTOCOL.md; PROTOCOL-CONFORMANCE R2): a legacy population (``population/v2`` compact rows of the validated
P1/cu=whole plan, or D4's rule-generated kind table for shapes no plan exists at) is read and every row / kind whose ``fn`` is ``Input``
is REMOVED; VU indices are renumbered contiguously (RU membership, kinds, gate intervals of the retained units are unchanged).  The
result is a NEW population identity: ``partition_digest`` = sha256 over (construction, basis partition digest, excluded kinds,
retained rows) with ``partition_digest_kind = "a9/legacy-minus-inputs/v1"``; the basis (file, content sha256, digest, counts, what was
excluded) is recorded in ``basis`` so the derivation is checkable.  This is a labelled interim population, not A1's: the instance that
names it says so (``instance_v5``) and the certificate carries the label.

Views (``e2e.partition_sample.PopulationView``) give the sampler the ranking primitives; :func:`install` puts :func:`load_population_v5`
in front of A3's ``load_population`` (D4's ``population_rules.install`` pattern), so stage1 / stage2 / verify and their ``bind`` and
pin checks run A3's code unchanged over the v5 view.

    python -m record_v5.population_v5 from-v2 population.compact.json --out DIR       # DIR/population.v5.json
    python -m record_v5.population_v5 from-rules population.rules.json --out DIR
    python -m record_v5.population_v5 check population.v5.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

from record_v5 import deps

deps.install()

from e2e import partition_sample as ps  # noqa: E402
from e2e.population_v2 import POPULATION_V2, ROW_FIELDS, CompactPopulation, Kind, PopulationError  # noqa: E402

SCHEMA = "veritor/a9/population-v5/v1"
CONSTRUCTION = "legacy-minus-inputs"
PARTITION_DIGEST_KIND = "a9/legacy-minus-inputs/v1"
#: [E10 EOD 2026-09-11] the partition identities a population.v5.json may carry: v1 (attempt 6) and E6's v2 (coverage table, literals,
#: token-selection form in the identity; e2e/partitions/legacy_minus_inputs_v2.py).  load_population_v5 accepts both (the v2 digest
#: re-derives through E6's `check`); the verifier's `record_v5` instance format should admit an instance declaring either
#: (e2e/verifier_v3/reverify.install_instance_format -> partition_sample.check_instance_format).
PARTITION_DIGEST_KINDS = (PARTITION_DIGEST_KIND, "a9/legacy-minus-inputs/v2")
COUNT_BASIS = {"rows": "validated_plan_minus_inputs", "rules": "analytical_census_minus_inputs"}
EXCLUDED_FN = ("Input",)
FILE = "population.v5.json"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canon(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def _partition_digest(basis_digest: str, excluded: list[str], payload: Any) -> str:
    return _sha(_canon({"kind": PARTITION_DIGEST_KIND, "construction": CONSTRUCTION, "basis_partition_digest": basis_digest, "excluded_kinds": excluded, "payload": payload}))


# -- building ----------------------------------------------------------------------------------------------------------------------


def from_v2(path: Path) -> dict[str, Any]:
    """``population/v2`` compact rows minus the Input rows.  O(rows)."""

    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != POPULATION_V2:
        raise PopulationError(f"{path}: schema {doc.get('schema')!r} != {POPULATION_V2!r}")
    pop = CompactPopulation(doc, path=path)
    pop.check(digest=False)
    kinds = pop.kinds
    excluded_ids = {k.id for k in kinds if k.fn in EXCLUDED_FN}
    excluded_names = sorted(k.name for k in kinds if k.id in excluded_ids)
    kept_kinds = [k for k in kinds if k.id not in excluded_ids]
    remap = {k.id: i for i, k in enumerate(kept_kinds)}
    rows, nxt, removed_vus, removed_gates = [], 0, 0, 0
    for r in pop.rows:  # [first_vu, count, ru, kind, lo, stride, unit_size]
        if r[3] in excluded_ids:
            removed_vus += r[1]
            removed_gates += r[1] * r[6]
            continue
        rows.append([nxt, r[1], r[2], remap[r[3]], r[4], r[5], r[6]])
        nxt += r[1]
    new_kinds = [Kind(i, k.fn, k.form, k.primitive, k.out_bits, k.size).to_json() for i, k in enumerate(kept_kinds)]
    validation = None
    cv = path.parent / "compact_validation.json"
    if cv.exists():  # rec_p1_population.py writes COMPACT-P1's verdict verbatim beside the rows
        art = json.loads(cv.read_text())
        if art.get("partition_digest") not in (None, pop.partition_digest):
            raise PopulationError(f"{cv}: partition_digest disagrees with {path.name}")
        validation = {"artifact": str(cv), "status": art.get("status"), "scope": art.get("scope") or art.get("codes"), "source": art.get("verdict_source", "compact_validation.json"),
                      "sha256": _sha(cv.read_bytes())}
    out = {
        "schema": SCHEMA, "construction": CONSTRUCTION, "basis_kind": "rows",
        "basis": {"schema": POPULATION_V2, "path": str(path), "content_sha256": pop.content_sha256, "partition_digest": pop.partition_digest,
                  "plan_digest": pop.plan_digest, "program_digest": pop.program_digest, "vu_count": pop.vu_count, "ru_count": pop.ru_count, "validation": validation,
                  "excluded": {"fn": list(EXCLUDED_FN), "kinds": excluded_names, "vus": removed_vus, "gates": removed_gates, "rows": len(pop.rows) - len(rows)}},
        "vu_count": nxt, "ru_count": pop.ru_count, "gates": pop.gates, "instances": pop.instances,
        "plan_digest": pop.plan_digest, "plan_digest_kind": pop.plan_digest_kind, "program_digest": pop.program_digest,
        "partition_digest": _partition_digest(pop.partition_digest, excluded_names, rows), "partition_digest_kind": PARTITION_DIGEST_KIND,
        "kinds": {"entries": new_kinds}, "row_fields": list(ROW_FIELDS), "rows": rows, "cu_intervals": pop.cu_intervals,
        "label": "legacy population minus Input kinds (interim, labelled; A1's population replaces it)",
    }
    out["content_sha256"] = _sha(_canon({k: v for k, v in out.items() if k != "content_sha256"}))
    return out


def from_rules(path: Path) -> dict[str, Any]:
    """D4's rule-generated kind table (``veritor/rec6k/population-rules/v1``) minus the Input kinds (MODELED counts stay MODELED)."""

    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != "veritor/rec6k/population-rules/v1":
        raise PopulationError(f"{path}: schema {doc.get('schema')!r} is not D4's rule population")
    kept = [k for k in doc["kinds"] if k["fn"] not in EXCLUDED_FN]
    excluded = sorted(k["kind"] for k in doc["kinds"] if k["fn"] in EXCLUDED_FN)
    removed = sum(int(k["count"]) for k in doc["kinds"] if k["fn"] in EXCLUDED_FN)
    acc, kinds = 0, []
    for k in kept:  # already in sorted key order (D4 checks it)
        kinds.append({"kind": k["kind"], "fn": k["fn"], "form": k["form"], "count": int(k["count"]), "offset": acc})
        acc += int(k["count"])
    out = {
        "schema": SCHEMA, "construction": CONSTRUCTION, "basis_kind": "rules",
        "basis": {"schema": doc["schema"], "path": str(path), "content_sha256": _sha(raw), "partition_digest": doc["partition_digest"], "plan_digest": doc["plan_digest"],
                  "program_digest": doc["program_digest"], "vu_count": int(doc["vu_count"]), "ru_count": int(doc["ru_count"]),
                  "construction": doc.get("construction"), "label": doc.get("label"), "rules": doc.get("rules"),
                  "excluded": {"fn": list(EXCLUDED_FN), "kinds": excluded, "vus": removed}},
        "LP": doc["LP"], "STEPS": doc["STEPS"], "vu_count": acc, "ru_count": int(doc["ru_count"]), "instances": int(doc.get("instances", 1)),
        "plan_digest": doc["plan_digest"], "plan_digest_kind": doc.get("plan_digest_kind_note"), "program_digest": doc["program_digest"],
        "program_digest_kind": doc.get("program_digest_kind"),
        "partition_digest": _partition_digest(doc["partition_digest"], excluded, [[k["kind"], k["count"]] for k in kinds]), "partition_digest_kind": PARTITION_DIGEST_KIND,
        "kinds": {"table": kinds},
        "label": "rule-generated (MODELED) kind table minus Input kinds (interim, labelled; no plan exists at this shape)",
    }
    out["content_sha256"] = _sha(_canon({k: v for k, v in out.items() if k != "content_sha256"}))
    return out


def write(doc: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
    return path


# -- views -----------------------------------------------------------------------------------------------------------------------


class V5RowsView(ps.PopulationView):
    """Rows basis: lookups by :class:`CompactPopulation` over the filtered rows (schema swapped for the constructor only)."""

    schema = SCHEMA

    def __init__(self, doc: dict[str, Any], *, path: Path | None = None) -> None:
        self.doc, self.path = doc, path
        inner = dict(doc)
        inner["schema"] = POPULATION_V2
        self.pop = CompactPopulation(inner, path=path)
        self.vu_count, self.ru_count = int(doc["vu_count"]), int(doc["ru_count"])
        self.plan_digest, self.program_digest, self.partition_digest = doc["plan_digest"], doc["program_digest"], doc["partition_digest"]
        self.instances, self.content_sha256 = int(doc.get("instances", 1)), doc["content_sha256"]
        # rows tile [0, vu_count) contiguously
        nxt = 0
        for r in doc["rows"]:
            if r[0] != nxt:
                raise PopulationError(f"v5 rows do not tile: row starts at {r[0]}, expected {nxt}")
            nxt += r[1]
        if nxt != self.vu_count:
            raise PopulationError(f"v5 rows sum {nxt} != vu_count {self.vu_count}")
        if any(k["fn"] in EXCLUDED_FN for k in doc["kinds"]["entries"]):
            raise PopulationError("v5 population still contains an Input kind")

    def ru_of(self, vu: int) -> int:
        return self.pop.ru_of(vu)

    def ru_size(self, ru: int) -> int:
        return self.pop.ru_size(ru)

    def vu_at(self, ru: int, k: int) -> int:
        return self.pop.vu_at(ru, k)

    def kind_of(self, vu: int) -> dict[str, Any]:
        k = self.pop.kind_of(vu)
        return {**k.to_json(), "kind": k.name}

    def gate_interval(self, vu: int) -> tuple[int, int]:
        return self.pop.gate_interval(vu)

    def kind_counts(self) -> dict[str, int]:
        return self.pop.kind_counts()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "construction": CONSTRUCTION, "partition_digest_kind": PARTITION_DIGEST_KIND, "basis": self.doc["basis"]}


class V5RulesView(ps.PopulationView):
    """Rules basis (one RU per instance; VU rank = kind offset + index within kind, kinds in sorted key order) -- D4's RuleView shape."""

    schema = SCHEMA

    def __init__(self, doc: dict[str, Any], *, path: Path | None = None) -> None:
        self.doc, self.path = doc, path
        self.vu_count, self.ru_count = int(doc["vu_count"]), int(doc["ru_count"])
        if self.ru_count != 1:
            raise PopulationError("rules population: one RU per instance")
        self.plan_digest, self.program_digest, self.partition_digest = doc["plan_digest"], doc["program_digest"], doc["partition_digest"]
        self.instances, self.content_sha256 = int(doc.get("instances", 1)), doc["content_sha256"]
        table = doc["kinds"]["table"]
        self._kinds = [k["kind"] for k in table]
        self._counts = [int(k["count"]) for k in table]
        self._offsets = [int(k["offset"]) for k in table]
        ends, acc = [], 0
        for c in self._counts:
            acc += c
            ends.append(acc)
        self._ends = ends
        if acc != self.vu_count or any(o != (ends[i - 1] if i else 0) for i, o in enumerate(self._offsets)):
            raise PopulationError("rules population: offsets / counts do not tile vu_count")
        if any(k.split("|")[0] in EXCLUDED_FN for k in self._kinds):
            raise PopulationError("v5 population still contains an Input kind")

    def ru_of(self, vu: int) -> int:
        if not 0 <= vu < self.vu_count:
            raise IndexError(vu)
        return 0

    def ru_size(self, ru: int) -> int:
        if ru != 0:
            raise IndexError(ru)
        return self.vu_count

    def vu_at(self, ru: int, k: int) -> int:
        if ru != 0 or not 0 <= k < self.vu_count:
            raise IndexError((ru, k))
        return k

    def kind_index(self, vu: int) -> tuple[int, int]:
        if not 0 <= vu < self.vu_count:
            raise IndexError(vu)
        b = bisect_right(self._ends, vu)
        return b, vu - self._offsets[b]

    def kind_of(self, vu: int) -> dict[str, Any]:
        b, j = self.kind_index(vu)
        k = self._kinds[b]
        return {"fn": k.split("|")[0], "form": k.split("|", 1)[1] if "|" in k else "", "kind": k, "index_within_kind": j, "count": self._counts[b]}

    def gate_interval(self, vu: int) -> None:
        return None

    def kind_counts(self) -> dict[str, int]:
        return dict(zip(self._kinds, self._counts))

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "construction": CONSTRUCTION, "partition_digest_kind": PARTITION_DIGEST_KIND, "basis": self.doc["basis"], "kinds": len(self._kinds)}


def view(doc: dict[str, Any], *, path: Path | None = None) -> ps.PopulationView:
    if doc.get("schema") != SCHEMA:
        raise PopulationError(f"schema {doc.get('schema')!r} != {SCHEMA!r}")
    if doc.get("content_sha256") != _sha(_canon({k: v for k, v in doc.items() if k != "content_sha256"})):
        raise PopulationError("v5 population content sha256 does not re-derive")
    return V5RowsView(doc, path=path) if doc.get("basis_kind") == "rows" else V5RulesView(doc, path=path)


_orig_load_population = ps.load_population


def load_population_v5(path: Path, *, check: str = "fast", instances: int | None = None) -> ps.PopulationView:
    path = Path(path)
    doc = json.loads(path.read_bytes())
    if doc.get("schema") != SCHEMA:
        return _orig_load_population(path, check=check, instances=instances)
    v = view(doc, path=path)
    if check == "full" and doc.get("basis_kind") == "rows":
        if doc.get("partition_digest_kind") == "a9/legacy-minus-inputs/v2":
            # [E10 EOD 2026-09-11] E6's v2 identity (e2e/partitions/legacy_minus_inputs_v2.py): the same schema and rows, the digest
            # is sha256 over {rows' v1 digest, kind-table digest, coverage-table sha256, literals, token_select}; its module re-derives it
            from e2e.partitions import legacy_minus_inputs_v2 as v2  # E6's file, read-only

            problems = v2.check(doc, None)
            if problems:
                raise PopulationError("v2 partition digest does not re-derive: " + "; ".join(problems[:3]))
        else:
            # re-derive the v5 partition digest from the rows (the analogue of population/v2's O(N_V) re-derivation; here O(rows))
            expect = _partition_digest(doc["basis"]["partition_digest"], doc["basis"]["excluded"]["kinds"], doc["rows"])
            if expect != doc["partition_digest"]:
                raise PopulationError("v5 partition digest does not re-derive from the rows")
    if instances is not None and instances != v.instances:
        v = ps.ReplicatedView(v, instances)
    return v


def install() -> None:
    """Make ``e2e.partition_sample`` (and everything importing its ``load_population`` by attribute) read the v5 schema.
    Process-wide: called by the runner's entry points (``main``), never at import time (A7 finding 22:05Z: importing a record_v5
    module from a test must not change what A3's tests see); tests use :func:`installed` (a context manager with teardown)."""

    ps.load_population = load_population_v5


def uninstall() -> None:
    ps.load_population = _orig_load_population


@contextlib.contextmanager
def installed():
    """``with population_v5.installed(): ...`` -- the v5 loader for the block, the original restored afterwards."""

    prev = ps.load_population
    install()
    try:
        yield
    finally:
        ps.load_population = prev


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("from-v2", "from-rules"):
        b = sub.add_parser(name)
        b.add_argument("path")
        b.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("path")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            v = load_population_v5(Path(a.path), check="full")
            print(f"[population_v5] OK {a.path}: N_V={v.vu_count} N_R={v.ru_count} partition {v.partition_digest[:16]} basis excluded {v.doc['basis']['excluded']}")
            return 0
        doc = from_v2(Path(a.path)) if a.cmd == "from-v2" else from_rules(Path(a.path))
        p = write(doc, Path(a.out) / FILE)
        print(f"[population_v5] {doc['construction']} ({doc['basis_kind']}): N_V={doc['vu_count']} (basis {doc['basis']['vu_count']}, excluded {doc['basis']['excluded']['vus']} "
              f"in {doc['basis']['excluded']['kinds']}) N_R={doc['ru_count']} partition {doc['partition_digest'][:16]} -> {p}")
        return 0
    except PopulationError as e:
        print(f"[population_v5] REFUSED: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
