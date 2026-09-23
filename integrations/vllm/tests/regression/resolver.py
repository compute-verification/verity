"""Where a reference record's bytes are: `artifact://r19/<row>/<rel>` -> a local file, sha256-checked against `fixtures.toml`.

`fixtures.toml` (this directory; small, committed) lists every reference file this harness may read: `[rows.<row>.files."<rel>"]`
with `sha256`, `bytes`, `role` (record | program | manifest | ...) and, per row, the places the bytes are known to live (`local.<machine>`
= a row dir on that pod, `artifacts.records` / `artifacts.programs` = `art:` ids in the data store once put, `oracle` = the vault
extraction dir).  Nothing large is in Git: the file is an index, the bytes come from one of these, in this order:

    1. $VERITY_REGRESSION_ROWS_ROOT/<row>/<rel>          a directory of row dirs (tp2: /workspace/regress/rows -> symlinks to the live
                                                          row dirs under /workspace/cp/sweep/...; vault: /vault/tmp/oracle)
    2. [rows.<row>.local.*]/<rel>                         a recorded row-dir path that exists on THIS machine (tp2's live dirs)
    3. store_io.fetch([rows.<row>.artifacts.<role>])/<rel> the data API (records tree / programs tree), when those ids are set
    4. nothing -> the check is SKIPPED by name ("input <rel> not resolvable here"); never a silent pass

A resolved file whose recorded sha256 is known and whose size is <= VERIFY_LIMIT is hashed and must match: a reference that differs
from the frozen index is `InputMismatch`, a failure, not a skip -- the oracle changed under the test.  Files above the limit (the
multi-GB fold records, 100+ MB manifests) are checked by size only unless VERITY_REGRESSION_VERIFY_ALL=1.

`Row` is the handle the checks receive: `row.path(rel)`, `row.json(rel)`, `row.workload()` (the in-repo `workloads/<row>.json`),
`row.programs_root()` (a dir holding `build_request*/instances.json.gz`, or None), `row.candidate(rel)` (the v2 output under
$VERITY_REGRESSION_CANDIDATE/<row>/<rel>, or None: self-check mode), `row.decisions(check)` (the labelled decisions of
`[[rows.<row>.decisions]]` naming that check; `sanction()` consults them when a result differs from v1).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
VLLM_ROOT = HERE.parents[1]                     # integrations/vllm
FIXTURES = HERE / "fixtures.toml"
EXPECTED_DIR = HERE / "expected"
VERIFY_LIMIT = 64 * 1024 * 1024

ROWS_ROOT_ENV = "VERITY_REGRESSION_ROWS_ROOT"
CANDIDATE_ENV = "VERITY_REGRESSION_CANDIDATE"


class InputMismatch(AssertionError):
    """A resolved reference file does not have the sha256 / size `fixtures.toml` froze for it."""


class NotResolvable(LookupError):
    """No replica of the file is reachable from this machine (the check skips by name)."""


def load_fixtures() -> dict[str, Any]:
    with open(FIXTURES, "rb") as f:
        return tomllib.load(f)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class Row:
    def __init__(self, key: str, spec: dict[str, Any], *, fixtures: dict[str, Any]):
        self.key = key
        self.spec = spec
        self.number = int(spec["row"])
        self.cls = spec.get("class")
        self.files: dict[str, dict[str, Any]] = spec.get("files") or {}
        self._fixtures = fixtures
        self._fetched: dict[str, Path] = {}

    # ---- locating -------------------------------------------------------------------------------------------------------------
    def _roots(self) -> list[Path]:
        roots: list[Path] = []
        env = os.environ.get(ROWS_ROOT_ENV)
        if env:
            roots.append(Path(env) / self.key)
        for _machine, p in sorted((self.spec.get("local") or {}).items()):
            roots.append(Path(p))
        oracle = self.spec.get("oracle")
        if oracle:
            roots.append(Path(oracle))
        return [r for r in roots if r.is_dir()]

    def _store_root(self, role: str) -> Path | None:
        art = (self.spec.get("artifacts") or {}).get(role)
        if not art:
            return None
        if art not in self._fetched:
            from . import store_io
            try:
                self._fetched[art] = store_io.fetch(art)
            except Exception as e:  # noqa: BLE001 -- the store not being reachable is a skip, not an error
                raise NotResolvable(f"{self.key}: store fetch of {role} {art} failed: {e}") from e
        return self._fetched[art]

    def path(self, rel: str, *, verify: bool = True) -> Path:
        """The local path of `artifact://r19/<row>/<rel>` (see the module docstring for the order); InputMismatch / NotResolvable."""
        for root in self._roots():
            p = root / rel
            if p.is_file():
                return self._checked(rel, p) if verify else p
        for role in ("records", "programs"):
            root = self._store_root(role)
            if root is not None and (root / rel).is_file():
                p = root / rel
                return self._checked(rel, p) if verify else p
        raise NotResolvable(f"{self.key}: {rel} not resolvable here (roots tried: {[str(r) for r in self._roots()]}; artifacts: {self.spec.get('artifacts')})")

    def _checked(self, rel: str, p: Path) -> Path:
        rec = self.files.get(rel)
        if not rec:
            return p                                       # not frozen in the index: used as found (the check records that it did)
        size = p.stat().st_size
        if rec.get("bytes") is not None and int(rec["bytes"]) != size:
            raise InputMismatch(f"{self.key}/{rel}: {size} bytes on disk, fixtures.toml froze {rec['bytes']}")
        if rec.get("sha256") and (size <= VERIFY_LIMIT or os.environ.get("VERITY_REGRESSION_VERIFY_ALL") == "1"):
            got = sha256_file(p)
            if got != rec["sha256"]:
                raise InputMismatch(f"{self.key}/{rel}: sha256 {got[:16]} on disk != frozen {rec['sha256'][:16]}")
        return p

    def has(self, rel: str) -> bool:
        try:
            self.path(rel, verify=False)
            return True
        except NotResolvable:
            return False

    def json(self, rel: str) -> Any:
        p = self.path(rel)
        if rel.endswith(".gz"):
            with gzip.open(p, "rt") as f:
                return json.load(f)
        with open(p) as f:
            return json.load(f)

    def jsonl(self, rel: str) -> list[Any]:
        with open(self.path(rel)) as f:
            return [json.loads(line) for line in f if line.strip()]

    def workload(self) -> Path:
        """The declared workload manifest (in-repo `integrations/vllm/workloads/<row>.json`) -- the same file row_pod.sh's `$WL` was."""
        p = VLLM_ROOT / "workloads" / f"{self.key}.json"
        if not p.is_file():
            raise NotResolvable(f"{self.key}: workload manifest {p} not in this tree")
        return p

    def programs_root(self) -> Path | None:
        """A directory holding the row's request Programs (`build_request*/instances.json.gz`) -- the live row dir or the fetched
        programs tree -- or None when none is reachable."""
        for root in self._roots():
            if any(root.glob("build_request*/instances.json.gz")):
                return root
        try:
            root = self._store_root("programs")
        except NotResolvable:
            root = None
        if root is not None and any(root.glob("build_request*/instances.json.gz")):
            return root
        return None

    def candidate(self, rel: str) -> Path | None:
        """The v2 candidate's file for this row (`$VERITY_REGRESSION_CANDIDATE/<row>/<rel>`), None in self-check mode."""
        root = os.environ.get(CANDIDATE_ENV)
        if not root:
            return None
        p = Path(root) / self.key / rel
        return p if p.is_file() else None

    def candidate_root(self) -> Path | None:
        root = os.environ.get(CANDIDATE_ENV)
        return (Path(root) / self.key) if root and (Path(root) / self.key).is_dir() else None

    # ---- expected -------------------------------------------------------------------------------------------------------------
    def expected(self) -> dict[str, Any]:
        p = EXPECTED_DIR / f"{self.key}.json"
        if not p.is_file():
            return {}
        with open(p) as f:
            return json.load(f)

    def decisions(self, check: str) -> list[dict[str, Any]]:
        """The labelled decisions on this row for `check` (`[[rows.<row>.decisions]]` in fixtures.toml, design rule Draft 3 §7: a
        difference from the v1 reference is sanctioned only by a recorded decision naming the finding or the changed policy)."""
        return [d for d in (self.spec.get("decisions") or []) if isinstance(d, dict) and d.get("check") == check]


def rows(selection: str | None = None) -> list[Row]:
    """The reference rows of fixtures.toml, by row number; `selection` = 'green' | 'by-name' | None (all)."""
    fx = load_fixtures()
    out = [Row(k, v, fixtures=fx) for k, v in (fx.get("rows") or {}).items()]
    if selection:
        out = [r for r in out if r.spec.get("selection") == selection]
    return sorted(out, key=lambda r: r.number)


def canonical(obj: Any) -> str:
    """Canonical JSON for the diff (sorted keys, compact, no NaN)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str)


NOT_APPLICABLE = "not_applicable"


def not_applicable(why: str) -> dict[str, str]:
    """A check's declaration that a leaf the record carries does not apply to the candidate (a Build-only candidate has no Commit to key a
    binding by): `{"not_applicable": why}` in `actual`.  `diff` does not count it as a difference and `is_not_applicable` finds them so the
    test records what was declared inapplicable, by key and reason."""
    return {NOT_APPLICABLE: why}


def is_not_applicable(x: Any) -> bool:
    return isinstance(x, dict) and set(x) == {NOT_APPLICABLE} and isinstance(x[NOT_APPLICABLE], str)


def not_applicable_keys(actual: Any, path: str = "$") -> list[str]:
    if is_not_applicable(actual):
        return [f"{path}: {actual[NOT_APPLICABLE]}"]
    if isinstance(actual, dict):
        return [s for k, v in actual.items() for s in not_applicable_keys(v, f"{path}.{k}")]
    return []


def diff(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Every leaf of `expected` that `actual` lacks or disagrees on (keys `actual` has beyond `expected` are not a difference: the
    expected document is the contract, lifted from the record; a leaf `actual` declares `not_applicable(why)` is not a difference either
    -- the declaration is recorded by the test).  Lists are compared whole."""
    out: list[str] = []
    if is_not_applicable(actual):
        return out
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected object, got {type(actual).__name__}"]
        for k, ev in expected.items():
            if k not in actual:
                out.append(f"{path}.{k}: missing in actual (expected {canonical(ev)[:120]})")
            else:
                out.extend(diff(ev, actual[k], f"{path}.{k}"))
        return out
    if canonical(expected) != canonical(actual):
        out.append(f"{path}: expected {canonical(expected)[:160]} != actual {canonical(actual)[:160]}")
    return out


DECISION_FIELDS = ("check", "since", "finding", "reason", "expected")


def sanction(row: Row, check: str, act: dict[str, Any], problems: list[str]) -> list[str]:
    """Consult the row's labelled decisions for `check` once `act` differs from the v1 contract (`problems` = `diff(v1, act)`).
    A decision whose `expected` values `act` ALL reproduces sanctions the difference: returns [] and records
    `act["sanctioned_by"] = finding`.  Otherwise the problems name both sides -- v1's and every decision's own differences -- and a
    decision missing any of DECISION_FIELDS (or with an empty `expected`) never sanctions anything (fail closed).  Rows without a
    decision for `check` get `problems` back unchanged."""
    decisions = row.decisions(check)
    if not problems or not decisions:
        return problems
    out = [f"v1: {p}" for p in problems]
    for d in decisions:
        label = f"decision {d.get('finding') or '?'} (since {d.get('since') or '?'})"
        missing = [k for k in DECISION_FIELDS if not d.get(k)]
        if missing:
            out.append(f"{label}: malformed in fixtures.toml (missing {missing}); a decision without them sanctions nothing")
            continue
        dp = diff(d["expected"], act)
        if not dp:
            act["sanctioned_by"] = d["finding"]
            return []
        out.extend(f"{label}: {p}" for p in dp)
    return out
