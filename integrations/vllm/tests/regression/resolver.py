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
ENGINE_ENV = "VERITY_REGRESSION_ENGINE"        # which builder recomputes `actual` (checks/manifest_digest.py ENGINES); default = the v1 baseline
BASELINE_ENGINE = "v1"


def engine() -> str:
    """`VERITY_REGRESSION_ENGINE` as set (default `v1`, the baseline the reference rows' Commits ran).  The value is validated by
    the check that uses it (`manifest_digest.engine()`); here it only decides whether a `pending_review` decision may be consulted."""
    return os.environ.get(ENGINE_ENV, BASELINE_ENGINE)


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
        """A directory holding the row's request Programs (`build_request*/instances.json.gz` beside the derive's `result.json`) -- the
        live row dir, or, when the Programs and the records live in different trees (the store's `programs` / `records` artifacts,
        the pod's staged programs dir: `instances.json.gz` + `manifest.json` only), a merged view assembled under the scratch dir: one
        symlink per frozen `build_request*/…` file of fixtures.toml, each resolved through `path()` (sha / size checked).  None when no
        Programs are reachable.  Both engines read `result.json` (attention impls observed) beside the instances, so a split layout
        must be re-joined here, from the frozen index, never by guessing."""
        root: Path | None = None
        for r in self._roots():
            if any(r.glob("build_request*/instances.json.gz")):
                root = r
                break
        if root is None:
            try:
                root = self._store_root("programs")
            except NotResolvable:
                root = None
            if root is None or not any(root.glob("build_request*/instances.json.gz")):
                return None
        if all((d / "result.json").is_file() for d in root.glob("build_request*") if (d / "instances.json.gz").is_file()):
            return root                                    # the live layout: everything the builders read is beside the instances
        return self._merged_programs_view()

    def _merged_programs_view(self) -> Path:
        """`<scratch>/programs/<rel>` -> the resolved reference file, for every frozen `build_request*/…` (and `build/rank*/build_request*/…`)
        entry of this row; rebuilt idempotently (stale links replaced)."""
        view = scratch(self) / "programs"
        view.mkdir(parents=True, exist_ok=True)
        n = 0
        for rel in sorted(self.files):
            if not (rel.startswith("build_request") or rel.startswith("build/")) or "/" not in rel:
                continue
            try:
                src = self.path(rel)
            except NotResolvable:
                continue                                   # a frozen file no replica holds here: the builder names it if it needs it
            dst = view / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            n += 1
        if not any(view.glob("build_request*/instances.json.gz")) and not any(view.glob("build/rank*/build_request*/instances.json.gz")):
            raise NotResolvable(f"{self.key}: merged Programs view has no instances.json.gz ({n} files linked under {view})")
        return view

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


SCRATCH_ENV = "VERITY_REGRESSION_SCRATCH"


def scratch(row: Row) -> Path:
    """`$VERITY_REGRESSION_SCRATCH/r19-regress/<row>` (default: the system temp dir): rebuilt manifests, their logs, the merged Programs view."""
    import tempfile
    p = Path(os.environ.get(SCRATCH_ENV) or tempfile.gettempdir()) / "r19-regress" / row.key
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def diff(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Every leaf of `expected` that `actual` lacks or disagrees on (keys `actual` has beyond `expected` are not a difference: the
    expected document is the contract, lifted from the record).  Lists are compared whole."""
    out: list[str] = []
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
#: `status` of a decision: `accepted` (the default when absent: Daniel confirmed it; sanctions under every engine) or
#: `pending_review` (labelled by a lane, not yet confirmed: sanctions ONLY a non-baseline actual -- the v2 engine or a candidate --
#: so the v1 baseline stays untouched; the v1 engine reproducing it would be a change to the reference, FAIL).
DECISION_STATUSES = ("accepted", "pending_review")


def decision_status(d: dict[str, Any]) -> str:
    return str(d.get("status") or "accepted")


def is_baseline(act: dict[str, Any]) -> bool:
    """True when `act` was recomputed by the v1 baseline (engine `v1`, not a candidate's document)."""
    return engine() == BASELINE_ENGINE and act.get("source") != "candidate"


def sanction(row: Row, check: str, act: dict[str, Any], problems: list[str]) -> list[str]:
    """Consult the row's labelled decisions for `check` once `act` differs from the v1 contract (`problems` = `diff(v1, act)`).
    A decision whose `expected` values `act` ALL reproduces sanctions the difference: returns [] and records
    `act["sanctioned_by"] = finding` (+ `act["decision_status"]`).  Otherwise the problems name both sides -- v1's and every
    decision's own differences -- and a decision missing any of DECISION_FIELDS (or with an empty `expected`), or with an unknown
    `status`, never sanctions anything (fail closed).  A `pending_review` decision is not consulted for a baseline actual
    (`is_baseline`): the v1 engine must still reproduce v1.  Rows without a decision for `check` get `problems` back unchanged."""
    decisions = row.decisions(check)
    if not problems or not decisions:
        return problems
    out = [f"v1: {p}" for p in problems]
    baseline = is_baseline(act)
    for d in decisions:
        status = decision_status(d)
        label = f"decision {d.get('finding') or '?'} (since {d.get('since') or '?'}, {status})"
        missing = [k for k in DECISION_FIELDS if not d.get(k)]
        if missing:
            out.append(f"{label}: malformed in fixtures.toml (missing {missing}); a decision without them sanctions nothing")
            continue
        if status not in DECISION_STATUSES:
            out.append(f"{label}: unknown status {status!r} (one of {DECISION_STATUSES}); sanctions nothing")
            continue
        if status == "pending_review" and baseline:
            out.append(f"{label}: not consulted -- the {BASELINE_ENGINE} engine must reproduce the v1 reference; a pending decision sanctions v2 / candidate results only")
            continue
        dp = diff(d["expected"], act)
        if not dp:
            act["sanctioned_by"] = d["finding"]
            act["decision_status"] = status
            return []
        out.extend(f"{label}: {p}" for p in dp)
    return out


def matched(act: dict[str, Any], problems: list[str]) -> str:
    """Which side a passing check reproduced: `v1` or `decision <finding> (<status>)`; `neither` when problems remain."""
    if problems:
        return "neither"
    if act.get("sanctioned_by"):
        return f"decision {act['sanctioned_by']} ({act.get('decision_status') or 'accepted'})"
    return "v1"
