"""Where a reference record's bytes are: `artifact://r19/<row>/<rel>` -> a local file, sha256-checked against `files.toml`.

`files.toml` (this directory; a registered fixture, `fixtures/artifacts.json`, that `research data fetch-fixtures` fetches) lists every
reference file this harness may read: `[rows.<row>.files."<rel>"]` with `sha256`, `bytes`, `role` (record | program | manifest | ...).
`fixtures.toml` (committed) holds, per row, the places the bytes are known to live (`local.<machine>` = a row dir on that pod,
`artifacts.records` / `artifacts.programs` = `art:` ids in the data store once put, `oracle` = the vault extraction dir), with the
labelled decisions and retirements.  Nothing large is in Git: the bytes come from one of these, in this order:

    1. $VERITY_REGRESSION_ROWS_ROOT/<row>/<rel>          a directory of row dirs (tp2: /workspace/regress/rows -> symlinks to the live
                                                          row dirs under /workspace/cp/sweep/...; vault: /vault/tmp/oracle)
    2. [rows.<row>.local.*]/<rel>                         a recorded row-dir path that exists on THIS machine (tp2's live dirs)
    3. store_io.fetch([rows.<row>.artifacts.<role>])/<rel> the data API (records tree / programs tree), when those ids are set
    3b. store_io.fetch([artifacts.<name>])/<row>/<rel>     the top-level shared trees, laid out `<row_id>/<rel>` (e.g. `commit_logs`:
                                                          the `commit.log` the per-row records trees omitted, put from the oracle)
    4. nothing -> the check is SKIPPED by name ("input <rel> not resolvable here"); never a silent pass

A resolved file whose recorded sha256 is known and whose size is <= VERIFY_LIMIT is hashed and must match: a reference that differs
from the frozen index is `InputMismatch`, a failure, not a skip -- the oracle changed under the test.  Files above the limit (the
multi-GB fold records, 100+ MB manifests) are checked by size only unless VERITY_REGRESSION_VERIFY_ALL=1.

`Row` is the handle the checks receive: `row.path(rel)`, `row.json(rel)`, `row.workload()` (the in-repo `workloads/<row>.json`),
`row.programs_root()` (a dir holding `build_request*/instances.json.gz`, or None), `row.candidate(rel)` (the v2 output under
$VERITY_REGRESSION_CANDIDATE/<row>/<rel>, or None: self-check mode), `row.decisions(check)` (the labelled decisions of
`[[rows.<row>.decisions]]` naming that check; `sanction()` consults them when a result differs from v1).

$VERITY_REGRESSION_CANDIDATE may also name Attempt ids (`r2026…`, comma-separated): the v2 candidate is then what those Attempts
published -- `row.candidate_attempts()` are their manifests (matched to the row by `params.row`), `row.candidate_root()` the union
of their `vllm-build/v1` / `vllm-match-record/v1` / `vllm-verdict/v1` / `execution-evidence/v1` trees laid out like a row dir.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from verity_vllm.config import workload_file

HERE = Path(__file__).resolve().parent
VLLM_ROOT = HERE.parents[1]  # integrations/vllm
FIXTURES = HERE / "fixtures.toml"
FILES = HERE / "files.toml"
EXPECTED_DIR = HERE / "expected"
VERIFY_LIMIT = 64 * 1024 * 1024

ROWS_ROOT_ENV = "VERITY_REGRESSION_ROWS_ROOT"
CANDIDATE_ENV = "VERITY_REGRESSION_CANDIDATE"             # a directory of row dirs, or attempt ids (`r2026…`, comma-separated)
CANDIDATE_CACHE_ENV = "VERITY_REGRESSION_CANDIDATE_CACHE"  # where the attempts' output trees are unioned (default: the temp dir)
ENGINE = "v2"                                  # the builder that recomputes `actual`: Q_module_body_v1 via verity_vllm.query (recorded in every result)
ORACLE_ENV = "VERITY_REGRESSION_ORACLE"        # `expected` (default: expected/<row>.json, re-baselined on v2) | `frozen` (lift from the R2 frozen v1 records)
ORACLES = ("expected", "frozen")
RECORD_ACTUAL_ENV = "VERITY_REGRESSION_RECORD_ACTUAL"   # a directory: test_reproduces writes every `actual` to <dir>/<row>/<check>.json (rebaseline.py input)
REBUILD_ENV = "VERITY_REGRESSION_REBUILD"      # "1": run the checks that recompute from a row's Programs (`REBUILD = True`: a manifest rebuild, step_segmentation's decode; CPU minutes to half an hour a row)
STORE_REQUIRED_ENV = "VERITY_STORE_REQUIRED"   # "1" (a recorded check, the post-train job): an input that does not resolve fails, never skips
REGISTRY = HERE.parents[3] / "fixtures" / "artifacts.json"   # the fixtures too large for git (`research data fetch-fixtures`)


def store_required() -> bool:
    return os.environ.get(STORE_REQUIRED_ENV) == "1"


def rebuilds() -> bool:
    return os.environ.get(REBUILD_ENV) == "1"


def registered(p: Path) -> str | None:
    """The `art:` id `fixtures/artifacts.json` registers for `p` (a fixture git does not track), else None."""
    try:
        reg = json.loads(REGISTRY.read_text())
        rel = str(p.resolve().relative_to(REGISTRY.parents[1]))
    except (OSError, ValueError):
        return None
    return next((art for art, path in reg.items() if path == rel), None)


def oracle() -> str:
    """Which contract `expected` is: `expected` (default) = expected/<row>.json, the harness's frozen contract (re-baselined on v2's
    outputs after the flip; the labelled decisions of fixtures.toml are the recorded reasons it differs from the R2 records);
    `frozen` = lift from the R2 frozen v1 records themselves (`check.expected(row)`) -- the historical comparison, where the same
    decisions sanction the differences."""
    o = os.environ.get(ORACLE_ENV, ORACLES[0])
    if o not in ORACLES:
        raise ValueError(f"{ORACLE_ENV}={o!r}: one of {ORACLES}")
    return o


def record_result(row_key: str, check: str, exp: Any, act: Any, problems: list[str]) -> Path | None:
    """When `$VERITY_REGRESSION_RECORD_ACTUAL` names a directory, write the check's result document to `<dir>/<row>/<check>.json`:
    `{expected, actual, problems, matched, engine, oracle}` (canonical JSON) -- the input `rebaseline.py` merges into expected/.
    Never raises (a recording failure is not a test failure)."""
    d = os.environ.get(RECORD_ACTUAL_ENV)
    if not d:
        return None
    try:
        p = Path(d) / row_key / f"{check}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = {"schema": "verity-vllm/regression-result/v1", "row": row_key, "check": check, "expected": exp, "actual": act,
               "problems": list(problems), "matched": matched(act, problems), "engine": ENGINE, "oracle": oracle()}
        p.write_text(canonical(doc) + "\n")
        return p
    except Exception:
        return None


class InputMismatch(AssertionError):
    """A resolved reference file does not have the sha256 / size `files.toml` froze for it."""


class NotResolvable(LookupError):
    """No replica of the file is reachable from this machine (the check skips by name)."""


def load_fixtures() -> dict[str, Any]:
    with open(FIXTURES, "rb") as f:
        return tomllib.load(f)


def load_files() -> dict[str, dict[str, Any]]:
    """`[rows.<row>.files]` of files.toml, by row key; NotResolvable until `research data fetch-fixtures` has fetched it."""
    if not FILES.is_file():
        raise NotResolvable(f"{FILES.name}, the frozen sha256 / bytes of every reference file, is not here: it is the registered fixture "
                            f"{registered(FILES)}; `research data fetch-fixtures` fetches it (suites.py does before the suite runs)")
    with open(FILES, "rb") as f:
        return {k: v.get("files") or {} for k, v in (tomllib.load(f).get("rows") or {}).items()}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class Row:
    def __init__(self, key: str, spec: dict[str, Any], *, fixtures: dict[str, Any], indexed: bool = False):
        self.key = key
        self.spec = spec
        self.number = int(spec["row"])
        self.cls = spec.get("class")
        self._files: dict[str, dict[str, Any]] | None = None if indexed else (spec.get("files") or {})
        self._fixtures = fixtures
        self._fetched: dict[str, Path] = {}
        #: `<rel>` -> "<finding> (<status>)" for every input this row resolved by a DECISION's pin instead of the frozen v1 pin
        self.pinned_by: dict[str, str] = {}

    @property
    def files(self) -> dict[str, dict[str, Any]]:
        """The row's frozen reference files (`<rel>` -> sha256 / bytes / role): files.toml's for an indexed row, read on first use."""
        if self._files is None:
            self._files = load_files().get(self.key) or {}
        return self._files

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
        return self._fetch(art, role)

    def _fetch(self, art: str, label: str) -> Path:
        # the store's trees/ is a cache its eviction may clear under a running suite: a tree that went is fetched again
        if art not in self._fetched or not self._fetched[art].is_dir():
            from . import store_io
            try:
                self._fetched[art] = store_io.fetch(art)
            except Exception as e:  # the store not being reachable is a skip, not an error
                raise NotResolvable(f"{self.key}: store fetch of {label} {art} failed: {e}") from e
        return self._fetched[art]

    def _shared_roots(self) -> list[Path]:
        """`<fetched shared tree>/<row_id>` for every top-level `[artifacts]` entry (step 3b); trees the store cannot give are skipped."""
        out: list[Path] = []
        for name, art in sorted((self._fixtures.get("artifacts") or {}).items()):
            if not art:
                continue
            try:
                root = self._fetch(art, name) / self.key
            except NotResolvable:
                continue
            if root.is_dir():
                out.append(root)
        return out

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
        for root in self._shared_roots():
            if (root / rel).is_file():
                p = root / rel
                return self._checked(rel, p) if verify else p
        raise NotResolvable(f"{self.key}: {rel} not resolvable here (roots tried: {[str(r) for r in self._roots()]}; artifacts: {self.spec.get('artifacts')}; "
                            f"shared: {sorted((self._fixtures.get('artifacts') or {}).keys())})")

    def _checked(self, rel: str, p: Path) -> Path:
        rec = self.files.get(rel)
        if not rec:
            return p                                       # not frozen in the index: used as found (the check records that it did)
        problem = self._pin_problem(rel, p, rec)
        if problem is None:
            self.pinned_by.pop(rel, None)
            return p
        # the frozen v1 pin failed: a labelled decision may own a sanctioned pin for this input (`pins."<rel>" = { bytes, sha256 }`:
        # the manifest a decision's promotion produces, read downstream as an INPUT by coverage / replay_partition) -- the decision,
        # not the check, owns it
        tried: list[str] = []
        for d in self.spec.get("decisions") or []:
            pin = ((d.get("pins") or {}).get(rel)) if isinstance(d, dict) else None
            if not pin or not d.get("finding"):
                continue
            status = decision_status(d)
            label = f"decision {d['finding']} ({status})"
            if status not in DECISION_STATUSES:
                tried.append(f"{label}: unknown status")
                continue
            dp = self._pin_problem(rel, p, pin)
            if dp is None:
                self.pinned_by[rel] = f"{d['finding']} ({status})"
                return p
            tried.append(f"{label}: {dp}")
        raise InputMismatch(f"{self.key}/{rel}: v1 pin: {problem}" + ("".join(f"; {t}" for t in tried)))

    @staticmethod
    def _pin_problem(rel: str, p: Path, pin: dict[str, Any]) -> str | None:
        """None when `p` has the `bytes` / `sha256` of `pin` (sha only under VERIFY_LIMIT or VERIFY_ALL), else what differs."""
        size = p.stat().st_size
        if pin.get("bytes") is not None and int(pin["bytes"]) != size:
            return f"{size} bytes on disk, pinned {pin['bytes']}"
        if pin.get("sha256") and (size <= VERIFY_LIMIT or os.environ.get("VERITY_REGRESSION_VERIFY_ALL") == "1"):
            got = sha256_file(p)
            if got != pin["sha256"]:
                return f"sha256 {got[:16]} on disk != pinned {str(pin['sha256'])[:16]}"
        return None

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
        """The declared workload manifest (`integrations/vllm/workloads/<row>.json`, fetched by its art id and checked against its
        digest, `config.workload_file`) -- the same file row_pod.sh's `$WL` was."""
        p = workload_file(VLLM_ROOT / "workloads" / f"{self.key}.json")
        if not p.is_file():
            raise NotResolvable(f"{self.key}: workload manifest {p} not in this tree")
        return p

    def programs_root(self) -> Path | None:
        """A directory holding the row's request Programs (`build_request*/instances.json.gz` beside the derive's `result.json`) -- the
        live row dir, or, when the Programs and the records live in different trees (the store's `programs` / `records` artifacts,
        the pod's staged programs dir: `instances.json.gz` + `manifest.json` only), a merged view assembled under the scratch dir: one
        hard link (a symlink across filesystems) per frozen `build_request*/…` file of files.toml, each resolved through `path()`
        (sha / size checked).  None when no
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
            try:
                os.link(src, dst)        # the store's trees/ is a cache eviction may clear under a rebuild that reads for an hour
            except OSError:
                dst.symlink_to(src)
            n += 1
        if not any(view.glob("build_request*/instances.json.gz")) and not any(view.glob("build/rank*/build_request*/instances.json.gz")):
            raise NotResolvable(f"{self.key}: merged Programs view has no instances.json.gz ({n} files linked under {view})")
        return view

    def candidate(self, rel: str) -> Path | None:
        """The v2 candidate's file for this row (`$VERITY_REGRESSION_CANDIDATE/<row>/<rel>`, or the same path inside the union of the
        candidate Attempts' output trees when the variable names attempt ids), None in self-check mode."""
        root = self.candidate_root()
        if root is None:
            return None
        p = root / rel
        return p if p.is_file() else None

    def candidate_root(self) -> Path | None:
        """`$VERITY_REGRESSION_CANDIDATE` is either a directory of row dirs (`<root>/<row>/`) or a comma/space-separated list of Attempt
        ids (`r2026…`): then this row's candidate dir is the union of its attempts' `vllm-build/v1` / `vllm-match-record/v1` /
        `vllm-verdict/v1` / `execution-evidence/v1` trees, fetched once per process under `$VERITY_REGRESSION_CANDIDATE_CACHE/<row>/`
        (default: the temp dir).  None when nothing names this row."""
        env = os.environ.get(CANDIDATE_ENV)
        if not env:
            return None
        ids = candidate_attempt_ids()
        if not ids:
            return (Path(env) / self.key) if (Path(env) / self.key).is_dir() else None
        mine = [d["id"] for d in self.candidate_attempts()]
        if not mine:
            return None
        cache = Path(os.environ.get(CANDIDATE_CACHE_ENV) or (Path(tempfile.gettempdir()) / "verity-regression-candidates"))
        dest = cache / self.key
        marker = dest / "ATTEMPTS.json"
        if marker.is_file():
            try:
                if sorted(json.loads(marker.read_text()).get("attempts", {})) == sorted(mine):
                    return dest
            except (OSError, ValueError):
                pass
            shutil.rmtree(dest, ignore_errors=True)
        from . import store_io
        try:
            return store_io.fetch_attempt(mine, dest)
        except Exception as e:  # the store not being reachable is a skip, not an error
            raise NotResolvable(f"{self.key}: candidate attempts {mine} not fetchable here: {e}") from e

    def candidate_attempts(self) -> list[dict[str, Any]]:
        """The candidate Attempt manifests naming this row (`params.row` == this row's key), in the order given."""
        return [d for d in candidate_attempt_docs() if (d.get("params") or {}).get("row") == self.key]

    # ---- expected -------------------------------------------------------------------------------------------------------------
    def expected_path(self) -> Path:
        return EXPECTED_DIR / f"{self.key}.json"

    def expected(self) -> dict[str, Any]:
        p = self.expected_path()
        if not p.is_file():
            return {}
        with open(p) as f:
            return json.load(f)

    def decisions(self, check: str) -> list[dict[str, Any]]:
        """The labelled decisions on this row for `check` (`[[rows.<row>.decisions]]` in fixtures.toml, design rule Draft 3 §7: a
        difference from the v1 reference is sanctioned only by a recorded decision naming the finding or the changed policy)."""
        return [d for d in (self.spec.get("decisions") or []) if isinstance(d, dict) and d.get("check") == check]

    def retired(self, check: str) -> dict[str, Any] | None:
        """`[rows.<row>.retired.<check>]` of fixtures.toml: the check cannot run on this row's record with the current code, and why
        (`reason`), since which change (`since`), and what would bring it back (`needs`).  The test skips it by that reason; nothing
        else lets a check that does not resolve skip under VERITY_STORE_REQUIRED=1."""
        r = (self.spec.get("retired") or {}).get(check)
        return r if isinstance(r, dict) else None


SCRATCH_ENV = "VERITY_REGRESSION_SCRATCH"


def scratch(row: Row) -> Path:
    """`$VERITY_REGRESSION_SCRATCH/r19-regress/<row>/<pid>` (default: the system temp dir): rebuilt manifests, their logs, the merged
    Programs view.  One per process: xdist workers rebuilding one row (#57's manifest and its refusal) never share a file."""
    import tempfile
    p = Path(os.environ.get(SCRATCH_ENV) or tempfile.gettempdir()) / "r19-regress" / row.key / str(os.getpid())
    p.mkdir(parents=True, exist_ok=True)
    return p

_ATTEMPT_DOCS: dict[str, dict[str, Any]] = {}


def candidate_attempt_ids() -> list[str]:
    """The attempt ids `$VERITY_REGRESSION_CANDIDATE` names (comma / whitespace separated); [] when it is a directory or unset."""
    env = os.environ.get(CANDIDATE_ENV) or ""
    toks = [t for t in re.split(r"[,\s]+", env) if t]
    ids = [t for t in toks if re.fullmatch(r"r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}", t)]
    return ids if ids and len(ids) == len(toks) else []


def candidate_attempt_docs() -> list[dict[str, Any]]:
    """The manifests of the candidate attempts, fetched once per process (`store_io.attempt`), plus the chain they cite: an attempt's
    `inputs` name artifacts (`build=art:…`, `match=art:…`) whose manifests carry the publishing attempt in `meta.run_id` -- naming the
    Commit attempt alone resolves the Match and Build attempts behind it.  An unreachable store is NotResolvable."""
    from . import store_io
    out: list[dict[str, Any]] = []
    queue = list(candidate_attempt_ids())
    seen: set[str] = set()
    while queue:
        rid = queue.pop(0)
        if rid in seen:
            continue
        seen.add(rid)
        if rid not in _ATTEMPT_DOCS:
            try:
                _ATTEMPT_DOCS[rid] = store_io.attempt(rid)
            except Exception as e:
                raise NotResolvable(f"candidate attempt {rid} not readable here: {e}") from e
        doc = _ATTEMPT_DOCS[rid]
        out.append(doc)
        for art in (doc.get("inputs") or {}).values():
            if isinstance(art, str) and art.startswith("art:"):
                try:
                    up = (store_io.show(art).get("meta") or {}).get("run_id")
                except Exception:  # noqa: BLE001 -- a cited artifact nobody can read is a broken chain: attempt_provenance names it
                    continue
                if isinstance(up, str) and re.fullmatch(r"r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}", up) and up not in seen:
                    queue.append(up)
    return out


def rows(selection: str | None = None) -> list[Row]:
    """The reference rows of fixtures.toml, by row number; `selection` = 'green' | 'by-name' | None (all)."""
    fx = load_fixtures()
    out = [Row(k, v, fixtures=fx, indexed=True) for k, v in (fx.get("rows") or {}).items()]
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
#: what `[rows.<row>.retired.<check>]` must state (`Row.retired`)
RETIREMENT_FIELDS = ("reason", "since", "needs")
#: `status` of a decision: `accepted` (the default when absent: confirmed) or `pending_review` (labelled, not yet confirmed); both
#: sanction.
DECISION_STATUSES = ("accepted", "pending_review")


def decision_status(d: dict[str, Any]) -> str:
    return str(d.get("status") or "accepted")


def sanction(row: Row, check: str, act: dict[str, Any], problems: list[str]) -> list[str]:
    """Consult the row's labelled decisions for `check` once `act` differs from the v1 contract (`problems` = `diff(v1, act)`).
    A decision whose `expected` values `act` ALL reproduces sanctions the difference: returns [] and records
    `act["sanctioned_by"] = finding` (+ `act["decision_status"]`).  Otherwise the problems name both sides -- v1's and every
    decision's own differences -- and a decision missing any of DECISION_FIELDS (or with an empty `expected`), or with an unknown
    `status`, never sanctions anything (fail closed).  Rows without a decision for `check` get `problems` back unchanged."""
    decisions = row.decisions(check)
    if not problems or not decisions:
        return problems
    out = [f"v1: {p}" for p in problems]
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
        dp = diff(d["expected"], act)
        if not dp:
            act["sanctioned_by"] = d["finding"]
            act["decision_status"] = status
            return []
        out.extend(f"{label}: {p}" for p in dp)
    return out


def matched(act: dict[str, Any], problems: list[str]) -> str:
    """Which side a passing check reproduced: the oracle (`expected` = expected/<row>.json, `frozen` = the R2 v1 records) or
    `decision <finding> (<status>)`; `neither` when problems remain."""
    if problems:
        return "neither"
    if act.get("sanctioned_by"):
        return f"decision {act['sanctioned_by']} ({act.get('decision_status') or 'accepted'})"
    return oracle()
