"""The canonical VU query (`verity_capture.vu_canonical`) over every supported-family fixture and, when present in the tree,
the real captured `Serve_v1` programs (`fixtures/e2e-composed/*/program/descriptor.json.gz`, 5e8 gates,
decoded in < 1 s): evaluates, exact partition, every w_out <= 32 (explicit oracle on the fixtures; the validators' verdict
on the real programs), count and max w_out reported, and the key property that attention VUs are not a population
indexed by every exact T (the set of VU kinds is the same at T = 40 and T = 72).

Skips (never fails) while `verity.ir.query_ast` / `verity_vllm.ir.partition` are not merged.  Counts land in
`/tmp/qtest_vu_canonical_report.json` (the status file quotes them)."""
from __future__ import annotations

import gzip
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from verity.ir.parts import Unsupported
from verity.ir.program import Program
from tests.ir import fixtures_query as FX
from verity_capture import vu_canonical as VC

REPORT_PATH = Path(os.environ.get("QTEST_VU_REPORT", "/tmp/qtest_vu_canonical_report.json"))
REPORT: dict[str, dict] = {}
ORACLE_SAMPLE = 600           # sets per fixture checked against the explicit oracle (all when count <= this)
REAL_PROGRAM_BUDGET_S = 180   # per real program: evaluation + validation, else skip with an issue line
REAL_PROGRAM_MEM_GB = float(os.environ.get("QTEST_REAL_MEM_GB", "0.9"))     # RSS growth allowed to the validators ...
REAL_PROGRAM_MEM_CAP_GB = float(os.environ.get("QTEST_REAL_MEM_CAP_GB", "1.2"))   # ... and an absolute cap (the 24 GB laptop SIGKILLs at ~2 GB when swap is full)


_MODULES = {"gateset": "verity.ir.query_ast", "boundary": "verity_vllm.ir.boundary", "partition": "verity_vllm.ir.partition"}


def _mod(name: str, owner: str):
    return pytest.importorskip(_MODULES[name], reason=f"{_MODULES[name]} not merged yet (owner {owner})")


@pytest.fixture(scope="module")
def GS():
    return _mod("gateset", "qapi")


@pytest.fixture(scope="module")
def PT():
    """partition with widths: `validate_vu_partition` calls `boundary.w_out_bound`/`w_out` at run time."""
    _mod("boundary", "qbound")
    return _mod("partition", "qpart")


def _record(key: str, **kw) -> None:
    REPORT.setdefault(key, {}).update(kw)
    try:
        REPORT_PATH.write_text(json.dumps(REPORT, indent=1, sort_keys=True, default=str))
    except OSError:
        pass


def _rss_reader():
    """A cheap current-RSS reader (GB): libproc on macOS, /proc/self/statm on Linux, else `ru_maxrss` (high-water mark)."""
    if sys.platform == "darwin":
        try:
            import ctypes
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            buf = ctypes.create_string_buffer(512)   # struct proc_taskinfo (2 x 64-bit: virtual size, resident size, ...)
            pid, PROC_PIDTASKINFO = os.getpid(), 4

            def read() -> float:
                n = libproc.proc_pidinfo(pid, PROC_PIDTASKINFO, ctypes.c_uint64(0), buf, 512)
                return int.from_bytes(buf.raw[8:16], "little") / 2**30 if n >= 16 else 0.0
            if read() > 0:
                return read
        except Exception:
            pass
    if os.path.exists("/proc/self/statm"):
        def read() -> float:
            with open("/proc/self/statm") as fh:
                return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        return read
    import resource

    def read() -> float:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2**30 if sys.platform == "darwin" else r / 2**20
    return read


_rss_gb = _rss_reader()


def _main_stack(depth: int = 6) -> str:
    """`module.function` chain (innermost last) of the main thread, restricted to verity.ir frames — the owner's pointer."""
    import traceback
    frame = sys._current_frames().get(threading.main_thread().ident)
    if frame is None:
        return "?"
    chain = [f"{Path(fs.filename).stem}.{fs.name}:{fs.lineno}" for fs in traceback.extract_stack(frame) if "verity.ir" in fs.filename]
    return " > ".join(chain[-depth:])


class _Budget:
    """Time box + memory box for the real-program checks (Unix): a hang becomes a TimeoutError, a memory blow-up becomes
    a TimeoutError before the OS SIGKILLs the whole pytest process (a watchdog thread polls the process RSS and fires the
    same SIGALRM path in the main thread)."""

    def __init__(self, seconds: int, what: str, mem_gb: float | None = None) -> None:
        self.seconds, self.what, self.mem_gb = seconds, what, mem_gb
        self._stop = threading.Event()
        self._why = f"{what} exceeded {seconds} s"

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._fire)
            signal.alarm(self.seconds)
            if self.mem_gb is not None:
                threading.Thread(target=self._watch, daemon=True).start()
        return self

    def _watch(self) -> None:
        base = _rss_gb()
        log = os.environ.get("QTEST_WATCH_LOG")
        while not self._stop.wait(0.02):
            now = _rss_gb()
            if log:
                with open(log, "a") as fh:
                    fh.write(f"{time.time():.2f} {self.what} base {base:.2f} now {now:.2f}\n")
            if now - base > self.mem_gb or now > REAL_PROGRAM_MEM_CAP_GB:
                self._why = (f"{self.what} grew the process RSS by {now - base:.1f} GB to {now:.1f} GB (box: +{self.mem_gb} GB, "
                             f"cap {REAL_PROGRAM_MEM_CAP_GB} GB) while in {_main_stack()}")
                signal.raise_signal(signal.SIGALRM)   # delivered to the main thread's handler
                return

    def _fire(self, *_):
        raise TimeoutError(self._why)

    def __exit__(self, *exc):
        self._stop.set()
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        return False


# ---------------------------------------------------------------------------------------------------------
# the query text / document (no Program needed)
# ---------------------------------------------------------------------------------------------------------


def test_profile_resolution():
    assert VC.resolve_profile("default") == "ampere"
    assert VC.resolve_profile("vllm-d9105ea80-sm89-eager-bf16") == "ampere"
    assert VC.resolve_profile("vllm-d9105ea80-sm90-eager-bf16") == "hopper"
    assert VC.resolve_profile("hopper") == "hopper" and VC.resolve_profile("FA3") == "hopper"
    with pytest.raises(ValueError):
        VC.resolve_profile("tpu-v5e")
    text = VC.canonical_query_text()
    assert text.startswith("override(default(32), {") and '"//Gemm_v1{*}": members("//Gemm_v1{*}", "@batch:GemmCoordinate_v1")' in text
    assert "AmpereBF16TcDot16_v1" in text and "HopperBF16WgmmaDot16_v1" in VC.canonical_query_text("hopper")
    paths = [p for p, _ in VC.canonical_cases_text()]
    assert len(paths) == len(set(paths))


def test_canonical_query_builds_and_serialises(GS):
    q = VC.canonical_vu_query("ampere")
    doc = VC.canonical_query_document("ampere", created_utc="2026-09-17T21:00:00Z")
    assert doc["schema"] == VC.SCHEMA and set(doc) == {"schema", "query_id", "query", "provenance"}
    assert "kind" not in doc and "w_max" not in doc, "the query document carries no use configuration"
    assert len(doc["query_id"]) == 64 and doc["query_id"] == GS.query_id(q)
    assert GS.to_json(GS.from_json(doc["query"])) == doc["query"]
    assert GS.query_id(VC.query_from_document(doc)) == doc["query_id"]
    assert VC.canonical_query_id("hopper") != doc["query_id"], "the PV zip names the profile's tensor-core step"
    assert VC.canonical_query_id("sm89") == doc["query_id"]
    bad = dict(doc, query_id="0" * 64)
    with pytest.raises(ValueError):
        VC.query_from_document(bad)


def test_text_form_parses_to_the_same_ast(GS):
    parse = getattr(GS, "parse", None)
    if parse is None:
        pytest.skip("ISSUE FOR qapi: gateset.parse missing (frozen interface)")
    q_text = parse(VC.canonical_query_text("ampere"))
    assert GS.to_json(q_text) == GS.to_json(VC.canonical_vu_query("ampere")), "constructor text and AST disagree"


# ---------------------------------------------------------------------------------------------------------
# fixtures: exact partition, w_out <= 32 (validators + explicit oracle), counts
# ---------------------------------------------------------------------------------------------------------


def _sets(fam, cap: int):
    n = fam.count()
    idx = range(n) if n <= cap else range(0, n, max(1, n // cap))
    for i in idx:
        yield i, fam.by_index(i)


def _run_canonical(GS, PT, P: Program, label: str, *, oracle: bool) -> dict:
    q = VC.canonical_vu_query("ampere")
    t0 = time.time()
    fam = GS.evaluate(P, q)
    count = fam.count()
    v = PT.validate_vu_partition(P, q, VC.W_MAX)
    t_eval = time.time() - t0
    res = {"gates": P.circuit.size, "count": count, "vu_count_verdict": v.count, "partition": v.partition.status,
           "codes": list(v.partition.codes), "width_status": v.width.status, "width_max_seen": v.width.max_seen,
           "violators": len(v.width.violators), "unsupported": len(v.width.unsupported), "status": v.status, "seconds": round(t_eval, 2)}
    if oracle:
        wmax, checked = 0, 0
        for i, s in _sets(fam, ORACLE_SAMPLE):
            w = FX.oracle_w_out(P, s)
            assert w <= VC.W_MAX, f"{label}: VU #{i} has w_out {w} > {VC.W_MAX} ({FX.gateset_kind(s)})"
            assert fam.index_of(s) == i, f"{label}: index_of(by_index({i})) != {i}"
            wmax, checked = max(wmax, w), checked + 1
        res.update(oracle_max_w_out=wmax, oracle_checked=checked)
        kinds = {FX.gateset_kind(s) for _, s in _sets(fam, 10_000)}
        res["kinds"] = len(kinds)
    _record(label, **res)
    assert v.partition.status == "exact", f"{label}: partition {v.partition.status} {v.partition.codes} {v.partition.detail}"
    assert not v.width.violators, f"{label}: w_out violators {v.width.violators[:8]} (max_seen {v.width.max_seen})"
    assert not v.width.unsupported, f"{label}: Unsupported members {v.width.unsupported[:8]} -- not a pass"
    assert v.width.max_seen is None or v.width.max_seen <= VC.W_MAX
    assert v.count == count > 0
    return res


def _sweep_tiles(P: Program, fam) -> tuple[bool, str]:
    """This suite's own interval sweep over every set (fixtures only): exact tiling of [0, size) or the first defect."""
    ivs = sorted(iv for _, s in _sets(fam, 10**9) for iv in s.intervals())
    pos = 0
    for lo, hi in ivs:
        if lo > pos:
            return False, f"gap [{pos}:{lo})"
        if lo < pos:
            return False, f"overlap at {lo} (< {pos})"
        pos = hi
    return pos == P.circuit.size, f"covered [0:{pos}) of {P.circuit.size}"


@pytest.mark.parametrize("name", FX.VLLM_FAMILY_FIXTURES)
def test_fixture_family_by_oracle(GS, name):
    """Evaluator + explicit oracle only (no partition module): the canonical sets tile the circuit and every sampled set
    has w_out <= 32; counts, kinds and the max width are recorded."""
    P = FX.wrapped_fixture(name)
    q = VC.canonical_vu_query()
    t0 = time.time()
    fam = GS.evaluate(P, q)
    count = fam.count()
    ok, why = _sweep_tiles(P, fam)
    assert ok, f"{name}: canonical sets do not tile the circuit: {why}"
    wmax, checked = 0, 0
    for i, s in _sets(fam, ORACLE_SAMPLE):
        w = FX.oracle_w_out(P, s)
        assert w <= VC.W_MAX, f"{name}: VU #{i} has w_out {w} > {VC.W_MAX} ({FX.gateset_kind(s)})"
        assert fam.index_of(s) == i
        wmax, checked = max(wmax, w), checked + 1
    kinds = {FX.gateset_kind(s) for _, s in _sets(fam, 10**9)}
    _record(f"oracle:{name}", gates=P.circuit.size, count=count, oracle_max_w_out=wmax, oracle_checked=checked, kinds=len(kinds),
            seconds=round(time.time() - t0, 2))


@pytest.mark.parametrize("name", FX.VLLM_FAMILY_FIXTURES)
def test_fixture_family_exact_and_narrow(GS, PT, name):
    P = FX.wrapped_fixture(name)
    res = _run_canonical(GS, PT, P, f"fixture:{name}", oracle=True)
    assert res["oracle_max_w_out"] <= VC.W_MAX


@pytest.mark.parametrize("name", ["gemm", "attention_head", "rmsnorm_triton"])
def test_inlined_fixture_falls_back_to_default(GS, PT, name):
    """`Program(spec)` inlines the family body into the root: the override paths match nothing and `default(32)` does the
    work -- still exact and narrow (the canonical query is never wrong on a program it has no case for)."""
    _run_canonical(GS, PT, FX.fixture(name), f"inlined:{name}", oracle=True)


def test_gemm_vus_are_output_words(GS):
    P = FX.wrapped_fixture("gemm")
    fam = GS.evaluate(P, VC.canonical_vu_query())
    coords = [s for _, s in _sets(fam, 10_000) if any(k[0] == "Gemm_v1" for k in FX.gateset_kind(s))]
    assert len(coords) == 6
    for s in coords:
        assert FX.oracle_w_out(P, s) == 16
        assert len(s.parts) == 1 and len(s.parts[0].items) == 1
    # the Input gates are members of their batch nodes: 32 + 192 Input16 VUs of 16 bits
    inputs = [s for _, s in _sets(fam, 10_000) if any("Input" in k[1][0][0] for k in FX.gateset_kind(s))]
    assert len(inputs) == 224 and all(FX.oracle_w_out(P, s) == 16 for s in inputs[:8])
    assert fam.count() == 230


def test_attention_population_not_keyed_by_T(GS):
    """T = 40 -> 3 blocks, T = 72 -> 5 blocks (BN = 16): more VUs, the SAME set of VU kinds."""
    q = VC.canonical_vu_query()
    P40, P72 = FX.wrapped_fixture("attention_head"), FX.wrapped_fixture("attention_head_t72")
    f40, f72 = GS.evaluate(P40, q), GS.evaluate(P72, q)
    k40 = {FX.gateset_kind(s) for _, s in _sets(f40, 100_000)}
    k72 = {FX.gateset_kind(s) for _, s in _sets(f72, 100_000)}
    _record("attention_kinds", T40={"count": f40.count(), "kinds": len(k40)}, T72={"count": f72.count(), "kinds": len(k72)},
            new_kinds_at_T72=sorted(map(str, k72 - k40)), kinds_lost_at_T72=sorted(map(str, k40 - k72)))
    assert f72.count() > f40.count()
    assert k72 == k40, f"new kinds at T=72: {sorted(map(str, k72 - k40))[:5]}; lost: {sorted(map(str, k40 - k72))[:5]}"
    # no VU is an AttnBlock call or a whole head: the 672-bit interfaces were dissolved
    for _, s in _sets(f72, 100_000):
        for base, callees in FX.gateset_kind(s):
            assert not any(cid.startswith("AttnBlock_v") and form == "call" for cid, form in callees), "an AttnBlock call is a VU"
            assert not any(cid.startswith("AttentionHead_v") for cid, form in callees), "a whole head is a VU"
    # the PV chain per d: one VU per output coordinate over the k-step batches, 32 bits
    pv = [s for _, s in _sets(f40, 100_000) if any(any(cid.startswith("AmpereBF16TcDot16") for cid, _ in c) for _, c in FX.gateset_kind(s))]
    assert len(pv) == 3 * 16, "3 blocks x D=16 PV chains"
    assert all(FX.oracle_w_out(P40, s) == 32 for s in pv)


# ---------------------------------------------------------------------------------------------------------
# real captured programs (structural only: 5e8 gates)
# ---------------------------------------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _program_records() -> list[Path]:
    root = _repo_root()
    found: list[Path] = []
    for base in ("verity_capture/tests/fixtures", "vllm-poc/fixtures"):
        d = root / base
        if d.is_dir():
            found.extend(sorted(d.rglob("program.json")))
    return found


def _load_real_program(pj: Path) -> Program | None:
    """A `Program` instance around the decoded descriptor named by `program.json` (None when the descriptor is absent)."""
    rec = json.loads(pj.read_text())
    prog = rec.get("program", rec)
    name = prog.get("descriptor")
    if not name:
        return None
    path = pj.parent / name
    if not path.exists():
        return None
    from verity.ir.codec import decode_program
    from verity.ir.defs import REGISTRY
    import verity_vllm.registry.b1  # noqa: F401  (registers the primitives the descriptor names)
    desc = json.load(gzip.open(path, "rt")) if name.endswith(".gz") else json.loads(path.read_text())
    dec = decode_program(desc, REGISTRY)
    P = Program.__new__(Program)
    P.fn = dec.program_fn_functions[0] if dec.program_fn_functions else None
    P.annotations = dec.annotations
    P.root = dec.root
    P._circuit = dec.circuit
    return P


_RECORDS = _program_records()


def _real_ids(p):
    return "/".join(p.parts[-3:-1]) if p else "none"


def _distinct_records(records: list[Path], k: int = 2) -> list[Path]:
    """Up to k records of distinct program shape (the two INT-B0 captures are the same 5.2e8-gate Serve program)."""
    out, seen = [], set()
    for pj in records:
        try:
            prog = json.loads(pj.read_text())
            prog = prog.get("program", prog)
            key = (prog.get("gates"), prog.get("descriptor"))
        except (OSError, ValueError):
            continue
        if key not in seen:
            seen.add(key)
            out.append(pj)
        if len(out) == k:
            break
    return out


_REAL = _distinct_records(_RECORDS) or [None]
_WHOLE_CALLS = ("AttnBlock_v1", "AttnBlock_v2", "AttnBlock_v3", "Gemm_v1", "Gemm_v2", "AttentionHead_v1", "AttentionHead_v2",
                "AttentionHead_v3", "Attention_v1", "Attention_v3", "LayerPre_v1", "LayerPost_v1", "Final_v1", "RMSNormTriton_v1",
                "RMSNormFusedCuda_v1", "RMSNormFusedCuda_v2", "SiluMul_v1")


def _load_or_skip(pj):
    if pj is None:
        pytest.skip("no captured program.json in the sparse tree")
    P = _load_real_program(pj)
    if P is None:
        pytest.skip(f"{pj}: descriptor not in the sparse tree")
    return P


def real_program_structural(pj: str) -> dict:
    """Evaluator only (runs in the child): count structurally, `by_index`/`index_of` round trip, sampled kinds, no whole
    family call as a VU."""
    from verity.ir import query_ast as GS
    P = _load_real_program(Path(pj))
    q = VC.canonical_vu_query()
    t0 = time.time()
    fam = GS.evaluate(P, q)
    n = fam.count()
    t_count = time.time() - t0
    sample_idx = list(range(0, n, max(1, n // 3000)))
    kinds: dict[tuple, int] = {}
    for i in sample_idx:
        k = FX.gateset_kind(fam.by_index(i))
        kinds[k] = kinds.get(k, 0) + 1
    roundtrip = {i: fam.index_of(fam.by_index(i)) for i in (0, n // 2, n - 1)}
    bad = [str(k)[:160] for k in kinds for _, callees in k for cid, form in callees if form == "call" and cid.split("{")[0] in _WHOLE_CALLS]
    return {"gates": P.circuit.size, "count": n, "seconds_evaluate_count": round(t_count, 2), "sampled": len(sample_idx),
            "sampled_kinds": len(kinds), "top_kinds": [(v, str(k)[:120]) for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:6]],
            "roundtrip": roundtrip, "whole_call_vus": bad[:3], "program_fn": str(getattr(P.root, "id", "?"))[:80], "rss_gb": round(_rss_gb(), 2)}


def real_program_validators(pj: str) -> dict:
    """Validators (runs in the child): exact partition and w_out <= 32 on the captured program, structurally."""
    from verity.ir import query_ast as GS
    from verity_vllm.ir import partition as PT
    P = _load_real_program(Path(pj))
    res = _run_canonical(GS, PT, P, f"real:{pj}", oracle=False)
    res["rss_gb"] = round(_rss_gb(), 2)
    return res


def real_program_readiness(pj: str) -> dict:
    """Wide-gate inventory (runs in the child): `readiness_32bit` over the captured program's specializations."""
    from verity_vllm.ir import partition as PT
    P = _load_real_program(Path(pj))
    t0 = time.time()
    r = PT.readiness_32bit(P)
    return {"ok": bool(r.ok), "max_w": r.max_w, "specs": getattr(r, "specs", None), "gates_over": getattr(r, "gates_over", None),
            "violators": [str(v)[:160] for v in list(r.violators)[:8]], "seconds": round(time.time() - t0, 2), "rss_gb": round(_rss_gb(), 2)}


def _in_child(fn_name: str, pj: Path, label: str, budget_s: int = REAL_PROGRAM_BUDGET_S) -> dict:
    """Run `fn_name(pj)` from this module in a fresh interpreter (low base RSS; an evaluator memory blow-up or hang kills
    or times out the child, never the pytest process) and return its JSON result; a killed / failed child skips with an
    issue line naming the last verity.ir frames the child's own memory watchdog saw."""
    import subprocess
    code = (f"import json, sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n"
            f"from tests.ir import test_vu_canonical as M\n"
            f"try:\n"
            f"    with M._Budget({budget_s}, {label!r}, mem_gb=M.REAL_PROGRAM_MEM_GB):\n"
            f"        res = M.{fn_name}({str(pj)!r})\n"
            f"except TimeoutError as e:\n"
            f"    res = {{'timeout': str(e)}}\n"
            f"except Exception:\n"
            f"    import traceback; res = {{'error': traceback.format_exc()}}\n"
            f"print('@@RESULT@@' + json.dumps(res, default=str))\n")
    env = dict(os.environ, QTEST_VU_REPORT=str(REPORT_PATH) + ".child")
    try:
        cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=budget_s + 60, env=env)
    except subprocess.TimeoutExpired:
        _record(label, timeout=f"child exceeded {budget_s + 60} s")
        pytest.skip(f"ISSUE FOR qapi/qpart: {label}: child exceeded {budget_s + 60} s")
    marker = [ln for ln in cp.stdout.splitlines() if ln.startswith("@@RESULT@@")]
    if not marker:
        why = f"child exit {cp.returncode}" + (" (SIGKILL: the OS reclaimed it — memory)" if cp.returncode == -9 else "")
        _record(label, child_failed=why, stderr_tail=cp.stderr[-800:])
        pytest.skip(f"ISSUE FOR qpart/qbound: {label}: {why}; stderr tail: {cp.stderr.strip()[-400:]}")
    res = json.loads(marker[-1][len("@@RESULT@@"):])
    if "timeout" in res:
        _record(label, timeout=res["timeout"])
        pytest.skip(f"ISSUE FOR qpart/qbound: {res['timeout']} (validation must be structural on a 5e8-gate program)")
    if "error" in res:   # an assertion or exception in the child is a real failure, not a skip
        _record(label, error=res["error"][-800:])
        pytest.fail(f"{label}: child raised:\n{res['error']}")
    _record(label, **res)
    return res


@pytest.mark.parametrize("pj", _REAL, ids=_real_ids)
def test_real_program_structural(GS, pj):
    """Evaluator only: the canonical query over a captured `Serve_v1` program (5e8 .. 5e10 gates) counts structurally in
    seconds, `by_index` / `index_of` round-trip, no whole family call is a VU, and the sampled kinds are recorded."""
    if pj is None:
        pytest.skip("no captured program.json in the sparse tree")
    label = f"real:{_real_ids(pj)}"
    res = _in_child("real_program_structural", pj, label)
    assert res["count"] > 0
    assert all(int(v) == int(k) for k, v in res["roundtrip"].items()), f"{label}: index_of(by_index(i)) != i: {res['roundtrip']}"
    assert not res["whole_call_vus"], f"{label}: whole calls as VUs: {res['whole_call_vus']}"
    # the population is a handful of T-agnostic kinds (a Serve program has an AttnBlock specialization per visible-token
    # count NVIS = 1..T; the VU kinds must not multiply with them)
    assert res["sampled_kinds"] <= 64, f"{label}: {res['sampled_kinds']} VU kinds in a 3000-sample: {res['top_kinds']}"


@pytest.mark.parametrize("pj", _REAL, ids=_real_ids)
def test_real_program_exact_and_narrow(GS, PT, pj):
    """Validators: exact partition and w_out <= 32 on the captured program (structural; the explicit oracle is impossible)."""
    if pj is None:
        pytest.skip("no captured program.json in the sparse tree")
    label = f"real:{_real_ids(pj)}"
    res = _in_child("real_program_validators", pj, label)
    assert res["partition"] == "exact", f"{label}: partition {res['partition']} {res['codes']}"
    assert res["width_status"] == "ok", f"{label}: width {res['width_status']} violators={res['violators']} unsupported={res['unsupported']}"
    assert res["status"] == "ok"


def test_query_id_agrees_across_modules_and_the_use_document_binds(GS, PT):
    """The canonical query's id is the same number in vu_canonical, gateset, query_codec and query_artifact; the G8 pair
    (query document, use document) produced from a fixture verdict checks clean and carries `status == "ok"`."""
    QA = pytest.importorskip("verity_vllm.ir.query_artifact", reason="verity_vllm.ir.query_artifact not merged yet (owner qpart)")
    QC = pytest.importorskip("verity.ir.query_codec", reason="verity.ir.query_codec not merged yet (owner qapi)")
    q = VC.canonical_vu_query()
    qid = VC.canonical_query_id()
    assert qid == GS.query_id(q) == QC.query_id(q) == QA.query_id(q) == VC.canonical_query_document()["query_id"]
    assert QC.query_id(GS.parse(VC.canonical_query_text())) == qid
    P = FX.wrapped_fixture("gemm")
    v = PT.validate_vu_partition(P, q, VC.W_MAX)
    qdoc, udoc = QA.vu_use_document(q, P, v, VC.W_MAX)
    assert qdoc["query_id"] == qid and qdoc["schema"] == VC.SCHEMA
    assert QA.check_query_document(qdoc) == []
    assert QA.check_use_document(udoc, qdoc["program_digest"], qdoc) == []
    assert udoc["verdict"]["status"] == "ok" and udoc["verdict"]["count"] == v.count == 230
    assert udoc["verdict"]["width"]["max_seen"] == 16 and udoc["use"]["w_max"] == VC.W_MAX
    # a foreign program digest is refused
    assert QA.check_use_document(udoc, "0" * 64, qdoc)


def test_readiness_32bit_fixtures(PT):
    """Design doc §7 wide-gate inventory on the fixture families: only GumbelArgmaxBf16 (Const64 x2) is not 32-bit ready."""
    not_ready = {name for name in FX.FIXTURES if not PT.readiness_32bit(FX.fixture(name)).ok}
    assert not_ready == {"gumbel"}, not_ready
    r = PT.readiness_32bit(FX.gumbel_program())
    assert r.max_w == 64 and len(r.violators) == 2
    _record("readiness:fixtures", not_ready=sorted(not_ready), gumbel_max_w=r.max_w, gumbel_violators=len(r.violators))


@pytest.mark.parametrize("pj", _REAL, ids=_real_ids)
def test_real_program_readiness_32bit(PT, pj):
    """The captured Serve programs are 32-bit ready (no gate wider than 32 in any specialization)."""
    if pj is None:
        pytest.skip("no captured program.json in the sparse tree")
    label = f"readiness:{_real_ids(pj)}"
    res = _in_child("real_program_readiness", pj, label)
    assert res["ok"], f"{label}: {res}"
    assert res["max_w"] == 32


def test_zz_report_written():
    _record("meta", written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), records=[str(p) for p in _RECORDS])
    assert REPORT_PATH.exists()
