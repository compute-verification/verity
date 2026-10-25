"""A single-GPU row's required-value manifest of record (`row_stages.SingleRow`), and at B >= 2 the work its Build runs beside the
derives: the workload Program's compose, which loads each component as its derive finishes, and the manifest, which builds its
components while the compose runs and takes the workload digest once it is composed."""
from __future__ import annotations

import contextlib
import glob
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from verity_vllm.pipeline import row_records as R
from verity_vllm.program import prefix_plan as PP


def required_manifest(row: Any, stage: str, digest: Callable[[], str] | None = None) -> int | None:
    """Build the required-value manifest of record (the query of record over the Build's request Programs, under the tap and
    guarded-max policies the row runs: all by default, NORM_TAP=0 etc. for A/B only) -> 0 | 3 | 1.  `digest`: the workload digest,
    asked for once a batch's manifest has started (`build-global --workload-digest -`), so its components build while the workload
    Program composes; "" (no workload Program) -> None, and no manifest."""
    wd = ""
    policy = R.tap_policy_flags(row.ck_)
    row.tl(f"{stage}.required_manifest", "start", ("query", R.QUERY_OF_RECORD))
    if row.B > 1 and glob.glob(f"{row.d}/build_request_LP*_T*"):
        wd = "-" if digest else R.workload_digest(row.d)
        cmd = row.cli("manifest", "build-global", "--fixture", row.wl, "--programs-root", row.d,
                      *(["--workload-digest", wd] if wd else []), *policy, "--out", f"{row.d}/manifest.json")
    else:
        cmd = row.cli("manifest", "build", "--program", f"{row.d}/build_request", "--workload", row.wl, *policy,
                      "--out", f"{row.d}/manifest.json")
    log = f"{row.d}/manifest.log"
    piped = wd == "-"
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.PIPE if piped else None, cwd=row.tree,
                             env=row.child_env)
        if piped:
            wd = ""
            try:
                wd = digest()
            finally:
                with contextlib.suppress(BrokenPipeError):
                    p.stdin.write(f"{wd}\n".encode())
                    p.stdin.close()
        rc = p.wait()
    if piped and not wd:
        if os.path.exists(f"{row.d}/manifest.json"):
            os.remove(f"{row.d}/manifest.json")
        row.tl(f"{stage}.required_manifest", "end", ("rc", "none"))
        return None
    if rc == 0:
        row.tl(f"{stage}.required_manifest", "end", ("rc", 0))
        row.log(f"required-value manifest of record ({R.QUERY_OF_RECORD}, B={row.B}{', workload digest ' + wd[:16] if wd else ''}): {R.tail(log)}")
        return 0
    if os.path.exists(f"{row.d}/manifest.json"):
        os.remove(f"{row.d}/manifest.json")
    hit = next((ln for ln in open(log, errors="replace") if "PRECHECK_FAIL_REGISTRY_UNAVAILABLE" in ln), None)
    if hit is not None:
        row.log(f"PRECHECK_FAIL_REGISTRY_UNAVAILABLE: {hit.rstrip(chr(10))}")
        row.tl(f"{stage}.required_manifest", "end", ("rc", 3), ("fail_class", "PRECHECK_FAIL_REGISTRY_UNAVAILABLE"))
        return 3
    row.tl(f"{stage}.required_manifest", "end", ("rc", 1))
    row.log(f"required-value manifest NOT built (rc={rc}; see manifest.log): {R.tail(log)}")
    return 1


def follow_compose(row: Any, to: int) -> tuple[Any, str, str]:
    """B >= 2: the workload Program's compose starts beside the derives (`global-program --follow`) and loads each component as its
    derive finishes; the required-value manifest starts once every request's shape is derived and builds its components while the
    compose runs, taking the workload digest when the compose ends; and the envelope (`build_request`, which no request runs unless
    its (LP, T) is a request's shape, and then both wait for it) may still derive -> (their future: (compose rc, manifest rc or None
    when none was built), the derives-done marker, the derives-failed marker).  `build` keeps that manifest only for a PASS Build."""
    done, stop = f"{row.d}/build_workload.derives_done", f"{row.d}/build_workload.derives_failed"
    for f in (done, stop):
        if os.path.exists(f):
            os.remove(f)
    wl = row.path(row.wl)
    dirs = [*([f"{row.d}/build_request"] if (row.LP, row.T) in R.request_shapes(wl) else []),
            *(f"{row.d}/build_request_{PP.shape_name(s)}" for s in R.extra_shapes(wl, row.LP, row.T))]
    cmd = row.cli("global-program", "--workload", row.wl, *(x for d in dirs for x in ("--derived", d)), "--out", f"{row.d}/build_workload",
                  "--checkpoint-dir", row.ck, "--follow", str(to), "--follow-done", done, "--follow-stop", stop, py=row.py312)
    pool = ThreadPoolExecutor(max_workers=3)
    compose = pool.submit(row.run_tl, "build.global_program", to, f"{row.d}/build_workload.log", cmd)

    def workload_digest() -> str:
        return R.workload_digest(row.d) if compose.result() == 0 and not row.signalled else ""

    def manifest() -> int | None:
        return row.required_manifest("build", workload_digest) if wait_derived(row, dirs, done, stop) else None
    man = pool.submit(manifest)
    fut = pool.submit(lambda: (compose.result(), man.result()))
    pool.shutdown(wait=False)
    return fut, done, stop


def wait_derived(row: Any, dirs: list[str], done: str, stop: str) -> bool:
    """Wait until every one of `dirs` is derived (True), or the derives end without them, fail, or the row is signalled (False)."""
    while not os.path.exists(stop) and not row.signalled:
        over = os.path.exists(done)
        if all(R.derived_ok(d) for d in dirs):
            return True
        if over:
            return False
        time.sleep(1)
    return False
