"""The config run's and the Program cache's parts of a single-GPU row (`row_stages.SingleRow`) and a TP2 row (`row_tp.TpRow`: the word
check over the ranks' components, the record per rank), docs/vllm-config-sweep-plan.md §2-§4.

A config run (`--config-run 1`) is the Build, the strict word check of its manifest (`word_check`), ONE instrumented Commit (`--pairs 1
--only-arm instrumented`: no control arm, no pairs) under bounded staging with a k-unit replay stratified by family, and
`config_record.json` (`config_record`), written however the run ends.  No Match, no manifest-verify rebuild, no VU export: the Commit's
decision is the instrumented run's own verdict (`commit_verdict`: coverage, openings, the replay).  Without a Match there is no account
for the executed-prefix class (`check.executed_prefix`) to link to: when every request ran to its cap the class has nothing to classify and the
replay's population is the whole manifest; a request stopped before its cap fails the Commit by name, as a missing input.  Every derive of
a Build goes through the host's Program cache when one is named (`derive_cached`, `pipeline.program_cache`).
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

from verity_vllm.pipeline import config_record as CR
from verity_vllm.pipeline import program_cache as PC
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline.row_driver import Stop

WORD_CHECK_TIMEOUT = 14400


def _tail(path: str, n: int = 240) -> str:
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return ""
    return (lines[-1] if lines else "")[:n]


def word_check(row: Any) -> None:
    """The strict word check of the Build's manifest (`epoch word-check`, as a row of record's job runs it) before any Commit; a FAIL stops
    the run (exit 20)."""
    env = row.env.environ
    rc = row.run_tl("build.word_check", WORD_CHECK_TIMEOUT, f"{row.d}/strict_word.log",
                    row.cli("epoch", "word-check", row.row, row.d, "--out", f"{row.d}/strict_word.json", "--world", str(getattr(row, "W", 1)),
                            "--build-word-check", env.get("VERITY_WORD_CHECK", ""), "--build-max-gates", env.get("VERITY_QWORD_MAX_GATES", "")))
    row.verdict("word-check", "PASS" if rc == 0 else "FAIL", f"rc={rc} {_tail(f'{row.d}/strict_word.log')}")
    if rc != 0:
        row.log("strict word check FAIL -> stop")
        row.tl("row", "end", ("ok", "false"), ("stopped_at", "word_check"))
        raise Stop(20)


def commit_verdict(row: Any, rck: int, wall: int) -> None:
    """The Commit's decision by its one instrumented run (`commit/summary.json`, only-arm): there is no control arm or Match for `verdict
    from-record` to grade."""
    sm = R.load_json(f"{row.d}/commit/summary.json") or {}
    ok = "PASS" if rck == 0 and sm.get("commit_pass") is True else "FAIL"
    row.verdict("commit", ok, f"rc={rck} wall={wall}s config run: 1 instrumented run, commit_pass={sm.get('commit_pass')}")
    row.tl("stage.commit", "end", ("ok", ok), ("wall_s", wall))
    if ok != "PASS":
        row.log("COMMIT FAIL -> stop")
        row.tl("row", "end", ("ok", "false"), ("stopped_at", "commit"))
        raise Stop(12)


def config_record(row: Any) -> dict[str, Any]:
    """`config_record.json` beside the row (`pipeline.config_record`) and its verdict line."""
    st = f"{row.d}/stages.txt"
    stages = open(st, errors="replace").read().splitlines() if os.path.isfile(st) else []
    rec = CR.write(row.d, row.row, tree=row.env.environ.get("RESEARCH_SOURCE_SHA") or row.code9,
                   device=R.load_json(f"{row.d}/target_family.json"), stages={"lines": stages[-24:]})
    rp = rec["replay"]
    row.verdict("config", rec["outcome"], rec["why"] or f"replay {rp.get('equal')}/{rp.get('evaluated')} equal (k={rp.get('k')}, "
                                                          f"{rp.get('strata_by')} strata), run root {str(rec['commit'].get('run_root'))[:16]}")
    return rec


def device_id(row: Any) -> dict[str, Any] | None:
    """What the derives are for, part of a derive's cache key: the row's declared target when it declares one (the Build derives for it
    on any host, a GPU-free one included), else the GPU model the host derives on (name, compute capability); None when neither is known."""
    if not hasattr(row, "_device_id"):
        declared = getattr(row, "tgt", "")
        first: list[str] = []
        if not declared:
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
                first = (r.stdout.splitlines() or [""])[0].split(",") if r.returncode == 0 else []
            except OSError:
                first = []
        row._device_id = ({"declared_target": declared} if declared else
                          {"name": first[0].strip(), "compute_cap": first[1].strip()} if len(first) == 2 else None)
    return row._device_id


def derive_cached(row: Any, st: str, to: int, lf: str, cmd: list[str], out: str, *extra: tuple[str, Any]) -> int:
    """One derive through the host's Program cache (`--program-cache`): an entry of the same key is restored into `out`; else the derive
    runs and, on success, publishes its dir.  Without the cache (or a key) it is the plain derive."""
    root = row.cb.program_cache
    tree = row.env.environ.get("RESEARCH_SOURCE_SHA")
    k = PC.key(cmd, workload=row.path(row.wl), tree=tree, device=device_id(row)) if root else None
    hit = PC.lookup(root, k) if k else None
    if hit:
        PC.restore(hit, out)
        with open(lf, "w") as f:
            f.write(f"program cache hit {k[:16]}: {hit}\n")
        row.tl(st, "mark", ("cache", "hit"), ("key", k[:16]))
        return 0
    rc = row.run_tl(st, to, lf, cmd, *extra)
    if k and rc == 0 and PC.publish(root, k, out, {"row": row.row, "stage": st, "tree": tree, "device": device_id(row)}):
        row.tl(st, "mark", ("cache", "published"), ("key", k[:16]))
    return rc
