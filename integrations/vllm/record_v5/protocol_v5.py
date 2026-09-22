"""[A9 contract 2026-09-10] Steps 1-4 -- A3's contract-ordered live driver (``e2e.verifier_v3.driver``) composed by import, with the v5
population reader and instance loader installed, the F policy passed as A3's --exemption-policy (prover unchanged, descriptor limit raised), and A2's
``output_copy_checks`` added to the step-4 check.  D4's ``record_6k/protocol_6k.py`` pattern; nothing of A3's is edited.

Order (docs/PROTOCOL.md): prover phase 0 (step 1: boundary commitments incl. v_out(R) for every RU, weights root) -> verifier step-2 checks
(A3 ``steps.step2_check``: v_in(C) = registered weights root + request commitments, v_out(C) = the served outputs, typing) -> r_q (seed held
in this process) -> prover phase 2 (step 3: replay of J, interiors, exemption request with the two per-RU fields) -> verifier step-4 checks
(A3 ``steps.step4_check``: F ⊆ J, |F| <= B_ex RUs, retained interiors committed; PLUS A2 ``output_copy_checks``: the step-1 boundary copy and
the step-3 interior copy of every served token agree, both authenticated) -> r_s over J \\ F -> ``ru_fields`` (R17/R18).

    python -m record_v5.protocol_v5 --config run.json --record REC --instance-dir REC/instance --population population.v5.json
                                    --py-vllm PY --semantic-profile P.json [--unsupported-kinds REC/unsupported_kinds.json]
                                    [--expressibility REC/expressibility_v5.json] [driver args...]
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

from record_v5 import ROOT, deps
from record_v5 import config as cfgmod
from record_v5 import freeze
from record_v5 import instance_v5 as i5
from record_v5 import population_v5 as p5

deps.install()

from e2e import partition_sample as ps  # noqa: E402
from e2e.verifier_v3 import driver as drv  # noqa: E402
from e2e.verifier_v3 import steps as v3steps  # noqa: E402

_ORIG_LOAD_INSTANCE = ps.load_instance


def install_loaders() -> None:
    """Point A3's sampler / driver at the v5 population and instance loaders -- for this process, from ``main`` only (A7 finding
    22:05Z: a module import must not install anything process-wide; tests wrap what they need in ``loaders_installed()``)."""

    p5.install()
    ps.load_instance = i5.load_instance_v5
    drv.ps.load_instance = i5.load_instance_v5


@contextlib.contextmanager
def loaders_installed():
    prev = ps.load_instance
    install_loaders()
    try:
        yield
    finally:
        p5.uninstall()
        ps.load_instance = prev
        drv.ps.load_instance = prev

_RECORD = {"dir": None}
_Popen = drv.subprocess.Popen


class _PopenV5(_Popen):
    """A3's prover command runs unchanged (its ``--exemption-policy`` carries the F policy); this wrapper only raises the prover's descriptor
    limit -- D4 attempt 1 died in build_commitments with EMFILE (one memmap per RU step vs the pod's 1,024 soft limit; decisions.md 17:59Z)
    -- and records the limits beside the record."""

    def __init__(self, cmd, *a, **kw):
        if isinstance(cmd, list) and "verity_vllm_adapter.selective_prover" in cmd:
            kw["preexec_fn"] = _raise_nofile
            # [E10 EOD 2026-09-11] the interior trees of J from E1's interpreter values under E2's layout (INTEGRATOR D8; brief: "the interior
            # tree comes from E2's layout code over E1's interpreter values"): the runner names the staged rows (VERITOR_INTERIOR_VALUES =
            # <dir>/<req>/<step>/<call>.jsonl, E2's stage_e1 output) and the pinned call order (VERITOR_INTERIOR_CALLS); E2's prover flags
            # --interior-values / --interior-calls [/ --interior-strict] are appended here (A3's driver builds the command; the appended arguments
            # are recorded in logs/prover-interior.json and echoed below).  Unset = the legacy capture-only prover, unchanged.
            iv = os.environ.get("VERITOR_INTERIOR_VALUES")
            if iv and "--interior-values" not in cmd:
                extra = ["--interior-values", iv]
                if os.environ.get("VERITOR_INTERIOR_CALLS"):
                    extra += ["--interior-calls", os.environ["VERITOR_INTERIOR_CALLS"]]
                if os.environ.get("VERITOR_INTERIOR_STRICT") == "1":
                    extra += ["--interior-strict"]
                cmd = list(cmd) + extra
                if _RECORD["dir"] is not None:
                    (_RECORD["dir"] / "logs").mkdir(parents=True, exist_ok=True)
                    (_RECORD["dir"] / "logs" / "prover-interior.json").write_text(json.dumps({
                        "appended_by": "record_v5/protocol_v5._PopenV5 (E10)", "arguments": extra, "values_dir": iv,
                        "calls": os.environ.get("VERITOR_INTERIOR_CALLS"), "strict": os.environ.get("VERITOR_INTERIOR_STRICT") == "1",
                        "meaning": "phase 2 of the prover builds one interior tree per RU of J from these value rows (E1 interpreter, E2 layout interior-layout/chunks-4096/v1) "
                                   "and records the roots in commitments.json#interior_roots; the capture's own interior trees are not used for the interior words"}, indent=1) + "\n")
                print(f"[protocol_v5] prover interior arguments appended: {' '.join(extra)}", flush=True)
            # The prover's environment is A3's (driver.py: PYTHONPATH=. only).  The runner-side src/ append of 7ee14f8a (W-A9-10) was
            # removed on A3's request (for-a9.md §9, W-A9-11 landed 9dd2560e): the prover must not need src/, and the append would mask a recurrence.
            # [E10 NIGHT 2026-09-12] the prover's peak RSS (table field commit_peak_rss_gb) was never recorded: with VERITOR_PROVER_TIME=<file>
            # the prover runs under GNU `/usr/bin/time -v -o <file>` (stdio passes through untouched, exit status is the prover's); costlog_v5
            # reads the file into `prover_time` (peak RSS, user/sys/wall).  Unset = unchanged.
            pt = os.environ.get("VERITOR_PROVER_TIME")
            if pt and os.path.exists("/usr/bin/time") and cmd[0] != "/usr/bin/time":
                cmd = ["/usr/bin/time", "-v", "-o", pt] + list(cmd)
                print(f"[protocol_v5] prover wrapped in /usr/bin/time -v -o {pt}", flush=True)
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if _RECORD["dir"] is not None:
                (_RECORD["dir"] / "logs").mkdir(parents=True, exist_ok=True)
                (_RECORD["dir"] / "logs" / "prover-rlimit.json").write_text(json.dumps({"parent_soft": soft, "parent_hard": hard, "prover_soft_set_to": min(hard, 1_048_576) if hard != resource.RLIM_INFINITY else 1_048_576,
                                                                                         "reason": "D4 RECORD-6K attempt 1: EMFILE in build_commitments (3,200 memmapped RU steps > 1,024)"}) + "\n")
            print(f"[protocol_v5] prover command: {' '.join(map(str, cmd))} (RLIMIT_NOFILE soft {soft} -> hard {hard})", flush=True)
        super().__init__(cmd, *a, **kw)


def _raise_nofile() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = hard if hard != resource.RLIM_INFINITY else 1_048_576
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, target), hard))


drv.subprocess.Popen = _PopenV5

_orig_step4 = v3steps.step4_check


def step4_check_v5(inst, *, J, r_q, instance_digest, c0, c2, exreq, ru_of_request, selectable_uncovered=None):
    """A3's step-4 check, then A2's output-copy agreement over the retained RUs' requests (a disagreeing or missing copy refuses the step)."""

    problems, F, committed = _orig_step4(inst, J=J, r_q=r_q, instance_digest=instance_digest, c0=c0, c2=c2, exreq=exreq, ru_of_request=ru_of_request,
                                         selectable_uncovered=selectable_uncovered)
    rec = _RECORD["dir"]
    t0 = time.perf_counter()
    result: dict[str, Any] = {"schema": "veritor/a9/output-copy-checks/v1", "checker": "verity_vllm_adapter.obligations.output_copy_checks (A2, merged b47e0d85)", "rows": [], "status": None}
    try:
        from verity_vllm.capture import CaptureReader
        from verity_vllm.commit import TreeCache
        from verity_vllm_adapter.obligations import output_copy_checks

        sel = rec / "adapter" / "selective"
        reader = CaptureReader(sel)
        cache = TreeCache(reader, recompute_leaves=True)
        retained_rids = sorted(rid for rid, r in ru_of_request.items() if r in set(committed))
        rows = output_copy_checks(reader, c2, cache, request_ids=retained_rids)
        bad = [r for r in rows if not r.get("ok")]
        result.update(rows=rows, requests=retained_rids, rows_total=len(rows), rows_bad=len(bad), status="ok" if rows and not bad else ("no rows" if not rows else "DISAGREE"))
        if bad:
            problems = problems + [f"output_copy_checks: {len(bad)} of {len(rows)} served-token copies disagree or are missing (A2 R4): " + "; ".join(str(b.get("problem", b))[:120] for b in bad[:3])]
        elif not rows and retained_rids:
            problems = problems + ["output_copy_checks: no interior copy rows for the retained RUs (nothing to compare; the step-3 commitment holds no sampler.select copies)"]
    except Exception as e:  # noqa: BLE001 -- reported, never silently passed
        result.update(status="NOT DONE", error=f"{type(e).__name__}: {e}")
        problems = problems + [f"output_copy_checks NOT DONE ({type(e).__name__}: {str(e)[:160]}): step 4 cannot pass without the agreement check"]
    result["seconds"] = round(time.perf_counter() - t0, 3)
    (rec / "verify").mkdir(parents=True, exist_ok=True)
    (rec / "verify" / "output_copy_checks.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"[protocol_v5] step 4 + output_copy_checks: {result['status']} ({result.get('rows_total', 0)} rows, {result.get('rows_bad', 0)} bad) in {result['seconds']} s", flush=True)
    return problems, F, committed


A3_HAS_OUTPUT_COPIES = "output_copy_checks" in inspect.getsource(v3steps) or "output_copy_checks" in inspect.getsource(drv)
if not A3_HAS_OUTPUT_COPIES:
    # the A3 drop on mainline a8e38f9a runs step4_check without A2's agreement check; A3's later drop (3a10d04b) has it -- then nothing is added
    v3steps.step4_check = step4_check_v5
    drv.steps.step4_check = step4_check_v5


def main(argv: list[str] | None = None) -> int:
    install_loaders()
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in argv:
        print("[protocol_v5] --config is required (the run config carries the F policy and the legacy guard)")
        return 2
    i = argv.index("--config")
    cfg = cfgmod.load(argv[i + 1])
    del argv[i:i + 2]
    # the F policy is A3's driver/prover option (--exemption-policy gamble|exempt-unsupported); the run config's name is mapped, never both given
    if "--exemption-policy" in argv:
        print("[protocol_v5] E_CONFIG: --exemption-policy comes from the run config (f_policy), not the command line")
        return 2
    argv += ["--exemption-policy", cfgmod.A3_POLICY[cfg.f_policy]]
    # A3 9dd2560e/b0f84e4e (mainline 89d5c984): the driver selects the instance/population loaders by name (--instance-format, default
    # 'contract' = partition-intervals/v1) and refuses a mismatch before anything is drawn; the v5 record declares a9/legacy-minus-inputs/v1
    if "--instance-format" not in argv:
        argv += ["--instance-format", "record_v5"]
    rec = Path(argv[argv.index("--record") + 1])
    _RECORD["dir"] = rec
    freeze.check(rec, "protocol")
    if cfg.is_production and "--condition-nonempty" in argv:
        print("[protocol_v5] E_CONFIG: --condition-nonempty (redraw of an empty r_q) is a reproducibility-mode option; production keeps a zero-size sample as a valid outcome")
        return 2
    if cfg.is_production and "--allow-legacy-seed-file" in argv:
        print("[protocol_v5] E_CONFIG: --allow-legacy-seed-file is not a production option (R7: the seed stays in the verifier process)")
        return 2
    (rec / "protocol_v5.json").write_text(json.dumps({
        "schema": "veritor/a9/protocol-composition/v1", "run_id": cfg.run_id, "f_policy": cfg.f_policy, "randomness_mode": cfg.randomness_mode,
        "epoch": cfg.epoch.as_json(), "driver": drv.__file__, "steps": v3steps.__file__, "population_reader": "record_v5/population_v5.load_population_v5 (installed into e2e.partition_sample)",
        "instance_loader": "record_v5/instance_v5.load_instance_v5", "prover": "A3 verity_vllm_adapter.selective_prover (unchanged; --exemption-policy from the run config)", "a3_exemption_policy": cfgmod.A3_POLICY[cfg.f_policy],
        "step4_output_copies": "A3's steps.py runs A2's output_copy_checks itself" if A3_HAS_OUTPUT_COPIES else "added by record_v5.protocol_v5.step4_check_v5 (A2 output_copy_checks over the retained RUs)",
        "argv": argv, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    # [USER DIRECTIVE 20:10Z §5] a prover abort is not a verifier rejection: the driver's exit reason is classified and recorded
    t0 = time.time()
    try:
        rc = drv.main(argv)
        reason = None
    except SystemExit as e:  # the driver's refusals / aborts / timeouts all leave through SystemExit with the reason
        rc = e.code if isinstance(e.code, int) else 3
        reason = str(e.code) if not isinstance(e.code, int) else None
    except Exception as e:  # noqa: BLE001 -- an uncaught driver exception (attempt 4 of V5-TINY-bex0-inc-elem: KeyError at driver.py:314, W-A9-13)
        # is a driver error, recorded as such with its traceback; the run must still get its protocol_exit.json
        import traceback
        rc, reason = 1, f"driver exception {type(e).__name__}: {e}"
        (rec / "logs").mkdir(parents=True, exist_ok=True)
        (rec / "logs" / "driver-traceback.txt").write_text(traceback.format_exc())
    finally:
        freeze.check(rec, "protocol-end")
    vx = json.loads((rec / "verifier_exit.json").read_text()) if (rec / "verifier_exit.json").exists() else None
    kind = REASON_CODE_KIND.get((vx or {}).get("reason_code"), None) or classify_exit(rc, reason)
    (rec / "protocol_exit.json").write_text(json.dumps({"schema": "veritor/a9/protocol-exit/v1", "rc": rc, "reason": reason, "kind": kind,
                                                        "kind_source": "verifier_exit.json#reason_code (A3 ckpt 5, W-A9-4)" if vx else "message text (classify_exit; no verifier_exit.json)",
                                                        "verifier_exit": vx,
                                                        "wall_s": round(time.time() - t0, 3), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    print(f"[protocol_v5] driver exit rc={rc} kind={kind}" + (f" reason={reason[:200]}" if reason else ""), flush=True)
    return rc


# A3's verifier_exit.json reason codes -> the runner's kinds (OK 0, CHRONOLOGY_MISMATCH 1, USAGE 2, STEP2_REFUSED 20, PROVER_ABORT 30, PROVER_EXIT 31, TIMEOUT 32, STEP4_REFUSED 40)
REASON_CODE_KIND = {"OK": "completed", "CHRONOLOGY_MISMATCH": "sampler_verify_problems", "USAGE": "driver_error", "STEP2_REFUSED": "verifier_refusal_step2",
                    "STEP4_REFUSED": "verifier_refusal_step4", "PROVER_ABORT": "prover_abort", "PROVER_EXIT": "prover_exit", "TIMEOUT": "timeout"}
EXIT_KINDS = ("completed", "sampler_verify_problems", "verifier_refusal_step2", "verifier_refusal_step4", "prover_abort", "prover_exit", "timeout", "driver_error")


def classify_exit(rc: int, reason: str | None) -> str:
    """The driver's exit, told apart: a verifier REFUSAL (step 2 / step 4, the verifier's decision on the prover's record), a PROVER ABORT
    (the prover wrote handshake/ABORT or exited before its phase), a timeout, or a driver error (none of those: a bug or a missing file)."""
    if reason is None:
        return "completed" if rc == 0 else "sampler_verify_problems" if rc == 1 else "driver_error"
    r = reason.lower()
    if r.startswith("driver exception"):
        return "driver_error"
    if "step2 refused" in r or "step2_refused" in r:
        return "verifier_refusal_step2"
    if "step4 refused" in r or "step4_refused" in r:
        return "verifier_refusal_step4"
    if "prover abort" in r:
        return "prover_abort"
    if "prover exited" in r:
        return "prover_exit"
    if "timeout" in r:
        return "timeout"
    return "driver_error"


if __name__ == "__main__":
    raise SystemExit(main())
