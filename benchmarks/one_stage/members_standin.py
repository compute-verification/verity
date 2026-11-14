"""v2's check of a per-member case on a stand-in window of many members over few templates, through `case_files`.

    members_standin.py OUT FLOCK_CIRCUIT LEAN_BIN_DIR [--templates T] MEMBERS...        e.g.  300 100

Each window holds MEMBERS members of one unit: `test_audit_two_members`' two templates (`ScaledMmS8Coordinate` at
K = 128 and K = 256) with half the units each, recorded, committed and written as serving writes a window, under
``stratified:1``: a template is a stratum, so one member of each is sampled and the rest share their template's circuit
unsampled, as a served window's members do. ``--templates T`` spreads the same MEMBERS over T templates instead
(`ks_of`: a further K each, `window`), so the program has T nodes and the audit T sessions at one member count. The
served harness's audit step (``records_audit.py audit`` on a local challenger, ``--deployed-key``, FLOCK_CHALLENGER=1)
audits it and writes the case; `case_files.run` then runs v2's ``check`` on it. Per window, OUT/m<MEMBERS>/result.json
(OUT/m<MEMBERS>-t<T>/ under ``--templates``) holds the audit's exit, seconds and peak, the members' staged directories
with no circuit file of their own (each a file the per-member case named before each member named its template's),
and the check's line, exit, seconds and peak, with ``accepted`` whether it accepts the case."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import case_files as CF  # noqa: E402

from verity.circuits.defs import bind  # noqa: E402
from verity.circuits.evaluation import evaluate_batch  # noqa: E402
from verity.commitments.windows import interface as IF  # noqa: E402
from verity.circuits.ops import gemm as G  # noqa: E402
from verity.protocols.proofs.spot.verifier.one_stage import draw as D, members as M, proofs as PF, registration as R  # noqa: E402

KS = (128, 256)
DESIGN = "stratified:1"
RUN = "ab" * 64


def ks_of(templates: int) -> tuple[int, ...]:
    """The K of each of ``templates`` templates: `KS`, then the other multiples of the step's 32 from `KS`'s least up
    (160, 192, 224, 288, ...), so two templates are `KS` and each further one is the next smallest circuit."""
    ks = (*KS, *(k for k in range(KS[0], G.S8_K_LIMIT, 32) if k not in KS))
    if not 1 <= templates <= len(ks):
        raise ValueError(f"{templates} templates: the stand-in has from 1 to {len(ks)}, one a K from {KS[0]} below "
                         f"{G.S8_K_LIMIT}")
    return ks[:templates]


def window(d: Path, members: int, ks: tuple[int, ...] = KS) -> Path:
    """The stand-in window of ``members`` one-unit members (module docstring) over a template for each K of ``ks``,
    written to ``d``: the templates hold the members as evenly as they go, the first ``members % len(ks)`` one more."""
    templates = tuple(bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM120_S8) for k in ks)
    ns = [members // len(ks) + (i < members % len(ks)) for i in range(len(ks))]
    if not ns[-1]:
        raise ValueError(f"{members} members over {len(ks)} templates leave a template none: give at least {len(ks)}")
    win = IF.Window(IF.Circuit("standin-members", tuple(zip(templates, ns))), RUN, D.parse_design(DESIGN), 1)
    rng = np.random.default_rng(0)
    for t, k, n in zip(templates, ks, ns):
        x, w = (rng.integers(-128, 128, (n, k)).astype(np.int8) for _ in range(2))
        sx, sw = ((np.abs(rng.standard_normal(n)) / 127).astype(np.float32) for _ in range(2))
        ins = [x.view(np.uint8), sx.view(np.uint32), w.view(np.uint8), sw.view(np.uint32)]
        win.record(t, ins, [np.asarray(evaluate_batch(t, *ins, kernel="reference"), dtype=np.uint16)])
    win.write(win.close(salt_key=bytes(32)), d, {"label": "standin"}, registrar=R)
    return d


def audit_argv(win: Path, out: Path, main: Path, flock: Path, lean: Path, ports: int = len(KS)) -> list[str]:
    """The served harness's audit step on ``win``, a session's port for each of its ``ports`` templates."""
    m0 = os.pathsep.join(str(ROOT / p) for p in ("", "benchmarks/numerical/python", "verity/protocols/proofs/flock/prover/python",
                                                  "adapters/vllm"))
    return [sys.executable, str(ROOT / "benchmarks" / "prove_path" / "records_audit.py"), "audit", "--window", str(win),
            "--law", DESIGN, "--circuit", hashlib.sha512((win / "circuit.json").read_bytes()).hexdigest(), "--run", RUN,
            "--challenger", "local", "--main", str(main), "--m0-pythonpath", m0, "--bin", str(flock),
            "--lean", str(lean / "flock-verify"), "--firewall", str(lean / "flock-firewall"), "--ports", str(ports),
            "--out", str(out), "--deployed-key"]


def one(root: Path, members: int, flock: Path, lean: Path, ks: tuple[int, ...] = KS) -> dict:
    d = root / (f"m{members}" if ks == KS else f"m{members}-t{len(ks)}")
    d.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"standin-m{members}-"))
    t = time.monotonic()
    win = window(work / "window", members, ks)
    res: dict = {"members": members, "templates": len(ks), "window_s": round(time.monotonic() - t, 1)}
    main = work / "main.txt"
    main.write_text(subprocess.run(["git", "-C", str(ROOT), "rev-list", "--first-parent", "HEAD"], capture_output=True,
                                   text=True, check=True).stdout)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FC_", "FLOCK_")) and k != "PYTHONPATH"}
    env.update(RAYON_NUM_THREADS=str(os.cpu_count()), RUST_MIN_STACK=str(64 << 20), FLOCK_CHALLENGER="1")
    t = time.monotonic()
    p = subprocess.run(audit_argv(win, work / "audit", main, flock, lean, len(ks)), capture_output=True, text=True,
                       env=env)
    (d / "audit.log").write_text(p.stdout)
    (d / "audit.err").write_text(p.stderr)
    res["audit"] = {"rc": p.returncode, "seconds": round(time.monotonic() - t, 1),
                    "peak_rss_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    staged = [x for x in (work / "audit" / "window").glob("m*") if x.is_dir() and x.name[1:].isdigit()]
    res["staged"] = {"members": len(staged), "no_own_circuit": sum(not (x / "circuit.txt").exists() for x in staged)}
    cfile = work / "audit" / "window" / "case-v2-members.json"
    for name in ("case-v2-members-error.txt", "summary.json"):
        if (work / "audit" / "window" / name).is_file():
            (d / name).write_bytes((work / "audit" / "window" / name).read_bytes())
    if cfile.is_file():
        case = json.loads(cfile.read_text())
        res["case"] = {"members": len(case["params"]["proofs"]["members"]), "blobs": len(case["blobs"]),
                       "statements": len(json.loads(bytes.fromhex(case["evidence"]))["proofs"]["statements"])}
        got = CF.run(case, PF.v2_executable(), "check", work / "check")
        (d / "check.err").write_text((work / "check" / "stderr").read_text()[-20000:])
        line = json.loads(got["line"]) if got["line"] else {}
        res["check"] = {**got, "accepted": got["rc"] == M.EXITS["accept"] and line.get("verdict") == "accept"}
    (d / "result.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("flock", type=Path, help="flock-circuit, as check_build.sh bin builds it")
    ap.add_argument("lean", type=Path, help="the directory of the Lean verifier's and the ZK server's executables")
    ap.add_argument("members", type=int, nargs="+", help="a window's members, a window each")
    ap.add_argument("--templates", type=int, default=len(KS), help="the templates the members are spread over")
    a = ap.parse_args()
    for n in a.members:
        print(json.dumps(one(a.out, n, a.flock, a.lean, ks_of(a.templates)), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
