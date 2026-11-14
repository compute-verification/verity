"""v2's check of a per-member case on a stand-in window of many members over few templates, through `case_files`.

    members_standin.py OUT FLOCK_CIRCUIT LEAN_BIN_DIR MEMBERS...        e.g.  300 100

Each window holds MEMBERS members of one unit: `test_audit_two_members`' two templates (`ScaledMmS8Coordinate` at
K = 128 and K = 256) with half the units each, recorded, committed and written as serving writes a window, under
``stratified:1``: a template is a stratum, so one member of each is sampled and the rest share their template's circuit
unsampled, as a served window's members do. The served harness's audit step (``records_audit.py audit`` on a local
challenger, ``--deployed-key``, FLOCK_CHALLENGER=1) audits it and writes the case; `case_files.run` then runs v2's
``refusal`` on it. Per window, OUT/m<MEMBERS>/result.json holds the audit's exit, seconds and peak, the members' staged
directories with no circuit file of their own (each a file the per-member case named before each member named its
template's), and the refusal's line, exit, seconds and peak, with ``all_passed`` whether the line is
`members.ALL_PASSED` (every check before the per-member form's refusal passed)."""
from __future__ import annotations

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
TEMPLATES = tuple(bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM120_S8) for k in KS)
DESIGN = "stratified:1"
RUN = "ab" * 64


def window(d: Path, members: int) -> Path:
    """The stand-in window of ``members`` one-unit members (module docstring), written to ``d``."""
    n = members // len(TEMPLATES)
    win = IF.Window(IF.Circuit("standin-members", tuple((t, n) for t in TEMPLATES)), RUN, D.parse_design(DESIGN), 1)
    rng = np.random.default_rng(0)
    for t, k in zip(TEMPLATES, KS):
        x, w = (rng.integers(-128, 128, (n, k)).astype(np.int8) for _ in range(2))
        sx, sw = ((np.abs(rng.standard_normal(n)) / 127).astype(np.float32) for _ in range(2))
        ins = [x.view(np.uint8), sx.view(np.uint32), w.view(np.uint8), sw.view(np.uint32)]
        win.record(t, ins, [np.asarray(evaluate_batch(t, *ins, kernel="reference"), dtype=np.uint16)])
    win.write(win.close(salt_key=bytes(32)), d, {"label": "standin"}, registrar=R)
    return d


def audit_argv(win: Path, out: Path, main: Path, flock: Path, lean: Path) -> list[str]:
    m0 = os.pathsep.join(str(ROOT / p) for p in ("", "benchmarks/numerical/python", "verity/protocols/proofs/flock/prover/python",
                                                  "adapters/vllm"))
    return [sys.executable, str(ROOT / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "one_stage" / "records_audit.py"), "audit", "--window", str(win),
            "--law", DESIGN, "--circuit", hashlib.sha512((win / "circuit.json").read_bytes()).hexdigest(), "--run", RUN,
            "--challenger", "local", "--main", str(main), "--m0-pythonpath", m0, "--bin", str(flock),
            "--lean", str(lean / "verity-verify-proof"), "--firewall", str(lean / "flock-firewall"), "--ports", str(len(KS)),
            "--out", str(out), "--deployed-key"]


def one(root: Path, members: int, flock: Path, lean: Path) -> dict:
    d = root / f"m{members}"
    d.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"standin-m{members}-"))
    t = time.monotonic()
    win = window(work / "window", members)
    res: dict = {"members": members, "templates": len(KS), "window_s": round(time.monotonic() - t, 1)}
    main = work / "main.txt"
    main.write_text(subprocess.run(["git", "-C", str(ROOT), "rev-list", "--first-parent", "HEAD"], capture_output=True,
                                   text=True, check=True).stdout)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FC_", "FLOCK_")) and k != "PYTHONPATH"}
    env.update(RAYON_NUM_THREADS=str(os.cpu_count()), RUST_MIN_STACK=str(64 << 20), FLOCK_CHALLENGER="1")
    t = time.monotonic()
    p = subprocess.run(audit_argv(win, work / "audit", main, flock, lean), capture_output=True, text=True, env=env)
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
        got = CF.run(case, PF.v2_executable(), "refusal", work / "refusal")
        (d / "refusal.err").write_text((work / "refusal" / "stderr").read_text()[-20000:])
        line = json.loads(got["line"]) if got["line"] else {}
        res["refusal"] = {**got, "all_passed": line.get("refusal") == M.ALL_PASSED and line.get("decided") is True}
    (d / "result.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    return res


def main() -> int:
    root, flock, lean, *sizes = sys.argv[1:]
    for n in map(int, sizes):
        print(json.dumps(one(Path(root), n, Path(flock), Path(lean)), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
