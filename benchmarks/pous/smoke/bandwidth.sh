#!/usr/bin/env bash
# The bandwidth audit's P2 session run on a processor, one honest run: the toy's 8-unit store served through the
# reference decode on the CPU, against the auditor gate's party shape (`p2_party_at_leaf.py`, in `tests/benchmarks/pous`)
# on this machine, its close signed at the deployed parameters under a test key made without a keygen, so a rehearsal that
# certifies nothing, for finding what breaks. The session is the auditor's (`p2_parties.SESSION`: 6 windows of 20 s, 4
# passes, 1,024 openings, a 1 s allowance: `bandwidth.py session`'s defaults), which the Lean holds every request to, so a run takes about three minutes of
# window time. From the tree's root, with `uv sync --all-packages --extra torch-cpu` done and elan on the PATH:
#
#   bash benchmarks/pous/smoke/bandwidth.sh
#
# It builds `poub-p2-protocol` and asks it the honest session's case; exit 0 where it answers accept.
set -o pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
export PATH="$HOME/.elan/bin:$PATH" LEAN_ABORT_ON_PANIC=1
PY=${PY:-$ROOT/.venv/bin/python}
J=verity/protocols/accounting/memory/poub/verifier/lean
(cd "$J" && lake build poub-p2-protocol) > "$(mktemp)" 2>&1 || { echo "poub-p2-protocol doesn't build"; exit 7; }
EXE=$ROOT/$J/.lake/build/bin/poub-p2-protocol
echo "poub-p2-protocol sha512 $(sha512sum "$EXE" | cut -c1-128)"
exec "$PY" - "$EXE" <<'PY'
import hashlib, json, sys, tempfile
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT), str(ROOT / "tests/benchmarks/pous"), str(ROOT / "tests/verity/protocols/accounting/memory/pous/verifier"),
                str(ROOT / "benchmarks/pous"), str(ROOT / "verity/protocols/accounting/memory/poub/prover/p2_stage")]
import test_pous_p2_bandwidth as T  # noqa: E402

exe, tmp = sys.argv[1], Path(tempfile.mkdtemp())
reg = T.BT.toy(tmp / "toy")


def run(name: str, trusted: str) -> dict | None:
    (tmp / f"challenger-{name}").mkdir()
    T.mc.SessionLog.create(tmp / f"challenger-{name}" / "sessions")
    party = (f"cd {ROOT} && exec {sys.executable} -S -E {ROOT / 'tests/benchmarks/pous/p2_party_at_leaf.py'} "
             f"--device {tmp / ('party-' + name)} --listen 127.0.0.1:0 --registration {tmp / 'toy' / 'registration.json'} "
             f"--challenger {tmp / ('challenger-' + name)} --session {hashlib.sha512(name.encode()).hexdigest()[:64]} "
             f"--trusted-sha512 {trusted}")
    private = tmp / "private" / name
    rc = T.B.main(["session", "--store", str(tmp / "toy" / "store"), "--layout", str(tmp / "toy" / "layout.json"), "--node", reg.node.decode(),
                   "--party", party, "--party-host", "127.0.0.1", "--device", "cpu", "--ref-decode", "--workers", "2", "--chunk", "5",
                   "--out", str(tmp / name), "--expect", "accept", "--case-dir", str(private), "--case-exe", exe])
    answer = json.loads((private / "answer.json").read_text()) if (private / "answer.json").exists() else None
    print(f"{name}: session rc={rc}; the Lean says {({k: answer.get(k) for k in ('verdict', 'reason')}) if answer else None}", flush=True)
    return answer


honest = run("honest", T.AT_LEAF)
sys.exit(0 if (honest or {}).get("verdict") == "accept" else 1)
PY
