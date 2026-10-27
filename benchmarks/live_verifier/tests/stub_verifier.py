"""verifier.py as the host's runner starts it, with the prover in this process: given the gate's command line, it runs one paced
session over the remote audit's in-process harness (`benchmarks/remote_session/tests/test_session.py`'s `run`, a CPU prover on
a pass clock, the Lean verdict) with its transcript in `--transcript`, and leaves `--out` as verifier.py does: `verifier.json`
(its sessions, registration and registry), `challenger/`, the registry at `--registry`. It prints the listening line the gate's
`status` reads."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RS = Path(__file__).resolve().parents[2] / "remote_session"
sys.path.insert(0, str(RS / "tests"))

import test_session as T  # noqa: E402
import verifier  # noqa: E402
import wire  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    for flag in ("--out", "--transcript", "--registry", "--port"):
        ap.add_argument(flag, required=True)
    a, _ = ap.parse_known_args()
    out, transcript = Path(a.out), Path(a.transcript)
    out.mkdir(parents=True, exist_ok=True)
    (out / "challenger").mkdir(exist_ok=True)
    print(f"verifier: listening on 127.0.0.1:{a.port}", flush=True)
    got: dict = {}

    def then(v) -> None:
        got.update(registration=wire.registration_json(v.registration), registry=v.registry_json())

    sessions, _ = T.run("honest", paces=[verifier.pace_of("2:8:40:2", 1.0)], transcript=transcript, then=then)
    Path(a.registry).write_text(json.dumps(got["registry"], indent=1))
    (out / "verifier.json").write_text(json.dumps(
        {"schema": "verity/remote-session/verifier/v0", "peer": ["127.0.0.1", 0], "registration": got["registration"],
         "verdict": "lean", "sessions": sessions, "registry": got["registry"]}, indent=1))
    for k, s in enumerate(sessions):
        print(f"verifier: session {k}: accepted={s['accepted']} failure={s['failure']}", flush=True)
    return 0 if all(s["accepted"] for s in sessions) else 1


if __name__ == "__main__":
    sys.exit(main())
