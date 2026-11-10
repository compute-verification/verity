# PoUB's verifier as a process (`verity/protocols/poub/verifier/remote_session`)

The verifier of PoUB's live sessions as a process a machine runs. `verifier_process.py` is the verifier's `main`
(`verity/protocols/poub/verifier/session/verifier.py`) on its command line, handed the two things a verifier imports
none of: the LMS check of a session's signatures, and the challenger's process that `--challenger serve` starts.

## Interface

What code outside this directory may use.

| Entry point | What it is |
|---|---|
| `python verity/protocols/poub/verifier/remote_session/verifier_process.py --port PORT --out OUT [OPTIONS]` | The verifier's process: it serves one prover over TCP and writes `verifier.json` under `--out`. Its other forms are `--registry REGISTRY.json --close NODE_HEX`, which closes a node's registration, and `--memory-of VERIFIER_JSON [--host-bytes N]`, which prints the memory line of each accepted session in a kept `verifier.json`. The options are `session/verifier.py`'s, and its docstring lists them. The file puts the repository's root on `sys.path` itself, so it runs under `python3 -I`. |
| `verifier_process.lms_verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool` and `verifier_process.SERVE` | The signature check, LMS over SHA-512 at the key's parameters, and the path of the challenger's process. A caller that drives `session.verifier.main` itself hands it these. |

Run by its path today: the session harness's scripts (`benchmarks/remote_session/run.sh` and `node.sh`) and
live_verifier's host (`verity/protocols/poub/device/live_verifier/host.py`), which starts it for each run. Imported as a
module: the harness's line of record (`benchmarks/remote_session/auditor_line.py`) and two tests.
