# The live challenger

The proof service's challenger, as it runs on the auditor's host: it measures nothing. `serve.py` serves one session's coins
to one prover client (`verity.core.protocols.sampled_proofs.remote`), with its record signed by the host's LMS key and written
before the first coin, and its release log signed when the session ends. The coins are the operating system's, as the Lean
verifier of record reads them.

- `auditor.py` is the host's service: the `vy-challenger-gate` verbs, run as the `auditor` user from a pinned tree, open one
  session a window and record each in a signed, append-only ledger before its first coin. It also keeps the periods and the
  model registry (`models.py`: each served model's weights, committed by the auditor with its own salts).
- `coin_link.py` is the network certifier's coin link for a window's session: the one path its coins take to the peer,
  counted and re-clocked on the link's grids.
- `publish.sh` puts the host's pins log, ledger, model registry and every ended session into the evidence store; the control
  pod runs it on a loop.
- `run.sh` is a development session: `serve.py` as a `research run` on another machine, which tunnels its port to the
  prover's machine (`PEER`). It needs `PORT`, `SCHEME`, `FAMILIES`, `SESSION` and `PEER`, so it runs through
  `research run ... -- bash benchmarks/live_challenger/run.sh` (its header has the full command), not bare
  `verity bench live_challenger`.

A session reads the source tree, the challenger's key directory (`~/.local/state/live-challenger/key`, never under `out/`) and
`/etc/vy/peers`. It writes `record.json`, `key.json`, `releases.jsonl`, `hellos.jsonl` and the signed close (`challenger.json`)
to its `out/`; a verifier reads them from there, never from the prover.

Status: in service on the auditor's host. The suite (`tests/`) runs `serve.py` as a process over loopback with a test key and the
Lean verifier's coins stubbed (`tests/stub_serve.py`), and covers the gate, its periods, models, load and rehearsals, and the coin
link.
