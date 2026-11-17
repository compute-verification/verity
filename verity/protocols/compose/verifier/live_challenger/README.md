# The live challenger

Every coin the proof system's verifier relies on is flipped by the challenger, and the challenger runs on the
auditor's own device, never on a machine the developer controls. This directory is the service that runs there. It
measures nothing: for each session it serves one prover client its coins, with a record signed by the host's LMS key
and written before the first coin, and a release log signed when the session ends. The coins are the operating
system's, as the Lean verifier reads them.

Around that, the host keeps a signed, append-only ledger with one session a run, recorded before its first coin; its
periods; each lane's declared days, the program it serves on a UTC day, signed before the day's first run
opens; and the model registry, each served model's weights committed by the auditor with its own salts.

It is in service on the auditor's device.

## Try it

~~~sh
uv run tools/check/suites.py verity/protocols/compose/verifier/live_challenger
~~~

The suite runs the challenger as a process over loopback with a test key and the Lean verifier's coins stubbed, and
covers the gate, its periods, declared days, models, load and rehearsals, and the network attester's coin link. It
needs no GPU and takes about two minutes. Each file's docstring says what it serves and its flags.

## What a session reads and writes

A session reads the source tree, the challenger's key directory (`~/.local/state/live-challenger/key`, never under
`out/`) and `/etc/vy/peers`. It writes `record.json`, `key.json`, `releases.jsonl`, `hellos.jsonl` and the signed close
(`challenger.json`) to its `out/`; a verifier reads them from there, never from the prover.
