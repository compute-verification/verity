# The ZK server

In a recursive session the developer's workers hold the values and prove the statement, and the auditor checks what
they proved. The ZK server stands between the workers and the auditor: it is the developer's only path from one to the
other. It commits every message the workers send, and for each statement it releases one zero-knowledge proof that the
workers' proof was accepted, on a fixed schedule. [Flock's live protocol](../../../flock/prover/live/PROTOCOL.md),
§10, says what it computes.

It is code and not analysis: the session loop is `Flock.ZkServer`, in [Flock's
verifier](../../../flock/verifier/README.md), and this package, `core_zk_server`, is the release side.

## Try it

~~~sh
uv run tools/check/suites.py verity-core-zk-server --quick
~~~

It holds the release side's Lean to its Python and to Rust's frozen fixtures, and runs it around Flock's session loop
against a scripted auditor.

## The deployment

The reference deployment puts the workers, the ZK server and the verifier on machines of their own, linked so that the
ZK server is the only exit. None of it is in place today: the ZK server runs on the workers' machines.
