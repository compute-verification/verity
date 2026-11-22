# The ZK server

The ZK server is the developer's only path from its workers to the auditor, trusted by the developer and by no one else:
it commits every message the workers send, masks, checks or fixes everything else, and releases Π_out's
zero-knowledge proofs that V_in accepts, on a fixed schedule. Flock's live protocol (`verity/protocols/proof/flock/prover/live/PROTOCOL.md`, §10) says what
it computes. Its Lean is two packages: `Flock.ZkServer` in Flock's Lean verifier (`flock-zk-server`: the session loop,
each rep's frame, the tap and the coin check), and this one, `core_zk_server`, the release side (§10.5: the record
`firewall/v0`, one Π_out session's decision over `Flock.ZkServer.Contract`, and `core-zk-server session`).

The reference deployment: the workers on the developer's GPU servers; one data diode from them to the ZK server, on CPU
servers of its own from another vendor; the verifier's challenger sending each live coin to both directly; the ZK server
the only exit, running only the compiled Lean, mixing the operating system's entropy with a hardware generator's, never
sending to the workers, and releasing fixed-size messages on a fixed schedule; optionally a release gate and a framing
filter. None of its elements is in place yet: the ZK server runs on the workers' machines, relays the coins to them and
serves their reads from its tape. The statements the elements realize: `Flock.zk_server_blind`,
`Flock.Guarantees.RecursiveZKLoop`, `Flock.Guarantees.SessionLoopZKAnyDeployed`,
`Flock.ZkServer.Contract.ZkServerComputes` and `Flock.zk_server_lean_loop`.

~~~sh
uv run pytest tests/verity/protocols/proof/flock/prover/test_honest_reqs.py tests/verity/protocols/proof/flock/prover/test_honest_session.py
~~~
