# The network attester's device (`verity.protocols.accounting.network.attester`)

The host programs that run the network attester in front of a server: `network_attester_serve.py` (the relay in front
of any TCP server, and the offline check of what it published), `load_relay.py` (the weights' load and its commit),
`connection_rows_run.py` (a run's lengths and content rows), with `runner_cli.py` and `runner_parts.py`;
`stall_probe.py` (what blocks a tick), `attester_key.py` (a device's LMS key on disk) and `coin_link.py` (the link for an
execution phase's challenges). The relay itself and its pacer are in `../verifier`.

`program/lean/` is the Lake package `certifier_program`, the network attester's program in executable Lean, which
`../verifier/lean` and `security/Proofs` require by path; its difftests run in the Lean check.

Try it:

~~~sh
uv run python verity/protocols/accounting/network/attester/network_attester_serve.py rehearse --fake --out /tmp/net --requests 4
~~~
