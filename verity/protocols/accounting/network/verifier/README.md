# The network attester's verifier side (`verity.protocols.accounting.network.verifier`)

What the auditor runs and trusts of network attestation: the relay and its pacer (`network_attester`, `pacer`), the grid
and the signed records (`grid`, `commitment`, `certificate`), the I/O and ingress ties' Definitions (`io_tie`,
`io_request`), the connection rows and their Python check (`connection_rows`, `connection_rows_check`), the transfer
rows and the output value (`transfer_rows`, `out_commit`), the class circuit's check (`blocks`) and the deployment
manifest (`parties`). Each module's docstring states its part.

`lean/` is the Lake package `network_check`: the executables `network-check`, `network-records`, `network-period` and
`network-rows` (each `id | check PARAMS COMMITMENT EVIDENCE | serve`), which `run`'s `onsite` Lean requires by path.
`PROTOCOL.md` is the scheme's description, and `protocol/` its parts.

Try it:

~~~sh
uv run tools/check/suites.py verity/protocols/accounting/network/verifier
~~~
