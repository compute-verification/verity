# The network attester's prover side (`verity.protocols.accounting.network.prover`)

The Boolean Definitions of network attestation that its verifier doesn't read: the scan's ingress and egress
(`ingress_scan`, `egress`), the ingress-ids tie (`io_ids`), the lengths rows' block (`lengths_row`) and the response
tie's (`response_tie`), each with a circuit-check sidecar (`<module>.circuit-check.py`, its pins beside it). Nothing on
the tip proves their blocks: the network attester's piece writes NOT RUN.

It also holds the engine's admission and the request framing the scheme attaches to a serve (`admission`, `id_relay`,
`ids_protocol`, `serve_head`), which are the adapter's.

Try it:

~~~sh
uv run tools/check/suites.py verity/protocols/accounting/network/prover
~~~
