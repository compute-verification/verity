# network_traces

Network attestation (`verity/protocols/resource_accounting/network/`) on real served traffic: per-token timing traces,
the network attester in front of a served endpoint as a relay, the zero-knowledge proofs of their proof blocks, and the
network chain end to end. Each driver's docstring says what it runs, what it publishes and what its verifier receives:
`chain_record.py` (the chain, from the gate's fetched runs to one report), `leak_guard.py` and the prover side's
`released_keys.py` (what a record may release), `served_session.py` (the served session's runbook),
`certifier_piece.py` (network attestation's piece of the one-stage composite), `wipe_egress.py` (a memory wipe's egress
count) and `leak_bounds.py` (K, the one calculator of the leak bound).

## Running it

- `verity bench network_traces` records one chunk of per-token vLLM timing traces on a leased GPU (`record.py`).
- K per monitoring period on a fetched network attester run and its scan's summary:
  `uv run benchmarks/network_traces/chain_record.py leak --warden-run RUN --scan SUMMARY --out FILE`. `verity bench
  certifier_term` prints the network attester's terms at the load of a node, a datacenter and the world.
- Results are read from the evidence store, never from this README.
- Tests: `uv run tools/check/suites.py verity-network-traces`.

## A cheating server

`deviations.py` puts a deviating process between the relay and the server's id front, plants a random string, and tries
each class of way to send it out (`--class`), each through a network attester of its own; its docstring has the
classes and the exit codes. The dry run, against the stand-in id endpoint, then against a served front whose
`HOST:PORT` is the value of an environment variable the command names, then the table once the server's record of the
tries' sampled ids is there:

    uv run benchmarks/network_traces/deviations.py try --stand-in --out DIR
    FRONT=HOST:PORT uv run benchmarks/network_traces/deviations.py try --front-env FRONT --out DIR \
        [--cell-bytes B] [--tail S] [-- --window T --delay D ...]
    uv run benchmarks/network_traces/deviations.py table --out DIR --ids REQUESTS.jsonl

## A second count from a packet capture

`capture_flows.py` reads a capture taken beside a run through the relay, at a point that is neither the server's
process nor the network attester's, and counts the bytes between the two a second time. Its docstring says which bytes
are compared, why they must agree and what it refuses.

**Taking the capture**, where every packet between the server and the network attester passes:

    tcpdump -i IFACE -U -s 130 -w run.pcap 'ip host SERVER'

- The filter keeps only the flows with the server's address on one side: add one `or` clause for each further address
  (`ip6 host` for an IPv6 one), and `vlan and (…)` on an interface whose frames carry a VLAN tag (`vlan and vlan and
  (…)` for two).
- The capture length is a trade. `-s 130` keeps every header the reader reads, and where a packet's headers are shorter
  (a common packet's are 54 to 66 bytes) the first bytes of its content: the start of each segment, a request's start
  and a token id among them. A length below a packet's headers is refused, so no one length both keeps every header
  and keeps no content; choose the least the capture point's headers need.
- Start it before the run's first connection and stop it once the file has stopped growing after the last has closed:
  a flow without its SYN or its FIN is only a floor.
- The file is pcap; `tcpdump -r IN -w OUT.pcap` rewrites a pcapng one.

**On the server's loopback**, where the network attester reaches the server through a forward that ends on its machine:

    tcpdump -i lo -U -s 54 -w run.pcap 'ip host 127.0.0.1 and tcp port PORT'

`-s 54` keeps no byte of content, since a TCP packet's headers on a Linux loopback are never under 54 bytes over IPv4
(74 over IPv6: `-s 74` and `'ip6 host ::1 and tcp port PORT'`). Read it with `--server 127.0.0.1:PORT --certifier
127.0.0.1`.

**Reading it:**

    python benchmarks/network_traces/capture_flows.py --capture run.pcap --server SERVER[:PORT] \
        --certifier CERTIFIER --run RUN --out capture_flows.json

`RUN` is the relay's run directory (`network_attester_serve.py serve`); without it the capture is summed and listed
only. It exits 0 when it read the capture and 2 when it refused it.
