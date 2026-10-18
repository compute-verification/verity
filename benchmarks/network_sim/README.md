# Network simulation

The device graph, and a mock datacenter of Linux network namespaces that runs the real network certifier. Both read the
accounting layer's declarations, `datacenter.Datacenter` (`benchmarks/datacenter`).

- `graph.py`: declarations and exports. `rtx_node` is the measured node (8 GPUs on two sockets, each GPU on a PCIe root
  port of its own, one NIC on socket 1) as one block, with memory accounting's capacities as named assumptions; a
  capacity not measured yet is `math.inf`, the bound that claims nothing. `leaf_spine` puts such nodes behind a
  leaf-spine fabric, each through its block's certifier, with one host outside every block. The exports read any
  `Datacenter` that validates: the netns plan `mockdc.py` stands up, and Shadow's GML. `python graph.py --blocks 2
  --netns` prints a plan.
- `records.py`: the certifier's published records as `datacenter.observe.Record`s (link, window, counts, digest,
  certifier), the records an honest run declares, and their audit, with `datacenter.claims`' claims and reasons.
- `mockdc.py`: a declaration's netns plan stood up on one host, the certifier's proxy on each block's uplink, and both
  audits.
- `tests/`: the graph's tests, and the records adapter's on an injected clock
  (`uv run tools/verity/check/suites.py network_sim`). The mock datacenter is a script, not a test: it runs in real time.

## The mock datacenter

Every namespace is named `na-*`. Each block (`na-b0`, standing in for a node of 8 GPUs) has one interface, to its
certifier's namespace (`na-b0-cert`), whose other interface goes to a leaf switch. By default there are four blocks, two
leaves joined by a spine (`na-spine0`), and one host outside every block (`na-ext0`, on `na-leaf0`). Only the switches
forward IP. A certifier forwards nothing: a block reaches the fabric only through the connections the proxy makes and
accepts, and an iptables rule in the certifier's namespace drops whatever the block opens. Each fabric link has netem,
2 ms ± 0.5 ms per hop at 400 Gbit/s by default. Nothing in the root namespace changes: each veth is created inside its
namespaces, and deleting the namespaces deletes them.

That layout is `graph.leaf_spine(4, leaves=2, spines=1)`'s. Any other declaration works whose blocks each have one
boundary link, to a certifier, and which has a device of kind `outside`: `--declaration FILE.py:NAME` names a
`Datacenter`, or a function returning one.

It needs Linux with iproute2, iptables, setpriv and passwordless sudo, and a Python with numpy. Run it on
vy-nebius-cpu-2 (its system `python3` has numpy), not on an agent VM. Every command takes the same `--declaration`, or
`--blocks`, `--leaves` and `--spines`, and `run` refuses a lab that is up in another shape.

~~~sh
python3 benchmarks/network_sim/mockdc.py up       # create the namespaces, then `status`
python3 benchmarks/network_sim/mockdc.py status   # list them, and check what reaches what
python3 benchmarks/network_sim/mockdc.py run --out DIR [--block b1=late ...]
python3 benchmarks/network_sim/mockdc.py down     # stop their processes, and delete every na-* namespace and veth
~~~

As a recorded run, which leaves the lab up:

~~~sh
research run --on vy-nebius-cpu-2 --project verity --source . --cwd source -- \
  bash -c 'M=benchmarks/network_sim/mockdc.py; python3 -B $M down && python3 -B $M up && python3 -B $M run --out "$OUT"'
~~~

## What a run shows

`run` plays `benchmarks/network_traces/tests/test_active_replay.py`'s synthetic trace, three windows of 60 buckets of
100 ms, on every block at once. In each block runs the server (`active_replay.Server`), in its certifier's namespace the
proxy (`verity.core.protocols.certifier.active.Proxy`, unmodified), and on the outside host the peer, whose connection
crosses the fabric. Each block's server plays one strategy: `honest`; `bits`, whose write times carry a random bit per
session; `late`, which holds a session in window 0 past every sync; or `extra`, which writes one event at a position the
Program doesn't have. Each block is audited twice. The certifier's audit is `audit.Audit`'s `check_link` and
`check_ingress_link` on its raw and committed records. The datacenter record check is `records.audit` on its records as
`datacenter.observe.Record`s against the ones an honest run of the Program publishes: each link-window needs one, with
the declared counts and digest, and each carried window leaves log2 of the sync set's size bits of timing advice. Each
block's wire is also compared with the honest block's. `DIR/summary.json` has the verdicts and the timing, and `DIR/b0/`
a block's records (`records.json` as `Record`s), logs and audit.

Run `r20261018-071731-392a`, on vy-nebius-cpu-2 at a load of about 50:

| Block | Strategy | Certifier audit | Datacenter record check | Wire against the honest block's |
|---|---|---|---|---|
| b0 | honest | accepted | 6 windows carried, 24 bits of advice | |
| b1 | bits | accepted | 6 windows carried, 24 bits of advice | identical |
| b2 | late | refused: `responses` window 0, `status:coverage-never-established` | refused: `b2.nic0`–`b2.cert` window 0, occupancy | differs in window 0 |
| b3 | extra | refused: `responses` window 2, `mismatch` | refused: `b3.nic0`–`b3.cert` window 2, occupancy | differs in window 2 |

`bits` is the covert channel the certifier closes: it is accepted because its wire is the honest one, byte for byte. The
latest tick was 4.3 ms late against the proxy's 25 ms slack. Each bucket reached the peer about 5.5 ms after its tick
from the blocks on `na-leaf0`, and about 10 ms after it from those behind the spine. The first run, before the
declarations (`r20261018-065536-38c4`, at a load of about 240), gave the same certifier verdicts.

## Limits

- All the processes share one host's monotonic clock, so one origin times every proxy. Blocks on separate hosts would
  need their clocks synchronized to within the slack.
- `mockdc.py` has its own server, proxy and peer coroutines because `active_replay`'s bind `127.0.0.1`. A host parameter
  there would let it reuse them.
- `records.audit` repeats `datacenter.claims.audit`'s record loop rather than calling it, because `claims.audit` derives
  the expected digests from a placed circuit's crossings, and the lab's traffic is a replay Program.
