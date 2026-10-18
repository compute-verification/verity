# Network simulation

The device graph, and a mock datacenter of Linux network namespaces that runs the real network certifier.

- `graph.py`: the device graph, a Python mirror of the Lean physical model
  (`Security/Definitions/Accounting/Physical.lean`'s `Hardware`, `Security/Definitions/Accounting/Device.lean`'s
  `Device`), its leaf-spine generator, its validation (every capacity a named assumption, every block's boundary link
  certified), and its two exports, the netns plan and Shadow's GML. `python graph.py --blocks 2 --netns` prints a plan.
- `mockdc.py`: the netns plan stood up on one host, the certifier's proxy on each block's uplink, and the audit.
- `tests/`: the graph's tests (`uv run tools/verity/check/suites.py network_sim`). The mock datacenter is a script, not a
  test: it runs in real time.

## The mock datacenter

Every namespace is named `na-*`. Each block (`na-b0`, standing in for a node of 8 GPUs) has one interface, to its
certifier's namespace (`na-b0-cert`), whose other interface goes to a leaf switch. By default there are four blocks, two
leaves joined by a spine (`na-spine0`), and one host outside every block (`na-ext0`, on `na-leaf0`). Only the switches
forward IP. A certifier forwards nothing: a block reaches the fabric only through the connections the proxy makes and
accepts, and an iptables rule in the certifier's namespace drops whatever the block opens. Each fabric link has netem,
2 ms ± 0.5 ms per hop at 400 Gbit/s by default. Nothing in the root namespace changes: each veth is created inside its
namespaces, and deleting the namespaces deletes them.

It needs Linux with iproute2, iptables, setpriv and passwordless sudo, and a Python with numpy. Run it on
vy-nebius-cpu-2 (its system `python3` has numpy), not on an agent VM. Every command takes the same `--blocks`,
`--leaves` and `--spines`, and `run` refuses a lab that is up in another shape.

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
Program doesn't have. The audit is `audit.Audit`'s `check_link` and `check_ingress_link` on each block's raw and
committed records, and each block's wire is compared with the honest block's. `DIR/summary.json` has the verdicts and the
timing, and `DIR/b0/` a block's records, logs and audit.

Run `r20261018-065536-38c4`, on vy-nebius-cpu-2 at a load of about 240:

| Block | Strategy | Audit | Wire against the honest block's |
|---|---|---|---|
| b0 | honest | accepted | |
| b1 | bits | accepted | identical |
| b2 | late | refused: `responses` window 0, `status:coverage-never-established` | differs in window 0 |
| b3 | extra | refused: `responses` window 2, `mismatch` | differs in window 2 |

`bits` is the covert channel the certifier closes: it is accepted because its wire is the honest one, byte for byte. The
latest tick was 4.2 ms late against the proxy's 25 ms slack. Each bucket reached the peer about 5.5 ms after its tick
from the blocks on `na-leaf0`, and about 10 ms after it from those behind the spine.

## Limits

- All the processes share one host's monotonic clock, so one origin times every proxy. Blocks on separate hosts would
  need their clocks synchronized to within the slack.
- `mockdc.py` has its own server, proxy and peer coroutines because `active_replay`'s bind `127.0.0.1`. A host parameter
  there would let it reuse them.
