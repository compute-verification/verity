# certifier_timing

How many bits an interval a server can move through **when** the network certifier releases its bytes. The certifier
fixes which interval a cell leaves in, so what is left to a server is the release's time inside the interval: the
jitter term, which the accounting charges as log₂ D_j a release interval and which is 0 only at D_j = 1. This
benchmark puts one number on it, checks that the number sees a channel, and states the number a budget asks for.

## The number

A run has three parties, each a process (`parties.py`):

- **The server** is the adversary. It picks a uniform bit for each interval and acts only through when and how it
  writes its own bytes for this interval and the next. The bytes are the same whatever the bit.
- **The certifier** is the device under test: the network certifier of this checkout or of another (`--tree`), as
  `verity.protocols.certifier.verifier` has it, writing each interval on its loop (`--paced-ms 0`) or through its
  pacer.
- **The observer** is the client side. It timestamps the first byte of each interval twice: on its own clock once
  its read returns, waiting in the kernel for it (`plain`) or polling without waiting (`spin`), and with the stamp
  the kernel gave the bytes as it took them off the network. It never has the server's bits.

The attacks (`parties.Plan`), over interval t's release time R_t:

| Attack | Bit 1 | Bit 0 |
| --- | --- | --- |
| `write` | the next interval's bytes in one write, at most `--lead-ms` before R_t | the same bytes midway between R_t and the next boundary |
| `burst` | the next interval's bytes in `--burst` writes, from `--lead-ms` before R_t to an eighth of that after it | as `write` |
| `delay` | the interval's own bytes half an interval late | the bytes at the boundary |

The network certifier fails a delayed interval closed, so `delay` runs only against the guard's stand-in.

`channel.measure` fits a decoder on the even intervals, one threshold on the arrival's offset after R_t or on its
`step` (the offset less its neighbours' trimmed mean, which drops a drift), and scores it on the odd ones. A decoder
wrong at rate e on uniform bits shows at least 1 − H(e) bits an interval: `bits`, with `lower` and `upper` from e's
95% Wilson interval. `upper` is 1 − H at the interval's end farther from one half, so a decoder that came out wrong
more often than not never makes it small.

**Two stamps, two columns.** Every case is scored on each of the observer's stamps, with a decoder of its own:

| Stamp | What it is | What moves it besides the certifier |
| --- | --- | --- |
| `process` | the observer's monotonic clock once the read that holds the interval's first byte returned | the network, the observer's kernel, and the observer's own scheduling and wake-up |
| `kernel` | the kernel's receive stamp on that read (`SO_TIMESTAMPNS`, read from `recvmsg`'s ancillary data) | the network and the observer's kernel up to where it takes a segment in |

On a TCP stream the kernel returns, with each read, the stamp of the last segment the read took bytes from: its wall
clock as its network stack took the segment in, before any socket or process saw it. Over loopback or a veth pair
that is as the sender's write reached the device. The observer reads one interval's bytes at most a read, so a read's
stamp is of that interval's bytes, and it moves the stamp to its monotonic clock by the two clocks' difference as it
started. Two things are told rather than hidden: a scored interval whose read carried no stamp, or a wall clock that
was set during the case, makes the case no measurement; and segments the kernel merged while they waited to be read
carry the last one's stamp, so an interval read that late has a later interval's, and the row counts them
(`merged`). A host whose kernel gives no stamp leaves the `kernel` column empty and says why (a case's `unstamped`).
A client can stamp as the kernel does, so the number is over both columns.

`run` prints one JSON line and writes it to `--out`/result.json:

| Key | What it is |
| --- | --- |
| `value`, `unit` | the most bits an interval any attack showed on either stamp, over that attack's pooled runs |
| `lower`, `upper`, `n` | its 95% bounds and scored intervals |
| `best` | the attack, observer and stamp `value` is of |
| `ceiling` | the largest upper bound of any attack on either stamp: what the run shows every attack built here under |
| `commit`, `run` | the commit measured from and the run's id |
| `certifier`, `host` | the certifier's commit and settings; the host, its cores and its clock |
| `guard`, `target` | the guard's two readings; the budget's bits an interval, and whether `ceiling` is under each end |
| `rows` | each attack with each observer, pooled, most bits first: the `process` stamp's reading and, under `kernel`, the kernel stamp's; `halted`, how many times the certifier's host's cores entered an idle state that stops them while the row's cases ran (null where the host counts none); then the guard's cases and any control, which are no part of the number (`role`) |
| `cases` | each run: its settings, hosts and problems, both stamps' readings (`client`, `kernel`), and the certifier's own release times scored the same way (`sender`) |

## Run it

On a team machine, never an agent's VM: the number is of the host it ran on.

~~~sh
verity bench certifier_timing run                     # this checkout's certifier, paced 50 ms into a 100 ms interval
verity bench certifier_timing run --paced-ms 0        # the same certifier writing each interval on its loop
verity bench certifier_timing run --tree /path/to/checkout --tree-commit SHA   # another checkout's certifier
verity bench certifier_timing guard                   # the guard alone
verity bench certifier_timing target --interval-ms 100
verity bench certifier_timing score DIR               # recorded cases again, as one result
verity bench certifier_timing table RESULT.json       # results as a table's rows
~~~

The certifier imports only the standard library, `verity` and numpy, so a `python3` of 3.12 or later that has numpy
runs `benchmarks/certifier_timing/run.py` with the checkout's root on `PYTHONPATH`. Under `research run` the result and
every case's timestamps go to `$OUT`. `run` exits 1 when the guard fails and 2 when a case is no measurement: the
certifier failed a connection, a tick ran past its slack, or the parties disagree on which interval is which.

## The guard

Every `run` first measures a channel it planted and one it didn't, against `relay`, a stand-in that paces nothing
and passes bytes on as they come. A server that delays an interval by half an interval on bit 1 must show at least
0.9 bits an interval, and the same server ignoring its bits at most 0.05 (`PLANTED_MIN`, `IGNORED_MAX`), on each of
the observer's two stamps. A run whose guard fails reports no number.

## The target

A stated parameter, not a measurement. The certifier's timing channel may carry 300 GB (2.4e12 bits) a year over all
compute, spread evenly over its nodes, each behind one certifier. The nodes are the census's world
(`benchmarks/census/adversary.json`, `world-inference-2027`: 9.36e30 FLOP a year) on eight-GPU nodes
(`verity/protocols/inference_only/rho/data/hardware.toml`, `location.h200-hgx`) at two utilizations: 0.4, the
fleet's (`verity/protocols/inference_only/rho/data/reference.toml`, `fleet.world-h200`: 4.687e7 nodes), which is a
bit a node every 616 seconds, and 1, 1.875e7 nodes, a bit every 246 seconds. `channel.SECONDS_A_BIT` holds the two
rounded, 600 and 240: so at an interval of τ seconds the certifier's share is at most τ/240 to τ/600 bits an
interval. `target` prints both ends and the scored intervals it takes to show a number that small
(`channel.samples_to_show`): `least`, at which a run whose decoder came out wrong exactly half the time has its
upper bound at the target and every other run shows more, and `nine_in_ten`, at which nine runs in ten of a decoder
truly at chance show it. A run scores half its intervals. The time a run takes to show the target is about the same
at every τ, since the target grows with τ as the intervals a second fall.

## What it establishes

- **A lower bound, of the attacks built here.** `value` is what these servers move to these observers under a
  one-threshold decoder. Nothing here bounds the channel from above: a reading of 0 says these attacks carry nothing
  the run could see, at its `ceiling`.
- **On one host.** `run`'s three parties share a host, its cores and its monotonic clock, over loopback. A certifier
  is the auditor's hardware on a machine the developer doesn't control, so what the server does to the certifier's
  host through shared cores is more than a deployed server can do, and a loopback path is quieter than a network.
- **`serve` and `observe`** put the observer on another host (below). The server still shares the certifier's host
  and clock.
- **One connection.** A case's observer holds one connection. A receiver holding several of a link's sees each
  release once on each, and can average away what differs between them; no case here does.

## The candidates

Each is a flag beside the certifier, which this benchmark never changes. The number is of the host it ran on and of
what else that host was doing, so a candidate is compared with the certifier as is in one `research run` on one host,
each `run` in turn and then again, and `score` pools each side's repeats. Two runs on two hosts compare the hosts.

| Flag | What it does |
| --- | --- |
| `--pin` | the certifier's loop, its pacer, the server and the observer each on a core of the run's own |
| `--gate-ms BEFORE,AFTER` | the certifier reads no server bytes from BEFORE ms ahead of each release to AFTER ms past it |
| `--waker-ms MS` | a writer wakes a reader over loopback every MS ms, so the host's network path is in use at every release |
| `--awake` | a process spins at idle priority on each of the run's CPUs, so none halts: a guest's cores under `idle=poll`, without root |
| `serve --launch DEV,AFTER,OPEN[,MODE]` | the kernel, or the NIC, releases: a `taprio` schedule on the device the certifier's bytes leave by lets them pass for OPEN ms from AFTER ms past each release time |

The certifier's egress is TCP, which takes no launch time from its socket (`SO_TXTIME`), so the release is put on
the device: under `--launch` the certifier's connections send at a priority of their own
(`parties.LAUNCH_PRIORITY`), the schedule holds that priority's packets to the gate and passes every other of the
host's always, and the server aims at the gate. MODE says who holds the gate:

| MODE | Who releases | It takes |
| --- | --- | --- |
| `gate` (default) | the kernel's timer, on the host's monotonic clock | `sch_taprio` |
| `gate-nic` | the NIC, the whole schedule offloaded | a NIC that runs the schedule, its clock kept to TAI |
| `txtime` | the kernel's ETF queue, at the stamp the schedule gave each packet | `sch_taprio`, `sch_etf` |
| `txtime-nic` | the NIC, launching each packet at its stamp | a NIC with launch time, its clock kept to TAI |

Every mode takes the right to administer the host's network (root's, for `tc qdisc`), `tc` and a device with two
transmit queues. A case whose gated queue carried fewer packets than the case has intervals, or dropped any, is no
measurement (`launch.json`, the device's queue counts).

## Two hosts

~~~sh
verity bench certifier_timing serve --listen 0.0.0.0 --port 47812 --attack write --out OUT/write-spin   # the certifier's host
verity bench certifier_timing observe HOST:47812 --observer spin --out SEEN/write-spin                   # the other host
verity bench certifier_timing score OUT/write-spin --observer SEEN
~~~

`serve` takes one connection, the case's observer, and `observe` tries the certifier until it answers. Nothing joins
the two clocks. Which interval a chunk is comes from its bytes (interval t's are all the byte t mod 256) and from the
certifier's record of the connection, and `step` reads through the clocks' drift. Both of the observer's stamps are
its own host's, each counted from its first arrival. `serve --certifier relay --attack
delay`, with and without `--ignore`, is the guard across the two hosts, and `serve --ignore` an attack's control.

`pair.sh serve|observe ADDR PORT START OUT [serve's flags]` runs one side of every case: the guard's two, each
attack with each observer, and a control. Both sides start each case at a time of day they work out from START
alone, so a case that fails takes no other with it; `score A/out/* --observer B/seen` is then the result.

`netns.sh OUT [serve's flags]` runs both sides on one host without root: a user namespace holds two network
namespaces joined by a veth pair, the certifier's device `cert0`, which `--launch cert0,AFTER,OPEN[,MODE]` can
gate. Both namespaces are one kernel's, on one clock: a gate there is still the host's software releasing, and shows
only that the schedule holds the certifier's bytes and what a software gate leaves.

## Why a directory of its own

`network_traces/tick_load.py --secret-bit` asks the same question of a whole rehearsal: a served endpoint, the
certifier's records and their check, at a served workload's traffic. Here the certifier runs alone and every
interval carries a bit, so the number moves only with the certifier's release path, comes with its guard, and is
measured the same way on any checkout's certifier.
