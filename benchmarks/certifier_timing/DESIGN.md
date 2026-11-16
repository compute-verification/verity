# certifier_timing: design

How the number is taken, the target it is held to, how a candidate is compared, the two-host scripts, and why the
benchmark has a directory of its own. [`README.md`](README.md) says what it measures and how to run it.

## The number

A run has three parties, each a process (`parties.py`):

- **The server** is the adversary. It picks a uniform bit for each interval and acts only through when and how it
  writes its own bytes for this interval and the next. The bytes are the same whatever the bit.
- **The network attester** is the device under test, of this checkout or of another (`--tree`), as
  `verity.protocols.resource_accounting.network.verifier` has it, writing each interval on its loop (`--paced-ms 0`) or through its
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

The network attester fails a delayed interval closed, so `delay` runs only against the guard's stand-in.

`channel.measure` fits a decoder on the even intervals, one threshold on the arrival's offset after R_t or on its
`step` (the offset less its neighbours' trimmed mean, which drops a drift), and scores it on the odd ones. A decoder
wrong at rate e on uniform bits shows at least 1 − H(e) bits an interval: `bits`, with `lower` and `upper` from e's
95% Wilson interval. `upper` is 1 − H at the interval's end farther from one half, so a decoder that came out wrong
more often than not never makes it small.

**Two stamps, two columns.** Every case is scored on each of the observer's stamps, with a decoder of its own:

| Stamp | What it is | What moves it besides the network attester |
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
| `certifier`, `host` | the network attester's commit and settings; the host, its cores and its clock |
| `guard`, `target` | the guard's two readings; the budget's bits an interval, and whether `ceiling` is under each end |
| `rows` | each attack with each observer, pooled, most bits first: the `process` stamp's reading and, under `kernel`, the kernel stamp's; `halted`, how many times the network attester's host's cores entered an idle state that stops them while the row's cases ran (null where the host counts none); then the guard's cases and any control, which are no part of the number (`role`) |
| `cases` | each run: its settings, hosts and problems, both stamps' readings (`client`, `kernel`), and the network attester's own release times scored the same way (`sender`) |

## The target

A stated parameter, not a measurement. The network attester's timing channel may carry 300 GB (2.4e12 bits) a year over all
compute, spread evenly over its nodes, each behind one network attester. The nodes are the census's world
(`benchmarks/census/adversary.json`, `world-inference-2027`: 9.36e30 FLOP a year) on eight-GPU nodes
(`benchmarks/slowdown/data/hardware.toml`, `location.h200-hgx`) at two utilizations: 0.4, the
fleet's (`benchmarks/slowdown/data/reference.toml`, `fleet.world-h200`: 4.687e7 nodes), which is a
bit a node every 616 seconds, and 1, 1.875e7 nodes, a bit every 246 seconds. `channel.SECONDS_A_BIT` holds the two
rounded, 600 and 240: so at an interval of τ seconds the network attester's share is at most τ/240 to τ/600 bits an
interval. `target` prints both ends and the scored intervals it takes to show a number that small
(`channel.samples_to_show`): `least`, at which a run whose decoder came out wrong exactly half the time has its
upper bound at the target and every other run shows more, and `nine_in_ten`, at which nine runs in ten of a decoder
truly at chance show it. A run scores half its intervals. The time a run takes to show the target is about the same
at every τ, since the target grows with τ as the intervals a second fall.

## The candidates

Each is a flag beside the network attester, which this benchmark never changes. The number is of the host it ran on and of
what else that host was doing, so a candidate is compared with the network attester as is in one `research run` on one host,
each `run` in turn and then again, and `score` pools each side's repeats. Two runs on two hosts compare the hosts.

## Two hosts

`pair.sh serve|observe ADDR PORT START OUT [serve's flags]` runs one side of every case: the guard's two, each
attack with each observer, and a control. Both sides start each case at a time of day they work out from START
alone, so a case that fails takes no other with it; `score A/out/* --observer B/seen` is then the result.

`netns.sh OUT [serve's flags]` runs both sides on one host without root: a user namespace holds two network
namespaces joined by a veth pair, the network attester's device `cert0`, which `--launch cert0,AFTER,OPEN[,MODE]` can
gate. Both namespaces are one kernel's, on one clock: a gate there is still the host's software releasing, and shows
only that the schedule holds the network attester's bytes and what a software gate leaves.

## Why a directory of its own

Here the network attester runs alone and every interval carries a bit, so the number moves only with the network attester's release
path, comes with its guard, and is measured the same way on any checkout's network attester.
