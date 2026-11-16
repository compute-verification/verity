# certifier_timing

This benchmark measures how many bits an interval a server can move through **when** the network attester releases its
bytes. The network attester fixes which interval a cell leaves in, so what is left to a server is the release's time
inside the interval: the jitter term, which the accounting charges as log₂ D_j a release interval and which is 0 only
at D_j = 1. Three processes on one host play the parts: a server that times its writes by a secret bit an interval, the
network attester under test, and an observer on the client side that timestamps each interval's first byte and decodes
it. The number is the most bits an interval any of these attacks showed, with its 95% bounds.

Every `run` first runs a guard, a channel it planted and one it didn't, and a run whose guard fails reports no number.
The target the number is held to, τ/240 to τ/600 bits an interval of τ seconds, is a stated parameter, not a
measurement.

## Try it

~~~sh
uv run verity bench certifier_timing target --interval-ms 100
~~~

It prints, in a second and on any machine, the target's two ends at a 100 ms interval and how many scored intervals a
run takes to show a number that small. The measurement itself is `run`, below, on a team machine.

## What it establishes

- **A lower bound, of the attacks built here.** `value` is what these servers move to these observers under a
  one-threshold decoder. Nothing here bounds the channel from above: a reading of 0 says these attacks carry nothing
  the run could see, at its `ceiling`.
- **On one host.** `run`'s three parties share a host, its cores and its monotonic clock, over loopback. A network attester
  is the auditor's hardware on a machine the developer doesn't control, so what the server does to the network attester's
  host through shared cores is more than a deployed server can do, and a loopback path is quieter than a network.
- **`serve` and `observe`** put the observer on another host (below). The server still shares the network attester's host
  and clock.
- **One connection.** A case's observer holds one connection. A receiver holding several of a link's sees each
  release once on each, and can average away what differs between them; no case here does.

## Run it

On a team machine, never an agent's VM: the number is of the host it ran on.

~~~sh
verity bench certifier_timing run                     # this checkout's network attester, paced 50 ms into a 100 ms interval
verity bench certifier_timing run --paced-ms 0        # the same network attester writing each interval on its loop
verity bench certifier_timing run --tree /path/to/checkout --tree-commit SHA   # another checkout's network attester
verity bench certifier_timing guard                   # the guard alone
verity bench certifier_timing target --interval-ms 100
verity bench certifier_timing score DIR               # recorded cases again, as one result
verity bench certifier_timing table RESULT.json       # results as a table's rows
~~~

The network attester imports only the standard library, `verity` and numpy, so a `python3` of 3.12 or later that has numpy
runs `benchmarks/certifier_timing/run.py` with the checkout's root on `PYTHONPATH`. Under `research run` the result and
every case's timestamps go to `$OUT`. `run` exits 1 when the guard fails and 2 when a case is no measurement: the
network attester failed a connection, a tick ran past its slack, or the parties disagree on which interval is which.

For a case with the observer on another host:

~~~sh
verity bench certifier_timing serve --listen 0.0.0.0 --port 47812 --attack write --out OUT/write-spin   # the network attester's host
verity bench certifier_timing observe HOST:47812 --observer spin --out SEEN/write-spin                   # the other host
verity bench certifier_timing score OUT/write-spin --observer SEEN
~~~
