# certifier_timing

How many bits an interval a server can move through **when** the network attester releases its bytes. The network
attester fixes which interval a cell leaves in, so what is left to a server is the release's time inside the interval:
the jitter term, which the accounting charges as log₂ D_j a release interval and which is 0 only at D_j = 1. This
benchmark puts one number on it, checks that the number sees a channel (the guard), and states the number a budget
asks for (`target`). `run.py`'s docstring has every verb, the result's keys and the guard; `parties.py`'s the three
parties and the attacks; `channel.py`'s the decoder, the bound it scores and the target.

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

With the observer on another host (the server still shares the network attester's host and clock):

~~~sh
verity bench certifier_timing serve --listen 0.0.0.0 --port 47812 --attack write --out OUT/write-spin   # the network attester's host
verity bench certifier_timing observe HOST:47812 --observer spin --out SEEN/write-spin                   # the other host
verity bench certifier_timing score OUT/write-spin --observer SEEN
~~~

Under `research run` the result and every case's timestamps go to `$OUT`. `run` exits 1 when the guard fails and 2 when
a case is no measurement. A candidate change to the release path (`--pin`, `--gate-ms`, `--waker-ms`, `--awake`,
`serve --launch`) is compared with the network attester as is in one `research run` on one host, each `run` in turn
and then again, and `score` pools each side's repeats: two runs on two hosts compare the hosts.

## What it establishes

- **A lower bound, of the attacks built here.** `value` is what these servers move to these observers under a
  one-threshold decoder. Nothing here bounds the channel from above: a reading of 0 says these attacks carry nothing
  the run could see, at its `ceiling`.
- **On one host**, unless the observer is on another. A deployed network attester is the auditor's hardware on a
  machine the developer doesn't control, so what the server does to its host through shared cores here is more than a
  deployed server can do, and a loopback path is quieter than a network.
- **One connection.** A receiver holding several of a link's connections sees each release once on each, and can
  average away what differs between them; no case here does.
