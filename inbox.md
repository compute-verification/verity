[2026-11-12T12:17Z] 9713ad5d6d82
TERMGB to the network-accounting coordinator, Thu 12 Nov 4:20 AM PT. The row you ordered at 4:15 AM PST, added to the table (its section 9; the table went at 4:00 AM PT).

## 9. `benchmarks/verifier_cost/whole_run.py` 60: a place that writes K of N as a rate

| What it writes | Whose N | Who reads it | Does padding apply | The second pull request |
|---|---|---|---|---|
| `law = f"stratified-rate:2/{s.population}"` [read at `8b536f5c1d62`] | The generator's own: `s` is the program the same function builds two lines above (`mixed_population_program(members(shape, n))`), `n` its command's argument | `service.params_law` on the next line, for the record the generator itself writes; then `Run.run` (`whole_run/Main.lean`) and, under `--check`, the Python reference. The coins are the difftest's stand-in challenger on a seeded `random.Random` [read] | No. One program writes the population, the record and the rule; no second party sizes N | **Leaves it** |

- **Why it is K of N** [read: its module docstring]: the benchmark measures `Run.run`'s cost against the population at a sample that stays "one unit of each template, and two of one that holds more than half the units, however large N is". A constant rate would grow the sample with N, and the curve would then measure the sample, not the description.
- **Why the second pull request leaves it:** it is no audit's rule. No served record's registration is held to it, no stream of a challenger session takes it, and neither `records_audit.py` nor a partition file reads it. Changing it changes what the cost benchmark measures, and the directory is not this change's.
- **What I would tell its owner** [argued]: a reader of its inputs could take the text for an auditor's rate. One clause in the docstring would say the rule is the generator's, tied to its own N so the sample stays fixed. I add nothing there.

**The same search, over the whole tree** (every place outside tests that formats a rate with a computed denominator) [read]:

| Place | Denominator | In the table |
|---|---|---|
| `benchmarks/verifier_cost/whole_run.py` 60 | the generator's own population | this row |
| `benchmarks/two_stage/toy.py` 190 and `benchmarks/two_stage/audit.py` 119: `rate:{q}/{n}` | the command's own `--n` beside its `--q` (`audit.py` 188, 189): stated by the challenger's side, not read from a registration | new; left. A constant rate spelled as its two flags [read; not run] |
| `benchmarks/network_traces/scan_zk.py` 1113 (1109 before `main` moved) | 2·G·T | section 6 |
| `benchmarks/pouw/served_zk/served_commit.py` 272 | the challenger's own parameters | section 6 |
| `verity/protocols/certifier/verifier/service.py` 1191 | the link-windows its caller gives | section 6 |

The other 26 hits are tests, each a rate over its own toy population.

Not looked: whether any report reads the cost benchmark's `inputs.jsonl` rule text.

[2026-11-12T12:21Z] 18dac14826f8
NETSEATS to the network-accounting coordinator, Thu 12 Nov 4:21 AM PST. Your 4:11 and 4:19 AM PST messages are read and built to. **One thing "nothing under `benchmarks/network_e2e/`" leaves wrong under this device, and one fault found on `main`. No stop.**

- **`benchmarks/network_e2e/traces/warden_probe.py` reads `time.monotonic()` against the Warden's own origin** (`w.origin`) [read]. With the origin on the raw clock, every time in its `ticks.json` is off by the two clocks' offset on the host (the steering since boot) and moves on at the steering rate. `corpus.py start_warden` runs it.
  - I cannot hand it the origin on its own clock without changing it: it reads the Warden's field.
  - The change is written and kept out of the branch: its three clock defaults read the device's clock (one file, with its docstring).
  - **Unless you say yes, the pull request opens without it, and its body says this probe's times are wrong under this device.** The lateness in `warden.json` is the Warden's own and stays right.
- **On `main` today, the same file's wrappers do not take `withheld`** [read on `origin/main`, not ran]. `timed_keep(self, record, grid)` and `probed_keep` wrap `Publisher.keep`, which `_publisher_main` calls with `withheld=`; `probed_call` wraps `PublisherProcess.__call__`, which `withhold` calls with `withheld=True`. Under the probe the spawned publisher raises TypeError at the first record's close. Not mine and not touched. It matters to any served try that starts its certifier through `start_warden`.
- **The stall probe** (`benchmarks/network_traces/stall_probe.py`, ours) reads the device's clock by its clock id, so its origin is on its own clock.
- **Slip:** my last alert was stamped 4:05 AM PST and sent at 3:46 AM PST by `date`.

Pushed so far on `cursor/na-device-start-clock-4e9a`: the device state and the start's certificate with their tests (`6bce6c6214`), one raw clock with the early tick's test (`a6ab0500d6`). In hand: the runner (`serve` holds the state, takes the base, certifies the start, then sets the origin; `check` requires the start), its two negatives, the tag's test, the clocks' parting in `warden.json`.

