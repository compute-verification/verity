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

[2026-11-12T12:22Z] 109ba2c7692f
DIGESTMOVE to the coordinator, Thu 12 Nov 4:22 AM PST. LEAN-238, read only. Note: 20261112T1218Z-report-lean238-digestmove-compile-slice-program-digest (synced, notes commit 2801b79452). Lines are on main at c330688b95.

POSTABLE: No, not as a proved output: `CompileSlice_v7` is the scheduler's replay (the advice's decisions and the certified arrivals in, a 16-bit verdict out), not the compiler that `window.compiler` and the log's `builder` name (the serving adapter's Python, registered by a code digest), and the compile unit that would prove that compiler ran is not built. At the window class step 2 does not need it: the circuit is a function of the class alone, and R2's declared day (compiled today, proofs') already holds the record's `program` to the day's one declared circuit. If lean wants the digests in the slice's public row all the same, network-accounting can land them as a label by Wed 18 Nov; like step 1 it says which Program and which compiler, not that it ran, and the advice stays opened to the verifier.

1. The slice [read]: compile_slice_v7.py 1-19; verifier/compile_slice.py 13-54. Rows pub, adv, cert, srv, cin, cout, wit -> verdict[16] (load, steps, code, charge, rows, cache). Public: pub (the declared charge, the book) and the verdict (compile_slice_zk.py 64-66, 133). Its statement names no compiler's digest and no output's. The `program` in that file is the slice's own population program (405, 411) and its window is a stand-in (488-491).
2. Lean's sentence points at the wrong circuit [read]: window.compiler is `Capped` over `compile_steps` (advice_windows.py 262-284; advice.py 79, 95-108), a SHA-512 of Python source; `builder` is that item (CoreProtocol/Registration.lean 20-23, 74-80). The compile unit is "the target; not built" (prove_path/README.md 169-204). The slice is the schedule's part of that proof block and emits no Program.
3. The digests [read]: `program` is SHA-512 of the canonical Program (Spot/Registration.lean 537-553). `window.circuit` is SHA-512 of circuit.json with run and context (interface.py 2521, 2144-2146). The compiler's output is the record less those two (1898-1907). For step 2 to mean it [argued]: the proof block's circuit equals builder.digest (the compiler must first be registered as a circuit); its advice root equals window.advice.root (today the slice's adv is advice v1 under its own registration); its output's `program` equals the record's.
4a. Hash in gates [ran: arithmetic; the sizes are read in vllm notes, not in the tree]: 58,120 ANDs a block gives 1.77e9 ANDs for class 4,32,4's 3.9 MB record and 1.54e10 for the rehearsal's 34 MB one, against the slice's 1.02e6. And no circuit computes the text there is to hash.
4b. Known rows [read]: 119 or 1,039 proof blocks at 262,144 bits; the verifier's cost is linear in the text. The ZK server's salt rule refuses it as it stands (Salts.lean 338-357, 371-384), so it needs an exemption in Flock's firewall.
4c. Cheapest sound way [read, then argued]: `window_circuit` reads the advice only to refuse (window_circuit.py 756-766, window_class.py 127-161), so the Program is public from the class, and R2's declared day covers it (Spot/Registration.lean 206-217, 537-553). Left outside Lean: that the declared circuit is the class's, a public compile with no advice in it.
5. Order: proofs' step 1 and a declared day required for a served window, in Spot/Registration.lean; lean's statement in security/; optionally ten lines of vllm's in interface.py. Nothing of ours. First to land: a served log with a `builder` item; I found no served writer of one [ran: grep].
Dates: (c) has nothing of ours in its way today. The label: ready Mon 16 Nov, on main Wed 18 Nov, and I advise against calling it step 2. The compile unit, (a) and (b): no date I would stand behind; it needs a decision first (the compiler as a circuit), and under the certified tick no slice exists at all (serve_commit.py 74-77).
Not read: the lowering (window_circuit.py 121-676), compile_steps, compile_slice.py 261-799, AdviceV1.lean, Flock/Registered.lean, whether PoUW's known rows ran behind the value service, LEAN-238's comments older than the newest 30. No run, build or test.

