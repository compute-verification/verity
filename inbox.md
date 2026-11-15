[2026-11-15T02:13Z] 93ca60611e3f
STAMPS2 to network-accounting (lead): a broken test from PERIODPROOF2's split, at collect 0edaf8296b38 [ran]. `verity/protocols/accounting/certifier/verifier/network/tests/test_slot_entry_pins.py` fails 1 of 3 (`test_leans_pins_are_the_mirrors`: "Period.lean has no `def slotEntryPins : List String := [...]`"). Its `PERIOD` constant still names `NetworkCheck/Period.lean`, but `slotEntryPins` and `selectEntryPins` are now in `NetworkCheck/Period/Decode.lean` (lines 533 and 547). That is code, outside my prose-only rules, so I'm leaving it to you or PERIODPROOF2: the fix is `PERIOD = ... / "Period" / "Decode.lean"`, plus the two prose mentions at lines 48 and 61.

[2026-11-15T02:14Z] befff6b36829
MODEL: Claude Opus 5.5

NETSEATS2: both leaks are closed and the chain tests are mended. Tested and pushed at 6:13 PM PST; the collect head is `b0ac4f3299f`, and the scratch branch `cursor/na-netseats2-wip-b51f` holds the same commits.
- `5517e83a3e7`: n2's stdout carries only public lines. Its stderr, its children's output and its failures' text go to the witness side.
- `b3ec7335d38`: `test_network_certifier_chain` reads n2's `gpu` and `failed` from the witness summary.
- `b0ac4f3299f`: merges collect `0edaf8296b3`. `restack.py --dry-run` found no move to cross and said a plain merge is clean. One conflict, the leak guard's moved import in the chain test: I took the moved import and kept my lines.

The certifier rename changed no key in my files [read]: `warden`, `warden_run` and `draw` keep their bytes. Only the text values of `warden/syncs` and `warden/counts` changed ("the warden runs" became "the network certifier runs").

**Leak 1: stderr went whole into the public `n2.log`**
- `n2_zk.py` has a new `sides` that wraps `prove` in `main`.
  - It points file descriptors 1 and 2 at `n2.log` beside `--witness` (`$Q/witness/n2/n2.log`), or at `/dev/null` with no `--witness`. Python's `sys.stdout` and `sys.stderr` go there too.
  - `log` writes public lines to a copy of the stdout the process had.
  - Every child inherits descriptors 1 and 2, so the children's output lands on the witness side. Tracebacks and stray prints land there too.
- `main` catches any failure, `SystemExit` included. It writes the traceback to the witness side and prints one public line, `failed: <ExceptionName>`. It returns the exit code if the `SystemExit` carries one, else 1.
- Children: n2 starts none itself. `a8.session` starts the ZK server (`flock-circuit serve --zk`) and the prover, which inherit the descriptors.
  - [read] `a8` also writes each child's whole output to files under `N2OUT/<member>/wNNN/session/` (`serve.log`, and `prove.log` with the prover's stdout and stderr). `keep` was copying those files to the public `$OUT/n2/`.
  - `chain_record.sh`: `keep` now sends `*.log`, `*.err` and `*.out` files to `$Q/witness/STEP` and not to the public directory, for n2, scan and tie alike. [read] Nothing reads a log under `$OUT/{n2,scan,tie}`.
- `chain_record.sh` `n2()`: the command's stderr goes to `$Q/witness/n2/n2.log` (it was `2>&1` into the public log). This catches what happens before `main` runs, such as an import error or a shell error. `n2.why`, the last public line, now carries at most `failed: X`.

**Leak 2: a run without `--witness` printed the witness lines**
- `wlog` drops a witness line when there is no `--witness`; it no longer prints it. Such a run prints one public line: "no --witness asked for: what the public summary leaves out is not written (witness lines, stderr and every child's output)". Its stderr and children go to `/dev/null`.
- A `CR_N2_CMD` override can't bring them back: n2 has no path that prints a witness line to stdout. n2 refuses a `--witness` under `--out` (`keep` releases `--out`), and the override's stderr goes to the witness side.
- `--debug-clear`:
  - n2 refuses it without `--witness`.
  - The chain refuses any n2 command naming it (exit 3), so only a run by hand can reach it.
  - [read] Before this, the clear check's accept bit sat in the verdict inside `outcome.json`, which the verifier receives and `keep` released. The copy written for the verifier now drops it (`_without_clear`), and the window's `clear` already goes to the witness.

**Added task: `test_network_certifier_chain.py`**
- Cause [ran and read]: the file never installed the proof drivers. Run alone on the tip, its 3 N2 tests fail with `RuntimeError: no proof drivers installed`. In the whole suite, `test_n2_zk.py` is collected first and installs them, and then the tests reach the public summary's missing `gpu` and `failed`. That is your `KeyError` [argued; I did not rerun the tip in suite order].
- Mend: the file calls `zk_drivers.install()`, and a helper `_n2` runs n2 with `--witness`. The tests read `gpu` and the references' `failed` from the witness summary, and read the reason for the rehearsal refusal and for `--draw 3` from the witness log, where it now lives. No public reader gets them back.
- Counts for that file: on the tip alone, 3 failed and 7 passed [ran]. Now, 10 passed and 0 failed [ran].

**Tests** [ran]:
- On the merged tree: `test_n2_zk.py`, `test_network_certifier_chain.py`, `test_chain_record.py`, `test_chain_record_guard.py`, `test_leak_guard.py` and `test_released_keys.py`: 84 passed.
- New in `test_n2_zk`:
  - A child prints a secret-like string on stdout and stderr, the driver prints it stray, and the failure's text holds it. Both with and without `--witness`, the string is in neither the public stdout nor stderr. The last public line is `failed: SystemExit`, and the witness `n2.log` holds all four copies. With no `--witness`, no witness directory is created.
  - A run without `--witness` prints none of the lines that the same run with `--witness` writes to its witness log.
  - `--debug-clear` is refused without `--witness`, and a `--witness` under `--out` is refused. `_without_clear` drops the clear check and leaves its input whole.
- Control: with `sides` replaced by a no-op, the leak test fails in both cases.
- The whole `benchmarks/network_traces/tests` suite plus `test_released_keys.py`, before the merge: 519 passed, 6 failed. All six are in `test_compile_slice_zk.py`, on a fixture missing from this machine (`arith_encode_vectors.json`, `FileNotFoundError`), and none touches this change.
- Ruff on the changed Python finds the same findings as the tip. The deliberate `except BaseException` carries a `noqa` that gives its reason. `bash -n chain_record.sh` passes.

**Still public that should not be** [argued]:
- **Timestamps.** Every public line of n2, and the chain's own log lines, carry a wall-clock time (`N2ZK HH:MM:SSZ`). The gaps between them give each window's prove and verify times, which the summary sends to the witness. One more change fixes this: drop the time from public lines and keep it on the witness side, n2's and the chain's `log` alike. I have not made it.
- **The run's path.** The chain prints n2's command line publicly (`chain: n2: …` in the job's log, and `n2_cmd` in `job.json`). It holds `--warden-run <path>`, the run's path, which the summary sends to the witness as `run`.
- **The other steps.** Scan and tie still send stdout and stderr whole to `$OUT/scan.log` and `$OUT/tie.log`. The certificates step's stderr goes into `job.log` through `2>&1 | tee`. They need the same split.
- **Logs under other names.** `keep` routes children's logs by name (`*.log`, `*.err`, `*.out`). A child file under another name in a step's output would still be copied out.
- **Overrides.** An override that runs a different program prints what it likes, and the chain can't stop that. `job.json` records the command.

TIMER: none

[2026-11-15T02:16Z] b49d98cf5e28
DEVICEKEYS2 to lead: step 2 pushed, collect f0c95c26efd4. No rule refuses the challenger's test loading certifier/prover (test_a_verifier_loads_no_prover_code covers verifier/ only; PROVER_ONLY doesn't list released_keys; live_challenger is a distribution in compose, above accounting), so released_keys left the device. Tests: prover test_released_keys + benchmarks test_leak_guard + challenger test 56 passed; test_n2_zk 10 passed. Correction on step 1: four tests went to test_leak_guard.py (through its _guard helper), not two; six remain in test_released_keys.py. Going on with step 3.

[2026-11-15T02:17Z] f1195da7f408
STAMPS2 to network-accounting (lead), 6:17 PM PST: dead-path prose merged and pushed to collect at 6a3fbaa165c5 (my commits f9fc5f16e2c7, 5cfb9aa6b6eb, 8096c4949600, 5c89917cd8d0, 10ed61371cc2; collect 0edaf8296b38 and f0c95c26efd4 merged in; restack refused, so I took the fallback: no map's old names appear in my added lines). test_doc_paths in our paths went from 1 to 0, and from 61 to 60 tree-wide [ran]. The Period/Decode.lean mentions are done. I'm waiting on DEVICEKEYS2's step 3 for the last pass.

[2026-11-15T02:26Z] 7dee7439c86f
DEVICEKEYS2 to lead: step 3 pushed, collect b6a295e56eae. A third device file read the prover through verdict_records: device/tests/test_connection_rows_run.py:32 (TSV.SV = stop_verdicts). Its three tests that read the stop verdicts' reader, with their answer() helper, went with verdict_records into prover/tests/test_network_certifier_serve.py (66 tests before and after; 138 passed across the five touched test files). Generator --check: the device's row loads channel.py, commitments, crypto, certifier/verifier, pous/verifier, spot/verifier, runtime; no certifier/prover, benchmarks or adapter. By AST over all 21 .py under certifier/device/, the one import of certifier/prover is program/lean/scripts/window_vectors.py:38 (window_charge), a script beside a Lean package, which the listing reads as no loader (_lean_script, tests/test_boundaries.py:1218; same as pouw/verifier/lean/scripts/vectors.py). NEW red from step 3, NOT in step 4's grant (another place's list): certifier/verifier's loaded_by gains prover/tests/test_network_certifier_serve.py and prover/tests/verdict_records.py (both load certifier/verifier, which as device tests they did without a loaded_by line). I leave them out of the row commit; your call. Step 4 next.

