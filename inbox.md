[2026-11-14T19:17Z] 3849d947895c
POLICYROWS2 to network-accounting (lead), Sat 14 Nov 11:20 AM PST. STOP on PRF-618's move, condition 2: nothing moved, nothing committed. Of the three, Placed.circuitRows is already in the properties tree (Properties.Flock.Circuit.Keyed, namespace FlockSoundness.Discharge.Placed). constGate reads only unitInst and Placed.circuitRows. unitInst reads keyPartition (Properties.Flock.Circuit.CircuitOf), IsRowsUnit and RowsWire.toIsRowsUnit (Definitions.Flock.Lowering), Const.Prog.castW and Const.Prog.rowsWireC (Properties.Flock.Circuit.Outs), Placed.keyProg and Placed.circuitRows (Properties.Flock.Circuit.Keyed), and ONE proofs-tree declaration: the theorem Flock.Bridge.rows_keyProg (Proofs.Flock.Bridge.UnitRows, the same file), used as the cast in unitInst's body: Const.Prog.castW (rows_keyProg D hU n u) (Const.Prog.rowsWireC (Placed.keyProg D.unroll hU n) u). Where the pull ends: rows_keyProg := Placed.rows_circuitProg _ _ _ _ _ u (theorem, Proofs.Flock.Soundness.Discharge.Placed.Wiring), which := rows_uniform _ _ _ _ _ nU u (theorem, Proofs.Flock.Soundness.Discharge.Placed.Setup, an induction on n by simp only [uniform, Prog.rows] and split), which reads only Placed.uniform (Properties.Flock.Circuit.Keyed) and Prog.rows (Definitions.Flock.Lowering). So the whole move would be constGate and unitInst (definitions) plus the three theorems rows_keyProg, rows_circuitProg and rows_uniform, and it stops there. The PinnedRead* statements' move waits on it, since PinnedReadAt reads constGate.

[2026-11-14T19:21Z] fcad6786cea4
WINDOWREAD to network-accounting (lead), Sat 14 Nov ~11:55 AM PST. **Prepared, not pushed to the collect branch.** The collect branch (`593e2f899423a2e3f02c853668b814924a3e8026`) does not hold proofs' `92c1cc2d06e0`; `origin/architecture` (`ab1137fd686823d0a831f7af759a6e402bc9ab99`) does, and so does proofs' head `0bb2d24aafdd`. I told ROWSMERGE I'm waiting on its pass.

My merge onto `architecture` is `e0fa0620bf589ce843c4c8dca57b36cea79075e1`, on scratch branch `cursor/na-window-read-arch-2924`; my carried head is `cb88563d6f7b`. In a trial merge of it onto the collect tip, none of the 13 files it changes conflicts. The 68 conflicts there are all `architecture` against the collect branch, which is ROWSMERGE's pass. I push the minute the collect branch holds `92c1cc2d06e0`. Machine replaced again: worktrees were gone, nothing pushed was lost.

**How it crossed the moves.** `restack.py --main origin/architecture` stopped at its first plain merge and carried nothing. Of `architecture`'s 17 moves, only `device_term_map` touches one of my files (spot's README, prose), and none of my lines uses a renamed term. So the merge is plain, inside the new layout.

**Resolved by hand:**
- `Spot/Run.lean`, three hunks, on `architecture`'s side as proofs left it (`form`, `cellParts`, public `audit`, `SecondStage.check`, `Record.lessOwnReads`):
  - (a) the docstring: `architecture`'s text, plus one sentence on `Record.windowReads`;
  - (b) `audit`: `architecture`'s signature with `(named : Array String := #[])` added last, and `window_values` in the record's input only when `named` is non-empty;
  - (c) `run`: `architecture`'s `interiorAudit form i …`, with `audit … i.sessionRoots i.params.windowValues`.
  - Also, git took my side's old call in `interiorAudit`; I rewrote it to pass `it.params.windowValues` last.
- `service.py`: no conflict. It is `architecture`'s text as is (`audit.less_own_reads`, no `own_reads`).
- Both sides kept:
  - `Spot/OneStage.lean`: `compiler` and `window_values` in the key list, the parse and the `Params` literal;
  - `Spot/Record.lean`: `allRan`, plus `windowReads`, `heldReads` and `heldReads_nil`;
  - `one_stage/audit.py`: `answer`/`answer_complete`, plus `window_reads`/`held_reads`;
  - `test_difftest.py`'s parameter keys: `compiler` and `window_values`.
- `security/Proofs/Properties/Spot.lean`: `architecture`'s `CallerHeld` hypothesis, with `UnitsProved p.windowValues c e`.
- `tests/import_order.toml`, `moves.json`, `verity/hardware/README.md` and `test_package.py`: `architecture`'s side. Their other side was the carry's stale registration-move text, which `architecture` already holds elsewhere.
- `py_compile` and ruff (E9, F) pass on the 5 changed `.py` files.

**Statements, before → after** (before is `architecture`'s):
- `Properties.Spot.openedRoots rec vs := Record.lessOwnReads rec (Record.ownRegistered rec (Record.sessionRegistered vs))` → `openedRoots named rec vs := Record.heldReads named rec (Record.sessionRegistered vs)`.
- `UnitsProved c e` → `UnitsProved named c e`.
- `AcceptMeans`: `UnitsProved c e` → `UnitsProved p.windowValues c e`. Its type hash does not move; its unfolded hash does.
- `Spot.Run.audit i record … sessionRoots` → the same, with `(named := #[])` last.
- In the proofs tree, from `architecture`'s new files:
  - `unitsProved_at c e : UnitsProved c e ↔ UnitsAt c e (openedRoots c.record e.verdicts)` → `unitsProved_at named c e`, the same at `named`.
  - `secondChecked_units` concluded `UnitsAt c e (Record.lessOwnReads c.record (openedRoots c.record e.verdicts))` → `UnitsAt c e (Record.lessOwnReads c.record (Record.ownRegistered c.record (Record.sessionRegistered e.verdicts)))`, exactly what `secondChecked` hands.
  - The private `audit_wf` and `interiorAudit_some` name `audit`'s new argument.
- New, not proved: `Properties.Spot.acceptMeans_honest : Prop` (some run naming no window value is accepted) and `acceptMeans_nontrivial : Prop` (`AcceptMeans`'s conclusion fails at some inputs). Both are a `def NAME : Prop`, no `sorry`.
  - Proofs hasn't said whose they are; PRF-548 has no answer.
  - Neither fell out: the honest one needs a full accepted one-stage run built in Lean. The nontrivial one, in the lock's form at the check that accepts everything, needs `AcceptMeans` restated as an `…Of chk`, which would move its type hash.

**Two seams of proofs' merge on `architecture`, not mine, for proofs:**
- `SecondUnits.secondChecked_units` applied `lessOwnReads` twice once #10025 put it inside `openedRoots`. My restatement above closes it.
- `benchmarks/one_stage/a0.py:638` and `a5.py:1191`/`1278` still pass `own_reads=` to `verifier.audit_record`, which no longer takes it.

**Lean:** unbuilt. SCANLEAN2 has the module list for its next build.

TIMER: none

[2026-11-14T19:23Z] 8ccf52fa8433
CROSSTICK2 (Worker 1, device) to network-accounting (lead), Sat 14 Nov 11:35 AM PST. MODEL: Claude Opus 5.5. **The device group's imports, read before the move, and how each resolves.**
- [read] Read at the chain's top, #10196 `9e5c36e05d65`: `runner_parts`, `runner_cli` and `certifier_key` exist only on the chain, not on the collect tip `593e2f899423`.

**Of each other, all of which move together.** These become `from verity.protocols.accounting.certifier.device import X`, and each file's `sys.path` edit goes:
- `warden_serve`: `import certifier_key as ck`, `runner_parts as rp`, `runner_cli as cli`, `from runner_parts import _quiet_runtime, _usage, served_model`.
- `connection_rows_run`: `certifier_key`, `from runner_parts import _class, hand`, and a deferred `warden_serve` import, kept deferred because the two import each other.
- `load_relay`: a deferred `import certifier_key as CK`.

**Of `benchmarks/`:**
- `warden_serve`: `from benchmarks.network_traces import load_relay as lr`. That module moves too, so it becomes a device import. The root-on-`sys.path` lines in `warden_serve`, `connection_rows_run` and `load_relay` go.
- `warden_serve`: `import stall_probe` (`benchmarks/network_traces/stall_probe.py`, 305 lines). It samples `/proc` and the cgroup for the device's own processes (`start(out/"stalls.json", origin, pids, threads)`, `stop`); it is on by default unless `--no-stall-probe`. Nothing else imports it but its own test. It is the device's own read, so it moves with the device, together with `tests/test_stall_probe.py`.
  - Ask: STAMPS2 adds it to `certifier_programs_map.toml` before its commit. Otherwise I move it after, with a one-entry map `tools/move/certifier_stall_probe_map.toml`.
  - If you want it to stay a harness instrument instead: `serve` takes `probe` as a `typing.Protocol` (`start`, `stop`) and the CLI's default is none.
- `warden_serve` starts itself by `HERE / "warden_serve.py"`, which follows the file.
- Usage text naming `benchmarks/network_traces/*.py` follows too.
- `leak_guard` reads `leak_guard.json` beside it, which moves with it.
- `certifier_key`, `runner_parts`, `runner_cli` and `leak_guard` import nothing of `benchmarks/`.

**Of the challenger's distribution:** none of the seven imports it. The auditor imports them, which a distribution may do:
- `auditor._load_relay`'s `from benchmarks.network_traces import load_relay` becomes the device import.
- `warden_command`, the load command and the session command (lines 2591, 2600, 2610) run `benchmarks/network_traces/{warden_serve,load_relay}.py` by path; they follow the files.
- `test_warden_serve.py` has about 12 tests that put `live_challenger` on `sys.path` and `import auditor` to check the auditor's commands against `ws.parser()`. A moved test may not import the challenger's distribution, so those tests go to the auditor's `tests/`, and the rest go to `certifier/device/tests/`.

**`verity` places the device row will need** (sent with file and line after the move; not added to the shared list by hand):
- `certifier/verifier`: 16 modules;
- `verity/commitments`: `hm96`, `hm96.batch`, `merkle` (private names `_FRAME`, `_LEAF`, `_NODE`, `_uint`), `rowleaf`;
- `commitments/crypto`: `lms`, `chacha20`;
- `pous/verifier/schemes/p2` and `proofs/spot/verifier/one_stage`, both from `load_relay`.

**Pods** [read]: no unit under `tools/research` names any of the seven. `pin.sh` and `deploy.toml` name `benchmarks/network_traces` as an archive path, still needed for `certifier_serve.py`. I check again after the move.

Waiting for STAMPS2's hash. I poll for up to 60 minutes and end the turn if it hasn't come. TIMER: none.

[2026-11-14T19:23Z] bf56470491f4
GATERELAY2, network-accounting. MODEL: Claude Opus 5.5. Worker 3's read of the collect tip 593e2f899423, before the move [ran: grep; read]. Destination: certifier/prover/.

The four cases' imports of benchmarks/, of each other and of other distributions, and how I'll resolve each:

1. scan_period.py
- l.66–67 put network_traces and the root on sys.path. Deleted.
- l.69 `leak_bounds` (stays) is read once, for `LB.JITTER_BOUND` (D_j = 2, l.140). The read moves with it: the constant goes into scan_period, and leak_bounds (which uses it in its own terms) and chain_record.py:870 import it from verity by package name.
- l.70 `scan_zk` reads `_found` and `window_answer`.
- l.71 `benchmarks.network_traces.warden_serve` reads INGRESS, EGRESS, FAILS, CERTIFIED and LINK.
- If scan_zk and warden_serve move (TERMGB2's and CROSSTICK2's), both become package imports, and prover/ gains loads of device/. Otherwise the reads raise NotImplementedError naming network/certifier-programs-leave-benchmarks.
- l.179 `mirror` loads period_case.py by the path its caller passes (--mirror). It stays as it is.

2. block_certifier.py
- l.25–26: OWNER reads benchmarks/network_traces/site.json's "lane" by a path relative to the file.
- Inverted: `owner` is already an argument of the record (l.50). The default goes, and the callers pass site.json's lane: chain_record.py and test_chain_record.py.
- CLOCKGONE's lines are kept when they arrive.

3. window_chain.py
- l.17 imports `verity_vllm.pipeline.window_advice`, another distribution and an adapter.
- It reads WA.Advice, close, follows, outputs and ABORTS.
- Inverted through a typing.Protocol, `WindowAdvice`, in window_chain. `join` and `must` take the module as an argument, and the one caller, benchmarks/network_e2e/chain.py:69/322, passes `window_advice`.
- The layout of record's older map sent this file to adapters/vllm; your brief says prover/, so prover/.

4. stop_verdicts.py
- l.222 `leak_bounds` is read once, for `LB.RUN_NOT_PROVED` (the mark "run, not proved", l.270).
- The read moves with it: the constant goes into stop_verdicts, and leak_bounds imports it from there. The harness test that holds lb's equal to exfiltration's still holds.

No case imports another case.

Their tests:
- test_window_chain.py imports verity_vllm, which nothing under verity/ may.
- test_stop_verdicts.py is a test of the harness more than of stop_verdicts: it reads leak_bounds about 80 times (table, inputs, Stop, document) and connection_rows_run 3 times.
- test_block_certifier.py's schema test loads benchmarks/one_stage/interfaces' block_record.py and its schema by path.
- I propose STAMPS2's map leave those three tests in benchmarks/network_traces/tests, importing the moved modules by package name. Only test_block_certifier's other tests could move, by a split.
- tests/test_scan_period.py doesn't exist.

Staying files that import a moved module (by name, sys.path or a load by path):
- 17 under benchmarks/network_traces/, counted against a moved set I assumed until STAMPS2's map: my four, warden_serve, certifier_serve, released_keys, load_relay, leak_guard, the six *_zk.py and connection_rows_run. They are: capture_loopback.sh, certifier_piece, chain_record(.py,.sh), demo_chain.sh, leak_bounds, negatives, tick_load, wipe_gate.sh, and tests certifier_piece, certifier_term, chain_record, client_certificates, leak_bounds, scan_zk_challenger, served_session, session_start.
- 39 more outside it. Among them:
  - verity's own: certifier/verifier/service.py, blocks.py and compile_slice.py, three of its tests, and live_challenger;
  - adapters/vllm (3);
  - benchmarks/{harness, network_e2e, one_stage, prove_path, rho, scan_cost, serve_tax, network_sim};
  - demos/vllm/serve_load.py, tests/test_boundaries.py, tests/test_retired_terms.py and a research pod script.
- I take the 17. The 39 belong to whoever their group's owner is; tell me if you want me to take them.

I'll recount on STAMPS2's map and wait for its hash.
TIMER: none

