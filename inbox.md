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

[2026-11-14T19:24Z] ca18aec70e0c
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 11:24 AM PST. INVENTORY (3a), collect head 9b88525827e685a7ab6909c4235acd8c5cbdd097, nothing pushed since. (i) NetTiming.SecurityProofs.PeriodLeakAccepted: staging guarantee, the period leak at the check's numbers over the older Deploy, proved; collect has it as NetTiming.SecurityProofs.MonitoringPeriodLeakAccepted over Compiled, proved, in the lock; repointed (push 1). (ii) NetTiming.SecurityProofs.PeriodTiedCount: in no Lean file or lock on staging either; the record names it as to be restated, always open; nothing to bring. (iii) Properties.NetworkPeriodLeak.PeriodLeakRegistered with theorem periodLeakRegistered: staging Property, on the period check's rows-form acceptance with the certifier's LMS, key and session the auditor's certifier-key/v1 record (NetTiming.NetworkCertifierRecord, also absent) and A.key cert.id = some (hexOf key), the period leak at the registered inputs' values; proved from NetTiming.periodLeak_registered (staging PeriodReg.lean, older Deploy) with rowsWf_of_accept and cellRows_wf_of_rowsWf; collect has nothing at the check's acceptance, only SampledLeakRegistered per phase, and #9350's PooledLeakRegistered (unmerged) is the registered period form it would now stand on. NOT BROUGHT: PERIODPROOF2 has not answered (asked 11:07 AM); a restatement over today's foundations should stand on #9350's form, not beside it. (iv) rows-state: Properties.NetworkPeriod.RowsStateHeld, proved (rowsStateHeld, _honest, _nontrivial), over verifier code NetworkCheck.Period.stateHeldIn, auditStateEntries and Audit.stateFree, a tighter charge; collect has stateEntries only, charges a tied audit a.free always, and RowsStateTied (proved) refuses a mismatched state. Bringing it changes what the verifier publishes (smaller K) ahead of a proof redone over the collect check: your ruling. (v) leak_bounds per-period name: MonitoringPeriodLeakAccepted on collect; the record reads it. 3c: make.py rerun at today's names, m1 still says the network certifier is not established; the one premise is certifier-key, open because the lock does not list Properties.NetworkPeriodLeak.periodLeakRegistered; m1 equals m1-key-open; fixtures left. STOPPED on PERIODPROOF2's word for (iii) and yours for (iv) and leak_bounds.WEIGHTS. SCANLEAN2 not told: no Lean module added. TIMER: none.

[2026-11-14T19:24Z] b3531cb44fc1
SCANLEAN2 to network-accounting (lead), Sat 14 Nov 11:24 AM PST. **Build run 1 (`r20261114-190001-c39c`, tip `88eb3c6906055f9284d9d4e2a58ca4a5309e45e4`) found one error, mine, now repaired on the branch. Run 2 (`r20261114-192327-ef10`) is building tip `593e2f899423a2e3f02c853668b814924a3e8026`, on research CPU machine 15.**
- **Built clean at run 1's tip:**
  - `network_check`'s default targets (`NetworkCheck` and its six executables), with my `ScanOk.named` change in `Period.lean`;
  - `NciCheck`, `NciCheck.RowsProtocol` and `inference-only-rows`;
  - `Properties.NetworkScan` with its submodules;
  - `Proofs.Properties.NetworkScan`;
  - every module of `Proofs.Properties.Nci`, among them PROVENANCE2's `RecordDeclared` (with the applied mend), `RecordRoles` and `RecordProvenance`, and POLICYROWS2's `RecordForm` and `Lemmas.Opening` as they stood at that tip.
- **The error:** `security/Properties/NetworkPeriod.lean:404:67`, ``Invalid field `key` … `NetworkCheck.Period.Params.key` ``, in my `RowsExecutionPhaseAccepted`. I wrote it against a `Params` with a `key` field, and DEVICEKEYS2's #10091 merge (`5b1210d25df`) put the keys in `Params.entries`. So it is my statement meeting that merge, not DEVICEKEYS2's error. `Proofs.Properties.NetworkPeriod` did not build, only because it imports that file.
- **Mended:** `5c99ce96cc8`. One conjunct, before `p.key = k.publicKey`, after `(∃ x ∈ p.entries, x.key = k.publicKey)`; the docstring now reads "under parameters one of whose entries' keys is the public key of a key at the deployed LMS parameters". The ratchet gate passes. Pushed after a fetch and a merge (a fast-forward), collect tip `fe72d9114e09`.
- **Left red: none known.** Run 2 checks the mend and adds POLICYROWS2's later commits, CLOCKGONE's and #9867/#9760's merges, and `Properties.Spot` and `Proofs.Properties.Spot` for WINDOWREAD. #10103 is not on the branch yet.
- **How long:** about 23 min from launch to end. Setup is about 3.5 min (Mathlib's cache comes down; main's build cache needs credentials the machine doesn't hold, so in-repo modules build from source). `network_check` takes 1 min, the policy package 1.3 min, and the proofs package 15.4 min, most of it Flock's soundness modules.

[2026-11-14T19:24Z] ac24970d8352
MEND9319 to network-accounting (lead). **#9827 (`21196b7d766198695f80f89536f89f5d0597cb6a`) is merged: the collect branch is at `281fc68ebda503aa0f9c680643ecea29c644d576` after my push.**

- **The carry.** `restack.py` carried the head across `ac8b3b51e490` (`clean`, `869125b13c48`). Across `8469d5dcc164` it stopped at the move's base on `tools/research/src/research/store/moves.json`. I merged that base by hand, keeping both moves' entries, dither's then clear's; the file is byte for byte #9639's resolution. Rerun, the carry was `clean` (`adf6c1fbf0f4`).
- **The merge into the collect branch.** `moves.json` conflicted again, on the order of the same moves. I took the collect side's file, which holds every line of the change's and #9663's added `PeriodChecked.lean` line besides [ran: compared]. The change's one file, `benchmarks/one_stage/tests/test_kept_compose_cache.py`, merged by itself; it differs from the head's only by the collect side's `SV.RULE` for `"subset:1"`. My first push was refused because the collect branch had moved; the remerge was clean.
- **Nothing fails to compile.** The test file compiles (`py_compile` [ran]).
- **Lean:** none in the change.

[2026-11-14T19:24Z] 7ea81fd0a083
SCANLEAN2 to network-accounting (lead), Sat 14 Nov 11:26 AM PST. **Correction to my 11:24 line: CLOCKGONE's deletion is not on the branch at run 2's tip `593e2f899423`.** `Period.lean` still holds the clock form and `Audit.bits`. Neither CLOCKGONE's merge nor #10103 is in run 2; each gets its own pass when it lands.

[2026-11-14T19:25Z] 66599edfd83c
TERMGB2 (Worker 2, ZK drivers) to network-accounting (lead), Sat 14 Nov 11:24 AM PST. **My group's imports before the move, and how I will resolve each.** [read]

I read them on STAMPS2's tip `329c3f314299` (`cursor/na-runners-gone-stamps2-04bd`), since its chain goes onto the collect branch (`593e2f899423`) before the move. GATERELAY2's device move `7dd8e645163c` is already on the collect branch.

**Proofs' move fixes three of mine itself.** Proofs' move is `0cb16dfc645`, map `tools/move/audit_code_map.toml`, on `origin/architecture` `be0f6a3ca361`, and **not yet on the collect branch**. On `origin/architecture`, its hand commit already rewrote my files:
- `a0` becomes `spot/prover/audit/template_audit`, in `warden_zk` and `n2_zk`;
- `a5_coins` becomes `spot/device/challenger_session`, in `scan_zk` and `a5_warden`;
- `a5` becomes `spot/prover/audit/window_audit`, in `a5_warden`.

Its map also takes most of `benchmarks/prove_path/` into SPOT: `prove_path`, `run.sh`, `class_size`, `window_cost`, `class_rows`, `records_audit` and their tests. What is left of `prove_path` for me is `served_statics_zk`, `served_tie_zk` and `served_tie_window`, with their three tests. These fixes reach us only when the collect branch takes `origin/architecture` across `0cb16dfc645` with restack. The move needs that to happen first, or STAMPS2's map collides with proofs' over `prove_path`.

**Imports of `benchmarks/` and how I resolve them:**
1. **`a7`, `a8`, `prove_leg` (`benchmarks/one_stage/`, the proof system's benchmark drivers; they stay there).** What they read: `a7` gives `M_MAX`, `statement_checks`, `frame_ports`, `opens` and `public_bcs`; `a8` gives `session`, `received`, `PIECE`, `find`, `needles` and `cost`; `prove_leg` gives `frame_ports`.
   - Every driver already reaches them only through `warden_zk._one_stage()`/`n2_zk._one_stage()`, which returns one namespace `M`. `leak_scan` already takes `a8` as an argument.
   - Resolution: inversion. The moved drivers take `M` typed by one `typing.Protocol` (`ProofDrivers`), and `_one_stage()` becomes the builder in one benchmark-side entry, `benchmarks/network_traces/zk_drivers.py`. The `*_zk.sh` files call that entry, which hands `M` to the moved `main`.
   - In `scan_zk`, the lazy `from prove_leg import frame_ports` goes through `M` too.
2. **`warden_serve` (in `scan_zk`, `n2_tie_zk` and `warden_replay`; CROSSTICK2's, going to the certifier's `device/`).**
   - `scan_zk` and `n2_tie_zk` read only `WS.bf`, which is the certifier verifier's `blob_files`. I import that directly; no tie is left.
   - `warden_replay` reads more and runs `warden_serve.py` by path. It follows CROSSTICK2's new path.
3. **`warden_chain` (`n2_zk`: `pairs`, `link`).** It imports only `verity`. If STAMPS2's map moves it to `prover/`, this is a sibling import. If it stays, `n2_zk` takes `pairs` and `link` through `M`. **Your word: does `warden_chain` move, and in whose group?**
4. **`compile_slice_zk`'s `_encodings()` and corpus readers.** They load `benchmarks/network_e2e/encodings` (`replay`, `replay_vectors`, `cache_vectors`) off `sys.path`, plus fixtures under it. The vLLM scheduler's replay is not Verity's. Those functions (`_encodings`, `engine_class`, `_corpus` and the reads of `TRACE_8EB1`) go to the benchmark-side entry and arrive through `M`. That fixture is already gone from the collect branch and STAMPS2's tip, moved to the store, so the corpus path fails today regardless.
5. **The traffic generators loaded by file.** `scan_zk` and `egress_zk` load `certifier/verifier/tests/{scan,egress}_traffic.py` with `spec_from_file_location`. These are the certifier's own test traffic, so the load becomes a path relative to the new place. If the boundaries test counts a by-file load from `verity`, they become an argument too.

**Other distributions:**
- `verity_flock` is used for `circuit` (`BC_BYTES`, `.txt`, `_slot_log` and other format constants) and `class_statement` (`UnitStager`, `Binding`, `Member`). It is used in `egress_zk`, `warden_zk`, `n2_zk`, `served_statics_zk`, `served_tie_zk` and `a5_warden`. It goes through `M` the same way: the entry imports it.
- `verity_vllm` is used for `commit.store_dump` (`served_tie_window`) and for `commit.index` and `pipeline.serving_rows` (`a5_warden`). If those files move, they go through an argument. If they stay under `benchmarks/`, nothing changes.
- `leak_scan` puts `flock/prover/agreement/` on `sys.path` for `redigest.layout`; that directory is in no place. Resolution: inversion, with `public_pieces` and `scan` taking the reader as they already take `a8`.
- Where the `served_tie_zk` cost constants (`FC.K_MAX`, `_ranges`, `_pow`) don't thread quickly, the body raises `NotImplementedError("network/certifier-programs-leave-benchmarks")`, and I'll name each one.

**Each other's imports** become sibling imports under `prover/`:
- `scan_zk` imports `warden_zk` and `leak_scan`;
- `egress_zk` imports `warden_zk` and `leak_scan`;
- `warden_zk` imports `leak_scan`;
- `n2_zk` imports `leak_scan`;
- `n2_tie_zk` imports `n2_zk` and `leak_scan`;
- `compile_slice_zk` imports `warden_zk`, `n2_zk` and `leak_scan`;
- `served_statics_zk` imports `served_tie_zk`, `served_tie_window` and `leak_scan`;
- `served_tie_zk` imports `served_tie_window` and `leak_scan`;
- `a5_warden` imports `warden_zk` (`WardenRowIdentity`, `flip`).

The `sys.path.insert(HERE)` lines go.

**Shared lists that would need to grow** [argued, to be confirmed with the generator's dry run after the move]. The certifier prover's place loads `circuits`, `commitments`, the certifier's `verifier` and SPOT's `verifier`. The drivers also load:
- `verity/commitments/crypto` (`lms`);
- `verity/protocols/proofs/spot/device` (`coins`, `challenger_session`);
- `verity/protocols/proofs/spot/prover` (`template_audit`).

Each moved driver also needs a line in the certifier verifier's `loaded_by`, and in SPOT verifier's if it is listed. I'll send the exact list after the move. I grow nothing.

Next: I poll the inbox and the collect branch for STAMPS2's hash for up to 60 minutes.

[2026-11-14T19:28Z] 79c617fe512c
GATERELAY2, network-accounting. MODEL: Claude Opus 5.5. Pushed 6b0edc2daa3: device_party.open_session's docstring also signs the run's start (certificate.DeviceState.start for base and ordinal over windows · window · bucket_ms; certificate.Issuer.certify_start under start_link). This is from DEVICEKEYS2's line, its facts checked against certificate.py and warden_serve.py [read]. Docstring only; it compiles [ran]. Collect head after the push: 0f531d913b7e. No file resolved by hand, no statement changed. Still waiting for STAMPS2's hash: no certifier_programs_map commit on the collect branch as of 11:30 AM PST.
TIMER: none

[2026-11-14T19:28Z] e9b64c2ce09f
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 11:28 AM PST. PUSH 2: collect head 9848a6a970c48e0fb153efe4f0f0f218afd910fc, four commits. 91c8f667a40: the record's weights share is leak_bounds.WEIGHTS (GLM-5.3 in FP4) with its workload card as source, and the census model is no longer cited. 77eb09b0b9b: reverts f141f384013. After 91c8f667a40 nothing reads total_params, weights_bytes or weights_datatype of the census's adversary model (figure1 reads weights_bytes from its own workload card), so withdraw your TOP-424 line; the census validates against its schema. 325c29a7d4c: the state-charge test expects free with RowsStateHeld unpinned, its docstring saying why (the check charges the whole free inputs, RowsStateTied refuses a mismatched state). 9848a6a970c: make.py's description gains one sentence, since run on this tree it writes an m1 that is m1-key-open: 'The committed m1 was cut at a lock that lists Properties.NetworkPeriodLeak.periodLeakRegistered; at one that doesn't, this tree's among them, certifier-key is open in m1 too and it is m1-key-open.' Fixtures unchanged. No Lean statement added. Tests: onsite_record 221 of 222; the one left is certifier-key, waiting on periodLeakRegistered in a lock; onsite_periods 30 of 30. PeriodLeakRegistered: #9350 (head 5e514f92d46f) is not on the collect branch, so it is left unstated. I am free for SCANLEAN2's mending. TIMER: none.

[2026-11-14T19:30Z] 7be3a646bf34
RUNNERSGONE (5.1, the older certifier form's drivers deleted): merged and pushed to cursor/na-architecture-b51f. My head 5e7582763180de2fad6ed5ba516d76ed5ef77721 (the top of the stack on STAMPS2's 329c3f31429, the relay cut with it). The collect branch's head after my push is 2d1275ee44e712cab1dfa065850881d7dd21efcb. Six of my pushes were refused as non-fast-forward, so I merged the collect branch again each time. One of those merges crossed the layout move 7dd8e645163c (certifier_device_map) with tools/move/restack.py --main 7dd8e645163c, which came out clean, before merging the tip; restack --dry-run found no other move. Files I resolved by hand, porting the collect side's edits to where the cut moved the code (relay_check, connection_rows_check, record_files, runner_parts, certifier_key): warden_serve.py, connection_rows_run.py, chain_record.py, n2_zk.py, scan_period.py, warden_chain.py, negatives.py, README.md, leak_guard.json, leak_guard.py, the relay_check.py, connection_rows_check.py, record_files.py and runner_parts.py modules, warden.py, content_rows.py, PROTOCOL.md, period_case.py, auditor.py, active.py, stop_verdicts.py, one_stage/warden_replay.py, and the tests test_connection_rows_run, test_warden_serve, test_session_start, test_warden_chain, test_scan_zk, test_out_request_values, test_leak_guard, test_certifier_piece, test_network_certifier_warden and test_onsite_record. Kept deleted although the collect side edited them: active_replay.py, certifier_serve.py, tick_load.py, test_active_replay.py and test_client_certificates.py. Their changes (cpu_ranges, the sorted certificate JSON) went to runner_parts and record_files. Does not compile: nothing; py_compile is clean on every .py I changed or resolved. Tests that fail on the collect branch's own tip too, so not from my merge: test_connection_rows_run::test_a_deployment_record_of_another_session_than_the_one_handed_gets_no_file (the start check's problem now comes before the windows'), and test_network_period_protocol::test_a_period_whose_frames_overrun_a_u32_is_refused (period_case.check unpacks 3, the test hands 2). Run r20261114-183440-4bd9 is cancelled; I launched no other run. No PR is opened and none is marked. I changed no statements.

[2026-11-14T19:31Z] 08c657abfcea
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 11:34 AM PST. 5.2: your (d) reading for PoUW does not hold as stated. The four PoUW files (benchmarks/pouw/tests/conftest.py, test_salt_bind.py, test_served_zk.py, verity/protocols/accounting/pouw/verifier/tests/test_pouw_salt.py) use the active certifier's whole salt and reveal path, not only `Proxy`: Proxy(...) with salt, registered and prover_key, bucketer.tick, commit_weights, serve_reveals, CertifierClient, weights_message, fresh_salt and served_window. Those names need 1,237 of active.py's 1,776 lines plus online.py and served.py, and the warden has no salt or reveal path, so a small helper would be the older certifier moved into PoUW's tests, a stand-in. Proposal: 5.2 deletes active.py, online.py and served.py, and those four files are left failing and named until compute-accounting's 4.2 moves PoUW's salt onto the warden (they are then the only readers left). Alternative: active.py, online.py, served.py and their tests stay until 4.2, and 5.2 lands without them. I take the proposal at 11:50 AM PST unless you say otherwise. Circuits: I picked ingress_scan.IngressScanBucket at the scan test's class (B=512, r=2, lanes=2, C=4, IDB=17, EVERY=2, WITHIN=3, D=5, KO=8, KI=100), the only other gated Definition; its pin holds and its digest takes 4.5 s. network_sim: the ten names are now benchmarks/network_sim/workload/canonical.py, text unchanged, and test_framing and test_lab pass (18). Service rewrite: the certifier's service, head_check, link_window_lean, period and tie_service tests pass (72), and network/tests/test_lean_audit.py passes (11). Nothing pushed (5.1 and CLOCKGONE are not on the collect branch). My head f3569956f41 (local), collect 593e2f899423.

[2026-11-14T19:31Z] 8a05b1f4d014
MEND9319 to network-accounting (lead). **#9883 (`f712a6405bbdff14ff56eea2a6097161562173b4`) is NOT merged: its carry conflicts with the collect branch (`0f531d913b7e`), so I left it, as you said.**

- **What it touches:** four files under `adapters/vllm/`: `verity_vllm/pipeline/tick_plugin.py` (+67 −9), `serve_admit.py` (+8 −7), `ids_front.py` (+3 −2) and `tests/pipeline/test_serve_admit_tick.py` (+150 −3). Under vLLM's own server, the tick's middleware now gives a lost release's request its disconnect and answers it with 400 or the stream's end. `TickAdmit.LOST` becomes the module-level `serve_admit.LOST`. It has no Lean.
- **The carry.** `restack.py` carried the head across `ac8b3b51e490` (`clean`, `f850807a42b4`), then across `e4a8682800d6` and `8469d5dcc164` (`clean`, `8fe8c4b2ed9f`). The head holds neither `ac8b3b51e490` nor `8469d5dcc164`, so the carry reran the clear and registration maps on the branch's text.
- **The merge of `8fe8c4b2ed9f` into the collect branch.** The four adapter files merge by themselves. Four files the change itself never edits conflict, each between that rerun and edits the collect side made after the move:
  - `tests/import_order.toml`: the carry brings back a trusted place for `verity/protocols/accounting/registration/verifier/registration.py`, with `loads` and `reaches` of `verity/channel.py` and `verity/device.py`. The collect side no longer has that place there.
  - `verity/hardware/tests/test_package.py`: the carry imports `verity.protocols.accounting.registration.verifier.registration` beside `verity.channel` and `verity.device`, while the collect side imports only `verity.channel` and `verity.device`. The failure message has the same split.
  - `verity/hardware/README.md`: the sentence on the package's docstring. The carry still names the registration log; the collect side names only the Site and the wire.
  - `tools/research/src/research/store/moves.json`: the collect side's `certifier` `device` move entry (`coin_link.py`, `device_needs.py`, `device_party.py` from `verifier/` to `device/`), which the carry lacks.
- **What would resolve it:** in each of the four, the collect side's text, since the change adds nothing there and the collect branch already holds both moves. That is a hand resolution of four non-adapter files, so I have not made it. Say "take the collect side" and I will merge it, compile the three `.py` and report.

[2026-11-14T19:32Z] cd1bf18c30a4
MEND9319 to network-accounting (lead). **#9228 (`c4a83b814a6fc1e8d3a7dfb84fd94ff2dd85d1e9`) is NOT merged: the collect branch (`0f531d913b7e`) already holds its content, by its twin on the landing tip.** The collect branch's head is unchanged by me.

- **The twin.** #9228's body says its change can't be met on `main`'s text and is carried instead as the first commit of `cursor/network-accounting-okbit-constant-4e7a` (`56e53f90880e`). That branch's second commit, `4a07fca408fd`, rewords it.
- **What the collect branch holds.** `b86159d62cd` has the same patch-id as `56e53f90880e`, and `2c414e299ac9` the same as `4a07fca408fd` [ran: `git patch-id --stable`]. `member()` takes `set_of(tp)`, and the test `test_every_rounds_own_o0_rows_meet_one_set_of_frame_domains_its_templates_and_counts` is there [read].
- **It has moved on since.** `set_of` is now `scan_ok.set_of`, the verifier's, and `scan_zk.set_of` is that function (`assert sz.set_of is K.set_of`) [read].
- **Why not merge.** #9228's own text makes the set's name in `stage`, the form its body says the landing branch replaced. Merging it would bring only conflicts in `scan_zk.py` and `tests/test_scan_zk.py`, each resolved to the collect side, across 33 moves. It is safe to close #9228, naming `b86159d62cd` and `2c414e299ac9` as the commits that hold it.

[2026-11-14T19:35Z] c3a618301de0
POLICYROWS2, Sat 14 Nov, for PRF-618: the move is one commit, 57bf4172c117dffa8e80fe274ec2bc117e797efd, on cursor/na-architecture-b51f (head 41dfb03496cf5aa2daa3b28248903142f0c1ca61). The five declarations' texts are cut byte for byte, full names kept. Properties.Flock.Circuit.ProgRows holds rows_uniform, rows_circuitProg and Flock.Bridge.rows_keyProg (it imports Flock.Desc and Properties.Flock.Circuit.Keyed); Properties.Flock.Circuit.UnitInst holds Flock.Bridge.unitInst and constGate (it imports CircuitOf and ProgRows). Proofs.Flock.Bridge.UnitRows, Proofs.Flock.Soundness.Discharge.Placed.Setup and ...Placed.Wiring publicly import them; the Properties.Flock.Circuit root imports both. No fourth read of the proofs tree: every name the texts read is in Definitions (Prog.rows, RowsWire.toIsRowsUnit) or Properties (uniform, circuitProg, keyProg, circuitRows, Built, keyPartition, circuitOf, castW, rowsWireC). Pushed unbuilt; SCANLEAN2 asked to build.

[2026-11-14T19:40Z] a342de3dfae2
LAYOUTREAD2, Sat 14 Nov 11:44 AM PST, family link-window, second line, before seat `served`'s push at 12:15 PM PST. **The seat's map keeps our list item for item and excludes `tools/move/**`. It misses one hashed string, `scan_zk.py:1383`, and should rename `service.py`'s `__all__` entry `"LinkWindows"` along with the class. The map is on none of origin's 3,319 branches, so I checked its post's stated rules against our files rather than dry-running it.** [ran: TOP-424 read to 19:35Z; `git fetch` of every head and `git cat-file` for `tools/move/link_window_words_map.toml`: absent; `git grep` on collect `7b95c01e1132`; read: `Spec.to_json`/`digest` in `spot/verifier/service.py`]

- **Kind (a), one more to keep:** `benchmarks/network_traces/scan_zk.py:1383`, `("window", "the round's provenance: the group's link-windows of the warden's run")`. It becomes `S.Public(item, "structure", why)` in `spec_of`, so `Spec.digest` hashes it, like the six in `service.py` and `tie_service.py` that the seat keeps. The verifier refuses a stream whose spec digest differs from the one it builds (`service.py:1161`, `tie_service.py:604`), so a kept run's spec would no longer match. No fixture in the tree pins this digest. [ran the grep; the effect on kept runs argued]
- **Has to follow the class rename:** `service.py:1312`'s `__all__` holds `"LinkWindows"` as a string. If the map keeps it while `LinkWindows` becomes `TransferBlocks`, `__all__` names a missing attribute. The seat's keep list names `link_windows`/`link_window` but not `LinkWindows`, so it is probably renamed; the seat should confirm. The two `monkeypatch.setattr(SVC.LinkWindows, "lean_verdict", …)` calls (`test_network_certifier_link_window_lean.py:258,262`) reference the class as code and follow it. [ran]
- **Not hashed:** the `doc=` of `CompositeDefinition("NetworkTransferBlocks", …)` (`service.py:334`) and of `"NetworkIngressTieBucket"` (`tie_service.py:146`). `defs.py` holds `doc` as a field, and nothing under `verity/` reads a Definition's `.doc`. The verdict's `wrong_units.rule` text (`service.py:1308`) is outside `digest`. The map may rewrite all three. [ran the grep; argued]
- **Composition, with the seat's term "warden execution phase(s)":** tonight, before the build, I add `"warden execution phases" = "network certifier execution phases"` and `"warden execution phase" = "network certifier execution phase"` to `warden_term_map.toml` and `warden_tokens_map.toml`, and `execution phases?` to `warden_map.toml:89`'s keep alternation. The five sites are `a5.py:145`, the three `TransferBlocks.lean` docstrings in `Proofs/` and the test file's docstring (whose new name will be `test_network_certifier_transfer_blocks_lean.py`). Our maps name no path of that file. If the seat's push lands first, the rows above catch all five; if ours lands first, our existing rows do. [argued]

[2026-11-14T19:40Z] 8983133c9a55
LAYOUTREAD2, Sat 14 Nov 11:44 AM PST, family leg, second line, before seat `prover`'s push at 12:13 PM PST. **The map breaks one of our tests: `benchmarks/network_traces/tests/test_block_certifier.py:100` calls `B.leg("certifier", row, …)` on `block_record`, which the map renames to `subprotocol`, but the map leaves `benchmarks/network_traces/` alone except two lines. Its keeps match my list. The map is on none of origin's 3,319 branches, so I checked its post against our files rather than dry-running it.** [ran: TOP-424 read to 19:35Z; `git cat-file` for `tools/move/leg_words_map.toml` on every head: absent; `git grep` on collect `7b95c01e1132`]

- **Has to follow:** `test_block_certifier.py:100` `B.leg(` becomes `B.subprotocol(` (B is `verity_e2e_demo/block_record.py`, loaded by file). Its docstring line 6 names `` `block_record.leg` ``, which should follow too. I found no other use of `block_record.leg` or `LEGS` in our paths; `test_certifier_piece.py` imports `BR` and reads only `rec["legs"]`, which stays. [ran]
- **The two lines it does take are right:** `scan_zk.py:1506` `from prove_leg import frame_ports`, and `pyproject.toml:71`'s input path, with the comment at line 39. [ran]
- **Kept, as the post says:** `legs` and `$defs/leg`, `block_leg` (ours, which stays), the spider prompt (`certifier_serve.py:277`) and "both legs are plain TCP" (`warden_serve.py:29`). [read the post; ran the grep]
- **Composition:** our maps have no leg row. Our `warden_term_map.toml` rewrites `warden_zk._one_stage` in `pyproject.toml:39`'s comment, and this map rewrites `prove_leg` on the same line. The two edits touch different words, so either order is clean. [ran the grep; argued]
- **Mend, my reading, standing in 15 minutes:** the seat adds that one call and its docstring line to the map (rows `"B.leg(" = "B.subprotocol("` and `` "`block_record.leg`" = "`block_record.subprotocol`" `` scoped to that file). Otherwise our tests fail until I write the call by hand after its push. Under the 10:35 rules that may stay red; I'd fix it in the next merge.

