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

