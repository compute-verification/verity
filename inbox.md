[2026-11-14T20:25Z] 2ec790ebf51b
SCANLEAN2 to network-accounting (lead), Sat 14 Nov 12:25 PM PST. **Run 3 (`r20261114-195116-1b8f`, collect tip `4e301b7f46e2e2708d6db71a143d2878b9f6e930`) built every target with no error. Run 4 (`r20261114-202435-9c76`) is building tip `343ec72678d84aa7ced309c872b5bdecfbee8226` on research CPU machine 14 and should end about 12:55 PM PST.** Machine 3 refused run 4 for memory.
- **Run 3, per module, every one built:**
  - `network_check` (all seven targets) and the policy package's whole default build (509 jobs, `NciCheck.RecordText` and `inference-only-rows` among them);
  - `Definitions.NetworkCertifier.ResponseTie` and `Definitions.Accounting.Guarantees.LinkEnds`;
  - `Properties.NetworkPeriod`, `Properties.NetworkScan`, `Properties.Nci.Record`, `.Opening`, `.PinnedRead`, `.RecordForm`, `Properties.Spot`, and `Properties.Flock.Circuit` with `ProgRows` and `UnitInst`;
  - `Proofs.Properties.NetworkPeriod` (my mend `1cbca437ad3d` holds), `Proofs.Properties.NetworkScan`, every `Proofs.Properties.Nci` module, `Proofs.Properties.Spot`, `Proofs.Accounting.Actual.LinkEnds`, `Proofs.Flock.Bridge.UnitRows`, `Proofs.Flock.Soundness.Discharge.Placed.Setup` and `.Wiring`, and `Proofs.Flock.SpotFlock.UnitRows`.
- **No fourth read:** all 30 proofs-tree modules whose text names `constGate`, `unitInst`, `rows_circuitProg`, `rows_keyProg` or `rows_uniform` built. So no proofs-tree read of the moved gate is left without its import.
- **Nothing to dispatch and nothing mended:** run 3 had no error. Proofs' `architecture` break (`44a32989bec`) is not on the collect branch.
- **For soundness, NET-430's Lean:** at collect tip `4e301b7f46e2` (build run `r20261114-195116-1b8f`, no error), these build: the pinned-read statements in `Properties.Nci.PinnedRead` (`PinnedReadAt`, `PinnedRead`, `ReturnsRow`, `PinnedReadBinds`), their witness shapes `PinnedReadHonest`, `PinnedReadNontrivial` and `OpenedRowReturned`, the proofs in `Proofs.Properties.Nci.Lemmas.Opening`, `NciCheck.RecordText` with its `inputsOf` refusal, and `RowsOkHeldAt` in `Properties.NetworkPeriod` with its proof in `Proofs.Properties.NetworkPeriod`. Built means Lake elaborated each module. The statements that are `def … : Prop` are stated, not proved. No Lean check (axioms, records) ran. `RowsOkHeldAt`'s exception comes with #10103, which is not on the branch.
- **Run 4 builds what came since:**
  - CLOCKGONE's deletion of the clock form (`Period.lean`, `Clocks.lean`, `RecordsMain`, and `Definitions`/`Proofs.NetworkCertifier`, now built as `Proofs.NetworkCertifier` and `Definitions.NetworkCertifier`);
  - DEVICEKEYS2's `1697f7f1dd4` (`entriesOrderRefusal`, `entriesKeyRefusal`, the new `#guard`, the rewritten section of `Properties.NetworkPeriod`);
  - `core_world`'s deletion (no lakefile or import names it);
  - the certifier-programs layout move;
  - `#9589`'s merge.

  DEVICEKEYS2's later work and WINDOWREAD's merge go in run 5 if they land after `343ec72678d8`.
- **Run 4's order:** our packages first (`network_check`, `nci_check`, our `Properties`, our `Proofs` with `Proofs.NetworkCertifier`, then `Definitions`), and last the modules above SPOT (`Properties.Spot`, `Properties.Flock.Circuit.*`, `Proofs.Properties.Spot`, the Flock bridges and the 30 gate readers). Each is its own `lake build`, so a break above SPOT can't hide ours. The script names the renamed spellings too (`Certifier`, `InferenceOnly`) and builds whichever exists.
- **How long:** run 3 took 26 min of work (proofs 19 min) from source, Mathlib included.

[2026-11-14T20:27Z] 2402c4f17f5f
PERIODPROOF2: #10103's head d6430eb02f335a0f2be12da7d6c5b126ae9a9bca is in the collect branch; its head after my push is 4163f90b8ff1c88cef0a54ed85862907b6cbcdb1 (fetched back: equal) [ran].

- Path: two plain merges of the collect branch, at 3e950d3 and then 073c78f3d59e (the clock-form deletion), then `tools/move/restack.py` across e97a534ec829 (`certifier_programs_map`). It came back `clean`; a plain merge would have conflicted in `scan_zk.py`, its test and `scan_cost/README.md` [ran].
- Resolved by hand:
  - Lean: `NetworkCheck/Period.lean`, `Period/Start.lean`, `ConnectionRowsGuards.lean`, `NetworkCheck.lean`, `security/{Properties,Proofs/Properties}/NetworkPeriod.lean`, `security/Properties/NetworkScan.lean`.
  - Python and data: `network/tests/period_case.py`, `test_network_period_protocol.py`, the certifier's `verifier/README.md`, `warden.py`, `benchmarks/network_traces/{README.md,chain_record.py,scan_period.py,scan_zk.py}` and its tests `test_chain_record`, `test_leak_bounds` and `test_scan_zk`, and `scan_cost/public.json`.
- Compiles: `lake build NetworkCheck NetworkCheck.ConnectionRowsGuards` passes, 126 jobs, before the restack; the restack touched no Lean of mine [ran]. `py_compile` passes on the 86 `.py` files changed [ran].
- Does not compile, or not known: `security/` is pushed unbuilt (rule 3). Not run: suites, Lean check, records.
- Statements changed. Before, then after:
  - My start check `startCheck`/`startChecks` is renamed `scanStartCheck`/`scanStartChecks`, with `_getElem?`, `_none` and `_of_accept` renamed to match. The collect branch's own `startChecks` (signed session starts) took the name first. The check list is `startChecks ++ ws ++ [oneDelta, rowsWfCheck, connectionRowsCheck] ++ as ++ wired ++ (state ++ tiedInput) ++ scanStartChecks ++ ok ++ held`.
  - `StartCharged`: was `match p.form with | .rows _ _ scan => if p.first = 0 then 0 else scan.start | .clock _ => 0`; now the `.rows` arm alone. `scanStartChecks` and `connectionRowsCheck` likewise lose their `.clock` arms. These follow the clock-form deletion.
  - `RowsStartCharged`'s honest and acceptAll lemmas take the 3-field `Evidence` `⟨#[], #[], #[]⟩`.
- Question, yes/no: should `scanStartCheck` charge a later execution phase of a period instead of refusing it (`start-carried`)? I take NO. A period stays one execution phase, and I change no expectation without your word. As it stands, the collect branch's multi-phase cases `rows-two-stretches`, `rows-restart`, `undecided-and-reject` and `delta` expect ACCEPT, meet `start-carried`, and will fail in the period suite [argued, not run].
- `period_case.build` now picks `first` before the listed entries. `start-charged` and a later carry raise it to at least 1, and `start-unpinned` and the "more" carry hold it at 0. My first merge set it after the entries [read].
- ONSITEMAIN2's question: NO. None of #9350, #8829, #9641 or #10103 holds `PeriodLeakRegistered`/`periodLeakRegistered`, `periodLeak_registered`/`PeriodLeakRegisteredWf` or `RowsStateHeld`/`rowsStateHeld` under any name [ran: `git grep` on all four heads]. The nearest is #9350's `NetTiming.Guarantees.PooledLeakRegistered` (proved `NetTiming.pooledLeakRegistered`, over `PhaseRegistered`), which is pooled and not on the rows-form acceptance. So (1) comes as a statement only.
- Runs, already cancelled: r20261114-171142-2057, r20261114-174333-79c6, r20261114-170028-cfc3, r20261114-165220-2414. Done rc 0: r20261114-180132-a756 (#8829's records) and r20261114-174638-1a89 (suites at b8c7a6ee); their outputs were not fetched.
- Next: #8829, #9350 and #9641 into the collect branch, one line each. #9641 has `check_clock`, which the clock-form deletion removes.

[2026-11-14T20:28Z] e0037f30010f
MODEL: Claude Opus 5.5
GATERELAY2 (Worker 3: the cases and what stays), 12:27 PM PST. Pushed to cursor/na-architecture-b51f.
- Head: 178cc3e16145. The collect branch after the push: 178cc3e16145 [ran].
- My base: a38521777993, which holds the Layout-move e97a534ec829 [ran].
- I merged the branch's tip 4163f90b8ff once. It was clean, with no file resolved by hand [ran]. There was no new move to cross.
- Everything touched passes py_compile, and the `.sh` drivers pass `bash -n` [ran]. No Lean and no statement changed.

The four cases. None imports benchmarks/ or another distribution now [ran: test_boundaries and test_integration_boundaries list none of the four modules].
- scan_period: it read nothing from benchmarks/. I deleted the three `sys.path` lines and the six `noqa: E402` waivers they left unused.
- block_certifier: `OWNER` read benchmarks/network_traces/site.json. It is gone, and `block_leg(..., owner=None)` takes the owner from its caller. `chain_record` no longer calls it, since CLOCKGONE's change is in. test_block_certifier passes `owner`.
- window_chain: it read `verity_vllm.pipeline.window_advice`. That is inverted through the `typing.Protocol` `WindowAdvice`, so `join(windows, release, WA)` takes the advice module as an argument. network_e2e/chain.py:300 and its test_chain pass the vLLM adapter's module.
- stop_verdicts: it read `leak_bounds`' `RUN_NOT_PROVED`, `Stop` and `NO_RECORDS`. These moved into stop_verdicts, and leak_bounds binds the same names to them, so every `LB.Stop` and `LB.NO_RECORDS` still holds.

What stays imports moved modules by package name. In benchmarks/network_traces/:
- chain_record.py:713 and :425 (`kind_of`, which arrived with the merge);
- deviations.py:127;
- compile_slice_zk.py:100, :714 and :817;
- tests/test_certifier_piece.py, test_leak_bounds.py, test_served_session.py, test_chain_record.py (the unused n2_zk load dropped), test_warden_chain.py, test_capture_flows.py and test_negatives.py. The moved tests' helpers come from `verity.protocols.accounting.certifier.{device,prover}.tests.test_*`.

The drivers run `python -m verity.protocols.accounting.certifier.{device,prover}.X`: capture_loopback, chain_record, egress_zk, n2_tie_zk, n2_zk, scan_period, scan_zk, warden_zk (line 33's broken `warden_serve.py rehearse` included) and wipe_gate.

The anchors:
- network_e2e/chain.py: dropped its `sys.path` to network_traces, which it no longer needs.
- network_e2e/traces/corpus.py's docstring now says why that path is still on: `warden_serve`'s own bare imports.
- verity_vllm/program/predict/scan_state.py:27: the comment names `verity.protocols.accounting.certifier.prover.scan_zk.W63`.
- certifier_term/run.py, one_stage test_served_verdicts and scan_cost test_scan_cost name only files that stay. They need nothing.

For the close's lists, from my group. Grow none by hand.
- BENCHMARKS_IMPORTERS (tests/test_boundaries.py:130):
  - verity/protocols/accounting/certifier/prover/tests/test_block_certifier.py:80 loads benchmarks/one_stage/interfaces/verity_e2e_demo/block_record.py by path, with its schema and its period-0 fixture (E2E at :24). The test holds the leg to the block record.
  - .../prover/tests/test_stop_verdicts.py:26 puts benchmarks/network_traces on sys.path for `import leak_bounds as LB` at :28.
- KNOWN (tests/test_integration_boundaries.py:35): .../prover/tests/test_window_chain.py:10 imports `verity_vllm.pipeline` (the test hands `WA` to `join`).
- Either list's entry goes if those three tests move back to benchmarks/network_traces/tests, which I'd propose for a later map.

Not mine, found on the way:
- prover/scan_zk.py:1331 and :1334 use `RG` with no import (ruff F821). It is already so on a38521777993.
- prover/tests/test_scan_zk.py:44 loads benchmarks/network_traces/scan_zk.py by its old path.
- one_stage/tests/test_a5_warden.py:32 does `import test_warden_serve` bare.
- device/tests/test_connection_rows_run.py:32 does `import test_stop_verdicts` bare.
- I edited only the lines of prover/tests/test_block_certifier.py and test_window_chain.py that my API change needed: `owner=`, `WA`, and their head's `sys.path` lines. TERMGB2 can skip those.
- For SUITESPLIT: the prover suite runs lake and needs flock-verify, so it belongs in suites.py's `AFTER_LEAN`. network_traces/pyproject.toml's untraced `test_leak_guard`, `test_scan_zk_challenger` and `test_scan_zk_lean_one`, its `test_block_certifier` reads entry and its flock-verify artifact now name moved tests. I have drafts and wrote none of them.

Tests, by file [ran, locally]:
- Pass: prover test_block_certifier (2) and test_window_chain (4).
- network_e2e test_chain: 8 pass and 1 fails. That one is `..._carry_once`: its fixture endpoint_carry/advice_trace.jsonl is an art: id not fetched here, so it never reaches `join`.
- test_stop_verdicts, test_leak_bounds and device test_certifier_term pass 65 only with device/ and prover/ on PYTHONPATH. Without that they stop at collection on `certifier_key` and `leak_scan`, the moved programs' own bare imports.
- The staying tests above, with only the root and network_traces on the path, stop at collection on `certifier_key` (3), `leak_scan` (1) and `test_stop_verdicts` (1). With the moved directories on the path, only test_certifier_piece and test_warden_chain still fail, on test_scan_zk.py:44's old path.
- tests/test_boundaries is red on the place listing, regenerated at the close.
TIMER: none

[2026-11-14T20:30Z] b5955b67e143
NETSEATS2, push 1 of this turn: LOAD leaves the certifier manifest. The commit is 5d478462cec2f698c08d2ae35c900e245b1907ac. The collect head after the push is 6f26856db0c87c5766ac75a4d91afce459575846. [ran]

The machine was new: uptime 0 min, and the worktree was gone. Nothing was lost, since 073c78f3d59e was pushed. [ran]

Readers of the LOAD message name [read]:
- device_party.py: the one sender, in `SENDS` and in `hand_out_run`, through `Opened.load`. Its `open_session` raises NotImplementedError, so nothing reaches that send. It decides nothing.
- The manifest's two tests, test_network_certifier_parties.py and test_network_certifier_device_party.py. Also MAX_LOAD, which only the bounds test reads.
- demos/onsite_served/config.toml lists the channel's messages. I regenerated it with its generator, `served_table.configuration` and `K.render`, at its own commit; one line changed.
- A pin over the configuration's text: test_run.py's `UNCAPPED_ONSITE`, the onsite configuration's digest, which I recomputed with `K.digest` over the test's own `config()`.
- No reader in device_needs.py.
- ROWSPROTO2's `ROWS_MANIFEST` (inference_only/verifier/parties.py) has no channels.
- No Lean mirrors the manifest: `NetworkCheck.Certificate.parametersManifest` is the schedule's parameters. Nothing for SCANLEAN2.
- main has no LOAD.
- No reader decides on the load, so I did not stop.

Statements, before and after [read: the diff]:
- parties.py: the docstring's table line and the bullet for the `load` message go. So do `LOAD`, `MAX_LOAD`, its Schema and both names in `__all__`. The channel to the verifier is now out-record, certificate, registration, deployment, groups.
- device_party.py: "hands out the run's deployment and load" becomes "the run's deployment". `Opened` is the deployment alone. `hand_out_run` sends DEPLOYMENT only.
- test_run.py's comment on `UNCAPPED_ONSITE`: "the onsite configuration's hash before the network certifier's response cap and its lengths rows' form were parameters" becomes "with neither the network certifier's response cap nor its lengths rows' form set". The old wording is no longer true, because the hash is not the old one. The value is 1cd21b01…e17a.
- `test_the_certifiers_manifest_declares_exactly_the_files_a_warden_run_publishes` passes unchanged. [ran]

Crossing the move: STAMPS2's `Move by tools/move/certifier_programs_map.toml` (e97a534ec82) came onto the collect branch while I worked. I crossed it with `restack.py --move-commit e97a534ec82 --onto origin/cursor/na-architecture-b51f`, which came out clean (161a10f4b214), then fast-forwarded to it. A plain merge of the collect head after that was clean. I resolved no files by hand. Everything compiles. [ran]

Tests [ran]:
- The manifest's two tests, test_auditor_warden, test_onsite_served, test_run and test_inference_only_parties: 346 passed and 1 failed. The failure is `test_the_served_window_runs_once_with_every_piece_s_hooks`, which fails at the base too: its key variables H1_KEY_SHA512 and WIPE_KEY_SHA512 come back redacted in this environment.
- After the last merge, the manifest's tests, test_auditor_warden and test_run: 306 passed.

TIMER: none

[2026-11-14T20:33Z] 0d30886c5d05
LAYOUTREAD2, Sat 14 Nov 12:42 PM PST, two maps dry-run ahead of their pushes, from the seats' branches: `law` (`law_words_map.toml` at `77298074e05`, push 1:10 PM PST) and `sampled-proofs` (`sampled_proofs_words_map.toml` at `f842c51a014`, push 1:30 PM PST). **Neither breaks anything of ours, and the integrator may carry across both as they are. law touches none of our files. sampled-proofs touches 23 lines in 16 of ours: prose, and seven calls of SPOT's Lake executable that follow its lakefile. Every kept byte survives.** [ran: each map run for real with `rename.py` on a scratch worktree of collect `6f26856db0c8`, our paths and the certifier's Lean in `security/` diffed, residuals grepped; the maps as on those branches, which may still change before the push]

- **law:** 36 files changed tree-wide, none in `verity/protocols/accounting/certifier/`, `benchmarks/network_traces/`, `benchmarks/network_e2e/`, `benchmarks/certifier_*/`, the certifier's Lean or our `tools/move/warden_*map.toml`. [ran]
- **sampled-proofs, sites in ours:** prose in `certifier/verifier/PROTOCOL.md` (3), `advice.py`, `cache_step.py`, `protocol/advice-v0.md`, `NetworkCheck/Period.lean`'s module doc (2), `security/Definitions/NetworkCertifier/Guarantees.lean` and `SampledAudit.lean` (docstrings), `network_e2e/encodings/pyproject.toml` (a comment), and the comments of four `.sh` scripts. It also changes `lake build sampled-proofs-verifier` to `lake build spot-verifier` in `benchmarks/network_traces/{chain_record,compile_slice_zk,egress_zk,n2_tie_zk,n2_zk,scan_zk,warden_zk}.sh`. [ran]
- **Lake executables:** the six it renames are all SPOT's, defined in `verity/protocols/proofs/spot/verifier/lean/`; none is ours. Our only callers are those seven scripts, and the map rewrites all seven. No `sampled-proofs-*` executable name is left in our paths. [ran]
- **Kept bytes, checked after the run:** the format ids `verity/sampled-proofs/one-stage/v1` (`Period.lean:354`'s `oneStage`, `test_certifier_piece.py:213`, `certifier/prover/tests/test_block_certifier.py:71`) and `…/two-stage/v1` (`network/tests/period_case.py:2539`); the flag `--sampled-proofs` (declared at `served_verdicts.py:2224` and passed at `test_certifier_piece.py:226`, both unchanged); the log name `build-sampled-proofs.log`. None of our hashed reasons, record keys or difftest texts holds the word. [ran]
- **link-window, leg, firewall:** not on `architecture` or any branch yet. I dry-run each when it lands.

[2026-11-14T20:35Z] 11db14c02990
POLICYROWS2 to network-accounting (lead), Sat 14 Nov 12:40 PM PST, for NET-430.

**Item 1: outcome (c). The check does not register the root gates' constants from the pin, and registers no constant of any gate.** I stopped and changed nothing. No statement was added or changed, there is no commit, and the collect head is still 6f26856db0c87c5766ac75a4d91afce459575846.

Where the pin goes (`verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/Record.lean`):
- The pin is not a root. `pinOf` (line 570) is the closure digest of the reader template's Definition (`DefinitionDigest.closureDigest`), where that digest is among the instance's `pins`. It is never compared with any root, domain id or gate.
- `rolesOf` (line 606) uses it to pick the pinned readers and hold them to the layout: an opening's key, ports and verdict are the layout's, and a table read sits in the layout's root row (input row 1) or domain-id row (input row 2) (lines 614–619).
- `openingHeld` (line 635) uses it as `pinned`.
- `Case.heldAt` (lines 707–716) uses it as the `pin` of `.opened name param pin`.

What the check does with the root rows:
- `openingHeld` (lines 635–637) asks three things. The opening's reader is pinned. Its table's `param` is θ. A table read by the same reader covers the opening's units, once of the table's root and once of its domain id (`covered`, line 624). A table read (`RecordText.TableRead`, RecordText.lean line 178) carries only reader, port, table, `root : Bool`, start and units, so it holds no root value.
- θ is `isTheta` (line 559). The log's first record registers a value under the parameter's name, its `root` equals the run's record's `window.registered_roots[param]`, and its dimensions match the leaves. That decides whether the root row is classed θ. It puts no value on the row's gates.
- The check says so itself: "it holds no row to a value, an opening's verdict among them" (line 193; again at lines 630–632). Nothing under `NciCheck/` produces registered constants.
- In `PinnedReadAt`, `consts` is a free parameter, held only to `consts (constGate …) = some true` and to "registered constants hold in `Yc`".

So soundness is right that nothing ties `q.rootOf Yc` to anything the check fixes. One correction to its proposed shape: the half that is missing is not "`consts` at the root gates are the pin's bytes", because the pin is a Definition's digest. It would tie the root gates' value to the registered root of the table's θ parameter (`window.registered_roots[param]`, the same root the log's first record registers). The check decides only that the row is declared to hold that root. Whatever binds a θ table-read row's gates to it is outside `NciCheck`. That is your and soundness's ruling to make.

**Item 2: soundness's reading is necessary but not sufficient, so I propose a restatement and changed nothing.**
- `CollisionIn H xs` (`security/Definitions/Core/Hash.lean:43`) is a collision between any two different messages of `xs`, including two within one half.
- With the witness soundness means (the unit's opening equals `table`'s opening at the index), each list is L ++ L. That removes the collisions between the halves. `¬ CollisionIn (L ++ L)` is still `¬ CollisionIn L`.
- The outer L, `RowLeaf.outerHashed` (`security/Definitions/Hash/RowLeaf.lean:124`), holds at least three different messages: `saltPrefix ++ salt` and `leafPrefix ++ commitString …` (`Hm96.hashed`, `Hm96.lean:86`), then the frame leaf's `leafMsg`, then the path's nodes.
- So `¬ CollisionIn L` still needs those messages' SHA-512 digests (`Verity.sha512` = `Flock.Sha512.hash`) to differ pairwise. That is evaluating SHA-512, which the kernel won't do. Even by that witness, `PinnedReadNontrivial` as written is not provable.

The smallest restatement: replace the last conjunct `¬ ReturnsRow ct hB n u q Yc table` with "the unit opens the table's own opening at the index, and returns another row":

    q.opening ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) =
        table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)) ∧
      q.hOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) ≠
        Hash.EmbeddingRead.h q.count q.words (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
          (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)))

With this, `ReturnsRow`'s first disjunct fails. Its two collision disjuncts reduce to collisions among one opening's own hashed inputs: SHA-512 on the table's fixed data, which nothing the unit returns produces. So the verdict bit still does the work, and no SHA-512 evaluation is needed. The docstring would say exactly that. I'll make the change once you or soundness say yes.

