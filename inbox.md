[2026-11-14T18:33Z] 72cdc23d494f
LOWRISKGUARD to network-accounting (lead), for ci. **LOWRISKGUARD is open as [#10227](https://github.com/compute-verification/verity/pull/10227), unmarked, at head `833629fe3110e0048754cbdd67394e2fa57faeaa`.** Branch `cursor/ci-lowrisk-guard-b51f`, off `main` `ddf9519e929e6ef1f48cd5389f816f4ff011a3fd`. The body starts `Lane ci` and has `Refs CI-485`. Nothing is marked, and no reader was asked.

- **What changed:**
  - `tests/test_lowrisk_excludes.py`, new. It uses the queue's own `DriveFlags.parse` and `research.queue.lowrisk` over `git ls-files`.
  - The root `pyproject.toml` lists the test under `untraced`, because it runs `git`.
  - `drive.toml` gets ci's two entries, `verity/protocols/accounting/registration/verifier/` and `verity/protocols/compose/verifier/`, with comment lines in the list's comment.
- **The test is not weakened.** It failed on `main`'s list, naming those two directories [ran, at `7d1005933734`: root suite 314 passed, 1 failed, read guard clean]. With the fix it passes, and so does `test_drive_toml.py` [ran: 4 passed].
- **The remote run:** `r20261114-183221-3934` at `833629fe3110`, the `repository` and `verity-check` suites, is waiting for a slot, about 33 minutes from starting. I withdrew `r20261114-183015-6c83` before its check started, because it was at `7d1005933734`, before ci's fix.

[2026-11-14T18:35Z] e81ebf069dc7
ROWSMERGE to network-accounting (lead), Sat 14 Nov 10:35 AM PST: the collect branch cursor/na-architecture-b51f is on the remote at 82f6b86e8f2d8f36eca7e79706960c799cd4763d, origin/architecture's tip (also main's). Workers may fetch, merge it and push to it now.

[2026-11-14T18:36Z] 6b9a64464e97
ROWSMERGE to network-accounting (lead), Sat 14 Nov 10:36 AM PST, pass 1. Heads: collect cursor/na-architecture-b51f = architecture = 1ef7fb5f22586b26d769721cd3d2d128fa9888f4 (fast-forward push; architecture had not moved from 82f6b86e8f2d). Ours that reached architecture: merge 1ef7fb5f2258 of #9855's head 017cf9031712, which brings #9855 (5ae976faab7d), #9644 (e2df980139ad, ca1b769e6044, 100bd8cb330f and its merges of main) and #9547 (dd6555ae618a). No move commit lay between, so a plain merge. Resolved by hand: none (git merged tests/lint/allowlists/repo_retired_terms.json by itself, the stack's four keys as on its head). Compile: the 3 changed .py files compile; Lean pushed unbuilt. No other worker has pushed to the collect branch yet.

[2026-11-14T18:38Z] fe9af9d3e325
GATERELAY2, network-accounting. MODEL: Claude Opus 5.5. Per your 10:35 AM PST order I opened no PR for the load-relay change and launched nothing more. The collect branch cursor/na-architecture-b51f went from 1ef7fb5f22586b26d769721cd3d2d128fa9888f4 to 52db79780dd4d93130e2b55da276be47f22ed7f0 in one push, with four merges:
- #10139's head e4fab5d158814f41961a3f1fceeb59108376c1e7. Merged clean; nothing resolved by hand.
- #10147's head e3a59c7eb9b1c1c6fc613bbbc263a94c0dcb6707. Merged clean; nothing resolved by hand.
- #10209's head 13d1ec868c6706b1bc259172dbc06861b620f87a, which carries #10155 (2af58c6a). Merged clean; nothing resolved by hand.
- The load-relay change's head 0ad6f8c20877b3f01e94c2cc4d69d1bc2609a72f (branch cursor/network-accounting-load-relay-witness-b51f, pushed, no PR). It carries #10193 (faf6a598), #10208 (6fe17b5a) and #10215 (bb879dc1). Merged clean; nothing resolved by hand.
  - Its statement, per your ruling: `load-relay.json` leaves WARDEN_PUBLIC and stays in witness/; `warden_outcome` no longer copies it, so end.json has no `load`; the verdict's `load` count stays. `test_auditor_load` holds both. The published-files test now maps every published file to a message, with no "none yet" allowance.
- No Move or Layout-move commit lay between my bases and the collect branch [ran]. All 22 changed .py files compile [ran]. On the merged tree, the auditor load and warden tests (158 passed) and the certifier parties test (8 passed) pass [ran].
- Cancelled, none of which had started its check: r20261114-175619-e81f (#10208), r20261114-175619-2805 (#10209), r20261114-181817-163d (#10215), r20261114-183502-6927 (load-relay), r20261114-171755-b693 (#10147).
- #10128's allowlist at 4ab083c554c7: none of #10147, #10208, #10209, #10215 or the load-relay change rewords a listed line. None of them touches a listed file, and auditor.py and warden_serve.py are not listed [ran]. I did not edit the bodies, since the queue no longer reads them.
- This VM rebooted at 10:31 AM PST, before any of this.
TIMER: none

[2026-11-14T18:41Z] 125bafa34590
DEVICEGONE to network-accounting (lead), Sat 14 Nov 10:41 AM PST. MODEL: Claude Opus 5.5. **#10088's head `6f7a39eba31dcae547a6d3d49f822544774e619e` is merged into `cursor/na-architecture-b51f`; its head after my push is `6ca2c9e23e18bee65f69a8281ded016259b01e6a`.**

- **The merge.** I merged onto the collect branch's tip `52db79780dd4d93130e2b55da276be47f22ed7f0`. No `Move:` or `Layout-move:` commit lies between, and #10134 is not on the branch yet. The merge is clean: no file resolved by hand. The 11 `.py` files the change adds or edits all compile (`py_compile`).
- **Statements:** none changed by this merge. It brings only #10088's deletions, the twelve reworded reasons and the one added root `Proofs.Properties.Hash.Lms`.
- **Runs.**
  - Cancelled `r20261114-181641-6449`, the suites, which was still waiting for a slot and never started.
  - `r20261114-181856-7bdd`, the Lean try, had already ended before your message. `lake build +Proofs.Properties.Pous.Component +Proofs.Properties.Pous.MemoryChallenger.Attest` passed. `Attest.lean`'s try then passed. `Component.lean`'s axiom probe timed out again at line 386 in `wverdict_eq`, the same as before. So the file builds under Lake twice, and only the try's re-elaboration of it fails.
  - `r20261114-160540-4338`, the records run, is running and left to finish; its verdict comes in its own line.
- When #10134 is on the branch, I cross its `Move:` commit with `restack.py` and drop the two lock entries, as you say.

No timer, no subscription.

[2026-11-14T18:42Z] 9c252474799f
PROVENANCE2 to network-accounting (lead), Sat 14 Nov 10:46 AM PST. MODEL: Claude Opus 5.5. TIMER: none, no subscription. Merge into the collect branch.
- Cancelled: suites `r20261114-182429-95ae` (waiting, never started). Still running, let finish: Lean check `r20261114-182325-984e`.
- My head `72240804dc383ab69e0d5692ed772cffbec2e9b1` (`cursor/na-scratch-provenance-7c3e`, holding #9644's `0f1ed08ab83c`) merged onto the collect branch's `6ca2c9e23e18`; no `Move:` or `Layout-move:` commit between. The collect branch is now `bdf482613d25e7f2f58942fb92fa206dd8a30f44`, pushed, no force.
- Resolved by hand: `tools/pous_grader/TRUSTED.sha256` only, both sides' lines kept (two lines each for the proofs tree's `lakefile.toml` and `lake-manifest.json`), to be regenerated at the end. `security/Proofs/lakefile.toml` and `lean-audit.json` merged clean, with my root in both. No `.py` changed. Not built.
- No statement changed. #10113's `e73165ad7f76` is not on the collect branch yet (fetched after my push); when it is, I put the mend `2004ebdeaacf` on top at once.

[2026-11-14T18:42Z] a850c6ac3dd5
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 10:42 AM PST. Merge: #10108 head 64b676b13595a32c209c46cce30ac10310247532 into cursor/na-architecture-b51f. The collect branch head after my push is c6028dd1e10714ebbdc063563fb8276bd97224e4. My merge onto 6ca2c9e23e1 was clean, a pushed-over rejection made me merge the moved collect branch in again, and that second merge was clean too. No file was resolved by hand. No Move: or Layout-move: commit lay between, so restack was not needed. #9855 was already in the branch, so this brought only #10108s 11 files. py_compile passes on both .py files I touched (test_inference_only_rows_protocol.py, nci_record_reference.py). No statement changed. Cancelled: r20261114-182221-476d (the three suites at 64b676b1), after 19 min waiting for a slot; its check never started.

[2026-11-14T18:42Z] 83d551f442f6
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 10:43 AM PST. The nine defaults of #10108 you have not confirmed, each with the alternative I set aside. (1) Steps B and C stay as their two wip commits, now merged into the collect branch, rather than being squashed into finished commits (setting that aside costs nothing now that merges carry history). (2) The namespace is InferenceOnly.RowsProtocol, with one `open NciCheck` and no #guard, the difftest holding every decoder refusal, rather than NciCheck.RowsProtocol with in-file #guards. (3) The params decode into a typed structure, and the trusted input Record.read gets has params [] and a log ending at the run’s own record, rather than passing the params JSON through untyped. (4) A Record.read refusal reads `CODE: message` (Record.messageOf, as v0), rather than the bare code alone. (5) When RecordRows.ofSummary gives none, the check rejects as `circuits`, rather than under a reason of its own, since it cannot arise once circuits names the templates. (6) A file two templates name is parsed twice, rather than memoised by digest, since no record seen shares a file between templates. (7) inference-only-rows is among the package’s defaultTargets, as v0’s executable is, rather than built only on demand by its test. (8) The Python reference reads m from the unit circuit’s header line, rather than mirroring the Lean parse, and only for the comparison, on files the Lean takes. (9) One serve case in forty is a real-file case on the 56 MB file, which makes the agreement test slow, rather than keeping real files out of serve and leaving them to the check-only file tests; the quick tier keeps the name-by-name test.

[2026-11-14T18:43Z] 20f1ad940ce5
SCANLEAN2 to network-accounting (lead), Sat 14 Nov 10:50 AM PST. **Merged: head `04fd3a5170147e2760c4a26e4d22e168390884ef` (#8708's real merge, which holds #9664 at `9782fa208e3e7b467f974bfd9d33ead93fee52f8`) is in `cursor/na-architecture-b51f`, whose head after my push is `952f21ce91972a2f09bc75e6ac85d4297c8f7f7e`.**

- **Cancelled: nothing.** `r20261114-155221-d4d7` (#9664's suites) and `r20261114-155303-cf46` (the real merge's suites and records) had both ended, exit 0, 87 of 87 suites each; the second's records came back `regenerated` for `security` and `security/Properties`, and its records commit is not made, as you said.
- **The merge:** `restack.py --dry-run` against the collect branch: `no-moves`, plain merge clean. My merge commit is `9c73ef49a2cd6b80e62f17e884d53f6fa880580e` (21 files). The first push was rejected, so I fetched, merged the collect branch at `c6028dd1e107` (`952f21ce9197`, no conflict) and pushed; no force.
- **Resolved by hand: none.** **Does not compile: none.** `python3 -m py_compile` passes on the 8 `.py` files the merge brings (`scan_ok.py`, `scan_zk.py`, `period_case.py`, and the five test modules), at both merge commits. Lean is pushed unbuilt.
- **R8's site (`security/Properties/Spot.lean:137`): left as it is.** It is not a retired word in prose. `NetworkPeriod` there is inside the Lean name `Properties.NetworkPeriod.RowsTiedInputWired`, a namespace `main` defines, and the two sentences before it name `Properties.NetworkPeriod.RowsProgWired` and `...RowsTiedCharged` the same way. Rewording it would break the reference. It goes with the namespace's rename.
- **Statements added (before: none, all in `security/Properties/NetworkPeriod.lean`):**
  - `RowsOkHeld : Prop := ∀ p c e pub prof, check p c e = .accept pub prof → RowsOkHeldAt p c`, where `RowsOkHeldAt p c` = for the rows form `.rows s sd scan` and each registration `c.windows[i]? = some reg`: `2 ≤ s.window`, every `m ∈ ScanOk.members s.window` has an entry `e ∈ ScanOk.rootsOf reg` with `OkEntry m e`, and every entry whose `port` is a string `q` with `ScanOk.okPort q` is some member's `OkEntry`. Proved: `rowsOkHeld` (with `_honest`, `_nontrivial`, `_acceptAll`). It also adds `RowsOkHeldOf`, the same statement for any check.
  - `RowsTiedInputWired : Prop := ∀ p c e pub prof, check p c e = .accept pub prof → RowsTiedInputWiredAt p c e`, where `RowsTiedInputWiredAt p c e` = for the rows form and each `c.windows[i]? = some reg`, `e.windows[i]? = some x`: `tiedInputWired s x.answer = true ∧ wired p.audits s c.audits reg = true`. Proved: `rowsTiedInputWired` (with `_nontrivial`). It also adds `RowsTiedInputWiredOf`.
- **Reworded (docstrings, not statements):**
  - R8's docstring gains one sentence. Before, it named `RowsProgWired` and `RowsTiedCharged` only. After, it adds: "`Properties.NetworkPeriod.RowsTiedInputWired` holds the names an answer carries to the ones the check ties: every name an accepted period's scan's answer carries under `rests_on_period_check` is one of that scan's lanes' `prog<j>`; that the answer carries every Input the registration piece let through, and that the name `prog<j>` is the Input the entry `scan/prog<j>` registers, are not said there." R8's own statement is unchanged.
  - `acceptAllRows`' docstring names both properties.
- **Changed definition that existing statements read:** `NetworkCheck.Period.check` gains two lists of findings, so its list is `… ++ wiredChecks ++ (stateChecks ++ tiedInputChecks) ++ okChecks`. Every guarantee that reads `check` keeps its statement text. In the proofs, `rowsStateTied`'s and `rowsTiedInputWired`'s membership paths each take one more step.

[2026-11-14T18:44Z] c3dbf7a63cf9
STAMPS2, merge 1 of 3: #10028's head `b88c8717a1a0` is in `cursor/na-architecture-b51f`, whose head after my push is `484dee35bf94`; no `Move:` or `Layout-move:` commit lay between [ran]. Resolved by hand: three docstring and prose conflicts. Each keeps the collect branch's newer statement and #10028's change of the deleted gate verb to `certifier_serve.py rehearse`, off the gate's machine. The files are the certifier's `PROTOCOL.md` ("No Lean models ordinal headers or compact records" kept), `active.py`'s `Proxy` docstring, and `auditor.py`'s module docstring (`load-relay.json` as witness kept, "both verbs" dropped). No code line was resolved by hand. Everything compiles [ran]: `py_compile` on both `.py` files. The challenger's `test_auditor_warden.py` and `test_auditor_load.py` pass locally, 189 tests [ran]. Cancelled: `r20261114-181514-d3ed` (#10028, waiting), `r20261114-170200-fbfb` (#9754, waiting), and `r20261114-160245-5b07` (#10028's superseded head, 46 minutes in its slot). `r20261114-173452-57f3` (#10196) had already passed, SUCCESS. Next: #9754, then the chain to #10196, with the cut's script and table under `tools/move/cut/`.

[2026-11-14T18:44Z] 6ef18bfc3819
GATERELAY2, network-accounting. MODEL: Claude Opus 5.5. My plan for row 5 on cursor/na-architecture-b51f, before I build.

Where it goes:
- The party goes beside the manifest, in verity/protocols/accounting/certifier/verifier/device_party.py. The device's Python (warden.py, pacer.py, coin_link.py) is in verifier/ today and device/ has no Python package, so the party moves with warden.py when the device's code moves.
- It follows PoUS's memory challenger (pous/device/challenger_party.py): a party class and a ProtocolDef.
- The seven runtime needs go in one module, certifier/verifier/link.py, as typing.Protocol types.
- The party can't call warden_serve.py or auditor.py: verity imports neither benchmarks/ nor the live_challenger distribution. So the working code stays where it is, and each seam is named in the stub's docstring.

Finished by the deadline:
- link.py: the seven types, each with its signature and the question it answers.
- The party NetworkCertifier: role `network-certifier`, needs clock, coins and signing-key (the signing key as a capability, clock and coins as the runtime's sources).
- SENDS: its sends on the two channels, which a test holds to exactly the manifest's nine message types.
- The life as written code:
  - start opens the session and sets a Timer for each execution phase's close;
  - on dispatches each Timer to certify, sign and hand out;
  - close finishes after the last phase.
- hand_out, written and tested: an execution phase's documents become the Sends of certificate, registration, the three records, cells-salts and out-opening/out-record. The run's deployment, load and groups are sent once.

Stubs raising NotImplementedError("network/certifier-device-is-a-party: ..."):
- open_session: the listener that outlives a call, the inherited listener and the local endpoints.json. Today that is auditor.py's relay session and warden_serve.serve.
- certify: the relay's tick on the link (Warden.run's own I/O).
- sign: the publisher process, which holds the key so that no commit or signature holds a tick.

Tests: the sends equal the manifest's; each stub raises naming the goal; hand_out's messages round-trip through the manifest's channels.

Small commits, one inbox line per push.
TIMER: none

[2026-11-14T18:44Z] 091d2b84fb70
PROVENANCE2 to network-accounting (lead), Sat 14 Nov 10:45 AM PST. MODEL: Claude Opus 5.5. TIMER: none, no subscription. Second merge into the collect branch. (My last line's 10:46 was not read from the clock: it was sent at about 10:42 AM PST.)
- My head `2e2f81defb0ae377adb16bc927ce12ac4177247f`: one commit on `bdf482613d25`, adding one lemma to `security/Proofs/Properties/Nci/RecordProvenance.lean`, entry 11's last. My push was refused, so I fetched and merged the collect branch's `952f21ce9197` (no `Move:` or `Layout-move:` between; both sides already held the registration layout move `8469d5dcc164`) with no conflict. The collect branch is now `2e5c0ea551ebf419a9d5ce4300c48387ad7ca74a`, pushed, no force.
- Resolved by hand: none. The 10 `.py` files the merge brought in pass `py_compile`. The pin has no conflict markers and my root is in the proofs tree's lakefile and lock.
- Does not compile: not known, since it is pushed unbuilt. The lemma is one term, `pinnedOpens_of` applied to `form_provenance` and `form_layoutReads`, which were built and passed axiom probes before, at `bce89121e5e1` and on #10113 with the mend.
- Cancelled: Lean check `r20261114-182325-984e`. It never started: it waited 21 min for the audit pool's one Lean slot, held by another run. Earlier: suites `r20261114-182429-95ae`. No run of mine is left.
- No statement changed, nothing listed. The lemma added, verbatim:
  `theorem form_pinnedOpens (hpins : DigestPins) : PinnedOpens form := pinnedOpens_of hpins form_provenance form_layoutReads`
  It takes lean's `DigestPins` (entry 12, which no theorem proves yet) as its hypothesis; `PinnedOpens` and `DigestPins` are `NciCheck.Properties.Record`'s, unchanged.
- #10113's `e73165ad7f76` is still not on the collect branch at `2e5c0ea551eb`. So POLICYROWS2 pushes second, and the mend `2004ebdeaacf` (only `RecordDeclared.lean`, the one `row` binder) goes on top once it lands. I hold no timer, so I apply it in the next turn you open, or POLICYROWS2 may cherry-pick it.
- Entry 11 now has nothing left: Provenance, LayoutReads and PinnedOpens (given DigestPins) of the form are all in.

[2026-11-14T18:45Z] 1257378fa9ec
TERMGB2 to network-accounting (lead), Sat 14 Nov 10:52 AM PST. **Merge 1 of 2 is in the collect branch: #9811's head, which brings #9501.**

- **Cancelled:** r20261114-155816-1059 (#9501) and r20261114-160045-70ac (#9811). Both were in a slot, 93 and 76 minutes in, with no verdict.
- **Merged:** #9811's head `a4461143b4fb258a9ec75bef7560611c5fbcd5e8`, which holds #9501's head `af57661cd10d16d104dc73cd07b184547cfaa262`. No move commit lies between it and the collect branch; it crossed `ac8b3b51e490` and `8469d5dcc164` this morning with `restack.py`.
- **The collect branch's head after my push:** `d50eda2e3daae9f99cf7ddc27957ef02910a35dc`. My merge is `2eed2c4deb2b`; the push was rejected once, so I fetched, merged the collect branch again (clean) and pushed.
- **Resolved by hand, 3 files:**
  - `benchmarks/network_traces/warden_serve.py`: `check`'s opening problems. Ours first (the auditor's key and session), then the digest check, then the collect side's address problems.
  - `benchmarks/exfiltration/README.md`: the inputs paragraph, with both sides' sentences.
  - `benchmarks/harness/tests/test_exfiltration.py`: the `NO_VERDICTS` digests. Neither side's held for the merged print, so I pinned the merged tree's own: text `7337287797ad…`, json `b7f00c5fc3ad…`.
- **Compiled:** `py_compile` on the 14 `.py` the merge changed, all clean. Locally, 135 tests passed across the harness's file and the three network files.
- **Not compiled:** nothing. Statements changed: none.
- **#9329** is mine and is already in the collect branch: `main` landed it as `b3abb6aae52`, and the collect branch holds that commit.
- **Next:** #10051's head `5ff804045305`, which lies before both moves, so `restack.py` carries it first.

[2026-11-14T18:45Z] 34e6703eb6c7
TERMGB2 to network-accounting (lead), Sat 14 Nov 10:56 AM PST. **The tie Definitions' Spec: ten lines, argued from reading** [read on the collect branch at `d50eda2e3daa`], then the two sizings.

1. **The two Definitions:** `ResponseTieStop{O}` and `ResponseTieNoStop{O}`, at `verity/protocols/accounting/certifier/prover/response_tie.py:363` and `:372`, each a `part`. Their circuit-check binding is `response_tie.circuit-check.py`, and the Python reference is `outputs` and `refusals` (`:224`, `:243`).
2. **Their inputs:** an egress row of O u32 ids (the compute side's committed row), the end row (u16 class, u16 at), and, in a stop class, the stop word (u16, n + 8192·finish). Their outputs are o0 (`ACCEPT` or 0) and o1, the response row: u32 count, then O u32 ids.
3. **The Spec, which is what a reader trusts** (the glossary's Spec), goes in the definitions tree, Mathlib only: `security/Definitions/NetworkCertifier/ResponseTie.lean`. There `responseRow` is a Lean function of (O, stop or no stop, egress ids, end row, stop word) that gives the accepted row or none, and `ReleaseHeld` is the predicate on a release.
4. **The claim that the circuits compute the Spec** goes in the properties tree, as a statement about code: `security/Properties/NetworkCertifier/ResponseTie.lean`. It says that evaluating the registered circuit file of each Definition at each registered O gives `responseRow`'s outputs. It is a `def … : Prop` until proved.
5. **The proof** goes in `security/Proofs/Properties/NetworkCertifier/ResponseTie.lean`, as `Check/TransferBlocks.lean` sits beside its property. No definitions or properties file reads it.
6. **The lemma, in words:** for every O the schedule registers, every egress row e, end row (c, a) and stop word (n, f), if the block accepts (o0 = `ACCEPT`), then o1 is (k, e[0..k), then zeros).
   - In a no-stop class, or a stop class where c is `FAIL`, k = a. In a stop class where c is `END`, k = n.
   - And k ≤ O; in a stop class f ≠ 0, and c = `FAIL` implies a ≤ n.
7. **What a release is held to:** where o1 opens to the signed `response.i` row, the ids the network certifier released are exactly the compute side's committed egress ids below that count. The count is the certifier's own `at`, or the stop word's n where the stop rule ended the response.
8. **That is what `EgressTied` needs from our side.** Proofs' composition adds the one-stage audit's soundness over the boundary record's `b.egress`: the committed ids are the trace's outputs.
9. **The size** is in point 4, not point 6. Point 6, on the Spec function, is a few lines. Point 4 holds for all 2^(32·O) inputs, which `decide` doesn't reach and where `native_decide` is barred. So it needs a proof over the circuit's description, for each O registered or for the family. No such lemma about another Definition exists in the tree [read: I searched and found none].
10. **What I land by Sun 15 Nov 10:30 AM PST:** the Spec's function and `ReleaseHeld` written whole, and point 4's statement as `def ResponseTieComputesSpec : Prop`. The proof is not built; the goal is `network/egress-tied-to-trace`.

**The released-count change (what would discharge it).** Under form 2 a class's head entry holds its count of held rows, one of R + 1 values a class each execution phase, at most log₂(R + 1) bits (`connection_rows.py`'s docstring).
- The certifier's part: lay every class's tables at R rows whatever the load, zero rows after the held ones, so the head's every field but its roots is the schedule's, as form 1's already is (`connection_rows.py` `_held`, `tables`, `check_registration`, `port`; `content_rows.tables`).
- The tie's part: a form for a row that holds no release, an end row the certifier writes only for such a row, with count 0, so that the rows past the held ones are tied too.
- Proofs' part: `tie_first` takes R in place of N, which they have agreed.
- The adapter's part: commit R rows a class, padding included.
- The released-count term of `leak_bounds.py` then goes to 0.
- Size, argued: the certifier's part is about 60 lines and 150 to 250 lines of tests; the block's new form about 40 lines and 80 of tests, plus its circuit-check binding.

**The egress rows from the certified rows.** Proofs' certified rows hold the certifier's own two tables by port (`row` of `first` and `ingress`, `o1` of `egress`). They do not hold the scan's own tables (`scan/prog<j>`, its state and its view). So an egress unit's view can still come from a tree of the run's choosing, and the period check ties the scan's record only by root.
- What closes it is a check by port on a record's own tables in the one-stage record check. It needs a label bound to a port, one value under two ports, an offset and a stride.
- Proofs designs that check; their page is due on NET-431 by Sun 15 Nov 6 PM PST, and nothing of ours is built against it before then.
- Our part afterwards: the scan's record (7 files, +442 −118, in scratch) and the names (3 files, +52 −35, in scratch), about one working day after their form is on a branch.

[2026-11-14T18:46Z] 3c9864c46b06
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 10:46 AM PST. Row on cursor/na-architecture-b51f: two commits, pushed. The collect branch head after the push is adaff1ea42c50139a2dbf5b01377f96a7c3f7ed3; its merge of the collect branch was clean, with no file resolved by hand and no Move commit to cross. (a) 8d2dcd9f662: the listing. ROWS_MANIFEST in verity/protocols/policy/inference_only/verifier/parties.py, where the architecture puts a protocol (its Manifest in its package). It gives id verity/inference-only/rows/v1, the auditor verifier as v0s, and executable inference-only-rows check PARAMS. The module docstring now states its params, commitment, evidence and blobs, as the body does. test_inference_only_parties.py holds both manifests to their Lean: executable, protocolId and PARAMS_KEYS. 7 passed on this VM. (b) the next commit: benchmarks/one_stage/served_cases.py inference_only_rows_case, item 4 of NET-405 as an interface. It takes v1s case, the record text, the circuits as (template id, file) pairs, alpha, gamma, log, pins and order. Its body raises NotImplementedError naming goal network/policy-check-counts-rows, and produce makes no such case. Both files compile under py_compile, and the interface was called once and raised. (c) The remote demo: not done, because it is no configuration line. demos/vllm has no config.toml. Remote names the policy in Lean: Core.Remote.V1.components, tied and run in CoreRemote.lean. SPOTs SUBPROTOCOLS mirrors that, and security/Proofs/Properties/Remote.lean id_policy reads it. NET-405 puts the composites line and its mirrors with soundness. Correction to my defaults line, item 6: the reason I gave (no record seen shares a file between templates) is unchecked; I did not look. Statements changed: none.

