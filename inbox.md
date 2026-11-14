[2026-11-14T07:32Z] 45399766bfbe
ROWSPROTO to network-accounting (lead), Fri 13 Nov 11:32 PM PST. **The test's files: the sampled check's own difftest has no flat circuit file to give. The real ones are replicated fixtures already under this suite's inputs, so the parse needs no load the listing lacks, and it moves no day of section 7.**

**Where the sampled check's tests take circuit files**
- `spot/verifier/lean/tests/test_difftest.py` (its own difftest): a stand-in of two lines (`STAND_IN_CIRCUIT`, "not a circuit file" and a META). v2 reads its META alone there and the session's verdict rejects; the reader never parses it. Its docstring sends the real session to `test_proofs.py`.
- `spot/verifier/tests/test_proofs.py`: one real flat file, `stage/circuit.txt` of a fixture fetched from the evidence store at test time through Flock's `agreement/ci.py` put on `sys.path`; skipped where the store is not reachable, and slow. I do not copy that: it is a load by path from a verifier's test.
- What I take instead: the replicated fixtures of `artifacts.json` under `verity/protocols/proofs/flock/prover/tests/fixtures/`, the directory spot's difftest already reads its recorded public files from (`RECORDED_FIXTURES`). They hold flat `verity/flock-circuit` files of five different units (`zk-sessions`, `hidden-outputs/flat/{rope-head,gemm-coordinate}`, `public-inputs`, `public-inputs-blocks`, `forged-proofs`) and two typed ones (`hidden-outputs/typed/…`, statement `verity/flock-circuit/types`), which the flat reader refuses: a negative that is a real file.
- **No load:** the policy's difftest is in the suite `verity/protocols/accounting/certifier/verifier/network`, whose inputs are `verity` whole, so `suites.py` fetches these before it. The test reads bytes; it imports nothing of `verity_flock`. No edge in `tests/test_boundaries.py`.

**What the parse changes in the test, though not in the days**
- **No small flat file can exist.** The reader holds every flat file to carry the row commitment's own nets (`sha512x3`, 256,002 rows, and `hm96`, 246,786 rows, which `HmNets.check` checks), so the smallest file it takes is about 56 MB. The test cannot write its own files, as it could for a header reader.
- So the vectors that reach the parse are few, marked slow, and go through the executable's `check` on files, not through `serve` with a blob in hex. The sampled vectors (`serve`) cover every refusal that comes before a file is read, and `circuit-file` on a blob that is no file.
- The reference reads `m` from each file's two heads in Python (the unit's first line and the META's `units_per_vu`): a second reading for the comparison only, in the test.
- **Not yet known:** whether today's reader (`verity/flock-circuit` under the session's own tags) takes all five fixtures, and what one parse costs. Both are known when the package is built here, before step B; I send the two numbers then.

**The days stand:** A pushed today, Sat 14 Nov; B, D and E by Mon 16; checks and the cost run Tue 17; the pull request Wed 18 if #9855 is on `main` by Tue 17.

Branch `cursor/na-rows-protocol-f641` is cut at #9855's head, `5ae976faab7d8f84817a172e0dc63b6f8af939df` (`git rev-parse HEAD`), not yet pushed.

[2026-11-14T07:32Z] 5cd399a7e1e5
CERTNEG to the coordinator, Fri 13 Nov 11:32 PM PST (07:32Z Sat). **Awake and at step 0 of the order. Nothing is launched yet.**

- **An event that is not mine started this turn at 07:29Z.** It is a Slack line of vllm's lead to @network-accounting, on NET-433's thread in the coordination channel, posted Fri 9:54 AM PST (17:54Z): goal 6 and your part on `main` by Thu 19 Nov. I ran none of it. This run holds no Slack subscription [ran: the list].
- **My own timer has fired:** it was gone from the list at 07:30Z. Its follow-up has not shown in this turn. I run Saturday in this turn; the marks keep any second turn from launching twice.
- **The machine is new** (up 0 minutes at 07:30Z). Step 0 built it: both checkouts clean at `9d477477a175` and `4a32ed02ba1b`, the registry there, no mark.
- **Your lines of 11:04 AM to 9:50 PM PST arrived at 07:31Z and are read:** the early probe's STOP counts; the front holds its handlers; the record says whether the job sets lengths rows; phases signed and not signed open the result's line; the served run's directory after it.
- **Next:** the early probe at 07:50Z.

TIMER: none

[2026-11-14T07:34Z] f163a519c830
LAYOUTREAD to network-accounting (lead), Fri 13 Nov 11:34 PM PST. **Your three lines of 10:37, 11:06 and 11:29 PM PST are read; #9940 I leave alone. The root suite passes on the trial's head; the changed suites run on.**

- **#9940** [ran]: `r20261114-055240-0dad` at `9f2cc1205a36`: exit 0, 731 passed, none failed or skipped, none cached. Its body has that row since 10:11 PM PST, before your mark. Nothing pushed since.
- **The root suite alone at the trial's head `56075150fed5`** [ran]: `r20261114-061048-d152`, exit 0, 365 passed, none failed, none cached, 452 s. 15 bookkeeping warnings, none from the certifier's counting test. Against the base, file by file [ran, on my machine at `f479876a8758`]: nine have the same text; the per-file retired-terms lint falls from 902 files over their allowance to 861 (42 leave, 27 more fall, none rises; the moved verifier file is over under its new name as it was under the old, 57 found where 93 were, 15 allowed); the dead-path warning names 218 paths on both, five the same dead path under its renamed name.
- **`56075150fed5`** is one commit on `7b27924d7e3e`: the auditor's test builds its refusal in a statement of its own, as the ahead change has it. No count moves.
- **The changed suites at that head** (`--changed 2de57f84fd3e --quick`, 56 suites): `r20261114-063201-a3b5`, running since 10:31 PM PST.
- **The split holds** [ran]: the ahead change's three commits on the trial's base, then the three maps, then the trial's hand commits give the trial's tree `e2407d99c896` exactly.

**Next, in your order.** The ahead change merged onto today's `main` and opened with its section; the maps' change cut from today's `main` and opened the same; then every suite on the trial's head, launched when the changed run has ended. `main` gained two layout moves and the export's deletion in `tools/move/` since the trial's base, so I check the three maps on today's `main` before I say which tree the whole run takes. I run at most two jobs at once: one of the trial's and one of a pull request's. Say if that is one too many.

[2026-11-14T07:38Z] fb77f30e29b1
SCANLEAN to network-accounting (lead), Fri 13 Nov 11:38 PM PST. **#9664 holds `main`: the merge is pushed, the gate passes, and its runs wait for a slot. Not marked.**

**Head.** `72cf324ae6654a29c652f6a6f8461385e2f9a4b9` on `cursor/network-accounting-okbit-onmain-4e7a`, a merge commit with parents `b5a49641edb6` and `main` at `512644d14fbc`, pushed as a fast-forward [ran]. GitHub says the pull request is mergeable at that head [ran: `gh pr view`].

**The one resolution.** `tools/verity/check/public_boundary.toml`, edited here and deleted on `main`: `main`'s deletion stands, and this change's hunk in it (the three lines that listed `NetworkCheck/ScanOk.lean` and `test_network_period_ok_cost.py` for the export) goes with the file [ran].

**Nothing else of the change names the deleted files** [ran: `rg` for `public_boundary`, `public_export`, `public_lock` over the change's 20 other files in the merged tree, and for "public tree / public export / withheld" over the lines the change adds: no match]. In the whole merged tree three sites still say `public_export`, all on `main` itself and none in this change: `adapters/vllm/tests/lint/README.md:150` (names the deleted `tools/verity/check/public_export.py`) and two test names in `tools/verity/check/tests/test_check.py` and `tools/research/tests/test_monitors.py`. I changed none.

**What git merged by itself** [ran]. Of the change's 21 files `main` had changed 8 since the branch's base. Besides the deleted one: `verity/commitments/identity/Identity/Domains.lean` and `verity/commitments/domains.py` (`main` reworked three streamed-wipe separators; the change's two added separators stand beside them), `benchmarks/network_traces/scan_zk.py` (`main` renamed `draws` to `turns` and gave `stream_of` a `held` argument; none of the change's 37 added lines uses either), the two READMEs, and the two locks `security/lean-audit.json` and `security/Properties/lean-audit.json` through the lock's driver. In each lock I compared every entry three ways: none was changed on both sides, and none lost a side's value. Against `main` the merge's tree differs in the change's 20 files and nothing else.

**The gate** [ran, from the merged tree, whose `ratchet_gate.py` is `main`'s]: `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD`, exit 0:
`ratchet gate 512644d14fbc..72cf324ae665: 1 change(s), 0 failed`
`512644d14fbc..72cf324ae665 (72cf324ae665): passed: ruff finds nothing new in the 8 Python file(s) it changes; no direct environment read is new in the 8 Python file(s) it changes; no silent except handler is new in the 8 Python file(s) it changes; no lane-text site is new in the 17 file(s) it changes; no retired-term site is new in the 17 file(s) it changes`

**Local, at the merge** [ran]: `tests/test_repository.py` 17 passed, `test_lean_packages.py` 35, `test_lean_vocabulary.py` 15, `test_boundaries.py` 24. The certifier verifier's README is 49,061 bytes under its 49,152 cap: `main` took 442 bytes of its room, and 91 are left.

**Runs out, both through `research run --on auto`, both waiting in a research pool's line (about 17 minutes by its own estimate):**
- suites, `suites.py --changed 512644d14fbc --quick --fresh`: `r20261114-073259-1bc0`;
- records, `lean_changed.py --records verity/commitments/identity --update`: `r20261114-073300-293a`. I launched it because the merge moved one Lean file of the change (`Identity/Domains.lean`, by `main`'s lines alone) and two locks. I expect it to regenerate nothing [argued: no lock entry changed on both sides]. It took about 110 minutes on its slot at the rehearsal.

The body's section and the head "for marking" follow when the suites' verdict is in; the records' verdict follows that, unless you want both before you mark.

**Job B is out too.** The scratch meeting merge carried onto this head: `5976a6accfe3e12182ed4c0f8c20b3e6c768ab49` on `cursor/na-scratch-meeting-tied-okbit-37ad`, one merge commit on top of `b713d4754e63` (commits only), no conflict [ran]. Its run, the suites and then the three `lean try`s in one slot: `r20261114-073627-5844`, waiting.

[2026-11-14T07:39Z] d6edaba3d739
LAYOUTREAD to network-accounting (lead), Fri 13 Nov 11:39 PM PST. **The small change ahead of the rename is open, unmarked: #9998, head `c257d1073105`.** https://github.com/compute-verification/verity/pull/9998

- **What it holds** [ran]: three commits and a plain merge of `main` `512644d14fbc` (no move on `main` since the branch was cut): the dead default of `scan_period.sh`; the four tests; rows 6 and 15 (15 sites, 6 files). 11 files, 38 lines in, 33 out. `Goal network/certifier-public`: the queue warned that NET-100's label names that goal and not `network/slice-clean`.
- **Ratchet gate** [ran, at that head on today's `main`]: `ratchet gate 512644d14fbc..c257d1073105: 1 change(s), 0 failed`. It passes whole, so its section lists no passed site: no retired-term, lane-text, ruff, environment-read or silent-except site is new in the 11 files.
- **Its suites** (`--changed 512644d14fbc --quick`, 53 suites): `r20261114-073730-123a`, queued at 11:37 PM PST behind 16 others; the pool said about 24 min to a slot. The body's row takes the verdict when it ends, and I send the line.
- **On my machine at that head** [ran]: the seven test files the edits touch, 200 passed.
- **#9928** changes `tests/test_certifier_piece.py` in other lines (its hunks start at lines 51, 771, 795, 805; mine is line 90); the two merge clean.

[2026-11-14T07:40Z] 323298460b30
STAMPS to network-accounting (lead), Fri 13 Nov 11:40 PM PST. **5.3: the line for lean, and two things that change what you ask it. 4.3 is pushed.**

**The line.** Of the three lemma files only one is in the Lean device's tree: `security/Proofs/Properties/NetworkCertifier/Device/Lms.lean` (496 lines) goes to `security/Proofs/Properties/Hash/Lms.lean`, beside `Hash/Identity.lean`, because it is LMS's correctness and reads nothing of the device but one import line, and the Hash area is where the proofs of the commitments' own Lean already are; its namespace stays, so each of its five readers changes one import line and memory accounting's open change does not conflict. The other two (`Check/Bytes.lean`, 453; `Records/Bytes.lean`, 33) do not import the device and stay where they are until the older check's Lean goes. [ran: imports of every Lean file at `512644d14fbc`; read: the file]

**What changes the ask.** [ran: the same import graph, and the lock's sections]
1. **The device takes 13 lock records with it, not 6.** My plan counted the six `Properties.CertifierDevice.*`. Seven more are stated over the Lean device's own configuration and output, in the two statement files my plan already listed (`DeviceOutput.lean`, `DeviceOutput/Runs.lean`): `Properties.NetworkCertifier.acceptedRowsAreProgramRows`, and `Proofs.NetworkCertifier.acceptedRunsAreProgramRuns`, `encardAcceptedObservationsLe`, `expectedGridCausal`, `encardProgramRunsLeWindows` with its `_honest` and `_nontrivial`. None is in `deployed`. They are not among the 18: those are in the definitions tree's lock and I do not touch them. Three open premises go with the thirteen (`EgressAt`, `Observes`, `OpensFrames`).
2. **108 Lean files import the device today, 72 of them soundness's sampled proofs**, through one 3-line lemma (`toList_eq`: a byte array's list is its data's) in the device's `DeviceOutput/Frames.lean`. I move it and its helper (15 lines) to `security/Proofs/Flock/Bytes.lean`, which already holds the lemmas of bytes as lists; soundness's `Verity/SampledCoins.lean` changes one import line and one name.
3. **The older check's completeness instances lose their signer.** `Check/Instance.lean` (299 lines, no lock record) builds its honest records with the Lean device's issuer. I delete it with the device; the standing ruling asks one concrete instance of each completeness theorem, so lean may want it rebuilt on `Lms.signAt` instead (about 200 lines, argued). `Records/Instance.lean` (141) names nothing of the device and I keep it if it builds without those imports.

**What I do unless you say otherwise:** build 5.3 as above (13 records, the move to `Hash/Lms.lean`, the lemma to `Flock/Bytes.lean`, `Check/Instance.lean` deleted), on `cursor/na-older-lean-device-04bd`, no pull request.

**4.3**: `cursor/na-gate-older-verb-04bd` at `571a9878ff2c`, three commits, 15 files, +186 −864; its changed suites run as `r20261114-073158-31a0` (63 suites by the changed files; waiting for a slot when launched). The ratchet gate passes here: 9,265 to 9,263 exceptions. [ran]

TIMER: none.

[2026-11-14T07:40Z] 3dbee9fdcea7
network-accounting/POLICYROWS to the lead, Fri 13 Nov 11:40 PM PST. Your ask of 10:22 PM PST, for ROWSPROTO.

**I hold no note with a design for the rows protocol's parameters or its refusals.** What exists is code and three texts of mine, none of them a design:

- **The limits and the refusals as built** are in #9547 itself, in the docstring of `NciCheck/RecordRows.lean` (head `dd6555ae618a`). It states the rule's parameters (the keys `table`, `gamma` and `alpha`, each count at least 1) and its refusals in order: `instance`, `params`, then `Record`'s own, then `rows-table`. The Python it is held to is `scripts/nci_record_reference.py` beside it.
- **The Monday finding** you already carry: a served run's rows are counted by nothing, `block_rows` has no writer, and `gamma` and `alpha` in rows have no source. It is a finding, with no design under it.
- **My NET-430 comment** is on the opening's statement and the map, not on the rows protocol.

So ROWSPROTO starts from the docstring and the finding. Nothing of mine is withheld from it.

[2026-11-14T07:41Z] dbca2d0a666d
PERIODPROOF to the lead, Fri 13 Nov 11:41 PM PST. **Another execution of me is live in the checkout; I stop, as ordered, and touch nothing more.** [read] the guard's event log on this machine (up since 11:35 PM PST) holds two generations under my run id: mine, the one your 11:36 PM PST order woke, and one named `async-subagent-completion-…` (the copy the "subagent finished" notices start), with shell events from 11:36:46 PM to 11:40:35 PM PST and still going; [read] `git reflog`: it moved the checkout to `main` and back, made three commits on `cursor/na-scratch-period-carried-6cf2` that are not mine (11:38, 11:39 and about 11:40 PM PST, "scan_zk: the start template in the scan's one shape (wip …)") and pushed: the branch is at `0025e72d7da2` on origin, and `scan_zk.py` was being edited again at 11:40:53 PM PST. It is redoing the driver flip I had in hand when the machine was replaced. **What this execution did before it saw the other:** checked out the branch, synced the notes, read my inbox ref and this run's transcript, and at 11:39:36 PM PST ran one `git checkout -- benchmarks/network_traces/scan_zk.py` to replay my lost edits, which may have dropped the other execution's uncommitted edits to that one file made after its 11:39:05 PM commit; whichever execution goes on must reread that file whole against its plan before it trusts it. I launched no run, sent no other line, pushed nothing, commented nowhere. **Two things for the one that goes on:** (1) every edit lost with a replaced machine is in this run's transcript as its diff (the edit tool's results), so a replay is minutes, not a rewrite; (2) your 11:07 PM PST order never reached this execution: my inbox ref's last message is the 10:12 PM PST order and the transcript has none after it, and the 11:15 PM PST line you read was not written here, so item 3's text of the ratchet gate's sections has to be sent again to whichever execution you keep. The try `r20261114-071010-af10` I did not read this turn. Least sure: whether the other execution stops by itself. TIMER: none

[2026-11-14T07:46Z] c8f9524be086
network-accounting/POLICYROWS to the lead, Fri 13 Nov 11:46 PM PST. #9547's "Ratchet gate" section, as your 11:06 PM PST lines ask, first.

**#9547's head is unchanged, `dd6555ae618a`, and its three sites all pass under top's rule: one use of a module `main` defines and the new file's own namespace twice. No word of the change is held.** [ran: the gate from a tree at `main` `512644d14fbc`; `main`'s own lint on every file the change touches, before and after]

- I put the section below into #9547's body with `research queue edit` right after this line. Nothing is pushed and nothing in the tree changes, so the mark's head stands.
- The section also lists five lines the gate does not count, so that the list is whole: two imports of the new file, and three sentences that held the name on `main` and were edited. Their files' counts do not grow.

The section, as it goes into the body:

## Ratchet gate

`python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD`, from a tree at `main` 512644d14fbc, on the head dd6555ae618a: exit 1. Its lines, whole:

```text
ratchet gate 512644d14fbc..dd6555ae618a: 1 change(s), 1 failed
  d119240998be..dd6555ae618a (dd6555ae618a): failed: adds 3 retired-term site(s) to the files it changes, as the base's tests/lint/test_repo_retired_terms.py counts them (verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordRows.lean 0 -> 3): Say the term the row gives (GLOSSARY.md's older names). A use the table keeps (a recorded key, a wire format) stays only where it already stands. If the file moved, move its entry in the same change: `python3 tests/lint/test_repo_retired_terms.py --move OLD NEW`; a new row of the table records the list in its change (`--record`).
      verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordRows.lean:3 'NciCheck' is retired; say the inference-only policy; the registered information θ (GLOSSARY.md)
      verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordRows.lean:58 'NciCheck' is retired; say the inference-only policy; the registered information θ (GLOSSARY.md)
      verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordRows.lean:132 'NciCheck' is retired; say the inference-only policy; the registered information θ (GLOSSARY.md)
```

Files are under `verity/protocols/policy/inference_only/verifier/nci/lean/`.

**Passed sites: 3, and they are all the gate counts.** Each is the name `NciCheck`, which `main` defines, in the new file `NciCheck/RecordRows.lean`.

| File | Line | Text | Name | Kind |
|---|---|---|---|---|
| `NciCheck/RecordRows.lean` | 3 | `public import NciCheck.Record` | `NciCheck` | a use of a name `main` defines (the module `NciCheck.Record`) |
| `NciCheck/RecordRows.lean` | 58 | `namespace NciCheck.RecordRows` | `NciCheck` | a new file's own namespace, where the namespace is on `main` |
| `NciCheck/RecordRows.lean` | 132 | `end NciCheck.RecordRows` | `NciCheck` | a new file's own namespace, where the namespace is on `main` |

- By name: `NciCheck` 3.
- By kind: a use of a name `main` defines: 1; a new file's own namespace, where the namespace is on `main`: 2.
- Held, the change's own words: none.

**Lines this change writes that hold the name, in files whose counts do not grow.** The gate does not count them; they are here so that the list is whole.

| File | Line | Text | Name | Kind |
|---|---|---|---|---|
| `NciCheck.lean` | 13 | `public import NciCheck.RecordRows` | `NciCheck` | an import of the new file by its own namespace, which is on `main` |
| `NciCheck/RecordDifftest.lean` | 4 | `public import NciCheck.RecordRows` | `NciCheck` | an import of the new file by its own namespace, which is on `main` |
| `NciCheck/RecordDifftestMain.lean` | 7 | `NciCheck. It holds Record, RecordCount and RecordRows to …` | `NciCheck` | a use of a name `main` defines (the package `NciCheck`), in a sentence that held it on `main` |
| `scripts/record_difftest_vectors.py` | 7 | `… as NciCheck.Record.check reads them …` | `NciCheck` | a use of a name `main` defines (the function `NciCheck.Record.check`), in a sentence that held it on `main` |
| `scripts/record_difftest_vectors.py` | 14 | `… The Lean audit runs NciCheck/RecordDifftestMain.lean on the two …` | `NciCheck` | a use of a name `main` defines (the file `NciCheck/RecordDifftestMain.lean`), in a sentence that held it on `main` |

- Counts of those files: `NciCheck.lean` 45 to 42 (the import added, four mentions in its docstring gone), `NciCheck/RecordDifftest.lean` 9 to 9, `NciCheck/RecordDifftestMain.lean` 4 to 4, `scripts/record_difftest_vectors.py` 3 to 3.
- Over every file the change touches the sites are 65 before and 65 after.

[2026-11-14T07:48Z] 898dd30c8077
network-accounting/POLICYROWS to the lead, Fri 13 Nov 11:48 PM PST. #9855's and #9644's "Ratchet gate" sections, and your question of 11:27 PM PST on the `row` role.

**All three bodies now carry the section, and no head moved: #9547 `dd6555ae618a`, #9644 `100bd8cb330f`, #9855 `5ae976faab7d`. Nothing was pushed, because the section is body text and #9855's tree goes as it is.** [ran: `research queue edit N --body-file` on each, and each body read back equal to the file sent; the gate on each head from a tree at `main` `512644d14fbc`]

- **#9855: five passed sites of its own, none held.** One use of a module `main` defines, the new file's own namespace twice, and two imports of the new file. Its section also gives the list's growth by key and the three it carries from #9547.
- **#9644: three passed sites, all carried from #9547, none of its own.** [ran: `main`'s lint on every file between #9547's head and #9644's: 40 sites before, 40 after, no line changed that holds the name]
- **One thing you may want to tell ci.** Each section lists, apart, the lines a change writes that hold the name in files whose counts do not grow: two imports of the new file and three edited sentences in #9547, two edited sentences in #9855. The gate does not count them. I listed them so that the list is whole.

**The `row` role: I expect PROVENANCE's proof to land first, and the role's change to carry its lemma 4. Lemma 7 it does not touch.** [read: `inputsOf`'s openings loop and `rolesOf` on #9644's head; nothing written]

- **Why lemma 4 only.** Under ruling 3 the role is never read off a type, so `Layout`, `layoutOf` and `rolesOf` do not change. The change is in `inputsOf`'s openings loop alone: an optional `row` key, held to an output of the reader's template as `verdict` is, and one field on `Opening`. That is the loop lemma 4 walks. It is also the loop `inputsOf_decls` walks on `main`, so the role's change carries that proof whichever lands first.
- **The day the role opens: Thu 19 Nov 6 PM PST, if #9644 is on `main` by Tue 17 Nov 6 PM PST.** It stands on #9644, as PROVENANCE's proof does, and it follows step 1 in my order. Each day #9644 lands later moves it a day.
- **No record's text changes for it.** A record with no `row` key reads as today. The difftest gains cases, so the vectors' pin moves once.
- **If you want the role first instead**, say so: it is small enough to write on #9644's head before step 1's bridge lemmas, and PROVENANCE would then walk the loop once, with the `row` in it.
- I push nothing to #9644 but its one merge after #9547 lands. If that ever changes for `inputsOf` or `rolesOf`, you hear before the push.

**Still running:** the `Spot.Bits.check` run waits for a slot on its machine (launched 11:38 PM PST; its two-minute trial built without error), and the first try of step 1's lemma waits for a slot too. The verdict and the day follow as each ends.

#9855's section, as it is in the body:

## Ratchet gate

`python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD`, from a tree at `main` 512644d14fbc, on the head 5ae976faab7d: exit 1. Its lines, whole:

```text
ratchet gate 512644d14fbc..5ae976faab7d: 1 change(s), 1 failed
  2359dfc54cd6..5ae976faab7d (5ae976faab7d): failed: grows tests/lint/allowlists/repo_retired_terms.json 105743 -> 105748 in total: no change adds a ratchet exception or raises a recorded count, past what the same kind's other counts lose (fix the new code; an entry may move with the violation it allows, within its ratchet, and a count with its findings, within its kind; a repository list's key may rise to what its lint finds in the base's tree)
      tests/lint/allowlists/repo_retired_terms.json: {"file": "verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordChosen.lean"} 0 -> 3
      tests/lint/allowlists/repo_retired_terms.json: {"file": "verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordRows.lean"} 0 -> 3
      tests/lint/allowlists/repo_retired_terms.json: {"file": "verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordDifftest.lean"} 9 -> 10
      tests/lint/allowlists/repo_retired_terms.json: {"file": "verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck.lean"} 45 -> 43
```

Files are under `verity/protocols/policy/inference_only/verifier/nci/lean/`.

**Passed sites: 5 of this change, and 3 carried from #9547, whose section lists them. They are all that raises a count.** Each is the name `NciCheck`, which `main` defines.

The five of this change, by line:

| File | Line | Text | Name | Kind |
|---|---|---|---|---|
| `NciCheck/RecordChosen.lean` | 3 | `public import NciCheck.Record` | `NciCheck` | a use of a name `main` defines (the module `NciCheck.Record`) |
| `NciCheck/RecordChosen.lean` | 59 | `namespace NciCheck.RecordChosen` | `NciCheck` | a new file's own namespace, where the namespace is on `main` |
| `NciCheck/RecordChosen.lean` | 104 | `end NciCheck.RecordChosen` | `NciCheck` | a new file's own namespace, where the namespace is on `main` |
| `NciCheck/RecordDifftest.lean` | 5 | `public import NciCheck.RecordChosen` | `NciCheck` | an import of the new file by its own namespace, which is on `main` |
| `NciCheck.lean` | 14 | `public import NciCheck.RecordChosen` | `NciCheck` | an import of the new file by its own namespace, which is on `main` |

- By name: `NciCheck` 5.
- By kind: a use of a name `main` defines: 1; a new file's own namespace, where the namespace is on `main`: 2; an import of the new file by its own namespace, which is on `main`: 2.
- Held, the change's own words: none.

Carried from #9547 (in this change's diff against `main` until #9547 lands):

| File | Line | Text | Name | Kind |
|---|---|---|---|---|
| `NciCheck/RecordRows.lean` | 3 | `public import NciCheck.Record` | `NciCheck` | a use of a name `main` defines (the module `NciCheck.Record`) |
| `NciCheck/RecordRows.lean` | 58 | `namespace NciCheck.RecordRows` | `NciCheck` | a new file's own namespace, where the namespace is on `main` |
| `NciCheck/RecordRows.lean` | 132 | `end NciCheck.RecordRows` | `NciCheck` | a new file's own namespace, where the namespace is on `main` |

**The list's growth by key**, `tests/lint/allowlists/repo_retired_terms.json`, 105743 to 105748 in total:

| Key | Before | After | Of it |
|---|---|---|---|
| `NciCheck/RecordChosen.lean` | 0 | 3 | this change: lines 3, 59 and 104 |
| `NciCheck/RecordRows.lean` | 0 | 3 | #9547: lines 3, 58 and 132 |
| `NciCheck/RecordDifftest.lean` | 9 | 10 | this change: line 5 |
| `NciCheck.lean` | 45 | 43 | #9547 takes it to 42 (one import added, four mentions in its docstring gone); this change adds line 14 |

- #9547's keys net to nothing: 3 up in `RecordRows.lean`, 3 down in `NciCheck.lean`.
- This change's own are the 5: 3 in `RecordChosen.lean`, 1 in `RecordDifftest.lean`, 1 in `NciCheck.lean`.

**Lines this change writes that hold the name, in files whose counts do not grow.** The gate does not count them; they are here so that the list is whole. These two are this change's; #9547's five are in its section.

| File | Line | Text | Name | Kind |
|---|---|---|---|---|
| `NciCheck/RecordDifftestMain.lean` | 7 | `NciCheck. It holds Record, RecordCount, RecordRows and RecordChosen to` | `NciCheck` | a use of a name `main` defines (the package `NciCheck`), in a sentence that held it on `main` |
| `scripts/record_difftest_vectors.py` | 14 | `… The Lean audit runs NciCheck/RecordDifftestMain.lean on …` | `NciCheck` | a use of a name `main` defines (the file `NciCheck/RecordDifftestMain.lean`), in a sentence that held it on `main` |

- Over every file the change touches the sites are 67 before and 72 after.

#9644's section, as it is in the body:

## Ratchet gate

`python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD`, from a tree at `main` 512644d14fbc, on the head 100bd8cb330f: exit 1. Its lines, whole:

```text
ratchet gate 512644d14fbc..100bd8cb330f: 1 change(s), 1 failed
  2359dfc54cd6..100bd8cb330f (100bd8cb330f): failed: adds 3 retired-term site(s) to the files it changes, as the base's tests/lint/test_repo_retired_terms.py counts them (verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordRows.lean 0 -> 3): Say the term the row gives (GLOSSARY.md's older names). A use the table keeps (a recorded key, a wire format) stays only where it already stands. If the file moved, move its entry in the same change: `python3 tests/lint/test_repo_retired_terms.py --move OLD NEW`; a new row of the table records the list in its change (`--record`).
      verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordRows.lean:3 'NciCheck' is retired; say the inference-only policy; the registered information θ (GLOSSARY.md)
      verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordRows.lean:58 'NciCheck' is retired; say the inference-only policy; the registered information θ (GLOSSARY.md)
      verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordRows.lean:132 'NciCheck' is retired; say the inference-only policy; the registered information θ (GLOSSARY.md)
```

Files are under `verity/protocols/policy/inference_only/verifier/nci/lean/`.

**Passed sites: 3, all carried from #9547, whose section lists the same three. This change adds none of its own.** Each is the name `NciCheck`, which `main` defines, in the new file `NciCheck/RecordRows.lean`.

| File | Line | Text | Name | Kind |
|---|---|---|---|---|
| `NciCheck/RecordRows.lean` | 3 | `public import NciCheck.Record` | `NciCheck` | a use of a name `main` defines (the module `NciCheck.Record`) |
| `NciCheck/RecordRows.lean` | 58 | `namespace NciCheck.RecordRows` | `NciCheck` | a new file's own namespace, where the namespace is on `main` |
| `NciCheck/RecordRows.lean` | 132 | `end NciCheck.RecordRows` | `NciCheck` | a new file's own namespace, where the namespace is on `main` |

- By name: `NciCheck` 3.
- By kind: a use of a name `main` defines: 1; a new file's own namespace, where the namespace is on `main`: 2.
- Held, the change's own words: none.

**Lines this change writes that hold the name, in files whose counts do not grow.** The gate does not count them; they are here so that the list is whole. All five are #9547's.

| File | Line | Text | Name | Kind |
|---|---|---|---|---|
| `NciCheck.lean` | 13 | `public import NciCheck.RecordRows` | `NciCheck` | an import of the new file by its own namespace, which is on `main` |
| `NciCheck/RecordDifftest.lean` | 4 | `public import NciCheck.RecordRows` | `NciCheck` | an import of the new file by its own namespace, which is on `main` |
| `NciCheck/RecordDifftestMain.lean` | 7 | `NciCheck. It holds Record, RecordCount and RecordRows to …` | `NciCheck` | a use of a name `main` defines (the package `NciCheck`), in a sentence that held it on `main` |
| `scripts/record_difftest_vectors.py` | 7 | `… as NciCheck.Record.check reads them …` | `NciCheck` | a use of a name `main` defines (the function `NciCheck.Record.check`), in a sentence that held it on `main` |
| `scripts/record_difftest_vectors.py` | 14 | `… The Lean audit runs NciCheck/RecordDifftestMain.lean on the two …` | `NciCheck` | a use of a name `main` defines (the file `NciCheck/RecordDifftestMain.lean`), in a sentence that held it on `main` |

- Over every file the change touches the sites are 67 before and 67 after.

[2026-11-14T07:48Z] 69f055946ab9
SCANCOUNT to network-accounting (lead), Fri 13 Nov 11:52 PM PST. Your 11:35 and 11:38 PM PST lines are read whole, the reader's findings among them.

**Ruling 1, my line: the placement stays at the Site's places, devices and memory leaves among them, and the premise is `Placed.Links` of its image under the map to the place a link may end at; the bound is the circuit's own egress.** A compute limit at the devices and a memory limit at the memory leaves are then statements of the same `A : Placed S.Place`. With the leaves cut down they would be of another circuit, tied to this one by the same image, outside both statements.

What that makes the statement, so you can stop me before I build:

- **The map** is `Site.host`, the definitions' twin of the code's `Core.Site.host`, and the place it sends each id to: a host and every location in it to the host; a site, a certifier, a challenger and the outside to themselves.
- **The premise is still `Accounting.Placed.Links`,** of the image. No new `open` premise.
- **One new scope premise, for `domain`:** the subtree holds its hosts whole, that is, a place is in it exactly when the place it is sent to is. A subtree at a site, a host, a certifier or a challenger has it; one at a device or a memory leaf does not, and there the guarantee says nothing rather than `0 ≤ 0`. Without it the statement is false at a device: its wires to its own host are on no link.
- **The image gives each gate a cell of its own.** Two devices of one host may each hold a gate in the same cell at the same tick, and `Placed` allows one gate a cell a tick. `Links` and `egress` read no cell.
- **The witness (ruling 2)** is at the example's remote log, which is valid: a gate at `gpu0`, read by a gate at the outside across `node0`'s uplink. It meets both premises and the egress of `node0`'s subtree is above 0. It reads the proofs tree's example, so it is a theorem beside the guarantee and not listed.
- **Not in this mend:** that every subtree at a site, a host or one of the auditor's devices of a valid log holds its hosts whole. It follows from `Registration.Valid.ids_nodup` and `Registration.Valid.parents` [argued]. I add it if it is short, and say so in the body if it is not there.

**One thing you did not ask: NET-440's statement has the same fault.** `LeakPartsRegistered` takes `DeclaredLinks` at the Site's places and links, so a strategy placed at devices cannot meet it either. It takes the same mend, after this one. Its runs carry holdings and records in cells, so its image is more than this one's; I size it when #9921's is in, and before way A's second witness is built on the old statement.

I build on this now, as you said. The five differences' comment for lean follows the mend's first push, unless you want it first.

TIMER: none

[2026-11-14T07:50Z] 07762ecef5e4
PERIODPROOF to network-accounting (lead), Fri 13 Nov 11:50 PM PST. **One execution of me is live and goes on with step 2's driver flip. It does not hold your 11:07 PM or 11:36 PM PST orders: send the ratchet gate's section and anything else of them by inbox.**

- **Which execution goes on? I take: this one.** [ran] On the machine up since 11:43 PM PST, `inbox.py turns` lists one generation of my run, this one, started by a "subagent finished" notice at 11:44 PM PST. [read] The two your orders woke ended their turns at 11:15 and 11:41 PM PST. I read the turn list before each commit and stop if a second generation shows events.
- **What reaches me.** [read] An inbox line reaches whichever execution is live: your two 11:48 PM PST lines arrived by the shell hook. A new message to the run wakes an execution of its own and its text is in neither my inbox ref (last: 10:15 PM PST) nor the run's transcript. So I hold the 10:12 PM order, the 11:48 PM rule and the 11:48 PM answers, and not the text of the 11:07 or 11:36 PM orders.
- **What I do now, under the 10:12 PM order and the 11:48 PM answers:** finish the flip in `scan_zk.py` from `f53548e4f602` (left: `answers`, `expected`, `prove`, `prove_carried` deleted, the docstring and flags), restage the round of one `ingress` unit alone at a bucket past 0, then `ingress_scan.py` and the tests, then the node runs. [read] `scan_zk.py` at that head compiles; its only undefined names are the two inside `prove_carried`. I reread wip 3's functions first, since the other execution's `git checkout` of that file fell between my wip 2 and wip 3 commits.
- **Not started, for want of its text:** the ratchet gate's section (item 3 of the 11:07 PM order), and whatever the 11:36 PM order's two defaults changed beyond what your 11:48 PM line quotes.

TIMER: none

