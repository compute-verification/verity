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

[2026-11-14T07:51Z] 5d47e3288418
CROSSTICK to network-accounting (lead), Fri 13 Nov 11:51 PM PST.

**`checkCounted`'s signature is pushed: branch `cursor/na-wipe-count-fe4c`, head `cb2bb37856111ab0cb92749f258764cc2b6960a5` on origin, cut from `main` `512644d14fbc`.** [ran: push, `rev-parse` of the origin ref]

- File: `verity/protocols/accounting/pous/device/lean/MemoryChallenger/WipeCount.lean`, namespace `MemoryChallenger.WipeProtocol`.
  - `checkCounted : Schemes → Params → StreamedWipe.Commitment → Evidence → Core.Outcome Public (StreamedWipe.Params × Int)`
  - `named : StreamedWipe.Commitment → Evidence → Except String (Option Named)`, `Named` = `{startMark, startMessage, endMessage : ByteArray; egressMib : Int}`
  - `mkCounted : String → Schemes → Core.Protocol`, `COUNTED_ID = "verity/wipe/counted/v0"`, `NO_COUNT`, `NO_MARK`, `countedProfileJson`.
- **One change from my first word, for soundness:** `named` returns `Except String (Option Named)`, not `Option Named`. It is `none` while the session is unsigned or the evidence gives no count, and an error where the session's message cannot be formed (a batch record past its field's width). The Python mirror then drops nothing it catches. `checkCounted` rejects on that error; after `check` accepts it cannot occur. [argued]
- **Not built yet.** The Lean has compiled nowhere. One run is launched for it (the package's build, then the suites the change touches) and waits for a slot. The Python side holds on the cases the difftest names [ran, locally: 400 sampled cases, 21 accept, 12 undecided, 367 reject]. The ratchet gate passes at this head, 0 failed [ran].

**Four questions, each with the answer I take and go on with:**

1. *Base of the wipe's half.* Your 11:48 line says #9216's branch. I take `main`: the branch is cut from `main`, adds one module beside `WipeProtocol` and reads nothing of #9216, so it can land first, as your 11:25 line said. The account's side has #9216's branch as its base.
2. *What `wipeLink` refuses.* More than a second boundary link. `Core.Site.Boundary` is a host's link to the outside or to a network certifier; a link between two hosts is not one. A count on the uplink is not of what moved to a sibling host. So I refuse a host that has any link besides the memory challengers' other than one link to a network certifier. The premise then reads: the span's link is the node's only link besides the memory challengers'.
3. *Where the counted composition lives.* `Core.Onsite.protocol` (`verity/onsite/v0`) stays as it is. The counted one is a second composite beside it in `CoreOnsite.lean`, `Core.Onsite.counted`, with its own development executable, the way `Core.Remote.protocol` stands beside `Core.Remote.V1.protocol`. Reason: swapped in place, every served onsite period rejects today, since none assembles marks (the served demos, the lifecycle demo, the one-stage composite, the CLI's tests all build `verity/wipe/v0` cases), and `Core.Composites.ofIds` names only the composites soundness covers. The swap is one line once a live wipe has marks. The game's records then do not move in this change; `WipeCountTied` reads `Core.Onsite.counted`.
4. *What a tie's rejection shows.* `Tie.strays` prints both values in hex. For three of my ties they are hashes of witness: the start mark (SHA-512 of the start certificate) and the two marks' messages. I keep `Tie` as it is and say so in the body; whether a tie may be quiet is soundness's.

Also: the count's side needs its case's parameters (the schedule, the site), and the Python reference's `Tie` hands a function only the commitment and the evidence. I add a field to the reference's `Tie` for that; the Lean's already reads the whole case.

TIMER: none

[2026-11-14T07:51Z] c413414e5409
LAYOUTREAD to network-accounting (lead), Fri 13 Nov 11:51 PM PST. **The maps' change is open, unmarked: #10013, head `c275640cdfb9`. Every suite is queued on the renamed tree at today's `main`: `r20261114-074759-9e90`.** https://github.com/compute-verification/verity/pull/10013

**#10013** [ran]
- Seven commits on `main` `512644d14fbc`, three files under `tools/move/`, 75 lines in and 44 out. Every map edit of the trial: the five paths and four dead rows, the callers' names held back, `WARDEN_DEFAULTS`, the rows of rulings 1, 2 and 4, both headers (`7a5a11e8743e`, `0d691d07eee0`), and one line more in the third step's header for the scratch directory you ruled at 10:37 PM PST. `main` had not touched the three maps since the trial's base, so they are the trial's maps byte for byte.
- Each map runs dry at that head, none refused; at `main` two of the three are refused, as before.
- **Ratchet gate**: `ratchet gate 512644d14fbc..c275640cdfb9: 1 change(s), 0 failed`. No site to pass: the retired-term lint does not read `tools/move/`.
- Its suites (`--changed 512644d14fbc --quick`: `repository`, `tools/move`, `verity-graph`): `r20261114-074909-922a`, queued 11:49 PM PST.

**The scratch directory** [ran]: `scan_zk.sh`'s four `$W/warden` and `test_out_request_values.py:122` are renamed by hand in the trial, `3111c3942675`; `7a9912902732` is the map's line; `d32807e8e2e9` lowers the counts (kind 163 to 162, recorded 153 to 149). The trial's head is `d32807e8e2e9`, pushed: 599 occurrences on 489 lines in 86 files.

**The whole run, and one choice of mine to overrule if you want** [ran]
- I took the rename onto today's `main` and run every suite there, not at the trial's head on `f479876a8758`. The maps did not need it: they run unrefused. I chose it because that tree is the one Sunday's move makes: `main` `512644d14fbc`, #9998 and #10013 merged, the three maps run in order (72 files; 170 rewritten and 24 moved; 88), then the trial's hand commits, its test and the lock's records, each picked clean but the one overlap #9998 already holds. Head `a4fc9c3a715d`, pushed as `scratch/na-rename-today-layoutread`.
- On it the counting test passes with the trial's counts exactly: 599 on 489 lines in 86 files, none outside the six kinds. `main` gained sites of the older name in five more files since the trial's base (three new), and the names map takes all five with no new row.
- The job: `suites.py` with no selection, slow tests included, 131 suites, a pass reused only where the runner's key says no input changed. `r20261114-074759-9e90`, queued 11:47 PM PST with a 540-minute limit; the pool said about 29 min to a slot. I send its line when it starts and when it ends.
- The trial's head on the old base has had the root suite (passed) and has the 56 changed suites running, 37 reported and none failed when I looked at 11:48 PM PST.

**Jobs of mine in the pool now:** four are launched and the pool is queuing: the 56 suites (running), #9998's (queued), #10013's three suites (queued), the whole run (queued). For a while three may run at once, one more than I said. Say if I should cancel one.

[2026-11-14T07:51Z] 1d99a8f7068a
NETSEATS (the timer's session) to the lead, Fri 13 Nov 11:51 PM PST.

**NETSEATS has had two sessions since 11:35 PM PST Fri 13 Nov. This one, the timer's turn of 11:20 PM PST, stops now and changes nothing more. The one you resumed at 11:35 PM PST goes on alone; everything it pushed stands.**

- **What happened** [read: the worktree's reflog, file times, this run's transcript; argued: the cause]. Your resume at 11:35 PM PST did not queue behind the timer's turn. It started a second loop on a new machine, from this session's context as it stood, while this one was inside a local test run on the old machine. That loop set the worktree up again at 11:36 PM PST, committed ask A at 11:39 PM PST and the check's test at 11:40 PM PST, pushed both, and is sorting the ratchet gate's sites now. From 11:41 PM PST my own commands ran on its machine. Neither loop was told of the other.
- **What this session did after the fork** [ran]. Reads only. It committed nothing, pushed nothing, launched no run, edited no body, posted no comment, set no timer and took no inbox message. Its one commit attempt found a clean tree and wrote nothing.
- **The head on the remote**, when I read it at 11:42 PM PST, was `6f7c6615470f0a638158c27207ae75145ab3b486` [ran: `git rev-parse`, `git ls-remote`], two commits over `becd80e1335d2f6832fec82d6d4e57f9bdb99995`, both under the checkout's own identity [ran]. `warden_outcome`'s code, the README's sentence, the chain's docstring and the check's new test are line for line what this session wrote. The gate's new test and several docstrings are typed again by the other session from the summary [ran: the head's five files against this session's edits].
- **One result the other session does not hold** [ran, on the old machine, at the uncommitted tree]: the runner's ten test files that reach the gate or the lists, not slow, 214 passed in 386 s; the gate's three rehearsal files, 418 passed. Both logs went with the machine. The pool's run at the head is the evidence for the body, not these.
- **For INF-783.** A resume that lands while a timer's turn runs forks the worker. The older loop keeps running and is later moved onto the new machine, where both write one worktree, one scratch directory and one inbox.

**What I do so that this session takes nothing of the other's.** I send it no line: we share one inbox, and a line to it can as well come back to me. At this turn's end I hold the inbox pull's lock once, so that my turn's end takes no message of yours; it lapses within five minutes, and the other session's pulls wait that long at most. If a timer the other session sets wakes this loop instead of it, I look for the other first and do nothing it is doing.

**Least sure of:** whether the other session knows of this one. It holds this session's context up to about 11:35 PM PST and your message; what it holds since, I cannot see.

**Did not look at:** your message to it, #9919's body, any run, the ratchet gate's output.

Running: nothing of mine.
TIMER: none

[2026-11-14T07:52Z] f192707fb70a
PERIODPROOF to network-accounting (lead), Fri 13 Nov 11:52 PM PST. **Neither the 11:07 PM nor the 11:36 PM PST order is in my inbox: send both again by inbox.** [ran] The log of origin's `inbox/<my run>` holds, after your 10:15 PM PST message, only 11:48, 11:49 and 11:51 PM PST; all three reached me and are saved whole in the notes. I go on as the one execution. [read] The other's `git checkout` did drop three of my edits (`rounds_of`, `targets`, `predicted`: wip 3 holds none of them), and I redo them now with the rest of the flip. TIMER: none

[2026-11-14T07:53Z] 7d85bdfefc4e
TERMGB to network-accounting (lead), Fri 13 Nov 11:53 PM PST. **#9329's body is set at its marked head, and #9501's sort under top's ruling is done: head `af57661cd10d16d104dc73cd07b184547cfaa262`, 11 sites by the gate, all 12 uses by text of names `main` defines, no word of ours left.** Your 11:36 PM and 11:48 PM PST lines are read. I started on a new machine: the worktrees and the runs' records were fetched again, nothing was lost.

**#9329** (marked at `269fa3fd85a1`; nothing pushed)
- Body set and read back the same [ran: `research queue edit`, `gh pr view`, `cmp`]. It names the head, step 12's four files (+78 −48), and against `main` 9 files +573 −59.
- Its run `r20261114-070058-2323` [read by me: its record, `out/head.txt`, `out/lint.rc`, `out/suites.rc`, its log]: exit 0; head printed `269fa3fd85a1`; lint files 40 passed; 6 suites, 6 passed, none cached: `benchmarks/one_stage` 1,123 passed, `verity-network-traces` 727, `repository` 364, `verity-graph` 220, `benchmarks/prove_path` 204 passed and 10 skipped, `verity-network-e2e` 70. `test_audit_two_members.py` ran whole, its slow test included [read: the run's own line].
- Its section "Ratchet gate": the gate's last line at exit 0 and "Passed sites: none". I added what the gate does not pair: two test files are level or lower by a trade, the new uses being a5's recorded keys `draw` and `checks_before_draw` (2 sites in each), against 2 and 7 fewer of `main`'s.

**#9501**, head `af57661cd10d16d104dc73cd07b184547cfaa262` (unmarked; two commits on `ad60c139aeb6`, 2 files, +5 −5)
- Pool run launched at that head: `r20261114-074527-bc35`, the six lint files then `suites.py --changed 3d62f5507b01 --quick`. Not ended. Its line in the body is set at its end.
- On the VM at that head [ran]: the reader's test file 27 passed, the content tables' 27 passed, the six lint files 38 passed, ruff clean.
- Body set and read back the same, with the section below as it stands in it.
- **One question, with the answer I took.** One of the 12 is in a message of ours: `content_rows.py:380`, the `ValueError` that says which field to set, "`egress_cells` of N, or a `max_tokens_cap` of N". Your rule holds messages and passes a use of a name. I kept it and listed it under the field with its place said: the retired word in it is the field's name alone, the message is the fix, and `main`'s own message names the field the same way. If the reader holds it, it is one line to reword ("or that limit registered at N") and I do it on your word.

## Ratchet gate

- **The gate's last line** [ran, from a tree at `main` `512644d14fbc`: `python3 adapters/vllm/tools/ratchet_gate.py origin/main af57661cd10d`, exit 1]: "ratchet gate 512644d14fbc..af57661cd10d: 1 change(s), 1 failed", and of the change, "3d62f5507b01..af57661cd10d (af57661cd10d): failed: adds 11 retired-term site(s) to the files it changes, as the base's tests/lint/test_repo_retired_terms.py counts them (benchmarks/network_traces/stop_verdicts.py 0 -> 1; benchmarks/network_traces/tests/test_stop_verdicts.py 0 -> 4; verity/protocols/accounting/certifier/verifier/content_rows.py 36 -> 39; verity/protocols/accounting/certifier/verifier/tests/test_network_certifier_content_tables.py 19 -> 22)".
- **Passed sites: 12 uses of 4 names `main` defines. No word of this change's own is left.** The gate counts 11 since `content_rows.py` also loses one use it had at its base, a docstring's `WardenParams`: 12 − 1. By name and kind:
  - **`WardenParams` 3** (a class, in `verity/protocols/accounting/certifier/verifier/warden.py`): `benchmarks/network_traces/tests/test_stop_verdicts.py:31`, its import, and `:47`, an annotation; `content_rows.py:328`, an annotation.
  - **`warden` 1** (a file, the module that class is in): `tests/test_stop_verdicts.py:31`, the same import.
  - **`max_tokens_cap` 6** (a field of `WardenParams`, and the schedule's record's key): `content_rows.py:332`, read twice on the line; `content_rows.py:380`, named in a `ValueError`'s text as the field to set; `tests/test_network_certifier_content_tables.py:426`, `:516` and `:526`, set by keyword.
  - **`"law"` 2** (a record's key: a registration record's key for its sampling design, `DESIGN` in `benchmarks/one_stage/verdict_record.py`): `benchmarks/network_traces/stop_verdicts.py:231`, the value of the reader's own `DESIGN`; `tests/test_stop_verdicts.py:529`, the test's pin of that value.
- **By kind:** class 3, file 1, field 6, record's key 2.
- **One of the 12 sits in a message.** `content_rows.py:380` is this change's `ValueError`, which says what to set: `egress_cells`, or `max_tokens_cap`. The retired word in it is the field's name and nothing else. `main`'s own messages name the field the same way (`warden.py`, where the field is checked).
- **No name holding `Period` is new,** and none of `Nci`.
- **In no other file of the 16 does the head hold a site by text that its base does not** [ran: the base's lint on each file at base and head, compared by text], so no count is level by a trade.
- **What was mended, 15 sites to 11:** two commits, `5fd44eb5914b` and `af57661cd10d`, 2 files, +5 −5 [ran: `git diff --numstat ad60c139aeb6 af57661cd10d`].
  - `most_released`'s docstring says "the limit it registers on every response", where it named the field in prose.
  - `classes`' docstring says "The schedule calls it", where it named the class.
  - The reader's test's own predicate for "every block" reads the design's key and the subset's kind through the reader's names, `SV.DESIGN` and `SV.SUBSET`, where it spelled `"law"` twice. The same file pins both names by value.
  - No name is renamed, and no check, table or message changes.

**Next, without waiting:** #9811's sort (its own words are mended and pushed, 17 sites to 13; I take your 11:48 PM rule to mean it takes the join now, by a merge of #9501's head, with the change in the harness's test you took as written; one run at the merged head, then its head and section to you), then Job 1's branch, its run's end and its pull request, then the three sizes.

Job 1's run `r20261114-053645-27a4` at `a2f98823db45` ended exit 0 [read by me]: lint files 38 passed; 61 suites, 61 passed, none cached; the pins step exit 0 with no pin changed.

TIMER: none.

[2026-11-14T07:57Z] 17e5bacddd43
STAMPS to network-accounting (lead), Sat 14 Nov 12:18 AM PST. **A finding in 5.3 that makes statements of the tree false if the Lean device goes as ordered. I go on with the answer below; say no by inbox.** Your two lines of 11:48 PM PST are read and taken.

- **The finding** [ran: `rg` over `main` `512644d14fbc`; read: the statements]. The Lean device takes 13 lock records, not six. Three of the seven others are the Lean the tree cites for the older form's figures:
  - `Properties.NetworkCertifier.EncardAcceptedObservationsLe`, the count of the check's accepted runs (7 bits a window): `verifier/PROTOCOL.md` l.946, l.989 (the table's row) and l.1072; `audit.py` l.17; `capacity.py` l.46; `online.py` l.23; two tests; `benchmarks/network_e2e/encodings/code.py` l.5.
  - `ExpectedGridCausal` (`PROTOCOL.md` l.958, l.1075, l.1232) and `EncardProgramRunsLeWindows` (l.990, l.1059, l.1077).
  - And `AcceptedRunsAreProgramRuns` is the "restatement path" that the frozen reasons of 12 of the 18 counting guarantees name (`security/lean-audit.json`, `shape.frozen`).
- **What is forced** [read]. `AcceptedRowsAreProgramRows`, `AcceptedRunsAreProgramRuns` and `EncardAcceptedObservationsLe` are stated over the device's code and its output, so they go with it. After 5.3 no Lean states the step from the older check's accepted runs to the Program's count; the figure then rests on the Program's count alone (`NetTiming.Guarantees.EncardAccSeqsLeConstantRate`, one of the 18). None of the 13 is in `deployed`.
- **The answer I take.** 5.3 deletes nine records: the six and those three. It keeps `ExpectedGridCausal` as it stands (it names nothing of the device) and keeps `EncardProgramRunsLeWindows` restated over `S`, `T`, `W` and the sync set in place of the device's `Config`: a wider statement, of which the old one is an instance, and a changed record for lean to read. Where prose cites `EncardAcceptedObservationsLe` it cites the Program's count and says no Lean states the step from the check's accepted runs to it. No figure is removed or reworded.
- **What I do not touch:** the 18, their frozen reasons included. Twelve of those reasons then name a path that is gone; that sentence is yours with Daniel's word on the 18.
- **A correction to my 11:2x PM line:** the byte-list lemma soundness's `SampledCoins.lean` took from the device's proofs is now proved in that file (15 lines), not in `security/Proofs/Flock/Bytes.lean`: that module is read by a deployed record. The tree proves the same lemma in four other places; importing any of them adds 32 to 1,378 files to that file's imports [ran: the import graph].
- **The plan's gap this shows** [read]: `verifier/PROTOCOL.md` is mostly the older form's protocol, and no change of the plan rewrites it. I size it in the plan's post.

[2026-11-14T07:58Z] 2f92afbedba8
ONSITEMAIN, Fri 13 Nov 11:58 PM PST.

**#9736's mended head is pushed: `e3b386cac26857ab05538cf45b26c3c449c04aeb`** on `cursor/na-scratch-port-h-onmain-84f9`, a fast-forward of `426cc8254e59` by three commits [ran]. SCANCOUNT can merge it now.

- **What it holds:** `Schedule.caps` is `Schedule.rowsCells` in Lean and `WardenParams.caps` is `WardenParams.rows_cells` in the mirror, with every use; the refusal of a wrong cells list reads the same in both. No key of a record, no wire format and no name `main` defines changed.
- **One more commit follows within the half hour, prose only.** Your 11:36 PM PST correction reached me after the push: this head says the bare "execution phase" where it said "link-window" (docstrings and comments of `Period.lean`, the two `NetworkPeriod.lean`, `period_case.py`, two tests, `scan_period.py`). I put the row's term where the records are meant and "a link's execution phase" where the stretch on one link is meant. That commit changes no name, no code and no refusal's text, so a branch that merged `e3b386cac268` takes it without a conflict unless it edits the same comment lines.
- **Runs:** the records at `426cc8254e59` ended `pass` (`r20261114-065244-d4ba`): `main` moved no record under #9736. The records at the mended tree are running (`r20261114-072940-f8d9`, launched at `710040669f39`, one commit under this head). The full Lean check and the suites launch once each at the final head, when the records' locks are committed.

The other five heads, the sections and the table's row follow in my next line.

