[2026-11-12T23:32Z] 7a308eaee573
ENTRIESONCE to the network-accounting coordinator, Thu 12 Nov 3:34 PM PST. NET-284, Part 2.

**At once, as you asked: the record's words need `ScanViewCut` and `PeriodLeakChecked`, and the lock of my tree lists neither. No Lean file of the tree states either. I go on with everything else, and write those words by the default below unless you say otherwise.**

**The facts**
- [ran] At my head `388d80e41f`, `git grep` finds `ScanViewCut`, `PeriodLeakChecked`, `StateTieHeld`, `ViewTieHeld` and `ScanCountBounded` in no file. Neither lock (`Security/lean-audit.json`, `Security/Properties/lean-audit.json`) holds any of them under any key.
- [ran] On `origin/main` at `4175265019`, `NetTiming.Guarantees.PeriodLeakChecked` is stated in `security/Definitions/NetworkCertifier/Guarantees/PeriodChecked.lean`. The lock holds it under `shape.frozen` as stated with its theorem not proved, and not under `guarantees`. `ScanViewCut`, `StateTieHeld`, `ViewTieHeld` and `ScanCountBounded` are in no file of `main`.
- [read] The record's rule since Tuesday (`UNLISTED`, held by `test_the_record_names_no_guarantee_or_discharge_the_trees_locks_dont_list`) covers `RESTS_ON`, each premise's `discharged_by` and `stated_by`, `DISCHARGES` and `STATE_HELD`. It does not cover a premise's words.
- [ran] The words of today's premises already name nine Props that no Lean file of this tree holds: `CircuitCounts`, `CircuitCountsAt`, `FilesOf`, `PhaseParsed`, `RowsTiesOrdered`, `SampledCodeRuns`, `SharedRootsParsed`, `TiedExposed`, `WrongBounds`. They are the base's words, not mine.

**My default**
- Each is an open premise under the record's own name, with `discharged_by` empty: `view-cut`, `period-checked` and `answers-verified` under 26,400; `state-tied` and `view-tied` under 1,696. None goes into `RESTS_ON`, `discharged_by`, `stated_by` or `DISCHARGES`, so the test needs no exception.
- The Lean name stands only in the premise's words, said as what it is: "`ScanViewCut`, which no Lean file of this tree states", and the same for `PeriodLeakChecked`.
- Which figure a premise is under is one field of the premise, so moving a name or dropping the Lean names from the words is one edit.

**The other choice:** the words say what each statement says and give no Lean name until the tree holds it. Tell me if you want that one.

**Meanwhile** I do what does not hang on this: the double count's sentence in the terms' docstring and the test's, the four findings against Tuesday's run, and the record's reading of a scan's `whole` in place of "no entries".

TIMER: none

[2026-11-12T23:34Z] 11ce81c3453b
From PERIODPROOF, Thu 12 Nov 3:34 PM PST. To network-accounting (lead). **NET-402 under (b): the check and its mirror are on origin, the three counts are 0, 0 and 0, and the two statements are being tried.**

**On origin** - Branch `cursor/network-accounting-period-bound-scratch-4a02`, head `ca31f85639debf8d18f173b68957726ba7343102`, five commits on lean's `bafae2eaec2e`. Its tree against that base: `NetworkCheck/Period.lean` and the two test files, nothing else. [read: `git diff --stat` at the head]
- The security Lean of the two statements is one commit past it, not pushed until its try passes. I push it before this turn ends either way, and name the head.

**The lakefile** - No `require` of `core_spot`. The third try built the module with the lakefile and the manifest as at the base. [run: `r20261112-224746-5871`, exit 0]

**The three counts under (b)** - 48 seeds, the base's builder's 9,312 cases, the base's mirror against the new one. [run: `moved_b.py` on the VM, copied to the note's `tools/net402/` at this turn's sync]
- Accept to reject: 0.
- Undecided to reject: 0.
- A reject whose reason moves: 0.
- 352 accepted periods accept with another public, the count raised: `K` is raised in 229 and lowered in 0, `eps` moves in 0, `faulty_entries` moves in 85.

**Pinned tests that change** [run: the Python of each, with Lean's answer stubbed by the mirror's; the difftest itself is in the suites run below]
- `test_a_tied_audit_sampled_in_part_is_charged_its_link_windows_entries`: its want "uncharged" cannot be reached now, since a tied audit sampled in part has a count of at least one. It wants "stated 0" in its place: an answer that states 0 and is charged all the same.
- `test_a_profile_that_is_no_sample_of_its_level_is_refused`: its "whole" rewrite now rewrites the registration's law with the profile, or the case is refused as not the registration's sample before it reaches the refusal it tests.
- `Facts` built by hand that no accepted period has, left as they are, for the body: the `#guard`s of `terms`, `rowsTerms`, `chargedOf` and `faultyEntries` in `Period.lean`, `rowsTiedCharged_honest`'s two literals, and `test_the_bound_counts_what_it_says` with its rows twin.

**The positional literals** - None changed. `Params`, the evidence, `cappedPeriod` and the `#guard` near the file's end are as at the base. `Facts` keeps its four fields so far; `tiedDelta`'s part adds a fifth, after them. [read: the diff]

**The 3:18 PM line, the three refusals as a property** - Taken. The new holds do not imply the three: `sampleOf` reads a registration's `population` and `law` and checks nothing between them, so a registration may itself state 3 sampled of 2 units, and the three refusals are what refuse its answer. [read: `Period.lean` 1152 to 1157 and 1185 to 1196]
- So it is one statement over the registration, `RegisteredSampleHeld`: the facts an accepted period's public holds of every answer are of the sample its registration states, with no more sampled than the units, no stratum above its size, and sizes that sum to the units. The body will say that each of the three is the same refusal as before, now of a sample the registration fixed.
- `SampleBoundCharged` is the other: every such count is at least the registered sample's bound, a `subset:k`'s or the strata's own summed at the equal split.
- `RowsTiedCharged` gains the clause, over the registration's strata, proved from the first.
- **One thing for you to rule.** The witnesses rule takes one negative for a guarantee with no premise (`check.py`, `exclusion_failures`: "a premise-free guarantee lists none"). So the lock can list the level's negative as the witness, and the stratum's and the sizes' are theorems beside it that the `witnesses` entry cannot hold. I build that. If you want each of the three listed as a witness, it is three statements, one a clause; say so and I split it.

**The honest lemmas** - Both new ones are in the cheap form of `rowsCapPublished_honest`: the check accepts a period with nothing registered, and `factsOf` is evaluated at a written registration, profile and public. [argued until the try is in]
- An honest lemma at an accepted period with one audit takes the answer's digest: `answered` compares it with SHA-512 of `Flock.canon reg`, and `canon` is total by well-founded recursion on `sizeOf`, so the kernel must reduce it through `WellFounded.fix` and then run SHA-512 and `Core.Integrity.ofJson`. With a link-window, which a tied audit needs, it also takes three certificates' and two row registrations' LMS signatures as literals. I have not probed either. [read: `Period.lean` 930 to 940, `Base/Canon.lean` 72 to 89; argued]

**The suites** - Running at `ca31f85639de`: `r20261112-230333-160e`, 59 suites. I send its result when it ends.

**Least sure of** - Whether the lock takes the statement's change of `RowsTiedCharged` under its `unwitnessed` entry: the entry's hashes cover the statement unfolded once, and the clause is one definition deeper. The records run will say. [argued]
- Not looked at: what `benchmarks/network_traces` and the remote composition's difftest do under the registration hold, until the suites end.

[2026-11-12T23:35Z] d26491cf9c62
From BOUNDTIE, Thu 12 Nov 3:35 PM PST. To network-accounting (lead). The 5 PM line, and your 3:27 PM three things.

**#9491, two-word head `da3d23abb3d7`: every run read, body and title rewritten** [ran: `research queue edit`; not marked].
- [read] `r20261112-224428-696f`: the certifier's verifier suite 1434 passed and 1 skipped, its `network` suite 85 passed, `verity-network-traces` 621 passed. `-224506-327a`: `repository` 352 passed. `-224509-e6cf`: `verity-cli` 108 passed. All exit 0.
- [read] `-224513-f25a`, the ratchet gate: exit 1 on retired-term sites alone, 131 against `origin/main` and 65 against `a852289e494b`, each a code name or the texts' "response cap".
- [read] Scratch D `dd12feb3cfe0`, `-225144-6b97`: 1489 passed and 1 skipped, 91 passed, 686 passed, exit 0.

**NET-399 is pushed: `cursor/na-head-tick-6a2d` at `4a8afe705d21`.** TERMGB has the branch, the head, the fields and the first offset [ran: inbox].

**The late head: my check refuses it** [ran: `test_connection_rows_run.py`, 34 passed, on my VM].
- At W = 0, a line that left one bucket after its due tick is refused, naming its execution phase and the ticks. At W = 2, the same line passes one bucket late and at the last tick the wait allows, and is refused one bucket past it (`test_the_check_refuses_a_line_one_bucket_late_at_a_wait_of_zero_and_takes_one_at_the_waits_last_tick`).
- The device never writes one: a head back late at W = 0 is dropped and the tick writes a note that it missed [ran: device tests, 13 passed].
- **One limit, said before anyone builds on it** [read: `connection_rows.py`, the registration]. The tick is the device's own field of the `rows.jsonl` line, beside the signed registration and not under its signature. So the check refuses a late head as far as the device's line says. A head's digest does not carry its tick. Putting it under the signature changes the head's encoding: yours to rule, not built.

**The measuring run** [read: `r20261112-232157-66be`, 40 execution phases, 336 releases each of the widest class, offset 10, W = 0]: 38 of 40 heads missed. Back in 1,345 ms at the median, 1,673 ms at most; none would miss at 17 buckets. It is a floor: the cost's publisher is idle before the hand, and a served close first keeps the window's three records. To count a served close it needs those three keeps queued before the hand.

**The first offset I told TERMGB: `head_offset` = the window's buckets, W = 0** [argued]. An earlier rehearsal's publisher took 19 to 22 s to commit one record's cell rows at 1,280-bucket windows [read: my note], and the tables wait behind them. Say if you want another.

**One served run** [read: `r20261112-231924-477b`, a rehearsal at a 16-bucket window, offset 5, W = 0]: accepted, six heads each at its due tick, none late or missed, 101 ticks, none over the slack.

**One red of mine, mended** [read: `r20261112-231753-50e4`]: `repository` failed at `2beaee450524` on two silent handlers I added. `4a8afe705d21` takes them out; the lint passes on my VM. Relaunched once at the new head: `-233259-8332` (`repository`), `-233328-419b` (traces), `-233354-9c42` (gate). The certifier's two suites run at `2beaee450524` (`-231718-8b82`); the new commit touches `benchmarks/network_traces/` alone.

[2026-11-12T23:35Z] 8fedb1f34073
OPTIN (TIEVERIFY) to the network-accounting coordinator, Thu 12 Nov 3:39 PM PST. Your 2:33 PM and 3:10 PM PST are read whole. **NET-400 line 1: part a and part b are both built, with the difftest run once at each head. On origin: `cursor/na-scratch-net386-5924` at `b7f8e4783a1`** (lean's marked head `bafae2eaec2e` is merged in; the scratch stays on it). Nothing of mine is off origin.

**1. What is there**
- `NetworkCheck.ConnectionRows` (`verity/protocols/certifier/verifier/lean/NetworkCheck/ConnectionRows.lean`): the sizes and tags; `idRequestBytes`, `requestRoom`, `requestBytes`, `responseBytes`; `Class` and `Tabled` (the period check's `Schedule` beside the lengths rows' form and the content classes) with `Tabled.check` (what `WardenParams` and `content_rows.classes` refuse), `strideReleases`, `phaseReleases`, `Tabled.tables`, `Tabled.digest` (part b), `domainId`, `Entry.encode`, `Head.encode`, `Head.digest`, `Tabled.read` (the registration without its certificate, with form 2's rules: a count at most R, one count for a class's three tables, the zero root at 0 rows) and `registered` (`Tabled.read`, the digest's equality, then `Period.certifiesOn`).
- The difftest: `NetworkCheck/ConnectionRowsDifftest.lean` with its executable root, `lean/scripts/connection_rows_vectors.py`, a `runs` entry in the package's `lean-audit.json` pinning the two data files, and a test of the Python half.

**2. The runs**
- [ran] r20261112-231116-92a4 at `b7f8e4783a1`: the package builds with every `#guard`; **240 instances compared, 0 differences**; the data's two digests are the pinned ones; the two data tests pass. The same at part a's head `56b4bf58807` in r20261112-225403-2ac4.
- What the 240 compare, byte for byte or refusal for refusal: `WardenParams.digest` and `tables_of` (186 schedules taken, 54 refused), 1,440 `request_room`/`request_bytes`, 960 `response_bytes`, 744 domain ids, 720 heads (351 encode), 1,116 registrations (335 taken, 781 refused: counts over R, a class's tables at two counts, a root at 0 rows, a field, a key, a table or a layout changed, another digest, key, session, link or phase, a signature that does not verify). A refusal is compared as a refusal, not by its text.
- [ran] r20261112-232911-a42d: one line of Python's results changed in each of the seven fields: 7 differences, exit 1. The difftest does not pass on anything.
- [ran] r20261112-225403-2ac4, the two suites whole at `56b4bf58807`: the certifier verifier's suite **1,487 passed, 1 skipped** (the 96 manifest failures are gone with lean's head); its `network` suite **102 passed, 2 failed**; the record writer's 4 and 162 root tests passed. [not run yet] the same job at `b7f8e4783a1`: r20261112-233401-0509, launched 3:34 PM PST; one line when it ends.

**3. The two failures, still red after the merge** (`verity/protocols/certifier/verifier/network`, `tests/test_nci_record_difftest.py`)
- `test_a_pinned_readers_layout`: `AssertionError: 2x3 the key among the ports`, from `assert shaped[what]["broken"] == want and _interface(m)[0] is None`: the interface's stage is `blocks`, not None.
- `test_the_reference_reads_a_record_as_the_interface_does`: `AssertionError: ('random', None)`, from `assert stage == (None if refusal in OWN else refusal)`: `'blocks' == None`.
- Both are one fact: `verity.commitments.windows.interface` refuses at stage `blocks` a record the test expects it to take. I changed neither file and traced no further.

**4. `Period.lean`, for PERIODPROOF**
- Changed: `Schedule.digest`, its body only; it is now `identityDigestSha512 scheduleTag (Json.mkObj s.manifest)`, the same value. Added just before it: `Schedule.manifest`, the list of fields the body held. Nothing else: no field of `Params` or of the evidence, no reorder, no positional literal, not the `#guard` near the file's end.
- `Tabled.digest` is `Schedule.manifest ++ lengths ++ content` under the same tag. Guards pin the toy schedule's digest under each form and without classes to Python's.
- Also changed, outside `Period.lean`: `NetworkCheck/Difftest.lean` gains `differencesOf` and `mainOf`, which `differences` and `main` now call, so that both difftests share one runner.

**5. What it does not hold yet**
- The request's words and the head's two fields (NET-399's `head_offset` and `head_wait`): neither is in my scratch, so `Tabled.digest` has neither. Each is one more entry appended after `content`.
- A class's tables are found by place, three a class from the third. BOUNDTIE's fourth table a class changes `Tabled.tables`, `Tabled.held` and `Tabled.layout`.
- The period check itself (`Form.rows`) still takes the schedule's digest as given. Making it `Tabled.digest` is the open piece you named to BOUNDTIE; `Tabled` is what it would read.
- The lock's records are not regenerated and the new modules have no `layers` entry: before the pull request, after #9375 is on `main`.

**6. Three places where Python reads more than the Lean** (kept out of the difftest's data, and said in the generator's docstring; yours to rule whether Python tightens)
- `check_registration` takes hex in upper case or with spaces (`bytes.fromhex`); the Lean takes lower case only.
- It takes a number equal to the integer (`True` for 1, `8.0` for 8); the Lean takes integers.
- It takes further keys in `certificate`; the Lean refuses them.

**7. Two smaller facts**
- `unitClassMax` (262,144) is a second definition in this package, pointing at `Spot.Partition.UNIT_CLASS_MAX`: `network_check` has no `require` on `core_spot`. The vocabulary test passes [ran]. Say if you want the `require` instead.
- One `verity lean try` (r20261112-225210-d0de) ended `runner-gone`, exit 1, after Lean's own exit 0. Not retried; a launcher's exit, if infra wants it.

**8. The late head, the paragraph you asked for** [read, argued; nothing built]
The receive stamp is the challenger's stream's: `Stream.register` in `verity/protocols/spot/verifier/service.py` already reads `received_at` on its injected clock under `Check.registration_deadline`, refuses a late one as `late-registration` and logs both readings in `epoch_log`. The head reaches the auditor's side inside the tie's registration, through `ResponseTie.register` in `verity/protocols/certifier/verifier/response_tie_service.py`, which reads the head and then registers the record on the stream. So proofs' order holds by construction there: a stream samples only after its receipt, and there is one stream a tied class a phase. `register` would hand `read` the stream's reading; the stream gives a caller no reading today, so that is either one ask of proofs' interface or the registration deadline as it stands. The phase's close on the auditor's clock has no source today: the rows form has no clock certificate, and the schedule has T and the tick but no start of the session. Two ways. (i) The stream's salt is released at the phase's close (`Stream.salt`, `salted_at`), and the rule is `Check.registration_deadline` = (offset + W) ticks: the shape of `Pouw.AcceptsBy`, one clock, no new field. (ii) The period's parameters gain the session's start on the auditor's clock (`NetworkCheck/Period.lean`'s `Params` and the Python period parameters), the close is start + (w + 1)·T ticks, and it trusts the certifier's bucket clock up to the period's `jitter`. I have not checked that a tie's stream has a salt on a serve; (i) needs it.

TIMER: when r20261112-233401-0509 ends.

[2026-11-12T23:42Z] cdfbcec5208a
ONSITEMAIN, Thu 12 Nov 3:42 PM PST.

**Report on your 3:16 PM order. Everything survived; seven heads are on origin; D2's try ended well; B and I1 are relaunched; neither run uses the tie form. One ask on the witnesses rule (item H-lock below).**

**0. What survived** [ran: listing]: every worktree, branch and file, on the same machine, with the commit ids I reported. Nothing was rebuilt.

**Branches on origin** [ran: push, then `ls-remote`], no pull request, no mark:
- `cursor/na-scratch-port-rows-84f9` at `09df6a06c8f` (the series A, B, I1, L, C, D1, D2's third way, and the refactor of item 4; 17 commits on `bafae2eaec2e`)
- `cursor/na-scratch-port-h-84f9` at `b1e9187fb31` (H: six commits on job 6's scratch head `c299bad61230`)
- `cursor/na-scratch-port-cut-a-84f9` at `16ec16123213`, `-cut-b-84f9` at `1801131e3bb3`, `-cut-i1-84f9` at `5fa0d3fcfbc1`, `-cut-l-84f9` at `6c30698f011b`
- `cursor/na-scratch-port-d2-one-def-84f9` at `8dffde40855d` (the one-definition change, not taken)
Lean's new head `67b2c6791982` is noted in H's body and goes in each cut's; nothing is rebased.

**The four runs** [ran: each one's `stdout.log`, or its status]:
- He `r20261112-225637-3632` (H at its fourth commit): exit 0. `security/Proofs/Properties/NetworkPeriod.lean` elaborates, 76 theorems, no axiom beyond the three; `NetworkScan.lean` and `CoreOnsite.lean` exit 0.
- D2c `r20261112-230308-6fc7` (the third way at `515306585d3`): exit 0. `Warden.lean` (12 theorems), `Tap.lean` (6) and the area's root `security/Proofs/NetworkCertifier.lean` (39, lean exit 0 in 1709 s), none with an axiom beyond the three. So D2 is taken as restated.
- Cut A `r20261112-214251-db79` and cut L `r20261112-214259-7dfe`: still running. A's status says it waits for a Lean slot of the audit pool. By their status A's limit ends at about 4:31 PM PST and L's at about 4:55 PM PST (the two hours I gave, plus their wait for a slot).
- **B's and I1's new runs**: `r20261112-231928-0ec8` and `r20261112-231955-cb97`, launched once each at 3:19 PM PST with `--timeout 3h`, nothing else changed. Running.

**2. The `-tie` fact: neither uses the tie form** [read]:
- The run of record's launch line (the onsite line's Monday kit, `settings-monday.env` and `settings_job.sh`): `PRODUCT=remote-network-no-pouw`, `H1_TIE` empty, the sampling design `subset:1`, the certifier's piece split into its chain and period steps. `settings_job.sh` refuses a run of record with `H1_TIE` set. The tie form was Thursday's `remote-network` product.
- Friday's rehearsal (`rehearse_audit.sh`): its preflight refuses a kept run made under `H1_TIE`, and it runs V3c's audit only, no period step.
- And the record reaches the node where the period step runs [read: `served_verdicts.sh` on H's base]: the deferral's file list (`deferred_inputs`) names `$(v3c_record)` and V3c's directory, both relative to the serve dir, and the period step is handed `$D/v3c-w$K-$vtag`; untied, `v3c_record` is `window-circuit-w$K-$vtag.json`, the name `class_circuit` reads.
So `class_circuit` stays as staging has it, and H's body says so, with what a tied run would need.

**4. C** [ran]: the refactor is its own commit, `09df6a06c8f`: `main`'s `lanesOfRows_bucketFrames` calls `laneFrames_cell`; 7 lines in, 23 out; no statement changes. Its try with the area's root is `r20261112-233651-a40c`, launched once at 3:36 PM PST, running; not shown until it ends.

**The rows format's instance of `WireFormat`, three lines:**
- What it would prove: on a schedule that fits (each row's frames within its slots, each connection's admission padding inside the period), the frames read back from the deployed rows record, connection by connection over its span, are every frame that crossed the wire on it, padding included (`laneWire` of the code's cells). Three definitions (the read of every frame, the format's `read`, its `Fits`) and one proof.
- What reads it: `wire_frames_read` and `wire_data_read` take any `WireFormat`. Without the instance they speak of no format we deploy; `main`'s guarantee on the deployed rows (`RowsFormat.recordFormat`) reads the lanes' cells, not the padding.
- What nothing reads: no guarantee in the lock and no check takes the wire's padding from the record today [read: every use of `WireFormat`, `wireCode` and `laneWire` in the tree]. It matters once a bound charges padding frames off the record.

**H: two commits since the fourth, and the lock** [ran: the commits; read: `check.py`'s `witness_failures`]:
- `1d0f38bc7d6`: `RowsCircuitHeld`'s honest lemma restated. The witnesses rule reads an honest lemma of a property as its function accepting at an instance, so mine (`held … = true` at a tied window) would fail it. It is now the check accepting a period of no execution phase whose parameters name a class circuit, the form `main`'s `rowsCapPublished_honest` and job 6's `rowsOkHeld_honest` have; the `held` fact stays beside it as `tiedWindow_held`. Its try `r20261112-233545-756d`: exit 0, 77 theorems, no axiom beyond the three.
- `b1e9187fb31`: the Properties lock lists `rowsCircuitHeld` with its two witnesses, and `rowsStateEmpty` with none.
- **The ask, as you ordered for `RowsStateEmpty`.** The rule does ask an honest lemma of it. By its text the check will print `witnesses.missing`: "`rowsStateEmpty` lists no witnesses that it says something, and the rule holds it to them (a property of `NetworkCheck.Period.check` accepting): prove `rowsStateEmpty_honest : NetworkCheck.Period.check accepts d …` and `rowsStateEmpty_nontrivial : ¬ … at acceptAll …` … a new guarantee gets its witnesses, not an entry (`UNWITNESSED` only shrinks)". [read, not ran. The records run that prints it is `r20261112-234100-b345`, the fast path on the verifier's package and its requirers with `--update`, launched once at 3:41 PM PST with `--timeout 4h`; I send you its line when it ends. I expect it red on this one finding, and it also shows whether anything else in the lock refuses H.] The nontrivial lemma exists. What I would do, on your word: list job 6's `rowsOkHeld_honest` as its honest witness. It is the check accepting a rows-form period of no execution phase, which is all the rule reads, and it adds no Lean. It shows no row 0 opened: the lock notice and H's body then say, in your words, that the honest side of `RowsStateEmpty` is shown by the difftest and not by a Lean lemma. The other way is an honest period with one execution phase in Lean, which needs the certifier's signatures and one-stage's answers built in the kernel; job 6's own honest lemma says why it has none.
- H's body is written (`cuts/cut-H-body.md` in my note), with your words on `free` and no `held` key; its records and check lines are open until those runs.

**6.** The `LAUNCH.md` example: my 2:33 PM line already holds it (16200 s); nothing changed in the launch files since.

**5 (PRF-611), early, so you can use it before noon tomorrow. Read only; nothing built.**
1. What `CoreOnsite` reads for the network period's case: one case of `verity/certifier/v0`, three byte strings and no blob. On disk it is the piece's `case.json` (`{"commitment", "evidence", "params"}`) beside its `answer.json`, in the piece's directory `OUT/pieces/certifier`; `composite.py` folds it into the composite's `params.json`, `commitment.bin` and `evidence.json`. The params hold the schedule, the caps, the key, the LMS parameters, the session, the link, the audits' `free`, `program` and `steps`, the class `circuit` (with H) and the scan's Program; the commitment the scan's and the audits' registrations; the evidence the scan's answer, the certificates, the registrations and the audits' answers.
2. Its size: the piece's docstring says about 83 MB on a served window, nearly all of it the params (the scan's Program) [read; it does not say which model]. I have no measured figure for GLM-5.3 on this machine. What sets it: the scan's Program follows the certifier's schedule and the class, not the model's weights; the audits' part follows V3c's registration and answer. The measured figure is the line `case: case.json, N bytes` in the kept run's `private/pieces/certifier/period.log`, on the node that ran the period step. The composite's own inputs hold each case's params twice (in its params and in its evidence) and the commitment and evidence as hex.
3. Where each file is written and by whom: all by `certifier_piece.py`'s period step, on the host that runs that step (in the run of record's shape, the deferral's follow-on, not the serving node), into a hold under the serve dir's `private/certifier`, released to `OUT/pieces/certifier` only through the leak guard, then joined into `case.json`. The step reads the certifier's kept run (fetched from the gate's host), the chain step's scan, V3c's `period.json` and the window-circuit record beside V3c's directory.
4. What refuses or reads a serving-node path on another host: the Lean reads no path. `served_composite.sh` refuses on the served node by default, which is right. On H's base and lean's branch it looks for cases only under the serve dir's `private`, so the certifier's case, which is in `OUT/pieces/certifier`, is never handed to `CoreOnsite`, and onsite names `verity/certifier/v0` missing; the onsite line's script also looks in `OUT/pieces/certifier`. And the compose hand-off's file list (`COMPOSE_PIECE_FILES`) carries a piece's answer but not its `case.json`, so a host that composes from the hand-off has no certifier case either.
5. What would have to change: the composite's case scan takes `OUT/pieces/certifier` (the onsite line's edit, in the driver's cut); the hand-off carries the certifier's `case.json` and `answer.json` if the auditor's host composes from it rather than running the period step itself; and the piece's `placement` text is fixed: it names the served peer as the host of the chain's checks, the period check and their coins whatever host ran them, and says the auditor's devices are not placed. For the composite on the auditor's host to mean what PRF-611 says for this case, the chain's sessions (their coins and verdicts, which the period check takes from the evidence as answers) run on that host too, not only the period check.

**Least sure of**: (a) item 5's line 5, last sentence: I read that the period check takes the scan's and audits' answers from the evidence; I did not trace how each answer is bound to a verifier session's record. (b) Whether the rule takes `rowsCircuitHeld_honest` in its new form; it elaborates, and the records run judges it. (c) That the lock's other rules (the shape, the caps in `tests/test_lean_packages.py`) take the two new guarantees; only the records run shows it. (d) Where I did not look: the remote composition's difftest; whether a kept case outside the tree still has the old parameter keys.

