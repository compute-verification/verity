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

