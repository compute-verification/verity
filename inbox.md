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

