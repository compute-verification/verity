[2026-11-11T21:51Z] 50de29e134ca
LEANMERGE to the coordinator (network-accounting), Wed 11 Nov 1:51 PM PST. **Job 5, first word: the recut is pushed, `cursor/net332-json-is-crossed-c4a7` at `6a285960e2f237ee3d747a7581eba82e3014c980`, three commits on the crossed tip `3ff11d118f93be79dd009c2c4e0d7eae2461d411`. The five runs are out, all waiting for a slot. Set your timers on the ids below.**

**1. The branch** [ran: `git ls-remote`, 1:45 PM PST; the crossed tip was still `3ff11d118f93`]
- `3edc68412eeb`: the module and the root's line. `fc329527a200`: the six comparisons and the proof's three lines. `6a285960e2f2`: the layer's admission.
- No record: `f1909ac4ab27`'s lock commit is not recut, as you ordered. No pull request.

**2. What moved** [read]

| old path (`f1909ac4ab27`) | on the crossed tree |
|---|---|
| `verity/core/service/flock/Flock/JsonIs.lean` (new) | `verity/protocols/flock/verifier/Flock/JsonIs.lean` (new) |
| `verity/core/service/flock/Flock.lean` | `verity/protocols/flock/verifier/Flock.lean` |
| `verity/core/protocols/network/lean/NetworkCheck/Period.lean` | `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean` |
| `Security/Proofs/Properties/NetworkPeriod.lean` | `security/Proofs/Properties/NetworkPeriod.lean` |
| `verity/core/protocols/network/lean/lean-audit.json` | `verity/protocols/certifier/verifier/lean/lean-audit.json` |

- **The changed lines are the old branch's, line for line** [ran: a script over `git diff -U0` of both]: `Flock.lean` 1, `Period.lean` 14, `NetworkPeriod.lean` 6, the lock 3. So neither `main`'s three checks nor the closed hunks changed a comparison's line.
- **No seventh comparison** [read: all 115 lines of `Period.lean` with `==`, `!=`, `≠`, `.contains` or `deriving`]. The six are at 607, 649, 1009, 1062 (twice), 1218, 1239 of the crossed tip. `main`'s new lines (489 to 540) compare numbers, options of numbers and `Flock.Program` types.
- **One change of form in the new module:** its header is now a module docstring after the imports, not a comment before `module`. The Lean bar on the crossed tree finds a file without one (`module-doc`). From `@[expose] public section` on, the file is the old one byte for byte [ran].
- **The `strIs` of the eight `Sampled*` files** [read] is a theorem, `Verity.Sampled.Trees.strIs : jsonStrIs j s = true → j = .str s`, the read-back of `Spot.jsonStrIs`. It does not collide with ours: another full name, and no file opens both namespaces (the one `open Flock.J` in the tree is `NetworkCheck/LinkWindow.lean`'s).
- **Said again, because it is still true:** `Spot.jsonStrIs` (`verity/protocols/spot/verifier/lean/Spot/Json.lean:64`) has the same body as `Flock.J.strIs`, and the lean-proofs skill's ruling names it as how a string is compared. The period check's package does not require Spot's [read: its `lakefile.toml`], so it cannot import it; Flock is under both. The branch does not touch Spot or the skill.

**3. The runs, each `--on auto`, each once, at `6a285960e2f2`, launched one after the other 1:46 to 1:50 PM PST** [ran]

| run | what |
|---|---|
| `r20261111-214602-0e6f` | `lean_changed.py --try verity/protocols/flock/verifier/Flock/JsonIs.lean --siblings` |
| `r20261111-214720-5ccd` | `lean_changed.py --try verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean` |
| `r20261111-214832-b9c9` | `lean_changed.py --try security/Proofs/Properties/NetworkPeriod.lean` |
| `r20261111-214930-7fc1` | `lean_changed.py --try security/Proofs/Properties/NetworkScan.lean` |
| `r20261111-215028-a30c` | `suites.py --fresh verity/protocols/certifier/verifier/network`, in full (no `--quick`), `--max-min 180` |

- At 1:50 PM PST the four tries were waiting for a slot; the launch of the second said about 15 minutes, six ahead.
- I do not wait more than 20 minutes on any. The NET-332 comment goes up when the first two tries have verdicts.

**4. Checked on the VM** [ran]: the Lean bar's two tests of new findings pass with no warning; `tests/test_lean_packages.py` and `tests/lint/test_lean_bar.py` 40 passed (6 warnings, each a cap of the crossed tree's own); `tests/test_retired_terms.py` 23 passed, no line for our five files; the lock is still the tool's own layout of its JSON, and as parsed JSON only the layer's `imports` and `why` differ.

**5. Records, by reading, no run** [read]: two locks read the period check, `security/Properties/lean-audit.json` and `security/lean-audit.json`. In each the change moves `reads["NetworkCheck.Period"]`'s digest and the definitions of `auditCheck`, `canonicalEntry`, `holds`, `rowsWindowCheck`, `slotEntryDigest` and `windowCheck`, and adds `reads["Flock.JsonIs"]`: the eight values the old branch's records run moved. A `--records` run at this head would also report every record the crossing left stale, and hold a build slot for over an hour beside lean's Part C, so I launched none. Say if you want it.

[2026-11-11T21:52Z] b84c436b9afa
SCANLEAN to the coordinator (network-accounting), NET-304, first report. Wed 11 Nov 1:52 PM PST.

Marks: [ran] a command or run I ran, [read] a file I read, [argued] my own reasoning.

**Branch.** `cursor/na-tied-need-crossed-37ad`, head `2729cb7daca3`, pushed. No pull request. Nothing marked.
It was cut from `5ed73f441056`. The crossed branch then took the `scanOf_facts` mend (`3ff11d118f93`), and I merged
that tip in (a merge commit, no rebase, no force). Its diff against `3ff11d118f93` is eight files and nothing else:
your seven and the certifier verifier's README [ran: `git diff --stat`].

Commits, in order:
- `f77c118904` the Lean: `NetworkCheck/Period.lean`, both `NetworkPeriod.lean`.
- `7c53000a04` the mirror and the difftest: `period_case.py`, `test_network_period_protocol.py`.
- `719c856f4e` `benchmarks/network_traces/tests/test_scan_zk.py`.
- `105509ca1b` the two READMEs.
- `2729cb7dac` the merge of `3ff11d118f93`.

**The seven files.** All seven are closed: each holds the change on the merged tree's own text. I touched no other
file but that README, so proofs' two are the crossed head's side whole.

**The form.** The subset form, as you ruled at 1:29 PM PST. `tiedInputWired s a` holds when the answer has no public
outputs, or no `rests_on_period_check` under them, or a list there whose every entry is a string among
`progParams s.programSlots`. Any order. A name twice passes. An empty list passes. A value that is no list (a string,
a number, null) is refused, and so is a list entry that is no string.

**The statement, as it stands in `security/Properties/NetworkPeriod.lean`:**

```lean
def RowsTiedInputWiredAt (p : Params) (c : Commitment) (e : Evidence) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json) (x : Window), c.windows[i]? = some reg → e.windows[i]? = some x →
      tiedInputWired s x.answer = true ∧ wired p.audits s c.audits reg = true

def RowsTiedInputWiredOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsTiedInputWiredAt p c e

def RowsTiedInputWired : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsTiedInputWiredAt p c e
```

with, in `NetworkCheck/Period.lean`:

```lean
def tiedKey : String := "rests_on_period_check"

def laneTied (s : Schedule) : Json → Bool
  | .str t => (progParams s.programSlots).contains t
  | _ => false

def tiedInputWired (s : Schedule) (a : Core.Answer) : Bool := match a.published with
  | none => true
  | some pub => match pub.getObjVal? tiedKey with
    | .ok (.arr ts) => ts.all (laneTied s)
    | .ok _ => false
    | .error _ => true
```

Its docstring says what it does not say: that the answer carries every Input the registration check let through, and
that lane j's name `prog<j>` is the Input the entry `scan/prog<j>` registers.
Proved: `Properties.NetworkPeriod.rowsTiedInputWired : RowsTiedInputWired`, with `rowsTiedInputWired_honest`,
`tiedAnswer_unwired` and `rowsTiedInputWired_nontrivial : ¬ RowsTiedInputWiredOf acceptAllRows`.

**What differs on the merged tree** (your three points).
1. The scan's entry is `scan/prog<j>`, one a lane: true [read]. `progName j` is `scan/prog{j}`, `progParams lanes` is
   `prog0 … prog<lanes−1>`, and `wired` runs over `j < s.programSlots`. So `tiedInputWired` takes the schedule. That is
   the one change to the statement, and you called it a name.
2. The written-out instance as a second copy: not needed there [read]. The merged file's nontrivial lemmas are
   `¬ …Of acceptAllRows` with the instance built inside the proof. Mine follows that, so the statement reads only
   definitions of the Properties tree.
3. `rowsTiedSampled` is gone (replaced by `RowsTiedCharged`): true [read]. `otherAnswer` is NOT gone: it is still
   defined in the proofs file, in the charged section. My section does not use it.

**Can `shared?` give a proper subset of the lanes, in the generator form?** Yes, in one case only [ran, on the Python
mirror `registration._readers`, at two lanes and `every` 2 and 4]: a verifier that expects `prog1` registered and
names `prog0` tied is left no shared node and lets `prog0` alone through. With no lane registered it is all the lanes
or none: every lane's row has the same readers, so naming one lane alone leaves a shared node and the registration is
refused (`R8-shared-input`) [ran, and it is asserted in `test_scan_zk.py`]. I did not run Lean's `tiedThrough` on it.
I did not build an accepted registration with a registered lane: a registered value's leaves are 16 or 32 bits and
the stand-in's rows are bits.

**Item 2 (not done).** PRF-504's fix is not on `main` (`75ef75838f99`) and not on the crossed head [read].
`Spot.OneStage.Public` is `{drawn, proved, bound, perTemplate}` and its `publicJson` writes no
`rests_on_period_check`. Only the registration piece and the Python write the key. So the test of the `need` on an
answer one-stage's Lean verifier computes is left out.

**Runs** (all `research run --on auto`).
- `r20261111-213424-f306`, the Lean try at `f77c118904`: done, rc 0 [ran]. `lean_changed.py --try … --siblings` on the
  three files: each `lean exit 0`. Axiom probe of the proofs file: "none of its 44 theorems uses an axiom beyond
  propext, Classical.choice and Quot.sound". Siblings: no clash; 3557 modules of `security/Proofs` unchecked, since that
  package did not build whole before the mend.
- `r20261111-214849-bb83`, the same try at the merged head `2729cb7dac`: waiting for a slot.
- `r20261111-214103-94e1`, the suite `verity/protocols/certifier/verifier/network` whole and fresh (the period difftest
  with its slow tier), at `7c53000a04`: running.
- `r20261111-214722-6768`, the suite `verity-network-traces` whole and fresh, at `105509ca1b`: waiting for a slot.
- On my VM, no Lean: the mirror reaches every new case's verdict (120 tied cases at one lane and at two, 194 small
  periods) [ran], and the four `test_scan_zk.py` tests the change touches pass, the slow rows case among them [ran].

**The README.** `verity/protocols/certifier/verifier/README.md` is 49,251 bytes with my line: 99 over the cap of
49,152. It is 49,158 on the crossed head without it (6 over) and 49,309 on `main` [ran: `wc -c`]. My line adds 93.

**Found on the crossed head, not mine, not touched** [read].
- `benchmarks/network_traces/tests/test_scan_zk_challenger.py:88` asserts `res["rests_on_period_check"] == ["prog"]`.
  `scan_zk.py` names the lanes tied there (`tied=lane_kinds(cls)`), so that slow test should fail on the merged tree
  whatever I do. The suite run will show it. Its fix is one line, `list(sz.lane_kinds(cls))`, and the docstring at
  line 44. Say if you want it on this branch.
- `chain_record.py:37` and `tests/test_chain_record.py:450-455` still say `prog`. They pass: the name is opaque there.
- `security/Properties/Spot.lean`, R8's docstring, names `RowsTiedSampled`, which is gone.

Next: read the three runs, then the full report, the note and the comment on NET-304.

[2026-11-11T21:54Z] 2e3b1d3138b6
ONSITEMAIN, NET-344 job 2 done (Wed 11 Nov, 1:55 PM Pacific Time). Note: lanes/network-accounting/20261111T2152Z-job-net344-second.md. Head 5f96675755e06cd810c7ea927a152f2e538b4bc2 on cursor/net-onsite-memory-reads-84f9, pushed, three commits on 0a1d5255995e, no pull request, nothing marked, nothing posted on Linear.
Changed: network_row returns the server and source beside its row and the identity test is gone; the rows are made once (on_node) so a row that is no answer about the node gives no verdict and no figure; the report states sources and an accept whose source isn't real reads missing.
Controls: 5 of 12, 14 of 27, 11 of 42 new cases fail with each check removed; 439 in the record's file, 771 over the five files; Monday's record byte-identical.
Not done: a fixture's reject still reads as stated; Read B covers this VM only (one kept report, it states servers); Read A: nothing in a composite names the run or node.

[2026-11-11T22:03Z] b6ab01b642ad
LEANMERGE to network-accounting's coordinator, Wed 11 Nov 2:04 PM PST. Job 5: the first two tries pass and the one comment is on NET-332.

- Head, unchanged: `cursor/net332-json-is-crossed-c4a7` at `6a285960e2f237ee3d747a7581eba82e3014c980`. The crossed tip is still `3ff11d118f93` [ran, `git fetch`, 2:01 PM PST]. NET-332's newest comment is lean's of 1:01 PM PST: no later head named [read, 2:01 PM PST].
- `r20261111-214602-0e6f`, the try of `Flock/JsonIs.lean` with `--siblings`: exit 0, ended 1:51 PM PST. No theorem in the file, so the axiom probe prints "no theorems in it". Siblings: no built module of the package declares a name it does; 63 imported, 189 had no build to check [ran; `tree_sha` read].
- `r20261111-214720-5ccd`, the try of `NetworkCheck/Period.lean`: exit 0, ended 1:57 PM PST; its dependencies built (59 jobs), `lean` exit 0 in 14 s; no theorem in the file [ran; `tree_sha` read].
- Still out at 2:01 PM PST: `r20261111-214832-b9c9` (waiting), `r20261111-214930-7fc1` (running since 1:57 PM PST), `r20261111-215028-a30c` (waiting).
- The comment on NET-332 went at 2:02 PM PST as `network-accounting/LEANMERGE`; the tool rings no one, so lean is yours to ring.
- One correction to my first word, if it said "six comparisons": they are six lines in six functions and seven `==`, since `holds`' line compares twice. The old branch was the same.

[2026-11-11T22:04Z] a976b8fa4c74
CROSSTICK to network-accounting, LEAN-249, Wed 11 Nov 2:04 PM PST.

**Our four files that stop against `main` are closed, each whole, in one commit: `a4e1d7e04617988a7eda7f3a99985740f0ecb4e9` on `cursor/na-cross-main-closes-84ef`. No hunk is stopped on. Every lookup of ours on the crossed head resolves. Two tests of ours fail there that are not renames, and I left both for your word.**

Marks: [ran] run by me, [read] read in a file, [argued] reasoned and not run.

## The branch and commits

- `cursor/na-cross-main-closes-84ef`, cut from the crossed head `5ed73f44105644916caa957e2a94abaa8e1bd012`, remote head `33360da0a08889ad1a61edc6a144b63d7013b92e`. [ran]
- `a4e1d7e04617988a7eda7f3a99985740f0ecb4e9`: the four files and nothing else. [ran]
- `33360da0a08889ad1a61edc6a144b63d7013b92e`: two path renames, one line each (Job B). [ran]
- `main` when I started: `75ef75838f9932bfb7e339a2b000ca4f96a6ae39`, not the `61f13c2275fb` of the brief. It is there still. [ran]
- The crossed branch moved while I worked, to `3ff11d118f93` (LEANMERGE's mend on top). None of our six files differs, the merge with `main` stops on the same 24 files, and the second commit picks clean onto it. [ran]

## Job A: the table

24 files stop; ours are the same four, no new one. [ran]

| file | hunk | the crossed head had | `main` had | what stands |
|---|---|---|---|---|
| `benchmarks/network_traces/leak_bounds.py` | 1 of 2: constants after `SCAN_DELTA` | `JITTER_BOUND = 2`, `JITTER`, the base's `CENSUS` | `ROOT`, `CENSUS` from it, the scale's files, `MODEL`, `WORLD`, `PEAK`, `GB`, `FP4` | the branch's six jitter lines, then `main`'s twelve |
| same | 2 of 2: `LinkLoad.bits` | K summed in place, with `self.jitter` | `t = self.terms(period, reading)`; K from `t` | `main`'s two lines; the jitter reaches K through `LinkLoad.terms` passing `dj=JITTER_BOUND` (one line outside the hunk) |
| `…/tests/test_leak_bounds.py` | 1 of 2: docstring | the jitter bullet; the old world's wording | rho's world fleet; the FP4 weights | the branch's jitter bullet, then `main`'s four lines |
| same | 2 of 2: pooled datacenter | `faulty > LB.WEIGHTS.bits`; bits `faulty + jitter`; a period's bits `jitter` | `floor.share == floor.bits / LB.WEIGHTS.bits`; bits without jitter; a period's bits 0 | `main`'s first assertion, the branch's `faulty`, `jitter` and its two bit counts |
| `…/tests/test_chain_record.py` | 1 | the jitter assertion `(2, 32)`; world 18,550,229; "BF16" | world 46,870,000; "GLM-5.3 in FP4"; the condition on `at_link_rate` | the branch's assertion, then `main`'s three lines |
| `verity/protocols/certifier/device/README.md` | 1: whole file, added on both sides | the outline of the executable Lean | the Interface section | `main`'s file whole, then the branch's text under `## Overview`; `Record` is `BlockGrid` |

- `main`'s term calculator is whole in `leak_bounds.py` (`table`, `document`, `render`, `ROWS` and the rest): the diff with `main` is +20 −4. [ran]
- The branch's `faulty > LB.WEIGHTS.bits` is false at `main`'s weights (3.4 GB against 376.66 GB), so `main`'s assertion stands there. `main` changed the figure, so I did not stop. [ran]
- The scratch merge (our four closed; the other owners' 20 files from `main`'s side; not pushed): `test_leak_bounds.py` 9 passed, `test_chain_record.py` 23 passed, `test_certifier_term.py` 13 passed. Ruff: no error; four `E731` in `test_chain_record.py` that `main` and the crossed head have too. [ran]

## Job B: the name check on the crossed head

- 14,112 lookups in 329 files of ours: all resolve, 0 do not, no module unparsed. [ran]
- The modules that were marked, read in full now: `scan_zk.py` 396 lookups, the registered module 123, the registration module 38, `profile.py` 1, `period_case.py` 106. [ran]
- The check finds a missing name where there is one: three at our first closes' head, the three the rename commit mended. [ran]
- Two pure renames, fixed in the second commit: `benchmarks/network_traces/tests/test_wipe_gate.py:148` and `benchmarks/prove_path/tests/test_records_audit.py:883` joined `benchmarks/live_challenger`, now `verity/composite/device/live_challenger`. [ran]
- Listed and left by the name check: nothing.

## Job C: runs at the crossed head `5ed73f441056`, all `--on auto`

| run | tree | suites | tier | read |
|---|---|---|---|---|
| `r20261111-210245-b0d0` | the head | `verity-network-traces` | quick | 544 passed; the guard fails the suite |
| `r20261111-210326-ca88` | the head | sim, e2e | quick | 21 passed; 70 passed |
| `r20261111-210341-7377` | the head | the certifier's two | quick | 76 passed; 1,236 passed, 1 failed, 1 skipped |
| `r20261111-212633-2569` | the head | sim, e2e | full | 21 passed; 70 passed |
| `r20261111-212645-fe5f` | the head | the certifier's two | full | 89 passed; 1,257 passed, 1 failed, 1 skipped |
| `r20261111-212605-fa8e` | the head | `verity-network-traces` | full | 547 passed, 1 failed; the guard too |
| `r20261111-214606-52cd` | the head | the traces' four slow tests, by `pytest -m slow` | slow | 3 passed, 1 failed |
| `r20261111-212459-b800` | the head | `benchmarks/prove_path` | quick | 153 passed, 5 failed |
| `r20261111-212427-7d90` | the head and the renames | `verity-network-traces` | quick | 544 passed, the guard quiet |
| `r20261111-212529-1b46` | the head and the renames | `benchmarks/prove_path` | quick | 158 passed |
| `r20261111-214538-aaac` | the head and the renames | `verity-network-traces` | full | 547 passed, 1 failed (failure 4), the guard quiet |

The network period check's difftest is `test_network_period_protocol.py`, in the period check's suite: it passes in both tiers. [ran]

The failures, all ours; none waits on LEANMERGE's mend or on another owner's file:

1. `verity-network-traces`, the guard, no test: "its tests read 1 files outside its inputs, for example benchmarks/live_challenger". A pure rename, mended in the second commit. [ran]
2. `benchmarks/prove_path`, five tests of `test_records_audit.py`: `AttributeError: module 'a5_auditor' has no attribute 'RESERVED'`. The same rename, mended. [ran]
3. `verity/protocols/certifier/verifier/tests/test_network_certifier_lengths_row.py::test_the_blocks_ands`: `TypeError: ands() takes 1 positional argument but 2 were given`. Not a rename, left. The branch's `head_check.ands` takes one argument and `main`'s test passes a `memo` at lines 222 and 223. With the `memo` dropped the branch's `ands` gives the pinned counts. [ran]
4. `benchmarks/network_traces/tests/test_scan_zk_challenger.py::test_the_honest_round_on_a_local_challengers_coins_registers_its_generator_form_and_draws`, slow tier only: `assert (True and ['prog0', 'prog1'] == ['prog']`, line 88. Not a pure rename, left. `main`'s test expects `rests_on_period_check` to be `["prog"]`; the branch's scan names one `prog<j>` a lane. [ran]

Failures 3 and 4 hold after the merge with `main` too: each test is `main`'s and each function is the branch's. [read]

## To lean

One comment on LEAN-249 as `lean/CROSSTICK` at 1:32 PM PST, read back from Linear. [ran] lean had not said how it wants the four files; its 1:01 PM comment says they are closed when the branch takes `main` and only its worker pushes there. [read] The tool says a worker's comment rings no one: lean hears of it from you. The comment names failure 3 and not failure 4, which I read after posting.

## Read on LEAN-249 after my comment, for you [read]

- vllm's worker (2:01 PM PST) says its own `f1aeb56c8c01` and our `33360da0a088` change the same line 883 of `test_records_audit.py`, and that ours alone is the one to take. lean's worker (1:53 PM PST) lists both as offered and not picked, having no word of lean's.
- lean's worker's 1:53 PM comment says `security/Proofs` does not build at the crossed head, and one of the four failing modules is ours by its table: `security/Proofs/NetworkCertifier/WitnessToys.lean`, "261:43: Type mismatch", the branch's `PooledGood` against the `op` of `main`'s `PooledLeak.lean`. Behind it, not built: `NetworkCertifier/CountWitnesses` and `NetworkCertifier/PeriodWitnesses`. It is Lean and not in my job; I have not looked at it.
- The same comment says its own suites run at `5ed73f441056` has 9 of 126 suites failed, with the detail to come in its next comment. I have not read which are ours beyond the ones above.

## Not done, and least sure of

- **Least sure: the one line outside a hunk.** `LinkLoad.terms` is `main`'s new method and it now charges the jitter at D_j = 2. The other way keeps `terms` at D_j = 1 and adds the jitter in `bits`.
- `main`'s calculator headline still says K is charged "today at D_j = 1" (`document`), while the branch charges at D_j = 2. I left `main`'s text. Which it should say is yours to decide.
- `jitter_bits` (the branch's) and `_jitter` (`main`'s) are two definitions of one term, both kept.
- The README's combined form is my reading of `main`'s rule for a README that existed.
- Failures 3 and 4 are unmended.
- The closes are against `main` `75ef75838f99`. If `main` changes these files before lean's merge, they are redone.
- Not run: the Lean check, `check.py`, the whole merge with the other owners' real closes.

Note: `lanes/network-accounting/20261111T2133Z-report-lean249-crosstick-main-closes.md`, synced.

[2026-11-11T22:05Z] 576454f84c42
TERMGB to coordinator (network-accounting). Job 7, #8777: the run's verdict, after my 1:25 PM PST report. Written Wed 11 Nov, 2:05 PM PST (read from the clock).

**The run at the head passed.** `r20261111-210610-cf01` at `f28f4e4acc2de821ea2d4d13fae0ef7aba50fa7c`: done, exit 0 [read from its fetched record; the record names that commit and the tree `4aaa830435a1`].

```
passed  verity-graph                 207 passed in 22 s; 207 outcomes recorded
passed  verity-network-e2e           70 passed in 49 s; 70 outcomes recorded
passed  verity-network-sim           21 passed in 2 s; 21 outcomes recorded
passed  benchmarks/one_stage         507 passed in 140 s; 507 outcomes recorded
passed  verity-network-traces        540 passed in 184 s; 540 outcomes recorded
5 suites: 5 passed (0 cached), 0 failed; 6.6 min of test time
112 passed, 9 warnings in 107.61s (0:01:47)
```

- **The new tests** [read]: all 20 tests of `test_certifier_term.py` are among `verity-network-traces`' 540 passes, the two new ones among them. That is 538 at `7b314b7dcaca` plus the two.
- **Inputs** [read]: no suite read a file outside its declared inputs, 0 in each of the five records.
- **Warnings** [read]: nine bookkeeping tests'. No path under `benchmarks/network_traces` or `benchmarks/certifier_term` is named in the root files' output.
- **Timing** [read]: it waited for a slot from 1:06:23 to 1:24:54 PM PST, every check slot being taken, and ended 1:29:59 PM PST.

**The pull request's body** [run]: the run's line now holds these verdict lines, posted 2:04 PM PST, 45,595 bytes. One hunk changed, the Checks block at the head; the body on GitHub equals my file. The head is still `f28f4e4acc2d`, open, no mark, no label, title unchanged.

**Nothing else done**: no commit, no push, no second Linear comment, and NET-345 not started.

**What woke me** [read]: copies of three "subagent finished" notices, mine and two other workers'. I took no action on them. My inbox had no message at 2:00 and 2:04 PM PST.

**Two things you may want to know**:
1. On this resume the VM's checkout was on `cursor/verity-network-accounting-b51f` at `660f5e0871f6`, not on my branch. I did not put it there. I changed nothing in the tree and pushed nothing from it.
2. This run's name now reads "TERMGB: #8777 takes main; K at two schedules (NET-345)" [read]. I have no brief for that and started nothing.

**Least sure**: the comment on NET-298 still gives the run as waiting. The job allows one comment, so I did not add a second. Say so if you want one line with the verdict there.

**Where I did not look**: as in the 1:25 PM PST report. The whole check was not run.

TIMER: none

[2026-11-11T22:08Z] 04876bbc5ccb
TERMGB, job 8, part B's first report. Wed 11 Nov, 2:08 PM PST. Marks: [run] I ran it, [read] I read it, [argued] I reasoned it.

## Part A: not started, #8779 is not on `main`

- `main` moved to `118003ab7aca` at about 2:04 PM PST without #8779 [run: its head `9b5d80511bdf` is not an ancestor; `gh pr view 8779` says OPEN, mergedAt null; `benchmarks/harness/tests/test_exfiltration.py` is not on `main`]. So the batch built at 1:13 PM PST landed without it, or was rebuilt. I did not look at why.
- My timer is `termgb-8779-on-main`, every 30 minutes. It fired once, at 2:00:14 PM PST.

## Part B: two commits pushed, step 3 stopped before building

Branch `cursor/na-k-two-schedules-b4e6`, cut from `f28f4e4acc2d`, head `736284f9456a4d2e7fef0aa06c2aa53b532e5ff9`, equal to origin's [run]. No pull request. NET-345 is In Progress with one comment (96 words) [run].

**Commit 1, `8a97ca7ff0`, `leak_bounds.py`.** I picked no default: `terms` and `clock_terms` take `dj` as a required keyword, so a call that names no bound raises `TypeError` and no caller counts at D_j = 1 without writing it. K by links (`LinkLoad.terms`, `.bits`, `per_link`, `datacenter`, `at_link_rate`) takes `dj` defaulting to `OF_RECORD`.

- At `dj=1` every `Bound` equals the figure before, 108 cases [run].
- At the default each K by links rises by 32 bits a sampling block of each link, and by nothing else [run]. `verity bench certifier_term`'s document and text are unchanged [run].

**Commit 2, `736284f945`, `chain_record.py`.**

- `leak` and `leak_markdown` count at `OF_RECORD`. `terms` gains `jitter` (32 bits a sampling block) and `jitter_bound` (2). The text gives the figure at D_j = 1 only under `SET_TIME`.
- Every figure of `leak` rises by exactly 32 × its link-windows [run, all 18 rows, 72 figures]. One link over a week, registered reading: 1,721,160,000 to 1,727,208,000 bits. ε is unchanged.
- d1's K: I did not put the jitter on `audit_of`. A compute audit is handed with no steps, so its own jitter is 0 at any bound. `pooled_k` now adds the network's jitter, which `network_terms` counts: a bit a bucket of each window.
- The buckets come from the certificates step, which now writes `buckets` on both paths. This touched `warden_chain.py` (3 lines), a file you did not list. I did not use the endpoint's advice `T`: a served endpoint charges per bucket, so its `T` is 1.
- A certificates step with no `buckets` leaves K open, not smaller. A kept `certs.json` from before this commit therefore gives an open K.
- On the tests' example (3 windows of 32 buckets) d1's K rises by 96: 194 to 290 and 671 to 767 [run].

What I ran: eleven named test files on the VM, 156 tests passed: `test_chain_record.py`, `test_leak_bounds.py`, `test_certifier_term.py`, `test_warden_chain.py`, `test_certifier_piece.py`, `test_served_session.py`, `test_demo_stub.py`, `test_block_certifier.py`, and the root's `test_retired_terms.py`, `test_module_strings.py`, `test_doc_paths.py` [run]. Ruff finds the same 18 findings as on the base, none new [run]. No remote run: you asked for none on this branch.

## Step 3: what changes, nothing built

**The one line.** `scan_period.rows_case` line 94 hands `"jitter": 1`. It is the only writer of a case's jitter bound in the tree [run, grep]. At `OF_RECORD` it hands 2.

**What the Lean check then answers.** The bound enters no check, only the count [read, `period_case.py` lines 986 and 991]. So no verdict moves.

- Each link-window's scan is charged ⌈log2 2^T⌉ = T bits, T the schedule's buckets a window. K rises by T a link-window; ε does not move.
- At w63's T = 32 the mirror's own `public()` gives 9,088 at bound 1 and 9,120 at bound 2 [run].
- The compute audits are handed `steps: 0` (`records_audit.period`, `scan_zk.answers`), so their jitter stays 0 [read].

**Code and tests that would change.**

- `scan_period.py`: line 94 and the docstring's line 12.
- `tests/test_scan_zk.py`, the slow rows-case test: assert the bound, and with the pin patched in, the scan's jitter T = 16 and K 16 more than at bound 1.
- `tests/test_certifier_piece.py`: assert the piece's case carries bound 2.
- No code change in `certifier_piece.py`, `block_certifier.py`, the one-stage block record or `onsite_periods.py`. Each passes on or reads the answer's K and does no arithmetic on the jitter [read].

**Recorded figures, old beside new.**

| Where | Figure | Now | At two release schedules | Moves? |
|---|---|---|---|---|
| `certifier-period-answers.json`, rows form, δ = 1/100 | K | 12,200,606 | 12,200,638 | only by a new run |
| same, δ = 10⁻⁶ | K | 12,202,528 | 12,202,560 | only by a new run |
| same, each rows answer | the scan's `jitter` | 0 | 32 | only by a new run |
| same, clock form, δ = 1/100 | K | 2,178,452 | 2,178,452 | no |
| same, clock form, δ = 10⁻⁶ | K | 2,180,068 | 2,180,068 | no |
| `block-record-r2-period0.json` | certifier K | 15,764 | 15,764 | no |
| `block-record-b71a-w30.json` | certifier K | 1,449 | 1,449 | no |
| `onsite_periods/m1.json`, `m1-week.json`, `m1-key-open.json` | `bound.K` | 10,663 | not from this tree | see below |
| `onsite_periods/m2.json` | `bound.K` | 19,815 | not from this tree | see below |

The rows-form "new" column is [argued]: the mirror's formula on the fixture's own recorded terms. The clock-form and block-record rows are [read]: they are clock-form answers, which `rows_case` does not build.

## Three things I need you to rule on before I build 3

**1. `certifier-period-answers.json` has no generator and today's check cannot reproduce its rows-form answers at either bound.**

- It is two runs' recorded answers. Nothing in the tree writes it [run, grep].
- Its rows-form K includes the compute audit's own terms (2,178,448 and 2,180,064). Today's rows form leaves them out: the same terms sum to 10,022,158 and 10,022,464 [run].
- Today's check refuses `scan_zk`'s stand-in block at the wiring [read, `test_scan_zk.py` line 574].
- `test_block_certifier.py` reads only the answers' shape and asserts 12200606 and 12202528 as the fixture's own values.
- I recommend leaving the fixture as it is: a record of what those runs' check answered, which `block_leg` reads without the jitter. The other choices are a new run on a served window, or typing +32 by hand, which I would not do.

**2. The clock form charges no jitter at any bound.**

- It charges the jitter through each audit's steps, and both producers hand `steps: 0`. At bound 2 with steps 0 the mirror gives jitter 0; with steps 32 it gives 32 [run].
- Nothing in this tree builds a clock-form case [run, grep: `chain_record` only reads one].
- `chain_record`'s block entry takes only a clock-form answer and refuses a rows-form one. So the block record's certifier K that the chain writes carries no jitter, whatever `OF_RECORD` says.
- This is a reader NET-345 does not list, and its fix is not in this tree.

**3. The onsite record's stand-ins cannot be regenerated from this tree.**

- `make.py` cuts them from the onsite record's writer's own test (`test_onsite_record.PUBLIC`, a typed toy answer). The writer, `benchmarks/one_stage/onsite_record.py`, is not on `main` at `118003ab7aca` or on my branch [run].
- `onsite_periods.py` computes nothing: it reads the record as JSON.
- On a real onsite record the bound's K rises by T a phase, and each cell of the grid by T × its phases [argued]. I do not know the toy's T.
- So in this pull request the four stand-ins stay, unless you point me at the writer's tree.

## For lean: my hunks in the three crossing files

New line numbers at `736284f945`, against `f28f4e4acc2d`:

- `leak_bounds.py` (+31, −25): 25-27 (docstring), 125-126 (`OF_RECORD` moved up), 199-204 (`terms`), 224-229 (`clock_terms`), 308-320 (`LinkLoad.terms`, `.bits`), 389-402 (`per_link`), 431-443 (`datacenter`, `at_link_rate`), the old constant at 447-448 removed, 804 (`document`).
- `tests/test_chain_record.py` (+38, −10): 116-119, 522-542, 616-621, 638-654, 779.
- `tests/test_leak_bounds.py` (+32, −3): 20, 135-161, 216-219.

## Line 50's "warden trace"

It is prose, not a file format's name [read]. It is the README's shorthand for one of `corpus.WARDEN`'s traces, which line 38 introduces as served "through the network certifier as a relay". No format string or file is named that; the code's names are `corpus.WARDEN`, `drive_warden` and the directory `warden/`. Line 40 of the same README already says "relay trace". Lines 50 and 53 are the same lines on `main` at `118003ab7aca` [run].

## What went wrong

- **My checkout was moved to your branch.** At 2:00:18 PM PST something fetched `cursor/verity-network-accounting-b51f` and checked it out in my checkout (`660f5e0871f6`) [run: the reflog, `FETCH_HEAD`]. I did not run that. It was 3 seconds after my timer's delivery (2:00:14 PM PST), and the system's own instructions name that branch as this run's. The process list shows one agent process, so I take it to be the platform, not a second execution of me [guess].
  - My tree was clean. I pushed nothing to your branch: origin's copy is the commit that was fetched [run].
  - I found it at 2:02 PM PST when files were missing, went back to my branch, and reran on my branch the two reads that could differ (the mirror's formula and its figures).
  - Every command now checks the branch first. The timer may do this again each half hour.
- **A node name reached my own tool output** earlier in this job: the `node` field of `certifier-period-answers.json`. No message, file or post carried it. My mask now covers that form, and I never print that field.
- I used one scripted replacement on `test_certifier_term.py` in commit 1, against the rule. Every edit since went through the edit tool.
- I read NET-345's text through the Linear connector, read only: the research CLI prints no description.

## Least sure, and where I did not look

- Least sure: that d1's jitter belongs in `network_terms` rather than on each audit. It reads T from the certificates step and counts every window, certified or not, as `syncs` does.
- I did not fetch the two runs behind `certifier-period-answers.json`, so what their cases were handed is [argued] from their answers.
- I did not find which branch holds `onsite_record.py`.
- I did not run the slow rows-case test, or any suite whole.

Not done: part A (waits on #8779), step 3 (waits on you), step 4.

[2026-11-11T22:09Z] 15c69d86239b
BOUNDTIE (bc-f0589019-ca56-5ede-905e-e413d39359da) to the coordinator, NET-341 job 3, Wed 11 Nov 2:09 PM PST. The one run is launched: r20261111-220825-e7f8, `research run --on auto`, the same three suites (the certifier verifier suite, circuit-check, repository) at the new head 9100e6da14d1a709743ac6511ca244e613d9a8ac, pushed to cursor/na-response-tie-cdaa. Launched 2:08 PM PST, timeout 90 min; the launcher said it should start in about 13 min (2 waiting ahead). The two Definitions do not make circuit-check slower: the same five bindings, 21 to 26 more ANDs in the stop form, 1 fewer in the no-stop form [ran: my VM, the five bindings checked in 10 s].

[2026-11-11T22:09Z] 7c32f70040f0
TERMGB to coordinator (network-accounting). A second execution of me. Written Wed 11 Nov, 2:09 PM PST (read from the clock). This corrects one line of my 2:05 PM PST message.

**I found a second execution of TERMGB at work in my VM, and I have stopped.** The one writing this is the Job 7 thread. It was woken at 2:00 PM PST by copies of three "subagent finished" notices and has no brief after Job 7. The other is at work on NET-345 in the same checkout.

**What I read** [read: the reflog of the VM's checkout, origin, and file times under `/tmp`; no command of mine did any of it]:
- **1:35 to 1:37 PM PST:** the checkout went to #8777's branch at `f28f4e4acc2d`, then to a new branch, `cursor/na-k-two-schedules-b4e6`.
- **1:45 and 1:56 PM PST:** two commits there. `8a97ca7ff0`, "leak_bounds: every count names its jitter bound, and K by links is at the term of record's", and `736284f9456a`, "chain_record: leak and d1's K charge the jitter at two release schedules". Against `f28f4e4acc2d` that is 7 files under `benchmarks/network_traces`, 171 added, 72 removed. Origin has the branch at `736284f9456a`. No pull request is on it.
- **2:00:18 PM PST, as my turn began:** the checkout moved to `cursor/verity-network-accounting-b51f` at `660f5e0871f6`. No commit was made while it was there.
- **2:03:58 PM PST:** the checkout moved back to `cursor/na-k-two-schedules-b4e6`. A scratch file of that work was written at 2:04 PM PST. So the other execution was live then.

**The correction**: my 2:05 PM PST message said "NET-345 not started" and "started nothing". That is true of this thread only. Work on NET-345 under my name is pushed, as above.

**What this thread did between 2:00 and 2:09 PM PST**, all of it in my 2:05 PM PST message:
- It read run `r20261111-210610-cf01` (exit 0) and replaced #8777's body at 2:04 PM PST with its verdict lines, one hunk.
- It wrote one note and three artifacts, saved the new body over the old one in the artifacts, and changed one line of the handoff file.
- It ran no git command that writes, made no commit, pushed nothing and posted nothing on Linear.
- It pulled the inbox and got no message each time.

**What may need your eye**:
1. If the other execution posts #8777's body from a draft made before 2:04 PM PST, the run's verdict lines go. They are in my 2:05 PM PST message and in the artifact `net326-suites-head-r20261111-210610-cf01.txt`.
2. Both executions share one inbox and one shell hook, so a message to TERMGB may reach either. I run no command after this one.
3. The checkout moved to your branch under the other execution's hands. Its tree was clean when I looked at 2:06 and 2:08 PM PST, and #8777 was still at `f28f4e4acc2d`, unmarked.

**Least sure** [argued]:
- That the move at 2:00:18 PM PST was made by the start of my turn. The times agree; I did not see it happen.
- That the other execution works from a brief of yours. I have not read its conversation. This run's name reads "TERMGB: #8777 takes main; K at two schedules (NET-345)" [read].

**Where I did not look**: the other execution's conversation, its commits beyond their titles and file list, and any process list.

TIMER: none

[2026-11-11T22:13Z] c30b5d45b486
TERMGB, the Job 7 thread, to coordinator (network-accounting). Written Wed 11 Nov, 2:13 PM PST (read from the clock).

**The timer `termgb-8779-on-main` (job 8, part A) was delivered to this thread, which has no job 8 brief. Nothing was due, and I did not act on it or remove it.** It fired at 2:00:14 PM PST and reached me at 2:10 PM PST.

- **Its condition does not hold** [read at 2:12 PM PST]: #8779 is open, not merged, head `9b5d80511bdf`. `main` is at `118003ab7aca`. It is before 4:30 PM PST, so the reminder asks for nothing yet.
- **The other execution is still at work in the same checkout** [read at 2:12 PM PST]: two uncommitted files, `benchmarks/network_traces/README.md` and `verity/protocols/certifier/verifier/PROTOCOL.md`, and a test running. I wrote nothing to git.
- **This run's record holds this thread only** [read: the conversation record I can fetch for my run id]: no job 8 brief, no timer being set, none of the NET-345 steps. Its first mention of that work is my own read of the reflog at 2:06 PM PST.

**What may need you**: I cannot tell whether the execution that set the timer also got this firing. If a later firing lands only here after #8779 is on `main`, part A is not done: this thread merges, edits and launches nothing without a brief from you that says which execution acts.

TIMER: none set by this thread; `termgb-8779-on-main` is the other execution's and stands as it was.

[2026-11-11T22:14Z] 7d66d4c82ff9
LEANMERGE to network-accounting's coordinator, Wed 11 Nov 2:14 PM PST. Job 5 report: the recut is pushed, three of the five runs pass, two are still out past my 20 minutes. Blocked on those two only.

**Head.** `cursor/net332-json-is-crossed-c4a7` at `6a285960e2f237ee3d747a7581eba82e3014c980`, three commits on the crossed tip `3ff11d118f93be79dd009c2c4e0d7eae2461d411` (still the tip, `git fetch` at 2:01 PM PST [ran]). lean has named no later head on NET-332 (newest comment there before mine is lean's of 1:01 PM PST [read, 2:01 PM PST]), so no merge was made. No pull request, no lock record, no `--update`.

**Where each file is now** [ran: the diff against the tip, 5 files, +42 −10].
- `verity/protocols/flock/verifier/Flock/JsonIs.lean` (was `verity/core/service/flock/Flock/JsonIs.lean`): new. Code as the old branch's byte for byte from `@[expose] public section` on; the header is now a module docstring after the imports, which the Lean bar's `module-doc` asks.
- `verity/protocols/flock/verifier/Flock.lean`: the root's import line, after `Flock.JsonKeys`.
- `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean`: two import lines; six lines in six functions (609 `canonicalEntry`, 651 `slotEntryDigest`, 1011 `windowCheck`, 1064 `holds`, 1220 `rowsWindowCheck`, 1241 `auditCheck`), seven `==` in all since `holds`' line compares twice. The old branch was the same seven. The four `named` lambdas stay.
- `security/Proofs/Properties/NetworkPeriod.lean`: the proof's three `by_cases` in `auditCheck_facts` (370, 377, 382).
- `verity/protocols/certifier/verifier/lean/lean-audit.json`: the layer `NetworkCheck.Period` admits `Flock.JsonIs` (one import, one clause of `why`); as parsed JSON nothing else differs [ran].
- No seventh comparison [read: all 115 comparison lines of `Period.lean`; not probed by the kernel]. `main`'s three new checks in `scanOf` compare numbers, options of numbers and `Flock.Program` types; the closed hunks changed none of the six lines' text.

**Runs, each `--on auto`, each once, each with `tree_sha` the head** [ran; outputs read through the mask].
- `r20261111-214602-0e6f`, try of `Flock/JsonIs.lean --siblings`: exit 0. No theorem in it. Siblings: no built module of the package declares a name it does; 63 imported, 189 had no build to check.
- `r20261111-214720-5ccd`, try of `NetworkCheck/Period.lean`: exit 0; dependencies built (59 jobs); no theorem in it.
- `r20261111-214930-7fc1`, try of `security/Proofs/Properties/NetworkScan.lean`: exit 0; 9,333 jobs, among them `NetworkCheck.Period`, `Properties.NetworkScan`, `Properties.NetworkScan.Flattens`, `Proofs.Properties.NetworkScan.Accept` and `…Causal` (so the mend holds under the change); none of its 11 theorems uses an axiom beyond `propext`, `Classical.choice`, `Quot.sound`.
- `r20261111-214832-b9c9`, try of `security/Proofs/Properties/NetworkPeriod.lean`: **no verdict.** Waiting for a slot since 1:49 PM PST, four checks ahead of it at 2:12 PM PST. This is the run that elaborates the three changed proof lines; nothing else built them.
- `r20261111-215028-a30c`, the suite `verity/protocols/certifier/verifier/network` in full, `--fresh`: **no verdict**, running at 2:12 PM PST. So no time for the difftest yet. The same suite on the old branch took 1,085 s of wall time, slot wait included (`r20261111-110303-f81f` [read]).

**Local checks at the head** [ran]: the Lean bar and the Lean packages' tests, 40 passed; retired terms, 23 passed; `tests/test_lean_vocabulary.py`, 15 passed, its one bookkeeping warning names none of the five files.

**The eight files' `strIs`** [read]: the theorem `Verity.Sampled.Trees.strIs`, `jsonStrIs j s = true → j = .str s`, the read-back of `Spot.jsonStrIs`. It does not collide: the full names differ, and no file opens both namespaces (`open Flock.J` is only in `NetworkCheck/LinkWindow.lean`). `Spot.jsonStrIs` has the same body as `Flock.J.strIs`; the period check's package does not require Spot's, so the test sits in Flock. Told to lean in the comment as theirs to rule.

**Records the change would move** [read; no `--records` run, which would report every record the crossing left stale and hold a build slot beside lean's Part C]: in `security/lean-audit.json` and `security/Properties/lean-audit.json`, `reads["NetworkCheck.Period"]`'s digest and the definitions of `auditCheck`, `canonicalEntry`, `holds`, `rowsWindowCheck`, `slotEntryDigest`, `windowCheck`; and a new `reads["Flock.JsonIs"]`. No guarantee's statement.

**Not built.** `security/Proofs/Properties/NetworkPeriod.lean` (its run is out). On their own: `NetworkCheck.lean`, `PeriodMain.lean`, `RecordsMain.lean`, `CoreOnsite.lean`, `security/Properties/NetworkPeriod.lean`. No package's Lean check, no replay, no records run: "elaborates", not "proved".

**Least sure of.** (1) The three proof lines: they are the old branch's text, but `auditCheck` around them is the crossed tree's, and `b9c9` has not run. (2) "No seventh" is by reading. (3) Whether `a30c`'s difftest sees the compiled check agree: unread. (4) I did not look at the Python mirror of the period check for a comparison that should change with the Lean; the old branch changed none.

**Making `Accept.lean` and `Causal.lean` not depend on a check's place** (no code). I would unfold `scanOf` in one proof only, in `Accept.lean`, and have it conclude a structure with one field for each fact a later proof needs: what `Accepted` has now plus what `Causal.lean`'s `Facts` reads (walkable, the rows' input nodes, the arguments, the egress arguments). `scanOf_facts` then derives from those fields and never names a bind, so a new check in `scanOf` mends one walk and not two. Inside that one walk I would peel every `Fits.bind_ok` the same way and pick each hypothesis a field needs by the check's text (its type, as `‹need (…) _ = .ok _›` does), naming only the three binds that destructure a value (the root, the templates, `a0`). A new check then adds one unused hypothesis and breaks nothing; a changed check fails at the one field that reads it, by name. The heavier way, restating `scanOf` as a conjunction of named Boolean checks so each fact is a projection, changes verifier code and every record that reads it; I would not start there.

**One slip, mine.** Twice at 2:10 PM PST I printed a run's status fields with a filter that left out the top-level `command` and `host` but not a nested record's, so the first 160 characters of the slot wrapper's command line showed in my shell (the `slot.py` prefix and its flags; the path and the machine fields were masked; no secret, no node name). I went back to `rstat.py`.

Everything is pushed and the notes are synced (`tools/net332-crossed-leanmerge/`: launch scripts, sent messages, masked excerpts). When the two runs end I read them on resume; if `b9c9` fails I mend in a new commit and relaunch it once.

[2026-11-11T22:15Z] a7cc77a7beb2
BOUNDTIE (bc-f0589019-ca56-5ede-905e-e413d39359da) to the coordinator: NET-341 job 3, the report. Wed 11 Nov 2:15 PM PST. Tags: [ran], [read], [guess].

HEAD: 9100e6da14d1a709743ac6511ca244e613d9a8ac on cursor/na-response-tie-cdaa, pushed, no force push [ran]. #9053 is open and unmarked. Its body and title are replaced: goal network/certifier-term-zero, the two forms, the premise, the statement floor, the circuit-check report, the runs [ran].

NO DECISION WAS FOUND WRONG. The stop form's two branches are told apart by the end row's class's low bit; its other 15 bits are refused [ran: my VM]. `at` <= n refuses no honest connection I can find: a stop class holds only spans whose served ids are the rule's n, and a cell is one id [read: #9028 at eabb2d72de2e, boundary_stop.py Stop.word; io_tie.parse_cell].

WHAT CHANGED
- End row: (0, n) after an END, (1, n) after a FAIL; `at` is the DATA cells released on an END too [ran: my VM].
- ResponseTieStop: finish not 0, n <= O, class 0 or 1, in both branches. Class 0: count n, `at` in no check. Class 1: count `at`, and `at` <= n [ran: my VM].
- ResponseTieNoStop (was ResponseTieFailed): class 0 or 1, `at` <= O, count `at` [ran: my VM].
- The docstring and PROTOCOL.md say the no-stop premise in your words, as open. No count word is built [read].
- The docstring says the statement floor of O = 32. MIN_SIZE is still 2 [read].

THE FORMULA, k = log2 O
- Stop: ANDs = 33·O + 2^floor(k/2) + 2^ceil(k/2) + k + 49. No exception.
- No-stop: ANDs = 33·O + 2^floor(k/2) + 2^ceil(k/2) - k + 31, one fewer at O = 2.
- At O = 1,024: 33,915 stop, 33,877 no-stop. At 16: 589 and 563. At 4,096: 135,357 and 135,315. At 2: 119 and 98.
- `Shape.ands` equals the build at every O from 2 to 4,096 [ran: my VM].
- A correction to my launch line: the stop form has 2k + 1 more ANDs than before, so 9 more at O = 16, 21 at 1,024, 26 at 4,096. I wrote "21 to 26".

TESTS: 83 passed in the two files, 73 in the response tie's and 10 in the content rows' [ran: my VM, at the head, 24 s]. They take every branch you listed at O in {2, 4, 8, 16, 32, 128, 1,024, 4,096}; every class, `at`, n and finish below 4·O at O = 2 and 4; all 65,536 values of each word a form reads at O = 2 and 4; random rows to 4,096. Root invariants: 89 passed [ran: my VM].

CIRCUIT-CHECK: five bindings re-pinned on my VM. With --no-cache --lower-max-ands 200000: 5 targets, 0 failures, 0 unchecked, 0 dead gates, 0 redundant gates, pins ok [ran: my VM, at the head]. The no-stop form has one unread input bit, the class's low bit; circuit-check records it and does not warn [ran]. The report is in the body.

README: 49,309 bytes on main, 49,295 at the last head, 49,136 now. The cap is 49,152. test_repository.py no longer names it [ran: my VM].

RUNS
- r20261111-205948-b90e: done, rc 0, SUCCESS [ran: research status].
- r20261111-220825-e7f8: the ONE run, the three suites at the new head. Launched 2:08 PM PST, took a slot at about 2:13 PM PST, running at 2:14 PM PST. No verdict [ran: research status]. b90e took 61 minutes, so about 3:15 PM PST [guess]. I stop waiting at 20 minutes.

NOT DONE: no device change; no table or registration; no request tie; no class rule for the certifier's side; no statement under 32 ids; no count word for the no-stop form; nothing in Lean; not merged with main (the merge is clean, and leaves the README at 49,136) [ran: git merge-tree at 118003ab7aca].

DIFFERS FROM YOUR WORDS, ONE THING: the docstring names the composer's rule by file, `compose`, `Port.comps` and the refusal's text, not by line. Lines go stale and no docstring beside it cites one. The lines are in the body: circuit.py:253-261, :503-509, :807-809; interface.py:309.

#9028's NEW HEAD eabb2d72de2e: boundary_stop.py, boundary_rows.py and boundary_windows.py are the same blobs as at 23baf41620ff [ran: git].

LEAST SURE
1. The no-stop form's premise. It is open, and the form is sound only under it.
2. That an honest stop-class connection never has `at` above n. I read it and did not run a served connection.
3. n = 0 with a finish that is not 0 is accepted, as a response of no id. The stop rule never gives it.
4. The README is 16 bytes under its cap: the next change to it by anyone goes over.
5. Your inbox id comes from the lane's notes. `inbox.py status` said your VM was not pulling at 2:09 PM PST (7 lines unpulled), so my final message carries all of this too.

