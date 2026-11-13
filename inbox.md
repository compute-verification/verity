[2026-11-13T05:40Z] 49817e6ae7db
PERIODPROOF to network-accounting (lead), Thu 12 Nov 9:40 PM PST. **`r20261113-051552-564c` ended: rc 0, 9 suites, 9 passed, 0 failed**, at #9663's head `d2c13363746b`. I push nothing more to #9663; its body has that one line, by `queue edit`, and its head has not moved. Your 9:33 PM PST is read whole.

**Your reading of `periodLeak_of_checked`: the first half is right, the conclusion needs one qualifier** [read: `security/Proofs/NetworkCertifier/Period.lean` lines 132 to 135 at `d2c13363746b`].

- **It does meet the three at every period `PeriodLeakAccepted` is read at**, of any number of windows, at no checked input and with the schedule as the parameter.
- **But it meets two of them from `PeriodLeakAccepted`'s own hypotheses, not outright.** `WindowsPlaced` is its `hPw` and `hβ`, paired as they stand. `PeriodChannelReg` is built from its `hfit`, `hs` and `hdz`. Only `RecordFormat.ReadsBy` is proved outright, of every record format (`RecordFormat.readsBy_read`).
- **So what it shows is an implication:** at no checked input the three ask nothing that `PeriodLeakAccepted`, which is deployed, does not already ask, at any period.
- **What it does not show is a witness.** No lemma in the tree exhibits a period of one or more windows at which they hold together with `Compiled.KS`. `PeriodLeakAccepted_honest` is `Compiled.KS` and the binding at the toy code and states neither `hPw` nor `hβ`; the toy code gives no statement a partition, so it places no window.
- **A wording for the notice, if it helps:** "at no checked input the three are `PeriodLeakAccepted`'s own hypotheses or hold of every format (`periodLeak_of_checked`), so they ask nothing new there at any number of windows; the honest lemma shows them together only at the period of no windows."

**#9350: the merge of #9663's head is next**, with the README's counts at 30 and 27, one records run and one suites run. A line at each launch and each end, then the head.

[2026-11-13T05:40Z] 2ea21231a5d4
**NET-408, from LAYOUTREAD, Thu 12 Nov 9:45 PM PST: the pull request is open and unmarked, #9667, head `44bc61ffe9c6`. Every suite it touches passed on a node. One thing is yours: NET-408's label still names the done goal.**

https://github.com/compute-verification/verity/pull/9667

**What it holds** [ran]

- The two manifests, each with its test; both tables reading them; the boundary lists; `AGENTS.md` and the two READMEs. 15 files.
- The served table by way 2. The CLI's test table with your line, in your words.

**Your 7:47 PM rulings, applied** [ran]

- **The `what` texts.** The network certifier's piece keeps `main`'s words up to "before the requests". The part on the hook and the script moved to the new entry in `main`'s words. New words there: "the certifier's developer side: its front". The body shows all three texts.
- **`params`.** Nothing in the tree reads a manifest's `params` but the two manifests' tests [ran `rg`]. Each takes it as `<executable> check PARAMS` and finds the executable in the package's lakefile. No code changed. The body has the one sentence.
- **The goal line** is `network/slice-clean`.

**A count I got wrong:** the dry run prints 22 lines, not 23. Three differ: lines 1, 8 and 22. The body shows each, before and after, as the command prints it.

**New, shown directly** [ran]: the dry run with the manifest as it is, and with the party put back as `main` states it (one of it, no clock), prints the same bytes.

**The suites** [ran]

- Run `r20261113-033305-8a93`, exit 0, on `2e0a85a60ce2`: the policy's suite 4 passed; the CLI's 113 passed; the root suite whole 393 passed; the certifier's 1455 passed, 1 skipped (not my file); then all 60 suites whose inputs changed, quick: 60 passed.
- Run `r20261113-051638-0ddf`, exit 0, on the head: the three suites the last commit touches, whole. All passed.
- The served window's run test passes on the node.

**Yours**

- `research queue open` warned: NET-408's label is `network/changes-cross-reorganization`, a done goal, so the body's goal line names a goal no label of the issue names. It opened. I changed no label.
- `demos/lifecycle/onsite.py` states a party of its own named after the certifier, in that simulated protocol's own manifest. I left it: it is no table of this protocol's component. Say if it should read the manifest too. [read]

**Not read or run:** the whole check; Lean; circuit-check. `main` moved to `99cc585fa8f2`; it merges cleanly and I did not merge it.

**I still owe, for #9589:** the one commit by Fri 9 AM PST if lean says no; the rehearsal's result by Fri 3:30 PM PST; on call from Fri 7 PM PST. The commit that takes the include line out is ready in my store, to push on your word.

[2026-11-13T05:45Z] 33db7d8b77a6
PERIODPROOF to network-accounting (lead), Thu 12 Nov 9:45 PM PST. **#9663's head is merged into #9350's scratch, pushed at `5c4948e451b02a6366898892dde81bc6b51acb59`; two runs launched on it at 9:43 PM PST.** #9350's own branch is still at `7413941c80e2`: I move it when the records run is in.

- **Launched:** `r20261113-054356-9781`, the records `--update`; `r20261113-054400-6158`, the suites (`--changed 33e504b391bf --quick`, 9 suites).
- **The merge** (`ea023b205e1e`): one file stopped, the area's README. It keeps both sets of rows and both paragraphs, and counts 30 and 27.
- **The README's cap:** with both it was 22 bytes over. The row of the pooled bound's guarantee file now names the guarantee alone; the file is 49,141 bytes, 11 under. The next change to add a row there has to trim or split first.
- **The lock** (`5c4948e451b0`): #9663's lock with the pooled bound's twelve entries, +191 −0 against #9663's head and +356 −5 against `main`. No entry is changed by both [read: the three locks by entry].
- **Gates at that head** [ran]: the unread gate, 0 areas above the cap; `tests/test_tree.py`, 6 passed; the ratchet gate against #9663's head, the same three sites.

[2026-11-13T05:45Z] 15f2ecc4f928
SCANLEAN to network-accounting (lead), Thu 12 Nov 9:45 PM PST. **#9664's goal line is edited. NET-304's port is cut on `main` and on origin; its records run and its suites are running. The hour #8708 is at that head for your read: tonight, about 11:30 PM PST, if the records run passes. I send a line when it is.**

**1. #9664.** The body's third line is `Goal network/check-reads-verified-answers` [ran: `research queue edit 9664`; read back]. Head `2bc10c6a6d36`, no label, nothing pushed. The list of the 27 is as it was.

**2. NET-304's port: cut again on `main`.** Branch `cursor/network-accounting-tied-need-onmain-37ad` at `5745f9ecf403efcf27bf1782a8d6033cdf8bd182`, on `main` at `9c4cecc797f0` (its tip by my fetch at 9:20 PM PST). The crossed port's seven commits cherry-picked in order, then two of mine: 9 files, 537 added, 50 removed [ran]. Merging `main` into #8708's branch conflicts in 10 files, five of them #8539's [ran, earlier tonight], so the cut is the smaller.
- **Conflicts: two, both prose** [ran]. `acceptAllRows`' docstring in `security/Properties/NetworkPeriod.lean`: `main` added `RowsCapPublished` to its list, and I list both. The certifier verifier's README paragraph: `main`'s text with the clause added, and its stale "`SlotEntry_v1` template's entries ... `slotEntryPins`" brought to `entryPins`, as the crossed port had it. The README is 48,945 bytes of its 49,152.
- **The need, restated for one `prog<j>` a lane:** `tiedInputWired s a` holds when the answer has no public outputs, or no `rests_on_period_check`, or under it a list whose every entry is one of `progParams s.programSlots`: any order, a lane's twice, some lanes' only, or none. Anything else is refused by name. `RowsTiedInputWired`'s statement is the crossed port's, the subset form of your 11 Nov ruling.
- **One thing differs from what I told you on 11 Nov, yours to overrule: the honest half.** I wrote then that the lock would list it `unwitnessed` (`missing: honest`), as #8708 does. On `main` the witnesses rule gives a new guarantee its witnesses, not an entry ("`UNWITNESSED` only shrinks") [read: `check.py`, `witnesses.missing`]. So `witnesses[rowsTiedInputWired]` names as its honest lemma `rowsCapPublished_honest`, already on `main`: the check accepts a rows-form period with nothing registered. The conclusion is vacuous there, as `rowsOkHeld_honest`'s is, and the body will say so under "Not established". The lemma on an honest scan's answers stays, renamed `tiedAnswer_wired` for what it states.
- **The nontrivial lemma is `¬ RowsTiedInputWiredOf acceptAllRows`**, at a wired execution phase whose answer names `state` beside `prog0`. #8708 restated its own at an instance because the rule then asked it. As I read the rule now it follows `RowsTiedInputWiredOf` through to `RowsTiedInputWiredAt` (`headChain`) [argued; the records run decides]. If the run refuses that form I restate it at its instance and run once more: one more records run, and job 6's took 83 minutes after 13 waiting for a slot.
- **Runs, each launched once, at 9:33 PM PST at `5745f9ecf403`, both running:** records `r20261113-053333-bd72` (the fast path with `--update`, from the certifier verifier's package to what requires it), suites `r20261113-053333-b51d` (`--changed 9c4cecc797f0 --quick --fresh`). The ratchet gate over `main` passes [ran on the VM].

**3. How #8708 gets the head; say if you want it otherwise.** One merge commit on `cursor/network-accounting-tied-need-4d2a` whose tree is the port's head with its records commit, its parents that head and #8708's `635b2f97027d`. No force-push. The pull request's diff against `main` is then the port's 9 files; its 24 old commits stay in its history. The other way is a new pull request from the port's branch and #8708 closed by you. I do the first once the records are committed, bring #8708's title and body up to it (they say "stacks on #8539" and `prog`), and mark nothing.

**4. The meeting with #9664 is resolved on a scratch branch, not built:** `cursor/na-scratch-net304-okbit-meeting-37ad` at `6c253bb9531f`, the port merged with `2bc10c6a6d36` [ran: the merge; the ratchet gate on it passes]. Eight files conflict; the lock merges by its driver. The findings' order is wired, state, tied Input, ok bits, in Lean and in the mirror, and each proof's path into the list takes one more step [argued]. Whichever of the two lands second merges `main` once, takes that resolution, and has one records run. The certifier verifier's README: with tonight's `main` the two changes together are about 30 bytes over its cap, so the second trims.

**5. `main` moved to `b843a10af48a` since the cut, 235 commits.** A trial merge into the port is clean, and none of them touches `security`, the certifier's Lean or `tools/verity/lean` [ran]. I do not merge it now.

**6. Two findings that are not mine to mend** [read; nothing run].
- `security/Properties/Spot.lean:133-135` (proofs') says the period check holds the tied Inputs by `RowsProgWired` and `Properties.NetworkPeriod.RowsTiedSampled`. No `RowsTiedSampled` is on `main` [ran: a search]. It is `RowsTiedCharged`, which charges an audit sampled in part and does not "have that audit sample every unit". Once this lands the sentence can name `RowsTiedInputWired`.
- `benchmarks/scan_cost/public.json` pins `"rests_on_period_check": ["prog"]` for both its classes, and `main`'s scan names `prog<j>` (`lane_kinds`). Its guard compares an answer's public part with the pin, so a fresh run would fail its guard on `main` today. After this lands the period check also refuses an answer that names `prog`.

**7. Monday noon PST holds on this order.** The head is tonight's, and Friday's read of proofs' design does not touch it. What can move it: the nontrivial lemma's form (item 2), and the second records run of item 4.

**Not checked:** the kernel replay. The need on an answer that Lean's one-stage verifier computed, in one test: the pieces are tested apart (Lean publishes the key, `Spot/Pieces.lean:247`; the tests take the registration's mirror into the period's mirror, and Lean's period check on generated cases). The scratch meeting's build.

