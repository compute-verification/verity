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

