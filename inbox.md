[2026-11-14T01:11Z] 53aaba91d7a7
PERIODPROOF to the lead, Fri 13 Nov 5:11 PM PST. **Return. The four are carried, each `clean`, each pushed as a fast-forward, each with its runs ended green at the head and its body set. The rehearsal of #9641 on SCANLEAN's tip is done on a scratch branch: it elaborates, its records are the head's, and 89 of 89 suites pass. I marked nothing and hold no run.**

**The four** (heads [rev-parse; each equals the remote's and the pull request's]):

| | outcome | head | at the head | what the records moved |
|---|---|---|---|---|
| #9663 | `clean` | `8187d01dfefcd4299c7ae6fd14917026446334ba` | records `r20261113-222434-0902` exit 0, `pass` ×3, keeps none; suites `r20261113-222522-253c` 10 of 10 | `8187d01dfefc`: the honest witness's record to its new key, the lock alone, 36 for 36, same `type_hash` |
| #8829 | `clean` | `bffa4e603f98d8bca8d8bc21a396341e1e3e2192` | records `r20261113-224646-f008` exit 0, `pass` ×3, keeps none; suites `r20261113-224731-2a1c` 10 of 10 | `bd44a4e9b4df`: ten guarantee records to new keys, four `reads` modules, two digests (`domain`, `open`) to #9663's values; 70 for 62 |
| #9350 | `clean` | `5e514f92d46f78c3e11a8e2eb2885d21aa0a67aa` | records `r20261113-224726-6d23` exit 0, `pass` ×3, keeps none; suites `r20261113-224808-f785` 4 of 4 | nothing at this head; its records commit is the one you read this morning |
| #9641 | `clean` | `7236dd249c296835abb406129c6d8ca0329f6588` | the job `r20261113-184419-f08c` at the commit below: 63 of 63 suites, two locks regenerated | `7236dd249c29`: two locks, +7 −8, as you read it |

- Every suites run: quick, fresh, none from the cache, the node's default temporary directory. Every records run: the fast path, without the kernel replay or the tests, launched by its own command line. [run]
- **By hand, beyond the tool:** #9663's names commit `fbbbaae31cc3` (7 files, 46 for 46); #8829's K mend `c441db46a4c4` (one file, 21 for 18), its names commit `a06af1520007` (10 files, 72 for 72) and its hand merge of #9663's head (two files, four blocks, each as the branch has it; tree equal to `bd44a4e9b4df`'s [run]). #9350's merge of #9663's head was clean.
- **Built on another:** #8829 and #9350 each hold #9663 at `8187d01dfefc`. #9641 holds none of them.
- **Bodies set:** all four. #9663's and #9350's with the runs at the head; #8829's rewritten whole, with a title that says `MonitoringPeriodLeakChecked`; #9641's as set at 1:25 PM PST, its Goal line kept. Each says what the carry did by hand; #8829's and #9350's say the three `why` texts wait for the later of the two.
- **`main` today (`eba3ea48d4db`, with the Dither move):** the tool's dry run for the four is `clean`, plain merge clean [run]. Nothing carried for it, as you said.

**The rehearsal** (nothing pushed to #9641):

- **Scratch head `d53e73e85398ca85f078fcac4ce372250a31a789`** on `cursor/na-scratch-period-bound-meeting-4a02`: #9641's `7236dd249c29`, the merge of SCANLEAN's `080b7d735e95` (`3efe4fd67b47`), one mend (`8429124757de`), and a clean merge of **SCANLEAN's tip `b713d4754e63d9fef8d16561a683a49893a1496f`**, the tip I took (locks only, by the driver). Through it the branch holds `main` at `d119240998be`, after the second chain and before the Dither move.
- **Six blocks in three files, each the other side's lines first and then mine, but one sentence:**
  - `security/Properties/NetworkPeriod.lean`, three. The module's account: the ok bits' paragraph, then mine. **The one sentence**, both lists in it: what `RowsProgWiredOf`, `RowsStateTiedOf`, `RowsTiedInputWiredOf`, `RowsTiedChargedOf`, `RowsCapPublishedOf`, `RowsOkHeldOf` and `AnswersOf` must fail at, for those six properties and for `RegisteredSampleHeld` and `SampleBoundCharged`. The definitions: the ok bits' (`OkEntry` to `unnamedCommitment`), then mine (`sampleReg` to `statedPublic`).
  - `security/Proofs/Properties/NetworkPeriod.lean`, one: the ok bits' section, then mine.
  - `period_case.py`, two: `TIE_UNHELD`, `UNOK`, `UNCOMPUTED`, then my `TIED_DELTA`; and the cases' verdicts, both sides' rows in one table.
  - Both locks and `Period.lean` merged by themselves.
- **One mend the merge needed:** `check_clock` reads an audit's finding one list further in (one more `List.mem_append_left`), since `okChecks` is now last.
- **Where my findings sit:** #9641 adds no list to `check`. Its refusals are inside a scan's answer and an audit's answer, so they are in `ws` and `as`: `ws ++ [oneDelta] ++ as ++ wiredChecks ++ (stateChecks ++ tiedInputChecks) ++ okChecks`, mine before all four lists, in Lean and in the mirror [read: `Period.lean` 1544 and `period_case.py`'s `check` at the scratch head].
- **Runs**, the first three at `8429124757de`, whose Lean and Python are the scratch head's:
  - `verity lean try` on the proofs file, `r20261113-232408-4dad`: Lean exit 0; none of its 82 theorems uses an axiom beyond the three.
  - The lock's records by its own command line, `r20261113-232409-063b`: exit 0; `security/Proofs` `pass` (204,388 declarations in 4,051 modules); `security` and `security/Properties` `regenerated`, four lines: the check's digest `b62c201d…` and its module's. **Both regenerated locks equal the scratch head's, byte for byte** [run: no difference], because SCANLEAN's records commit writes the same lines: #9641 does not change `check`'s own text. So the head needs no records commit, and I ran no second records run.
  - The suites changed since `7236dd249c29`, `r20261113-232408-b57e`: exit 0, 89 of 89, none cached. The certifier's network suite, with the period's difftest run whole, 105 passed; the certifier's verifier 1,441; `verity-vllm` 17,154; `research` 5,009.

**What quotes the certifier's `K` or the leak bound** (your 4:08 and 4:21 PM PST lines; I wrote neither wording anywhere, and edited no body for it):

- Bodies: #9641's (the check's `K`; one line of counts, "`K` is raised in 229 and lowered in none"); #8829's (the K lemmas, as terms, no number); #9663's and #9350's (the leak bound's statements, no number).
- Tracked files of mine: `security/Proofs/Properties/NetworkPeriod/Fails.lean`, `security/Definitions/NetworkCertifier/Guarantees/MonitoringPeriodFails.lean`, the `MonitoringPeriod*` and `PooledReg*` proof files and the README of `security/Proofs/NetworkCertifier/`, and on #9641 the two `NetworkPeriod.lean` files, `Period.lean` and `period_case.py`. None holds a measured figure.

**Least sure of:**

1. **#8829's K lemmas.** They build and the three guarantees' statements did not change, but that `tied`, `faulty` and `charged` as free parameters are what a later proof needs at `main`'s terms is [argued] from `main`'s `rowsKBoundsCount`, not shown by a use.
2. **The rehearsal's last step.** The suites ran one locks-only merge below the scratch head, and the head's records pass is [argued] from the byte equality, not run.
3. **The one sentence.** I joined both lists in one sentence as you asked; read its wording before it goes to #9641.

**Not looked at:** the whole `verity check`, a kernel replay and `circuit-check` at any head; any build at a merge with `main` at `eba3ea48d4db` (merge-tree and the dry run only); a records run by its own command line at #9641's `7236dd249c29` (its records came from the job's script, so the gate sees none until its merged head); #8829's README is 50,738 bytes, over its cap, left for the README pull request; comms' #9737; the three `why` texts, unchanged.

**Open with me:** the merge of #9641 when you say #8708 is on `main`; the one commit of the `why` texts on the later of #9350 and #8829; the README pull request after #9663 and #9350 land; NET-427 on your word.

TIMER: none

