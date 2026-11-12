[2026-11-12T16:09Z] 4aa0953a250a
CERTNEG, Thu 12 Nov 8:09 AM PST. **The G = 8 run is launched, once: `r20261112-160747-765d` (NET-385). My estimate is 156 to 170 minutes of its 180, so it fits by 10 to 24 minutes and can still end at the limit with no result.**

**The launch** [ran]
- `research run --on NODE --project verity --source . --cwd clone --pool network --as network-accounting --mem-gb 128 --timeout 10800 -- env SC_GROUP=8 bash benchmarks/scan_cost/run.sh`, from the clean checkout at `f919f768d435`; the run's record names that tree.
- The baseline's own machine, slot `n0`, taken at once. It started at 8:08:10 AM PST; its limit ends it at 11:08 AM PST.
- Nothing is pushed to #9290's branch.

**The estimate (item 1), written before the launch.** Baseline figures are [read] from `r20261112-135528-7404`'s `stdout.log`, `out/sczk/summary.json` and `out/rss.jsonl`; the two local probes are [ran] on my VM (no proving); the G = 8 column is [argued] from them.

| Part of the scan | G = 1, s [read] | G = 8, s [argued] | Grows with the group? |
| --- | --- | --- | --- |
| start, builds, the host's read | 19 | 70 (up to 520 if the node's builds are cold for this tree) | the host's read does, about ×8 [ran: `--dry` here, 4 s and 33 s] |
| Programs and lowering | 277 | 300 | no: lowering is a template's; the Program is 81.02 MB at G = 1 and 81.04 MB at G = 8, the partition 494 bytes at both [ran] |
| `block` round | 31 | 60 to 120 | yes: one session, 8 units become 64 |
| `honest` round | 983 | 3,990 to 4,100 | yes: 4 sessions become 18 |
| `chain` round | 1,016 | 4,090 to 4,200 | yes: 4 sessions become 18 |
| `tampered` round | 69 | 510 to 560 | yes: 1 session becomes 8 |
| `unread` round | 281 | 280 to 290 | no: one unit at any group |
| copy, removal, result | 13 | 55 | with the kept bytes |
| **the run** | **2,687 (44.8 min)** | **9,350 to 9,700 (156 to 162 min); to 170 min with cold builds** | |

- **Inside `honest` at G = 8:** staging 75; the registration and the draw before the first session 265 to 375 (202 at G = 1, paid once a round); the 18 sessions 2,376 (first 70.9, last 96.1, eight ingress at 214 to 218, eight egress at 59.0) and their servers' starts 51; the Lean verifier 443; the scan of what the verifier received 745 (213 at G = 1: it reads every file under `verifier/` and the round then holds ten copies of the big circuits, not three).
- **Sessions in the scan:** 11 at G = 1, 46 at G = 8 (1 + 18 + 18 + 8 + 1) [ran: `--dry`, which the host's read passes at G = 8].
- **Peak memory:** the largest prover process about 23 to 24 GB (22.61 GB at G = 1, for 30 blocks at m = 26 + 5; a member at G = 8 has 31 or 32 blocks at the same m), about 31 GB with its server (6.97 GB), under 50 GB for the job with the driver. The slot gives 128 GB.
- **It fits three hours by 10 to 24 minutes** if a session in a group costs what it costs alone. Sessions and their Lean verifier are 59% of the estimate, so a session that costs a fifth more in a group ends the run at the limit with no result. That is the same assumption as your 93 and 17.
- **Not known before the run:** whether the node's builds are warm for this tree (`main` is merged in since the baseline); what the Lean registration check costs at 512 units (its Python twin is flat: 18.1 s at 64 units, 16.7 s at 512 [ran]).

**If it ends at the limit**
- The scan's log, `lean.jsonl` and `rss.jsonl` are written to `out/` as it goes, so the honest round's session seconds and the Lean verifier's CPU seconds survive. The stage split, the answers and the guard do not: no result.
- A limit of 240 minutes would cover the top of the estimate with a quarter to spare. A group of 6 comes to 126 to 135 minutes by the same arithmetic, a group of 4 to 93 to 99.

**One thing I found on the way** [read]: the baseline run carries no label in the store. The benchmarks' rule asks each measured run for one with its numbers. I will label this run when it ends and the baseline the same way, unless you say not to.

I report when it ends, or at the limit.

[2026-11-12T16:17Z] 576287581966
PERIODPROOF to the network-accounting coordinator, Thu 12 Nov 8:17 AM PST. **NET-367's pull request is open and unmarked: [#9350](https://github.com/compute-verification/verity/pull/9350), head `b782fc88d460`.** It contains #9286 at `f7902b5a99`, still that pull request's head [run: gh]; its first lines say so, and that it is not to be marked before #9286 and lean's merge are on `main`.

- **Statement** [read `security/Definitions/NetworkCertifier/Guarantees/PooledReg.lean`]: `PooledLeakRegistered`, the event form, bound `(1 - q)^Fmax + Σ e_i`, at every secret of a type `Sec` with no set of secrets. A hypothesis `z ∈ Zs` of the bound was an unnamed premise to the lock's check (records run 1, its one finding); a set is the subtype.
- **No new `open` premise, and no other lane's record moves** [run `r20261112-134801-521b`, rc 0 at `d1b7aa7701`: one record kept, security's, committed as `b782fc88d4`; `open`, `trust_gap`, `unwitnessed`, `deployed` equal the base's]. At the head [run `r20261112-155232-cf09`]: all three packages pass, nothing to regenerate.
- **Head tries** [run]: aggregator with siblings `r20261112-134735-ea2a` exit 0, 42 theorems, no further axiom. Earlier heads: `…121452-30ca` exit 1 (fixed), `…122240-d813`, `…125728-460a`, `…132111-d6e3` exit 0; records `…130555-8d60` rc 1 (that finding). Cancelled: `…123652-a7ec` and `…132714-a6d6` unstarted, `…130707-acf8` when the head moved.
- **Test merge with lean's `f3837f57d1`** (local, of `d1b7aa7701`; the seven new files are the head's bytes) [run]: nine tries, all exit 0, no further axiom: `…135801-16b9` (aggregator, siblings, 40 theorems), `…141808-8e63`, `…142129-af22`, `…142642-c590`, `…142855-8072`, `…144901-ca58`, `…150719-27b1`, `…153407-ee3e`, `…154949-807c`.
- **Suites** `--changed c330688b9582 --quick` [run `r20261112-142618-b392`]: 9 passed, 0 failed.
- **`merge-tree`** [run]: head with `main` `8252fe8dff65` clean; with lean's branch 24 files, none under `security/`, `tools/verity/lean` or `tests/lint`, each lean's own conflict with `main`.
- **Gates** [run]: `unread_gate.py origin/main HEAD`, 0 areas above the cap. `ratchet_gate.py` (warn): 3 retired-term sites, two imports and one `open` of modules named with older terms. The list falls 15 in total; my "16" at 5:22 AM PST was wrong.
- **Unread count.** Before: 3 modules, 33 declarations, the cap [run locally: `unread.py` on the published facts of `main`'s `33866d93e66e`; `RecordRows` 27, `SampledAuditInstance` 3, `Warden` 3]. You read it right. After: 3 and 33 [argued]: each new module holds a declaration the guarantee's proof reads [read `Proofs/NetworkCertifier/PooledReg.lean` 74, 77, 98] or the lock names. Not measured: the fast path runs no ratchet.
- **Checked and carried at this set** [argued; definitions read]. `leakAt_regR` is about 45 lines on `leakAt_of_trace`'s 30.
  - Checked (`LeakAtChecked`, `Checked.lean` 146): about 110 lines more, plus a statement and witnesses. It needs the set disjoint from the classes sampled in whole, which holds only outside the phase's event, and a valid run at the trace. `CheckedWitness` is stated at the extractor's witness; restated at the trace it is a new `open` premise.
  - Carried (`LeakAtCarried`, `Carried.lean` 77): 150 to 200 lines beyond the checked. It needs the opened rows among the registered openings, `CarriedRoots` and `CheckedClosing` at the trace's rows, and the pool's collision term.
- **NET-293's proof branch (`ff5d7a3d5aec`) needs carrying** [run: merge-tree]. Clean with `main`; with lean's branch two files conflict, `security/Definitions/NetworkCertifier/Witness.lean` and the area's README. One merge commit, one aggregator try (26 minutes cold on the merged tree today, 7 warm) and one records run (15 to 31 minutes today): about an hour of one slot.
- **Slots**: two at once for about 90 minutes in all, in stretches between 5:06 and 6:55 AM PST, three for two of them when a try's client stopped waiting; one at a time since 6:55.
- **Owed before the mark**: the merge of `main`, the README rows, and the diff showing none of #9286's files.
- **Least sure**: the unread count after. **Not looked at**: one-stage's code for a scan's sample, a stratified sample's rate, whether the deployed order meets `PhasesAdaptive`.

**On your 6:50 AM PST message** [read proofs' scratch file from line 226; argued, nothing built]:
1. `pooled_of_event` does not take the charge as built: its `hev` has one constant a phase and it reads `pr_pooled_le`. With the charge it gains `cl` and its two side conditions, `hev … ≤ e i + Ex (cl z i ω)`, and the term `Ex fun ω => Σ_i cl z i ω (ω i)`; today's is the instance at zero. It is a lemma with one caller, so that costs no record. NET-367's guarantee stands as it is, at charge zero. I see no change to it now that surely avoids a changed record later: a charge needs `pr_pooled_ex_le` on proofs' `ex_prod_add_le` (#9332, not on `main`), and what `PhaseErrors` says with a charge depends on FINDER's statement.
2. For NET-382 I recommend changing this guarantee's record, one guarantee with the charge. Nothing reads `PooledLeakRegistered` (no bridge, not `deployed`), and a second listing is two records, two frozen shapes and two more witnesses for one bound. `PooledLeakAccepted` is the other case: lean's bridge reads it, so its charge form is listed beside it.
3. Sessions a sample: I do not know the count. One-stage v2 holds a statement to exactly one session [read `verity/protocols/spot/verifier/one_stage/proofs.py` 550 to 558], one statement a member with a sampled unit. So it is the number of members the scan's link-window is proved as, 1 if the scan is one member. The period check reads one answer with one level of `2T` units [read `Period.lean` 861, 1087], not the member count. The run of record's evidence settles it; I did not read it.

TIMER: none

