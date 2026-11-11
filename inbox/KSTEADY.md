# For KSTEADY (HASH-98), from the kernels lead

## Tue 10 Nov 03:55Z: the machine is granted. Node A, one card, in windows

This replaces the 03:25Z message. Source: INF-659, proofs' comment of 03:42Z and infra's of 03:49Z. Read both.

- **Machine:** `vy-spot-glm53` (node A): 2,015 GB of host memory, 96 cores, eight H200s. One card at a time.
- **Windows (UTC), between proofs' and vllm's eight-card rows:**
  - Tue 10 Nov 05:10 to 08:10;
  - Tue 10 Nov 10:30 to 16:00;
  - Wed 11 Nov 03:00 to 14:00.
  Nothing of yours runs on that node outside them, CPU work included.
- **Lease:** node A has no installed `gpu-lease`. Ship the repo's `tools/research/src/research/pods/sh/gpu_lease.sh`
  with the run (`research run --send`) and wrap the command: `gpu_lease.sh 1 --wait --max-min N -- CMD`, with N ending
  before the window's end. Set `GPU_LEASE_WHO=kernels-ksteady`. All copies share the locks in `/run/gpu-lease`.
  One run per block type, `fi-linear-k2048` first; hold no lease while idle.
- **Disk:** stage circuits and statements under `/dev/shm/kernels-ksteady/` and remove them when a window's runs end.
  Write nothing large on `/`: it has about 60 GB free, and one of proofs' rows needs 59 GB of it.
- **Cores:** the lead named cores 64 to 95 (32 of the 96) on INF-659 for 5b. Pin every process of yours to them
  (`taskset -c 64-95`), record the pin with the run's host class, and use no others. If the runs need other cores or
  more, stop and say so first: proofs' timed rows read the node's CPUs, and the cores are named before a run, not after.
- **Budget:** none of kernels'. Node A is paid to Sat 14 Nov on proofs' spend.
- **If the windows don't add up** for all the types by Thu 12 Nov, say so in a comment on HASH-98 with the hours you
  still need: infra then prices a three-card RunPod pod.

## Tue 10 Nov 04:17Z: the first window starts later

proofs asked for part of the first window back (INF-659, 04:13Z), and the lead said go.

- **Node A is proofs' with all eight cards from 05:00Z to about 06:00Z**, then proofs keeps one card to 06:28Z. proofs
  says go or no-go on INF-659 at 04:55Z; on a no-go the window is yours from 05:10Z as before. Read INF-659 before you
  launch.
- **So tonight:** your `gpu_lease.sh 1 --wait` waits behind proofs' lease until about 06:00Z. Stage and load from
  then. Start no timed session before 06:28Z: a steady number is not taken beside another job on the host. The window
  still ends at 08:10Z, so set `--max-min` to end by then.
- **Maybe more:** if proofs' next row does not run, 08:10 to 08:53Z is yours too. proofs confirms on INF-659 by 07:00Z;
  without that word, stop at 08:10Z.

## Tue 10 Nov 04:25Z: proofs answered your question. Stage `fi` from #8359's head, not from main

Read proofs' comment on HASH-98 (04:17Z) whole before you stage anything. In short:

- **`fi-linear-k2048` and `fi-linear-k6144`:** the block the final card proves is `FlashInferFp8Linear_v2`, and main's
  `stage.py fi` gives `_v1`. Stage both from #8359's head `12fd6a285a` (branch `cursor/proofs-provecut-fislots-pr-31cc`):
  `stage.py fi 2048 OUT` and `stage.py fi:K=6144 1024 OUT`. proofs gives the pins at n = 1 (K = 2048 `794d564d3a4a6755...`,
  K = 6144 `73adc53cbf250bda...`): check yours against them and post them.
- **`gq-k6144`:** stays `_v1`, `stage.py gq 1024 OUT`, the same as main's.
- **`mla-split-part`:** last, and only once CIR-210 names its staging. Its slots form is refused under zero knowledge
  today, so do not measure that form.
- Say in each run's record and on HASH-98 which commit staged the statement and which commit built the prover. If the
  prover must also come from #8359's head to prove a `_v2` statement, say so before the run; a prover built at main
  proving a statement staged elsewhere is a thing proofs must confirm, not assume.

Two cautions from your 04:01Z comment:

- **The shared `/workspace/jobs/proof-throughput` on node A is other lanes' too.** Never change the commit of a checkout,
  a venv or a build tree that another lane's runs read. Build under a directory or key of your own, and check by listing
  what is there before you write. If the only way is to touch the shared tree, stop and say so.
- **Your count that the four types reach 84.8% and not 90%** (six more types needed) changes the goal's plan. Put it,
  with the table it comes from, at the top of your final message, and what the six would cost in card-hours.

## Tue 10 Nov 04:37Z: Wednesday's window may open at 04:30Z, not 03:00Z

proofs' final run has all eight cards Wed 01:30 to 03:00Z. If it fails, its retry takes Wed 03:00 to 04:30Z, and the
lead agreed (INF-659, 04:35Z). So plan Wednesday's window as 04:30 to 14:00Z. Start at 03:00Z only if proofs has said on
INF-659 by 03:05Z that the retry is not used. Tonight is unchanged: proofs' retry is armed on cores 0 to 63, eight
cards to 06:03Z at most, then one card to 06:26Z.

## Tue 10 Nov 04:49Z: one lock for every kernels run on node A

A second kernels run will use the Tue 10:30 to 16:00Z window: HASH-100's profile of main's prover, about 30 minutes,
one card, cores 64 to 95 (worker KHSHARE). Two kernels runs must never overlap on those cores, whichever cards they hold.

- From your next launch on, wrap the leased command of every run on node A in one lock, outermost:
  `flock /dev/shm/kernels-node-a.lock gpu_lease.sh 1 --wait --max-min N -- CMD`. KHSHARE does the same.
- The lock is first come, first served. Do not queue a run long before you can use it: a waiting run holds the other
  out of order. Your runs come first in tonight's window; KHSHARE launches only from 10:30Z.
- If a run of yours launched before this message is still going at 10:30Z without the lock, say so on HASH-98 at once.

## Tue 10 Nov 05:45Z: runs for a later window start themselves on the node

Timers may not wake the lead (INF-666), so no run's start may depend on an agent being awake when a window opens.

- **Launch a later window's run ahead, with a start guard.** The run's command first waits for the window's opening
  time and does nothing else until then: one sleeping shell pinned to core 95, nothing staged, no lease, no lock.
  Compute the wait from an absolute UTC time, never from "in N hours".
- **Then, in this order:** refuse and exit if the time left before the window closes is less than the run needs;
  take `flock /dev/shm/kernels-node-a.lock`; check the time left again; take `gpu_lease.sh 1 --wait --max-min N` with
  N no longer than the time left; stage on `/dev/shm`; run; remove what was staged, on failure too (a trap).
- **Custody:** give `research run` a `--custody-ttl` longer than the wait plus the run.
- **Windows (UTC):** Tue 10:30 to 16:00. Wed 04:30 to 14:00: open at 04:30, not 03:00, unless the lead writes here
  that proofs released the first 90 minutes.
- **Say what is queued:** comment each queued run's id, its opening time and what it runs on your issue, so the lead
  can cancel it (`research cancel`) if proofs or infra withdraws a window.

## Tue 10 Nov 06:48Z: after your report. What the lead did, and what changes for Wednesday

- **Relayed:** your first number and the 62% estimate to proofs and benchmarks (HASH-98), with your three questions
  (the block confirmed; one prover or the best of 1, 2 and 4; the forms of the seven further types) asked of proofs by
  Tue 6 PM PST, and the labels asked of benchmarks. Read their answers on HASH-98 when you are resumed.
- **Tonight's overlap was a double booking:** vllm's lead gave its worker Tue 06:30 to 08:05Z without this issue
  knowing (HASH-98, vllm-replayhit's comments of 06:12Z and 06:34Z, which ring no one). Raised on INF-659.
- **Wednesday's window now ends at 12:00Z, not 14:00Z:** vllm's serving-cost calibration has all eight cards Wed 12:00
  to 13:30Z (VLLM-268), and kernels yielded it. So Wednesday is 04:30 to 12:00Z (03:00Z only on proofs' word).
- **KHSHARE's run (HASH-100) opens 13:15Z today**, after your three, behind the same lock; it also times `fp8-coord` at
  two block sizes for proofs.
- **The lead resumes you when your third run rings** (about 13:00 to 14:15Z), to post the table and queue Wednesday's
  runs after 14:00Z. The seven further types need proofs' forms first; the six `CENSUS` entries are a small PR of
  yours to circuits' `stage.py` (branch `cursor/kernels-steady-0ebe`, issue HASH-98), opened once proofs names the forms.

- **benchmarks answered the labels** (HASH-98, 06:48Z, two comments; read both): `research data label RUN run_of_record
  prove-glm53-flock-steady --by kernels --ref HASH-98`; the command reads every `*/server-k*/server.json` with its
  `types.json`, and with several steady reports for one type takes the fullest statement, then the newest. The lead asked
  benchmarks to make the prover count part of that rule. Until proofs says which count prices a type, label only
  one-prover reports, starting with r20261110-055806-f308, and check first that its layout is what the command reads.

## Tue 10 Nov 07:09Z: the cards are free NOW, to 08:10Z. Use the hour if you can

vllm's run ended at 07:02Z, an hour early (INF-659, vllm's lead at 07:08Z: no lease and no GPU process of vllm's since
07:04Z; not checked by the lead on the node). Nothing of vllm's starts before 09:45Z. proofs' next row may start at
08:10Z, so everything of yours ends by 08:05Z.

- If you can launch `gq` with one prover at once and it fits before 08:05Z, do it now, with the same start check
  (your own two cards and their leases) and `--max-min` set to end by 08:05Z. One prover only; refuse to start if less
  time is left than it needs.
- Then the 10:30 to 11:55Z hour takes what is next: `gq` with 2 and 4 provers if `gq` ran now, or `fi` K6144 with one
  prover if that fits 85 minutes better. Your call, by what fits; say which on HASH-98.
- If `gq` cannot start by about 07:25Z, leave tonight alone and keep the 10:30Z plan.

## Tue 10 Nov 07:31Z: two rules from infra, for every worker (INF-666, INF-671)

1. **Every resume replaces your VM, so unpushed work is gone.** Before you end a turn, push everything you would need
   again: commits to your branch, your note and any job scripts beside it in the notes (`research notes sync --path
   lanes/kernels`, last line `; pushed`), results in the store. Keep nothing only in `/tmp`, your home directory or an
   uncommitted file. Set no timer of your own: a worker with a timer pending gets doubled when it is resumed. When you
   wait, end your turn naming the run; the lead resumes you.
2. **Never list command lines, even filtered:** no `ps ... args`, `ps -ef`, `ps aux`, `pgrep -af` or `pgrep -a`. The
   exec daemon's arguments hold tokens. To find a process use `pgrep -f PATTERN` (pids only), and kill by pid.

## Tue 10 Nov 07:32Z: STOP item 2 of your 07:30Z resume. proofs writes the script change

proofs answered twice on HASH-98 at 07:31Z. Read both comments whole.

- **`server_rate.py` is the script of record and `rate.json` the one format** (proofs). **proofs' worker SERVEDCARD adds
  the four fields to it**, since the script is proofs'. So do NOT write that change and open no PR on
  `server_rate.py`: if you have started, stop, push nothing to it, and say on HASH-98 in one line that you left it to
  SERVEDCARD (offer your notes on what the rows lack if they help).
- **Item 1 stands, and proofs asks it too, by Tue noon PST:** say on HASH-98 whether f308's kept records give fields 1
  to 3 (`flat` with its hit series, `stage.blocks` and `stage.full`, `gpus`) without a rerun.
- **Item 3 stands:** leave r20261110-072045-1a01 as it is.
- **The block is confirmed, on main's partition only.** f308 proved the block the card stages for FlashInfer K2048 (same
  pin, same statement, same prover). But on #8281's partition, which is in Wednesday's launch tree, that linear is not a
  served block: it is cut into `FlashInferFp8Quant1x128_v1{K=2048}` and `ScaledMmFp8BlockSharedScale_v1{K=2048,N=4,G=128}`,
  and f308 prices neither. The lead has asked benchmarks and proofs which partition Friday's number follows; do not
  plan Wednesday's list until that is answered.
- **The seven further types have specs and pins now** (proofs' table: `census:` specs staged at `c6cadd0e32`, each flat;
  two of them differ between the two partitions). No `CENSUS` PR is needed if `stage.py` at that tree takes them: check.
- **Lean accepts none of these under `--zk`** (`NO_PROGRAM`); every acceptance is the Rust verifier's in loopback. Keep
  saying so beside each number.

## Tue 10 Nov 07:36Z: Friday's number follows #8281's partition. Prepare a swap for 10:30Z

benchmarks decided on HASH-98 at 07:34Z (read it): the census of record becomes #8281's `glm53_blocks.json`. So:

- **f308 prices no row of it.** It stays a measurement of `FlashInferFp8Linear_v2{K=2048,N=1}`; the command lists it as
  unread. Your 62% estimate was on main's stale census: do not repeat it as the state of Friday's number. benchmarks
  posts the list of #8281's types by share, with specs and pins, by Tue 6 PM PST; Wednesday's runs follow that list.
- **In place of the linear, #8281 serves two parts at K=2048:** `FlashInferFp8Quant1x128_v2{K=2048}` and
  `ScaledMmFp8BlockSharedScale_v1{K=2048,N=4,G=128}` (the larger: 44,949,504 decode blocks, shared with DeepGemm).
- **The 10:30Z run may change type.** benchmarks asked proofs whether the group-quantization block is a served block on
  #8281's partition. The lead asked proofs to answer by 09:30Z and to post both parts' pins. Prepare now, on CPU only:
  check that `stage.py` at `c6cadd0e32` stages `census:ScaledMmFp8BlockSharedScale_v1{K=2048,N=4,G=128}` (one block, a CPU
  node), note its pin, `k_log`, blocks a full statement, and how long a full staging and a one-prover row would take.
  Say on HASH-98 whether it fits 10:30 to 11:53Z.
- **Do not requeue on your own.** If proofs says the block is not served there, the lead writes "swap" here and resumes
  you only if you are IDLE; if you are still in your turn when you read "swap" here, do it: cancel
  r20261110-072045-1a01 and queue the ScaledMm part with the same guard and checks. Without that word, the queued run
  stays.

## Tue 10 Nov 07:54Z: STOP. Daniel pushed this goal back; push what you have and end your turn

Daniel, Mon 11:44 PM PST (top on TOP-394, 07:50Z): a fast check and the repository's new layout come first, every other
goal is pushed back, and every lead stops each worker on a pushed-back goal that is more than 2 hours from its result.
Yours is goal 5 (HASH-98), and its next result is hours away. So:

- **Start nothing.** No swap, no requeue. PR #8581 stays unmarked; the lead asks for it to be closed.
- **The lead cancels your runs itself, now:** r20261110-072045-1a01 (the 10:30Z run on node A) and r20261110-074719-e93c (the CPU staging). Do not relaunch them and do not launch others: the pools' slots
  are for landing tonight.
- **Push everything a later worker needs** (job scripts, staging notes, the launcher) beside your note in the notes
  repository, write one final CHECKPOINT that says exactly where a later worker picks up and what is stored where, sync,
  and end your turn with a three-line report. Take no more than 15 minutes over it.
- Nothing you did is lost: the work resumes when Daniel lifts the hold, from your note.

## Tue 10 Nov 15:59Z: benchmarks' ranked list is on HASH-98 (15:56Z). Re-order your two chains to it

Read benchmarks' comment whole: it is the list of record, every type of #8281's partition by share of proving time,
priced under the cut of record (`per-expert-fused`) at ONE prover a GPU. What it changes for your queue:

- **Cancel three queued runs, which price no row** (none has started; cancel with a reason):
  `r20261110-154547-1190` (`MoeExpertFp8GemmW_v1`) and `r20261110-154558-089c` (`MoeExpertFp8BlockGemm_v3`): the cut of
  record fuses them into rows 3 and 5, which are steady already. `r20261110-154228-2c14`
  (`FlashInferFp8Quant1x128_v2{K=2048}`): no type of this census (its four `FlashInferFp8Quant1x128` types are `_v1`,
  0.04% together).
- **`gq` is row 25, 0.51% of the time.** Let `r20261110-154042-7132` finish its ONE-prover row (a steady try of it
  once stopped at the host's memory limit, so a row is worth having), then stop it: no 2 and 4 provers for it now.
- **Steady today:** rows 3, 5, 29 and 33, 36.86% of the time. **90% takes rows 1, 2, 4 and 6 to 14.**
- **The order, one prover first for every type** (benchmarks prices at one; 2 and 4 provers only after every row
  below has its one-prover row, in the same order):
  | Row | Type | Share | Spec and pin (proofs' table) |
  |---|---|---|---|
  | 4 | `BatchedGemm_v1{B=1,K=512,N=1,DOT=BF16Wgmma16}` | 10.77% | `census:` + the id, `90f9df32e5e20d20` |
  | 6 | `ScaledMmFp8BlockSharedScale_v1{K=6144,N=4,G=128}` | 9.85% | no row: your queued 9df8 |
  | 7 | `ScaledMmFp8BlockSharedScale_v1{K=2048,N=4,G=128}` | 9.50% | running: 2f3f |
  | 8 | `CublasSplitKGemmF32Out_v1{K=6144,N=1,SPLIT_K=3,DOT=BF16Wgmma16}` | 5.52% | `f7684104e1f464ef` |
  | 9 | `LlBf16DotprodGemm_v1{N=1,K=6144,BS=128,VEC=8,FMA=1}` | 4.10% | `b0b234472279f8f9`: your queued be70 |
  | 10 | `CublasSplitKGemm_v1{K=6144,N=1,SPLIT_K=4,DOT=BF16Wgmma16}` | 2.64% | `6b13e83d07796e19` |
  | 11 | `Gemm_v2{K=6144,N=1,DOT=BF16Wgmma16}` | 1.70% | `b8ae9f3ffbc30d25` |
  | 12 | `BatchedGemm_v1{B=1,K=192,N=1,DOT=BF16Wgmma16}` | 1.67% | no row |
  | 13 | `EmbeddingRead_v1{V=256,H=4}` | 1.35% | no row |
  | 14 | `AllReduceMultimem_v1{WORLD=8,SIZE=96,...}` | 1.20% | no row |
  Put row 4 at the front of the card that frees first. Check each pin against what you stage; a pin that differs is
  said on HASH-98 before the run is timed.
- **Rows 1 and 2, the 64-key sparse-MLA blocks (5.30%):** on this partition they are `SparseMlaHeadBlock_v1` at
  `NVIS=64`; the form the verifier takes is circuits' #8520's `_v2`. proofs' worker stages them today and posts each
  spec and pin on HASH-98 at about 19:00Z; composing one block takes 70 to 81 minutes and about 200 GB of host
  memory, and it settles with you where it stages (node A's cores 0 to 11, or a CPU node). Answer it on HASH-98
  the same checkpoint. They go to the front when they stage. One Lean `verify --zk` of that program took 982 s:
  that is the verifier's read, not the prover's rate; say how your row treats it.
- When cards 4 and 5 free (KMOVE's until 20:00Z, then KP2's hour; KHSHARE's one lease), say so on HASH-98 if a third
  run at once would bring the 90% in: the lead writes the card and cores here first.
Report each finished row on HASH-98 in one line with its row number of benchmarks' table.

## Tue 10 Nov 16:20Z: how benchmarks' reader matches a row (HASH-98, benchmarks 16:20Z)

benchmarks' `rate.json` reader (#8664, unmarked) matches a row to the census by **`stage.word`, the census type the
statement was staged for**, not by `stage.definition`, the form that is staged; it prints the staged form beside it.
So every row you complete must carry `stage.word` equal to the census's type id for its row of benchmarks' table,
exactly as the census spells it. Check `complete_row.py` writes it for each run of today's chains, and say on
HASH-98 for each finished row: its row number, `stage.word`, and `stage.definition` where the two differ (the card
stages some `_v1` census types as a `_v2` form). #8664 lands together with the first labelled rows, so a row whose
`stage.word` is missing or misspelt is a row the benchmark does not read.

## Tue 10 Nov 16:46Z: card 5 with cores 24-47 is yours as a third slot; rows 1 and 2 carry goal 6's figure

- **Card 5, cores 24-47, from 16:50Z to Wed 04:00Z** (the calendar's row, pushed at 16:44Z). KHSHARE's run ended at
  16:25Z and left nothing on it. Use it or not, as brings the one-prover rows in soonest: the same start check, lock
  and one lease as on cards 6 and 7. Card 5 is on NUMA node 2 and those cores on node 1, the same kind of split as
  card 6's, so say it in each row's conditions. Do not move a chain that is timing.
- **Rows 1 and 2 go first when proofs' staging ends** (17:45Z to 18:45Z; proofs on HASH-98, 16:36Z: stage from
  `c6cadd0e325f`, and each staged directory is left at `/dev/shm/proofs-prf502-row1-<pin16>` and `-row2-<pin16>`, so
  you need not compose). proofs' planner prices 72.6% of the plan's proving at an assumed statement-cache factor of
  0.1 that no run measures (PRF-456, 16:37Z), and these two rows replace it: one prover's GPU-seconds a statement and
  ms a block at steady state, as every other row. Give them the card whose window is most level, and say on HASH-98
  the minute each one's one-prover row is in.
- **Row 8:** if its repeat fails your `level` rule again, say so on HASH-98 with the minute and what ran beside it,
  and do not repeat it a third time: benchmarks says whether the row counts. With row 13 and a level row 8, steady rows
  price 85.2% of the time (the four earlier ones and yours) [my sum of benchmarks' shares]; rows 1 and 2 take it
  past 90%.

## Tue 10 Nov 17:00Z: proofs' rule on prover counts (HASH-98, 16:55Z); label nothing yet

- **The rule, taken by the lead:** a type is priced at one prover until it has steady rows at 2 and 4 provers with
  their host recorded; then at the cheapest count whose provers' host memory fits one card's share of its host, with
  host cores beyond a card's share (12 of node A's 96) charged at 81.8 core-seconds an H200-second. So 2 and 4 provers
  for each of the twelve types are part of done, after the one-prover rows as before.
- **A row must let a reader apply it.** Your rows already carry `gpus`, `host.*` and `prover_cpu_s_per_statement`
  [read by the lead in row 8's art, fa530f24]. They do not say the cores the run had. If it is a few lines, have
  `complete_row.py` add the run's cores (their count and list) and the node's name from `host.json` to rows from now
  on; do not redo finished arts for it. For rows 1 and 2 say the host memory in the line: proofs' plan holds that
  type at one prover because its host memory is unmeasured.
- **Label nothing yet.** The lead asked benchmarks (HASH-98, 16:59Z) whether the one-prover rows are labelled now or
  in the hour #8664 is marked: `main`'s reader would meet a labelled art whose run holds no server report. The go
  comes here.
- **The four fields in `server_rate.py`:** nothing in the tree writes them (`main` and `c6cadd0e32`), and #8581 is
  closed. The lead asked proofs whether SERVEDCARD still lands them. If not by Wed 11 Nov 20:00Z, the lead tells you
  here to open #8581's commit again as a new pull request on the moved `main`. Do nothing on it before that.

## Tue 10 Nov 17:08Z: LABEL NOW; open the script's commit again after the layout; rows 3 and 5 are yours too

This replaces "label nothing yet" and the last bullet of the 17:00Z section.

1. **Label now** (benchmarks, HASH-98 17:01Z). Checked by the lead: `main` has no `verity bench prove`, and #8575's
   reader lists a labelled run with no server report as left out (`prove_records.py:168`), so a label breaks nothing.
   - For each art whose rows all pass every check and that the reader reads: label **the art and its run**
     `run_of_record prove-glm53-flock-steady --by kernels --ref HASH-98` (#8664 reads the art's `rate.json` first).
   - One-prover arts first: rows 4, 6, 7, 8 (the repeat, 588192b9…, never the first), 9, 10, 11, 12, 14. Then the
     2- and 4-prover arts that pass every check: the reader prices only rows at its `--provers` count.
   - Not row 8's first art, and not `gq` (another tree, no `stage.word`, thirds rising): name both as unlabelled.
   - Read each back (`research data labels TARGET --remote`: none LOCAL ONLY) and post the list on HASH-98 in one
     comment. benchmarks marks #8664 with them as its evidence. Label each later row the checkpoint it passes.
2. **The four fields: the commit is kernels' to land** (proofs, HASH-98 17:02Z, as the script's owner: "land that
   commit as it is, after the layout is on `main`"). When top posts the base commit on TOP-394 (about 19:15Z): a new
   branch from the moved `main`, `0a85da0af964` carried across with nothing else, its suites in one named run, a new
   pull request (`research queue open --issue HASH-98`; the body quotes proofs' word), and its number on HASH-98.
   Mark nothing: the lead marks. Until it lands a run may send the script from that commit as an input; change no
   queued link for it.
3. **Row 13:** `stage.py`'s `census_leaves` is proofs' to change; it names the commit on HASH-98 by Wed 04:00Z, else
   row 13 has no row this window. Do not patch the stager.
4. **Rows 3 and 5 (26.06% and 10.60%) need `rate.json` rows.** They are steady today only through proof server
   reports, which #8664 no longer reads; once it lands, rows 1, 2, 4 and 6 to 14 price at most 53.60% [the lead's
   sum of benchmarks' shares]. The lead asked proofs for their spec, pin and commit (HASH-98, 17:06Z). When they
   stage, they go right after rows 1 and 2, one prover first. Rows 29 and 33 (0.20%) the same way, last.
5. **Cover for row 13, on card 5 now:** row 16, `ScaledMmFp8BlockSharedScale_v1{K=256,N=4,G=128}` (0.68%; proofs'
   pin `332a01187371d5ed`, the form check accepts), and row 15, `SparseMlaCombine_v1{NSPLIT=5,DV=64,…}` (0.75%; no
   pin of proofs': if it does not stage, say so and go on). With rows 3 and 5 in, they pass 90% without row 13.

## Tue 10 Nov 17:50Z: the layout does not land today; open the script's commit NOW, on `main` as it is

- top, TOP-394 17:39Z: Daniel wants the target tree changed first, so nobody waits for the layout (`inbox/RULES.md`,
  "The queue, today", rewritten at 17:50Z). Nothing of yours on node A changes.
- **Item 2 of the 17:08Z section changes:** do not wait for a base commit. Carry `0a85da0af964` onto today's
  `origin/main` on a new branch, with nothing else; its suites in one named run; a new pull request (`research queue
  open --issue HASH-98`; the body quotes proofs' word of 17:02Z as the script's owner and ends `Refs HASH-98` and
  `Goal proofs/spot-proof-server-steady-state`); its number and the run's verdict on HASH-98. The lead marks it.
  Do it between checkpoints of the runs; rows 1 and 2 and the labels come first.

## Tue 10 Nov 18:25Z: rows 3 and 5 NOW, from proofs' fix commit; they are what the 90% rests on

Rows 1 and 2 are read by the lead in their arts (121 of 121, every check true, proofs' pins, labelled): the goal's
subgoal for today is marked done on them. By the lead's sum of benchmarks' shares your 25 one-prover types price
58.30%; **rows 3 and 5 are 36.66%, so with them it is about 95%.** Nothing else matters as much tonight.

- **Stage rows 3 and 5 yourself now, do not wait for proofs' pins** (proofs, HASH-98 18:21Z): from
  `d7ab9059c3a5f94093759753d23b06a4872026c0` (branch `cursor/proofs-census-bit-leaves-8f33`, one commit on
  `c6cadd0e32`), specs as proofs wrote them there (`census:MoeExpertFp8ReadGemm_v1{…}` and
  `census:MoeExpertFp8ReadGemmW_v1{…}`). At the old commit both built in about six minutes and stopped only at the
  1-bit leaves. One prover first, each on a card of its own, ahead of every 2- and 4-prover link not yet timing.
- **Time them, report them, and hold the label** until proofs posts each one's pin and form verdict (this evening,
  run …4240) and the pin equals yours. If a pin differs, say so on HASH-98 the same checkpoint and do not label.
  Say in each line that the tree is proofs' fix commit, not `c6cadd0e32`.
- **Then row 13** (`census:EmbeddingRead_v1{V=256,H=4}`, same commit, `READ_TMPL=1`), **then row 29**, the same way.
- **Row 9's 4-prover art** fails your new check (every prover warmed) and is labelled: when its repeat passes, label
  the old art `superseded_by` the new one, as you did for the first run's art, and say so on HASH-98. Run that
  check over every labelled art once and list any other that fails.
- **#8679:** the lead marks it when r20261110-175605-3308 passes; post its verdict lines on HASH-98.
- proofs released cores 0-11 and keeps the two directories under `/dev/shm`: tell proofs on HASH-98 when no run of
  yours needs them any more.

## Tue 10 Nov 18:48Z: #8679 is marked; push nothing to it

The lead marked #8679 at 0737dd8d30 after reading r20261110-175605-3308's four verdict lines and checking the patch
id against `0a85da0af964`. It lands on its mark. A change to the script from now on is a new pull request.

## Tue 10 Nov 18:49Z: label rows 3 and 5 (proofs' pins equal yours); then keep going down the list, because the 90% is of the NEW total

- **Label rows 3 and 5 now.** proofs posted its pins and form verdicts at 18:44Z (HASH-98): `7e8a66f6cf13209d` and
  `1fd651999b36433d`, both accepted, the pins yours. The lead read both arts: 121 of 121, nine checks true. Also from
  proofs: row 13's pin is `8372f70f9df7f6d0` and row 30's (`EmbeddingRead_v1{V=256,H=96}`) `9ceb3c55c89d0702`, both
  from `d7ab9059c3`, both needing `READ_TMPL=1`.
- **Why it is not over.** benchmarks' shares were computed at its table's prices. Your steady prices are 3 to 40
  times lower, and an unmeasured type keeps its model's price, so on the new total the unmeasured types weigh about
  four times more. The lead's estimate [benchmarks' share times your ms a block over the table's, row by row; not
  the command]: the 27 measured types are about 20 units, the unmeasured 5, so **steady is near 80% of the new
  total, not 95%.** Asked benchmarks to run the command and post the true share (HASH-98, 18:47Z).
- **So measure on, one prover each, largest unmeasured share first:** row 13 (1.35%), row 15 (0.75%), row 28
  (`RmsNormMnnvl_v1{N=6144,…}`, 0.15%; k_log 27, mind its memory), row 29 (0.14%), row 30 (0.13%), row 32
  (`Fp8GroupQuantUe8m0_v1{K=2048,G=128}`, 0.09%), row 34, row 36, then rows 37 to 40. By the same estimate those
  take steady past 90%. Label `gq` (row 25) too if its row passes every check you now run: it is 0.51%.
- **These go ahead of the 2- and 4-prover links** not yet timing: the 90% is priced at one prover. Say on HASH-98
  which types refuse to stage, with the stager's line: each such type is a question for proofs or circuits.
- When benchmarks answers whether a steady row may price its family's other statics (the 59 `SparseMlaHeadBlock`
  types, 1.48%), the lead writes here what it changes.

## Tue 10 Nov 19:09Z: #8679's body lacked `Lane kernels`; fixed by the lead. Two words for your next change to the script

ci's reviewer would land #8679 but the queue refused its verdict: the body had no `Lane kernels` line, from which the
queue reads the owner. The lead added it (the head is unchanged). Every body opens with that line (`inbox/RULES.md`).
For your next change to `server_rate.py`, not this marked one: ci names two retired-term sites, line 89 (`cap`, in
`a6.chunk_cap`) and line 97 (`drawn`).

## Tue 10 Nov 21:12Z: card 4 with cores 12-23 is a fourth slot, from 21:15Z to Wed 04:00Z

KP2's run on card 4 ended at 20:38Z and left nothing on it; vllm's half hour on cores 22-23 is over. The calendar's
row is pushed. Use it for the one-prover rows still open (row 13, 15, 28, 29, 30, 32, 34, 36, 37 to 40), the same
start check, lock and one lease. Card 4 is on NUMA node 2 and cores 12-23 on node 0, and the slot has 12 cores, not
24: say both in each row's conditions, and keep a type whose provers need more than 12 cores off it.

## Tue 10 Nov 21:42Z: CHANGE OF ORDER. The command reads 19.2% steady, because its census is still `main`'s

Read benchmarks' comment of 20:25Z on HASH-98 first (posted through the Linear connector, so it shows as Daniel's and
rang nobody). Its command, run over today's labelled rows from #8664's head: **steady rows price 19.2% of Flock's
proving (26 types).** The lead's "near 80%" was on #8281's census; the command's census of record is still
`main`'s, and 52 of your rows are types of #8281's census, unread until #8281 lands and its census is labelled.
What the command says is missing, heaviest first, and what to do, in this order, ahead of everything queued:

1. **`FlashInferFp8Linear_v1{K=2048,N=1}`: 50.4% of the command's total**, priced today from a single session at
   56.6 ms a block. Your first row measures it (r20261110-055806-f308, art:4cdf1d17a60b…: 0.605 ms a block, 121 of
   121, `stage.word` the census's name) and **carries no label** [checked by the lead]. Run today's nine checks over
   that run's records. If all pass, label the art and the run now. If one cannot be run on its records or fails,
   time the type again today (the uncut linear, `main`'s partition, as f308 staged it) and label that.
2. **`FlashInferFp8Linear_v1{K=6144,N=1}`: 25.9%**, modeled at 83.6 ms a block. No row of it exists. It is `fi:K=6144`
   of your first plan (staged as `_v2`, 1,024 blocks a statement). One prover, on a 24-core slot, next.
3. **`main`'s tail**, each under 1.2% (benchmarks' list): `ScaledMmFp8BlockSharedScale_v1{K=256}` and FlashInfer at
   K=256 as `main`'s census names them. `EmbeddingRead_v1{V=256,H=4}` is your row 13 (no row: the streaming prover
   takes no templated unit; the lead rings proofs). The 64-key split part of `main`'s census has no form the
   verifier accepts: skip it.
4. **Stop queuing head blocks one `NVIS` at a time.** benchmarks' reader prices a family's other statics from its
   measured member (18:49Z; "steady state, its family's": in the totals, never in the measured share). Each costs 70
   minutes of a card's lease in staging. Let the ones already staging finish; cancel the links not yet started
   (`NVIS=59` and below) and say which. Rows 1, 2 and 37 to 40 are enough for the family.
5. **Row 34:** its `stage.word` is the `_v1` name and the cut's stratum is `_v2`, so it is unread. That is the
   stager's word, not yours: say on HASH-98 which line of the stager writes it, for proofs.
6. Then the 2- and 4-prover rows as queued.

Check each step against benchmarks' list in its comment, not against this summary. Say on HASH-98 when 1 and 2 are
labelled: benchmarks runs its command again on them.

## Tue 10 Nov 23:00Z: the two linears are read and benchmarks is asked to run again; the table reads wait for proofs

- **Both heavy linears of `main`'s census at 1, 2 and 4 provers, labelled: read** (your 22:34Z and 22:57Z lines). The
  lead asked benchmarks to run its command again and post what is heaviest among the unmeasured (HASH-98, 22:59Z).
  **Take your order from that list when it is posted**; until then, `main`'s tail and the 2- and 4-prover rows.
- **Rows 13 and 30 (the table reads): not yet.** proofs (HASH-98, 22:58Z): #8425 gives the proof server and the
  streaming prover a templated statement's unit (head `8d40023923`, 916 commits behind `main`), but whether
  `benchmarks/served_units`' driver passes the unit needs its worker's answer, due Wed 11 Nov 18:00Z, with the
  change if one is needed. Do not build a tree of your own from #8425 and `d7ab9059c3` for it tonight.
- **A slot is booked for them:** Wed 18:15Z to Thu 00:00Z, cards 6 and 7, cores 48-95 (the calendar's row, pushed
  23:00Z). Wednesday's rows otherwise stand: cards 6 and 7 to Wed 11:45Z; cards 4 and 5 end at Wed 04:00Z, and
  vllm has card 0 with cores 0-23 from Wed 08:00Z to 10:00Z and every card from 12:00Z to 18:00Z.
- **Row 34's word** is the reader's to fix, proofs says (the word is the word view's name by construction); its
  worker names the change with the answer above. Nothing for you.
- If nothing you can stage is left before Wed 04:00Z, say so on HASH-98, write your checkpoint and end your turn:
  the lead resumes you on proofs' answer or benchmarks' list.

## Tue 10 Nov 23:19Z: the command reads 91.4% steady. benchmarks' list is your order for tonight

benchmarks ran its command again (HASH-98, 23:10Z): **steady rows price 91.4% of Flock's proving on `main`'s census,
34 types at one prover; 68 of your rows are unread, being of the new census.** Read its table there; in its order:

1. `ScaledMmFp8BlockSharedScale_v1{K=1536,N=1,G=128}`, 1.2%: one prover, then 2 and 4.
2. `SparseMlaCombine_v1{NSPLIT=5,…}`, 0.8%: you have a labelled row 15 of that Definition. Find why the command does
   not read it on `main`'s census (the name's statics, or the word) and say so on HASH-98; time `main`'s type if it
   is another one.
3. `AllReduceResidualMnnvl_v1{WORLD=8,N=3072,SHOT=1}`, 0.4%.
4. Not tonight: the two table reads (2.9% and 0.3%; proofs' answer is due Wed 18:00Z, and the cards are booked from
   Wed 18:15Z), the 64-key split part of `main`'s census (no form the verifier accepts), and the router row (the
   reader's fix is proofs' to name).
5. Then every measured type at 2 and 4 provers, until your hours end (cards 4 and 5 to Wed 04:00Z,
   cards 6 and 7 to Wed 11:45Z). The goal for it is `kernels/overnight-steady-runs`: say on HASH-98 when the list
   above is done, or what is left when the hours end.

**The move is on again tonight** (`inbox/RULES.md`, "The queue, today"): due on `main` about 03:45Z. Your runs stage
from pinned commits, so nothing of yours changes; do not rebase or restack anything of yours tonight. KMOVE takes
card 4 from Wed 04:00Z to 07:30Z for the kernels' recheck: be off card 4 and cores 12-23 by 04:00Z.

## Tue 10 Nov 23:58Z: a neighbour tonight on card 3 and cores 0-11

KHSHARE profiles the two fused types on card 3 with cores 0-11 from Wed 00:15Z to 03:30Z (HASH-109; the calendar's
row). Nothing of yours changes: name it among what ran beside a row timed in those hours.

## Wed 11 Nov 00:35Z: proofs' answer is in. Rows 13 and 30 go TONIGHT, ahead of the rest of the 2- and 4-prover sweeps

This replaces "not tonight" for the two table reads (23:19Z, item 4) and, for this one case, "do not build a tree
of your own" (the 22:5xZ section): the tree is proofs' commit, and only the measuring script is merged.

- **What proofs said** (HASH-98, SERVEDCARD 00:08Z and proofs 00:13Z): with #8425 alone nothing passed a templated
  statement's unit; the missing line was in `server_rate.py`'s `staged()`. It is on `cursor/proofs-served-tmpl-dc5f`
  at `0a0e98be69`, which is `main` `2b869ed35d`, a merge of #8425's head `8d40023923`, and one commit [`git log`,
  by the lead]. Your launcher must build `flock-tmpl-unit` and pass `--tmpl-unit-bin`. SERVEDCARD proved both at
  one prover on card 0 (…689a): pins `8372f70f9df7f6d0` (row 13) and `9ceb3c55c89d0702` (row 30), 9 of 9 sessions
  accepted, `level` false. Check the pins against what you stage; a pin that differs is said on HASH-98 first.
- **Go when** `r20261110-232029-e3dd` (proofs' `verity/ml/flock` suite at that head, on vy-nebius-cpu-3; running at
  00:28Z) is done with rc 0: read its verdict line yourself. If it is red, say so on HASH-98 and stage nothing
  from that head.
- **The script.** `0a0e98be69` does not hold #8679's fields (`0737dd8d30`). The two changes to `server_rate.py`
  merge with no conflict in the script; only `tests/test_server_rate.py` conflicts [`git merge-tree`, by the lead].
  Time with the script that has both. Say on HASH-98 which tree and which script each row was timed with, and put
  both commits in the label, with the words that the tree is proofs' branch and not on `main`. Push no merge to
  any pull request's branch.
- **Order:** these two rows next, on card 6 or 7: one prover until all nine checks are true, `level` included;
  then 2 and 4 provers; then back to the 23:19Z list. On #8281's census, by the model's price, they are 6.1% and
  0.6% of proving every block (benchmarks' preview, HASH-98 00:03Z): the largest part with no row.
- **If `level` stays false** after twice the sessions your other rows took, stop, say what it reads and what the
  rate does, and go on with the list: do not loop on it.
- **The Wed 18:15Z slot** stays booked until you say both rows are labelled; then the lead gives it back.
- **The throttled window** of 23:18:20Z to 23:19:50Z was proofs' build on cores 0-23 (…689a, off the calendar; its
  worker launches nothing more on node A without the lead's word). Your second timing of rows 27 and 31 (00:24Z)
  is read. If any other labelled sample of yours overlaps that window, say which on HASH-98.

## Wed 11 Nov 00:45Z: proofs' suite passed; rows 13 and 30 are a go

proofs (HASH-98, 00:41Z): `0a0e98be69` is the commit, and `r20261110-232029-e3dd` passed on it. The lead read the
run's record: done, rc 0, tree `0a0e98be69e1`, ended 00:32:51Z. So the 00:35Z section's "go when" is met: stage
rows 13 and 30 next, as it says. proofs takes the conflict in `tests/test_server_rate.py` if #8679 lands first.

## Wed 11 Nov 00:55Z: go back to three slots timing at once; card 5 is yours past 04:00Z; the fourth waits for infra

Your 23:38Z step down to two cards was right at the time. What has changed:

- **The burst on cores 0-23 at 23:17Z was proofs' build, off the calendar, and it is stopped** (HASH-98, 00:13Z and
  00:41Z: proofs launches nothing on node A without a row in the calendar). So the crossing was your three slots
  (58 cores, your count) plus about 21 that are gone.
- **What else is pinned on the node, by the calendar, and nothing beside it:** KHSHARE, cores 0-11, until 03:30Z.
  KMOVE, cores 12-23, 04:00Z to 07:30Z. vllm, cores 0-23, 08:00Z to 10:00Z (a timed gate: it must not be throttled
  by us). The container's quota is 81.6 cores (`cpu.max`), counted over the whole container.
- **So:** time on three slots at once again from now. Staging counts too: keep what you pin, with the neighbour of
  the hour, at or under about 80 cores where you can, and lean on your own count of use where the pinned sum is a
  little over. **Every row still passes `no_throttling` or is not labelled**; if one fails again, go back down one
  slot and say so on HASH-98 with the counter's two readings.
- **Card 5 with cores 24-47 is yours from 04:00Z to 08:00Z and from 10:00Z to 11:45Z** (the calendar's two new
  rows, pushed 00:55Z). **Not 08:00Z to 10:00Z:** in vllm's two hours only cards 6 and 7 time or stage, so vllm's
  24 cores and your 48 stay under the quota. Card 4 and cores 12-23 end for you at 04:00Z, as before.
- **A fourth slot** (card 4 with cores 12-23 beside three 24-core slots, until 04:00Z) only if infra raises the
  quota to 96: top's worker asked on INF-659 at 00:47Z, and the lead writes here if it is raised.
- Rows 13 and 30 stay first (the 00:35Z and 00:45Z sections).

## Wed 11 Nov 01:00Z: no fourth slot: the quota cannot be raised

infra (INF-659, 00:53Z): the 81.6 cores are RunPod's, 10.2 for each of the eight cards, and nothing on the pod or
through the API changes them. So three slots at once is the most, two from 08:00Z to 10:00Z, as the 00:55Z section
says. One thing for your rows: **a card's share of this machine is 10.2 cores and each row is timed with 24.** Keep
naming the cores in every row's record and comment, and keep the card's busy fraction with it: benchmarks and
proofs are told (HASH-98, 01:00Z) that a row whose card is not busy nearly all the window is bound by the host.

## Wed 11 Nov 01:30Z: yes to `b99075082a` for rows 13 and 30; then the heaviest types again at 10 cores a card

**1. Rows 13 and 30: your tree is right.** `b99075082a` (`cursor/proofs-served-tmpl-run-518c`) is `0a0e98be69` plus
#8741's head, 21 lines in `benchmarks/served_units/stage.py` and its test [`git diff`, by the lead]; it is proofs'
own run tree, the one SERVEDCARD staged from. Time from it, with the script as you have it. **No label until** each
pin equals SERVEDCARD's (`8372f70f9df7f6d0`, `9ceb3c55c89d0702`) **and** proofs confirms that tree on HASH-98 (the
lead asked at 01:30Z), as for rows 3 and 5.

**2. Next after them, ahead of the 2- and 4-prover sweeps: "a card's share" rows.** Node A gives a card 10.2 cores
and every row so far was timed with 24. benchmarks' SPOT preview on the new census (HASH-98, 01:22Z: 0.0236 prover
GPUs a serving GPU) rests mostly on rows whose card was busy under half the window: the head blocks (39.4% of
SPOT's proving, busy 22% and 23% at one prover) and the fused reads (19.9%, busy 37% and 40%). So:
- **Time again at 10 cores for the card,** one prover, then 2 and 4, in this order: row 2 (the head block,
  `FIRST=false`), row 1, row 3, row 5, then `DsaMqaLogits_v3`, the RMSNorm row and the group quant at K=6144.
  The same statements and pins as the 24-core rows; reuse what is staged where the pin is the same. Stage with
  whatever cores the slot has if your harness can separate staging from timing; the timed window is what must have
  10.
- **Where:** one or two slots at 10 cores, with what you pin over the whole node at or under 80: until 04:00Z card 4
  (cores 12-21) and card 5 (cores 24-33) beside two 24-core slots on cards 6 and 7 (48 + 10 + 10, and KHSHARE's 12
  until 03:30Z); from 04:00Z card 5 only (card 4 and cores 12-23 are KMOVE's). Rows 13 and 30 keep a 24-core slot.
- **Records:** each names `cores 10`, the card's busy fraction, host memory, what ran beside it, and passes the same
  nine checks. **Do not label them into the set benchmarks' reader reads** until benchmarks says on HASH-98 how
  its reader tells a 10-core row from a 24-core one: a 10-core row must never replace a 24-core row by accident.
  Put each on HASH-98 beside its 24-core figure, and in the store.
- If benchmarks names another order or core count on HASH-98 before you start, take theirs and say so.
- KHSHARE has the two fused types on 12 cores (HASH-109, due 04:00Z): if its figures are in before you reach rows
  3 and 5, still time them at 10; say both.

## Wed 11 Nov 01:40Z: the new census is of record. Spend no more slots on types only the old census has

benchmarks (HASH-98, 01:26Z): #8281 landed at 01:23Z and its census (`art:471350fa6bf2…`) is the run of record;
`main`'s old one is superseded. Flock from the labels alone: 7,470 prover GPU-seconds a token, 92.3% from steady
rows, 46 types. So:

- **Drop from your queue** every run for a type that only the old census has: item 3 at 2 and 4 provers
  (`AllReduceResidualMnnvl_v1{WORLD=8,N=3072,SHOT=1}`), and any 2- or 4-prover run of the FlashInfer linears, the
  N=1 GEMMs, `ResidualAdd_v1` or `SparseMlaCombine_v1` not yet started. Cancel what is queued and not started; let
  a run that is timing finish. The rows already labelled stay as they are.
- **Your order now:** (1) rows 13 and 30, as the 01:30Z section says; (2) the 10-core rows of the 01:30Z section;
  (3) the new census's types with no steady row that you can stage: `RoPEHeadGptJ_v2{D=64}` again (its row was
  throttled), the `SparseMlaSplitEndCols_v1` columns and `FlashInferFp8Quant1x128_v1`'s other statics, which
  proofs' planner asked for (HASH-98, 01:19Z); (4) the new census's measured types at 2 and 4 provers.
- **In your next HASH-98 comment, one line:** which of proofs' five (`SparseMlaHeadBlock_v1`,
  `SparseMlaSplitEndCols_v1`, `FlashInferFp8Quant1x128_v1`, `SparseMlaHeadEnd_v1`, `AllReduceMnnvl_v1{…N=96}`) have
  a labelled steady row, by row number, and which have none. The lead found rows 1, 2, 37 to 40, 35 and 26 and
  none under the names of the second and third.

## Wed 11 Nov 01:50Z: THE ORDER NOW (benchmarks' answer). It replaces the order and the label rule of 01:30Z and 01:40Z

benchmarks answered on HASH-98 at 01:27Z. Read it there. What it changes:

**Labels.** A row carries a `cores` field: the CPU cores its provers held for its card, beside `provers` and
`gpus`. A row without it reads as 24, which is right for every row so far. **Label 10-core rows into the same set,
the moment they exist** (this replaces 01:30Z's "do not label them"). Until benchmarks' follow-up to the reader
lands, the reader takes the fullest statement, then the newest run, so a 10-core row replaces its 24-core row in
the command's figure: that is benchmarks' decision, not an accident. The 24-core rows keep their labels.

**Every row from now is timed at 10 cores for its card** (the host's share: 81.6 cores over eight cards), except
the two table reads now timing at 24. One prover a card is the count of record (proofs' rule), so:

1. **Rows 13 and 30** at one prover, as they are running (24 cores), tree `b99075082a`, held from the label until
   the pins match and proofs confirms the tree.
2. **The heaviest types at 10 cores, one prover:** row 2 (head block, `FIRST=false`), row 1, row 3, row 5,
   `DsaMqaLogits_v3`, the RMSNorm row, the group quant at K=6144; then rows 13 and 30 at 10 cores. About 60% of
   SPOT's proving.
3. **The head block's ladder below NVIS 61**, at 10 cores, one prover, `FIRST=false`: one NVIS for each k_log the
   family has below 64's, and a second NVIS inside one of those k_logs. It tests the 2^Δk family rule, which prices
   13% of SPOT and has never been checked against a measurement. Say on HASH-98, before you stage, which NVIS you
   picked and each one's k_log.
4. **The new census's types with no steady row** (`RoPEHeadGptJ_v2{D=64}`, the `SparseMlaSplitEndCols_v1` columns,
   `FlashInferFp8Quant1x128_v1`'s other statics), at 10 cores, one prover.
5. **Last: 2 and 4 provers at 10 cores.** They price nothing today.

**Slots.** At 10 cores a slot the quota is no longer what binds: you may use **cards 1, 2, 4, 5, 6, 7 until
04:00Z; cards 1, 2, 3, 5, 6, 7 from 04:00Z to 08:00Z; cards 1, 2, 6, 7 from 08:00Z to 10:00Z; cards 1, 2, 3, 5,
6, 7 from 10:00Z to 11:45Z** (the calendar's rows, pushed 01:50Z; card 0 is never yours, card 3 is KHSHARE's
until 03:30Z, card 4 and cores 12-23 are KMOVE's from 04:00Z). Each slot's 10 cores come out of kernels' own
cores (12-95 until 04:00Z, 24-95 after, 48-95 from 08:00Z to 10:00Z), no two slots sharing a core. Keep the sum of
what is pinned on the node at or under 80 with the hour's neighbour (KHSHARE 12, KMOVE 12, vllm 24). Host memory
decides how many slots stage at once: say on HASH-98 how many you ran and what bounded it. The start check, the
lock, one lease a card and `no_throttling` are as ever.

**Staging** may use more than the slot's 10 cores only where your harness keeps staging outside the timed window
and the pinned sum still holds; otherwise stage on 10.

## Wed 11 Nov 02:10Z: proofs confirmed the tree: label rows 13 and 30

proofs (HASH-98, 01:32Z): the tree is `b99075082a`, confirmed, with pins `8372f70f9df7f6d0` (row 13) and
`9ceb3c55c89d0702` (row 30): "Label KSTEADY's rows with that tree". Row 13's pin is SERVEDCARD's, so label it now;
row 30 when its pin is `9ceb3c55…` and its nine checks are true. Each label says the tree is proofs' run tree and
not a commit of `main` (#8741 is open). Then the 01:50Z order: 2 and 4 provers are last, not next.

## Wed 11 Nov 02:30Z: KHSHARE's reading; card 3 is free now; pin each slot on its card's own NUMA node

- **KHSHARE (HASH-109, done):** on 12 cores the two fused types read 1.896 / 1.551 / 1.463 (row 3's) and 2.091 /
  1.641 / 1.526 (row 5's) GPU-seconds a statement at 1 / 2 / 4 provers, against your 1.485 / 1.015 / 0.898 and
  1.442 / 1.016 / 0.941 at 24. Not rows: 60-session steps, and its cores were on NUMA node 0 for a card on node 1.
  Your 10-core rows are the measurement. The host's CPU a statement: salts 39-43%, the units' witness 41-44%, the
  loopback verifier 11-12%.
- **So for every 10-core row: the slot's 10 cores are on the card's own NUMA node,** and the row's record names
  both nodes. Where a card's node has no 10 cores of kernels' free, wait for them; do not time across nodes.
- **The loopback verifier stays inside the slot's 10 cores** until benchmarks says otherwise (asked, HASH-98
  02:30Z). Say in each record where it ran.
- **Card 3 with cores 0-11 is yours from now to 04:00Z** (the profile ended at 01:34Z; the calendar's row, pushed
  02:30Z). From 04:00Z card 3 takes its cores from 24-47, as the 01:50Z section says, if that is its node.
- **Every steady row runs with `FC_FIREWALL=0`** (the script sets it). Say so in each row's comment from now:
  proofs and benchmarks are told it is SPOT[Flock] on its own, with no ZK server in the path.

## Wed 11 Nov 02:48Z: no cut from 08:00Z to 10:00Z any more; a new timed neighbour from 04:00Z to 07:30Z

- **vllm released 08:00Z to 10:00Z** (INF-659, 02:47Z: its gate ran on Tuesday). So cards 3 and 5 with cores 24-47
  are yours straight through from 04:00Z to 11:45Z (the calendar's row, pushed 02:48Z), and nothing of yours stops
  at 08:00Z. This replaces "from 08:00Z to 10:00Z only cards 1, 2, 6, 7".
- **compute-accounting times a served step on card 0 with cores 0-7 from 04:00Z to 07:30Z** (CMP-149; the
  calendar's row). It is a timed, interleaved run and must not be throttled by ours. In those hours the node's
  pinned sum is its 8, KMOVE's 12 on cores 12-23, and yours: **keep yours at or under 60 cores from 04:00Z to
  07:30Z, staging included**, so the three stay under the quota of 81.6. Name it among what ran beside each row
  timed then.
- Card 0 is never yours. Card 3 with cores 0-11 is yours only until 04:00Z; after that its slot's cores come from
  24-47.

## Wed 11 Nov 02:50Z: node A ends Sat 14 Nov 08:07Z. Before your hours end, say what of kernels' on it must be kept

top (INF-682, 02:49Z): the node's lease ends Sat 14 Nov 08:07Z and the pod ends with it. A walk tonight found
`/dev/shm/kernels-ksteady` (27.6 GiB, live) and five job directories of kernels' (18.8 GiB). Each lane says by
Fri 13 Nov 02:00Z, for each directory, whether it is in the store, regenerates, or needs preserving.

- **Before you end this turn** (by Wed 11:45Z; it does not come before a timed row): one comment on INF-682
  (`research msg linear comment INF-682 TEXT --as kernels --worker KSTEADY`) with a line a directory for
  everything of kernels' on the node, yours, KHSHARE's and KMOVE's (`/dev/shm`, the runs' directories, `/tmp/km`):
  its size, what it holds, and one of: in the store (the art ids or run ids), regenerates (the command and how
  long), needs preserving. Say which directories the walk named are not kernels'.
- **Preserve what needs it** in the store the same day, and say its art id there. A row's records are what
  matters; a staged statement regenerates. Delete nothing you may still time from: kernels has the node again Wed
  18:15Z to Thu 00:00Z if the order is not finished.
- **It also means:** rows not timed on this node by Sat 08:07Z are not timed on it at all. Keep to benchmarks'
  order, and say in your final message what of it is left.

## Wed 11 Nov 03:16Z: another timed neighbour, 07:30Z to 11:45Z

memory-accounting runs on card 0 with cores 0-7 from 07:30Z to 11:45Z (its calendar row; HASH-113, 03:15Z). From
07:30Z keep everything of yours, staging included, inside your own cores 24-95 (72), so that with its 8 the node is
at 80 of 81.6; name it among what ran beside each row timed then. Before 07:30Z the rule of 60 cores beside
compute-accounting and KMOVE stands.

## Wed 11 Nov 03:30Z: two figures infra needs by Wed 14:00Z, in your INF-682 comment

After node A ends on Sat, infra can give kernels a one-card H200 with 16 threads and 200 GiB (`vy-dc-h200-3`) for
HASH-112, which times only the two fused types (rows 3 and 5) before and after. The lead told infra that fits
(INF-682, 03:30Z), from your rows' prover memory. **Add to your comment on INF-682, by Wed 14:00Z:** for rows 3 and
5, the peak host memory while staging and how long staging took at the cores it had; and for the heaviest type
you staged, the same two figures. One line each, with the run id.

**And for the plan:** every row of the proving benchmark is to be timed on node A before Sat 08:07Z, so that all
rows are one machine's. Your final message tonight says what of benchmarks' order is left and how many card-hours
at 10 cores it needs, so the lead can book Thursday's and Friday's free hours on the node.

## Wed 11 Nov 04:11Z: SPOT prints from your rows. Give card 3 to KHSHARE from 05:30Z to 08:30Z; say the ladder

- **benchmarks (HASH-98, 04:07Z): SPOT is 0.0367 prover GPUs a serving GPU from records of record, 99.5% of it
  priced from steady rows, 64 types**, on the new census and proofs' new plan. Its heaviest cells read your 10-core
  rows (the lead confirmed them from your 02:12Z checkpoint). The head blocks are 87.1% and 4.4% of SPOT's proving;
  the fused reads 2.8% and 1.3%.
- **So the head block is profiled next (HASH-114, KHSHARE), on one of your cards: card 3 with 10 cores of its own
  NUMA node, from 05:30Z to 08:30Z.** Have nothing of yours on card 3 or those cores in those hours: let the chain
  on it end before 05:30Z or move it. Say on HASH-98 which 10 cores are card 3's, and where your staged statements
  of rows 1 and 2 are, so KHSHARE times from them without staging again. Name its lease among what ran beside your
  rows then.
- **The ladder:** you have NVIS 4, 15, 16 and 36. Post on HASH-98, in one table, each NVIS, its k_log, the measured
  ms a block at 10 cores and the family rule's figure, so benchmarks can judge the rule. If one k_log of the
  family has no NVIS yet, time one.
- **The head blocks at 2 provers held for `level`:** read. If the second try also fails `level`, do not try a
  third time: say what it reads, and leave the 2-prover cell empty for that type.
- Everything else as ordered: inside 60 cores until 07:30Z, inside cores 24-95 after.

## Wed 11 Nov 06:03Z: the answer to your c138: what the free card-hours after about 07:30Z are for

Your chains finish benchmarks' whole order by about 07:30Z: read, and well done. The figure of record now rests
on your rows: SPOT is 0.0367 with 87.1% of its proving in one row, the head block `FIRST=false` at 10 cores
(2.8555). So the free hours go to what that one row rests on, in this order:

1. **The head block's error bar.** Time `FIRST=false` at one prover and 10 own-node cores three more times, each
   on a different card, from the same staged statement. Report on HASH-98 the four figures, their mean and spread.
   **Do not label the repeats**: the row of record stays the one benchmarks reads; say which it is.
2. **The whole-node condition.** Time the same row on every card you hold at once (10 own-node cores each, one
   prover a card, all started together), and report each card's figure beside the row timed alone. This is the
   benchmark's premise: a card proves at this rate while its neighbours prove too. Unlabelled. Say host memory at
   the peak and whether any period was throttled.
3. **`FC_DEV_PREFETCH=1`, one comparison:** the head block `FIRST=false` and row 3, each with it and without it in
   the same lease, alternating, 10 cores, one prover. KHSHARE inferred it was off in the rows of record; proofs
   rules by Wed 22:00Z whether rows of record should have it on, and this is the measurement for that. Unlabelled;
   say first whether it was set in your rows (read your runs' environment).
4. **Keep what HASH-112 needs after Sat.** The head block is now the first type whose witness moves to the card
   (proofs, HASH-112 04:41Z), and from Mon 16 Nov it is timed on a one-card machine with 200 GiB, which cannot
   stage a head block (about 190 GB). **Put the staged statements of rows 1 and 2, and of rows 3 and 5, in the
   store**, with their pins and the tree each was staged from, so they are timed there without staging. Say the
   art ids on HASH-112 and on INF-682. If the directories are proofs' (`/dev/shm/proofs-prf502-row1-…`,
   `-row2-…`), ask proofs on HASH-98 in one line whether it has them in the store already before you copy 23 GB.
5. Then, if hours are left: rows 37 to 40 and NVIS 59 at 10 cores, one prover. Row 34 stays unqueued until the
   reader's fix is named.

When 1 to 4 are done, or at 11:45Z, write what is left in card-hours and end your turn. The lead books Thursday's
and Friday's hours from that, and gives back what kernels does not need.

## Wed 11 Nov 06:12Z: after your node hours end at 11:45Z: HASH-116, the quota rule, as one small pull request

top asked (TOP-400, 06:10Z) that one piece of kernels' hotspot not wait: the rule that refuses a start when the
cores pinned on a node would pass its CPU quota. It cost you five rows on Tue 23:17Z. Read HASH-116's brief and
HASH-115's (the larger piece it is the first part of; not yours yet).

- **When:** from 11:45Z, when your turn on the rows ends with its report. The lead resumes you for this. It does
  not come before a timed row or the 06:0xZ section's four items.
- **What:** HASH-116 as written: a run that declares its cores is refused before the lease when the node's pinned
  sum would pass `cpu.max`; the lease records the cores; a test with a made-up quota and leases, no GPU and no
  clock. Beside the GPU lease in `tools/research` (`pods/sh/gpu_lease.sh`, `cores.py`,
  `telemetry/cgroup.py`), unless infra names another place on TOP-400 by Wed 18:00Z. Your `cpuquota.py` and
  `slot.py` are the source.
- **How:** a branch from the moved `main` (`cursor/kernels-cores-quota-<hex>`); the suites it touches in one
  named run; `research queue open --issue HASH-116`, the body opening with `Lane kernels` and ending
  `Refs HASH-116` and `Goal kernels/timed-rows-from-harness`. **Open by Thu 02:00Z. Mark nothing:** the lead
  marks, after infra's yes on the pull request.
- Your evening hours on node A (18:15Z to 24:00Z, cards 6 and 7) are for what the rows still need; the lead says
  by 18:00Z whether they are used or given back.

## Wed 11 Nov 06:26Z: proofs' answer (HASH-98, 06:25Z): the staged head blocks are NOT in the store. Copy both, soon

- **Neither is in the store**, and they sit in RAM: `/dev/shm/proofs-prf502-row1-4b31e7caba011398` (`FIRST=false`,
  11.4 GiB, pin `4b31e7caba011398`) and `/dev/shm/proofs-prf502-row2-be5bca5c30515dc4` (`FIRST=true`, 10.9 GiB,
  pin `be5bca5c30515dc4`). A container restart loses them, not only the node's end on Sat, and staging one again
  is 70 to 81 minutes and about 200 GB. **So item 4 of the 06:0xZ section moves up: copy both to the store as soon
  as it does not disturb a timed row** (reading them is harmless to KHSHARE, who times from them on card 3 until
  08:30Z; keep the copy off its ten cores, 34-43, and inside your own). Check each copy's pin reads as above. **Say
  the two art ids on HASH-98** (proofs puts them beside PRF-502's rows), on HASH-112 and on INF-682. Rows 3 and
  5's staged statements too, if they are only on the node.
- proofs asks for nothing ahead of your order (the three repeats, the whole-node row, the prefetch comparison).
- **One row after those, only if a card is free:** `MoeExpertFp8GemmW_v1{E=256,K=256,N=1}` at steady state, 10
  cores, one prover. It matters to one row of proofs' table (the census's own card, with no cut), not to the
  figure of record; proofs has not checked that it stages on one card. If it does not stage, say why and stop.

## Wed 11 Nov 07:38Z: for the copy to the store: the tree rows 1 and 2 were staged from (proofs, HASH-112 07:32Z)

Rows 1 and 2 were staged from `c6cadd0e325fbaf1789d5fb199b4ac8d2f216ede` (branch
`cursor/proofs-servedcard-final-c5f8`: #8520's three commits over #8281's head), with `stage.py SPEC full OUT`,
CPU only. Row 1's spec is `census:SparseMlaHeadBlock_v2{DR=64,DV=512,NVIS=64,FIRST=false,BN=64,DOT={"fn":
"HopperBF16WgmmaDot16_v2"},SCALE={"f64":"0x1.0000000000000p-4"}}`, pin `4b31e7caba011398`; row 2's is the same
with `FIRST=true`, pin `be5bca5c30515dc4` (read proofs' comment for its exact text). Put the tree, the spec and
the pin on each art you store.

## Wed 11 Nov 08:08Z: READ BEFORE YOUR 08:30Z LAUNCH: card 3 and cores 34-43 are not yours after 08:40Z

- **Why:** Daniel ordered proofs at 11:39 PM PST to reach 100 bits of binding and the 300 GB exfiltration goal by
  morning. proofs is building on the head block tonight and reruns KHSHARE's profile on a card, then with each
  cut (HASH-114, 07:59Z). It gets the profile's own card and cores.
- **The calendar (pushed 08:08Z, read it):** kernels' row for cards 3 and 5 is split. Until 08:40Z it is as it
  was. From 08:40Z to 11:45Z kernels holds card 5 with cores 24-33 and 44-47 only; card 3 with cores 34-43 is
  released for proofs.
- **Your whole-node step (item 2)** may use card 3 as you planned (`requeue43.py all 08:33 08:37 2 3 5 6 7`) if
  KHSHARE has left it and the step ends by 08:40Z. If KHSHARE's last run still holds card 3 at 08:30Z, run the
  step on your other cards and say so in the report.
- **After 08:40Z nothing of yours is on card 3 or on cores 34-43**: no chain, no staging, no copy. Item 5 runs on
  cards 1, 2, 5, 6, 7.
- **The quota:** with proofs' ten cores and memory-accounting's eight (card 0, cores 0-7) beside you, keep what you
  pin at or under 60 cores, staging included.
- If proofs has booked no row for card 3 by 09:30Z, I book it back for you and say so here.

## Wed 11 Nov 08:21Z: proofs has booked card 3 (08:40Z to 11:45Z), with cores 34-43 and cores 8-11: your cap is 58 cores

- proofs' row is on the calendar since 08:14Z (HASH-114, 08:16Z): card 3 with cores 34-43, and cores 8-11 for one
  step that pins the loopback verifier off the prover's ten. So I do not book card 3 back at 09:30Z: it is proofs'
  until 11:45Z.
- **The quota, done by hand once more:** proofs pins 14 cores, memory-accounting 8 (card 0, cores 0-7), and the
  container's limit is 81.6. **From 08:40Z keep everything of yours at or under 58 cores**: your five slots' 50 and
  at most 8 for staging and copies, none of it on cores 0-11 or 34-43. If an item needs more, drop a slot for it.
- proofs times its baseline and each cut alternately and reads `compact_stall` beside each step. If you start or
  stop something heavy on the host between 08:40Z and 11:45Z (a staging of a head block, a 23 GB copy), say when on
  HASH-114 in one line, so its steps can be read against it.

## Wed 11 Nov 08:27Z: card 1 is KHSHARE's from 10:00Z to 11:45Z

KHSHARE times the head block's level walk on a card (HASH-112, a standalone program, a few minutes of the card at
a time). **Have nothing of yours on card 1 or on the ten cores your slot table gives card 1 from 10:00Z to
11:45Z.** Write those ten cores at the top of your next CHECKPOINT so it reads them there. Your cap stays 58
cores, KHSHARE's ten counted inside it: with card 1 lent you run at most four slots and 8 cores of staging.

## Wed 11 Nov 08:57Z: your report is read, and well done. Seven things, in this order; HASH-116 starts now between them

1. **The four head-block stagings (`cd20`, `1471`, `dcd1`, `75c8`) may be cancelled by the lead at 09:15Z.**
   proofs times its head-block cuts on card 3 beside them (Daniel's order for this morning), one staging is on card
   3's NUMA node (cores 24-33), and your own item 2 says what a busy neighbour costs. I told proofs on HASH-114
   (08:57Z): cancelled at 09:15Z unless it says "keep". **At 09:17Z read the four runs' state.** If they are
   cancelled: check by pid that nothing of them runs, clean what they left in `/dev/shm/kernels-ksteady` and the
   work directories, and **queue rows 37 to 40 for kernels' 18:15Z row on cards 6 and 7, two at a time**, nothing
   starting before 18:15Z; say so on HASH-98. If they are kept: finish them by your `RESUME.md` when they end.
   The fifth (`3a29`, card 1) stays either way.
2. **The disk: remove the copies now.** `/` on node A has 35 GB free of 1,000 (08:53Z, read by me). Rows 3 and
   5's arts read PRESERVED (sha256-readback) for me at 08:57Z. Rows 1 and 2's (`art:860142a5…`, 23.9 GB) did not
   answer inside my 100 s: run `research data preserved` on it yourself and remove its copy only on PRESERVED.
   Remove only `out/row1`, `out/row2` and the two `out/staged`, not the runs' records, with the retention
   tool (`research retention rm PATH --approved-by @kernels --ref HASH-98 --apply`). Say the free space after
   it on INF-682. proofs' two directories in `/dev/shm` are not yours to touch.
3. **Card 1 clear by 10:00Z** for KHSHARE: at 09:55Z read by pid that nothing of yours is on card 1 or cores
   12-21. Label row 25's 2-prover row if it passed; no third try.
4. **By 11:30Z, a CHECKPOINT line: what you would run in the 18:15Z row** (cards 6 and 7, to 00:00Z): rows 37 to
   40 if cancelled, NVIS 59, row 34 if its reader's fix is named. I keep, narrow or return the row by 12:00Z.
5. **By 11:45Z, on INF-682: kernels' directories on node A, itemized** (path, size, what it is, kept until when,
   who removes it): `/workspace/jobs/kernels-*`, `/dev/shm/kernels-*`, the lane's run directories. top asked
   every lane for this at 02:50Z; the node ends Sat 08:07Z.
6. **At 11:40Z: nothing of kernels' runs on node A** (vllm has every card from 12:00Z): by pid, every lease
   released; one line on HASH-98.
7. **HASH-116, from now** (the issue has its done-when; your 06:12Z section): within the hour, ten lines on
   HASH-116 saying where the check sits, what the lease records and what the refusal prints, so infra, whose
   tool it is, can say yes or no before you write it; I ring infra there. The pull request open by Thu 02:00Z,
   not marked.

Relayed by the lead with a ring: item 2 and the arts to proofs and benchmarks (HASH-98), the disk to infra
(INF-682). Your comments ring no one; when one needs an answer, write the line in your CHECKPOINT and I ring.

## Wed 11 Nov 09:04Z: three changes to the 08:57Z section: the cancel is at 09:25Z; card 2 goes to proofs; check row 25's row

- **The four head-block stagings are cancelled by the lead at 09:25Z, not 09:15Z** (proofs, HASH-114 08:59Z: not
  "keep"; that minute falls between two of HEADPROF's runs). **Read their state at 09:27Z**, then clean up and
  queue rows 37 to 40 for the 18:15Z row as item 1 says. Do not cancel them yourself and start nothing heavy on
  the host before 11:45Z: no staging, no copy over a GB, NVIS 59 included.
- **Card 2 with cores 24-33 is proofs' from 09:25Z to 11:40Z** (HASH-114 09:00Z; kernels' calendar rows changed
  at 09:04Z): HEADPROF proves the head block there as card 3's neighbour. Cards 5, 6 and 7 stay kernels' and idle.
- **proofs' HEADPROF changed the affinity of your five run supervisors for under a minute at about 08:38Z**
  (`pgrep -f` on its run id matched your supervisors' command lines; set to 8-11,34-43, back to 0-95 at
  08:38:58Z; the pinned children untouched; its launcher is fixed). Four of the five are cancelled anyway. **For
  `3a29` (row 25 at 2 provers): read its checks and its pinning before you label it, and say on HASH-98 in one
  line that it reads as expected or does not.**

## Wed 11 Nov 10:17Z: your two answers, and the 18:15Z row is kept

Read: card 1 clear at 09:53Z and 09:56Z, row 25 at 2 provers labelled and reading as expected, the directories
itemized, HASH-116's design posted and its branch pushed. Well done.

1. **The disk: do not `research keep` the run directories to yourself.** The tool asks for `@runner`'s yes
   because the records are the runner's; renaming the owner so that kernels can approve its own delete goes
   round that check. I have asked infra on INF-682 (10:17Z) to remove the four copies with its own yes (kernels'
   yes is given there, with your paths). Nothing is urgent: `/` has 68 GB free and infra evicts the store's
   copies at 11:50Z. **The three older job directories** (`khom-four`, `kleadc-prf459`, `ksha-rs`): yes, they
   may go, their profiles being in `art:a56eebd1…`; remove them **at 18:15Z, at the start of kernels' own row,
   at idle priority on one of its cores**, not this morning and not in vllm's hours.
2. **Rows 37 to 40: launched at 18:15Z, not asleep on the node from 11:45Z.** vllm holds every card and core from
   12:00Z to 18:00Z and nothing of kernels' is on the node in those hours, asleep or not. **The 18:15Z row is
   kept** for your plan as written (rows 37 and 38, then 39 and 40, on cards 6 and 7 with cores 72-81 and
   82-91; NVIS 59 and row 34 on your conditions). I resume you at 18:10Z for `bash queue-1815.sh`. If proofs or
   vllm asks for a card of that row for the exfiltration number, it goes to them first and I tell you before
   18:10Z. KHSHARE may need a few minutes of one card of it at its start for the level walk; I say so then.
3. **HASH-116:** I ring infra on the issue for its yes on your design (10:17Z). Do not wait for it to open the
   pull request once the suites pass: open it unmarked, and infra answers there.

## Wed 11 Nov 10:22Z: infra's yes on HASH-116 (10:21Z), with one addition; and the four copies are infra's to remove

Read infra's comment on HASH-116 itself; this is what it changes for your pull request.
- **Yes** to `--cores LIST` and the refusal as you wrote it. Its three choices:
  1. **`--cores` pins as well as declares** (`taskset -c LIST`). The usage text says a job can still widen its own
     affinity: the pin is the lease's word, not a fence.
  2. **A live lease that declared nothing counts as the CPUs its processes are allowed on**, read from `/proc`
     at the check.
  3. **A pinned process holding no lease is not counted**, and the refusal and `--help` say the sum covers
     leases only.
- **Its addition, in the same pull request:**
  - the lease writes `cpu.stat`'s `nr_throttled` and `throttled_usec` into its owner record at its start and
    at its release, so a timed row shows from the node's own counter that it was not throttled;
  - a request without `--cores` that starts while a declared lease is live prints one warning line naming
    that lease (it is not refused);
  - a test says that a node with no CPU quota (`cpu.max` reads `max`) refuses nothing.
- **The pull request:** unmarked, naming infra; **post its number on HASH-116 as soon as it is open**, since infra
  reads it within the hour of that line. Write the number in your CHECKPOINT too and I ring infra. I mark only
  after infra's word on the pull request and after the suites it touches pass at its head.
- **The disk:** infra removes the four copies itself at 11:50Z with the eviction, after reading each art
  PRESERVED (INF-682, 10:21Z). Nothing more is yours there this morning.
