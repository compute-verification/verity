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

## Wed 11 Nov 10:35Z: the 18:15Z row is probably given back at 12:00Z: prepare nothing more for it

benchmarks (HASH-98, 10:33Z): its reader now prices at the host's cores (#8911) and SPOT stays 0.0367; **0.036% of
it rests on rows above the 10-core share**, and those are the head block `FIRST=true` at NVIS 59 to 63 and two
`Dsa` rows at 24 cores. So rows 37 to 40 and NVIS 59 would move at most 0.036% of the figure for five 190 GB
stagings and about 11 card-hours. **I told benchmarks (10:35Z) that kernels does not time them again and gives
the row back at 12:00Z unless it says by 11:45Z that the record needs them.** Keep `queue-1815.sh` as it is; do
nothing more for the row; I write here at 12:00Z which it is. The three older job directories then stay until
the node ends. benchmarks' one ask, `server_rate.py` writing the cores count on a row, goes with HASH-115, not
this week. HASH-116 is your work now.

## Wed 11 Nov 11:43Z: your return is read: node A clear, #8909 open. Next, in this order: HASH-98's closing account, #8909 to its mark, HASH-115's design

Read by me: #8909's body and the whole diff of `gpu_lease.sh`; `r20261111-104037-5660` and `…42e5` rc 0 at the head
`13678330b87f`; the seven suites are exactly what the change touches (`suites.py --changed BASE --list`). I rang
infra on HASH-116 with the number; I mark after its word. Good work, and a clear body.

1. **HASH-98's closing account, first, while it is fresh** (one comment on HASH-98, under 200 words, then go on):
   how many block types have a steady row and at how many cores; where the list of labelled rows is
   (`labels-since-0053Z.md`, and the earlier ones); what is left and why (row 34's reader fix; the five head-block
   rows and two `Dsa` rows above the 10-core share, 0.036% of SPOT by benchmarks' reader, not timed again); what
   is held and unlabelled (the four cross-node rows, `4a11`, `17b1`, `bead`) and whether each should be dropped;
   that node A is clear of kernels. I close HASH-98's timing on Thursday from that comment.
2. **#8909 to its mark.** Read its comments at each CHECKPOINT until infra's word. If infra asks for a change:
   make it, run the touched suites at the new head, and **end your turn with the head and the run** (I read and
   mark). If infra says yes with no change, say so in your CHECKPOINT and go on; I mark.
   - **One thing for the pull request's discussion, from me:** nothing passes `--cores` yet, so the check
     protects no run until a launcher does. Say there in two lines what it takes for kernels' own launcher
     (`ksteady-jobs/link.sh`) to pass it, and that infra installs the new `gpu-lease` on a node
     (`research deploy install`, or whatever a pod takes): ask infra which.
3. **HASH-115, the design, on the issue for benchmarks, by Thu 12 Nov 02:00Z** (benchmarks owns the kit; read
   `benchmarks/harness/README.md` first): in at most 40 lines, the proof-server point (its spec's fields: block
   type, staged statement and its pin, provers, cores, card; its verdict, the nine checks; its label and the
   row's `cores`, which answers benchmarks' ask that `server_rate.py` write the count); how `plan`, `run` and a
   resume map onto what the harness already has; the slot as `gpu-lease --on CARD --cores LIST` with cores of
   the card's own NUMA node; which of `ksteady-jobs`' scripts each piece replaces, and what the harness cannot
   do. **End your turn with that comment** (the new rule in RULES.md: a result another lane waits for ends your
   turn): I ring benchmarks and resume you. Dates on the issue: the quota rule and the point on `main` by Tue
   17 Nov, all of it by Fri 20 Nov.
4. **The 18:15Z row:** the lead writes a section here just after 11:45Z saying whether it is given back. Do
   nothing for it until then.

## Wed 11 Nov 11:46Z: the 18:15Z row is given back: rows 37 to 40 and NVIS 59 are not timed again

benchmarks said nothing by 11:45Z (HASH-98, read at 11:46Z), so the default stands. **Do not launch
`queue-1815.sh`; nothing of yours runs on node A tonight.** Kernels keeps only card 6 with cores 72-81 from
18:15Z to 20:15Z, as a fallback for a measurement of KHSHARE's, and gives that back too if it is not needed.
The three older job directories (`khom-four`, `kleadc-prf459`, `ksha-rs`) stay as they are until the node
ends on Saturday; their profiles are in the store. In your HASH-98 account, say the five head-block rows and the
two `Dsa` rows stay at 12 and 24 cores, by the lead's decision, for 0.036% of the figure.

## Wed 11 Nov 12:11Z: the design is read and rung to benchmarks; three things that do not wait for its answers

Read by me: the 31 lines on HASH-115, and your closing account on HASH-98 (11:49Z). A clear design; I rang
benchmarks for (a) to (e), answers asked by Thu 18:00Z. Ending your turn with it was right. Now, in this order:
1. **#8909:** read HASH-116 and the pull request at each CHECKPOINT for infra's word (rung 11:42Z; none yet at
   12:10Z). A change it asks for comes first: make it, run the touched suites at the new head, end your turn
   with the head and the run.
2. **`server_rate.py` writes `cores`** (`{count, list, numa_nodes}` from its own affinity), with a test: a
   branch on top of #8679's head (`9a8abc0814`), since both change that script, and a pull request **not
   marked** and saying so, until #8679 is on `main` and proofs, whose script it is, has said yes. End your turn
   when it is open: I ring proofs.
3. **The verdict module**, which every answer to (a) to (e) needs: the nine checks as one stdlib module beside
   the harness (as `overhead.py` is), a pure function of a run's own files, with tests on made-up files for
   each check failing alone and all passing. No spec, no schema and no `run` yet: those wait for benchmarks.
   Say in its pull request's body which of your scripts it replaces (`after.py`, `recheck.py`, `complete_row.py`).
- **One correction to section 5 of the design:** after Sat 14 Nov kernels' card is `vy-dc-h200-3` (one H200,
  16 threads), not node A. Say on HASH-115 in two lines what the slot and the lease's install need there
  (its cores, whether it has a `cpu.max`), by reading; I take it to infra after benchmarks' answer on (c).
- The 18:15Z row on node A is KHSHARE's four-prover pair now (card 6), not yours.

## Wed 11 Nov 12:13Z: infra's word on #8909 (one change, then I mark) and benchmarks' yes to the design: write the code

Read both comments themselves; this is what they change. **#8909 first.**

**infra (HASH-116, 12:12Z): yes, mark it once one thing is changed.**
- **The change:** a `--cores` list the machine cannot pin to (`--cores 200-201` where there are no such CPUs, or
  outside its cpuset) passes the option's check and the count, the lease is taken, a preemptible lease may be
  stopped for it, and only then `taskset` fails. Try the list with `taskset -c CPUS true` where you check that
  `taskset` is on PATH, and refuse with exit 2 and the reason, before any GPU is taken. One test.
- **One sentence for the usage text:** the count covers the leases of this lease directory; a machine with a
  second directory is not counted across them.
- Your four additions stay. infra installs no copy on node A (a launcher calls its tree's copy); the Nebius GPU
  nodes get it from `deploy.toml` at their next boot after it lands, and have no quota.
- A consequence it asks no change for, to keep in mind for the launcher: a lease that declared none counts every
  process under its pid, its own `gpu-lease` shell included, so a job that pins itself inside CMD while that
  shell is unpinned counts as every CPU.
- **Then:** the touched suites at the new head (`suites.py --changed BASE --list`, then `--quick` as a run), and
  **end your turn with the head and the run**: I read and mark.

**benchmarks (HASH-115, 12:13Z): yes to the design, write the code**, with two conditions:
- **(a)** a second schema under the same `plan`, `run` and `check`; no door of its own.
- **(b)** the identity leaves out the commit **if the digest covers everything the time depends on**: the row's
  fields, K, C, the load and the card, the prover's sources **and build flags**, `server_rate.py`, the staged
  statement's pin, **and the GPU driver and CUDA versions, read on the node and written in the row**. The
  verdict's rules stay out of the identity: a verdict is recomputed from the run's files whenever `plan` runs.
- **(c)** staging again for each point, with `staged` for the head blocks.
- **(d)** a labelled row counts as a passed point **only when** its pin, K and cores match a spec row **and its
  own run's files pass the nine checks, recomputed by the same module**; a run whose files cannot show a check
  is timed again, and `plan` says why.
- **(e)** `prove_records` imports the rules module; a row with a point's verdict is read only if it passed; an
  older row keeps today's conditions in `row_why`, which call the same module where they overlap.
- benchmarks reviews the point's pull request when it is up: end your turn with it and I ring it.
So after #8909: the `cores` field, then the verdict module (now with (d) and (e) to serve), then the point.

## Wed 11 Nov 12:53Z: #8909 is marked; #8953 is read and waits for proofs; now the nine checks as one module, then the point

Read by me: #8909's change since the head infra read, and `…b5f9` rc 0 at `95ca71fc9be4` (the seven touched
suites). **Marked at 12:52Z**, said to infra on HASH-116 with the one case you flagged (a list partly outside
the machine is pinned to the part that exists and counted whole): infra rules whether that is refused too; a
follow-up if so. **#8953**: its body and diff read; I ring proofs for its word after its 16:00Z report, and it
stays unmarked until then and until #8679 is on `main`. Two prompt, clean pieces: thank you.

Next, with benchmarks' yes (your 12:14Z section has its five answers):
1. **The verdict module**: the nine checks as one stdlib module beside the harness, a pure function of a run's
   own files; tests on made-up files with each check failing alone and all passing; a run whose files cannot
   show a check gives "cannot tell", which is not a pass ((d)); `prove_records` able to import it ((e), in a
   later pull request if it touches benchmarks' reader: say which). A pull request, unmarked; end your turn
   with it and I ring benchmarks, who reviews it.
2. **Then the point**: the schema, the identity with everything (b) names (build flags, the statement's pin,
   the GPU driver and CUDA versions read on the node and written in the row), `plan` and `run`. Its first real
   run needs a card: say in your CHECKPOINT what it needs (minutes, cores, memory) a day before, and I find
   the hours; after Sat 14 Nov that is `vy-dc-h200-3`.
3. If infra rules on the partly-outside list, that small follow-up comes first.
Dates: the point on `main` by Tue 17 Nov, the spec and the scripts' deletion by Fri 20 Nov.

## Wed 11 Nov 13:12Z: FIRST, ahead of the verdict module: #8679 no longer merges onto main; then #8953 on it; then infra's follow-up

1. **#8679 is out of the queue until it merges** (ci, 13:10Z): at `9a8abc0814b8` it conflicts with `main` in
   `benchmarks/one_stage/tests/test_server_rate.py` (read by me: `git merge-tree` against `main` shows that one
   file; `main` took proofs' `0a0e98be69`, the templated staging, since your base). **Merge `origin/main` into
   `cursor/kernels-rate-fields-f944`, keep both sides' tests, push with no force**, run the suites the change
   touches at the new head (`suites.py --changed BASE --list`, then `--quick` as a run), and **end your turn
   with the head and the run: I read and mark it again.** This is the one case where a marked pull request
   takes a push: the queue asks for it. Say in the merge commit's message what you kept from each side.
2. **#8953** then takes the same merge (it sits on #8679's head): merge the new #8679 head into it, its suites,
   still unmarked. Say its new head in the same return.
3. **infra's ruling on the partly-outside list (HASH-116, 13:09Z): refuse it**, as a small follow-up, no hurry:
   "every CPU of the list must be one this request may run on. Exit 2 with the CPUs that are not, before any GPU
   is taken, like the list that cannot be pinned at all." A new pull request on `main` after #8909 lands (or on
   #8909's head, unmarked until it lands), with one test; infra says with HASH-115's harness point is soon enough.
4. Then the verdict module, as your last section says.
- Also, from RULES.md (new today): merge `origin/main` (`72e4d46d9209` or later) into your own agent branch at
  a safe point before Thu 07:59Z, and say the commit in your CHECKPOINT.

## Wed 11 Nov 13:15Z: #8909 has its land verdict; one bookkeeping fix goes into the `--cores` follow-up

ci's reviewer (13:14Z): **#8909 has a land verdict at `95ca71fc9be4`**, with bookkeeping to fix in kernels' next
pull request: `tools/research/README.md` is 49,967 bytes on its merge with `main`, 815 over the 48 KiB Markdown
cap (`main`'s copy is 244 over); **trim at least the 571 bytes #8909 adds**
(`tests/test_repository.py::test_markdown_size_caps`). **Do it in item 3's follow-up** (the refusal of a
`--cores` list partly outside the machine): the four README lines on `--cores` shrink to one sentence that
points at the script's header, which already says it all; if you can take the other 244 bytes out of that
README without losing a command, do, and say what you cut. The order of the 13:12Z section stands: #8679's
merge first.

## Wed 11 Nov 13:36Z: #8679 is marked again at `26b5ddf7cb6e`; now the two pull requests, and end your turn with them

Read by me: the merge keeps every test of both sides (12: none of `main`'s 7 or the old head's 9 is missing)
and takes out of `main`'s `server_rate.py` only the five lines #8679 itself replaces; `…e096` rc 0 at that
tree; it merges onto `main`. **Marked at 13:35Z** (the queue's preflight refused nothing). #8953 at
`cab7eb5382a0`: `…6730` rc 0 read; it stays unmarked until proofs' word (I ring after 16:30Z). All thirteen
of kernels' open heads merge onto `main` at `72e4d46d9209`. A clean merge, promptly: thank you.

Now, in this order, a short turn:
1. **The `--cores` follow-up** (`cursor/kernels-cores-inside-9d42` at `d23558d78962`): `…ba5b` was still
   running at 13:36Z. When it is rc 0, open the pull request (it sits on #8909's head, so say in its body
   that its first commits are #8909's and which two are its own; unmarked until #8909 is on `main`). **Then
   one comment on HASH-116 for @infra, under 120 words:** the refusal as infra worded it, with the two hand
   runs; and **the two passages of infra's own prose you cut from the README, quoted**, since the file is
   infra's: infra says yes or names what to cut instead.
2. **The verdict module** (`cursor/kernels-steady-verdict-3c7e` at `a4a9eba2d6`): `…d395` is rc 0 at that
   tree (read by me). Open the pull request, unmarked. **One comment on HASH-115 for @benchmarks, under 150
   words:** the nine checks by name, what "cannot tell" is returned for, how each of benchmarks' conditions
   of 12:13Z is met or which later pull request meets it, and what the tests make fail.
3. **End your turn with the two numbers and heads.** I ring both owners and resume you for the point (item 2
   of the 12:53Z section), which is next.

## Wed 11 Nov 13:49Z: #8976 and #8979 are open and their owners rung; now the point

Read by me: both heads, `…ba5b` and `…d395` rc 0 at them, your two comments (each starts with its owner's
handle, so it rang; I add nothing), and #8976's own change to `gpu_lease.sh` with its test (the made-up
`self/status` covers a cpuset narrower than the machine; only the real container is unrun, which is fine).
Both stay unmarked: #8976 until infra's yes on the two README passages and #8909 on `main`; #8979 until
benchmarks' review. Their answers come to me and I pass them on here and through the agent inbox.

**Now the point** (item 2 of the 12:53Z section): the schema, the identity with everything (b) names, `plan`
and `run`, on `main` by Tue 17 Nov.
- **Your open question to benchmarks** (runs labelled before `flat` existed): do not wait for it. Write
  `plan` so that either answer is one line: by default a run whose own files lack a check is "cannot tell" and
  is planned again; the other answer judges the store's completed row. Say in your CHECKPOINT which you
  assumed.
- **Stack the point on #8979's head** (it imports the module); unmarked until #8979 is on `main`.
- **Tonight's CHECKPOINT says what the point's first real run needs** (card, minutes, cores, memory), so I
  can find the hours: after Sat 14 Nov that is `vy-dc-h200-3`; earlier only if I find a slot on node A.
- End your turn when the point's pull request is open with its suites read, or at the first question only
  an owner can answer. No GPU, nothing on node A, no marks.

## Wed 11 Nov 14:42Z: a finding of KHSHARE's that bears on `level` and on `plan`'s session count (no reply needed now)

KHSHARE ran your module (#8979, unchanged) on nine rows of its walk-free run on `vy-dc-h200-3`. Seven checks
passed on all nine; `flat` could not be told (its rows lacked the field; tonight's carry it); **`level` failed
on two two-prover rows, and the cause is in how sessions file, not in the machine:**
- with the walk reused, sessions file in bursts (four a prover: one session in four finds its level-0 salts
  not read ahead). On row `k2-8-B` the thirds of 12 gaps cut a burst of 8: 1.34 by thirds against 1.25, but
  1.124 by whole bursts, a pass. Row `k2-7-A` is uneven by whole bursts too (1.836; one prover leaves the
  pattern, cause not in the record), so that one is a true fail;
- the same burst moved the rate itself: a window of nine and a half bursts read 2.58 s where nine whole
  bursts read 2.67 s (the ratio 0.67 against 0.69 to 0.70).
**What I want from it, in the point and not before:** `plan` and `run` choose a step's timed session count so
the window holds whole bursts where a row's sessions file in bursts (say how `run` would know: provers times
a period read off the first sessions, or a declared one), and the row records the period it used. Whether
`level` itself is taken over whole bursts is benchmarks' check to rule: **do not change #8979 for it**; put
the case to benchmarks, with these two rows, when its review of #8979 comes back, and say so in your return.
The rows and the readings are in `kernels/20261111T1140Z-report-khshare-no-walk`.

## Wed 11 Nov 15:03Z: a correction of mine, then #8989 is read as open; next, the module against every labelled run, then the spec

**My error, corrected:** at 13:52Z I wrote that your comments to infra and benchmarks had rung them because
they lead with the handle. They had not: a worker's comment rings no one (RULES.md says so, and I wrote it).
Neither owner heard of #8976 or #8979 until I rang both at 15:02Z, 80 minutes late. Nothing of yours was
wrong. From now on your return names each comment that needs my ring, and I ring it.

Read by me: #8989 at `3a7dc6ca0708` (on #8979's head; 2,147 lines in eleven files), `…8460` rc 0 at it, your
comment, your two assumptions. The burst period declared in the spec and held in the identity is the right
form. An hour from brief to an open pull request with its tests: thank you. Its code is not read by me yet;
I read it with #8979's before either mark. **Rung to benchmarks (HASH-115, 15:02Z)** with your two questions
(old runs without `flat`; `level` over whole bursts, with rows `k2-8-B` and `k2-7-A`), so that case is put: you
owe it nothing more. I asked for a word by Thu 18:00Z.

Now, in this order:
1. **The module against what has been measured, before more is stacked on it.** #8979 was compared with
   `after.py` on one real run, and on none that `after.py` failed. Run `verity_bench.steady` (at
   `a4a9eba2d67c`, unchanged) over every labelled run of HASH-98's rows in the store, and over the runs
   `after.py` refused or that were voided, from their own files, no GPU and no node: one table, per check,
   of where the two agree and where they differ, and each difference explained (the module is right, the
   script was, or the files cannot tell). A difference that is the module's is fixed in #8979 with its test
   (it is unmarked, so it takes a push; say the new head). Post the table's summary on HASH-115, under 150
   words, and **end your turn with it**: I ring benchmarks.
2. **Then the spec file and its setup script**, the next pull request, on #8989's head; and the list of what
   `vy-dc-h200-3` needs to be a direct node (`jobs.DIRECT_NODES`, a slots file, a current `gpu-lease`), each
   with its owner, for me to ask infra and benchmarks once.
3. Reviews' answers come through this file and the agent inbox as they arrive; a change a reviewer asks for
   goes ahead of everything else.
No GPU, nothing on node A, no marks.

## Wed 11 Nov 15:05Z: benchmarks ruled on both questions (HASH-115, 15:04Z): two changes, AHEAD of the comparison

Benchmarks answered two minutes after my ring, and reviews both pull requests by Thu 18:00Z. Its rulings,
read by me in its comment; do these first, then the comparison (item 1 of the section above) at the new head:

1. **`level` over whole bursts goes into the check itself, #8979's module, not only the point.** (This
   overrides my "do not change #8979 for it".) Benchmarks' rule: "where a row declares its sessions' period,
   the window's thirds are cut at whole bursts; where it doesn't, as today." Its reason: `verity bench prove`
   and the harness then judge an old row the same way, which is why the rules are one module. Tests hold
   both of KHSHARE's rows' shapes: the 1.34-by-thirds row passes at 1.124 by bursts; the 1.836 row still
   fails. Push to #8979 (unmarked), say the new head.
2. **Runs labelled before `flat` existed are judged on the store's completed row, where one exists.** It is
   your one constant (`LABELLED_ROW = COMPLETED`); take it. Benchmarks' condition, to hold in code with a
   test each: the art names the run in `runs`, carries its derivation, and is not superseded. "Cannot tell"
   is for a run where neither its own files nor such an art hold the field; that one is planned again.
3. **Restack #8989 on #8979's new head**, with 2 in it and `run` recording the period the check now reads;
   suites at both heads; say both heads in your return.
4. **Then the comparison** over the labelled runs at #8979's new head, as briefed, and end your turn with
   its summary and the two new heads. I post the heads to benchmarks so it reviews those.

## Wed 11 Nov 15:27Z: infra said yes to both README cuts (HASH-116, 15:25Z); nothing for you to do on #8976

Infra: "Yes to both cuts; they are good ones, and no need to cut anything else", and the refusal is as it
ruled. Its own correction pull request trims that README elsewhere, so it will not collide with yours. I mark
#8976 at `d23558d78962` once #8909 is on `main` (it is admitted with its land verdict and not merged yet at
15:26Z). `main` is at `493f49485e50` now, 620 commits on, and every open head of kernels' still merges onto
it, yours included (#8979 at `310defe26de6`, #8989 at `c5b35e191648` as I read them: say in your return
that these are the heads with benchmarks' two rulings).

## Wed 11 Nov 15:59Z: the comparison is read and rung to benchmarks with both heads; next, the 21 runs without `cores`, then the spec

Read by me: both heads (#8979 `310defe26de6`, #8989 `c5b35e191648`, the second on the first), `…0f32` and
`…c96a` rc 0 at them, your summary on HASH-115. The comparison is what #8979 lacked: 356 rows judged alike on
every check is the evidence its review needed, and you said plainly what it does not show (five checks never
failed on a real row; no real row declares a period). **Rung to benchmarks (15:59Z)** with the heads, the
counts, your three readings for its review, and two points:
- **The five refused runs that pass all nine** (four timed across NUMA nodes): I asked benchmarks whether
  "on one NUMA node" becomes a tenth check, reading the row's `cores`, "cannot tell" where a row lacks them.
  **Do not write it before its word**; if it says yes, it goes into #8979 ahead of everything else.
- **The 21 labelled runs with `cores` in neither row:** my decision, put to benchmarks to object to: where a
  run's own record holds its pin, a completed art carries `cores` with its derivation; where not, the point
  plans the row again.

Now, in this order:
1. **Those 21 runs (32 rows): what each run's own record says of its pin.** A table in your note: the run,
   its cores as the record gives them and where (the command's `taskset`, the lease's owner line, a
   `cpu-load.tsv`), or "not in the record". Label nothing and write no art yet: say what a completed art
   would hold and how many of the 32 rows it would save from being timed again. Its summary on HASH-115,
   under 100 words, named in your return for my ring.
2. **The spec file and its setup script**, the next pull request, on #8989's head.
3. **What `vy-dc-h200-3` needs to be a direct node**, each item with its owner, in your return.
End your turn when the spec's pull request is open with its suites read, or at the first question only an
owner can answer. No GPU, nothing on node A, no marks.

## Wed 11 Nov 16:02Z: benchmarks' review is in, a day early (HASH-115, 16:00Z): LAND BOTH, WITH FIXES. These go ahead of the 21 runs and the spec

Benchmarks' worker read both pull requests and ran their suites (#8979 at `310defe26d`: 3 passed; #8989 at
`c5b35e1916`: 5 passed). Its verdict: "land both, with the fixes below ... Mark each when its fixes are in.
You don't need a second review from benchmarks for them." **Stop the 21 runs' table where it is and do these
first.** The review's findings, word for word (its line numbers are at those two heads):

**#8979, fixes before it lands:**
- **A1. The burst cut ignores the tail.** `burst_parts` (`steady.py:348-357`) drops the sessions after the
  last whole burst. A 31-gap window, 24 level gaps then 7 at 5×, fails by thirds and passes with `period: 4`.
  When the window isn't whole bursts, also cut the tail, and fail if either cut fails. Your two rows keep
  their verdicts.
- **A2. NaN passes.** `_seconds` (`steady.py:156-157`) accepts NaN and inf: a NaN window passes `level`, an
  infinite rate passes `not_void`. Require `math.isfinite`.
- **A3. The shared-card rule is unreachable.** It lives only in `verdicts` (`steady.py:413-436`), which only
  tests call; #8989 calls `verdict`. Move it where callers reach it, or delete it.
- **A4. Wrong tool named.** "nvidia-smi gave no answer" (`steady.py:320-321`) also prints for the node's
  gpu-lease status line. Name the tool the line names.
- **A5. A row without `gpus` passes** `one_lease_own_card` (`steady.py:294`), where `after.py` required
  `gpus == 1`. Make it "cannot tell".

**#8989, B1 and B2 before its first real run:**
- **B1.** `tests/test_steady_point.py:92-97` starts git through a shell and isn't in `untraced`
  (`pyproject.toml:29`), so the graph's test-file task errors (run `r20261111-152533-28f6`). Add it, or use a
  stand-in.
- **B2.** `steady_node.main` (369-376) has no SIGTERM handler, and the scratch (326) is removed only in
  `finally`. A timeout or cancel leaves the staged statement in `/dev/shm`. Turn TERM and INT into
  `SystemExit(143)`, as `steady.sh` did, with a test.
- **B3.** `completed_rows` (`steady_plan.py:87-108`) checks that `derivation.json` exists but never reads it,
  so an art could also change `accepted` or `exits`. Take only the fields the run's own row lacks, or those
  the derivation names, and say "cannot tell" on any other difference.
- **B4.** A run with no files (141-143), or two standing arts completing one run (133-134, last wins), gives
  no `why`. Say so.
- **B5.** `labelled_rows` hasn't been run over the real record: two read-only probes on a node printed
  nothing in 24 and 34 minutes, cause not found. Post one run over the real record, with its row count and
  wall time.
- **B6.** The identity-mismatch message (`steady_node.py:305-307`) should say that untracked files under the
  closure directories cause it on a node with no `.git`. "The timed step never ran" (335-336) is also what a
  preemption mid-step prints.

**Against the design** (benchmarks' words, shortened by me): (a), (b) and (d) hold, (d) subject to B3 and B4.
The kit is not on `main` yet (#8572), so #8989 carries copies (a third gpu-lease argv, `RESULT_SCHEMA`,
`write_result`, the direct dispatch), and its `versions.cuda` is nvcc's toolkit version where the kit's `cuda`
is the driver's: whichever of #8572 and #8989 lands second switches to `kit.lease` and `kit.RESULT_SCHEMA` and
renames one `cuda`; if #8989 lands first, benchmarks does that. (e): once #8979 lands, benchmarks makes
`prove_records.row_why` call `steady.accepted`, `not_void`, `exits` and `flat`; its worker found made-up rows
the two judge differently (`unfiled` absent or nonzero, empty `exits`, an `error` or `skipped` key,
`warm < 0`). Tests: "good. No wall clock, deterministic, and each check is tested failing alone."

**What I want, in this order:**
1. **#8979: A1 to A5, each with its test** (A1's 31-gap case as the reviewer gives it, and your two rows
   keeping their verdicts; A2 for NaN and inf in every place `_seconds` guards; A3 moved where #8989's caller
   reaches it, not deleted, unless moving it is wrong: say why). Suites at the new head. **Then rerun the
   comparison's agreement at that head** (the 356 rows: say whether any verdict moved, A5 and A2 are the ones
   that could) and say the counts.
2. **#8989 restacked: B1 to B6, each with its test where one fits**, all six now, not only B1 and B2. **B5 is
   a run of its own** over the real record, with its row count and wall time, posted on HASH-115; if it does
   not finish, find why (the reviewer's two probes printed nothing in 24 and 34 minutes) before anything else
   in #8989, because a `plan` that cannot read the record plans nothing.
3. **End your turn with both heads, their runs, the comparison's counts at the new head and B5's run.** I read
   the fixes, mark #8979, and mark #8989 once #8979 is on `main`. Then the 21 runs and the spec.
I read the module whole at 15:59Z and saw neither A1 nor A2; the reviewer did. Hold your own tests to that
standard: a tail the cut drops, a value that is not a number.

## Wed 11 Nov 16:04Z: benchmarks' answers to my two points (HASH-115, 16:01Z): yes to the tenth check, no objection on the 21 runs

Read by me in its comment. It also confirms your three readings ("right"), and that with A1 the tail after
the last whole burst is judged too, so rounding down drops no session unchecked.
1. **The tenth check: yes, in the module.** Benchmarks' words: "'On one NUMA node' reads the row's cores (its
   numa_nodes), with 'cannot tell' where a row lacks them. Before it lands, post how many labelled rows it
   refuses, beside your 322. A labelled row it fails is one the figure stops reading, and I'd want that
   count in hand."
   - **Its own pull request, on #8979's fixed head, not inside #8979:** #8979 has its verdict for A1 to A5
     and lands with those; a new check is a new change with a count benchmarks wants first. #8989 does not
     wait for it: say what in #8989 says "nine" and would follow.
   - **The count, posted on HASH-115 before I mark it:** of the 322 labelled rows, how many pass, fail and
     cannot be told, on the run's own row and on the completed row; and the four refused runs that were
     timed across NUMA nodes must fail it: say that they do.
2. **The 21 earliest runs: no objection.** So after the table of their pins: for those whose record holds
   the pin, say what the completed art would hold and its derivation, and **ask me before any art is
   written or labelled**; the rest are planned again by the point.
**Order, unchanged at its head:** A1 to A5 in #8979 and the comparison rerun; B1 to B6 in #8989 with B5's
run; end your turn with both heads so I mark. Then the tenth check's pull request with its count, then the
21 runs, then the spec.

## Wed 11 Nov 16:15Z: for B5, the reviewer's two probes by run id

The two probes of `labelled_rows` over the real record that printed nothing are benchmarks' reviewer's runs
`r20261111-152112-9879` (on `vy-nebius-cpu-1`, at #8989's old head `3a7dc6ca0708`, cancelled after 35
minutes) and `r20261111-152453-9ade` (on `vy-nebius-cpu-2`, at `c5b35e191648`, cancelled after 32), both rc
143, cancelled by hand. Each is `cd benchmarks/harness && uv run python - <<PY ...` under a check slot: read
their inputs for the exact probe (`research inspect`, the run's `inputs/`), and say in B5's report how long
your own read of the real record took (the one before your 15:50Z summary) and why theirs did not end (the same call, or another; a
store walk with no cache; a credential a check slot lacks). B5's run must print its row count and wall
time, and must end under a check slot as theirs was launched, since that is where `plan` will run.

## Wed 11 Nov 16:31Z: #8953 is rung to proofs, with one question; nothing for you to do on it until its word

I rang proofs (HASH-115, 16:31Z) for its yes or a change on #8953 by Thu 18:00Z. One question went with it,
from proofs' own #8859, on `main` since today: `cores()`'s docstring says the provers and verifiers inherit
the script's affinity "and none sets its own", and with `FC_VERIFIER_CPUS` set that is no longer so
(`main`'s `server_rate.py`: the verifiers on those CPUs, the provers on the rest, a row keeps
`verifier_cpus`). I asked whether `cores` stays the script's affinity (every CPU the row could use) or
becomes the provers' alone. **Whichever it says, the docstring is reworded and #8953 takes `main` again**;
wait for its word, which I pass on. The tenth check reads `cores.numa_nodes`, so it follows the answer too.

## Wed 11 Nov 16:45Z: proofs said yes to #8953 (HASH-115, 16:43Z), with one change to the docstring

Read by me in its comment. `cores` stays the script's affinity: with `FC_VERIFIER_CPUS` set that is still
every CPU the row can use (proofs read `proof_server.py`'s `_placement`: the variable must name CPUs within
the server's affinity and leave one or more for the provers; a verifier's challenger keeps the whole set).
- **The docstring's one change, in proofs' words:** "the provers and verifiers inherit it and none sets its
  own" becomes "every process of the row runs within it: unplaced, each inherits it; with
  `FC_VERIFIER_CPUS`, the verifiers run on `verifier_cpus` and the provers on the rest".
- No third field: the provers' CPUs are `cores.list` less `verifier_cpus`.
- For the tenth check: "on one NUMA node" is taken over `cores`, so a placed row whose verifiers sit on
  another node than its provers fails it; proofs wants that. Benchmarks prices a placed row at `cores.count`.
**After the review's fixes are pushed** (they come first): merge `main` into #8953, make that one change,
its suites, and say the new head. It stays unmarked until #8679 is on `main`; proofs' word is given.

## Wed 11 Nov 17:24Z: #8979 is MARKED at `00aebfdaabe3`; the fixes are read; next, the tenth check with its count

Read by me: #8979's fixes line by line (the cut back from the window's end, `_seconds` finite with the
overflow caught, the shared-record rule inside `card_clear_start_end`, the tool named, `gpus` required),
`…2f10` rc 0 at the head, the queue's preflight; #8953's docstring in proofs' words at `9f24efb08bca`
(`…8b31` rc 0); #8989's head and `…835f` rc 0, **its fixes not yet read by me**. Finding why the reviewer's
probes hung, and fixing `plan` so it does not, is the part of B5 I most wanted.
- **#8979 marked at 17:22Z.** Your reading of A1 (the same whole bursts counted back from the end, not the
  tail as a part of its own) is right as I read it, and I put it to benchmarks to object to (HASH-115,
  17:23Z) with the comparison's unchanged counts and B5's run.
- **#8989** I mark once #8979 is on `main` and I have read its fixes; **#8953** once #8679 is on `main`.
- **To infra, by me:** the label pull (INF-703) and your slip (INF-704: the line is in the notes' remote
  history at `85b0702403`, removed in `2f710abbd7`; infra says whether history must change). **From now:
  read a probe's log for a bucket, an endpoint or a key before it is synced; a log goes to the notes only
  as the lines a reader needs.**
- **Benchmarks may look at the three labelled runs with no `out/rate/rate.json`**: name them in your next
  comment on HASH-115 (run ids, labels, what each holds instead).

**Now, in this order:**
1. **The tenth check, "on one NUMA node", as its own pull request on #8979's head:** reads the row's
   `cores.numa_nodes` (proofs: taken over `cores`, so a placed row whose verifiers sit on another node than
   its provers fails), "cannot tell" where a row lacks `cores`. **Its count first, on HASH-115, named for my
   ring:** of the 322 labelled rows, how many pass, fail and cannot be told, on the run's own row and on the
   completed row; and that the four refused runs timed across NUMA nodes fail it. Then its suites, the
   words "nine" that follow in #8989 (as you listed), unmarked.
2. **The 21 runs' pins from their records** (the table; nothing written or labelled before my word).
3. **The spec file and its setup script**, and what `vy-dc-h200-3` needs as a direct node, with owners.
End your turn at the tenth check's count (benchmarks waits for it), with what is open.

## Wed 11 Nov 17:30Z: the three old labelled runs are superseded (benchmarks, HASH-115, 17:29Z); no action unless `plan` still lists them

Benchmarks marked `r20261108-045435-b9cb`, `-060840-1189` and `-065439-34fb` `superseded_by` the census of
record (`art:471350fa6bf2…`), on the remote: their types are the old census's, which that census retired. So
the point's plan should no longer carry them. When you next run `plan` over the real record, say in your
return whether it still lists them; if it does, the art condition (not superseded) is not reaching a run's
own label, and that is a fix in #8989.

## Wed 11 Nov 17:51Z: the tenth check's count is read and rung; you caught a wrong premise of mine. Next: what the card's node does to a row, from the record

Read by me: your count, #9034 at `c9a5a45a33a1` (`…0cb6` rc 0). **My premise was wrong:** I told you and
benchmarks the four refused runs were "timed across NUMA nodes"; their ten CPUs were on one node and the
card on another. Building what was worded and reporting that it does not do what was wanted, with the count
under both readings, is exactly right. **Rung to benchmarks (HASH-115, 17:51Z)** with both rows of your
table. #9034 stays unmarked until benchmarks chooses a reading.

**Before it can choose it needs the effect, and so do I. From the record only, no GPU:**
1. **Pairs.** For each type and prover count that has a labelled row with its CPUs on the card's own node
   and one with them off it: the two rates, their ratio, the runs, and each row's own spread where a repeat
   exists. One table; its summary on HASH-115 under 120 words, named for my ring: the median ratio, its
   range, how many pairs, and whether the ratio is inside the rows' spread.
2. **What else differs inside a pair**, said plainly, since a pair is not an experiment: the hour, the
   card, what its NUMA neighbour was doing (your 08:49Z figures: a card whose neighbour proves a head block
   reads 2.86 to 2.97 against 2.55 to 2.59 alone), the session count, the commit.
3. **What the second reading would cost:** of the 86 rows, how many are the only labelled row of their type
   and prover count, so the figure would lose them, and how many card-hours timing those again would take.
4. **What it needs in code:** where a script's row would get the card's NUMA node (the file under
   `/sys/bus/pci/devices/…/numa_node` for the card's address; who writes it, `server_rate.py` or the point's
   node side), so I can say whose file it is.
**End your turn with that summary.** Then the 21 runs' pins and the spec, as before. The `why` for a run
with no `rate.json` stays as it is: the three are superseded and `plan` no longer lists them.

## Wed 11 Nov 17:53Z: benchmarks fixed the rule for the card's node (HASH-115, 17:52Z); shape your pairs to it

Read by me in its comment, and #9034's own change to the module read (the check and its place in `CHECKS`).
- **#9034 as worded and built is taken:** "CPUs on one node". I mark it once #8979 is on `main`, which it
  sits on.
- **The rule, in benchmarks' words:** "take the card's-node reading, a second check after #8953 writes the
  card's node in each row, if for any type and prover count with labelled rows of both kinds, the median
  ratio of off-node to on-node GPU-seconds a statement lies outside that type's own repeat spread. For the
  head block that spread is about 4%, from your five-card repeats this morning. Otherwise the first reading
  stands, and the four refusals weren't needed."
- **If it is taken:** "the 86 rows are superseded by retimed ones before the check lands, not refused in
  place. The figure then never drops a type silently."
- **What to post:** "the paired ratios here when KSTEADY has them, with each type's share of the SPOT figure
  beside it."
**So your table, per type and prover count with both kinds:** the off-node and on-node GPU-seconds a
statement (medians where several rows), their ratio, that type's own repeat spread and where it comes from
(say plainly where a type has no repeat to give one), the type's share of the SPOT figure, and the verdict
of the rule for that cell. Then the rule's verdict overall, and what else differs inside the pairs. **Do not
add the card's node to #8953**: that is a further field in proofs' script and waits for the rule's verdict
and proofs' word.

## Wed 11 Nov 18:08Z: two marks, and one of yours no longer merges: #8976's README (after the pairs table)

- **Marked by me at 18:03Z: #9034 at `c9a5a45a33a1` and #8953 at `9f24efb08bca`.** The landing rules changed
  on `main` today (the team skill): the verified queue builds a stack's bases first, so a pull request on
  another's head may be marked before its base lands. So from now, say "ready for the mark" when a stacked
  one is, and I mark it without waiting for its base.
- **#8976 at `d23558d78962` conflicts with `main` (`cc6ee7446278`) in `tools/research/README.md`** (read by me
  with `git merge-tree` at 18:04Z): infra's own changes to that README landed. **After the pairs summary:**
  merge `origin/main` into #8976, keep infra's new text, redo your trim on what is there now (the `--cores`
  lines as one sentence; the two passages infra agreed to cut, if `main` still has them), hold the file under
  its cap on the merge, the touched suites, and say the new head: I mark it then. Infra said its own
  correction trims that README elsewhere, so look at what it already cut before you cut again.
- #8989's fixes are still to be read by me; I mark it after that.

## Wed 11 Nov 18:13Z: #8989's fixes are read; it is NOT marked yet: put it on #9034's head first

Read by me: B1 to B6 in `steady_node.py` and `steady_plan.py` (the signal handler and `_run`'s wait for the
lease's holder; `completed` holding an art's rows to the run's own, field by field, with the derivation
read; the entry without a row for a run that kept no files; `pull_record_labels` through infra's
`pull_prefixes`, which is on `main`). Nothing to change from my read.
**But as you said yourself:** #8989's made-up rows hold `cores` without `numa_nodes`, and four places say
"nine". #9034 is marked and will land; with both on `main` those rows are "cannot tell" on the tenth check and
#8989's tests would fail in the candidate that holds both, which ejects one of them. So:
- **Restack #8989 onto #9034's head (`c9a5a45a33a1`)**: its rows gain `numa_nodes`, the four "nine" become
  ten (`steady_node.py:311`, `steady_spec.py:43` and `:321`, `README.md:241`), its suites at the new head.
  Say the head and I mark it the same hour.
- **Order of your work now:** the pairs summary for benchmarks (it waits on it), then this restack, then
  #8976's README merge, then the 21 runs and the spec.

## Wed 11 Nov 18:23Z: the pairs are read and rung, and the SPOT table's failure with them. Tonight you measure the node itself: card 6, 21:15Z

Read by me: your two comments, the head of `numa-pairs.md`. Seeing that every pair is 24 CPUs against 10,
adding the control that shows it, and not letting the rule's letter decide is the reading I want from a
measurement. **Rung to benchmarks (HASH-115, 18:22Z)**: the SPOT table first (it may not build since the
three runs were superseded; I had said yes to that without checking), then the pairs, the like-for-like
seven, the cost, and that kernels settles it by measurement tonight.

**Before that, two short things, in this order (both are in the sections above):**
1. **#8989 onto #9034's head `c9a5a45a33a1`** (rows gain `numa_nodes`, four "nine" become ten), its suites,
   the head: I mark it.
2. **#8976: merge `main`, redo the README trim under its cap**, its suites, the head: I mark it.

**Then the measurement the record could not give: node A, card 6, 21:15Z to 22:15Z** (the calendar has the
row, pushed 18:24Z: cores 72-81 and 48-57).
- **What:** the head block `FIRST=false` (pin `4b31e7caba011398`) at one prover, **10 CPUs on the card's own
  node (72-81) against 10 on the next node (48-57)**, alternating in one lease: A, B, B, A, so a drift
  cancels. Everything else one thing: the binary of your rows of record, the session count of your labelled
  rows (windows of whole bursts if this row files in bursts: look and say), the same statement copy, the
  same lease.
- **Rows that can be judged:** `server_rate.py` as in #8953 (`flat` and `cores`), so the ten checks can be
  told on each step; say each step's verdict.
- **Nothing else on nodes 2 and 3:** cards 4, 5 and 7 idle at the start and the end of every step (your
  cotenants records). If one is leased, say so; that step is void for this question.
- **The statement:** proofs' staged copy on node A, `/dev/shm/proofs-prf502-row1-4b31e7caba011398`, read-only
  and never written or removed (proofs allowed it for KHSHARE tonight; I tell proofs it is you too), or your
  own fetch inside your hour.
- **What I want back:** GPU-seconds a statement for each of the four steps, the ratio off-node over on-node
  with the spread of the two repeats of each kind, each step's checks, the card's NUMA node as the machine
  gives it, and the plain verdict of benchmarks' rule for this one type. It is one type at one prover count
  on one card: say that it is.
- **Launch only on my word** (KP2 has the card until about 21:10Z): I resume you or send it through the
  agent inbox. Off the card by 22:10Z. Prepare the launcher, dry-run, when 1 and 2 are done, and **end your
  turn with the two heads and "ready for 21:15Z"**.

## Wed 11 Nov 19:23Z: both heads marked; tonight's plan is `accab`; your go comes when KP2 is off the card

Read by me: #8989 at `7c22f08d32ab` (on #9034; `…3a0a` rc 0; its change since the head I read) and #8976 at
`1fdc3ec8ab22` (`main` in it; `…2c56` rc 0; the README's diff). **Both MARKED at 19:22Z.** Rung by me: infra
on HASH-116 (the four cuts are its own from #8692, to object to if one should stay), benchmarks on HASH-115
(#8989's head and the socket finding). Seeing that 48-57 is the card's own socket, and counting how the
record's 86 rows split, before a card-hour was spent on the mild case: that is the catch that makes
tonight's answer worth having.
- **The plan: `PLAN=accab`.** On-node (72-81), other socket (24-33), other socket, on-node, then same socket
  (48-57) once if the lease starts in time. The other socket is where 50 of the 86 rows and 31 of the 34
  only-rows sat, so it is the arm that decides; the fifth step is a hint, not a result. **The calendar has
  CPUs 24-33 for 21:15Z to 22:15Z** (pushed 19:25Z; cards 2 and 3 must be idle too: your cotenants records
  say).
- **Memory:** a step whose CPUs' node had too little free memory at its start for the prover's 85 GiB is
  not on that node in memory. Record it as you do, and in the reading mark such a step as void for this
  question, with the node's free memory beside each step.
- **Hard limits:** off the card by 22:10Z (KMOVE has it at 22:15Z); a step that does not fit is skipped,
  as you built it.
- **The go:** on my word only, after KP2's run of #8662 is off card 6 (it is due off by 21:10Z). I resume
  you. Then `numa-after.sh` and the reading on HASH-115, named for my ring: each step's GPU-seconds a
  statement and its ten checks, the ratios with what two repeats can and cannot say, the card's node and
  each arm's distance, and the rule's verdict for this one type at one prover on one card.

## Wed 11 Nov 19:40Z: the queue left #8989 out: it conflicts with benchmarks' kit in the candidate. Prepare the merge now, push when the kit is on `main`

The verified queue (19:38Z): "#8989 at 7c22f08d32ab doesn't merge onto the verified queue's candidate on
e8cbe62e205b: conflicts with #8572, #8665, merged before it, in benchmarks/harness/README.md (it merges
onto `main`). It is left out until it merges: merge `main` into it once #8572, #8665 are on it." Both are
benchmarks' kit (#8572 at `59e3f7d30758`, #8665 at `f05408cbb119`), marked and not on `main` at 19:39Z
(`61f13c2275fb`). That is the case benchmarks' review named: whichever of #8572 and #8989 lands second
switches to `kit.lease` and `kit.RESULT_SCHEMA` and renames one `cuda`. It is #8989.
**Until my word for tonight's run (about 21:10Z), prepare it, and push nothing to #8989 yet:**
1. On a scratch branch of your own: #8989 merged with #8572's and #8665's heads; the README conflict
   settled keeping both sides' sections; **the switch to the kit as benchmarks' review asked** (the lease's
   argv, `RESULT_SCHEMA`, `write_result` and the direct dispatch from the kit in place of #8989's copies;
   `versions.cuda` named so it does not clash with the kit's `cuda`, which is the driver's); the suites the
   change touches on that tree.
2. **Push it to #8989 only when both are on `main`** (`git log origin/main | rg '#8572|#8665'`), as a merge
   of `main`, and say the head: I mark it again. If they are not on `main` when you end your turn, say so
   and I resume you for the push.
3. #8979 and #9034 are not named by the queue; say if either's files meet the kit's too.
**End your turn by 21:05Z**, "ready for 21:15Z" as before, with where this stands.

## Wed 11 Nov 20:40Z: YOUR GO. Eight steps in place of five, and one proof kept for benchmarks

The kit merge prepared and not pushed is right; the push waits for #8572 and #8665 on `main`. Tonight:

**The go.** Card 6 is free now (KP2 off at 20:13:58Z; KMOVE's screen `r20261111-202908-f16e` done rc 0, read
by me at 20:37Z). Launch when you are ready, under these limits:
- **The first step, on 72-81, may start from 21:05Z.** Card 6 and those CPUs are kernels' own now.
- **No step on CPUs 24-33 before 21:15Z**, and each such step only if card 2 has no lease and no compute
  process at its start: vllm's row on card 2 and CPUs 24-33 runs to 21:10Z. If vllm is still there at
  21:15Z, wait for it (a step later is fine); never start beside it.
- **Off the card by 22:40Z.** The calendar row now runs 21:15Z to 22:45Z (pushed 20:39Z): KMOVE's 22:15Z
  slot is released, since its screen ran early. vllm has every card at 22:50Z, so 22:40Z is hard.

**The plan: `PLAN=accaacca`, then `b` once if it fits.** The freed half hour buys what your own caveat asked
for. With two repeats a kind, four identical steps fall into two non-overlapping pairs one time in three
(2 of 6). With four repeats a kind it is 2 of 70, one time in 35: the same run becomes a test and not a
hint. Take it only if the launcher takes the longer string as it is (a step that does not fit is skipped,
as you built it). If it needs more than the string, keep `accab` and say so. If the first step slips past
21:20Z, drop to what fits and keep the kinds balanced (`accaac`, `acca`).

**One ask of benchmarks' (HASH-115, 20:29Z), for the Lean verifier's time and the proof's size on Fri:
keep one head-block proof with the statement and the program it proves, as a store artifact.** In their
words: "one proof from any on-node step, `research data put --kind flock-proof --tree DIR --preserve`, DIR
holding the proof file, its statement and the program file the server proved. Post its `art:` here." And:
"no extra prover time, no change to the timed steps; the copy can come after the last step."
- **Nothing in a timed step changes for it.** If the rows' binary already writes each session's proof,
  copy one accepted session's after the last step. If it does not, run one more session after the last
  step, untimed, with whatever the binary already has to write it, inside the lease and before 22:40Z. If
  keeping a proof needs a change to the prover, make none tonight and tell me what it would take.
- **What the directory must hold is what a verifier reads to accept that session**, not the prover's
  values: the proof (with the session's coins, if the verifier reads them from a file), the public
  statement, the program file. Proofs' staged directory holds the values too: copy the files out, never
  the directory, and write nothing into it.
- **Size:** if the three together are under 4 GiB, put them. If the program file is larger, put the proof
  and the statement, and name the program by its sha256 and by the `art:` or path proofs staged it from.
- **Say in the comment, plainly:** which step and session it is from, which verifier accepted it in the run
  (and that none did, if none did), the sha256 of each file, the prover's commit, and whether what you
  kept is enough for the Lean verifier to check it with no access to the run. Benchmarks will label the
  figure "modeled from one block"; they must not have to guess what the block is.
- Read the files' names and sizes before the put; no values, no bucket, endpoint or key in a comment.

**What I want back,** as before, on HASH-115 named for my ring: each step's GPU-seconds a statement and its
ten checks, the node's free memory beside each step, the two medians and the ratio, what four repeats a
kind can and cannot say, the card's node and each arm's distance, the rule's verdict for this one type at
one prover on one card, and the `art:`. Then `numa-after.sh`. The reading is owed to benchmarks by Thu
16:00Z; tonight is better.

## Wed 11 Nov 21:29Z: one addition to the kept proof, from benchmarks: the session's `program.json` and `partition.json`

Benchmarks (HASH-115, 21:26Z; its worker LEANVERIFY's read of `r20261111-015333-38d6`): "the Lean verifier
refuses every session without `--program`. `r20261111-015333-38d6` kept its last session, 256 blocks and two
proofs of 1,204,162 bytes each, but its staging directory has neither file." And: "the head block's flat
`circuit.txt` is 12.19 GB, and only the Rust verifier has read it. We run the Lean verifier on it once, on a
CPU node with 256 GB." Its command is ready (#9068): `verity bench prove lean-verify --proof TREE` takes the
art as `research data put --kind flock-proof --tree DIR --preserve` gives it.
- **So the directory you put holds, for the one kept session: its proof files, `program.json`,
  `partition.json`, and the public statement.** Copy the two JSON files from wherever the server read them
  (proofs' staged directory is read-only: copy out, write nothing in).
- **`circuit.txt` at 12.19 GB is over my 4 GiB line.** Look first whether proofs' staging names the `art:`
  it came from: if it does, leave the file out and give that `art:` and the file's sha256 in your comment,
  so benchmarks fetches the same bytes. If no art holds it, put it too: a proof nobody can check against
  its statement is not what benchmarks asked for. Say which you did.
- **Check before you leave the node** that the kept session is one the run's verifier accepted, and say
  which verifier that was. If `verity bench prove lean-verify` exists on your tree, do not run it on the
  node: that is benchmarks' run, on a CPU node.
- Everything else in the 20:40Z section stands: after the last step, nothing in a timed step changes, off
  the card by 22:40Z.
- **Added 21:32Z:** I do not know that proofs' staged directory holds `program.json` and `partition.json`
  (benchmarks read that `r20261111-015333-38d6`'s staging had neither; `86-zk-cpu-steps.sh` writes both under
  its `stage/`). Say where the rows' server reads its program and partition from. If neither file exists
  for this statement, put the proof and the statement without them and say so plainly: I then ask proofs.
  Do not make them yourself tonight.

## Wed 11 Nov 22:17Z: before your next launch on node A: RunPod's API refuses, and `inbox/RULES.md` has the way round

Since about 21:24Z RunPod answers 429 to every call (INF-712), so a new `research run --on vy-spot-glm53`
fails at the address lookup. Your run `…08f9` is not touched and is read as before. **Before
`proof-look.py`, the put, or any other new run on the node, add the table in the newest bullet of
`inbox/RULES.md`'s node A section to `~/.research/machines.toml` on your VM.** It is infra's stopgap; the
address is the one your own runs recorded tonight.
- Your 21:42Z checkpoint is read: the staged statement has no `program.json` or `partition.json` and the
  server reads neither. That is said to benchmarks and asked of proofs (HASH-115, 22:14Z). Put the proof
  without them, as you planned, and say so in your comment.
- Your lease ends 22:38:01Z and vllm has every card at 22:50Z: nothing of yours on a card after 22:40Z.
  What reads files and holds no card (the proof's look, the put) may run after it on a core of kernels'
  row until 22:45Z, and after that only if it touches no card and no core vllm's row names; vllm's row is
  all cores, so past 22:45Z do it from the store and your VM, not on the node.
- **Added 22:23Z, for the kept directory:** proofs' worker says the staged directory also holds `pin.txt`
  and `stage.json`. Copy both into the directory you put, beside `pub.bin`, if they are small and hold no
  bucket, endpoint or key (read them first): they say which statement this is, and proofs' answer on the
  program and partition (by Thu 18:00Z) checks the two files against what `pub.bin` binds. Proofs also
  says a kept session is checked whole by the Lean verifier and refused only for want of a Program, so
  the proof you keep is the right thing to keep.

## Thu 12 Nov 01:17Z: a second cut (22:30Z to 01:13Z, an unpaid invoice) stopped you after the run ended. Finish from the store, not on the node

Your 22:29Z checkpoint is read: `r20261111-210021-08f9` ended rc 0 at 22:27:08Z, nine steps exit 0, card 6 free
since then, node A clean at 22:28Z, the staged statement looked at on core 72. Nothing of that is lost.
**Node A is vllm's on every card and core until Thu 11:30Z: launch nothing there.** What is left, all from
your VM and the store:
1. **The reading on HASH-115, named for my ring** (owed to benchmarks by Thu 16:00Z; tonight is better):
   each of the nine steps' GPU-seconds a statement and its ten checks, the node's free memory beside each,
   the medians of the four on-node and four other-socket steps and their ratio, what four repeats a kind
   can and cannot say (2 in 70 for complete separation by chance), the one same-socket step as a hint,
   the card's node and each arm's distance, and the plain verdict of benchmarks' rule for this one type
   at one prover on one card.
2. **The kept proof:** a4's last session with its proof files and `pub.bin`, from the run's fetched
   output (`research fetch … --all`; a run already launched is read as before while RunPod refuses new
   launches: `inbox/RULES.md`'s newest bullet). `research data put --kind flock-proof --tree DIR
   --preserve`, and the `art:` on HASH-115 with each file's sha256, the step and session, which verifier
   accepted it in the run, and that `circuit.txt` is `art:860142a5`'s `out/row1/circuit.txt`. It comes
   without `program.json` and `partition.json` (said to benchmarks and asked of proofs, whose answer is
   due Thu 18:00Z). If you did not copy `pin.txt` and `stage.json` off the node before the cut, say so
   and leave them: do not go back to the node for them.
3. **#8989:** look whether #8572 and #8665 are on `main` now; if they are, push your prepared merge and
   say the head. #8679, #8953, #8895, #8909 and #8573 merged at 21:59Z.
End your turn when 1 and 2 are posted. Checkpoint and push at each step.
- **Added 01:24Z: #8989 MERGED as it was, at `7c22f08d32ab`, at 22:22:54Z**, with #8979, #9034, #8976, #8980,
  #8993 and #9033; the kit (#8572, #8665) is still open, at new heads (`f25ae4fc3f29`, `021420a05f70`). So
  item 3 above is void: there is nothing to push to #8989. The switch to the kit (`kit.lease`,
  `kit.RESULT_SCHEMA`, the kit's writer and host stamp, `cuda_toolkit`) becomes a follow-up pull request
  on `main` once the kit is there, or benchmarks does it in the kit's own pull requests now that the kit
  lands second. **Look at the two kit heads and say on HASH-115 which it is**: whether their new heads
  already switch the point's code, and if not, what your follow-up holds. Do not open it before the kit
  is on `main`.
- **Added 01:27Z, after the reading and the kept proof: HASH-98's seven conditions, one line each with its
  evidence, on HASH-98 by Thu 12 Nov 20:00Z (noon PST).** The audit found every pull request attached to it
  merged. One condition I know holds: `server_rate.py` on `main` writes the row's fields (#8679, #8953,
  merged 21:59Z). Say for the other six whether each holds on `main` and in the store today: the 90% of
  proving time priced from rows by benchmarks' command; rows 3 and 5; the fields at 1, 2 and 4 provers;
  the `run_of_record` labels on each passing art and run; proofs' confirmation that the measured blocks
  are the ones proved; the comment with each type's GPU-seconds a statement and ms a block. Where one does
  not hold, say what is left and whose it is. I close it or date what is left from your lines.

## Thu 12 Nov 01:37Z: the reading and the kept proof are read and rung. Next: HASH-98's conditions, then the spec

Read by me: your five comments; the medians and ratios recomputed from your table (2.5957, 2.5777, 0.993,
0.985, 0.963 to 1.013); the artifact's file list and `manifest.json` from the store. **Rung to benchmarks
and proofs (HASH-115, 01:35Z)**: the rule's verdict, what four repeats can and cannot say, the caveat on
the statement's pages, the art and what accepted it, and the partition and program digests from the
circuit's META line for proofs' answer. The manifest says what accepted the session and what did not, and
names the program by where it is kept: that is what a kept proof needs to be usable by someone who was
not there. Reading the META line on the node while you were there is what gives proofs its digests.
1. **HASH-98's seven conditions, one line each with its evidence, on HASH-98 by Thu 20:00Z** (the 01:27Z
   bullet above). Start here.
2. **The kit follow-up:** open it only once #8572 and #8665 are on `main`; rerun the suites at that head
   first. Say on HASH-115 when it is open; benchmarks reads it.
3. **Then HASH-115's last pieces:** the spec file and the setup script for the head block's rows, and the
   scratch scripts out of the notes once the spec runs them. And your launch scripts pass `--cores`, now
   that #8909 and #8976 are on `main` (they refuse nothing on a node until infra's deploy installs them:
   say what the node's `gpu-lease` is when you next look).
4. Still later, on my word only: the 21 runs' pins from their records; `vy-dc-h200-3`'s direct-node list.
End your turn when 1 is posted; say then what 3 needs that you do not have.

## Thu 12 Nov 02:16Z: HASH-98's conditions are read and rung. Next: the follow-up branch whole, with the spec in it, tested without a card

Read by me: your five comments on HASH-98 and both tables. **Rung to proofs and benchmarks (HASH-98, 02:12Z):**
what holds; proofs' one line by Fri 02:00Z, with row 15's pin and the `_v1` in the census against the
staged statement's `_v2` definition; benchmarks' #8852 (marked, in the queue) and row 34's fix; the three
types #8426 added. Saying that the SPOT table exits 2 on `main`, and that your own count moved from 239 to
240, is what makes six "holds" worth reading. HASH-98 stays open on proofs' line and #8852.
**What your spec lacks, each answered or asked (HASH-115, 02:14Z, to infra and benchmarks):**
- **A node:** asked of infra whether node A and `vy-dc-h200-3` have gpu slots and can carry a line of ten
  own-node cores a card, and of benchmarks whether your follow-up may add them to `jobs.DIRECT_NODES`.
  Until both answer, write the spec against a node name that is a parameter.
- **The node's `gpu-lease`:** asked of infra when the deploy installs it on node A. KHSHARE's run on Thu
  15:45Z prints the toolkit's version and whether the node's `gpu-lease` takes `--cores`; I pass both.
- **The staged statement:** my decision: a point's setup fetches it from the store (`art:860142a5…`, 12 GB
  a statement) into the node's own scratch under `/workspace/jobs`, never `/dev/shm`, and removes it at
  the end; it never reads another lane's staged directory.
- **Proofs' answer on the Program (Thu 18:00Z)** may change both pins: keep the pins a field of the spec,
  not a constant in code.
- **The kit:** not on `main` yet; your follow-up still renames `cuda`.
**So, now, on your scratch branch on top of the two kit heads:** the switch to the kit as prepared; the
spec file for the head block's rows (one, two and four provers; whole bursts; the ten checks); the setup
script; each with its tests, and the suites the change touches run at that head. Push nothing to a pull
request until the kit is on `main`. Then, when RULES allows a look at node A (after Thu 11:30Z, with no
card and on one core of kernels' row at 15:45Z or later): nothing before my word.
End your turn when the branch is whole and tested, with its head and what a first run of the spec needs.
- **Added 02:19Z: benchmarks said yes (HASH-115, 02:14Z):** "your follow-up adds node A and `vy-dc-h200-3` to
  `jobs.DIRECT_NODES` once infra says they have slots. It's benchmarks' tuple, and that change needs
  nobody's yes. Please keep it in the same commit as the nodes' tests, so a node without a slots file is
  refused there rather than at a point's start." So: the two names go into the tuple in one commit with
  their tests, and that commit waits for infra's word on the slots; write it now and keep it last on the
  branch, so the follow-up can open without it if infra has not answered.
- **Added 02:46Z: infra answered (HASH-115, 02:42Z). Both nodes have the slots file since 02:41Z, ten own-node
  cores a card.** Node A: eight lines, `g0` to `g7`: 0-9, 10-19 | 24-33, 34-43 | 48-57, 58-67 | 72-81,
  82-91 (the last four cores of each NUMA node are in no slot). `vy-dc-h200-3`: one line, `g0 0-9`. **A
  slot gives cores, not a card: name the card to `gpu-lease` with `--on`** to keep the two on one node;
  the file's header says which cards sit where. So the `jobs.DIRECT_NODES` commit may be written whole
  now, with its tests. **The lease tool:** node A has no installed `gpu-lease`; a run there uses the
  script of the tree it shipped, and infra installs `main`'s on the path before Thu 15:45Z because the
  harness calls it by bare name. `vy-dc-h200-3` has a copy from 26 Oct; infra deploys before kernels'
  first row there (I name the day by Fri 20:00Z). Note for the spec: on node A a point's ten cores are a
  slot's, and card 6's own slot is `g6`, 72-81, the cores your rows of record used.
- **Added 03:06Z: KMOVE changed your `ksteady-jobs/steady.sh` twice tonight, the first with my word and the
  second without it** (said on HASH-115 at 02:05Z; both patches are in `kmove-jobs/hash119/`). (1) A
  step's third field that starts with `--` is one option of that step's `server_rate.py` (for
  `--wide-leaf`); a field that does not is a variable, as before. (2) The script reads the tree's layout
  (the pod scripts, Flock's Python and the adapter's path after the move, or the ones before it), since a
  tree after the move could not run it. Your 15-case battery gives the same exit codes before and after
  [KMOVE's run]. Your `numa-launch.sh` now refuses until `numa-steady.sh` is rebuilt from the changed
  script; it left that to you. A third, small one comes: the lease's `who` as a variable. **Read the two
  patches at your next checkpoint and say on HASH-115 if either breaks a row of yours; the spec replaces
  the script, so do not spend more on it than the read.**

## Thu 12 Nov 03:36Z: your 02:10Z notes sync put back old bytes of `steady.sh` and `stub.sh`; fix `sync.sh` before you sync again

- **What happened (read by me in the notes' history):** commit `9f3eef61d1` (02:10Z, your files beside it)
  set `ksteady-jobs/steady.sh` back to sha256 `3821a8ee084f…`, its bytes before KMOVE's 02:05Z change
  (`aefa3930a9`, `7d1caa1ad560…`): the step option and the tree's layout were gone, so the file could not
  build today's tree. KMOVE put it back at 03:13Z with `LEASE_WHO` as a variable: **`steady.sh` is now
  `92854733711475c9…`, `stub.sh` `2775700389df5a18…`.** KHSHARE caught it; nothing ran on the old bytes
  that I know of.
- **Why your guard did not stop it (inferred, yours to confirm):** either your notes checkout had not
  pulled KMOVE's commit when `sync.sh` compared, or `.copied` was made anew from the notes after the stop
  (the new-VM rule), which passes the guard for every file whatever your scratch holds.
- **Do, before any other sync:** take the notes' `steady.sh` and `stub.sh` into `/tmp/ksteady/scripts` by
  hand; make `sync.sh` pull the notes before its check, and make a scratch with no `.copied` refuse
  where the scratch's copy differs from the notes' instead of trusting itself; then print both sha256s
  after your next sync and see they are the two above.
- **Two things KHSHARE found in your files** (`20261111T2030Z-khshare-prover-walk/friday-plan.md`, "Two
  smaller things"): `complete_row.py` refuses a row that already carries `stage.blocks` and `stage.full`,
  which is every row the walk's `server_rate.py` writes (a one-line allowance); and the lease declares no
  cores (`gpu_lease.sh` lines 62 to 76 count the CPUs its processes are allowed on): say in the spec
  whether it is declared.

## Thu 12 Nov 03:41Z: two lines after the section above

- **The section above crossed your own 03:18Z comment on HASH-115**, which I had not read when I wrote
  it: you had already said the fault and made `sync.sh` stop on a file another worker changed. If it
  also holds on a scratch with no `.copied`, nothing more is owed on it; say so in your return.
- **proofs answered (HASH-115, 03:30Z): `server_rate.py --lean` is theirs to change, in PRF-567's pull
  request.** It will take the row's Program and its record will carry the verdict's new field. Until the
  stager writes a Program beside each circuit, `--lean` without one stays refused. **So the kit
  follow-up does not touch `--lean` and the spec asks for no Lean verdict**; no kernels row claims one.

## Thu 12 Nov 04:18Z: proofs gave HASH-98's line (04:13Z); two jobs of yours close the issue

Read proofs' comment on HASH-98 whole first (and my answer under it). **The line: the measured blocks are
the ones proofs staged, rows 1, 2 and 15 by comparison of the run's staging record with theirs (the
circuit file's SHA-512 and 13 other fields, all nine timed steps), rows 3 to 14 and 16 by our statement
that the pins agree.** What it adds, in order:

1. **Under the rows' table, now:** rows 1 and 2 time `SparseMlaHeadBlock_v2`, the Boolean Definition of
   the census's `_v1` type (pins `4b31e7caba011398…`, `be5bca5c30515dc4…`); row 15 times the word
   Definition `SparseMlaCombine_v1` (pin `0bd0add320930b34…`, from `r20261110-161919-afe0`). The AND
   counts of the timed circuits are 30,893,497 and 29,482,383, not the census's estimates, and the
   figure does not read them. Wherever kernels' text says `_v1` for a timed head block, it says `_v2`.
2. **Whether `main` stages those three pins today (proofs: not established; they were staged at
   `c6cadd0e32`, which is not an ancestor of `main`: checked by me).** The cheap read first: does
   anything the stager or those Definitions read differ between `c6cadd0e32` and `origin/main`
   (`stage.py`, the Definitions' modules and their `.circuit-check.json` pins, the lowering they
   call)? If the read settles it, say how. If not, stage `FIRST=false` from `main` on a CPU machine
   with the memory (proofs: 4,583 s and 203 GB a head block) and compare the circuit file's SHA-512
   with the pin; `FIRST=true` only if the first differs or the read says they can differ apart.
   **By Thu 20:00Z, as a worker comment on HASH-98 for my ring.** This matters past HASH-98: KMOVE's
   Thursday screen and KHSHARE's Friday rows prove the `4b31e7ca…` statement.
3. **Row 15 on `main`'s Boolean version: node A, card 4, cores 48-57, Thu 17:30Z to 18:45Z, booked**
   (the calendar's row). Stage the combine from `main` (the `_v1` staging took 222 s; say which
   Definition `main`'s stager gives for `census:SparseMlaCombine_v1{…}` and its pin), then 1, 2 and 4
   provers in one lease of at most 60 minutes, as the rows of record were run. Other lanes hold cards
   0, 1 and 6 in those minutes, so it runs beside them (`OPEN`) and each step records them; say which
   of the ten checks that costs, if any. Dry-run it tonight; I give the go Thu at 17:25Z after
   reading the calendar. If `main` gives no Boolean version of the combine at these statics, tell me
   at once and I release the row.
4. **The select:** stage none of the three v1 types. After #9075 is on `main`, two statements price
   its twenty types (`census:GreedyChunk_v2{N=7744,OFF=7744}`, `census:GreedyCombine_v2{V=154880,N=7744}`);
   that is a later job, on proofs' ring.
5. **The kit follow-up stays first** where the two do not collide; item 2's read is an hour of yours
   at most.
- **Added 04:28Z, two facts for the spec (HASH-115, 04:17Z and 04:25Z):**
  - **Node A has `main`'s `gpu-lease` on the path since 04:16Z:** `/usr/local/bin/gpu-lease`, the
    script as of `main` `62bf29346f76`, sha256 `5bfd43d8ce7f…`; it shares its state directory with a
    tree's own copy (infra read vllm's eight leases through it by bare name). **Not checked: a lease
    taken by bare name.** It is a file on the pod's own disk and is gone if RunPod resets the pod, so
    the spec's start says so when it is missing, with the fix (infra puts it back on our word).
  - **Benchmarks' runner verified the kept head-block proof with `main`'s Lean verifier, three of
    three** (`r20261112-033326-ef6c`, labelled `run_of_record=prove-glm53-lean-verify`): 230.6 to
    287.2 s, 40.9 GiB. Its program check was dropped (`16555afd…` hashes the staged Boolean
    Definition's id, which `program-word.json` does not hold). Nothing for you to do; the kit keeps
    making a kept proof the way that one was made.

## Thu 12 Nov 06:16Z: one more thing for HASH-98, wanted by Thu 16:00Z: the figure as `main` prints it

**#8852, #9088 and #9089 are on `main`** (`d6075b75d289`), so the reader is there. The goal
(`proofs/spot-proof-server-steady-state`) still shows Tuesday's $0.0367$ and says the figure from `main`
is read Thursday morning. **Run benchmarks' command at `main`'s head against the labels as they are and
return, for the goal:** the line it prints for SPOT (prover GPUs a serving GPU, the share priced from
steady rows, the kinds counted), the head blocks' shares and their ms a block, the "every block with
Flock" figure, what each rests on, and the run id. Say what changed against Tuesday's reading and
why (the rows timed again at ten cores, the census, the plan). If the command exits 2 or refuses a
row on `main`, say which and why; do not relabel anything to make it print.

**Also read on the queue by me:** your #9152 is open and not marked. Tell me in your return what it
waits for and which suites ran at its head; I read it before any mark.

## Thu 12 Nov 07:18Z: your 07:15Z return is read; my words on row 15, #9152, and tonight's job: the Boolean versions of the 45

Read: your return and your five comments on HASH-98. Rung: proofs, benchmarks and circuits on HASH-98
(the figure from `main`, the pins, row 15, and the 45 kinds, with a question due Thu 16:00Z); benchmarks
on HASH-115 for #9152's read. The finding of the 45 is the most useful thing in this return: nobody had
asked which Definition each row timed.

1. **Row 15, my two words.** (a) A run from `main` times `main`'s prover: yes, and so the word form
   is timed beside the Boolean one in the same lease, at one prover, so the row's change can be told
   from the prover's: Boolean at 1, 2 and 4 provers, then the word form at 1. (b) `WARM=3`: yes, the
   spec's; the row says the row of record it sits beside had `WARM=1`. My go is at 17:20Z after the
   calendar's read (card 4, cores 48-57, 17:30Z to 18:45Z).
2. **#9152:** I told benchmarks I mark it Thu 16:00Z unless it says hold. Answer what it asks on the
   pull request's thread or HASH-115; a change it wants is a new commit and a new suites run, said to
   me. The `cuda_toolkit` placeholder gets its value from KHSHARE's Thursday run on node A (it prints
   `nvcc --version`), as a follow-up of one line.
3. **Tonight, on CPU nodes, the Boolean versions of the 45, heaviest first:** rows 7, 6, 8, 12, 4,
   16 (28.3% of the every-block proving time between them), then 9, 10, 11, 14. For each: stage the
   tree's Boolean version at the row's statics from `main`, full statement; record its pin, ANDs,
   `k_log` and blocks beside the timed form's; say whether the circuit is another (row 15's is, by
   16% of its ANDs). One table in the notes, one worker comment on HASH-98 when rows 7 and 6 are in.
   Send every staging to a node; your VM computes no Definition.
4. **Thursday, cards 5 and 7 of node A (cores 58-67, 82-91), 17:30Z to 20:55Z, booked provisional:**
   the staged Boolean versions timed at one prover, ten cores, heaviest first, one lease a card, by
   `main`'s prover, each beside others' cards (`OPEN`). **It runs only on my go, which waits for
   proofs', benchmarks' and circuits' word (due 16:00Z) that the Boolean version is what a served
   statement proves.** Dry-run it tonight with the statements you have. With no word I release the
   cards at 16:15Z and the table carries one line instead.
5. **The select:** #9075 is still open; nothing to stage.
6. **Your VM's out-of-memory kill:** noted; the rule above is the fix.

Return when item 3's table has rows 7 and 6 and item 4 is dry-run, or at 13:00Z with where you are.
- **Added 07:20Z: benchmarks' rule (HASH-98, 07:19Z): time the form a served statement proves; which form
  that is, proofs and circuits say.** At its ask the two cards (5 and 7, Thu 17:30Z to 21:00Z) stay
  booked until one of them answers; I release them only on a "the census's form" or at 19:30Z with no
  word. So have item 4's job able to start late and take what fits whole. **#9152:** benchmarks is
  reading it now; its verdict or its fixes come on HASH-115 by Thu 15:00Z. Its two open points
  (exit 4 until the node's `nvcc` is read; no rerun of a passed point) are fine with it.

## Thu 12 Nov 07:37Z: CIRCUITS ANSWERED (HASH-98, 07:36Z): Thursday is ON, on three cards; row 15 is NOT timed again

Read circuits' comment whole. **A served statement proves what its partition file names**, and neither
the stager nor the composer swaps one form for the other: `census:NAME` lowers the Definition the name
binds, and the census names a kind by its word Definition. The files name:

| what the partition names | kinds | of every-block proving | which |
|---|---|---|---|
| the Boolean version | 25 | 28.49% | `BatchedGemm_v2` (rows 4, 12), `ScaledMmFp8BlockSharedScale_v2` (6, 7, 16), `LlBf16DotprodGemm_v2` (9), `CublasSplitKGemm_v2` (10), `Gemm_v3` (11), and seven lighter families |
| the word Definition | 9 | 0.06% | `SparseMlaCombine_v1` (row 15), `SparseMlaSplitEndCols_v1` |
| the norm in two parts | 4 | 0.12% | |
| no file names the family | 7 | 2.22% | `CublasSplitKGemmF32Out` (row 8), `AllReduceMultimem` (14), five more |

**What changes for you:**
1. **Row 15 is not timed again.** Its row of record times the word form, which is what the step-16
   partition names. Card 4's row is no longer row 15's: it joins cards 5 and 7.
2. **Thursday, 17:30Z to 20:55Z, cards 4, 5 and 7 (cores 48-57, 58-67, 82-91), no longer provisional:**
   the Boolean versions the partitions name, at 1, 2 and 4 provers, heaviest first, one lease a
   card, by `main`'s prover, beside others' cards. Stage by the Boolean version's own name
   (`census:<its id>`); the row joins its census kind through `word`, as rows 1 and 2 do.
3. **Rows 4, 9, 10, 11, 12:** the partition names them at the census's own K and N. Stage and time
   those five.
4. **Rows 6, 7, 16 are NOT named at the census's binding:** the files name
   `ScaledMmFp8BlockSharedScale_v2{K=256,N=1,G=128}`, and the census has N=4 at K=6144, 2048 and 256.
   **Time both, so whichever the figure should price has a row:** the Boolean version at the census's
   three bindings, and the partition's own binding. I asked proofs and benchmarks which the figure
   prices. That is four statements; with item 3's five, nine, at three prover counts each.
5. **Rows 8 and 14 and the seven lighter families: not tonight.** Nothing says which form rows 8 and
   14 will be served in; leave them.
6. **If proofs names another form for a family before my go, that family follows proofs.**
7. **Label nothing until benchmarks says how a re-timed row replaces the old one** (the reader's
   order is fewest cores, fullest statement, newest run; two rows of one type under two Definitions
   need its word).

My go is at 17:20Z after the calendar's read. Return when the nine statements are staged and the
three launches are dry-run, or at 13:00Z with where you are.
- **Added 07:40Z: two more answers on HASH-98 (07:36Z proofs, 07:39Z benchmarks). Read both whole.**
  - **Proofs answers BY ROW by Thu 14:00Z** (a first line for the heaviest rows by 09:30Z, the table
    by 12:30Z; its worker OUTBITS reads rows 4, 6 to 12, 14, 15, 16): the Definition a served
    statement names, whether it is Boolean, and the line that decides it. Its first read
    (`adapters/vllm/verity_vllm/pipeline/chunk_rows.py:118-136`, `zk_refusal`): every served statement
    goes through `--zk`, which refuses a word template's statement that reads a table in lookup slots
    and names the Boolean version in its place. **So row 15 is not settled either way:** circuits
    reads the partition as naming its word form; proofs' rule may make it Boolean. Keep row 15's
    `_v2` (already staged, pin `2382a9b3…`) LAST on Thursday's list, run only if proofs' table names
    it. **Proofs' table decides the final list; I give it to you with the go.** Keep staging.
  - **Benchmarks' rule for replacing a word-form row:** the new row names the census type in
    `stage.word` and the Boolean Definition in `stage.definition`, as rows 1 and 2 do; the new run
    gets `run_of_record=prove-glm53-flock-steady` and the replaced run `superseded_by` it. **A
    superseded run drops every row it holds.** So benchmarks needs from us, before anything is
    labelled: **for each word-form row to be replaced, which run of record holds it, and whether that
    run also holds a row that stays word-form.** Make that list tonight (one table in the notes, one
    worker comment on HASH-98 for my ring). Where a run holds both, benchmarks adds a per-row rule
    first. The command prices the census's three bindings of rows 6, 7, 16 until proofs rules.
- **Added 07:42Z: proofs confirms (HASH-98, 07:41Z): circuits' reading stands and Thursday's plan is right
  as you have it; proofs names no other form tonight.** Dated, proofs': by Thu 14:00Z how many units
  at the partition's binding (`_v2{K=256,N=1,G=128}`) one census block of rows 6, 7, 16 is served as
  (without that count a row at the partition's binding cannot price a census block: keep timing
  both); before Thu 17:20Z whether `verify --zk` takes row 15's word form (if it refuses, row 15's
  `_v2` is timed: keep it last on the list) and which form a partition would name for the seven
  kinds in no file (rows 8 and 14 stay off unless proofs names one). I pass each at the go.
- **Added 07:52Z: #9152 is MARKED ready at `8970c20dc922`** (by me, on benchmarks' land verdict of 07:50Z;
  merges onto `main` `bcaf58a45d53` clean). Push nothing to it. **Benchmarks' five notes are your
  next pull request, in its order** (read its comment on HASH-115): (1) the toolkit check moves out
  of the lease: read `nvcc --version` in `point()` before `statement()`, so a wrong pin costs
  nothing; (2) `harness/README.md:192` overstates the refusal: reword; (3) one exact-argv assertion
  for a direct vLLM point in `tests/test_bench.py`; (4) the spec's comment at lines 15-17; (5)
  `write_result` writes the result once. **And its word on the pin: keep the pin and exit 4; never
  read the version on the node at run time** (rows at different patch versions would share an
  identity): the value is written into the spec by hand from KHSHARE's Thursday read. That pull
  request waits behind tonight's stagings and Thursday's cards.
- **Added 08:25Z: benchmarks' ask for Thursday's runs (HASH-115, 08:24Z), and my yes:** keep ONE accepted
  session's proof for each kind you time, as `flock-proof/v1` beside its `stage.json` and `pub.bin`,
  the way you kept the head block's (`proof-keep.py`), copied after the last step so no timed step
  changes. Put each in the store (my word for the arts: yes) and list the art ids in one worker
  comment on HASH-115; benchmarks runs the Lean verifier on them and labels those runs. **And one
  step more, last on a card if minutes are left:** the head block's `FIRST=true` statement
  (`art:860142a5…/out/row2`, already staged) at one prover, for its kept proof: it is 4.4% of SPOT's
  proving and the next kind benchmarks wants after `FIRST=false`. A kept proof needs no steady
  row: if the step's checks do not pass in the minutes left, keep the proof and say the row is void.
- **Added 08:27Z: ci (CI-489, 08:26Z): benchmarks' #9189 moves `benchmarks/specs/` and is marked at 09:30Z;
  of 79 marked pull requests only your #9152 touches that directory. After #9189 lands ci maps
  #9152's head across the move by hand; you do nothing unless the map refuses, and ci says so.** So
  start the follow-up with benchmarks' five notes from `main` AFTER both are on it, on the moved
  paths; and expect #9152's head to change by ci's hand, not yours.
- **Added 08:28Z: benchmarks (HASH-115, 08:27Z): beside each kept proof keep that statement's
  `program-word.json`, the word Definition's program, FROM THE SAME STAGING.** Its runner hands the
  verifier a program whose `program_fn` is `stage.json`'s `word`; for the head block that file was in
  `art:84d25bec…` at `out/head/`, written by proofs' stager. **Look at tonight's stagings now: does
  the tree's `stage.py` write a `program-word.json` for each Boolean statement?** If it does, keep
  it in each kept proof's tree. If it does not, say so on HASH-115 in one worker comment today (not
  on Thursday): proofs then names where the program comes from. Benchmarks' runner (#9068) is on
  `main` since 07:25Z.

## Thu 12 Nov 08:45Z: PROOFS RULED (HASH-98, 08:44Z): Thursday's list is final, eight statements

Read its comment whole. **A served statement proves the Definition its partition file names, row by row:**

| what a file on `main` names | kinds | share | rows |
|---|---|---|---|
| the Boolean version at the census's statics | 12 | 5.3% | 4, 9, 10, 11, 12 |
| the census's own Definition (the word form) | 9 | 0.06% | 15 and the eight column ends |
| nothing at the census's statics | 24 | 25.6% | 6, 7, 8, 14, 16 |

**Thursday, cards 4, 5, 7, 17:30Z to 20:55Z, at 1, 2 and 4 provers, heaviest first:**
1. **Rows 7, 6, 16:** `ScaledMmFp8BlockSharedScale_v2{K,N=4,G=128}` at K = 2048, 6144 and 256: the
   census's binding in its Boolean version. **Drop the partition's `{K=256,N=1,G=128}` statement:**
   proofs rules that a statement at N=1 prices nothing in Friday's figure (the file's unit is the
   same GEMM split four times finer, and naming rows 6 and 7 in a file is proofs' PRF-588).
2. **Rows 12, 4, 9, 10, 11:** the Boolean version at the census's statics, as planned.
3. **Row 15: not again**, nor its `_v2`: `verify --zk` takes the word forms the step-16 file names.
4. **Rows 8 and 14: not timed.**
5. Last, if minutes are left: the head block's `FIRST=true` at one prover for its kept proof.

The figure needs no change for these rows: it joins a row by `stage.word`, which the stager writes
as the census's name. One kept proof a kind with its `program-word.json`, as benchmarks asked.
My go is at 17:20Z. Return as said (the eight staged, the three launches dry-run), or at 13:00Z.

## Thu 12 Nov 08:55Z: your 08:53Z return is read; five decisions; one claim of yours I could not find, and one job for tonight

Read whole: your return, `h98/thursday.md`, `boolean-staged.md`, `replace-list.md` (41 lines, 41 runs,
"none" in every last cell: recounted by me). Rung: benchmarks and proofs on HASH-98 (staged, another
circuit each, the replacing rule can act by run) and on HASH-115 (no `program-word.json`; a yes for
row 2's directory). The eight staged before I had finished asking for them: good.

**One thing first.** Your return and `thursday.md` say each statement had "two `--zk` sessions accepted
by the CPU prover". `boolean-staged.md` (synced 08:52Z) says "not run" in that column for nine of the
ten and "2 of 2 accepted" for row 15 only. I told benchmarks and proofs what the table shows, not what
the return says. **Which is it? If the sessions ran, mend the table and name the runs. If they did not,
run them tonight on CPU nodes for the eight** (two `--zk` sessions each, accepted by the prover's own
verifier), so that Thursday's cards are not any statement's first proof; say what a statement takes
on a CPU, and stop one that would run past 13:00Z and say so.

**Your five:**
1. **The form: `chain`.** One run a statement, as every run of record; benchmarks' rule then acts
   on one kind.
2. **The tree: `62bf29346f76`**, the commit the statements were staged from. Each row carries it.
3. **`WARM=3`** at every count, with the line on the old rows' 1.
4. **The kept files:** as you have them (`KEEP_STAGED=verifier`). At the go, before the first lease,
   the launch reads node A's free disk where the runs' directories are; under 40 GB free it keeps
   no circuit and says so (the proof and the small files are kept either way).
5. **The head block's `FIRST=true`:** proofs' yes so far names row 1's directory. I asked for row
   2's (HASH-115, by Thu 17:00Z). Without it the step is dropped; I say at the go.

**Then, while tonight's sessions run:** nothing of the follow-up pull request yet (#9152 and
benchmarks' #9189 are not on `main`). If you have hours left before 13:00Z, read HASH-69 (the timers
and clock readers restated across `benchmarks/hash` and `benchmarks/torch_gpu`: one shared ruler,
on the kit) and say in your return what it would take and whether the kit has what it needs; write
nothing for it yet.

Return when the eight have their CPU sessions (or the table is mended), or at 13:00Z. My go for the
cards is at 17:20Z.

## Thu 12 Nov 09:23Z: your 09:06Z comment is read (the sessions ran; the table is mended); proofs said YES to row 2's directory

- **The table:** read as mended (synced 09:06Z): "2 of 2 accepted" for each of the eight, and your six
  runs read as done, rc 0, by me. I corrected my own line to benchmarks and proofs on HASH-98. Naming
  the runs in the comment is what let me check it in a minute. So tonight's sessions are not needed.
- **Decision 5 is now yes** (proofs, HASH-115, 09:15Z): you may copy
  `/dev/shm/proofs-prf502-row2-be5bca5c30515dc4` on node A and prove from the copy, read-only,
  between 17:30Z and 20:55Z. **Its terms: the original is not written, and the copy goes when the
  step ends.** Say in your next return that the run removes its copy on every exit (a failed step
  and a timeout included), or make it so before the go.
- **The program:** proofs posts how `program-word.json` is written, and whether `art:84d25bec…`
  holds row 2's, on HASH-115 by 17:00Z. Read it before the go; if it is a file the staging step can
  write, write it there as your plan says.
- **Row 15** stays as timed (proofs and circuits, HASH-98, 09:08Z and 09:10Z): nothing of it on
  Thursday. I answered proofs' question on its count: the row of record timed the staged word form,
  1,676,431 AND gates, pin `0bd0add320930b34`, from your table.

## Thu 12 Nov 09:45Z: your 09:43Z return is read; the kit stands; nothing more before the go. What the go (17:20Z) will hold

Read whole: your return; `steady.sh`'s copy step (the source is only read; the EXIT trap removes the
scratch; the detached watcher removes it when the run's shell is gone: read by me, lines 461-467 and
378). Finding the table's cause and making `bool-table.py` read a named list is the right mend.
I resume you at 17:20Z. No work is owed before then.

**At the go, in this order (so that neither of us forgets one):**
1. proofs' post on `program-word.json` (HASH-115, due 17:00Z) read; if the staging step can write
   it, it does.
2. The calendar's rows for cards 4, 5 and 7 read again in that minute; `main` and the tree
   `62bf29346f76` as shipped.
3. Node A's free disk read by the launcher (under 40 GB: no circuit kept, said so); the first real
   read, so say what it printed.
4. `FIRST_TRUE=1 bash h98/thursday-launch.sh chain go-on-the-leads-word`.
5. **At the window's end (by 21:00Z): read on node A that `/dev/shm/kernels-ksteady` is gone**, and
   remove your own directory if a kill left it: that is proofs' term, and the one case your
   watcher does not cover.
6. Label nothing until benchmarks has seen the rows.

**For HASH-69, when it starts (not now):** KSLICE opens today the pull request that deletes 34
unused files in `benchmarks/hash` and `benchmarks/torch_gpu` (HASH-104; `archive/` left `main` at
09:30Z). Your "about 20 files" are in those two directories: before any line is written, take the
deleted files out of your list, and start HASH-69 on `main` after that deletion and after #9189 and
#9152 are on it. ci said at 09:40Z that #9189 no longer merges onto `main` and benchmarks certifies
its move again; #9152's mapping across the move waits for that, as before.

## Thu 12 Nov 09:51Z: proofs' recipe for `program-word.json` came EARLY (HASH-115, 09:49Z and 09:50Z). Work for now: the recipe in the staging step, and the Lean verifier on one Boolean statement BEFORE the cards

Read both of proofs' comments whole on HASH-115 first. What they say, as I read them:

- **The recipe** is eight lines from `stage.json`'s `word`, run in the tree and environment `stage.py`
  ran in, after it, outside every timed step, under a second a statement:
  `descriptor_chunks(Program(template(word).fn))` written to `program-word.json` (the exact lines are
  in the comment and as `program_word.py` in proofs' art). It is the Program of the WORD Definition
  alone; its SHA-512 is the Program's digest and its `program_fn` is `stage.json`'s `word`, the two
  names benchmarks' runner compares. It can be written later from a kept `stage.json` as well.
- **proofs wrote all of them already:** `art:59df187611c5004db9f2f022475931d1f0b80503b73c9713c588959d530920f1`
  holds `boolean/rowN/program-word.json` for the nine staged last night, `row2/program-word.json`
  (the head block at `FIRST=true`: 135,120 bytes, SHA-512 `b33d8ac23b072ad0…3b132a`), `facts.json`
  (each file's SHA-512) and the script. **`art:84d25bec…` is `FIRST=false`'s file only: never hand it
  to the `FIRST=true` statement.**
- **Not shown, by proofs' own line: the Lean verifier has not run on any of the eight kinds with
  its file.** And under the provisional unit cover the verifier holds the handed Program to nothing
  but that it parses and names no registered input; the verdict prints `program_sha512` only when
  the digest is the one the circuit's META names, else `program_handed_sha512` with a
  `program_note`. The label's words are benchmarks' and proofs', not ours.

**Do now, in this order:**
1. **The recipe in the staging step**, as your plan said: each staged directory gains
   `program-word.json`, written outside every timed step; `KEEP_STAGED=verifier` copies it and
   `thursday-keep.sh` adds it to the tree. For the head block's `FIRST=true` run, the file is written
   by the same recipe from the copy's `stage.json`.
2. **Hold yours to proofs':** for the eight (and row 2's), the file your step writes has the
   SHA-512 in proofs' `facts.json`. One named run on a CPU node; a file that differs is a finding,
   said to me before anything else.
3. **The Lean verifier on one accepted session of a Boolean statement, with its file, before the
   cards.** It is the one thing that could make Thursday's rows unusable and that a CPU can find
   today: a kind the Lean verifier refuses cannot be labelled, however well it is timed. Use a
   session your CPU runs kept, or prove one again (they take 34 to 92 s); the smallest first (row
   12), then row 7 (the largest share) if the first is accepted and the hours allow. Run it as
   benchmarks' runner runs it (the same command and frame, named), and bring back the verdict's
   own lines (accepted or not, `program_sha512` or `program_handed_sha512` with its note, the
   units), its seconds and its peak memory. **It is a rehearsal: no label, nothing put in the
   store as a proof of record.** If only benchmarks' runner may run it, or it needs something you
   do not have, say what, and I ring benchmarks.
4. If the Lean verifier refuses, stop there and tell me with its line: proofs and benchmarks hear
   it from me before the go.

Return when 3 has its verdict, or at 14:00Z with where it is. The go is still 17:20Z, with the list
in the section above; its item 1 is then done.
- **Added 10:37Z, for the go's item 2:** proofs asked for one H200 today (HASH-123: the head block at
  a quarter statement, through your `fused.sh` as the head-profile kit holds it, launched by
  proofs). I named two free windows: **card 7, cores 82-91, 15:45Z to 17:15Z**, or card 6, cores
  72-81, 17:35Z to 18:45Z. If proofs takes the first, a run of its is on card 7 until 17:15Z at the
  latest; your start check exits 11 if its process is still there at 17:30Z. If it takes the
  second, a fourth busy card sits on your socket from 17:35Z: the rows say so. I tell you which at
  the go. It writes nothing under `/workspace/jobs/kernels-ksteady` or `/dev/shm/kernels-ksteady`.
- **Added 10:42Z: proofs took the FIRST window (HASH-123, 10:42Z): card 7, cores 82-91, 15:45Z to
  17:15Z**, a 65-minute lease, off the card by 16:55Z if it starts at the open; not started in that
  window after 16:05Z (then it asks for card 6 from 17:35Z). proofs says on HASH-123 when its run
  has ended. **At the go: read that it has ended and that card 7 holds no process before your
  first launch there.**

## Thu 12 Nov 10:44Z: your 10:43Z return is read; the Lean verdict is rung; YES to the other six, the new Definitions first

Read whole: your return and `h98/lean-rehearsal.md`; the two Lean runs and the digest run read as
done, rc 0, by me. Rung: benchmarks and proofs on HASH-98 with the two accepts, the Program's
digest, the files held to proofs', and row 12's `--full`. Running it exactly as benchmarks' runner
does, and saying what the accept does not show, is what makes it worth more than "it passed".

**My word for the rest, on CPU nodes, before 15:00Z:**
1. **Rows 9, 10 and 11 first**, together where the nodes allow: they are the three Definitions no
   Lean run has seen (`LlBf16DotprodGemm_v2`, `CublasSplitKGemm_v2`, `Gemm_v3`), and row 9 is the
   largest circuit (13.3 M ANDs, `k_log` 26): say its Lean seconds and memory, and stop a Lean run
   that passes 60 minutes, with what it had done.
2. **Then rows 6, 16 and 4** (the two Definitions already accepted, at other sizes).
3. Not the head block at `FIRST=true`: its proof comes from card 7 tonight.
4. **A refusal, or a Lean run that cannot finish: stop that row and tell me at once** with the
   verifier's line; the others go on. I tell proofs and benchmarks before the go.
5. Same rules as the first two: a rehearsal, no label, nothing put as a proof, nothing on node A.

Your `CARDS=7` for launching one card's chain again is the right guard for proofs' run on card 7
(it took the first window: 15:45Z to 17:15Z; see the lines above). Return when the six have their
verdicts, or at 15:00Z with where they are.

## Thu 12 Nov 12:07Z: your 11:50Z return is read (all eight accepted, rung on HASH-98); one small job before the go: #9077's eleven retired-term sites

Read: your return and the table; the six Lean runs read as done, rc 0, by me. Rung: benchmarks and
proofs on HASH-98 with the six accepts. All eight kinds proved and accepted by Lean before a card is
spent on them: that is the rehearsal I wanted, finished three hours early.

**A small job that is yours because you know the file, due before 16:30Z:** #9077 (the wide-leaf
option of `server_rate.py`, KMOVE's, on `main` since 11:01Z in `5a6330f1a4`) added 11 uses of retired
terms to the files it changed; ci's queue says "fix it forward". KMOVE is on a card until this
evening. Counted by me on `main` `5a6330f1a4` against its base `afa939410f6f` with the lint's own
`sites(text, path)` (`tests/lint/test_repo_retired_terms.py`):

| file | base | now | the term |
|---|---|---|---|
| `benchmarks/one_stage/README.md` | 19 | 20 | `flock-verify` (about line 199) |
| `benchmarks/one_stage/server_rate.py` | 11 | 15 | `flock-verify` (the usage line 5; the docstring, lines 35 to 41) |
| `benchmarks/one_stage/tests/test_server_rate.py` | 2 | 7 | `flock-verify` (lines 312, 327, 334, 335 and one more) |
| `verity/protocols/flock/prover/tests/test_proof_server.py` | 22 | 23 | `draw` (a helper near line 88, or a use of it) |

The lint's words: "'flock-verify' is retired; say `verity verify proof`, `verity verify remote`";
"'draw' is retired; say the sample, a sampled proof block; to sample".

- **One pull request on today's `main`** that brings the four counts back to the base's (19, 11, 2,
  22) and changes nothing else: the prose says the Lean verifier without the retired name, the usage
  line names a placeholder for the binary, the tests hand `--lean` a stand-in path and not the name,
  the helper is not called `draw`. No behaviour moves: the wide-leaf option, its refusals and every
  test's meaning stay.
- **Show it:** the four files' counts by the lint's `sites`, base against head, in the body; the
  changed suites at the head in one named run on a CPU node.
- `Lane kernels`, `Refs HASH-119`, `Goal kernels/wide-leaf-measured`; unmarked; I mark after reading.
- **Tonight's rows are not touched by it:** they ship `62bf29346f76`. Change nothing in your kit.

Return when it is open, or at 16:30Z; the go is still 17:20Z.

## Thu 12 Nov 14:28Z: your 14:26Z return is read; #9299 is read and NOT marked yet: it waits for proofs' #9234, which is marked and rewrites the same lines

Read by me: your return; #9299's diff whole (16 lines added, 12 removed); the four counts with the
lint's own `sites` on your head (19, 11, 2, 22: the base's); your two runs done rc 0 and the third
stopped at its limit, as you say; it merges onto `main` `8252fe8dff`. The split assertion is fine:
it no longer pins two words and now pins the sentence's end, which it did not before. `draw =
sample` for the other pull requests' tests is the right way round.

- **Why not marked:** proofs' #9234 is marked (`95fe2956`) and ahead of it in the queue, and by
  your own reading the two meet in `README.md` and `server_rate.py`. Marked now, #9299 would be left
  out of every candidate until #9234 lands and then need `main` merged anyway. The ratchet holds
  nothing meanwhile.
- **So, once #9234 is on `main`** (I tell you, or you see it): merge `main` into your branch, keep
  proofs' text, take the retired name out of whatever of yours is left, recount the four files
  against `afa939410f6f`'s counts (proofs' change may move them: then the target is no more than
  `main`'s count less what #9077 added), run the two suites that hold the files, and tell me the
  head. Not during your card window: before the go or after 21:00Z.
- **The go is at 17:20Z**, with the list in the section of 09:4xZ and the lines added since:
  proofs' run on card 7 until 16:54Z at the latest; `CARDS=7` if that card's chain must go again.
