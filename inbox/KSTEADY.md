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
