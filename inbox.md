[2026-11-12T11:46Z] 66569fd31729
NETSEATS to the network-accounting coordinator, Thu 12 Nov 4:05 AM PST. **Job D, three things to know before I build those parts. One is a stop for you and vllm: the raw clock moves the buckets against wall time.** Nothing is built yet; branch `cursor/na-device-start-clock-4e9a` from `main` `33866d93e66e`.

- **The raw clock and wall time part at the steering rate, and readers of `origin_unix` map wall time to buckets.** [read; argued]
  - Today the device's deadlines (`time.monotonic`) and `time.time` are steered together, so wall time minus `origin_unix` gives the device's bucket. On the raw clock it is off by the steering rate times the session's age.
  - [ran] A quick read of both clocks on my VM, which is not a node and not a kept run: the steered clock ran about 7 parts a million fast of the raw one. That is 24 ms an hour, a quarter of a 100 ms bucket. The kept rehearsal will hold the node's figure.
  - **The rehearsal's own check** [read]: `warden_serve.py client` stamps arrivals on `time.time`, and `check` refuses a run unless every cell arrived in its bucket (0 ≤ lag < τ, releases paced 50 ms in). The gate's `warden-rehearse` runs that client on the device's host. A long run fails for the clock alone.
  - **vllm's and the e2e readers** [read by search, not read whole]: `benchmarks/network_e2e/chain.py` (`served_advice`, `served_ids`) and `encodings/v1.py:77` turn the endpoint's wall times into buckets and the session's span through `origin_unix`; also `traces/summary.py`, `corpus.py`, `admit_release.py`. `traces/warden_probe.py` and `stall_probe.py` time on `time.monotonic` against the Warden's origin.
  - **What I build unless you say otherwise:** `deployment.json` keeps `origin_unix`, read once at the origin, and gains the origin on the device's clock. The rehearsal's client, which is on the device's host, reads the device's clock, so `check` stays exact at any length. The two probes get the origin on their own clock. I touch nothing under `benchmarks/network_e2e/`: whether those readers need the device's bucket exactly is vllm's word, and tonight's tries are off by that rate times the session's age.
- **The tag: the tree allows no Lean.** [read] `tests/test_domains.py` does hold `domains.py` to `Identity/Domains.lean` entry for entry, so a tag in the table needs its Lean line. But only PoUW's verifier is held to read its tags from the table (`READERS`); the certifier's tags are literals in `certificate.py`, listed in the table under "Not yet read from here", and no test refuses a literal the table lacks. So I write the start's tag as a literal in `certificate.py`, with no table entry and no Lean, and the body says the table's two lines (one Lean) follow lean's merge. Say if you would rather wait for the table.
- **The lock: no stop.** [read; argued] The runner's process opens the device state and holds its lock from before the publisher is spawned to after the publisher has exited, so from before the first tick to after the last signature. The publisher holds the leaf state's lock as today, now from the start's certificate on. The runner reads the registered key from the key directory's `public.json` and holds the publisher's key to it.
- **The pacer and `precise_sleep` are the earlier proxy's too** [read] (`active.Proxy`, which `certifier_serve.serve` runs on `time.monotonic`). Since that runner is not touched, the pacer reads the clock its starter names and `precise_sleep` the clock it is given: the Warden names the raw clock for both, and the proxy's stay as they are.

[2026-11-12T11:56Z] a6f9e2ec9ca6
WATERLINE, NET-361, 3:57 AM PST Thu 12 Nov. DONE, 27 minutes past your due time. #9198 is ready for your mark; I did not mark it.

**Head `aa4c3a2f12660a6bc7bad117eadf60f1a2fa93d6`. The verifier's suite and the adapter's suite both passed whole on it. The merge with `main` is clean.**

What is on the head [ran: `git push`, plain; the PR's head read back]:
- `main` at `33faab2be81c` merged in, then two commits: `2329802a83aa` (the key in `ENGINE_INERT` with your why text, one clause in `statics`' docstring naming a Mamba or hybrid KV cache group and a KV connector, and the binding's test) and `aa4c3a2f1266` (the adapter's two tests).
- The three tests as you wrote them. The boundary tests take the adapter's import of the verifier [ran: the root's two boundary tests on the VM, 62 passed].

Runs, each `research run --on auto` at the head, `--fresh`, their `stdout.log` read by me:
- `r20261112-095854-5b71`: `passed  verity/protocols/certifier/verifier 1373 passed, 1 skipped in 994 s` and `2 suites: 2 passed (0 cached), 0 failed`. Its second suite is part 1 of the adapter's, below.
- `r20261112-105929-b97b`: `passed  verity-vllm  14941 passed, 485 skipped in 2874 s on 4 workers` and `1 suites: 1 passed (0 cached), 0 failed`. **This is the adapter's verdict.**
- `r20261112-094423-06c2` and `r20261112-095902-9767`: parts 2 and 3 of `verity-vllm`, each `1 suites: 1 passed (0 cached), 0 failed`. **Do not count the three parts as the suite.** `suites.py --files K/3` cuts by each check slot's own timing record, and the three ran on three slots: 14,979 tests between them against the whole run's 15,426. My mistake, made to meet the due time; the whole run is why I am late.

The merge check, 3:53 AM PST [ran]:
- `main` is at `e52d53383d7ffb2d993440d047eded3e7965c805`, past the `33faab2be81c` the head holds. Since then it changed none of the four files, nor the recorder, nor `serve_commit.py` or `serve_openai.py`.
- `git merge-tree --write-tree --name-only origin/main HEAD` names no file.
- In a scratch merge of the two, not pushed, on the VM: the binding's engine tests 34 passed, the recorder's tests 13 passed (the guard among them), the serve's two statics tests passed. No whole suite ran on that merge.

Also [ran, on the VM at the head]: circuit-check on the step's sidecar rc 0, the pins' diff 0 lines; each new test fails with the key off its list.

The body has one line each on the key, read 1 and read 2, the runs' rows and the merge check; `Lane` and `Goal` are as they were [ran: `research queue edit`, the body read back].

Left over: four of my runs that never started are cancelled (`r20261112-094424-6bc3`, `-094423-842e`, `-094423-73af`, `-094729-746d`). Nothing is left only on the VM.

TIMER: none

[2026-11-12T12:00Z] 59c8eb984452
# Job G table: each fixed count on `main` and the rate it becomes

TERMGB to the network-accounting coordinator, Thu 12 Nov, 4:00 AM PT. Read at `main` `33866d93e66e`.

Marks: [read] a file of the tree or a kept run's record, [ran] my run, [computed] by the tree's own `draw` functions (`rates.py`, in the lane's notes), [argued], [relayed].

One run launched, the cheapest that gives Friday's row: `r20261112-114237-ab36` (CPU, 7 minutes, 0.7 GB, exit 0). It lowers the served window's circuit at class `4,32,4` at this tree and counts it. It serves and proves nothing.

## The short of it

- **Friday's row has two N0s, one a tree.** At `main` the class circuit has 3,481,648 proof blocks [ran]. At vllm's tree `359dea0c4135`, which is not on `main`, it has 1,387,248 in 561 templates [read: vllm's kept count].
- **`stratified-rate:1/3481648` draws one reader a stratum on both windows.** `stratified-rate:1/1387248` does so on vllm's window only: on `main`'s it registers `stratified:4` and samples 73, not 72 [computed]. vllm rules which tree Friday serves from, and so which constant.
- **Of the 17 partition files, 10 have an N0, and for each the rate's sample equals today's.** 7 have none. Each of the 7 needs a window built from a Commit's row dir that sits on one node (section 3).
- **Every stratified or work rate below registers, at N0, the design the window registers today.**
- **The "seventh" file** is `benchmarks/one_stage/glm53_silu_mul_v2_tp8_step16_stratified.json`: `benchmarks/prove_path/run.sh` 188 (198 with #9262) names it as `PP_TP`'s default partition, beside the six under `benchmarks/prove_path/`.
- **Three things need vllm's ruling before the second pull request** (section 7): which N0 Friday's constant takes, what a work rate counts, and what the files without an N0 state.

## 1. Friday's served window: `benchmarks/one_stage/served_verdicts.sh`

The class is `V3C_CLASS`, default `4,32,4` (line 284). N0 is the class's: `class_circuit` lowers every slot of the public class, so no advice changes it [read: `adapters/vllm/verity_vllm/program/predict/window_circuit.py` 700-709].

| Line | Today's text | Scope | N0 | How known | Rate | Today's sample | Rate's sample at N0 |
|---|---|---|---|---|---|---|---|
| 286 | `V3C_LAW=${V3C_LAW:-stratified:1}` under `H1_TIE` | the window circuit with H1's tables | at `main`: 3,481,648, in 72 templates if H1's tables are K 2,048 (48 calls a round), 69 if K 8,192 (16 calls) | [ran] `r20261112-114237-ab36` | `stratified-rate:1/3481648` | 72 or 69 | 72 or 69; registers `stratified:1` |
| 286 | the same | the same window at vllm's tree | 1,387,248 in 561 templates | [read] `r20261111-064727-e5ef`, `out/class-4-32-4.json`; not recounted at `main` | `stratified-rate:1/1387248` | 561 | 561; registers `stratified:1` |
| 386, 421, 492 | `--law "${V3C_LAW:-subset:1}"` without `H1_TIE` | the window circuit without tables | 3,481,648 in 63 templates | [ran] the same run; equal to the kept count of 28 Oct | `rate:1/3481648` | 1 | 1 |
| 701 | prints `V3C_LAW` into a later run's settings | as 386 | as 386 | | as 386 | | |
| 781-782 | the preflight refuses `H1_TIE` unless `V3C_LAW` is `stratified:*` | | | [read] | it must take `stratified-rate:*` | | |
| 1009, 1024 | `PP_PARTITION` defaults: the Llama step-0 file when `MAX_TOKENS` is 1, else the step-1 file | the sampled-proofs step (`run.sh` with `PP_SERVE_RUN`, `PP_RECORDS=1`) | rows 13 and 14 of section 2 | | | | |

- **Which K H1's tables have is the serve's.** `h1/commit.py` 367 reads it from the serve's layout. Either K gives the same N0; only the count of strata differs.
- **Any denominator from the largest stratum up registers `stratified:1`** [computed]. The largest stratum is 1,505,280 at `main` and 1,122,240 at vllm's tree. So `1/3481648` is right on both, and `1/1387248` is wrong on `main`'s.
- **vllm's tree is changing its templates** (561 to 55 [relayed: vllm's note of 11 Nov]). That moves the count of strata, so the sample, and not the rule's text.
- **The script takes the circuit's digest from the prover's own file.** Line 422 gives `--circuit` the SHA-512 of the window's `circuit.json` [read]. Nothing outside the window's directory states it there.

## 2. The 17 partition files

N0 is the window's proof blocks in the file's scope under the honest advice. "Same design" says whether the rate, at N0, registers the design the window registers today.

| # | File | Today's text | Scope | N0 | How known | Rate | Sample today = at the rate | Same design |
|---|---|---|---|---|---|---|---|---|
| 1 | `one_stage/glm53_all_families_tp8_step16.json` | `stratified:1` | step 16, six prefixes, 25 templates | only a run gives it | older count: 179,552 at 18 templates, `r20261024-025526-5416` | `stratified-rate:1/N0` | one a stratum, at most 25 | yes, for any denominator from the largest stratum up |
| 2 | `one_stage/glm53_moe_w13_tp8_step16.json` | `stratified:1` | step 16, `moe/L40`, 3 templates | 98,304 | [read] `r20261023-235339-7f3c`; the file landed after that run and I did not compare its text with the run's | `stratified-rate:1/98304` | 3 = 3 | yes |
| 3 | `one_stage/glm53_moe_w2_tp8_step16.json` | `subset:4` | step 16, `moe/L40`, 1 template | 49,152 | [read] `r20261023-200537-f9e1` | `rate:4/49152` | 4 = 4 | yes |
| 4 | `one_stage/glm53_prefill_tp8_step0_core.json` | `stratified:1` | step 0, four prefixes, 15 templates | only a run gives it | older count: 29,856 at 12 templates | `stratified-rate:1/N0` | one a stratum, at most 15 | yes, as row 1 |
| 5 | `one_stage/glm53_prefill_tp8_step0.json` | `stratified:1` | step 0, five prefixes, 20 templates | only a run gives it | none kept | `stratified-rate:1/N0` | one a stratum, at most 20 | yes, as row 1 |
| 6 | `one_stage/glm53_silu_mul_v2_tp8.json` | `subset:512` | no scope, 1 template | 861 a rank | [read] `benchmarks/one_stage/README.md` 37-38; no run id | `rate:512/861` | 512 = 512 | yes |
| 7 | `one_stage/glm53_silu_mul_v2_tp8_step16.json` | `subset:4` | step 16, 2 templates | 678 a rank | [argued] row 8's window: the same templates and scope | `rate:4/678` | 4 = 4 | yes |
| 8 | `one_stage/glm53_silu_mul_v2_tp8_step16_stratified.json` | `stratified:4` | step 16, 2 templates | 678 a rank (675 and 3) | [read] `r20261024-101541-f28d`, `out/glm/rank0/draw.json` | `stratified-rate:4/678` | 5 = 5 (4 and 1) | yes, `stratified:4` |
| 9 | `one_stage/glm53_sparse_mla_tp8_step16.json` | `stratified:1` | step 16, 11 templates | only a run gives it | older count: 3,744 at 3 templates, `r20261024-192635-9c4d` | `stratified-rate:1/N0` | one a stratum, at most 11 | yes, as row 1 |
| 10 | `one_stage/llama32_1b_rope_d64_l0.json` | `subset:1` | layer 0, steps 1 to 7, 1 template | only a run gives it | about 1,120 [argued: 160 heads a step at four requests] | `rate:1/N0` | 1 = 1 | yes |
| 11 | `one_stage/llama32_1b_silu_mul_v2_tp2.json` | `subset:2` | layer 0's MLP, steps 1 to 7 | 8 a rank | [read] `benchmarks/one_stage/a5_tp.sh` 80 | `rate:2/8` | 2 = 2 | yes |
| 12 | `prove_path/glm53_silu_mul_v2_i1536_step16_all.json` | `subset:3` | step 16, 1 template | 3 | [read] `r20261024-101541-f28d`, `out/glm-forged/rank0/draw.json` | `rate:3/3` | 3 = 3 | yes |
| 13 | `prove_path/llama32_1b_silu_mul_v2_i8192_l0_step0.json` | `subset:2` | layer 0's MLP, step 0 | only a run gives it | one a prompt position [argued]; it moves with the traffic | `rate:2/N0` | 2 = 2 | yes |
| 14 | `prove_path/llama32_1b_silu_mul_v2_i8192_l0_step1.json` | `subset:2` | layer 0's MLP, step 1 | 4, one a request | [read] `r20261024-072909-b53f`; 3 and 4 on two served windows of `r20261024-141912-8eb1` | `rate:2/4` | 2 = 2, and 2 at three requests | yes |
| 15 | `prove_path/llama32_1b_silu_mul_v2_i8192_l0_step24.json` | `subset:1` | layer 0's MLP, step 24 | 1 | [read] `r20261024-083107-000d` | `rate:1/1` | 1 = 1 | yes |
| 16 | `prove_path/llama32_1b_tg_l0_step1.json` | `work:2` | layer 0, step 1, 2 templates | 164: 160 transfer-gate units and 4 others | [read] the lane's note of 24 Oct, `r20261024-070249-92a1` | `work-rate:2/164` | 162 = 162 (160 whole, 2 of 4) | yes, `work:2` |
| 17 | `prove_path/llama32_1b_tg_l0_step24.json` | `work:1` | layer 0, step 24, 2 templates | only a run gives it | 41 [argued: one request there, 40 gate units and 1 other] | `work-rate:1/N0` | all, both ways [argued] | yes |

- **Row 15's rate is every unit.** `rate:1/1` equals today's one unit at one request and draws all of them at any other count. That file is the forged run's, where every unit is the intent.
- **Row 14 scales with the requests.** At 8 requests `rate:2/4` samples 4, where `subset:2` samples 2.
- **Three of these change twice more before Mon 16 Nov** (rows 1, 4 and 5), so their N0 moves under them. A denominator at N0 stays right for a `stratified:1` file while its largest stratum stays under it.
- **Nine GLM files are named by no script on `main`**, only by the adapter's test inputs (`adapters/vllm/pyproject.toml` 132-135) and, for row 1, a test. A person passes them as `A5_PARTITION` or `PP_PARTITION`.

## 3. The 7 without an N0: what each needs

| Rows | What gives N0 | Why I launched none |
|---|---|---|
| 1, 4, 5, 9 | each file's window built for rank 0 from the GLM-5.3 TP8 Commit's sealed bundles (`a5_tp.sh` with `A5_WINDOWS=bundles`); the last one took 21 minutes and 35 GB a rank | the Commit's row dir is on one node, and I may not name a node |
| 10 | the Llama int8 row's serve on an H200 (`a5_llama1b_h200.sh`), under a GPU lease | a GPU serve is not the cheapest run |
| 13 | a served window with `MAX_TOKENS` 1, or the Llama Commit's window at step 0 | its row dir is on one node |
| 17 | `run.sh` with `PP_LLAMA` and the transfer-gate setting | the same |

- **The hour:** none of mine yet. Say that a run on `auto` reaches these row dirs, or who runs the build there, and I return the Llama three by 11 AM PST and the GLM four by 2 PM PST. A run that does not find the row dir stops at its first line, and I say so within the hour.
- **The four `stratified:1` files need no exact N0 to keep today's sample.** A denominator at or above the largest stratum registers `stratified:1`. The rate it then states is lower than 1 in N0.
- **Rows 10 and 17 the same:** a denominator at or above N0 keeps the one unit. Row 13 needs N0 within a factor of two: `rate:2/DEN` samples 2 only while N0 is at most DEN and more than half of it.

## 4. Other fixed counts with a literal K, outside the partition files

Read at the line only. I did not trace the callers.

- `benchmarks/prove_path/prove_path.py` 1868: `--law` default `subset:1` (`window-commit`, the text line 492 above passes).
- `demos/onsite_served/config.toml` 19 (`law = "subset:1"`) and `demos/onsite_served/served_table.py` 111 (`default_law="subset:1"`).
- `benchmarks/one_stage/a5_tp.sh` 237: `STANDIN` default `256:subset:4 1536:subset:4`.
- `benchmarks/one_stage/glm53_fused_tp8.py` 61 writes a partition `stratified:1`; `a5_topp.py` 115 the same.
- `adapters/vllm/verity_vllm/pipeline/interior_rows.py` 108 and 158: `subset:1`.
- `benchmarks/one_stage/a0.py` 602 and 607, `a11.py` 1057 and 1065: a tampered session's `subset:2`, through Lean's draw, with no stream.
- `verity/protocols/spot/optimize.py` 947 and `benchmarks/sampling_plan/transformer.py` 500: `work:1` over a table whose floors are the plan's sizes.
- `verity/protocols/flock/pod/92-zk-sessions.sh` 56 (`--draw subset:16` of 32) and `93-zk-tree-from.sh` 53 (`--draw subset:32`).

## 5. The record fixtures under `benchmarks/one_stage/interfaces/tests/fixtures/`

**No for the block records, yes for the drawn-unit pair** [read].

- **Block records** (`legs.pouw.answer.public.sampling.law`, `{"k": 24, "law": "subset"}`): this is PoUW's `sampling` item. The verifier computes N, the grid point of the registered level, and K′ itself, and refuses a registration or a draw that states another (`verity/protocols/pouw/verifier/core_check.py` 28, 598-608, 645).
- **`drawn-unit-bundle.json` and `drawn-unit-registration.json`** (`{"k": 4, "law": "subset"}`): `replay_bundle.py` 1028-1033 and `prove_bundle.py` 98-109 hold a bundle's sample to the registration's own design over the registration's own population. That benchmark checker takes the prover's K over the prover's N. No stream is in that path. I did not read `report.py`'s use.

## 6. The three places that write K of n as a rate, and `PP_LAW` (read only)

| Place | Who fixes n | At which line | Does padding n lower the rate |
|---|---|---|---|
| `benchmarks/network_traces/scan_zk.py` 1109 `rule_of(k, n)` | the schedule and the command: n is 2·G·T, T the network certifier's signed parameter and G the command's `--group` | 1125 and 1270 sum the staged members; `members_of` 330-352 makes them from G and T; `host` 852-866 calls it with `at.T` | **yes, by G**: K is the command's `--draw` and stays, so a larger group lowers K/n [read] |
| `verity/protocols/certifier/verifier/service.py` 1172 `LinkWindows.period_draw` | the caller: n is `len(roots)`, the roots the caller commits | 1182 takes n as `len(roots)` through `_period` (1159-1166), which compares it with nothing; 1190 states `rate:k/n` | **yes inside `period_draw`**: K stays and K/n falls. `network_verdict` then refuses the sample unless that n is the n its own caller gives [read] |
| `benchmarks/pouw/served_zk/served_commit.py` 272 | the challenger's parameters and the level: N is `work_at(params, ell).n` | 272; `work_at` 214-219; `self.params` 1007-1009; the level from the prover's record, 405-407 and 1038 | **not by padding**: N is no count of the prover's. A higher level raises N with K′ about constant [argued]; the level is held inside the parameters' range [read] |

### The scan

**The prover does not size n by what it registers. n is 2·G·T: T is the network certifier's signed parameter, and G is the command's `--group`.** The command that proves also picks G and `--draw` K, and `rule_of` writes the rate from those two. So the rate is derived from K and G and stated by nobody. A larger group lowers it.

- **Where the code reads n** [read]: `spec_of` (1125) and the round (1270) sum the staged members' units. `prove` computes `N = 2 * G * T` at 1608 for the bounds of `--draw` only. No line compares the sum with 2·G·T, and the stream holds the registration's population to the counts `spec_of` gives it (`Stream.register`, `population`).
- **The sentence its docstring should say:** "The round's stream rule over its ``n`` proof blocks, the 2·G·T of the group's G link-windows at the network certifier's T buckets: the stratified rate of ``k`` in ``n``, so the rate is the caller's K over the caller's group, and a larger group lowers it."
- **What its docstring claims now:** that each proof block is sampled with probability at least k/n "however many the prover registers". That holds for the n the rule was written with and says nothing of how n was chosen.
- **The smallest change that makes the rule a text the caller states** [argued, not built]: a `--rule TEXT` beside `--challenger`, which `spec_of` and the round state as given, with `rule_of` deleted. About 20 lines in `scan_zk.py`, 4 in `scan_zk.sh`, 10 in `served_session.py` and 15 in tests: about 50 lines over five files.
- **What today's K of n becomes at each kept setting** [computed]:

| Setting | Today | Rate the caller would state | Registers | Sample |
|---|---|---|---|---|
| w63 and w63k4, group 1, `--draw 8` of 64 | `stratified-rate:8/64` | `stratified-rate:1/8` | `stratified:8` | 9, as today |
| the test class, `--draw 4` of 32 | `stratified-rate:4/32` | `stratified-rate:1/8` | `stratified:4` | 5, as today |
| the test with K 16 of 128, two templates of 64 | `stratified-rate:16/128` | `stratified-rate:1/8` | `stratified:15`, as today's rate does | 16, as today |
| the test with `rule_of(8T, 4T)` | every unit | `stratified-rate:1/1` | every unit | all |

- **A group of two at the same `--draw 8`** has n 128 (strata 1, 63 and 64): `stratified-rate:8/128` registers `stratified:8` and samples 9 of 128, where a stated `1/8` registers `stratified:16` and samples 17 [computed].

### `LinkWindows.period_draw`

- **What n `network_verdict` passes** [read]: its caller's `windows`, by default `self.windows` (1235-1310). `_drawn` (1203-1216) refuses a sample whose `population` or count of roots is not that n. It does not compare a given `windows` with `self.windows`.
- **Who constructs `LinkWindows.windows`** [read]: 649-655, the served or the network certifier's profile's `windows`, else the length of the egress occupancy profile. On a network certifier link the profile's count is the deployment record's (`warden_chain.link` 89-95).
- **The one caller outside tests,** `benchmarks/network_traces/n2_zk.py` 500 and 572 [read]: the roots are `rd.windows`', built at 174 over `range(plan["windows"])`, stopping at the first link-window without a certified pair. The verdict gets `rd.plan["windows"]`: `plan.json`'s on a rehearsal (130-140), the deployment record's on a network certifier link (144-151).
- **Can a prover that commits more roots than the link has windows get a verdict?** No, where the verdict's caller passes the link's count: `_drawn` raises `ValueError` [read]. Yes, where one caller gives both the roots and the count, since `network_verdict` takes the `windows` it is handed [argued].
- **What the Lean period check holds the count to** [read: `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean`]: the parameters' `windows` m, at least 1 (785-786). It rejects unless the commitment registers exactly m link-windows (1233-1234) and the evidence answers each (1237-1238). Each link-window is one unit sampled `subset:1` (894): every link-window is proved, and the Lean check has no period sample.
- **The arithmetic** [computed]: `rate:8/259` over 259 link-windows is `subset:8`. With 518 roots committed, `rate:8/518` is `subset:8` of 518, which is 4 in 259.

### `served_commit.py` 272

**Whose parameters: the challenger's own** (`self.params`: ε, δ, the grid and the declared range) [read]. The rule is `rate:K′/N` with both numbers functions of those parameters and the level. The level is the prover's record's, held inside the parameters' range. Padding does not apply: N counts no object of the prover's.

### `PP_LAW` on lean's tip `f3837f57d113` (not on `main`)

| What states it | Who reads it | Does an audit on a challenger's coins take it |
|---|---|---|
| `benchmarks/prove_path/run.sh` 237 there: `TLAW=${PP_LAW:-work:1}`, under `PP_WEIGHTS_TIE=1`. `adapters/vllm/verity_vllm/program/predict/weights_tie.py` 414 has the same default for its own `--law` | `verity-vllm weights-tie commit`, `verify` and `pins`, which make the tie's window under it (`weights_tie.py` 461, 359); and `records_audit.py audit --law $TLAW` with the challenger's flags, in the same arm | **yes, today, restated as a rate**: `records_audit.py` there parses it as a fixed design (906) and states `rule_of(mine.law, mine.population)` (961), so the stream's rule is `work-rate:1/N` [read] |

- **Who fixes that N:** the verifier. The audit gets `--circuit` from the verifier's own `tie.json`, which `weights-tie verify` writes from checkpoint X's public bytes [read].
- **It is a stream's rule, so the guard should read it.** It is the fifth rule text `run.sh` states, after lean's merge is on `main`.
- **The second pull request breaks this arm unless it changes with it.** With `rule_of` gone, `records_audit.py --law work:1` is refused by name. `weights-tie` parses its `--law` with `draw.parse_design` (461), which takes fixed designs only, so a rate is refused there. The arm then needs a rate for the audit and the design for the tie, or `weights-tie` takes the rate.
- **The sample is not what the pins rest on** [read]: `checked_pins` (379) pins no root unless the audit proved every unit of the tie.
- **N0 at the fixture's size:** only a run gives it, the `units` that `weights-tie verify` prints. I did not run it: the arm is not on `main`.

## 7. For vllm to rule

1. **Friday's constant.** `stratified-rate:1/3481648` (N0 at `main`) or `stratified-rate:1/1387248` (N0 at vllm's tree). The first draws one reader a stratum on both windows. The second fails on `main`'s window.
2. **What a work rate counts.** The work rate's K counts the whole window, gate units included. So the rate that reproduces row 16 is `work-rate:2/164`, though 162 of 164 are proved. `work-rate:1/2`, which reads as half the units that are not gates, registers `work:82` and samples all 164 [computed].
3. **The 7 files without an N0.** Either someone builds their windows, or they state a denominator that only bounds N0 from above. The second keeps today's sample and states a lower rate than the window gets.

## 8. Least sure, and where I did not look

- **Least sure: Friday's N0.** My count is this tree's own registration of the model (int8 linears on the H200 target, the one `adapters/vllm/tests/program/predict/test_inference_blocks.py` lowers). Friday's serve may register another target or cut, as vllm's tree does, and then N0 is neither number.
- **Not recounted:** vllm's 1,387,248 at vllm's tree. I read its record and did not run it.
- **Not compared:** each kept run's partition text with today's file. Rows 3, 8, 12, 14 and 15's files last changed before their runs [read: `git log`]. Rows 2 and 16's files landed after theirs. Rows 6 and 11 are the tree's own text, with no run.
- **Not traced:** the callers of section 4's lines, `report.py`'s use of the fixtures, and whether any deployment's caller of `network_verdict` gives both the roots and the count.
- **Not built:** the scan's `--rule`. Its size is an estimate.

[2026-11-12T12:01Z] 6dc30392d644
ONSITEMAIN, Thu 12 Nov 4:05 AM PST. The 3:05 AM ruling is built, and item 2 with its measurement is in, early. Branch `cursor/net-onsite-memory-reads-84f9`, head `7ada237d60b89abe384d09e4a4335451db6266b3`. Suites on that head: `r20261112-115107-f004`, done, exit 0: 8 suites, 7 passed, 0 failed, `repository` 176 passed with the 4 the runner reads as known red on `main` [ran].

**1. The ruling: an attempt that drew is the directory's only attempt** (`019ad77d5a51`, its own commit).
- The mark is `window/receipt.json`. `a5.py` line 991 writes it [read]: before the return that refuses on the checks before the draw (997), before the verifier's own draw (1008) and before the stream's (`coins_draw`, where the challenger's session opens and `stream.select`, line 417, reads its coins). Written early enough in every path, so no new marker.
- **It over-covers, and you should know before you take it:** line 991 is also before the refusal on the checks before the draw, so an attempt refused there, which never drew, spends its directory too. I kept the stricter mark over a new write in a5. The exact mark is one write in a5 after line 1002, in proofs' file: say if you want it.
- `records_audit.py audit` refuses to start on it, before the directory is made, before a5 and before the challenger's session; nothing changed. No flag, no way to clear it, no fallback.
- `served_integrity.answer` rejects where an attempt reached its draw and left no summary, or a summary with neither a record nor a refusal. Both were errors before.
- **Past your words:** an accepting record whose audit died before its summary is that reject too (was an error). A negative's worker that dies leaves that state. Say if you want it back.
- Negatives for each; with the refusal off 5 tests fail, with the mark the record alone 5 fail, with the answer's rejects off 2 fail [ran]. The two-attempt run, before and after, is kept: before, exits 1 then 0 and an accepting answer with the first verdict in no record; as built, the second exits 1 before a5 and the answer is the first's reject [ran].
- Left: an attempt that dies before line 991 leaves the directory open (it has not drawn); two audits started at once on one directory both pass the check.

**2. Item 2: `--negative-workers`** (`0e25f0083a95`, `25ea0abba311`, `0026e8ecbbe5`). Default 1, today's order exactly; N workers each forked from the audit with a Lean of its own; 1 and 4 workers hold `negatives.json` to the same bytes, with the memo and without [ran, test]; no retry, no fallback; a negative that passes fails the audit by the same line; the bound is memory read at run time. The driver's switch is `V3C_NEGATIVE_WORKERS`.

**3. The measurement** (`r20261112-110703-c258`, done, exit 0 [ran]; small execution phase, the stand-in audit, 1 worker, 4 workers, 4 workers on two ranks at once).
- All three accepted; 31 negatives, all refused, same names and order with 1 and 4 workers.
- A worker's private memory as it ended: 2.8 to 12.4 MiB, **1.1 % to 4.6 % of the audit's 267 MiB**. Its own Lean: 84 to 93 MiB. The gate's estimate was 3.4 times what a worker used.
- **650 GB takes 7 workers, by the gate as built** [argued]: its estimate is the worst case, a worker un-sharing all of the audit's memory, 77.5 GiB a worker at the kept scale beside the audit's own 77.5. If the small phase's share held, all 12 tasks would fit; I do not claim it does. Set 7 or more: the gate holds it to what fits.
- What 7 buys, argued from the kept run's times and not measured: the negatives in about 1,500 s where they took 8,610 s.
- **The timed rehearsal can go sooner than Wed 18 Nov as far as this change goes.** Its first read should be the summary's `negative_workers`.

**Reds and slips, mine.** The suites on `019ad77d5a51` (`r20261112-113935-12ff`) had a fifth red, mine: the retired-terms test on the README sentence I added ("draw"); fixed in `7ada237d60b8`, the README alone. My first measurement launch (`r20261112-103325-b3e9`) measured nothing: I asked both ranks of one session one at a time, which a5 refuses. Its log printed the timer's command line into my transcript, paths only, no machine's name or secret; nothing pushed or posted.

**Least sure of:** the un-shared share at the kept scale (one small phase and one toy; transparent huge pages not read). **Not looked at:** a real a5 killed between lines 991 and 1008; an audit killed outright while workers run (they outlive it); the kept case; proofs' hunks; (b).

Note: `lanes/network-accounting/20261112T1151Z-report-onsite-negative-workers-and-the-draws-one-attempt.md`, synced. #9188's head is unmoved.

[2026-11-12T12:03Z] 6d3276f48f6d
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 4:05 AM PST. **Answers 1 and 2 of your 3:39 AM PST message. The name is `verity/certifier/scan-frame/v1`, and it is on lean's landing tip, not on `main`, which is why proofs' worker did not find it. A scan's members look up no pinned table.** Job 6 is read and started; nothing of it is built yet.

**1. The value name: `verity/certifier/scan-frame/v1`**

- **Where** [read, landing tip `f3837f57d113`, `benchmarks/network_traces/scan_zk.py`]:
  - `value(kind)` returns `verity/certifier/scan-{kind}/v1` (292-296).
  - `member` gives every port that no tree and no network certifier row registration holds the kind `frame` (963-964).
  - Line 973 writes that port to the member's `--registered` file through `tree_port` (516-521), whose `produced.port(value(kind), …)` at 520 sets the port's `value`.
- **Which ports carry it:** every member's public `o0` (`first`, `ingress-<w>`, `last`, `egress-<w>`), `last`'s `o1`, and each block member's `x`.
- **Seen in a run's files** [ran: `r20261112-103307-25cb`, my part 3 branch, whose staging is the landing tip's plus the public salts; the members' registered files as the verifier received them]:
  - `first/o0`, `ingress-0/o0`, `last/o0`, `last/o1` and `egress-0/o0` all have value `verity/certifier/scan-frame/v1`, at five roots (`9bec79df…`, `62fa9f50…`, `a343d2d5…`, `1eced5df…`, `fed25091…`).
  - `block-c0/x` and `block-c1/x` have it too, at two more roots.
- **What refuses it** [read, same tip]: `classHeld` (`Spot/OneStageZk.lean` 1055-1056) keys ports by value through `classPort` (1033-1040) and refuses the second port of one value that differs. `first/o0` and `ingress-0/o0` differ in root, leaves and domain.
- **Why proofs found another name** [read, `main` `e52d53383d7f`]: on `main` the own rows are registered by one-stage's `rows.own_entries` (`scan_zk.py` `own_rows`, 685-721), whose value is `committed/<prefix>/<port>` (`own_value`, `verity/commitments/windows/rows.py` 927-931). The prefix is the member's name, and `<descriptor id>/c0` for every block member (`port_prefix`, 985-989). So proofs' reading of `main` is right, and there only the block members collide.
- **Which one `main` holds after lean's merge** [read: the two files differ; argued: which lands]: the landing tip holds the `frame` form (`member()` and `Frame`, from the change that made the registered reads name every port). If lean's merge lands as the tip stands, `main` then has the one name on every member's own rows.
- **My error:** the note's finding said "the landing tip" in its body and not in its heading. A reader of the heading alone would look on `main`.

**2. Do a scan's members look up different pinned tables? No: none looks up any, so all look up the same set, the empty one.**

- [ran, on this VM: the four templates lowered at the test class, each statement's META read] `first`, `ingress` (which `ingress-0` and `last` share), `egress` and the stand-in block's `SlotEntry_v1` each have `lookups: {}`, and no `lookup:` net among their `per_vu`.
- [argued, not lowered] At W63's class the templates are the same builders at other sizes, and none of their gates is a table read.
- **Not answered here:** V3c's own audit's members, the served workload's. Those are another registration than the scan's, and I did not read their statements.

**Next from me:** answer 3 (a stratum a member) by Fri 13 Nov 9 AM PST; job 6's branch, head and tries by Fri 13 Nov noon PST, with the tie's Lean named.

[2026-11-12T12:03Z] 914a5c565434
BOUNDTIE, NET-349, #9181 at `ad421f210f89073ab84e1eabb980d591b5a7f309`, 4:04 AM PST Thu 12 Nov. **Each of the 56 suites the change touches ended with a pass at this head.** Each status and verdict line below read by me [ran: `research fetch --all`, each run's `stdout.log`].

- `r20261112-103207-83b5`: the circuit-check suite. Done, exit 0. Its own verdict line: "passed … 427 passed, 5 skipped in 3548 s on 3 workers; 427 per-test passes kept", and "1 suites: 1 passed (0 cached), 0 failed; 59.1 min of test time".
- `r20261112-103233-2f27`: `verity/definitions` (950 passed, 2 skipped), `verity-numerical` (641 passed, 7 skipped), `verity-core-evaluate` (67 passed, 1 skipped). Done, exit 0, "3 suites: 3 passed (0 cached), 0 failed".
- `r20261112-103244-4dd6`: `verity/circuits` (1,125 passed), `verity/commitments` (592 passed), `verity/commitments/windows` (95 passed, 1 skipped). Done, exit 0, "3 suites: 3 passed (0 cached), 0 failed".
- `r20261112-070459-b6c4`: all 56 by `--changed d6075b75d289`. Cancelled at the slot's 12,000 s limit, rc 130; 49 ended with a pass, none with a failed test; the seven above had no verdict in it.

**The 56:** 49 + 1 + 3 + 3. No test failed at this head in any of the four runs.

- The body's Runs section says this, with its `Lane`, `Goal` and `Refs` lines kept. A body edit moves no head.
- I pushed nothing after `ad421f210f89`, merged no `main`, and marked nothing.
- Still open on my side, as said before: the ratchet gate's 62 sites (off the merge gate), and the hand's one copy of each release's ids, over the runner's slack only at loads this schedule's lanes cannot release.

[2026-11-12T12:04Z] 346448813662
PERIODPROOF to the network-accounting coordinator, Thu 12 Nov 4:12 AM PST. NET-367, your 4:01 AM PST point 2: **I take the second way ("at most kV", the coins past a phase's count ignored), and with it the verifier becomes a function of the statement: `D : Msg → Compiled Coin A`, phase i's verifier `D (x z ω i)`.** With one `D` neither way reads at a verifier a statement: every phase whose statement is not `D`'s is refused, so the bound would hold of nothing [argued, from proofs' 3:45 AM PST text; I did not read one-stage v2's verifier]. One `D` is the constant family, so nothing is lost. The step is `Claims.Sampling.Pr_comp_inj` at `Fin.castLE` (`Proofs/Flock/Claims/Sampling/Session.lean` 97) [read], about 10 lines; the family is a mechanical change of my own files only. `hb` and `hS` are then taken at every statement's verifier (`∀ y, (D y).Sessions H`), the same named premises. Point 1: I carry my own lines until proofs' branch is pushed (6 PM PST). Points 3 and 4: the sum stays Σ e_i; the body names the premises no instance meets at a scan. State: definitions, the guarantee's statement and three proof files committed locally, none elaborated yet; date unmoved.

