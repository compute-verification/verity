
05:45Z Tue (coordinator), compute-research on the crux: **no registered Definition reproduces a Pearl+ tile yet**; Pearl-C's C-Flock statement leaves its TileRead session a parameter until one lands. The pieces exist in ml.circuits.silicon, exact: HOPPER_E4M3_WGMMA_K32 (the chain) and HOPPER_BF16_WGMMA_K16 (the peel). **TileRead (Pearl+'s chain plus peel at pinned roundings) is the missing circuit.** **Fallback for Tue: sampled proofs prove the stock FP8 linears' Definitions, and Pearl+'s tiles are checked by its reference verifier.** So check first that Pearl+'s served y is bit-for-bit stock FP8's (the same cleaned product), which makes that fallback sound for the served values. If it isn't, say so at once.

05:47Z Tue (coordinator), **circuits overrides the fallback** (TOP-159): Pearl+'s y is **not** any FP8 Definition's output, so proving stock FP8 linears would not cover the served values. Its cleaned product U is exactly PoUW's pc8 units, and y adds a small unscale. **circuits writes that Definition (pc8 plus unscale) and sizes pc8 at Llama 1B by Tue 10 AM PST.** For the Tue 4 PM run, circuits recommends that **sampled proofs sample only the non-linear blocks, with the linears covered by PoUW** (Pearl+'s reference verifier). Plan the run that way: a sampling design that excludes the linears' blocks. Name how a non-linear block's input y is tied to Pearl+'s committed U (the unscale) so the record doesn't overclaim, and say what's unbound until circuits' Definition lands.

05:50Z Tue (coordinator), circuits' correction: pc8's units take Pearl+'s operations as primitives, so no gate-level circuit reproduces a tile yet. **TileRead** (the chain plus the peel, from the two silicon atoms) is the missing Definition, and **circuits builds it: a scoped estimate and a first working cut by Tue 10 AM PST.** The Tue 4 PM plan stands: sample only the non-linear blocks, and the linears are covered by PoUW.

06:05Z Tue (coordinator): **vy-dc-h200-2 stays up continuously to Thu 5 Nov 02:00Z** (lease 01:55Z); no stop after f55c, and it's yours for the Pearl+ bring-up from f55c's end (~06:30Z). kernels' step kernels/pearlplus-gate-h200 (Pearl+ gated and timed on one H200, due Tue 12 PM PST, INF-439) also needs an H200. Share vy-dc-h200-2 with kernels through gpu-lease, and reuse their gate and per-linear timing as your bring-up's step rather than duplicating it.

06:05Z Tue (coordinator), kernels' first Pearl+ number (TOP-159): **bit-exact at all four Llama 3.2 1B linears**, but **5-10x slower than FP8 per linear and 14x per decode layer**, mostly hashing. kernels works the hashing on its own H200 Tue 6-10 AM PST. They briefly used vy-dc-h200-2's free GPU. Reuse their gate and numbers rather than re-measuring per linear: your bring-up's job is the served vLLM run (tokens/s against stock FP8 end to end, output quality, the verifier on retained passes) and the profile of the served step beyond the linears.

06:30Z Tue (coordinator), compute-research's PLUS-UTIL profile (TOP-159; sm_120, Llama 1B): Pearl+ decode step **3.2x** stock, mixed **6.2x**; hashing is **53%** of decode GPU time. Declared utilization **0.28%** at decode against stock's own **0.79%**, so batch size and load matter as much as hashing. **vllm's lever: a big batch plus CUDA graphs for prefill (~26% declared)**; kernels' levers are a shallower row commit and the prefill commit grid. Make the served run's load prefill-heavy with a large batch (as compute-accounting's settings: 1,024-id prompts, max_tokens 1, 8 a step at T 8192), take CUDA graphs where Pearl+'s whole-step capture allows (e2e.py --whole-step), and report declared and credited utilization from served_shares.py. H1's T 512 window registration doesn't bind a Pearl+ run.

06:50Z Tue (coordinator), answers to your 05:50Z note:
- **The split is confirmed:** you take the served e2e (vLLM, whole model, verify_run, quality, served-step profile). kernels takes per-linear timing and the hashing profile.
- **H200:** vy-dc-h200-2 stays up to Thu 02:00Z, but f55c holds its GPU lease until 11:08Z at the latest. I asked infra to **wake vy-dc-h200 now** (instead of Tue 23:30Z) for your bring-up. Use whichever comes first.
- **Ruling (i) or (ii): (ii).** Cut the circuit at the linears, so each linear's y is a **registered value** of the run's record (committed under hiding leaves, read by the next non-linear block), never a public-input port, which the verifier refuses. The linears' sites carry no Definition, so no proof block is drawn there, and PoUW covers them. (i) would describe BF16 linears that aren't what's served. Lift `protocols.refusals` only for a sampling design that draws no block at an executor's site, with a test that it still refuses one that does. circuits may object by Tue 8 AM PST (asked).
- **The interim Python tie: yes, for Tue if it fits.** At each drawn tile, verify_run.py also opens the tile's rows of served y and holds them to unscale(U) bit for bit, and opens the call's A rows and holds them to PoUW's A rows. That needs the Commit to keep linear inputs and outputs at row grain, with openings kept. Size it after bring-up. If it's under ~4 h, do it.
- **Record wording:** yours stands. The linears are replay-verified on drawn tiles (a development shortcut); the sampled non-linear blocks are proved on their committed inputs; the served values between them are bound to U only by the interim Python tie (or not bound, if it's not built). Never claim a served token is proved.

06:55Z Tue (coordinator): **circuits accepts (ii)**: it keeps every value under a hiding leaf. Their notes: (1) the record says the registered y isn't yet tied to PoUW's credited rows (or names the interim Python tie, if you build it); (2) keep the cut points stable, so TileRead later fills each linear site; (3) training confirms that inference-only's counts read a circuit with the linears cut out. I'm asking training.

06:45Z Tue (coordinator): **vy-dc-h200 is up now** (one H200, idle, deadline Thu 5 Nov 02:00Z, lease 01:55Z). Start Pearl+'s served bring-up there now. Run 1 is still planned on vy-dc-h200 at ~Tue 23:45Z, so finish or hand off by ~23:00Z, or tell me if run 1 should move to vy-dc-h200-2 (free after f55c, by 11:08Z at the latest).

06:55Z Tue (coordinator), training on inference-only over a circuit cut at the linears (TOP-159): **each cut linear needs its own count.** Its gates go into its session's block by shape, and c is calibrated the same way. Its y counts as internal input only where the record ties y to PoUW's credited product for that session. Where it's untied, the verdict says so plainly. So the cut must carry each linear site's shape (m, k, n and the scheme) in the registered circuit, for the counts to add. pearlplus keeps that in the cut; ibcounts' tool adds the cut linears' gates by shape and marks y tied or untied. Read training's details on TOP-159.

07:25Z Tue (coordinator): **TileRead is #7066** (circuits; cursor/circuits-pearl-tileread-21f3 at 3def12b945, base the run tree; circuit-check 14/0): PearlCChain, PearlCTileReadCell, PearlCTileRead and **PearlCUnscale** (y = RN_bf16(RN_f32(RN_f32(U*RN_f32(1/alpha_A))*RN_f32(1/alpha_B)))), word and Boolean. circuits lands it on the run tree; take that head. **Circuits' gap fix, which replaces the interim Python tie: draw PearlCUnscale blocks over committed U and alpha at each linear site**, so the served y is bound to U by proof. U itself stays covered by PoUW's verifier on drawn tiles; TileRead can later make U provable too. With that, the record's wording changes from 'y not bound to U' to 'y bound to U by sampled unscale blocks'. Keep the cut points stable, and give each site its shape for training's counts.

07:30Z Tue (coordinator), correction to 07:25Z: #7066 is NOT being merged into the run tree (neither circuits' agents nor vllm's merge PRs). **Merge its head commit 3def12b945 (cursor/circuits-pearl-tileread-21f3) into your own branch** (git merge of the commit) and leave #7066 open. circuits approves it as owner at that head.

07:35Z Tue (coordinator): **the integration branch `remote` now exists** at 3def12b945: the run tree ea3f9c5635 plus circuits' #7066 (TileRead, PearlCUnscale), a fast-forward. It is Daniel's one branch for the remote protocol. Reviewed heads get a **plain git merge into `remote`** (never into a PR's own base, which would mark that PR merged), and the PRs stay open as the review record. Merge `remote` into your branch to stay current. Only the coordinator, or a worker the coordinator names, pushes to `remote`; push your work to your own branch and say in your report which head is ready to integrate. Run 1 launches from `remote`.

08:22Z Tue (coordinator): **`remote` is now 19abfa44cf**, with kernels' #7075 integrated: Llama 3.2 1B in the sm_90a H200 session (`h200.sh MODEL=llama32_1b`, `glm53_timing.py --model`), cuBLASLt FP8 baselines, the pearlc-served arm, the prefill split under deferred-a, and the clock lock (LOCK_MHZ). Bytes are unchanged. Merge `remote` into your branch and build on kernels' harness rather than duplicating it. Two levers would change the commitment: a tile leaf over U alone, and a shorter row-seed chain. Those are compute-accounting's call; don't use them unless they rule.

08:40Z Tue (coordinator), compute-accounting (TOP-159, 12:35 AM PST):
- **Pearl+'s PoUW verdict now covers the served y.** Each call's real rows of y are committed as hm96 rows under the run commitment, and on every accepted tile the reference verifier opens them and checks each word against unscale(U). Its PR is #7076. I asked for a narrow head on `remote` to integrate; don't merge #7076's branch, which carries 3,969 unrelated files.
- **On kernels' synthetic H200 passes:** REJECT only where a row's debit is over its cap, which the 3-draw screen credits. kernels drops those rows.
- **Modeled utilization:** Pearl+ about 6.0% declared and 5.5% credited (stock int8 31.2%, H1 1.4%). Hashing is ~69% of the step.

**Their four asks of you:**
1. a served transcript directory (pass-NNNNN plus `coins.json`);
2. `POUW_CMD` wired to `served_passes.py verify --passes <dir> --out "$3/verify.json"`;
3. y staged on the `--whole-step` and grouped paths, which don't commit it yet;
4. the y root tied to the block input that sampled proofs commit (training's tie). With circuits' PearlCUnscale blocks sampled at the linear sites, that ties y to U by proof as well.

08:45Z Tue (coordinator): **`remote` is now 08d5d36ee0**, with compute-accounting's Pearl+ verdict covering y. served_y commits each call's real rows of y on retained passes; verify_run checks them against unscale(U) on drawn tiles, then runs served_passes and served_shares at Pearl+'s credit. Merge `remote` into your branch.

09:45Z Tue (coordinator): **`remote` is now 6f86e8412c**, with training's #7083 merged: the inference-only verdict step, ibcounts' block-counts, and tiee2e's run-2 tree (the H1 tie A-E and the ok-salt fix). Merge `remote` into your branch. **For the 4 PM run (training):** set IO_CLASS, REPO, REV and LEASE_MIN as served, and have `lake` on the served node. The verdict runs in served_verdicts.sh step 5, beside sampled proofs. Rehearsal 2 accepts retroactively.

10:20Z Tue (coordinator): **`remote` is now 4bd367fc06**, with kernels' debit-over-cap fix (volunteer_pass gives up a pass's rows over a debit cap before the sampling coin; decode and prefill ACCEPT on 459 draws; the control REJECTs). Merge `remote`. The pouw suite has 4 failures from compute-accounting's served-y commits; compute-accounting fixes them.

10:30Z Tue (coordinator): **`remote` is now 44b84e9a57**; merge it. Inherited reds that touch your path: served_y isn't a public interface (test_vllm_public_api), an E402 count in test_pearl_c_vllm.py, and four Pearl+ tests in benchmarks/pouw (compute-accounting's PEARLREMOTE is on those). **Live coins:** the run's sampled proofs must take the live challenger (PP_CHALLENGER=gate:...), not the local development session that's now the default. Also, for the Pearl+ tie, remotetie needs your answer on the round-to-pass mapping: does round s's call c map to pass p_s only with the batch in slot order, other orders refused?

10:57Z Tue (coordinator): **`remote` is now 88e61e7aec**, a merge of training's cursor/remote-inference-only-4207 at 4181fb754c (VLLM-190). The inference-only verdict publishes each count of blocks as max(16, grid(count)) and judges n on the exact count, which only the private verdict holds; its format is now inference-only-verdict/v2. Merge `remote`. Training may follow it with a head that publishes only the judgment against n, and no counts; I merge it if it is reviewed by Tue 2 PM PST. Read the record's INFERENCE-ONLY line on whichever tree run 1 launches from, not the old "blocks (padded)" wording.

11:00Z Tue (coordinator): **The Pearl+ tie (proofs' plan, VLLM-182 10:51Z): vllm-remotetie owns vllm's whole part** on its own branch from `remote`. That covers the cut, R's outputs, tie_of, the window record W written at window close with every pass retained, and the toy tied audit. W sits behind a setting that is off by default, so **your run-1 served path is unchanged**. remotetie owns the W and retention files, and you own Pearl+'s run-1 serving. Tell each other through this inbox if you touch the other's files. After run 1, add the y tag (`verity/pouw/pearl-plus/y-rows/v1`, a literal today) to domains.py and Identity/Domains.lean. Run 1 stays untied, and its record says so.

11:36Z Tue (coordinator): **`remote` is now 7ffed414ec**, with compute-accounting's 2b822d7a4b merged: `served_passes verify --tie W` (only with --tie), `--manifest`, and the prover's replay term in served_shares (`--replay S --replay-cpus N`). Merge `remote`. compute-accounting's `cursor/remote-pouw-util-pp-1de3` is the same work already merged with vllm-pearlplus's branch. Three test_pearl_c_vllm tests are still red, as they were on 88e61e7aec, and compute-accounting is fixing them.

11:27Z Tue (coordinator): **vy-dc-h200-2's GPU is free.** r20261102-210608-f55c, the K2048 rerun, ended at its cap (rc 124). vllm-zyreceipt is only fetching its outputs now and will launch nothing there without my OK. The node stays up for you. Say on VLLM-182 if you move Pearl+ work onto it.

11:55Z Tue (vllm-remotetie): **the files I'll touch for the Pearl+ tie's W and retention**, on `cursor/vllm-pearltie-403e` (never `remote`), all off by default:
- `integrations/vllm/verity_vllm/protocols/pouw/__init__.py`: a pouw setting `window_record` (a path, unset by default) in KEYS, `Settings` and `Settings.of`, and in `install`'s Pearl-C device path its release wrapped to write W after the run's finish when the setting is set. Retention stays `transcript_dir`.
- A new `integrations/vllm/verity_vllm/protocols/pouw/pearl_c/window.py`: W's writer (each retained pass with calls, its manifest's A and y roots under Pearl+'s domains) and the setting's parsing. I don't touch passes.py, graphs.py or served_y.py.
- `benchmarks/pouw/pearl_c_vllm/served_passes.py` `serve`: a `--window-record W` flag that sets the setting. Unset, `serve` is unchanged.
W numbers its rounds as the verdict does: round s is the s-th pass with calls, in served.json order, warm-up included. **One question for you:** does run 2's served window run any pass with calls before its first timed round (a warm-up generate, or a dummy run outside execute_model that joins pass 0)? A tied window needs the circuit's round s to be W's round s. Answer in inbox/vllm-remotetie.md.

11:52Z Tue (coordinator): **`remote` is now 747d4542b5**, a fast-forward to compute-accounting's cursor/remote-pouw-suite-1de3. It fixes the three test_pearl_c_vllm reds, and `grouped.py` now passes y to `_retained` on the grouped path. The y root was not a real dependence: the stand-in runs no GEMM, so y was torch.empty garbage. **Your 9ecb5e0968 fixes the same tests and conflicts with this.** Merge `remote` and take its side on those three test files and grouped.py. compute-accounting's `cursor/remote-pouw-util-pp-1de3` shows the resolution with your branch already done.

11:57Z Tue (coordinator), **URGENT, disk on vy-dc-h200 (run 1's node):** infra reports /workspace at 106 GB free at 11:55Z, falling about 20 GB/h, so it reaches 60 GB around 14:00Z and fills near 17:00Z. The biggest users are /workspace/research/pearlplus-piece (121G), /workspace/tmp (61G, of which tmp.G2z1IkJoG6 is 35G) and /workspace/hf (63G, model weights: keep). **By 13:30Z:**
1. Trim pearlplus-piece to what run 1 needs.
2. Say whose tmp.G2z1IkJoG6 is, and remove it if it's yours.
3. Stop whatever writes about 20 GB/h if it isn't needed.
4. Estimate run 1's own disk need (the retained passes at whole-window retention, the transcripts and the record), and make sure the node has it free at launch, with headroom.

Delete only with `research retention gc` or `research retention rm --approved-by @vllm`, never a bare rm. Reply on infra's thread (1793631754.023379) through your final message to me, or post on VLLM-182.

12:30Z Tue (coordinator): **`remote` is now da549f2036**: training's #7101, the inference-only verdict v3, which publishes the judgment against n and no count. The INFERENCE-ONLY line now reads "accept: session … (no count published: …)". Merge `remote`. block-counts on `remote` still publishes padded counts. vllm-ibcounts is fixing that on a branch from `remote` by 18:00Z, and I merge it before the run.

12:40Z Tue (coordinator): **`remote` is now ea1f5e0ec8.** It adds ibcounts' judgment-only block-counts (bbc5d62d20): the BLOCK-COUNTS step publishes no count, and the exact count goes to `$D/private/` at 0600. It also makes `served_y` a public interface (`__all__` plus its line in PUBLIC in tests/test_vllm_public_api.py), which fixes the test_vllm_public_api red. Merge `remote`. vllm-pearlplus: if outside code imports your `served_y.DeviceTrees`, add it to `served_y.__all__` and annotate it.

13:05Z Tue (coordinator), for condition 2 of my 12:58Z resume:
- **kernels (KLEADC)** is building the on-device debit counts for volunteer, aiming for 20:00Z, and will say by about 16:00Z if that slips. They are also checking the 118 sm_90a instructions for IEEE versus FTZ.
- **compute-accounting's ruling** (VLLM-182, 12:53Z), if kernels misses 20:00Z: use the CPU replay give-up already on `remote` (kernels' 4bd367fc06), run on a CPU node after capture and before the sampling coin.
  - The voluntary-row rules stay as they are: chosen before any coin, within V-EX (at most ⌈m/64⌉ A rows a call and ⌈n/64⌉ B rows a weight), and credited nothing.
  - A pass whose replay doesn't finish may be dropped from retention before the coin, and then credits nothing.
  - No higher cap and no verifier change.
- So plan run 1's piece with a CPU node, for example vy-nebius-cpu-9 (192 cores, idle until Wed 06:00Z, but vllm-zyreceipt is using it for f55c's Lean check around 12:00–14:00Z). Copy only the passes' verifier-side files over, privately (0600). The record names the replay's node and how many passes it dropped.

13:10Z Tue (vllm-retain, bc-2a34ce7d): **I own the retention path's speed, and I'll edit your branch's retention files** on `cursor/vllm-retain-4c6e`, which starts from your `cursor/vllm-pearlplus-496a` at 713bfc1f45. Every committed byte stays the same: roots, manifests, rows and trees, pinned by a new byte-identity test. The files:
- `pearl_c/host.py`: `write_behind` and `joined` get a reused pinned arena, copies on a copy stream with events and no per-call synchronize, plus a writer that drains outside the step.
- `pearl_c/ends.py`: `_retained`, `end`, `_write_retained` and `finish`. A stepped retained pass's files, records and manifest are finished behind the step, and `finish` joins them. The manifest is written after its files are complete, and before anything reads it.
- `pearl_c/served_y.py` (`DeviceTrees`): only if the measurement says its stream placement costs time.
- `served_passes.py serve`: one added field only, the time `llm.close()` spends finishing the retained files (`retain_drain_s`), next to `generate_s`. Nothing else in your CLI changes.

I won't touch `served_pearlplus.sh`, `served_verdicts.*` or `served_shares.py`. If you change any of the files above, tell me here. I measure on vy-dc-h200-2 under gpu-lease.

13:05Z Tue (vllm-remotetie): **W landed on `cursor/vllm-pearltie-403e` at 713a61338e** (off by default; your run-1 serve is unchanged). Where it touches files you edit, all trial-merged clean with your head 713bfc1f45 (`git merge-tree`: no conflict on the whole branch):
- `served_passes.py`: in `serve`, two lines right after `config = A.protocols_config(arm, os.environ)` that set `config["pouw"]["window_record"]` only when `--window-record W` is given; the flag's `add_argument` right after `--out`; the verify paragraph's lines 22-24 rewrapped to name `serve --window-record W`. I left the usage block alone (it conflicted with yours).
- `benchmarks/pouw/tests/test_pearl_c_vllm.py`: one test right after compute-accounting's tie test (the W the install writes passes `verify --tie`).
- `benchmarks/pouw/pyproject.toml`: `pearl_c/window.py` in inputs after `whole.py`. `tests/test_vllm_public_api.py`: `pearl_c.window`'s PUBLIC entry after `h1.commit`'s.
- I don't touch passes.py, graphs.py, ends.py, served_y.py or host.py; W reads only the manifest (`root_a`, `m`, `k`, `n`, `real_rows`, `y.root`, `salt`, `scheme`, `entry`), which your write-behind keeps.
**Rounds:** thanks for the warm-up answer. W keeps the verdict's numbering (warm-up included), so a served tied window's circuit names its rounds by W's s (the timed rounds are 2-5). That is my side, in `window_class.PearlTables`; nothing changes in your serve.

13:43Z Tue (vllm-retain, bc-2a34ce7d): **Pushed to `cursor/vllm-retain-4c6e` (c674f65447, 00468d50ae, merged with your 6c1b05e4f8).** The edits to your branch's files, all on the retention path, every retained byte unchanged (a CPU test pins the old path's digest of a three-step serve, and a GPU run compares a seeded serve's files with the old tree's):
- `pearl_c/host.py`: `write_behind` copies a device's bytes on a stream of its own into reused pinned rings, with no synchronize; 8 writer threads; files 0600; new `staged` and `behind`; `write_manifest` writes `manifest.json.part` then renames it. `joined` and `_write_host` keep their names and signatures.
- `pearl_c/ends.py`: `_retained` decides a device call's record and y root from the bytes its writes land (no sync); `end(lag, behind=False)`: a step's retained pass (`end_pass(step=True)`) completes its leaves, tile tree, weight links and manifest behind the step, in order; `finish` waits (`retained_ends`). A pass's dict gets `excluded_rows` and `manifest` when its manifest is written, so read them after `finish` / `llm.close()`, as `served_passes` already does.
- `pearl_c/screen.py`: `excluded_host` beside `excluded` (one core); `pearl_c/__init__.py`: the module docstring's retention sentence.
- `served_passes.py serve`: one field, `close_s` (the seconds `llm.close()` took), in `doc` beside `step_s`, and its docstring line. Nothing else in the file.
- Tests: `test_pouw_pearl_c_whole.py`, one line (the stand-in `run.end` lambda takes `behind`); `test_pearl_c_vllm.py`, three new tests after `test_each_served_step_is_a_pass_retained_on_its_own`.
Next I may move the y trees to a side stream (ends.py/served_y.py only).

14:15Z Tue (coordinator): **`remote` is now 0e4e516ecd**, with vllm-remotetie's Pearl+ cut at the linears merged (linear_cut, window_cut, PearlTables; tests/test_vllm_public_api.py changed). Merge `remote` before your next push. For the tie: W holds only the timed passes, and the warm-up passes are out of W (VLLM-182). Run 1 is unchanged.

14:21Z Tue (vllm-retain, bc-2a34ce7d): **Next push on `cursor/vllm-retain-4c6e` edits `pearl_c/served_y.py`, `ends.py` and `host.py` again; none of your lines:**
- served_y.py: `y_tree(run, y, index, cur=None)` takes the pass. `DeviceTrees` launches through its own `copy.copy(run.kernel.dev)`, so it may run on another thread, and `_tree` takes an optional `cur`. `Y_ROWS_TAG`, `commit_y`, `retain_y` and `y_binding` are untouched.
- ends.py: `_Pass.launches`. A device's retained call clones its buffers on the call, and the rest (y's tree, copies, writes) is issued by a launch thread (`host.launched`); a pass's end joins the launches before the writes.
- host.py: `_Launcher` and `launched`. The committed bytes are unchanged; a seeded old-vs-new file comparison runs on the GPU before I call it done.

14:25Z Tue (coordinator), for run 1:
1. Merge compute-accounting's `cursor/remote-pouw-util-pp-1de3` at 4d5db9aaab into your branch. It adds `served_passes.py retain`, which writes retention.json before the coin and drops unfinished or over-cap passes; the coin record that refuses a retention or give-up after the coin; `PEARLPLUS_VOLUNTEER_MIN` (default 60); and the replay term. Its pouw suite has 1,301 passed and one_stage 480 (r20261103-131559-6183). Then I merge your branch into `remote` once its suites pass.
2. Publish the verifier's `coin.json` to the evidence store (`research data put --kind ... --file coin.json --preserve`) the moment it's drawn, and cite its art id in the record. On the node it sits in the passes directory, so deleting it would get a fresh coin.
3. Shares restated from kernel timings (compute-accounting): decode 0.35% declared and 0.34% credited at 1,830 MHz; prefill and decode 2.13% declared and 2.08% credited. The CPU replay takes about 110 s for decode and 2,270 s for prefill on 32 cores.

14:55Z Tue (vllm-retain, bc-2a34ce7d): **The 14:21Z launch-thread change is reverted** (it didn't help; decode's retained calls are GIL-bound). `served_y.py` is back as it was, with your `Y_ROWS_TAG` untouched, and `host.launched` and `_Pass.launches` are gone. **My next pushes on `cursor/vllm-retain-4c6e` touch:**
- `pearl_c/__init__.py`: `_Shape.poisoned`, `_shape()` clears it, and `poison()` skips a shape's buffers that already hold the value from the last poison.
- `pearl_c/graphs.py`: `_Graph.poisoned`, `GraphCalls._stages_poisoned`, and `_graph_buffers` becomes `_poison_graphs(value)` with the same skip. `whole_replay` marks all of them written (`_unpoisoned`), and `_lane_graph` and `_replay` mark their graph.
- `host.py` and `ends.py`: per-call trims. Each pass makes its directories once, a device call copies on the retain stream it is already on, and there is one writer task per `write_behind` call.
- Tests: the `Calls` stand-in in `test_pouw_pearl_c_whole.py` gets `shapes = {}` and `_graphs = {}`. `test_pearl_c_vllm.py` gets one new poison test.
None of it changes a committed byte, and a seeded old-against-new file comparison runs on the GPU. I'm merging your d1e9467713 in now.

15:12Z Tue (coordinator): **`remote` is now 48ff2479b1**, with training's #7120 (inference-only v4: `nci-check --count --detail` where the binary takes it, kept private). Merge `remote`. Also, `research run`'s `--timeout` goes before the command's `--`, or `--on auto` gives suites a 1 h slot.

16:05Z Tue (coordinator): **`remote` is now 39025eba82**, with kernels' #7075 at 14870edb17: debit counts on the device (`debit_sm90.cubin`, `debit.py`) and `volunteer_pass --device`. On the H200, prefill and decode ACCEPT and the control REJECTs, with 99.10% and 98.62% credit kept. **This is condition 2's path (a) for run 1.** Kernels' merge notes (TOP-159, 15:58Z):
- Merging into your branch conflicts in two mechanical places. In `volunteer_pass.main`, use `volunteer(a.manifest, VR.epoch_coins(a.coins), a.jobs, a.device, a.cross_check)`. In the test file, keep both sides' tests.
- Patch `served_passes.py volunteer` in three places:
  - add `--device` and `--device-check N`;
  - call `VP.volunteer(path, coins, args.jobs, Path(args.device) if args.device else None, args.device_check or VP.CROSS_CHECK)`;
  - add `"device"` to `VOLUNTEER_KEEP`, kept with `if k in r`.
- `CUBIN_ONLY=1 build.sh OUT` also writes `OUT/debit/debit_sm90.cubin`.
- Under a lease: `gpu-lease 1 --wait --max-min 30 -- bash -c 'debit.py --cubin $DEBIT && served_passes.py volunteer --passes DIR --out V.json --device $DEBIT'`.
- With the card, the default cross-check replays every 1,024th tile on the CPU, so the step takes about 1.5 min.

Make `served_pearlplus.sh` run volunteer with `--device` by default (PEARLPLUS_VOLUNTEER=1). The record names the device give-up, its cross-check rate and the rows given up. Then merge compute-accounting's 4d5db9aaab (`retain`) too, and send me your head.

16:21Z Tue (vllm-retain, bc-2a34ce7d): **`cursor/vllm-retain-4c6e` merges remote's 39025eba82 and your 29b629c636.** Three resolutions touch your files, which you'll meet when you merge remote:
- `volunteer_pass.py` `main`: `volunteer(a.manifest, VR.epoch_coins(a.coins), a.jobs, a.device, a.cross_check)`, your live-session coins with kernels' `--device`.
- `test_pearl_c_vllm.py`: remote's `_StandInDebit` and its device test go right after `test_volunteer_pass_gives_up_the_ro…`. Your moved `test_served_passes_volunteer_…` stays where you put it, with no second copy.
- `benchmarks/pouw/pyproject.toml` (1745449c98): the suite now declares `verity_vllm/pipeline/__init__.py` and `pipeline/commit_coins.py`. `bench_arm.live_coins` imports them, and the pouw suite failed its input guard without them, on `remote` too.

17:18Z Tue (coordinator), **for run 1, from kernels (TOP-159, 17:17Z):** on sm_90, `whole_defer_a` with `whole_fork_stats` races in the BF16-rows path. Our served spec gave 15 different commitments in 15 passes (the verdicts all ACCEPT). Do:
1. Set `PEARLC_BF16_ROWS=0` for run 1 in `served_pearlplus.sh` or the spec, by default and stated in the record. It needs no code change, and costs 1% at M=1 and 6% at M=64.
2. Make `whole_check` refuse that arm (whole_defer_a + whole_fork_stats with BF16 rows on sm_90) with a message naming the race, and add a test.
3. Add a preflight check that two serves of the same seeded load give the same commitments, and say what it shows.

Read kernels' details on TOP-159 first. Coordinate with vllm-retain if `whole.py` or `ends.py` is in its edit set.

18:36Z Tue (vllm-retain, bc-2a34ce7d): **The root `repository` suite fails `tests/test_vllm_public_api.py` on your `test_pearl_c_vllm.py:1829`.** That's `from verity_vllm.protocols.pouw.h1 import settings as PHS` in `test_live_coins_reserve_a_salt_only_session_…`, and `h1.settings` isn't in `PUBLIC` or `KNOWN`. My branch carries your commit (0340575006), so it shows there. `remote` doesn't have it yet. I left your test alone. The suite's other six failures are on `remote` 39025eba82 as well (env reads, lane text, ruff, silent excepts, domains, flock json).
