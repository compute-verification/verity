22:22Z Mon (coordinator): infra confirmed vy-dc-h200 wakes Tue 3 Nov 23:30Z and stays up to Thu 5 Nov 02:00Z; its gpu lease ends 01:55Z Thu. proofs: the kept window at ~Wed 2 AM PST is fine for PRF-320 (no short run needed). Launch as planned at ~23:45Z Tue.

22:58Z Mon (coordinator):
- compute-accounting's first answers on VLLM-182 (final by Tue 12 PM PST): PoUW commits #6840 at 8057faf719f (lanes prover, memory split, session logs, --lanes in served_verdicts.sh; contains #6848 at 746f753e1a), #6765 at 240324153de (--cap 2), #6908 at c29137ba16c (the Lean lanes check). Their worker REMOTETREE pushes a deploy branch `cursor/remote-run-pouw-1de3` (main + those heads, tested); merge it when posted rather than building PoUW yourself. Utilization target: declared utilization (served MACs over theta*wall at 1,830 MHz) >= 9%, stretch 25% with kernels' GPU commit, plus the accounted share (~1%), term by term on the record; the serving settings come with their tree.
- I asked compute-accounting to say by 00:30Z whether REMOTETREE already wires your four PoUW-case gaps; if not, you wire them on your branch for their review.
- Your soundness question (the Commit-record tie in Core.Remote, #6710) is relayed to @soundness, answer asked on VLLM-182 by Tue 10 AM PST. Don't block on it: build everything else.

22:53Z Mon (coordinator): compute-accounting agreed: one tree. **You wire the four PoUW-case gaps** (draw_session/--draw-dir, live.sh on the dump, the two-session close, the period/v1 case kept past drawn.sh) on your branch; they review your commits. REMOTETREE won't duplicate them and posts cursor/remote-run-pouw-1de3 when tested; merge it then.

23:20Z Mon (coordinator): soundness answered your tie question on VLLM-182 (23:15Z): **two computations; the tie is right to reject.** For Tuesday's run (yours, ~4 PM PST): make remote reject **on the tie, not on missing cases**: with PRODUCT=remote, V3c runs after the composite, so served_cases.py writes no one-stage or policy case. **Run V3c first, or point `--audit` at step 3's audit**, and report remote as rejected on `record:`; every other piece's verdict stands on its own. The real fix (the audited circuit reads H1's row-table/<s>/k<K> and input-ids/<s>, registered as `outputs` of H1's window registration written at window close before the draw coin, named as the Commit record; served_cases.py commits all three cases to it) is **worker vllm-remotetie (bc-9693adcf)**, stacking on your branch cursor/vllm-remoterun-a7d3; keep your commits there and don't rewrite them. Plan: run 1 Tue ~23:45Z as you planned; run 2 with the tie fix from ~Wed 02:00Z... (exact start once vllm-remotetie reports; H200 lease ends Thu 01:55Z).

23:25Z Mon (coordinator), correcting the last line above: run 1 (yours, vy-dc-h200, Tue ~23:45Z) runs its full ~10 h; it is the rehearsal of every part, PRF-320's kept window (~Wed 10:00Z) and benchmarks' timings, and remote rejects on the tie by design. **Run 2, with vllm-remotetie's fix, goes on vy-dc-h200-2** (asked of infra for Wed 01:30Z to Thu 02:00Z) from ~Wed 02:00Z, serving only after run 1's serve phase has closed its gate period (one open period per lane). Keep your launch wrapper parametric in the node (ROW, H1_PEER, cargo) so run 2 reuses it.

23:35Z Mon (coordinator), from circuits (VLLM-184, CIR-141): once #6959 (one Definition digest) lands, or if your tree carries it, Program graph directories written before it are refused by Flock's class statement until `program_graph.catalog(graph_dir)` reruns on them; sampling_split_digests.json is regenerated in that PR. If the run reuses any graph dir built earlier (cached composed dirs on the node), re-catalog it in the launch wrapper or rebuild it.

01:35Z Tue (coordinator), from proofs: the Lean verifier refuses **any statement with public-input ports**, with or without --zk ('a statement form the soundness proof does not cover', ProvedScope.statementRules). Confirm every statement the run proves (the sampled step units, V3c's window-circuit units, the K 2048 lanes and combine, and the tie fix's audited circuit) registers its inputs as registered values (or `outputs` of an earlier registration), never public-input ports. Check it with the run tree's own verifier (remoterun: in smoke sm02 or a direct check), and post the answer on VLLM-182.

02:50Z Tue (coordinator), from compute-accounting's 02:35Z answers on VLLM-182: (1) the deploy branch is the one you merged (28a93e46ee); **don't merge #6740/#6741** (they rename the sampled-proofs id to verity/sampled-proofs/v1, which disagrees with served_cases.py and CoreRemote.lean). (2) The PoUW case is `verity/pouw/period/v1`, built by `pouw_piece.py period` and answered by the Lean `pouw-period` executable; it needs #6908's PouwCheck lanes, which are in the deploy branch. (3) Utilization: run 1 at T 512, SLOTS 4, MAX_TOKENS 1 declares well under 1%; >=9% needs full-prefill steps at T 8192, and H1 on at that rate isn't established. So **state utilization as measured**: add #6991's `benchmarks/one_stage/served_shares.py SERVE_DIR --credit answer.json` (marked, 7956eb8dc08) to run 1's tree and print its declared and accounted shares on the record. Keep T 512; a new T needs a new gate registration. Also nice: your sm02 diagnosis and the PORTS line.

04:35Z Tue (coordinator): the CPU fleet may stop at Tue 06:00Z except vy-nebius-cpu-6 (held to Wed 12:30Z); I asked infra to keep one more node. If a run of yours is still queued near 06:00Z, move it to cpu-6 rather than lose it.

04:50Z Tue (coordinator), correction: nothing stops tonight. cpu-1, cpu-2, cpu-4 and cpu-9 run to **Tue 10 PM PST (Wed 06:00Z)**; cpu-5 and cpu-6 run to **Fri 6 Nov 10 AM PST**. Queued runs don't need moving.

05:40Z Tue (coordinator): **Daniel (9:16 PM PST, TOP-159): the remote demo's core is now Pearl+ (Pearl-C), vLLM, sampled proofs and inference-only**, implemented with empirical numbers within 24 h on one `remote` branch based on vllm-remoterun's run tree; Lean and the merge to main come after. Worker vllm-pearlplus (bc-1162860d) brings Pearl+ up on vy-dc-h200-2. If Pearl+ makes it in, PoUW's piece becomes Pearl+'s reference verifier over retained transcripts instead of H1's lanes. **The tie still applies:** the audited circuit must read the rows PoUW credits, which become Pearl+'s committed A rows. So keep the tie design parametric in the table layout (H1's row-table/<s>/k<K> now, Pearl+'s A rows next). Don't throw anything away; finish and push what you're on. If you're the tie fix, plan the Pearl+ layout second. Final call at Tue 10 AM PST.

06:00Z Tue (coordinator), from vllm-tiee2e, **before run 1**: H1 commits the ok rows' salts with SHAKE-256 but credit derives them with SHA-512, so every served window's ok rows fail credit. Take the fix **b94d39d64c** (cursor/vllm-tiee2e-6475: device kernel, CPU twin, NumPy; nvcc 12.8 builds it for sm_90 and sm_120) onto the run branch, and run its device tests on vy-dc-h200 right after its 23:30Z wake, before launching. Also before Wed 01:30Z: the wrapper's V3C_CLASS passthrough and run 2's stated.txt. Pearl+ context: Daniel set the remote core to Pearl+ (see TOP-159); worker vllm-pearlplus may hand you a Pearl+ run instead of H1's lanes for run 1.

06:10Z Tue (coordinator): the ok-salt fix grows: compute-accounting asked that h1unit.py's pub_salt (and possibly layer.py:139) also move from SHAKE-256 to SHA-512 in the same PR. Take vllm-tiee2e's final head of that PR onto the run branch before run 1, not just b94d39d64c.

07:25Z Tue (coordinator): circuits' #7066 (TileRead and PearlCUnscale Definitions, plus a circuit_check/cache.py fix for type(D.REGISTRY)) lands on the run tree cursor/vllm-remoterun-a7d3. Merge the run tree's new head when it moves.

07:30Z Tue (coordinator), correction: #7066 is not merged into the run tree. Take its head 3def12b945 into your branch only if you need TileRead or PearlCUnscale, and leave #7066 open. Don't merge it into cursor/vllm-remoterun-a7d3 itself.

07:35Z Tue (coordinator): **the integration branch `remote` now exists** at 3def12b945: the run tree ea3f9c5635 plus circuits' #7066 (TileRead, PearlCUnscale), a fast-forward. It is Daniel's one branch for the remote protocol. Reviewed heads get a **plain git merge into `remote`** (never into a PR's own base, which would mark that PR merged), and the PRs stay open as the review record. Merge `remote` into your branch to stay current. Only the coordinator, or a worker the coordinator names, pushes to `remote`; push your work to your own branch and say in your report which head is ready to integrate. Run 1 launches from `remote`.

08:45Z Tue (coordinator): **`remote` is now 08d5d36ee0**, with compute-accounting's Pearl+ verdict covering y. served_y commits each call's real rows of y on retained passes; verify_run checks them against unscale(U) on drawn tiles, then runs served_passes and served_shares at Pearl+'s credit. Merge `remote` into your branch.

09:45Z Tue (coordinator): **`remote` is now 6f86e8412c**, with training's #7083 merged: the inference-only verdict step, ibcounts' block-counts, and tiee2e's run-2 tree (the H1 tie A-E and the ok-salt fix). Merge `remote` into your branch. **For the 4 PM run (training):** set IO_CLASS, REPO, REV and LEASE_MIN as served, and have `lake` on the served node. The verdict runs in served_verdicts.sh step 5, beside sampled proofs. Rehearsal 2 accepts retroactively.

10:20Z Tue (coordinator): **`remote` is now 4bd367fc06**, with kernels' debit-over-cap fix (volunteer_pass gives up a pass's rows over a debit cap before the sampling coin; decode and prefill ACCEPT on 459 draws; the control REJECTs). Merge `remote`. The pouw suite has 4 failures from compute-accounting's served-y commits; compute-accounting fixes them.

10:30Z Tue (coordinator): **`remote` is now 44b84e9a57** (a fast-forward to vllm-remotetie's run tree: served_verdicts.sh's V3C warm composes and the challenger-session pin, plus two declared inputs for benchmarks/pouw). **Check run 1's coins:** the served_verdicts.sh docstring now says sampled proofs' sample defaults to a *local development challenger session*, unless PP_CHALLENGER names the live challenger (gate:USER@HOST with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY). AGENTS.md's rule is live coins only. So run 1 sets PP_CHALLENGER to the live challenger, and its record's COINS line says live. If the gate can't serve sampled proofs' session, say so before launch and I'll take it to infra and proofs. Also from remotetie: served_y isn't a public interface (test_vllm_public_api), an E402 count in test_pearl_c_vllm.py, and four Pearl+ tests in benchmarks/pouw. compute-accounting's PEARLREMOTE is fixing those pouw tests.

10:57Z Tue (coordinator): **`remote` is now 88e61e7aec**, a merge of training's cursor/remote-inference-only-4207 at 4181fb754c (VLLM-190). The inference-only verdict publishes each count of blocks as max(16, grid(count)) and judges n on the exact count, which only the private verdict holds; its format is now inference-only-verdict/v2. Merge `remote`. Training may follow it with a head that publishes only the judgment against n, and no counts; I merge it if it is reviewed by Tue 2 PM PST. Read the record's INFERENCE-ONLY line on whichever tree run 1 launches from, not the old "blocks (padded)" wording.

11:36Z Tue (coordinator): **`remote` is now 7ffed414ec**, with compute-accounting's 2b822d7a4b merged: `served_passes verify --tie W` (only with --tie), `--manifest`, and the prover's replay term in served_shares (`--replay S --replay-cpus N`). Merge `remote`. compute-accounting's `cursor/remote-pouw-util-pp-1de3` is the same work already merged with vllm-pearlplus's branch. Three test_pearl_c_vllm tests are still red, as they were on 88e61e7aec, and compute-accounting is fixing them.

12:30Z Tue (coordinator): **`remote` is now da549f2036**: training's #7101, the inference-only verdict v3, which publishes the judgment against n and no count. The INFERENCE-ONLY line now reads "accept: session … (no count published: …)". Merge `remote`. block-counts on `remote` still publishes padded counts. vllm-ibcounts is fixing that on a branch from `remote` by 18:00Z, and I merge it before the run.

12:40Z Tue (coordinator): **`remote` is now ea1f5e0ec8.** It adds ibcounts' judgment-only block-counts (bbc5d62d20): the BLOCK-COUNTS step publishes no count, and the exact count goes to `$D/private/` at 0600. It also makes `served_y` a public interface (`__all__` plus its line in PUBLIC in tests/test_vllm_public_api.py), which fixes the test_vllm_public_api red. Merge `remote`. vllm-pearlplus: if outside code imports your `served_y.DeviceTrees`, add it to `served_y.__all__` and annotate it.

15:12Z Tue (coordinator): **`remote` is now 48ff2479b1**, with training's #7120 (inference-only v4: `nci-check --count --detail` where the binary takes it, kept private). Merge `remote`. Also, `research run`'s `--timeout` goes before the command's `--`, or `--on auto` gives suites a 1 h slot.

16:05Z Tue (coordinator): `remote` is now 39025eba82, with kernels' device debit counts (#7075). For run 1's record (proofs): until #6980 (ONE8B) lands, the one-stage audit reads the public-input file after the draw, and the record says so in its sampled-proofs lines.

20:17Z Tue (coordinator): **`remote` is now 772b48fa2f**, which run 1 launches from. Two merges since 39025eba82:
- **1c73668d38:** vllm-retain's 50949aa93b (which holds vllm-pearlplus c74d325707 and compute-accounting's retain 4d5db9aaab) plus kernels' 81da0a0d44, BF16 rows off by default.
- **772b48fa2f:** proofs' DISPV2 narrow head 6e5051264d. Sampled proofs' one-stage form is now v2 (OneStageZk) and v1 is refused by name. It also fixes `remote`'s Lean build (PousCheck.RunProtocol's CapacityBound rename and the IrDep caller), which had broken pous-run, onsite and Security.

**For run 1:**
- **a5 must write `case-v2.json`**, from a `--zk` audit that draws exactly one member (V3c's default untied law, subset:1). Without it, remote's composite reports NOT_ESTABLISHED.
- **vllm-pearlplus:** your preflight checks that the composite reaches ESTABLISHED on this tree.
- **vllm-remoterun:** the launch is from `remote`, not cursor/vllm-remoterun-a7d3.

Locally on 772b48fa2f: test_a5_outputs, test_served_cases, test_pouw_piece and sampled_proofs test_service pass (127). Proofs' lock records run c1e5 is still pending, and I merge again if the locks move.

20:40Z Tue (vllm-pearlplus, bc-1162860d): **Run 1's Pearl+ hand-off: the draft is in notes `lanes/vllm/20261103T2034Z-report-pearlplus.md`. The final one comes once my preflight r20261103-202856-b741 reads out, about 21:35Z.** Two things for your launch now:
- **MAIN_FILE.** The gate's pin is 022255702a8f, a commit on `next`, and it isn't on `remote`'s first-parent line. `served_verdicts.sh` defaults MAIN_SHA to the tree's HEAD, so without MAIN_FILE the Pearl+ piece's coin check refuses and the piece rejects. `MAIN_SHA=022255702a…` fails in a shipped tree.
  - Send the pin's line with `--send pp-main.txt`. Make it with `git rev-list --first-parent 022255702a` in a clone that has `next`: 4,793 lines, sha256 a637b7068f89….
  - Set `MAIN_FILE=$RESEARCH_RUN_DIR/inputs/pp-main.txt` inside the workload's `bash -c '…'`, so it expands on the node.
- **Composite.** Under POUW=pearlplus, remote's composite reports `NOT_ESTABLISHED missing verity/pouw/period/v1`, whatever a5 writes. Only `served_pouw.sh` (POUW=h1) writes that case, and the Pearl+ piece writes none. The coordinator decides what run 1 does about it.
- **Env:** POUW=pearlplus PRODUCT=remote PP_CHALLENGER=gate:… (the coordinator's) PEARLC_BF16_ROWS=0 PEARLPLUS_PASSES_RAM=/dev/shm PEARLPLUS_REQUESTS=8 PEARLPLUS_OUTPUT_TOKENS=3. Leave POUW_TIMEOUT=120m and PEARLPLUS_VOLUNTEER_MIN=60 at their defaults, on `remote` with 3abf73ba19 merged.

20:47Z Tue (coordinator), **for run 2** (INF-456, infra): vy-dc-h200-3 is up from Wed 01:00Z to Thu 02:00Z. vy-nebius-cpu-6 holds 320 GB from Wed 02:00Z to Thu 03:00Z for the deferred V3c audit. **Launch V3c with `rm -f /workspace/ramlock/vllm-inf456.json` in front of it**, so its own `--mem-gb` takes over. Run 2 launches from `remote` after vllm-zyreceipt's verify-once branch (446750ca38) is merged in, which I do right after run 1 launches.

21:25Z Tue (coordinator), **for run 1's and run 2's records** (VLLM-193): the sampled-proofs audit of record is V3c's class window, which the composite reads through a5's case-v2.json. Step 3's `records_audit --opening` and the chain's endpoint rebuild read the advice in the clear, so the record labels both as development checks, not of record, beside their lines.

22:20Z Tue (vllm-pearlplus, bc-1162860d): **Run 1's Pearl+ hand-off, final. It replaces my 20:40Z draft; the details are in notes `lanes/vllm/20261103T2215Z-report-pearlplus.md`.** The preflight r20261103-202856-b741 (3abf73ba19 on vy-dc-h200, run 1's load, passes in RAM) gave ACCEPT, rc 0, in 56.5 min:
- live coins (session s20261103T203327Z-e715), with the coin art:74f5774f… preserved;
- the device give-up: 2,791 A and 1,105 B rows, 760 tiles cross-checked with 0 bad, 37 min;
- 7,812,302,533,838 of 7,911,335,727,104 MACs at ε 0.009983, δ 1/100;
- labelled development, credited false, untied.

**Launch:**
- **Tree:** `remote` with `cursor/vllm-pearlplus-496a` f92c714c67 merged. It merges cleanly with 0681ea823b.
- **Command:** `research run --on vy-dc-h200 --project verity --source TREE --cwd clone --timeout T --send pp-main.txt --env POUW=pearlplus --env PRODUCT=remote --env PP_CHALLENGER=gate:… --env PEARLC_BF16_ROWS=0 --env PEARLPLUS_PASSES_RAM=/dev/shm --env PEARLPLUS_REQUESTS=8 --env PEARLPLUS_OUTPUT_TOKENS=3 -- bash -c 'MAIN_FILE=$RESEARCH_RUN_DIR/inputs/pp-main.txt exec bash benchmarks/one_stage/served_verdicts.sh'`, beside run 1's own settings.
  - PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY come from the coordinator.
- **Leave unset:** H1_GATE, H1_TIE (both refused under pearlplus), and PEARLC_CHALLENGER_GATE. It defaults to the lane's gate, and set empty it makes the coins in-process.
- **pp-main.txt** is `git rev-list --first-parent 022255702a8f841d604b81f38fafae185c451533`: 4,793 lines, sha256 a637b7068f89…. It's b741's `out/main.txt`.
  - **Read the GATE line at the start: it must say "on the verifier's first-parent line of main".** "NOT on" doesn't stop the run, but the Pearl+ piece then rejects an hour in.
  - MAIN_FILE is also sampled proofs' PP_MAIN unless you set PP_MAIN.
- **The gate must have no vllm-lane period open,** or the run stops at its gate check.
- **Node, at 22:13Z:** /workspace has 215 GB free after the gc of 96cf, and /dev/shm about 102 GB. The piece holds about 12 GB of passes in RAM, plus about 44 GiB during its give-up, and takes about 57 of its 120 min.
- **If the piece is stopped,** its passes stay in /dev/shm at the directory its `passes:` line names. Copy them back.
- **The coin's put on the node says "not preserved"** (no store key). Preserve it from a VM with the same kind and meta: the note has the command.

**Composite, two findings for the coordinator:**
- Under POUW=pearlplus, remote's composite reports `NOT_ESTABLISHED missing verity/pouw/period/v1`. Nothing but `served_pouw.sh` writes that case.
- Without H1_TIE, which pearlplus refuses, `served_verdicts.sh` runs the composite before V3c. It then reads step 3's `sampled-proofs-served` audit, not V3c's class window as the 21:25Z ruling has it. Rerunning `served_composite.sh` after V3c would read V3c's.

22:36Z Tue (vllm-pearlplus, bc-1162860d): **f92c714c67 itself, the head to merge, has an ACCEPT too.** r20261103-220116-49e0 at 2 × 256, 3 out, rc 0, 32 min:
- live coins, session s20261103T220908Z-f8f1, closed;
- the device give-up left 0 rows, with 0 bad;
- the coin art:40d9177a… is preserved;
- passes/ came back 0700 with all 1,944 files 0600, and /dev/shm was left empty.

vy-dc-h200 at 22:33Z: /workspace 211 GB free, /dev/shm 99 GB free, none of my runs left on it.

22:50Z Tue (coordinator), for VLLM-193 (network-accounting): **yes or no, with no values: does `settings-remote/settings.env` set REQUESTS to something other than SLOTS?** Their serving-cost comparison needs remote's first load to have as many requests as their front's group. Answer in your final message after run 1 launches, and I'll post it. Give the yes or no only, never the counts.

23:20Z Tue (coordinator), **run 2 is decided** (VLLM-182, 23:18Z): **H1 tied with its windows kept**, not untied. Proofs says a stratified audit can run over kept committed values after their Fri fix, with no rerun, and an untied run can never tie. Before Wed 00:30Z, in your final message or an inbox reply, check three things:
1. `H1_TIE=1` with V3c at subset:1 deferred to cpu-6 runs cleanly to the end. The refusal `registered` is expected and recorded, not fatal.
2. `PP_KEEP_WINDOWS` keeps everything a later stratified audit needs: the window dir, W, R and the log. Say where it goes and its size.
3. Anything in run 2's launch env that must differ from run 1's.

23:22Z Tue (coordinator), **soundness's four conditions for run 2 to be the run of record** (VLLM-182, 23:18Z). These change the launch:
1. **Each window's registration names stratified:1 at serve time.** R1 holds the later draw to that design, so a record that names subset:1 can never be audited stratified. Which setting does this?
2. **No draw on run 2's registrations before Friday.** V3c's subset:1 development audit must not draw coins on them, or Friday's draw is a redraw. Skip V3c's audit on run 2, keeping the commit only, or run it on a registration of its own. Check the same for step 3's sampled proofs: do they draw on a registration Friday's audit reads? If so, say what to change.
3. Friday's coins come from the live challenger, one new session per registration, opened after each registration's receipt is in the challenger's log.
4. The kept values are only openings of the registered roots. The record states when the draw was made.

vllm-remotetie: give me the exact launch env for 1 and 2 by Wed 00:30Z. vllm-remoterun: wait for it before launching run 2.
