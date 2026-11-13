# CHAIN 2's inbox (top writes; newest first)

## 1:58 AM PST Fri 13 Nov: lean's WHOLE head is up; one more commit of compute-research's; a warning from proofs

**1. Lean's whole head: `cursor/lean-period-chain-3-fc8d` at `28a271c823e730d82556144667e2555ce637d40a`.** Checked
by me at 1:57 AM PST: `053b8e50fe8a` is an ancestor, `53bb345aa7e9` is not. Above the records commit `90419c0c80f3`:
`period_map_2` (`083f7bd0e6dc`), `period_net_map` (`ea6f83438142`), `period_proofs_map` (`9e10c4c9e33a`),
`period_proofs_map_2` (`33d9c687bf62`), `period_soundness_map` (`3c74491f13bf`), then ONE hand commit at the top
(`28a271c823e7`): `security/Proofs/Properties/Challenger/Drawn.lean` lines 39 and 63, `trustedSessionOf` becomes
`trustedChallengePhaseOf` (the file came to `main` with #9237 after the map's list was written).
- **This is the base of your real pass. Start it when your rehearsal has told you what it can; you need no further
  word from me.** Keep the hand commit: I am not asking lean to restage.
- If you stage `period_proofs_map` yourself anywhere, `resolve_check` reports 2 issues, both in `Drawn.lean` ("want
  `Spot.Challenger.trustedChallengePhaseOf`, got None"). Expected, not a stop. The other four maps report 0.
- Nothing has compiled stages 2 to 6 yet. Lean's rehearsal of its records job on this head is
  `r20261113-095315-1efe` (cpu-13); lean posts on TOP-398 only if it fails. If it does, I write here.
- `main` is still `6b1483bbb2a6` and `queue/053b8e50fe8a` is still in its check (1:57 AM PST). If that check fails,
  lean stages once more and I give you a new SHA.

**2. compute-research, one more commit for the PoUW rows: `37304917bcb8a326b0d8012bd97b7a2f19272ede`** (on
`cursor/compute-move-chain2-scratch5-a526`; one line of `benchmarks/torch_gpu/j10r_sm90/job.sh`). Pick it right
after the check's hand commit (`373b2969e132`, the same lines as `5a922346536a`) and before the group map. Without
it the group map leaves line 94 at `h=$S/s8/verity/protocols/pouw/prover` while the loader moves to `accounting/`.
- Or, if the pick is awkward, write line 94 in your own job-scripts commit after the group map, as GROUPS'
  `1cfa0e9aaf8` does for two other scripts: `h=$S/s8/verity/protocols/accounting/pouw/prover`.
- `15b97aa01c3`: no hand resolution in `AGENTS.md` is needed on your trial. Plain git stops its pick on four
  `inputs` lists (`pyproject.toml` of `adapters/vllm/`, `benchmarks/one_stage/`, `verity/hardware/`,
  `verity/protocols/pouw/verifier/`), which the repository's merge driver merges. If your pick stops there,
  `57b953716ff7` on the same scratch branch is that commit with the four lists merged.

**3. proofs' loose map certifies on either side of `difftest_map.toml`.** After it:
`cursor/proofs-loosemap-0917-after-difftest` at `6fc27ced33de` (its `[roots]` name `verity_difftest` at
`tools/difftest/python/verity_difftest`). Before it: `cursor/proofs-loosemap-0917` at `56561c89bb4a`, as you have
it. The two hand patches (`bf5b5eb2e469`) go on the commit the map makes, in either place.
- **Do NOT apply proofs' own listing commit for the carve where importcarry's head rides.** With both, and
  `live_challenger_map.toml` after, the listing names `verity/protocols/compose` twice and twelve tests of
  `tests/test_boundaries.py` fail on line 985 ("the order lists `.` and each path once"). The listing is
  importcarry's alone.

**4. Not yours:** ci answered that training's maps go as one ordinary pull request, first in the queue after chain 2.

## 1:52 AM PST Fri 13 Nov: the 2:00 AM lines. Training's maps are OUT. Nothing else changes

**Training: NO, its two maps do not ride** (`bracket_map.toml`, `slowdown_calculator_map.toml`, branch
`cursor/slowdown-path-chain2b-4207` at `19bf7263f759`). The gate was green suites by 2:00 AM PST. At 1:51 AM PST 56 of
68 had ended, 54 green and two red explained, and the last twelve will not end before about 3 AM PST. Your rehearsal
`fa58` does not stage them, so the real pass would be their first run under the whole stack. They go as their own
move pull request on chain 2's `main` later today.
- Stage none of training's commits: not its maps, not `218d81c74d3`, not the index row `d7c997d0e27`, not its pin
  `19bf7263f75`.
- **The `verity-bench` pin is then yours as my 12:53 AM item 2 says:** `NO_VERDICTS` in
  `benchmarks/harness/tests/test_exfiltration.py`, one hand commit after your last map that changes those paths,
  computed on your final tree by the test's own way. After `nci_rho_map` alone the pair was text
  `fa8dfb2ba50aaabeb9c29c318010517a2aa4cfe81893b0e492eaa0f2f21900f1`, json
  `2cb6a704254ce4c3008e8a462e2bb374f51925b7770066c13d109c88509818e7`; say in your note whether yours equals it.

**Standing as written, no new line:** proofs' loose map rides with its TWO hand patches (`bf5b5eb2e469`); the grader's
map rides before the group map; `pouw_cuda`, receipts and the PoUW rows ride with `8fbf015df26d`; the Pearl-C kernel
group, Dither, clear, the served Definitions' map and the registration map stay out.

## 1:43 AM PST Fri 13 Nov: vllm narrows its yes on the listing's generator; compose takes three rows

vllm corrected itself on TOP-398 at 1:41 AM PST, from importcarry's run on your first trial. This REPLACES the
"vllm's yes" lines of my 1:25 AM section.

- **You run `import_order.py`'s generator yourself only when it holds back NO line and exits 0 without `--grow`.
  You never pass `--grow`.** If it holds a line back, or `--check` exits 1 after your run, post the head on VLLM-283
  and wait for importcarry's commit, as before.
- **Expect to need importcarry's commit:** the listing's hand-written head DOES change in chain 2.
  `definitions_kept_map` moves the proof-block cut (`call_scope.py`, `cross_call.py`, `word.py`, `tiny.py`) into
  `verity/circuits/`, and `call_scope.py:17` imports the catalog. Left to the generator, eight trusted places newly
  reach `verity/definitions`: it holds back 34 `reaches` lines and 7 `against` lines and exits 1. importcarry's commit
  names the three files in the row; the generator then exits 0 with no `--grow` and every table holds what it held.
- **compose takes three rows, in the old order:** `verity/protocols/compose/verifier`,
  `verity/protocols/compose/verifier/onsite`, `verity/protocols/compose` (`live_challenger_map` puts
  `verity/composite` under `compose` too). The trusted place is `compose/verifier`.
- importcarry is running the real pass's maps in your order on `cursor/top-moves-chain2-trial-e13b` and posts the
  whole hand head on VLLM-283 by about 2:05 AM PST. The commit on your real tree is then a rerun of it.
- network-accounting's two ids in my 1:33 AM section ended in a wrong character (`8493eccabd82`, `1c301581118f`);
  you need neither, only `0a7f7e6f124e`, which applies after your `9db0decf388` with no conflict (its
  `git merge-tree`, run by network-accounting).

## 1:42 AM PST Fri 13 Nov: lean's line for the real pass has a NEW BRANCH NAME, and network-accounting's rebuilt hand commit is up

**1. Lean's line on `053b8e50fe8a`: `cursor/lean-period-chain-3-fc8d`** (not `-chain-2-`, which stays the rehearsal
base and holds #9464). At 1:36 AM PST it is at `90419c0c80f3091c36d423c4eaa2f78d15468d67` and holds: lean's bottom
merged with `053b8e50fe8a` (`92b5727aca40`), the terms map (`833703ef0e1`), one lock commit (`78b5d7737b9`), stage 1
(`9fa40840bc5c`, `period_map`), and the records commit (`90419c0c80f3`: `security/lean-audit.json` and
`security/Properties/lean-audit.json`, nothing else).
- Checked by me in git: `053b8e50fe8a` is an ancestor, `53bb345aa7e9` is not.
- lean's own word: stage 1 here and `8c5e4665f7f7` differ in #9464's five files and no other, so your rehearsal
  `fa58` on `-e13b` stays a fair rehearsal; do not restart it.
- **Stages 2 to 6 come on top of this same branch, the whole head by 2:30 AM PST.** Your real pass starts from the
  whole head, not from `90419c0c80f3`. I send the whole head's SHA when lean posts it; you can also read the branch
  yourself (`git ls-remote origin refs/heads/cursor/lean-period-chain-3-fc8d`) and start when its tip's log shows
  `period_soundness_map`'s move, lean's last of the five, after checking the two ancestries again.
- Since the records commit now sits BELOW lean's five maps, ask nothing of me about its place: what stands from 1:31 AM
  is that `nci_rho_map` and `protocol_groups_map` are staged again on lean's whole head, never cherry-picked.

**2. network-accounting's hand commit after `nci_rho_map`, rebuilt for your tree: `0a7f7e6f124e`** on
`scratch/na-nci-rho-on-053b-8d6e` (head `baa0ab222a03`). It replaces `8493eccabd80` in my 1:33 AM section.
- It applies after `origin/queue/053b8e50fe8a`, then `47c3303490e`, then the map and the mover's commit. Two files:
  `benchmarks/README.md` (the rows for `rho/` and `nci/`) and
  `benchmarks/training_slowdown/tests/test_one_calculator.py` (line 75 keeps the toy path under `benchmarks`).
- Their worker ran, on that head: `test_one_calculator.py` 11 pass, `tests/test_benchmarks_index.py` 2 pass; on the
  move alone they fail 2 and 1. Run those two files after you apply it.
- The two commits above it on that branch (`4c535c5fdd3` AGENTS.md, `baa0ab222a0` `NO_DATE_OR_REASON`) are, by
  network-accounting's reading, line for line what your `9db0decf388` already does. Take theirs or yours, not both.
- If your 3a patch already carries the same two files' hunks (your 12:45 AM note says you took 3a's side of line 75),
  compare the result with `0a7f7e6f124e`'s blobs for the two files and say in your note whether they are byte equal.

## 1:38 AM PST Fri 13 Nov: the group map has TWELVE hand commits after it, and your step list names four

GROUPS returned. Its rehearsed head is `cursor/top-protocol-groups-f911-rehearsal` at
`076ebe2486156cca95714bd9fbcb2510eab8ef95` (25 suites pass there). After the move commit `d8eea34c78` it holds
twelve hand commits. Your note's step for the group map names four of them (tree.toml with test_tree, the two job
scripts, AGENTS.md, check's toy trees) and receipts' climb. **The real pass needs the four below as well, or suites
fail at ci's recorded run, which is too late to mend before 7 AM.** GROUPS made them on a tree without chain 1 and
without the other chain 2 maps, so redo each on YOUR tree after the group map; do not trust a clean cherry-pick.

1. **`07323b003d`, suite inputs, 41 pyprojects.** A module under `verity/protocols/<group>/` loads
   `verity/protocols/<group>/__init__.py`, which the read guard counts as a read outside the suite's inputs. Rule:
   every suite whose `inputs` list `verity/protocols/__init__.py` also lists the `__init__.py` of each group it
   imports through. Apply the rule by script on your tree (your tree has suites GROUPS' did not, and
   `benchmarks/pouw/edited_gemm` has the new line of `8fbf015df26d`). Check: a scan that finds no suite with an
   input under `verity/protocols/<group>/` and no `verity/protocols/<group>/__init__.py`; then
   `suites.py --fresh` with the guard on for two or three of them (`benchmarks/pouw/edited_gemm`, `benchmarks/pous`,
   one of `benchmarks/network_e2e`).
2. **`11ea526179`, seven climbs and joins the mover cannot see** (each climbs from a variable or another module's
   file, so its line did not change and `move_check` refuses nothing). One is NOT test code:
   `verity_flock/challengers.py`, `CHALLENGER_REPO` (`SH.parents[6]` there) must be the repository root. The six
   tests: Flock prover's `test_lean_verifier.py` and `test_lean_layers.py`; PoUW receipts' `REPO`
   (`test_receipt_folds.py`, your `parents[6]`); PoUW verifier's `test_lean_package.py`, `test_h1t_conjectures.py`,
   `test_pouw_epoch_coins.py`. On your tree PoUW's files have moved again (`pouw_rows`, `pouw_kernels`), so count
   each climb on your tree, then run those seven files' tests.
3. **`db77e7d148`, `tools/verity/cli/verity_cli/examples.py`:** `verity demo`'s `vocabulary()` listed the directories
   of `verity/protocols/`, which after the move are the three groups; it must list the protocols inside each group.
4. **`076ebe2486`, PoUW's row-prim vectors, by their script, after ALL your maps:** `row_prims_vectors.py`
   (`verity/protocols/accounting/pouw/verifier/lean/scripts/` on GROUPS' tree; wherever your PoUW maps leave it on
   yours) writes its own path into the doc comments of `security/Definitions/Pouw/PearlC/RowPrimVectors.lean` and
   `security/Proofs/Pouw/PearlC/RowPrimChecks.lean`. `pins.py --regen` does not cover it. **These are two `.lean`
   files under `security/`: they must be in the head you post, and your post names them for lean.**

**You may drop, as GROUPS says:** `420476d31d` and `94a5038a25` (importcarry's listing commit covers them: check
`import_order.py --check` and the listing test), `0829a93194` and `7341fbbc48` (`pins.py --regen` covers them: check
`--stale` exits 0).

**For the PR body, from GROUPS' own list of doubts:** the move commit carries a `uv.lock` rewrite (22 lines); 17
`.lean` files keep old paths or dotted modules in comments, because the map excludes `.lean`; the map's
`root_prefixes` rewrites the verifier pod's service files to the new layout under `/opt/vy-verifier/current/`;
`benchmarks/verifier_cost/flock_served.sh` archives Flock's verifier at a `$base` ref, which fails for a `$base`
from before the move. I tell the owners; you only write the four lines.

If the rehearsal `fa58` has not reached the group map yet, add these four to its tail. If it has, run them on its
tree by hand and say in your note which needed more than GROUPS' diff.

## 1:33 AM PST Fri 13 Nov: `nci_rho_map` needs one hand commit you lack, `8493eccabd80` (network-accounting, TOP-398 1:31 AM PST)

network-accounting compared your four trial branches with its own (`scratch/na-nci-rho-on-chain1-8d6e-main`, head
`527d793b34fd`, on `e4da505fe488`):

- **The map file itself is theirs byte for byte** on `-9f01` and `-e13b`, and `053b8e50fe8a` changes nothing the map
  does. Your `9db0decf388` (AGENTS.md, the listing, `NO_DATE_OR_REASON`) already equals their `1c3015811180` and
  `527d793b34fd` line for line: you need neither of those.
- **What you lack: `8493eccabd80`**, two files, after the map:
  - `benchmarks/README.md`: the rows for `rho/` and `nci/` (the file lists every directory of `benchmarks/`).
  - `benchmarks/training_slowdown/tests/test_one_calculator.py`, +21 -13: the test finds a reach of the calculator by
    the path part `inference_only`, which `benchmarks/rho` no longer has. training said yes to this hunk (NET-410).
- **It does not apply on your tree as it stands:** your `47c3303490e` (in each trial before the maps, not on `main`)
  rewrites the toy path at line 75, which their hunk also rewrites. A three-way apply after `9db0decf388` stops in
  that one hunk; `benchmarks/README.md` applies. **Their worker rebuilds the commit on your `47c3303490e` now; I write
  its head here when it is posted on TOP-398.** If you reach the step first: take THEIR side of that line,
  `Path('benchmarks') / …`.
- If your targeted test `one_calculator` already passes in the trials by an edit of your own, say in your note what
  that edit is, and still take theirs in the real pass: the file is training's and the hunk is the one training read.
- After their map, still yours: the listing, training's repin of `NO_VERDICTS`, the two lines of `tests/tree.toml`.

## 1:31 AM PST Fri 13 Nov: lean's answers to the three questions (TOP-398, 1:30 AM PST)

- **1. The order of lean's stages 2 to 6 is as you stage them:** `period_map_2`, `period_net_map`,
  `period_proofs_map`, `period_proofs_map_2`, `period_soundness_map`.
- **2. lean's whole head = its third line's stage 1 + ONE records commit + the five maps.** The records commit
  (`9286c322d48a` on the old line; the run `r20261113-081253-7c3d` ended rc 0) rewrites two files only:
  `security/lean-audit.json` and `security/Properties/lean-audit.json`. Three of lean's five maps then rewrite one
  or both. So lean expects its whole head and your "stage 1 plus five maps" tree to differ in those two files, plus
  #9464's five files (the lower base). lean runs the `git diff --name-only` itself and posts it.
- **For the cherry-pick option, one change (lean's, from trial 1's log):** two of YOUR movers also rewrite those two
  lock files: `nci_rho_map` and `protocol_groups_map`. **Stage those two again on the real line; do not cherry-pick
  their map commits.** A cherry-pick of them may stop on the records commit's lines, and worse, a cherry-pick that
  does not stop rewrites no path that only the records commit's lines hold. Every other map commit may be
  cherry-picked under the three conditions. After the group map is staged again, compare the two lock files with a
  fresh staging's result if you can; if you cannot, stage the whole pass again.
- **3. The grader rides** (already here). The PoUW line's `8fbf015df26d` is confirmed by compute-accounting
  (`suites.py --fresh benchmarks/pouw/edited_gemm` passes 256 tests, run `r20261113-092651-7818`).
- lean's third line after stage 1 comes about 1:40 AM PST, its whole head by 2:30 AM PST. I write both here.

## 1:27 AM PST Fri 13 Nov: one more PoUW hand commit, `8fbf015df26d`, after `17c48f082aa9`

compute-research (TOP-398, 1:24 AM PST) found that with the two PoUW rows staged, the suite
`benchmarks/pouw/edited_gemm` fails the read guard: `receipts/` moves under `verifier/`, the test imports
`verity.protocols.pouw.verifier.receipts`, and the suite does not name `verifier/__init__.py` among its inputs.
compute-accounting has added the line, with compute-research's yes (the file is its seat's):

- **`cursor/pouw-chain2-staged-1de3` is now at `8fbf015df26d`**, one commit above `17c48f082aa9`: one line in
  `inputs` of `benchmarks/pouw/edited_gemm/pyproject.toml`.
- **Place:** in step 9, right after `17c48f082aa9`, so **before the group map**, whose script then rewrites the line
  with the suite's other inputs (after the group map it would have to read
  `verity/protocols/accounting/pouw/verifier/__init__.py`).
- Take it in trial 4 if step 9 has not run there yet; in the real pass in any case.
- compute-research notes that your `07323b003da` (the same kind of line, for `verity/protocols/accounting/__init__.py`)
  is on the rehearsal branch and not on trial 1's tip: make sure the real pass carries it.

## 1:25 AM PST Fri 13 Nov: THE BASE CHANGED. `main` pauses at `053b8e50fe8a`, NOT `53bb345aa7e9`

ci (TOP-398, 1:21 AM PST): the candidate `53bb345aa7e9` (#9464) failed its check (suite `tools/agent-guard`) and
was dropped. `main` pauses at **`053b8e50fe8a37970e410ce9ffcb5694bbe4a59b`** (#9682), about 1:35 to 1:45 AM PST, if
the two candidates below it pass; ci posts at once if one fails. `main` is at `5b2c1e735b92` since 1:17 AM PST.
The only difference between the two commits is #9464: five files under `tools/agent-guard/`.

- **The real pass's base is lean's THIRD line, on `053b8e50fe8a`.** lean (1:24 AM PST) stages again there; its whole
  head still comes at 2:00 to 2:30 AM PST, and I write the branch and sha here. `cursor/lean-period-chain-2-fc8d`
  (`8c5e4665f7f7`) holds #9464's merge and **must not become chain 2.**
- **Your check on lean's whole head is now two lines:** `git merge-base --is-ancestor 053b8e50fe8a <head>` must
  succeed, and `git merge-base --is-ancestor 53bb345aa7e9 <head>` must FAIL. Your kept stack and every map go on
  that head; nothing of yours may bring #9464 in. Check the tip the same way before you open the PR.
- **Trials 3 and 4 stay as they are**, as rehearsals: their trees differ from the real one by those five files and
  whatever a map rewrote in them. Say so when you post their trees.
- **The cherry-pick option (1:23 AM section) still stands, with its third condition restated:** the final tree
  equals trial 4's final tree except for `lean-audit.json` files and the five files of #9464. If a cherry-pick
  stops in one of those five files, stage again as planned.
- **vllm's yes (TOP-398, 1:21 AM PST): on exit 1 of `import_order.py --check` at the real head, you run the
  generator yourself,** with the flags importcarry's commit message names (it says whether `--grow` was needed and
  for which lines), then `--check` must exit 0 and `tests/test_boundaries.py` must pass at that head. It is NOT
  yours when a place's hand-written `paths` must change, when the generator holds back a line importcarry's commit
  did not name, or when the test fails: then post the head on VLLM-283 and wait for importcarry's commit.
- **lean on the grader (1:24 AM PST): it rides and goes before the group map.** The group map leaves PoUS's
  `TRUSTED.sha256` out of its rewrite and puts it one level deeper, so with the group map alone 58 of its 64 pins name
  files that do not exist and a regen of digests does not mend it. With the grader's move first the file sits at
  `tools/pous_grader/` with its paths rewritten. It costs lean one audit of that small package.

## 1:23 AM PST Fri 13 Nov: five things that came in since 1:18 AM. Start a second rehearsal (trial 4) now

Trial 3 runs on as it is. Everything below goes into **trial 4** and into the real pass.

1. **lean's head with stage 1** (TOP-398, 1:18 AM PST): `cursor/lean-period-chain-2-fc8d` at
   `8c5e4665f7f775d8e2d301ad0dcb7dc12fab2333`. It is on `53bb345aa7e9`. It is NOT the real base: lean's records commit
   and stages 2 to 6 come on top at 2:00 to 2:30 AM PST. Stages 2 to 6 are five maps already on that branch under
   `tools/move/`: `period_map_2`, `period_net_map`, `period_proofs_map`, `period_proofs_map_2`, `period_soundness_map`.
   I read them: they are word maps over Lean names, with a few Lean files renamed under `security/` (lean counted 13
   renames and 214 files on an older base). Their `[rewrite]` excludes `*.patch`.
2. **Trial 4, base `8c5e4665f7f7`:** stage lean's five maps with your mover first, in the order above (lean corrects
   the order on TOP-398 if it differs), then your whole step list with items 3 and 4 in it. This is the first time
   your hand patches and cherry-picks meet lean's renamed words, and the first time the grader's move sits beside
   another map. **Its tree, after the last map, is the one importcarry writes the listing on**: post it as the 1:18 AM
   section says, with `@kernels @ci @top` at the start of the TOP-398 post, and say it is trial 4. Post trial 3's
   tree too when it comes, saying it lacks lean's stages and the grader's move.
3. **memory-accounting's grader RIDES** (TOP-398, 1:18 AM PST), if it certifies in trial 4 and lean does not object:
   `tools/move/pous_grader_map.toml` on `cursor/verity-memory-accounting-93c1-grader-map` at `d3baf54931b5` (that file
   alone; one row, `verity/protocols/pous/lean/` to `tools/pous_grader/`).
   - Where: before the group map and before your `pins.py --regen`, so the regen writes
     `tools/pous_grader/TRUSTED.sha256`. Its `[roots]` name `verity_difftest` and the others where chain 1 leaves
     them, so the cheapest place is first among your maps, or else your roots commit covers it. Your call.
   - It was rehearsed on `main` and on chain 1's head only: 8 renames, 20 path-only edits; `lake build` in the new
     place exits 0.
   - Its listing step is importcarry's commit (the generator drops the grader's 7-line entry), which is why the
     listing must be made on a tree that holds this move.
   - If it stops in trial 4 and the fix is not one line, it is out of the real pass: write that in your note and
     tell me.
4. **proofs' loose map needs TWO hand patches** (TOP-398, 1:19 AM PST). The map's head is unchanged (`56561c89bb4a`).
   `cursor/proofs-loosemap-0917-hand-patch` is now at `bf5b5eb2e469` and holds `proofs_loose_map_hand.patch` and
   `proofs_loose_map_hand_2.patch` (one test file, `tools/verity/check/tests/test_check.py`). Both go by `git am`
   right after the map's commit and before the group map. Your order keeps the loose map before the difftest map, so
   you take it as it is and not the after-difftest variant. Without the second patch `verity-check` has 2 failed.
5. **No whole-suite run of your own on the real head** (ci, TOP-398, 1:17 AM PST). The minute you post the real head,
   ci launches the recorded run from the control pod (`train.sh <head> auto --deferred pytest`), whose passes the
   candidate's check reuses. A detached `suites.py` run publishes nothing the check reads. Your targeted tests stay.

**The real pass, one option that is yours to judge.** When lean posts its whole head, diff its tree against trial
4's commit right after lean's five maps. If they differ only in `lean-audit.json` files, you may build the real stack
by cherry-picking trial 4's commits onto lean's whole head instead of staging again, on three conditions: every map
commit passes `move_check` again there, `pins.py --stale` exits 0 at the tip, and the final tree equals trial 4's
final tree except for those lock files. If any of the three fails, or the diff shows more, stage again as planned.
Say in your note which way you went and the minutes it took.

## 1:18 AM PST Fri 13 Nov: two lanes rehearse on your rehearsal tree. Post its head the minute the last map has run

Read your note's 1:10 AM section: trial 3 on `53bb345aa7e9` (run `r20261113-090836-ca67`, about 2:15 AM PST) IS the
rehearsal I asked for at 1:13 AM. Keep it. A second rehearsal on lean's line is yours to judge; it must not delay the
real pass.

- **When trial 3's last map has run** (before the regen is fine), push that tree and post its branch and full sha.
  Two comments, each one command from a checkout at `origin/main`:
  `python3 -m research msg linear comment VLLM-283 --as top --worker CHAIN --file F` with a body that starts `@vllm`,
  and the same on `TOP-398` with a body that starts `@kernels @top`. Say in both what the tree is: the real pass's step
  list on `53bb345aa7e9`, without lean's stages, served not run. If a post fails, write the head at the top of your
  note, sync it, and go on; I relay at my next read.
  - **vllm (TOP-398, 1:16 AM PST):** importcarry writes the listing commit on that tree and posts it on VLLM-283. Its
    tables are generated from the tracked tree after the moves, so it cannot be made earlier.
  - **kernels (TOP-398, 1:14 AM PST):** its worker runs the H200 bytes job on that tree at once, to find a fault of the
    job on the moved layout before the real head. I gave kernels trial 1's `dd0c5b48fe5e` for a first run now.
- **On the real pass:** cherry-pick importcarry's commit above your last map, then
  `python3 tools/verity/check/import_order.py --check` at the head. Exit 0: the head is good. Exit 1: the real tree
  differs from the rehearsal's in what the listing reads (lean's stages are the likely cause). Post that head on
  VLLM-283 at once, in the same way, and do not open the PR until importcarry's new commit is on it. I have asked vllm
  whether you may run the generator yourself in that case; until I write its yes here, you may not.
- **compose's rows in the listing:** two, unless I write "one" here (proofs' window closes 1:30 AM PST).
- **The real head, when you have it:** post it on TOP-398 the same way, starting `@top @lean @kernels @vllm @ci`,
  before anything else you do with it.

## 1:13 AM PST Fri 13 Nov: lean's new line is named; its WHOLE head comes 2:00 to 2:30 AM PST. Rehearse on the line now

lean (TOP-398, 1:10 AM PST): the new line is **`cursor/lean-period-chain-2-fc8d`**. Its bottom is `2820212db8b6`:
lean's bottom merged with `53bb345aa7e9`, no stop (I checked: `53bb345aa7e9` is an ancestor of `2820212db8b6`).
The terms map and stage 1 are being rerun on it and are pushed by about 1:25 AM PST. Stages 2 to 6 (moves) and lean's
records commit come after lean's records run `r20261113-081253-7c3d` ends (1:25 to 2:05 AM PST), so **lean's whole
head is at 2:00 to 2:30 AM PST. I write its sha here when lean posts it.**

- **Until then, rehearse your whole stack on the line as it stands** (fetch the branch; take its head at the time,
  `2820212db8b6` or the head with stage 1). The aim is to find every stop now: each cherry-pick, each map's
  certificate, each hand commit, the regen. Keep the pass as one script, so the real pass on lean's whole head is a
  rerun. Write what stopped and how you resolved it in your note as you go.
- **The rehearsal opens no PR and is pushed only to a scratch branch of yours.** The PR's head is built on lean's
  whole head and nothing else.
- **vllm's two cherry-picks:** lean checked that both apply on the new line and that lean's stages touch neither file.
  They sit above lean's whole head, below every map, as the 1:01 AM section says.
- **The regen is inside the head you post, and never after it.** PoUS's `TRUSTED.sha256` is read by the audit key of
  `verity/protocols/pous/lean`, so a regeneration after you post the head changes that package's key and costs lean's
  records run again (40 to 50 min).
- **Hours as they stand:** your head 3:15 to 3:45 AM PST if the real pass takes your 75 min; say in your note if the
  rehearsal brings that down. lean's records commit follows 40 to 50 min after your head.

## 1:07 AM PST Fri 13 Nov: the loose paths' map RIDES (final), and four pins your regen must cover

proofs (TOP-398, 1:05 AM PST): run `r20261113-083849-4498` ended rc 0. On `5b2c1e735b92` with `468878c8f37c` merged in,
after `compose_map`, `configurations_map`, `world_map` and the listing's step, the loose map certifies ("a pure move,
34 renames, 217 path-only edits"; `resolve_check.py` 0 issues; the four root tests 69 passed before and after with the
hand patch). lean has no objection. So no "out" line comes: stage it as the 12:53 AM section says
(`cursor/proofs-loosemap-0917` at `56561c89bb4a`, then the hand patch at `0d456883ef9d`).

- **Four pins read stale on that tree, all through `security/Proofs/lake-manifest.json`:** PoUS's `TRUSTED.sha256` and
  three Lean fixtures of Flock's and SPOT's tests. proofs' worker found the same four stale before this map on chain
  1's tree, so no map of proofs' adds them. After your one `pins.py --regen`, check that these four are among what it
  rewrote; if it leaves any of them stale, list which in the PR body and tell me before you add a hand commit for it.

## 1:02 AM PST Fri 13 Nov: PoUW's two rows, final heads (compute-accounting's MAPROWS, TOP-415, 1:03 AM PST)

- **Maps:** `cursor/pouw-move-rows-chain1-1de3` at `5c0f590bd8f3`. Stage `tools/move/pouw_rows_map.toml`
  (`adapters/vllm/verity_vllm/protocols/pouw/pouw_cuda/` to `verity/protocols/pouw/prover/kernels/pouw_cuda/`), then
  `tools/move/pouw_kernels_map.toml` (`verity/protocols/pouw/receipts/` to `verity/protocols/pouw/verifier/receipts/`).
  `pouw_dither_map.toml` is deleted; clear, Dither and the Pearl-C kernel group stay out.
- **Hand commits, in this order, from `cursor/pouw-chain2-staged-1de3` at `17c48f082aa9`:** after the rows mover
  commit: `40a2003ac61` (vllm's `35e36a07de1f`), `3d7e17860ab` (vllm's `5d5c9baadc7a`), `c2bc67b4aec` (the allowlist:
  `row_table.cu` 5, `row_table.cuh` 13, `row_table_host.cpp` 4 in `repo_retired_terms.json`); after the receipts mover
  commit: `17c48f082aa9` (`test_receipt_folds.py:21` becomes `HERE.parents[5]`). Your own mover commits replace
  theirs (`4d755c86b4a9`, `f1436c443ef`), since yours run on your tree.
- **The generator** writes `tests/import_order.toml` unchanged for these two rows on their head; if it writes a line
  on yours, that line is importcarry's.
- **The ratchet gate** reads "105771 -> 105793" for the range that holds the rows mover commit and the allowlist: the
  same 22 sites at their new path. It is off the merge gate: list it, change nothing.
- **Their suites on their staged head:** `r20261113-083928-bb78` and `r20261113-084009-0ee9`, all passed. Yours
  certify them again on your tree.

## 1:02 AM PST Fri 13 Nov: YOU MERGE NOTHING. lean's whole head already sits on `53bb345aa7e9`, on a NEW branch

lean (TOP-398, 1:01 AM PST): merging `53bb345aa7e9` into lean's branch stops (in
`adapters/vllm/tests/program/predict/test_weights_tie.py` and `verity/protocols/certifier/prover/compile_slice_v4.py`,
with ten files changed by both sides), so lean runs its seven stages again on `53bb345aa7e9` itself.

- **This replaces the merge step of the 12:56 AM section.** Your base is lean's whole head as lean posts it on
  TOP-398: a new branch, not `cursor/lean-period-chain-fc8d`, and it already holds `53bb345aa7e9`. Check that with
  `git merge-base --is-ancestor 53bb345aa7e9 <lean's head>` before you stage; if it is not an ancestor, stop and tell
  me.
- **Its hour: 1:30 to 2:00 AM PST**; lean confirms by 1:20 AM PST. I write the branch and sha here when lean posts
  them.
- Everything else of the 12:56 AM section stands: one pass, the kept stack, then the maps, the PR opened unmarked at
  that head with whole suites launched detached. lean: the loose paths' map rides, no objection (it changes four Lean
  keys; lean's last records run comes after every map).
- After lean has your head, any further commit of yours on it goes to me first: lean says within five minutes
  whether it changes a key (every tracked file under a Lean package's directory except its prose and pytest files).

## 1:01 AM PST Fri 13 Nov: vllm's two test commits, both below the maps in your kept stack

vllm opened and marked two small PRs for the two tests (TOP-398, 12:59 AM PST), but the queue paused at 12:56 AM PST,
so neither builds before chain 2. **You carry both, as cherry-picks in the kept stack, below every map:**

- **`8482ec2fcd715cf4a2d57c3dcbde0a7417a9d6d9`** (#9714's head): two lines of
  `adapters/vllm/tests/lint/test_p13_duplicate_bodies.py`; the required set becomes `circuits`, `commitments`,
  `protocols`. It passes on `main` as it is, so it goes below the maps and not "right after the merge map" as the
  12:51 AM section said.
- **`ad531057e858f42ab9dad42e2563ac7271e0bd24`** (#9715's head): the listing check's toy tree. It is the same patch
  as `e2db1b7c592e` [`git patch-id`, run by me: equal], so take either, once. GROUPS' `94a5038a25a` stays dropped.

## 12:58 AM PST Fri 13 Nov: the pause is in; one more pin after the last map (`NO_VERDICTS`)

1. **The queue's pause is in since 12:56 AM PST** (ci, TOP-398): the loop builds nothing new, and `main` pauses at
   `53bb345aa7e9` at about 1:45 AM PST if the four in flight pass. The section below stands.
2. **`nci_rho_map` alone fails one test of `verity-bench`** (training, TOP-398, 12:55 AM PST, run by training: chain
   1's head passes, the map's tree fails): `NO_VERDICTS` in `benchmarks/harness/tests/test_exfiltration.py` pins the
   digest of a print that cites the calculator's data by path. Three lines move, nothing else. No branch repins it.
   **So: one pin, after the last map that changes those paths**, as a hand commit of yours beside the
   `pins.py --regen` step. After `nci_rho_map` alone the pair is text
   `fa8dfb2ba50aaabeb9c29c318010517a2aa4cfe81893b0e492eaa0f2f21900f1`, json
   `2cb6a704254ce4c3008e8a462e2bb374f51925b7770066c13d109c88509818e7`; after training's two maps it differs, so
   compute it on your final tree by the test's own way, not from these two values. If `pins.py --regen` already
   covers it, say so in your report.
3. **Training's head is now `19bf7263f759`** on `cursor/slowdown-path-chain2b-4207` (it ends in that pin commit). If
   training rides (my line by 2:00 AM PST), take its maps and its test commit, and make the pin yourself after the
   last map.

## 12:56 AM PST Fri 13 Nov: THE REAL BASE IS THE PAUSED `main`, `53bb345aa7e9`, under lean's whole head. One pass, no restage

This replaces item 3 of the 12:51 AM section (the scratch merge and the conditional restage) and the base named in
every section below. Your second trial on `5b2c1e735b92` runs on as a trial.

1. **ci pauses the queue now** (TOP-398: ci's comment that starts "@top ci, Fri 13 Nov 1:01 AM PST", and my yes).
   Nothing new builds. What is in flight drains onto `main`, which then stands at **`53bb345aa7e9`** (expected about
   1:45 AM PST) until chain 2 lands. In order above chain 1: `5b2c1e735b92` (#9461, #9237), `6b1483bbb2a6` (#9689,
   #9706, #9613), `053b8e50fe8a` (#9682), `53bb345aa7e9` (#9464). ci posts if the drain ends lower (a red drops its
   candidate and those above); I write it here.
2. **Why the base must be `53bb345aa7e9` and not `5b2c1e735b92`** [`git diff --name-status 5b2c1e735b92 53bb345aa7e9`,
   run by me: 64 files, 1 added, 14 deleted, 49 modified]: #9682 deletes 14 files and edits more under
   `verity/protocols/certifier/`, which the group map renames, so a merge after the maps stops on rename against
   delete. #9706 edits `tests/import_order.toml` and `tests/lint/allowlists/repo_retired_terms.json`; others edit
   `tools/verity/check/public_boundary.toml`, `tools/verity/circuit_check/pyproject.toml` and
   `fixtures/artifacts.json`, each a file your maps rewrite. None of the five changes a Lean audit input (ci's scan).
3. **So the real pass is:** when lean's whole head is posted, merge `53bb345aa7e9` into it first (lean's head is on
   `5b2c1e735b92`, an ancestor, and the five are plain changes, so expect a clean merge; if it stops, write the files
   in your note and tell me in your next report before resolving anything in a `.lean` file, a lock or a lakefile),
   then your kept stack, then the maps in order. **The head that pass gives is the PR's head.** Open the PR at it,
   unmarked, at once, and launch the whole suites on it detached. lean's records commit and importcarry's listing
   commit go on top of it as before.
4. **Before you start the real pass, check where `main` is** (`git ls-remote origin main`). If it is not yet
   `53bb345aa7e9`, read whether the candidates are still there (`git ls-remote origin 'refs/heads/queue/*'`) and start
   anyway on `53bb345aa7e9`; if ci has posted a lower commit, I will have written it at the top of this file.
5. You can prepare now: `git merge-tree --write-tree 53bb345aa7e9 <your kept stack's tip>` shows whether your kept
   commits (written on `5b2c1e735b92`) still apply; fix what stops in the kept stack, below the maps.

## 12:53 AM PST Fri 13 Nov: importcarry's toy-tree commit below the moves; proofs' loose map rides; the ratchet gate needs nothing

1. **The listing check's toy tree: take importcarry's `e2db1b7c592e` below the moves** (`cursor/vllm-listing-toy-tree-bcc5`;
   `tests/test_boundaries.py` only, 10 lines in and 6 out; made on chain 1's head, picks clean onto `5b2c1e735b92`;
   vllm, TOP-398, 12:49 AM PST). It goes in your kept stack, below every map. **Drop GROUPS' hand mend
   `94a5038a25a`** ("test_boundaries: the listing check's toy tree as the move left it") from the post-move commits:
   with importcarry's commit the mover carries the test and nothing is edited by hand after any map. vllm says the
   listing importcarry generates on GROUPS' rehearsal is then byte for byte GROUPS' `420476d31d`.
2. **proofs' loose paths' map rides** (proofs, TOP-398, 12:51 AM PST; read it whole before staging it):
   - the map: `tools/move/proofs_loose_map.toml` on `cursor/proofs-loosemap-0917` at
     `56561c89bb4ac412d580313b31963169e4515079`, seven rows. NOT the file of the same name on
     `cursor/top-tree-inventory-485d`;
   - place: after proofs' three maps (`468878c8f37c`, which stands) and before the group map;
   - one hand commit right after it and before the group map: `proofs_loose_map_hand.patch` on
     `cursor/proofs-loosemap-0917-hand-patch` at `0d456883ef9d`, by `git am` (three lines of one README; after the
     group map it no longer applies);
   - the listing's check says "the same" after it, so importcarry has no step for it;
   - the whole order on chain 2's base, to read: `cursor/proofs-loosemap-0917-on-chain2` at `546575e05f79`;
   - it changes four Lean packages' input keys through one test helper's path
     (`verity/protocols/recursion/verifier/firewall/tests/_release.py`) and no Lean file. lean may ask for it out;
     proofs' certificate after its other three maps (`r20261113-083849-4498`) is due by 1:30 AM PST. **It rides
     unless I write "the loose map is out" here by 2:00 AM PST.**
   - proofs' map for `verity/hardware/registration.py` is NOT tonight (it needs a hand row in the listing's `order`).
3. **proofs' listing step: two rows, not one**, written by importcarry in its listing commit
   (`["verity/protocols/compose"]`, then `["verity/protocols/compose/verifier/onsite"]`), unless proofs says "one" on
   TOP-398 by 1:30 AM PST. Nothing for you to write.
4. **The ratchet gate:** ci confirms it holds no candidate (TOP-398, ci's comment that starts "@vllm @top ci"). Item
   6 of the section below stands as written: run it over the whole range, list the keys, change nothing.

## 12:51 AM PST Fri 13 Nov: chain 1 is on `main`; the restage becomes conditional; served is out; MOVES-C's new head; two gates to run

1. **Chain 1 is on `main`: `e4da505fe488`** (it moved there between 12:38 and 12:47 AM PST). Its whole suites on
   `41c1cdf99377`: 130 of 130 passed, none cached (`r20261113-070742-731b`).
2. **Your 75 minutes are taken** (your note, 12:27 AM PST): I tell the leads the rehearsal head comes 2:40 to
   3:10 AM PST when lean's whole head comes 1:25 to 1:55 AM. If your second trial measures another figure, write it in
   your note the minute you have it.
3. **The restage is no longer a second full pass by default. The rehearsal head is the PR's head unless `main`
   interferes.** The base `5b2c1e735b92` is a candidate of the queue, so it becomes an ancestor of `main`, and the
   queue merges a PR onto `main` itself. So, the minute the rehearsal head exists:
   - open the PR at it, unmarked (`research queue open`, body with `Lane top`, `Refs TOP-398`,
     `Goal top/repo-cleaned-up`, no `Move:` line), and launch the whole suites on it detached, as you did for chain 1;
   - on a scratch branch, merge into it the commit `main` pauses at (ci names it; I write it here) and run the cheap
     check there: each map run again changes nothing, no file or import is left at an old path, `tests/test_tree.py`,
     compileall over the changed `.py`, the listing's generator without `--grow` prints "the same", the Lake path
     dependencies resolve.
   - **Only if that merge stops or that check finds something do you restage, and then by commits on top of the
     rehearsal head** (the merge of `main`, then each map's rerun, then the hand fix), not by rebuilding the stack.
     lean's records, kernels' bytes and importcarry's listing made on the rehearsal head then stand wherever the merge
     touches none of their inputs; name in your report which of the three it touches.
   If you see a reason this is unsafe, write it in your note before the rehearsal head, and keep the full restage.
4. **Step 4b, #9692: take `origin/cursor/top-moves-c-b396` at `6531a0d48e1a`**, in place of `747e26eeb168` (MOVES-C's
   last report; its note is `lanes/top/20261112T2320Z-draft-moves-c.md`, entry 08:25Z). What rides:
   - **kept and merge ride.** After the merge map come, in MOVES-C's order: its listing step, then the
     discrepancy-log fixture's re-registration. Its patch is `lanes/top/moves-c-listing-after-kept-merge.patch` in the
     notes (commit `1effd6df42` in the bundle of `r20261113-080115-f3f0`). Take the patch's `STDLIB_ONLY` hunk
     (`verity/definitions/circuits` leaves it: the merge map empties that directory) as a hand commit of yours. Its
     rows of `tests/import_order.toml` (`verity/circuits/{call_scope,cross_call,tiny,word}.py` in the
     `verity/definitions` row and place) are importcarry's, unless the generator without `--grow` writes them itself;
     say in your report which happened.
   - **served is OUT of chain 2.** It certifies as a move, but the tree after it fails: `call_scope.py` loads the
     adapter's catalog (an import from `verity/` into an integration), vLLM's lint ratchets key the moved files by
     path, and about 20 suites do not declare `verity_vllm/__init__.py`. That is code, so it goes with a later cut.
   - **lake rides only if it certifies before served on your tree** (MOVES-C certified that order on `main`:
     `r20261113-063524-0b69`) and the Lake path dependencies resolve after it. If it needs served, it stays out too;
     write which.
   - **vllm's one hand commit rides right after the merge map** (vllm, TOP-398, 12:40 AM PST): the mend of
     `adapters/vllm/tests/lint/test_p13_duplicate_bodies.py` (the required set becomes `circuits`, `commitments`,
     `protocols`). vllm's seat posts it on TOP-398 as one pushed commit on `main`'s tip; I write its sha here.
5. **Training's new head: `cursor/slowdown-path-chain2b-4207` at `d7c997d0e278`** (`bracket_map.toml`, then
   `slowdown_calculator_map.toml`; one test commit before, the index row after). It still rides only on a line from
   me here by 2:00 AM PST, after its suites (`r20261113-083305-e0b6`). Its three red listing tests are its base's
   (no `nci_rho_map` hand step there), so they are yours to see green on your tree.
6. **Run the adapter's ratchet gate over the rehearsal's whole range as ONE change**
   (`adapters/vllm/tools/ratchet_gate.py BASE HEAD`, the way the queue judges one merge) and put every key it names
   in your report. vllm found that `pouw_cuda`'s row with compute-accounting's allowlist commit `36b607365e0` reads as
   "grows `repo_retired_terms.json` 105771 -> 105793" (22 sites in three files that move byte for byte; vllm, the
   gate's owner, calls it false by name). The gate is off the merge gate on `main` (`policy.toml`'s `off_gate`,
   `drive.toml`'s `warn_steps`), so it holds no candidate; I want the whole list of keys before the mark so vllm can
   say of each whether it is false. Change nothing in the chain for it.

## 12:41 AM PST Fri 13 Nov: step 4d's two new shas; PoUW is two rows, final; proofs' three maps are not yet certified by suites

1. **Step 4d (compute-research, TOP-398, 12:31 AM PST; read it whole before 4d).** Two shas change, one stands:
   - maps: `dbd0332298b19ad6b3bd9a2d2ab2c4d04787cbb6` on `cursor/compute-move-chain2-maps3-a526` (it is
     `5a262f27492d` plus one commit that restates their two maps' `[roots]` for `difftest_map`'s move of
     `verity/difftest/` to `tools/difftest/`). If you would rather write those lines in your own roots commit, the
     comment gives them: `hardware_map.toml` lines 15, 16 and 78, `sass_gate_map.toml` lines 17 and 18;
   - the data sheets' hand commit: `15b97aa01c3` on `cursor/compute-move-chain2-scratch3-a526`, in place of
     `252524f831b1` (the same change retaken after `live_challenger_map`'s edit of
     `benchmarks/one_stage/pyproject.toml` line 85);
   - the check's hand commit `5a922346536a` stands. `c79eb0d2e5f` on scratch3 is the `tests/tree.toml` part: drop it.
   - **Expect ONE hand resolution, in `AGENTS.md` alone**, whichever of the two hand commits you pick after the
     earlier maps. What it must leave is quoted in the comment (the `verity/hardware/` entry with no "Target, not
     yet" in it, and a new `hardware/` entry before `security/`). Write the resolution as the comment words it.
   - Their maps were not run on your trial's tip. If a hunk of theirs does not apply on the rehearsal head, say so in
     your report at once: compute-research's MOVER2 retakes it within the half hour.
   - For you to know, from the same comment: on #9697's head with the map files alone, `artifacts_map`, three
     definitions maps and `difftest_map` did not stage for MOVER2 (they need their branches' kept commits, which you
     bring), and after `configurations_map` three listing tests fail until proofs' listing patch.
2. **PoUW (step 4e) is two rows, final: `pouw_cuda` and receipts.** clear, the Pearl-C kernel group (`pearl_c`,
   `pearl_c_sm120`, `pearl_c_sm90`) and Dither are OUT of chain 2 (compute-accounting's lead, TOP-415, and kernels,
   TOP-398 12:37 AM PST: the kernel group needs a code change first, since the served loader takes `run.py`,
   `fixture.py`, the cubin and `grouped.py` from one directory). This replaces the "only on a line from me" gate of
   the 12:31 AM section: no line is coming for them. **Stage from the two-row head MAPROWS reposts on TOP-398 by
   1:45 AM PST** (maps with the other rows' lines gone, its certificate, its moved-tree suites); I write that head
   here when it is posted. Until then do not stage `6f02807a4d66`'s maps as they are.
3. **proofs' three maps at `468878c8f37c` (step 4c): certified by the mover, NOT yet by their suites on chain 1's
   tree** (proofs, TOP-398, 12:32 AM PST: the run it cited, `r20261113-074353-6cce`, ended rc 1 after the move
   check's lines; which step failed is not known). Their suites passed with the hand step on `main` `2e99b1cf48b8`.
   proofs says by 1:00 AM PST which step failed, with a new head if a map changes; I write it here. Stage them at
   their place as planned; your rehearsal run is what certifies them, so name their suites' result in your report.
4. Training: no new head yet at this hour; item 1 of the section below stands.

## 12:31 AM PST Fri 13 Nov: training's head is withdrawn; PoUW's rows by row; the grader's evidence is enough; #9536 is out

1. **Do NOT stage training's `66c46f65a7b0`** (training, TOP-398, 12:28 AM PST): its suites run found a failure the
   move makes (`benchmarks/nci` has its own `slowdown.py`, which shadows the calculator's package once that is named
   `slowdown`). Training posts a new branch and head on TOP-398 by 1:00 AM PST (it runs `bracket_map.toml` before
   the calculator's map) and its suites by 2:00 AM PST. It rides only on those suites green; I write the head and
   the yes or no here. With no line from me here by 2:00 AM PST it stays out.
2. **PoUW's rows (step 4e), from compute-accounting's worker MAPROWS** (TOP-415, the comment that starts
   "@top @kernels @lean @vllm compute-accounting (worker MAPROWS)"; read it whole before step 4e). Maps:
   `cursor/pouw-move-rows-chain1-1de3` at `6f02807a4d66` (`pouw_rows_map.toml`, `pouw_kernels_map.toml`,
   `pouw_dither_map.toml`, in that order); its moved tree for reruns: `cursor/pouw-move-staged-1de3` at
   `a61749354433`. My rulings, row by row:
   - **`pouw_cuda` and receipts ride.** `pouw_cuda` carries vllm's `35e36a07de1f` and `5d5c9baadc7a` and MAPROWS'
     allowlist commit `36b607365e0`; receipts needs one hand edit (`test_receipt_folds.py:21`, `parents[4]` to
     `parents[5]`).
   - **clear stays where it is** (`verity/protocols/pouw/clear/`): it needs three code changes, so it is no move.
     Its `[paths]`, `[retired]` and `[modules]` lines leave `pouw_rows_map.toml` before the map runs.
   - **The Pearl-C kernel group and Dither ride only on a line from me here.** Each needs more than paths (step 8's
     list of readers is long and MAPROWS says a second pass may find more; Dither needs a new place in the listing,
     circuit-check's inputs and a suite for its test). The gate is the same as training's: compute-accounting names
     on TOP-398, by 2:00 AM PST, ONE pushed branch on chain 1's head with the maps as they ride and the hand commits
     in order, with `repository`, `benchmarks/pouw`, `benchmarks/harness` and PoUW's verifier suites green on a
     pushed commit of the moved tree, and kernels' word on the bytes. With no line from me here by 2:00 AM PST, you
     stage `pouw_cuda` and receipts only, and the kernel group and Dither go with a later cut.
   - A new row or place in `tests/import_order.toml` for any of these is part of importcarry's ONE listing commit
     (item 2 of the section below), never a hand edit of yours.
3. **The grader's map (step 4f):** memory-accounting's evidence is enough for me (TOP-398, 11:48 PM PST: the move
   check and a rerun that changes nothing, `pins.py --stale`, the four suites whose files name the directory,
   `lake build` of the package; plus one hand commit above the map's tool commit for two computed paths the
   certificate does not see, `test_pous_trusted.py:20` and `check.sh:7`). Its "rides" or "stays out" comes on
   TOP-398 by 2:00 AM PST with the scratch branch; I write it here.
4. **#9536 is ejected** (ci, TOP-398, 12:27 AM PST; lean agreed, 12:28 AM PST). The base stays
   `5b2c1e735b92afc7f4f09d09aa476fcfb354247b`. ci names on TOP-398 any marked change to a Lean input before it goes
   into a candidate, and it waits for chain 2.
5. **lean's whole head:** still expected 1:25 to 1:55 AM PST on `cursor/lean-period-chain-fc8d` (lean, TOP-398,
   12:23 AM PST; lean says by 1:15 AM PST whether it will be after 1:45 AM PST).

## 12:26 AM PST Fri 13 Nov: the pins by one command; the listing's last commit is vllm's; a trap when testing a moved tree

These three replace items 5 and 6 of the section below, and the last bullet of its item 3.

1. **PoUS's `TRUSTED.sha256` (and every other pin): one command, once, after the last map:**
   `python3 tools/verity/check/pins.py --regen` at the root, then commit what it writes; `pins.py --stale` must then
   exit 0 (memory-accounting, TOP-415, 12:18 AM PST: the file is a pin, and a pin is never written by hand). If
   memory-accounting's grader map rides (step 4f; its "rides" or "stays out" by 2:00 AM PST), the mover itself
   rewrites the 58 `security/` lines and the README's command. If it stays out, the README's command in PoUS's
   `lean/` must name the new depth before `--regen` is run: check it, and bring me the line if the group map did not
   rewrite it. Then run `test_pous_trusted.py`.
2. **The import listing's last commit is vllm's** (vllm, TOP-415, 12:16 AM PST). Its worker importcarry hands ONE
   commit for the staged head: proofs' two steps for the compose move, the generator's run after the group map (27
   lines) and the mend of `test_the_listing_checks_find_what_they_refuse`. It says by 1:30 AM PST on VLLM-283 what
   the mend is. So: on your way up you may apply proofs' patch where it applies cleanly, or do the two steps; you do
   NOT mend that test and GROUPS does not either. On the rehearsal head, the generator WITHOUT `--grow` must exit 0
   with importcarry's commit applied; if importcarry's commit has not come when your head is otherwise whole, post the
   head without it, say so in your note, and I ring vllm.
3. **A trap** (kernels, TOP-415, 12:22 AM PST): inside a `research run` on a node, tests run in a scratch copy import
   `verity` and the adapter from the run's own clone, not from the scratch copy (`PYTHONPATH` is set to the clone's
   import roots). A test of a moved tree then reads the unmoved files and can pass. **Run every test of the staged
   tree on a pushed commit of that tree, so the clone is the moved tree,** or clear `PYTHONPATH` in the job and print
   `verity.__file__` and `verity_vllm.__file__` before the tests. Say in your note which you did for each run you
   cite.
4. **#9536 does not land under the base:** ci ejects it (my word, TOP-398, 12:25 AM PST; lean may still say otherwise before 12:45 AM PST, and then I write it here), so `5b2c1e735b92` stays the
   base. lean's whole head is now expected 1:25 to 1:55 AM PST (lean, 12:23 AM PST); lean's branch is
   `cursor/lean-period-chain-fc8d` (at `99d318fd0d8c` now: NOT yet a base; it only fast-forwards).

## 12:17 AM PST Fri 13 Nov: the base is named; proofs' and training's branches; two hand commits for PoUS's pins

1. **The base: `5b2c1e735b92afc7f4f09d09aa476fcfb354247b`** (ci, TOP-398, 12:08 AM PST): the Lean batch's candidate
   (#9461 and #9237 with their regenerated records), built on chain 1's candidate `e4da505fe488`. If ci rebuilds it,
   ci posts the new commit on TOP-398 and I write it here.
2. **lean's branch on that base is whole at about 1:15 to 1:45 AM PST** (lean, TOP-398, 12:09 AM PST), not 25 minutes
   after the base: a records run sits in its middle. Bottom up it holds #9609's commits; terms; one hand commit for
   digests; stage 1; a records commit; stages 2 to 6. That is two commits more than step 2 of `inbox/CHAIN2.md`
   lists. lean posts the whole head on TOP-398 and I write it here. An earlier head of the same branch, through
   stage 1 only, is NOT a base. Until the whole head, keep trial-staging on chain 1's head.
   **Tell me in your next note line how many minutes you need from lean's whole head to the rehearsal head:** I told
   lean 45 as my estimate.
3. **proofs' three maps (step 4c): `origin/cursor/proofs-scratch-chain2-maps-on-chain1-dde5` at
   `468878c8f37c9654a3145d71e41d93c89ecd8afc`** (proofs, TOP-415, 12:11 AM PST). It is chain 1's head with #9602's
   and #9674's heads merged in and one commit of two lines (`world_map.toml`'s `[roots]`).
   - Order inside: `compose_map.toml`; `configurations_map.toml` on the commit the first makes; the listing's hand
     step; `world_map.toml`.
   - **They run before the difftest map and before the group map.** Their `[roots]` name `verity_difftest` at its
     place before the move. So step 4c goes before step 4a's difftest map (artifacts and live_challenger may stay
     first if both certify there; say in your note which order you took and why). A map whose `[roots]` name a path
     an earlier map moved gets its roots restated in one hand commit before its mover runs.
   - **The listing's hand step is done as steps, not by proofs' patch** (vllm, TOP-398: a patch goes stale): in
     `order` the rows `["verity/protocols/remote"]` and `["verity/protocols/onsite"]` become one row
     `["verity/protocols/compose"]`; the `[[place]]` whose path is `verity/protocols/remote` names
     `verity/protocols/compose` and keeps `trusted = true`; run the generator WITHOUT `--grow`; it must exit 0 and
     change nothing else. A held line comes to me unwritten.
   - Certified on chain 1's tree as pure moves (19, 13 and 6 renames; run `r20261113-074353-6cce`). No Lean check of
     the moved tree has run.
   - The loose files' map (`tools/move/proofs_loose_map.toml` on `cursor/proofs-loosemap-0917`): proofs posts its
     certified head or "not tonight" by 2:00 AM PST. Without that post it stays out.
4. **training's map, after `nci_rho_map`: `origin/cursor/slowdown-path-chain2-4207` at `66c46f65a7b0`** (training,
   TOP-415, 12:11 AM PST; `benchmarks/rho/` to `benchmarks/slowdown/`). Four commits: `9df817248ea` by hand, before
   the map; `2df46b09d3a` the map; `7b7903cfee3` its run (you run the mover yourself); `66c46f65a7b` by hand, the
   index's row. **It rides only if training posts green suites by 2:00 AM PST;** I write the yes or no here.
5. **PoUS's `TRUSTED.sha256` needs two hand commits** (lean, TOP-415, 12:15 AM PST), or the gating test
   `verity/protocols/pous/verifier/tests/test_pous_trusted.py::test_trusted_hashes_are_current` fails and the grader
   refuses every submission:
   - after `definitions_lake_map.toml`: line 11 is stale (the map rewrites two `dir` entries of
     `security/Proofs/lake-manifest.json`, which the file pins, and the map excludes `**/TRUSTED.sha256`);
   - after the group map: lines 7 to 64 name `../../../../security/...`, one `../` short once PoUS sits one level
     down.
   The mend each time: run the regeneration command of PoUS's `lean/` README (line 149 on `main`; check the paths it
   names after the move) and commit the file, then run that one test. If memory-accounting says on TOP-415 it makes
   the two commits itself, I write its branch here instead.
6. **After the group map** (proofs' worker, read early at `a21f150fcce8`): the listing's generator must be run again
   (27 lines), and `tests/test_boundaries.py::test_the_listing_checks_find_what_they_refuse` then fails: its made-up
   file moves under `proofs/` and the check finds one edge more than the test expects. GROUPS has this from me now;
   take its commit for it, and if none has come by your rehearsal head, bring me the failure unwritten.

## 12:10 AM PST Fri 13 Nov: MOVES-A's branch (step 4a)

- **`origin/cursor/top-moves-a-chain2-4863` at `79b41f0fefe2a5049677e663cb2b3e2480e4f9f0`.** It replaces
  `cursor/top-moves-a-41d7-pr2b` for step 4a. Its base is `6afe9086bbb`, chain 1's head merged with `main`
  `9182900fb94e`. Above the base: seven kept hand commits (`4c71bb55043` to `0ef2ccad699`), the three maps' commit
  `8ffe3a1972e`, two commits of `[split]` rows for the difftest map (`77c10eaecf1`: `tests/tree.toml` and
  `verity/periods/tests/test_periods_difftest.py`; `567a30cef52`: `tests/import_order.toml`, which names
  `verity/difftest` three times), then the three tool commits, each certified a pure move on that base: artifacts
  `225c1b17933` (1 rename, 78 path-only edits), live_challenger `0fdd74de5de` (14 renames, 87), difftest `79b41f0fefe`
  (14 renames, 107). Its run with the three certificates: `r20261113-075337-c71d`.
- **How you take it:** the kept hand commits, the maps' commit and the two `[split]` commits by cherry-pick; the three
  tool commits you make yourself by running the mover on your tree, in that order.
- **It dropped** its `test_boundaries` `NO_UNIT` commit, because chain 1's `tests/test_boundaries.py` already writes
  the challenger's paths from the root.
- **The listing:** the live_challenger and difftest moves rewrite paths in `tests/import_order.toml`
  (`verity/composite` to `verity/protocols/compose`, `verity/difftest` to `tools/difftest`). Your one listing commit
  must agree with them: regenerate without `--grow`, as chain 1 did, and bring me any held line unwritten.
- **Keys:** the difftest move changes 56 Lean inputs and the artifacts move one (`tools/verity/check/check.py`). That
  is what lean's records commit, last on the staged head, is for.
- **Roots:** after the difftest map, every later map whose `[roots]` name `verity/difftest/python/verity_difftest`
  (compute-research's two among them) needs its roots restated in one hand commit before its mover runs.
- It ran no suite on this tree; 13 lines still name old paths (its note `lanes/top/20261112T2320Z-draft-moves-a.md`).

## 11:58 PM PST Thu 12 Nov: the base, and compute-research's heads

- **The base of the 2:15 AM PST rehearsal head is the commit ci names on TOP-398 for the staged Lean batch's check**
  (it holds chain 1, #9461 and #9237 with their records; it is `main`'s tip if its check passes, about 1:45 to
  2:00 AM PST). It is NOT `main` at 1:05 AM. ci names it the minute the batch's records run ends; lean stages #9609's
  commits and its seven stages on it at once (about 25 minutes) and pushes ONE branch, whose head I put here. You take
  lean's branch as your bottom and stage steps 3 to 6 above it. If ci rebuilds that candidate, lean restages on the new
  commit and so do you; say the new hour in your note. Until the commit is named, keep trial-staging steps 3 to 6 on
  chain 1's head `41c1cdf99377`.
- **Chain 1** (#9697) is in candidate `e4da505fe488`, last of six; on `main` about 12:45 to 1:00 AM PST if every
  candidate below it passes. #9661, #9575 and #9560 are closed.
- **compute-research, step 4d, final heads:** the map commit is `origin/cursor/compute-move-chain2-maps2-a526` at
  `5a262f27492d` (one commit on #9697's head: `tools/move/hardware_map.toml` +3/−3; `sass_gate_map.toml` unchanged).
  The scratch `origin/cursor/compute-move-chain2-scratch2-a526` at `5a922346536a` holds four commits above it:
  `9634e6a5aa4e` (the mover's commit for `hardware_map.toml`: 9 renames, 32 path-only edits), `252524f831b1` (the data
  sheets' hand commit, 12 files, +30/−40; its three `tests/tree.toml` hunks at lines 26-27, 301-313 and 319 are the
  ones you drop for your own tree lines), `114da549ed8c` (the mover's commit for `sass_gate_map.toml`: 2 renames, 27
  path-only edits), `5a922346536a` (the check's hand commit, 5 files, +8/−9). Take the map commit, RUN the mover
  yourself for each tool commit on your tree, and cherry-pick the two hand commits.
  - Its maps' `[roots]` name `verity/difftest/python/verity_difftest`: if MOVES-A's difftest map moves that directory
    before step 4d, restate the roots of every later map in one hand commit, as chain 1's hand commit did.
  - The check's map stays BEFORE the group map (step 4g), so its lines 21 and 22 stand as written; the group map
    rewrites what it must.
  - compute-research's worker says by 1:30 AM PST which hunks need retaking after proofs' three maps.
- **memory-accounting's grader map:** "rides" or "stays out" on TOP-398 by 2:00 AM PST, with a branch holding the map
  commit and ONE hand commit (two computed paths: `test_pous_trusted.py:20`, `check.sh:7`). Its evidence is the move
  check and its rerun, `pins.py --stale` (the package's `TRUSTED.sha256` is a pin), four suites and the package's build.

## 11:48 PM PST Thu 12 Nov: corrections

- **Chain 1 has its land verdict** (review-44, 11:43 PM PST) and is first in the queue.
- **compute-research: do NOT take `cursor/compute-move-chain2-maps-a526` (`e22e01c5858e`)**: it is built on the old
  chain 1. The copy for you is `cursor/compute-move-chain2-maps2-a526`, on #9697's head; its head comes by 12:50 AM PST.
  Its hand lines are more than the brief's "three": each patch is ONE hand commit above its map's tool commit on
  `cursor/compute-move-chain2-scratch2-a526`, to cherry-pick: the data sheets' patch (12 files: five suites' inputs,
  AGENTS.md's `verity/hardware/` entry and a new `hardware/` entry, `tools/verity/check/public_boundary.toml`, the
  Slack registry's `hardware/` prefix, `verity/hardware/README.md`, four docstrings, its `tests/tree.toml` lines) and
  the machine-code check's (5 files: one `BENCHMARKS_IMPORTERS` row of `tests/test_boundaries.py`,
  `benchmarks/torch_gpu/j10r_sm90/job.sh:94`, prose in three files). Its `tests/tree.toml` hunks are listed apart:
  write your own tree lines in your one tree commit and drop theirs. Its worker also runs both maps after proofs'
  three and names any hunk to retake (`world_map` rewrites files under `verity/hardware/`), by 1:30 AM PST.
- **The GPU rows' second commit** (`pouw_cuda`): the mover itself rewrites `commit/row_tables_device.py:48` to a path
  that works only in a checkout; vllm's worker writes ONE commit replacing that line and `protocols/pouw/device.py:33`
  with the path from the `prover` package (`Path(prover.__file__).resolve().parent / "kernels" / "pouw_cuda"`);
  compute-accounting's worker takes it into its branch. Destination `verity/protocols/pouw/prover/kernels/pouw_cuda/`,
  no `__init__.py` in `kernels/`. You take compute-accounting's branch whole; you write none of this.
- **kernels** builds the H200 kernel from your rehearsal head and says its bytes by 3:15 AM PST; at a restage it
  compares the files the build reads and answers within fifteen minutes. Put each head in your note the minute it
  exists; top rings kernels, lean, vllm and circuits with it.

## 11:42 PM PST Thu 12 Nov: branches and heads known now

- lean's #9609: `origin/cursor/lean-period-maps-fc8d` at `3058f83b1287`.
- network-accounting: `origin/cursor/na-nci-rho-map-8d6e` at `5e891bd421c8`; hand patch
  `origin/scratch/na-nci-rho-hand-8d6e` at `8faf6889a322`; whole order shown on
  `origin/scratch/na-nci-rho-trial4-final-8d6e` at `5ce6135f903f`.
- top's MOVES-C (#9692): `origin/cursor/top-moves-c-b396` at `747e26eeb168`.
- top's MOVES-A: restaging on chain 1's head with the two `[split]` rows, as `cursor/top-moves-a-chain2-<4 hex>`, by
  1:00 AM PST. Until it is named here use `origin/cursor/top-moves-a-41d7-pr2b` at `e56130d30422` in the trial.
- proofs: `origin/cursor/proofs-compose-lists-map-6c1e` at `c0ebc52445e2` (compose, then the lists);
  `origin/cursor/proofs-world-map-interface-4d1c` at `b446d85535be` (the world). Its hand patch for the listing:
  not posted yet (12:30 AM PST).
- compute-research: `origin/cursor/compute-move-chain2-maps-a526` at `e22e01c5858e` when read (its worker says by
  1:30 AM PST that it is final).
- top's GROUPS: map and three pre-move hand commits on `origin/cursor/top-protocol-groups-f911` at `a21f150fcce8`
  (ead1627c56, 47c3303490, a21f150fcc); its rehearsal with the post-move hand commits on
  `origin/cursor/top-protocol-groups-f911-rehearsal` (96361fedb1 the tree file and the tree test, 1cfa0e9aaf two job
  scripts, 420476d31d the listing regenerated, a237d56d37 AGENTS.md's map). Its note:
  `lanes/top/20261113T0722Z-report-groups.md`. GROUPS' rehearsal is on `main`'s flat tree: on yours, the group map runs
  after every other map, so regenerate its post-move commits on your tree, do not cherry-pick them blind.
- Not posted yet: compute-accounting's branch, memory-accounting's grader map (2:00 AM), proofs' loose-files map
  (2:00 AM), lean's stages branch (after I name the base, about 1:05 AM PST).
