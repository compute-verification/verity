Lane circuits
Refs CIR-264
Goal vllm/production-vllm-reproducible

## What changed and why

vLLM at the pin `d9105ea80` verifies a request's draft tokens by a sampled accept when the temperature is not zero: draft token i is accepted when its log-probability in the target's row is above log(u) for a uniform u, and the first refused position (the bonus row when none is refused) is sampled again from its row with the refused token masked. The tree stated only the greedy accept (`Accept_v1`). This adds the sampled accept and its resample as registered Definitions, read from the PTX and SASS of the four Triton kernels that `rejection_sample` launches, at GLM-5.3's vocabulary of 154,880 with five draft tokens, and holds them to a capture of those kernels on a card.

- `verity/definitions/registry/sampled_accept.py` (new). `SampledAccept_v1{K,V,TILE}` takes the K + 1 rows of logits after top-p, the K draft tokens, the request's seed, the rows' positions and the temperature, and returns `Accept_v1`'s outputs: the verdicts, the accepted count and the appended tokens. The whole is far past one Flock block, so its body is calls of four unit Definitions and their wiring: `AcceptTileHead_v1`, `AcceptTileSum_v1`, `AcceptRowEnd_v1` and `AcceptResample_v1`, with `AcceptExpLane_v1`, `AcceptUniform_v1` and `LogitAtStep_v1` under them. The module's docstring holds the reading (the order of each sum, each `exp` and `log` with its exact form, the comparison, the uniform) and the domain. A numpy reference in the same module computes the same words from the kernels' layouts, written apart from the Definitions' bodies.
- `verity/definitions/registry/sampled_accept.circuit-check.py` (new): the sidecar's bindings.
- `adapters/vllm/tests/program/sampled_accept_vectors.py` and `test_sampled_accept.py` (new): ten hand vectors (u at its floor, a log-probability equal to log(u) and one place either side, a draft token outside the top-p set, a placeholder, every position refused, every position accepted, a refused token the resample would pick again, the warp's two halves disagreeing), random requests, and each Definition against the reference.
- `adapters/vllm/tests/properties/sampled_accept_capture_gpu.py` (new): launches the pin's kernels on a card, keeps each request's operands and stored words, compares them with the reference in the job, and replays a kept capture off the card (`--replay`), where it also evaluates the Definition on the first N requests.
- `verity/definitions/evidence.json`: the records of the kept PTX, the capture and its replay, which the conformance text names. `verity/definitions/tests/catalog_ids.json`: the eight new ids.

Every instruction of the kernels is stated by a primitive that was already registered. No registered Definition's words change and no existing registry module is edited. Nothing is deleted: the greedy accept stays, since the kernels verify greedily at temperature zero and `Accept_v1` states that.

## What others own, and what is open

- **The `logits` port has no writer in this pull request.** The Definition reads the rows after temperature and top-p, -inf outside the kept set, because that is what the kernels read. The registered unit that writes that masked row, and a re-cut of the accept's units to whole rows of 32 lanes, are CIR-273's pull request, stacked on this one.
- **At this head `Q_call v1` refuses the resample's Definitions as `gate-too-wide`** (10 Definitions in the report below: nine bindings of `AcceptResample_v1` and the small `SampledAccept_v1` that calls three of them; they count as unchecked, which never fails). The cause is the chunk's 64-bit select of the row's position. CIR-273's pull request cures it: there the resample selects the rows' 32-bit stream keys [the lead's word for that commit; it is not in this branch and I did not run it].
- **vLLM's split top-p pipeline takes at most 64 rows a call under a draft** [vllm's reading, on CIR-264; not read by me]. The Definition is one request's, so the limit changes nothing in it.
- **How leakage counts the masked row** is asked of proofs on CIR-264.
- **One case is outside the Definition's domain.** The rejection kernel's one warp holds its sum in two halves that can differ in the last place. When lane 0 accepts a draft token that is not the last and lanes 16 to 31 refuse it, the upper half stops and the next position reads lanes that no longer run. The Definition states lane 0's arithmetic there and its domain leaves the case out. The hand vector `halves-split` builds it; on the card 1 of its 200 stored words differs from lane 0's arithmetic (below).

## circuit-check

`uv run circuit-check verity/definitions/registry/sampled_accept.circuit-check.py --update-pins` in a clone of `66f957e940d7` (run `r20261113-042405-da93`): **31 targets, 0 failures, 30 unchecked (warnings, which never fail), 10 Definitions `Q_call v1` refuses.** It wrote no pin: the clone's `git status` and `git diff` are empty after it. Each target, as the run printed it (`u` units, `b` committed bits):

~~~text
unchecked definition AcceptExpLane_v1  failures=0 unchecked=1 q_call=1u/32b
unchecked definition AcceptResample_v1{K=2,N=16,FIRST=False,LAST=True}  failures=0 unchecked=1 q_call=refused:gate-too-wide
unchecked definition AcceptResample_v1{K=2,N=2,FIRST=False,LAST=False}  failures=0 unchecked=1 q_call=refused:gate-too-wide
unchecked definition AcceptResample_v1{K=2,N=2,FIRST=False,LAST=True}  failures=0 unchecked=1 q_call=refused:gate-too-wide
unchecked definition AcceptResample_v1{K=2,N=2,FIRST=True,LAST=False}  failures=0 unchecked=1 q_call=refused:gate-too-wide
unchecked definition AcceptResample_v1{K=2,N=32,FIRST=False,LAST=False}  failures=0 unchecked=1 q_call=refused:gate-too-wide
unchecked definition AcceptResample_v1{K=2,N=32,FIRST=True,LAST=False}  failures=0 unchecked=1 q_call=refused:gate-too-wide
unchecked definition AcceptResample_v1{K=5,N=32,FIRST=False,LAST=False}  failures=0 unchecked=1 q_call=refused:gate-too-wide
unchecked definition AcceptResample_v1{K=5,N=32,FIRST=False,LAST=True}  failures=0 unchecked=1 q_call=refused:gate-too-wide
unchecked definition AcceptResample_v1{K=5,N=32,FIRST=True,LAST=False}  failures=0 unchecked=1 q_call=refused:gate-too-wide
unchecked definition AcceptRowEnd_v1{T=17,N=16,AT=0}  failures=0 unchecked=1 q_call=1u/32b
unchecked definition AcceptRowEnd_v1{T=17,N=16,AT=512}  failures=0 unchecked=1 q_call=1u/32b
unchecked definition AcceptRowEnd_v1{T=19,N=3328,AT=4096}  failures=0 unchecked=1 q_call=1u/32b
unchecked definition AcceptTileHead_v1{N=16,FIRST=False,LAST=False}  failures=0 unchecked=1 q_call=78u/1535b
unchecked definition AcceptTileHead_v1{N=16,FIRST=False,LAST=True}  failures=0 unchecked=1 q_call=77u/1503b
unchecked definition AcceptTileHead_v1{N=16,FIRST=True,LAST=False}  failures=0 unchecked=1 q_call=78u/1535b
unchecked definition AcceptTileHead_v1{N=512,FIRST=False,LAST=False}  failures=0 unchecked=1 q_call=2558u/50143b
unchecked definition AcceptTileHead_v1{N=512,FIRST=True,LAST=False}  failures=0 unchecked=1 q_call=2558u/50143b
unchecked definition AcceptTileHead_v1{N=7424,FIRST=False,LAST=True}  failures=0 unchecked=1 q_call=37117u/727487b
unchecked definition AcceptTileHead_v1{N=8192,FIRST=False,LAST=False}  failures=0 unchecked=1 q_call=40958u/802783b
unchecked definition AcceptTileHead_v1{N=8192,FIRST=True,LAST=False}  failures=0 unchecked=1 q_call=40958u/802783b
unchecked definition AcceptTileSum_v1{N=16,AT=512,LAST=True}  failures=0 unchecked=1 q_call=1u/32b
unchecked definition AcceptTileSum_v1{N=4096,AT=0,LAST=False}  failures=0 unchecked=1 q_call=8064u/258048b
unchecked definition AcceptTileSum_v1{N=4096,AT=4096,LAST=True}  failures=0 unchecked=1 q_call=1u/32b
unchecked definition AcceptTileSum_v1{N=512,AT=0,LAST=False}  failures=0 unchecked=1 q_call=896u/28672b
unchecked definition AcceptTileSum_v1{N=512,AT=0,LAST=True}  failures=0 unchecked=1 q_call=1u/32b
unchecked definition AcceptTileSum_v1{N=6,AT=0,LAST=True}  failures=0 unchecked=1 q_call=1u/32b
unchecked definition AcceptUniform_v1  failures=0 unchecked=1 q_call=1u/32b
ok        definition LogitAtStep_v1  failures=0 unchecked=0 q_call=3u/65b
unchecked definition SampledAccept_v1{K=2,V=8208,TILE=512}  failures=0 unchecked=2 q_call=refused:gate-too-wide
ok        definition SampledAccept_v1{K=5,V=154880,TILE=8192}  failures=0 unchecked=0 by-calls: 9 callees, not lowered whole
unchecked redundant-gates/boolean: 6311 over 6 target(s)
unchecked redundant-gates/ir: 2560 over 1 target(s)
unchecked vectors/none: 23 over 23 target(s)
circuit-check: vectors a realization compared with the reference on: 7 random, 0 edge-only, 23 none
circuit-check: 31 target(s), 0 failure(s), 30 unchecked (warnings, which never fail) on 29 Definition(s), 10 Definition(s) Q_call v1 refuses
~~~

- **`ok`, 2 targets.** `LogitAtStep_v1`, and `SampledAccept_v1{K=5,V=154880,TILE=8192}`, GLM-5.3's Call, which is checked by its calls: it is past what circuit-check builds whole, and each of its 9 callees is a target above.
- **`vectors/none`, 23 targets.** No realization compared the target with its reference, because the target is too large to lower whole; only the reference's totality was checked. The 7 small targets were lowered and compared with the reference on random vectors.
- **`redundant-gates/boolean`, 6 targets, and `redundant-gates/ir`, 1 target.** 6,311 ANDs on identical operands in the lowerings of the six small targets, and 2,560 values the small Call's chunks each work out again.
- **Every unit at GLM-5.3's statics is inside a Flock block** [read from the report's `block` entries: the largest are `AcceptRowEnd_v1{T=19,N=3328,AT=4096}` and the two `AcceptTileSum_v1{N=4096,..}` at k = 26 of 27]. The small Call whole is past a block and is held to it by its cut, as the sidecar says.

## Runs, and what I read in each

The head is `66f957e940d7428d0904626920d265a4f38e93c7`. The capture and its replay ran at its parent `903c545eb3d3`; the head differs from it only in `evidence.json`'s entry for those two runs and in the conformance text that names them. Times are UTC.

- **`r20261113-034537-8871`, the capture on a card** (node A, card 0, an H200, compute capability 9.0; cores 0-9; one lease held 85 s). The pin's four kernels, launched with the sampler's arguments at 154,880 logits and five draft tokens, one request of 1 + 5 rows a launch, no model loaded. 410 requests (10 hand vectors, 400 random), 2,460 rows. **409 are inside the domain: 80,226 stored words compared with the reference, 0 differ.** Accepted draft tokens a request: none 236, one 57, two 37, three 19, four 17, five 44. Each request launched three times, once into buffers filled with `0xA5`: 0 launches disagree, and the words the kernels leave unwritten are the expected ones on all 410. The 410 as one batch with their rows shuffled: 0 requests differ from their own launch. The PTX and the machine code on the node are the ones the Definitions were read from, for all four kernels. The gate held: vLLM `0.28.1rc1.dev472+gd9105ea80`, Triton 3.7.1, the pin's two source files by SHA-256.
- **The same run, the one request outside the domain (`halves-split`).** On the card its count is 6 and its six tokens are lane 0's; 1 of its 200 words differs from lane 0's arithmetic: the kept log-sum-exp of the last position tested is `0xc1af360a` where lane 0's is `0x40e00000`. So after a split the card's sums are not lane 0's arithmetic, and the tokens agree only because this vector's later rows put all their mass on the draft token [the second half is my inference from the vector; no other split was tested].
- **`r20261113-034958-1553`, the capture replayed off the card** (waited 10.5 min for a slot, held it 16.5 min). The kept operands and stored words against the reference again: the same counts. **The registered `SampledAccept_v1{K=5,V=154880,TILE=8192}` evaluated on 11 requests** (nine hand vectors and two random ones), its 12 output words against the card's count and tokens and against the reference's: **264 words compared, 0 differ.** It is 11 requests and not 409 because one evaluation at this size takes over a minute.
- **`r20261113-042405-da93`, circuit-check at the head** (waited 15.7 min, held 49 s): the report above.
- **`r20261113-044914-2cc1`, the suites this change reaches, at the head** (`suites.py --changed 3498d4ad65 --quick`; no wait, held its slot 120 min). **59 suites: 59 passed (0 cached), 0 failed**; 34,182 tests passed and 488 skipped over the 59 [summed by me from `out/suites.json`], with no read outside a suite's inputs. Among them `verity-vllm` 14,356 passed, 400 skipped (it ran `test_sampled_accept.py` whole, slow tests included); `verity/definitions` 841 passed; `verity/circuits` 1,120 passed; `repository` 349 passed. `out/suites.json` and each suite's log are in the run. **The run itself ended at its 120-minute limit (exit 143)** during its second step, `repository` whole, which the next run did instead.
- **`r20261113-065139-a23b`, `repository` whole and the ratchet gate, at the head** (no wait, held 5.6 min). `repository`: **352 passed, 0 failed**, no line of its log names a file of this change; `out/repository/suites.json` is in the run. `adapters/vllm/tools/ratchet_gate.py 3498d4ad65 HEAD`: **1 change, 0 failed** (no new environment read, silent except, lane-text site or retired-term site).
- **Three style warnings in this change's files, left as they are.** In the same suites run, `tests/lint/test_ruff_vllm.py` (ruff under vLLM's own rules, a bookkeeping warning) lists 97 findings that are not in its allowlist; three are in this change's files: the import block of `sampled_accept_vectors.py` and of `test_sampled_accept.py` (`I001`: vLLM's rules want one import a line) and one implicit string concatenation in `sampled_accept_capture_gpu.py` (`ISC004`). The integration's own ruff (`adapters/vllm/ruff.toml`) passes on the three files [run locally]. Fixing them would move the head off the one every run here is at, and the capture script off the text the capture ran; a later change can take them.
- **`r20261112-135317-f990`, the kept PTX** (earlier, compiled without a card): each kernel's Triton IR, LLVM IR, PTX, cubin and SASS at this specialization, which the reading in the docstring is of.
- **`r20261112-154819-c4ae`, an interpreter of the kept PTX against the reference, off the card** (earlier, at `c40961a349`): 1,010 requests, 1,009 inside the domain, 197,890 words compared, 0 differ. It holds the order of the sums, the control flow and the inlined log; the capture is what holds `ex2.approx.f32` on the card.
- Locally at the head: the retired-terms and lane-text lints (`tests/lint/test_repo_retired_terms.py`, `test_repo_lane_text.py`): 17 passed, 2 bookkeeping warnings, neither naming a file of this change.

## Not run

- A served request: the capture launches the kernels itself, with the arguments the sampler gives them. No model was loaded and no vLLM serve ran.
- Any split of the warp's halves other than the one hand vector.
- `circuit-check --all`, the Lean check, Flock's build, and any suite this change's files are not inputs of.
- The slow tier of the suites other than `repository`'s and `test_sampled_accept.py`'s (`--quick`).
- A merge of `main`: every run above is at this branch's own head, cut from `3498d4ad65`.
