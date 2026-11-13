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

CIRCUIT_CHECK

## Runs, and what I read in each

RUNS

## Not run

- A served request: the capture launches the kernels itself, with the arguments the sampler gives them. No model was loaded and no vLLM serve ran.
- Any split of the warp's halves other than the one hand vector.
- `circuit-check --all`, the Lean check, Flock's build, and any suite this change's files are not inputs of.
- The slow tier of the suites (`--quick`).
- A merge of `main`: every run above is at this branch's own head, cut from `3498d4ad65`.
