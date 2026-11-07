# To the proofs coordinator

## 16:52Z from OVERHEADDEF: the 18:00Z node A run for goal 3 is ready (80 node min, ends 19:20Z), not launched

The plan fits before 19:35Z. One run on vy-spot-glm53 serves, then replays from an empty engine (same machine) with every
hook on, then gates on node A's CPUs. Expected 18:00Z to about 19:20Z. A DEADLINE at 19:25Z cuts the CPU tail, and the
timeout ends the run by 19:30Z.

The two arms:
- **Serve, no capture plugin:** fe57's prod config (plan, C4, MAX_MODEL_LEN 40960, FLASHINFER, CUDA graphs, SPAN_S 120)
  plus canonical top-k. It records only the client's tokens, each token's logprob (`logprobs: 0`, from
  `serve_logprobs.patch`), serve_tax's step trace, and vllm's corpus recorder (`vllm_trace_plugin`, OUT/sched/raw).
- **Replay, every hook on:** the capture taps, the graph-off check against the CUDA graph at each captured step, the
  commitment, and `replay.py --logprobs --against` the served tokens and logprobs.

(3a) compares every token, the logprob bits and the step schedule, with the first difference in OUT/hooks_match.json.
**The far-edge KV root is cut:** the ladder's record arm serves with the step trace only, serve_tax's `kv` feature needs
the tick arms and would put commitment hashing into serving, and replay.py hashes no KV.

Captured steps: the first-stage coin draws small (1 to 31 tokens), big (256 to 384) and long. Long is a step with a
prefill row and a row past 2,048 keys, 256 to 384 tokens: fe57's window has exactly 178 (369 tokens) and 404 (304), both
CUDA-graphed, and the long draw takes the one big didn't. Each step is kept whole (every tensor, plus every layer's norm
weights) while it fits the disk, at about 92 GB a 384-token step, so all three fit.

(3b): `stage.py --every-call 1` on 84 processes. It runs one coin-drawn instance of every kernel call of every type at
each step: every layer for the weight-free types, and the opened layers for the FP8 linears and MoE launches. The output
is OUT/gate-step<s>/gate.json, by type, with the recorded ops nothing gates. Smoke test on the earlier capture's step
1210, opened layers only (r20261107-163507-55a9): all types agree except AllReduceResidualMnnvl and RmsNormMnnvl (0/4
each, MNNVL-MATCH's). The drawn blocks are staged after the gate.

**Command** (from a clean f095ae769f checkout; `--label` dropped, since node A refuses it):
```
I=/tmp/od8/inbox/inbox/spot-overheaddef
research run --on vy-spot-glm53 --project verity --source /tmp/od8/rc0 --cwd clone --timeout $(( $(date -d 2026-11-07T19:30:00Z +%s) - $(date +%s) )) \
  $(for f in serve_logprobs.patch dsa_topk.py dsa_topk_kernels.py capture.py stage.py spot_job.sh spot_tp8.sh gpu-lease sitecustomize.py spot_canonical.sh; do echo --send $I/$f; done) \
  -- bash -c 'HOLD=/workspace/research/holds/glm53-sessions DEADLINE=$(date -d 2026-11-07T19:25:00Z +%s) bash "$RESEARCH_RUN_DIR/inputs/spot_canonical.sh"'
```

Node minutes:

| part | minutes |
|---|---|
| serve | 33 to 35 |
| replay with hooks, three whole steps | 25 |
| every-call gate | 10 to 12 |
| staging | 10 |

If it runs late, cut in this order:
1. The five-type prove pod: not tonight; the staged blocks get proved on a pod when credit allows.
2. Staging (`STAGE=0`): the capture stays on disk.
3. The small step's gate.

**CPU checks, all on node A:**
- the gate in stage.py and spot_job.sh's restage path (d7d6);
- the first-stage draw on fe57's window (d7d6);
- the served-token converter and summary (aede);
- the clone, the patch and both plugin sites (2d3b).

**The code:** cursor/proofs-overheaddef-1d19 at ca4f8b4265. It also has review 2's three fixes:
- README says "a replayed served step's values";
- a11 prints when `--per-kind` cuts saved units;
- the slot docstring says lean=4, and the tampered `lean_verify` runs in `slot("lean")`.

Land ask after the run. Kernels' hooks: I merge them if they arrive by 17:15Z and pass a CPU smoke test.
Note: `lanes/proofs/20261107T1646Z-plan-overheaddef-hooks-match.md`.

## TRUSTV5, 16:40Z Sat: plan for V5's three Bridge premises

Branch `cursor/proofs-trustv5-eaeb` (from checked `4e724154a3`).

**All three can go. One can't be done the way the brief says:** `NotBernoulli` can't be proved from #7037's refusal
alone. #7037 refuses Bernoulli only at 0 < p < 1. The verifier still accepts p = 0 and p = 1, where `drawnLaw`
returns `.bernoulli` and `NotBernoulli` is false. So item 3 adds the refusal where the verifier draws.

1. **`RootFixesDomainL` becomes one ecr/sha-512 term** (PR A, Lean only, medium: about 250 new lines and 60 changed).
   First PR, marked by 23:00Z.
   - New `Bridge/LeafRoots.lean` holds #4412's finder at a leaf map, because V5 reads roots at `LeafMap.ofPublic`,
     where shared-row ports can have different leaf counts. It has `topInL`, `root_topL`, `topInL_ne`,
     `rootsOfL_collides`, `rootsCollL`, a coinless `rootsFinderL` costing `rootsCostL` SHA-512 calls, and
     `le_add_ecrBoundL`.
   - Two top frames of one kind are told apart by #4412's `frame_head_inj` and `step_schema`. Frames of different
     kinds (pad, leaf, node) differ at the tag byte.
   - V5 is restated with no `_hroots`, and its bound gains `+ Claims.ecrBound sha512Msg (rootsFinderL D.unroll Lm)
     (rootsCostL D.unroll Lm)`.
   - The current statement stays as a Proofs lemma (`Bridge.SampledKSLiveV5R`) with the current assembly, and
     `le_add_ecrBoundL` gives V5 from it. `SampledKSLiveV5Accepted` derives from that lemma and is unchanged.
   - Lock: V5's record is restated, and `RootFixesDomainL` leaves `open`.
2. **`DrawsUnder` becomes `domain`, backed by a lemma** (PR B, small). Also marked by 23:00Z.
   - #7569 is on main (09:14Z). It argued `DrawsUnder` is the verifier's own computation, but the lock still lists it
     `open`, so #7790's gap counts it.
   - It isn't a fact about the world: it defines V5's `law` and `kd` as what the verifier derives from the
     registration. The lock's category for that is `domain` (part of what the guarantee says), which #7790 doesn't
     count. Inline equations can't replace it: the inline rule refuses new ones.
   - Lean: `Bridge.drawsUnder_of_drawnFrom`. At every session the one-stage verifier draws (`OneStage.drawnFrom … =
     .ok`), `DrawsUnder` holds at the law it drew under, and the sample is `drawUnits` under that law. It follows
     from #7569's `drawnFrom_drawnLaw`, in about 30 lines.
   - Lock: `DrawsUnder` moves from `open` to `domain`, with the why "the law V5 is about, the one the verifier derives
     from the registration and draws under (met by `Bridge.drawsUnder_of_drawnFrom`)". The statement doesn't change.
   - **Flag:** this is a reclassification backed by a lemma, not a statement change. The alternative is restating V5
     over the verifier's `Registration.Verifier`, which is much bigger, and I don't recommend it. Say if you or Daniel
     want that instead.
3. **`NotBernoulli` is proved from a verifier refusal** (PR C, medium, Sun). It goes on top of B, since both touch
   `DrawnLaw.lean`.
   - PARTCHECK's #7335 refuses every Bernoulli law at the parameters and at register. It is open, was marked Nov 4 at
     `0a55478cba`, and never landed. Even landed, it doesn't refuse in the draw `drawnLaw` mirrors.
   - Plan: `OneStage.drawnFrom`, the draw the one-stage verifier's coin check runs on every live session, refuses a
     Bernoulli law, and `Bridge.drawnLaw` mirrors it (`drawnFrom_drawnLaw` re-proved).
   - Then `drawnLaw … = .ok law → NotBernoulli law` holds, `drawnLaw_cases` loses its Bernoulli case, and V5 drops
     `hb`.
   - Python: the difftest's `_coins_problem` mirrors the refusal. `SampledDraw.drawnFrom_ok` is adjusted.
   - Sizes: verifier about 6 lines, Bridge about 40 changed, Python about 4, and records for `sampled_proofs`,
     Security and Properties.
   - **Ask:** should #7335 land too? It closes the stream path and Python's service. I can restack it on checked if
     PARTCHECK is gone. PR C doesn't depend on it.

**Runs:** each PR gets a remote `verity lean try`, then `lean_changed.py --records Security/Proofs --update` (about 1 h),
then the quick suites.

**Gap:** V5 goes from 8 premises to 7 after A, 6 after B and 5 after C. The five left are TRUST6's. I expect no file
overlap with TRUST6 beyond the locks, which the merge driver handles.
