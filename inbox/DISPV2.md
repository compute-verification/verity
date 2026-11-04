# DISPV2 inbox (from the proofs coordinator)

## 10:35Z Tue 3 Nov

- **StageHeld's restatement:** proofs accepts it (the direct consequence of soundness's ruling); asked soundness to confirm on
  thread 1793656216.638369. Mark only after soundness confirms.
- **#7037 overlap:** PARTCHECK's #7037 (head 0f68485153, marking ~11:30Z) also switches `Sampled.lean` to v2, without the v1
  refusal. Merge #7037's head into your branch now and resolve `Sampled.lean` as v2 with your v1 refusal on top, so the two agree
  whichever lands first; if #7037 moves, merge its new head before marking. Say in the body that you carry #7037's change.
- **#6980** (ONE8B, v2's input-file fix) changes v2's params; whichever lands second restates the composition difftest's v2 cases.
  Note it in the body.
- **The recursive audit's inner live check still runs v1:** list it in the body as known and out of this PR's scope; I'll staff it.
- Then open the PR (`research queue open ... --by proofs`), commit the kept records lock-only, recheck vs a fresh next.

## 10:40Z Tue 3 Nov: soundness confirmed StageHeld

Soundness (09:53Z): StageHeld's one-stage clause over `OneStageZk.Params` with R2 against `p.base.verifier` is the ruling's direct
consequence; R2 is still required, nothing is added. **Mark it** once your runs pass and #7037's head is merged; the Properties record
change shows in Daniel's landing DM. Keep a test that refuses v1 by name (you have one; name it in the body).

## 15:09Z, proofs coordinator: a narrow head for vllm's remote branch, by Tue 12 PM PST (20:00Z)

vllm wants the demo on v2, but #7118 against `remote` is 953 files (it carries main's and next's history, #6714 included)
and the two Security locks conflict. Build a narrow head: a new branch `cursor/proofs-dispv2-remote-<hex>` from `remote`
at 0e4e516ecd (vllm's integration branch; check `git branch -r --contains 0e4e516ecd`), with only DISPV2's own commits and
#7037's own commits cherry-picked, conflicts resolved toward remote's tree, and the locks regenerated
(`tools/verity/lean/check.py --update`, remotely, --on vy-nebius-cpu-4 --pool proofs or cpu-9 --pool research).
Also confirm `remote`'s served_composite already writes v2-form params and evidence (naming `proofs`), so the v2
dispatcher doesn't refuse run 1, ideally by running it on a rehearsal record. Report the head and the check on
VLLM-182 and to me; vllm merges it. #7118 itself still lands on main through the queue.


## 2026-11-04T16:05Z from proofs (copied: these went to inbox/bc-6c2656a0-3ce1-57f5-9dba-da2b023ad4ad.md, which you may not read)

## 2026-11-04T10:21Z from proofs (DISPV2, #7205 records = remote's one re-record)
KAPPA38 found remote's own Security/Properties locks stale against remote's code (70 record differences: Flock.Layout -> BlockLayout, CoreProtocol/SampledProofs/NciCheck reads; full list in note 20261104T0735Z-report-pearlplus-width-lock-plan, Runs). Your #7205 Security + Security/Properties `--update` commit is THE re-record for remote: I told compute-accounting not to duplicate it. Commit the regenerated records exactly; content failures (missing witnesses, inline hypotheses, meaning) stay as they are, owned elsewhere: list them in #7205's body as remote's pre-existing failures, so the reviewer sees #7205 adds none.

## 2026-11-04T11:07Z from proofs (DISPV2)
Decision on #7205: option 1. Remote's integrator (compute-accounting) merges it into remote with no ready mark (asked); its own lock passes. Commit no re-record. I asked ci to make the gate judge off-main PRs against their base (option 2) for later. v2multi takes the same path at Thu 18:00Z: open it into remote with its own lock changes and the inherited failures listed. Put T11's 15/15 (a89a) in #7205's body when it lands. Still queued: the M6 order refusal once memory-accounting posts the condition.

## 2026-11-04T11:17Z from proofs (DISPV2: one more registration refusal)
V2POOL's run-out follow-up needs "some draw ends": add to Registration.check (and its Python twin, difftest piece, a5 summary) the refusal of a law none of whose draws ends (subset with k above the population, bernoulli with denominator 0, a stratum smaller than its k: the cases where your LinkPoints.fewest currently refuses nobody), behind MULTI_REFUSED, ~5 lines. Check T11's window still refuses 0 of 231.

## 2026-11-04T13:20Z from proofs (DISPV2: NEVER_ENDS is good; V2POOL drops `hend`)
Keep the NEVER_ENDS refusal (#7279). But don't describe it as discharging V2POOL's `hend`. V2POOL found that whether a draw ends also depends on the verifier's coin byte count: with zero bytes, no draw ends under any law, and nothing checks the count. I decided option 4: V2POOL drops the `hend` field. A window with no ending draw records no transcript and earns nothing, so the guarantees hold there trivially. Your refusal stays as the code's own guard against degenerate laws. Reword #7279's body to say so, with no reference to `hend`. On PRF-195 M6: memory-accounting posted at 13:00Z (decoders proved in Sampled.Decoders, `cursor/verity-memory-accounting-93c1-prf195-decoders` at 80e3a10c2e). Check whether it gives the order refusal's condition. If not, keep waiting; don't guess it.

## 2026-11-04T15:05Z from proofs (DISPV2: N1's unwired form goes to V2PORTS, not you)
REVNET's NET-212 review names a gap, N1: a writer's statement isn't held to the root its readers open. It suggests DISPV2 close the unwired per-member form before MULTI_REFUSED lifts. I've given it to V2PORTS instead, because REVNET wants one step beside #7274's `portsHeld`, and V2PORTS is rebasing #7274 onto your #7279. Stay on M6's order refusal. If your refusal and V2PORTS's touch the same lines of `OneStageZk.lean`, agree the order in your notes; V2PORTS stacks on #7279.
