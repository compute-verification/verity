[2026-11-09T16:02Z] 3bacca81c19f
from TWOINSTANCE (bc-d092e1b8), 16:05Z, re your 14:57Z PRIORITY

Done:
- The reshape to one template with two units a request, one sampled (ns = #[2], ks = #[1]), is in (aa04fb8c36).
- The docstring and the PR body now say non-vacuity is NOT reduced to one existence fact. It needs three things, none
  proved: Sites, an accepted run, and a concrete Ref (the toy is evidence only).
- secondProblem_held now holds at every nonempty ascending sample (TWOOPENS's guard).
- Head 5076f1ea6c is pushed. Instance (14 thms) and InstanceBoundary (23) pass on cpu-5 with std axioms. #8363 is
  re-marked, and its body is updated.

Next, the accepted run, before Sites. The plan is a new module, SpotFlock/InstanceRun.lean.
- Proved for every R, the part where the vacuity hid:
  - At the zero coin, bnd0's first stage samples #[0] and the second stage draws unit 0. Each stage reads 82 bytes:
    2*k*width + 64, with width = 9. The kernel evaluates each on 82 zero bytes, reusing SecondWitness's fuel twins;
    drawUnits_append extends that to the whole coin.
  - So secondDraw = some, and unit 0 is in checked.
  - two, partition, InScopeTwo (scopeOk_sound), the block, replayOf and the replay block all follow from Ref.scope.
  - stmtsOf (encList [xS]) = some [xS] follows from rdList_encList.
- Taken as a bundle, the facts about the session statement, which depend on R's Flock statement (call it
  Session R ...):
  - the run's statement x (the boundary record) and round-1 record m1;
  - the session statement xS and its witness wF;
  - (V true).k xS = (V true).k R.x;
  - sessionOk 0 xS, and readsOk 0 x m1 xS coin0 coin0;
  - Flock's check of xS at wF, and setupOf true xS isSome.
- Target: ok_held : Session R ... -> exists r, r.Ok (held R bnd0 ...).
- Honest caveat: Session's satisfiability is the concrete-Ref question. In particular, codesOk/pinOk at a
  one-template registration aren't shown satisfiable. The toy is evidence, but its registration has five templates.

Base choice: #8363 stays on onevalue, since counted_held needs DrawnAt (integration-only). I can split the
bytemplate-only part off if TWOOPENS prefers.

ETA for InstanceRun's first try: ~17:15Z.

[2026-11-09T16:14Z] 6a1a2e971764
Coordinator 16:16Z (top ring): five lock open premises from LEAN-186 are past due (Sun 8 Nov): Claims.Sampling.SessionRuns, Claims.SamplingGeneric.SessionOpensTwo, SamplingDeploy.Code.InteriorFixed, SamplingDeploy.Code.RefusesTwo, SamplingDeploy.SessionsIn. Re-date each to 2026-11-30 (PRF-460) in #8309's lock edit (or discharge any your restatement already proves), alongside Sites joining open. One records run covers both.

[2026-11-09T16:20Z] 4b307e981984
TWOINSTANCE 16:27Z. My records run r20261109-144859-3bb3 (cpu-1, at my head 5076f1ea6c, which contains your onevalue tip) failed on one check only, in your file:

  compile-time: Flock/SpotFlock/ReadsAt.lean:56 runs code while it compiles (elab), which can write anywhere the build can; list Proofs.Flock.SpotFlock.ReadsAt under `compile_time` with the reason if it must

(failure code compile-time.source). Nothing in my files failed. If your records run 40f0 hit the same thing, ignore this.

The accepted run's first try is in flight: InstanceRun.lean, r20261109-161915-5fad on cpu-5, ETA ~16:45Z. It proves, for every R at the zero coin, sample #[0], the second draw drawing unit 0, unit 0 checked, and `ok_held : Session R desc q c₀ → ∃ r : CheckedRun, r.Ok (held R bnd0 desc q c₀)`. `Session` holds the facts that read R's Flock statement (rounds, sessionOk, readsOk at the zero coins, Flock check, setup); its existence isn't shown.

[2026-11-09T16:27Z] dfcf0f774cf5
TWOVERIFY (bc-3772f5d9), 16:45Z. The proofs coordinator's 16:20Z order has me on the toy's instance of `scopeOk_members`. That covers:
- closed forms for `memberProg`'s length;
- the replay design's strata, which are the layout's own stratum sizes, since `filesReplay` only maps each slot;
- a per-unit check for `FilesFill`;
- a theorem from one Boolean descriptor check, plus `Built` for each member, to `scopeOk` and on to `(partition h s).isSome && (two h s).isSome`.

It all goes in one new file, `Security/Proofs/Flock/SpotFlock/ScopeDesc.lean`, on `cursor/proofs-twoverify-3e16`, which is `checked` with #8354 merged in, not on your stack. I won't touch `two`, `partition`, `Code.lean`, `Sites` or #8309. If you're proving anything about `memberProg`'s length, `filesReplay`'s classes or strata, or `FilesFill`, tell me and I'll reuse it rather than duplicate it.

