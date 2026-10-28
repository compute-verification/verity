import Proofs.Flock.Claims.Recursion.IVStarCalls
import Proofs.Flock.Claims.Recursion.CallsDecision

/-!
# `c_V` at the decisions the deployed reader makes

`nextR_calls_le` reads `hV` only at a decision `actR` makes, run at the session its setup returns against `H`, so `hV`
need hold only there (`nextR_calls_le'`, `innerCalls_R_le_acc'`, `lamLin2_R_ge_acc'`). Over every `f`, no `c_V` bounds
`decisionAtS`: it digests one file per proof. The session against `H` need not be the decision's own (`setupOf` is at
SHA-512), so `hV` takes it apart from the decision's.
-/

namespace RecDeploy

open Flock
open Claims (Msg Out Prover run coinSeq)
open FlockDeploy (Coin Ins)

/-- **A step of `nextR`'s calls**, with `c_V` bounding a verdict only at a decision `actR` makes and the session its
setup returns. -/
theorem nextR_calls_le' (rd : Msg → Option FlockDeploy.FinalP) (H : Msg → Msg) (x : Msg) (cB cV : ℕ)
    (hB : ∀ v, FlockDeploy.setupOf false x = some v → FlockDeploy.setupCallsOf (Claims.hOf H) v.1.read ≤ cB)
    (hV : ∀ as cs s spec sts link c₁ mids f spec' sts',
      FlockDeploy.actR rd x as cs = .decide s spec sts link c₁ mids f →
      ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H).1 = .ok (spec', sts') →
      ((FlockDeploy.decisionAtS s spec' sts' x link c₁ mids f).run.eval H).2.length ≤ cV)
    (as : List Msg) (cs : List Coin) : ((FlockDeploy.nextR rd x as cs).eval H).2.length ≤ cB + cV := by
  cases ha : FlockDeploy.actR rd x as cs with
  | rej => simp [FlockDeploy.nextR, ha, Claims.OC.eval]
  | coin c => simp [FlockDeploy.nextR, ha, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    obtain ⟨v, _, _, _, _, _, _, hset, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some (FlockDeploy.actR_decide ha)
    have hs : s = v.1 := congrArg FlockDeploy.DecR.s hd
    have h1 := hB v hset
    rw [← hs] at h1
    have hl := (FlockDeploy.setupM_eval H s.read).2
    have key : ((FlockDeploy.nextR rd x as cs).eval H).2 = ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H).2 ++
        (match ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H).1 with
         | .ok w => ((FlockDeploy.decisionAtS s w.1 w.2 x link c₁ mids f).run.eval H).2
         | .error _ => []) := by
      simp only [FlockDeploy.nextR, ha, FlockDeploy.KS.outOf_log, FlockDeploy.decisionSF, ExceptT.run_bind,
        Claims.OC.bind_eq, Claims.OC.eval_bind]
      rcases hv : (FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
      cases r with
      | error e => first | rfl | simp [Claims.OC.eval]
      | ok w =>
        obtain ⟨a, b⟩ := w
        first | rfl | simp [Claims.OC.eval]
    rw [key, List.length_append, hl]
    split
    · rename_i w hw
      have h2 := hV as cs s spec sts link c₁ mids f w.1 w.2 ha (by rw [hw])
      omega
    · first | omega | (simp only [List.length_nil, Nat.add_zero]; omega) | (simp; omega)

/-- `innerCalls_R_le_acc` with `hV` at `nextR`'s decisions (`nextR_calls_le'`). -/
theorem innerCalls_R_le_acc' (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (rd : Msg → Option FlockDeploy.FinalP)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (H : Msg → Msg)
    {x : Msg} {IV : Claims.Verifier Coin} {H' : Msg → Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
    {r : Fin ((V IV (deployedB b).st).k x) → Coin} (hacc : (run H' P (V IV (deployedB b).st) x ω r).out = .acc)
    (cB cV : ℕ)
    (hB : ∀ v, FlockDeploy.setupOf false (xIOf x) = some v → FlockDeploy.setupCallsOf (Claims.hOf H) v.1.read ≤ cB)
    (hV : ∀ as cs s spec sts link c₁ mids f spec' sts',
      FlockDeploy.actR rd (xIOf x) as cs = .decide s spec sts link c₁ mids f →
      ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H).1 = .ok (spec', sts') →
      ((FlockDeploy.decisionAtS s spec' sts' (xIOf x) link c₁ mids f).run.eval H).2.length ≤ cV) :
    ∀ cs ws, (((recBoundAt rel (deployedB b) L).record
      (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar).innerCalls H x cs ws).length ≤
        2 ^ 28 + (cB + cV) :=
  innerCalls_deployedB_le_acc rel b L (FlockDeploy.nextR rd) colsE wE join rowIn final w H hacc (cB + cV)
    (fun as cs => nextR_calls_le' rd H (xIOf x) cB cV hB hV as cs)
    (fun as cs m hm => nextR_msg rd H (xIOf x) as cs m hm)

/-- **`lamLin2_R_ge_acc` with `hV` at `nextR`'s decisions**: `c_V` need bound a verdict only at a decision `actR` makes
and the session its setup returns. -/
theorem lamLin2_R_ge_acc' (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (rd : Msg → Option FlockDeploy.FinalP)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (H : Msg → Msg)
    {x : Msg} {IV : Claims.Verifier Coin} {H' : Msg → Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
    {r : Fin ((V IV (deployedB b).st).k x) → Coin} (hacc : (run H' P (V IV (deployedB b).st) x ω r).out = .acc)
    (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) (ps : List (Msg × Out)) (cB cV : ℕ)
    (hB : ∀ v, FlockDeploy.setupOf false (xIOf x) = some v → FlockDeploy.setupCallsOf (Claims.hOf H) v.1.read ≤ cB)
    (hV : ∀ as cs s spec sts link c₁ mids f spec' sts',
      FlockDeploy.actR rd (xIOf x) as cs = .decide s spec sts link c₁ mids f →
      ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H).1 = .ok (spec', sts') →
      ((FlockDeploy.decisionAtS s spec' sts' (xIOf x) link c₁ mids f).run.eval H).2.length ≤ cV)
    (hc : 2 ^ 28 + (cB + cV) ≤ 2 ^ 55) :
    (∀ cs ws, (((recBoundAt rel (deployedB b) L).record
      (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar).innerCalls H x cs ws).length ≤
        2 ^ 28 + (cB + cV)) ∧
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel (deployedB b) L).record
        (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x (2 ^ 28 + (cB + cV)) (2 ^ 55) ∧
    (Claims.Recursion.setupCalls H
      (recBound rel (deployedB b) L (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar)
      ((recBound rel (deployedB b) L
        (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar).vstar x ps)).length ≤
        2 ^ 55 :=
  ⟨innerCalls_R_le_acc' rel b L rd colsE wE join rowIn final w H hacc cB cV hB hV,
    lamLin2_deployedB_ge_acc rel b L (FlockDeploy.nextR rd) colsE wE join rowIn final w decI H hacc hS ps hc⟩

end RecDeploy
