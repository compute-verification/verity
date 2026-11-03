import Proofs.Properties.Verity.Recursive.Finders
import Proofs.Flock.Claims.Recursion.BitsLin2
import Proofs.Flock.Claims.Recursion.CostExp
import Proofs.Flock.Claims.Recursion.InnerAbort
import Proofs.Flock.Claims.Recursion.Compose
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.Claims.Recursion.DeployFacts

/-!
# A play's expected SHA-512 calls

`Dep.ex_playCost_le`: averaged over a play's sample at the draw `ξ`, its calls (`Dep.playCost`) are at most its budget
`Dep.playBudget = (1 + r) · t' + r + N_I · nI`:
* the run's calls are at most `t'` at every sample (`SessionIn.calls_le`);
* `E*`'s program reads the oracle only through reruns (`noHash_served`), so its calls are at most `t'` a rerun
  (`Claims.Sampling.calls_le_of_noHash`), and its reruns are `r` in expectation (`Claims.Recursion.efficient`);
* its inner views make at most `nI` calls at each of the inner `E^rep`'s reruns where `E_O` returns witnesses, since a
  run then accepts (`Claims.Recursion.EO_some_acc`), and those reruns are `N_I` in expectation
  (`Claims.erepProg_cost_exp`).
-/

namespace Verity.Recursive

open scoped ENNReal
open Claims (Msg)
open FlockDeploy (Coin)
open RecDeploy
open Accounting (Device)

theorem cast_sum_map {D : Type} (l : List D) (f : D → ℕ) :
    (((l.map f).sum : ℕ) : ℝ) = (l.map fun d => (f d : ℝ)).sum := by
  induction l with
  | nil => simp
  | cons a l ih => simp [List.map_cons, List.sum_cons, ih]

/-- **`E^rep` served by `E₂` reads the oracle only through reruns** where `E₂` does. -/
theorem noHash_served {Coin W W₂ Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (V : Claims.Verifier Coin) (E₂ : Claims.Extractor Coin W₂ V)
    (hE₂ : ∀ x r c, Claims.Sampling.NoHash (E₂.ext x r c))
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Claims.Seen) (m k K : Msg → ℕ)
    (opens : Claims.Seen → List Op) (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg)
    (r₀ : Fin (V.k x) → Coin) (c : (Claims.Recursion.served V E₂ dec m k K opens assemble check).C x) :
    Claims.Sampling.NoHash ((Claims.Recursion.served V E₂ dec m k K opens assemble check).ext x r₀ c) := by
  dsimp only [Claims.Recursion.served]
  refine Claims.Recursion.noHash_serve (fun d => ?_) ?_
  · unfold Claims.Recursion.outerStage
    exact Claims.Sampling.NoHash.bind (hE₂ x d.1 d.2) fun _ => Claims.Sampling.NoHash.ret _
  · refine Claims.Sampling.NoHash.rerun _ _ fun v => ?_
    split
    · exact Claims.Sampling.NoHash.firstWait _ _ _ _ _
    · exact Claims.Sampling.NoHash.ret _

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

namespace Dep

variable (q : Dep)

/-- `E*`'s program reads the oracle only through reruns. -/
theorem noHash_E (x : Msg) (r : Fin (q.D.V.k x) → Coin) (c : q.E.C x) : Claims.Sampling.NoHash (q.E.ext x r c) :=
  noHash_served q.D.V q.EO (fun x r c => Claims.Recursion.noHash_EO q.D q.I _ q.decO x r c)
    (Claims.Recursion.dec FlockDeploy.sha512 q.D) (fun x => q.I.waits (Claims.Recursion.xI q.D x))
    (fun x => q.I.k (Claims.Recursion.xI q.D x)) (fun x => q.I.cap (Claims.Recursion.xI q.D x)) q.I.opens
    (fun x => q.I.assemble (Claims.Recursion.xI q.D x)) (fun x => q.I.check (Claims.Recursion.xI q.D x)) x r c

/-- **`E*`'s calls are at most `t` a rerun** where every run of the prover makes at most `t`. -/
theorem E_calls_le {Ω : Type} (P : Claims.Prover Ω) (x : Msg) (t : ℕ)
    (ht : ∀ ω r, runCalls (Claims.run FlockDeploy.sha512 P q.D.V x ω r) ≤ t) (ω : Ω) (r : Fin (q.D.V.k x) → Coin)
    (c : q.E.C x) :
    (q.E.eval FlockDeploy.sha512 P x ω r c).2.1 ≤ t * (q.E.eval FlockDeploy.sha512 P x ω r c).2.2 :=
  Claims.Sampling.calls_le_of_noHash FlockDeploy.sha512 (Claims.run FlockDeploy.sha512 P q.D.V x ω) t (ht ω)
    (q.noHash_E x r c)

/-- **`E*`'s expected reruns are `runs`** (`Claims.Recursion.efficient`). -/
theorem ex_E_runs_le {Ω : Type} (P : Claims.Prover Ω) (x : Msg) (ω : Ω) :
    Claims.Ex (fun s : (Fin (q.D.V.k x) → Coin) × q.E.C x =>
      ((q.E.eval FlockDeploy.sha512 P x ω s.1 s.2).2.2 : ℝ)) ≤ q.runs x := by
  refine ((Claims.Recursion.efficient FlockDeploy.sha512 q.D q.I (deployZkRec _root_.Flock.Readable.relation)
    q.decO (RecDeploy.sessionAccepts _ _ _)).2 x Ω P ω).trans (le_of_eq ?_)
  unfold Dep.runs
  push_cast
  try ring

/-- **The extraction's expected inner-view calls are `N_I · nI`** where every accepted statement's inner views make at
most `nI` calls (`hn`): each of the inner `E^rep`'s reruns reads one view, where `E_O` returns witnesses and so the run
accepts (`EO_some_acc`), and the reruns are `N_I` in expectation (`Claims.erepProg_cost_exp`). -/
theorem ex_ivCalls_le (nI : Msg → ℕ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    {Ω : Type} (P : Claims.Prover Ω) (x : Msg) (ω : Ω) :
    Claims.Ex (fun s : (Fin (q.D.V.k x) → Coin) × q.E.C x => (q.ivCallsAt P x ω s.1 s.2 : ℝ)) ≤
      ((Claims.Recursion.NI q.D q.I x * nI x : ℕ) : ℝ) := by
  set w : (Fin (q.D.V.k x) → Coin) × q.EO.C x → ℝ := fun dr =>
    (((q.EO.eval FlockDeploy.sha512 P x ω dr.1 dr.2).1.elim 0 fun ws =>
      (q.D.innerCalls FlockDeploy.sha512 x (Claims.coinSeq dr.1) ws).length : ℕ) : ℝ)
  have hw : ∀ dr, 0 ≤ w dr := fun _ => Nat.cast_nonneg _
  have hwn : ∀ dr, w dr ≤ nI x := by
    intro dr
    simp only [w]
    cases h : (q.EO.eval FlockDeploy.sha512 P x ω dr.1 dr.2).1 with
    | none => simp
    | some ws =>
      have hacc := Claims.Recursion.EO_some_acc FlockDeploy.sha512 q.D q.I
        (deployZkRec _root_.Flock.Readable.relation) q.decO P x ω dr.1 dr.2 h
      exact_mod_cast hn x ⟨Ω, P, ω, dr.1, hacc⟩ _ ws
  have hEw : Claims.Ex w ≤ nI x :=
    (Claims.Ex_mono hwn).trans_eq (Claims.Ex_const _)
  calc Claims.Ex (fun s : (Fin (q.D.V.k x) → Coin) × q.E.C x => (q.ivCallsAt P x ω s.1 s.2 : ℝ))
      = Claims.Ex (fun s : ((Fin (q.D.V.k x) → Coin) × q.EO.C x) ×
            (Fin (q.I.waits (Claims.Recursion.xI q.D x)) → Fin (q.I.cap (Claims.Recursion.xI q.D x)) →
              (Fin (q.D.V.k x) → Coin) × q.EO.C x) =>
          (((Claims.erepProg (q.I.waits (Claims.Recursion.xI q.D x)) (q.I.k (Claims.Recursion.xI q.D x))
              (q.I.cap (Claims.Recursion.xI q.D x)) q.I.opens (q.I.assemble (Claims.Recursion.xI q.D x))
              (q.I.check (Claims.Recursion.xI q.D x)) s.1 s.2).rerunsAt FlockDeploy.sha512
            (Claims.servedRun FlockDeploy.sha512
              (Claims.Recursion.outerStage q.EO (Claims.Recursion.dec FlockDeploy.sha512 q.D) x)
              (Claims.run FlockDeploy.sha512 P q.D.V x ω))).map w).sum) := by
        refine (Claims.Ex_assoc (α := Fin (q.D.V.k x) → Coin) (β := q.EO.C x)
          (γ := Fin (q.I.waits (Claims.Recursion.xI q.D x)) → Fin (q.I.cap (Claims.Recursion.xI q.D x)) →
            (Fin (q.D.V.k x) → Coin) × q.EO.C x) _).trans ?_
        refine congrArg Claims.Ex (funext fun s => ?_)
        exact cast_sum_map _ _
    _ ≤ (1 + q.I.waits (Claims.Recursion.xI q.D x) * q.I.k (Claims.Recursion.xI q.D x)) * Claims.Ex w :=
        Claims.erepProg_cost_exp FlockDeploy.sha512 _ hw _ _ _ _ _ _
    _ ≤ (1 + q.I.waits (Claims.Recursion.xI q.D x) * q.I.k (Claims.Recursion.xI q.D x)) * nI x :=
        mul_le_mul_of_nonneg_left hEw (by positivity)
    _ = ((Claims.Recursion.NI q.D q.I x * nI x : ℕ) : ℝ) := by
        unfold Claims.Recursion.NI
        push_cast
        try ring

variable (V : Vocab d α) {k : Verity.Registration.Composite} {θ c : ByteArray} {F : Array ByteArray}
  {σ : Game.Strategy (V.game θ F c)}

/-- **A play's expected calls are at most its budget**: averaged over the sample at the draw `ξ`, the run's calls (at
most `t'`), `E*`'s (at most `t'` a rerun) and its reruns (`runs` in expectation), and its inner views' (`N_I · nI` in
expectation, where every accepted statement's inner views make at most `nI` calls, `hn`). -/
theorem ex_playCost_le (S : SessionIn V q k θ F c σ) (nI : Msg → ℕ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x) (ξ : S.Ξ) :
    Claims.Ex (fun s : q.Sample V S ξ => (q.playCost V S ξ s : ℝ)) ≤ q.playBudget V S nI ξ := by
  have ht : ∀ ω r, runCalls (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r) ≤ q.tCost V S :=
    fun ω r => (S.calls_le ξ ω r).trans (le_max_right _ _)
  have hpt : ∀ s : q.Sample V S ξ, (q.playCost V S ξ s : ℝ) ≤
      (q.tCost V S : ℝ) + (q.tCost V S + 1) *
        ((q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).2.2 : ℝ) + (q.ivCalls V S ξ s : ℝ) := by
    intro s
    have h1 := ht s.1 s.2.1
    have h2 := q.E_calls_le (S.P ξ) (S.x ξ) (q.tCost V S) ht s.1 s.2.1 s.2.2
    have : q.playCost V S ξ s ≤ q.tCost V S +
        q.tCost V S * (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).2.2 +
        (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).2.2 + q.ivCalls V S ξ s := by
      unfold Dep.playCost
      omega
    have h3 : ((q.playCost V S ξ s : ℕ) : ℝ) ≤ ((q.tCost V S +
        q.tCost V S * (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).2.2 +
        (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).2.2 + q.ivCalls V S ξ s : ℕ) : ℝ) :=
      Nat.cast_le.2 this
    push_cast at h3
    linarith
  have hE := fun ω => q.ex_E_runs_le (S.P ξ) (S.x ξ) ω
  have hI := fun ω => q.ex_ivCalls_le nI hn (S.P ξ) (S.x ξ) ω
  have ht0 : (0 : ℝ) ≤ q.tCost V S + 1 := by positivity
  calc Claims.Ex (fun s : q.Sample V S ξ => (q.playCost V S ξ s : ℝ))
      ≤ Claims.Ex (fun s : q.Sample V S ξ => (q.tCost V S : ℝ) + (q.tCost V S + 1) *
          ((q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).2.2 : ℝ) + (q.ivCalls V S ξ s : ℝ)) :=
        Claims.Ex_mono hpt
    _ = Claims.Ex (fun ω : S.Ω => Claims.Ex (fun s : (Fin (q.D.V.k (S.x ξ)) → Coin) × q.E.C (S.x ξ) =>
          (q.tCost V S : ℝ) + (q.tCost V S + 1) *
            ((q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) ω s.1 s.2).2.2 : ℝ) +
            (q.ivCallsAt (S.P ξ) (S.x ξ) ω s.1 s.2 : ℝ))) :=
        Claims.Ex_prod _
    _ ≤ Claims.Ex (fun _ : S.Ω => (q.tCost V S : ℝ) + (q.tCost V S + 1) * (q.runs (S.x ξ) : ℝ) +
          ((Claims.Recursion.NI q.D q.I (S.x ξ) * nI (S.x ξ) : ℕ) : ℝ)) := by
        refine Claims.Ex_mono fun ω => ?_
        rw [Claims.Ex_add, Claims.Ex_add, Claims.Ex_const, Claims.Ex_mul_left]
        exact add_le_add (add_le_add le_rfl (mul_le_mul_of_nonneg_left (hE ω) ht0)) (hI ω)
    _ = (q.playBudget V S nI ξ : ℝ) := by
        rw [Claims.Ex_const]
        unfold Dep.playBudget
        push_cast
        ring

end Dep

end Verity.Recursive
