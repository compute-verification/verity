import Proofs.Flock.Claims.ZK.NonVacuous
import Proofs.Flock.Claims.ZK.Interact

/-!
# Refutation witnesses: the shared claim forms

Every claim definition carries a refutation witness: a theorem `<Claim>_fails` that exhibits a concrete toy instance at
which the claim is false, so the definition constrains something. A claim whose definition no instance can refute is
vacuous. This file holds the toys and the witnesses of `Definitions.Core.Interaction`'s shared forms; Flock's are
`Witnesses.Flock`, the sampling compiler's `Witnesses.Sampling`, the ZK-recursion compiler's `Witnesses.Recursion`.

The toys: `accV` accepts after one round and `rejV` rejects; `silentP` sends the empty message and calls nothing;
`noneE` gives up at once and `onceE` reruns the judged run once and gives up; `noFinder` outputs nothing; `pairFinder`
outputs `([0], [1])`, a collision of the constant hash `constH`, whose outputs are 64 bytes.

* `KS_fails`, `KSLin_fails`, `Bits_fails`: at `accV`, `silentP` and a statement with no witness, every extractor fails
  at `κ < 1` (at `λ > 0` for `Bits`), whatever `γ`: the run is accepted surely and `B` finds nothing.
* `Complete_fails`: `rejV` fails at `δ_c < 1` wherever the scope and the relation meet. Its tapes are a nonempty
  type, so the prover with no tape, which met its first form at every verifier and `δ_c = 0`, is refused by its type.
* `ScopeMatches_fails`: an empty scope and a check that takes everything.
* `ZKmalR_fails` (`ZKMal.not_zkmalR_collSim`): the colliding simulator, which met the form before `Fork`, fails it.
  `ZKmalR_vacuous`: the form's honest prover is a parameter, so a prover that ignores its witness and calls nothing
  (`ZKMal.quietP`) meets it at `ε_zk = 0` and `γ = 0`, at the simulator that runs that prover (`quietSim`). The
  families' `ZK` read the masked record's own reference prover and simulator, so it meets theirs only at a record
  whose own prover ignores its witness (`Flock.ZK_of_quiet_record`).
  `SimCost_fails`: the colliding simulator's two calls exceed a bound of `1`.
* `ExtractRunsLe_fails`, `ExtractRunsExpLe_fails`: `onceE` reruns once, above a bound of `0`.
* `Nontrivial_fails`: a recorded run that `rejV` rejects.
* `StrictBirthday_fails`, `ExpectedCR_fails`: `pairFinder` finds `constH`'s collision at no cost;
  `StrictBirthday_holds`, `ExpectedCR_holds`: `noFinder` meets both, so they are satisfiable.
* `Hm96Hiding_fails`: a constant mask hides nothing below `δ₁ = 1/2`; `Hm96Hiding_holds`: a uniform mask with a
  constant salt hash hides at `δ₁ = 0`.
* `JsonSpec_fails`: a canonical printer that never returns. `JsonSpec` reads `Json.parse`, `==` and `canon`, which the
  kernel can't unfold, so no proof can show it satisfiable.

`ERepCost` is a theorem at every record (`Claims.ERepCost` in `Proofs.Flock.Claims.ERepCost`), so it has no refutation.
-/

namespace Claims.Witness

open Claims
open scoped ENNReal

/-- The verifier that accepts after one round. -/
def accV (Coin : Type) : Verifier Coin := ⟨fun _ => 1, fun _ _ _ => .ret .acc⟩

/-- The verifier that rejects after one round. -/
def rejV (Coin : Type) : Verifier Coin := ⟨fun _ => 1, fun _ _ _ => .ret .rej⟩

/-- The prover that sends the empty message and calls nothing. -/
def silentP : Prover Unit := ⟨fun _ _ _ => .ret []⟩

/-- The extractor that gives up at once. -/
def noneE {Coin W : Type} (V : Verifier Coin) : Extractor Coin W V where
  C _ := Unit
  ext _ _ _ := .ret none

/-- The extractor that reruns the judged run once and gives up. -/
def onceE {Coin W : Type} (V : Verifier Coin) : Extractor Coin W V where
  C _ := Unit
  ext _ r₀ _ := .rerun r₀ fun _ => .ret none

/-- The finder that outputs nothing. -/
def noFinder : Finder where
  S := Unit
  out _ := none
  cost _ := 0

/-- `noFinder` at every prover and statement. -/
def noFinders : FinderFamily := fun _ _ _ _ _ => noFinder

/-- The constant hash, whose outputs are 64 bytes. -/
def constH : Msg → Msg := fun _ => List.replicate 64 0

/-- The finder that outputs `([0], [1])` at no cost. -/
def pairFinder : Finder where
  S := Unit
  out _ := some ([0], [1])
  cost _ := 0

section Run

variable {Coin : Type} [Inhabited Coin]

theorem run_accV_out {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((accV Coin).k x) → Coin) :
    (run H P (accV Coin) x ω r).out = .acc := by
  simp [run, accV, stepWith, OC.eval]

theorem run_accV_vLog {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((accV Coin).k x) → Coin) : (run H P (accV Coin) x ω r).vLog = [] := by
  simp [run, accV, stepWith, OC.eval]

theorem run_silentP_accV_pLog (H : Msg → Msg) (x : Msg) (ω : Unit) (r : Fin ((accV Coin).k x) → Coin) :
    (run H silentP (accV Coin) x ω r).pLog = [] := by
  simp [run, accV, silentP, stepWith, OC.eval]

theorem run_rejV_out {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((rejV Coin).k x) → Coin) :
    (run H P (rejV Coin) x ω r).out = .rej := by
  simp [run, rejV, stepWith, OC.eval]

end Run

section Finders

theorem noFinder_pr (H : Msg → Msg) : noFinder.Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by simp [noFinder])]
  exact Pr_false

theorem pairFinder_pr : pairFinder.Pr constH = 1 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_true_intro ⟨[0], [1], rfl, by decide, rfl⟩]
  exact ZKMal.Pr_true

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `B` finds nothing against `accV`, which calls nothing. -/
theorem B_accV_pr {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (P : Prover Ω) (x : Msg)
    (hk : 0 < (accV Coin).k x) : (B H (accV Coin) P x hk).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by simp [B, run_accV_vLog, findColl])]
  exact Pr_false

theorem prB_accV {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (P : Prover Ω) (x : Msg) :
    PrB H (accV Coin) P x = 0 := by
  unfold PrB
  split_ifs with hk
  · exact B_accV_pr H P x hk
  · rfl

theorem strictBirthday_B_accV {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (P : Prover Ω) (x : Msg)
    (hk : 0 < (accV Coin).k x) : StrictBirthday H (B H (accV Coin) P x hk) := fun q _ => by
  rw [B_accV_pr]; positivity

/-- At `accV`, on a statement with no witness, every extractor fails on every run. -/
theorem prBad_accV {W Ω : Type} [Fintype Ω] (H : Msg → Msg) (R : Rel W) (E : Extractor Coin W (accV Coin))
    (x : Msg) (P : Prover Ω) [Nonempty Ω] (hR : ∀ w, ¬ R x w) : PrBad H (accV Coin) R E x P = 1 := by
  have hb : ∀ s : Ω × (Fin ((accV Coin).k x) → Coin) × E.C x,
      ((run H P (accV Coin) x s.1 s.2.1).out = .acc ∧ ¬ R.ext x (E.eval H P x s.1 s.2.1 s.2.2).1) ↔ True := by
    intro s
    refine iff_true_intro ⟨run_accV_out H P x s.1 s.2.1, ?_⟩
    cases (E.eval H P x s.1 s.2.1 s.2.2).1 with
    | none => exact id
    | some w => exact hR w
  unfold PrBad
  rw [Pr_congr hb]
  exact ZKMal.Pr_true

end Finders

section Claims

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`KS` fails** at `accV` on a statement with no witness, at every extractor, every `κ < 1` and every `γ`. -/
theorem KS_fails (H : Msg → Msg) (R : Rel W) (E : Extractor Coin W (accV Coin)) (κ γ : Msg → ℝ) (x : Msg)
    (hR : ∀ w, ¬ R x w) (hκ : κ x < 1) : ¬ KS H (accV Coin) R E κ γ := by
  intro h
  have h1 := h x Unit silentP
  rw [prBad_accV H R E x silentP hR, prB_accV, mul_zero, Real.sqrt_zero, add_zero] at h1
  linarith

/-- **`KSLin` fails** there too, at the finders that find nothing and every `β`. -/
theorem KSLin_fails (H : Msg → Msg) (R : Rel W) (E : Extractor Coin W (accV Coin)) (κ γ β : Msg → ℝ) (x : Msg)
    (hR : ∀ w, ¬ R x w) (hκ : κ x < 1) : ¬ KSLin H (accV Coin) R E κ γ β noFinders := by
  intro h
  have h1 := h x Unit silentP
  rw [prBad_accV H R E x silentP hR, prB_accV, mul_zero, Real.sqrt_zero, add_zero, noFinders, noFinder_pr,
    mul_zero, add_zero] at h1
  linarith

omit [Fintype Coin] [Nonempty Coin] in
/-- `silentP` against `accV` makes no SHA-512 call. -/
theorem calls_silentP_accV (H : Msg → Msg) (x : Msg) (ω : Unit) (r : Fin ((accV Coin).k x) → Coin) :
    (run H silentP (accV Coin) x ω r).pLog.length + (run H silentP (accV Coin) x ω r).vLog.length ≤ 1 := by
  rw [run_silentP_accV_pLog, run_accV_vLog]; simp

/-- **`Bits` fails** at `accV` on a statement with no witness, at every extractor and every `λ > 0`. -/
theorem Bits_fails (H : Msg → Msg) (R : Rel W) (E : Extractor Coin W (accV Coin)) (lam : Msg → ℝ) (x : Msg)
    (hR : ∀ w, ¬ R x w) (hlam : 0 < lam x) : ¬ Bits H (accV Coin) R E lam := by
  intro h
  have h1 := h x Unit silentP 1 le_rfl (calls_silentP_accV H x) (strictBirthday_B_accV H silentP x)
  rw [prBad_accV H R E x silentP hR, Nat.cast_one, one_mul] at h1
  have : (2 : ℝ) ^ (-lam x) < 1 := Real.rpow_lt_one_of_one_lt_of_neg (by norm_num) (by linarith)
  linarith

-- The `first` alternatives cover `Complete` with and without its `Out64 H` premise, so one proof elaborates on both.
set_option linter.unreachableTactic false in
set_option linter.unusedTactic false in
omit [Nonempty Coin] in
/-- **`Complete` fails** at `rejV`, at every prover with a tape and every `δ_c < 1`, at a statement in scope with a
witness. The hash is `constH`, whose outputs are 64 bytes. -/
theorem Complete_fails {Ω : Type} [Nonempty Ω] (P : W → Prover Ω) (R : Rel W) (Scope : Msg → Prop) (δc : Msg → ℝ)
    {x : Msg} {w : W} (hs : Scope x) (hr : R x w) (hδ : δc x < 1) :
    ¬ Complete constH (rejV Coin) P R Scope δc := by
  intro h
  have h1 : 1 - δc x ≤
      Pr fun r : Fin ((rejV Coin).k x) → Coin => (run constH (P w) (rejV Coin) x (Classical.arbitrary Ω) r).out = .acc := by
    first
    | exact h x w hs hr _
    | exact h (fun _ => List.length_replicate ..) x w hs hr _
  rw [Pr_congr fun r => iff_false_intro (by rw [run_rejV_out]; exact Out.noConfusion), Pr_false] at h1
  linarith

/-- **`ScopeMatches` fails** at an empty scope and a check that takes everything. -/
theorem ScopeMatches_fails : ¬ ScopeMatches (fun _ => False) (fun _ => true) := fun h => (h []).2 rfl

/-- **`ZKmalR` fails** at the colliding simulator, the instance that met the form that charged the simulator's calls,
at the quiet prover, every `ε_zk < 1` and every `γ`. -/
theorem ZKmalR_fails {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (u v : Msg)
    (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1) :
    ¬ ZKmalR H (fun _ => 1) (fun _ : Unit => ZKMal.quietP N Ω []) (fun _ _ => True) (ZKMal.collSim u v) εzk γ :=
  ZKMal.not_zkmalR_collSim H _ _ u v εzk γ x () trivial one_pos hε

/-- The simulator that runs the quiet prover against the verifier for `k x` rounds. -/
def quietSim (k : Msg → ℕ) : Simulator where
  C _ := Unit
  prog x _ := ZKMal.spRun (fun _ => .ret []) (k x) []

/-- **`ZKmalR` is met by a prover that ignores its witness**, at `ε_zk = 0`, `γ = 0` and every relation: the quiet
prover's transcript against any `V*` is `quietSim`'s. -/
theorem ZKmalR_vacuous {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (k : Msg → ℕ) (R : Rel W) : ZKmalR H k (fun _ => ZKMal.quietP N Ω []) R (quietSim k) 0 0 := by
  intro x w _ Θ Vs θ hk Ev
  let τ := ZKMal.interact (fun as => ((Vs.next x as θ).eval H).1) (fun _ => []) (k x) []
  have hτ : ∀ ω, (runStar H (ZKMal.quietP N Ω []) Vs x ω θ).msgs = τ := fun ω => by
    rw [ZKMal.runStar_msgs, hk]
    rfl
  have hS : ∀ c, (quietSim k).out H Vs θ x c = τ := fun _ =>
    ZKMal.spRun_eval H (fun as => ((Vs.next x as θ).eval H).1) (fun _ => .ret []) (k x) []
  have e₁ : (Pr fun ω : N × Ω => Ev (runStar H (ZKMal.quietP N Ω []) Vs x ω θ).msgs) = Pr fun _ : N × Ω => Ev τ :=
    Pr_congr fun ω => by rw [hτ ω]
  have e₂ : (Pr fun c : (quietSim k).C x => Ev ((quietSim k).out H Vs θ x c)) =
      Pr fun _ : (quietSim k).C x => Ev τ :=
    Pr_congr fun c => by rw [hS c]
  rw [e₁, e₂]
  by_cases hE : Ev τ
  · simp only [hE, ZKMal.Pr_true]; simp
  · simp only [hE, Pr_false]; simp

/-- **`SimCost` fails** at the colliding simulator, whose two calls exceed a bound of `1`. -/
theorem SimCost_fails (H : Msg → Msg) (u v : Msg) : ¬ SimCost H (ZKMal.collSim u v) fun _ => 1 := by
  intro h
  have h1 := h [] fun _ => .rej
  have hc : (fun c : (ZKMal.collSim u v).C [] =>
      ((((ZKMal.collSim u v).prog [] c).eval H fun _ => .rej).2 : ℝ)) = fun _ => 2 :=
    funext fun _ => by simp [ZKMal.collSim, SP.eval]
  rw [hc, Ex_const] at h1
  norm_num at h1

omit [Fintype Coin] [Nonempty Coin] in
/-- **`ExtractRunsLe` fails** at `onceE`, which reruns once, at a bound of `0`. -/
theorem ExtractRunsLe_fails (H : Msg → Msg) (V : Verifier Coin) :
    ¬ ExtractRunsLe H (onceE V : Extractor Coin W V) fun _ => 0 := by
  intro h
  have h1 := h [] Unit silentP () (fun _ => default) ()
  simp [Extractor.eval, onceE, RP.eval] at h1

/-- **`ExtractRunsExpLe` fails** at `onceE`, at a bound of `0`. -/
theorem ExtractRunsExpLe_fails (H : Msg → Msg) (V : Verifier Coin) :
    ¬ ExtractRunsExpLe H (onceE V : Extractor Coin W V) fun _ => 0 := by
  intro h
  have h1 := h [] Unit silentP ()
  have hc : (fun s : (Fin (V.k []) → Coin) × (onceE V : Extractor Coin W V).C [] =>
      (((onceE V : Extractor Coin W V).eval H silentP [] () s.1 s.2).2.2 : ℝ)) = fun _ => 1 :=
    funext fun _ => by simp [Extractor.eval, onceE, RP.eval]
  rw [hc, Ex_const] at h1
  norm_num at h1

omit [Fintype Coin] [Nonempty Coin] in
/-- **`Nontrivial` fails** at a recorded run that `rejV` rejects. -/
theorem Nontrivial_fails (H : Msg → Msg) :
    ¬ Nontrivial H (rejV Coin) [⟨[], [], fun _ => default⟩] := by
  rintro ⟨-, h⟩
  have h1 := h _ (List.mem_singleton_self _)
  rw [run_rejV_out] at h1
  exact Out.noConfusion h1

end Claims

section Assumptions

/-- **`StrictBirthday` fails** at `constH` and `pairFinder`, which finds its collision with no call. -/
theorem StrictBirthday_fails : ¬ StrictBirthday constH pairFinder := by
  intro h
  have h1 := h 0 fun _ => by simp [pairFinder]
  rw [pairFinder_pr] at h1
  norm_num at h1

/-- `StrictBirthday` is satisfiable: `noFinder` meets it at every hash. -/
theorem StrictBirthday_holds (H : Msg → Msg) : StrictBirthday H noFinder := fun q _ => by
  rw [noFinder_pr]; positivity

/-- **`ExpectedCR` fails** at `constH` and `pairFinder`. -/
theorem ExpectedCR_fails : ¬ ExpectedCR constH pairFinder := by
  intro h
  unfold ExpectedCR at h
  rw [pairFinder_pr] at h
  have hc : (fun s : pairFinder.S => (pairFinder.cost s : ℝ)) = fun _ => 0 := funext fun _ => by simp [pairFinder]
  rw [hc, Ex_const] at h
  norm_num at h

/-- `ExpectedCR` is satisfiable: `noFinder` meets it at every hash. -/
theorem ExpectedCR_holds (H : Msg → Msg) : ExpectedCR H noFinder := by
  unfold ExpectedCR
  rw [noFinder_pr]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

/-- At a constant mask, hm96's hiding needs `δ₁ ≥ 1/2`: the mask is `true` surely, a uniform bit half the time. -/
theorem half_le_of_hm96Hiding_const {δ₁ : ℝ≥0∞} (h : Hm96Hiding (fun _ : Unit => true) (id : Unit → Unit) δ₁) :
    2⁻¹ ≤ δ₁ := by
  have h1 := h fun p => p.1 = true
  simp [PrENNReal, Fintype.sum_prod_type] at h1
  rw [← ENNReal.inv_two_add_inv_two] at h1
  exact (ENNReal.add_le_add_iff_left (ENNReal.inv_ne_top.2 two_ne_zero)).1 h1

/-- **`Hm96Hiding` fails** at a constant mask and `δ₁ = 0`. -/
theorem Hm96Hiding_fails : ¬ Hm96Hiding (fun _ : Unit => true) (id : Unit → Unit) 0 := fun h => by
  have := half_le_of_hm96Hiding_const h
  simp at this

/-- `Hm96Hiding` is satisfiable: a uniform mask (the salt itself) with a constant salt hash hides at `δ₁ = 0`. -/
theorem Hm96Hiding_holds {Y : Type} [Fintype Y] [Nonempty Y] : Hm96Hiding (id : Y → Y) (fun _ => ()) 0 := by
  intro E
  classical
  have hn : (Fintype.card Y : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have ht : (Fintype.card Y : ℝ≥0∞) ≠ ∞ := ENNReal.natCast_ne_top _
  have hc : (Fintype.card Y : ℝ≥0∞) * ((Fintype.card Y : ℝ≥0∞) * Fintype.card Y)⁻¹ = (Fintype.card Y : ℝ≥0∞)⁻¹ := by
    rw [ENNReal.mul_inv (Or.inl hn) (Or.inl ht), ← mul_assoc, ENNReal.mul_inv_cancel hn ht, one_mul]
  have key : ∀ x : Y, (∑ _y : Y, ((Fintype.card (Y × Y) : ℝ≥0∞))⁻¹ * (if E (x, ()) then 1 else 0)) =
      (Fintype.card Y : ℝ≥0∞)⁻¹ * (if E (x, ()) then 1 else 0) := fun x => by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Fintype.card_prod, Nat.cast_mul, ← mul_assoc, hc]
  unfold PrENNReal
  rw [Fintype.sum_prod_type, add_zero]
  exact le_of_eq (Finset.sum_congr rfl fun x _ => (key x).symm)

/-- **`JsonSpec` fails** at a canonical printer that never returns. -/
theorem JsonSpec_fails (print : Lean.Json → String) : ¬ JsonSpec print fun _ => .error "" := fun h => by
  obtain ⟨_, hs⟩ := h.canon_ok .null Claims.Json.Good.null
  cases hs

end Assumptions

end Claims.Witness
