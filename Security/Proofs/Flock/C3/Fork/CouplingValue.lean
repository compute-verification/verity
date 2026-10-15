import Proofs.Flock.Soundness.Game.Union
import Proofs.Flock.C3.Fork.CouplingLock

/-!
# a1l, step 4: the coupling's oracle game is no better than the oracle table

`holoOrcL_value_le`: at any level-0 table `C₀`, the oracle game of `lockHoloL` is accepted no more often than
`holoTable` at the same registered word. `holoRep` is `holoRepO` with a rep that Ligerito rejects handing on nothing
(`holoRep_eq_map`), so where both reps' Ligerito accept, the two games are the same; elsewhere the oracle game
rejects.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}

theorem value_coin_mono {β γ C : Type} [Fintype C] [Nonempty C] {p : β → Prop} {q : γ → Prop}
    {k : C → Game β} {k' : C → Game γ} (h : ∀ c, value p (k c) ≤ value q (k' c)) :
    value p (Game.coin C k) ≤ value q (Game.coin C k') :=
  value_bind_mono h (Game.coin C Game.ret)

/-- `holoRep` is `holoRepO` handing on nothing where Ligerito rejects. -/
theorem holoRep_eq_map (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) :
    holoRep A S sch hm C₀ extra = (holoRepO A S sch hm C₀ extra).map fun p => if p.2 then p.1 else none := by
  simp only [holoRep, holoRepO, Game.map, Game.bind_eq, Game.pure_eq, Game.bind_assoc]
  refine congrArg (Game.bind _) (funext fun zc => ?_)
  rcases zc with _ | zc
  · rfl
  dsimp only
  simp only [Game.bind_assoc]
  refine congrArg (Game.bind _) (funext fun lc => ?_)
  rcases lc with _ | lc
  · rfl
  dsimp only
  simp only [Game.bind_assoc]
  refine congrArg (Game.bind _) (funext fun op => ?_)
  rcases op with _ | op
  · rfl
  dsimp only
  simp only [Game.bind_assoc, Game.bind_ret]

/-- **The coupling's oracle game against the oracle table**: at any level-0 table `C₀`, `holoOrcL` accepts no more
often than `holoTable` does at the same registered word `w`. -/
theorem holoOrcL_value_le {Sl : Type} (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ)
    (rsch : Schedule) (w C₀ : Oracle F) :
    value (fun b : Bool => b = true)
        (holoOrcL (D := D) (Sl := Sl) A S sch hm ptLocal mPts sh units τ rsch w C₀) ≤
      value (fun o : Bool × Oracle F => o.1 = true) (holoTable A S sch hm ptLocal mPts sh units τ rsch w) := by
  let body : Oracle F → Game (Bool × Oracle F) := fun C => Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    (interleave (holoRep A S sch hm C (extraClaims S ptLocal mPts pts))
      (holoRep A S sch hm C (extraClaims S ptLocal mPts pts))).bind fun reps =>
    (regPhase A S sh units τ false (fun op => ligerito A rsch op.T op.b w) reps).bind fun ok => Game.ret (ok, C)
  have htab : value (fun o : Bool × Oracle F => o.1 = true) (holoTable A S sch hm ptLocal mPts sh units τ rsch w) =
      ⨆ C, value (fun o : Bool × Oracle F => o.1 = true) (body C) := rfl
  rw [htab]
  refine le_trans ?_ (le_iSup (fun C => value (fun o : Bool × Oracle F => o.1 = true) (body C)) C₀)
  refine value_coin_mono fun pts => ?_
  rw [holoRep_eq_map, interleave_map]
  simp only [Game.map, Game.bind_assoc, Game.bind_ret]
  refine value_bind_mono (fun reps => ?_) _
  obtain ⟨⟨o₁, v₁⟩, ⟨o₂, v₂⟩⟩ := reps
  by_cases hv : v₁ = true ∧ v₂ = true
  · obtain ⟨rfl, rfl⟩ := hv
    refine value_bind_mono (fun okR => ?_) _
    simp only [value_send, value_ret, Bool.true_and]
    exact iSup_le fun _ => le_rfl
  · refine le_trans (value_bind_le_of_forall _ _ _ 0 fun okR => ?_) zero_le
    have hf : (v₁ && v₂ && okR) = false := by
      cases v₁ <;> cases v₂ <;> simp_all
    simp only [value_send, value_ret, hf]
    exact iSup_le fun _ => by simp
