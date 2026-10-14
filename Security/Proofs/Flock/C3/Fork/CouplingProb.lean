import Proofs.Flock.C3.Fork.CouplingBad
import Proofs.Flock.C3.Fork.CouplingValue

/-!
# a1l, step 7: the compiled holographic table on a clash-free set, against one prover

`holoTableCL_prob_le`: on a set `W` of outcomes no two of which clash, a prover makes the compiled holographic table
end accepted in `W` no more often than the oracle table's value at the registered word. It is `table_prob_clashFree`'s
argument on `lockHoloL`: the oracle prover sends `W`'s tables (`tab₀W` at the prover's cap, `chH` at the reps' and R's),
so `holo_bad_zero` leaves the coupling no bad term, and the coupling's oracle game is no better than the oracle table
(`holoOrcL_value_le`).
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Recursion
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

/-- **The compiled holographic table on a clash-free set, against one prover**: accepted in `W` no more often than
the oracle table's value at the registered word `x.w`. -/
theorem holoTableCL_prob_le [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hlen : 2 ≤ sch.levels.length)
    (hk0 : (sch.levels.headD default).lanes = sch.k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (sh : SparseShape)
    (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule)
    (hsh : rsch.levels = [] ∨ 2 ≤ rsch.levels.length ∧ (rsch.levels.headD default).lanes = rsch.k0)
    (x : HoloReg F D) (W : OutHL F D Sl → Prop) (hW : ∀ o o', W o → W o' → ¬ ClashHL H E Lf sch rsch x o o')
    (σ : Strategy (holoTableCL A H E Lf S sch hm ptLocal mPts sh units τ rsch x.cap)) :
    prob (fun o : OutHL F D Sl => o.accepted = true ∧ W o)
        (holoTableCL A H E Lf S sch hm ptLocal mPts sh units τ rsch x.cap) σ ≤
      value (fun o : Bool × Oracle F => o.1 = true) (holoTable A S sch hm ptLocal mPts sh units τ rsch x.w) := by
  obtain ⟨m, k0, levels, ext⟩ := sch
  obtain _ | ⟨l₀, _ | ⟨l₁, rest⟩⟩ := levels
  · simp at hlen
  · simp at hlen
  revert σ
  show ∀ σ : Strategy (Game.send (Cap D)
    (holoTableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts sh units τ rsch x.cap)), _
  intro σ
  have hfix₀ : ∀ s t d, getEO (Lock.getPair (getRep 0) (getRep 1)) (getRep 0)
      ((((fun _ => none), (fun _ => none)), (fun _ => none)) : (IRep F D × IRep F D) × IRep F D) s =
        some (t, d) → t = chH H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ rsch W x s d := by
    intro s t d h
    simp [getEO, Lock.getPair, getRep] at h
  have L := lockHoloL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts sh units τ rsch hsh
    (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (WC W) σ.1) x.w σ.1 x.cap
  have hcpl := Lock.expect_le_coupled (fun s d _ _ => chH H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ rsch W x s d)
    (fun out : OutHL F D Sl => out.accepted = true ∧ W out) (fun v : Bool => v = true)
    (fun (i : (IRep F D × IRep F D) × IRep F D) (out : OutHL F D Sl) => out.accepted = true ∧ W out ∧
      ¬ (AgreeRep 0 (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (WC W) σ.1) i.1.1 out.pend.1.reqs out.opens ∧
        AgreeRep 1 (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (WC W) σ.1) i.1.2 out.pend.2.reqs out.opens ∧
        AgreeRep 2 x.w i.2 out.pendR.reqs out.opens))
    (fun i a b hR hE hB => by
      obtain ⟨-, hacc, -⟩ := hR
      by_contra hn
      exact hB ⟨hE.1, hE.2, fun h => hn (hacc hE.1 h.1 h.2.1 h.2.2)⟩)
    L σ.2
  have hbad : Lock.leafE (fun s d _ _ => chH H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ rsch W x s d)
      (fun (i : (IRep F D × IRep F D) × IRep F D) (out : OutHL F D Sl) => ind (fun out : OutHL F D Sl =>
        out.accepted = true ∧ W out ∧
          ¬ (AgreeRep 0 (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (WC W) σ.1) i.1.1 out.pend.1.reqs out.opens ∧
            AgreeRep 1 (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (WC W) σ.1) i.1.2 out.pend.2.reqs out.opens ∧
            AgreeRep 2 x.w i.2 out.pendR.reqs out.opens)) out) L σ.2 ≤ 0 := by
    have h := leafE_le_of_fixed (chH H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ rsch W x) (f' := fun _ _ => 0)
      (fun i a b hR hi => holo_bad_zero H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ rsch x W hW σ.1 i a b hR hi)
      L σ.2 hfix₀
    rw [Lock.leafE_const _ (fun _ => 0), expect_const] at h
    exact h
  have horc : expect (ind fun v : Bool => v = true)
      (holoOrcL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts sh units τ rsch x.w
        (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (WC W) σ.1))
      (Lock.oracleStrat (fun s d _ _ => chH H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ rsch W x s d) L σ.2) ≤
      (value (fun o : Bool × Oracle F => o.1 = true)
        (holoTable A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts sh units τ rsch x.w)).toReal := by
    rw [← prob_toReal]
    exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (value_le_one _ _))
      ((prob_le_value _ _ _).trans (holoOrcL_value_le A S _ hm ptLocal mPts sh units τ rsch x.w _))
  have e : (prob (fun o : OutHL F D Sl => o.accepted = true ∧ W o)
      (holoTableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts sh units τ rsch x.cap) σ).toReal =
    expect (ind fun o : OutHL F D Sl => o.accepted = true ∧ W o)
      (holoTableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts sh units τ rsch x.cap σ.1) σ.2 :=
    prob_toReal _ _ σ
  refine (ENNReal.toReal_le_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (ne_top_of_le_ne_top ENNReal.one_ne_top (value_le_one _ _))).1 ?_
  linarith

end FlockC3.Fork
