import Proofs.Flock.Recursive.ForkInnerC
import Proofs.Flock.Recursive.ValueFinder
import Proofs.Flock.Recursive.LeafFinder
import Proofs.Flock.Recursive.SaltFinder

/-!
# Recursion: gap 2 off a registrant's row off a verifier's value

`zk_sessions_recursive_fork_on` and `zk_sessions_recursive_fork_innerL` on V*'s bridge off
`RegColl ∨ ValueColl ∨ LeafColl ∨ SaltColl ∨ TopColl` (`Flock.Assumptions.VBridgeCOff` at
`ValueColl zs vsrc ∨ LeafColl zs ∨ SaltColl lv cap salt zs ∨ TopColl wr st reg bcs'`), which
`FlockC3.RegLevels.value_tie`'s collision case, a read's leaf off the table's, a read of R's level-0 top off its node's
row under the held salts and a held top other than the registered one let the discharge use (`value_tie_coll`,
`FlockC3.RegLevels.leaf_of_accepts`, `FlockC3.RegLevels.salt_tie`, `topOf_held`): `valueFinder`, `leafFinder`,
`saltFinder` and `topFinder` finding a collision are four more forks, and off them and `regFinder`'s the bad event has
probability `0` (`prob_valueColl_le`, `prob_leafColl_le`, `prob_saltColl_le`, `prob_topCollAt_le`, `prob_regColl_le` at
no budget).
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

section Chain

open Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL Composed
  ZkReg

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **`zk_sessions_recursive_fork_on` off `RegColl ∨ ValueColl ∨ LeafColl ∨ SaltColl ∨ TopColl`**: the value, the
table leaves', the held salts' or the held top's finder finds a collision, or `zk_sessions_recursive_fork_on`'s
conclusion. -/
theorem zk_sessions_recursive_fork_onV
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (lv : FlockSoundness.Accounting.Level) (cap : Cap Refine.Dig)
    (salt : ℕ → ByteArray) (wr : List (Cm × I.Coin) → Lean.Json × String) (st : Flock.Stmt) (reg : Lean.Json)
    (bcs' : Array ByteArray) (hstk : StackedTopAt lv st reg cap salt bcs')
    (dg : ByteArray → ByteArray) (N : ℕ) {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : VBridgeOff (fun R ω t R₂ => RegColl zs R ω t R₂ ∨ ValueColl zs vsrc R ω t R₂ ∨ LeafColl zs R ω t R₂ ∨
      SaltColl lv cap salt zs R ω t R₂ ∨ TopColl wr st reg bcs' t) zs dg N k Rw x wmsg)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (rd : (L.Ω → List (Cm × I.Coin) → Cm → I.Msg) → L.Ω → List (Cm × I.Coin) → List (I.Msg × I.Coin) → Prop)
    (hrd : ∀ (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) ω t (l : List I.Msg), t.length = I.r →
      (∀ k < I.r, Option.Rel (same k (t.map Prod.snd)) l[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) →
      rd ref ω t (l.zip (t.map Prod.snd)))
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hcmt : VCommits zs Sk cmt) (hdec : VDecodesR wmsg same Sk msg) (hfull : VFullR I.r Sk)
    {εin : ℝ≥0∞} (hin : ∀ ref, InnerAtLeavesVZ zs dg N k Rw σ x wmsg (rd ref) εin) :
    (valueFinder zs vsrc dg N σ).Finds H512 ∨ (leafFinder zs dg N σ).Finds H512 ∨
      (saltFinder lv cap salt zs σ).Finds H512 ∨ (topFinder wr st reg bcs' lv cap salt hstk zs σ).Finds H512 ∨
      (regFinder zs dg N σ).Finds H512 ∨
      RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin := by
  by_cases hv : (valueFinder zs vsrc dg N σ).Finds H512
  · exact Or.inl hv
  refine Or.inr ?_
  by_cases hl : (leafFinder zs dg N σ).Finds H512
  · exact Or.inl hl
  refine Or.inr ?_
  by_cases hsa : (saltFinder lv cap salt zs σ).Finds H512
  · exact Or.inl hsa
  refine Or.inr ?_
  by_cases hto : (topFinder wr st reg bcs' lv cap salt hstk zs σ).Finds H512
  · exact Or.inl hto
  refine Or.inr ?_
  by_cases h0 : (regFinder zs dg N σ).Finds H512
  · exact Or.inl h0
  by_cases h1 : RoundFork zs dg N k Rw σ cmt
  · exact Or.inr (Or.inl h1)
  by_cases h2 : SessionFork zs dg N k Rw M σ
  · exact Or.inr (Or.inr (Or.inl h2))
  refine Or.inr (Or.inr (Or.inr ?_))
  have hno := allAt_of_not_exAt h2
  have hB : expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ 0 0 0 0 0 τ) s) ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) := by
    refine expAt_mono_all (hno.mono fun ω t s hs => ?_)
    unfold sumV sumS
    exact exG_mono_outcome _ _ fun rs hrs => Finset.sum_le_sum fun j _ =>
      ZkOuter.boundIf_le _ _ _ _ _ fun hf => hs ⟨rs, hrs, j, hf⟩
  have hsl : expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackIf dg N k Rw M ρ 0 0 0 τ) s) ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) := by
    refine expAt_mono_all (hno.mono fun ω t s hs => ?_)
    unfold sumV sumS
    exact exG_mono_outcome _ _ fun rs hrs => Finset.sum_le_sum fun j _ =>
      ZkOuter.slackIf_le _ _ _ _ fun hf => hs ⟨rs, hrs, j, hf⟩
  have hX : prob (fun o => RegColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 ∨ ValueColl zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1 ∨
        LeafColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 ∨ SaltColl lv cap salt zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 ∨
        TopColl wr st reg bcs' o.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      ENNReal.ofReal ((0 : ℝ) ^ 2 / 2 ^ (513 : ℝ)) +
        (ENNReal.ofReal ((0 : ℝ) ^ 2 / 2 ^ (513 : ℝ)) + (ENNReal.ofReal ((0 : ℝ) ^ 2 / 2 ^ (513 : ℝ)) +
          (ENNReal.ofReal ((0 : ℝ) ^ 2 / 2 ^ (513 : ℝ)) + ENNReal.ofReal ((0 : ℝ) ^ 2 / 2 ^ (513 : ℝ))))) :=
    (prob_or_le _ _ _ _).trans (add_le_add (prob_regColl_le zs dg N σ (Finder.cr_of_not_finds h0 0))
      ((prob_or_le _ _ _ _).trans (add_le_add (prob_valueColl_le zs vsrc dg N σ (Finder.cr_of_not_finds hv 0))
        ((prob_or_le _ _ _ _).trans (add_le_add (prob_leafColl_le zs dg N σ (Finder.cr_of_not_finds hl 0))
          ((prob_or_le _ _ _ _).trans (add_le_add (prob_saltColl_le lv cap salt zs σ (Finder.cr_of_not_finds hsa 0))
            (prob_topCollAt_le wr st reg bcs' lv cap salt hstk zs σ (Finder.cr_of_not_finds hto 0)))))))))
  refine (zk_sessions_recursive_full_on hA3 zs dg N hk hRw hM hρ hr le_rfl 0 0 0 0 σ x wmsg
    (fun R ω t R₂ => RegColl zs R ω t R₂ ∨ ValueColl zs vsrc R ω t R₂ ∨ LeafColl zs R ω t R₂ ∨
      SaltColl lv cap salt zs R ω t R₂ ∨ TopColl wr st reg bcs' t) hbr hX same rd hrd Sk cmt msg hcmt hdec hfull
    (fun _ => 0) (fun i hi ω pos => Finder.cr_of_not_finds (fun hf => h1 ⟨i, hi, ω, pos, hf⟩) 0) hin).trans
    (add_le_add (add_le_add ?_ ?_) le_rfl)
  · simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, ENNReal.ofReal_zero, add_zero]
    exact hB
  · simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, Real.sqrt_zero, ENNReal.ofReal_zero,
      mul_zero, add_zero]
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    exact mul_le_mul' le_rfl hsl

/-- **Gap 2 at the committed rounds off `RegColl ∨ ValueColl ∨ LeafColl ∨ SaltColl ∨ TopColl`, the inner term
given**: `zk_sessions_recursive_fork_innerL` with V*'s bridge as
`Flock.Assumptions.VBridgeCOff (ValueColl zs vsrc ∨ LeafColl zs ∨ SaltColl lv cap salt zs ∨ TopColl wr st reg bcs')`,
the value, the table leaves', the held salts' and the held top's finders finding a collision four more forks. -/
theorem zk_sessions_recursive_fork_innerLV
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (lv : FlockSoundness.Accounting.Level) (cap : Cap Refine.Dig)
    (salt : ℕ → ByteArray) (wr : List (Cm × I.Coin) → Lean.Json × String) (st : Flock.Stmt) (reg : Lean.Json)
    (bcs' : Array ByteArray) (hstk : StackedTopAt lv st reg cap salt bcs')
    (dg : ByteArray → ByteArray) (N : ℕ) {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hbr : Flock.Assumptions.VBridgeCOff (fun R ω t R₂ => ValueColl zs vsrc R ω t R₂ ∨ LeafColl zs R ω t R₂ ∨
      SaltColl lv cap salt zs R ω t R₂ ∨ TopColl wr st reg bcs' t) zs dg N k Rw x wmsg Sk cmt msg)
    {D : Type} (H : List UInt8 → D) (pick : I.Stmt → List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair)
    {εin : ℝ≥0∞} (hr0 : 0 < I.r)
    (hleaf : ¬ InnerForkRec zs dg N k Rw σ x wmsg H pick →
      ∀ ref, InnerAtLeavesL σ (fun R ω t o => AcceptsV (zs R ω t) o) x (W := ((j : Fin S) → (Vs j).Bits) × I.Stmt)
        (fun R ω t _ o => (layers (zs R ω t o.1) dg N k Rw o.2, x R ω)) (fun w => wmsg w.1) ref εin) :
    (valueFinder zs vsrc dg N σ).Finds H512 ∨ (leafFinder zs dg N σ).Finds H512 ∨
      (saltFinder lv cap salt zs σ).Finds H512 ∨ (topFinder wr st reg bcs' lv cap salt hstk zs σ).Finds H512 ∨
      (regFinder zs dg N σ).Finds H512 ∨
      RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨ InnerForkRec zs dg N k Rw σ x wmsg H pick ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin := by
  by_cases hf : InnerForkRec zs dg N k Rw σ x wmsg H pick
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl hf)))))))
  have hrd : ∀ (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) ω t (l : List I.Msg), t.length = I.r →
      (∀ k < I.r, Option.Rel (fun a b => k + 1 < I.r → a = b) l[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) →
      ∃ m, l.zip (t.map Prod.snd) = relabelL (ref ω) m t := by
    intro ref ω t l ht h
    obtain ⟨m, hm⟩ := exists_of_rel (h (t.length - 1) (by omega)) (by
      simp only [ne_eq, Option.map_eq_none_iff, List.getElem?_eq_none_iff, Nat.not_le]; omega)
    exact ⟨m, zip_eq_relabelL (ref ω) m t l hm fun k hk =>
      option_eq_of_rel (R := fun a b => k + 1 < I.r → a = b) (fun _ _ hab => hab (by omega)) (h k (by omega))⟩
  rcases zk_sessions_recursive_fork_onV hA3 zs vsrc lv cap salt wr st reg bcs' hstk dg N hk hRw hM hρ hr σ x wmsg hbr.1
      (fun i _ a b => i + 1 < I.r → a = b) (fun ref ω t u => ∃ m, u = relabelL (ref ω) m t) hrd
      (fun i j => if i + 1 < I.r then Sk i j else ∅) cmt msg hbr.2.1 hbr.2.2.1 hbr.2.2.2
      (fun ref => InnerAtLeavesL.toZ (hleaf hf ref)) with h | h | h | h | h | h | h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr (Or.inl h))
  · exact Or.inr (Or.inr (Or.inr (Or.inl h)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h)))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr h)))))))

end Chain

end FlockSoundness.Discharge.Recursion

#print axioms FlockSoundness.Discharge.Recursion.zk_sessions_recursive_fork_onV

#print axioms FlockSoundness.Discharge.Recursion.zk_sessions_recursive_fork_innerLV
