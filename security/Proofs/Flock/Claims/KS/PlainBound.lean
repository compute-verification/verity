module

public import Proofs.Flock.Claims.KS.Mixture
public import Proofs.Flock.Claims.KS.Rounds
public import Proofs.Flock.Claims.KS.SessionSC
public import Proofs.Flock.Claims.KS.TableBound
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Soundness.Game

/-! The plain table bound at a statement of one table from a mixture of salted sessions after `Commit`, its hypotheses on the deployed prover left open. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec

local notation "𝔽" => FlockLevel3.GF128

theorem tables_head_one {zk : Bool} {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h : tables zk x = [(spec, st)]) : (tables zk x).head? = some (spec, st) := by
  rw [h]
  rfl

open Classical in
theorem plain_le_sessionsEV (Vr : Claims.Verifier Coin) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts) {Ω : Type} (P : Claims.Prover Ω)
    (e : Claims.Escapes sha512 Vr P x) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (τ : Type) [Fintype τ] [Nonempty τ] (cap₀ : Model.Cap Refine.Dig)
    (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀))
    (hacc : (Claims.Pr fun r : Fin (Vr.k x) → Coin => (Claims.run sha512 P Vr x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p) ≤
      (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
        (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ)
    (hmean : (∑ t, epsSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀
        l₁ rest ext hk0 hm ptLocal spec.mPts [] [] cap₀ (σ t)) / Fintype.card τ ≤ εtable false x + e.term ω) :
    Claims.accPr sha512 Vr P x ω ≤ εtable false x +
      e.term ω +
      Claims.Pr fun r : Fin (Vr.k x) → Coin => (Claims.run sha512 P Vr x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
  have hper : ∀ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
      (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t) ≤
      epsSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
          hk0 hm ptLocal spec.mPts [] [] cap₀ (σ t) := by
    intro t
    have hs := session_sound_of_table_accL_sc execArith execArith_correct Refine.H512 Refine.enc512 Refine.leaf512
      (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hsch hm ptLocal spec.mPts hlay [] [] cap₀
      (fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
      (fun _ h => h.1) (Claims.Flock.oracleOf C) (σ t)
    have e : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL))) ∧
        ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (Claims.Flock.oracleOf C)) = ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512
          Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL)) :=
      congrArg ind (funext fun _ => propext (and_iff_left hT))
    have e₀ : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL))) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (outs.getD 0 dfltOutL) ∧
        Off₀L Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL)) = fun _ => 0 := by
      funext outs
      unfold ind
      split_ifs with h0
      · exact absurd ⟨h0.2.2, h0.2.1⟩ h0.1.2
      · rfl
    rw [← e]
    refine le_trans hs (le_of_eq ?_)
    simp only [List.length_nil]
    rw [e₀, expect_const, add_zero]
  have hsum := Finset.sum_le_sum fun t (_ : t ∈ Finset.univ) => hper t
  have hcard : (0 : ℝ) ≤ Fintype.card τ := Nat.cast_nonneg _
  have hdiv := div_le_div_of_nonneg_right hsum hcard
  have hsplit : Claims.accPr sha512 Vr P x ω ≤
      (Claims.Pr fun r : Fin (Vr.k x) → Coin => (Claims.run sha512 P Vr x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p) +
      Claims.Pr fun r : Fin (Vr.k x) → Coin => (Claims.run sha512 P Vr x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    unfold Claims.accPr
    refine le_trans (Claims.Pr_mono fun r hr => ?_) (Claims.Pr_or_le _ _)
    by_cases hq : ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
        Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p
    · exact Or.inr ⟨hr, hq⟩
    · exact Or.inl ⟨hr, hq⟩
  have key : Claims.accPr sha512 Vr P x ω ≤ εtable false x +
      e.term ω +
      Claims.Pr fun r : Fin (Vr.k x) → Coin => (Claims.run sha512 P Vr x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P Vr x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    linarith [hsplit, hacc, hmean, hdiv]
  exact key

open Classical in
theorem plain_le_sessionsE {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts) {Ω : Type} (P : Claims.Prover Ω)
    (e : Claims.Escapes sha512 (V false) P x) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (τ : Type) [Fintype τ] [Nonempty τ] (cap₀ : Model.Cap Refine.Dig)
    (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀))
    (hacc : (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p) ≤
      (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
        (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ)
    (hmean : (∑ t, epsSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m k0 l₀
        l₁ rest ext hk0 hm ptLocal spec.mPts [] [] cap₀ (σ t)) / Fintype.card τ ≤ εtable false x + e.term ω) :
    Claims.accPr sha512 (V false) P x ω ≤ εtable false x +
      e.term ω +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p :=
  plain_le_sessionsEV (V false) L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay P e ω C hT τ cap₀ σ hacc hmean

open Classical in
theorem plain_le_sessions {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)])
    (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts) {Ω : Type} (P : Claims.Prover Ω)
    (ev : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Ω → (Fin ((V false).k x) → Coin) →
      (Fin ((V false).k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin ((V false).k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (V false) x ω s).vLog ++
        (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω s).vLog) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (τ : Type) [Fintype τ] [Nonempty τ] (cap₀ : Model.Cap Refine.Dig)
    (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀))
    (hacc : (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p) ≤
      (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
        (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ)
    (hadv : ∀ r ℓ, (∑ t, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
        k0 l₀ l₁ rest ext hk0 hm ptLocal spec.mPts [] [] cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) => ev r ℓ ω s.1 s.2)
    (hsc : (∑ t, expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL))
      (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : Fin ((V false).k x) → Coin => ev₁ ω s) :
    Claims.accPr sha512 (V false) P x ω ≤ εtable false x +
      (Claims.Flock.oldEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch) ev
        ev₁ tied tied₁).term ω +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p :=
  plain_le_sessionsE L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay P
    (Claims.Flock.oldEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch) ev ev₁
      tied tied₁) ω C hT τ cap₀ σ hacc
    (mean_epsSL_le execArith Refine.H512 Refine.enc512 Refine.leaf512 h L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal [] []
      cap₀ σ sha512 (V false) P (plainJ (tables_head_one h) hsch) ev ev₁ tied tied₁ ω hadv hsc)

open Classical in
theorem plain_tableBoundθAt_of_sessions (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → List 𝔽 → Prop)
    {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts)
    (hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, Ok x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain rel) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hmix : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      ∃ (ev : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Ω → (Fin ((V false).k x) → Coin) →
          (Fin ((V false).k x) → Coin) → Prop)
        (ev₁ : Ω → (Fin ((V false).k x) → Coin) → Prop)
        (_tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl sha512
          ((Claims.run sha512 P (V false) x ω s).vLog ++
            (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).vLog))
        (_tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω s).vLog),
        ∀ ω (C : Fin (wordLen false x) → List 𝔽), (∀ p, Ok x (C p)) →
          ∃ (τ : Type) (_ : Fintype τ) (_ : Nonempty τ) (cap₀ : Model.Cap Refine.Dig)
            (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512
              (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀)),
            (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
                ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
                  Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧
                    u ≠ C p) ≤
              (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
                  ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
                  ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
                    (outs.getD 0 dfltOutL)))
                (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
                  ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ ∧
            (∀ r ℓ, (∑ t, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
                k0 l₀ l₁ rest ext hk0 hm ptLocal spec.mPts [] [] cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) => ev r ℓ ω s.1 s.2) ∧
            (∑ t, expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
                SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
                  (outs.getD 0 dfltOutL))
              (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
                ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : Fin ((V false).k x) → Coin => ev₁ ω s) :
    Claims.Flock.TableBoundθAt sha512 (deployPlain rel) (wordLen false) Ok x := by
  intro Ω _ _ P
  obtain ⟨ev, ev₁, tied, tied₁, hω⟩ := hmix Ω P
  refine ⟨Claims.Flock.oldEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch)
    ev ev₁ tied tied₁, oldEsc_fitsFork_plainJ rel P x (tables_head_one h) hsch ev ev₁ tied tied₁,
    fun ω C hok hg => ?_⟩
  obtain ⟨τ, _, _, cap₀, σ, hacc, hadv, hsc⟩ := hω ω C hok
  exact plain_le_sessions h L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay P ev ev₁ tied tied₁ ω C (hnc C hok hg) τ
    cap₀ σ hacc hadv hsc

def plainTab {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts) : TabSpec spec.mPts :=
  ⟨L.model (Refine.regionsOf st), m, k0, l₀, l₁, rest, ext, hsch, hm, ptLocal, hlay⟩

open Classical Discharge.Tables in
theorem plain_tableBoundθAt_of_live (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → List 𝔽 → Prop)
    {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts)
    (hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, Ok x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain rel) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hlive : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      ∃ (ev : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Ω → (Fin ((V false).k x) → Coin) →
          (Fin ((V false).k x) → Coin) → Prop)
        (ev₁ : Ω → (Fin ((V false).k x) → Coin) → Prop)
        (_tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl sha512
          ((Claims.run sha512 P (V false) x ω s).vLog ++
            (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).vLog))
        (_tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω s).vLog),
        ∀ ω (C : Fin (wordLen false x) → List 𝔽), (∀ p, Ok x (C p)) →
          ∃ (pL : LiveOutJ [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length spec.mPts → Prop)
            (opn : Array ByteArray → Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length →
              Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
            (dg : ByteArray → ByteArray)
            (dec : Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length → ℕ → ByteArray → List Msg)
            (one : Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length → ℕ → Bool)
            (fin : Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length → Fin 2 → Array ByteArray →
              List Msg)
            (capOf : Array ByteArray → Fin [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length →
              Model.Cap Refine.Dig)
            (N : ℕ) (rootB₀ : Array ByteArray)
            (hdec : ∀ j, DecodesTable
              (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
              (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
                ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOf rootB₀ ⟨0, Nat.one_pos⟩) (Claims.Flock.oracleOf C) o ∧
                ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
              pL opn dec one fin capOf
              (fun j => repsOf (D := Refine.Dig) execArith
                ([plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].get j))
              j)
            (τ : Type) (_ : Fintype τ) (_ : Nonempty τ)
            (σL : τ → Strategy
              (liveSession dg [plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay].length spec.mPts N))
            (hσ : ∀ u, (σL u).1 = rootB₀),
            (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
                ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
                  Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧
                    u ≠ C p) ≤
              (∑ u, (prob pL _ (σL u)).toReal) / Fintype.card τ ∧
            (∀ r ℓ, (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
                k0 l₀ l₁ rest ext hk0 hm ptLocal spec.mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩) r.val (ℓ.val + 1)
                (simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
                  (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀
                  (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) => ev r ℓ ω s.1 s.2) ∧
            (∑ u, expect (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
                SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
                  (outs.getD 0 dfltOutL))
              (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
                ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩))
              (simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
                (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀
                (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : Fin ((V false).k x) → Coin => ev₁ ω s) :
    Claims.Flock.TableBoundθAt sha512 (deployPlain rel) (wordLen false) Ok x := by
  refine plain_tableBoundθAt_of_sessions rel Ok h L m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal hlay hnc
    fun Ω _ _ P => ?_
  obtain ⟨ev, ev₁, tied, tied₁, hω⟩ := hlive Ω P
  refine ⟨ev, ev₁, tied, tied₁, fun ω C hok => ?_⟩
  obtain ⟨pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hdec, τ, _, _, σL, hσ, htr, hadv, hsc⟩ := hω ω C hok
  exact ⟨τ, inferInstance, inferInstance, capOf rootB₀ ⟨0, Nat.one_pos⟩, fun u => simAt execArith Refine.H512
    Refine.enc512 Refine.leaf512 (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N
    hdec rootB₀ (σL u) (hσ u),
    le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm ptLocal hlay) _ pL opn dg dec one fin capOf N hdec rootB₀ σL hσ),
    hadv, hsc⟩

end FlockDeploy.KS
