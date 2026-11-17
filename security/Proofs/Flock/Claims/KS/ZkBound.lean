module

public import Proofs.Flock.Claims.KS.MixtureZk
public import Proofs.Flock.Claims.KS.Refit
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
import Proofs.Flock.Claims.KS.PlainBound
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Soundness.Game

/-! The masked table bound from masked sessions after `Commit`. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)

local notation "𝔽" => FlockLevel3.GF128

def Off₀Z {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap₀ : Model.Cap D) (C : Model.Oracle 𝔽)
    (out : OutZC 𝔽 D Sl) : Prop :=
  ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C) out p

def AccOneZ {D Sl : Type} [Inhabited D] [Inhabited Sl] (outs : List (OutZC 𝔽 D Sl)) : Prop :=
  (outs.getD 0 dfltOutZC).accepted = true

open Classical in
theorem zk_tableBoundθAt_of_sessions {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D]
    [Inhabited Sl] (A : Model.Arith 𝔽 K) (hA : A.Correct) (Hs : Hashes 𝔽 D Sl)
    (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → List 𝔽 → Prop)
    {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables true x = [(spec, st)]) {mPts : ℕ}
    (T : TabZK 𝔽 mPts) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hsch : T.sch = Accounting.m1Schedule st.wideLeaf st.m)
    (ht : T.t = Accounting.m1Pad st.wideLeaf st.m) (he : T.e = 2) (hS : T.S.shape mPts = acct spec st) (hN : T.N ≤ 2 ^ 13)
    (hq : T.q = 168) (hδ : 7 / 16 ≤ T.δ)
    (hNp : T.N ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.wideLeaf st.m).1.k0))
    (hnc : ∀ C : Fin (wordLen true x) → List 𝔽, (∀ p, Ok x (C p)) →
      ¬ Claims.Flock.GoodTable (deployZk rel) (wordLen true) x C → ¬ T.Committed A (Claims.Flock.oracleOf C))
    (hmix : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      ∃ (j : Claims.Flock.IZ T.sch.levels.length → Fin ((V true).k x)) (_ : Function.Injective j)
        (ev : Claims.Flock.IZ T.sch.levels.length → Ω → (Fin ((V true).k x) → Coin) →
          (Fin ((V true).k x) → Coin) → Prop)
        (ev₁ : Ω → (Fin ((V true).k x) → Coin) → Prop)
        (_tied : ∀ i ω s s', ev i ω s s' → Claims.TiedColl sha512
          ((Claims.run sha512 P (V true) x ω s).vLog ++
            (Claims.run sha512 P (V true) x ω (Claims.forkCoins (j i) s s')).vLog))
        (_tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog),
        ∀ ω (C : Fin (wordLen true x) → List 𝔽), (∀ p, Ok x (C p)) →
          ∃ (τ : Type) (_ : Fintype τ) (_ : Nonempty τ) (cap₀ : Model.Cap D)
            (σ : τ → Strategy (sessAtZ [] [] (tabAfterZC A Hs T cap₀))),
            (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
                ¬ ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
                  Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧
                    u ≠ C p) ≤
              (∑ t, expect (ind fun outs => AccOneZ outs ∧
                  ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC))
                (sessAtZ [] [] (tabAfterZC A Hs T cap₀)) (σ t)) / Fintype.card τ ∧
            (∀ r ℓ, (∑ t, advRSZ A Hs T [] [] cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inl (r, ℓ)) ω s.1 s.2) ∧
            (∀ r : Fin 2, (∑ t, advPadsSZ A Hs T [] [] cap₀ r.val (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inr (.inl r)) ω s.1 s.2) ∧
            (∀ r : Fin 2, (∑ t, advTauSZ A Hs T [] [] cap₀ r.val (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inr (.inr r)) ω s.1 s.2) ∧
            (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) => SelfClashZ Hs T (outs.getD 0 dfltOutZC))
              (sessAtZ [] [] (tabAfterZC A Hs T cap₀)) (σ t)) / Fintype.card τ ≤
              Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ ω s) :
    Claims.Flock.TableBoundθAt sha512 (deployZk rel) (wordLen true) Ok x := by
  intro Ω _ _ P
  obtain ⟨j, hj, ev, ev₁, tied, tied₁, hω⟩ := hmix Ω P
  refine ⟨Claims.Flock.oldEscZ sha512 (V true) P x T.sch T.N j ev ev₁ tied tied₁,
    oldEscZ_fitsFork_zk rel P x (tables_head_one h) h25 h35 hsch T.N hNp j hj ev ev₁ tied tied₁,
    fun ω C hok hg => ?_⟩
  obtain ⟨τ, _, _, cap₀, σ, hacc, hadv, hP, hT, hsc⟩ := hω ω C hok
  have hTc := hnc C hok hg
  have hper : ∀ t, expect (ind fun outs => AccOneZ outs ∧
        ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC))
      (sessAtZ [] [] (tabAfterZC A Hs T cap₀)) (σ t) ≤ epsSZ A Hs T [] [] cap₀ (σ t) := by
    intro t
    have hs := session_sound_of_table_accZ A hA Hs T [] [] cap₀
      (fun outs => AccOneZ outs ∧ ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC))
      (fun _ h i hi => by obtain rfl := Nat.le_zero.1 hi; exact h.1) (Claims.Flock.oracleOf C) (σ t)
    have e : (ind fun outs : List (OutZC 𝔽 D Sl) =>
        (AccOneZ outs ∧ ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC)) ∧
          ¬ T.Committed A (Claims.Flock.oracleOf C)) = ind fun outs => AccOneZ outs ∧
          ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC) :=
      congrArg ind (funext fun _ => propext (and_iff_left hTc))
    have e₀ : (ind fun outs : List (OutZC 𝔽 D Sl) =>
        (AccOneZ outs ∧ ¬ Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutZC)) ∧
          ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ (Claims.Flock.oracleOf C))
            (outs.getD 0 dfltOutZC) p) = fun _ => 0 := by
      funext outs
      unfold ind
      split_ifs with h0
      · exact absurd h0.2 h0.1.2
      · rfl
    simp only [List.length_nil] at hs
    rw [e, e₀, expect_const, add_zero] at hs
    exact hs
  have hmean := mean_epsSZ_le FlockLevel3.card_F A Hs T [] [] cap₀ σ h h25 h35 hsch ht he hS hN hq hδ sha512
    (V true) P j ev ev₁ tied tied₁ ω hadv hP hT hsc
  have hsum := Finset.sum_le_sum fun t (_ : t ∈ Finset.univ) => hper t
  have hcard : (0 : ℝ) ≤ Fintype.card τ := Nat.cast_nonneg _
  have hdiv := div_le_div_of_nonneg_right hsum hcard
  have hsplit : Claims.accPr sha512 (V true) P x ω ≤
      (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧ u ≠ C p) +
      Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    unfold Claims.accPr
    refine le_trans (Claims.Pr_mono fun r hr => ?_) (Claims.Pr_or_le _ _)
    by_cases hq : ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
        Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧ u ≠ C p
    · exact Or.inr ⟨hr, hq⟩
    · exact Or.inl ⟨hr, hq⟩
  have key : Claims.accPr sha512 (V true) P x ω ≤ εtable true x +
      (Claims.Flock.oldEscZ sha512 (V true) P x T.sch T.N j ev ev₁ tied tied₁).term ω +
      Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    linarith [hsplit, hacc, hmean, hdiv]
  exact key

open Classical in
def ZkLiveAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (Ok : Claims.Msg → List 𝔽 → Prop) (x : Claims.Msg) {mPts : ℕ}
    (T : TabZK 𝔽 mPts) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      ∃ (j : Claims.Flock.IZ T.sch.levels.length → Fin ((V true).k x)) (_ : Function.Injective j)
        (ev : Claims.Flock.IZ T.sch.levels.length → Ω → (Fin ((V true).k x) → Coin) →
          (Fin ((V true).k x) → Coin) → Prop)
        (ev₁ : Ω → (Fin ((V true).k x) → Coin) → Prop)
        (_tied : ∀ i ω s s', ev i ω s s' → Claims.TiedColl sha512
          ((Claims.run sha512 P (V true) x ω s).vLog ++
            (Claims.run sha512 P (V true) x ω (Claims.forkCoins (j i) s s')).vLog))
        (_tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog),
        ∀ ω (C : Fin (wordLen true x) → List 𝔽), (∀ p, Ok x (C p)) →
          ∃ (pL : LiveOutJ [T].length mPts → Prop) (opn : Array ByteArray → Fin [T].length → OpensZ 𝔽 D Sl)
            (dg : ByteArray → ByteArray) (dec : Fin [T].length → ℕ → ByteArray → List Msg)
            (one : Fin [T].length → ℕ → Bool) (fin : Fin [T].length → Fin 2 → Array ByteArray → List Msg)
            (capOf : Array ByteArray → Fin [T].length → Model.Cap D) (N : ℕ) (rootB₀ : Array ByteArray)
            (hdec : ∀ i, DecodesTable (outJ Hs [T]) (fun _ o => o.accepted = true ∧
                ¬ Off₀Z Hs T (capOf rootB₀ ⟨0, Nat.one_pos⟩) (Claims.Flock.oracleOf C) o)
              pL opn dec one fin capOf (repJ A Hs [T]) i)
            (τ : Type) (_ : Fintype τ) (_ : Nonempty τ) (σL : τ → Strategy (liveSession dg [T].length mPts N))
            (hσ : ∀ u, (σL u).1 = rootB₀),
            (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
                ¬ ∃ (p : Fin (wordLen true x)) (u : List 𝔽),
                  Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) p.val = some u ∧
                    u ≠ C p) ≤
              (∑ u, (prob pL _ (σL u)).toReal) / Fintype.card τ ∧
            (∀ r ℓ, (∑ u, advRSZ A Hs T [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩) r.val (ℓ.val + 1)
                (simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inl (r, ℓ)) ω s.1 s.2) ∧
            (∀ r : Fin 2, (∑ u, advPadsSZ A Hs T [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩) r.val
                (simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inr (.inl r)) ω s.1 s.2) ∧
            (∀ r : Fin 2, (∑ u, advTauSZ A Hs T [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩) r.val
                (simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                ev (.inr (.inr r)) ω s.1 s.2) ∧
            (∑ u, expect (ind fun outs : List (OutZC 𝔽 D Sl) => SelfClashZ Hs T (outs.getD 0 dfltOutZC))
              (sessAtZ [] [] (tabAfterZC A Hs T (capOf rootB₀ ⟨0, Nat.one_pos⟩)))
              (simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ ≤
              Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ ω s

theorem zk_tableBoundθAt_of_live {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D]
    [Inhabited Sl] (A : Model.Arith 𝔽 K) (hA : A.Correct) (Hs : Hashes 𝔽 D Sl)
    (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → List 𝔽 → Prop)
    {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables true x = [(spec, st)]) {mPts : ℕ}
    (T : TabZK 𝔽 mPts) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hsch : T.sch = Accounting.m1Schedule st.wideLeaf st.m)
    (ht : T.t = Accounting.m1Pad st.wideLeaf st.m) (he : T.e = 2) (hS : T.S.shape mPts = acct spec st) (hN : T.N ≤ 2 ^ 13)
    (hq : T.q = 168) (hδ : 7 / 16 ≤ T.δ)
    (hNp : T.N ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.wideLeaf st.m).1.k0))
    (hnc : ∀ C : Fin (wordLen true x) → List 𝔽, (∀ p, Ok x (C p)) →
      ¬ Claims.Flock.GoodTable (deployZk rel) (wordLen true) x C → ¬ T.Committed A (Claims.Flock.oracleOf C))
    (hlive : ZkLiveAt A Hs Ok x T) :
    Claims.Flock.TableBoundθAt sha512 (deployZk rel) (wordLen true) Ok x := by
  refine zk_tableBoundθAt_of_sessions A hA Hs rel Ok h T h25 h35 hsch ht he hS hN hq hδ hNp hnc fun Ω _ _ P => ?_
  obtain ⟨j, hj, ev, ev₁, tied, tied₁, hω⟩ := hlive Ω P
  refine ⟨j, hj, ev, ev₁, tied, tied₁, fun ω C hok => ?_⟩
  obtain ⟨pL, opn, dg, dec, one, fin, capOf, N, rootB₀, hdec, τ, _, _, σL, hσ, htr, hadv, hP, hT, hsc⟩ := hω ω C hok
  exact ⟨τ, inferInstance, inferInstance, capOf rootB₀ ⟨0, Nat.one_pos⟩,
    fun u => simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u),
    le_trans htr (mean_live_le_simAtZ A Hs T _ pL opn dg dec one fin capOf N hdec rootB₀ σL hσ),
    hadv, hP, hT, hsc⟩

end FlockDeploy.KS
