module

public import Batteries.Data.ByteArray
public import Proofs.Flock.Claims.Bridge.Live
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
import Proofs.Flock.Claims.Bridge.Hm96Tree
import Proofs.Flock.Soundness.Game
import VCVio.Prelude.Core

/-! A mixture of live strategies bounds the runs of `V false`, which reveals each round's whole coin where the live session reveals part of it. -/

@[expose] public section

namespace Claims.Bridge.Live

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Tables

def mergeC {K M : ℕ} {X : Type} (n : Fin K → ℕ) (u r : Fin K → Fin M → X) : Fin K → Fin M → X :=
  fun j i => if i.val < n j then r j i else u j i

def swapC {K M : ℕ} (X : Type) (n : Fin K → ℕ) :
    (Fin K → Fin M → X) × (Fin K → Fin M → X) ≃ (Fin K → Fin M → X) × (Fin K → Fin M → X) where
  toFun p := (mergeC n p.1 p.2, mergeC n p.2 p.1)
  invFun p := (mergeC n p.1 p.2, mergeC n p.2 p.1)
  left_inv p := by
    refine Prod.ext (funext fun j => funext fun i => ?_) (funext fun j => funext fun i => ?_) <;>
      simp only [mergeC] <;> split_ifs <;> rfl
  right_inv p := by
    refine Prod.ext (funext fun j => funext fun i => ?_) (funext fun j => funext fun i => ?_) <;>
      simp only [mergeC] <;> split_ifs <;> rfl

open Classical in
theorem Pr_mergeC {K M : ℕ} {X : Type} [Fintype X] [Nonempty X] (n : Fin K → ℕ)
    (E : (Fin K → Fin M → X) → Prop) :
    Pr (fun p : (Fin K → Fin M → X) × (Fin K → Fin M → X) => E (mergeC n p.1 p.2)) = Pr E := by
  refine (Pr_equiv (swapC X n) (fun q => E q.1)).trans ?_
  rw [Pr_eq_Ex, Ex_prod, Pr_eq_Ex]
  exact congrArg Ex (funext fun a => Ex_const' (ind E a))

def mergeAt {K L M : ℕ} {X : Type} (h : L = K) (n : Fin K → ℕ) (u : Fin K → Fin M → X) (r : Fin L → Fin M → X) :
    Fin K → Fin M → X :=
  mergeC n u fun j => r (Fin.cast h.symm j)

open Classical in
theorem Pr_mergeAt {K L M : ℕ} {X : Type} [Fintype X] [Nonempty X] (h : L = K) (n : Fin K → ℕ)
    (E : (Fin K → Fin M → X) → Prop) :
    Pr (fun p : (Fin K → Fin M → X) × (Fin L → Fin M → X) => E (mergeAt h n p.1 p.2)) = Pr E := by
  subst h
  exact Pr_mergeC n E

theorem Pr_evalAt {K : ℕ} {D Y : Type} [Fintype D] [DecidableEq D] [Fintype Y] [Nonempty Y] (d : Fin K → D)
    (G : (Fin K → Y) → Prop) : Pr (fun f : Fin K → D → Y => G fun j => f j (d j)) = Pr G := by
  refine (Pr_equiv ((Equiv.piCongrRight fun j => Equiv.funSplitAt (d j) Y).trans
    (Equiv.arrowProdEquivProdArrow (Fin K) (fun _ => Y) fun j => {e // e ≠ d j} → Y)) fun q => G q.1).trans ?_
  rw [Pr_eq_Ex, Ex_prod, Pr_eq_Ex]
  exact congrArg Ex (funext fun a => Ex_const' (ind G a))

def fnC {K M : ℕ} {X D : Type} (n : Fin K → ℕ) (key : Fin K → (Fin K → Fin M → X) → D)
    (f : Fin K → D → Fin M → X) (r : Fin K → Fin M → X) : Fin K → Fin M → X :=
  mergeC n (fun j => f j (key j r)) r

theorem Pr_fnC {K M : ℕ} {X D : Type} [Fintype X] [Nonempty X] [Fintype D] [DecidableEq D] (n : Fin K → ℕ)
    (key : Fin K → (Fin K → Fin M → X) → D) (E : (Fin K → Fin M → X) → Prop) :
    Pr (fun p : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) => E (fnC n key p.1 p.2)) = Pr E := by
  calc Pr (fun p : (Fin K → D → Fin M → X) × (Fin K → Fin M → X) => E (fnC n key p.1 p.2))
      = Pr (fun q : (Fin K → Fin M → X) × (Fin K → D → Fin M → X) => E (fnC n key q.2 q.1)) :=
        Pr_equiv (Equiv.prodComm _ _) fun q => E (fnC n key q.2 q.1)
    _ = Ex fun r => Pr fun f : Fin K → D → Fin M → X => E (fnC n key f r) :=
        Pr_prod fun r f => E (fnC n key f r)
    _ = Ex fun r => Pr fun u : Fin K → Fin M → X => E (mergeC n u r) :=
        congrArg Ex (funext fun r => Pr_evalAt (fun j => key j r) fun u => E (mergeC n u r))
    _ = Pr (fun q : (Fin K → Fin M → X) × (Fin K → Fin M → X) => E (mergeC n q.2 q.1)) :=
        (Pr_prod fun r u => E (mergeC n u r)).symm
    _ = Pr (fun p : (Fin K → Fin M → X) × (Fin K → Fin M → X) => E (mergeC n p.1 p.2)) :=
        Pr_equiv (Equiv.prodComm _ _) fun p => E (mergeC n p.1 p.2)
    _ = Pr E := Pr_mergeC n E

def fnAt {K L M : ℕ} {X D : Type} (h : L = K) (n : Fin K → ℕ) (key : Fin K → (Fin K → Fin M → X) → D)
    (f : Fin K → D → Fin M → X) (r : Fin L → Fin M → X) : Fin K → Fin M → X :=
  fnC n key f fun j => r (Fin.cast h.symm j)

theorem Pr_fnAt {K L M : ℕ} {X D : Type} [Fintype X] [Nonempty X] [Fintype D] [DecidableEq D] (h : L = K)
    (n : Fin K → ℕ) (key : Fin K → (Fin K → Fin M → X) → D) (E : (Fin K → Fin M → X) → Prop) :
    Pr (fun p : (Fin K → D → Fin M → X) × (Fin L → Fin M → X) => E (fnAt h n key p.1 p.2)) = Pr E := by
  subst h
  exact Pr_fnC n key E

open Classical in
theorem liveMix_eq {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω τ : Type} [Fintype τ] [Nonempty τ]
    (P' : Prover (Ω × τ)) (x : Msg) (ω : Ω) (pL : LiveOutJ (J x) (mPts x) → Prop) :
    (∑ u, (prob pL (liveSession dg (J x) (mPts x) (N x))
      (liveStrat J mPts N dg w H P' x (ω, u))).toReal) / Fintype.card τ =
      Pr (fun p : τ × (Fin (N x + 2) → LCoin nMax) => pL (liveOut J mPts N dg w H P' x (ω, p.1) p.2)) := by
  symm
  calc Pr (fun p : τ × (Fin (N x + 2) → LCoin nMax) => pL (liveOut J mPts N dg w H P' x (ω, p.1) p.2))
      = Ex fun u => Pr fun r' : Fin (N x + 2) → LCoin nMax => pL (liveOut J mPts N dg w H P' x (ω, u) r') :=
        Pr_prod (fun u r' => pL (liveOut J mPts N dg w H P' x (ω, u) r'))
    _ = Ex fun u => (prob pL (liveSession dg (J x) (mPts x) (N x))
          (liveStrat J mPts N dg w H P' x (ω, u))).toReal := by
        refine congrArg Ex (funext fun u => ?_)
        rw [liveStrat, prob_eqRec, prob_toReal, Pr_eq_Ex]
        exact ex_outcome H P' (fun x => N x + 2) (fun x => sessionK dg (J x) (mPts x) (N x))
          (fun x => sessionIf (w x) dg (N x)) x (ω, u) (sessionLe (w x) dg (N x) _) (ind pL)
    _ = _ := rfl

open Classical in
theorem Pr_le_liveMix {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω τ : Type} [Fintype τ] [Nonempty τ]
    (P' : Prover (Ω × τ)) (x : Msg) (ω : Ω) (pL : LiveOutJ (J x) (mPts x) → Prop) {R : Type} [Fintype R]
    (E : R → Prop) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → R) (hΦ : Pr (fun p => E (Φ p)) = Pr E)
    (hsim : ∀ u r', E (Φ (u, r')) → pL (liveOut J mPts N dg w H P' x (ω, u) r')) :
    Pr E ≤ (∑ u, (prob pL (liveSession dg (J x) (mPts x) (N x))
      (liveStrat J mPts N dg w H P' x (ω, u))).toReal) / Fintype.card τ := by
  rw [← hΦ, liveMix_eq J mPts N dg w H P' x ω pL]
  exact Pr_mono fun p hp => hsim p.1 p.2 hp

open Classical in
theorem liveMix_le_Pr {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω τ : Type} [Fintype τ] [Nonempty τ]
    (P' : Prover (Ω × τ)) (x : Msg) (ω : Ω) (pL : LiveOutJ (J x) (mPts x) → Prop) {R : Type} [Fintype R]
    (E : R → Prop) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → R) (hΦ : Pr (fun p => E (Φ p)) = Pr E)
    (hsim : ∀ u r', pL (liveOut J mPts N dg w H P' x (ω, u) r') → E (Φ (u, r'))) :
    (∑ u, (prob pL (liveSession dg (J x) (mPts x) (N x))
      (liveStrat J mPts N dg w H P' x (ω, u))).toReal) / Fintype.card τ ≤ Pr E := by
  rw [← hΦ, liveMix_eq J mPts N dg w H P' x ω pL]
  exact Pr_mono fun p hp => hsim p.1 p.2 hp

theorem fst_eqRec {α M : Type} {k k' : M → Game α} (h : Game.send M k' = Game.send M k)
    (s : Strategy (Game.send M k')) : (h ▸ s : Strategy (Game.send M k)).1 = s.1 := by
  obtain ⟨-, hk⟩ := Game.send.inj h
  cases eq_of_heq hk
  rfl

theorem liveStrat_fst {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    (liveStrat J mPts N dg w H P x ω).1 = (w x).root ((P.next x [] ω).eval H).1 :=
  fst_eqRec (liveSession_eq dg (J x) (mPts x) (N x)).symm _

end Claims.Bridge.Live
