module

public import Proofs.Flock.Claims.Recursion.OpenLowerJointV
public import Proofs.Flock.Claims.Recursion.PartAcceptsSetup
public import Proofs.Flock.Claims.Recursion.VerdictRowECoin
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

/-! Row E's chain at the joint witnesses (`RowEJoint`), which nothing fixes to an honest prover's, carried by the `rec-open` pieces' holds since the deployed last message carries no chain block. -/

@[expose] public section

namespace RecDeploy.Top

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB
open RecDeploy.Rows (accHolds sideAt coefRead rowDPieces)

variable (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)

def ChainHolds (x : Msg) (ms : List Msg) (cs : List Coin) (ch : ℕ → ℕ → Array F128) : Prop :=
  ∀ q ∈ rowDPieces key x, accHolds (sideAt rd) coefRead (fun _ _ => ch) q x ms cs

open Classical in
noncomputable def chainC (x : Msg) (ms : List Msg) (cs : List Coin) : ℕ → ℕ → Array F128 :=
  if h : ∃ ch, ChainHolds rd key x ms cs ch then Classical.choose h else fun _ _ => #[]

def accPiece (q : OpenPiece) (x : Msg) (ms : List Msg) (cs : List Coin) : Prop :=
  accHolds (sideAt rd) coefRead (fun _ _ => chainC rd key x ms cs) q x ms cs

noncomputable abbrev accRC (x : Msg) (ms : List Msg) (cs : List Coin) : ℕ → Array F128 :=
  accR rd (fun x' ms' => chainC rd key x' ms' cs) x ms cs

theorem accPiece_of_chain {x : Msg} {ms : List Msg} {cs : List Coin} {ch : ℕ → ℕ → Array F128}
    (h : ChainHolds rd key x ms cs ch) : ∀ q ∈ rowDPieces key x, accPiece rd key q x ms cs := by
  intro q hq
  have he : ∃ ch, ChainHolds rd key x ms cs ch := ⟨ch, h⟩
  have hs := Classical.choose_spec he q hq
  have hc : chainC rd key x ms cs = Classical.choose he := by rw [chainC, dif_pos he]
  unfold accPiece
  rw [hc]
  exact hs

theorem targetE_accRC
    (hE : ∀ (P' : Msg → List Msg → List Coin → Prop) (chainAt : Msg → List Msg → ℕ → ℕ → Array F128),
      (∀ x ms cs, P' x ms cs → ∀ q ∈ rowDPieces key x, accHolds (sideAt rd) coefRead chainAt q x ms cs) →
        TargetE rd P' (accR rd chainAt))
    (P : Msg → List Msg → List Coin → Prop)
    (hP : ∀ x ms cs, P x ms cs → ∀ q ∈ rowDPieces key x, accPiece rd key q x ms cs) :
    TargetE rd P (accRC rd key) := by
  intro x ms cs hA D cls i hi
  obtain ⟨h1, h2, h3, h4⟩ := hA
  exact hE (fun x' ms' cs' => P x' ms' cs' ∧ cs' = cs) (fun x' ms' => chainC rd key x' ms' cs)
    (fun x' ms' cs' h q hq => by
      obtain ⟨h, rfl⟩ := h
      exact hP x' ms' cs' h q hq)
    x ms cs ⟨h1, h2, h3, h4, rfl⟩ D cls i hi

section Joint

variable (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace) (L : LayoutB)
  (shape : Claims.Recursion.Form → Prop)
  (chk : ℕ → Msg → (ℕ → Coin) → Msg → Claims.Flock.Wit → List Msg → Prop)

def RowEJoint : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (xs : List Msg) (ws : List Claims.Flock.Wit), OpenLower.JointV H D L chk x c xs ws →
      ∀ os last, Bd.final ((L.msgs x ws).getD (Bd.DE.V.k (Claims.Recursion.xI D x) - 1) []) = some (os, last) →
        os.length + 1 = Bd.DE.V.k (Claims.Recursion.xI D x) →
        ∃ ch, ChainHolds rd key (Claims.Recursion.xI D x) (Bd.opened os last)
          ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) ch

theorem piecesComputes_accPiece (h : RowEJoint rd key H D Bd L shape chk) (ownQ : Msg → OpenPiece → Option ℕ) :
    ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (xs : List Msg) (ws : List Claims.Flock.Wit), OpenLower.JointV H D L chk x c xs ws →
        ∀ s, PiecesComputes Bd D (rowDPieces key) (accPiece rd key) ownQ s x c (L.msgs x ws) := by
  intro x st hx hf c xs ws hJ s q hq _ os last hfin hlen
  obtain ⟨ch, hch⟩ := h x st hx hf c xs ws hJ os last hfin hlen
  exact accPiece_of_chain rd key hch q hq

end Joint

end RecDeploy.Top
