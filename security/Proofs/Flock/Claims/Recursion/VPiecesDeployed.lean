module

public import Proofs.Flock.Claims.Recursion.NonVacuous
public import Proofs.Flock.Claims.Recursion.CapPieces
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

/-! The V\* parts' pieces of the verdict and their owners, for `recKSDecompR_of_targets`. -/

@[expose] public section

namespace RecDeploy.VPieces

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP VerdictR)
open Claims.Recursion.VStarB

inductive VPiece where
  | res (q : ResPiece)
  | opn (q : OpenPiece)
  | cap (q : CapPiece)

def vrows (resP : Msg → List ResPiece) (openP : Msg → List OpenPiece) (capP : Msg → List CapPiece) (x : Msg) :
    List VPiece :=
  (resP x).map .res ++ (openP x).map .opn ++ (capP x).map .cap

def vholds (resH : ResPiece → Msg → List Msg → List Coin → Prop) (openH : OpenPiece → Msg → List Msg → List Coin → Prop)
    (capH : CapPiece → Msg → List Msg → List Coin → Prop) : VPiece → Msg → List Msg → List Coin → Prop
  | .res q, x, ms, cs => resH q x ms cs
  | .opn q, x, ms, cs => openH q x ms cs
  | .cap q, x, ms, cs => capH q x ms cs

abbrev vholdsD (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (sideAt : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (rowAt : Msg → List Msg → CapPiece → ByteArray) : VPiece → Msg → List Msg → List Coin → Prop :=
  vholds (resHolds elemAt vvAt) (openHolds sideAt posAt) (capHolds rowAt)

def noB : Msg → VerdictR → Option ℕ := fun _ _ => none

def keep (x : Msg) (s : ℕ) : Option ℕ := if s < nS x then some s else none

def vown (ownR : Msg → ResPiece → Option ℕ) (ownO : Msg → OpenPiece → Option ℕ) (ownC : Msg → CapPiece → Option ℕ)
    (x : Msg) : Top.Piece VPiece → Option ℕ
  | .dense => keep x 0
  | .row (.res q) => (ownR x q).bind (keep x)
  | .row (.opn q) => (ownO x q).bind (keep x)
  | .row (.cap q) => (ownC x q).bind (keep x)

theorem keep_lt {x : Msg} {s t : ℕ} (h : keep x s = some t) : t < nS x := by
  unfold keep at h
  split at h
  · cases h
    assumption
  · cases h

theorem keep_eq {x : Msg} {s t : ℕ} (h : keep x s = some t) : s = t := by
  unfold keep at h
  split at h
  · cases h
    rfl
  · cases h

theorem keep_of_lt {x : Msg} {s : ℕ} (h : s < nS x) : keep x s = some s := by
  unfold keep
  rw [if_pos h]

variable (ownR : Msg → ResPiece → Option ℕ) (ownO : Msg → OpenPiece → Option ℕ) (ownC : Msg → CapPiece → Option ℕ)

theorem vown_lt {x : Msg} {p : Top.Piece VPiece} {s : ℕ} (h : vown ownR ownO ownC x p = some s) : s < nS x := by
  rcases p with _ | (q | q | q)
  · exact keep_lt h
  all_goals
    obtain ⟨t, -, ht⟩ := Option.bind_eq_some_iff.1 h
    exact keep_lt ht

theorem ownP_lt {x : Msg} {r : VerdictP (Top.Piece VPiece)} {s : ℕ} (h : ownP noB (vown ownR ownO ownC) x r = some s) :
    s < nS x := by
  cases r with
  | base q => simp [ownP, noB] at h
  | piece p => exact vown_lt ownR ownO ownC h

theorem hR_vown (rd : Msg → Option FinalP) (rows : Msg → List VPiece) (holds : VPiece → Msg → List Msg → List Coin → Prop)
    (acc : Msg → List Msg → List Coin → ℕ → Array _root_.Flock.F128) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (h : Top.Targets rd rows holds acc) :
    ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recR rd b aux nOf hd rest fin wE final w).parse x = some st →
      Claims.Recursion.Form.Dense st.form →
      ∀ r ∈ (splitR rd rows holds acc h).checks (Claims.Recursion.xI (recR rd b aux nOf hd rest fin wE final w) x),
        ∀ s, ownP noB (vown ownR ownO ownC) x r = some s → s < (recR rd b aux nOf hd rest fin wE final w).nS x :=
  fun _ _ _ _ _ _ _ hs => ownP_lt ownR ownO ownC hs

theorem nS_pos {x : Msg} {r : XR} (h : readX x = some r) : 0 < nS x := by
  obtain ⟨_, -, h0⟩ := capAt_readX h
  exact h0

theorem vown_some (resP : Msg → List ResPiece) (openP : Msg → List OpenPiece) (capP : Msg → List CapPiece) {x y : Msg}
    (h0 : 0 < nS x) (hr : ∀ q ∈ resP y, ∃ s < nS x, ownR x q = some s)
    (ho : ∀ q ∈ openP y, ∃ s < nS x, ownO x q = some s) (hc : ∀ q ∈ capP y, ∃ s < nS x, ownC x q = some s) :
    ∀ p ∈ Top.pieces (vrows resP openP capP) y, ∃ s, vown ownR ownO ownC x p = some s := by
  intro p hp
  rcases List.mem_cons.1 hp with rfl | hp
  · exact ⟨0, keep_of_lt h0⟩
  obtain ⟨p', hp', rfl⟩ := List.mem_map.1 hp
  simp only [vrows, List.mem_append, List.mem_map] at hp'
  rcases hp' with (⟨q, hq, rfl⟩ | ⟨q, hq, rfl⟩) | ⟨q, hq, rfl⟩
  · obtain ⟨s, hs, e⟩ := hr q hq
    exact ⟨s, by simp only [vown, e, Option.bind_some, keep_of_lt hs]⟩
  · obtain ⟨s, hs, e⟩ := ho q hq
    exact ⟨s, by simp only [vown, e, Option.bind_some, keep_of_lt hs]⟩
  · obtain ⟨s, hs, e⟩ := hc q hq
    exact ⟨s, by simp only [vown, e, Option.bind_some, keep_of_lt hs]⟩

theorem piecesComputes_vown (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (D : Claims.Recursion.Deploy Coin) (resP : Msg → List ResPiece) (openP : Msg → List OpenPiece)
    (capP : Msg → List CapPiece) (resH : ResPiece → Msg → List Msg → List Coin → Prop)
    (openH : OpenPiece → Msg → List Msg → List Coin → Prop) (capH : CapPiece → Msg → List Msg → List Coin → Prop)
    {s : ℕ} {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (hdense : Top.DenseInner (Claims.Recursion.xI D x))
    (hres : PiecesComputes Bd D resP resH ownR s x c ms) (hopen : PiecesComputes Bd D openP openH ownO s x c ms)
    (hcap : PiecesComputes Bd D capP capH ownC s x c ms) :
    PiecesComputes Bd D (Top.pieces (vrows resP openP capP)) (Top.pholds (vholds resH openH capH))
      (vown ownR ownO ownC) s x c ms := by
  intro p hp hown os last hf hk
  rcases List.mem_cons.1 hp with rfl | hp
  · exact hdense
  obtain ⟨p', hp', rfl⟩ := List.mem_map.1 hp
  simp only [vrows, List.mem_append, List.mem_map] at hp'
  rcases hp' with (⟨q, hq, rfl⟩ | ⟨q, hq, rfl⟩) | ⟨q, hq, rfl⟩
  · obtain ⟨t, ht, hk'⟩ := Option.bind_eq_some_iff.1 hown
    rw [keep_eq hk'] at ht
    exact hres q hq ht os last hf hk
  · obtain ⟨t, ht, hk'⟩ := Option.bind_eq_some_iff.1 hown
    rw [keep_eq hk'] at ht
    exact hopen q hq ht os last hf hk
  · obtain ⟨t, ht, hk'⟩ := Option.bind_eq_some_iff.1 hown
    rw [keep_eq hk'] at ht
    exact hcap q hq ht os last hf hk

end RecDeploy.VPieces
