module

public import Proofs.Flock.Claims.KS.PlainSimW
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
import Proofs.Flock.Claims.Recursion.PlainK

/-! Self-clashes past level 0 collide in the run's log, so `PlainClashRun` follows from a raw clash at any level. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge.Tables Claims.Bridge.Live

def PlainLevelClash (ms : MerkleScheme) (lv : Flock.Level) (lanes : ℕ) (L : List (Claims.Msg × Claims.Msg)) : Prop :=
  ∃ (root : ByteArray) (q : ℕ) (row row' : Array F128), row ≠ row' ∧ Opening ms lv lanes root L q row ∧
    Opening ms lv lanes root L q row'

theorem plainLevelClash_tied {ms : MerkleScheme} {lv : Flock.Level} {lanes : ℕ} {L : List (Claims.Msg × Claims.Msg)}
    (h : PlainLevelClash ms lv lanes L) : Claims.TiedColl sha512 L := by
  obtain ⟨_, _, _, _, hne, h₁, h₂⟩ := h
  exact tiedColl_of_append_self (tiedColl_of_openings h₁ h₂ hne)

def PlainRawClashRunL {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
    (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out = .acc ∧
      ((∃ (q : ℕ) (a b : List FlockLevel3.GF128),
        (q, a) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧
        (q, b) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧ a ≠ b) ∨
        (∃ (ms : MerkleScheme) (lv : Flock.Level) (lanes : ℕ),
          PlainLevelClash ms lv lanes (Claims.run sha512 P (V false) x ω (Φ (u, r'))).vLog) ∨
        Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω (Φ (u, r'))).vLog)

def PlainRawClashRunLK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
    (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).out = .acc ∧
      ((∃ (q : ℕ) (a b : List FlockLevel3.GF128),
        (q, a) ∈ plainRawCols (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).seen ∧
        (q, b) ∈ plainRawCols (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).seen ∧ a ≠ b) ∨
        (∃ (ms : MerkleScheme) (lv : Flock.Level) (lanes : ℕ),
          PlainLevelClash ms lv lanes (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).vLog) ∨
        Claims.TiedColl sha512 (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).vLog)

theorem plainClashRun_of_rawLK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (h : PlainRawClashRunLK keep P x ω mPts N dg w P' Φ pL) :
    PlainClashRunK keep P x ω mPts N dg w P' Φ pL := fun u r' hp => by
  obtain ⟨hacc, ⟨q, a, b, ha, hb, hab⟩ | ⟨_, _, _, hc⟩ | ht⟩ := h u r' hp
  · exact ⟨hacc, plainK_selfClash_tied keep hacc ha hb hab⟩
  · exact ⟨hacc, plainLevelClash_tied hc⟩
  · exact ⟨hacc, ht⟩

theorem plainClashRun_of_rawL {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (h : PlainRawClashRunL P x ω mPts N dg w P' Φ pL) :
    PlainClashRun P x ω mPts N dg w P' Φ pL :=
  plainClashRun_of_rawLK id (Ω := Ω) (P := P) (x := x) (ω := ω) (mPts := mPts) (N := N) (dg := dg) (w := w) (τ := τ) (P' := P') (Φ := Φ) (pL := pL) (h := h)

theorem plainRawClashRunL_of_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (h : PlainRawClashRun P x ω mPts N dg w P' Φ pL) :
    PlainRawClashRunL P x ω mPts N dg w P' Φ pL := fun u r' hp => by
  obtain ⟨hacc, hc | ht⟩ := h u r' hp
  · exact ⟨hacc, .inl hc⟩
  · exact ⟨hacc, .inr (.inr ht)⟩

end FlockDeploy.KS
