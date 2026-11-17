module

public import Proofs.Flock.Claims.Bridge.Live
public import Proofs.Flock.Claims.DeployVK
public import Proofs.Flock.Claims.KS.ConflictTied
public import Proofs.Flock.Claims.KS.PlainBound
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
import Proofs.Flock.Claims.KS.LiveMix

/-! The deployed events a simulator's bounds read: `P`'s roots, an accepting run with a collision, and an accepting fork with a collision. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

def plainRoot {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) : Array ByteArray :=
  ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray

theorem plainRoot_of_first {Ω τ : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) {w : Claims.Msg → Wire 65536 1 mPts} {P' : Claims.Prover (Ω × τ)}
    (hw : ∀ a, (w x).root a = ((linkOf a).getD default).1.toArray)
    (h1 : ∀ u, ((P'.next x [] (ω, u)).eval sha512).1 = ((P.next x [] ω).eval sha512).1) (u : τ) :
    (liveStrat (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u)).1 = plainRoot P x ω := by
  rw [liveStrat_fst, hw, h1]
  rfl

def plainEv₁ {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (s : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (V false) x ω s).out = .acc ∧ Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω s).vLog

def plainEv₁K (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (s : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (VK keep) x ω s).out = .acc ∧ Claims.TiedColl sha512 (Claims.run sha512 P (VK keep) x ω s).vLog

def plainForkEv {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) {m k0 : ℕ}
    {l₀ l₁ : Accounting.Level} {rest : List Accounting.Level} {ext : ℕ}
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) {Ω : Type} (P : Claims.Prover Ω)
    (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) (ω : Ω) (s s' : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (V false) x ω s).out = .acc ∧
    (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).out = .acc ∧
    Claims.TiedColl sha512 ((Claims.run sha512 P (V false) x ω s).vLog ++
      (Claims.run sha512 P (V false) x ω (Claims.forkCoins (plainJ (tables_head_one h) hsch r ℓ) s s')).vLog)

def plainForkEvK (keep : Bool → Bool) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) {m k0 : ℕ}
    {l₀ l₁ : Accounting.Level} {rest : List Accounting.Level} {ext : ℕ}
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) {Ω : Type} (P : Claims.Prover Ω)
    (r : Fin 2) (ℓ : Fin (l₀ :: l₁ :: rest).length) (ω : Ω) (s s' : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (VK keep) x ω s).out = .acc ∧
    (Claims.run sha512 P (VK keep) x ω (Claims.forkCoins (plainJ (x := x) (tables_head_one h) hsch r ℓ) s s')).out = .acc ∧
    Claims.TiedColl sha512 ((Claims.run sha512 P (VK keep) x ω s).vLog ++
      (Claims.run sha512 P (VK keep) x ω (Claims.forkCoins (plainJ (x := x) (tables_head_one h) hsch r ℓ) s s')).vLog)

def PlainClashRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') → plainEv₁ P x ω (Φ (u, r'))

def PlainClashRunK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') → plainEv₁K keep P x ω (Φ (u, r'))

/-- `SelfClashL` also counts clashes at later levels, which this does not cover. -/
def PlainRawClashRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 mPts) {τ : Type} (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin) (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
    (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out = .acc ∧
      ((∃ (q : ℕ) (a b : List 𝔽),
        (q, a) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧
        (q, b) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧ a ≠ b) ∨
        Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω (Φ (u, r'))).vLog)

theorem plainClashRun_of_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (h : PlainRawClashRun P x ω mPts N dg w P' Φ pL) :
    PlainClashRun P x ω mPts N dg w P' Φ pL := fun u r' hp => by
  obtain ⟨hacc, ⟨q, a, b, ha, hb, hab⟩ | ht⟩ := h u r' hp
  · exact ⟨hacc, plain_selfClash_tied hacc ha hb hab⟩
  · exact ⟨hacc, ht⟩

end FlockDeploy.KS
