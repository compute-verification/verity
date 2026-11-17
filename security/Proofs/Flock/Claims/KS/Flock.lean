module

public import Proofs.Flock.Claims.KS.Wait
public import Proofs.Flock.Claims.KS.Fork
public import Definitions.Claims
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
import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.Prob

/-! Flock's KS at `E^rep` from one wait's bound, with `K > 0`, `ε_table ≥ 0`, `k ≥ 8` and the relation splitting by table as side conditions. -/

@[expose] public section

namespace Claims

namespace Flock

theorem check_sound (parse : Msg → Option Stmt) : ∀ x w, check parse x w = true → R parse x w := by
  intro x w h
  unfold check at h
  split at h
  · rename_i s hs
    exact ⟨s, hs, of_decide_eq_true h⟩
  · simp at h

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

def ExtractsAt (D : Deploy Coin Col T) (x : Msg) (vs : List Seen) (j : ℕ) : Prop :=
  ∃ (M : T) (t : Stmt), D.table x j = some t ∧ RelS t (D.read [M]) ∧
    (1 - D.θ M) * D.len M ≤ ((Finset.univ : Finset (Fin (D.len M))).filter fun i =>
      D.enc M i.val = fill (D.len M) (vs.flatMap fun v => D.cols v j) i).card

def Extracts (D : Deploy Coin Col T) (x : Msg) (vs : List Seen) : Prop :=
  ∃ s, D.parse x = some s ∧ ∀ j < s.tables, ExtractsAt D x vs j

def Splits (D : Deploy Coin Col T) (x : Msg) : Prop :=
  ∀ s, D.parse x = some s → ∀ Ms : List T, Ms.length = s.tables →
    (∀ j (hj : j < Ms.length), ∃ t, D.table x j = some t ∧ RelS t (D.read [Ms[j]])) → RelS s (D.read Ms)

theorem flatMap_singleton_flatMap {α β : Type} (f : α → ℕ → List β) (j : ℕ) :
    ∀ vs : List α, ((vs.flatMap fun v => [f v]).flatMap fun o => o j) = vs.flatMap fun v => f v j
  | [] => rfl
  | v :: vs => by
    simp only [List.flatMap_cons, List.cons_append, List.nil_append]
    rw [flatMap_singleton_flatMap f j vs]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem assembles_of_extracts (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (hlw : ListWithin D.len D.enc D.θ decode) (x : Msg) (vs : List Seen) (hsp : Splits D x) (h : Extracts D x vs) :
    Assembles (fun v => [D.cols v]) (assemble D.parse D.table decode D.read x) (check D.parse x) vs := by
  obtain ⟨s, hs, hall⟩ := h
  have hsome := assemble_isSome (table := D.table) (decode := decode) (read := D.read)
    (opens := vs.flatMap fun v => [D.cols v]) hs fun j hj => by
      obtain ⟨M, t, ht, hR, hag⟩ := hall j hj
      refine ⟨t, ht, M, ?_, hR⟩
      rw [flatMap_singleton_flatMap]
      exact hlw _ M hag
  obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 hsome
  refine ⟨w, hw, ?_⟩
  obtain ⟨Ms, rfl, hlen, hMs⟩ := assemble_eq_some hs hw
  unfold check
  rw [hs]
  exact decide_eq_true (hsp s hs Ms hlen fun j hj => by
    obtain ⟨t, ht, -, hR⟩ := hMs j hj
    exact ⟨t, ht, hR⟩)

def OneWait (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x, e.Fits (D.weight x) ∧
    ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (D.k x) (Extracts D x) / accPr H D.V P x ω ^ D.k x) ≤
        D.εtable x + e.term ω + D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem rate_le (D : Deploy Coin Col T) (x : Msg) (hk : 0 < D.k x) : D.N₀ x / (Real.exp 1 * D.k x) ≤ 1 / 4 := by
  have hk' : (0 : ℝ) < D.k x := by exact_mod_cast hk
  have he := Real.exp_pos 1
  rw [div_le_iff₀ (mul_pos he hk')]
  have hc : 4 * D.N₀ x / Real.exp 1 ≤ D.k x := Nat.le_ceil _
  rw [div_le_iff₀ he] at hc
  linarith

def NoAcc (H : Msg → Msg) (V : Verifier Coin) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin), (run H P V x ω r).out ≠ .acc

theorem KS_of_oneWait (H : Msg → Msg) (D : Deploy Coin Col T) (hK : ∀ x, 0 < D.cap x)
    (hk : ∀ x, (8 ≤ D.k x ∧ Splits D x) ∨ NoAcc H D.V x) (hεt : ∀ x, 0 ≤ D.εtable x) (h : OneWait H D) :
    KS H D := by
  intro decode hlw x Ω _ _ P
  rcases hk x with ⟨hk8, hsp⟩ | hno
  swap
  · have h0 : PrBad H D.V (R D.parse) (E D decode) x P = 0 := by
      unfold PrBad
      rw [Pr_congr (q := fun _ => False) fun s => ⟨fun hs => hno Ω P s.1 s.2.1 hs.1, False.elim⟩, Pr_false]
    rw [h0]
    exact add_nonneg (κ_pos D x (hεt x)).le (Real.sqrt_nonneg _)
  obtain ⟨e, hfit, hw⟩ := h x Ω P
  have hwait : ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (D.k x)
        (Assembles (fun v => [D.cols v]) (assemble D.parse D.table decode D.read x) (check D.parse x)) /
          accPr H D.V P x ω ^ D.k x) ≤
        (D.εtable x + e.term ω) + D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω := by
    intro ω hε
    refine le_trans ?_ ((hw ω hε).trans (le_of_eq (by ring)))
    refine mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right ?_ (pow_nonneg hε.le _)) hε.le
    unfold accFail
    exact Pr_mono fun σ ⟨hacc, hno⟩ => ⟨hacc, fun hex => hno (assembles_of_extracts D decode hlw x _ hsp hex)⟩
  have hb := ERep_trunc H D.V (R D.parse) D.waits D.k D.cap (fun v => [D.cols v])
    (assemble D.parse D.table decode D.read) (check D.parse) (check_sound D.parse) x P (hK x) hk8
    (fun ω => D.εtable x + e.term ω) (fun ω => add_nonneg (hεt x) (e.term_nonneg ω)) (rate_le D x (by omega)) hwait
  have hEx : 4 * Ex (fun ω => D.εtable x + e.term ω) ≤
      4 * D.εtable x + Real.sqrt (γ D x * PrB H D.V P x) := by
    rw [Ex_add, Ex_const]
    have := e.le (D.weight x) hfit
    unfold γ
    linarith
  show PrBad H D.V (R D.parse)
      (ERep D.V D.waits D.k D.cap (fun v => [D.cols v]) (assemble D.parse D.table decode D.read) (check D.parse)) x
        P ≤
    κ D x + Real.sqrt (γ D x * PrB H D.V P x)
  unfold κ
  linarith

end Flock

end Claims
