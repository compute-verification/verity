module

public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.Flock.SamplingGeneric.Efficient
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
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
import Proofs.Flock.Claims.Sampling.Cost
import Proofs.Flock.Claims.Sampling.Wait
import Proofs.Flock.SamplingGeneric.Bits

/-!
# Refutation witnesses: the generic sampled compiler's cost claims

`Compiled.Bits` and `Compiled.Efficient` each fail at a toy instance, on the toy statement of `Witnesses.Sampling`.

* `sksL` (`Witnesses.SamplingGeneric`) accepts every run while no witness is in its relation at `δ = 0`, and there
  `λ > 0`: `Compiled.Bits_fails`.
* `effL`: a compiled verifier over `findA`, a proof system whose verifier accepts after one round and whose extractor
  assembles `()` at once, with its one proof block's session seen as accepted, under a commitment whose opening check
  makes one SHA-512 call (`callCm`) that the compiled verifier counts as none. Every run is accepted, so the compiled
  extractor reaches its first inner run's extraction, whose opening check calls (`ownLog_E_pos`), above the own-call
  bound `0`: `Compiled.Efficient_fails`.

The lock's `meets` witnesses of these claims are `twoSessions_meets` (the contract with two sampled sessions) and
`twoOpenCalls_meets`: the opening checks' count holds at `twoL`, so `CompilerBits` and `CompilerEfficient` read
antecedents that hold together.
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Sampling (AuditStatement Opening CommitmentScheme lamLin forkOn waitAccC flatMapRP ownLog)
open Claims.Witness.Sampling (toyStmt sksD_noR slack_le_two)

/-! ## `Compiled.Bits` fails at `sksL` -/

theorem maxBelow_zero (n : ℕ) : maxBelow n (fun _ => 0) = 0 := by
  unfold maxBelow
  generalize List.range n = l
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.map_cons, List.foldr_cons, ih, max_self]

section Bits

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem sksL_γ (x : Msg) : (sksL Coin).γ x = 0 := by
  have h : (sksL Coin).γs x = 0 := by
    unfold Compiled.γs
    refine (congrArg (maxBelow _) (funext fun u => ?_)).trans (maxBelow_zero _)
    exact zero_div _
  unfold Compiled.γ
  rw [h, mul_zero, mul_zero]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- `sksL`'s one proof block's shape makes no wait, so its session's expected reruns are its replay. -/
theorem sksL_es (x : Msg) : (sksL Coin).es x = 1 := by
  show max ((1 : ℝ) + ((0 : ℕ) : ℝ) * ((0 : ℕ) : ℝ)) 0 = 1
  exact (max_eq_left (by norm_num)).trans (by norm_num)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem sksL_work (x : Msg) : (sksL Coin).work x ≤ 2 ^ 100 := by
  have he := sksL_es (Coin := Coin) x
  have hko : (sksL Coin).kOut x = 2 ^ 10 * 3 := rfl
  have hki : (sksL Coin).kIn x = 2 ^ 10 * 3 := rfl
  unfold Compiled.work
  rw [he, hko, hki, show (sksL Coin).nPU x = 1 from rfl, show (sksL Coin).openCalls x = 0 from rfl]
  norm_num

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem sksL_lam_pos (x : Msg) :
    0 < lamLin ((sksL Coin).κ 0 x) ((sksL Coin).γ x) 1 ((sksL Coin).work x) := by
  have hκ := sksL_κ (Coin := Coin) x
  have hκ0 := (sksL Coin).κ_pos le_rfl x
  have hw := sksL_work (Coin := Coin) x
  have hw0 := (sksL Coin).work_pos x
  have h3 : Real.logb 2 3 < 2 := by
    rw [Real.logb_lt_iff_lt_rpow (by norm_num) (by norm_num)]; norm_num
  have h1 : Real.logb 2 3 < Real.logb 2 (1 / (sksL Coin).κ 0 x) := by
    refine Real.logb_lt_logb (by norm_num) (by norm_num) ?_
    rw [lt_div_iff₀ hκ0]; linarith
  have hγ : Real.logb 2 ((sksL Coin).γ x) = 0 := by rw [sksL_γ, Real.logb_zero]
  have hW : Real.logb 2 (2 * 1 * (sksL Coin).work x) ≤ 101 := by
    rw [Real.logb_le_iff_le_rpow (by norm_num) (by positivity)]
    norm_num; linarith
  unfold lamLin
  have := lt_min (lt_min h1 (show Real.logb 2 3 < 255.5 - Real.logb 2 ((sksL Coin).γ x) / 2 by rw [hγ]; linarith))
    (show Real.logb 2 3 < 256 - Real.logb 2 (2 * 1 * (sksL Coin).work x) by linarith)
  linarith

theorem sksL_expectedCR (H : Msg → Msg) (x : Msg) :
    ExpectedCR H (Claims.Sampling.Bext H (sksL Coin).V (sksL Coin).E Unit silentP x) := by
  unfold ExpectedCR
  rw [sksL_bext]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

/-- **`Compiled.Bits` fails** at `sksL`, at `δ = 0`: the silent prover is accepted, no witness is in the relation, and
`λ > 0`. -/
theorem compiledBits_fails (H : Msg → Msg) : ¬ (sksL Coin).Bits H := by
  intro h
  have hp : PrBad H (sksL Coin).V (Claims.Sampling.R H (sksL Coin).commit (sksL Coin).parse (sksL Coin).partition 0)
      (sksL Coin).E [] silentP = 1 :=
    prBad_accV H _ _ [] silentP (sksD_noR (Coin := Coin) H 0 le_rfl [])
  have h2 : PrBad H (sksL Coin).V (Claims.Sampling.R H (sksL Coin).commit (sksL Coin).parse (sksL Coin).partition 0)
      (sksL Coin).E [] silentP ≤ (1 : ℕ) * (2 : ℝ) ^ (-lamLin ((sksL Coin).κ 0 []) ((sksL Coin).γ []) 1
        ((sksL Coin).work [])) :=
    h 0 le_rfl [] (sksL_scope []) Unit silentP 1 le_rfl (calls_silentP_accV (Coin := Coin) H [])
      (strictBirthday_B_accV (Coin := Coin) H silentP []) (sksL_expectedCR H [])
  rw [hp, Nat.cast_one, one_mul] at h2
  have : (2 : ℝ) ^ (-lamLin ((sksL Coin).κ 0 []) ((sksL Coin).γ []) 1
      ((sksL Coin).work [])) < 1 :=
    Real.rpow_lt_one_of_one_lt_of_neg (by norm_num) (by linarith [sksL_lam_pos (Coin := Coin) []])
  linarith

end Bits

/-! ## `Compiled.Efficient` fails at `effL` -/

/-- **A proof system whose extractor assembles `()` at once**: `toyA`'s verifier and relation, its extractor one wait
over one coin vector that stops at no accepted rerun, its assemble step `()` and its check passing everything. -/
def findA (Coin : Type) : Arg Coin :=
  { toyA Coin with
    assemble := fun _ _ => (some (), 0)
    check := fun _ _ => true
    waits := fun _ => 1
    cap := fun _ => 1 }

/-- The revealed opening: the input's value at position `0`. -/
def opening₀ : Opening Bool := ⟨0, true, []⟩

/-- A commitment whose every opening passes after one SHA-512 call. -/
def callCm : CommitmentScheme Bool := ⟨fun _ _ => .query [] fun _ => .ret true, fun _ _ _ => []⟩

/-- **A compiled verifier over `findA` whose opening check calls more than it counts**: it accepts after one round,
samples its one proof block, whose session is the run's statement seen as accepted, and its session's witness reveals
`opening₀` under `callCm`, counted as no call. -/
noncomputable def effL (Coin : Type) : Compiled Coin (findA Coin) where
  V := accV Coin
  parse _ := some toyStmt
  scopeCheck _ := true
  partition _ := some Claims.Witness.Sampling.toyPart
  commit := callCm
  stage2 _ := Finset.univ
  checked _ _ := [0]
  sh _ _ := ()
  κb _ _ := 0
  γb _ _ := 0
  rounds _ _ := Finset.univ
  session x _ _ := some (x, ⟨[], .acc⟩)
  iroots _ _ _ := []
  reveal _ _ _ _ := [([], opening₀)]
  openCalls _ := 0
  nOut _ := 1
  nIn _ := 1
  Tape := Unit
  P _ := silentP
  δc _ _ := 0

section OwnCalls

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

theorem exists_cons_of_length_pos {α : Type} {l : List α} (h : 0 < l.length) : ∃ a l', l = a :: l' :=
  List.exists_cons_of_ne_nil (List.ne_nil_of_length_pos h)

theorem ownLog_flatMapRP_head {R β γ : Type} (H : Msg → Msg) (run : R → Trace) (f : β → RP R (List γ)) (b : β)
    (bs : List β) : (ownLog H run (f b)).length ≤ (ownLog H run (flatMapRP f (b :: bs))).length := by
  rw [Claims.Sampling.ownLog_flatMapRP, List.map_cons, List.sum_cons]
  exact Nat.le_add_right _ _

/-- A wait whose first sample is accepted keeps it first. -/
theorem waitAccC_head {R : Type} (H : Msg → Msg) (run : R → Trace) {k : ℕ} (hk : 0 < k) (r : R) (rs : List R)
    (hr : (run r).out = .acc) : ∃ l, ((waitAccC k (r :: rs) []).eval H run).1 = (r, (run r).seen) :: l := by
  rw [Claims.Sampling.waitAccC_eval_nil, Claims.Sampling.accList,
    List.filter_cons_of_pos (p := fun r => decide ((run r).out = .acc)) (decide_eq_true hr), List.take_cons hk,
    List.map_cons]
  exact ⟨_, rfl⟩

/-- **The compiled extractor makes a SHA-512 call of its own** wherever every run is accepted and each accepted run's
extractions make one: its waits keep their first samples, so the first inner run's extractions are made. -/
theorem ownLog_E_pos (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (L.V.k x) → Coin) (c : L.E.C x) {s : AuditStatement} (hs : L.parse x = some s)
    (hacc : ∀ r, (run H P L.V x ω r).out = .acc)
    (hrun : ∀ r v c', 0 < (ownLog H (run H P L.V x ω) (L.runExt x s r v c')).length) :
    0 < (ownLog H (run H P L.V x ω) (L.E.ext x r₀ c)).length := by
  have h₀ : (run H P L.V x ω r₀).seen.out = .acc := hacc r₀
  obtain ⟨a, as, ha⟩ :=
    exists_cons_of_length_pos (l := List.finRange (L.kOut x)) (by rw [List.length_finRange]; exact L.kOut_pos x)
  obtain ⟨b, bs, hb⟩ :=
    exists_cons_of_length_pos (l := List.finRange (L.kIn x)) (by rw [List.length_finRange]; exact L.kIn_pos x)
  obtain ⟨l₀, ls, hl⟩ :=
    exists_cons_of_length_pos (l := List.finRange (L.capOut x)) (by rw [List.length_finRange]; exact L.capOut_pos x)
  obtain ⟨m₀, ms, hm⟩ :=
    exists_cons_of_length_pos (l := List.finRange (L.capIn x)) (by rw [List.length_finRange]; exact L.capIn_pos x)
  simp only [Compiled.E, ownLog, hs, ite_eq_left h₀]
  rw [Claims.Sampling.ownLog_bind, Claims.Sampling.ownLog_of_noHash _ _ (Claims.Sampling.NoHash.waitAccC _ _ _),
    List.nil_append, Claims.Sampling.ownLog_bind, List.length_append]
  refine Nat.lt_of_lt_of_le ?_ (Nat.le_add_right _ _)
  rw [hl, List.map_cons]
  obtain ⟨o, ho⟩ := waitAccC_head H (run H P L.V x ω) (L.kOut_pos x) (c.1 l₀) (ls.map c.1) (hacc _)
  rw [ho, ha, List.zip_cons_cons]
  refine Nat.lt_of_lt_of_le ?_ (ownLog_flatMapRP_head _ _ _ _ _)
  dsimp only
  rw [Claims.Sampling.ownLog_bind, Claims.Sampling.ownLog_of_noHash _ _ (Claims.Sampling.NoHash.waitAccC _ _ _),
    List.nil_append, hm, List.map_cons]
  obtain ⟨o', ho'⟩ := waitAccC_head H (run H P L.V x ω) (L.kIn_pos x) (forkOn (L.stage2 x) (c.1 l₀) (c.2.1 a m₀))
    (ms.map fun l => forkOn (L.stage2 x) (c.1 l₀) (c.2.1 a l)) (hacc _)
  rw [ho', hb, List.zip_cons_cons]
  refine Nat.lt_of_lt_of_le ?_ (ownLog_flatMapRP_head _ _ _ _ _)
  exact hrun _ _ _

end OwnCalls

section Efficient

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- Each of `effL`'s runs' extractions makes a SHA-512 call of its own: its one slot's session finds `()` at once, and
checking the opening it reveals makes `callCm`'s one call. -/
theorem effL_runExt (H : Msg → Msg) (x : Msg) (run : (Fin ((effL Coin).V.k x) → Coin) → Trace)
    (r : Fin ((effL Coin).V.k x) → Coin) (v : Seen) (c : (effL Coin).SlotEC x) :
    0 < (ownLog H run ((effL Coin).runExt x toyStmt r v c)).length := by
  have h1 : 0 < (effL Coin).nPU x := Nat.one_pos
  obtain ⟨d, ds, hd⟩ :=
    exists_cons_of_length_pos (l := List.finRange ((effL Coin).nPU x)) (by rw [List.length_finRange]; exact h1)
  unfold Compiled.runExt
  rw [hd, show (effL Coin).checked x r = [0] from rfl, List.zip_cons_cons]
  refine Nat.lt_of_lt_of_le ?_ (ownLog_flatMapRP_head _ _ _ _ _)
  dsimp only
  rw [dite_eq_left h1, Compiled.unitExt, show (effL Coin).session x 0 v = some (x, ⟨[], .acc⟩) from rfl]
  dsimp only
  rw [dite_eq_left (show (findA Coin).shape x = (effL Coin).sh x 0 from rfl)]
  exact Nat.succ_pos _

/-- **`Compiled.Efficient` fails** at `effL`: every run is accepted, so the compiled extractor reaches its first inner
run's extraction, whose opening check makes a SHA-512 call of its own, while the own-call bound is `0`. -/
theorem compiledEfficient_fails (H : Msg → Msg) : ¬ (effL Coin).Efficient H := by
  rintro ⟨-, -, h⟩
  have c : (effL Coin).E.C [] := Classical.arbitrary _
  have h1 := h [] Unit silentP () (fun _ => default) c
  have h0 : (effL Coin).kOut [] * (effL Coin).kIn [] * (effL Coin).nPU [] * (effL Coin).openCalls [] = 0 := by
    rw [show (effL Coin).openCalls [] = 0 from rfl, mul_zero]
  have hpos := ownLog_E_pos (effL Coin) H silentP [] () (fun _ => default) c (s := toyStmt) rfl
    (fun r => run_accV_out _ _ _ _ r) fun r v c' => effL_runExt H [] _ r v c'
  exact Nat.lt_irrefl 0 (Nat.lt_of_lt_of_le hpos (h1.trans_eq h0))

end Efficient

/-! ## The opening checks' count holds beside the contract -/

section Meets

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The opening checks' count holds at `twoL`** (`CompilerBits`' and `CompilerEfficient`'s premise beyond the
contract): `trueCm`'s opening is its value, so checking it makes no call. With `twoSessions_meets`, `CompilerBits` and
`CompilerEfficient` read antecedents that hold together. -/
theorem twoOpenCalls_meets (H : Msg → Msg) : (twoL Coin).OpenCallsLe H := by
  intro x s _ u v w
  exact Nat.le_of_eq rfl

end Meets

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.SamplingGeneric

/-- **`Compiled.Bits` fails** at some compiled verifier. -/
theorem Compiled.Bits_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : Compiled Coin A), ¬ L.Bits H :=
  ⟨_, _, Witness.SamplingGeneric.compiledBits_fails H⟩

/-- **`Compiled.Efficient` fails** at some compiled verifier. -/
theorem Compiled.Efficient_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : Compiled Coin A), ¬ L.Efficient H :=
  ⟨_, _, Witness.SamplingGeneric.compiledEfficient_fails H⟩

end Claims.SamplingGeneric
