import Definitions.Core.Assumptions.Accel
import Mathlib.Algebra.Order.Monoid.Canonical.Basic
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# The machine 𝓜: an honest semantics, the rate bound, and the coverage premise's refusal

* `s8Open`: a semantics with one MMA opcode, `s8` by `s8` into 32-bit accumulators, computing `C + Σ a·b` exactly
  (`intDot`), the deployed kernels' `wgmma`. It meets (P1) on every card (`s8Open_exact`) and (P2)
  (`s8Open_productForm`), so the headline's hypotheses on the semantics are met.
* `units_le_of_fits`: a run meeting the model uses each resource at most at its rate over its seconds, and so at most
  the device model's limits over a window of ticks (`fits_toDevice`, the table's refinement of `Accounting.Device`).
* `cardWithinAccel_fails` (the nontrivial lemma): the coverage premise refuses a card that changes its memory in no
  time. Every instruction uses part of an issue slot (`use_issue_pos`), so a run of zero seconds runs none and leaves
  the memory as it was (`run_mem_of_issue_zero`).
-/

open scoped NNRat

namespace Verity.Accel

/-- The exact integer dot: `C + Σ a·b`, as a 32-bit two's complement word. -/
def intDot (fa fb : Fmt) : Dot := fun acc a b =>
  (BitVec.ofInt 32 ((BitVec.ofNat 32 acc).toInt +
    (List.zipWith (fun p q => fa.intVal p * fb.intVal q) a b).sum)).toNat

/-- A multiplier output's signed product. -/
def signed (t : Verity.TC.Spec.Cls × Verity.TC.Spec.Cls × Bool × ℕ × ℕ) : ℤ :=
  if t.2.2.1 then -(t.2.2.2.1 : ℤ) else (t.2.2.2.1 : ℤ)

theorem signed_mulOut_s8 (p q : ℕ) :
    signed (mulOut .s8 .s8 p q) = Fmt.intVal .s8 p * Fmt.intVal .s8 q := by
  generalize hz : Fmt.intVal .s8 p * Fmt.intVal .s8 q = z
  have hm : mulOut .s8 .s8 p q = (.finite, .finite, decide (z < 0), z.natAbs, 0) := by
    rw [← hz]; rfl
  rw [hm]
  unfold signed
  by_cases h : z < 0
  · simp only [decide_eq_true h, ↓reduceIte]
    show -((z.natAbs : ℕ) : ℤ) = z
    rw [Int.ofNat_natAbs_of_nonpos h.le, neg_neg]
  · simp only [decide_eq_false h, Bool.false_eq_true, ↓reduceIte]
    show ((z.natAbs : ℕ) : ℤ) = z
    exact Int.natAbs_of_nonneg (not_lt.mp h)

/-- **The exact `s8` dot is in product form** at every `k`. -/
theorem s8_productForm (k : ℕ) : ProductForm .s8 .s8 k (intDot .s8 .s8) := by
  refine ⟨fun c ps _ _ => (BitVec.ofInt 32 ((BitVec.ofNat 32 c).toInt + (ps.map signed).sum)).toNat, ?_⟩
  intro c a b ha hb
  have ha' : a.length ≤ k := le_of_eq ha
  have hb' : b.length ≤ k := le_of_eq hb
  try dsimp only
  rw [List.take_of_length_le ha', List.take_of_length_le hb', List.map_zipWith]
  simp only [intDot, signed_mulOut_s8]

/-- **An honest semantics**: one MMA opcode, `s8` by `s8` into 32-bit accumulators, exact (`intDot`), dense and 2:4
sparse; no lane or read-modify-write opcodes. -/
def s8Open : Open where
  Lane := Empty
  pipe := fun e => nomatch e
  lane := fun e => nomatch e
  Atom := Empty
  atom := fun e => nomatch e
  Mma := Unit
  fa _ := .s8
  fb _ := .s8
  wide _ := false
  dot _ _ := intDot .s8 .s8
  dotSp _ _ := intDot .s8 .s8

/-- `s8Open` meets (P1) on every card: its integer dot is exact at any range. -/
theorem s8Open_exact (c : Card) : c.Exact s8Open := by
  intro o k acc a b _ _ _ _ _ _
  rfl

/-- `s8Open` meets (P2). -/
theorem s8Open_productForm : Verity.Assumptions.AccelProductForm s8Open :=
  fun _ k => ⟨s8_productForm k, s8_productForm (k / 2)⟩

namespace Machine

variable {m : Machine}

/-- **The rate bound**: a run meeting the model uses each resource at most `ρ_r T`. -/
theorem units_le_of_fits {Θ : Type} [DecidableEq Θ] (P : m.Prog Θ) (σ : m.State Θ) (sched : List Θ) (T : ℚ≥0)
    (h : Fits P σ sched T) (r : Res) : units (run P σ sched).2 r ≤ m.card.ρ r * T :=
  le_self_add.trans (h.2 r)

/-- **The table refines the device model**: a run meeting the model in `len` ticks of `τ` seconds uses at most the
device's tensor MACs, lane ops (32 an issue slot) and device-memory bits summed over the window (`Card.toDevice`). -/
theorem fits_toDevice {Θ : Type} [DecidableEq Θ] (P : m.Prog Θ) (σ : m.State Θ) (sched : List Θ) (τ : ℚ≥0)
    (len : ℕ) {ι : Type} (cap : ι → ℕ) (h : Fits P σ sched (len * τ)) :
    units (run P σ sched).2 .tensor ≤ len * ((m.card.toDevice τ cap).rate .tc : ℚ≥0) ∧
      32 * units (run P σ sched).2 .issue ≤ len * ((m.card.toDevice τ cap).rate .cuda : ℚ≥0) ∧
      32 * units (run P σ sched).2 .dram ≤ len * ((m.card.toDevice τ cap).bw : ℚ≥0) := by
  have key : ∀ (r : Res) (k : ℚ≥0), k * units (run P σ sched).2 r ≤ len * (⌈k * m.card.ρ r * τ⌉₊ : ℚ≥0) := by
    intro r k
    calc k * units (run P σ sched).2 r ≤ k * (m.card.ρ r * (len * τ)) := by
          gcongr; exact units_le_of_fits P σ sched _ h r
      _ = len * (k * m.card.ρ r * τ) := by ring
      _ ≤ len * (⌈k * m.card.ρ r * τ⌉₊ : ℚ≥0) := by gcongr; exact Nat.le_ceil _
  have h1 := key .tensor 1
  simp only [one_mul] at h1
  exact ⟨h1, key .issue 32, key .dram 32⟩

theorem port_issue (s : Store) : port s .issue = 0 := by cases s <;> rfl

theorem rd_issue (s : Store) : rd s .issue = 0 := by
  cases s <;> first | rfl | simp [rd, port]

/-- Every instruction uses part of an issue slot. -/
theorem use_issue_pos (i : m.Instr) : 0 < i.use .issue := by
  rcases i with ⟨o, _, _, _⟩ | ⟨_, _, _⟩
  · cases o <;>
      simp only [Instr.use, Card.use, slot, rd_issue, port_issue, mul_zero, add_zero, zero_add, ↓reduceIte,
        reduceCtorEq, and_false] <;>
      positivity
  · simp only [Instr.use, slot, ↓reduceIte]
    norm_num

theorem units_append (a b : List m.Instr) (r : Res) : units (a ++ b) r = units a r + units b r := by
  simp [units]

/-- A step that runs no instruction leaves the state as it was. -/
theorem step_none {Θ : Type} [DecidableEq Θ] (P : m.Prog Θ) (σ : m.State Θ) (θ : Θ)
    (h : (step P σ θ).2 = none) : (step P σ θ).1 = σ := by
  unfold step at h ⊢
  rcases hP : (P θ)[σ.pc θ]? with _ | (_ | _) <;> simp only [hP] at h ⊢ <;>
    first | rfl | (split at h <;> simp at h) | simp_all

/-- **A run that uses no issue slot runs no instruction**, so it leaves the memory as it was. -/
theorem run_mem_of_issue_zero {Θ : Type} [DecidableEq Θ] (P : m.Prog Θ) :
    ∀ (σ : m.State Θ) (sched : List Θ), units (run P σ sched).2 .issue = 0 → (run P σ sched).1.mem = σ.mem
  | _, [], _ => rfl
  | σ, θ :: rest, h => by
    simp only [run, units_append] at h ⊢
    obtain ⟨h1, h2⟩ := add_eq_zero.mp h
    have hn : (step P σ θ).2 = none := by
      cases hs : (step P σ θ).2 with
      | none => rfl
      | some i =>
        rw [hs] at h1
        have := use_issue_pos i
        simp [units] at h1
        exact absurd h1 this.ne'
    rw [run_mem_of_issue_zero P _ rest h2, step_none P σ θ hn]

end Machine

/-- **The nontrivial lemma: the coverage premise refuses a card that computes in no time.** If `can` says the card
takes some memory `M` in `D` to `f M ≠ M` in zero seconds, `CardWithinAccel c can` is false: a run of zero seconds
uses no issue slot (G2), so it runs no instruction and leaves `M` as it was. -/
theorem cardWithinAccel_fails (c : Card) (can : (c.Mem → c.Mem) → ℚ≥0 → (c.Mem → Prop) → Prop)
    (f : c.Mem → c.Mem) (D : c.Mem → Prop) (M : c.Mem) (hcan : can f 0 D) (hD : D M) (hf : f M ≠ M) :
    ¬ Verity.Assumptions.CardWithinAccel c can := by
  rintro ⟨x, _, _, h⟩
  obtain ⟨P, sched, hrun⟩ := h f 0 D hcan
  obtain ⟨hfits, hmem⟩ := hrun ⟨M, fun _ => 0⟩ hD (fun _ => rfl)
  have h0 : Machine.units (Machine.run P ⟨M, fun _ => 0⟩ sched).2 .issue = 0 :=
    le_antisymm ((Machine.units_le_of_fits P _ sched 0 hfits .issue).trans (by simp)) zero_le
  exact hf (hmem.symm.trans (Machine.run_mem_of_issue_zero P _ sched h0))

end Verity.Accel
