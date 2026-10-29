import Definitions.Pouw.Assumptions.Dimension.W1Accel
import Proofs.Pouw.Dimension.H200MeasuredProofs
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.NNRat
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-!
# W1 over the accelerator model: proofs

`freshWrites` (a run changes at most the words it writes, at every grid point), `w1FreshWords` (the card-level bound
from `CardWithinAccelW1`), and the H200 instance's one-tick witness (`h200FreshTick`, `w1TimeH200_le`).
-/

open scoped NNRat

namespace Pouw.Dimension.W1Accel

open Verity Verity.Accel Verity.Accel.Machine

variable {ι : Type} {m : Machine ι}

/-- `M'` agrees with `M` outside some `k` locations. -/
def Within (M M' : m.Mem) (k : ℕ) : Prop := ∃ W : List m.Loc, W.length ≤ k ∧ ∀ l, l ∉ W → M' l = M l

theorem Within.refl (M : m.Mem) (k : ℕ) : Within M M k := ⟨[], Nat.zero_le _, fun _ _ => rfl⟩

theorem Within.trans {M M' M'' : m.Mem} {a b : ℕ} (h₁ : Within M M' a) (h₂ : Within M' M'' b) :
    Within M M'' (a + b) := by
  obtain ⟨W, hW, hW'⟩ := h₁
  obtain ⟨V, hV, hV'⟩ := h₂
  refine ⟨W ++ V, by rw [List.length_append]; omega, fun l hl => ?_⟩
  rw [List.mem_append, not_or] at hl
  rw [hV' l hl.2, hW' l hl.1]

theorem within_writeIn (s : m.Store) (x : Operand) (M₀ M : m.Mem) (w : Word) :
    Within M (x.writeIn s M₀ M w) 1 := by
  unfold Operand.writeIn
  split
  next l _ => exact ⟨[l], le_rfl, fun l' hl' => Function.update_of_ne (List.ne_of_not_mem_cons hl') _ _⟩
  next => exact Within.refl M 1

theorem within_foldl_writeIn (o : m.Op) (dst : Fin (m.nOut o) → Operand) (M₀ : m.Mem) (v : Fin (m.nOut o) → Word) :
    ∀ (is : List (Fin (m.nOut o))) (M : m.Mem),
      Within M (is.foldl (fun M' i => (dst i).writeIn (m.outStore o i) M₀ M' (v i)) M) is.length
  | [], M => Within.refl M 0
  | i :: is, M => by
      rw [List.foldl_cons, List.length_cons, Nat.add_comm]
      exact (within_writeIn (m.outStore o i) (dst i) M₀ M (v i)).trans (within_foldl_writeIn o dst M₀ v is _)

theorem within_execOp (o : m.Op) (src : Fin (m.nIn o) → Operand) (dst : Fin (m.nOut o) → Operand)
    (pred : Option ℕ) (M : m.Mem) : Within M (execOp o src dst pred M) (m.nOut o) := by
  unfold execOp
  try dsimp only
  split
  · exact Within.refl M _
  · have h := within_foldl_writeIn o dst M (m.eval o fun i => (src i).readIn (m.inStore o i) M)
      (List.finRange (m.nOut o)) M
    rwa [List.length_finRange] at h

variable {Θ : Type} [DecidableEq Θ]

theorem within_step (P : m.Prog Θ) (σ : m.State Θ) (θ : Θ) :
    Within σ.mem (step P σ θ).1.mem (((step P σ θ).2.toList.map outWords).sum) := by
  unfold step
  cases (P θ)[σ.pc θ]? with
  | none => exact Within.refl _ _
  | some i =>
    cases i with
    | op o src dst pred => exact within_execOp o src dst pred σ.mem
    | bra p t => exact Within.refl _ _

/-- `tick`'s step: thread `θ` steps, and its instruction joins the tick's. -/
def tickF (P : m.Prog Θ) (acc : m.State Θ × List m.Instr) (θ : Θ) : m.State Θ × List m.Instr :=
  ((step P acc.1 θ).1, acc.2 ++ (step P acc.1 θ).2.toList)

theorem tick_eq (P : m.Prog Θ) (σ : m.State Θ) (θs : List Θ) : tick P σ θs = θs.foldl (tickF P) (σ, []) := rfl

theorem within_tick_foldl (P : m.Prog Θ) :
    ∀ (θs : List Θ) (acc : m.State Θ × List m.Instr), ∃ k,
      Within acc.1.mem (θs.foldl (tickF P) acc).1.mem k ∧
        ((θs.foldl (tickF P) acc).2.map outWords).sum = (acc.2.map outWords).sum + k
  | [], acc => ⟨0, Within.refl _ _, rfl⟩
  | θ :: θs, acc => by
      obtain ⟨k, hk, hs⟩ := within_tick_foldl P θs (tickF P acc θ)
      refine ⟨((step P acc.1 θ).2.toList.map outWords).sum + k, (within_step P acc.1 θ).trans hk, ?_⟩
      rw [List.foldl_cons, hs]
      simp only [tickF, List.map_append, List.sum_append, Nat.add_assoc]

theorem within_tick (P : m.Prog Θ) (σ : m.State Θ) (θs : List Θ) :
    Within σ.mem (tick P σ θs).1.mem (((tick P σ θs).2.map outWords).sum) := by
  obtain ⟨k, hk, hs⟩ := within_tick_foldl P θs (σ, [])
  rw [tick_eq, hs]
  simpa using hk

theorem within_run (P : m.Prog Θ) :
    ∀ (sched : List (List Θ)) (σ : m.State Θ), Within σ.mem (run P σ sched).1.mem (writes (run P σ sched).2)
  | [], σ => Within.refl _ _
  | θs :: rest, σ => by
      have h := (within_tick P σ θs).trans (within_run P rest (tick P σ θs).1)
      simpa [run, writes, List.flatten_cons, List.map_append, List.sum_append] using h

/-- The locations of `Ls` whose word the run changes are at most the words it writes. -/
theorem changed_card_le (P : m.Prog Θ) (sched : List (List Θ)) (σ : m.State Θ) (Ls : Finset m.Loc) :
    (Ls.filter fun l => (run P σ sched).1.mem l ≠ σ.mem l).card ≤ writes (run P σ sched).2 := by
  obtain ⟨W, hW, hW'⟩ := within_run P sched σ
  refine le_trans (Finset.card_le_card fun l hl => ?_) ((List.toFinset_card_le W).trans hW)
  rw [Finset.mem_filter] at hl
  rw [List.mem_toFinset]
  by_contra h
  exact hl.2 (hW' l h)

/-- **Fresh writes need writes**, proved. -/
theorem freshWrites : FreshWrites := by
  intro ι m Θ _ G P sched S σ Ls
  calc ∑ l ∈ Ls, (S.filter fun g => (run P (σ g) sched).1.mem l ≠ (σ g).mem l).card
      = ∑ g ∈ S, (Ls.filter fun l => (run P (σ g) sched).1.mem l ≠ (σ g).mem l).card := by
        simp only [Finset.card_filter]
        exact Finset.sum_comm
    _ ≤ ∑ g ∈ S, writes (run P (σ g) sched).2 :=
        Finset.sum_le_sum fun g _ => changed_card_le P sched (σ g) Ls

/-- **W1's fresh-words bound on the card**, proved from `CardWithinAccelW1`. -/
theorem w1FreshWords : W1FreshWords := by
  intro ι c cap oracle cw canW hW f w D hf G S M hD Ls hLs
  obtain ⟨x, -, -, h⟩ := hW
  obtain ⟨e, P, sched, hP⟩ := h f w D hf
  let σ : G → (c.machine x cap oracle).State ℕ := fun g => ⟨c.decode e (M g), fun _ => 0⟩
  have hrun : ∀ g ∈ S, (run P (σ g) sched).1.mem = c.decode e (f (M g)) ∧
      cw * (writes (run P (σ g) sched).2 : ℚ≥0) ≤ w := fun g hg => hP (M g) (σ g) (hD g hg) (fun _ => rfl) rfl
  have hdec : ∀ (N : c.Mem) l, l ∈ Ls → c.decode e N l = N l := fun N l hl => by
    simp [Card.decode, hLs l hl]
  have hcount : ∑ l ∈ Ls, (S.filter fun g => f (M g) l ≠ M g l).card =
      ∑ l ∈ Ls, (S.filter fun g => (run P (σ g) sched).1.mem l ≠ (σ g).mem l).card := by
    refine Finset.sum_congr rfl fun l hl => congrArg Finset.card (Finset.filter_congr fun g hg => ?_)
    rw [(hrun g hg).1, hdec _ l hl]
    show f (M g) l ≠ M g l ↔ f (M g) l ≠ c.decode e (M g) l
    rw [hdec _ l hl]
  have hle := freshWrites ι (c.machine x cap oracle) ℕ G P sched S σ Ls
  rw [hcount]
  calc cw * ((∑ l ∈ Ls, (S.filter fun g => (run P (σ g) sched).1.mem l ≠ (σ g).mem l).card : ℕ) : ℚ≥0)
      ≤ cw * ((∑ g ∈ S, writes (run P (σ g) sched).2 : ℕ) : ℚ≥0) :=
        mul_le_mul_of_nonneg_left (by exact_mod_cast hle) (by simp)
    _ = ∑ g ∈ S, cw * (writes (run P (σ g) sched).2 : ℚ≥0) := by
        rw [Nat.cast_sum, Finset.mul_sum]
    _ ≤ ∑ _g ∈ S, w := Finset.sum_le_sum fun g hg => (hrun g hg).2
    _ = S.card * w := by rw [Finset.sum_const, nsmul_eq_mul]

/-- **At `h200Measured`, W1 is 32 a word**, proved from `H100.h32H200Measured`. -/
theorem w1H200Measured32 : W1H200Measured32 := fun _ _ _ _ h => by
  simpa using h 32 H100.h32H200Measured

/-! ## The H200 instance's tick -/

/-- **One tick of the H200's instance writes 84,480 fresh words**, proved. -/
theorem h200FreshTick : H200FreshTick := by
  intro x
  dsimp only
  refine ⟨fun o ho => ?_, fun r => ?_, ?_⟩
  · rw [List.mem_append, List.mem_replicate, List.mem_replicate] at ho
    rcases ho with ⟨-, rfl⟩ | ⟨-, rfl⟩
    · exact And.intro rfl (Or.inl ⟨rfl, by decide, le_rfl⟩)
    · exact rfl
  · rw [List.map_append, List.map_replicate, List.map_replicate, List.sum_append, List.sum_replicate,
      List.sum_replicate, nsmul_eq_mul, nsmul_eq_mul]
    cases r <;> norm_num [h200Use, opUse, port, rd, Lane.pipe, H200.ρ]
  · rw [List.map_append, List.map_replicate, List.map_replicate, List.sum_append, List.sum_replicate,
      List.sum_replicate]
    norm_num [fresh, Lane.nOut]

/-- The open parameters at θ's floors: every floor met with equality, every open dot the accumulator. -/
def floorOpen : Open where
  slack := 1
  b1 := 1 / 8
  sync := 1
  madHi := 2
  dot := fun _ _ _ _ c _ _ => c
  dotSp := fun _ _ _ _ c _ _ => c

/-- **The time reading is at most 32/5 a fresh word**, proved. -/
theorem w1TimeH200_le : W1TimeH200Le := by
  intro c hc
  have hfl : H200.card.floors floorOpen := by
    show 1 ≤ floorOpen.sync ∧ 1 / 8 ≤ floorOpen.b1 ∧ 2 ≤ floorOpen.madHi ∧ 1 ≤ floorOpen.slack
    exact ⟨le_rfl, le_rfl, le_rfl, le_rfl⟩
  obtain ⟨havail, hfit, hfresh⟩ := h200FreshTick floorOpen
  have h := hc floorOpen hfl _ havail hfit
  rw [hfresh] at h
  have h' := NNRat.coe_le_coe.mpr h
  simp only [H200.ρ] at h'
  push_cast at h'
  rw [← NNRat.coe_le_coe]
  push_cast
  linarith

end Pouw.Dimension.W1Accel
