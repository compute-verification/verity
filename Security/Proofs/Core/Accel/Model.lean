import Definitions.Core.Assumptions.Accel
import Mathlib.Algebra.Order.Monoid.Canonical.Basic
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# The machine 𝓜: an honest semantics, the rate bound, and the card premise's refutations

* `s8Open`: a semantics with one MMA opcode, `s8` by `s8` into 32-bit accumulators, computing `C + Σ a·b` exactly
  (`intDot`), the deployed kernels' `wgmma`. It meets (P1) on every card (`s8Open_exact`) and (P2)
  (`s8Open_productForm`), so the headline's hypotheses on the semantics are met.
* `units_le_of_fits`: a run meeting the model uses each resource at most at its rate over its seconds, and so at most
  the device model's limits summed over a window of ticks (`fits_toDevice`, at `Card.toDevice`).
* `cardWithinAccel_fails`: the card premise is false for a card that changes its memory in no time. Every instruction
  uses part of an issue slot (`use_issue_pos`), so a run of zero seconds runs none and leaves the memory as it was
  (`run_mem_of_issue_zero`).
* `cardWithinAccel_fails_rate`: it is false for a card that, in `T` seconds, changes more words of shared memory, L2 or
  device memory than the store's resource carries in `T` seconds. Each word an instruction writes there is a unit of
  that resource (`outTo_le_use`), so a run changes at most its units (`changed_run`).
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

/-- **The window bound**: a run meeting the model in `len` ticks of `τ` seconds uses at most the device's tensor MACs,
CUDA-core ops (32 an issue slot, and every pipe's lanes) and device-memory bits summed over the window
(`Card.toDevice`). It builds no per-tick `Accounting.Schedule`. -/
theorem fits_toDevice {Θ : Type} [DecidableEq Θ] (P : m.Prog Θ) (σ : m.State Θ) (sched : List Θ) (τ : ℚ≥0)
    (len : ℕ) {ι : Type} (cap : ι → ℕ) (h : Fits P σ sched (len * τ)) :
    units (run P σ sched).2 .tensor ≤ len * ((m.card.toDevice τ cap).rate .tc : ℚ≥0) ∧
      32 * units (run P σ sched).2 .issue + units (run P σ sched).2 .alu + units (run P σ sched).2 .imad +
          units (run P σ sched).2 .shfl + units (run P σ sched).2 .f64 + units (run P σ sched).2 .sfu ≤
        len * ((m.card.toDevice τ cap).rate .cuda : ℚ≥0) ∧
      32 * units (run P σ sched).2 .dram ≤ len * ((m.card.toDevice τ cap).bw : ℚ≥0) := by
  have u := fun r => units_le_of_fits P σ sched _ h r
  have key : ∀ (r : Res) (k : ℚ≥0), k * units (run P σ sched).2 r ≤ len * (⌈k * m.card.ρ r * τ⌉₊ : ℚ≥0) := by
    intro r k
    calc k * units (run P σ sched).2 r ≤ k * (m.card.ρ r * (len * τ)) := by gcongr; exact u r
      _ = len * (k * m.card.ρ r * τ) := by ring
      _ ≤ len * (⌈k * m.card.ρ r * τ⌉₊ : ℚ≥0) := by gcongr; exact Nat.le_ceil _
  have h1 := key .tensor 1
  simp only [one_mul] at h1
  refine ⟨h1, ?_, key .dram 32⟩
  calc _ ≤ 32 * (m.card.ρ .issue * (len * τ)) + m.card.ρ .alu * (len * τ) + m.card.ρ .imad * (len * τ) +
          m.card.ρ .shfl * (len * τ) + m.card.ρ .f64 * (len * τ) + m.card.ρ .sfu * (len * τ) := by
        gcongr <;> exact u _
    _ = len * ((32 * m.card.ρ .issue + m.card.ρ .alu + m.card.ρ .imad + m.card.ρ .shfl + m.card.ρ .f64 +
          m.card.ρ .sfu) * τ) := by ring
    _ ≤ _ := by gcongr; exact Nat.le_ceil _

theorem port_issue (s : Store) : port s .issue = 0 := by cases s <;> rfl

theorem regWr_issue (s : Store) : regWr s .issue = 0 := by simp [regWr]

/-- Every instruction uses part of an issue slot. -/
theorem use_issue_pos (i : m.Instr) : 0 < i.use .issue := by
  rcases i with ⟨o, _, _, _⟩ | ⟨_, _, _⟩
  · cases o <;>
      simp only [Instr.use, Card.use, slot, port_issue, regWr_issue, mul_zero, add_zero, zero_add, ↓reduceIte,
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

/-- The words of store `t` where `M'` differs from `M`. -/
def changed (t : Store) (M M' : m.Mem) : Finset (Fin (m.card.size t)) :=
  Finset.univ.filter fun q => M' ⟨t, q⟩ ≠ M ⟨t, q⟩

theorem changed_self (t : Store) (M : m.Mem) : changed t M M = ∅ := by
  simp [changed]

theorem changed_trans (t : Store) (M M' M'' : m.Mem) :
    (changed t M M'').card ≤ (changed t M M').card + (changed t M' M'').card := by
  refine (Finset.card_le_card fun q hq => ?_).trans (Finset.card_union_le _ _)
  simp only [changed, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union] at hq ⊢
  by_contra h
  rw [not_or, not_not, not_not] at h
  exact hq (h.2.trans h.1)

theorem changed_update (t : Store) (M : m.Mem) (l : m.Loc) (w : Word) :
    (changed t M (Function.update M l w)).card ≤ if l.1 = t then 1 else 0 := by
  split_ifs with h
  · refine Finset.card_le_one.mpr fun a ha b hb => ?_
    simp only [changed, Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
    have ha' : (⟨t, a⟩ : m.Loc) = l := by
      by_contra e
      exact ha (Function.update_of_ne e _ _)
    have hb' : (⟨t, b⟩ : m.Loc) = l := by
      by_contra e
      exact hb (Function.update_of_ne e _ _)
    exact eq_of_heq (Sigma.mk.inj_iff.mp (ha'.trans hb'.symm)).2
  · rw [Nat.le_zero, Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
    intro q hq
    simp only [changed, Finset.mem_filter, Finset.mem_univ, true_and] at hq
    apply hq
    rw [Function.update_of_ne]
    rintro rfl
    exact h rfl

theorem changed_writeIn (t s : Store) (x : Operand) (M₀ M : m.Mem) (w : Word) :
    (changed t M (x.writeIn s M₀ M w)).card ≤ if s = t then 1 else 0 := by
  rcases x with i | ⟨r, off⟩ | w'
  · exact changed_update t M (m.loc s i) w
  · exact changed_update t M (m.loc s ((M₀ (m.loc .reg r)).toNat + off)) w
  · simp [Operand.writeIn, Operand.locIn, changed_self]

theorem changed_foldl (t : Store) (o : Op m.sem) (dst : Fin o.nOut → Operand) (M₀ : m.Mem)
    (v : Fin o.nOut → Word) : ∀ (l : List (Fin o.nOut)) (M : m.Mem),
      (changed t M (l.foldl (fun M' i => (dst i).writeIn (o.outStore i) M₀ M' (v i)) M)).card ≤
        (l.filter fun i => o.outStore i = t).length
  | [], M => by simp [changed_self]
  | i :: l, M => by
    rw [List.foldl_cons]
    refine (changed_trans t M ((dst i).writeIn (o.outStore i) M₀ M (v i)) _).trans ?_
    have h1 := changed_writeIn t (o.outStore i) (dst i) M₀ M (v i)
    have h2 := changed_foldl t o dst M₀ v l ((dst i).writeIn (o.outStore i) M₀ M (v i))
    by_cases h : o.outStore i = t
    · rw [ite_eq_left h] at h1
      rw [List.filter_cons_of_pos (by simpa using h), List.length_cons]
      omega
    · rw [ite_eq_right h] at h1
      rw [List.filter_cons_of_neg (by simpa using h)]
      omega

theorem changed_execOp (t : Store) (o : Op m.sem) (src : Fin o.nIn → Operand) (dst : Fin o.nOut → Operand)
    (pred : Option (ℕ × ℕ)) (M : m.Mem) :
    (changed t M (execOp o src dst pred M)).card ≤ ((List.finRange o.nOut).filter fun i => o.outStore i = t).length := by
  simp only [execOp]
  split_ifs
  · simp [changed_self]
  · exact changed_foldl t o dst M _ _ M

theorem port_ne_reg {t : Store} {r : Res} (h : port t r = 1) : t ≠ .reg := by
  rintro rfl
  cases r <;> simp [port] at h

/-- An opcode writes at most its units of resource `r` in the store `r` prices. -/
theorem outTo_le_use (t : Store) (r : Res) (h : port t r = 1) (o : Op m.sem) :
    ((((List.finRange o.nOut).filter fun i => o.outStore i = t).length : ℕ) : ℚ≥0) ≤ m.card.use o r := by
  have ht := port_ne_reg h
  cases o with
  | mma o sh sa sb => simp [Op.outStore, Ne.symm ht]
  | mmaSp o sh sa sb => simp [Op.outStore, Ne.symm ht]
  | lane o => simp [Op.outStore, Ne.symm ht]
  | move s t' n k =>
    by_cases e : t' = t
    · subst e
      rw [List.filter_eq_self.mpr (fun i _ => by simp [Op.outStore]), List.length_finRange]
      show ((k * n : ℕ) : ℚ≥0) ≤ slot (((max k 1 : ℕ) : ℚ≥0) / 32) r + (n : ℚ≥0) * port s r +
        ((k * n : ℕ) : ℚ≥0) * port t' r + if s = .reg ∧ t' = .reg ∧ r = .alu then ((k * n : ℕ) : ℚ≥0) else 0
      rw [h, mul_one]
      exact le_add_right (le_add_left le_rfl)
    · simp [Op.outStore, e]
  | atom s a =>
    by_cases e : s = t
    · subst e
      show ((((List.finRange 2).filter fun i : Fin 2 => (if i.val = 0 then s else .reg) = s).length : ℕ) : ℚ≥0) ≤
        slot (1 / 32) r + port s r + regWr s r
      rw [show List.finRange 2 = [0, 1] from rfl, h]
      simp [Ne.symm ht]
      exact le_add_right (le_add_left le_rfl)
    · show ((((List.finRange 2).filter fun i : Fin 2 => (if i.val = 0 then s else .reg) = t).length : ℕ) : ℚ≥0) ≤ _
      rw [show List.finRange 2 = [0, 1] from rfl]
      simp [e, Ne.symm ht]
  | red s t' n a =>
    by_cases e : t' = t
    · subst e
      rw [List.filter_eq_self.mpr (fun i _ => by simp [Op.outStore]), List.length_finRange]
      show (n : ℚ≥0) ≤ slot (1 / 32) r + (n : ℚ≥0) * (port s r + port t' r + regWr t' r)
      rw [h]
      calc (n : ℚ≥0) = n * 1 := (mul_one _).symm
        _ ≤ n * (port s r + 1 + regWr t' r) := by gcongr; exact le_add_right (le_add_left le_rfl)
        _ ≤ _ := le_add_left le_rfl
    · simp [Op.outStore, e]

theorem changed_step {Θ : Type} [DecidableEq Θ] (t : Store) (r : Res) (h : port t r = 1) (P : m.Prog Θ)
    (σ : m.State Θ) (θ : Θ) :
    ((changed t σ.mem (step P σ θ).1.mem).card : ℚ≥0) ≤ units (step P σ θ).2.toList r := by
  have ht := port_ne_reg h
  rcases hP : (P θ)[σ.pc θ]? with _ | (⟨o, src, dst, pred⟩ | ⟨p, tg, l⟩)
  · simp [step, hP, changed_self]
  · simp only [step, hP, Option.toList_some, units, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
      add_zero, Instr.use]
    exact (Nat.cast_le.mpr (changed_execOp t o src dst pred σ.mem)).trans (outTo_le_use t r h o)
  · simp only [step, hP]
    split_ifs
    · rcases l with _ | rr
      · simp [changed_self]
      · have := changed_update t σ.mem (m.loc .reg rr) (BitVec.ofNat 32 (σ.pc θ + 1))
        rw [ite_eq_right (show ¬ (m.loc Store.reg rr).fst = t from fun e => ht e.symm)] at this
        simp only [Option.elim_some]
        rw [Nat.le_zero.mp this]
        simp
    · simp [changed_self]

/-- **A run changes at most its units of resource `r` in the store `t` it prices** (`port t r = 1`: shared memory,
L2 or device memory), each word it writes there a unit. -/
theorem changed_run {Θ : Type} [DecidableEq Θ] (t : Store) (r : Res) (h : port t r = 1) (P : m.Prog Θ) :
    ∀ (σ : m.State Θ) (sched : List Θ),
      ((changed t σ.mem (run P σ sched).1.mem).card : ℚ≥0) ≤ units (run P σ sched).2 r
  | σ, [] => by simp [run, changed_self]
  | σ, θ :: rest => by
    simp only [run, units_append]
    have h1 := changed_step t r h P σ θ
    have h2 := changed_run t r h P (step P σ θ).1 rest
    calc ((changed t σ.mem (run P (step P σ θ).1 rest).1.mem).card : ℚ≥0)
        ≤ (changed t σ.mem (step P σ θ).1.mem).card +
            (changed t (step P σ θ).1.mem (run P (step P σ θ).1 rest).1.mem).card := by
          exact_mod_cast changed_trans t _ _ _
      _ ≤ _ := add_le_add h1 h2

end Machine

/-- **The card premise is false for a card that computes in no time.** If `can` says the card
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

/-- **The card premise is false for a card that writes faster than its table.** If
`can` says the card takes some memory `M` in `D` to an `f M` differing from `M` in more words of shared memory, L2 or
device memory than the store's resource carries in `T` seconds (`ρ_r T`, `port t r = 1`), `CardWithinAccel c can` is
false: a run changes at most its units of `r` in `t` (`Machine.changed_run`), and a run of `T` seconds uses at most
`ρ_r T` (G2). -/
theorem cardWithinAccel_fails_rate (c : Card) (can : (c.Mem → c.Mem) → ℚ≥0 → (c.Mem → Prop) → Prop) (t : Store)
    (r : Res) (h : port t r = 1) (f : c.Mem → c.Mem) (T : ℚ≥0) (D : c.Mem → Prop) (M : c.Mem) (hcan : can f T D)
    (hD : D M) (hf : c.ρ r * T < ((Finset.univ.filter fun q : Fin (c.size t) => f M ⟨t, q⟩ ≠ M ⟨t, q⟩).card : ℚ≥0)) :
    ¬ Verity.Assumptions.CardWithinAccel c can := by
  rintro ⟨x, _, _, hc⟩
  obtain ⟨P, sched, hrun⟩ := hc f T D hcan
  obtain ⟨hfits, hmem⟩ := hrun ⟨M, fun _ => 0⟩ hD (fun _ => rfl)
  have h1 := Machine.changed_run (m := ⟨c, x⟩) t r h P ⟨M, fun _ => 0⟩ sched
  rw [hmem] at h1
  exact absurd (h1.trans (Machine.units_le_of_fits P _ sched T hfits r)) (not_le.mpr hf)

end Verity.Accel
