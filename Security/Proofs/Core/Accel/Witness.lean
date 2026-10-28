import Definitions.Core.Accel.H200
import Proofs.Core.Accel.Model
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.NNRat
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith

/-!
# The honest lemma: the deployed INT8 GEMM meets the H200's model

The machine is the H200's table at the exact `s8` semantics (`hm`, `s8Open`). The kernel (`kernel n`) multiplies `n × n`
`s8` matrices, `n` a multiple of 256 and at most 2048: `A` sits in device memory row-major and `B` column-major, a byte
an entry, four to a word (`Placed`); `C` row-major, a word an entry. Three phases:

* **loads** (`load`): `2n` moves device memory → shared memory, each a row of `A` or a column of `B`;
* **MMAs** (`mma`): `n³/524288` `wgmma.m64n256k32.s32.s8.s8` with `A` and `B` from shared memory, slice by slice along
  `k`, each adding its 64 × 256 block's products of the slice to the block's accumulators in registers (zero at the
  first slice);
* **stores** (`store`): `n²/4` four-word moves registers → device memory.

Each instruction is the whole text of a thread of its own (`prog`), and the schedule steps the threads in order
(`sched`), so a run is the kernel's instructions in order (`run_plan`).

* `kernel_fits`: the run meets the model (`Machine.Fits`) in `secs n` seconds, each resource's need at its rate one
  after another (`n³` tensor units, the moves' and MMAs' issue slots, shared memory's and device memory's words, the
  code fetched at four words an instruction).
* `kernel_computes`: it leaves `C = A·B` exactly, each entry's sum a 32-bit word.
* `gemm_cardWithinAccel`: so the card premise holds at the honest instance (`gemmCan`, the card running this
  kernel), at the exact `s8` semantics, which meets (P1) and (P2).
-/

open scoped NNRat

namespace Verity.Accel.H200.Resident

open Machine (Operand)

/-- The H200's machine at the exact `s8` semantics. -/
abbrev hm : Machine := ⟨card, s8Open⟩

/-- An instruction of the kernel: an opcode, its input operands and its output operands. -/
abbrev Ins : Type := Op s8Open × (ℕ → Operand) × (ℕ → Operand)

/-- The instruction on the machine, unpredicated. -/
def Ins.instr (a : Ins) : hm.Instr := .op a.1 (fun i => a.2.1 i) (fun i => a.2.2 i) none

/-- The instruction's effect on the stores. -/
def Ins.exec (a : Ins) (M : hm.Mem) : hm.Mem := Machine.execOp a.1 (fun i => a.2.1 i) (fun i => a.2.2 i) none M

/-- A plan's program: instruction `θ` is the whole text of thread `θ`. -/
def prog (plan : List Ins) : hm.Prog ℕ := fun θ => (plan[θ]?.map Ins.instr).toList

/-- A plan's schedule: its threads in order. -/
def sched (plan : List Ins) : List ℕ := List.range plan.length

/-! ## A run is the plan -/

theorem run_prog (plan : List Ins) : ∀ (θs : List ℕ) (σ : hm.State ℕ), θs.Nodup → (∀ θ ∈ θs, σ.pc θ = 0) →
    (Machine.run (prog plan) σ θs).2 = θs.filterMap (fun θ => plan[θ]?.map Ins.instr) ∧
      (Machine.run (prog plan) σ θs).1.mem = (θs.filterMap fun θ => plan[θ]?).foldl (fun M a => a.exec M) σ.mem
  | [], σ, _, _ => ⟨rfl, rfl⟩
  | θ :: θs, σ, hnd, hpc => by
    obtain ⟨hθ, hnd'⟩ := List.nodup_cons.mp hnd
    have h0 : σ.pc θ = 0 := hpc θ (List.mem_cons_self ..)
    rcases ha : plan[θ]? with _ | a
    · have hs : Machine.step (prog plan) σ θ = (σ, none) := by
        simp [Machine.step, prog, ha]
      obtain ⟨ih1, ih2⟩ := run_prog plan θs σ hnd' (fun θ' h => hpc θ' (List.mem_cons_of_mem _ h))
      simp only [Machine.run, hs, ih1, ih2, List.filterMap_cons, ha, Option.map_none, Option.toList_none,
        List.nil_append]
      exact ⟨trivial, trivial⟩
    · have hs : Machine.step (prog plan) σ θ = (⟨a.exec σ.mem, Function.update σ.pc θ 1⟩, some a.instr) := by
        simp [Machine.step, prog, ha, h0, Ins.instr, Ins.exec]
      obtain ⟨ih1, ih2⟩ := run_prog plan θs ⟨a.exec σ.mem, Function.update σ.pc θ 1⟩ hnd'
        (fun θ' h => by
          show Function.update σ.pc θ 1 θ' = 0
          rw [Function.update_of_ne (fun e : θ' = θ => hθ (e ▸ h))]
          exact hpc θ' (List.mem_cons_of_mem _ h))
      simp only [Machine.run, hs, ih1, ih2, List.filterMap_cons, ha, Option.map_some, Option.toList_some,
        List.singleton_append, List.foldl_cons]
      exact ⟨trivial, trivial⟩

theorem filterMap_range_getElem {α β : Type} (g : α → β) : ∀ is : List α,
    (List.range is.length).filterMap (fun s => is[s]?.map g) = is.map g
  | [] => rfl
  | a :: is => by
    rw [List.length_cons, List.range_succ_eq_map, List.filterMap_cons, List.filterMap_map]
    simpa [Function.comp_def] using filterMap_range_getElem g is

/-- **A run of a plan is the plan**: its instructions, and its stores the plan's instructions applied in order. -/
theorem run_plan (plan : List Ins) (σ : hm.State ℕ) (hpc : ∀ θ, σ.pc θ = 0) :
    (Machine.run (prog plan) σ (sched plan)).2 = plan.map Ins.instr ∧
      (Machine.run (prog plan) σ (sched plan)).1.mem = plan.foldl (fun M a => a.exec M) σ.mem := by
  obtain ⟨h1, h2⟩ := run_prog plan (sched plan) σ List.nodup_range (fun θ _ => hpc θ)
  refine ⟨h1.trans (filterMap_range_getElem _ plan), h2.trans ?_⟩
  have : (List.range plan.length).filterMap (fun s => plan[s]?) = plan := by
    simpa using filterMap_range_getElem id plan
  rw [sched, this]

theorem code_prog (plan : List Ins) : Machine.code (prog plan) (sched plan) = plan.length := by
  unfold Machine.code sched
  rw [List.Nodup.dedup List.nodup_range,
    List.map_congr_left (g := fun _ => 1) (fun θ hθ => by
      simp [prog, List.getElem?_eq_getElem (List.mem_range.mp hθ)])]
  simp

theorem units_prog (plan : List Ins) (σ : hm.State ℕ) (hpc : ∀ θ, σ.pc θ = 0) (r : Res) :
    Machine.units (Machine.run (prog plan) σ (sched plan)).2 r = (plan.map fun a => card.use a.1 r).sum := by
  rw [(run_plan plan σ hpc).1]
  simp [Machine.units, Ins.instr, Machine.Instr.use, List.map_map, Function.comp_def]

/-! ## The kernel -/

/-- `len` consecutive words from store `s`, starting at word `a`, to store `t`, starting at word `b`. -/
def copyN (s t : Store) (len a b : ℕ) : Ins := (.move s t len 1, fun i => .direct (a + i), fun i => .direct (b + i))

/-- Word `w` of row `i` of `A`, in device memory and in shared memory. -/
def addrA (n i w : ℕ) : ℕ := i * (n / 4) + w

/-- Word `w` of column `j` of `B`, in device memory and in shared memory. -/
def addrB (n j w : ℕ) : ℕ := n * n / 4 + j * (n / 4) + w

/-- Entry `(i, j)` of `C` in device memory. -/
def dramC (n i j : ℕ) : ℕ := n * n / 2 + i * n + j

/-- Accumulator `(i, j)` of `C` in the registers. -/
def acc (n i j : ℕ) : ℕ := i * n + j

/-- Load `c`: row `c` of `A` for `c < n`, else column `c - n` of `B`, device memory to shared memory. -/
def load (n c : ℕ) : Ins :=
  if c < n then copyN .dram .smem (n / 4) (addrA n c 0) (addrA n c 0)
  else copyN .dram .smem (n / 4) (addrB n (c - n) 0) (addrB n (c - n) 0)

/-- The `wgmma`: `s8` by `s8` into 32-bit accumulators, `64 × 256 × 32`, both operands in shared memory. -/
def mmaOp : Op s8Open := .mma () ⟨64, 256, 32⟩ .smem .smem

/-- The blocks of 64 rows and 256 columns a slice has. -/
def blocks (n : ℕ) : ℕ := n / 64 * (n / 256)

/-- MMA `p`: slice `κ = p / blocks n`, block `(i₀, j₀)`. Its inputs are the block's 64 rows of `A`'s slice (8 words
each), its 256 columns of `B`'s slice (8 words each), and its accumulators (zero at `κ = 0`); its outputs, the
accumulators. -/
def mma (n p : ℕ) : Ins :=
  let κ := p / blocks n
  let e := p % blocks n
  let i₀ := 64 * (e / (n / 256))
  let j₀ := 256 * (e % (n / 256))
  (mmaOp,
    fun i =>
      if i < 512 then .direct (addrA n (i₀ + i / 8) (8 * κ + i % 8))
      else if i < 2560 then .direct (addrB n (j₀ + (i - 512) / 8) (8 * κ + (i - 512) % 8))
      else if κ = 0 then .imm 0
      else .direct (acc n (i₀ + (i - 2560) / 256) (j₀ + (i - 2560) % 256)),
    fun o => .direct (acc n (i₀ + o / 256) (j₀ + o % 256)))

/-- Store `w`: four accumulators of row `w / (n/4)` to `C`. -/
def store (n w : ℕ) : Ins :=
  copyN .reg .dram 4 (acc n (w / (n / 4)) (4 * (w % (n / 4)))) (dramC n (w / (n / 4)) (4 * (w % (n / 4))))

/-- The MMAs a kernel runs: `n/32` slices of `blocks n` each. -/
def mmas (n : ℕ) : ℕ := n / 32 * blocks n

/-- **The kernel**: the loads, then the MMAs, then the stores. -/
def kernel (n : ℕ) : List Ins :=
  (List.range (2 * n)).map (load n) ++ (List.range (mmas n)).map (mma n) ++ (List.range (n * n / 4)).map (store n)

/-- Byte `k` of a word, the lowest first. -/
def byteOf (w : Word) (k : ℕ) : BitVec 8 := BitVec.ofNat 8 (w.toNat / 2 ^ (8 * k))

/-- `A` and `B` in device memory: entry `(i, l)` of `A` is byte `l % 4` of word `l / 4` of row `i`, entry `(l, j)` of
`B` byte `l % 4` of word `l / 4` of column `j`. -/
def Placed (n : ℕ) (A B : Fin n → Fin n → BitVec 8) (M : hm.Mem) : Prop :=
  (∀ i l : Fin n, byteOf (M (hm.loc .dram (addrA n i (l / 4)))) (l % 4) = A i l) ∧
    ∀ l j : Fin n, byteOf (M (hm.loc .dram (addrB n j (l / 4)))) (l % 4) = B l j

/-! ## The kernel's units -/

/-- The kernel's issue slots: a move's `1/32`, an MMA's one. -/
def issueNeed (n : ℕ) : ℚ≥0 := ((2 * n + n * n / 4 : ℕ) : ℚ≥0) / 32 + mmas n

/-- Shared memory's bank words: the loads write `A` and `B`, and each MMA reads 2,560. -/
def smemNeed (n : ℕ) : ℚ≥0 := (2 * n * (n / 4) + mmas n * 2560 : ℕ)

/-- Device memory's words: the loads read `A` and `B`, the stores write `C`, and the code is fetched at four words an
instruction. -/
def dramNeed (n : ℕ) : ℚ≥0 := (2 * n * (n / 4) + n * n / 4 * 4 + 4 * (2 * n + mmas n + n * n / 4) : ℕ)

/-- **The kernel's seconds** on the H200: each resource's need at its rate, one after another. -/
def secs (n : ℕ) : ℚ≥0 :=
  (n : ℚ≥0) ^ 3 / card.ρ .tensor + issueNeed n / card.ρ .issue + smemNeed n / card.ρ .smem +
    dramNeed n / card.ρ .dram

theorem blocks_mul (n : ℕ) (h : 256 ∣ n) : mmas n * 524288 = n ^ 3 := by
  obtain ⟨q, rfl⟩ := h
  simp only [mmas, blocks]
  rw [show 256 * q / 32 = 8 * q by omega, show 256 * q / 64 = 4 * q by omega, show 256 * q / 256 = q by omega]
  ring

theorem kernel_length (n : ℕ) : (kernel n).length = 2 * n + mmas n + n * n / 4 := by
  simp [kernel, Nat.add_assoc]

theorem load_op (n c : ℕ) : (load n c).1 = .move .dram .smem (n / 4) 1 := by
  unfold load; split <;> rfl

theorem kernel_use (n : ℕ) (r : Res) : ((kernel n).map fun a => card.use a.1 r).sum =
    ((2 * n : ℕ) : ℚ≥0) * card.use (x := s8Open) (.move .dram .smem (n / 4) 1) r +
      (mmas n : ℚ≥0) * card.use mmaOp r + ((n * n / 4 : ℕ) : ℚ≥0) * card.use (x := s8Open) (.move .reg .dram 4 1) r := by
  simp only [kernel, List.map_append, List.sum_append, List.map_map, Function.comp_def, load_op]
  simp [mma, store, copyN, List.map_const', List.sum_replicate, nsmul_eq_mul]

theorem rho_pos (r : Res) : 0 < card.ρ r := by
  cases r <;> norm_num [card, clock]

theorem le_rho_mul {ρ q T : ℚ≥0} (hρ : 0 < ρ) (h : q / ρ ≤ T) : q ≤ ρ * T := by
  rw [mul_comm]; exact (div_le_iff₀ hρ).mp h

/-- **The kernel meets the model** in `secs n` seconds: its code fits (G1), and each resource's units, with the code
fetched, are within its rate over the run (G2). -/
theorem kernel_fits (n : ℕ) (h256 : 256 ∣ n) (hn : n ≤ 2048) (σ : hm.State ℕ) (hpc : ∀ θ, σ.pc θ = 0) :
    Machine.Fits (prog (kernel n)) σ (sched (kernel n)) (secs n) := by
  have hcode := code_prog (kernel n)
  have hlen := kernel_length n
  refine ⟨?_, fun r => ?_⟩
  · rw [hcode, hlen]
    show _ ≤ 141 * 2 ^ 30 / 16
    obtain ⟨q, rfl⟩ := h256
    have hq : q ≤ 8 := by omega
    interval_cases q <;> norm_num [mmas, blocks]
  · rw [units_prog _ σ hpc, kernel_use, hcode, hlen]
    have hwa : (⟨64, 256, 32⟩ : Shape).wa (s8Open.fa ()) = 512 := rfl
    have hwb : (⟨64, 256, 32⟩ : Shape).wb (s8Open.fb ()) = 2048 := rfl
    have hw : card.weight (s8Open.fa ()) = 1 := rfl
    cases r <;> simp only [mmaOp, Card.use, slot, port, hwa, hwb, hw, reduceCtorEq, false_and, and_false,
      ite_true, ite_false, mul_zero, mul_one, zero_add, add_zero, Nat.cast_ofNat, zero_le, max_self]
    · rw [show ((64 * 256 * max 32 1 : ℕ) : ℚ≥0) = 524288 by norm_num]
      have h3 : (mmas n : ℚ≥0) * 524288 = (n : ℚ≥0) ^ 3 := by exact_mod_cast blocks_mul n h256
      rw [h3]
      refine le_rho_mul (rho_pos _) ?_
      exact le_add_right (le_add_right (le_add_right le_rfl))
    · refine le_rho_mul (rho_pos _) (le_add_right (le_add_right (le_add_left (le_of_eq ?_))))
      simp only [issueNeed]
      push_cast
      ring
    · refine le_rho_mul (rho_pos _) (le_add_right (le_add_left (le_of_eq ?_)))
      simp only [smemNeed]
      push_cast
      ring
    · refine le_rho_mul (rho_pos _) (le_add_left (le_of_eq ?_))
      simp only [dramNeed]
      push_cast
      ring

/-! ## Writes -/

section Writes

variable {α β γ : Type} [DecidableEq β]

theorem foldl_update_not_mem (f : α → β) (g : α → γ) (b : β) :
    ∀ (l : List α) (M : β → γ), (∀ a ∈ l, f a ≠ b) →
      l.foldl (fun M' a => Function.update M' (f a) (g a)) M b = M b
  | [], _, _ => rfl
  | a :: l, M, h => by
    rw [List.foldl_cons, foldl_update_not_mem f g b l _ (fun a' ha' => h a' (by simp [ha'])),
      Function.update_of_ne (h a (by simp)).symm]

theorem foldl_update_mem (f : α → β) (g : α → γ) :
    ∀ (l : List α) (M : β → γ), (∀ a ∈ l, ∀ a' ∈ l, f a = f a' → a = a') → l.Nodup → ∀ a ∈ l,
      l.foldl (fun M' a => Function.update M' (f a) (g a)) M (f a) = g a
  | [], _, _, _, _, h => absurd h (by simp)
  | a₀ :: l, M, hinj, hnd, a, ha => by
    rw [List.foldl_cons]
    have hnd' := List.nodup_cons.mp hnd
    rcases List.mem_cons.mp ha with rfl | ha'
    · rw [foldl_update_not_mem f g (f a) l _ (fun a' ha' e =>
        hnd'.1 (hinj a' (by simp [ha']) a (by simp) e ▸ ha')), Function.update_self]
    · exact foldl_update_mem f g l _ (fun a ha a' ha' => hinj a (by simp [ha]) a' (by simp [ha'])) hnd'.2 a ha'

theorem execOp_direct {m : Machine} (o : Op m.sem) (src : Fin o.nIn → Operand) (d : Fin o.nOut → ℕ) (M : m.Mem) :
    Machine.execOp o src (fun i => .direct (d i)) none M =
      (List.finRange o.nOut).foldl (fun M' i => Function.update M' (m.loc (o.outStore i) (d i))
        (o.eval (fun i => (src i).readIn (o.inStore i) M) i)) M := by
  simp [Machine.execOp, Machine.Operand.writeIn, Machine.Operand.locIn]

theorem loc_eq_iff {m : Machine} (s s' : Store) (q q' : ℕ) :
    m.loc s q = m.loc s' q' ↔ s = s' ∧ q % m.card.size s = q' % m.card.size s' := by
  constructor
  · intro h
    obtain rfl : s = s' := congrArg Sigma.fst h
    exact ⟨rfl, congrArg Fin.val (eq_of_heq (Sigma.mk.inj h).2)⟩
  · rintro ⟨rfl, h⟩
    simp only [Machine.loc, h]

end Writes

/-! ## Copies -/

theorem copy_exec (s t : Store) (len a b : ℕ) (M : hm.Mem) :
    (copyN s t len a b).exec M =
      (List.finRange (1 * len)).foldl (fun M' (i : Fin (1 * len)) => Function.update M'
        (hm.loc t (b + i.val)) (M (hm.loc s (a + i.val)))) M := by
  simp only [Ins.exec, copyN]
  refine (execOp_direct (m := hm) (.move s t len 1) _ (fun i => b + i.val) M).trans ?_
  congr 1
  funext M' i
  have hi : (i : ℕ) < len := by have := i.isLt; simp only [Op.nOut] at this; omega
  simp [Op.eval, getD, Op.outStore, Op.inStore, Op.nIn, Machine.Operand.readIn, Machine.Operand.locIn,
    Nat.mod_eq_of_lt hi, hi]

theorem copy_at (s t : Store) (len a b : ℕ) (M : hm.Mem) (u : Store) (q : ℕ) (hq : q < hm.card.size u)
    (hb : b + len ≤ hm.card.size t) :
    (copyN s t len a b).exec M (hm.loc u q) =
      if u = t ∧ b ≤ q ∧ q < b + len then M (hm.loc s (a + (q - b))) else M (hm.loc u q) := by
  rw [copy_exec]
  split_ifs with hin
  · obtain ⟨rfl, h1, h2⟩ := hin
    have := foldl_update_mem (fun i : Fin (1 * len) => hm.loc u (b + i.val))
      (fun i => M (hm.loc s (a + i.val))) (List.finRange (1 * len)) M
      (fun i _ j _ e => by
        have e2 := ((loc_eq_iff (m := hm) u u _ _).mp e).2
        have hi := i.isLt
        have hj := j.isLt
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)] at e2
        exact Fin.ext (by omega))
      (List.nodup_finRange _) ⟨q - b, by omega⟩ (List.mem_finRange _)
    simp only at this
    rwa [show b + (q - b) = q by omega] at this
  · apply foldl_update_not_mem
    intro i _ e
    obtain ⟨rfl, e2⟩ := (loc_eq_iff (m := hm) t u _ _).mp e
    have hi := i.isLt
    rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt hq] at e2
    exact hin ⟨rfl, by omega, by omega⟩

/-! ## The loads -/

theorem load_eq (n c : ℕ) (h4 : 4 ∣ n) : load n c = copyN .dram .smem (n / 4) (c * (n / 4)) (c * (n / 4)) := by
  unfold load
  split
  · simp [addrA]
  · rename_i hc
    obtain ⟨k, rfl⟩ := h4
    obtain ⟨d, rfl⟩ : ∃ d, c = 4 * k + d := ⟨c - 4 * k, by omega⟩
    have e1 : 4 * k / 4 = k := by omega
    have e2 : 4 * k * (4 * k) / 4 = 4 * k * k := by
      rw [show 4 * k * (4 * k) = 4 * (4 * k * k) by ring]
      exact Nat.mul_div_cancel_left _ (by norm_num)
    simp only [addrB, e1, e2, Nat.add_sub_cancel_left]
    congr 1 <;> ring

theorem loads_exec (n : ℕ) (h4 : 4 ∣ n) (hsz : 2 * n * (n / 4) ≤ hm.card.size .smem)
    (hsd : hm.card.size .smem ≤ hm.card.size .dram) (M : hm.Mem) :
    ∀ c ≤ 2 * n, ∀ (u : Store) (q : ℕ), q < hm.card.size u →
      ((List.range c).map (load n)).foldl (fun M a => a.exec M) M (hm.loc u q) =
        if u = .smem ∧ q < c * (n / 4) then M (hm.loc .dram q) else M (hm.loc u q)
  | 0, _, u, q, _ => by simp
  | c + 1, hc, u, q, hq => by
    have hb : c * (n / 4) + n / 4 ≤ hm.card.size .smem := by
      have := Nat.mul_le_mul_right (n / 4) hc
      rw [add_mul, one_mul] at this
      omega
    rw [List.range_succ, List.map_append, List.foldl_append, List.map_cons, List.map_nil, List.foldl_cons,
      List.foldl_nil, load_eq n c h4, copy_at .dram .smem (n / 4) (c * (n / 4)) (c * (n / 4)) _ u q hq hb]
    have ih := loads_exec n h4 hsz hsd M c (by omega)
    by_cases hu : u = .smem
    · subst hu
      by_cases h1 : c * (n / 4) ≤ q ∧ q < c * (n / 4) + n / 4
      · rw [ite_eq_left ⟨rfl, h1⟩, ih .dram _ (by omega), ite_eq_right (by simp),
          ite_eq_left ⟨rfl, by rw [add_mul, one_mul]; omega⟩, show c * (n / 4) + (q - c * (n / 4)) = q by omega]
      · rw [ite_eq_right (fun h => h1 h.2), ih .smem q hq]
        by_cases h2 : q < c * (n / 4)
        · rw [ite_eq_left ⟨rfl, h2⟩, ite_eq_left ⟨rfl, by rw [add_mul, one_mul]; omega⟩]
        · rw [ite_eq_right (fun h => h2 h.2), ite_eq_right (fun h => by rw [add_mul, one_mul] at h; omega)]
    · rw [ite_eq_right (fun h => hu h.1), ih u q hq, ite_eq_right (fun h => hu h.1), ite_eq_right (fun h => hu h.1)]

/-! ## The stores -/

theorem store_eq (n w : ℕ) (h4 : 4 ∣ n) : store n w = copyN .reg .dram 4 (4 * w) (n * n / 2 + 4 * w) := by
  obtain ⟨k, rfl⟩ := h4
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · simp [store, acc, dramC]
  have e1 : 4 * k / 4 = k := by omega
  have e2 : w / k * (4 * k) + 4 * (w % k) = 4 * w := by
    have := Nat.div_add_mod w k
    rw [show w / k * (4 * k) = 4 * (k * (w / k)) by ring]
    omega
  simp only [store, acc, dramC, e1, e2, Nat.add_assoc]

theorem stores_exec (n : ℕ) (h4 : 4 ∣ n) (hsd : n * n / 2 + n * n ≤ hm.card.size .dram)
    (hsr : n * n ≤ hm.card.size .reg) (M : hm.Mem) :
    ∀ m ≤ n * n / 4, ∀ (u : Store) (q : ℕ), q < hm.card.size u →
      ((List.range m).map (store n)).foldl (fun M a => a.exec M) M (hm.loc u q) =
        if u = .dram ∧ n * n / 2 ≤ q ∧ q < n * n / 2 + 4 * m then M (hm.loc .reg (q - n * n / 2))
        else M (hm.loc u q)
  | 0, _, u, q, _ => by simp
  | m + 1, hm', u, q, hq => by
    have hb : n * n / 2 + 4 * m + 4 ≤ hm.card.size .dram := by omega
    rw [List.range_succ, List.map_append, List.foldl_append, List.map_cons, List.map_nil, List.foldl_cons,
      List.foldl_nil, store_eq n m h4, copy_at .reg .dram 4 (4 * m) (n * n / 2 + 4 * m) _ u q hq hb]
    have ih := stores_exec n h4 hsd hsr M m (by omega)
    by_cases hu : u = .dram
    · subst hu
      by_cases h1 : n * n / 2 + 4 * m ≤ q ∧ q < n * n / 2 + 4 * m + 4
      · rw [ite_eq_left ⟨rfl, h1⟩, ih .reg _ (by omega), ite_eq_right (by simp), ite_eq_left ⟨rfl, by omega, by omega⟩,
          show 4 * m + (q - (n * n / 2 + 4 * m)) = q - n * n / 2 by omega]
      · rw [ite_eq_right (fun h => h1 h.2), ih .dram q hq]
        by_cases h2 : n * n / 2 ≤ q ∧ q < n * n / 2 + 4 * m
        · rw [ite_eq_left ⟨rfl, h2⟩, ite_eq_left ⟨rfl, h2.1, by omega⟩]
        · rw [ite_eq_right (fun h => h2 h.2), ite_eq_right (fun h => by omega)]
    · rw [ite_eq_right (fun h => hu h.1), ih u q hq, ite_eq_right (fun h => hu h.1), ite_eq_right (fun h => hu h.1)]

/-! ## The MMAs -/

/-- MMA `p`'s output `o`: its accumulator's index. -/
def accOut (n p o : ℕ) : ℕ :=
  acc n (64 * (p % blocks n / (n / 256)) + o / 256) (256 * (p % blocks n % (n / 256)) + o % 256)

theorem mma_exec (n p : ℕ) (M : hm.Mem) :
    (mma n p).exec M =
      (List.finRange (mma n p).1.nOut).foldl (fun M' o => Function.update M'
        (hm.loc .reg (accOut n p o.val))
        ((mma n p).1.eval (fun i => ((mma n p).2.1 i.val).readIn ((mma n p).1.inStore i) M) o)) M := by
  simp only [Ins.exec]
  exact execOp_direct (m := hm) _ _ (fun o => accOut n p o.val) M

theorem hwa : (⟨64, 256, 32⟩ : Shape).wa .s8 = 512 := by decide

theorem hwb : (⟨64, 256, 32⟩ : Shape).wb .s8 = 2048 := by decide

theorem mma_eval (n p : ℕ) (v : Fin (mma n p).1.nIn → Word) (o : Fin (mma n p).1.nOut) :
    (mma n p).1.eval v o =
      BitVec.ofInt 32 ((getD v (2560 + o.val)).toInt +
        (List.zipWith (fun a b => Fmt.s8.intVal a * Fmt.s8.intVal b) (rowA .s8 ⟨64, 256, 32⟩ (getD v) (o.val / 256))
          (colB .s8 ⟨64, 256, 32⟩ 512 (getD v) (o.val % 256))).sum) := by
  show mmaEval (intDot .s8 .s8) .s8 .s8 false ⟨64, 256, 32⟩ (getD v) o.val = _
  simp only [mmaEval, accWords, accC, Bool.false_eq_true, ↓reduceIte, Nat.div_one, Nat.mod_one, mul_zero, pow_zero,
    intDot, BitVec.ofNat_toNat, BitVec.setWidth_eq, hwa, hwb]

/-- MMA `p`'s input words in `M`, as its opcode reads them. -/
def mmaV (n p : ℕ) (M : hm.Mem) : ℕ → Word :=
  getD (fun i : Fin (mma n p).1.nIn => ((mma n p).2.1 i.val).readIn ((mma n p).1.inStore i) M)

theorem mma_in (n p : ℕ) (M : hm.Mem) (i : ℕ) (hi : i < 18944) :
    mmaV n p M i =
      if i < 512 then
        M (hm.loc .smem (addrA n (64 * (p % blocks n / (n / 256)) + i / 8) (8 * (p / blocks n) + i % 8)))
      else if i < 2560 then
        M (hm.loc .smem (addrB n (256 * (p % blocks n % (n / 256)) + (i - 512) / 8) (8 * (p / blocks n) + (i - 512) % 8)))
      else if p / blocks n = 0 then 0
      else M (hm.loc .reg
        (acc n (64 * (p % blocks n / (n / 256)) + (i - 2560) / 256) (256 * (p % blocks n % (n / 256)) + (i - 2560) % 256))) := by
  have hn : (mma n p).1.nIn = 18944 := rfl
  simp only [mmaV, getD, show i < (mma n p).1.nIn by rw [hn]; exact hi, ↓reduceDIte]
  have ha : (⟨64, 256, 32⟩ : Shape).wa (s8Open.fa ()) = 512 := rfl
  have hb : (⟨64, 256, 32⟩ : Shape).wb (s8Open.fb ()) = 2048 := rfl
  simp only [mma, mmaOp, Op.inStore, ha, hb, Machine.Operand.readIn, Machine.Operand.locIn]
  by_cases h1 : i < 512
  · simp [h1]
  · by_cases h2 : i < 2560
    · simp [h1, h2]
    · by_cases h3 : p / blocks n = 0 <;> simp [h1, h2, h3]

/-! ## Bytes -/

theorem byte_toInt (w : Word) (k : ℕ) : Fmt.s8.intVal (w.toNat / 2 ^ (8 * k) % 2 ^ 8) = (byteOf w k).toInt := by
  rw [byteOf, BitVec.toInt_eq_toNat_cond, BitVec.toNat_ofNat]
  simp only [Fmt.intVal]
  have h : w.toNat / 2 ^ (8 * k) % 2 ^ 8 < 256 := Nat.mod_lt _ (by norm_num)
  split_ifs <;> push_cast <;> omega

/-- Entry `(i, l)` of `A` as MMA operands read it from shared memory in `M`. -/
def aOf (n : ℕ) (M : hm.Mem) (i l : ℕ) : ℤ := (byteOf (M (hm.loc .smem (addrA n i (l / 4)))) (l % 4)).toInt

/-- Entry `(l, j)` of `B` as MMA operands read it from shared memory in `M`. -/
def bOf (n : ℕ) (M : hm.Mem) (l j : ℕ) : ℤ := (byteOf (M (hm.loc .smem (addrB n j (l / 4)))) (l % 4)).toInt

theorem zipWith_ofFn_sum (f : ℕ → ℕ → ℤ) (F G : ℕ → ℕ) : ∀ k : ℕ,
    (List.zipWith f (List.ofFn fun l : Fin k => F l) (List.ofFn fun l : Fin k => G l)).sum =
      ∑ l ∈ Finset.range k, f (F l) (G l)
  | 0 => by simp
  | k + 1 => by
    simp only [List.ofFn_succ', Fin.val_castSucc, Fin.val_last, List.concat_eq_append]
    rw [List.zipWith_append (by simp), List.sum_append, zipWith_ofFn_sum f F G k, Finset.sum_range_succ]
    simp

/-- MMA `p`'s products for output `o = 256 r + s`: `A`'s row `i₀ + r` against `B`'s column `j₀ + s` over slice `κ`. -/
theorem mma_sum (n p : ℕ) (M : hm.Mem) (r s : ℕ) (hr : r < 64) (hs : s < 256) :
    (List.zipWith (fun x y => Fmt.s8.intVal x * Fmt.s8.intVal y)
      (rowA .s8 ⟨64, 256, 32⟩ (mmaV n p M) r) (colB .s8 ⟨64, 256, 32⟩ 512 (mmaV n p M) s)).sum =
      ∑ l ∈ Finset.range 32, aOf n M (64 * (p % blocks n / (n / 256)) + r) (32 * (p / blocks n) + l) *
        bOf n M (32 * (p / blocks n) + l) (256 * (p % blocks n % (n / 256)) + s) := by
  simp only [rowA, colB]
  rw [zipWith_ofFn_sum (fun x y => Fmt.s8.intVal x * Fmt.s8.intVal y)
    (fun l => elem Fmt.s8.bits (mmaV n p M) (r * Fmt.s8.rowLen 32 + l))
    (fun l => elem Fmt.s8.bits (fun w => mmaV n p M (512 + w)) (s * Fmt.s8.rowLen 32 + l)) (Fmt.s8.rowLen 32)]
  simp only [show Fmt.s8.rowLen 32 = 32 from rfl, show Fmt.s8.bits = 8 from rfl]
  refine Finset.sum_congr rfl fun l hl => ?_
  have hl : l < 32 := Finset.mem_range.mp hl
  simp only [elem, show ¬((8 : ℕ) = 64) by decide, ↓reduceIte]
  rw [show (r * 32 + l) * 8 / 32 = 8 * r + l / 4 by omega, show (r * 32 + l) * 8 % 32 = 8 * (l % 4) by omega,
    show (s * 32 + l) * 8 / 32 = 8 * s + l / 4 by omega, show (s * 32 + l) * 8 % 32 = 8 * (l % 4) by omega,
    mma_in n p M _ (by omega), mma_in n p M _ (by omega), byte_toInt, byte_toInt]
  simp only [show 8 * r + l / 4 < 512 by omega, show ¬ 512 + (8 * s + l / 4) < 512 by omega,
    show 512 + (8 * s + l / 4) < 2560 by omega, ite_true, ite_false, aOf, bOf]
  rw [show (8 * r + l / 4) / 8 = r by omega, show (8 * r + l / 4) % 8 = l / 4 by omega,
    show (512 + (8 * s + l / 4) - 512) / 8 = s by omega, show (512 + (8 * s + l / 4) - 512) % 8 = l / 4 by omega,
    show (32 * (p / blocks n) + l) / 4 = 8 * (p / blocks n) + l / 4 by omega,
    show (32 * (p / blocks n) + l) % 4 = l % 4 by omega]

/-! ## One MMA's effect -/

theorem divmod_unique {a b c d Q : ℕ} (hb : b < Q) (hd : d < Q) (h : a * Q + b = c * Q + d) :
    a = c ∧ b = d := by
  have hQ : 0 < Q := by omega
  have h1 : (a * Q + b) / Q = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hQ, Nat.div_eq_of_lt hb, zero_add]
  have h2 : (c * Q + d) / Q = c := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hQ, Nat.div_eq_of_lt hd, zero_add]
  have h3 : (a * Q + b) % Q = b := by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]
  have h4 : (c * Q + d) % Q = d := by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd]
  exact ⟨by rw [← h1, ← h2, h], by rw [← h3, ← h4, h]⟩

theorem acc_lt (n i j : ℕ) (hi : i < n) (hj : j < n) : acc n i j < n * n := by
  simp only [acc]
  nlinarith

theorem sq_le_reg (n : ℕ) (hn : n ≤ 2048) : n * n ≤ hm.card.size .reg := by
  show _ ≤ 132 * (65536 + 4494)
  nlinarith

/-- The block `(i/64, j/256)` of entry `(i, j)`, numbered as the MMAs number them. -/
def blk (n i j : ℕ) : ℕ := i / 64 * (n / 256) + j / 256

theorem blocks_eq (Q : ℕ) : blocks (256 * Q) = 4 * Q * Q := by
  simp only [blocks]
  rw [show 256 * Q / 64 = 4 * Q by omega, show 256 * Q / 256 = Q by omega]

theorem blk_lt (n i j : ℕ) (h256 : 256 ∣ n) (hi : i < n) (hj : j < n) : blk n i j < blocks n := by
  obtain ⟨Q, rfl⟩ := h256
  rw [blocks_eq]
  simp only [blk, show 256 * Q / 256 = Q by omega]
  have h1 : i / 64 < 4 * Q := by omega
  have h2 : j / 256 < Q := by omega
  nlinarith

/-- Entry `(i, j)` is in block `e` exactly when its block coordinates are `e`'s. -/
theorem blk_iff (n i j e : ℕ) (h256 : 256 ∣ n) (hj : j < n) :
    blk n i j = e ↔ i / 64 = e / (n / 256) ∧ j / 256 = e % (n / 256) := by
  obtain ⟨Q, rfl⟩ := h256
  simp only [blk, show 256 * Q / 256 = Q by omega]
  have hQ : 0 < Q := by omega
  constructor
  · rintro rfl
    have hj' : j / 256 < Q := by omega
    exact ⟨by rw [Nat.add_comm, Nat.add_mul_div_right _ _ hQ, Nat.div_eq_of_lt hj', zero_add],
      by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj']⟩
  · rintro ⟨h1, h2⟩
    rw [h1, h2, Nat.div_add_mod']

theorem blk_geom (n e : ℕ) (h256 : 256 ∣ n) (he : e < blocks n) :
    64 * (e / (n / 256)) + 64 ≤ n ∧ 256 * (e % (n / 256)) + 256 ≤ n := by
  obtain ⟨Q, rfl⟩ := h256
  rw [blocks_eq] at he
  rw [show 256 * Q / 256 = Q by omega]
  have hQ : 0 < Q := by
    rcases Nat.eq_zero_or_pos Q with rfl | h
    · simp at he
    · exact h
  have h1 : e / Q < 4 * Q := (Nat.div_lt_iff_lt_mul hQ).mpr (by nlinarith)
  have h2 : e % Q < Q := Nat.mod_lt _ hQ
  omega

theorem blocks_pos (n : ℕ) (h256 : 256 ∣ n) (hn : 0 < n) : 0 < blocks n := by
  obtain ⟨Q, rfl⟩ := h256
  rw [blocks_eq]
  have : 0 < Q := by omega
  positivity

/-- An MMA writes no store but the registers. -/
theorem mma_other (n p : ℕ) (M : hm.Mem) (u : Store) (hu : u ≠ .reg) (q : ℕ) :
    (mma n p).exec M (hm.loc u q) = M (hm.loc u q) := by
  rw [mma_exec]
  exact foldl_update_not_mem _ _ _ _ _ fun o _ h => hu ((loc_eq_iff (m := hm) _ _ _ _).mp h).1.symm

/-- **One MMA on the accumulators**: MMA `p` (slice `κ = p / blocks n`, block `e = p % blocks n`) adds its slice's
products to each accumulator of block `e`, from zero at `κ = 0`, and leaves every other accumulator. -/
theorem mma_reg (n p : ℕ) (h256 : 256 ∣ n) (hn0 : 0 < n) (hn : n ≤ 2048) (M : hm.Mem) (i j : ℕ) (hi : i < n)
    (hj : j < n) :
    (mma n p).exec M (hm.loc .reg (acc n i j)) =
      if blk n i j = p % blocks n then
        BitVec.ofInt 32 ((if p / blocks n = 0 then 0 else (M (hm.loc .reg (acc n i j))).toInt) +
          ∑ l ∈ Finset.range 32, aOf n M i (32 * (p / blocks n) + l) * bOf n M (32 * (p / blocks n) + l) j)
      else M (hm.loc .reg (acc n i j)) := by
  have he : p % blocks n < blocks n := Nat.mod_lt _ (blocks_pos n h256 hn0)
  obtain ⟨gi, gj⟩ := blk_geom n (p % blocks n) h256 he
  set i₀ := 64 * (p % blocks n / (n / 256)) with hi₀
  set j₀ := 256 * (p % blocks n % (n / 256)) with hj₀
  have hsz := sq_le_reg n hn
  have hout : ∀ o : ℕ, o < 16384 → accOut n p o < n * n := fun o ho => acc_lt n _ _ (by omega) (by omega)
  have hacc := acc_lt n i j hi hj
  have hno : (mma n p).1.nOut = 16384 := rfl
  rw [mma_exec]
  by_cases hb : blk n i j = p % blocks n
  · rw [ite_eq_left hb]
    obtain ⟨h1, h2⟩ := (blk_iff n i j _ h256 hj).mp hb
    have ho : (i % 64) * 256 + j % 256 < (mma n p).1.nOut := by rw [hno]; omega
    have hf : accOut n p ((i % 64) * 256 + j % 256) = acc n i j := by
      simp only [accOut, acc]
      rw [← hi₀, ← hj₀, show (i % 64 * 256 + j % 256) / 256 = i % 64 by omega,
        show (i % 64 * 256 + j % 256) % 256 = j % 256 by omega, show i₀ + i % 64 = i by omega,
        show j₀ + j % 256 = j by omega]
    have key := foldl_update_mem (fun o : Fin (mma n p).1.nOut => hm.loc .reg (accOut n p o.val))
      (fun o => (mma n p).1.eval (fun i => ((mma n p).2.1 i.val).readIn ((mma n p).1.inStore i) M) o)
      (List.finRange _) M
      (fun a _ a' _ e => by
        have e2 := ((loc_eq_iff (m := hm) _ _ _ _).mp e).2
        have ha : a.val < 16384 := hno ▸ a.isLt
        have ha' : a'.val < 16384 := hno ▸ a'.isLt
        rw [Nat.mod_eq_of_lt ((hout _ ha).trans_le hsz), Nat.mod_eq_of_lt ((hout _ ha').trans_le hsz)] at e2
        simp only [accOut, acc] at e2
        obtain ⟨e3, e4⟩ := divmod_unique (by omega) (by omega) e2
        exact Fin.ext (by omega))
      (List.nodup_finRange _) ⟨_, ho⟩ (List.mem_finRange _)
    simp only [hf] at key
    rw [key, mma_eval]
    change BitVec.ofInt 32 ((mmaV n p M (2560 + _)).toInt +
      (List.zipWith _ (rowA .s8 ⟨64, 256, 32⟩ (mmaV n p M) _) (colB .s8 ⟨64, 256, 32⟩ 512 (mmaV n p M) _)).sum) = _
    rw [mma_sum n p M _ _ (by omega) (by omega), mma_in n p M _ (by omega)]
    rw [show (i % 64 * 256 + j % 256) / 256 = i % 64 by omega,
      show (i % 64 * 256 + j % 256) % 256 = j % 256 by omega, ← hi₀, ← hj₀, show i₀ + i % 64 = i by omega,
      show j₀ + j % 256 = j by omega, ite_eq_right (by omega : ¬ 2560 + (i % 64 * 256 + j % 256) < 512),
      ite_eq_right (by omega : ¬ 2560 + (i % 64 * 256 + j % 256) < 2560),
      show (2560 + (i % 64 * 256 + j % 256) - 2560) / 256 = i % 64 by omega,
      show (2560 + (i % 64 * 256 + j % 256) - 2560) % 256 = j % 256 by omega, show i₀ + i % 64 = i by omega,
      show j₀ + j % 256 = j by omega]
    split_ifs <;> simp
  · rw [ite_eq_right hb]
    apply foldl_update_not_mem
    intro o _ e
    have e2 := ((loc_eq_iff (m := hm) _ _ _ _).mp e).2
    have ho : o.val < 16384 := hno ▸ o.isLt
    rw [Nat.mod_eq_of_lt ((hout _ ho).trans_le hsz), Nat.mod_eq_of_lt (hacc.trans_le hsz)] at e2
    simp only [accOut, acc] at e2
    obtain ⟨e3, e4⟩ := divmod_unique (by omega) (by omega) e2
    exact hb ((blk_iff n i j _ h256 hj).mpr ⟨by omega, by omega⟩)

/-! ## The MMAs' invariant -/

/-- After the MMAs of the slices before `κ` and of the first `e` blocks of slice `κ` from `M₀`: every store but the
registers is `M₀`'s, and accumulator `(i, j)` holds the sum over the slices its block has had (anything before its
first). -/
def MmaInv (n : ℕ) (M₀ M : hm.Mem) (κ e : ℕ) : Prop :=
  (∀ u, u ≠ .reg → ∀ q, M (hm.loc u q) = M₀ (hm.loc u q)) ∧
  ∀ i j, i < n → j < n → (blk n i j < e ∨ 1 ≤ κ) →
    M (hm.loc .reg (acc n i j)) =
      BitVec.ofInt 32 (∑ l ∈ Finset.range (32 * (κ + if blk n i j < e then 1 else 0)), aOf n M₀ i l * bOf n M₀ l j)

theorem ofInt_toInt_add (S T : ℤ) :
    BitVec.ofInt 32 ((BitVec.ofInt 32 S).toInt + T) = BitVec.ofInt 32 (S + T) := by
  rw [BitVec.ofInt_add, BitVec.ofInt_toInt, ← BitVec.ofInt_add]

theorem mma_step (n κ e : ℕ) (h256 : 256 ∣ n) (hn0 : 0 < n) (hn : n ≤ 2048) (he : e < blocks n)
    (M₀ M : hm.Mem) (h : MmaInv n M₀ M κ e) :
    MmaInv n M₀ ((mma n (κ * blocks n + e)).exec M) κ (e + 1) := by
  obtain ⟨hother, hreg⟩ := h
  have hB := blocks_pos n h256 hn0
  have hdiv : (κ * blocks n + e) / blocks n = κ := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt he, zero_add]
  have hmod : (κ * blocks n + e) % blocks n = e := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt he]
  have hA : aOf n M = aOf n M₀ := by
    funext i l
    simp only [aOf, hother .smem (show (Store.smem : Store) ≠ .reg by decide)]
  have hB' : bOf n M = bOf n M₀ := by
    funext l j
    simp only [bOf, hother .smem (show (Store.smem : Store) ≠ .reg by decide)]
  refine ⟨fun u hu q => by rw [mma_other n _ M u hu q]; exact hother u hu q, fun i j hi hj hc => ?_⟩
  rw [mma_reg n _ h256 hn0 hn M i j hi hj, hdiv, hmod, hA, hB']
  by_cases hb : blk n i j = e
  · rw [ite_eq_left hb, ite_eq_left (show blk n i j < e + 1 by omega)]
    by_cases hκ : κ = 0
    · subst hκ
      simp
    · rw [ite_eq_right hκ, hreg i j hi hj (Or.inr (by omega)), ite_eq_right (show ¬ blk n i j < e by omega),
        ofInt_toInt_add, Nat.add_zero, show 32 * (κ + 1) = 32 * κ + 32 by ring, Finset.sum_range_add]
  · rw [ite_eq_right hb, hreg i j hi hj (by omega)]
    congr 3
    by_cases h1 : blk n i j < e
    · rw [ite_eq_left h1, ite_eq_left (show blk n i j < e + 1 by omega)]
    · rw [ite_eq_right h1, ite_eq_right (show ¬ blk n i j < e + 1 by omega)]

theorem mma_next (n κ : ℕ) (h256 : 256 ∣ n) (M₀ M : hm.Mem) (h : MmaInv n M₀ M κ (blocks n)) :
    MmaInv n M₀ M (κ + 1) 0 := by
  refine ⟨h.1, fun i j hi hj _ => ?_⟩
  have hb := blk_lt n i j h256 hi hj
  rw [h.2 i j hi hj (Or.inl hb), ite_eq_left hb, ite_eq_right (Nat.not_lt_zero _)]

theorem mma_slices (n : ℕ) (h256 : 256 ∣ n) (hn0 : 0 < n) (hn : n ≤ 2048) (M₀ : hm.Mem) :
    ∀ κ, MmaInv n M₀ (((List.range (κ * blocks n)).map (mma n)).foldl (fun M a => a.exec M) M₀) κ 0
  | 0 => ⟨fun _ _ _ => by simp, fun i j _ _ h => by omega⟩
  | κ + 1 => by
    have ih := mma_slices n h256 hn0 hn M₀ κ
    have hblocks : ∀ e, e ≤ blocks n →
        MmaInv n M₀ (((List.range (κ * blocks n + e)).map (mma n)).foldl (fun M a => a.exec M) M₀) κ e := by
      intro e
      induction e with
      | zero => intro _; simpa using ih
      | succ e ihe =>
        intro he
        rw [show κ * blocks n + (e + 1) = (κ * blocks n + e) + 1 by ring, List.range_succ, List.map_append,
          List.foldl_append, List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil]
        exact mma_step n κ e h256 hn0 hn (by omega) M₀ _ (ihe (by omega))
    rw [show (κ + 1) * blocks n = κ * blocks n + blocks n by ring]
    exact mma_next n κ h256 M₀ _ (hblocks (blocks n) le_rfl)

/-- **The MMA phase**: from `M₀`, the kernel's MMAs leave every store but the registers as it was, and accumulator
`(i, j)` holding row `i` of `A` dotted with column `j` of `B`, as shared memory holds them in `M₀`, wrapped to a word. -/
theorem mmas_exec (n : ℕ) (h256 : 256 ∣ n) (hn0 : 0 < n) (hn : n ≤ 2048) (M₀ : hm.Mem) :
    (∀ u, u ≠ .reg → ∀ q,
      ((List.range (mmas n)).map (mma n)).foldl (fun M a => a.exec M) M₀ (hm.loc u q) = M₀ (hm.loc u q)) ∧
    ∀ i j, i < n → j < n →
      ((List.range (mmas n)).map (mma n)).foldl (fun M a => a.exec M) M₀ (hm.loc .reg (acc n i j)) =
        BitVec.ofInt 32 (∑ l ∈ Finset.range n, aOf n M₀ i l * bOf n M₀ l j) := by
  have h := mma_slices n h256 hn0 hn M₀ (n / 32)
  have h32 : 1 ≤ n / 32 := by omega
  refine ⟨h.1, fun i j hi hj => ?_⟩
  rw [mmas, h.2 i j hi hj (Or.inr h32), ite_eq_right (Nat.not_lt_zero _), Nat.add_zero,
    show 32 * (n / 32) = n by omega]

/-! ## The honest lemma -/

theorem idx_lt {a b i w : ℕ} (hi : i < a) (hw : w < b) : i * b + w < a * b := by nlinarith

/-- **The kernel computes `C = A·B`**: from any state holding `A` and `B`, it leaves each entry of `C` the exact sum,
a 32-bit word. -/
theorem kernel_computes (n : ℕ) (hn0 : 0 < n) (h256 : 256 ∣ n) (hn : n ≤ 2048) (A B : Fin n → Fin n → BitVec 8)
    (σ : hm.State ℕ) (hpc : ∀ θ, σ.pc θ = 0) (hP : Placed n A B σ.mem) (i j : Fin n) :
    (Machine.run (prog (kernel n)) σ (sched (kernel n))).1.mem (hm.loc .dram (dramC n i j)) =
      BitVec.ofInt 32 (∑ l : Fin n, (A i l).toInt * (B l j).toInt) := by
  have h4 : 4 ∣ n := Nat.dvd_trans (by norm_num) h256
  have hsz : 2 * n * (n / 4) ≤ hm.card.size .smem := by
    show _ ≤ 132 * 65536
    have := Nat.mul_le_mul (Nat.mul_le_mul_left 2 hn) (show n / 4 ≤ 512 by omega)
    omega
  have hsd : hm.card.size .smem ≤ hm.card.size .dram := by
    show 132 * 65536 ≤ 141 * 2 ^ 30 / 4
    norm_num
  have hsd' : n * n / 2 + n * n ≤ hm.card.size .dram := by
    show _ ≤ 141 * 2 ^ 30 / 4
    have := Nat.mul_le_mul hn hn
    omega
  have hdd : hm.card.size .dram = card.size .dram := rfl
  have hss : hm.card.size .smem = card.size .smem := rfl
  have h2k : 2 * n * (n / 4) = 2 * (n * (n / 4)) := Nat.mul_assoc 2 n (n / 4)
  have hq4 : n * n / 4 = n * (n / 4) := Nat.mul_div_assoc n h4
  have hnn : n * n = 4 * (n * (n / 4)) := by
    rw [← Nat.mul_assoc, Nat.mul_comm 4 n, Nat.mul_assoc, Nat.mul_div_cancel' h4]
  have hL := loads_exec n h4 hsz hsd σ.mem (2 * n) le_rfl
  have hM := mmas_exec n h256 hn0 hn (((List.range (2 * n)).map (load n)).foldl (fun M a => a.exec M) σ.mem)
  have hS := stores_exec n h4 hsd' (sq_le_reg n hn)
    (((List.range (mmas n)).map (mma n)).foldl (fun M a => a.exec M)
      (((List.range (2 * n)).map (load n)).foldl (fun M a => a.exec M) σ.mem)) (n * n / 4) le_rfl
  have hacc := acc_lt n i j i.isLt j.isLt
  have hdc : dramC n i j - n * n / 2 = acc n i j := by simp only [dramC, acc]; omega
  rw [(run_plan _ σ hpc).2, kernel, List.foldl_append, List.foldl_append,
    hS .dram (dramC n i j) (by simp only [dramC, acc] at hacc ⊢; omega),
    ite_eq_left ⟨rfl, by simp only [dramC]; omega, by simp only [dramC, acc] at hacc ⊢; omega⟩, hdc,
    hM.2 i j i.isLt j.isLt, ← Fin.sum_univ_eq_sum_range]
  congr 1
  refine Finset.sum_congr rfl fun l _ => ?_
  have hl4 : (l : ℕ) / 4 < n / 4 := by omega
  have ha := idx_lt i.isLt hl4
  have hb := idx_lt j.isLt hl4
  simp only [aOf, bOf]
  rw [hL .smem _ (by simp only [addrA]; omega), ite_eq_left ⟨rfl, by simp only [addrA]; omega⟩,
    hL .smem _ (by simp only [addrB]; omega), ite_eq_left ⟨rfl, by simp only [addrB]; omega⟩, hP.1 i l, hP.2 l j]

/-- **The honest instance** (the card running the deployed GEMM): in `secs n` seconds, on every memory holding some `A`
and `B`, to the memory the kernel leaves, which holds `C = A·B` (`gemm_computes`). -/
def gemmCan (n : ℕ) (f : card.Mem → card.Mem) (T : ℚ≥0) (D : card.Mem → Prop) : Prop :=
  T = secs n ∧ D = (fun M => ∃ A B, Placed n A B M) ∧
    f = fun M => (Machine.run (prog (kernel n)) ⟨M, fun _ => 0⟩ (sched (kernel n))).1.mem

/-- **The card premise holds at the honest instance**: at the exact `s8` semantics, which meets (P1) and (P2), the
kernel is a program of the machine that meets the model and computes the card's map. -/
theorem gemm_cardWithinAccel (n : ℕ) (h256 : 256 ∣ n) (hn : n ≤ 2048) :
    Verity.Assumptions.CardWithinAccel card (gemmCan n) := by
  refine ⟨s8Open, s8Open_exact _, s8Open_productForm, ?_⟩
  rintro f T D ⟨rfl, rfl, rfl⟩
  refine ⟨prog (kernel n), sched (kernel n), fun σ _ hpc => ⟨kernel_fits n h256 hn σ hpc, ?_⟩⟩
  obtain ⟨M, pc⟩ := σ
  obtain rfl : pc = fun _ => 0 := funext hpc
  rfl

/-- The honest instance's map leaves `C = A·B`. -/
theorem gemm_computes (n : ℕ) (hn0 : 0 < n) (h256 : 256 ∣ n) (hn : n ≤ 2048) {f : card.Mem → card.Mem} {T : ℚ≥0}
    {D : card.Mem → Prop} (h : gemmCan n f T D) (A B : Fin n → Fin n → BitVec 8) (M : card.Mem)
    (hP : Placed n A B M) (i j : Fin n) :
    f M (hm.loc .dram (dramC n i j)) = BitVec.ofInt 32 (∑ l : Fin n, (A i l).toInt * (B l j).toInt) := by
  obtain ⟨-, -, rfl⟩ := h
  exact kernel_computes n hn0 h256 hn A B ⟨M, fun _ => 0⟩ (fun _ => rfl) hP i j

end Verity.Accel.H200.Resident
