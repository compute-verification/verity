import Definitions.Core.Accel.H200

/-!
# The non-vacuity witness: an honest INT8 GEMM meets the H200's model (trusted)

An honest run meets every hypothesis (rule I): the naive tiled INT8 kernel below computes `C = A·B` exactly for any
`n × n` `s8` matrices `A` and `B` (`n` a multiple of 512, the three matrices fitting in HBM), meets G1 and G2 on the
H200 instance at every value of its open parameters, and takes at most `n³/540672 + n²/512` ticks, so its tensor rate
tends to the data sheet's 132 · 4096 s8 MACs a tick. `Witness` is the statement; its proof is to come after the red
team.

**The kernel.** `A` sits in HBM row-major and `B` column-major, a byte an entry, four to a word; `C` row-major, a word
an entry. For each 512 × 512 tile of `C`, in order:

* a **load stream** of `64n` four-word copies HBM → registers (`load`): for each slice `κ` of 32 along `k`, `A`'s 512
  rows and `B`'s 512 columns of that slice into register buffer `κ % 3`, 132 copies a tick;
* an **MMA stream** of `64n` `mma.sync.m16n8k32.s32.s8.s8.s32` (`mma`), 16 ticks behind the loads, 132 a tick (the
  tensor core's whole rate): for each slice, the 32 × 64 MMAs of the tile, each adding its 16 × 8 block's products
  of the slice to the block's accumulators (zero at the first slice);
* a **store stream** of 65,536 four-word copies registers → HBM (`store`), 151 a tick (604 of HBM's 606 words).

A tick of the first two streams uses 540,672 tensor units, 136.125 issue slots and 528 HBM words; a tick of the third,
4.72 issue slots and 604 HBM words. Each instruction is the text of a thread of its own (`prog`), and tick `t` steps
the threads of the plan's tick `t` (`sched`), so the schedule is fixed before the run, as the model requires.
-/

open scoped NNRat

namespace Verity.Accel.H200.Naive

open Machine (Operand)

/-- An instruction of the kernel: an H200 opcode, its input operands and its output operands. -/
abbrev Ins : Type := {o : Op Store // Avail o} × (ℕ → Operand) × (ℕ → Operand)

/-- The instruction on the H200 machine, unpredicated. -/
def Ins.instr {ι : Type} {x : Open} {cap : ι → ℕ} {oracle : ℕ} (a : Ins) : (machine x cap oracle).Instr :=
  .op a.1 (fun i => a.2.1 i) (fun i => a.2.2 i) none

/-- A plan's program: the instruction at slot `s` of tick `t` is the whole text of thread `(t, s)`. -/
def prog {ι : Type} {x : Open} {cap : ι → ℕ} {oracle : ℕ} (plan : List (List Ins)) :
    (machine x cap oracle).Prog (ℕ × ℕ) :=
  fun θ => ((plan.getD θ.1 [])[θ.2]?.map Ins.instr).toList

/-- A plan's schedule: tick `t` steps threads `(t, 0), (t, 1), …` in order. -/
def sched (plan : List (List Ins)) : List (List (ℕ × ℕ)) :=
  plan.mapIdx fun t is => (List.range is.length).map fun s => (t, s)

/-- Four consecutive words from store `s`, starting at word `a`, to store `t`, starting at word `b`. -/
def copy4 (s t : Store) (h : Avail (.copy s t 4)) (a b : ℕ) : Ins :=
  (⟨.copy s t 4, h⟩, fun i => .direct (a + i), fun i => .direct (b + i))

/-! ## Layout -/

/-- Accumulator `(r, c)` of the tile: register word `512 r + c`. -/
def acc (r c : ℕ) : ℕ := r * 512 + c

/-- Word `q` of row `r` of `A`'s slice in buffer `b`. -/
def panelA (b r q : ℕ) : ℕ := 262144 + b * 8192 + r * 8 + q

/-- Word `q` of column `c` of `B`'s slice in buffer `b`. -/
def panelB (b c q : ℕ) : ℕ := 262144 + b * 8192 + 4096 + c * 8 + q

/-- Word `w` of row `i` of `A` in HBM. -/
def dramA (n i w : ℕ) : ℕ := i * (n / 4) + w

/-- Word `w` of column `j` of `B` in HBM. -/
def dramB (n j w : ℕ) : ℕ := n * n / 4 + j * (n / 4) + w

/-- Entry `(i, j)` of `C` in HBM. -/
def dramC (n i j : ℕ) : ℕ := n * n / 2 + i * n + j

/-! ## The kernel -/

/-- Copy `c` of tile `(I, J)`'s load stream: slice `κ = c / 2048`; the first 1024 copies are `A`'s rows, two a row,
the next 1024 `B`'s columns. -/
def load (n I J c : ℕ) : Ins :=
  let κ := c / 2048
  let e := c % 2048
  let h := (Or.inl le_rfl : Avail (.copy .dram .reg 4))
  if e < 1024 then copy4 .dram .reg h (dramA n (512 * I + e / 2) (8 * κ + 4 * (e % 2))) (panelA (κ % 3) (e / 2) (4 * (e % 2)))
  else
    let e := e - 1024
    copy4 .dram .reg h (dramB n (512 * J + e / 2) (8 * κ + 4 * (e % 2))) (panelB (κ % 3) (e / 2) (4 * (e % 2)))

/-- The MMA: `s8` by `s8`, `16 × 8 × 32`, both operands in registers, wrapping. -/
def mmaOp : Op Store := .mma .s8 .s8 false ⟨16, 8, 32⟩ .reg .reg

/-- MMA `p` of the tile's stream: slice `κ = p / 2048`, block `(i₀, j₀)` of 16 rows and 8 columns. Its inputs are the
block's 16 rows of `A` (8 words each), its 8 columns of `B` (8 words each), and its accumulators (zero at `κ = 0`); its
outputs, the accumulators. -/
def mma (p : ℕ) : Ins :=
  let κ := p / 2048
  let e := p % 2048
  let i₀ := 16 * (e / 64)
  let j₀ := 8 * (e % 64)
  (⟨mmaOp, ⟨rfl, by decide, Or.inl ⟨rfl, rfl, rfl, rfl, rfl⟩⟩⟩,
    fun i =>
      if i < 128 then .direct (panelA (κ % 3) (i₀ + i / 8) (i % 8))
      else if i < 192 then .direct (panelB (κ % 3) (j₀ + (i - 128) / 8) ((i - 128) % 8))
      else if κ = 0 then .imm 0
      else .direct (acc (i₀ + (i - 192) / 8) (j₀ + (i - 192) % 8)),
    fun o => .direct (acc (i₀ + o / 8) (j₀ + o % 8)))

/-- Copy `w` of tile `(I, J)`'s store stream: four accumulators of row `w / 128` to `C`. -/
def store (n I J w : ℕ) : Ins :=
  copy4 .reg .dram (Or.inl le_rfl) (acc (w / 128) (4 * (w % 128)))
    (dramC n (512 * I + w / 128) (512 * J + 4 * (w % 128)))

/-- Items `k t, …, k t + k - 1` of a stream of `len`, those that exist. -/
def chunk (len k t : ℕ) : List ℕ := List.range' (k * t) (min k (len - k * t))

/-- Tile `(I, J)`: `⌈64n / 132⌉ + 16` ticks of loads and MMAs, then 435 ticks of stores. -/
def tile (n I J : ℕ) : List (List Ins) :=
  (List.range ((64 * n + 131) / 132 + 16)).map (fun τ =>
      (chunk (64 * n) 132 τ).map (load n I J) ++
        if 16 ≤ τ then (chunk (64 * n) 132 (τ - 16)).map mma else []) ++
    (List.range 435).map fun u => (chunk 65536 151 u).map (store n I J)

/-- **The kernel's plan**: the tiles of `C`, row by row. -/
def kernel (n : ℕ) : List (List Ins) :=
  (List.range (n / 512)).flatMap fun I => (List.range (n / 512)).flatMap fun J => tile n I J

/-! ## The statement -/

/-- Byte `k` of a word, the lowest first. -/
def byteOf (w : Word) (k : ℕ) : BitVec 8 := BitVec.ofNat 8 (w.toNat / 2 ^ (8 * k))

/-- `A` and `B` in HBM: entry `(i, l)` of `A` is byte `l % 4` of word `l / 4` of row `i`, entry `(l, j)` of `B` byte
`l % 4` of word `l / 4` of column `j`. -/
def Placed {ι : Type} {x : Open} {cap : ι → ℕ} {oracle : ℕ} (n : ℕ) (A B : Fin n → Fin n → BitVec 8)
    (M : (machine x cap oracle).Mem) : Prop :=
  (∀ i l : Fin n, byteOf (M ((machine x cap oracle).loc Store.dram (dramA n i (l / 4)))) (l % 4) = A i l) ∧
    ∀ l j : Fin n, byteOf (M ((machine x cap oracle).loc Store.dram (dramB n j (l / 4)))) (l % 4) = B l j

/-- **The witness**: on the H200 at any open parameters, from any state holding `A` and `B`, the kernel meets G1 and
G2, leaves `A·B` in `C` exactly (each entry's sum, a 32-bit word), and takes at most `n³/540672 + n²/512` ticks. -/
def Witness : Prop :=
  ∀ {ι : Type} (x : Open) (cap : ι → ℕ) (oracle : ℕ) (n : ℕ), 0 < n → 512 ∣ n → 3 * n ^ 2 ≤ 2 * size .dram →
    ∀ (A B : Fin n → Fin n → BitVec 8) (σ : (machine x cap oracle).State (ℕ × ℕ)), (∀ θ, σ.pc θ = 0) →
      Placed n A B σ.mem →
      Machine.Fits (prog (kernel n)) σ (sched (kernel n)) ∧
      (∀ i j : Fin n, (Machine.run (prog (kernel n)) σ (sched (kernel n))).1.mem
          ((machine x cap oracle).loc Store.dram (dramC n i j)) =
        BitVec.ofInt 32 (∑ l : Fin n, (A i l).toInt * (B l j).toInt)) ∧
      ((sched (kernel n)).length : ℚ≥0) ≤ (n : ℚ≥0) ^ 3 / 540672 + (n : ℚ≥0) ^ 2 / 512

end Verity.Accel.H200.Naive
