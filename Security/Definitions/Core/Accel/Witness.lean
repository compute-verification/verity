import Definitions.Core.Accel.H200

/-!
# The non-vacuity witness: an honest INT8 GEMM meets the H200's model (trusted)

An honest run meets every hypothesis (rule I): the resident `wgmma` kernel below computes `C = A·B` exactly for any
`n × n` `s8` matrices `A` and `B` (`n` a multiple of 256, at most 2048), meets G1, G2 and the code fetch on the H200
instance at every value of its open parameters with a backlog of at least a tick (θ's floor), and takes at most
`2n + 21n³/10485760 + n²/604 + 1` ticks. At `n = 2048` that is 28,244 ticks against the 15,888 the tensor core's
rate needs for the products alone (17,190 at the data sheet's 1830 MHz). `Witness` is the statement.

**The kernel.** `A` sits in HBM row-major and `B` column-major, a byte an entry, four to a word; `C` row-major, a word
an entry. Three phases, one after the other:

* **loads** (`load`): `2n` bulk copies HBM → shared memory, one a tick, each a row of `A` or a column of `B`
  (`n/4` words, at most 512 of HBM's 606 a tick);
* **MMAs** (`mma`): `n³/524288` `wgmma.m64n256k32.s32.s8.s8` with `A` and `B` from shared memory, one a tick (524,288
  of the tensor core's 540,672 units at the tick's clock, 2,560 of the banks' 4,224 words) with an idle tick after
  every 20 (`mmaTick`: 20 · 524,288 ≤ 21 · 499,712, so with a tick of backlog every window meets `tensorLong` even at
  the 1830 MHz cap, lifted until a clock check on the H200 rules on it), slice
  by slice along `k`, each adding its 64 × 256 block's products of the slice to the block's accumulators in registers
  (zero at the first slice);
* **stores** (`store`): `n²/4` four-word copies registers → HBM, 151 a tick (604 of HBM's 606 words).

Each instruction is the text of a thread of its own (`prog`), so the code is the instruction count: at `n = 2048`,
1,069,056 instructions, 4,276,224 words fetched beside 6,291,456 data words, within the 17,117,575 the run's HBM port
carries without the backlog. Tick `t` steps the threads of the plan's tick `t` (`sched`). `A` and `B` (`n²/2` words) fit
in shared memory and `C`'s accumulators (`n²`) in the register file, both pooled (`Card`).
-/

open scoped NNRat

namespace Verity.Accel.H200.Resident

open Machine (Operand)

/-- An instruction of the kernel: an H200 opcode, its input operands and its output operands. -/
abbrev Ins : Type := {o : Opcode Store // Avail o} × (ℕ → Operand) × (ℕ → Operand)

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

/-- `len` consecutive words from store `s`, starting at word `a`, to store `t`, starting at word `b`. -/
def copyN (s t : Store) (len : ℕ) (h : Avail (.copy s t len)) (a b : ℕ) : Ins :=
  (⟨.copy s t len, h⟩, fun i => .direct (a + i), fun i => .direct (b + i))

/-! ## Layout -/

/-- Word `w` of row `i` of `A`, in HBM and in shared memory. -/
def addrA (n i w : ℕ) : ℕ := i * (n / 4) + w

/-- Word `w` of column `j` of `B`, in HBM and in shared memory. -/
def addrB (n j w : ℕ) : ℕ := n * n / 4 + j * (n / 4) + w

/-- Entry `(i, j)` of `C` in HBM. -/
def dramC (n i j : ℕ) : ℕ := n * n / 2 + i * n + j

/-- Accumulator `(i, j)` of `C` in the register file. -/
def acc (n i j : ℕ) : ℕ := i * n + j

/-! ## The kernel -/

/-- Load `c`: row `c` of `A` for `c < n`, else column `c - n` of `B`, HBM to shared memory in one bulk copy. -/
def load (n c : ℕ) : Ins :=
  let h : Avail (.copy .dram .smem (n / 4)) := Or.inr (Or.inr ⟨rfl, by decide⟩)
  if c < n then copyN .dram .smem (n / 4) h (addrA n c 0) (addrA n c 0)
  else copyN .dram .smem (n / 4) h (addrB n (c - n) 0) (addrB n (c - n) 0)

/-- The `wgmma`: `s8` by `s8`, `64 × 256 × 32`, both operands in shared memory, wrapping. -/
def mmaOp : Opcode Store := .mma .s8 .s8 false ⟨64, 256, 32⟩ .smem .smem

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
  (⟨mmaOp, ⟨rfl, by decide, Or.inr ⟨rfl, by decide, by decide, by decide, Or.inr rfl, rfl, rfl⟩⟩⟩,
    fun i =>
      if i < 512 then .direct (addrA n (i₀ + i / 8) (8 * κ + i % 8))
      else if i < 2560 then .direct (addrB n (j₀ + (i - 512) / 8) (8 * κ + (i - 512) % 8))
      else if κ = 0 then .imm 0
      else .direct (acc n (i₀ + (i - 2560) / 256) (j₀ + (i - 2560) % 256)),
    fun o => .direct (acc n (i₀ + o / 256) (j₀ + o % 256)))

/-- Store `w`: four accumulators of row `w / (n/4)` to `C`. -/
def store (n w : ℕ) : Ins :=
  copyN .reg .dram 4 (Or.inl le_rfl) (acc n (w / (n / 4)) (4 * (w % (n / 4))))
    (dramC n (w / (n / 4)) (4 * (w % (n / 4))))

/-- Items `k t, …, k t + k - 1` of a stream of `len`, those that exist. -/
def chunk (len k t : ℕ) : List ℕ := List.range' (k * t) (min k (len - k * t))

/-- MMA tick `j`: idle when `j % 21 = 20`, else MMA `j - j / 21`, so that 21 consecutive ticks run at most 20
`wgmma`s (M1's long-window cap, 499,712 units a tick, against a `wgmma`'s 524,288: 20 · 524,288 ≤ 21 · 499,712;
`tensorLong` is 540,672 while M14 holds, and the idle ticks keep the kernel within either). -/
def mmaTick (n j : ℕ) : List Ins := if j % 21 = 20 then [] else [mma n (j - j / 21)]

/-- The MMAs a kernel runs: `n/32` slices of `blocks n` each. -/
def mmas (n : ℕ) : ℕ := n / 32 * blocks n

/-- **The kernel's plan**: the loads, then the MMAs with an idle tick after every 20, then the stores. -/
def kernel (n : ℕ) : List (List Ins) :=
  (List.range (2 * n)).map (fun c => [load n c]) ++
    (List.range (mmas n + mmas n / 20)).map (mmaTick n) ++
    (List.range ((n * n / 4 + 150) / 151)).map fun u => (chunk (n * n / 4) 151 u).map (store n)

/-! ## The statement -/

/-- Byte `k` of a word, the lowest first. -/
def byteOf (w : Word) (k : ℕ) : BitVec 8 := BitVec.ofNat 8 (w.toNat / 2 ^ (8 * k))

/-- `A` and `B` in HBM: entry `(i, l)` of `A` is byte `l % 4` of word `l / 4` of row `i`, entry `(l, j)` of `B` byte
`l % 4` of word `l / 4` of column `j`. -/
def Placed {ι : Type} {x : Open} {cap : ι → ℕ} {oracle : ℕ} (n : ℕ) (A B : Fin n → Fin n → BitVec 8)
    (M : (machine x cap oracle).Mem) : Prop :=
  (∀ i l : Fin n, byteOf (M ((machine x cap oracle).loc Store.dram (addrA n i (l / 4)))) (l % 4) = A i l) ∧
    ∀ l j : Fin n, byteOf (M ((machine x cap oracle).loc Store.dram (addrB n j (l / 4)))) (l % 4) = B l j

/-- **The witness**: on the H200 at any open parameters with a backlog of at least a tick (θ's floor), from any state
holding `A` and `B`, the kernel meets G1, G2 and the code fetch, leaves `A·B` in `C` exactly (each entry's sum, a
32-bit word), and takes at most `2n + 21n³/10485760 + n²/604 + 1` ticks. -/
def Witness : Prop :=
  ∀ {ι : Type} (x : Open) (cap : ι → ℕ) (oracle : ℕ) (n : ℕ), 1 ≤ x.slack → 0 < n → 256 ∣ n → n ≤ 2048 →
    ∀ (A B : Fin n → Fin n → BitVec 8) (σ : (machine x cap oracle).State (ℕ × ℕ)), (∀ θ, σ.pc θ = 0) →
      Placed n A B σ.mem →
      Machine.Fits (prog (kernel n)) σ (sched (kernel n)) ∧
      (∀ i j : Fin n, (Machine.run (prog (kernel n)) σ (sched (kernel n))).1.mem
          ((machine x cap oracle).loc Store.dram (dramC n i j)) =
        BitVec.ofInt 32 (∑ l : Fin n, (A i l).toInt * (B l j).toInt)) ∧
      ((sched (kernel n)).length : ℚ≥0) ≤ 2 * (n : ℚ≥0) + 21 * (n : ℚ≥0) ^ 3 / 10485760 + (n : ℚ≥0) ^ 2 / 604 + 1

end Verity.Accel.H200.Resident
