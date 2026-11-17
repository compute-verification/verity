module

public import Definitions.Core.Accel.Isa
public import Definitions.Registration.Device
public import Mathlib.Data.NNRat.Floor

/-! A card's table of numbers (`Card`), the units each ACCEL op uses of it (`Card.use`) and the device it gives the
device model (`Card.toDevice`). -/

@[expose] public section

open scoped NNRat

namespace Verity.Accel

/-- **A card's table.** -/
structure Card where
  /-- the words each store holds -/
  size : Store → ℕ
  size_pos : ∀ s, 0 < size s
  /-- each resource's units a second, at the clock's ceiling -/
  ρ : Res → ℚ≥0
  /-- the tensor units a MAC of each format uses (an operand pair's, by `A`'s format) -/
  weight : Fmt → ℚ≥0
  /-- P1's range: an integer MMA is exact while its accumulator and products sum within `2^exact` in magnitude -/
  exact : ℕ
  /-- the instructions a program's code holds, all threads' texts together -/
  text : ℕ

/-- A word's read or write at a store: a bank word of shared memory, a word of L2's or a word of device memory's
bandwidth (registers have none). -/
def port : Store → Res → ℚ≥0
  | .smem, .smem => 1
  | .l2, .l2 => 1
  | .dram, .dram => 1
  | _, _ => 0

/-- A register word an atomic or a reduction writes: a lane of the `alu`. -/
def regWr (t : Store) (r : Res) : ℚ≥0 := if t = .reg ∧ r = .alu then 1 else 0

/-- One issue slot, or a lane's share of one. -/
def slot (q : ℚ≥0) (r : Res) : ℚ≥0 := if r = .issue then q else 0

namespace Card

variable (c : Card) {x : Open}

/-- A location: a store and a word of it. -/
abbrev Loc : Type := Σ s : Store, Fin (c.size s)

/-- The stores' contents: a word at every location. -/
abbrev Mem : Type := c.Loc → Word

/-- **The units of each resource an opcode uses.** -/
def use : Opcode x → Res → ℚ≥0
  | .mma o sh sa sb, r =>
    (if r = .tensor then ((sh.m * sh.n * max sh.k 1 : ℕ) : ℚ≥0) * c.weight (x.fa o) else 0) + slot 1 r +
      (sh.wa (x.fa o) : ℚ≥0) * port sa r + (sh.wb (x.fb o) : ℚ≥0) * port sb r
  | .mmaSp o sh sa sb, r =>
    (if r = .tensor then ((sh.m * sh.n * max (sh.k / 2) 1 : ℕ) : ℚ≥0) * c.weight (x.fa o) else 0) + slot 1 r +
      (sh.waSp (x.fa o) : ℚ≥0) * port sa r + (sh.wb (x.fb o) : ℚ≥0) * port sb r
  | .lane o, r => slot (1 / 32) r + if (x.pipe o).res = some r then 1 else 0
  | .move s t n k, r =>
    slot (((max k 1 : ℕ) : ℚ≥0) / 32) r + (n : ℚ≥0) * port s r + ((k * n : ℕ) : ℚ≥0) * port t r +
      if s = .reg ∧ t = .reg ∧ r = .alu then ((k * n : ℕ) : ℚ≥0) else 0
  | .atom s _, r => slot (1 / 32) r + port s r + regWr s r
  | .red s t n _, r => slot (1 / 32) r + (n : ℚ≥0) * (port s r + port t r + regWr t r)

/-- **(P1) exactness**: at the semantics `x`, every integer MMA opcode with 32-bit accumulators computes `C + Σ a·b`
exactly (as a 32-bit two's complement word) while `|C| + Σ |a·b| < 2^exact`. -/
def Exact (x : Open) : Prop :=
  ∀ (o : x.Mma) (k acc : ℕ) (a b : List ℕ), intPair (x.fa o) (x.fb o) = true → x.wide o = false →
    a.length = k → b.length = k → acc < 2 ^ 32 →
    |(BitVec.ofNat 32 acc).toInt| + (List.zipWith (fun p q => |(x.fa o).intVal p * (x.fb o).intVal q|) a b).sum <
      2 ^ c.exact →
    x.dot o k acc a b = (BitVec.ofInt 32 ((BitVec.ofNat 32 acc).toInt +
      (List.zipWith (fun p q => (x.fa o).intVal p * (x.fb o).intVal q) a b).sum)).toNat

/-- The card as the device model's device at ticks of `τ` seconds, each limit rounded up. The side channels `cap`
are the host links', not the card's. -/
def toDevice {ι : Type} (τ : ℚ≥0) (cap : ι → ℕ) : ResourceAccounting.Device ι where
  chip := 32 * (c.size .reg + c.size .smem + c.size .l2)
  mem := 32 * c.size .dram
  rate
    | .tc => ⌈c.ρ .tensor * τ⌉₊
    | .cuda => ⌈(32 * c.ρ .issue + c.ρ .alu + c.ρ .imad + c.ρ .shfl + c.ρ .f64 + c.ρ .sfu) * τ⌉₊
    | .oracle => ⌈c.ρ .issue * τ⌉₊
  bw := ⌈32 * c.ρ .dram * τ⌉₊
  cap := cap

end Card

end Verity.Accel
