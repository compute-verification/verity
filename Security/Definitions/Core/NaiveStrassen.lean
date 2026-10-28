/-!
# Naive tiled against Strassen in the tensor-core model (trusted)

Exact int8 matrix products, priced by the tensor core alone (G2's tensor resource over P1's exact domains), additions
free.

**Atoms.** An exact path for one sub-product is an MMA format (`Atom`): the tensor units a MAC uses, in eighths of an
s8 MAC (`weight8`), the most consecutive integers it holds per operand entry (`cap`), and, for a float format, the
instruction's `k` and the width `w` within which its dot of `k` products is exact (`k · ⌊d_A/2⌋ · ⌊d_B/2⌋ ≤ 2^w`).
These are the ACCEL model's numbers (`Accel.H200`, `Accel.Sm120`, `Accel.Principles`): each card's `weight`, the
formats' integer runs, and P1's widths (24 where FP32 bounds the dot, 14 for the H200's FP8). Taking every float dot
as exact as FP32 allows (f16 and tf32 are `Open.dot` on both cards) and accumulation across MMAs as exact only adds
atoms. `b1` is at `Open.b1 = 1/8`; costs are monotone in the weights, so a lower bound here holds at every
`Open.b1 ≥ 1/8` and, on the H200, every `Open.sync ≥ 1`.

**Schedules** (`Sched`) multiply two `2^l × 2^l` block matrices whose entries range over `a` and `b` consecutive
integers: an atom; a digit split of one operand, `x - c = x₀ + d·x₁` with `2 ≤ d < a`, `x₀` over `d` values and `x₁`
over `⌈a/d⌉`; or a 2 × 2 scheme from a given list, each product on the combination its coefficients make, which
ranges over `(a - 1)·‖u‖₁ + 1` values, reduced mod `m` when `m > 0` (a CRT residue). An atom at depth budget `l`
multiplies `2^l`-leaf blocks, so it costs `8^l · weight8`; naive tiled, the s8 atom at `(256, 256)`, costs `8^(l+1)`.
-/

namespace Verity.Accel.NaiveStrassen

/-- An exact path for a sub-product: one MMA format. -/
structure Atom where
  /-- tensor units a MAC uses, in eighths of an s8 MAC -/
  weight8 : Nat
  /-- the most consecutive integers an operand entry may range over -/
  cap : Nat
  /-- a float format's instruction `k` and exact width `w`; `none` for an integer format, exact on its whole range -/
  coupling : Option (Nat × Nat)
  deriving BEq, DecidableEq, Repr

/-- `f` holds a product of operands ranging over `a` and `b` values exactly. -/
def Atom.holds (f : Atom) (a b : Nat) : Bool :=
  a ≤ f.cap && b ≤ f.cap &&
    match f.coupling with
    | none => true
    | some (k, w) => k * (a / 2) * (b / 2) ≤ 2 ^ w

/-- A 2 × 2 bilinear scheme: product `t` multiplies `∑ u[t][p] A_p` by `∑ v[t][q] B_q`, and `C_r = ∑ w[r][t] M_t`,
blocks in the order `11, 12, 21, 22`. -/
structure Scheme where
  u : List (List Int)
  v : List (List Int)
  w : List (List Int)
  deriving BEq, DecidableEq, Repr

/-- The number of products. -/
def Scheme.R (s : Scheme) : Nat := s.u.length

/-- The `ℓ₁` norm of an integer vector. -/
def l1 (c : List Int) : Nat := (c.map Int.natAbs).foldl (· + ·) 0

/-- The scheme computes `C = A·B` (Brent's equations over the 2 × 2 blocks). -/
def Scheme.correct (s : Scheme) : Bool :=
  s.v.length == s.R && (s.u ++ s.v).all (·.length == 4) && s.w.length == 4 && s.w.all (·.length == s.R) &&
    (List.range 4).all fun p => (List.range 4).all fun q => (List.range 4).all fun r =>
      ((List.range s.R).map fun t => s.u[t]![p]! * s.v[t]![q]! * s.w[r]![t]!).foldl (· + ·) 0 ==
        if p / 2 = r / 2 ∧ p % 2 = q / 2 ∧ q % 2 = r % 2 then 1 else 0

/-- Strassen's scheme. -/
def strassen : Scheme where
  u := [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]]
  v := [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
  w := [[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]]

/-- Strassen–Winograd's scheme, its `S_i` and `T_i` expanded over the input blocks. -/
def winograd : Scheme where
  u := [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, -1, -1], [0, 0, 0, 1], [0, 0, 1, 1], [-1, 0, 1, 1], [1, 0, -1, 0]]
  v := [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, -1, 1], [-1, 1, 0, 0], [1, -1, 0, 1], [0, -1, 0, 1]]
  w := [[1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0], [1, 0, 0, -1, 0, 1, 1], [1, 0, 0, 0, 1, 1, 1]]

/-- The trivial block split: `C_ik = ∑_j A_ij B_jk`, eight products. -/
def trivial : Scheme where
  u := [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0],
    [0, 0, 0, 1]]
  v := [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
    [0, 0, 0, 1]]
  w := [[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1]]

/-- The schemes of the family. -/
def schemes : List Scheme := [strassen, winograd, trivial]

/-- The values a `±` combination of `ν` entries ranging over `a` values each takes, reduced mod `m` if `m > 0`. -/
def range (m a ν : Nat) : Nat :=
  let r := (a - 1) * ν + 1
  if m = 0 then r else min r m

/-- A recursion tree. -/
inductive Sched where
  /-- one MMA format for the whole sub-product -/
  | atom (f : Atom)
  /-- `A`'s digit split at `d`: the low digit's subtree, then the high digit's -/
  | splitA (d : Nat) (lo hi : Sched)
  /-- `B`'s digit split at `d` -/
  | splitB (d : Nat) (lo hi : Sched)
  /-- a scheme, a subtree for each product -/
  | scheme (s : Scheme) (kids : Fin s.R → Sched)

namespace Sched

/-- `S` multiplies `2^l`-leaf blocks with operands over `a` and `b` values (mod `m` if `m > 0`), by atoms from `fs`
and schemes from `ss`. -/
def valid (fs : List Atom) (ss : List Scheme) (m : Nat) : Nat → Nat → Nat → Sched → Bool
  | _, a, b, .atom f => fs.contains f && f.holds a b
  | l, a, b, .splitA d x y => 2 ≤ d && d < a && x.valid fs ss m l d b && y.valid fs ss m l ((a + d - 1) / d) b
  | l, a, b, .splitB d x y => 2 ≤ d && d < b && x.valid fs ss m l a d && y.valid fs ss m l a ((b + d - 1) / d)
  | 0, _, _, .scheme _ _ => false
  | l + 1, a, b, .scheme s kids =>
    ss.contains s && (List.finRange s.R).all fun t =>
      (kids t).valid fs ss m l (range m a (l1 s.u[t.val]!)) (range m b (l1 s.v[t.val]!))

/-- Tensor units, in eighths of an s8 MAC per leaf MAC (`8^(l+1)` for naive tiled). -/
def cost : Nat → Sched → Nat
  | l, .atom f => 8 ^ l * f.weight8
  | l, .splitA _ x y => x.cost l + y.cost l
  | l, .splitB _ x y => x.cost l + y.cost l
  | l, .scheme s kids => ((List.finRange s.R).map fun t => (kids t).cost (l - 1)).foldl (· + ·) 0

/-- Some node is a scheme of fewer than eight products (Strassen or Winograd). -/
def fast : Sched → Bool
  | .atom _ => false
  | .splitA _ x y => x.fast || y.fast
  | .splitB _ x y => x.fast || y.fast
  | .scheme s kids => s.R < 8 || (List.finRange s.R).any fun t => (kids t).fast

end Sched

/-! ## The cards' atoms (`b1` at `Open.b1 = 1/8`) -/

/-- INT8 MMA, s32 accumulate: weight 1, all 256 values. -/
def s8 : Atom := ⟨8, 256, none⟩
/-- Binary MMA (AND/XOR-popcount) on `{0, 1}`, at `Open.b1 = 1/8`. -/
def b1 : Atom := ⟨1, 2, none⟩

/-- The H200 SXM at its data sheet's instruction `k` (f16 and bf16 at 8, tf32 at 4): s8, e4m3, e5m2, f16, bf16,
tf32, b1. -/
def h200Sheet : List Atom :=
  [s8, ⟨8, 33, some (32, 14)⟩, ⟨8, 17, some (32, 14)⟩, ⟨16, 4097, some (8, 24)⟩, ⟨16, 513, some (8, 24)⟩,
    ⟨32, 4097, some (4, 24)⟩, b1]

/-- The H200 SXM at the measured instruction `k` (f16 and bf16 at 16, tf32 at 8). -/
def h200Measured : List Atom :=
  [s8, ⟨8, 33, some (32, 14)⟩, ⟨8, 17, some (32, 14)⟩, ⟨16, 4097, some (16, 24)⟩, ⟨16, 513, some (16, 24)⟩,
    ⟨32, 4097, some (8, 24)⟩, b1]

/-- The RTX PRO 6000 (sm_120) at its data sheet's `k`: s8, e4m3, e5m2, e2m1, nvf4 (`k = 64`, weight 1/2), f16,
bf16, tf32, b1. -/
def rtx6000Sheet : List Atom :=
  [s8, ⟨8, 33, some (32, 24)⟩, ⟨8, 17, some (32, 24)⟩, ⟨8, 9, some (32, 24)⟩, ⟨4, 9, some (64, 24)⟩,
    ⟨16, 4097, some (8, 24)⟩, ⟨16, 513, some (8, 24)⟩, ⟨32, 4097, some (4, 24)⟩, b1]

/-- The RTX PRO 6000 at the measured `k` (f16 and bf16 at 16, f16 at 8 at twice the weight, tf32 at 4). -/
def rtx6000Measured : List Atom :=
  [s8, ⟨8, 33, some (32, 24)⟩, ⟨8, 17, some (32, 24)⟩, ⟨8, 9, some (32, 24)⟩, ⟨4, 9, some (64, 24)⟩,
    ⟨16, 4097, some (16, 24)⟩, ⟨32, 4097, some (8, 24)⟩, ⟨16, 513, some (16, 24)⟩, ⟨32, 4097, some (4, 24)⟩, b1]

/-- The four instances. -/
def instances : List (List Atom) := [h200Sheet, h200Measured, rtx6000Sheet, rtx6000Measured]

/-! ## The statements -/

/-- **Theorem 1**: at every depth budget `l ≤ L`, every schedule over `fs` for int8 operands that applies Strassen
or Winograd somewhere costs at least naive tiled's `8^(l+1)` plus 32, i.e. `1 + 4·8^(-l)` times naive.

D4's guarantee (`Verity.Guarantees.NaiveBeatsStrassenServed`) states it at `L = 12` on all four instances. Served
inner and outer dimensions are at most 65536 (DeepSeek-V4's and Llama-3.1-70B's MLPs), and a leaf is at least 16:
sm_120 has no `wgmma`, its `mma.sync` tiles are 16 wide, and the H200 is taken at `Open.sync = 1`. So a recursion on
a served product is at most `log₂(65536 / 16) = 12` deep. Deeper schedules pad past the served size, and the size
form `NaiveBeatsStrassenUpTo` at `c = 4096` bounds them at every depth. Both are about int8 operands at `(256, 256)`.

The exact tables hold further on three instances: to depth 13 on the H200 (both `k`) and the RTX PRO 6000 at the
measured `k`. On the RTX PRO 6000's data sheet they stop at 12: with b1 at `1/8`, a schedule reaches `0.998267` of
naive at depth 13. -/
def NaiveBeatsStrassen (fs : List Atom) (L : Nat) : Prop :=
  ∀ l ≤ L, ∀ S : Sched, S.valid fs schemes 0 l 256 256 = true → S.fast = true → 8 ^ (l + 1) + 32 ≤ S.cost l

/-- **Theorem 1 in size**: an `n × n` product padded to `2^l` leaves of size `b`, at any depth `l`, loses to naive
tiled's `8 n³` whenever `n ≤ c·b`.

D4's guarantee (`Verity.Guarantees.NaiveBeatsStrassenServedSize`) states it at `c = 4096 = 65536 / 16`
(dimensions at most 65536, leaves at least 16), for int8 operands at `(256, 256)`. The proof uses Theorem 1 up to
depth 12, the least cost at depth 13 (above `8^13`), and `cost ≥ 7^l` from depth 14 on (`7^14 > 2^39`).

The exact tables give larger constants, such as `c = 16113` on the H200's data sheet
(`2^14 G_14^(1/3) = 16113.9`, with `G_14` the least cost at depth 14), but certifying them needs tables to depth
16. -/
def NaiveBeatsStrassenUpTo (fs : List Atom) (c : Nat) : Prop :=
  ∀ (l b n : Nat) (S : Sched), S.valid fs schemes 0 l 256 256 = true → S.fast = true → 1 ≤ n → n ≤ 2 ^ l * b →
    n ≤ c * b →
    8 * n ^ 3 < b ^ 3 * S.cost l

/-- Pairwise coprime. -/
def coprime : List Nat → Bool
  | [] => true
  | m :: ms => ms.all (Nat.gcd m · == 1) && coprime ms

/-- A root-CRT schedule for inner dimension `K` at depth `l`: pairwise coprime moduli whose product is at least
`128·(255K − 381)`, which every exact one needs (an entry of `A·B` takes at least that many values), each with a
schedule over `ℤ_m` on residues of `min(256, m)` values. -/
def crtValid (fs : List Atom) (K l : Nat) (rs : List (Nat × Sched)) : Bool :=
  !rs.isEmpty && coprime (rs.map (·.1)) && 128 * (255 * K - 381) ≤ (rs.map (·.1)).foldl (· * ·) 1 &&
    rs.all fun (m, S) => 2 ≤ m && S.valid fs schemes m l (min 256 m) (min 256 m)

/-- **Theorem 2 (root CRT)**: every root-CRT schedule for an `n × n` int8 product, padded to `2^l` leaves of size
`b`, with at least two moduli or a fast scheme, costs more than naive tiled for every `2 ≤ n ≤ N` (`N = 494` on the
H200; at `n = 1` the count `128·(255K − 381)` is vacuous). -/
def NaiveBeatsCrtStrassen (fs : List Atom) (N : Nat) : Prop :=
  ∀ (l b n : Nat) (rs : List (Nat × Sched)), 2 ≤ n → n ≤ N → n ≤ 2 ^ l * b → crtValid fs n l rs = true →
    (2 ≤ rs.length ∨ rs.any (·.2.fast) = true) → 8 * n ^ 3 < b ^ 3 * (rs.map fun r => r.2.cost l).foldl (· + ·) 0

/-! ## The non-vacuity witness -/

/-- One Winograd at depth 1: `M₁` and `M₂` on s8, the other five on f16 (their ranges up to `1021 × 256`). It costs
`8 + 8 + 5·16 = 96 = 8^2 + 32`, so Theorem 1's bound is attained at `l = 1`. -/
def winogradOnce : Sched :=
  .scheme winograd fun t => if t.val < 2 then .atom s8 else .atom ⟨16, 4097, some (8, 24)⟩

end Verity.Accel.NaiveStrassen
