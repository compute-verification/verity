module

public import PouwCommit.Rows
public import PouwCommit.Draws
public import Identity.Domains.Pouw

/-!
# A pass's records and give-up (`pearl_c_work`, `clear_work.check_records`)

What the prover declares for each matmul of a pass before the coin, and the root that binds it:

* **`UnitRecord`**: the call's real rows (the rest are zero filler), its kept width |K|, the A rows and its weight's B rows
  that fail the per-row rules (excluded), and the rows it gives up (voluntary, V-EX). `UnitRecord.reason` is its
  constructor's refusal, in `UnitRecord.__post_init__`'s words. `UnitRecord.credited` is the band rule: a voluntary A
  row takes its band of 64 A rows' credit, and an excluded or voluntary B row its block of 64 columns'.
* **V-EX's count caps** (`checkRecords`): at most ⌈m/64⌉ voluntary A rows per record and ⌈n/64⌉ voluntary B rows per weight,
  and every unit of a weight declares the same excluded and the same voluntary B rows.
* **`recordsRoot`**: `tagged("verity/pouw/pearl-c/records/v2", salt, weights_root, (w, split(w))…, record…)`, each split
  `Split(k)` (no side channels: the served passes keep every channel, so |K| = k and `k_unit` = k rounded up to 128) and
  each record its v2 layout, with the voluntary rows appended when it has any.

The scheme is Pearl-C v1 -h3 on the H100's record (`pearl-c-h100-v1-h3`), the deployed one: `Scheme` holds its domain
(`PearlC.check`), its grid (`PearlC.tiles`, row-major tiles of `tileM` × `tileN`, the last row block narrower) and its
`wref`. A test instance (the manifest's `instance`) sets `k_min` and the tile.
-/

@[expose] public section

namespace PouwCheck.PearlPlus

open PouwCommit

/-- One declared GEMM: `A (m × k) · Bᵀ`, `B` its weight's `n × k` rows. -/
structure Matmul where
  index : Nat
  m : Nat
  k : Nat
  n : Nat
  weight : Nat
  deriving BEq, Repr, Inhabited

/-- One matmul as its prover declares it before the coin. -/
structure UnitRecord where
  matmul : Matmul
  realRows : Nat
  kKept : Nat
  excludedA : List Nat := []
  excludedB : List Nat := []
  voluntaryA : List Nat := []
  voluntaryB : List Nat := []
  deriving BEq, Repr, Inhabited

/-- Rows `0 ≤ r < bound`, strictly increasing. -/
def increasingBelow (rows : List Nat) (bound : Nat) : Bool :=
  rows.all (· < bound) && (rows.zip rows.tail).all fun (a, b) => a < b

namespace UnitRecord

/-- `UnitRecord.__post_init__`'s refusal, or none. -/
def reason (r : UnitRecord) : Option String :=
  let s := r.matmul
  if !(0 < r.realRows && r.realRows ≤ s.m && 0 < r.kKept && r.kKept ≤ s.k) then
    some "a record's real rows and kept width lie within its shape"
  else if ![(r.excludedA, r.realRows), (r.excludedB, s.n), (r.voluntaryA, r.realRows), (r.voluntaryB, s.n)].all
      fun (rows, bound) => increasingBelow rows bound then
    some "excluded and voluntary rows are strictly increasing and in range"
  else if r.excludedA.any (r.voluntaryA.contains ·) || r.excludedB.any (r.voluntaryB.contains ·) then
    some "a row is excluded or voluntary, not both"
  else none

/-- The band rule's band of A rows and block of B rows, at absolute indices: the game's 64 × 64 tiling. -/
def BAND : Nat := 64

/-- The tile's credited rows and columns, by the band rule (`UnitRecord.credited`): an A row that is real, not excluded,
and whose band of `BAND` rows holds no voluntary A row; a column whose block of `BAND` B rows holds no excluded or
voluntary B row. At the deployed 64 × 64 tile, a tile holding a voluntary row or an excluded B row has no credited cell. -/
def credited (r : UnitRecord) (rows cols : List Nat) : List Nat × List Nat :=
  (rows.filter fun i => i < r.realRows && !r.excludedA.contains i && !r.voluntaryA.any (· / BAND == i / BAND),
   cols.filter fun j => !(r.excludedB ++ r.voluntaryB).any (· / BAND == j / BAND))

end UnitRecord

/-- Little-endian bytes of `x` mod 2^(8·w). -/
def le (w x : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity w
  for i in [0:w] do out := out.push (x / 256 ^ i % 256).toUInt8
  return out

def le32s (xs : List Nat) : ByteArray := xs.foldl (fun acc x => acc ++ le 4 x) ByteArray.empty

/-- `UnitRecord.encode`: `<QQIIIIIII` index, weight, m, k, n, real rows, |K|, #excluded A, #excluded B, then the excluded
rows; with voluntary rows, `<II` their counts and then the rows. -/
def UnitRecord.encode (r : UnitRecord) : ByteArray :=
  let u := r.matmul
  let base := le 8 u.index ++ le 8 u.weight ++ le32s [u.m, u.k, u.n, r.realRows, r.kKept, r.excludedA.length,
    r.excludedB.length] ++ le32s r.excludedA ++ le32s r.excludedB
  if r.voluntaryA.isEmpty && r.voluntaryB.isEmpty then base
  else base ++ le32s [r.voluntaryA.length, r.voluntaryB.length] ++ le32s r.voluntaryA ++ le32s r.voluntaryB

def ALIGN : Nat := 128

/-- `Split(k).k_unit`: the kept width rounded up to whole 128-channel blocks. -/
def kUnit (k : Nat) : Nat := (k + ALIGN - 1) / ALIGN * ALIGN

/-- `Split(k).encode()`: no side channels and no calibration digest. -/
def splitEncode (k : Nat) : ByteArray := le32s [k, 0] ++ tagged Domains.POUW_PEARL_C_SPLIT [.bytes ByteArray.empty]

/-- `records_root(salt, weights_root, splits, records)`, `splits` each weight's k (`Split(k)`). -/
def recordsRoot (salt weightsRoot : ByteArray) (splits : List (Nat × Nat)) (records : List UnitRecord) : ByteArray :=
  let sorted := splits.mergeSort fun a b => a.1 ≤ b.1
  tagged Domains.POUW_PEARL_C_RECORDS
    ([.bytes salt, .bytes weightsRoot] ++ sorted.flatMap (fun (w, k) => [.int w, .bytes (splitEncode k)]) ++
      records.map (.bytes ·.encode))

def VOLUNTARY_PER : Nat := 64

/-- V-EX's count cap on `rows` rows: ⌈rows/64⌉. -/
def voluntaryCap (rows : Nat) : Nat := (rows + VOLUNTARY_PER - 1) / VOLUNTARY_PER

/-- `check_records` on records built from the workload's matmuls in order, `splits` each weight's k: the split's `k_unit`
and |K| on every record, one set of excluded and one of voluntary B rows per weight, and V-EX's count caps. -/
def checkRecords (splits : List (Nat × Nat)) (records : List UnitRecord) : Option String := Id.run do
  let mut exB : List (Nat × List Nat) := []
  let mut volB : List (Nat × List Nat) := []
  for r in records do
    let u := r.matmul
    let k := (splits.lookup u.weight).getD 0
    if (u.k, r.kKept) != (kUnit k, k) then
      return some s!"matmul {u.index}: k and |K| are not its weight's split"
    match exB.lookup u.weight with
    | some e => if e != r.excludedB then return some s!"weight {u.weight}: its units declare different excluded rows"
    | none => exB := (u.weight, r.excludedB) :: exB
    match volB.lookup u.weight with
    | some e => if e != r.voluntaryB then return some s!"weight {u.weight}: its units declare different voluntary rows"
    | none => volB := (u.weight, r.voluntaryB) :: volB
    if r.voluntaryA.length > voluntaryCap u.m || r.voluntaryB.length > voluntaryCap u.n then
      return some (s!"matmul {u.index}: more voluntary rows than V-EX allows (⌈m/64⌉ = {voluntaryCap u.m} of A, " ++
        s!"⌈n/64⌉ = {voluntaryCap u.n} of B)")
  return none

/-! ## The scheme: Pearl-C v1 -h3 on the H100's record -/

def NAME : String := "pearl-c-h100-v1-h3"
def SCHEMES : List String := ["pearl-c-sm120-v1-h3", "pearl-c-sm120-unpromoted-v1-h3", NAME]
def K_MAX : Nat := 2 ^ 16
/-- k's alignment: G = 4 atoms of k = 32 per promotion. -/
def WINDOW : Nat := 128
def G : Nat := 4
/-- The ranks of the noise: r = 32 (`pearl_kw.R`), and the peel's BF16 depth 2r. -/
def R : Nat := 32
def PEEL_K : Nat := 2 * R
/-- The H100's prices (`Prices.h100`): an FP32 add 32, a BF16 MAC 2, forming 64 + 4 per activation element. -/
def FADD : Nat := 32
def BF16_MAC : Nat := 2
def FORMING : Nat := 64 + 4

/-- The scheme's instance: Pearl's `k_min` and the tile, 1,024 and 64 × 64 as deployed. -/
structure Scheme where
  kMin : Nat := 1024
  tileM : Nat := 64
  tileN : Nat := 64
  deriving BEq, Repr, Inhabited

namespace Scheme

/-- The ks below `k_min` the domain takes: the H100 record's census k = 256, each credited row capped at 4ρ there. -/
def censusKs (s : Scheme) : List Nat := if 256 < s.kMin then [256] else []

/-- `PearlC.check`'s refusal of a matmul outside the domain, or none. -/
def domainReason (s : Scheme) (u : Matmul) : Option String :=
  let rows := min s.tileM 16
  if (s.kMin ≤ u.k || s.censusKs.contains u.k) && u.k ≤ K_MAX && u.k % WINDOW == 0 && u.m % rows == 0 &&
      u.n % s.tileN == 0 && u.m ≤ 2 ^ 24 && u.n ≤ 2 ^ 24 then none
  else
    let also := if s.censusKs.isEmpty then ""
      else s!" or k in {", ".intercalate (s.censusKs.map toString)} (the census's)"
    some (s!"{NAME}: Shape(m={u.m}, k={u.k}, n={u.n}) is outside {s.kMin} <= k <= {K_MAX}{also} with {WINDOW} | k " ++
      s!"(whole promotion groups), {rows} | m, {s.tileN} | n, m, n <= 2^24")

/-- `grid_tiles`: consecutive `tileM × tileN` tiles in row-major order, each as (its rows, its columns). -/
def tiles (s : Scheme) (m n : Nat) : List (List Nat × List Nat) :=
  let blocks (len w : Nat) := (List.range ((len + w - 1) / w)).map fun b => List.range' (b * w) (min w (len - b * w))
  (blocks m s.tileM).flatMap fun rows => (blocks n s.tileN).map fun cols => (rows, cols)

end Scheme

/-- `PearlC.wref(Shape(m, k, n))` on the H100's record under rev1: the chain with its promotion adds less each word's first,
the fused A′·F_B, the BF16 peel and A's forming. -/
def wref (m k n : Nat) : Nat :=
  m * n * (k + FADD * (k / WINDOW)) - FADD * m * n + m * k * R + BF16_MAC * PEEL_K * m * n + FORMING * m * k

/-- `PearlC.credit_of`: `wref` less ρ's squared accumulate (qa = 4 per activation element). -/
def creditOf (m k n : Nat) : Nat := wref m k n - 4 * m * k

end PouwCheck.PearlPlus
