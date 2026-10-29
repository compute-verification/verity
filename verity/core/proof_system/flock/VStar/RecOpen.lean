import VStar.Sha
import VStar.Residuals

/-!
# `RecOpen{LANES, H}`: one inner Ligerito opening (`verity_flock.rec_open`)

The opened row's plain SHA-512 leaf climbs its `H` siblings by the low bits `dirs` holds, each level's node
`SHA-512(left ‖ right)` with the current digest on the side its bit names (`t = d ∧ (cur ⊕ sib)` swaps them), and the node
it reaches, then 512 zero bits, is the first result (`out`); the second (`acc_out`) is `acc` with the row's consistency terms
`Σ_l coef_r[l] · row[l]` added to its two GF(2^128) elements (`consStructure`, a `GfResiduals` of two residuals), its other
words as they were. The running sums come first (`rec_open._steps`), then the climb.

Ports, u16 words: `row` `8 LANES`, `aux` `A(H) = 64 max(1, ⌈H / 2⌉)` (the siblings bottom up, 32 words each; at `H = 0`
one row the unit does not read), `acc` 64, `coef` `16 LANES`, `dirs` 64, whose words 0 and 1 hold the position's `H` low
bits; `out` 64 and `acc_out` 64.
-/

namespace VStar

open Flock.HmNets

def ROW_BLOCK : Nat := 64
def NODE_WORDS : Nat := 32
def OUT_WORDS : Nat := 64
def ACC_WORDS : Nat := 64
def DIRS_WORDS : Nat := 64

def rowWords (LANES : Nat) : Nat := 8 * LANES
def auxWords (H : Nat) : Nat := max 1 ((NODE_WORDS * H + ROW_BLOCK - 1) / ROW_BLOCK) * ROW_BLOCK
def coefWords (LANES : Nat) : Nat := 16 * LANES

/-- `rec_open.port_words`: `row`, `aux`, `acc`, `coef`, `dirs`. -/
def recOpenPorts (LANES H : Nat) : List Nat := [rowWords LANES, auxWords H, ACC_WORDS, coefWords LANES, DIRS_WORDS]

/-- `rec_open._check`: whole SHA-512 blocks of row, and the position's `H` bits in `dirs`' first two words. -/
def checkRecOpen (LANES H : Nat) : Except String Unit := do
  if LANES < 8 || LANES % 8 != 0 then
    throw s!"RecOpen's LANES is a positive multiple of 8 (whole SHA-512 blocks of row), not {LANES}"
  if 32 < H then throw s!"RecOpen's H is 0 to 32 (the position's bits are dirs' first two words), not {H}"

/-- `rec_open.cons_structure`: `w` is the row's LANES elements then acc's two, `v` the coefficients (`c0[l]`, then
`c1[l]`), and residual `r` is `Σ_l c_r[l] · row[l] + acc_r`. -/
def consStructure (LANES : Nat) : GfS where
  nw := LANES + 2
  nv := 2 * LANES
  ops := (List.range LANES).map (fun l => [Term.w l]) ++ (List.range (2 * LANES)).map fun i => [Term.v i]
  res := (List.range 2).map fun r =>
    { mul := (List.range LANES).map fun l => (LANES + r * LANES + l, l), add := [Term.w (LANES + r)] }

/-- `rec_open._Forms.swap`: `t = d ∧ (a ⊕ s)` bit by bit, the pair `(a ⊕ t, s ⊕ t)`. -/
def swap (d : Lin) (a s : List Lin) : M (List Lin × List Lin) := do
  let t ← (a.zip s).mapM fun x => Flock.HmNets.and d (x.1 ^^^ x.2)
  pure (List.zipWith (· ^^^ ·) a t, List.zipWith (· ^^^ ·) s t)

/-- The climb from level `h`: each sibling swapped by bit `h` of `d`, then the pair's SHA-512. -/
def climbFrom (d : List Lin) : Nat → List Lin → List (List Lin) → M (List Lin)
  | _, cur, [] => pure cur
  | h, cur, s :: rest => do
    let lr ← swap (d.getD h Lin.zero) cur s
    let cur ← shaFrom iv (lr.1 ++ lr.2) 0
    climbFrom d (h + 1) cur rest

/-- The siblings' forms, 512 each, bottom up. -/
def sibForms (aux : List Lin) (H : Nat) : List (List Lin) := (List.range H).map fun h => (aux.drop (512 * h)).take 512

/-- `rec_open._climb`: the row's leaf, the climb, then 512 zero bits. -/
def climbRow (H : Nat) (row aux d : List Lin) : M (List Lin) := do
  let leaf ← shaFrom iv row 0
  let top ← climbFrom d 0 leaf (sibForms aux H)
  pure (top ++ List.replicate 512 Lin.zero)

/-- `rec_open._steps` over forms: `(out, acc_out)`'s bits. -/
def recOpen (LANES H : Nat) (row aux acc coef dirs : List Lin) : M (List Lin × List Lin) := do
  let t ← residualForms (consStructure LANES) (row ++ acc.take 256).toArray coef.toArray
  let out ← climbRow H row aux (dirs.take 32)
  pure (out, t ++ acc.drop 256)

/-- Consecutive slices of `bits` of `16 w` forms for each `w` of `ws`. -/
def slices (bits : List Lin) : List Nat → List (List Lin)
  | [] => []
  | w :: ws => bits.take (16 * w) :: slices (bits.drop (16 * w)) ws

end VStar
