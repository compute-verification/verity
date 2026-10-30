import PoubCheck.Basic

/-!
# A pass's plan: `receipts.Plan`

R runs `(first, count)` in read order, in a store of `blocks` blocks: count ≥ 1, inside the store, pairwise disjoint,
and no run starting where the one before it ends. Position p is the p-th block of the runs concatenated.
Its digest is `SHA512(PLAN_DOMAIN ‖ u32be R ‖ (u64be first ‖ u64be count) × R)`. `Plan.make?` refuses (`none`) every
other plan, as `Plan(runs, blocks)` raises; runs arrive as integers, so a negative one is refused here too.
-/

namespace PoubCheck

def PLAN_DOMAIN : ByteArray := tag0 "verity/poub/plan/v0"

structure Plan where
  blocks : Nat
  runs : Array (Nat × Nat)
  /-- The positions' running ends: `ends[r]` is the positions of runs `0 … r`. -/
  ends : Array Nat
deriving Inhabited

/-- `store_blocks`: a u64, at least 1. -/
def storeBlocks (blocks : Nat) : Bool := 1 ≤ blocks && blocks < 2 ^ 64

private def endsOf (runs : Array (Nat × Nat)) : Array Nat :=
  (runs.foldl (fun (acc : Array Nat × Nat) r => (acc.1.push (acc.2 + r.2), acc.2 + r.2)) (#[], 0)).1

private def lexLt (a b : Nat × Nat) : Bool := a.1 < b.1 || (a.1 == b.1 && a.2 < b.2)

/-- `Plan(runs, blocks)`, or `none` where it raises. -/
def Plan.make? (runs : Array (Int × Int)) (blocks : Nat) : Option Plan := do
  if !storeBlocks blocks then none
  if !(1 ≤ runs.size && runs.size < 2 ^ 32) then none
  if runs.any fun (f, c) => f < 0 || c < 1 || f + c > blocks then none
  let rs := runs.map fun (f, c) => (f.toNat, c.toNat)
  if (Array.range (rs.size - 1)).any fun k => rs[k]!.1 + rs[k]!.2 == rs[k + 1]!.1 then none
  let ordered := rs.qsort lexLt
  if (Array.range (ordered.size - 1)).any fun k => ordered[k]!.1 + ordered[k]!.2 > ordered[k + 1]!.1 then none
  some ⟨blocks, rs, endsOf rs⟩

/-- Every block, in store order. -/
def Plan.whole (blocks : Nat) : Plan := ⟨blocks, #[(0, blocks)], #[blocks]⟩

/-- L, the pass's positions. -/
def Plan.length (pl : Plan) : Nat := pl.ends.back?.getD 0

/-- The block at position `p < L`. -/
def Plan.block (pl : Plan) (p : Nat) : Nat :=
  let r := bisectRight pl.ends p
  pl.runs[r]!.1 + p - (if r = 0 then 0 else pl.ends[r - 1]!)

def Plan.encode (pl : Plan) : ByteArray :=
  pl.runs.foldl (fun acc (f, c) => acc ++ be 8 f ++ be 8 c) (be 4 pl.runs.size)

def Plan.digest (pl : Plan) : ByteArray := sha512 (PLAN_DOMAIN ++ pl.encode)

/-- `spans` of `Plan.changed`: runs inside a store of `blocks`, as sorted half-open intervals, disjoint. -/
private def spans (runs : Array (Int × Int)) (blocks : Nat) : Option (Array (Nat × Nat)) := do
  if runs.any fun (f, c) => f < 0 || c < 1 || f + c > blocks then none
  let out := (runs.map fun (f, c) => (f.toNat, (f + c).toNat)).qsort lexLt
  if (Array.range (out.size - 1)).any fun k => out[k]!.2 > out[k + 1]!.1 then none
  some out

/-- `Plan.changed(added, removed, blocks)`: this plan's blocks less `removed`, each of which it holds, plus `added`,
none of which it then holds, in store order, in a store of `blocks` blocks, at least this plan's; `none` otherwise or
if no block is left. -/
def Plan.changed (pl : Plan) (added removed : Array (Int × Int)) (blocks : Nat) : Option Plan := Id.run do
  if blocks < pl.blocks then return none
  let some gone := spans removed blocks | return none
  let some more := spans added blocks | return none
  let mut kept : Array (Nat × Nat) := #[]
  let mut g := 0
  for (f, c) in pl.runs.qsort lexLt do
    let mut a := f
    let b := f + c
    for _ in [0:gone.size] do
      if g < gone.size && gone[g]!.1 < b then
        let r := gone[g]!
        if r.1 < a || r.2 > b then return none
        if r.1 > a then kept := kept.push (a, r.1)
        a := r.2
        g := g + 1
      else break
    if a < b then kept := kept.push (a, b)
  if g < gone.size then return none
  let mut out : Array (Nat × Nat) := #[]
  for (a, b) in (kept ++ more).qsort lexLt do
    match out.back? with
    | some (oa, ob) =>
      if a < ob then return none
      if a == ob then out := out.pop.push (oa, b) else out := out.push (a, b)
    | none => out := out.push (a, b)
  return Plan.make? (out.map fun (a, b) => ((a : Int), ((b - a : Nat) : Int))) blocks

end PoubCheck
