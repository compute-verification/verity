/-
`committedBits` as `SampledProofs.Partition` first computed it: its own walk of one activation of the Definition, every
gate unrolled, to each gate's last reader. `Partition.committedBits` reads the same off `Flock.Qcall.gates`' layout
(`Gates.readTo`), and the `committed_bits` difftest piece holds the two equal on every Definition a `Q_call` partition
cuts. The verifier never runs this.
-/
import Flock.Qcall

namespace SampledProofs.Partition

open Flock (V)

/-- `partition._committed_bits` by its own walk: per unit `[lo, hi)` of one activation of definition `d`, the summed
declared widths of its gates that are Input gates, or that a later gate of the activation reads or the activation
returns, constants included. -/
def committedBitsRef (P : Flock.Program.Prog) (d : Nat) (units : Array (Nat × Nat)) : V (Array Nat) := do
  let defn := P.defs[d]!
  let lo := defn.paramLeaves
  let n := defn.gates
  let starts := (defn.params.foldl (fun (acc : Array Nat × Nat) (_, t) => (acc.1.push acc.2, acc.2 + t.leaves)) (#[], 0)).1
  let top : Flock.Program.Scope := .top d lo (some starts)
  let mut prim : Array Nat := if defn.prim then #[d] else Array.mkEmpty n
  let mut last : Array Int := Array.replicate n (-1)
  let mut stack : Array (Flock.Program.Scope × Nat × Nat) := if defn.prim then #[] else #[(top, 0, 0)]
  while stack.size > 0 do
    let (sc, k, m) := stack.back!
    let body := (P.defs[sc.d]!).body
    if k ≥ body.nodes.size then
      stack := stack.pop
      continue
    let node := body.nodes[k]!
    if m ≥ node.members then
      stack := stack.set! (stack.size - 1) (sc, k + 1, 0)
      continue
    stack := stack.set! (stack.size - 1) (sc, k, m + 1)
    let callee := P.defs[node.fn]!
    let o := sc.off + body.offsets[k]!
    if !callee.prim then
      stack := stack.push (.child sc node.fn (o + m * callee.gates) k m, 0, 0)
      continue
    let j := o + m - lo
    for g in ← Flock.Program.operands P sc node m do
      if lo ≤ g && g - lo < j then last := last.set! (g - lo) j
    prim := prim.push node.fn
  if prim.size != n then throw s!"{defn.key}: {prim.size} gates laid out, {n} declared"
  for g in ← if defn.prim then pure #[lo] else defn.body.ret.toArray.mapM (Flock.Program.resolve P top) do
    if lo ≤ g && g < lo + n then last := last.set! (g - lo) n
  return units.map fun (a, b) => (List.range (b - a)).foldl (fun bits i =>
    let f := P.defs[prim[a + i]!]!
    if Flock.Program.isInputId f.key || last[a + i]! ≥ b then
      bits + match f.ret with
        | .value w => w
        | _ => 0
    else bits) 0

end SampledProofs.Partition
