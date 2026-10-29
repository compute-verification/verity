/-
# The per-gate reference of `Flock.Annot`

`Annot.check` decides a meaning on an annotation's plans, at the size of the description. This module lays the
annotation out gate by gate, as the reference the check is held to (`Flock.Test.Annot`, `Properties.Annot.CheckAgrees`,
and difftest): `sets`, the family's sets in family order, each a list of root-gate intervals; `labelled`, the gates
each label holds; `expand`, each gate's labels; and `reference`, the meaning decided on those, gate by gate. Its walk is
over every activation the family reaches, so it is O(unrolled circuit), and no verifier calls it.

The layout is the descriptor's (`Flock.Program`, §6): member `m` of node `j` of an activation at offset `off` is the
gates `[off + offsets[j] + m·G, off + offsets[j] + (m + 1)·G)`, `G` the callee's gates (one for a primitive). An input
gate is one a primitive whose id is an input id computes (`Program.isInputId`), as `Prog.gate` locates it.
-/
import Flock.Annot

namespace Flock.Annot

open Flock.Program

/-- One set of a family: its gates, and the Definition its members call when it is of one node (`group` reads it). -/
structure GateSet where
  callee : Option Nat
  gates : Array (Nat × Nat)
  deriving Inhabited, BEq

section
variable (P : Prog) (F : Family)

/-- Plan `p`'s sets on the activation at offset `off`, appended to `acc` in family order. -/
def walk : Nat → Nat → Nat → Array GateSet → V (Array GateSet)
  | 0, _, _, _ => throw "fuel: a walk takes more steps than this reference follows"
  | fuel + 1, p, off, acc => do
    let pl := F.plans[p]!
    let body := (P.defs[pl.d]!).body
    let g := fun (j : Nat) => memberGates P body.nodes[j]!
    let at_ := fun (j m : Nat) => off + body.offsets[j]! + m * g j
    let mut acc := acc
    for e in pl.entries do
      let fn := body.nodes[e.j]!.fn
      match e.kind with
      | .atomic k =>
        for (a, b) in e.members do
          for t in [0:ceilDiv (b - a) k] do
            let s := a + t * k
            acc := acc.push { callee := some fn, gates := norm #[(at_ e.j s, at_ e.j (min (s + k) b))] }
      | .whole =>
        if total e.members > 0 then
          acc := acc.push { callee := some fn, gates := norm (e.members.map fun (a, b) => (at_ e.j a, at_ e.j b)) }
      | .zip ns =>
        for (a, b) in e.members do
          for m in [a:b] do
            acc := acc.push { callee := none, gates := norm (ns.map fun j => (at_ j m, at_ j (m + 1))) }
      | .table rows l =>
        for v in [0:l] do
          let js := (Array.range rows.size).filter fun j => rows[j]! == some v
          acc := acc.push { callee := none, gates := norm (js.map fun j => (at_ j 0, at_ j body.nodes[j]!.members)) }
      | .descend s =>
        for (a, b) in e.members do
          for m in [a:b] do
            acc ← walk fuel s (at_ e.j m) acc
    return acc

end

/-- The family's sets, in family order. -/
def sets (P : Prog) (F : Family) : V (Array GateSet) := walk P F Steps.FUEL F.top 0 #[]

/-- The gates each label holds: set `i` for label `i` without a map, else the union of the sets the map sends to it. -/
def labelled (P : Prog) (F : Family) : V (Array (Array (Nat × Nat))) := do
  let ss ← sets P F
  match F.map with
  | none => return ss.map (·.gates)
  | some (.affine a b n) =>
    let mut out := Array.replicate n #[]
    for i in [0:ss.size] do
      let v := (a * i + b) % n
      out := out.set! v (out[v]! ++ ss[i]!.gates)
    return out.map norm
  | some (.group byName) =>
    let mut keys : Array String := #[]
    let mut out : Array (Array (Nat × Nat)) := #[]
    for s in ss do
      let some c := s.callee | refuse "term" "group over sets of several nodes"
      let k := if byName then baseName (P.defs[c]!).id else (P.defs[c]!).key
      match keys.idxOf? k with
      | some i => out := out.set! i (out[i]! ++ s.gates)
      | none =>
        keys := keys.push k
        out := out.push s.gates
    return out.map norm

/-- `Annot.expand`: each gate of the circuit, its labels in increasing order. -/
def expand (P : Prog) (D : Desc) : V (Array (Array Nat)) := do
  let ls ← labelled P (← eval P D)
  let mut out := Array.replicate (P.defs[P.root]!).gates #[]
  for l in [0:ls.size] do
    for (a, b) in ls[l]! do
      for g in [a:b] do
        out := out.set! g (out[g]!.push l)
  return out

/-- Whether gate `g` is an input gate: the primitive `Prog.gate` locates it at has an input id. -/
def isInputGate (P : Prog) (g : Nat) : V Bool := do
  let (f, _, _) ← P.gate g
  return isInputId (P.defs[f]!).key

/-- The meaning decided gate by gate: the tiling on the family's sets, each gate in increasing order (`input-held`,
`overlap`, `gate-in-no-set`), then `too-many` on the labels, then `too-large` on the largest label's gates. -/
def reference (Φ : Meaning) (P : Prog) (D : Desc) : V Verdict := do
  let F ← eval P D
  let ss ← sets P F
  let ls ← labelled P F
  let count := ls.size
  let refused := fun (code : String) => (pure { accept := false, code, count } : V Verdict)
  if let some t := Φ.tiling then
    let mut held := Array.replicate (P.defs[P.root]!).gates 0
    for s in ss do
      for (a, b) in s.gates do
        for g in [a:b] do
          held := held.set! g (held[g]! + 1)
    for g in [0:held.size] do
      let input ← isInputGate P g
      if t == .partition && input && held[g]! > 0 then return ← refused "input-held"
      if t == .partition && !input && held[g]! > 1 then return ← refused "overlap"
      if !input && held[g]! == 0 then return ← refused "gate-in-no-set"
  if let some n := Φ.maxCount then
    if count > n then return ← refused "too-many"
  if let some c := Φ.maxGates then
    if (match F.map with | some (.affine ..) => true | _ => false) then
      refuse "meaning-unsupported" "max_gates on an affine label map"
    if ls.any (total · > c) then return ← refused "too-large"
  return { accept := true, count }

end Flock.Annot
