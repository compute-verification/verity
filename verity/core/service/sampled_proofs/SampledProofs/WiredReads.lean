/-
A per-member one-stage audit's wired reads (`OneStageZk`, `proofs.wires`): for each instance of a member's session, the
writer's committed output each of its wired rows reads and the writer unit's position, from the wires alone. The
session's setup opens each instance's row against them (`Flock.RegWired.check`), inside the session.

The window's wiring (`Wires.Wiring`, `Wires.check`'s summary) gives each reader row's writer `(template, unit, out)`
(`Wires.Summary.writerOf`). The verifier's own partition gives each template's first unit (`Partition.evaluate`); its
parameters' members (each named by the window's member key, `<template>/c<j>`) give the member holding a writer unit and
the unit's position, its rank among the member's units (leaf `i` of a member's output row is its `i`-th unit's, as v2's
per-member statement holds its units to the member's); and the registration gives the root of each member's port
(`<member>/<port>`), its leaves and domain the verifier's own `ports`. All of it is fixed before any coin:

* `ctxOf`: the context; each template the wiring names is one template of the partition, of its units, and no
  registration port is named twice;
* `derive`: a session's wired reads, one `Flock.RegWired.Port` per wired row of the member's template, one read per
  session instance (its unit the instance's `units.indices`, which the verifier derived), refused when a wire meets only
  some of the member's units, a writer unit is in no member, the registration holds no root for its output or the root's
  leaves are not its member's units; its work is the session's instances times the wires, never the member's units;
* `boundHeld`: the member's bound ports (`Flock.RegBound`) at the registration's roots and the verifier's domains, so a
  writer's statement is held to the committed rows its readers open.

Nothing here reads a row's value: only positions, roots, domains and registration names, which are public.
-/
import Flock
import SampledProofs.Json
import SampledProofs.Wires
import SampledProofs.Partition
import Std.Data.HashSet

namespace SampledProofs.WiredReads

open Lean (Json)
open Flock (V)

/-- A member of the window: its name (the window's member key its ports are registered under) and its units. -/
structure Member where
  name : String
  units : Array (Nat × Nat)

/-- What a session's wired reads are derived from: the wiring's digest, the wiring and its summary, each template's
first unit and count in the population, the window's members, and the registration's ports as writers' outputs. -/
structure Ctx where
  digest : String
  wiring : Wires.Wiring
  summary : Wires.Summary
  firsts : Array (String × Nat × Nat)
  members : Array Member
  writers : Std.HashMap String Flock.RegWired.Writer

/-- The parameters' `proofs.wires`: the window's wiring (`Wires.Wiring.ofJson`), refused unless `Wires.check` accepts it,
with its canonical JSON's SHA-512. -/
def specOf (j : Json) : V (String × Wires.Wiring × Wires.Summary) := do
  let c ← (Wires.Wiring.ofJson j).mapError (s!"proofs: wires: {·}")
  match Wires.check c with
  | .ok s => pure (← canonSha512 j, c, s)
  | .error r =>
    throw s!"proofs: wires: the wiring is refused ({r.code.name}{(r.wire.map (s!", wire {·}")).getD ""})"

/-- The registration's ports (`roots`) as writers' outputs: each root as the registration holds it, its leaves and domain
the verifier's own `ports` (R3 holds the registration's to them); a port either leaves out is none. A port the
registration names twice is refused: a port is one root. -/
def writersOf (record : Json) (ports : Array (String × Json)) : V (Std.HashMap String Flock.RegWired.Writer) := do
  let own : Std.HashMap String Json := ports.foldl (fun m (p, v) => m.insert p v) {}
  let roots := match getD record "roots" with | .arr rs => rs | _ => #[]
  let mut seen : Std.HashSet String := {}
  let mut out : Std.HashMap String Flock.RegWired.Writer := {}
  for r in roots do
    let .str port := getD r "port" | continue
    if seen.contains port then throw s!"wired reads: the registration names port {port} twice: a port is one root"
    seen := seen.insert port
    match getD r "root", own[port]? with
    | .str root, some v =>
      match Flock.unhex root, nat? (getD v "leaves"), getD v "domain" with
      | .ok rb, some leaves, .str d =>
        if let .ok db := Flock.unhex d then out := out.insert port { name := port, root := rb, domain := db, leaves }
      | _, _, _ => pure ()
    | _, _ => pure ()
  pure out

/-- The context (module docstring, `ctxOf`): the wiring `spec`, the templates' first units from the verifier's own program
and query (`Partition.evaluate`), each template the wiring names one of them with the wiring's units, and the members
and registration's ports. -/
def ctxOf (spec : String × Wires.Wiring × Wires.Summary) (program : Program) (query : Json) (record : Json)
    (ports : Array (String × Json)) (members : Array Member) : V Ctx := do
  let rs ← Partition.evaluate program query
  let firsts := rs.map fun r => (r.template, r.first, r.count)
  for (tid, t) in spec.2.1.templates do
    match firsts.filter (·.1 == tid) with
    | #[(_, _, n)] =>
      if n != t.units then
        throw s!"proofs: wires: template {tid} has {t.units} units, and the verifier's partition {n}"
    | _ => throw s!"proofs: wires: template {tid} is not one template of the verifier's partition"
  pure { digest := spec.1, wiring := spec.2.1, summary := spec.2.2, firsts, members,
         writers := ← writersOf record ports }

/-- Row `r` of template `tid`'s port `p`: the port itself when uncut, else `p#r` (`interface.row_name`). -/
def rowName (c : Wires.Wiring) (tid p : String) (r : Nat) : String :=
  if c.cutsOf tid p == 1 then p else s!"{p}#{r}"

/-- Population unit `g`'s template, its first unit and its index among the template's units. -/
def templateAt (ctx : Ctx) (g : Nat) : Option (String × Nat × Nat) :=
  (ctx.firsts.find? fun (_, lo, n) => lo ≤ g && g < lo + n).map fun (t, lo, _) => (t, lo, g - lo)

/-- How many of `units` lie below `g`: `g`'s position in its member's rows, which are its units' rows in order. -/
def rank (units : Array (Nat × Nat)) (g : Nat) : Nat := units.foldl (fun acc (a, b) => acc + (min b g - min a g)) 0

/-- The units in `units`. -/
def count (units : Array (Nat × Nat)) : Nat := units.foldl (fun acc (a, b) => acc + (b - a)) 0

/-- Whether unit `g` is in one of `units`. -/
def within (units : Array (Nat × Nat)) (g : Nat) : Bool := units.any fun (a, b) => a ≤ g && g < b

/-- The read of the row a wire fills from unit `v` of template `tid`'s output `out`: its member's output root, as the
registration holds it with the member's units as leaves, and its rank among the member's units. -/
def readOf (ctx : Ctx) (row : String) (tid : String) (v : Int) (out : String) : V Flock.RegWired.Read := do
  let some (_, lo, _) := ctx.firsts.find? (·.1 == tid)
    | throw s!"wired reads: template {tid}, which writes row {row}, is not a template of the verifier's partition"
  let g := lo + v.toNat
  let some m := ctx.members.find? (within ·.units g)
    | throw s!"wired reads: unit {v} of {tid}, which writes row {row}, is in no member"
  let port := s!"{m.name}/{out}"
  let some w := ctx.writers[port]?
    | throw s!"wired reads: the registration commits no root for {port}, which writes row {row}"
  if w.leaves != count m.units then
    throw s!"wired reads: {port}'s root has {w.leaves} leaves, and member {m.name} {count m.units} units"
  pure { writer := w, pos := rank m.units g }

/-- **A session's wired reads** (module docstring, `derive`): each row of the member's template that a wire meeting the
member's units fills, with one read per session instance of statement `st` (`readOf`, at the writer `Summary.writerOf`
gives the instance's unit's row), the instance's unit its `units.indices`, which the verifier derived. -/
def derive (ctx : Ctx) (m : Member) (st : Flock.Stmt) : V (Array Flock.RegWired.Port) := do
  if !Flock.Registered.derived st.partition then
    throw "wired reads: the statement's units are taken as stated, and a wired row's writer binds only as far as its unit does: the verifier derives them (--partition, --program)"
  let some (lo0, _) := m.units[0]? | throw s!"wired reads: member {m.name} holds no unit"
  let some (tid, first, _) := templateAt ctx lo0 | throw s!"wired reads: member {m.name}'s unit {lo0} is in no template"
  let ws := ctx.summary.wires.filter fun w =>
    w.reader == tid && m.units.any fun (a, b) => first + w.start < b && a < first + w.start + w.readerUnits
  if let some w := ws.find? fun w => m.units.any fun (a, b) => a < first + w.start || first + w.start + w.readerUnits < b then
    throw s!"wired reads: a wire fills port {w.port} of only some of member {m.name}'s units: members are cut at every wire's ends"
  let indices ← Flock.Registered.unitIndices st.pub
  let units ← indices.mapM fun g => do
    if !within m.units g then throw s!"wired reads: the statement's unit {g} is not member {m.name}'s"
    pure (g - first)
  let rows := ws.flatMap fun w => (List.range w.cuts).map fun j => (w.port, w.cut0 + j)
  rows.toArray.mapM fun (p, r) => do
    let name := rowName ctx.wiring tid p r
    let reads ← units.mapM fun u => do
      let some (wt, v, out) := ctx.summary.writerOf tid u p r
        | throw s!"wired reads: no wire fills row {name} of unit {u} of {tid}"
      readOf ctx name wt v out
    pure { name, reads }

/-- **The member's bound ports are the registration's** (module docstring, `boundHeld`): each bound port's root the
registration's for `<member>/<port>` and its domain the verifier's own. -/
def boundHeld (ctx : Ctx) (m : Member) (bound : Array Flock.RegBound.Bound) : V Unit :=
  bound.forM fun b => do
    let port := s!"{m.name}/{b.name}"
    match ctx.writers[port]? with
    | none => throw s!"wired reads: member {m.name} binds port {b.name}, and the registration commits no root for {port}"
    | some w =>
      if b.root != w.root || b.domain != w.domain then
        throw s!"wired reads: member {m.name} binds port {b.name} to another root or domain than the registration's {port}"

/-! The writer's rank and a member's units. -/

#guard rank #[(5, 6), (0, 2)] 0 == 0
#guard rank #[(5, 6), (0, 2)] 1 == 1
#guard rank #[(5, 6), (0, 2)] 5 == 2
#guard rank #[(10, 14)] 12 == 2
#guard count #[(5, 6), (0, 2)] == 3
#guard within #[(5, 6), (0, 2)] 1 && !within #[(5, 6), (0, 2)] 2

end SampledProofs.WiredReads
