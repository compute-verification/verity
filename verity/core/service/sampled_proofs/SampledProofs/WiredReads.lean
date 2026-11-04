/-
A per-member one-stage audit's wired reads (`OneStageZk`, `proofs.wires`): which writer's committed output each drawn
unit's wired row reads, from the wires alone, and the statement's stated writers held to it.

The window's wiring (`Wires.Wiring`, `Wires.check`'s summary) says which writer `(template, unit, out)` fills each reader
row. The verifier's own partition gives each template's first unit, its parameters' members (each named by the window's
member key, `<template>/c<j>`) the member holding a writer unit, and the registration the root of each member's port
(`<member>/<port>`, its leaves and domain the verifier's own `ports`), all fixed before any coin:

* `derive`: a member's `--wired` reads (`Flock.RegWired.Port`): each row its units' wires fill, with the outputs of the
  writer members its wire's writers lie among (`Wires.Wire.span`). `Flock.RegWired.check` opens every table row of
  them against its stated writer's root, in the session's setup.
* `check`: on the session's drawn units, `Wires.Summary.drawnOk` with `same` the public check that the drawn instance's
  table row states the layout's writer (`<member>/<out>`) and position (the writer unit's rank among its member's units),
  computed from the wire per row (`Wires.Wire.writerAt`); the statement's units derived; each wired port the layout
  row's width; and every port's frame-v3-sha512 root and domain the registration's for its member, so the writer's
  statement proves the committed row its readers open.

Nothing here reads a row's value: only `b ‖ c`'s positions, roots, paths and registration names, which are public.
-/
import Flock
import SampledProofs.WiresDrawn
import SampledProofs.Partition

namespace SampledProofs.WiredReads

open Lean (Json)
open Flock (V)

/-- A member of the window: its name, the window's member key its ports are registered under, and its units. -/
structure Member where
  name : String
  units : Array (Nat × Nat)

/-- What a session's wired reads are derived and checked from: the wiring and its summary, each template's first unit
and count in the population, the window's members, and the registration's ports as writers' outputs. -/
structure Ctx where
  wiring : Wires.Wiring
  summary : Wires.Summary
  firsts : Array (String × Nat × Nat)
  members : Array Member
  writers : Std.HashMap String Flock.RegWired.Writer

/-- The parameters' `proofs.wires`: the window's wiring (`Wires.Wiring.ofJson`), refused unless `Wires.check` accepts it. -/
def specOf (j : Json) : V (Wires.Wiring × Wires.Summary) := do
  let c ← (Wires.Wiring.ofJson j).mapError (s!"proofs: wires: {·}")
  match Wires.check c with
  | .ok s => pure (c, s)
  | .error r =>
    throw s!"proofs: wires: the wiring is refused ({r.code.name}{(r.wire.map (s!", wire {·}")).getD ""})"

/-- The registration's ports (`roots`, the last naming each) as writers' outputs: each root as the registration holds it,
its leaves and domain the verifier's own `ports` (R3 holds the registration's to them). A port either leaves out is none. -/
def writersOf (record : Json) (ports : Array (String × Json)) : Std.HashMap String Flock.RegWired.Writer :=
  let own : Std.HashMap String Json := ports.foldl (fun m (p, v) => m.insert p v) {}
  let roots := match getD record "roots" with | .arr rs => rs | _ => #[]
  roots.foldl (fun m r =>
    match getD r "port", getD r "root" with
    | .str port, .str root =>
      match own[port]?, Flock.unhex root with
      | some v, .ok rb =>
        match nat? (getD v "leaves"), getD v "domain" with
        | some leaves, .str d =>
          match Flock.unhex d with
          | .ok db => m.insert port { name := port, root := rb, domain := db, leaves }
          | .error _ => m.erase port
        | _, _ => m.erase port
      | _, _ => m.erase port
    | _, _ => m) {}

/-- The context: the wiring `spec`, the templates' first units from the verifier's own program and query
(`Partition.evaluate`), each template the wiring names one of them, and the members and registration's ports. -/
def ctxOf (spec : Wires.Wiring × Wires.Summary) (program : Program) (query : Json) (record : Json)
    (ports : Array (String × Json)) (members : Array Member) : V Ctx := do
  let rs ← Partition.evaluate program query
  let firsts := rs.map fun r => (r.template, r.first, r.count)
  for (tid, _) in spec.1.templates do
    if (firsts.filter (·.1 == tid)).size != 1 then
      throw s!"proofs: wires: template {tid} is not one template of the verifier's partition"
  pure { wiring := spec.1, summary := spec.2, firsts, members, writers := writersOf record ports }

/-- Row `r` of template `tid`'s port `p`: the port itself when uncut, else `p#r` (`interface.row_name`). -/
def rowName (c : Wires.Wiring) (tid p : String) (r : Nat) : String :=
  if c.cutsOf tid p == 1 then p else s!"{p}#{r}"

/-- The u16 words of a row of template `tid`'s port `p`, cut as the wiring cuts it. -/
def rowWordsOf (c : Wires.Wiring) (tid p : String) : Nat :=
  match c.templates.lookup tid with
  | some t => match t.rows.lookup p with
    | some q => q.words / c.cutsOf tid p
    | none => 0
  | none => 0

/-- Population unit `g`'s template and its index among the template's units. -/
def templateAt (ctx : Ctx) (g : Nat) : Option (String × Nat) :=
  (ctx.firsts.find? fun (_, lo, n) => lo ≤ g && g < lo + n).map fun (t, lo, _) => (t, g - lo)

/-- How many of `units` lie below `g`: `g`'s position in its member's rows, which are its units' rows in order. -/
def rank (units : Array (Nat × Nat)) (g : Nat) : Nat := units.foldl (fun acc (a, b) => acc + (min b g - a)) 0

/-- Unit `u` of template `tid` as a writer: its member's key and its position among the member's units. -/
def writerAt (ctx : Ctx) (tid : String) (u : Int) : Option (String × Nat) :=
  match ctx.firsts.find? (·.1 == tid) with
  | none => none
  | some (_, lo, n) =>
    if u < 0 || n ≤ u.toNat then none
    else
      let g := lo + u.toNat
      (ctx.members.find? fun m => m.units.any fun (a, b) => a ≤ g && g < b).map fun m => (m.name, rank m.units g)

/-- The registration ports of the outputs `w`'s writers may be: `<member>/<out>` of each member of its writer template
that its writers' span (`Wires.Wire.span`) meets. -/
def writerPorts (ctx : Ctx) (w : Wires.Wire) : Array String :=
  match ctx.firsts.find? (·.1 == w.writer) with
  | none => #[]
  | some (_, lo, _) =>
    let ga := lo + w.span.1.toNat
    let gb := lo + w.span.2.toNat
    (ctx.members.filter fun m => m.units.any fun (a, b) => a ≤ gb && ga < b).map (s!"{·.name}/{w.out}")

/-- **The member's wired reads** (`Flock.RegWired`'s `--wired` file): each row of its template that a wire meeting its
units fills, with the registration's roots of the writer members' outputs that wire's writers lie among. -/
def derive (ctx : Ctx) (m : Member) : V (Array Flock.RegWired.Port) := do
  let some (lo, _) := m.units[0]? | throw s!"wired reads: member {m.name} holds no unit"
  let some (tid, u0) := templateAt ctx lo | throw s!"wired reads: member {m.name}'s unit {lo} is in no template"
  let first := lo - u0
  let ws := ctx.summary.wires.filter fun w =>
    w.reader == tid && m.units.any fun (a, b) => first + w.start < b && a < first + w.start + w.readerUnits
  let rows := ws.flatMap fun w => (List.range w.cuts).map fun j => (rowName ctx.wiring tid w.port (w.cut0 + j), w)
  rows.toArray.mapM fun (name, w) => do
    let writers ← (writerPorts ctx w).mapM fun n => match ctx.writers[n]? with
      | some x => pure x
      | none => throw s!"wired reads: the registration commits no root for {n}, an output member {m.name}'s {name} reads"
    pure { name, writers }

/-- Why drawn instance `i` (unit `u` of template `tid`)'s wired row `r` of port `p`, filled by the wires from `wr`, is not
the row its statement states: no such port, another width, a writer in no member, or a stated writer or position not the
layout's. `es`: each wired port's stated writers and positions, per table row. -/
def rowWhy (ctx : Ctx) (st : Flock.Stmt) (es : Array (String × Array String × Array Nat)) (tid : String) (u i : Nat)
    (p : String) (r : Nat) (wr : String × Int × String) : Option String :=
  let name := rowName ctx.wiring tid p r
  match Flock.Registered.portOf st.c name, es.find? (·.1 == name) with
  | none, _ => some s!"wired reads: the layout wires row {name} of {tid}, which is no port of the statement's circuit"
  | some _, none => some s!"wired reads: the verifier's wired reads hold no port {name}, which the layout wires"
  | some q, some (_, ws, ps) =>
    let words := rowWordsOf ctx.wiring tid p
    if st.c.ports[q]!.bits != 16 * words then
      some s!"wired reads: port {name}'s rows are {st.c.ports[q]!.bits} bits, and the layout's row {16 * words}"
    else match writerAt ctx wr.1 wr.2.1 with
      | none => some s!"wired reads: unit {wr.2.1} of {wr.1}, which the layout wires {name} from, is in no member"
      | some (m, pos) =>
        let want := s!"{m}/{wr.2.2}"
        let ρ := Flock.RegWired.refOf st.pub st.c.ports.size q i
        if ws.getD ρ "" == want && ps.getD ρ 0 == pos then none
        else some s!"wired reads: unit {u} of {tid} reads {name} from {ws.getD ρ ""} at position {ps.getD ρ 0}, and the layout wires it from {want} at position {pos}"

/-- Why member `m`'s statement's ports are not the registration's: a port whose frame-v3-sha512 root or domain in the
public header is not the registration's root and the verifier's domain for `<m>/<port>`, or that the registration holds no
root for. -/
def heldWhy (ctx : Ctx) (m : Member) (st : Flock.Stmt) : Option String :=
  st.c.ports.findSome? fun p =>
    let reg := s!"{m.name}/{p.name}"
    match ctx.writers[reg]?, st.pub.fv3Bytes "roots" p.name, st.pub.fv3Bytes "domain_ids" p.name with
    | none, _, _ => some s!"wired reads: the registration commits no root for {reg}, member {m.name}'s port {p.name}"
    | some w, .ok r, .ok d =>
      if r == w.root && d == w.domain then none
      else some s!"wired reads: port {p.name}'s frame-v3-sha512 root or domain in the public file is not the registration's {reg}"
    | some _, _, _ => some s!"wired reads: the public file names no frame-v3-sha512 root and domain for port {p.name}"

/-- **The drawn units' wired rows are the layout's** (module docstring, `check`): the statement's units derived, its
ports the registration's (`heldWhy`), and `Wires.Summary.drawnOk` on the drawn units (`draw`'s, every instance without
one) with `same` the public check `rowWhy`, against member `m`'s wired reads `ports`; a refusal names the first row that
fails. -/
def check (ctx : Ctx) (m : Member) (ports : Array Flock.RegWired.Port) (draw : Option Json) (st : Flock.Stmt) :
    V Unit := do
  if !Flock.Registered.derived st.partition then
    throw "wired reads: the statement's units are taken as stated, and a wired row's writer binds only as far as its unit does: the verifier derives them (--partition, --program)"
  if let some why := heldWhy ctx m st then throw why
  let indices ← Flock.Registered.unitIndices st.pub
  let drawn ← match draw with
    | none => pure (List.range st.pub.n)
    | some j => do
      let d ← Flock.Draw.UnitDraw.ofJson j
      pure d.proved.toList
  let es ← ports.mapM fun p => do
    let (ws, ps, _) ← Flock.RegWired.entry st.pub p
    pure (p.name, ws, ps)
  let units ← drawn.mapM fun i => do
    let some g := indices[i]? | throw s!"wired reads: the draw draws instance {i}, which units.indices does not name"
    let some tu := templateAt ctx g | throw s!"wired reads: unit {g} is in no template of the verifier's partition"
    pure (tu, i)
  let why := fun ((tid, u, p, r) : Wires.Key) (wr : String × Int × String) =>
    match units.lookup (tid, u) with
    | some i => rowWhy ctx st es tid u i p r wr
    | none => some s!"wired reads: unit {u} of {tid} is no drawn instance"
  if ctx.summary.drawnOk (fun k wr => (why k wr).isNone) (units.map (·.1)) then pure ()
  else
    let bad := (units.map (·.1)).findSome? fun (tid, u) =>
      (ctx.summary.rowsAt tid u).findSome? fun ((p, r), wr) => why (tid, u, p, r) wr
    throw (bad.getD "wired reads: a drawn unit's wired row is not its writer's")

end SampledProofs.WiredReads
