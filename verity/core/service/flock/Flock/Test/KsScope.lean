/-
`KsScope`: the `--zk` statement passes `tagsWhy`, and a typed statement or one of public outputs is refused; every named
statement that passes it reads its record's sample (`hm96Rows`), the draw `verify --zk` hands `checkOf`. A unit of
one input group that returns the outputs passes `filesWhy`, and with a `sha512/row-seg/v1` output row too; with that row
one compression short, or an output row of two ports, it is refused, and so are a unit reading two input groups and a
tail stage returning the outputs (both in `scopeOkI`, which `ProvedScope.check` admits under hidden outputs). A
shared-row public file in the form `SampledKSLiveV5` covers (two input ports' tables, its circuit's public input ports
parsing, the file loading over them, no two roots one) passes `sharedWhy` and the whole check; one whose circuit's
public input ports do not parse, one that does not load and a per-instance file are refused. At one table any sample
passes `drawWhy`; at more, a subset sample whose `k` the tables divide passes, and a sample of `k` they don't divide, a sample
that does not parse and no sample are refused. A templated statement's circuit, `Flock.Test.Tmpl`'s unit as
`Flock.Test.TmplScope` builds it, is held to `scopeOkT` on its template: it passes at one table and is refused at two,
with outputs a tail stage computes, and without a template.
-/
module

public import Flock.KsScope
public import Flock.Test.TmplScope
public import Flock.Test.Wires
meta import Flock.KsScope
meta import Flock.Test.TmplScope
meta import Flock.Test.Wires

@[expose] public section

namespace Flock.Test.KsScope

open Lean (Json toJson)
open Flock Flock.KsScope

#guard tagsWhy Tags.circuit == none
#guard (tagsWhy Tags.circuitTypes).isSome
#guard (tagsWhy Tags.circuitEb90718f).isSome
#guard (tagsWhy { Tags.circuit with retainRounds := false }).isSome
#guard (tagsWhy { Tags.circuit with merkleLeaf := "flock-leaf/sha512-unsalted" }).isSome
#guard Tags.all.all fun t => (tagsWhy t).isSome || t.hm96Rows

/-- A unit of one input group, returning the outputs. -/
def one : Circuit := Wires.circ #[Wires.net #[⟨0, 1, #[1]⟩] 1 #[1] #[(#[0], #[0])]] #[1] #[]

#guard ProvedScope.scopeOkH one
#guard filesWhy Tags.circuit 1 one default ByteArray.empty == none
#guard ProvedScope.scopeOkI (Wires.rms 0) && filesWhy Tags.circuit 1 (Wires.rms 0) default ByteArray.empty == some SCOPE_WHY
#guard ProvedScope.scopeOkI Wires.tail && filesWhy Tags.circuit 1 Wires.tail default ByteArray.empty == some SCOPE_WHY

/-- `one` with a `sha512/row/v2` input row `x` and a `sha512/row-seg/v1` output row `y` of 1,024 bits in `chunks`
compressions, its one segment's two and none past them when well formed. -/
def segOut (chunks : Nat) : Circuit :=
  { one with
    ports := #[{ name := "x", words := 64, chunks := 1, nb := 0, bits := 1024, v2 := true },
      { name := "y", words := 64, chunks, nb := 0, bits := 1024, v2 := true, segs := some (HmRow.segShape 1024).1 }],
    outRowPorts := #[1] }

#guard ProvedScope.scopeOkH (segOut 2) && filesWhy Tags.circuit 1 (segOut 2) default ByteArray.empty == none
#guard !ProvedScope.scopeOkH (segOut 1) && filesWhy Tags.circuit 1 (segOut 1) default ByteArray.empty == some SCOPE_WHY
#guard (ProvedScope.scopeWhy true (segOut 1)).startsWith "rows that are neither" &&
  (ProvedScope.scopeWhy true (segOut 1)).endsWith "one output port per output row"
#guard !ProvedScope.scopeOkH { segOut 2 with outRowPorts := #[2] }

/-- `one` with two input ports, `x` and `y`. -/
def two : Circuit :=
  { one with ports := #[{ name := "x", words := 64, chunks := 1, nb := 1 }, { name := "y", words := 64, chunks := 1, nb := 1 }] }

/-- A shared-row public file of `two`'s one instance: `x`'s table of two rows and `y`'s of one, the instance reading row 1
of `x` and row 0 of `y`; `cut` bytes short of its length. -/
def shared (cut : Nat := 0) : ByteArray :=
  let header := Json.mkObj [("format", Json.str Tags.circuit.publicFormat), ("statement", Json.str Tags.circuit.statement),
    (Tags.circuit.publicCircuitKey, Json.str (hexOf two.sha)), ("instances", toJson 1), ("rows", Json.bool false),
    ("units", HmRow.unitsEntry two #[0]), ("shared_rows", Json.mkObj [("x", toJson 2), ("y", toJson 1)])]
  let rows := (Array.range (3 * HmRow.BC_BYTES)).map (·.toUInt8)
  let refs : Array UInt8 := #[1, 0, 0, 0, 0, 0, 0, 0]
  let body := ByteArray.mk (rows ++ refs)
  (header.compress ++ "\n").toUTF8 ++ body.extract 0 (body.size - cut)

/-- `checkOf`'s refusal of `c` under `tags` at the public file `f`. -/
def whyAt (tags : Tags) (c : Circuit) (f : ByteArray) (count : Nat) (draw : Option Json) : Option String :=
  match checkOf (.ok c) tags f count draw with
  | .ok () => none
  | .error e => some e

#guard ProvedScope.scopeOkH two
#guard ((HmOut.loadPublic Tags.circuit two (shared)).toOption.bind (·.tables)).map (fun t => (t.counts, t.starts, t.refs))
  == some (#[2, 1], #[0, 2], #[1, 0])
#guard sharedWhy Tags.circuit two (shared) == none
#guard whyAt Tags.circuit two (shared) 1 none == none
#guard ((sharedWhy Tags.circuit two (shared 1)).getD "").startsWith s!"{TABLES_WHY} that does not load"
#guard ((sharedWhy Tags.circuit { two with metaJ := Json.mkObj [("shared_reads", Json.arr #[])] } (shared)).getD
  "").startsWith s!"{TABLES_WHY} whose circuit's public input ports do not parse"
#guard (whyAt Tags.circuit { two with metaJ := Json.mkObj [("shared_reads", Json.arr #[])] } (shared) 1 none).isSome
#guard collidesT two default { counts := #[2, 1], starts := #[0, 2], refs := #[1, 0] } == false

def n (x : Nat) : Json := Lean.toJson x

def subset (k : Nat) : Json := Json.mkObj [("law", "subset"), ("population", n 8), ("k", n k),
  ("units", Json.arr ((Array.range k).map n))]

#guard drawWhy 1 none == none && drawWhy 1 (some (subset 3)) == none
#guard drawWhy 2 (some (subset 4)) == none && drawWhy 4 (some (subset 4)) == none
#guard (drawWhy 2 (some (subset 3))).isSome && (drawWhy 4 (some (subset 6))).isSome
#guard (drawWhy 2 (some (Json.mkObj [("law", "subset")]))).isSome
#guard (drawWhy 2 none).isSome

/-- `checkOf`'s refusal of `c` under `tags`, at an empty public file, which no statement loads. -/
def why (tags : Tags) (c : Circuit) (count : Nat) (draw : Option Json) : Option String :=
  whyAt tags c ByteArray.empty count draw

#guard why Tags.circuitTypes one 1 none == some (ProvedScope.refusal (tagsWhy Tags.circuitTypes).get!)
#guard (why Tags.circuit one 1 none).isSome

/-- `Flock.Test.Tmpl`'s unit as a templated circuit (`Circuit.tmpl`), as `TmplScope` builds it. -/
def tmpl : Option Circuit := TmplScope.unit.toOption.map fun tu => { (default : Circuit) with tmpl := some tu }

/-- `unitWhy` and `filesWhy` on `tmpl` (changed by `f`) at the templated statement's tags and `count` tables. -/
def tmplWhy (count : Nat) (f : Circuit → Circuit := id) : Option (Option String × Option String) :=
  tmpl.map fun c =>
    (unitWhy (TmplScope.tags true) count (f c), filesWhy (TmplScope.tags true) count (f c) default ByteArray.empty)

#guard tmpl.isSome
#guard tmplWhy 1 == some (none, none)
#guard tmplWhy 2 == some (some TMPL_WHY, some TMPL_WHY)
#guard tmplWhy 1 (fun c => { c with outNet := 1 }) == some (some TMPL_WHY, some TMPL_WHY)
#guard tmplWhy 1 (fun c => { c with tmpl := none }) == some (some TMPL_WHY, some TMPL_WHY)
#guard tmpl.map (unitWhy (TmplScope.tags false) 1) == some (some TMPL_WHY)
#guard unitWhy Tags.circuit 2 one == none

end Flock.Test.KsScope
