import Difftest
import VStar

/-!
# V*'s difftest executable

The functions `verity_flock`'s builders are held to (`tests/test_vstar_difftest.py`), each answering for one part of V*'s
circuits on the parameters it is given: a part's rows as a laid-out netlist (`sha`, `climb`, `residuals`: the netlist's
SHA-256, bytes and rows), a unit's lowering text (`rec_open_unit`, `inner_unit`), and a statement's whole circuit file
(`rec_open_circuit`, `inner_circuit`: its SHA-512 and bytes).
-/

namespace VStar.DT

open Lean (Json)
open Flock.HmNets

def input' (j : Json) (keys : List String) : Except String (String → Json) := jObj j keys "input"

def digestsJson (d : Digests) (rows : Nat) : Json :=
  Json.mkObj [("sha256", Flock.hexOf d.sha256.finish), ("bytes", (d.bytes : Json)), ("rows", (rows : Json))]

/-- A part laid out with input ports `ins` and its outputs as one port, as `IL.layout(C, ins, ["y"])` and
`IL.netlist(lay, "part")` lay out and write it. -/
def partNet (st : St) (ins : Array Nat) (outs : List Lin) : Except String Json := do
  if let some v := deadRow st outs.toArray then throw s!"AND {v} reaches no output"
  let l := layout st ins #[outs.length] outs.toArray
  pure (digestsJson ((emitNet hashPut "part" l none).run {}).2 l.useful)

/-- `rec_open._sha` on `bits` input bits after `pre` prefix bytes, from the chaining value `cv`. -/
def sha (j : Json) : Except String Json := do
  let o ← input' j ["cv", "bits", "pre"]
  let cv ← (← jArr (o "cv") "cv").mapM (jNat · "cv")
  let n ← jNat (o "bits") "bits"
  let pre ← jNat (o "pre") "pre"
  unless cv.size == 8 && cv.all (· < 2 ^ 64) do throw "cv: eight 64-bit words"
  unless n % 8 == 0 && pre % 128 == 0 do throw "a message of whole bytes after whole blocks"
  let r := (do let x ← input n; shaFrom (cv.toList.map (BitVec.ofNat 64 ·)) x pre).run St.empty
  partNet r.2 #[n] r.1

/-- `rec_open._climb` of a row of `row` bits over `H` siblings by the low bits of a 16-bit `d`. -/
def climb (j : Json) : Except String Json := do
  let o ← input' j ["H", "row"]
  let H ← jNat (o "H") "H"
  let n ← jNat (o "row") "row"
  unless H ≤ 16 && n % 8 == 0 do throw "at most 16 siblings, a row of whole bytes"
  let r := (do
    let row ← input n
    let aux ← input (512 * H)
    let d ← input 16
    climbRow H row aux d).run St.empty
  partNet r.2 #[n, 512 * H, 16] r.1

/-- `rec_residuals.residual_forms` on ports `w` and `v` of `S.nw` and `S.nv` elements. -/
def residuals (j : Json) : Except String Json := do
  let S ← GfS.ofJson j
  checkS S
  let r := (do
    let w ← input (128 * S.nw)
    let v ← input (128 * S.nv)
    residualForms S w.toArray v.toArray).run St.empty
  partNet r.2 #[128 * S.nw, 128 * S.nv] r.1

def unitJson (u : UnitC) : Except String Json := do
  let l ← u.laid
  pure (digestsJson (← unitDigests u l) l.useful)

def circuitJson (u : UnitC) (sub : Sub) (o : String → Json) : Except String Json := do
  let l ← u.laid
  let b : Bindings := ⟨← jStr (o "partition") "partition", ← jStr (o "program") "program", o "program_digests"⟩
  let e ← emitCircuit hashPut u l sub b
  let d := (e.run {}).2
  pure (Json.mkObj [("sha512", Flock.hexOf d.sha512.finish), ("bytes", (d.bytes : Json))])

def recOpenSub (LANES H : Nat) : Sub :=
  { id := s!"rec-open/l{LANES}-h{H}", template := "rec-open", inputs := ["row", "aux", "acc", "coef", "dirs"],
    outputs := ["out", "acc_out"] }

def innerSub (S : RepS) (name : String) : Sub :=
  { id := name, template := "rec-residuals", inputs := S.ports.map (·.1) ++ ["v"], outputs := ["out"] }

/-- `rec_outer.lowering(LANES, H).text`. -/
def recOpenLowering (j : Json) : Except String Json := do
  let o ← input' j ["LANES", "H"]
  unitJson (← recOpenUnit (← jNat (o "LANES") "LANES") (← jNat (o "H") "H"))

/-- `circuit.compose` of `RecOpen{LANES, H}`'s lowering under the given bindings. -/
def recOpenCircuit (j : Json) : Except String Json := do
  let o ← input' j ["LANES", "H", "partition", "program", "program_digests"]
  let L ← jNat (o "LANES") "LANES"
  let H ← jNat (o "H") "H"
  circuitJson (← recOpenUnit L H) (recOpenSub L H) o

/-- `rec_residuals.lowering(S, name).text`. -/
def innerLowering (j : Json) : Except String Json := do
  let o ← input' j ["S", "name"]
  unitJson (← innerUnit (← RepS.ofJson (o "S")) (← jStr (o "name") "name"))

/-- `circuit.compose` of `InnerRepCheck{S}`'s lowering under the given bindings. -/
def innerCircuit (j : Json) : Except String Json := do
  let o ← input' j ["S", "name", "partition", "program", "program_digests"]
  let S ← RepS.ofJson (o "S")
  let name ← jStr (o "name") "name"
  circuitJson (← innerUnit S name) (innerSub S name) o

end VStar.DT

open VStar.DT in
def main : IO Unit := Difftest.serve (Difftest.table [
  ("sha", sha), ("climb", climb), ("residuals", residuals), ("rec_open_unit", recOpenLowering),
  ("rec_open_circuit", recOpenCircuit), ("inner_unit", innerLowering), ("inner_circuit", innerCircuit)])
