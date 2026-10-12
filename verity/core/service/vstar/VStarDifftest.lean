import Difftest
import VStar
import Flock.RecResiduals

/-!
# V*'s difftest executable

The functions `verity_flock`'s builders are held to (`tests/test_vstar_difftest.py`), each answering for one part of V*'s
circuits on the parameters it is given: a part's rows as a laid-out netlist (`sha`, `climb`, `residuals`: the netlist's
SHA-256, bytes and rows), a unit's lowering text (`rec_open_unit`, `inner_unit`), and a statement's whole circuit file
(`rec_open_circuit`, `inner_circuit`: its SHA-512 and bytes); the algebra's structure and its parts (`structure`,
`spec`, `parts`), and V*'s statements over an inner session's shape (`statements`). `verifier_parts` holds the Lean
verifier's `rec-residuals/…` nets (`Flock.RecResiduals`) to V*'s units, and `verifier_algebra` the verifier's own algebra
(`Flock.RecAlgebra`, its parts and their nets, holo shapes included) to `rec_algebra` and `rec_residuals`.
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

/-- A JSON value's SHA-256 as `json.dumps(sort_keys=True, separators=(",", ":"))` writes it. -/
def jsonSha256 (j : Json) : Except String String := do
  pure (Flock.hexOf (((({} : Sha256S).update (← pyDumps j).toUTF8)).finish))

def shapeOf (j : Json) : Except String Algebra.Shape := do
  let o ← input' j ["m", "k_log", "claims"]
  pure { m := ← jNat (o "m") "m", kLog := ← jNat (o "k_log") "k_log", claims := ← jNat (o "claims") "claims" }

def gfSummary (S : GfS) : Except String (List (String × Json)) := do
  let products : Nat := (S.res.map (·.mul.length)).foldl (· + ·) 0
  pure [("sha256", ← jsonSha256 S.toJson), ("nw", (S.nw : Json)), ("nv", (S.nv : Json)), ("ops", (S.ops.length : Json)),
    ("res", (S.res.length : Json)), ("products", (products : Json))]

def natsJson (xs : List Nat) : Json := Json.arr (xs.toArray.map fun (i : Nat) => (i : Json))

/-- `rec_algebra.structure(Shape(m, k_log, claims))[0]`: its SHA-256 and sizes. -/
def structureFn (j : Json) : Except String Json := do
  pure (Json.mkObj (← gfSummary (← Algebra.structure_ (← shapeOf j))))

/-- `rec_residuals.spec(Shape(m, k_log, claims))`: the structure's summary, the ports, and the whole S's SHA-256. -/
def specFn (j : Json) : Except String Json := do
  let S ← Algebra.spec (← shapeOf j)
  pure (Json.mkObj (("S", ← jsonSha256 S.toJson) :: ("ports", Json.arr (S.ports.toArray.map fun (nm, n) =>
    Json.arr #[(nm : Json), (n : Json)])) :: (← gfSummary S.gf)))

/-- `rec_residuals.parts(S, max_regions, max_slots, max_lone_slots)`, each part whole. -/
def partsFn (j : Json) : Except String Json := do
  let o ← input' j ["S", "max_regions", "max_slots", "max_lone_slots"]
  let ps ← parts (← RepS.ofJson (o "S")) (← jNat (o "max_regions") "max_regions") (← jNat (o "max_slots") "max_slots")
    (← jNat (o "max_lone_slots") "max_lone_slots")
  pure (Json.arr (ps.map fun p => Json.mkObj [("S", p.S.toJson), ("res", natsJson p.res), ("v", natsJson p.v)]))

/-- `rec_vstage.stage`'s statements over an inner session of shape `(m, k_log, claims)`: each level's `RecOpen` and each
algebra part's S (its SHA-256, sizes, residuals and v elements). -/
def statementsFn (j : Json) : Except String Json := do
  let st ← statements (← shapeOf j)
  let out ← st.mapM fun s => match s with
    | .recOpen level L H n => pure (Json.mkObj [("name", s.name), ("dir", s.dir), ("instances", (n : Json)),
        ("level", (level : Json)), ("LANES", (L : Json)), ("H", (H : Json))])
    | .inner _ _ p => do pure (Json.mkObj ([("name", (s.name : Json)), ("dir", (s.dir : Json)), ("instances", (s.instances : Json)),
        ("S", ((← jsonSha256 p.S.toJson) : Json)), ("ports", (p.S.ports.length : Json)), ("res_of", natsJson p.res),
        ("v_of", natsJson p.v)] ++ (← gfSummary p.S.gf)))
    | s => throw s!"{s.name}: not one of statements' kinds"
  pure (Json.arr out)

/-- Each of V*'s algebra statements over an inner session of shape `(m, k_log, claims)`: its name and whether the Lean
verifier's net for it (`Flock.RecResiduals`, which a `rec-residuals/…` unit must equal) is V*'s unit's; and whether the
verifier refuses the part past the last. -/
def verifierPartsFn (j : Json) : Except String Json := do
  let sh ← shapeOf j
  let st ← statements sh
  let some fsh := Flock.RecResiduals.shape? sh.m sh.kLog sh.claims | throw "the verifier refuses a shape V* takes"
  let mut out : Array Json := #[]
  let mut n := 0
  for s in st do
    if let .inner j name p := s then
      n := n + 1
      let mine ← netOf name (← (← innerUnit p.S name).laid)
      let (S, ns, nv) ← Flock.RecResiduals.part fsh j
      let theirs ← netOf (Flock.RecResiduals.unitName sh.m sh.kLog sh.claims j) (Flock.RecResiduals.unitLaid S ns nv)
      out := out.push (Json.mkObj [("name", (name : Json)), ("equal", (decide (mine = theirs) : Json))])
  pure (Json.mkObj [("parts", Json.arr out), ("past_refused", ((Flock.RecResiduals.part fsh n).toOption.isNone : Json))])

def recTerm : Flock.RecOpen.Term → Term
  | .w i => .w i
  | .v i => .v i
  | .wt i u => .wt i u

def algTerm : Flock.RecAlgebra.Term → Term
  | .w i => .w i
  | .v i => .v i
  | .wt i u => .wt i u

/-- The verifier's whole S (`RecAlgebra.structureOf`) in `gf2k`'s form. -/
def algGfS (S : Flock.RecAlgebra.GfS) : GfS :=
  { nw := S.nw, nv := S.nv, ops := (S.ops.map fun o => (o.map algTerm).toList).toList,
    res := (S.res.map fun r => ({ mul := r.mul.toList, add := (r.add.map algTerm).toList } : Res)).toList }

/-- A row port's name in `rec_residuals.spec`: `m<i>` for round `i`, then the running sums'. -/
def portName (rounds p : Nat) : String :=
  if p < rounds then s!"m{p}" else if p == rounds then "acc0" else "acc1"

/-- The Lean verifier's own algebra at `Shape(m, k_log, claims, holo)` (`Flock.RecAlgebra`, `Flock.RecResiduals`), as
`rec_residuals.spec` and `parts` give it: the whole S's summary, and per part its S as `InnerRepCheck`'s (row ports named
as `spec` names them), its residuals and v elements, and with `units` the net `RecResiduals.check` holds a unit to, as
`rec_residuals.lowering(p.S, name)` writes it. -/
def verifierAlgebraFn (j : Json) : Except String Json := do
  let o ← input' j ["m", "k_log", "claims", "holo", "units"]
  let (m, k, cl) := (← jNat (o "m") "m", ← jNat (o "k_log") "k_log", ← jNat (o "claims") "claims")
  let holo ← match o "holo" with | .bool b => pure b | _ => throw "holo: not a boolean"
  let units ← match o "units" with | .bool b => pure b | _ => throw "units: not a boolean"
  let some sh0 := Flock.RecResiduals.shape? m k cl | throw "the verifier refuses the shape"
  let sh := { sh0 with holo }
  let S ← Flock.RecAlgebra.structureOf sh
  let sizes := Flock.RecResiduals.portSizes sh
  let gs ← Flock.RecResiduals.groups S sizes
  let nr := (Flock.RecAlgebra.rounds sh).size
  let mut out : Array Json := #[]
  for i in [0:gs.size] do
    let (R, P, Vs) := gs[i]!
    let (pS, ns, nv) ← Flock.RecResiduals.part sh i
    let rep : RepS := { gf := { nw := ns.foldl (· + ·) 0, nv, ops := pS.ops.map (·.map recTerm),
                                res := pS.res.map fun r => ({ mul := r.mul, add := r.add.map recTerm } : Res) },
                        ports := (P.toList.zip ns).map fun (p, n) => (portName nr p, n) }
    let name := Flock.RecResiduals.unitName m k cl i
    let unit ← if units then do
        let r := (Flock.RecResiduals.unitNet' pS ns nv).run St.empty
        unitJson { name, inPorts := Flock.RecResiduals.portWords ns nv, outPorts := [Flock.RecResiduals.RES_OUT_WORDS],
                   st := r.2, outs := r.1.toArray }
      else pure Json.null
    out := out.push (Json.mkObj [("name", (name : Json)), ("S", ← jsonSha256 rep.toJson), ("res", natsJson R.toList),
      ("v", natsJson Vs.toList), ("unit", unit)])
  pure (Json.mkObj ([("parts", Json.arr out)] ++ (← gfSummary (algGfS S))))

end VStar.DT

open VStar.DT in
def main : IO Unit := Difftest.serve (Difftest.table [
  ("sha", sha), ("climb", climb), ("residuals", residuals), ("rec_open_unit", recOpenLowering),
  ("rec_open_circuit", recOpenCircuit), ("inner_unit", innerLowering), ("inner_circuit", innerCircuit),
  ("structure", structureFn), ("spec", specFn), ("parts", partsFn), ("statements", statementsFn),
  ("verifier_parts", verifierPartsFn), ("verifier_algebra", verifierAlgebraFn)])
