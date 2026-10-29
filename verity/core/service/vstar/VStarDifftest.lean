import Difftest
import VStar
import Flock.RecResiduals
import Flock.RecSetup

/-!
# V*'s difftest executable

The functions `verity_flock`'s builders are held to (`tests/test_vstar_difftest.py`), each answering for one part of V*'s
circuits on the parameters it is given: a part's rows as a laid-out netlist (`sha`, `climb`, `residuals`: the netlist's
SHA-256, bytes and rows), a unit's lowering text (`rec_open_unit`, `inner_unit`), and a statement's whole circuit file
(`rec_open_circuit`, `inner_circuit`: its SHA-512 and bytes); the algebra's structure and its parts (`structure`,
`spec`, `parts`), and V*'s statements over an inner session's shape (`statements`); `spec_holo`, `statements_holo`,
`inner_circuit_holo` and `verifier_parts_holo` are those over a holo inner session (`VStar.Holo`). `verifier_parts` holds the Lean
verifier's `rec-residuals/…` nets (`Flock.RecResiduals`) to V*'s units, and `verifier_algebra` the verifier's own algebra
(`Flock.RecAlgebra`, its parts and their nets, holo shapes included) to `rec_algebra` and `rec_residuals`. `inner_draw`
holds `rec_live.inner_draw`, the prover's reading of the inner session's draw, to the verifier's (`Flock.RecSetup.innerDrawn`).
`link_unit` holds V*'s link unit (`VStar.linkUnit`) to `rec_link.lowering`, and `link_tree` the verifier's rows trees
(`Flock.RecLinkTree`) to `rec_link.Tree`.
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

def circuitJson (u : UnitC) (sub : Sub) (o : String → Json) (holo : Bool := false) : Except String Json := do
  let l ← u.laid
  let b : Bindings := ⟨← jStr (o "partition") "partition", ← jStr (o "program") "program", o "program_digests"⟩
  let e ← emitCircuitOf holo hashPut u l sub b
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

/-- `circuit.compose` of `InnerRepCheck{S}`'s lowering under the given bindings; `holo`: with META `holo`
(`rec_residuals.statement(..., holo=True)`). -/
def innerCircuit (holo : Bool) (j : Json) : Except String Json := do
  let o ← input' j ["S", "name", "partition", "program", "program_digests"]
  let S ← RepS.ofJson (o "S")
  let name ← jStr (o "name") "name"
  circuitJson (← innerUnit S name) (innerSub S name) o holo

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

/-- `rec_residuals.spec(Shape(m, k_log, claims, holo=holo))`: the structure's summary, the ports, and the whole S's
SHA-256. -/
def specFn (holo : Bool) (j : Json) : Except String Json := do
  let S ← if holo then Algebra.specHolo (← shapeOf j) else Algebra.spec (← shapeOf j)
  pure (Json.mkObj (("S", ← jsonSha256 S.toJson) :: ("ports", Json.arr (S.ports.toArray.map fun (nm, n) =>
    Json.arr #[(nm : Json), (n : Json)])) :: (← gfSummary S.gf)))

/-- `rec_residuals.parts(S, max_regions, max_slots, max_lone_slots)`, each part whole. -/
def partsFn (j : Json) : Except String Json := do
  let o ← input' j ["S", "max_regions", "max_slots", "max_lone_slots"]
  let ps ← parts (← RepS.ofJson (o "S")) (← jNat (o "max_regions") "max_regions") (← jNat (o "max_slots") "max_slots")
    (← jNat (o "max_lone_slots") "max_lone_slots")
  pure (Json.arr (ps.map fun p => Json.mkObj [("S", p.S.toJson), ("res", natsJson p.res), ("v", natsJson p.v)]))

/-- `rec_vstage.stage`'s statements over an inner session of shape `(m, k_log, claims)`, holo or plain: each level's
`RecOpen` and each algebra part's S (its SHA-256, sizes, residuals and v elements). -/
def statementsFn (holo : Bool) (j : Json) : Except String Json := do
  let st ← statementsOf (← shapeOf j) holo
  let out ← st.mapM fun s => match s with
    | .recOpen level L H n => pure (Json.mkObj [("name", s.name), ("dir", s.dir), ("instances", (n : Json)),
        ("level", (level : Json)), ("LANES", (L : Json)), ("H", (H : Json))])
    | .inner _ _ p => do pure (Json.mkObj ([("name", (s.name : Json)), ("dir", (s.dir : Json)), ("instances", (s.instances : Json)),
        ("S", ((← jsonSha256 p.S.toJson) : Json)), ("ports", (p.S.ports.length : Json)), ("res_of", natsJson p.res),
        ("v_of", natsJson p.v)] ++ (← gfSummary p.S.gf)))
    | s => throw s!"{s.name}: not one of statements' kinds"
  pure (Json.arr out)

/-- Each of V*'s algebra statements over an inner session of shape `(m, k_log, claims)`, holo or plain: its name and
whether the Lean verifier's net for it (`Flock.RecResiduals`, which a `rec-residuals/…` unit must equal; at a holo shape
the one a statement whose META says `holo` is held to) is V*'s unit's; and whether the verifier refuses the part past the
last. -/
def verifierPartsFn (holo : Bool) (j : Json) : Except String Json := do
  let sh ← shapeOf j
  let st ← statementsOf sh holo
  let some fsh0 := Flock.RecResiduals.shape? sh.m sh.kLog sh.claims | throw "the verifier refuses a shape V* takes"
  let fsh := { fsh0 with holo }
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

/-- `rec_live.inner_draw` on an inner session's record: its draw, or the verifier's refusal of a session over it
(`Flock.RecSetup.innerDrawn`, the same at the inner statement's files and at the verifier's inner coins). -/
def innerDraw (j : Json) : Except String Json := do
  let o ← input' j ["record"]
  let record ← jStr (o "record") "record"
  pure <| match Flock.RecSetup.innerDrawn (some (.coins Json.null record)) with
    | .ok () => Json.mkObj [("draw", (Flock.RecCoins.recordDraw record).getD Json.null)]
    | .error e => Json.mkObj [("refused", (e : Json))]

def hexB (j : Json) (ctx : String) : Except String ByteArray := do Flock.unhex (← jStr j ctx)

def f128 (j : Json) (ctx : String) : Except String Flock.F128 := do Flock.f128OfHex (← jStr j ctx)

def arrOf {α : Type} (f : Json → String → Except String α) (j : Json) (ctx : String) : Except String (Array α) := do
  (← jArr j ctx).mapM (f · ctx)

def pair (j : Json) (ctx : String) : Except String (Json × Json) := do
  let a ← jArr j ctx
  unless a.size == 2 do throw s!"{ctx}: a pair"
  pure (a[0]!, a[1]!)

/-- `VStar.In.Src` as the test writes it: F elements and bytes as hex (`{hi:016x}{lo:016x}` an element), a message row and
a top node as `[words, salt]` and `[node, salt]`, the fresh salts as `{dir: {port: [salt, ...]}}`. -/
def srcOf (j : Json) : Except String In.Src := do
  let o ← jObj j ["msgs", "openings", "tops", "acc_salts", "kappa", "eq_a", "vs", "fresh"] "src"
  let row : Json → String → Except String In.Row := fun x ctx => do
    let (w, s) ← pair x ctx
    pure { words := ← arrOf jNat w ctx, salt := ← hexB s ctx }
  let opening : Json → String → Except String In.Opening := fun x ctx => do
    let q ← jObj x ["rep", "level", "j", "pos", "row", "sibs"] ctx
    pure { rep := ← jNatAt q "rep" ctx, level := ← jNatAt q "level" ctx, j := ← jNatAt q "j" ctx,
            pos := ← jNatAt q "pos" ctx, row := ← arrOf f128 (q "row") ctx, sibs := ← arrOf hexB (q "sibs") ctx }
  let node : Json → String → Except String (ByteArray × ByteArray) := fun x ctx => do
    let (n, s) ← pair x ctx
    pure (← hexB n ctx, ← hexB s ctx)
  let fj := o "fresh"
  let fresh : String → String → Nat → ByteArray := fun d p i => match (fj.getObjValD d).getObjValD p with
    | .arr a => match a[i]? with
      | some (.str s) => (Flock.unhex s).toOption.getD .empty
      | _ => .empty
    | _ => .empty
  pure { msgs := ← arrOf (arrOf row) (o "msgs") "msgs", openings := ← arrOf opening (o "openings") "openings",
         tops := ← arrOf (arrOf (arrOf node)) (o "tops") "tops", accSalts := ← arrOf hexB (o "acc_salts") "acc_salts",
         kappa := ← arrOf (arrOf (arrOf (arrOf f128))) (o "kappa") "kappa",
         eqA := ← arrOf (arrOf (arrOf f128)) (o "eq_a") "eq_a", vs := ← arrOf (arrOf f128) (o "vs") "vs", fresh }

/-- Bytes' SHA-512, its first 8 bytes as hex: one row's or one salt's fingerprint. -/
def short (b : ByteArray) : Json := Flock.hexOf ((Flock.Sha512.hash b).extract 0 8)

def u16le (ws : Array Nat) : ByteArray := ws.foldl (fun b w => (b.push (w % 256).toUInt8).push (w / 256 % 256).toUInt8) .empty

/-- One V* part's port inputs (`VStar.In`) on the stager's inputs `src`: `stmt` a level's statement (`{"kind": "level",
"level", "LANES", "H", "n"}`), a part of the algebra's (`{"kind": "inner", "ports": [[name, n], ..], "v": [..]}`) or a
cap's (`{"kind": "cap", "level", "N"}`), `prog` its program's SHA-512 hex. Each port, in the circuit's order, as its table
rows' and salts' fingerprints (`short` of a row's u16 little-endian words, of a salt) and its refs. -/
def partInFn (j : Json) : Except String Json := do
  let o ← input' j ["src", "prog", "stmt"]
  let src ← srcOf (o "src")
  let prog ← jStr (o "prog") "prog"
  let st := o "stmt"
  let p ← match ← jStr (st.getObjValD "kind") "stmt.kind" with
    | "level" => do
      let q ← jObj st ["kind", "level", "LANES", "H", "n"] "stmt"
      In.levelIn src prog (← jNatAt q "level" "stmt") (← jNatAt q "LANES" "stmt") (← jNatAt q "H" "stmt") (← jNatAt q "n" "stmt")
    | "inner" => do
      let q ← jObj st ["kind", "ports", "v"] "stmt"
      let ports ← (← jArr (q "ports") "ports").mapM fun x => do
        let (nm, n) ← pair x "port"
        pure (← jStr nm "port", ← jNat n "port")
      In.algebraIn src prog ports.toList (← arrOf jNat (q "v") "v").toList
    | "cap" => do
      let q ← jObj st ["kind", "level", "N"] "stmt"
      In.capIn src (← jNatAt q "level" "stmt") (← jNatAt q "N" "stmt")
    | k => throw s!"stmt.kind: level, inner or cap, not {k}"
  let port : String × In.PortIn → Json := fun (nm, q) => Json.arr #[(nm : Json), Json.mkObj [
    ("rows", Json.arr (q.rows.map fun (r : In.Row) => short (u16le r.words))),
    ("salts", Json.arr (q.rows.map fun (r : In.Row) => short r.salt)), ("refs", natsJson q.refs.toList)]]
  pure (Json.mkObj [("ports", Json.arr (p.map port))])

/-- A rep's tops in the order `VStar.In.Src.tops` lists them: `level0`, the node count of level 0's top (`root_B`, at
`fast100(m)`'s first cap), then `later`, each later level's top's, in the order the rep's schedule commits them
(`RecVStar.topSizes` of `repRounds`, or `holoRepRounds` with `holo`). -/
def topSizesFn (j : Json) : Except String Json := do
  let o ← input' j ["m", "k_log", "claims", "holo"]
  let m ← jNatAt o "m" "input"
  let kLog ← jNatAt o "k_log" "input"
  let claims ← jNatAt o "claims" "input"
  let holo ← match o "holo" with
    | .bool b => pure b
    | _ => throw "input.holo: not a boolean"
  let some s := Flock.fast100 m | throw s!"m = {m}: no fast100 schedule"
  let some rs := (if holo then Flock.RecVStar.holoRepRounds else Flock.RecVStar.repRounds) m kLog claims
    | throw s!"m = {m}, k_log = {kLog}: no rep schedule"
  pure (Json.mkObj [("level0", ((2 ^ (s.levels.getD 0 default).cap : Nat) : Json)),
    ("later", natsJson (Flock.RecVStar.topSizes rs))])

/-- `rec_link.lowering(UB, PB).text`. -/
def linkLowering (j : Json) : Except String Json := do
  let o ← input' j ["UB", "PB"]
  unitJson (← linkUnit (← jNat (o "UB") "UB") (← jNat (o "PB") "PB"))

/-- `rec_link.Tree` at depth `UB + PB` over `rows`, `[unit, row, b ‖ c]` each (`b ‖ c` hex), in increasing order: its root
and the opening of `pos`, `[unit, row]` (`Flock.RecLinkTree.rowsRoot`, `rowsPath`). -/
def linkTree (j : Json) : Except String Json := do
  let o ← input' j ["UB", "PB", "rows", "pos"]
  let UB ← jNat (o "UB") "UB"
  let PB ← jNat (o "PB") "PB"
  let rows ← (← jArr (o "rows") "rows").mapM fun x => do
    let a ← jArr x "row"
    unless a.size == 3 do throw "row: [unit, row, bc]"
    pure ((← jNat a[0]! "unit", ← jNat a[1]! "row"), ← hexB a[2]! "bc")
  let p ← jArr (o "pos") "pos"
  unless p.size == 2 do throw "pos: [unit, row]"
  let root ← Flock.RecLinkTree.rowsRoot UB PB rows.toList
  let path ← Flock.RecLinkTree.rowsPath UB PB rows.toList (← jNat p[0]! "pos") (← jNat p[1]! "pos")
  pure (Json.mkObj [("root", (Flock.hexOf root : Json)), ("path", Json.arr (path.toArray.map fun s => (Flock.hexOf s : Json)))])

end VStar.DT

open VStar.DT in
def main : IO Unit := Difftest.serve (Difftest.table [
  ("sha", sha), ("climb", climb), ("residuals", residuals), ("rec_open_unit", recOpenLowering),
  ("rec_open_circuit", recOpenCircuit), ("inner_unit", innerLowering), ("inner_circuit", innerCircuit false),
  ("structure", structureFn), ("spec", specFn false), ("parts", partsFn), ("statements", statementsFn false),
  ("verifier_parts", verifierPartsFn false), ("verifier_algebra", verifierAlgebraFn), ("spec_holo", specFn true),
  ("statements_holo", statementsFn true), ("inner_circuit_holo", innerCircuit true),
  ("verifier_parts_holo", verifierPartsFn true), ("inner_draw", innerDraw), ("part_in", partInFn),
  ("top_sizes", topSizesFn), ("link_unit", linkLowering), ("link_tree", linkTree)])
