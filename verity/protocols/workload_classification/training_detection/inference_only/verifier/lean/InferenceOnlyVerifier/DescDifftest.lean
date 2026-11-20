module

public import InferenceOnlyVerifier.Desc
public import InferenceOnlyVerifier.Difftest
meta import InferenceOnlyVerifier.Desc

/-!
# The difftest of the description-sized check: against the unrolled one, and against the Python reference

`scripts/desc_difftest_vectors.py` writes instances of `InferenceOnlyCheck.Desc` (θ and the owners per root node), each one expanded
gate by gate where it is well-formed, and `scripts/inference_only_reference.py`'s `check_desc` on each. `InferenceOnlyCheck/DescDifftestMain.lean`
fails unless `InferenceOnlyCheck.Desc.check` gives Python's result on every instance and `InferenceOnlyCheck.check`'s on every expansion: per
inference block its size and fresh input bits, the verdict and the first failing inference block, or the first malformation's code. The Lean audit runs
it (`runs` in `lean-audit.json`): compiled evaluation, a test and not a proof.
-/

@[expose] public section

namespace InferenceOnlyVerifier.DescDifftest

open Lean (Json)

/-- How `a` differs from `b`, one line per field of `InferenceOnlyCheck.Difftest.fields`. -/
def compare (i : Nat) (what : String) (a b : Json) : List String :=
  InferenceOnlyVerifier.Difftest.fields.filterMap fun k =>
    let x := a.getObjValD k
    let y := b.getObjValD k
    if x == y then none else some s!"instance {i}: {k}: Desc {x.compress}, {what} {y.compress}"

/-- How instance line `i` differs: `InferenceOnlyCheck.Desc.check` against Python's result and, on an expansion, against
`InferenceOnlyCheck.check` on it. -/
def differences (i : Nat) (inst exp python : String) : List String :=
  match Json.parse inst, Json.parse exp, Json.parse python with
  | .error e, _, _ => [s!"instance {i}: not JSON: {e}"]
  | _, .error e, _ => [s!"instance {i}: the expansion doesn't parse: {e}"]
  | _, _, .error e => [s!"instance {i}: Python's results don't parse: {e}"]
  | .ok j, .ok x, .ok py =>
    let desc := Desc.check j
    compare i "Python" desc py ++ (if x.isNull then [] else compare i "InferenceOnlyVerifier.check" desc (check x))

/-- `lake env lean --run InferenceOnlyCheck/DescDifftestMain.lean INSTANCES EXPANDED PYTHON N`: fails unless the three files have
`N` lines each and no line differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, exp, py, n] := args
    | IO.eprintln "usage: InferenceOnlyVerifier/DescDifftestMain.lean INSTANCES EXPANDED PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let e ← lines exp
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size ∨ a.size != e.size then
    IO.eprintln s!"desc difftest: {a.size} instances, {e.size} expansions and {b.size} results, not {n}"
    return 1
  let tasks := (List.range a.size).map fun i => Task.spawn fun _ => differences i a[i]! e[i]! b[i]!
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"desc difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

-- A grid part of two lanes of 524,289 periods each, 1,048,578 references, is summarized a lane a piece: it gets the
-- verdict the same grid gets at 2 periods, not `unwalked`.
#guard let big := r#"{"program":"{\"annotations\":{},\"boolean\":{\"Op8_v1\":\"StandIn[Op8]_v1\"},\"callees\":[\"RegisteredInput8_v1\",\"Op8_v1\",\"F_v1\",\"Const1[0x0]_v1\"],\"definitions\":{\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"F_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x0\",[\"Array\",1048578,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"F_v1\"},\"Op8_v1\":{\"form\":\"primitive\",\"id\":\"Op8_v1\",\"params\":[[\"a0\",[\"Value\",8]]],\"ret\":[\"Value\",8]},\"R_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"grid\":[1,[[\"n\",0,0,0,[[524289,0]]],[\"n\",1,0,0,[[524289,0]]]]]},\"type\":1}],\"fn\":2,\"form\":\"call\",\"out\":0}],\"params\":[],\"ret\":{\"affine\":[\"n\",2,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"R_v1\"},\"RegisteredInput8_v1\":{\"form\":\"primitive\",\"id\":\"RegisteredInput8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"StandIn[Op8]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":3,\"form\":\"primitive\",\"out\":2}],\"params\":[[\"a0\",[\"Array\",8,[\"Value\",1]]]],\"ret\":{\"explicit\":[[\"n\",0,0],[\"n\",0,0],[\"n\",0,0],[\"n\",0,0],[\"n\",0,0],[\"n\",0,0],[\"n\",0,0],[\"n\",0,0]]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[Op8]_v1\"}},\"program_fn\":\"R_v1\",\"root\":\"R_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",8],[\"Array\",1048578,[\"Value\",8]],[\"Value\",1]]}","theta":[],"owner":["input","input",0],"gamma":10,"alpha":[8,1]}"#
  match Json.parse big, Json.parse ((big.replace "1048578" "4").replace "524289" "2") with
  | .ok j, .ok k => Desc.check j == Desc.check k && (Desc.check j).getObjValD "error" != Json.str "unwalked"
  | _, _ => false

-- A whole lane of runs of 2 and two levels, the first repeating the run 1,048,577 times (a composite reading 4,194,308
-- references to a parameter of 4 leaves), is refused by name: `Piece.pisWalk`'s levels.
#guard match Json.parse r#"{"program":"{\"annotations\":{},\"boolean\":{\"Op_v1\":\"StandIn[Op]_v1\"},\"callees\":[\"Input8_v1\",\"Op_v1\",\"F_v1\",\"Const1[0x0]_v1\"],\"definitions\":{\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"F_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"grid\":[2,[[\"p\",0,0,1,[[1048577,0],[2,2]]]]]},\"type\":2}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x0\",[\"Array\",4,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"F_v1\"},\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Op_v1\":{\"form\":\"primitive\",\"id\":\"Op_v1\",\"params\":[[\"a0\",[\"Array\",4194308,[\"Value\",8]]]],\"ret\":[\"Value\",8]},\"R_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":4,\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,4]},\"type\":1}],\"fn\":2,\"form\":\"call\",\"out\":0}],\"params\":[],\"ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"R_v1\"},\"StandIn[Op]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":3,\"form\":\"primitive\",\"out\":4}],\"params\":[[\"a0\",[\"Array\",4194308,[\"Array\",8,[\"Value\",1]]]]],\"ret\":{\"strided\":[\"n\",0,0,1,0,0,8]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[Op]_v1\"}},\"program_fn\":\"R_v1\",\"root\":\"R_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",8],[\"Array\",4,[\"Value\",8]],[\"Array\",4194308,[\"Value\",8]],[\"Array\",4194307,[\"Value\",8]],[\"Value\",1]]}","theta":[],"owner":["input",0],"gamma":10,"alpha":[8,1]}"# with
  | .ok j => let r := Desc.check j; r.getObjValD "error" == Json.str "unwalked" &&
      ((r.getObjValD "message").getStr?.toOption.getD "").startsWith "intervals of a lane's levels"
  | .error _ => false

-- Part of such a lane, all of it but its first leaf (a call whose callee reads all but the first of its 4,194,308
-- leaves), is refused by name: `Piece.pisWalk`'s rows.
#guard match Json.parse r#"{"program":"{\"annotations\":{},\"boolean\":{\"Op_v1\":\"StandIn[Op]_v1\"},\"callees\":[\"Input8_v1\",\"Op_v1\",\"F_v1\",\"H_v1\",\"Const1[0x0]_v1\"],\"definitions\":{\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"F_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"grid\":[2,[[\"p\",0,0,1,[[1048577,0],[2,2]]]]]},\"type\":2}],\"fn\":3,\"form\":\"call\",\"out\":0}],\"params\":[[\"x0\",[\"Array\",4,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"F_v1\"},\"H_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,1,4194307]},\"type\":3}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x0\",[\"Array\",4194308,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"H_v1\"},\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Op_v1\":{\"form\":\"primitive\",\"id\":\"Op_v1\",\"params\":[[\"a0\",[\"Array\",4194307,[\"Value\",8]]]],\"ret\":[\"Value\",8]},\"R_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":4,\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,4]},\"type\":1}],\"fn\":2,\"form\":\"call\",\"out\":0}],\"params\":[],\"ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"R_v1\"},\"StandIn[Op]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":4,\"form\":\"primitive\",\"out\":4}],\"params\":[[\"a0\",[\"Array\",4194307,[\"Array\",8,[\"Value\",1]]]]],\"ret\":{\"strided\":[\"n\",0,0,1,0,0,8]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[Op]_v1\"}},\"program_fn\":\"R_v1\",\"root\":\"R_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",8],[\"Array\",4,[\"Value\",8]],[\"Array\",4194308,[\"Value\",8]],[\"Array\",4194307,[\"Value\",8]],[\"Value\",1]]}","theta":[],"owner":["input",0],"gamma":10,"alpha":[8,1]}"# with
  | .ok j => let r := Desc.check j; r.getObjValD "error" == Json.str "unwalked" &&
      ((r.getObjValD "message").getStr?.toOption.getD "").startsWith "rows of a read"
  | .error _ => false

-- Every other leaf of a call's output of 2,097,154 leaves, read by a root node, is refused by name: `down`'s call.
#guard match Json.parse r#"{"program":"{\"annotations\":{},\"boolean\":{\"Op_v1\":\"StandIn[Op]_v1\"},\"callees\":[\"Input8_v1\",\"Op_v1\",\"H_v1\",\"R_v1\",\"Const1[0x0]_v1\"],\"definitions\":{\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"H_v1\":{\"bindings\":{},\"body\":{\"nodes\":[],\"params\":[[\"x0\",[\"Array\",2097154,[\"Value\",8]]]],\"ret\":{\"affine\":[\"p\",0,0,2097154]},\"ret_type\":[\"Array\",2097154,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"H_v1\"},\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Op_v1\":{\"form\":\"primitive\",\"id\":\"Op_v1\",\"params\":[[\"a0\",[\"Array\",1048577,[\"Value\",8]]]],\"ret\":[\"Value\",8]},\"R_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2097154,\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,2097154]},\"type\":0}],\"fn\":2,\"form\":\"call\",\"out\":0},{\"args\":[{\"refs\":{\"strided\":[\"n\",1,0,1048577,0,2,1048577]},\"type\":1}],\"fn\":1,\"form\":\"primitive\",\"out\":2}],\"params\":[],\"ret\":{\"affine\":[\"n\",2,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"R_v1\"},\"StandIn[Op]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":4,\"form\":\"primitive\",\"out\":3}],\"params\":[[\"a0\",[\"Array\",1048577,[\"Array\",8,[\"Value\",1]]]]],\"ret\":{\"strided\":[\"n\",0,0,1,0,0,8]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[Op]_v1\"}},\"program_fn\":\"R_v1\",\"root\":\"R_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Array\",2097154,[\"Value\",8]],[\"Array\",1048577,[\"Value\",8]],[\"Value\",8],[\"Value\",1]]}","theta":[],"owner":["input",0,0],"gamma":10,"alpha":[8,1]}"# with
  | .ok j => let r := Desc.check j; r.getObjValD "error" == Json.str "unwalked" &&
      ((r.getObjValD "message").getStr?.toOption.getD "").startsWith "leaves of a call's output read one by one"
  | .error _ => false

-- A composite reading every other leaf of its parameter of 2,097,154 leaves is refused by name: its `used` as
-- intervals (`ivsCount`).
#guard match Json.parse r#"{"program":"{\"annotations\":{},\"boolean\":{\"Op_v1\":\"StandIn[Op]_v1\"},\"callees\":[\"Input8_v1\",\"Op_v1\",\"F_v1\",\"R_v1\",\"Const1[0x0]_v1\"],\"definitions\":{\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"F_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"strided\":[\"p\",0,0,1048577,0,2,1048577]},\"type\":0}],\"fn\":1,\"form\":\"primitive\",\"out\":1}],\"params\":[[\"x0\",[\"Array\",2097154,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"F_v1\"},\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Op_v1\":{\"form\":\"primitive\",\"id\":\"Op_v1\",\"params\":[[\"a0\",[\"Array\",1048577,[\"Value\",8]]]],\"ret\":[\"Value\",8]},\"R_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2097154,\"out\":2},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,2097154]},\"type\":2}],\"fn\":2,\"form\":\"call\",\"out\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"R_v1\"},\"StandIn[Op]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":4,\"form\":\"primitive\",\"out\":3}],\"params\":[[\"a0\",[\"Array\",1048577,[\"Array\",8,[\"Value\",1]]]]],\"ret\":{\"strided\":[\"n\",0,0,1,0,0,8]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[Op]_v1\"}},\"program_fn\":\"R_v1\",\"root\":\"R_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Array\",1048577,[\"Value\",8]],[\"Value\",8],[\"Array\",2097154,[\"Value\",8]],[\"Value\",1]]}","theta":[],"owner":["input",0],"gamma":10,"alpha":[8,1]}"# with
  | .ok j => let r := Desc.check j; r.getObjValD "error" == Json.str "unwalked" &&
      ((r.getObjValD "message").getStr?.toOption.getD "").startsWith "runs of a read, as intervals one by one"
  | .error _ => false

-- A call returning every other leaf of its parameter of 2,097,154 leaves, read whole by a root node, is refused by
-- name: `throughArg`.
#guard match Json.parse r#"{"program":"{\"annotations\":{},\"boolean\":{\"Op_v1\":\"StandIn[Op]_v1\"},\"callees\":[\"Input8_v1\",\"Op_v1\",\"H_v1\",\"R_v1\",\"Const1[0x0]_v1\"],\"definitions\":{\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"H_v1\":{\"bindings\":{},\"body\":{\"nodes\":[],\"params\":[[\"x0\",[\"Array\",2097154,[\"Value\",8]]]],\"ret\":{\"strided\":[\"p\",0,0,1048577,0,2,1048577]},\"ret_type\":[\"Array\",1048577,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"H_v1\"},\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Op_v1\":{\"form\":\"primitive\",\"id\":\"Op_v1\",\"params\":[[\"a0\",[\"Array\",1048577,[\"Value\",8]]]],\"ret\":[\"Value\",8]},\"R_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2097154,\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,2097154]},\"type\":0}],\"fn\":2,\"form\":\"call\",\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"n\",1,0,1048577]},\"type\":1}],\"fn\":1,\"form\":\"primitive\",\"out\":2}],\"params\":[],\"ret\":{\"affine\":[\"n\",2,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"R_v1\"},\"StandIn[Op]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":4,\"form\":\"primitive\",\"out\":3}],\"params\":[[\"a0\",[\"Array\",1048577,[\"Array\",8,[\"Value\",1]]]]],\"ret\":{\"strided\":[\"n\",0,0,1,0,0,8]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[Op]_v1\"}},\"program_fn\":\"R_v1\",\"root\":\"R_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Array\",2097154,[\"Value\",8]],[\"Array\",1048577,[\"Value\",8]],[\"Value\",8],[\"Value\",1]]}","theta":[],"owner":["input",0,0],"gamma":10,"alpha":[8,1]}"# with
  | .ok j => let r := Desc.check j; r.getObjValD "error" == Json.str "unwalked" &&
      ((r.getObjValD "message").getStr?.toOption.getD "").startsWith "leaves of a parameter taken through an argument"
  | .error _ => false

-- A batch of one member mapped over the inner axis of an argument of 1,048,577 rows of one leaf, whose member returns
-- its parameter whole, read whole by a root node, is refused by name: `throughArg`'s rows of a mapped axis.
#guard match Json.parse r#"{"program":"{\"annotations\":{},\"boolean\":{\"Op_v1\":\"StandIn[Op]_v1\"},\"callees\":[\"Input8_v1\",\"Op_v1\",\"H_v1\",\"Const1[0x0]_v1\"],\"definitions\":{\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"H_v1\":{\"bindings\":{},\"body\":{\"nodes\":[],\"params\":[[\"x0\",[\"Array\",1048577,[\"Value\",8]]]],\"ret\":{\"affine\":[\"p\",0,0,1048577]},\"ret_type\":[\"Array\",1048577,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"H_v1\"},\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Op_v1\":{\"form\":\"primitive\",\"id\":\"Op_v1\",\"params\":[[\"a0\",[\"Array\",1048577,[\"Value\",8]]]],\"ret\":[\"Value\",8]},\"R_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":1048577,\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,1048577]},\"type\":2}],\"axes\":[1],\"fn\":2,\"form\":\"batch\",\"n\":1,\"out\":3},{\"args\":[{\"refs\":{\"affine\":[\"n\",1,0,1048577]},\"type\":1}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[],\"ret\":{\"affine\":[\"n\",2,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"R_v1\"},\"StandIn[Op]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":3,\"form\":\"primitive\",\"out\":4}],\"params\":[[\"a0\",[\"Array\",1048577,[\"Array\",8,[\"Value\",1]]]]],\"ret\":{\"strided\":[\"n\",0,0,1,0,0,8]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[Op]_v1\"}},\"program_fn\":\"R_v1\",\"root\":\"R_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",8],[\"Array\",1048577,[\"Value\",8]],[\"Array\",1048577,[\"Array\",1,[\"Value\",8]]],[\"Array\",1,[\"Array\",1048577,[\"Value\",8]]],[\"Value\",1]]}","theta":[],"owner":["input",0,0],"gamma":10,"alpha":[8,1]}"# with
  | .ok j => let r := Desc.check j; r.getObjValD "error" == Json.str "unwalked" &&
      ((r.getObjValD "message").getStr?.toOption.getD "").startsWith "rows of a mapped axis a batch member's parameter"
  | .error _ => false

-- A batch of one member mapped over the inner axis of an argument of 1,048,578 rows given as two parts, whose member
-- reads its first leaf and all but its first two, is refused by name before `memberRows` stops at the first run:
-- `nodeReads`' runs of a mapped argument.
#guard match Json.parse r#"{"program":"{\"annotations\":{},\"boolean\":{\"Op_v1\":\"StandIn[Op]_v1\"},\"callees\":[\"Input8_v1\",\"Op_v1\",\"C_v1\",\"Const1[0x0]_v1\"],\"definitions\":{\"C_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"concat\":[{\"affine\":[\"p\",0,0,1]},{\"affine\":[\"p\",0,2,1048576]}]},\"type\":4}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x0\",[\"Array\",1048578,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"C_v1\"},\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Op_v1\":{\"form\":\"primitive\",\"id\":\"Op_v1\",\"params\":[[\"a0\",[\"Array\",1048577,[\"Value\",8]]]],\"ret\":[\"Value\",8]},\"R_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":1048578,\"out\":1},{\"args\":[{\"refs\":{\"concat\":[{\"affine\":[\"n\",0,524289,524289]},{\"affine\":[\"n\",0,0,524289]}]},\"type\":2}],\"axes\":[1],\"fn\":2,\"form\":\"batch\",\"n\":1,\"out\":3}],\"params\":[],\"ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Array\",1,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"R_v1\"},\"StandIn[Op]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":3,\"form\":\"primitive\",\"out\":5}],\"params\":[[\"a0\",[\"Array\",1048577,[\"Array\",8,[\"Value\",1]]]]],\"ret\":{\"strided\":[\"n\",0,0,1,0,0,8]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[Op]_v1\"}},\"program_fn\":\"R_v1\",\"root\":\"R_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",8],[\"Array\",1048578,[\"Value\",8]],[\"Array\",1048578,[\"Array\",1,[\"Value\",8]]],[\"Array\",1,[\"Value\",8]],[\"Array\",1048577,[\"Value\",8]],[\"Value\",1]]}","theta":[],"owner":["input",0],"gamma":10,"alpha":[8,1]}"# with
  | .ok j => let r := Desc.check j; r.getObjValD "error" == Json.str "unwalked" &&
      ((r.getObjValD "message").getStr?.toOption.getD "").startsWith "runs of a mapped argument a member reads"
  | .error _ => false

-- A batch of 1,048,577 members, each returning its two gates out of order, read whole with one gate inside it by a
-- root node of another inference block, is refused by name: `freshWalk`'s frame expanded copy by copy.
#guard match Json.parse r#"{"program":"{\"annotations\":{},\"boolean\":{\"Op_v1\":\"StandIn[Op]_v1\"},\"callees\":[\"Const1[0x0]_v1\",\"Op_v1\",\"G_v1\",\"R_v1\"],\"definitions\":{\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"G_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[],\"fn\":0,\"form\":\"primitive\",\"out\":0}],\"params\":[],\"ret\":{\"explicit\":[[\"n\",1,0],[\"n\",0,0]]},\"ret_type\":[\"Array\",2,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"G_v1\"},\"Op_v1\":{\"form\":\"primitive\",\"id\":\"Op_v1\",\"params\":[[\"a0\",[\"Array\",2097155,[\"Value\",1]]]],\"ret\":[\"Value\",8]},\"R_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":2,\"form\":\"batch\",\"n\":1048577,\"out\":1},{\"args\":[{\"refs\":{\"concat\":[{\"affine\":[\"n\",0,0,2097154]},{\"explicit\":[[\"n\",0,5]]}]},\"type\":2}],\"fn\":1,\"form\":\"primitive\",\"out\":3}],\"params\":[],\"ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"R_v1\"},\"StandIn[Op]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":0,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"a0\",[\"Array\",2097155,[\"Value\",1]]]],\"ret\":{\"strided\":[\"n\",0,0,1,0,0,8]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[Op]_v1\"}},\"program_fn\":\"R_v1\",\"root\":\"R_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",1],[\"Array\",1048577,[\"Array\",2,[\"Value\",1]]],[\"Array\",2097155,[\"Value\",1]],[\"Value\",8]]}","theta":[],"owner":[0,1],"gamma":10,"alpha":[8,1]}"# with
  | .ok j => let r := Desc.check j; r.getObjValD "error" == Json.str "unwalked" &&
      ((r.getObjValD "message").getStr?.toOption.getD "").startsWith "copies of a member's gates, expanded one by one"
  | .error _ => false

end InferenceOnlyVerifier.DescDifftest
