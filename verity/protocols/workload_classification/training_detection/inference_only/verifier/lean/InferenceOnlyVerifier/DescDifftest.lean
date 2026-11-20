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

-- A whole lane of two levels, the first of radix 1,048,577 (a composite reading 2,097,154 references to one parameter leaf),
-- is refused by name: `Piece.pisWalk`'s levels.
#guard match Json.parse r#"{"program":"{\"annotations\":{},\"boolean\":{\"Op_v1\":\"StandIn[Op]_v1\"},\"callees\":[\"RegisteredInput8_v1\",\"Op_v1\",\"F_v1\",\"Const1[0x0]_v1\"],\"definitions\":{\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"F_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"grid\":[1,[[\"p\",0,0,0,[[1048577,0],[2,0]]]]]},\"type\":2}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x0\",[\"Array\",1,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"F_v1\"},\"Op_v1\":{\"form\":\"primitive\",\"id\":\"Op_v1\",\"params\":[[\"a0\",[\"Array\",2097154,[\"Value\",8]]]],\"ret\":[\"Value\",8]},\"R_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,1]},\"type\":1}],\"fn\":2,\"form\":\"call\",\"out\":0}],\"params\":[],\"ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"R_v1\"},\"RegisteredInput8_v1\":{\"form\":\"primitive\",\"id\":\"RegisteredInput8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"StandIn[Op]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":3,\"form\":\"primitive\",\"out\":4}],\"params\":[[\"a0\",[\"Array\",2097154,[\"Array\",8,[\"Value\",1]]]]],\"ret\":{\"strided\":[\"n\",0,0,1,0,0,8]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[Op]_v1\"}},\"program_fn\":\"R_v1\",\"root\":\"R_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",8],[\"Array\",1,[\"Value\",8]],[\"Array\",2097154,[\"Value\",8]],[\"Array\",2097153,[\"Value\",8]],[\"Value\",1]]}","theta":[],"owner":["input",0],"gamma":10,"alpha":[8,1]}"# with
  | .ok j => let r := Desc.check j; r.getObjValD "error" == Json.str "unwalked" &&
      ((r.getObjValD "message").getStr?.toOption.getD "").startsWith "intervals of a lane's levels"
  | .error _ => false

-- Part of such a lane, all its rows but the first (a call whose callee reads all but the first of its 2,097,154 leaves),
-- is refused by name: `Piece.pisWalk`'s rows.
#guard match Json.parse r#"{"program":"{\"annotations\":{},\"boolean\":{\"Op_v1\":\"StandIn[Op]_v1\"},\"callees\":[\"RegisteredInput8_v1\",\"Op_v1\",\"F_v1\",\"H_v1\",\"Const1[0x0]_v1\"],\"definitions\":{\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"F_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"grid\":[1,[[\"p\",0,0,0,[[1048577,0],[2,0]]]]]},\"type\":2}],\"fn\":3,\"form\":\"call\",\"out\":0}],\"params\":[[\"x0\",[\"Array\",1,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"F_v1\"},\"H_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,1,2097153]},\"type\":3}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x0\",[\"Array\",2097154,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"H_v1\"},\"Op_v1\":{\"form\":\"primitive\",\"id\":\"Op_v1\",\"params\":[[\"a0\",[\"Array\",2097153,[\"Value\",8]]]],\"ret\":[\"Value\",8]},\"R_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,1]},\"type\":1}],\"fn\":2,\"form\":\"call\",\"out\":0}],\"params\":[],\"ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"R_v1\"},\"RegisteredInput8_v1\":{\"form\":\"primitive\",\"id\":\"RegisteredInput8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"StandIn[Op]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":4,\"form\":\"primitive\",\"out\":4}],\"params\":[[\"a0\",[\"Array\",2097153,[\"Array\",8,[\"Value\",1]]]]],\"ret\":{\"strided\":[\"n\",0,0,1,0,0,8]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[Op]_v1\"}},\"program_fn\":\"R_v1\",\"root\":\"R_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",8],[\"Array\",1,[\"Value\",8]],[\"Array\",2097154,[\"Value\",8]],[\"Array\",2097153,[\"Value\",8]],[\"Value\",1]]}","theta":[],"owner":["input",0],"gamma":10,"alpha":[8,1]}"# with
  | .ok j => let r := Desc.check j; r.getObjValD "error" == Json.str "unwalked" &&
      ((r.getObjValD "message").getStr?.toOption.getD "").startsWith "rows of part of a lane"
  | .error _ => false

end InferenceOnlyVerifier.DescDifftest
