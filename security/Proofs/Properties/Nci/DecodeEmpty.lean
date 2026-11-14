module

public import Circuits.Program
import Proofs.Flock.SessionLoop.Concrete.Fwd
import Batteries.Tactic.OpenPrivate
import Std.Data.HashMap.Lemmas
import Std.Data.HashSet.Lemmas
import all Circuits.Program
import all Base.Canon
import all Base.JsonParse
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold
import all Init.Data.Array.Attach
import all Init.WF

/-!
# The decoder's answer on the Program of no node

`decode_empty`: `Flock.Program.decode` reads `programEmpty`, a descriptor whose root has no node, as the Program
`progEmpty`, a term, with no hypothesis. `decode` keeps its definitions and its state in hash maps keyed by strings,
and the kernel computes no such lookup; everything else of it the kernel runs. So the answer is reached in three moves.

* **`decode`'s own lines, with its lookups named.** `decodeT`, `decodeTail`, `walk`, `walkRest` and `getDefRest` are
  the lines of `decode` and of its `getDef`, cut at the walk of the definitions and at `getDef`'s three lookups; each
  cut is an equation the kernel checks by unfolding both sides (`decode_eq`, `walk_eq`, `getDef_eq`), for every input.
* **The three lookups, by the library's lemmas.** In a state whose index does not hold `fid` and whose busy set does
  not, over a descriptor that defines `fid` as `d`, `getDef` is `getDefRest` at `d` (`getDef_fresh`).
* **The rest, by the kernel.** On `programEmpty` the lines before the walk run to the descriptor of one definition
  (`decodeT_empty`, for every walk), and the lines after the lookups run to a Program (`answerEmpty_isOk`).
-/

namespace Flock.Program.Closed

open Lean (Json JsonNumber)
open private getDef composite staticFns DecState DecState.defs DecState.index DecState.busy Desc Desc.types Desc.callees
  Desc.ddefs liftV at_ paramsOf buildNode natJ? from Circuits.Program

/-- A descriptor whose root has no node. -/
@[expose] public def programEmpty : String :=
  "{\"annotations\":{},\"callees\":[],\"definitions\":{\"Root_v1\":{\"bindings\":{},\"body\":{\"nodes\"" ++
  ":[],\"params\":[],\"ret\":{\"concat\":[]},\"ret_type\":[\"Tuple\",[]]},\"form\":\"composite\",\"id\"" ++
  ":\"Root_v1\"}},\"program_fn\":\"Root_v1\",\"root\":\"Root_v1\",\"schema\":\"verity-ir/descriptor/v1" ++
  "\",\"types\":[]}"

/-- The decoder's monad: its state over its refusals. -/
abbrev DM := StateT DecState (Except String)

/-- What an accepted answer holds, the default where it refuses. -/
def getOk {ε α : Type} [Inhabited α] (x : Except ε α) : α :=
  match x with
  | .ok a => a
  | .error _ => default

theorem ok_get {ε α : Type} [Inhabited α] {x : Except ε α} (h : x.toOption.isSome = true) : x = .ok (getOk x) := by
  cases x with
  | ok a => rfl
  | error e => simp [Except.toOption] at h

/-- A text's bytes read back as the text. -/
theorem fromUTF8?_toUTF8 (s : String) : String.fromUTF8? s.toUTF8 = some s := by
  have h : s.toUTF8.IsValidUTF8 := s.isValidUTF8
  first
    | (unfold String.fromUTF8?; rw [dif_pos h]; rfl)
    | (simp only [String.fromUTF8?, dif_pos h]; rfl)
    | simp [String.fromUTF8?, String.fromUTF8, h]

/-! ### `decode`'s lines -/

/-- `decode`'s walk of the definitions: the root, then the functions the program's name binds. -/
def walk (desc : Desc) (rootId programFn : String) : DM Nat := do
  let r ← getDef Steps.FUEL desc rootId
  for f in recordFunctionIds programFn do
    if desc.ddefs.contains f then discard (getDef Steps.FUEL desc f)
  return r

/-- `decode`'s lines after its walk of the definitions. -/
def decodeTail (top : Std.TreeMap.Raw String Json) (rootId programFn : String)
    (r : Except String (Nat × DecState)) : V Prog := do
  let (root, st) ← r
  let rd := st.defs[root]!
  if rd.prim || !rd.params.isEmpty then throw s!"root '{rootId}': the root is a composite without parameters"
  for d in st.defs do
    if d.prim || d.key == rootId then continue
    for k in [0:d.body.nodes.size] do
      let c := st.defs[d.body.nodes[k]!.fn]!
      if c.prim && (isInputId c.key || (c.key.startsWith "Registered" && isInputId (c.key.drop 10).toString)) then
        throw s!"definition '{d.key}' node {k}: an input node ({c.key}) appears only in the root's body"
  let bare := Json.mkObj (top.toArray.filter (·.1 != "annotations")).toList
  return { defs := st.defs, index := st.index, root, programFn, sha512 := Sha512.hash (← canon bare).toUTF8 }

/-- `decode` once its bytes are read as a text, its walk of the definitions left to `walkOf`. -/
def decodeT (walkOf : Desc → String → String → DM Nat) (text : String) : V Prog := do
  let j ← match JsonParse.parse text with
    | .ok j => pure j
    | .error e => throw s!"program: not JSON: {e}"
  let .obj top := j | throw "program: not an object"
  match (j.getObjValAs? String "schema").toOption with
  | some s => if s != SCHEMA then throw s!"unexpected descriptor schema '{s}' (this verifier reads {SCHEMA})"
  | none => throw s!"unexpected descriptor schema (this verifier reads {SCHEMA})"
  let .ok (.obj dkv) := j.getObjVal? "definitions" | throw "program: definitions"
  for (fid, d) in dkv.toArray do
    let .obj _ := d | throw s!"definition '{fid}': not an object"
  let freeDefs := dkv.toArray.map fun (fid, d) => (fid, match d with
    | .obj kvs => Json.mkObj ((kvs.toArray.filter (·.1 != "body")).toList ++
        ((d.getObjVal? "body").toOption.map (fun b => ["params", "ret_type"].filterMap fun k =>
          (b.getObjVal? k).toOption.map (k, ·))).getD [])
    | x => x)
  let free := Json.mkObj ((top.toArray.filter (fun (k, _) => k != "definitions" && k != "annotations")).toList ++
    [("definitions", Json.mkObj freeDefs.toList)])
  if let some e := domainError free then throw s!"descriptor outside the canonical JSON domain (PROTOCOL.md §1): {e}"
  let .ok (.arr callees) := j.getObjVal? "callees" | throw "program: callees"
  let .ok (.arr tys) := j.getObjVal? "types" | throw "program: types"
  let types ← tys.mapM Ty.ofJson
  let desc : Desc := { ddefs := dkv.toArray.foldl (fun m (k, v) => m.insert k v) {}, callees, types }
  let some rootId := (j.getObjValAs? String "root").toOption | throw "program: root"
  let some programFn := (j.getObjValAs? String "program_fn").toOption | throw "program: program_fn"
  decodeTail top rootId programFn ((walkOf desc rootId programFn).run {})

/-- `decode` is its lines. -/
theorem decode_eq (bytes : ByteArray) :
    decode bytes = match String.fromUTF8? bytes with
      | some text => decodeT walk text
      | none => throw "program: not UTF-8" := by
  kernel_rfl'

/-- The walk after its first `getDef`. -/
def walkRest (desc : Desc) (programFn : String) (r : Except String (Nat × DecState)) :
    Except String (Nat × DecState) :=
  r >>= fun (a, s) => (do
    for f in recordFunctionIds programFn do
      if desc.ddefs.contains f then discard (getDef Steps.FUEL desc f)
    return a : DM Nat) s

theorem walk_eq (desc : Desc) (rootId programFn : String) (s : DecState) :
    walk desc rootId programFn s = walkRest desc programFn (getDef Steps.FUEL desc rootId s) := by
  kernel_rfl'

/-! ### `getDef`'s lines, cut at its three lookups -/

/-- `getDef` after its three lookups, at the definition `d` it found. -/
def getDefRest (n : Nat) (desc : Desc) (fid : String) (d : Json) : DM Nat := do
  let form := (d.getObjValAs? String "form").toOption.getD ""
  let id := (d.getObjValAs? String "id").toOption.getD ""
  let defn ← if form == "primitive" then do
      if id != fid then throw s!"definition '{fid}': a primitive is keyed by its id, not '{id}'"
      let params ← liftV (paramsOf ((d.getObjVal? "params").toOption.getD Json.null) s!"primitive {fid}")
      let ret ← liftV (Ty.ofJson ((d.getObjVal? "ret").toOption.getD Json.null))
      let .value w := ret | throw s!"a primitive returns exactly one Value<w>; {fid} declares {ret.render}"
      let input := inputId? fid
      if let some (_, iw) := input then
        if !(params.isEmpty && w == iw) then throw s!"primitive {fid} signature differs from the registry"
      pure ({ key := fid, id, prim := true, params, ret, binding := input.map (·.1), gates := 1,
              paramLeaves := params.foldl (· + ·.2.leaves) 0, outLeaves := 1 } : Defn)
    else composite n desc fid d form id
  let s ← get
  set { s with defs := s.defs.push defn, index := s.index.insert fid s.defs.size, busy := s.busy.erase fid }
  return s.defs.size

/-- `getDef` at what the descriptor holds for `fid`. -/
def getDefK3 (n : Nat) (desc : Desc) (fid : String) : Option Json → DM Nat
  | some d => getDefRest n desc fid d
  | none => throw s!"KeyError: no definition {fid}"

/-- `getDef` at whether `fid` is being decoded. -/
def getDefK2 (n : Nat) (desc : Desc) (fid : String) (od : Option Json) (busy : Bool) : DM Nat :=
  if busy then throw s!"definition '{fid}' depends on itself" else getDefK3 n desc fid od

/-- `getDef` at what the index holds for `fid`. -/
def getDefK1 (n : Nat) (desc : Desc) (fid : String) (od : Option Json) : Option Nat → DM Nat
  | some i => pure i
  | none => do
    let s ← get
    getDefK2 n desc fid od (s.busy.contains fid)

/-- `getDef` is its lines: the index's lookup, then the busy set's, then the descriptor's. -/
theorem getDef_eq (n : Nat) (desc : Desc) (fid : String) :
    getDef (n + 1) desc fid = (do
      let s ← get
      getDefK1 n desc fid desc.ddefs[fid]? s.index[fid]? : DM Nat) := by
  unfold getDef
  kernel_rfl'

/-- **`getDef` given its three lookups.** In a state whose index does not hold `fid` and whose busy set does not,
over a descriptor that defines `fid` as `d`, `getDef` runs its lines after the lookups at `d`. -/
theorem getDef_fresh (n : Nat) (desc : Desc) (fid : String) (s : DecState) (d : Json)
    (hi : s.index[fid]? = none) (hb : s.busy.contains fid = false) (hd : desc.ddefs[fid]? = some d) :
    getDef (n + 1) desc fid s = getDefRest n desc fid d s := by
  have h1 : getDef (n + 1) desc fid s = getDefK1 n desc fid desc.ddefs[fid]? s.index[fid]? s := by
    rw [getDef_eq]
    first
      | kernel_rfl'
      | skip
  have h2 : getDefK1 n desc fid desc.ddefs[fid]? none s
      = getDefK2 n desc fid desc.ddefs[fid]? (s.busy.contains fid) s := by
    kernel_rfl'
  have h3 : getDefK2 n desc fid (some d) false s = getDefRest n desc fid d s := by
    kernel_rfl'
  rw [h1, hi, h2, hb, hd, h3]

/-! ### The Program of no node -/

/-- The parsed text of `programEmpty`, null where the parse refuses. -/
def docEmpty : Json := (JsonParse.parse programEmpty).toOption.getD .null

/-- Its pairs. -/
def topEmpty : Std.TreeMap.Raw String Json :=
  match docEmpty with
  | .obj top => top
  | _ => ∅

/-- Its root's definition. -/
def rootEmpty : Json :=
  ((docEmpty.getObjVal? "definitions").toOption.bind fun d => (d.getObjVal? "Root_v1").toOption).getD .null

/-- The descriptor `decode` hands its walk. -/
def descEmpty : Desc :=
  { ddefs := (∅ : Std.HashMap String Json).insert "Root_v1" rootEmpty, callees := #[], types := #[] }

/-- The fold `decode` builds its definitions with, on one definition. -/
theorem fold_one (d : Json) :
    (#[("Root_v1", d)] : Array (String × Json)).foldl (fun m (k, v) => m.insert k v) {}
      = (∅ : Std.HashMap String Json).insert "Root_v1" d := by
  kernel_rfl'

/-- `decode`'s lines before the walk, on `programEmpty`: the descriptor of the one definition, for every walk. -/
theorem decodeT_empty (walkOf : Desc → String → String → DM Nat) :
    decodeT walkOf programEmpty
      = decodeTail topEmpty "Root_v1" "Root_v1" (walkOf descEmpty "Root_v1" "Root_v1" {}) := by
  kernel_rfl'

theorem index_start (fid : String) : (({} : DecState).index)[fid]? = none := by
  first
    | exact Std.HashMap.getElem?_empty
    | exact Std.HashMap.getElem?_emptyWithCapacity
    | simp

theorem busy_start (fid : String) : ({} : DecState).busy.contains fid = false := by
  first
    | exact Std.HashSet.contains_empty
    | exact Std.HashSet.contains_emptyWithCapacity
    | simp

theorem ddefs_root : descEmpty.ddefs["Root_v1"]? = some rootEmpty := by
  show ((∅ : Std.HashMap String Json).insert "Root_v1" rootEmpty)["Root_v1"]? = some rootEmpty
  exact Std.HashMap.getElem?_insert_self

theorem fuel_succ : Steps.FUEL - 1 + 1 = Steps.FUEL := by decide +kernel

/-- `getDef` on the root of `programEmpty`, from the start: its lines after the lookups. -/
theorem getDef_empty :
    getDef Steps.FUEL descEmpty "Root_v1" {} = getDefRest (Steps.FUEL - 1) descEmpty "Root_v1" rootEmpty {} := by
  have h := getDef_fresh (Steps.FUEL - 1) descEmpty "Root_v1" {} rootEmpty (index_start _) (busy_start _) ddefs_root
  rw [fuel_succ] at h
  exact h

/-- What `decode` answers on `programEmpty`, no lookup by a hash left in what decides it. -/
def answerEmpty : V Prog :=
  decodeTail topEmpty "Root_v1" "Root_v1"
    (walkRest descEmpty "Root_v1" (getDefRest (Steps.FUEL - 1) descEmpty "Root_v1" rootEmpty {}))

/-- The Program of no node. -/
public def progEmpty : Prog := getOk answerEmpty

theorem answerEmpty_isOk : answerEmpty.toOption.isSome = true := by decide +kernel

/-- **The decoder reads `programEmpty`**, with no hypothesis. -/
public theorem decode_empty : decode programEmpty.toUTF8 = .ok progEmpty := by
  rw [decode_eq, fromUTF8?_toUTF8]
  show decodeT walk programEmpty = .ok progEmpty
  rw [decodeT_empty walk, walk_eq, getDef_empty]
  exact ok_get answerEmpty_isOk

end Flock.Program.Closed
