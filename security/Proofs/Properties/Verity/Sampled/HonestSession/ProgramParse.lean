module

public import Proofs.Properties.Verity.Sampled.HonestSession.ProgramProof
public import Proofs.Properties.Verity.Sampled.HonestSession.JsonCanonFacts
import Proofs.Flock.Verifier.Hash
import Circuits.Program
import Flock.ZkServer
import Proofs.Flock.Recursive.ZkServerJson
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
import all Init.Data.String.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Legacy
import all Init.Data.String.Slice
import all Init.Data.String.Search
import all Init.Data.String.TakeDrop
import all Init.Data.String.Pattern.String
import all Init.Data.String.Lemmas.Pattern.String.ForwardPattern
import all Init.Data.ToString.Basic
import all Init.Data.Repr
import all Init.Data.Nat.ToString
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Stream
import all Init.Data.Slice.Basic
import all Init.Data.Slice.Array.Basic
import all Init.Data.Array.Subarray
import all Init.Data.Range.Basic

/-!
# `Flock.Program.decode` on the descriptor (`ProgramFacts.program_parse`, `design_eq`, `lawAt_of`, `program_sha`)

The descriptor's bytes are its JSON printed (`canon_programJ`), so `decode` reads `norm programJ`
(`ZkServerJson.parse_printJson`), which is `programJ` (`norm_programJ`), and `fwd_def` carries `decodeOf` over it rule by
rule to `decV` (`decode_V`, `decode_file`). Each rule is one stop of that walk: the descriptor's id checks, the node keys'
sort (read by `qsort_of_sorted`, never run), `Ty`'s well-founded functions at the descriptor's types (`tyBeq_v16`,
`tyOfJson_v16`, `leavesK`), the width check
`Widths.bodyBad` (as `bodyBadK`), loops as structural recursions (`forInSt`, `forInEx`), each decoded definition folded into
its name (`defIn16`, `defNot16`, `defUnit`, `defRoot`), and the program's SHA-512, which stays `Sha512.hash` of the printed
bare descriptor (`canon_bare`), never evaluated. A consumer reads `decV` through `decV_defs`, `decV_root`, `decV_ids` and
`decV_units`.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.ProgramParse

open Flock
open Lean (Json)
open ProgramProof

theorem printable_of (j : Json) (h : (ZkServer.jsonChars j).all (fun c => decide (c.toNat < 0x7f)) = true) :
    JsonCanon.Printable j := by
  intro c hc
  rw [ZkServer.printJson, String.toList_ofList] at hc
  exact of_decide_eq_true (List.all_eq_true.1 h c hc)

theorem printable_programJ : JsonCanon.Printable programJ := printable_of _ (by decide +kernel)

theorem printable_programBare : JsonCanon.Printable programBare := printable_of _ (by decide +kernel)

theorem canon_programJ : canon programJ = .ok (ZkServer.printJson programJ) :=
  JsonCanon.jsonFacts.canon_plain plain_programJ printable_programJ

theorem canon_programBare : canon programBare = .ok (ZkServer.printJson programBare) :=
  JsonCanon.jsonFacts.canon_plain plain_programBare printable_programBare

/-- `norm` leaves the descriptor as it is: its members are already in key order, as the parser's trees insert them. -/
theorem norm_programJ : FlockSoundness.Discharge.Recursion.ZkServerJson.norm programJ = programJ := by kernel_rfl'

/-- The last stop: the bare descriptor `decode` hashes is `programBare`, whose canonical text is its printed JSON. -/
theorem canon_bare (j : Json) (h : j = programBare) : canon j = .ok (ZkServer.printJson programBare) := by
  subst h; exact canon_programBare

theorem ends_in16 : "Input16_v1".endsWith "_v1" = true := by
  show String.Slice.Pattern.BackwardSliceSearcher.endsWith "_v1".toSlice "Input16_v1".toSlice = true
  rw [String.Slice.Pattern.BackwardSliceSearcher.endsWith_iff]
  refine ⟨"Input16", ?_⟩
  rw [String.copy_toSlice, String.copy_toSlice, ← String.toList_inj, String.toList_append]
  decide +kernel

theorem drop_in16 : (("Input16_v1".drop 5).dropEnd 3).toString = "16" := by
  show (("Input16_v1".drop 5).dropEnd 3).copy = "16"
  rw [← String.toList_inj, String.Slice.toList_copy_dropEnd, String.toList_copy_drop]
  decide +kernel

theorem front16 : "16".front = '1' := by
  rw [String.front_eq, String.front?_eq]
  decide +kernel

theorem isNat16 : "16".toSlice.isNat = true :=
  String.Slice.isNat_of_isDigit (s := "16".toSlice) (by simp +decide) (by simp +decide)

theorem toNat16 : "16".toNat? = some 16 := by
  rw [← String.toNat?_toSlice, String.Slice.toNat?_eq_some_ofDigitChars isNat16, String.copy_toSlice]
  decide +kernel

theorem inputId_in16 (s : String) (h : s = "Input16_v1") : Program.inputId? s = some (Program.RUN, 16) := by
  subst h
  rw [ProgramProof.inputId_K]; unfold ProgramProof.widthAfterK
  have hs : "Input16_v1".startsWith "Input" = true := by simp +decide
  have hl : "Input".length = 5 := by simp +decide
  simp only [hs, ends_in16, hl, drop_in16]
  simp +decide [toNat16, String.all_bool_eq]

theorem inputId_not16' (s : String) (h : s = "Not16_v1") : Program.inputId? s = none := by
  subst h; exact inputId_not16

/-- A node's keys, already ascending, sort to themselves: the sort is read by `qsort_of_sorted`, never run. -/
theorem qsort_sorted (a : Array String) (h : a.toList.Pairwise (· < ·)) : a.qsort (· < ·) = a :=
  Array.toList_inj.1 (Properties.Pouw.MonitoringPeriod.SecurityProofs.Sort.qsort_of_sorted h)

/-- `Ty`'s equality test (well-founded) at the descriptor's two types. -/
theorem tyBeq_v16 (a b : Program.Ty) (ha : a = .value 16) (hb : b = .value 16) : (a == b) = true := by
  subst ha hb
  show Program.Ty.beq _ _ = true
  simp [Program.Ty.beq]

theorem tyBeq_a16 (a b : Program.Ty) (ha : a = .array 1 (.value 16)) (hb : b = .array 1 (.value 16)) :
    (a == b) = true := by
  subst ha hb
  show Program.Ty.beq _ _ = true
  simp [Program.Ty.beq]

/-- The program function's id holds no binding record, so it names no functions. -/
theorem recordFunctionIds_hs (s : String) (h : s = "HsProg_v0") : Program.recordFunctionIds s = #[] := by
  subst h
  have hc : ("HsProg_v0".contains '{') = false := by simp +decide
  simp [Program.recordFunctionIds, Program.recordPairs, hc]

theorem isInputId_not16 (s : String) (h : s = "Not16_v1") : Program.isInputId s = false := by
  subst h; simp [Program.isInputId, inputId_not16]

theorem registered_not16 (s : String) (h : s = "Not16_v1") : s.startsWith "Registered" = false := by
  subst h; simp +decide

section Widths
open Flock.Program Flock.Program.Widths

/-- The descriptor's types: `Value w` and `Array 1 (Value w)`. -/
def simpleTy : Ty → Bool
  | .value _ => true
  | .array n (.value _) => n == 1
  | _ => false

def homoS : Ty → Option Nat
  | .value w => some w
  | .array _ (.value w) => some w
  | _ => none

/-- `Widths.homo` (well-founded), structurally at simple types (`homo_eq`). -/
def homoK (t : Ty) : Option Nat := if simpleTy t then homoS t else homo t

theorem homo_simple (t : Ty) (h : simpleTy t = true) : homo t = homoS t := by
  cases t with
  | value w => simp [homo, rangeWidths, Ty.leaves, homoS]
  | array n e =>
    cases e with
    | value w =>
      simp only [simpleTy, beq_iff_eq] at h
      subst h
      simp [homo, rangeWidths, Ty.leaves, homoS]
    | _ => simp [simpleTy] at h
  | _ => simp [simpleTy] at h

theorem homo_eq : homo = homoK := by
  funext t
  unfold homoK
  split
  · exact homo_simple t ‹_›
  · rfl

/-- `Widths.seqBad` with `homoK` for `homo`. -/
def seqBadK (s : Src) (t : Ty) (seq : Seq) : Widths.M (Option (Nat × Part)) := do
  let ht := homoK t
  for q in seq.parts, a in seq.starts do
    if !(← partOk s t ht a q) then return some (a, q)
  return none

/-- `Widths.bodyBad` with `homoK` and `seqBadK`: the kernel evaluates it where every type is simple. -/
def bodyBadK (defs : Array Defn) (key : String) (B : Body) : Widths.M (Option String) := do
  let outs := B.nodes.map (·.out)
  let s : Src := { params := B.params.map (·.2), nodes := outs, hp := B.params.map (homoK ·.2), hn := outs.map homoK,
                   below := 0 }
  for k in [0:B.nodes.size] do
    let node := B.nodes[k]!
    let callee := defs[node.fn]!
    for i in [0:node.args.size] do
      let a := node.args[i]!
      let sk := { s with below := k }
      if let some (pos, q) ← seqBadK sk a.ty a.seq then
        let pn := (callee.params[i]?.map (·.1)).getD s!"{i}"
        return some (refusal B key s!"node {k} ({node.form.name} {callee.id}) argument '{pn}'" sk a.ty pos q)
  let sr := { s with below := B.nodes.size }
  if let some (pos, q) ← seqBadK sr B.retType B.ret then
    return some (refusal B key "the body's return" sr B.retType pos q)
  return none

theorem seqBad_K : seqBad = seqBadK := by
  funext s t seq
  unfold seqBad seqBadK
  rw [homo_eq]
  rfl

theorem bodyBad_K : bodyBad = bodyBadK := by
  funext defs key B
  unfold bodyBad bodyBadK
  rw [homo_eq, seqBad_K]
  rfl

/-- The walk's rule: `bodyBad`'s run is `bodyBadK`'s, which the kernel evaluates on simple types. -/
theorem bodyBad_run (defs : Array Defn) (key : String) (B : Body) (w : Nat) :
    (bodyBad defs key B).run w = (bodyBadK defs key B).run w := by rw [bodyBad_K]

end Widths

/-- The state monad's bind, the pair destructured by a match the walk reduces on its constructor: no projection of the
state is left behind in the values the walk carries. -/
theorem stBindM {σ ε α β : Type} (x : StateT σ (Except ε) α) (f : α → StateT σ (Except ε) β) (s : σ) :
    (x >>= f) s = (x s >>= fun p => match p with | (a, s') => f a s') := rfl

theorem mapM_st_nil {σ ε α β : Type} (xs : Array α) (f : α → StateT σ (Except ε) β) (s : σ) (h : xs = #[]) :
    (xs.mapM f : StateT σ (Except ε) (Array β)) s = .ok (#[], s) := by
  subst h; rw [Array.mapM_empty]; rfl

/-- A loop of `Except` over a list, by structural recursion. -/
def forInEx {ε α β : Type} (f : α → β → Except ε (ForInStep β)) : List α → β → Except ε β
  | [], b => .ok b
  | a :: as, b => f a b >>= fun r => match r with
    | .done b => .ok b
    | .yield b => forInEx f as b

theorem forIn_list_ex {ε α β : Type} (l : List α) (b : β) (f : α → β → Except ε (ForInStep β)) :
    (forIn l b f : Except ε β) = forInEx f l b := by
  induction l generalizing b with
  | nil => rfl
  | cons a as ih =>
    rw [List.forIn_cons]
    show (f a b >>= fun r => _) = (f a b >>= fun r => _)
    congr 1
    funext r
    cases r with
    | done b' => rfl
    | yield b' => exact ih b'

theorem forIn_arr_ex {ε α β : Type} (xs : Array α) (b : β) (f : α → β → Except ε (ForInStep β)) :
    (forIn xs b f : Except ε β) = forInEx f xs.toList b := by
  rw [← Array.forIn_toList, forIn_list_ex]

theorem forIn_range_ex {ε β : Type} (r : Std.Legacy.Range) (b : β) (f : Nat → β → Except ε (ForInStep β)) :
    (forIn r b f : Except ε β) = forInEx f (List.range' r.start r.size r.step) b := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', forIn_list_ex]

section Decode
open Flock.Program

/-- `Ty.leaves` (well-founded), by structural recursion at a value and an array of values (`leaves_eq`). -/
def leavesK : Ty → Nat
  | .value _ => 1
  | .array n (.value _) => n * 1
  | t => t.leaves

theorem leaves_eq : @Ty.leaves = leavesK := by
  funext t
  cases t with
  | value w => simp [leavesK, Ty.leaves]
  | array n e => cases e <;> simp [leavesK, Ty.leaves]
  | tuple fs => simp [leavesK]
  | record fs => simp [leavesK]

/-- `Ty.ofJson` (well-founded) at the descriptor's two types. -/
theorem tyOfJson_v16 (j : Json) (h : j = v16) : Ty.ofJson j = .ok (.value 16) := by
  subst h; rw [Ty.ofJson.eq_def]; rfl

theorem tyOfJson_a16 (j : Json) (h : j = .arr #[.str "Array", jn 1, v16]) :
    Ty.ofJson j = .ok (.array 1 (.value 16)) := by
  subst h; rw [Ty.ofJson.eq_def]
  show (do let e ← Ty.ofJson v16; pure (Ty.array 1 e) : V Ty) = _
  rw [tyOfJson_v16 _ rfl]; rfl

/-! The parameters' repeated-name check at the descriptor's parameter lists, none or one parameter: its set of names
hashes with `String.hash`, which the walk blocks. -/

theorem dup_params_nil (ps : Array (String × Ty)) (n : Nat) (h : ps = #[]) (hn : n = 0) :
    ((Array.foldl (fun s p => s.insert p.1) (∅ : Std.HashSet String) ps 0 n).size != n) = false := by
  subst h hn; rfl

theorem dup_params_one (ps : Array (String × Ty)) (n : Nat) (h : ps.size = 1) (hn : n = 1) :
    ((Array.foldl (fun s p => s.insert p.1) (∅ : Std.HashSet String) ps 0 n).size != n) = false := by
  subst hn
  rcases ps with ⟨_ | ⟨x, _ | ⟨y, l⟩⟩⟩
  · simp at h
  · show (((∅ : Std.HashSet String).insert x.1).size != 1) = false
    simp [Std.HashSet.size_insert]
  · simp at h

/-! The decoded definitions, named: the walk folds each record into its name as it is decoded (`fold_in16`, …), so the
state it carries stays small. -/

def defIn16 : Defn :=
  { key := "Input16_v1", id := "Input16_v1", prim := true, params := #[], ret := .value 16,
    binding := some RUN, gates := 1, paramLeaves := 0, outLeaves := 1 }

def defNot16 : Defn :=
  { key := "Not16_v1", id := "Not16_v1", prim := true, params := #[("a", .value 16)], ret := .value 16,
    gates := 1, paramLeaves := 1, outLeaves := 1 }

def unitBody : Body :=
  { params := #[("x", .value 16)], retType := .value 16,
    nodes := #[{ form := .prim, fn := 1, args := #[{ ty := .value 16, seq := Seq.ofParts #[.affine true 0 0 1] }],
                 out := .value 16 }],
    ret := Seq.ofParts #[.affine false 0 0 1], offsets := #[0, 1] }

def defUnit : Defn :=
  { key := templateId, id := templateId, prim := false, params := #[("x", .value 16)], ret := .value 16,
    body := unitBody, gates := 1, paramLeaves := 1, outLeaves := 1 }

def rootBody : Body :=
  { params := #[], retType := .value 16,
    nodes := #[{ form := .batch, fn := 0, args := #[], out := .array 1 (.value 16), axes := #[], n := 1 },
      { form := .call, fn := 2, args := #[{ ty := .value 16, seq := Seq.ofParts #[.affine false 0 0 1] }],
        out := .value 16 }],
    ret := Seq.ofParts #[.affine false 1 0 1], offsets := #[0, 1, 2] }

def defRoot : Defn :=
  { key := rootId, id := rootId, prim := false, params := #[], ret := .value 16, body := rootBody,
    gates := 2, paramLeaves := 0, outLeaves := 1 }

theorem fold_in16 (k i : String) (p : Bool) (ps : Array (String × Ty)) (r : Ty) (bi : Option String) (bd : Body)
    (g pl ol : Nat) (hk : k = defIn16.key) (hi : i = defIn16.id) (hp : p = defIn16.prim) (hps : ps = defIn16.params)
    (hr : r = defIn16.ret) (hbi : bi = defIn16.binding) (hbd : bd = defIn16.body) (hg : g = defIn16.gates)
    (hpl : pl = defIn16.paramLeaves) (hol : ol = defIn16.outLeaves) : Defn.mk k i p ps r bi bd g pl ol = defIn16 := by
  subst hk hi hp hps hr hbi hbd hg hpl hol; rfl

theorem fold_not16 (k i : String) (p : Bool) (ps : Array (String × Ty)) (r : Ty) (bi : Option String) (bd : Body)
    (g pl ol : Nat) (hk : k = defNot16.key) (hi : i = defNot16.id) (hp : p = defNot16.prim) (hps : ps = defNot16.params)
    (hr : r = defNot16.ret) (hbi : bi = defNot16.binding) (hbd : bd = defNot16.body) (hg : g = defNot16.gates)
    (hpl : pl = defNot16.paramLeaves) (hol : ol = defNot16.outLeaves) : Defn.mk k i p ps r bi bd g pl ol = defNot16 := by
  subst hk hi hp hps hr hbi hbd hg hpl hol; rfl

theorem fold_unit (k i : String) (p : Bool) (ps : Array (String × Ty)) (r : Ty) (bi : Option String) (bd : Body)
    (g pl ol : Nat) (hk : k = defUnit.key) (hi : i = defUnit.id) (hp : p = defUnit.prim) (hps : ps = defUnit.params)
    (hr : r = defUnit.ret) (hbi : bi = defUnit.binding) (hbd : bd = defUnit.body) (hg : g = defUnit.gates)
    (hpl : pl = defUnit.paramLeaves) (hol : ol = defUnit.outLeaves) : Defn.mk k i p ps r bi bd g pl ol = defUnit := by
  subst hk hi hp hps hr hbi hbd hg hpl hol; rfl

theorem fold_root (k i : String) (p : Bool) (ps : Array (String × Ty)) (r : Ty) (bi : Option String) (bd : Body)
    (g pl ol : Nat) (hk : k = defRoot.key) (hi : i = defRoot.id) (hp : p = defRoot.prim) (hps : ps = defRoot.params)
    (hr : r = defRoot.ret) (hbi : bi = defRoot.binding) (hbd : bd = defRoot.body) (hg : g = defRoot.gates)
    (hpl : pl = defRoot.paramLeaves) (hol : ol = defRoot.outLeaves) : Defn.mk k i p ps r bi bd g pl ol = defRoot := by
  subst hk hi hp hps hr hbi hbd hg hpl hol; rfl

end Decode

/-! **`decode` on the descriptor's JSON**: the walk of `decodeOf` from `programJ`, rule by rule. -/
fwd_def decV decode_V : Program.decodeOf (some (.ok programJ))
    by [hs_contains_insert, hs_contains_empty, hs_contains_erase, hm_get_insert, hm_get_empty, fst_mk, snd_mk,
      forIn_st_nil, forIn_range_st, stPure, splitOn_root, splitOn_unit, all_list, toNat_one, repr_one, stBindM,
      stGet, stSet, stModify, iteApp, domainError_none, mapM_except, tyOfJson_v16, tyOfJson_a16, leaves_eq,
      fold_in16, fold_not16, fold_unit, fold_root, forIn_arr_ex, forIn_range_ex,
      foldl_insert_get, inputId_in16, inputId_not16', qsort_sorted, tyBeq_v16, tyBeq_a16, recordFunctionIds_hs,
      isInputId_not16, registered_not16, bodyBad_run, mapM_st_nil, canon_bare, dup_params_nil, dup_params_one]
    block [String.hash, Flock.canon, ZkServer.printJson, Sha512.hash] shallow []

/-! What a consumer reads of `decV`: its definitions by name, its root, and its one unit. -/

theorem decV_defs : decV.defs = #[defIn16, defNot16, defUnit, defRoot] := by kernel_rfl'

theorem decV_root : decV.root = 3 := by kernel_rfl'

theorem decV_programFn : decV.programFn = "HsProg_v0" := by kernel_rfl'

theorem decV_ids : decV.defs.toList.map (·.id) = ["Input16_v1", "Not16_v1", templateId, rootId] := by
  rw [decV_defs]; rfl

/-- The root calls the template once: one proof block (`nUnits`). -/
theorem decV_units :
    ((decV.defs[decV.root]!).body.nodes.toList.filter fun x => (decV.defs[x.fn]!).key == templateId).length = nUnits := by
  rw [decV_defs, decV_root]; decide +kernel

theorem decV_sha512 : decV.sha512 = Sha512.hash (ZkServer.printJson programBare).toUTF8 := by kernel_rfl'

/-- The registered bytes are the descriptor's JSON printed. -/
theorem programFile_eq : programFile = (ZkServer.printJson programJ).toUTF8 := by
  unfold programFile
  rw [canon_programJ]
  rfl

theorem desc_bytes : (⟨desc.toArray⟩ : ByteArray) = programFile := by
  unfold desc
  generalize programFile = b
  cases b
  simp

/-- **`decode programFile = .ok decV`**. -/
theorem decode_file : Program.decode programFile = .ok decV := by
  rw [Program.decode_eq_of, Program.readJson, programFile_eq, Properties.CanonRoundTrip.fromUTF8?_toUTF8, Option.map_some,
    FlockSoundness.Discharge.Recursion.ZkServerJson.parse_printJson plain_programJ, norm_programJ]
  exact decode_V

theorem parse_file : Spot.Program.parse programFile = .ok ⟨.ofProg decV, decV⟩ := by
  unfold Spot.Program.parse
  rw [decode_file]
  rfl

theorem program_eq : program = ⟨.ofProg decV, decV⟩ := by
  unfold program
  rw [parse_file]
  rfl

/-- **`ProgramFacts.program_parse`**. -/
theorem program_parse : Spot.Program.parse ⟨desc.toArray⟩ = .ok program := by
  rw [desc_bytes, parse_file, program_eq]

/-- **`ProgramFacts.design_eq`**. -/
theorem design_eq : Flock.Bridge.designAt desc query samplingDesign none nUnits = .subset 1 := design_eq_of_parse program_parse

/-- **`ProgramFacts.lawAt_of`**. -/
theorem lawAt_of : ∀ (v : Spot.Docket.Verifier), v.program = program → v.query = query → v.samplingDesign = samplingDesign →
    v.workTable = none → samplingDesignAt v nUnits = .ok (Flock.Bridge.designAt desc query samplingDesign none nUnits) :=
  lawAt_of_parse program_parse

/-- **`ProgramFacts.program_sha`**: the hash `decode` takes is of the bare descriptor's printed text, never evaluated. -/
theorem program_sha : program.sha512 = programSha := by
  rw [program_eq]
  show decV.sha512 = programSha
  rw [decV_sha512]
  unfold programSha
  rw [canon_programBare]
  rfl

/-- A loop of `ByteArray.foldl` whose step appends two hex digits keeps its text hex and grows it by two a byte. -/
theorem loop_hex (g : String → UInt8 → String)
    (hg : ∀ s x, s.toList.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) = true →
      (g s x).toList.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) = true ∧ (g s x).length = s.length + 2)
    (as : ByteArray) (stop : Nat) (h : stop ≤ as.size) :
    ∀ (i j : Nat) (s : String), s.toList.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) = true →
      (ByteArray.foldlM.loop (m := Id) (fun b x => pure (g b x)) as stop h i j s).run.toList.all
        (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) = true ∧
      (ByteArray.foldlM.loop (m := Id) (fun b x => pure (g b x)) as stop h i j s).run.length =
        s.length + 2 * min i (stop - j) := by
  intro i
  induction i with
  | zero => intro j s hs; unfold ByteArray.foldlM.loop; split <;> simp [hs]
  | succ i ih =>
    intro j s hs
    unfold ByteArray.foldlM.loop
    split
    · obtain ⟨h1, h2⟩ := hg s as[j] hs
      obtain ⟨h3, h4⟩ := ih (j+1) _ h1
      exact ⟨h3, h4.trans (by rw [h2]; omega)⟩
    · simp [hs]; omega

theorem foldl_hex (g : String → UInt8 → String) (hg1 : ∀ s x, (g s x).toList = s.toList ++ (g "" x).toList)
    (hg2 : ∀ n, n < 256 → (g "" (UInt8.ofNat n)).toList.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) = true ∧
      (g "" (UInt8.ofNat n)).toList.length = 2)
    (b : ByteArray) (hb : b.size = 64) : Flock.HmRow.isSha512Hex (b.foldl g "") = true := by
  have hg : ∀ s x, s.toList.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) = true →
      (g s x).toList.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) = true ∧ (g s x).length = s.length + 2 := by
    intro s x hs
    obtain ⟨a1, a2⟩ := hg2 x.toNat x.toNat_lt
    rw [UInt8.ofNat_toNat] at a1 a2
    rw [← String.length_toList, ← String.length_toList, hg1, List.all_append, hs, a1, List.length_append, a2]
    exact ⟨rfl, rfl⟩
  have key := loop_hex g hg b b.size (Nat.le_refl _) (b.size - 0) 0 "" rfl
  unfold Flock.HmRow.isSha512Hex
  rw [String.all_eq_all_toSlice, String.Slice.all_bool_eq, String.copy_toSlice]
  unfold ByteArray.foldl ByteArray.foldlM
  simp only [Nat.le_refl, ↓reduceDIte]
  refine Bool.and_eq_true_iff.mpr ⟨?_, key.1⟩
  rw [beq_iff_eq, key.2, hb]; rfl

/-- **A 64-byte string's hex is a SHA-512 digest's**, as `HmRow.checkPartitionObject` reads its `program`. -/
theorem hexOf_sha512Hex (b : ByteArray) (hb : b.size = 64) : Flock.HmRow.isSha512Hex (Flock.hexOf b) = true := by
  unfold Flock.hexOf
  refine foldl_hex _ ?_ ?_ b hb
  · intro s x; simp [String.toList_push]
  · decide +kernel

/-- The registration's program digest in hex is one (`popMatches`' partition object names it). -/
theorem programSha_hex : Flock.HmRow.isSha512Hex (Flock.hexOf programSha) = true :=
  hexOf_sha512Hex _ (Flock.Sha512.hash_size _)

end Verity.Sampled.Instance.HonestSession.ProgramParse
