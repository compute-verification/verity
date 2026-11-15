module

public import Proofs.Properties.Verity.Sampled.HonestSession.ProgramProof
public import Proofs.Properties.Verity.Sampled.HonestSession.JsonCanonFacts
import all Circuits.Program
import all Flock.Firewall
import all Proofs.Flock.Recursive.FirewallJson
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
# The rules of `Flock.Program.decode`'s walk over the descriptor (`ProgramFacts.program_parse`)

The descriptor's bytes are its JSON printed (`canon_programJ`), so `decode` reads `norm programJ`
(`FirewallJson.parse_printJson`), and `fwd_walk` carries `decodeOf` over it rule by rule. Each rule below is one stop of that
walk: the descriptor's id checks, the node keys' sort (read by `qsort_of_sorted`, never run), `Ty`'s equality at the
descriptor's types, the width check `Widths.bodyBad` (as `bodyBadK`, which the kernel evaluates), and the program's SHA-512,
which stays `Sha512.hash` of the printed bare descriptor (`canon_bare`), never evaluated. The walk itself is not here yet.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.ProgramParse

open Flock
open Lean (Json)
open ProgramProof

theorem printable_of (j : Json) (h : (Firewall.jsonChars j).all (fun c => decide (c.toNat < 0x7f)) = true) :
    JsonCanon.Printable j := by
  intro c hc
  rw [Firewall.printJson, String.toList_ofList] at hc
  exact of_decide_eq_true (List.all_eq_true.1 h c hc)

theorem printable_programJ : JsonCanon.Printable programJ := printable_of _ (by decide +kernel)

theorem printable_programBare : JsonCanon.Printable programBare := printable_of _ (by decide +kernel)

theorem canon_programJ : canon programJ = .ok (Firewall.printJson programJ) :=
  JsonCanon.jsonFacts.canon_plain plain_programJ printable_programJ

theorem canon_programBare : canon programBare = .ok (Firewall.printJson programBare) :=
  JsonCanon.jsonFacts.canon_plain plain_programBare printable_programBare

/-- The last stop: the bare descriptor `decode` hashes is `norm programBare`, whose canonical text is `programBare`'s. -/
theorem canon_bare (j : Json) (h : j = FlockSoundness.Discharge.Recursion.FirewallJson.norm programBare) : canon j = .ok (Firewall.printJson programBare) := by
  subst h
  rw [JsonCanon.jsonFacts.canon_norm plain_programBare, canon_programBare]

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
  unfold Program.inputId? Program.widthAfter?
  have hs : "Input16_v1".startsWith "Input" = true := by simp +decide
  have hl : "Input".length = 5 := by simp +decide
  simp only [hs, ends_in16, hl, drop_in16]
  simp +decide [toNat16, String.all_bool_eq]

theorem inputId_not16' (s : String) (h : s = "Not16_v1") : Program.inputId? s = none := by
  subst h; exact inputId_not16

/-- A node's keys, already ascending, sort to themselves: the sort is read by `qsort_of_sorted`, never run. -/
theorem qsort_sorted (a : Array String) (h : a.toList.Pairwise (· < ·)) : a.qsort (· < ·) = a :=
  Array.toList_inj.1 (Properties.Pouw.Period.SecurityProofs.Sort.qsort_of_sorted h)

/-- `composite`'s match on a node object (`keys`), at a node the walk evaluates to an object: its keys' branch, so the sort
under it is no longer under a binder. -/
def isObj : Json → Bool
  | .obj _ => true
  | _ => false

def objKvs : Json → Std.TreeMap.Raw String Json compare
  | .obj kvs => kvs
  | _ => {}

private theorem keys_match (β : Type) (nd : Json) (f : Std.TreeMap.Raw String Json compare → β) (g : Json → β)
    (h : isObj nd = true) : Program.composite.match_3 (fun _ => β) nd f g = f (objKvs nd) := by
  cases nd <;> simp [isObj] at h
  rfl

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

/-- The decoder state's definitions, read off the state the walk built: the read no longer carries its hash maps. -/
private theorem ds_defs (d : Array Program.Defn) (i : Std.HashMap String Nat) (b : Std.HashSet String) (w : Nat) :
    (Program.DecState.mk d i b w).defs = d := rfl

theorem form_beq_false (a b : Program.Form) (h : (a == b) = false) : (a == b) = false := h

end Verity.Sampled.Instance.HonestSession.ProgramParse
