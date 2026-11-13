module

public import Flock.With.SetupEq
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Json.Text
public import Proofs.Properties.Verity.Sampled.Record.PartCong
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic

/-!
# Alike partition objects set the session up alike

A record states a member's partition object as text. Parsed back, it is alike the object the verifier holds
(`Claims.Json.Same`: equal leaves, arrays alike element by element, objects with one key set and alike values), and
equal to it only up to the shape of the tree an object keeps its keys in. The session's setup reads a partition
object only through `view`, and every reader there returns the same on alike objects:

* `chkObj_same`, `digest_same`, `chkPart_same`: the form check, the tagged digest and the binding check;
* `tm_same`: the templates a template query names;
* `derive_same`, `qwordUnits_same`, `qcallUnits_same`, `qword_same`: the units derived from the verifier's program;
* `strat_same`, `work_same`: a stratified and a work sample's design;
* `view_same`: so two alike objects have one view.

An object that passes the binding check has a canonical text (`canon_of_chkPart`): the check hashes it.

`PartSame x y` is what a session's two readings of the partition may be: one partition, or two alike objects. On
both, `Stmt.setupTables` is the same (`setupTables_same`).

A reader that matches on a value which is only alike on the two sides is restated with each read of the object an
argument: `chkObjV` (`HmRow.checkPartitionObject`, which matches on the query's parameters and on a parameter's
value), `tmV` (`TemplateQuery.templatesOf`, on the list of templates) and `chkPartV` (`HmRow.checkPartition`, on the
object). The three are private to this file, so no statement another file reads names one, and each is the
verifier's function by `rfl` (`chkObj_view`, `tm_view`, `chkPart_view`), so a change to the verifier breaks this
file's build and nothing else. The other readers read the object by key only and are rewritten in place.
-/

@[expose] public section

namespace Flock.PartCong

open Lean (Json)
open Claims.Json (Same)
open HmRow (PARTITION_V1 isSha512Hex EVALUABLE Q_CALL)

/-- The value at `k`, `null` where there is none: how the readers step into an object. -/
private abbrev sub (j : Json) (k : String) : Json := (j.getObjVal? k).toOption.getD Json.null

/-- Two reads of a key that a reader matching on them cannot tell apart: two refusals, or two alike values. -/
def AlikeE (x y : Except String Json) : Prop :=
  (∃ e e', x = .error e ∧ y = .error e') ∨ ∃ a b, x = .ok a ∧ y = .ok b ∧ Same a b

/-- Two optional values that a reader matching on them cannot tell apart: both missing, or both there and alike. -/
private def Alike (x y : Option Json) : Prop := (x = none ∧ y = none) ∨ ∃ a b, x = some a ∧ y = some b ∧ Same a b

theorem alikeE_get {j j' : Json} (h : Same j j') (k : String) : AlikeE (j.getObjVal? k) (j'.getObjVal? k) := by
  cases hv : j.getObjVal? k with
  | ok v =>
    obtain ⟨v', hv', hs⟩ := h.getObjVal hv
    exact Or.inr ⟨v, v', rfl, hv', hs⟩
  | error e =>
    cases hv' : j'.getObjVal? k with
    | ok v' =>
      obtain ⟨v, hv2, -⟩ := h.symm.getObjVal hv'
      rw [hv] at hv2
      cases hv2
    | error e' => exact Or.inl ⟨e, e', rfl, rfl⟩

private theorem alike_get {j j' : Json} (h : Same j j') (k : String) :
    Alike (j.getObjVal? k).toOption (j'.getObjVal? k).toOption := by
  rcases alikeE_get h k with ⟨e, e', e₁, e₂⟩ | ⟨a, b, e₁, e₂, hs⟩
  · rw [e₁, e₂]; exact Or.inl ⟨rfl, rfl⟩
  · rw [e₁, e₂]; exact Or.inr ⟨a, b, rfl, rfl, hs⟩

private theorem same_sub {j j' : Json} (h : Same j j') (k : String) : Same (sub j k) (sub j' k) := by
  show Same ((j.getObjVal? k).toOption.getD Json.null) ((j'.getObjVal? k).toOption.getD Json.null)
  rcases alike_get h k with ⟨e₁, e₂⟩ | ⟨a, b, e₁, e₂, hs⟩
  · rw [e₁, e₂]; exact Same.null
  · rw [e₁, e₂]; exact hs

theorem getStr_same {v v' : Json} (h : Same v v') : v.getStr? = v'.getStr? := by cases h <;> rfl

theorem getNat_same {v v' : Json} (h : Same v v') : v.getNat? = v'.getNat? := by cases h <;> rfl

private theorem asStr_same {j j' : Json} (h : Same j j') (k : String) :
    j.getObjValAs? String k = j'.getObjValAs? String k := getStr_same (same_sub h k)

private theorem asNat_same {j j' : Json} (h : Same j j') (k : String) :
    j.getObjValAs? Nat k = j'.getObjValAs? Nat k := getNat_same (same_sub h k)

/-- An object's keys, sorted, and none for another value: the readers' test of which keys an object has. -/
private def keysOf (j : Json) : Array String :=
  match j with
  | .obj kvs => (kvs.toArray.map (·.1)).qsort (· < ·)
  | _ => #[]

theorem keysArr_same {m n : Std.TreeMap.Raw String Json} (hm : m.WF) (hn : n.WF)
    (hk : ∀ k, (m.get? k).isSome = (n.get? k).isSome) : m.toArray.map (·.1) = n.toArray.map (·.1) := by
  have e : ∀ t : Std.TreeMap.Raw String Json, t.toArray.map (·.1) = t.keys.toArray := fun t => by
    rw [Std.TreeMap.Raw.toArray_keys]
    exact Std.TreeMap.Raw.map_fst_toArray_eq_keysArray
  rw [e m, e n, FlockComplete.keys_eq_of_same hm hn hk]

private theorem keysOf_same : ∀ {a b : Json}, Same a b → keysOf a = keysOf b
  | _, _, .null => rfl
  | _, _, .bool _ => rfl
  | _, _, .num _ => rfl
  | _, _, .str _ => rfl
  | _, _, .arr _ _ => rfl
  | _, _, .obj hm hn hk _ => congrArg (fun x : Array String => x.qsort (· < ·)) (keysArr_same hm hn hk)

/-- A template's name from its read as a string, with `templatesOf`'s refusal otherwise. -/
private def strOr (r : Except String String) : V String :=
  match r with
  | .ok s => pure s
  | .error _ => throw "partition: query-params: a template is not a string"

/-- The names in an array of templates, as `templatesOf` reads them. -/
private def mapStrs (a : Array Json) : V (Array String) := a.mapM fun t => strOr t.getStr?

private theorem mapStrs_same {a b : Array Json} (hs : a.size = b.size)
    (he : ∀ (i : Nat) (h₁ : i < a.size) (h₂ : i < b.size), Same a[i] b[i]) : mapStrs a = mapStrs b := by
  have e : a.map Json.getStr? = b.map Json.getStr? := by
    apply Array.ext
    · rw [Array.size_map, Array.size_map]; exact hs
    · intro i h₁ h₂
      rw [Array.getElem_map, Array.getElem_map]
      exact getStr_same (he i _ _)
  have m : ∀ x : Array Json, mapStrs x = (x.map Json.getStr?).mapM strOr := fun x =>
    (Array.mapM_map (f := Json.getStr?) (g := strOr) (xs := x)).symm
  rw [m a, m b, e]

/-- `TemplateQuery.templatesOf`'s text, with each read of the parameters an argument (`tm_view`). -/
private def tmV (name : String) (version : Nat) (keys : Array String) (tmpl : Option String) (tsv : Except String Json)
    (ms : Array Json → V (Array String)) : Option (V (Array String)) :=
  if (name, version) == ("Q_template_instance", 0) then some do
    match keys, tmpl with
    | #["template"], some t => pure #[t]
    | _, _ => throw "partition: query-params: Q_template_instance v0 takes exactly {template}, a string"
  else if (name, version) == ("Q_template_instances", 0) then some do
    if keys != #["templates"] then throw "partition: query-params: Q_template_instances v0 takes exactly {templates}"
    let ts ← match tsv with
      | .ok (.arr a) => ms a
      | _ => throw "partition: query-params: templates is not a list"
    if ts.isEmpty || (List.range ts.size).any (fun i => (ts.extract 0 i).contains ts[i]!) then
      throw "partition: query-params: templates is not a non-empty list of distinct strings"
    pure ts
  else none

set_option smartUnfolding false in
private theorem tm_view (name : String) (version : Nat) (params : Json) :
    TemplateQuery.templatesOf name version params =
      tmV name version (keysOf params) (params.getObjValAs? String "template").toOption
        (params.getObjVal? "templates") mapStrs := by rfl

set_option smartUnfolding false in
private theorem tmV_tsv (name : String) (version : Nat) (keys : Array String) (tmpl : Option String)
    {x y : Except String Json} (h : AlikeE x y) :
    tmV name version keys tmpl x mapStrs = tmV name version keys tmpl y mapStrs := by
  rcases h with ⟨e, e', rfl, rfl⟩ | ⟨a, b, rfl, rfl, hs⟩
  · rfl
  · cases hs with
    | @arr a' b' hs he =>
      calc tmV name version keys tmpl (.ok (.arr a')) mapStrs
          = tmV name version keys tmpl (.ok (.arr b')) (fun _ => mapStrs a') := rfl
        _ = tmV name version keys tmpl (.ok (.arr b')) (fun _ => mapStrs b') := by rw [mapStrs_same hs he]
        _ = tmV name version keys tmpl (.ok (.arr b')) mapStrs := rfl
    | _ => rfl

/-- **Alike parameters name the same templates.** -/
theorem tm_same {ps ps' : Json} (h : Same ps ps') (name : String) (version : Nat) :
    TemplateQuery.templatesOf name version ps = TemplateQuery.templatesOf name version ps' := by
  rw [tm_view name version ps, tm_view name version ps', keysOf_same h, asStr_same h "template"]
  exact tmV_tsv name version _ _ (alikeE_get h "templates")

/-- The parameters' value at each key, as `checkPartitionObject` matches on it. -/
private def pxOf (ps : Json) : String → Option Json := fun k => (ps.getObjVal? k).toOption

/-- The templates a query names at the parameters `ps`. -/
private def tmOf (ps : Json) : String → Nat → Option (V (Array String)) := fun name version =>
  TemplateQuery.templatesOf name version ps

private theorem alike_px {ps ps' : Json} (h : Same ps ps') (k : String) : Alike (pxOf ps k) (pxOf ps' k) := alike_get h k

private theorem tmOf_same {ps ps' : Json} (h : Same ps ps') : tmOf ps = tmOf ps' :=
  funext fun name => funext fun version => tm_same h name version

/-- `HmRow.checkPartitionObject`'s text, with each read of the object an argument (`chkObj_view`). -/
private def chkObjV (okeys : Array String) (fmt : Option String) (prog : String) (qkeys : Array String) (name : String)
    (ver : Option Json) (ps : Json) (pkeys : Array String) (px : String → Option Json)
    (tm : String → Nat → Option (V (Array String))) : V (String × Nat) := do
  if okeys != #["format", "program", "query"] then throw "partition: the object's keys are not format, program, query"
  if fmt != some PARTITION_V1 then throw s!"partition: not a {PARTITION_V1} object"
  if !isSha512Hex prog then throw "partition: program is not a SHA-512 digest"
  if qkeys != #["name", "params", "version"] then throw "partition: the query's keys are not name, version, params"
  if name.isEmpty then throw "partition: the query's name is not a non-empty string"
  let version ← match ver with
    | some (.num n) => if n.exponent == 0 && n.mantissa ≥ 0 then pure n.mantissa.toNat else throw "partition: the query's version is not a natural number"
    | _ => throw "partition: the query's version is not a natural number"
  match ps with
  | .obj _ => pure ()
  | _ => throw "partition: the query's params are not an object"
  if EVALUABLE.contains (name, version) then
    if pkeys != #["W", "X"] then throw s!"partition: Q_word v{version}'s parameters are not X, W"
    for k in ["X", "W"] do
      match px k with
      | some (.num n) => if !(n.exponent == 0 && n.mantissa > 0) then throw s!"partition: {k} is not a positive integer"
      | _ => throw s!"partition: {k} is not a positive integer"
  if (name, version) == Q_CALL then
    if pkeys != #["X"] then throw "partition: Q_call v1's parameters are not X"
    match px "X" with
    | some (.num n) =>
      if !(n.exponent == 0 && n.mantissa > 0 && n.mantissa ≤ Qcall.PARAM_MAX) then
        throw s!"partition: X is not an integer in 1..{Qcall.PARAM_MAX}"
    | _ => throw s!"partition: X is not an integer in 1..{Qcall.PARAM_MAX}"
  if let some ts := tm name version then discard ts
  return (name, version)

set_option smartUnfolding false in
private theorem chkObj_view (obj : Json) :
    HmRow.checkPartitionObject obj =
      chkObjV (keysOf obj) (obj.getObjValAs? String "format").toOption
        ((obj.getObjValAs? String "program").toOption.getD "") (keysOf (sub obj "query"))
        (((sub obj "query").getObjValAs? String "name").toOption.getD "")
        ((sub obj "query").getObjVal? "version").toOption (sub (sub obj "query") "params")
        (keysOf (sub (sub obj "query") "params")) (pxOf (sub (sub obj "query") "params"))
        (tmOf (sub (sub obj "query") "params")) := by rfl

section

variable {ok qk pk : Array String} {f : Option String} {p nm : String} {v : Option Json} {ps : Json}
  {px : String → Option Json} {tm : String → Nat → Option (V (Array String))}

set_option smartUnfolding false in
private theorem chkObjV_ver {va vb : Option Json} (h : Alike va vb) :
    chkObjV ok f p qk nm va ps pk px tm = chkObjV ok f p qk nm vb ps pk px tm := by
  rcases h with ⟨rfl, rfl⟩ | ⟨a, b, rfl, rfl, hs⟩
  · rfl
  · cases hs <;> rfl

set_option smartUnfolding false in
private theorem chkObjV_ps {psa psb : Json} (h : Same psa psb) :
    chkObjV ok f p qk nm v psa pk px tm = chkObjV ok f p qk nm v psb pk px tm := by
  cases h <;> rfl

set_option smartUnfolding false in
private theorem chkObjV_px {pxa pxb : String → Option Json} (hX : Alike (pxa "X") (pxb "X"))
    (hW : Alike (pxa "W") (pxb "W")) :
    chkObjV ok f p qk nm v ps pk pxa tm = chkObjV ok f p qk nm v ps pk pxb tm := by
  unfold chkObjV
  simp only [List.forIn_cons, List.forIn_nil]
  generalize pxa "X" = xa at hX ⊢
  generalize pxb "X" = xb at hX ⊢
  generalize pxa "W" = wa at hW ⊢
  generalize pxb "W" = wb at hW ⊢
  rcases hX with ⟨rfl, rfl⟩ | ⟨a, b, rfl, rfl, hs⟩ <;> rcases hW with ⟨rfl, rfl⟩ | ⟨a', b', rfl, rfl, hs'⟩
  · rfl
  · cases hs' <;> rfl
  · cases hs <;> rfl
  · cases hs <;> cases hs' <;> rfl

end

/-- **Alike partition objects pass the form check alike**, with one query name and version. -/
theorem chkObj_same {a b : Json} (h : Same a b) : HmRow.checkPartitionObject a = HmRow.checkPartitionObject b := by
  have hq := same_sub h "query"
  have hp := same_sub hq "params"
  rw [chkObj_view a, chkObj_view b, keysOf_same h, asStr_same h "format", asStr_same h "program", keysOf_same hq,
    asStr_same hq "name", keysOf_same hp, tmOf_same hp]
  exact (chkObjV_ver (alike_get hq "version")).trans
    ((chkObjV_ps hp).trans (chkObjV_px (pxa := pxOf (sub (sub a "query") "params"))
      (pxb := pxOf (sub (sub b "query") "params")) (alike_px hp "X") (alike_px hp "W")))


theorem digest_same {a b : Json} (h : Same a b) : HmRow.partitionDigest a = HmRow.partitionDigest b := by
  unfold HmRow.partitionDigest
  rw [FlockComplete.JsonRT.canon_same a b h]

/-- `HmRow.checkPartition`'s text at an object, with each read of the object an argument (`chkPart_view`). -/
private def chkPartV (c : Circuit) (obj : Option Json) (chk : V (String × Nat)) (dig : V ByteArray) (prog : Option String) :
    V HmRow.PartitionFinding := do
  let part := (c.metaJ.getObjVal? "partition").toOption.getD Json.null
  let rule := (part.getObjValAs? String "rule").toOption.getD ""
  let digest := (part.getObjValAs? String "digest").toOption.getD ""
  match rule, obj with
  | "verity/partition/v1", none => throw "partition: the circuit binds a verity/partition/v1 digest; the verifier's partition object is required"
  | "verity/partition/v1", some _ =>
    let (name, version) ← chk
    if hexOf (← dig) != digest then
      throw "partition: the circuit's partition digest is not the verifier's partition object's"
    if prog != (c.metaJ.getObjValAs? String "program_sha512").toOption then
      throw "partition: the partition object is over another program than META's program_sha512"
    let note := if (TemplateQuery.templatesOf name version Json.null).isSome then
        s!"units taken as stated, not derived: the verifier's copy of the program was not given (--program) to evaluate {name} v{version}"
      else if EVALUABLE.contains (name, version) || (name, version) == Q_CALL then
        s!"units taken as stated, not derived: the verifier's copy of the program was not given (--program) to evaluate {name} v{version}"
      else s!"units taken as stated, not derived: the verifier does not evaluate {name} v{version}"
    return { rule, query := s!"{name} v{version}", digest, derived := false, note }
  | _, some _ => throw "partition: the circuit binds the provisional unit cover, not a verity/partition/v1 object"
  | _, none => return { rule, digest, derived := false, note := "units taken as stated, not derived: the provisional unit cover names no query" }

set_option smartUnfolding false in
private theorem chkPart_view (c : Circuit) (o : Json) :
    HmRow.checkPartition c (some o) =
      chkPartV c (some o) (HmRow.checkPartitionObject o) (HmRow.partitionDigest o)
        (o.getObjValAs? String "program").toOption := by rfl

set_option smartUnfolding false in
private theorem chkPartV_obj (c : Circuit) (a b : Json) (chk : V (String × Nat)) (dig : V ByteArray) (prog : Option String) :
    chkPartV c (some a) chk dig prog = chkPartV c (some b) chk dig prog := by rfl

/-- **Alike partition objects pass the binding check alike.** -/
theorem chkPart_same {a b : Json} (h : Same a b) (c : Circuit) :
    HmRow.checkPartition c (some a) = HmRow.checkPartition c (some b) := by
  rw [chkPart_view c a, chkPart_view c b, chkObj_same h, digest_same h, asStr_same h "program"]
  exact chkPartV_obj c a b _ _ _

/-- The binding check passes at an object only where the object's tagged digest returned. -/
private theorem chkPartV_dig {c : Circuit} {o : Json} {chk : V (String × Nat)} {dig : V ByteArray}
    {prog : Option String} {f : HmRow.PartitionFinding} (h : chkPartV c (some o) chk dig prog = .ok f) :
    ∃ d, dig = .ok d := by
  cases dig with
  | ok d => exact ⟨d, rfl⟩
  | error e =>
    exfalso
    unfold chkPartV at h
    dsimp only at h
    split at h
    · cases h
    · cases chk with
      | error e' => cases h
      | ok nv => cases h
    · cases h
    · rename_i hn hx
      cases hn

/-- **A partition object that passes the binding check has a canonical text**: the check passes at an object only
under a `verity/partition/v1` binding, where it hashes the object's canonical text (`HmRow.partitionDigest`). -/
theorem canon_of_chkPart {c : Circuit} {o : Json} {f : HmRow.PartitionFinding}
    (h : HmRow.checkPartition c (some o) = .ok f) : ∃ s, Flock.canon o = .ok s := by
  rw [chkPart_view] at h
  obtain ⟨d, hd⟩ := chkPartV_dig h
  cases hc : Flock.canon o with
  | ok s => exact ⟨s, rfl⟩
  | error e =>
    unfold HmRow.partitionDigest at hd
    rw [hc] at hd
    cases hd

theorem derive_same {a b : Json} (h : Same a b) (c : Circuit) (program : ByteArray) (indices : Array Nat) :
    HmRow.deriveTemplateUnits c a program indices = HmRow.deriveTemplateUnits c b program indices := by
  have hq := same_sub h "query"
  unfold HmRow.deriveTemplateUnits
  simp only [asStr_same hq "name", asNat_same hq "version", tm_same (same_sub hq "params"), asStr_same h "program"]

theorem qwordUnits_same {a b : Json} (h : Same a b) (c : Circuit) (program : ByteArray) (indices : Array Nat) :
    HmRow.deriveQwordUnits c a program indices = HmRow.deriveQwordUnits c b program indices := by
  have hq := same_sub h "query"
  have hp := same_sub hq "params"
  unfold HmRow.deriveQwordUnits
  simp only [asNat_same hq "version", asNat_same hp "X", asNat_same hp "W", asStr_same h "program"]

theorem qcallUnits_same {a b : Json} (h : Same a b) (c : Circuit) (program : ByteArray) (indices : Array Nat) :
    HmRow.deriveQcallUnits c a program indices = HmRow.deriveQcallUnits c b program indices := by
  have hp := same_sub (same_sub h "query") "params"
  unfold HmRow.deriveQcallUnits
  simp only [asNat_same hp "X", asStr_same h "program"]

theorem qword_same {a b : Json} (h : Same a b) (c : Circuit) (pub : Public) (program : ByteArray)
    (f : HmRow.PartitionFinding) : HmRow.qwordFinding c pub a program f = HmRow.qwordFinding c pub b program f := by
  have hq := same_sub h "query"
  unfold HmRow.qwordFinding
  simp only [asStr_same hq "name", asNat_same hq "version", qwordUnits_same h, qcallUnits_same h]

theorem strata_same {a b : Json} (h : Same a b) (law : String) (program : ByteArray) :
    HmRow.templateStrataWith With.std law a program = HmRow.templateStrataWith With.std law b program := by
  have hp := same_sub (same_sub h "query") "params"
  unfold HmRow.templateStrataWith
  simp only [chkObj_same h, tm_same hp, asStr_same h "program"]

theorem strat_same {a b : Json} (h : Same a b) (program : ByteArray) (k : Nat) :
    HmRow.stratifiedLaw a program k = HmRow.stratifiedLaw b program k := by
  rw [← With.stratifiedLawWith_std a program k, ← With.stratifiedLawWith_std b program k]
  unfold HmRow.stratifiedLawWith
  rw [strata_same h]

theorem work_same {a b : Json} (h : Same a b) (program : ByteArray) (k : Nat) (table : Array (String × Nat × Nat)) :
    HmRow.workLaw a program k table = HmRow.workLaw b program k table := by
  rw [← With.workLawWith_std a program k table, ← With.workLawWith_std b program k table]
  unfold HmRow.workLawWith
  rw [strata_same h]

private theorem mk_congr {c c' : Circuit → V HmRow.PartitionFinding} {s s' : ByteArray → Nat → V (Draw.Law × Nat)}
    {w w' : ByteArray → Nat → Array (String × Nat × Nat) → V (Draw.Law × Nat)} {t t' : Bool}
    {d d' : Circuit → ByteArray → Array Nat → V String}
    {q q' : Circuit → Public → ByteArray → HmRow.PartitionFinding → V (Option HmRow.PartitionFinding)}
    (h1 : c = c') (h2 : s = s') (h3 : w = w') (h4 : t = t') (h5 : d = d') (h6 : q = q') :
    PartView.mk c s w t d q = PartView.mk c' s' w' t' d' q' := by
  subst h1 h2 h3 h4 h5 h6
  rfl

/-- **Alike partition objects have one view**: every reader of the session's setup returns the same on both. -/
theorem view_same {a b : Json} (h : Same a b) : view a = view b := by
  have hq := same_sub h "query"
  have e1 : (fun c => HmRow.checkPartition c (some a)) = fun c => HmRow.checkPartition c (some b) :=
    funext fun c => chkPart_same h c
  have e2 : HmRow.stratifiedLaw a = HmRow.stratifiedLaw b := funext fun p => funext fun k => strat_same h p k
  have e3 : HmRow.workLaw a = HmRow.workLaw b :=
    funext fun p => funext fun k => funext fun t => work_same h p k t
  have e4 : (TemplateQuery.templatesOf (((sub a "query").getObjValAs? String "name").toOption.getD "")
        (((sub a "query").getObjValAs? Nat "version").toOption.getD 0) Json.null).isSome =
      (TemplateQuery.templatesOf (((sub b "query").getObjValAs? String "name").toOption.getD "")
        (((sub b "query").getObjValAs? Nat "version").toOption.getD 0) Json.null).isSome := by
    rw [asStr_same hq "name", asNat_same hq "version"]
  have e5 : (fun c => HmRow.deriveTemplateUnits c a) = fun c => HmRow.deriveTemplateUnits c b :=
    funext fun c => funext fun p => funext fun i => derive_same h c p i
  have e6 : (fun c pub => HmRow.qwordFinding c pub a) = fun c pub => HmRow.qwordFinding c pub b :=
    funext fun c => funext fun pub => funext fun p => funext fun f => qword_same h c pub p f
  exact mk_congr e1 e2 e3 e4 e5 e6

/-- **Partitions the session's setup reads alike**: one partition, or two objects that are alike. -/
def PartSame (x y : Option Json) : Prop := x = y ∨ ∃ a b, x = some a ∧ y = some b ∧ Same a b

theorem PartSame.map_view {x y : Option Json} (h : PartSame x y) : x.map view = y.map view := by
  rcases h with rfl | ⟨a, b, rfl, rfl, hs⟩
  · rfl
  · exact congrArg some (view_same hs)

/-- **Alike partitions set the session up alike.** -/
theorem setupTables_same (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : Nat)
    (tables : List (String × ByteArray)) (draw : Option Json) {pa pb : Option Json} (h : PartSame pa pb)
    (program : Option ByteArray) (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) :
    Stmt.setupTables tags circuitFile publicFile coins count tables draw pa program work stratified closure
        publicInputs =
      Stmt.setupTables tags circuitFile publicFile coins count tables draw pb program work stratified closure
        publicInputs :=
  setupTables_congr tags circuitFile publicFile coins count tables draw h.map_view program work stratified closure
    publicInputs

end Flock.PartCong
