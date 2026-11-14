module

public import Proofs.Properties.Verity.SampledTwo.FormSetup
public import Proofs.Properties.Verity.SampledTwo.FormStmts
public import Proofs.Flock.SpotFlock.ParamsWf
public import Proofs.Core.CanonicalJson
public import Batteries.Tactic.OpenPrivate
import all Lean.Data.Json.Basic

/-!
# What each member's statement prints of the member reads back

`SessionCoins.StmtsPrint` (`FormSetup.lean`) asks, of each member a template's statement names, that its partition
object is a well-formed tree (`Claims.Json.Wf`) and its registered reads printable (`Claims.Json.Good`: every object
a well-formed tree, every number an integer). Here it is a theorem, `stmtsPrint_of`, with no hypothesis on the play.

The form's parameters are read from the tree `Core.canonicalJson` returns of the parameters' bytes
(`SessionMsgs.formOf`), which is printable (`Verity.CanonicalJson.canonicalJson_good`). Each reader on the way to a
member hands on a part of the tree it was given, or that tree with one key's value replaced by a printable one:

* a field of a printable value is printable (`good_getD`, `good_field?`), and so are its entries (`good_entries`);
* a printable object less a key is printable (`good_less`), and so is one with a key's value replaced by a
  printable value (`good_put`): its keys are distinct, and none of the rest is the key put;
* the form splits off its `form` key (`split_good`) and reads the interior audit's parameters from the interior's
  `params` with its `proofs` replaced by the interior's `members` (`members_print`);
* the per-member form's reader reads the members from that `proofs` and the rest as the one statement's parameters,
  at `proofs` of no file (`membersParams_print`);
* each member's registered reads are the `registered` field of its entry of the members' object, and its partition
  object is the one statement's (`membersOf_reads`), which is the `partition` field of the tree less its `proofs`
  (`partition_wf_zk`, `SpotFlock.ParamsWf.partition_wf`).

The readers are walked join point by join point (`SpotFlock.DoWalk`).
-/

@[expose] public section

namespace Verity.SampledTwo

namespace FormPrint

open Lean (Json)
open Claims.Json (Good Wf)
open Spot (getD field? entries)
open FlockComplete.JsonRT (Distinct good_mkObj)
open Properties.Spot.SecurityProofs (bind_eq_ok)
open Properties.Spot.Units (field?_obj)
open SpotFlock.ReadsJson (getElem?_of_mem)
open SpotFlock.ParamsWf (distinct_toList partition_wf array_mapM_mem)

/-! ## Printable values -/

/-- **A printable value is well formed.** -/
theorem wf_of_good {j : Json} (h : Good j) : Wf j := by
  induction h with
  | null => exact .null
  | bool b => exact .bool b
  | num m => exact .num _
  | str s => exact .str s
  | arr _ ih => exact .arr ih
  | obj hm _ ih => exact .obj hm ih

/-- **A field of a printable value is printable**, read as `Spot.getD` reads it. -/
theorem good_getD {u : Json} (hu : Good u) (k : String) : Good (getD u k) := by
  cases hu with
  | obj hm h =>
    simp only [getD, field?_obj]
    cases hk : Std.TreeMap.Raw.get? _ k with
    | none => exact .null
    | some v => exact h k v hk
  | _ => exact .null

/-- **A field a printable value holds is printable.** -/
theorem good_field? {u v : Json} (hu : Good u) {k : String} (h : field? u k = some v) : Good v := by
  have e : getD u k = v := by
    unfold Spot.getD
    rw [h]
    rfl
  rw [← e]
  exact good_getD hu k

/-- **The entries of a printable value are printable.** -/
theorem good_entries {j : Json} (hj : Good j) : ∀ kv ∈ entries j, Good kv.2 := by
  intro kv hkv
  cases hj with
  | obj hm hv =>
    refine hv kv.1 kv.2 ?_
    rw [Std.TreeMap.Raw.get?_eq_getElem?]
    exact getElem?_of_mem hm hkv
  | _ => simp [Spot.entries] at hkv

/-- The entries of a value less the key `k`, as the parameters' readers rebuild an object without one of its keys. -/
def less (j : Json) (k : String) : List (String × Json) := ((entries j).filter (·.1 != k)).toList

theorem less_distinct {j : Json} (hj : Good j) (k : String) : Distinct (less j k) := by
  cases hj with
  | obj hm _ =>
    show ((_ : Array (String × Json)).filter (·.1 != k)).toList.Pairwise _
    rw [Array.toList_filter]
    exact (distinct_toList hm).filter _
  | _ => simp [less, Spot.entries]

theorem less_ne {j : Json} {k : String} {kv : String × Json} (h : kv ∈ less j k) : kv.1 ≠ k := by
  have h' := (Array.mem_filter.1 (Array.mem_toList_iff.1 h)).2
  simpa using h'

theorem less_good {j : Json} (hj : Good j) {k : String} {kv : String × Json} (h : kv ∈ less j k) : Good kv.2 :=
  good_entries hj kv (Array.mem_filter.1 (Array.mem_toList_iff.1 h)).1

/-- **A printable object less a key is printable.** -/
theorem good_less {j : Json} (hj : Good j) (k : String) : Good (Json.mkObj (less j k)) :=
  good_mkObj (less_distinct hj k) fun _ h => less_good hj h

/-- **A printable object with a key's value replaced by a printable one is printable.** -/
theorem good_put {j v : Json} (hj : Good j) (hv : Good v) (k : String) :
    Good (Json.mkObj (less j k ++ [(k, v)])) := by
  refine good_mkObj ?_ fun p hp => ?_
  · refine List.pairwise_append.2 ⟨less_distinct hj k, List.pairwise_singleton _ _, fun a ha b hb => ?_⟩
    obtain rfl := List.mem_singleton.1 hb
    intro hc
    exact less_ne ha (Std.LawfulEqCmp.eq_of_compare hc)
  · rcases List.mem_append.1 hp with hp | hp
    · exact less_good hj hp
    · obtain rfl := List.mem_singleton.1 hp
      exact hv

/-- A list of entries with no key twice has distinct keys, as `Json.mkObj` reads them. -/
theorem distinct_of_nodup {l : List (String × Json)} (h : (l.map (·.1)).Nodup) : Distinct l :=
  (List.pairwise_map.1 h).imp fun {_ _} hab hc => hab (Std.LawfulEqCmp.eq_of_compare hc)

/-! ## The one statement's readers (`Spot.OneStageZk`) -/

section zk

open private without committedOf in Spot.OneStageZk.paramsOfJson

private theorem without_less (j : Json) (k : String) : without j k = less j k := by
  cases j <;> simp [without, less, Spot.entries]

/-- **The registered reads `committedOf` returns are the tree's `registered` field.** -/
private theorem committedOf_reads {m : Json} {w : String × String × Option Json} (h : committedOf m = .ok w) :
    ∀ r, w.2.2 = some r → r = getD m "registered" := by
  unfold committedOf at h
  repeat' first
    | (jp_step h to w with (fun x : Option Json => ∀ r, x = some r → r = getD m "registered") by
        intro r hr; subst hr; rename_i hp; cases hp; first | (symm; assumption) | done)
    | (have h' := bind_eq_ok.1 h; clear h; obtain ⟨_, _, h⟩ := h')
    | throws h
    | split at h
  cases h
  assumption

/-- **The partition object `Spot.OneStage.paramsOfJson` returns is well formed**, where the tree it read is. -/
theorem partition_wf_base {rest : Json} {b : Spot.OneStage.Params} {o : Json}
    (hb : Spot.OneStage.paramsOfJson rest = .ok b) (ho : b.verifier.partition = some o) (hrest : Wf rest) :
    Wf o := by
  unfold Spot.OneStage.paramsOfJson at hb
  obtain ⟨q, hq, hb⟩ := bind_eq_ok.1 hb
  split at hb
  · cases hb
  cases hb
  exact partition_wf hrest hq ho

/-- **The one statement's partition object is well formed**, where the tree its parameters were read from is
printable. -/
theorem partition_wf_zk {J : Json} {p : Spot.OneStageZk.Params} (hJ : Good J)
    (h : Spot.OneStageZk.paramsOfJson J = .ok p) : Wf p.own.partition := by
  unfold Spot.OneStageZk.paramsOfJson at h
  do_walk h to p
  all_goals
    cases h
    refine partition_wf_base (by assumption) (by assumption) ?_
    rw [without_less]
    exact wf_of_good (good_less hJ "proofs")

/-- **What `membersOf` hands each member**: the shared partition object, and for its registered reads the
`registered` field of an entry of the members' object. -/
theorem membersOf_reads {shared : Spot.OneStageZk.Own} {j : Json} {ms : Array Spot.OneStageZk.Member}
    (h : Spot.OneStageZk.membersOf shared j = .ok ms) :
    ∀ m ∈ ms, m.own.partition = shared.partition ∧
      ∃ kv ∈ entries j, ∀ r, m.own.registered = some r → r = getD kv.2 "registered" := by
  unfold Spot.OneStageZk.membersOf at h
  do_walk h to ms
  all_goals
    cases h
    intro m hm
    obtain ⟨⟨name, mj⟩, hx, hf⟩ := array_mapM_mem (by assumption) m hm
    dsimp only at hf
    do_walk hf to m
    all_goals
      cases hf
      exact ⟨rfl, _, hx, committedOf_reads (by assumption)⟩

/-- **Each member of the per-member form's parameters has a well-formed partition object and printable registered
reads**, where the tree the parameters were read from is printable. -/
theorem membersParams_print {J : Json} {a : Spot.OneStageZk.Params} (hJ : Good J)
    (h : Spot.OneStageZk.membersParamsOfJson J = .ok a) :
    ∀ ms, a.members = some ms → ∀ m ∈ ms, Wf m.own.partition ∧ ∀ r, m.own.registered = some r → Good r := by
  unfold Spot.OneStageZk.membersParamsOfJson at h
  do_walk h to a
  all_goals
    cases h
    intro ms hms m hm
    obtain rfl := Option.some.inj hms
    have hpr := good_field? hJ (by assumption)
    obtain ⟨hpart, kv, hkv, hreg⟩ := membersOf_reads (by assumption) m hm
    refine ⟨?_, fun r hr => ?_⟩
    · rw [hpart]
      refine partition_wf_zk ?_ (by assumption)
      rw [without_less]
      refine good_put hJ (good_mkObj ?_ ?_) "proofs"
      · exact distinct_of_nodup (by simp)
      · intro p hp
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
        rcases hp with rfl | rfl | rfl
        · exact .str _
        · exact .str _
        · exact .null
    · rw [hreg r hr]
      exact good_getD (good_entries (good_getD hpr "members") kv hkv) "registered"

end zk

/-! ## The two-stage form's parameters -/

/-- **The form's rest of a printable tree is printable** (`split`). -/
theorem split_good {j j' : Json} {what : String} (hj : Good j)
    (h : VStar.Recursive.SampledTwo.split j what = .ok j') : Good j' := by
  unfold VStar.Recursive.SampledTwo.split at h
  do_walk h to j'
  all_goals
    cases h
    exact good_less hj "form"

/-- **Each member of the form's parameters has a well-formed partition object and printable registered reads**,
where the tree the parameters were read from is printable. -/
theorem members_print {j : Json} {p : VStar.Recursive.SampledTwo.Params} (hj : Good j)
    (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) :
    ∀ m ∈ p.members, Wf m.own.partition ∧ ∀ r, m.own.registered = some r → Good r := by
  replace h := (VStar.Recursive.SampledTwo.paramsOfJson_ok h).1
  unfold VStar.Recursive.SampledTwo.paramsParsed at h
  do_walk h to p
  all_goals
    cases h
    have hj' := split_good hj (by assumption)
    refine membersParams_print ?_ (by assumption) _ (by assumption)
    refine good_put (good_getD (good_getD hj' "interior") "params") (good_mkObj ?_ ?_) "proofs"
    · exact List.pairwise_singleton _ _
    · intro p hp
      obtain rfl := List.mem_singleton.1 hp
      exact good_getD (good_getD hj' "interior") "members"

/-- **The form's parameters of a case are read from the tree `Core.canonicalJson` returns of its parameters'
bytes.** -/
theorem formOf_params {k : Case} {p : VStar.Recursive.SampledTwo.Params}
    {e : VStar.Recursive.SampledTwo.Evidence} (h : SessionMsgs.formOf k = some (p, e)) :
    ∃ j, Core.canonicalJson k.params = .ok j ∧ VStar.Recursive.SampledTwo.paramsOfJson j = .ok p := by
  unfold SessionMsgs.formOf at h
  cases hp : (Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson) with
  | error w =>
    rw [hp] at h
    cases h
  | ok p' =>
    cases he : VStar.Recursive.SampledTwo.evidenceOf k.evidence with
    | error w =>
      rw [hp, he] at h
      cases h
    | ok e' =>
      rw [hp, he] at h
      cases h
      exact bind_eq_ok.1 hp

end FormPrint

/-- **`SessionCoins.StmtsPrint`**, at any case and any plays: each member of the form's parameters has a well-formed
partition object and printable registered reads (`FormPrint.members_print`), the parameters being read from the
tree `Core.canonicalJson` returns, which is printable (`Verity.CanonicalJson.canonicalJson_good`). -/
theorem stmtsPrint_of (C : CaseOf) (Good : Verity.Exchange → Prop) : SessionCoins.StmtsPrint C Good := by
  intro o k p e _ _ _ hf t m hm
  obtain ⟨j, hj, hp⟩ := FormPrint.formOf_params hf
  exact FormPrint.members_print (Verity.CanonicalJson.canonicalJson_good hj) hp m (memberAt_mem hm)

end Verity.SampledTwo
