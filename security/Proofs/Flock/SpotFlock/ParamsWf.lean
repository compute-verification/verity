module

public import Proofs.Flock.SpotFlock.HeldOfReads
public import Proofs.Flock.SpotFlock.DoWalk
public import Proofs.Properties.Spot.Units
public import Batteries.Tactic.OpenPrivate

/-!
# What the parameters' readers hand on of a well-formed tree

Three things the two-stage form's proofs take as premises are properties of the tree its parameters were read from,
and hold where that tree is well formed (`Claims.Json.Wf`: every object a well-formed map):

* **the partition object** (`partition_wf`): the object `Spot.OneStage.paramsParsed` returns as the verifier's
  partition is the tree's `partition` field, so it is well formed;
* **the registered reads** (`readsWf_of_params`): each template's registered reads, as
  `VStar.Recursive.SampledTwo.paramsOfJson` returns them in `circuits`, is a field of an entry of the interior's
  `proofs`, so they are well formed (`SpotFlock.HeldOf.ReadsWf`);
* **the ports** (`ports_nodup`): the ports `paramsParsed` returns are the entries of the tree's `ports` object, so no
  name is there twice.

The form reads its boundary's and its interior's parameters through `Core.canonicalJson` (`paramsAt`), so for those
two the tree is the one `canonicalJson` returned (`paramsAt_ok`); `ports_nodup_form` and `partition_wf_form` state
the ports and the partition there.

## How the readers are walked

`paramsParsed` is one `do` block whose `match`es and `if`s each leave a join point for the rest of the block. It is
walked once, join point by join point (`jp_step`, `do_walk` of `SpotFlock.DoWalk`), which is linear in the block.

A `match` on an array literal (the one key `templates`, an entry's four keys) is one `split` does not split:
`match_templates_ok` and `match_entry_ok` take its cases by hand, once, for any arms.

## What this does not show

That the tree `Core.canonicalJson` returns is well formed: it is the hypothesis `hcj` of the theorems that name
`canonicalJson`, and holds where `canonicalJson` parses with the verifier's own parser
(`FlockComplete.JsonRT.parse_wf`).
-/

@[expose] public section

namespace SpotFlock.ParamsWf

open Lean (Json)
open Claims.Json (Wf)
open Spot (getD)
open Properties.Spot.SecurityProofs (bind_eq_ok)
open Properties.Spot.Units (wf_getD)
open SpotFlock.ReadsJson (wf_obj wf_mkObj getElem?_of_mem)
open SpotFlock.HeldOf (ReadsWf)
open private obj from Spot.OneStage
open private without from VStarRecursive.SampledTwo

/-! ## `mapM` in `Except` -/

theorem list_mapM_mem {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {r : List β}, l.mapM f = .ok r → ∀ y ∈ r, ∃ x ∈ l, f x = .ok y
  | [], r, h, y, hy => by
    rw [List.mapM_nil] at h
    cases h
    cases hy
  | a :: l, r, h, y, hy => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := bind_eq_ok.1 h
    obtain ⟨bs, hl, h⟩ := bind_eq_ok.1 h
    cases h
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨a, List.mem_cons_self, hb⟩
    · obtain ⟨x, hx, hxy⟩ := list_mapM_mem hl y hy
      exact ⟨x, List.mem_cons_of_mem _ hx, hxy⟩

/-- **Each value an `Array.mapM` returns is the function's at an element.** -/
theorem array_mapM_mem {α β ε : Type} {f : α → Except ε β} {a : Array α} {r : Array β} (h : a.mapM f = .ok r) :
    ∀ y ∈ r, ∃ x ∈ a, f x = .ok y := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : a.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    cases h
    intro y hy
    obtain ⟨x, hx, hxy⟩ := list_mapM_mem hl y (Array.mem_toList_iff.2 hy)
    exact ⟨x, Array.mem_toList_iff.1 hx, hxy⟩

theorem list_mapM_self {α ε : Type} {f : α → Except ε α} (hf : ∀ a b, f a = .ok b → b = a) :
    ∀ {l r : List α}, l.mapM f = .ok r → r = l
  | [], r, h => by
    rw [List.mapM_nil] at h
    cases h
    rfl
  | a :: l, r, h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := bind_eq_ok.1 h
    obtain ⟨bs, hl, h⟩ := bind_eq_ok.1 h
    cases h
    rw [hf a b hb, list_mapM_self hf hl]

/-- **An `Array.mapM` of a function that returns its argument or throws returns the array.** -/
theorem array_mapM_self {α ε : Type} {f : α → Except ε α} (hf : ∀ a b, f a = .ok b → b = a) {a r : Array α}
    (h : a.mapM f = .ok r) : r = a := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : a.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    cases h
    rw [list_mapM_self hf hl]

/-! ## The entries of a well-formed object -/

/-- **The entries of a well-formed value are well formed.** -/
theorem wf_entries {j : Json} (hj : Wf j) : ∀ kv ∈ Spot.entries j, Wf kv.2 := by
  intro kv hkv
  cases j with
  | obj m =>
    obtain ⟨hm, hv⟩ := wf_obj hj
    exact hv kv.1 kv.2 (getElem?_of_mem hm hkv)
  | _ => simp [Spot.entries] at hkv

/-- The entries of a well-formed object, in its order, have distinct keys. -/
theorem distinct_toList {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) :
    FlockComplete.JsonRT.Distinct m.toArray.toList := by
  rw [Std.TreeMap.Raw.toList_toArray]
  exact Std.TreeMap.Raw.distinct_keys_toList hm

/-- **No key is twice among the entries of a well-formed value.** -/
theorem entries_nodup {j : Json} (hj : Wf j) : ((Spot.entries j).map (·.1)).toList.Nodup := by
  cases j with
  | obj m =>
    obtain ⟨hm, -⟩ := wf_obj hj
    show (m.toArray.map (·.1)).toList.Nodup
    rw [Array.toList_map]
    refine List.pairwise_map.2 ((distinct_toList hm).imp fun {a b} hab e => hab ?_)
    rw [e]
    exact Std.ReflCmp.compare_self
  | _ => simp [Spot.entries]

/-- **A well-formed object less a key is well formed**, as the form's parser rebuilds it (`without`). -/
private theorem wf_without {j : Json} (hj : Wf j) (k : String) : Wf (Json.mkObj (without j k)) := by
  refine wf_mkObj ?_ fun kv hkv => wf_entries hj kv ?_
  · cases j with
    | obj m =>
      obtain ⟨hm, -⟩ := wf_obj hj
      show ((m.toArray.filter (·.1 != k)).toList).Pairwise _
      rw [Array.toList_filter]
      exact (distinct_toList hm).filter _
    | _ => simp [without, Spot.entries, FlockComplete.JsonRT.Distinct]
  · exact (Array.mem_filter.1 (Array.mem_toList_iff.1 hkv)).1

/-! ## The stream's parameters (`Spot.OneStage.paramsParsed`) -/

private theorem obj_ok {a b : Json} {w : String} (h : obj a w = .ok b) : b = a := by
  unfold obj at h
  split at h
  · exact (Except.ok.inj h).symm
  · cases h

/-- **The parameters' ports are the entries of the tree's `ports`, and their partition object is its `partition`.** -/
theorem paramsParsed_reads {j : Json} {p : Spot.OneStage.Params} (h : Spot.OneStage.paramsParsed j = .ok p) :
    p.verifier.ports = Spot.entries (getD j "ports") ∧ ∀ o, p.verifier.partition = some o → o = getD j "partition" := by
  unfold Spot.OneStage.paramsParsed at h
  repeat' first
    | (jp_step h to p with (fun w : Option Json => ∀ o, w = some o → o = getD j "partition") by
        intro o ho; cases ho <;> (apply obj_ok; assumption))
    | (jp_step h to p with (fun _ => True) by trivial)
    | head_zeta at h
    | (have h' := bind_eq_ok.1 h; clear h; obtain ⟨_, _, h⟩ := h')
    | throws h
    | split at h
  cases h
  refine ⟨?_, by assumption⟩
  have ho := obj_ok (by assumption : obj (getD j "ports") "ports" = Except.ok _)
  subst ho
  refine array_mapM_self ?_ (by assumption)
  rintro ⟨q, v⟩ b hb
  repeat' first
    | exact (Except.ok.inj hb).symm
    | (have h' := bind_eq_ok.1 hb; clear hb; obtain ⟨_, _, hb⟩ := h')
    | throws hb
    | split at hb

/-- **The parameters list no port twice**, where the tree they were read from is well formed. -/
theorem ports_nodup {j : Json} {p : Spot.OneStage.Params} (hj : Wf j) (h : Spot.OneStage.paramsParsed j = .ok p) :
    (p.verifier.ports.map (·.1)).toList.Nodup := by
  rw [(paramsParsed_reads h).1]
  exact entries_nodup (wf_getD hj "ports")

/-- **The parameters' partition object is well formed**, where the tree they were read from is. -/
theorem partition_wf {j : Json} {p : Spot.OneStage.Params} (hj : Wf j) (h : Spot.OneStage.paramsParsed j = .ok p)
    {o : Json} (ho : p.verifier.partition = some o) : Wf o := by
  rw [(paramsParsed_reads h).2 o ho]
  exact wf_getD hj "partition"

/-! ## The templates' circuits (`Spot.Run.templateCircuits`) -/

/-- **A `match` on the one key `templates` and a list returns only at a list**, in its first arm. The pattern is an
array literal, which `split` does not split, so the cases are taken by hand. -/
theorem match_templates_ok {α ε : Type} {ko : Option (Array String)} {d : Json} {A : Array Json → Except ε α}
    {e : ε} {r : α}
    (h : (match (ko, d) with
      | (some #["templates"], .arr es) => A es
      | _ => throw e) = .ok r) : ∃ es, d = .arr es ∧ A es = .ok r := by
  rcases ko with _ | ⟨⟨l⟩⟩
  · cases h
  rcases l with _ | ⟨a, _ | ⟨b, t⟩⟩
  · cases h
  · by_cases ha : a = "templates"
    case neg => exfalso; lit_walk h ha
    subst ha
    cases d with
    | arr es => exact ⟨es, rfl, h⟩
    | _ => cases h
  · cases h

/-- **A `match` on an entry's four keys and its fields returns only in its first arm**, a tuple whose last field is
the entry's last. -/
theorem match_entry_ok {ε : Type} {ko : Option (Array String)} {a b c d : Json}
    {cond : String → String → Std.TreeMap.Raw String Json compare → Json → Bool} {e₁ e₂ : ε}
    {r : String × String × Json × Json}
    (h : (match ko, a, b, c, d with
      | some #["circuit", "header", "registered", "template"], .str t, .str cc, .obj hh, rr =>
        if cond t cc hh rr then pure (t, cc, Json.obj hh, rr) else throw e₁
      | _, _, _, _, _ => (throw e₂ : Except ε (String × String × Json × Json))) = .ok r) : r.2.2.2 = d := by
  rcases ko with _ | ⟨⟨l⟩⟩
  · cases h
  rcases l with _ | ⟨k₁, _ | ⟨k₂, _ | ⟨k₃, _ | ⟨k₄, _ | ⟨k₅, l⟩⟩⟩⟩⟩
  · cases h
  · cases h
  · cases h
  · cases h
  · by_cases h₁ : k₁ = "circuit"
    case neg => exfalso; lit_walk h h₁
    subst h₁
    by_cases h₂ : k₂ = "header"
    case neg => exfalso; lit_walk h h₂
    subst h₂
    by_cases h₃ : k₃ = "registered"
    case neg => exfalso; lit_walk h h₃
    subst h₃
    by_cases h₄ : k₄ = "template"
    case neg => exfalso; lit_walk h h₄
    subst h₄
    cases a <;> first | throws h | skip
    cases b <;> first | throws h | skip
    cases c <;> first | throws h | skip
    rename_i t cc hh
    have h' : (if cond t cc hh d = true then pure (t, cc, Json.obj hh, d) else throw e₁ :
        Except ε (String × String × Json × Json)) = .ok r := h
    split at h'
    · cases h'
      rfl
    · cases h'
  · cases h

/-- **The templates' registered reads are well-formed trees**, where the `proofs` they were read from are: each
is the `registered` field of an entry of its `templates`. -/
theorem templateCircuits_readsWf {proofs query : Json} {cs : Array (String × String × Json × Json)}
    (hj : Wf proofs) (h : Spot.Run.templateCircuits proofs query = .ok cs) : ReadsWf cs := by
  unfold Spot.Run.templateCircuits at h
  do_walk h to cs
  · cases h
    intro c hc
    simp at hc
  · obtain ⟨es, hts, h⟩ := match_templates_ok h
    do_walk h to cs
    cases h
    intro c hc
    obtain ⟨e, he, hce⟩ := array_mapM_mem (by assumption) c hc
    have hwe : Wf e := by
      have hw := wf_getD hj "templates"
      rw [hts] at hw
      cases hw with
      | arr hall => exact hall e he
    rw [match_entry_ok hce]
    exact wf_getD hwe "registered"

/-! ## The two-stage form's parameters (`VStar.Recursive.SampledTwo.paramsOfJson`) -/

/-- **The form's rest of a well-formed tree is well formed** (`split`). -/
theorem split_wf {j j' : Json} {what : String} (hj : Wf j) (h : VStar.Recursive.SampledTwo.split j what = .ok j') :
    Wf j' := by
  unfold VStar.Recursive.SampledTwo.split at h
  do_walk h to j'
  cases h
  exact wf_without hj "form"

/-- **Parameters `paramsAt` returns are `paramsParsed`'s of the tree `Core.canonicalJson` returned.** -/
theorem paramsAt_ok {x : Json} {q : Spot.OneStage.Params} (h : VStar.Recursive.SampledTwo.paramsAt x = .ok q) :
    ∃ b j, Core.canonicalJson b = .ok j ∧ Spot.OneStage.paramsParsed j = .ok q := by
  unfold VStar.Recursive.SampledTwo.paramsAt at h
  obtain ⟨s, -, h⟩ := bind_eq_ok.1 h
  obtain ⟨j, hj, h⟩ := bind_eq_ok.1 h
  exact ⟨_, j, hj, h⟩

/-- **What the form's parameters are read from**: the boundary's and the interior's by `paramsAt`, and the
templates' circuits by `Spot.Run.templateCircuits` from the `proofs` of the interior's `params`, in the form's rest
of the tree. -/
theorem paramsOfJson_reads {j : Json} {p : VStar.Recursive.SampledTwo.Params}
    (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) :
    ∃ j' xb xi query, VStar.Recursive.SampledTwo.split j "parameters" = .ok j' ∧
      VStar.Recursive.SampledTwo.paramsAt xb = .ok p.boundary ∧
      VStar.Recursive.SampledTwo.paramsAt xi = .ok p.interior ∧
      Spot.Run.templateCircuits (getD (getD (getD j' "interior") "params") "proofs") query = .ok p.circuits := by
  unfold VStar.Recursive.SampledTwo.paramsOfJson at h
  do_walk h to p
  cases h
  exact ⟨_, _, _, _, by assumption, by assumption, by assumption, by assumption⟩

/-- **The form's templates' registered reads are well-formed trees** (`ReadsWf`, the premise of
`SpotFlock.HeldOf.refReadsWf_form`), where the tree the form's parameters were read from is well formed. -/
theorem readsWf_of_params {j : Json} {p : VStar.Recursive.SampledTwo.Params} (hj : Wf j)
    (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) : ReadsWf p.circuits := by
  obtain ⟨j', _, _, query, hs, -, -, hc⟩ := paramsOfJson_reads h
  exact templateCircuits_readsWf (wf_getD (wf_getD (wf_getD (split_wf hj hs) "interior") "params") "proofs") hc

/-- **`ReadsWf` of the parameters a case's bytes parse to**, where `Core.canonicalJson` returns well-formed trees. -/
theorem readsWf_of_bytes (hcj : ∀ b j, Core.canonicalJson b = .ok j → Wf j) {a : ByteArray}
    {p : VStar.Recursive.SampledTwo.Params}
    (h : (Core.canonicalJson a >>= VStar.Recursive.SampledTwo.paramsOfJson) = .ok p) : ReadsWf p.circuits := by
  obtain ⟨j, hj, hp⟩ := bind_eq_ok.1 h
  exact readsWf_of_params (hcj a j hj) hp

/-- **The form's boundary and interior each list no port twice**, where `Core.canonicalJson` returns well-formed
trees: what `Verity.SampledTwo.PortsDistinctAt` asks of an accepted play's two registrations. -/
theorem ports_nodup_form (hcj : ∀ b j, Core.canonicalJson b = .ok j → Wf j) {j : Json}
    {p : VStar.Recursive.SampledTwo.Params} (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) :
    (p.boundary.verifier.ports.map (·.1)).toList.Nodup ∧ (p.interior.verifier.ports.map (·.1)).toList.Nodup := by
  obtain ⟨_, _, _, _, -, hb, hi, -⟩ := paramsOfJson_reads h
  obtain ⟨b₁, j₁, hc₁, hp₁⟩ := paramsAt_ok hb
  obtain ⟨b₂, j₂, hc₂, hp₂⟩ := paramsAt_ok hi
  exact ⟨ports_nodup (hcj b₁ j₁ hc₁) hp₁, ports_nodup (hcj b₂ j₂ hc₂) hp₂⟩

/-- **The form's boundary's and interior's partition objects are well formed**, where `Core.canonicalJson` returns
well-formed trees. -/
theorem partition_wf_form (hcj : ∀ b j, Core.canonicalJson b = .ok j → Wf j) {j : Json}
    {p : VStar.Recursive.SampledTwo.Params} (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) :
    (∀ o, p.boundary.verifier.partition = some o → Wf o) ∧
      ∀ o, p.interior.verifier.partition = some o → Wf o := by
  obtain ⟨_, _, _, _, -, hb, hi, -⟩ := paramsOfJson_reads h
  obtain ⟨b₁, j₁, hc₁, hp₁⟩ := paramsAt_ok hb
  obtain ⟨b₂, j₂, hc₂, hp₂⟩ := paramsAt_ok hi
  exact ⟨fun _ ho => partition_wf (hcj b₁ j₁ hc₁) hp₁ ho, fun _ ho => partition_wf (hcj b₂ j₂ hc₂) hp₂ ho⟩

end SpotFlock.ParamsWf
