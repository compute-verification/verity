module

public import Proofs.Properties.Nci.Record
public import Proofs.Properties.Nci.RecordBlocks
public import Proofs.Properties.Nci.RecordProgram

/-!
# The wiring's templates are the Program's

`templates_of_read`: on a record `read` accepts, the template query accepts the wiring's template ids, and the
wiring's templates are the Program's (`Templates`): a template of the wiring has the units the root has of it, and is
the template of their Definition (`TemplateOf`).

The record's text gives its templates from the root's ranges, one a range (`templatesOf_spec`, `templatesOf_tmplOf`),
with distinct ids and distinct port names (`tmpls_named`). A unit of a template is a member of that template's one
range, and the text's template of that range is read off the unit's own Definition (`unit_tmpl`). `finish` takes the
wiring's templates from the text's (`finish_wiring`), so their ids are the ranges' templates; the template query reads
the ids it is asked for as a set (`evaluate_congr`), so it accepts the wiring's ids with the same ranges
(`evaluate_wiring`). A template read off a Definition is that Definition's template (`templateOf_of`), which gives
`Templates` (`templates_of`).
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open RecordText
open Lean (Json)

/-- `finish` takes its wiring's templates from the front's own piece of the text. -/
theorem finish_wiring {f : Front} {pin : String → Option String} {c : Case} (h : finish f pin = .ok c) :
    ∃ W₀, Spot.Wires.Wiring.ofJson (piece f) = .ok W₀ ∧ c.W.templates = W₀.templates := by
  unfold finish at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i W hW
      have hW₀ : ∃ W₀, Spot.Wires.Wiring.ofJson (piece f) = .ok W₀ ∧ W.templates = W₀.templates := by
        unfold wiringOf at hW
        split at hW
        · rename_i W₀ hW₀
          cases hW
          exact ⟨W₀, hW₀, rfl⟩
        · cases hW
      split at h
      · split at h
        · cases h
        · split at h
          · cases h
          · split at h
            · cases h
            · split at h
              · cases h
              · cases h
                exact hW₀
      · cases h

end NciCheck.Record.SecurityProofs

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog Scope Defn)
open Flock.TemplateQuery (TRange)
open Flock.Sigma (bits)
open Spot.Bits (bitsOf)
open NciCheck.RecordText (Tmpl tmplOf templatesOf fieldsOf outerOf)
open NciCheck.Record.SecurityProofs (bitsOf_eq tmplOf_spec tmplOf_names piece_templates eq_of_nodup_map read_front
  finish_text front_text finish_wiring)
open NciCheck.Record.SecurityProofs.Json (bind_ok mapM_ok)

/-! ## The template query reads its templates as a set -/

/-- A loop whose body gives the steps another's gives, wherever that one accepts, ends where that one ends. -/
theorem forIn_transfer {σ : Type} {f g : ℕ → σ → Except String (ForInStep σ)}
    (hfg : ∀ k s x, f k s = .ok x → g k s = .ok x) :
    ∀ (l : List ℕ) (s x : σ), forIn l s f = .ok x → forIn l s g = .ok x
  | [], s, x, hx => by
    rw [List.forIn_nil] at hx ⊢
    exact hx
  | k :: l, s, x, hx => by
    rw [List.forIn_cons] at hx ⊢
    obtain ⟨a, ha, hx⟩ := bind_ok hx
    rw [hfg k s a ha]
    cases a with
    | done s' => exact hx
    | yield s' => exact forIn_transfer hfg l s' x hx

/-- **The template query reads the templates asked for as a set**: where it accepts one list, it accepts any list
with the same members, with the same ranges. -/
theorem evaluate_congr {d : Flock.TemplateQuery.Desc} {a b : Array String} {rs : Array TRange}
    (hab : ∀ t, a.contains t = b.contains t) (h : Flock.TemplateQuery.evaluate d a = .ok rs) :
    Flock.TemplateQuery.evaluate d b = .ok rs := by
  unfold Flock.TemplateQuery.evaluate at h ⊢
  cases hroot : d.def? d.root with
  | none =>
    simp only [hroot] at h
    cases h
  | some root =>
    simp only [hroot] at h ⊢
    obtain ⟨r, hloop, h⟩ := bind_ok h
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop ⊢
    rw [forIn_transfer ?_ _ _ _ hloop]
    · simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h ⊢
      split at h
      · cases h
      · rename_i hm
        have hb : ¬ (!(Array.filter (fun t => !r.fst.any fun r => r.template == t && decide (r.count > 0))
            b).isEmpty) = true := by
          intro hne
          apply hm
          rw [Bool.not_eq_true', Array.isEmpty_eq_false_iff] at hne ⊢
          obtain ⟨t, ht⟩ := Array.exists_mem_of_ne_empty _ hne
          obtain ⟨htb, hp⟩ := Array.mem_filter.1 ht
          have hta : t ∈ a := Array.contains_iff_mem.1 ((hab t).trans (Array.contains_iff_mem.2 htb))
          exact Array.ne_empty_of_mem (Array.mem_filter.2 ⟨hta, hp⟩)
        rw [ite_eq_right hb]
        exact h
    · intro k s x hx
      by_cases hi : d.isInputNode root.nodes[k]! = true
      · rw [ite_eq_left hi] at hx ⊢
        exact hx
      · rw [ite_eq_right hi] at hx ⊢
        by_cases hc : (!(root.nodes[k]!.form == "call" || root.nodes[k]!.form == "batch") ||
            !a.contains (d.callees.getD root.nodes[k]!.fn "")) = true
        · rw [ite_eq_left hc] at hx
          simp only [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at hx
          cases hx
        · rw [ite_eq_right hc] at hx
          rw [ite_eq_right (by rw [← hab]; exact hc)]
          exact hx

/-! ## The text's templates, one a range -/

/-- Where a map that may refuse gives a list, each of its members is the map of the member at its place. -/
theorem mapM_getElem? {α β : Type} (f : α → Except String β) :
    ∀ (l : List α) (l' : List β), l.mapM f = .ok l' → ∀ (i : ℕ) (y : β), l'[i]? = some y →
      ∃ x, l[i]? = some x ∧ f x = .ok y
  | [], l', h, i, y, hy => by
    rw [List.mapM_nil] at h
    cases h
    simp at hy
  | x :: l, l', h, i, y, hy => by
    rw [List.mapM_cons] at h
    obtain ⟨y0, hy0, h⟩ := bind_ok h
    obtain ⟨ys, hys, h⟩ := bind_ok h
    cases h
    cases i with
    | zero =>
      rw [List.getElem?_cons_zero] at hy
      cases hy
      exact ⟨x, List.getElem?_cons_zero, hy0⟩
    | succ i =>
      rw [List.getElem?_cons_succ] at hy
      obtain ⟨x', hx', hf⟩ := mapM_getElem? f l ys hys i y hy
      exact ⟨x', by rw [List.getElem?_cons_succ]; exact hx', hf⟩

/-- **The text's templates are read off the root's ranges**: where the text's reader gives the templates `ts`, the
template query accepts the text's template ids, which are its ranges' templates in order, and `ts` are the ranges'
templates, one each. The same content as `Record`'s private `templatesOf_ok`, read the same way; one of the two goes
when that one is public. -/
theorem templatesOf_spec {P : Prog} {doc : Json} {ts : List Tmpl} (h : templatesOf P doc = .ok ts) :
    ∃ (tids : Array String) (rs : Array TRange),
      Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs ∧
        tids = rs.map (·.template) ∧ rs.toList.mapM (tmplOf P) = .ok ts := by
  unfold templatesOf at h
  simp (config := { iota := false }) only [bind, Except.bind, pure, Except.pure, throw, throwThe,
    MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rename_i tsp _ hdup _ rs₀ hev _ rs hrs hne
  have hrs' : rs₀ = rs := Except.ok.inj hrs
  have hmap : rs.map (fun r => (r.template, r.count)) = tsp := Classical.not_not.1 fun hc => hne (bne_iff_ne.2 hc)
  refine ⟨tsp.map (·.1), rs, by rw [← hrs']; exact hev, ?_, h⟩
  rw [← hmap, Array.map_map]
  rfl

/-- Each of the text's templates is the text's reading of a range. -/
theorem templatesOf_tmplOf {P : Prog} {doc : Json} {ts : List Tmpl} (h : templatesOf P doc = .ok ts) :
    ∀ t ∈ ts, ∃ r, tmplOf P r = .ok t := by
  obtain ⟨tids, rs, -, -, hmapM⟩ := templatesOf_spec h
  intro t ht
  obtain ⟨r, -, hr⟩ := (mapM_ok _ _ _ hmapM t).1 ht
  exact ⟨r, hr⟩

private theorem tids_eq {ts : List Tmpl} {rs : Array TRange}
    (hmap : ts.map (fun t => (t.tid, t.units)) = rs.toList.map (fun r => (r.template, r.count))) :
    ts.map (·.tid) = rs.toList.map (·.template) := by
  have h := congrArg (List.map (·.1)) hmap
  rw [List.map_map, List.map_map] at h
  exact h

/-- **The text's templates have distinct ids, and each distinct port names.** -/
theorem tmpls_named {P : Prog} {doc : Json} {ts : List Tmpl} (hts : templatesOf P doc = .ok ts) :
    (ts.map (·.tid)).Nodup ∧ ∀ t ∈ ts, (t.rows.map (·.1) ++ t.outs.map (·.1)).Nodup := by
  obtain ⟨tids, rs, hev, -, -⟩ := templatesOf_spec hts
  obtain ⟨hnd, hmap⟩ := templatesOf_ranges hts hev
  refine ⟨by rw [tids_eq hmap]; exact hnd, fun t ht => ?_⟩
  obtain ⟨r, hr⟩ := templatesOf_tmplOf hts t ht
  exact tmplOf_names hr

private theorem range_at {ts : List Tmpl} {rs : Array TRange}
    (hmap : ts.map (fun t => (t.tid, t.units)) = rs.toList.map (fun r => (r.template, r.count))) {i : ℕ} {t : Tmpl}
    (ht : ts[i]? = some t) : i < rs.size ∧ rs[i]!.template = t.tid ∧ rs[i]!.count = t.units := by
  have hlen : ts.length = rs.size := by simpa using congrArg List.length hmap
  obtain ⟨hlt, -⟩ := List.getElem?_eq_some_iff.1 ht
  have hi : i < rs.size := by rw [← hlen]; exact hlt
  have hi' : i < rs.toList.length := by rw [Array.length_toList]; exact hi
  have h := congrArg (fun l => l[i]?) hmap
  simp only [List.getElem?_map, ht, Option.map_some] at h
  rw [List.getElem?_eq_getElem hi'] at h
  simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq, Array.getElem_toList] at h
  rw [getElem!_pos rs i hi]
  exact ⟨hi, h.1.symm, h.2.symm⟩

private theorem get_of_toList {rs : Array TRange} {i : ℕ} {r : TRange} (hi : i < rs.size)
    (hr : rs.toList[i]? = some r) : rs[i]! = r := by
  have hi' : i < rs.toList.length := by rw [Array.length_toList]; exact hi
  rw [getElem!_pos rs i hi]
  rw [List.getElem?_eq_getElem hi', Option.some.injEq, Array.getElem_toList] at hr
  exact hr

/-- **A unit's template is read off its own Definition.** Where the text's reader gives the templates `ts`: a unit
`u` of the template `tid` (`UnitAt`) has a template of `ts` with that id and more units than `u`, which is the
text's reading of a range and whose Definition is the unit's. -/
theorem unit_tmpl {P : Prog} {doc : Json} {ts : List Tmpl} {tids : Array String} {rs : Array TRange}
    (hP : ReadProg P) (hts : templatesOf P doc = .ok ts)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) :
    ∃ t ∈ ts, t.tid = tid ∧ u < t.units ∧ t.defn = sc.d ∧ ∃ r, tmplOf P r = .ok t := by
  obtain ⟨i, t, hi, -, hti, htid, -, hlt, rfl⟩ := (unitAt_iff hts hrs tid u sc).1 hu
  obtain ⟨tids₀, rs₀, hev, -, hmapM⟩ := templatesOf_spec hts
  rw [evaluate_ranges hev hrs] at hmapM
  obtain ⟨r, hr, htm⟩ := mapM_getElem? (tmplOf P) _ _ hmapM i t hti
  have hri := get_of_toList hi hr
  subst hri
  have hk := (range_node (read_bodies hP).1 hP.2.2.2.2.2.2 (readProg_inputs hP) hrs hi).1
  refine ⟨t, List.mem_of_getElem? hti, htid, hlt, ?_, _, htm⟩
  rw [(tmplOf_spec htm).1, (child_root P _ _).1, Array.getElem?_eq_getElem hk,
    getElem!_pos (P.defs[P.root]!).body.nodes _ hk]
  rfl

/-! ## The wiring's templates -/

/-- **The template query accepts a wiring's template ids** where they are the ids of the text's templates, with the
ranges it gives of any other list. -/
theorem evaluate_wiring {P : Prog} {doc : Json} {ts : List Tmpl} (hts : templatesOf P doc = .ok ts)
    {W : List (String × Spot.Wires.Template)} (hp1 : ∀ tid t', (tid, t') ∈ W → ∃ t ∈ ts, t.tid = tid)
    (hp2 : ∀ t ∈ ts, ∃ t', (t.tid, t') ∈ W) :
    ∃ rs, Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.map (·.1)).toArray = .ok rs := by
  obtain ⟨tids, rs, hev, htids, -⟩ := templatesOf_spec hts
  obtain ⟨-, hmap⟩ := templatesOf_ranges hts hev
  refine ⟨rs, evaluate_congr (fun t => ?_) hev⟩
  apply Bool.eq_iff_iff.2
  rw [Array.contains_iff_mem, Array.contains_iff_mem, htids]
  constructor
  · intro ht
    have h1 : t ∈ ts.map (·.tid) := by
      rw [tids_eq hmap]
      obtain ⟨r, hr, he⟩ := Array.mem_map.1 ht
      exact List.mem_map.2 ⟨r, Array.mem_toList_iff.2 hr, he⟩
    obtain ⟨tm, htm, rfl⟩ := List.mem_map.1 h1
    obtain ⟨t', ht'⟩ := hp2 tm htm
    exact List.mem_toArray.2 (List.mem_map.2 ⟨(tm.tid, t'), ht', rfl⟩)
  · intro ht
    obtain ⟨⟨tid, t'⟩, hm, he0⟩ := List.mem_map.1 (List.mem_toArray.1 ht)
    have he0' : tid = t := he0
    subst he0'
    obtain ⟨tm, htm, he⟩ := hp1 tid t' hm
    have h1 : tid ∈ ts.map (·.tid) := List.mem_map.2 ⟨tm, htm, he⟩
    rw [tids_eq hmap] at h1
    obtain ⟨r, hr, he'⟩ := List.mem_map.1 h1
    exact Array.mem_map.2 ⟨r, Array.mem_toList_iff.1 hr, he'⟩

private theorem names_inj {α : Type} [Inhabited α] (ps : Array (String × α)) (hnd : (ps.toList.map (·.1)).Nodup)
    (i i' : ℕ) (hi : i < ps.size) (hi' : i' < ps.size) (h : ps[i]!.1 = ps[i']!.1) : i = i' := by
  have h1 : i < (ps.toList.map (·.1)).length := by simpa using hi
  have h2 : i' < (ps.toList.map (·.1)).length := by simpa using hi'
  have hEq : (ps.toList.map (·.1))[i] = (ps.toList.map (·.1))[i'] := by
    rw [getElem!_pos ps i hi, getElem!_pos ps i' hi'] at h
    simpa using h
  rcases Nat.lt_trichotomy i i' with hlt | heq | hgt
  · exact absurd hEq (List.pairwise_iff_getElem.1 hnd i i' h1 h2 hlt)
  · exact heq
  · exact absurd hEq.symm (List.pairwise_iff_getElem.1 hnd i' i h2 h1 hgt)

/-- **A template read off a Definition is that Definition's template**: a wiring's template whose rows and outputs
are those of a text's template, which has a row a parameter of the Definition `d`, of a whole number of 16-bit
words, and an output a field of `d`'s return, under distinct names, is `d`'s template (`TemplateOf`). -/
theorem templateOf_of {d : Defn} {t : Tmpl} {t' : Spot.Wires.Template}
    (hrows : t.rows = d.params.toList.map (fun p => (p.1, bits p.2 / 16, outerOf p.2)))
    (hbits : ∀ p ∈ d.params.toList, bits p.2 ≠ 0 ∧ bits p.2 % 16 = 0)
    (houts : t.outs = (fieldsOf d.ret).zipIdx.map (fun x => ("o" ++ toString x.2, (bits x.1 + 15) / 16)))
    (hnames : (t.rows.map (·.1) ++ t.outs.map (·.1)).Nodup)
    (hr : ∀ p q, (p, q) ∈ t'.rows ↔
      ∃ words outer, (p, words, outer) ∈ t.rows ∧ q.words = words ∧ q.outer = outer)
    (ho : ∀ o n, (o, n) ∈ t'.outs ↔ (o, n) ∈ t.outs) : TemplateOf d t' := by
  have hnd : (d.params.toList.map (·.1)).Nodup := by
    have h := (List.nodup_append.1 hnames).1
    rw [hrows, List.map_map] at h
    exact h
  have hmem : ∀ i (hi : i < d.params.size), d.params[i]! ∈ d.params.toList := fun i hi => by
    rw [getElem!_pos d.params i hi]
    exact Array.mem_toList_iff.2 (Array.getElem_mem hi)
  refine ⟨fun i i' hi hi' h => names_inj d.params hnd i i' hi hi' h, fun i hi => ?_, fun p q => ?_, fun o n => ?_⟩
  · rw [bitsOf_eq]
    exact Nat.dvd_of_mod_eq_zero (hbits _ (hmem i hi)).2
  · rw [hr, hrows]
    constructor
    · rintro ⟨words, outer, hm, hw, ho'⟩
      obtain ⟨x, hx, he⟩ := List.mem_map.1 hm
      obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hx)
      simp only [Prod.mk.injEq] at he
      refine ⟨i, hi, ?_, ?_, ?_⟩
      · rw [getElem!_pos d.params i hi]; exact he.1.symm
      · rw [getElem!_pos d.params i hi, bitsOf_eq, hw]; exact he.2.1.symm
      · rw [getElem!_pos d.params i hi, ho']; exact he.2.2.symm
    · rintro ⟨i, hi, rfl, hw, ho'⟩
      exact ⟨_, _, List.mem_map.2 ⟨d.params[i]!, hmem i hi, rfl⟩, by rw [hw, bitsOf_eq], ho'⟩
  · rw [ho, houts]
    constructor
    · intro hm
      obtain ⟨⟨ty, f⟩, hx, he⟩ := List.mem_map.1 hm
      obtain ⟨hlt, hget⟩ := List.getElem?_eq_some_iff.1 (List.mem_zipIdx_iff_getElem?.1 hx)
      simp only [Prod.mk.injEq] at he
      refine ⟨f, hlt, he.1.symm, ?_⟩
      rw [getElem!_pos _ f hlt, hget, bitsOf_eq]
      exact he.2.symm
    · rintro ⟨f, hf, rfl, rfl⟩
      refine List.mem_map.2 ⟨((fieldsOf d.ret)[f]!, f), List.mem_zipIdx_iff_getElem?.2 ?_, ?_⟩
      · rw [getElem!_pos _ f hf]
        exact List.getElem?_eq_getElem hf
      · rw [bitsOf_eq]

/-- **A wiring whose templates are the text's has the Program's templates** (`Templates`): where the text's reader
gives the templates `ts`, the template query gives the ranges `rs`, and each template of the wiring `W` has the id,
the units, the rows and the outputs of one of `ts`. -/
theorem templates_of {P : Prog} {doc : Json} {ts : List Tmpl} (hP : ReadProg P) (hts : templatesOf P doc = .ok ts)
    {tids : Array String} {rs : Array TRange}
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs) {W : Spot.Wires.Wiring}
    (hp1 : ∀ tid t', (tid, t') ∈ W.templates → ∃ t ∈ ts, t.tid = tid ∧ t'.units = t.units ∧
        (∀ p q, (p, q) ∈ t'.rows ↔
          ∃ words outer, (p, words, outer) ∈ t.rows ∧ q.words = words ∧ q.outer = outer) ∧
        ∀ o n, (o, n) ∈ t'.outs ↔ (o, n) ∈ t.outs) : Templates P rs W := by
  obtain ⟨hd, -⟩ := tmpls_named hts
  obtain ⟨-, hmap⟩ := templatesOf_ranges hts hrs
  intro tid t' hmem
  obtain ⟨t, ht, htid, hunits, hrows, houts⟩ := hp1 tid t' hmem
  refine ⟨fun u => ⟨fun hu => ?_, ?_⟩, fun u sc hu => ?_⟩
  · obtain ⟨i, hi⟩ := List.getElem?_of_mem ht
    obtain ⟨hilt, htmpl, hcount⟩ := range_at hmap hi
    exact ⟨P.child P.rootScope rs[i]!.node u, (unitAt_iff hts hrs tid u _).2
      ⟨i, t, hilt, htmpl.trans htid, hi, htid, hcount.symm, by rw [← hunits]; exact hu, rfl⟩⟩
  · rintro ⟨sc, hu⟩
    obtain ⟨i, t₂, -, -, hi₂, htid₂, -, hlt, -⟩ := (unitAt_iff hts hrs tid u sc).1 hu
    have he : t₂ = t :=
      eq_of_nodup_map (·.tid) ts hd t₂ (List.mem_of_getElem? hi₂) t ht (htid₂.trans htid.symm)
    rw [hunits, ← he]
    exact hlt
  · obtain ⟨t₂, ht₂, htid₂, -, hdefn, r, hr⟩ := unit_tmpl hP hts hrs hu
    have he : t₂ = t := eq_of_nodup_map (·.tid) ts hd t₂ ht₂ t ht (htid₂.trans htid.symm)
    subst he
    obtain ⟨-, -, -, hr1, hr2, ho1, -⟩ := tmplOf_spec hr
    rw [hdefn] at hr1 hr2 ho1
    exact templateOf_of hr1 hr2 ho1 (tmplOf_names hr) hrows houts

/-- **On a record `read` accepts, the wiring's templates are the Program's**: the template query accepts the
wiring's template ids, and with the ranges it gives, a template of the wiring has the units the root has of it and is
the template of their Definition. -/
theorem templates_of_read {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c) :
    ∃ rs, Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg c.f.P)
        (c.W.templates.map (·.1)).toArray = .ok rs ∧ Templates c.f.P rs c.W := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, -⟩ := finish_text hc
  have hts := front_text hf
  obtain ⟨W₀, hW₀, hWt⟩ := finish_wiring hc
  obtain ⟨hd, hn⟩ := tmpls_named hts
  obtain ⟨hp1, hp2⟩ := piece_templates hd hn hW₀
  obtain ⟨rs, hrs⟩ := evaluate_wiring hts (W := c.W.templates)
    (fun tid t' hm => by
      rw [hWt] at hm
      obtain ⟨t, ht, he, -⟩ := hp1 tid t' hm
      exact ⟨t, ht, he⟩)
    (fun t ht => by rw [hWt]; exact hp2 t ht)
  exact ⟨rs, hrs, templates_of (readProg_of_read h) hts hrs (by rw [hWt]; exact hp1)⟩

end NciCheck.Properties.Record.SecurityProofs
