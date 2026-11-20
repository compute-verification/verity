module

public import Proofs.Properties.InferenceOnly.Record
public import Proofs.Properties.InferenceOnly.RecordBlocks
public import Proofs.Properties.InferenceOnly.RecordProgram

/-!
# The wiring's templates are the Program's

`templates_of_read`: on a record `read` accepts, the template query accepts the wiring's template ids, and the
wiring's templates are the Program's (`Templates`): a template of the wiring has the units the root has of it, and is
the template of their Definition (`TemplateOf`).

The record's text gives its templates from the root's ranges, one a template, each read off its first range
(`templatesOf_spec`, `templatesOf_tmplOf`), with distinct ids and distinct port names (`tmpls_named`). A unit of a
template is a member of one of that template's ranges, all of which call one Definition (`range_fn`), and the text's
template is read off it (`unit_tmpl`). `finish` takes the
wiring's templates from the text's (`FinishTemplates`, a hypothesis), so their ids are the ranges' templates; the template query reads
the ids it is asked for as a set (`evaluate_congr`), so it accepts the wiring's ids with the same ranges
(`evaluate_wiring`). A template read off a Definition is that Definition's template (`templateOf_of`), which gives
`Templates` (`templates_of`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Spot.RecordText
open Lean (Json)

/-- **`finish`'s wiring has the front's templates**: each template of the wiring is one of the front's, with its
units, its rows by name with their words and outer lengths, and its outputs by name, and each of the front's is one
of the wiring's. `RecordText.wiring` builds the wiring's templates from `templatesOf`'s (`Tmpl.wired`); proving it
of that construction replaces this hypothesis. -/
def FinishTemplates : Prop :=
  ∀ (f : Front) (pin : String → Option String) (c : Case), finish f pin = .ok c →
    (∀ tid t', (tid, t') ∈ c.W.templates → ∃ t ∈ f.ts, t.tid = tid ∧ t'.units = t.units ∧
        (∀ p q, (p, q) ∈ t'.rows ↔
          ∃ words outer, (p, words, outer) ∈ t.rows ∧ q.words = words ∧ q.outer = outer) ∧
        ∀ o n, (o, n) ∈ t'.outs ↔ (o, n) ∈ t.outs) ∧
      ∀ t ∈ f.ts, ∃ t', (t.tid, t') ∈ c.W.templates

end InferenceOnlyVerifier.Record.SecurityProofs

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog Scope Defn fieldsOf)
open Flock.TemplateQuery (TRange)
open Flock.Sigma (bits)
open Spot.Bits (bitsOf)
open Spot.RecordText (Tmpl tmplOf templatesOf outerOf)
open InferenceOnlyVerifier.Record.SecurityProofs (ranges_spec bitsOf_eq tmplOf_spec tmplOf_names eq_of_nodup_map read_front
  finish_text front_text FinishTemplates)
open InferenceOnlyVerifier.Record.SecurityProofs.Json (bind_ok mapM_ok)

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

/-! ## The text's templates, one a template -/

/-- Each of the text's templates is the text's reading of a range. -/
theorem templatesOf_tmplOf {P : Prog} {doc : Json} {ts : List Tmpl} (h : templatesOf P doc = .ok ts) :
    ∀ t ∈ ts, ∃ r, tmplOf P r = .ok t := by
  obtain ⟨tids, rs, -, -, -, hmapM⟩ := templatesOf_spec h
  intro t ht
  obtain ⟨x, -, hx⟩ := (mapM_ok _ _ _ hmapM t).1 ht
  obtain ⟨-, -, r, -, -, hr⟩ := tmplAt_ok hx
  exact ⟨_, hr⟩

/-- **The text's templates have distinct ids, and each distinct port names.** -/
theorem tmpls_named {P : Prog} {doc : Json} {ts : List Tmpl} (hts : templatesOf P doc = .ok ts) :
    (ts.map (·.tid)).Nodup ∧ ∀ t ∈ ts, (t.rows.map (·.1) ++ t.outs.map (·.1)).Nodup := by
  obtain ⟨tids, rs, hev, -, -, -⟩ := templatesOf_spec hts
  refine ⟨(templatesOf_ranges hts hev).1, fun t ht => ?_⟩
  obtain ⟨r, hr⟩ := templatesOf_tmplOf hts t ht
  exact tmplOf_names hr

/-- **The ranges of one template call one Definition**, of a Program the record check reads: a range's template is
its callee's key, and the decoder indexes each Definition under its own key. -/
theorem range_fn {P : Prog} {tids : Array String} {rs : Array TRange} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs) {r r' : TRange}
    (hr : r ∈ rs.toList) (hr' : r' ∈ rs.toList) (h : r.template = r'.template) :
    ((P.defs[P.root]!).body.nodes[r.node]!).fn = ((P.defs[P.root]!).body.nodes[r'.node]!).fn := by
  have hrc := InferenceOnlyVerifier.Properties.SecurityProofs.Walk.walkable_recut P hP.2.2.2.2.1
  obtain ⟨-, hall⟩ := ranges_spec (read_bodies hP).1 hP.2.2.2.2.2.2 (readProg_inputs hP) hrs
  obtain ⟨hk, ht, -⟩ := hall r hr
  obtain ⟨hk', ht', -⟩ := hall r' hr'
  obtain ⟨⟨bytes, hdec⟩, -⟩ := hP
  obtain ⟨hroot, -, -⟩ := SecurityProofs.BodiesOk.decode_root bytes P hdec
  obtain ⟨st, hPd, hinv, -⟩ := SecurityProofs.BodiesOk.decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, SecurityProofs.BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hPd]; exact hinv.1
  have hfn : ∀ k < (P.defs[P.root]!).body.nodes.size, ((P.defs[P.root]!).body.nodes[k]!).fn < P.defs.size :=
    fun k hk => Nat.lt_trans ((hok P.root hroot).2.1 k hk).1 hroot
  obtain ⟨-, huniq⟩ := SecurityProofs.DecodeKeys.decode_keys bytes P hdec
  have h1 := huniq _ (hfn _ hk)
  have h2 := huniq _ (hfn _ hk')
  rw [← ht, h, ht', h2] at h1
  exact (Option.some.inj h1).symm

/-- **A unit's template is read off its own Definition.** Where the text's reader gives the templates `ts`: a unit
`u` of the template `tid` (`UnitAt`) has a template of `ts` with that id and more units than `u`, which is the
text's reading of a range and whose Definition is the unit's. -/
theorem unit_tmpl {P : Prog} {doc : Json} {ts : List Tmpl} {tids : Array String} {rs : Array TRange}
    (hP : ReadProg P) (hts : templatesOf P doc = .ok ts)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) :
    ∃ t ∈ ts, t.tid = tid ∧ u < t.units ∧ t.defn = sc.d ∧ ∃ r, tmplOf P r = .ok t := by
  obtain ⟨t, ht, htid, hlt⟩ := (unitAt_iff hts hrs tid u).1 ⟨sc, hu⟩
  have hu' := hu
  unfold UnitAt at hu'
  obtain ⟨i, m, hi, hti, -, -, rfl⟩ := hu'
  obtain ⟨tids₀, rs₀, hev, -, -, hmapM⟩ := templatesOf_spec hts
  rw [evaluate_ranges hev hrs] at hmapM
  obtain ⟨x, -, hx⟩ := (mapM_ok _ _ _ hmapM t).1 ht
  obtain ⟨hxt, -, r, hr, hrt, htm⟩ := tmplAt_ok hx
  refine ⟨t, ht, htid, hlt, ?_, _, htm⟩
  obtain ⟨-, hall⟩ := ranges_spec (read_bodies hP).1 hP.2.2.2.2.2.2 (readProg_inputs hP) hrs
  have hrl : r ∈ rs.toList := Array.mem_toList_iff.2 hr
  have hil : rs[i]! ∈ rs.toList := by
    rw [getElem!_pos rs i hi]
    simp
  have hk := (hall r hrl).1
  rw [(tmplOf_spec htm).1, (child_root P _ _).1]
  show (((P.defs[P.root]!).body.nodes[r.node]?).map (·.fn)).getD 0 = _
  rw [Array.getElem?_eq_getElem hk, Option.map_some, Option.getD_some, ← getElem!_pos (P.defs[P.root]!).body.nodes r.node hk]
  exact range_fn hP hrs hrl hil (hrt.trans (hxt.symm.trans (htid.trans hti.symm)))

/-! ## The wiring's templates -/

/-- **The template query accepts a wiring's template ids** where they are the ids of the text's templates, with the
ranges it gives of any other list. -/
theorem evaluate_wiring {P : Prog} {doc : Json} {ts : List Tmpl} (hts : templatesOf P doc = .ok ts)
    {W : List (String × Spot.Wires.Template)} (hp1 : ∀ tid t', (tid, t') ∈ W → ∃ t ∈ ts, t.tid = tid)
    (hp2 : ∀ t ∈ ts, ∃ t', (t.tid, t') ∈ W) :
    ∃ rs, Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.map (·.1)).toArray = .ok rs := by
  obtain ⟨tids, rs, hev, -, htids, -⟩ := templatesOf_spec hts
  obtain ⟨-, hmap⟩ := templatesOf_ranges hts hev
  have hmem : ∀ t, t ∈ tids ↔ t ∈ ts.map (·.tid) := fun t => by
    have he : ts.map (·.tid) = tids.toList := by
      rw [htids, ← hmap, List.map_map]
      rfl
    rw [he, Array.mem_toList_iff]
  refine ⟨rs, evaluate_congr (fun t => ?_) hev⟩
  apply Bool.eq_iff_iff.2
  rw [Array.contains_iff_mem, Array.contains_iff_mem, hmem]
  constructor
  · intro ht
    obtain ⟨tm, htm, rfl⟩ := List.mem_map.1 ht
    obtain ⟨t', ht'⟩ := hp2 tm htm
    exact List.mem_toArray.2 (List.mem_map.2 ⟨(tm.tid, t'), ht', rfl⟩)
  · intro ht
    obtain ⟨⟨tid, t'⟩, hm, he0⟩ := List.mem_map.1 (List.mem_toArray.1 ht)
    have he0' : tid = t := he0
    subst he0'
    obtain ⟨tm, htm, he⟩ := hp1 tid t' hm
    exact List.mem_map.2 ⟨tm, htm, he⟩

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
  intro tid t' hmem
  obtain ⟨t, ht, htid, hunits, hrows, houts⟩ := hp1 tid t' hmem
  refine ⟨fun u => ⟨fun hu => ?_, ?_⟩, fun u sc hu => ?_⟩
  · exact (unitAt_iff hts hrs tid u).2 ⟨t, ht, htid, by rw [← hunits]; exact hu⟩
  · intro hu
    obtain ⟨t₂, ht₂, htid₂, hlt⟩ := (unitAt_iff hts hrs tid u).1 hu
    have he : t₂ = t := eq_of_nodup_map (·.tid) ts hd t₂ ht₂ t ht (htid₂.trans htid.symm)
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
theorem templates_of_read (hFT : FinishTemplates) {j tr : Json} {c : InferenceOnlyVerifier.Record.Case}
    (h : InferenceOnlyVerifier.Record.read j tr = .ok c) :
    ∃ rs, Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg c.f.P)
        (c.W.templates.map (·.1)).toArray = .ok rs ∧ Templates c.f.P rs c.W := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨hp1, hp2⟩ := hFT f pin c hc
  obtain ⟨rfl, -⟩ := finish_text hc
  have hts := front_text hf
  obtain ⟨rs, hrs⟩ := evaluate_wiring hts (W := c.W.templates)
    (fun tid t' hm => by
      obtain ⟨t, ht, he, -⟩ := hp1 tid t' hm
      exact ⟨t, ht, he⟩)
    hp2
  exact ⟨rs, hrs, templates_of (readProg_of_read h) hts hrs hp1⟩

end InferenceOnlyVerifier.Properties.Record.SecurityProofs
