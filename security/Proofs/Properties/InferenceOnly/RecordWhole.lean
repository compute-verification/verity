module

public import Proofs.Properties.InferenceOnly.RecordBlocks

/-!
# What the record check tests of a wire's two rows

`Record.unwhole` refuses a record with a wire whose offset is no first word of a field of its writer's return of a
word or more, whose field is not the bits of the row it fills, or whose two rows' leaves do not line up. This module
states what a record it accepts has, of each wire, in the terms of the two templates' Definitions (`whole_spec`): the
wire's field is the row's words, whole; and either the reader's parameter has leaves of one bit and is none of what
its Definition passes on, or each side's leaves are of one width and the writer's divides the reader's.

The steps: what `unwhole` tests of one wire (`unwhole_none`), a template's leaves among the leaves of templates with
distinct ids (`leaves_find`), and a name's entry in a list of named values with distinct names (`lookup_map`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Spot.RecordText (Tmpl)
open Flock.Program
open Flock.Sigma (bits)

/-- The member with a key, in a list whose keys are distinct. -/
theorem find_of_nodup {α : Type} (key : α → String) : ∀ (l : List α), (l.map key).Nodup → ∀ a ∈ l,
    l.find? (fun x => key x == key a) = some a
  | [], _, _, ha => by cases ha
  | x :: l, h, a, ha => by
    rw [List.find?_cons]
    by_cases hx : key x = key a
    · rw [show (key x == key a) = true from beq_iff_eq.2 hx]
      exact congrArg some (eq_of_nodup_map key (x :: l) h x List.mem_cons_self a ha hx)
    · rw [show (key x == key a) = false from beq_eq_false_iff_ne.2 hx]
      rcases List.mem_cons.1 ha with ha' | ha'
      · exact absurd (by rw [ha']) hx
      · exact find_of_nodup key l (List.nodup_cons.1 (by simpa using h)).2 a ha'

/-- A name's value, in a list of named values whose names are distinct. -/
theorem lookup_map {α β : Type} (k : α → String) (v : α → β) : ∀ (l : List α), (l.map k).Nodup → ∀ a ∈ l,
    (l.map fun x => (k x, v x)).lookup (k a) = some (v a)
  | [], _, _, ha => by cases ha
  | x :: l, h, a, ha => by
    rw [List.map_cons, List.lookup_cons]
    by_cases hx : k a = k x
    · rw [show (k a == k x) = true from beq_iff_eq.2 hx]
      exact congrArg (fun y => some (v y)) (eq_of_nodup_map k (x :: l) h x List.mem_cons_self a ha hx.symm)
    · rw [show (k a == k x) = false from beq_eq_false_iff_ne.2 hx]
      rcases List.mem_cons.1 ha with ha' | ha'
      · exact absurd (by rw [ha']) hx
      · exact lookup_map k v l (List.nodup_cons.1 (by simpa using h)).2 a ha'

/-- A name not among a list's names is looked up past it. -/
theorem lookup_append_of_not_mem {β : Type} : ∀ (l l' : List (String × β)) (a : String), a ∉ l.map (·.1) →
    (l ++ l').lookup a = l'.lookup a
  | [], _, _, _ => rfl
  | x :: l, l', a, h => by
    rw [List.map_cons, List.mem_cons, not_or] at h
    rw [List.cons_append, List.lookup_cons, show (a == x.1) = false from beq_eq_false_iff_ne.2 h.1]
    exact lookup_append_of_not_mem l l' a h.2

/-- Two leaf widths, each found, the writer's nonzero and dividing the reader's. -/
theorem widths_divide {a b : Option Nat}
    (h : (match a, b with
      | some r, some o => o != 0 && r % o == 0
      | _, _ => false) = true) :
    ∃ r o, a = some r ∧ b = some o ∧ o ≠ 0 ∧ r % o = 0 := by
  match a, b, h with
  | some r, some o, h =>
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at h
    exact ⟨r, o, rfl, rfl, h.1, h.2⟩
  | none, _, h => simp at h
  | some _, none, h => simp at h

/-- A member of a list, or of the list it is inserted into, is in the insertions. -/
theorem mem_foldl_insert {x : Int} : ∀ (l t : List Int), (x ∈ t ∨ x ∈ l) → x ∈ l.foldl (fun t y => t.insert y) t
  | [], _, h => by simpa using h
  | y :: l, t, h => by
    rw [List.foldl_cons]
    refine mem_foldl_insert l (t.insert y) ?_
    rcases h with h | h
    · exact Or.inl (List.mem_insert_iff.2 (Or.inr h))
    · rcases List.mem_cons.1 h with rfl | h
      · exact Or.inl (List.mem_insert_iff.2 (Or.inl rfl))
      · exact Or.inr h

/-- **Every offset step a wire's index takes is listed** by `offsetSteps`, where it lists them. -/
theorem accum_mem (m : Nat) : ∀ (ps : List (Nat × Int)) (x : Nat) {os : List Int},
    offsetSteps m ps = some os → Circuits.Wire.accum ps x ∈ os
  | [], _, os, h => by
    simp only [offsetSteps, Option.some.injEq] at h
    subst h
    simp [Circuits.Wire.accum]
  | (r, st) :: ps, x, os, h => by
    simp only [offsetSteps] at h
    obtain ⟨rest, hr, h⟩ := Option.bind_eq_some_iff.1 h
    have ih := accum_mem m ps (x / r) hr
    simp only [Circuits.Wire.accum]
    split at h
    · rename_i hs
      obtain rfl := Option.some.inj h
      rw [beq_iff_eq.1 hs, Int.mul_zero, Int.zero_add]
      exact ih
    · split at h
      · cases h
      · rename_i hr0
        simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or] at hr0
        split at h
        · cases h
        · obtain rfl := Option.some.inj h
          refine mem_foldl_insert _ _ (Or.inr (List.mem_flatMap.2 ⟨x % r, ?_, List.mem_map.2 ⟨_, ih, rfl⟩⟩))
          exact List.mem_range.2 (Nat.mod_lt _ (Nat.pos_of_ne_zero hr0.1))

/-- **What `unwhole` tests of one wire**, with the words `Spot.Wires.check` gives its row: its offset steps are listed
(`offsetSteps`), and at each, the writer is a template, and the offset is the first word of a field of its
Definition's return of a word or more; that field's bits, where its leaves are found, are the row's words'; and the
reader's row is closed, or each side has a leaf width and the writer's divides the reader's. -/
theorem unwhole_none {P : Prog} {ts : List Tmpl} {ls : List Leaves} {s : Spot.Wires.Summary}
    (h : unwhole P ts ls s = none) {w : Spot.Wires.Wire} {n : Nat} (hw : (w, n) ∈ s.wires.zip s.rowWords) :
    ∃ os, offsetSteps (((ts.find? (·.tid == w.writer)).map fun t => (fieldsOf (P.defs[t.defn]!).ret).length).getD 0)
        (Circuits.Wire.opairs w) = some os ∧ ∀ d ∈ os,
    ∃ t j, ts.find? (·.tid == w.writer) = some t ∧ j < (fieldsOf (P.defs[t.defn]!).ret).length ∧
      Spot.Edges.fieldWord (P.defs[t.defn]!).ret j = ((w.offset : Int) + d).toNat ∧
      0 < Spot.Edges.wordsOf (fieldsOf (P.defs[t.defn]!).ret)[j]! ∧
      (∀ b, ((ls.find? (·.tid == w.writer)).bind fun l => l.bits.lookup ("o" ++ toString j)) = some b →
        b = 16 * n) ∧
      (((ls.find? (·.tid == w.reader)).map fun l => l.closed.contains w.port).getD false = true ∨
        ∃ r o, ((ls.find? (·.tid == w.reader)).bind fun l => l.leaf.lookup w.port).join = some r ∧
          ((ls.find? (·.tid == w.writer)).bind fun l => l.leaf.lookup ("o" ++ toString j)).join = some o ∧
          o ≠ 0 ∧ r % o = 0) := by
  unfold unwhole at h
  have h0' := List.findSome?_eq_none_iff.1 h (w, n) hw
  dsimp only at h0'
  split at h0'
  · cases h0'
  rename_i os hos
  refine ⟨os, hos, fun d hd => ?_⟩
  have h0 := List.findSome?_eq_none_iff.1 h0' d hd
  dsimp only at h0
  split at h0
  · cases h0
  · rename_i j hf
    obtain ⟨t, ht, hj⟩ := Option.bind_eq_some_iff.1 hf
    obtain ⟨hjp, hjr, -⟩ := List.find?_range_eq_some.1 hj
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hjp
    refine ⟨t, j, ht, List.mem_range.1 hjr, hjp.1, hjp.2, ?_⟩
    split at h0
    · cases h0
    · rename_i hsh
      refine ⟨fun b hb => ?_, ?_⟩
      · rw [hb] at hsh
        simp only at hsh
        split at hsh
        · rename_i he
          exact beq_iff_eq.1 he
        · cases hsh
      · simp only [ite_eq_left_iff, reduceCtorEq, imp_false, Classical.not_not, Bool.or_eq_true] at h0
        rcases h0 with hc | hc
        · exact Or.inl hc
        · exact Or.inr (widths_divide hc)

/-- A name among a list's names is looked up in it. -/
theorem lookup_append_of_some {β : Type} : ∀ (l l' : List (String × β)) (a : String) (b : β), l.lookup a = some b →
    (l ++ l').lookup a = some b
  | [], _, _, _, h => by cases h
  | x :: l, l', a, b, h => by
    rw [List.lookup_cons] at h
    rw [List.cons_append, List.lookup_cons]
    split
    · rename_i he
      rw [he] at h
      exact h
    · rename_i he
      rw [he] at h
      exact lookup_append_of_some l l' a b h

/-- **The template `t` is its Definition's**: its rows are the Definition's parameters, each with its words and its
outer length, its outputs the fields of the Definition's return type, named `o<j>`, each with its bits rounded up to
words, and no two of its ports have one name. -/
structure Shaped (P : Prog) (t : Tmpl) : Prop where
  rows : t.rows = (P.defs[t.defn]!).params.toList.map fun p => (p.1, bits p.2 / 16, Spot.RecordText.outerOf p.2)
  outs : t.outs =
    (fieldsOf (P.defs[t.defn]!).ret).zipIdx.map fun x => ("o" ++ toString x.2, (bits x.1 + 15) / 16)
  names : (t.rows.map (·.1) ++ t.outs.map (·.1)).Nodup

/-- A template the text's reader gives is its Definition's. -/
theorem shaped_of_tmplOf {P : Prog} {r : Flock.TemplateQuery.TRange} {t : Tmpl} (h : Spot.RecordText.tmplOf P r = .ok t) :
    Shaped P t :=
  ⟨(tmplOf_spec h).2.2.2.1, (tmplOf_spec h).2.2.2.2.2.1, tmplOf_names h⟩

/-- A template's leaves, among the leaves of templates with distinct ids. -/
theorem leaves_find {P : Prog} {passed : Array (List Nat)} {ts : List Tmpl} (hd : (ts.map (·.tid)).Nodup) {t : Tmpl}
    (ht : t ∈ ts) : (ts.map (leavesOf P passed)).find? (·.tid == t.tid) = some (leavesOf P passed t) := by
  rw [List.find?_map]
  show Option.map (leavesOf P passed) (ts.find? fun x => x.tid == t.tid) = _
  rw [find_of_nodup (·.tid) ts hd t ht]
  rfl

namespace Shaped

variable {P : Prog} {t : Tmpl} (h : Shaped P t)
include h

/-- No two parameters of a template's Definition have one name. -/
theorem params_nodup : ((P.defs[t.defn]!).params.toList.map (·.1)).Nodup := by
  have := (List.nodup_append.1 h.names).1
  rw [h.rows, List.map_map] at this
  exact this

/-- No two outputs of a template have one name. -/
theorem outs_nodup :
    ((fieldsOf (P.defs[t.defn]!).ret).zipIdx.map fun x => "o" ++ toString x.2).Nodup := by
  have := (List.nodup_append.1 h.names).2.1
  rw [h.outs, List.map_map] at this
  exact this

/-- No parameter of a template's Definition has an output's name. -/
theorem out_not_param {j : Nat} (hj : j < (fieldsOf (P.defs[t.defn]!).ret).length) :
    "o" ++ toString j ∉ (P.defs[t.defn]!).params.toList.map (·.1) := by
  intro hm
  have hd := (List.nodup_append.1 h.names).2.2
  refine hd ("o" ++ toString j) ?_ ("o" ++ toString j) ?_ rfl
  · rw [h.rows, List.map_map]
    exact hm
  · rw [h.outs, List.map_map]
    exact List.mem_map.2 ⟨((fieldsOf (P.defs[t.defn]!).ret)[j], j),
      List.mem_zipIdx_iff_getElem?.2 (List.getElem?_eq_getElem hj), rfl⟩

/-- An output's words, by name. -/
theorem outs_lookup {j : Nat} (hj : j < (fieldsOf (P.defs[t.defn]!).ret).length) :
    t.outs.lookup ("o" ++ toString j) = some ((bits (fieldsOf (P.defs[t.defn]!).ret)[j] + 15) / 16) := by
  rw [h.outs]
  exact lookup_map (fun x : Ty × Nat => "o" ++ toString x.2) (fun x => (bits x.1 + 15) / 16) _ h.outs_nodup
    ((fieldsOf (P.defs[t.defn]!).ret)[j], j) (List.mem_zipIdx_iff_getElem?.2 (List.getElem?_eq_getElem hj))

/-- An output's bits, by name, among the template's leaves. -/
theorem bits_lookup (passed : Array (List Nat)) {j : Nat} (hj : j < (fieldsOf (P.defs[t.defn]!).ret).length) :
    (leavesOf P passed t).bits.lookup ("o" ++ toString j) = some (bits (fieldsOf (P.defs[t.defn]!).ret)[j]) :=
  lookup_map (fun x : Ty × Nat => "o" ++ toString x.2) (fun x => bits x.1) _ h.outs_nodup
    ((fieldsOf (P.defs[t.defn]!).ret)[j], j) (List.mem_zipIdx_iff_getElem?.2 (List.getElem?_eq_getElem hj))

/-- An output's leaf width, by name, among the template's leaves. -/
theorem out_leaf (passed : Array (List Nat)) {j : Nat} (hj : j < (fieldsOf (P.defs[t.defn]!).ret).length) :
    (leavesOf P passed t).leaf.lookup ("o" ++ toString j) =
      some (Widths.homo (fieldsOf (P.defs[t.defn]!).ret)[j]) := by
  show ((P.defs[t.defn]!).params.toList.map (fun x : String × Ty => (x.1, Widths.homo x.2)) ++
    (fieldsOf (P.defs[t.defn]!).ret).zipIdx.map fun x : Ty × Nat => ("o" ++ toString x.2, Widths.homo x.1)).lookup
      ("o" ++ toString j) = _
  rw [lookup_append_of_not_mem _ _ _ (by rw [List.map_map]; exact h.out_not_param hj)]
  exact lookup_map (fun x : Ty × Nat => "o" ++ toString x.2) (fun x => Widths.homo x.1) _ h.outs_nodup
    ((fieldsOf (P.defs[t.defn]!).ret)[j], j) (List.mem_zipIdx_iff_getElem?.2 (List.getElem?_eq_getElem hj))

/-- A parameter's leaf width, by name, among the template's leaves. -/
theorem param_leaf (passed : Array (List Nat)) {i : Nat} (hi : i < (P.defs[t.defn]!).params.size) :
    (leavesOf P passed t).leaf.lookup (P.defs[t.defn]!).params[i].1 =
      some (Widths.homo (P.defs[t.defn]!).params[i].2) := by
  show ((P.defs[t.defn]!).params.toList.map (fun x : String × Ty => (x.1, Widths.homo x.2)) ++
    (fieldsOf (P.defs[t.defn]!).ret).zipIdx.map fun x : Ty × Nat => ("o" ++ toString x.2, Widths.homo x.1)).lookup
      (P.defs[t.defn]!).params[i].1 = _
  exact lookup_append_of_some _ _ _ _
    (lookup_map (fun x : String × Ty => x.1) (fun x => Widths.homo x.2) _ h.params_nodup
      (P.defs[t.defn]!).params[i] (Array.mem_toList_iff.2 (Array.getElem_mem hi)))

/-- A closed row is a parameter of one-bit leaves that is none of what its Definition hands on. -/
theorem closed_spec (passed : Array (List Nat)) {i : Nat} (hi : i < (P.defs[t.defn]!).params.size)
    (hc : (leavesOf P passed t).closed.contains (P.defs[t.defn]!).params[i].1 = true) :
    Widths.homo (P.defs[t.defn]!).params[i].2 = some 1 ∧
      i ∉ passed[t.defn]?.getD (List.range (P.defs[t.defn]!).params.size) := by
  have hm := List.contains_iff_mem.1 hc
  change (P.defs[t.defn]!).params[i].1 ∈ (P.defs[t.defn]!).params.toList.zipIdx.filterMap _ at hm
  obtain ⟨⟨⟨name, ty⟩, i'⟩, hx, he⟩ := List.mem_filterMap.1 hm
  have hx' := List.mem_zipIdx_iff_getElem?.1 hx
  simp only at he
  split at he
  · rename_i hcond
    cases he
    obtain ⟨hi', hget⟩ := List.getElem?_eq_some_iff.1 hx'
    have hii : i' = i := by
      refine List.nodup_iff_eq_of_getElem_eq.1 h.params_nodup i' i (by simpa using hi') (by simpa using hi) ?_
      rw [List.getElem_map, List.getElem_map, hget, Array.getElem_toList]
    subst hii
    have hty : (P.defs[t.defn]!).params[i'].2 = ty := by
      have := congrArg Prod.snd hget
      rw [Array.getElem_toList] at this
      exact this
    simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true', List.contains_eq_mem, decide_eq_false_iff_not]
      at hcond
    exact ⟨hty ▸ hcond.1, hcond.2⟩
  · cases he

end Shaped

/-- **What the record check tests of a wire**, in the terms of its two templates' Definitions and the words
`Spot.Wires.check` gives its row: the word each index `x` of the wire reads from (`Wire.offsetAt`) is the first word of
a field of the writer's return of a word or more, and that field is the row's words, whole; and either the reader's parameter has leaves of one bit and is none
of what its Definition hands on, or each side's leaves are of one width and the writer's divides the reader's. -/
theorem whole_spec {P : Prog} {passed : Array (List Nat)} {ts : List Tmpl} (hd : (ts.map (·.tid)).Nodup)
    (hs : ∀ t ∈ ts, Shaped P t) {s : Spot.Wires.Summary}
    (h : unwhole P ts (ts.map (leavesOf P passed)) s = none) {w : Spot.Wires.Wire} {n : Nat}
    (hw : (w, n) ∈ s.wires.zip s.rowWords)
    {tr tw : Tmpl} (htr : tr ∈ ts) (htw : tw ∈ ts) (hr : tr.tid = w.reader) (hwt : tw.tid = w.writer)
    {i : Nat} (hi : i < (P.defs[tr.defn]!).params.size) (hp : (P.defs[tr.defn]!).params[i].1 = w.port) (x : Nat) :
    ∃ j, j < (fieldsOf (P.defs[tw.defn]!).ret).length ∧ Spot.Edges.fieldWord (P.defs[tw.defn]!).ret j = w.offsetAt x ∧
      0 < Spot.Edges.wordsOf (fieldsOf (P.defs[tw.defn]!).ret)[j]! ∧ bits (fieldsOf (P.defs[tw.defn]!).ret)[j]! = 16 * n ∧
      ((Widths.homo (P.defs[tr.defn]!).params[i].2 = some 1 ∧
          i ∉ passed[tr.defn]?.getD (List.range (P.defs[tr.defn]!).params.size)) ∨
        ∃ r o, Widths.homo (P.defs[tr.defn]!).params[i].2 = some r ∧
          Widths.homo (fieldsOf (P.defs[tw.defn]!).ret)[j]! = some o ∧ o ≠ 0 ∧ r % o = 0) := by
  obtain ⟨os, hos, hall⟩ := unwhole_none h hw
  obtain ⟨t, j, hft, hj, hfw, hb, h1, h2⟩ := hall _ (accum_mem _ _ x hos)
  have hfw' := find_of_nodup (·.tid) ts hd tw htw
  rw [hwt] at hfw'
  rw [hfw'] at hft
  obtain rfl := Option.some.inj hft
  have hlw := leaves_find (P := P) (passed := passed) hd htw
  have hlr := leaves_find (P := P) (passed := passed) hd htr
  rw [hwt] at hlw
  rw [hr] at hlr
  rw [hlw] at h1
  rw [hlr, hlw] at h2
  simp only [Option.bind_some, Option.map_some, Option.getD_some, ← hp] at h1 h2
  rw [(hs tw htw).bits_lookup passed hj] at h1
  rw [(hs tr htr).param_leaf passed hi, (hs tw htw).out_leaf passed hj] at h2
  have e : (fieldsOf (P.defs[tw.defn]!).ret)[j] = (fieldsOf (P.defs[tw.defn]!).ret)[j]! :=
    (getElem!_pos (fieldsOf (P.defs[tw.defn]!).ret) j hj).symm
  rw [e] at h1 h2
  refine ⟨j, hj, hfw, hb, h1 _ rfl, ?_⟩
  rcases h2 with h2 | ⟨r, o, hr', ho', h3⟩
  · exact Or.inl ((hs tr htr).closed_spec passed hi h2)
  · exact Or.inr ⟨r, o, by simpa using hr', by simpa using ho', h3⟩

end InferenceOnlyVerifier.Record.SecurityProofs
