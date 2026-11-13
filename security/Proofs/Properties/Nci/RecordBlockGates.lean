module

public import Proofs.Properties.Nci.Record
public import Proofs.Properties.Nci.RecordBlockSize
public import Proofs.Properties.Nci.RecordBlocks
public import Proofs.Properties.Nci.RecordForm
public import Proofs.Properties.Nci.RecordGates
public import Proofs.Properties.Nci.RecordLeafWide
public import Proofs.Properties.Nci.RecordProgram
public import Proofs.Properties.Nci.RecordRead
public import Proofs.Properties.Nci.RecordTemplates
public import Mathlib.Tactic

/-!
# A block has the summary's gates, or none

In every circuit a record the check reads lays out, a block has the gates the summary counts, or no gate
(`block_gates`): the second half of `Properties.Record.Counts`.

A block is a slot's units (`Divides`), and a unit's gates are its Definition's. The units' gate ranges are apart
(`units_apart`); every gate of a unit is a computed gate of the circuit, since it is computed (`unitsComputed`) and
no wire covers it, a wired gate being an input gate (`OwnRows`); and every computed gate of the circuit lies in a
unit. So a block's size is the gates of its slot's units (`block_size`): range by range of the root, the range's
units in the slot times its Definition's gates (`block_ranges`).

A template's grids are whole periods of the slots and tile its units (`tmpl_grids`), so each of the `S` slots holds
the template's units over `S` (`tmpl_slot_card`), and the block of such a slot has the summary's gates. A unit's slot
is below `S` (`tmpl_slot_lt`), so a block of any other number has no gate. `block_gates_of` is the statement for any
Program, text and circuit; `block_gates` is it of a record `read` accepts.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Grid Tmpl gridsOf templatesOf slotIn)
open NciCheck.Record.SecurityProofs (form slotOf OneGrid slotOf_of_mem gridsOf_ok gridsOf_period gridsOf_covers
  tiles_ordered tiles_sum mem_of_lookup eq_of_nodup_map read_front finish_text front_text)
open Lean (Json)

/-! ## Counting over `Fin` -/

/-- The members of `Fin n` a predicate on the naturals holds of are as many as the naturals below `n` it holds
of. -/
private theorem card_fin (q : ℕ → Prop) [DecidablePred q] (n : ℕ) :
    (Finset.univ.filter fun v : Fin n => q v.val).card = ((List.range n).filter fun v => decide (q v)).length := by
  have key : ∀ m : ℕ, (∑ v ∈ Finset.range m, if q v then (1 : ℕ) else 0) =
      ((List.range m).filter fun v => decide (q v)).length := by
    intro m
    induction m with
    | zero => simp
    | succ m ih =>
      rw [Finset.sum_range_succ, ih, List.range_succ, List.filter_append, List.length_append]
      by_cases hq : q m
      · simp [hq]
      · simp [hq]
  rw [Finset.card_filter, ← key n]
  exact Fin.sum_univ_eq_sum_range (fun v => if q v then (1 : ℕ) else 0) n

/-- A sum over `Fin n` of a function of the index is the sum over a list of `n` elements, where the function at an
index is a function of the list's element there. -/
private theorem sum_fin_map {α : Type} (g : α → ℕ) : ∀ (l : List α) (n : ℕ) (f : ℕ → ℕ), l.length = n →
    (∀ i (hi : i < l.length), f i = g l[i]) → ∑ i : Fin n, f i.val = (l.map g).sum
  | [], n, f, hn, _ => by
    subst hn
    exact Fin.sum_univ_zero _
  | a :: l, n, f, hn, h => by
    subst hn
    have ih := sum_fin_map g l l.length (fun i => f (i + 1)) rfl fun i hi => h (i + 1) (Nat.succ_lt_succ hi)
    have h0 : f 0 = g a := h 0 (Nat.succ_pos _)
    rw [List.map_cons, List.sum_cons, ← ih, ← h0]
    show ∑ i : Fin (l.length + 1), f i.val = _
    exact (Fin.sum_univ_succ fun i : Fin (l.length + 1) => f i.val).trans rfl

/-- The text's template at a range's place has the range's template and its count for units. -/
private theorem tmpl_of_range {ts : List Tmpl} {rs : Array TRange}
    (hmap : ts.map (fun t => (t.tid, t.units)) = rs.toList.map (fun r => (r.template, r.count))) {i : ℕ}
    (hi : i < rs.size) : ∃ t, ts[i]? = some t ∧ t.tid = rs[i]!.template ∧ t.units = rs[i]!.count := by
  have hlen : ts.length = rs.size := by simpa using congrArg List.length hmap
  have hi' : i < ts.length := by omega
  have h : (ts.map fun t => (t.tid, t.units))[i]? = (rs.toList.map fun r => (r.template, r.count))[i]? := by
    rw [hmap]
  refine ⟨ts[i], List.getElem?_eq_getElem hi', ?_⟩
  rw [getElem!_pos rs i hi]
  simpa [hi, hi'] using h

/-! ## A template's grids -/

/-- **A template's grids**, where the check reads the text's grids: no unit lies in two of them, each has a row
length of at least one and is whole periods of the slots, each lies in the template's units, and together they are
the template's units. -/
theorem tmpl_grids {ts : List Tmpl} {doc : Json} {S : ℕ} {per : List (String × List Grid)}
    (h : gridsOf ts doc = .ok (S, per)) (hd : (ts.map (·.tid)).Nodup) {t : Tmpl} (ht : t ∈ ts) :
    OneGrid ((per.lookup t.tid).getD []) ∧ (∀ g ∈ (per.lookup t.tid).getD [], 0 < g.q) ∧
      (∀ g ∈ (per.lookup t.tid).getD [], g.q * S ∣ g.units) ∧
      (∀ g ∈ (per.lookup t.tid).getD [], g.start + g.units ≤ t.units) ∧
      (((per.lookup t.tid).getD []).map (·.units)).sum = t.units := by
  obtain ⟨gs, hgs⟩ := (gridsOf_ok h).2.2 t ht
  cases hl : per.lookup t.tid with
  | none =>
    exact absurd hl (by
      intro hn
      have := List.lookup_eq_none_iff.1 hn (t.tid, gs) hgs
      simp at this)
  | some gs' =>
    have hmem := mem_of_lookup _ _ _ hl
    obtain ⟨t', ht', he, h1, hq⟩ := (gridsOf_ok h).2.1 t.tid gs' hmem
    have : t' = t := eq_of_nodup_map (·.tid) ts hd t' ht' t ht he
    subst this
    obtain ⟨-, hin, hpw⟩ := tiles_ordered gs' 0 t'.units h1
    have hone : OneGrid gs' := hpw.imp fun hab => Or.inl hab
    show OneGrid gs' ∧ (∀ g ∈ gs', 0 < g.q) ∧ (∀ g ∈ gs', g.q * S ∣ g.units) ∧
      (∀ g ∈ gs', g.start + g.units ≤ t'.units) ∧ (gs'.map (·.units)).sum = t'.units
    exact ⟨hone, hq, gridsOf_period h t'.tid gs' hmem, fun g hg => (hin g hg).2.1,
      (Nat.zero_add _).symm.trans (tiles_sum gs' 0 t'.units h1)⟩

/-- **A unit of a template has a slot, below the number of slots.** -/
theorem tmpl_slot_lt {ts : List Tmpl} {doc : Json} {S : ℕ} {per : List (String × List Grid)}
    (h : gridsOf ts doc = .ok (S, per)) (hd : (ts.map (·.tid)).Nodup) {t : Tmpl} (ht : t ∈ ts) {u : ℕ}
    (hu : u < t.units) : ∃ s, slotIn S per t.tid u = some s ∧ s < S := by
  obtain ⟨g, hg, h1, h2⟩ := gridsOf_covers h hd ht hu
  have hs : slotOf S ((per.lookup t.tid).getD []) u = some ((u - g.start) / g.q % S) :=
    slotOf_of_mem S _ u (tmpl_grids h hd ht).1 g hg h1 h2
  exact ⟨_, hs, Nat.mod_lt _ (gridsOf_ok h).1⟩

/-- **A slot holds a template's units over the slots**, counted among the naturals below the template's units. -/
theorem tmpl_slot_units {ts : List Tmpl} {doc : Json} {S : ℕ} {per : List (String × List Grid)}
    (h : gridsOf ts doc = .ok (S, per)) (hd : (ts.map (·.tid)).Nodup) {t : Tmpl} (ht : t ∈ ts) {s : ℕ}
    (hs : s < S) : ((List.range t.units).filter fun u => slotIn S per t.tid u == some s).length = t.units / S := by
  obtain ⟨hg, hq, hper, hin, hsum⟩ := tmpl_grids h hd ht
  exact slot_units S ((per.lookup t.tid).getD []) t.units s hs hg hq hper hin hsum

/-- The same, counted in `Fin`: the units of a template whose slot is `s` are the template's units over the
slots. -/
theorem tmpl_slot_card {ts : List Tmpl} {doc : Json} {S : ℕ} {per : List (String × List Grid)}
    (h : gridsOf ts doc = .ok (S, per)) (hd : (ts.map (·.tid)).Nodup) {t : Tmpl} (ht : t ∈ ts) {s : ℕ}
    (hs : s < S) {tid : String} {n : ℕ} (htid : t.tid = tid) (hn : t.units = n) :
    (Finset.univ.filter fun v : Fin n => (slotIn S per tid v.val).getD 0 = s).card = n / S := by
  subst htid hn
  refine ((card_fin (fun v => (slotIn S per t.tid v).getD 0 = s) t.units).trans ?_).trans
    (tmpl_slot_units h hd ht hs)
  refine congrArg List.length (List.filter_congr fun v hv => ?_)
  show decide ((slotIn S per t.tid v).getD 0 = s) = (slotIn S per t.tid v == some s)
  obtain ⟨s', hs', -⟩ := tmpl_slot_lt h hd ht (List.mem_range.1 hv)
  rw [hs']
  by_cases hss : s' = s
  · subst hss
    simp
  · simp [hss]

/-! ## A block's size, range by range -/

/-- **A block's gates, range by range.** Where the computed gates of a circuit lie in units, unit `v` of range `i`
the gates `[off i v, off i v + G i)`, the units' gates apart, all of them computed and all of block `slot i v`, and
range `i` has `cnt i` units in block `u`: block `u` has, range by range, `cnt i` times `G i` gates. -/
theorem block_ranges (D : Verity.Circuit) {n : ℕ} (U : Verity.Partition D n) (at_ : Fin D.N → ℕ)
    (hinj : Function.Injective at_) {m : ℕ} (units G cnt : Fin m → ℕ)
    (off slot : (i : Fin m) → Fin (units i) → ℕ)
    (hapart : ∀ (i i' : Fin m) (v : Fin (units i)) (v' : Fin (units i')), i.val ≠ i'.val ∨ v.val ≠ v'.val →
      off i v + G i ≤ off i' v' ∨ off i' v' + G i' ≤ off i v)
    (hin : ∀ x, D.isInput x = false →
      ∃ i v, off i v ≤ at_ x ∧ at_ x < off i v + G i ∧ (U.unit x).map Fin.val = some (slot i v))
    (hall : ∀ i v g, off i v ≤ g → g < off i v + G i → ∃ x, at_ x = g ∧ D.isInput x = false) (u : Fin n)
    (hc : ∀ i, (Finset.univ.filter fun v : Fin (units i) => slot i v = u.val).card = cnt i) :
    D.size (U.gates u) = ∑ i, cnt i * G i := by
  rw [← slot_sum units G cnt slot u.val hc]
  refine block_size D U at_ hinj (ι := (i : Fin m) × Fin (units i)) (fun a => off a.1 a.2) (fun a => G a.1)
    (fun a => slot a.1 a.2) ?_ ?_ ?_ u
  · rintro ⟨i, v⟩ ⟨i', v'⟩ hne
    refine hapart i i' v v' ?_
    by_cases h1 : i.val = i'.val
    · have hi : i = i' := Fin.ext h1
      subst hi
      refine Or.inr fun h2 => hne ?_
      rw [Fin.ext h2]
    · exact Or.inl h1
  · intro x hx
    obtain ⟨i, v, h1, h2, h3⟩ := hin x hx
    exact ⟨⟨i, v⟩, h1, h2, h3⟩
  · rintro ⟨i, v⟩ g h1 h2
    exact hall i v g h1 h2

/-! ## The summary's gates -/

/-- **A block has the summary's gates, or none**, of any Program the record check reads, text whose templates and
grids the check reads, and circuit the wires lay out with the slots for blocks: block `u` has, template by template,
the template's units over the slots times its Definition's gates, or no gate. -/
theorem block_gates_of {P : Prog} {doc : Json} {ts : List Tmpl} {S : ℕ} {per : List (String × List Grid)}
    {tids : Array String} {rs : Array TRange} {W : Spot.Wires.Wiring} {Sw : Spot.Wires.Summary}
    {D : Verity.Circuit} {at_ : Fin D.N → ℕ} {n : ℕ} {U : Verity.Partition D n}
    (hP : ReadProg P) (hts : templatesOf P doc = .ok ts) (hgr : gridsOf ts doc = .ok (S, per))
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids = .ok rs)
    (hO : OwnRows P rs W) (hL : LaysOut P rs W Sw D at_) (hDiv : Divides P rs (slotIn S per) at_ U) (u : Fin n) :
    D.size (U.gates u) = (ts.map fun t => t.units / S * (P.defs[t.defn]!).gates).sum ∨ U.gates u = ∅ := by
  obtain ⟨-, hmap⟩ := templatesOf_ranges hts hrs
  have hnd : (ts.map (·.tid)).Nodup := (tmpls_named hts).1
  have hr : P.root < P.defs.size := (read_bodies hP).1
  have hcomp := unitsComputed hP hrs
  have hslot : ∀ {tid : String} {v : ℕ} {sc : Scope}, UnitAt P rs tid v sc →
      ∃ s, slotIn S per tid v = some s ∧ s < S := by
    intro tid v sc hunit
    obtain ⟨t, ht, htid, hlt, -, -⟩ := unit_tmpl hP hts hrs hunit
    rw [← htid]
    exact tmpl_slot_lt hgr hnd ht hlt
  by_cases hu : u.val < S
  · left
    have hlen : ts.length = rs.size := by simpa using congrArg List.length hmap
    have hsum : ∑ i : Fin rs.size, rs[i.val]!.count / S *
          (P.defs[((P.defs[P.root]!).body.nodes[rs[i.val]!.node]!).fn]!).gates =
        (ts.map fun t => t.units / S * (P.defs[t.defn]!).gates).sum := by
      refine sum_fin_map (fun t : Tmpl => t.units / S * (P.defs[t.defn]!).gates) ts rs.size
        (fun i => rs[i]!.count / S * (P.defs[((P.defs[P.root]!).body.nodes[rs[i]!.node]!).fn]!).gates) hlen ?_
      intro i hi
      have hir : i < rs.size := by omega
      obtain ⟨t, hget, htt, htu⟩ := tmpl_of_range hmap hir
      have hti : ts[i] = t := Option.some.inj ((List.getElem?_eq_getElem hi).symm.trans hget)
      show rs[i]!.count / S * (P.defs[((P.defs[P.root]!).body.nodes[rs[i]!.node]!).fn]!).gates =
        ts[i].units / S * (P.defs[ts[i].defn]!).gates
      rw [hti, htu]
      rcases Nat.eq_zero_or_pos rs[i]!.count with h0 | hpos
      · rw [h0, Nat.zero_div, Nat.zero_mul, Nat.zero_mul]
      · have hunit : UnitAt P rs rs[i]!.template 0 (P.child P.rootScope rs[i]!.node 0) :=
          (unitAt_iff hts hrs _ _ _).2
            ⟨i, t, hir, rfl, hget, htt, htu, Nat.lt_of_lt_of_eq hpos htu.symm, rfl⟩
        obtain ⟨t', ht', htid', -, hdefn, -⟩ := unit_tmpl hP hts hrs hunit
        have : t' = t :=
          eq_of_nodup_map (·.tid) ts hnd t' ht' t (List.mem_of_getElem? hget) (htid'.trans htt.symm)
        subst this
        rw [hdefn, (child_root P _ _).1]
    rw [← hsum]
    refine block_ranges D U at_ hL.1 (fun i : Fin rs.size => rs[i.val]!.count)
      (fun i => (P.defs[((P.defs[P.root]!).body.nodes[rs[i.val]!.node]!).fn]!).gates)
      (fun i => rs[i.val]!.count / S)
      (fun i v => (P.child P.rootScope rs[i.val]!.node v.val).off)
      (fun i v => (slotIn S per rs[i.val]!.template v.val).getD 0) ?_ ?_ ?_ u ?_
    · intro i i' v v' hne
      exact units_apart hP hrs i.isLt i'.isLt v.isLt v'.isLt hne
    · intro x hx
      obtain ⟨tid, v, sc, s, hunit, h1, h2, hs, hmapu⟩ := hDiv x hx
      obtain ⟨i, t, hi, hti, -, -, htu, hvt, rfl⟩ := (unitAt_iff hts hrs tid v sc).1 hunit
      refine ⟨⟨i, hi⟩, ⟨v, Nat.lt_of_lt_of_eq hvt htu⟩, h1, h2, ?_⟩
      show (U.unit x).map Fin.val = some ((slotIn S per rs[i]!.template v).getD 0)
      rw [hti, hs]
      exact hmapu
    · intro i v g h1 h2
      obtain ⟨hk, hcnt⟩ := range_node hr hP.2.2.2.2.2.2 (readProg_inputs hP) hrs i.isLt
      obtain ⟨t, hget, htt, htu⟩ := tmpl_of_range hmap i.isLt
      have hunit : UnitAt P rs rs[i.val]!.template v.val (P.child P.rootScope rs[i.val]!.node v.val) :=
        (unitAt_iff hts hrs _ _ _).2
          ⟨i.val, t, i.isLt, rfl, hget, htt, htu, Nat.lt_of_lt_of_eq v.isLt htu.symm, rfl⟩
      have hlt : g < gates P := unit_gate_lt hP hk (Nat.lt_of_lt_of_eq v.isLt hcnt) h2
      have hnw : ¬ Wired P rs W Sw g := by
        rintro ⟨r, l, lo, up, -, hrl⟩
        obtain ⟨p, ops, sc', hgate, hinp⟩ := hO.1 r l g lo up hrl
        have hno := hcomp _ _ _ g p ops sc' hunit h1 h2 hgate
        rw [hinp] at hno
        cases hno
      obtain ⟨x, hx⟩ := (hL.2.1 g).2 ⟨hlt, hnw⟩
      obtain ⟨p, ops, sc', hgate, -, hinp, -⟩ := hL.2.2 x
      rw [hx] at hgate
      refine ⟨x, hx, ?_⟩
      rw [hinp]
      exact hcomp _ _ _ g p ops sc' hunit h1 h2 hgate
    · intro i
      obtain ⟨t, hget, htt, htu⟩ := tmpl_of_range hmap i.isLt
      exact tmpl_slot_card hgr hnd (List.mem_of_getElem? hget) hu htt htu
  · right
    by_contra hne
    obtain ⟨x, hx⟩ := Finset.nonempty_iff_ne_empty.2 hne
    have hxu : U.unit x = some u := by
      simpa only [Verity.Partition.gates, Finset.mem_filter, Finset.mem_univ, true_and] using hx
    have hxi : D.isInput x = false := by
      cases hb : D.isInput x with
      | false => rfl
      | true =>
        have hnone := (U.input_iff x).2 hb
        rw [hxu] at hnone
        exact absurd hnone (by simp)
    obtain ⟨tid, v, sc, s, hunit, -, -, hs, hmapu⟩ := hDiv x hxi
    obtain ⟨s', hs', hlt⟩ := hslot hunit
    rw [hxu] at hmapu
    have h1 : some u.val = some s := hmapu
    have h2 : s = s' := Option.some.inj (hs.symm.trans hs')
    have h3 : u.val = s := Option.some.inj h1
    omega

/-- **A block has the summary's gates, or none**: in a circuit a record the check reads lays out, every block
has the gates the summary counts, or no gate. -/
theorem block_gates {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c)
    {d : Verity.RegisteredCircuit} (hd : Served form c d) :
    ∀ u, d.D.size (d.𝓘.gates u) = (form.summarize c).gates ∨ d.𝓘.gates u = ∅ := by
  intro u
  obtain ⟨P, doc, W, Sw, rs, at_, hdec, -, -, -, -, -, -, -, hrs, -, hO, -, -, -, hL, -, hDiv⟩ := hd
  have hPc : c.f.P = P := Except.ok.inj ((decode_of_read h).symm.trans hdec)
  subst hPc
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, hgr, -⟩ := finish_text hc
  have hDiv' : Divides c.f.P rs (slotIn c.S c.grids) at_ d.𝓘 := hDiv
  show d.D.size (d.𝓘.gates u) = (c.f.ts.map fun t => t.units / c.S * (c.f.P.defs[t.defn]!).gates).sum ∨
    d.𝓘.gates u = ∅
  exact block_gates_of (readProg_of_read h) (front_text hf) hgr hrs hO hL hDiv' u

end NciCheck.Properties.Record.SecurityProofs
