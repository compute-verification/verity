module

public import Proofs.Properties.Nci.RecordInputs
public import Proofs.Properties.Nci.BaselineSteps

/-!
# What a table read and an opening declare

`RecordText.inputsOf` reads a table read and each port of an opening as a declaration of a rectangle
(`RecordText.rect`): the reader's units `[at, at + units)`, every row of the port's cut. A second walk down its stages
(`inputsOf_declared`), by one invariant through each loop (`Declared`, with `BaselineSteps.forIn_list_prefix`), says
which declaration each is:

* every table read the declarations list has its declaration, the rectangle of its reader, port and units, filled
  by the table's root or its domain id;
* every declaration an opening fills is the rectangle of one of that opening's ports, over the opening's reader and
  units, and the opening is the one the declaration names by its index.

With the first walk's disjointness (`inputsOf_decls`) that gives what fills a row:

* `covers_iff`, `rect_covers`: the rows a declaration covers, and a rectangle's: the reader's, the port's, the units
  `[at, at + units)` and the rows below the cut. A rectangle over a cut of 0 covers no row.
* `fillAt_of_covers`: among declarations no two of which fill one row, the one that covers a row is what fills it.
  Two declarations that cover one row are not disjoint (`covers_not_disjoint`), so the first that covers it has the
  same fill.
* `table_fill`: a row of a table read's rectangle is filled by that table read.
* `opening_of_fill`: a row an opening fills is a port of that opening, of its reader, at one of its units.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open NciCheck.RecordText (Row Tmpl Decl Fill Inputs TableRead Opening inputsOf fillAt rect)
open NciCheck.Properties.SecurityProofs.BaselineSteps (forIn_list_prefix)
open Lean (Json)

/-! ## The rows a declaration covers -/

/-- **The rows a declaration's rectangle covers**: its reader's and its port's, at its reader units and its rows. -/
theorem covers_iff (w : Spot.Wires.Wire) (r : Row) :
    (w.indexOf r.tid r.unit r.port r.row).isSome = true ↔
      w.reader = r.tid ∧ w.port = r.port ∧ w.start ≤ r.unit ∧ r.unit < w.start + w.readerUnits ∧
        w.cut0 ≤ r.row ∧ r.row < w.cut0 + w.cuts := by
  unfold Spot.Wires.Wire.indexOf
  split
  · rename_i h
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    simp only [Option.isSome_some, true_iff]
    exact ⟨h.1.1.1.1.1, h.1.1.1.1.2, h.1.1.1.2, h.1.1.2, h.1.2, h.2⟩
  · rename_i h
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    simp only [Option.isSome_none, Bool.false_eq_true, false_iff]
    rintro ⟨h1, h2, h3, h4, h5, h6⟩
    exact h ⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩

/-- **The rows a rectangle covers**: the reader's and the port's, at the units `[start, start + units)` and the rows
below the cut `k`. Over a cut of 0 it covers none. -/
theorem rect_covers (reader port : String) (start units k : Nat) (r : Row) :
    ((rect reader port start units k).indexOf r.tid r.unit r.port r.row).isSome = true ↔
      reader = r.tid ∧ port = r.port ∧ start ≤ r.unit ∧ r.unit < start + units ∧ r.row < k := by
  rw [covers_iff]
  show reader = r.tid ∧ port = r.port ∧ start ≤ r.unit ∧ r.unit < start + k * (units * 1) / k ∧ 0 ≤ r.row ∧
    r.row < 0 + k ↔ _
  constructor
  · rintro ⟨h1, h2, h3, h4, -, h6⟩
    have hk : 0 < k := by omega
    rw [Nat.mul_one, Nat.mul_div_cancel_left _ hk] at h4
    exact ⟨h1, h2, h3, h4, by omega⟩
  · rintro ⟨h1, h2, h3, h4, h5⟩
    have hk : 0 < k := by omega
    refine ⟨h1, h2, h3, ?_, Nat.zero_le _, by omega⟩
    rw [Nat.mul_one, Nat.mul_div_cancel_left _ hk]
    exact h4

/-- Two rectangles that cover one row are not disjoint. -/
theorem covers_not_disjoint {a b : Spot.Wires.Wire} {r : Row}
    (ha : (a.indexOf r.tid r.unit r.port r.row).isSome = true)
    (hb : (b.indexOf r.tid r.unit r.port r.row).isSome = true) : Spot.Wires.disjoint a b = false := by
  obtain ⟨a1, a2, a3, a4, a5, a6⟩ := (covers_iff a r).1 ha
  obtain ⟨b1, b2, b3, b4, b5, b6⟩ := (covers_iff b r).1 hb
  unfold Spot.Wires.disjoint
  simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq, decide_eq_false_iff_not, Nat.not_le]
  exact ⟨⟨⟨⟨⟨a1.trans b1.symm, a2.trans b2.symm⟩, by omega⟩, by omega⟩, by omega⟩, by omega⟩

/-- Among pairwise disjoint declarations, the first that covers a row has the fill of any that covers it. -/
theorem find?_covers : ∀ (ds : List Decl), (ds.map (·.w)).Pairwise (fun a b => Spot.Wires.disjoint a b = true) →
    ∀ (d : Decl) (r : Row), d ∈ ds → (d.w.indexOf r.tid r.unit r.port r.row).isSome = true →
      (ds.find? fun d => (d.w.indexOf r.tid r.unit r.port r.row).isSome).map (·.fill) = some d.fill
  | [], _, _, _, hd, _ => absurd hd List.not_mem_nil
  | e :: es, hp, d, r, hd, hc => by
    rw [List.map_cons, List.pairwise_cons] at hp
    rw [List.find?_cons]
    cases he : (e.w.indexOf r.tid r.unit r.port r.row).isSome with
    | true =>
      rcases List.mem_cons.1 hd with rfl | hd'
      · rfl
      · have hdis := hp.1 d.w (List.mem_map.2 ⟨d, hd', rfl⟩)
        rw [covers_not_disjoint he hc] at hdis
        cases hdis
    | false =>
      rcases List.mem_cons.1 hd with rfl | hd'
      · rw [he] at hc
        cases hc
      · exact find?_covers es hp.2 d r hd' hc

/-- **Among declarations no two of which fill one row, the one that covers a row is what fills it.** -/
theorem fillAt_of_covers {ds : List Decl} (hdis : Spot.Wires.pairwiseDisjoint (ds.map (·.w)) = true) {d : Decl}
    (hd : d ∈ ds) {r : Row} (hc : (d.w.indexOf r.tid r.unit r.port r.row).isSome = true) :
    fillAt ds r = some d.fill :=
  find?_covers ds ((Spot.Wires.pairwiseDisjoint_iff _).1 hdis) d r hd hc

/-- What fills a row is the fill of a declaration that covers it. -/
theorem fillAt_some {ds : List Decl} {r : Row} {fill : Fill} (h : fillAt ds r = some fill) :
    ∃ d ∈ ds, (d.w.indexOf r.tid r.unit r.port r.row).isSome = true ∧ d.fill = fill := by
  unfold fillAt at h
  cases hf : ds.find? (fun d => (d.w.indexOf r.tid r.unit r.port r.row).isSome) with
  | none =>
    rw [hf] at h
    cases h
  | some d =>
    rw [hf] at h
    have hc := List.find?_some hf
    exact ⟨d, List.mem_of_find?_eq_some hf, hc, Option.some.inj h⟩

/-! ## Which declaration a table read and an opening's port are -/

/-- **What the walk keeps** of the declarations `ds`, the table reads `roots` and the openings `os` read so far, with
the cuts of `W`: each table read has its declaration, the rectangle of its reader, port and units, filled by its
table's root or domain id; and each declaration an opening fills is the rectangle of one of that opening's ports, over
its reader and units. -/
def Declared (W : Spot.Wires.Wiring) (ds : List Decl) (roots : List TableRead) (os : List Opening) : Prop :=
  (∀ rd ∈ roots, ∃ d ∈ ds, d.w = rect rd.reader rd.port rd.start rd.units (W.cutsOf rd.reader rd.port) ∧
      d.fill = .table rd.table rd.root) ∧
    ∀ d ∈ ds, ∀ i, d.fill = .opening i → ∃ o, os[i]? = some o ∧ ∃ p ∈ o.ports,
      d.w = rect o.reader p o.start o.units (W.cutsOf o.reader p)

/-- Declarations none of which an opening fills, with no table read and no opening. -/
theorem declared_start {W : Spot.Wires.Wiring} {ds : List Decl} (h : ∀ d ∈ ds, ∀ i, d.fill ≠ .opening i) :
    Declared W ds [] [] :=
  ⟨fun _ hrd => absurd hrd List.not_mem_nil, fun d hd i hi => absurd hi (h d hd i)⟩

/-- A declaration no opening fills, after the others. -/
theorem declared_push {W : Spot.Wires.Wiring} {a : Array Decl} {roots : List TableRead} {os : List Opening}
    (d : Decl) (hd : ∀ i, d.fill ≠ .opening i) (h : Declared W a.toList roots os) :
    Declared W (a.push d).toList roots os := by
  obtain ⟨hr, ho⟩ := h
  rw [Array.toList_push]
  refine ⟨fun rd hrd => ?_, fun e he i hi => ?_⟩
  · obtain ⟨d', hd', hw⟩ := hr rd hrd
    exact ⟨d', List.mem_append_left _ hd', hw⟩
  · rcases List.mem_append.1 he with he | he
    · exact ho e he i hi
    · rw [List.mem_singleton.1 he] at hi
      exact absurd hi (hd i)

/-- A table read, with its declaration. -/
theorem declared_root {W : Spot.Wires.Wiring} {a : Array Decl} {roots : List TableRead} {os : List Opening}
    (reader port table : String) (root : Bool) (start units bits : Nat) (h : Declared W a.toList roots os) :
    Declared W (a.push ⟨rect reader port start units (W.cutsOf reader port), bits, .table table root⟩).toList
      (roots ++ [⟨reader, port, table, root, start, units⟩]) os := by
  obtain ⟨hr, ho⟩ := declared_push (W := W) (roots := roots) (os := os)
    ⟨rect reader port start units (W.cutsOf reader port), bits, .table table root⟩ (fun i e => by cases e) h
  refine ⟨fun rd hrd => ?_, ho⟩
  rcases List.mem_append.1 hrd with hrd | hrd
  · exact hr rd hrd
  · rw [List.mem_singleton.1 hrd]
    exact ⟨_, by rw [Array.toList_push]; exact List.mem_append_right _ (List.mem_singleton.2 rfl), rfl, rfl⟩

/-- One more opening, which no declaration names yet. -/
theorem declared_open {W : Spot.Wires.Wiring} {ds : List Decl} {roots : List TableRead} {os : List Opening}
    (o : Opening) (h : Declared W ds roots os) : Declared W ds roots (os ++ [o]) := by
  refine ⟨h.1, fun d hd i hi => ?_⟩
  obtain ⟨o', ho', hp⟩ := h.2 d hd i hi
  obtain ⟨hlt, -⟩ := List.getElem?_eq_some_iff.1 ho'
  exact ⟨o', by rw [List.getElem?_append_left hlt]; exact ho', hp⟩

/-- One more port of the last opening, with its declaration. -/
theorem declared_port {W : Spot.Wires.Wiring} {a : Array Decl} {roots : List TableRead} {os : List Opening}
    (reader table key : String) (ports : List String) (verdict : String) (start units : Nat) (p : String)
    (bits : Nat) (h : Declared W a.toList roots (os ++ [⟨reader, table, key, ports, verdict, start, units⟩])) :
    Declared W (a.push ⟨rect reader p start units (W.cutsOf reader p), bits, .opening os.length⟩).toList roots
      (os ++ [⟨reader, table, key, ports ++ [p], verdict, start, units⟩]) := by
  obtain ⟨hr, ho⟩ := h
  rw [Array.toList_push]
  refine ⟨fun rd hrd => ?_, fun e he i hi => ?_⟩
  · obtain ⟨d', hd', hw⟩ := hr rd hrd
    exact ⟨d', List.mem_append_left _ hd', hw⟩
  · rcases List.mem_append.1 he with he | he
    · obtain ⟨o', ho', q, hq, hw⟩ := ho e he i hi
      by_cases hlt : i < os.length
      · rw [List.getElem?_append_left hlt] at ho' ⊢
        exact ⟨o', ho', q, hq, hw⟩
      · obtain ⟨hlt', -⟩ := List.getElem?_eq_some_iff.1 ho'
        rw [List.length_append, List.length_singleton] at hlt'
        have hi' : i = os.length := by omega
        subst hi'
        rw [List.getElem?_append_right (Nat.le_refl _), Nat.sub_self, List.getElem?_cons_zero,
          Option.some.injEq] at ho'
        subst ho'
        exact ⟨⟨reader, table, key, ports ++ [p], verdict, start, units⟩,
          by rw [List.getElem?_append_right (Nat.le_refl _), Nat.sub_self, List.getElem?_cons_zero], q,
          List.mem_append_left _ hq, hw⟩
    · rw [List.mem_singleton.1 he] at hi
      cases hi
      rw [List.mem_singleton.1 he]
      exact ⟨_, by rw [List.getElem?_append_right (Nat.le_refl _), Nat.sub_self, List.getElem?_cons_zero], p,
        List.mem_append_right _ (List.mem_singleton.2 rfl), rfl⟩

/-- An opening, read after the others: the loop over its ports, each step of which declares one port's rectangle
for the opening the declarations will name next, keeps what the walk keeps. -/
theorem declared_opening {W : Spot.Wires.Wiring} {roots : List TableRead} {os : List Opening} {α : Type}
    (reader table key verdict : String) (start units : Nat)
    {f : α → Array Decl × List String → Except String (ForInStep (Array Decl × List String))}
    (hf : ∀ x s s', f x s = .ok s' → ∃ p bits, s' = .yield
      (s.1.push ⟨rect reader p start units (W.cutsOf reader p), bits, .opening os.length⟩, s.2 ++ [p]))
    (l : Array α) (a a' : Array Decl) (ports : List String) (h : Declared W a.toList roots os)
    (hl : forIn l (a, []) f = .ok (a', ports)) :
    Declared W a'.toList roots (os ++ [⟨reader, table, key, ports, verdict, start, units⟩]) := by
  rw [← Array.forIn_toList] at hl
  exact forIn_list_prefix
    (fun _ s => Declared W s.1.toList roots (os ++ [⟨reader, table, key, s.2, verdict, start, units⟩])) f
    l.toList [] _ _
    (by
      intro _ x s s' hs hx
      obtain ⟨p, bits, rfl⟩ := hf x s s' hx
      exact ⟨_, rfl, declared_port reader table key s.2 verdict start units p bits hs⟩)
    (declared_open _ h) hl

/-- **Which declaration a table read and an opening's port are**, of the declarations `inputsOf` returns: each
table read has its declaration, the rectangle of its reader, port and units, filled by its table's root or domain id;
and each declaration an opening fills is the rectangle of one of that opening's ports, over its reader and units. -/
theorem inputsOf_declared {W : Spot.Wires.Wiring} {ts : List Tmpl} {doc : Json} {x : Inputs}
    (h : inputsOf ts W doc = .ok x) : Declared W x.decls x.roots x.openings := by
  unfold inputsOf at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat' split at h
  all_goals try (cases h; done)
  rename_i d1 h1 _ _ _ _ d2 h2 _ _ _ _ d3 h3 _ _ _ _ d4 h4 _
  cases h
  have b1 : Declared W d1.toList [] [] := by
    rw [← Array.forIn_toList] at h1
    refine forIn_list_prefix (fun _ a => Declared W a.toList [] []) _ _ [] _ _ ?_ (declared_start ?_) h1
    · intro _ r b s hb hs
      repeat' split at hs
      all_goals try (cases hs; done)
      all_goals
        cases hs
        exact ⟨_, rfl, declared_push _ (fun i e => by cases e) hb⟩
    · intro d hd i hi
      obtain ⟨wi, -, rfl⟩ := List.mem_map.1 hd
      cases hi
  have b2 : Declared W d2.1.toList d2.2 [] := by
    rw [← Array.forIn_toList] at h2
    refine forIn_list_prefix (fun _ s => Declared W s.1.toList s.2 []) _ _ [] _ _ ?_ b1 h2
    intro _ r b s hb hs
    repeat' split at hs
    all_goals try (cases hs; done)
    cases hs
    exact ⟨_, rfl, declared_root _ _ _ _ _ _ _ hb⟩
  have b3 : Declared W d3.1.toList d2.2 d3.2 := by
    rw [← Array.forIn_toList] at h3
    refine forIn_list_prefix (fun _ s => Declared W s.1.toList d2.2 s.2) _ _ [] _ _ ?_ b2 h3
    intro _ o b s hb hs
    repeat' split at hs
    all_goals try (cases hs; done)
    rename_i hin
    cases hs
    exact ⟨_, rfl, declared_opening _ _ _ _ _ _ (by
      intro pj s' s'' hs'
      repeat' split at hs'
      all_goals try (cases hs'; done)
      cases hs'
      exact ⟨_, _, rfl⟩) _ _ _ _ hb hin⟩
  rw [← Array.forIn_toList] at h4
  refine forIn_list_prefix (fun _ a => Declared W a.toList d2.2 d3.2) _ _ [] _ _ ?_ b3 h4
  intro _ g b s hb hs
  repeat' split at hs
  all_goals try (cases hs; done)
  cases hs
  exact ⟨_, rfl, declared_push _ (fun i e => by cases e) hb⟩

/-! ## What fills a table read's rows and an opening's -/

/-- **A row of a table read's rectangle is filled by that table read**: of the declarations `inputsOf` returns, at a
unit the table read covers and a row below its port's cut. -/
theorem table_fill {W : Spot.Wires.Wiring} {ts : List Tmpl} {doc : Json} {x : Inputs}
    (h : inputsOf ts W doc = .ok x) {rd : TableRead} (hrd : rd ∈ x.roots) {u row : Nat} (hu : rd.start ≤ u)
    (hu' : u < rd.start + rd.units) (hrow : row < W.cutsOf rd.reader rd.port) :
    fillAt x.decls ⟨rd.reader, u, rd.port, row⟩ = some (.table rd.table rd.root) := by
  obtain ⟨d, hd, hw, hfill⟩ := (inputsOf_declared h).1 rd hrd
  rw [← hfill]
  refine fillAt_of_covers (inputsOf_decls h).1 hd ?_
  rw [hw]
  exact (rect_covers _ _ _ _ _ _).2 ⟨rfl, rfl, hu, hu', hrow⟩

/-- **A row an opening fills is a port of that opening**, of the declarations `inputsOf` returns: the opening is the
one the fill names, its reader is the row's template, the row's port is one of its ports and the row's unit one of
its units. -/
theorem opening_of_fill {W : Spot.Wires.Wiring} {ts : List Tmpl} {doc : Json} {x : Inputs}
    (h : inputsOf ts W doc = .ok x) {r : Row} {i : Nat} (hf : fillAt x.decls r = some (.opening i)) :
    ∃ o, x.openings[i]? = some o ∧ o.reader = r.tid ∧ r.port ∈ o.ports ∧ o.start ≤ r.unit ∧
      r.unit < o.start + o.units := by
  obtain ⟨d, hd, hc, hfill⟩ := fillAt_some hf
  obtain ⟨o, ho, p, hp, hw⟩ := (inputsOf_declared h).2 d hd i hfill
  rw [hw] at hc
  obtain ⟨h1, h2, h3, h4, -⟩ := (rect_covers _ _ _ _ _ _).1 hc
  exact ⟨o, ho, h1, h2 ▸ hp, h3, h4⟩

end NciCheck.Properties.Record.SecurityProofs
