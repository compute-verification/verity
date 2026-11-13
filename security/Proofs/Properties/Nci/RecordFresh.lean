module

public import Proofs.Properties.Nci.RecordBlockGates
public import Proofs.Properties.Nci.RecordFreshRows
public import Proofs.Properties.Nci.RecordKinds
public import Proofs.Properties.Nci.RecordSlots
public import Proofs.Properties.Nci.RecordTally
public import Mathlib.Tactic

/-!
# The fresh rows of a slot are within the summary's fresh

`fresh_rows : FreshRows`: on a record `Record.read` accepts, the rows of one slot whose kind is neither `inside` nor
`theta` are, between them, no more bits than the summary's fresh bits (`Record.Summary.fresh`).

The summary's fresh is at least a block's input bits less the slot's `inside` and `theta` tallies (`fresh_ge`), so it
is enough that the fresh rows and the two tallies fit together in a block's input bits. Three facts give that:

* **The rows of a slot** (`slotList`: each template's units of the slot, each row of the template, each cut of the
  row) are, with their bits, no more than a block's input bits (`slotList_bits`): a slot holds a template's units
  over the slots (`tmpl_slot_units`), and a port's cuts times a row's bits is at most the port's bits.
* **A declaration's tally is its rows' bits.** The rows of the slot a declaration's wire covers (`slotRows`) number
  what the tally counts (`slotRows_card`, by `inSlot_count`), they are rows of the slot, and each has at least the
  declaration's bits (`decl_fits`). No row is covered twice (`inputsOf_decls`).
* **A row covered by an `inside` or `theta` declaration has that kind** (`fill_of_cover`), so it is none of the
  fresh rows.

`covers_le` is the count: a set of rows and disjoint covers that miss it, each cover weighted by no more than its
rows' bits, are together within the bits of all the rows.
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open RecordText
open Lean (Json)
open NciCheck.Properties.Record.SecurityProofs (inputsOf_decls wired_of_read wire_shape shape_spec cover_apart
  fill_of_cover tmpls_named tmpl_grids tmpl_slot_units)

/-! ## Sums -/

/-- A sum over the members of a list is no more than the sum along the list. -/
private theorem sum_toFinset_le {ρ : Type} [DecidableEq ρ] (b : ρ → ℕ) :
    ∀ l : List ρ, ∑ x ∈ l.toFinset, b x ≤ (l.map b).sum
  | [] => by simp
  | a :: l => by
    rw [List.toFinset_cons, List.map_cons, List.sum_cons]
    by_cases ha : a ∈ l.toFinset
    · rw [Finset.insert_eq_of_mem ha]
      exact (sum_toFinset_le b l).trans (Nat.le_add_left _ _)
    · rw [Finset.sum_insert ha]
      exact Nat.add_le_add_left (sum_toFinset_le b l) _

/-- A sum along a list of lists is bounded list by list. -/
private theorem sum_flatMap_le {α β : Type} (b : β → ℕ) (f : α → List β) (g : α → ℕ) :
    ∀ l : List α, (∀ a ∈ l, ((f a).map b).sum ≤ g a) → ((l.flatMap f).map b).sum ≤ (l.map g).sum
  | [], _ => by simp
  | a :: l, h => by
    rw [List.flatMap_cons, List.map_append, List.sum_append, List.map_cons, List.sum_cons]
    exact Nat.add_le_add (h a List.mem_cons_self)
      (sum_flatMap_le b f g l fun x hx => h x (List.mem_cons_of_mem _ hx))

/-- The sums over the members numbered 0 and over those numbered 1 are one sum over the list. -/
private theorem sum_two {α : Type} (k f : α → ℕ) : ∀ l : List α,
    ((l.filter fun a => k a == 0).map f).sum + ((l.filter fun a => k a == 1).map f).sum =
      (l.map fun a => if k a = 0 ∨ k a = 1 then f a else 0).sum
  | [] => by simp
  | a :: l => by
    have ih := sum_two k f l
    rw [List.filter_cons, List.filter_cons, List.map_cons, List.sum_cons]
    by_cases h0 : k a = 0
    · simp only [h0, beq_self_eq_true, ite_true, List.map_cons, List.sum_cons, true_or]
      simp only [show ((0 : ℕ) == 1) = false from rfl, Bool.false_eq_true, ite_false]
      omega
    · by_cases h1 : k a = 1
      · simp only [h1, beq_self_eq_true, ite_true, List.map_cons, List.sum_cons, or_true]
        simp only [show ((1 : ℕ) == 0) = false from rfl, Bool.false_eq_true, ite_false]
        omega
      · simp only [beq_iff_eq, h0, h1, ite_false, or_self]
        omega

/-- **Disjoint covers that miss a set fit beside it.** `F` is a set of rows within `R`, and each `d` of `ds` covers
some rows `cov d`, no row twice. Where each cover of weight not zero lies in `R`, misses `F`, and weighs no more than
the bits of each row it covers, the bits of `F` and the covers' weights are together within the bits of `R`. -/
private theorem covers_le {ρ δ : Type} [DecidableEq ρ] (b : ρ → ℕ) (R : Finset ρ) (cov : δ → Finset ρ)
    (wt : δ → ℕ) : ∀ (ds : List δ) (F : Finset ρ), F ⊆ R → ds.Pairwise (fun a b => Disjoint (cov a) (cov b)) →
      (∀ d ∈ ds, wt d ≠ 0 → cov d ⊆ R ∧ Disjoint F (cov d) ∧ ∀ r ∈ cov d, wt d ≤ b r) →
      ∑ x ∈ F, b x + (ds.map fun d => (cov d).card * wt d).sum ≤ ∑ x ∈ R, b x
  | [], F, hF, _, _ => by simpa using Finset.sum_le_sum_of_subset hF
  | d :: ds, F, hF, hp, h => by
    rw [List.pairwise_cons] at hp
    rw [List.map_cons, List.sum_cons]
    by_cases hw : wt d = 0
    · rw [hw, Nat.mul_zero, Nat.zero_add]
      exact covers_le b R cov wt ds F hF hp.2 fun d' hd' => h d' (List.mem_cons_of_mem _ hd')
    · obtain ⟨hR, hdis, hle⟩ := h d List.mem_cons_self hw
      have ih := covers_le b R cov wt ds (F ∪ cov d) (Finset.union_subset hF hR) hp.2 fun d' hd' hw' => by
        obtain ⟨hR', hdis', hle'⟩ := h d' (List.mem_cons_of_mem _ hd') hw'
        exact ⟨hR', Finset.disjoint_union_left.2 ⟨hdis', hp.1 d' hd'⟩, hle'⟩
      rw [Finset.sum_union hdis] at ih
      have hc : (cov d).card * wt d ≤ ∑ x ∈ cov d, b x := by
        calc (cov d).card * wt d = ∑ _x ∈ cov d, wt d := by rw [Finset.sum_const, smul_eq_mul]
          _ ≤ ∑ x ∈ cov d, b x := Finset.sum_le_sum hle
      omega

/-! ## The rows of a slot a wire covers -/

/-- **The rows of slot `s` a wire covers**, each as reader, unit, port and cut: the wire's units whose slot in the
grids `gs` is `s`, and the wire's cuts. -/
def slotRows (S : ℕ) (gs : List Grid) (s : ℕ) (w : Spot.Wires.Wire) : Finset (String × ℕ × String × ℕ) :=
  (((List.range' w.start w.readerUnits).filter fun u => slotOf S gs u == some s).toFinset ×ˢ
    Finset.Ico w.cut0 (w.cut0 + w.cuts)).image fun p => (w.reader, p.1, w.port, p.2)

/-- A row of `slotRows` is a row of the wire's reader and port, in the slot, below the wire's last cut, and one the
wire covers. -/
theorem slotRows_mem {S : ℕ} {gs : List Grid} {s : ℕ} {w : Spot.Wires.Wire} {x : String × ℕ × String × ℕ}
    (h : x ∈ slotRows S gs s w) :
    x.1 = w.reader ∧ x.2.2.1 = w.port ∧ slotOf S gs x.2.1 = some s ∧ x.2.2.2 < w.cut0 + w.cuts ∧
      (w.indexOf x.1 x.2.1 x.2.2.1 x.2.2.2).isSome = true := by
  obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 h
  rw [Finset.mem_product, List.mem_toFinset, List.mem_filter, List.mem_range'_1, Finset.mem_Ico] at hp
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4, h5⟩ := hp
  refine ⟨rfl, rfl, by simpa using h3, h5, ?_⟩
  simp only [Spot.Wires.Wire.indexOf, beq_self_eq_true, Bool.true_and, decide_eq_true h1, decide_eq_true h2,
    decide_eq_true h4, decide_eq_true h5, ite_true, Option.isSome_some]

/-- **The rows of a slot a wire covers number what the tally counts**: the wire's units in the slot
(`Record.inSlot`) times its cuts. -/
theorem slotRows_card (S : ℕ) (gs : List Grid) (s : ℕ) (w : Spot.Wires.Wire) (hg : OneGrid gs)
    (hq : ∀ g ∈ gs, 0 < g.q) (hs : s < S) :
    (slotRows S gs s w).card = inSlot S gs w.start (w.start + w.readerUnits) s * w.cuts := by
  rw [inSlot_count S gs _ _ s hg hq hs, Nat.add_sub_cancel_left]
  unfold slotRows
  rw [Finset.card_image_of_injective _ (fun p q h => by
      simp only [Prod.mk.injEq, true_and] at h
      exact Prod.ext h.1 h.2),
    Finset.card_product, List.toFinset_card_of_nodup ((List.nodup_range' ..).filter _), Nat.card_Ico,
    Nat.add_sub_cancel_left]

/-- **No row is covered by two disjoint wires.** -/
theorem slotRows_apart {S : ℕ} {s : ℕ} (gs gs' : List Grid) {a b : Spot.Wires.Wire}
    (h : Spot.Wires.disjoint a b = true) : Disjoint (slotRows S gs s a) (slotRows S gs' s b) :=
  Finset.disjoint_left.2 fun _ ha hb => cover_apart h (slotRows_mem ha).2.2.2.2 (slotRows_mem hb).2.2.2.2

/-! ## The rows of a slot -/

/-- **The rows of slot `s`**, each as reader, unit, port and cut: each template's units of the slot, each row of the
template, each cut of the row's port. -/
def slotList (ts : List Tmpl) (W : Spot.Wires.Wiring) (S : ℕ) (per : List (String × List Grid)) (s : ℕ) :
    List (String × ℕ × String × ℕ) :=
  ts.flatMap fun t => ((List.range t.units).filter fun u => slotIn S per t.tid u == some s).flatMap fun u =>
    t.rows.flatMap fun row => (List.range (W.cutsOf t.tid row.1)).map fun i => (t.tid, u, row.1, i)

/-- A cut of a row of a template's unit of the slot is in `slotList`. -/
theorem slotList_mem {ts : List Tmpl} {W : Spot.Wires.Wiring} {S : ℕ} {per : List (String × List Grid)} {s : ℕ}
    {t : Tmpl} (ht : t ∈ ts) {u : ℕ} (hu : u < t.units) (hs : slotIn S per t.tid u = some s)
    {row : String × ℕ × ℕ} (hrow : row ∈ t.rows) {i : ℕ} (hi : i < W.cutsOf t.tid row.1) :
    (t.tid, u, row.1, i) ∈ slotList ts W S per s :=
  List.mem_flatMap.2 ⟨t, ht, List.mem_flatMap.2 ⟨u, List.mem_filter.2 ⟨List.mem_range.2 hu, by simp [hs]⟩,
    List.mem_flatMap.2 ⟨row, hrow, List.mem_map.2 ⟨i, List.mem_range.2 hi, rfl⟩⟩⟩⟩

/-- **The rows of a slot are, with their bits, no more than a block's input bits**: each template's units over the
slots times its rows' 16-bit words, as the summary counts them. -/
theorem slotList_bits {ts : List Tmpl} {doc : Json} {S : ℕ} {per : List (String × List Grid)}
    (h : gridsOf ts doc = .ok (S, per)) (hd : (ts.map (·.tid)).Nodup) (hn : ∀ t ∈ ts, (t.rows.map (·.1)).Nodup)
    (W : Spot.Wires.Wiring) {s : ℕ} (hs : s < S) :
    ((slotList ts W S per s).map fun x => rowBits ts W x.1 x.2.2.1).sum ≤
      (ts.map fun t => t.units / S * (t.rows.map fun (_, words, _) => 16 * words).sum).sum := by
  refine sum_flatMap_le _ _ _ ts fun t ht => ?_
  rw [← tmpl_slot_units h hd ht hs]
  refine (sum_flatMap_le _ _ (fun _ => (t.rows.map fun (_, words, _) => 16 * words).sum) _ fun u _ => ?_).trans ?_
  · refine sum_flatMap_le _ _ _ t.rows fun row hrow => ?_
    have hrb : ((fun x : String × ℕ × String × ℕ => rowBits ts W x.1 x.2.2.1) ∘ fun i => (t.tid, u, row.1, i)) =
        fun _ => 16 * row.2.1 / W.cutsOf t.tid row.1 := funext fun _ => rowBits_eq W hd ht (hn t ht) hrow
    rw [List.map_map, hrb, List.map_const', List.sum_replicate, List.length_range, smul_eq_mul]
    obtain ⟨name, words, off⟩ := row
    exact Nat.mul_div_le _ _
  · rw [List.map_const', List.sum_replicate, smul_eq_mul]

/-- **A unit with a slot is one of its template's units.** -/
theorem slot_in_tmpl {ts : List Tmpl} {doc : Json} {S : ℕ} {per : List (String × List Grid)}
    (h : gridsOf ts doc = .ok (S, per)) (hd : (ts.map (·.tid)).Nodup) {t : Tmpl} (ht : t ∈ ts) {u s : ℕ}
    (hu : slotIn S per t.tid u = some s) : u < t.units := by
  obtain ⟨g, hg, -⟩ := Option.map_eq_some_iff.1 hu
  have hin := (tmpl_grids h hd ht).2.2.2.1 g (List.mem_of_find?_eq_some hg)
  have hp := List.find?_some hg
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
  omega

/-! ## A declaration's bits -/

/-- A looked-up row is the first row of its name. -/
private theorem find?_of_lookup {β : Type} : ∀ (l : List (String × β)) (a : String) (b : β), l.lookup a = some b →
    ∃ row, l.find? (·.1 == a) = some row ∧ row.2 = b
  | [], _, _, h => by cases h
  | (k, v) :: l, a, b, h => by
    rw [List.lookup_cons] at h
    rw [List.find?_cons]
    by_cases hk : k = a
    · subst hk
      simp only [beq_self_eq_true] at h ⊢
      exact ⟨(k, v), rfl, Option.some.inj h⟩
    · have h1 : (a == k) = false := beq_eq_false_iff_ne.2 fun e => hk e.symm
      have h2 : (k == a) = false := beq_eq_false_iff_ne.2 hk
      simp only [h1] at h
      simp only [h2]
      exact find?_of_lookup l a b h

/-- **A wire with words reads a row of its reader**, and its words are the row's over the port's cut. -/
theorem rowWords_row {ts : List Tmpl} {W : Spot.Wires.Wiring} {w : Spot.Wires.Wire} (h : rowWords ts W w ≠ 0) :
    ∃ t row, ts.find? (·.tid == w.reader) = some t ∧ t.rows.find? (·.1 == w.port) = some row ∧
      rowWords ts W w = row.2.1 / W.cutsOf w.reader w.port := by
  unfold rowWords at h ⊢
  cases ht : ts.find? (·.tid == w.reader) with
  | none =>
    rw [ht] at h
    exact absurd rfl h
  | some t =>
    rw [ht] at h
    simp only [Option.bind_some] at h ⊢
    cases hl : t.rows.lookup w.port with
    | none =>
      rw [hl] at h
      exact absurd rfl h
    | some v =>
      obtain ⟨row, hrow, hv⟩ := find?_of_lookup _ _ _ hl
      refine ⟨t, row, rfl, hrow, ?_⟩
      rw [← hv]

/-- **A declaration with bits fits its rows**, on a record the check reads: its reader is a template and its port a
row of it, its bits are no more than a row's (`rowBits`), and its cuts are cuts of the port. -/
theorem decl_fits {j tr : Json} {c : Case} (h : read j tr = .ok c) {d : Decl} (hd : d ∈ c.x.decls)
    (hb : d.bits ≠ 0) :
    ∃ t row, c.f.ts.find? (·.tid == d.w.reader) = some t ∧ t.rows.find? (·.1 == d.w.port) = some row ∧
      d.bits ≤ 16 * row.2.1 / c.W.cutsOf d.w.reader d.w.port ∧
      d.w.cut0 + d.w.cuts ≤ c.W.cutsOf d.w.reader d.w.port := by
  obtain ⟨f, pin, -, hfin⟩ := read_front h
  obtain ⟨hcf, -, text, -, hx, hcuts, hwires⟩ := finish_text hfin
  subst hcf
  have hcut : ∀ tid p, text.cutsOf tid p = c.W.cutsOf tid p := fun tid p => by
    simp only [Spot.Wires.Wiring.cutsOf, hcuts]
  obtain ⟨-, rest, hdecls, hrest⟩ := inputsOf_decls hx
  rw [hdecls] at hd
  rcases List.mem_append.1 hd with hw | hr
  · obtain ⟨⟨w, i⟩, hwi, rfl⟩ := List.mem_map.1 hw
    have hwm : w ∈ c.W.wires := hwires ▸ List.mem_of_getElem? (List.mem_zipIdx_iff_getElem?.1 hwi)
    obtain ⟨t, row, ht, hrow, hrw⟩ := rowWords_row (ts := c.f.ts) (W := text) (w := w) fun h0 =>
      hb (show 16 * rowWords c.f.ts text w = 0 by rw [h0])
    obtain ⟨rt, wt, hsh, -⟩ := wire_shape (wired_of_read h) w hwm
    obtain ⟨-, -, p, ow, -, -, -, hc, -⟩ := shape_spec hsh
    refine ⟨t, row, ht, hrow, ?_, hc⟩
    show 16 * rowWords c.f.ts text w ≤ _
    rw [hrw, hcut]
    exact Nat.mul_div_le_mul_div_assoc _ _ _
  · obtain ⟨-, t, ht, -, hc, hrow⟩ := hrest d hr
    rcases hrow with ⟨-, h0⟩ | ⟨row, hrow, hbits⟩
    · exact absurd h0 hb
    · exact ⟨t, row, ht, hrow, by rw [hbits, hcut], by rw [← hcut]; exact hc⟩

/-! ## The fresh rows -/

/-- **The fresh rows of a slot are no more bits than the summary's fresh**, on a record the check reads. -/
theorem fresh_rows : FreshRows := by
  intro j tr c h s hs F hF
  obtain ⟨f, pin, hfr, hfin⟩ := read_front h
  obtain ⟨hcf, hgr, text, -, hx, -, -⟩ := finish_text hfin
  have hts := front_text hfr
  subst hcf
  obtain ⟨hd, hnames⟩ := tmpls_named hts
  have hn : ∀ t ∈ c.f.ts, (t.rows.map (·.1)).Nodup := fun t ht => (List.nodup_append.1 (hnames t ht)).1
  obtain ⟨hdisj, -⟩ := inputsOf_decls hx
  have hpw : c.x.decls.Pairwise fun a b => Spot.Wires.disjoint a.w b.w = true :=
    List.pairwise_map.1 ((Spot.Wires.pairwiseDisjoint_iff _).1 hdisj)
  have hone : ∀ tid, OneGrid (c.gridsAt tid) := fun tid => (gridsOf_lookup hgr tid).1.imp fun hab => Or.inl hab
  have hq : ∀ tid, ∀ g ∈ c.gridsAt tid, 0 < g.q := fun tid => (gridsOf_lookup hgr tid).2
  have hkind : ∀ k : Kind, k.index = 0 ∨ k.index = 1 → k ≠ Kind.inside → k ≠ Kind.theta → False := by
    intro k
    cases k <;> simp [Kind.index]
  let b : String × ℕ × String × ℕ → ℕ := fun x => rowBits c.f.ts c.W x.1 x.2.2.1
  let R : Finset (String × ℕ × String × ℕ) := (slotList c.f.ts c.W c.S c.grids s).toFinset
  let cov : Decl → Finset (String × ℕ × String × ℕ) := fun d => slotRows c.S (c.gridsAt d.w.reader) s d.w
  let wt : Decl → ℕ := fun d =>
    if (c.kind (some d.fill)).index = 0 ∨ (c.kind (some d.fill)).index = 1 then d.bits else 0
  have hFR : F ⊆ R := fun x hx => by
    obtain ⟨⟨t, ht, htid, row, hrow, hp⟩, hcut, hslot, -, -⟩ := hF x hx
    obtain ⟨tid, u, p, i⟩ := x
    simp only at htid hp hcut hslot
    subst htid hp
    exact List.mem_toFinset.2 (slotList_mem ht (slot_in_tmpl hgr hd ht hslot) hslot hrow hcut)
  have hcov : ∀ d ∈ c.x.decls, wt d ≠ 0 → cov d ⊆ R ∧ Disjoint F (cov d) ∧ ∀ r ∈ cov d, wt d ≤ b r := by
    intro d hdm hw
    have hk : (c.kind (some d.fill)).index = 0 ∨ (c.kind (some d.fill)).index = 1 := by
      by_contra hk
      exact hw (ite_eq_right hk)
    have hwt : wt d = d.bits := ite_eq_left hk
    obtain ⟨t, row, hft, hfrow, hbits, hcuts⟩ := decl_fits h hdm (hwt ▸ hw)
    have htm : t ∈ c.f.ts := List.mem_of_find?_eq_some hft
    have htid : t.tid = d.w.reader := by simpa using List.find?_some hft
    have hrm : row ∈ t.rows := List.mem_of_find?_eq_some hfrow
    have hrn : row.1 = d.w.port := by simpa using List.find?_some hfrow
    refine ⟨fun x hx => ?_, Finset.disjoint_left.2 fun x hxF hx => ?_, fun x hx => ?_⟩
    · obtain ⟨h1, h2, h3, h4, -⟩ := slotRows_mem hx
      obtain ⟨tid, u, p, i⟩ := x
      simp only at h1 h2 h3 h4
      subst h1 h2
      have hslot : slotIn c.S c.grids t.tid u = some s := by rw [htid]; exact h3
      have hm := slotList_mem (W := c.W) htm (slot_in_tmpl hgr hd htm hslot) hslot hrm (i := i)
        (by rw [htid, hrn]; omega)
      rw [htid, hrn] at hm
      exact List.mem_toFinset.2 hm
    · obtain ⟨-, -, -, -, hidx⟩ := slotRows_mem hx
      have hfill := fill_of_cover hdisj hdm (r := ⟨x.1, x.2.1, x.2.2.1, x.2.2.2⟩) hidx
      obtain ⟨-, -, -, hni, hnt⟩ := hF x hxF
      have hrk : c.rowKind ⟨x.1, x.2.1, x.2.2.1, x.2.2.2⟩ = c.kind (some d.fill) := by
        show c.kind (fillAt c.x.decls _) = _
        rw [hfill]
      rw [hrk] at hni hnt
      exact hkind _ hk hni hnt
    · obtain ⟨h1, h2, -⟩ := slotRows_mem hx
      rw [hwt]
      show d.bits ≤ rowBits c.f.ts c.W x.1 x.2.2.1
      rw [h1, h2]
      unfold rowBits
      rw [hft]
      simp only [Option.bind_some, hfrow, Option.map_some, Option.getD_some]
      exact hbits
  have key := covers_le b R cov wt c.x.decls F hFR (hpw.imp fun hab => slotRows_apart _ _ hab) hcov
  have hR : ∑ x ∈ R, b x ≤ (summarize c).inputs :=
    (sum_toFinset_le b _).trans (slotList_bits hgr hd hn c.W hs)
  have hT : (tally c.S (c.x.decls.map fun d => (c.kind (some d.fill), c.gridsAt d.w.reader, d)) s)[Kind.inside.index]! +
      (tally c.S (c.x.decls.map fun d => (c.kind (some d.fill), c.gridsAt d.w.reader, d)) s)[Kind.theta.index]! =
        (c.x.decls.map fun d => (cov d).card * wt d).sum := by
    rw [tally_eq _ _ _ _ (kind_index_lt _), tally_eq _ _ _ _ (kind_index_lt _), List.filter_map, List.filter_map,
      List.map_map, List.map_map]
    refine (sum_two (fun d : Decl => (c.kind (some d.fill)).index) _ c.x.decls).trans ?_
    refine congrArg List.sum (List.map_congr_left fun d _ => ?_)
    show (if _ then inSlot c.S (c.gridsAt d.w.reader) d.w.start (d.w.start + d.w.readerUnits) s * d.w.cuts * d.bits
      else 0) = (slotRows c.S (c.gridsAt d.w.reader) s d.w).card * wt d
    rw [slotRows_card _ _ _ _ (hone _) (hq _) hs]
    by_cases hk : (c.kind (some d.fill)).index = 0 ∨ (c.kind (some d.fill)).index = 1
    · rw [ite_eq_left hk, show wt d = d.bits from ite_eq_left hk]
    · rw [ite_eq_right hk, show wt d = 0 from ite_eq_right hk, Nat.mul_zero]
  have hge := fresh_ge c s hs
  show ∑ x ∈ F, b x ≤ (summarize c).fresh
  omega

end NciCheck.Record.SecurityProofs
