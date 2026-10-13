import Proofs.Flock.C3.Reps.ResRead

/-!
# The checks' whole reads: `RsReadCol` and `ResReadCol`

`RecOpening.check` refuses a `rec-opening` part whose computed rows read an element of a claim's ring switch in fewer
than 128 bits (`rsUnread`), and both it and `RecResiduals.check` a part whose computed rows read a row-port element in
some but not all of them (`partlyRead`), all over `readIns`, the input columns the net's computed rows read.

* `getD_readIns`: `readIns` marks exactly those columns; `readCol_of_row`, `row_of_readCol`: they are the audit's
  `ReadCol` at `unitLog ≤ 32` (`Rows.ofNet_a`, `Rows.ofNet_b`).
* `check_rsWhole`, `check_whole`: what each check accepted, with its refusals passed.
* `readCol_whole`: a part's net that passed `partlyRead` reads a row-port element whole where it reads one of its bits;
  `readCol_openWhole` at the opening's check, and `openReadCol_whole`, what `openingZeroAtL_wReadO` reads, at
  `unitLog ≤ 32`.
* `readCol_rs`: the opening's part reads each column of its ring-switch elements.

This file imports no session, so the sessions read these directly: `RsReadCol` is `rsReadCol` (`RsFacts`, from
`readCol_rsAt`) and `ResReadCol` is `resReadCol` (`ReadWholeAt`).
-/

namespace FlockC3.Reps.ReadWhole

open Flock

/-! ## `readIns` -/

theorem size_foldl_set (xs : List ℕ) (rd : Array Bool) :
    (xs.foldl (fun rd c => rd.setIfInBounds c true) rd).size = rd.size := by
  induction xs generalizing rd with
  | nil => rfl
  | cons x xs ih => rw [List.foldl_cons, ih, Array.size_setIfInBounds]

theorem getD_foldl_set (xs : List ℕ) (rd : Array Bool) (col : ℕ) :
    (xs.foldl (fun rd c => rd.setIfInBounds c true) rd).getD col false = true ↔
      rd.getD col false = true ∨ (col < rd.size ∧ col ∈ xs) := by
  induction xs generalizing rd with
  | nil => simp only [List.foldl_nil, Array.getD_eq_getD_getElem?, List.not_mem_nil, and_false, or_false]
  | cons x xs ih =>
    rw [List.foldl_cons, ih, Array.size_setIfInBounds]
    by_cases hx : x = col
    · subst hx
      by_cases hl : x < rd.size
      · simp only [Array.getD_eq_getD_getElem?, Array.size_setIfInBounds, hl, getElem?_pos,
          Array.getElem_setIfInBounds_self, Option.getD_some, true_and, true_or, List.mem_cons, and_self, or_true]
      · simp only [Array.getD_eq_getD_getElem?, Array.size_setIfInBounds, hl, not_false_eq_true, getElem?_neg,
          Option.getD_none, Bool.false_eq_true, false_and, or_self, List.mem_cons, true_or, and_true]
    · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds_ne hx, List.mem_cons, Ne.symm hx,
        false_or]

theorem size_foldl_rows (g : Net) (L : List ℕ) (rd : Array Bool) :
    (L.foldl (fun rd r => (g.a.row r ++ g.b.row r).foldl (fun rd c => rd.setIfInBounds c true) rd) rd).size =
      rd.size := by
  induction L generalizing rd with
  | nil => rfl
  | cons r L ih => rw [List.foldl_cons, ih, ← Array.foldl_toList, size_foldl_set]

theorem getD_foldl_rows (g : Net) (L : List ℕ) (rd : Array Bool) (col : ℕ) :
    (L.foldl (fun rd r => (g.a.row r ++ g.b.row r).foldl (fun rd c => rd.setIfInBounds c true) rd) rd).getD col
        false = true ↔
      rd.getD col false = true ∨ (col < rd.size ∧ ∃ r ∈ L, col ∈ (g.a.row r).toList ++ (g.b.row r).toList) := by
  induction L generalizing rd with
  | nil => simp only [List.foldl_nil, List.not_mem_nil, false_and, exists_const, and_false, or_false]
  | cons r L ih =>
    rw [List.foldl_cons, ih, ← Array.foldl_toList, getD_foldl_set, size_foldl_set, Array.toList_append]
    constructor
    · rintro ((h | ⟨h1, h2⟩) | ⟨h1, r', h2, h3⟩)
      · exact Or.inl h
      · exact Or.inr ⟨h1, r, List.mem_cons_self, h2⟩
      · exact Or.inr ⟨h1, r', List.mem_cons_of_mem _ h2, h3⟩
    · rintro (h | ⟨h1, r', h2, h3⟩)
      · exact Or.inl (Or.inl h)
      · rcases List.mem_cons.1 h2 with rfl | h2
        · exact Or.inl (Or.inr ⟨h1, h3⟩)
        · exact Or.inr ⟨h1, r', h2, h3⟩

/-- **`readIns` marks exactly the input columns a computed row reads.** -/
theorem getD_readIns (g : Net) (col : ℕ) :
    (RecResiduals.readIns g).getD col false = true ↔
      col < min (g.inWords * WORD) g.constPos ∧ ∃ r, min (g.inWords * WORD) g.constPos ≤ r ∧ r < g.constPos ∧
        col ∈ (g.a.row r).toList ++ (g.b.row r).toList := by
  unfold RecResiduals.readIns
  dsimp only
  rw [getD_foldl_rows]
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_replicate, Array.size_replicate, List.mem_range'_1]
  constructor
  · rintro (h | ⟨h1, r, ⟨h2, h3⟩, h4⟩)
    · split at h <;> simp only [Option.getD_some, Option.getD_none, Bool.false_eq_true] at h
    · exact ⟨h1, r, h2, by omega, h4⟩
  · rintro ⟨h1, r, h2, h3, h4⟩
    exact Or.inr ⟨h1, r, ⟨h2, by omega⟩, h4⟩

/-- **All 128 of element `q`'s columns are read** where `readBits` counts 128. -/
theorem getD_of_readBits {rd : Array Bool} {ns : List ℕ} {q : ℕ} (h : RecResiduals.readBits rd ns q = 128)
    {b : ℕ} (hb : b < 128) : rd.getD (RecResiduals.rowCol ns 0 (128 * q + b)) false = true := by
  unfold RecResiduals.readBits at h
  have h' : ((List.range 128).filter fun b => rd.getD (RecResiduals.rowCol ns 0 (128 * q + b)) false).length =
      (List.range 128).length := by rw [h, List.length_range]
  exact List.length_filter_eq_length_iff.1 h' b (List.mem_range.2 hb)

theorem readBits_pos {rd : Array Bool} {ns : List ℕ} {q b : ℕ} (hb : b < 128)
    (h : rd.getD (RecResiduals.rowCol ns 0 (128 * q + b)) false = true) : RecResiduals.readBits rd ns q ≠ 0 := by
  unfold RecResiduals.readBits
  exact Nat.pos_iff_ne_zero.1 (List.length_pos_of_mem (List.mem_filter.2 ⟨List.mem_range.2 hb, h⟩))

/-- `RecResiduals.rowCol` is the bridge's, at the part's ports (`partRep`'s, unnamed). -/
theorem rowCol_eq : ∀ (ns : List ℕ) (k q : ℕ),
    RecResiduals.rowCol ns k q = FlockVBridge.OfVStar.rowCol (ns.map ("", ·)) k q
  | [], _, _ => rfl
  | n :: ns, k, q => by
    show (if q < 128 * n then k + q else RecResiduals.rowCol ns (k + 16 * RecResiduals.portWordsOf n) (q - 128 * n)) =
      (if q < 128 * n then k + q else
        FlockVBridge.OfVStar.rowCol (ns.map ("", ·)) (k + 16 * VStar.portWordsOf n) (q - 128 * n))
    rw [rowCol_eq ns]
    rfl

/-! ## The refusals -/

/-- **`partlyRead` passes only where each element is read in no bit or in all 128.** -/
theorem readBits_of_partlyRead {rd : Array Bool} {ns : List ℕ} (h : RecResiduals.partlyRead rd ns = none)
    {q : ℕ} (hq : q < ns.sum) : RecResiduals.readBits rd ns q = 0 ∨ RecResiduals.readBits rd ns q = 128 := by
  unfold RecResiduals.partlyRead at h
  have := List.find?_eq_none.1 h (q, RecResiduals.readBits rd ns q) (List.mem_map.2 ⟨q, List.mem_range.2 hq, rfl⟩)
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, not_and, Decidable.not_not] at this
  by_cases h0 : RecResiduals.readBits rd ns q = 0
  · exact Or.inl h0
  · exact Or.inr (this h0)

/-- **`rsUnread` passes only where each ring-switch element of the part's w is read in all 128 bits.** -/
theorem readBits_of_rsUnread {sh : Shape} {W : Array ℕ} {rd : Array Bool} {ns : List ℕ}
    (h : RecOpening.rsUnread sh W rd ns = none) {q : ℕ} (hq : q < W.size) {k u : ℕ}
    (hk : (RecOpening.wNames sh)[W[q]!]? = some (.rs k u)) : RecResiduals.readBits rd ns q = 128 := by
  unfold RecOpening.rsUnread at h
  have := List.filterMap_eq_nil_iff.1 (List.head?_eq_none_iff.1 h) q (List.mem_range.2 hq)
  simp only [hk] at this
  by_cases h128 : RecResiduals.readBits rd ns q = 128
  · exact h128
  · simp only [bne_iff_ne, ne_eq, h128, not_false_eq_true, ↓reduceIte, reduceCtorEq] at this

theorem getElem?_of_wi {sh : Shape} {k : RecAlgebra.Key} {e : ℕ} (h : RecOpening.wi sh k = some e) :
    (RecOpening.wNames sh)[e]? = some k := by
  unfold RecOpening.wi at h
  generalize RecOpening.wNames sh = xs at h ⊢
  rcases xs with ⟨l⟩
  rw [List.idxOf?_toArray, List.idxOf?_eq_some_iff] at h
  obtain ⟨hl, he, -⟩ := h
  rw [List.getElem?_toArray, List.getElem?_eq_getElem hl, he]

/-- **`partW` is `wIdx` of the part's group's row ports.** -/
theorem partW_eq {sh : Shape} {S0 : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : RecOpening.structureOf sh = .ok S0) (hg : RecResiduals.groups S0 (RecOpening.portSizes sh) = .ok gs)
    {p : ℕ} (hp : p < gs.size) {W : Array ℕ} (h : RecOpening.partW sh p = .ok W) :
    W = FlockVBridge.PartS.wIdx (RecOpening.portSizes sh) gs[p].2.1 := by
  unfold RecOpening.partW at h
  obtain ⟨S1, hS1, h⟩ := FlockVBridge.PartS.bind_ok h
  rw [hS] at hS1
  cases hS1
  obtain ⟨gs1, hg1, h⟩ := FlockVBridge.PartS.bind_ok h
  rw [hg] at hg1
  cases hg1
  dsimp only at h
  split at h
  · rename_i R P V hj
    rw [Array.getElem?_eq_getElem hp, Option.some.injEq] at hj
    rw [hj]
    simp only [pure, Except.pure, Except.ok.injEq] at h
    exact h.symm
  · cases h

theorem getElem?_of_wiA {sh : Shape} {k : RecAlgebra.Key} {e : ℕ} (h : RecAlgebra.wi sh k = some e) :
    (RecAlgebra.wNames sh)[e]? = some k := by
  unfold RecAlgebra.wi at h
  generalize RecAlgebra.wNames sh = xs at h ⊢
  rcases xs with ⟨l⟩
  rw [List.idxOf?_toArray, List.idxOf?_eq_some_iff] at h
  obtain ⟨hl, he, -⟩ := h
  rw [List.getElem?_toArray, List.getElem?_eq_getElem hl, he]

/-- **The residuals' `partW` is `wIdx` of the part's group's row ports.** -/
theorem partW_eqR {sh : Shape} {S0 : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : RecAlgebra.structureOf sh = .ok S0) (hg : RecResiduals.groups S0 (RecResiduals.portSizes sh) = .ok gs)
    {p : ℕ} (hp : p < gs.size) {W : Array ℕ} (h : RecResiduals.partW sh p = .ok W) :
    W = FlockVBridge.PartS.wIdx (RecResiduals.portSizes sh) gs[p].2.1 := by
  unfold RecResiduals.partW at h
  obtain ⟨S1, hS1, h⟩ := FlockVBridge.PartS.bind_ok h
  rw [hS] at hS1
  cases hS1
  obtain ⟨gs1, hg1, h⟩ := FlockVBridge.PartS.bind_ok h
  rw [hg] at hg1
  cases hg1
  dsimp only at h
  split at h
  · rename_i R P V hj
    rw [Array.getElem?_eq_getElem hp, Option.some.injEq] at hj
    rw [hj]
    simp only [pure, Except.pure, Except.ok.injEq] at h
    exact h.symm
  · cases h

/-- **What `RecOpening.check` accepted reads its ring-switch elements whole, every other element whole or not at
all, and keys each ring switch's bits by its claim's port**: the verifier's net for part `j`, whose computed rows read
each element of a claim's ring switch at `partW`'s slots in all 128 bits, each row-port element in all 128 or in none,
and whose bits of claim `i`'s ring-switch element are keyed by ports named `m<i>`. -/
theorem check_rsWhole {c : Circuit} (h : RecOpening.check c = .ok ()) {m cl j : ℕ}
    (hp : RecOpening.params c.unit.name = some (m, cl, j)) {sh : Shape} (hsh : RecOpening.shape? m cl = some sh) :
    ∃ S ns nv W, RecOpening.part sh j = .ok (S, ns, nv) ∧
      HmNets.netOf (RecOpening.unitName m cl j) (RecResiduals.unitLaid S ns nv) = .ok c.unit ∧
      RecOpening.partW sh j = .ok W ∧ RecOpening.rsUnread sh W (RecResiduals.readIns c.unit) ns = none ∧
      RecResiduals.partlyRead (RecResiduals.readIns c.unit) ns = none ∧
      RecResiduals.misnamed c ns W.size (RecOpening.rsName sh W) = none := by
  unfold RecOpening.check at h
  rw [hp] at h
  dsimp only at h
  rw [hsh] at h
  dsimp only at h
  split at h
  · cases h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i S ns nv hpart
      split at h
      · cases h
      · split at h
        · cases h
        · rename_i g hg
          split at h
          · rename_i hcg
            subst hcg
            split at h
            · cases h
            · rename_i W hW
              split at h
              · rename_i hn
                split at h
                · rename_i hw
                  split at h
                  · rename_i hmis
                    exact ⟨S, ns, nv, W, hpart, hg, hW, hn, hw, hmis⟩
                  · cases h
                · cases h
              · cases h
          · cases h

/-- **What `RecResiduals.check` accepted reads its row-port elements whole or not at all, keyed by their rounds'
ports**: the verifier's net for part `j`, whose computed rows read each row-port element in all 128 bits or in none,
and whose bits of an element of w a round holds are keyed by ports named for each round holding it. -/
theorem check_whole {c : Circuit} (h : RecResiduals.check c = .ok ()) {m k cl j : ℕ}
    (hp : RecResiduals.params c.unit.name = some (m, k, cl, j)) {sh : Shape}
    (hsh : RecResiduals.shape? m k cl = some sh) {holo : Bool} (hh : RecResiduals.holoOf c = .ok holo) :
    ∃ S ns nv, RecResiduals.part { sh with holo } j = .ok (S, ns, nv) ∧
      HmNets.netOf (RecResiduals.unitName m k cl j) (RecResiduals.unitLaid S ns nv) = .ok c.unit ∧
      RecResiduals.partlyRead (RecResiduals.readIns c.unit) ns = none ∧ ∃ W,
        RecResiduals.partW { sh with holo } j = .ok W ∧
        RecResiduals.misnamed c ns W.size (fun a => RecResiduals.roundNames (RecAlgebra.rounds { sh with holo })
          (RecAlgebra.wNames { sh with holo })[W[a]!]?) = none := by
  unfold RecResiduals.check at h
  rw [hp] at h
  dsimp only at h
  rw [hsh] at h
  dsimp only at h
  split at h
  · cases h
  · rw [hh] at h
    dsimp only at h
    split at h
    · cases h
    · rename_i S ns nv hpart
      split at h
      · cases h
      · split at h
        · cases h
        · rename_i g hg
          split at h
          · rename_i hcg
            subst hcg
            split at h
            · rename_i hn
              split at h
              · cases h
              · rename_i W hW
                split at h
                · rename_i hmis
                  exact ⟨S, ns, nv, hpart, hg, hn, W, hW, hmis⟩
                · cases h
            · cases h
          · cases h

theorem part_ok {sh : Shape} {p : ℕ} {x : RecOpen.GfS × List ℕ × ℕ} (h : RecResiduals.part sh p = .ok x) :
    ∃ S0 gs, RecAlgebra.structureOf sh = .ok S0 ∧ RecResiduals.groups S0 (RecResiduals.portSizes sh) = .ok gs := by
  unfold RecResiduals.part at h
  obtain ⟨S0, hS0, h⟩ := FlockVBridge.PartS.bind_ok h
  dsimp only at h
  split at h
  · cases h
  · obtain ⟨gs, hg, -⟩ := FlockVBridge.PartS.bind_ok h
    exact ⟨S0, gs, hS0, hg⟩

/-! ## The audit's reads -/

section Net

open FlockSoundness FlockSoundness.Audit
open FlockVBridge.OfVStar (ReadCol rowCol partRep)

/-- **A net row from the inputs' end to the constant is a computed row** of the audit's rows, reading its columns. -/
theorem readCol_of_row {net : Net} (hU : Built net) (hu : net.unitLog ≤ 32) {r col : ℕ}
    (hr : min (net.inWords * WORD) net.constPos ≤ r) (hr' : r < net.constPos)
    (hc : col ∈ (net.a.row r).toList ++ (net.b.row r).toList) : ReadCol (Rows.ofNet hU) col := by
  have h1 : (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp = net.constPos := Rows.ofNet_one hU
  have hn : (Rows.ofNet hU).nIn = min (net.inWords * WORD) net.constPos := Rows.ofNet_nIn hU
  have hj : r - (Rows.ofNet hU).nIn < (Rows.ofNet hU).nComp := by omega
  have hij : (Rows.ofNet hU).nIn + (⟨r - (Rows.ofNet hU).nIn, hj⟩ : Fin _).val = r := by simp only; omega
  have ha := Rows.ofNet_a hU hu ⟨r - (Rows.ofNet hU).nIn, hj⟩
  have hb := Rows.ofNet_b hU hu ⟨r - (Rows.ofNet hU).nIn, hj⟩
  rw [hij] at ha hb
  rw [← ha, ← hb, ← List.map_append] at hc
  obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hc
  exact ⟨⟨_, hj⟩, x, hx, rfl⟩

/-- **A column a computed row reads is in a net row** from the inputs' end to the constant. -/
theorem row_of_readCol {net : Net} (hU : Built net) (hu : net.unitLog ≤ 32) {col : ℕ}
    (h : ReadCol (Rows.ofNet hU) col) :
    ∃ r, min (net.inWords * WORD) net.constPos ≤ r ∧ r < net.constPos ∧
      col ∈ (net.a.row r).toList ++ (net.b.row r).toList := by
  obtain ⟨i, x, hx, rfl⟩ := h
  have h1 : (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp = net.constPos := Rows.ofNet_one hU
  have hn : (Rows.ofNet hU).nIn = min (net.inWords * WORD) net.constPos := Rows.ofNet_nIn hU
  have hi := i.isLt
  refine ⟨(Rows.ofNet hU).nIn + i.val, by omega, by omega, ?_⟩
  rw [← Rows.ofNet_a hU hu i, ← Rows.ofNet_b hU hu i, ← List.map_append]
  exact List.mem_map.2 ⟨x, hx, rfl⟩

/-- **The unit `RecOpening.check` accepted for part `p` reads each column of its ring-switch elements**: bit `b` of the
part's row-port element `q`, slot `e` of the opening's w holding ring switch `i`'s element `t`. -/
theorem readCol_rs {c : Flock.Circuit} (hU : Built c.unit) (h : RecOpening.check c = .ok ()) (hu : c.unit.unitLog ≤ 32)
    {m cl p : ℕ} (hp : RecOpening.params c.unit.name = some (m, cl, p)) {sh : Flock.Shape}
    (hsh : RecOpening.shape? m cl = some sh) {S0 : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : RecOpening.structureOf sh = .ok S0) (hg : RecResiduals.groups S0 (RecOpening.portSizes sh) = .ok gs)
    (hpg : p < gs.size) {S₁ : RecOpen.GfS} {ns : List ℕ} {nv : ℕ} (hpart : RecOpening.part sh p = .ok (S₁, ns, nv))
    {q e : ℕ} (he : (FlockVBridge.PartS.wIdx (RecOpening.portSizes sh) gs[p].2.1)[q]? = some e) {i t : ℕ}
    (hwi : RecOpening.wi sh (.rs i t) = some e) {b : ℕ} (hb : b < 128) :
    ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 (128 * q + b)) := by
  obtain ⟨S₂, ns₂, nv₂, W, hpart₂, -, hW, hn, -⟩ := check_rsWhole h hp hsh
  obtain ⟨rfl, rfl, rfl⟩ : S₁ = S₂ ∧ ns = ns₂ ∧ nv = nv₂ := by
    simpa only [Prod.mk.injEq] using Except.ok.inj (hpart.symm.trans hpart₂)
  obtain rfl := partW_eq hS hg hpg hW
  obtain ⟨hq, hqe⟩ := Array.getElem?_eq_some_iff.1 he
  have hk : (RecOpening.wNames sh)[(FlockVBridge.PartS.wIdx (RecOpening.portSizes sh) gs[p].2.1)[q]!]? =
      some (.rs i t) := by
    rw [getElem!_pos (FlockVBridge.PartS.wIdx (RecOpening.portSizes sh) gs[p].2.1) q hq, hqe]
    exact getElem?_of_wi hwi
  obtain ⟨-, r, hr, hr', hc⟩ := (getD_readIns c.unit _).1 (getD_of_readBits (readBits_of_rsUnread hn hq hk) hb)
  rw [rowCol_eq] at hc
  exact readCol_of_row hU hu hr hr' hc

/-- **The unit `RecResiduals.check` accepted reads a row-port element whole where it reads one of its bits.** -/
theorem readCol_whole {c : Flock.Circuit} (hU : Built c.unit) (hu : c.unit.unitLog ≤ 32) {S : RecOpen.GfS}
    {ns : List ℕ} {nv : ℕ} {name : String} (hn : HmNets.netOf name (RecResiduals.unitLaid S ns nv) = .ok c.unit)
    (hw : RecResiduals.partlyRead (RecResiduals.readIns c.unit) ns = none) {q b : ℕ} (hq : q < ns.sum)
    (hb : b < 128) (hr : ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 (128 * q + b))) {b₀ : ℕ}
    (hb₀ : b₀ < 128) : ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 (128 * q + b₀)) := by
  have hlt := FlockVBridge.OfVStar.rowCol_lt_nIn hU hn (q := 128 * q + b) (by omega)
  have hn' : (Rows.ofNet hU).nIn = min (c.unit.inWords * WORD) c.unit.constPos := Rows.ofNet_nIn hU
  obtain ⟨r, hr1, hr2, hmem⟩ := row_of_readCol hU hu hr
  have hset : (RecResiduals.readIns c.unit).getD (RecResiduals.rowCol ns 0 (128 * q + b)) false = true := by
    rw [getD_readIns, rowCol_eq]
    exact ⟨by rw [← hn']; exact hlt, r, hr1, hr2, hmem⟩
  rcases readBits_of_partlyRead hw hq with h0 | h128
  · exact absurd h0 (readBits_pos hb hset)
  · obtain ⟨-, r', hr1', hr2', hmem'⟩ := (getD_readIns c.unit _).1 (getD_of_readBits h128 hb₀)
    rw [rowCol_eq] at hmem'
    exact readCol_of_row hU hu hr1' hr2' hmem'

/-- **The unit `RecOpening.check` accepted for part `p` reads a row-port element whole where it reads one of its bits**
(`readCol_whole` at the part's net, whose `partlyRead` the check passed). -/
theorem readCol_openWhole {c : Flock.Circuit} (hU : Built c.unit) (h : RecOpening.check c = .ok ())
    (hu : c.unit.unitLog ≤ 32) {m cl p : ℕ} (hp : RecOpening.params c.unit.name = some (m, cl, p)) {sh : Flock.Shape}
    (hsh : RecOpening.shape? m cl = some sh) {S₁ : RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecOpening.part sh p = .ok (S₁, ns, nv)) {q b : ℕ} (hq : q < ns.sum) (hb : b < 128)
    (hr : ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 (128 * q + b))) {b₀ : ℕ} (hb₀ : b₀ < 128) :
    ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 (128 * q + b₀)) := by
  obtain ⟨S₂, ns₂, nv₂, -, hpart₂, hn, -, -, hw, -⟩ := check_rsWhole h hp hsh
  obtain ⟨rfl, rfl, rfl⟩ : S₁ = S₂ ∧ ns = ns₂ ∧ nv = nv₂ := by
    simpa only [Prod.mk.injEq] using Except.ok.inj (hpart.symm.trans hpart₂)
  exact readCol_whole hU hu hn hw hq hb hr hb₀

/-- **An element a `rec-opening` part holds is read whole or not at all** (what `openingZeroAtL_wReadO` reads, with the
unit's `unitLog ≤ 32` that `OpeningZeroAtL` carries): `readCol_openWhole` at each circuit. -/
theorem openReadCol_whole (sh : Flock.Shape) :
    ∀ (c : Flock.Circuit) (hU : Built c.unit) (m cl p : ℕ), Flock.RecOpening.check c = .ok () →
      c.unit.unitLog ≤ 32 → Flock.RecOpening.params c.unit.name = some (m, cl, p) →
      Flock.RecOpening.shape? m cl = some sh →
      ∀ (S₁ : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ), Flock.RecOpening.part sh p = .ok (S₁, ns, nv) →
      ∀ a < ns.sum, ∀ b < 128, ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 (128 * a + b)) →
        ∀ b₀ < 128, ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 (128 * a + b₀)) :=
  fun _ hU _ _ _ hc hu hp hsh _ _ _ hpart _ hq _ hb hr _ hb₀ => readCol_openWhole hU hc hu hp hsh hpart hq hb hr hb₀

end Net

end FlockC3.Reps.ReadWhole
