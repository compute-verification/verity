module

public import Proofs.Flock.SpotFlock.Facts

/-!
# The wiring's clauses decided once per template give the ones stated bit by bit

The design decides three clauses of the registered wiring once per template, over the rows one unit of the template's
circuit reads its source bits at (`sourceRows`) and the rows it writes (`writtenRows`): `sourceRowsNamed`,
`writtenRowsUnwired`, `writtenRowsRead`. The statements the connected program's proofs read range over every source
bit and every written bit of the members' programs (`sourcesNamed`, `writtenUnwired`, `writtenRead`). Each follows
from its clause, because a bit's row is one of its template's:

* a source bit's row is one its member's circuit's units read at (`sourceBits_row`): the row map (`Bits.rowBit`)
  reads a source gate's row off its input column's source, whatever the unit;
* a written bit's row is one its member's circuit's units write (`writtenBits_row`): a returned output's bit is given
  the name of one of the circuit's output row ports (`Bits.outPos`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-! ## A commitment position's row is one of its circuit's -/

/-- A source's commitment position has the same row at every unit. -/
theorem srcPos_row {c : _root_.Flock.Circuit} {u : ℕ} {s : Spot.Bits.Src} {rp : Spot.Bits.RowPos}
    (hs : Spot.Bits.srcPos c u s = some rp) :
    (Spot.Bits.srcPos c 0 s).map (fun x : Spot.Bits.RowPos => x.row) = some rp.row := by
  have key : ∀ v, (Spot.Bits.srcPos c v s).map (fun x : Spot.Bits.RowPos => x.row) =
      (Spot.Bits.srcPos c u s).map (fun x : Spot.Bits.RowPos => x.row) := by
    intro v
    cases s <;> first | rfl | simp [Spot.Bits.srcPos]
  exact (key 0).trans (congrArg (Option.map fun x : Spot.Bits.RowPos => x.row) hs)

/-- **A source gate's row is one its circuit's units read at.** -/
theorem row_mem_sourceRows {c : _root_.Flock.Circuit} {n g : ℕ} {rp : Spot.Bits.RowPos}
    (hr : Spot.Bits.rowBit c n g = some rp) : rp.row ∈ sourceRows c := by
  unfold Spot.Bits.rowBit at hr
  dsimp only at hr
  split at hr
  · cases hr
  · rename_i hc
    have hB : ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn ≠ 0 := by
      intro h0
      apply hc
      first | simp [h0] | (rw [h0]; simp)
    unfold sourceRows
    try dsimp only
    exact List.mem_filterMap.2 ⟨(g - 2) % ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn,
      List.mem_range.2 (Nat.mod_lt _ (Nat.pos_of_ne_zero hB)), srcPos_row hr⟩

/-- **A returned output bit's row is one its circuit's units write.** -/
theorem row_mem_writtenRows {c : _root_.Flock.Circuit} {u k j t : ℕ} {rp : Spot.Bits.RowPos}
    (hr : Spot.Bits.outRowBit c u k j t = some rp) : rp.row ∈ writtenRows c := by
  unfold Spot.Bits.outRowBit at hr
  obtain ⟨x, -, hx⟩ := Option.bind_eq_some_iff.1 hr
  unfold Spot.Bits.outPos at hx
  dsimp only at hx
  obtain ⟨o, ho, rfl⟩ := Option.map_eq_some_iff.1 hx
  unfold writtenRows
  exact List.mem_map.2 ⟨o, List.mem_of_find?_eq_some ho, rfl⟩

variable (h : Held)

/-- Member `t`'s circuit is its template's reference statement's. -/
theorem memberCircuit_ms {t : ℕ} (ht : t < h.ms.length) : memberCircuit h t = h.ms[t].1 := by
  simp [memberCircuit, Held.ms]

variable {h} in
/-- **A source bit's row is one its member's circuit's units read at**, and its member is one of the members. -/
theorem sourceBits_row {x t : ℕ} {r : Spot.Bits.RowPos} (hs : (x, t, r) ∈ sourceBits h.ms) :
    t < h.ms.length ∧ r.row ∈ sourceRows (memberCircuit h t) := by
  unfold sourceBits at hs
  obtain ⟨m, hm, hx⟩ := List.mem_flatMap.1 hs
  have hm' := List.mem_range.1 hm
  obtain ⟨g, hg, hgx⟩ := List.mem_filterMap.1 hx
  have hci : h.ms.getD m default = h.ms[m] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm', Option.getD_some]
  rw [hci] at hg hgx
  cases hκ : Spot.Bits.consts h.ms[m].1 h.ms[m].2.size g with
  | some b =>
    rw [hκ] at hgx
    simp at hgx
  | none =>
    rw [hκ] at hgx
    have hgx' : (Spot.Bits.rowBit h.ms[m].1 h.ms[m].2.size g).map (fun r => (Spot.Bits.base h.ms m + g, m, r)) =
        some (x, t, r) := hgx
    obtain ⟨r', hr', he⟩ := Option.map_eq_some_iff.1 hgx'
    have h2 : m = t := congrArg (fun p => p.2.1) he
    have h3 : r' = r := congrArg (fun p => p.2.2) he
    subst h2
    subst h3
    refine ⟨hm', ?_⟩
    rw [memberCircuit_ms h hm']
    exact row_mem_sourceRows hr'

variable {h} in
/-- **A written bit's row is one its member's circuit's units write**, and its member is one of the members. -/
theorem writtenBits_row {y t : ℕ} {r : Spot.Bits.RowPos} (hs : (y, t, r) ∈ writtenBits h.ms) :
    t < h.ms.length ∧ r.row ∈ writtenRows (memberCircuit h t) := by
  unfold writtenBits at hs
  obtain ⟨m, hm, hx⟩ := List.mem_flatMap.1 hs
  have hm' := List.mem_range.1 hm
  obtain ⟨u, hu, hx⟩ := List.mem_flatMap.1 hx
  obtain ⟨k, hk, hx⟩ := List.mem_flatMap.1 hx
  obtain ⟨j, hj, hx⟩ := List.mem_flatMap.1 hx
  obtain ⟨b, hb, hx⟩ := List.mem_filterMap.1 hx
  have hci : h.ms.getD m default = h.ms[m] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm', Option.getD_some]
  rw [hci] at hu hk hj hx
  cases hg : Spot.Bits.outGate h.ms[m].1 h.ms[m].2.size u k j b with
  | none =>
    rw [hg] at hx
    cases hr : Spot.Bits.outRowBit h.ms[m].1 u k j b with
    | none =>
      rw [hr] at hx
      simp at hx
    | some r' =>
      rw [hr] at hx
      simp at hx
  | some a =>
    rw [hg] at hx
    cases hr : Spot.Bits.outRowBit h.ms[m].1 u k j b with
    | none =>
      rw [hr] at hx
      simp at hx
    | some r' =>
      rw [hr] at hx
      have he : (Spot.Bits.base h.ms m + Spot.Bits.posOf (Spot.Bits.keyProg h.ms[m].1 h.ms[m].2.size) a, m, r') =
          (y, t, r) := Option.some.inj hx
      have h2 : m = t := congrArg (fun p => p.2.1) he
      have h3 : r' = r := congrArg (fun p => p.2.2) he
      subst h2
      subst h3
      refine ⟨hm', ?_⟩
      rw [memberCircuit_ms h hm']
      exact row_mem_writtenRows hr

/-! ## The clauses stated bit by bit -/

/-- **Every source bit is at a port a link or a wire names** (`sourcesNamed`), where every row a template's units
read source bits at is (`sourceRowsNamed`). -/
theorem sourcesNamed_of_rows (hc : sourceRowsNamed h = true) : sourcesNamed h = true := by
  unfold sourcesNamed
  rw [List.all_eq_true]
  intro s hs
  obtain ⟨ht, hrow⟩ := sourceBits_row (x := s.1) (t := s.2.1) (r := s.2.2) hs
  unfold sourceRowsNamed at hc
  exact List.all_eq_true.1 (List.all_eq_true.1 hc s.2.1 (List.mem_range.2 ht)) s.2.2.row hrow

/-- **No wire names a port a member writes that no link names** (`writtenUnwired`), where none names a row a
template's units write that no link names (`writtenRowsUnwired`). -/
theorem writtenUnwired_of_rows (hc : writtenRowsUnwired h = true) : writtenUnwired h = true := by
  unfold writtenUnwired
  rw [List.all_eq_true]
  intro y hy
  obtain ⟨ht, hrow⟩ := writtenBits_row (y := y.1) (t := y.2.1) (r := y.2.2) hy
  unfold writtenRowsUnwired at hc
  exact List.all_eq_true.1 (List.all_eq_true.1 hc y.2.1 (List.mem_range.2 ht)) y.2.2.row hrow

/-- **Every port a member writes that no link names is some wire's writer port** (`writtenRead`), where every row a
template's units write that no link names is (`writtenRowsRead`). -/
theorem writtenRead_of_rows (hc : writtenRowsRead h = true) : writtenRead h = true := by
  unfold writtenRead
  rw [List.all_eq_true]
  intro y hy
  obtain ⟨ht, hrow⟩ := writtenBits_row (y := y.1) (t := y.2.1) (r := y.2.2) hy
  unfold writtenRowsRead at hc
  exact List.all_eq_true.1 (List.all_eq_true.1 hc y.2.1 (List.mem_range.2 ht)) y.2.2.row hrow

/-- **Every returned output bit of a circuit's unit sits on a computed row**, where the decision holds
(`outsComputed`): its column is past the unit net's input columns and before its constant. -/
theorem outsComputed_spec {c : _root_.Flock.Circuit} (hc : outsComputed c = true) :
    ∀ k < c.upv, ∀ j < (c.leavesOut.getD k #[]).size, ∀ t < 16,
      (Spot.Bits.Rows.ofNet c.unit).nIn ≤ Spot.Bits.outCol c j t ∧
        Spot.Bits.outCol c j t < (Spot.Bits.Rows.ofNet c.unit).nIn + (Spot.Bits.Rows.ofNet c.unit).nComp := by
  intro k hk j hj t ht
  unfold outsComputed at hc
  try dsimp only at hc
  have h1 := List.all_eq_true.1 hc k (List.mem_range.2 hk)
  have h2 := List.all_eq_true.1 h1 j (List.mem_range.2 hj)
  have h3 := List.all_eq_true.1 h2 t (List.mem_range.2 ht)
  rw [Bool.and_eq_true] at h3
  exact ⟨of_decide_eq_true h3.1, of_decide_eq_true h3.2⟩

/-- **Where the design evaluates, every returned output bit of each member's unit sits on a computed row**
(`outsOk`), at the member as the partition lists it. -/
theorem outs_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) {t : ℕ} (ht : t < h.ms.length) :
    ∀ k < h.ms[t].1.upv, ∀ j < (h.ms[t].1.leavesOut.getD k #[]).size, ∀ b < 16,
      (Spot.Bits.Rows.ofNet h.ms[t].1.unit).nIn ≤ Spot.Bits.outCol h.ms[t].1 j b ∧
        Spot.Bits.outCol h.ms[t].1 j b <
          (Spot.Bits.Rows.ofNet h.ms[t].1.unit).nIn + (Spot.Bits.Rows.ofNet h.ms[t].1.unit).nComp := by
  have hc : outsOk h = true := (wiring_of_two h hd).2.2.2.2.2.2.2.2.2.2
  unfold outsOk at hc
  have h1 : outsComputed (memberCircuit h t) = true := List.all_eq_true.1 hc t (List.mem_range.2 ht)
  rw [memberCircuit_ms h ht] at h1
  exact outsComputed_spec h1

/-- **Where the design evaluates, every source bit is at a port a link or a wire names, no wire names a port a
member writes that no link names, and some wire reads it.** -/
theorem namedRows_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) :
    sourcesNamed h = true ∧ writtenUnwired h = true ∧ writtenRead h = true := by
  obtain ⟨-, -, hn, -, -, hu, hr, -, -, -⟩ := wiring_of_two h hd
  exact ⟨sourcesNamed_of_rows h hn, writtenUnwired_of_rows h hu, writtenRead_of_rows h hr⟩

end SpotFlock
