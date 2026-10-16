import Proofs.Flock.C3.Reps.ReadRow
import Proofs.Flock.VBridge.OpeningUnit

/-!
# The `rec-residuals` unit reads the columns its products read (`readCol_resUnit`)

`RecResiduals.unitLaid S ns nv` is the unit `RecResiduals.check` and `RecOpening.check` accept for a part: its S, row
ports of `ns` elements and a `v` of `nv`. Every input column a row of its builder reads is one a computed row of the net
reads, under any name; in particular each column a product of two one-term operands reads.

* The builder (`rowReads_partRun`): a residual's product ANDs each position's pair of operand bits (Karatsuba's leaves,
  `mem_leaves`), so where bit `j` of two one-term operands are input bits `c₁ ≠ c₂` (`termCol`: `w`'s on the row ports,
  `v`'s after them) a row reads both (`ResReads`).
* The layout and the net (`laidRes_reads`, `readCol_resUnit`): the builder's `k`-th row is the laid-out row
  `inRows + k`, a computed row of the net at `unitLog ≤ 32` (`readCol_of_laid`).
* An element (`readCol_resUnit_el`): a product of `v`'s element `i` and `w`'s element `e` reads all 128 of `e`'s
  columns.
* At the checks (`readCol_opening`, `readCol_residuals`): the unit `RecOpening.check` or `RecResiduals.check` accepted
  for part `p` reads each column its part's products read.
-/

namespace FlockC3.Reps.RowRead

section Builder

open Flock.HmNets hiding and
open Flock.RecOpen
open FlockSoundness.Discharge.Hm (vars length_vars inRowsOf)
open FlockVBridge.OfVStar (partRep partIn partRun partOuts rowForms rowCol vBase innerIn)

/-- The input bit carrying bit `j` of term `t`'s forms in a part's unit: `w`'s on the row ports (`rowCol`), `v`'s after
them (`vBase`); none past the part's elements. -/
def termCol (ns : List ℕ) (nv : ℕ) : Term → ℕ → Option ℕ
  | .w e, j => if 128 * e + j < 128 * ns.sum then some (rowCol (partRep ns nv).ports 0 (128 * e + j)) else none
  | .wt e u, j =>
    if 128 * (e + j) + u < 128 * ns.sum then some (rowCol (partRep ns nv).ports 0 (128 * (e + j) + u)) else none
  | .v i, j => if 128 * i + j < 128 * nv then some (vBase (partRep ns nv) + (128 * i + j)) else none

/-- **Column `c` is one a product of the part's residuals reads**: a product of two one-term operands whose bits `j` are
input bits `c₁ ≠ c₂`, `c` one of them. -/
def ResReads (S : GfS) (ns : List ℕ) (nv : ℕ) (c : ℕ) : Prop :=
  ∃ r ∈ S.res, ∃ ab ∈ r.mul, ∃ t₁ t₂, S.ops.getD ab.1 [] = [t₁] ∧ S.ops.getD ab.2 [] = [t₂] ∧
    ∃ j < 128, ∃ c₁ c₂, termCol ns nv t₁ j = some c₁ ∧ termCol ns nv t₂ j = some c₂ ∧ c₁ ≠ c₂ ∧ (c = c₁ ∨ c = c₂)

theorem getD_rowForms : ∀ (ps : List (String × ℕ)) (k q : ℕ), q < 128 * (ps.map (·.2)).sum →
    (rowForms ps k).getD q Lin.zero = Lin.var (rowCol ps k q)
  | [], _, _, hq => by simp at hq
  | p :: ps, k, q, hq => by
    have hl : (vars k (128 * p.2)).length = 128 * p.2 := length_vars _ _
    simp only [List.map_cons, List.sum_cons] at hq
    simp only [FlockVBridge.OfVStar.rowForms, FlockVBridge.OfVStar.rowCol]
    by_cases h : q < 128 * p.2
    · simp only [h, ↓reduceIte]
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by rw [hl]; exact h), ← List.getD_eq_getElem?_getD]
      exact getD_inputs _ _ _ h
    · simp only [h, ↓reduceIte]
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by rw [hl]; omega), hl,
        ← List.getD_eq_getElem?_getD]
      exact getD_rowForms ps _ _ (by omega)

theorem getD_termForms_wt (W V : List Lin) (i u : ℕ) {j : ℕ} (hj : j < 128) :
    (termForms W V (.wt i u)).getD j Lin.zero = W.getD (128 * (i + j) + u) Lin.zero := by
  simp [termForms, List.getD_eq_getElem?_getD, hj]

theorem vBase_eq (ns : List ℕ) (nv : ℕ) :
    vBase (partRep ns nv) = 16 * ((partRep ns nv).ports.map fun p => VStar.portWordsOf p.2).sum := rfl

theorem rowCol_lt_vBase {ns : List ℕ} {nv q : ℕ} (hq : q < 128 * ns.sum) :
    rowCol (partRep ns nv).ports 0 q < vBase (partRep ns nv) := by
  have h := FlockVBridge.OfVStar.rowCol_lt (partRep ns nv).ports 0 q (by rw [FlockVBridge.OfVStar.ports_snd]; exact hq)
  rw [vBase_eq]
  omega

theorem vBase_lt_innerIn (ns : List ℕ) {nv q : ℕ} (hq : q < 128 * nv) :
    vBase (partRep ns nv) + q < innerIn (partRep ns nv) := by
  have h := FlockVBridge.OfVStar.innerIn_eq (partRep ns nv)
  have hp := FlockVBridge.OfVStar.le_portWords nv
  have hn : (partRep ns nv).gf.nv = nv := rfl
  rw [hn] at h
  omega

/-- **`termCol` is where the term's forms read**: bit `j` of term `t`'s forms on the part's `w` and `v` is the input bit
`termCol` gives. -/
theorem termCol_spec {ns : List ℕ} {nv : ℕ} {t : Term} {j c : ℕ} (hj : j < 128) (h : termCol ns nv t j = some c) :
    (termForms (rowForms (partRep ns nv).ports 0) (vars (vBase (partRep ns nv)) (128 * nv)) t).getD j Lin.zero =
      Lin.var c ∧ c < innerIn (partRep ns nv) := by
  have hv := vBase_lt_innerIn ns (nv := nv) (q := 0)
  cases t with
  | w e =>
    simp only [termCol] at h
    split at h
    · next hq =>
      obtain rfl := Option.some.inj h
      have hq' : 128 * e + j < 128 * ((partRep ns nv).ports.map (·.2)).sum := by
        rw [FlockVBridge.OfVStar.ports_snd]; exact hq
      have hlt := rowCol_lt_vBase (nv := nv) hq
      refine ⟨by rw [getD_termForms_w _ _ _ hj, getD_rowForms _ _ _ hq'], ?_⟩
      by_cases h0 : 0 < nv
      · have := hv (by omega); omega
      · have := FlockVBridge.OfVStar.innerIn_eq (partRep ns nv); omega
    · cases h
  | wt e u =>
    simp only [termCol] at h
    split at h
    · next hq =>
      obtain rfl := Option.some.inj h
      have hq' : 128 * (e + j) + u < 128 * ((partRep ns nv).ports.map (·.2)).sum := by
        rw [FlockVBridge.OfVStar.ports_snd]; exact hq
      have hlt := rowCol_lt_vBase (nv := nv) hq
      refine ⟨by rw [getD_termForms_wt _ _ _ _ hj, getD_rowForms _ _ _ hq'], ?_⟩
      have := FlockVBridge.OfVStar.innerIn_eq (partRep ns nv); omega
    · cases h
  | v i =>
    simp only [termCol] at h
    split at h
    · next hq =>
      obtain rfl := Option.some.inj h
      exact ⟨by rw [getD_termForms_v _ _ _ hj]; exact getD_inputs _ _ _ hq, vBase_lt_innerIn ns hq⟩
    · cases h

/-- **A residual's product ANDs bit `j` of its operands**: where those are input bits `u₁ ≠ u₂`, a row reads each. -/
theorem reads_residualForms_of {S : GfS} {W V : List Lin} {r : Res} (hr : r ∈ S.res) {ab : ℕ × ℕ} (hab : ab ∈ r.mul)
    {j : ℕ} (hj : j < 128) {u₁ u₂ : Var} (h₁ : (opForms S W V ab.1).getD j Lin.zero = Lin.zero ^^^ Lin.var u₁)
    (h₂ : (opForms S W V ab.2).getD j Lin.zero = Lin.zero ^^^ Lin.var u₂) (hne : u₁ ≠ u₂) {u : Var}
    (hu : u = u₁ ∨ u = u₂) : Reads (residualForms S W V) u := by
  have hxy := mem_leaves 7 (opForms S W V ab.1) (opForms S W V ab.2) j (Nat.lt_of_lt_of_le hj (by decide))
  rw [h₁, h₂] at hxy
  have hs : ∀ v : Var, (Lin.zero ^^^ Lin.var v).s ≠ [] := fun v => by rw [s_zero_xor_var]; exact List.cons_ne_nil _ _
  have hne' : (Lin.zero ^^^ Lin.var u₁).s ≠ (Lin.zero ^^^ Lin.var u₂).s := by
    rw [s_zero_xor_var, s_zero_xor_var]
    exact fun h => hne (List.cons.inj h).1
  have hv : u ∈ (Lin.zero ^^^ Lin.var u₁).s ++ (Lin.zero ^^^ Lin.var u₂).s := by
    rw [s_zero_xor_var, s_zero_xor_var]
    rcases hu with rfl | rfl
    · exact List.mem_append_left _ (List.mem_cons.2 (Or.inl rfl))
    · exact List.mem_append_right _ (List.mem_cons.2 (Or.inl rfl))
  have hr0 := reads_leafSums (p := (opForms S W V ab.1, opForms S W V ab.2)) hxy (hs u₁) (hs u₂) hne' hv
    (r.mul.map fun ab => (opForms S W V ab.1, opForms S W V ab.2)) (List.replicate (3 ^ 7) Lin.zero)
    (List.mem_map.2 ⟨ab, hab, rfl⟩)
  exact reads_residualsGo S W V
    (reads_residual S W V r (by rw [List.isEmpty_iff]; exact List.ne_nil_of_mem hab) hr0) S.res hr

/-- **The part's builder has a row reading each column its products read**, an input bit. -/
theorem rowReads_partRun {S : GfS} {ns : List ℕ} {nv c : ℕ} (h : ResReads S ns nv c) :
    RowReads (partRun S ns nv).2 c ∧ c < innerIn (partRep ns nv) := by
  obtain ⟨r, hr, ab, hab, t₁, t₂, h₁, h₂, j, hj, c₁, c₂, hc₁, hc₂, hne, hc⟩ := h
  obtain ⟨hf₁, hl₁⟩ := termCol_spec hj hc₁
  obtain ⟨hf₂, hl₂⟩ := termCol_spec hj hc₂
  refine ⟨?_, by rcases hc with rfl | rfl <;> assumption⟩
  unfold FlockVBridge.OfVStar.partRun
  rw [← FlockVBridge.residualForms_eq]
  exact reads_residualForms_of hr hab hj (by rw [getD_opForms h₁ hj, hf₁]) (by rw [getD_opForms h₂ hj, hf₂]) hne hc _

/-- **The laid-out unit has a computed row for each input bit a builder row reads**: the `k`-th AND, at row
`inRows + k`, its forms' columns those of their variables. -/
theorem laidRes_reads {S : GfS} {ns : List ℕ} {nv c : ℕ} (hr : RowReads (partRun S ns nv).2 c)
    (hlt : c < innerIn (partRep ns nv)) :
    ∃ k, inRowsOf (partIn ns nv) + k < (Flock.RecResiduals.unitLaid S ns nv).constPos ∧
      c ∈ ((Flock.RecResiduals.unitLaid S ns nv).ra.getD (inRowsOf (partIn ns nv) + k) #[]).toList ++
        ((Flock.RecResiduals.unitLaid S ns nv).rb.getD (inRowsOf (partIn ns nv) + k) #[]).toList := by
  have hst : ((Flock.RecResiduals.unitNet' S ns nv).run St.empty).2 = (partRun S ns nv).2 := by
    rw [FlockVBridge.OfVStar.unitNet'_run]
  have hN : (partIn ns nv).foldl (· + ·) 0 = innerIn (partRep ns nv) := by
    rw [FlockVBridge.OfVStar.partIn, FlockVBridge.OfVStar.foldl_replicate, FlockVBridge.OfVStar.innerIn_partRep]
    exact Nat.mul_comm _ _
  obtain ⟨r, hr, hv⟩ := hr
  rw [← hst] at hr
  obtain ⟨k, hk, hrk⟩ := List.mem_iff_getElem.1 (List.mem_reverse.2 hr)
  obtain ⟨h1, -, h3, -⟩ := layout_rows ((Flock.RecResiduals.unitNet' S ns nv).run St.empty).2 (partIn ns nv)
    (Array.replicate Flock.RecResiduals.RES_OUT_WORDS 16) ((Flock.RecResiduals.unitNet' S ns nv).run St.empty).1.toArray
  have hspan := groupWords_pos ((Array.replicate Flock.RecResiduals.RES_OUT_WORDS 16).foldl (· + ·) 0)
  have hro := le_roundUp (groupWords ((partIn ns nv).foldl (· + ·) 0) * Flock.WORD +
    ((Flock.RecResiduals.unitNet' S ns nv).run St.empty).2.rows.length) _ hspan
  have hk' : k < ((Flock.RecResiduals.unitNet' S ns nv).run St.empty).2.rows.length := by
    rw [← List.length_reverse]; exact hk
  refine ⟨k, ?_, ?_⟩
  · show groupWords ((partIn ns nv).foldl (· + ·) 0) * Flock.WORD + k < (layout _ (partIn ns nv) _ _).constPos
    rw [h1]
    omega
  · obtain ⟨ha, hb⟩ := h3 k hk
    show c ∈ ((layout _ (partIn ns nv) _ _).ra.getD _ #[]).toList ++
      ((layout _ (partIn ns nv) _ _).rb.getD _ #[]).toList
    rw [ha, hb, hrk]
    rcases List.mem_append.1 hv with hv | hv
    · exact List.mem_append_left _ (mem_cols hv (by rw [hN]; exact hlt))
    · exact List.mem_append_right _ (mem_cols hv (by rw [hN]; exact hlt))

end Builder

section Net

open FlockSoundness FlockSoundness.Audit
open FlockVBridge.OfVStar (ReadCol partRep partRun rowCol vBase innerIn)

/-- **The `rec-residuals` unit reads every input column its builder reads**: under any name, a row of the part's
residual computation (`partRun`) reading input bit `c` is a computed row of the net reading column `c`. -/
theorem readCol_resUnit {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {name : String} {net : Flock.Net}
    (hU : Built net) (hn : Flock.HmNets.netOf name (Flock.RecResiduals.unitLaid S ns nv) = .ok net)
    (hu : net.unitLog ≤ 32)
    {c : ℕ} (hr : RowReads (partRun S ns nv).2 c) (hc : c < innerIn (partRep ns nv)) : ReadCol (Rows.ofNet hU) c := by
  obtain ⟨k, hk, hmem⟩ := laidRes_reads hr hc
  rw [← FlockVBridge.OfVStar.inWords_unitLaid hn] at hk hmem
  exact readCol_of_laid hU hn hu hk hmem

/-- **The `rec-residuals` unit reads each column its products read** (`ResReads`). -/
theorem readCol_of_resReads {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {name : String} {net : Flock.Net}
    (hU : Built net) (hn : Flock.HmNets.netOf name (Flock.RecResiduals.unitLaid S ns nv) = .ok net)
    (hu : net.unitLog ≤ 32)
    {c : ℕ} (hr : ResReads S ns nv c) : ReadCol (Rows.ofNet hU) c :=
  readCol_resUnit hU hn hu (rowReads_partRun hr).1 (rowReads_partRun hr).2

/-- **A product of `v`'s element `i` and `w`'s element `e` reads all 128 of `e`'s columns.** -/
theorem readCol_resUnit_el {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {name : String} {net : Flock.Net}
    (hU : Built net) (hn : Flock.HmNets.netOf name (Flock.RecResiduals.unitLaid S ns nv) = .ok net)
    (hu : net.unitLog ≤ 32)
    {r : Flock.RecOpen.Res} (hr : r ∈ S.res) {ab : ℕ × ℕ} (hab : ab ∈ r.mul) {i e : ℕ}
    (h₁ : S.ops.getD ab.1 [] = [.v i]) (h₂ : S.ops.getD ab.2 [] = [.w e]) (hi : i < nv) (he : e < ns.sum) {b : ℕ}
    (hb : b < 128) : ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 (128 * e + b)) := by
  have hq : 128 * e + b < 128 * ns.sum := by omega
  have hlt := rowCol_lt_vBase (nv := nv) hq
  refine readCol_of_resReads hU hn hu ⟨r, hr, ab, hab, .v i, .w e, h₁, h₂, b, hb, vBase (partRep ns nv) + (128 * i + b),
    rowCol (partRep ns nv).ports 0 (128 * e + b), ?_, ?_, by omega, Or.inr rfl⟩
  · simp only [termCol, show 128 * i + b < 128 * nv by omega, ↓reduceIte]
  · simp only [termCol, hq, ↓reduceIte]

/-- **The unit `RecOpening.check` accepted for part `p` reads each column its part's products read.** -/
theorem readCol_opening {c : Flock.Circuit} (hU : Built c.unit) (h : Flock.RecOpening.check c = .ok ())
    (hu : c.unit.unitLog ≤ 32) {m cl p : ℕ} (hp : Flock.RecOpening.params c.unit.name = some (m, cl, p))
    {sh : Flock.Shape} (hsh : Flock.RecOpening.shape? m cl = some sh) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : Flock.RecOpening.part sh p = .ok (S, ns, nv)) {col : ℕ} (hr : ResReads S ns nv col) :
    ReadCol (Rows.ofNet hU) col := by
  obtain ⟨S₁, ns₁, nv₁, hpart₁, hn⟩ := FlockVBridge.OfVStar.check_opening h hp hsh
  rw [hpart] at hpart₁
  cases hpart₁
  exact readCol_of_resReads hU hn hu hr

/-- **The unit `RecResiduals.check` accepted for part `p` reads each column its part's products read.** -/
theorem readCol_residuals {c : Flock.Circuit} (hU : Built c.unit) (h : Flock.RecResiduals.check c = .ok ())
    (hu : c.unit.unitLog ≤ 32) {m k cl p : ℕ} (hp : Flock.RecResiduals.params c.unit.name = some (m, k, cl, p))
    {sh : Flock.Shape} (hsh : Flock.RecResiduals.shape? m k cl = some sh) {holo : Bool}
    (hh : Flock.RecResiduals.holoOf c = .ok holo) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : Flock.RecResiduals.part { sh with holo } p = .ok (S, ns, nv)) {col : ℕ} (hr : ResReads S ns nv col) :
    ReadCol (Rows.ofNet hU) col := by
  obtain ⟨S₁, ns₁, nv₁, hpart₁, hn⟩ := FlockVBridge.OfVStar.check_part_at h hp hsh hh
  rw [hpart] at hpart₁
  cases hpart₁
  exact readCol_of_resReads hU hn hu hr

end Net

end FlockC3.Reps.RowRead
