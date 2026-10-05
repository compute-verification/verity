import FlockSoundness.Discharge.ZkLink.OracleP
import FlockSoundness.CompiledAlign
import FlockSoundness.Protocol.Discharge.ZkLink.LockLevel

/-!
# The compiled masked level 0 in lockstep with the oracle one

`CompiledLock`'s lockstep with level 0's rows `w₀` lanes wide (`AgreeZ`: `AgreeRep` with level 0's lanes `< w₀`, the
witness lanes and both reps' extra lanes of one tree). `lockRecZ` is `lockRec` over `AgreeZ`; `lockLigPadZ` runs M1's
masked level 0 (`ligeritoPadZK`) beside its compiled form (`ligeritoPadZC`), whose level-0 consistency value is computed
from the opened rows (`padColumnO_eq`). The folds and the OOD samples are `CompiledLock`'s (`lockFold`, `lockTwoOod`),
the masked lane rounds and `ρ` the same game on both sides (`lockFoldZK`, `Lock.same`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-! ## Agreement with level 0's wide rows -/

omit [Fintype F] [DecidableEq F] in
theorem AgreeZ.of_append {r : ℕ} {C₀ : Oracle F} {w₀ : ℕ} {i : IRep F D} {reqs reqs' : List Req}
    {o : Opens F D} (h : AgreeZ r C₀ w₀ i (reqs ++ reqs') o) : AgreeZ r C₀ w₀ i reqs o :=
  fun req hr => h req (List.mem_append_left _ hr)

omit [Fintype F] [DecidableEq F] in
theorem AgreeZ.last {r : ℕ} {C₀ : Oracle F} {w₀ : ℕ} {i : IRep F D} {reqs : List Req} {req : Req}
    {o : Opens F D} (h : AgreeZ r C₀ w₀ i (reqs ++ [req]) o) {t : Oracle F} (ht : tabAt C₀ i req.level = some t)
    (j : ℕ) (hj : j < req.positions.length) (lane : ℕ) (hl : lane < if req.level = 0 then w₀ else 2 ^ req.lanes) :
    (o r req.level j).1.getD lane 0 = t lane (req.positions.getD j 0) :=
  h req (List.mem_append_right _ (List.mem_singleton_self _)) t ht j hj lane hl

theorem queryRel_leafZ (A : Arith F K) (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (i : IRep F D) (lv : Level)
    (ch : List K) (start : ℕ) (st : LigState F K) (stC : LigStateC F K D) (h : StRelZ r C₀ w₀ i st stC)
    (hlev : stC.level ≠ 0) (w : Fin lv.queries → F) (α : Fin (ceilLog2 lv.queries) → F) (β : F) (qO : Quad K)
    (qC : Opens F D → Quad K)
    (hq : ∀ o, AgreeZ r C₀ w₀ i (stC.reqs ++ [⟨stC.level, positions A lv.queries lv.logLen w,
        ch.length, pathLen lv⟩]) o → qC o = qO) :
    QueryRelZ r C₀ w₀ lv ch.length stC i i
      { st with
        s := (st.s + A.embed β *
          consValue A (List.ofFn α) ch st.prev (positions A lv.queries lv.logLen w)),
        q := qO,
        cons := (st.cons ++ [⟨lv.logCols, positions A lv.queries lv.logLen w, List.ofFn α, β,
          start⟩]) }
      { stC with
        s := (fun o => stC.s o + A.embed β * consValueO A (List.ofFn α) ch
          (fun j => (o r stC.level j).1) (positions A lv.queries lv.logLen w)),
        q := qC,
        cons := (stC.cons ++ [⟨lv.logCols, positions A lv.queries lv.logLen w, List.ofFn α, β,
          start⟩]),
        reqs := (stC.reqs ++ [⟨stC.level, positions A lv.queries lv.logLen w, ch.length,
          pathLen lv⟩]) } := by
  refine ⟨rfl, ⟨h.levels, h.oods, by simp [h.cons], h.prevLevel, h.prev, fun o ho => ⟨?_, hq o ho⟩⟩,
    rfl, rfl, rfl, _, rfl, positions_length A _ _ w, positions_lt A _ _ w⟩
  simp only
  rw [(h.sq o ho.of_append).1]
  congr 2
  refine consValueO_eq A _ ch st.prev _ _ fun j hj lane hl => ?_
  exact AgreeZ.last ho h.prev j hj lane (by simpa [hlev] using hl)

/-! ## The recursive rounds -/

omit [Field K] [Fintype K] [DecidableEq K] in
theorem StRelZ.update {r : ℕ} {C₀ : Oracle F} {w₀ : ℕ} {i : IRep F D} {st : LigState F K}
    {stC : LigStateC F K D} (h : StRelZ r C₀ w₀ i st stC) (ℓ : ℕ) (x : Option (Oracle F × Cap D))
    (hℓ : stC.level < ℓ) (hreq : ∀ req ∈ stC.reqs, req.level ≤ stC.level) :
    StRelZ r C₀ w₀ (Function.update i ℓ x) st stC where
  levels := h.levels
  oods := h.oods
  cons := h.cons
  prevLevel := h.prevLevel
  prev := by rw [tabAt_update C₀ i ℓ _ x (by omega)]; exact h.prev
  sq o ho := h.sq o fun req hr t ht => ho req hr t (by
    rw [tabAt_update C₀ i ℓ _ x (by have := hreq req hr; omega)]; exact ht)

omit [Fintype F] [DecidableEq F] in
theorem RecRelZ.reject (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (sch : Schedule) (i : IRep F D) :
    RecRelZ r C₀ w₀ sch i false (PendL.reject : PendL F D) :=
  ⟨fun _ _ => rfl, rfl, by simp [PendL.reject], fun ℓ _ h => by simp [PendL.reject] at h,
    by simp [PendL.reject]⟩

/-! ## Level 0 -/

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- On a row agreeing with `C` at `p` on its `w₀` lanes, the padded column from the row is the oracle's. -/
theorem padColumnO_eq (A : Arith F K) (d cols t : ℕ) (c₀ : F) (ch : List K) {e : ℕ} (ρ : Fin e → K)
    (xl : Fin e → ℕ) (ybar : Fin t → K) (row : List F) (C : Oracle F) (p w₀ : ℕ)
    (h : ∀ lane < w₀, row.getD lane 0 = C lane p) (hch : 2 ^ ch.length ≤ w₀) (hxl : ∀ i, xl i < w₀) :
    padColumnO A d cols t c₀ ch ρ xl ybar row p = padColumn A d cols t c₀ ch ρ xl ybar C p := by
  unfold padColumnO padColumn foldColumn
  congr 2
  · exact Finset.sum_congr rfl fun lane hl => by
      dsimp only
      rw [h lane ((Finset.mem_range.1 hl).trans_le hch)]
  · exact Finset.sum_congr rfl fun i _ => by rw [h _ (hxl i)]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- On agreeing rows, the batched value of a column function of the rows is the oracle column's. -/
theorem consValuePad_eq (A : Arith F K) (α : List F) (col : List F → ℕ → K) (v : ℕ → K) (rows : ℕ → List F)
    (pos : List ℕ) (h : ∀ j < pos.length, col (rows j) (pos.getD j 0) = v (pos.getD j 0)) :
    ((pos.zipIdx).map fun pj => A.embed (eqList α pj.2) * col (rows pj.2) pj.1).sum = consValueV A α v pos := by
  unfold consValueV
  refine congrArg List.sum (List.map_congr_left fun pj hpj => ?_)
  obtain ⟨p, j⟩ := pj
  have hj : j < pos.length ∧ pos.getD j 0 = p := by
    rw [List.mem_iff_getElem] at hpj
    obtain ⟨n, hn, hpn⟩ := hpj
    rw [List.getElem_zipIdx] at hpn
    simp only [Prod.mk.injEq, zero_add] at hpn
    obtain ⟨h1, h2⟩ := hpn
    subst h2
    simp only [List.length_zipIdx] at hn
    exact ⟨hn, by rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hn]; simp [h1]⟩
  simp only
  rw [← hj.2, h j hj.1]

/-- Rep `r`'s extra lanes lie in level 0's rows. -/
theorem extraLane_lt (k0 e r : ℕ) (hr : r < 2) (i : Fin e) : extraLane k0 e r i < 2 ^ k0 + 2 * e := by
  unfold extraLane
  have : e * r ≤ e := by
    calc e * r ≤ e * 1 := Nat.mul_le_mul_left e (by omega)
      _ = e := mul_one e
  have := i.isLt
  rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl <;> simp at * <;> omega

end FlockSoundness.Discharge.ZkLink
