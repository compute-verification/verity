import Proofs.Flock.Soundness.Discharge.Layout.Scope
import Proofs.Flock.Soundness.Discharge.Hidden.ParseCommit
import Proofs.Flock.Soundness.Discharge.Aliased.Rows
import Proofs.Flock.Soundness.SparseRows
import Proofs.Flock.Soundness.ExecParse
import Flock.TmplOrder
import Flock.ProvedScope

/-!
# A templated statement the check passes (PRF-133, ASK-112 option A)

`ProvedScope.check` passes a templated circuit (`Circuit.tmpl`) on its description (`ProvedScope.scopeOkT`), never on the
rows it unrolls to. Its statement circuit, the flat circuit its template unrolls to (`HmOut.unrolled`), is then one the
hidden-output headlines cover at the unit's program (`unrolled_scope`): `HmOut.parseStmt` reads it, it carries no
template, its unit is `Built`, it is in `Layout.scopeOkI`, and the wires feed neither the unit nor the outputs. The unit
reads only port bits among its inputs because the template is ordered (`portReadsB_unroll`); the public file loads at it
as at the committed circuit (`publicOf_unrolled`).
-/

namespace FlockSoundness.Discharge.FailClosed

open Flock Flock.Tmpl

private theorem portSpan_mem {t : Tmpl} {c : ℕ} (h : t.portSpan c (c + 1) = true) : c ∈ portBits t.inGroups := by
  unfold Tmpl.portSpan at h
  rw [List.any_eq_true] at h
  obtain ⟨g, hg, hq⟩ := h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hq
  rw [Discharge.Aliased.portBits_eq]
  refine List.mem_flatMap.2 ⟨g, hg, List.mem_range'_1.2 ⟨hq.1, ?_⟩⟩
  have := hq.2
  rw [← Array.foldl_toList, foldl_add_eq_sum, Nat.zero_add] at this
  omega

/-- **An ordered template's unrolled net reads, among its input columns, only port bits** (`Layout.portReadsB`), checked
on the description (`Tmpl.ordered`): every logical row past the input rows reads input columns only at port bits
(`TmplOrder.rows_good`). -/
theorem portReadsB_unroll {t : Tmpl} {n : Net} (ho : t.ordered = true) (h : Tmpl.unroll t = .ok n)
    (hlog : n.unitLog ≤ 32) : Layout.portReadsB n = true := by
  unfold Tmpl.unroll at h
  obtain ⟨acc, hr, h2⟩ := TmplOrder.bind_ok h
  obtain ⟨-, ha, hb, -, hcp, hig, -, hle, -, -, -, -, -, hw0, -⟩ := Net.ofRows_ok h2
  have hg := TmplOrder.rows_good ho hr
  have hinc : n.inWords * WORD = t.inCols := by rw [hw0]; rfl
  have hw : 2 ^ n.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hlog
  rw [hinc, hcp] at hle
  unfold Layout.portReadsB
  simp only [List.all_eq_true, List.mem_range, List.mem_append, decide_eq_true_eq]
  intro i hi x hx hxn
  rw [hinc, hcp, Nat.min_eq_left hle] at hi hx hxn
  rw [hig]
  rw [SparseRows.ofRows_row ha hw, SparseRows.ofRows_row hb hw] at hx
  have hy1 : t.inCols ≤ t.inCols + i := Nat.le_add_right _ _
  have hy2 : t.inCols + i < t.constPos := by omega
  have hreads : TmplOrder.Reads t.inCols t.constPos (fun c => t.portSpan c (c + 1) = true) (t.inCols + i) x := by
    rcases hx with hx | hx
    · exact (hg _ hy1 hy2).1 x hx
    · exact (hg _ hy1 hy2).2 x hx
  rcases hreads with e | ⟨-, hp⟩
  · omega
  · exact portSpan_mem (hp hxn)

/-- The public file loads at a circuit unrolled as at the circuit: `publicOf` reads neither its nets nor its template. -/
theorem publicOf_unrolled (t : Tags) (c : Circuit) (f : ByteArray) :
    ProvedScope.publicOf t (HmOut.unrolled c) f = ProvedScope.publicOf t c f := by
  unfold HmOut.unrolled
  split
  · split
    · rfl
    · rfl
  · rfl

/-- **A templated circuit the check passes, unrolled, is a statement the hidden-output headlines cover**: the check's
form (hidden outputs, `--zk`, no registered reads of the verifier's own, at most one table) and a templated statement's
tags; `HmOut.parseStmt` reads the statement circuit, which carries no template, has a `Built` unit, is in
`Layout.scopeOkI`, and whose wires feed neither the unit nor the outputs. -/
theorem unrolled_scope {t : Tags} {zk reg : Bool} {count : ℕ} {cf : ByteArray} {tables : List (String × ByteArray)}
    {c : Circuit} {tu : TmplUnit} (hty : t.typed = false) (hc : ProvedScope.circuitOf t cf tables = .ok c)
    (ht : c.tmpl = some tu) (hs : ProvedScope.scopeOkT t.hiddenOutputs zk reg count c tu = true) :
    t.hiddenOutputs = true ∧ zk = true ∧ reg = false ∧ count ≤ 1 ∧ t.isTmpl = true ∧
      HmOut.parseStmt t cf tables = .ok (HmOut.unrolled c) ∧ (HmOut.unrolled c).tmpl = none ∧
      Built (HmOut.unrolled c).unit ∧ Layout.scopeOkI (HmOut.unrolled c) = true ∧
      ProvedScope.wiresFeed (HmOut.unrolled c) = false := by
  unfold ProvedScope.scopeOkT at hs
  simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq, decide_eq_true_eq] at hs
  obtain ⟨⟨⟨⟨⟨⟨⟨hh, hz⟩, hr⟩, hn⟩, hig1⟩, hout⟩, hlog⟩, hrows⟩ := hs
  rw [ProvedScope.circuitOf_hidden hh] at hc
  have hi : t.isTmpl = true := by
    cases hi : t.isTmpl
    · rw [parseCommit_flat hi] at hc
      rw [parse_tmpl_none hty hc] at ht
      cases ht
    · rfl
  rw [parseCommit_tmpl hi] at hc
  obtain ⟨tu', n, ht', hun, hl, -, -, -, he⟩ := unrolled_of_parseTmpl hc
  rw [ht] at ht'
  cases ht'
  have hu : (HmOut.unrolled c).unit = n := (unrolled_unit hc).trans hl
  have hun' := hun
  unfold Tmpl.unroll at hun'
  obtain ⟨acc, -, h2⟩ := TmplOrder.bind_ok hun'
  obtain ⟨hul, -, -, -, -, hig, -⟩ := Net.ofRows_ok h2
  have ho : tu.t.ordered = true := by
    have := tu.ok
    simp only [Tmpl.unrollOk, Bool.and_eq_true] at this
    exact this.1.1.1.1.1.2
  have hlog' : n.unitLog ≤ 32 := by rw [hul]; exact hlog
  have e1 : Layout.v1Ports (HmOut.unrolled c) = ProvedScope.v1Ports c := by rw [he]; rfl
  have e2 : Layout.bitRows (HmOut.unrolled c) = ProvedScope.bitRows c := by rw [he]; rfl
  have e3 : (HmOut.unrolled c).outNet = c.outNet ∧ (HmOut.unrolled c).unitNet = c.unitNet := by
    rw [he]; exact ⟨rfl, rfl⟩
  refine ⟨hh, hz, hr, hn, hi, parseStmt_of_tmpl hi hc, unrolled_tmpl hc, unrolled_built hc, ?_, ?_⟩
  · unfold Layout.scopeOkI
    rw [hu, e1, e2, hrows, portReadsB_unroll ho hun hlog']
    simp [hlog']
  · unfold ProvedScope.wiresFeed
    rw [hu, hig, hig1, e3.1, e3.2, hout]
    simp

end FlockSoundness.Discharge.FailClosed
