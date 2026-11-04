import Proofs.Flock.Soundness.TmplUnitAt

/-!
# A hidden-output file's statement circuit

A templated file's statement circuit is the flat circuit its template unrolls to (`HmOut.unrolled`): the circuit
`HmOut.parseTmpl` commits, its unit net `Tmpl.unroll`'s, carrying no template. `HmOut.parseStmt` reads it, and any
other file's circuit as `HmOut.parse` does. A consumer of `HmOut.parseStmt … = .ok c` splits it with `parseStmt_cases`:
the flat file's `HmOut.parse`, or the templated file's committed circuit `ct` with `c = HmOut.unrolled ct`, which is `ct`
with its unit net the logical one (`unrolled_of_parseTmpl`, `unrolled_unit`, `unrolled_built`).
-/

namespace FlockSoundness

open Flock

variable {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c ct : Circuit}

theorem parseStmt_flat (ht : tags.isTmpl = false) :
    HmOut.parseStmt tags cf tables = HmOut.parse tags cf tables := by
  simp [HmOut.parseStmt, ht]

theorem parseStmt_tmpl (ht : tags.isTmpl = true) :
    HmOut.parseStmt tags cf tables = (HmOut.parseTmpl tags cf tables).map HmOut.unrolled := by
  rw [HmOut.parseStmt, if_pos ht]
  cases HmOut.parseTmpl tags cf tables <;> rfl

theorem parseStmt_of_parse (ht : tags.isTmpl = false) (h : HmOut.parse tags cf tables = .ok c) :
    HmOut.parseStmt tags cf tables = .ok c := by
  rw [parseStmt_flat ht]; exact h

theorem parseStmt_of_tmpl (ht : tags.isTmpl = true) (h : HmOut.parseTmpl tags cf tables = .ok ct) :
    HmOut.parseStmt tags cf tables = .ok (HmOut.unrolled ct) := by
  rw [parseStmt_tmpl ht, h]; rfl

/-- **The bridge every consumer of the statement circuit reads**: a flat file's is `HmOut.parse`'s, a templated file's
is the circuit `HmOut.parseTmpl` commits, unrolled. -/
theorem parseStmt_cases (h : HmOut.parseStmt tags cf tables = .ok c) :
    (tags.isTmpl = false ∧ HmOut.parse tags cf tables = .ok c) ∨
      (tags.isTmpl = true ∧ ∃ ct, HmOut.parseTmpl tags cf tables = .ok ct ∧ c = HmOut.unrolled ct) := by
  cases ht : tags.isTmpl
  · exact .inl ⟨rfl, by rwa [parseStmt_flat ht] at h⟩
  · refine .inr ⟨rfl, ?_⟩
    rw [parseStmt_tmpl ht] at h
    cases hp : HmOut.parseTmpl tags cf tables with
    | error e => rw [hp] at h; cases h
    | ok ct =>
      rw [hp] at h
      have e : (Except.ok (HmOut.unrolled ct) : V Circuit) = .ok c := h
      cases e
      exact ⟨ct, rfl, rfl⟩

/-- A flat file's `HmOut.parse` reads the statement circuit (`parseStmt_cases`'s flat side, at tags known flat). -/
theorem parse_of_parseStmt (ht : tags.isTmpl = false) (h : HmOut.parseStmt tags cf tables = .ok c) :
    HmOut.parse tags cf tables = .ok c := by
  rwa [parseStmt_flat ht] at h

theorem unrolled_flat (h : c.tmpl = none) : HmOut.unrolled c = c := by
  unfold HmOut.unrolled
  rw [h]

/-- **A templated file's statement circuit is its committed circuit with the logical unit net**: `ct` with its unit
(net `0`) `Tmpl.unroll`'s net, `lunit ct`, and no template; every other field is `ct`'s. -/
theorem unrolled_of_parseTmpl (h : HmOut.parseTmpl tags cf tables = .ok ct) :
    ∃ (tu : TmplUnit) (n : Net), ct.tmpl = some tu ∧ Tmpl.unroll tu.t = .ok n ∧ lunit ct = n ∧ ct.unitNet = 0 ∧
      ct.unitNet < ct.nets.size ∧ ct.unit = tu.net ∧
      HmOut.unrolled ct = { ct with nets := ct.nets.set! ct.unitNet ("unit", n), tmpl := none } := by
  obtain ⟨tu, c₀, P, -, he, hu0, hn0⟩ := TmplValue.parseTmpl_facts h
  obtain ⟨-, n, -, -, -, hun⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  have ht : ct.tmpl = some tu := by rw [he]
  have hsz : ct.unitNet < ct.nets.size := by rw [he]; exact P.unitNet
  have hu : ct.unit = tu.net := by
    rw [he]
    show (c₀.nets.getD c₀.unitNet default).2 = tu.net
    rw [hu0, hn0]
  refine ⟨tu, n, ht, hun, lunit_tmpl ht hun, by rw [he]; exact hu0, hsz, hu, ?_⟩
  unfold HmOut.unrolled
  rw [ht]
  simp only [hun]

/-- A templated file's statement circuit's unit is the committed circuit's logical net. -/
theorem unrolled_unit (h : HmOut.parseTmpl tags cf tables = .ok ct) : (HmOut.unrolled ct).unit = lunit ct := by
  obtain ⟨tu, n, -, -, hl, -, hsz, -, he⟩ := unrolled_of_parseTmpl h
  rw [he, hl]
  show ((ct.nets.set! ct.unitNet ("unit", n)).getD ct.unitNet default).2 = n
  simp [Array.getD_eq_getD_getElem?, hsz]

theorem unrolled_tmpl (h : HmOut.parseTmpl tags cf tables = .ok ct) : (HmOut.unrolled ct).tmpl = none := by
  obtain ⟨_, _, _, _, _, _, _, _, he⟩ := unrolled_of_parseTmpl h
  rw [he]

/-- **A templated file's statement circuit's unit is `Built`**: `Tmpl.unroll`'s net (`built_tmpl`). -/
theorem unrolled_built (h : HmOut.parseTmpl tags cf tables = .ok ct) : Built (HmOut.unrolled ct).unit := by
  obtain ⟨tu, -, ht, -⟩ := unrolled_of_parseTmpl h
  rw [unrolled_unit h]
  exact built_tmpl ht

end FlockSoundness
