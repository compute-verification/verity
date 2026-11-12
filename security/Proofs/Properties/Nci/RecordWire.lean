module

public import Proofs.Properties.Nci.RecordBlocks
public import Spot.WiresSound

/-!
# The wire of a row

What `Spot.Wires.check` has tested of a wire, and what it gives of a row the wire covers.

* `wire_shape`, `shape_spec`: each wire of a wiring the check accepts passed `Spot.Wires.shape` and
  `Spot.Wires.boundsOk`: its reader and writer are templates of the wiring, its port a row of the reader and its
  output an output of the writer, its rows are among the port's cut, and a row of the cut has the output's words.
* `cut_divides`: a port's cut divides its outer length and its words.
* `writerOf_wire`, `indexOf_some`: a row's writer is given by a wire that covers the row, and what covering says.
* `wire_of_writer`: all of it for a row with a writer, in one statement: the wire, its two templates and its two
  ports, the row within the port's cut, the words of a row of the cut, and both units within their templates.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open NciCheck.Record.SecurityProofs (mem_of_lookup)

/-! ## What `Spot.Wires.check` tests of a wire -/

/-- What `Spot.Wires.shape` has tested of a wire it passes: the reader and the writer are templates of the wiring,
the port is a row of the reader and the output an output of the writer, the wire's rows are among the port's cut,
its cut is whole, and a row of the cut has the output's words. -/
theorem shape_spec {c : Spot.Wires.Wiring} {w : Spot.Wires.Wire} {rt wt : Spot.Wires.Template}
    (h : Spot.Wires.shape c w = .ok (rt, wt)) :
    c.templates.lookup w.reader = some rt ∧ c.templates.lookup w.writer = some wt ∧
      ∃ p ow, rt.rows.lookup w.port = some p ∧ wt.outs.lookup w.out = some ow ∧ 1 ≤ w.cuts ∧
        w.cut0 + w.cuts ≤ c.cutsOf w.reader w.port ∧ (w.cuts = 1 ∨ w.radix.getLast? = some w.cuts) ∧
        w.units % w.cuts = 0 ∧ p.words / c.cutsOf w.reader w.port = ow := by
  unfold Spot.Wires.shape at h
  split at h
  · rename_i rt' wt' hr hw
    split at h
    · rename_i p ow hp how
      dsimp only at h
      split at h
      · cases h
      · rename_i hc
        split at h
        · cases h
        · rename_i hwd
          cases h
          simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq,
            beq_iff_eq] at hc
          exact ⟨hr, hw, p, ow, hp, how, hc.1.1.1, hc.1.1.2, hc.1.2, hc.2, by simpa using hwd⟩
    · cases h
  · cases h

/-- Each wire of a wiring `Spot.Wires.check` accepts passed its shape and its bounds. -/
theorem wire_shape {c : Spot.Wires.Wiring} {S : Spot.Wires.Summary} (h : Spot.Wires.check c = .ok S) :
    ∀ w ∈ c.wires, ∃ rt wt, Spot.Wires.shape c w = .ok (rt, wt) ∧ Spot.Wires.boundsOk w rt wt = true := by
  intro w hw
  have hc := Spot.Wires.check_wireCode h w hw
  unfold Spot.Wires.wireCode at hc
  split at hc
  · cases hc
  · rename_i rt wt hs
    split at hc
    · rename_i hb
      exact ⟨rt, wt, hs, hb⟩
    · cases hc

/-- A port's cut divides its outer length and its words, in a wiring `Spot.Wires.check` accepts. -/
theorem cut_divides {c : Spot.Wires.Wiring} {S : Spot.Wires.Summary} (h : Spot.Wires.check c = .ok S)
    {tid p : String} {t : Spot.Wires.Template} {q : Spot.Wires.Port} (ht : c.templates.lookup tid = some t)
    (hq : t.rows.lookup p = some q) :
    0 < c.cutsOf tid p ∧ q.outer % c.cutsOf tid p = 0 ∧ q.words % c.cutsOf tid p = 0 := by
  have hcut : Spot.Wires.cutsOk c = true := by
    unfold Spot.Wires.check at h
    split at h
    · cases h
    · rename_i hc
      simpa using hc
  unfold Spot.Wires.Wiring.cutsOf
  cases hl : c.cuts.lookup tid with
  | none => simp [Nat.mod_one]
  | some kv =>
    simp only [Option.getD_some]
    cases hk : kv.lookup p with
    | none => simp [Nat.mod_one]
    | some k =>
      simp only [Option.getD_some]
      unfold Spot.Wires.cutsOk at hcut
      have h3 := List.all_eq_true.1 hcut _ (mem_of_lookup _ _ _ hl)
      simp only [ht] at h3
      have h4 := List.all_eq_true.1 h3 _ (mem_of_lookup _ _ _ hk)
      simp only [hq, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h4
      exact ⟨h4.1.1, h4.1.2, h4.2⟩

/-- What a wire that covers a row says of the row. -/
theorem indexOf_some {w : Spot.Wires.Wire} {t : String} {u : Nat} {p : String} {row i : Nat}
    (h : w.indexOf t u p row = some i) :
    w.reader = t ∧ w.port = p ∧ w.start ≤ u ∧ u < w.start + w.readerUnits ∧ w.cut0 ≤ row ∧
      row < w.cut0 + w.cuts ∧ i = (u - w.start) * w.cuts + (row - w.cut0) := by
  unfold Spot.Wires.Wire.indexOf at h
  split at h
  · rename_i hcnd
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hcnd
    obtain ⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩ := hcnd
    exact ⟨h1, h2, h3, h4, h5, h6, (Option.some.inj h).symm⟩
  · cases h

/-- A row's writer is given by a wire of the wiring that covers the row. -/
theorem writerOf_wire {c : Spot.Wires.Wiring} {S : Spot.Wires.Summary} (h : Spot.Wires.check c = .ok S)
    {t : String} {u : Nat} {p : String} {row : Nat} {wt : String} {wu : Int} {out : String}
    (hw : S.writerOf t u p row = some (wt, wu, out)) :
    ∃ w ∈ c.wires, ∃ i, w.indexOf t u p row = some i ∧ wt = w.writer ∧ wu = w.writerAt i ∧ out = w.out := by
  unfold Spot.Wires.Summary.writerOf at hw
  rw [(Spot.Wires.check_sound h).1] at hw
  obtain ⟨w, hwm, hwe⟩ := List.exists_of_findSome?_eq_some hw
  obtain ⟨i, hi, he⟩ := Option.map_eq_some_iff.1 hwe
  simp only [Prod.mk.injEq] at he
  exact ⟨w, hwm, i, hi, he.1.symm, he.2.1.symm, he.2.2.symm⟩

/-- **The wire of a row with a writer**, in a wiring `Spot.Wires.check` accepts: a wire of the wiring covers the row
and names its writer; the wire passed its shape and its bounds; the reader and the writer are templates of the wiring,
the port a row of the reader and the output an output of the writer; the row is within the port's cut, which divides
the port's outer length and its words, and a row of the cut has the output's words; the reader's unit and the writer's
are within their templates. -/
theorem wire_of_writer {c : Spot.Wires.Wiring} {S : Spot.Wires.Summary} (h : Spot.Wires.check c = .ok S)
    {tid : String} {u : Nat} {port : String} {row : Nat} {wt : String} {wu : Int} {out : String}
    (hw : S.writerOf tid u port row = some (wt, wu, out)) :
    ∃ w ∈ c.wires, ∃ rt wtT p ow, w.reader = tid ∧ w.port = port ∧ w.writer = wt ∧ w.out = out ∧
      Spot.Wires.shape c w = .ok (rt, wtT) ∧ Spot.Wires.boundsOk w rt wtT = true ∧
      c.templates.lookup tid = some rt ∧ c.templates.lookup wt = some wtT ∧
      rt.rows.lookup port = some p ∧ wtT.outs.lookup out = some ow ∧
      row < c.cutsOf tid port ∧ p.words / c.cutsOf tid port = ow ∧ 0 < c.cutsOf tid port ∧
      p.outer % c.cutsOf tid port = 0 ∧ p.words % c.cutsOf tid port = 0 ∧ u < rt.units ∧ 0 ≤ wu ∧
      wu.toNat < wtT.units ∧ ∃ i, w.indexOf tid u port row = some i ∧ wu = w.writerAt i := by
  obtain ⟨w, hwm, idx, hidx, rfl, rfl, rfl⟩ := writerOf_wire h hw
  obtain ⟨hrd, hpt, -, hu, -, hrow, -⟩ := indexOf_some hidx
  obtain ⟨rt, wtT, hsh, hb⟩ := wire_shape h w hwm
  obtain ⟨hlr, hlw, p, ow, hlp, hlo, -, hk, -, hd, hwd⟩ := shape_spec hsh
  obtain ⟨hk0, hko, hkw⟩ := cut_divides h hlr hlp
  have hi := Spot.Wires.indexOf_lt hd hidx
  have hs := w.writerAt_mem_span idx hi
  have hb' := hb
  unfold Spot.Wires.boundsOk at hb'
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hb'
  subst hrd hpt
  exact ⟨w, hwm, rt, wtT, p, ow, rfl, rfl, rfl, rfl, hsh, hb, hlr, hlw, hlp, hlo, by omega, hwd, hk0, hko, hkw,
    by omega, by omega, by omega, idx, hidx, rfl⟩

end NciCheck.Properties.Record.SecurityProofs
