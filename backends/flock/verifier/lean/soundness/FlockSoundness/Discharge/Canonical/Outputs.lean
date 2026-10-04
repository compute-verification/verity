import FlockSoundness.Discharge.Canonical.Commit
import FlockSoundness.Discharge.Canonical.OutCopy
import FlockSoundness.Discharge.HiddenExec.KeyH

set_option Elab.async false

/-!
# Output rows of runs of output ports, exact length

`HiddenExec.outBitRow` puts output word `w` in output row `nIn + jo`, one per output port (`outPortOf`), at bit
`16 (w - outAt jo) + t` of a row of `16 · words` bits zero-padded to whole blocks. A canonical output row (`PROTOCOL.md`
§16.16) is one per run of output ports (META `out_row_ports`, `HmOut.outGroup c j = (first, count, start, words)`), of
exactly the run's `16 · words` bits. So output word `w` is in run `outRunOf c w` (the first whose words end past `w`), at
bit `16 (w - start) + t`.

* `outRunOf`, `outBitRowC`: that bit, as a row bit of `keyProg` (`Fin (rowW c q)`); `outBitRowC_spec` reads it back;
* `outBitRowC_ne_none`: every output word's bit is a canonical output row's;
* `out_copyC`: `HiddenExec.out_copy_key` at runs, without `outNet = unitNet` and `sha512/row/v1` rows.

When a tail's last stage returns the outputs (`outNet ≠ unitNet`), `HmRow.parse` requires every `leavesOut` empty, so no
unit column is a returned output (`Layout.OutColAt` fails) and `out_copyC` holds vacuously: `keyProg` (the unit net's rows
alone) does not compute a tail's outputs. The statement-level `OutputsCommittedC` (`Statements.lean`) covers them.

Every declaration here is proved. `outGroup_next` makes the runs consecutive (`start` is a prefix sum of `outPorts`'
words), and `HmRow.parse`'s `out_row_ports` check makes them cover the output ports (`hsum`), so `outRunOf_spec` finds the
run of every output word; `formatOk`'s exact rows give `16 (w - start) + t < 16 · words = ℓ`. `out_copyC` is
`out_copyHC` (`OutCopy.lean`: `out_copyH`'s route at runs) at that run, and `check_tail` at a tail.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3 Model Placed
open FlockSoundness.Discharge.Layout (keyN OutColAt)
open FlockSoundness.Discharge.HiddenExec (AcceptedH siteAddrH outWordOf)

/-- **The run holding output word `w`**: the first output row `j` whose run's words end past `w`. -/
def outRunOf (c : Circuit) (w : ℕ) : ℕ :=
  ((List.range c.outRowPorts.size).find? fun j =>
    decide (w < (HmOut.outGroup c j).2.2.1 + (HmOut.outGroup c j).2.2.2)).getD 0

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

variable (c) in
/-- **A canonical output row's bit**: copy `d.1`'s output `d.2.1`, bit `d.2.2`, of unit `u` is bit `16 (w - start) +
d.2.2` of `u`'s output row `nIn + j`, where the copy's output word `w` is in run `j` (`outRunOf`) from word `start`;
none past the row. -/
def outBitRowC (u : Fin n) (d : ℕ × ℕ × ℕ) : Option (Fin n × (q : Fin c.ports.size) × Fin (rowW c q)) :=
  if hq : HmOut.nIn c + outRunOf c (outWordOf c d) < c.ports.size then
    if hk : 16 * (outWordOf c d - (HmOut.outGroup c (outRunOf c (outWordOf c d))).2.2.1) + d.2.2 <
        rowW c ⟨_, hq⟩ then
      some (u, ⟨⟨_, hq⟩, ⟨_, hk⟩⟩)
    else none
  else none

theorem outBitRowC_spec {u u' : Fin n} {d : ℕ × ℕ × ℕ} {q : Fin c.ports.size} {k : Fin (rowW c q)}
    (h : outBitRowC c u d = some (u', ⟨q, k⟩)) :
    u' = u ∧ q.val = HmOut.nIn c + outRunOf c (outWordOf c d) ∧
      k.val = 16 * (outWordOf c d - (HmOut.outGroup c (outRunOf c (outWordOf c d))).2.2.1) + d.2.2 := by
  unfold outBitRowC at h
  split at h
  · split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨hu, hs⟩ := h
      subst hu
      cases hs
      exact ⟨rfl, rfl, rfl⟩
    · exact absurd h (by simp)
  · exact absurd h (by simp)

theorem foldl_add_sum {xs : List ℕ} : ∀ a, xs.foldl (· + ·) a = a + xs.sum := by
  induction xs with
  | nil => intro a; rfl
  | cons x xs ih => intro a; rw [List.foldl_cons, ih, List.sum_cons]; omega

/-- `HmOut.outGroup`'s `start` and `words` as sums: the output words of the ports before run `j`'s first, and of its
ports. -/
theorem outGroup_sums (c : Circuit) (j : ℕ) :
    (HmOut.outGroup c j).2.2.1 = ((c.outPorts.toList.take (c.outRowPorts.toList.take j).sum).map (·.2)).sum ∧
      (HmOut.outGroup c j).2.2.2 = (((c.outPorts.toList.drop (c.outRowPorts.toList.take j).sum).take
        (c.outRowPorts.getD j 0)).map (·.2)).sum := by
  simp only [HmOut.outGroup, ← Array.foldl_toList, Array.toList_extract, List.extract_eq_take_drop, List.drop_zero,
    Nat.sub_zero, Nat.add_sub_cancel_left, Hidden.foldl_snd_eq, foldl_add_sum, Nat.zero_add]
  constructor <;> trivial

/-- **Runs are consecutive**: run `j + 1` starts where run `j` ends. -/
theorem outGroup_next {c : Circuit} {j : ℕ} (hj : j < c.outRowPorts.size) :
    (HmOut.outGroup c j).2.2.1 + (HmOut.outGroup c j).2.2.2 = (HmOut.outGroup c (j + 1)).2.2.1 := by
  obtain ⟨e1, e2⟩ := outGroup_sums c j
  have hl : j < c.outRowPorts.toList.length := by simpa using hj
  have hg : c.outRowPorts.getD j 0 = c.outRowPorts.toList[j] := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some, Array.getElem_toList]
  rw [e1, e2, (outGroup_sums c (j + 1)).1, List.take_succ_eq_append_getElem hl, List.sum_append, List.sum_singleton,
    ← hg, List.take_add, List.map_append, List.sum_append]

/-- **The run holding an output word**: when the runs cover the output ports (`HmRow.parse`'s `out_row_ports`
check), output word `w` is in run `outRunOf c w`, from its `start`. -/
theorem outRunOf_spec {c : Circuit} (hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size) {w : ℕ}
    (hw : w < c.outWords) :
    outRunOf c w < c.outRowPorts.size ∧ (HmOut.outGroup c (outRunOf c w)).2.2.1 ≤ w ∧
      w < (HmOut.outGroup c (outRunOf c w)).2.2.1 + (HmOut.outGroup c (outRunOf c w)).2.2.2 := by
  have h0 : (HmOut.outGroup c 0).2.2.1 = 0 := by simp [(outGroup_sums c 0).1]
  have hN : (HmOut.outGroup c c.outRowPorts.size).2.2.1 = c.outWords := by
    have hs : (c.outRowPorts.toList.take c.outRowPorts.size).sum = c.outPorts.size := by
      rw [← Array.length_toList, List.take_length, ← hsum, ← Array.foldl_toList, foldl_add_sum, Nat.zero_add]
    rw [(outGroup_sums c _).1, hs, ← Array.length_toList, List.take_length, Circuit.outWords, ← Array.foldl_toList,
      Hidden.foldl_snd_eq, Nat.zero_add]
  unfold outRunOf
  rcases hf : (List.range c.outRowPorts.size).find?
      (fun j => decide (w < (HmOut.outGroup c j).2.2.1 + (HmOut.outGroup c j).2.2.2)) with _ | i
  · exfalso
    have hN0 : 0 < c.outRowPorts.size := by
      rcases Nat.eq_zero_or_pos c.outRowPorts.size with h | h
      · rw [h, h0] at hN; omega
      · exact h
    have := List.find?_eq_none.1 hf (c.outRowPorts.size - 1) (List.mem_range.2 (by omega))
    simp only [decide_eq_true_eq] at this
    rw [outGroup_next (by omega), Nat.sub_add_cancel hN0, hN] at this
    omega
  · obtain ⟨hp, hi, hmin⟩ := List.find?_range_eq_some.1 hf
    have hi' := List.mem_range.1 hi
    simp only [decide_eq_true_eq] at hp
    simp only [Option.getD_some]
    refine ⟨hi', ?_, hp⟩
    rcases Nat.eq_zero_or_pos i with rfl | hpos
    · rw [h0]; exact Nat.zero_le _
    · have := hmin (i - 1) (by omega)
      simp only [Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_false_iff_not, not_lt] at this
      rw [outGroup_next (by omega), Nat.sub_add_cancel hpos] at this
      exact this

/-- **Every output word's bit is a canonical output row's**: in a canonical circuit whose rows are the input rows then
one per run of output ports (`HmOut.checkRows`), the runs covering the output ports (`hsum`, which `HmRow.parse`
checks), each of an output word's 16 bits is a bit of its run's output row. -/
theorem outBitRowC_ne_none (hcr : HmOut.checkRows c = .ok ())
    (hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size) (hfmt : formatOk c = true) (u : Fin n)
    {d : ℕ × ℕ × ℕ} (hw : outWordOf c d < c.outWords) (ht : d.2.2 < 16) : outBitRowC c u d ≠ none := by
  obtain ⟨hj, hle, hlt⟩ := outRunOf_spec hsum hw
  have hq := (Hidden.checkRows_out hcr hj).1
  have hb := formatOk_out hfmt hj
  have hrw : rowW c ⟨_, hq⟩ = (c.ports.getD (HmOut.nIn c + outRunOf c (outWordOf c d)) default).bits := by
    unfold rowW
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hq, Option.getD_some]
    rfl
  unfold outBitRowC
  rw [dite_eq_left hq, dite_eq_left (by rw [hrw, hb]; omega)]
  exact Option.some_ne_none _

variable {T : Model.Statement}

/-- **A returned output's column holds its canonical output row's bit** (`out_copy_key` at runs): in a satisfying
witness of an accepted statement on a canonical circuit `c`, at block `o` and VU `g`, the VU's column on copy `u'`'s
output `j`, bit `t` (`Layout.OutColAt`) holds the witness's bit at the message column of its output row's bit
(`outBitRowC`). No condition on the out net: with a tail, no column is a returned output.

*Route*: the module docstring's. -/
theorem out_copyC (a : AcceptedH T) (hce : a.st.c = c) (hfmt : formatOk c = true) {z : Witness T.m}
    (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w) {u' j t : ℕ}
    (hoc : OutColAt c hU x.val u' j t) {u v : Fin n} {q : Fin c.ports.size} {k : Fin (rowW c q)}
    (hb : outBitRowC c u (u', j, t) = some (v, ⟨q, k⟩)) :
    T.block z o (a.col hg x) = z (siteAddrH a o (Typed.msgCol a.st.c g q.val (k.val / 8) (k.val % 8))) := by
  subst hce
  obtain ⟨-, hq, hk⟩ := outBitRowC_spec hb
  obtain ⟨hu', hj, ht, h1, h2, hx⟩ := hoc
  have P := parse_facts a.parse
  obtain ⟨ls, hls⟩ := P.check
  by_cases ho : a.st.c.outNet = a.st.c.unitNet
  swap
  · have := check_tail P.typed ho hls u'
    omega
  obtain ⟨-, -, hlo⟩ := Layout.check_out P.typed ho hls
  have hget : a.st.c.leavesOut.getD u' #[] = a.st.c.leavesOut[u']! := by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hwj : outWordOf a.st.c (u', j, t) = (a.st.c.leavesOut.getD u' #[])[j] := getElem!_pos _ j hj
  have hwlt : outWordOf a.st.c (u', j, t) < a.st.c.outWords := by
    rw [hwj]
    refine (hlo u' hu').2 _ ?_
    rw [← hget]
    exact Array.getElem_mem_toList _
  obtain ⟨hjr, hle, hlt⟩ := outRunOf_spec (Hidden.parse_rowPorts a.parse).2 hwlt
  obtain ⟨hrows, hwords⟩ := outRowsC_of hU a.parse a.checkRows hfmt
  have hw : (HmOut.outGroup a.st.c (outRunOf a.st.c (outWordOf a.st.c (u', j, t)))).2.2.1 +
      (outWordOf a.st.c (u', j, t) - (HmOut.outGroup a.st.c (outRunOf a.st.c (outWordOf a.st.c (u', j, t)))).2.2.1) =
      (a.st.c.leavesOut.getD u' #[])[j] := by
    rw [← hwj]
    omega
  have := out_copyHC a ho hrows hz o hg x hu' hj ht h1 h2 hx hjr (by rw [hwords _ hjr]; omega) (by omega) hw
  rw [hq, hk]
  exact this

end FlockSoundness.Discharge.Canonical
