import FlockSoundness.Discharge.Canonical.Commit
import FlockSoundness.Discharge.HiddenExec.KeyH

set_option Elab.async false

/-!
# Output rows of runs of output ports, exact length

`HiddenExec.outBitRow` puts output word `w` in output row `nIn + jo`, one per output port (`outPortOf`), at bit
`16 (w - outAt jo) + t` of a row of `16 · words` bits zero-padded to whole blocks. A canonical output row (`PROTOCOL.md`
§16.15) is one per run of output ports (META `out_row_ports`, `HmOut.outGroup c j = (first, count, start, words)`), of
exactly the run's `16 · words` bits. So output word `w` is in run `outRunOf c w` (the first whose words end past `w`), at
bit `16 (w - start) + t`.

* `outRunOf`, `outBitRowC`: that bit, as a row bit of `keyProg` (`Fin (rowW c q)`); `outBitRowC_spec` reads it back;
* `outBitRowC_ne_none`: every output word's bit is a canonical output row's;
* `out_copyC`: `HiddenExec.out_copy_key` at runs, without `outNet = unitNet` and `sha512/row/v1` rows.

When a tail's last stage returns the outputs (`outNet ≠ unitNet`), `HmRow.parse` requires every `leavesOut` empty, so no
unit column is a returned output (`Layout.OutColAt` fails) and `out_copyC` holds vacuously: `keyProg` (the unit net's rows
alone) does not compute a tail's outputs. The statement-level `OutputsCommittedC` (`Statements.lean`) covers them.

**Proof work.**
* `outBitRowC_spec`: `outBitRow_spec`'s, proved here.
* `outBitRowC_ne_none`: small. `HmOut.checkRows` makes the runs consecutive and covering (`outGroup`'s `start` is a
  prefix sum of `outPorts`' words, and the runs' counts sum to `outPorts.size`), so `outRunOf` finds the run of `w` and
  `16 (w - start) + t < 16 · words = ℓ` by `formatOk`'s `outRowsExact`. One prefix-sum lemma on `outGroup` is new.
* `out_copyC`: moderate. `out_copyH`'s route (`Hidden.msg_copy`: Δ's output pair copies the computed column to the
  output row's message column) at runs: `Hidden.outPairs` already writes one pair per output word into its run's row
  (`Hidden.outSrc` over `outGroup`; `outGroup_single` is the one-port case `out_copyH` uses), and `Hidden.OutRows` /
  `out_rows_v1` need their run form (`outRows_runs`: output row `j` is port `nIn + j` with `16 · words` bits). The tail
  case is `HmRow.parse`'s `leavesOut` check, one `check_facts` lemma.
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

/-- **Every output word's bit is a canonical output row's**: in a canonical circuit whose rows are the input rows then
one per run of output ports (`HmOut.checkRows`), each of an output word's 16 bits is a bit of its run's output row.

*Route*: the module docstring's. -/
theorem outBitRowC_ne_none (hcr : HmOut.checkRows c = .ok ()) (hfmt : formatOk c = true) (u : Fin n)
    {d : ℕ × ℕ × ℕ} (hw : outWordOf c d < c.outWords) (ht : d.2.2 < 16) : outBitRowC c u d ≠ none := by
  sorry

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
  sorry

end FlockSoundness.Discharge.Canonical
