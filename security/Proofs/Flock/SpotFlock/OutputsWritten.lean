module

public import Proofs.Flock.SpotFlock.NeededBitsHeld
public import Proofs.Flock.SpotFlock.WiringRows
public import Proofs.Flock.Claims.Recursion.ResLowerDeployed

/-! # Every output word's bit of a member's unit is a written bit

`SpotFlock.outputsWritten_of_two`: at an accepted design, each bit below `16 · w` of output row `nIn + o` of member
`t`'s circuit, `w` that output row's own words (`(HmOut.outGroup c o).2.2.2`), at leaf `u < n_t`, is a written bit
(`writtenBits`); `outputsWritten_at_port` is the same at a port index. Every output word is returned by some instance's
leaf (`RecDeploy.ResLower.check_complete`), and that leaf's bit is a copy row's, where every returned output bit sits on
a computed row (`OutsComputed`).

The bits of a padded output row past `16 · w` (`HmRow.checkRows`' `padded`) are zero padding no leaf returns, so no
written bit sits there. `OutsComputed` is `Spot.Bits.check`'s own condition ("a returned output bit … sits on no
computed row"). The design's evaluation decides it of every member (`outsOk`, a clause of `wiringOk`), so an accepted
design gives it (`outsComputed_of_two`, from `outs_of_two`) and the fact takes no hypothesis beyond the design. -/

@[expose] public section

namespace Flock.Bridge.NeededBitsOf

open FlockSoundness

/-- **Every returned output bit sits on a computed row** of the unit net (`Spot.Bits.check`'s condition): the column
`outCol c j t` of each instance's leaf `j`, bit `t`, is past the input rows and before the constant. -/
def OutsComputed (c : Flock.Circuit) : Prop :=
  ∀ k < c.upv, ∀ j < (c.leavesOut.getD k #[]).size, ∀ t < 16,
    (Spot.Bits.Rows.ofNet c.unit).nIn ≤ Spot.Bits.outCol c j t ∧
      Spot.Bits.outCol c j t < (Spot.Bits.Rows.ofNet c.unit).nIn + (Spot.Bits.Rows.ofNet c.unit).nComp

section words

variable (c : Flock.Circuit) (n : ℕ)

/-- **Word `outAt o + d` of output port `o` is that port's** (`outPortOf`), for `d` below its words. -/
theorem outPortOf_at {o d : ℕ} (ho : o < c.outPorts.size) (hd : d < (c.outPorts.getD o default).2) :
    FlockSoundness.Discharge.Hidden.outAt c o + d < c.outWords ∧
      FlockSoundness.Discharge.HiddenExec.outPortOf c (FlockSoundness.Discharge.Hidden.outAt c o + d) = o := by
  have hsucc := FlockSoundness.Discharge.Hidden.outAt_succ c ho
  rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD'] at hsucc
  have hmono := outAt_mono c (show o + 1 ≤ c.outPorts.size by omega)
  rw [FlockSoundness.Discharge.HiddenExec.outAt_size] at hmono
  have hw : FlockSoundness.Discharge.Hidden.outAt c o + d < c.outWords := by omega
  refine ⟨hw, ?_⟩
  obtain ⟨hlt, hle, hlt2⟩ := FlockSoundness.Discharge.HiddenExec.outPortOf_spec c hw
  rcases Nat.lt_trichotomy
      (FlockSoundness.Discharge.HiddenExec.outPortOf c (FlockSoundness.Discharge.Hidden.outAt c o + d)) o with h | h | h
  · have h1 := outAt_mono c (show FlockSoundness.Discharge.HiddenExec.outPortOf c
      (FlockSoundness.Discharge.Hidden.outAt c o + d) + 1 ≤ o by omega)
    have h2 := FlockSoundness.Discharge.Hidden.outAt_succ c hlt
    rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD'] at h2
    omega
  · exact h
  · have h1 := outAt_mono c (show o + 1 ≤
      FlockSoundness.Discharge.HiddenExec.outPortOf c (FlockSoundness.Discharge.Hidden.outAt c o + d) by omega)
    omega

/-- **Every output word is returned by some instance's leaf**, of a parsed statement in the hidden-output scope. -/
theorem covered_of_parse {tags : Flock.Tags} {cf : ByteArray} {tables : List (String × ByteArray)}
    (hty : tags.typed = false) (hcS : HmOut.parseStmt tags cf tables = .ok c) (hU : Built c.unit)
    (hs : ProvedScope.scopeOkH c = true) :
    ∀ w < c.outWords, ∃ k < c.upv, ∃ j : ℕ, (c.leavesOut.getD k #[])[j]? = some w := by
  rw [FlockSoundness.Discharge.FailClosed.scopeOkH_eq] at hs
  obtain ⟨-, ho, -⟩ := FlockSoundness.Discharge.Layout.scopeOkH_spec hU hs
  obtain ⟨cs, hcs, hce⟩ := FlockSoundness.commit_of_parseStmt hty hcS
  have hC := FlockSoundness.Discharge.Hidden.committed_of_commit hcs
  obtain ⟨nets, he⟩ := FlockSoundness.Discharge.Hidden.Committed.unrolled_eq hty hC
  obtain ⟨hty', ⟨ls, hls⟩, -⟩ := FlockSoundness.Discharge.Hidden.Committed.check hty hC
  subst hce
  rw [he] at ho ⊢
  intro w hw
  have hret := _root_.RecDeploy.ResLower.check_complete hty' ho hls hw
  unfold FlockSoundness.Discharge.HiddenExec.returned at hret
  obtain ⟨k, hk, hwk⟩ := List.mem_flatMap.1 hret
  obtain ⟨j, hj⟩ := Array.mem_iff_getElem?.1 (Array.mem_toList_iff.1 hwk)
  exact ⟨k, List.mem_range.1 hk, j, hj⟩

/-- **Bit `bit < 16 · w` of output row `nIn + o` at leaf `u` is a copy row's bit**: some instance `k`'s leaf `j`, bit
`bit % 16`, has a gate (`outGate`) and its commitment position is that bit. -/
theorem written_of_word (hcov : ∀ w < c.outWords, ∃ k < c.upv, ∃ j : ℕ, (c.leavesOut.getD k #[])[j]? = some w)
    (hr : c.outRowPorts = Array.replicate c.outPorts.size 1) (hC : OutsComputed c) {u o bit : ℕ} (hu : u < n)
    (ho : o < c.outPorts.size) (hb : bit < 16 * (c.outPorts.getD o default).2) :
    ∃ k j t x, Spot.Bits.outGate c n u k j t = some x ∧
      Spot.Bits.outRowBit c u k j t = some ⟨(c.ports.getD (HmOut.nIn c + o) default).name, u, bit⟩ := by
  obtain ⟨hw, hpo⟩ := outPortOf_at c ho (show bit / 16 < (c.outPorts.getD o default).2 by omega)
  obtain ⟨k, hk, j, hj⟩ := hcov _ hw
  obtain ⟨hjs, -⟩ := Array.getElem?_eq_some_iff.1 hj
  obtain ⟨h1, h2⟩ := hC k hk j hjs (bit % 16) (Nat.mod_lt _ (by omega))
  have hg : ∃ x, Spot.Bits.outGate c n u k j (bit % 16) = some x := by
    cases hx : Spot.Bits.outGate c n u k j (bit % 16) with
    | some x => exact ⟨x, rfl⟩
    | none =>
      exfalso
      unfold Spot.Bits.outGate at hx
      dsimp only at hx
      split at hx
      · cases hx
      · rename_i hne
        apply hne
        simp only [Bool.and_eq_true, decide_eq_true_eq]
        exact ⟨⟨⟨⟨⟨hu, hk⟩, hjs⟩, Nat.mod_lt _ (by omega)⟩, h1⟩, h2⟩
  obtain ⟨x, hx⟩ := hg
  refine ⟨k, j, bit % 16, x, hx, ?_⟩
  · unfold Spot.Bits.outRowBit
    rw [hj]
    show Spot.Bits.outPos c u (FlockSoundness.Discharge.Hidden.outAt c o + bit / 16) (bit % 16) = _
    rw [outPos_eq hr hw, hpo]
    congr 2
    omega

end words

end Flock.Bridge.NeededBitsOf

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy FlockSoundness FlockSoundness.Discharge Exec Flock Flock.Readable

/-- **At an accepted design, every returned output bit of each member's unit sits on a computed row**: the design
decides it of every member (`outsOk`, `outs_of_two`). -/
theorem outsComputed_of_two (h : Held) {s : AuditStatement} {d : Design.Two} (hd : two h s = some d)
    {t : ℕ} (ht : t < h.ms.length) : _root_.Flock.Bridge.NeededBitsOf.OutsComputed h.ms[t].1 := by
  unfold _root_.Flock.Bridge.NeededBitsOf.OutsComputed
  exact outs_of_two h hd ht

/-- The fact on output port `o`'s words (`outPorts`' second): `outputsWritten_of_two` words it on the output row's. -/
theorem outputsWritten_words_of_two (h : Held) {s : AuditStatement} {d : Design.Two} (hd : two h s = some d)
    {t : ℕ} (ht : t < h.ms.length) {u : ℕ} (hu : u < h.ms[t].2.size) {o : ℕ} (ho : o < h.ms[t].1.outPorts.size)
    {bit : ℕ} (hb : bit < 16 * (h.ms[t].1.outPorts.getD o default).2) :
    ∃ x, (x, t, (⟨(h.ms[t].1.ports.getD (_root_.Flock.HmOut.nIn h.ms[t].1 + o) default).name, u, bit⟩ :
      Spot.Bits.RowPos)) ∈ writtenBits h.ms := by
  obtain ⟨tags, cf, tables, hty, hcS⟩ := member_parsed h hd ht
  obtain ⟨hU, -, -, hR, -⟩ := member_facts h hd ht
  have hcov := _root_.Flock.Bridge.NeededBitsOf.covered_of_parse _ hty hcS hU
    (scoped_of_two h hd _ (List.getElem_mem ht))
  obtain ⟨k, j, b, x, hx, hr⟩ :=
    _root_.Flock.Bridge.NeededBitsOf.written_of_word _ _ hcov hR.1.1 (outsComputed_of_two h hd ht) hu ho hb
  exact ⟨_, _root_.Flock.Bridge.NeededBitsOf.mem_writtenBits_of ht hx hr⟩

/-- **Every output word's bit of a member's unit is a written bit**, at an accepted design: bit `bit` below `16 · w`
of member `t`'s output row `nIn + o`, `w` that output row's own words (`HmOut.outGroup`'s fourth), at leaf `u < n_t`.
A padded row's bits from `16 · w` on are not in it. -/
theorem outputsWritten_of_two (h : Held) {s : AuditStatement} {d : Design.Two} (hd : two h s = some d)
    {t : ℕ} (ht : t < h.ms.length) {u : ℕ} (hu : u < h.ms[t].2.size) {o : ℕ} (ho : o < h.ms[t].1.outPorts.size)
    {bit : ℕ} (hb : bit < 16 * (_root_.Flock.HmOut.outGroup h.ms[t].1 o).2.2.2) :
    ∃ x, (x, t, (⟨(h.ms[t].1.ports.getD (_root_.Flock.HmOut.nIn h.ms[t].1 + o) default).name, u, bit⟩ :
      Spot.Bits.RowPos)) ∈ writtenBits h.ms := by
  obtain ⟨_, -, -, hR, -⟩ := member_facts h hd ht
  rw [_root_.FlockSoundness.Discharge.Hidden.outGroup_single hR.1.1 ho] at hb
  exact outputsWritten_words_of_two h hd ht hu ho hb

/-- **The same at a port index**: bit `bit` below `16 · w` of member `t`'s row port `q`, an output row
(`nIn ≤ q < ports.size`), `w` the words of output row `q − nIn`. -/
theorem outputsWritten_at_port (h : Held) {s : AuditStatement} {d : Design.Two} (hd : two h s = some d)
    {t : ℕ} (ht : t < h.ms.length) {u : ℕ} (hu : u < h.ms[t].2.size) {q : ℕ}
    (hq : _root_.Flock.HmOut.nIn h.ms[t].1 ≤ q) (hqs : q < h.ms[t].1.ports.size) {bit : ℕ}
    (hb : bit < 16 * (_root_.Flock.HmOut.outGroup h.ms[t].1 (q - _root_.Flock.HmOut.nIn h.ms[t].1)).2.2.2) :
    ∃ x, (x, t, (⟨(h.ms[t].1.ports.getD q default).name, u, bit⟩ : Spot.Bits.RowPos)) ∈ writtenBits h.ms := by
  obtain ⟨_, -, -, hR, -⟩ := member_facts h hd ht
  have hsz := congrArg Array.size hR.1.1
  rw [Array.size_replicate] at hsz
  have hn : _root_.Flock.HmOut.nIn h.ms[t].1 = h.ms[t].1.ports.size - h.ms[t].1.outRowPorts.size := rfl
  have ho : q - _root_.Flock.HmOut.nIn h.ms[t].1 < h.ms[t].1.outPorts.size := by omega
  obtain ⟨x, hx⟩ := outputsWritten_of_two h hd ht hu ho hb
  have he : _root_.Flock.HmOut.nIn h.ms[t].1 + (q - _root_.Flock.HmOut.nIn h.ms[t].1) = q := by omega
  rw [he] at hx
  exact ⟨x, hx⟩

end SpotFlock
