import FlockSoundness.Discharge.Canonical.Format
import FlockSoundness.Discharge.FailClosed.Scope

set_option Elab.async false

/-!
# The verifier's obligation: the statement check accepts canonical circuits only

`parse_formatOk` is the one statement about the executable verifier the canonical headline (`Headline.lean`) adds: a
circuit the statement check passes (`FailClosed.Checks`) outside `Layout.scopeOk` is at a `--zk` statement with hidden
outputs, its unit returns the outputs, and it is in the canonical format (`formatOk`).

The check (`Flock.ProvedScope.check`, verifier change V1, `PROTOCOL.md` §16.16) accepts a circuit that `scopeOk` or
`canonicalOk` accepts; `canonicalOk` is `--zk`, hidden outputs, `outNet = unitNet` and the executable's
`ProvedScope.formatOk`. That is `formatOk` with its `readsRows` (Δ's key at every stacked unit input is not a wire or a
fixed column) as `cutsCovered`, a mask over the leaf cuts: every cut port of every copy has one
(`readsRows_of_cutsCovered`; at a cut port `srcAt` reads a wire or a fixed column only where no leaf cut names it), and
`portReadsB` as the port mask's (`FailClosed.portReadsB_eq`). `formatOk_of_provedScope` is the translation.

Every declaration here is proved.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock Exec

/-- **At a cut port a leaf cut names, Δ's key is a row bit or the zero**, never a wire's or a fixed column's. -/
theorem srcAt_cut {c : Circuit} {k p t : ℕ} (hp : (c.unit.inGroups.getD 0 default).bits.size ≤ p)
    {e : ℕ × ℕ × ℕ} (he : e ∈ c.leafCuts.toList) (hek : e.1 = k) (hep : e.2.1 = p) :
    (∀ net slot col, Aliased.srcAt c k p t ≠ .wire net slot col) ∧ ∀ col, Aliased.srcAt c k p t ≠ .lit col := by
  have hne : c.leafCuts.toList.filter (fun e => decide (e.1 = k ∧ e.2.1 = p)) ≠ [] := by
    intro h0
    have := List.filter_eq_nil_iff.1 h0 e he
    simp [hek, hep] at this
  obtain ⟨e', l, hl⟩ := List.exists_cons_of_ne_nil hne
  unfold Aliased.srcAt
  rw [if_neg (by omega), hl]
  refine ⟨fun _ _ _ => ?_, fun _ => ?_⟩ <;> dsimp only <;> split <;> simp

/-- **At a leaf port, Δ's key is a row bit or the zero.** -/
theorem srcAt_leafPort {c : Circuit} {k p t : ℕ} (hp : p < (c.unit.inGroups.getD 0 default).bits.size) :
    (∀ net slot col, Aliased.srcAt c k p t ≠ .wire net slot col) ∧ ∀ col, Aliased.srcAt c k p t ≠ .lit col := by
  rcases Aliased.srcAt_leaf c k t hp with h | ⟨q, byte, bit, h, -⟩ <;> rw [h] <;> simp

/-- **`cutsCovered` names a leaf cut at every cut port of every copy.** -/
theorem cut_of_covered {c : Circuit} (h : ProvedScope.cutsCovered c = true) {k p : ℕ} (hk : k < c.upv)
    (hp0 : (c.unit.inGroups.getD 0 default).bits.size ≤ p) (hp : p < (portCols c.unit.inGroups).size) :
    ∃ e ∈ c.leafCuts.toList, e.1 = k ∧ e.2.1 = p := by
  unfold ProvedScope.cutsCovered at h
  simp only [List.all_eq_true, List.mem_range] at h
  have hm := h k hk (p - (c.unit.inGroups.getD 0 default).bits.size) (by omega)
  rw [show (c.unit.inGroups.getD 0 default).bits.size + (p - (c.unit.inGroups.getD 0 default).bits.size) = p by
    omega] at hm
  have hidx : k * (portCols c.unit.inGroups).size + p < c.upv * (portCols c.unit.inGroups).size := by
    have : (k + 1) * (portCols c.unit.inGroups).size ≤ c.upv * (portCols c.unit.inGroups).size :=
      Nat.mul_le_mul_right _ hk
    rw [Nat.add_mul, Nat.one_mul] at this
    omega
  rw [FailClosed.maskOf_getD _ hidx, decide_eq_true_iff] at hm
  obtain ⟨e, he, hf⟩ := List.mem_filterMap.1 hm
  split at hf
  · rename_i hlt
    have hf' := Option.some.inj hf
    refine ⟨e, he, ?_, ?_⟩
    · have := congrArg (· / (portCols c.unit.inGroups).size) hf'
      rwa [Nat.mul_comm e.1, Nat.mul_comm k, Nat.mul_add_div (by omega), Nat.mul_add_div (by omega),
        Nat.div_eq_of_lt hlt, Nat.div_eq_of_lt hp, Nat.add_zero, Nat.add_zero] at this
    · have := congrArg (· % (portCols c.unit.inGroups).size) hf'
      rwa [Nat.mul_comm e.1, Nat.mul_comm k, Nat.mul_add_mod, Nat.mul_add_mod, Nat.mod_eq_of_lt hlt,
        Nat.mod_eq_of_lt hp] at this
  · cases hf

/-- **`cutsCovered` gives `readsRows`**: every stacked unit input's key is a row bit, the zero, or a column outside every
port. -/
theorem readsRows_of_cutsCovered {c : Circuit} (hU : Built c.unit) (h : ProvedScope.cutsCovered c = true) :
    readsRows c = true := by
  unfold readsRows
  refine List.all_eq_true.2 fun x hx => ?_
  have hx := List.mem_range.1 hx
  have hk : x / (c.unit.inWords * WORD) < c.upv := Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hx)
  have hkey : (∀ net slot col, Aliased.srcKey c x ≠ .wire net slot col) ∧ ∀ col, Aliased.srcKey c x ≠ .lit col := by
    unfold Aliased.srcKey
    split
    · rename_i hm
      obtain ⟨p, t, hp, ht, hpt⟩ := Aliased.mem_portBits hm
      rw [hpt, FlockSoundness.portOf_eq (Aliased.portBits_nodup hU) hp ht, Nat.add_sub_cancel_left]
      by_cases hp0 : p < (c.unit.inGroups.getD 0 default).bits.size
      · exact srcAt_leafPort hp0
      · obtain ⟨e, he, hek, hep⟩ := cut_of_covered h hk (Nat.le_of_not_lt hp0) hp
        exact srcAt_cut (Nat.le_of_not_lt hp0) he hek hep
    · exact ⟨fun _ _ _ => by simp, fun _ => by simp⟩
  obtain ⟨hnw, hnl⟩ := hkey
  rcases hs : Aliased.srcKey c x with _ | ⟨q, byte, bit⟩ | ⟨net, slot, col⟩ | ⟨col⟩ | ⟨k, off⟩
  · rfl
  · rfl
  · exact absurd hs (hnw _ _ _)
  · exact absurd hs (hnl _)
  · rfl

/-- **The executable's `formatOk` is `formatOk`**, at a built unit. -/
theorem formatOk_of_provedScope {c : Circuit} (hU : Built c.unit) (h : ProvedScope.formatOk c = true) :
    formatOk c = true := by
  simp only [ProvedScope.formatOk, Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨hrow, hout⟩, hcut⟩, hpr⟩, hu⟩, hg⟩ := h
  simp only [formatOk, Bool.and_eq_true]
  exact ⟨⟨⟨⟨⟨hrow, hout⟩, readsRows_of_cutsCovered hU hcut⟩, by rw [← FailClosed.portReadsB_eq]; exact hpr⟩, hu⟩, hg⟩

/-- **Every circuit the statement check passes outside `Layout.scopeOk` is canonical**: at a `--zk` statement with hidden
outputs, its unit returning the outputs, in the canonical format. -/
theorem parse_formatOk {t : Tags} {zk : Bool} {I : Inputs} (h : FailClosed.Checks t zk I)
    (hv : Layout.scopeOk (FailClosed.circ t I) = false) :
    zk = true ∧ t.hiddenOutputs = true ∧ (FailClosed.circ t I).outNet = (FailClosed.circ t I).unitNet ∧
      formatOk (FailClosed.circ t I) = true := by
  obtain ⟨hz, hh, ho, hf⟩ := FailClosed.Checks.canonical_spec h hv
  exact ⟨hz, hh, ho, formatOk_of_provedScope (FailClosed.Checks.built h) hf⟩

end FlockSoundness.Discharge.Canonical

namespace FlockSoundness.Discharge.FailClosed

open Flock Exec ZkExec

/-- **A `--zk` circuit the check passes outside `Layout.scopeOk` is canonical** (`Canonical.parse_formatOk`). -/
theorem ScopeZ.formatOk {I : Inputs} (h : ScopeZ I) (hv : Layout.scopeOk (circZ I) = false) :
    Canonical.formatOk (circZ I) = true :=
  (Canonical.parse_formatOk h hv).2.2.2

/-- **A `--zk` statement whose circuit is outside `Layout.scopeOk` has hidden outputs.** -/
theorem ScopeZ.hidden {I : Inputs} (h : ScopeZ I) (hv : Layout.scopeOk (circZ I) = false) :
    I.tags.hiddenOutputs = true := by
  rw [← zkTags_hiddenOutputs]; exact (Canonical.parse_formatOk h hv).2.1

end FlockSoundness.Discharge.FailClosed
