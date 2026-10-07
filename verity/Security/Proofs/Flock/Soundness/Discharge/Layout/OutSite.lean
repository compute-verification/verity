import Proofs.Flock.Soundness.Discharge.Layout.OutRegion
import Proofs.Flock.Soundness.Discharge.Layout.Site

set_option Elab.async false

/-!
# The drawn units' returned outputs: the population's

A drawn statement's outputs are `HmRow.drawn`'s: instance `r`'s `nOut` output words are proved unit `D.proved[r]`'s in
the registered population's public file (`drawn_outputs`), which holds `n · nOut` words, each below `2^16`
(`loadPublic_outputs`). So at a proved unit's slot, region "out"'s output `j` of the slot's leaf is the population's
(`outVal_drawn`), and a satisfying witness holds its bits at the slot's output columns (`out_drawn`; at a drawn unit's
site in its table's accepted statement, `out_site`).
-/

namespace FlockSoundness.Discharge.Layout

open Flock Refine Model

set_option maxHeartbeats 2000000 in
/-- **`HmRow.drawn`'s outputs**: the proved units' `nOut` output words, unit by unit. -/
theorem drawn_outputs {c : Circuit} {pop pub : Public} {d : Lean.Json} (h : HmRow.drawn c pop d = .ok pub) :
    ∃ D, Draw.UnitDraw.ofJson d = .ok D ∧ pub.outputs =
      D.proved.foldl (fun acc u => acc ++ pop.outputs.extract (u * c.outWords) (u * c.outWords + c.outWords)) #[] := by
  unfold HmRow.drawn at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨_, ‹_›, rfl⟩

theorem two_bytes_lt (a b : UInt8) : a.toNat + 256 * b.toNat < 2 ^ 16 := by
  have := a.toNat_lt
  have := b.toNat_lt
  omega

set_option maxHeartbeats 2000000 in
/-- **The population's outputs**: a public file holds `n · nOut` output words, each two bytes (below `2^16`). -/
theorem loadPublic_outputs {tags : Tags} {c : Circuit} {file : ByteArray} {pop : Public}
    (h : HmRow.loadPublic tags c file = .ok pop) :
    pop.outputs.size = pop.n * c.outWords ∧ ∀ x ∈ pop.outputs, x < 2 ^ 16 := by
  unfold HmRow.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨by simp, fun x hx => ?_⟩
    simp only [Array.mem_map, Array.mem_range] at hx
    obtain ⟨i, -, rfl⟩ := hx
    exact two_bytes_lt _ _

theorem getD_lt_of_forall {a : Array ℕ} {B : ℕ} (h : ∀ x ∈ a, x < B) (hB : 0 < B) (i : ℕ) : a.getD i 0 < B := by
  unfold Array.getD
  split
  · exact h _ (Array.getElem_mem _)
  · exact hB

/-- **Region "out"'s output at a proved unit's slot is the population's**: at block `o` and unit slot `k` whose instance
`o · c.g + k / upv` is proved, output `j` of the slot's leaf (an output word below `nOut`) is proved unit
`D.proved[o · c.g + k / upv]`'s in the population. -/
theorem outVal_drawn {c : Circuit} {pop pub : Public} {d : Lean.Json} {D : Draw.UnitDraw}
    (h : HmRow.drawn c pop d = .ok pub) (hD : Draw.UnitDraw.ofJson d = .ok D)
    (hsz : pop.outputs.size = pop.n * c.outWords) (hlt : ∀ u ∈ D.proved, u < D.population)
    {o k j : ℕ} (hr : o * c.g + k / c.upv < D.proved.size)
    (hlo : (c.leavesOut.getD (k % c.upv) #[])[j]! < c.outWords) :
    outVal c pub o k j =
      pop.outputs.getD (D.proved[o * c.g + k / c.upv] * c.outWords + (c.leavesOut.getD (k % c.upv) #[])[j]!) 0 := by
  obtain ⟨D', hD', hpopn, hn, -, -⟩ := drawn_spec h
  obtain ⟨D'', hD'', hout⟩ := drawn_outputs h
  rw [hD] at hD' hD''
  cases hD'
  cases hD''
  unfold outVal Circuit.instOf
  rw [hn]
  simp only [hr, ↓reduceIte]
  rw [hout]
  refine foldl_extract_getD pop.outputs D.proved (fun u hu => ?_) 0 hr hlo
  have := hlt u hu
  rw [hsz, ← hpopn]
  nlinarith

/-- **The returned outputs a satisfying witness of a drawn statement holds at a proved unit's slot** are the unit's in
the registered population's public file: in an untyped circuit whose unit returns the outputs, at block `o` and unit slot
`k` (below `2^⌊log₂ count⌋`) whose instance `o · c.g + k / upv` is proved, the bit at column `16 j + t` of the slot's
returned-output group is bit `t` of the population's output word `lo[j]` of unit `D.proved[o · c.g + k / upv]`, for the
leaf's outputs `lo` (words below `nOut`, `hlo`), under `out_pinned`'s conditions on the output group. -/
theorem out_drawn {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {d : Lean.Json} {partition : Option Lean.Json} {program : Option ByteArray}
    {st : Stmt} (h : Stmt.setupH tags circuitFile publicFile coins tables (some d) partition program = .ok st)
    {pop : Public} (hpop : HmRow.loadPublic tags st.c publicFile = .ok pop)
    {D : Draw.UnitDraw} (hD : Draw.UnitDraw.ofJson d = .ok D) (hlt : ∀ u ∈ D.proved, u < D.population)
    (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    (ho : st.c.outNet = st.c.unitNet) (hty : st.c.typed = none)
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {k j t : ℕ} (hk : k < st.c.unitRange.count) (hk2 : k < 2 ^ st.c.unitRange.count.log2)
    (hr : o.val * st.c.g + k / st.c.upv < D.proved.size)
    (hw : (st.c.unit.outGroups.getD 0 default).words = 2 ^ (st.c.unit.outGroups.getD 0 default).words.log2)
    (hb : ∀ j < (st.c.leavesOut.getD (k % st.c.upv) #[]).size, outBits st.c j = 16)
    (hlo : ∀ j < (st.c.leavesOut.getD (k % st.c.upv) #[]).size, (st.c.leavesOut.getD (k % st.c.upv) #[])[j]! <
      st.c.outWords)
    (hfit : 16 * (st.c.leavesOut.getD (k % st.c.upv) #[]).size ≤ 128 * (st.c.unit.outGroups.getD 0 default).words)
    (hj : j < (st.c.leavesOut.getD (k % st.c.upv) #[]).size) (ht : t < 16) :
    bit ((L.model regs).block z o (inBlockOr st.c.kLog (outAddr st.c k j t))) =
      (pop.outputs.getD (D.proved[o.val * st.c.g + k / st.c.upv] * st.c.outWords +
        (st.c.leavesOut.getD (k % st.c.upv) #[])[j]!) 0).testBit t := by
  obtain ⟨pop', hpop', hdr⟩ := setupH_drawn h
  rw [hpop] at hpop'
  cases hpop'
  obtain ⟨hsz, h16⟩ := loadPublic_outputs hpop
  rw [← outVal_drawn hdr hD hsz hlt hr (hlo j hj)]
  refine out_pinned L regs hregs (setupH_regions h) ho hty hz o hk hk2 hw hb (fun j' hj' => ?_) hfit hj ht
  rw [outVal_drawn hdr hD hsz hlt hr (hlo j' hj')]
  exact getD_lt_of_forall h16 (Nat.two_pow_pos _) _

variable {T : Model.Statement}

/-- **The returned outputs at a drawn unit's site are the population's**: in a satisfying witness of an accepted drawn
statement of an untyped circuit whose unit returns the outputs, at a site (`BcSite`: block `o`, template instance `g`, proved unit `u`),
the bit at column `16 j + t` of the returned-output group of the template instance's copy `u'` is bit `t` of the
population's output word `lo[j]` of `u`, for the copy's leaf's outputs `lo`, under `out_pinned`'s conditions. -/
theorem out_site (a : Placed.Accepted T) {pop : Public} {o : Fin (2 ^ T.nbl)} {g u : ℕ} (hs : BcSite a pop o g u)
    (ho : a.st.c.outNet = a.st.c.unitNet) (hty : a.st.c.typed = none)
    {z : Witness T.m} (hz : T.Satisfies z) {u' j t : ℕ} (hu' : u' < a.st.c.upv)
    (hk : g * a.st.c.upv + u' < a.st.c.unitRange.count)
    (hk2 : g * a.st.c.upv + u' < 2 ^ a.st.c.unitRange.count.log2)
    (hw : (a.st.c.unit.outGroups.getD 0 default).words = 2 ^ (a.st.c.unit.outGroups.getD 0 default).words.log2)
    (hb : ∀ j < (a.st.c.leavesOut.getD u' #[]).size, outBits a.st.c j = 16)
    (hlo : ∀ j < (a.st.c.leavesOut.getD u' #[]).size, (a.st.c.leavesOut.getD u' #[])[j]! < a.st.c.outWords)
    (hfit : 16 * (a.st.c.leavesOut.getD u' #[]).size ≤ 128 * (a.st.c.unit.outGroups.getD 0 default).words)
    (hj : j < (a.st.c.leavesOut.getD u' #[]).size) (ht : t < 16) :
    bit (z (siteAddr a o (outAddr a.st.c (g * a.st.c.upv + u') j t))) =
      (pop.outputs.getD (u * a.st.c.outWords + (a.st.c.leavesOut.getD u' #[])[j]!) 0).testBit t := by
  obtain ⟨d, D, hd, hD, hlt, hregs, hpop, -, hu⟩ := hs
  have hm : (g * a.st.c.upv + u') % a.st.c.upv = u' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hu']
  have hdv : (g * a.st.c.upv + u') / a.st.c.upv = g := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hu', zero_add]
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, regs, hT⟩ := a
  subst hT
  dsimp only at hd hregs hpop hu hm hdv hk hk2 hw hb hlo hfit hj ho hty hu' ⊢
  subst hd
  obtain ⟨hr, rfl⟩ := Array.getElem?_eq_some_iff.1 hu
  have := out_drawn h hpop hD hlt (setupH_layout ht' h) regs hregs ho hty hz o hk hk2 (by rw [hdv]; exact hr) hw
    (by rw [hm]; exact hb) (by rw [hm]; exact hlo) (by rw [hm]; exact hfit) (by rw [hm]; exact hj) ht
  simp only [hm, hdv] at this
  exact this

end FlockSoundness.Discharge.Layout
