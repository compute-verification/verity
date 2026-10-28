import Proofs.Flock.Soundness.Discharge.Inst.Place
import Proofs.Flock.Soundness.Discharge.Integrate.Bridge
import Proofs.Flock.Soundness.Discharge.Hidden.Unpin
import Proofs.Flock.Soundness.Discharge.Hidden.Flat
import Proofs.Flock.Soundness.Discharge.Inst.Order

/-!
# A pinned statement's witness satisfies its statement without Δ's pinned copies

`Hidden.sat_unpin` needs two facts of a witness of an accepted statement with pinned outputs: its forced-zero row
carries 0, and at each pinned bit `unpin`'s `B` evaluates to 1. Both hold at the layout of `Hidden.unpin st`:
- the forced-zero row (`Hm.zeroCol`, the first unit slot's row past its useful rows) is empty in `unpin`'s `A`
  (`Layout.placed_zero`), and its own copies are `(zero, zero)` twice, so it carries 0 (`Pinned.zero_block'`);
- a pinned bit is a linear computed row (`LinearBit`: of any net, its local `B` the slot's constant alone), so `unpin`'s
  `B` there is the slot's constant (`placedB_linear`), whose row reads the pin in both statements (`placed_const`: no
  pinned bit is a slot's constant), so it carries 1.

So a witness of the accepted statement satisfies `unpin`'s (`sat_pinned`). Nothing here reads a row size, a `G` or a
producer family.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Refine FlockSoundness.Discharge.Inst

/-- **Column `x` is a linear computed row** of template instance `g`: column `o` of net `n`'s slot `i`, past the net's
input rows and below its constant, whose local `B` (a lookup slot's table side included) is the constant alone. -/
def LinearBit (c : Circuit) (x : ℕ) : Prop :=
  ∃ g < c.g, ∃ n i o, x = pos c g (code n i o) ∧ n < c.nets.size ∧ n ≠ c.shaNet ∧ n ≠ c.hmNet ∧
    i < c.perVu.getD n 0 ∧ netIn c n ≤ o ∧ o < (netOf c n).constPos ∧
    ∀ j, entB c n o j = if j = (netOf c n).constPos then 1 else 0

section placed

variable {st : Stmt} (L : FlockSoundness.Layout st)
include L

theorem pos_const_eq (C : CheckFacts st.c) {g n i : ℕ} (hn : n < st.c.nets.size) :
    pos st.c g (code n i (netOf st.c n).constPos) =
      ((st.c.rangeOf n).pos0 + (g * st.c.perVu.getD n 0 + i)) * netW st.c n + (netOf st.c n).constPos := by
  rw [pos_eq]
  unfold Range.slot
  rw [(net_range L C hn).2.2.2]

/-- **A linear row's `B` is its slot's constant alone.** -/
theorem placedB_linear (C : CheckFacts st.c) {g n i o : ℕ} (hg : g < st.c.g) (hn : n < st.c.nets.size)
    (hi : i < st.c.perVu.getD n 0) (hlo : netIn st.c n ≤ o) (hhi : o < (netOf st.c n).constPos)
    (hB : ∀ j, entB st.c n o j = if j = (netOf st.c n).constPos then 1 else 0)
    (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = pos st.c g (code n i o)) :
    placedB st r j = if j.val = pos st.c g (code n i (netOf st.c n).constPos) then 1 else 0 := by
  have hW : 0 < netW st.c n := Nat.two_pow_pos _
  have hc : (netOf st.c n).constPos < netW st.c n := (L.fit n).1
  have ho : o < netW st.c n := lt_trans hhi hc
  show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
  rw [L.delta_b, hr, rangesEntry_net L C hn hg hi ho, delta_comp_net L C hn hg hi hlo hhi, add_zero,
    ite_slot_eq (v := slotEntryB st (.net n) o) hW hc hB, pos_const_eq L C hn]

/-- **No linear row is a slot's constant.** -/
theorem linear_ne_const (C : CheckFacts st.c) {x : ℕ} (hx : LinearBit st.c x) {g n i : ℕ}
    (hn : n < st.c.nets.size) (hg : g < st.c.g) (hi : i < st.c.perVu.getD n 0) :
    x ≠ pos st.c g (code n i (netOf st.c n).constPos) := by
  obtain ⟨g', hg', n', i', o', rfl, hn', -, -, hi', -, hhi', -⟩ := hx
  intro he
  obtain ⟨hm, hk, -, hsl⟩ := net_range L C hn
  obtain ⟨hm', hk', -, hsl'⟩ := net_range L C hn'
  rw [pos_eq, pos_eq] at he
  obtain ⟨rfl, -, h⟩ := eq_slot L hm' hm hk' hk (net_slot_lt L C hn' hg' hi')
    (by rw [hsl']; exact lt_trans hhi' (L.fit n').1) (net_slot_lt L C hn hg hi) (by rw [hsl]; exact (L.fit n).1) he
  omega

end placed

/-- `Pinned.mulVec_single` at a row whose one entry is named by its index. -/
theorem mulVec_single' {N : ℕ} (M : Matrix (Fin N) (Fin N) (ZMod 2)) {r : Fin N} {kv : ℕ}
    (h : ∀ j : Fin N, M r j = if j.val = kv then 1 else 0) (hk : kv < N) (v : Fin N → ZMod 2) :
    M.mulVec v r = v ⟨kv, hk⟩ :=
  Pinned.mulVec_single M (fun j => (h j).trans (by
    by_cases hj : j.val = kv
    · rw [if_pos hj, if_pos (Fin.ext hj)]
    · rw [if_neg hj, if_neg (fun e => hj (congrArg Fin.val e))])) v

/-- **A witness of an accepted statement with pinned outputs satisfies its statement without Δ's pinned copies**
(`Hidden.unpin`), when every pinned bit is a linear row: at `unpin`'s layout `L` and its flat twin's unit facts `F`. -/
theorem sat_pinned {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout (Hidden.unpin st))
    (F : Aliased.UnitFacts tags (Hidden.flatTwin st)) (hW : Integrate.NetWidths st.c) (hleaf : st.leaf = default)
    (hda : st.da = (HmOut.delta st.c st.pin).1) (hdb : st.db = (HmOut.delta st.c st.pin).2) (hwf : StmtWF st)
    (hlin : ∀ x ∈ Hidden.pinnedCols st.c, LinearBit st.c x)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {z : Model.Witness st.m}
    (hz : (stmtOf st hwf regs).Satisfies z) : (stmtOf (Hidden.unpin st) (Hidden.unpin_wf hwf) regs).Satisfies z := by
  have hK : Integrate.NetKinds st.c := F.parse.kinds
  have C : CheckFacts st.c := F.check
  have hcnt : ∀ r : ℕ, ((Hidden.pinnedCols st.c).count r : ZMod 2) = 1 → r ∈ Hidden.pinnedCols st.c := by
    intro r hr
    by_contra hm
    rw [List.count_eq_zero.2 hm, Nat.cast_zero] at hr
    exact zero_ne_one hr
  by_cases hs₀ : 0 < st.c.unitRange.count
  swap
  · have hnp : Hidden.pinnedCols st.c = [] := by
      refine List.eq_nil_iff_forall_not_mem.2 fun x hx => hs₀ ?_
      obtain ⟨g, hg, -⟩ := hlin x hx
      rw [L.unit_count]
      exact Nat.mul_pos (Nat.zero_lt_of_lt hg) C.upv_pos
    have hPR : Pinned.PinRows (stmtOf st hwf regs) (stmtOf (Hidden.unpin st) (Hidden.unpin_wf hwf) regs).A₀
        (stmtOf (Hidden.unpin st) (Hidden.unpin_wf hwf) regs).B₀ ⟨0, Nat.two_pow_pos _⟩ (fun _ => 0) := by
      intro r j
      refine ⟨(Hidden.matA_unpin hda r.val j.val).trans ?_, (Hidden.matB_unpin hdb r.val j.val).trans ?_⟩ <;>
        simp [Hidden.pinCnt, hnp] <;> rfl
    exact (Pinned.sat_unpin hPR hz (fun _ _ h => absurd h zero_ne_one) (fun _ _ h => absurd h zero_ne_one)).1
  have huse : st.c.unit.useful = st.c.unit.constPos + 1 := F.useful
  have hZ : Hm.zeroCol st.c < 2 ^ st.c.kLog := L.unit_slot_lt hs₀ C.unit_zero
  have hA0 : ∀ j, (stmtOf (Hidden.unpin st) (Hidden.unpin_wf hwf) regs).A₀ ⟨Hm.zeroCol st.c, hZ⟩ j = 0 := by
    intro j
    have hA : ∀ j', sparseEntry st.c.unit.a st.c.unit.useful j' = 0 := fun j' => by
      unfold sparseEntry
      rw [ite_eq_right (by have h : st.c.unit.a.rows ≤ st.c.unit.useful := F.rowsA; omega)]
    have hB : ∀ j', sparseEntry st.c.unit.b st.c.unit.useful j' +
        lookupEntry st.c.unit.lookup st.c.unit.useful j' = 0 := fun j' => by
      rw [L.unit_lookup]
      unfold sparseEntry
      rw [ite_eq_right (by have h : st.c.unit.b.rows ≤ st.c.unit.useful := F.rowsB; omega)]
      rfl
    show matA (Hidden.unpin st) (Hm.zeroCol st.c) j.val = 0
    exact (Integrate.placedA_eqW (st := Hidden.unpin st) hW hK L hleaf ⟨Hm.zeroCol st.c, hZ⟩ j).symm.trans
      (L.placed_zero hs₀ C.unit_zero huse hA hB _ j rfl).1
  have hPR := Hidden.pinRows_unpin hda hdb hwf regs ⟨Hm.zeroCol st.c, hZ⟩ rfl
  refine (Hidden.sat_unpin hda hdb hwf (zero := ⟨Hm.zeroCol st.c, hZ⟩) rfl hz
    (fun o _ _ => Pinned.zero_block' hPR hA0 hz o) fun o r hr => ?_).1
  obtain ⟨g, hg, n, i, o', hx, hn, hsha, hhm, hi, hlo, hhi, hBe⟩ := hlin r.val (hcnt r.val hr)
  have hk : pos st.c g (code n i (netOf st.c n).constPos) < 2 ^ st.c.kLog :=
    pos_lt L C hn hg hi (L.fit n).1
  let k : Fin (2 ^ st.c.kLog) := ⟨_, hk⟩
  have hkv : k.val = pos st.c g (code n i (netOf st.c n).constPos) := rfl
  have hBr : ∀ j, (stmtOf (Hidden.unpin st) (Hidden.unpin_wf hwf) regs).B₀ r j =
      if j.val = pos st.c g (code n i (netOf st.c n).constPos) then 1 else 0 := fun j =>
    (Integrate.placedB_eqW (st := Hidden.unpin st) hW hK L hleaf r j).symm.trans
      (placedB_linear L C hg hn hi hlo hhi hBe r j hx)
  refine (mulVec_single' _ hBr hk _).trans ?_
  have hrow := rowAt_const_eq hn hsha hhm hi (L.fit n).1
  have hconst := placed_const L [] F.nets C hg hrow k
  have hnk : (Hidden.pinnedCols st.c).count k.val = 0 :=
    List.count_eq_zero.2 fun hm => linear_ne_const L C (hlin _ hm) hn hg hi hkv
  have hpc : ∀ j, Hidden.pinCnt st.c k.val j = 0 := fun j => by rw [Hidden.pinCnt, hnk, Nat.zero_mul]
  have hAk : ∀ j, (stmtOf st hwf regs).A₀ k j = if j = (stmtOf st hwf regs).pin then 1 else 0 := fun j =>
    (Hidden.matA_unpin hda k.val j.val).trans (by
      rw [hpc, Nat.cast_zero, add_zero]
      exact (Integrate.placedA_eqW (st := Hidden.unpin st) hW hK L hleaf k j).symm.trans (hconst j hkv).1)
  have hBk : ∀ j, (stmtOf st hwf regs).B₀ k j = if j = (stmtOf st hwf regs).pin then 1 else 0 := fun j =>
    (Hidden.matB_unpin hdb k.val j.val).trans (by
      rw [hpc, Nat.cast_zero, add_zero]
      exact (Integrate.placedB_eqW (st := Hidden.unpin st) hW hK L hleaf k j).symm.trans (hconst j hkv).2)
  exact (hz.1 o k).symm.trans ((congrArg₂ (· * ·) (Pinned.mulVec_single _ hAk _) (Pinned.mulVec_single _ hBk _)).trans
    ((congrArg₂ (· * ·) (hz.2.1 o) (hz.2.1 o)).trans (one_mul 1)))

end FlockSoundness.Discharge.HiddenExec
