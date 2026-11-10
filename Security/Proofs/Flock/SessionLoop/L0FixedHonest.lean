module

public import Proofs.Flock.SessionLoop.L0Fixed
public import Proofs.Flock.SessionLoop.FinalC

/-!
# `L0Fixed` at the honest values

At the honest values the deployed source's masked words below `h_ab` are `vsWord` plus the rep's pads
(`Loop.vsWord_honestVals`, `Masked.srcE_masked_pad`), and from `zp 64` to `h_ab` they are level 0's opening at the coins
plus the same pads (`Masked.ligOodE`, `ligMsgE`, `ligSumsE`). Cancelling the pads gives `vsWord = l0Clear`.

* **`Loop.l0Fixed_honestVals`**: `L0Fixed` holds at the honest values, at every live statement and witness of the
  masked relation.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open FlockComplete FlockComplete.LigLock

namespace Loop

private theorem cancel_pad {a b p : _root_.Flock.F128} (h : a + p = b + p) : a = b := by
  rw [← FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self a p, h,
    FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self]

/-- Word `t` of a masked lane message. -/
private theorem mm_word {src : _root_.FlockComplete.RepSrc} {lay : _root_.Flock.Zk.PadLayout}
    {h : Array _root_.Flock.F128} {i : ℕ}
    {P : _root_.Flock.F256 × _root_.Flock.F256} (e : mmOf src lay i = maskMsg lay h i P) {t : ℕ} (ht : t < 4) :
    src.masked (lay.lgMsg i t) = l0Word (if t < 2 then P.1 else P.2) (t % 2) + h[lay.lgMsg i t]! := by
  obtain rfl | rfl | rfl | rfl : t = 0 ∨ t = 1 ∨ t = 2 ∨ t = 3 := by omega
  · exact congrArg (fun q : _root_.Flock.F256 × _root_.Flock.F256 => q.1.c0) e
  · exact congrArg (fun q : _root_.Flock.F256 × _root_.Flock.F256 => q.1.c1) e
  · exact congrArg (fun q : _root_.Flock.F256 × _root_.Flock.F256 => q.2.c0) e
  · exact congrArg (fun q : _root_.Flock.F256 × _root_.Flock.F256 => q.2.c1) e

/-- Word `t` of a masked pair sum. -/
private theorem e_word {src : _root_.FlockComplete.RepSrc} {lay : _root_.Flock.Zk.PadLayout}
    {h : Array _root_.Flock.F128} {i : ℕ}
    {m : _root_.Flock.F256}
    (e : (⟨src.masked (lay.lgE i 0), src.masked (lay.lgE i 1)⟩ : _root_.Flock.F256) =
      m + ⟨h[lay.lgE i 0]!, h[lay.lgE i 1]!⟩) {t : ℕ} (ht : t < 2) :
    src.masked (lay.lgE i t) = l0Word m t + h[lay.lgE i t]! := by
  obtain rfl | rfl : t = 0 ∨ t = 1 := by omega
  · exact congrArg _root_.Flock.F256.c0 e
  · exact congrArg _root_.Flock.F256.c1 e

/-- **`L0Fixed` holds at the honest values**: the deployed source masks `vsWord` under the rep's pads below `h_ab`, and
from `zp 64` on its masked words are level 0's opening at the coins under the same pads (`ligOodE`, `ligMsgE`,
`ligSumsE`). -/
theorem l0Fixed_honestVals (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hw : Masked.RZk x w) :
    L0Fixed (CoinLayer.session Loop.loopNr Masked.dataE).honestVals x w := by
  intro ν τ c n hn s spec hs j r extras he k hlo hhi
  have hlay : ligLay x j = layE x j := by
    unfold ligLay ligSt
    rw [tabsOf_stAt x j]
    rfl
  have hood : (layE x j).nOod0 = 1 := rfl
  have hpad : ligPad x τ j r = (padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h := by
    unfold ligPad
    rw [hlay]
  have hv : ∀ i, i < (layE x j).hab → (ligSrc x c w τ j r).masked i =
      vsWord (CoinLayer.session Loop.loopNr Masked.dataE).honestVals x w ν c τ j r i + (ligPad x τ j r)[i]! := by
    intro i hi
    rw [vsWord_honestVals x w ν c τ j r (k := i) (by rw [nE_eq_hab]; exact hi), hpad]
    exact srcE_masked_pad x c w τ j r hi
  suffices hs' : (ligSrc x c w τ j r).masked k =
      l0Clear (ligCoin c) (ligP0 x j r) (zkTab x j).sch (ligTab x w τ j) (ligBasis x c w τ j r extras) (layE x j)
        ((UE.tapeZk τ).extra j.val) r.val k + (ligPad x τ j r)[k]! from
    cancel_pad ((hv k hhi).symm.trans hs')
  have hab : (layE x j).hab = (layE x j).zp 64 + 1 + 4 * (zkTab x j).sch.k0 + 4 := rfl
  have hoo : (layE x j).lgOod 0 = (layE x j).zp 64 := rfl
  have hmsg : ∀ i t, (layE x j).lgMsg i t = (layE x j).zp 64 + 1 + 4 * i + t := fun _ _ => rfl
  have hlgE : ∀ i t, (layE x j).lgE i t = (layE x j).zp 64 + 1 + 4 * (zkTab x j).sch.k0 + 2 * i + t :=
    fun _ _ => rfl
  rw [hab] at hhi
  rcases Nat.lt_or_ge k ((layE x j).zp 64 + 1) with h1 | h1
  · obtain rfl : k = (layE x j).lgOod 0 := by rw [hoo]; omega
    have e := ligOodE x hx w hw τ c n hn s spec hs j r extras he
    try dsimp only at e
    rw [hlay] at e
    rw [l0Clear_ood]
    exact e
  rcases Nat.lt_or_ge k ((layE x j).zp 64 + 1 + 4 * (zkTab x j).sch.k0) with h2 | h2
  · obtain ⟨i, t, hi, ht, rfl⟩ : ∃ i t, i < (zkTab x j).sch.k0 ∧ t < 4 ∧ k = (layE x j).lgMsg i t :=
      ⟨(k - ((layE x j).zp 64 + 1)) / 4, (k - ((layE x j).zp 64 + 1)) % 4, by omega, by omega,
        by rw [hmsg]; omega⟩
    have e := ligMsgE x hx w hw τ c n hn s spec hs j r extras he i hi
    try dsimp only at e
    rw [hlay] at e
    rw [l0Clear_msg _ _ _ _ _ _ _ _ hood hi ht]
    exact mm_word e ht
  · obtain ⟨i, t, hi, ht, rfl⟩ : ∃ i t, i < 2 ∧ t < 2 ∧ k = (layE x j).lgE i t :=
      ⟨(k - ((layE x j).zp 64 + 1 + 4 * (zkTab x j).sch.k0)) / 2,
        (k - ((layE x j).zp 64 + 1 + 4 * (zkTab x j).sch.k0)) % 2, by omega, by omega, by rw [hlgE]; omega⟩
    have e := ligSumsE x hx w hw τ c n hn s spec hs j r extras he
    try dsimp only at e
    rw [hlay] at e
    rw [l0Clear_e _ _ _ _ _ _ _ _ hood ht]
    obtain ⟨e0, e1⟩ := e
    obtain rfl | rfl : i = 0 ∨ i = 1 := by omega
    · exact e_word e0 ht
    · exact e_word e1 ht

end Loop

end Claims.ZKServer
