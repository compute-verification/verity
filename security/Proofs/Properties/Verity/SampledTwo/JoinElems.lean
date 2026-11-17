module

public import Proofs.Properties.Verity.SampledTwo.DevAgrees
public import Proofs.Properties.Verity.SampledTwo.CoinMap

/-! `FirstElems` and `SecondElems` from the coin map's agreement, at any locator where each stage's bytes
fit in its round's releases. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims

def FirstFits (C : CaseOf) (L : CoinMap.Locate) (Acc Good : Verity.Exchange → Prop) : Prop :=
  ∀ o, Acc o → Good o → ∃ inp, inputOn C o = some inp ∧ inp.source.size ≤ 16 * CoinMap.fitOf L 0 [] o.log

def SecondFits (A : Live₂) (C : CaseOf) (L : CoinMap.Locate) (Acc Good : Verity.Exchange → Prop) : Prop :=
  ∀ o, Acc o → Good o → ∃ inp it, inputOn C o = some inp ∧ inp.interior = some it ∧
    ∀ j < A.h.stage, stageWidth it.source j ≤ 16 * CoinMap.fitOf L (j + 1) [] o.log

theorem firstElems_of (A : Live₂) (x : Msg) (C : CaseOf) (L : CoinMap.Locate) (reads : ℕ → Prop)
    (Acc Good : Verity.Exchange → Prop) (h0 : reads 0) (hfit : FirstFits C L Acc Good) :
    FirstElems A x C (CoinMap.dev L) Acc Good (CoinMap.ElemAgrees L A x reads) := by
  intro o hA hG r hAg i hi c hc
  obtain ⟨inp, hin, hsz⟩ := hfit o hA hG
  refine ⟨inp, CoinMap.fitOf L 0 [] o.log, hin, hsz, ?_⟩
  intro j hj
  exact hAg i (by rw [hi]; exact h0) c (by rw [hi]; exact hc) j (by rw [hi]; exact hj)

theorem secondElems_of (A : Live₂) (x : Msg) (C : CaseOf) (L : CoinMap.Locate) (reads : ℕ → Prop)
    (Acc Good : Verity.Exchange → Prop) (hr : ∀ n, 0 < n → n ≤ A.h.stage → reads n)
    (hfit : SecondFits A C L Acc Good) :
    SecondElems A x C (CoinMap.dev L) Acc Good (CoinMap.ElemAgrees L A x reads) := by
  intro o hA hG r hAg i hpos hle c hc
  obtain ⟨inp, it, hin, hit, hsz⟩ := hfit o hA hG
  have h1 : i.val - 1 + 1 = i.val := by omega
  have hw := hsz (i.val - 1) (by omega)
  rw [h1] at hw
  refine ⟨inp, it, CoinMap.fitOf L i.val [] o.log, hin, hit, hw, ?_⟩
  intro j hj
  exact hAg i (hr i.val hpos hle) c hc j hj

theorem readsAt_of_le (h : SpotFlock.Held) {n : ℕ} (hn : n ≤ h.stage) : readsAt h n := by
  have h2 : h.stage + 2 ≤ SpotFlock.off h h.ms.length := SpotFlock.stage_le_off h h.ms.length
  unfold readsAt
  exact ⟨by omega, by omega⟩

end Verity.SampledTwo
