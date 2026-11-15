module

public import Proofs.Properties.Verity.SampledTwo.DevAgrees
public import Proofs.Properties.Verity.SampledTwo.CoinMap

/-!
# The stages' releases and the run's coins, from the coin map's agreement

`devAgrees_formReading` (`DevAgrees.lean`) takes, for the two stages' rounds, `FirstElems` and `SecondElems`: the
release the coin map reads for a stage's round and the run's coin there have the same elements below some `u`, and the
stage's bytes are at most `16 u`. The coin map's agreement (`CoinMap.ElemAgrees`) gives the same elements below the
count of the round's releases that fit in its coin (`CoinMap.fitOf`). So the two follow at any locator where each
stage's bytes are at most 16 bytes an element that fits:

* `FirstFits`, `SecondFits`: on an accepted play whose log is the device's, the run of record has an input (and an
  interior), and the first stage's bytes (each second stage's slice, `stageWidth`) are at most 16 bytes an element of
  the round's releases that fit;
* `firstElems_of`, `secondElems_of`: `FirstElems` and `SecondElems` at the coin map's release (`CoinMap.dev`) and
  agreement, from them, where the agreement reads the stages' rounds;
* `readsAt_of_le`: the run's verifier reads the stages' rounds (`readsAt` at rounds `0` to `stage`).

Nothing here reads the form's check: `Acc` and `Good` are any predicates, and the locator is any. That a locator for
the whole run has the stages' locator's `fitOf` at the stages' rounds is SESSIONCOINS's (`SessionCoins.fitOf_congr`,
`fitOf_run_stage`, `SessionDev.lean`).
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- **The first stage's bytes fit in round 0's releases**: on a play `Acc` accepts whose log is the device's (`Good`),
the run of record has an input, and its first stage's bytes are at most 16 bytes an element of the releases the
locator `L` puts on round 0 that fit in its coin. -/
def FirstFits (C : CaseOf) (L : CoinMap.Locate) (Acc Good : Verity.Exchange → Prop) : Prop :=
  ∀ o, Acc o → Good o → ∃ inp, inputOn C o = some inp ∧ inp.source.size ≤ 16 * CoinMap.fitOf L 0 [] o.log

/-- **The second stage's bytes fit in rounds `1` to `stage`'s releases**: on a play `Acc` accepts whose log is the
device's, the run of record has an input with an interior, and the second stage's bytes of coin `j` (`stageWidth`) are
at most 16 bytes an element of the releases the locator `L` puts on round `j + 1` that fit in its coin. -/
def SecondFits (A : Live₂) (C : CaseOf) (L : CoinMap.Locate) (Acc Good : Verity.Exchange → Prop) : Prop :=
  ∀ o, Acc o → Good o → ∃ inp it, inputOn C o = some inp ∧ inp.interior = some it ∧
    ∀ j < A.h.stage, stageWidth it.source j ≤ 16 * CoinMap.fitOf L (j + 1) [] o.log

/-- **`FirstElems` at the coin map's release and agreement**, where the agreement reads round 0 and the first stage's
bytes fit in its releases: the elements below what fits are the same, by the agreement. -/
theorem firstElems_of (A : Live₂) (x : Msg) (C : CaseOf) (L : CoinMap.Locate) (reads : ℕ → Prop)
    (Acc Good : Verity.Exchange → Prop) (h0 : reads 0) (hfit : FirstFits C L Acc Good) :
    FirstElems A x C (CoinMap.dev L) Acc Good (CoinMap.ElemAgrees L A x reads) := by
  intro o hA hG r hAg i hi c hc
  obtain ⟨inp, hin, hsz⟩ := hfit o hA hG
  refine ⟨inp, CoinMap.fitOf L 0 [] o.log, hin, hsz, ?_⟩
  intro j hj
  exact hAg i (by rw [hi]; exact h0) c (by rw [hi]; exact hc) j (by rw [hi]; exact hj)

/-- **`SecondElems` at the coin map's release and agreement**, where the agreement reads rounds `1` to `stage` and the
second stage's bytes fit in their releases. -/
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

/-- **The run's verifier reads the stages' rounds**: rounds `0` to `stage` are before the statements' round and the
sessions'. -/
theorem readsAt_of_le (h : SpotFlock.Held) {n : ℕ} (hn : n ≤ h.stage) : readsAt h n := by
  have h2 : h.stage + 2 ≤ SpotFlock.off h h.ms.length := SpotFlock.stage_le_off h h.ms.length
  unfold readsAt
  exact ⟨by omega, by omega⟩

end Verity.SampledTwo
