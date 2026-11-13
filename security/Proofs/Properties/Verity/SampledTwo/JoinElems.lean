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
* `readsAt_of_le`: the run's verifier reads the stages' rounds (`readsAt` at rounds `0` to `stage`);
* `usedOf_congr`, `fitOf_congr`: two locators that put the same releases on a round count the same elements there, and
  the same elements that fit. A locator for the whole run that is the stages' locator at the stages' rounds then has
  the stages' locator's `fitOf` there.

Nothing here reads the form's check: `Acc` and `Good` are any predicates, and the locator is any.
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

/-- **Two locators that put the same releases on round `n` count the same elements there.** -/
theorem usedOf_congr {L L' : CoinMap.Locate} {n : ℕ}
    (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n)
    (pre log : List Verity.Release) : CoinMap.usedOf L n pre log = CoinMap.usedOf L' n pre log := by
  induction log generalizing pre with
  | nil => simp only [CoinMap.usedOf]
  | cons r log ih =>
    simp only [CoinMap.usedOf, ih (pre ++ [r])]
    by_cases hl : CoinMap.locAt L pre r.frame r.answers = some n
    · rw [if_pos hl, if_pos ((h pre r.frame r.answers).1 hl)]
    · rw [if_neg hl, if_neg (mt (h pre r.frame r.answers).2 hl)]

/-- **Two locators that put the same releases on round `n` count the same elements that fit in its coin.** -/
theorem fitOf_congr {L L' : CoinMap.Locate} {n : ℕ}
    (h : ∀ pre fr b, CoinMap.locAt L pre fr b = some n ↔ CoinMap.locAt L' pre fr b = some n)
    (pre log : List Verity.Release) : CoinMap.fitOf L n pre log = CoinMap.fitOf L' n pre log := by
  induction log generalizing pre with
  | nil => simp only [CoinMap.fitOf]
  | cons r log ih =>
    have hu : CoinMap.usedOf L n [] pre = CoinMap.usedOf L' n [] pre := usedOf_congr h [] pre
    simp only [CoinMap.fitOf, ih (pre ++ [r]), hu]
    by_cases hl : CoinMap.locAt L pre r.frame r.answers = some n ∧
        CoinMap.usedOf L' n [] pre + CoinMap.widthOf r.frame ≤ 65536
    · rw [if_pos hl, if_pos ⟨(h pre r.frame r.answers).1 hl.1, hl.2⟩]
    · rw [if_neg hl, if_neg fun h' => hl ⟨(h pre r.frame r.answers).2 h'.1, h'.2⟩]

end Verity.SampledTwo
