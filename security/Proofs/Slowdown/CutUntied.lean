module

public import Proofs.Core.Prob
public import Proofs.Slowdown.CutCharged

/-!
# The charged count with untied words: from cells to the blocks' reads, and over a monitoring period

**The bridge** (`readsOn_untie`): blocks whose reads outside `θ` meet `ReadsOn d P G k T S` while every word is tied,
each block of `G` reading at most `w` words of a set `𝒲` of words of at most `z` gates, all the blocks at most `n` of
them counted with repeats and `G`'s blocks at most `s`, meet, untied at `𝒲`, `ReadsOn` at `k + z w`, `T + z n` and
`S + z (2 k + z w) s`. A block reads, untied, at most its tied reads `r` and `z` gates for each word of `𝒲` it reads,
`v` (`Untied.card_unregistered_untie_words_le`); `r ≤ k` and `v ≤ w` on `G` give `(r + z v)² ≤ r² + z (2 k + z w) v`.
With each word read by at most one block and at most `U` words, `n` is `U` and `s` is `min(U, |G| w)`
(`readsOn_untie_of_untiedWords`), and the blocks outside `G` that each read more than `w` words are at most
`U / (w + 1)` (`card_charged_le`).

**The period** (`HEChargedUntied`): on each outcome where at most `U` words are untied, the bridge gives `HECharged`'s
premise at the untied circuit, so its conclusion holds there, and so does the count of charged blocks; an outcome where
the period is accepted and they fail is one where it is accepted with more than `U` words untied, whose probability the
premise bounds (`Verity.Game.prPMFReal_mono`). The witnesses are at the toy (`HEChargedUntied_honest`,
`HEChargedUntied_nontrivial`).
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.Cut.SecurityProofs

open Verity Finset

/-- **The bridge from untied words to the blocks' reads.** Tied, the blocks meet `ReadsOn d P G k T S`; each word of
`𝒲` has at most `z` gates, each block of `G` reads at most `w` of them, all the blocks read at most `n` of them,
counted with repeats, and `G`'s blocks at most `s`. Untied at `𝒲`, they meet `ReadsOn` at `k + z w`, `T + z n` and
`S + z (2 k + z w) s`. -/
theorem readsOn_untie {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) (G : Finset (Fin m)) {k T S z w n s : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (hz : ∀ E ∈ 𝒲, E.card ≤ z) (hR : ReadsOn d P G k T S)
    (hw : ∀ u ∈ G, (wordsRead d.D 𝒲 (P.gates u)).card ≤ w)
    (hn : ∑ u, (wordsRead d.D 𝒲 (P.gates u)).card ≤ n) (hs : ∑ u ∈ G, (wordsRead d.D 𝒲 (P.gates u)).card ≤ s) :
    ReadsOn (d.untie (𝒲.biUnion id)) P G (k + z * w) (T + z * n) (S + z * (2 * k + z * w) * s) := by
  obtain ⟨hk, hT, hS⟩ := hR
  have hu : ∀ u, ((d.untie (𝒲.biUnion id)).unregistered (P.gates u)).card ≤
      (d.unregistered (P.gates u)).card + z * (wordsRead d.D 𝒲 (P.gates u)).card :=
    fun u => Untied.card_unregistered_untie_words_le d 𝒲 hz (P.gates u)
  refine ⟨fun u hu' => (hu u).trans (Nat.add_le_add (hk u hu') (Nat.mul_le_mul_left z (hw u hu'))), ?_, ?_⟩
  · calc ∑ u, ((d.untie (𝒲.biUnion id)).unregistered (P.gates u)).card
        ≤ ∑ u, ((d.unregistered (P.gates u)).card + z * (wordsRead d.D 𝒲 (P.gates u)).card) :=
          sum_le_sum fun u _ => hu u
      _ = ∑ u, (d.unregistered (P.gates u)).card + z * ∑ u, (wordsRead d.D 𝒲 (P.gates u)).card := by
          rw [sum_add_distrib, mul_sum]
      _ ≤ T + z * n := Nat.add_le_add hT (Nat.mul_le_mul_left z hn)
  · have hsq : ∀ u ∈ G, ((d.untie (𝒲.biUnion id)).unregistered (P.gates u)).card ^ 2 ≤
        (d.unregistered (P.gates u)).card ^ 2 + z * (2 * k + z * w) * (wordsRead d.D 𝒲 (P.gates u)).card := by
      intro u hu'
      have hr := hk u hu'
      have hv := hw u hu'
      calc ((d.untie (𝒲.biUnion id)).unregistered (P.gates u)).card ^ 2
          ≤ ((d.unregistered (P.gates u)).card + z * (wordsRead d.D 𝒲 (P.gates u)).card) ^ 2 :=
            Nat.pow_le_pow_left (hu u) 2
        _ = (d.unregistered (P.gates u)).card ^ 2 +
              z * (2 * (d.unregistered (P.gates u)).card + z * (wordsRead d.D 𝒲 (P.gates u)).card) *
                (wordsRead d.D 𝒲 (P.gates u)).card := by ring
        _ ≤ (d.unregistered (P.gates u)).card ^ 2 + z * (2 * k + z * w) * (wordsRead d.D 𝒲 (P.gates u)).card := by
            gcongr
    calc ∑ u ∈ G, ((d.untie (𝒲.biUnion id)).unregistered (P.gates u)).card ^ 2
        ≤ ∑ u ∈ G, ((d.unregistered (P.gates u)).card ^ 2 +
            z * (2 * k + z * w) * (wordsRead d.D 𝒲 (P.gates u)).card) := sum_le_sum hsq
      _ = ∑ u ∈ G, (d.unregistered (P.gates u)).card ^ 2 +
            z * (2 * k + z * w) * ∑ u ∈ G, (wordsRead d.D 𝒲 (P.gates u)).card := by
          rw [sum_add_distrib, mul_sum]
      _ ≤ S + z * (2 * k + z * w) * s := Nat.add_le_add hS (Nat.mul_le_mul_left _ hs)

/-- **The bridge at `UntiedWords`**: with each untied word read by at most one block, the words any blocks read,
counted with repeats, are at most the untied words, and `G`'s blocks read at most `w` each; so at most `U` of them give
`ReadsOn` at `k + z w`, `T + z U` and `S + z (2 k + z w) min(U, |G| w)`. -/
theorem readsOn_untie_of_untiedWords {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) (G : Finset (Fin m))
    {k T S z w U : ℕ} (𝒲 : Finset (Finset (Fin d.D.N))) (hW : UntiedWords d P G 𝒲 z w) (hU : 𝒲.card ≤ U)
    (hR : ReadsOn d P G k T S) :
    ReadsOn (d.untie (𝒲.biUnion id)) P G (k + z * w) (T + z * U) (S + z * (2 * k + z * w) * min U (G.card * w)) :=
  readsOn_untie d P G 𝒲 hW.1 hR hW.2.2 ((Untied.sum_card_wordsRead_le P 𝒲 hW.2.1 univ).trans hU)
    (le_min ((Untied.sum_card_wordsRead_le P 𝒲 hW.2.1 G).trans hU)
      ((sum_le_card_nsmul G _ w hW.2.2).trans_eq (smul_eq_mul _ _)))

/-- **The charged blocks are few**: when each of at most `U` words is read by at most one block, the blocks outside
`G` that each read more than `w` of them are at most `U / (w + 1)`. -/
theorem card_charged_le {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) (G : Finset (Fin m)) {w U : ℕ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (hone : OneReader P 𝒲) (hU : 𝒲.card ≤ U)
    (hover : ∀ u ∈ Gᶜ, w < (wordsRead d.D 𝒲 (P.gates u)).card) : (w + 1) * Gᶜ.card ≤ U :=
  (Nat.mul_le_mul_left _ (card_le_card fun u hu => by
      simp only [Untied.readersOver, mem_filter, mem_univ, true_and]; exact hover u hu)).trans
    ((Untied.card_readersOver_le P 𝒲 hone w).trans hU)

theorem HEChargedUntied : Guarantees.HEChargedUntied := by
  intro Ω ι V _ _ _ _ _ d W val k m c γ T S z w U P G untied accepted βw βx b ε hw hx hb hval hshare hH hE hR hU hlim
  refine le_trans (Verity.Game.prPMFReal_mono _ fun x hbad => ⟨hbad.1, not_le.1 fun hle => hbad.2 ?_⟩) hlim
  obtain ⟨hN, hγ⟩ := HECharged ι V (d.untie ((untied x).biUnion id)) (W x) (val x) (k + z * w) m c γ (T + z * U)
    (S + z * (2 * k + z * w) * min U ((G x).card * w)) P (G x) βw βx b hw hx hb (hval x) (hshare x) (hH x) (hE x)
    (readsOn_untie_of_untiedWords d P (G x) (untied x) (hU x) hle (hR x))
  exact ⟨hN, hγ, card_charged_le d P (G x) (untied x) (hU x).2.1 hle⟩

theorem HEChargedUntied_honest : Guarantees.HEChargedUntied_honest := by
  obtain ⟨hval, hshare, hH, hE, hR, -, -⟩ := HECharged_honest
  refine ⟨fun _ => hval, fun _ => hshare, fun _ => Untied.hardness_untie Toy.d _ _ _ _ _ hH, fun _ => hE,
    fun _ => hR, fun _ => ⟨fun _ h => absurd h (notMem_empty _), fun _ h => absurd h (notMem_empty _),
      fun u _ => by simp [wordsRead]⟩, ?_⟩
  unfold Slowdown.Assumptions.PeriodUntiedAtMost
  rw [Verity.Game.prPMFReal_congr _ fun _ => show (True ∧ 0 < (∅ : Finset (Finset (Fin Toy.C.N))).card) ↔ False by
    simp, Verity.Game.prPMFReal_false]

/-- Untied at its three input words, the toy with them registered has the input gates outside `θ` of the toy with
nothing registered. -/
theorem toyReg_unreg :
    unregisteredInputs (Toy.dReg.untie (Toy.inputWords.biUnion id)) = unregisteredInputs Toy.d := by decide

/-- (H) holds at the untied circuit, at costs 1 and every `k`. -/
theorem toyReg_hardness_untied (k : ℕ) :
    Assumptions.Hardness (Toy.dReg.untie (Toy.inputWords.biUnion id)) Toy.arrays Toy.val ![1, 1] k := by
  intro S hS
  have hκ : kappaWords (Toy.dReg.untie (Toy.inputWords.biUnion id)) S = kappaWords Toy.d S :=
    congrArg (fun X : Finset (Fin Toy.C.N) => kappa Toy.C X (S.biUnion id)) toyReg_unreg
  exact (toy_hardness k S ((le_of_eq hκ.symm).trans hS)).trans (Nat.cast_le.2 (le_of_eq hκ.symm))

theorem HEChargedUntied_nontrivial : Guarantees.HEChargedUntied_nontrivial := by
  have hfail : ¬ ((Toy.W.N : ℝ) ≤ ((∑ u ∈ (univ : Finset (Fin 1))ᶜ, (Toy.W.inside (Toy.P1.gates u)).card : ℕ) : ℝ) +
      workHEOn (0 + 1 * 0) (0 + 1 * (2 * 0 + 1 * 3) * min 0 ((univ : Finset (Fin 1)).card * 3)) (1 * 1) 1
        Toy.W.O) := by
    rw [toy_N, compl_univ, sum_empty, Nat.min_eq_left (Nat.zero_le _)]
    norm_num [workHEOn]
  refine ⟨⟨()⟩, one_pos, fun _ => toy_values, fun _ => toy_noSharedPair, fun _ => toyReg_hardness_untied _,
    fun _ => toy_exports 1, fun _ => by unfold ReadsOn; decide,
    fun _ => ⟨by decide, fun _ _ u v _ _ => Subsingleton.elim u v, by decide⟩, fun h => ?_, fun h => ?_⟩
  · unfold Slowdown.Assumptions.PeriodUntiedAtMost at h
    have h1 : (1 : ℝ) ≤ 0 := le_trans (le_of_eq (Verity.Game.prPMFReal_true _).symm)
      (le_trans (Verity.Game.prPMFReal_mono _ fun _ _ => ⟨trivial, by show 0 < Toy.inputWords.card; decide⟩) h)
    norm_num at h1
  · have h1 : (1 : ℝ) ≤ 0 := le_trans (le_of_eq (Verity.Game.prPMFReal_true _).symm)
      (le_trans (Verity.Game.prPMFReal_mono _ fun _ _ => ⟨trivial, fun h' => hfail h'.1⟩) h)
    norm_num at h1

end Slowdown.Cut.SecurityProofs
