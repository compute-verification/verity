import Definitions.Nci.Reads
import Proofs.Nci.Pull
import Proofs.Nci.Threads

/-!
# Reads in a set of gates, the pull into the contained circuit, and Theorem 1 over a read measure

* `novelReads_le_readsIn`: what a set reads novelly is at most all it reads, `readsIn d R univ`.
* `novelReads_pull_le_readsIn`: through a containment `M : c ≼ d`, a piece of a unit of `d` reads novelly at most what
  its unit reads in `M`'s image, baseline included, so no condition on the baselines is needed: a value the piece reads
  is a gate of `c` outside the piece or an input of `c`, and `M` sends either to a gate of its image that the unit reads.
* `novelReads_piece_le_readsIn`: the same through a partial containment, for the piece of `S` a unit computes, with the
  image of `S` and of the gates `S` reads in place of `M`'s image.
* `threaded_reads_by`: Theorem 1 (`threaded_reads`) with reads counted by any measure `ρ`. Its proof reads the count
  twice, through `ρ R ≤ S` and through `ρ R` paying for the weights used and the thread values read, so it goes through
  for any `ρ`; `threaded_count` (`Proofs.Nci.Threads`) is the counting. Theorem 1 is the instance `ρ = novelReads`.
-/

namespace Nci.SecurityProofs

open Verity Finset

theorem novelReads_le_readsIn (d : Registered) (R : Finset (Fin d.D.N)) : novelReads d R ≤ readsIn d R univ := by
  unfold novelReads readsIn
  rw [inter_univ]
  exact sum_le_sum_of_subset sdiff_subset

/-- **The pull into the contained circuit**: through a containment `M : c ≼ d`, the piece of `c` that a unit of `d`
induces (`pull`) reads at most what the unit reads in `M`'s image, baseline included. -/
theorem novelReads_pull_le_readsIn {c d : Registered} (M : Containment c.D d.D) {n : ℕ} (P : Partition d.D n)
    (u : Fin n) : novelReads c ((pull M P).gates u) ≤ readsIn d (P.gates u) (univ.map ⟨M.emb, M.emb_inj⟩) := by
  unfold novelReads readsIn
  calc c.D.width (c.D.into ((pull M P).gates u) \ c.θ) ≤ c.D.width (c.D.into ((pull M P).gates u)) :=
        sum_le_sum_of_subset sdiff_subset
    _ = d.D.width ((c.D.into ((pull M P).gates u)).map ⟨M.emb, M.emb_inj⟩) := width_map (M := M) _
    _ ≤ d.D.width (d.D.into (P.gates u) ∩ univ.map ⟨M.emb, M.emb_inj⟩) := by
        refine sum_le_sum_of_subset fun b hb => ?_
        obtain ⟨a, -, rfl⟩ := mem_map.1 hb
        exact mem_inter.2 ⟨into_pull_map u hb, mem_map_of_mem _ (mem_univ a)⟩

/-- **The pull through a partial containment**: the piece of `c` that a unit of `d` computes (`piece`) reads novelly at
most what the unit reads in the image of `S` and of the gates `S` reads, baseline included: a value the piece reads is a
gate of `S` or a gate outside `S` that `S` reads, and `M` sends either to a gate the unit reads, of the same width. -/
theorem novelReads_piece_le_readsIn {c d : Registered} (M : PartialContainment c d) {n : ℕ} (P : Partition d.D n)
    (u : Fin n) : novelReads c (piece M P u) ≤ readsIn d (P.gates u) ((M.S ∪ c.D.into M.S).image M.emb) := by
  have hA : ∀ a ∈ c.D.into (piece M P u) \ c.θ, a ∈ M.S ∪ c.D.into M.S := fun a ha =>
    into_piece_subset u (mem_sdiff.1 ha).1
  have hmaps : ∀ a ∈ c.D.into (piece M P u) \ c.θ,
      M.emb a ∈ d.D.into (P.gates u) ∩ (M.S ∪ c.D.into M.S).image M.emb := by
    intro a ha
    refine mem_inter.2 ⟨?_, mem_image_of_mem _ (hA a ha)⟩
    have hain := (mem_sdiff.1 ha).1
    simp only [Circuit.into, mem_filter, mem_univ, true_and] at hain
    obtain ⟨hna, g, hg, hag⟩ := hain
    obtain ⟨hgS, hgu⟩ := mem_piece.1 hg
    simp only [Circuit.into, mem_filter, mem_univ, true_and]
    refine ⟨fun hau => ?_, M.emb g, hgu, ?_⟩
    · by_cases haS : a ∈ M.S
      · exact hna (mem_piece.2 ⟨haS, hau⟩)
      · have hc := P.gates_subset_computed u hau
        simp only [Circuit.computed, mem_filter, mem_univ, true_and] at hc
        rw [M.input_emb a ((mem_union.1 (hA a ha)).resolve_left haS)] at hc
        exact Bool.false_ne_true hc.symm
    · rw [M.reads_emb g hgS]
      exact mem_image_of_mem _ hag
  unfold novelReads readsIn Circuit.width
  calc ∑ a ∈ c.D.into (piece M P u) \ c.θ, c.D.w a
      = ∑ b ∈ (c.D.into (piece M P u) \ c.θ).image M.emb, d.D.w b := by
        rw [sum_image fun a ha b hb h => M.emb_inj a (hA a (mem_coe.1 ha)) b (hA b (mem_coe.1 hb)) h]
        exact sum_congr rfl fun a ha => (M.emb_w a (hA a ha)).symm
    _ ≤ ∑ b ∈ d.D.into (P.gates u) ∩ (M.S ∪ c.D.into M.S).image M.emb, d.D.w b :=
        sum_le_sum_of_subset (image_subset_iff.2 hmaps)

/-- **Theorem 1 over a read measure** `ρ`: for a `(d, e)`-threaded `c` whose small subcircuits pay for the weights they
use and the thread values they read, both read by `ρ` (small: at most `Γ` computed gates and `ρ R ≤ S`), a small
subcircuit `R` holds at most `ρ R Λ / b_a` counted multiplications, `Λ = T.bound S b_w d e`. -/
theorem threaded_reads_by (c : Registered) (T : Threading c) (ρ : Finset (Fin c.D.N) → ℝ) (Γ : ℕ) (S b_w b_a : ℝ)
    (d e : ℕ) (hbw : 0 < b_w) (hba : 0 < b_a) (hd : 0 < d)
    (hT : ∀ R, R ⊆ c.D.computed ∧ c.D.size R ≤ Γ ∧ ρ R ≤ S → ∀ t, (T.threadReads R t).card < d →
      (T.threadMults R t).card ≤ e * (T.threadReads R t).card)
    (hW : ∀ R, R ⊆ c.D.computed ∧ c.D.size R ≤ Γ ∧ ρ R ≤ S →
      b_w * (T.weightsUsed R).card + b_a * ∑ t, (T.threadReads R t).card ≤ ρ R)
    (R : Finset (Fin c.D.N)) (hR : R ⊆ c.D.computed ∧ c.D.size R ≤ Γ ∧ ρ R ≤ S) :
    b_a * (T.mults R).card ≤ ρ R * T.bound S b_w d e :=
  threaded_count T R hbw hba hd (hT R hR) (hW R hR) hR.2.2

example : Guarantees.ThreadedReads := by
  intro c T Γ S b_w b_a d e hbw hba hd hT hW
  exact threaded_reads_by c T (fun R => (novelReads c R : ℝ)) Γ S b_w b_a d e hbw hba hd hT hW

end Nci.SecurityProofs
