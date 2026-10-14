import Mathlib
import Proofs.Accounting.Bandwidth.AuditRO

/-!
# The bandwidth audit v0 with relayed tags: the proof

An accepted run with low ingress falls in one of three events, by `correct`, the blocks whose committed tag is right,
and `relayed`, those among them whose contents the chip never queried:

- fewer than `k` correct, and every sampled block among them: `((k − 1)/N)^s` (`AuditRO.card_sample`);
- at least `m` relayed: `C(N, m)·2^c/2^(h·m)` (`card_relay_le`). For a fixed store, an oracle with `m` relayed blocks is
  determined by those blocks, the side bits the chip read and the oracle off the `m` points, because replaying the
  chip from its side bits recomputes the tags (`Chip.runSide_sideTrace`) without querying those points;
- otherwise at least `k − m` queried: `AuditRO`'s compression at `k − m` blocks, for the device with its side sources
  fixed at the oracle (`RelayDevice.atOracle`).
-/

namespace Bandwidth

open Pous (Bits pr)
open scoped ENNReal

variable {N b h S c G q : ℕ}

namespace RelayDevice

/-- The device at one oracle: a `Device` whose side sources hold what the relay device's hold for `O`. -/
def atOracle (A : RelayDevice N b h S c G q) (O : Oracle N b h) : Device N b h S c G q where
  state := A.state
  mem := A.mem
  side E := A.side E O
  chip := A.chip
  bounded := A.bounded

/-- The blocks whose committed tag is the oracle's. -/
def correct (A : RelayDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) : Finset (Fin N) :=
  Finset.univ.filter (fun j => A.tags E O j = O (j, E j))

/-- The blocks with a correct tag whose contents the chip never queried the oracle at. -/
def relayed (A : RelayDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) : Finset (Fin N) :=
  A.correct E O \ queried (A.atOracle O) E O

end RelayDevice

namespace AuditRelay

open RelayDevice

theorem card_relay_le (A : RelayDevice N b h S c G q) (E : Store N b) (m : ℕ)
    [DecidablePred (fun O : Oracle N b h => m ≤ (A.relayed E O).card)] :
    (Finset.univ.filter (fun O : Oracle N b h => m ≤ (A.relayed E O).card)).card * 2 ^ (h * m) ≤
      N.choose m * 2 ^ c * Fintype.card (Oracle N b h) := by
  classical
  by_cases hmN : m ≤ N
  swap
  · rw [Finset.filter_false_of_mem (fun O _ hm => hmN (hm.trans ((Finset.card_le_univ _).trans (by simp))))]
    simp
  set ch := A.chip (A.state E) with hch
  let pad : Oracle N b h → Bits c := fun O i => (ch.sideTrace (A.world E O)).getD i false
  let P : Finset (Fin N) → Finset (Fin N × Bits b) := fun K => K.image (fun j => (j, E j))
  let fibR : Finset (Fin N) × Bits c → Finset (Oracle N b h) := fun p =>
    Finset.univ.filter (fun O => p.1 ⊆ A.relayed E O ∧ pad O = p.2)
  let codes : Finset (Finset (Fin N) × Bits c) := (Finset.univ : Finset (Fin N)).powersetCard m ×ˢ Finset.univ
  have hP : ∀ K, (P K).card = K.card := fun K =>
    Finset.card_image_of_injective _ (fun j₁ j₂ he => (Prod.mk.inj he).1)
  have hs : ∀ O, (fun n => (ch.sideTrace (A.world E O)).getD n false) = AuditRO.stream c (pad O) := by
    intro O
    funext n
    simp only [AuditRO.stream, pad]
    split_ifs with hn
    · rfl
    · have hlen := Chip.sideTrace_length (A.world E O) ch
      have hside : ch.sideBits (A.world E O) ≤ c := (A.bounded (A.state E) (A.world E O)).1
      rw [List.getD_eq_default _ _ (by omega)]
  have hfib : ∀ p ∈ codes, (fibR p).card ≤ (2 ^ h) ^ (N * 2 ^ b - m) := by
    rintro ⟨K, st⟩ hp
    simp only [codes, Finset.mem_product, Finset.mem_powersetCard, Finset.mem_univ, and_true] at hp
    obtain ⟨-, hK⟩ := hp
    calc (fibR (K, st)).card ≤ (Finset.univ : Finset (↥(P K)ᶜ → Bits h)).card := by
          refine Finset.card_le_card_of_injOn (fun O x => O x) (fun _ _ => Finset.mem_univ _) ?_
          intro O₁ h₁ O₂ h₂ he
          simp only [fibR, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at h₁ h₂
          have hoff : ∀ x, x ∉ P K → O₁ x = O₂ x := fun x hx => congrFun he ⟨x, by simpa using hx⟩
          have hq : ∀ x ∈ ch.queries (A.world E O₁), O₂ x = (A.world E O₁).oracle x := by
            intro x hx
            refine (hoff x (fun hxP => ?_)).symm
            obtain ⟨j, hj, rfl⟩ := Finset.mem_image.1 hxP
            exact (Finset.mem_sdiff.1 (h₁.1 hj)).2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, hx⟩)
          have hrun : ch.run (A.world E O₁) = ch.run (A.world E O₂) := by
            rw [← Chip.runSide_sideTrace (A.world E O₁) O₂ ch hq,
              ← Chip.runSide_sideTrace (A.world E O₂) O₂ ch (fun _ _ => rfl), hs O₁, hs O₂, h₁.2, h₂.2]
            rfl
          funext x
          by_cases hx : x ∈ P K
          · obtain ⟨j, hj, rfl⟩ := Finset.mem_image.1 hx
            have r₁ := (Finset.mem_sdiff.1 (h₁.1 hj)).1
            have r₂ := (Finset.mem_sdiff.1 (h₂.1 hj)).1
            simp only [correct, Finset.mem_filter, Finset.mem_univ, true_and] at r₁ r₂
            rw [← r₁, ← r₂]
            exact congrFun hrun j
          · exact hoff x hx
      _ = (2 ^ h) ^ (N * 2 ^ b - m) := by simp [hP, hK]
  have hsub : Finset.univ.filter (fun O : Oracle N b h => m ≤ (A.relayed E O).card) ⊆ codes.biUnion fibR := by
    intro O hO
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hO
    obtain ⟨K, hK, hcard⟩ := Finset.exists_subset_card_eq hO
    refine Finset.mem_biUnion.2 ⟨(K, pad O), ?_, ?_⟩
    · simp [codes, Finset.mem_product, Finset.mem_powersetCard, hcard]
    · simp [fibR, hK]
  have hcodes : codes.card = N.choose m * 2 ^ c := by
    simp [codes, Finset.card_product, Finset.card_powersetCard]
  have hmD : m ≤ N * 2 ^ b := hmN.trans (Nat.le_mul_of_pos_right _ (by positivity))
  have e : (2 ^ h) ^ (N * 2 ^ b - m) * 2 ^ (h * m) = (2 ^ h) ^ (N * 2 ^ b) := by
    rw [pow_mul 2 h m, ← pow_add, Nat.sub_add_cancel hmD]
  have horacle : Fintype.card (Oracle N b h) = (2 ^ h) ^ (N * 2 ^ b) := by simp
  calc (Finset.univ.filter (fun O : Oracle N b h => m ≤ (A.relayed E O).card)).card * 2 ^ (h * m)
      ≤ (codes.card * (2 ^ h) ^ (N * 2 ^ b - m)) * 2 ^ (h * m) := by
        gcongr
        rw [Finset.filter_congr_decidable]
        refine (Finset.card_le_card hsub).trans (Finset.card_biUnion_le.trans ?_)
        exact (Finset.sum_le_card_nsmul _ _ _ hfib).trans (le_of_eq (smul_eq_mul _ _))
    _ = N.choose m * 2 ^ c * Fintype.card (Oracle N b h) := by
        rw [hcodes, horacle, ← e]
        ring

end AuditRelay

end Bandwidth

namespace Bandwidth.SecurityProofs

open Pous (Bits pr)
open scoped ENNReal
open Bandwidth.AuditRO Bandwidth.AuditRelay Bandwidth.RelayDevice

/-- **The bandwidth audit v0 with relayed tags, in the random-oracle model** (`Bandwidth.Guarantees.AuditRelay`). -/
theorem AuditRelay : Bandwidth.Guarantees.AuditRelay := by
  classical
  intro N b h S c G q s k m lam A
  have hsplit : ∀ x : Store N b × Oracle N b h × (Fin s → Fin N),
      A.Accepts x.1 x.2.1 x.2.2 ∧ A.ingress x.1 x.2.1 + S + c + m * b + G * b + lam < k * b →
      (((A.correct x.1 x.2.1).card < k ∧ ∀ i, x.2.2 i ∈ A.correct x.1 x.2.1) ∨
        m ≤ (A.relayed x.1 x.2.1).card) ∨
      (k - m ≤ (queried (A.atOracle x.2.1) x.1 x.2.1).card ∧
        (A.atOracle x.2.1).ingress x.1 x.2.1 + S + c + G * b + lam < (k - m) * b) := by
    rintro ⟨E, O, I⟩ ⟨hacc, hlow⟩
    dsimp only at hacc hlow ⊢
    have hI : ∀ i, I i ∈ A.correct E O := fun i => by simpa [correct] using hacc i
    by_cases hk : (A.correct E O).card < k
    · exact Or.inl (Or.inl ⟨hk, hI⟩)
    by_cases hm : m ≤ (A.relayed E O).card
    · exact Or.inl (Or.inr hm)
    refine Or.inr ⟨?_, ?_⟩
    · have h1 : (A.correct E O).card ≤ (A.relayed E O).card + (queried (A.atOracle O) E O).card :=
        Finset.card_le_card_sdiff_add_card
      omega
    · have e : (k - m) * b = k * b - m * b := Nat.sub_mul k m b
      change A.ingress E O + S + c + G * b + lam < (k - m) * b
      rw [e]
      omega
  have hΩ : Fintype.card (Store N b × Oracle N b h × (Fin s → Fin N)) =
      Fintype.card (Store N b) * (Fintype.card (Oracle N b h) * N ^ s) := by
    simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]
  by_cases hN : N ^ s = 0
  · rw [pr_eq_zero_of_card _ (by rw [hΩ, hN]; simp)]
    exact bot_le
  refine (pr_mono hsplit).trans ?_
  refine (pr_or_le _ _).trans (add_le_add ((pr_or_le _ _).trans (add_le_add ?_ ?_)) ?_)
  · -- the sample misses the blocks without a correct tag
    have hc := pr_le_of_card (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (A.correct x.1 x.2.1).card < k ∧ ∀ i, x.2.2 i ∈ A.correct x.1 x.2.1) (N ^ s) ((k - 1) ^ s) hN (by
      rw [hΩ]
      refine (card_filter_prod_le _ _ (Fintype.card (Oracle N b h) * ((k - 1) ^ s * N ^ s))
        (fun E => ?_)).trans (le_of_eq (by ring))
      refine card_filter_prod_le _ _ _ (fun O => ?_)
      by_cases hk : (A.correct E O).card < k
      · simp only [hk, true_and]
        rw [card_sample]
        exact Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
      · simp [hk])
    refine hc.trans (le_of_eq ?_)
    rw [Nat.cast_pow, Nat.cast_pow, div_eq_mul_inv, div_eq_mul_inv, mul_pow, ENNReal.inv_pow]
  · -- relayed tags
    have hc := pr_le_of_card (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        m ≤ (A.relayed x.1 x.2.1).card) (2 ^ (h * m)) (N.choose m * 2 ^ c) (by positivity) (by
      rw [hΩ]
      refine (card_filter_prod_le _ _ (N.choose m * 2 ^ c * Fintype.card (Oracle N b h) * N ^ s)
        (fun E => ?_)).trans (le_of_eq (by ring))
      refine (card_filter_prod_le' _ _ (N.choose m * 2 ^ c * Fintype.card (Oracle N b h))
        (fun _ => ?_)).trans (le_of_eq (by simp))
      exact card_relay_le A E m)
    refine hc.trans (le_of_eq ?_)
    push_cast
    rfl
  · -- compression at `k − m` blocks
    refine (pr_le_of_card _ (2 ^ lam) (q.choose (k - m)) (by positivity) (by
      rw [hΩ]
      refine (card_filter_prod_le' _ _ (q.choose (k - m) * Fintype.card (Store N b)) (fun y => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]; ring))
      exact card_compress_le (A.atOracle y.1) y.1 (k - m) lam)).trans (le_of_eq ?_)
    push_cast
    rfl

end Bandwidth.SecurityProofs
