import Mathlib
import Proofs.Accounting.RateAudit.Audit

/-!
# The rate audit with relayed tags: the proof

`Bandwidth.SecurityProofs.AuditRelay`'s argument on the rate audit's game, for a `RelayDevice` whose side sources may
depend on the oracle. An accepted run whose compute is within `F` and whose footprint is short of the floor at `k − m`
words falls in one of three events, by `correct`, the words whose committed tag is right, and `relayed`, those among
them whose contents the chip never queried under the stream:

- fewer than `k` correct, and every sampled word among them: `((k − 1)/N)^s` (`Bandwidth.AuditRO.card_sample`);
- at least `m` relayed: `C(N, m)·2^L/2^(h·m)` (`card_relay_le`). For a fixed store and stream, an oracle with `m`
  relayed words is determined by those words, the side bits the chip read and the oracle off the `m` points, because
  replaying the chip from its side bits recomputes the tags (`Chip.runSide_sideTrace`) without querying those points;
- otherwise at least `k − m` queried: `RateAudit`'s compression at `k − m` words (`card_compress_le`), which never
  reads the side sources.
-/

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal

namespace Chip

variable {N b h ρ : ℕ} {α : Type}

/-- The side-source bits a run read, in order. -/
def sideTrace (w : World N b h ρ) : Chip N b h ρ α → List Bool
  | done _ => []
  | tag x k => (k (w.oracle x)).sideTrace w
  | ingress a k => (k (w.mem a)).sideTrace w
  | side a k => w.side a :: (k (w.side a)).sideTrace w
  | stream i k => (k (w.stream i)).sideTrace w
  | rebuild j k => (k (w.store j)).sideTrace w

/-- A run's value, recomputed from an oracle, device memory, the certified stream, the store and a stream of the
side-source bits it read. -/
def runSide (O : Oracle N b h ρ) (mem : ℕ → Bool) (R : Bits ρ) (E : Payloads N b) : Chip N b h ρ α → (ℕ → Bool) → α
  | done a, _ => a
  | tag x k, f => (k (O x)).runSide O mem R E f
  | ingress a k, f => (k (mem a)).runSide O mem R E f
  | side _ k, f => (k (f 0)).runSide O mem R E (fun n => f (n + 1))
  | stream i k, f => (k (R i)).runSide O mem R E f
  | rebuild j k, f => (k (E j)).runSide O mem R E f

/-- A run reads at most its link bits from the side sources. -/
theorem sideTrace_length_le (w : World N b h ρ) (c : Chip N b h ρ α) : (c.sideTrace w).length ≤ c.linkBits w := by
  induction c with
  | done a => simp [sideTrace, linkBits]
  | tag x k ih => simpa [sideTrace, linkBits] using ih _
  | ingress a k ih => simpa [sideTrace, linkBits] using ih _
  | side a k ih => simpa [sideTrace, linkBits] using ih _
  | stream i k ih => exact (ih _).trans (Nat.le_succ _)
  | rebuild j k ih => simpa [sideTrace, linkBits] using ih _

/-- Given device memory, the stream and the store, the run is a function of the oracle at its queries and the side bits
it read: any oracle that agrees with the world's at the run's queries recomputes it from those bits. -/
theorem runSide_sideTrace (w : World N b h ρ) (O' : Oracle N b h ρ) (c : Chip N b h ρ α)
    (ho : ∀ x ∈ c.queries w, O' x = w.oracle x) :
    c.runSide O' w.mem w.stream w.store (fun n => (c.sideTrace w).getD n false) = c.run w := by
  induction c with
  | done a => simp [runSide, run]
  | tag x k ih =>
    simp only [queries, List.mem_cons, forall_eq_or_imp] at ho
    obtain ⟨hx, ho⟩ := ho
    simp only [runSide, run, sideTrace, hx]
    exact ih _ ho
  | ingress a k ih => exact ih _ ho
  | side a k ih =>
    simp only [runSide, run, sideTrace, List.getD_cons_zero, List.getD_cons_succ]
    exact ih _ ho
  | stream i k ih => exact ih _ ho
  | rebuild j k ih => exact ih _ ho

end Chip

variable {N b h ρ S L q : ℕ}

namespace RelayDevice

/-- The words whose committed tag is the oracle's under the stream. -/
def correct (A : RelayDevice N b h ρ S L q) (E : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) : Finset (Fin N) :=
  Finset.univ.filter (fun j => A.tags E R O j = O (R, j, E j))

/-- The words with a correct tag whose contents the chip never queried the oracle at, under the stream. -/
def relayed (A : RelayDevice N b h ρ S L q) (E : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) : Finset (Fin N) :=
  A.correct E R O \ queried A E R O

end RelayDevice

open RelayDevice

theorem card_relay_le (A : RelayDevice N b h ρ S L q) (E : Payloads N b) (R : Bits ρ) (m : ℕ)
    [DecidablePred (fun O : Oracle N b h ρ => m ≤ (A.relayed E R O).card)] :
    (Finset.univ.filter (fun O : Oracle N b h ρ => m ≤ (A.relayed E R O).card)).card * 2 ^ (h * m) ≤
      N.choose m * 2 ^ L * Fintype.card (Oracle N b h ρ) := by
  classical
  by_cases hmN : m ≤ N
  swap
  · rw [Finset.filter_false_of_mem (fun O _ hm => hmN (hm.trans ((Finset.card_le_univ _).trans (by simp))))]
    simp
  set D := Fintype.card (Bits ρ × Fin N × Bits b) with hD
  set ch := A.chip (A.state E) with hch
  let pad : Oracle N b h ρ → Bits L := fun O i => (ch.sideTrace (A.world E R O)).getD i false
  let P : Finset (Fin N) → Finset (Bits ρ × Fin N × Bits b) := fun K => K.image (fun j => (R, j, E j))
  let fibR : Finset (Fin N) × Bits L → Finset (Oracle N b h ρ) := fun p =>
    Finset.univ.filter (fun O => p.1 ⊆ A.relayed E R O ∧ pad O = p.2)
  let codes : Finset (Finset (Fin N) × Bits L) := (Finset.univ : Finset (Fin N)).powersetCard m ×ˢ Finset.univ
  have hP : ∀ K, (P K).card = K.card := fun K =>
    Finset.card_image_of_injective _ (fun j₁ j₂ he => (Prod.mk.inj (Prod.mk.inj he).2).1)
  have hs : ∀ O, (fun n => (ch.sideTrace (A.world E R O)).getD n false) = Bandwidth.AuditRO.stream L (pad O) := by
    intro O
    funext n
    simp only [Bandwidth.AuditRO.stream, pad]
    split_ifs with hn
    · rfl
    · have hlen := Chip.sideTrace_length_le (A.world E R O) ch
      have hside : ch.linkBits (A.world E R O) ≤ L := (A.bounded (A.state E) (A.world E R O)).1
      rw [List.getD_eq_default _ _ (by omega)]
  have hfib : ∀ p ∈ codes, (fibR p).card ≤ (2 ^ h) ^ (D - m) := by
    rintro ⟨K, st⟩ hp
    simp only [codes, Finset.mem_product, Finset.mem_powersetCard, Finset.mem_univ, and_true] at hp
    obtain ⟨-, hK⟩ := hp
    calc (fibR (K, st)).card ≤ (Finset.univ : Finset (↥(P K)ᶜ → Bits h)).card := by
          refine Finset.card_le_card_of_injOn (fun O x => O x) (fun _ _ => Finset.mem_univ _) ?_
          intro O₁ h₁ O₂ h₂ he
          simp only [fibR, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at h₁ h₂
          have hoff : ∀ x, x ∉ P K → O₁ x = O₂ x := fun x hx => congrFun he ⟨x, by simpa using hx⟩
          have hq : ∀ x ∈ ch.queries (A.world E R O₁), O₂ x = (A.world E R O₁).oracle x := by
            intro x hx
            refine (hoff x (fun hxP => ?_)).symm
            obtain ⟨j, hj, rfl⟩ := Finset.mem_image.1 hxP
            exact (Finset.mem_sdiff.1 (h₁.1 hj)).2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, hx⟩)
          have hrun : ch.run (A.world E R O₁) = ch.run (A.world E R O₂) := by
            rw [← Chip.runSide_sideTrace (A.world E R O₁) O₂ ch hq,
              ← Chip.runSide_sideTrace (A.world E R O₂) O₂ ch (fun _ _ => rfl), hs O₁, hs O₂, h₁.2, h₂.2]
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
      _ = (2 ^ h) ^ (D - m) := by simp [hP, hK, hD]
  have hsub : Finset.univ.filter (fun O : Oracle N b h ρ => m ≤ (A.relayed E R O).card) ⊆ codes.biUnion fibR := by
    intro O hO
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hO
    obtain ⟨K, hK, hcard⟩ := Finset.exists_subset_card_eq hO
    refine Finset.mem_biUnion.2 ⟨(K, pad O), ?_, ?_⟩
    · simp [codes, Finset.mem_product, Finset.mem_powersetCard, hcard]
    · simp [fibR, hK]
  have hcodes : codes.card = N.choose m * 2 ^ L := by
    simp [codes, Finset.card_product, Finset.card_powersetCard]
  have hmD : m ≤ D := by
    have : D = 2 ^ ρ * (N * 2 ^ b) := by simp [hD]
    rw [this]
    exact hmN.trans ((Nat.le_mul_of_pos_right _ (by positivity)).trans (Nat.le_mul_of_pos_left _ (by positivity)))
  have e : (2 ^ h) ^ (D - m) * 2 ^ (h * m) = (2 ^ h) ^ D := by
    rw [pow_mul 2 h m, ← pow_add, Nat.sub_add_cancel hmD]
  have horacle : Fintype.card (Oracle N b h ρ) = (2 ^ h) ^ D := by simp [hD]
  calc (Finset.univ.filter (fun O : Oracle N b h ρ => m ≤ (A.relayed E R O).card)).card * 2 ^ (h * m)
      ≤ (codes.card * (2 ^ h) ^ (D - m)) * 2 ^ (h * m) := by
        gcongr
        rw [Finset.filter_congr_decidable]
        refine (Finset.card_le_card hsub).trans (Finset.card_biUnion_le.trans ?_)
        exact (Finset.sum_le_card_nsmul _ _ _ hfib).trans (le_of_eq (smul_eq_mul _ _))
    _ = N.choose m * 2 ^ L * Fintype.card (Oracle N b h ρ) := by
        rw [hcodes, horacle, ← e]
        ring

end Accounting.Rate

namespace Accounting.SecurityProofs

open Accounting.Rate Accounting.Rate.RelayDevice
open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (card_filter_prod_le card_filter_prod_le' pr_le_of_card pr_mono pr_or_le pr_eq_zero_of_card)
open Bandwidth.AuditRO (card_sample)

/-- The rate audit's bound with relayed tags, for a relay device whose side sources may depend on the oracle: the floor
at `k − m` words, at `((k − 1)/N)^s + C(N, m)·2^L/2^(h·m) + C(q, k − m)/2^λ`. -/
theorem _root_.Accounting.Rate.RelayDevice.auditRelay {N b h ρ S L q : ℕ} (A : RelayDevice N b h ρ S L q)
    (s k m F cTag cEnc lam : ℕ) :
    pr (fun x : Payloads N b × (Bits ρ × Oracle N b h ρ × (Fin s → Fin N)) =>
        A.Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧ A.compute cTag cEnc x.1 x.2.1 x.2.2.1 ≤ F ∧
          A.footprint x.1 x.2.1 x.2.2.1 * cEnc < floor b ρ S L F cTag cEnc (k - m) lam)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m)
        + ((q.choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  classical
  rcases Nat.eq_zero_or_pos (k - m) with hkm | hk0
  · refine (pr_le_of_card _ 1 0 one_ne_zero ?_).trans (by simp)
    rw [Finset.filter_false_of_mem (fun x _ hx => by simp [floor, hkm] at hx), Finset.card_empty, zero_mul,
      zero_mul]
  have hsplit : ∀ x : Payloads N b × (Bits ρ × Oracle N b h ρ × (Fin s → Fin N)),
      (A.Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧ A.compute cTag cEnc x.1 x.2.1 x.2.2.1 ≤ F ∧
        A.footprint x.1 x.2.1 x.2.2.1 * cEnc < floor b ρ S L F cTag cEnc (k - m) lam) →
      (((A.correct x.1 x.2.1 x.2.2.1).card < k ∧ ∀ i, x.2.2.2 i ∈ A.correct x.1 x.2.1 x.2.2.1) ∨
        m ≤ (A.relayed x.1 x.2.1 x.2.2.1).card) ∨
      (k - m ≤ (queried A x.1 x.2.1 x.2.2.1).card ∧ taken A x.1 x.2.1 x.2.2.1 + lam < (k - m) * b + ρ) := by
    rintro ⟨E, R, O, I⟩ ⟨hacc, hc, hf⟩
    dsimp only at hacc hc hf ⊢
    have hI : ∀ i, I i ∈ A.correct E R O := fun i => by simpa [correct] using hacc i
    by_cases hk : (A.correct E R O).card < k
    · exact Or.inl (Or.inl ⟨hk, hI⟩)
    by_cases hm : m ≤ (A.relayed E R O).card
    · exact Or.inl (Or.inr hm)
    have hq : k - m ≤ (queried A E R O).card := by
      have h1 : (A.correct E R O).card ≤ (A.relayed E R O).card + (queried A E R O).card :=
        Finset.card_le_card_sdiff_add_card
      omega
    exact Or.inr ⟨hq, taken_lt_of_floor A E R O hq hc hf⟩
  have hΩ : Fintype.card (Payloads N b × (Bits ρ × Oracle N b h ρ × (Fin s → Fin N))) =
      Fintype.card (Payloads N b) * (Fintype.card (Bits ρ) * (Fintype.card (Oracle N b h ρ) * N ^ s)) := by
    simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]
  by_cases hN : N ^ s = 0
  · rw [pr_eq_zero_of_card _ (by rw [hΩ, hN]; simp)]
    exact bot_le
  refine (pr_mono hsplit).trans ?_
  refine (pr_or_le _ _).trans (add_le_add ((pr_or_le _ _).trans (add_le_add ?_ ?_)) ?_)
  · -- the sample misses the words without a correct tag
    have hc := pr_le_of_card (fun x : Payloads N b × (Bits ρ × Oracle N b h ρ × (Fin s → Fin N)) =>
        (A.correct x.1 x.2.1 x.2.2.1).card < k ∧ ∀ i, x.2.2.2 i ∈ A.correct x.1 x.2.1 x.2.2.1) (N ^ s)
        ((k - 1) ^ s) hN (by
      rw [hΩ]
      refine (card_filter_prod_le _ _
        (Fintype.card (Bits ρ) * (Fintype.card (Oracle N b h ρ) * ((k - 1) ^ s * N ^ s)))
        (fun E => ?_)).trans (le_of_eq (by ring))
      refine card_filter_prod_le _ _ _ (fun R => ?_)
      refine card_filter_prod_le _ _ _ (fun O => ?_)
      by_cases hk : (A.correct E R O).card < k
      · simp only [hk, true_and]
        rw [card_sample]
        exact Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
      · simp [hk])
    refine hc.trans (le_of_eq ?_)
    rw [Nat.cast_pow, Nat.cast_pow, div_eq_mul_inv, div_eq_mul_inv, mul_pow, ENNReal.inv_pow]
  · -- relayed tags
    have hc := pr_le_of_card (fun x : Payloads N b × (Bits ρ × Oracle N b h ρ × (Fin s → Fin N)) =>
        m ≤ (A.relayed x.1 x.2.1 x.2.2.1).card) (2 ^ (h * m)) (N.choose m * 2 ^ L) (by positivity) (by
      rw [hΩ]
      refine (card_filter_prod_le _ _
        (Fintype.card (Bits ρ) * (N.choose m * 2 ^ L * Fintype.card (Oracle N b h ρ) * N ^ s))
        (fun E => ?_)).trans (le_of_eq (by ring))
      refine card_filter_prod_le _ _ _ (fun R => ?_)
      refine (card_filter_prod_le' _ _ (N.choose m * 2 ^ L * Fintype.card (Oracle N b h ρ))
        (fun _ => ?_)).trans (le_of_eq (by simp))
      exact card_relay_le A E R m)
    refine hc.trans (le_of_eq ?_)
    push_cast
    rfl
  · -- compression at `k − m` words
    refine (pr_le_of_card _ (2 ^ lam) (q.choose (k - m)) (by positivity) (by
      rw [hΩ, card_filter_equiv (Equiv.prodAssoc (Payloads N b) (Bits ρ) (Oracle N b h ρ × (Fin s → Fin N))).symm]
      refine (card_filter_prod_le' _ _ (q.choose (k - m) * Fintype.card (Payloads N b × Bits ρ)) (fun y => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]; ring))
      exact card_compress_le A y.1 (k - m) lam hk0)).trans (le_of_eq ?_)
    push_cast
    rfl

end Accounting.SecurityProofs
