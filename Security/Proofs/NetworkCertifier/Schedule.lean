import Proofs.NetworkCertifier.Advice

/-!
# Theorem 3 for the shipped schedule: the constant-rate Program

`docs/network-transparency/timing-channel.md` §2.4, for `constantRate` (`Definitions.NetworkCertifier.Schedule`). Its
advice is one value per execution block, so `encard_accSeqs_le_advice` applies as it stands. Causality
(`constantRate_rows_causal`: a step's rows read only its block's advice) sharpens it to the blocks the horizon reaches.
Each count is stated for any finite advice per block and for strings of `k` bits (`_bits`), the protocol's.

The witness at the end shows that the blocks bound is met exactly by a covert sender.
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Z In Act Wire : Type}

/-- Strings of `k` bits for each of `E` execution blocks number `2^(E·k)`. -/
theorem card_bits_pow (E k : ℕ) : Fintype.card (Fin k → Bool) ^ E = 2 ^ (E * k) := by
  rw [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, ← pow_mul, Nat.mul_comm k E]

theorem encard_accSeqs_le_constantRate {Adv : Type} [Fintype Adv] (T g E : ℕ)
    (frames : List In → Fin E → Adv → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D : ℕ) :
    (AccSeqs (constantRate (S := S) T g E frames) op n D).encard ≤
      ((Fintype.card Adv ^ E * D ^ n : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_advice (constantRate (S := S) T g E frames) op n D
  rwa [Fintype.card_fun, Fintype.card_fin] at this

theorem encard_accSeqs_le_constantRate_bits (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D : ℕ) :
    (AccSeqs (constantRate (S := S) T g E frames) op n D).encard ≤ ((2 ^ (E * k) * D ^ n : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_constantRate (S := S) T g E frames op n D
  rwa [card_bits_pow] at this

/-! ### Causality: a step's rows read only its execution block's advice -/

theorem constantRate_rows_causal {Adv : Type} (T g E : ℕ)
    (frames : List In → Fin E → Adv → Fin L → List LogicalFrame) (t : ℕ) (ins : List In) (a a' : Fin E → Adv)
    (ha : ∀ e : Fin E, (e : ℕ) = t / (g * T) → a e = a' e) :
    (constantRate (S := S) T g E frames).rows t ins a = (constantRate (S := S) T g E frames).rows t ins a' := by
  funext ℓ
  simp only [constantRate]
  split_ifs with h
  · rw [ha ⟨t / (g * T), h⟩ rfl]
  · rfl

/-! ### `|a|` bits per execution block opened -/

theorem encard_accSeqs_le_constantRate_windows {Adv : Type} [Fintype Adv] (T g E : ℕ)
    (frames : List In → Fin E → Adv → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In) (n D m : ℕ)
    (hm : n ≤ m * (g * T)) :
    (AccSeqs (constantRate (S := S) T g E frames) op n D).encard ≤
      ((Fintype.card Adv ^ min E m * D ^ n : ℕ) : ℕ∞) := by
  classical
  by_cases hne : Nonempty (Fin E → Adv)
  swap
  · have : AccSeqs (constantRate (S := S) T g E frames) op n D = ∅ := by
      ext o; simp only [Set.mem_empty_iff_false, iff_false]
      rintro ⟨-, -, a, -⟩; exact hne ⟨a⟩
    rw [this, Set.encard_empty]; exact bot_le
  let a0 : Fin E → Adv := Classical.choice hne
  let adv : List (Obs L K S) → Fin E → Adv := fun o =>
    if h : AcceptedUpTo (constantRate (S := S) T g E frames) op n o then h.choose else a0
  have adv_spec : ∀ o ∈ AccSeqs (constantRate (S := S) T g E frames) op n D,
      Accepted (constantRate (S := S) T g E frames) op n o (adv o) := by
    intro o ho; simp only [adv, ho.2.2, ↓reduceDIte]; exact ho.2.2.choose_spec
  let cut : (Fin E → Adv) → Fin (min E m) → Adv := fun a i =>
    a ⟨i, lt_of_lt_of_le i.2 (min_le_left _ _)⟩
  let code : List (Obs L K S) → (Fin (min E m) → Adv) × (Fin n → ℕ) := fun o => (cut (adv o), jitterCode o)
  have hinj : Set.InjOn code (AccSeqs (constantRate (S := S) T g E frames) op n D) := by
    intro o ho o' ho' h
    simp only [code, Prod.mk.injEq] at h
    obtain ⟨hσ, hj⟩ := h
    have acc := adv_spec o ho
    have acc' := adv_spec o' ho'
    apply eq_of_stepwise ho.1 ho'.1
    intro t ht hpre
    obtain ⟨h1, hr, hs⟩ := acc t ht
    obtain ⟨h2, hr', hs'⟩ := acc' t ht
    apply Obs.ext'
    · rw [hr, hr', ingressUpTo_congr op hpre]
      apply constantRate_rows_causal
      intro e he
      have htm : t < m * (g * T) := lt_of_lt_of_le ht hm
      have hem : (e : ℕ) < m := by
        rw [he]; exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm (g * T) m]; exact htm)
      have hσe := congrFun hσ ⟨e, lt_min e.2 hem⟩
      simpa only [cut, Fin.eta] using hσe
    · exact jitter_eq_of_jitterCode hj ht h1 h2
    · funext κ; rw [hs κ, hs' κ]
  calc (AccSeqs (constantRate (S := S) T g E frames) op n D).encard
      ≤ (((univ : Finset (Fin (min E m) → Adv)) ×ˢ
          Fintype.piFinset fun _ : Fin n => range D).card : ℕ∞) := by
        apply encard_le_card_of_injOn code hinj
        intro o ho
        simp only [code, mem_product, mem_univ, true_and]
        exact jitterCode_mem o ho.1 ho.2.1
    _ = ((Fintype.card Adv ^ min E m * D ^ n : ℕ) : ℕ∞) := by
        rw [card_product, Fintype.card_piFinset]
        simp

theorem encard_accSeqs_le_constantRate_windows_bits (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D m : ℕ) (hm : n ≤ m * (g * T)) :
    (AccSeqs (constantRate (S := S) T g E frames) op n D).encard ≤ ((2 ^ (min E m * k) * D ^ n : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_constantRate_windows (S := S) T g E frames op n D m hm
  rwa [card_bits_pow] at this

theorem encard_decodable_le_constantRate {Adv : Type} [Fintype Adv]
    (I : Interaction L K S Z In Act Wire) (T g E : ℕ)
    (frames : List In → Fin E → Adv → Fin L → List LogicalFrame) (n D m : ℕ) (hm : n ≤ m * (g * T))
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T g E frames) I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z) :
    Zs.encard ≤ ((Fintype.card Adv ^ min E m * D ^ n : ℕ) : ℕ∞) := by
  calc Zs.encard
      ≤ {o | ∃ z, o = I.obsRun z n ∧ AcceptedUpTo (constantRate (S := S) T g E frames) I.op n o}.encard :=
        encard_decodable_le I _ n view hW2 Zs hacc dec hdec
    _ ≤ (AccSeqs (constantRate (S := S) T g E frames) I.op n D).encard := by
        apply Set.encard_le_encard
        rintro o ⟨z, rfl, ha⟩
        refine ⟨I.length_obsRun z n, ?_, ha⟩
        intro x hx
        obtain ⟨k, xs, as, rfl⟩ := I.mem_obsRun z n x hx
        exact hJ k xs as
    _ ≤ _ := encard_accSeqs_le_constantRate_windows T g E frames I.op n D m hm

theorem encard_decodable_le_constantRate_bits
    (I : Interaction L K S Z In Act Wire) (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (n D m : ℕ) (hm : n ≤ m * (g * T))
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T g E frames) I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z) :
    Zs.encard ≤ ((2 ^ (min E m * k) * D ^ n : ℕ) : ℕ∞) := by
  have := encard_decodable_le_constantRate I T g E frames n D m hm view hW2 hJ Zs hacc dec hdec
  rwa [card_bits_pow] at this

/-! ## Witness: the blocks bound is met -/

/-- One frame of one execution block, sent at bucket 1 if the block's one advice bit is set and at bucket 0
otherwise. -/
def oneFrame : List Unit → Fin 1 → (Fin 1 → Bool) → Fin 1 → List LogicalFrame :=
  fun _ _ v _ => [⟨[true], if v 0 then 1 else 0⟩]

/-- The covert sender of the joint witness: its secret bit picks the bucket its one frame leaves in. -/
def delayObs (k : ℕ) (_ : List Unit) (as : List Bool) : Obs 1 1 1 :=
  { rows := fun _ _ => if decide (k = 0) = !(as.getD 0 false) then some [true] else none
    jitter := 0
    status := fun _ => Status.complete }

def delayCertifier : Certifier 1 1 1 Unit Bool (Obs 1 1 1) where
  obs := delayObs
  wire := delayObs
  fault _ := False

def delayInteraction : Interaction 1 1 1 Bool Unit Bool (Obs 1 1 1) where
  certifier := delayCertifier
  op _ _ := ()
  snd z _ _ := z

instance instDecidableAcceptedOneFrame (k : ℕ) (op : ℕ → List (Obs 1 1 1) → Unit) (o : List (Obs 1 1 1))
    (a : Fin 1 → Fin 1 → Bool) : Decidable (Accepted (constantRate (S := 1) 2 1 1 oneFrame) op k o a) := by
  unfold Accepted; infer_instance

/-- **The blocks bound is met by a covert sender.** One execution block of two buckets (`g = 1`, `T = 2`),
`oneFrame`, and one advice bit. The sender's secret bit delays its frame by one bucket, and each run is accepted with
the matching bit. Every hypothesis of `EncardDecodableLeConstantRate` holds (W2, J with `D = 1`, acceptance, a wire
decoder, a horizon of `n = 2` within `m = 1` block), and both secrets are decoded, which is the bound
`2^(min(E, m)·k) · D^n = 2^1 · 1^2`. -/
theorem constantRate_joint_witness :
    Regenerates delayInteraction.certifier (fun _ o => o) ∧ JitterBounded delayInteraction.certifier 1 ∧
      (∀ z ∈ (Set.univ : Set Bool),
        AcceptedUpTo (constantRate (S := 1) 2 1 1 oneFrame) delayInteraction.op 2 (delayInteraction.obsRun z 2)) ∧
      (∀ z ∈ (Set.univ : Set Bool),
        (fun w : List (Obs 1 1 1) => (w.head?.map fun x => decide (x.rows 0 0 = none)).getD false)
          (delayInteraction.wireRun z 2) = z) ∧
      2 ≤ 1 * (1 * 2) ∧ (Set.univ : Set Bool).encard = ((2 ^ (min 1 1 * 1) * 1 ^ 2 : ℕ) : ℕ∞) := by
  refine ⟨fun _ _ _ => rfl, fun _ _ _ => by simp [delayInteraction, delayCertifier, delayObs], ?_, ?_, by decide, ?_⟩
  · intro z _
    exact ⟨fun _ _ => z, by cases z <;> decide +kernel⟩
  · intro z _
    cases z <;> decide +kernel
  · rw [Set.encard_univ, ENat.card_eq_coe_fintype_card]
    decide +kernel

end NetTiming
