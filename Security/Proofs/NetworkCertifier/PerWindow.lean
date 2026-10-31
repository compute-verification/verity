import Proofs.NetworkCertifier.Schedule
import Proofs.NetworkCertifier.Ingress

/-!
# Per window: the blocks the accounting charges

The guarantees count steps; these count the monitoring period's whole blocks and factor each bound per block: the
reference schedule's `2^k · D^(g·T)` per execution block, from any execution block's start and whatever the run showed
before it, and the ingress bound's `(r ℓ + 1)^T` per `T` buckets (`encard_ingress_obsDecodable_le` and its content
version at `n = m·T`). Each takes its parent's hypotheses at that length and no other.

The egress count from block `e₀` codes a sequence continuing a fixed prefix by the advice of blocks `e₀` to `e₀ + m − 1`
and the jitter classes after the prefix. That code fixes the sequence: before block `e₀` it is the prefix, and after it
each step's rows read only its own block's advice (`constantRate_rows_causal`), with the FIFO empty at the block's
first bucket.
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Z Msg In Act Wire : Type}

/-- The ingress bound at `n = m·T`, read per link-window. -/
theorem ingress_bound_eq_perWindow (r : Fin L → ℕ) (T m D : ℕ) :
    (∏ ℓ, (r ℓ + 1)) ^ (m * T) * D ^ (m * T) = ((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m := by
  rw [pow_mul', pow_mul', ← mul_pow, Finset.prod_pow]

/-- **`m` execution blocks from block `e₀`, for any finite advice per block.** The accepted sequences through block
`e₀ + m − 1` that continue a given prefix `p` of the first `e₀` blocks number at most `(#Adv · D^(g·T))^m`. -/
theorem encard_accSeqs_le_constantRate_from {Adv : Type} [Fintype Adv] (T g E : ℕ)
    (frames : List In → Fin E → Adv → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (e₀ m D : ℕ) (hE : e₀ + m ≤ E) (p : List (Obs L K S)) :
    {o | o ∈ AccSeqs (constantRate (S := S) T g E frames) op ((e₀ + m) * (g * T)) D ∧
        o.take (e₀ * (g * T)) = p}.encard ≤ (((Fintype.card Adv * D ^ (g * T)) ^ m : ℕ) : ℕ∞) := by
  classical
  by_cases hne : Nonempty (Fin E → Adv)
  swap
  · have : {o | o ∈ AccSeqs (constantRate (S := S) T g E frames) op ((e₀ + m) * (g * T)) D ∧
        o.take (e₀ * (g * T)) = p} = ∅ := by
      ext o; simp only [Set.mem_empty_iff_false, iff_false]
      rintro ⟨⟨-, -, a, -⟩, -⟩; exact hne ⟨a⟩
    rw [this, Set.encard_empty]; exact bot_le
  let a0 : Fin E → Adv := Classical.choice hne
  let adv : List (Obs L K S) → Fin E → Adv := fun o =>
    if h : AcceptedUpTo (constantRate (S := S) T g E frames) op ((e₀ + m) * (g * T)) o then h.choose else a0
  have adv_spec : ∀ o ∈ AccSeqs (constantRate (S := S) T g E frames) op ((e₀ + m) * (g * T)) D,
      Accepted (constantRate (S := S) T g E frames) op ((e₀ + m) * (g * T)) o (adv o) := by
    intro o ho; simp only [adv, ho.2.2, ↓reduceDIte]; exact ho.2.2.choose_spec
  have hBn : e₀ * (g * T) ≤ (e₀ + m) * (g * T) := Nat.mul_le_mul_right _ (Nat.le_add_right _ _)
  have hlen : ∀ o : List (Obs L K S), o.length = (e₀ + m) * (g * T) →
      (o.drop (e₀ * (g * T))).length = m * (g * T) := by
    intro o ho; rw [List.length_drop, ho, ← Nat.sub_mul, Nat.add_sub_cancel_left]
  let cut : (Fin E → Adv) → Fin m → Adv := fun a i => a ⟨e₀ + i, by have := i.2; omega⟩
  let code : List (Obs L K S) → (Fin m → Adv) × (Fin (m * (g * T)) → ℕ) :=
    fun o => (cut (adv o), jitterCode (o.drop (e₀ * (g * T))))
  have hinj : Set.InjOn code {o | o ∈ AccSeqs (constantRate (S := S) T g E frames) op ((e₀ + m) * (g * T)) D ∧
      o.take (e₀ * (g * T)) = p} := by
    intro o ho o' ho' h
    simp only [code, Prod.mk.injEq] at h
    obtain ⟨hσ, hj⟩ := h
    have acc := adv_spec o ho.1
    have acc' := adv_spec o' ho'.1
    have hp : o.take (e₀ * (g * T)) = o'.take (e₀ * (g * T)) := ho.2.trans ho'.2.symm
    have hl := ho.1.1
    have hl' := ho'.1.1
    apply eq_of_stepwise hl hl'
    intro t ht hpre
    by_cases htB : t < e₀ * (g * T)
    · have h1 : t < (o.take (e₀ * (g * T))).length := by rw [List.length_take]; omega
      calc o[t] = (o.take (e₀ * (g * T)))[t] := List.getElem_take.symm
        _ = (o'.take (e₀ * (g * T)))[t]'(hp ▸ h1) := List.getElem_of_eq hp h1
        _ = o'[t] := List.getElem_take
    obtain ⟨h1, hr, hs⟩ := acc t ht
    obtain ⟨h2, hr', hs'⟩ := acc' t ht
    have hgT : 0 < g * T := Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at ht; omega
    apply Obs.ext'
    · rw [hr, hr', ingressUpTo_congr op hpre]
      apply constantRate_rows_causal
      intro e he
      have hlo : e₀ ≤ (e : ℕ) := by rw [he]; exact (Nat.le_div_iff_mul_le hgT).mpr (by omega)
      have hhi : (e : ℕ) < e₀ + m := by rw [he]; exact (Nat.div_lt_iff_lt_mul hgT).mpr ht
      have hσe := congrFun hσ ⟨e - e₀, by omega⟩
      have hidx : e = ⟨e₀ + (e - e₀), by omega⟩ := Fin.ext (by simp only; omega)
      rw [hidx]
      exact hσe
    · have hj' := jitter_eq_of_jitterCode hj (t := t - e₀ * (g * T)) (by rw [← hlen o hl]; simp only [List.length_drop]; omega)
        (by simp only [List.length_drop]; omega) (by simp only [List.length_drop]; omega)
      simp only [List.getElem_drop, Nat.add_sub_cancel' (Nat.le_of_not_lt htB)] at hj'
      exact hj'
    · funext κ; rw [hs κ, hs' κ]
  calc _ ≤ (((univ : Finset (Fin m → Adv)) ×ˢ
          Fintype.piFinset fun _ : Fin (m * (g * T)) => range D).card : ℕ∞) := by
        apply encard_le_card_of_injOn code hinj
        intro o ho
        simp only [code, mem_product, mem_univ, true_and]
        exact jitterCode_mem _ (hlen o ho.1.1) fun x hx => ho.1.2.1 x (List.mem_of_mem_drop hx)
    _ = (((Fintype.card Adv * D ^ (g * T)) ^ m : ℕ) : ℕ∞) := by
        rw [card_product, Fintype.card_piFinset]
        simp only [card_univ, Fintype.card_fun, Fintype.card_fin, card_range, prod_const]
        rw [mul_pow, ← pow_mul']

/-- **`|a|` bits per execution block, from any block's start.** `encard_accSeqs_le_constantRate_from` with `|a|`-bit
advice. The steps counted need only lie within the `E` blocks: with `g·T = 0` there is one sequence, the empty one. -/
theorem encard_accSeqs_le_constantRate_perWindow (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (e₀ m D : ℕ) (hn : (e₀ + m) * (g * T) ≤ E * (g * T)) (p : List (Obs L K S)) :
    {o | o ∈ AccSeqs (constantRate (S := S) T g E frames) op ((e₀ + m) * (g * T)) D ∧
        o.take (e₀ * (g * T)) = p}.encard ≤ (((2 ^ k * D ^ (g * T)) ^ m : ℕ) : ℕ∞) := by
  rcases Nat.eq_zero_or_pos (g * T) with h0 | hpos
  · calc _ ≤ ({[]} : Set (List (Obs L K S))).encard := by
          apply Set.encard_le_encard
          rintro o ⟨⟨hl, -⟩, -⟩
          rw [h0, Nat.mul_zero] at hl
          exact List.eq_nil_of_length_eq_zero hl
      _ = 1 := Set.encard_singleton _
      _ ≤ _ := by
          rw [h0, pow_zero, mul_one]
          exact_mod_cast Nat.one_le_iff_ne_zero.mpr (pow_ne_zero _ (pow_ne_zero _ two_ne_zero))
  · have := encard_accSeqs_le_constantRate_from (S := S) T g E frames op e₀ m D
      (Nat.le_of_mul_le_mul_right hn hpos) p
    rwa [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin] at this

/-- **`|a|` bits per execution block through the wire, from any block's start (W2 and J).** Secrets whose runs show
the same `p` through the first `e₀` execution blocks, accepted through the `m` after them and decoded from the wire,
number at most `(2^k · D^(g·T))^m`. -/
theorem encard_decodable_le_constantRate_perWindow
    (I : Interaction L K S Z In Act Wire) (T g E k : ℕ)
    (frames : List In → Fin E → (Fin k → Bool) → Fin L → List LogicalFrame) (e₀ m D : ℕ)
    (hn : (e₀ + m) * (g * T) ≤ E * (g * T))
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z) (p : List (Obs L K S)) (hpre : ∀ z ∈ Zs, I.obsRun z (e₀ * (g * T)) = p)
    (hacc : ∀ z ∈ Zs, AcceptedUpTo (constantRate (S := S) T g E frames) I.op ((e₀ + m) * (g * T))
      (I.obsRun z ((e₀ + m) * (g * T))))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z ((e₀ + m) * (g * T))) = z) :
    Zs.encard ≤ (((2 ^ k * D ^ (g * T)) ^ m : ℕ) : ℕ∞) := by
  have hinj : Set.InjOn (fun z => I.obsRun z ((e₀ + m) * (g * T))) Zs := by
    intro z hz z' hz' h
    simp only at h
    rw [← hdec z hz, ← hdec z' hz', I.wireRun_eq view hW2, I.wireRun_eq view hW2, h]
  rw [← hinj.encard_image]
  refine le_trans (Set.encard_le_encard ?_)
    (encard_accSeqs_le_constantRate_perWindow T g E k frames I.op e₀ m D hn p)
  rintro _ ⟨z, hz, rfl⟩
  refine ⟨⟨I.length_obsRun z _, ?_, hacc z hz⟩, ?_⟩
  · intro x hx
    obtain ⟨t, xs, as, rfl⟩ := I.mem_obsRun z _ x hx
    exact hJ t xs as
  · rw [I.obsRun_take z (Nat.mul_le_mul_right _ (Nat.le_add_right _ _))]
    exact hpre z hz

theorem encard_ingress_obsDecodable_le_perWindow (I : Interaction L K S Msg In Act Wire) (T m D : ℕ)
    (r : Fin L → ℕ) (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg) :
    Ms.encard ≤ ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞) := by
  rw [← ingress_bound_eq_perWindow]
  exact encard_ingress_obsDecodable_le I (m * T) D r hW4 hr hJ c Ms hacc hc dec hdec

theorem encard_ingress_obsDecodable_le_contents_perWindow (I : Interaction L K S Msg In Act Wire) (T m D : ℕ)
    (r : Fin L → ℕ) (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg) :
    Ms.encard ≤ Cs.encard * ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞) := by
  rw [← ingress_bound_eq_perWindow]
  exact encard_ingress_obsDecodable_le_contents I (m * T) D r hW4 hr hJ Cs Ms hacc hc dec hdec

end NetTiming
