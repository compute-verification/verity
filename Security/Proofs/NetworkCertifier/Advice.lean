import Proofs.NetworkCertifier.Counting
import Proofs.NetworkCertifier.Occupancy

/-!
# Theorem 3: the advice caps

`docs/network-transparency/timing-channel.md` §2.4. Everything is counted on accepted observation sequences: the rows
at every step equal the Program's rows for one declared advice string, every status is complete, and the jitter classes
are below `D`.
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Z In Act Wire : Type}

theorem Obs.ext' {x y : Obs L K S} (h1 : x.rows = y.rows) (h2 : x.jitter = y.jitter) (h3 : x.status = y.status) :
    x = y := by
  cases x; cases y; simp_all

/-- **W4 at the certifier: timing is occupancy.** Under packing, every recorded egress row is the fill of some frame
sequence at its count, written as a one-bucket grid. -/
theorem recorded_row_eq_fill {In Act Wire : Type} (w : Certifier L K S In Act Wire) (hW4 : Packed w)
    (k : ℕ) (xs : List In) (as : List Act) (ℓ : Fin L) :
    ∃ (f : ℕ → List Bool) (c : Fin 1 → Fin (S + 1)), (fun _ : Fin 1 => (w.obs k xs as).rows ℓ) = fill f c :=
  exists_fill_of_leftPacked _ (fun _ => hW4 k xs as ℓ)

/-- The jitter code of a sequence: its jitter classes, `0` past its end. -/
def jitterCode {n : ℕ} (o : List (Obs L K S)) : Fin n → ℕ := fun t => (o[(t : ℕ)]?.map Obs.jitter).getD 0

theorem jitterCode_mem {n D : ℕ} (o : List (Obs L K S)) (hl : o.length = n) (hD : ∀ x ∈ o, x.jitter < D) :
    jitterCode (n := n) o ∈ Fintype.piFinset fun _ : Fin n => range D := by
  rw [Fintype.mem_piFinset]; intro t
  have ht : (t : ℕ) < o.length := by rw [hl]; exact t.2
  simp only [jitterCode, List.getElem?_eq_getElem ht, Option.map_some, Option.getD_some, mem_range]
  exact hD _ (List.getElem_mem ht)

theorem jitter_eq_of_jitterCode {n : ℕ} {o o' : List (Obs L K S)} (h : jitterCode (n := n) o = jitterCode o')
    {t : ℕ} (ht : t < n) (h1 : t < o.length) (h2 : t < o'.length) : o[t].jitter = o'[t].jitter := by
  have := congrFun h ⟨t, ht⟩
  simpa [jitterCode, List.getElem?_eq_getElem h1, List.getElem?_eq_getElem h2] using this

/-- The prefix induction shared by both caps: two accepted sequences of length `n` agree once they agree, step by
step, on everything the verifier cannot pin down from the prefix. -/
theorem eq_of_stepwise {n : ℕ} {o o' : List (Obs L K S)} (hl : o.length = n) (hl' : o'.length = n)
    (step : ∀ t (ht : t < n), o.take t = o'.take t →
      o[t]'(by omega) = o'[t]'(by omega)) : o = o' := by
  have key : ∀ t ≤ n, o.take t = o'.take t := by
    intro t
    induction t with
    | zero => simp
    | succ t ih =>
      intro ht
      have hpre := ih (by omega)
      rw [← List.take_concat_get' o t (by omega), ← List.take_concat_get' o' t (by omega), hpre,
        step t (by omega) hpre]
  have := key n le_rfl
  rwa [← hl, List.take_length, hl, ← hl', List.take_length] at this

/-- **Theorem 3, the `2^A` cap.** With `Fintype.card Adv = 2^A`, the accepted observation sequences of length `n` with
jitter below `D` number at most `2^A · D^n`. -/
theorem encard_accSeqs_le_advice {Adv : Type} [Fintype Adv] (P : EgressPrediction L S In Adv)
    (op : ℕ → List (Obs L K S) → In) (n D : ℕ) :
    (AccSeqs P op n D).encard ≤ ((Fintype.card Adv * D ^ n : ℕ) : ℕ∞) := by
  classical
  by_cases hne : Nonempty Adv
  swap
  · have : AccSeqs P op n D = ∅ := by
      ext o; simp only [Set.mem_empty_iff_false, iff_false]
      rintro ⟨-, -, a, -⟩; exact hne ⟨a⟩
    rw [this, Set.encard_empty]; exact bot_le
  let a0 : Adv := Classical.choice hne
  let adv : List (Obs L K S) → Adv := fun o => if h : AcceptedUpTo P op n o then h.choose else a0
  have adv_spec : ∀ o ∈ AccSeqs P op n D, Accepted P op n o (adv o) := by
    intro o ho; simp only [adv, ho.2.2, ↓reduceDIte]; exact ho.2.2.choose_spec
  let code : List (Obs L K S) → Adv × (Fin n → ℕ) := fun o => (adv o, jitterCode o)
  have hinj : Set.InjOn code (AccSeqs P op n D) := by
    intro o ho o' ho' h
    simp only [code, Prod.mk.injEq] at h
    obtain ⟨ha, hj⟩ := h
    have acc := adv_spec o ho
    have acc' := adv_spec o' ho'
    apply eq_of_stepwise ho.1 ho'.1
    intro t ht hpre
    obtain ⟨h1, hr, hs⟩ := acc t ht
    obtain ⟨h2, hr', hs'⟩ := acc' t ht
    apply Obs.ext'
    · rw [hr, hr', ha, ingressUpTo_congr op hpre]
    · exact jitter_eq_of_jitterCode hj ht h1 h2
    · funext κ; rw [hs κ, hs' κ]
  calc (AccSeqs P op n D).encard
      ≤ (((univ : Finset Adv) ×ˢ Fintype.piFinset fun _ : Fin n => range D).card : ℕ∞) := by
        apply encard_le_card_of_injOn code hinj
        intro o ho
        simp only [code, mem_product, mem_univ, true_and]
        exact jitterCode_mem o ho.1 ho.2.1
    _ = ((Fintype.card Adv * D ^ n : ℕ) : ℕ∞) := by
        rw [card_product, Fintype.card_piFinset]; simp

/-- **Theorem 3, the fixed advice budget.** With the advice `E` strings of exactly `k` bits, one per execution block,
the accepted sequences number at most `2^(E·k) · D^n`. -/
theorem encard_accSeqs_le_adviceBits (E k : ℕ) (P : EgressPrediction L S In (Fin E → Fin k → Bool))
    (op : ℕ → List (Obs L K S) → In) (n D : ℕ) :
    (AccSeqs P op n D).encard ≤ ((2 ^ (E * k) * D ^ n : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_advice P op n D
  rwa [Fintype.card_fun, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Fintype.card_fin, ← pow_mul'] at this

/-- **Theorem 3, the clock-sync form.** This is the `2^A` cap with the advice type `Ac × (Fin W → Sync)`: one
clock-sync value per window from `Sync`, for example an offset δ and a rate ρ, 16 × 8 values. Over `W` windows the
accepted sequences number at most `#Ac · #Sync^W · D^n`, that is `A_c + W·log2 #Sync` bits plus jitter.

Instantiated for constant-rate release, this bounds the timing channel when the Program's only timing advice is the
clock sync. The shipped schedule is such a Program by its definition (`constantRate`), and
`EncardAccSeqsLeConstantRate` states this bound for it. A sync alphabet that varies by window is covered by
`encard_accSeqs_le_advice` directly. -/
theorem encard_accSeqs_le_sync {Ac Sync : Type} [Fintype Ac] [Fintype Sync] (W : ℕ)
    (P : EgressPrediction L S In (Ac × (Fin W → Sync))) (op : ℕ → List (Obs L K S) → In) (n D : ℕ) :
    (AccSeqs P op n D).encard ≤ ((Fintype.card Ac * Fintype.card Sync ^ W * D ^ n : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_advice P op n D
  rwa [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin] at this

/-- **A fixed clock.** With the clock-sync value fixed (`Sync = Unit`), the accepted sequences number at most
`#Ac · D^n`. Instantiated for constant-rate release under a fixed logical schedule, timing carries nothing beyond the
jitter, and nothing at all under J (`D = 1`): the zero-capacity theorem, with content advice. -/
theorem encard_accSeqs_le_fixedClock {Ac : Type} [Fintype Ac] (W : ℕ)
    (P : EgressPrediction L S In (Ac × (Fin W → Unit))) (op : ℕ → List (Obs L K S) → In) (n D : ℕ) :
    (AccSeqs P op n D).encard ≤ ((Fintype.card Ac * D ^ n : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_sync W P op n D
  simpa using this

end NetTiming
