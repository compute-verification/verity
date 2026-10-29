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
theorem encard_accSeqs_le_advice {Adv : Type} [Fintype Adv] (P : Program L S In Adv)
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

/-- **Theorem 3, the clock-sync form.** This is the `2^A` cap with the advice type `Ac × (Fin W → Sync)`: one
clock-sync value per window from `Sync`, for example an offset δ and a rate ρ, 16 × 8 values. Over `W` windows the
accepted sequences number at most `#Ac · #Sync^W · D^n`, that is `A_c + W·log2 #Sync` bits plus jitter.

Instantiated for constant-rate release, this bounds the timing channel when the Program's only timing advice is the
clock sync. The shipped schedule is such a Program by its definition (`constantRate`), and
`EncardAccSeqsLeConstantRate` states this bound for it. A sync alphabet that varies by window is covered by
`encard_accSeqs_le_advice` directly. -/
theorem encard_accSeqs_le_sync {Ac Sync : Type} [Fintype Ac] [Fintype Sync] (W : ℕ)
    (P : Program L S In (Ac × (Fin W → Sync))) (op : ℕ → List (Obs L K S) → In) (n D : ℕ) :
    (AccSeqs P op n D).encard ≤ ((Fintype.card Ac * Fintype.card Sync ^ W * D ^ n : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_advice P op n D
  rwa [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin] at this

/-- **A fixed clock.** With the clock-sync value fixed (`Sync = Unit`), the accepted sequences number at most
`#Ac · D^n`. Instantiated for constant-rate release under a fixed logical schedule, timing carries nothing beyond the
jitter, and nothing at all under J (`D = 1`): the zero-capacity theorem, with content advice. -/
theorem encard_accSeqs_le_fixedClock {Ac : Type} [Fintype Ac] (W : ℕ)
    (P : Program L S In (Ac × (Fin W → Unit))) (op : ℕ → List (Obs L K S) → In) (n D : ℕ) :
    (AccSeqs P op n D).encard ≤ ((Fintype.card Ac * D ^ n : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_sync W P op n D
  simpa using this

/-- **Theorem 3, the block cap, under the two dataflow conditions.** The advice is content advice `Ac` and timing
advice `At β` per block `β`. The cells `C` carry the Program's occupancy predictions `occ c : U c`, at step `step c`,
in block `blk c`. Then the accepted sequences number at most
`#Ac · D^n · ∏_β min(#At β, ∏_{c ∈ β} #U c)`. -/
theorem encard_accSeqs_le_blocks {Ac B C : Type} {At : B → Type} {U : C → Type}
    [Fintype Ac] [Fintype B] [DecidableEq B] [∀ β, Fintype (At β)] [Fintype C] [DecidableEq C] [∀ c, Fintype (U c)]
    (P : Program L S In (Ac × ((β : B) → At β))) (op : ℕ → List (Obs L K S) → In) (n D : ℕ)
    (step : C → ℕ) (blk : C → B) (occ : (c : C) → List In → Ac × ((β : B) → At β) → U c)
    (h1 : FramesReadNoTimingAdvice P step occ) (h2 : OccupancyReadsOwnBlock blk occ) :
    (AccSeqs P op n D).encard ≤
      ((Fintype.card Ac * D ^ n *
        ∏ β, min (Fintype.card (At β)) (∏ c ∈ univ.filter (fun c => blk c = β), Fintype.card (U c)) : ℕ) : ℕ∞) := by
  classical
  by_cases hne : Nonempty (Ac × ((β : B) → At β))
  swap
  · have : AccSeqs P op n D = ∅ := by
      ext o; simp only [Set.mem_empty_iff_false, iff_false]
      rintro ⟨-, -, a, -⟩; exact hne ⟨a⟩
    rw [this, Set.encard_empty]; exact bot_le
  let a0 : Ac × ((β : B) → At β) := Classical.choice hne
  let adv : List (Obs L K S) → Ac × ((β : B) → At β) := fun o => if h : AcceptedUpTo P op n o then h.choose else a0
  have adv_spec : ∀ o ∈ AccSeqs P op n D, Accepted P op n o (adv o) := by
    intro o ho; simp only [adv, ho.2.2, ↓reduceDIte]; exact ho.2.2.choose_spec
  let prodU : B → ℕ := fun β => ∏ c ∈ univ.filter (fun c => blk c = β), Fintype.card (U c)
  let Kset : Finset B := univ.filter fun β => Fintype.card (At β) ≤ prodU β
  let code : List (Obs L K S) → Ac × (Fin n → ℕ) × ((β : B) → At β ⊕ ((c : {c // blk c = β}) → U c.1)) := fun o =>
    ((adv o).1, jitterCode o, fun β => if β ∈ Kset then Sum.inl ((adv o).2 β)
      else Sum.inr fun c => occ c.1 (ingressUpTo op o (step c.1)) (adv o))
  have hinj : Set.InjOn code (AccSeqs P op n D) := by
    intro o ho o' ho' h
    have hc : (code o).1 = (code o').1 := by rw [h]
    have hj : (code o).2.1 = (code o').2.1 := by rw [h]
    have hb : (code o).2.2 = (code o').2.2 := by rw [h]
    simp only [code] at hc hj hb
    have acc := adv_spec o ho
    have acc' := adv_spec o' ho'
    apply eq_of_stepwise ho.1 ho'.1
    intro t ht hpre
    obtain ⟨hl1, hr, hs⟩ := acc t ht
    obtain ⟨hl2, hr', hs'⟩ := acc' t ht
    have hins := ingressUpTo_congr op hpre
    have hocc : ∀ c, step c = t → occ c (ingressUpTo op o t) (adv o) = occ c (ingressUpTo op o t) (adv o') := by
      intro c hct
      have hbc := congrFun hb (blk c)
      by_cases hK : blk c ∈ Kset
      · simp only [hK, ↓reduceIte] at hbc
        exact h2 c _ _ _ hc (Sum.inl.inj hbc)
      · simp only [hK, ↓reduceIte] at hbc
        have := congrFun (Sum.inr.inj hbc) ⟨c, rfl⟩
        simp only at this
        rw [hct] at this
        rw [this, hins]
    apply Obs.ext'
    · rw [hr, hr', h1 t _ _ _ hc hocc, hins]
    · exact jitter_eq_of_jitterCode hj ht hl1 hl2
    · funext κ; rw [hs κ, hs' κ]
  let blockSet : (β : B) → Finset (At β ⊕ ((c : {c // blk c = β}) → U c.1)) := fun β =>
    if β ∈ Kset then univ.map Function.Embedding.inl else univ.map Function.Embedding.inr
  have hblock : ∀ β, (blockSet β).card = min (Fintype.card (At β)) (prodU β) := by
    intro β
    by_cases hK : β ∈ Kset
    · have hle : Fintype.card (At β) ≤ prodU β := by simpa [Kset] using hK
      simp only [blockSet, hK, ↓reduceIte, card_map, card_univ]; exact (min_eq_left hle).symm
    · have hlt : prodU β < Fintype.card (At β) := by simpa [Kset] using hK
      simp only [blockSet, hK, ↓reduceIte, card_map, card_univ, Fintype.card_pi]
      rw [min_eq_right hlt.le]
      simp only [prodU]
      exact (Finset.prod_subtype (univ.filter fun c => blk c = β) (by simp) (fun c => Fintype.card (U c))).symm
  calc (AccSeqs P op n D).encard
      ≤ (((univ : Finset Ac) ×ˢ (Fintype.piFinset fun _ : Fin n => range D) ×ˢ Fintype.piFinset blockSet).card
          : ℕ∞) := by
        apply encard_le_card_of_injOn code hinj
        intro o ho
        refine Finset.mem_product.mpr ⟨mem_univ _, Finset.mem_product.mpr ⟨jitterCode_mem o ho.1 ho.2.1, ?_⟩⟩
        rw [Fintype.mem_piFinset]; intro β
        by_cases hK : β ∈ Kset <;> simp [code, blockSet, hK]
    _ = ((Fintype.card Ac * D ^ n *
          ∏ β, min (Fintype.card (At β)) (∏ c ∈ univ.filter (fun c => blk c = β), Fintype.card (U c)) : ℕ) :
          ℕ∞) := by
        rw [card_product, card_product, Fintype.card_piFinset, Fintype.card_piFinset]
        simp only [card_univ, card_range, prod_const, Fintype.card_fin, hblock, prodU, mul_assoc]

/-- **The left-packed instance.** Cells are the (link, step) pairs of the horizon, and the Program's occupancy at a
cell is the count of its row, a value in `Fin (S + 1)`. Then the block cap reads
`#Ac · D^n · ∏_β min(#At β, (S + 1)^|β|)`. -/
theorem encard_accSeqs_le_blocks_counts {Ac B : Type} {At : B → Type}
    [Fintype Ac] [Fintype B] [DecidableEq B] [∀ β, Fintype (At β)]
    (P : Program L S In (Ac × ((β : B) → At β))) (op : ℕ → List (Obs L K S) → In) (n D : ℕ)
    (blk : Fin L × Fin n → B)
    (h1 : FramesReadNoTimingAdvice P (fun c : Fin L × Fin n => (c.2 : ℕ))
      (fun c ins a => rowCount (P.rows c.2 ins a c.1)))
    (h2 : OccupancyReadsOwnBlock blk (fun c : Fin L × Fin n => fun ins a => rowCount (P.rows c.2 ins a c.1))) :
    (AccSeqs P op n D).encard ≤
      ((Fintype.card Ac * D ^ n *
        ∏ β, min (Fintype.card (At β)) ((S + 1) ^ (univ.filter (fun c => blk c = β)).card) : ℕ) : ℕ∞) := by
  have := encard_accSeqs_le_blocks (U := fun _ => Fin (S + 1)) P op n D _ blk _ h1 h2
  simpa [Fintype.card_fin, prod_const] using this

/-- **The composed statement (W2, J and both conditions).** A decoder reads the wire and recovers every secret in `Zs`,
all of whose runs are accepted up to `n`. Then `Zs` is no larger than the block cap. -/
theorem encard_decodable_le_blocks {Ac B C : Type} {At : B → Type} {U : C → Type}
    [Fintype Ac] [Fintype B] [DecidableEq B] [∀ β, Fintype (At β)] [Fintype C] [DecidableEq C] [∀ c, Fintype (U c)]
    (I : Interaction L K S Z In Act Wire) (P : Program L S In (Ac × ((β : B) → At β))) (n D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (step : C → ℕ) (blk : C → B) (occ : (c : C) → List In → Ac × ((β : B) → At β) → U c)
    (h1 : FramesReadNoTimingAdvice P step occ) (h2 : OccupancyReadsOwnBlock blk occ)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo P I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z) :
    Zs.encard ≤
      ((Fintype.card Ac * D ^ n *
        ∏ β, min (Fintype.card (At β)) (∏ c ∈ univ.filter (fun c => blk c = β), Fintype.card (U c)) : ℕ) : ℕ∞) := by
  calc Zs.encard ≤ {o | ∃ z, o = I.obsRun z n ∧ AcceptedUpTo P I.op n o}.encard :=
        encard_decodable_le I P n view hW2 Zs hacc dec hdec
    _ ≤ (AccSeqs P I.op n D).encard := by
        apply Set.encard_le_encard
        rintro o ⟨z, rfl, ha⟩
        refine ⟨I.length_obsRun z n, ?_, ha⟩
        intro x hx
        obtain ⟨k, xs, as, rfl⟩ := I.mem_obsRun z n x hx
        exact hJ k xs as
    _ ≤ _ := encard_accSeqs_le_blocks P I.op n D step blk occ h1 h2

end NetTiming
