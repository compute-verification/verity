module

public import Definitions.WeightSecurity.Guarantees
public import Proofs.Core.Game.Basic
public import Proofs.Core.Game.Interleave
public import Proofs.Core.Game.Union
public import Proofs.Spot.ExfiltrationBound
public import Proofs.WeightSecurity.Theft

/-! `WeightSecurity.SecurityProofs.TheftServed : WeightSecurity.Guarantees.TheftServed`, `Theft`'s leakage lemma at an
audited deployment, with `TheftServedReq` and `TheftServedMinEntropy`. On a run whose releases are isolated and whose
audit leaves fewer than `M` incorrect proof blocks undetected, the outsider's view is its view against the datacenter
that deviates through the served channel with the run's own covert string (`Served.view_eq_resp`): each window's advice
and bucket, the values its release changed, and the halt. Those values number at most `F · M + (L + 1) · |O|`
(`Served.sum_card_changed_le`): through the windows accepted before the first rejection, which sampled none of their
incorrect proof blocks, the changes count against those blocks, which count in `Served.lead`, and the rejected window
and the lag's windows change at most `|O|` values each. So the outsider steals only where the run leaves `M` undetected,
its releases aren't isolated, or some honest-use party `Ou.withString ch s` steals, which `HonestUseHard` bounds by `p`
each (`theftServed_isolated`). Local releases are isolated (`Served.isolated_of_local`), which gives `TheftServed`.

`card_servedChannel_le` counts the served channel's covert strings,
`|Adv|^T · |Tick|^T · (Σ_{j ≤ K} C(T·|O|, j)) · |V|^K · (T + 1)`, the overwrites by the Hamming-ball count
`Population.Exfil.card_devBall_le`, and `card_servedChannel_le_rpow` bounds that by two to `servedBits`, the names of
the overwritten values by `(T·|O| + 1)^K`, which gives `TheftServedReq`. The witnesses are proved at the toy
deployment. -/

@[expose] public section

namespace WeightSecurity

open Finset

section View

variable {R X M Θ' : Type}

theorem Outsider.view_length (O : Outsider R X M Θ') (r : R) (resp : List (X × M) → X → M) :
    ∀ n, (O.view r resp n).length = n
  | 0 => rfl
  | n + 1 => by simp [Outsider.view, Outsider.view_length O r resp n]

/-- **Two responders that agree along the outsider's view give it the same view.** -/
theorem Outsider.view_congr (O : Outsider R X M Θ') (r : R) (resp resp' : List (X × M) → X → M) :
    ∀ q, (∀ n < q, resp (O.view r resp n) (O.query r (O.view r resp n)) =
        resp' (O.view r resp n) (O.query r (O.view r resp n))) → O.view r resp q = O.view r resp' q
  | 0, _ => rfl
  | q + 1, h => by
    have ih := Outsider.view_congr O r resp resp' q fun n hn => h n (Nat.lt_succ_of_lt hn)
    show (O.query r (O.view r resp q), resp (O.view r resp q) (O.query r (O.view r resp q))) :: O.view r resp q =
      (O.query r (O.view r resp' q), resp' (O.view r resp' q) (O.query r (O.view r resp' q))) :: O.view r resp' q
    rw [h q (Nat.lt_succ_self q), ← ih]

end View

namespace Served

variable {O V Tick Adv R Q Θ Θ' : Type}

theorem count_le_length (L : ℕ) : ∀ l : List (Served O V Tick Adv), count L l ≤ l.length
  | [] => le_rfl
  | w :: l => by
    have := count_le_length L l
    simp only [count, List.length_cons]
    split_ifs
    · exact Nat.succ_le_succ this
    · exact Nat.succ_le_succ (min_le_left _ _)

/-- **The halt comes at most `L + 1` windows after the accepted ones.** -/
theorem count_le_accLen (L : ℕ) : ∀ l : List (Served O V Tick Adv), count L l ≤ accLen l + (L + 1)
  | [] => Nat.zero_le _
  | w :: l => by
    have := count_le_accLen L l
    simp only [count, accLen]
    split_ifs
    · omega
    · rw [zero_add]
      exact Nat.succ_le_succ (min_le_right _ _)

theorem accLen_le_count (L : ℕ) : ∀ l : List (Served O V Tick Adv), accLen l ≤ count L l
  | [] => le_rfl
  | w :: l => by
    have := accLen_le_count L l
    simp only [accLen, count]
    split_ifs <;> omega

/-- **The windows before the first rejection are accepted.** -/
theorem accepted_of_lt_accLen : ∀ (l : List (Served O V Tick Adv)) (n : ℕ) (w : Served O V Tick Adv),
    n < accLen l → l[n]? = some w → w.accepted
  | [], _, _, h, _ => by simp [accLen] at h
  | v :: l, n, w, h, hw => by
    by_cases hv : v.accepted
    · rcases n with _ | n
      · simp only [List.getElem?_cons_zero, Option.some.injEq] at hw
        exact hw ▸ hv
      · simp only [accLen, hv, ite_true] at h
        exact accepted_of_lt_accLen l n w (by omega) (by simpa using hw)
    · simp [accLen, hv] at h

/-- **The windows accepted before the first rejection count in `lead`** when no accepted window sampled an incorrect
proof block: each of them sampled none. -/
theorem sum_incorrectAt_le_lead : ∀ l : List (Served O V Tick Adv), (¬ ∃ w ∈ l, w.accepted ∧ w.sampledIncorrect) →
    ∑ n ∈ range (accLen l), incorrectAt l n ≤ lead l
  | [] => fun _ => by simp [accLen]
  | w :: l => fun h => by
    by_cases ha : w.accepted
    · have hd : ¬ w.sampledIncorrect := fun hd => h ⟨w, List.mem_cons_self, ha, hd⟩
      have ih := sum_incorrectAt_le_lead l fun ⟨v, hv, hva⟩ => h ⟨v, List.mem_cons_of_mem _ hv, hva⟩
      have hs : ∀ n, incorrectAt (w :: l) (n + 1) = incorrectAt l n := fun n => by simp [incorrectAt]
      have h0 : incorrectAt (w :: l) 0 = w.incorrect := by simp [incorrectAt]
      simp only [accLen, lead, ite_eq_left ha, ite_eq_right hd]
      rw [sum_range_succ', h0]
      simp only [hs]
      omega
    · simp [accLen, ha]

variable [Fintype O]

/-- **Local releases are isolated**: each window's bound, summed. -/
theorem isolated_of_local (C : Θ → Q × Adv → O → V) (E : ℕ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv)) (hloc : Local C E Ou L θ r l) :
    Isolated C E Ou L θ r l := by
  intro k hk
  rw [mul_sum]
  refine sum_le_sum fun n hn => ?_
  have hn' : n < count L l := lt_of_lt_of_le (mem_range.1 hn) (hk.trans (accLen_le_count L l))
  have hlen : n < l.length := lt_of_lt_of_le hn' (count_le_length L l)
  have hw : l[n]? = some l[n] := List.getElem?_eq_getElem hlen
  refine (hloc n hn' _ hw).trans (le_of_eq ?_)
  simp [incorrectAt, hw]

/-- **Through the halt, the releases change at most `F · lead + (L + 1) · |O|` values** when the releases are isolated
and no accepted window sampled an incorrect proof block: through the last window accepted before the first rejection,
at most `F` per incorrect proof block of those windows, which count in `lead`, and after it at most `|O|` a window. -/
theorem sum_card_changed_le (C : Θ → Q × Adv → O → V) (F : ℕ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv)) (hiso : Isolated C F Ou L θ r l)
    (hacc : ¬ ∃ w ∈ l, w.accepted ∧ w.sampledIncorrect) :
    ∑ n ∈ range (count L l), (changed C Ou L θ r l n).card ≤ F * lead l + (L + 1) * Fintype.card O := by
  have hca := count_le_accLen L l
  rw [← sum_range_add_sum_Ico _ (accLen_le_count L l)]
  apply add_le_add
  · calc ∑ n ∈ range (accLen l), (changed C Ou L θ r l n).card
        ≤ F * ∑ n ∈ range (accLen l), incorrectAt l n := hiso _ le_rfl
      _ ≤ F * lead l := Nat.mul_le_mul_left _ (sum_incorrectAt_le_lead l hacc)
  · calc ∑ n ∈ Ico (accLen l) (count L l), (changed C Ou L θ r l n).card
        ≤ ∑ _n ∈ Ico (accLen l) (count L l), Fintype.card O := sum_le_sum fun n _ => card_le_univ _
      _ = (count L l - accLen l) * Fintype.card O := by rw [sum_const, Nat.card_Ico, smul_eq_mul]
      _ ≤ (L + 1) * Fintype.card O := Nat.mul_le_mul_right _ (by omega)

variable [DecidableEq O]

/-- The values a run's releases changed through the halt, by window and slot: the overwrite of its covert string. -/
noncomputable def overwrite (C : Θ → Q × Adv → O → V) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv)) (T : ℕ) : Fin T × O → Option V :=
  fun p => if p.1.val < count L l ∧ p.2 ∈ changed C Ou L θ r l p.1.val then l[p.1.val]?.map (·.egress p.2) else none

theorem card_overwrite_le (C : Θ → Q × Adv → O → V) (F M : ℕ)
    (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv))
    (T : ℕ) (hiso : Isolated C F Ou L θ r l) (hesc : ¬ Undetected M l) :
    (univ.filter fun p => overwrite C Ou L θ r l T p ≠ none).card ≤ F * M + (L + 1) * Fintype.card O := by
  classical
  have hacc : ¬ ∃ w ∈ l, w.accepted ∧ w.sampledIncorrect := fun h => hesc (Or.inl h)
  have hlead : lead l < M := lt_of_not_ge fun h => hesc (Or.inr h)
  calc (univ.filter fun p => overwrite C Ou L θ r l T p ≠ none).card
      ≤ ((range (count L l)).sigma fun n => changed C Ou L θ r l n).card := by
        refine card_le_card_of_injOn (fun p => ⟨p.1.val, p.2⟩) (fun p hp => ?_) (fun p _ p' _ h => ?_)
        · have hp' : overwrite C Ou L θ r l T p ≠ none := (mem_filter.1 hp).2
          have hc : p.1.val < count L l ∧ p.2 ∈ changed C Ou L θ r l p.1.val := by
            by_contra hc
            exact hp' (ite_eq_right hc)
          exact mem_sigma.2 ⟨mem_range.2 hc.1, hc.2⟩
        · simp only [Sigma.mk.inj_iff, heq_eq_eq] at h
          exact Prod.ext (Fin.ext h.1) h.2
    _ = ∑ n ∈ range (count L l), (changed C Ou L θ r l n).card := card_sigma _ _
    _ ≤ F * lead l + (L + 1) * Fintype.card O := sum_card_changed_le C F Ou L θ r l hiso hacc
    _ ≤ F * M + (L + 1) * Fintype.card O := by have := hlead.le; gcongr

variable [Fintype V] [Fintype Adv] [Inhabited Adv] [Fintype Tick] [Inhabited Tick]

/-- **An isolated run that leaves fewer than `M` incorrect proof blocks undetected is the served channel's deviation
with its own covert string**: each window's advice and bucket, the run's overwrite and its halt. -/
theorem view_eq_resp (C : Θ → Q × Adv → O → V) (F M : ℕ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (T L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv)) (hiso : Isolated C F Ou L θ r l)
    (hesc : ¬ Undetected M l) :
    ∃ s : (servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)).S,
      view Ou L r l T =
        Ou.view r ((servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)).resp C s θ r) T := by
  classical
  have hcl := count_le_length L l
  refine ⟨(fun t => (l[t.val]?.map (·.adv)).getD default, fun t => (l[t.val]?.map (·.tick)).getD default,
    ⟨overwrite C Ou L θ r l T, card_overwrite_le C F M Ou L θ r l T hiso hesc⟩,
    ⟨min (count L l) T, Nat.lt_succ_of_le (min_le_right _ _)⟩), ?_⟩
  refine Outsider.view_congr Ou r _ _ T fun n hn => ?_
  have hlen := Outsider.view_length Ou r (fun h _ => reply L l h.length) n
  have hv : view Ou L r l n = Ou.view r (fun h _ => reply L l h.length) n := rfl
  generalize Ou.view r (fun h _ => reply L l h.length) n = V at hlen hv ⊢
  subst hlen
  simp only [Channel.resp, servedChannel]
  rw [reply]
  by_cases hnc : V.length < count L l
  · have hw : l[V.length]? = some l[V.length] := List.getElem?_eq_getElem (lt_of_lt_of_le hnc hcl)
    rw [ite_eq_left hnc, dite_eq_left ⟨lt_min hnc hn, hn⟩, dite_eq_left hn]
    simp only [hw, Option.map, Option.getD]
    refine congrArg some (Prod.ext (funext fun o => ?_) rfl)
    by_cases ho : o ∈ changed C Ou L θ r l V.length
    · simp [overwrite, hnc, ho, hw]
    · have he : l[V.length].egress o = C θ ((Ou.query r (view Ou L r l V.length)).1, l[V.length].adv) o := by
        simpa [changed, hw] using ho
      rw [hv] at he
      simp only [overwrite, hnc, ho, and_false, ↓reduceIte]
      exact he
  · rw [ite_eq_right hnc, dite_eq_right fun h => hnc (lt_of_lt_of_le h.1 (min_le_left _ _))]

end Served

namespace SecurityProofs

open Verity.Game WeightSecurity.Assumptions
open scoped ENNReal

open Classical in
theorem avg_indicator {α : Type} [Fintype α] (P : α → Prop) :
    avg (fun z => if P z then (1 : ℝ≥0∞) else 0) = ENNReal.ofReal (Claims.Pr P) := by
  unfold avg Claims.Pr Claims.prUniform
  rw [← mul_sum, sum_boole]
  rcases Nat.eq_zero_or_pos (Fintype.card α) with h0 | hpos
  · have : IsEmpty α := Fintype.card_eq_zero_iff.1 h0
    simp
  · rw [ENNReal.ofReal_div_of_pos (by exact_mod_cast hpos), ENNReal.ofReal_natCast, ENNReal.ofReal_natCast,
      div_eq_mul_inv, mul_comm]

theorem avg_sum_univ {α ι : Type} [Fintype α] [Fintype ι] (f : ι → α → ℝ≥0∞) :
    avg (fun z => ∑ i, f i z) = ∑ i, avg (f i) := by
  unfold avg
  simp only [mul_sum]
  exact sum_comm

theorem avg_le_const {α : Type} [Fintype α] (x : ℝ≥0∞) : avg (fun _ : α => x) ≤ x := by
  rcases isEmpty_or_nonempty α with h | h
  · simp [avg]
  · exact (avg_const x).le

theorem value_const_le {α : Type} (P : Prop) [Decidable P] (g : Verity.Game α) :
    value (fun _ => P) g ≤ if P then 1 else 0 := by
  by_cases hP : P
  · rw [ite_eq_left hP]
    exact value_le_one _ _
  · rw [ite_eq_right hP]
    exact (value_mono (q := fun _ => False) (fun _ h => hP h) g).trans_eq (value_false g)

theorem ite_exists_le_sum {ι : Type} [Fintype ι] (P : ι → Prop) [Decidable (∃ i, P i)] [DecidablePred P] :
    (if ∃ i, P i then (1 : ℝ≥0∞) else 0) ≤ ∑ i, if P i then 1 else 0 := by
  by_cases h : ∃ i, P i
  · obtain ⟨i, hi⟩ := h
    rw [ite_eq_left ⟨i, hi⟩]
    calc (1 : ℝ≥0∞) = if P i then 1 else 0 := (ite_eq_left hi).symm
      _ ≤ ∑ i, if P i then 1 else 0 :=
        single_le_sum (f := fun i => if P i then (1 : ℝ≥0∞) else 0) (fun _ _ => zero_le) (mem_univ i)
  · rw [ite_eq_right h]
    exact zero_le

/-- **The leakage lemma over a game of coins.** Every party's run is a game `g z` at the weights' and outsider's coins
`z`. Wherever the outsider wins, either the bad event happens or some honest-use party `O.withString ch s` steals the
weights. Then the outsider wins with probability at most the bad event's plus `|ch.S| · p`, the union bound over the
covert strings with `HonestUseHard` for each. -/
theorem value_leak {Ω R Θ X Y M Θ' α : Type} [Fintype Ω] [Fintype R] (C : Θ → X → Y) (gen : Ω → Θ)
    (Stolen : Θ → Θ' → Prop) (q : ℕ) (A : Set (Outsider R X Y Θ')) (p : ℝ) (O : Outsider R X M Θ')
    (ch : Channel R X Y M) (hhon : HonestUseHard C gen Stolen q A p) (hsim : Simulable A O ch)
    (g : Ω × R → Verity.Game α) (Win Bad : Ω × R → α → Prop)
    (hcov : ∀ z a, Win z a →
      Bad z a ∨ ∃ s : ch.S, Stolen (gen z.1) ((O.withString ch s).model z.2 (honest C (gen z.1)) q)) :
    avg (fun z => value (Win z) (g z)) ≤
      avg (fun z => value (Bad z) (g z)) + ENNReal.ofReal (Fintype.card ch.S * p) := by
  classical
  have hz : ∀ z, value (Win z) (g z) ≤ value (Bad z) (g z) + ∑ s : ch.S,
      if Stolen (gen z.1) ((O.withString ch s).model z.2 (honest C (gen z.1)) q) then 1 else 0 := fun z =>
    calc value (Win z) (g z)
        ≤ value (fun a => Bad z a ∨ ∃ s : ch.S, Stolen (gen z.1) ((O.withString ch s).model z.2 (honest C (gen z.1)) q))
            (g z) := value_mono (hcov z) _
      _ ≤ value (Bad z) (g z) +
            value (fun _ => ∃ s : ch.S, Stolen (gen z.1) ((O.withString ch s).model z.2 (honest C (gen z.1)) q)) (g z) :=
          value_or_le _ _ _
      _ ≤ _ := add_le_add le_rfl ((value_const_le _ _).trans (ite_exists_le_sum _))
  have hs : ∀ s : ch.S, avg (fun z : Ω × R =>
      if Stolen (gen z.1) ((O.withString ch s).model z.2 (honest C (gen z.1)) q) then (1 : ℝ≥0∞) else 0) ≤
        ENNReal.ofReal p := fun s =>
    (avg_indicator _).trans_le (ENNReal.ofReal_le_ofReal (hhon _ (hsim s)))
  calc avg (fun z => value (Win z) (g z))
      ≤ avg (fun z => value (Bad z) (g z) + ∑ s : ch.S,
          if Stolen (gen z.1) ((O.withString ch s).model z.2 (honest C (gen z.1)) q) then (1 : ℝ≥0∞) else 0) :=
        avg_mono hz
    _ = avg (fun z => value (Bad z) (g z)) + ∑ s : ch.S, avg (fun z : Ω × R =>
          if Stolen (gen z.1) ((O.withString ch s).model z.2 (honest C (gen z.1)) q) then (1 : ℝ≥0∞) else 0) := by
        rw [avg_add, avg_sum_univ]
    _ ≤ avg (fun z => value (Bad z) (g z)) + ∑ _s : ch.S, ENNReal.ofReal p := by
        gcongr with s
        exact hs s
    _ = avg (fun z => value (Bad z) (g z)) + ENNReal.ofReal (Fintype.card ch.S * p) := by
        rw [sum_const, card_univ, nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg _), ENNReal.ofReal_natCast]

/-- **The served channel's covert strings**: advice and a bucket per window, an overwrite of at most `K` of the `T · |O|`
values, and the halt. -/
theorem card_servedChannel_le (R Q Adv O V Tick : Type) [Fintype Adv] [Fintype O] [DecidableEq O] [Fintype V]
    [Nonempty V] [Fintype Tick] (T K : ℕ) :
    Fintype.card (servedChannel R Q Adv O V Tick T K).S ≤
      Fintype.card Adv ^ T * Fintype.card Tick ^ T *
        ((∑ j ∈ range (K + 1), (T * Fintype.card O).choose j) * Fintype.card V ^ K) * (T + 1) := by
  classical
  have hsub : Fintype.card {d : Fin T × O → Option V // (univ.filter fun p => d p ≠ none).card ≤ K} ≤
      (∑ j ∈ range (K + 1), (T * Fintype.card O).choose j) * Fintype.card V ^ K := by
    have hball := Population.Exfil.card_devBall_le (Fin T × O) V K
    rw [Fintype.card_prod, Fintype.card_fin] at hball
    refine le_trans ?_ hball
    let img : {d : Fin T × O → Option V // (univ.filter fun p => d p ≠ none).card ≤ K} →
        Population.Exfil.Dev (Fin T × O) V :=
      fun d => ⟨univ.filter fun p => d.1 p ≠ none, fun p => (d.1 p.1).getD (Classical.arbitrary V)⟩
    have hrec : ∀ d, (fun p => if h : p ∈ (img d).1 then some ((img d).2 ⟨p, h⟩) else none) = d.1 := by
      intro d
      funext p
      by_cases hp : d.1 p = none
      · have : p ∉ (img d).1 := by simp [img, hp]
        rw [dite_eq_right this, hp]
      · have hm : p ∈ (img d).1 := by simp [img, hp]
        rw [dite_eq_left hm]
        obtain ⟨v, hv⟩ := Option.ne_none_iff_exists'.1 hp
        simp [img, hv]
    rw [← card_univ]
    refine card_le_card_of_injOn img (fun d _ => ?_) (fun d _ d' _ h => ?_)
    · simp only [Finset.mem_coe, Population.Exfil.devBall, mem_sigma, mem_filter, mem_univ, true_and, and_true]
      exact d.2
    · exact Subtype.ext ((hrec d).symm.trans ((congrArg (fun D : Population.Exfil.Dev (Fin T × O) V =>
        fun p => if h : p ∈ D.1 then some (D.2 ⟨p, h⟩) else none) h).trans (hrec d')))
  calc Fintype.card (servedChannel R Q Adv O V Tick T K).S
      = Fintype.card ((Fin T → Adv) × (Fin T → Tick) ×
          {d : Fin T × O → Option V // (univ.filter fun p => d p ≠ none).card ≤ K} × Fin (T + 1)) :=
        Fintype.card_congr' rfl
    _ = Fintype.card Adv ^ T * (Fintype.card Tick ^ T *
          (Fintype.card {d : Fin T × O → Option V // (univ.filter fun p => d p ≠ none).card ≤ K} * (T + 1))) := by
        simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]
    _ ≤ Fintype.card Adv ^ T * (Fintype.card Tick ^ T *
          ((∑ j ∈ range (K + 1), (T * Fintype.card O).choose j) * Fintype.card V ^ K * (T + 1))) := by
        gcongr
    _ = _ := by ring

/-- **At most `K` of `N` positions: at most `(N + 1)^K` choices**, the binomial expansion's terms bounding the
binomial coefficients, `N.choose j ≤ N^j`. -/
theorem sum_choose_le_pow (N K : ℕ) : ∑ j ∈ range (K + 1), N.choose j ≤ (N + 1) ^ K := by
  rw [add_pow]
  refine sum_le_sum fun j hj => ?_
  have hjK : j ≤ K := Nat.lt_succ_iff.1 (mem_range.1 hj)
  have hc : N.choose j ≤ N ^ j * K.choose j :=
    calc N.choose j ≤ j.factorial * N.choose j := Nat.le_mul_of_pos_left _ (Nat.factorial_pos j)
      _ = N.descFactorial j := (Nat.descFactorial_eq_factorial_mul_choose N j).symm
      _ ≤ N ^ j := Nat.descFactorial_le_pow N j
      _ ≤ N ^ j * K.choose j := Nat.le_mul_of_pos_right _ (Nat.choose_pos hjK)
  simpa using hc

/-- **Two to the served channel's bits** is the product of its parts' counts. -/
theorem rpow_servedBits (Adv O V Tick : Type) [Fintype Adv] [Inhabited Adv] [Fintype O] [Fintype V] [Nonempty V]
    [Fintype Tick] [Inhabited Tick] (T K : ℕ) :
    (2 : ℝ) ^ servedBits Adv O V Tick T K =
      (Fintype.card Adv * Fintype.card Tick) ^ T * (Fintype.card V * (T * Fintype.card O + 1)) ^ K * (T + 1) := by
  have e : ∀ x : ℝ, 0 < x → (2 : ℝ) ^ Real.logb 2 x = x := fun x hx => Real.rpow_logb two_pos (by norm_num) hx
  have m : ∀ (n : ℕ) (y : ℝ), (2 : ℝ) ^ ((n : ℝ) * y) = ((2 : ℝ) ^ y) ^ n := fun n y => by
    rw [mul_comm, Real.rpow_mul zero_le_two, Real.rpow_natCast]
  have hA : (0 : ℝ) < Fintype.card Adv := by exact_mod_cast Fintype.card_pos
  have hTk : (0 : ℝ) < Fintype.card Tick := by exact_mod_cast Fintype.card_pos
  have hV : (0 : ℝ) < Fintype.card V := by exact_mod_cast Fintype.card_pos
  have hN : (0 : ℝ) < T * Fintype.card O + 1 := by positivity
  have hT : (0 : ℝ) < T + 1 := by positivity
  unfold servedBits
  rw [Real.rpow_add two_pos, Real.rpow_add two_pos, m, m, Real.rpow_add two_pos, Real.rpow_add two_pos, e _ hA,
    e _ hTk, e _ hV, e _ hN, e _ hT]

/-- **The served channel's covert strings number at most two to its bits**: `card_servedChannel_le` with the
overwrites' positions counted at most `(T · |O| + 1)^K`. -/
theorem card_servedChannel_le_rpow (R Q Adv O V Tick : Type) [Fintype Adv] [Inhabited Adv] [Fintype O] [DecidableEq O]
    [Fintype V] [Nonempty V] [Fintype Tick] [Inhabited Tick] (T K : ℕ) :
    (Fintype.card (servedChannel R Q Adv O V Tick T K).S : ℝ) ≤ (2 : ℝ) ^ servedBits Adv O V Tick T K := by
  rw [rpow_servedBits, mul_pow, mul_pow]
  have hs := sum_choose_le_pow (T * Fintype.card O) K
  have hle : Fintype.card (servedChannel R Q Adv O V Tick T K).S ≤
      Fintype.card Adv ^ T * Fintype.card Tick ^ T * (Fintype.card V ^ K * (T * Fintype.card O + 1) ^ K) * (T + 1) :=
    calc Fintype.card (servedChannel R Q Adv O V Tick T K).S
        ≤ Fintype.card Adv ^ T * Fintype.card Tick ^ T *
            ((∑ j ∈ range (K + 1), (T * Fintype.card O).choose j) * Fintype.card V ^ K) * (T + 1) :=
          card_servedChannel_le R Q Adv O V Tick T K
      _ ≤ Fintype.card Adv ^ T * Fintype.card Tick ^ T *
            ((T * Fintype.card O + 1) ^ K * Fintype.card V ^ K) * (T + 1) :=
          Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ (Nat.mul_le_mul_right _ hs))
      _ = _ := by rw [Nat.mul_comm ((T * Fintype.card O + 1) ^ K) (Fintype.card V ^ K)]
  exact_mod_cast hle

/-- **`TheftServed` at isolated releases**: the bound with the probability that the releases aren't isolated, each
incorrect proof block's changes counted wherever they fall. On a run whose releases are isolated and whose audit leaves
fewer than `M` incorrect proof blocks undetected, the outsider's view is the served channel's deviation with the run's
own covert string (`Served.view_eq_resp`). -/
theorem theftServed_isolated {Ω R Θ Q Adv O V Tick Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv]
    [Fintype O] [DecidableEq O] [Fintype V] [Fintype Tick] [Inhabited Tick]
    (C : Θ → Q × Adv → O → V) (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop) (A : Set (Outsider R (Q × Adv) (O → V) Θ'))
    (p : ℝ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (T L F M : ℕ) (δ : ℝ≥0∞)
    (run : Θ → R → Verity.Game (List (Served O V Tick Adv))) (hhon : HonestUseHard C gen Stolen T A p)
    (hsim : Simulable A Ou (servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)))
    (hesc : Served.UndetectedLe M δ run) :
    avg (fun z : Ω × R => value (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T)))
        (run (gen z.1) z.2)) ≤
      ENNReal.ofReal (Fintype.card (servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)).S * p) +
        δ + avg (fun z : Ω × R => value (fun l => ¬ Served.Isolated C F Ou L (gen z.1) z.2 l) (run (gen z.1) z.2)) := by
  classical
  have hcov : ∀ (z : Ω × R) (l : List (Served O V Tick Adv)),
      Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T)) →
        (¬ Served.Isolated C F Ou L (gen z.1) z.2 l ∨ Served.Undetected M l) ∨
          ∃ s : (servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)).S, Stolen (gen z.1)
            ((Ou.withString (servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)) s).model z.2
              (honest C (gen z.1)) T) := by
    intro z l hst
    by_cases hiso : Served.Isolated C F Ou L (gen z.1) z.2 l
    · by_cases he : Served.Undetected M l
      · exact Or.inl (Or.inr he)
      · obtain ⟨s, hs⟩ := Served.view_eq_resp C F M Ou T L (gen z.1) z.2 l hiso he
        refine Or.inr ⟨s, ?_⟩
        rw [withString_model, Outsider.model, ← hs]
        exact hst
    · exact Or.inl (Or.inl hiso)
  refine (value_leak C gen Stolen T A p Ou _ hhon hsim (fun z => run (gen z.1) z.2)
    (fun z l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T)))
    (fun z l => ¬ Served.Isolated C F Ou L (gen z.1) z.2 l ∨ Served.Undetected M l) hcov).trans ?_
  calc avg (fun z : Ω × R => value (fun l => ¬ Served.Isolated C F Ou L (gen z.1) z.2 l ∨ Served.Undetected M l)
          (run (gen z.1) z.2)) + ENNReal.ofReal
          (Fintype.card (servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)).S * p)
      ≤ avg (fun z : Ω × R => value (fun l => ¬ Served.Isolated C F Ou L (gen z.1) z.2 l) (run (gen z.1) z.2) + δ) +
          ENNReal.ofReal (Fintype.card (servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)).S * p) := by
        refine add_le_add (avg_mono fun z => ?_) le_rfl
        exact (value_or_le _ _ _).trans (add_le_add le_rfl (hesc _ _))
    _ ≤ (avg (fun z : Ω × R => value (fun l => ¬ Served.Isolated C F Ou L (gen z.1) z.2 l) (run (gen z.1) z.2)) + δ) +
          ENNReal.ofReal (Fintype.card (servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)).S * p) := by
        rw [avg_add]
        gcongr
        exact avg_le_const δ
    _ = _ := by ring

/-- **`TheftServed` as a simulation**, for any event and not only a stolen model. Except where the releases aren't
isolated or the audit leaves `M` incorrect proof blocks undetected, the outsider's whole view of the deployment is what
it rebuilds (`Outsider.rebuild`) from one covert string `s` of the served channel and the answers that honest use of the
served function gives the outsider with `s` written in: the served function's API plus one message of at most
`servedBits` bits (`card_servedChannel_le_rpow`). Whatever the outsider ends up holding is a function of that view. -/
theorem theftServed_simulation {Ω R Θ Q Adv O V Tick Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv]
    [Fintype O] [DecidableEq O] [Fintype V] [Fintype Tick] [Inhabited Tick]
    (C : Θ → Q × Adv → O → V) (gen : Ω → Θ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (T L F M : ℕ)
    (δ : ℝ≥0∞) (run : Θ → R → Verity.Game (List (Served O V Tick Adv))) (hesc : Served.UndetectedLe M δ run) :
    avg (fun z : Ω × R => value (fun l => ¬ ∃ s : (servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)).S,
        Served.view Ou L z.2 l T = Ou.rebuild (servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)) s z.2
          ((Ou.withString (servedChannel R Q Adv O V Tick T (F * M + (L + 1) * Fintype.card O)) s).view z.2
            (honest C (gen z.1)) T)) (run (gen z.1) z.2)) ≤
      δ + avg (fun z : Ω × R => value (fun l => ¬ Served.Isolated C F Ou L (gen z.1) z.2 l) (run (gen z.1) z.2)) := by
  classical
  calc _ ≤ avg (fun z : Ω × R => value (Served.Undetected M) (run (gen z.1) z.2) +
          value (fun l => ¬ Served.Isolated C F Ou L (gen z.1) z.2 l) (run (gen z.1) z.2)) := by
        refine avg_mono fun z => (value_mono (fun l hl => ?_) _).trans (value_or_le _ _ _)
        by_contra h
        obtain ⟨s, hs⟩ := Served.view_eq_resp C F M Ou T L (gen z.1) z.2 l (not_not.mp (not_or.mp h).2)
          (not_or.mp h).1
        exact hl ⟨s, hs.trans (rebuild_view Ou _ C s (gen z.1) z.2 T).symm⟩
    _ ≤ _ := by
        rw [avg_add]
        exact add_le_add ((avg_mono fun z => hesc _ _).trans (avg_le_const δ)) le_rfl

theorem TheftServed : Guarantees.TheftServed := by
  intro Ω R Θ Q Adv O V Tick Θ' _ _ _ _ _ _ _ _ _ C gen Stolen A p Ou T L E M δ run hhon hsim hesc
  exact (theftServed_isolated C gen Stolen A p Ou T L E M δ run hhon hsim hesc).trans (add_le_add le_rfl
    (avg_mono fun z => value_mono (fun l hl hloc => hl (Served.isolated_of_local C E Ou L _ _ l hloc)) _))

theorem TheftServedReq : Guarantees.TheftServedReq := by
  intro Ω R Θ Q Adv U W Tick Θ' _ _ _ _ _ _ _ _ _ _ C gen Stolen A p Ou T L F M δ run hhon hsim hesc
  refine (theftServed_isolated C gen Stolen A p Ou T L F M δ run hhon hsim hesc).trans
    (add_le_add (add_le_add ?_ le_rfl) le_rfl)
  rcases le_or_gt 0 p with hp | hp
  · exact ENNReal.ofReal_le_ofReal
      (mul_le_mul_of_nonneg_right (card_servedChannel_le_rpow R Q Adv U W Tick T _) hp)
  · rw [ENNReal.ofReal_of_nonpos (mul_nonpos_of_nonneg_of_nonpos (Nat.cast_nonneg _) hp.le)]
    exact zero_le

theorem TheftServedMinEntropy : Guarantees.TheftServedMinEntropy := by
  intro Ω R Θ Q Adv U W Tick Θ' _ _ _ _ _ _ _ _ _ _ C gen Stolen K Ou T L F M δ run hhon hesc
  have h := TheftServedReq C gen Stolen Set.univ ((2 : ℝ) ^ (-K)) Ou T L F M δ run hhon
    (fun _ => Set.mem_univ _) hesc
  rwa [← Real.rpow_add two_pos, ← sub_eq_add_neg] at h

namespace ToyServed

open WeightSecurity.ToyServed

theorem card_W : Fintype.card (W × Unit) = 256 := by simp [W]

theorem honest_C (θ : W) : honest C θ = honest C (0, 0) := rfl

/-- Honest use of the toy's served function reveals nothing: every party's output is one fixed guess. -/
theorem guess_le (P : Outsider Unit (Unit × Unit) (Unit → Fin 16) W) :
    Claims.Pr (fun z : W × Unit => Stolen (gen z.1) (P.model z.2 (honest C (gen z.1)) 2)) ≤ 1 / 256 := by
  calc Claims.Pr (fun z : W × Unit => Stolen (gen z.1) (P.model z.2 (honest C (gen z.1)) 2))
      ≤ Pouw.pr (fun z : W × Unit => z.1 = P.model () (honest C (0, 0)) 2) := Pouw.pr_mono fun z h => by
        obtain ⟨θ, ⟨⟩⟩ := z
        have h' : P.model () (honest C θ) 2 = θ := h
        rw [honest_C] at h'
        exact h'.symm
    _ = 1 / 256 := by
        rw [Pouw.pr_eq_card _ {(P.model () (honest C (0, 0)) 2, ())} fun z => by simp [Prod.ext_iff], card_singleton,
          card_W]
        norm_num

/-- The honest run's audit leaves no incorrect proof block undetected. -/
theorem honest_undetected : Served.UndetectedLe 1 0 WeightSecurity.ToyServed.honestRun := by
  classical
  intro θ r
  refine le_of_eq ?_
  unfold WeightSecurity.ToyServed.honestRun
  rw [value_ret, ite_eq_right]
  rintro (⟨w, hw, -, hd⟩ | h)
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at hw
    rcases hw with rfl | rfl <;> exact hd
  · simp [Served.lead] at h

/-- The thief steals the weights from the leaking run on every coin. -/
theorem leak_steals (z : WeightSecurity.ToyServed.W × Unit) : value (fun l => WeightSecurity.ToyServed.Stolen
    (WeightSecurity.ToyServed.gen z.1) (WeightSecurity.ToyServed.thief.out z.2
      (Served.view WeightSecurity.ToyServed.thief 0 z.2 l 2)))
    (WeightSecurity.ToyServed.leakRun (WeightSecurity.ToyServed.gen z.1) z.2) = 1 := by
  classical
  unfold WeightSecurity.ToyServed.leakRun
  rw [value_ret, ite_eq_left]
  simp [WeightSecurity.ToyServed.Stolen, WeightSecurity.ToyServed.gen, WeightSecurity.ToyServed.thief, Served.view,
    Served.reply, Served.count, Outsider.view]

/-- The leaking run's releases change one slot per window, which has one incorrect proof block. -/
theorem leak_local (z : WeightSecurity.ToyServed.W × Unit) : value (fun l => ¬ Served.Local WeightSecurity.ToyServed.C 1
    WeightSecurity.ToyServed.thief 0 (WeightSecurity.ToyServed.gen z.1) z.2 l)
    (WeightSecurity.ToyServed.leakRun (WeightSecurity.ToyServed.gen z.1) z.2) = 0 := by
  classical
  unfold WeightSecurity.ToyServed.leakRun
  rw [value_ret, ite_eq_right (not_not.2 ?_)]
  intro n _ w hw
  have hwr : w.incorrect = 1 := by
    rcases n with _ | _ | n
    · simp only [List.getElem?_cons_zero, Option.some.injEq] at hw
      rw [← hw]
    · simp only [List.getElem?_cons_succ, List.getElem?_cons_zero, Option.some.injEq] at hw
      rw [← hw]
    · simp at hw
  rw [hwr, mul_one]
  exact (card_le_univ _).trans_eq Fintype.card_unit

theorem leak_isolated (z : WeightSecurity.ToyServed.W × Unit) : value (fun l => ¬ Served.Isolated
    WeightSecurity.ToyServed.C 1 WeightSecurity.ToyServed.thief 0 (WeightSecurity.ToyServed.gen z.1) z.2 l)
    (WeightSecurity.ToyServed.leakRun (WeightSecurity.ToyServed.gen z.1) z.2) = 0 :=
  le_antisymm ((value_mono (fun l hl hloc => hl (Served.isolated_of_local WeightSecurity.ToyServed.C 1
    WeightSecurity.ToyServed.thief 0 _ _ l hloc)) _).trans_eq (leak_local z)) zero_le

/-- The toy's covert capacity at one changed unit: `16` values, `3` names and `3` halts. -/
theorem rpow_bits :
    (2 : ℝ) ^ servedBits Unit Unit (Fin 16) Unit 2 (1 * 0 + (0 + 1) * Fintype.card Unit) = 144 := by
  rw [rpow_servedBits]
  norm_num [Fintype.card_unit, Fintype.card_fin]

theorem rpow_neg_eight : (2 : ℝ) ^ (-(8 : ℝ)) = 1 / 256 := by
  rw [Real.rpow_neg zero_le_two, show (8 : ℝ) = ((8 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  norm_num

theorem rpow_bits_sub :
    (2 : ℝ) ^ (servedBits Unit Unit (Fin 16) Unit 2 (1 * 0 + (0 + 1) * Fintype.card Unit) - 8) = 144 / 256 := by
  rw [Real.rpow_sub two_pos, rpow_bits, show (8 : ℝ) = ((8 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  norm_num

end ToyServed

theorem TheftServedHonest : Guarantees.TheftServedHonest :=
  ⟨fun P _ => ToyServed.guess_le P, fun _ => Set.mem_univ _, ToyServed.honest_undetected⟩

theorem TheftServedNontrivial : Guarantees.TheftServedNontrivial := by
  classical
  intro h
  simp only [ToyServed.leak_steals, ToyServed.leak_local, avg_const, add_zero] at h
  have hc : Fintype.card (servedChannel Unit Unit Unit Unit (Fin 16) Unit 2
      (1 * 0 + (0 + 1) * Fintype.card Unit)).S ≤ 144 :=
    (card_servedChannel_le Unit Unit Unit Unit (Fin 16) Unit 2 _).trans (by simp [Finset.sum_range_succ])
  have hlt : ENNReal.ofReal ((Fintype.card (servedChannel Unit Unit Unit Unit (Fin 16) Unit 2
      (1 * 0 + (0 + 1) * Fintype.card Unit)).S : ℝ) * (1 / 256)) < 1 := by
    rw [ENNReal.ofReal_lt_one]
    have : (Fintype.card (servedChannel Unit Unit Unit Unit (Fin 16) Unit 2
        (1 * 0 + (0 + 1) * Fintype.card Unit)).S : ℝ) ≤ 144 := by exact_mod_cast hc
    linarith
  exact absurd h (not_le.2 hlt)

theorem TheftServedReqNontrivial : Guarantees.TheftServedReqNontrivial := by
  intro h
  simp only [ToyServed.leak_steals, ToyServed.leak_isolated, ToyServed.rpow_bits, avg_const, add_zero] at h
  have hlt : ENNReal.ofReal ((144 : ℝ) * (1 / 256)) < 1 := by
    rw [ENNReal.ofReal_lt_one]
    norm_num
  exact absurd h (not_le.2 hlt)

theorem TheftServedMinEntropyHonest : Guarantees.TheftServedMinEntropyHonest :=
  ⟨fun P _ => (ToyServed.guess_le P).trans_eq ToyServed.rpow_neg_eight.symm, ToyServed.honest_undetected⟩

theorem TheftServedMinEntropyNontrivial : Guarantees.TheftServedMinEntropyNontrivial := by
  intro h
  simp only [ToyServed.leak_steals, ToyServed.leak_isolated, ToyServed.rpow_bits_sub, avg_const, add_zero] at h
  have hlt : ENNReal.ofReal ((144 : ℝ) / 256) < 1 := by
    rw [ENNReal.ofReal_lt_one]
    norm_num
  exact absurd h (not_le.2 hlt)

end SecurityProofs

end WeightSecurity
