module

public import Definitions.WeightSecurity.Guarantees
public import Proofs.Core.Game.Basic
public import Proofs.Core.Game.Interleave
public import Proofs.Core.Game.Union
public import Proofs.SampledProofs.ExfiltrationBound
public import Proofs.WeightSecurity.Theft

/-! `WeightSecurity.SecurityProofs.TheftServed : WeightSecurity.Guarantees.TheftServed`, `Theft`'s leakage lemma at an
audited deployment. On a run whose releases are all local and whose audit doesn't escape at the budget `M`, the
outsider's view is its view against the datacenter that deviates through the served channel with the run's own covert
string (`Served.view_eq_resp`): each window's advice and bucket, the values its release changed, and the halt. Those
values number at most `E · M + (L + 1) · |O|` (`Served.sum_card_changed_le`): the windows accepted before the first
rejection drew none of their incorrect proof blocks, so their incorrect proof blocks count in `Served.lead`, and the
rejected window and the lag's windows change at most `|O|` values each. So the outsider steals only where the run
escapes, a release isn't local, or some honest-use party `Ou.withString ch s` steals, which `HonestUseHard` bounds by
`p` each.

`card_servedChannel_le` counts the served channel's covert strings,
`|Adv|^T · |Tick|^T · (Σ_{j ≤ K} C(T·|O|, j)) · |V|^K · (T + 1)`, the overwrites by the Hamming-ball count
`Population.Exfil.card_devBall_le`. The witnesses are proved at the toy deployment. -/

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

open Classical in
/-- The number of windows accepted before the first rejection. -/
noncomputable def accLen : List (Served O V Tick Adv) → ℕ
  | [] => 0
  | w :: l => if w.accepted then accLen l + 1 else 0

/-- Window `n`'s number of incorrect proof blocks, `0` past the run. -/
def wrongAt (l : List (Served O V Tick Adv)) (n : ℕ) : ℕ := (l[n]?.map Served.wrong).getD 0

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

/-- **The windows accepted before the first rejection count in `lead`** when no accepted window drew an incorrect proof
block: each of them drew none. -/
theorem sum_wrongAt_le_lead : ∀ l : List (Served O V Tick Adv), (¬ ∃ w ∈ l, w.accepted ∧ w.drawnWrong) →
    ∑ n ∈ range (accLen l), wrongAt l n ≤ lead l
  | [] => fun _ => by simp [accLen]
  | w :: l => fun h => by
    by_cases ha : w.accepted
    · have hd : ¬ w.drawnWrong := fun hd => h ⟨w, List.mem_cons_self, ha, hd⟩
      have ih := sum_wrongAt_le_lead l fun ⟨v, hv, hva⟩ => h ⟨v, List.mem_cons_of_mem _ hv, hva⟩
      have hs : ∀ n, wrongAt (w :: l) (n + 1) = wrongAt l n := fun n => by simp [wrongAt]
      have h0 : wrongAt (w :: l) 0 = w.wrong := by simp [wrongAt]
      simp only [accLen, lead, if_pos ha, if_neg hd]
      rw [sum_range_succ', h0]
      simp only [hs]
      omega
    · simp [accLen, ha]

variable [Fintype O]

/-- **Through the halt, the releases change at most `E · lead + (L + 1) · |O|` values** when every release is local and no
accepted window drew an incorrect proof block. -/
theorem sum_card_changed_le (C : Θ → Q × Adv → O → V) (E : ℕ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv)) (hloc : Local C E Ou L θ r l)
    (hacc : ¬ ∃ w ∈ l, w.accepted ∧ w.drawnWrong) :
    ∑ n ∈ range (count L l), (changed C Ou L θ r l n).card ≤ E * lead l + (L + 1) * Fintype.card O := by
  classical
  have hcl := count_le_length L l
  have hca := count_le_accLen L l
  rw [← sum_filter_add_sum_filter_not (range (count L l)) (· < accLen l)]
  apply add_le_add
  · calc ∑ n ∈ (range (count L l)).filter (· < accLen l), (changed C Ou L θ r l n).card
        ≤ ∑ n ∈ (range (count L l)).filter (· < accLen l), E * wrongAt l n := by
          refine sum_le_sum fun n hn => ?_
          have hn' : n < count L l := mem_range.1 (mem_filter.1 hn).1
          have hlen : n < l.length := lt_of_lt_of_le hn' hcl
          have hw : l[n]? = some l[n] := List.getElem?_eq_getElem hlen
          refine (hloc n hn' _ hw).trans (le_of_eq ?_)
          simp [wrongAt, hw]
      _ ≤ ∑ n ∈ range (accLen l), E * wrongAt l n :=
          sum_le_sum_of_subset_of_nonneg (fun n hn => mem_range.2 (mem_filter.1 hn).2) fun _ _ _ => Nat.zero_le _
      _ = E * ∑ n ∈ range (accLen l), wrongAt l n := (mul_sum _ _ _).symm
      _ ≤ E * lead l := Nat.mul_le_mul_left _ (sum_wrongAt_le_lead l hacc)
  · calc ∑ n ∈ (range (count L l)).filter (fun n => ¬ n < accLen l), (changed C Ou L θ r l n).card
        ≤ ∑ _n ∈ (range (count L l)).filter (fun n => ¬ n < accLen l), Fintype.card O :=
          sum_le_sum fun n _ => card_le_univ _
      _ = ((range (count L l)).filter (fun n => ¬ n < accLen l)).card * Fintype.card O := by
          rw [sum_const, smul_eq_mul]
      _ ≤ (L + 1) * Fintype.card O := by
          refine Nat.mul_le_mul_right _ ?_
          calc ((range (count L l)).filter (fun n => ¬ n < accLen l)).card ≤ (Ico (accLen l) (count L l)).card :=
                card_le_card fun n hn => by
                  simp only [mem_filter, mem_range, not_lt] at hn
                  exact mem_Ico.2 ⟨hn.2, hn.1⟩
            _ ≤ L + 1 := by rw [Nat.card_Ico]; omega

variable [DecidableEq O] [Fintype V] [DecidableEq V] [Fintype Adv] [Inhabited Adv] [Fintype Tick] [Inhabited Tick]

/-- The values a run's releases changed through the halt, by window and slot: the overwrite of its covert string. -/
noncomputable def overwrite (C : Θ → Q × Adv → O → V) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv)) (T : ℕ) : Fin T × O → Option V :=
  fun p => if p.1.val < count L l ∧ p.2 ∈ changed C Ou L θ r l p.1.val then l[p.1.val]?.map (·.egress p.2) else none

theorem card_overwrite_le (C : Θ → Q × Adv → O → V) (E M : ℕ)
    (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv))
    (T : ℕ) (hloc : Local C E Ou L θ r l) (hesc : ¬ Escapes M l) :
    (univ.filter fun p => overwrite C Ou L θ r l T p ≠ none).card ≤ E * M + (L + 1) * Fintype.card O := by
  classical
  have hacc : ¬ ∃ w ∈ l, w.accepted ∧ w.drawnWrong := fun h => hesc (Or.inl h)
  have hlead : lead l < M := lt_of_not_ge fun h => hesc (Or.inr h)
  calc (univ.filter fun p => overwrite C Ou L θ r l T p ≠ none).card
      ≤ ((range (count L l)).sigma fun n => changed C Ou L θ r l n).card := by
        refine card_le_card_of_injOn (fun p => ⟨p.1.val, p.2⟩) (fun p hp => ?_) (fun p _ p' _ h => ?_)
        · have hp' : overwrite C Ou L θ r l T p ≠ none := (mem_filter.1 hp).2
          have hc : p.1.val < count L l ∧ p.2 ∈ changed C Ou L θ r l p.1.val := by
            by_contra hc
            exact hp' (if_neg hc)
          exact mem_sigma.2 ⟨mem_range.2 hc.1, hc.2⟩
        · simp only [Sigma.mk.inj_iff, heq_eq_eq] at h
          exact Prod.ext (Fin.ext h.1) h.2
    _ = ∑ n ∈ range (count L l), (changed C Ou L θ r l n).card := card_sigma _ _
    _ ≤ E * lead l + (L + 1) * Fintype.card O := sum_card_changed_le C E Ou L θ r l hloc hacc
    _ ≤ E * M + (L + 1) * Fintype.card O := by gcongr; exact hlead.le

/-- **A local run that doesn't escape is the served channel's deviation with its own covert string**: each window's
advice and bucket, the run's overwrite and its halt. -/
theorem view_eq_resp (C : Θ → Q × Adv → O → V) (E M : ℕ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (T L : ℕ) (θ : Θ) (r : R) (l : List (Served O V Tick Adv)) (hloc : Local C E Ou L θ r l) (hesc : ¬ Escapes M l) :
    ∃ s : (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)).S,
      view Ou L r l T = Ou.view r ((servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)).resp C s θ r) T := by
  classical
  have hcl := count_le_length L l
  refine ⟨(fun t => (l[t.val]?.map (·.adv)).getD default, fun t => (l[t.val]?.map (·.tick)).getD default,
    ⟨overwrite C Ou L θ r l T, card_overwrite_le C E M Ou L θ r l T hloc hesc⟩,
    ⟨min (count L l) T, Nat.lt_succ_of_le (min_le_right _ _)⟩), ?_⟩
  refine Outsider.view_congr Ou r _ _ T fun n hn => ?_
  have hlen := Outsider.view_length Ou r (fun h _ => reply L l h.length) n
  simp only [Channel.resp, servedChannel]
  rw [hlen, reply]
  by_cases hnc : n < count L l
  · have hw : l[n]? = some l[n] := List.getElem?_eq_getElem (lt_of_lt_of_le hnc hcl)
    rw [if_pos hnc, dif_pos ⟨lt_min hnc hn, hn⟩, dif_pos hn]
    simp only [hw, Option.map, Option.getD]
    refine congrArg some (Prod.ext (funext fun o => ?_) rfl)
    by_cases ho : o ∈ changed C Ou L θ r l n
    · simp [overwrite, hnc, ho, hw]
    · have he : l[n].egress o = C θ ((Ou.query r (view Ou L r l n)).1, l[n].adv) o := by
        simpa [changed, hw] using ho
      simp only [overwrite, hnc, ho, and_false, ↓reduceIte, Option.getD_none]
      exact he
  · rw [if_neg hnc, dif_neg fun h => hnc (lt_of_lt_of_le h.1 (min_le_left _ _))]

end Served

namespace SecurityProofs

open Verity.Game

open Classical in
theorem avg_indicator {α : Type} [Fintype α] (P : α → Prop) :
    avg (fun z => if P z then (1 : ℝ≥0∞) else 0) = ENNReal.ofReal (Claims.Pr P) := by
  unfold avg Claims.Pr
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

theorem value_const_le {α : Type} (P : Prop) [Decidable P] (g : Game α) :
    value (fun _ => P) g ≤ if P then 1 else 0 := by
  by_cases hP : P
  · rw [if_pos hP]
    exact value_le_one _ _
  · rw [if_neg hP]
    exact (value_mono (q := fun _ => False) (fun _ h => hP h) g).trans_eq (value_false g)

theorem ite_exists_le_sum {ι : Type} [Fintype ι] (P : ι → Prop) [Decidable (∃ i, P i)] [DecidablePred P] :
    (if ∃ i, P i then (1 : ℝ≥0∞) else 0) ≤ ∑ i, if P i then 1 else 0 := by
  by_cases h : ∃ i, P i
  · obtain ⟨i, hi⟩ := h
    rw [if_pos ⟨i, hi⟩]
    calc (1 : ℝ≥0∞) = if P i then 1 else 0 := (if_pos hi).symm
      _ ≤ ∑ i, if P i then 1 else 0 :=
        single_le_sum (f := fun i => if P i then (1 : ℝ≥0∞) else 0) (fun _ _ => zero_le) (mem_univ i)
  · rw [if_neg h]
    exact zero_le

/-- **The leakage lemma over a game of coins.** Every party's run is a game `g z` at the weights' and outsider's coins
`z`. Wherever the outsider wins, either the bad event happens or some honest-use party `O.withString ch s` steals the
weights. Then the outsider wins with probability at most the bad event's plus `|ch.S| · p`, the union bound over the
covert strings with `HonestUseHard` for each. -/
theorem value_leak {Ω R Θ X Y M Θ' α : Type} [Fintype Ω] [Fintype R] (C : Θ → X → Y) (gen : Ω → Θ)
    (Stolen : Θ → Θ' → Prop) (q : ℕ) (A : Set (Outsider R X Y Θ')) (p : ℝ) (O : Outsider R X M Θ')
    (ch : Channel R X Y M) (hhon : HonestUseHard C gen Stolen q A p) (hsim : Simulable A O ch)
    (g : Ω × R → Game α) (Win Bad : Ω × R → α → Prop)
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
      _ ≤ _ := add_le_add_left ((value_const_le _ _).trans (ite_exists_le_sum _)) _
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
    [DecidableEq V] [Nonempty V] [Fintype Tick] (T K : ℕ) :
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
        rw [dif_neg this, hp]
      · have hm : p ∈ (img d).1 := by simp [img, hp]
        rw [dif_pos hm]
        obtain ⟨v, hv⟩ := Option.ne_none_iff_exists'.1 hp
        simp [img, hv]
    rw [← card_univ]
    refine card_le_card_of_injOn img (fun d _ => ?_) (fun d _ d' _ h => ?_)
    · simp only [Population.Exfil.devBall, mem_sigma, mem_filter, mem_univ, true_and, and_true]
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

theorem TheftServed : Guarantees.TheftServed := by
  intro Ω R Θ Q Adv O V Tick Θ' _ _ _ _ _ _ _ _ _ _ C gen Stolen A p Ou T L E M δ run hhon hsim hesc
  classical
  have hcov : ∀ (z : Ω × R) (l : List (Served O V Tick Adv)),
      Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T)) →
        (¬ Served.Local C E Ou L (gen z.1) z.2 l ∨ Served.Escapes M l) ∨
          ∃ s : (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)).S, Stolen (gen z.1)
            ((Ou.withString (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)) s).model z.2
              (honest C (gen z.1)) T) := by
    intro z l hst
    by_cases hloc : Served.Local C E Ou L (gen z.1) z.2 l
    · by_cases he : Served.Escapes M l
      · exact Or.inl (Or.inr he)
      · obtain ⟨s, hs⟩ := Served.view_eq_resp C E M Ou T L (gen z.1) z.2 l hloc he
        refine Or.inr ⟨s, ?_⟩
        rw [withString_model, Outsider.model, ← hs]
        exact hst
    · exact Or.inl (Or.inl hloc)
  refine (value_leak C gen Stolen T A p Ou _ hhon hsim (fun z => run (gen z.1) z.2)
    (fun z l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T)))
    (fun z l => ¬ Served.Local C E Ou L (gen z.1) z.2 l ∨ Served.Escapes M l) hcov).trans ?_
  calc avg (fun z : Ω × R => value (fun l => ¬ Served.Local C E Ou L (gen z.1) z.2 l ∨ Served.Escapes M l)
          (run (gen z.1) z.2)) + ENNReal.ofReal
          (Fintype.card (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)).S * p)
      ≤ avg (fun z : Ω × R => value (fun l => ¬ Served.Local C E Ou L (gen z.1) z.2 l) (run (gen z.1) z.2) + δ) +
          ENNReal.ofReal (Fintype.card (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)).S * p) := by
        gcongr with z
        exact (value_or_le _ _ _).trans (add_le_add_left (hesc _ _) _)
    _ ≤ (avg (fun z : Ω × R => value (fun l => ¬ Served.Local C E Ou L (gen z.1) z.2 l) (run (gen z.1) z.2)) + δ) +
          ENNReal.ofReal (Fintype.card (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)).S * p) := by
        rw [avg_add]
        gcongr
        exact avg_le_const δ
    _ = _ := by ring

namespace ToyServed

open WeightSecurity.ToyServed

theorem card_W : Fintype.card (W × Unit) = 256 := by simp [W]

/-- Honest use of the toy's served function reveals nothing: every party's output is one fixed guess. -/
theorem guess_le (P : Outsider Unit (Unit × Unit) (Unit → Fin 16) W) :
    Claims.Pr (fun z : W × Unit => Stolen (gen z.1) (P.model z.2 (honest C (gen z.1)) 2)) ≤ 1 / 256 := by
  classical
  unfold Claims.Pr
  rw [card_W]
  have h1 : (univ.filter fun z : W × Unit => Stolen (gen z.1) (P.model z.2 (honest C (gen z.1)) 2)).card ≤ 1 := by
    refine card_le_one.2 fun a ha b hb => ?_
    simp only [mem_filter, mem_univ, true_and, Stolen, gen, id] at ha hb
    have hab : a.1 = b.1 := ha.symm.trans hb
    exact Prod.ext hab (Subsingleton.elim _ _)
  have : ((univ.filter fun z : W × Unit => Stolen (gen z.1) (P.model z.2 (honest C (gen z.1)) 2)).card : ℝ) ≤ 1 := by
    exact_mod_cast h1
  rw [Nat.cast_ofNat]
  exact div_le_div_of_nonneg_right this (by norm_num)

end ToyServed

theorem TheftServedHonest : Guarantees.TheftServedHonest := by
  classical
  refine ⟨fun P _ => ToyServed.guess_le P, fun _ => Set.mem_univ _, fun θ r => le_of_eq ?_⟩
  unfold WeightSecurity.ToyServed.honestRun
  rw [value_ret, if_neg]
  rintro (⟨w, hw, -, hd⟩ | h)
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at hw
    rcases hw with rfl | rfl <;> exact hd
  · simp [Served.lead] at h

theorem TheftServedNontrivial : Guarantees.TheftServedNontrivial := by
  classical
  intro h
  have hsteal : ∀ z : WeightSecurity.ToyServed.W × Unit, value (fun l => WeightSecurity.ToyServed.Stolen
      (WeightSecurity.ToyServed.gen z.1) (WeightSecurity.ToyServed.thief.out z.2
        (Served.view WeightSecurity.ToyServed.thief 0 z.2 l 2)))
      (WeightSecurity.ToyServed.leakRun (WeightSecurity.ToyServed.gen z.1) z.2) = 1 := by
    intro z
    unfold WeightSecurity.ToyServed.leakRun
    rw [value_ret, if_pos]
    simp [WeightSecurity.ToyServed.Stolen, WeightSecurity.ToyServed.gen, WeightSecurity.ToyServed.thief, Served.view,
      Served.reply, Served.count, Outsider.view]
  have hloc : ∀ z : WeightSecurity.ToyServed.W × Unit, value (fun l => ¬ Served.Local WeightSecurity.ToyServed.C 1
      WeightSecurity.ToyServed.thief 0 (WeightSecurity.ToyServed.gen z.1) z.2 l)
      (WeightSecurity.ToyServed.leakRun (WeightSecurity.ToyServed.gen z.1) z.2) = 0 := by
    intro z
    unfold WeightSecurity.ToyServed.leakRun
    rw [value_ret, if_neg (not_not.2 ?_)]
    intro n _ w hw
    have hwr : w.wrong = 1 := by
      rcases n with _ | _ | n <;> simp_all
    rw [hwr, mul_one]
    exact (card_le_univ _).trans_eq Fintype.card_unit
  simp only [hsteal, hloc, avg_const, add_zero] at h
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

end SecurityProofs

end WeightSecurity
