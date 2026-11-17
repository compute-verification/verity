module

public import Definitions.WeightSecurity.Cut
public import Proofs.WeightSecurity.Served

/-! `theft_cut`: `Theft` at an honest cut's channel. -/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Finset WeightSecurity WeightSecurity.Assumptions

variable {Ω R X Y Y' Θ Θ' F V : Type}

theorem view_eq_cut (κ : Cut X Y) {C : Θ → X → Y} {Cf : Θ → X × F → Y} {size : F → ℕ} {inClass : F → Prop}
    (hcut : κ.Cuts C Cf size inClass) (real : Θ → R → List (X × Y) → X → Y) (O : Outsider R X Y Θ') (q M : ℕ)
    (θ : Θ) (r : R) (hw : FaultsWithin real Cf size inClass O q M θ r) :
    ∃ s, O.view r (real θ r) q = O.view r ((κ.channel R q M).resp C s θ r) q := by
  classical
  obtain ⟨φ, hφ, hM⟩ := hw
  have hh : ∀ t : Fin q, ∃ h : κ.H, κ.size h ≤ size (φ t) ∧
      Cf θ (O.query r (O.view r (real θ r) t), φ t) =
        κ.sim (O.query r (O.view r (real θ r) t)) h (C θ (κ.ask (O.query r (O.view r (real θ r) t)) h)) :=
    fun t => hcut θ _ (φ t) (hφ t t.2).1
  choose s hs using hh
  have hsM : ∑ t, κ.size (s t) ≤ M :=
    calc ∑ t, κ.size (s t) ≤ ∑ t : Fin q, size (φ t) := sum_le_sum fun t _ => (hs t).1
      _ = ∑ n ∈ range q, size (φ n) := Fin.sum_univ_eq_sum_range (fun n => size (φ n)) q
      _ ≤ M := hM
  refine ⟨⟨s, hsM⟩, ?_⟩
  suffices key : ∀ n ≤ q, O.view r (real θ r) n = O.view r ((κ.channel R q M).resp C ⟨s, hsM⟩ θ r) n from
    key q le_rfl
  intro n
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
    intro hn
    have hv := ih (Nat.le_of_succ_le hn)
    have hlen : (O.view r (real θ r) n).length < q := by rw [Outsider.view_length]; omega
    simp only [Outsider.view]
    rw [← hv]
    refine congrArg₂ List.cons (Prod.ext rfl ?_) rfl
    simp only [Channel.resp, Cut.channel, hlen, ↓reduceDIte]
    rw [(hφ n (by omega)).2]
    have := (hs ⟨n, by omega⟩).2
    simp only [Outsider.view_length] at this ⊢
    exact this

theorem leaksThrough_cut [Fintype Ω] [Fintype R] (κ : Cut X Y) {C : Θ → X → Y} {Cf : Θ → X × F → Y}
    {size : F → ℕ} {inClass : F → Prop} (hcut : κ.Cuts C Cf size inClass) (gen : Ω → Θ)
    (real : Θ → R → List (X × Y) → X → Y) (O : Outsider R X Y Θ') (q M : ℕ) :
    LeaksThrough C gen real O q (κ.channel R q M)
      (Claims.Pr fun z : Ω × R => ¬ FaultsWithin real Cf size inClass O q M (gen z.1) z.2) :=
  Pouw.pr_mono fun z hno hw => hno (view_eq_cut κ hcut real O q M (gen z.1) z.2 hw)

theorem theft_cut [Fintype Ω] [Fintype R] (κ : Cut X Y) {C : Θ → X → Y} {Cf : Θ → X × F → Y}
    {size : F → ℕ} {inClass : F → Prop} (hcut : κ.Cuts C Cf size inClass) (gen : Ω → Θ)
    (Stolen : Θ → Θ' → Prop) (q : ℕ) (A : Set (Outsider R X Y Θ')) (p : ℝ)
    (real : Θ → R → List (X × Y) → X → Y) (O : Outsider R X Y Θ') (M : ℕ)
    (hhon : HonestUseHard C gen Stolen q A p) (hsim : Simulable A O (κ.channel R q M)) :
    Claims.Pr (fun z : Ω × R => Stolen (gen z.1) (O.model z.2 (real (gen z.1) z.2) q)) ≤
      Fintype.card (κ.channel R q M).S * p +
        Claims.Pr (fun z : Ω × R => ¬ FaultsWithin real Cf size inClass O q M (gen z.1) z.2) :=
  SecurityProofs.Theft C gen Stolen q A p real O (κ.channel R q M) _ hhon
    (leaksThrough_cut κ hcut gen real O q M) hsim

theorem dropFaults_query (P : Outsider R (X × F) Y Θ') (r : R) (h : List (X × Y)) :
    P.dropFaults.query r h = (P.query r (P.relift r h)).1 := rfl

theorem relift_view (P : Outsider R (X × F) Y Θ') (Cf : Θ → X × F → Y) (θ : Θ) (r : R) :
    ∀ n, P.relift r (P.dropFaults.view r (faultingAs P Cf θ r) n) = P.view r (honest Cf θ) n
  | 0 => rfl
  | n + 1 => by
    simp only [Outsider.view, Outsider.relift, faultingAs, honest, dropFaults_query, relift_view P Cf θ r n]

theorem dropFaults_model (P : Outsider R (X × F) Y Θ') (Cf : Θ → X × F → Y) (θ : Θ) (r : R) (q : ℕ) :
    P.dropFaults.model r (faultingAs P Cf θ r) q = P.model r (honest Cf θ) q :=
  congrArg (P.out r) (relift_view P Cf θ r q)

theorem mem_view (O : Outsider R X Y Θ') (r : R) (resp : List (X × Y) → X → Y) :
    ∀ q n, n < q → (O.query r (O.view r resp n), resp (O.view r resp n) (O.query r (O.view r resp n))) ∈
      O.view r resp q
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | q + 1, n, h => by
    rcases Nat.lt_succ_iff_lt_or_eq.1 h with h | rfl
    · exact List.mem_cons_of_mem _ (mem_view O r resp q n h)
    · exact List.mem_cons_self

theorem sum_view (O : Outsider R X Y Θ') (r : R) (resp : List (X × Y) → X → Y) (f : X → ℕ) :
    ∀ q, ((O.view r resp q).map fun e => f e.1).sum = ∑ n ∈ range q, f (O.query r (O.view r resp n))
  | 0 => rfl
  | q + 1 => by
    rw [sum_range_succ, ← sum_view O r resp f q]
    simp only [Outsider.view, List.map_cons, List.sum_cons]
    omega

theorem faultedUseHard_of_cut [Fintype Ω] [Fintype R] (κ : Cut X Y) {C : Θ → X → Y} {Cf : Θ → X × F → Y}
    {size : F → ℕ} {inClass : F → Prop} (hcut : κ.Cuts C Cf size inClass) (gen : Ω → Θ)
    (Stolen : Θ → Θ' → Prop) (q M : ℕ) (A : Set (Outsider R X Y Θ')) (Af : Set (Outsider R (X × F) Y Θ')) (p : ℝ)
    (hA : ∀ P ∈ Af, Simulable A P.dropFaults (κ.channel R q M)) :
    HonestUseHard C gen Stolen q A p →
      HonestUseHard Cf gen Stolen q (Af ∩ PatternsIn inClass q ∩ FaultBudget size q M)
        (Fintype.card (κ.channel R q M).S * p) := by
  intro hhon P ⟨⟨hPA, hcl⟩, hsz⟩
  have hw : ∀ z : Ω × R, FaultsWithin (faultingAs P Cf) Cf size inClass P.dropFaults q M (gen z.1) z.2 := by
    intro z
    refine ⟨fun n => (P.query z.2 (P.view z.2 (honest Cf (gen z.1)) n)).2, fun n hn => ⟨?_, ?_⟩, ?_⟩
    · exact hcl z.2 (honest Cf (gen z.1)) _ (mem_view P z.2 _ q n hn)
    · simp only [faultingAs, dropFaults_query, relift_view]
    · have := sum_view P z.2 (honest Cf (gen z.1)) (fun x => size x.2) q
      rw [← this]
      exact hsz z.2 (honest Cf (gen z.1))
  calc Claims.Pr (fun z : Ω × R => Stolen (gen z.1) (P.model z.2 (honest Cf (gen z.1)) q))
      = Claims.Pr (fun z : Ω × R =>
          Stolen (gen z.1) (P.dropFaults.model z.2 (faultingAs P Cf (gen z.1) z.2) q)) := by
        simp only [dropFaults_model]
    _ ≤ Fintype.card (κ.channel R q M).S * p + Claims.Pr (fun z : Ω × R =>
          ¬ FaultsWithin (faultingAs P Cf) Cf size inClass P.dropFaults q M (gen z.1) z.2) :=
        theft_cut κ hcut gen Stolen q A p (faultingAs P Cf) P.dropFaults M hhon (hA P hPA)
    _ = Fintype.card (κ.channel R q M).S * p := by
        have h0 : Claims.Pr (fun z : Ω × R =>
            ¬ FaultsWithin (faultingAs P Cf) Cf size inClass P.dropFaults q M (gen z.1) z.2) = 0 :=
          Pouw.pr_eq_zero fun z h => h (hw z)
        rw [h0, add_zero]

/-- So `TheftFaulted` charges such a pattern `κ`'s hint and takes no `FaultedUseHard` there. -/
theorem not_interior_of_cuts (κ : Cut X Y) {Cf : Θ → X × F → Y} {size : F → ℕ} {inClass : F → Prop} {φ₀ φ : F}
    (hcut : κ.Cuts (fun θ x => Cf θ (x, φ₀)) Cf size inClass) (hφ : inClass φ) (hne : φ ≠ φ₀) :
    ¬ κ.Interior Cf size φ₀ φ := by
  rintro (h | h)
  · exact hne h
  · refine h fun θ x φ' hφ' => ?_
    subst hφ'
    exact hcut θ x _ hφ

theorem cuts_map (κ : Cut X Y) {C : Θ → X → Y} {Cf : Θ → X × F → Y} {size : F → ℕ} {inClass : F → Prop}
    (f : Y → Y') (g : Y' → Y) (hgf : ∀ θ x, g (f (C θ x)) = C θ x) (hcut : κ.Cuts C Cf size inClass) :
    (κ.map f g).Cuts (fun θ x => f (C θ x)) (fun θ x => f (Cf θ x)) size inClass := by
  intro θ x φ hφ
  obtain ⟨h, hs, he⟩ := hcut θ x φ hφ
  refine ⟨h, hs, ?_⟩
  show f (Cf θ (x, φ)) = f (κ.sim x h (g (f (C θ (κ.ask x h)))))
  rw [hgf, he]

theorem decode_congr {a b : ℕ → List V → V} {N : ℕ} (h : ∀ n < N, a n (decode a n) = b n (decode a n)) :
    ∀ m ≤ N, decode a m = decode b m := by
  intro m
  induction m with
  | zero => intro _; rfl
  | succ m ih =>
    intro hm
    simp only [decode]
    rw [← ih (Nat.le_of_succ_le hm), h m hm]

theorem cuts_forced [Fintype V] {C : Θ → X → List V} {Cf : Θ → X × F → List V} {size : F → ℕ}
    {inClass : F → Prop} {step : Θ → X → List V → V} {N : ℕ} {force : X → (Fin N → Option V) → X}
    (hC : ForcedServing C step N force) (P : Θ → X → F → Finset ℕ)
    (hP : ∀ θ x φ, inClass φ → (P θ x φ).card ≤ size φ)
    (hdec : ∀ θ x φ, inClass φ → ∃ next : ℕ → List V → V,
      Cf θ (x, φ) = decode next N ∧ ∀ n < N, n ∉ P θ x φ → next n = step θ x) :
    (forcedCut X V N force).Cuts C Cf size inClass := by
  classical
  intro θ x φ hφ
  obtain ⟨next, hCf, hagree⟩ := hdec θ x φ hφ
  let d : Fin N → Option V := fun i => if (i : ℕ) ∈ P θ x φ then some (next i (decode next i)) else none
  refine ⟨d, ?_, ?_⟩
  · show (univ.filter fun i => (d i).isSome).card ≤ size φ
    refine le_trans (Finset.card_le_card_of_injOn (fun i : Fin N => (i : ℕ)) ?_ ?_) (hP θ x φ hφ)
    · intro i hi
      have hi' : (d i).isSome := (Finset.mem_filter.1 hi).2
      show (i : ℕ) ∈ P θ x φ
      by_contra hn
      simp [d, hn] at hi'
    · intro i _ j _ h
      exact Fin.ext h
  · show Cf θ (x, φ) = C θ (force x d)
    rw [hC, hCf]
    refine decode_congr (fun n hn => ?_) N le_rfl
    simp only [forcing, hn, ↓reduceDIte, d]
    by_cases hnP : n ∈ P θ x φ
    · simp [hnP]
    · simp [hnP, hagree n hn hnP]

theorem cuts_tokenGate [Fintype V] {C : Θ → X → List V} {Cf : Θ → X × F → List V} {size : F → ℕ}
    {inClass : F → Prop} {step : Θ → X → List V → V} {N : ℕ} {force : X → (Fin N → Option V) → X}
    (hC : ForcedServing C step N force) (pos : F → Finset ℕ) (hpos : ∀ φ, inClass φ → (pos φ).card ≤ size φ)
    (hloc : ∀ θ x φ, inClass φ → ∃ next : ℕ → List V → V,
      Cf θ (x, φ) = decode next N ∧ ∀ n < N, n ∉ pos φ → next n = step θ x) :
    (forcedCut X V N force).Cuts C Cf size inClass :=
  cuts_forced hC (fun _ _ φ => pos φ) (fun _ _ φ hφ => hpos φ hφ) hloc

theorem cuts_oneToken [Fintype V] {C : Θ → X → List V} {Cf : Θ → X × F → List V} {size : F → ℕ}
    {inClass : F → Prop} {step : Θ → X → List V → V} {force : X → (Fin 1 → Option V) → X}
    (hC : ForcedServing C step 1 force)
    (hone : ∀ θ x φ, inClass φ → (Cf θ (x, φ)).length = 1)
    (hempty : ∀ θ x φ, inClass φ → size φ = 0 → Cf θ (x, φ) = decode (fun _ => step θ x) 1) :
    (forcedCut X V 1 force).Cuts C Cf size inClass := by
  classical
  refine cuts_forced hC (fun _ _ φ => if size φ = 0 then ∅ else {0}) (fun _ _ φ _ => ?_) (fun θ x φ hφ => ?_)
  · split_ifs with h <;> simp; omega
  · by_cases h0 : size φ = 0
    · exact ⟨fun _ => step θ x, hempty θ x φ hφ h0, fun _ _ _ => rfl⟩
    · obtain ⟨v, hv⟩ := List.length_eq_one_iff.1 (hone θ x φ hφ)
      refine ⟨fun _ _ => v, by simp [hv, decode], fun n hn hnP => ?_⟩
      simp [h0] at hnP
      omega

/-- For a wide block, whose output is at least the `N` tokens' bits, the hint is no longer than the fault. -/
theorem cuts_release [Fintype V] {C : Θ → X → List V} {Cf : Θ → X × F → List V} {size : F → ℕ}
    {inClass : F → Prop} {N : ℕ}
    (hlen : ∀ θ x φ, inClass φ → (Cf θ (x, φ)).length = N)
    (hempty : ∀ θ x φ, inClass φ → size φ = 0 → Cf θ (x, φ) = C θ x) :
    (releaseCut X V N).Cuts C Cf size inClass := by
  intro θ x φ hφ
  by_cases h0 : size φ = 0
  · exact ⟨none, Nat.zero_le _, by simp [releaseCut, hempty θ x φ hφ h0]⟩
  · exact ⟨some ⟨_, hlen θ x φ hφ⟩, show 1 ≤ size φ by omega, rfl⟩

theorem card_forcedChannel_le [Fintype V] [Nonempty V] (N : ℕ) (force : X → (Fin N → Option V) → X) (R : Type)
    (q M : ℕ) : Fintype.card ((forcedCut X V N force).channel R q M).S ≤ (Fintype.card V * (q * N + 1)) ^ M := by
  classical
  have hinj : Fintype.card ((forcedCut X V N force).channel R q M).S ≤
      Fintype.card {d : Fin q × Fin N → Option V // (univ.filter fun p => d p ≠ none).card ≤ M} := by
    refine Fintype.card_le_of_injective (fun s => ⟨fun p => s.1 p.1 p.2, ?_⟩) fun s s' h => ?_
    · refine le_trans (le_of_eq ?_) s.2
      rw [card_filter, Fintype.sum_prod_type]
      refine sum_congr rfl fun t _ => ?_
      show _ = (univ.filter fun i => (s.1 t i).isSome).card
      rw [card_filter]
      refine sum_congr rfl fun i _ => ?_
      dsimp only
      by_cases h : s.1 t i = none <;> simp [h]
    · exact Subtype.ext (funext fun t => funext fun i => congrArg (fun d => d.1 (t, i)) h)
  calc Fintype.card ((forcedCut X V N force).channel R q M).S
      ≤ Fintype.card {d : Fin q × Fin N → Option V // (univ.filter fun p => d p ≠ none).card ≤ M} := hinj
    _ ≤ (∑ j ∈ range (M + 1), (q * N).choose j) * Fintype.card V ^ M := by
        have := card_overwrites_le (Fin q × Fin N) V M
        rwa [Fintype.card_prod, Fintype.card_fin, Fintype.card_fin] at this
    _ ≤ (q * N + 1) ^ M * Fintype.card V ^ M := Nat.mul_le_mul_right _ (sum_choose_le_pow _ _)
    _ = (Fintype.card V * (q * N + 1)) ^ M := by rw [mul_pow, Nat.mul_comm]

theorem card_releaseChannel_le [Fintype V] [Nonempty V] (N : ℕ) (R : Type) (q M : ℕ) :
    Fintype.card ((releaseCut X V N).channel R q M).S ≤ (Fintype.card V ^ N * (q + 1)) ^ M := by
  classical
  have : Nonempty (List.Vector V N) := ⟨List.Vector.replicate N (Classical.arbitrary V)⟩
  have hinj : Fintype.card ((releaseCut X V N).channel R q M).S ≤
      Fintype.card {d : Fin q → Option (List.Vector V N) // (univ.filter fun t => d t ≠ none).card ≤ M} := by
    refine Fintype.card_le_of_injective (fun s => ⟨s.1, ?_⟩) fun s s' h => ?_
    · refine le_trans (le_of_eq ?_) s.2
      rw [card_filter]
      refine sum_congr rfl fun t _ => ?_
      cases s.1 t <;> simp [releaseCut]
    · have h' := congrArg Subtype.val h
      exact Subtype.ext h'
  calc Fintype.card ((releaseCut X V N).channel R q M).S
      ≤ Fintype.card {d : Fin q → Option (List.Vector V N) // (univ.filter fun t => d t ≠ none).card ≤ M} := hinj
    _ ≤ (∑ j ∈ range (M + 1), q.choose j) * (Fintype.card V ^ N) ^ M := by
        have := card_overwrites_le (Fin q) (List.Vector V N) M
        rwa [Fintype.card_fin, card_vector] at this
    _ ≤ (q + 1) ^ M * (Fintype.card V ^ N) ^ M := Nat.mul_le_mul_right _ (sum_choose_le_pow _ _)
    _ = (Fintype.card V ^ N * (q + 1)) ^ M := by rw [mul_pow, Nat.mul_comm]

end WeightSecurity.SecurityProofs
