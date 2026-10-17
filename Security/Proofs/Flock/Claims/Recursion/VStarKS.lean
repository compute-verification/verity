import Proofs.Flock.Claims.Recursion.InnerAcceptsKS
import Proofs.Flock.Claims.Recursion.VStar

/-!
# The recursion's KS from V*'s parts

`KS_of_flockKS_coll` (`InnerAcceptsKS.lean`) reads two code facts that V*'s parts don't give at the deployed readers
(`VStar.lean`): `CallsOpen`, an opening at every inner round, also those an inner view that rejects early doesn't play,
and `CallsRecord`'s first half, every call of the inner view's run recorded, also where two parts hold different
messages. It reads them at four places only, and V*'s parts give what each place needs:

* `Pr_bad_le`, `Pr_bad_le_fork`: a fork's differing openings are a collision `Be` finds wherever the played rounds'
  openings are recorded (`forkColl_of_openingPlayed`, `sum_opensDiffer_le_played`, from `CallsOpenPlayed`).
* `Pr_innerRejects_le_fork`: `Be` reads only true evaluations, which needs only `CallsRecord`'s second half
  (`calls_true_of`, from `CallsTrue`).
* `PrBAbort_le_PrBI`: `B` at `P_I`'s aborting runs finds collisions `B_I` finds, where the inner verifier's calls are
  recorded off a collision (`PrBAbort_le_PrBI_coll`, from `CallsLogColl`).

This file is that chain with those three facts in place of `CallsOpen` and `CallsRecord`, each proof as its original's
with the one call changed (`Pr_bad_le_played`, `Pr_bad_le_fork_played`, `Pr_innerRejects_le_fork_true`,
`bad_rej_le_played`, `innerKS_of_KSAbort_played`, `KS_of_flockKS_played`). `KS_of_parts` is the recursion's KS from
Flock's KS at the outer deployment, the inner protocol's KS against provers that may abort, the session facts, and
V*'s parts at the deployed readers (`codeFacts_of_staging`).
-/

namespace Claims.Recursion.VStar

section Coupling

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **A draw is bad at most as `√(k_I·k(x)·Pr[Be])`**: `Pr_bad_le`, with the played rounds' openings
(`sum_opensDiffer_le_played`). -/
theorem Pr_bad_le_played {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} (hV : InnerViewRun H D I DO) (hO : CallsOpenPlayed H D I DO opening)
    (hT : CallsTrue H D I DO) (hR : InnerRounds D I) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      Real.sqrt (I.V.k (xI D x) * D.V.k x * PrBe H D I DO decO P x) := by
  have h1 : Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      ∑ j : Fin (I.V.k (xI D x)), Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
        (eoOut H D I DO decO P x t.1 t.2).isSome = true ∧
          eoMsg H D I DO decO P x t.1 t.2 j ≠ opAt H D I DO decO P x t.1 j ((List.range j).map (coinSeq t.2.1))) := by
    refine (Pr_mono fun t ht => ?_).trans (Pr_exists_le _)
    obtain ⟨hs, hna⟩ := ht
    simp only [Agrees, not_forall] at hna
    obtain ⟨j, hj, hne⟩ := hna
    exact ⟨⟨j, hj⟩, hs, hne⟩
  have h2 : ∀ j : Fin (I.V.k (xI D x)), Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (eoOut H D I DO decO P x t.1 t.2).isSome = true ∧
        eoMsg H D I DO decO P x t.1 t.2 j ≠ opAt H D I DO decO P x t.1 j ((List.range j).map (coinSeq t.2.1))) ≤
      Real.sqrt (Pr (OpensDiffer H D I DO decO P x j)) := fun j => by
    have hp := Pr_plur_fork j (fun ω d => (eoOut H D I DO decO P x ω d).isSome = true)
      (fun ω d => eoMsg H D I DO decO P x ω d j)
    refine (le_of_eq (Pr_congr fun t => ?_)).trans (hp.trans (Real.sqrt_le_sqrt (Pr_mono fun t ht => ?_)))
    · simp only [opAt, fork_preCoins]
    · obtain ⟨hg₁, hg₂, hf⟩ := ht
      obtain ⟨ws, hws⟩ := Option.isSome_iff_exists.1 hg₁
      obtain ⟨ws', hws'⟩ := Option.isSome_iff_exists.1 hg₂
      refine ⟨ws, ws', eoGives_of_some hws, eoGives_of_some hws', ?_⟩
      simpa [eoMsg, hws, hws', dec] using hf
  have hcs := Real.sum_sqrt_mul_sqrt_le (Finset.univ : Finset (Fin (I.V.k (xI D x))))
    (f := fun _ : Fin (I.V.k (xI D x)) => (1 : ℝ)) (g := fun j => Pr (OpensDiffer H D I DO decO P x j))
    (fun _ => zero_le_one) (fun j => Pr_nonneg _)
  simp only [Real.sqrt_one, one_mul, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    mul_one] at hcs
  have h4 := sum_opensDiffer_le_played hV hO hT hR decO x P
  calc Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2)
      ≤ ∑ j : Fin (I.V.k (xI D x)), Real.sqrt (Pr (OpensDiffer H D I DO decO P x j)) :=
        h1.trans (Finset.sum_le_sum fun j _ => h2 j)
    _ ≤ Real.sqrt (I.V.k (xI D x)) * Real.sqrt (∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j)) :=
        hcs
    _ = Real.sqrt (I.V.k (xI D x) * ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j)) :=
        (Real.sqrt_mul (Nat.cast_nonneg _) _).symm
    _ ≤ Real.sqrt (I.V.k (xI D x) * (D.V.k x * PrBe H D I DO decO P x)) :=
        Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left h4 (Nat.cast_nonneg _))
    _ = Real.sqrt (I.V.k (xI D x) * D.V.k x * PrBe H D I DO decO P x) := by rw [mul_assoc]

/-- **A draw is bad at most as `√(k_I·Σ_{j<k_I} Pr[forkColl j])`**: `Pr_bad_le_fork`, with the played rounds'
openings (`forkColl_of_openingPlayed`). -/
theorem Pr_bad_le_fork_played {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} (hV : InnerViewRun H D I DO) (hO : CallsOpenPlayed H D I DO opening)
    (hT : CallsTrue H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      Real.sqrt (I.V.k (xI D x) * ∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j)) := by
  have h1 : Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      ∑ j : Fin (I.V.k (xI D x)), Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
        (eoOut H D I DO decO P x t.1 t.2).isSome = true ∧
          eoMsg H D I DO decO P x t.1 t.2 j ≠ opAt H D I DO decO P x t.1 j ((List.range j).map (coinSeq t.2.1))) := by
    refine (Pr_mono fun t ht => ?_).trans (Pr_exists_le _)
    obtain ⟨hs, hna⟩ := ht
    simp only [Agrees, not_forall] at hna
    obtain ⟨j, hj, hne⟩ := hna
    exact ⟨⟨j, hj⟩, hs, hne⟩
  have h2 : ∀ j : Fin (I.V.k (xI D x)), Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (eoOut H D I DO decO P x t.1 t.2).isSome = true ∧
        eoMsg H D I DO decO P x t.1 t.2 j ≠ opAt H D I DO decO P x t.1 j ((List.range j).map (coinSeq t.2.1))) ≤
      Real.sqrt (Pr (OpensDiffer H D I DO decO P x j)) := fun j => by
    have hp := Pr_plur_fork j (fun ω d => (eoOut H D I DO decO P x ω d).isSome = true)
      (fun ω d => eoMsg H D I DO decO P x ω d j)
    refine (le_of_eq (Pr_congr fun t => ?_)).trans (hp.trans (Real.sqrt_le_sqrt (Pr_mono fun t ht => ?_)))
    · simp only [opAt, fork_preCoins]
    · obtain ⟨hg₁, hg₂, hf⟩ := ht
      obtain ⟨ws, hws⟩ := Option.isSome_iff_exists.1 hg₁
      obtain ⟨ws', hws'⟩ := Option.isSome_iff_exists.1 hg₂
      refine ⟨ws, ws', eoGives_of_some hws, eoGives_of_some hws', ?_⟩
      simpa [eoMsg, hws, hws', dec] using hf
  have hcs := Real.sum_sqrt_mul_sqrt_le (Finset.univ : Finset (Fin (I.V.k (xI D x))))
    (f := fun _ : Fin (I.V.k (xI D x)) => (1 : ℝ)) (g := fun j => Pr (OpensDiffer H D I DO decO P x j))
    (fun _ => zero_le_one) (fun j => Pr_nonneg _)
  simp only [Real.sqrt_one, one_mul, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    mul_one] at hcs
  calc Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2)
      ≤ ∑ j : Fin (I.V.k (xI D x)), Real.sqrt (Pr (OpensDiffer H D I DO decO P x j)) :=
        h1.trans (Finset.sum_le_sum fun j _ => h2 j)
    _ ≤ Real.sqrt (I.V.k (xI D x)) * Real.sqrt (∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j)) :=
        hcs
    _ = Real.sqrt (I.V.k (xI D x) * ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j)) :=
        (Real.sqrt_mul (Nat.cast_nonneg _) _).symm
    _ ≤ Real.sqrt (I.V.k (xI D x) * ∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j)) :=
        Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
          (Finset.sum_le_sum fun j _ => Pr_mono fun t => forkColl_of_openingPlayed hV hO hT decO x P t)
          (Nat.cast_nonneg _))

/-- **A rejected inner view is a collision at every fork**: `Pr_innerRejects_le_fork`, where `Be` reads only true
evaluations by `CallsTrue` and `CallsNil` (`calls_true_of`). -/
theorem Pr_innerRejects_le_fork_true {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} (hA : InnerAcceptsColl H D I DO) (hT : CallsTrue H D I DO) (hN0 : CallsNil H D)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) (j : ℕ) :
    Pr (InnerRejects H D I DO decO P x) ≤ Pr (forkColl H D I DO decO P x j) := by
  rw [← (unifMap_judged (Ω := Ω) (R := Fin (D.V.k x) → Coin) (C := (EO D I DO decO).C x)).Pr]
  refine Pr_mono fun t ht => ?_
  obtain ⟨ws, hg, hrej⟩ := ht
  rcases hA decO x Ω P t.1 t.2.1 t.2.2.2.1 ws hg with hacc | ⟨u, v, huv, -⟩
  · exact absurd hacc hrej
  · have hws : ((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2.2.1).1.getD [] = ws := by
      rw [hg.2]; rfl
    have htrue : ∀ p ∈ D.calls x (((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2.2.1).1.getD []) ++
        D.calls x (((EO D I DO decO).eval H P x t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2).1.getD []),
        H p.1 = p.2 :=
      fun p hp => (List.mem_append.1 hp).elim (calls_true_of hT hN0 t.1 (t.2.1, t.2.2.2.1) p)
        (calls_true_of hT hN0 t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) p)
    have hsub : ∀ p ∈ D.calls x ws, p ∈ D.calls x (((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2.2.1).1.getD []) ++
        D.calls x (((EO D I DO decO).eval H P x t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2).1.getD []) :=
      fun p hp => List.mem_append_left _ (by rw [hws]; exact hp)
    obtain ⟨⟨u', v'⟩, huv'⟩ := Option.isSome_iff_exists.1
      (findColl_isSome_of_sub H htrue hsub (Option.isSome_iff_exists.2 ⟨(u, v), huv⟩))
    exact ⟨u', v', huv', findColl_spec H _ htrue u' v' huv'⟩

/-- **Bad reruns and a rejecting judged draw together cost what `InnerKS` holds for bad reruns**: `bad_rej_le`, from
the played rounds' openings and `CallsTrue` (`Pr_bad_le_played`, `Pr_bad_le_fork_played`,
`Pr_innerRejects_le_fork_true`). -/
theorem bad_rej_le_played {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} (hV : InnerViewRun H D I DO) (hO : CallsOpenPlayed H D I DO opening)
    (hT : CallsTrue H D I DO) (hR : InnerRounds D I) (hA : InnerAcceptsColl H D I DO) (hN0 : CallsNil H D)
    (hM : RoundsMargin H D I) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    NI D I x * Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) +
        Pr (InnerRejects H D I DO decO P x) ≤
      NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x * PrBe H D I DO decO P x) := by
  have hn0 : (0 : ℝ) ≤ NI D I x := Nat.cast_nonneg _
  by_cases hE : Pr (InnerRejects H D I DO decO P x) = 0
  · rw [hE, add_zero]
    exact mul_le_mul_of_nonneg_left (Pr_bad_le_played hV hO hT hR decO x P) hn0
  obtain ⟨t, ws, hg, -⟩ : ∃ t, InnerRejects H D I DO decO P x t := by
    by_contra hn
    refine hE ((Pr_congr fun t => ⟨fun h => hn ⟨t, h⟩, False.elim⟩).trans Pr_false)
  obtain ⟨hkI, hk3⟩ := hM x Ω P t.1 t.2.1 hg.1
  let S : ℕ → ℝ := fun j => Pr (forkColl H D I DO decO P x j)
  have hk : 0 < D.V.k x := by omega
  have hES : ∀ j, Pr (InnerRejects H D I DO decO P x) ≤ S j := fun j =>
    Pr_innerRejects_le_fork_true hA hT hN0 decO x P j
  have htot : (D.V.k x : ℝ) * PrBe H D I DO decO P x = ∑ j ∈ Finset.range (D.V.k x), S j := by
    rw [PrBe_eq_avg H D I DO decO P x hk, mul_div_cancel₀ _ (by exact_mod_cast hk.ne'),
      Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j))]
  have hsplit : ∑ j ∈ Finset.range (D.V.k x), S j =
      ∑ j ∈ Finset.range (I.V.k (xI D x)), S j + ∑ j ∈ Finset.Ico (I.V.k (xI D x)) (D.V.k x), S j :=
    (Finset.sum_range_add_sum_Ico S (hR x)).symm
  have hB : 3 * Pr (InnerRejects H D I DO decO P x) ≤ ∑ j ∈ Finset.Ico (I.V.k (xI D x)) (D.V.k x), S j := by
    have hcard : (3 : ℝ) ≤ (Finset.Ico (I.V.k (xI D x)) (D.V.k x)).card := by
      rw [Nat.card_Ico]
      exact_mod_cast (by omega : 3 ≤ D.V.k x - I.V.k (xI D x))
    calc 3 * Pr (InnerRejects H D I DO decO P x)
        ≤ (Finset.Ico (I.V.k (xI D x)) (D.V.k x)).card * Pr (InnerRejects H D I DO decO P x) :=
          mul_le_mul_of_nonneg_right hcard (Pr_nonneg _)
      _ = ∑ _j ∈ Finset.Ico (I.V.k (xI D x)) (D.V.k x), Pr (InnerRejects H D I DO decO P x) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ j ∈ Finset.Ico (I.V.k (xI D x)) (D.V.k x), S j := Finset.sum_le_sum fun j _ => hES j
  have hAa : ∑ j ∈ Finset.range (I.V.k (xI D x)), S j ≤ I.V.k (xI D x) := by
    calc ∑ j ∈ Finset.range (I.V.k (xI D x)), S j ≤ ∑ _j ∈ Finset.range (I.V.k (xI D x)), (1 : ℝ) :=
          Finset.sum_le_sum fun j _ => Pr_le_one _
      _ = I.V.k (xI D x) := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]
  have hA0 : 0 ≤ ∑ j ∈ Finset.range (I.V.k (xI D x)), S j := Finset.sum_nonneg fun j _ => Pr_nonneg _
  have hbad := Pr_bad_le_fork_played hV hO hT decO x P
  rw [Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j))] at hbad
  have hn1 : (1 : ℝ) ≤ NI D I x := by
    have : 1 ≤ NI D I x := by simp only [NI]; omega
    exact_mod_cast this
  have hm := sqrt_margin hn1 (by exact_mod_cast hkI : (1 : ℝ) ≤ I.V.k (xI D x)) hA0 hAa (Pr_nonneg _)
    (Pr_le_one _) hB
  calc (NI D I x : ℝ) * Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
        Bad H D I DO decO P x t.1 t.2) + Pr (InnerRejects H D I DO decO P x)
      ≤ NI D I x * Real.sqrt (I.V.k (xI D x) * ∑ j ∈ Finset.range (I.V.k (xI D x)), S j) +
          Pr (InnerRejects H D I DO decO P x) :=
        add_le_add (mul_le_mul_of_nonneg_left hbad hn0) le_rfl
    _ ≤ NI D I x * Real.sqrt (I.V.k (xI D x) * (∑ j ∈ Finset.range (I.V.k (xI D x)), S j +
          ∑ j ∈ Finset.Ico (I.V.k (xI D x)) (D.V.k x), S j)) := hm
    _ = NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x * PrBe H D I DO decO P x) := by
        rw [mul_assoc, htot, hsplit]

/-- **`InnerKS` from `Inner.KSAbort` where V* meets `InnerAcceptsColl`**: `innerKS_of_KSAbort_coll`, from the played
rounds' openings, `CallsTrue` and the inner verifier's calls off a collision (`PrBAbort_le_PrBI_coll`,
`bad_rej_le_played`). -/
theorem innerKS_of_KSAbort_played [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {opening : Opening} {decC : Msg → Msg → Coin} (hKS : I.KSAbort H)
    (hR : InnerRounds D I) (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hLC : CallsLogColl H D I DO)
    (hO : CallsOpenPlayed H D I DO opening) (hA : InnerAcceptsColl H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) : InnerKS H D I DO := by
  intro decO _ x Ω _ _ P
  have h1 := PrBadGood_le_abort_rej hV hpc hR decO x P
  have h2 := hKS (xI D x) Ω (innerP H D I DO decO P x decC) ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
    (keepA H D I DO decO P x)
  have h3 := PrBAbort_le_PrBI_coll hLC hT hN0 hpc hR decO x P
  have h4 := bad_rej_le_played hV hO hT hR hA hN0 hM decO x P
  have hBA0 : 0 ≤ PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x) (keepA H D I DO decO P x) := by
    unfold PrBAbort; split
    · exact Pr_nonneg _
    · exact le_rfl
  have h5 : Real.sqrt (I.γ (xI D x) * PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x)
      (keepA H D I DO decO P x)) ≤ Real.sqrt (I.γ (xI D x) * PrBI H D I DO decO P x) := by
    by_cases hγ : 0 ≤ I.γ (xI D x)
    · exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left h3 hγ)
    · rw [Real.sqrt_eq_zero'.2 (mul_nonpos_of_nonpos_of_nonneg (not_le.1 hγ).le hBA0)]
      exact Real.sqrt_nonneg _
  linarith [h2.trans (add_le_add le_rfl h5)]

end Coupling

section RecursionKS

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **The recursion's KS from Flock's KS at the outer deployment and the inner protocol's KS against provers that may
abort, where V* meets `InnerAcceptsColl`**: `KS_of_flockKS_coll`, from the played rounds' openings
(`CallsOpenPlayed`), the recorded calls' truth (`CallsTrue`) and the inner verifier's calls off a collision
(`CallsLogColl`) in place of `CallsOpen` and `CallsRecord`. -/
theorem KS_of_flockKS_played [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (opening : Opening) (decC : Msg → Msg → Coin) (hF : Flock.KS H DO)
    (hKS : I.KSAbort H) (hRef : Refuses H D) (hRounds : InnerRounds D I) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : SessionRounds D I DO) (hS : SessionAccepts D) (hR : SessionRuns H D I DO)
    (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hLC : CallsLogColl H D I DO)
    (hO : CallsOpenPlayed H D I DO opening) (hA : InnerAcceptsColl H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) : KSA H D I DO :=
  KS_of H D I DO hRef hRounds (outerKS_of_flockKS hF hN hε hRd hS hR)
    (innerKS_of_KSAbort_played hKS hRounds hV hT hLC hO hA hM hpc hN0)

/-- **The recursion's KS from V\*'s parts**: `KS_of_flockKS_played`, with every code fact about the readers and V\*
from V*'s parts at the deployed readers (`codeFacts_of_staging`): the compiled verifier reads through them
(`ReadsThrough`), every part's SHA-512 gadgets compute `H` (`PartCallsTrue`), an accepted run's first pairs show its
coins (`RunPairs`), the layout spans no more than the inner and staging rounds, and V*'s parts meet
`VStarPartsStaging` at the dense, holo and sparse shapes. -/
theorem KS_of_parts [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (L : Layout) (opening : Opening) (decC decS : Msg → Msg → Coin) (hF : Flock.KS H DO) (hKS : I.KSAbort H)
    (hRef : Refuses H D) (hRounds : InnerRounds D I) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : SessionRounds D I DO) (hS : SessionAccepts D) (hR : SessionRuns H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hRT : ReadsThrough H D I.V L.msgs L.calls) (hT : ∀ s, PartCallsTrue H D DO L s)
    (hP : RunPairs H D I decS) (hL : ∀ x, L.rounds x ≤ I.V.k (xI D x) + D.kS x)
    (hD : VStarPartsStaging D I DO H L opening decS Form.Dense)
    (hH : VStarPartsStaging D I DO H L opening decS Form.Holo)
    (hSp : VStarPartsStaging D I DO H L opening decS Form.Sparse) : KSA H D I DO := by
  obtain ⟨hV, hN0, hTr, hO, hLC, hA⟩ := codeFacts_of_staging hRef hRT hT hP hL hD hH hSp
  exact KS_of_flockKS_played H D I DO opening decC hF hKS hRef hRounds hN hε hRd hS hR hV hTr hLC hO hA hM hpc hN0

end RecursionKS

end Claims.Recursion.VStar
