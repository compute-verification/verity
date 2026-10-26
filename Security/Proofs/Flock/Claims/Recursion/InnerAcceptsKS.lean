import Proofs.Flock.Claims.Recursion.InnerAccepts
import Proofs.Flock.Claims.Recursion.FlockKSDeploy

/-!
# The recursion's KS where the inner view may reject on a collision

`innerKS_of_KSAbort` reads `InnerAccepts`: wherever `E_O` gives witnesses on an accepted run, the inner view is
accepted. At SHA-512 V* meets the weaker `InnerAcceptsColl` (`InnerAccepts.lean`): accepted, or two of the witnesses'
recorded calls collide. This file proves `InnerKS` at the same bound, `γ_eA` unchanged, from `InnerAcceptsColl` and
one more code fact, `RoundsMargin`: on an accepted run the inner protocol takes a round and the compiled verifier at
least three rounds past the inner ones.

* The inner stage fails where `E^rep` fails against `P_I`'s aborting runs, at a bad rerun, or where the judged draw's
  inner view rejects (`PrBadGood_le_abort_rej`): `InnerAccepts` only closed the last case.
* A bad draw costs at most `√(k_I·Σ_{j<k_I} Pr[forkColl j])` (`Pr_bad_le_fork`): the differing openings are collisions
  at the inner rounds' forks only.
* A rejecting judged draw is a collision at every fork (`Pr_innerRejects_le_fork`), so at the forks past the inner
  rounds, which `Pr_bad_le_fork` leaves unused, it costs at most a third of their sum. Those forks' spare mass covers
  it (`sqrt_margin`), so `N_I·Pr[bad] + Pr[rejects] ≤ N_I·√(k_I·k(x)·Pr[Be])` (`bad_rej_le`), the term `InnerKS`
  holds for bad reruns alone.

`KSFlock_of_flockKS_V_coll` is `KSFlock_of_flockKS_V` with `InnerAcceptsColl` and the margin for `InnerAccepts`.
-/

namespace Claims.Recursion

open Claims

section Coupling

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The compiled verifier's rounds past the inner ones** (a code fact): on an accepted run, the inner protocol takes a
round at the inner statement, and the compiled verifier at least three more than it. -/
def RoundsMargin (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
    0 < I.V.k (xI D x) ∧ I.V.k (xI D x) + 3 ≤ D.V.k x

/-- **A draw is bad at most as `√(k_I·Σ_{j<k_I} Pr[forkColl j])`**: `Pr_bad_le`'s argument, stopped before the inner
rounds' forks are bounded by all of `Be`'s. -/
theorem Pr_bad_le_fork {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} (hV : InnerViewRun H D I DO) (hO : CallsOpenPlayed H D I DO opening)
    (hT : CallsTrue H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      Real.sqrt (I.V.k (xI D x) * ∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j)) := by
  exact (Pr_bad_le_opens H D I DO decO x P).trans (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
    (Finset.sum_le_sum fun j _ => Pr_mono fun t => forkColl_of_opening hV hO hT decO x P t) (Nat.cast_nonneg _)))

/-- **Spare mass under the root**: `n·√(a·A) + E ≤ n·√(a·(A+B))` when `B` holds three `E`s, `E ≤ 1`, `A ≤ a`,
`a ≥ 1` and `n ≥ 1`, since `(√(a·A) + E)² ≤ a·A + 2a·E + a·E`. -/
theorem sqrt_margin {n a A B E : ℝ} (hn : 1 ≤ n) (ha : 1 ≤ a) (hA0 : 0 ≤ A) (hAa : A ≤ a) (hE0 : 0 ≤ E)
    (hE1 : E ≤ 1) (hB : 3 * E ≤ B) : n * Real.sqrt (a * A) + E ≤ n * Real.sqrt (a * (A + B)) := by
  have ha0 : 0 ≤ a := by linarith
  have hu0 : 0 ≤ Real.sqrt (a * A) := Real.sqrt_nonneg _
  have hu : Real.sqrt (a * A) ≤ a :=
    (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hAa ha0)).trans_eq (Real.sqrt_mul_self ha0)
  have hsq : Real.sqrt (a * A) ^ 2 = a * A := Real.sq_sqrt (mul_nonneg ha0 hA0)
  have h1 : Real.sqrt (a * A) * E ≤ a * E := mul_le_mul_of_nonneg_right hu hE0
  have h2 : E * E ≤ E * 1 := mul_le_mul_of_nonneg_left hE1 hE0
  have h3 : E ≤ a * E := by nlinarith
  have h4 : a * (3 * E) ≤ a * B := mul_le_mul_of_nonneg_left hB ha0
  have hv : Real.sqrt (a * A) + E ≤ Real.sqrt (a * (A + B)) := by
    calc Real.sqrt (a * A) + E = Real.sqrt ((Real.sqrt (a * A) + E) ^ 2) := (Real.sqrt_sq (by positivity)).symm
      _ ≤ Real.sqrt (a * (A + B)) := Real.sqrt_le_sqrt (by nlinarith)
  nlinarith [mul_le_mul_of_nonneg_left hv (by linarith : (0 : ℝ) ≤ n)]

theorem unifMap_drop {Ω R C Q : Type} [Fintype Ω] [Fintype R] [Fintype C] [Fintype Q] [Nonempty Q] :
    UnifMap (fun t : Ω × R × C × Q => (t.1, t.2.1, t.2.2.1)) := by
  let e : Ω × R × C × Q ≃ (Ω × R × C) × Q :=
    { toFun := fun t => ((t.1, t.2.1, t.2.2.1), t.2.2.2)
      invFun := fun s => (s.1.1, s.1.2.1, s.1.2.2, s.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have hfst : UnifMap (fun p : (Ω × R × C) × Q => p.1) := fun G => by
    rw [Ex_prod]
    simp only [Ex_const]
  exact (UnifMap.equiv e).comp hfst

/-- **The inner stage fails where `E^rep` fails against `P_I`'s aborting runs, at a bad rerun, or where the judged
draw's inner view rejects**: `PrBadGood_le_abort`'s argument, where the judged draw's acceptance is read off the draw
where its inner view doesn't reject, in place of `InnerAccepts`. -/
theorem PrBadGood_le_abort_rej {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decC : Msg → Msg → Coin} (hV : InnerViewRun H D I DO) (hpc : PublicCoin H I decC) (hR : InnerRounds D I)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) :
    PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
        (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
        (fun x => I.check (xI D x)) x P ≤
      I.PrBadAbort H (xI D x) (innerP H D I DO decO P x decC) (keepA H D I DO decO P x) +
        NI D I x * Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
          Bad H D I DO decO P x t.1 t.2) + Pr (InnerRejects H D I DO decO P x) := by
  let Yb : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
      (Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) → Prop :=
    fun s => (innerF H D I DO decO P x decC s.1 (s.2.1, s.2.2.1)).out = .acc ∧
      ¬ I.R.ext (xI D x) ((erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens
        (I.assemble (xI D x)) (I.check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).eval H
          (innerF H D I DO decO P x decC s.1)).1
  let Zb : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
      (Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) → Prop :=
    fun s => ∃ d ∈ (erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens (I.assemble (xI D x))
      (I.check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
        (servedRun H (Claims.outerStage (EO D I DO decO) (dec H D) x) (run H P D.V x s.1)), Bad H D I DO decO P x s.1 d
  let Jb : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
      (Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) → Prop :=
    fun s => InnerRejects H D I DO decO P x (s.1, s.2.1, s.2.2.1)
  have hYZ : PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
      (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
      (fun x => I.check (xI D x)) x P ≤ Pr Yb + Pr Zb + Pr Jb := by
    refine (Pr_mono fun s hs => ?_).trans ((Pr_or_le (fun s => Yb s ∨ Zb s) Jb).trans
      (add_le_add (Pr_or_le Yb Zb) le_rfl))
    by_cases hz : Zb s
    · exact Or.inl (Or.inr hz)
    by_cases hj : Jb s
    · exact Or.inr hj
    left; left
    have hnb : ∀ d ∈ (erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens (I.assemble (xI D x))
        (I.check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
          (servedRun H (Claims.outerStage (EO D I DO decO) (dec H D) x) (run H P D.V x s.1)),
        ¬ Bad H D I DO decO P x s.1 d := fun d hd hb => hz ⟨d, hd, hb⟩
    have heq := RP.eval_congr H _ (innerF H D I DO decO P x decC s.1) _ fun d hd =>
      servedRun_seen_eq hV hpc (hnb d hd)
    have hd₀ : (s.2.1, s.2.2.1) ∈ (erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens
        (I.assemble (xI D x)) (I.check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
          (servedRun H (Claims.outerStage (EO D I DO decO) (dec H D) x) (run H P D.V x s.1)) := List.mem_cons_self
    have hgood := hs.2.1 _ hd₀
    obtain ⟨ws, hws⟩ : ∃ ws, eoOut H D I DO decO P x s.1 (s.2.1, s.2.2.1) = some ws := by
      rcases ho : eoOut H D I DO decO P x s.1 (s.2.1, s.2.2.1) with _ | ws
      · refine absurd ⟨hs.1, fun h => ?_⟩ hgood
        rw [show ((EO D I DO decO).eval H P x s.1 s.2.1 s.2.2.1).1 = none from ho] at h
        exact h
      · exact ⟨ws, rfl⟩
    have ha : Agrees H D I DO decO P x s.1 (s.2.1, s.2.2.1) := by
      by_contra hn
      exact hnb _ hd₀ ⟨by rw [hws]; rfl, hn⟩
    refine ⟨?_, fun hI => hs.2.2 ?_⟩
    · rw [innerF_eq_innerRun hpc hws ha]
      have hacc : (D.innerView H x s.2.1 ws).out = .acc := by
        by_contra hne
        exact hj ⟨ws, eoGives_of_some hws, hne⟩
      rw [hV decO x Ω P s.1 s.2.1 s.2.2.1 ws (eoGives_of_some hws)] at hacc
      exact hacc
    · show (innerR D I).ext x (((erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens
        (I.assemble (xI D x)) (I.check (xI D x)) (s.2.1, s.2.2.1) s.2.2.2).serve
          (Claims.outerStage (EO D I DO decO) (dec H D) x)).eval H (run H P D.V x s.1)).1
      rw [(RP.serve_eval H _ _ _).1, heq]
      exact (innerR_ext D I x _).2 hI
  have hY : Pr Yb = I.PrBadAbort H (xI D x) (innerP H D I DO decO P x decC) (keepA H D I DO decO P x) := by
    have hΦ : UnifMap (fun s : Ω × ((Fin (I.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
        (Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) →
          (Fin (I.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
        (s.1, (extendDraw s.2.1).1, (extendDraw s.2.1).2, fun i j => extendDraw (s.2.2 i j))) := by
      let e : Ω × ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
          (Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ≃
          Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
          (Fin (I.waits (xI D x)) → Fin (I.cap (xI D x)) → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :=
        { toFun := fun s => (s.1, s.2.1.1, s.2.1.2, s.2.2)
          invFun := fun s => (s.1, (s.2.1, s.2.2.1), s.2.2.2)
          left_inv := fun _ => rfl
          right_inv := fun _ => rfl }
      exact (UnifMap.self.prod ((unifMap_extendDraw (hR x)).prod
        (UnifMap.pi (UnifMap.pi (unifMap_extendDraw (hR x)) _) _))).comp (UnifMap.equiv e)
    rw [← hΦ.Pr Yb]
    refine Pr_congr fun s => ?_
    show (innerF H D I DO decO P x decC s.1 (extendDraw s.2.1)).out = .acc ∧
      ¬ I.R.ext (xI D x) ((erepProg (I.waits (xI D x)) (I.k (xI D x)) (I.cap (xI D x)) I.opens
        (I.assemble (xI D x)) (I.check (xI D x)) (extendDraw s.2.1) (fun i j => extendDraw (s.2.2 i j))).eval H
          (innerF H D I DO decO P x decC s.1)).1 ↔ _
    have hf : (fun d => innerF H D I DO decO P x decC s.1 (extendDraw d)) =
        abortRun H (innerP H D I DO decO P x decC) I.V (xI D x) (keepA H D I DO decO P x) s.1 :=
      funext fun d => (abortRun_eq (hR x) s.1 d).symm
    rw [erepProg_eval_comp, hf, ← abortRun_eq (hR x)]
    rfl
  have hZ : Pr Zb ≤ NI D I x * Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      Bad H D I DO decO P x t.1 t.2) := by
    have hN : (NI D I x : ℝ) = 1 + (I.waits (xI D x) : ℝ) * (I.k (xI D x) : ℝ) := by
      simp only [NI]; push_cast; ring
    rw [hN]
    exact Pr_exists_rerun_le H D.V (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x)) (fun x => I.k (xI D x))
      (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x)) (fun x => I.check (xI D x)) x P
      (Bad H D I DO decO P x)
  have hJ : Pr Jb = Pr (InnerRejects H D I DO decO P x) :=
    (unifMap_drop (Ω := Ω) (R := Fin (D.V.k x) → Coin) (C := (EO D I DO decO).C x)).Pr
      (InnerRejects H D I DO decO P x)
  linarith

/-- **Bad reruns and a rejecting judged draw together cost what `InnerKS` holds for bad reruns**:
`N_I·Pr[bad] + Pr[rejects] ≤ N_I·√(k_I·k(x)·Pr[Be])`. A bad draw costs at most `√(k_I·A)`, `A` the inner rounds'
forks' collisions (`Pr_bad_le_fork`); a rejecting draw at most each fork's, so a third of `B`, the forks' past the
inner rounds (`Pr_innerRejects_le_fork`, `RoundsMargin`), and `k(x)·Pr[Be] = A + B` (`sqrt_margin`). Where no run is
accepted, no draw rejects, and `Pr_bad_le` alone gives it. -/
theorem bad_rej_le {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} (hV : InnerViewRun H D I DO) (hO : CallsOpenPlayed H D I DO opening)
    (hT : CallsTrue H D I DO) (hR : InnerRounds D I) (hA : InnerAcceptsColl H D I DO) (hN0 : CallsNil H D)
    (hM : RoundsMargin H D I)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    NI D I x * Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) +
        Pr (InnerRejects H D I DO decO P x) ≤
      NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x * PrBe H D I DO decO P x) := by
  have hn0 : (0 : ℝ) ≤ NI D I x := Nat.cast_nonneg _
  by_cases hE : Pr (InnerRejects H D I DO decO P x) = 0
  · rw [hE, add_zero]
    exact mul_le_mul_of_nonneg_left (Pr_bad_le hV hO hT hR decO x P) hn0
  obtain ⟨t, ws, hg, -⟩ : ∃ t, InnerRejects H D I DO decO P x t := by
    by_contra hn
    refine hE ((Pr_congr fun t => ⟨fun h => hn ⟨t, h⟩, False.elim⟩).trans Pr_false)
  obtain ⟨hkI, hk3⟩ := hM x Ω P t.1 t.2.1 hg.1
  let S : ℕ → ℝ := fun j => Pr (forkColl H D I DO decO P x j)
  have hk : 0 < D.V.k x := by omega
  have hES : ∀ j, Pr (InnerRejects H D I DO decO P x) ≤ S j := fun j =>
    Pr_innerRejects_le_fork hA hT hN0 decO x P j
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
  have hbad := Pr_bad_le_fork hV hO hT decO x P
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

/-- **`InnerKS` from `Inner.KSAbort` where V* meets `InnerAcceptsColl`**: `innerKS_of_KSAbort`, with the judged draw's
rejection paid from the spare forks (`PrBadGood_le_abort_rej`, `bad_rej_le`), at the same bound. -/
theorem innerKS_of_KSAbort_coll [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {opening : Opening} {decC : Msg → Msg → Coin} (hKS : I.KSAbort H)
    (hR : InnerRounds D I) (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hL : CallsLogColl H D I DO)
    (hO : CallsOpenPlayed H D I DO opening) (hA : InnerAcceptsColl H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) : InnerKS H D I DO := by
  intro decO _ x Ω _ _ P
  have h1 := PrBadGood_le_abort_rej hV hpc hR decO x P
  have h2 := hKS (xI D x) Ω (innerP H D I DO decO P x decC) ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
    (keepA H D I DO decO P x)
  have h3 := PrBAbort_le_PrBI hL hT hN0 hpc hR decO x P
  have h4 := bad_rej_le hV hO hT hR hA hN0 hM decO x P
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
abort, where V* meets `InnerAcceptsColl`**: `KS_of_flockKS` with `innerKS_of_KSAbort_coll`. -/
theorem KS_of_flockKS_coll [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (opening : Opening) (decC : Msg → Msg → Coin) (hF : Flock.KS H DO)
    (hKS : I.KSAbort H) (hRef : Refuses H D) (hRounds : InnerRounds D I) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : SessionRounds D I DO) (hS : SessionAccepts D) (hR : SessionRuns H D I DO)
    (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hL : CallsLogColl H D I DO)
    (hO : CallsOpenPlayed H D I DO opening) (hA : InnerAcceptsColl H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) :
    KSA H D I DO :=
  KS_of H D I DO hRef hRounds (outerKS_of_flockKS hF hN hε hRd hS hR)
    (innerKS_of_KSAbort_coll hKS hRounds hV hT hL hO hA hM hpc hN0)

/-- **KS of the recursion over Flock where V* meets `InnerAcceptsColl`**: `KSFlock_of_flockKS` with
`KS_of_flockKS_coll`. The margin is the compiled verifier's: on an accepted run, three rounds past the inner
deployment's, which takes a round at every statement. -/
theorem KSFlock_of_flockKS_coll [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (DI DO : Flock.Deploy Coin Col T)
    (opening : Opening) (decC : Msg → Msg → Coin) (hF : Flock.KS H DO) (NI : Msg → ℕ) (Ok : Msg → Col → Prop)
    (hNI : ∀ x, (NI x : ℝ) ≤ DI.N₀ x) (hεI : ∀ x, 0 ≤ DI.εtable x) (hVkI : ∀ x, 0 < DI.V.k x)
    (hKI : ∀ x, 0 < DI.cap x) (hkI : ∀ x, (8 ≤ DI.k x ∧ Flock.Splits DI x) ∨ Flock.NoAcc H DI.V x)
    (hone : ∀ x, (∃ s, DI.parse x = some s ∧ s.tables = 1) ∨ Flock.NoAcc H DI.V x) (hz : ∀ x, Ok x 0)
    (hok : Flock.ColsOk H DI NI Ok) (hconf : Flock.ConflictTied H DI NI) (htab : TableBoundθOnAcc H DI NI Ok)
    (hlastI : AccLast H DI.V) (hfreeI : DecideFree H DI.V) (hrefI : Refusable H DI.V)
    (hRef : Refuses H D) (hRounds : ∀ x, DI.V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : ∀ decI, SessionRounds D (Inner.ofFlock DI decI) DO) (hS : SessionAccepts D)
    (hR : ∀ decI, SessionRuns H D (Inner.ofFlock DI decI) DO)
    (hV : ∀ decI, InnerViewRun H D (Inner.ofFlock DI decI) DO)
    (hT : ∀ decI, CallsTrue H D (Inner.ofFlock DI decI) DO)
    (hL : ∀ decI, CallsLogColl H D (Inner.ofFlock DI decI) DO)
    (hO : ∀ decI, CallsOpenPlayed H D (Inner.ofFlock DI decI) DO opening)
    (hA : ∀ decI, InnerAcceptsColl H D (Inner.ofFlock DI decI) DO)
    (hM : ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
      DI.V.k (xI D x) + 3 ≤ D.V.k x)
    (hpc : ∀ decI, PublicCoin H (Inner.ofFlock DI decI) decC) (hN0 : CallsNil H D) : KSFlockA H D DI DO :=
  fun decI h => KS_of_flockKS_coll H D _ DO opening decC hF
    (KSAbort_of_tableBoundOnAcc H DI NI Ok decC hNI hεI hVkI hKI hkI hone hz hok hconf htab
      (coinsShown_of_publicCoin (hpc decI)) hlastI hfreeI hrefI decI h) hRef hRounds hN hε
    (hRd decI) hS (hR decI) (hV decI) (hT decI) (hL decI) (hO decI) (hA decI)
    (fun x Ω P ω r hacc => ⟨hVkI _, hM x Ω P ω r hacc⟩) (hpc decI) hN0

end RecursionKS

end Claims.Recursion

namespace FlockDeploy

open Claims.Recursion

/-- **KS of the recursion over Flock's deployed verifier as the inner protocol, where V* meets `InnerAcceptsColl`**, at
SHA-512: `KSFlock_of_flockKS_V` with `InnerAcceptsColl` and the compiled verifier's margin (`KSFlock_of_flockKS_coll`)
for `InnerAccepts`. -/
theorem KSFlock_of_flockKS_V_coll {Col T : Type} [DecidableEq Col] [Zero Col] (zk : Bool) (D : Deploy Coin)
    (DI DO : Claims.Flock.Deploy Coin Col T) (hVI : DI.V = V zk) (opening : Opening)
    (decC : Claims.Msg → Claims.Msg → Coin) (hF : Claims.Flock.KS sha512 DO) (NI : Claims.Msg → ℕ)
    (Ok : Claims.Msg → Col → Prop) (hNI : ∀ x, (NI x : ℝ) ≤ DI.N₀ x) (hεI : ∀ x, 0 ≤ DI.εtable x)
    (hVkI : ∀ x, 0 < DI.V.k x) (hKI : ∀ x, 0 < DI.cap x)
    (hkI : ∀ x, (8 ≤ DI.k x ∧ Claims.Flock.Splits DI x) ∨ Claims.Flock.NoAcc sha512 DI.V x)
    (hone : ∀ x, (∃ s, DI.parse x = some s ∧ s.tables = 1) ∨ Claims.Flock.NoAcc sha512 DI.V x) (hz : ∀ x, Ok x 0)
    (hok : Claims.Flock.ColsOk sha512 DI NI Ok) (hconf : Claims.Flock.ConflictTied sha512 DI NI)
    (htab : TableBoundθOnAcc sha512 DI NI Ok) (hRef : Refuses sha512 D) (hRounds : ∀ x, DI.V.k (xI D x) ≤ D.V.k x)
    (hN : SessionNums D DO) (hε : TableErrNonneg D) (hRd : ∀ decI, SessionRounds D (Inner.ofFlock DI decI) DO)
    (hS : SessionAccepts D) (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock DI decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock DI decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock DI decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock DI decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock DI decI) DO opening)
    (hA : ∀ decI, InnerAcceptsColl sha512 D (Inner.ofFlock DI decI) DO)
    (hM : ∀ (x : Claims.Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin),
      (Claims.run sha512 P D.V x ω r).out = .acc → DI.V.k (xI D x) + 3 ≤ D.V.k x)
    (hpc : ∀ decI, PublicCoin sha512 (Inner.ofFlock DI decI) decC) (hN0 : CallsNil sha512 D) :
    KSFlockA sha512 D DI DO :=
  KSFlock_of_flockKS_coll sha512 D DI DO opening decC hF NI Ok hNI hεI hVkI hKI hkI hone hz hok hconf htab
    (hVI ▸ accLast_V_sha512 zk) (hVI ▸ decideFree_V sha512 zk) (hVI ▸ refusable_V sha512 zk) hRef hRounds hN hε
    hRd hS hR hV hT hL hO hA hM hpc hN0

end FlockDeploy
