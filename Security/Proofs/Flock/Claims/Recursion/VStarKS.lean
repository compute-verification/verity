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
  recorded off a collision (`PrBAbort_le_reg`, from `CallsLogColl`).

A part may also fall back on a tie between one of its recorded calls and an evaluation the V* setup makes at its
statement, which isn't among the recorded calls alone: each place then charges its fork `Be`'s collision or `Br`'s
registered tie (`forkReg`), and the forks' charges sum to `k(x)·(Pr[Be] + Pr[Br])` (`sum_forkReg_le`). So the inner
stage's two finder terms both read `k(x)·(Pr[Be] + Pr[Br])` (`InnerKSR`), and `KS_of_reg` is `KS_of` with that sum,
split under the root into `KSA`'s `√(γ_eA·Pr[Be]) + √(γ_eA·Pr[Br])` (`sqrt_mul_add_le`).

This file is that chain with those three facts in place of `CallsOpen` and `CallsRecord` (`Pr_bad_le_played`,
`Pr_bad_le_fork_played`, `Pr_innerRejects_le_fork_true`, `bad_rej_le_played`, `innerKSR_of_KSAbort`,
`KS_of_flockKS_played`). `KS_of_parts` is the recursion's KS from Flock's KS at the outer deployment, the inner
protocol's KS against provers that may abort, the session facts, and V*'s parts at the deployed readers
(`codeFacts_of_staging`).
-/

namespace Claims.Recursion.VStar

section Coupling

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **A draw is bad at most as `√(k_I·k(x)·(Pr[Be] + Pr[Br]))`**: `Pr_bad_le`, with the played rounds' openings
(`sum_opensDiffer_le_played`). -/
theorem Pr_bad_le_played {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} {fixed : Msg → ℕ → Option Msg} {pub : Msg → ℕ → Bool} (hV : InnerViewRun H D I DO)
    (hO : CallsOpenPlayed H D I DO opening fixed pub)
    (hT : CallsTrue H D I DO) (hR : InnerRounds D I) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      Real.sqrt (I.V.k (xI D x) * D.V.k x * (PrBe H D I DO decO P x + PrBr H D I DO decO P x)) := by
  rw [mul_assoc]
  exact (Pr_bad_le_opens H D I DO decO x P).trans
    (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (sum_opensDiffer_le_played hV hO hT hR decO x P)
      (Nat.cast_nonneg _)))

/-- **A draw is bad at most as `√(k_I·Σ_{j<k_I} Pr[forkReg j])`**: `Pr_bad_le_fork`, with the played rounds'
openings (`forkColl_of_openingPlayed`), each fork charging `Be`'s collision or `Br`'s registered tie. -/
theorem Pr_bad_le_fork_played {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} {fixed : Msg → ℕ → Option Msg} {pub : Msg → ℕ → Bool} (hV : InnerViewRun H D I DO)
    (hO : CallsOpenPlayed H D I DO opening fixed pub)
    (hT : CallsTrue H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      Real.sqrt (I.V.k (xI D x) * ∑ j : Fin (I.V.k (xI D x)), Pr (forkReg H D I DO decO P x j)) := by
  exact (Pr_bad_le_opens H D I DO decO x P).trans (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
    (Finset.sum_le_sum fun j _ => Pr_mono fun t => forkColl_of_openingPlayed hV hO hT decO x P t)
    (Nat.cast_nonneg _)))

/-- **A rejected inner view is what every fork charges**: `Pr_innerRejects_le_fork`, where V* accepts off a collision or
a registered tie (`InnerAcceptsR`), each `Be`'s or `Br`'s at the fork (`coll_or_reg`), and `Be` reads only true
evaluations by `CallsTrue` and `CallsNil` (`calls_true_of`). -/
theorem Pr_innerRejects_le_fork_true {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} (hA : InnerAcceptsR H D I DO) (hT : CallsTrue H D I DO) (hN0 : CallsNil H D)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) (j : ℕ) :
    Pr (InnerRejects H D I DO decO P x) ≤ Pr (forkReg H D I DO decO P x j) := by
  rw [← (unifMap_judged (Ω := Ω) (R := Fin (D.V.k x) → Coin) (C := (EO D I DO decO).C x)).Pr]
  refine Pr_mono fun t ht => ?_
  obtain ⟨ws, hg, hrej⟩ := ht
  rcases hA decO x Ω P t.1 t.2.1 t.2.2.2.1 ws hg with hacc | hc
  · exact absurd hacc hrej
  · have hws : ((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2.2.1).1.getD [] = ws := by
      rw [hg.2]; rfl
    have htrue : ∀ p ∈ D.calls x (((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2.2.1).1.getD []) ++
        D.calls x (((EO D I DO decO).eval H P x t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2).1.getD []),
        H p.1 = p.2 :=
      fun p hp => (List.mem_append.1 hp).elim (calls_true_of hT hN0 t.1 (t.2.1, t.2.2.2.1) p)
        (calls_true_of hT hN0 t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) p)
    rw [← hws] at hc
    unfold forkReg forkColl regColl
    exact coll_or_reg htrue (collOrReg_of_callsColl htrue (Or.inl hc))

/-- **Bad reruns and a rejecting judged draw together cost what `InnerKSR` holds for bad reruns**: `bad_rej_le`, from
the played rounds' openings and `CallsTrue` (`Pr_bad_le_played`, `Pr_bad_le_fork_played`,
`Pr_innerRejects_le_fork_true`), the forks' charges summing to at most `k(x)·(Pr[Be] + Pr[Br])`
(`sum_forkReg_le`). -/
theorem bad_rej_le_played {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} {fixed : Msg → ℕ → Option Msg} {pub : Msg → ℕ → Bool} (hV : InnerViewRun H D I DO)
    (hO : CallsOpenPlayed H D I DO opening fixed pub)
    (hT : CallsTrue H D I DO) (hR : InnerRounds D I) (hA : InnerAcceptsR H D I DO) (hN0 : CallsNil H D)
    (hM : RoundsMargin H D I) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    NI D I x * Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) +
        Pr (InnerRejects H D I DO decO P x) ≤
      NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x * (PrBe H D I DO decO P x + PrBr H D I DO decO P x)) := by
  have hn0 : (0 : ℝ) ≤ NI D I x := Nat.cast_nonneg _
  by_cases hE : Pr (InnerRejects H D I DO decO P x) = 0
  · rw [hE, add_zero]
    exact mul_le_mul_of_nonneg_left (Pr_bad_le_played hV hO hT hR decO x P) hn0
  obtain ⟨t, ws, hg, -⟩ : ∃ t, InnerRejects H D I DO decO P x t := by
    by_contra hn
    refine hE ((Pr_congr fun t => ⟨fun h => hn ⟨t, h⟩, False.elim⟩).trans Pr_false)
  obtain ⟨hkI, hk3⟩ := hM x Ω P t.1 t.2.1 hg.1
  let S : ℕ → ℝ := fun j => Pr (forkReg H D I DO decO P x j)
  have hES : ∀ j, Pr (InnerRejects H D I DO decO P x) ≤ S j := fun j =>
    Pr_innerRejects_le_fork_true hA hT hN0 decO x P j
  have htot : ∑ j ∈ Finset.range (D.V.k x), S j ≤
      (D.V.k x : ℝ) * (PrBe H D I DO decO P x + PrBr H D I DO decO P x) := by
    rw [← Fin.sum_univ_eq_sum_range (fun j => Pr (forkReg H D I DO decO P x j))]
    exact sum_forkReg_le H D I DO decO P x
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
  rw [Fin.sum_univ_eq_sum_range (fun j => Pr (forkReg H D I DO decO P x j))] at hbad
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
    _ ≤ NI D I x * Real.sqrt (I.V.k (xI D x) * (D.V.k x * (PrBe H D I DO decO P x + PrBr H D I DO decO P x))) := by
        rw [← hsplit]
        exact mul_le_mul_of_nonneg_left
          (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left htot (Nat.cast_nonneg _))) hn0
    _ = NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x * (PrBe H D I DO decO P x + PrBr H D I DO decO P x)) := by
        rw [mul_assoc]

/-- **The inner protocol's KS, carried to the served runs, with registered ties**: `InnerKS` with
`k(x)·(Pr[Be] + Pr[Br])` for both of its finders' terms. A tie between a recorded call and an evaluation the V* setup
makes isn't among the recorded calls alone, so where a part falls back on one, the inner `B`'s collisions and the
inner rounds' differing openings are `Be`'s or `Br`'s at a fork. -/
def InnerKSR [DecidableEq Col] [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
        (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
        (fun x => I.check (xI D x)) x P ≤
      I.κ (xI D x) + Real.sqrt (I.γ (xI D x) * D.V.k x * (PrBe H D I DO decO P x + PrBr H D I DO decO P x)) +
        NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x * (PrBe H D I DO decO P x + PrBr H D I DO decO P x))

/-- **`InnerKSR` from `Inner.KSAbort` where V* meets `InnerAcceptsR`**: `innerKS_of_KSAbort_coll`, from the played
rounds' openings, `CallsTrue` and the inner verifier's calls off a collision or a registered tie (`PrBAbort_le_reg`,
`bad_rej_le_played`). -/
theorem innerKSR_of_KSAbort [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {opening : Opening} {fixed : Msg → ℕ → Option Msg} {pub : Msg → ℕ → Bool}
    {decC : Msg → Msg → Coin} (hKS : I.KSAbort H)
    (hR : InnerRounds D I) (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hLC : CallsLogColl H D I DO)
    (hO : CallsOpenPlayed H D I DO opening fixed pub) (hA : InnerAcceptsR H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) : InnerKSR H D I DO := by
  intro decO _ x Ω _ _ P
  have h1 := PrBadGood_le_abort_rej hV hpc hR decO x P
  have h2 := hKS (xI D x) Ω (innerP H D I DO decO P x decC) ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
    (keepA H D I DO decO P x)
  have h3 := PrBAbort_le_reg hLC hT hN0 hpc hR decO x P
  have h4 := bad_rej_le_played hV hO hT hR hA hN0 hM decO x P
  have hBA0 : 0 ≤ PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x) (keepA H D I DO decO P x) := by
    unfold PrBAbort; split
    · exact Pr_nonneg _
    · exact le_rfl
  have h5 : Real.sqrt (I.γ (xI D x) * PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x)
      (keepA H D I DO decO P x)) ≤
      Real.sqrt (I.γ (xI D x) * D.V.k x * (PrBe H D I DO decO P x + PrBr H D I DO decO P x)) := by
    by_cases hγ : 0 ≤ I.γ (xI D x)
    · rw [mul_assoc]
      exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left h3 hγ)
    · rw [Real.sqrt_eq_zero'.2 (mul_nonpos_of_nonpos_of_nonneg (not_le.1 hγ).le hBA0)]
      exact Real.sqrt_nonneg _
  linarith [h2.trans (add_le_add le_rfl h5)]

end Coupling

section RecursionKS

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- `√(a·(q + r)) ≤ √(a·q) + √(a·r)` at `a, q, r ≥ 0`. -/
theorem sqrt_mul_add_le {a q r : ℝ} (ha : 0 ≤ a) (hq : 0 ≤ q) (hr : 0 ≤ r) :
    Real.sqrt (a * (q + r)) ≤ Real.sqrt (a * q) + Real.sqrt (a * r) := by
  have h1 := Real.sq_sqrt (mul_nonneg ha hq)
  have h2 := Real.sq_sqrt (mul_nonneg ha hr)
  have h3 := Real.sqrt_nonneg (a * q)
  have h4 := Real.sqrt_nonneg (a * r)
  rw [Real.sqrt_le_left (add_nonneg h3 h4)]
  nlinarith [mul_nonneg h3 h4]

/-- **The recursion's KS with registered ties**: `KS_of` with `InnerKSR` for `InnerKS`. Both of the inner stage's
finder terms read `k(x)·(Pr[Be] + Pr[Br])`, so they sit under one root `√(γ_eA·(Pr[Be] + Pr[Br]))`, which is at most
`KSA`'s two terms `√(γ_eA·Pr[Be]) + √(γ_eA·Pr[Br])`. -/
theorem KS_of_reg [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (hRef : Refuses H D) (hO : OuterKS H D I DO) (hI : InnerKSR H D I DO) : KSA H D I DO := by
  intro decO hdec x Ω _ _ P
  set A : ℝ := (NI D I x : ℝ)
  set p := PrB H D.V P x
  set q := PrBe H D I DO decO P x
  set r := PrBr H D I DO decO P x
  set G := ∑ s : Fin (D.nS x), Real.sqrt ((D.nums x s).γ * D.V.k x)
  have hA : (1 + (I.waits (xI D x) : ℝ) * (I.k (xI D x) : ℝ)) = A := by simp only [A, NI]; push_cast; ring
  have hA0 : 0 ≤ A := Nat.cast_nonneg _
  have hp0 : 0 ≤ p := PrB_nonneg H D.V P x
  have hq0 : 0 ≤ q := PrBe_nonneg H D I DO decO P x
  have hr0 : 0 ≤ r := PrBr_nonneg H D I DO decO P x
  have hG0 : 0 ≤ G := Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _
  have hsplit := servedERep_bad_le_good H D.V (R D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
    (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
    (fun x => I.check (xI D x)) x P
  rw [hA] at hsplit
  have hgood : PrBadGood H D.V (R D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
      (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
      (fun x => I.check (xI D x)) x P ≤
      PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
      (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
      (fun x => I.check (xI D x)) x P := by
    refine Pr_mono fun t h => ⟨h.1, h.2.1, fun hi => h.2.2 ?_⟩
    obtain ⟨s, hs, hc⟩ := hRef x Ω P t.1 t.2.1 h.1
    exact R_of_innerR D I hs hc _ hi
  have hin := hI decO hdec x Ω P
  have hEO : PrBad H D.V outR (EO D I DO decO) x P ≤
      ∑ s : Fin (D.nS x), (D.nums x s).κ + G * Real.sqrt p := by
    refine (PrBad_EO_le H D I DO decO x P).trans ?_
    rw [Finset.sum_mul, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun s _ => (hO decO hdec x Ω P s).trans (le_of_eq ?_)
    rw [Real.sqrt_mul (mul_nonneg (Nums.γ_nonneg _) (Nat.cast_nonneg _))]
  have hB : A * (G * Real.sqrt p) = Real.sqrt (γA D I x * p) := by
    have hγ : γA D I x = (A * G) ^ 2 := rfl
    rw [hγ, Real.sqrt_mul' _ hp0, Real.sqrt_sq (mul_nonneg hA0 hG0), mul_assoc]
  have hqr0 : 0 ≤ q + r := add_nonneg hq0 hr0
  have hBe : Real.sqrt (I.γ (xI D x) * D.V.k x * (q + r)) + A * Real.sqrt (I.V.k (xI D x) * D.V.k x * (q + r)) =
      Real.sqrt (γeA D I x * (q + r)) := by
    have hγe : γeA D I x = (Real.sqrt (I.γ (xI D x) * D.V.k x) + A * Real.sqrt (I.V.k (xI D x) * D.V.k x)) ^ 2 :=
      rfl
    rw [hγe, Real.sqrt_mul' (I.γ (xI D x) * D.V.k x) hqr0, Real.sqrt_mul' ((I.V.k (xI D x) : ℝ) * D.V.k x) hqr0,
      Real.sqrt_mul' (_ ^ 2) hqr0,
      Real.sqrt_sq (add_nonneg (Real.sqrt_nonneg _) (mul_nonneg hA0 (Real.sqrt_nonneg _)))]
    ring
  have hγe0 : 0 ≤ γeA D I x := sq_nonneg _
  have hroot := sqrt_mul_add_le hγe0 hq0 hr0
  have hκ : κ D I x = I.κ (xI D x) + A * ∑ s : Fin (D.nS x), (D.nums x s).κ := rfl
  show PrBad H D.V (R D I) (served D.V (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
    (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
    (fun x => I.check (xI D x))) x P ≤ _
  rw [served_eq, hκ]
  have hmul := mul_le_mul_of_nonneg_left hEO hA0
  nlinarith [hsplit, hgood, hin, hmul, hB, hBe, hroot]

/-- **The recursion's KS from Flock's KS at the outer deployment and the inner protocol's KS against provers that may
abort, where V* meets `InnerAcceptsR`**: `KS_of_flockKS_coll`, from the played rounds' openings
(`CallsOpenPlayed`), the recorded calls' truth (`CallsTrue`) and the inner verifier's calls off a collision or a
registered tie (`CallsLogColl`) in place of `CallsOpen` and `CallsRecord`, at `KS_of_reg`. -/
theorem KS_of_flockKS_played [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (opening : Opening) (decC : Msg → Msg → Coin) {fixed : Msg → ℕ → Option Msg}
    {pub : Msg → ℕ → Bool} (hF : Flock.KS H DO)
    (hKS : I.KSAbort H) (hRef : Refuses H D) (hRounds : InnerRounds D I) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : SessionRounds D I DO) (hS : SessionAccepts D) (hR : SessionRuns H D I DO)
    (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hLC : CallsLogColl H D I DO)
    (hO : CallsOpenPlayed H D I DO opening fixed pub) (hA : InnerAcceptsR H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) : KSA H D I DO :=
  KS_of_reg H D I DO hRef (outerKS_of_flockKS hF hN hε hRd hS hR)
    (innerKSR_of_KSAbort hKS hRounds hV hT hLC hO hA hM hpc hN0)

/-- **The recursion's KS from V\*'s parts**: `KS_of_flockKS_played`, with every code fact about the readers and V\*
from V*'s parts at the deployed readers (`codeFacts_of_staging`): the compiled verifier reads through them
(`ReadsThrough`), its V* statements number at least `D.nS` (`VstarLen`), every part's SHA-512 gadgets compute `H`
(`PartCallsTrue`), an accepted run's first pairs show its coins (`RunPairs`), the readers read the layout's public
rounds off the run (`ReadsPub`), the last inner round opens (`FinalOpens`), the layout spans no more than the inner
and staging rounds, and V*'s parts meet `VStarPartsStaging` at the dense, holo and sparse shapes. -/
theorem KS_of_parts [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (L : Layout) (opening : Opening) (decC decS : Msg → Msg → Coin) (hF : Flock.KS H DO) (hKS : I.KSAbort H)
    (hRef : Refuses H D) (hRounds : InnerRounds D I) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : SessionRounds D I DO) (hS : SessionAccepts D) (hR : SessionRuns H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hRT : ReadsThrough H D I.V L.msgs L.calls) (hVL : VstarLen D)
    (hT : ∀ s, PartCallsTrue H D DO L s) (hP : RunPairs H D I decS) (hPub : ReadsPub H D I DO L)
    (hFin : FinalOpens H D I DO opening) (hL : ∀ x, L.rounds x ≤ I.V.k (xI D x) + D.kS x)
    (hD : VStarPartsStaging D I DO H L opening decS Form.Dense)
    (hH : VStarPartsStaging D I DO H L opening decS Form.Holo)
    (hSp : VStarPartsStaging D I DO H L opening decS Form.Sparse) : KSA H D I DO := by
  obtain ⟨hV, hN0, hTr, hO, hLC, hA⟩ := codeFacts_of_staging hRef hRT hVL hT hP hPub hFin hL hD hH hSp
  exact KS_of_flockKS_played H D I DO opening decC hF hKS hRef hRounds hN hε hRd hS hR hV hTr hLC hO hA hM hpc hN0

end RecursionKS

end Claims.Recursion.VStar
