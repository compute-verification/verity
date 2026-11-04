import Proofs.Flock.Claims.Recursion.PlurLin
import Proofs.Flock.Claims.Recursion.BoundForks

/-!
# A bad draw, linearly

`Pr_bad_le` bounds a draw that `P_I`'s runs don't show by `√(k_I·k(x)·Pr[Be])`, the root from `Pr_plur_fork`. Here the
same draw is bounded linearly (`Pr_bad_le_lin`). At each inner round `j` it misses `P_I` at most as often as a link at
the fork at `j` succeeds, plus `1/(M+1)` (`Pr_plur_fork_lin`), and where a link succeeds, two extractions forked at `j`
open round `j` differently, so their calls collide wherever the inner rounds' forks bind (`ForksBind`, `linkOut_coll`):
at the deployed records directly (`RecDeploy.forksBind_C`), and wherever the played rounds' openings are recorded
(`forksBind_of_opening`).

The deviation finder `Fdev` (`Definitions.Claims.RecursionLin`) draws the round `j` uniformly, then a tape, a prefix, a
draw and the loop's `M` draws. It outputs the collision between the first draw and the loop's first draw where `E_O`
gives witnesses (`linkOut`); its declared cost is one draw's calls, then the loop's (`linkCost`), at most twice one
draw's in expectation (`Ex_Fdev_cost_le`). So `Pr[Bad] ≤ k_I·Pr[Fdev] + k_I/(M+1)`, and under `ExpectedCR` at `Fdev`,
`Pr[Bad] ≤ k_I·(2·E[c]/2^256 + 1/(M+1))` at one draw's calls `c` (`Pr_bad_le_ecr`).
-/
namespace Claims.Recursion

open Claims

section BadLin

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The inner rounds' forks bind wherever the played rounds' openings are recorded** (`forkColl_of_opening`). -/
theorem forksBind_of_opening {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {opening : Opening} (hV : InnerViewRun H D I DO) (hO : CallsOpenPlayed H D I DO opening)
    (hT : CallsTrue H D I DO) : ForksBind H D I DO :=
  fun decO x _ P _ t h => forkColl_of_opening hV hO hT decO x P t h

/-- **A link that succeeds names a collision**: the first draw and the loop's first where `E_O` gives witnesses open
round `j` differently, at two draws forked at `j` (`fork_fork`), so their calls collide (`ForksBind`). -/
theorem linkOut_coll {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hFk : ForksBind H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} (P : Prover Ω) {M : ℕ}
    (j : Fin (I.V.k (xI D x))) (ω : Ω) (r : Fin (D.V.k x) → Coin) (u : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
    (us : Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
    (hg : (eoOut H D I DO decO P x ω (fork j r u.1, u.2)).isSome = true)
    (hd : firstDiffers (fun v => (eoOut H D I DO decO P x ω (fork j r v.1, v.2)).isSome = true)
      (fun v => eoMsg H D I DO decO P x ω (fork j r v.1, v.2) j) (eoMsg H D I DO decO P x ω (fork j r u.1, u.2) j)
      M us) :
    ∃ a b, linkOut H D I DO decO P x ((j, ω, r, u, us) : LinCoins D I DO decO Ω x M) = some (a, b) ∧
      Collision H a b := by
  obtain ⟨v, hv, hgv, hfv⟩ := firstPick_of_firstDiffers M us hd
  obtain ⟨ws, hws⟩ := Option.isSome_iff_exists.1 hg
  obtain ⟨ws', hws'⟩ := Option.isSome_iff_exists.1 hgv
  have hod : OpensDiffer H D I DO decO P x j (ω, fork j r u.1, v.1, u.2, v.2) := by
    refine ⟨ws, ws', eoGives_of_some hws, ?_, ?_⟩
    · simp only [fork_fork]
      exact eoGives_of_some hws'
    · simp only [fork_fork]
      simpa [eoMsg, hws, hws', dec] using Ne.symm hfv
  have hfc := hFk decO x Ω P j _ hod
  simp only [forkColl, fork_fork] at hfc
  have hv' : firstPick (linkG H D I DO decO P x j ω r) M us = some v := hv
  simp only [linkOut, hv', Option.getD_some]
  exact hfc

/-- **A bad draw, linearly**: at most `k_I` times the link finder's success, plus `k_I/(M+1)`. -/
theorem Pr_bad_le_lin {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hFk : ForksBind H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (M : ℕ) (hk : 0 < I.V.k (xI D x)) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      I.V.k (xI D x) * (Fdev H D I DO decO P x M hk).Pr H + I.V.k (xI D x) / ((M : ℝ) + 1) := by
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
      Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
          (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
        ∃ a b, linkOut H D I DO decO P x ((j, t) : LinCoins D I DO decO Ω x M) = some (a, b) ∧ Collision H a b) +
        1 / ((M : ℝ) + 1) := fun j => by
    have hp := Pr_plur_fork_lin (j : ℕ) (fun ω d => (eoOut H D I DO decO P x ω d).isSome = true)
      (fun ω d => eoMsg H D I DO decO P x ω d j) M
    refine (le_of_eq (Pr_congr fun t => ?_)).trans (hp.trans (add_le_add (Pr_mono fun t ht => ?_) le_rfl))
    · simp only [opAt, fork_preCoins]
    · exact linkOut_coll hFk decO x P j t.1 t.2.1 t.2.2.1 t.2.2.2 ht.1 ht.2
  have hF : (I.V.k (xI D x) : ℝ) * (Fdev H D I DO decO P x M hk).Pr H =
      ∑ j : Fin (I.V.k (xI D x)), Pr (fun t : Ω × (Fin (D.V.k x) → Coin) ×
          ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) × (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
        ∃ a b, linkOut H D I DO decO P x ((j, t) : LinCoins D I DO decO Ω x M) = some (a, b) ∧ Collision H a b) := by
    have hP : (Fdev H D I DO decO P x M hk).Pr H = Ex fun j : Fin (I.V.k (xI D x)) =>
        Pr fun t : Ω × (Fin (D.V.k x) → Coin) × ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
          (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
        ∃ a b, linkOut H D I DO decO P x ((j, t) : LinCoins D I DO decO Ω x M) = some (a, b) ∧ Collision H a b :=
      Pr_prod (fun s : LinCoins D I DO decO Ω x M => ∃ a b, linkOut H D I DO decO P x s = some (a, b) ∧ Collision H a b)
    rw [hP]
    unfold Ex
    rw [Fintype.card_fin, mul_div_cancel₀ _ (by exact_mod_cast hk.ne')]
  calc Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2)
      ≤ ∑ j : Fin (I.V.k (xI D x)), (Pr (fun t : Ω × (Fin (D.V.k x) → Coin) ×
          ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) × (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
        ∃ a b, linkOut H D I DO decO P x ((j, t) : LinCoins D I DO decO Ω x M) = some (a, b) ∧ Collision H a b) +
          1 / ((M : ℝ) + 1)) :=
        h1.trans (Finset.sum_le_sum fun j _ => h2 j)
    _ = I.V.k (xI D x) * (Fdev H D I DO decO P x M hk).Pr H + I.V.k (xI D x) / ((M : ℝ) + 1) := by
        rw [Finset.sum_add_distrib, ← hF, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
          mul_one_div]

/-- **The link finder costs at most two draws in expectation**: at each round, tape and prefix, `Ex_linkCost_le`;
the draws forked at `j` from a uniform prefix are uniform (`Ex_fork`). -/
theorem Ex_Fdev_cost_le (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M : ℕ)
    (hk : 0 < I.V.k (xI D x)) :
    Ex (fun s : (Fdev H D I DO decO P x M hk).S => ((Fdev H D I DO decO P x M hk).cost s : ℝ)) ≤
      2 * Ex (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
        (drawCost H D I DO decO P x t.1 t.2 : ℝ)) := by
  have hoff : ∀ j : ℕ, PrefixDet (k := D.V.k x) (Coin := Coin) fun _ => j := fun _ _ _ _ => rfl
  have hω : ∀ (j : Fin (I.V.k (xI D x))) (ω : Ω),
      (Ex fun r : Fin (D.V.k x) → Coin =>
        Ex fun b : ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
            (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
          (linkCost (linkG H D I DO decO P x j ω r)
            (fun v => drawCost H D I DO decO P x ω (fork j r v.1, v.2)) M b : ℝ)) ≤
        2 * Ex fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => (drawCost H D I DO decO P x ω d : ℝ) := by
    intro j ω
    let F : (Fin (D.V.k x) → Coin) → ℝ := fun r =>
      Ex fun c : (EO D I DO decO).C x => (drawCost H D I DO decO P x ω (r, c) : ℝ)
    have hu : ∀ r, (Ex fun v : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
        (drawCost H D I DO decO P x ω (fork j r v.1, v.2) : ℝ)) = Ex fun r' => F (fork j r r') := fun r => Ex_prod _
    calc _ ≤ Ex fun r : Fin (D.V.k x) → Coin => 2 * Ex fun v : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
            (drawCost H D I DO decO P x ω (fork j r v.1, v.2) : ℝ) :=
          Ex_mono fun r => Ex_linkCost_le _ _ M
      _ = 2 * Ex fun p : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) => F (fork j p.1 p.2) := by
          rw [Ex_mul_left, Ex_prod (fun p : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) => F (fork j p.1 p.2))]
          exact congrArg (2 * ·) (congrArg Ex (funext hu))
      _ = 2 * Ex F := by rw [Ex_fork (hoff j) F]
      _ = _ := congrArg (2 * ·)
          (Ex_prod (fun d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => (drawCost H D I DO decO P x ω d : ℝ))).symm
  have hj : ∀ j : Fin (I.V.k (xI D x)),
      Ex (fun rest : Ω × (Fin (D.V.k x) → Coin) × ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
          (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
        (linkCost (linkG H D I DO decO P x j rest.1 rest.2.1)
          (fun v => drawCost H D I DO decO P x rest.1 (fork j rest.2.1 v.1, v.2)) M rest.2.2 : ℝ)) ≤
        2 * Ex (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
          (drawCost H D I DO decO P x t.1 t.2 : ℝ)) := by
    intro j
    rw [Ex_prod (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
      (drawCost H D I DO decO P x t.1 t.2 : ℝ)), ← Ex_mul_left]
    exact (Ex_prod _).trans_le (Ex_mono fun ω => (Ex_prod _).trans_le (hω j ω))
  calc Ex (fun s : (Fdev H D I DO decO P x M hk).S => ((Fdev H D I DO decO P x M hk).cost s : ℝ))
      = Ex fun j : Fin (I.V.k (xI D x)) =>
          Ex fun rest : Ω × (Fin (D.V.k x) → Coin) × ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) ×
              (Fin M → (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) =>
            (linkCost (linkG H D I DO decO P x j rest.1 rest.2.1)
              (fun v => drawCost H D I DO decO P x rest.1 (fork j rest.2.1 v.1, v.2)) M rest.2.2 : ℝ) :=
        Ex_prod (fun s : LinCoins D I DO decO Ω x M => ((Fdev H D I DO decO P x M hk).cost s : ℝ))
    _ ≤ Ex fun _ : Fin (I.V.k (xI D x)) => 2 * Ex (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
          (drawCost H D I DO decO P x t.1 t.2 : ℝ)) := Ex_mono hj
    _ = _ := by
        have : Nonempty (Fin (I.V.k (xI D x))) := ⟨⟨0, hk⟩⟩
        exact Ex_const _

/-- **A bad draw under `ExpectedCR` at the link finder**: at most `k_I·(2·E[c]/2^256 + 1/(M+1))`, at one draw's calls
`c`. Linear in the collision term, where `Pr_bad_le` is `√(k_I·k(x)·Pr[Be])`. -/
theorem Pr_bad_le_ecr {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hFk : ForksBind H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (M : ℕ) (hk : 0 < I.V.k (xI D x)) (hcr : ExpectedCR H (Fdev H D I DO decO P x M hk)) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      I.V.k (xI D x) * (2 * Ex (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
        (drawCost H D I DO decO P x t.1 t.2 : ℝ)) / 2 ^ (256 : ℝ)) + I.V.k (xI D x) / ((M : ℝ) + 1) := by
  have hF : (Fdev H D I DO decO P x M hk).Pr H ≤
      2 * Ex (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
        (drawCost H D I DO decO P x t.1 t.2 : ℝ)) / 2 ^ (256 : ℝ) :=
    hcr.trans (div_le_div_of_nonneg_right (Ex_Fdev_cost_le H D I DO decO x P M hk) (by positivity))
  have := mul_le_mul_of_nonneg_left hF (Nat.cast_nonneg (I.V.k (xI D x)) : (0 : ℝ) ≤ I.V.k (xI D x))
  linarith [Pr_bad_le_lin hFk decO x P M hk]

end BadLin

end Claims.Recursion
