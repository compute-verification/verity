import Proofs.Flock.Claims.Recursion.FlockKSLin
import Proofs.Flock.Claims.Recursion.CostExp
import Proofs.Flock.Soundness.Game.LockLin

/-!
# A cap position's off mass at the abort law, linearly, as `FC`'s hits

`TableBoundAbortLin` charges a cap position `q` to `FC`'s run and loop at `q`. Here is the charge, for any opening `op`
of `q` that a run makes exactly where it accepts and queries `q` (`opensAt`), and whose two values at runs sharing the
coins before `q`'s round tie a collision among their calls:

* **The plurality after the round** (`capPlur`): the value that runs forked at the round from `d₀` open most often. It
  reads only `d₀`'s coins before the round (`capPlur_prefix`).
* **A fork at a fixed round of a uniform run is uniform** (`Ex_capFork`), `Ex_fork` at a constant round.
* **One base run's forks** (`off_le_succ`): `LinkFinder.succ_ge` at `drawGame`, one reading per run: forks open off
  their plurality at most as often as the link finder over them succeeds, plus `1/(eM)`.
* **The link** (`Pr_off_le_capHit`): the finder's success is `FC`'s hit at the fork (`fork_fork`: `FC`'s loop from a
  fork is the finder's loop), and averaging over the base run, a run opens `q` off the plurality after `q`'s round with
  probability at most `FC`'s hit at `q` plus `1/(M+1)`.
-/

namespace Claims.Recursion

open Claims Claims.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Rewinding

section CapLink

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- The value `f` opens most often over uniform draws. -/
noncomputable def plurOf {Dr Val : Type} [Fintype Dr] [Fintype Val] [Nonempty Val] (f : Dr → Option Val) : Val :=
  (Finset.exists_max_image Finset.univ (fun v => Pr fun e => f e = some v) Finset.univ_nonempty).choose

theorem plurOf_max {Dr Val : Type} [Fintype Dr] [Fintype Val] [Nonempty Val] (f : Dr → Option Val) (v : Val) :
    (Pr fun e => f e = some v) ≤ Pr fun e => f e = some (plurOf f) :=
  (Finset.exists_max_image Finset.univ (fun v => Pr fun e => f e = some v) Finset.univ_nonempty).choose_spec.2 v
    (Finset.mem_univ v)

/-- **The plurality opening after round `j`**: the value that runs forked at `j` from `d₀`, with fresh abort coins,
open most often. -/
noncomputable def capPlur (V : Verifier Coin) {A Val : Type} [Fintype A] [Fintype Val] [Nonempty Val] (x : Msg)
    (j : ℕ) (op : (Fin (V.k x) → Coin) × A → Option Val) (d₀ : (Fin (V.k x) → Coin) × A) : Val :=
  plurOf fun e : (Fin (V.k x) → Coin) × A => op (fork j d₀.1 e.1, e.2)

omit [Nonempty Coin] [Inhabited Coin] in
/-- The plurality after round `j` reads only the coins before it. -/
theorem capPlur_prefix (V : Verifier Coin) {A Val : Type} [Fintype A] [Fintype Val] [Nonempty Val] (x : Msg) (j : ℕ)
    (op : (Fin (V.k x) → Coin) × A → Option Val) {d₀ d₀' : (Fin (V.k x) → Coin) × A}
    (h : ∀ i : Fin (V.k x), i.val < j → d₀.1 i = d₀'.1 i) : capPlur V x j op d₀ = capPlur V x j op d₀' := by
  have hf : fork j d₀.1 = fork j d₀'.1 := funext fun r => funext fun i => by
    by_cases hi : i.val < j
    · simp only [fork, hi, ↓reduceIte, h i hi]
    · simp only [fork, hi, ↓reduceIte]
  unfold capPlur
  rw [hf]

omit [Inhabited Coin] in
/-- **A fork at a cap's round of a uniform run is uniform.** -/
theorem Ex_capFork (V : Verifier Coin) (Cp : Caps Coin) {A : Type} [Fintype A] [Nonempty A] (x : Msg)
    (q : Fin (Cp.width x)) (Ψ : (Fin (V.k x) → Coin) × A → ℝ) :
    (Ex fun p : ((Fin (V.k x) → Coin) × A) × ((Fin (V.k x) → Coin) × A) => Ψ (capFork V Cp x q p.1 p.2)) =
      Ex Ψ := by
  have hoff : PrefixDet (k := V.k x) (Coin := Coin) fun _ => Cp.round x q := fun _ _ _ => rfl
  calc (Ex fun p : ((Fin (V.k x) → Coin) × A) × ((Fin (V.k x) → Coin) × A) => Ψ (capFork V Cp x q p.1 p.2))
      = Ex fun r₀ : Fin (V.k x) → Coin => Ex fun _ : A => Ex fun e : (Fin (V.k x) → Coin) × A =>
          Ψ (fork (Cp.round x q) r₀ e.1, e.2) :=
        (Claims.Bridge.Ex_prod _).trans (Claims.Bridge.Ex_prod _)
    _ = Ex fun p : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) =>
          (fun r => Ex fun a : A => Ψ (r, a)) (fork (Cp.round x q) p.1 p.2) := by
        rw [Claims.Bridge.Ex_prod (fun p : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) =>
          (fun r => Ex fun a : A => Ψ (r, a)) (fork (Cp.round x q) p.1 p.2))]
        refine congrArg Ex (funext fun r₀ => ?_)
        rw [Ex_const, Claims.Bridge.Ex_prod]
    _ = Ex fun r => Ex fun a : A => Ψ (r, a) := Ex_fork hoff fun r => Ex fun a : A => Ψ (r, a)
    _ = Ex Ψ := (Claims.Bridge.Ex_prod Ψ).symm

/-- **One base run's forks, linearly**: draws open off their plurality at most as often as the link finder over them
succeeds, plus `1/(eM)`. -/
theorem off_le_succ {Dr Val : Type} [Fintype Dr] [Nonempty Dr] [Fintype Val] [DecidableEq Val] [Nonempty Val]
    (f : Dr → Option Val) (M : ℕ) (hM : 1 ≤ M) :
    (Pr fun e => (f e).isSome ∧ f e ≠ some (plurOf f)) ≤
      (Ex fun e => Ex fun vs : Fin M → Dr =>
        ind (LinkFinder.Succ (rdAt (fun e (_ : Unit) => f e) ())) (e, (List.finRange M).map vs)) +
        1 / (Real.exp 1 * M) := by
  have hX : ∀ v, LinkFinder.pRd (drawGame Dr) (drawS Dr) (rdAt (fun e (_ : Unit) => f e) ()) v ≤
      LinkFinder.pRd (drawGame Dr) (drawS Dr) (rdAt (fun e (_ : Unit) => f e) ()) (plurOf f) := fun v => by
    simp only [pRd_rdAt, expect_drawGame]
    have := plurOf_max f v
    rwa [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Pr_eq_Ex] at this
  have h := LinkFinder.succ_ge (drawGame Dr) (drawS Dr) (rdAt (fun e (_ : Unit) => f e) ()) hX M hM
  rw [sRd_rdAt, zero_add, LinkFinder.expect_finderG] at h
  simp only [pRd_rdAt, expect_drawGame, expectN_drawGame] at h
  have hoff : (Pr fun e => (f e).isSome ∧ f e ≠ some (plurOf f)) =
      (∑ v, Ex (ind fun e => f e = some v)) - Ex (ind fun e => f e = some (plurOf f)) := by
    rw [Claims.Bridge.Pr_eq_Ex]
    exact expect_off_eq (fun e (_ : Unit) => f e) (drawGame Dr) (drawS Dr) (fun _ => plurOf f) ()
  rw [hoff]
  linarith

/-- **A cap position's off mass, linearly, as `FC`'s hits there**: for an opening of `q` that a run makes exactly
where it accepts and queries `q`, and whose two different values at runs sharing the coins before `q`'s round tie a
collision among their calls, a run opens `q` off the plurality after `q`'s round with probability at most that of
`FC`'s run and loop at `q` finding a collision, plus `1/(M+1)`. -/
theorem Pr_off_le_capHit (H : Msg → Msg) (V : Verifier Coin) (Cp : Caps Coin) {Ω A Val : Type} [Fintype Ω]
    [Nonempty Ω] [Fintype A] [Nonempty A] [Fintype Val] [DecidableEq Val] [Nonempty Val] (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (M : ℕ) (q : Fin (Cp.width x)) (ω : Ω)
    (op : (Fin (V.k x) → Coin) × A → Option Val)
    (hop : ∀ d, (op d).isSome ↔ opensAt H V Cp P x keep q ω d)
    (htie : ∀ d d' v v', (∀ i : Fin (V.k x), i.val < Cp.round x q → d.1 i = d'.1 i) → op d = some v →
      op d' = some v' → v ≠ v' →
        TiedColl H ((abortRun H P V x keep ω d).vLog ++ (abortRun H P V x keep ω d').vLog)) :
    (Pr fun d => (op d).isSome ∧ op d ≠ some (capPlur V x (Cp.round x q) op d)) ≤
      (Pr fun s : ((Fin (V.k x) → Coin) × A) × (Fin M → (Fin (V.k x) → Coin) × A) =>
        ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, s.1, s.2) = some (u, v) ∧ Collision H u v) +
        1 / ((M : ℝ) + 1) := by
  classical
  rcases Nat.eq_zero_or_pos M with rfl | hM
  · refine (Pr_le_one _).trans ?_
    rw [Nat.cast_zero, zero_add, div_one]
    exact le_add_of_nonneg_left (Pr_nonneg _)
  have hstep : ∀ (d₀ e : (Fin (V.k x) → Coin) × A) (vs : Fin M → (Fin (V.k x) → Coin) × A),
      LinkFinder.Succ (rdAt (fun e (_ : Unit) => op (capFork V Cp x q d₀ e)) ()) (e, (List.finRange M).map vs) →
        ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, capFork V Cp x q d₀ e, vs) = some (u, v) ∧
          Collision H u v := by
    intro d₀ e vs hs
    simp only [LinkFinder.Succ] at hs
    rcases hs with h2 | ⟨v, hv, e', hn, hne⟩
    · have := rdAt_card_le (fun e (_ : Unit) => op (capFork V Cp x q d₀ e)) () e
      omega
    rw [rdAt_eq_singleton] at hv
    have hne_iff : ∀ e₁, (rdAt (fun e (_ : Unit) => op (capFork V Cp x q d₀ e)) () e₁).Nonempty ↔
        opensAt H V Cp P x keep q ω (capFork V Cp x q d₀ e₁) := fun e₁ => by
      rw [← hop]
      unfold rdAt
      dsimp only
      cases op (capFork V Cp x q d₀ e₁) <;> simp
    have hsome : (rdAt (fun e (_ : Unit) => op (capFork V Cp x q d₀ e)) () e').Nonempty := by
      unfold LinkFinder.nextRead at hn
      simpa using List.find?_some hn
    obtain ⟨v', hv'⟩ : ∃ v', op (capFork V Cp x q d₀ e') = some v' := by
      unfold rdAt at hsome
      cases h : op (capFork V Cp x q d₀ e') with
      | none => simp [h] at hsome
      | some w => exact ⟨w, rfl⟩
    have hvv : v ≠ v' := fun h => hne ((rdAt_eq_singleton _ () e' v).2 (by simp only [hv', h]))
    have hcf : ∀ w, capFork V Cp x q (capFork V Cp x q d₀ e) w = capFork V Cp x q d₀ w := fun w => by
      simp only [capFork, fork_fork]
    have hfp : firstPick (fun w => opensAt H V Cp P x keep q ω (capFork V Cp x q d₀ w)) M vs = some e' := by
      rw [← find?_finRange_map, ← hn]
      unfold LinkFinder.nextRead
      congr 1
      funext e₁
      exact decide_eq_decide.2 (hne_iff e₁).symm
    have hopen : opensAt H V Cp P x keep q ω (capFork V Cp x q d₀ e) := (hop _).1 (by simp [hv])
    have hout : (FC H V Cp P x keep M q.pos).out (q, ω, capFork V Cp x q d₀ e, vs) =
        findColl ((abortRun H P V x keep ω (capFork V Cp x q d₀ e)).vLog ++
          (abortRun H P V x keep ω (capFork V Cp x q d₀ e')).vLog) := by
      show (if opensAt H V Cp P x keep q ω (capFork V Cp x q d₀ e) then _ else none) = _
      rw [ite_eq_left hopen]
      simp only [hcf, hfp, Option.getD_some]
    have htied := htie (capFork V Cp x q d₀ e) (capFork V Cp x q d₀ e') v v'
      (fun i hi => (fork_agree (Cp.round x q) d₀.1 e.1 i hi).symm.trans (fork_agree (Cp.round x q) d₀.1 e'.1 i hi))
      hv hv' hvv
    have hlog : ∀ p ∈ (abortRun H P V x keep ω (capFork V Cp x q d₀ e)).vLog ++
        (abortRun H P V x keep ω (capFork V Cp x q d₀ e')).vLog, H p.1 = p.2 := fun p hp => by
      rcases List.mem_append.1 hp with hp | hp
      · exact abortRun_vLog H P V x keep ω _ p hp
      · exact abortRun_vLog H P V x keep ω _ p hp
    obtain ⟨⟨u, w⟩, huw⟩ := Option.isSome_iff_exists.1 (findColl_isSome H hlog htied)
    exact ⟨u, w, hout.trans huw, findColl_spec H _ hlog u w huw⟩
  have hd₀ : ∀ d₀ : (Fin (V.k x) → Coin) × A,
      (Pr fun e => (op (capFork V Cp x q d₀ e)).isSome ∧
        op (capFork V Cp x q d₀ e) ≠ some (capPlur V x (Cp.round x q) op d₀)) ≤
        (Ex fun e => Pr fun vs : Fin M → (Fin (V.k x) → Coin) × A =>
          ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, capFork V Cp x q d₀ e, vs) = some (u, v) ∧
            Collision H u v) + 1 / (Real.exp 1 * M) := by
    intro d₀
    refine (off_le_succ (fun e => op (capFork V Cp x q d₀ e)) M hM).trans (add_le_add (Ex_mono fun e => ?_) le_rfl)
    rw [Claims.Bridge.Pr_eq_Ex]
    exact Ex_mono fun vs => ind_le_of_imp (hstep d₀ e vs)
  have hL : (Pr fun d => (op d).isSome ∧ op d ≠ some (capPlur V x (Cp.round x q) op d)) =
      Ex fun d₀ => Pr fun e => (op (capFork V Cp x q d₀ e)).isSome ∧
        op (capFork V Cp x q d₀ e) ≠ some (capPlur V x (Cp.round x q) op d₀) := by
    rw [Claims.Bridge.Pr_eq_Ex, ← Ex_capFork V Cp x q, Claims.Bridge.Ex_prod]
    refine congrArg Ex (funext fun d₀ => ?_)
    rw [Claims.Bridge.Pr_eq_Ex]
    refine congrArg Ex (funext fun e => ?_)
    have hc : capPlur V x (Cp.round x q) op (capFork V Cp x q d₀ e) = capPlur V x (Cp.round x q) op d₀ :=
      capPlur_prefix V x _ op fun i hi => (fork_agree (Cp.round x q) d₀.1 e.1 i hi).symm
    simp only [ind, hc]
    split_ifs <;> rfl
  have hR : (Pr fun s : ((Fin (V.k x) → Coin) × A) × (Fin M → (Fin (V.k x) → Coin) × A) =>
        ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, s.1, s.2) = some (u, v) ∧ Collision H u v) =
      Ex fun d₀ => Ex fun e => Pr fun vs : Fin M → (Fin (V.k x) → Coin) × A =>
        ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, capFork V Cp x q d₀ e, vs) = some (u, v) ∧
          Collision H u v := by
    rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod, ← Ex_capFork V Cp x q, Claims.Bridge.Ex_prod]
    exact congrArg Ex (funext fun d₀ => congrArg Ex (funext fun e => (Claims.Bridge.Pr_eq_Ex _).symm))
  have hM' : (M : ℝ) + 1 ≤ Real.exp 1 * M := by
    have hR1 : (1 : ℝ) ≤ M := by exact_mod_cast hM
    have := Real.add_one_le_exp (1 : ℝ)
    nlinarith
  have he : 1 / (Real.exp 1 * M) ≤ 1 / ((M : ℝ) + 1) := one_div_le_one_div_of_le (by positivity) hM'
  rw [hL, hR]
  calc (Ex fun d₀ => Pr fun e => (op (capFork V Cp x q d₀ e)).isSome ∧
          op (capFork V Cp x q d₀ e) ≠ some (capPlur V x (Cp.round x q) op d₀))
      ≤ Ex fun d₀ => (Ex fun e => Pr fun vs : Fin M → (Fin (V.k x) → Coin) × A =>
          ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, capFork V Cp x q d₀ e, vs) = some (u, v) ∧
            Collision H u v) + 1 / (Real.exp 1 * M) := Ex_mono hd₀
    _ = (Ex fun d₀ => Ex fun e => Pr fun vs : Fin M → (Fin (V.k x) → Coin) × A =>
          ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, capFork V Cp x q d₀ e, vs) = some (u, v) ∧
            Collision H u v) + 1 / (Real.exp 1 * M) := by rw [Ex_add, Ex_const]
    _ ≤ _ := add_le_add le_rfl he

end CapLink

end Claims.Recursion
