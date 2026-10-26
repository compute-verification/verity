import Proofs.Flock.Claims.Recursion.CapLink

/-!
# A cap position's off mass at the abort law, at read sets

`Pr_off_le_capHit` charges an opening that a run makes at one value per position. A deployed run can open a position
twice, to different rows, and then no single value per run is the model's opening there. Here the run reads a set of
values at the position (`rd`), and the link finder's two-reading term (`LinkFinder.sRd`) takes the runs that read two:

* **The plurality single reading after the round** (`capPlurRd`): the value that runs forked at the round from `d₀`
  read alone most often. It reads only `d₀`'s coins before the round (`capPlurRd_prefix`).
* **One base run's forks** (`offRd_le_succ`): `LinkFinder.succ_ge` at `drawGame`: forks that read anything but their
  plurality alone, at most as often as the link finder over them succeeds, plus `1/(eM)`.
* **The link** (`Pr_rdOff_le_capHit`): where a run reads something exactly where it accepts and queries `q`, and any
  two different values that two runs sharing the coins before `q`'s round read (one run twice included) tie a
  collision among their calls, a run reads anything but the plurality after `q`'s round alone with probability at most
  `FC`'s hit at `q` plus `1/(M+1)`.
-/

namespace Claims.Recursion

open Claims Claims.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Rewinding

section CapLinkRd

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- The value `f` reads alone most often over uniform draws. -/
noncomputable def plurRd {Dr Val : Type} [Fintype Dr] [Fintype Val] [Nonempty Val] (f : Dr → Finset Val) : Val :=
  (Finset.exists_max_image Finset.univ (fun v => Pr fun e => f e = {v}) Finset.univ_nonempty).choose

theorem plurRd_max {Dr Val : Type} [Fintype Dr] [Fintype Val] [Nonempty Val] (f : Dr → Finset Val) (v : Val) :
    (Pr fun e => f e = {v}) ≤ Pr fun e => f e = {plurRd f} :=
  (Finset.exists_max_image Finset.univ (fun v => Pr fun e => f e = {v}) Finset.univ_nonempty).choose_spec.2 v
    (Finset.mem_univ v)

/-- **The plurality single reading after round `j`**: the value that runs forked at `j` from `d₀`, with fresh abort
coins, read alone most often. -/
noncomputable def capPlurRd (V : Verifier Coin) {A Val : Type} [Fintype A] [Fintype Val] [Nonempty Val] (x : Msg)
    (j : ℕ) (rd : (Fin (V.k x) → Coin) × A → Finset Val) (d₀ : (Fin (V.k x) → Coin) × A) : Val :=
  plurRd fun e : (Fin (V.k x) → Coin) × A => rd (fork j d₀.1 e.1, e.2)

omit [Nonempty Coin] [Inhabited Coin] in
/-- The plurality single reading after round `j` reads only the coins before it. -/
theorem capPlurRd_prefix (V : Verifier Coin) {A Val : Type} [Fintype A] [Fintype Val] [Nonempty Val] (x : Msg)
    (j : ℕ) (rd : (Fin (V.k x) → Coin) × A → Finset Val) {d₀ d₀' : (Fin (V.k x) → Coin) × A}
    (h : ∀ i : Fin (V.k x), i.val < j → d₀.1 i = d₀'.1 i) : capPlurRd V x j rd d₀ = capPlurRd V x j rd d₀' := by
  have hf : fork j d₀.1 = fork j d₀'.1 := funext fun r => funext fun i => by
    by_cases hi : i.val < j
    · simp only [fork, hi, ↓reduceIte, h i hi]
    · simp only [fork, hi, ↓reduceIte]
  unfold capPlurRd
  rw [hf]

theorem tiedColl_dup {H : Msg → Msg} {l : List (Msg × Msg)} (h : TiedColl H (l ++ l)) (l' : List (Msg × Msg)) :
    TiedColl H (l ++ l') := by
  obtain ⟨u, v, hc, hu, hv⟩ := h
  simp only [List.map_append, List.mem_append, or_self] at hu hv
  exact TiedColl.append_right ⟨u, v, hc, hu, hv⟩ l'

/-- **One base run's forks at read sets, linearly**: draws read anything but their plurality single reading alone at
most as often as the link finder over them succeeds, plus `1/(eM)`. -/
theorem offRd_le_succ {Dr Val : Type} [Fintype Dr] [Nonempty Dr] [Fintype Val] [DecidableEq Val] [Nonempty Val]
    (f : Dr → Finset Val) (M : ℕ) (hM : 1 ≤ M) :
    (Pr fun e => (f e).Nonempty ∧ f e ≠ {plurRd f}) ≤
      (Ex fun e => Ex fun vs : Fin M → Dr => ind (LinkFinder.Succ f) (e, (List.finRange M).map vs)) +
        1 / (Real.exp 1 * M) := by
  have hX : ∀ v, LinkFinder.pRd (drawGame Dr) (drawS Dr) f v ≤
      LinkFinder.pRd (drawGame Dr) (drawS Dr) f (plurRd f) := fun v => by
    simp only [LinkFinder.pRd, expect_drawGame]
    have := plurRd_max f v
    rwa [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Pr_eq_Ex] at this
  have h := LinkFinder.succ_ge (drawGame Dr) (drawS Dr) f hX M hM
  rw [LinkFinder.expect_finderG] at h
  simp only [expect_drawGame, expectN_drawGame] at h
  have hr := LinkFinder.rRd_eq (drawGame Dr) (drawS Dr) f
  have hoff : (Pr fun e => (f e).Nonempty ∧ f e ≠ {plurRd f}) =
      LinkFinder.rRd (drawGame Dr) (drawS Dr) f - LinkFinder.pRd (drawGame Dr) (drawS Dr) f (plurRd f) := by
    unfold LinkFinder.rRd LinkFinder.pRd
    rw [← expect_sub, expect_drawGame, Claims.Bridge.Pr_eq_Ex]
    refine congrArg Ex (funext fun e => ?_)
    by_cases h2 : f e = {plurRd f}
    · simp [ind, h2]
    · by_cases h1 : (f e).Nonempty <;> simp [ind, h1, h2]
  rw [hoff]
  linarith

/-- **A cap position's off mass at read sets, linearly, as `FC`'s hits there**: for read sets of `q` that a run makes
nonempty exactly where it accepts and queries `q`, any two different values of which, read by runs sharing the coins
before `q`'s round (one run twice included), tie a collision among their calls, a run reads anything but the plurality
after `q`'s round alone with probability at most that of `FC`'s run and loop at `q` finding a collision, plus
`1/(M+1)`. -/
theorem Pr_rdOff_le_capHit (H : Msg → Msg) (V : Verifier Coin) (Cp : Caps Coin) {Ω A Val : Type} [Fintype Ω]
    [Nonempty Ω] [Fintype A] [Nonempty A] [Fintype Val] [DecidableEq Val] [Nonempty Val] (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (M : ℕ) (q : Fin (Cp.width x)) (ω : Ω)
    (rd : (Fin (V.k x) → Coin) × A → Finset Val)
    (hop : ∀ d, (rd d).Nonempty ↔ opensAt H V Cp P x keep q ω d)
    (htie : ∀ d d' u u', (∀ i : Fin (V.k x), i.val < Cp.round x q → d.1 i = d'.1 i) → u ∈ rd d → u' ∈ rd d' →
      u ≠ u' → TiedColl H ((abortRun H P V x keep ω d).vLog ++ (abortRun H P V x keep ω d').vLog)) :
    (Pr fun d => (rd d).Nonempty ∧ rd d ≠ {capPlurRd V x (Cp.round x q) rd d}) ≤
      (Pr fun s : ((Fin (V.k x) → Coin) × A) × (Fin M → (Fin (V.k x) → Coin) × A) =>
        ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, s.1, s.2) = some (u, v) ∧ Collision H u v) +
        1 / ((M : ℝ) + 1) := by
  classical
  rcases Nat.eq_zero_or_pos M with rfl | hM
  · refine (Pr_le_one _).trans ?_
    rw [Nat.cast_zero, zero_add, div_one]
    exact le_add_of_nonneg_left (Pr_nonneg _)
  have hhit : ∀ (b : (Fin (V.k x) → Coin) × A) (vs : Fin M → (Fin (V.k x) → Coin) × A),
      opensAt H V Cp P x keep q ω b →
      TiedColl H ((abortRun H P V x keep ω b).vLog ++ (abortRun H P V x keep ω (capFork V Cp x q b
        ((firstPick (fun v => opensAt H V Cp P x keep q ω (capFork V Cp x q b v)) M vs).getD b))).vLog) →
      ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, b, vs) = some (u, v) ∧ Collision H u v := by
    intro b vs hopen htied
    have hout : (FC H V Cp P x keep M q.pos).out (q, ω, b, vs) =
        findColl ((abortRun H P V x keep ω b).vLog ++ (abortRun H P V x keep ω (capFork V Cp x q b
          ((firstPick (fun v => opensAt H V Cp P x keep q ω (capFork V Cp x q b v)) M vs).getD b))).vLog) := by
      show (if opensAt H V Cp P x keep q ω b then _ else none) = _
      rw [ite_eq_left hopen]
    have hlog : ∀ p ∈ (abortRun H P V x keep ω b).vLog ++ (abortRun H P V x keep ω (capFork V Cp x q b
        ((firstPick (fun v => opensAt H V Cp P x keep q ω (capFork V Cp x q b v)) M vs).getD b))).vLog,
        H p.1 = p.2 := fun p hp => by
      rcases List.mem_append.1 hp with hp | hp
      · exact abortRun_vLog H P V x keep ω _ p hp
      · exact abortRun_vLog H P V x keep ω _ p hp
    obtain ⟨⟨u, w⟩, huw⟩ := Option.isSome_iff_exists.1 (findColl_isSome H hlog htied)
    exact ⟨u, w, hout.trans huw, findColl_spec H _ hlog u w huw⟩
  have hstep : ∀ (d₀ e : (Fin (V.k x) → Coin) × A) (vs : Fin M → (Fin (V.k x) → Coin) × A),
      LinkFinder.Succ (fun e => rd (capFork V Cp x q d₀ e)) (e, (List.finRange M).map vs) →
        ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, capFork V Cp x q d₀ e, vs) = some (u, v) ∧
          Collision H u v := by
    intro d₀ e vs hs
    have hcf : ∀ w, capFork V Cp x q (capFork V Cp x q d₀ e) w = capFork V Cp x q d₀ w := fun w => by
      simp only [capFork, fork_fork]
    simp only [LinkFinder.Succ] at hs
    rcases hs with h2 | ⟨v, hv, e', hn, hne⟩
    · obtain ⟨u, hu, u', hu', huu⟩ := Finset.one_lt_card.1 h2
      exact hhit _ vs ((hop _).1 ⟨u, hu⟩) (tiedColl_dup (htie _ _ u u' (fun _ _ => rfl) hu hu' huu) _)
    · have hsome : (rd (capFork V Cp x q d₀ e')).Nonempty := by
        unfold LinkFinder.nextRead at hn
        simpa using List.find?_some hn
      obtain ⟨u', hu', hne'⟩ : ∃ u' ∈ rd (capFork V Cp x q d₀ e'), u' ≠ v := by
        by_contra hall
        simp only [not_exists, not_and, not_not] at hall
        obtain ⟨w, hw⟩ := hsome
        exact hne (Finset.eq_singleton_iff_unique_mem.2 ⟨by rw [← hall w hw]; exact hw, hall⟩)
      have hfp : firstPick (fun w => opensAt H V Cp P x keep q ω (capFork V Cp x q d₀ w)) M vs = some e' := by
        rw [← find?_finRange_map, ← hn]
        unfold LinkFinder.nextRead
        congr 1
        funext e₁
        exact decide_eq_decide.2 (hop _).symm
      refine hhit _ vs ((hop _).1 (by rw [hv]; exact Finset.singleton_nonempty v)) ?_
      simp only [hcf, hfp, Option.getD_some]
      exact htie _ _ v u'
        (fun i hi => (fork_agree (Cp.round x q) d₀.1 e.1 i hi).symm.trans (fork_agree (Cp.round x q) d₀.1 e'.1 i hi))
        (by rw [hv]; exact Finset.mem_singleton_self v) hu' (Ne.symm hne')
  have hd₀ : ∀ d₀ : (Fin (V.k x) → Coin) × A,
      (Pr fun e => (rd (capFork V Cp x q d₀ e)).Nonempty ∧
        rd (capFork V Cp x q d₀ e) ≠ {capPlurRd V x (Cp.round x q) rd d₀}) ≤
        (Ex fun e => Pr fun vs : Fin M → (Fin (V.k x) → Coin) × A =>
          ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, capFork V Cp x q d₀ e, vs) = some (u, v) ∧
            Collision H u v) + 1 / (Real.exp 1 * M) := by
    intro d₀
    refine (offRd_le_succ (fun e => rd (capFork V Cp x q d₀ e)) M hM).trans
      (add_le_add (Ex_mono fun e => ?_) le_rfl)
    rw [Claims.Bridge.Pr_eq_Ex]
    exact Ex_mono fun vs => ind_le_of_imp (hstep d₀ e vs)
  have hL : (Pr fun d => (rd d).Nonempty ∧ rd d ≠ {capPlurRd V x (Cp.round x q) rd d}) =
      Ex fun d₀ => Pr fun e => (rd (capFork V Cp x q d₀ e)).Nonempty ∧
        rd (capFork V Cp x q d₀ e) ≠ {capPlurRd V x (Cp.round x q) rd d₀} := by
    rw [Claims.Bridge.Pr_eq_Ex, ← Ex_capFork V Cp x q, Claims.Bridge.Ex_prod]
    refine congrArg Ex (funext fun d₀ => ?_)
    rw [Claims.Bridge.Pr_eq_Ex]
    refine congrArg Ex (funext fun e => ?_)
    have hc : capPlurRd V x (Cp.round x q) rd (capFork V Cp x q d₀ e) = capPlurRd V x (Cp.round x q) rd d₀ :=
      capPlurRd_prefix V x _ rd fun i hi => (fork_agree (Cp.round x q) d₀.1 e.1 i hi).symm
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
  calc (Ex fun d₀ => Pr fun e => (rd (capFork V Cp x q d₀ e)).Nonempty ∧
          rd (capFork V Cp x q d₀ e) ≠ {capPlurRd V x (Cp.round x q) rd d₀})
      ≤ Ex fun d₀ => (Ex fun e => Pr fun vs : Fin M → (Fin (V.k x) → Coin) × A =>
          ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, capFork V Cp x q d₀ e, vs) = some (u, v) ∧
            Collision H u v) + 1 / (Real.exp 1 * M) := Ex_mono hd₀
    _ = (Ex fun d₀ => Ex fun e => Pr fun vs : Fin M → (Fin (V.k x) → Coin) × A =>
          ∃ u v, (FC H V Cp P x keep M q.pos).out (q, ω, capFork V Cp x q d₀ e, vs) = some (u, v) ∧
            Collision H u v) + 1 / (Real.exp 1 * M) := by rw [Ex_add, Ex_const]
    _ ≤ _ := add_le_add le_rfl he

end CapLinkRd

end Claims.Recursion
