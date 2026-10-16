import Proofs.Flock.Claims.Recursion.Compose
import Proofs.Flock.Claims.Aggregate

/-!
# The recursive audit's fork form, in the claims' error terms

`Flock.Guarantees.RecursiveAuditC3At` concludes in fork form: the accepted false statements have probability `0`, or one
of the prover's own runs finds a SHA-512 collision, or a bound holds. `KS` and `KSLin` bound the bad event on every
prover by `κ + √(γ·Pr[B]) + β·Pr[Fe]`. Each kind of disjunct lands in one of those terms:

* the bad event is inside acceptance (`PrBad_le_acc`), so acceptance at probability `0` contributes nothing;
* a collision among the verifier's own SHA-512 calls in one run is one `B` outputs (`Pr_VColl_le_PrB`), so it folds into
  `B`'s term at `(√γ + 1)²` (`KS_of_parts`, `KSLin_of_parts`);
* a collision read off records the verifier never hashes in the clear (a registrant's records, a read's openings, held
  salts, a registered value) is read off the extractor's output: one run, the extractor on it, then the pair
  (`oneRunExt`), a finder of `Fe`, `Pr[Fe]` at most its calls over `2^256` under `ExpectedCR` (`oneRunExt_Pr_le`);
* a collision between two runs forked at a round, each read through the extractor (the inner rounds' forks), is
  `forkExt`'s: `B`'s two runs, `E` on each, then the pair (`forkExt_Pr_le`).
-/

namespace Claims

/-- `findColl` finds a collision whenever its list of evaluations of `H` has one. -/
theorem findColl_isSome (H : Msg → Msg) :
    ∀ (l : List (Msg × Msg)), (∀ p ∈ l, H p.1 = p.2) → ∀ u v, u ∈ l.map Prod.fst → v ∈ l.map Prod.fst →
      Collision H u v → (findColl l).isSome := by
  intro l
  induction l with
  | nil => intro _ u _ hu; simp at hu
  | cons hd tl ih =>
    obtain ⟨q, a⟩ := hd
    intro hl u v hu hv hc
    have hq : H q = a := hl (q, a) (by simp)
    have htl : ∀ p ∈ tl, H p.1 = p.2 := fun p hp => hl p (by simp [hp])
    have hfind : ∀ p ∈ tl, p.1 ≠ q → H p.1 = H q →
        (tl.find? fun p => decide (p.1 ≠ q) && decide (p.2 = a)).isSome := by
      intro p hp hne hH
      refine List.find?_isSome.2 ⟨p, hp, ?_⟩
      simp only [Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨hne, by rw [← htl p hp, hH, hq]⟩
    simp only [findColl]
    split
    · rfl
    · rename_i hnone
      simp only [List.map_cons, List.mem_cons] at hu hv
      rcases hu with rfl | hu
      · rcases hv with rfl | hv
        · exact absurd rfl hc.1
        · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hv
          have h := hfind p hp (fun e => hc.1 e.symm) hc.2.symm
          rw [hnone] at h
          exact absurd h (by simp)
      · rcases hv with rfl | hv
        · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hu
          have h := hfind p hp hc.1 hc.2
          rw [hnone] at h
          exact absurd h (by simp)
        · exact ih htl u v hu hv hc

section Transport

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- A collision among the verifier's own SHA-512 calls in the run `t`. -/
def VColl (H : Msg → Msg) (t : Trace) : Prop :=
  ∃ u v, u ∈ t.vLog.map Prod.fst ∧ v ∈ t.vLog.map Prod.fst ∧ Collision H u v

theorem PrB_nonneg {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) : 0 ≤ PrB H V P x := by
  unfold PrB; split
  · exact Pr_nonneg _
  · exact le_rfl

theorem PrB_le_one {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) : PrB H V P x ≤ 1 := by
  unfold PrB; split
  · exact Pr_le_one _
  · exact zero_le_one

/-- **One run's collision is `B`'s**: `B`'s first run is the prover's at a uniform tape and coins, and `B` outputs a
collision whenever the verifier's calls in it have one. -/
theorem Pr_VColl_le_PrB {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) (hk : 0 < V.k x) :
    Pr (fun s : Ω × (Fin (V.k x) → Coin) => VColl H (run H P V x s.1 s.2)) ≤ PrB H V P x := by
  have : Nonempty (Fin (V.k x)) := ⟨⟨0, hk⟩⟩
  simp only [PrB, hk, ↓reduceDIte]
  rw [B_Pr_eq]
  calc Pr (fun s : Ω × (Fin (V.k x) → Coin) => VColl H (run H P V x s.1 s.2))
      = Pr (fun s : Fin (V.k x) × Ω × (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) =>
          VColl H (run H P V x s.2.1 s.2.2.1)) := by
        rw [Pr_snd (α := Fin (V.k x))
          (fun s : Ω × (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => VColl H (run H P V x s.1 s.2.1))]
        exact (Pr_drop (γ := Fin (V.k x) → Coin) fun ω r => VColl H (run H P V x ω r)).symm
    _ ≤ _ := by
        refine Pr_mono fun s ⟨u, v, hu, hv, hc⟩ => ?_
        refine findColl_isSome H _ (fun p hp => ?_) u v (by simp [hu]) (by simp [hv]) hc
        rcases List.mem_append.1 hp with hp | hp
        · exact (run_logs H P V x _ _).2 p hp
        · exact (run_logs H P V x _ _).2 p hp

omit [Nonempty Coin] in
/-- **The bad event is inside acceptance**: `Pr[run = acc ∧ ¬R(x, E(x))] ≤ Pr[run = acc]`. -/
theorem PrBad_le_acc {W : Type} (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V) (x : Msg)
    {Ω : Type} [Fintype Ω] (P : Prover Ω) :
    PrBad H V R E x P ≤ Pr fun s : Ω × (Fin (V.k x) → Coin) => (run H P V x s.1 s.2).out = .acc := by
  unfold PrBad
  rw [← Pr_drop (γ := E.C x) fun ω r => (run H P V x ω r).out = .acc]
  exact Pr_mono fun s h => h.1

/-- **The parts of a fork-form bound, assembled, with a linear term**: on every prover the bad event is at most `κ`,
`B`'s term at `γ`, a collision among the verifier's calls in the judged run, and `β·Pr[Fe]`; then `KSLin` holds at
`(√γ + 1)²`. -/
theorem KSLin_of_parts {W : Type} (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V)
    {κ γ β : Msg → ℝ} (Fe : FinderFamily) (hγ : ∀ x, 0 ≤ γ x) (hk : ∀ x, 0 < V.k x)
    (h : ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      PrBad H V R E x P ≤ κ x + Real.sqrt (γ x * PrB H V P x) +
        Pr (fun s : Ω × (Fin (V.k x) → Coin) => VColl H (run H P V x s.1 s.2)) + β x * (Fe Ω P x).Pr H) :
    KSLin H V R E κ (fun x => (Real.sqrt (γ x) + 1) ^ 2) β Fe := by
  intro x Ω _ _ P
  have h₁ := h x Ω P
  have h₂ := Pr_VColl_le_PrB H V P x (hk x)
  have h₃ := sqrt_add_lin (hγ x) zero_le_one (PrB_nonneg H V P x) (PrB_le_one H V P x)
  rw [one_mul] at h₃
  linarith

/-- **The parts of a fork-form bound, assembled**: on every prover the bad event is at most `κ`, `B`'s term at `γ` and
a collision among the verifier's calls in the judged run; then `KS` holds at `(√γ + 1)²`. -/
theorem KS_of_parts {W : Type} (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V)
    {κ γ : Msg → ℝ} (hγ : ∀ x, 0 ≤ γ x) (hk : ∀ x, 0 < V.k x)
    (h : ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      PrBad H V R E x P ≤ κ x + Real.sqrt (γ x * PrB H V P x) +
        Pr (fun s : Ω × (Fin (V.k x) → Coin) => VColl H (run H P V x s.1 s.2))) :
    KS H V R E κ fun x => (Real.sqrt (γ x) + 1) ^ 2 := by
  intro x Ω _ _ P
  have h₁ := h x Ω P
  have h₂ := Pr_VColl_le_PrB H V P x (hk x)
  have h₃ := sqrt_add_lin (hγ x) zero_le_one (PrB_nonneg H V P x) (PrB_le_one H V P x)
  rw [one_mul] at h₃
  linarith

/-! ## Collisions read off the extractor's output -/

/-- **One run, the extractor, then a pair**: run the prover once at its tape and coins, run `E` on that run, and compute
a pair from the run and `E`'s output by an oracle program `pick`. Its cost is the run's SHA-512 calls (both parties'),
`E`'s, and `pick`'s. -/
def oneRunExt {W : Type} (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V)
    (pick : Msg → Trace → Option W → OC (Option (Msg × Msg))) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) : Finder where
  S := Ω × (Fin (V.k x) → Coin) × E.C x
  out s := ((pick x (run H P V x s.1 s.2.1) (E.eval H P x s.1 s.2.1 s.2.2).1).eval H).1
  cost s := (run H P V x s.1 s.2.1).pLog.length + (run H P V x s.1 s.2.1).vLog.length +
    (E.eval H P x s.1 s.2.1 s.2.2).2.1 + ((pick x (run H P V x s.1 s.2.1) (E.eval H P x s.1 s.2.1 s.2.2).1).eval H).2.length

/-- `oneRunExt` at every prover and statement. -/
def oneRunExtFamily {W : Type} (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V)
    (pick : Msg → Trace → Option W → OC (Option (Msg × Msg))) : FinderFamily :=
  fun _ _ _ P x => oneRunExt H V E pick P x

/-- An event on which `pick` outputs a collision has probability at most `oneRunExt`'s. -/
theorem Pr_le_oneRunExt {W : Type} (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V)
    (pick : Msg → Trace → Option W → OC (Option (Msg × Msg))) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (Ev : Ω × (Fin (V.k x) → Coin) × E.C x → Prop)
    (hEv : ∀ s, Ev s → ∃ u v,
      ((pick x (run H P V x s.1 s.2.1) (E.eval H P x s.1 s.2.1 s.2.2).1).eval H).1 = some (u, v) ∧ Collision H u v) :
    Pr Ev ≤ (oneRunExt H V E pick P x).Pr H :=
  Pr_mono hEv

/-- **`oneRunExt`'s cost**: a run's calls `T`, `E`'s `N_h` and `pick`'s `c`, each at most. -/
theorem oneRunExt_cost_le {W : Type} (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V)
    (pick : Msg → Trace → Option W → OC (Option (Msg × Msg))) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) {T Nh c : ℕ} (hT : ∀ ω r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ T)
    (hE : ∀ ω r₀ e, (E.eval H P x ω r₀ e).2.1 ≤ Nh) (hc : ∀ t w, ((pick x t w).eval H).2.length ≤ c)
    (s : (oneRunExt H V E pick P x).S) : (oneRunExt H V E pick P x).cost s ≤ T + Nh + c := by
  have h₁ := hT s.1 s.2.1
  have h₂ := hE s.1 s.2.1 s.2.2
  have h₃ := hc (run H P V x s.1 s.2.1) (E.eval H P x s.1 s.2.1 s.2.2).1
  simp only [oneRunExt]
  omega

/-- **`oneRunExt` under `ExpectedCR`**: it finds a collision with probability at most `(T + N_h + c)/2^256`. -/
theorem oneRunExt_Pr_le {W : Type} (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V)
    (pick : Msg → Trace → Option W → OC (Option (Msg × Msg))) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) {T Nh c : ℕ} (hT : ∀ ω r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ T)
    (hE : ∀ ω r₀ e, (E.eval H P x ω r₀ e).2.1 ≤ Nh) (hc : ∀ t w, ((pick x t w).eval H).2.length ≤ c)
    (hCR : ExpectedCR H (oneRunExt H V E pick P x)) :
    (oneRunExt H V E pick P x).Pr H ≤ ((T + Nh + c : ℕ) : ℝ) / 2 ^ (256 : ℝ) := by
  refine hCR.trans (div_le_div_of_nonneg_right ?_ (by positivity))
  calc Ex (fun s => ((oneRunExt H V E pick P x).cost s : ℝ))
      ≤ Ex (fun _ : (oneRunExt H V E pick P x).S => ((T + Nh + c : ℕ) : ℝ)) :=
        Ex_mono fun s => by exact_mod_cast oneRunExt_cost_le H V E pick P x hT hE hc s
    _ = _ := Ex_const _

/-- **`B`'s fork, the extractor on both runs, then a pair**: guess a round `j`, run the prover twice from a shared start
with fresh continuations after round `j` (`B`'s runs), run `E` on each, and compute a pair from both runs and both
outputs by `pick`. Its cost is both runs' SHA-512 calls, both of `E`'s, and `pick`'s. -/
def forkExt {W : Type} (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V)
    (pick : Msg → Trace → Option W → Trace → Option W → OC (Option (Msg × Msg))) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) : Finder where
  S := BCoins Ω Coin (V.k x) × E.C x × E.C x
  ne := ⟨(⟨⟨0, hk⟩, Classical.arbitrary _, fun _ => default, fun _ => default⟩, Classical.arbitrary _,
    Classical.arbitrary _)⟩
  out s :=
    let r₂ := forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2
    ((pick x (run H P V x s.1.2.1 s.1.2.2.1) (E.eval H P x s.1.2.1 s.1.2.2.1 s.2.1).1 (run H P V x s.1.2.1 r₂)
      (E.eval H P x s.1.2.1 r₂ s.2.2).1).eval H).1
  cost s :=
    let r₂ := forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2
    let t₁ := run H P V x s.1.2.1 s.1.2.2.1
    let t₂ := run H P V x s.1.2.1 r₂
    t₁.pLog.length + t₁.vLog.length + t₂.pLog.length + t₂.vLog.length +
      (E.eval H P x s.1.2.1 s.1.2.2.1 s.2.1).2.1 + (E.eval H P x s.1.2.1 r₂ s.2.2).2.1 +
      ((pick x t₁ (E.eval H P x s.1.2.1 s.1.2.2.1 s.2.1).1 t₂ (E.eval H P x s.1.2.1 r₂ s.2.2).1).eval H).2.length

omit [Nonempty Coin] in
/-- An event on which `pick` outputs a collision has probability at most `forkExt`'s. -/
theorem Pr_le_forkExt {W : Type} (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V)
    (pick : Msg → Trace → Option W → Trace → Option W → OC (Option (Msg × Msg))) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) (Ev : (forkExt H V E pick P x hk).S → Prop)
    (hEv : ∀ s, Ev s → ∃ u v, (forkExt H V E pick P x hk).out s = some (u, v) ∧ Collision H u v) :
    Pr Ev ≤ (forkExt H V E pick P x hk).Pr H :=
  Pr_mono hEv

omit [Nonempty Coin] in
/-- **`forkExt`'s cost**: two runs' calls `T` each, two of `E`'s `N_h` each, and `pick`'s `c`. -/
theorem forkExt_cost_le {W : Type} (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V)
    (pick : Msg → Trace → Option W → Trace → Option W → OC (Option (Msg × Msg))) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) {T Nh c : ℕ}
    (hT : ∀ ω r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ T)
    (hE : ∀ ω r₀ e, (E.eval H P x ω r₀ e).2.1 ≤ Nh) (hc : ∀ t w t' w', ((pick x t w t' w').eval H).2.length ≤ c)
    (s : (forkExt H V E pick P x hk).S) : (forkExt H V E pick P x hk).cost s ≤ 2 * T + 2 * Nh + c := by
  have h₁ := hT s.1.2.1 s.1.2.2.1
  have h₂ := hT s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2)
  have h₃ := hE s.1.2.1 s.1.2.2.1 s.2.1
  have h₄ := hE s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2
  have h₅ := hc (run H P V x s.1.2.1 s.1.2.2.1) (E.eval H P x s.1.2.1 s.1.2.2.1 s.2.1).1
    (run H P V x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2))
    (E.eval H P x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2).1
  simp only [forkExt]
  omega

omit [Nonempty Coin] in
/-- **`forkExt` under `ExpectedCR`**: it finds a collision with probability at most `(2T + 2N_h + c)/2^256`. -/
theorem forkExt_Pr_le {W : Type} (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V)
    (pick : Msg → Trace → Option W → Trace → Option W → OC (Option (Msg × Msg))) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) {T Nh c : ℕ}
    (hT : ∀ ω r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ T)
    (hE : ∀ ω r₀ e, (E.eval H P x ω r₀ e).2.1 ≤ Nh) (hc : ∀ t w t' w', ((pick x t w t' w').eval H).2.length ≤ c)
    (hCR : ExpectedCR H (forkExt H V E pick P x hk)) :
    (forkExt H V E pick P x hk).Pr H ≤ ((2 * T + 2 * Nh + c : ℕ) : ℝ) / 2 ^ (256 : ℝ) := by
  refine hCR.trans (div_le_div_of_nonneg_right ?_ (by positivity))
  calc Ex (fun s => ((forkExt H V E pick P x hk).cost s : ℝ))
      ≤ Ex (fun _ : (forkExt H V E pick P x hk).S => ((2 * T + 2 * Nh + c : ℕ) : ℝ)) :=
        Ex_mono fun s => by exact_mod_cast forkExt_cost_le H V E pick P x hk hT hE hc s
    _ = _ := Ex_const _

end Transport

end Claims
