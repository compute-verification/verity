module

public import Definitions.Slowdown.Security
public import Proofs.Slowdown.FitRate
public import Proofs.ResourceAccounting.Actual.Outside
public import Proofs.Flock.Claims.Prob

/-!
# No computation is feasible, accounted for and prohibited

`secure`: `M ∈ H F → M ∈ A F x_s x_b x_w → M ∉ B F p Y`, the three sets of `Definitions.Slowdown.Security`, when `Y`
is at least the fit bound (`Frame.bound`) and the bits a covered window can read of a reference training algorithm
(`Frame.unaccounted`) are fewer than `N* b_w`. Its steps:

* What the limits leave. In a feasible computation whose declared circuit holds `x_s` bits on a server at a cut, the
  server carries at most `m − x_s` bits outside the declared circuit across it (`state_outside`); one whose window reads
  `x_b` bits of the declared circuit from outside its server reads at most `b ω − x_b` others from there
  (`ingress_outside`); and one whose declared circuit does `x_w` work in a window does at most `Φ ω − x_w` outside it
  there (`work_outside`).
* What a covered window reads of the reference training algorithm's novel gates: what its server carries across the
  cut that opens it plus what crosses the server's boundary, each split into the part outside the declared circuit,
  which the limits leave, and the part the algorithm shares with it (`window_reads_le`).
* The count. `fitRate_of_reads_but` bounds the counted multiplications by the fit bound over the covered windows plus
  those placed in the missed windows, which number at most the work the limits leave there plus those shared with the
  declared circuit (`card_missed_le`).

`secure_pr` is the claim over the audits' coins: where every computation is feasible and the audits are sound with
error `δ` (`Sound`), they accept a prohibited computation with probability at most `δ`. `single_m`, `single_b` and
`single_Φ` are the limits of a frame of one server, and `server_m`, `server_b` and `server_Φ` those of a server of
`n` GPUs with its host and switch.
-/

@[expose] public section

open InferenceOnly InferenceOnly.SecurityProofs

namespace Slowdown.SecurityProofs

open Verity ResourceAccounting ResourceAccounting.SecurityProofs.Actual Finset Slowdown.Security

/-- **What a window reads in `G`, beside what is accounted for.** Outside `S`, `X` carries at most `r_s` bits across
`a` and `A[X, [a, b))` reads at most `r_b` bits from outside `X`; of `S`, at most `o_s` bits carried and `o_b` bits read
are in `G`. Then in the developer's declaration the window reads at most `(r_s + o_s) + (r_b + o_b)` bits of `G`. -/
theorem window_reads_le {L : Type*} [DecidableEq L] (A : Placed L) {ℓ : L → L → ℕ} (hC : A.Causal ℓ)
    {X Y : Finset L} (hXY : X ⊆ Y) {θ : Finset (Fin A.N)} (hθ : ∀ a ∈ θ, A.isInput a = true)
    {t a b : ℕ} (hb : b ≤ t) (S G : Finset (Fin A.N)) {r_s r_b o_s o_b : ℕ}
    (hrs : A.stateIn X a Sᶜ ≤ r_s)
    (hrb : A.width (((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ Sᶜ) ≤ r_b)
    (hos : A.stateIn X a (S ∩ G) ≤ o_s)
    (hob : A.width (((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ (S ∩ G)) ≤ o_b) :
    readsIn (A.declaration Y t θ hθ) (A.region X (Set.Ico a b)) G ≤ (r_s + o_s) + (r_b + o_b) := by
  have hR : A.region X (Set.Ico a b) ⊆ A.region Y (Set.Iio t) :=
    region_subset A hXY fun s hs => Set.mem_Iio.2 (hs.2.trans_le hb)
  have h1 := width_into_region_inter_le_boundary A X a b G (forward_of_causal A hC _)
  have h2 : A.stateIn X a G ≤ r_s + o_s := (stateIn_le_split A X a S G).trans (Nat.add_le_add hrs hos)
  have h3 : A.width (((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ G) ≤ r_b + o_b :=
    (width_inter_le_split A _ S G).trans (Nat.add_le_add hrb hob)
  exact (readsIn_declaration A hθ hR G).le.trans (h1.trans (Nat.add_le_add h2 h3))

open Classical in
/-- **The counted multiplications placed in the windows of `I`**: in each, those outside `S` number at most the work
the compute rate leaves beside the `x_w` that `S` does there, `Φ ω − x_w` at a price of at least 1 a counted
multiplication, and those in `S` at most `o_w`. -/
theorem card_missed_le {L : Type*} [DecidableEq L] (A : Placed L) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t : ℕ) (κ : Cuts t) (ω : ℕ)
    {𝒳 : Set (Finset L)} (compute : Finset L → ℕ) (π : Fin A.N → ℕ) (Φ x_w o_w : ℕ) (S : Finset (Fin A.N))
    (c : RegisteredCircuit) (M : Containment c.D (A.declaration (univ.biUnion X) t θ hθ).D) (T : Threading c)
    (I : Finset (Fin (n * κ.J)))
    (hW : A.ComputeRate 𝒳 π compute) (hX : ∀ i, X i ∈ 𝒳)
    (hω : ∀ j : Fin κ.J, κ.τ j.succ - κ.τ j.castSucc ≤ ω) (hΦ : ∀ i, compute (X i) ≤ Φ)
    (hπ : ∀ g ∈ T.M, 1 ≤ π (M.emb g))
    (hx : ∀ (i : Fin n) (j : Fin κ.J), finProdFinEquiv (i, j) ∈ I →
      x_w ≤ A.work π (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) ∩ S))
    (ho : ∀ (i : Fin n) (j : Fin κ.J), finProdFinEquiv (i, j) ∈ I →
      (T.M.filter fun g => M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) ∩ S).card ≤ o_w) :
    (T.M.filter fun g => ∃ i j, finProdFinEquiv (i, j) ∈ I ∧
      M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))).card ≤ I.card * (Φ * ω - x_w + o_w) := by
  have hsub : (T.M.filter fun g => ∃ i j, finProdFinEquiv (i, j) ∈ I ∧
      M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))) ⊆
      I.biUnion fun u => T.M.filter fun g => M.emb g ∈ A.region (X (finProdFinEquiv.symm u).1)
        (Set.Ico (κ.τ (finProdFinEquiv.symm u).2.castSucc) (κ.τ (finProdFinEquiv.symm u).2.succ)) := by
    intro g hg
    obtain ⟨hgs, i, j, hu, hgr⟩ := mem_filter.1 hg
    refine mem_biUnion.2 ⟨_, hu, mem_filter.2 ⟨hgs, ?_⟩⟩
    rw [Equiv.symm_apply_apply]
    exact hgr
  have hwin : ∀ (i : Fin n) (j : Fin κ.J), finProdFinEquiv (i, j) ∈ I →
      (T.M.filter fun g => M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))).card ≤
        Φ * ω - x_w + o_w := by
    intro i j hu
    have hout : (T.M.filter fun g => M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) \ S).card ≤
        Φ * ω - x_w := by
      have h1 : (T.M.filter fun g => M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) \ S).card ≤
          A.work π (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) \ S) := by
        convert card_filter_le_work A π _ T.M ⟨M.emb, M.emb_inj⟩ hπ
      exact h1.trans ((work_outside A hW (hX i) _ _ S (hx i j hu)).trans
        (Nat.sub_le_sub_right (Nat.mul_le_mul (hΦ i) (hω j)) _))
    refine (card_le_card fun g hg => ?_).trans ((card_union_le _ _).trans (Nat.add_le_add hout (ho i j hu)))
    obtain ⟨hgs, hgr⟩ := mem_filter.1 hg
    by_cases hS : M.emb g ∈ S
    · exact mem_union_right _ (mem_filter.2 ⟨hgs, mem_inter.2 ⟨hgr, hS⟩⟩)
    · exact mem_union_left _ (mem_filter.2 ⟨hgs, mem_sdiff.2 ⟨hgr, hS⟩⟩)
  refine (card_le_card hsub).trans (card_biUnion_le.trans ?_)
  calc _ ≤ ∑ _u ∈ I, (Φ * ω - x_w + o_w) := sum_le_sum fun u hu =>
        hwin _ _ (by rwa [Prod.mk.eta, Equiv.apply_symm_apply])
    _ = I.card * (Φ * ω - x_w + o_w) := by rw [sum_const, smul_eq_mul]

/-- **No computation is feasible, accounted for and prohibited.** At a frame `F`, amounts `x_s`, `x_b` and `x_w`, and
a recipe `p` with `b_w ≥ 0`, `w_e > 0` and the bits a covered window can read of a reference training algorithm
(`F.unaccounted x_s x_b p`) fewer than `N* b_w`: a computation that is feasible on the frame's hardware (`H F`) and
consistent with what the audits account for (`A F x_s x_b x_w`) holds no reference training algorithm of more than `Y`
counted multiplications that meets the recipe's premises (`B F p Y`), for every `Y` at least the fit bound
(`F.bound x_s x_b x_w p`). -/
theorem secure {L E : Type*} [Fintype E] [DecidableEq L] (F : Frame L E) (x_s x_b x_w : ℕ) (p : Recipe) (Y : ℝ)
    (hbw : 0 ≤ p.b_w) (hwe : 0 < p.w_e) (hgap : (F.unaccounted x_s x_b p : ℝ) < p.Nstar * p.b_w)
    (hY : F.bound x_s x_b x_w p ≤ Y) (M : Computation L E) :
    M ∈ H F → M ∈ A F x_s x_b x_w → M ∉ B F p Y := by
  rintro ⟨hC, hW, -, hMem, hL⟩ hA ⟨c, emb, T, start, hT, hlt⟩
  have hω : ∀ j : Fin F.cuts.J, F.cuts.τ j.succ - F.cuts.τ j.castSucc ≤ F.ω := fun j =>
    le_sup (f := fun j : Fin F.cuts.J => F.cuts.τ j.succ - F.cuts.τ j.castSucc) (mem_univ j)
  have hm : ∀ i, F.hw.M (F.server i) ≤ F.m := fun i => le_sup (f := fun i => F.hw.M (F.server i)) (mem_univ i)
  have hb : ∀ i, linkCapacity F.hw.ends F.hw.β (F.server i) ≤ F.b := fun i =>
    le_sup (f := fun i => linkCapacity F.hw.ends F.hw.β (F.server i)) (mem_univ i)
  have hΦ : ∀ i, F.hw.compute (F.server i) ≤ F.Φ := fun i =>
    le_sup (f := fun i => F.hw.compute (F.server i)) (mem_univ i)
  have hS : ∀ (i : Fin F.n) (j : Fin F.cuts.J), finProdFinEquiv (i, j) ∉ F.missed →
      (readsIn (M.run.declaration (univ.biUnion F.server) F.t M.registered M.registered_input)
        (M.run.region (F.server i) (Set.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ)))
        ((c.θᶜ).map ⟨emb.emb, emb.emb_inj⟩) : ℝ) ≤ ((F.unaccounted x_s x_b p : ℕ) : ℝ) := by
    intro i j hu
    obtain ⟨hxs, hxb⟩ := (hA i j).1 hu
    have hτ : F.cuts.τ j.succ ≤ F.t := (F.cuts.mono (Fin.le_last _)).trans F.cuts.last.le
    have hrs : M.run.stateIn (F.server i) (F.cuts.τ j.castSucc) M.declaredᶜ ≤ F.m - x_s :=
      (state_outside M.run (T := Set.univ) (fun X hX t _ => hMem X hX t) (F.unit i) (Set.mem_univ _) M.declared
        hxs).trans (Nat.sub_le_sub_right (hm i) _)
    have hrb : M.run.width (((M.run.into (M.run.region (F.server i)
        (Set.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ)))).filter fun g => M.run.x g ∉ F.server i) ∩
        M.declaredᶜ) ≤ F.b * F.ω - x_b :=
      (ingress_outside M.run hL (F.server i) _ _ M.declared hxb).trans
        (Nat.sub_le_sub_right (Nat.mul_le_mul (hb i) (hω j)) _)
    exact Nat.cast_le.2 (window_reads_le M.run hC (subset_biUnion_of_mem F.server (mem_univ i)) M.registered_input
      hτ M.declared _ hrs hrb (hT.shared_carried i j hu) (hT.shared_read i j hu))
  have hk := fitRate_of_reads_but M.run F.hw.ℓ F.hw.ends F.hw.β M.usage.link F.n F.server M.registered
    M.registered_input F.t F.cuts ((F.unaccounted x_s x_b p : ℕ) : ℝ) c emb T start F.missed p.Nstar p.b_w p.w_e
    F.disjoint hC hL hbw hwe.le (Nat.cast_nonneg _) hS hgap fun Q hQ => ⟨hT.reach Q hQ, hT.fits Q hQ⟩
  have hcount := card_missed_le M.run F.n F.server M.registered M.registered_input F.t F.cuts F.ω F.hw.compute
    M.usage.π F.Φ x_w p.o_w M.declared c emb T F.missed hW F.unit hω hΦ hT.load (fun i j hu => (hA i j).2 hu)
    hT.shared_work
  have hc : 0 < 4 * p.w_e * (p.Nstar * p.b_w - ((F.unaccounted x_s x_b p : ℕ) : ℝ)) :=
    mul_pos (mul_pos (by norm_num) hwe) (sub_pos.2 hgap)
  have hreuse : ((F.n * F.cuts.J : ℕ) : ℝ) * T.reuse * p.Nstar * ((F.unaccounted x_s x_b p : ℕ) : ℝ) ^ 2 ≤
      ((F.n * F.cuts.J : ℕ) : ℝ) * p.reuse * p.Nstar * ((F.unaccounted x_s x_b p : ℕ) : ℝ) ^ 2 :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right
      (mul_le_mul_of_nonneg_left (Nat.cast_le.2 hT.reuse) (Nat.cast_nonneg _)) (Nat.cast_nonneg _)) (sq_nonneg _)
  have h1 := hk.trans (add_le_add hreuse (mul_le_mul_of_nonneg_left (Nat.cast_le.2 hcount) hc.le))
  have h2 : (T.M.card : ℝ) ≤ F.bound x_s x_b x_w p := by
    unfold Frame.bound
    rw [← sub_le_iff_le_add, le_div_iff₀' hc, mul_sub]
    linarith
  exact absurd (h2.trans hY) (not_le.2 hlt)

/-- **The audits accept a prohibited computation with probability at most `δ`**, where every computation the coins
give is feasible and the audits are sound with error `δ`. -/
theorem secure_pr {L E : Type*} [Fintype E] [DecidableEq L] (F : Frame L E) (x_s x_b x_w : ℕ) (p : Recipe) (Y : ℝ)
    (hbw : 0 ≤ p.b_w) (hwe : 0 < p.w_e) (hgap : (F.unaccounted x_s x_b p : ℝ) < p.Nstar * p.b_w)
    (hY : F.bound x_s x_b x_w p ≤ Y) {Ω : Type} [Fintype Ω] (accept : Ω → Prop) (M : Ω → Computation L E) (δ : ℝ)
    (hH : ∀ ω, M ω ∈ H F) (hA : Sound F x_s x_b x_w accept M δ) :
    Claims.Pr (fun ω => accept ω ∧ M ω ∈ B F p Y) ≤ δ :=
  (Claims.Pr_mono fun ω h => ⟨h.1, fun hin => secure F x_s x_b x_w p Y hbw hwe hgap hY (M ω) (hH ω) hin h.2⟩).trans hA

section Single

variable {L E : Type*} (hw : Hardware L E) (X : Finset L) (hX : X ∈ hw.units) (t : ℕ) (cuts : Cuts t)
  (missed : Finset (Fin (1 * cuts.J)))

theorem single_m : (Frame.single hw X hX t cuts missed).m = hw.M X :=
  sup_const (s := (univ : Finset (Fin 1))) univ_nonempty _

theorem single_b [Fintype E] : (Frame.single hw X hX t cuts missed).b = linkCapacity hw.ends hw.β X :=
  sup_const (s := (univ : Finset (Fin 1))) univ_nonempty _

theorem single_Φ : (Frame.single hw X hX t cuts missed).Φ = hw.compute X :=
  sup_const (s := (univ : Finset (Fin 1))) univ_nonempty _

end Single

section Server

variable (n die host fabric chip dram ram queue world store bus peer net : ℕ) (t : ℕ) (cuts : Cuts t)
  (missed : Finset (Fin (1 * cuts.J)))

/-- A server of `n` GPUs carries `n (chip + dram) + ram + queue` bits across a tick. -/
theorem server_m :
    (Frame.single (server n die host fabric chip dram ram queue world store bus peer net) (Loc.inside n)
      (Set.mem_univ _) t cuts missed).m = n * (chip + dram) + ram + queue :=
  (single_m _ _ _ t cuts missed).trans (server_memory n die host fabric chip dram ram queue world store bus peer net)

/-- A server of `n` GPUs reads `net` bits a tick from outside. -/
theorem server_b :
    (Frame.single (server n die host fabric chip dram ram queue world store bus peer net) (Loc.inside n)
      (Set.mem_univ _) t cuts missed).b = net :=
  (single_b _ _ _ t cuts missed).trans (server_link n die host fabric chip dram ram queue world store bus peer net)

/-- A server of `n` GPUs does `n die + host + fabric` work a tick. -/
theorem server_Φ :
    (Frame.single (server n die host fabric chip dram ram queue world store bus peer net) (Loc.inside n)
      (Set.mem_univ _) t cuts missed).Φ = n * die + host + fabric :=
  (single_Φ _ _ _ t cuts missed).trans (server_compute n die host fabric chip dram ram queue world store bus peer net)

end Server

end Slowdown.SecurityProofs
