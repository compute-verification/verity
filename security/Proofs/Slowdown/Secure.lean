module

public import Definitions.Slowdown.Security
public import Proofs.Slowdown.FitRate
public import Proofs.ResourceAccounting.Actual.Outside
public import Definitions.ResourceAccounting.BlockProperties

/-!
# No computation is feasible, accounted for and prohibited

`secure`: `M ∈ H F → M ∈ A F x_s x_b x_w y_b → M ∉ B F p Y`, the three sets of `Definitions.Slowdown.Security`, when `Y`
is at least the fit bound (`Frame.bound`) and the bits a covered window can read of a reference training algorithm
(`Frame.unaccounted`) are fewer than `N* b_w`. Its steps:

* What the limits leave. In a feasible computation whose declared circuit holds `x_s` bits on a server at a cut, the
  server carries at most `m − x_s` bits outside the declared circuit across it (`state_outside`); one whose window reads
  `x_b` bits of the declared circuit from outside its server reads at most `b ω − x_b` others from there
  (`ingress_outside`), and at most `y_b − x_b` when it reads `y_b` in all; and one whose declared circuit does `x_w` work in a window does at most `Φ ω − x_w` outside it
  there (`work_outside`).
* What a covered window reads of the reference training algorithm's novel gates: what its server carries across the
  cut that opens it plus what crosses the server's boundary, each split into the part outside the declared circuit,
  which the limits leave, and the part the algorithm shares with it (`window_reads_le`).
* The count. `fitRate_of_reads_but` bounds the counted multiplications by the fit bound over the covered windows plus
  those placed in the missed windows, which number at most the work the limits leave there plus those shared with the
  declared circuit (`card_missed_le`).

`secure_work` is the claim from work alone: at a frame whose every window the memory and bandwidth audits miss, `Y` at
least `n J (Φ ω − x_w + o_w)`, with no premise on the recipe's other parameters.

Over the audits' runs, with `ExceptWith` the tree's "fails on an accepted run with probability at most `ε`":
`accounted_of_windows` adds the audits' errors over the windows (an accepted run is in `A` outside
`|missed| δ_w + (n J − |missed|) (δ_s + δ_b + δ_y)`), and `secure_pr` is `secure` there (where every run is feasible and an
accepted run is in `A` outside `δ`, an accepted run is not in `B` outside `δ`). `single_m`, `single_b` and `single_Φ`
are the limits of a frame of one server, and `server_m`, `server_b` and `server_Φ` those of a server of `n` GPUs with
its host and switch.
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

/-- **No computation is feasible, accounted for and prohibited.** At a frame `F`, amounts `x_s`, `x_b`, `x_w` and
`y_b`, and a recipe `p` with `b_w ≥ 0`, `w_e > 0` and the bits a covered window can read of a reference training
algorithm (`F.unaccounted x_s x_b y_b p`) fewer than `N* b_w`: a computation that is feasible on the frame's hardware
(`H F`) and consistent with what the audits account for (`A F x_s x_b x_w y_b`) holds no reference training algorithm
of more than `Y` counted multiplications that meets the recipe's premises (`B F p Y`), for every `Y` at least the fit
bound (`F.bound x_s x_b x_w y_b p`). -/
theorem secure {L E : Type*} [Fintype E] [DecidableEq L] (F : Frame L E) (x_s x_b x_w y_b : ℕ) (p : Recipe) (Y : ℝ)
    (hbw : 0 ≤ p.b_w) (hwe : 0 < p.w_e) (hgap : (F.unaccounted x_s x_b y_b p : ℝ) < p.Nstar * p.b_w)
    (hY : F.bound x_s x_b x_w y_b p ≤ Y) (M : Computation L E) :
    M ∈ H F → M ∈ A F x_s x_b x_w y_b → M ∉ B F p Y := by
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
        ((c.θᶜ).map ⟨emb.emb, emb.emb_inj⟩) : ℝ) ≤ ((F.unaccounted x_s x_b y_b p : ℕ) : ℝ) := by
    intro i j hu
    obtain ⟨hxs, hxb, hyb⟩ := (hA i j).1 hu
    have hτ : F.cuts.τ j.succ ≤ F.t := (F.cuts.mono (Fin.le_last _)).trans F.cuts.last.le
    have hrs : M.run.stateIn (F.server i) (F.cuts.τ j.castSucc) M.declaredᶜ ≤ F.m - x_s :=
      (state_outside M.run (T := Set.univ) (fun X hX t _ => hMem X hX t) (F.unit i) (Set.mem_univ _) M.declared
        hxs).trans (Nat.sub_le_sub_right (hm i) _)
    have hrb : M.run.width (((M.run.into (M.run.region (F.server i)
        (Set.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ)))).filter fun g => M.run.x g ∉ F.server i) ∩
        M.declaredᶜ) ≤ min (F.b * F.ω) y_b - x_b := by
      have h1 := (ingress_outside M.run hL (F.server i) _ _ M.declared hxb).trans
        (Nat.sub_le_sub_right (Nat.mul_le_mul (hb i) (hω j)) _)
      have h2 := width_inter_add_compl M.run ((M.run.into (M.run.region (F.server i)
        (Set.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ)))).filter fun g => M.run.x g ∉ F.server i) M.declared
      unfold Placed.ingress at hyb
      rw [Finset.filter_congr_decidable] at hyb
      have hxb' : x_b ≤ M.run.width (((M.run.into (M.run.region (F.server i)
        (Set.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ)))).filter fun g => M.run.x g ∉ F.server i) ∩ M.declared) :=
        hxb
      have hyb' : M.run.width ((M.run.into (M.run.region (F.server i)
        (Set.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ)))).filter fun g => M.run.x g ∉ F.server i) ≤ y_b := hyb
      omega
    exact Nat.cast_le.2 (window_reads_le M.run hC (subset_biUnion_of_mem F.server (mem_univ i)) M.registered_input
      hτ M.declared _ hrs hrb (hT.shared_carried i j hu) (hT.shared_read i j hu))
  have hk := fitRate_of_reads_but M.run F.hw.ℓ F.hw.ends F.hw.β M.usage.link F.n F.server M.registered
    M.registered_input F.t F.cuts ((F.unaccounted x_s x_b y_b p : ℕ) : ℝ) c emb T start F.missed p.Nstar p.b_w p.w_e
    F.disjoint hC hL hbw hwe.le (Nat.cast_nonneg _) hS hgap fun Q hQ => ⟨hT.reach Q hQ, hT.fits Q hQ⟩
  have hcount := card_missed_le M.run F.n F.server M.registered M.registered_input F.t F.cuts F.ω F.hw.compute
    M.usage.π F.Φ x_w p.o_w M.declared c emb T F.missed hW F.unit hω hΦ hT.load (fun i j hu => (hA i j).2 hu)
    hT.shared_work
  have hc : 0 < 4 * p.w_e * (p.Nstar * p.b_w - ((F.unaccounted x_s x_b y_b p : ℕ) : ℝ)) :=
    mul_pos (mul_pos (by norm_num) hwe) (sub_pos.2 hgap)
  have hreuse : ((F.n * F.cuts.J : ℕ) : ℝ) * T.reuse * p.Nstar * ((F.unaccounted x_s x_b y_b p : ℕ) : ℝ) ^ 2 ≤
      ((F.n * F.cuts.J : ℕ) : ℝ) * p.reuse * p.Nstar * ((F.unaccounted x_s x_b y_b p : ℕ) : ℝ) ^ 2 :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right
      (mul_le_mul_of_nonneg_left (Nat.cast_le.2 hT.reuse) (Nat.cast_nonneg _)) (Nat.cast_nonneg _)) (sq_nonneg _)
  have h1 := hk.trans (add_le_add hreuse (mul_le_mul_of_nonneg_left (Nat.cast_le.2 hcount) hc.le))
  have h2 : (T.M.card : ℝ) ≤ F.bound x_s x_b x_w y_b p := by
    unfold Frame.bound
    rw [← sub_le_iff_le_add, le_div_iff₀' hc, mul_sub]
    linarith
  exact absurd (h2.trans hY) (not_le.2 hlt)

/-- **Work alone.** At a frame whose every window the memory and bandwidth audits miss, a feasible computation in
`A` holds no reference training algorithm of more than `Y` counted multiplications, for every `Y` at least
`n J (Φ ω − x_w + o_w)`: what the compute rate leaves beside the `x_w` work the declared circuit does in each window.
No premise on the recipe's other parameters. -/
theorem secure_work {L E : Type*} [Fintype E] [DecidableEq L] (F : Frame L E) (hF : F.missed = univ)
    (x_s x_b x_w y_b : ℕ) (p : Recipe) (Y : ℝ)
    (hY : ((F.n * F.cuts.J * (F.Φ * F.ω - x_w + p.o_w) : ℕ) : ℝ) ≤ Y) (M : Computation L E) :
    M ∈ H F → M ∈ A F x_s x_b x_w y_b → M ∉ B F p Y := by
  classical
  rintro ⟨hC, hW, -, -, hL⟩ hA ⟨c, emb, T, start, hT, hlt⟩
  have hω : ∀ j : Fin F.cuts.J, F.cuts.τ j.succ - F.cuts.τ j.castSucc ≤ F.ω := fun j =>
    le_sup (f := fun j : Fin F.cuts.J => F.cuts.τ j.succ - F.cuts.τ j.castSucc) (mem_univ j)
  have hΦ : ∀ i, F.hw.compute (F.server i) ≤ F.Φ := fun i =>
    le_sup (f := fun i => F.hw.compute (F.server i)) (mem_univ i)
  have hin : ∀ i j, finProdFinEquiv (i, j) ∈ F.missed := fun i j => hF ▸ mem_univ _
  obtain ⟨P, hP⟩ := ResourceAccounting.SecurityProofs.Actual.ServerBlocksAt M.run F.hw.ℓ (fun _ => M.run.N) F.hw.ends
    F.hw.β M.usage.link F.n F.server M.registered M.registered_input F.t F.cuts ∅ F.disjoint hC
    (computeRate_N M.run _) hL
  have hcount := card_missed_le M.run F.n F.server M.registered M.registered_input F.t F.cuts F.ω F.hw.compute
    M.usage.π F.Φ x_w p.o_w M.declared c emb T F.missed hW F.unit hω hΦ hT.load (fun i j hu => (hA i j).2 hu)
    hT.shared_work
  have hall : T.M.card ≤ F.missed.card * (F.Φ * F.ω - x_w + p.o_w) := by
    refine (card_le_card fun g hg => mem_filter.2 ⟨hg, ?_⟩).trans hcount
    have hne : (pull emb P).unit g ≠ none := fun h =>
      absurd (((pull emb P).input_iff g).1 h) (by simp [(mem_filter.1 (T.M_computed hg)).2])
    obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 hne
    obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective u
    have h2 := mem_pull_gates.1 ((Partition.mem_gates _).2 hu)
    exact ⟨i, j, hin i j, (Iff.of_eq (congrArg (fun s => emb.emb g ∈ s) (hP i j).1)).1 h2⟩
  rw [hF, card_univ, Fintype.card_fin] at hall
  exact absurd ((Nat.cast_le.2 hall).trans hY) (not_le.2 hlt)

section Runs

open scoped ENNReal

variable {L E : Type*} [DecidableEq L] {Ω : Type} (pr : (Ω → Prop) → ℝ≥0∞)
  (mono : ∀ p q : Ω → Prop, (∀ r, p r → q r) → pr p ≤ pr q) (accepts : Ω → Prop) (M : Ω → Computation L E)

include mono

/-- **An accepted run is not prohibited, outside `δ`**: over runs whose computations are all feasible, if an accepted
run's computation is in `A` outside probability `δ`, it is not in `B` outside `δ`. `pr` is any monotone probability on
the runs. -/
theorem secure_pr [Fintype E] (F : Frame L E) (x_s x_b x_w y_b : ℕ) (p : Recipe) (Y : ℝ)
    (hbw : 0 ≤ p.b_w) (hwe : 0 < p.w_e) (hgap : (F.unaccounted x_s x_b y_b p : ℝ) < p.Nstar * p.b_w)
    (hY : F.bound x_s x_b x_w y_b p ≤ Y) (δ : ℝ≥0∞) (hH : ∀ ω, M ω ∈ H F)
    (hA : ExceptWith pr accepts (fun ω => M ω ∈ A F x_s x_b x_w y_b) δ) :
    ExceptWith pr accepts (fun ω => M ω ∉ B F p Y) δ :=
  (mono _ _ fun ω h => ⟨h.1, fun hin => h.2 (secure F x_s x_b x_w y_b p Y hbw hwe hgap hY (M ω) (hH ω) hin)⟩).trans hA

/-- **The audits' errors add over the windows.** If at each covered window an accepted run's declared circuit holds
`x_s` bits at the cut outside probability `δ_s` and reads `x_b` bits from outside its server outside `δ_b`, and the
window reads at most `y_b` bits from there outside `δ_y`, and at each missed window the declared circuit does `x_w`
work outside `δ_w`, then an accepted run's computation is in `A` outside
`|missed| δ_w + (n J − |missed|) (δ_s + δ_b + δ_y)`. `pr` is any monotone probability on the runs with the union
bound. -/
theorem accounted_of_windows
    (union : ∀ {κ : Type} (s : Finset κ) (p : κ → Ω → Prop), pr (fun r => ∃ x ∈ s, p x r) ≤ ∑ x ∈ s, pr (p x))
    (F : Frame L E) (x_s x_b x_w y_b : ℕ) (δ_s δ_b δ_w δ_y : ℝ≥0∞)
    (hs : ∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      ExceptWith pr accepts (fun ω => x_s ≤ (M ω).carries F i j (M ω).declared) δ_s)
    (hb : ∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      ExceptWith pr accepts (fun ω => x_b ≤ (M ω).reads F i j (M ω).declared) δ_b)
    (hy : ∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      ExceptWith pr accepts (fun ω => (M ω).run.ingress (F.server i) (F.span j) ≤ y_b) δ_y)
    (hw : ∀ i j, finProdFinEquiv (i, j) ∈ F.missed →
      ExceptWith pr accepts
        (fun ω => x_w ≤ (M ω).run.work (M ω).usage.π ((M ω).window F i j ∩ (M ω).declared)) δ_w) :
    ExceptWith pr accepts (fun ω => M ω ∈ A F x_s x_b x_w y_b)
      (F.missed.card * δ_w + (F.n * F.cuts.J - F.missed.card : ℕ) * (δ_s + δ_b + δ_y)) := by
  classical
  let good : Option Bool → Fin F.n → Fin F.cuts.J → Ω → Prop := fun k i j ω =>
    match k with
    | none => x_s ≤ (M ω).carries F i j (M ω).declared
    | some true => x_b ≤ (M ω).reads F i j (M ω).declared
    | some false => (M ω).run.ingress (F.server i) (F.span j) ≤ y_b
  let bad : Fin (F.n * F.cuts.J) → Ω → Prop := fun u ω => accepts ω ∧
    ¬ ((u ∉ F.missed → ∀ k, good k (finProdFinEquiv.symm u).1 (finProdFinEquiv.symm u).2 ω) ∧
      (u ∈ F.missed → x_w ≤ (M ω).run.work (M ω).usage.π
        ((M ω).window F (finProdFinEquiv.symm u).1 (finProdFinEquiv.symm u).2 ∩ (M ω).declared)))
  have hu : ∀ u : Fin (F.n * F.cuts.J),
      finProdFinEquiv ((finProdFinEquiv.symm u).1, (finProdFinEquiv.symm u).2) = u :=
    fun u => finProdFinEquiv.apply_symm_apply u
  have hbad : ∀ u, pr (bad u) ≤ if u ∈ F.missed then δ_w else δ_s + δ_b + δ_y := by
    intro u
    split_ifs with h
    · exact (mono _ _ fun ω hω => ⟨hω.1, fun hx => hω.2 ⟨fun hn => absurd h hn, fun _ => hx⟩⟩).trans
        (hw _ _ (by rw [hu]; exact h))
    · refine (mono _ (fun ω => ∃ k ∈ (univ : Finset (Option Bool)), accepts ω ∧
          ¬ good k (finProdFinEquiv.symm u).1 (finProdFinEquiv.symm u).2 ω) fun ω hω => ?_).trans
        ((union _ _).trans ?_)
      · by_contra hno
        exact hω.2 ⟨fun _ k => not_not.1 fun hk => hno ⟨k, mem_univ _, hω.1, hk⟩, fun hm => absurd hm h⟩
      · rw [Fintype.sum_option, Fintype.sum_bool, add_assoc]
        exact add_le_add (hs _ _ (by rw [hu]; exact h))
          (add_le_add (hb _ _ (by rw [hu]; exact h)) (hy _ _ (by rw [hu]; exact h)))
  have hsum : ∑ u : Fin (F.n * F.cuts.J), (if u ∈ F.missed then δ_w else δ_s + δ_b + δ_y) =
      F.missed.card * δ_w + (F.n * F.cuts.J - F.missed.card : ℕ) * (δ_s + δ_b + δ_y) := by
    rw [sum_ite, sum_const, sum_const, nsmul_eq_mul, nsmul_eq_mul, filter_mem_eq_inter, univ_inter,
      filter_not, filter_mem_eq_inter, univ_inter, card_univ_sdiff, Fintype.card_fin]
  refine (mono _ (fun ω => ∃ u ∈ (univ : Finset (Fin (F.n * F.cuts.J))), bad u ω) fun ω hω => ?_).trans
    ((union _ _).trans ((sum_le_sum fun u _ => hbad u).trans hsum.le))
  by_contra hno
  refine hω.2 fun i j => ?_
  by_contra hij
  refine hno ⟨finProdFinEquiv (i, j), mem_univ _, hω.1, fun hg => hij ⟨fun hn => ?_, by simpa using hg.2⟩⟩
  have h3 := hg.1 hn
  simp only [Equiv.symm_apply_apply] at h3
  exact ⟨h3 none, h3 (some true), h3 (some false)⟩

end Runs

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
