module

public import Definitions.Slowdown.Guarantees.Secure
public import Definitions.Slowdown.Guarantees.Run
public import Proofs.Core.Game.Union
public import Proofs.Core.Game.Prob
public import Proofs.Slowdown.FitRate
public import Proofs.ResourceAccounting.Actual.Outside

/-!
# No computation is feasible, accounted for and prohibited

`Secure` proves `Guarantees.Secure`: `M ∈ H F.hw → M ∈ A F x_s x_b x_w y_b → M ∉ B F tb sh Y`, the three sets of
`Definitions.Slowdown.Security`, at a training bound `tb` that holds, when `Y` is at least the bound on training
(`Frame.bound`) and `tb` is defined at the bits a covered window can read of a reference training algorithm
(`Frame.unaccounted`, `Frame.BoundAtMost`). It is one path for every training bound: `fit_holds` is that the fit
bound holds (`Training.fit`), and another route to a bound on training is another `Training` with its proof. Its
steps:

* What the limits leave. In a feasible computation whose server holds `x_s` bits of the declared circuit at a cut,
  the server carries at most `m − x_s` bits outside the declared circuit across it (`state_outside`); one whose links
  deliver to a window's server `x_b` bits of the declared circuit and `y_b` bits in all reads at most
  `min (b ω) y_b − x_b` others from outside the server (`delivered_outside`); and one whose declared circuit does `x_w`
  work in a window of `len` ticks does at most `Φ len − x_w` outside it there (`work_outside`).
* What a covered window reads of the reference training algorithm's novel gates: what its server carries across the
  cut that opens it plus what crosses the server's boundary, each split into the part outside the declared circuit,
  which the limits leave, and the part the algorithm shares with it (`window_reads_le`).
* The count. The windows divide the reference training algorithm (`ServerBlocksAt`, pulled along the containment), so
  the training bound, which holds, bounds its counted multiplications in the covered windows; those placed in the
  missed windows number at most the work the limits leave there plus those shared with the declared circuit
  (`card_missed_le`).

Where the memory and bandwidth audits miss every window no counted multiplication is in a covered one, the bound is
the missed windows' sum (`bound_of_missed`), and nothing is asked of the training bound.

`RunSecure` proves `Guarantees.RunSecure` from `Secure`: with the probability of an event the run's value against an
optimal developer (`Game.value`), which is monotone and has the union bound, each audit's soundness at its own
acceptance is soundness at the verifier's, the errors add over the windows, and at an outcome where every window's
clauses hold the computation is in `A`, so `Secure` takes it out of `B`. `RunComplete` is the union bound for an
honest strategy.

`single_m`, `single_b` and `single_Φ` are the limits of a frame of one server, and `server_m`, `server_b` and
`server_Φ` those of a server of `n` GPUs with its host and switch.
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
the compute rate leaves over the window's ticks beside the `x_w` that `S` does there, at a price of at least 1 a
counted multiplication, and those in `S` at most `o_w`. -/
theorem card_missed_le {L : Type*} [DecidableEq L] (A : Placed L) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t : ℕ) (κ : Cuts t)
    {𝒳 : Set (Finset L)} (compute : Finset L → ℕ) (π : Fin A.N → ℕ) (Φ x_w o_w : ℕ) (S : Finset (Fin A.N))
    (c : RegisteredCircuit) (M : Containment c.D (A.declaration (univ.biUnion X) t θ hθ).D) (T : Threading c)
    (I : Finset (Fin (n * κ.J)))
    (hW : A.ComputeRate 𝒳 π compute) (hX : ∀ i, X i ∈ 𝒳) (hΦ : ∀ i, compute (X i) ≤ Φ)
    (hπ : ∀ g ∈ T.M, 1 ≤ π (M.emb g))
    (hx : ∀ (i : Fin n) (j : Fin κ.J), finProdFinEquiv (i, j) ∈ I →
      x_w ≤ A.work π (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) ∩ S))
    (ho : ∀ (i : Fin n) (j : Fin κ.J), finProdFinEquiv (i, j) ∈ I →
      (T.M.filter fun g => M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) ∩ S).card ≤ o_w) :
    (T.M.filter fun g => ∃ i j, finProdFinEquiv (i, j) ∈ I ∧
      M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))).card ≤
      ∑ u ∈ I, (Φ * (κ.τ (finProdFinEquiv.symm u).2.succ - κ.τ (finProdFinEquiv.symm u).2.castSucc) - x_w + o_w) := by
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
        Φ * (κ.τ j.succ - κ.τ j.castSucc) - x_w + o_w := by
    intro i j hu
    have hout : (T.M.filter fun g => M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) \ S).card ≤
        Φ * (κ.τ j.succ - κ.τ j.castSucc) - x_w := by
      have h1 : (T.M.filter fun g => M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) \ S).card ≤
          A.work π (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) \ S) := by
        convert card_filter_le_work A π _ T.M ⟨M.emb, M.emb_inj⟩ hπ
      exact h1.trans ((work_outside A hW (hX i) _ _ S (hx i j hu)).trans
        (Nat.sub_le_sub_right (Nat.mul_le_mul_right _ (hΦ i)) _))
    refine (card_le_card fun g hg => ?_).trans ((card_union_le _ _).trans (Nat.add_le_add hout (ho i j hu)))
    obtain ⟨hgs, hgr⟩ := mem_filter.1 hg
    by_cases hS : M.emb g ∈ S
    · exact mem_union_right _ (mem_filter.2 ⟨hgs, mem_inter.2 ⟨hgr, hS⟩⟩)
    · exact mem_union_left _ (mem_filter.2 ⟨hgs, mem_sdiff.2 ⟨hgr, hS⟩⟩)
  exact (card_le_card hsub).trans (card_biUnion_le.trans (sum_le_sum fun u hu =>
    hwin _ _ (by rwa [Prod.mk.eta, Equiv.apply_symm_apply])))

/-- Where the memory and bandwidth audits miss every window, the bound on training is the missed windows' sum. -/
theorem bound_of_missed {L E : Type*} [Fintype E] (F : Frame L E) (hF : F.missed = univ) (x_s x_b x_w y_b : ℕ)
    (tb : Training) (sh : Shared) :
    F.bound x_s x_b x_w y_b tb sh =
      ((∑ u : Fin (F.n * F.cuts.J), (F.Φ * F.len (finProdFinEquiv.symm u).2 - x_w + sh.o_w) : ℕ) : ℝ) := by
  unfold Frame.bound
  rw [hF, card_univ, Fintype.card_fin, Nat.sub_self, tb.bound_zero, zero_add]

/-- The fit bound's count. -/
theorem fit_bound (p : Recipe) (k : ℕ) (S : ℝ) :
    (Training.fit p).bound k S = k * p.reuse * p.Nstar * S ^ 2 / (4 * p.w_e * (p.Nstar * p.b_w - S)) :=
  rfl

/-- **The fit bound holds** (`Training.fit`): `Fit.fitBoundSlots_but` at the slots that are the weights, with the
counted multiplications in the windows left out taken off both sides. -/
theorem fit_holds (p : Recipe) : (Training.fit p).Holds := by
  classical
  intro c T k Q I start S ⟨hbw, hwe, hS0, hgap⟩ ⟨hreuse, hR, hF⟩ hread
  have hk := Fit.fitBoundSlots_but T (Fit.idSlots T) Q I start p.Nstar S p.b_w p.w_e hbw hwe.le hgap hS0 hread
    (fun u hu hne => by
      have h1 := hR u hu (hne.mono fun g hg => ((Fit.mem_keep_threadMults T _ _ u g).1 hg).1)
      have h2 := Fit.card_threadSlots (Fit.keepSlots (Fit.idSlots T) (Fit.outside Q I))
        (Fit.idSlots_injOn T) univ u
      rw [Fit.keep_threadWeights_univ] at h2
      exact h1.trans h2.ge)
    (fun j hj => by
      have h2 := Fit.card_slotsUsed (Fit.idSlots T) (Fit.idSlots_injOn T) (Q.gates j)
      rw [h2]
      exact hF j hj)
  have hsplit : T.M.card = (T.M.filter fun g => ∀ j ∈ I, g ∉ Q.gates j).card +
      (T.M.filter fun g => ∃ j ∈ I, g ∈ Q.gates j).card := by
    rw [← card_union_of_disjoint (disjoint_filter.2 fun g _ h ⟨j, hj, hg⟩ => h j hj hg)]
    refine congrArg card (Finset.ext fun g => ?_)
    simp only [mem_union, mem_filter]
    refine ⟨fun hg => ?_, fun hg => hg.elim (·.1) (·.1)⟩
    by_cases h : ∃ j ∈ I, g ∈ Q.gates j
    · exact Or.inr ⟨hg, h⟩
    · exact Or.inl ⟨hg, fun j hj hgj => h ⟨j, hj, hgj⟩⟩
  have hc : 0 < 4 * p.w_e * (p.Nstar * p.b_w - S) := mul_pos (mul_pos (by norm_num) hwe) (sub_pos.2 hgap)
  have hre : ((k - I.card : ℕ) : ℝ) * T.reuse * p.Nstar * S ^ 2 ≤
      ((k - I.card : ℕ) : ℝ) * p.reuse * p.Nstar * S ^ 2 :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right
      (mul_le_mul_of_nonneg_left (Nat.cast_le.2 hreuse) (Nat.cast_nonneg _)) (Nat.cast_nonneg _)) (sq_nonneg _)
  rw [hsplit] at hk
  push_cast at hk
  change _ ≤ ((k - I.card : ℕ) : ℝ) * p.reuse * p.Nstar * S ^ 2 / (4 * p.w_e * (p.Nstar * p.b_w - S))
  rw [le_div_iff₀' hc]
  linarith

theorem Secure : Guarantees.Secure := by
  classical
  intro L E _ _ F x_s x_b x_w y_b tb sh Y htb ⟨hvalid, hY⟩ M
  rintro ⟨hC, hW, hMem, hL⟩ hA ⟨c, emb, T, start, hT, hlt⟩
  have hω : ∀ j : Fin F.cuts.J, F.cuts.τ j.succ - F.cuts.τ j.castSucc ≤ F.ω := fun j =>
    le_sup (f := F.len) (mem_univ j)
  have hm : ∀ i, F.hw.M (F.server i) ≤ F.m := fun i => le_sup (f := fun i => F.hw.M (F.server i)) (mem_univ i)
  have hb : ∀ i, linkCapacity F.hw.ends F.hw.β (F.server i) ≤ F.b := fun i =>
    le_sup (f := fun i => linkCapacity F.hw.ends F.hw.β (F.server i)) (mem_univ i)
  have hΦ : ∀ i, F.hw.compute (F.server i) ≤ F.Φ := fun i =>
    le_sup (f := fun i => F.hw.compute (F.server i)) (mem_univ i)
  have hS : ∀ (i : Fin F.n) (j : Fin F.cuts.J), finProdFinEquiv (i, j) ∉ F.missed →
      (readsIn (M.run.A.declaration (univ.biUnion F.server) F.t M.registered M.registered_input)
        (M.run.A.region (F.server i) (Set.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ)))
        ((c.θᶜ).map ⟨emb.emb, emb.emb_inj⟩) : ℝ) ≤ ((F.unaccounted x_s x_b y_b sh : ℕ) : ℝ) := by
    intro i j hu
    obtain ⟨hxs, hxb, hyb⟩ := (hA i j).1 hu
    have hτ : F.cuts.τ j.succ ≤ F.t := (F.cuts.mono (Fin.le_last _)).trans F.cuts.last.le
    have hrs : M.run.A.stateIn (F.server i) (F.cuts.τ j.castSucc) M.declaredᶜ ≤ F.m - x_s :=
      state_outside ((hMem _ (F.unit i) _).trans (hm i)) M.declared hxs
    have hrb : M.run.A.width (((M.run.A.into (M.run.A.region (F.server i)
        (Set.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ)))).filter fun g => M.run.A.x g ∉ F.server i) ∩
        M.declaredᶜ) ≤ min (F.b * F.ω) y_b - x_b := by
      change x_b ≤ M.delivered F i j M.declared at hxb
      change M.delivered F i j univ ≤ y_b at hyb
      unfold Computation.delivered at hxb hyb
      have h1 := delivered_outside M.run.A hL (F.server i) (F.cuts.τ j.castSucc) (F.cuts.τ j.succ) M.declared
        (x := x_b) (y := y_b) (by convert hxb) (by convert hyb)
      exact (le_of_eq (by congr)).trans (h1.trans (Nat.sub_le_sub_right
        (min_le_min_right _ (Nat.mul_le_mul (hb i) (hω j))) _))
    exact Nat.cast_le.2 (window_reads_le M.run.A hC (subset_biUnion_of_mem F.server (mem_univ i)) M.registered_input
      hτ M.declared _ hrs hrb (hT.shared_carried i j hu) (hT.shared_read i j hu))
  obtain ⟨P, hP⟩ := ResourceAccounting.SecurityProofs.Actual.ServerBlocksAt M.run.A F.hw.ℓ
    (fun _ => M.run.A.N) F.hw.ends F.hw.β M.usage.link F.n F.server M.registered M.registered_input F.t F.cuts
    ((c.θᶜ).map ⟨emb.emb, emb.emb_inj⟩) F.disjoint hC (computeRate_N M.run.A _) hL
  have hQ : Divides F M emb (pull emb P) := fun i j g =>
    mem_pull_gates.trans (Iff.of_eq (congrArg (fun s => emb.emb g ∈ s) (hP i j).1))
  have hread : ∀ u ∉ F.missed, (freshInput c ((pull emb P).gates u) : ℝ) ≤
      ((F.unaccounted x_s x_b y_b sh : ℕ) : ℝ) := by
    intro u hu
    obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective u
    have h1 : freshInput c ((pull emb P).gates (finProdFinEquiv (i, j))) ≤
        readsIn (M.run.A.declaration (univ.biUnion F.server) F.t M.registered M.registered_input)
          (P.gates (finProdFinEquiv (i, j))) ((c.θᶜ).map ⟨emb.emb, emb.emb_inj⟩) :=
      freshInput_pull_le_readsIn_compl emb P (finProdFinEquiv (i, j))
    rw [(hP i j).1] at h1
    exact (Nat.cast_le.2 h1).trans (hS i j hu)
  have hcov : ((T.M.filter fun g => ∀ j ∈ F.missed, g ∉ (pull emb P).gates j).card : ℝ) ≤
      tb.bound (F.n * F.cuts.J - F.missed.card) ((F.unaccounted x_s x_b y_b sh : ℕ) : ℝ) := by
    by_cases hF : F.missed = univ
    · have hempty : (T.M.filter fun g => ∀ j ∈ F.missed, g ∉ (pull emb P).gates j) = ∅ := by
        refine filter_eq_empty_iff.2 fun g hg hno => ?_
        have hne : (pull emb P).unit g ≠ none := fun h =>
          absurd (((pull emb P).input_iff g).1 h) (by simp [(mem_filter.1 (T.M_computed hg)).2])
        obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 hne
        exact hno u (hF ▸ mem_univ u) ((Partition.mem_gates _).2 hu)
      rw [hempty, hF, card_univ, Fintype.card_fin, Nat.sub_self, tb.bound_zero]
      simp
    · exact htb T (pull emb P) F.missed start _ (hvalid hF) (hT.prem _ hQ) hread
  have hcount : _ ≤ ∑ u ∈ F.missed, (F.Φ * F.len (finProdFinEquiv.symm u).2 - x_w + sh.o_w) :=
    card_missed_le M.run.A F.n F.server M.registered M.registered_input F.t F.cuts F.hw.compute
      M.usage.π F.Φ x_w sh.o_w M.declared c emb T F.missed hW F.unit hΦ hT.load (fun i j hu => (hA i j).2 hu)
      hT.shared_work
  have hsplit : T.M.card ≤ (T.M.filter fun g => ∀ j ∈ F.missed, g ∉ (pull emb P).gates j).card +
      (T.M.filter fun g => ∃ i j, finProdFinEquiv (i, j) ∈ F.missed ∧
        emb.emb g ∈ M.run.A.region (F.server i) (Set.Ico (F.cuts.τ j.castSucc) (F.cuts.τ j.succ))).card := by
    refine (card_le_card fun g hg => ?_).trans (card_union_le _ _)
    by_cases h : ∀ j ∈ F.missed, g ∉ (pull emb P).gates j
    · exact mem_union_left _ (mem_filter.2 ⟨hg, h⟩)
    · push_neg at h
      obtain ⟨u, hu, hgu⟩ := h
      obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective u
      exact mem_union_right _ (mem_filter.2 ⟨hg, i, j, hu, (hQ i j g).1 hgu⟩)
  have h2 : (T.M.card : ℝ) ≤ F.bound x_s x_b x_w y_b tb sh := by
    unfold Frame.bound
    exact (Nat.cast_le.2 (hsplit.trans (Nat.add_le_add_left hcount _))).trans
      (by rw [Nat.cast_add]; exact add_le_add hcov le_rfl)
  exact absurd (h2.trans hY) (not_le.2 hlt)

/-- The links of a feasible computation deliver to a window's server at most `b ω` bits. -/
theorem delivered_le_capacity {L E : Type*} [Fintype E] [DecidableEq L] (F : Frame L E) (M : Computation L E)
    (hM : M ∈ H F.hw) (i : Fin F.n) (j : Fin F.cuts.J) : M.delivered F i j univ ≤ F.b * F.ω := by
  unfold Computation.delivered
  simp only [inter_univ]
  refine (le_of_eq (sum_congr rfl fun τ _ => ?_)).trans ((delivered_le M.run.A hM.2.2.2 (F.server i) _ _).trans
    (Nat.mul_le_mul (le_sup (f := fun i => linkCapacity F.hw.ends F.hw.β (F.server i)) (mem_univ i))
      (le_sup (f := F.len) (mem_univ j))))
  congr

section Run

open scoped ENNReal

theorem RunSecure : Guarantees.RunSecure := by
  classical
  intro L E _ _ α g world F x_s x_b x_w y_b tb sh Y acc_s acc_b acc_y acc_w δ_s δ_b δ_y δ_w htb hY hH hs hb hy hw
  let acc : α → Prop := fun a => acc_s a ∧ acc_b a ∧ acc_y a ∧ acc_w a
  let good : Option Bool → Fin F.n → Fin F.cuts.J → α → Prop := fun k i j a =>
    match k with
    | none => world a ∈ HeldAtLeast F i j (x_s a)
    | some true => world a ∈ DeliveredAtLeast F i j (x_b a)
    | some false => world a ∈ DeliveredAtMost F i j (y_b a)
  let bad : Fin (F.n * F.cuts.J) → α → Prop := fun u a => acc a ∧
    ¬ ((u ∉ F.missed → ∀ k, good k (finProdFinEquiv.symm u).1 (finProdFinEquiv.symm u).2 a) ∧
      (u ∈ F.missed → world a ∈ WorkAtLeast F (finProdFinEquiv.symm u).1 (finProdFinEquiv.symm u).2 (x_w a)))
  have hu : ∀ u : Fin (F.n * F.cuts.J),
      finProdFinEquiv ((finProdFinEquiv.symm u).1, (finProdFinEquiv.symm u).2) = u :=
    fun u => finProdFinEquiv.apply_symm_apply u
  have hbad : ∀ u, Game.value (bad u) g ≤ if u ∈ F.missed then δ_w else δ_s + δ_b + δ_y := by
    intro u
    split_ifs with h
    · exact (Game.value_mono (fun a ha => ⟨ha.1.2.2.2, fun hx => ha.2 ⟨fun hn => absurd h hn, fun _ => hx⟩⟩) g).trans
        (hw _ _ (by rw [hu]; exact h))
    · refine (Game.value_mono (q := fun a => ∃ k ∈ (univ : Finset (Option Bool)), acc a ∧
          ¬ good k (finProdFinEquiv.symm u).1 (finProdFinEquiv.symm u).2 a) (fun a ha => ?_) g).trans
        ((Game.value_exists_finset_le _ _ g).trans ?_)
      · by_contra hno
        exact ha.2 ⟨fun _ k => not_not.1 fun hk => hno ⟨k, mem_univ _, ha.1, hk⟩, fun hm => absurd hm h⟩
      · rw [Fintype.sum_option, Fintype.sum_bool, add_assoc]
        exact add_le_add
          ((Game.value_mono (fun a ha => ⟨ha.1.1, ha.2⟩) g).trans (hs _ _ (by rw [hu]; exact h)))
          (add_le_add ((Game.value_mono (fun a ha => ⟨ha.1.2.1, ha.2⟩) g).trans (hb _ _ (by rw [hu]; exact h)))
            ((Game.value_mono (fun a ha => ⟨ha.1.2.2.1, ha.2⟩) g).trans (hy _ _ (by rw [hu]; exact h))))
  have hsum : ∑ u : Fin (F.n * F.cuts.J), (if u ∈ F.missed then δ_w else δ_s + δ_b + δ_y) =
      F.missed.card * δ_w + (F.n * F.cuts.J - F.missed.card : ℕ) * (δ_s + δ_b + δ_y) := by
    rw [sum_ite, sum_const, sum_const, nsmul_eq_mul, nsmul_eq_mul, filter_mem_eq_inter, univ_inter,
      filter_not, filter_mem_eq_inter, univ_inter, card_univ_sdiff, Fintype.card_fin]
  refine (Game.value_mono (q := fun a => ∃ u ∈ (univ : Finset (Fin (F.n * F.cuts.J))), bad u a)
    (fun a ha => ?_) g).trans
    ((Game.value_exists_finset_le _ _ g).trans ((sum_le_sum fun u _ => hbad u).trans hsum.le))
  by_contra hno
  refine ha.2 (Secure F _ _ _ _ tb sh (Y a) htb (hY a) (world a) (hH a) fun i j => ?_)
  by_contra hij
  refine hno ⟨finProdFinEquiv (i, j), mem_univ _, ha.1, fun hg => hij ⟨fun hn => ?_, fun hm => ?_⟩⟩
  · have h3 := hg.1 hn
    simp only [Equiv.symm_apply_apply] at h3
    exact ⟨h3 none, h3 (some true), h3 (some false)⟩
  · have h4 := hg.2 hm
    simp only [Equiv.symm_apply_apply] at h4
    exact h4

/-- **An honest developer's run is accepted, outside the audits' errors**: where the verifier accepts whenever each
of the four audits does, a strategy `honest` that memory's audit rejects with probability at most `η_s`, bandwidth's
`η_b`, the count `η_y` and work's `η_w` is rejected with probability at most their sum. -/
theorem RunComplete {α : Type} (g : Game α) (honest : g.Strategy) (accept acc_s acc_b acc_y acc_w : α → Prop)
    (η_s η_b η_y η_w : ℝ≥0∞) (hacc : ∀ a, acc_s a → acc_b a → acc_y a → acc_w a → accept a)
    (hs : Game.prob (fun a => ¬ acc_s a) g honest ≤ η_s) (hb : Game.prob (fun a => ¬ acc_b a) g honest ≤ η_b)
    (hy : Game.prob (fun a => ¬ acc_y a) g honest ≤ η_y) (hw : Game.prob (fun a => ¬ acc_w a) g honest ≤ η_w) :
    Game.prob (fun a => ¬ accept a) g honest ≤ η_s + η_b + η_y + η_w := by
  have h1 : Game.prob (fun a => ¬ accept a) g honest ≤
      Game.prob (fun a => ((¬ acc_s a ∨ ¬ acc_b a) ∨ ¬ acc_y a) ∨ ¬ acc_w a) g honest :=
    Game.prob_mono (fun a ha => by
      by_contra hno
      obtain ⟨h123, h4⟩ := not_or.1 hno
      obtain ⟨h12, h3⟩ := not_or.1 h123
      obtain ⟨h1, h2⟩ := not_or.1 h12
      exact ha (hacc a (not_not.1 h1) (not_not.1 h2) (not_not.1 h3) (not_not.1 h4))) g honest
  exact h1.trans ((Game.prob_or_le _ _ g honest).trans (add_le_add ((Game.prob_or_le _ _ g honest).trans
    (add_le_add ((Game.prob_or_le _ _ g honest).trans (add_le_add hs hb)) hy)) hw))

end Run

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
