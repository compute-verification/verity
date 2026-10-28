import Definitions.Nci.Guarantees.Info
import Proofs.Nci.Support
import Proofs.Nci.Threads

/-!
# Proofs of NCI's guarantees with reads counted by information

A set of gates takes at most `2^width` values (`support_le_width`), so `novelInfo ≤ novelReads`
(`novelInfo_le_novelReads`), with equality when the novel reads are novel input gates, which range freely
(`novelInfo_eq_novelReads_of_inputs`). So `WeightsSupport` gives `WeightsIncompressible`
(`weightsIncompressible_of_support`), and the two say the same at a subcircuit whose novel reads are novel input gates
(`weightsSupport_iff_of_inputs`).

Theorem 1's proof reads `novelReads` only through incompressibility and the cap `S`, so it carries over with `novelInfo`
in its place (`threaded_count`, `threaded_reads_info`). A run of `c` lifts to a run of a declaration that contains it,
with the baseline at its committed values (`liftRun`), so the piece of `c` inside a set `W` of `d`'s gates reads no more
information than `W` (`novelInfo_le_of_containment`). Supports multiply at most over a union (`support_union_le`), and a
function of `k` bits takes at most `2^k` values (`support_le_of_factor`): that bounds a set read by a state and an
ingress (`InfoOfState`). A set of gates fed by no novel input outside `P` takes at most `2^width(P)` values
(`support_le_of_feeds`).
-/

namespace Nci.SecurityProofs

open Verity Finset

section Support

variable (d : Registered) (θv : d.D.Transcript)

theorem mem_admissible_self : θv ∈ admissible d θv :=
  mem_filter.2 ⟨mem_univ _, fun _ _ => rfl⟩

theorem support_pos (T : Finset (Fin d.D.N)) : 0 < support d θv T := by
  classical
  unfold support
  exact card_pos.2 ⟨_, mem_image_of_mem _ (mem_admissible_self d θv)⟩

theorem support_mono {T T' : Finset (Fin d.D.N)} (h : T ⊆ T') : support d θv T ≤ support d θv T' := by
  classical
  unfold support
  exact card_image_le_of_factor fun a _ b _ hab => funext fun g => congrFun hab ⟨g.1, h g.2⟩

theorem support_union_le (A B : Finset (Fin d.D.N)) :
    support d θv (A ∪ B) ≤ support d θv A * support d θv B := by
  classical
  unfold support
  calc ((admissible d θv).image (values (A ∪ B))).card
      ≤ ((admissible d θv).image fun a => (values A a, values B a)).card :=
        card_image_le_of_factor fun a _ b _ hab => funext fun g => by
          rcases mem_union.1 g.2 with h | h
          · exact congrFun (congrArg Prod.fst hab) ⟨g.1, h⟩
          · exact congrFun (congrArg Prod.snd hab) ⟨g.1, h⟩
    _ ≤ ((admissible d θv).image (values A) ×ˢ (admissible d θv).image (values B)).card :=
        card_le_card fun x hx => by
          obtain ⟨a, ha, rfl⟩ := mem_image.1 hx
          exact mem_product.2 ⟨mem_image_of_mem _ ha, mem_image_of_mem _ ha⟩
    _ = _ := card_product _ _

/-- The gates of `T` take at most `2^k` values when their values on every run are those of a function of `k` bits. -/
theorem support_le_of_factor {T : Finset (Fin d.D.N)} {k : ℕ} (f : BitVec k → (g : T) → BitVec (d.D.w g))
    (h : ∀ a ∈ admissible d θv, ∃ s, values T a = f s) : support d θv T ≤ 2 ^ k := by
  classical
  unfold support
  calc ((admissible d θv).image (values T)).card ≤ (univ.image f).card :=
        card_le_card fun x hx => by
          obtain ⟨a, ha, rfl⟩ := mem_image.1 hx
          obtain ⟨s, hs⟩ := h a ha
          rw [hs]
          exact mem_image_of_mem f (mem_univ s)
    _ ≤ univ.card := card_image_le
    _ = 2 ^ k := by rw [card_univ, card_bitVec]

/-- The gates of a subset of `R` take no more values than `R`'s novel reads. -/
theorem support_le_support_into {R T : Finset (Fin d.D.N)} (hR : R ⊆ d.D.computed) (hT : T ⊆ R) :
    support d θv T ≤ support d θv (d.D.into R \ d.θ) := by
  classical
  unfold support
  refine card_image_le_of_factor fun a ha b hb hab => funext fun g => ?_
  have ha' := (mem_filter.mp ha).2
  have hb' := (mem_filter.mp hb).2
  refine eval_agree_of_into hR (fun x hx => ?_) g (hT g.2)
  by_cases hθ : x ∈ d.θ
  · rw [d.D.eval_input a (d.θ_input x hθ), d.D.eval_input b (d.θ_input x hθ), ha' x hθ, hb' x hθ]
  · exact congrFun hab ⟨x, mem_sdiff.mpr ⟨hx, hθ⟩⟩

/-- Novel input gates take every value. -/
theorem support_eq_of_inputs {U : Finset (Fin d.D.N)} (hU : ∀ g ∈ U, d.D.isInput g = true ∧ g ∉ d.θ) :
    support d θv U = 2 ^ d.D.width U := by
  classical
  have himg : (admissible d θv).image (values U) = univ := by
    refine eq_univ_of_forall fun p => mem_image.2 ⟨fun h => if hh : h ∈ U then p ⟨h, hh⟩ else θv h, ?_, ?_⟩
    · exact mem_filter.2 ⟨mem_univ _, fun g hg => dif_neg fun h => (hU g h).2 hg⟩
    · funext g
      show d.D.eval _ g = p g
      rw [d.D.eval_input _ (hU g g.2).1]
      exact dif_pos g.2
  unfold support
  rw [himg, card_univ, Fintype.card_pi]
  simp only [card_bitVec]
  rw [prod_coe_sort U fun g => 2 ^ d.D.w g, prod_pow_eq_pow_sum]
  rfl

end Support

section Info

variable (d : Registered) (θv : d.D.Transcript) (R : Finset (Fin d.D.N))

theorem novelInfo_le_iff (x : ℝ) :
    novelInfo d θv R ≤ x ↔ (support d θv (d.D.into R \ d.θ) : ℝ) ≤ 2 ^ x :=
  Real.logb_le_iff_le_rpow (by norm_num) (Nat.cast_pos.2 (support_pos d θv _))

theorem le_novelInfo_iff (x : ℝ) :
    x ≤ novelInfo d θv R ↔ (2 : ℝ) ^ x ≤ support d θv (d.D.into R \ d.θ) :=
  Real.le_logb_iff_rpow_le (by norm_num) (Nat.cast_pos.2 (support_pos d θv _))

/-- **Information is at most width.** -/
theorem novelInfo_le_novelReads : novelInfo d θv R ≤ novelReads d R := by
  rw [novelInfo_le_iff, Real.rpow_natCast]
  have := support_le_width d θv (d.D.into R \ d.θ)
  unfold novelReads
  exact_mod_cast this

/-- **Information is width on novel inputs**: when `R`'s novel reads are novel input gates. -/
theorem novelInfo_eq_novelReads_of_inputs (hU : ∀ g ∈ d.D.into R \ d.θ, d.D.isInput g = true) :
    novelInfo d θv R = novelReads d R := by
  have hs := support_eq_of_inputs d θv fun g hg => ⟨hU g hg, (mem_sdiff.1 hg).2⟩
  unfold novelInfo novelReads
  rw [hs, Nat.cast_pow, Nat.cast_ofNat, ← Real.rpow_natCast, Real.logb_rpow (by norm_num) (by norm_num)]

end Info

theorem smallInfo_of_small {c : Registered} (θv : c.D.Transcript) {Γ : ℕ} {S : ℝ} {R : Finset (Fin c.D.N)}
    (hR : Small c Γ S R) : SmallInfo c θv Γ S R :=
  ⟨hR.1, hR.2.1, (novelInfo_le_novelReads c θv R).trans hR.2.2⟩

/-- **Support form gives the width form**: `WeightsSupport` gives `WeightsIncompressible` at the same cap and rates. -/
theorem weightsIncompressible_of_support {c : Registered} {θv : c.D.Transcript} {T : Threading c} {Γ : ℕ}
    {S b_w b_a : ℝ} (h : Assumptions.WeightsSupport c θv T Γ S b_w b_a) :
    Assumptions.WeightsIncompressible c T Γ S b_w b_a := fun R hR =>
  ((le_novelInfo_iff c θv R _).2 (h R (smallInfo_of_small θv hR))).trans (novelInfo_le_novelReads c θv R)

/-- **Where the two forms agree**: at a subcircuit whose novel reads are novel input gates, `WeightsSupport`'s
inequality is `WeightsIncompressible`'s. -/
theorem weightsSupport_iff_of_inputs {c : Registered} (θv : c.D.Transcript) (T : Threading c) (b_w b_a : ℝ)
    (R : Finset (Fin c.D.N)) (hU : ∀ g ∈ c.D.into R \ c.θ, c.D.isInput g = true) :
    (2 : ℝ) ^ (b_w * (T.weightsUsed R).card + b_a * ∑ t, (T.threadReads R t).card) ≤
        support c θv (c.D.into R \ c.θ) ↔
      b_w * (T.weightsUsed R).card + b_a * ∑ t, (T.threadReads R t).card ≤ novelReads c R := by
  rw [← le_novelInfo_iff, novelInfo_eq_novelReads_of_inputs c θv R hU]

section Threads

variable {c : Registered} (T : Threading c) (θv : c.D.Transcript) (Γ : ℕ) (S : ℝ) (d e : ℕ)

/-- Threading by information at any cap follows from threading by width at a cap no subcircuit reaches. -/
theorem threadedInfo_of_threaded (h : T.Threaded Γ (c.D.width univ) d e) : T.ThreadedInfo θv Γ S d e := by
  intro R hR
  have hle : novelReads c R ≤ c.D.width univ := by
    unfold novelReads Circuit.width
    exact sum_le_sum_of_subset (subset_univ _)
  exact h R ⟨hR.1, hR.2.1, by exact_mod_cast hle⟩

theorem threaded_of_threadedInfo (h : T.ThreadedInfo θv Γ S d e) : T.Threaded Γ S d e :=
  fun R hR => h R (smallInfo_of_small θv hR)

/-- **`(1, 0)`-threaded by information from thread closure**, as `threaded_one_zero`. -/
theorem threadedInfo_one_zero (hθ : ∀ g t, T.thread g = some t → g ∉ c.θ)
    (hclosed : ∀ h t, c.D.isInput h = false → T.thread h = some t → ∃ g ∈ c.D.reads h, T.thread g = some t) :
    T.ThreadedInfo θv Γ S 1 0 :=
  threadedInfo_of_threaded T θv Γ S 1 0 (threaded_one_zero T Γ _ hθ hclosed)

end Threads

/-- **Theorem 1 with reads counted by information.** -/
theorem threaded_reads_info (c : Registered) (θv : c.D.Transcript) (T : Threading c) (Γ : ℕ) (S b_w b_a : ℝ)
    (d e : ℕ) (hbw : 0 < b_w) (hba : 0 < b_a) (hd : 0 < d) (hT : T.ThreadedInfo θv Γ S d e)
    (hW : Assumptions.WeightsSupport c θv T Γ S b_w b_a) (R : Finset (Fin c.D.N)) (hR : SmallInfo c θv Γ S R) :
    b_a * (T.mults R).card ≤ novelInfo c θv R * T.bound S b_w d e :=
  threaded_count T R hbw hba hd (hT R hR) ((le_novelInfo_iff c θv R _).2 (hW R hR)) hR.2.2

theorem ThreadedReadsInfo : Guarantees.ThreadedReadsInfo := threaded_reads_info

open Classical in
/-- A run of `C` lifted through `M` to a run of `D`: `b` at the images of `C`'s gates, `θv` elsewhere. -/
noncomputable def liftRun {C D : Circuit} (M : Containment C D) (θv : D.Transcript) (b : C.Transcript)
    (h : Fin D.N) : BitVec (D.w h) :=
  if hh : ∃ g, M.emb g = h then BitVec.ofNat (D.w h) (b hh.choose).toNat else θv h

theorem liftRun_emb {C D : Circuit} (M : Containment C D) (θv : D.Transcript) (b : C.Transcript) (g : Fin C.N) :
    liftRun M θv b (M.emb g) = BitVec.ofNat (D.w (M.emb g)) (b g).toNat := by
  have hh : ∃ g', M.emb g' = M.emb g := ⟨g, rfl⟩
  have hc : hh.choose = g := M.emb_inj hh.choose_spec
  rw [liftRun, dif_pos hh, hc]

theorem pull_liftRun {C D : Circuit} (M : Containment C D) (θv : D.Transcript) (b : C.Transcript) :
    M.pull (liftRun M θv b) = b := by
  funext g
  apply BitVec.eq_of_toNat_eq
  show ((liftRun M θv b (M.emb g)).cast (M.emb_w g)).toNat = (b g).toNat
  rw [BitVec.toNat_cast, liftRun_emb, BitVec.toNat_ofNat, M.emb_w g]
  exact Nat.mod_eq_of_lt (b g).isLt

/-- Through a containment, `c`'s gates take no more values over its runs with the baseline at `θc` than their images
over `d`'s with the baseline at `θv`, when `θc` is what `θv` gives the images of `c`'s baseline. -/
theorem support_le_of_containment {c d : Registered} (M : Containment c.D d.D)
    (hθ : ∀ g, c.D.isInput g = true → M.emb g ∈ d.θ → g ∈ c.θ) {θc : c.D.Transcript} {θv : d.D.Transcript}
    (hv : ∀ g ∈ c.θ, M.emb g ∈ d.θ → (θv (M.emb g)).cast (M.emb_w g) = θc g)
    {U : Finset (Fin c.D.N)} {U' : Finset (Fin d.D.N)} (hU : ∀ g ∈ U, M.emb g ∈ U') :
    support c θc U ≤ support d θv U' := by
  classical
  let φ : ((g : U') → BitVec (d.D.w g)) → (g : U) → BitVec (c.D.w g) :=
    fun p g => (p ⟨M.emb g.1, hU g.1 g.2⟩).cast (M.emb_w g.1)
  unfold support
  calc ((admissible c θc).image (values U)).card ≤ (((admissible d θv).image (values U')).image φ).card := by
        refine card_le_card fun x hx => ?_
        obtain ⟨b, hb, rfl⟩ := mem_image.1 hx
        refine mem_image.2 ⟨values U' (liftRun M θv b), mem_image_of_mem _ ?_, ?_⟩
        · refine mem_filter.2 ⟨mem_univ _, fun h hh => ?_⟩
          by_cases hr : ∃ g, M.emb g = h
          · obtain ⟨g, rfl⟩ := hr
            have hg : g ∈ c.θ := hθ g (by rw [← M.isInput_emb]; exact d.θ_input _ hh) hh
            apply BitVec.eq_of_toNat_eq
            rw [liftRun_emb, BitVec.toNat_ofNat, (mem_filter.1 hb).2 g hg, ← hv g hg hh, BitVec.toNat_cast]
            exact Nat.mod_eq_of_lt (θv (M.emb g)).isLt
          · rw [liftRun, dif_neg hr]
        · funext g
          show (d.D.eval (liftRun M θv b) (M.emb g.1)).cast (M.emb_w g.1) = c.D.eval b g.1
          rw [M.eval_emb, pull_liftRun]
    _ ≤ ((admissible d θv).image (values U')).card := card_image_le

/-- A novel read of the piece of `c` inside `W` is, through the containment, a novel read of `W`. -/
theorem emb_into_filter {c d : Registered} (M : Containment c.D d.D)
    (hθ : ∀ g, c.D.isInput g = true → M.emb g ∈ d.θ → g ∈ c.θ) {W : Finset (Fin d.D.N)} (hW : W ⊆ d.D.computed) :
    ∀ g ∈ c.D.into (univ.filter fun g => M.emb g ∈ W) \ c.θ, M.emb g ∈ d.D.into W \ d.θ := by
  intro a ha
  rw [mem_sdiff] at ha ⊢
  obtain ⟨ha, haθ⟩ := ha
  simp only [Circuit.into, mem_filter, mem_univ, true_and] at ha ⊢
  obtain ⟨hna, g, hg, hag⟩ := ha
  have hgi : c.D.isInput g = false := by
    rw [← M.isInput_emb]
    exact (mem_filter.1 (hW hg)).2
  refine ⟨⟨hna, M.emb g, hg, ?_⟩, fun h => haθ (hθ a ?_ h)⟩
  · rw [M.reads_emb g hgi]
    exact mem_map_of_mem _ hag
  · rw [← M.isInput_emb]
    exact d.θ_input _ h

/-- **Information through a containment**: the piece of `c` inside a set `W` of `d`'s computed gates reads no more
information than `W`. -/
theorem novelInfo_le_of_containment {c d : Registered} (M : Containment c.D d.D)
    (hθ : ∀ g, c.D.isInput g = true → M.emb g ∈ d.θ → g ∈ c.θ) {θc : c.D.Transcript} {θv : d.D.Transcript}
    (hv : ∀ g ∈ c.θ, M.emb g ∈ d.θ → (θv (M.emb g)).cast (M.emb_w g) = θc g) {W : Finset (Fin d.D.N)}
    (hW : W ⊆ d.D.computed) : novelInfo c θc (univ.filter fun g => M.emb g ∈ W) ≤ novelInfo d θv W := by
  unfold novelInfo
  refine Real.logb_le_logb_of_le (by norm_num) (Nat.cast_pos.2 (support_pos c θc _)) ?_
  exact_mod_cast support_le_of_containment M hθ hv (emb_into_filter M hθ hW)

theorem ThreadedContainedInfo : Guarantees.ThreadedContainedInfo := by
  intro c d M θc θv T Γ K b_w b_a dd e hθ hv hbw hba hdd hT hW W hWc hΓ hK
  have hinfo := novelInfo_le_of_containment M hθ hv hWc
  have hPc : (univ.filter fun g => M.emb g ∈ W) ⊆ c.D.computed := fun g hg => by
    have := (mem_filter.1 (hWc (mem_filter.1 hg).2)).2
    rw [M.isInput_emb] at this
    exact mem_filter.2 ⟨mem_univ _, this⟩
  have hPs : c.D.size (univ.filter fun g => M.emb g ∈ W) ≤ Γ := by
    refine le_trans ?_ hΓ
    unfold Circuit.size
    rw [← card_map ⟨M.emb, M.emb_inj⟩]
    exact card_le_card fun x hx => by
      obtain ⟨g, hg, rfl⟩ := mem_map.1 hx
      exact (mem_filter.1 hg).2
  calc b_a * ((T.mults (univ.filter fun g => M.emb g ∈ W)).card : ℝ)
      ≤ novelInfo c θc (univ.filter fun g => M.emb g ∈ W) * T.bound K b_w dd e :=
        threaded_reads_info c θc T Γ K b_w b_a dd e hbw hba hdd hT hW _ ⟨hPc, hPs, hinfo.trans hK⟩
    _ ≤ novelInfo d θv W * T.bound K b_w dd e :=
        mul_le_mul_of_nonneg_right hinfo (le_max_of_le_left (Nat.cast_nonneg _))

theorem InfoOfState : Guarantees.InfoOfState := by
  intro D θv R Sst Sin k f hR hin hf
  have hs : support D θv (D.D.into R \ D.θ) ≤ 2 ^ (k + D.D.width Sin) :=
    calc support D θv (D.D.into R \ D.θ) ≤ support D θv (Sst ∪ Sin) := support_mono D θv hin
      _ ≤ support D θv Sst * support D θv Sin := support_union_le D θv Sst Sin
      _ ≤ 2 ^ k * 2 ^ D.D.width Sin := Nat.mul_le_mul (support_le_of_factor D θv f hf) (support_le_width D θv Sin)
      _ = 2 ^ (k + D.D.width Sin) := (pow_add 2 k _).symm
  refine ⟨(support_le_support_into D θv hR (subset_refl R)).trans hs, ?_⟩
  rw [novelInfo_le_iff, ← Nat.cast_add, Real.rpow_natCast]
  exact_mod_cast hs

theorem ThreadedOfState : Guarantees.ThreadedOfState := by
  intro c d M θc θv T Γ b_w b_a dd e W Sst Sin k f hθ hv hbw hba hdd hT hW hWc hΓ hin hf
  have hK := (InfoOfState d θv W Sst Sin k f hWc hin hf).2
  exact (ThreadedContainedInfo c d M θc θv T Γ _ b_w b_a dd e hθ hv hbw hba hdd hT hW W hWc hΓ hK).trans
    (mul_le_mul_of_nonneg_right hK (le_max_of_le_left (Nat.cast_nonneg _)))

section Feeds

theorem feeds_of_read {D : Circuit} {g x y : Fin D.N} (hg : D.Feeds g x) (hx : x ∈ D.reads y) : D.Feeds g y := by
  revert hx
  induction hg with
  | refl => exact fun hx => .step hx (.refl y)
  | step hgh _ ih => exact fun hx => .step hgh (ih hx)

/-- A gate's value is determined by the input gates that feed it. -/
theorem eval_agree_of_feeds {D : Circuit} {a b : D.Transcript} :
    ∀ t, (∀ g, D.isInput g = true → D.Feeds g t → a g = b g) → D.eval a t = D.eval b t := by
  intro t
  induction t using WellFoundedLT.induction with
  | _ t ih => ?_
  intro h
  cases hi : D.isInput t
  · rw [D.eval_op a hi, D.eval_op b hi]
    exact D.op_congr t _ _ fun x hx =>
      ih x (D.reads_lt t x hx) fun g hg hf => h g hg (feeds_of_read hf hx)
  · rw [D.eval_input a hi, D.eval_input b hi]
    exact h t hi (.refl t)

/-- **What a set's values carry when novel inputs `P` alone feed it**: at most `2^width(P)` values, whatever the gates
compute. A declared value that is a function of the baseline and the prompts that feed it carries at most their
width. -/
theorem support_le_of_feeds (d : Registered) (θv : d.D.Transcript) {F P : Finset (Fin d.D.N)}
    (h : ∀ g, ∀ t ∈ F, d.D.isInput g = true → g ∉ d.θ → d.D.Feeds g t → g ∈ P) :
    support d θv F ≤ 2 ^ d.D.width P := by
  classical
  refine le_trans ?_ (support_le_width d θv P)
  unfold support
  refine card_image_le_of_factor fun a ha b hb hab => funext fun t => eval_agree_of_feeds t.1 fun g hg hf => ?_
  by_cases hθ : g ∈ d.θ
  · rw [(mem_filter.1 ha).2 g hθ, (mem_filter.1 hb).2 g hθ]
  · have hab' : d.D.eval a g = d.D.eval b g := congrFun hab ⟨g, h g t.1 t.2 hg hθ hf⟩
    rwa [d.D.eval_input a hg, d.D.eval_input b hg] at hab'

end Feeds

end Nci.SecurityProofs
