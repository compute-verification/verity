import Proofs.Pouw.SakLb

/-!
# The served Dither sampling block, selected or not: SAK(matmul) from its residual at the selection's rate

`Definitions.Pouw.Dither.ditherSelected` is the served design: the check coin carries a selection coin beside the cells'
coin, and a block its selection coin doesn't pick passes with no cell checked. `Proofs.Pouw.SakLb` proves the sampling
half of SAK(matmul) for `noFoldRound`, whose check runs on every coin; this file proves it for the selected block:

* `ok_selected_iff`: the check passes exactly where the block isn't picked or every checked cell is right;
* `prob_imp_eq`, `prob_imp_le`: a check picked with probability `π` that then passes with probability `ρ` passes with
  probability `1 − π + π ρ` (at most that when a coin space is empty);
* `detects_selected`: a cells' check that detects at `g` detects at `1 − π + π g` once selected;
* `pointwise_selected`, `sakMatmul_of_residual_selected`: `SAKResidual` and a cells' check that detects at `g` give
  `SAKMatmul` on the selected block at a rate with `(1 − π + π g(1 − f)) e^{η f pub} ≤ 1`;
* `sakMatmul_of_residual_bern`: with the block's `s ≥ 1` cells drawn independently and uniformly, `η · pub ≤ π`
  suffices (`SakLb.rate_bern`);
* `workSound_sakMatmul_selected`: `WorkSound`'s SAK(matmul) premise on every sampling block of a period of selected
  blocks, at `WorkSound`'s rate `η H_w / H_mm,w` against `pub_mm,w`, from the residual at `pub_mm,w + lam_w` and
  `η H_w / H_mm,w · (pub_mm,w + lam_w) ≤ π`;
* `selected_honest`, `selected_unpicked`: the honest words pass every check coin, and a block that isn't picked
  passes and is accepted on any answer, so it is never answered;
* `selected_skip`, `selected_nontrivial`: words wrong at every cell pass with probability exactly `1 − π`, so past
  `η · pub > −log(1 − π)` skipping the block beats moment 1 (`SakLb.rate_bern_sharp`), and `π` is the selected
  block's whole detection.

The other premises of `WorkSound` read the round only through `RoundProto`'s fields, so they are stated at any `P`:
`WorkSound` holds at `P = ditherSelected k Sel picked` with `sel c = picked c.1`, and the SAK(matmul) premise it takes
is, at this `P`, `SAKResidual`'s at the selection's rate. Nothing here is a premise of a guarantee: `SAKResidual` is a
proof-only definition, a hypothesis of these lemmas.
-/

namespace Pouw.SakLb

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Finset

section Selection

theorem prob_imp_eq {Sel C : Type} [Fintype Sel] [Fintype C] [Nonempty Sel] [Nonempty C] (p : Sel → Prop)
    (q : C → Prop) : prob (fun c : Sel × C => p c.1 → q c.2) = 1 - prob p + prob p * prob q := by
  classical
  unfold prob
  have hS : (0 : ℝ) < Fintype.card Sel := by exact_mod_cast Fintype.card_pos
  have hC : (0 : ℝ) < Fintype.card C := by exact_mod_cast Fintype.card_pos
  have hsplit := Finset.card_filter_add_card_filter_not (s := (univ : Finset (Sel × C)))
    (fun c : Sel × C => p c.1 → q c.2)
  have hneg : (univ.filter fun c : Sel × C => ¬ (p c.1 → q c.2)) =
      (univ.filter p) ×ˢ (univ.filter fun c => ¬ q c) := by
    ext c; simp
  have hnq := Finset.card_filter_add_card_filter_not (s := (univ : Finset C)) q
  rw [hneg, Finset.card_product, card_univ, Fintype.card_prod] at hsplit
  rw [card_univ] at hnq
  have h1 : ((univ.filter fun c : Sel × C => p c.1 → q c.2).card : ℝ) +
      (univ.filter p).card * (univ.filter fun c => ¬ q c).card = Fintype.card Sel * Fintype.card C := by
    exact_mod_cast hsplit
  have h2 : ((univ.filter q).card : ℝ) + (univ.filter fun c => ¬ q c).card = Fintype.card C := by
    exact_mod_cast hnq
  have e : (1 - ((univ.filter p).card : ℝ) / Fintype.card Sel +
      (univ.filter p).card / Fintype.card Sel * ((univ.filter q).card / Fintype.card C)) *
        (Fintype.card Sel * Fintype.card C) =
      Fintype.card Sel * Fintype.card C - (univ.filter p).card * Fintype.card C +
        (univ.filter p).card * (univ.filter q).card := by
    field_simp
  have h3 : ((univ.filter fun c : Sel × C => p c.1 → q c.2).card : ℝ) =
      Fintype.card Sel * Fintype.card C - (univ.filter p).card * Fintype.card C +
        (univ.filter p).card * (univ.filter q).card := by
    linear_combination h1 - ((univ.filter p).card : ℝ) * h2
  rw [Fintype.card_prod, Nat.cast_mul, div_eq_iff (by positivity), e]
  convert h3 using 3
  ext c; simp

theorem prob_imp_le {Sel C : Type} [Fintype Sel] [Fintype C] (p : Sel → Prop) (q : C → Prop) :
    prob (fun c : Sel × C => p c.1 → q c.2) ≤ 1 - prob p + prob p * prob q := by
  have hπ1 := prob_le_one p
  have hρ0 := prob_nonneg q
  have hρ1 := prob_le_one q
  rcases isEmpty_or_nonempty Sel with hS | hS
  · have h0 : prob (fun c : Sel × C => p c.1 → q c.2) = 0 := by unfold prob; simp
    have hp : prob p = 0 := by unfold prob; simp
    rw [h0, hp]; norm_num
  rcases isEmpty_or_nonempty C with hC | hC
  · have h0 : prob (fun c : Sel × C => p c.1 → q c.2) = 0 := by unfold prob; simp
    rw [h0]; nlinarith [prob_nonneg p]
  exact (prob_imp_eq p q).le

variable (k : NoFold) (Sel : Type) [Fintype Sel] (picked : Sel → Prop)

/-- The selected block's check passes exactly where the block isn't picked or every checked cell is right. -/
theorem ok_selected_iff (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (c : Sel × k.Chk)
    (T : (noFoldRound k).Tr) :
    (ditherSelected k Sel picked).ok a d c T ↔ (picked c.1 → ∀ q, k.cells c.2 q ∈ right k a d T) :=
  imp_congr_right fun _ => ok_iff_mem_right k a d c.2 T

/-- **The selected block detects at `1 − π + π g`** where its cells' check detects at `g`, `π` the probability that
its selection coin picks it. -/
theorem detects_selected (g : ℝ → ℝ) (hdet : Detects k g) (S : Finset (Cell k)) :
    prob (fun c : Sel × k.Chk => picked c.1 → ∀ q, k.cells c.2 q ∈ S) ≤
      1 - prob picked + prob picked * g ((S.card : ℝ) / Fintype.card (Cell k)) :=
  (prob_imp_le picked fun c : k.Chk => ∀ q, k.cells c q ∈ S).trans
    (by linarith [mul_le_mul_of_nonneg_left (hdet S) (prob_nonneg picked)])

/-- **One dither on the selected block**: a program right on a fraction `r` of the words at cost at least
`r · pub − lam`, under a cells' check that detects at `g` and a rate with `(1 − π + π g(1 − f)) e^{η f pub} ≤ 1`, has
`pass · e^{η · shortfall} ≤ 1` at `pub − lam`. -/
theorem pointwise_selected (g : ℝ → ℝ) (hdet : Detects k g) (pub lam η cost : ℝ) (hpub : 0 ≤ pub) (hη : 0 ≤ η)
    (hrate : ∀ f : ℝ, 0 ≤ f → f ≤ 1 →
      (1 - prob picked + prob picked * g (1 - f)) * Real.exp (η * (f * pub)) ≤ 1)
    (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (T : (noFoldRound k).Tr)
    (hcost : rightFrac k a d T * pub - lam ≤ cost) :
    prob (fun c => (ditherSelected k Sel picked).ok a d c T) *
      Real.exp (η * roundShortfall (pub - lam) cost) ≤ 1 := by
  set r := rightFrac k a d T with hr
  have hr0 := rightFrac_nonneg k a d T
  have hr1 := rightFrac_le_one k a d T
  have hpass : prob (fun c => (ditherSelected k Sel picked).ok a d c T) ≤
      1 - prob picked + prob picked * g (1 - (1 - r)) := by
    have hf : (fun c => (ditherSelected k Sel picked).ok a d c T) =
        fun c : Sel × k.Chk => picked c.1 → ∀ q, k.cells c.2 q ∈ right k a d T :=
      funext fun c => propext (ok_selected_iff k Sel picked a d c T)
    rw [hf, sub_sub_cancel]
    exact detects_selected k Sel picked g hdet (right k a d T)
  have hsh : roundShortfall (pub - lam) cost ≤ (1 - r) * pub := by
    unfold roundShortfall
    apply max_le (mul_nonneg (by linarith) hpub)
    linarith
  have hexp : Real.exp (η * roundShortfall (pub - lam) cost) ≤ Real.exp (η * ((1 - r) * pub)) :=
    Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left hsh hη)
  have hg := hrate (1 - r) (by linarith) (by linarith)
  have hg0 : 0 ≤ 1 - prob picked + prob picked * g (1 - (1 - r)) := le_trans (prob_nonneg _) hpass
  calc _ ≤ (1 - prob picked + prob picked * g (1 - (1 - r))) * Real.exp (η * ((1 - r) * pub)) :=
        mul_le_mul hpass hexp (Real.exp_pos _).le hg0
    _ ≤ 1 := hg

/-- **`SAKMatmul` on the selected block from its residual**: `SAKResidual` at `pub` and `lam`, a cells' check that
detects at `g` and a rate with `(1 − π + π g(1 − f)) e^{η f pub} ≤ 1` at every fraction `f` give `SAKMatmul` at
`pub − lam` on `ditherSelected`. -/
theorem sakMatmul_of_residual_selected (G : Accel) (μ : G.Res → ℝ) (Reg : Set (noFoldRound k).Inst)
    (pub lam η : (noFoldRound k).Inst → ℝ) (κdeg : ℕ → ℝ) (g : ℝ → ℝ)
    (hres : SAKResidual G k μ Reg pub lam κdeg) (hdet : Detects k g)
    (hrate : ∀ a ∈ Reg, ∀ f : ℝ, 0 ≤ f → f ≤ 1 →
      (1 - prob picked + prob picked * g (1 - f)) * Real.exp (η a * (f * pub a)) ≤ 1)
    (hpub : ∀ a ∈ Reg, 0 ≤ pub a) (hη : ∀ a ∈ Reg, 0 ≤ η a) :
    SAKMatmul G (ditherSelected k Sel picked) μ Reg (fun a => pub a - lam a) η κdeg := by
  intro g1 g2 p1 p2 a ha B St hSt s Mw hext
  obtain ⟨Deg, hDeg, hcost⟩ := hres g1 g2 p1 p2 a ha B St hSt s Mw hext
  refine ⟨Deg, hDeg, wordsMoment_le_one G _ μ _ _ a s Mw Deg fun d hd => ?_⟩
  exact pointwise_selected k Sel picked g hdet (pub a) (lam a) (η a) _ (hpub a ha) (hη a ha) (hrate a ha) a d _
    (hcost d hd)

/-- **`SAKMatmul` on the selected block at its Bernoulli rate**: with the block's `s ≥ 1` cells drawn independently
and uniformly, `η · pub ≤ π` suffices (`rate_bern`; past `−log(1 − π)` it fails, `selected_nontrivial`). -/
theorem sakMatmul_of_residual_bern (G : Accel) (μ : G.Res → ℝ) (Reg : Set (noFoldRound k).Inst)
    (pub lam η : (noFoldRound k).Inst → ℝ) (κdeg : ℕ → ℝ) (hres : SAKResidual G k μ Reg pub lam κdeg)
    (e : k.Chk ≃ (Fin k.s → Cell k)) (he : ∀ c q, k.cells c q = e c q) (hs : 1 ≤ k.s)
    (hrate : ∀ a ∈ Reg, η a * pub a ≤ prob picked) (hpub : ∀ a ∈ Reg, 0 ≤ pub a) (hη : ∀ a ∈ Reg, 0 ≤ η a) :
    SAKMatmul G (ditherSelected k Sel picked) μ Reg (fun a => pub a - lam a) η κdeg :=
  sakMatmul_of_residual_selected k Sel picked G μ Reg pub lam η κdeg _ hres (detects_iid k e he)
    (fun a ha f hf0 hf1 => by
      have h := rate_bern k.s hs (prob picked) (η a * pub a) (prob_nonneg picked) (hrate a ha) f hf0 hf1
      have he' : η a * (f * pub a) = η a * pub a * f := by ring
      rw [he']
      exact h)
    hpub hη

/-- **`WorkSound`'s SAK(matmul) premise on a period of selected blocks**: on every sampling block `w`, at
`WorkSound`'s rate `η H_w / H_mm,w` against `pub_mm,w`, from the residual at `pub_mm,w + lam_w` and the selection's
rate `η H_w / H_mm,w · (pub_mm,w + lam_w) ≤ π`. At `pub_mm ≤ H_mm` and `lam = 0` that is `π ≥ η H_w`: the block is
picked with probability at least `q H_w / C` at `η = q / C`, proportional to its cost. -/
theorem workSound_sakMatmul_selected (G : Accel) {nsb : ℕ} (μ : G.Res → ℝ)
    (Reg : Fin nsb → Set (noFoldRound k).Inst) (Hmm Hsha Ho pubmm lam : Fin nsb → ℝ) (η : ℝ) (κdmm : ℕ → ℝ)
    (hres : ∀ w, SAKResidual G k μ (Reg w) (fun _ => pubmm w + lam w) (fun _ => lam w) κdmm)
    (e : k.Chk ≃ (Fin k.s → Cell k)) (he : ∀ c q, k.cells c q = e c q) (hs : 1 ≤ k.s)
    (hH : ∀ w, 0 < Hmm w ∧ 0 < Hsha w ∧ 0 ≤ Ho w) (hpub : ∀ w, 0 ≤ pubmm w + lam w) (hη : 0 ≤ η)
    (hrate : ∀ w, η * (Hmm w + Hsha w + Ho w) / Hmm w * (pubmm w + lam w) ≤ prob picked) :
    ∀ w, SAKMatmul G (ditherSelected k Sel picked) μ (Reg w) (fun _ => pubmm w)
      (fun _ => η * (Hmm w + Hsha w + Ho w) / Hmm w) κdmm := by
  intro w
  obtain ⟨h1, h2, h3⟩ := hH w
  have h := sakMatmul_of_residual_bern k Sel picked G μ (Reg w) (fun _ => pubmm w + lam w) (fun _ => lam w)
    (fun _ => η * (Hmm w + Hsha w + Ho w) / Hmm w) κdmm (hres w) e he hs (fun _ _ => hrate w)
    (fun _ _ => hpub w) (fun _ _ => div_nonneg (mul_nonneg hη (by linarith)) h1.le)
  simp only [add_sub_cancel_right] at h
  exact h

/-- **Honest**: the honest words pass every check coin of the selected block. -/
theorem selected_honest (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (c : Sel × k.Chk) :
    (ditherSelected k Sel picked).ok a d c (noFoldHonest k a d) :=
  fun _ _ => rfl

/-- A block its selection coin doesn't pick passes, opens and is accepted on any words, root and answer, so it is
never answered. -/
theorem selected_unpicked (o : k.Orc) (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (c : Sel × k.Chk)
    (hc : ¬ picked c.1) (T : (noFoldRound k).Tr) (r : k.Root) (y : (noFoldRound k).Ans) :
    (ditherSelected k Sel picked).ok a d c T ∧ (ditherSelected k Sel picked).opens o r c T ∧
      (ditherSelected k Sel picked).accept o a d c r y :=
  ⟨fun h => absurd h hc, fun h => absurd h hc, fun h => absurd h hc⟩

/-- **Skipping the block**: words wrong at every cell pass the selected block with probability exactly `1 − π`, at
`s ≥ 1` cells. -/
theorem selected_skip [Nonempty Sel] [Nonempty k.Chk] (hs : 1 ≤ k.s) (a : (noFoldRound k).Inst)
    (d : (noFoldRound k).Draw) (T : (noFoldRound k).Tr) (hT : right k a d T = ∅) :
    prob (fun c => (ditherSelected k Sel picked).ok a d c T) = 1 - prob picked := by
  have hf : (fun c : Sel × k.Chk => picked c.1 → (noFoldRound k).ok a d c.2 T) =
      fun c : Sel × k.Chk => picked c.1 → (fun _ : k.Chk => False) c.2 := by
    funext c
    refine propext (imp_congr_right fun _ => ?_)
    rw [ok_iff_mem_right, hT]
    exact ⟨fun h => Finset.notMem_empty _ (h ⟨0, hs⟩), False.elim⟩
  have h0 : prob (fun _ : k.Chk => False) = 0 := by unfold prob; simp
  have h := prob_imp_eq picked (fun _ : k.Chk => False)
  rw [h0, mul_zero, add_zero] at h
  show prob (fun c : Sel × k.Chk => picked c.1 → (noFoldRound k).ok a d c.2 T) = _
  rw [hf]
  exact h

/-- **Nontrivial**: the selected block's rate can't pass `−log(1 − π)`. At `s ≥ 1` cells and `π < 1`, the words
that are the honest ones plus one at every cell pass with probability `1 − π`, and at any rate `x > −log(1 − π)`
their moment `(1 − π) e^x` (shortfall `pub` at no cost, `x = η · pub`) exceeds 1, so `sakMatmul_of_residual_bern`'s
`η · pub ≤ π` is within a constant of the best rate. -/
theorem selected_nontrivial [Nonempty Sel] [Nonempty k.Chk] (hs : 1 ≤ k.s) (hπ : prob picked < 1)
    (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) (x : ℝ) (hx : -Real.log (1 - prob picked) < x) :
    1 < prob (fun c => (ditherSelected k Sel picked).ok a d c (fun i j b => noFoldHonest k a d i j b + 1)) *
      Real.exp x := by
  have hT : right k a d (fun i j b => noFoldHonest k a d i j b + 1) = ∅ := by
    refine Finset.filter_false_of_mem fun y _ h => ?_
    have h1 : (1 : Word) = 0 := by simpa [noFoldHonest] using h
    exact absurd h1 (by decide)
  rw [selected_skip k Sel picked hs a d _ hT]
  have h := rate_bern_sharp k.s hs (prob picked) x hπ hx
  rw [sub_self, zero_pow (by omega), mul_zero, add_zero, mul_one] at h
  exact h

end Selection

end Pouw.SakLb
