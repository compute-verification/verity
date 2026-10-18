import Proofs.Accounting.Headline.Luck
import Proofs.Accounting.TS.ProofsSound
import Models.Accounting.Guarantees.Headline

/-!
# RT-HEADLINE scratch (never lands): what #2482's headlines say about a cheater with a wrong unit

* (A) `accepts_wrong_le`: in 𝒫_full the headline's proof-layer premises (`Of`, binding, `SoundAt`, `Pins`) bound the
  probability of an accepted window whose committed values are not right by `S·(δ_bind + n_π·δ_ZK)`. So `RightFixed`'s
  antecedent holds on every accepted path off that event: the headline is not vacuous for a wrong-unit cheater.
* (C) `sampled_not_luck`, `sampled_no_split`: a sampled round (one wrong unit of 16, drawn w.p. 1/16) passes w.p. 15/16,
  so no receipt split meets `Pinned`, `CubeOperands` and `ReceiptLuckAt ½` for it: the headline can't be instantiated
  on 𝒫_samp with an undrawn wrong unit.
* (D) `adapt_*`: a skipper whose share reads the round's cube bit (skip iff the bit is set, always right) meets `Pinned`
  and `ReceiptLuckAt ½` but no `LuckAt p` for `p < 1`; so `CubeOperands` is the premise that excludes it.
* (E) `lift_*`: every play's game lifts to a receipt game meeting `Pl.game = G.game`, `Pinned` and `ReceiptLuckAt` at
  any `pr`; with that split `CubeOperands` forces the round honest. The three are free alone; the luck's content is in
  `CubeOperands` together with the split the modeler picks.
-/

namespace RTHeadline

open Accounting.TS Accounting.Assumptions.TS Accounting.SecurityProofs.TS Pouw.RelativeSecurity

theorem prob_of_false {Ω : Type} [Fintype Ω] (p : Ω → Prop) (h : ∀ ω, ¬ p ω) : prob p = 0 := by
  classical
  unfold prob
  rw [Finset.filter_false_of_mem (fun ω _ => h ω)]
  simp

theorem prob_pos_of {Ω : Type} [Fintype Ω] (p : Ω → Prop) (ω : Ω) (h : p ω) : 0 < prob p := by
  classical
  unfold prob
  apply div_pos
  · exact_mod_cast Finset.card_pos.mpr ⟨ω, Finset.mem_filter.mpr ⟨Finset.mem_univ _, h⟩⟩
  · exact_mod_cast Fintype.card_pos_iff.mpr ⟨ω⟩

/-! ## (A) 𝒫_full: accepted with a wrong committed value is the crypto event -/

theorem accepts_wrong_le {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V K T L U : Type} (P : Play Ω S K T L U)
    (pf : ProofLayer Ω S nπ V) (δbind δzk : ℝ) (hOf : pf.Of P)
    (hB : ∀ w, prob (fun x => pf.Moved x w) ≤ δbind) (hZ : pf.SoundAt δzk) (hPin : pf.Pins) :
    prob (fun x => P.accepts x ∧ ¬ pf.right (pf.committed x)) ≤ S * (δbind + nπ * δzk) := by
  have hw : ∀ w, prob (fun x => pf.Moved x w ∨ ∃ i ∈ (Finset.univ : Finset (Fin nπ)), pf.Unsound x w i) ≤
      δbind + nπ * δzk := fun w => by
    refine (prob_or_le _ _).trans (add_le_add (hB w) ((prob_exists_le _ _).trans ?_))
    calc ∑ i : Fin nπ, prob (fun x => pf.Unsound x w i) ≤ ∑ _i : Fin nπ, δzk :=
          Finset.sum_le_sum fun i _ => hZ w i
      _ = nπ * δzk := by simp
  have hE : prob (fun x => ∃ w ∈ (Finset.univ : Finset (Fin S)),
      pf.Moved x w ∨ ∃ i ∈ (Finset.univ : Finset (Fin nπ)), pf.Unsound x w i) ≤ S * (δbind + nπ * δzk) := by
    calc prob _ ≤ ∑ w : Fin S, prob (fun x => pf.Moved x w ∨
          ∃ i ∈ (Finset.univ : Finset (Fin nπ)), pf.Unsound x w i) := prob_exists_le _ _
      _ ≤ ∑ _w : Fin S, (δbind + nπ * δzk) := Finset.sum_le_sum fun w _ => hw w
      _ = S * (δbind + nπ * δzk) := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  refine le_trans (prob_mono fun x hx => ?_) hE
  obtain ⟨hacc, hr⟩ := hx
  by_contra hE'
  have hM : ∀ w, ¬ pf.Moved x w := fun w h => hE' ⟨w, Finset.mem_univ _, Or.inl h⟩
  have hU : ∀ w i, ¬ pf.Unsound x w i := fun w i h =>
    hE' ⟨w, Finset.mem_univ _, Or.inr ⟨i, Finset.mem_univ _, h⟩⟩
  exact hr (hPin _ (rel_committed pf x (hOf.1 x hacc) hM hU))

/-- At the headline's budget, `checks·(δ_bind + n_π·δ_ZK) ≤ 2^-41`, a wrong-unit cheater is accepted w.p. ≤ 2^-41. -/
theorem accepts_wrong_le_budget {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V K T L U : Type} (P : Play Ω S K T L U)
    (pf : ProofLayer Ω S nπ V) (δbind δzk : ℝ) (hbud : (S : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41)
    (hOf : pf.Of P) (hB : ∀ w, prob (fun x => pf.Moved x w) ≤ δbind) (hZ : pf.SoundAt δzk) (hPin : pf.Pins) :
    prob (fun x => P.accepts x ∧ ¬ pf.right (pf.committed x)) ≤ (1 / 2) ^ 41 :=
  (accepts_wrong_le P pf δbind δzk hOf hB hZ hPin).trans hbud

/-! ## (C) 𝒫_samp: one wrong unit of 16, drawn with probability 1/16 -/

abbrev Ω16 : Type := Fin 16 × Finset (Fin 1)

/-- The sampled round: the wrong unit's share 1/16, and the round passes unless the draw is the wrong unit. -/
noncomputable def sampledGame : RoundGame Ω16 1 := .round (fun _ => 1 / 16) (fun ω => ω.1 ≠ 0) (fun _ => .done)

theorem prob_undrawn : prob (fun ω : Ω16 => ω.1 ≠ 0) = 15 / 16 := by
  rw [prob_fst (fun a : Fin 16 => a ≠ 0), prob_eq_card]
  have h1 : (Finset.univ.filter fun a : Fin 16 => a ≠ 0).card = 15 := by decide
  rw [h1]
  norm_num [Fintype.card_fin]

theorem sampled_not_luck : ¬ LuckAt (7 / 8) sampledGame := by
  intro h
  obtain ⟨-, h2, -⟩ := h
  have e : (fun ω : Ω16 => ω.1 ≠ 0 ∧ (0 : ℝ) < 1 / 16) = fun ω => ω.1 ≠ 0 := by
    funext ω
    apply propext
    exact ⟨fun h => h.1, fun h => ⟨h, by norm_num⟩⟩
  have h2' : prob (fun ω : Ω16 => ω.1 ≠ 0 ∧ (0 : ℝ) < 1 / 16) ≤
      7 / 8 * prob (fun _ : Ω16 => (0 : ℝ) < 1 / 16) := h2
  have hpass : prob (fun ω : Ω16 => ω.1 ≠ 0 ∧ (0 : ℝ) < 1 / 16) = 15 / 16 := by
    exact (congrArg prob e).trans prob_undrawn
  rw [hpass] at h2'
  have := prob_le_one (fun _ : Ω16 => (0 : ℝ) < 1 / 16)
  linarith

/-- No receipt split of the sampled round meets the headline's luck premises. -/
theorem sampled_no_split (R : Type) [CommRing R] (G : ReceiptGame Ω16 1) (hG : sampledGame = G.game) :
    ¬ (G.Pinned ∧ G.CubeOperands R ∧ ReceiptLuckAt (1 / 2) G) := by
  rintro ⟨hP, hC, hR⟩
  exact sampled_not_luck (Headline.luck_of_receipt_half R sampledGame G hG hP hC hR)

/-! ## (D) a skipper whose share reads the round's cube bit -/

abbrev Ω2 : Type := Unit × Finset (Fin 1)

/-- Skip (share 1) iff the bit is set; always pass, always right, always the honest receipt. -/
def adaptG : ReceiptGame Ω2 1 :=
  .round (fun ω => if ω.2 = ∅ then 0 else 1) (fun _ => True) (fun _ => True) (fun _ => True) (fun _ => .done)

theorem adapt_pinned : adaptG.Pinned := And.intro (fun _ _ => trivial) (fun _ => trivial)

theorem adapt_receipt_luck : ReceiptLuckAt (1 / 2) adaptG := by
  refine And.intro ?_ (fun _ => trivial)
  have h0 : prob (fun ω : Ω2 => True ∧ ¬ True ∧ 0 < (if ω.2 = ∅ then (0 : ℝ) else 1)) = 0 :=
    prob_of_false _ fun ω h => h.2.1 trivial
  have h1 : prob (fun ω : Ω2 => ¬ True ∧ 0 < (if ω.2 = ∅ then (0 : ℝ) else 1)) = 0 :=
    prob_of_false _ fun ω h => h.1 trivial
  show prob (fun ω : Ω2 => True ∧ ¬ True ∧ 0 < (if ω.2 = ∅ then (0 : ℝ) else 1)) ≤
    1 / 2 * prob (fun ω : Ω2 => ¬ True ∧ 0 < (if ω.2 = ∅ then (0 : ℝ) else 1))
  rw [h0, h1]
  norm_num

theorem adapt_not_luck (p : ℝ) (hp : p < 1) : ¬ LuckAt p adaptG.game := by
  intro h
  obtain ⟨-, h2, -⟩ := h
  have h2' : prob (fun ω : Ω2 => True ∧ 0 < (if ω.2 = ∅ then (0 : ℝ) else 1)) ≤
      p * prob (fun ω : Ω2 => 0 < (if ω.2 = ∅ then (0 : ℝ) else 1)) := h2
  simp only [true_and] at h2'
  have hq : 0 < prob (fun ω : Ω2 => 0 < (if ω.2 = ∅ then (0 : ℝ) else 1)) :=
    prob_pos_of _ ((), {0}) (by simp)
  nlinarith [mul_pos (sub_pos.mpr hp) hq]

/-- So `CubeOperands` fails for it, at any ring: it is the premise that excludes the bit-reading skipper. -/
theorem adapt_not_cube (R : Type) [CommRing R] : ¬ adaptG.CubeOperands R := fun hC =>
  adapt_not_luck (7 / 8) (by norm_num)
    (Headline.luck_of_receipt_half R adaptG.game adaptG rfl adapt_pinned hC adapt_receipt_luck)

/-! ## (E) the trivial split: `Pl.game = G.game`, `Pinned` and `ReceiptLuckAt` cost nothing on their own -/

/-- Any play's ideal game as a receipt game: every word right, the receipt the honest one exactly when the round
passes. -/
def lift {Ω : Type} : {n : ℕ} → RoundGame Ω n → ReceiptGame Ω n
  | _, .done => .done
  | _, .round d pass next => .round d pass (fun _ => True) pass (fun ω => lift (next ω))

theorem lift_game {Ω : Type} : ∀ {n : ℕ} (g : RoundGame Ω n), (lift g).game = g
  | _, .done => rfl
  | _, .round d pass next => by
    show RoundGame.round d pass (fun ω => (lift (next ω)).game) = RoundGame.round d pass next
    congr 1
    funext ω
    exact lift_game (next ω)

theorem lift_pinned {Ω : Type} : ∀ {n : ℕ} (g : RoundGame Ω n), (lift g).Pinned
  | _, .done => trivial
  | _, .round _ _ next => And.intro (fun _ h => h) (fun ω => lift_pinned (next ω))

theorem lift_receipt_luck {Ω : Type} [Fintype Ω] (pr : ℝ) :
    ∀ {n : ℕ} (g : RoundGame Ω n), ReceiptLuckAt pr (lift g)
  | _, .done => trivial
  | _, .round d pass next => by
    refine And.intro ?_ (fun ω => lift_receipt_luck pr (next ω))
    have h0 : prob (fun ω => pass ω ∧ ¬ True ∧ 0 < d ω) = 0 := prob_of_false _ fun ω h => h.2.1 trivial
    have h1 : prob (fun ω => ¬ True ∧ 0 < d ω) = 0 := prob_of_false _ fun ω h => h.1 trivial
    show prob (fun ω => pass ω ∧ ¬ True ∧ 0 < d ω) ≤ pr * prob (fun ω => ¬ True ∧ 0 < d ω)
    rw [h0, h1, mul_zero]

/-- With the trivial split, `CubeOperands` holds only where the round is honest: its content is the cheater's. -/
theorem lift_cube_honest {α ι : Type} [Fintype ι] [DecidableEq ι] (R : Type) [CommRing R] {n : ℕ}
    (d : α × Finset ι → ℝ) (pass : α × Finset ι → Prop) (next : α × Finset ι → RoundGame (α × Finset ι) n)
    (h : (lift (RoundGame.round d pass next)).CubeOperands R) : ∀ ω, d ω = 0 := by
  obtain ⟨hd01, ⟨δ, e, -, hdδ, hne, hright⟩, -⟩ := h
  intro ω
  rcases lt_or_eq_of_le (hd01 ω).1 with hpos | hzero
  · rw [hdδ ω] at hpos
    obtain ⟨S, hS⟩ := hne ω.1 hpos
    exact absurd (hright (ω.1, S) trivial) hS
  · exact hzero.symm

/-! ## Axioms -/

#print axioms accepts_wrong_le
#print axioms accepts_wrong_le_budget
#print axioms sampled_not_luck
#print axioms sampled_no_split
#print axioms adapt_pinned
#print axioms adapt_receipt_luck
#print axioms adapt_not_luck
#print axioms adapt_not_cube
#print axioms lift_game
#print axioms lift_pinned
#print axioms lift_receipt_luck
#print axioms lift_cube_honest

end RTHeadline
