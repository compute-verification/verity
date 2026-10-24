import Proofs.Accounting.TS.Drawn
import Definitions.Accounting.Guarantees.TS.Fixed

/-!
# An honest run meets option A's hypotheses (non-vacuity)

The witness Rule I asks of `Accounting.Guarantees.TS.UnitFixedOfWords` and `Accounting.Guarantees.TS.AdditiveFixed`:
an honest word layer, with a K-chunk running sum and a glue, that meets every hypothesis of both statements.

One round (`S = 1`) with one coin bit (`m = 1`, `C = Bool`), a draw bit (`B = Bool`) and four units, four draws a round
(`s = 4`), values, noise and words in `ℤ`, the noise being the coin bit `n`:

* `chunk0` (unit 0) computes `S₀ = 2·(3 + n)` from two inputs committed before the coin;
* `chunk1` (unit 1) computes `S₁ = 5·(7 + n) + S₀`: its product of two inputs committed before the coin, plus `chunk0`'s
  running sum, an additive operand written after the coin;
* `glue` (unit 2) computes `G = S₁·2`, its noise not read, its multiplicative operand `S₁` written after the coin by
  `chunk1`, which is always proved;
* `prod` (unit 3) computes `G·(11 + n)`, its multiplicative operand `G` written after the coin by `glue`.

Draws 0, 1 and 3 name units 0, 1 and 3; draw 2 names `glue` when the draw bit is set and `prod` when it isn't. So
`glue` is sampled, not always proved, and priced: draw 2 lands on it with probability `1/2`, and its late cone (`glue`
and `prod`) weighs `1 ≤ 1/2 · 3`, the mean round weight `N_w` being `3`. `prod` is fixed through the priced `glue`,
`glue` through the always-proved `chunk1`, and the chunks by their bound operands. `prod` also reads `glue`'s other
input, so every position is opened on every path.

Each unit writes its words at every output slot and its receipt commits them. Every unit is paid (`mode = paid`) and
`fixed` is the total weight, `glue` weighing nothing. So every unit is fixed under option A, computed, and good, and
`chunk0` steers `prod` (through `chunk1` and `glue`): `FixedWords` asks of `prod` what `Fixed ∧ Computed` doesn't. For
`AdditiveFixed`, `chunk1`'s multiplicative operands are bound before its coin, it is computed on every path, and its
additive operand takes two values on two continuations from its coin.

This witness tests no skip. No unit is skipped, so it shows the hypotheses can all hold of an honest run, not that a
skipped, priced writer's cone is charged at its draw probability: every unit is good, nothing here charges a unit that
isn't (`DrawnLayer.Covers`), and `Priced` holds of `glue` without a run in which `glue` is skipped.
-/

namespace Accounting.SecurityProofs.TS.FixedMeets

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- The window's coins: one round, one coin bit, a draw bit. -/
abbrev FCoins := Paths (Paths Bool 1 × Bool) 1

/-- The coin bit. -/
def fbit (y : FCoins) : Bool := coinAt (coinAt y 0).1 0

/-- A bit as `0` or `1`. -/
def nzb (b : Bool) : ℤ := if b then 1 else 0

/-- The noise on `y`: its coin bit. -/
def nz (y : FCoins) : ℤ := nzb (fbit y)

/-- Draw `i` on draw bit `b`: draw 2 names `glue` when `b` is set and `prod` when it isn't; draw `i` otherwise names
unit `i`. -/
def drawF (b : Bool) (i : Fin 4) : Fin 4 := if i = 2 ∧ b = false then 3 else i

/-- Each unit's slots' positions: `chunk0` reads `0, 1` and writes `2`; `chunk1` reads `3, 4` and `2` (additively) and
writes `5`; `glue` reads `5, 6` and writes `7`; `prod` reads `7, 8` and `6` and writes `9`. -/
def posF : Fin 4 → Fin 5 → Fin 10 :=
  ![![0, 1, 2, 2, 2], ![3, 4, 2, 5, 5], ![5, 6, 7, 7, 7], ![7, 8, 9, 9, 6]]

/-- Each position's honest value under noise `n`. -/
def hval (ℓ : Fin 10) (n : ℤ) : ℤ :=
  ![2, 3, 2 * (3 + n), 5, 7, 5 * (7 + n) + 2 * (3 + n), 2, (5 * (7 + n) + 2 * (3 + n)) * 2, 11,
    (5 * (7 + n) + 2 * (3 + n)) * 2 * (11 + n)] ℓ

/-- The inputs are committed before the coin, the outputs after it. -/
def cutF : Fin 10 → Fin 1 × Fin 2 :=
  ![(0, 0), (0, 0), (0, 1), (0, 0), (0, 0), (0, 1), (0, 0), (0, 1), (0, 0), (0, 1)]

/-- Each output's writer. -/
def srcF : Fin 10 → Option (Fin 1 × Fin 4) :=
  ![none, none, some (0, 0), none, none, some (0, 1), none, some (0, 2), none, some (0, 3)]

/-- Slots `0, 1` are multiplicative; `chunk1`'s slot `2` is additive. -/
def roleF (u : Fin 1 × Fin 4) : Fin 5 → SlotRole :=
  ![SlotRole.mul, SlotRole.mul, if u.2 = 1 then SlotRole.add else SlotRole.other, SlotRole.other, SlotRole.other]

/-- Whether the unit's work reads its noise: all but `glue`. -/
def eFac (u : Fin 1 × Fin 4) : ℤ := if u.2 = 2 then 0 else 1

/-- A unit's work: `a₀·(a₁ + e·n)`. -/
def prodF (u : Fin 1 × Fin 4) (a : {k : Fin 5 // roleF u k = SlotRole.mul} → ℤ) (n : ℤ) : ℤ :=
  a ⟨0, rfl⟩ * (a ⟨1, rfl⟩ + eFac u * n)

/-- What a unit's additive operand adds: `chunk1`'s running sum. -/
def accF (u : Fin 1 × Fin 4) (b : {k : Fin 5 // roleF u k = SlotRole.add} → ℤ) : ℤ :=
  if h : roleF u 2 = SlotRole.add then b ⟨2, h⟩ else 0

/-- What a unit writes: its words. -/
def outF (u : Fin 1 × Fin 4) (v : Fin 5 → ℤ) (n : ℤ) (_ : Fin 5) : ℤ :=
  v 0 * (v 1 + eFac u * n) + if u.2 = 1 then v 2 else 0

/-- A unit's receipt slot: its first output slot. -/
def receiptF (u : Fin 1 × Fin 4) : Fin 5 := if u.2 = 1 then 3 else 2

/-- The drawn layer: draws by `drawF`, every proof accepted, opening the honest values. -/
def fdl : DrawnLayer Bool Bool 1 1 4 (Fin 4) (Fin 5) (Fin 10) ℤ where
  draw := drawF
  pos _ := posF
  coin _ _ := 0
  cut := cutF
  src := srcF
  accepted _ _ _ := True
  opened y w i k := hval (posF (drawF (coinAt y w).2 i) k) (nz y)
  rel _ _ _ _ := True
  right _ _ _ _ := True

/-- The word layer: each unit's words are its honest derivation at its committed slots. -/
noncomputable def fpf : WordLayer Bool Bool 1 1 4 (Fin 4) (Fin 5) (Fin 10) ℤ ℤ ℤ where
  toDrawnLayer := fdl
  role := roleF
  noise y _ := nz y
  words y u := prodF u (fun k => fdl.view y u.1 u.2 k.1) (nz y) + accF u (fun k => fdl.view y u.1 u.2 k.1)
  prod := prodF
  acc := accF
  output := outF
  receipt := receiptF
  fold _ w := w

/-- Every unit is paid. -/
def fmode : FCoins → Fin 1 × Fin 4 → Mode := fun _ _ => .paid

/-- Each unit's words: `glue` none, the others one. -/
noncomputable def fwt (_ : Fin 1) (j : Fin 4) : ℝ := if j = 2 then 0 else 1

/-- The total weight. -/
noncomputable def ftot : ℝ := ∑ w : Fin 1, ∑ j : Fin 4, fwt w j

/-- The play: one round, always accepted, with the total weight fixed. -/
noncomputable def fPlay : Play (Paths Bool 1 × Bool) 1 Unit Unit Unit Unit where
  game := .round (fun _ => 0) (fun _ => True) (fun _ => .done)
  accepts _ := True
  run _ :=
    { run := ⟨fun _ => 0, 0, 0⟩, nword := fun _ _ => 0, nfold := fun _ => 0, nfill := fun _ => 0,
      words := fun _ => 0, fixed := ftot }

/-- The run's folded words: the total weight. -/
noncomputable def ffolded : FCoins → ℝ := fun _ => ftot

/-- The paths with coin bit `1` and `0`. -/
def xt : FCoins := (((true, ()), false), ())
def xf : FCoins := (((false, ()), false), ())

/-- On either draw bit, the drawn units read every position. -/
theorem read_all : ∀ ℓ : Fin 10, ∀ b : Bool, ∃ i : Fin 4, ∃ k : Fin 5, posF (drawF b i) k = ℓ := by decide

theorem opens_iff (y : FCoins) (ℓ : Fin 10) (v : ℤ) : fdl.Opens y ℓ v ↔ v = hval ℓ (nz y) := by
  constructor
  · rintro ⟨w, i, k, hp, -, rfl⟩
    have hp' : posF (drawF (coinAt y w).2 i) k = ℓ := hp
    subst hp'
    rfl
  · rintro rfl
    obtain ⟨i, k, rfl⟩ := read_all ℓ (coinAt y 0).2
    exact ⟨0, i, k, rfl, trivial, rfl⟩

/-- A plurality whose every continuation opens one value, a function of the path that the continuations keep, is that
value. -/
theorem plural_of_val {X V : Type} [Fintype X] [Nonempty V] (sp : X → X → X) (opens : X → V → Prop) (f : X → V)
    (hf : ∀ y v, opens y v ↔ v = f y) (x : X) (hx : ∀ z, f (sp x z) = f x) : plural sp opens x = f x := by
  classical
  by_contra hne
  have hall : tally sp opens x (f x) = Fintype.card X := by
    unfold tally
    rw [Finset.filter_true_of_mem, Finset.card_univ]
    intro z _
    exact ⟨(hf _ _).2 (hx z).symm, fun v' hv' => ((hf _ _).1 hv').trans (hx z)⟩
  have hnone : tally sp opens x (plural sp opens x) = 0 := by
    unfold tally
    rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    rintro z - ⟨h, -⟩
    exact hne (((hf _ _).1 h).trans (hx z))
  have hle := Plurality.tally_le_plural (sp := sp) (opens := opens) x (f x)
  have hpos : 0 < Fintype.card X := Fintype.card_pos_iff.2 ⟨x⟩
  omega

/-- Each position is cut at its round's start (an input) or after the coin (an output). -/
theorem cut2 : ∀ ℓ : Fin 10, (cutF ℓ).2 = 0 ∨ (cutF ℓ).2 = 1 := by decide

/-- An input's honest value doesn't read the noise bit. -/
theorem hval_input : ∀ ℓ : Fin 10, (cutF ℓ).2 = 0 → ∀ b b' : Bool, hval ℓ (nzb b) = hval ℓ (nzb b') := by decide

/-- A continuation from an output's cut, after the coin, keeps the coin bit. -/
theorem nz_cont (x z : FCoins) (ℓ : Fin 10) (h : (cutF ℓ).2 = 1) : nz (fdl.cont ℓ x z) = nz x := by
  show nz (spliceAt (cutF ℓ).1 (cutF ℓ).2 x z) = nz x
  rw [h, Subsingleton.elim (cutF ℓ).1 0]
  rfl

/-- A position's honest value is the same on every continuation from its cut. -/
theorem cont_val (x z : FCoins) (ℓ : Fin 10) : hval ℓ (nz (fdl.cont ℓ x z)) = hval ℓ (nz x) := by
  rcases cut2 ℓ with h | h
  · exact hval_input ℓ h (fbit (fdl.cont ℓ x z)) (fbit x)
  · rw [nz_cont x z ℓ h]

theorem committed_eq (x : FCoins) (ℓ : Fin 10) : fpf.committed x ℓ = hval ℓ (nz x) :=
  plural_of_val (fdl.cont ℓ) (fun y v => fdl.Opens y ℓ v) (fun y => hval ℓ (nz y)) (fun y v => opens_iff y ℓ v) x
    (fun z => cont_val x z ℓ)

theorem view_eq (x : FCoins) (w : Fin 1) (j : Fin 4) : fpf.view x w j = fun k => hval (posF j k) (nz x) :=
  funext fun k => committed_eq x (posF j k)

theorem out_ok : ∀ (b : Bool) (w : Fin 1) (j : Fin 4) (k : Fin 5), srcF (posF j k) = some (w, j) →
    hval (posF j k) (nzb b) = outF (w, j) (fun k' => hval (posF j k') (nzb b)) (nzb b) k := by decide

theorem receipt_ok : ∀ (b : Bool) (w : Fin 1) (j : Fin 4),
    hval (posF j (receiptF (w, j))) (nzb b) =
      prodF (w, j) (fun k => hval (posF j k.1) (nzb b)) (nzb b) + accF (w, j) (fun k => hval (posF j k.1) (nzb b)) := by
  decide

theorem computed_all (x : FCoins) (u : Fin 1 × Fin 4) : fpf.Computed x u := by
  obtain ⟨w, j⟩ := u
  refine ⟨rfl, fun k hk => ?_⟩
  rw [committed_eq, view_eq]
  exact out_ok (fbit x) w j k hk

theorem folds_all (x : FCoins) (u : Fin 1 × Fin 4) : fpf.Folds x u := by
  obtain ⟨w, j⟩ := u
  show fpf.committed x (fpf.pos w j (receiptF (w, j))) =
    prodF (w, j) (fun k => fpf.view x w j k.1) (nz x) + accF (w, j) (fun k => fpf.view x w j k.1)
  rw [committed_eq, view_eq]
  exact receipt_ok (fbit x) w j

theorem modes_run : ModesRun fpf fmode := fun x u => ⟨fun _ => computed_all x u, fun _ => folds_all x u⟩

theorem wt_nonneg (w : Fin 1) (j : Fin 4) : 0 ≤ fwt w j := by
  unfold fwt
  split_ifs <;> norm_num

open Classical in
theorem sum_le_tot (p : Fin 1 → Fin 4 → Prop) : (∑ w, ∑ j, if p w j then fwt w j else 0) ≤ ftot :=
  Finset.sum_le_sum fun w _ => Finset.sum_le_sum fun j _ => by
    show (if p w j then fwt w j else 0) ≤ fwt w j
    split_ifs
    · exact le_rfl
    · exact wt_nonneg w j

open Classical in
theorem counts_fixed : CountsFixed fPlay fpf fwt ffolded := by
  intro x _
  show (∑ w, ∑ j, if fpf.FixedWords fwt x (w, j) then fwt w j else 0) / ((1 : ℕ) : ℝ) ≤ ftot ∧
    (∑ w, ∑ j, if fpf.Folds x (w, j) then fwt w j else 0) / ((1 : ℕ) : ℝ) ≤ ftot
  rw [Nat.cast_one, div_one, div_one]
  exact ⟨sum_le_tot fun w j => fpf.FixedWords fwt x (w, j), sum_le_tot fun w j => fpf.Folds x (w, j)⟩

theorem writer_reads : fpf.WriterReads := by
  have h : ∀ ℓ : Fin 10, ∀ w : Fin 1, ∀ j : Fin 4, srcF ℓ = some (w, j) → ∃ k : Fin 5, posF j k = ℓ := by decide
  rintro ℓ ⟨w, j⟩ hs
  exact h ℓ w j hs

theorem draw_fix (b : Bool) (i : Fin 4) (h : i ≠ 2) : drawF b i = i := by
  unfold drawF
  exact ite_eq_right fun h' => h h'.1

/-- Every unit but `glue` is always proved. -/
theorem always_proved (v : Fin 1 × Fin 4) (h : v.2 ≠ 2) : fpf.AlwaysProved v := fun b => ⟨v.2, draw_fix b v.2 h⟩

/-- **`glue` is sampled**: no draw names it when the draw bit isn't set. -/
theorem glue_sampled : ¬ fpf.AlwaysProved (0, 2) := fun h => by
  have h2 : ∀ i : Fin 4, drawF false i ≠ 2 := by decide
  obtain ⟨i, hi⟩ := h false
  exact h2 i hi

/-- A late edge out of `glue` or `prod` lands on `glue` or `prod`. -/
theorem late_closed (u v : Fin 1 × Fin 4) (h : fpf.Late u v) (hu : u.2 = 2 ∨ u.2 = 3) : v.2 = 2 ∨ v.2 = 3 := by
  have key : ∀ (j : Fin 4) (k : Fin 5) (u : Fin 1 × Fin 4), srcF (posF j k) = some u → (u.2 = 2 ∨ u.2 = 3) →
      j = 2 ∨ j = 3 := by decide
  obtain ⟨k, hk, -⟩ := h
  exact key v.2 k u hk hu

/-- `glue`'s late cone is `glue` and `prod`. -/
theorem cone_glue (v : Fin 1 × Fin 4) (h : Relation.ReflTransGen fpf.Late (0, 2) v) : v.2 = 2 ∨ v.2 = 3 := by
  induction h with
  | refl => exact Or.inl rfl
  | tail _ hl ih => exact late_closed _ _ hl ih

theorem fwt2 (w : Fin 1) : fwt w 2 = 0 := by
  unfold fwt
  exact ite_eq_left rfl

theorem fwt_ne2 (w : Fin 1) (j : Fin 4) (h : j ≠ 2) : fwt w j = 1 := by
  unfold fwt
  exact ite_eq_right h

open Classical in
theorem cone_term (v : Fin 1 × Fin 4) :
    (if Relation.ReflTransGen fpf.Late (0, 2) v then fwt v.1 v.2 else 0) ≤ if v = (0, 3) then (1 : ℝ) else 0 := by
  by_cases hc : Relation.ReflTransGen fpf.Late (0, 2) v
  · rw [ite_eq_left hc]
    rcases cone_glue v hc with h | h
    · rw [h, fwt2]
      split_ifs <;> norm_num
    · have hv : v = (0, 3) := Prod.ext (Subsingleton.elim _ _) h
      rw [ite_eq_left hv]
      exact (fwt_ne2 v.1 v.2 (by rw [h]; decide)).le
  · rw [ite_eq_right hc]
    split_ifs <;> norm_num

open Classical in
/-- `glue`'s late cone weighs `1`, `prod`'s weight. -/
theorem coneWt_glue : fpf.coneWt fwt (0, 2) ≤ 1 := by
  unfold DrawnLayer.coneWt
  calc (∑ v : Fin 1 × Fin 4, if Relation.ReflTransGen fpf.Late (0, 2) v then fwt v.1 v.2 else 0)
      ≤ ∑ v : Fin 1 × Fin 4, if v = (0, 3) then (1 : ℝ) else 0 := Finset.sum_le_sum fun v _ => cone_term v
    _ = 1 := by simp [Finset.sum_ite_eq']

theorem sum_fwt : ∑ j, fwt 0 j = 3 := by
  rw [Fin.sum_univ_four, fwt_ne2 0 0 (by decide), fwt_ne2 0 1 (by decide), fwt2, fwt_ne2 0 3 (by decide)]
  norm_num

/-- Draw 2 lands on `glue` on half the draw bits. -/
theorem prob_glue : prob (fun b : Bool => drawF b 2 = 2) = 1 / 2 := by
  have h : ∀ inst : DecidablePred (fun b : Bool => drawF b 2 = 2),
      (@Finset.filter _ _ inst Finset.univ).card = 1 := by
    intro inst
    rw [Finset.card_eq_one]
    refine ⟨true, Finset.ext fun b => ?_⟩
    rw [Finset.mem_filter, Finset.mem_singleton]
    simp only [Finset.mem_univ, true_and]
    clear inst
    cases b <;> decide
  unfold prob
  rw [h, Fintype.card_bool]
  norm_num

/-- **`glue` is priced at its public fan-out**: draw 2 lands on it with probability `1/2`, and its late cone weighs `1`,
at most `1/2` of the mean round weight `N_w = 3` (one round). -/
theorem priced_glue : fpf.Priced fwt (0, 2) := by
  refine ⟨2, ?_⟩
  show fpf.coneWt fwt (0, 2) ≤
    prob (fun b : Bool => drawF b 2 = 2) * ((∑ w : Fin 1, ∑ j, fwt w j) / ((1 : ℕ) : ℝ))
  have h : prob (fun b : Bool => drawF b 2 = 2) * ((∑ w : Fin 1, ∑ j, fwt w j) / ((1 : ℕ) : ℝ)) = 3 / 2 := by
    rw [prob_glue, Fin.sum_univ_one, sum_fwt]
    norm_num
  rw [h]
  linarith [coneWt_glue]

theorem bound_of_none : ∀ (j : Fin 4) (k : Fin 5), srcF (posF j k) = none →
    (cutF (posF j k)).1 < 0 ∨ (cutF (posF j k)).1 = 0 ∧ (cutF (posF j k)).2 ≤ (0 : Fin 1).castSucc := by
  decide

theorem fixed_all (u : Fin 1 × Fin 4) : fpf.Fixed fwt u := by
  obtain ⟨w, j⟩ := u
  obtain rfl : w = 0 := Subsingleton.elim w 0
  intro k _
  rcases h : srcF (posF j k) with _ | v
  · exact Or.inl (bound_of_none j k h)
  · refine Or.inr ⟨v, h, ?_⟩
    by_cases hv : v.2 = 2
    · obtain rfl : v = (0, 2) := Prod.ext (Subsingleton.elim _ _) hv
      exact Or.inr priced_glue
    · exact Or.inl (always_proved v hv)

theorem credited_fixed (w : Fin 1) (j : Fin 4) (_ : fwt w j ≠ 0) : fpf.Fixed fwt (w, j) := fixed_all (w, j)

theorem fixedWords_all (x : FCoins) (u : Fin 1 × Fin 4) : fpf.FixedWords fwt x u :=
  ⟨fixed_all u, computed_all x u, fun t _ => computed_all x t⟩

theorem good_all (x : FCoins) (w : Fin 1) (j : Fin 4) : fpf.Good fmode x w j := fun _ _ => rfl

theorem late01 : fpf.Late (0, 0) (0, 1) := ⟨2, rfl, Or.inr ⟨rfl, by decide⟩⟩

theorem late12 : fpf.Late (0, 1) (0, 2) := ⟨0, rfl, Or.inr ⟨rfl, by decide⟩⟩

theorem prod_late : ¬ fpf.Bound (0, 3) 0 := by
  show ¬ ((cutF (posF 3 0)).1 < 0 ∨ (cutF (posF 3 0)).1 = 0 ∧ (cutF (posF 3 0)).2 ≤ (0 : Fin 1).castSucc)
  decide

theorem mulLate23 : fpf.MulLate (0, 2) (0, 3) := ⟨0, rfl, rfl, prod_late⟩

theorem steers_prod : fpf.Steers (0, 0) (0, 3) :=
  ⟨(0, 2), Relation.ReflTransGen.tail (Relation.ReflTransGen.single late01) late12, mulLate23⟩

theorem tot_pos : 0 < ftot := by
  have h1 : fwt 0 0 = 1 := by
    unfold fwt
    split_ifs with h
    · exact absurd h (by decide)
    · rfl
  have h2 : fwt 0 0 ≤ ∑ j, fwt 0 j := Finset.single_le_sum (fun j _ => wt_nonneg 0 j) (Finset.mem_univ 0)
  have h3 : ∑ j, fwt 0 j ≤ ftot :=
    Finset.single_le_sum (f := fun w => ∑ j, fwt w j) (fun w _ => Finset.sum_nonneg fun j _ => wt_nonneg w j)
      (Finset.mem_univ 0)
  linarith

/-- **`UnitFixedOfWords`'s hypotheses hold of an honest run** with a K-chunk running sum and a sampled glue: nonnegative
weights, positive in total, every credited unit fixed, the modes the run's, the labelling counting fixed words, every
writer reading what it writes, every path accepted and every unit good, `chunk0` steering `prod` through a late
multiplicative operand, and `glue`, which writes it, sampled and priced, not always proved. -/
theorem unitFixedOfWords_meets :
    (∀ w j, 0 ≤ fwt w j) ∧ 0 < ftot ∧ (∀ w j, fwt w j ≠ 0 → fpf.Fixed fwt (w, j)) ∧ ModesRun fpf fmode ∧
      CountsFixed fPlay fpf fwt ffolded ∧ fpf.WriterReads ∧ (∀ x, fPlay.accepts x) ∧
      (∀ x w j, fpf.Good fmode x w j) ∧ (∀ x u, fpf.FixedWords fwt x u) ∧ fpf.Steers (0, 0) (0, 3) ∧
      ¬ fpf.Bound (0, 3) 0 ∧ fpf.MulLate (0, 2) (0, 3) ∧ ¬ fpf.AlwaysProved (0, 2) ∧ fpf.Priced fwt (0, 2) :=
  ⟨wt_nonneg, tot_pos, credited_fixed, modes_run, counts_fixed, writer_reads, fun _ => trivial, good_all,
    fixedWords_all, steers_prod, prod_late, mulLate23, glue_sampled, priced_glue⟩

theorem bound_chunk1 : ∀ k, fpf.role (0, 1) k = SlotRole.mul → fpf.Bound (0, 1) k := by
  have h : ∀ k : Fin 5, roleF (0, 1) k = SlotRole.mul →
      (cutF (posF 1 k)).1 < 0 ∨ (cutF (posF 1 k)).1 = 0 ∧ (cutF (posF 1 k)).2 ≤ (0 : Fin 1).castSucc := by decide
  exact h

theorem nz_xt : nz xt = 1 := rfl

theorem nz_splice : nz (spliceAt (0 : Fin 1) (fpf.coin 0 1) xt xf) = 0 := rfl

theorem additive_moves :
    fpf.committed xt (fpf.pos 0 1 2) ≠ fpf.committed (spliceAt (0 : Fin 1) (fpf.coin 0 1) xt xf) (fpf.pos 0 1 2) := by
  rw [committed_eq, committed_eq, nz_xt, nz_splice]
  decide

/-- **`AdditiveFixed`'s hypotheses hold of an honest K-chunk** whose running sum is written after its coin: `chunk1`'s
multiplicative operands are bound before its coin and it is computed on every path, while its additive operand, which
`chunk0` writes after that coin, takes two values on two continuations from it. -/
theorem additiveFixed_meets :
    (∀ k, fpf.role (0, 1) k = SlotRole.mul → fpf.Bound (0, 1) k) ∧ (∀ x, fpf.Computed x (0, 1)) ∧
      fpf.role (0, 1) 2 = SlotRole.add ∧ fpf.Late (0, 0) (0, 1) ∧
      fpf.committed xt (fpf.pos 0 1 2) ≠ fpf.committed (spliceAt (0 : Fin 1) (fpf.coin 0 1) xt xf) (fpf.pos 0 1 2) :=
  ⟨bound_chunk1, fun x => computed_all x (0, 1), rfl, late01, additive_moves⟩

end Accounting.SecurityProofs.TS.FixedMeets
