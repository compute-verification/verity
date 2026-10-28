import Proofs.Pouw.Period
import Proofs.Pouw.Period.Accept
import Proofs.Accounting.ComposePeriod.Honest

/-!
# An honest live-salt period meets `H1CheckedCreditPeriod`'s premises

The positive twin of `Properties.Pouw.Period.H1CheckedCreditPeriod`, as
`Accounting.SecurityProofs.ComposePeriod.Honest` is of `CircuitWorkPeriod`: a period of `n ≥ 1` windows at H1's
instance's own numbers, whose every window the deployed check (`PouwCheck.protocol.check`) accepts at the
all-protocols leg's parameters (`Accept.h1Params`), and which meets every premise of the instance
(`h1_meets`).

The leg's check takes these numbers: `verity/pouw/v1` accepts the loop's honest window with 128 real tiles of `N = 144`,
all 24 drawn tiles proved, at `ε = 1/2` and `δ = 10⁻⁶`. So the `N_p = 144` and `K_d = 24` that `DrawIsCoin` and this
witness use are the leg's actual check's.

* The circuit (`circuit`): 128 committed words, one a unit (H1's 128 real tiles a window), no operands, so no unit
  writes another late (`reaches_iff`) and each real tile's late cone is its own word; one round, live. The
  population is the grid's `N_p = 144`: tiles `128 … 143` are fillers of no words. So every real tile's share is
  `1/128` (`shareCone_eq`) and `ρ = 8/9` meets `ρ·N_p·share ≤ 1` with equality.
* The scheme is the ComposePeriod twin's (operand luck pinned, `r_B = ½`, guess `½`), so the lucks are `½ ≤ 7/8` and
  `½ ≤ ½`.
* The windows (`windows`): every window registers the loop's honest registration (`Accept.h1Reg`) and the service
  answers it with the units of the model's draw at the window's draw coin, ascending (`unitsOf`), so the check accepts
  every window on every path (`accepted_all`, from `Accept.check_h1`), publishes `N = 144`, `K′ = 24` and exactly
  those units (`drawIsCoin`), and its registration doesn't read the coins (`RegisteredBeforeDraw`).
* The salt is the window's round coin (`SaltIsLiveCoin` by definition), every tile is live and paid with its words
  right, and the openings are in `Unit`, so binding holds at `δ_bind = 0`, the service's soundness at `δ_ks = 0`, and
  the staged and flat relations are `True` (`StagedPins`).

The conclusion then speaks of the whole period: its accepted prefix is all `n` windows (`prefixLen_all`) and its
credited live words are `128·n` (`checkedLive_all`), so `h1_credit` is `H1CheckedCreditPeriod`'s bound at `128·n` live
words, not at none.
-/

namespace Properties.Pouw.Period.SecurityProofs.HonestH1

open Properties.Pouw.Period Properties.Pouw.Period.SecurityProofs Properties.Pouw.Period.SecurityProofs.Accept
open PouwCheck (Params UnitsOk Ascending)
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposePeriod

/-! ## The circuit -/

/-- 128 committed words of no operands, one a unit, one live round. -/
def circuit : Declared Unit where
  dag := List.replicate 128 Honest.word
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun _ => 0
  seedNames := fun _ _ => false
  units := 128
  unit := fun i => i

theorem hr1 : 0 < circuit.rounds := Nat.one_pos

theorem hK24 : 24 ≤ 144 := by decide

theorem length_dag : circuit.dag.length = 128 := List.length_replicate

theorem wordGates_eq : wordGates circuit = List.range 128 := by
  unfold wordGates
  rw [length_dag]
  refine List.filter_eq_self.2 fun i hi => ?_
  show ((List.replicate 128 Honest.word)[i]?).any isWord = true
  rw [List.getElem?_replicate_of_lt (List.mem_range.1 hi)]
  decide

theorem unitWords_eq (u : ℕ) : unitWords circuit u = if u < 128 then [u] else [] := by
  unfold unitWords
  rw [wordGates_eq]
  show (List.range 128).filter (· == u) = _
  rw [List.filter_beq, List.count_range]
  by_cases h : u < 128 <;> simp [h]

theorem tileWords_eq (t : Fin 144) : tileWords Honest.scheme circuit 144 t = if t.val < 128 then 1 else 0 := by
  unfold tileWords
  rw [unitWords_eq]
  by_cases h : t.val < 128 <;> simp [h, Honest.scheme]

theorem wordsOf_eq : wordsOf Honest.scheme circuit = 128 := by
  unfold wordsOf creditedWords
  rw [wordGates_eq]
  simp [Honest.scheme]

theorem not_reads (j i : ℕ) : ¬ circuit.Reads j i := by
  show ¬ ((List.replicate 128 Honest.word)[j]?.any fun g => g.args.contains i) = true
  rw [List.getElem?_replicate]
  split <;> simp [Honest.word]

/-- **No unit writes another late**: the circuit's late cones are its units. -/
theorem reaches_iff (u v : ℕ) : circuit.Reaches u v ↔ u = v := by
  constructor
  · intro h
    have h' : Relation.ReflTransGen circuit.UnitLate u v := h
    induction h' with
    | refl => rfl
    | tail _ hbc _ =>
      obtain ⟨i, j, -, -, -, hl⟩ := hbc
      exact absurd hl.1 (not_reads j i)
  · rintro rfl
    exact Relation.ReflTransGen.refl

theorem tileCone_eq (t : Fin 144) :
    tileCone Honest.scheme circuit 144 t = tileWeight Honest.scheme circuit 144 t := by
  unfold tileCone
  rw [Finset.sum_eq_single t]
  · exact if_pos ((reaches_iff _ _).2 rfl)
  · intro t' _ ht'
    exact if_neg fun h => ht' (Fin.ext ((reaches_iff _ _).1 h)).symm
  · intro h
    exact absurd (Finset.mem_univ t) h

theorem shareCone_eq (t : Fin 144) :
    shareCone Honest.scheme circuit 144 t = if t.val < 128 then 1 / 128 else 0 := by
  unfold shareCone
  rw [tileCone_eq, tileWeight, tileWords_eq, wordsOf_eq]
  by_cases h : t.val < 128 <;> simp [h]

theorem shareCone_nonneg (t : Fin 144) : 0 ≤ shareCone Honest.scheme circuit 144 t := by
  rw [shareCone_eq]
  split_ifs <;> norm_num

/-- The real tiles' sum: tiles `0 … 127` of the 144. -/
theorem sum_ite_lt (c : ℝ) : ∑ t : Fin 144, (if t.val < 128 then c else 0) = 128 * c := by
  have h := Fin.sum_univ_add (a := 128) (b := 16) (fun t => if t.val < 128 then c else 0)
  have h1 : ∀ i : Fin 128, (if (Fin.castAdd 16 i).val < 128 then c else 0) = c := fun i => if_pos i.isLt
  have h2 : ∀ i : Fin 16, (if (Fin.natAdd 128 i).val < 128 then c else 0) = 0 := fun i =>
    if_neg (by show ¬ (128 + i.val < 128); omega)
  simp only [h1, h2, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, add_zero, smul_zero,
    Nat.cast_ofNat] at h
  exact h

theorem sum_shareCone : ∑ t : Fin 144, shareCone Honest.scheme circuit 144 t = 1 := by
  simp only [shareCone_eq]
  rw [sum_ite_lt]
  norm_num

theorem sum_tileWords : ∑ t : Fin 144, (tileWords Honest.scheme circuit 144 t : ℝ) = 128 := by
  simp only [tileWords_eq, Nat.cast_ite, Nat.cast_one, Nat.cast_zero]
  rw [sum_ite_lt]
  norm_num

theorem circuit_wf : circuit.Wf := by
  refine ⟨fun i hi => ⟨Nat.one_pos, lt_of_lt_of_eq hi length_dag⟩, fun _ _ => Nat.one_pos, fun _ _ _ _ _ => rfl,
    fun _ _ _ _ _ _ => rfl, fun i _ j _ h => ?_⟩
  have e : i = j := h
  rw [e]

theorem scope {δs : ℝ} (hδs : 0 < δs) : Scope Honest.scheme circuit δs :=
  ⟨circuit_wf, fun _ _ _ => rfl, hδs, fun _ => by norm_num [Honest.scheme, LuckForm.value],
    fun _ _ => Or.inl (Nat.zero_le _)⟩

/-! ## The period, its windows and its proofs -/

/-- A window's coins: its round's coin, the draw's permutation and the proofs' (none). -/
abbrev Ω1 : Type := Paths Bool circuit.rounds × (Equiv.Perm (Fin 144) × Unit)

/-- A window's honest stages: every tile paid, its words right and its checked values honest. -/
def stages : SplitStages Bool (Fin 144) 1 :=
  .stage (fun _ _ => .paid) (fun _ _ => True) (fun _ _ => True) fun _ => .done

/-- The honest period of `n` windows: every tile live, each window's stages `stages`, whatever the past. -/
def period : (n : ℕ) → Period Bool Ω1 (Fin 144) circuit.rounds n
  | 0 => .done
  | n + 1 => .window (fun _ => true) stages fun _ => period n

/-- The check's parameters: the leg's (`h1_params`). -/
def prm : Params := h1Params ""

/-- Every window's registration: the loop's honest one. -/
def reg : PouwCheck.Registration := h1Reg "" "" "" "" ByteArray.empty

/-- The units of the model's draw at permutation coin `β`, ascending. -/
noncomputable def unitsOf (β : Equiv.Perm (Fin 144) × Unit) : Array ℕ :=
  (Finset.sort (Finset.univ.image fun i : Fin 24 => (drawOf hK24 β i).val) (· ≤ ·)).toArray

/-- The period's windows as the check reads them: the honest registration, and the service's answer accepting the
units of the window's draw. -/
noncomputable def windows (n : ℕ) : Executions Ω1 n where
  reg := fun _ _ => reg
  ev := fun x w => h1Evidence reg (unitsOf (coinAt x w).2)

/-- The service's openings, in `Unit`. -/
def opened {n : ℕ} : Paths Ω1 n → Fin n → ℕ → ℕ → Unit := fun _ _ _ _ => ()

/-- A salt's bytes from a round coin. -/
def enc (b : Bool) : ByteArray := ⟨#[if b then 1 else 0]⟩

/-- Each window's salt: its round coin's. -/
def salt {n : ℕ} (x : Paths Ω1 n) (w : Fin n) : ByteArray := enc (coinAt (coinAt x w).1 ⟨0, hr1⟩)

/-- The staged and the flat statements: every opening meets them. -/
def staged : ByteArray → Fin 144 → (ℕ → Unit) → Prop := fun _ _ _ => True

def flat : ByteArray → Fin 144 → (ℕ → Unit) → Prop := fun _ _ _ => True

/-- The period's proofs under the check. -/
noncomputable def run (n : ℕ) : Outcomes Bool (Equiv.Perm (Fin 144) × Unit) circuit.rounds 24 144 n Unit :=
  checkedRun hK24 prm (windows n) opened (stagedAt salt staged) (flatAtCoin hr1 enc flat)

noncomputable def layer (n : ℕ) :
    DrawnLayer Bool (Equiv.Perm (Fin 144) × Unit) n circuit.rounds 24 (Fin 144) ℕ (Fin n × ℕ) Unit :=
  checkedLayer circuit hr1 hK24 prm (windows n) opened (stagedAt salt staged) (flatAtCoin hr1 enc flat)

/-- The honest run's anatomy: the 128 words fixed. -/
noncomputable def anatomy : Anatomy Unit Unit Unit Unit := { Honest.anatomy with fixed := 128 }

/-- The play: the period's ideal game at `liveCone`, accepted on every path. -/
noncomputable def play (n : ℕ) : Play Ω1 n Unit Unit Unit Unit where
  game := periodGame (tileStage circuit 144) (liveCone Honest.scheme circuit 144) 1 1 24 (period n)
  accepts := fun _ => True
  run := fun _ => anatomy

/-! ## The check accepts every window, and its draw is the coin's -/

theorem ascending_of_pairwise : ∀ (l : List ℕ), l.Pairwise (· < ·) → Ascending l
  | [], _ => trivial
  | [_], _ => trivial
  | _ :: b :: t, h => ⟨List.rel_of_pairwise_cons h List.mem_cons_self, ascending_of_pairwise (b :: t) h.of_cons⟩

theorem draw_inj (β : Equiv.Perm (Fin 144) × Unit) : Function.Injective fun i : Fin 24 => (drawOf hK24 β i).val :=
  fun _ _ h => Fin.castLE_injective hK24 (β.1.injective (Fin.ext h))

/-- **The draw's units are 24 ascending distinct units below 144.** -/
theorem unitsOk (β : Equiv.Perm (Fin 144) × Unit) : UnitsOk (unitsOf β) 24 144 := by
  refine ⟨?_, ?_, fun u hu => ?_⟩
  · simp only [unitsOf, List.size_toArray, Finset.length_sort]
    rw [Finset.card_image_of_injective _ (draw_inj β), Finset.card_univ, Fintype.card_fin]
  · simp only [unitsOf, List.toList_toArray]
    exact ascending_of_pairwise _
      (((Finset.pairwise_sort (s := Finset.univ.image fun i : Fin 24 => (drawOf hK24 β i).val)
        (r := (· ≤ ·))).and (Finset.sort_nodup (s := Finset.univ.image fun i : Fin 24 => (drawOf hK24 β i).val)
          (r := (· ≤ ·)))).imp fun h => lt_of_le_of_ne h.1 h.2)
  · simp only [unitsOf, List.toList_toArray, Finset.mem_sort, Finset.mem_image] at hu
    obtain ⟨i, -, rfl⟩ := hu
    exact (drawOf hK24 β i).isLt

/-- **The deployed check accepts every window on every path.** -/
theorem accepted_all (n : ℕ) (x : Paths Ω1 n) (w : Fin n) :
    Accepts prm ((windows n).reg x w) ((windows n).ev x w) :=
  ⟨_, _, check_h1 "" "" "" "" ByteArray.empty (unitsOf (coinAt x w).2) (unitsOk _)⟩

theorem acceptedUpTo_all (n : ℕ) (x : Paths Ω1 n) (w : Fin n) : (windows n).AcceptedUpTo prm x w :=
  fun w' _ => accepted_all n x w'

/-- **The check's draw is the period's draw coin**: it publishes `N = 144`, `K′ = 24` and the draw's units. -/
theorem drawIsCoin (n : ℕ) : DrawIsCoin hK24 prm (windows n) := by
  intro x w pub prof hc
  have he := hc.symm.trans (check_h1 "" "" "" "" ByteArray.empty (unitsOf (coinAt x w).2) (unitsOk _))
  injection he with hp
  subst hp
  refine ⟨congrArg PouwWindow.Work.n work_h1, rfl, fun u => ?_⟩
  show u ∈ (unitsOf (coinAt x w).2).toList ↔ _
  simp only [unitsOf, List.toList_toArray, Finset.mem_sort, Finset.mem_image, Finset.mem_univ, true_and]

theorem serviceSound (n : ℕ) : ServiceSound prm (windows n) opened (stagedAt salt staged) 0 :=
  fun _ => le_of_eq (Honest.prob_none fun _ ⟨_, _, _, _, _, h⟩ => h trivial)

theorem prefixLen_all (n : ℕ) (x : Paths Ω1 n) : (windows n).prefixLen prm x = n := by
  classical
  unfold Executions.prefixLen
  rw [Finset.filter_true_of_mem fun w _ => acceptedUpTo_all n x w, Finset.card_univ, Fintype.card_fin]

/-! ## The honest windows' claims -/

theorem period_every {P : (Fin 144 → Bool) → SplitStages Bool (Fin 144) 1 → Prop} (h : P (fun _ => true) stages) :
    ∀ n, (period n).EveryWindow P
  | 0 => trivial
  | n + 1 => ⟨h, fun _ => period_every h n⟩

theorem liveAt_period : ∀ (n : ℕ) (x : Paths Ω1 n) (w : Fin n), (period n).liveAt x w = fun _ => true
  | 0, _, w => w.elim0
  | n + 1, x, w => by
    refine Fin.cases ?_ (fun w' => ?_) w
    · rfl
    · exact liveAt_period n x.2 w'

theorem liveWords_true (t : Fin 144) :
    liveWords Honest.scheme circuit 144 (fun _ => true) t = tileWords Honest.scheme circuit 144 t := by
  unfold liveWords
  exact if_pos rfl

/-- **The accepted prefix's live words are `128·n`.** -/
theorem checkedLive_all (n : ℕ) (x : Paths Ω1 n) :
    checkedLive Honest.scheme circuit (period n) prm (windows n) x = 128 * n := by
  unfold checkedLive
  calc _ = ∑ _w : Fin n, (128 : ℝ) := Finset.sum_congr rfl fun w _ => by
          rw [if_pos (acceptedUpTo_all n x w), liveAt_period n x w]
          simp only [liveWords_true]
          exact sum_tileWords
    _ = 128 * n := by simp [mul_comm]

theorem paid_share (s : Mode) (hs : s ≠ .paid) (t : Fin 144) (c : Bool) :
    (fun c : Bool => if (fun (_ : Bool) (_ : Fin 144) => Mode.paid) c t = s then (1 : ℝ) else 0) c = 0 :=
  ite_eq_right fun h => hs h.symm

theorem every_perLiveTile {P : Kind → ReceiptGame Bool 1 → ReceiptGame Bool 1 → Prop}
    (h : ∀ k (t : Fin 144), P k (tileGame (fun _ _ => Mode.paid) (fun _ _ => True) (fun _ _ => True) .folded t)
      (tileGame (fun _ _ => Mode.paid) (fun _ _ => True) (fun _ _ => True) .whole t)) (n : ℕ) :
    (period n).EveryWindow (fun live W => PerLiveTile Honest.scheme circuit live W P) :=
  period_every (P := fun live W => PerLiveTile Honest.scheme circuit live W P)
    ⟨fun t _ _ k _ _ => h k t, fun _ => trivial⟩ n

/-- **The honest period passes every prefix of its game**, at any weights and costs. -/
theorem acceptsUpTo_period (wt : (Fin 144 → Bool) → Fin 144 → ℝ) (a b : ℝ) :
    ∀ (n j : ℕ) (x : Paths Ω1 n),
      RoundGame.acceptsUpTo (periodGame (tileStage circuit 144) wt a b 24 (period n)) j x
  | 0, _, _ => trivial
  | _ + 1, 0, _ => trivial
  | n + 1, j + 1, x =>
    ⟨fun v _ => by simp [stages, SplitStages.joint, SplitStages.sameAt], acceptsUpTo_period wt a b n j x.2⟩

theorem ite_words_le (c : Prop) {_ : Decidable c} (j : Fin 144) :
    (if c then (tileWords Honest.scheme circuit 144 j : ℝ) else 0) ≤ tileWords Honest.scheme circuit 144 j := by
  split_ifs
  exacts [le_rfl, Nat.cast_nonneg _]

theorem ite_zero_le (c : Prop) {_ : Decidable c} (j : Fin 144) :
    (if c then 0 else (tileWords Honest.scheme circuit 144 j : ℝ)) ≤ tileWords Honest.scheme circuit 144 j := by
  split_ifs
  exacts [Nat.cast_nonneg _, le_rfl]

theorem sum_div_le {n : ℕ} (hn : 0 < n) (f : Fin n → Fin 144 → ℝ)
    (hf : ∀ w j, f w j ≤ tileWords Honest.scheme circuit 144 j) : (∑ w, ∑ j, f w j) / n ≤ 128 := by
  rw [div_le_iff₀ (Nat.cast_pos.mpr hn)]
  calc ∑ w, ∑ j, f w j ≤ ∑ _w : Fin n, (128 : ℝ) :=
        Finset.sum_le_sum fun w _ => (Finset.sum_le_sum fun j _ => hf w j).trans_eq sum_tileWords
    _ = 128 * n := by simp [mul_comm]

/-! ## Every premise -/

/-- **An honest live-salt period meets every premise of `H1CheckedCreditPeriod`**, in its order, at its numbers
(`N_p = 144`, `K_d = 24`, one round, `ρ = 8/9`, `w_max = 1`, lucks `7/8` and `½`), any `δ_s > 0`,
`δ_bind = δ_ks = 0` and 128 fixed and folded words a window; the deployed check accepts every window on every path,
the accepted prefix is the whole period and its live words are `128·n`. -/
theorem h1_meets (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) :
    circuit.units ≤ 144 ∧ circuit.rounds = 1 ∧ Scope Honest.scheme circuit δs ∧
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * shareCone Honest.scheme circuit 144 t ≤ 1) ∧
    (∀ j, stageShare (tileStage circuit 144) (shareCone Honest.scheme circuit 144) j ≤ 1) ∧
    (∀ t : Fin 144, 0 < shareCone Honest.scheme circuit 144 t →
      tileLuck Honest.scheme circuit 144 t ≤ 7 / 8 ∧ tileLuck0 Honest.scheme circuit 144 t ≤ 1 / 2) ∧
    (play n).game = periodGame (tileStage circuit 144) (liveCone Honest.scheme circuit 144) 1 1 24 (period n) ∧
    (period n).EveryWindow (fun live _ => DeadConeDead Honest.scheme circuit 144 live) ∧
    PrefixOf (layer n) (play n).game ∧
    (∀ w, prob (fun x => (layer n).MovedIn x w) ≤ 0) ∧
    RegisteredBeforeDraw (windows n) ∧ DrawIsCoin hK24 prm (windows n) ∧
    ServiceSound prm (windows n) opened (stagedAt salt staged) 0 ∧
    SaltIsLiveCoin hr1 (salt (n := n)) enc ∧ StagedPins staged flat ∧
    (period n).EveryWindow
      (fun live W => PerLiveTile Honest.scheme circuit live W (fun k G _ => OperandHMu (Honest.scheme.mu k) G)) ∧
    (period n).EveryWindow
      (fun live W => PerLiveTile Honest.scheme circuit live W (fun k G _ => ReceiptLuckAt (Honest.scheme.rB k) G)) ∧
    (period n).EveryWindow
      (fun live W => PerLiveTile Honest.scheme circuit live W (fun k _ G0 => GuessLuckAt (Honest.scheme.guess k) G0)) ∧
    UnitFixed (play n) (layer n) (periodMode circuit (period n) (run n))
      (fun _ t => (tileWords Honest.scheme circuit 144 t : ℝ)) (fun _ => 128) ∧
    (∀ x w, Accepts prm ((windows n).reg x w) ((windows n).ev x w)) ∧
    (∀ x, (windows n).prefixLen prm x = n) ∧
    ∀ x, checkedLive Honest.scheme circuit (period n) prm (windows n) x = 128 * n := by
  refine ⟨by decide, rfl, scope hδs, fun t => ?_, fun j => ?_, fun t ht => ?_, rfl,
    period_every (P := fun live _ => DeadConeDead Honest.scheme circuit 144 live) (fun _ _ ht _ => by simp at ht) n,
    fun x j _ => acceptsUpTo_period _ _ _ n j x, fun w => ?_, fun _ _ _ => rfl, drawIsCoin n, serviceSound n,
    fun _ _ => rfl, fun _ _ _ h => h,
    every_perLiveTile (P := fun k G _ => OperandHMu (Honest.scheme.mu k) G)
      (fun k t => Honest.operandHMu_of_zero _ _ _ _ _ (paid_share .folded (by decide) t)) n,
    every_perLiveTile (P := fun k G _ => ReceiptLuckAt (Honest.scheme.rB k) G)
      (fun k t => Honest.receiptLuck_of_zero _ _ _ _ _ (paid_share .folded (by decide) t)) n,
    every_perLiveTile (P := fun k _ G0 => GuessLuckAt (Honest.scheme.guess k) G0)
      (fun k t => Honest.guessLuck_of_zero _ _ _ _ _ (paid_share .whole (by decide) t)) n,
    fun _ x _ => ⟨sum_div_le hn _ fun w j => ?_, sum_div_le hn _ fun w j => ?_⟩,
    accepted_all n, prefixLen_all n, checkedLive_all n⟩
  · rw [shareCone_eq]
    split_ifs <;> norm_num
  · unfold stageShare
    calc ∑ v, (if tileStage circuit 144 v = j then shareCone Honest.scheme circuit 144 v else 0)
        ≤ ∑ v, shareCone Honest.scheme circuit 144 v := Finset.sum_le_sum fun v _ => by
          split_ifs
          exacts [le_rfl, shareCone_nonneg v]
      _ = 1 := sum_shareCone
  · have hlt : t.val < 128 := by
      by_contra h
      rw [shareCone_eq, if_neg h] at ht
      exact lt_irrefl 0 ht
    have hk : tileKind circuit 144 t = some (kindAt circuit t.val) := by
      show (unitWords circuit t.val).head?.map (kindAt circuit) = _
      rw [unitWords_eq, if_pos hlt]
      rfl
    constructor
    · unfold tileLuck
      rw [hk]
      norm_num [Honest.scheme, Scheme.foldLuck]
    · unfold tileLuck0
      rw [hk]
      norm_num [Honest.scheme]
  · refine le_of_eq (Honest.prob_none ?_)
    rintro x ⟨i, -, h⟩
    exact h (funext fun _ => Subsingleton.elim _ _)
  · exact ite_words_le _ j
  · exact ite_zero_le _ j

/-- **`H1CheckedCreditPeriod`'s conclusion at the honest live-salt period**: `h1_meets` gives the instance every
premise, so except with probability `δ_s` its fixed words are at least `(1 − ε(n))₊` of its `128·n` live words. -/
theorem h1_credit (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) :
    prob (fun x => (play n).accepts x ∧ n * ((play n).run x).fixed <
      creditPrefix Honest.scheme circuit 144 ((windows n).prefixLen prm x)
        (checkedLive Honest.scheme circuit (period n) prm (windows n) x) 1 1 (67 / 8) δs) ≤ δs + n * (0 + 0) := by
  obtain ⟨hN, hM, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hStaged, hMu, hRB, hGs,
    hUF, -, -, -⟩ := h1_meets n hn hδs
  exact h1CheckedCreditPeriod Honest.scheme circuit hn hN hr1 hM hK24 prm (windows n) opened salt enc staged flat
    (period n) (play n) (fun _ => 128) δs 0 0 hS hρN hwmax hlk hG hDD hPre hMoved hReg hDraw hServ hSalt hStaged hMu
    hRB hGs hUF

end Properties.Pouw.Period.SecurityProofs.HonestH1
