import Proofs.Accounting.ComposePeriod
import Proofs.Accounting.Compose.SplitRateClosed

/-!
# An honest period meets `CircuitWorkPeriod`'s premises

The positive twin of `Accounting.Guarantees.ComposePeriod.CircuitWorkPeriod` (brief rule I, as I8-COIN's `Keyed.lean`
twins are of their layers): a period of `n ≥ 1` windows of a circuit of one gate, one credited word, one tile and one
live round, every tile live in every window, every stage paid with its words right and its checked values honest, and
every drawn tile's proof accepted (`honest_meets`). It meets every premise at live numbers: `K_d = 1`, `ρ = 1`,
`w_max = 1`, lucks `p = p0 = ½` (the operand luck pinned, `r_B = ½`, the guess bound `½`), costs `a = b = 1` and
`SplitRateClosed`'s rate there, `Λ = 4`; binding at `δ_bind = 0` and joint soundness at `δ_ks = 0` (its openings are in
`Unit`, so each is its committed value, and every relation holds), and `UnitFixed` at one fixed and one folded word a
window. Its play is accepted on every path, its accepted prefix is the whole period, and its credited live words are
`n`, so the conclusion speaks of `n` live words, not of none.
-/

namespace Accounting.SecurityProofs.ComposePeriod.Honest

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

theorem prob_none {Ω : Type} [Fintype Ω] {p : Ω → Prop} (h : ∀ ω, ¬ p ω) : prob p = 0 := by
  have hp : p = fun _ => False := funext fun ω => propext ⟨h ω, False.elim⟩
  rw [hp]
  exact Accounting.SecurityProofs.TS.Rounds.prob_false

/-! ## The circuit, its scheme and the honest period -/

/-- The word: step `0` of an `S8Dot_v1` body of `K = L = 32` (its `T_1`, a served step), committed. -/
def word : Accounting.Gates.Gate Unit where
  op := .const 0
  args := []
  lo := 0
  hi := 0
  step := 0
  prim := "S8Dot_v1"
  site := .dot 32 32 [] 0
  commit := true

/-- One gate, one call site, one live round, one unit, every value committed before the round's coin. -/
def circuit : Declared Unit where
  dag := [word]
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun _ => 0
  seedNames := fun _ _ => false
  units := 1
  unit := fun _ => 0

/-- Every kind credited, its operand luck pinned (`0`), its receipt luck and guess bound `½`. -/
noncomputable def scheme : Scheme Unit where
  floor := fun _ => ⟨fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0⟩
  credited := fun _ => true
  span := fun _ => 0
  form := fun _ => .pinned
  mu := fun _ => 0
  rB := fun _ => 1 / 2
  guess := fun _ => 1 / 2
  seedRate := fun _ => 0
  wholeRate := fun _ => 0

/-- A window's honest stages: its tile paid, its words right and its checked values honest. -/
def stages : SplitStages Bool (Fin 1) 1 :=
  .stage (fun _ _ => .paid) (fun _ _ => True) (fun _ _ => True) fun _ => .done

/-- The honest period of `n` windows: every tile live, each window's stages `stages`, whatever the past. -/
def period : (n : ℕ) → Period Bool (Paths Bool 1 × (Equiv.Perm (Fin 1) × Unit)) (Fin 1) 1 n
  | 0 => .done
  | n + 1 => .window (fun _ => true) stages fun _ => period n

/-- Its proofs' run: every drawn tile's proof accepted, opening values in `Unit`, every relation and rightness
holding. -/
def run (n : ℕ) : PeriodRun Bool (Equiv.Perm (Fin 1) × Unit) 1 1 1 n Unit where
  accepted := fun _ _ _ => True
  opened := fun _ _ _ _ => ()
  rel := fun _ _ _ _ => True
  right := fun _ _ _ _ => True

/-- The period's proofs, `Declared.periodLayer` at the honest run. -/
def layer (n : ℕ) : DrawnLayer Bool (Equiv.Perm (Fin 1) × Unit) n 1 1 (Fin 1) ℕ (Fin n × ℕ) Unit :=
  circuit.periodLayer (Nat.one_pos : 0 < circuit.rounds) (le_rfl : 1 ≤ 1) (run n)

/-- The honest run's anatomy: one fixed word a window. -/
noncomputable def anatomy : Anatomy Unit Unit Unit Unit where
  run := ⟨fun _ => 0, 0, 0⟩
  nword := fun _ _ => 0
  nfold := fun _ => 0
  nfill := fun _ => 0
  words := fun _ => 0
  fixed := 1

/-- The play: the period's ideal game at `liveCone`, accepted on every path. -/
noncomputable def play (n : ℕ) : Play (Paths Bool 1 × (Equiv.Perm (Fin 1) × Unit)) n Unit Unit Unit Unit where
  game := periodGame (tileStage circuit 1) (liveCone scheme circuit 1) 1 1 1 (period n)
  accepts := fun _ => True
  run := fun _ => anatomy

/-! ## The circuit's numbers -/

theorem tileWords_eq (t : Fin 1) : tileWords scheme circuit 1 t = 1 := by
  rw [Fin.fin_one_eq_zero t]
  decide

theorem wordsOf_eq : wordsOf scheme circuit = 1 := by decide

theorem tileKind_eq (t : Fin 1) : tileKind circuit 1 t = some (kindAt circuit 0) := by
  rw [Fin.fin_one_eq_zero t]
  rfl

theorem tileCone_eq (t : Fin 1) : tileCone scheme circuit 1 t = 1 := by
  classical
  rw [Fin.fin_one_eq_zero t]
  unfold tileCone tileWeight
  rw [Fin.sum_univ_one,
    ite_eq_left (Relation.ReflTransGen.refl : circuit.Reaches (0 : Fin 1).val (0 : Fin 1).val), tileWords_eq,
    wordsOf_eq]
  norm_num

theorem shareCone_eq (t : Fin 1) : shareCone scheme circuit 1 t = 1 := by
  have h0 : tileWords scheme circuit 1 t ≠ 0 := by
    rw [tileWords_eq]
    exact one_ne_zero
  unfold shareCone
  rw [ite_eq_right h0, tileCone_eq]

theorem circuit_wf : circuit.Wf := by
  refine ⟨fun _ _ => ⟨Nat.one_pos, Nat.one_pos⟩, fun _ _ => Nat.one_pos, fun _ _ _ _ _ => rfl,
    fun _ _ _ _ _ _ => rfl, fun i hi j hj _ => ?_⟩
  obtain rfl : i = 0 := Nat.lt_one_iff.mp (List.mem_range.mp (List.mem_filter.mp hi).1)
  obtain rfl : j = 0 := Nat.lt_one_iff.mp (List.mem_range.mp (List.mem_filter.mp hj).1)
  rfl

theorem scope {δs : ℝ} (hδs : 0 < δs) : Scope scheme circuit δs :=
  ⟨circuit_wf, fun _ _ _ => rfl, hδs, fun _ => by norm_num [scheme, LuckForm.value],
    fun _ _ => Nat.zero_le _⟩

/-- `SplitRateClosed` at `K_d = 1`, `ρ = 1`, `w_max = 1`, `p = p0 = ½`, `a = b = 1`: `Λ = 4`. -/
theorem splitRate : SplitRate 1 1 (1 / 2) (1 / 2) 1 1 1 4 := by
  have h := Accounting.SecurityProofs.Compose.Rates.splitRateClosed 1 1 (1 / 2) (1 / 2) 1 1 1 le_rfl one_pos
    zero_le_one (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) one_pos one_pos
  convert h using 1
  norm_num

/-! ## The honest window's claims -/

theorem operandHMu_of_zero {Ω : Type} [Fintype Ω] (μ : ℝ) (d : Ω → ℝ) (pass right same : Ω → Prop)
    (hd : ∀ ω, d ω = 0) : OperandHMu μ (ReceiptGame.round d pass right same fun _ => .done) :=
  ⟨fun ω => by rw [hd ω]; exact ⟨le_rfl, zero_le_one⟩,
    ⟨Unit, Unit, ⟨fun _ => (), fun _ => (), fun _ => 0, fun _ => (), fun _ x => x⟩,
      fun _ h => absurd h (lt_irrefl 0), fun _ _ => rfl, hd, fun _ _ => rfl⟩,
    fun _ => trivial⟩

theorem receiptLuck_of_zero {Ω : Type} [Fintype Ω] (r : ℝ) (d : Ω → ℝ) (pass right same : Ω → Prop)
    (hd : ∀ ω, d ω = 0) : ReceiptLuckAt r (ReceiptGame.round d pass right same fun _ => .done) := by
  have e1 : prob (fun ω => same ω ∧ ¬right ω ∧ 0 < d ω) = 0 :=
    prob_none fun ω h => absurd (h.2.2.trans_eq (hd ω)) (lt_irrefl 0)
  have e2 : prob (fun ω => ¬right ω ∧ 0 < d ω) = 0 :=
    prob_none fun ω h => absurd (h.2.trans_eq (hd ω)) (lt_irrefl 0)
  exact ⟨e1.le.trans (le_of_eq (by rw [e2, mul_zero])), fun _ => trivial⟩

theorem guessLuck_of_zero {Ω : Type} [Fintype Ω] (g : ℝ) (d : Ω → ℝ) (pass right same : Ω → Prop)
    (hd : ∀ ω, d ω = 0) : GuessLuckAt g (ReceiptGame.round d pass right same fun _ => .done) := by
  have e1 : prob (fun ω => pass ω ∧ 0 < d ω) = 0 :=
    prob_none fun ω h => absurd (h.2.trans_eq (hd ω)) (lt_irrefl 0)
  have e2 : prob (fun ω => 0 < d ω) = 0 :=
    prob_none fun ω h => absurd (h.trans_eq (hd ω)) (lt_irrefl 0)
  exact ⟨fun ω => by rw [hd ω]; exact ⟨le_rfl, zero_le_one⟩, e1.le.trans (le_of_eq (by rw [e2, mul_zero])),
    fun _ => trivial⟩

theorem paid_share (s : Mode) (hs : s ≠ .paid) (t : Fin 1) (c : Bool) :
    (fun c : Bool => if (fun (_ : Bool) (_ : Fin 1) => Mode.paid) c t = s then (1 : ℝ) else 0) c = 0 :=
  ite_eq_right fun h => hs h.symm

/-- **A claim of every live credited tile of the honest stages**: each of their tile games has share `0`. -/
theorem perLiveTile_stages {live : Fin 1 → Bool} {P : Kind → ReceiptGame Bool 1 → ReceiptGame Bool 1 → Prop}
    (h : ∀ k t, P k (tileGame (fun _ _ => Mode.paid) (fun _ _ => True) (fun _ _ => True) .folded t)
      (tileGame (fun _ _ => Mode.paid) (fun _ _ => True) (fun _ _ => True) .whole t)) :
    PerLiveTile scheme circuit live stages P :=
  ⟨fun t _ _ k _ _ => h k t, fun _ => trivial⟩

theorem period_every {P : (Fin 1 → Bool) → SplitStages Bool (Fin 1) 1 → Prop} (h : P (fun _ => true) stages) :
    ∀ n, (period n).EveryWindow P
  | 0 => trivial
  | n + 1 => ⟨h, fun _ => period_every h n⟩

theorem every_perLiveTile {P : Kind → ReceiptGame Bool 1 → ReceiptGame Bool 1 → Prop}
    (h : ∀ k t, P k (tileGame (fun _ _ => Mode.paid) (fun _ _ => True) (fun _ _ => True) .folded t)
      (tileGame (fun _ _ => Mode.paid) (fun _ _ => True) (fun _ _ => True) .whole t)) (n : ℕ) :
    (period n).EveryWindow (fun live W => PerLiveTile scheme circuit live W P) :=
  period_every (P := fun live W => PerLiveTile scheme circuit live W P) (perLiveTile_stages h) n

theorem liveAt_period : ∀ (n : ℕ) (x : Paths (Paths Bool 1 × (Equiv.Perm (Fin 1) × Unit)) n) (w : Fin n),
    (period n).liveAt x w = fun _ => true
  | 0, _, w => w.elim0
  | n + 1, x, w => by
    refine Fin.cases ?_ (fun w' => ?_) w
    · rfl
    · exact liveAt_period n x.2 w'

/-- **The honest period passes every prefix of its game**, at any weights and costs: every window's checked values
are honest. -/
theorem acceptsUpTo_period (wt : (Fin 1 → Bool) → Fin 1 → ℝ) (a b : ℝ) :
    ∀ (n j : ℕ) (x : Paths (Paths Bool 1 × (Equiv.Perm (Fin 1) × Unit)) n),
      RoundGame.acceptsUpTo (periodGame (tileStage circuit 1) wt a b 1 (period n)) j x
  | 0, _, _ => trivial
  | _ + 1, 0, _ => trivial
  | n + 1, j + 1, x =>
    ⟨fun v _ => by simp [stages, SplitStages.joint, SplitStages.sameAt], acceptsUpTo_period wt a b n j x.2⟩

theorem sum_div_le_one {n : ℕ} (hn : 0 < n) (f : Fin n → Fin 1 → ℝ) (hf : ∀ w j, f w j ≤ 1) :
    (∑ w, ∑ j, f w j) / n ≤ 1 := by
  rw [div_le_one (Nat.cast_pos.mpr hn)]
  calc ∑ w, ∑ j, f w j ≤ ∑ _w : Fin n, (1 : ℝ) :=
        Finset.sum_le_sum fun w _ => by rw [Fin.sum_univ_one]; exact hf w 0
    _ = n := by simp

/-! ## Every premise -/

/-- **An honest period meets every premise of `CircuitWorkPeriod`**, in its order, at `K_d = 1`, `ρ = 1`, `w_max = 1`,
`p = p0 = ½`, `a = b = 1`, `Λ = 4`, any `δ_s > 0`, `δ_bind = δ_ks = 0`, one fixed and one folded word a window; and
it is accepted on every path, with the whole period its accepted prefix and `n` credited live words. -/
theorem honest_meets (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) :
    circuit.units ≤ 1 ∧ Scope scheme circuit δs ∧
    (∀ t : Fin 1, 1 * ((1 : ℕ) : ℝ) * shareCone scheme circuit 1 t ≤ 1) ∧
    (∀ j, stageShare (tileStage circuit 1) (shareCone scheme circuit 1) j ≤ 1) ∧
    (∀ t : Fin 1, 0 < shareCone scheme circuit 1 t →
      tileLuck scheme circuit 1 t ≤ 1 / 2 ∧ tileLuck0 scheme circuit 1 t ≤ 1 / 2) ∧
    SplitRate 1 1 (1 / 2) (1 / 2) 1 1 1 4 ∧
    (play n).game = periodGame (tileStage circuit 1) (liveCone scheme circuit 1) 1 1 1 (period n) ∧
    (period n).EveryWindow (fun live _ => DeadConeDead scheme circuit 1 live) ∧
    PrefixOf (layer n) (play n).game ∧
    (∀ w, prob (fun x => (layer n).MovedIn x w) ≤ 0) ∧ (layer n).JointSound 0 ∧ (layer n).Pins ∧
    (period n).EveryWindow (fun live W => PerLiveTile scheme circuit live W (fun k G _ => OperandHMu (scheme.mu k) G)) ∧
    (period n).EveryWindow
      (fun live W => PerLiveTile scheme circuit live W (fun k G _ => ReceiptLuckAt (scheme.rB k) G)) ∧
    (period n).EveryWindow
      (fun live W => PerLiveTile scheme circuit live W (fun k _ G0 => GuessLuckAt (scheme.guess k) G0)) ∧
    UnitFixed (play n) (layer n) (periodMode circuit (period n) (run n)) (fun _ t => (tileWords scheme circuit 1 t : ℝ))
      (fun _ => 1) ∧
    (∀ x, (play n).accepts x) ∧ (∀ x, (run n).prefixLen x = n) ∧
    ∀ x, creditedLive scheme circuit (period n) (run n) x = n := by
  have hc : ∀ x w, (run n).Credited x w := fun _ _ _ _ _ => trivial
  refine ⟨le_rfl, scope hδs, fun t => by rw [shareCone_eq]; norm_num, fun j => ?_, fun t _ => ?_, splitRate, rfl,
    period_every (P := fun live _ => DeadConeDead scheme circuit 1 live) (fun t _ ht _ => by simp at ht) n,
    fun x j _ => acceptsUpTo_period _ _ _ n j x, fun w => ?_, fun w => ?_, fun _ _ _ _ _ => trivial,
    every_perLiveTile (P := fun k G _ => OperandHMu (scheme.mu k) G)
      (fun k t => operandHMu_of_zero _ _ _ _ _ (paid_share .folded (by decide) t)) n,
    every_perLiveTile (P := fun k G _ => ReceiptLuckAt (scheme.rB k) G)
      (fun k t => receiptLuck_of_zero _ _ _ _ _ (paid_share .folded (by decide) t)) n,
    every_perLiveTile (P := fun k _ G0 => GuessLuckAt (scheme.guess k) G0)
      (fun k t => guessLuck_of_zero _ _ _ _ _ (paid_share .whole (by decide) t)) n,
    fun _ x _ => ⟨sum_div_le_one hn _ fun w j => ?_, sum_div_le_one hn _ fun w j => ?_⟩,
    fun _ => trivial, fun x => ?_, fun x => ?_⟩
  · unfold stageShare
    rw [Fin.sum_univ_one, shareCone_eq]
    split_ifs <;> norm_num
  · have hk := tileKind_eq t
    constructor
    · unfold tileLuck
      rw [hk]
      norm_num [scheme, Scheme.foldLuck]
    · unfold tileLuck0
      rw [hk]
      norm_num [scheme]
  · refine le_of_eq (prob_none ?_)
    rintro x ⟨i, -, h⟩
    exact h (funext fun _ => Subsingleton.elim _ _)
  · refine le_of_eq (prob_none ?_)
    rintro x ⟨i, -, h⟩
    exact h trivial
  · split_ifs <;> simp [tileWords_eq]
  · split_ifs <;> simp [tileWords_eq]
  · unfold PeriodRun.prefixLen
    rw [Finset.filter_true_of_mem fun w _ => hc x w, Finset.card_univ, Fintype.card_fin]
  · simp [creditedLive, hc, liveAt_period, liveWords, tileWords_eq]

/-- **`CircuitWorkPeriod`'s conclusion at the honest period**, from the period layer's covers: `honest_meets` gives
`circuitWorkPeriod_of_covers` every premise in its own form. -/
theorem honest_conclusion (hC : PeriodCoversStmt) (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) :
    prob (fun x => (play n).accepts x ∧
      ¬ LiveWorkAt (creditedLive scheme circuit (period n) (run n) x) (wordsOf scheme circuit : ℝ) 1 1
        (rOfPrefix scheme circuit 1 ((run n).prefixLen x) 1 1 4 δs) (n * ((play n).run x).fixed) (n * 1)) ≤
      δs + n * (0 + 0) := by
  obtain ⟨hN, hS, hρN, hwmax, hlk, hSR, hG, hDD, hPre, hMoved, hJS, hPins, hMu, hRB, hGs, hUF, -, -, -⟩ :=
    honest_meets n hn hδs
  exact circuitWorkPeriod_of_covers hC scheme circuit hn hN Nat.one_pos le_rfl (period n) (play n) (run n)
    (fun _ => 1) 1 (1 / 2) (1 / 2) 1 1 1 4 δs 0 0 hS one_pos hρN hwmax hlk zero_le_one zero_le_one (by norm_num) hSR
    hG hDD hPre hMoved hJS hPins hMu hRB hGs hUF

end Accounting.SecurityProofs.ComposePeriod.Honest
