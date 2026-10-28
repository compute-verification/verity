import Proofs.Pouw.PeriodJ10R
import Proofs.Accounting.ComposePeriod.Honest
import Proofs.Accounting.TS.Keyed

/-!
# An honest J10-R period meets `H1J10RCheckedCreditPeriod`'s premises

The positive twin of `Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod`, as `Proofs.Pouw.Period.Honest` is of
`H1CheckedCreditPeriod`, conditional on `Credit.window`'s verdict on the windows' inputs. Over any sampling block's tile
stratum `s_T` of `σ` (`N_p = |stratum σ s_T|`, `1 ≤ k s_T`), take a period of `n ≥ 1` windows whose registration is a
function of the window's round coin (`regOf`) and the rest of whose inputs a function of its round and draw coins
(`evOf`). Given three facts of those inputs, the witness meets every premise of the statement (`j10r_meets`):

* `Credit.window` credits every window (`hacc`);
* the registered salt, hex-decoded, is the round coin's bytes (`hsalt`), so `SaltIsLiveCoin` holds with no round trip
  through `hex?`;
* the units `Credit.descriptorDrawn` draws are the tile coordinate's at the draw coin (`hdraw`), `TileDrawIsCoin`'s
  body on every window.

They are facts of the code on honest windows, which Lean doesn't evaluate here (`Credit.window` reads SHA-512 trees and
C-Flock's proofs). The compiled verifier evidences them: PouwCheck's `test_difftest` has `pouw-check-difftest`'s
`window` entry (`Credit.window`) credit honest J10-R windows (`window_instance`'s `j10r`) as the Python verifier does,
and its `window-drawn` entry draw the Python verifier's units on generated coins.

* The circuit (`circuit N`): `N = N_p` committed words of no operands, one a unit, one live round, so every tile's late
  cone is its own word and its share `1/N` (`shareCone_eq`): `ρ = 1` meets `ρ·N_p·share ≤ 1` with equality.
* The scheme is the ComposePeriod twin's (operand luck pinned, `r_B = ½`, guess `½`).
* The registration reads the round coin and not the draw coin, so it is fixed before the draw
  (`registeredBeforeDraw`, by `coinAt_spliceAt`).
* Every tile is live and paid with its words right, the openings are in `Unit` and the unit's relation is `True`, so
  binding holds at `δ_bind = 0` and the service's soundness at `δ_ks = 0`; the three lucks hold of the honest games
  (`SharedNoiseLucks`), and the play fixes and folds `N` words a window (`UnitFixed`).

The conclusion then speaks of the whole period: its accepted prefix is all `n` windows (`prefixLen_all`) and its live
words are `N·n` (`creditLive_all`), so `j10r_credit` is the statement's bound at `N·n` live words, not at none.
-/

namespace Properties.Pouw.PeriodJ10R.SecurityProofs.HonestJ10R

open Properties.Pouw.PeriodJ10R Properties.Pouw.PeriodJ10R.SecurityProofs
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposePeriod

/-! ## The circuit -/

/-- `N` committed words of no operands, one a unit, one live round. -/
def circuit (N : ℕ) : Declared Unit where
  dag := List.replicate N Honest.word
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun _ => 0
  seedNames := fun _ _ => false
  units := N
  unit := fun i => i

theorem hr1 (N : ℕ) : 0 < (circuit N).rounds := Nat.one_pos

theorem length_dag (N : ℕ) : (circuit N).dag.length = N := List.length_replicate

theorem wordGates_eq (N : ℕ) : wordGates (circuit N) = List.range N := by
  unfold wordGates
  rw [length_dag]
  refine List.filter_eq_self.2 fun i hi => ?_
  show ((List.replicate N Honest.word)[i]?).any isWord = true
  rw [List.getElem?_replicate_of_lt (List.mem_range.1 hi)]
  decide

theorem unitWords_eq (N u : ℕ) : unitWords (circuit N) u = if u < N then [u] else [] := by
  unfold unitWords
  rw [wordGates_eq]
  show (List.range N).filter (· == u) = _
  rw [List.filter_beq, List.count_range]
  by_cases h : u < N <;> simp [h]

theorem tileWords_eq (N : ℕ) (t : Fin N) : tileWords Honest.scheme (circuit N) N t = 1 := by
  unfold tileWords
  rw [unitWords_eq, if_pos t.isLt]
  simp [Honest.scheme]

theorem wordsOf_eq (N : ℕ) : wordsOf Honest.scheme (circuit N) = N := by
  unfold wordsOf creditedWords
  rw [wordGates_eq]
  simp [Honest.scheme]

theorem not_reads (N j i : ℕ) : ¬ (circuit N).Reads j i := by
  show ¬ ((List.replicate N Honest.word)[j]?.any fun g => g.args.contains i) = true
  rw [List.getElem?_replicate]
  split <;> simp [Honest.word]

/-- **No unit writes another late**: the circuit's late cones are its units. -/
theorem reaches_iff (N u v : ℕ) : (circuit N).Reaches u v ↔ u = v := by
  constructor
  · intro h
    have h' : Relation.ReflTransGen (circuit N).UnitLate u v := h
    induction h' with
    | refl => rfl
    | tail _ hbc _ =>
      obtain ⟨i, j, -, -, -, hl⟩ := hbc
      exact absurd hl.1 (not_reads N j i)
  · rintro rfl
    exact Relation.ReflTransGen.refl

theorem tileCone_eq (N : ℕ) (t : Fin N) :
    tileCone Honest.scheme (circuit N) N t = tileWeight Honest.scheme (circuit N) N t := by
  unfold tileCone
  rw [Finset.sum_eq_single t]
  · exact if_pos ((reaches_iff _ _ _).2 rfl)
  · intro t' _ ht'
    exact if_neg fun h => ht' (Fin.ext ((reaches_iff _ _ _).1 h)).symm
  · intro h
    exact absurd (Finset.mem_univ t) h

theorem shareCone_eq (N : ℕ) (t : Fin N) : shareCone Honest.scheme (circuit N) N t = 1 / N := by
  unfold shareCone
  rw [tileCone_eq, tileWeight, tileWords_eq, wordsOf_eq]
  simp

theorem pos_of_fin {N : ℕ} (t : Fin N) : (0 : ℝ) < N := Nat.cast_pos.mpr (Nat.lt_of_le_of_lt (Nat.zero_le _) t.isLt)

theorem sum_shareCone {N : ℕ} (hN : 0 < N) : ∑ t : Fin N, shareCone Honest.scheme (circuit N) N t = 1 := by
  simp only [shareCone_eq, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

theorem sum_tileWords (N : ℕ) : ∑ t : Fin N, (tileWords Honest.scheme (circuit N) N t : ℝ) = N := by
  simp [tileWords_eq]

theorem circuit_wf (N : ℕ) : (circuit N).Wf := by
  refine ⟨fun i hi => ⟨Nat.one_pos, lt_of_lt_of_eq hi (length_dag N)⟩, fun _ _ => Nat.one_pos,
    fun _ _ _ _ _ => rfl, fun g hg => by obtain rfl := List.eq_of_mem_replicate hg; rfl,
    fun i _ j _ h => ?_⟩
  have e : i = j := h
  rw [e]

theorem scope (N : ℕ) {δs : ℝ} (hδs : 0 < δs) : Scope Honest.scheme (circuit N) δs :=
  ⟨circuit_wf N, fun _ _ _ => rfl, hδs, fun _ => by norm_num [Honest.scheme, LuckForm.value],
    fun _ _ => Or.inl (Nat.zero_le _)⟩

/-! ## The period, its windows and its proofs -/

/-- A window's honest stages: every tile paid, its words right and its checked values honest. -/
def stages (N : ℕ) : SplitStages Bool (Fin N) 1 :=
  .stage (fun _ _ => .paid) (fun _ _ => True) (fun _ _ => True) fun _ => .done

/-- A window's coins: its round's coin, the draw's permutation of the `N` tiles and the rest of the draw `B`. -/
abbrev WCoin (N : ℕ) (B : Type) : Type := Paths Bool (circuit N).rounds × (Equiv.Perm (Fin N) × B)

/-- The honest period of `n` windows: every tile live, each window's stages `stages`, whatever the past. -/
def period (Ω : Type) (N : ℕ) : (n : ℕ) → Period Bool Ω (Fin N) (circuit N).rounds n
  | 0 => .done
  | n + 1 => .window (fun _ => true) (stages N) fun _ => period Ω N n

section Windows

variable {nB mB : ℕ} (σ : Fin nB → Fin mB) (sT : Fin mB)

/-- A window's coins over the tile stratum: its round's coin, the stratum's permutation and the rest of the block's
draw (none). -/
abbrev Ω1 : Type := Coin Bool Unit (circuit (tileCount σ sT)).rounds σ sT

/-- The period's windows as `Credit.window` reads them: window `w` registers `regOf` of its round coin, and the rest of
its inputs are `evOf` of its round coin and its draw coin. -/
def windows (regOf : Bool → Registered) (evOf : Bool → Equiv.Perm (Fin (tileCount σ sT)) × Unit → Answered)
    (n : ℕ) : CreditWindows (Ω1 σ sT) n where
  reg x w := regOf (coinAt (coinAt x w).1 ⟨0, hr1 _⟩)
  ev x w := evOf (coinAt (coinAt x w).1 ⟨0, hr1 _⟩) (coinAt x w).2

end Windows

/-- The service's openings, in `Unit`. -/
def opened {Ω : Type} {n : ℕ} : Paths Ω n → Fin n → ℕ → ℕ → Unit := fun _ _ _ _ => ()

/-- Every unit's place: one place, which the relation `tile` doesn't read. -/
def place {Ω : Type} {n Np : ℕ} : Paths Ω n → Fin n → Fin Np → Place := fun _ _ _ => ⟨0, 0, 0, 0, 0, 0, 0, 0, [], []⟩

/-- The unit's relation: every opening meets it. -/
def tile {Np : ℕ} : Noise → List (List ℕ) → Fin Np → (ℕ → Unit) → Prop := fun _ _ _ _ => True

/-- The honest run's anatomy: the `N` words fixed. -/
noncomputable def anatomy (N : ℕ) : Anatomy Unit Unit Unit Unit := { Honest.anatomy with fixed := N }

/-- The play: the period's ideal game at `liveCone`, drawing `Kd` tiles a window, accepted on every path. -/
noncomputable def play (B : Type) (N Kd n : ℕ) : Play (WCoin N B) n Unit Unit Unit Unit where
  game := periodGame (tileStage (circuit N) N) (liveCone Honest.scheme (circuit N) N) 1 1 Kd (period (WCoin N B) N n)
  accepts := fun _ => True
  run := fun _ => anatomy N

/-! ## The windows' facts -/

/-- **The registration is fixed before the draw coin**: it reads only the round coin, which the splice at window `w`
after its one round keeps. -/
theorem registeredBeforeDraw {nB mB : ℕ} (σ : Fin nB → Fin mB) (sT : Fin mB) (regOf : Bool → Registered)
    (evOf : Bool → Equiv.Perm (Fin (tileCount σ sT)) × Unit → Answered) (n : ℕ) :
    RegisteredBeforeDraw (windows σ sT regOf evOf n) := by
  intro x z w
  show regOf (coinAt (coinAt (spliceAt w.val (circuit (tileCount σ sT)).rounds x z) w).1 ⟨0, hr1 _⟩) =
    regOf (coinAt (coinAt x w).1 ⟨0, hr1 _⟩)
  rw [Accounting.SecurityProofs.TS.Keyed.coinAt_spliceAt]
  rfl

theorem period_every {Ω : Type} {N : ℕ} {P : (Fin N → Bool) → SplitStages Bool (Fin N) 1 → Prop}
    (h : P (fun _ => true) (stages N)) : ∀ n, (period Ω N n).EveryWindow P
  | 0 => trivial
  | n + 1 => ⟨h, fun _ => period_every h n⟩

theorem liveAt_period {Ω : Type} {N : ℕ} :
    ∀ (n : ℕ) (x : Paths Ω n) (w : Fin n), (period Ω N n).liveAt x w = fun _ => true
  | 0, _, w => w.elim0
  | n + 1, x, w => by
    refine Fin.cases ?_ (fun w' => ?_) w
    · rfl
    · exact liveAt_period n x.2 w'

theorem liveWords_true {N : ℕ} (t : Fin N) :
    liveWords Honest.scheme (circuit N) N (fun _ => true) t = tileWords Honest.scheme (circuit N) N t := by
  unfold liveWords
  exact if_pos rfl

theorem paid_share {N : ℕ} (s : Mode) (hs : s ≠ .paid) (t : Fin N) (c : Bool) :
    (fun c : Bool => if (fun (_ : Bool) (_ : Fin N) => Mode.paid) c t = s then (1 : ℝ) else 0) c = 0 :=
  ite_eq_right fun h => hs h.symm

theorem every_perLiveTile {Ω : Type} {N : ℕ} {P : Kind → ReceiptGame Bool 1 → ReceiptGame Bool 1 → Prop}
    (h : ∀ k (t : Fin N), P k (tileGame (fun _ _ => Mode.paid) (fun _ _ => True) (fun _ _ => True) .folded t)
      (tileGame (fun _ _ => Mode.paid) (fun _ _ => True) (fun _ _ => True) .whole t)) (n : ℕ) :
    (period Ω N n).EveryWindow (fun live W => PerLiveTile Honest.scheme (circuit N) live W P) :=
  period_every (Ω := Ω) (N := N) (P := fun live W => PerLiveTile Honest.scheme (circuit N) live W P)
    ⟨fun t _ _ k _ _ => h k t, fun _ => trivial⟩ n

/-- **The honest period passes every prefix of its game**, at any weights, costs and draw size. -/
theorem acceptsUpTo_period {B : Type} {N : ℕ} (wt : (Fin N → Bool) → Fin N → ℝ) (a b : ℝ) (Kd : ℕ) :
    ∀ (n j : ℕ) (x : Paths (WCoin N B) n),
      RoundGame.acceptsUpTo (periodGame (tileStage (circuit N) N) wt a b Kd (period (WCoin N B) N n)) j x
  | 0, _, _ => trivial
  | _ + 1, 0, _ => trivial
  | n + 1, j + 1, x =>
    ⟨fun v _ => by simp [stages, SplitStages.joint, SplitStages.sameAt], acceptsUpTo_period wt a b Kd n j x.2⟩

theorem ite_words_le {N : ℕ} (c : Prop) {_ : Decidable c} (j : Fin N) :
    (if c then (tileWords Honest.scheme (circuit N) N j : ℝ) else 0) ≤ tileWords Honest.scheme (circuit N) N j := by
  split_ifs
  exacts [le_rfl, Nat.cast_nonneg _]

theorem ite_zero_le {N : ℕ} (c : Prop) {_ : Decidable c} (j : Fin N) :
    (if c then 0 else (tileWords Honest.scheme (circuit N) N j : ℝ)) ≤ tileWords Honest.scheme (circuit N) N j := by
  split_ifs
  exacts [Nat.cast_nonneg _, le_rfl]

theorem sum_div_le {N n : ℕ} (hn : 0 < n) (f : Fin n → Fin N → ℝ)
    (hf : ∀ w j, f w j ≤ tileWords Honest.scheme (circuit N) N j) : (∑ w, ∑ j, f w j) / n ≤ N := by
  rw [div_le_iff₀ (Nat.cast_pos.mpr hn)]
  calc ∑ w, ∑ j, f w j ≤ ∑ _w : Fin n, (N : ℝ) :=
        Finset.sum_le_sum fun w _ => (Finset.sum_le_sum fun j _ => hf w j).trans_eq (sum_tileWords N)
    _ = N * n := by simp [mul_comm]

/-! ## Every premise -/

section Meets

variable {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB)
  (unitOf : Fin nB → ℕ) (regOf : Bool → Registered)
  (evOf : Bool → Equiv.Perm (Fin (tileCount σ sT)) × Unit → Answered) (enc : Bool → ByteArray)

/-- **Every window credited**, its prefix every window. -/
theorem prefixLen_all (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β)) (n : ℕ) (x : Paths (Ω1 σ sT) n) :
    (windows σ sT regOf evOf n).prefixLen x = n := by
  classical
  unfold CreditWindows.prefixLen
  have h : ∀ w ∈ (Finset.univ : Finset (Fin n)), (windows σ sT regOf evOf n).AcceptedUpTo x w :=
    fun _ _ _ _ => hacc _ _
  rw [Finset.filter_true_of_mem h, Finset.card_univ, Fintype.card_fin]

/-- **The accepted prefix's live words are `N·n`.** -/
theorem creditLive_all (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β)) (n : ℕ) (x : Paths (Ω1 σ sT) n) :
    creditLive Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
      (windows σ sT regOf evOf n) x = tileCount σ sT * n := by
  unfold creditLive
  calc _ = ∑ _w : Fin n, (tileCount σ sT : ℝ) := Finset.sum_congr rfl fun w _ => by
          have hw : (windows σ sT regOf evOf n).AcceptedUpTo x w := fun _ _ => hacc _ _
          rw [if_pos hw, liveAt_period n x w]
          simp only [liveWords_true]
          exact sum_tileWords _
    _ = tileCount σ sT * n := by simp [mul_comm]

/-- **An honest J10-R period meets every premise of `H1J10RCheckedCreditPeriod`**, in its order, at `N_p = |stratum σ
s_T|` one-word units, one round, `ρ = 1`, the twin's lucks, any `δ_s > 0`, `δ_bind = δ_ks = 0` and `N_p` fixed and
folded words a window, given that `Credit.window` credits every window (`hacc`), that the registered salt is the
round coin's bytes (`hsalt`) and that `Credit.descriptorDrawn`'s units are the tile coordinate's at the draw coin
(`hdraw`); the accepted prefix is then the whole period and its live words are `N_p·n`. -/
theorem j10r_meets (hKd : 1 ≤ k sT) (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β))
    (hsalt : ∀ c, (PouwCheck.hex? (regOf c).salt).getD ByteArray.empty = enc c)
    (hdraw : ∀ c β, ∃ d : Array ℕ, creditDrawn (regOf c) (evOf c β) = .ok d ∧
      ∀ u : ℕ, u ∈ d.toList ↔ ∃ i : Fin (k sT), unitOf (tileAt σ sT (drawOf (hk sT) β i)) = u)
    (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) :
    (circuit (tileCount σ sT)).units ≤ tileCount σ sT ∧ (circuit (tileCount σ sT)).rounds = 1 ∧
    Scope Honest.scheme (circuit (tileCount σ sT)) δs ∧
    (∀ t : Fin (tileCount σ sT),
      (1 : ℝ) * (tileCount σ sT) * shareCone Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t ≤ 1) ∧
    (∀ j, stageShare (tileStage (circuit (tileCount σ sT)) (tileCount σ sT))
      (shareCone Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)) j ≤ 1) ∧
    (∀ t : Fin (tileCount σ sT), 0 < shareCone Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t →
      tileLuck Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t ≤ 7 / 8 ∧
        tileLuck0 Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t ≤ 1 / 2) ∧
    (play Unit (tileCount σ sT) (k sT) n).game =
      periodGame (tileStage (circuit (tileCount σ sT)) (tileCount σ sT))
        (liveCone Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)) 1 1 (k sT)
        (period (Ω1 σ sT) (tileCount σ sT) n) ∧
    (period (Ω1 σ sT) (tileCount σ sT) n).EveryWindow
      (fun live _ => DeadConeDead Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) live) ∧
    PrefixOf (tileLayer (circuit (tileCount σ sT)) (hr1 _) σ k hk sT unitOf (windows σ sT regOf evOf n) opened
      (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) (j10rAtCoin (hr1 _) enc place tile))
      (play Unit (tileCount σ sT) (k sT) n).game ∧
    (∀ w, prob (fun x => (tileLayer (circuit (tileCount σ sT)) (hr1 _) σ k hk sT unitOf (windows σ sT regOf evOf n)
      opened (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile)
      (j10rAtCoin (hr1 _) enc place tile)).MovedIn x w) ≤ 0) ∧
    RegisteredBeforeDraw (windows σ sT regOf evOf n) ∧ TileDrawIsCoin σ k hk sT unitOf (windows σ sT regOf evOf n) ∧
    TileServiceSound σ sT unitOf (windows σ sT regOf evOf n) opened
      (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) 0 ∧
    Properties.Pouw.Period.SaltIsLiveCoin (hr1 _) (saltOf (windows σ sT regOf evOf n)) enc ∧
    SharedNoiseLucks Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n) ∧
    UnitFixed (play Unit (tileCount σ sT) (k sT) n)
      (tileLayer (circuit (tileCount σ sT)) (hr1 _) σ k hk sT unitOf (windows σ sT regOf evOf n) opened
        (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) (j10rAtCoin (hr1 _) enc place tile))
      (periodMode (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
        (tileRun σ k hk sT unitOf (windows σ sT regOf evOf n) opened
          (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) (j10rAtCoin (hr1 _) enc place tile)))
      (fun _ t => (tileWords Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t : ℝ))
      (fun _ => (tileCount σ sT : ℝ)) ∧
    (∀ x w, WindowAccepts ((windows σ sT regOf evOf n).reg x w) ((windows σ sT regOf evOf n).ev x w)) ∧
    (∀ x, (windows σ sT regOf evOf n).prefixLen x = n) ∧
    ∀ x, creditLive Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
      (windows σ sT regOf evOf n) x = tileCount σ sT * n := by
  have hN : 0 < tileCount σ sT := Nat.lt_of_lt_of_le hKd (hk sT)
  refine ⟨le_rfl, rfl, scope _ hδs, fun t => ?_, fun j => ?_, fun t _ => ?_, rfl,
    period_every (Ω := Ω1 σ sT) (N := tileCount σ sT)
      (P := fun live _ => DeadConeDead Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) live) (fun _ _ ht _ => by simp at ht) n,
    fun x j _ => acceptsUpTo_period _ _ _ _ n j x, fun w => ?_, registeredBeforeDraw σ sT regOf evOf n,
    fun _ _ _ => hdraw _ _, fun w => ?_, fun _ _ => hsalt _,
    ⟨every_perLiveTile (P := fun k G _ => OperandHMu (Honest.scheme.mu k) G)
      (fun k t => Honest.operandHMu_of_zero _ _ _ _ _ (paid_share .folded (by decide) t)) n,
    every_perLiveTile (P := fun k G _ => ReceiptLuckAt (Honest.scheme.rB k) G)
      (fun k t => Honest.receiptLuck_of_zero _ _ _ _ _ (paid_share .folded (by decide) t)) n,
    every_perLiveTile (P := fun k _ G0 => GuessLuckAt (Honest.scheme.guess k) G0)
      (fun k t => Honest.guessLuck_of_zero _ _ _ _ _ (paid_share .whole (by decide) t)) n⟩,
    fun _ x _ => ⟨sum_div_le (N := tileCount σ sT) hn _ fun w j => ?_,
      sum_div_le (N := tileCount σ sT) hn _ fun w j => ?_⟩,
    fun _ _ => hacc _ _, prefixLen_all σ sT regOf evOf hacc n,
    creditLive_all σ sT regOf evOf hacc n⟩
  · rw [shareCone_eq, one_mul, mul_one_div_cancel (pos_of_fin t).ne']
  · unfold stageShare
    calc ∑ v, (if tileStage (circuit (tileCount σ sT)) (tileCount σ sT) v = j then
          shareCone Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) v else 0)
        ≤ ∑ v, shareCone Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) v :=
          Finset.sum_le_sum fun v _ => by
          split_ifs
          exacts [le_rfl, by rw [shareCone_eq]; exact div_nonneg zero_le_one (pos_of_fin v).le]
      _ = 1 := sum_shareCone hN
  · have hk' : tileKind (circuit (tileCount σ sT)) (tileCount σ sT) t = some (kindAt (circuit (tileCount σ sT)) t.val) := by
      show (unitWords (circuit (tileCount σ sT)) t.val).head?.map (kindAt (circuit (tileCount σ sT))) = _
      rw [unitWords_eq, if_pos t.isLt]
      rfl
    constructor
    · unfold tileLuck
      rw [hk']
      norm_num [Honest.scheme, Scheme.foldLuck]
    · unfold tileLuck0
      rw [hk']
      norm_num [Honest.scheme]
  · refine le_of_eq (Honest.prob_none ?_)
    rintro x ⟨i, -, h⟩
    exact h (funext fun _ => Subsingleton.elim _ _)
  · exact le_of_eq (Honest.prob_none fun _ ⟨_, _, _, _, _, h⟩ => h trivial)
  · exact ite_words_le _ j
  · exact ite_zero_le _ j

/-- **`H1J10RCheckedCreditPeriod`'s conclusion at the honest J10-R period**: `j10r_meets` gives the instance every
premise, so except with probability `δ_s` its fixed words are at least `(1 − ε(n))₊` of its `N_p·n` live words. -/
theorem j10r_credit (hKd : 1 ≤ k sT) (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β))
    (hsalt : ∀ c, (PouwCheck.hex? (regOf c).salt).getD ByteArray.empty = enc c)
    (hdraw : ∀ c β, ∃ d : Array ℕ, creditDrawn (regOf c) (evOf c β) = .ok d ∧
      ∀ u : ℕ, u ∈ d.toList ↔ ∃ i : Fin (k sT), unitOf (tileAt σ sT (drawOf (hk sT) β i)) = u)
    (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) :
    prob (fun x => (play Unit (tileCount σ sT) (k sT) n).accepts x ∧
      n * ((play Unit (tileCount σ sT) (k sT) n).run x).fixed <
        creditPrefix Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)
          ((windows σ sT regOf evOf n).prefixLen x)
          (creditLive Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
            (windows σ sT regOf evOf n) x) 1 1 (j10rRate (k sT) 1) δs) ≤ δs + n * (0 + 0) := by
  obtain ⟨hN, hM, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hLucks, hUF, -, -, -⟩ :=
    j10r_meets σ k hk sT unitOf regOf evOf enc hKd hacc hsalt hdraw n hn hδs
  exact h1J10RCheckedCreditPeriod σ k hk sT unitOf Honest.scheme (circuit (tileCount σ sT)) hn hN (hr1 _) hM hKd
    (windows σ sT regOf evOf n) opened enc place tile (period (Ω1 σ sT) (tileCount σ sT) n)
    (play Unit (tileCount σ sT) (k sT) n) (fun _ => (tileCount σ sT : ℝ)) 1 δs 0 0 hS one_pos le_rfl hρN
    hwmax hlk hG hDD hPre hMoved hReg hDraw hServ hSalt hLucks hUF

end Meets

end Properties.Pouw.PeriodJ10R.SecurityProofs.HonestJ10R
