import Proofs.Pouw.PeriodJ10RStrata
import Proofs.Pouw.PeriodJ10R.Honest

/-!
# An honest J10-R period over several tile strata meets `H1J10RCheckedCreditPeriodStrata`'s premises, and the
conclusion fails at a verifier that accepts everything

The twins of `Properties.Pouw.PeriodJ10RStrata.H1J10RCheckedCreditPeriodStrata`, as `HonestJ10R.j10r_meets` is of the
one-stratum statement.

*Honest* (`strata_meets_period`, `strata_meets_stratum`, `strata_credit`). Over any sampling block `σ` with draw `k` and
any finite set `S` of its strata, `1 ≤ k s` at each `s ∈ S`, take a period of `n ≥ 1` windows whose registration is a
function of the window's round coin (`regOf`) and the rest of whose inputs a function of its round coin and the block's
draw (`evOf`). Given three facts of those inputs, the witness meets every premise of the statement:

* `Credit.window` credits every window (`hacc`);
* the registered salt, hex-decoded, is the round coin's bytes (`hsalt`);
* the proof blocks `Credit.descriptorDrawn` draws are the union over `S` of the strata's coordinates of the block's draw
  (`hdraw`), `StrataDrawIsCoin`'s body on every window.

At each stratum `s` the instance is `HonestJ10R`'s at `N_s = |stratum σ s|` with the rest of the block's draw as `Ξ`:
`N_s` one-word proof blocks, one live round, `ρ = 1`, the twin's lucks, `δ_bind = δ_ks = 0`, `N_s` words fixed and folded a
window. The accepted prefix is all `n` windows and stratum `s`'s live words are `N_s·n` (`stratumLive_all`), so
`strata_credit` is the statement's bound at `∑_{s ∈ S} N_s·n` live words.

*Nontrivial* (`strata_nontrivial`). The statement reads its verdict in its conclusion only through the accepted prefix
and the strata's live words; `StrataBoundBy` is the conclusion at any verdict `acc`, and at `Credit.window`'s it is the
statement's own (`strataBound_window`). At the verdict that credits every window, over the honest instance's circuits,
scheme, periods and rates, the provers that fix no word (`idlePlay`, accepted on every path) break it: once
`Λ_s·ln(1/δ_s(s)) < n` at each `s ∈ S` and `S` is non-empty, the summed credit is positive on every path, so the event's
probability is `1`, above any `∑_{s ∈ S} (δ_s(s) + n·(δ_bind(s) + δ_ks(s))) < 1`.
-/

namespace Properties.Pouw.PeriodJ10RStrata.SecurityProofs.HonestStrata

open Properties.Pouw.PeriodJ10R Properties.Pouw.PeriodJ10R.SecurityProofs Properties.Pouw.PeriodJ10RStrata
  Properties.Pouw.PeriodJ10RStrata.SecurityProofs
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposePeriod

/-! ## The honest instance -/

section Instance

variable {nB mB : ℕ} (σ : Fin nB → Fin mB)

/-- Stratum `s`'s view of a window's coins at the honest instance: `HonestJ10R`'s window coins with the rest of the
block's draw. -/
abbrev View (s : Fin mB) : Type := Coin Bool (Rest σ s) (HonestJ10R.circuit (tileCount σ s)).rounds σ s

/-- The honest circuit has one round a window, the statement's `hM`. -/
theorem rounds1 (N : ℕ) : (HonestJ10R.circuit N).rounds = 1 :=
  rfl

/-- The period's windows as `Credit.window` reads them: window `w` registers `regOf` of its round coin, and the rest of
its inputs are `evOf` of its round coin and the block's draw. -/
def windows (regOf : Bool → Registered) (evOf : Bool → BlockDraw σ → Answered) (n : ℕ) :
    CreditWindows (StrataCoin Bool 1 σ) n where
  reg x w := regOf (coinAt (coinAt x w).1 ⟨0, Nat.one_pos⟩)
  ev x w := evOf (coinAt (coinAt x w).1 ⟨0, Nat.one_pos⟩) (coinAt x w).2

end Instance

section Meets

variable {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
  (S : Finset (Fin mB)) (unitOf : Fin nB → ℕ) (regOf : Bool → Registered) (evOf : Bool → BlockDraw σ → Answered)
  (enc : Bool → ByteArray)

/-- **The period-wide premises**, and the accepted prefix is every window: the registration reads only the round coin,
which the splice at window `w` after its one round keeps; the salt is the round coin's bytes; the draw is the strata's. -/
theorem strata_meets_period (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β))
    (hsalt : ∀ c, (PouwCheck.hex? (regOf c).salt).getD ByteArray.empty = enc c)
    (hdraw : ∀ c β, ∃ d : Array ℕ, creditDrawn (regOf c) (evOf c β) = .ok d ∧
      ∀ u : ℕ, u ∈ d.toList ↔ ∃ s ∈ S, ∃ b ∈ tileDraw σ k hk s (β s), unitOf b = u) (n : ℕ) :
    RegisteredBeforeDraw (windows σ regOf evOf n) ∧ StrataDrawIsCoin σ k hk S unitOf (windows σ regOf evOf n) ∧
    Properties.Pouw.Period.SaltIsLiveCoin Nat.one_pos (saltOf (windows σ regOf evOf n)) enc ∧
    ∀ x, (windows σ regOf evOf n).prefixLen x = n := by
  classical
  refine ⟨fun x z w => ?_, fun _ _ _ => hdraw _ _, fun _ _ => hsalt _, fun x => ?_⟩
  · show regOf (coinAt (coinAt (spliceAt w.val 1 x z) w).1 ⟨0, Nat.one_pos⟩) =
      regOf (coinAt (coinAt x w).1 ⟨0, Nat.one_pos⟩)
    rw [Accounting.SecurityProofs.TS.Keyed.coinAt_spliceAt]
    rfl
  · unfold CreditWindows.prefixLen
    have h : ∀ w ∈ (Finset.univ : Finset (Fin n)), (windows σ regOf evOf n).AcceptedUpTo x w :=
      fun _ _ _ _ => hacc _ _
    rw [Finset.filter_true_of_mem h, Finset.card_univ, Fintype.card_fin]

/-- **Stratum `s`'s live words are `N_s·n`**: every window credited, every tile live, one word each. -/
theorem stratumLive_all (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β)) (n : ℕ) (s : Fin mB)
    (x : Paths (StrataCoin Bool 1 σ) n) :
    stratumLive Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (rounds1 _) σ s
      (HonestJ10R.period (View σ s) (tileCount σ s) n) (windows σ regOf evOf n) x = tileCount σ s * n := by
  unfold stratumLive
  calc _ = ∑ _w : Fin n, (tileCount σ s : ℝ) := Finset.sum_congr rfl fun w _ => by
          have hw : (windows σ regOf evOf n).AcceptedUpTo x w := fun _ _ => hacc _ _
          rw [if_pos hw, HonestJ10R.liveAt_period n _ w]
          simp only [HonestJ10R.liveWords_true]
          exact HonestJ10R.sum_tileWords _
    _ = tileCount σ s * n := by simp [mul_comm]

/-- **An honest J10-R period meets every per-stratum premise at each stratum `s`**, in the statement's order, at
`HonestJ10R`'s instance at `N_s = |stratum σ s|` on stratum `s`'s view (`layerAt`, `runAt`, `creditAt`): `Scope` at
any `δ_s > 0`, the live share at `ρ = 1`, the stage share, H1's lucks, the play's game, `DeadConeDead`, `PrefixOf`,
binding at `δ_bind = 0`, `TileServiceSound` at `δ_ks = 0`, `SharedNoiseLucks` and `UnitFixed` at `N_s` fixed and folded
words a window. -/
theorem strata_meets_stratum (s : Fin mB) (hKd : 1 ≤ k s) (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) :
    Scope Honest.scheme (HonestJ10R.circuit (tileCount σ s)) δs ∧
    (∀ t : Fin (tileCount σ s),
      (1 : ℝ) * (tileCount σ s) * shareCone Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) t ≤ 1) ∧
    (∀ j, stageShare (tileStage (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s))
      (shareCone Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s)) j ≤ 1) ∧
    (∀ t : Fin (tileCount σ s), 0 < shareCone Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) t →
      tileLuck Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) t ≤ 7 / 8 ∧
        tileLuck0 Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) t ≤ 1 / 2) ∧
    (HonestJ10R.play (Rest σ s) (tileCount σ s) (k s) n).game =
      periodGame (tileStage (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s))
        (liveCone Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s)) 1 1 (k s)
        (HonestJ10R.period (View σ s) (tileCount σ s) n) ∧
    (HonestJ10R.period (View σ s) (tileCount σ s) n).EveryWindow
      (fun live _ => DeadConeDead Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) live) ∧
    PrefixOf (layerAt (HonestJ10R.circuit (tileCount σ s)) (HonestJ10R.hr1 _) (rounds1 _) σ k hk s unitOf
      (windows σ regOf evOf n) HonestJ10R.opened enc HonestJ10R.place HonestJ10R.tile)
      (HonestJ10R.play (Rest σ s) (tileCount σ s) (k s) n).game ∧
    (∀ w, prob (fun y => (layerAt (HonestJ10R.circuit (tileCount σ s)) (HonestJ10R.hr1 _) (rounds1 _) σ k hk s unitOf
      (windows σ regOf evOf n) HonestJ10R.opened enc HonestJ10R.place HonestJ10R.tile).MovedIn y w) ≤ 0) ∧
    TileServiceSound σ s unitOf (creditAt (rounds1 (tileCount σ s)) σ s (windows σ regOf evOf n)) (pullAt (rounds1 (tileCount σ s)) σ s HonestJ10R.opened)
      (relAt (rounds1 (tileCount σ s)) σ s (windows σ regOf evOf n) HonestJ10R.place HonestJ10R.tile) 0 ∧
    SharedNoiseLucks Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (HonestJ10R.period (View σ s) (tileCount σ s) n) ∧
    UnitFixed (HonestJ10R.play (Rest σ s) (tileCount σ s) (k s) n)
      (layerAt (HonestJ10R.circuit (tileCount σ s)) (HonestJ10R.hr1 _) (rounds1 _) σ k hk s unitOf (windows σ regOf evOf n)
        HonestJ10R.opened enc HonestJ10R.place HonestJ10R.tile)
      (periodMode (HonestJ10R.circuit (tileCount σ s)) (HonestJ10R.period (View σ s) (tileCount σ s) n)
        (runAt (HonestJ10R.circuit (tileCount σ s)) (HonestJ10R.hr1 _) (rounds1 _) σ k hk s unitOf (windows σ regOf evOf n)
          HonestJ10R.opened enc HonestJ10R.place HonestJ10R.tile))
      (fun _ t => (tileWords Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) t : ℝ))
      (fun _ => (tileCount σ s : ℝ)) := by
  have hN : 0 < tileCount σ s := Nat.lt_of_lt_of_le hKd (hk s)
  refine ⟨HonestJ10R.scope _ hδs, fun t => ?_, fun j => ?_, fun t _ => ?_, rfl,
    HonestJ10R.period_every (Ω := View σ s) (N := tileCount σ s)
      (P := fun live _ => DeadConeDead Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) live)
      (fun _ _ ht _ => by simp at ht) n,
    fun x j _ => HonestJ10R.acceptsUpTo_period _ _ _ _ n j x, fun w => ?_, fun w => ?_,
    ⟨HonestJ10R.every_perLiveTile (P := fun k G _ => OperandHMu (Honest.scheme.mu k) G)
      (fun k t => Honest.operandHMu_of_zero _ _ _ _ _ (HonestJ10R.paid_share .folded (by decide) t)) n,
    HonestJ10R.every_perLiveTile (P := fun k G _ => ReceiptLuckAt (Honest.scheme.rB k) G)
      (fun k t => Honest.receiptLuck_of_zero _ _ _ _ _ (HonestJ10R.paid_share .folded (by decide) t)) n,
    HonestJ10R.every_perLiveTile (P := fun k _ G0 => GuessLuckAt (Honest.scheme.guess k) G0)
      (fun k t => Honest.guessLuck_of_zero _ _ _ _ _ (HonestJ10R.paid_share .whole (by decide) t)) n⟩,
    fun _ x _ => ⟨HonestJ10R.sum_div_le (N := tileCount σ s) hn _ fun w j => ?_,
      HonestJ10R.sum_div_le (N := tileCount σ s) hn _ fun w j => ?_⟩⟩
  · rw [HonestJ10R.shareCone_eq, one_mul, mul_one_div_cancel (HonestJ10R.pos_of_fin t).ne']
  · unfold stageShare
    calc ∑ v, (if tileStage (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) v = j then
          shareCone Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) v else 0)
        ≤ ∑ v, shareCone Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) v :=
          Finset.sum_le_sum fun v _ => by
          split_ifs
          exacts [le_rfl, by rw [HonestJ10R.shareCone_eq]; exact div_nonneg zero_le_one (HonestJ10R.pos_of_fin v).le]
      _ = 1 := HonestJ10R.sum_shareCone hN
  · have hk' : tileKind (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) t =
        some (kindAt (HonestJ10R.circuit (tileCount σ s)) t.val) := by
      show (unitWords (HonestJ10R.circuit (tileCount σ s)) t.val).head?.map
        (kindAt (HonestJ10R.circuit (tileCount σ s))) = _
      rw [HonestJ10R.unitWords_eq, if_pos t.isLt]
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
  · exact HonestJ10R.ite_words_le _ j
  · exact HonestJ10R.ite_zero_le _ j

/-- **`H1J10RCheckedCreditPeriodStrata`'s conclusion at the honest J10-R period over `S`**: `strata_meets_period` and
`strata_meets_stratum` give the instance every premise, so except with probability `∑_{s ∈ S} δ_s(s)` its fixed words,
summed over `S`, are at least the sum over `S` of `(1 − ε_s(n))₊` of stratum `s`'s `N_s·n` live words. -/
theorem strata_credit (hKd : ∀ s ∈ S, 1 ≤ k s) (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β))
    (hsalt : ∀ c, (PouwCheck.hex? (regOf c).salt).getD ByteArray.empty = enc c)
    (hdraw : ∀ c β, ∃ d : Array ℕ, creditDrawn (regOf c) (evOf c β) = .ok d ∧
      ∀ u : ℕ, u ∈ d.toList ↔ ∃ s ∈ S, ∃ b ∈ tileDraw σ k hk s (β s), unitOf b = u)
    (n : ℕ) (hn : 0 < n) {δs : Fin mB → ℝ} (hδs : ∀ s ∈ S, 0 < δs s) :
    prob (fun x : Paths (StrataCoin Bool 1 σ) n =>
      (∀ s ∈ S, (HonestJ10R.play (Rest σ s) (tileCount σ s) (k s) n).accepts (pathsAt Bool (rounds1 (tileCount σ s)) σ s n x)) ∧
      n * ∑ s ∈ S, ((HonestJ10R.play (Rest σ s) (tileCount σ s) (k s) n).run (pathsAt Bool (rounds1 (tileCount σ s)) σ s n x)).fixed <
        ∑ s ∈ S, creditPrefix Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s)
          ((windows σ regOf evOf n).prefixLen x)
          (stratumLive Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (rounds1 _) σ s
            (HonestJ10R.period (View σ s) (tileCount σ s) n) (windows σ regOf evOf n) x) 1 1 (j10rRate (k s) 1)
          (δs s)) ≤
      ∑ s ∈ S, (δs s + n * (0 + 0)) := by
  obtain ⟨hReg, hDraw, hSalt, -⟩ := strata_meets_period σ k hk S unitOf regOf evOf enc hacc hsalt hdraw n
  have hm := fun s (hs : s ∈ S) => strata_meets_stratum σ k hk unitOf regOf evOf enc s (hKd s hs) n hn (hδs s hs)
  exact h1J10RCheckedCreditPeriodStrata σ k hk S unitOf Honest.scheme (fun s => HonestJ10R.circuit (tileCount σ s)) hn
    (fun _ _ => le_rfl) (fun _ => HonestJ10R.hr1 _) (fun s => rounds1 (tileCount σ s)) hKd (windows σ regOf evOf n) HonestJ10R.opened enc
    (fun _ => HonestJ10R.place) (fun _ => HonestJ10R.tile) (fun s => HonestJ10R.period (View σ s) (tileCount σ s) n)
    (fun s => HonestJ10R.play (Rest σ s) (tileCount σ s) (k s) n) (fun s _ => (tileCount σ s : ℝ)) (fun _ => 1) δs
    (fun _ => 0) (fun _ => 0) hReg hDraw hSalt (fun s hs => (hm s hs).1) (fun _ _ => ⟨one_pos, le_rfl⟩)
    (fun s hs => (hm s hs).2.1) (fun s hs => (hm s hs).2.2.1) (fun s hs => (hm s hs).2.2.2.1)
    (fun s hs => (hm s hs).2.2.2.2.1) (fun s hs => (hm s hs).2.2.2.2.2.1) (fun s hs => (hm s hs).2.2.2.2.2.2.1)
    (fun s hs => (hm s hs).2.2.2.2.2.2.2.1) (fun s hs => (hm s hs).2.2.2.2.2.2.2.2.1)
    (fun s hs => (hm s hs).2.2.2.2.2.2.2.2.2.1) (fun s hs => (hm s hs).2.2.2.2.2.2.2.2.2.2)

end Meets

/-! ## The verifier that accepts everything -/

section By

variable {Ω : Type} {n : ℕ}

/-- **Window `w` and every earlier one are credited** by the verdict `acc` on path `x`. -/
def UpToBy (acc : Paths Ω n → Fin n → Prop) (x : Paths Ω n) (w : Fin n) : Prop :=
  ∀ w' : Fin n, w' ≤ w → acc x w'

open Classical in
/-- **The accepted prefix at verdict `acc`**: `CreditWindows.prefixLen` with `acc` for `Credit.window`'s verdict. -/
noncomputable def prefixLenBy (acc : Paths Ω n → Fin n → Prop) (x : Paths Ω n) : ℕ :=
  (Finset.univ.filter fun w : Fin n => UpToBy acc x w).card

end By

open Classical in
/-- **Stratum `s`'s live words at verdict `acc`**: `stratumLive` with `acc` for `Credit.window`'s verdict. -/
noncomputable def stratumLiveBy {ι K C : Type} (Sc : Scheme K) (Cd : Declared ι) {m nB mB n : ℕ} (hM : Cd.rounds = m)
    (σ : Fin nB → Fin mB) (s : Fin mB)
    (Wp : WindowPlay C (Coin C (Rest σ s) Cd.rounds σ s) (Fin (tileCount σ s)) Cd.rounds n)
    (acc : Paths (StrataCoin C m σ) n → Fin n → Prop) (x : Paths (StrataCoin C m σ) n) : ℝ :=
  ∑ w : Fin n, if UpToBy acc x w then
    ∑ t : Fin (tileCount σ s), (liveWords Sc Cd (tileCount σ s) (Wp.liveAt (pathsAt C hM σ s n x) w) t : ℝ) else 0

/-- **`H1J10RCheckedCreditPeriodStrata`'s conclusion at verdict `acc`**: except with probability `δ`, an accepted
period's fixed words, summed over `S`, are at least the sum over `S` of each stratum's credit of `acc`'s accepted
prefix. -/
def StrataBoundBy {ι K T L U C : Type} [Fintype C] {nB mB n : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (S : Finset (Fin mB)) (Sc : Scheme K) (Cd : Fin mB → Declared ι) (hM : ∀ s, (Cd s).rounds = 1)
    (Wp : (s : Fin mB) → WindowPlay C (Coin C (Rest σ s) (Cd s).rounds σ s) (Fin (tileCount σ s)) (Cd s).rounds n)
    (Pl : (s : Fin mB) → Play (Coin C (Rest σ s) (Cd s).rounds σ s) n K T L U)
    (acc : Paths (StrataCoin C 1 σ) n → Fin n → Prop) (ρ δs : Fin mB → ℝ) (δ : ℝ) : Prop :=
  prob (fun x : Paths (StrataCoin C 1 σ) n => (∀ s ∈ S, (Pl s).accepts (pathsAt C (hM s) σ s n x)) ∧
    n * ∑ s ∈ S, ((Pl s).run (pathsAt C (hM s) σ s n x)).fixed <
      ∑ s ∈ S, creditPrefix Sc (Cd s) (tileCount σ s) (prefixLenBy acc x)
        (stratumLiveBy Sc (Cd s) (hM s) σ s (Wp s) acc x) 1 1 (j10rRate (k s) (ρ s)) (δs s)) ≤ δ

/-- **At `Credit.window`'s verdict it is the statement's conclusion**, so `strata_nontrivial`, `StrataBoundBy` at the
verdict that credits every window, is the conclusion with that verdict in place of `Credit.window`'s. -/
theorem strataBound_window {ι K T L U C : Type} [Fintype C] {nB mB n : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (S : Finset (Fin mB)) (Sc : Scheme K) (Cd : Fin mB → Declared ι) (hM : ∀ s, (Cd s).rounds = 1)
    (ws : CreditWindows (StrataCoin C 1 σ) n)
    (Wp : (s : Fin mB) → WindowPlay C (Coin C (Rest σ s) (Cd s).rounds σ s) (Fin (tileCount σ s)) (Cd s).rounds n)
    (Pl : (s : Fin mB) → Play (Coin C (Rest σ s) (Cd s).rounds σ s) n K T L U) (ρ δs : Fin mB → ℝ) (δ : ℝ) :
    prob (fun x : Paths (StrataCoin C 1 σ) n => (∀ s ∈ S, (Pl s).accepts (pathsAt C (hM s) σ s n x)) ∧
      n * ∑ s ∈ S, ((Pl s).run (pathsAt C (hM s) σ s n x)).fixed <
        ∑ s ∈ S, creditPrefix Sc (Cd s) (tileCount σ s) (ws.prefixLen x)
          (stratumLive Sc (Cd s) (hM s) σ s (Wp s) ws x) 1 1 (j10rRate (k s) (ρ s)) (δs s)) ≤ δ ↔
      StrataBoundBy σ k S Sc Cd hM Wp Pl (fun x w => WindowAccepts (ws.reg x w) (ws.ev x w)) ρ δs δ :=
  Iff.rfl

/-- **The prover that does nothing at stratum `s`**: the honest play's game, accepted on every path, fixing no word. -/
noncomputable def idlePlay (B : Type) (N Kd n : ℕ) : Play (HonestJ10R.WCoin N B) n Unit Unit Unit Unit :=
  { HonestJ10R.play B N Kd n with run := fun _ => { HonestJ10R.anatomy N with fixed := 0 } }

theorem prefixLenBy_all {Ω : Type} {n : ℕ} (x : Paths Ω n) : prefixLenBy (fun _ _ => True) x = n := by
  classical
  unfold prefixLenBy
  rw [Finset.filter_true_of_mem fun _ _ _ _ => trivial, Finset.card_univ, Fintype.card_fin]

theorem stratumLiveBy_all {nB mB n : ℕ} (σ : Fin nB → Fin mB) (s : Fin mB) (x : Paths (StrataCoin Bool 1 σ) n) :
    stratumLiveBy Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (rounds1 _) σ s
      (HonestJ10R.period (View σ s) (tileCount σ s) n) (fun _ _ => True) x = (tileCount σ s : ℝ) * n := by
  unfold stratumLiveBy
  calc _ = ∑ _w : Fin n, (tileCount σ s : ℝ) := Finset.sum_congr rfl fun w _ => by
          rw [if_pos (show UpToBy (fun _ _ => True) x w from fun _ _ => trivial), HonestJ10R.liveAt_period n _ w]
          simp only [HonestJ10R.liveWords_true]
          exact HonestJ10R.sum_tileWords _
    _ = tileCount σ s * n := by simp [mul_comm]

theorem zeroCone_circuit (N : ℕ) : zeroCone Honest.scheme (HonestJ10R.circuit N) N = 0 := by
  unfold zeroCone
  exact Finset.sum_eq_zero fun t _ => if_neg (by rw [HonestJ10R.tileWords_eq]; exact one_ne_zero)

/-- **A stratum's whole-period credit is positive once `Λ·ln(1/δ_s) < n`**: `ε(n) = Λ·ln(1/δ_s)/n` at `N·n` live
words. -/
theorem creditPrefix_circuit_pos {N n : ℕ} (hN : 0 < N) (hn : 0 < n) {Λ δs : ℝ} (hΛ : Λ * Real.log (1 / δs) < n) :
    0 < creditPrefix Honest.scheme (HonestJ10R.circuit N) N n ((N : ℝ) * n) 1 1 Λ δs := by
  have hN' : (0 : ℝ) < N := Nat.cast_pos.mpr hN
  have hn' : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have heps : epsPrefix Honest.scheme (HonestJ10R.circuit N) N n ((N : ℝ) * n) 1 1 Λ δs =
      Λ * Real.log (1 / δs) / n := by
    unfold epsPrefix rOfPrefix
    rw [zeroCone_circuit, HonestJ10R.wordsOf_eq, min_self, mul_zero, mul_zero, add_zero, one_mul,
      div_eq_div_iff (mul_pos hN' hn').ne' hn'.ne']
    ring
  have h1 : 0 < 1 - Λ * Real.log (1 / δs) / n := by
    rw [sub_pos, div_lt_one hn']
    exact hΛ
  unfold creditPrefix
  rw [heps, max_eq_left h1.le]
  exact mul_pos h1 (mul_pos hN' hn')

theorem prob_all {Ω : Type} [Fintype Ω] (a : Ω) : prob (fun _ : Ω => True) = 1 := by
  classical
  have : Nonempty Ω := ⟨a⟩
  rw [prob_eq_card, Finset.filter_true_of_mem fun _ _ => trivial, Finset.card_univ]
  exact div_self (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)

/-- **`H1J10RCheckedCreditPeriodStrata`'s conclusion fails at the verifier that accepts everything**: at the honest
instance's circuits, scheme, periods and rates over a non-empty `S` (`N_s = |stratum σ s|`, `1 ≤ k s`,
`Λ_s = j10rRate (k s) 1`), with every window credited, the provers that fix no word (`idlePlay`) are accepted on every
path with less than the summed credit, so the bound fails at every `∑_{s ∈ S} (δ_s(s) + n·(δ_bind(s) + δ_ks(s))) < 1`
once `Λ_s·ln(1/δ_s(s)) < n` at each `s ∈ S`. Its statement is `StrataBoundBy` at that verdict and those plays,
unfolded, so it negates the conclusion's own `prob … ≤ …`. -/
theorem strata_nontrivial {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Finset (Fin mB)) (hS : S.Nonempty) (hKd : ∀ s ∈ S, 1 ≤ k s) (n : ℕ) (hn : 0 < n)
    {δs δbind δks : Fin mB → ℝ} (hΛ : ∀ s ∈ S, j10rRate (k s) 1 * Real.log (1 / δs s) < n)
    (hδ : ∑ s ∈ S, (δs s + n * (δbind s + δks s)) < 1) :
    ¬ prob (fun x : Paths (StrataCoin Bool 1 σ) n =>
        (∀ s ∈ S, (idlePlay (Rest σ s) (tileCount σ s) (k s) n).accepts (pathsAt Bool (rounds1 (tileCount σ s)) σ s n x)) ∧
        n * ∑ s ∈ S, ((idlePlay (Rest σ s) (tileCount σ s) (k s) n).run (pathsAt Bool (rounds1 (tileCount σ s)) σ s n x)).fixed <
          ∑ s ∈ S, creditPrefix Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s)
            (prefixLenBy (fun _ _ => True) x)
            (stratumLiveBy Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (rounds1 _) σ s
              (HonestJ10R.period (View σ s) (tileCount σ s) n) (fun _ _ => True) x) 1 1 (j10rRate (k s) 1) (δs s)) ≤
      ∑ s ∈ S, (δs s + n * (δbind s + δks s)) := by
  intro h
  have hpos : 0 < ∑ s ∈ S, creditPrefix Honest.scheme (HonestJ10R.circuit (tileCount σ s)) (tileCount σ s) n
      ((tileCount σ s : ℝ) * n) 1 1 (j10rRate (k s) 1) (δs s) :=
    Finset.sum_pos (fun s hs => creditPrefix_circuit_pos (Nat.lt_of_lt_of_le (hKd s hs) (hk s)) hn (hΛ s hs)) hS
  have hall := (Pouw.RelativeSecurity.prob_mono (p := fun _ : Paths (StrataCoin Bool 1 σ) n => True) fun x _ => by
    refine ⟨fun _ _ => trivial, ?_⟩
    simp only [prefixLenBy_all, stratumLiveBy_all]
    show (n : ℝ) * ∑ _s ∈ S, (0 : ℝ) < _
    rw [Finset.sum_const_zero, mul_zero]
    exact hpos).trans h
  rw [prob_all (Paths.const (Paths.const true 1, fun s => Equiv.refl _) n)] at hall
  linarith

end Properties.Pouw.PeriodJ10RStrata.SecurityProofs.HonestStrata
