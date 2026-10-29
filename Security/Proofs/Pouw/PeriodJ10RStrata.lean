import Properties.Pouw.PeriodJ10RStrata
import Proofs.Pouw.PeriodJ10R

/-!
# Proof of `H1J10RCheckedCreditPeriodStrata`

At each stratum `s ∈ S`, what `Proofs.Pouw.PeriodJ10R` applies to its one stratum:
`Accounting.SecurityProofs.ComposePeriod.CircuitCreditPeriod` at `tileRun`, on stratum `s`'s view of the period's coins
(`pathsAt`). Its joint soundness comes from `StrataDrawIsCoin`'s half that puts stratum `s`'s drawn proof blocks among the
proof blocks `Credit.window`'s draw demands, with its `TileServiceSound` (`jointSound_strata`); its pins from the period's
salt being its live coin, read on the view (`saltAt`). Its accepted prefix and live words are the period's
(`prefixLen_eq`, `creditedLive_eq`, `stratumLive_symm`), and the view is a relabelling of the period's coins, so the
bound holds on the period's coins (`stratum_bound`). An accepted period whose fixed words, summed over `S`, fall short of
the summed credit has a stratum whose own fall short, so the union bound gives the statement.
-/

namespace Properties.Pouw.PeriodJ10RStrata.SecurityProofs

open Properties.Pouw.PeriodJ10R Properties.Pouw.PeriodJ10R.SecurityProofs Properties.Pouw.PeriodJ10RStrata
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-! ## Stratum `s`'s view, window by window -/

theorem coinAt_pathsCongr {α β : Type} (e : α ≃ β) :
    ∀ {n : ℕ} (x : Paths α n) (w : Fin n), coinAt (pathsCongr e n x) w = e (coinAt x w)
  | 0, _, w => w.elim0
  | _ + 1, _, ⟨0, _⟩ => rfl
  | _ + 1, x, ⟨w + 1, h⟩ => coinAt_pathsCongr e x.2 ⟨w, Nat.lt_of_succ_lt_succ h⟩

/-- **A window's coins on stratum `s`'s view are its coins' `atStratum`.** -/
theorem coinAt_view {C : Type} {r m nB mB : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB) {n : ℕ}
    (y : Paths (Coin C (Rest σ s) r σ s) n) (w : Fin n) :
    coinAt y w = atStratum h σ s (coinAt ((pathsAt C h σ s n).symm y) w) := by
  conv_lhs => rw [← (pathsAt C h σ s n).apply_symm_apply y]
  exact coinAt_pathsCongr (atStratum h σ s) _ w

/-- The view's rounds' coins are the window's, relabelled by `r = m`. -/
theorem coinAt_atStratum_fst {C : Type} {r m nB mB : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB)
    (z : StrataCoin C m σ) (hr : 0 < r) (hm : 0 < m) :
    coinAt (atStratum h σ s z).1 ⟨0, hr⟩ = coinAt z.1 ⟨0, hm⟩ := by
  subst h
  rfl

/-- The view's tile stratum's permutation is the block's at `s`. -/
theorem atStratum_perm {C : Type} {r m nB mB : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB)
    (z : StrataCoin C m σ) : (atStratum h σ s z).2.1 = z.2 s :=
  rfl

/-- **The period's salt is its live coin on stratum `s`'s view.** -/
theorem saltAt {C : Type} {r nB mB n : ℕ} (h : r = 1) (hr : 0 < r) (σ : Fin nB → Fin mB) (s : Fin mB)
    (ws : CreditWindows (StrataCoin C 1 σ) n) {enc : C → ByteArray}
    (hs : Properties.Pouw.Period.SaltIsLiveCoin Nat.one_pos (saltOf ws) enc) :
    Properties.Pouw.Period.SaltIsLiveCoin hr (saltOf (creditAt h σ s ws)) enc := by
  intro y w
  show saltOf ws ((pathsAt C h σ s n).symm y) w = _
  rw [hs, coinAt_view h σ s y w, coinAt_atStratum_fst h σ s _ hr Nat.one_pos]

/-- **Joint soundness at stratum `s`'s run**: a window whose drawn tile of stratum `s` has an opening failing its
relation is one `Credit.window` credits with that tile's window proof block among the proof blocks its draw demanded, since
`StrataDrawIsCoin` puts every stratum's drawn proof blocks there, which stratum `s`'s `TileServiceSound` bounds. -/
theorem jointSound_strata {ι C V : Type} [Fintype C] (Cd : Declared ι) (hr : 0 < Cd.rounds) {m nB mB n : ℕ}
    (hM : Cd.rounds = m) (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Finset (Fin mB)) (s : Fin mB) (hsS : s ∈ S) (unitOf : Fin nB → ℕ) (ws : CreditWindows (StrataCoin C m σ) n)
    (opened : Paths (StrataCoin C m σ) n → Fin n → ℕ → ℕ → V) (enc : C → ByteArray)
    (place : Paths (StrataCoin C m σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop) {δks : ℝ}
    (hD : StrataDrawIsCoin σ k hk S unitOf ws)
    (hS : TileServiceSound σ s unitOf (creditAt hM σ s ws) (pullAt hM σ s opened) (relAt hM σ s ws place tile) δks) :
    (layerAt Cd hr hM σ k hk s unitOf ws opened enc place tile).JointSound δks := by
  intro w
  refine (Pouw.RelativeSecurity.prob_mono fun y hy => ?_).trans (hS w)
  obtain ⟨i, hacc, hrel⟩ := hy
  obtain ⟨d, hd, hmem⟩ := hD ((pathsAt C hM σ s n).symm y) w hacc
  refine ⟨hacc, d, hd, drawOf (hk s) (coinAt y w).2 i,
    (hmem _).2 ⟨s, hsS, tileAt σ s (drawOf (hk s) (coinAt y w).2 i), ?_, rfl⟩, hrel⟩
  rw [coinAt_view hM σ s y w]
  unfold tileDraw
  exact Finset.mem_image.2 ⟨i, Finset.mem_univ _, rfl⟩

/-- **Stratum `s`'s live words on its view** are its `creditLive` at the period's `Credit.window` inputs there. -/
theorem stratumLive_symm {ι K C : Type} (Sc : Scheme K) (Cd : Declared ι) {m nB mB n : ℕ} (hM : Cd.rounds = m)
    (σ : Fin nB → Fin mB) (s : Fin mB) (Wp : Period C (Coin C (Rest σ s) Cd.rounds σ s) (Fin (tileCount σ s)) Cd.rounds n)
    (ws : CreditWindows (StrataCoin C m σ) n) (y : Paths (Coin C (Rest σ s) Cd.rounds σ s) n) :
    stratumLive Sc Cd hM σ s Wp ws ((pathsAt C hM σ s n).symm y) = creditLive Sc Cd Wp (creditAt hM σ s ws) y := by
  unfold stratumLive
  simp only [Equiv.apply_symm_apply]
  rfl

/-- **The period's accepted prefix on stratum `s`'s view** is the view's. -/
theorem prefixLen_symm {C : Type} {r m nB mB n : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB)
    (ws : CreditWindows (StrataCoin C m σ) n) (y : Paths (Coin C (Rest σ s) r σ s) n) :
    ws.prefixLen ((pathsAt C h σ s n).symm y) = (creditAt h σ s ws).prefixLen y :=
  rfl

/-! ## One stratum's bound, and the union -/

/-- **Stratum `s`'s bound on the period's coins**: `CircuitCreditPeriod` at stratum `s`'s run, as
`Proofs.Pouw.PeriodJ10R` applies it, carried from the view to the period's coins. -/
theorem stratum_bound {ι K T L U C V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C]
    [Nonempty V] {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Finset (Fin mB)) (s : Fin mB) (hsS : s ∈ S) (unitOf : Fin nB → ℕ) (Sc : Scheme K) (Cd : Declared ι) {n : ℕ}
    (hn : 0 < n) (hN : Cd.units ≤ tileCount σ s) (hr : 0 < Cd.rounds) (hM : Cd.rounds = 1) (hKd : 1 ≤ k s)
    (ws : CreditWindows (StrataCoin C 1 σ) n) (opened : Paths (StrataCoin C 1 σ) n → Fin n → ℕ → ℕ → V)
    (enc : C → ByteArray) (place : Paths (StrataCoin C 1 σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop)
    (Wp : Period C (Coin C (Rest σ s) Cd.rounds σ s) (Fin (tileCount σ s)) Cd.rounds n)
    (Pl : Play (Coin C (Rest σ s) Cd.rounds σ s) n K T L U)
    (folded : Paths (Coin C (Rest σ s) Cd.rounds σ s) n → ℝ) (ρ δs δbind δks : ℝ)
    (hDraw : StrataDrawIsCoin σ k hk S unitOf ws)
    (hSalt : Properties.Pouw.Period.SaltIsLiveCoin Nat.one_pos (saltOf ws) enc)
    (hS : Scope Sc Cd δs) (hρ : 0 < ρ) (hρ1 : ρ ≤ 1)
    (hρN : ∀ t : Fin (tileCount σ s), ρ * (tileCount σ s) * shareCone Sc Cd (tileCount σ s) t ≤ 1)
    (hwmax : ∀ j, stageShare (tileStage Cd (tileCount σ s)) (shareCone Sc Cd (tileCount σ s)) j ≤ 1)
    (hlk : ∀ t : Fin (tileCount σ s), 0 < shareCone Sc Cd (tileCount σ s) t →
      tileLuck Sc Cd (tileCount σ s) t ≤ 7 / 8 ∧ tileLuck0 Sc Cd (tileCount σ s) t ≤ 1 / 2)
    (hG : Pl.game = periodGame (tileStage Cd (tileCount σ s)) (liveCone Sc Cd (tileCount σ s)) 1 1 (k s) Wp)
    (hDD : Wp.EveryWindow (fun live _ => DeadConeDead Sc Cd (tileCount σ s) live))
    (hPre : PrefixOf (layerAt Cd hr hM σ k hk s unitOf ws opened enc place tile) Pl.game)
    (hMoved : ∀ w, prob (fun y => (layerAt Cd hr hM σ k hk s unitOf ws opened enc place tile).MovedIn y w) ≤ δbind)
    (hServ : TileServiceSound σ s unitOf (creditAt hM σ s ws) (pullAt hM σ s opened) (relAt hM σ s ws place tile)
      δks)
    (hLucks : SharedNoiseLucks Sc Cd Wp)
    (hUF : UnitFixed Pl (layerAt Cd hr hM σ k hk s unitOf ws opened enc place tile)
      (periodMode Cd Wp (runAt Cd hr hM σ k hk s unitOf ws opened enc place tile))
      (fun _ t => (tileWords Sc Cd (tileCount σ s) t : ℝ)) folded) :
    prob (fun x : Paths (StrataCoin C 1 σ) n => Pl.accepts (pathsAt C hM σ s n x) ∧
      n * (Pl.run (pathsAt C hM σ s n x)).fixed <
        creditPrefix Sc Cd (tileCount σ s) (ws.prefixLen x) (stratumLive Sc Cd hM σ s Wp ws x) 1 1
          (j10rRate (k s) ρ) δs) ≤ δs + n * (δbind + δks) := by
  obtain ⟨hMu, hRB, hGs⟩ := hLucks
  have hK0 : 0 < k s := hKd
  have h := Accounting.SecurityProofs.ComposePeriod.CircuitCreditPeriod Sc Cd hn hN hr (hk s) Wp Pl
    (tileRun σ k hk s unitOf (creditAt hM σ s ws) (pullAt hM σ s opened) (relAt hM σ s ws place tile)
      (rightAt hM hr σ s enc place tile))
    folded ρ (7 / 8) (1 / 2) 1 1 1 (j10rRate (k s) ρ) δs δbind δks hS hρ hρN hwmax hlk one_pos one_pos
    (j10rRate_pos hKd hρ) (rate_j10r hKd hρ hρ1) hG hDD hPre hMoved
    (jointSound_strata Cd hr hM σ k hk S s hsS unitOf ws opened enc place tile hDraw hServ)
    (pins_j10r (place := pullAt hM σ s place) (tile := tile) hr (saltAt hM hr σ s ws hSalt)) hMu hRB hGs hUF
  simp only [prefixLen_eq σ k hk s unitOf hK0 (creditAt hM σ s ws) (pullAt hM σ s opened),
    creditedLive_eq Sc Cd σ k hk s unitOf hK0 (creditAt hM σ s ws) (pullAt hM σ s opened) _ _ Wp] at h
  refine le_of_eq_of_le (prob_equiv (pathsAt C hM σ s n) _) (le_of_eq_of_le ?_ h)
  congr 1
  funext y
  simp only [Equiv.apply_symm_apply]
  exact congrArg (fun L => Pl.accepts y ∧ n * (Pl.run y).fixed <
      creditPrefix Sc Cd (tileCount σ s) ((creditAt hM σ s ws).prefixLen y) L 1 1 (j10rRate (k s) ρ) δs)
    (stratumLive_symm Sc Cd hM σ s Wp ws y)

/-- **`H1J10RCheckedCreditPeriodStrata`**: each stratum's bound (`stratum_bound`), then the union bound over `S`: on an
accepted period whose summed fixed words fall short of the summed credit, some stratum's fall short of its own. -/
theorem h1J10RCheckedCreditPeriodStrata : H1J10RCheckedCreditPeriodStrata := by
  intro ι K T L U C V _ _ _ _ _ _ nB mB σ k hk S unitOf Sc Cd n hn hN hr hM hKd ws opened enc place tile Wp Pl folded
    ρ δs δbind δks _hReg hDraw hSalt hS hρ hρN hwmax hlk hG hDD hPre hMoved hServ hLucks hUF
  calc _ ≤ prob (fun x : Paths (StrataCoin C 1 σ) n => ∃ s ∈ S, (Pl s).accepts (pathsAt C (hM s) σ s n x) ∧
          n * ((Pl s).run (pathsAt C (hM s) σ s n x)).fixed <
            creditPrefix Sc (Cd s) (tileCount σ s) (ws.prefixLen x) (stratumLive Sc (Cd s) (hM s) σ s (Wp s) ws x)
              1 1 (j10rRate (k s) (ρ s)) (δs s)) := by
        refine Pouw.RelativeSecurity.prob_mono fun x hx => ?_
        obtain ⟨hacc, hlt⟩ := hx
        by_contra hne
        push Not at hne
        refine absurd hlt (not_lt.2 ?_)
        rw [Finset.mul_sum]
        exact Finset.sum_le_sum fun s hs => hne s hs (hacc s hs)
    _ ≤ ∑ s ∈ S, prob (fun x : Paths (StrataCoin C 1 σ) n => (Pl s).accepts (pathsAt C (hM s) σ s n x) ∧
          n * ((Pl s).run (pathsAt C (hM s) σ s n x)).fixed <
            creditPrefix Sc (Cd s) (tileCount σ s) (ws.prefixLen x) (stratumLive Sc (Cd s) (hM s) σ s (Wp s) ws x)
              1 1 (j10rRate (k s) (ρ s)) (δs s)) :=
        prob_exists_le S _
    _ ≤ _ := Finset.sum_le_sum fun s hs =>
        stratum_bound σ k hk S s hs unitOf Sc (Cd s) hn (hN s hs) (hr s) (hM s) (hKd s hs) ws opened enc (place s)
          (tile s) (Wp s) (Pl s) (folded s) (ρ s) (δs s) (δbind s) (δks s) hDraw hSalt (hS s hs) (hρ s hs).1
          (hρ s hs).2 (hρN s hs) (hwmax s hs) (hlk s hs) (hG s hs) (hDD s hs) (hPre s hs) (hMoved s hs)
          (hServ s hs) (hLucks s hs) (hUF s hs)

end Properties.Pouw.PeriodJ10RStrata.SecurityProofs
