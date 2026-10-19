import Definitions.Accounting.Compose

/-!
# PoUW's work over an accounting period of `n` windows (trusted)

The vocabulary of `Models.Accounting.Guarantees.ComposePeriod` (INT-PERIOD,
note:compute-accounting/20261019T0739Z-draft-int-period), answering RT-INT-1
(note:compute-accounting/20261019T0528Z-draft-rt-int): `CircuitWork`'s one window reaches `ε < 1%` only at about
10,552 equal live rounds, and a served window has 18. A period is `n` windows of one declared circuit, run in order,
each with its own live coins and its own draw, the cheater's play in each window any function of the earlier windows'
coins. Each window is one round of the period's game, so `ServiceBudget`'s per-window budget composes over the period
(`Guarantees.ComposePeriod.PeriodBudget`) and the tail is taken once for the period: the logical term is divided by
`n`.

Each window says which of its tiles are live (RT2-INT-1, note:compute-accounting/20261019T1100Z-draft-rt2-int). On
SERVE-H1's masked unit (`PouwH1Tile_v2`, #2639) a tile whose rows are all the canonical empty row is dead: its receipt
is the canonical empty receipt at every coin, so no guess bound holds of it, and 52.8% of a served window's units are
dead. A dead tile's words count for nothing: its share weight is `0` (`liveCone`), the period's proofs read it as
skipped whole (`periodMode`), and the conclusion counts live words only (`periodLive`, `LiveWorkAt`). Weights stay
over the declared circuit's words (`wordsOf`), so the static premises on `ρ`, the stage share and the lucks are
`CircuitWork`'s, and a period of unequal loads is bounded as a share of its total words.

* `consGame`: a window's one round, then the rest of the period.
* `Period`, `Period.EveryWindow`, `Period.windowAt`, `Period.liveAt`: the cheater's play over `n` windows, each
  window's live tiles and split stages (`SplitStages`) chosen after every earlier window's coins (its rounds' coins, its
  draw and its proofs' coins); a claim of every window, whatever the past; the window played, and its live tiles, at
  window `w` along the period's coins.
* `liveWords`, `liveCone`, `DeadConeDead`: a tile's words when live; its late cone's live words over the circuit's;
  a dead tile's cone holds no live words.
* `periodGame`: the period's ideal game, window `w`'s round its `serviceGame` at weights of its live tiles.
* `periodMode`: the run's modes as the period's proofs read them: a live tile `(w, t)` has window `w`'s mode along
  its rounds' coins (`tileMode` per window), and a dead one is skipped whole.
* `PerLiveTile`: `PerTile` on the window's live tiles.
* `PeriodRun`, `Declared.periodLayer`: the period's proofs as one `Accounting.TS.DrawnLayer` over `n` rounds, each
  window `Declared.layer`'s: positions are `(window, gate)`, so each window's positions are its own, cut at its own
  coins (a registered input at its window's first coin, which binds no earlier than the service does) and written by
  its own units; the draw is each window's; only the run-dependent half (`PeriodRun`) is abstract.
* `periodLive`, `LiveWorkAt`, `rOfPeriod`: the period's live words per window; `WorkAt` on live words, with the
  shares over the declared words; the period's logical bound `Λ·ln(1/δ_s)/n + min(a, b)·D₀`.
-/

namespace Accounting.Compose

open Accounting.TS Pouw.RelativeSecurity

/-! ## The period's play and its ideal game -/

/-- **A window's round, then the rest of the period**: the one round of `g`, its share and check, followed by `next`
on its coins in place of `g`'s end. -/
def consGame {Ω : Type} {n : ℕ} : RoundGame Ω 1 → (Ω → RoundGame Ω n) → RoundGame Ω (n + 1)
  | .round d pass _, next => .round d pass next

/-- **The cheater's play over a period of `n` windows** of one circuit of `M` rounds: window by window, its live tiles
`live` (fixed before the window's first coin: on SERVE-GATE's served gate, the tiles whose row block holds an admitted
slot), its split stages (`SplitStages`: each tile's mode, rightness and checked values per stage, on the window's
rounds' coins), then the rest of the period as a function of the window's coins `Ω` (its rounds' coins, its draw and
its proofs' coins). -/
inductive Period (C Ω V : Type) (M : ℕ) : ℕ → Type
  | done : Period C Ω V M 0
  | window {n : ℕ} (live : V → Bool) (W : SplitStages C V M) (next : Ω → Period C Ω V M n) :
      Period C Ω V M (n + 1)

namespace Period

variable {C Ω V : Type} {M : ℕ}

/-- **`P` of every window, whatever the past**: of each window's live tiles and split stages along every path of
earlier windows' coins. -/
def EveryWindow (P : (V → Bool) → SplitStages C V M → Prop) : {n : ℕ} → Period C Ω V M n → Prop
  | _, .done => True
  | _, .window live W next => P live W ∧ ∀ ω, EveryWindow P (next ω)

/-- **The window played at window `w`** along the period's coins `x`. -/
def windowAt : {n : ℕ} → Period C Ω V M n → Paths Ω n → Fin n → SplitStages C V M
  | _, .done, _, w => w.elim0
  | _, .window _ W next, x, w =>
    Fin.cases (motive := fun _ => SplitStages C V M) W (fun w' => (next x.1).windowAt x.2 w') w

/-- **The live tiles of window `w`** along the period's coins `x`. -/
def liveAt : {n : ℕ} → Period C Ω V M n → Paths Ω n → Fin n → V → Bool
  | _, .done, _, w => w.elim0
  | _, .window live _ next, x, w =>
    Fin.cases (motive := fun _ => V → Bool) live (fun w' => (next x.1).liveAt x.2 w') w

end Period

/-- **A tile's live words**: its credited words (`tileWords`) when it is live, and none when it is dead. -/
def liveWords {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (live : Fin Np → Bool) (t : Fin Np) : ℕ :=
  if live t then tileWords Sc Cd Np t else 0

open Classical in
/-- **A tile's late cone's live weight**: `tileCone` at live words, over the declared circuit's words (`wordsOf`), not
the window's live words, so every window's shares are in one unit and the period's sum is a share of its total. -/
noncomputable def liveCone {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (live : Fin Np → Bool)
    (t : Fin Np) : ℝ :=
  ∑ t' : Fin Np, if Cd.Reaches t.val t'.val then (liveWords Sc Cd Np live t' : ℝ) / (wordsOf Sc Cd : ℝ) else 0

/-- **A dead tile's late cone holds no live words**: no tile a dead tile reaches along late writes carries live
words. Then a dead tile's `liveCone` is `0`, and reading it as skipped whole (`periodMode`) leaves every live tile of
words good or not as the run left it. It is computed from the IR and the window's live tiles, and it holds for every
Program COMPILE accepts, whatever is live: a credited tile reaches only itself (`kindF = 1`), and a tile of no words
reaches no tile of words (`zeroCone = 0`). -/
def DeadConeDead {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (live : Fin Np → Bool) : Prop :=
  ∀ t t' : Fin Np, live t = false → Cd.Reaches t.val t'.val → liveWords Sc Cd Np live t' = 0

/-- **The period's ideal game**: window `w`'s round is its `serviceGame` (its share `a` per unit of folded and `b` per
unit of whole work at weights `w live` of its live tiles `live`, passing when every tile its draw names has honest
checked values), and the windows follow one another, each on its own coins. -/
noncomputable def periodGame {C Ξ : Type} {N M : ℕ} (stg : Fin N → ℕ) (w : (Fin N → Bool) → Fin N → ℝ) (a b : ℝ)
    (K : ℕ) : {n : ℕ} → Period C (Paths C M × (Equiv.Perm (Fin N) × Ξ)) (Fin N) M n →
      RoundGame (Paths C M × (Equiv.Perm (Fin N) × Ξ)) n
  | _, .done => .done
  | _, .window live W next =>
    consGame (serviceGame (Ξ := Ξ) stg W (w live) a b K) fun ω => periodGame stg w a b K (next ω)

/-- **The run's modes as the period's proofs read them**: a live tile `t` of window `w` has window `w`'s mode at its
rounds' coins (`tileMode` of the window played there), and a dead one is skipped whole, so it is never good and folds
nothing. -/
def periodMode {ι C B : Type} {Np n : ℕ} (Cd : Declared ι)
    (Wp : Period C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (x : Paths (Paths C Cd.rounds × B) n)
    (u : Fin n × Fin Np) : Mode :=
  if Wp.liveAt x u.1 u.2 then ((Wp.windowAt x u.1).joint (tileStage Cd Np)).mode (coinAt x u.1).1 u.2 else .whole

/-- **A claim of every live credited tile at its stage, whatever the past**: `PerTile` on the tiles `live` names. A
dead tile carries no claim: on SERVE-H1's masked unit its whole skip passes at every coin, so no guess bound holds of
it (RT2-INT's `guess_false_dead`). -/
def PerLiveTile {ι K C : Type} {Np M : ℕ} (Sc : Scheme K) (Cd : Declared ι) (live : Fin Np → Bool)
    (W : SplitStages C (Fin Np) M) (P : Kind → ReceiptGame C 1 → ReceiptGame C 1 → Prop) : Prop :=
  W.EveryStage fun j mode right same => ∀ t : Fin Np, live t = true → tileStage Cd Np t = j → ∀ k,
    tileKind Cd Np t = some k → Sc.credited k = true →
      P k (tileGame mode right same .folded t) (tileGame mode right same .whole t)

/-! ## The period's proofs, their static half from the IR -/

/-- **The period's proofs' run-dependent half**: `ProofRun` with a window index. On the period's coins, whether window
`w`'s proof of its `i`-th drawn tile was accepted and the values its opening gives its slots, and each tile's relation
and rightness on its slots' values in window `w`. -/
structure PeriodRun (C B : Type) (m s Np n : ℕ) (V : Type) where
  accepted : Paths (Paths C m × B) n → Fin n → Fin s → Prop
  opened : Paths (Paths C m × B) n → Fin n → Fin s → ℕ → V
  rel : Paths (Paths C m × B) n → Fin n → Fin Np → (ℕ → V) → Prop
  right : Paths (Paths C m × B) n → Fin n → Fin Np → (ℕ → V) → Prop

/-- **The period's proofs as the IR defines them**: `Declared.layer` in every window, its positions tagged by their
window. Window `w`'s tile `t` reads `(w, ℓ)` for each of its slots `ℓ`, its noise reads window `w`'s coin of its round,
position `(w, ℓ)` is cut at window `w`'s coin `Declared.layer` cuts `ℓ` at and written by window `w`'s unit of `ℓ`, and
window `w`'s draw is its own (`drawOf`). No position crosses windows: a value one window hands the next (decode
feedback, a KV cache) is the next window's input, cut where that window's IR cuts it, so the period's late cones stay
inside their windows. A registered input is cut at its window's first coin, no earlier than the service commits it.
The draw is uniform over every tile, dead ones included, as the served gate's is; a drawn dead tile is proved like any
other, and its words count for nothing (`periodMode`, `liveCone`). -/
def Declared.periodLayer {ι C Ξ V : Type} (Cd : Declared ι) {n Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (pr : PeriodRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np n V) :
    DrawnLayer C (Equiv.Perm (Fin Np) × Ξ) n Cd.rounds Kd (Fin Np) ℕ (Fin n × ℕ) V where
  draw := drawOf hK
  pos := fun w t k => (w, (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length))
  coin := fun _ t => ⟨min (Cd.roundOf t.val) (Cd.rounds - 1),
    Nat.lt_of_le_of_lt (Nat.min_le_right _ _) (Nat.sub_lt hr Nat.one_pos)⟩
  cut := fun ℓ => (ℓ.1, ⟨min (max (Cd.cutOf ℓ.2) (Cd.commitRound ℓ.2)) Cd.rounds,
    Nat.lt_succ_of_le (Nat.min_le_right _ _)⟩)
  src := fun ℓ => if h : ℓ.2 < Cd.dag.length ∧ Cd.binding ℓ.2 ≠ some .coin ∧ Cd.unit ℓ.2 < Np then
    some (ℓ.1, ⟨Cd.unit ℓ.2, h.2.2⟩) else none
  accepted := pr.accepted
  opened := pr.opened
  rel := pr.rel
  right := pr.right

/-! ## The period's conclusion -/

/-- **The period's live words per window**: the mean over its `n` windows of the words of each window's live tiles,
along the period's coins. With every tile live it is `N_w` when the tiles hold the circuit's credited words. -/
noncomputable def periodLive {ι K C B : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (Wp : Period C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (x : Paths (Paths C Cd.rounds × B) n) : ℝ :=
  (∑ w : Fin n, ∑ t : Fin Np, (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ)) / n

/-- **Work on live words**: `WorkAt` with the live words `N_live` in place of `N_w`, the shares `d_F, d₀` still of the
declared circuit's `N_w` words. All but `(d_F + d₀)·N_w` of the `N_live` live words are fixed right and all but
`d₀·N_w` folded, at cost `a·d_F + b·d₀ < r`. At `N_live = N_w` it is `WorkAt`. As a share of the live words the
deficit is `(d_F + d₀)·N_w/N_live`. -/
def LiveWorkAt (Nlive Nw a b r fixed folded : ℝ) : Prop :=
  ∃ dF d0 : ℝ, 0 ≤ dF ∧ 0 ≤ d0 ∧ Nlive - (dF + d0) * Nw ≤ fixed ∧ Nlive - d0 * Nw ≤ folded ∧ a * dF + b * d0 < r

/-- **The period's logical bound** `r = Λ·ln(1/δ_s)/n + min(a, b)·D₀`: the service's budget at `δ_s` over the period,
per window, and the cone weight on tiles of no words (`zeroCone`), which every window carries and no draw detects. At
`n = 1` it is `rOf`. -/
noncomputable def rOfPeriod {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np n : ℕ) (a b Λ δs : ℝ) : ℝ :=
  Λ * Real.log (1 / δs) / n + min a b * zeroCone Sc Cd Np

end Accounting.Compose
