import Definitions.Accounting.Compose

/-!
# PoUW's work over a monitoring period of `n` windows (trusted)

The vocabulary of `Definitions.Accounting.Guarantees.ComposePeriod`. `CircuitWork`'s one window reaches `ε < 1%` only
at about 10,552 equal live rounds, and the served H1 window has 18. A period is up to `n` windows of one declared
circuit, run in order, each with its own live coins and its own draw, the cheater's play in each window any function
of the earlier windows' coins. Each window is one round of the period's game, so `ServiceBudget`'s per-window budget
composes over the period (`Guarantees.ComposePeriod.PeriodBudget`) and the tail is taken once for the period: the
logical term is divided by the number of windows credited.

Each window says which of its tiles are live. On a masked tile Definition a dead tile's receipt is the canonical empty
receipt at every coin, so no guess bound holds of it. A dead tile's words count for nothing: its share weight is `0`
(`liveCone`), the period's proofs read it as skipped whole (`periodMode`), and the conclusion counts live words only
(`creditedLive`, `LiveWorkAt`). Weights stay over the declared circuit's words (`wordsOf`), so the static premises on
`ρ`, the stage share and the lucks are `CircuitWork`'s, and a period of unequal loads is bounded as a share of its
total words.

A period is credited up to its first window with a refused proof: `n` is the most windows it may hold, and the
windows from the first failure on count as skipped whole, as dead tiles do. That window is a stopping time of the
period's coins, which the cheater may steer, and the tail at a stopping time is `budget_tail`'s maximal form
(`Guarantees.ComposePeriod.PeriodTailMax`, over the game's prefixes: `RoundGame.acceptsUpTo`,
`RoundGame.deficiencyUpTo`).

* `consGame`: a window's one round, then the rest of the period.
* `Period`, `Period.EveryWindow`, `Period.windowAt`, `Period.liveAt`: the cheater's play over `n` windows, each
  window's live tiles and split stages (`SplitStages`) chosen after every earlier window's coins (its rounds' coins, its
  draw and its proofs' coins); a claim of every window, whatever the past; the window played, and its live tiles, at
  window `w` along the period's coins.
* `liveWords`, `liveCone`, `DeadConeDead`: a tile's words when live; its late cone's live words over the circuit's;
  a dead tile's cone holds no live words.
* `periodGame`: the period's ideal game, window `w`'s round its `serviceGame` at weights of its live tiles.
* `RoundGame.acceptsUpTo`, `RoundGame.deficiencyUpTo`: a game's first `j` rounds pass; their shares' sum.
* `PeriodRun`, `PeriodRun.Credited`, `PeriodRun.prefixLen`: the period's proofs' run-dependent half; window `w` and
  every earlier one had every drawn tile's proof accepted; the number of such windows, the accepted prefix.
* `periodMode`: the run's modes as the period's proofs read them: a live tile `(w, t)` of a credited window has window
  `w`'s mode along its rounds' coins (`tileMode` per window), and every other tile is skipped whole.
* `PerLiveTile`: `PerTile` on the window's live tiles.
* `Declared.periodLayer`: the period's proofs as one `Accounting.TS.DrawnLayer` over `n` rounds, each window
  `Declared.layer`'s: positions are `(window, gate)`, so each window's positions are its own, cut at its own coins (a
  registered input at its window's first coin, which binds no earlier than the service does) and written by its own
  proof blocks; the draw is each window's; only the run-dependent half (`PeriodRun`) is abstract.
* `PrefixOf`: `DrawnLayer.Of`'s game clause on every prefix: right drawn tiles in the first `j` windows pass the
  game's first `j` rounds.
* `creditedLive`, `LiveWorkAt`, `rOfPrefix`: the credited windows' live words; `WorkAt` on live words, with the
  shares over the declared words; the period's logical bound over its accepted prefix of `j` windows,
  `Λ·ln(1/δ_s) + j·min(a, b)·D₀`.
* `epsPrefix`, `creditPrefix`: that bound as a share `ε(j)` of the prefix's live words, and the period's credit in
  live words, `(1 − ε(j))₊` of them.
-/

namespace Accounting.Compose

open Accounting.TS Pouw.RelativeSecurity

/-! ## The period's play and its ideal game -/

/-- **A window's round, then the rest of the period**: the one round of `g`, its share and check, followed by `next`
on its coins in place of `g`'s end. -/
def consGame {Ω : Type} {n : ℕ} : RoundGame Ω 1 → (Ω → RoundGame Ω n) → RoundGame Ω (n + 1)
  | .round d pass _, next => .round d pass next

namespace RoundGame

variable {Ω : Type}

/-- **The game's first `j` rounds pass** along `x` (every round when `j` is past its last). At `j = 0` it holds, and at
`j ≥ n` it is `RoundGame.accepts`. -/
def acceptsUpTo : {n : ℕ} → RoundGame Ω n → ℕ → Paths Ω n → Prop
  | _, .done, _, _ => True
  | _, .round _ _ _, 0, _ => True
  | _, .round _ pass next, j + 1, x => pass x.1 ∧ acceptsUpTo (next x.1) j x.2

/-- **The shares of the game's first `j` rounds** along `x`: `RoundGame.deficiency` cut after round `j`. -/
def deficiencyUpTo : {n : ℕ} → RoundGame Ω n → ℕ → Paths Ω n → ℝ
  | _, .done, _, _ => 0
  | _, .round _ _ _, 0, _ => 0
  | _, .round d _ next, j + 1, x => d x.1 + deficiencyUpTo (next x.1) j x.2

end RoundGame

/-- **The cheater's play over a period of `n` windows** of one circuit of `M` rounds: window by window, its live tiles
`live` (fixed before the window's first coin: on the served gate, the tiles whose row block holds an admitted slot, so
a window's gate admits every slot before its first coin; a window that admits in a later round is not of this form),
its split stages (`SplitStages`: each tile's mode, rightness and
checked values per stage, on the window's rounds' coins), then the rest of the period as a function of the window's
coins `Ω` (its rounds' coins, its draw and its proofs' coins). -/
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

/-- **The period's proofs' run-dependent half**: `ProofRun` with a window index. On the period's coins, whether window
`w`'s proof of its `i`-th drawn tile was accepted and the values its opening gives its slots, and each tile's relation
and rightness on its slots' values in window `w`. -/
structure PeriodRun (C B : Type) (m s Np n : ℕ) (V : Type) where
  accepted : Paths (Paths C m × B) n → Fin n → Fin s → Prop
  opened : Paths (Paths C m × B) n → Fin n → Fin s → ℕ → V
  rel : Paths (Paths C m × B) n → Fin n → Fin Np → (ℕ → V) → Prop
  right : Paths (Paths C m × B) n → Fin n → Fin Np → (ℕ → V) → Prop

/-- **Window `w` is credited** on the period's coins `x`: it and every earlier window had every drawn tile's proof
accepted. The period stops at its first window with a refused proof, with no retry, and credits the windows before
it. -/
def PeriodRun.Credited {C B V : Type} {m s Np n : ℕ} (pr : PeriodRun C B m s Np n V) (x : Paths (Paths C m × B) n)
    (w : Fin n) : Prop :=
  ∀ w' : Fin n, w' ≤ w → ∀ i, pr.accepted x w' i

open Classical in
/-- **The accepted prefix's length**: the number of credited windows, which are the first `prefixLen` windows. Its
first window past them is the period's first failure, a stopping time of its coins. -/
noncomputable def PeriodRun.prefixLen {C B V : Type} {m s Np n : ℕ} (pr : PeriodRun C B m s Np n V)
    (x : Paths (Paths C m × B) n) : ℕ :=
  (Finset.univ.filter fun w : Fin n => pr.Credited x w).card

open Classical in
/-- **The run's modes as the period's proofs read them**: a live tile `t` of a credited window `w` has window `w`'s
mode at its rounds' coins (`tileMode` of the window played there), and every other tile, dead or in a window past the
accepted prefix, is skipped whole, so it is never good and folds nothing. -/
noncomputable def periodMode {ι C B V : Type} {s Np n : ℕ} (Cd : Declared ι)
    (Wp : Period C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (pr : PeriodRun C B Cd.rounds s Np n V)
    (x : Paths (Paths C Cd.rounds × B) n) (u : Fin n × Fin Np) : Mode :=
  if pr.Credited x u.1 ∧ Wp.liveAt x u.1 u.2 = true then
    ((Wp.windowAt x u.1).joint (tileStage Cd Np)).mode (coinAt x u.1).1 u.2
  else .whole

/-- **A claim of every live credited tile at its stage, whatever the past**: `PerTile` on the tiles `live` names. A
dead tile carries no claim: on a masked tile Definition its whole skip passes at every coin, so no guess bound holds
of it. -/
def PerLiveTile {ι K C : Type} {Np M : ℕ} (Sc : Scheme K) (Cd : Declared ι) (live : Fin Np → Bool)
    (W : SplitStages C (Fin Np) M) (P : Kind → ReceiptGame C 1 → ReceiptGame C 1 → Prop) : Prop :=
  W.EveryStage fun j mode right same => ∀ t : Fin Np, live t = true → tileStage Cd Np t = j → ∀ k,
    tileKind Cd Np t = some k → Sc.credited k = true →
      P k (tileGame mode right same .folded t) (tileGame mode right same .whole t)

/-! ## The period's proofs, their static half from the IR -/

/-- **The period's proofs as the IR defines them**: `Declared.layer` in every window, its positions tagged by their
window. Window `w`'s tile `t` reads `(w, ℓ)` for each of its slots `ℓ`, its noise reads window `w`'s coin of its round,
position `(w, ℓ)` is cut at window `w`'s coin `Declared.layer` cuts `ℓ` at and written by window `w`'s proof block of
`ℓ`, and window `w`'s draw is its own (`drawOf`). No position crosses windows: a value one window hands the next
(decode feedback, a KV cache) is the next window's input, cut where that window's IR cuts it, so the period's late
cones stay inside their windows. A registered input is cut at its window's first coin, no earlier than the service
commits it. The draw is uniform over every tile, dead ones included, as the served gate's is; a drawn dead tile is
proved like any other, and its words count for nothing (`periodMode`, `liveCone`). -/
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

/-- **The game is its drawn tiles' proofs', on every prefix**: `DrawnLayer.Of`'s game clause cut after each window.
When every tile the draws of the first `j` windows name is right at its committed values, the game's first `j` rounds
pass. A window's round reads only its own coins and draw, so a window past the prefix doesn't bear on it. -/
def PrefixOf {C B : Type} [Fintype C] [Fintype B] {n m s : ℕ} {J Q Pos V : Type} [Nonempty V]
    (pf : DrawnLayer C B n m s J Q Pos V) (g : RoundGame (Paths C m × B) n) : Prop :=
  ∀ x j, (∀ w : Fin n, w.val < j → ∀ i, pf.right x w (pf.drawnAt x w i) (pf.view x w (pf.drawnAt x w i))) →
    RoundGame.acceptsUpTo g j x

/-! ## The period's conclusion -/

open Classical in
/-- **The credited windows' live words**: the words of the live tiles of every window of the accepted prefix
(`PeriodRun.Credited`), along the period's coins. With every tile live and every window credited it is `n·N_w` when
the tiles hold the circuit's credited words. -/
noncomputable def creditedLive {ι K C B V : Type} (Sc : Scheme K) (Cd : Declared ι) {s Np n : ℕ}
    (Wp : Period C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (pr : PeriodRun C B Cd.rounds s Np n V)
    (x : Paths (Paths C Cd.rounds × B) n) : ℝ :=
  ∑ w : Fin n, if pr.Credited x w then ∑ t : Fin Np, (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ) else 0

/-- **Work on live words**: `WorkAt` with the live words `N_live` in place of `N_w`, the shares `d_F, d₀` still of the
declared circuit's `N_w` words. All but `(d_F + d₀)·N_w` of the `N_live` live words are fixed right and all but
`d₀·N_w` folded, at cost `a·d_F + b·d₀ < r`. At `N_live = N_w` it is `WorkAt`. As a share of the live words the
deficit is `(d_F + d₀)·N_w/N_live`. -/
def LiveWorkAt (Nlive Nw a b r fixed folded : ℝ) : Prop :=
  ∃ dF d0 : ℝ, 0 ≤ dF ∧ 0 ≤ d0 ∧ Nlive - (dF + d0) * Nw ≤ fixed ∧ Nlive - d0 * Nw ≤ folded ∧ a * dF + b * d0 < r

/-- **The period's logical bound over an accepted prefix of `j` windows**, `r = Λ·ln(1/δ_s) + j·min(a, b)·D₀`: the
service's budget at `δ_s`, once for the period, and the cone weight on tiles of no words (`zeroCone`), which every
credited window carries and no draw detects. Per credited window it is `Λ·ln(1/δ_s)/j + min(a, b)·D₀`; at `j = 1` it
is `rOf`. -/
noncomputable def rOfPrefix {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np j : ℕ) (a b Λ δs : ℝ) : ℝ :=
  Λ * Real.log (1 / δs) + j * (min a b * zeroCone Sc Cd Np)

/-- **The period's `ε` at its accepted prefix, in live words**: `ε(j) = r(j)·N_w/(min(a, b)·N_live)`, the logical bound
at the realized prefix `j` (`rOfPrefix`) as a share of the prefix's live words `N_live` (`creditedLive`). Under
`LiveWorkAt` the words not fixed right are at most `(d_F + d₀)·N_w`, and `min(a, b)·(d_F + d₀) ≤ a·d_F + b·d₀ < r(j)`,
so they are fewer than `ε(j)·N_live`. It is read at the realized prefix, never at the period's cap `n`. -/
noncomputable def epsPrefix {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np j : ℕ) (Nlive a b Λ δs : ℝ) : ℝ :=
  rOfPrefix Sc Cd Np j a b Λ δs * (wordsOf Sc Cd : ℝ) / (min a b * Nlive)

/-- **The period's credit in live words**: `(1 − ε(j))₊·N_live`, the prefix's live words discounted by `ε` at the
realized prefix (`epsPrefix`). A prefix whose `ε` is at least `1` earns nothing. -/
noncomputable def creditPrefix {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np j : ℕ) (Nlive a b Λ δs : ℝ) : ℝ :=
  max (1 - epsPrefix Sc Cd Np j Nlive a b Λ δs) 0 * Nlive

end Accounting.Compose
