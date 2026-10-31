import Properties.Pouw.PeriodJ10R

/-!
# A period's credit over several tile strata of the sampling block, J10-R

`Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod` over a finite set `S` of the sampling block's tile strata in place
of its one stratum `s_T`. Each stratum `s ∈ S` is one contraction `K` of the window's layout (H1's served layout has
two, 16,384 and 65,536), with its own population `N_s = |stratum σ s|` (`tileCount σ s`), its own draw `k s`, declared circuit
`Cd s`, live share `ρ s` and rate `Λ_s = j10rRate (k s) (ρ s)`, so its own `ε_s(j)`. The conclusion sums the strata:
except with probability `∑_{s ∈ S} (δ_s(s) + n·(δ_bind(s) + δ_ks(s)))`, an accepted period's fixed words, summed over
`S`, are at least the sum over `S` of each stratum's `(1 − ε_s(j))₊` of its live words in the windows before the first
window `Credit.window` refuses.

* `BlockDraw`, `Rest`, `StrataCoin`: the sampling block's stratified draw, one permutation of each of its strata, drawn
  independently (`FlockSoundness.Audit.Law.stratified`'s coordinates; stratum `s`'s draw is its permutation's first
  `k s` images, `tileDraw`); the rest of it beside stratum `s`; and a window's coins, its rounds' coins and the block's
  draw.
* `atStratum`, `pathsCongr`, `pathsAt`, `pullAt`, `creditAt`: stratum `s`'s view. A window's coins split at `s`
  (`Equiv.piSplitAt`) are the one-stratum statement's `Coin` at `s_T = s` and `Ξ = Rest σ s`: the rounds' coins, the
  stratum's permutation and the rest of the block's draw. The period's coins split window by window, and a period-wide
  input (`Credit.window`'s inputs, the service's openings, a tile's place) is read on them.
* `layerAt`, `runAt`, `relAt`, `rightAt`: `tileLayer`, `tileRun`, `pinnedAt` and `j10rAtCoin` at stratum `s`'s view.
* `StrataDrawIsCoin`: on every accepted window, the proof blocks `Credit.window`'s draw demands (`creditDrawn`) are the window
  proof blocks of the tile strata's coordinates of the block's draw, `k s` of stratum `s` for each `s ∈ S`.
* `stratumLive`: stratum `s`'s live words in the windows before the first refusal.
* `H1J10RCheckedCreditPeriodStrata`: the statement.

*Period-wide and per stratum.* A window has one registration, one salt, one draw and one verdict, whatever its strata,
so these are period-wide, stated once on the period's coins (`StrataCoin`):
* `Credit.window`'s inputs (`ws`) and so the accepted prefix, which every stratum's credit reads;
* `RegisteredBeforeDraw`, of `ws`;
* `Properties.Pouw.Period.SaltIsLiveCoin`, of the window's one salt: every proof block's noise, in every stratum, reads it
  through its call's `seed_c`;
* `StrataDrawIsCoin`, in place of `TileDrawIsCoin` (below);
* the service's openings `opened` (of window proof blocks, whatever their stratum), the salt's encoding `enc`, the scheme
  `Sc`, the period's length `n` and the block (`σ`, `k`, `S`, `unitOf`), fixed at the period's open.

The rest is per stratum, the one-stratum statement's premise at stratum `s`'s view, for each `s ∈ S`: the data (`Cd s`,
`Wp s`, `Pl s`, `folded s`, `place s`, `tile s`, `ρ s`, `δ_s(s)`, `δ_bind(s)`, `δ_ks(s)`), `Scope`, the live share
and stage share bounds, H1's lucks, the play's game, `DeadConeDead`, `PrefixOf`, binding at `δ_bind(s)`,
`TileServiceSound` at `δ_ks(s)` (C-Flock's soundness at the stratum's drawn proof blocks), `SharedNoiseLucks` (the stratum's
tile games) and `UnitFixed`.

*Why the draw's premise is period-wide.* `TileDrawIsCoin` says the window's drawn proof blocks are exactly the `k s_T` of the
tile stratum. A window over several strata draws `∑_{s ∈ S} k s` proof blocks, which no one stratum's coordinate covers, so
read per stratum it is false at every honest window with two non-empty strata, and a statement taking it per stratum
would say nothing. `StrataDrawIsCoin` says they are exactly the union over `S` of the strata's coordinates; at one
stratum it is `TileDrawIsCoin`'s body on the block's draw. A code fact once its code exists, as `TileDrawIsCoin` is:
the verifier draws the tiles inside the sampling block's stratified draw, one stratum a `K`, and reads `k s`
from it; `Credit.window` today draws by its own sampling design, one stratum a `K` template (`Credit.strata`). Where the draw's
stream is expanded from the live coin by SHA-512, reading it as the uniform permutations uses SHA-512 as a random
oracle (R-ACCEL-3).

*The form it covers.* The verifier accepts this statement's form when each window of the period is an H1 window over
J10-R's proof block on `PouwH1Tile_v5` whose proof blocks are at one or more `K`, each `K` one stratum of `S`, with the same strata
(and so the same `N_s` and `k s`) in every window, as the period's open commits them; a window with another set of
`K`s, or a `K` split over strata or strata merged, has a draw `StrataDrawIsCoin` doesn't describe.

*The bound.* The strata's permutations are independent coordinates of the block's draw, but the bound doesn't use it:
every stratum's game reads the window's one round coin and the prover adapts across strata, so the strata's failure
events are not independent, and the union bound, `∑_{s ∈ S}` of the one-stratum bounds, is what holds.

The proof is `Proofs.Pouw.PeriodJ10RStrata`'s: at each stratum, what `Proofs.Pouw.PeriodJ10R` applies
(`CircuitCreditPeriod` at `tileRun`), with the stratum's joint soundness from `StrataDrawIsCoin` and its
`TileServiceSound`, carried to the period's coins by `pathsAt`, then the union bound.
-/

namespace Properties.Pouw.PeriodJ10RStrata

open Properties.Pouw.PeriodJ10R
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-! ## The sampling block's draw and stratum `s`'s view of it -/

/-- **The sampling block's draw**: a permutation of each stratum `s` of `σ`, independently; stratum `s`'s draw is the
proof blocks of its first `k s` images (`tileDraw`). -/
abbrev BlockDraw {nB mB : ℕ} (σ : Fin nB → Fin mB) : Type := (s : Fin mB) → Equiv.Perm (Fin (tileCount σ s))

/-- **The rest of the block's draw beside stratum `s`**: the other strata's permutations. -/
abbrev Rest {nB mB : ℕ} (σ : Fin nB → Fin mB) (s : Fin mB) : Type :=
  (j : {j : Fin mB // j ≠ s}) → Equiv.Perm (Fin (tileCount σ j.1))

/-- **A window's coins over the block's strata**: its `m` rounds' coins and the block's draw. -/
abbrev StrataCoin (C : Type) (m : ℕ) {nB mB : ℕ} (σ : Fin nB → Fin mB) : Type := Paths C m × BlockDraw σ

/-- **Stratum `s`'s view of a window's coins**: the one-stratum statement's `Coin` at `s_T = s`, the rounds' coins (`r`
of them, `r = m`), then the block's draw split at `s` into the stratum's permutation and the rest. -/
def atStratum {C : Type} {r m nB mB : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB) :
    StrataCoin C m σ ≃ Coin C (Rest σ s) r σ s :=
  (Equiv.cast (congrArg (Paths C) h.symm)).prodCongr (Equiv.piSplitAt s _)

/-- **A relabelling of each window's coins, on the period's coins**, window by window. -/
def pathsCongr {α β : Type} (e : α ≃ β) : (n : ℕ) → Paths α n ≃ Paths β n
  | 0 => Equiv.refl Unit
  | n + 1 => e.prodCongr (pathsCongr e n)

/-- **Stratum `s`'s view of the period's coins**: `atStratum` in every window. -/
def pathsAt (C : Type) {r m nB mB : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB) (n : ℕ) :
    Paths (StrataCoin C m σ) n ≃ Paths (Coin C (Rest σ s) r σ s) n :=
  pathsCongr (atStratum h σ s) n

/-- **A period-wide input read on stratum `s`'s view** of the period's coins. -/
def pullAt {C β : Type} {r m nB mB : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB) {n : ℕ}
    (f : Paths (StrataCoin C m σ) n → β) : Paths (Coin C (Rest σ s) r σ s) n → β :=
  fun y => f ((pathsAt C h σ s n).symm y)

/-- **The period's `Credit.window` inputs on stratum `s`'s view**: each window's registration and the rest of its
inputs, the period's at the coins the view relabels. -/
def creditAt {C : Type} {r m nB mB n : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB)
    (ws : CreditWindows (StrataCoin C m σ) n) : CreditWindows (Coin C (Rest σ s) r σ s) n where
  reg := pullAt h σ s ws.reg
  ev := pullAt h σ s ws.ev

/-! ## The one-stratum statement's proofs at stratum `s`'s view -/

/-- **Stratum `s`'s proof block relation**: `pinnedAt` at the window's salt and the place of the stratum's tiles, on its view. -/
def relAt {C V : Type} {r m nB mB n : ℕ} (h : r = m) (σ : Fin nB → Fin mB) (s : Fin mB)
    (ws : CreditWindows (StrataCoin C m σ) n) (place : Paths (StrataCoin C m σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop) :
    Paths (Coin C (Rest σ s) r σ s) n → Fin n → Fin (tileCount σ s) → (ℕ → V) → Prop :=
  pinnedAt (saltOf (creditAt h σ s ws)) (pullAt h σ s place) tile

/-- **Stratum `s`'s tile rightness**: `j10rAtCoin` at the place of the stratum's tiles, on its view. -/
def rightAt {C V : Type} {r m nB mB n : ℕ} (h : r = m) (hr : 0 < r) (σ : Fin nB → Fin mB) (s : Fin mB)
    (enc : C → ByteArray) (place : Paths (StrataCoin C m σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop) :
    Paths (Coin C (Rest σ s) r σ s) n → Fin n → Fin (tileCount σ s) → (ℕ → V) → Prop :=
  j10rAtCoin hr enc (pullAt h σ s place) tile

/-- **Stratum `s`'s proofs with `Credit.window`'s verdict**: `tileRun` at the stratum, on its view, with the period's
`Credit.window` inputs and the service's openings. -/
noncomputable def runAt {ι C V : Type} (Cd : Declared ι) (hr : 0 < Cd.rounds) {m nB mB n : ℕ} (hM : Cd.rounds = m)
    (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (s : Fin mB) (unitOf : Fin nB → ℕ)
    (ws : CreditWindows (StrataCoin C m σ) n) (opened : Paths (StrataCoin C m σ) n → Fin n → ℕ → ℕ → V)
    (enc : C → ByteArray) (place : Paths (StrataCoin C m σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop) :
    PeriodRun C (Equiv.Perm (Fin (tileCount σ s)) × Rest σ s) Cd.rounds (k s) (tileCount σ s) n V :=
  tileRun σ k hk s unitOf (creditAt hM σ s ws) (pullAt hM σ s opened) (relAt hM σ s ws place tile)
    (rightAt hM hr σ s enc place tile)

/-- **Stratum `s`'s proofs as the IR defines them**: `tileLayer` at the stratum, on its view. -/
noncomputable def layerAt {ι C V : Type} (Cd : Declared ι) (hr : 0 < Cd.rounds) {m nB mB n : ℕ} (hM : Cd.rounds = m)
    (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (s : Fin mB) (unitOf : Fin nB → ℕ)
    (ws : CreditWindows (StrataCoin C m σ) n) (opened : Paths (StrataCoin C m σ) n → Fin n → ℕ → ℕ → V)
    (enc : C → ByteArray) (place : Paths (StrataCoin C m σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop) :
    DrawnLayer C (Equiv.Perm (Fin (tileCount σ s)) × Rest σ s) n Cd.rounds (k s) (Fin (tileCount σ s)) ℕ (Fin n × ℕ) V :=
  tileLayer Cd hr σ k hk s unitOf (creditAt hM σ s ws) (pullAt hM σ s opened) (relAt hM σ s ws place tile)
    (rightAt hM hr σ s enc place tile)

/-! ## The draw and the credit -/

/-- **`Credit.window`'s draw is the block's draw at the tile strata**: on every path and every window `Credit.window`
credits, the proof blocks its draw demands proofs of (`creditDrawn`) are the window proof blocks (`unitOf`) of the tile strata's
coordinates of the block's draw at the window's coins, `tileDraw`'s `k s` proof blocks of stratum `s` for each `s ∈ S`, and
no others. At one stratum it is `TileDrawIsCoin`'s body on the block's draw; read per stratum it would be false at a
window over two non-empty strata. A code fact once its code exists. Code change: the verifier draws the tiles
inside the sampling block's stratified draw, as one subcircuit of the integrity registration, one stratum a `K`, and
reads `k s` from it; today `Credit.window` draws by its own sampling design, one stratum a `K` template. Where the draw's
stream is expanded from the live coin by SHA-512, reading it as the uniform permutations uses SHA-512 as a random
oracle (R-ACCEL-3). -/
def StrataDrawIsCoin {C : Type} {m nB mB n : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (S : Finset (Fin mB)) (unitOf : Fin nB → ℕ)
    (ws : CreditWindows (StrataCoin C m σ) n) : Prop :=
  ∀ (x : Paths (StrataCoin C m σ) n) (w : Fin n), WindowAccepts (ws.reg x w) (ws.ev x w) →
    ∃ d : Array ℕ, creditDrawn (ws.reg x w) (ws.ev x w) = .ok d ∧
      ∀ u : ℕ, u ∈ d.toList ↔ ∃ s ∈ S, ∃ b ∈ tileDraw σ k hk s ((coinAt x w).2 s), unitOf b = u

open Classical in
/-- **Stratum `s`'s live words over the accepted prefix**: the words of the stratum's live tiles (`Period.liveAt` of its
period, on its view) in each window before the first window `Credit.window` refuses. -/
noncomputable def stratumLive {ι K C : Type} (Sc : Scheme K) (Cd : Declared ι) {m nB mB n : ℕ} (hM : Cd.rounds = m)
    (σ : Fin nB → Fin mB) (s : Fin mB)
    (Wp : Period C (Coin C (Rest σ s) Cd.rounds σ s) (Fin (tileCount σ s)) Cd.rounds n)
    (ws : CreditWindows (StrataCoin C m σ) n) (x : Paths (StrataCoin C m σ) n) : ℝ :=
  ∑ w : Fin n, if ws.AcceptedUpTo x w then
    ∑ t : Fin (tileCount σ s), (liveWords Sc Cd (tileCount σ s) (Wp.liveAt (pathsAt C hM σ s n x) w) t : ℝ) else 0

/-- **(L)'s credit over the windows served H1's verdict credits, at J10-R's tile proof block, over several tile strata.** For
every scheme `Sc` and period of at most `n ≥ 1` windows, each window's `Credit.window` inputs any function of the
period's coins (`ws`), over a finite set `S` of the sampling block's tile strata of `σ`, each `s ∈ S` with its own
declared circuit `Cd s` (`(Cd s).units ≤ N_s = |stratum σ s|`, one round a window), draw `k s ≥ 1`, live share
`0 < ρ s ≤ 1` and rate `Λ_s = j10rRate (k s) (ρ s)`: under the period-wide premises `RegisteredBeforeDraw`,
`StrataDrawIsCoin` and `SaltIsLiveCoin`, and at each `s ∈ S` the premises `H1J10RCheckedCreditPeriod` takes at its one
stratum, at stratum `s`'s view (`layerAt`, `runAt`, `relAt`), with `TileDrawIsCoin` replaced by `StrataDrawIsCoin`,
except with probability `∑_{s ∈ S} (δ_s(s) + n·(δ_bind(s) + δ_ks(s)))` an accepted period's fixed words summed over
`S` are at least the sum over `S` of `(1 − ε_s(j))₊` of stratum `s`'s live words in the windows before the first
refusal, `ε_s` at the stratum's `N_s`, `Cd s`, `Λ_s` and `δ_s(s)`. -/
def H1J10RCheckedCreditPeriodStrata : Prop :=
  ∀ {ι K T L U C V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Nonempty V]
    {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Finset (Fin mB)) (unitOf : Fin nB → ℕ) (Sc : Scheme K) (Cd : Fin mB → Declared ι) {n : ℕ} (_hn : 0 < n)
    (_hN : ∀ s ∈ S, (Cd s).units ≤ tileCount σ s) (hr : ∀ s, 0 < (Cd s).rounds) (hM : ∀ s, (Cd s).rounds = 1)
    (_hKd : ∀ s ∈ S, 1 ≤ k s)
    (ws : CreditWindows (StrataCoin C 1 σ) n) (opened : Paths (StrataCoin C 1 σ) n → Fin n → ℕ → ℕ → V)
    (enc : C → ByteArray) (place : (s : Fin mB) → Paths (StrataCoin C 1 σ) n → Fin n → Fin (tileCount σ s) → Place)
    (tile : (s : Fin mB) → Noise → List (List ℕ) → Fin (tileCount σ s) → (ℕ → V) → Prop)
    (Wp : (s : Fin mB) → Period C (Coin C (Rest σ s) (Cd s).rounds σ s) (Fin (tileCount σ s)) (Cd s).rounds n)
    (Pl : (s : Fin mB) → Play (Coin C (Rest σ s) (Cd s).rounds σ s) n K T L U)
    (folded : (s : Fin mB) → Paths (Coin C (Rest σ s) (Cd s).rounds σ s) n → ℝ) (ρ δs δbind δks : Fin mB → ℝ),
    RegisteredBeforeDraw ws → StrataDrawIsCoin σ k hk S unitOf ws →
    Properties.Pouw.Period.SaltIsLiveCoin Nat.one_pos (saltOf ws) enc →
    (∀ s ∈ S, Scope Sc (Cd s) (δs s)) →
    (∀ s ∈ S, 0 < ρ s ∧ ρ s ≤ 1) →
    (∀ s ∈ S, ∀ t : Fin (tileCount σ s), ρ s * (tileCount σ s) * shareCone Sc (Cd s) (tileCount σ s) t ≤ 1) →
    (∀ s ∈ S, ∀ j, stageShare (tileStage (Cd s) (tileCount σ s)) (shareCone Sc (Cd s) (tileCount σ s)) j ≤ 1) →
    (∀ s ∈ S, ∀ t : Fin (tileCount σ s), 0 < shareCone Sc (Cd s) (tileCount σ s) t →
      tileLuck Sc (Cd s) (tileCount σ s) t ≤ 7 / 8 ∧ tileLuck0 Sc (Cd s) (tileCount σ s) t ≤ 1 / 2) →
    (∀ s ∈ S, (Pl s).game =
      periodGame (tileStage (Cd s) (tileCount σ s)) (liveCone Sc (Cd s) (tileCount σ s)) 1 1 (k s) (Wp s)) →
    (∀ s ∈ S, (Wp s).EveryWindow (fun live _ => DeadConeDead Sc (Cd s) (tileCount σ s) live)) →
    (∀ s ∈ S, PrefixOf (layerAt (Cd s) (hr s) (hM s) σ k hk s unitOf ws opened enc (place s) (tile s)) (Pl s).game) →
    (∀ s ∈ S, ∀ w, prob (fun y => (layerAt (Cd s) (hr s) (hM s) σ k hk s unitOf ws opened enc (place s)
      (tile s)).MovedIn y w) ≤ δbind s) →
    (∀ s ∈ S, TileServiceSound σ s unitOf (creditAt (hM s) σ s ws) (pullAt (hM s) σ s opened)
      (relAt (hM s) σ s ws (place s) (tile s)) (δks s)) →
    (∀ s ∈ S, SharedNoiseLucks Sc (Cd s) (Wp s)) →
    (∀ s ∈ S, UnitFixed (Pl s) (layerAt (Cd s) (hr s) (hM s) σ k hk s unitOf ws opened enc (place s) (tile s))
      (periodMode (Cd s) (Wp s) (runAt (Cd s) (hr s) (hM s) σ k hk s unitOf ws opened enc (place s) (tile s)))
      (fun _ t => (tileWords Sc (Cd s) (tileCount σ s) t : ℝ)) (folded s)) →
    prob (fun x : Paths (StrataCoin C 1 σ) n => (∀ s ∈ S, (Pl s).accepts (pathsAt C (hM s) σ s n x)) ∧
      n * ∑ s ∈ S, ((Pl s).run (pathsAt C (hM s) σ s n x)).fixed <
        ∑ s ∈ S, creditPrefix Sc (Cd s) (tileCount σ s) (ws.prefixLen x)
          (stratumLive Sc (Cd s) (hM s) σ s (Wp s) ws x) 1 1 (j10rRate (k s) (ρ s)) (δs s)) ≤
      ∑ s ∈ S, (δs s + n * (δbind s + δks s))

end Properties.Pouw.PeriodJ10RStrata
