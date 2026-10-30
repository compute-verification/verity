import Definitions.Core.Formalism
import Properties.Pouw.PeriodJ10R

/-!
# The served PoUW's values in AND gates (`PouwCheck.Credit.window`, J10-R)

`Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod` is stated in words: except with probability
`δ_s + n·(δ_bind + δ_ks)`, an accepted period's fixed words are at least `(1 − ε(j))₊` of the live words of the windows
before the first refusal. `RightValues` states that bound in AND gates of the registered declared circuit under its
pinned lowering (`Verity.creditSize lower D`): except with probability δ_coin + δ_crypto,
an accepted period's right values cover at least `(1 − ε_skip)₊·S` AND gates. It is the values guarantee, "declared
size, values certified".

*Not a compute claim.* Right values aren't work: one cheap faulty value upstream can make its
whole cone right, and values that read no live coin can be replayed. So this says which of the declared circuit's
values are certified, never what the prover spent. A compute guarantee for the served PoUW is a separate statement,
which adds SAK to H1.

* `BlockMap`: the public map from words to gates. Proof block `t`'s gates in `lower D`, the registered declaration `D`
  under the lowering `lower` pinned at registration (the record carries its digest), and word `i`'s bits there,
  which are outputs of its block. Each unit is one instance of its `K`'s `PouwH1WindowTile_v1{UNIT=…,K=…}`
  (`Credit.layoutOf`), and its proof block is that instance's gates.
* `BlockMap.size`: block `t`'s credit size, the AND count (`Verity.Circuit.andSize`) of its gates outside `ledger`,
  the gates of `lower D` another protocol already credits (sampled proofs certify the served GEMMs too). One
  ledger, so no gate is counted twice, and the blocks' sizes sum to at most the credit `Verity.creditSize lower D`
  (`Proofs.Pouw.ServedValues`' `sum_size_le_creditSize`).
* `liveSize`, `creditedSize`: the credited size `S`, the size of each live block of credited words over the windows
  before the first refusal: `creditLive` with each block's size in place of its words. It is public, read off the
  registration, the live tiles and the verdict.
* `rightSize`: the size of the blocks the run computed with their late cones, `DrawnLayer.Good`: the blocks whose
  values are right. E1(c) (`accounting-ts-unit-fixed`) says a good block's words are its own right values: its unit
  and every unit that writes it a value after its coin are computed, none skipped and guessed. "Right" is the
  accounting model's `DrawnLayer.Good`: nothing yet ties it to the committed values `Credit.window` checks.
* `sizePerWord`, `epsSkip`, `creditFloor`: the most size a credited word carries, `σ = max_t size_t / words_t` over the
  blocks of credited words. Then `ε_skip = σ·ε(j)·N_live/S`, and the right values must cover `(1 − ε_skip)₊·S` AND
  gates. When every block of credited words carries the same size per word, `S = σ·N_live` and `ε_skip = ε(j)`.
* `RightValues`: except with H1's probability, an accepted period's right blocks have total size at least
  `(1 − ε_skip)₊·S`.

*No algorithmic premise.* Being about values, it takes no SAK and no premise beyond H1's: `TileDrawIsCoin` (the draw),
`TileServiceSound` (the proof service's soundness, open), `RegisteredBeforeDraw`, `SaltIsLiveCoin` and
`SharedNoiseLucks`, with the structure H1 reads (`Scope`, the live share `ρ`, the game and its prefix, the binding at
`δ_bind`). `SharedNoiseLucks` is its one computational premise: the fold receipts' detection, which is to be
proved in the slot's model or designed away by the no-fold receipt, not assumed. It drops H1's `UnitFixed` (and
the play's `folded` words): it reads the right blocks themselves, not the play's abstract `fixed` count.

Its `δ_s` is δ_coin and its `n·(δ_bind + δ_ks)` is δ_crypto. `BlockMap` is a named map from words to gates of the
registered circuit, the unit the PoUW component's `Holds` is restated in.
-/

namespace Properties.Pouw.ServedValues

open Properties.Pouw.PeriodJ10R
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **The declared circuit's proof blocks and its words' bits**, the public map from words to gates. `gates t` is proof
block `t`'s gates in `lower D`, the registered declaration `D` under the lowering `lower` pinned at registration: the
gates of the instance of its subcircuit template at its shape. `bits i` is word `i`'s bits, gate `i` of the word-level
circuit `Cd` lowered to `lower D`'s one-bit gates. The blocks are disjoint and hold no input gate. Each word of block
`t` (`unitWords Cd t`) is an output of the block: its bits are nonempty and among the block's exits
(`Verity.Circuit.exits`: the block's gates that a gate outside it reads, or that are outputs of `lower D`). It is fixed
at registration with `D`, `lower` and `Cd`, so each block's size is public. -/
structure BlockMap {ι : Type} (Cd : Declared ι) (Np : ℕ) (lower : Verity.Declaration → Verity.Circuit)
    (D : Verity.Declaration) where
  gates : Fin Np → Finset (Fin (lower D).N)
  bits : ℕ → Finset (Fin (lower D).N)
  disjoint : ∀ t t' : Fin Np, t ≠ t' → Disjoint (gates t) (gates t')
  computed : ∀ t, gates t ⊆ (lower D).computed
  nonempty : ∀ t : Fin Np, ∀ i ∈ unitWords Cd t.val, (bits i).Nonempty
  exits : ∀ t : Fin Np, ∀ i ∈ unitWords Cd t.val, bits i ⊆ (lower D).exits (gates t)

/-- **Proof block `t`'s size**: the credit size of its gates that no other protocol credits, their AND count in
`lower D` outside `ledger`. `ledger` is the gates of `lower D` another protocol's credit already counts, one ledger
for every protocol, so that no gate is counted twice. -/
noncomputable def BlockMap.size {ι : Type} {Cd : Declared ι} {Np : ℕ} {lower : Verity.Declaration → Verity.Circuit}
    {D : Verity.Declaration} (M : BlockMap Cd Np lower D) (ledger : Finset (Fin (lower D).N)) (t : Fin Np) : ℕ :=
  (lower D).andSize (M.gates t \ ledger)

/-- **A block's live size**: its size `sz t` when it carries live words (`liveWords`), and none when it is dead or of
no credited words. -/
def liveSize {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (live : Fin Np → Bool) (sz : Fin Np → ℕ)
    (t : Fin Np) : ℕ :=
  if liveWords Sc Cd Np live t = 0 then 0 else sz t

open Classical in
/-- **The credited size `S`**: the live size of every block of the windows before the first refusal
(`CreditWindows.AcceptedUpTo`), `creditLive` with each block's size in place of its words. -/
noncomputable def creditedSize {ι K C B : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (Wp : Period C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (ws : CreditWindows (Paths C Cd.rounds × B) n)
    (sz : Fin Np → ℕ) (x : Paths (Paths C Cd.rounds × B) n) : ℝ :=
  ∑ w : Fin n, if ws.AcceptedUpTo x w then ∑ t : Fin Np, (liveSize Sc Cd Np (Wp.liveAt x w) sz t : ℝ) else 0

open Classical in
/-- **The right blocks' size**: the size of every block of credited words that `right` names in each window. At
`DrawnLayer.Good` it is the size whose values the period certifies right: a good block is live in a credited window
(`periodMode` skips every other block whole). -/
noncomputable def rightSize {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ} (sz : Fin Np → ℕ)
    (right : Fin n → Fin Np → Prop) : ℝ :=
  ∑ w : Fin n, ∑ t : Fin Np, if right w t ∧ tileWords Sc Cd Np t ≠ 0 then (sz t : ℝ) else 0

/-- **The most size a credited word carries**, `σ = max_t sz_t / words_t` over the blocks of credited words, and `0`
when there are none. It is public, fixed by the registration. -/
noncomputable def sizePerWord {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (sz : Fin Np → ℕ) : ℝ :=
  ⨆ t : Fin Np, if tileWords Sc Cd Np t = 0 then 0 else (sz t : ℝ) / (tileWords Sc Cd Np t : ℝ)

/-- **ε_skip at the accepted prefix**: `σ·ε(j)·N_live/S`, the words `ε(j)` leaves unfixed (`epsPrefix`) at the most
size a word carries, as a share of the credited size `S`. When every block of credited words carries `σ` a word,
`S = σ·N_live` and it is `ε(j)`. -/
noncomputable def epsSkip {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np j : ℕ) (Nlive S : ℝ) (sz : Fin Np → ℕ)
    (a b Λ δs : ℝ) : ℝ :=
  sizePerWord Sc Cd Np sz * epsPrefix Sc Cd Np j Nlive a b Λ δs * Nlive / S

/-- **The size the right values must cover**: `(1 − ε_skip)₊·S`, the credited size discounted by `ε_skip` at the
realized prefix. A prefix whose `ε_skip` is at least `1` earns nothing. -/
noncomputable def creditFloor {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np j : ℕ) (Nlive S : ℝ)
    (sz : Fin Np → ℕ) (a b Λ δs : ℝ) : ℝ :=
  max (1 - epsSkip Sc Cd Np j Nlive S sz a b Λ δs) 0 * S

/-- **The served PoUW's values in AND gates.** Take `H1J10RCheckedCreditPeriod`'s period and premises without
`UnitFixed` (and without the play's `folded` words): a declared circuit `Cd` and a scheme `Sc` run as a period of at
most `n ≥ 1` windows over the sampling block's tile stratum `s_T` of `σ`, with `Credit.window`'s inputs (`ws`) any
function of the period's coins. Add the registered declaration `D`, the lowering `lower` pinned at registration,
`lower D`'s proof blocks and words' bits (`M : BlockMap`), and the gates other protocols credit (`ledger`). Then,
except with probability `δ_s + n·(δ_bind + δ_ks)` (δ_coin plus δ_crypto), an accepted period's right blocks have total
credit size at least `(1 − ε_skip)₊·S` AND gates. Here `S` is the credited size of the windows before the first
refusal (`creditedSize`) and `ε_skip = σ·ε(j)·N_live/S` (`epsSkip`, at H1's `ε(j)` and live words). A block is right
(`DrawnLayer.Good` under `periodMode`) when the run computed it and every block that writes it a value after its coin.
E1(c) (`accounting-ts-unit-fixed`) reads that as its words being its own right values.

It is a statement about values, not a compute claim: right values aren't work. It needs no
algorithmic premise, only H1's. `TileDrawIsCoin`, `TileServiceSound`, `RegisteredBeforeDraw` and `SaltIsLiveCoin` are
about the draw, the service and the salt. `SharedNoiseLucks` is the one computational premise: the fold receipts'
detection, which is to be proved or designed away by the no-fold receipt. -/
def RightValues : Prop :=
  ∀ {ι K T L U C Ξ V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty V] {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (sT : Fin mB) (unitOf : Fin nB → ℕ) (Sc : Scheme K) (Cd : Declared ι) {n : ℕ} (_hn : 0 < n)
    (_hN : Cd.units ≤ tileCount σ sT) (hr : 0 < Cd.rounds) (_hM : Cd.rounds = 1) (_hKd : 1 ≤ k sT)
    (D : Verity.Declaration) (lower : Verity.Declaration → Verity.Circuit) (M : BlockMap Cd (tileCount σ sT) lower D)
    (ledger : Finset (Fin (lower D).N)) (ws : CreditWindows (Coin C Ξ Cd.rounds σ sT) n)
    (opened : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → ℕ → ℕ → V) (enc : C → ByteArray)
    (place : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → Fin (tileCount σ sT) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ sT) → (ℕ → V) → Prop)
    (Wp : Period C (Coin C Ξ Cd.rounds σ sT) (Fin (tileCount σ sT)) Cd.rounds n)
    (Pl : Play (Coin C Ξ Cd.rounds σ sT) n K T L U) (ρ δs δbind δks : ℝ),
    Scope Sc Cd δs →
    0 < ρ → ρ ≤ 1 →
    (∀ t : Fin (tileCount σ sT), ρ * (tileCount σ sT) * shareCone Sc Cd (tileCount σ sT) t ≤ 1) →
    (∀ j, stageShare (tileStage Cd (tileCount σ sT)) (shareCone Sc Cd (tileCount σ sT)) j ≤ 1) →
    (∀ t : Fin (tileCount σ sT), 0 < shareCone Sc Cd (tileCount σ sT) t →
      tileLuck Sc Cd (tileCount σ sT) t ≤ 7 / 8 ∧ tileLuck0 Sc Cd (tileCount σ sT) t ≤ 1 / 2) →
    Pl.game = periodGame (tileStage Cd (tileCount σ sT)) (liveCone Sc Cd (tileCount σ sT)) 1 1 (k sT) Wp →
    Wp.EveryWindow (fun live _ => DeadConeDead Sc Cd (tileCount σ sT) live) →
    PrefixOf (tileLayer Cd hr σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
      (j10rAtCoin hr enc place tile)) Pl.game →
    (∀ w, prob (fun x => (tileLayer Cd hr σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
      (j10rAtCoin hr enc place tile)).MovedIn x w) ≤ δbind) →
    RegisteredBeforeDraw ws → TileDrawIsCoin σ k hk sT unitOf ws →
    TileServiceSound σ sT unitOf ws opened (pinnedAt (saltOf ws) place tile) δks →
    Properties.Pouw.Period.SaltIsLiveCoin hr (saltOf ws) enc → SharedNoiseLucks Sc Cd Wp →
    prob (fun x => Pl.accepts x ∧
      rightSize Sc Cd (M.size ledger) (fun w t =>
        (tileLayer Cd hr σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
          (j10rAtCoin hr enc place tile)).Good
          (periodMode Cd Wp (tileRun σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
            (j10rAtCoin hr enc place tile))) x w t) <
        creditFloor Sc Cd (tileCount σ sT) (ws.prefixLen x) (creditLive Sc Cd Wp ws x)
          (creditedSize Sc Cd Wp ws (M.size ledger) x) (M.size ledger) 1 1 (j10rRate (k sT) ρ) δs) ≤
      δs + n * (δbind + δks)

end Properties.Pouw.ServedValues
