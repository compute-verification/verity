import Definitions.Pouw.Assumptions.ServedWork
import Properties.Pouw.PeriodJ10R

/-!
# The served PoUW protocol's work: priced work on the prover's machines from an accepted period

`Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod` says what an accepted period of served H1 (`PouwCheck.Period`,
the verdict `PouwCheck.Credit.window` on each window, J10-R's noise) fixes: except with probability
`δ_s + n·(δ_bind + δ_ks)`, its fixed words are at least `(1 − ε(j))₊` of the live words of its windows before the first
refusal. `ServedWork` turns those words into work: the prover's machines, wherever they ran, spent at least `p` a
credited word of that, less the luck allowance, except with that probability plus `n·κ_deg(B)`.

It is the remote form. The prover's machines are its whole inventory, one machine of the accelerator model 𝓜
(`Pouw.Dither.Accel`) with its device premises (G1, G2, P1, P2) as hypotheses; no network certifier, wipe or latency
check enters, and no outside machine is priced: every program of the inventory exchanges no bits with machines outside
it (`G.ext = 0`), which is what "the inventory" means. The claim is about the work, not where it ran.

* `saltCoin`: window `w`'s round-0 coin, the coin its salt is (`Properties.Pouw.Period.SaltIsLiveCoin`) and its tiles'
  noise reads (`j10rAtCoin`).
* `FixedBeforeSalt`: a function of the period's coins and a window that reads only the windows before it, so it is fixed
  before that window's salt.
* `FixedAreRight`: the bridge from the play's fixed words to the inventory's outputs. On every accepted path the period's
  fixed words (`n · fixed`, `UnitFixed`'s scale: the good units' words summed over the windows) are at most the credited
  words the inventory's programs get right, window by window (`rightWords` at the window's instance and salt).
* `ServedWork`: the guarantee.

**The type bridge.** H1's words are abstract in the play (`Anatomy.fixed`, which `UnitFixed` bounds below by the good
units' words). `SakLb.SAKResidual` speaks of a program's output and the fraction of its words that are right. The bridge
is `FixedAreRight`, at the tiles' credited words (`tileWords`, `UnitFixed`'s weights), and `SAKServed` is
`SAKResidual`'s claim at the served types (a window's tiles, its salt as the dither, `p` a credited word): the right
share by tiles is at most the right share by words.

**Where ε(j), the lucks and κ_deg enter.** `ε(j)` is `creditPrefix`'s, at the realized prefix `j` and the rate
`j10rRate (k s_T) ρ`, as the words guarantee has it; `SharedNoiseLucks` is that guarantee's premise and enters only
through it. `κ_deg(B)` is `SAKServed`'s degenerate salts at `B` bits of advice, once a window: each window's instance,
advice and program are fixed before its salt (`FixedBeforeSalt`), so its salt is uniform given them and falls in their
degenerate set with probability at most `κ_deg(B)` (the union over `n` windows).

**What is left out.** The commitment's hashing is not credited, so SHA-512's premise (the SHA analogue of SAK, `SAKSha`;
`Pouw.Assumptions.ShaSpeed` in Lean) doesn't enter. The proof system's soundness (`ProofSystemSound` in prose) enters as
the words guarantee's `TileServiceSound`.

**Non-vacuity is open.** 𝓜 as written prices register reductions (a register word that an atomic or a reduction
writes costs an `alu` lane, and L2 writes are priced), so the four Russians' method pays at least an `alu` lane a MAC.
`SAKMatmul` as written still fails at a dual price that leaves those pipes cheap, and `SAKResidual` is not adopted, so
whether `SAKServed` holds on 𝓜 is open.

The proof is `Proofs.Pouw.ServedWork`; `Proofs.Pouw.ServedWork.Honest` has its honest and nontrivial lemmas.
-/

namespace Properties.Pouw.Served

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Properties.Pouw.PeriodJ10R

/-- **Window `w`'s salt coin**: its round-0 coin, which `SaltIsLiveCoin` makes its salt and `j10rAtCoin`'s noise reads. -/
def saltCoin {C B : Type} {m n : ℕ} (hm : 0 < m) (x : Paths (Paths C m × B) n) (w : Fin n) : C :=
  coinAt (coinAt x w).1 ⟨0, hm⟩

/-- **Fixed before the window's salt**: `f x w` reads only the windows before `w` (`spliceAt w 0` keeps those and takes
all of window `w` and later from `z`). Of the prover's instance, advice and program for window `w`: chosen from the
period's past, never from window `w`'s salt or anything after it. -/
def FixedBeforeSalt {C B X : Type} {m n : ℕ} (f : Paths (Paths C m × B) n → Fin n → X) : Prop :=
  ∀ (x z : Paths (Paths C m × B) n) (w : Fin n), f (spliceAt w.val 0 x z) w = f x w

/-- **The fixed words are the inventory's right words**: on every accepted path, the play's fixed words over the period
(`n · fixed`) are at most the credited words the inventory's outputs get right, summed over the windows (`right x w`).
The bridge from the play's anatomy, which `Play` leaves abstract, to what the prover's machines output: the words the
prover commits are its programs' outputs. A named premise until the play's fixed words are defined from the committed
values (the adaptive model), as `UnitFixed` is. -/
def FixedAreRight {Ω K T L U : Type} {n : ℕ} (Pl : Play Ω n K T L U) (right : Paths Ω n → Fin n → ℝ) : Prop :=
  ∀ x, Pl.accepts x → n * (Pl.run x).fixed ≤ ∑ w, right x w

/-- **The served PoUW protocol's work** (H1, J10-R, `Credit.window`'s verdict), in the remote form. Under
`H1J10RCheckedCreditPeriod`'s premises and setting, let the prover's whole inventory be one machine `G` of 𝓜 at its
device premises, whose window-`w` program `work x w` exchanges no bits outside it and reads the window's salt coin
(`saltCoin`) and advice `state x w` of at most `B` bits, on the registrable instance `inst x w`, all three fixed before
the salt (`FixedBeforeSalt`). With SAK at the served words (`SAKServed`: `p ≥ 0` a credited word, luck `lam`, degenerate
salts `κ_deg`) and the play's fixed words the inventory's right words (`FixedAreRight`), then except with probability
`δ_s + n·(δ_bind + δ_ks) + n·κ_deg(B)`, an accepted period's priced work on the inventory is at least `p` times
`(1 − ε(j))₊` of its accepted prefix's live words, less the windows' luck allowances:
`Σ_w cost_μ(work_w) ≥ p · creditPrefix − Σ_w lam(inst_w)`. Whether it is non-vacuous on 𝓜 is open (module doc). -/
def ServedWork : Prop :=
  ∀ {ι K T L U C Ξ V I W St : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype C] [Fintype Ξ]
    [Nonempty C] [Nonempty Ξ] [Nonempty V] {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (unitOf : Fin nB → ℕ) (Sc : Scheme K) (Cd : Declared ι)
    {n : ℕ} (_hn : 0 < n) (_hN : Cd.units ≤ tileCount σ sT) (hr : 0 < Cd.rounds) (_hM : Cd.rounds = 1)
    (_hKd : 1 ≤ k sT) (ws : CreditWindows (Coin C Ξ Cd.rounds σ sT) n)
    (opened : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → ℕ → ℕ → V) (enc : C → ByteArray)
    (place : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → Fin (tileCount σ sT) → Place)
    (tile : Noise → List (List ℕ) → Fin (tileCount σ sT) → (ℕ → V) → Prop)
    (Wp : Period C (Coin C Ξ Cd.rounds σ sT) (Fin (tileCount σ sT)) Cd.rounds n)
    (Pl : Play (Coin C Ξ Cd.rounds σ sT) n K T L U)
    (folded : Paths (Coin C Ξ Cd.rounds σ sT) n → ℝ) (ρ δs δbind δks : ℝ)
    (G : Pouw.Dither.Accel) (μ : G.Res → ℝ) (Reg : Set I) (hon : I → C → Fin (tileCount σ sT) → W) (p : ℝ) (lam : I → ℝ)
    (κdeg : ℕ → ℝ) (B : ℕ) (inst : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → I)
    (state : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → St)
    (work : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → G.Mc.Alg (St × C) (Fin (tileCount σ sT) → W)),
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
    UnitFixed Pl (tileLayer Cd hr σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
        (j10rAtCoin hr enc place tile))
      (periodMode Cd Wp (tileRun σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
        (j10rAtCoin hr enc place tile)))
      (fun _ t => (tileWords Sc Cd (tileCount σ sT) t : ℝ)) folded →
    G.G1 → G.G2 → G.P1 → G.P2 →
    Pouw.Assumptions.SAKServed G μ Reg hon (fun t => (tileWords Sc Cd (tileCount σ sT) t : ℝ)) p lam κdeg →
    (∃ e : St → Fin (2 ^ B), Function.Injective e) →
    (∀ x w i, G.ext (work x w) i = 0) →
    (∀ x w, inst x w ∈ Reg) → 0 ≤ p →
    FixedBeforeSalt (fun x w => (inst x w, state x w, work x w)) →
    FixedAreRight Pl (fun x w => Pouw.Assumptions.rightWords hon (fun t => (tileWords Sc Cd (tileCount σ sT) t : ℝ)) (inst x w)
      (saltCoin hr x w) (G.Mc.eval (work x w) (state x w, saltCoin hr x w))) →
    prob (fun x => Pl.accepts x ∧
      ∑ w, G.cost μ (work x w) (state x w, saltCoin hr x w) <
        p * creditPrefix Sc Cd (tileCount σ sT) (ws.prefixLen x) (creditLive Sc Cd Wp ws x) 1 1 (j10rRate (k sT) ρ) δs -
          ∑ w, lam (inst x w)) ≤
      δs + n * (δbind + δks) + n * κdeg B

end Properties.Pouw.Served
