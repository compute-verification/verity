module

public import Definitions.Pouw.PearlC.DeviceRev1
public import Definitions.Core.Interaction

/-!
# The sampled audit sampled as the run samples: tickets from the records (trusted; definitions only)

`GGapSampled` and `GGapSampledTol` (`Tile.lean`) sample every tile by its share of the layout's `W_ref`, fixed before the
adversary acts. Pearl+'s run samples otherwise (`Run.tickets`, `ticketWeight`): a tile's ticket is its credited cells at
`wref(1, |K|, n)/n`, read off the records the prover commits with the transcript before the coin, so the sample depends on
the transcript. This file states γ for that sample.

- **A declaration** (`Decl`) is what a record declares of a committed unit before the coin: its kept width |K| and the
  rows and columns it credits (real, neither excluded nor voluntary). The prover fixes its declarations with the
  transcript, so the statements take them as any function of the oracle and the salt.
- **The sample's weight** of a tile (`drawWeight`) is a weight `Wd` at its unit's declaration, and nothing on a blank
  unit's tile. Pearl-C's (`pearlCDrawRev1`) is `ticketWeight`'s: the tile's declared credited cells, each at
  `wrefRowRev1 d |K| n / n`, rev1's `W_ref` of one row at the declared kept width with its promotions counted per whole
  group, as `wref(1, |K|, n)` counts them.
- **The tickets** (`ticketsOf`) are the weights scaled to integers by the lcm of their denominators, as `Run.tickets`
  scales them, and **the claimed work** (`claimedDraw`) is their total, the run's `Σ` ticket weight.
- **The probability** is the average over the oracle and the salt (`Claims.Ex`) of each transcript's sample probability
  over its own tickets (`pr`), so that every `(H, s)` weighs the same whatever its number of tickets. A transcript with
  no tickets has no sample, and every event has probability `0` on it.
- **The audit's check of a sampled tile** passes on a good tile whose declaration is consistent with it (`declOK`): the
  declared kept width is within the unit's `k`, and every declared credited row of the tile passes Π's rules.
- **`GGapSampledDrawTol`:** the tolerant audit accepts (every sampled tile passes, is of a blank unit, or is over its
  inert-work bound) while `T` is below `(1 − γ)·(1 − u(X_O))` of the claimed work, with probability at most
  `δ + η(q, N)`, for every table `u` with a miss bound at `δ` (`MissBound`). **`GGapSampledDrawTolThreshold`** is it at
  the deployed table, `u 0 = ε_s` and `u x = 1` otherwise, at `(1 − ε_s)^t`. **`GGapSampledDraw`** is the audit that
  accepts only checked tiles, at `(1 − ε_s)^t`. The deduction is a share of the claimed work, `u(X_O)·W_claimed`, where
  `GGapSampledTol`'s is a share of the layout's.

The bound `q ≤ 2^64` is written as its numeral, so that it is a plain range.
-/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-- What a record declares of a committed unit before the coin (`UnitRecord`): its kept width |K| (`kept`), and the rows
and columns it credits (`rows`, `cols`: real, neither excluded nor voluntary). -/
structure Decl where
  kept : ℕ
  rows : Finset ℕ
  cols : Finset ℕ

variable {Q R S : Type}

/-- The sample's weight of tile `g` of `τ` under the declarations `dec`: `Wd` at its unit's declaration, and nothing on a
blank unit's tile. -/
def drawWeight (TR : TileRules Q R S) (Wd : Layout → ℕ → Decl → ℚ) (U : Workload) (τ : Transcript)
    (dec : ℕ → Decl) (g : ℕ) : ℚ :=
  if (τ ((TR.tiling U.layout).unit g)).isNone then 0 else Wd U.layout g (dec ((TR.tiling U.layout).unit g))

/-- The integer tickets of the weights `v` of tiles `0, …, NT − 1`, as `Run.tickets` scales them: each weight's numerator
times the lcm of their denominators over its own. A nonnegative weight's tickets are the lcm times the weight. -/
def ticketsOf (NT : ℕ) (v : ℕ → ℚ) (g : ℕ) : ℕ :=
  if g < NT then (v g).num.toNat * ((range NT).lcm (fun i => (v i).den) / (v g).den) else 0

/-- The sample's tickets of `τ` under `dec`. -/
def drawTickets (TR : TileRules Q R S) (Wd : Layout → ℕ → Decl → ℚ) (U : Workload) (τ : Transcript)
    (dec : ℕ → Decl) : ℕ → ℕ :=
  ticketsOf (TR.tiling U.layout).NT (drawWeight TR Wd U τ dec)

/-- The work `τ` claims under `dec`: the sample's weights of all its tiles, the run's `Σ` ticket weight. -/
def claimedDraw (TR : TileRules Q R S) (Wd : Layout → ℕ → Decl → ℚ) (U : Workload) (τ : Transcript)
    (dec : ℕ → Decl) : ℝ :=
  ∑ g ∈ range (TR.tiling U.layout).NT, (drawWeight TR Wd U τ dec g : ℝ)

/-- Tile `g`'s declaration is consistent with `τ`: its kept width is within the unit's `k`, and every credited row it
declares in the tile passes Π's rules. The audit checks both of a sampled tile. -/
def declOK (TR : TileRules Q R S) (U : Workload) (τ : Transcript) (dec : ℕ → Decl) (g : ℕ) : Prop :=
  (dec ((TR.tiling U.layout).unit g)).kept ≤ (U.layout.shape ((TR.tiling U.layout).unit g)).k ∧
    ∀ i ∈ (TR.tiling U.layout).rows g, i ∈ (dec ((TR.tiling U.layout).unit g)).rows →
      TR.rowOK (U.layout.shape ((TR.tiling U.layout).unit g)) (actOf τ ((TR.tiling U.layout).unit g)) i

open Classical in
/-- The good tiles of `τ` whose declaration is consistent with it: those a sampled tile's check passes. -/
noncomputable def goodDeclTiles (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S)
    (τ : Transcript) (dec : ℕ → Decl) : Finset ℕ :=
  (goodTiles P TR U H s τ).filter (declOK TR U τ dec)

/-- **The γ-gap property under the tolerant sampled audit, sampled as the run samples.** For every in-domain workload, budget
`T`, oracle bound `q ≤ 2^64`, adversary, declarations `dec` (fixed with the transcript before the coin) and miss-bound
table `u` at `δ` (`MissBound`): averaged over the oracle and the salt, the probability over `t` samples by the transcript's
tickets (`drawTickets`) that the audit accepts (every sampled tile passes, is of a blank unit, or is over its inert-work
bound) while `T` is below `(1 − γ)·(1 − u(X_O))` of the claimed work, `X_O` the samples on tiles over their bound, is
at most `δ + η(q, N)`. -/
def GGapSampledDrawTol [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (Wd : Layout → ℕ → Decl → ℚ) (γ : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 18446744073709551616 → ∀ A : Adversary CM,
    A.Bounded U T q → ∀ dec : (Q → R) → S → ℕ → Decl, ∀ (t : ℕ) (u : ℕ → ℝ) (δ : ℝ), MissBound t u δ →
      Claims.Ex (fun x : (Q → R) × S =>
        pr (fun c : Fin t → Ticket (drawTickets TR Wd U (A.transcript U x.1 x.2) (dec x.1 x.2))
            (TR.tiling U.layout).NT =>
          AuditAcceptsTol (goodDeclTiles P TR U x.1 x.2 (A.transcript U x.1 x.2) (dec x.1 x.2) ∪
              blankTiles TR U (A.transcript U x.1 x.2))
            (overCapTiles TR U x.1 x.2 (A.transcript U x.1 x.2)) c ∧
          T < (1 - γ) * (1 - u (drawsIn (overCapTiles TR U x.1 x.2 (A.transcript U x.1 x.2)) c)) *
            claimedDraw TR Wd U (A.transcript U x.1 x.2) (dec x.1 x.2)))
        ≤ δ + η q U.layout.N

/-- **The gap property under the sampled audit, sampled as the run samples**: `GGapSampledDrawTol`'s audit when it accepts
only checked tiles (every sampled tile passes, or is of a blank unit), with `ε_s ≤ 1` of the claimed work deducted, at
`(1 − ε_s)^t + δ(q, N)`. -/
def GGapSampledDraw [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (Wd : Layout → ℕ → Decl → ℚ) (g : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 18446744073709551616 → ∀ A : Adversary CM,
    A.Bounded U T q → ∀ dec : (Q → R) → S → ℕ → Decl, ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      Claims.Ex (fun x : (Q → R) × S =>
        pr (fun c : Fin t → Ticket (drawTickets TR Wd U (A.transcript U x.1 x.2) (dec x.1 x.2))
            (TR.tiling U.layout).NT =>
          AuditAccepts (goodDeclTiles P TR U x.1 x.2 (A.transcript U x.1 x.2) (dec x.1 x.2) ∪
              blankTiles TR U (A.transcript U x.1 x.2)) c ∧
          T < (1 - g) * (1 - εs) * claimedDraw TR Wd U (A.transcript U x.1 x.2) (dec x.1 x.2)))
        ≤ (1 - εs) ^ t + δ q U.layout.N

/-- **The γ-gap property under the tolerant sampled audit, sampled as the run samples, at the deployed table**:
`GGapSampledDrawTol`'s event at the threshold table `u 0 = ε_s`, `u x = 1` otherwise, at `(1 − ε_s)^t + η(q, N)`. The
deployed verifier's `ε_s` is the least with `(1 − ε_s)^t ≤ δ`. A run whose samples include a tile over its inert-work
bound is accepted and certifies nothing; any other accepted run certifies `(1 − γ)·(1 − ε_s)` of the claimed work. -/
def GGapSampledDrawTolThreshold [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (Wd : Layout → ℕ → Decl → ℚ) (γ : ℝ)
    (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 18446744073709551616 → ∀ A : Adversary CM,
    A.Bounded U T q → ∀ dec : (Q → R) → S → ℕ → Decl, ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      Claims.Ex (fun x : (Q → R) × S =>
        pr (fun c : Fin t → Ticket (drawTickets TR Wd U (A.transcript U x.1 x.2) (dec x.1 x.2))
            (TR.tiling U.layout).NT =>
          AuditAcceptsTol (goodDeclTiles P TR U x.1 x.2 (A.transcript U x.1 x.2) (dec x.1 x.2) ∪
              blankTiles TR U (A.transcript U x.1 x.2))
            (overCapTiles TR U x.1 x.2 (A.transcript U x.1 x.2)) c ∧
          T < (1 - γ) * (1 - (if drawsIn (overCapTiles TR U x.1 x.2 (A.transcript U x.1 x.2)) c = 0 then εs else 1)) *
            claimedDraw TR Wd U (A.transcript U x.1 x.2) (dec x.1 x.2)))
        ≤ (1 - εs) ^ t + η q U.layout.N

/-! ## Pearl-C's sample under rev1 -/

/-- **rev1's `W_ref` of one row of `n` words at kept width `k`**, as the records' tickets count it (`wref(1, |K|, n)`): each
word's promotions per whole group of `32·G` channels (`⌊k/(32·G)⌋`), the first one out, with the chain's part cut at `0`
as `wref`'s natural subtraction cuts it. It is `wrefDevRev1` of one row at `k` when `32·G` divides `k > 0`, and at most
that at any `k` up to such a width. -/
def wrefRowRev1 (d : PearlCDevice) (k n : ℕ) : ℚ :=
  (if d.G = 0 then (n : ℚ) * k
    else max ((n : ℚ) * k + d.prices.add * n * ((k / (32 * d.G) : ℕ) : ℚ) - d.prices.add * n) 0) +
    (k : ℚ) * Params.pi.r + 2 * d.prices.bf16 * Params.pi.r * n + ((d.prices.costs.fs : ℚ) + d.prices.costs.qa) * k

/-- **The sample's weight of a Pearl-C audit tile under rev1**, as the run's `ticketWeight` weighs it: the tile's declared
credited cells (its rows among `dc.rows`, its columns among `dc.cols`), each at `wrefRowRev1 d |K| n / n`. -/
def pearlCDrawRev1 (d : PearlCDevice) (L : Layout) (g : ℕ) (dc : Decl) : ℚ :=
  wrefRowRev1 d dc.kept (L.shape ((auditTiling 64 64 L).unit g)).n *
      (((auditTiling 64 64 L).rows g ∩ dc.rows).card * ((auditTiling 64 64 L).cols g ∩ dc.cols).card : ℚ) /
    (L.shape ((auditTiling 64 64 L).unit g)).n

end Pouw.PearlC
