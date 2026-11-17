module

public import Definitions.Pouw.PearlC.DeviceRev1
public import Definitions.Core.Interaction

/-! The sampled audit as the run samples it, by tickets read from the records the prover commits with the trace before the coin. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

structure Decl where
  kept : ℕ
  rows : Finset ℕ
  cols : Finset ℕ

variable {Q R S : Type}

def drawWeight (TR : TileRules Q R S) (Wd : Layout → ℕ → Decl → ℚ) (U : Workload) (τ : Trace)
    (dec : ℕ → Decl) (g : ℕ) : ℚ :=
  if (τ ((TR.tiling U.layout).unit g)).isNone then 0 else Wd U.layout g (dec ((TR.tiling U.layout).unit g))

def ticketsOf (NT : ℕ) (v : ℕ → ℚ) (g : ℕ) : ℕ :=
  if g < NT then (v g).num.toNat * ((range NT).lcm (fun i => (v i).den) / (v g).den) else 0

def drawTickets (TR : TileRules Q R S) (Wd : Layout → ℕ → Decl → ℚ) (U : Workload) (τ : Trace)
    (dec : ℕ → Decl) : ℕ → ℕ :=
  ticketsOf (TR.tiling U.layout).NT (drawWeight TR Wd U τ dec)

def claimedDraw (TR : TileRules Q R S) (Wd : Layout → ℕ → Decl → ℚ) (U : Workload) (τ : Trace)
    (dec : ℕ → Decl) : ℝ :=
  ∑ g ∈ range (TR.tiling U.layout).NT, (drawWeight TR Wd U τ dec g : ℝ)

/-- The audit checks both of a sampled tile. -/
def declOK (TR : TileRules Q R S) (U : Workload) (τ : Trace) (dec : ℕ → Decl) (g : ℕ) : Prop :=
  (dec ((TR.tiling U.layout).unit g)).kept ≤ (U.layout.shape ((TR.tiling U.layout).unit g)).k ∧
    ∀ i ∈ (TR.tiling U.layout).rows g, i ∈ (dec ((TR.tiling U.layout).unit g)).rows →
      TR.rowOK (U.layout.shape ((TR.tiling U.layout).unit g)) (actOf τ ((TR.tiling U.layout).unit g)) i

open Classical in
noncomputable def goodDeclTiles (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S)
    (τ : Trace) (dec : ℕ → Decl) : Finset ℕ :=
  (goodTiles P TR U H s τ).filter (declOK TR U τ dec)

def GGapSampledDrawTol [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (Wd : Layout → ℕ → Decl → ℚ) (γ : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 18446744073709551616 → ∀ A : Adversary CM,
    A.Bounded U T q → ∀ dec : (Q → R) → S → ℕ → Decl, ∀ (t : ℕ) (u : ℕ → ℝ) (δ : ℝ), MissBound t u δ →
      Claims.Ex (fun x : (Q → R) × S =>
        pr (fun c : Fin t → Ticket (drawTickets TR Wd U (A.trace U x.1 x.2) (dec x.1 x.2))
            (TR.tiling U.layout).NT =>
          AuditAcceptsTol (goodDeclTiles P TR U x.1 x.2 (A.trace U x.1 x.2) (dec x.1 x.2) ∪
              blankTiles TR U (A.trace U x.1 x.2))
            (overInertBoundTiles TR U x.1 x.2 (A.trace U x.1 x.2)) c ∧
          T < (1 - γ) * (1 - u (drawsIn (overInertBoundTiles TR U x.1 x.2 (A.trace U x.1 x.2)) c)) *
            claimedDraw TR Wd U (A.trace U x.1 x.2) (dec x.1 x.2)))
        ≤ δ + η q U.layout.N

def GGapSampledDraw [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (Wd : Layout → ℕ → Decl → ℚ) (g : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 18446744073709551616 → ∀ A : Adversary CM,
    A.Bounded U T q → ∀ dec : (Q → R) → S → ℕ → Decl, ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      Claims.Ex (fun x : (Q → R) × S =>
        pr (fun c : Fin t → Ticket (drawTickets TR Wd U (A.trace U x.1 x.2) (dec x.1 x.2))
            (TR.tiling U.layout).NT =>
          AuditAccepts (goodDeclTiles P TR U x.1 x.2 (A.trace U x.1 x.2) (dec x.1 x.2) ∪
              blankTiles TR U (A.trace U x.1 x.2)) c ∧
          T < (1 - g) * (1 - εs) * claimedDraw TR Wd U (A.trace U x.1 x.2) (dec x.1 x.2)))
        ≤ (1 - εs) ^ t + δ q U.layout.N

/-- The deployed verifier's `ε_s` is the least with `(1 − ε_s)^t ≤ δ`. A run whose samples include a tile over its
inert-work bound is accepted and certifies nothing; any other accepted run certifies `(1 − γ)·(1 − ε_s)` of the claimed work. -/
def GGapSampledDrawTolThreshold [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (Wd : Layout → ℕ → Decl → ℚ) (γ : ℝ)
    (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 18446744073709551616 → ∀ A : Adversary CM,
    A.Bounded U T q → ∀ dec : (Q → R) → S → ℕ → Decl, ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      Claims.Ex (fun x : (Q → R) × S =>
        pr (fun c : Fin t → Ticket (drawTickets TR Wd U (A.trace U x.1 x.2) (dec x.1 x.2))
            (TR.tiling U.layout).NT =>
          AuditAcceptsTol (goodDeclTiles P TR U x.1 x.2 (A.trace U x.1 x.2) (dec x.1 x.2) ∪
              blankTiles TR U (A.trace U x.1 x.2))
            (overInertBoundTiles TR U x.1 x.2 (A.trace U x.1 x.2)) c ∧
          T < (1 - γ) * (1 - (if drawsIn (overInertBoundTiles TR U x.1 x.2 (A.trace U x.1 x.2)) c = 0 then εs else 1)) *
            claimedDraw TR Wd U (A.trace U x.1 x.2) (dec x.1 x.2)))
        ≤ (1 - εs) ^ t + η q U.layout.N

/-- Promotions count per whole group of `32·G` channels: it is `wrefDevRev1` of one row at `k` when `32·G` divides `k > 0`,
and at most that at any `k` up to such a width. -/
def wrefRowRev1 (d : PearlCDevice) (k n : ℕ) : ℚ :=
  (if d.G = 0 then (n : ℚ) * k
    else max ((n : ℚ) * k + d.prices.add * n * ((k / (32 * d.G) : ℕ) : ℚ) - d.prices.add * n) 0) +
    (k : ℚ) * Params.pi.r + 2 * d.prices.bf16 * Params.pi.r * n + ((d.prices.costs.fs : ℚ) + d.prices.costs.qa) * k

def pearlCDrawRev1 (d : PearlCDevice) (L : Layout) (g : ℕ) (dc : Decl) : ℚ :=
  wrefRowRev1 d dc.kept (L.shape ((auditTiling 64 64 L).unit g)).n *
      (((auditTiling 64 64 L).rows g ∩ dc.rows).card * ((auditTiling 64 64 L).cols g ∩ dc.cols).card : ℚ) /
    (L.shape ((auditTiling 64 64 L).unit g)).n

end Pouw.PearlC
