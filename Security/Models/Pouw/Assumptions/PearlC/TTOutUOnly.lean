import Models.Pouw.Assumptions.PearlC.TTOutRev1

/-!
# TT_OUT over U alone (trusted; the game under U-only binding, and the named `Prop`)

Under the `-h1` hashing format's U-only messages (P1 of `internal/pouw/hashing-sm120-options.md`) the verifier checks a
unit's useful words U and not its ticket words C̃. A unit is then correct when its activations pass, its debit is within
the cap, its weight rows pass and its U words are right (`CorrectU`); `correctSetU` and the γ-gap property `GγU` are
`correctSet` and `Gγ` with that. `TTOutU` is `TTOut` over `correctSetU`, and `TTOutPearlCDevUOnly` is it at rev1's
protocol, so its credit, cap, W_ref, debit and domain are rev1's (the granted ones at `G = 0`).

It asserts **more** than the (C̃, U) row: every unit correct under both bindings is correct under U alone, so its event
contains the (C̃, U) row's (`ttOutPearlCDevRev1_of_uOnly`). It is a new conjecture, not a weakening.
-/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

/-- A unit correct under U-only binding: admissible activations, its replayed debit within the cap, every row of its
weight passing the rules, and every useful word on its `m × n` entries has the reference FP32 value. Its ticket words
are not read. -/
def CorrectU (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (u : ℕ) (c : Commit) : Prop :=
  P.actOK (U.layout.shape u) c.act ∧ P.capOK H s U u c.act ∧
    (∀ j < (U.layout.shape u).n, P.weightOK (U.layout.shape u) (U.weight (U.layout.wid u)) j) ∧
    ∀ i < (U.layout.shape u).m, ∀ j < (U.layout.shape u).n, wordQ (c.useful i j) = P.useful H s U u c.act i j

open Classical in
/-- `K′`: the units of `τ` correct under U-only binding. -/
noncomputable def correctSetU (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) : Finset ℕ :=
  (range U.layout.N).filter fun u => ∃ c, τ u = some c ∧ CorrectU P U H s u c

/-- The adversary wins under U-only binding: the honest work of its U-correct units exceeds `T / (1 − γ)`. -/
def WinsU (P : Protocol Q R S) (U : Workload) (γ T : ℝ) (τ : Transcript) (H : Q → R) (s : S) : Prop :=
  T / (1 - γ) < ∑ u ∈ correctSetU P U H s τ, P.Wref U.layout u

/-- **The γ-gap property under U-only binding** `G_γ^U` with error `η(q, N)`. -/
def GγU [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S => WinsU P U γ T (A.transcript U x.1 x.2) x.1 x.2) ≤ η q U.layout.N

end Pouw.PearlC

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **TT_OUT(γ₀) over U alone**: `TTOut` with `K′ = correctSetU`, the units whose useful words are correct and whose
debit is within the cap, whatever their ticket words. -/
def TTOutU [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - γ₀) < ∑ u ∈ correctSetU P U x.1 x.2 (A.transcript U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.transcript U x.1 x.2) u))
      ≤ ε q U.layout.N

/-- **TT_OUT(1/400) over U alone for Pearl-C at a device record `d`**: `TTOutU` at `pearlCProtocolDevRev1 d sem ρ`,
`pearlCDomainDev d`, `γ₀ = 1/400` and `εPearlC`. It is `TTOutPearlCDevRev1` with the ticket words unchecked.

- **What it adds to the (C̃, U) row.** U is four BF16 k16 steps accumulated into C̃'s word, so U reads C̃ down to the bits
  its first step keeps: a word's U ignores a +1-ulp move of C̃ on about 30% of words and a +4-ulp move on about 5%
  (`pearlc_u_only_skips.py`; bc-3006c44a's `r20260930-063237-5549`: ≤ 1.6% at 16 ulps). A tile is credited only if all
  its U words are right, so a C̃ that is off anywhere beyond that slack fails the tile.
- **The debit carries over unchanged.** No lone atom skip that changes C̃ leaves U unchanged (0 per family at `k = 1,024`,
  promoted and unpromoted; 0 of 4,096 words at `k = 8,192`, `r20260930-063237-5549`). Promotion stays forced: U from the
  unpromoted C̃ is the pinned U on at most 78% of words at `k = 1,024` and 12% at `k = 8,192`, and no tile passes.
- **What it carries beyond the (C̃, U) row.** Cross-atom sets whose effect on C̃ lands inside U's slack, not only sets
  that leave C̃ unchanged. Finding one needs each atom's effect on the word's rounding, which is its products, as for
  the sets the (C̃, U) row carries.
- **The falsifiers** are `TTOutPearlCDev`'s, with correct U words in place of correct C̃ and U words, plus programs
  that compute C̃ only approximately (lower precision, fewer promotions, dropped low-order work) and still reproduce U.
- **The cost model** is `TTOutPearlCDevRev1`'s: a tensor-core instruction is charged its whole fragment shape, never per
  output word. -/
def TTOutPearlCDevUOnly [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : PearlCDevice)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutU CM (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400) εPearlC

end Pouw.PearlC.Assumptions
