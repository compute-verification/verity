import Definitions.Pouw.PearlC.Instance
import Definitions.Pouw.PearlC.Tile
import Definitions.Pouw.PearlC.Device

/-!
# The named assumption TT_OUT (trusted; the cheap-binding lane's file)

`Prop`s taken as hypotheses by the statements that use them, never `axiom`s.
-/

namespace Pouw.PearlC.Assumptions

open Finset

variable {Q R S : Type}

/-- **TT_OUT(γ₀), output tickets force the noisy chain**, for any cost model `CM`, protocol `P`, domain `D`, target
`γ₀` and error `ε(q, N)`: in the game of `Pouw.PearlC.Gγ`, with `K` the units whose ticket words and useful output words
are both correct and whose debit is within the cap, `Pr_{H,s}[ ∑_{u ∈ K} credit(u) > T / (1 − γ₀) ] ≤ ε(q, N)`, where
`credit(u)` is the reference's value on the unit's committed activations and the realized noise. The named instance is
`TTOutPearlC`. -/
def TTOut [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - γ₀) < ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.transcript U x.1 x.2) u))
      ≤ ε q U.layout.N

/-- **The named assumption TT_OUT(1/400) for Pearl-C**: `TTOut` at `pearlCProtocol sem c ρ`, `pearlCDomain Params.pi`,
`γ₀ = 1/400` and `εPearlC`.

**It is falsified** (rating D): the first promotion of every word adds into a +0 total, an exact identity any program
may skip, yet `creditOf` and `wrefOf` price it (`TTOutRev1.lean`). rev1 (`TTOutPearlCDevRev1`, `TTOutPearlCH100Rev1`) is
its repair.

- **The credit is debited per unit.** It is `creditOf` minus the unit's replayed debit (`PearlCSem.unitDebit`): every atom
  of a one-group joint skip, per output word, on the noisy codes under the ticket's schedule (P1's `jointFlags`), and
  every forming element that is salt-dead or needed by no unflagged atom (P3). So the assumption is not about work the
  replay proves skippable.
- **The cap `ρ`.** A unit whose debit exceeds `ρ·creditOf` is rejected (`capOK`, in `Correct`). It enters neither sum.
- **Cross-group joint skips are the second hole, and this assumption carries them.** Sets of atoms from different groups
  whose rounding effects cancel can be jointly skippable while no group-local flag is set (`joint_cross_group`: the chain
  `+1 ×4 | −1 ×4`). No replay count bounds them, and neither `pearlCDomain` nor the cap does. By counting, such sets exist
  for generic data, and a greedy search with the products in hand finds them: P1's census and `pearlc_census.py
  --cross-words`, reported beside γ, not in it. TT_OUT asserts that no program can find and exploit them for more than
  γ₀ of the credit, because finding one needs each atom's effect on the output's rounding, which is its products.

- **The cost model.** `CM` is the H100 accounting (W1 at the measured H100 prices, one unit = one dense FP8 MAC). It
  charges every operation a program executes, at its price, and it is the same `CostModel` for the honest reference
  (`AdmitsRef`) and for every adversary. A tensor-core instruction (`wgmma`, `mma.sync`, of any kind, shape, precision
  or sparsity) is charged its whole fragment shape, however few of its output words or lanes a program uses, never per
  output word; per-word pricing would make per-word joint skip sets count (`repair-rows-rev1-and-v2-cap.md`). The
  unit's input is the model's clean quantizer output (the scaled activations
  and weights as FP32 words); whatever that quantizer computes is outside the unit for both programs, and nothing inside
  the unit is free for either. Excluding glue changes only the unit boundary, never an executed operation's price.
  That the honest H100 program fits `W_ref` under this `CM` is a measured claim, the kernel lane's, not part of this
  assumption.
- **The costs `c`.** `Costs.adopted` when the model adopts Pearl's headroom scale as its quantizer, even with PoUW off;
  `Costs.current` otherwise. The salt-dependent cast `A′ = e4m3(D)` is protocol forming in both.
- **The shared definitions it reads.** U is `PeelExact.peelExactVal`, the pinned order of `pearl_c_u.py` at `76134352`:
  it depends on the individual codes through the fused chains, the Gram matrix and P_B's FMA chain, so two operand
  rows with equal exact products can give different U words. The debit's flags are `Skip.unitFlags`, P1's `jointFlags`
  under G4, whose reference implementation is `pearlc_census.debit` (PR #451, `f6efc6b9`), and `needed` is `Skip.needed`.
- **`sem`.** Its remaining pieces become shared definitions as they land (`Instance.lean`): `noise` (E_A per unit, from
  (salt, A root, B root, unit index); F_A, F_B and E_B per job, from (salt, B root)); `fc`, §5's forming chain; and `alpha`, `beta` and `floor`. P3's salt-dead predicate is the shared `SaltDead.saltDead`.
  `TTOutPearlC` at any other of these is a different assumption; replacing one with the landed definition is a re-grant.
- **What falsifies it.** A program of the model that outputs correct ticket and useful words within the cap, on some
  in-domain family, for less than `(1 − γ₀)` of their debited credit. The families to try:
  - cross-group cancelling sets found without computing their products (the census's greedy set, and P1's outlier
    families);
  - aligned spikes (`aligned-spikes-rR`, Gaussian and ±1 backgrounds) beyond what the debit removes;
  - identity tiles of a frozen chain;
  - exact runs long enough for Strassen;
  - forming that depends on the activations alone;
  - programs that fuse or skip the noisy cast (A′ is not bound verbatim);
  - activations that are zero, constant, sparse, low rank, saturated, grid-aligned, or in the noise span.

  The activation families can't make the chain cheap for this reason. Every committed row that passes the rules
  carries fresh salted noise of scale σ·α ≥ 1 quantized unit on every element, and at most k/64 of its entries lie
  above 8σ. So each product's low bits depend on the salt, and no atom is predictable before the noise is drawn. Zero
  rows fail the noise floor and are rejected. The standing test is the census `benchmarks/pouw/pearlc_census.py`,
  which covers all of these families under `--forming s5`. -/
def TTOutPearlC [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : PearlCSem Q R S)
    (c : Costs) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocol sem c ρ) (pearlCDomain Params.pi) (1 / 400) εPearlC

/-- **TT_OUT(γ₀) per tile**: as `TTOut`, with the tiles of `TR` in place of the units. `K` is the good tiles of the
committed units (`goodTiles`: debit within the cap, and every ticket and useful word of the credited rows correct), and
`Pr_{H,s}[ ∑_{g ∈ K} credit(g) > T / (1 − γ₀) ] ≤ ε(q, N)`. The named instance is `TTOutTilePearlC`. -/
def TTOutTile [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - γ₀) < ∑ g ∈ goodTiles P TR U x.1 x.2 (A.transcript U x.1 x.2),
        TR.credit x.1 x.2 U g (actOf (A.transcript U x.1 x.2) ((TR.tiling U.layout).unit g)))
      ≤ ε q U.layout.N

/-- **The named assumption TT_OUT(1/400) for Pearl-C, per audit tile**: `TTOutTile` at `pearlCProtocol sem c ρ`, its
tiles `pearlCTiles sem c ρ` (the audit's `64 × 64` grid), `pearlCDomain Params.pi`, `γ₀ = 1/400` and `εPearlC`. It is
`TTOutPearlC` with three changes, for a verifier that checks drawn tiles only (the red team's F1). It is falsified as
`TTOutPearlC` is, by the first-promotion skip (rating D), and rev1 per tile (`TTOutTilePearlCDevRev1`) is its repair.

- **Tiles, not units.** It credits the good tiles of any committed unit, including a unit some of whose other tiles are
  wrong or over the cap. So it also speaks about partial units, where `TTOutPearlC` credits only whole correct units.
- **Credited rows, per row.** A tile's rows that pass Π's rules are credited; a row that fails them stays in its unit,
  bound, and earns no credit (the plan's one per-row rule). `TTOutPearlC` credits nothing of a unit with such a row.
- **The cap per tile.** A tile is good only if its own replayed debit on its credited rows (`PearlCSem.tileDebit`: P1's
  flags over their words, and the forming their words don't need) is at most `ρ·creditOf` times their share. On a whole
  unit, the tiles' debits add up to at least the unit's.
- **A fixed credit.** A good tile earns `(1 − ρ)·creditOf` times its credited rows' share, whatever its debit. Within the
  cap that is at most `TTOutPearlC`'s debited credit, so per tile the claim is weaker.

The rest is `TTOutPearlC`'s: the same cost model and charging rule, the same shared definitions and `sem`, the same
error, and the same second hole (cross-group joint skips within a word, carried here). The falsifiers are
`TTOutPearlC`'s, applied to any set of tiles: a program that outputs a set of good tiles' words for less than
`(1 − γ₀)` of their credit. The per-word census (`pearlc_census.py`) covers a tile of any size, since the debit is per
word, and an aligned-spike tile (R = 300, 74.6% of atoms flagged per word) is over the cap and not good. -/
def TTOutTilePearlC [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (c : Costs) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocol sem c ρ) (pearlCTiles sem c ρ) (pearlCDomain Params.pi) (1 / 400) εPearlC

/-- **TT_OUT(1/400) for Pearl-C at a device record** `d`: `TTOut` at `pearlCProtocolDev d sem ρ`,
`pearlCDomainDev d`, `γ₀ = 1/400` and `εPearlC`. At `devH100` it is `TTOutPearlC` at `Costs.adopted`
(`ttOutPearlCDev_h100`). At another record it is another named assumption: the same game, charging rule, error and
falsifiers, on that device's ticket chain, U, flags, salt-dead predicate and prices. `CM` is that device's W1 accounting
(`w1-complete/<device>`), which charges a tensor-core instruction its whole fragment shape, never per output word. -/
def TTOutPearlCDev [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : PearlCDevice)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocolDev d sem ρ) (pearlCDomainDev d) (1 / 400) εPearlC

/-- **TT_OUT(1/400) per audit tile at a device record** `d`: `TTOutTile` at `pearlCProtocolDev d sem ρ` and
`pearlCTilesDev d sem ρ`. At `devH100` it is `TTOutTilePearlC` at `Costs.adopted` (`ttOutTilePearlCDev_h100`). -/
def TTOutTilePearlCDev [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : PearlCDevice)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocolDev d sem ρ) (pearlCTilesDev d sem ρ) (pearlCDomainDev d) (1 / 400) εPearlC

end Pouw.PearlC.Assumptions
