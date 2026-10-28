import Definitions.Pouw.Assumptions.PearlC.TTOut
import Definitions.Pouw.PearlC.DeviceFp4

/-!
# TT_OUT for Pearl-C4 on sm_120 (trusted; staged)

`Prop`s taken as hypotheses by the statements that use them, never `axiom`s. No guarantee states them: the Pearl-C4
guarantees state the `W1` forms (`TTOutFp4W1`), and their proofs reduce to the forms here.
-/

namespace Pouw.PearlC.Assumptions

open Finset

variable {Q R S : Type}

/-- **TT_OUT(1/400) at an FP4 record** `d`: `TTOut` at `pearlCProtocolFp4 d sem ρ`, `pearlCDomainFp4 sem`,
`γ₀ = 1/400` and `εPearlC`. It is `TTOutPearlCDev`'s statement on `Fp4Device`'s maps. `PearlCDevice` reads E4M3 codes,
32-MAC atoms and promotion groups, so the FP4 line has its own record. At `G = 0` rev1 is this same statement, because
rev1's first-add term is zero with no promotion. The domain includes D-SK (`Fp4Sem.SaltKeyedB`), so at a `sem` whose
`B̃` is not keyed by A's salt it claims nothing. -/
def TTOutFp4 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : Fp4Device)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4 sem) (1 / 400) εPearlC

/-- **TT_OUT(1/400) per audit tile at an FP4 record** `d`: `TTOutTile` at `pearlCProtocolFp4 d sem ρ` and
`pearlCTilesFp4 d sem ρ`, the `64 × 64` audit grid. -/
def TTOutTileFp4 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : Fp4Device)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocolFp4 d sem ρ) (pearlCTilesFp4 d sem ρ) (pearlCDomainFp4 sem) (1 / 400) εPearlC

/-- **The named assumption `tt-out/fp4-sm120`: TT_OUT(1/400) for Pearl-C4 on sm_120** (`pearl-c-nvfp4-v0`, theory §2
and §5): `TTOutFp4` at `devFp4Sm120`.

- **The cost model.** `CM` is the sm_120 accounting in FP4 units: W1 at the node-2 prices, where one unit is one dense
  NVFP4 MAC with FP32 accumulation and an instruction is priced by its whole fragment. It charges every operation a
  program executes, and it is the same `CostModel` for the honest reference and for every adversary. That the honest
  sm_120 program fits `W_ref` under this `CM` is the kernel's measured claim, not part of this assumption. At W1
  (`TTOutFp4Sm120W1`, `Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1`) `CM` is the term
  `W1.w1Priced (W1Device.fp4Sm120 sha fetch) salt`.
- **The credit.** `creditFp4` less the unit's replayed debit (`Fp4Sem.unitDebit`), whose executable reference is
  `pearl_c4.tile_debit` (`9363e5012`). The debit has D-24's 2:4 row windows (`Fp4Dev.rowWin24`), F1′ (the base
  split's expected saving per row window under the exact noise law, `Fp4Dev.f1Row`, B̃'s at 10 times), the chain's
  firings whose lone skip leaves the word unchanged (`Fp4Skip.fp4PureFlags`), the positions whose A′ and B̃ codes are
  both salt-dead, the forming elements that are salt-dead or needed by no word, the sub-grid blocks
  (`Fp4Dev.subgridFlags4`), F2 (the masked int8 route on flat scale bytes, at the tile's, the rows' and the block
  columns' flatness, `Fp4Dev.int8Saving`, with `Fp4Dev.cL` the catalogue's minimum pinned per shape) and R1 (a row over
  `ρ` on its own F1′ earns nothing). A unit whose debit exceeds `ρ·creditFp4` is rejected. The row rule is D-NF alone
  (`Fp4Dev.RowAdmit4`: `0 ≤ β` and `0x08 ≤ e4m3(β) < 0x80`): D-SS's one-in-64 cap became the per-element charge on
  salt-dead codes, so a row with many screened blocks is charged for them rather than rejected.
- **The domain is the grant's.** `pearlCDomainFp4` bounds the shapes as the reference does (`PearlC4.check`), with
  `n ≥ 4096` in place of its `n ≥ 128`: `0 < m ≤ 2^24`, `4096 ≤ n ≤ 2^18`, both multiples of 64, and `1024 ≤ k ≤ 2^16`
  with `128 ∣ k`. So the assumption says nothing about a unit outside those shapes.
- **The base-split fix** (theory §6.2–6.5).
  F1′, F2 and R1 answer the base split (A′ = A0 + ΔA with A0 near rank one and ΔA 2:4) and the masked int8 route on
  flat scales. The falsifier for this row is an admitted tile, its credited debit at most `1/400`, on which
  an exact program saves more than its debit plus 0.25% in W1. Open (§6.5): gathering across blocks on operands flat on
  both sides, whether the sorted pairing is the best layout inside a block, and adaptive regrouping after the salt.
- **Exactness is per run.** Nothing here assumes the chain is exact. A claim that equates `C̃` with the exact product
  carries a per-word exact-run condition (option (b); `PeelFp4Proofs.ExactRuns4`).
- **The salt-dead predicate judges the E2M1 code.** `Fp4Dev.saltDead4` asks whether the element's code value after the
  noisy block scale and the cast is the same at every draw, not whether the noise atom's word is. A forming skip needs
  the code, and a dominant element's code can be constant while its word is not.
- **ρ_D is proved from `PinnedScales`, which is argued, not proved.** `Fp4Dev.RhoDFp4` holds under
  `CodeProofs.PinnedScales` (`CodeProofs.rhoDFp4At_ninety`, with `CodeProofs.rhoDFp4At_ninety_eq : RhoDFp4At 90 =
  RhoDFp4` by `rfl`). That the reference's `stats` meet `PinnedScales` is argued, not proved. The β and α₀ bounds rest
  on the FP32 error bound of `s5_stats`' `ρ²`, about `(k/256 + 13)·u ≈ 270u ≈ 2^-15.9` at `k = 2^16`, against the
  `2⁻¹⁰` allowance, and on the numerical table of `s5_stats`' error. There is no Lean
  lemma about `s5_stats` until `s5_stats` is pinned.
- **D-SK is a domain rule.** `pearlCDomainFp4 sem` requires `Fp4Sem.SaltKeyedB`: `B̃`'s lines are drawn from
  `seed_B = H_"pearl-c4/v1/seed-B"(salt ‖ root_B)` (SHA-512, `pearl_c4.labelled`), so `B̃` is a function of the salt
  and root_B, and the ticket, U and the debit compare against the `B̃` the verifier forms itself. A variant that fixes
  `B̃` before A's salt is outside D₄ and needs the per-salt split debit.
- **The shared definitions it reads.** The ticket is `Fp4.cTilde` at `Fp4.sm120Nvf4`, U is `Fp4Dev.peelFp4` (the FP4
  chains feeding `PeelExact`'s BF16 peel), the forming is `Fp4Dev.formRow`, and the salt-dead predicate is
  `Fp4Dev.saltDead4`.
- **`sem`.** The noise lines (`Fp4Sem.lines`) and the row scales `alpha` and `beta` become shared definitions as they
  land. `TTOutFp4Sm120` at any other of these is a different assumption, and replacing one with the landed definition
  is a re-grant.
- **What it carries.** As for FP8: joint skips of several firings within a word that no lone flag sees, which no replay
  count bounds. At `G = 0` the whole `k/64`-firing chain is one group, so these are the only joint skips.
- **What falsifies it.** A program of the model that outputs correct ticket and useful words within the cap, on some
  in-domain family, for less than `(1 − γ₀)` of their debited credit. The families to try are `TTOutPearlC`'s:
  cancelling sets of firings, aligned spikes, identity tiles, exact runs, forming that reads the activations alone,
  fused or skipped noisy casts, and degenerate activations. Here they apply to the E2M1 grid, where a block's codes
  saturate at `6·s_b`. -/
def TTOutFp4Sm120 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : Fp4Sem Q R S)
    (ρ : ℚ) : Prop :=
  TTOutFp4 CM devFp4Sm120 sem ρ

/-- **`tt-out/fp4-sm120` per audit tile**: `TTOutTileFp4` at `devFp4Sm120`. It changes `TTOutFp4Sm120` as
`TTOutTilePearlC` changes `TTOutPearlC`: good tiles of any committed unit, credited rows per row, the cap per tile, and a
fixed credit of `(1 − ρ)·creditFp4` times the credited rows' share. -/
def TTOutTileFp4Sm120 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileFp4 CM devFp4Sm120 sem ρ

end Pouw.PearlC.Assumptions
