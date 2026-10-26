import Definitions.Pouw.Assumptions.Dither

/-!
# Pooling the fleet's checks: the pointwise check moment (trusted)

The premise that lets the checks of concurrent replicas pool into one bound, at `ln(1/δ)` for the whole fleet
instead of `ln(G/δ)` a replica (CMP-152):

* `checkMoment`: a words program's moment over the check coin alone, at one dither;
* `SAKCheck`: `SAKMatmul` at every dither outside `Deg`, at a price `ζ` a block.

`SecurityProofs.Pool` proves that it pools (`pool_luck`), that at `ζ = 0` it implies `SAKMatmul`
(`sakMatmul_of_sakCheck`), and that `SAKMatmul`'s form, averaged over the block's dither, does not pool
(`not_pool_of_average_moments`, `parity_all_pass`).
-/

namespace Pouw.Dither

open Pouw.RelativeSecurity

open Classical in
/-- **A words program's check moment at a dither** (CMP-152): for a words program `Mw` on advice `s` at the dither `d`,
the average over the check coin alone of `e^{η · (pub − cost_μ)₊}` where the check passes on the words it computes,
`0` elsewhere. `wordsMoment` is its average over the dithers outside `Deg`. -/
noncomputable def checkMoment (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (pub η : ℝ) (a : P.Inst) {St : Type}
    (s : St) (Mw : G.Mc.Alg (St × P.Draw) P.Tr) (d : P.Draw) : ℝ :=
  avg fun c : P.Chk =>
    if P.ok a d c (G.Mc.eval Mw (s, d)) then Real.exp (η * roundShortfall pub (G.cost μ Mw (s, d))) else 0

end Pouw.Dither

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Pouw.Dither

/-- `pouw-sak-check` (proposed; CMP-152, pending Daniel). **SAK(matmul) at every dither**: `SAKMatmul` with the
average over the dither taken out. In the model at its device premises, on every registrable instance `a ∈ Reg`,
every words program that exchanges no bits with outside machines, on any advice `s` of at most `B` bits that reads
nothing of the block's own coins, has check moment at most `e^{ζ a}` at every dither outside a set `Deg` of
probability at most `κdeg B`: `E_c[1_{pass(d, c)} · e^{η (pub − cost_μ)₊}] ≤ e^{ζ a}` for every `d ∉ Deg`.

**Why it's needed.** The fleet's blocks are not one sequence: a block's work can read the dithers of blocks that root
after it, and two concurrent blocks can each read the other's. A moment averaged over the block's own dither, at every
value of the others' (`SAKMatmul`'s form), then bounds each block but not their product
(`SecurityProofs.Pool.not_pool_of_average_moments`): blocks that guess their dither from the others' pass together
with probability 1/2 however many they are (`SecurityProofs.Pool.parity_all_pass`). Conditioning on every dither at
once leaves only the check coins, which the challenger releases one after another after each block's root, so this
form pools (`SecurityProofs.Pool.pool_luck`).

**Its price, `ζ`.** At a fixed dither, a program that skips `j` of the block's `N` words passes a check of one of them
with probability `1 − (j − r)/N`, `r` the skipped words that are right anyway; averaged over the dither `r` is about
`p j` (`p` the free-guess rate, `SAKMatmul`'s `1 − p`). `ζ a` pays for `r` above its mean: at selection probability
`π` and `L = ln(1/κ)` it is about `π (√(2 p L / N) + 2L / (3N))`, and the dithers past that margin are in `Deg`
(probability `e^{-L}`). A dither family whose free guesses come all together (every word right on the same dithers)
has no such margin, and the premise is false for it.

At `ζ = 0` it implies `SAKMatmul` (`SecurityProofs.Pool.sakMatmul_of_sakCheck`). -/
def SAKCheck (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (Reg : Set P.Inst) (pub η ζ : P.Inst → ℝ)
    (κdeg : ℕ → ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ a ∈ Reg, ∀ (B : ℕ) (St : Type), (∃ e : St → Fin (2 ^ B), Function.Injective e) →
    ∀ (s : St) (Mw : G.Mc.Alg (St × P.Draw) P.Tr), (∀ i, G.ext Mw i = 0) →
      ∃ Deg : P.Draw → Prop, prob Deg ≤ κdeg B ∧
        ∀ d, ¬ Deg d → checkMoment G P μ (pub a) (η a) a s Mw d ≤ Real.exp (ζ a)

end Pouw.Assumptions
