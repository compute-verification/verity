import Definitions.Pouw.Dither

/-!
# SAK at the served words: the work premise of `Properties.Pouw.Served.ServedWork` (trusted)

`Pouw.SakLb.SAKResidual` (`Proofs.Pouw.SakLb`) states symmetric algorithmic knowledge for exact matrix products as one
pointwise premise on the no-fold round's cells: outside a set of the round's dithers of probability at most `κdeg B`, a
program that exchanges no bits and reads at most `B` bits of advice costs at least the fraction of the words it gets
right times `pub`, less a luck allowance `lam`. It is a proof-only definition over `noFoldRound`'s types.

`SAKServed` is the same claim at the served protocol's types (`PouwCheck.Period`, H1's design): a window's words are
its tiles' (`Fin Np → W`), the dither is the window's round-0 coin (its salt, `Properties.Pouw.Period.SaltIsLiveCoin`),
and a tile counts as right only when all of its words are the honest ones (`rightWords`), each weighted by its
credited words `wt t`. At `pub = p · Σ_t wt t` the right fraction by tiles is at most the right fraction by words, so
`SakLb.SAKResidual` at the same `μ`, `lam` and `κdeg`, on a round whose cells are the tiles' credited words, gives
`SAKServed` pointwise: it asks no more of 𝓜.

**Non-vacuity is open.** 𝓜 as `Definitions.Core.Accel` writes it prices register reductions: a register word that an
atomic or a reduction writes costs an `alu` lane, L2 writes are priced, and an MMA pays at least one product an output.
The four Russians' method then pays at least an `alu` lane a MAC, but `SAKMatmul` as written still fails at a dual
price that leaves those pipes cheap, and `SAKResidual`, which would carry them, is not adopted. Whether `SAKServed`
holds on 𝓜 is open.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Pouw.Dither

open Classical in
/-- **The credited words a window's output gets right**: the words `wt t` of each tile `t` whose output `T t` is the
honest one `hon a d t` of the instance `a` at the dither `d`, and nothing for a tile with any word wrong. -/
noncomputable def rightWords {I D W : Type} {Np : ℕ} (hon : I → D → Fin Np → W) (wt : Fin Np → ℝ) (a : I) (d : D)
    (T : Fin Np → W) : ℝ :=
  ∑ t : Fin Np, if T t = hon a d t then wt t else 0

/-- `pouw-sak-served` (proposed; `SakLb.SAKResidual` at the served words). **SAK at the served words**: in
𝓜 at its device premises, on every registrable instance `a ∈ Reg`, every program that exchanges no bits with outside
machines, on any advice `s` of at most `B` bits fixed before the window's salt `d`, costs at least `p` a credited word
it gets right, less the luck allowance `lam a`, at every salt outside a set `Deg` of probability at most `κdeg B`:
`p · rightWords − lam ≤ cost_μ`.

`hon a d` is the window's honest output at instance `a` under salt `d`: each tile's words of J10-R's noised product at
the noise and fold keys its place reads from `d` (`J10R.unitNoise`, `J10R.unitKeys`). `p` is the public best's priced
cost a credited word at the period's dual prices `μ`, one price for the tile stratum (one `K` template). `Reg` is every
instance the protocol lets a prover register, so a prover that registers its cheapest instance gets that instance's
words at `p` each, no less. The advice is anything computed before the salt within (G1)'s memory, as `SAKMatmul`'s
(the un-noised products, the weights' preprocessing), and the degenerate salts are those on which J10-R's noise leaves
the product's structure.

It is falsified by any program on the card, with any advice, that gets a share of a window's tiles right on most
salts at a measured priced cost below `p` a credited word of them, less `lam`. Whether it holds of 𝓜 as written is
open (module doc). -/
def SAKServed (G : Accel) (μ : G.Res → ℝ) {I D W : Type} [Fintype D] {Np : ℕ} (Reg : Set I)
    (hon : I → D → Fin Np → W) (wt : Fin Np → ℝ) (p : ℝ) (lam : I → ℝ) (κdeg : ℕ → ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ a ∈ Reg, ∀ (B : ℕ) (St : Type), (∃ e : St → Fin (2 ^ B), Function.Injective e) →
    ∀ (s : St) (Mw : G.Mc.Alg (St × D) (Fin Np → W)), (∀ i, G.ext Mw i = 0) →
      ∃ Deg : D → Prop, prob Deg ≤ κdeg B ∧
        ∀ d, ¬ Deg d → p * rightWords hon wt a d (G.Mc.eval Mw (s, d)) - lam a ≤ G.cost μ Mw (s, d)

end Pouw.Assumptions
