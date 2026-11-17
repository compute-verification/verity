module

public import Definitions.Pouw.Dither

/-! Symmetric algorithmic knowledge at the served words, the work premise of `Properties.Pouw.Served.ServedWork`; whether it holds on 𝓜 is open. -/

@[expose] public section

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
