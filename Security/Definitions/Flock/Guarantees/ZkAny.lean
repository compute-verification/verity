module

public import Definitions.Core.Game
public import Definitions.Flock.ZK.AnyValues

/-!
# Zero knowledge at any values

What the ZK server releases is simulated for any worker and any committed values, valid or not: the simulator reads the
public inputs, the auditor's coins and samples, and the protocol's public outcome (each verdict and where a session
stopped), and nothing of the values. An outcome that correlates with the values costs a factor of the number of outcomes
`|St|`, `log₂ |St|` bits.

* `ZkAnyOutcome`: the generic form. A release that is, at each outcome `s`, a branch `B s` simulated by `S s` within
  `ε s` is simulated within `ε s` jointly with that outcome, within `Σ_s (Pr[S s ∈ E] + ε s)` overall, and so within
  `|St|` times the simulator that draws the outcome uniformly, plus `Σ_s ε s`.
* `ZkAnyGate`: the check-then-release instance, two outcomes: where the server's check passes, the view is a derived view
  `d` that a simulator matches at any values; where it fails, only the cut leaves.
* `TableReleaseAny`: `ZkAnyGate` for one table of C-Flock's ZK server, exactly: on any witness `w` carrying the public
  region data and a dummy `w₀`, with no premise that `w`'s values pass the inner constraint, the release behind a check
  that passes only where they do (`Table.Gated`) is bounded by `S_shvzk(x, e)` on `w₀` plus the dummy's pre-final view.
-/

@[expose] public section

namespace Flock.Guarantees

open Verity.Game
open scoped ENNReal

-- As in `SHVZK`: a table's randomness is a product of up to fifteen factors.
set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Zero knowledge with a public outcome, generic**: a release `rel` that is, at each outcome `s = out ω`, the branch
`B s ω`, where each branch is simulated by `S s` within `ε s` (one-sided, for every event), is simulated within `ε s`
jointly with each outcome, within `Σ_s (Pr[S s ∈ E] + ε s)` overall, and within `|St|` times the simulator that draws
the outcome uniformly, plus `Σ_s ε s`. -/
def ZkAnyOutcome : Prop :=
  ∀ {Ω St Ωs V : Type} [Fintype Ω] [Fintype St] [Fintype Ωs] (rel : Ω → V) (out : Ω → St) (B : St → Ω → V)
    (S : St → Ωs → V) (ε : St → ℝ≥0∞),
    (∀ ω, rel ω = B (out ω) ω) →
    (∀ s (E : V → Prop), prCoin (fun ω => E (B s ω)) ≤ prCoin (fun σ => E (S s σ)) + ε s) →
    ∀ E : V → Prop,
      (∀ s, prCoin (fun ω => out ω = s ∧ E (rel ω)) ≤ prCoin (fun σ => E (S s σ)) + ε s) ∧
      prCoin (fun ω => E (rel ω)) ≤ ∑ s, (prCoin (fun σ => E (S s σ)) + ε s) ∧
      prCoin (fun ω => E (rel ω)) ≤
        (Fintype.card St : ℝ≥0∞) * prCoin (fun p : St × Ωs => E (S p.1 p.2)) + ∑ s, ε s

/-- **Check-then-release, generic**: for every witness `w` with the public property `Pub` (whatever its values), where
the cut `cut w` is simulated by `simCut` within `εc`, the derived view `d w` by `simFull` within `εf`, and the view is
the derived view wherever the check `go w` passes, the release (the view where the check passes, the cut where it
fails) is simulated on each branch, and overall within the sum of both simulators plus `εf + εc`. -/
def ZkAnyGate : Prop :=
  ∀ {Wt Ω Ωf Ωc V P : Type} [Fintype Ω] [Fintype Ωf] [Fintype Ωc]
    (Pub : Wt → Prop) (view d : Wt → Ω → V) (cut : Wt → Ω → P) (go : Wt → Ω → Bool)
    (simFull : Ωf → V) (simCut : Ωc → P) (εf εc : ℝ≥0∞),
    (∀ w, Pub w → ∀ E : P → Prop, prCoin (fun ω => E (cut w ω)) ≤ prCoin (fun σ => E (simCut σ)) + εc) →
    (∀ w, Pub w → ∀ E : V → Prop, prCoin (fun ω => E (d w ω)) ≤ prCoin (fun σ => E (simFull σ)) + εf) →
    (∀ w, Pub w → ∀ ω, go w ω = true → view w ω = d w ω) →
    ∀ w, Pub w → ∀ E : V ⊕ P → Prop,
      prCoin (fun ω => go w ω = true ∧ E (.inl (view w ω))) ≤ prCoin (fun σ => E (.inl (simFull σ))) + εf ∧
      prCoin (fun ω => go w ω = false ∧ E (.inr (cut w ω))) ≤ prCoin (fun σ => E (.inr (simCut σ))) + εc ∧
      prCoin (fun ω => E (if go w ω then .inl (view w ω) else .inr (cut w ω))) ≤
        prCoin (fun σ => E (.inl (simFull σ))) + prCoin (fun σ => E (.inr (simCut σ))) + (εf + εc)

/-- **One table's release behind the ZK server's check, at any values**: at a non-degenerate coin vector (`Rank`,
`PadOnto`, `W^r` bijective, `PadsOnto`, `β^r ≠ 0`), for a witness `w` and a dummy `w₀` carrying the public region data,
and a check `go` on the server's randomness that passes only where `w`'s values pass the inner constraint at the drawn
pads (`Table.Gated`), and with no premise that they pass it anywhere, the release is the view where `go` passes and the
pre-final view where it doesn't (`Table.release`), and each branch is bounded by its simulator: `S_shvzk(x, e)` on `w₀`
and the dummy's pre-final view. -/
def TableReleaseAny : Prop :=
  ∀ {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
    [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
    [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL),
    T.Rank → T.PadOnto → (∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) → T.PadsOnto →
    (∀ r, T.β r ≠ 0) →
    ∀ (go : T.Rand → Bool) (w w₀ : W), T.sReg w = T.pubReg → T.sReg w₀ = T.pubReg → T.Gated go w →
    ∀ E : T.View ⊕ T.PreView → Prop,
      prCoin (fun ω => go ω = true ∧ E (.inl (T.view w ω))) ≤ prCoin (fun σ => E (.inl (T.sim w₀ σ))) ∧
      prCoin (fun ω => go ω = false ∧ E (.inr (T.preView w ω))) ≤ prCoin (fun ω => E (.inr (T.preView w₀ ω))) ∧
      prCoin (fun ω => E (T.release go w ω)) ≤
        prCoin (fun σ => E (.inl (T.sim w₀ σ))) + prCoin (fun ω => E (.inr (T.preView w₀ ω)))

end Flock.Guarantees
