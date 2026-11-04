module

public import Definitions.Core.Game
public import Definitions.Claims.Hm96
public import Definitions.Flock.ZK.AnyValues
public import Definitions.Flock.ZK.RealRelease

/-!
# Zero knowledge at any values

What the ZK server releases is simulated for any worker and any committed values, valid or not: the simulator reads the
public inputs, the auditor's coins and samples, and the protocol's public outcome (each verdict and where a session
stopped), and nothing of the values. An outcome that correlates with the values costs a factor of the number of outcomes
`|St|`, `log₂ |St|` bits.

* `ZkAnyOutcome`: the generic form. A release that is, at each outcome `s`, a branch `B s` simulated by `S s` within
  `ε s` is simulated within `ε s` jointly with that outcome, within `Σ_s (Pr[S s ∈ E] + ε s)` overall, and so within
  `|St|` times the simulator that draws the outcome uniformly, plus `Σ_s ε s`.
* `ZkAnyGate`: the check-then-release instance, two outcomes: where the server's check passes, the view is a derived
  view `d` that a simulator matches; where it fails, only the cut leaves. Nothing reads the values beyond these
  premises, which an instance proves at every witness.
* `TableReleaseAny`: `ZkAnyGate` for one table of C-Flock's ZK server, exactly: on any witness `w` carrying the public
  region data and a dummy `w₀`, with no premise that `w`'s values pass the inner constraint, the release behind a check
  that passes only where they do (`Table.Gated`) is bounded by `S_shvzk(x, e)` on `w₀` plus the dummy's pre-final view.
* `SessionReleaseAny`: `ZkAnyOutcome` for a session of `J` tables, the outcome being where it stopped: the tables whose
  files left show their views, each behind its check (`Session.Gated`), and the others their pre-final views.
* `TableReleaseHm96`, `SessionReleaseHm96`: the same releases with every leaf a real hm96 leaf on both branches, within
  `2·δ₁` a leaf left unopened of simulators with real leaves; the session's at witnesses fixed before the server's draws
  (`Session.WitnessesFirst`), per outcome in `ZKOut`'s form.
-/

@[expose] public section

namespace Flock.Guarantees

open Verity.Game FlockSoundness.ZK
open scoped ENNReal

-- As in `SHVZK`: a table's randomness is a product of up to fifteen factors.
set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Zero knowledge with a public outcome, generic**: a release `rel` that is, at each outcome `s = out ω`, the branch
`B s ω` (`Splits`), where each branch is simulated by `S s` within `ε s` (`Within`), is simulated within `ε s` jointly
with each outcome, within `Σ_s (Pr[S s ∈ E] + ε s)` overall, and within `|St|` times the simulator that draws the
outcome uniformly, plus `Σ_s ε s`. -/
def ZkAnyOutcome : Prop :=
  ∀ {Ω St Ωs V : Type} [Fintype Ω] [Fintype St] [Fintype Ωs] (rel : Ω → V) (out : Ω → St) (B : St → Ω → V)
    (S : St → Ωs → V) (ε : St → ℝ≥0∞),
    Splits rel out B → (∀ s, Within (B s) (S s) (ε s)) →
    ∀ E : V → Prop,
      (∀ s, prCoin (fun ω => out ω = s ∧ E (rel ω)) ≤ prCoin (fun σ => E (S s σ)) + ε s) ∧
      prCoin (fun ω => E (rel ω)) ≤ ∑ s, (prCoin (fun σ => E (S s σ)) + ε s) ∧
      prCoin (fun ω => E (rel ω)) ≤
        (Fintype.card St : ℝ≥0∞) * prCoin (fun p : St × Ωs => E (S p.1 p.2)) + ∑ s, ε s

/-- **Check-then-release, generic**: where the cut `cut` is simulated by `simCut` within `εc`, the derived view `d` by
`simFull` within `εf`, and the view is the derived view wherever the check `go` passes (`AgreesWhere`), the release (the
view where the check passes, the cut where it fails) is simulated on each branch, and overall within the sum of both
simulators plus `εf + εc`. -/
def ZkAnyGate : Prop :=
  ∀ {Ω Ωf Ωc V P : Type} [Fintype Ω] [Fintype Ωf] [Fintype Ωc]
    (view d : Ω → V) (cut : Ω → P) (go : Ω → Bool) (simFull : Ωf → V) (simCut : Ωc → P) (εf εc : ℝ≥0∞),
    Within cut simCut εc → Within d simFull εf → AgreesWhere go view d →
    ∀ E : V ⊕ P → Prop,
      prCoin (fun ω => go ω = true ∧ E (.inl (view ω))) ≤ prCoin (fun σ => E (.inl (simFull σ))) + εf ∧
      prCoin (fun ω => go ω = false ∧ E (.inr (cut ω))) ≤ prCoin (fun σ => E (.inr (simCut σ))) + εc ∧
      prCoin (fun ω => E (if go ω then .inl (view ω) else .inr (cut ω))) ≤
        prCoin (fun σ => E (.inl (simFull σ))) + prCoin (fun σ => E (.inr (simCut σ))) + (εf + εc)

/-- **One table's release behind the ZK server's check, at any values**: at coins non-degenerate for the table
(`Table.Simulable`), for a witness `w` and a dummy `w₀` carrying the public region data (`Table.RegPublic`), and a check
`go` on the server's randomness that passes only where `w`'s values pass the inner constraint at the drawn pads
(`Table.Gated`), and with no premise that they pass it anywhere, the release is the view where `go` passes and the
pre-final view where it doesn't (`Table.release`), and each branch is bounded by its simulator: `S_shvzk(x, e)` on `w₀`
and the dummy's pre-final view. -/
def TableReleaseAny : Prop :=
  ∀ {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
    [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
    [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL)
    (go : T.Rand → Bool) (w w₀ : W),
    T.Simulable → T.RegPublic w → T.RegPublic w₀ → T.Gated go w →
    ∀ E : T.View ⊕ T.PreView → Prop,
      prCoin (fun ω => go ω = true ∧ E (.inl (T.view w ω))) ≤ prCoin (fun σ => E (.inl (T.sim w₀ σ))) ∧
      prCoin (fun ω => go ω = false ∧ E (.inr (T.preView w ω))) ≤ prCoin (fun ω => E (.inr (T.preView w₀ ω))) ∧
      prCoin (fun ω => E (T.release go w ω)) ≤
        prCoin (fun σ => E (.inl (T.sim w₀ σ))) + prCoin (fun ω => E (.inr (T.preView w₀ ω)))

/-- **A session's release behind the ZK server's checks, at any values**: at coins non-degenerate for each of its `J`
tables, for witnesses `w` and dummies `w₀` carrying the public region data, an outcome `out ω` (where the session
stopped, a function of the server's randomness) and the tables whose files leave at each outcome (`full`), where a
table's file leaves only if its values pass the inner constraint at its drawn pads (`Session.Gated`), and with no
premise that they pass it anywhere, the session's release (`Session.releaseAt`) is bounded at each outcome, jointly with
it, by the simulator at that outcome (`Session.simAt`), overall by the sum over outcomes, and so by `|St|` times the
simulator that draws the outcome uniformly. -/
def SessionReleaseAny : Prop :=
  ∀ {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
    [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
    [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] {J : ℕ}
    (T : Fin J → FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) {St : Type} [Fintype St]
    (out : (∀ j, (T j).Rand) → St) (full : St → Fin J → Bool) (w w₀ : Fin J → W),
    (∀ j, (T j).Simulable) → (∀ j, (T j).RegPublic (w j)) → (∀ j, (T j).RegPublic (w₀ j)) →
    FlockSoundness.ZK.Session.Gated T out full w →
    ∀ E : (∀ j, (T j).View ⊕ (T j).PreView) → Prop,
      (∀ s, prCoin (fun ω => out ω = s ∧ E (FlockSoundness.ZK.Session.releaseAt T full w (out ω) ω)) ≤
        prCoin (fun σ => E (FlockSoundness.ZK.Session.simAt T full w₀ s σ))) ∧
      prCoin (fun ω => E (FlockSoundness.ZK.Session.releaseAt T full w (out ω) ω)) ≤
        ∑ s, prCoin (fun σ => E (FlockSoundness.ZK.Session.simAt T full w₀ s σ)) ∧
      prCoin (fun ω => E (FlockSoundness.ZK.Session.releaseAt T full w (out ω) ω)) ≤
        (Fintype.card St : ℝ≥0∞) *
          prCoin (fun p : St × (∀ j, (T j).SimRand × (T j).Rand) =>
            E (FlockSoundness.ZK.Session.simAt T full w₀ p.1 p.2))

/-- **One table's release behind the ZK server's check, at any values, with real hm96 leaves**: `TableReleaseAny` with
every leaf real. Where the check passes, the view's hidden leaves are hm96 leaves under fresh salts (`Hm96.viewR`); where
it fails, the same run's pre-final messages leave, every leaf of them an hm96 leaf, `τ`'s commitment included
(`Hm96.preR`, `Hm96.TauLeaf`). From hm96 hiding at `δ₁` a leaf, each branch is within `2·δ₁` a leaf it leaves unopened of
its simulator with real leaves: `S_shvzk(x, e)` on `w₀` within `2·|Hid|·δ₁`, and `w₀`'s pre-final messages within
`2·|PreLeaves|·δ₁`. No premise reads the values. -/
def TableReleaseHm96 : Prop :=
  ∀ {F K W Salt B D Hi VL C : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
    [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]
    [AddCommGroup Salt] [AddCommGroup B] [AddCommGroup Hi]
    (T : FlockSoundness.ZK.Table F K W Salt (B × Salt) D Hi VL) (M : T.Hm96 C) (digT : F → B) {δ₁ : ℝ≥0∞}
    (go : T.Rand → Bool) (w w₀ : W),
    Claims.Hm96Hiding M.My M.cs δ₁ → M.TauLeaf digT → T.Simulable → T.RegPublic w → T.RegPublic w₀ →
    T.Gated go w →
    ∀ E : T.View ⊕ T.PreView → Prop,
      prCoin (fun p : T.Rand × (T.Hid → Salt) => go p.1 = true ∧ E (.inl (M.viewR w p.1 p.2))) ≤
          prCoin (fun σ : T.SimRand × (T.Hid → Salt) => E (.inl (M.simR w₀ σ.1 σ.2))) +
            2 * Fintype.card T.Hid * δ₁ ∧
        prCoin (fun p : T.Rand × (T.Hid → Salt) => go p.1 = false ∧ E (.inr (M.preR w p.1 p.2))) ≤
          prCoin (fun p : T.Rand × (T.Hid → Salt) => E (.inr (M.preR w₀ p.1 p.2))) +
            2 * Fintype.card T.PreLeaves * δ₁ ∧
        prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.releaseR go w p.1 p.2)) ≤
          prCoin (fun σ : T.SimRand × (T.Hid → Salt) => E (.inl (M.simR w₀ σ.1 σ.2))) +
            prCoin (fun p : T.Rand × (T.Hid → Salt) => E (.inr (M.preR w₀ p.1 p.2))) +
              (2 * Fintype.card T.Hid * δ₁ + 2 * Fintype.card T.PreLeaves * δ₁)

/-- **A session's release behind the ZK server's checks, at any values, with real hm96 leaves**: `SessionReleaseAny`
with every leaf real, at witnesses fixed before the server's draws for the session (`Session.WitnessesFirst`: as a
function `wit` of those draws, they read none of them). At each outcome `s`, jointly with it, the release is bounded by
the simulator given `s` (`Session.simR`, which reads the dummies `w₀` and not the witnesses) plus `2·δ₁` for each leaf
the release at `s` leaves unopened (`Session.unopened`), `ZKOut`'s per-outcome form; overall by the sum over outcomes,
and by `|St|` times the simulator that draws the outcome uniformly plus the outcomes' errors, `ZKOutLeak`'s form. No
premise reads the values. -/
def SessionReleaseHm96 : Prop :=
  ∀ {F K W Salt B D Hi VL C : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
    [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]
    [AddCommGroup Salt] [AddCommGroup B] [AddCommGroup Hi] {J : ℕ}
    (T : Fin J → FlockSoundness.ZK.Table F K W Salt (B × Salt) D Hi VL) (M : ∀ j, (T j).Hm96 C)
    (digT : Fin J → F → B) {δ₁ : ℝ≥0∞} {St : Type} [Fintype St] (out : (∀ j, (T j).Rand) → St)
    (full : St → Fin J → Bool) (wit : (∀ j, (T j).Rand × ((T j).Hid → Salt)) → Fin J → W) (w₀ : Fin J → W),
    FlockSoundness.ZK.Session.WitnessesFirst wit →
    (∀ j, Claims.Hm96Hiding (M j).My (M j).cs δ₁) → (∀ j, (M j).TauLeaf (digT j)) → (∀ j, (T j).Simulable) →
    (∀ p j, (T j).RegPublic (wit p j)) → (∀ j, (T j).RegPublic (w₀ j)) →
    (∀ p, FlockSoundness.ZK.Session.Gated T out full (wit p)) →
    ∀ E : (∀ j, (T j).View ⊕ (T j).PreView) → Prop,
      (∀ s, prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => out (fun j => (p j).1) = s ∧
          E (FlockSoundness.ZK.Session.releaseR T M full (wit p) (out fun j => (p j).1) p)) ≤
        prCoin (fun σ => E (FlockSoundness.ZK.Session.simR T M full w₀ s σ)) +
          2 * FlockSoundness.ZK.Session.unopened T full s * δ₁) ∧
      prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) =>
          E (FlockSoundness.ZK.Session.releaseR T M full (wit p) (out fun j => (p j).1) p)) ≤
        ∑ s, (prCoin (fun σ => E (FlockSoundness.ZK.Session.simR T M full w₀ s σ)) +
          2 * FlockSoundness.ZK.Session.unopened T full s * δ₁) ∧
      prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) =>
          E (FlockSoundness.ZK.Session.releaseR T M full (wit p) (out fun j => (p j).1) p)) ≤
        (Fintype.card St : ℝ≥0∞) *
          prCoin (fun q : St × (∀ j, ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt))) =>
            E (FlockSoundness.ZK.Session.simR T M full w₀ q.1 q.2)) +
          ∑ s, 2 * FlockSoundness.ZK.Session.unopened T full s * δ₁

end Flock.Guarantees
