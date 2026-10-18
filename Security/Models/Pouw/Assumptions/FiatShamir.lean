import Definitions.Pouw.FiatShamir

/-!
# Relative security's named premises under Fiat–Shamir draws (trusted)

The premises of the Fiat–Shamir statements in `Models.Pouw.Guarantees.FiatShamir`, over decision 102's seeds
(`Definitions.Pouw.FiatShamir`). Each stage's draw is SHA-512 of commitments the prover has made, read as a random
oracle at the window's salt (`random-oracle`, the statements' probability model, not a premise). The live coins
are each round's check.

* `SteerBound`: a try's draw steers with probability at most `μ^m` (decision 101's per-try `μ^m`; Daniel, 18 Oct
  02:39Z). `FiatShamirFloor` takes it.
* `ServiceSoundStagedFS`: `ServiceSoundStaged` with its stage draws from the oracle, the staged service sound up to its
  proof layer at every oracle. It is the premise from which the staged service's live statement at unsteered draws
  (`LiveFloorUnsteered`) is to be derived with K's scope and the machine's charge; no guarantee reads it yet.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity

/-- `pouw-fs-steer` (proposed). **A try steers with probability at most `μ^m`.** For every adversary within `T` oracle
calls, every value `r` of its own coins, every oracle `H` and every input `x` (a stage's
`tag ‖ s ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j)`, committed or only tried), redrawing the oracle at `x` alone
steers the adversary's run with probability at most `μ^m`: a steer needs `m` coordinates of the draw, each matched
with probability at most `μ` (decision 101: "steering needs grinding (per try ≤ μ^m)"; Daniel, 18 Oct 02:39Z). The
rest of the oracle is fixed, so the event may depend on the adversary's run before the try and after it, but not on
the draw at `x` except through the event itself.

This is where Fiat–Shamir's risk sits, and what the analysis's `Steer` must make true together with
`LiveFloorUnsteered`: the steer events must hold every draw at which the live coins alone don't bound the floor's
event, and still be rare at every input. It fails if a cheaper run needs fewer coordinates matched than `m` (stages
whose cells' errors share coordinates, so one matched coordinate lets many escape), or if the inputs aren't fresh per
stage (two stages naming one input share a draw: decision 102 binds the forward and position and refuses repeats). -/
def SteerBound {J N Live : Type} [Fintype J] [Fintype N] (Mc : Machine) (S : FSService J N Live) (T : ℕ)
    (Steer : SteerMap Mc S) (μ : ℝ) (m : ℕ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : FSAdv Mc S Ω), FSWithin Mc S M T → ∀ (r : Ω) (H : S.Q → N) (x : S.Q),
    prob (fun d : N => Steer Ω M r (Function.update H x d) x) ≤ μ ^ m

section StagedFiatShamir

variable {R St n L np B n' : ℕ} {Ar : Type} [CommRing Ar]

/-- `pouw-fs-staged-service` (proposed). **The staged service under Fiat–Shamir draws is sound up to its proof
layer.** `ServiceSoundStaged` with decision 102's seeds: stage `(w, s)`'s draw is the oracle's answer at the input its
commitment names (`pre`), and each round's check is live (S4, round protocol with C1 per stage and C4 per draw; S5's
live coins now the checks only, on `trusted-nebius-host`). For every adversary within `T` oracle calls, every value `r`
of its own coins and every oracle `H`: its run's play fixes each stage's error table at the stage's operand
commitment, which the stage's input hashes (`root(A_j)`, `root(B_j)`), so before its draw (the play's form,
`StagedPlay`; a re-rolled root is a new input and a new draw, which `SteerBound` prices); every run the verifier
accepts is accepted by the play's game at its realized coins (`stagedCoins`) or breaks binding or zero knowledge in
some round (N1, X-RO commitments; F1 and F2, C-Flock's binding and zero knowledge), at most `δb` and `δz` a round
over the live coins. -/
def ServiceSoundStagedFS (Mc : Machine) (S : StagedFS R St n L np B n') (T : ℕ) (run : RunMapFS Mc S Ar)
    (δb δz : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : FSAdv Mc S Ω), FSWithin Mc S M T →
    ∀ (r : Ω) (H : S.Q → Finset (DBits n L np B n')),
      (∀ w, prob (fun l : Fin R → Fin St × DCells n n' B =>
        (run Ω M r H).Bind w (stagedCoins (S.draws H (Mc.eval M ((H, l), r)).1) l)) ≤ δb) ∧
      (∀ w, prob (fun l : Fin R → Fin St × DCells n n' B =>
        (run Ω M r H).ZK w (stagedCoins (S.draws H (Mc.eval M ((H, l), r)).1) l)) ≤ δz) ∧
      ∀ l, S.verify (Mc.eval M ((H, l), r)).1 (S.draws H (Mc.eval M ((H, l), r)).1) l (Mc.eval M ((H, l), r)).2 = true →
        (run Ω M r H).play.game.accepts (stagedCoins (S.draws H (Mc.eval M ((H, l), r)).1) l) ∨
          ∃ w, (run Ω M r H).Bind w (stagedCoins (S.draws H (Mc.eval M ((H, l), r)).1) l) ∨
            (run Ω M r H).ZK w (stagedCoins (S.draws H (Mc.eval M ((H, l), r)).1) l)

end StagedFiatShamir

end Pouw.Assumptions
