module

public import Definitions.Claims.SamplingGeneric

/-!
# The sampled compiler's knowledge soundness, in two steps

`CompilerKS` in the two steps `Claims.Sampling`'s proofs take at Flock, each a statement a proof can target alone:

* `KSOf` (`Sampling.KS_of`, generic): `A`'s knowledge soundness at the sampled proof blocks' sessions (`SessionKS`), the
  commitment's binding and the contract give `Compiled.KS`. A failed extraction at an accepted run is a failed
  session or a collision among the extraction's SHA-512 calls; failed sessions are rare where the sample is likely to
  hit an incorrect proof block (`Sessions.draws`).
* `SessionKSOf` (`Sampling.SessionKS_of`, generic): `A`'s `KS`, its rounds its shape's and the contract give
  `SessionKS`: at each session `A`'s knowledge soundness at the prover `Sessions.runs` shows, a session's fork being the
  compiled verifier's at one of its `k(x)` rounds.

`compilerKS_of` composes them.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (forkOn)
open Claims.Recursion (serve viewing)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **A sampled proof block's session is extracted** at the run `run'` on coins `r`, at `A`'s extractor coins `c`: the
run shows the block's session at the block's shape, and `A`'s extractor there, its reruns served as in `unitExt`, finds
a witness in `A`'s relation at the session's statement. -/
def UnitOk (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace) (u : ℕ)
    (r : Fin (L.V.k x) → Coin) (c : A.EC (L.sh x u)) : Prop :=
  ∃ xS vS, ∃ h : A.shape xS = L.sh x u, L.session x u (run' r).seen = some (xS, vS) ∧
    A.R.ext xS ((serve (fun r' => viewing (L.view x u) (L.place x u xS r r'))
      (A.ext xS (L.coinsOf x u xS r) (cast (congrArg A.EC h.symm) c))).eval H run').1

/-- **`A`'s knowledge soundness at the sampled proof blocks' sessions** (`Sampling.SessionKS`, generic): on every prover,
a nonnegative weight `β` of each tape and coins `(ω, r)`, averaging at most `Pr[B]`, such that at `(ω, r)`, over fresh
second-stage coins and `A`'s extractor's coins at each proof block, the run accepts and the sampled proof block a
selection `sel` picks is not extracted with probability at most `κ_s + √(k(x)·γ_s·β(ω, r))`. -/
def SessionKS (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    ∃ β : Ω → (Fin (L.V.k x) → Coin) → ℝ, (∀ ω r, 0 ≤ β ω r) ∧
      Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x ∧
      ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x))),
        Pr (fun p : (Fin (L.V.k x) → Coin) × ((u : Fin (L.nPU x)) → A.EC (L.sh x u)) =>
          (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
            ∃ u, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some u ∧
              u.val ∈ L.checked x (forkOn (L.stage2 x) r p.1) ∧
              ¬ L.UnitOk H x (run H P L.V x ω) u (forkOn (L.stage2 x) r p.1) (p.2 u)) ≤
          L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)

end Compiled

/-- **KS from the sessions' KS** (`Sampling.KS_of`, generic): a compiled sampled verifier that meets the contract, with
a binding commitment, whose sessions are knowledge sound (`SessionKS`), meets `KS`. -/
def KSOf : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), L.commit.Binds H → L.Sessions H → L.SessionKS H → L.KS H

/-- **The sessions' KS from the proof system's** (`Sampling.SessionKS_of`, generic): over a knowledge-sound proof
system whose rounds are its shape's, a compiled sampled verifier that meets the contract has knowledge-sound sessions. -/
def SessionKSOf : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), A.WF → A.KS H → L.Sessions H → L.SessionKS H

theorem compilerKS_of (hks : KSOf) (hsess : SessionKSOf) : CompilerKS := by
  intro Coin _ _ _ A L H hwf hA hB hS
  exact hks Coin A L H hB hS (hsess Coin A L H hwf hA hS)

end Claims.SamplingGeneric
