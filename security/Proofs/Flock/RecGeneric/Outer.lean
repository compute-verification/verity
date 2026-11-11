module

public import Definitions.Claims.RecursionGeneric.Outer
public import Proofs.Flock.Claims.Recursion.OuterKS
public import Proofs.Flock.Claims.Recursion.Serve

/-!
# Π_out's extraction, and the composite extractor's cost

The compiled verifier and the compiler over Π_out are `Definitions.Claims.RecursionGeneric.Outer`'s. Its extractor's
interface is stated, not assumed: `Arg.KS` is its knowledge soundness in `Claims.KS`'s form (`κ` plus `√(γ·Pr[B])`, `B`
Π_out's verifier's fork finder); `ExtractRunsLe` and `ExtractRunsExpLe` at `A.E` bound its reruns, strictly or in
expectation; and `Sampling.ExtractOwnLe` at `A.E` bounds the SHA-512 calls it makes itself, beside its reruns'. Running
time is reruns and calls, and `assemble`'s steps, which it returns beside its candidate.

`OuterSessionsKS` is that `E_O` meets `Claims.KS` where `A`'s extractor does and the compiler meets `Compiled.Sessions`
(lemma O1, the generic form of `Claims.Recursion.outerKS_of_flockKS`); `ExtractCost` is the composite extractor's cost
(lemma O2).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

/-- **O1: `E_O` is knowledge sound, from Π_out's** (the generic form of
`Claims.Recursion.outerKS_of_flockKS`): where `A`'s counted extractor is knowledge sound and the compiler meets
`Sessions`, `E_O` meets `Claims.KS` at `κ_O`, `γ_O`, from the sessions' bounds (`Bounded`). Per session: the session's
prover after the run's earlier messages (`sessProver`) on the judged run's coins before the session; `A`'s KS at its
staged statement, whose shape is the session's and whose errors are within the session's bounds; Jensen over the
earlier coins; its fork finder is the compiled verifier's `B` at a round guess among `k(x)`; a union over sessions
(`collect`). -/
def OuterSessionsKS : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (A : Arg Coin)
    (L : Compiled Coin A), A.KS H → L.Sessions H →
      Claims.KS H L.V L.toComp.RO L.toComp.EO L.toComp.κO L.toComp.γO

/-- **The composite extractor's cost** (lemma O2): `E*` (`Comp.E` at `L.toComp`) reruns the prover at most
`(1 + m_Π·K_Π)·(1 + Σ_s ρ(sh_s))` times on every coin where `A` reruns at most `ρ`; in expectation at most
`(1 + m_Π·k_Π)·(1 + Σ_s e(sh_s))` where `A` reruns at most `e` in expectation (Wald, over `E^rep`'s fresh reruns); and
it makes at most `(1 + m_Π·K_Π)·(Σ_s h(sh_s) + n)` SHA-512 calls of its own (`Sampling.ExtractOwnLe`) where `A` makes at
most `h` (`Sampling.ExtractOwnLe` at `A.E`) and `Open(Π, S)` at most `n` on any witnesses' inner messages (`Comp.innerLog`): each of
`E^rep`'s reruns is `E_O`'s sessions and the inner view's run, whose calls are `E*`'s own (`Comp.decProg`). The inner
`E^rep`'s and `E_O`'s reruns multiply; their errors do not (`Comp.κ`: `N_I` times `κ_O`). -/
def ExtractCost : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (A : Arg Coin)
    (L : Compiled Coin A) (I : Proto Coin W) (S : Server) (ρ h : A.Sh → ℕ) (e : A.Sh → ℝ) (n : Msg → ℕ),
    L.Sessions H →
      ((ExtractRunsLe H A.E fun x => ρ (A.shape x)) → ExtractRunsLe H (L.toComp.E I S) fun x =>
        (1 + I.waits (L.xI x) * I.cap (L.xI x)) * (1 + ∑ s : Fin (L.nS x), ρ (L.sh x s))) ∧
      ((ExtractRunsExpLe H A.E fun x => e (A.shape x)) → ExtractRunsExpLe H (L.toComp.E I S) fun x =>
        (1 + I.waits (L.xI x) * I.k (L.xI x) : ℝ) * (1 + ∑ s : Fin (L.nS x), e (L.sh x s))) ∧
      ((Claims.Sampling.ExtractOwnLe H A.E fun x => h (A.shape x)) →
        (∀ x r ws, (L.toComp.innerLog (S.IVStar I.V I.ans) H x r ws).length ≤ n x) →
        Claims.Sampling.ExtractOwnLe H (L.toComp.E I S) fun x =>
          (1 + I.waits (L.xI x) * I.cap (L.xI x)) * (∑ s : Fin (L.nS x), h (L.sh x s) + n x))

end Claims.RecGeneric
