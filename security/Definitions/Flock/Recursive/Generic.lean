module

public import Definitions.Claims.RecursionGeneric

/-!
# The recursion as a compiler: its two closed statements

`CompositionKS` (L1) and `CompilerKS` (L2) over `Definitions.Claims.RecursionGeneric`'s vocabulary.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

/-- **L1: the composition is knowledge sound, generic in `E_O`.** For every oracle `H`, Π_in `I` that is well formed
and knowledge sound against aborting provers in run-finder form (`Proto.WF`, `Proto.KSAbortLin`) and whose last step
reads no last coin (`Proto.LastCoinFree`), server `S` that binds (`Server.Binds`), compiled verifier with an extraction
`E_O` (`Comp`), and statement `x` at which `E_O` is knowledge sound (`Comp.OuterKSAt`) at nonnegative `κ_O(x)`, `γ_O(x)`
and the compiler meets its contract at `Open(Π_in, S)` and Π_in's reading of opened views (`Comp.Contract`): on every
prover and at every loop bound, `Pr[run = acc ∧ ¬R(x, E*(x))] ≤ Comp.err`, `E*` Π_in's `E^rep` served by `E_O`
(`Comp.E`). Π_in's `assemble` and `E_O` are any programs, their computation between reruns not counted, so `E*`'s
witness is the one the prover committed only where they are counted (Flock's `E^rep` at the deployed decoders). -/
def CompositionKS : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (I : Proto Coin W) (S : Server)
    (K : Comp Coin),
    I.WF → I.KSAbortLin H → I.LastCoinFree H → S.Binds → ∀ x : Msg, K.OuterKSAt H x → 0 ≤ K.κO x → 0 ≤ K.γO x →
    K.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (m : Fin I.nT → ℕ) (M : ℕ),
      PrBad H K.V (K.R I) (K.E I S) x P ≤ K.err I S H P x m M

/-- **L2: the recursion is a knowledge-sound compiler, generic in Π_in, Π_out and the server.** For every Π_in `I` as
in L1, server `S` that binds, Π_out `A` whose counted extractor is knowledge sound (`Arg.KS`), compiler `C` whose
compiled verifier `compile(Π_in, A, S)` meets Π_out's sessions' contract (`Compiled.Sessions`, among it each session's
errors within its bounds), and statement `x` where it meets the composition's (`Comp.Contract`): on every prover and at
every loop bound, `Pr[run = acc ∧ ¬R(x, E*(x))] ≤ Comp.err` at `compile(Π_in, A, S)`, with `κ_O = Σ_s κb_s` and
`γ_O = (Σ_s √(γb_s·k(x)))²`. `E*` is Π_in's `E^rep` served by `E_O`, and `E_O` replays the judged run and runs `A`'s
extractor at each session in order (`Compiled.EO`). The contract is met with any number of sequential sessions:
`Compiled.Accepts` is one-way, and `Compiled.Runs` reads each session's own verdict, not the compiled run's. -/
def CompilerKS : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (I : Proto Coin W) (S : Server)
    (A : Arg Coin) (C : Compiler Coin A),
    I.WF → I.KSAbortLin H → I.LastCoinFree H → S.Binds → A.KS H → (compile C I S).Sessions H →
    ∀ x : Msg, (compile C I S).toComp.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (m : Fin I.nT → ℕ) (M : ℕ),
      PrBad H (compile C I S).V ((compile C I S).toComp.R I) ((compile C I S).toComp.E I S) x P ≤
        (compile C I S).toComp.err I S H P x m M

end Claims.RecGeneric
