module

public import Proofs.Flock.RecGeneric.Outer
public import Proofs.Flock.RecGeneric.ZK

/-!
# The recursion as a compiler, generic in the outer argument and the server, over repetition-extracted inner protocols

Three layers, each a statement over the one below. The inner protocol is any whose knowledge soundness its repetition
extractor `E^rep` shows (`Proto`); an inner extractor that forks at a later round or extracts straight-line is not
covered.

* **L1, the composition** (`GenericRecursiveKS`): any inner protocol `Π` (`Proto`) over any server's commitments
  (`Server`), at any compiled verifier with any outer extraction (`Comp`), at each statement where the outer extraction
  is knowledge sound (`Comp.OuterKSAt`) and the compiler meets its contract (`Comp.Contract`). The composite
  extractor is `Π`'s `E^rep`, each rerun served by `E_O` (`Comp.E`); its error is `Comp.err`: `κ_Π + N_I·κ_O`,
  `√(N_I²·γ_O·Pr[B])`, `Π`'s own terms with its finders run on the composite's reruns, `E_O`'s witnesses outside `R_O`
  at each rerun those finders read (`Comp.PrOut` at weight `Σ_i w_i·d_i`), a rejecting judged run (`B_I`, `forkPlan`)
  and the deviation finder (`linkPlan`) with `Comp.PrOut` again, at weight `β = (N_I + Σ_i w_i·d_i)·k_Π`. The server
  binds (`Server.Binds`).
* **L2, the compiler** (`GenericRecursiveKSCompiled`): L1 at `compile(Π, A, S)` for any outer argument `A`
  (`Claims.SamplingGeneric.Arg`, the sampled audit's proof system for proof blocks) and
  compiler, its outer extraction session by session (`Compiled.EO`) and knowledge sound by `A`'s (`OuterSessionsKS`).
* **Zero knowledge** (`GenericRecursiveZK`, `Proofs.Flock.RecGeneric.ZK`): `Claims.ZKOut` of the ZK server's compiled
  prover, generic in the server's leaves and the outer phase, from the outer phase's `ZKOut`.

The Flock instance (`Proofs.Flock.RecGeneric.AtFlock`) is L1 at today's compiled record, its `E_O`, the hm96 server
and the element-level Flock: it gives the deployed theorem's bound (`Claims.Recursion.KSLin2`) with the gated link's
term in place of `Fdev`'s (`Claims.RecGeneric.KSLinkOn`), at the deployed decoders, at each statement where the
instance's contract holds.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The composite's knowledge error** at the inner protocol `I` over the server `S`, the prover `P`, the statement
`x`, `I`'s loop bounds `m` and the link's `M`:
`κ(x) + √(γ(x)·Pr[B]) + Σ_i w_i(x_I)·(Pr[F_i(m_i) on the reruns] + τ_i(x_I, m_i)) + Σ_i w_i·d_i·Pr_out + Pr[B_I] +
β(x)·(Pr[F_dev(M)] + 1/(M+1) + Pr_out)`, `F_i` read on the reruns as the inner view accepts (`innerModel`), `Pr_out` the
chance that `E_O` gives witnesses outside `R_O` at a rerun (`Comp.PrOut`, `0` where `R_O` holds at every witness), `B_I`
and `F_dev` as `E_O` gives (`verModel`). `F_dev` reads only whether `E_O` gives witnesses, not `R_O`, so the link pays
`Pr_out` where its first accepted rerun's witnesses fall outside `R_O`. -/
noncomputable def Comp.err (K : Comp Coin) (I : Proto Coin W) (S : Server) (H : Msg → Msg) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (x : Msg) (m : Fin I.nT → ℕ) (M : ℕ) : ℝ :=
  K.κ I x + Real.sqrt (K.γ I x * PrB H K.V P x) +
    ∑ i, (I.terms i).w (K.xI x) *
      (((I.terms i).F (m i)).PrAt H (K.innerModel (S.IVStar I.V I.ans) H P x) (K.xI x) +
        (I.terms i).tail (K.xI x) (m i)) +
    K.reads I x * K.PrOut H P x +
    (forkPlan (Coin := Coin) I.V.k).PrAt H (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) +
    K.β I x * ((linkPlan (Coin := Coin) I.V.k M).PrAt H (K.verModel (S.IVStar I.V I.ans) H P x) (K.xI x) +
      1 / ((M : ℝ) + 1) + K.PrOut H P x)

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric
open Claims.SamplingGeneric (Arg)

/-- **L1: the recursion's knowledge soundness, generic in the outer extraction, over inner protocols with repetition
extractors.** For every oracle `H`, inner protocol `Π` that is well formed and knowledge sound against aborting provers
in run-finder form (`Proto.WF`, `Proto.KSAbortLin`) and whose last step reads no last coin (`Proto.LastCoinFree`),
server `S` that binds (`Server.Binds`), compiled verifier with an outer extraction `E_O`, and statement `x` at which
`E_O` is knowledge sound (`Comp.OuterKSAt`) at nonnegative `κ_O(x)`, `γ_O(x)` and the compiler meets its contract at
`Open(Π, S)` and `Π`'s reading of opened views (`Comp.Contract`: code facts, and V\*'s two where some run accepts `x`):
on every prover and at every loop bound,
`Pr[run = acc ∧ ¬R(x, E*(x))] ≤ Comp.err`, `E*` the inner `E^rep` served by `E_O` (`Comp.E`). Every accepted
statement is covered wherever the contract holds at every statement. `Π`'s `assemble` and `E_O` are any programs, their
computation between reruns not counted, so `E*`'s witness is the one the prover committed only where they are counted
(Flock's `E^rep` at the deployed decoders), and bits are read only there. -/
def GenericRecursiveKS : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (I : Proto Coin W) (S : Server)
    (K : Claims.RecGeneric.Comp Coin),
    I.WF → I.KSAbortLin H → I.LastCoinFree H → S.Binds → ∀ x : Msg, K.OuterKSAt H x → 0 ≤ K.κO x → 0 ≤ K.γO x →
    K.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (m : Fin I.nT → ℕ) (M : ℕ),
      PrBad H K.V (K.R I) (K.E I S) x P ≤ K.err I S H P x m M

/-- **L2: the recursion as a compiler, generic in the outer argument and the server, over inner protocols with
repetition extractors.** For every
`Π` as in L1, server `S` that binds, outer argument `A` whose counted extractor is knowledge sound (`Arg.KS`; the
sampled audit's proof system for proof blocks, `Claims.SamplingGeneric.Arg`, its extractor `E^rep` at its counted
parts), and compiler `C` whose compiled verifier `compile(Π, A, S)` meets the outer sessions' contract
(`Compiled.Sessions`, among it each session's errors within its bounds), and statement `x` where it meets the
composition's (`Comp.Contract`): on every prover and at every loop bound, `Pr[run = acc ∧ ¬R(x, E*(x))] ≤ Comp.err`
at `compile(Π, A, S)`, with `κ_O = Σ_s κb_s` and `γ_O = (Σ_s √(γb_s·k(x)))²`. `E*` is the inner `E^rep` served by
`E_O`, and `E_O` replays the judged run and runs `A`'s extractor at each session in order (`Compiled.EO`): their costs
compose as `ExtractCost` says. It is L1 with `OuterSessionsKS`. The contract is met with any number of sequential
sessions: `Compiled.Accepts` is one-way, and `Compiled.Runs` reads each session's own verdict, not the compiled run's
(`EchoHolds` at an inner verifier of `k ≥ 2` rounds runs `k` sessions). -/
def GenericRecursiveKSCompiled : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (I : Proto Coin W) (S : Server)
    (A : Arg Coin) (C : Compiler Coin A),
    I.WF → I.KSAbortLin H → I.LastCoinFree H → S.Binds → A.KS H → (compile C I S).Sessions H →
    ∀ x : Msg, (compile C I S).toComp.Contract H (S.IVStar I.V I.ans) (fun v => I.opens (S.openView v)) x →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (m : Fin I.nT → ℕ) (M : ℕ),
      PrBad H (compile C I S).V ((compile C I S).toComp.R I) ((compile C I S).toComp.E I S) x P ≤
        (compile C I S).toComp.err I S H P x m M

end Flock.Guarantees
