module

public import Definitions.Claims.RecursionGeneric.Outer

/-!
# The recursion as a compiler, generic in Π_in, Π_out and the ZK server

The recursion proves Π_in's verifier inside Π_out's proofs. Π_in is any public-coin protocol whose knowledge soundness
its repetition extractor `E^rep` shows, stated against aborting provers in run-finder form (`Proto`); a Π_in whose
extractor forks at a later round or extracts straight-line is not covered. A ZK server (`Server`) commits to each of
Π_in's messages before the coin that answers it, and `Open(Π_in, S)` (`Server.IVStar`), Π_in's verifier over those
commitments, is what the compiled verifier's circuits V\* compute. Π_out is any proof system for proof blocks as the
sampled audit reads it (`Claims.SamplingGeneric.Arg`), knowledge sound by its own counted extractor (`Arg.KS`).

The composed system's extractor `E*` (`Comp.E`) is Π_in's `E^rep` at Π_in's statement, each rerun served by `E_O`, the
extraction of Π_out's witnesses from the compiled run, and read as the view of `Open(Π_in, S)` those witnesses give.
Its knowledge error is `Comp.err`, in Π_in's, Π_out's and the compiled verifier's terms.

That V\*'s circuits compute `Open(Π_in, S)` is the compiler's contract (`Comp.Contract`), stated at one statement: code
facts about the compiled verifier and `E_O`, and, where some run accepts the statement, V\*'s two (`ForksBind`: the
commitments bind across forks; `InnerAcceptsColl`: witnesses `E_O` gives on an accepted run give an accepting view of
`Open(Π_in, S)` or a SHA-512 collision). The bound holds at each statement where the contract holds, so it covers every
statement the compiled verifier accepts wherever the contract holds at every statement.

Two closed statements, the second over the first, stated in `Definitions.Flock.Recursive.Generic`:

* `CompositionKS` (L1): any Π_in, any binding server, any compiled verifier with any extraction `E_O`, at each
  statement where `E_O` is knowledge sound (`Comp.OuterKSAt`) and the contract holds.
* `CompilerKS` (L2): any Π_in, any binding server, any knowledge-sound Π_out and any compiler whose compiled verifier
  meets Π_out's sessions' contract (`Compiled.Sessions`), at each statement where the contract holds: `E_O` runs Π_out's
  extractor session by session (`Compiled.EO`), and its errors sum the sessions' bounds.

Zero knowledge of the composed system is not stated here. Z1 (`Flock.Guarantees.GenericRecursiveZK`) states it over
the ZK server's release (`Claims.RecGeneric.ZkRelease`, in `Proofs.Flock.RecGeneric.ZK`), whose definitions live in
`Proofs`.

The Flock instances: `Flock.Guarantees.RecursiveKSEvery` is `CompositionKS` at today's compiled record, its `E_O`, the
hm96 server and the element-level Flock (`Claims.RecGeneric.RecoversLinkOn` takes `CompositionKS`), and
`Flock.Guarantees.RecursiveKSCompiledEvery` is `CompilerKS` at Π_out the element-level Flock (`Arg.ofFlock`) and
today's compiler (`Claims.RecGeneric.ksCompiledOn_of` takes `CompilerKS`).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

section Err

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The composed system's knowledge error** at Π_in `I` over the server `S`, the prover `P`, the statement `x`, `I`'s
loop bounds `m` and the link's `M`:
`κ(x) + √(γ(x)·Pr[B]) + Σ_i w_i(x_I)·(Pr[F_i(m_i) on the reruns] + τ_i(x_I, m_i)) + Σ_i w_i·d_i·Pr_out + Pr[B_I] +
β(x)·(Pr[F_dev(M)] + 1/(M+1) + Pr_out)`. Here `κ = κ_Π + N_I·κ_O` and `γ = N_I²·γ_O` (`Comp.κ`, `Comp.γ`,
`N_I = 1 + m_Π·k_Π`); `F_i` are Π_in's own terms, read on the composed system's reruns as the view of `Open(Π_in, S)`
accepts (`innerModel`); `Pr_out` is the chance that `E_O` gives witnesses outside `R_O` at a rerun (`Comp.PrOut`, `0`
where `R_O` holds at every witness); `B_I` (`forkPlan`) finds a rejecting judged run and `F_dev` (`linkPlan`) the
deviation, both as `E_O` gives (`verModel`), at weight `β = (N_I + Σ_i w_i·d_i)·k_Π`. `F_dev` reads only whether `E_O`
gives witnesses, not `R_O`, so the link pays `Pr_out` where its first accepted rerun's witnesses fall outside `R_O`. -/
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

end Err

end Claims.RecGeneric
