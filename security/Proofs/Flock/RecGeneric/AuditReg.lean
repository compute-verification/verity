module

public import Definitions.Core.Scheme
public import Proofs.Flock.RecGeneric.Sampled
public import Proofs.Spot.Sampling.RegDraw

/-!
# The recursive audit at the design the prover registers, in the form a scheme reads

`Flock.Guarantees.RecursiveSampledKS` is the recursive audit at one sampling design fixed before the prover's strategy,
as a bound on acceptance. Here the design is `L R`, read from the registration `R` the prover sends before the coin
(`Pouw.Window.auditReg`), and the bound is `Verity.AuditBound`, the one the accounting layer reads
(`Verity.Scheme.AccountingLayer`).

The outer statement at a sample `S` is `stmt S R`: a compiled verifier that recomputes the draw from the coins inside
its statement is an instance when its verdict reads the coins only through the sample it draws. The prover's state after
registering is a strategy for every coin, so the outer prover sees the coin before it proves.

Statements only.
-/

@[expose] public section

namespace Flock.Guarantees

open Claims Claims.RecGeneric FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.SampledGeneric
open scoped ENNReal

/-- **The recursive audit at the registered design.** For every compiled verifier `K` over the inner protocol `I` and
the server `Sv` with L1's hypotheses at every sampled statement (as `RecursiveSampledKS` takes them), design `L R` read
from the registration, binding commitment `B` to the trace and witnesses that are sound for the proof blocks
(`JointExtractor.Sound`): at every strategy `σ` of the registered-design audit whose link finders meet ecr/sha-512 at
the trials' counted calls (`Trials.CR`), `Verity.AuditBound` holds at the design `L σ.1` it registered, with the
incorrect set `F` of the plurality of the trials, and the error
$$\mathbb E_S \max_{u \in S} \mathrm{Comp.err}(\mathrm{stmt}(S, R), P_S) + \varepsilon_{\rm bind}.$$
At every cutoff `δ`, a run is accepted while `L σ.1`'s sample misses `F` with probability at most `δ`, with
probability at most `δ` plus that error. -/
def RecursiveAuditBoundReg : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Hash.Msg → Hash.Msg) (I : Proto Coin W)
    (Sv : Server) (K : Claims.RecGeneric.Comp Coin) (mI : Fin I.nT → ℕ) (Mo : ℕ) (n : ℕ) (Reg Dh : Type)
    (Hc : List UInt8 → Dh) (L : Reg → SamplingDesign n) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (stmt : Finset (Fin n) → Reg → Hash.Msg)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening) (M : ℕ),
    I.WF → I.KSAbortLin H → I.LastCoinFree H → Sv.Binds →
    (∀ S R, K.OuterKSAt H (stmt S R) ∧ 0 ≤ K.κO (stmt S R) ∧ 0 ≤ K.γO (stmt S R) ∧
      K.Contract H (Sv.IVStar I.V I.ans) (fun v => I.opens (Sv.openView v)) (stmt S R)) →
    (jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).Sound Correct → 1 ≤ M →
    ∀ σ : Strategy (Pouw.Window.auditReg L (proofSystemOf H K.V stmt).session),
      ((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials (L σ.1)).CR (reg (atReg σ))
          (cont (atReg σ)) M →
      Verity.AuditBound (fun _ => L σ.1)
        (fun p => prob p (Pouw.Window.auditReg L (proofSystemOf H K.V stmt).session) σ) (fun o => o.1 = true)
        (fun _ => incorrect B Correct (reg (atReg σ))
          (((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials (L σ.1)).plur (reg (atReg σ))
            (cont (atReg σ))))
        (avg (fun ω => ⨆ u ∈ (L σ.1).draw ω,
            ENNReal.ofReal (K.err I Sv H (proverOf (x := stmt ((L σ.1).draw ω) σ.1) (cont (atReg σ) ω))
              (stmt ((L σ.1).draw ω) σ.1) mI Mo)) +
          ((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials (L σ.1)).bindErr (reg (atReg σ))
            (cont (atReg σ)) M)

end Flock.Guarantees
