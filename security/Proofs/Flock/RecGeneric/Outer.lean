module

public import Definitions.Claims.RecursionGeneric.Outer
public import Proofs.Flock.Claims.Recursion.OuterKS
public import Proofs.Flock.Claims.Recursion.Serve

/-! Π_out's extractor's interface and the composite extractor's cost, stated, not assumed; definitions, no proofs. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

def OuterSessionsKS : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (A : Arg Coin)
    (L : Compiled Coin A), A.KS H → L.Sessions H →
      Claims.KS H L.V L.toComp.RO L.toComp.EO L.toComp.κO L.toComp.γO

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
