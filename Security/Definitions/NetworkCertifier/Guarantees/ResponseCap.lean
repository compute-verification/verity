module

public import Definitions.Core.Assumptions
public import Definitions.NetworkCertifier.ResponseCap

/-!
# The response cap's guarantee

`Definitions.NetworkCertifier.ResponseCap`. The statement only; `Proofs.NetworkCertifier.ResponseCap` proves it.
-/

@[expose] public section

namespace NetTiming.Guarantees

open NetTiming

/-- **Each certified response holds at most its request's certified `max_tokens` tokens, hence at most the cap.** On a
schedule `s` registered with the response cap `cap`, each connection `c`'s request read at ingress as `mt c`: with the
network certifier running its code under the cap (`cappedCode`) by `trusted-host`, whatever the server sends and
whenever it arrives (`arr σ`), every connection's lane holds at most `cap` data cells, and at most `m` where its request
read as `some m`. In the id egress form a data cell is one token. -/
def ResponseCapped : Prop :=
  ∀ {κ Cell Srv Dev : Type} (s : CertifierSchedule κ) (cap : ℕ) (mt : κ → Option ℕ) (arr : Srv → κ → Arrivals Cell)
    (dev : Dev) (run : Dev → Srv → κ → List (ℕ × CellView Cell)),
    Verity.Assumptions.TrustedHost dev run (fun σ => cappedCode s cap mt (arr σ)) →
    ∀ σ c, dataCells (run dev σ c) ≤ cap ∧ ∀ m, mt c = some m → dataCells (run dev σ c) ≤ m

end NetTiming.Guarantees
