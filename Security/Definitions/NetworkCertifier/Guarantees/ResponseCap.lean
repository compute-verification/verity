module

public import Definitions.NetworkCertifier.ResponseCap

/-!
# The response cap's guarantee

`Definitions.NetworkCertifier.ResponseCap`. The statement only; `Proofs.NetworkCertifier.ResponseCap` proves it.
-/

@[expose] public section

namespace NetTiming.Guarantees

open NetTiming

/-- **Each certified response holds at most its request's certified `max_tokens` tokens, hence at most the cap.** On a
schedule `s` registered with the response cap `cap`, each connection `c`'s request read at ingress as `mt c` (`none`
where its field isn't a u32), the network certifier's code under the cap (`cappedCode`) sends every connection at most
`cap` data cells and at most its request's `max_tokens` (none for a request it couldn't read), whatever the server
sends and whenever it arrives (`arr`). In the id egress form a data cell is one token. Under `trusted-host` the network
certifier's run is this code, so each response it certifies is so bounded. -/
def ResponseCapped : Prop :=
  ∀ {κ Cell : Type} (s : CertifierSchedule κ) (cap : ℕ) (mt : κ → Option ℕ) (arr : κ → Arrivals Cell) (c : κ),
    dataCells (cappedCode s cap mt arr c) ≤ cap ∧ dataCells (cappedCode s cap mt arr c) ≤ (mt c).getD 0

end NetTiming.Guarantees
