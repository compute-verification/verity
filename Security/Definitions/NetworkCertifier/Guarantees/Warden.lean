import Definitions.NetworkCertifier.Warden
import Definitions.NetworkCertifier.AdviceString
import Definitions.NetworkCertifier.Assumptions.AdviceString
import Definitions.Core.Assumptions
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# The warden's channel bound

Under design v3 the warden (`Definitions.NetworkCertifier.Warden`) sends every cell in its bucket on the public
schedule, so the client's view of the server side is a function of the schedule, each connection's fail cell or none,
and the data in the cells (`WardenViewEqClientView`). Everything else the server does, its accept time, its send timing,
its ACKs and flow control, reaches the client only through the fail cells. A connection with `n` cells has at most
`n + 1` fail choices, so the client's views number at most `#schedules · ∏_c (cells_c + 1) · #data`
(`EncardWardenViewsLe`). With the data from an advice string within the caps (`BitsCount`, #3533's bound) and a
content term, a covert sender in the server moves at most
`log2 #schedules + Σ_c log2 (cells_c + 1) + Σ_w b w + k` bits to the client (`EncardWardenDecodableLe`).

#3533's bound counts runs the compiled Program accepts exactly, which the all-frames egress tie gives only when no step
unit can be faulty: every step unit drawn `all`, or a sampled audit at `δ ≥ miss(1)`. At a smaller `δ` an accepted
sampled audit gives `Accepted` by the circuit's Program only with the faulty units' committed gates held as advice
(`SampledAuditAccepted`), and the composed charge is `SampledLeakAccepted`, with `SampledAuditAccepted` as its
hypothesis (`Definitions/NetworkCertifier/Guarantees/SampledCharge`): every value of the inputs other than the
ingress, the faulty units' committed bits and `log2` of the number of possible faulty sets, except with probability
`δ + ε_ks + ε_bind` over the selection. Its last two terms are the content term `k` below, for the circuit's Program.

The warden runs its code by `trusted-host` (`Verity.Assumptions.TrustedHost`), read by name: it is the
auditor's device, on a host the prover doesn't control, and its code's output includes each cell's bucket, so the
assumption covers the warden's timing too (its CPU isolation and its monotonic clock, which the rename to
`trusted-host` states). The content term is a premise of the run, not an assumption: the egress data no proved unit
fixes takes at most `2^k` values (`k` is `ε` times the egress data bits, at the draw's `δ`).

The schedules are counted as given. Admission must not depend on the server (a fail that freed its lane early would let
the server time the next admission): a schedule chosen from the server's behavior is still among `Ss` and charged
`log2 #Ss`, which is that coupling's bound.
-/

namespace NetTiming.Guarantees

open Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The client's view factors through the fail cells.** The warden device `w` runs the warden's code on the schedule
and on what reaches it from the server, whose behavior `σ` (accept time, send timing, ACKs, flow control, its data)
reaches it as the arrivals `arr σ` (`trusted-host`). Then what the client receives from the server side is the
client view of the schedule `s`, the data in the cells, and each connection's fail cell. -/
def WardenViewEqClientView : Prop :=
  ∀ {κ Cell Srv Dev : Type} (arr : Srv → κ → Arrivals Cell) (w : Dev)
    (run : Dev → WardenSchedule κ × Srv → κ → List (ℕ × CellView Cell))
    (htrust : Verity.Assumptions.TrustedHost w run fun p => wardenCode p.1 (arr p.2))
    (s : WardenSchedule κ) (σ : Srv),
    run w (s, σ) = clientView s (fun c => (arr σ c).stream) (failCell s (arr σ))

/-- **The warden's views count.** Over schedules in `Ss`, each giving connection `c` at most `cap c` cells, and server
behaviors in `Srvs` whose data lies in `Ds`, the client's views of the server side number at most
`#Ss · ∏_c (cap c + 1) · #Ds`: `log2 #Ss + Σ_c log2 (cap c + 1)` bits besides the data. -/
def EncardWardenViewsLe : Prop :=
  ∀ {κ Cell Srv Dev : Type} [Fintype κ] (arr : Srv → κ → Arrivals Cell) (w : Dev)
    (run : Dev → WardenSchedule κ × Srv → κ → List (ℕ × CellView Cell))
    (htrust : Verity.Assumptions.TrustedHost w run fun p => wardenCode p.1 (arr p.2))
    (cap : κ → ℕ) (Ss : Set (WardenSchedule κ)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (Srvs : Set Srv) (Ds : Set (κ → List Cell)) (hD : ∀ σ ∈ Srvs, (fun c => (arr σ c).stream) ∈ Ds),
    {v | ∃ s ∈ Ss, ∃ σ ∈ Srvs, run w (s, σ) = v}.encard ≤ Ss.encard * ((∏ c, (cap c + 1) : ℕ) : ℕ∞) * Ds.encard

/-- **The warden's channel bound, for the whole run.** Each secret `z` in `Zs` of a covert sender in the server picks
a schedule in `Ss` and a server behavior, whose data is `dataOf a u`: what the compiled Program fixes from an advice
string `a` charged at most `b w` in every window `w` (`bits` counts), and the content `u` that no proved unit fixes,
one of at most `2^k` values. A decoder of the client's view of the server side recovers every secret. Then `Zs` has at
most `#Ss · ∏_c (cap c + 1) · 2^(Σ_w b w) · 2^k` elements: the sender moves at most
`log2 #Ss + Σ_c log2 (cap c + 1) + Σ_w b w + k` bits to the client, the schedule's parameters, the fail cells, the
advice string and the content together. -/
def EncardWardenDecodableLe : Prop :=
  ∀ {κ Cell Srv Dev Z Adv U : Type} {W : ℕ} [Fintype κ] (arr : Srv → κ → Arrivals Cell) (w : Dev)
    (run : Dev → WardenSchedule κ × Srv → κ → List (ℕ × CellView Cell))
    (htrust : Verity.Assumptions.TrustedHost w run fun p => wardenCode p.1 (arr p.2))
    (cap : κ → ℕ) (Ss : Set (WardenSchedule κ)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (bits : Adv → Fin W → ℕ) (hbits : BitsCount bits) (b : Fin W → ℕ)
    (dataOf : Adv → U → κ → List Cell) (Us : Set U) (k : ℕ) (hUs : Us.encard ≤ ((2 ^ k : ℕ) : ℕ∞))
    (Zs : Set Z) (schedOf : Z → WardenSchedule κ) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (hdata : ∀ z ∈ Zs, ∃ a, WithinCaps bits b a ∧ ∃ u ∈ Us, (fun c => (arr (srvOf z) c).stream) = dataOf a u)
    (dec : (κ → List (ℕ × CellView Cell)) → Z) (hdec : ∀ z ∈ Zs, dec (run w (schedOf z, srvOf z)) = z),
    Zs.encard ≤ Ss.encard * (((∏ c, (cap c + 1)) * 2 ^ (∑ i, b i) * 2 ^ k : ℕ) : ℕ∞)

end NetTiming.Guarantees
