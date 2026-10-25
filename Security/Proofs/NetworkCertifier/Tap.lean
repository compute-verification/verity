import Definitions.NetworkCertifier.Tap
import Proofs.Core.AuditorDevice

/-!
# The warden got what the certifier saw

On the one tap that runs its code by `trusted-host`, the warden's half of its run is the warden's code on what reached
the tap (`trustedHost_warden`), and the egress data the warden's code read on each connection is the data read back from
the certifier's records of the same run (`wardenCertifierEgress`). This is the lemma that replaces the named assumption
`warden-certifier-egress`, which tied two devices together.
-/

namespace NetTiming

open Verity.Assumptions

/-- **The warden half of the tap**: the client's view of the tap's run is the warden's code on the schedule and on what
reached the tap from the server, so `EncardWardenViewsLe` and its lemmas apply to it. -/
theorem trustedHost_warden {Conn Cell Rec Srv Dev : Type} (arr : Srv → Conn → Arrivals Cell)
    (tc : WardenTap Conn Cell Rec) (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × Rec)
    (htrust : TrustedHost tap run (tc.code.onUplink arr)) :
    TrustedHost tap (fun d p => (run d p).1) fun p => wardenCode p.1 (arr p.2) :=
  Verity.TapCode.trustedHost_relay htrust

/-- **The warden got what the certifier saw**: on the tap that runs its code by `trusted-host`, at every schedule `s`
and server behavior `σ`, the egress data reaching the warden on each connection is the data read back from the
certifier's records of the same run. -/
theorem wardenCertifierEgress {Conn Cell Rec Srv Dev : Type} (arr : Srv → Conn → Arrivals Cell)
    (tc : WardenTap Conn Cell Rec) (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × Rec)
    (htrust : TrustedHost tap run (tc.code.onUplink arr)) (s : WardenSchedule Conn) (σ : Srv) :
    (fun c => (arr σ c).stream) = tc.read (run tap (s, σ)).2 :=
  (Verity.TapCode.read_run htrust s σ).symm

end NetTiming
