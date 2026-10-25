import Definitions.Core.AuditorDevice
import Definitions.Core.Assumptions

/-!
# The auditor tap's run: its relay and its record are projections of one run

A tap that runs its code by `trusted-host` relays its code's relay and keeps its code's record, both on what reached it
from the server (`relay_run`, `record_run`). So what its relay got is what its record saw: the record read back is the
data that reached the tap (`read_run`), and the relay alone is a device that runs the relay's code by `trusted-host`
(`trustedHost_relay`).
-/

namespace Verity.TapCode

open Verity.Assumptions

variable {Cfg Seen Relay Rec Data Srv Dev : Type} {T : TapCode Cfg Seen Relay Rec Data} {seen : Srv → Seen} {d : Dev}
  {run : Dev → Cfg × Srv → Relay × Rec}

theorem relay_run (htrust : TrustedHost d run (T.onUplink seen)) (c : Cfg) (σ : Srv) :
    (run d (c, σ)).1 = T.relay c (seen σ) := by
  rw [htrust (c, σ)]
  rfl

theorem record_run (htrust : TrustedHost d run (T.onUplink seen)) (c : Cfg) (σ : Srv) :
    (run d (c, σ)).2 = T.record c (seen σ) := by
  rw [htrust (c, σ)]
  rfl

/-- **What the relay got is what the record saw**: on a tap that runs its code by `trusted-host`, the record of its run
on the server's behavior `σ`, read back, is the data of what reached it from `σ`, the input its relay ran on. -/
theorem read_run (htrust : TrustedHost d run (T.onUplink seen)) (c : Cfg) (σ : Srv) :
    T.read (run d (c, σ)).2 = T.data (seen σ) := by
  rw [record_run htrust]
  exact T.read_record c (seen σ)

/-- The tap's relay is a device that runs the relay's code by `trusted-host`. -/
theorem trustedHost_relay (htrust : TrustedHost d run (T.onUplink seen)) :
    TrustedHost d (fun d p => (run d p).1) fun p => T.relay p.1 (seen p.2) :=
  fun p => relay_run htrust p.1 p.2

end Verity.TapCode
