import Definitions.NetworkCertifier.Tap
import Definitions.NetworkCertifier.TapRows

/-!
# The named conditions of a monitoring period's statements

Definitions only. The statements over a monitoring period of sampled audits take these conditions, each named once:

* the network certifier registers its observations at every window (`CertifiersOpen`);
* each window's statement has its partition, and each of its possible faulty sets commits a bounded number of bits
  (`WindowsPlaced`);
* the network certifier's record format holds each schedule of `Ss`, which caps each connection's cells
  (`SchedulesHeld`);
* the schedule the network certifier runs for each of a covert sender's secrets is one of `Ss` (`SchedulesCovered`), a
  property of the compiled Lean audit over the network certifier's signed records;
* a decoder recovers each secret from the client's view (`SenderDecodes`), the count's own terms.
-/

namespace NetTiming

open Claims Claims.Sampling

section Period

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The network certifier registers its observations at every window**: for each secret `z` of `Zs` and window
`i`, at the statement `x z i` parses to, its openings `tops s` register the observations `obs z i` at the window's
operator `op i` under the statement's roots (`TapOpens`). -/
def CertifiersOpen {L K S n : ℕ} {Z : Type} (H : Msg → Msg) (D : Deploy Coin Col T)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (op : Fin n → ℕ → List (Obs L K S) → List Bool)
    (tops : AuditStatement → List (Opening Bool)) (Zs : Set Z) (x : Z → Fin n → Msg)
    (obs : Z → Fin n → List (Obs L K S)) : Prop :=
  ∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → TapOpens H D.commit (lay s) (k s) (op i) s (tops s) (obs z i)

/-- **The windows' registrations**: each window's statement `sw i` has the partition `Pw i`, and each of its possible
faulty sets at `δ` commits at most `β i` bits. -/
def WindowsPlaced {n : ℕ} (D : Deploy Coin Col T) (δ : ℝ) (sw : Fin n → AuditStatement) (Pw : Fin n → Partition)
    (β : Fin n → ℕ) : Prop :=
  (∀ i, D.partition (sw i) = some (Pw i)) ∧
    ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i

end Period

/-- **The schedules the network certifier's record format holds**: each schedule of `Ss` gives connection `c` at most
`cap c` cells, and the record format `f` holds it (`TapRecord.fits`). -/
def SchedulesHeld {Conn Cell Ob : Type} (f : TapRecord Conn Cell Ob) (cap : Conn → ℕ) (Ss : Set (WardenSchedule Conn)) :
    Prop :=
  ∀ s ∈ Ss, (∀ c, s.cells c ≤ cap c) ∧ f.fits s

/-- **The schedules a covert sender's secrets run on are counted**: the schedule the network certifier runs for each
secret `z` of `Zs` (`schedOf z`) is one of `Ss`. The sender doesn't choose `Ss`: the count charges for it, `log2 #Ss`
where `SchedulesHeld` is stated over it, or `log2 #Θ` where `Ss` is the schedules whose `(r, D)` is in the registered
set `Θ` and which fit the rows format. So this holds of every covert sender where `Ss` holds every schedule the network
certifier's signed records can show: a property of the compiled Lean audit the verification runs over those records,
which `trusted-host` makes the traffic that crossed, never of the device's code, and not an assumption. -/
def SchedulesCovered {Conn Z : Type} (Ss : Set (WardenSchedule Conn)) (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) :
    Prop :=
  ∀ z ∈ Zs, schedOf z ∈ Ss

/-- **A covert sender's secrets, recovered from the client's view**: the decoder `dec` recovers each secret `z` of `Zs`,
run on the schedule `schedOf z` and the server behavior `srvOf z`, from the cells the network certifier's run `run cert`
gives the client. These are the count's own terms, a quantified hypothesis on the adversary's strategy (its secrets,
schedules, server behaviors and decoder), with `run` the network certifier's by `trusted-host`: nothing about the
verifier's or the prover's code. -/
def SenderDecodes {Conn Cell Srv Dev Z Ob : Type}
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × Ob) (cert : Dev) (Zs : Set Z)
    (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv) (dec : (Conn → List (ℕ × CellView Cell)) → Z) : Prop :=
  ∀ z ∈ Zs, dec (run cert (schedOf z, srvOf z)).1 = z

end NetTiming
