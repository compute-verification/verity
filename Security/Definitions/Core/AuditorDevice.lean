/-!
# The auditor's device: one tap on each server's uplink

The auditor-trusted devices (AGENTS.md, "Auditor-trusted devices") are the auditor's hardware, each on a machine the
prover doesn't control, and trust in each is `trusted-host` (`Verity.Assumptions.TrustedHost`). On the network side
there is one such device per AI server: a tap on the server's uplink, which everything the server sends out reaches. The
active warden and the network certifier are this one device (Daniel, 25 Oct: "little taps on each AI server's uplink").
Its one code, run on what reaches it from the server, gives both what it relays on toward the client and what it
records. A protocol reads the device only through projections of that one run, so two readings of it, the warden's
cells and the certifier's records, can't see different traffic: that they saw the same is a lemma about the code, not
an assumption.

Definitions only. A tap's code (`TapCode`) is its relay and its record, each a function of its public configuration
and of what reached it, and its record carries the data it saw: reading the record back (`read`) gives the data of what
reached it (`data`). `TapCode.onUplink` is that code as the device runs it on a server's behavior.
-/

namespace Verity

/-- **An auditor tap's code**: on its public configuration `c` and what reached it from the server `x`, what it relays
on toward the client (`relay c x`) and what it records (`record c x`, its records and their signed certificates). The
record carries the data it saw: reading it back (`read`) gives the data of `x` (`data x`), at every configuration and
whatever reached it. -/
structure TapCode (Cfg Seen Relay Rec Data : Type) where
  relay : Cfg → Seen → Relay
  record : Cfg → Seen → Rec
  data : Seen → Data
  read : Rec → Data
  read_record : ∀ c x, read (record c x) = data x

/-- **The tap on a server's uplink**: its code on the configuration `p.1` and the server's behavior `p.2`, which reaches
the tap as `seen p.2`; the relay and the record are of one run, on the same input. `trusted-host` for the tap's device
`d`, run by `run`, is `Verity.Assumptions.TrustedHost d run (T.onUplink seen)`. -/
def TapCode.onUplink {Cfg Seen Relay Rec Data Srv : Type} (T : TapCode Cfg Seen Relay Rec Data) (seen : Srv → Seen) :
    Cfg × Srv → Relay × Rec :=
  fun p => (T.relay p.1 (seen p.2), T.record p.1 (seen p.2))

end Verity
