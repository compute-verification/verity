import Properties.Verity.Game
import Properties.NetworkCertifier.Period

/-!
# The network certifier's component at onsite, decoded: what `Verity.ComponentAccepts .network` gives

`Verity.ComponentAccepts .network s` (onsite's network component, `NetworkCheck.Period.protocol`, a protocol that reads
no blobs, so the same check on whatever blobs were sent, accepts its own case in what the prover sent) yields the
period that case decodes to (`NetworkPeriod`) and the period check's acceptance of it (`componentAccepts_network`),
which is `Properties.NetworkCertifier.PeriodAcceptMeans`'s hypothesis as it stands. With `PeriodAcceptMeans`,
every link-window of that period was accepted (`componentAccepts_network_windows`), as
`Verity.Subprotocol.NetworkWindowLe` (`window_le`) reads it at the compiled verdict.
-/

namespace Verity

open Properties.NetworkCertifier
open Core (canonicalJson)

/-- **The network certifier's case in what the prover sent decodes to the period `p`, `c`, `e`**: onsite's three byte
strings decode, and onsite's network component's case, on the blobs sent, decodes under the parameters onsite names
for it. -/
def NetworkPeriod (s : Sent) (p : NetworkCheck.Period.Params) (c : NetworkCheck.Period.Commitment)
    (e : NetworkCheck.Period.Evidence) : Prop :=
  ∃ n ps b d k ev, Decoded s n ps b d k ev ∧
    ∃ x ∈ Properties.Composition.triples s.components ps ev, x.1 = NetworkCheck.Period.protocol ∧
      (canonicalJson x.2.1.params.toUTF8 >>= NetworkCheck.Period.paramsOf) = .ok p ∧
        NetworkCheck.Period.commitmentOf x.2.2.commitment = .ok c ∧
          NetworkCheck.Period.evidenceOf x.2.2.evidence = .ok e

/-- **Onsite's network component accepting is the period check accepting the period its case decodes to**, the
component on any blobs being the period check itself (`Protocol.withoutBlobs_onBlobs`). -/
theorem componentAccepts_network {s : Sent} (h : ComponentAccepts .network s) :
    ∃ p c e, NetworkPeriod s p c e ∧ NetworkCheck.Period.protocol.Accepts p c e := by
  obtain ⟨n, ps, b, d, k, ev, hd, x, hx, h1, hda⟩ := h
  have h1' : x.1 = NetworkCheck.Period.protocol :=
    h1.trans (Core.Protocol.withoutBlobs_onBlobs NetworkCheck.Period.protocol (Core.Blobs.ofArray s.blobs))
  rw [h1'] at hda
  obtain ⟨p, c, e, hp, hc, he, hacc⟩ := hda
  exact ⟨p, c, e, ⟨n, ps, b, d, k, ev, hd, x, hx, h1', hp, hc, he⟩, hacc⟩

/-- **Period acceptance at onsite**: given `PeriodAcceptMeans`, when onsite's network component accepts, its case decodes to a
period whose every link-window was accepted in the parameters' form. -/
theorem componentAccepts_network_windows (hP : PeriodAcceptMeans) {s : Sent} (h : ComponentAccepts .network s) :
    ∃ p c e, NetworkPeriod s p c e ∧ PeriodWindowsAccepted p c e :=
  let ⟨p, c, e, hn, pub, prof, hacc⟩ := componentAccepts_network h
  ⟨p, c, e, hn, hP p c e pub prof hacc⟩

end Verity
