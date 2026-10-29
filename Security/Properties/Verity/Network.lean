import Properties.Verity.Game
import Properties.NetworkCertifier.Period

/-!
# The network certifier's component, decoded: what `Verity.ComponentAccepts k .network` gives

`Verity.ComponentAccepts k .network s` (composite `k`'s network component accepts its own case in what the prover sent;
only onsite has one, `NetworkCheck.Period.protocol`, a protocol that reads the blobs sent: in the clock form one-stage
v2 runs on each link-window's proof on them) yields the period that case decodes to (`NetworkPeriod`) and the period
check's acceptance of it on the blobs sent (`componentAccepts_network`), which is
`Properties.NetworkCertifier.PeriodAcceptMeans`'s hypothesis as it stands. With `PeriodAcceptMeans`, every link-window
of that period was accepted on those blobs, in the clock form its proof by v2 (`componentAccepts_network_windows`), as
`Verity.Subprotocol.NetworkWindowLe` (`window_le`) reads it at the compiled verdict.
-/

namespace Verity

open Properties.NetworkCertifier
open Core (canonicalJson)

/-- **The network certifier's case in what the prover sent composite `k` decodes to the period `p`, `c`, `e`**: the
three byte strings decode, and the network component's case among `k`'s components, on the blobs sent, decodes under
the parameters `k` names for it. -/
def NetworkPeriod (k : Verity.Registration.Composite) (s : Sent) (p : NetworkCheck.Period.Params)
    (c : NetworkCheck.Period.Commitment) (e : NetworkCheck.Period.Evidence) : Prop :=
  ∃ ps b ev, Decoded s ps b ev ∧
    ∃ x ∈ Properties.Composition.triples (s.components k) ps ev,
      x.1 = NetworkCheck.Period.protocol.onBlobs (Core.Blobs.ofArray s.blobs) ∧
      (canonicalJson x.2.1.params.toUTF8 >>= NetworkCheck.Period.paramsOf) = .ok p ∧
        NetworkCheck.Period.commitmentOf x.2.2.commitment = .ok c ∧
          NetworkCheck.Period.evidenceOf x.2.2.evidence = .ok e

/-- **A composite's network component accepting is the period check accepting the period its case decodes to, on the
blobs sent**: only onsite has one, and its case's check is the period check's on those blobs. -/
theorem componentAccepts_network {k : Verity.Registration.Composite} {s : Sent} (h : ComponentAccepts k .network s) :
    ∃ p c e, NetworkPeriod k s p c e ∧ ∃ pub prof,
      NetworkCheck.Period.check p c e (Core.Blobs.ofArray s.blobs) = .accept pub prof := by
  obtain ⟨P, hP, ps, b, ev, hd, x, hx, h1, hda⟩ := h
  have hP' : P = NetworkCheck.Period.protocol := by
    cases k
    · exact (Option.some.inj hP).symm
    · cases hP
  subst hP'
  rw [h1] at hda
  obtain ⟨p, c, e, hp, hc, he, hacc⟩ := hda
  exact ⟨p, c, e, ⟨ps, b, ev, hd, x, hx, h1, hp, hc, he⟩, hacc⟩

/-- **Period acceptance at a composite's network component**: given `PeriodAcceptMeans`, when composite `k`'s network
component accepts, its case decodes to a period whose every link-window was accepted in the parameters' form on the
blobs sent. -/
theorem componentAccepts_network_windows (hP : PeriodAcceptMeans) {k : Verity.Registration.Composite} {s : Sent}
    (h : ComponentAccepts k .network s) :
    ∃ p c e, NetworkPeriod k s p c e ∧ PeriodWindowsAccepted p c e (Core.Blobs.ofArray s.blobs) :=
  let ⟨p, c, e, hn, pub, prof, hacc⟩ := componentAccepts_network h
  ⟨p, c, e, hn, hP p c e _ pub prof hacc⟩

end Verity
