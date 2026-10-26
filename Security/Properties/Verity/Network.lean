import Properties.Verity.Game
import Properties.NetworkCertifier.Period

/-!
# The network certifier's component at onsite, decoded: what `Verity.ComponentAccepts .network` gives

`Verity.ComponentAccepts .network s` (onsite's network component, `NetworkCheck.Period.protocol`, accepts its own case
in what the prover sent) yields one decoded case, parameters, commitment and evidence, that the period check accepts
(`componentAccepts_network`), which is `Properties.NetworkCertifier.PeriodAcceptMeans`'s hypothesis as it stands. With
`PeriodAcceptMeans` (ONE-13), the period that case decodes to (`NetworkPeriod`) accepted every link-window
(`componentAccepts_network_windows`), as `Verity.Legs.NetworkWindowLe` (ONE-12's `window_le`) reads it at the compiled
verdict.
-/

namespace Verity

open Properties.NetworkCertifier
open Core (canonicalJson)

/-- **Onsite's network component accepting is the period check accepting a decoded case.** -/
theorem componentAccepts_network {s : Sent} (h : ComponentAccepts .network s) :
    ∃ θ b ev, Properties.Composition.DecodedAccepted NetworkCheck.Period.protocol θ b ev :=
  let ⟨_, _, _, _, _, _, _, _, hx, hd⟩ := h
  ⟨_, _, _, hx ▸ hd⟩

/-- **The period onsite's network case decodes to**: what the prover sent decodes to onsite's cases, one of them is
the network certifier's, and its parameters, commitment and evidence decode to `p`, `c` and `e`. -/
def NetworkPeriod (s : Sent) (p : NetworkCheck.Period.Params) (c : NetworkCheck.Period.Commitment)
    (e : NetworkCheck.Period.Evidence) : Prop :=
  ∃ (ps : List Core.Composition.Named) (b : ByteArray) (cs : List Core.Composition.Case),
    (canonicalJson s.params >>= Core.Composition.decodeParams) = .ok ps ∧
      Core.Composition.decodeCommitment s.commitment = .ok b ∧ Core.Composition.decodeEvidence s.evidence = .ok cs ∧
      ∃ x ∈ Properties.Composition.triples Core.Onsite.components ps cs, x.1 = NetworkCheck.Period.protocol ∧
        (canonicalJson x.2.1.params.toUTF8 >>= NetworkCheck.Period.paramsOf) = .ok p ∧
        NetworkCheck.Period.commitmentOf x.2.2.commitment = .ok c ∧
        NetworkCheck.Period.evidenceOf x.2.2.evidence = .ok e

/-- **ONE-13 at onsite**: given `PeriodAcceptMeans`, when onsite's network component accepts, its case decodes to a
period whose every link-window was accepted in the parameters' form. -/
theorem componentAccepts_network_windows (hP : PeriodAcceptMeans) {s : Sent} (h : ComponentAccepts .network s) :
    ∃ p c e, NetworkPeriod s p c e ∧ PeriodWindowsAccepted p c e := by
  obtain ⟨ps, b, cs, hps, hb, hcs, x, hx, h1, hd⟩ := h
  change x.1 = NetworkCheck.Period.protocol at h1
  rw [h1] at hd
  obtain ⟨p, c, e, hp, hc, he, pub, prof, hacc⟩ := hd
  exact ⟨p, c, e, ⟨ps, b, cs, hps, hb, hcs, x, hx, h1, hp, hc, he⟩, hP p c e pub prof hacc⟩

end Verity
