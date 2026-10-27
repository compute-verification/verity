import Properties.Verity.Game
import Properties.NetworkCertifier.Period

/-!
# The network certifier's component at onsite, decoded: what `Verity.ComponentAccepts .network` gives

`Verity.ComponentAccepts .network s` (onsite's network component, `NetworkCheck.Period.protocol`, a protocol that reads
no blobs, so the same check on whatever blobs were sent, accepts its own case in what the prover sent) yields one decoded
case, parameters, commitment and evidence, that the period check accepts (`componentAccepts_network`), which is
`Properties.NetworkCertifier.PeriodAcceptMeans`'s hypothesis as it stands. With
`PeriodAcceptMeans` (ONE-13), every link-window of that period was accepted (`componentAccepts_network_windows`), as
`Verity.Legs.NetworkWindowLe` (ONE-12's `window_le`) reads it at the compiled verdict.
-/

namespace Verity

open Properties.NetworkCertifier
open Core (canonicalJson)

/-- **Onsite's network component accepting is the period check accepting a decoded case.** -/
theorem componentAccepts_network {s : Sent} (h : ComponentAccepts .network s) :
    ∃ θ b ev, Properties.Composition.DecodedAccepted NetworkCheck.Period.protocol θ b ev := by
  obtain ⟨_, _, _, _, _, _, _, x, _, hx, hd⟩ := h
  rw [hx] at hd
  exact ⟨_, _, _, hd⟩

/-- **ONE-13 at onsite**: given `PeriodAcceptMeans`, when onsite's network component accepts, its case decodes to a
period whose every link-window was accepted in the parameters' form. -/
theorem componentAccepts_network_windows (hP : PeriodAcceptMeans) {s : Sent} (h : ComponentAccepts .network s) :
    ∃ (θ b ev : ByteArray) (p : NetworkCheck.Period.Params) (c : NetworkCheck.Period.Commitment)
      (e : NetworkCheck.Period.Evidence),
      (canonicalJson θ >>= NetworkCheck.Period.paramsOf) = .ok p ∧ NetworkCheck.Period.commitmentOf b = .ok c ∧
        NetworkCheck.Period.evidenceOf ev = .ok e ∧ PeriodWindowsAccepted p c e :=
  let ⟨θ, b, ev, p, c, e, hp, hc, he, pub, prof, hacc⟩ := componentAccepts_network h
  ⟨θ, b, ev, p, c, e, hp, hc, he, hP p c e pub prof hacc⟩

end Verity
