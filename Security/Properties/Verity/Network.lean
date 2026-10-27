import Properties.Verity.Game
import Properties.NetworkCertifier.Period

/-!
# The network certifier's component at onsite, decoded: what `Verity.ComponentAccepts .network` gives

`Verity.ComponentAccepts .network s` (onsite's network component, `NetworkCheck.Period.protocol`, a protocol that reads
no blobs, so the same check on whatever blobs were sent, accepts its own case in what the prover sent) yields one decoded
case, parameters, commitment and evidence, that the period check accepts (`componentAccepts_network`), which is
`Properties.NetworkCertifier.PeriodAcceptMeans`'s hypothesis as it stands. With
`PeriodAcceptMeans` (ONE-13), every link-window of that period was accepted (`componentAccepts_network_windows`), as
`Verity.Legs.NetworkWindowLe` (ONE-12's `window_le`) reads it at the compiled verdict. Conversely, a period that what
the prover sent decodes to (`NetworkPeriod`) and that the period check accepts is the component accepting
(`componentAccepts_network_of`).

`Vocab.NetworkSoundBy acc` is the network certifier's target (`Vocab.NetworkCertifierSound`) with `acc` in place of the
compiled component's acceptance in its verdict; at `ComponentAccepts .network` it is the target itself
(`Vocab.networkCertifierSound_by`). Its honest and nontrivial lemmas are `Verity.networkComponentBound_honest` and
`Verity.networkComponentBound_nontrivial` (`Proofs.Properties.Verity.Network`).
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

/-- **The period onsite's network case decodes to**: what the prover sent decodes under onsite's registered
composition, one of the components' cases is the network certifier's on the blobs sent, and its parameters, commitment
and evidence decode to `p`, `c` and `e`. -/
def NetworkPeriod (s : Sent) (p : NetworkCheck.Period.Params) (c : NetworkCheck.Period.Commitment)
    (e : NetworkCheck.Period.Evidence) : Prop :=
  ∃ n ps b d k ev, Decoded s n ps b d k ev ∧
    ∃ x ∈ Properties.Composition.triples s.components ps ev,
      x.1 = (component .network).onBlobs (Core.Blobs.ofArray s.blobs) ∧
      (canonicalJson x.2.1.params.toUTF8 >>= NetworkCheck.Period.paramsOf) = .ok p ∧
      NetworkCheck.Period.commitmentOf x.2.2.commitment = .ok c ∧ NetworkCheck.Period.evidenceOf x.2.2.evidence = .ok e

/-- **A decoded period the check accepts is onsite's network component accepting**: when what the prover sent
decodes to the period `p`, `c`, `e` (`NetworkPeriod`) and the period check accepts it, `ComponentAccepts .network`. -/
theorem componentAccepts_network_of {s : Sent} {p : NetworkCheck.Period.Params} {c : NetworkCheck.Period.Commitment}
    {e : NetworkCheck.Period.Evidence} (hn : NetworkPeriod s p c e)
    (hacc : ∃ pub prof, NetworkCheck.Period.protocol.check p c e = .accept pub prof) : ComponentAccepts .network s := by
  obtain ⟨n, ps, b, d, k, ev, hdec, x, hx, h1, hp, hc, he⟩ := hn
  refine ⟨n, ps, b, d, k, ev, hdec, x, hx, h1, ?_⟩
  rw [h1]
  exact ⟨p, c, e, hp, hc, he, hacc⟩

namespace Vocab

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **The network certifier's target at an acceptance `acc`**: `NetworkCertifierSound` with `acc` in place of the
compiled component's acceptance in the network verdict. At all `θ`, `F` and `c`, and for every strategy, the
probability that the committed blob passed the file check, NCI accepted the registration, `acc` holds of what the
prover sent, and more than `K` bits went through the network certifiers, is at most the network certifier's error at
that strategy. -/
def NetworkSoundBy (acc : Sent → Prop) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => (V.Registered o.params o.file ∧ RegistrationAccepts o.sent ∧ acc o.sent) ∧
        ¬ V.Holds θ F c σ .network o) (V.game θ F c) σ ≤ V.error .network θ F c σ

theorem networkCertifierSound_by : V.NetworkCertifierSound ↔ V.NetworkSoundBy (ComponentAccepts .network) :=
  Iff.rfl

end Vocab

end Verity
