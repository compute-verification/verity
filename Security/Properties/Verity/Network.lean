import Properties.Verity.Legs
import Properties.NetworkCertifier.Period

/-!
# The network certifier's component at onsite, decoded: what `Verity.ComponentAccepts .network` gives

`Verity.ComponentAccepts .network s` (onsite's network component, `NetworkCheck.Period.protocol`, accepts its own case
in what the prover sent) yields one decoded case, parameters, commitment and evidence, that the period check accepts
(`componentAccepts_network`), which is `Properties.NetworkCertifier.PeriodAcceptMeans`'s hypothesis as it stands. With
`PeriodAcceptMeans` (ONE-13), the period that case decodes to (`NetworkPeriod`) accepted every link-window
(`componentAccepts_network_windows`), as `Verity.Legs.NetworkWindowLe` (ONE-12's `window_le`) reads it at the compiled
verdict. Conversely, a decoded period the check accepts is the component accepting (`componentAccepts_network_of`).

`Vocab.NetworkBoundBy acc` is the network component's bound (`Vocab.ComponentBound .network`) with the compiled
component's acceptance replaced by `acc`; at `ComponentAccepts .network` it is the bound itself
(`componentBound_network_by`). Its honest and nontrivial lemmas are `Verity.networkComponentBound_honest` and
`Verity.networkComponentBound_nontrivial` (`Proofs.Properties.Verity.Network.Bound`).
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

/-- **A decoded period the check accepts is onsite's network component accepting**: when what the prover sent
decodes to the period `p`, `c`, `e` (`NetworkPeriod`) and the period check accepts it, `ComponentAccepts .network`. -/
theorem componentAccepts_network_of {s : Sent} {p : NetworkCheck.Period.Params} {c : NetworkCheck.Period.Commitment}
    {e : NetworkCheck.Period.Evidence} (hn : NetworkPeriod s p c e)
    (hacc : ∃ pub prof, NetworkCheck.Period.protocol.check p c e = .accept pub prof) : ComponentAccepts .network s := by
  obtain ⟨ps, b, cs, hps, hb, hcs, x, hx, h1, hp, hc, he⟩ := hn
  refine ⟨ps, b, cs, hps, hb, hcs, x, hx, h1, ?_⟩
  rw [h1]
  exact ⟨p, c, e, hp, hc, he, hacc⟩

namespace Vocab

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **The network component's bound at an acceptance `acc`**: `ComponentBound .network` with `acc` in place of the
compiled component's acceptance. At all `θ`, `F` and `c`, and for every strategy, the probability that `acc` holds of
what the prover sent, every release the network component's check read is in the device's log, more than `K` bits
went through the network certifiers, and there is no break, is at most the network certifier's published error. -/
def NetworkBoundBy (acc : Sent → Prop) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => acc o.sent ∧ (∀ r ∈ V.coinsRead .network o.sent, r ∈ o.log) ∧
        ¬ V.Holds θ F c σ .network o ∧ ¬ V.Break θ F c o) (V.game θ F c) σ ≤ V.delta θ .network

theorem componentBound_network_by : V.ComponentBound .network ↔ V.NetworkBoundBy (ComponentAccepts .network) :=
  Iff.rfl

end Vocab

end Verity
