module

public import Definitions.Poub.Remote.Outside

/-! That no device outside the node computes for it, as the network attesters observe it: `SketchBoundAt`, `Covered`
and `IngressCounted`. -/

@[expose] public section

namespace ResourceAccounting.Remote.Held.Assumptions

open ResourceAccounting.Remote.Held

variable {Ω : Type}

/-- **`certifier-coverage`** (every uplink of the node passes through a network attester): on every accepting outcome
where nothing but the verifier's messages entered the node over its uplinks during the credited passes, at every tick
of every credited pass no device outside the node computes for it on a registered block or on a receipt of one. A
computation off the node reaches the receipts the verifier takes only by coming into the node during the passes
(`Outside`), and every path in from another host is an uplink a certifier sits on, so with nothing else entering
nothing off the node computed for it.

It is what `ProviderLockdown T O applied lifted` became: the provider's security group, admitting packets only from and
to the auditor's devices over `[applied, lifted)`, is replaced by certifiers on every uplink, which count what enters
(`Ingress.counted`, observed in the line's event) rather than being trusted to block it, so the lockdown's interval
and its records go. The certifiers see every frame on an uplink, so two of the lockdown's gaps close: name resolution
over an uplink, and a flow opened before the lockdown, which a stateful rule kept admitting. What this assumes besides
certifiers on every uplink, the gaps left:

* the paths the node's host serves without an uplink: the provider's resolver and metadata service at link-local
  addresses, and its management plane;
* a GPU cluster's fabric between hosts and a shared filesystem the node mounts, unless a certifier sits on each;
* the verifier taking the session's messages from the node alone, over the session's channel, and the certifiers and
  the verifier being the auditor's devices (`trusted-device`). -/
def Covered (T : Trace Ω) (O : Outside T) (I : Ingress Ω) : Prop :=
  ∀ ω, T.accepts ω → Quiet T I.entered ω → ∀ p < T.passes, ∀ t, T.tick ω p ≤ t → t < T.tick ω (p + 1) →
    O.blocks ω t = ∅ ∧ O.receipts ω t = ∅

variable [Fintype Ω]

/-- **The certifiers' count of what enters, at δ_net** (`trusted-device`: each certifier is the auditor's device, running
its stated code): except with probability `δ` when the audit accepts, where the network attesters on the node's uplinks
sign nothing but the verifier's messages entering it over the credited passes, nothing else entered over them. `δ` is
δ_net, the certifiers' error at the session's boundary: their sampled audit's `δ + ε_ks + ε_bind`
(`NetTiming.Guarantees.SampledAuditAccepted`) at the ingress windows covering the credited span. -/
def IngressCounted (T : Trace Ω) (I : Ingress Ω) (δ : ℝ) : Prop :=
  T.Holds δ fun ω => Quiet T I.counted ω → Quiet T I.entered ω

/-- **The receipts' sketch bound, wherever computed**: except with probability `δ` when the audit accepts, every correct
receipt of a registered block in a credited pass was computed while the block was in node memory or on a chip, or while
a device outside the node computed on it for the node, during the pass. -/
def SketchBoundAt (T : Trace Ω) (O : Outside T) (δ : ℝ) : Prop :=
  T.Holds δ fun ω => ∀ p < T.passes, ∀ j ∈ T.correct ω p,
    ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t ∪ O.blocks ω t

end ResourceAccounting.Remote.Held.Assumptions
