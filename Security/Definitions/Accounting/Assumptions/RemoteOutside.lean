import Definitions.Accounting.Remote.Outside

/-!
# No device outside the node computes for it, as the network certifiers observe it (trusted)

A relay prover forwards each window's coin from a thin node to a helper outside it, which
holds the encoded store and computes the receipts. No block crosses the node's boundary links, so `Identity` holds and
the links' charge (`Links`) is untouched, yet the node holds nothing. What excludes it is where the receipts'
computation ran. `SketchBound` used to carry that inside its derivation (the sketch model's `Present`); here it is split
into what the receipts' sketch shows wherever the computation ran, and what the auditor's devices see of where it ran,
each a named `Prop` on a trace, the session's record of computation off the node for it
(`Accounting.Remote.Held.Outside`) and what entered the node over its uplinks (`Accounting.Remote.Held.Ingress`):

* `SketchBoundAt O δ`: except with probability `δ`, a correct receipt of a registered block was computed while the
  block was in node memory or on a chip, or while a device outside the node computed on it for the node, during its
  pass. The relay meets it: its helper computed on every block it receipted, and sent the receipts in.
* `Covered O I` (`certifier-coverage`): on every accepting outcome where nothing but the verifier's messages entered
  the node over its uplinks during the credited passes, no device outside the node computed for it then. It is what
  the provider's lockdown (`ProviderLockdown`) became, with the network certifiers in its place, and it says what the
  certifiers cover and what they don't.
* `IngressCounted I δ` (the certifiers' count, at δ_net): except with probability `δ` when the audit accepts, where the
  certifiers sign nothing but the verifier's messages entering the node over the credited passes, nothing else
  entered.

In the per-party names, `Covered` is `Covered` (every uplink from the prover's server passes through an active
certifier) and `IngressCounted` is `Faithful` of the certifiers (what they sign is what crossed their links); neither
names the prover, whose strategy is arbitrary. The line reads no `Secret`, `Confined` or `WitnessPathPrivate`: those
keep what the certifiers sign from the auditor, and the bits-read bound needs only that it was counted.

Statement C in full, no device outside the node computing on a registered block or a receipt during a credited pass,
was a premise of the memory line (`NoOutsideCompute`), on the prover's own conduct. It is now the event the certifiers
observe (`Quiet T I.counted`), inside the line's event, and their count's error δ_net in its bound. The three give the
sketch bound where the certifiers observed nothing else entering (`Accounting.SecurityProofs.Remote.sketchBound_of_covered`)
and `SketchBound` gives `SketchBoundAt` at every `O`. The relay breaks no premise: its receipts come in from the
helper, so the certifiers count them and the line credits that outcome nothing. How the split stands to the sketch
model's premises, under which `SketchBound` is derived (`Accounting.Guarantees.Remote.SketchBoundSession`):

* `Present` is the block clause over the passes together with the same statement without a location (`PresentAt`, in
  the proofs: the block obtained whole was on the node or computed on outside it for the node at a tick of the pass).
  `Present` gives `PresentAt`, and `PresentAt` with `Covered`, where nothing else entered on any accepting outcome,
  gives `Present` where the model's passes are among the trace's credited ones (`present_of_covered`,
  `presentAt_of_present`).
* `KeyBudget` states no location and needs none of this: it bounds pass `j`'s receipt-hash queries by `qtot j`, and
  `Σ qtot` is within the compressions the prover makes on the node or off it (`Q = F·C`, `timing.sketch_failure`'s
  `outside`), so it holds wherever the grinding ran.
-/

namespace Accounting.Remote.Held.Assumptions

open Accounting.Remote.Held

variable {Ω : Type}

/-- **`certifier-coverage`** (every uplink of the node passes through a network certifier): on every accepting outcome
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
  the verifier being the auditor's devices (`trusted-host`). -/
def Covered (T : Trace Ω) (O : Outside T) (I : Ingress Ω) : Prop :=
  ∀ ω, T.accepts ω → Quiet T I.entered ω → ∀ p < T.passes, ∀ t, T.tick ω p ≤ t → t < T.tick ω (p + 1) →
    O.blocks ω t = ∅ ∧ O.receipts ω t = ∅

variable [Fintype Ω]

/-- **The certifiers' count of what enters, at δ_net** (`trusted-host`: each certifier is the auditor's device, running
its stated code): except with probability `δ` when the audit accepts, where the network certifiers on the node's uplinks
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

end Accounting.Remote.Held.Assumptions
