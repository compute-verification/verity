import Definitions.Accounting.Poub.Outside

/-!
# No device outside the node computes on the store: statement C in full (trusted)

A relay prover forwards each window's coin from a thin node to a helper outside it, which
holds the encoded store and computes the receipts. No block crosses the node's boundary links, so `Identity` holds and
the links' charge (`Links`) is untouched, yet the node holds nothing. What excludes it is where the receipts'
computation ran. `SketchBound` used to carry that inside its derivation (the sketch model's `Present`); here it is split
into what the receipts' sketch shows wherever the computation ran, and where it ran, each a named `Prop` on a trace and
the session's record of computation off the node (`Accounting.Poub.Held.Outside`):

* `NoOutsideCompute O` (statement C in full; the trace's `Identity` is its links clause): on every accepting outcome,
  no device outside the node computes on a registered block or a receipt during a credited pass. It is a premise of the
  memory line (`Properties.PoubCheck.Line.LineHeld`), and the line says it is conditional on it until certifiers
  cover the gate node's boundary and certify that nothing but the verifier's messages entered the node over the
  credited span. What grounds it for a session run inside the provider's lockdown of the node is
  `ProviderLockdown`, under `trusted-host`, which says what the lockdown covers and what it doesn't.
* `SketchBoundAt O δ`: except with probability `δ`, a correct receipt of a registered block was computed while the block
  was in node memory or on a chip, or while a device outside the node computed on it, during its pass. The relay meets
  it: its helper computed on every block it receipted.

The two give `SketchBound` (`Accounting.SecurityProofs.Poub.sketchBound_of_noOutside`) and `SketchBound` gives
`SketchBoundAt` at every `O`, so the line under the two is as strong as under `SketchBound`, and the relay breaks only
`NoOutsideCompute`. How C stands to the sketch model's premises, under which `SketchBound` is derived
(`Accounting.Guarantees.Poub.SketchBoundSession`):

* `Present` is C's block clause over the passes together with the same statement without a location (`PresentAt`, in
  the proofs: the block obtained whole was on the node or computed on outside it at a tick of the pass). `Present`
  gives `PresentAt`, and `PresentAt` with C gives `Present` where the model's passes are among the trace's credited ones
  (`present_of_noOutside`, `presentAt_of_present`).
* `KeyBudget` states no location and needs none of C: it bounds pass `j`'s receipt-hash queries by `qtot j`, and
  `Σ qtot` is within the compressions the prover makes on the node or off it (`Q = F·C`, `timing.sketch_failure`'s
  `outside`), so it holds wherever the grinding ran. What C adds is where the computation on a block's sectors (the
  block clause) and on its receipts (the receipts clause) ran.
-/

namespace Accounting.Poub.Held.Assumptions

open Accounting.Poub.Held

variable {Ω : Type}

/-- **No outside compute** (statement C in full): on every accepting outcome, at every tick of every credited pass, no
device outside the node computes on a registered block or on a receipt of one. -/
def NoOutsideCompute (T : Trace Ω) (O : Outside T) : Prop :=
  ∀ ω, T.accepts ω → ∀ p < T.passes, ∀ t, T.tick ω p ≤ t → t < T.tick ω (p + 1) →
    O.blocks ω t = ∅ ∧ O.receipts ω t = ∅

/-- **`trusted-host`** (the provider's lockdown): on every accepting outcome, at every tick of `[applied,
lifted)`, an interval over which the host provider applied a lockdown to the node, no device outside the node
computes on a registered block or on a receipt of one. A lockdown is the provider's security group on the node's
network interfaces admitting packets only from and to its allow list, the auditor's devices (the verifier's and the
challenger's hosts); its records are infra's, the rules before, during and after, with the times applied and lifted.

What the lockdown covers: every packet between the node and another host that crosses the provider's network and that
a security group evaluates. What it doesn't, and so what this assumes besides the provider enforcing its rules:

* name resolution through the provider's resolver (a link-local address), which security groups don't evaluate:
  a channel out of the node;
* stateful flows: a connection opened before the lockdown was applied, which a stateful rule may keep admitting;
* the provider's own paths into the node: its metadata service, its management plane, a GPU cluster's fabric between
  hosts, and any shared filesystem the node mounts;
* the allow list's hosts themselves, which are the auditor's devices (the rest of `trusted-host`).

So it states the part of `NoOutsideCompute` a lockdown grounds: `[applied, lifted)` are on the verifier's clock, and
the line grounds `NoOutsideCompute` by it only for a session whose every credited pass lies inside one applied
interval (`Accounting.SecurityProofs.Poub.noOutsideCompute_of_lockdown`; `auditor_line.py`'s `--lockdown`, which
holds the session's signed span to the records' times at whole seconds, strictly inside). -/
def ProviderLockdown (T : Trace Ω) (O : Outside T) (applied lifted : ℕ) : Prop :=
  ∀ ω, T.accepts ω → ∀ t, applied ≤ t → t < lifted → O.blocks ω t = ∅ ∧ O.receipts ω t = ∅

variable [Fintype Ω]

/-- **The receipts' sketch bound, wherever computed**: except with probability `δ` when the audit accepts, every correct
receipt of a registered block in a credited pass was computed while the block was in node memory or on a chip, or while
a device outside the node computed on it, during the pass. -/
def SketchBoundAt (T : Trace Ω) (O : Outside T) (δ : ℝ) : Prop :=
  T.Holds δ fun ω => ∀ p < T.passes, ∀ j ∈ T.correct ω p,
    ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t ∪ O.blocks ω t

end Accounting.Poub.Held.Assumptions
