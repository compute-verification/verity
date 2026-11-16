module

public import Proofs.Properties.Verity.Sampled.Instance
public import Properties.Verity.LedgerPinned

/-!
# The deployed device's ledger lines, signed under a key the registration pins (stated, not proved)

`remote_soundness_deployedReady` reads the challenger's ledger through its instance's device: the field
`Instance.signer` of its argument `I` makes the vocabulary's signer the deployed device's (`Device.Deployment.signer`),
which signs the ledger lines of each play (`Device.Deployment.setupOf`'s `ledger`, its `opens` before any coin and its
`closes` at the close) under the parameters' key (`Device.pubOf`, `Binding.keyOf`), and the coupling reads those lines
as the device's own ledger (`Properties.Challenger.OpensOn` in `Binding.accepted_run`) at the deployed device's history
(`Instance.dev`, which `Instance.constructionsAt` assembles with). Nothing in the theorem says that key is the
auditor's: `hFP` (`Instance.FamiliesPinned`) pins the family the parameters name, not the key.

* `Instance.LedgerLinesPinned`: at every play the sampled verdict accepts, the parameters name a key and every ledger
  line the deployed device signs at that play verifies under it, a key the registration pins
  (`Properties.Verity.LinesPinned`);
* `Instance.LedgerLinesPinnedHonest`: some instance has an accepted play whose device signs at least one ledger line,
  at which `LedgerLinesPinned` holds;
* `Instance.LedgerLinesPinnedNontrivial`: wherever an accepted play's key is one the registration doesn't pin and its
  device signs a line, `LedgerLinesPinned` fails.

None of the three is proved.
-/

@[expose] public section

namespace Verity.Sampled

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

namespace Instance

variable (I : Instance d α)

/-- **Every ledger line the deployed device signs at an accepted play is signed under a key the registration pins**,
at composite `k`, the registration log `log` and the auditor's inputs `A`: at every play the sampled verdict accepts,
the parameters name a challenger key (`Binding.keyOf`), and every line of the device's ledger at that play
(`Device.Deployment.setupOf`'s `ledger`) is a ledger entry whose signature LMS verifies under that key, which the
registration pins (`Properties.Verity.LinesPinned`). Not proved.

It is what would discharge the part of `remote_soundness_deployedReady`'s argument `I` that its field `Instance.signer`
takes as given: that the device whose signatures the play's ledger lines carry, under the parameters' key
(`Device.pubOf`), is the auditor's. The theorem would take it beside `hFP` (`Instance.FamiliesPinned`), which pins the
family the parameters name and reads no key. -/
def LedgerLinesPinned (k : Verity.Registration.Composite) (log : Verity.Registration)
    (A : Verity.Registration.Auditor) : Prop :=
  ∀ θ F c o, Verity.Game.Reaches (I.base.game θ F c) o → I.base.Verdict k .sampled o →
    ∃ key, Binding.keyOf o.params = some key ∧
      Properties.Verity.LinesPinned log A key (I.deployment.setupOf I.base.families o).ledger

/-- **`LedgerLinesPinned`'s honest shape**: some instance, at composite `k`, has a play its sampled verdict accepts at
which the deployed device signs at least one ledger line, and `LedgerLinesPinned` holds of it at some registration log
and auditor's inputs. Not proved. -/
def LedgerLinesPinnedHonest (k : Verity.Registration.Composite) : Prop :=
  ∃ (I : Instance d α) (log : Verity.Registration) (A : Verity.Registration.Auditor),
    ∃ θ F c o, Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o ∧
      (I.deployment.setupOf I.base.families o).ledger ≠ [] ∧ I.LedgerLinesPinned k log A

/-- **`LedgerLinesPinned` is not vacuous**: at every instance, composite `k`, registration log and auditor's inputs,
a play the sampled verdict accepts whose parameters name a key the registration doesn't pin
(`Properties.Verity.KeyPinned`), and at which the deployed device signs a ledger line, makes `LedgerLinesPinned`
false. Not proved. -/
def LedgerLinesPinnedNontrivial (k : Verity.Registration.Composite) : Prop :=
  ∀ (I : Instance d α) (log : Verity.Registration) (A : Verity.Registration.Auditor) θ F c o key,
    Verity.Game.Reaches (I.base.game θ F c) o → I.base.Verdict k .sampled o →
    Binding.keyOf o.params = some key → ¬Properties.Verity.KeyPinned log A key.pub →
    (I.deployment.setupOf I.base.families o).ledger ≠ [] → ¬I.LedgerLinesPinned k log A

end Instance

end Verity.Sampled
