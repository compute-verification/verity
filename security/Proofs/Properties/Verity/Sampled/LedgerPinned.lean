module

public import Proofs.Properties.Verity.Sampled.Instance
public import Properties.Verity.LedgerPinned

/-! The deployed device's ledger lines, signed under a key the registration pins (`LedgerLinesPinned`, with
its honest and nontrivial forms): stated, not proved. -/

@[expose] public section

namespace Verity.Sampled

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

namespace Instance

variable (I : Instance d α)

/-- Every ledger line the deployed device signs at an accepted play verifies under a key the registration
pins. Not proved; it would discharge what `Instance.signer` takes as given in `remote_soundness_deployedReady`: that
the device whose signatures the play's ledger lines carry is the auditor's. -/
def LedgerLinesPinned (k : Verity.Registration.Composite) (log : Verity.Registration)
    (A : Verity.Registration.Auditor) : Prop :=
  ∀ θ F c o, Verity.Game.Reaches (I.base.game θ F c) o → I.base.Verdict k .sampled o →
    ∃ key, Binding.keyOf o.params = some key ∧
      Properties.Verity.LinesPinned log A key (I.deployment.setupOf I.base.families o).ledger

/-- `LedgerLinesPinned`'s honest shape: some instance has an accepted play whose device signs a ledger line
and at which `LedgerLinesPinned` holds. Not proved. -/
def LedgerLinesPinnedHonest (k : Verity.Registration.Composite) : Prop :=
  ∃ (I : Instance d α) (log : Verity.Registration) (A : Verity.Registration.Auditor),
    ∃ θ F c o, Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o ∧
      (I.deployment.setupOf I.base.families o).ledger ≠ [] ∧ I.LedgerLinesPinned k log A

/-- `LedgerLinesPinned` is not vacuous: an accepted play whose key the registration doesn't pin and whose
device signs a ledger line makes it false. Not proved. -/
def LedgerLinesPinnedNontrivial (k : Verity.Registration.Composite) : Prop :=
  ∀ (I : Instance d α) (log : Verity.Registration) (A : Verity.Registration.Auditor) θ F c o key,
    Verity.Game.Reaches (I.base.game θ F c) o → I.base.Verdict k .sampled o →
    Binding.keyOf o.params = some key → ¬Properties.Verity.KeyPinned log A key.pub →
    (I.deployment.setupOf I.base.families o).ledger ≠ [] → ¬I.LedgerLinesPinned k log A

end Instance

end Verity.Sampled
