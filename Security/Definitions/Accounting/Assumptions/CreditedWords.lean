import Definitions.Accounting.JointBridge.CreditedWords

/-!
# A credited word's named premise: its multiply-adds ran on the device's tensor cores (trusted)

The one premise `Properties.Accounting.JointBridge.WorkOfCreditedWords` adds to a PoUW source statement that credits
words (`Properties.Pouw.Period.H1CheckedCreditPeriod`): a `Prop` taken as a hypothesis, never an `axiom`. It has no
card yet; it needs one before a PR that takes it is marked ready.
-/

namespace Accounting.Assumptions

/-- **Words fixed right ran on the tensor cores** (`accounting-words-on-tensor-cores`): on every period PoUW's audit
accepts, the words the source game counts as fixed right (`fixed`, its owned right committed words less the adaptive
ones), at `m` multiply-adds each, are within the compute account's tensor-core multiply-adds over the period's window
(`Accounting.PeriodRun.tensorMacs`). It reads no price.

Why it is a premise: the source game is about the run its play reads (`Accounting.TS.Play.run`), and the device model's
schedule is a separate object; nothing PoUW's check sees ties a word of one to an op of the other. It fails for a
device that computes an owned word's multiply-adds on its CUDA cores (`IDP4A`, priced `p .cuda`, not `p .tc`), on
another device (help a network certifier bounds only in bits), under another account, outside the window, or that
needs fewer than `m` of them for a word (a shortcut the tile's Definition doesn't take). -/
def WordsOnTensorCores {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type}
    [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) (m : ℕ) {X : Type} (coin : Ω (r.own .pouw) ≃ X)
    (acc : ((i : I) → Ω i) → X → Prop) (fixed : ((i : I) → Ω i) → X → ℝ) : Prop :=
  ∀ ξ y, acc ξ (coin y) → (m : ℝ) * fixed ξ (coin y) ≤ (r.tensorMacs (Function.update ξ (r.own .pouw) y) : ℝ)

end Accounting.Assumptions
