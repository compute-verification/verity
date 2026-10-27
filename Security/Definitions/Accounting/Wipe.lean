import Definitions.Core.Blocks
import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe on the device model (trusted)

The wipe that opens a server-time block (`Verity.Wipe`, `Definitions.Core.Blocks`) when it is PoUS's streamed-label
wipe (`Pous.StreamedWipe`; `verity/core/protocols/pous/PROTOCOL.md`, "The streamed-label wipe"), as `verity/wipe/v0`
runs it: the memory challenger streams `n` labels of 512 bits into the device's store, then reveals `k` spot reads one
at a time and reads each label back under a deadline.

* **The labels** (`Labels n`): the table of `n` labels of 512 bits. The deployed labels are SHA-512's PRF of a key the
  challenger draws and keeps (`prf/sha-512`): `prf K` for a uniform key `K`.
* **The spot reads** (`Reads n k`): `k` indices into the labels, uniform and independent.
* **The prover's strategy** (`Strategy`): any fill of the device's state from its state at the boundary and the labels,
  with any egress, any answer to each spot read from the device's state when the fill ends, the egress, the reads
  revealed before it in order and its own index, and any state it leaves for the block. Nothing constrains it but its
  type. The type is what the auditor's devices enforce, each a premise on one party, as Daniel names them: the memory
  challenger is `Faithful` (it reveals read `j + 1` only once read `j`'s answer arrived on time, so no answer reads a
  later index) and `Secret` (it keeps its key, so nothing of the labels reaches the device but the stream), and the
  site is `Covered` (every bit that leaves the device during the wipe crosses a link a network certifier watches, so the
  outside holds nothing of the labels but the egress, and answers from it and the reads it is sent).
* **The observed event** (`Strategy.Observed`): what the auditor's devices see and sign. Every spot read's answer, as
  the challenger receives it, is its label (`wrong = 0` in the signed session), and the network certifiers' count of the
  egress is at most the bound the verdict publishes. The verdict checks the deadlines and the batch shape too; the event
  leaves them out, so it holds on every run the verdict accepts.
* **The wipe** (`Strategy.wipe`): the strategy as core's `Verity.Wipe`, its seed the challenger's key and its verdict an
  event of the run.
* **The two error terms** a survivor bound pays beyond the wipe's own. The labels' PRF advantage
  (`Pous.StreamedWipe.prfAdvantage`) against the distinguisher the adversary's moves make (`Strategy.distinguisher`: it
  runs the strategy on the labels it is given and outputs whether the run is accepted, keeps the foreign content and
  sends an egress in `A`). The network certifiers' miss, `δ_net` (`Strategy.deltaNet`): the probability that the
  verdict accepts while the egress the strategy sent is outside `A`, a set of at most `2^E` strings for the published
  `E`. The network certifier's guarantee bounds it (`NetTiming.Guarantees.SampledLeakAccepted`: except with probability
  `δ + ε_ks + ε_bind`, the accepted observation sequences number at most `2^bits`).
-/

namespace Accounting.Wipe

open Verity
open scoped ENNReal

/-- The streamed wipe's labels: a table of `n` labels of 512 bits (64 bytes). -/
abbrev Labels (n : ℕ) : Type := Fin n → Pous.Bits 512

/-- The streamed wipe's spot reads: `k` indices into the `n` labels. -/
abbrev Reads (n k : ℕ) : Type := Fin k → Fin n

/-- **A prover's strategy** for the streamed wipe of `n` labels with `k` spot reads, at boundary `b` of the schedule `s`
on the device `d`. Every field is the prover's choice.

* `fill`: the bits of the store the wipe's account counts as filled (`Verity.Wipe.fill`); the rest of the store and the
  on-chip state are counted as surviving.
* `filled σ L`: the device's state when the fill ends, from its state `σ` at the boundary and the labels `L`, and the
  `egress` bits that left the device during the wipe.
* `answer σ' ξ h i`: the answer to spot read `i`, from the device's state `σ'` when the fill ends, the egress `ξ` (what
  the outside holds) and the reads `h` revealed before it, in order.
* `carry` and `after`: as `Verity.Wipe`'s. -/
structure Strategy {ι : Type} (d : Accounting.Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B) (b : Fin B)
    (n k : ℕ) where
  fill : ℕ
  fill_le : fill ≤ d.mem
  egress : ℕ
  carry : Device.State d → s.Carried b
  filled : Device.State d → Labels n → Device.State d × (Fin egress → Bool)
  answer : Device.State d → (Fin egress → Bool) → List (Fin n) → Fin n → Pous.Bits 512
  after : Device.State d → (Fin egress → Bool) → Reads n k → Device.State d

namespace Strategy

variable {ι : Type} {d : Accounting.Device ι} {C : Circuit} {B : ℕ} {s : Blocks C B} {b : Fin B} {n k : ℕ}
  (τ : Strategy d s b n k)

/-- **The observed event** of the run from the device's state `σ` at the boundary, the labels `L` and the spot reads
`c`: every spot read's answer, as the memory challenger receives it, is its label, and the network certifiers' count of
the egress, `count`, is at most `K` bits. -/
def Observed (count : (Fin τ.egress → Bool) → ℕ) (K : ℕ) (σ : Device.State d) (L : Labels n) (c : Reads n k) :
    Prop :=
  (∀ i : Fin k, τ.answer (τ.filled σ L).1 (τ.filled σ L).2 (Pous.StreamedWipe.revealed c i) (c i) = L (c i)) ∧
    count (τ.filled σ L).2 ≤ K

/-- **The strategy as a wipe** of core's model, with the verdict `acc` of the run's state at the boundary, labels and
spot reads: its seed is the challenger's key, and its labels `prf` of it. -/
def wipe (acc : Device.State d → Labels n → Reads n k → Prop) {κ : ℕ} (prf : Pous.Bits κ → Labels n) :
    Verity.Wipe d s b (Pous.Bits κ) (Reads n k) where
  fill := τ.fill
  fill_le := τ.fill_le
  egress := τ.egress
  carry := τ.carry
  filled σ K := τ.filled σ (prf K)
  accepts σ K c := acc σ (prf K) c
  after := τ.after

/-- **The distinguisher the adversary's moves make**, against the labels: with coins the foreign content `z` and the
spot reads `c`, and labels `L`, it runs the strategy from `hold z` and outputs whether `acc` accepts, `keep` gives back
`z`, and the egress is in `A`. -/
def distinguisher (acc : Device.State d → Labels n → Reads n k → Prop) (A : Finset (Fin τ.egress → Bool)) {S : ℕ}
    (hold : Pous.Bits S → Device.State d)
    (keep : Device.State d → (Fin τ.egress → Bool) → Reads n k → Pous.Bits S) :
    (Pous.Bits S × Reads n k) × Labels n → Prop := fun x =>
  acc (hold x.1.1) x.2 x.1.2 ∧ keep (τ.filled (hold x.1.1) x.2).1 (τ.filled (hold x.1.1) x.2).2 x.1.2 = x.1.1 ∧
    (τ.filled (hold x.1.1) x.2).2 ∈ A

/-- **`δ_net`, the network certifiers' miss** at `A`: the probability, over a uniform foreign content `z`, key and spot
reads, that `acc` accepts the run from `hold z` while the egress it sent is outside `A`. -/
noncomputable def deltaNet (acc : Device.State d → Labels n → Reads n k → Prop) {κ : ℕ} (prf : Pous.Bits κ → Labels n)
    (A : Finset (Fin τ.egress → Bool)) {S : ℕ} (hold : Pous.Bits S → Device.State d) : ℝ≥0∞ :=
  Pous.pr fun x : Pous.Bits S × Pous.Bits κ × Reads n k =>
    acc (hold x.1) (prf x.2.1) x.2.2 ∧ (τ.filled (hold x.1) (prf x.2.1)).2 ∉ A

end Strategy

end Accounting.Wipe
