import Definitions.Accounting.Confidence
import Definitions.Pous.Model.M1p

/-!
# A wipe's deployed labels, and one audit as a distinguisher (trusted)

`Accounting.Assumptions.UniformLabels` draws a wipe's labels uniform given the auditor's devices' coins. The deployed
labels are not: they are SHA-512 of a 32-byte key the memory challenger draws and the block's index (PR #2726), a PRF
(claim `prf/sha-512`). The challenger uses the key only to compute labels: it streams them in the fill and compares
answers to them. So a run is a function of the other coins `ω` (the prover's, the auditor's devices'), the label table
`F : Fin N → Bits b` and the indices `c`, `S ((ω, F), c)`, and the deployed system is that run at `F = prf s` for a
uniform key `s`, as PoUS's streamed-label wipe states it (`Pous.StreamedWipe.winReal`, whose prover sees only the
labels).

* `tableLabel`: the label a run carries for block `i` at any read of the wipe, `F i` as a number, bit `k` with weight
  `2^k` (`Pous.bitsVal`), as `Actual.Run.seen` carries a gate's value.
* `heldEvent`: `Accounting.Guarantees.Actual.WipeReadHeld`'s event on the coins: the challenger at `V` accepts every
  read (`Ch j` of its view), each read arrives at a tick of `T`, and at least `K` reads find `X`'s state narrower than
  `m` bits at the arrival (`Actual.Run.mem`). One run decides it.
* `heldDistinguisher`: the distinguisher one audit makes of the system, in `Pous.StreamedWipe.prfAdvantage`'s form:
  with coins `(ω, c)` and labels `F`, it runs `S ((ω, F), c)` once and outputs `heldEvent`. It asks the labels only what
  the audit asks them (the fill streams every label, each read compares one), and it never reads a key.
-/

namespace Accounting

open Finset

namespace Actual

/-- **The label table's labels**: block `i`'s label at every read of the wipe, `F i` as a number, bit `k` with weight
`2^k` (`Pous.bitsVal`). -/
def tableLabel {Ω : Type*} {n N b : ℕ} (ω₀ : Ω × (Fin N → Pous.Bits b)) (_ : Fin n) (i : Fin N) : ℕ :=
  Pous.bitsVal (ω₀.2 i)

/-- **The store held read by read, as an event of one run**: the challenger at `V` accepts every read (`Ch j` of its
view, `Actual.Run.view`), each read arrives at a tick of `T`, and at least `K` reads find `X`'s state narrower than
`m` bits at the arrival (`Actual.Run.mem`). It is `Accounting.Guarantees.Actual.WipeReadHeld`'s event. -/
def heldEvent {Ω₀ I L : Type*} {n : ℕ} (S : Ω₀ × I → Actual.Run L) (X V : Finset L)
    (Ch : Fin n → Set (ℕ × L × ℕ × ℕ × ℕ) → Prop) (arrive : Ω₀ × I → Fin n → ℕ) (T : Finset ℕ) (m K : ℕ)
    (r : Ω₀ × I) : Prop :=
  (∀ j, Ch j ((S r).view V) ∧ arrive r j ∈ T) ∧ K ≤ (univ.filter fun j => (S r).mem X (arrive r j) < m).card

/-- **One audit as a distinguisher** of the label table: with coins `x.1 = (ω, c)` and labels `x.2 = F`, it runs the
system once on the coins `((ω, F), c)` and outputs `heldEvent`. -/
def heldDistinguisher {Ω L : Type*} {n N b : ℕ} (S : (Ω × (Fin N → Pous.Bits b)) × (Fin n → Fin N) → Actual.Run L)
    (X V : Finset L) (Ch : Fin n → Set (ℕ × L × ℕ × ℕ × ℕ) → Prop)
    (arrive : (Ω × (Fin N → Pous.Bits b)) × (Fin n → Fin N) → Fin n → ℕ) (T : Finset ℕ) (m K : ℕ)
    (x : (Ω × (Fin n → Fin N)) × (Fin N → Pous.Bits b)) : Prop :=
  heldEvent S X V Ch arrive T m K ((x.1.1, x.2), x.1.2)

end Actual

end Accounting
