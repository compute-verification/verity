import Definitions.Accounting.Confidence
import Definitions.Pous.Model.M1p

/-!
# The deployed labels over a server's life, and one audit as a distinguisher (trusted)

`Accounting.Assumptions.UniformLabels` draws the labels uniform given the auditor's devices' coins. The deployed labels
are not: wipe `w`'s label of block `i` is SHA-512 of the wipe's key and `i` (PR #2726), and the wipe's key is derived
from the auditor's secret for the server and the wipe (`streamed_wipe.wipe_key`), a PRF of the secrets (claim
`prf/sha-512`). The challenger uses the secrets only to compute labels: it streams them in the fill and compares
answers to them. So a run is a function of the other coins `ω` (the prover's, the auditor's devices'), the lifetime
label table `F : Fin (W * N) → Bits b` of `W` wipes of `N` blocks, and the indices `c`, `S ((ω, F), c)`, and the
deployed system is that run at `F = prf s` for uniform secrets `s`, as PoUS's streamed-label wipe states it for one
wipe (`Pous.StreamedWipe.winReal`, whose prover sees only the labels).

* `tableLabel`: the label a run carries for block `i` at read `j`, the table's entry for read `j`'s wipe `wipe j` and
  `i` (slot `finProdFinEquiv (wipe j, i)`), as a number, bit `k` with weight `2^k` (`Pous.bitsVal`), as
  `Actual.Run.seen` carries a gate's value.
* `heldEvent`: `Accounting.Guarantees.Actual.WipeReadHeld`'s event on the coins: the challenger at `V` accepts every
  read (`Ch j` of its view), each read arrives at a tick of `T`, and at least `K` reads find `X`'s state narrower than
  `m` bits at the arrival (`Actual.Run.mem`). One run decides it.
* `heldDistinguisher`: the distinguisher one audit makes of the system, in `Pous.StreamedWipe.prfAdvantage`'s form:
  with coins `(ω, c)` and a table `F`, it runs `S ((ω, F), c)` once and outputs `heldEvent`. It asks the table only what
  the audits ask it (each wipe's fill streams its labels, each read compares one), and it never reads the secrets.
-/

namespace Accounting

open Finset

namespace Actual

/-- **The lifetime table's labels**: block `i`'s label at read `j`, the entry `F (finProdFinEquiv (wipe j, i))` of read
`j`'s wipe as a number, bit `k` with weight `2^k` (`Pous.bitsVal`). -/
def tableLabel {Ω : Type*} {n W N b : ℕ} (wipe : Fin n → Fin W) (ω₀ : Ω × (Fin (W * N) → Pous.Bits b)) (j : Fin n)
    (i : Fin N) : ℕ :=
  Pous.bitsVal (ω₀.2 (finProdFinEquiv (wipe j, i)))

/-- **The store held read by read, as an event of one run**: the challenger at `V` accepts every read (`Ch j` of its
view, `Actual.Run.view`), each read arrives at a tick of `T`, and at least `K` reads find `X`'s state narrower than
`m` bits at the arrival (`Actual.Run.mem`). It is `Accounting.Guarantees.Actual.WipeReadHeld`'s event. -/
def heldEvent {Ω₀ I L : Type*} {n : ℕ} (S : Ω₀ × I → Actual.Run L) (X V : Finset L)
    (Ch : Fin n → Set (ℕ × L × ℕ × ℕ × ℕ) → Prop) (arrive : Ω₀ × I → Fin n → ℕ) (T : Finset ℕ) (m K : ℕ)
    (r : Ω₀ × I) : Prop :=
  (∀ j, Ch j ((S r).view V) ∧ arrive r j ∈ T) ∧ K ≤ (univ.filter fun j => (S r).mem X (arrive r j) < m).card

/-- **One audit as a distinguisher** of the label table `Fin M → Bits b`: with coins `x.1 = (ω, c)` and table
`x.2 = F`, it runs the system once on the coins `((ω, F), c)` and outputs `heldEvent`. -/
def heldDistinguisher {Ω L : Type*} {n M N b : ℕ} (S : (Ω × (Fin M → Pous.Bits b)) × (Fin n → Fin N) → Actual.Run L)
    (X V : Finset L) (Ch : Fin n → Set (ℕ × L × ℕ × ℕ × ℕ) → Prop)
    (arrive : (Ω × (Fin M → Pous.Bits b)) × (Fin n → Fin N) → Fin n → ℕ) (T : Finset ℕ) (m K : ℕ)
    (x : (Ω × (Fin n → Fin N)) × (Fin M → Pous.Bits b)) : Prop :=
  heldEvent S X V Ch arrive T m K ((x.1.1, x.2), x.1.2)

end Actual

end Accounting
