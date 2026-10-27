import Definitions.Accounting.Confidence
import Definitions.Pouw.Basic.Prob

/-!
# PoUW's work on one node, on the actual circuit (trusted)

The PoUW half of the converged GPU model (v3.1), on the
actual-circuit model rather than a device model of its own. A strategy `S` maps the verifier's live coins `r : Ω`
(uniform over a finite `Ω`, `Pouw.pr`) to a run (`Actual.Run`); the node is a location `X`, and the verifier sits at the
leaves `V`. The coins are the verifier's, so the strategy, fixed before them, is the prover's circuit for every value
they take; each coin reaches the node only through what crosses from `V`, at the tick the verifier sends it.

* `Verdict`: the verifier's side, the window `[ts, td)`, the `N` blocks, how it decodes block `k`'s word from what
  crossed into `V` before the deadline `td` (`word`), the right word at the coins (`target`), and its rule (`accept`),
  which reads the coins and everything that crossed into `V`.
* `good`, `goodAll`: the blocks whose word is right at the coin, and those right at every coin (v3.1's `goodAt`,
  `goodAll`).
* `credits X π m`: the node's own computed gates in the window outside a marking `m`, priced at the credit pricing `π`
  (an instruction class's credit, the most target multiply-adds it can stand for), `Placed.work` over `Placed.region`.
* `help X`: the bits the node and the verifier read, in the window, from gates placed outside both, `in_{X ∪ V}`
  (`Placed.ingress`): every word that reaches the node, or the verifier directly, from any other machine. The
  verifier's coins are placed at `V`, so they are no help; nor is what the node itself holds (its inputs and its own
  gates).
-/

namespace Accounting.PouwWork

open Finset

/-- What crosses into a set of leaves: each wire as its arrival tick, its reader's leaf and its source as seen
(`Actual.Run.crossing`). -/
abbrev Inbox (L : Type*) : Type _ := Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ)

/-- **PoUW's verdict** at the verifier's leaves `V`, on coins `Ω`: the window `[ts, td)`, the `N` blocks, block `k`'s
word as the verifier decodes it from what crossed into `V` before `td` (`word`), its right value at the coins
(`target`), and the acceptance rule on the coins and what crossed into `V` (`accept`). `ts` and `td` are the
verifier's ticks, never the prover's. -/
structure Verdict (Ω : Type) (L : Type*) where
  V : Finset L
  ts : ℕ
  td : ℕ
  N : ℕ
  word : Inbox L → ℕ → ℕ
  target : Ω → ℕ → ℕ
  accept : Ω → Inbox L → Prop

variable {Ω : Type} {L : Type*} (S : Ω → Actual.Run L) (P : Verdict Ω L)

/-- What crossed into the verifier's leaves at coin `r`, read during `I`. -/
def inbox (I : Set ℕ) (r : Ω) : Inbox L := (S r).crossing (↑P.V)ᶜ ↑P.V I

/-- The verifier accepts at coin `r`. -/
def Accepts (r : Ω) : Prop := P.accept r (inbox S P Set.univ r)

open Classical in
/-- The blocks whose word, as decoded from what crossed into `V` before `td`, is right at coin `r`. -/
noncomputable def good (r : Ω) : ℕ :=
  ((range P.N).filter fun k => P.word (inbox S P (Set.Iio P.td) r) k = P.target r k).card

open Classical in
/-- The blocks whose word is right at every coin: what the work floor counts. -/
noncomputable def goodAll : ℕ :=
  ((range P.N).filter fun k => ∀ r, P.word (inbox S P (Set.Iio P.td) r) k = P.target r k).card

/-- The node `X`'s credits at coin `r`: the price at `π r` of its computed gates placed in the window, those of the
marking `m r` left out. -/
noncomputable def credits (X : Finset L) (π : ∀ r, Fin (S r).A.N → ℕ) (m : ∀ r, Finset (Fin (S r).A.N)) (r : Ω) :
    ℕ :=
  (S r).A.work (π r) ((S r).A.region X (Set.Ico P.ts P.td) \ m r)

open Classical in
/-- The help the node `X` takes in at coin `r`: the width of the gates placed outside `X` and `V` that gates of `X` or
`V` read during the window, `in_{X ∪ V}([ts, td))`. -/
noncomputable def help (X : Finset L) (r : Ω) : ℕ := (S r).A.ingress (X ∪ P.V) (Set.Ico P.ts P.td)

end Accounting.PouwWork
