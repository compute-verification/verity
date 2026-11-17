module

public import Definitions.Pouw.Confidence
public import Definitions.Pouw.Basic.Prob

/-! PoUW's work on one node on the actual-circuit model, with the strategy fixed before the verifier's live coins. -/

@[expose] public section

namespace ResourceAccounting.PouwWork

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

end ResourceAccounting.PouwWork
