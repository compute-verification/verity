module

public import Definitions.Draft.Acceptance
public import Definitions.Pouw.PouwWork

/-!
# Pearl+'s useful work as served, on the actual circuit (draft)

The statement of useful work for Pearl+ at lean's acceptance (`Draft.Acceptance`: `transcript`, `AsWritten`,
`Accepts`), in `Sound`'s shape over every strategy, with `Theorem1`'s event: the node's priced work in the window is
below the credit less `ρ` times the help it took in only on a small share of the challenges. Definitions and statements
only; nothing here is proved but two small probes, and nothing is in a lock.

* **The auditor** (`Auditor`): the challenge device's leaves `Vd`, whose input gates hold the challenge (the salt's cells
  and the sample's cells), and the verifier's leaves `Vv`, which have no input gates and hear the challenge as a message
  (soundness's answer (2), Sat 21 Nov 9:37 PM PST). Ticks: the salt from `ts`, the sample from `tc`, the deadline `td`.
* **The challenge, chosen in one place** (`feasibleAt`): the feasible runs a challenge at a time (soundness's answer
  (1)), the device's input gates holding the challenge's encoding. `SoundAt` is `Sound` over such a family; at a
  constant family it is `Sound` (`soundAt_const`). The other answer (one feasible set, the index bound by the named
  uniform-draw assumption) is a change to `feasibleAt` alone.
* **What the code reads** (`Read`, `CodeReadsCredit`): the verifier's function `vf` and the credit `W` are arguments, as
  the definitions may not load the verifier's package. The one premise about them says: on a transcript `vf` accepts,
  the sampled openings sit at the positions the challenge names and verify against the roots, and `W` is the code's
  credit `cred` at the number `X_O` of opened tiles over the inert-work bound.
* **The tiles committed before the sample** (`committedRight`): defined once, without an extractor, over the strategy's
  own runs: tile `i` is right at `r` when some accepted run sharing `r`'s run before the sample opens it and every such
  run opens it within the tile rule. `Collides` is two such runs opening one position two ways.
* **The work** is `PouwWork.credits` on the node's leaves `X` over the window `[ts, td)` at the registered pricing
  `price`, with no gate left out: the price of every computed gate the node places in the window. **The help** is
  `PouwWork.help`: the width of the gates placed outside `X` and the auditor that `X` or the auditor read in the window.

Vocabulary: transcript, message, challenge, actual circuit, registered information, private.
-/

@[expose] public section

namespace Draft.PearlPlusUsefulWork

open ResourceAccounting Finset Draft.Acceptance

variable {L : Type*} [LinearOrder L]

/-- **The auditor's leaves and ticks.** `Vd` is the auditor-trusted challenge device: its input gates hold the
challenge, the salt in the cells `saltCells` and the sample in the cells `sampleCells`. `Vv` is the verifier, whose
verdict is cell `c` of leaf `y` at the deadline `td`. The salt and the sample are the device's two messages: no gate
reads the salt before `ts` nor the sample before `tc` (`DeviceAsWritten`), and the roots are heard between them. -/
structure Auditor (L : Type*) where
  Vd : Finset L
  Vv : Finset L
  y : L
  c : ℕ
  saltCells : Finset ℕ
  sampleCells : Finset ℕ
  ts : ℕ
  tc : ℕ
  td : ℕ

/-- The auditor's leaves, `Sound`'s `V`. -/
def Auditor.leaves (A : Auditor L) : Finset L := A.Vd ∪ A.Vv

/-- The verifier's transcript up to the deadline (lean's `transcript` at `Vv`): `Vv` has no input gates, so its first
message is empty and the rest is what crossed into `Vv` before `td`, the device's challenge among it. -/
noncomputable def vtranscript (A : Auditor L) (R : Actual.Run L) : List (List Bool) := transcript R A.Vv A.td

/-! ## The challenge: one definition -/

/-- **The one place the challenge is chosen.** The feasible runs at challenge `r`: those of `𝓕` whose device's input
gates hold `enc r`, its encoding (soundness, 9:37 PM PST: the feasible runs a challenge at a time, the challenge in the
device's input gates and heard by the verifier as a message). Were the other answer taken (one feasible set, the
challenge in the view bound to the index by the named uniform-draw assumption `UniformRandomBytes`), this definition
alone changes: to `𝓕`, with the guard moved into the event. -/
def feasibleAt {Ω : Type} (𝓕 : Set (Actual.Run L)) (A : Auditor L) (enc : Ω → List Bool) (r : Ω) :
    Set (Actual.Run L) :=
  {R | R ∈ 𝓕 ∧ challenge R A.Vd = enc r}

open Classical in
/-- **`E` is sound at `δ` against a family of feasible sets**, `𝓕 r` the feasible runs at challenge `r`: `Sound` with
`S r ∈ 𝓕 r` in place of `S r ∈ 𝓕`. It ranges over every strategy whose run at each challenge is feasible at it. -/
def SoundAt (Ω : Type) [Fintype Ω] (𝓕 : Ω → Set (Actual.Run L)) (V : Finset L)
    (acc : Set (ℕ × L × ℕ × ℕ × ℕ) → Prop) (E : (Ω → Actual.Run L) → Ω → Prop) (δ : ℝ) : Prop :=
  ∀ S : Ω → Actual.Run L, (∀ r, S r ∈ 𝓕 r) → CausalFrom S → (∀ r r', (S r).inputs (↑V)ᶜ = (S r').inputs (↑V)ᶜ) →
    ((univ.filter fun r => acc ((S r).view V) ∧ ¬E S r).card : ℝ) ≤ δ * Fintype.card Ω

/-- **Probe: at a constant family `SoundAt` is `Sound`**, so the family form adds no strength of its own. -/
theorem soundAt_const (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V : Finset L)
    (acc : Set (ℕ × L × ℕ × ℕ × ℕ) → Prop) (E : (Ω → Actual.Run L) → Ω → Prop) (δ : ℝ) :
    SoundAt Ω (fun _ => 𝓕) V acc E δ ↔ Sound Ω 𝓕 V acc E δ := Iff.rfl

/-! ## The premises about the feasible runs and the code -/

/-- **Kind 1, the feasible runs' SHA-512 budget.** `q ≤ 2^64`, and every feasible run places at most `q` SHA-512
evaluations before the deadline, by anyone. `shaEvals R t` counts them; the actual circuit has no definition of "one
SHA-512 evaluation" on `main` today, so it is an argument here, and which gates it counts is an open point. A table
from salt to tiles made before the salt is evaluations before the deadline, so it is in `q`. -/
def HashBudget (𝓕 : Set (Actual.Run L)) (A : Auditor L) (shaEvals : Actual.Run L → ℕ → ℕ) (q : ℕ) : Prop :=
  q ≤ 2 ^ 64 ∧ ∀ R ∈ 𝓕, shaEvals R A.td ≤ q

/-- **Kind 2, the device and the verifier as written**, on the feasible runs: the verifier's leaves have no input
gates; no gate reads the salt's cells before `ts` nor the sample's before `tc` (the challenge does not leave the device
early, anywhere, the device's own gates included); and nothing the verifier's leaves place is read off the auditor
before the deadline (Pearl+'s verifier sends nothing to the node). With lean's `AsWritten` at `Vv` this is soundness's
N1 mend: what is heard is what crossed, the verifier holds nothing else, and the challenge leaves when it is sent. -/
def DeviceAsWritten (𝓕 : Set (Actual.Run L)) (A : Auditor L) : Prop :=
  ∀ R ∈ 𝓕,
    (∀ g, R.A.x g ∈ A.Vv → R.A.isInput g = false) ∧
    (∀ g h, g ∈ R.A.reads h → R.A.isInput g = true → R.A.x g ∈ A.Vd → R.A.cell g ∈ A.saltCells → A.ts ≤ R.A.t h) ∧
    (∀ g h, g ∈ R.A.reads h → R.A.isInput g = true → R.A.x g ∈ A.Vd → R.A.cell g ∈ A.sampleCells →
      A.tc ≤ R.A.t h) ∧
    (∀ g h, g ∈ R.A.reads h → R.A.x g ∈ A.Vv → R.A.x h ∉ A.leaves → A.td ≤ R.A.t h)

/-- What Pearl+'s code reads off a transcript it accepts: the roots, each sampled tile's position and opened value, and
`X_O`, the number of opened tiles over the inert-work bound. -/
structure Read where
  roots : List (List Bool)
  opened : List (ℕ × List Bool)
  xo : ℕ

open Classical in
/-- **Kind 2, the one premise about the code's function `vf` and its credit `W`**, on transcripts alone: where `vf`
accepts `τ`, the code's reading `dec τ` opens tiles exactly at the positions the challenge names in `τ`
(`positions τ`), each opening verifies against the roots (`verifies`: frame-v3-sha512 multiproof verification against
the roots), `X_O` is the number of opened tiles outside the tile rule `ok`, and `W τ` is the code's credit `cred` at that
`X_O`, the registered tiles' tickets and MACs and `t` being registered information fixed in `cred`. Pearl+'s executable
is to be an instance: `vf` its accept bit on the transcript, `cred` `creditAtAssumption`'s credited number over the
registered tiles at `t` and `X_O` (`Properties.Pouw.PearlPlus.workIsTheCredit` is the existing property of that number;
the instance at the transcript is a lemma to prove). -/
def CodeReadsCredit (vf : List (List Bool) → Bool) (W : List (List Bool) → ℕ) (dec : List (List Bool) → Option Read)
    (positions : List (List Bool) → List ℕ) (verifies : List (List Bool) → ℕ → List Bool → Prop)
    (ok : ℕ → List Bool → Prop) (cred : ℕ → ℕ) : Prop :=
  ∀ τ, vf τ = true → ∃ ρd, dec τ = some ρd ∧ ρd.opened.map Prod.fst = positions τ ∧
    (∀ p ∈ ρd.opened, verifies ρd.roots p.1 p.2) ∧
    ρd.xo = (ρd.opened.filter fun p => decide (¬ ok p.1 p.2)).length ∧ W τ = cred ρd.xo

/-- What the verifier at `V` heard during `[a, b)`: `heard` is in order of the reader's tick, so what it heard before
`a` is a prefix of what it heard before `b`. -/
noncomputable def heardBetween (R : Actual.Run L) (V : Finset L) (a b : ℕ) : List (List Bool) :=
  (heard R V b).drop (heard R V a).length

/-- **Kind 2, the roots are heard between the salt and the sample**, on the feasible runs: the roots the code reads off
the transcript at the deadline are `rootsOf` what the verifier heard during `[ts, tc)`, after the salt's message can
have been read and before the sample's can. The code tells the three apart by the device's two messages; the
transcript does not yet carry each message's sender (soundness's (2) asks for it), so this is stated on runs. -/
def RootsBeforeSample (𝓕 : Set (Actual.Run L)) (A : Auditor L) (dec : List (List Bool) → Option Read)
    (rootsOf : List (List Bool) → List (List Bool)) : Prop :=
  ∀ R ∈ 𝓕, ∀ ρd, dec (vtranscript A R) = some ρd → ρd.roots = rootsOf (heardBetween R A.Vv A.ts A.tc)

/-- The sample: `t` tickets, each a tile `u < N` and one of its `w u` tickets, so a uniform ticket names `u` with
probability `w u / ∑ w`. The challenge is a salt from `Ω₀` and a sample. -/
abbrev Sample (N t : ℕ) (w : ℕ → ℕ) : Type := Fin t → Σ u : Fin N, Fin (w u)

/-- **Kind 2, the code reads the sample's tiles**, on the feasible runs at each challenge: the positions the code reads
off the transcript are the sample's tiles, in order. -/
def PositionsAreSample {Ω₀ : Type} {N t : ℕ} {w : ℕ → ℕ} (𝓕 : Ω₀ × Sample N t w → Set (Actual.Run L))
    (A : Auditor L) (positions : List (List Bool) → List ℕ) : Prop :=
  ∀ r, ∀ R ∈ 𝓕 r, positions (vtranscript A R) = List.ofFn fun k => ((r.2 k).1 : ℕ)

/-! ## The tiles committed before the sample -/

section Strategy

variable {Ω : Type} (S : Ω → Actual.Run L) (A : Auditor L)

/-- Two challenges give the same run before the sample: every gate placed before `tc`, anywhere, is the same. -/
def SamePrefix (r r' : Ω) : Prop := ∀ X : Finset L, (S r).before X A.tc = (S r').before X A.tc

/-- The tiles the code reads as opened in the run at `r`, none where it reads nothing. -/
noncomputable def openedAt (dec : List (List Bool) → Option Read) (r : Ω) : List (ℕ × List Bool) :=
  ((dec (vtranscript A (S r))).map Read.opened).getD []

open Classical in
/-- **The tiles committed under the roots heard before the sample, right**, defined once and without an extractor:
tile `i < N` is right at `r` when some accepted run sharing `r`'s run before the sample opens it, and every accepted run
sharing it opens it within the tile rule `ok`. A tile no accepted such run opens is not right. This reads the strategy's
own runs at other challenges (as a rewinding argument does), so it is a function of the strategy and the run before
the sample, not of one run. It counts what a challenge after the roots can find, not what the node holds: that the
roots bind one value a position is `Binding`'s, and that those values cost work is the floor's. -/
noncomputable def committedRight (vf : List (List Bool) → Bool) (dec : List (List Bool) → Option Read)
    (ok : ℕ → List Bool → Prop) (N : ℕ) (r : Ω) : Finset ℕ :=
  (range N).filter fun i =>
    (∃ r', SamePrefix S A r r' ∧ vf (vtranscript A (S r')) = true ∧ ∃ v, (i, v) ∈ openedAt S A dec r') ∧
    ∀ r', SamePrefix S A r r' → vf (vtranscript A (S r')) = true → ∀ v, (i, v) ∈ openedAt S A dec r' → ok i v

/-- **Two runs sharing `r`'s run before the sample open one position two ways**, under the same roots heard before the
sample (`RootsBeforeSample`): a SHA-512 collision, given the openings verify. -/
def Collides (dec : List (List Bool) → Option Read) (r : Ω) : Prop :=
  ∃ r', SamePrefix S A r r' ∧ ∃ i v v', (i, v) ∈ openedAt S A dec r ∧ (i, v') ∈ openedAt S A dec r' ∧ v ≠ v'

/-- The window `[ts, td)` at the auditor's leaves, the only fields `PouwWork.credits` and `PouwWork.help` read. -/
def window : PouwWork.Verdict Ω L :=
  ⟨A.leaves, A.ts, A.td, 0, fun _ _ => 0, fun _ _ => 0, fun _ _ => True⟩

/-- **The node's work**: `PouwWork.credits` on the node's leaves `X` over `[ts, td)` at the registered pricing `price`
(a computed gate's credit, the most target multiply-adds it can stand for), no gate left out. -/
noncomputable def work (X : Finset L) (price : ∀ R : Actual.Run L, Fin R.A.N → ℕ) (r : Ω) : ℕ :=
  PouwWork.credits S (window A) X (fun r => price (S r)) (fun _ => ∅) r

/-- **The help into the node**: `PouwWork.help`, the width of the gates placed outside `X` and the auditor that `X` or
the auditor read during `[ts, td)`. -/
noncomputable def helpIn (X : Finset L) (r : Ω) : ℕ := PouwWork.help S (window A) X r

end Strategy

/-! ## The premises of the third kind -/

/-- The strategies `Sound` ranges over at the family `𝓕`. -/
def Admissible {Ω : Type} (𝓕 : Ω → Set (Actual.Run L)) (V : Finset L) (S : Ω → Actual.Run L) : Prop :=
  (∀ r, S r ∈ 𝓕 r) ∧ CausalFrom S ∧ ∀ r r', (S r).inputs (↑V)ᶜ = (S r').inputs (↑V)ᶜ

open Classical in
/-- **Kind 3, the floor, ONE: the tile-cost conjecture on the actual circuit**, at the gap `g` with error
`(q + N)/2^128`. For every admissible strategy, on at most that share of the challenges does the node take no help,
no two runs sharing the run before the sample open a position two ways, and its priced work in `[ts, td)` stay below
`(1 − g)` times the MACs of the tiles committed right before the sample. It is `GGap` / the named TT_OUT assumption
read on the actual circuit with SHA-512 the fixed function, where `main`'s has a uniform oracle and an abstract cost
model: `main`'s `TTOutTilePearlCH100Rev1` (rev1, H100) is its random-oracle argument, and this is a conjecture with no
verifier at which it is proved. It is claimed only at feasible sets meeting `HashBudget` at `q`, lean's `AsWritten` and
`DeviceAsWritten`, at `g = 1/400` for Pearl+. -/
def TileCostFloor {Ω : Type} [Fintype Ω] (𝓕 : Ω → Set (Actual.Run L)) (A : Auditor L) (X : Finset L)
    (price : ∀ R : Actual.Run L, Fin R.A.N → ℕ) (vf : List (List Bool) → Bool)
    (dec : List (List Bool) → Option Read) (ok : ℕ → List Bool → Prop) (N : ℕ) (mac : ℕ → ℕ) (g : ℝ)
    (q : ℕ) : Prop :=
  ∀ S, Admissible 𝓕 A.leaves S →
    ((univ.filter fun r => helpIn S A X r = 0 ∧ ¬ Collides S A dec r ∧
        (work S A X price r : ℝ) < (1 - g) * ∑ i ∈ committedRight S A vf dec ok N r, (mac i : ℝ)).card : ℝ)
      ≤ ((q + N : ℝ) / 2 ^ 128) * Fintype.card Ω

open Classical in
/-- **Kind 3, binding: SHA-512 collision resistance at the strategy's finder.** For every admissible strategy, the
accepted challenges at which two runs sharing the run before the sample open a position two ways are at most a
`2q/2^256` share. It is `SHA512CRExpected` (`ecr/sha-512`) read on the runs, at the finder that plays the strategy at two
challenges sharing the run before the sample and outputs the two openings: at most `2q` evaluations, so error
`2q/2^256`. `SHA512CRExpected` is stated over a game, not a run, so it does not fit as written; the reduction from it to
this, building that finder from the runs, is a lemma to prove and is not stated here. -/
def Binding {Ω : Type} [Fintype Ω] (𝓕 : Ω → Set (Actual.Run L)) (A : Auditor L) (vf : List (List Bool) → Bool)
    (dec : List (List Bool) → Option Read) (q : ℕ) : Prop :=
  ∀ S, Admissible 𝓕 A.leaves S →
    ((univ.filter fun r => vf (vtranscript A (S r)) = true ∧ Collides S A dec r).card : ℝ)
      ≤ (2 * q / 2 ^ 256 : ℝ) * Fintype.card Ω

/-- **Kind 3, help at `ρ`, its own claim, separate from the floor.** Every admissible strategy has an admissible
strategy with the same transcripts, the same runs before the sample up to the same pairs of challenges, no help, and
priced work at most its own plus `ρ` per bit of help: each bit from outside stands in for at most `ρ` priced work.
Compute accounting stands behind no `ρ` today: this is a placeholder for a claim nobody has made. -/
def HelpAtRho {Ω : Type} (𝓕 : Ω → Set (Actual.Run L)) (A : Auditor L) (X : Finset L)
    (price : ∀ R : Actual.Run L, Fin R.A.N → ℕ) (ρ : ℝ) : Prop :=
  ∀ S, Admissible 𝓕 A.leaves S → ∃ S', Admissible 𝓕 A.leaves S' ∧ ∀ r,
    vtranscript A (S' r) = vtranscript A (S r) ∧ (∀ r', SamePrefix S A r r' ↔ SamePrefix S' A r r') ∧
    helpIn S' A X r = 0 ∧ (work S' A X price r : ℝ) ≤ work S A X price r + ρ * helpIn S A X r

/-! ## Lemmas to prove -/

open Classical in
/-- **Lemma to prove (pure probability, the `X_O` tail).** Over a salt in `Ω₀` and `t` work-weighted tickets, for any
right set `K` that may read the salt: the share of challenges at which the tiles outside `K` hold more than
`eps k` of the ticket weight, `k` the number of tickets outside `K`, is at most `δ`. At `eps 0 = εs` and `eps k ≥ 1`
for `k > 0` (over_cap = reject) it is `WorkWeightedSampling` at ticket weights with `δ = (1 − εs)^t`; at
over_cap = uncredited, where Pearl+ widens `ε` with `X_O`, it is not on `main`. It holds of the code's `eps` (its ε at
each `X_O`), which is the property to prove of the code. -/
def XOTail (Ω₀ : Type) [Fintype Ω₀] (N t : ℕ) (w : ℕ → ℕ) (eps : ℕ → ℝ) (δ : ℝ) : Prop :=
  ∀ K : Ω₀ → Finset ℕ,
    ((univ.filter fun x : Ω₀ × Sample N t w =>
        eps (univ.filter fun k => ((x.2 k).1 : ℕ) ∉ K x.1).card * ∑ i ∈ range N, (w i : ℝ) <
          ∑ i ∈ range N \ K x.1, (w i : ℝ)).card : ℝ)
      ≤ δ * Fintype.card (Ω₀ × Sample N t w)

/-- **Lemma to prove (a property of the code's credit, the greedy MAC bound).** Where the tiles outside `K` hold at
most `eps k` of the ticket weight, the credit at `X_O = k` is at most `(1 − g)` times the MACs of `K`. For Pearl+ it is
the step from ε of the ticket weight to `creditAtAssumption`'s bound, filled from the tiles with the most MACs a ticket,
and it holds of `creditAtAssumption`'s number, not of any `cred`. -/
def GreedyBound (N : ℕ) (w mac : ℕ → ℕ) (cred : ℕ → ℕ) (eps : ℕ → ℝ) (g : ℝ) : Prop :=
  ∀ (k : ℕ) (K : Finset ℕ), K ⊆ range N →
    ∑ i ∈ range N \ K, (w i : ℝ) ≤ eps k * ∑ i ∈ range N, (w i : ℝ) →
    (cred k : ℝ) ≤ (1 - g) * ∑ i ∈ K, (mac i : ℝ)

/-! ## The statement -/

/-- **Pearl+'s useful work as served, on the actual circuit.** For every strategy whose run at each challenge is
feasible at it, the share of challenges at which the verifier accepts and the node's priced work in `[ts, td)` is below
`W(transcript) − ρ·help` is at most `δdet + (q + N)/2^128 + 2q/2^256`, `δdet` the error the code's credit is computed at.
NOT `(1 − 1/400)·W`: the credit already holds both discounts.

Premises, by kind: (1) `HashBudget`; (2) lean's `AsWritten` at the verifier, `DeviceAsWritten`, `CodeReadsCredit`,
`RootsBeforeSample`, `PositionsAreSample`, and the two code properties `XOTail` and `GreedyBound` (lemmas to prove at
the executable); (3) `TileCostFloor` (the one floor), `Binding`, `HelpAtRho`. None is about one strategy.

It is a theorem to prove, by the union bound of `TileCostFloor` (after `HelpAtRho` removes the help), `Binding`, and the
run-level detection that `XOTail` gives through `CodeReadsCredit` and `PositionsAreSample`, closed by `GreedyBound`.

Not vacuous because: the honest strategy that computes every tile in `[ts, td)`, sends the roots before `tc` and the
sampled openings after it, with Pearl+'s verifier as `vf` and `creditAtAssumption` as `cred`, meets every premise but the
floor's, which is the conjecture; and its conclusion fails at the accept-all verifier, which `CodeReadsCredit` refuses
(`acceptAll_not_code`). -/
def UsefulWorkActual : Prop :=
  ∀ {L : Type*} [LinearOrder L] (Ω₀ : Type) [Fintype Ω₀] (N t : ℕ) (w mac : ℕ → ℕ)
    (𝓕 : Set (Actual.Run L)) (A : Auditor L) (enc : Ω₀ × Sample N t w → List Bool) (X : Finset L)
    (price : ∀ R : Actual.Run L, Fin R.A.N → ℕ) (shaEvals : Actual.Run L → ℕ → ℕ) (q : ℕ)
    (vf : List (List Bool) → Bool) (W : List (List Bool) → ℕ) (dec : List (List Bool) → Option Read)
    (positions : List (List Bool) → List ℕ) (verifies : List (List Bool) → ℕ → List Bool → Prop)
    (ok : ℕ → List Bool → Prop) (cred : ℕ → ℕ) (rootsOf : List (List Bool) → List (List Bool))
    (eps : ℕ → ℝ) (g ρ δdet : ℝ),
    HashBudget 𝓕 A shaEvals q →
    AsWritten 𝓕 A.Vv A.y A.c A.td vf {[]} →
    DeviceAsWritten 𝓕 A →
    CodeReadsCredit vf W dec positions verifies ok cred →
    RootsBeforeSample 𝓕 A dec rootsOf →
    PositionsAreSample (feasibleAt 𝓕 A enc) A positions →
    XOTail Ω₀ N t w eps δdet →
    GreedyBound N w mac cred eps g →
    TileCostFloor (feasibleAt 𝓕 A enc) A X price vf dec ok N mac g q →
    Binding (feasibleAt 𝓕 A enc) A vf dec q →
    HelpAtRho (feasibleAt 𝓕 A enc) A X price ρ →
    SoundAt (Ω₀ × Sample N t w) (feasibleAt 𝓕 A enc) A.leaves (Accepts A.y A.c A.td)
      (fun S r => (W (vtranscript A (S r)) : ℝ) - ρ * helpIn S A X r ≤ work S A X price r)
      (δdet + (q + N : ℝ) / 2 ^ 128 + 2 * q / 2 ^ 256)

/-! ## Probes -/

/-- **Probe: the accept-all verifier is refused by the code premise** wherever some transcript has no reading, so the
statement cannot be applied to it. -/
theorem acceptAll_not_code (W : List (List Bool) → ℕ) (dec : List (List Bool) → Option Read)
    (positions : List (List Bool) → List ℕ) (verifies : List (List Bool) → ℕ → List Bool → Prop)
    (ok : ℕ → List Bool → Prop) (cred : ℕ → ℕ) (τ₀ : List (List Bool)) (h : dec τ₀ = none) :
    ¬ CodeReadsCredit (fun _ => true) W dec positions verifies ok cred := by
  intro hc
  obtain ⟨ρd, hd, -⟩ := hc τ₀ rfl
  rw [h] at hd
  cases hd

/-- **Probe (soundness's N1 (b)): a message arriving at or after the deadline is not heard.** Lean's `heard` keeps
only wires whose reader is placed before `d`, so tiles arriving after the deadline are in no transcript the verdict
reads. -/
theorem late_not_heard (R : Actual.Run L) (V : Finset L) (d : ℕ) (h : ∀ g, R.A.x g ∈ V → d ≤ R.A.t g) :
    heard R V d = [] := by
  unfold heard
  rw [List.map_eq_nil_iff]
  apply List.eq_nil_of_length_eq_zero
  rw [Finset.length_sort]
  first
    | (simp only [Finset.card_eq_zero, Finset.image_eq_empty, Finset.filter_eq_empty_iff]
       rintro p - ⟨-, -, hx, ht⟩
       exact absurd ht (not_lt.mpr (h p.1 hx)))
    | (apply Finset.card_eq_zero.mpr
       convert Finset.image_empty (fun p : Fin R.A.N × Fin R.A.N =>
         toLex (R.A.t p.1, toLex (R.A.x p.1, Actual.position (R.seen p.2))))
       exact Finset.filter_eq_empty_iff.mpr fun p _ hp => absurd hp.2.2.2 (not_lt.mpr (h p.1 hp.2.2.1)))

end Draft.PearlPlusUsefulWork
