import Definitions.Pouw.Basic.Prob
import Definitions.Pouw.Basic.Mat
import Definitions.Pouw.Basic.QTree

/-!
# The γ-gap game (trusted; problem statement §3.2–§3.5)

The game `G_γ(Π, 𝒜, T, q)` of §3.3, with probabilities over the oracle `H` and the salt `s`:

~~~text
0. Oracle.      H : Q → R is uniform over all functions (Q, R finite).
1. Workload.    U = (layout, weights): N ≤ 2^64 units, unit u has shape (m_u, k_u, n_u), weight id wid u,
                index u. U is universally quantified outside the probability, as in §3.3's requirement.
2. Preprocess.  σ = A.pre.eval H : an adaptive oracle program (a QTree), chosen after U, with unbounded
                computation between queries; never a function of the salt. Its queries count towards q.
3. Salt.        s : S uniform, independent of H and hence of σ.
4. Online.      τ = CM.run A.online H U A.act σ s, with CM.cost ≤ T and (preprocessing queries) + CM.calls ≤ q
                for every H and s. Preprocessing costs nothing. The online program reads the workload and
                its activation input A.act (the units' own operands); it may commit other activations.
5. Score.       K = correctSet; A wins if  T / (1 − γ) < ∑_{u ∈ K} W_ref(u).
~~~

The cost model `CM` is a parameter fixed before the protocol and the workload: the adversary picks a program of
the model, and the model, not the adversary, says what it costs. M_4090 is one instance; every theorem holds for
every cost model.

Simplifications (each listed in `STATEMENTS.md` with its bridge or as a gap):
* ideal commitments: the online phase outputs the committed values themselves (`Transcript`);
* a finite oracle `Q → R` and a uniform salt type `S` (the chain's instance is `Fin (2^128)`);
* deterministic adversaries (`Definitions/Pouw/Basic/Prob.lean`).
-/

namespace Pouw

open Finset

/-! ## Workloads -/

/-- A unit's shape: `A` is `m × k`, `B` is `k × n`. -/
structure Shape where
  m : ℕ
  k : ℕ
  n : ℕ

/-- `W_mm = m·k·n`, the plain GEMM's multiply-accumulates. -/
def Shape.Wmm (sh : Shape) : ℕ := sh.m * sh.k * sh.n

/-- The declared layout: `N` units `0, …, N − 1` (unit `u`'s index is `u`), their shapes and weight ids. -/
structure Layout where
  N : ℕ
  shape : ℕ → Shape
  wid : ℕ → ℕ

/-- A declared workload: its layout and the weights `B_w`, fixed before the salt. -/
structure Workload where
  layout : Layout
  weight : ℕ → Mat

/-- The §3.2 default shape domain: `m` a positive multiple of 16, and `k, n ∈ [2^10, 2^16]` multiples of 64. -/
def Shape.InDefaultDomain (sh : Shape) : Prop :=
  0 < sh.m ∧ 16 ∣ sh.m ∧ 2 ^ 10 ≤ sh.k ∧ sh.k ≤ 2 ^ 16 ∧ 64 ∣ sh.k ∧ 2 ^ 10 ≤ sh.n ∧ sh.n ≤ 2 ^ 16 ∧ 64 ∣ sh.n

/-- The first milestone's shape `(m, k, n) = (2^12, 2^16, 2^16)` (§6, notes on D1). -/
def milestoneShape : Shape := ⟨2 ^ 12, 2 ^ 16, 2 ^ 16⟩

/-- Every unit's shape is in the shape domain `D`. §3.2's domain is `Layout.ShapesIn Shape.InDefaultDomain`. -/
def Layout.ShapesIn (D : Shape → Prop) (L : Layout) : Prop := ∀ u < L.N, D (L.shape u)

/-- The rows weight `w` serves: `∑_{u : w(u) = w} m_u`. -/
def Layout.rowsOf (L : Layout) (w : ℕ) : ℕ :=
  ∑ u ∈ (Finset.range L.N).filter (fun u => L.wid u = w), (L.shape u).m

/-- A workload is in the domain `D` (a predicate on layouts, part of `Π`: working decision W3) if it has at most
`2^64` units (the index is a 64-bit field), its layout satisfies `D`, and every weight entry a unit reads is int7.
The weights are otherwise arbitrary, degenerate ones included. Without the bound on `N`, a finite oracle gives two
units identical noise (`Pouw.Sanity.identical_checked`). -/
def Workload.InDomain (D : Layout → Prop) (U : Workload) : Prop :=
  U.layout.N ≤ 2 ^ 64 ∧ D U.layout ∧
    ∀ u < U.layout.N, ∀ l < (U.layout.shape u).k, ∀ j < (U.layout.shape u).n, InInt7 (U.weight (U.layout.wid u) l j)

/-! ## Transcripts and cost models -/

/-- The preprocessed state: any finite data. A function of the finite oracle can be encoded in it injectively. -/
abbrev State : Type := List ℤ

/-- Checked values: blocks of integer matrices, `checked t i j`. A protocol says which entries are compared. -/
abbrev Checked : Type := ℕ → Mat

/-- What the online phase commits to for one unit: the activations `A_u`, the checked values and the useful
output (ideal commitment: the values themselves). -/
structure Commit where
  act : Mat
  checked : Checked
  out : Mat

/-- A committed transcript: unit `u`'s commitment, or `none` (a blank, which is never correct). -/
abbrev Transcript : Type := ℕ → Option Commit

/-- An abstract cost model for the online phase: its programs, what a program outputs, what it costs and how many
oracle calls it makes, on oracle `H`, workload `U`, activation input `act` (unit `u`'s own operand is `act u`),
state `σ` and salt `s`. The model fixes `cost` and `calls`; an adversary only chooses a program. -/
structure CostModel (Q R S : Type) where
  Prog : Type
  run : Prog → (Q → R) → Workload → (ℕ → Mat) → State → S → Transcript
  cost : Prog → (Q → R) → Workload → (ℕ → Mat) → State → S → ℝ
  calls : Prog → (Q → R) → Workload → (ℕ → Mat) → State → S → ℕ

/-- The model's programs reach `H` only through counted queries: on fixed inputs, a run is the result
of an adaptive oracle program, and `calls` counts that program's queries. -/
def CostModel.Queries {Q R S : Type} (CM : CostModel Q R S) : Prop :=
  ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Mat) (σ : State) (s : S), ∃ t : QTree Q R Transcript,
    ∀ H : Q → R, CM.run prog H U act σ s = t.eval H ∧ CM.calls prog H U act σ s = t.calls H

/-! ## Protocols -/

/-- A protocol `Π` (§3.2): its checked values `Y(H, s, U, u, A_u)` (the noise is derived from `H` inside `Y`), the
entries of them the verifier compares, the honest reference `H_ref`'s output (its checked values and its decoded
useful output), and `W_ref`, `H_ref`'s cost per unit excluding hash calls. `W_ref` reads only the layout, never an
input value. -/
structure Protocol (Q R S : Type) where
  Y : (Q → R) → S → Workload → ℕ → Mat → Checked
  checkIdx : Shape → Finset (ℕ × ℕ × ℕ)
  refChecked : (Q → R) → S → Workload → ℕ → Mat → Checked
  refOut : (Q → R) → S → Workload → ℕ → Mat → Mat
  Wref : Layout → ℕ → ℕ

variable {Q R S : Type}

/-- The committed activations are int7 inside the unit's shape. -/
def ActOK (sh : Shape) (A : Mat) : Prop := ∀ i < sh.m, ∀ l < sh.k, InInt7 (A i l)

/-- The checked values are correct: the activations are in the domain, and every compared entry equals `Y`. -/
def CheckedOK (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (u : ℕ) (c : Commit) : Prop :=
  ActOK (U.layout.shape u) c.act ∧
    ∀ x ∈ P.checkIdx (U.layout.shape u), c.checked x.1 x.2.1 x.2.2 = P.Y H s U u c.act x.1 x.2.1 x.2.2

/-- The useful output is correct: it equals `A_u · B_{w(u)}` on the unit's `m × n` entries. -/
def OutOK (U : Workload) (u : ℕ) (c : Commit) : Prop :=
  ∀ i < (U.layout.shape u).m, ∀ j < (U.layout.shape u).n,
    c.out i j = mm (U.layout.shape u).k c.act (U.weight (U.layout.wid u)) i j

/-- A correct unit (§3.2): correct checked values and the correct useful output. -/
def Correct (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (u : ℕ) (c : Commit) : Prop :=
  CheckedOK P U H s u c ∧ OutOK U u c

open Classical in
/-- `K`: the units that are correct in the transcript `τ`. -/
noncomputable def correctSet (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) :
    Finset ℕ :=
  (range U.layout.N).filter fun u => ∃ c, τ u = some c ∧ Correct P U H s u c

open Classical in
/-- `K'` (§4.4): the units whose checked values are correct; the useful output is not required. -/
noncomputable def checkedSet (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) :
    Finset ℕ :=
  (range U.layout.N).filter fun u => ∃ c, τ u = some c ∧ CheckedOK P U H s u c

/-- `H_ref` is complete on the domain `D` (§3.2): for every in-domain workload, unit, oracle, salt and int7
activations, its checked values equal `Y` on every compared entry and its useful output equals `A·B`. -/
def Protocol.Complete (P : Protocol Q R S) (D : Layout → Prop) : Prop :=
  ∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, ∀ (H : Q → R) (s : S) (A : Mat), ActOK (U.layout.shape u) A →
    (∀ x ∈ P.checkIdx (U.layout.shape u),
        P.refChecked H s U u A x.1 x.2.1 x.2.2 = P.Y H s U u A x.1 x.2.1 x.2.2) ∧
    (∀ i < (U.layout.shape u).m, ∀ j < (U.layout.shape u).n,
        P.refOut H s U u A i j = mm (U.layout.shape u).k A (U.weight (U.layout.wid u)) i j)

/-- The honest transcript on activations `act`: `H_ref`'s output for every unit. -/
def honestTranscript (P : Protocol Q R S) (H : Q → R) (s : S) (U : Workload) (act : ℕ → Mat) : Transcript :=
  fun u => if u < U.layout.N then
    some ⟨act u, P.refChecked H s U u (act u), P.refOut H s U u (act u)⟩ else none

/-- The cost model runs `H_ref` at its claimed cost: one program, fixed before the workload and the activations
(a real kernel reads its operands; it does not embed them), outputs the honest transcript on every in-domain
workload and int7 activation input, with no preprocessing, at cost at most `∑_u W_ref(u)` and at most `qref U`
oracle calls. -/
def AdmitsRef (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (qref : Workload → ℕ) : Prop :=
  ∃ prog : CM.Prog, ∀ U : Workload, U.InDomain D → ∀ act : ℕ → Mat,
    (∀ u < U.layout.N, ActOK (U.layout.shape u) (act u)) → ∀ (H : Q → R) (s : S),
      CM.run prog H U act [] s = honestTranscript P H s U act ∧
      CM.cost prog H U act [] s ≤ ∑ u ∈ range U.layout.N, (P.Wref U.layout u : ℝ) ∧
      CM.calls prog H U act [] s ≤ qref U

/-! ## Adversaries and the game -/

/-- An adversary, chosen after the workload: a preprocessing oracle program (never reading the salt; unbounded
computation, counted queries), an online program of the cost model, and the activation input its online program
reads as the units' own operands (the committed activations may differ). -/
structure Adversary (CM : CostModel Q R S) where
  pre : QTree Q R State
  online : CM.Prog
  act : ℕ → Mat

/-- The transcript the adversary commits on oracle `H` and salt `s`. -/
def Adversary.transcript {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (H : Q → R) (s : S) :
    Transcript :=
  CM.run A.online H U A.act (A.pre.eval H) s

/-- The online phase costs at most `T`, and preprocessing and the online phase together make at most `q` oracle
calls, for every oracle and salt (surely). -/
def Adversary.Bounded {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (T : ℝ) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), CM.cost A.online H U A.act (A.pre.eval H) s ≤ T ∧
    A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

/-- `W = ∑_{u ∈ U} W_ref(u)`. -/
def totalWref (P : Protocol Q R S) (L : Layout) : ℝ := ∑ u ∈ range L.N, (P.Wref L u : ℝ)

/-- The adversary wins: the honest work of its correct units exceeds `T / (1 − γ)`. -/
def Wins (P : Protocol Q R S) (U : Workload) (γ T : ℝ) (τ : Transcript) (H : Q → R) (s : S) : Prop :=
  T / (1 - γ) < ∑ u ∈ correctSet P U H s τ, (P.Wref U.layout u : ℝ)

/-- **The γ-gap property** `G_γ` with error `η(q, N)`, in §3.3's quantifier order after `Π`: for every in-domain
workload, every budget `T ≥ 0`, every `q ≤ 2^64` and every adversary within `(T, q)`, `Pr_{H,s}[𝒜 wins] ≤ η`. -/
def Gγ [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S => Wins P U γ T (A.transcript U x.1 x.2) x.1 x.2) ≤ η q U.layout.N

/-- **§3.3's requirement** at `(γ, η)`: `∃ Π` whose honest reference is complete and runs in the cost model at
cost at most `W_ref` (so `W_ref` cannot be under-declared; `W_ref ≡ 0` would make `G_γ` trivial), such that `G_γ`
holds with error `η`. `W_ref` is input-independent by its type. -/
def Requirement [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∃ (P : Protocol Q R S) (qref : Workload → ℕ), P.Complete D ∧ AdmitsRef CM P D qref ∧ Gγ CM P D γ η

/-! ## The audit (§3.5) -/

/-- A work-weighted ticket: unit `u < N` owns `w u` tickets, so a uniform ticket selects `u` with probability
`w u / ∑_v w v`. -/
abbrev Ticket (w : ℕ → ℕ) (N : ℕ) : Type := Σ u : Fin N, Fin (w u)

/-- The audit with `t` independent work-weighted samples accepts iff every sampled unit is in `K` (the verifier
recomputes the sampled units and checks their checked values and useful output). -/
def AuditAccepts {w : ℕ → ℕ} {N t : ℕ} (K : Finset ℕ) (c : Fin t → Ticket w N) : Prop :=
  ∀ i, ((c i).1 : ℕ) ∈ K

end Pouw
