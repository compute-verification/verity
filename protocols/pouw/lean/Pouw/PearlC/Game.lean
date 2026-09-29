import Pouw.PearlC.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# The γ-gap game for an output-ticket FP8 protocol (trusted)

The game of the integer PoUW problem statement (§3.3), specialised to a protocol whose bound words are output words:
per unit, the ticket words (Pearl-C's pre-peel accumulator `C̃`) and the useful output words (`U`, after the checked
clean-up and before the unscale), both FP32 bit patterns, and both checked bit for bit.

~~~text
0. Oracle.      H : Q → R uniform over all functions (Q, R finite).
1. Workload.    U = (layout, weights): N ≤ 2^64 units; unit u has shape (m_u, k_u, n_u), weight id wid u, index u.
                U is quantified outside the probability.
2. Preprocess.  σ = A.pre.eval H: an adaptive oracle program, never a function of the salt; its queries count in q.
3. Salt.        s : S uniform, independent of H and hence of σ.
4. Online.      τ = CM.run A.online H U A.act σ s, with CM.cost ≤ T and (preprocessing + online) calls ≤ q, for every
                H and s. The online program chooses the committed activations adaptively.
5. Score.       K = correctSet; A wins if T / (1 − γ) < ∑_{u ∈ K} W_ref(u).
~~~

Simplifications, as in the integer package: ideal commitments (the online phase outputs the committed values), a finite
oracle and a uniform salt, deterministic adversaries. Values are kept as bit patterns (`ℕ`); what they mean is the
protocol's reference functions, which the named instance takes from the pinned tensor-core models.
-/

namespace Pouw.PearlC

open Finset

/-! ## Workloads -/

/-- A unit's shape: activations `A` are `m × k`, weights `B` are `k × n`. -/
structure Shape where
  m : ℕ
  k : ℕ
  n : ℕ

/-- `m·k·n`, the plain GEMM's multiply-accumulates. -/
def Shape.Wmm (sh : Shape) : ℕ := sh.m * sh.k * sh.n

/-- The declared layout: `N` units `0, …, N − 1` (unit `u`'s index is `u`), their shapes and weight ids. -/
structure Layout where
  N : ℕ
  shape : ℕ → Shape
  wid : ℕ → ℕ

/-- A matrix of bit patterns: E4M3 or BF16 codes of an operand, FP32 words of an output. -/
abbrev Codes : Type := ℕ → ℕ → ℕ

/-- A declared workload: its layout and the weights `B_w` (their codes), fixed before the salt. -/
structure Workload where
  layout : Layout
  weight : ℕ → Codes

/-! ## Transcripts and cost models -/

/-- The preprocessed state: any finite data. -/
abbrev State : Type := List ℤ

/-- What the online phase commits to for one unit (ideal commitment: the values themselves): the activations `A_u`,
the ticket words and the useful output words. -/
structure Commit where
  act : Codes
  ticket : Codes
  useful : Codes

/-- A committed transcript: unit `u`'s commitment, or `none` (a blank, which is never correct). -/
abbrev Transcript : Type := ℕ → Option Commit

/-- An abstract cost model for the online phase: its programs, what a program outputs, what it costs and how many
oracle calls it makes, on oracle `H`, workload `U`, activation input `act`, state `σ` and salt `s`. The model fixes
`cost` and `calls`; an adversary only chooses a program. The named instance is the H100 accounting (W1 at the
measured H100 prices, one unit = one dense FP8 multiply-accumulate). -/
structure CostModel (Q R S : Type) where
  Prog : Type
  run : Prog → (Q → R) → Workload → (ℕ → Codes) → State → S → Transcript
  cost : Prog → (Q → R) → Workload → (ℕ → Codes) → State → S → ℝ
  calls : Prog → (Q → R) → Workload → (ℕ → Codes) → State → S → ℕ

/-- The model's programs reach `H` only through counted queries. -/
def CostModel.Queries {Q R S : Type} (CM : CostModel Q R S) : Prop :=
  ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S), ∃ t : QTree Q R Transcript,
    ∀ H : Q → R, CM.run prog H U act σ s = t.eval H ∧ CM.calls prog H U act σ s = t.calls H

/-! ## Protocols -/

/-- An output-ticket protocol `Π`: the reference ticket words and useful output words of a unit, as functions of the
oracle (which derives the noise), the salt, the workload, the unit and its committed activations; the domain rules on
committed activations and on weights (for Pearl-C: finite codes, liveness, the noise floor); and two costs read from
the layout only, never from an input value: `credit`, the honest work the named assumption credits, and `Wref`, the
honest reference's whole cost (excluding hash calls). -/
structure Protocol (Q R S : Type) where
  ticket : (Q → R) → S → Workload → ℕ → Codes → Codes
  useful : (Q → R) → S → Workload → ℕ → Codes → Codes
  actOK : Shape → Codes → Prop
  weightOK : Shape → Codes → Prop
  credit : Layout → ℕ → ℝ
  Wref : Layout → ℕ → ℝ

variable {Q R S : Type}

/-- A workload is in the domain `D` (a predicate on layouts, part of `Π`) if it has at most `2^64` units, its layout
satisfies `D`, and every unit's weights satisfy the protocol's weight rules. -/
def Workload.InDomain (P : Protocol Q R S) (D : Layout → Prop) (U : Workload) : Prop :=
  U.layout.N ≤ 2 ^ 64 ∧ D U.layout ∧ ∀ u < U.layout.N, P.weightOK (U.layout.shape u) (U.weight (U.layout.wid u))

/-- A correct unit: its committed activations satisfy the domain rules, and every ticket word and every useful output
word on its `m × n` entries equals the protocol's reference, bit for bit. -/
def Correct (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (u : ℕ) (c : Commit) : Prop :=
  P.actOK (U.layout.shape u) c.act ∧
    ∀ i < (U.layout.shape u).m, ∀ j < (U.layout.shape u).n,
      c.ticket i j = P.ticket H s U u c.act i j ∧ c.useful i j = P.useful H s U u c.act i j

open Classical in
/-- `K`: the units that are correct in the transcript `τ`. -/
noncomputable def correctSet (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) :
    Finset ℕ :=
  (range U.layout.N).filter fun u => ∃ c, τ u = some c ∧ Correct P U H s u c

/-- The honest transcript on activations `act`: the reference words for every unit. -/
def honestTranscript (P : Protocol Q R S) (H : Q → R) (s : S) (U : Workload) (act : ℕ → Codes) : Transcript :=
  fun u => if u < U.layout.N then some ⟨act u, P.ticket H s U u (act u), P.useful H s U u (act u)⟩ else none

/-- The cost model runs the honest reference at its claimed cost: one program, fixed before the workload and the
activations, outputs the honest transcript on every in-domain workload and admissible activation input, with no
preprocessing, at cost at most `∑_u W_ref(u)` and at most `qref U` oracle calls. -/
def AdmitsRef (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (qref : Workload → ℕ) : Prop :=
  ∃ prog : CM.Prog, ∀ U : Workload, U.InDomain P D → ∀ act : ℕ → Codes,
    (∀ u < U.layout.N, P.actOK (U.layout.shape u) (act u)) → ∀ (H : Q → R) (s : S),
      CM.run prog H U act [] s = honestTranscript P H s U act ∧
      CM.cost prog H U act [] s ≤ ∑ u ∈ range U.layout.N, P.Wref U.layout u ∧
      CM.calls prog H U act [] s ≤ qref U

/-! ## Adversaries and the game -/

/-- An adversary, chosen after the workload: a preprocessing oracle program (never reading the salt), an online program
of the cost model, and the activation input its online program reads. -/
structure Adversary (CM : CostModel Q R S) where
  pre : QTree Q R State
  online : CM.Prog
  act : ℕ → Codes

/-- The transcript the adversary commits on oracle `H` and salt `s`. -/
def Adversary.transcript {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (H : Q → R) (s : S) :
    Transcript :=
  CM.run A.online H U A.act (A.pre.eval H) s

/-- The online phase costs at most `T`, and preprocessing and the online phase together make at most `q` oracle
calls, for every oracle and salt (surely). -/
def Adversary.Bounded {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (T : ℝ) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), CM.cost A.online H U A.act (A.pre.eval H) s ≤ T ∧
    A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

/-- The adversary wins: the honest work of its correct units exceeds `T / (1 − γ)`. -/
def Wins (P : Protocol Q R S) (U : Workload) (γ T : ℝ) (τ : Transcript) (H : Q → R) (s : S) : Prop :=
  T / (1 - γ) < ∑ u ∈ correctSet P U H s τ, P.Wref U.layout u

/-- **The γ-gap property** `G_γ` with error `η(q, N)`: for every in-domain workload, every budget `T ≥ 0`, every
`q ≤ 2^64` and every adversary within `(T, q)`, `Pr_{H,s}[𝒜 wins] ≤ η`. -/
def Gγ [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S => Wins P U γ T (A.transcript U x.1 x.2) x.1 x.2) ≤ η q U.layout.N

end Pouw.PearlC
