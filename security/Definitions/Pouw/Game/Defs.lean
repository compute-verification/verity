module

public import Definitions.Pouw.Basic.Prob
public import Definitions.Pouw.Basic.Mat
public import Definitions.Pouw.Basic.QTree
public import Definitions.Hash.Basic

/-! The gap game `G_g(Π, 𝒜, T, q)` at a cost model fixed before the protocol and the workload, which reads SHA-512 through its break event rather than a collision-resistance assumption. -/

@[expose] public section

namespace Pouw

open Finset

/-- A unit's shape: `A` is `m × k`, `B` is `k × n`. -/
structure Shape where
  m : ℕ
  k : ℕ
  n : ℕ

/-- `W_mm = m·k·n`, the plain GEMM's multiply-accumulates. -/
def Shape.Wmm (sh : Shape) : ℕ := sh.m * sh.k * sh.n

/-- The declared layout: `N` units `0, …, N − 1` (unit `u`'s index is `u`), their shapes and the id of each unit's
fixed operand. -/
structure Layout where
  N : ℕ
  shape : ℕ → Shape
  fixedId : ℕ → ℕ

/-- A declared workload: its layout and the fixed operands `B_w`. -/
structure Workload where
  layout : Layout
  /-- Declared before the salt and reusable across units. -/
  fixed : ℕ → Mat

def Shape.InDefaultDomain (sh : Shape) : Prop :=
  0 < sh.m ∧ 16 ∣ sh.m ∧ 2 ^ 10 ≤ sh.k ∧ sh.k ≤ 2 ^ 16 ∧ 64 ∣ sh.k ∧ 2 ^ 10 ≤ sh.n ∧ sh.n ≤ 2 ^ 16 ∧ 64 ∣ sh.n

/-- The first milestone's shape `(m, k, n) = (2^12, 2^16, 2^16)` (§6, notes on D1). -/
def milestoneShape : Shape := ⟨2 ^ 12, 2 ^ 16, 2 ^ 16⟩

def Layout.ShapesIn (D : Shape → Prop) (L : Layout) : Prop := ∀ u < L.N, D (L.shape u)

/-- The rows fixed operand `w` serves: `∑_{u : fixedId u = w} m_u`. -/
def Layout.rowsOf (L : Layout) (w : ℕ) : ℕ :=
  ∑ u ∈ (Finset.range L.N).filter (fun u => L.fixedId u = w), (L.shape u).m

/-- A workload is in the domain `D` (a predicate on layouts, part of `Π`: working decision W3) if it has at most
`2^64` units (the index is a 64-bit field), its layout satisfies `D`, and every fixed-operand entry a unit reads is
int7. The fixed operands are otherwise arbitrary, degenerate ones included. Without the bound on `N`, a finite oracle gives two
units identical noise (`Pouw.Sanity.identical_checked`). -/
def Workload.InDomain (D : Layout → Prop) (U : Workload) : Prop :=
  U.layout.N ≤ 2 ^ 64 ∧ D U.layout ∧
    ∀ u < U.layout.N, ∀ l < (U.layout.shape u).k, ∀ j < (U.layout.shape u).n, InInt7 (U.fixed (U.layout.fixedId u) l j)

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

/-- A committed trace: unit `u`'s commitment, or `none` (a blank, which is never correct). -/
abbrev Trace : Type := ℕ → Option Commit

/-- An abstract cost model for the online phase: its programs, what a program outputs, what it costs and how many
oracle calls it makes, on oracle `H`, workload `U`, activation input `act` (unit `u`'s own operand is `act u`),
state `σ` and salt `s`, and the SHA-512 its programs hash with (`sha`) and the inputs a run hashes with it (`hashed`).
The model fixes `cost`, `calls` and `hashed`; an adversary only chooses a program. By default a model hashes
nothing. -/
structure CostModel (Q R S : Type) where
  Prog : Type
  run : Prog → (Q → R) → Workload → (ℕ → Mat) → State → S → Trace
  cost : Prog → (Q → R) → Workload → (ℕ → Mat) → State → S → ℝ
  calls : Prog → (Q → R) → Workload → (ℕ → Mat) → State → S → ℕ
  sha : Hash.Msg → Hash.Digest := fun _ => default
  hashed : Prog → (Q → R) → Workload → (ℕ → Mat) → State → S → List Hash.Msg := fun _ _ _ _ _ _ => []

/-- The model's programs reach `H` only through counted queries: on fixed inputs, a run is the result
of an adaptive oracle program, and `calls` counts that program's queries. -/
def CostModel.Queries {Q R S : Type} (CM : CostModel Q R S) : Prop :=
  ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Mat) (σ : State) (s : S), ∃ t : QTree Q R Trace,
    ∀ H : Q → R, CM.run prog H U act σ s = t.eval H ∧ CM.calls prog H U act σ s = t.calls H

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
    c.out i j = mm (U.layout.shape u).k c.act (U.fixed (U.layout.fixedId u)) i j

/-- A correct unit (§3.2): correct checked values and the correct useful output. -/
def Correct (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (u : ℕ) (c : Commit) : Prop :=
  CheckedOK P U H s u c ∧ OutOK U u c

open Classical in
/-- `K`: the units that are correct in the trace `τ`. -/
noncomputable def correctSet (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Trace) :
    Finset ℕ :=
  (range U.layout.N).filter fun u => ∃ c, τ u = some c ∧ Correct P U H s u c

open Classical in
/-- `K'` (§4.4): the units whose checked values are correct; the useful output is not required. -/
noncomputable def checkedSet (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Trace) :
    Finset ℕ :=
  (range U.layout.N).filter fun u => ∃ c, τ u = some c ∧ CheckedOK P U H s u c

def Protocol.Complete (P : Protocol Q R S) (D : Layout → Prop) : Prop :=
  ∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, ∀ (H : Q → R) (s : S) (A : Mat), ActOK (U.layout.shape u) A →
    (∀ x ∈ P.checkIdx (U.layout.shape u),
        P.refChecked H s U u A x.1 x.2.1 x.2.2 = P.Y H s U u A x.1 x.2.1 x.2.2) ∧
    (∀ i < (U.layout.shape u).m, ∀ j < (U.layout.shape u).n,
        P.refOut H s U u A i j = mm (U.layout.shape u).k A (U.fixed (U.layout.fixedId u)) i j)

def honestTrace (P : Protocol Q R S) (H : Q → R) (s : S) (U : Workload) (act : ℕ → Mat) : Trace :=
  fun u => if u < U.layout.N then
    some ⟨act u, P.refChecked H s U u (act u), P.refOut H s U u (act u)⟩ else none

/-- One program, fixed before the workload and the activations: a real kernel reads its operands and does not embed them. -/
def AdmitsRef (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (qref : Workload → ℕ) : Prop :=
  ∃ prog : CM.Prog, ∀ U : Workload, U.InDomain D → ∀ act : ℕ → Mat,
    (∀ u < U.layout.N, ActOK (U.layout.shape u) (act u)) → ∀ (H : Q → R) (s : S),
      CM.run prog H U act [] s = honestTrace P H s U act ∧
      CM.cost prog H U act [] s ≤ ∑ u ∈ range U.layout.N, (P.Wref U.layout u : ℝ) ∧
      CM.calls prog H U act [] s ≤ qref U

/-- An adversary, chosen after the workload: a preprocessing oracle program (never reading the salt; unbounded
computation, counted queries), an online program of the cost model, and the activation input its online program
reads as the units' own operands (the committed activations may differ). -/
structure Adversary (CM : CostModel Q R S) where
  pre : QTree Q R State
  online : CM.Prog
  act : ℕ → Mat

/-- The trace the adversary commits on oracle `H` and salt `s`. -/
def Adversary.trace {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (H : Q → R) (s : S) :
    Trace :=
  CM.run A.online H U A.act (A.pre.eval H) s

/-- The inputs the adversary's online run hashes with SHA-512 on oracle `H` and salt `s`. -/
def Adversary.hashed {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (H : Q → R) (s : S) :
    List Hash.Msg :=
  CM.hashed A.online H U A.act (A.pre.eval H) s

/-- **The run breaks SHA-512**: two different inputs it hashed have one digest. -/
def Adversary.Breaks {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (H : Q → R) (s : S) : Prop :=
  Hash.CollisionIn CM.sha (A.hashed U H s)

/-- The online phase costs at most `T`, and preprocessing and the online phase together make at most `q` oracle
calls, for every oracle and salt (surely). -/
def Adversary.Bounded {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (T : ℝ) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), CM.cost A.online H U A.act (A.pre.eval H) s ≤ T ∧
    A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

/-- `W = ∑_{u ∈ U} W_ref(u)`. -/
def totalWref (P : Protocol Q R S) (L : Layout) : ℝ := ∑ u ∈ range L.N, (P.Wref L u : ℝ)

/-- The adversary wins: the honest work of its correct units exceeds `T / (1 − g)`. -/
def Wins (P : Protocol Q R S) (U : Workload) (g T : ℝ) (τ : Trace) (H : Q → R) (s : S) : Prop :=
  T / (1 - g) < ∑ u ∈ correctSet P U H s τ, (P.Wref U.layout u : ℝ)

/-- **The gap property** `G_g` at the gap `g` (an adversary's advantage `ε = g/(1 − g)`) with failure probability
`δ(q, N)`, in §3.3's quantifier order after `Π`: for every in-domain workload, every budget `T ≥ 0`, every
`q ≤ 2^64` and every adversary within `(T, q)`, `Pr_{H,s}[𝒜 wins ∧ 𝒜's run doesn't break SHA-512] ≤ δ`. -/
def GGap [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (g : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S => Wins P U g T (A.trace U x.1 x.2) x.1 x.2 ∧ ¬ A.Breaks U x.1 x.2)
      ≤ δ q U.layout.N

/-- The honest reference runs at cost at most `W_ref`, so `W_ref` cannot be under-declared: `W_ref ≡ 0` would make
`G_g` trivial. -/
def Requirement [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (D : Layout → Prop) (g : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∃ (P : Protocol Q R S) (qref : Workload → ℕ), P.Complete D ∧ AdmitsRef CM P D qref ∧ GGap CM P D g δ

/-- A work-weighted ticket: unit `u < N` owns `w u` tickets, so a uniform ticket selects `u` with probability
`w u / ∑_v w v`. -/
abbrev Ticket (w : ℕ → ℕ) (N : ℕ) : Type := Σ u : Fin N, Fin (w u)

/-- The audit with `t` independent work-weighted samples accepts iff every sampled proof block is in `K` (the verifier
recomputes the sampled proof blocks and checks their checked values and useful output). -/
def AuditAccepts {w : ℕ → ℕ} {N t : ℕ} (K : Finset ℕ) (c : Fin t → Ticket w N) : Prop :=
  ∀ i, ((c i).1 : ℕ) ∈ K

def drawsIn {w : ℕ → ℕ} {N t : ℕ} (O : Finset ℕ) (c : Fin t → Ticket w N) : ℕ :=
  (univ.filter fun i => ((c i).1 : ℕ) ∈ O).card

def AuditAcceptsTol {w : ℕ → ℕ} {N t : ℕ} (K O : Finset ℕ) (c : Fin t → Ticket w N) : Prop :=
  ∀ i, ((c i).1 : ℕ) ∈ K ∨ ((c i).1 : ℕ) ∈ O

/-- `u` is the sampling tolerance read off the samples; Clopper–Pearson's one-sided upper bound at `δ` is one. -/
def MissBound (t : ℕ) (u : ℕ → ℝ) (δ : ℝ) : Prop :=
  ∀ (N : ℕ) (w : ℕ → ℕ) (S : Finset ℕ), S ⊆ range N →
    pr (fun c : Fin t → Ticket w N => u (drawsIn S c) * ∑ g ∈ range N, (w g : ℝ) < ∑ g ∈ S, (w g : ℝ)) ≤ δ

end Pouw
