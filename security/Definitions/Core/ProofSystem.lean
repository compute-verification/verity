module

public import Mathlib
public import Definitions.Core.Interaction

/-! The interface a recursion reads of its inner proof system (`ProofSystem`) and its outer one (`ZkProofSystem`). -/

@[expose] public section

namespace Verity

open Claims
open scoped NNReal

/-- A run's sample space. The fork at round `j` keeps the first sample's coins before round `j` and takes everything
else from the second. -/
structure SampleSpace (Coin : Type) (k : ℕ) where
  S : Type
  [fin : Fintype S]
  [ne : Nonempty S]
  coins : S → Fin k → Coin
  fork : ℕ → S → S → S
  coins_fork : ∀ j a b i, coins (fork j a b) i = if i.val < j then coins a i else coins b i

attribute [instance] SampleSpace.fin SampleSpace.ne

section Spaces

variable (Coin : Type) [Fintype Coin] [Nonempty Coin]

def SampleSpace.plain (k : ℕ) : SampleSpace Coin k where
  S := Fin k → Coin
  coins := id
  fork j a b i := if i.val < j then a i else b i
  coins_fork _ _ _ _ := rfl

/-- The verifier's coins and the prover's fresh coins `A`, which the verifier never reads; a fork takes the second
sample's. -/
def SampleSpace.withAbort (k : ℕ) (A : Type) [Fintype A] [Nonempty A] : SampleSpace Coin k where
  S := (Fin k → Coin) × A
  coins := Prod.fst
  fork j a b := (fun i => if i.val < j then a.1 i else b.1 i, b.2)
  coins_fork _ _ _ _ := rfl

end Spaces

structure RunFn (Coin : Type) (k : ℕ) where
  Ω : Type
  [fin : Fintype Ω]
  [ne : Nonempty Ω]
  D : SampleSpace Coin k
  trace : Ω → D.S → Transcript

attribute [instance] RunFn.fin RunFn.ne

abbrev RunClass {Coin : Type} (V : Verifier Coin) : Type 1 := (x : Msg) → RunFn Coin (V.k x) → Prop

section Runs

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

def RunFn.plain (H : Msg → Msg) (V : Verifier Coin) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    RunFn Coin (V.k x) where
  Ω := Ω
  D := SampleSpace.plain Coin (V.k x)
  trace ω r := run H P V x ω r

/-- Each run is aborted after the last round unless `keep`; an aborted run sends nothing and is rejected. -/
def RunFn.abort (H : Msg → Msg) (V : Verifier Coin) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (A : Type) [Fintype A] [Nonempty A] (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) : RunFn Coin (V.k x) where
  Ω := Ω
  D := SampleSpace.withAbort Coin (V.k x) A
  trace ω t := if keep ω t.1 t.2 then run H P V x ω t.1 else ⟨[], [], [], .rej⟩

def RunClass.plain (H : Msg → Msg) (V : Verifier Coin) : RunClass V := fun x F =>
  ∃ (Ω : Type) (_ : Fintype Ω) (_ : Nonempty Ω) (P : Prover Ω), F = RunFn.plain H V x P

def RunClass.aborting (H : Msg → Msg) (V : Verifier Coin) : RunClass V := fun x F =>
  ∃ (Ω : Type) (_ : Fintype Ω) (_ : Nonempty Ω) (P : Prover Ω) (A : Type) (_ : Fintype A) (_ : Nonempty A)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool), F = RunFn.abort H V x P A keep

end Runs

/-- It sees the sample space only through its samples, their coins and their forks, so it runs unchanged over plain
and aborting runs. -/
structure PExtractor (Coin W : Type) (V : Verifier Coin) where
  C : Msg → Type → Type
  [fin : ∀ x (S : Type) [Fintype S], Fintype (C x S)]
  [ne : ∀ x (S : Type) [Nonempty S], Nonempty (C x S)]
  ext : (x : Msg) → (D : SampleSpace Coin (V.k x)) → D.S → C x D.S → RP D.S (Option W)

attribute [instance] PExtractor.fin PExtractor.ne

section Extractors

variable {Coin W : Type}

def PExtractor.toExtractor [Fintype Coin] [Nonempty Coin] {V : Verifier Coin} (E : PExtractor Coin W V) :
    Extractor Coin W V where
  C x := E.C x (Fin (V.k x) → Coin)
  ext x r₀ c := E.ext x (SampleSpace.plain Coin (V.k x)) r₀ c

/-- **`E^rep` over any sample space**: rerun the judged sample; if it accepted, make `m` waits, each over `K` fresh
samples and stopping at `k` accepted reruns, and return the first assembled candidate that `check` accepts. At the plain
sample it is `ERep`. -/
def PExtractor.rep {Op : Type} (V : Verifier Coin) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) : PExtractor Coin W V where
  C x S := Fin (m x) → Fin (K x) → S
  fin _ _ _ := inferInstance
  ne _ _ _ := inferInstance
  ext x _ d₀ ρ := .rerun d₀ fun v₀ =>
    if v₀.out = .acc then
      firstWait (k x) opens (assemble x) (check x) ((List.finRange (m x)).map fun i => (List.finRange (K x)).map (ρ i))
    else .ret none

variable {V : Verifier Coin}

def PExtractor.WaldLe (H : Msg → Msg) (E : PExtractor Coin W V) (N : Msg → ℕ) : Prop :=
  ∀ x (F : RunFn Coin (V.k x)) (ω : F.Ω) (f : F.D.S → ℝ), (∀ d, 0 ≤ f d) →
    Ex (fun s : F.D.S × E.C x F.D.S => (((E.ext x F.D s.1 s.2).rerunsAt H (F.trace ω)).map f).sum) ≤ N x * Ex f

def PExtractor.RunsLe (H : Msg → Msg) (E : PExtractor Coin W V) (M : Msg → ℕ) : Prop :=
  ∀ x (F : RunFn Coin (V.k x)) ω d c, ((E.ext x F.D d c).eval H (F.trace ω)).2.2 ≤ M x

noncomputable def PrBadOn (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : PExtractor Coin W V) (x : Msg)
    (F : RunFn Coin (V.k x)) : ℝ :=
  Pr fun s : F.Ω × F.D.S × E.C x F.D.S =>
    (F.trace s.1 s.2.1).out = .acc ∧ ¬ R.ext x ((E.ext x F.D s.2.1 s.2.2).eval H (F.trace s.1)).1

end Extractors

/-- How a collision term enters a knowledge error: `√(w · Pr[F])` or `w · Pr[F]`. -/
inductive TermForm where
  | root
  | lin
  deriving DecidableEq

inductive TermPrice where
  | strict
  | expected
  deriving DecidableEq

structure KTerm (Coin : Type) (V : Verifier Coin) where
  w : Msg → ℝ
  finder : (Msg → Msg) → (x : Msg) → RunFn Coin (V.k x) → Option Finder
  form : TermForm
  price : TermPrice

section Terms

variable {Coin W : Type} {V : Verifier Coin}

noncomputable def KTerm.val (τ : KTerm Coin V) (H : Msg → Msg) (x : Msg) (F : RunFn Coin (V.k x)) : ℝ :=
  match τ.finder H x F with
  | none => 0
  | some G =>
    match τ.form with
    | .root => Real.sqrt (τ.w x * G.Pr H)
    | .lin => τ.w x * G.Pr H

/-- **`B` at a prover's runs** (extraction-free): guess `j`, sample the tape and two samples, run at the first and at
its fork at round `j` with the second, and output the first collision among the verifier's SHA-512 calls in the two
runs. Its cost is both runs' calls, both parties'. At the plain sample it is `B`. -/
def RunFn.B {k : ℕ} (F : RunFn Coin k) (hk : 0 < k) : Finder where
  S := Fin k × F.Ω × F.D.S × F.D.S
  ne := ⟨(⟨0, hk⟩, Classical.arbitrary _, Classical.arbitrary _, Classical.arbitrary _)⟩
  out s := findColl ((F.trace s.2.1 s.2.2.1).vLog ++ (F.trace s.2.1 (F.D.fork s.1.val s.2.2.1 s.2.2.2)).vLog)
  cost s :=
    let t₁ := F.trace s.2.1 s.2.2.1
    let t₂ := F.trace s.2.1 (F.D.fork s.1.val s.2.2.1 s.2.2.2)
    t₁.pLog.length + t₁.vLog.length + t₂.pLog.length + t₂.vLog.length

/-- **`B`'s term**, `√(γ · Pr[B])`, priced by `StrictBirthday`, and `0` at a verifier of no rounds. -/
def KTerm.B (γ : Msg → ℝ) : KTerm Coin V where
  w := γ
  finder _ x F := if hk : 0 < V.k x then some (F.B hk) else none
  form := .root
  price := .strict

def KSOn (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : PExtractor Coin W V) (κ : Msg → ℝ)
    (terms : List (KTerm Coin V)) (cls : RunClass V) : Prop :=
  ∀ x F, cls x F → PrBadOn H V R E x F ≤ κ x + (terms.map fun τ => τ.val H x F).sum

end Terms

/-- What a protocol brings to a recursion as its inner. Its extractor is per hash, since a compiled system's reads the
inner views there. -/
structure ProofSystem (Coin W : Type) where
  V : Verifier Coin
  R : Rel W
  Scope : Msg → Prop
  scopeCheck : Msg → Bool
  E : (Msg → Msg) → PExtractor Coin W V
  N : Msg → ℕ
  M : Msg → ℕ
  κ : Msg → ℝ
  terms : List (KTerm Coin V)
  Tape : Type
  [tapeNe : Nonempty Tape]
  P : W → Prover Tape
  δc : (Msg → Msg) → Msg → ℝ
  /-- Whether the honest prover stops at a statement, read off the verifier's coins so far. -/
  stops : (Msg → Msg) → Msg → List Coin → Bool
  len : Msg → ℕ → ℕ

attribute [instance] ProofSystem.tapeNe

namespace ProofSystem

section Forms

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (S : ProofSystem Coin W)

def KS : Prop := KSOn H S.V S.R (S.E H) S.κ S.terms (RunClass.plain H S.V)

/-- **KS against provers that may abort** after the last round, reading every coin of the run and fresh coins of their
own. A recursion's inner owes this: the outer runs come after the inner rounds and decide what the extractor sees of
an inner rerun. -/
def KSAbort : Prop := KSOn H S.V S.R (S.E H) S.κ S.terms (RunClass.aborting H S.V)

def Wald : Prop := (S.E H).WaldLe H S.N ∧ (S.E H).RunsLe H S.M

def SelfChecking : Prop :=
  ∀ x (D : SampleSpace Coin (S.V.k x)) (runs : D.S → Transcript) d c w,
    (((S.E H).ext x D d c).eval H runs).1 = some w → S.R x w

def Complete : Prop := Claims.Complete H S.V S.P S.R S.Scope (S.δc H)

def StoppedAt (x : Msg) (cs : List Coin) : Prop := ∃ i ≤ cs.length, S.stops H x (cs.take i) = true

def PublicFailures : Prop :=
  Out64 H → ∀ x w, S.Scope x → S.R x w → ∀ ω : S.Tape,
    (∀ r, (run H (S.P w) S.V x ω r).out ≠ .acc → S.StoppedAt H x (List.ofFn r)) ∧
      Pr (fun r : Fin (S.V.k x) → Coin => S.StoppedAt H x (List.ofFn r)) ≤ S.δc H x

def PublicLengths : Prop :=
  Out64 H → ∀ x w, S.Scope x → S.R x w → ∀ (ω : S.Tape) r j a b,
    (run H (S.P w) S.V x ω r).msgs[j]? = some (a, b) → ¬ S.StoppedAt H x ((List.ofFn r).take j) →
      a.length = S.len x j

end Forms

def ScopeMatches {Coin W : Type} (S : ProofSystem Coin W) : Prop := Claims.ScopeMatches S.Scope S.scopeCheck

end ProofSystem

structure ZkProofSystem (Coin W : Type) extends ProofSystem Coin W where
  kZK : Msg → ℕ
  Nonce : Msg → Type
  Rest : Msg → Type
  [nonceFin : ∀ x, Fintype (Nonce x)]
  [nonceNe : ∀ x, Nonempty (Nonce x)]
  [restFin : ∀ x, Fintype (Rest x)]
  [restNe : ∀ x, Nonempty (Rest x)]
  Pz : W → (x : Msg) → Prover (Nonce x × Rest x)
  Sim : ℕ → Simulator
  εzk : ℕ → ℝ≥0 → Msg → ℝ
  γzk : ℕ → Msg → ℝ
  Hides : (Msg → Msg) → ℝ≥0 → Prop
  Shape : Type
  shape : Msg → Shape
  CS : Shape → Type → Type
  κS : Shape → ℝ
  wS : Shape → List ℝ
  NS : Shape → ℕ
  MS : Shape → ℕ
  εzkS : ℕ → ℝ≥0 → Shape → ℝ
  γzkS : ℕ → Shape → ℝ

attribute [instance] ZkProofSystem.nonceFin ZkProofSystem.nonceNe ZkProofSystem.restFin ZkProofSystem.restNe

namespace ZkProofSystem

variable {Coin W : Type}

def ZK (H : Msg → Msg) (O : ZkProofSystem Coin W) : Prop :=
  ∀ t, 0 < t → ∀ δ₁ : ℝ≥0, O.Hides H δ₁ → ∀ x₀,
    ZKmalR H O.kZK (fun w => O.Pz w x₀) (fun x w => x = x₀ ∧ O.R x w) (O.Sim t) (O.εzk t δ₁) (O.γzk t)

/-- The public shape fixes the extractor's coins and bounds the errors, so an outer statement chosen after the inner
coins has errors and extractor coins its shape fixes. -/
def ByShape (O : ZkProofSystem Coin W) : Prop :=
  ∀ x, (∀ H, (O.E H).C x = O.CS (O.shape x)) ∧ O.κ x ≤ O.κS (O.shape x) ∧
    List.Forall₂ (fun (τ : KTerm Coin O.V) w => τ.w x ≤ w) O.terms (O.wS (O.shape x)) ∧
    O.N x ≤ O.NS (O.shape x) ∧ O.M x ≤ O.MS (O.shape x) ∧
    (∀ t δ₁, O.εzk t δ₁ x ≤ O.εzkS t δ₁ (O.shape x)) ∧ ∀ t, O.γzk t x ≤ O.γzkS t (O.shape x)

end ZkProofSystem

end Verity
