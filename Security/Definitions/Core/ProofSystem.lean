import Mathlib
import Definitions.Core.Interaction

/-!
# Proof systems: what a protocol brings to a recursion

The interface a recursion `Rec(I, O)` reads of its inner protocol `I` and its outer protocol `O`, stated in the
interaction model (`Definitions/Core/Interaction.lean`, namespace `Claims`):

* A run is a function of a sample (`SampleSpace`): the verifier's coins, and whatever else the run samples, as the fresh
  coins of a prover that may abort after the last round (`SampleSpace.withAbort`). A fork at round `j` keeps the first
  sample's coins before `j` and takes everything else from the second.
* A prover is seen through its runs at one statement (`RunFn`): its tape, its sample space, and the trace at each.
  Knowledge soundness is stated at a class of these (`RunClass`): every prover (`plain`), or every prover that may abort
  after the last round (`aborting`).
* An extractor is given the sample space only abstractly (`PExtractor`), so one extractor runs over plain and aborting
  runs alike. `E^rep` is one (`PExtractor.rep`), and at the plain sample it is `ERep` (`toExtractor`).
* A knowledge error is `κ` plus collision terms (`KTerm`): a weight and a finder, under a square root or linear, priced
  by `StrictBirthday` or `ExpectedCR`. `KTerm.B` is `B`'s, the one `KS` has.
* `KSOn` is knowledge soundness at a class; `WaldLe` and `RunsLe` bound the extractor's reruns, weighted and at most.
* `ProofSystem` is what an inner brings, and `ZkSystem` what an outer adds: a masked prover, simulators with their
  errors, the hiding they rest on, and each statement's public shape, which fixes the extractor's coins and bounds the
  errors (`ByShape`).

Mathlib and Core only, as Core's layer is.
-/

namespace Verity

open Claims
open scoped NNReal

/-! ## Samples and runs -/

/-- **A sample space** of a run of `k` rounds: what one run samples (`S`, uniformly), the verifier's coins in a sample,
and the fork at round `j`, which keeps the first sample's coins before round `j` and takes everything else from the
second. -/
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

/-- The plain sample: the verifier's coins. -/
def SampleSpace.plain (k : ℕ) : SampleSpace Coin k where
  S := Fin k → Coin
  coins := id
  fork j a b i := if i.val < j then a i else b i
  coins_fork _ _ _ _ := rfl

/-- The sample of a run that may abort after its last round: the verifier's coins, and the prover's fresh coins `A`,
which the verifier never reads. A fork takes the second sample's. -/
def SampleSpace.withAbort (k : ℕ) (A : Type) [Fintype A] [Nonempty A] : SampleSpace Coin k where
  S := (Fin k → Coin) × A
  coins := Prod.fst
  fork j a b := (fun i => if i.val < j then a.1 i else b.1 i, b.2)
  coins_fork _ _ _ _ := rfl

end Spaces

/-- **A prover's runs at one statement**, as an extractor and a collision finder see them: its tape `Ω` (uniform), its
sample space, and the run at each tape and sample. -/
structure RunFn (Coin : Type) (k : ℕ) where
  Ω : Type
  [fin : Fintype Ω]
  [ne : Nonempty Ω]
  D : SampleSpace Coin k
  trace : Ω → D.S → Trace

attribute [instance] RunFn.fin RunFn.ne

/-- **A class of provers**, as the runs they have at each statement. -/
abbrev RunClass {Coin : Type} (V : Verifier Coin) : Type 1 := (x : Msg) → RunFn Coin (V.k x) → Prop

section Runs

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `P`'s runs against `V` at `x`, on the plain sample. -/
def RunFn.plain (H : Msg → Msg) (V : Verifier Coin) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    RunFn Coin (V.k x) where
  Ω := Ω
  D := SampleSpace.plain Coin (V.k x)
  trace ω r := run H P V x ω r

/-- `P`'s runs against `V` at `x`, each aborted after the last round unless `keep`, which reads the tape, every coin of
the run and the run's fresh coins `A`. An aborted run sends nothing and is rejected. -/
def RunFn.abort (H : Msg → Msg) (V : Verifier Coin) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (A : Type) [Fintype A] [Nonempty A] (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) : RunFn Coin (V.k x) where
  Ω := Ω
  D := SampleSpace.withAbort Coin (V.k x) A
  trace ω t := if keep ω t.1 t.2 then run H P V x ω t.1 else ⟨[], [], [], .rej⟩

/-- Every prover on a finite tape. -/
def RunClass.plain (H : Msg → Msg) (V : Verifier Coin) : RunClass V := fun x F =>
  ∃ (Ω : Type) (_ : Fintype Ω) (_ : Nonempty Ω) (P : Prover Ω), F = RunFn.plain H V x P

/-- Every prover on a finite tape that may abort after the last round, on any finite fresh coins. -/
def RunClass.aborting (H : Msg → Msg) (V : Verifier Coin) : RunClass V := fun x F =>
  ∃ (Ω : Type) (_ : Fintype Ω) (_ : Nonempty Ω) (P : Prover Ω) (A : Type) (_ : Fintype A) (_ : Nonempty A)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool), F = RunFn.abort H V x P A keep

end Runs

/-! ## Extractors -/

/-- **An extractor given the sample space abstractly**: over any sample space `D` of the verifier's rounds, from the
judged run's sample and its own coins (`C x D.S`, built from `D`'s samples), a rerun program over `D`'s samples. It sees
the space only through its samples, their coins and their forks, so it runs unchanged over plain and aborting runs. -/
structure PExtractor (Coin W : Type) (V : Verifier Coin) where
  C : Msg → Type → Type
  [fin : ∀ x (S : Type) [Fintype S], Fintype (C x S)]
  [ne : ∀ x (S : Type) [Nonempty S], Nonempty (C x S)]
  ext : (x : Msg) → (D : SampleSpace Coin (V.k x)) → D.S → C x D.S → RP D.S (Option W)

attribute [instance] PExtractor.fin PExtractor.ne

section Extractors

variable {Coin W : Type}

/-- The extractor at the plain sample, a `Claims.Extractor`. -/
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

/-- **Wald's bound on the reruns**: at every run function and tape, for every nonnegative weight `f` of a sample, the
expected sum of `f` over the samples `E` reruns at, over the judged sample and `E`'s coins, is at most `N(x)` times
`f`'s mean. -/
def PExtractor.WaldLe (H : Msg → Msg) (E : PExtractor Coin W V) (N : Msg → ℕ) : Prop :=
  ∀ x (F : RunFn Coin (V.k x)) (ω : F.Ω) (f : F.D.S → ℝ), (∀ d, 0 ≤ f d) →
    Ex (fun s : F.D.S × E.C x F.D.S => (((E.ext x F.D s.1 s.2).rerunsAt H (F.trace ω)).map f).sum) ≤ N x * Ex f

/-- **The reruns at most**: `E` reruns at most `M(x)` times, at every run function, tape, sample and coins. -/
def PExtractor.RunsLe (H : Msg → Msg) (E : PExtractor Coin W V) (M : Msg → ℕ) : Prop :=
  ∀ x (F : RunFn Coin (V.k x)) ω d c, ((E.ext x F.D d c).eval H (F.trace ω)).2.2 ≤ M x

/-- `Pr[run = acc ∧ ¬R(x, E(x))]` at a prover's runs: over the tape, the judged sample and `E`'s coins, `E` rerunning
the prover at the judged run's tape. -/
noncomputable def PrBadOn (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : PExtractor Coin W V) (x : Msg)
    (F : RunFn Coin (V.k x)) : ℝ :=
  Pr fun s : F.Ω × F.D.S × E.C x F.D.S =>
    (F.trace s.1 s.2.1).out = .acc ∧ ¬ R.ext x ((E.ext x F.D s.2.1 s.2.2).eval H (F.trace s.1)).1

end Extractors

/-! ## Knowledge errors -/

/-- How a collision term enters a knowledge error: `√(w · Pr[F])` or `w · Pr[F]`. -/
inductive TermForm where
  | root
  | lin
  deriving DecidableEq

/-- The named assumption that prices a term's finder: `StrictBirthday` (at a worst-case cost) or `ExpectedCR` (at an
expected one). -/
inductive TermPrice where
  | strict
  | expected
  deriving DecidableEq

/-- **A collision term** of a knowledge error: a weight, and a finder at each statement and prover's runs (none where
the term is `0`, as `B`'s is at a verifier of no rounds), entering as `form` and priced as `price` says. -/
structure KTerm (Coin : Type) (V : Verifier Coin) where
  w : Msg → ℝ
  finder : (Msg → Msg) → (x : Msg) → RunFn Coin (V.k x) → Option Finder
  form : TermForm
  price : TermPrice

section Terms

variable {Coin W : Type} {V : Verifier Coin}

/-- A collision term's value at a statement and a prover's runs. -/
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

/-- **Knowledge soundness at a class of provers**: at every statement and every prover's runs in the class,
`Pr[run = acc ∧ ¬R(x, E(x))] ≤ κ(x) + Σ terms`. -/
def KSOn (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : PExtractor Coin W V) (κ : Msg → ℝ)
    (terms : List (KTerm Coin V)) (cls : RunClass V) : Prop :=
  ∀ x F, cls x F → PrBadOn H V R E x F ≤ κ x + (terms.map fun τ => τ.val H x F).sum

end Terms

/-! ## Proof systems -/

/-- **A proof system**, what a protocol brings to a recursion as its inner: its verifier, relation and readable scope;
its extractor (at each hash, since a compiled system's reads the inner views there) with its reruns, and its knowledge
error; its honest prover on its tapes with its completeness error; and what hiding its messages needs public: when the
honest prover stops, and its messages' lengths. -/
structure ProofSystem (Coin W : Type) where
  V : Verifier Coin
  R : Rel W
  Scope : Msg → Prop
  scopeCheck : Msg → Bool
  E : (Msg → Msg) → PExtractor Coin W V
  /-- The extractor's reruns in expectation (`WaldLe`). -/
  N : Msg → ℕ
  /-- The extractor's reruns at most (`RunsLe`). -/
  M : Msg → ℕ
  κ : Msg → ℝ
  terms : List (KTerm Coin V)
  Tape : Type
  [tapeNe : Nonempty Tape]
  P : W → Prover Tape
  δc : (Msg → Msg) → Msg → ℝ
  /-- Whether the honest prover stops at a statement, read off the verifier's coins so far. -/
  stops : (Msg → Msg) → Msg → List Coin → Bool
  /-- The honest prover's message length in each round. -/
  len : Msg → ℕ → ℕ

attribute [instance] ProofSystem.tapeNe

namespace ProofSystem

section Forms

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (S : ProofSystem Coin W)

/-- **KS** against every prover. -/
def KS : Prop := KSOn H S.V S.R (S.E H) S.κ S.terms (RunClass.plain H S.V)

/-- **KS against provers that may abort** after the last round, reading every coin of the run and fresh coins of their
own. A recursion's inner owes this: the outer runs come after the inner rounds and decide what the extractor sees of
an inner rerun. -/
def KSAbort : Prop := KSOn H S.V S.R (S.E H) S.κ S.terms (RunClass.aborting H S.V)

/-- The extractor reruns `N` times in expectation, weighted by any nonnegative cost of a sample, and `M` at most. -/
def Wald : Prop := (S.E H).WaldLe H S.N ∧ (S.E H).RunsLe H S.M

/-- **The extractor checks its output**: every witness it returns is in the relation, over every sample space and
run. -/
def SelfChecking : Prop :=
  ∀ x (D : SampleSpace Coin (S.V.k x)) (runs : D.S → Trace) d c w,
    (((S.E H).ext x D d c).eval H runs).1 = some w → S.R x w

/-- **Complete**, at the honest prover's completeness error. -/
def Complete : Prop := Claims.Complete H S.V S.P S.R S.Scope (S.δc H)

/-- The honest prover has stopped by the coins `cs`: its stop holds at a prefix of them. -/
def StoppedAt (x : Msg) (cs : List Coin) : Prop := ∃ i ≤ cs.length, S.stops H x (cs.take i) = true

/-- **The honest prover's failures are public**: at a 64-byte hash, on a statement in scope and a witness in the
relation, every run the verifier doesn't accept has stopped, which reads only the statement and the coins, and the
honest prover stops with probability at most its completeness error. -/
def PublicFailures : Prop :=
  Out64 H → ∀ x w, S.Scope x → S.R x w → ∀ ω : S.Tape,
    (∀ r, (run H (S.P w) S.V x ω r).out ≠ .acc → S.StoppedAt H x (List.ofFn r)) ∧
      Pr (fun r : Fin (S.V.k x) → Coin => S.StoppedAt H x (List.ofFn r)) ≤ S.δc H x

/-- **The honest prover's message lengths are public**: at a 64-byte hash, on a statement in scope and a witness in the
relation, its message in round `j` has `len x j` bytes unless it has stopped by the coins before it. -/
def PublicLengths : Prop :=
  Out64 H → ∀ x w, S.Scope x → S.R x w → ∀ (ω : S.Tape) r j a b,
    (run H (S.P w) S.V x ω r).msgs[j]? = some (a, b) → ¬ S.StoppedAt H x ((List.ofFn r).take j) →
      a.length = S.len x j

end Forms

/-- The readable scope is the verifier's own scope check. -/
def ScopeMatches {Coin W : Type} (S : ProofSystem Coin W) : Prop := Claims.ScopeMatches S.Scope S.scopeCheck

end ProofSystem

/-- **A zero-knowledge proof system**, what a recursion's outer adds: its masked prover on a nonce and the rest of its
tape at each statement (`Pz`, over `kZK x` rounds); its simulators at each attempt budget `t` (`Sim t`), with their
errors `εzk t δ₁` and `γzk t` wherever its commitments hide up to `δ₁` (`Hides`); and each statement's public shape: its
extractor's coins (`CS`) and its errors' bounds (`κS`, the terms' weights `wS`, `NS`, `MS`, `εzkS`, `γzkS`), which
`ByShape` asks the statement's to meet. -/
structure ZkSystem (Coin W : Type) extends ProofSystem Coin W where
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

attribute [instance] ZkSystem.nonceFin ZkSystem.nonceNe ZkSystem.restFin ZkSystem.restNe

namespace ZkSystem

variable {Coin W : Type}

/-- **Zero knowledge** against malicious verifiers, statement by statement: at every attempt budget `t ≥ 1` and wherever
the commitments hide up to `δ₁`, `Sim t` simulates the masked prover at each statement within `εzk t δ₁`, plus `γzk t`
times the probability that `Fork` finds a collision. -/
def ZK (H : Msg → Msg) (O : ZkSystem Coin W) : Prop :=
  ∀ t, 0 < t → ∀ δ₁ : ℝ≥0, O.Hides H δ₁ → ∀ x₀,
    ZKmalR H O.kZK (fun w => O.Pz w x₀) (fun x w => x = x₀ ∧ O.R x w) (O.Sim t) (O.εzk t δ₁) (O.γzk t)

/-- **The public shape fixes the extractor's coins and bounds the errors**: at every statement, the extractor's coins
are its shape's, and the knowledge error, each collision term's weight, the reruns and the simulation errors are at most
its shape's. So an outer statement chosen after the inner coins has errors and extractor coins its shape fixes. -/
def ByShape (O : ZkSystem Coin W) : Prop :=
  ∀ x, (∀ H, (O.E H).C x = O.CS (O.shape x)) ∧ O.κ x ≤ O.κS (O.shape x) ∧
    List.Forall₂ (fun (τ : KTerm Coin O.V) w => τ.w x ≤ w) O.terms (O.wS (O.shape x)) ∧
    O.N x ≤ O.NS (O.shape x) ∧ O.M x ≤ O.MS (O.shape x) ∧
    (∀ t δ₁, O.εzk t δ₁ x ≤ O.εzkS t δ₁ (O.shape x)) ∧ ∀ t, O.γzk t x ≤ O.γzkS t (O.shape x)

end ZkSystem

end Verity
