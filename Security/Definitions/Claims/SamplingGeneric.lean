module

public import Definitions.Claims.Sampling
public import Definitions.Claims.Recursion

/-!
# Claims: the sampled audit as a compiler, generic in the proof system for proof blocks (a draft, before it freezes)

`Claims.Sampling` with its Flock-specific parts as parameters: the proof system the sessions of the sampled proof blocks
run (`Arg`, for Flock `Sampling.Deploy.flock`) and its extractor (for Flock `E^rep`, `Sampling.unitExt`). Everything
else is `Claims.Sampling`'s, read as it stands: the statement (`AuditStatement`), the witness (`Sampling.Wit`), the
relation at `δ` (`Sampling.R`: the incorrect set of the extracted trace is in `M_δ = {B : P_a(B) > δ}`) and its count
form (`Sampling.RCount`), the readable scope (`InScope`), the commitment scheme and its binding (`CommitmentScheme`,
`CommitmentScheme.Binds`), the waits, the extraction-based finder `Bext` and the claims' forms (`KSLinOn`, `BitsLin`,
`ExtractOwnLe`, `leafLayer`).

1. **The proof system for proof blocks** (`Arg`): a public-coin verifier over byte strings, its statements' public shape
   (`Sh`, `shape`) and the rounds a shape fixes, its witnesses and relation, and its extractor: coins per shape (`EC`),
   a rerun program at the judged run's coins (`ext`), and its errors per shape (`κ`, `γ`). It is knowledge sound in
   `Claims.KS`'s form (`Arg.KS`); its cost is reruns (`Arg.RerunsLe`, `Arg.RerunsExpLe`) and its own SHA-512 calls
   (`Arg.OwnLe`), both counted by `RP.eval`. Its zero knowledge is per public outcome (`ZkArg.ZK`, `ZKOut`): at each
   outcome (each verdict, where a session stopped), the honest view's probability of the outcome and an event is at
   most the simulator's, given the outcome, plus an error that reads the statement, the witness, the verifier and the
   outcome; the leak form over `n` outcomes is its corollary (`ZKOutLeak`).
2. **The compiled sampled verifier** (`Compiled`, `Sampling.Deploy` without Flock): its verifier, parser, scope check,
   partition and commitment; the rounds after the draw (`stage2`); the sampled proof blocks (`checked`); each one's
   session (`session`: its statement and view), public shape (`sh`) and coins (`place`, `coinsOf`); and what a
   session's witness reveals (`reveal`), each opening checked at most `openCalls` SHA-512 calls.
3. **The extractor** (`Compiled.E`): `Sampling.E` with each sampled proof block's extraction `Arg`'s extractor at its
   session (`Compiled.unitExt`), each rerun the judged run's coins with the session's rounds replaced (`place`), seen as
   the session's view, then what its witness reveals that passes its opening check. Every SHA-512 call of the
   extraction is an `RP.hash` (the openings it checks, `Arg`'s own) or a rerun's, so `Bext`'s cost is counted, and the
   extractor is fixed before `H`.
4. **The errors**: `κ_δ = c²·(δ + κ_s) + (1 + c)·(2k/K + e^{−k/8})`, `Sampling.κ` with Flock's `κ_F` replaced by the
   largest of `Arg`'s `κ` at the proof blocks' shapes (`κs`); `γ = c⁴·k(x)·γ_s`, with `γ_s` the largest of
   `Arg`'s `γ` per round at the proof blocks' shapes (a session's fork finder is the compiled verifier's `B` at a round
   guess among `k(x)`); `ε_bind = Pr[B_ext]`.
5. **The contract** (`Compiled.Sessions`): the code facts the compiler theorems read: the draw is the design's and fixed
   before the sessions; each sampled proof block's session on an accepted run accepts at the block's shape; a fork on
   its rounds is a run of a prover against `Arg`'s verifier; and a witness in `Arg`'s relation reveals openings under
   which its proof block is correct.
6. **The claims** of a compiled verifier (`Compiled.KS`, `Compiled.KSCount`, `Compiled.Bits`, `Compiled.Efficient`,
   and `ZkCompiled.ZK`, zero knowledge per public outcome relative to the published leaf layer).
7. **The compiler's statements** (`CompilerKS`, `CompilerKSCount`, `CompilerBits`, `CompilerEfficient`): each claim of
   every compiled verifier that meets the contract, from `Arg`'s.
8. **Flock** (`Arg.ofFlock`): Flock's `E^rep` at a decoder, its shape the numbers its errors read; `FlockArgKS` says
   that `Flock.KS` is its `Arg.KS`, so `FlockKS` is `CompilerKS` at Flock.

Statements only: definitions, no proofs. `Claims.Sampling` stays as it is, frozen; it becomes this block's Flock instance
when its extractor is restated as `Compiled.E` at `Arg.ofFlock`.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition CommitmentScheme Opening InScope waitAccC forkOn flatMapRP
  checkRevealed assemble Bext KSLinOn BitsLin ExtractOwnLe lamLin slack leafLayer Rhon)
open Claims.Recursion (serve viewing)

/-- The largest of `f 0, …, f (n − 1)`, and `0` when there is none. -/
noncomputable def maxBelow (n : ℕ) (f : ℕ → ℝ) : ℝ := ((List.range n).map f).foldr max 0

/-- The largest of `f 0, …, f (n − 1)`, and `0` when there is none. -/
def maxBelowN (n : ℕ) (f : ℕ → ℕ) : ℕ := ((List.range n).map f).foldr max 0

/-! ## 1. The proof system for proof blocks -/

/-- **A proof system for proof blocks**, as the sampled audit reads it: its statements' public shapes (`Sh`, `shape`),
its verifier and the rounds a shape fixes, its witnesses and relation, and its extractor: its coins (`EC`, a finite type
per shape), its rerun program at the judged run's coins (`ext`), and its errors per shape (`κ`, `γ`). A field, so the
extractor is fixed before the hash: it calls `H` only through `RP.hash`. -/
structure Arg (Coin : Type) where
  Sh : Type
  shape : Msg → Sh
  V : Verifier Coin
  rounds : Sh → ℕ
  W : Type
  R : Rel W
  EC : Sh → Type
  [fin : ∀ s, Fintype (EC s)]
  [ne : ∀ s, Nonempty (EC s)]
  ext : (x : Msg) → (Fin (V.k x) → Coin) → EC (shape x) → RP (Fin (V.k x) → Coin) (Option W)
  κ : Sh → ℝ
  γ : Sh → ℝ

attribute [instance] Arg.fin Arg.ne

namespace Arg

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- The proof system's extractor, as a claims-core extractor. -/
def E (A : Arg Coin) : Extractor Coin A.W A.V where
  C x := A.EC (A.shape x)
  ext := A.ext

/-- **The proof system is knowledge sound** at its errors per shape: `Claims.KS` of its extractor. -/
def KS (H : Msg → Msg) (A : Arg Coin) : Prop :=
  Claims.KS H A.V A.R A.E (fun x => A.κ (A.shape x)) (fun x => A.γ (A.shape x))

/-- Its rounds are its shape's. -/
def WF (A : Arg Coin) : Prop := ∀ x, A.V.k x = A.rounds (A.shape x)

/-- **Strict reruns**: at most `ρ` of the statement's shape, on every prover, tape and coin. -/
def RerunsLe (H : Msg → Msg) (A : Arg Coin) (ρ : A.Sh → ℕ) : Prop :=
  ExtractRunsLe H A.E fun x => ρ (A.shape x)

/-- **Expected reruns**: at most `e` of the statement's shape, at every prover and tape. -/
def RerunsExpLe (H : Msg → Msg) (A : Arg Coin) (e : A.Sh → ℝ) : Prop :=
  ExtractRunsExpLe H A.E fun x => e (A.shape x)

/-- **Its own SHA-512 calls**, not its reruns': at most `h` of the statement's shape, on every prover, tape and coin. -/
def OwnLe (H : Msg → Msg) (A : Arg Coin) (h : A.Sh → ℕ) : Prop :=
  ExtractOwnLe H A.E fun x => h (A.shape x)

end Arg

/-! ### Zero knowledge with a public outcome -/

/-- **A simulator given the outcome**: its coins, and its program, which also receives the protocol's public outcome
`o ∈ St` (each verdict, and where a session stopped). -/
structure SimOut (St : Type) where
  C : Msg → Type
  [fin : ∀ x, Fintype (C x)]
  [ne : ∀ x, Nonempty (C x)]
  prog : (x : Msg) → St → C x → SP (List (Msg × Out))

attribute [instance] SimOut.fin SimOut.ne

/-- The simulator given the outcome `o`, as the spec's `Simulator`. -/
def SimOut.given {St : Type} (S : SimOut St) (o : St) : Simulator where
  C := S.C
  prog x c := S.prog x o c

/-- **Zero knowledge with a public outcome**, per outcome (`ZKmalRx`'s one-sided form at each outcome): for `(x, w) ∈ R`,
every verifier `V*` of the protocol's `k(x)` rounds and tape `θ`, every outcome `o` and every event, the honest
transcript's probability of showing the outcome `o` and the event is at most the simulator's, given `o`, of the event,
plus `ε(x, w, V*, θ, o)`. `out` reads the outcome off the transcript, so the outcome is public; the simulator is fixed
before the witness and the verifier; `ε` is a function of the statement, the witness, the verifier and the outcome, such
as a collision finder's success built from them (`Fork`), never an assumed bound. Over `n` outcomes it gives the leak
form, `n` times the simulator that draws the outcome uniformly plus `Σ_o ε` (`ZKOutLeak`). -/
def ZKOut {W St : Type} (N Ω : Msg → Type) [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
    [∀ x, Nonempty (Ω x)] (H : Msg → Msg) (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N x × Ω x)) (R : Rel W)
    (out : Msg → List (Msg × Out) → St) (S : SimOut St) (ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ) :
    Prop :=
  ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ (o : St) (Ev : List (Msg × Out) → Prop),
    Pr (fun ω : N x × Ω x =>
        out x (runStar H (P w x) Vs x ω θ).msgs = o ∧ Ev (runStar H (P w x) Vs x ω θ).msgs) ≤
      Pr (fun c : S.C x => Ev ((S.given o).out H Vs θ x c)) + ε x w Θ Vs θ o

/-- **The leak form, from the per-outcome form**: over finitely many outcomes, on every event the honest transcript's
probability is at most the number of outcomes times the simulator's that draws the outcome uniformly, plus the sum of
the outcomes' errors. -/
def ZKOutLeak : Prop :=
  ∀ {W St : Type} [Fintype St] [Nonempty St] (N Ω : Msg → Type) [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] (H : Msg → Msg) (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N x × Ω x))
    (R : Rel W) (out : Msg → List (Msg × Out) → St) (S : SimOut St)
    (ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ),
    ZKOut N Ω H k P R out S ε →
    ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ (Ev : List (Msg × Out) → Prop),
      Pr (fun ω : N x × Ω x => Ev (runStar H (P w x) Vs x ω θ).msgs) ≤
        (Fintype.card St : ℝ) * Pr (fun p : St × S.C x => Ev ((S.given p.1).out H Vs θ x p.2)) +
          ∑ o, ε x w Θ Vs θ o

/-- **A proof system for proof blocks with zero knowledge**: its honest prover at each witness, on a tape of each
statement's own nonce and rest, its public outcome (`St`, read off a transcript by `out`), its simulator given the
outcome, and its error. -/
structure ZkArg (Coin : Type) extends Arg Coin where
  Nonce : Msg → Type
  Rest : Msg → Type
  [nonceFin : ∀ x, Fintype (Nonce x)]
  [nonceNe : ∀ x, Nonempty (Nonce x)]
  [restFin : ∀ x, Fintype (Rest x)]
  [restNe : ∀ x, Nonempty (Rest x)]
  P : W → (x : Msg) → Prover (Nonce x × Rest x)
  St : Type
  out : Msg → List (Msg × Out) → St
  S : SimOut St
  εzk : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ

attribute [instance] ZkArg.nonceFin ZkArg.nonceNe ZkArg.restFin ZkArg.restNe

/-- **The proof system is zero knowledge with a public outcome**, against every verifier of its rounds, at its error. -/
def ZkArg.ZK (H : Msg → Msg) {Coin : Type} (A : ZkArg Coin) : Prop :=
  ZKOut A.Nonce A.Rest H A.V.k A.P A.R A.out A.S A.εzk

/-! ## 2. The compiled sampled verifier -/

/-- **A compiled sampled verifier over the proof system `A`** (`Sampling.Deploy` with its Flock parts replaced by `A`):
the verifier, its parser, scope check and partition, the trace's commitment; the rounds after the draw (`stage2`), so
runs that keep the other rounds make the same draw; the slots of the proof blocks a run samples (`checked`); each one's
public shape (`sh x u`, fixed by the statement); its session as a run shows it (`session`: its statement in `A` and its
view); a run's coins with the session's rounds replaced by `A`'s coins (`place`) and the session's coins (`coinsOf`);
what a session's witness reveals (`reveal`: committed values, each with the roots it is committed under); the SHA-512
calls of checking what one session's witness reveals (`openCalls`); and the reference prover on its tapes with its
completeness error. -/
structure Compiled (Coin : Type) (A : Arg Coin) where
  V : Verifier Coin
  parse : Msg → Option AuditStatement
  scopeCheck : Msg → Bool
  partition : AuditStatement → Option Partition
  commit : CommitmentScheme Bool
  stage2 : (x : Msg) → Finset (Fin (V.k x))
  checked : (x : Msg) → (Fin (V.k x) → Coin) → List ℕ
  sh : Msg → ℕ → A.Sh
  session : Msg → ℕ → Seen → Option (Msg × Seen)
  place : (x : Msg) → ℕ → (xS : Msg) → (Fin (V.k x) → Coin) → (Fin (A.V.k xS) → Coin) → Fin (V.k x) → Coin
  coinsOf : (x : Msg) → ℕ → (xS : Msg) → (Fin (V.k x) → Coin) → Fin (A.V.k xS) → Coin
  reveal : AuditStatement → ℕ → Seen → A.W → List (List Msg × Opening Bool)
  openCalls : Msg → ℕ
  Tape : Type
  [tapeNonempty : Nonempty Tape]
  P : Sampling.Wit → Prover Tape
  δc : (Msg → Msg) → Msg → ℝ

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

instance instNonemptyTape (L : Compiled Coin A) : Nonempty L.Tape := L.tapeNonempty

/-- The statement's partition, as the verifier evaluates its query; empty when the query doesn't apply. -/
def part (L : Compiled Coin A) (s : AuditStatement) : Partition := (L.partition s).getD ⟨[]⟩

/-- The proof blocks at `x`, `0` on a statement that doesn't parse. -/
def nPU (L : Compiled Coin A) (x : Msg) : ℕ :=
  match L.parse x with
  | some s => (L.part s).blocks.length
  | none => 0

/-- The extractor's waits stop at `k = 2^40·(2·n_PU + 1)` accepted runs (`Sampling.Deploy.k`). -/
def k (L : Compiled Coin A) (x : Msg) : ℕ := 2 ^ 40 * (2 * L.nPU x + 1)

/-- The waits give up after `K = 2^128·k` reruns. -/
def cap (L : Compiled Coin A) (x : Msg) : ℕ := 2 ^ 128 * L.k x

/-- Session `u`'s view of a run. -/
def view (L : Compiled Coin A) (x : Msg) (u : ℕ) (v : Seen) : Seen := ((L.session x u v).map Prod.snd).getD ⟨[], .rej⟩

/-! ## 3. The extractor -/

open Classical in
/-- **A sampled proof block's extraction at an accepted run** `(r, v)`: `A`'s extractor at the block's session, its
judged coins the session's coins of `r`, each rerun `r` with the session's rounds replaced (`place`), seen as the
session's view; then what its witness reveals that passes its opening check. At a session whose shape is not the
block's, which no accepted run shows (`Sessions.shows`), nothing. -/
noncomputable def unitExt (L : Compiled Coin A) (x : Msg) (s : AuditStatement) (u : ℕ) (r : Fin (L.V.k x) → Coin)
    (v : Seen) (c : A.EC (L.sh x u)) : RP (Fin (L.V.k x) → Coin) (List (List Msg × Opening Bool)) :=
  match L.session x u v with
  | none => .ret []
  | some (xS, _) =>
    if h : A.shape xS = L.sh x u then
      (serve (fun r' => viewing (L.view x u) (L.place x u xS r r'))
          (A.ext xS (L.coinsOf x u xS r) (cast (congrArg A.EC h.symm) c))).bind fun
        | none => .ret []
        | some w => checkRevealed L.commit (L.reveal s u v w)
    else .ret []

/-- The extractor's coins: the outer wait's `K` coin vectors; for each of its `k` accepted runs, the inner wait's `K`
(used fresh on the second stage); and for each accepted inner run and each proof block, `A`'s extractor's coins at the
block's shape. -/
abbrev ECoins (L : Compiled Coin A) (x : Msg) :=
  (Fin (L.cap x) → Fin (L.V.k x) → Coin) × (Fin (L.k x) → Fin (L.cap x) → Fin (L.V.k x) → Coin) ×
    ((j : Fin (L.k x) × Fin (L.k x) × Fin (L.nPU x)) → A.EC (L.sh x j.2.2))

/-- **The sampling extractor** (`Sampling.E`, generic): replay the judged run; if it accepted, rerun on fresh coins
until `k` accepted runs are in hand; at each, rerun fresh on the second stage until `k` accepted runs are in hand (each
wait giving up after `K`); at each of those, extract every sampled proof block (`unitExt`); and assemble. -/
noncomputable def E (L : Compiled Coin A) : Extractor Coin Sampling.Wit L.V where
  C x := ECoins L x
  ext x r₀ c := .rerun r₀ fun v₀ =>
    match L.parse x with
    | none => .ret none
    | some s =>
      if v₀.out = .acc then
        (waitAccC (L.k x) ((List.finRange (L.cap x)).map c.1) []).bind fun outer =>
          (flatMapRP (fun jo : Fin (L.k x) × ((Fin (L.V.k x) → Coin) × Seen) =>
              (waitAccC (L.k x) ((List.finRange (L.cap x)).map fun l => forkOn (L.stage2 x) jo.2.1 (c.2.1 jo.1 l))
                  []).bind fun inner =>
                flatMapRP (fun li : Fin (L.k x) × ((Fin (L.V.k x) → Coin) × Seen) =>
                    flatMapRP (fun u : ℕ =>
                        if hu : u < L.nPU x then unitExt L x s u li.2.1 li.2.2 (c.2.2 (jo.1, li.1, ⟨u, hu⟩))
                        else .ret [])
                      (L.checked x li.2.1))
                  ((List.finRange (L.k x)).zip inner))
            ((List.finRange (L.k x)).zip outer)).bind fun found => .ret (some (assemble s found))
      else .ret none

/-! ## 4. The errors -/

/-- **The sessions' knowledge error**: the largest of `A`'s `κ` at the proof blocks' shapes (Flock's `κ_F` in
`Sampling.κ`). -/
noncomputable def κs (L : Compiled Coin A) (x : Msg) : ℝ := maxBelow (L.nPU x) fun u => A.κ (L.sh x u)

/-- **The sessions' `γ` per round**: the largest of `A`'s `γ` over its rounds at the proof blocks' shapes. -/
noncomputable def γs (L : Compiled Coin A) (x : Msg) : ℝ :=
  maxBelow (L.nPU x) fun u => A.γ (L.sh x u) / A.rounds (L.sh x u)

/-- **The knowledge error at `δ`**: `δ + ε_ks`, with `ε_ks = (c² − 1)·δ + c²·κ_s + (1 + c)·(2k/K + e^{−k/8})`
(`Sampling.κ` at `κ_s`). -/
noncomputable def κ (L : Compiled Coin A) (δ : ℝ) (x : Msg) : ℝ :=
  slack ^ 2 * (δ + L.κs x) + (1 + slack) * (2 * L.k x / L.cap x + Real.exp (-(L.k x : ℝ) / 8))

/-- **`γ`**: `c⁴·k(x)·γ_s`, with `k(x)` the compiled verifier's rounds: a session's fork at one of its `k_A` rounds is
the compiled verifier's `B` at one of `k(x)`. -/
noncomputable def γ (L : Compiled Coin A) (x : Msg) : ℝ := slack ^ 4 * (L.V.k x * L.γs x)

/-- The design's miss at `t` at `x`, clamped at `0`; `0` on a statement that doesn't parse (`Sampling.missAt`). -/
noncomputable def missAt (L : Compiled Coin A) (t : ℕ) (x : Msg) : ℝ :=
  match L.parse x with
  | some s => max 0 (s.design.miss (L.part s) t)
  | none => 0

/-- The sessions' expected reruns, strict reruns and own SHA-512 calls: the largest of `A`'s at the proof blocks'
shapes. -/
noncomputable def es (L : Compiled Coin A) (e : A.Sh → ℝ) (x : Msg) : ℝ := maxBelow (L.nPU x) fun u => e (L.sh x u)

def ρs (L : Compiled Coin A) (ρ : A.Sh → ℕ) (x : Msg) : ℕ := maxBelowN (L.nPU x) fun u => ρ (L.sh x u)

def hs (L : Compiled Coin A) (h : A.Sh → ℕ) (x : Msg) : ℕ := maxBelowN (L.nPU x) fun u => h (L.sh x u)

/-- The extractor's expected work in runs, `1 + k + k²·(1 + n_PU·e_s) + k²·n_PU·(openCalls + h_s)`: its expected reruns
and its own SHA-512 calls (`Sampling.work` at `A`'s). -/
noncomputable def work (L : Compiled Coin A) (e : A.Sh → ℝ) (h : A.Sh → ℕ) (x : Msg) : ℝ :=
  1 + L.k x + (L.k x : ℝ) ^ 2 * (1 + L.nPU x * L.es e x) + (L.k x : ℝ) ^ 2 * L.nPU x * (L.openCalls x + L.hs h x)

/-- **The sampling scope, on the statement's bytes**: it parses, its query evaluates, and the partition and statement are
in scope (`Sampling.Scope`). -/
def Scope (L : Compiled Coin A) (x : Msg) : Prop :=
  ∃ s P, L.parse x = some s ∧ L.partition s = some P ∧ InScope P s

/-! ## 5. The contract -/

/-- **The compiler's contract** at `H`: the code facts the compiler's statements read.

* `draws`: the draw misses a set `B` of slots with probability at most the design's `P_a(B)`.
* `stages`: the draw is fixed before the second stage: a fork on it keeps the sampled proof blocks.
* `places`: `place` replaces only rounds of the second stage, and `coinsOf` reads back what it placed.
* `shows`: an accepted run shows each sampled proof block's session accepting, at the block's shape.
* `staged`: a fork on the second stage keeps each session's statement.
* `runs`: a run with a session's rounds replaced shows the run of a prover against `A`'s verifier at the session's
  statement on those coins, and `A`'s verifier's SHA-512 calls there are among the compiled verifier's.
* `reveals`: a witness in `A`'s relation at a session reveals openings under which its proof block is correct. -/
structure Sessions (H : Msg → Msg) (L : Compiled Coin A) : Prop where
  draws : ∀ x s P, L.parse x = some s → L.partition s = some P → ∀ B : Finset ℕ,
    Pr (fun r : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x r, u ∉ B) ≤ s.design.Pa P B
  stages : ∀ x r d, L.checked x (forkOn (L.stage2 x) r d) = L.checked x r
  places : ∀ x u xS r d, (∀ i ∉ L.stage2 x, L.place x u xS r d i = r i) ∧
    L.coinsOf x u xS (L.place x u xS r d) = d ∧ L.place x u xS r (L.coinsOf x u xS r) = r
  shows : ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) r, (run H P L.V x ω r).out = .acc → ∀ u ∈ L.checked x r,
    ∃ xS vS, L.session x u (run H P L.V x ω r).seen = some (xS, vS) ∧ vS.out = .acc ∧ A.shape xS = L.sh x u
  staged : ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) r d u xS vS xS' vS',
    L.session x u (run H P L.V x ω r).seen = some (xS, vS) →
    L.session x u (run H P L.V x ω (forkOn (L.stage2 x) r d)).seen = some (xS', vS') → xS' = xS
  runs : ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) r u xS vS, L.session x u (run H P L.V x ω r).seen = some (xS, vS) →
    ∃ Ps : Prover Unit, ∀ d,
      L.session x u (run H P L.V x ω (L.place x u xS r d)).seen = some (xS, (run H Ps A.V xS () d).seen) ∧
      ∀ e ∈ (run H Ps A.V xS () d).vLog, e ∈ (run H P L.V x ω (L.place x u xS r d)).vLog
  reveals : ∀ x s P u v xS vS w, L.parse x = some s → L.partition s = some P → L.session x u v = some (xS, vS) →
    A.R xS w → ∃ blk, P.blocks[u]? = some blk ∧
      ∃ ι : ℕ → Bool, s.prog.Correct (s.prog.refIn blk (Sampling.Wit.val H L.commit s (assemble s (L.reveal s u v w))) ι) blk

/-! ## 6. The claims of a compiled verifier -/

/-- **KS**, on the statements in scope, at every `δ ≥ 0`: `Pr[acc ∧ incorrect ∉ M_δ] ≤ δ + ε_ks + ε_bind`, with
`ε_ks = κ − δ + √(γ·Pr[B])` and `ε_bind = Pr[B_ext]` (`Sampling.KS`, generic). -/
def KS (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ δ : ℝ, 0 ≤ δ →
    KSLinOn H L.V L.Scope (Sampling.R H L.commit L.parse L.partition δ) L.E (L.κ δ) L.γ (fun _ => 1) (Bext H L.V L.E)

/-- **KS, the count form**: fewer than `t` proof blocks are incorrect, except with probability
`max(0, miss(t)) + ε_ks + ε_bind` (`Sampling.KSCount`, generic). -/
def KSCount (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ t, KSLinOn H L.V L.Scope (Sampling.RCount H L.commit L.parse L.partition t) L.E (fun x => L.κ (L.missAt t x) x)
    L.γ (fun _ => 1) (Bext H L.V L.E)

/-- **Bits**, on the statements in scope, at every `δ ≥ 0`: `λ_δ = lamLin(κ_δ, γ, 1, work)`, `work` at `A`'s expected
reruns `e` and own calls `h` (`Sampling.Bits`, generic). -/
def Bits (H : Msg → Msg) (L : Compiled Coin A) (e : A.Sh → ℝ) (h : A.Sh → ℕ) : Prop :=
  ∀ δ : ℝ, 0 ≤ δ →
    BitsLin H L.V L.Scope (Sampling.R H L.commit L.parse L.partition δ) L.E (Bext H L.V L.E)
      fun x => lamLin (L.κ δ x) (L.γ x) 1 (L.work e h x)

/-- **Efficiency**: the extractor's reruns, at most `1 + K + k·(K + k·n_PU·ρ_s)` and `1 + k + k²·(1 + n_PU·e_s)`
expected; its own SHA-512 calls, at most `k²·n_PU·(openCalls + h_s)` (`Sampling.Efficient`, generic). -/
def Efficient (H : Msg → Msg) (L : Compiled Coin A) (ρ : A.Sh → ℕ) (e : A.Sh → ℝ) (h : A.Sh → ℕ) : Prop :=
  ExtractRunsLe H L.E (fun x => 1 + L.cap x + L.k x * (L.cap x + L.k x * (L.nPU x * L.ρs ρ x))) ∧
    ExtractRunsExpLe H L.E (fun x => 1 + L.k x + (L.k x : ℝ) ^ 2 * (1 + L.nPU x * L.es e x)) ∧
    ExtractOwnLe H L.E (fun x => L.k x ^ 2 * L.nPU x * (L.openCalls x + L.hs h x))

end Compiled

/-- **A masked compiled verifier**: the reference prover's tape at each statement a nonce and the rest of the
statement's own (`tapeOf`); the audit's public outcome (`St`, read off a transcript by `out`: each sampled session's
verdict and where it stopped); and the simulator, given the trace's published leaf layer and the outcome (`S`,
`Sampling.ZkDeploy`'s with the outcome). -/
structure ZkCompiled (Coin : Type) (A : Arg Coin) extends Compiled Coin A where
  Nonce : Msg → Type
  Rest : Msg → Type
  [nonceFin : ∀ x, Fintype (Nonce x)]
  [nonceNe : ∀ x, Nonempty (Nonce x)]
  [restFin : ∀ x, Fintype (Rest x)]
  [restNe : ∀ x, Nonempty (Rest x)]
  tapeOf : (x : Msg) → Nonce x × Rest x → Tape
  St : Type
  out : Msg → List (Msg × Out) → St
  S : SimOut (List Msg × St)

attribute [instance] ZkCompiled.nonceFin ZkCompiled.nonceNe ZkCompiled.restFin ZkCompiled.restNe

namespace ZkCompiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- The masked compiled verifier's reference prover at the witness `w` and the statement `x`. -/
def prover (L : ZkCompiled Coin A) (w : Sampling.Wit) (x : Msg) : Prover (L.Nonce x × L.Rest x) :=
  (L.P w).comap (L.tapeOf x)

/-- **Zero knowledge with a public outcome, relative to the published leaf layer** (`Sampling.ZK` in `ZKOut`'s form): of
the reference prover when no proof block is incorrect, against every verifier of the compiled verifier's rounds, per
outcome, at the simulator given the trace's leaf layer `l x` and the outcome, at the error `ε`. Hiding the leaf layer is
the commitment's (`Claims.Hm96Hiding` at Flock). -/
def ZK (H : Msg → Msg) (L : ZkCompiled Coin A)
    (ε : (x : Msg) → Sampling.Wit → (Θ : Type) → VStar Θ → Θ → List Msg × L.St → ℝ) : Prop :=
  ∀ l : Msg → List Msg,
    ZKOut L.Nonce L.Rest H L.V.k L.prover
      (fun x w => Rhon H L.commit L.parse L.partition x w ∧ leafLayer H L.commit L.parse x w = l x)
      (fun x ms => (l x, L.out x ms)) L.S ε

end ZkCompiled

/-! ## 7. The compiler's statements -/

/-- **The sampled compiler is knowledge sound**: every compiled sampled verifier that meets the contract over a
knowledge-sound proof system for proof blocks, with a binding commitment, meets `KS` at the proof system's errors. -/
def CompilerKS : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), A.WF → A.KS H → L.commit.Binds H → L.Sessions H → L.KS H

/-- **The count form**: as `CompilerKS`, `KSCount`. -/
def CompilerKSCount : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), A.WF → A.KS H → L.commit.Binds H → L.Sessions H → L.KSCount H

/-- **Bits**: as `CompilerKS`, `Bits` at the proof system's expected reruns and own calls. -/
def CompilerBits : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg) (e : A.Sh → ℝ) (h : A.Sh → ℕ), A.WF → A.KS H → L.commit.Binds H → L.Sessions H →
      A.RerunsExpLe H e → A.OwnLe H h → L.Bits H e h

/-- **Efficiency**: the compiled extractor's reruns and own calls, from the proof system's. -/
def CompilerEfficient : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg) (ρ : A.Sh → ℕ) (e : A.Sh → ℝ) (h : A.Sh → ℕ), L.Sessions H →
      A.RerunsLe H ρ → A.RerunsExpLe H e → A.OwnLe H h → L.Efficient H ρ e h

/-! ## 8. Flock -/

section Flock

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- `E^rep`'s accepted reruns per wait at the level-0 lengths `N₀`, `k = ⌈4·N₀/e⌉` (`Flock.Deploy.k`). -/
noncomputable def flockK (N₀ : ℝ) : ℕ := ⌈4 * N₀ / Real.exp 1⌉₊

/-- **Flock as a proof system for proof blocks**, at a decoder: `E^rep` (`Flock.E`), its shape the numbers its coins and
errors read (waits `m′`, truncation `K`, rounds, `ε_table`, `N₀`, the weight `C`), and its errors Flock's (`Flock.κ`,
`Flock.γ`). -/
noncomputable def Arg.ofFlock (D : Claims.Flock.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    Arg Coin where
  Sh := ℕ × ℕ × ℕ × ℝ × ℝ × ℝ
  shape x := (D.waits x, D.cap x, D.V.k x, D.εtable x, D.N₀ x, D.weight x)
  V := D.V
  rounds s := s.2.2.1
  W := Claims.Flock.Wit
  R := Claims.Flock.R D.parse
  EC s := Fin s.1 → Fin s.2.1 → Fin s.2.2.1 → Coin
  ext x r c := (Claims.Flock.E D decode).ext x r c
  κ s := 4 * s.2.2.2.1 + 2 ^ (-(s.1 : ℝ)) + 2 * (flockK s.2.2.2.2.1 : ℝ) / (s.2.1 : ℝ) +
    Real.exp (-(flockK s.2.2.2.2.1 : ℝ) / 8)
  γ s := 16 * (Real.sqrt (2 * (s.2.2.1 : ℝ) * s.2.2.2.2.2) + (s.2.2.1 : ℝ)) ^ 2

/-- **Flock's KS is `Arg.ofFlock`'s**: `Flock.KS` at a decoder within Flock's radius. -/
def FlockArgKS : Prop :=
  ∀ (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col] (decode : List (ℕ × Col) → List T),
    Claims.Flock.KS H D → Claims.Flock.ListWithin D.len D.enc D.θ decode → (Arg.ofFlock D decode).KS H

/-- **The sampled compiler at Flock**: every compiled sampled verifier over Flock at a decoder within its radius that
meets the contract, with a binding commitment, meets `KS`, from `Flock.KS` (`CompilerKS` and `FlockArgKS`). -/
def FlockKS : Prop :=
  ∀ (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col] (decode : List (ℕ × Col) → List T)
    (L : Compiled Coin (Arg.ofFlock D decode)), Claims.Flock.KS H D →
      Claims.Flock.ListWithin D.len D.enc D.θ decode → L.commit.Binds H → L.Sessions H → L.KS H

end Flock

end Claims.SamplingGeneric
