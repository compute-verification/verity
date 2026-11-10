module

public import Definitions.Claims.RecursionGeneric.Comp
public import Definitions.Claims.RecursionGeneric.Session
public import Definitions.Claims.SamplingGeneric

/-!
# Π_out, abstract, and the compiler over it

Π_out is a proof system for proof blocks as the sampled audit reads it (`Claims.SamplingGeneric.Arg`): a public-coin
verifier over byte strings, its relation, its errors per statement, and its extractor, `E^rep` at its counted parts
(`Arg.ext`: replay the judged run, then waits over fresh coins, each rerun's view opened by `opens` and a wait's
openings assembled by `assemble`, a counted program, and checked). Its coins (`Arg.EC`) are a type the statement's
public shape (`shape`) fixes. The compiled verifier stages each of Π_out's sessions' statements during its run, so only
its shape is fixed by the compiled statement: the extractor's coins are the shape's, and the compiled verifier bounds
each session's errors by numbers the compiled statement fixes (`κb`, `γb`).

`Compiled Coin A` is a compiled verifier that runs, after Π_in's rounds and the staging rounds (`pre`), `nS` sessions of
`A`'s verifier at the statements its staging rounds fix (`stmts`), session `s` at the public shape `sh x s` with its
errors within `κb x s`, `γb x s`, in order. A compiler (`Compiler`) makes one from the verifier its sessions prove;
`compile C Π S` is it at `Open(Π, S)`, Π_in's verifier over the ZK server's commitments.

`Compiled.EO` is the extraction of Π_out's witnesses: replay the judged run and, at each session in order, `A`'s
extractor at the session's staged statement, its reruns forks of the judged run at the session's first round, each seen
as the session's view. `Compiled.toComp` is the composition's input (`Comp`), with `κ_O = Σ_s κb_s` and
`γ_O = (Σ_s √(γb_s·k(x)))²`. `Compiled.Sessions` is what the compiler must give for it.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

/-- **A compiled verifier over Π_out `A`**: the verifier; whether a statement parses into the relation's form and its
statement of Π_in; the rounds before the sessions (`pre`, Π_in's rounds and the staging rounds); the sessions' count and
public shapes (`sh x s`) and the bounds on their errors (`κb x s`, `γb x s`: on `A`'s `κ` and `γ` at the session's
staged statement, `Bounded`); the sessions' statements the run's messages stage (`stmts`); session `s`'s view of a run
(`session`); Π_in's messages a run's sessions' statements and witnesses give (`msgs`); the SHA-512 evaluations a finder
searches at a rerun (`calls`, as `Comp.calls`); and the compiled verifier's own at a run's messages (`own`). -/
structure Compiled (Coin : Type) (A : Arg Coin) where
  V : Verifier Coin
  parses : Msg → Prop
  xI : Msg → Msg
  pre : Msg → ℕ
  nS : Msg → ℕ
  sh : Msg → ℕ → A.Sh
  κb : Msg → ℕ → ℝ
  γb : Msg → ℕ → ℝ
  stmts : Msg → List (Msg × Out) → List Msg
  session : Msg → ℕ → Seen → Seen
  msgs : Msg → List (Msg × A.W) → List Msg
  calls : (Msg → Msg) → (x : Msg) → (Fin (V.k x) → Coin) → List (Msg × A.W) → List (Msg × Msg)
  own : (Msg → Msg) → Msg → List (Msg × Out) → List (Msg × Msg)

/-- **A compiler**: from the verifier its sessions prove, the compiled verifier over Π_out `A`. -/
def Compiler (Coin : Type) (A : Arg Coin) : Type := Verifier Coin → Compiled Coin A

/-- **`compile(Π, A, S)`**: the compiler at `Open(Π, S)`, Π_in's verifier over the server's commitments. -/
def compile {Coin W : Type} [Inhabited Coin] {A : Arg Coin} (C : Compiler Coin A) (I : Proto Coin W) (S : Server) :
    Compiled Coin A :=
  C (S.IVStar I.V I.ans)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **Session `s`'s first round**: after the rounds before the sessions and the earlier sessions' rounds, each its
public shape's. -/
def off (L : Compiled Coin A) (x : Msg) (s : ℕ) : ℕ :=
  L.pre x + ((List.range s).map fun t => A.rounds (L.sh x t)).sum

/-- A session's coins placed from round `o` on of the compiled verifier's (`default` past them). -/
def placeAt {k n : ℕ} (o : ℕ) (r : Fin n → Coin) : Fin k → Coin := fun i => coinSeq r (i.val - o)

open Classical in
/-- **Session `s` of `E_O` at its staged statement `xO`**: where `xO` has the session's public shape, `A`'s extractor
at `xO`, its judged coins the judged run's from the session's first round on, each rerun the fork of the judged run at
that round onto the rerun's coins, seen as the session's view; its witness paired with `xO`. Where the shape differs
(which no accepted run shows, `Shapes`), none. -/
noncomputable def sessionProg (L : Compiled Coin A) (x : Msg) (r₀ : Fin (L.V.k x) → Coin) (s : Fin (L.nS x))
    (xO : Msg) (c : A.EC (L.sh x s)) : RP (Fin (L.V.k x) → Coin) (Option (Msg × A.W)) :=
  if h : A.shape xO = L.sh x s then
    (serve (fun r' : Fin (A.V.k xO) → Coin => viewing (L.session x s) (fork (L.off x s) r₀ (placeAt (L.off x s) r')))
        (A.ext xO (sessCoins (L.off x s) r₀ (A.V.k xO)) (cast (congrArg A.EC h.symm) c))).bind
      fun o => .ret (o.map fun w => (xO, w))
  else .ret none

/-- **`E_O`, the extraction of Π_out's witnesses**: replay the judged run; if it accepted, each session's extraction in
order (`sessionProg`) at the statement the run staged for it; all their witnesses with their statements, or none. -/
noncomputable def EO (L : Compiled Coin A) : Extractor Coin (List (Msg × A.W)) L.V where
  C x := (s : Fin (L.nS x)) → A.EC (L.sh x s)
  ext x r₀ c := .rerun r₀ fun v₀ =>
    if v₀.out = .acc then
      collect ((List.finRange (L.nS x)).map fun s => L.sessionProg x r₀ s ((L.stmts x v₀.msgs).getD s []) (c s))
    else .ret none

/-- **The composition's input** at `L`: `E_O` (`EO`) for the relation "every session's witness is in `A`'s relation at
its statement", with `κ_O = Σ_s κb_s` and `γ_O = (Σ_s √(γb_s·k(x)))²`, the sessions' bounds on `A`'s errors (a
session's fork finder guesses one of the compiled verifier's `k(x)` rounds). -/
noncomputable def toComp (L : Compiled Coin A) : Comp Coin where
  V := L.V
  xI := L.xI
  parses := L.parses
  WO := List (Msg × A.W)
  RO _ ws := ∀ p ∈ ws, A.R p.1 p.2
  EO := L.EO
  msgs := L.msgs
  calls := L.calls
  own := L.own
  κO x := ∑ s : Fin (L.nS x), L.κb x s
  γO x := (∑ s : Fin (L.nS x), Real.sqrt (L.γb x s * L.V.k x)) ^ 2

/-! ## The compiler's contract for `E_O` (code facts) -/

/-- The staged statements read only the rounds before the sessions. -/
def Stages (L : Compiled Coin A) : Prop := ∀ x ms, L.stmts x ms = L.stmts x (ms.take (L.pre x))

/-- On an accepted run, each session's staged statement has the session's public shape. -/
def Shapes (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (L.V.k x) → Coin), (run H P L.V x ω r).out = .acc →
    ∀ s : Fin (L.nS x), A.shape ((L.stmts x (run H P L.V x ω r).msgs).getD s []) = L.sh x s

/-- **The sessions' errors are within their bounds**: the bounds are nonnegative, and on an accepted run `A`'s errors
at each session's staged statement are at most the session's (`κb`, `γb`). -/
def Bounded (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ (x : Msg) (s : Fin (L.nS x)), 0 ≤ L.κb x s ∧ 0 ≤ L.γb x s ∧
    ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (L.V.k x) → Coin), (run H P L.V x ω r).out = .acc →
      A.κ ((L.stmts x (run H P L.V x ω r).msgs).getD s []) ≤ L.κb x s ∧
        A.γ ((L.stmts x (run H P L.V x ω r).msgs).getD s []) ≤ L.γb x s

/-- The sessions end by the compiled verifier's last round. -/
def Fits (L : Compiled Coin A) : Prop := ∀ x, L.off x (L.nS x) ≤ L.V.k x

/-- An accepted run shows each session accepting. -/
def Accepts (L : Compiled Coin A) : Prop :=
  ∀ (x : Msg) (s : ℕ) (v : Seen), s < L.nS x → v.out = .acc → (L.session x s v).out = .acc

/-- **The compiled verifier runs session `s` as `A`'s verifier at its staged statement**: on a run that accepts, each
fork at the session's first round shows through `session` the run of the prover's continuation after the run's earlier
messages (`sessProver`) against `A`'s verifier at the fork's coins from that round on, and `A`'s verifier's SHA-512
calls in that run are among the compiled verifier's. -/
def Runs (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ (x : Msg) (s : Fin (L.nS x)) (Ω : Type) (P : Prover Ω) (ω : Ω) (r d : Fin (L.V.k x) → Coin),
    (run H P L.V x ω r).out = .acc →
    (run H (sessProver P x ω ((run H P L.V x ω r).msgs.take (L.off x s))) A.V
        ((L.stmts x (run H P L.V x ω r).msgs).getD s []) ()
        (sessCoins (L.off x s) d _)).seen =
      L.session x s (run H P L.V x ω (fork (L.off x s) r d)).seen ∧
    ∀ e ∈ (run H (sessProver P x ω ((run H P L.V x ω r).msgs.take (L.off x s))) A.V
        ((L.stmts x (run H P L.V x ω r).msgs).getD s []) () (sessCoins (L.off x s) d _)).vLog,
      e ∈ (run H P L.V x ω (fork (L.off x s) r d)).vLog

/-- **The compiler's contract for `E_O`**: the code facts the extraction's knowledge soundness reads. -/
structure Sessions (H : Msg → Msg) (L : Compiled Coin A) : Prop where
  stages : L.Stages
  shapes : L.Shapes H
  bounded : L.Bounded H
  fits : L.Fits
  accepts : L.Accepts
  runs : L.Runs H

end Compiled

end Claims.RecGeneric
