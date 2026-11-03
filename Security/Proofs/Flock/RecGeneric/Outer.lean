module

public import Proofs.Flock.RecGeneric.Comp
public import Proofs.Flock.Claims.Recursion.OuterKS
public import Proofs.Flock.Claims.Recursion.Serve
public import Definitions.Claims.SamplingGeneric

/-!
# The outer argument, abstract, and the compiler over it

The outer argument is a proof system for proof blocks as the sampled audit reads it (`Claims.SamplingGeneric.Arg`): a
public-coin verifier over byte strings, its relation, and an extractor whose coins and errors are functions of the
statement's public shape (`shape`). The compiled verifier stages each outer session's statement during its run, so only
its shape is fixed by the compiled statement: the extractor's coins must be a type the shape fixes, and its errors
numbers the shape fixes.

The extractor's interface is stated, not assumed: `Arg.KS` is its knowledge soundness in `Claims.KS`'s form (`κ` plus
`√(γ·Pr[B])`, `B` the outer verifier's fork finder); `Arg.RerunsLe` and `Arg.RerunsExpLe` bound its reruns, strictly
or in expectation; and `Arg.OwnLe` bounds the SHA-512 calls it makes itself, beside its reruns'. Running time is reruns
and calls; the pure computation between them is not counted (`Claims.ExtractRunsLe`).

`Compiled Coin A` is a compiled verifier that runs, after its inner and staging rounds (`pre`), `nS` sessions of `A`'s
verifier at the statements its staging rounds fix (`stmts`), session `s` at the public shape `sh x s`, in order. A
compiler (`Compiler`) makes one from the inner verifier its sessions prove; `compile C Π S` is it at `Open(Π, S)`.

`Compiled.EO` is the outer extraction: replay the judged run and, at each session in order, `A`'s extractor at the
session's staged statement, its reruns forks of the judged run at the session's first round, each seen as the session's
view. `Compiled.toComp` is the composition's input (`Comp`), with `κ_O = Σ_s κ_A(sh_s)` and
`γ_O = (Σ_s √(γ_A(sh_s)·k(x)))²`. `Compiled.Sessions` is what the compiler must give for it; `OuterSessionsKS` is that
`E_O` then meets `Claims.KS` (lemma O1, the generic form of `Claims.Recursion.outerKS_of_flockKS`).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

/-- **A compiled verifier over the outer argument `A`**: the verifier; whether a statement parses into the relation's
form and its inner statement; the rounds before the sessions (`pre`, the inner rounds and the staging rounds); the
sessions' count and public shapes (`sh x s`); the sessions' statements the run's messages stage (`stmts`); session `s`'s
view of a run (`session`); the inner messages a run's sessions' statements and witnesses give (`msgs`); the SHA-512
evaluations a finder searches at a rerun (`calls`, as `Comp.calls`); and the compiled verifier's own at a run's messages
(`own`). -/
structure Compiled (Coin : Type) (A : Arg Coin) where
  V : Verifier Coin
  parses : Msg → Prop
  xI : Msg → Msg
  pre : Msg → ℕ
  nS : Msg → ℕ
  sh : Msg → ℕ → A.Sh
  stmts : Msg → List (Msg × Out) → List Msg
  session : Msg → ℕ → Seen → Seen
  msgs : Msg → List (Msg × A.W) → List Msg
  calls : (Msg → Msg) → (x : Msg) → (Fin (V.k x) → Coin) → List (Msg × A.W) → List (Msg × Msg)
  own : (Msg → Msg) → Msg → List (Msg × Out) → List (Msg × Msg)

/-- **A compiler**: from the inner verifier its sessions prove, the compiled verifier over `A`. -/
def Compiler (Coin : Type) (A : Arg Coin) : Type := Verifier Coin → Compiled Coin A

/-- **`compile(Π, A, S)`**: the compiler at `Open(Π, S)`, the inner verifier over the server's commitments. -/
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

/-- **`E_O`, the outer extraction**: replay the judged run; if it accepted, each session's extraction in order
(`sessionProg`) at the statement the run staged for it; all their witnesses with their statements, or none. -/
noncomputable def EO (L : Compiled Coin A) : Extractor Coin (List (Msg × A.W)) L.V where
  C x := (s : Fin (L.nS x)) → A.EC (L.sh x s)
  ext x r₀ c := .rerun r₀ fun v₀ =>
    if v₀.out = .acc then
      collect ((List.finRange (L.nS x)).map fun s => L.sessionProg x r₀ s ((L.stmts x v₀.msgs).getD s []) (c s))
    else .ret none

/-- **The composition's input** at `L`: `E_O` (`EO`) for the relation "every session's witness is in `A`'s relation at
its statement", with `κ_O = Σ_s κ_A(sh_s)` and `γ_O = (Σ_s √(γ_A(sh_s)·k(x)))²` (a session's fork finder guesses one of
the compiled verifier's `k(x)` rounds). -/
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
  κO x := ∑ s : Fin (L.nS x), A.κ (L.sh x s)
  γO x := (∑ s : Fin (L.nS x), Real.sqrt (A.γ (L.sh x s) * L.V.k x)) ^ 2

/-! ## The compiler's contract for the outer extraction (code facts) -/

/-- The staged statements read only the rounds before the sessions. -/
def Stages (L : Compiled Coin A) : Prop := ∀ x ms, L.stmts x ms = L.stmts x (ms.take (L.pre x))

/-- On an accepted run, each session's staged statement has the session's public shape. -/
def Shapes (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (L.V.k x) → Coin), (run H P L.V x ω r).out = .acc →
    ∀ s : Fin (L.nS x), A.shape ((L.stmts x (run H P L.V x ω r).msgs).getD s []) = L.sh x s

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

/-- **The compiler's contract for `E_O`**: the code facts `OuterSessionsKS` reads. -/
structure Sessions (H : Msg → Msg) (L : Compiled Coin A) : Prop where
  stages : L.Stages
  shapes : L.Shapes H
  fits : L.Fits
  accepts : L.Accepts
  runs : L.Runs H

end Compiled

/-- **O1: the outer extraction is knowledge sound, from the outer argument's** (the generic form of
`Claims.Recursion.outerKS_of_flockKS`): where `A` is knowledge sound at nonnegative errors and the compiler meets
`Sessions`, `E_O` meets `Claims.KS` at `κ_O`, `γ_O`. Per session: the session's prover after the run's earlier messages
(`sessProver`) on the judged run's coins before the session; `A`'s KS at its staged statement, whose shape is the
session's; Jensen over the earlier coins; its fork finder is the compiled verifier's `B` at a round guess among `k(x)`;
a union over sessions (`collect`). -/
def OuterSessionsKS : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (A : Arg Coin)
    (L : Compiled Coin A), A.WF → (∀ s, 0 ≤ A.κ s) → (∀ s, 0 ≤ A.γ s) → A.KS H → L.Sessions H →
      Claims.KS H L.V L.toComp.RO L.toComp.EO L.toComp.κO L.toComp.γO

/-- **The composite extractor's cost** (lemma O2): `E*` (`Comp.E` at `L.toComp`) reruns the prover at most
`(1 + m_Π·K_Π)·(1 + Σ_s ρ(sh_s))` times on every coin where `A` reruns at most `ρ`; in expectation at most
`(1 + m_Π·k_Π)·(1 + Σ_s e(sh_s))` where `A` reruns at most `e` in expectation (Wald, over `E^rep`'s fresh reruns); and
it makes at most `(1 + m_Π·K_Π)·(Σ_s h(sh_s) + n)` SHA-512 calls of its own (`Sampling.ExtractOwnLe`) where `A` makes at
most `h` (`Arg.OwnLe`) and `Open(Π, S)` at most `n` on any witnesses' inner messages (`Comp.innerLog`): each of
`E^rep`'s reruns is `E_O`'s sessions and the inner view's run, whose calls are `E*`'s own (`Comp.decProg`). The inner
`E^rep`'s and `E_O`'s reruns multiply; their errors do not (`Comp.κ`: `N_I` times `κ_O`). -/
def ExtractCost : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (A : Arg Coin)
    (L : Compiled Coin A) (I : Proto Coin W) (S : Server) (ρ h : A.Sh → ℕ) (e : A.Sh → ℝ) (n : Msg → ℕ),
    A.WF → L.Sessions H →
      (A.RerunsLe H ρ → ExtractRunsLe H (L.toComp.E I S) fun x =>
        (1 + I.waits (L.xI x) * I.cap (L.xI x)) * (1 + ∑ s : Fin (L.nS x), ρ (L.sh x s))) ∧
      (A.RerunsExpLe H e → ExtractRunsExpLe H (L.toComp.E I S) fun x =>
        (1 + I.waits (L.xI x) * I.k (L.xI x) : ℝ) * (1 + ∑ s : Fin (L.nS x), e (L.sh x s))) ∧
      (A.OwnLe H h → (∀ x r ws, (L.toComp.innerLog (S.IVStar I.V I.ans) H x r ws).length ≤ n x) →
        Claims.Sampling.ExtractOwnLe H (L.toComp.E I S) fun x =>
          (1 + I.waits (L.xI x) * I.cap (L.xI x)) * (∑ s : Fin (L.nS x), h (L.sh x s) + n x))

end Claims.RecGeneric
