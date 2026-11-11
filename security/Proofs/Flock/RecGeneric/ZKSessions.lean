module

public import Proofs.Flock.RecGeneric.ZK
public import Proofs.Flock.Claims.ZK.Compose

/-!
# The outer phase as sessions in turn, zero knowledge per outcome

The outer phase of the ZK server's release (`ZkRelease`) at Flock is the V\* sessions: one zero-knowledge session system
run in turn, each session at a statement the outer statement fixes before any session runs, on its own tape. This file
states that such an outer phase is zero knowledge per outcome (`Claims.ZKOut`), from each session's, as `ZkRelease.OuterZK`
reads it:

* **the session system** (`ZkSess`): the session prover at its witness (`PS`, `kS` rounds at a session statement), its
  public outcome (`outS`), and, at each compiled statement `x`, the session relation (`RS x`), the session's tape
  `NS x × ΩS x` and its simulator's coins `CS x` and program (`progS x`, given the session statement and outcome), the
  same at every session statement under `x`. Its zero knowledge (`ZkSess.ZK`) is `ZKOut` at each `x` within `ε_S`. The
  composed prover samples its tapes, and the composed simulator its coins, before the outer statement fixes the session
  statements, so their types are `x`'s; `RS x` may restrict the session statements to those whose own types match.
* **the layout** (`Sessions`): at `x`, `n x` sessions, session `i` at `xs x x_O i`, and `ko x` rounds in all, which the
  sessions' rounds make up at every outer statement (`Sessions.Shaped`, an equation, met by any number of sessions).
* **the composition**: the prover runs session `i` on its tape at its own history (`Sessions.PO`); the outcome is each
  session's, `none` for one the run doesn't reach (`Sessions.outO`); the simulator runs each session's simulator given
  its outcome, its questions after the messages so far, while the run goes on (`Sessions.simO`); the relation is every
  session's (`Sessions.RO`), a conjunction.

`GenericSessionsZK` (Z1-S): the composition is zero knowledge per outcome within the exact sum
`Σ_i E_{t_{<i}}[[the real prefix shows o_{<i} and reaches i] ε_S(x_i, w_i, V* after the real prefix, o_i)]`
(`Sessions.err`), the hybrid over the sessions in turn. `GenericSessionsZKCost`: its simulator makes at most `n` times
the session simulator's calls, so it is counted where the session simulator is.

`GenericRecursiveSessionsZK` and `GenericRecursiveSessionsZKCost` are Z1 and Z1-cost at the release of a first part
(`FirstPart`) and these sessions (`ZkRelease.ofSessions`).

A proof system's simulator, Flock's server's among them, flips coins of a type its statement fixes, which a session
statement's commitments may change. `ZkSess.ofZKOut` is its session system at finite sets `Xs x` of the session statements
each compiled statement admits: the session simulator under `x` samples one of the system's tapes for each statement in
`Xs x` and runs the system's simulator on its own statement's. `GenericSessionsOfZKOut` (Z1-L): that is zero knowledge
per outcome within the system's own error, and `GenericSessionsOfZKOutCost` (Z1-L-cost) counted at the system's calls.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- **A zero-knowledge session system at compiled statements**: the session prover at the witness `WS` (`PS`, on the
tape `NS x × ΩS x` under the compiled statement `x`, `kS` rounds at a session statement), its relation under `x` (`RS`),
its public outcome (`outS`), and its simulator under `x`: coins `CS x` and a program given the session statement and the
outcome (`progS x`). -/
structure ZkSess where
  WS : Type
  RS : Msg → Rel WS
  kS : Msg → ℕ
  NS : Msg → Type
  ΩS : Msg → Type
  [nFin : ∀ x, Fintype (NS x)]
  [nNe : ∀ x, Nonempty (NS x)]
  [oFin : ∀ x, Fintype (ΩS x)]
  [oNe : ∀ x, Nonempty (ΩS x)]
  PS : WS → (x : Msg) → Prover (NS x × ΩS x)
  StS : Type
  outS : Msg → List (Msg × Out) → StS
  CS : Msg → Type
  [cFin : ∀ x, Fintype (CS x)]
  [cNe : ∀ x, Nonempty (CS x)]
  progS : (x : Msg) → Msg → StS → CS x → SP (List (Msg × Out))

attribute [instance] ZkSess.nFin ZkSess.nNe ZkSess.oFin ZkSess.oNe ZkSess.cFin ZkSess.cNe

namespace ZkSess

/-- The session simulator under the compiled statement `x`, on the coins `CS x` at every session statement. -/
def simAt (S : ZkSess) (x : Msg) : SimOut S.StS where
  C _ := S.CS x
  prog xs o c := S.progS x xs o c

/-- **The sessions are zero knowledge per outcome** under each compiled statement `x`: `ZKOut` of the session prover on
`NS x × ΩS x`, against every verifier of the session's `kS` rounds, on every witness in `RS x`, at `simAt x` and
`εS x`. -/
def ZK (S : ZkSess) (H : Msg → Msg) (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ) : Prop :=
  ∀ x, ZKOut (fun _ => S.NS x) (fun _ => S.ΩS x) H S.kS (fun w _ => S.PS w x) (S.RS x) S.outS (S.simAt x) (εS x)

/-- `m` sessions' tapes under `x`. -/
def tapes (S : ZkSess) (x : Msg) : ℕ → Type
  | 0 => Unit
  | m + 1 => (S.NS x × S.ΩS x) × tapes S x m

/-- `m` sessions' simulator coins under `x`. -/
def coins (S : ZkSess) (x : Msg) : ℕ → Type
  | 0 => Unit
  | m + 1 => S.CS x × coins S x m

instance tapesFin (S : ZkSess) (x : Msg) : ∀ m, Fintype (S.tapes x m)
  | 0 => inferInstanceAs (Fintype Unit)
  | m + 1 => letI := tapesFin S x m; inferInstanceAs (Fintype ((S.NS x × S.ΩS x) × S.tapes x m))

instance tapesNe (S : ZkSess) (x : Msg) : ∀ m, Nonempty (S.tapes x m)
  | 0 => inferInstanceAs (Nonempty Unit)
  | m + 1 => letI := tapesNe S x m; inferInstanceAs (Nonempty ((S.NS x × S.ΩS x) × S.tapes x m))

instance coinsFin (S : ZkSess) (x : Msg) : ∀ m, Fintype (S.coins x m)
  | 0 => inferInstanceAs (Fintype Unit)
  | m + 1 => letI := coinsFin S x m; inferInstanceAs (Fintype (S.CS x × S.coins x m))

instance coinsNe (S : ZkSess) (x : Msg) : ∀ m, Nonempty (S.coins x m)
  | 0 => inferInstanceAs (Nonempty Unit)
  | m + 1 => letI := coinsNe S x m; inferInstanceAs (Nonempty (S.CS x × S.coins x m))

/-- **The session system of a zero-knowledge system whose simulator's coins are its statement's**, at the finite sets
`Xs x` of session statements each compiled statement admits: the session relation under `x` is `R` at the statements in
`Xs x`; the session simulator's coins under `x` are one tape of `S` for each statement in `Xs x`, and at a session
statement in `Xs x` it runs `S` on that statement's tape (elsewhere it sends nothing, and no witness is in the
relation there). -/
def ofZKOut {W : Type} (R : Rel W) (k : Msg → ℕ) (N Ω : Type) [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω]
    (P : W → Prover (N × Ω)) {St : Type} (out : Msg → List (Msg × Out) → St) (S : SimOut St)
    (Xs : Msg → Finset Msg) : ZkSess where
  WS := W
  RS x xs w := xs ∈ Xs x ∧ R xs w
  kS := k
  NS _ := N
  ΩS _ := Ω
  PS w _ := P w
  StS := St
  outS := out
  CS x := (v : Xs x) → S.C v
  progS x xs o c := if h : xs ∈ Xs x then S.prog xs o (c ⟨xs, h⟩) else .ret []

end ZkSess

/-- **`V*` after the prover messages `pre`**, as a verifier of `k` rounds at any statement: its answers at `x_O` after
`pre`. -/
def vUnder {Θ : Type} (Vs : VStar Θ) (xO : Msg) (pre : List Msg) (k : ℕ) : VStar Θ where
  k _ := k
  next _ as θ := Vs.next xO (pre ++ as) θ

/-- **The sessions' layout** for the session system `S`: at the compiled statement `x`, `n x` sessions, session `i` at
the statement `xs x x_O i` that the outer statement `x_O` fixes, and `ko x` rounds in all. -/
structure Sessions (S : ZkSess) where
  n : Msg → ℕ
  xs : Msg → Msg → ℕ → Msg
  ko : Msg → ℕ

namespace Sessions

variable {S : ZkSess} (L : Sessions S)

/-- The rounds of the `m` sessions from session `i` at `(x, x_O)`. -/
def rounds (x xO : Msg) : ℕ → ℕ → ℕ
  | 0, _ => 0
  | m + 1, i => S.kS (L.xs x xO i) + rounds x xO m (i + 1)

/-- **The sessions make up the outer phase's rounds** at every outer statement. -/
def Shaped : Prop := ∀ x xO, L.rounds x xO (L.n x) 0 = L.ko x

/-- The composed prover's message, `m` sessions from session `i` at `(x, x_O)`, at the sessions' witnesses `wo`: within
session `i`'s rounds, its prover at its statement, its own history and its tape; after them, the rest's. -/
def cnext (x xO : Msg) (wo : ℕ → S.WS) : (m i : ℕ) → List (Msg × Msg) → S.tapes x m → OC Msg
  | 0, _, _, _ => .ret []
  | m + 1, i, hist, t =>
    if hist.length < S.kS (L.xs x xO i) then (S.PS (wo i) x).next (L.xs x xO i) hist t.1
    else cnext x xO wo m (i + 1) (hist.drop (S.kS (L.xs x xO i))) t.2

/-- **The composed prover** at the sessions' witnesses `wo` under `x`, at the outer statement it is run at. -/
def PO (wo : ℕ → S.WS) (x : Msg) : Prover (S.tapes x (L.n x) × Unit) where
  next xO hist t := L.cnext x xO wo (L.n x) 0 hist t.1

/-- **The composed relation**: every session's witness is in `RS x` at its statement. -/
def RO (x : Msg) : Rel (ℕ → S.WS) := fun xO wo => ∀ i < L.n x, S.RS x (L.xs x xO i) (wo i)

/-- The outcomes of the `m` sessions from session `i` on the transcript `τ` of their rounds: each one's at its
statement on its rounds where the run reaches it, and `none` for each after a session that stops the run. -/
def outs (x xO : Msg) : (m i : ℕ) → List (Msg × Out) → List (Option S.StS)
  | 0, _, _ => []
  | m + 1, i, τ =>
    some (S.outS (L.xs x xO i) (τ.take (S.kS (L.xs x xO i)))) ::
      if ZKMal.goesOn (S.kS (L.xs x xO i)) (τ.take (S.kS (L.xs x xO i))) then
        outs x xO m (i + 1) (τ.drop (S.kS (L.xs x xO i)))
      else List.replicate m none

/-- **The composed outcome**: every session's, `none` where the run doesn't reach it. -/
def outO (x xO : Msg) (τ : List (Msg × Out)) : List (Option S.StS) := L.outs x xO (L.n x) 0 τ

/-- The `m` sessions from session `i` simulated after the transcript `ms`, given their outcomes: each session's
simulator given its outcome, its questions after the messages so far, on its coins, while the run goes on. -/
def csim (x xO : Msg) : (m i : ℕ) → List (Option S.StS) → List (Msg × Out) → S.coins x m → SP (List (Msg × Out))
  | 0, _, _, ms, _ => .ret ms
  | m + 1, i, some a :: os, ms, c =>
    spBind (spUnder (ms.map Prod.fst) (S.progS x (L.xs x xO i) a c.1)) fun τ =>
      if ZKMal.goesOn (S.kS (L.xs x xO i)) τ then csim x xO m (i + 1) os (ms ++ τ) c.2 else .ret (ms ++ τ)
  | _ + 1, _, _, ms, _ => .ret ms

/-- The composed simulator's program under `x`, at the outer statement `x_O`, given the sessions' outcomes. -/
def progO (x xO : Msg) (o : List (Option S.StS)) (c : S.coins x (L.n x)) : SP (List (Msg × Out)) :=
  L.csim x xO (L.n x) 0 o [] c

/-- **The composed simulator** under `x`, on every session's coins, at every outer statement. -/
def simO (x : Msg) : SimOut (List (Option S.StS)) where
  C _ := S.coins x (L.n x)
  prog xO o c := L.progO x xO o c

open Classical in
/-- The composed error of the `m` sessions from session `i` after the transcript `ms`, given their outcomes: session
`i`'s error at its statement, its witness and `V*` after `ms`, at its outcome; plus the rest's after `ms` and session
`i`'s real transcript, averaged over session `i`'s tape where that transcript shows its outcome and goes on. -/
noncomputable def errAt (H : Msg → Msg) (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ)
    (x xO : Msg) (wo : ℕ → S.WS) (Θ : Type) (Vs : VStar Θ) (θ : Θ) :
    (m i : ℕ) → List (Msg × Out) → List (Option S.StS) → ℝ
  | 0, _, _, _ => 0
  | m + 1, i, ms, some a :: os =>
    εS x (L.xs x xO i) (wo i) Θ (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i))) θ a +
      Ex fun t : S.NS x × S.ΩS x =>
        if S.outS (L.xs x xO i) (runStar H (S.PS (wo i) x)
              (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i))) (L.xs x xO i) t θ).msgs = a ∧
            ZKMal.goesOn (S.kS (L.xs x xO i)) (runStar H (S.PS (wo i) x)
              (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i))) (L.xs x xO i) t θ).msgs = true then
          errAt H εS x xO wo Θ Vs θ m (i + 1)
            (ms ++ (runStar H (S.PS (wo i) x) (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i)))
              (L.xs x xO i) t θ).msgs) os
        else 0
  | _ + 1, _, _, _ => 0

/-- **The composed error** under `x` at `(x_O, w_O, V*, θ)` and the outcomes `o`: the sum over the sessions of each
one's error at `V*` after the real run's messages before it, averaged over the earlier sessions' tapes where their real
transcripts show their outcomes in `o` and go on. -/
noncomputable def err (H : Msg → Msg) (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ)
    (x xO : Msg) (wo : ℕ → S.WS) (Θ : Type) (Vs : VStar Θ) (θ : Θ) (o : List (Option S.StS)) : ℝ :=
  L.errAt H εS x xO wo Θ Vs θ (L.n x) 0 [] o

end Sessions

/-- **A release's first part**, whose outer witness is in `WO`: `ZkRelease`'s rounds of leaves in their frames, the
workers' values, the outer statement and the outer witness, without the outer phase. -/
structure FirstPart (W : Type) (Lv : Leaves) (WO : Type) where
  nh : Msg → ℕ
  nl : Msg → ℕ → ℕ
  frame : Msg → List (Msg × Out) → List Msg → OC Msg
  Ωp : Msg → Type
  [pFin : ∀ x, Fintype (Ωp x)]
  [pNe : ∀ x, Nonempty (Ωp x)]
  msg : W → (x : Msg) → Ωp x → List (Msg × Msg) → OC (List Msg)
  xo : Msg → List (Msg × Out) → Msg
  wo : W → (x : Msg) → Ωp x → (Fin (∑ i ∈ Finset.range (nh x), nl x i) → Lv.Salt) → List (Msg × Out) → WO

attribute [instance] FirstPart.pFin FirstPart.pNe

/-- **The release whose outer phase is the sessions in turn**: the first part `F`, then the layout `L`'s composed
prover, relation, outcome and simulator. -/
def ZkRelease.ofSessions {W : Type} {Lv : Leaves} {S : ZkSess} (F : FirstPart W Lv (ℕ → S.WS)) (L : Sessions S) :
    ZkRelease W Lv where
  nh := F.nh
  nl := F.nl
  frame := F.frame
  Ωp := F.Ωp
  msg := F.msg
  ko := L.ko
  xo := F.xo
  WO := ℕ → S.WS
  wo := F.wo
  RO := L.RO
  NO x := S.tapes x (L.n x)
  ΩO _ := Unit
  PO := L.PO
  St := List (Option S.StS)
  outO := L.outO
  CO x := S.coins x (L.n x)
  progO := L.progO

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric

/-- **Z1-S: sessions in turn are zero knowledge per outcome**, from each session's. For every session system `S` that
is zero knowledge per outcome within `ε_S` (`ZkSess.ZK`) and layout `L` whose sessions make up its rounds
(`Sessions.Shaped`): under each compiled statement, the composed prover is zero knowledge per outcome against every
verifier of `ko` rounds, on every witness in the composed relation (every session's), at the composed simulator (each
session's simulator given its outcome, in turn) and within the composed error (`Sessions.err`: each session's error at
`V*` after the real prefix, averaged over the earlier sessions' tapes where the prefix shows its outcomes and reaches
the session). This is `ZkRelease.OuterZK` for an outer phase of these sessions. -/
def GenericSessionsZK : Prop :=
  ∀ (H : Msg → Msg) (S : ZkSess) (L : Sessions S)
    (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ),
    S.ZK H εS → L.Shaped →
      ∀ x, ZKOut (fun _ => S.tapes x (L.n x)) (fun _ => Unit) H (fun _ => L.ko x) (fun wo _ => L.PO wo x) (L.RO x)
        (L.outO x) (L.simO x) (L.err H εS x)

/-- **Z1-S-cost: the composed simulator is counted where the session simulator is.** Where the session simulator makes
at most `c_S(x)` calls on every coin, at every session statement and outcome and against every verifier, the composed
simulator makes at most `n(x)·c_S(x)`. -/
def GenericSessionsZKCost : Prop :=
  ∀ (H : Msg → Msg) (S : ZkSess) (L : Sessions S) (cS : Msg → ℕ),
    (∀ x xs o (c : S.CS x) ask, ((S.progS x xs o c).eval H ask).2 ≤ cS x) →
      ∀ x xO o (c : S.coins x (L.n x)) ask, ((L.progO x xO o c).eval H ask).2 ≤ L.n x * cS x

/-- **Z1 at an outer phase of sessions** (from Z1 and Z1-S): where the leaves hide within `δ`, the sessions are zero
knowledge per outcome within `ε_S` and make up the outer phase's rounds, the release of the first part `F` and the
sessions in turn is zero knowledge per outcome at every witness, at its simulator given the released outcome, within
`ZkRelease.err` at the composed error `Sessions.err`: each session's error at `V*` after the real prefix, averaged over
the first part where the server releases and over the earlier sessions' tapes, plus `δ` per leaf of the first part. -/
def GenericRecursiveSessionsZK : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (Lv : Leaves) (S : ZkSess) (L : Sessions S) (F : FirstPart W Lv (ℕ → S.WS))
    (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ) (δ : ℝ),
    Lv.Hides δ → S.ZK H εS → L.Shaped →
      ZKOut (ZkRelease.ofSessions F L).N (ZkRelease.ofSessions F L).Ω H (ZkRelease.ofSessions F L).k
        (ZkRelease.ofSessions F L).prover (fun _ _ => True) (ZkRelease.ofSessions F L).outcome
        (ZkRelease.ofSessions F L).sim ((ZkRelease.ofSessions F L).err H (L.err H εS) δ)

/-- **Z1-cost at an outer phase of sessions** (from Z1-cost and Z1-S-cost): where each frame makes at most `c_F(x)`
calls and the session simulator at most `c_S(x)`, the release's simulator given the outcome makes at most
`n_h(x)·(c_F(x) + 1) + k_O(x)` calls at the refusal and `n_h(x)·(c_F(x) + 1) + n(x)·c_S(x)` at an outer outcome. -/
def GenericRecursiveSessionsZKCost : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (Lv : Leaves) (S : ZkSess) (L : Sessions S) (F : FirstPart W Lv (ℕ → S.WS))
    (cF cS : Msg → ℕ),
    (∀ x pre ls, ((F.frame x pre ls).eval H).2.length ≤ cF x) →
    (∀ x xs o (c : S.CS x) ask, ((S.progS x xs o c).eval H ask).2 ≤ cS x) →
    ZKMal.SimCostLe H ((ZkRelease.ofSessions F L).sim.given none) (fun x => F.nh x * (cF x + 1) + L.ko x) ∧
      ∀ o, ZKMal.SimCostLe H ((ZkRelease.ofSessions F L).sim.given (some o))
        (fun x => F.nh x * (cF x + 1) + L.n x * cS x)

/-- **Z1-L: a zero-knowledge system is a session system at no cost in error.** Where a prover on a tape `N × Ω` is zero
knowledge per outcome within `ε` at a simulator whose coins are its statement's, its session system at any finite sets
`Xs` of session statements (`ZkSess.ofZKOut`) is zero knowledge per outcome within the same `ε`: one tape for each
statement in `Xs x`, of which the session reads its own statement's, is that statement's tape. -/
def GenericSessionsOfZKOut : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (R : Rel W) (k : Msg → ℕ) (N Ω : Type) [Fintype N] [Nonempty N] [Fintype Ω]
    [Nonempty Ω] (P : W → Prover (N × Ω)) {St : Type} (out : Msg → List (Msg × Out) → St) (S : SimOut St)
    (Xs : Msg → Finset Msg) (ε : Msg → W → (Θ : Type) → VStar Θ → Θ → St → ℝ),
    ZKOut (fun _ => N) (fun _ => Ω) H k (fun w _ => P w) R out S ε →
      (ZkSess.ofZKOut R k N Ω P out S Xs).ZK H fun _ => ε

/-- **Z1-L-cost: its session simulator is counted where the system's is.** Where the system's simulator makes at most
`c_S(x)` calls at every statement in `Xs x`, outcome and coin, against every verifier, so does the session simulator
under `x`, at every session statement. -/
def GenericSessionsOfZKOutCost : Prop :=
  ∀ (H : Msg → Msg) {W : Type} (R : Rel W) (k : Msg → ℕ) (N Ω : Type) [Fintype N] [Nonempty N] [Fintype Ω]
    [Nonempty Ω] (P : W → Prover (N × Ω)) {St : Type} (out : Msg → List (Msg × Out) → St) (S : SimOut St)
    (Xs : Msg → Finset Msg) (cS : Msg → ℕ),
    (∀ x, ∀ xs ∈ Xs x, ∀ o (c : S.C xs) ask, ((S.prog xs o c).eval H ask).2 ≤ cS x) →
      ∀ x xs o (c : (ZkSess.ofZKOut R k N Ω P out S Xs).CS x) ask,
        (((ZkSess.ofZKOut R k N Ω P out S Xs).progS x xs o c).eval H ask).2 ≤ cS x

end Flock.Guarantees
