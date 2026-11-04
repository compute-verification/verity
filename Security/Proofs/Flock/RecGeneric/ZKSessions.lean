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
  public outcome (`outS`), and, at each compiled statement `x`, the session's tape `NS x × ΩS x` and its simulator's
  coins `CS x` and program (`progS x`, given the session statement and outcome), the same at every session statement
  under `x`. Its zero knowledge (`ZkSess.ZK`) is `ZKOut` at each `x` within `ε_S`.
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

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- **A zero-knowledge session system at compiled statements**: the session prover at the witness `WS` (`PS`, on the
tape `NS x × ΩS x` under the compiled statement `x`, `kS` rounds at a session statement), its public outcome (`outS`),
and its simulator under `x`: coins `CS x` and a program given the session statement and the outcome (`progS x`). -/
structure ZkSess where
  WS : Type
  RS : Rel WS
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
`NS x × ΩS x`, against every verifier of the session's `kS` rounds, on every witness in `RS`, at `simAt x` and
`εS x`. -/
def ZK (S : ZkSess) (H : Msg → Msg) (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ) : Prop :=
  ∀ x, ZKOut (fun _ => S.NS x) (fun _ => S.ΩS x) H S.kS (fun w _ => S.PS w x) S.RS S.outS (S.simAt x) (εS x)

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

/-- **The composed relation**: every session's witness is in `RS` at its statement. -/
def RO (x : Msg) : Rel (ℕ → S.WS) := fun xO wo => ∀ i < L.n x, S.RS (L.xs x xO i) (wo i)

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

end Flock.Guarantees
