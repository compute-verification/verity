module

public import Definitions.Claims.RecursionGeneric.Plan
public import Definitions.Claims.RecursionGeneric.Server

/-!
# The inner protocol, abstract

`Proto Coin W` is what the recursion asks of an inner proof system: a public-coin verifier over byte strings with its
relation, readable scope and repetition extractor (`Claims.Recursion.Inner`), the bytes that answer a round (`ans`, its
coin's), and its knowledge error against provers that may abort after its rounds, stated with run finders
(`Proto.KSAbortLin`): `κ` plus, per term, a weight times a run finder's success (`Plan.lean`) plus a tail.

`Proto.WF` is what the recursion's proof reads of the terms: the answers decode to their coins, weights and tails are
nonnegative, and each term's finder meets a bad run at most `d` times as often as one run is bad (`RunFinder.ReadsLe`).

The inner extractor is the repetition extractor `E^rep`: it reruns the whole protocol on fresh coins, waits for accepted
reruns, and assembles a witness from their opened views. So the recursion covers inner protocols whose knowledge
soundness `E^rep` shows; an inner extractor that forks at a later round (tree extraction) or extracts straight-line
is not a `Proto`.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- **One term of an inner protocol's knowledge error**: at each loop bound `m`, a run finder `F m`; its weight
`w x`, its tail `tail x m` (what truncating the finder's loops at `m` costs), and `d x`, how many bad runs its plan
meets in expectation per bad run (`RunFinder.ReadsLe`). -/
structure Term (Coin : Type) where
  F : ℕ → RunFinder Coin
  w : Msg → ℝ
  tail : Msg → ℕ → ℝ
  d : Msg → ℕ

/-- **An inner protocol with a repetition extractor**: `Claims.Recursion.Inner` (its verifier, relation, scope,
repetition extractor's pieces, `κ` and `γ`), the bytes that answer a round before the last (`ans`, its coin's), and the
`nT` terms of its knowledge error. `γ` is unread: the terms carry the collision part of the error. -/
structure Proto (Coin W : Type) extends Claims.Recursion.Inner Coin W where
  ans : Coin → Msg
  nT : ℕ
  terms : Fin nT → Term Coin

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **What the recursion's proof reads of an inner protocol's terms**: its answers decode to their coins, `κ`, every
weight and every tail is nonnegative, and every term's finder, at every loop bound, meets a bad run at most `d` times
as often as one run is bad. -/
def Proto.WF (I : Proto Coin W) : Prop :=
  (∃ dec : Msg → Coin, ∀ c, dec (I.ans c) = c) ∧ (∀ x, 0 ≤ I.κ x) ∧ (∀ i x, 0 ≤ (I.terms i).w x) ∧
    (∀ i x m, 0 ≤ (I.terms i).tail x m) ∧ ∀ i m, ((I.terms i).F m).ReadsLe (I.terms i).d

/-- **The inner protocol's knowledge soundness against provers that may abort, with its error in run finders**: for
every statement, prover (any finite tape), way to abort after the inner rounds (`keep`, on any finite fresh coins per
run) and loop bounds `m`,
`Pr[run = acc ∧ ¬R(x, E^rep(x))] ≤ κ(x) + Σ_i w_i(x)·(Pr[F_i(m_i) finds a collision among the runs' calls] + τ_i(x, m_i))`,
`E^rep` and every finder seeing each run as it aborts (`runModel`). At Flock it is
`Claims.Recursion.KSFlockLin` (three terms: `B`, the caps' `FC` and level 0's `F0`). -/
def Proto.KSAbortLin (H : Msg → Msg) (I : Proto Coin W) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A]
    (keep : Ω → (Fin (I.V.k x) → Coin) → A → Bool) (m : Fin I.nT → ℕ),
    I.toInner.PrBadAbort H x P keep ≤
      I.κ x + ∑ i, (I.terms i).w x *
        (((I.terms i).F (m i)).PrAt H (runModel H I.V P x keep) x + (I.terms i).tail x (m i))

/-- **The inner verifier's last step reads no last coin**: at its last round, two coin lists that agree before the last
coin give one accept bit with the same calls. `Open(V, S)` reads `V`'s last step only through that bit (`agrees`), so
its last step then reads no last coin either, decision and calls. The composite's inner prover sends the plurality of
the reruns' last messages, decided at its own run's last coin, so the recursion reads this with `Comp.ForksBind` at the
last round. It holds at a public-coin verifier whose decision reads only the coins that answered earlier messages, and
at the element-level Flock it is `Claims.Recursion.Bound.LastStepFree`, proved at the deployed plain verifier and
reader. -/
def Proto.LastCoinFree (H : Msg → Msg) (I : Proto Coin W) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs cs' : List Coin), as.length = I.V.k x → cs.length = as.length →
    cs'.length = as.length → cs.dropLast = cs'.dropLast →
    (bindOC (I.V.next x as cs) fun b => OC.ret (decide (b = .acc))).eval H =
      (bindOC (I.V.next x as cs') fun b => OC.ret (decide (b = .acc))).eval H

end Claims.RecGeneric
