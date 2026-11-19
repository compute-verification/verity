module

public import Definitions.Pouw.Game.Deadline

/-!
# The deadline's named assumption (trusted; a `Prop` taken as a hypothesis, never an `axiom`)

The verifier's acceptance rule at the deadline is a protocol definition (`Pouw.AcceptsBy`): it reads two entries the
auditor's device signs, the salt's (`a`) and the docket's (`b`), and checks `0 ≤ ra.elapsed rb ≤ Δ` on that device's
clock. How that clock runs against real time is Core's premise `clock-within-drift` (`Assumptions.ClockWithin`), which
places the docket's entry within the real window `realWindow` after the salt's. What links that real window to the cost
model is this assumption about the physical run, the honest clock. The salt's unpredictability before its release is
not restated: preprocessing never reads the salt, by `Adversary`'s construction, and the program the assumption names
runs after the adversary's own preprocessing.

It is stated once, over what the deadline reads of a game's adversaries (`Pouw.Adversaries`), and both PoUW games read
it: the integer game at `Pouw.CostModel.adversaries` and Pearl-C's at `Pouw.PearlC.CostModel.adversaries`.
-/

@[expose] public section

namespace Pouw.Assumptions

variable {Q R S : Type}

/-- **`honest-clock`: the work behind a docket the device signed by a real deadline was done by then**, for an
adversary `A` of a game's adversaries `G` (each with its preprocessing, trace, online cost, oracle calls and SHA-512
inputs on a fixed workload), the protocol's encoding of a docket `dk`, the device's entries `st` (the salt's `a` and the
docket's `b`, signed at the real times `st.real`), the run's spending `run`, and a real window `d` from the salt's entry.

The cost that determines a trace whose digest the device's entry `b` signs by the real time `st.ts s + d` is spent by
then, and after the salt's entry, since only preprocessing runs before it and preprocessing never reads the salt. In
the cost model: an online program, run after `A`'s own preprocessing on some activation input, outputs `A`'s trace on
every oracle and salt on which `b` signs that trace's digest by `st.ts s + d`, hashing there only inputs `A`'s run
hashes; on every oracle and salt it costs at most what the run spends in `(st.ts s, st.ts s + d]` and makes at most the
oracle calls of `A`'s online phase. On a trace whose digest `b` doesn't sign by then, its output is unconstrained.

That program is the prover's computation stopped at `st.ts s + d`, which hashes a part of what the whole computation
hashes. It reads the real times of the device's own two entries only: when `b` is signed against when the verifier
reads its clock is not its business, which is `clock-within-drift`'s and the verifier's check of `ra.elapsed rb`; and
it says nothing of when any entry other than `a` and `b` was made, beyond what the cost model charges. It fails if the
cost model charges the computation that produced the signed trace less than the run spent on it by `b`'s signing. -/
def HonestClock {Adv Out : Type} {k : String} (G : Adversaries Q R S Adv Out) (A : Adv) (dk : Docket S Out k)
    (st : Stamps Q R S k) (run : TimedRun Q R S) (d : ℝ) : Prop :=
  ∃ A' : Adv, G.pre A' = G.pre A ∧ ∀ (H : Q → R) (s : S),
    (dk.digest (G.trace A H s) ∈ (st.docket H s).shown → st.tr H s ≤ st.ts s + d →
      G.trace A' H s = G.trace A H s ∧ ∀ m ∈ G.hashed A' H s, m ∈ G.hashed A H s) ∧
    G.cost A' H s ≤ run.spentIn H s (st.ts s) (st.ts s + d) ∧
    G.calls A' H s ≤ G.calls A H s

end Pouw.Assumptions
