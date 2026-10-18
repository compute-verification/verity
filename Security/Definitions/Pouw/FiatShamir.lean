import Definitions.Pouw.RelativeSecurity
import Definitions.Pouw.Basic.QTree

/-!
# Relative security under Fiat–Shamir draws: the vocabulary its statements read (trusted)

Decision 102's seeds: matmul `j`'s draw comes from `SHA-512(tag ‖ s ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j))`,
a hash of commitments the prover has already made, at the window's live salt `s`. A prover can re-roll a root through
its hiding salts and try again, so every try is a fresh draw. The statements over this vocabulary
(`Models.Pouw.Guarantees.FiatShamir`, `Models.Pouw.Assumptions.FiatShamir`) read SHA-512 at the window's salt as a
random oracle: a uniform function `H : Q → N` from the seed's inputs to a stage's draws, which the prover reaches only
through counted queries (`QTree`, as PoUW's game does). Every other coin stays live.

* `FSService`: a service whose stage draws are the oracle's answers at the inputs its commitment names (`pre`), and
  whose other coins (`Live`: each round's check of a stage and a cell-block) are drawn after the commitment;
* `FSWithin`: an adversary whose commitment is the result of one oracle program of at most `T` calls, the committed
  inputs counted;
* `ServiceFloorFS`: `ServiceFloor` over such adversaries, over the oracle, the live coins and the adversary's coins;
* `SteerMap`, `LiveFloorUnsteered`: an analysis's steer events (per adversary, own coins, oracle and input), and the
  live statement at draws outside them, where the draws are fixed and the live coins alone bound the floor's event;
* the staged window under these draws: `toPaths`, `stagedCoins` (its realized coins), `StagedFS` and `RunMapFS`.
-/

namespace Pouw.RelativeSecurity

section FiatShamir

/-- **A service under Fiat–Shamir draws** (decision 102), its stage draws in `N`, its stages `J` (on the staged
design, `Fin R × Fin St`) and its live coins `Live`, drawn after the commitment (each round's check). Stage `j`'s draw
is the oracle's answer at `pre c j`, the input commitment `c` names for it
(`tag ‖ s ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j)`); the verifier reads the oracle only there. -/
structure FSService (J N Live : Type) where
  Q : Type
  [instQ : Fintype Q]
  [decQ : DecidableEq Q]
  Com : Type
  Ans : Type
  pre : Com → J → Q
  verify : Com → (J → N) → Live → Ans → Bool

attribute [instance] FSService.instQ FSService.decQ

variable {J N Live : Type}

/-- The draws commitment `c` gets from the oracle `H`: stage `j`'s is `H (pre c j)`. -/
def FSService.draws (S : FSService J N Live) (H : S.Q → N) (c : S.Com) : J → N := fun j => H (S.pre c j)

/-- An adversary against `S` in the machine `Mc`: from the oracle, the live coins and its own coins `Ω`, a commitment
and its answers. -/
abbrev FSAdv (Mc : Machine) (S : FSService J N Live) (Ω : Type) : Type :=
  Mc.Alg (((S.Q → N) × Live) × Ω) (S.Com × S.Ans)

/-- **Within `T` oracle calls, committing first.** At each value `r` of its own coins the adversary's commitment is
the result of one adaptive oracle program (`QTree`), whatever the live coins, so it commits before they are drawn.
The program makes at most `T` calls against every oracle, counting the `|J|` inputs its commitment names, which the
verifier evaluates. Every query counts, whenever it is made: a try at a root it then drops is a query. Its answers may
read the oracle anywhere, since the draws are fixed by the commitment. -/
def FSWithin [Fintype J] (Mc : Machine) (S : FSService J N Live) {Ω : Type} (M : FSAdv Mc S Ω) (T : ℕ) : Prop :=
  ∀ r : Ω, ∃ t : QTree S.Q N S.Com, ∀ (H : S.Q → N) (l : Live),
    (Mc.eval M ((H, l), r)).1 = t.eval H ∧ t.calls H + Fintype.card J ≤ T

/-- **The service's adversary floor under Fiat–Shamir draws**: every adversary within `T` oracle calls has its
accepted runs charged (`max(|M|, cost)`) below `F` with probability at most `κ`, over a uniform oracle (the
random-oracle model), the live coins and its own coins. `ServiceFloor` with the stage draws taken from the oracle. -/
def ServiceFloorFS [Fintype J] [Fintype N] [Fintype Live] (Mc : Machine) (S : FSService J N Live) (T : ℕ)
    (F κ : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : FSAdv Mc S Ω), FSWithin Mc S M T →
    prob (fun ω : ((S.Q → N) × Live) × Ω =>
      S.verify (Mc.eval M ω).1 (S.draws ω.1.1 (Mc.eval M ω).1) ω.1.2 (Mc.eval M ω).2 = true ∧
        Mc.charge M ω < F) ≤ κ

/-- **An analysis's steer events**: per adversary, value of its own coins, oracle `H` and input `x`, whether the draw
`H x` steers the adversary's run. The event may read the whole oracle, so it may read the adversary's run up to the
try and after it; what it costs is `SteerBound`'s, which redraws the oracle at `x` alone. -/
abbrev SteerMap (Mc : Machine) (S : FSService J N Live) : Type 1 :=
  ∀ (Ω : Type) [Fintype Ω], FSAdv Mc S Ω → Ω → (S.Q → N) → S.Q → Prop

/-- **The live statement at unsteered draws**: for every adversary within `T` oracle calls, every oracle `H` and every
value `r` of its own coins, the live coins alone make its run accepted, charged below `F`, with no committed draw
steered, with probability at most `ε`. The oracle is fixed here, and with it the commitment and its draws, so only
the live coins (each round's check) are random: this is the live analysis with the draws chosen by the adversary
among the unsteered ones. -/
def LiveFloorUnsteered [Fintype J] [Fintype Live] (Mc : Machine) (S : FSService J N Live) (T : ℕ)
    (Steer : SteerMap Mc S) (F ε : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : FSAdv Mc S Ω), FSWithin Mc S M T → ∀ (H : S.Q → N) (r : Ω),
    prob (fun l : Live =>
      (∀ j, ¬ Steer Ω M r H (S.pre (Mc.eval M ((H, l), r)).1 j)) ∧
        S.verify (Mc.eval M ((H, l), r)).1 (S.draws H (Mc.eval M ((H, l), r)).1) l (Mc.eval M ((H, l), r)).2 = true ∧
          Mc.charge M ((H, l), r) < F) ≤ ε

end FiatShamir

section StagedFiatShamir

/-- `n` coins as a path of `n` rounds, coin `i` the `i`-th round's. -/
def toPaths {Ω : Type} : (n : ℕ) → (Fin n → Ω) → Paths Ω n
  | 0, _ => ()
  | n + 1, f => (f 0, toPaths n fun i => f i.succ)

variable {R St n L np B n' : ℕ}

/-- **The staged window's realized coins under Fiat–Shamir draws**: round `w`'s stage draws are the oracle's
(`d (w, s)`), and its check of a stage and a cell-block (`l w`) is live. -/
def stagedCoins (d : Fin R × Fin St → Finset (DBits n L np B n')) (l : Fin R → Fin St × DCells n n' B) :
    Paths (DStage St n L np B n') R :=
  toPaths R fun w => (toPaths St fun s => d (w, s), l w)

/-- **The staged service under Fiat–Shamir draws**: `R` rounds of `St` stages on the re-encoded design, stage
`(w, s)`'s draw (its design bits) from the oracle at the input its commitment names, each round's check live. -/
abbrev StagedFS (R St n L np B n' : ℕ) : Type 1 :=
  FSService (Fin R × Fin St) (Finset (DBits n L np B n')) (Fin R → Fin St × DCells n n' B)

/-- **The analysis's map from adversaries to their window runs under Fiat–Shamir draws**: for each adversary `M`
against the staged service, each value `r` of its own coins and each oracle `H`, its run, which the statements read
at its realized coins `stagedCoins (S.draws H c) l`. -/
abbrev RunMapFS (Mc : Machine) (S : StagedFS R St n L np B n') (Ar : Type) [CommRing Ar] : Type 1 :=
  ∀ (Ω : Type) [Fintype Ω], FSAdv Mc S Ω → Ω → (S.Q → Finset (DBits n L np B n')) → StagedRun R St n L np B n' Ar

end StagedFiatShamir

end Pouw.RelativeSecurity
