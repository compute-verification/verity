import Definitions.Pouw.RelativeSecurity
import Definitions.Pouw.Basic.QTree

/-!
# Relative security under Fiat–Shamir draws: the vocabulary its statements read (trusted)

Decision 102's seeds: matmul `j`'s draw comes from `SHA-512(tag ‖ s ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j))`,
a hash of commitments the prover has already made, at the epoch's live salt `s` (`Epoch.start`: every window of an
epoch reads one salt). A prover can re-roll a root through its hiding salts and try again, so every try is a fresh
draw. The hash is evaluated inside the prover's own proof (decision 105): the drawn unit's zero-knowledge proof derives
its seed from the roots and position it witnesses, and the verifier never evaluates it. The statements over this
vocabulary (`Models.Pouw.Guarantees.FiatShamir`, `Models.Pouw.Assumptions.FiatShamir`) read SHA-512 as a random
oracle, gap (4)'s `H` (`Pouw.WindowWork.StraightLineExtraction`) at the seed's domain: a uniform function `H : Q → N`
from the seed's inputs to a stage's draws, which the prover reaches only through counted queries (`QTree`, as PoUW's
game does). The named premise `Pouw.Assumptions.SeedExtraction` ties it to the seeds the proofs derive. Every other
coin stays live.

* `FSService`: a service whose drawn units' proofs derive their seeds in-circuit (`seed`), at the inputs its commitment
  names (`pre`), and whose other coins (`Live`: each round's check of a stage and a cell-block) are drawn after the
  commitment; it accepts at the derived seeds (`accepts`), and the analysis reads it at the oracle's (`draws`);
* `FSWithin`: an adversary whose commitment is the result of one oracle program of at most `T` calls, the committed
  inputs counted;
* `StateTerm`, `coll`: an analysis's state term `g` (per adversary, its own coins and the oracle) and its mean, the one
  collision term of the statements (gap (6)'s `κ + g(state)` and `coll`, gap (4)'s `β(H, s)` averaged);
* `ServiceFloorFS`: `ServiceFloor` over such adversaries, over the oracle, the live coins and the adversary's coins, its
  bound a function of the adversary;
* `GainMap`, `LiveFloorUnsteered`: an analysis's gain through each try (relative to the stage's savings, GRIND's `R q`),
  and the live statement at draws whose every committed seed realizes less than the premium `π`, where the draws are
  fixed and the live coins alone bound the floor's event;
* the staged window under these draws: `toPaths`, `stagedCoins` (its realized coins), `StagedFS` and `RunMapFS`.
-/

namespace Pouw.RelativeSecurity

section FiatShamir

/-- **A service under Fiat–Shamir draws** (decision 102), its stage draws in `N`, its stages `J` (on the staged
design, `Fin R × Fin St`) and its live coins `Live`, drawn after the commitment (each round's check). `pre c j` is the
input commitment `c` names for stage `j`'s seed (`tag ‖ s ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j)`). Distinct
stages name distinct inputs because the circuit binds the position through the unit's opening path (decisions
105–106), not because the verifier refuses repeats: it never sees the inputs. `seed c a j` is the seed that the proof
of stage `j`'s drawn unit in the answers `a` derives in-circuit, and `verify c d l a` is the verifier's check of the
answers at the seeds `d` and the live coins `l`. -/
structure FSService (J N Live : Type) where
  Q : Type
  [instQ : Fintype Q]
  [decQ : DecidableEq Q]
  Com : Type
  Ans : Type
  pre : Com → J → Q
  seed : Com → Ans → J → N
  verify : Com → (J → N) → Live → Ans → Bool

attribute [instance] FSService.instQ FSService.decQ

variable {J N Live : Type}

/-- **The service accepts** the commitment `c` and the answers `a` at the live coins `l`: its proofs pass at the seeds
they derive. The verifier never evaluates the oracle. -/
def FSService.accepts (S : FSService J N Live) (c : S.Com) (l : Live) (a : S.Ans) : Bool :=
  S.verify c (S.seed c a) l a

/-- The draws commitment `c` gets from the oracle `H`: stage `j`'s is `H (pre c j)`. -/
def FSService.draws (S : FSService J N Live) (H : S.Q → N) (c : S.Com) : J → N := fun j => H (S.pre c j)

/-- An adversary against `S` in the machine `Mc`: from the oracle, the live coins and its own coins `Ω`, a commitment
and its answers. -/
abbrev FSAdv (Mc : Machine) (S : FSService J N Live) (Ω : Type) : Type :=
  Mc.Alg (((S.Q → N) × Live) × Ω) (S.Com × S.Ans)

/-- **Within `T` oracle calls, committing first.** At each value `r` of its own coins the adversary's commitment is
the result of one adaptive oracle program (`QTree`), whatever the live coins, so it commits before they are drawn.
The program makes at most `T` calls against every oracle, counting the `|J|` inputs its commitment names, at which its
proofs are to derive their seeds. Every query counts, whenever it is made and by whichever of the prover's devices,
from the salt's release to the deadline: a try at a root it then drops is a query, and so is a query made for an
earlier window of the epoch, which reads the same oracle. Its answers may read the oracle anywhere, since the draws
are fixed by the commitment. -/
def FSWithin [Fintype J] (Mc : Machine) (S : FSService J N Live) {Ω : Type} (M : FSAdv Mc S Ω) (T : ℕ) : Prop :=
  ∀ r : Ω, ∃ t : QTree S.Q N S.Com, ∀ (H : S.Q → N) (l : Live),
    (Mc.eval M ((H, l), r)).1 = t.eval H ∧ t.calls H + Fintype.card J ≤ T

/-- **An analysis's state term** (gap (6)'s `g`, gap (4)'s `β(H, s)`): per adversary, value of its own coins and
oracle `H`, what the analysis charges to the state the run holds, such as a collision among its hash inputs or a proof
layer's failure at that state. No constant bounds a fixed-oracle event for every adversary: one that holds a collision
of SHA-512 in its commitment tree equivocates an opening at every live coin. So the statements' fixed-oracle premises
bound their events by a constant plus this term, and their conclusion carries its mean (`coll`). -/
abbrev StateTerm (Mc : Machine) (S : FSService J N Live) : Type 1 :=
  ∀ (Ω : Type) [Fintype Ω], FSAdv Mc S Ω → Ω → (S.Q → N) → ℝ

/-- **The collision term** of an adversary `M`: its state term's mean over the oracle and its own coins. -/
noncomputable def coll [Fintype N] {Mc : Machine} {S : FSService J N Live} (g : StateTerm Mc S) (Ω : Type)
    [Fintype Ω] (M : FSAdv Mc S Ω) : ℝ :=
  (∑ x : (S.Q → N) × Ω, g Ω M x.2 x.1) / (Fintype.card ((S.Q → N) × Ω) : ℝ)

/-- **The service's adversary floor under Fiat–Shamir draws**: every adversary `M` within `T` oracle calls has its
accepted runs charged (`max(|M|, cost)`) below `F` with probability at most `b M`, over a uniform oracle (the
random-oracle model), the live coins and its own coins. `ServiceFloor` with the stage draws taken from the oracle and
a bound that depends on the adversary. -/
def ServiceFloorFS [Fintype J] [Fintype N] [Fintype Live] (Mc : Machine) (S : FSService J N Live) (T : ℕ) (F : ℝ)
    (b : ∀ (Ω : Type) [Fintype Ω], FSAdv Mc S Ω → ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : FSAdv Mc S Ω), FSWithin Mc S M T →
    prob (fun ω : ((S.Q → N) × Live) × Ω =>
      S.accepts (Mc.eval M ω).1 ω.1.2 (Mc.eval M ω).2 = true ∧ Mc.charge M ω < F) ≤ b Ω M

/-- **An analysis's gain map** (GRIND's `R q`, note:compute-accounting/20261018T0700Z-draft-grind): per adversary,
value of its own coins, oracle `H` and input `x`, what its run realizes through the draw at `x`, in work units. That
is the work its units seeded at `x` skip and still pass with (lucky), less `κ` times the work they skip and fail with
(unlucky), at the analysis's exchange rate `κ`, and nothing at an input no committed stage's seed reads. It is
relative to the stage's savings (RT4's change 2): a stage whose cheat saves no work gains nothing, however correlated
its errors. It may read the whole oracle, so the adversary's run before the try and after it; what it costs is
`SteerBound`'s, which redraws the oracle at `x` alone. -/
abbrev GainMap (Mc : Machine) (S : FSService J N Live) : Type 1 :=
  ∀ (Ω : Type) [Fintype Ω], FSAdv Mc S Ω → Ω → (S.Q → N) → S.Q → ℝ

/-- **The live statement at unsteered draws, the premium priced into the floor**: for every adversary within `T`
oracle calls, every oracle `H` and every value `r` of its own coins, the live coins alone make its run accepted at the
oracle's draws, charged below `F`, with every committed input's gain below the premium `π`, with probability at most
`κ + g(M, r, H)`. The oracle is fixed here, and with it the commitment and its draws, so only the live coins (each
round's check) are random: this is the live analysis at draws chosen by the adversary among those whose every seed
realizes less than `π`.

`F` is where the premium goes. At such draws a run's savings are at most `(1 + κ)·U` plus `π` a seed group, `U` its
unlucky savings, which the checks catch at the zero-luck rate, so the staged instance's `F` is the live floor less
the premium: `(1 − ε_FS)·W` with `ε_FS = ε_live + (κ − 3)/4·ε_luck + π/(units a group)` (GRIND's composition; the
live analysis charges `κ = 3`, its luck `3/4`). -/
def LiveFloorUnsteered [Fintype J] [Fintype Live] (Mc : Machine) (S : FSService J N Live) (T : ℕ)
    (gain : GainMap Mc S) (π F κ : ℝ) (g : StateTerm Mc S) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : FSAdv Mc S Ω), FSWithin Mc S M T → ∀ (H : S.Q → N) (r : Ω),
    prob (fun l : Live =>
      (∀ j, gain Ω M r H (S.pre (Mc.eval M ((H, l), r)).1 j) < π) ∧
        S.verify (Mc.eval M ((H, l), r)).1 (S.draws H (Mc.eval M ((H, l), r)).1) l (Mc.eval M ((H, l), r)).2 = true ∧
          Mc.charge M ((H, l), r) < F) ≤ κ + g Ω M r H

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
