import Mathlib
import Definitions.Claims

/-!
# Relative security for matmul PoUW: the vocabulary its guarantees read (trusted)

The definitions `Definitions.Pouw.Guarantees.RelativeSecurity` and `Definitions.Pouw.Assumptions.RelativeSecurity` read, moved
verbatim from `Proofs.Pouw.RelativeSecurity` (same names, namespace and section variables), so the lock's `reads`
follow them into the spec:

* uniform probability (`prob`), `R` rounds of fresh coins (`Paths`), a prover's play (`RoundGame`, its acceptance
  and deficiency) and a round's stages (`StageGame`);
* the machine model (`AlgModel`, a run's `charge`), the service (`Service`) and its adversary floor (`ServiceFloor`;
  adversary by adversary, `ServiceFloorPerProver`, at an adversary's error `breakBound`, read through the analysis's
  collision finders, `Finders` and `finderTerm`, and `ecr/sha-512`'s `Claims.ecrBound` at budgets computed from the
  adversary's time);
* Theorem K's runs (`BitAff`, `bitInd`, `Slp`, `SlpG`), the re-encoded design's prefix words (`DBits`, `dPrefix`,
  `wRe`, `DCells`), affine products (`AffProd`), degree-2 errors (`CubeQuad`, `pairDiff`) and the spot check's
  deficiency (`spotDef`);
* the staged spot check (`StagePlay`, `StagedPlay`, `DStage`) and an adversary's window run on it (`StagedRun`);
* the spot check over a round's overlapping draws: its bit points (`drawInd`, `mergeAt`), K∥'s products (`AffMulAt`),
  the per-draw deficiency (`ovDef`), the play (`OvPlay`, its game and errors), the round's bits, cell-blocks, draws
  and coins (`OBits`, `OCells`, `ODraws`, `OSpot`), and an adversary's window run on it (`OverlapRun`);
* the parity luck's scope: one-sided products of any form (`LeftOnly`, `RightOnly`, `OneSided`; per draw,
  `OneSidedAt`), `Z/2³²` read mod 2 (`toF2`), a cell-block's frame (`dFrame`), products two-sided at a frame
  (`XBlind`, `YBlind`, `TwoSidedAt`) and charging purity (`SlpG.ChargePure`; across draws, `SlpG.ChargePureAt`).
-/

namespace Pouw.RelativeSecurity

open Finset

section Probability

/-- The probability of an event under the uniform distribution on a finite type. -/
noncomputable def prob {Ω : Type*} [Fintype Ω] (p : Ω → Prop) : ℝ := by
  classical exact ((univ.filter p).card : ℝ) / (Fintype.card Ω : ℝ)

end Probability

section Rounds

/-- The coins of `n` rounds, each fresh and uniform on `Ω`. -/
@[reducible] def Paths (Ω : Type) : ℕ → Type
  | 0 => Unit
  | n + 1 => Ω × Paths Ω n

instance Paths.instFintype (Ω : Type) [Fintype Ω] : ∀ n, Fintype (Paths Ω n)
  | 0 => inferInstanceAs (Fintype Unit)
  | n + 1 => @instFintypeProd Ω (Paths Ω n) _ (Paths.instFintype Ω n)

/-- A prover's play over `n` rounds of coins on `Ω`: per round, its deficiency `d` and its check `pass` as
functions of the round's coins, then the rest of the play, which may depend on them. -/
inductive RoundGame (Ω : Type) : ℕ → Type
  | done : RoundGame Ω 0
  | round {n : ℕ} (d : Ω → ℝ) (pass : Ω → Prop) (next : Ω → RoundGame Ω n) : RoundGame Ω (n + 1)

namespace RoundGame

variable {Ω : Type}

/-- The window accepts: every round's check passes. -/
def accepts : {n : ℕ} → RoundGame Ω n → Paths Ω n → Prop
  | _, .done, _ => True
  | _, .round _ pass next, x => pass x.1 ∧ (next x.1).accepts x.2

/-- The hidden deficiency `Σ_w d_w` along the realized coins. -/
def deficiency : {n : ℕ} → RoundGame Ω n → Paths Ω n → ℝ
  | _, .done, _ => 0
  | _, .round d _ next, x => d x.1 + (next x.1).deficiency x.2

end RoundGame

end Rounds

section Stages

/-- **A round's stages**: per stage, its cheat step `a` and the wrong share `u` its draw adds, then the rest of the
round, which may depend on the draw. -/
inductive StageGame (Ω : Type) : ℕ → Type
  | done : StageGame Ω 0
  | stage {m : ℕ} (a : ℝ) (u : Ω → ℝ) (next : Ω → StageGame Ω m) : StageGame Ω (m + 1)

namespace StageGame

variable {Ω : Type}

/-- The round's deficiency `D = Σ_s a_s` along its draws. -/
def defi : {m : ℕ} → StageGame Ω m → Paths Ω m → ℝ
  | _, .done, _ => 0
  | _, .stage a _ next, x => a + (next x.1).defi x.2

end StageGame

end Stages

section AlgModel

/-- A machine model: algorithms from inputs `I` to outputs `O` (coins are part of the input), what they compute, and
their cost. The axioms are the properties of `time` the reduction uses. -/
structure AlgModel where
  /-- The algorithms from `I` to `O`. -/
  Alg : Type → Type → Type
  /-- What an algorithm computes. -/
  eval : ∀ {I O : Type}, Alg I O → I → O
  /-- Its cost (an abstract `time`: priced work, worst case over inputs and coins). -/
  time : ∀ {I O : Type}, Alg I O → ℝ
  /-- One run's cost (RS-DEF §1.2's `cost_P(x; σ, ρ)`), which `time` bounds. -/
  cost : ∀ {I O : Type}, Alg I O → I → ℝ
  /-- The program's size `|M|` (RS-DEF's T4), charged with every run (`charge`, RS-RT's V1). -/
  size : ∀ {I O : Type}, Alg I O → ℝ
  /-- Run `f`, then `g` on its output. -/
  seq : ∀ {I O P : Type}, Alg I O → Alg O P → Alg I P
  /-- Run `f` and `g` on the same input. -/
  pair : ∀ {I O P : Type}, Alg I O → Alg I P → Alg I (O × P)
  /-- The projections (routing). -/
  fst : ∀ {I J : Type}, Alg (I × J) I
  snd : ∀ {I J : Type}, Alg (I × J) J
  eval_seq : ∀ {I O P : Type} (f : Alg I O) (g : Alg O P) (x : I), eval (seq f g) x = eval g (eval f x)
  eval_pair : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P) (x : I), eval (pair f g) x = (eval f x, eval g x)
  eval_fst : ∀ {I J : Type} (x : I × J), eval (fst (I := I) (J := J)) x = x.1
  eval_snd : ∀ {I J : Type} (x : I × J), eval (snd (I := I) (J := J)) x = x.2
  /-- Costs are nonnegative. -/
  time_nonneg : ∀ {I O : Type} (f : Alg I O), 0 ≤ time f
  /-- `time` is the worst case: every run costs at most `time`. -/
  cost_le_time : ∀ {I O : Type} (f : Alg I O) (x : I), cost f x ≤ time f
  /-- T4: a program's time is at least its size. -/
  size_le_time : ∀ {I O : Type} (f : Alg I O), size f ≤ time f
  size_nonneg : ∀ {I O : Type} (f : Alg I O), 0 ≤ size f
  /-- Subadditivity of sequential composition. -/
  time_seq_le : ∀ {I O P : Type} (f : Alg I O) (g : Alg O P), time (seq f g) ≤ time f + time g
  /-- Monotonicity: running `f` first costs at least `f`. (Not `g`: `g`'s worst case may lie outside `f`'s range.) -/
  le_time_seq : ∀ {I O P : Type} (f : Alg I O) (g : Alg O P), time f ≤ time (seq f g)
  /-- Subadditivity of side-by-side composition. -/
  time_pair_le : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P), time (pair f g) ≤ time f + time g
  /-- Monotonicity: running `f` beside `g` costs at least each. -/
  le_time_pair_left : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P), time f ≤ time (pair f g)
  le_time_pair_right : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P), time g ≤ time (pair f g)
  /-- Routing is free. -/
  time_fst : ∀ {I J : Type}, time (fst (I := I) (J := J)) = 0
  time_snd : ∀ {I J : Type}, time (snd (I := I) (J := J)) = 0

namespace AlgModel

variable (Mc : AlgModel)

/-- **A run's charge** (RS-DEF v2's `max(|M|, cost_M(σ, ρ))`, RS-RT's V1): the program's size and the run's cost.
Charging the run alone lets a coin-indexed table of accepted answers pass at `O(λ)` per run. -/
def charge {I O : Type} (f : Mc.Alg I O) (x : I) : ℝ := max (Mc.size f) (Mc.cost f x)

end AlgModel

end AlgModel

section Service

/-- The proof service's layer of a protocol (AGENTS.md's one proof service): live coins, messages and the verifier.
A protocol is a service with a problem (`correct`) and an honest Solve. -/
structure Service where
  Seed : Type
  [instSeed : Fintype Seed]
  lam : ℕ
  card_seed : Fintype.card Seed = 2 ^ lam
  Com : Type
  Ans : Type
  Coins : Type
  [instCoins : Fintype Coins]
  verify : Com → Seed → Ans → Bool

attribute [instance] Service.instSeed Service.instCoins

/-- **The service's adversary floor** in a machine: every adversary that commits first has its accepted runs charged
(`max(|M|, cost)`) below `F` with probability at most `κ`. It reads only the service, so it is `AdvFloor` of every protocol over the service
(`MatMachine.editFloor_iff`). -/
def ServiceFloor (Mc : AlgModel) (S : Service) (F κ : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (S.Seed × Ω) (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) →
    prob (fun ω : S.Seed × Ω => S.verify (Mc.eval M ω).1 ω.1 (Mc.eval M ω).2 = true ∧ Mc.charge M ω < F) ≤ κ

/-- **The service's adversary floor, adversary by adversary**: every adversary `M` that commits first has its accepted
runs charged below `F` with probability at most `κ Ω M`, a bound of its own, `⊤` where there is none. `ServiceFloor` is
one bound for every adversary, which a service on SHA-512 commitments meets only at 1: an adversary that hard-codes a
collision opens a leaf two ways. -/
def ServiceFloorPerProver (Mc : AlgModel) (S : Service) (F : ℝ)
    (κ : ∀ (Ω : Type) [Fintype Ω], Mc.Alg (S.Seed × Ω) (S.Com × S.Ans) → ENNReal) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (S.Seed × Ω) (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) →
    ENNReal.ofReal (prob fun ω : S.Seed × Ω => S.verify (Mc.eval M ω).1 ω.1 (Mc.eval M ω).2 = true ∧
      Mc.charge M ω < F) ≤ κ Ω M

/-- **The analysis's collision finders, adversary by adversary**: for each adversary `M` in `Mc` against `S`, each
value `r` of its own coins and each round `w`, the finder the analysis builds from `M` at `r` to turn round `w`'s break
into a collision of the commitments' hash (`Claims.Finder`: its coins, its output, and its SHA-512 calls on each coin).
Its cost counts everything it computes, the runs of `M` it plays and whatever else it reads off them, such as the error
table the analysis fixes from a commitment. Like `RunMap`, it is the analysis's; a service premise per prover takes one
for the binding breaks and one for the zero-knowledge breaks. The cost is declared, so the bounds never read it as a
number: they read a budget computed from `M`'s time (`breakBound`), and a finder whose expected cost exceeds it gives
no bound. -/
abbrev Finders (Mc : AlgModel) (S : Service) (R : ℕ) : Type 1 :=
  ∀ (Ω' : Type) [Fintype Ω'], Mc.Alg (S.Seed × Ω') (S.Com × S.Ans) → Ω' → Fin R → Claims.Finder

/-- **An adversary's finder term at one value of its coins, at the budget `q`**: the `ecr/sha-512` bounds of its
finders for every round (`Claims.ecrBound` at the hash `sha` and the budget), summed: `R · q / 2^256` where every
finder meets the assumption within `q` calls in expectation, `⊤` otherwise. -/
noncomputable def finderTerm {Mc : AlgModel} {S : Service} {R : ℕ} (sha : Claims.Msg → Claims.Msg)
    (fd : Finders Mc S R) (q : ℝ) (Ω : Type) [Fintype Ω] (M : Mc.Alg (S.Seed × Ω) (S.Com × S.Ans)) (r : Ω) :
    ENNReal :=
  ∑ w, Claims.ecrBound sha (fd Ω M r w) q

/-- **An adversary's error in the floors per prover**: at each value `r` of its coins, the luck term `ε`, the
constants `δb` and `δz` of each of the `R` rounds' binding and zero-knowledge breaks, and its binding and
zero-knowledge finder terms at `r` (`finderTerm`) at the weights `βb` and `βz`, capped at 1, a probability; then
averaged over `r`. The cap keeps one value of the coins whose finder fails `ecr/sha-512` (a prover that merely tries
its coins as a collision) from sending the whole bound to `⊤`: that value adds its share of the coins, `1/|Ω|`.

Each path's finders are read at a budget computed from the adversary: `cb · time M + db` SHA-512 calls for the
binding finders and `cz · time M + dz` for the zero-knowledge ones, at constants the analysis states (its finders'
reruns of `M` per unit of `M`'s time, and the calls they make besides). `M`'s time is the machine's worst case
(`AlgModel.time`), which `M` can't set low, and the finders' declared costs only have to fit under the budgets. Where
every finder fits and meets `ecr/sha-512`, the error is
`ε + R (δb + δz) + R (βb (cb · time M + db) + βz (cz · time M + dz)) / 2^256`, capped at 1. -/
noncomputable def breakBound {Mc : AlgModel} {S : Service} {R : ℕ} (sha : Claims.Msg → Claims.Msg)
    (fb fz : Finders Mc S R) (cb db cz dz : ℝ) (ε δb βb δz βz : ℝ) (Ω : Type) [Fintype Ω]
    (M : Mc.Alg (S.Seed × Ω) (S.Com × S.Ans)) : ENNReal :=
  ∑ r : Ω, (Fintype.card Ω : ENNReal)⁻¹ *
    min 1 (ENNReal.ofReal (ε + R * (δb + δz)) +
      ENNReal.ofReal βb * finderTerm sha fb (cb * Mc.time M + db) Ω M r +
      ENNReal.ofReal βz * finderTerm sha fz (cz * Mc.time M + dz) Ω M r)

end Service

section TheoremK

/-- An operand affine in the bits `z : V → R`: `c + Σ_v α_v z_v`. -/
structure BitAff (V R : Type) where
  c : R
  α : V → R

/-- Its value at the bits `z`. -/
def BitAff.eval {V R : Type} [Fintype V] [CommRing R] (A : BitAff V R) (z : V → R) : R := A.c + ∑ v, A.α v * z v

/-- The bit point with ones on `S`. -/
def bitInd {V R : Type} [DecidableEq V] [CommRing R] (S : Finset V) : V → R := fun v => if v ∈ S then 1 else 0

/-- The bit-pair second difference `f(e_u + e_v) − f(e_u) − f(e_v) + f(0)`. -/
def pairDiff {V R : Type} [DecidableEq V] [CommRing R] (f : (V → R) → R) (u v : V) : R :=
  f (bitInd {u, v}) - f (bitInd {u}) - f (bitInd {v}) + f (bitInd ∅)

/-- **A straight-line program over `R` on the bits** (Theorem K's runs): slot `j` is a product of two bit-affine
operands (`isProd j`: an MMA slot on bytes loaded from committed or coin data, or formed from them by adds), or a
linear combination of earlier slots plus a bit-affine word (accumulation, merges, copies; ring operations only, so
no shift and no fractional scale). -/
structure Slp (V R : Type) (N : ℕ) where
  isProd : Fin N → Bool
  opA : Fin N → BitAff V R
  opB : Fin N → BitAff V R
  lin : (j : Fin N) → Fin j.val → R
  off : Fin N → BitAff V R

namespace Slp

variable {V R : Type} [Fintype V] [CommRing R] {N : ℕ}

/-- Each slot's coefficients on the products (the closed form). -/
def coef (P : Slp V R N) (j : Fin N) : Fin N → R :=
  if P.isProd j then Pi.single j 1
  else ∑ k : Fin j.val, P.lin j k • P.coef ⟨k.val, k.isLt.trans j.isLt⟩
termination_by j.val
decreasing_by exact k.isLt

end Slp

/-- The design's bits: U's planes `β_p(i, j)` (row `i`, column `j < L`, plane `p`) and the coins `b(t, j, l)` of
`X`'s entries `(tL + j, l)`. -/
abbrev DBits (n L np B n' : ℕ) : Type := (Fin n × Fin L × Fin np) ⊕ (Fin B × Fin L × Fin n')

section Design

variable {R : Type} [CommRing R] {n L np B n' : ℕ}

/-- `P̃_(i, tL+j) = P_(i, tL+j) − 64 + U_ij` with `U_ij = −64 + Σ_p w_p β_p(i, j)` (the constants in `pc`): with `W`
stacked identities, U's column `j` serves every block. -/
def dP (pc : Fin n → Fin B → Fin L → R) (w : Fin np → R) (i : Fin n) (t : Fin B) (j : Fin L) :
    BitAff (DBits n L np B n') R :=
  ⟨pc i t j, Sum.elim (fun a => if a.1 = i ∧ a.2.1 = j then w a.2.2 else 0) fun _ => 0⟩

/-- `X_(tL+j, l) = ⌊x/s⌋_(tL+j, l) + b(t, j, l)` (`⌊x/s⌋` in `vq`). -/
def dX (vq : Fin B → Fin L → Fin n' → R) (t : Fin B) (j : Fin L) (l : Fin n') : BitAff (DBits n L np B n') R :=
  ⟨vq t j l, Sum.elim (fun _ => 0) fun a => if a = (t, j, l) then 1 else 0⟩

/-- Cell-block `(i, l, t)`'s target: block `t`'s increment `Σ_j P̃_(i, tL+j) X_(tL+j, l)` of the committed prefix
sum at cell `(i, l)`. -/
def dIncr (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) (i : Fin n)
    (l : Fin n') (t : Fin B) : (DBits n L np B n' → R) → R :=
  fun z => ∑ j, (dP pc w i t j).eval z * (dX vq t j l).eval z

/-- The committed prefix word `T_t(i, l) = Σ_{k < (t+1)L} P̃_ik X_kl`. -/
def dPrefix (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) (i : Fin n)
    (l : Fin n') (t : ℕ) : (DBits n L np B n' → R) → R :=
  fun z => ∑ t' : Fin B, if t'.val ≤ t then dIncr pc vq w i l t' z else 0

end Design

/-- The re-encoded design's plane weights `w = (1, 2, 4, 8, 16, 32, 65)`: `U = −64 + Σ_p w_p β_p` maps 7 bits onto
`{±1, …, ±64}`, and plane 0's weight is odd. -/
def wRe : Fin 7 → ZMod (2 ^ 32) := ![1, 2, 4, 8, 16, 32, 65]

/-- `Z/2³² → F₂`. -/
def toF2 : ZMod (2 ^ 32) →+* ZMod 2 := ZMod.castHom (by norm_num) (ZMod 2)

/-- A round's cell-blocks: cells `(i, l)` and blocks `t`. -/
abbrev DCells (n n' B : ℕ) : Type := Fin n × Fin n' × Fin B

end TheoremK

section Chain

variable {Vl Vr R : Type} [Fintype Vl] [Fintype Vr] [DecidableEq Vl] [DecidableEq Vr] [CommRing R]

/-- A product of two bit-affine operands, mixed or not. -/
def AffProd (π : (Vl ⊕ Vr → R) → R) : Prop :=
  ∃ A B : BitAff (Vl ⊕ Vr) R, ∀ z, π z = A.eval z * B.eval z

/-- A function of the bits that reads only U's side (`Vl`). -/
def LeftOnly (g : (Vl ⊕ Vr → R) → R) : Prop :=
  ∀ z z' : Vl ⊕ Vr → R, (∀ a, z (.inl a) = z' (.inl a)) → g z = g z'

/-- A function of the bits that reads only the coins' side (`Vr`). -/
def RightOnly (h : (Vl ⊕ Vr → R) → R) : Prop :=
  ∀ z z' : Vl ⊕ Vr → R, (∀ b, z (.inr b) = z' (.inr b)) → h z = h z'

/-- **A one-sided product**: a function of U's bits times a function of the coins, each of any form. -/
def OneSided (π : (Vl ⊕ Vr → R) → R) : Prop :=
  ∃ g h, LeftOnly g ∧ RightOnly h ∧ ∀ z, π z = g z * h z

end Chain

section Overlap

variable {St : ℕ} {ι : Fin St → Type} [∀ t, Fintype (ι t)] [∀ t, DecidableEq (ι t)] {R : Type} [CommRing R]

/-- The round's bit point at draws `U`: draw `s`'s bits are ones on `U s`. -/
def drawInd (U : ∀ s, Finset (ι s)) : (Σ s, ι s) → R := fun v => bitInd (U v.1) v.2

/-- The round's bit point at draws `U` with draw `t`'s bits replaced by `z`. -/
def mergeAt (U : ∀ s, Finset (ι s)) (t : Fin St) (z : ι t → R) : (Σ s, ι s) → R :=
  fun v => Function.update (fun s => (bitInd (U s) : ι s → R)) t z v.1 v.2

/-- **K∥'s operands**: a product of two operands affine in draw `t`'s bits at every bit point of the other draws',
for every draw `t`. -/
def AffMulAt (π : ((Σ s, ι s) → R) → R) : Prop :=
  ∀ (U : ∀ s, Finset (ι s)) (t : Fin St), ∃ A B : BitAff (ι t) R, ∀ z, π (mergeAt U t z) = A.eval z * B.eval z

end Overlap

section OverlapLuck

variable {St : ℕ} {κ ι : Fin St → Type} [∀ t, Fintype (κ t)] [∀ t, Fintype (ι t)] [∀ t, DecidableEq (ι t)]
  {R : Type} [CommRing R]

/-- A round's deficiency with overlapping draws (RS-D1E r2 §2.5): the share of cells and draws at which the cell's
error, on its own draw's bits with every other draw's at theirs, is nonzero at some bit point with at most two
ones. -/
noncomputable def ovDef (e : (Σ t, κ t) → ((Σ s, ι s) → R) → R) : ℝ :=
  prob fun a : (Σ t, κ t) × (∀ s, Finset (ι s)) =>
    ∃ S : Finset (ι a.1.1), S.card ≤ 2 ∧ e a.1 (mergeAt a.2 a.1.1 (bitInd S)) ≠ 0

/-- **A prover's play with overlapping draws** over `n` rounds: per round, its error table on every draw's bits, fixed
before the round's first draw (the round's run straight-line over every draw's bits), then the rest of the play,
which may depend on the round's draws and its check. -/
inductive OvPlay (κ ι : Fin St → Type) (A : Type) : ℕ → Type
  | done : OvPlay κ ι A 0
  | round {n : ℕ} (e : (Σ t, κ t) → ((Σ s, ι s) → A) → A) (next : (Σ t, κ t) × (∀ s, Finset (ι s)) → OvPlay κ ι A n) :
      OvPlay κ ι A (n + 1)

namespace OvPlay

variable {A : Type} [CommRing A]

/-- **The window's game** of a play with overlapping draws: round `w`'s deficiency is `ovDef` of its errors, and its
check passes where the drawn cell's error vanishes at the drawn bits of every draw. -/
noncomputable def game : {n : ℕ} → OvPlay κ ι A n → RoundGame ((Σ t, κ t) × (∀ s, Finset (ι s))) n
  | _, .done => .done
  | _, .round e next => .round (fun _ => ovDef e) (fun a => e a.1 (drawInd a.2) = 0) fun a => (next a).game

omit [∀ t, Fintype (κ t)] [∀ t, Fintype (ι t)] [∀ t, DecidableEq (ι t)] [CommRing A] in
/-- Round `w`'s error table along the realized coins. -/
def errAt : {n : ℕ} → OvPlay κ ι A n → Paths ((Σ t, κ t) × (∀ s, Finset (ι s))) n → Fin n → (Σ t, κ t) →
    ((Σ s, ι s) → A) → A
  | _, .done, _, w => w.elim0
  | _, .round e next, x, w =>
    Fin.cases (motive := fun _ => (Σ t, κ t) → ((Σ s, ι s) → A) → A) e (fun w' => (next x.1).errAt x.2 w') w

end OvPlay

end OverlapLuck

/-- **A straight-line program with products of any form**: as `Slp`, but slot `j`'s product is any function of the
bits (`prod j`: a product of bytes the run formed, affine or not). -/
structure SlpG (V R : Type) (N : ℕ) where
  isProd : Fin N → Bool
  prod : Fin N → (V → R) → R
  lin : (j : Fin N) → Fin j.val → R
  off : Fin N → BitAff V R

namespace SlpG

variable {V R : Type} [Fintype V] [CommRing R] {N : ℕ}

/-- The slots' values at the bits `z`. -/
def val (P : SlpG V R N) (z : V → R) (j : Fin N) : R :=
  if P.isProd j then P.prod j z
  else ∑ k : Fin j.val, P.lin j k * P.val z ⟨k.val, k.isLt.trans j.isLt⟩ + (P.off j).eval z
termination_by j.val
decreasing_by exact k.isLt

/-- The program's linear structure, as an `Slp` (its products' operands dropped). -/
def toSlp (P : SlpG V R N) : Slp V R N := ⟨P.isProd, fun _ => ⟨0, 0⟩, fun _ => ⟨0, 0⟩, P.lin, P.off⟩

/-- Each slot's coefficients on the products. -/
def coef (P : SlpG V R N) : Fin N → Fin N → R := P.toSlp.coef

end SlpG

section ParitySplit

variable {V : Type} {K : ℕ}

/-- A function of the bit points blind to a frame's `x` bits: it agrees on bit points that agree off them. -/
def XBlind (fr : Fin K ⊕ Fin K → V) (g : Finset V → ZMod 2) : Prop :=
  ∀ S S' : Finset V, (∀ v, (∀ j, fr (.inl j) ≠ v) → (v ∈ S ↔ v ∈ S')) → g S = g S'

/-- A function of the bit points blind to a frame's `y` bits. -/
def YBlind (fr : Fin K ⊕ Fin K → V) (g : Finset V → ZMod 2) : Prop :=
  ∀ S S' : Finset V, (∀ v, (∀ j, fr (.inr j) ≠ v) → (v ∈ S ↔ v ∈ S')) → g S = g S'

variable [DecidableEq V] {A : Type} [CommRing A]

/-- **A product two-sided at a frame**: mod 2 it reads both the frame's `x` bits and its `y` bits. -/
def TwoSidedAt (fr : Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) (π : (V → A) → A) : Prop :=
  ¬XBlind fr (fun S => φ (π (bitInd S))) ∧ ¬YBlind fr (fun S => φ (π (bitInd S)))

end ParitySplit

section ParityDesign

variable {Ar : Type} [CommRing Ar] {n L np B n' : ℕ}

/-- **Cell-block `(i, l, t)`'s frame at plane `p₀`**: its `x` bits are row `i` of U at plane `p₀` (U's column `j`
serves every block), its `y` bits the block's coins `b(t, ·, l)`. -/
def dFrame (p₀ : Fin np) (c : DCells n n' B) : Fin L ⊕ Fin L → DBits n L np B n' :=
  Sum.elim (fun j => .inl (c.1, j, p₀)) fun j => .inr (c.2.2, j, c.2.1)

/-- **Charging purity** of a run (RS-E-REF r6 VI.2.1): no product is two-sided at the frames of two cell-blocks
whose words read it with odd coefficients (mod 2 through `φ`, RS-E-REF r8 VIII.3 item 4: a word that reads a product
at an even scale is not charged it). Prefix chains, merges and partners are free: a product of block `t'` read by a
later block's word reads none of that block's coins. -/
def SlpG.ChargePure (φ : Ar →+* ZMod 2) (p₀ : Fin np) {N : ℕ} (P : SlpG (DBits n L np B n') Ar N)
    (word : DCells n n' B → Fin N) : Prop :=
  ∀ c c' i, c ≠ c' → φ (P.coef (word c) i) ≠ 0 → TwoSidedAt (dFrame p₀ c) φ (P.prod i) →
    φ (P.coef (word c') i) ≠ 0 → ¬TwoSidedAt (dFrame p₀ c') φ (P.prod i)

end ParityDesign

section Luck

variable {V R : Type} [Fintype V] [DecidableEq V] [CommRing R]

/-- A function of the bits of degree at most 2 on the cube: every square's second difference is its pair's,
whatever the other bits. Bit-affine words, products of two bit-affine operands, and their sums are such. -/
def CubeQuad (f : (V → R) → R) : Prop :=
  ∀ (S : Finset V) (u v : V), u ≠ v → u ∉ S → v ∉ S →
    f (bitInd (insert u (insert v S))) - f (bitInd (insert u S)) - f (bitInd (insert v S)) + f (bitInd S) =
      pairDiff f u v

end Luck

section Spot

variable {C V R : Type} [Fintype C] [Fintype V] [DecidableEq V] [CommRing R]

/-- A round's deficiency in a spot check: the share of cells whose error is nonzero at some bit point with at most
two ones (the cells whose word is not certified). -/
noncomputable def spotDef (e : C → (V → R) → R) : ℝ :=
  prob fun c : C => ∃ U : Finset V, U.card ≤ 2 ∧ e c (bitInd U) ≠ 0

end Spot

/-- **A round's staged spot play** (per-stage draws: decision 36 as RS-E-REF r5 V.3 repaired it): per stage, its
error table (its committed words minus their targets, per cell, as functions of the stage's bits, both noise blocks
of the stage's products drawn in its draw), fixed by the stage's operand commitment before its draw, then the rest of
the round, which may depend on the draw. -/
inductive StagePlay (C V A : Type) : ℕ → Type
  | done : StagePlay C V A 0
  | stage {m : ℕ} (e : C → (V → A) → A) (next : Finset V → StagePlay C V A m) : StagePlay C V A (m + 1)

namespace StagePlay

variable {C V A : Type} [Fintype C] [Fintype V] [DecidableEq V] [CommRing A]

omit [Fintype C] [Fintype V] [DecidableEq V] [CommRing A] in
/-- Stage `s`'s error table along the round's draws. -/
def errAt : {m : ℕ} → StagePlay C V A m → Paths (Finset V) m → Fin m → C → (V → A) → A
  | _, .done, _, s => s.elim0
  | _, .stage e next, x, s =>
    Fin.cases (motive := fun _ => C → (V → A) → A) e (fun s' => (next x.1).errAt x.2 s') s

omit [Fintype C] [Fintype V] in
/-- The round's check at stage `s` and cell `c`: the error vanishes at the stage's drawn bits. -/
def ok : {m : ℕ} → StagePlay C V A m → Paths (Finset V) m → Fin m → C → Prop
  | _, .done, _, s, _ => s.elim0
  | _, .stage e next, x, s, c =>
    Fin.cases (motive := fun _ => Prop) (e c (bitInd x.1) = 0) (fun s' => (next x.1).ok x.2 s' c) s

/-- **The round's stage game**, `M` stages in all: stage `s`'s step is `1/M` of its share of uncertified cells,
and its draw adds `1/M` of the share of cells it finds wrong. -/
noncomputable def sgame (M : ℕ) : {m : ℕ} → StagePlay C V A m → StageGame (Finset V) m
  | _, .done => .done
  | _, .stage e next =>
    .stage ((M : ℝ)⁻¹ * spotDef e) (fun S => (M : ℝ)⁻¹ * prob fun c => e c (bitInd S) ≠ 0) fun S => (next S).sgame M

end StagePlay

/-- **A prover's staged play** over `n` rounds of `M` stages: per round, its stage play, then the rest of the play,
which may depend on the round's draws and its check. -/
inductive StagedPlay (C V A : Type) (M : ℕ) : ℕ → Type
  | done : StagedPlay C V A M 0
  | round {n : ℕ} (P : StagePlay C V A M) (next : Paths (Finset V) M × (Fin M × C) → StagedPlay C V A M n) :
      StagedPlay C V A M (n + 1)

namespace StagedPlay

variable {C V A : Type} [Fintype C] [Fintype V] [DecidableEq V] [CommRing A] {M : ℕ}

/-- **The window's game** of a staged play: round `w`'s deficiency is its stage game's, and its check passes where
the drawn stage's error at the drawn cell vanishes at that stage's bits. -/
noncomputable def game : {n : ℕ} → StagedPlay C V A M n → RoundGame (Paths (Finset V) M × (Fin M × C)) n
  | _, .done => .done
  | _, .round P next =>
    .round (fun a => (P.sgame M).defi a.1) (fun a => P.ok a.1 a.2.1 a.2.2) fun a => (next a).game

omit [Fintype C] [Fintype V] [DecidableEq V] [CommRing A] in
/-- Round `w`'s stage `s` error table along the realized coins. -/
def errAt : {n : ℕ} → StagedPlay C V A M n → Paths (Paths (Finset V) M × (Fin M × C)) n → Fin n → Fin M → C →
    (V → A) → A
  | _, .done, _, w => w.elim0
  | _, .round P next, x, w =>
    Fin.cases (motive := fun _ => Fin M → C → (V → A) → A) (P.errAt x.1.1) (fun w' => (next x.1).errAt x.2 w') w

end StagedPlay

/-- A round's coins with `St` stages on the re-encoded design: each stage's bits, then the check's stage and
cell-block. -/
abbrev DStage (St n L np B n' : ℕ) : Type := Paths (Finset (DBits n L np B n')) St × (Fin St × DCells n n' B)

/-- A round's bits with overlapping draws: draw `t`'s design bits, stage `t` of shape `(n t, L, B t, n' t)`. -/
abbrev OBits (St : ℕ) (n : Fin St → ℕ) (L np : ℕ) (B n' : Fin St → ℕ) : Type := Σ t, DBits (n t) L np (B t) (n' t)

/-- A round's cell-blocks, over all its stages. -/
abbrev OCells (St : ℕ) (n n' B : Fin St → ℕ) : Type := Σ t, DCells (n t) (n' t) (B t)

/-- A round's draws with overlapping stages: draw `t`'s bits. -/
abbrev ODraws (St : ℕ) (n : Fin St → ℕ) (L np : ℕ) (B n' : Fin St → ℕ) : Type :=
  ∀ t, Finset (DBits (n t) L np (B t) (n' t))

/-- A round's coins with overlapping stages: the check's cell-block, among every stage's, and every stage's draw. -/
abbrev OSpot (St : ℕ) (n : Fin St → ℕ) (L np : ℕ) (B n' : Fin St → ℕ) : Type :=
  OCells St n n' B × ODraws St n L np B n'

section ParityOverlapRound

variable {St : ℕ} {Ar : Type} [CommRing Ar] {n B n' : Fin St → ℕ} {L np : ℕ}

/-- **K∥'s products for one-sided operands of any form**: at every bit point of the other draws, one-sided on draw
`t`'s bits, for every draw `t`; each side may depend on the other draws' bits in any way. -/
def OneSidedAt (π : (OBits St n L np B n' → Ar) → Ar) : Prop :=
  ∀ (W : ODraws St n L np B n') (t : Fin St), OneSided fun z => π (mergeAt W t z)

/-- **Charging purity across draws**: at every bit point of the draws, no product is two-sided at the frames of two
cell-blocks, of any stages, whose words read it with odd coefficients (mod 2 through `φ`). -/
def SlpG.ChargePureAt (φ : Ar →+* ZMod 2) (p₀ : Fin np) {N : ℕ} (P : SlpG (OBits St n L np B n') Ar N)
    (word : OCells St n n' B → Fin N) : Prop :=
  ∀ (W : ODraws St n L np B n') (c c' : OCells St n n' B) i, c ≠ c' → φ (P.coef (word c) i) ≠ 0 →
    TwoSidedAt (dFrame p₀ c.2) φ (fun z => P.prod i (mergeAt W c.1 z)) → φ (P.coef (word c') i) ≠ 0 →
      ¬TwoSidedAt (dFrame p₀ c'.2) φ fun z => P.prod i (mergeAt W c'.1 z)

end ParityOverlapRound

/-- **An adversary's window run on the staged design**, as the analysis reads it: its staged play (per round and
stage, the error table fixed by the stage's operand commitment before its draw), the per-round binding and
zero-knowledge break events, the noise reads `V`, and along every play `x` each round's stages as straight-line
programs over `Ar` on the stage's bits (`prog x w s`), on committed constants `pc x w s` and `vq x w s`, with the slot
of each cell's committed prefix word (`word x w s`). -/
structure StagedRun (R St n L np B n' : ℕ) (Ar : Type) [CommRing Ar] where
  play : StagedPlay (DCells n n' B) (DBits n L np B n') Ar St R
  Bind : Fin R → Paths (DStage St n L np B n') R → Prop
  ZK : Fin R → Paths (DStage St n L np B n') R → Prop
  V : Paths (DStage St n L np B n') R → ℝ
  N : Paths (DStage St n L np B n') R → Fin R → Fin St → ℕ
  prog : ∀ x w s, SlpG (DBits n L np B n') Ar (N x w s)
  pc : Paths (DStage St n L np B n') R → Fin R → Fin St → Fin n → Fin B → Fin L → Ar
  vq : Paths (DStage St n L np B n') R → Fin R → Fin St → Fin B → Fin L → Fin n' → Ar
  word : ∀ x w s, DCells n n' B → Fin (N x w s)

/-- **The analysis's map from adversaries to their window runs**: for each adversary `M` in `Mc` against the
service `S` and each value `r` of its own coins, its run. -/
abbrev RunMap (Mc : AlgModel) (S : Service) (R St n L np B n' : ℕ) (Ar : Type) [CommRing Ar] : Type 1 :=
  ∀ (Ω' : Type) [Fintype Ω'], Mc.Alg (S.Seed × Ω') (S.Com × S.Ans) → Ω' → StagedRun R St n L np B n' Ar

/-- **An adversary's window run with overlapping stages**, as the analysis reads it: its play over the round's draws
(per round, the error tables of every stage fixed before the round's first draw), the per-round binding
and zero-knowledge break events, the noise reads `V`, and along every play `x` each round's run as one straight-line
program over `Ar` on every draw's bits (`prog x w`), on committed constants per stage (`pc x w t`, `vq x w t`), with
the slot of each cell's committed prefix word (`word x w`). -/
structure OverlapRun (R St : ℕ) (n : Fin St → ℕ) (L np : ℕ) (B n' : Fin St → ℕ) (Ar : Type) [CommRing Ar] where
  play : OvPlay (fun t => DCells (n t) (n' t) (B t)) (fun t => DBits (n t) L np (B t) (n' t)) Ar R
  Bind : Fin R → Paths (OSpot St n L np B n') R → Prop
  ZK : Fin R → Paths (OSpot St n L np B n') R → Prop
  V : Paths (OSpot St n L np B n') R → ℝ
  N : Paths (OSpot St n L np B n') R → Fin R → ℕ
  prog : ∀ x w, SlpG (OBits St n L np B n') Ar (N x w)
  pc : Paths (OSpot St n L np B n') R → Fin R → ∀ t, Fin (n t) → Fin (B t) → Fin L → Ar
  vq : Paths (OSpot St n L np B n') R → Fin R → ∀ t, Fin (B t) → Fin L → Fin (n' t) → Ar
  word : ∀ x w, OCells St n n' B → Fin (N x w)

/-- **The analysis's map from adversaries to their window runs with overlapping stages**. -/
abbrev ORunMap (Mc : AlgModel) (S : Service) (R St : ℕ) (n : Fin St → ℕ) (L np : ℕ) (B n' : Fin St → ℕ) (Ar : Type)
    [CommRing Ar] : Type 1 :=
  ∀ (Ω' : Type) [Fintype Ω'], Mc.Alg (S.Seed × Ω') (S.Com × S.Ans) → Ω' → OverlapRun R St n L np B n' Ar

end Pouw.RelativeSecurity
