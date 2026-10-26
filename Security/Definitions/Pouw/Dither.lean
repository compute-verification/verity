import Definitions.Pouw.RelativeSecurity

/-!
# The no-fold Dither protocol's work-soundness: the vocabulary its guarantee reads (trusted)

The headline of the ACCEL program (`Definitions.Pouw.Guarantees.Dither.WorkSound`) is stated over:

* **the accelerator model** `G = (Σ, M, R, ρ, I, u)` (`Accel`): a machine's programs, the period's devices' resources
  with their data-sheet rates `ρ`, their memory `M`, a run's use of each resource `u`, the two device premises (G1)
  capacity and (G2) rate (`Accel.G1`, `Accel.G2`) and the arithmetic premises (P1) and (P2) (`Accel.P1`, `Accel.P2`),
  which MODEL-SPEC (CMP-109) instantiates from `Definitions.Core.Accel`;
* **prices** on the resources (`Accel.cost`): a run's cost `Σ_r μ_r u_r` at a price `μ_r ≥ 0` a unit of resource `r`
  (at `μ_r = λ_r / ρ_r`, the device-seconds it takes at the price vector `λ`);
* **a round** of a commit-then-check protocol (`RoundProto`): the words the prover computes after the first coin, their
  commitment (the SHA-512 Merkle root over the words' leaves), and the check, parametric in the protocol (R-ACCEL-2),
  with the no-fold protocol as one instance (`NoFold`, `noFoldRound`) whose knobs are parameters, so protocol v1 and
  v2 are instances;
* **a round program's moments** (`wordsMoment`, `commitMoment`): the moment bounds of the two "public best" premises,
  SAK(matmul) on the words and SHA-512's speed on their commitment (`Definitions.Pouw.Assumptions.Dither`'s `SAKMatmul`, `ShaSpeed`);
* **a monitoring period's play** round by round (`PeriodPlay`: per round a words program and a commitment program),
  the analysis's map from adversaries to plays (`PeriodMap`, `PeriodRun`), and the rounds' floor (`RoundsFloor`);
* **freed compute** (`Period`, `Period.FreedWithin`, `Period.DualOpt`, `FreedSound`): R-ACCEL-1's definition, kept in
  this one place, and its probability form, the headline's conclusion;
* the error (`epsWorkSound`).
-/

namespace Pouw.Dither

open Pouw.RelativeSecurity Finset

section Model

/-- **The accelerator model** at the granularity this guarantee reads: a machine `Mc` (its programs, what they compute
and their composition), the period's devices' resources `Res` (tensor cores, issue slots, shared-memory and register
ports, device memory, links), each at its data-sheet rate `rate` (`ρ`, units a second), their memory `mem` (`M`, bits),
and per run its use of each resource `use` (`u`), the bits it holds at its peak `held`, its wall time `wall`, and the
bits it reads from outside the period's devices `ext` (what the network certifier counts). `P1` and `P2` are the
arithmetic premises, exact operand formats and one accumulator per product, as MODEL-SPEC (CMP-109) defines them in
`Definitions.Core.Accel.Principles`. `θ = (M, ρ, u)` are the model's parameters. -/
structure Accel where
  /-- The programs, what they compute and how they compose. -/
  Mc : Machine
  /-- The period's devices' resources. -/
  Res : Type
  [instRes : Fintype Res]
  /-- `ρ`: each resource's data-sheet rate, units a second, over the period's devices. -/
  rate : Res → ℝ
  /-- `M`: the period's devices' memory, bits. -/
  mem : ℝ
  /-- `u`: a run's use of each resource, units. -/
  use : ∀ {I O : Type}, Mc.Alg I O → I → Res → ℝ
  /-- The bits a run holds at its peak. -/
  held : ∀ {I O : Type}, Mc.Alg I O → I → ℝ
  /-- A run's wall time, seconds. -/
  wall : ∀ {I O : Type}, Mc.Alg I O → I → ℝ
  /-- The bits a run reads from outside the period's devices (imported). -/
  ext : ∀ {I O : Type}, Mc.Alg I O → I → ℝ
  /-- (P1) exact operand formats, as MODEL-SPEC defines it. -/
  P1 : Prop
  /-- (P2) a product lives in one accumulator, as MODEL-SPEC defines it. -/
  P2 : Prop

attribute [instance] Accel.instRes

namespace Accel

variable (G : Accel)

/-- **(G1) capacity**: every program, its code and constants with what it holds at its peak, fits in the period's
devices' memory. A table of answers indexed by the live coins is a program, so G1 bounds it. -/
def G1 : Prop := ∀ {I O : Type} (f : G.Mc.Alg I O) (x : I), G.Mc.size f + G.held f x ≤ G.mem

/-- **(G2) rate**: every resource runs at a positive rate, and a run uses each at most at its rate over its wall time. -/
def G2 : Prop :=
  (∀ r, 0 < G.rate r) ∧
    ∀ {I O : Type} (f : G.Mc.Alg I O) (x : I) (r : G.Res), 0 ≤ G.use f x r ∧ G.use f x r ≤ G.rate r * G.wall f x

/-- **A run's cost at the prices `μ`**: `Σ_r μ_r u_r`, `μ_r` the price of a unit of resource `r`. At `μ_r = λ_r / ρ_r`
it is the device-seconds the run takes at the price vector `λ`; at a point mass it is one resource's busy time. -/
noncomputable def cost (μ : G.Res → ℝ) {I O : Type} (f : G.Mc.Alg I O) (x : I) : ℝ := ∑ r, μ r * G.use f x r

end Accel

end Model

section Round

/-- **A round of a commit-then-check protocol**: an instance the prover committed before the round (`Inst`, the
workload's operands), the round's first live coin `Draw` (σ₁, the noise), the words the prover computes after it
`Tr`, their commitment `commit` (in `Root`), the check's live coin `Chk` (σ₂), and whether the check passes on the
words (`ok`). The prover sends the commitment before σ₂ is drawn and answers the check with `Ans` (the opened words
and their openings); `accept` is what the verifier checks of the answer against the commitment, and `ok` what the
analysis reads of the words the commitment binds (`Extract` links them). The proof layer (C-Flock's proofs of the
opened words) is the service's, not the round's. -/
structure RoundProto where
  Inst : Type
  Draw : Type
  Chk : Type
  [instDraw : Fintype Draw]
  [instChk : Fintype Chk]
  Tr : Type
  Root : Type
  commit : Tr → Root
  ok : Inst → Draw → Chk → Tr → Prop
  Ans : Type
  accept : Inst → Draw → Chk → Root → Ans → Prop

attribute [instance] RoundProto.instDraw RoundProto.instChk

/-- A word: integers mod `2^32`. -/
abbrev Word : Type := ZMod (2 ^ 32)

/-- **The no-fold protocol's knobs** (PROTO-DESIGN, R-ACCEL-2): `A` is `n × m`, `B` is `m × n'` with `m = nb · L`,
`L` the segment length; the noise (`Draw`, and `noise`, the noised operands `(Ã, B̃)` from the committed operands and
the noise: HYB noise in protocol v1, whatever v2 freezes); the check's coin `Chk` and the `s` words it selects
(`cells`: the frozen protocol's stratified draw, any width, no power-of-two size); `merkle`, the commitment to the
words: each word a SHA-512 Merkle leaf (hm96 rows in frame-v3-sha512 trees), no fold; and `opens`, the verifier's
check that a word with its authentication path `Path` opens a root at a cell. -/
structure NoFold where
  n : ℕ
  n' : ℕ
  nb : ℕ
  L : ℕ
  Draw : Type
  [instDraw : Fintype Draw]
  noise : (Fin n → Fin (nb * L) → Word) × (Fin (nb * L) → Fin n' → Word) → Draw →
    (Fin n → Fin (nb * L) → Word) × (Fin (nb * L) → Fin n' → Word)
  s : ℕ
  Chk : Type
  [instChk : Fintype Chk]
  cells : Chk → Fin s → Fin n × Fin n' × Fin nb
  Root : Type
  merkle : (Fin n → Fin n' → Fin nb → Word) → Root
  Path : Type
  opens : Root → Fin n × Fin n' × Fin nb → Word → Path → Prop

attribute [instance] NoFold.instDraw NoFold.instChk

/-- **A no-fold word** at cell `(i, j)` and segment `b`: the noised product's prefix sum at the end of segment `b`,
`Σ_{t < (b+1) L} Ã_it B̃_tj` mod `2^32`. -/
def noFoldWord (k : NoFold) (ab : (Fin k.n → Fin (k.nb * k.L) → Word) × (Fin (k.nb * k.L) → Fin k.n' → Word))
    (i : Fin k.n) (j : Fin k.n') (b : Fin k.nb) : Word :=
  ∑ t : Fin (k.nb * k.L), if t.val < (b.val + 1) * k.L then ab.1 i t * ab.2 t j else 0

/-- **The no-fold round at its knobs**: the instance is the committed operands `(A, B)`, the first coin the noise, the
words every segment prefix sum of the noised product (`n n' nb` words), committed by `merkle`, and the check the `s`
cells and segments its coin selects, passing where every checked word is the noised product's. The answer is each
checked word with its path, accepted where every path opens the root and every word is the noised product's. -/
def noFoldRound (k : NoFold) : RoundProto where
  Inst := (Fin k.n → Fin (k.nb * k.L) → Word) × (Fin (k.nb * k.L) → Fin k.n' → Word)
  Draw := k.Draw
  Chk := k.Chk
  Tr := Fin k.n → Fin k.n' → Fin k.nb → Word
  Root := k.Root
  commit := k.merkle
  ok a d c T := ∀ q, T (k.cells c q).1 (k.cells c q).2.1 (k.cells c q).2.2 =
    noFoldWord k (k.noise a d) (k.cells c q).1 (k.cells c q).2.1 (k.cells c q).2.2
  Ans := Fin k.s → Word × k.Path
  accept a d c r y := ∀ q, k.opens r (k.cells c q) (y q).1 (y q).2 ∧
    (y q).1 = noFoldWord k (k.noise a d) (k.cells c q).1 (k.cells c q).2.1 (k.cells c q).2.2

/-- **The honest words**: every segment prefix sum of the noised product. -/
def noFoldHonest (k : NoFold) (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) : (noFoldRound k).Tr :=
  fun i j b => noFoldWord k (k.noise a d) i j b

end Round

section Moments

/-- The uniform average of `f`. -/
noncomputable def avg {Ω : Type} [Fintype Ω] (f : Ω → ℝ) : ℝ := (∑ ω, f ω) / Fintype.card Ω

/-- **A shortfall** at the constant `c` against the honest cost `H`: `(c − (cost + β · imported)/H)₊`, how far a
program's cost, with each imported bit worth `β`, falls short of `c` times the honest program's `H`. -/
noncomputable def roundShortfall (c β H cost imp : ℝ) : ℝ := max 0 (c - (cost + β * imp) / H)

open Classical in
/-- **A words program's moment** (SAK(matmul)'s): the average over the round's coins of `e^{η · shortfall}` where the
check passes on the words it computes, `0` where it fails, its cost at the prices `μ` against `c H`. The program is
fixed before the round's coins and computes the words from the first coin alone. -/
noncomputable def wordsMoment (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (c η β H : ℝ) (a : P.Inst)
    (Mw : G.Mc.Alg P.Draw P.Tr) : ℝ :=
  avg fun x : P.Draw × P.Chk =>
    if P.ok a x.1 x.2 (G.Mc.eval Mw x.1) then
      Real.exp (η * roundShortfall c β H (G.cost μ Mw x.1) (G.ext Mw x.1)) else 0

open Classical in
/-- **A commitment program's moment** (SHA-512 speed's): for a commitment program `Mh` beside a words program `Mw`, the
average over the round's coins of `e^{η · shortfall}` of `Mh`'s cost against `c H`, where the check passes on `Mw`'s
words, `0` where it fails. -/
noncomputable def commitMoment (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (c η β H : ℝ) (a : P.Inst)
    (Mw : G.Mc.Alg P.Draw P.Tr) (Mh : G.Mc.Alg P.Draw P.Root) : ℝ :=
  avg fun x : P.Draw × P.Chk =>
    if P.ok a x.1 x.2 (G.Mc.eval Mw x.1) then
      Real.exp (η * roundShortfall c β H (G.cost μ Mh x.1) (G.ext Mh x.1)) else 0

end Moments

section Period

/-- **A monitoring period's play**, round by round: per round, the instance committed before it, the round's words
program and commitment program (both fixed before the round's coins, on the history), then the rest of the play,
which may depend on the round's coins. -/
inductive PeriodPlay (G : Accel) (P : RoundProto) : ℕ → Type
  | done : PeriodPlay G P 0
  | round {k : ℕ} (a : P.Inst) (Mw : G.Mc.Alg P.Draw P.Tr) (Mh : G.Mc.Alg P.Draw P.Root)
      (next : P.Draw × P.Chk → PeriodPlay G P k) : PeriodPlay G P (k + 1)

namespace PeriodPlay

variable {G : Accel} {P : RoundProto}

/-- Every round's check passes on the words its words program computes. -/
def accepts : {k : ℕ} → PeriodPlay G P k → Paths (P.Draw × P.Chk) k → Prop
  | _, .done, _ => True
  | _, .round a Mw _ next, x => P.ok a x.1.1 x.1.2 (G.Mc.eval Mw x.1.1) ∧ (next x.1).accepts x.2

/-- Every round's commitment program computes the commitment to its words program's words, on every first coin, at
every node of the play. -/
def Computes : {k : ℕ} → PeriodPlay G P k → Prop
  | _, .done => True
  | _, .round _ Mw Mh next => (∀ d, G.Mc.eval Mh d = P.commit (G.Mc.eval Mw d)) ∧ ∀ y, (next y).Computes

/-- The rounds' use of a resource (words and commitment programs), summed along the realized coins. -/
def use : {k : ℕ} → PeriodPlay G P k → Paths (P.Draw × P.Chk) k → G.Res → ℝ
  | _, .done, _, _ => 0
  | _, .round _ Mw Mh next, x, r => G.use Mw x.1.1 r + G.use Mh x.1.1 r + (next x.1).use x.2 r

/-- The rounds' imported bits, summed along the realized coins. -/
def ext : {k : ℕ} → PeriodPlay G P k → Paths (P.Draw × P.Chk) k → ℝ
  | _, .done, _ => 0
  | _, .round _ Mw Mh next, x => G.ext Mw x.1.1 + G.ext Mh x.1.1 + (next x.1).ext x.2

/-- The rounds' cost at the prices `μ` along the realized coins. -/
noncomputable def cost {k : ℕ} (pl : PeriodPlay G P k) (μ : G.Res → ℝ) (x : Paths (P.Draw × P.Chk) k) : ℝ :=
  ∑ r, μ r * pl.use x r

end PeriodPlay

/-- **The prover of a monitoring period of `k` rounds**, one round at a time (RT-LEAN L1 and L18; RT-PROTO P8 with a
node's rounds in order). It keeps a state `St` across rounds, starting from `start` run on the oracle `Orc` (a uniform
`Orc` stands for SHA-512 as a random oracle; `Orc := Unit` for none). Round `w`, in the order of the messages:
`inst w` commits the round's instance; the verifier draws the round's dither coins `d`; `work w` is everything the
prover computes between that dither and its commitment, and returns the commitment, sent before the check coins are
drawn; the verifier draws the check coins `c`; `answer w` answers them. A round's programs see the earlier rounds'
coins only through the state, so nothing before round `w`'s commitment depends on round `w`'s check coins or on any
later coin. -/
structure Prover (G : Accel) (P : RoundProto) (Orc : Type) (k : ℕ) where
  St : Type
  start : G.Mc.Alg Orc St
  inst : Fin k → G.Mc.Alg St (St × P.Inst)
  work : Fin k → G.Mc.Alg (St × P.Draw) (St × P.Root)
  answer : Fin k → G.Mc.Alg (St × P.Chk) (St × P.Ans)

namespace Prover

variable {G : Accel} {P : RoundProto} {Orc : Type} {k : ℕ} (A : Prover G P Orc k)

/-- The state at the start of round `w` (it stays put past `k`), at the oracle `o` and the verifier's coins `x`. -/
def state (o : Orc) (x : Fin k → P.Draw × P.Chk) : ℕ → A.St
  | 0 => G.Mc.eval A.start o
  | w + 1 =>
    if h : w < k then
      (G.Mc.eval (A.answer ⟨w, h⟩) ((G.Mc.eval (A.work ⟨w, h⟩)
        ((G.Mc.eval (A.inst ⟨w, h⟩) (state o x w)).1, (x ⟨w, h⟩).1)).1, (x ⟨w, h⟩).2)).1
    else state o x w

/-- Round `w`'s input to `work`: the state after its instance's commitment, and its dither coins. -/
def workIn (o : Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) : A.St × P.Draw :=
  ((G.Mc.eval (A.inst w) (A.state o x w)).1, (x w).1)

/-- Round `w`'s input to `answer`: the state after its commitment, and its check coins. -/
def answerIn (o : Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) : A.St × P.Chk :=
  ((G.Mc.eval (A.work w) (A.workIn o x w)).1, (x w).2)

/-- Round `w`'s instance. -/
def instAt (o : Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) : P.Inst :=
  (G.Mc.eval (A.inst w) (A.state o x w)).2

/-- Round `w`'s commitment. -/
def root (o : Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) : P.Root := (G.Mc.eval (A.work w) (A.workIn o x w)).2

/-- Round `w`'s answer. -/
def ans (o : Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) : P.Ans := (G.Mc.eval (A.answer w) (A.answerIn o x w)).2

/-- **Accepted**: every round's answer passes the verifier's check against the round's commitment. -/
def Accepted (o : Orc) (x : Fin k → P.Draw × P.Chk) : Prop :=
  ∀ w, P.accept (A.instAt o x w) (x w).1 (x w).2 (A.root o x w) (A.ans o x w)

/-- The rounds' use of resource `r`: every round's three programs at their inputs. `start` runs before the period. -/
noncomputable def use (o : Orc) (x : Fin k → P.Draw × P.Chk) (r : G.Res) : ℝ :=
  ∑ w, (G.use (A.inst w) (A.state o x w) r + G.use (A.work w) (A.workIn o x w) r +
    G.use (A.answer w) (A.answerIn o x w) r)

/-- The rounds' imported bits. -/
noncomputable def ext (o : Orc) (x : Fin k → P.Draw × P.Chk) : ℝ :=
  ∑ w, (G.ext (A.inst w) (A.state o x w) + G.ext (A.work w) (A.workIn o x w) +
    G.ext (A.answer w) (A.answerIn o x w))

end Prover

/-- **An adversary's period as the analysis reads it**: its play, and per round the binding and zero-knowledge break
events of the proof layer. -/
structure PeriodRun (G : Accel) (P : RoundProto) (R : ℕ) where
  play : PeriodPlay G P R
  Bind : Fin R → Paths (P.Draw × P.Chk) R → Prop
  ZK : Fin R → Paths (P.Draw × P.Chk) R → Prop

/-- **The analysis's map from adversaries to their periods**: for each adversary `M` against the service `S` and each
value of its own coins, its run. -/
abbrev PeriodMap (G : Accel) (S : Service) (P : RoundProto) (R : ℕ) : Type 1 :=
  ∀ (Ω' : Type) [Fintype Ω'], G.Mc.Alg (S.Seed × Ω') (S.Com × S.Ans) → Ω' → PeriodRun G P R

/-- **The rounds' floor, adversary by adversary**: every adversary that commits first has its accepted periods whose
rounds (the credited part, as the analysis `run` cuts it out of the adversary's run, along the period's coins `e`)
cost below `F` at the prices `μ` with probability at most `κ Ω M`. -/
def RoundsFloor (G : Accel) (S : Service) (P : RoundProto) {R : ℕ} (run : PeriodMap G S P R)
    (e : S.Seed ≃ Paths (P.Draw × P.Chk) R) (μ : G.Res → ℝ) (F : ℝ)
    (κ : ∀ (Ω : Type) [Fintype Ω], G.Mc.Alg (S.Seed × Ω) (S.Com × S.Ans) → ENNReal) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : G.Mc.Alg (S.Seed × Ω) (S.Com × S.Ans)),
    (∀ σ σ' r, (G.Mc.eval M (σ, r)).1 = (G.Mc.eval M (σ', r)).1) →
    ENNReal.ofReal (prob fun ω : S.Seed × Ω => S.verify (G.Mc.eval M ω).1 ω.1 (G.Mc.eval M ω).2 = true ∧
      (run Ω M ω.2).play.cost μ (e ω.1) < F) ≤ κ Ω M

/-- **The headline's error over a monitoring period of `k` rounds**:
`ε = (1 − h)(1 − c_mm) + h (1 − c_sha) + ln(1/δ)/(η k) + β K/(k H)`, with `H = H_mm + H_sha` the honest round's cost
(its matmul part, the words, and its hashing part, the commitment), `h = H_sha / H` the hashing's share, `c_mm` and
`c_sha` the two public-best constants (SAK(matmul) and SHA-512's speed), the sampling luck at confidence `δ` and rate
`η`, and the network import (`K` bits at most, the network certifier's, each worth `β`). -/
noncomputable def epsWorkSound (cmm csha Hmm Hsha η δ β K : ℝ) (k : ℕ) : ℝ :=
  (Hmm * (1 - cmm) + Hsha * (1 - csha)) / (Hmm + Hsha) + Real.log (1 / δ) / (η * k) + β * K / (k * (Hmm + Hsha))

end Period

section Freed

/-- **A monitoring period's devices and the other work they could do**, in resource units: the period's length `T`
(seconds), the other jobs `J`, each with its value `val` per unit and its use `needs` per unit, and the honest use
`hon` of a credited round (the honest words and commitment programs'). -/
structure Period (G : Accel) where
  T : ℝ
  J : Type
  [instJ : Fintype J]
  val : J → ℝ
  needs : J → G.Res → ℝ
  hon : G.Res → ℝ

attribute [instance] Period.instJ

namespace Period

variable {G : Accel} (Pd : Period G)

/-- What the devices can do in the period (G2): `ρ_r T` of resource `r`. -/
def cap (r : G.Res) : ℝ := G.rate r * Pd.T

/-- A strategy fits the devices: its credited part (its rounds, `PeriodPlay.use`) uses `X`, its other work `z ≥ 0`
units of each job, and the two together use no resource past its capacity. -/
def Fits (X : G.Res → ℝ) (z : Pd.J → ℝ) : Prop := (∀ j, 0 ≤ z j) ∧ ∀ r, X r + ∑ j, z j * Pd.needs j r ≤ Pd.cap r

/-- A strategy's value: its credited work `W` plus its other work's. -/
def value (W : ℝ) (z : Pd.J → ℝ) : ℝ := W + ∑ j, Pd.val j * z j

/-- **Freed compute is at most `ε C`** (R-ACCEL-1, Daniel 25 Oct): a strategy credited `W` with other work `z` gets
no more credited work plus other work out of the period's devices than an honest strategy on the same devices, plus
`ε C`. An honest strategy is credited `W'` at the honest use `W' · hon` and does other work `z'` beside it. Its dual is
a price on the resources (`Definitions.Pouw.Guarantees.Dither.FreedOfFloor`). This is the definition's one place. -/
def FreedWithin (W : ℝ) (z : Pd.J → ℝ) (ε C : ℝ) : Prop :=
  ∃ (W' : ℝ) (z' : Pd.J → ℝ), 0 ≤ W' ∧ Pd.Fits (fun r => W' * Pd.hon r) z' ∧ Pd.value W z ≤ Pd.value W' z' + ε * C

/-- **`μ` is the period's dual price at credited work `W`**: prices are nonnegative, every job uses nonnegative
resources and is worth at most its use at `μ` (dual feasibility), and an honest strategy credited `W` does other work
worth exactly the capacity it leaves at `μ` (complementary slackness: the honest optimum `OPT_honest`). A choice of
the price, not a premise about the prover: by linear programming duality such a `μ` exists whenever the honest
strategy credited `W` fits. -/
def DualOpt (μ : G.Res → ℝ) (W : ℝ) : Prop :=
  (∀ r, 0 ≤ μ r) ∧ (∀ j r, 0 ≤ Pd.needs j r) ∧ (∀ j, Pd.val j ≤ ∑ r, μ r * Pd.needs j r) ∧
    ∃ z' : Pd.J → ℝ, Pd.Fits (fun r => W * Pd.hon r) z' ∧ ∑ j, Pd.val j * z' j = ∑ r, μ r * (Pd.cap r - W * Pd.hon r)

end Period

/-- **Freed compute at most `ε C`, adversary by adversary** (R-ACCEL-1's probability form): for every adversary that
commits first, the probability that its period is accepted and some other work `z` fits on the period's devices beside
its credited part (its rounds' use, as the analysis `run` cuts them out of its run) with
`W + W_other(z) > OPT_honest + ε C` (`¬ Period.FreedWithin`) is at most `κ Ω M`. The credited work is `W`. -/
def FreedSound (G : Accel) (S : Service) (P : RoundProto) {R : ℕ} (run : PeriodMap G S P R)
    (e : S.Seed ≃ Paths (P.Draw × P.Chk) R) (Pd : Period G) (W ε C : ℝ)
    (κ : ∀ (Ω : Type) [Fintype Ω], G.Mc.Alg (S.Seed × Ω) (S.Com × S.Ans) → ENNReal) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : G.Mc.Alg (S.Seed × Ω) (S.Com × S.Ans)),
    (∀ σ σ' r, (G.Mc.eval M (σ, r)).1 = (G.Mc.eval M (σ', r)).1) →
    ENNReal.ofReal (prob fun ω : S.Seed × Ω => S.verify (G.Mc.eval M ω).1 ω.1 (G.Mc.eval M ω).2 = true ∧
      ∃ z, Pd.Fits ((run Ω M ω.2).play.use (e ω.1)) z ∧ ¬ Pd.FreedWithin W z ε C) ≤ κ Ω M

end Freed

end Pouw.Dither
