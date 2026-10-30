import Definitions.Pouw.RelativeSecurity

/-!
# The no-fold Dither protocol's work-soundness: the vocabulary its guarantee reads (trusted)

The headline of the ACCEL program (`Definitions.Pouw.Guarantees.Dither.WorkSound`) is stated over:

* **the accelerator model** `G = (Σ, M, R, ρ, I, u)` (`Accel`): a machine's programs, the period's devices' resources
  with their data-sheet rates `ρ`, their memory `M`, a run's use of each resource `u`, the three device premises (G1)
  capacity, (G2) rate and (G3) priced oracle calls (`Accel.G1`, `Accel.G2`, `Accel.G3`), the oracle reached only
  through calls (`Accel.OracleCalls`) and the arithmetic premises (P1) and (P2) (`Accel.P1`, `Accel.P2`), which
  MODEL-SPEC instantiates from `Definitions.Core.Accel`;
* **prices** on the resources (`Accel.cost`): a run's cost `Σ_r μ_r u_r` at a price `μ_r ≥ 0` a unit of resource `r`
  (at `μ_r = λ_r / ρ_r`, the device-seconds it takes at the price vector `λ`);
* **a round** of a commit-then-check protocol (`RoundProto`): the words the prover computes after the first coin, their
  commitment (the SHA-512 Merkle root over the words' leaves), and the check, parametric in the protocol,
  with the no-fold protocol as one instance (`NoFold`, `noFoldRound`) whose knobs are parameters, so protocol v1 and
  v2 are instances;
* **a round program's moments** (`wordsMoment`, `commitMoment`): the moment bounds of the two "public best" premises,
  SAK(matmul) on the words and SHA-512's speed on their commitment (`Definitions.Pouw.Assumptions.Dither`'s
  `SAKMatmul`, `ShaSpeed`), in units of priced cost, outside a degenerate set of dithers;
* **a monitoring period's prover** sampling block by sampling block on instances registered before the period
  (`Prover`), what it sends (`Prover.transcript`), its timed use and oracle calls (`Prover.timedUse`, `Prover.calls`:
  its sampling blocks' work, from each block's dither coins to its commitment), and the service's
  verdict on the transcript and the auditor's devices' readings (`Verdict`, `Readings`, `Prover.Passes`);
* **freed compute** (`Period`, `Period.FreedWithin`, `Period.DualOpt`, `FreedSound`): the definition, kept in
  this one place, and its probability form, the headline's conclusion;
* the error (`epsWorkSound`).
-/

namespace Pouw.Dither

open Pouw.RelativeSecurity Finset

section Model

/-- **The accelerator model** at the granularity this guarantee reads: a machine `Mc` (its programs, what they compute
and their composition), the period's devices' resources `Res` (tensor cores, issue slots, shared-memory and register
ports, device memory, links), each at its data-sheet rate `rate` (`ρ`, units a second), their memory `mem` (`M`, bits),
and per run its use of each resource `use` (`u`), the bits it holds at its peak `held`, its wall time `wall`, the
bits it exchanges with machines outside the period's devices `ext` (what the network certifier counts), and its oracle
calls `calls`, each a step of the machine that uses `callUse`. `P1` and `P2` are the arithmetic
premises, exact operand formats and one accumulator per product, as MODEL-SPEC defines them in
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
  /-- The bits a run exchanges with machines outside the period's devices, in either direction (what the network
  certifier counts, into and out of the devices), other than the certifier's own coin stream. -/
  ext : ∀ {I O : Type}, Mc.Alg I O → I → ℝ
  /-- The oracle calls a run makes: SHA-512 evaluations through the machine's oracle-call step. -/
  calls : ∀ {I O : Type}, Mc.Alg I O → I → ℝ
  /-- The oracle-call step's use of each resource: the pinned public SHA-512 implementation's, a call. -/
  callUse : Res → ℝ
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

/-- **(G2) rate**: every resource runs at a positive rate, a run uses each at most at its rate over its wall time, and
its uses and exchanged bits are counts (nonnegative). -/
def G2 : Prop :=
  (∀ r, 0 < G.rate r) ∧
    ∀ {I O : Type} (f : G.Mc.Alg I O) (x : I),
      0 ≤ G.ext f x ∧ ∀ r : G.Res, 0 ≤ G.use f x r ∧ G.use f x r ≤ G.rate r * G.wall f x

/-- **(G3) oracle calls are priced steps**: a run's oracle calls are a count, and each is a step of
the machine that uses `callUse` of every resource, so a run uses at least its calls' `callUse`. -/
def G3 : Prop :=
  ∀ {I O : Type} (f : G.Mc.Alg I O) (x : I), 0 ≤ G.calls f x ∧ ∀ r, G.calls f x * G.callUse r ≤ G.use f x r

/-- **Programs reach the oracle only through their calls**, at an oracle that is a table
`eo : Orc ≃ (X → D)`: at every oracle and input, a program that reads the oracle computes the same at every oracle
that agrees with it on some set of at most as many points as its calls. With (G3), a program pays `callUse` for every
point of the table it reads; nothing reads the table as data. The oracle stays in the programs' input types (the
machine has no oracle step of its own), and this premise is what makes that input a query interface. -/
def OracleCalls {Orc X D : Type} (eo : Orc ≃ (X → D)) : Prop :=
  ∀ {J O : Type} (f : G.Mc.Alg (Orc × J) O) (o : Orc) (j : J), ∃ S : Finset X, (S.card : ℝ) ≤ G.calls f (o, j) ∧
    ∀ o', (∀ y ∈ S, eo o' y = eo o y) → G.Mc.eval f (o', j) = G.Mc.eval f (o, j)

/-- **A run's cost at the prices `μ`**: `Σ_r μ_r u_r`, `μ_r` the price of a unit of resource `r`. At `μ_r = λ_r / ρ_r`
it is the device-seconds the run takes at the price vector `λ`; at a point mass it is one resource's busy time. -/
noncomputable def cost (μ : G.Res → ℝ) {I O : Type} (f : G.Mc.Alg I O) (x : I) : ℝ := ∑ r, μ r * G.use f x r

/-- **An oracle call's priced cost** at the prices `μ`: `shaPub_μ = Σ_r μ_r callUse_r`, the public implementation's
cost a call. -/
noncomputable def shaPub (μ : G.Res → ℝ) : ℝ := ∑ r, μ r * G.callUse r

end Accel

end Model

section Round

/-- **A round of a commit-then-check protocol**: an instance registered before the period (`Inst`: the workload's
operands and the weight-side coins it reuses), the round's first live coin `Draw` (σ₁, the noise), the
words the prover computes after it `Tr`, their commitment (in `Root`), the check's live coin `Chk` (σ₂), whether the
check passes on the words (`ok`), and whether the checked cells of the words open under a root (`opens`). The prover
sends the commitment before σ₂ is drawn and answers the check with `Ans` (the opened words and their openings);
`accept` is what the verifier checks of the answer against the commitment, and `ok` and `opens` what the analysis reads
of the words the commitment binds (`ExtractAdv` links them).
* `opens o r c T`: the cells the check coin `c` selects open under the root `r` at `T`'s words, by the verifier's own
  opening check under the oracle. It is a property of the oracle, the root, the coin and the words, never of anything
  the prover supplies (condition 2).
* `commit o T` is the honest commitment, the Merkle root over `T`'s leaves; no premise asks a prover to compute it for
  every `T`.
* The round's check may select no cell: a round is checked independently with a live Bernoulli coin inside `Chk`, at a
  probability set from its public shape, and `ok` and `opens` hold where it isn't selected.

The commitment's hash is the random oracle `Orc` (a uniform table: SHA-512 as a random oracle, a proposed
assumption), which `commit`, `opens` and `accept` read. The proof layer (C-Flock's proofs of the opened words) is the
service's, not the round's. -/
structure RoundProto where
  Inst : Type
  Draw : Type
  Chk : Type
  [instDraw : Fintype Draw]
  [instChk : Fintype Chk]
  Orc : Type
  [instOrc : Fintype Orc]
  Tr : Type
  Root : Type
  commit : Orc → Tr → Root
  ok : Inst → Draw → Chk → Tr → Prop
  opens : Orc → Root → Chk → Tr → Prop
  Ans : Type
  accept : Orc → Inst → Draw → Chk → Root → Ans → Prop

attribute [instance] RoundProto.instDraw RoundProto.instChk RoundProto.instOrc

/-- A word: integers mod `2^32`. -/
abbrev Word : Type := ZMod (2 ^ 32)

/-- **The no-fold protocol's knobs** (PROTO-DESIGN): `A` is `n × m`, `B` is `m × n'` with `m = nb · L`,
`L` the segment length; the noise (`Draw`, and `noise`, the noised operands `(Ã, B̃)` from the committed operands and
the noise: HYB noise in protocol v1, whatever v2 freezes); the check's coin `Chk` and the `s` words it selects
(`cells`: the frozen protocol's stratified draw, any width, no power-of-two size); `merkle`, the commitment to the
words: each word a Merkle leaf (hm96 rows in frame-v3-sha512 trees), no fold, under the oracle `Orc` that stands for
SHA-512; and `opens`, the verifier's check under the oracle that a word with its authentication path `Path` opens a
root at a cell. -/
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
  Orc : Type
  [instOrc : Fintype Orc]
  Root : Type
  merkle : Orc → (Fin n → Fin n' → Fin nb → Word) → Root
  Path : Type
  opens : Orc → Root → Fin n × Fin n' × Fin nb → Word → Path → Prop

attribute [instance] NoFold.instDraw NoFold.instChk NoFold.instOrc

/-- **A no-fold word** at cell `(i, j)` and segment `b`: the noised product's prefix sum at the end of segment `b`,
`Σ_{t < (b+1) L} Ã_it B̃_tj` mod `2^32`. -/
def noFoldWord (k : NoFold) (ab : (Fin k.n → Fin (k.nb * k.L) → Word) × (Fin (k.nb * k.L) → Fin k.n' → Word))
    (i : Fin k.n) (j : Fin k.n') (b : Fin k.nb) : Word :=
  ∑ t : Fin (k.nb * k.L), if t.val < (b.val + 1) * k.L then ab.1 i t * ab.2 t j else 0

/-- **The no-fold round at its knobs**: the instance is the committed operands `(A, B)`, the first coin the noise, the
words every segment prefix sum of the noised product (`n n' nb` words), committed by `merkle`, and the check the `s`
cells and segments its coin selects, passing where every checked word is the noised product's, and opening where
every checked word has a path that opens the root at its cell. The answer is each checked word with its path,
accepted where every path opens the root and every word is the noised product's. -/
def noFoldRound (k : NoFold) : RoundProto where
  Inst := (Fin k.n → Fin (k.nb * k.L) → Word) × (Fin (k.nb * k.L) → Fin k.n' → Word)
  Draw := k.Draw
  Chk := k.Chk
  Orc := k.Orc
  Tr := Fin k.n → Fin k.n' → Fin k.nb → Word
  Root := k.Root
  commit := k.merkle
  ok a d c T := ∀ q, T (k.cells c q).1 (k.cells c q).2.1 (k.cells c q).2.2 =
    noFoldWord k (k.noise a d) (k.cells c q).1 (k.cells c q).2.1 (k.cells c q).2.2
  opens o r c T := ∀ q, ∃ p, k.opens o r (k.cells c q) (T (k.cells c q).1 (k.cells c q).2.1 (k.cells c q).2.2) p
  Ans := Fin k.s → Word × k.Path
  accept o a d c r y := ∀ q, k.opens o r (k.cells c q) (y q).1 (y q).2 ∧
    (y q).1 = noFoldWord k (k.noise a d) (k.cells c q).1 (k.cells c q).2.1 (k.cells c q).2.2

/-- **The honest words**: every segment prefix sum of the noised product. -/
def noFoldHonest (k : NoFold) (a : (noFoldRound k).Inst) (d : (noFoldRound k).Draw) : (noFoldRound k).Tr :=
  fun i j b => noFoldWord k (k.noise a d) i j b

end Round

section Moments

/-- The uniform average of `f`. -/
noncomputable def avg {Ω : Type} [Fintype Ω] (f : Ω → ℝ) : ℝ := (∑ ω, f ω) / Fintype.card Ω

/-- **A shortfall** against a public best: `(pub − cost)₊`, how far a program's priced cost falls short of the public
best's `pub`, in the same units (the prices `μ`). -/
noncomputable def roundShortfall (pub cost : ℝ) : ℝ := max 0 (pub - cost)

open Classical in
/-- **A words program's moment** (SAK(matmul)'s): for a words program `Mw` run on advice `s` fixed before the round's
coins (whatever the prover carried into the round) and the round's dither, the average over the round's coins of
`e^{η · (pub − cost_μ)₊}` where the dither is outside the degenerate set `Deg` and the check passes on the words it
computes, `0` elsewhere. `η` is a rate per unit of priced cost. -/
noncomputable def wordsMoment (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (pub η : ℝ) (a : P.Inst) {St : Type}
    (s : St) (Mw : G.Mc.Alg (St × P.Draw) P.Tr) (Deg : P.Draw → Prop) : ℝ :=
  avg fun x : P.Draw × P.Chk =>
    if ¬ Deg x.1 ∧ P.ok a x.1 x.2 (G.Mc.eval Mw (s, x.1)) then
      Real.exp (η * roundShortfall pub (G.cost μ Mw (s, x.1))) else 0

open Classical in
/-- **A commitment program's moment** (SHA-512 speed's, in sampling form; W1): at the oracle `o`, for a commitment
program `Mh` on the advice, the dither and the words of a words program `Mw` (its output is the prover's root), the
average over the round's coins of `e^{η · (pub − cost_μ(Mh))₊}` where the dither is outside `Deg`, the check passes on
`Mw`'s words and the checked cells open under `Mh`'s root (`RoundProto.opens`), `0` elsewhere. A program that hashes
part of the tree is caught where the check selects a cell outside that part. -/
noncomputable def commitMoment (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (pub η : ℝ) (o : P.Orc) (a : P.Inst)
    {St : Type} (s : St) (Mw : G.Mc.Alg (St × P.Draw) P.Tr) (Mh : G.Mc.Alg (P.Orc × (St × P.Draw) × P.Tr) P.Root)
    (Deg : P.Draw → Prop) : ℝ :=
  avg fun x : P.Draw × P.Chk =>
    if ¬ Deg x.1 ∧ P.ok a x.1 x.2 (G.Mc.eval Mw (s, x.1)) ∧
        P.opens o (G.Mc.eval Mh (o, (s, x.1), G.Mc.eval Mw (s, x.1))) x.2 (G.Mc.eval Mw (s, x.1)) then
      Real.exp (η * roundShortfall pub (G.cost μ Mh (o, (s, x.1), G.Mc.eval Mw (s, x.1)))) else 0

end Moments

section Prover

/-- **The prover of a monitoring period of `nsb` sampling blocks**, one sampling block at a time. The sampling
blocks' instances are registered before the period (`reg : Fin nsb → P.Inst`, an argument of what reads the prover),
so the prover never chooses one. It keeps a state `St` across sampling blocks,
starting from `start`, fixed before the period. Every program reads the oracle `P.Orc` (the block's hash),
so its queries are inside priced programs. The challenger issues every live coin. Sampling block `b`, in the order of
its coins:

* the challenger issues the block's dither coins `d`; `work b` is everything the prover computes between them and the
  block's commitment, and returns the commitment and the state the next block starts from;
* the challenger issues the block's check coin `c` (whether the block is selected, and which of its proof blocks a
  selected block proves), and `absorb b` writes it into the state. `absorb` is a function, not a program: it reads no
  oracle and uses nothing. Whatever it computes is advice to later blocks' work, and `SAKMatmul` and `ShaSpeed` hold
  at every advice of the size `ExtractAdv` gives, so it buys the prover nothing they don't already allow;
* `answer b` answers the check from the state after the block's work and every coin of the period. It is off
  the state path, since no later program reads it, so it may run beside later sampling blocks and arrive after their
  coins. An answer only opens committed values, which binding fixes (`ExtractAdv`), so what it reads changes only
  whether it is accepted.

**A sampling block is the set of calls whose coins go out together**: block `b + 1`'s coins go out
only after every root of block `b` is in. Calls that overlap in time but whose coins went out separately are separate
sampling blocks, in the order their coins went out. A block's work sees the earlier blocks' coins only through the
state, so nothing before block `b`'s commitment depends on its check coin or on any later coin.

**The challenger times the work** (`timedUse`, `calls`): each block's work runs from its dither coins to its
commitment, inside the period on its devices, one block at a time. Only selected blocks are answered, beside later
blocks. No later program reads an answer, and an honest answer's cost is in the honest cost's `W_ans` (`WorkSound`'s
`H_o`), which the headline's `ε` charges at full weight, so no answer's use or calls are credited or counted against
the period. Its exchanged bits are (`ext`): an answer that runs beside later blocks holds their coins, and what it
sends out crosses the boundary the network certifier watches.

The sampling blocks of a period are one sequence in the order their coins went out: a fleet whose replicas' coins
aren't serialized is one prover per replica (the fleet's bound is
`Definitions.Pouw.Guarantees.Dither.FleetSound`). -/
structure Prover (G : Accel) (P : RoundProto) (nsb : ℕ) where
  St : Type
  start : St
  work : Fin nsb → G.Mc.Alg (P.Orc × St × P.Draw) (St × P.Root)
  absorb : Fin nsb → St → P.Chk → St
  answer : Fin nsb → G.Mc.Alg (P.Orc × St × (Fin nsb → P.Draw × P.Chk)) P.Ans

namespace Prover

variable {G : Accel} {P : RoundProto} {k : ℕ} (A : Prover G P k)

/-- The state at the start of sampling block `b` (it stays put past the last block), at the oracle `o` and the
verifier's coins `x`: the previous block's work's, with its check coin absorbed. No answer is on this path. -/
def state (o : P.Orc) (x : Fin k → P.Draw × P.Chk) : ℕ → A.St
  | 0 => A.start
  | b + 1 =>
    if h : b < k then
      A.absorb ⟨b, h⟩ (G.Mc.eval (A.work ⟨b, h⟩) (o, state o x b, (x ⟨b, h⟩).1)).1 (x ⟨b, h⟩).2
    else state o x b

/-- Round `w`'s state with its dither coins. -/
def wordsIn (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) : A.St × P.Draw :=
  (A.state o x w, (x w).1)

/-- Round `w`'s input to `work`: the oracle, the state at the round's start, and its dither coins. -/
def workIn (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) : P.Orc × A.St × P.Draw :=
  (o, A.wordsIn o x w)

/-- Sampling block `b`'s input to `answer`: the oracle, the state after its work, and every coin of the period, later
blocks' among them. -/
def answerIn (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (b : Fin k) : P.Orc × A.St × (Fin k → P.Draw × P.Chk) :=
  (o, (G.Mc.eval (A.work b) (A.workIn o x b)).1, x)

/-- Round `w`'s commitment. -/
def root (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) : P.Root := (G.Mc.eval (A.work w) (A.workIn o x w)).2

/-- Sampling block `b`'s answer. -/
def ans (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (b : Fin k) : P.Ans :=
  G.Mc.eval (A.answer b) (A.answerIn o x b)

/-- **The transcript**: every round's commitment and answer, what the prover sends. -/
def transcript (o : P.Orc) (x : Fin k → P.Draw × P.Chk) : Fin k → P.Root × P.Ans :=
  fun w => (A.root o x w, A.ans o x w)

/-- **Accepted by the rounds' checks** on the registered instances `reg`: every round's answer passes the verifier's
check against its commitment. -/
def Accepted (reg : Fin k → P.Inst) (o : P.Orc) (x : Fin k → P.Draw × P.Chk) : Prop :=
  ∀ w, P.accept o (reg w) (x w).1 (x w).2 (A.root o x w) (A.ans o x w)

/-- **The sampling blocks' timed use** of resource `r`: every block's `work` at its input, from its dither coins to
its commitment, the phase the challenger times, one block at a time. An answer's use isn't counted: no
later program reads it, and the honest answers' cost is charged in `ε` through the honest cost's `W_ans`. -/
noncomputable def timedUse (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (r : G.Res) : ℝ :=
  ∑ b, G.use (A.work b) (A.workIn o x b) r

/-- **The sampling blocks' oracle calls**: every block's `work` at its input, the calls on the state
path that reaches the roots. On a run whose timed use fits the period's devices, (G3) caps them at a resource's
capacity over a call's use of it. -/
noncomputable def calls (o : P.Orc) (x : Fin k → P.Draw × P.Chk) : ℝ :=
  ∑ b, G.calls (A.work b) (A.workIn o x b)

/-- **The sampling blocks' exchanged bits** (`Accel.ext`): every block's `work` and `answer` at their inputs. An
answer reads every coin of the period, so it could send a later block's coins out, and an outside machine
could return that block's root to its work; the bits are counted here whichever program sends them. An
unselected block's answer never runs, and the prover that stands for a node answers it with a constant program, which
exchanges nothing. -/
noncomputable def ext (o : P.Orc) (x : Fin k → P.Draw × P.Chk) : ℝ :=
  ∑ b, (G.ext (A.work b) (A.workIn o x b) + G.ext (A.answer b) (A.answerIn o x b))

end Prover

/-- **The service's verdict on a period**: what the verifier accepts, from the oracle, the verifier's
coins, the transcript (`Prover.transcript`) and the readings `Obs` of the auditor's devices (the network certifier's
signed counts). It reads no code: the rounds' checks run inside C-Flock's zero-knowledge proofs (`ServiceSound` ties
it to `Prover.Accepted`). -/
abbrev Verdict (P : RoundProto) (k : ℕ) (Obs : Type) : Type :=
  P.Orc → (Fin k → P.Draw × P.Chk) → (Fin k → P.Root × P.Ans) → Obs → Prop

/-- **The auditor's devices' readings** of a prover's period (the network certifier's counts), at the oracle and the
coins. They are the devices' and depend on what the prover's programs exchange (`ImportWithinCertifier`). -/
abbrev Readings (G : Accel) (P : RoundProto) (k : ℕ) (Obs : Type) : Type 1 :=
  Prover G P k → P.Orc → (Fin k → P.Draw × P.Chk) → Obs

/-- **The service accepts the period**: the verdict on the prover's transcript and the devices' readings. -/
def Prover.Passes {G : Accel} {P : RoundProto} {k : ℕ} {Obs : Type} (A : Prover G P k) (verdict : Verdict P k Obs)
    (dev : Readings G P k Obs) (o : P.Orc) (x : Fin k → P.Draw × P.Chk) : Prop :=
  verdict o x (A.transcript o x) (dev A o x)

/-- **The headline's error over a monitoring period of `k` rounds**, round `w` at its registered instance's honest
cost `H w` (its matmul, hashing and other parts at the dual prices) and public bests `pubmm w` and
`pubsha w`:
`ε = (∑_w (H_w − pub_mm,w − pub_sha,w) + ln(1/δ)/η + β K) / ∑_w H_w`.
The terms: the model's slack against the public bests and the honest work that is neither (no lower bound, full
weight), the sampling luck at confidence `δ` and rate `η` a unit of priced cost (P15/P20: `η = k_draw/C`, so the term
is `ln(1/δ)/k_draw`), and the exchanged bits (`K` at most, the network certifier's count, each worth `β`). -/
noncomputable def epsWorkSound {k : ℕ} (H pubmm pubsha : Fin k → ℝ) (η δ β K : ℝ) : ℝ :=
  (∑ w, (H w - pubmm w - pubsha w) + Real.log (1 / δ) / η + β * K) / ∑ w, H w

end Prover

section Freed

/-- **A monitoring period's devices and the other work they could do**, in resource units: the period's length `T`
(seconds), the other jobs `J`, each with its value `val` per unit and its use `needs` per unit, and the honest use
`hon` of the credited work and its value `rv` (the credit, in the jobs' units: its honest cost at the dual
price). `WorkSound` reads it at `W = 1` credited unit: `hon` is the registered sampling blocks' total
honest use, their work and the selected blocks' answers beside it, whose blocks differ, and `rv` their credit. -/
structure Period (G : Accel) where
  T : ℝ
  J : Type
  [instJ : Fintype J]
  val : J → ℝ
  needs : J → G.Res → ℝ
  hon : G.Res → ℝ
  rv : ℝ

attribute [instance] Period.instJ

namespace Period

variable {G : Accel} (Pd : Period G)

/-- What the devices can do in the period (G2): `ρ_r T` of resource `r`. -/
def cap (r : G.Res) : ℝ := G.rate r * Pd.T

/-- A strategy fits the devices: its credited part (its sampling blocks' timed work, `Prover.timedUse`) uses `X`,
its other work `z ≥ 0` units of each job, and the two together use no resource past its capacity. -/
def Fits (X : G.Res → ℝ) (z : Pd.J → ℝ) : Prop := (∀ j, 0 ≤ z j) ∧ ∀ r, X r + ∑ j, z j * Pd.needs j r ≤ Pd.cap r

/-- A strategy's value: its `W` credited rounds at `rv` each, plus its other work's. -/
def value (W : ℝ) (z : Pd.J → ℝ) : ℝ := W * Pd.rv + ∑ j, Pd.val j * z j

/-- **Freed compute is at most `ε C`**: a strategy credited `W` with other work `z` gets
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

/-- **Freed compute at most `ε C`, prover by prover** (`Period.FreedWithin`'s probability form): for every prover, the
probability over the oracle and the verifier's coins that the service accepts its period (`Prover.Passes`, on its
transcript and the devices' readings `dev`) and some other work `z` fits on the period's devices beside its sampling
blocks' timed work (`Prover.timedUse`) with `W rv + W_other(z) > OPT_honest + ε C`
(`¬ Period.FreedWithin`) is at most
`κ`. `W` is the credited work. `ε` and `κ` are the prover's own: a prover whose commitment path makes more queries
has a larger walk-and-log cost and extraction error (`ExtractAdv`). Wrap corrections and peels (BET-HYBRID-2) are
serving work outside the words: they are other work here, never credit, so skipping them isn't a soundness
question. -/
def FreedSound (G : Accel) (P : RoundProto) (k : ℕ) {Obs : Type} (verdict : Verdict P k Obs)
    (dev : Readings G P k Obs) (Pd : Period G) (W : ℝ) (ε : Prover G P k → ℝ) (C : ℝ) (κ : Prover G P k → ℝ) :
    Prop :=
  ∀ A : Prover G P k, prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => A.Passes verdict dev ox.1 ox.2 ∧
    ∃ z, Pd.Fits (A.timedUse ox.1 ox.2) z ∧ ¬ Pd.FreedWithin W z (ε A) C) ≤ κ A

end Freed

end Pouw.Dither
