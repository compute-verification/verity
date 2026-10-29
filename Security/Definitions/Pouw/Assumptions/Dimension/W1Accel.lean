import Definitions.Core.Accel.H200
import Definitions.Core.Assumptions.Accel

/-!
# W1 as a refinement of the accelerator model (trusted statements; draft semantics)

**Draft semantics, pending review**.

`Assumptions.CardWithinAccel` carries the card to its accelerator model in time: a code the card runs in `t` ticks is a
program of `c.machine x` whose schedule has at most `t` ticks. W1 (`Pouw.Dimension.H100W`) is a different reading,
work: the sum over the instructions a program writes of each one's price per 32-bit register it writes. This file states
W1 over the same model, so a work bound on the card goes through one premise of `CardWithinAccel`'s shape, and says
what the time reading gives instead.

* **Writes** (`outWords`, `writes`): the words a run's instructions may write, each opcode's `nOut` (a branch none),
  every tick's together. A run is data-dependent: its branches read registers and its operands may be addresses a
  program computed (`Machine.Operand.indirect`), so which locations a run writes depends on the content.
* **`FreshWrites`** (any accelerator, proved in `Proofs.Pouw.Dimension.W1AccelProofs`): over a grid of initial states,
  the (location, grid point) pairs whose word a run changes are at most the words its instructions write, all grid
  points together. It needs nothing of the program: it holds with data-dependent control and addresses.
* **`CardWithinAccelW1`** (the work premise, `card-within-accel-w1(card, cw)`): a code the card runs at W1 work at most
  `w` on each content is a program of the model whose run writes at most `w / cw` words. At `h200Measured` every class
  is at least 32 a 32-bit register, and so is every load, atomic and reduction form measured on the H200 (the
  `sm90_w1.cu` probes), so `cw = 32` is the measured reading.
* **`W1FreshWords`** (proved from the premise): a code at W1 work at most `w` a content changes, on average over any
  grid of contents, at most `w / cw` words outside the register file. With every target word changed on at least a
  `1 − ε` share of the grid, that is `cw·(1 − ε)` a word.
* **The time reading** (`W1TimeH200`, `H200FreshTick`): one tick of the H200's instance fits 8448 `red.v8` from
  registers into L2 by `iadd3` and 8448 DFMAs, 84,480 words that aren't copies, against 540,672 tensor units a tick: so
  no time bound at more than 32/5 units a fresh word follows from `CardWithinAccel` on this instance
  (`w1TimeH200_le`). The instance's L2 rate is the lane loads' issue bound, a placeholder (`Accel.H200`'s docstring).

**Where `MH100w` differs.** `MH100w` is straight-line and oblivious: its `move` copies a fixed register for free, and a
target only has to differ from every free value (`DistinctOn`). The model's runs branch on data and store to computed
addresses. One store to a computed address changes the word of a different location at each grid point, so the
locations' contents as functions of the grid are as many new functions as there are addresses, from one write a grid
point; `DistinctOn` admits them all as targets. Here a target is a location whose word a run changes on a share of the
grid, and the count is of writes, moves and loads included: what a free move could make up, a grid point at a time,
is a write the premise prices.
-/

open scoped NNRat

namespace Pouw.Dimension.W1Accel

open Verity Verity.Accel

variable {ι : Type}

/-- The words an instruction may write: an opcode's outputs, a branch's none. -/
def outWords {m : Machine ι} : m.Instr → ℕ
  | .op o _ _ _ => m.nOut o
  | .bra _ _ => 0

/-- The words a run's instructions may write, every tick's together. -/
def writes {m : Machine ι} (iss : List (List m.Instr)) : ℕ := (iss.flatten.map outWords).sum

/-- **Fresh writes need writes** (any accelerator, program and schedule; data-dependent branches and addresses
included): for initial states `σ g` at the grid points `g ∈ S` and locations `Ls`, the pairs `(l, g)` whose word the
run from `σ g` changes number at most the words the runs write, all grid points together. -/
def FreshWrites : Prop :=
  ∀ (ι : Type) (m : Machine ι) (Θ : Type) [DecidableEq Θ] (G : Type) (P : m.Prog Θ) (sched : List (List Θ))
    (S : Finset G) (σ : G → m.State Θ) (Ls : Finset m.Loc),
    ∑ l ∈ Ls, (S.filter fun g => (Machine.run P (σ g) sched).1.mem l ≠ (σ g).mem l).card ≤
      ∑ g ∈ S, writes (Machine.run P (σ g) sched).2

/-- **`card-within-accel-w1(card, cw)`** (the work premise, draft): the card at W1 work is no more than its accelerator
model writing words at `cw` a word. `canW f w D` says the card, running one code, takes every content `M` with `D M` to
`f M` at W1 work at most `w` on each: the sum, over the instructions it executes on `M`, of each one's measured price per
32-bit register or memory word it writes (`h200Measured`'s classes, and a form outside them at its own measured row).
Then some program of `c.machine x`, at open parameters meeting θ's floors with the product form, under an encoding `e`
of the card's register words chosen before the content, takes `decode e M` to `decode e (f M)` on every such `M` in a run
that writes at most `w / cw` words.

It is `Assumptions.CardWithinAccel` with work for time: the schedule's length is unconstrained, and the run's writes
are bounded instead. Its falsification is a code the card runs whose W1 work is below `cw` times the words some
model program of it must write, which a census row below `cw` a word is: every form the card's compiled code holds,
priced per word it writes on the card (the `sm90_w1.cu` rows and CORE-H32's), each at least `cw`. -/
def CardWithinAccelW1 (c : Accel.Card) (cap : ι → ℕ) (oracle : ℕ) (cw : ℚ≥0)
    (canW : (c.Mem → c.Mem) → ℚ≥0 → (c.Mem → Prop) → Prop) : Prop :=
  ∃ x : Accel.Open, c.floors x ∧ Assumptions.AccelProductForm x ∧
    ∀ f w D, canW f w D → ∃ (e : ℕ → Accel.Enc) (P : (c.machine x cap oracle).Prog ℕ) (sched : List (List ℕ)),
      ∀ (M : c.Mem) (σ : (c.machine x cap oracle).State ℕ), D M → (∀ θ, σ.pc θ = 0) → σ.mem = c.decode e M →
        (Accel.Machine.run P σ sched).1.mem = c.decode e (f M) ∧
          cw * (writes (Accel.Machine.run P σ sched).2 : ℚ≥0) ≤ w

/-- **W1's fresh-words bound on the card** (draft semantics): under `CardWithinAccelW1 c cap oracle cw canW`, a code the
card runs at W1 work at most `w` a content in `D`, over contents `M g` in `D` at the grid points `g ∈ S`, changes at
most `|S|·w / cw` (location, grid point) pairs outside the register file: for every set `Ls` of locations of shared
memory, L2 or device memory, `cw · Σ_{l ∈ Ls} #{g ∈ S : f (M g) l ≠ M g l} ≤ |S| · w`. -/
def W1FreshWords : Prop :=
  ∀ (ι : Type) (c : Accel.Card) (cap : ι → ℕ) (oracle : ℕ) (cw : ℚ≥0)
    (canW : (c.Mem → c.Mem) → ℚ≥0 → (c.Mem → Prop) → Prop),
    CardWithinAccelW1 c cap oracle cw canW →
    ∀ f w D, canW f w D → ∀ (G : Type) (S : Finset G) (M : G → c.Mem), (∀ g ∈ S, D (M g)) →
      ∀ Ls : Finset (Σ s : Store, Fin (c.size s)), (∀ l ∈ Ls, l.1 ≠ .reg) →
        cw * ((∑ l ∈ Ls, (S.filter fun g => f (M g) l ≠ M g l).card : ℕ) : ℚ≥0) ≤ S.card * w

/-! ## The time reading on the H200's instance -/

/-- The words an opcode writes that aren't copies of its inputs: an MMA's accumulators, a lane or uniform function's
outputs (a shuffle's none), an atomic add's new word, a reduction's words; a copy's, a broadcast's and an exchange's
none. -/
def fresh : Op Store → ℕ
  | .mma _ _ _ sh _ _ | .mmaSp _ _ _ sh _ _ => sh.m * sh.n
  | .lane g | .uni g => if g = .shfl then 0 else g.nOut
  | .copy .. | .bcast .. => 0
  | .atom _ .add => 1
  | .atom _ .exch => 0
  | .red _ _ n _ => n
  | .dmma sh => 2 * sh.m * sh.n

/-- The units opcode `o` uses on the H200's instance at open parameters `x` (`Card.machine`'s `use`). -/
def h200Use (x : Open) (o : Op Store) (r : Res) : ℚ≥0 :=
  opUse (H200.card.weightOf x) (H200.card.scale x) x.madHi o r

/-- **W1 as a time bound on the H200's instance at `c` units a fresh word**: at every open parameters meeting θ's floors,
the opcodes of any list that fits one tick's units of every resource write at most `ρ tensor / c` fresh words. -/
def W1TimeH200 (c : ℚ≥0) : Prop :=
  ∀ x : Open, H200.card.floors x → ∀ ops : List (Op Store), (∀ o ∈ ops, H200.Avail o) →
    (∀ r, (ops.map fun o => h200Use x o r).sum ≤ H200.ρ r) → c * ((ops.map fresh).sum : ℚ≥0) ≤ H200.ρ .tensor

/-- **One tick of the H200's instance writes 84,480 fresh words**: at every open parameters, 8448 `red.v8` from
registers into L2 by `iadd3` and 8448 DFMAs are available, fit one tick's units of every resource (the issue slots and
the L2's and FP64 pipe's units exactly), and write 8 and 2 fresh words each. -/
def H200FreshTick : Prop :=
  ∀ x : Open,
    let ops := List.replicate 8448 (Op.red Store.reg Store.l2 8 Lane.iadd3) ++ List.replicate 8448 (Op.lane Lane.dfma)
    (∀ o ∈ ops, H200.Avail o) ∧ (∀ r, (ops.map fun o => h200Use x o r).sum ≤ H200.ρ r) ∧ (ops.map fresh).sum = 84480

/-- **The time reading is at most 32/5 a fresh word**: `W1TimeH200 c` implies `c ≤ 32/5` (540,672 tensor units a tick
over 84,480 words). So W1 at 32 a word is not a consequence of `CardWithinAccel` on this instance. -/
def W1TimeH200Le : Prop := ∀ c : ℚ≥0, W1TimeH200 c → c ≤ 32 / 5

end Pouw.Dimension.W1Accel
