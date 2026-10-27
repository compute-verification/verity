import Definitions.Core.Accel.Card

/-!
# The machine 𝓜 (trusted)

The machine at a card's numbers `card` and its semantics `sem`. Store `s` is `size s` words (`Loc`). A program is a
text per thread (`Prog`); a thread runs its text in order (`step`) and the threads share the stores. An instruction
is an ACCEL op on operands, predicated on a register bit, or a branch on one, the one data-dependent control.
A run is a function of its initial state and its schedule, the threads stepped one after another (`run`). The schedule
is fixed with the program, before the content: one chosen after it would be a free lookup table (step, of threads
writing different constants, the one that writes the answer). A card's data-dependent work is the branches its code
takes on words it computed.

A run of `T` seconds meets the model (`Fits`) if

* **(G1) capacity**: the texts of the threads it steps hold at most `text` instructions (code is data in device memory;
  the stores' capacities hold by construction, store `s` has exactly `size s` locations), and
* **(G2) rate**: over the run, its instructions use at most `ρ_r T` units of each resource `r`, device memory's carrying
  the code too, fetched once at 16 bytes an instruction.

The model has no latency, no per-clock limit and no SM boundaries (each resource and store is pooled over the card, and
any thread names any location), and a resource with no stated rate is free: each lets a program do more than the card.
-/

open scoped NNRat

namespace Verity.Accel

/-- **The machine** at a card's numbers and its semantics. -/
structure Machine where
  card : Card
  sem : Open

namespace Machine

variable (m : Machine)

/-- A location: a store and a word of it. -/
abbrev Loc : Type := m.card.Loc

/-- The stores' contents: a word at every location. -/
abbrev Mem : Type := m.card.Mem

/-- The stores' bits. -/
def bits : ℕ := 32 * (m.card.size .reg + m.card.size .smem + m.card.size .l2 + m.card.size .dram)

/-- Word `i` of store `s`, its index taken modulo the store's size. -/
def loc (s : Store) (i : ℕ) : m.Loc := ⟨s, ⟨i % m.card.size s, Nat.mod_lt _ (m.card.size_pos s)⟩⟩

/-- An operand: a fixed word of the store its opcode names; the word at a register's value plus an offset; or an
immediate the text carries (as an output, it discards the word). -/
inductive Operand
  | direct (i : ℕ)
  | indirect (r : ℕ) (off : ℕ)
  | imm (w : Word)

/-- An instruction: an opcode on operands, predicated on bit `b` of register `r` (`pred = some (r, b)`: it writes
only when the bit is set), or a branch, taken when its predicate bit is set (always, with none), to the instruction its
`target` names (a fixed index, a register's value plus an offset, or an immediate), writing the next instruction's index
to register `link` if it names one (a call; a return is a branch to that register). -/
inductive Instr
  | op (o : Op m.sem) (src : Fin o.nIn → Operand) (dst : Fin o.nOut → Operand) (pred : Option (ℕ × ℕ))
  | bra (pred : Option (ℕ × ℕ)) (target : Operand) (link : Option ℕ)

/-- A program: each thread's text. -/
abbrev Prog (Θ : Type) : Type := Θ → List m.Instr

/-- A run's state: the stores and each thread's program counter. -/
structure State (Θ : Type) where
  mem : m.Mem
  pc : Θ → ℕ

variable {m}

/-- Whether bit `p.2` of register `p.1` is set in `M`. -/
def bit (M : m.Mem) (p : ℕ × ℕ) : Bool := (M (m.loc .reg p.1)).getLsbD p.2

/-- The location an operand names in store `s` of `M`, or none for an immediate. -/
def Operand.locIn (s : Store) : Operand → m.Mem → Option m.Loc
  | .direct i, _ => some (m.loc s i)
  | .indirect r off, M => some (m.loc s ((M (m.loc .reg r)).toNat + off))
  | .imm _, _ => none

/-- The word an operand reads in store `s` of `M`. -/
def Operand.readIn (s : Store) (x : Operand) (M : m.Mem) : Word :=
  match x with
  | .imm w => w
  | _ => match x.locIn s M with
    | some l => M l
    | none => 0

/-- The instruction index a branch's target names in `M`. -/
def Operand.target : Operand → m.Mem → ℕ
  | .direct i, _ => i
  | .indirect r off, M => (M (m.loc .reg r)).toNat + off
  | .imm w, _ => w.toNat

/-- `M` with `w` written where the output operand `x` names in store `s` of `M₀` (the memory before the instruction,
so an address is read before any write). -/
def Operand.writeIn (s : Store) (x : Operand) (M₀ M : m.Mem) (w : Word) : m.Mem :=
  match x.locIn s M₀ with
  | some l => Function.update M l w
  | none => M

/-- An opcode's effect on `M`: every input read, then the outputs written in order, unless its predicate bit is clear. -/
def execOp (o : Op m.sem) (src : Fin o.nIn → Operand) (dst : Fin o.nOut → Operand) (pred : Option (ℕ × ℕ))
    (M : m.Mem) : m.Mem :=
  let v := o.eval fun i => (src i).readIn (o.inStore i) M
  if pred.any fun p => !bit M p then M
  else (List.finRange o.nOut).foldl (fun M' i => (dst i).writeIn (o.outStore i) M M' (v i)) M

/-- The units of resource `r` an instruction uses: its opcode's, or a branch's lane share of an issue slot. -/
def Instr.use : m.Instr → Res → ℚ≥0
  | .op o _ _ _ => m.card.use o
  | .bra _ _ _ => slot (1 / 32)

variable {Θ : Type} [DecidableEq Θ]

/-- Thread `θ` runs the instruction at its counter: the new state, and the instruction (none past its text's end). -/
def step (P : m.Prog Θ) (σ : m.State Θ) (θ : Θ) : m.State Θ × Option m.Instr :=
  match (P θ)[σ.pc θ]? with
  | none => (σ, none)
  | some (.op o src dst pred) =>
      (⟨execOp o src dst pred σ.mem, Function.update σ.pc θ (σ.pc θ + 1)⟩, some (.op o src dst pred))
  | some (.bra p t l) =>
      if p.all (bit σ.mem) then
        (⟨l.elim σ.mem fun r => Function.update σ.mem (m.loc .reg r) (BitVec.ofNat 32 (σ.pc θ + 1)),
          Function.update σ.pc θ (t.target σ.mem)⟩, some (.bra p t l))
      else (⟨σ.mem, Function.update σ.pc θ (σ.pc θ + 1)⟩, some (.bra p t l))

/-- **A run** of `P` from `σ` on a schedule (the threads stepped, in order): the final state, and the instructions run. -/
def run (P : m.Prog Θ) : m.State Θ → List Θ → m.State Θ × List m.Instr
  | σ, [] => (σ, [])
  | σ, θ :: rest =>
      let a := step P σ θ
      let b := run P a.1 rest
      (b.1, a.2.toList ++ b.2)

/-- The units of resource `r` a list of instructions uses. -/
def units (iss : List m.Instr) (r : Res) : ℚ≥0 := (iss.map fun i => i.use r).sum

/-- The instructions of code a run on `sched` holds: the texts of the threads it steps, each once. -/
def code (P : m.Prog Θ) (sched : List Θ) : ℕ := (sched.dedup.map fun θ => (P θ).length).sum

/-- **A run of `T` seconds meets the model**: its code fits (G1), and its instructions, with its code fetched once
(four words an instruction), use at most `ρ_r T` of each resource (G2). -/
def Fits (P : m.Prog Θ) (σ : m.State Θ) (sched : List Θ) (T : ℚ≥0) : Prop :=
  code P sched ≤ m.card.text ∧
    ∀ r, units (run P σ sched).2 r + (if r = .dram then 4 * (code P sched : ℚ≥0) else 0) ≤ m.card.ρ r * T

end Machine

end Verity.Accel
