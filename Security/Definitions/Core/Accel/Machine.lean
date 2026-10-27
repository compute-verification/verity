import Definitions.Accounting.Device
import Mathlib.Algebra.Field.Rat

/-!
# The accelerator model (trusted)

An accelerator is \(\mathcal G = (\Sigma, M, R, \rho, \mathcal I, u)\): stores \(\Sigma\) of \(M_s\) words each,
resources \(R\) of \(\rho_r\) units a tick each, and an instruction set \(\mathcal I\) whose every opcode has exact
semantics and uses \(u_r(o) \ge 0\) units of each resource. `Machine` is that tuple on an `Accounting.Device`, whose
coarse limits (on-chip bits, device-memory bits, op rates, bandwidth, side channels) it refines. The model states two
things about the card and nothing about algorithms:

* **(G1) Capacity.** Store `s` is `size s` words (`Loc`): a run holds at most \(M_s\) words in it at every tick because
  there are no other locations. Code is data too: it lives in device memory, so the texts of all the threads a run
  steps hold at most `text` instructions together (`Prog.Fits`), device memory's bytes at 16 bytes an instruction.
* **(G2) Rate.** Over every window of a run, the instructions issued use at most \(\rho_r\) units of each resource `r`
  a tick, plus a backlog of `slack` ticks (`Rate`). The code reaches the SMs from device memory, so the run's
  instructions and its code, fetched once (`fetchUse` units of `fetch` an instruction), share that path (`CodeFetch`).

Everything else is the instruction set's semantics: an opcode reads its input words from the stores `inStore` names
and writes `eval` of them to the stores `outStore` names, and nothing else moves a word. So a product an opcode forms
reaches another word only through further instructions, each of which pays its usage (P2 in `Accel.Principles`).

**Execution.** A program is a text per thread (`Prog`); a thread has a program counter and runs its text in order
(`step`), and the threads share the stores. A branch (`Instr.bra`) is the one data-dependent control: it reads a
register and sets the thread's counter. A run's schedule is the list, per tick, of the threads stepped in it, in order
(`run`). A thread past the end of its text does nothing and uses nothing.

**The schedule costs the adversary nothing.** A run is a function of its initial state and its schedule, and every
statement over the model takes the schedule after the state (the adversary chooses both). A card's data-dependent
distribution of work, atomics-based work stealing included, is therefore a schedule the adversary could have computed
from the state: it is one of the schedules the statement already ranges over, at the same instructions and so the same
units. Data the run receives later (a verifier's coins) starts a new run from the state it is written into, with a new
schedule. An address operand (`Operand.indirect`) is a register's value plus an offset, as a load, store or atomic
takes one, and an operand may be an immediate the text carries.

**Time.** A tick is the device's unit of time (an instance takes the SM clock at its maximum, so a tick is at least one
clock), as in `Accounting.Window`. Instructions take effect at issue: the model has no latency, which only helps a
program.

**Where the model is coarser than the card.** Every coarsening lets a program do more than the card, so a lower bound in
the model holds on the card:
* stores are pooled over the device (one register file, one shared memory), and any thread may name any location;
* a resource is pooled over the device's SMs, and the per-tick bound is on the whole device;
* a resource with no documented rate (the register file's ports, L2's bandwidth) is not a resource here: it is free;
* a thread's text bound is per thread, not over the card's whole program text.
-/

open scoped NNRat

namespace Verity.Accel

/-- A word: the stores hold 32-bit words, as a GPU's registers, shared memory and device memory do. -/
abbrev Word : Type := BitVec 32

/-- **An accelerator** \((\Sigma, M, R, \rho, \mathcal I, u)\) refining the device `toDevice`. -/
structure Machine (ι : Type) extends Accounting.Device ι where
  /-- the stores Σ -/
  Store : Type
  [decStore : DecidableEq Store]
  /-- `M_s`: the words store `s` holds -/
  size : Store → ℕ
  size_pos : ∀ s, 0 < size s
  /-- the register store: a branch's predicate and an address operand's index are read from it -/
  reg : Store
  /-- the resources `R` -/
  Res : Type
  /-- `ρ_r`: units of resource `r` the device supplies a tick -/
  ρ : Res → ℚ≥0
  /-- the backlog: work issued in a window completes within `slack` ticks after it -/
  slack : ℕ
  /-- the opcodes 𝓘 -/
  Op : Type
  /-- an opcode's input words -/
  nIn : Op → ℕ
  /-- an opcode's output words -/
  nOut : Op → ℕ
  /-- the store each input word is read from -/
  inStore : (o : Op) → Fin (nIn o) → Store
  /-- the store each output word is written to -/
  outStore : (o : Op) → Fin (nOut o) → Store
  /-- an opcode's exact semantics: its output words from its input words -/
  eval : (o : Op) → (Fin (nIn o) → Word) → Fin (nOut o) → Word
  /-- `u_r(o)`: the units of resource `r` one issue of opcode `o` uses -/
  use : Op → Res → ℚ≥0
  /-- the units of each resource a branch uses -/
  braUse : Res → ℚ≥0
  /-- the instructions all the texts of a run hold at most, together -/
  text : ℕ
  /-- the resource that carries code to the SMs (device memory's port) -/
  fetch : Res
  /-- its units an instruction of code uses, fetched once -/
  fetchUse : ℚ≥0

attribute [instance] Machine.decStore

variable {ι : Type}

namespace Machine

variable (m : Machine ι)

/-- A location: a store and a word of it. Store `s` has exactly `size s` of them (G1). -/
abbrev Loc : Type := Σ s : m.Store, Fin (m.size s)

/-- The stores' contents: a word at every location. -/
abbrev Mem : Type := m.Loc → Word

/-- Word `i` of store `s`, its index taken modulo the store's size. -/
def loc (s : m.Store) (i : ℕ) : m.Loc := ⟨s, ⟨i % m.size s, Nat.mod_lt _ (m.size_pos s)⟩⟩

/-- An operand: a fixed word of the store its opcode names; the word at a register's value plus an offset (an address
the program computed); or an immediate the text carries (as an output, it discards the word). Indices are taken modulo
the store's size. -/
inductive Operand
  | direct (i : ℕ)
  | indirect (r : ℕ) (off : ℕ)
  | imm (w : Word)

/-- An instruction of a text: an opcode on operands, predicated on a register (it writes only when the register is
nonzero), or a branch to `target` when register `p` is nonzero. -/
inductive Instr
  | op (o : m.Op) (src : Fin (m.nIn o) → Operand) (dst : Fin (m.nOut o) → Operand) (pred : Option ℕ)
  | bra (p : ℕ) (target : ℕ)

/-- A program: each thread's text. -/
abbrev Prog (Θ : Type) : Type := Θ → List m.Instr

/-- A run's state: the stores and each thread's program counter. -/
structure State (Θ : Type) where
  mem : m.Mem
  pc : Θ → ℕ

variable {m}

/-- The location an operand names in store `s` of `M`, or none for an immediate. -/
def Operand.locIn (s : m.Store) : Operand → m.Mem → Option m.Loc
  | .direct i, _ => some (m.loc s i)
  | .indirect r off, M => some (m.loc s ((M (m.loc m.reg r)).toNat + off))
  | .imm _, _ => none

/-- The word an operand reads in store `s` of `M`. -/
def Operand.readIn (s : m.Store) (x : Operand) (M : m.Mem) : Word :=
  match x with
  | .imm w => w
  | _ => match x.locIn s M with
    | some l => M l
    | none => 0

/-- `M` with `w` written where the output operand `x` names in store `s` of `M₀` (the memory before the instruction,
so an address is read before any write). -/
def Operand.writeIn (s : m.Store) (x : Operand) (M₀ M : m.Mem) (w : Word) : m.Mem :=
  match x.locIn s M₀ with
  | some l => Function.update M l w
  | none => M

/-- An opcode's effect on `M`: every input read, then the outputs written in order, unless the predicate is zero. -/
def execOp (o : m.Op) (src : Fin (m.nIn o) → Operand) (dst : Fin (m.nOut o) → Operand) (pred : Option ℕ)
    (M : m.Mem) : m.Mem :=
  let v := m.eval o fun i => (src i).readIn (m.inStore o i) M
  if pred.any fun p => M (m.loc m.reg p) = 0 then M
  else (List.finRange (m.nOut o)).foldl (fun M' i => (dst i).writeIn (m.outStore o i) M M' (v i)) M

/-- The units of resource `r` an instruction uses. -/
def Instr.use : m.Instr → m.Res → ℚ≥0
  | .op o _ _ _ => m.use o
  | .bra _ _ => m.braUse

variable {Θ : Type} [DecidableEq Θ]

/-- Thread `θ` runs the instruction at its counter: the new state, and the instruction (none past its text's end). -/
def step (P : m.Prog Θ) (σ : m.State Θ) (θ : Θ) : m.State Θ × Option m.Instr :=
  match (P θ)[σ.pc θ]? with
  | none => (σ, none)
  | some (.op o src dst pred) =>
      (⟨execOp o src dst pred σ.mem, Function.update σ.pc θ (σ.pc θ + 1)⟩, some (.op o src dst pred))
  | some (.bra p t) =>
      (⟨σ.mem, Function.update σ.pc θ (if σ.mem (m.loc m.reg p) ≠ 0 then t else σ.pc θ + 1)⟩, some (.bra p t))

/-- One tick: the threads `θs` step in order; the state after, and the instructions they ran. -/
def tick (P : m.Prog Θ) (σ : m.State Θ) (θs : List Θ) : m.State Θ × List m.Instr :=
  θs.foldl (fun acc θ => let r := step P acc.1 θ; (r.1, acc.2 ++ r.2.toList)) (σ, [])

/-- **A run** of `P` from `σ` on a schedule (per tick, the threads stepped in it): the final state, and per tick the
instructions run. -/
def run (P : m.Prog Θ) : m.State Θ → List (List Θ) → m.State Θ × List (List m.Instr)
  | σ, [] => (σ, [])
  | σ, θs :: rest =>
      let a := tick P σ θs
      let b := run P a.1 rest
      (b.1, a.2 :: b.2)

/-- The units of resource `r` the instructions run at tick `t` use. -/
def usage (iss : List (List m.Instr)) (t : ℕ) (r : m.Res) : ℚ≥0 :=
  ((iss.getD t []).map fun i => i.use r).sum

/-- **(G2) Rate**: over every window, the instructions run use at most `ρ_r` units of each resource `r` a tick, plus
the backlog. At `slack = 0` and a window of one tick it bounds each tick. -/
def Rate (iss : List (List m.Instr)) : Prop :=
  ∀ (w : Accounting.Window) (r : m.Res),
    ∑ t ∈ w.ticks, usage iss t r ≤ m.ρ r * ((w.len + m.slack : ℕ) : ℚ≥0)

/-- The instructions of code a run on `sched` holds: the texts of the threads it steps, each once. -/
def code (P : m.Prog Θ) (sched : List (List Θ)) : ℕ := (sched.flatten.dedup.map fun θ => (P θ).length).sum

/-- **(G1) Capacity of the code**: the texts of the threads a run steps fit in `text` instructions together. (The
stores' capacities hold by construction: store `s` has exactly `size s` locations.) -/
def Prog.Fits (P : m.Prog Θ) (sched : List (List Θ)) : Prop := code P sched ≤ m.text

/-- **(G2) Fetching the code**: the run's use of `fetch` and its code, fetched once, fit in `fetch`'s units over the
run and its backlog. -/
def CodeFetch (P : m.Prog Θ) (σ : m.State Θ) (sched : List (List Θ)) : Prop :=
  ∑ t ∈ Finset.range sched.length, usage (run P σ sched).2 t m.fetch + m.fetchUse * code P sched ≤
    m.ρ m.fetch * ((sched.length + m.slack : ℕ) : ℚ≥0)

/-- A run of `P` on `sched` from `σ` meets the model: its code fits (G1), its instructions meet the rates and its code
is fetched (G2). Its duration is `sched.length` ticks. -/
def Fits (P : m.Prog Θ) (σ : m.State Θ) (sched : List (List Θ)) : Prop :=
  P.Fits sched ∧ Rate (run P σ sched).2 ∧ CodeFetch P σ sched

end Machine

end Verity.Accel
