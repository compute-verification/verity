import Definitions.Accounting.JointBlocks
import Definitions.Core.Game

/-!
# A period as a causal game, and the integrity claim read off the commitment (trusted)

Soundness's rulings on the one theorem's form (26 Oct, LEAN-62):

* **R5, a causal game.** A period is one `Verity.Game` (`Definitions.Core.Game`) played after its parameters and its
  commitment: each device release is a `coin` drawn after the bytes it answers (a `send`), in the device's order, and
  the evidence is the last `send`. The game ends in the run `O`, what the verifier of record reads: the bytes sent, the
  coins released and the evidence. A prover's strategy (`Verity.Game.Strategy`) sees each coin only once it is drawn,
  so every byte is a function of the coins released before it, and nothing else.
* **R4, the claim read off the commitment.** The sampled-proofs claim reads no transcript that is a field of the run.
  It reads the openings the runs of the prover's strategy show under the run's commitment (`PeriodLive.opens`): every
  transcript those openings open to (`OpensTo`) has at most an `ε` share of the circuit's units wrong, where a unit
  that reads or writes a committed value with no opening to that transcript's value counts as wrong (`unbound`). Two
  openings of one position to two values (`DoubleOpen`), each shown by a run of the strategy with that commitment, are
  the period's `Break`: a collision along the two openings' paths under one root, from two runs of the strategy sharing
  the commitment (the fork form). Every opening the claim reads is in a run of the strategy, so the break is bounded by
  what the runs carry.

A period's view of the run (`PeriodLive`) is the device model's period (`Accounting.PeriodRun`) with every field read
off the run rather than off a product of coins: its partition, accounts and window, its commitment, the openings it
shows, its schedule, the network certifier's claim, and each subprotocol's verdict. A subprotocol's per-period
guarantee (`PeriodLive.Sound`) bounds its failure over the whole game at every strategy: its coins may come before,
between or after the other subprotocols', and the prover may adapt every byte to every coin released before it. Each
subprotocol proves it from its own game, the prover's moves in it with the other subprotocols' coins fixed at some
values (`Verity.Game.InducedAt`) being a strategy of that game. The joint guarantees are union bounds over the one
game (`Accounting.Guarantees.JointPeriodLive`, `JointHorizonLive`, `JointHorizonBlocksLive`).

The period's game may be a horizon's: one game, several periods, each read off the same run, each period's parameters
and commitment sent after the coins of the periods before it. A period's commitment is the run's, so in a horizon the
runs whose openings a claim reads are those that sent the same commitment.
-/

namespace Accounting

open Verity

section Opened

variable {C : Verity.Circuit} {n : ℕ}

/-- `X` is a transcript the openings `opens` open to: at every position some opening opens, `X` holds an opened
value. -/
def OpensTo (opens : (g : Fin C.N) → BitVec (C.w g) → Prop) (X : C.Transcript) : Prop :=
  ∀ g, (∃ v, opens g v) → opens g (X g)

open Classical in
/-- **The units `X` doesn't bind**: those wrong in `X`, and those that read or write a committed value (`Partition.io`)
with no opening to `X`'s value there. -/
noncomputable def unbound (P : Verity.Partition C n) (opens : (g : Fin C.N) → BitVec (C.w g) → Prop)
    (X : C.Transcript) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ P.Correct X u ∨ ∃ g ∈ P.io u, ¬ opens g (X g)

/-- **Two openings of one position**, to two values. -/
def DoubleOpen (opens : (g : Fin C.N) → BitVec (C.w g) → Prop) : Prop :=
  ∃ g v v', opens g v ∧ opens g v' ∧ v ≠ v'

/-- **The integrity claim read off the commitment**, at share `ε`: every transcript the openings `opens` open to has
at most an `ε` share of `P`'s units unbound. -/
def IntegrityHolds (P : Verity.Partition C n) (opens : (g : Fin C.N) → BitVec (C.w g) → Prop) (ε : ℝ) : Prop :=
  ∀ X, OpensTo opens X → ((unbound P opens X).card : ℝ) ≤ ε * n

end Opened

/-- The runs a strategy reaches: the game's output at some value of its coins, the strategy playing every `send`. -/
def Reaches {O : Type} : (G : Verity.Game O) → Verity.Game.Strategy G → O → Prop
  | .ret a, _, o => o = a
  | .send _ k, s, o => Reaches (k s.1) s.2 o
  | @Game.coin _ _ _ _ k, s, o => ∃ c, Reaches (k c) (s c) o

/-- **A period's view of the run `O` a causal game ends in**, on device `d` with accounts `α`, over the circuit `C`,
with commitments in `Cm`: the partition whose units the integrity claim counts, each resource's account, the period's
window, the run's commitment, the openings it shows, its schedule, the network certifier's claim, and each
subprotocol's verdict. The coins are the run's: nothing here reads a coin the game hasn't drawn. -/
structure PeriodLive {ι : Type} (d : Device ι) (α : Type) [Fintype α] (C : Verity.Circuit) (Cm O : Type) where
  /-- the number of units of the partition -/
  units : ℕ
  /-- the partition of the circuit's computed gates into units -/
  parts : Verity.Partition C units
  /-- the account a resource is charged to -/
  acct : Resource → α
  /-- the period's window -/
  win : Window
  /-- the commitment the run sent for the period -/
  commit : O → Cm
  /-- the run shows an opening of position `g` to `v` that passes the opening check under its commitment -/
  shows : O → (g : Fin C.N) → BitVec (C.w g) → Prop
  /-- the schedule the run realizes -/
  schedule : O → Schedule ι α
  /-- every schedule fits the device -/
  fits : ∀ o, (schedule o).Fits d
  /-- the network certifier's claim at `K`: what the network certifiers saw the node move in the period is within `K`
  bits -/
  commWithin : ℝ → O → Prop
  /-- a subprotocol's verdict on the run -/
  accepts : Audit → O → Prop

namespace PeriodLive

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {Cm O : Type}
  (r : PeriodLive d α C Cm O)

/-- **The openings of the run `o`'s commitment** in the game `G` at the strategy `s`: the values some run of `s` that
sent the same commitment shows at each position. -/
def opens (G : Verity.Game O) (s : Verity.Game.Strategy G) (o : O) (g : Fin C.N) (v : BitVec (C.w g)) : Prop :=
  ∃ o', Reaches G s o' ∧ r.commit o' = r.commit o ∧ r.shows o' g v

/-- **The claim a subprotocol certifies** at certificates `c` and price `p`, on the run `o` of the strategy `s`: the
integrity claim read off the commitment, and the device-model claims read off the run's schedule. -/
def Holds (c : Certificates) (p : Price) (G : Verity.Game O) (s : Verity.Game.Strategy G) : Audit → O → Prop
  | .sampled, o => IntegrityHolds r.parts (r.opens G s o) c.ε
  | .pouw, o => c.W ≤ ((r.schedule o).work r.win (r.acct .compute) p : ℝ)
  | .pous, o => c.S * r.win.len ≤ ((r.schedule o).held r.win (r.acct .capacity) : ℝ)
  | .poub, o => c.Q ≤ ((r.schedule o).read r.win (r.acct .bandwidth) : ℝ)
  | .network, o => r.commWithin c.K o

/-- **The break**: runs of the strategy that sent the run `o`'s commitment open one position to two values. -/
def Break (G : Verity.Game O) (s : Verity.Game.Strategy G) (o : O) : Prop := DoubleOpen (r.opens G s o)

/-- **A subprotocol's per-period guarantee over the causal game**, at certificates `c`, price `p` and error `δ`: at
every strategy, the probability over the coins the game draws that subprotocol `a` accepts while its claim fails,
with no break, is at most `δ`. -/
def Sound (G : Verity.Game O) (c : Certificates) (p : Price) (a : Audit) (δ : ℝ) : Prop :=
  ∀ s : Verity.Game.Strategy G,
    Verity.Game.prob (fun o => r.accepts a o ∧ ¬ r.Holds c p G s a o ∧ ¬ r.Break G s o) G s ≤ ENNReal.ofReal δ

end PeriodLive

/-- The wipe that opens a server-time block, read off the run: the bits of device memory its fill covers, the memory
challenger's verdict, and `kept`, the bits of foreign content beyond the committed carried state that the block can
read back from what the wipe leaves (`Accounting.BlockWipe`'s fields, on the run rather than on a product of coins). -/
structure WipeLive {ι : Type} (d : Device ι) (O : Type) where
  /-- the bits of device memory the fill covers -/
  fill : ℕ
  fill_le : fill ≤ d.mem
  /-- the memory challenger's verdict on the wipe -/
  accepts : O → Prop
  /-- the bits of foreign content beyond the committed carried state that survive the wipe into the block -/
  kept : O → ℕ

/-- **The wipe's survivor bound** at fraction `f`, on the run `o`: `BlockWipe.Survives`' bound. -/
def WipeLive.Survives {ι : Type} {d : Device ι} {O : Type} (w : WipeLive d O) (f : ℝ) (o : O) : Prop :=
  (w.kept o : ℝ) ≤ f * w.fill + 2 * 128 + (d.chip + (d.mem - w.fill))

/-- A horizon of `B` server-time blocks read off one run: each block's period, over its own circuit, the wipe that
opens it, the bits of weights its PoUS store holds, its PoUS challenge times on the run, and the read deadline. -/
structure BlocksLive {ι : Type} (d : Device ι) (α : Type) [Fintype α] (B : ℕ) (C : Fin B → Verity.Circuit)
    (Cm O : Type) where
  /-- block `b`'s period -/
  block : (b : Fin B) → PeriodLive d α (C b) Cm O
  /-- the wipe that opens block `b` -/
  wipe : Fin B → WipeLive d O
  /-- the bits of weights block `b`'s PoUS store holds (U) -/
  weights : Fin B → ℕ
  /-- block `b`'s PoUS challenge times on the run -/
  chal : Fin B → O → Finset ℕ
  /-- the read deadline, in ticks (T_read) -/
  readDeadline : ℕ

/-- **Block `b`'s memory term** on the run `o` (`BlocksRun.MemoryTerm`'s, ASK-132): at each of its PoUS challenge
times, the bits of its weights that its capacity account doesn't hold in device memory are at most
`0.07 · U + S_chip + ∑_i cap_i · T_read`. -/
def BlocksLive.MemoryTerm {ι : Type} [Fintype ι] {d : Device ι} {α : Type} [Fintype α] {B : ℕ}
    {C : Fin B → Verity.Circuit} {Cm O : Type} (s : BlocksLive d α B C Cm O) (b : Fin B) (o : O) : Prop :=
  ∀ t ∈ s.chal b o, (s.weights b : ℝ) - (((s.block b).schedule o t).mem ((s.block b).acct .capacity) : ℝ) ≤
    7 / 100 * s.weights b + d.chip + (∑ i, d.cap i * s.readDeadline : ℕ)

end Accounting
