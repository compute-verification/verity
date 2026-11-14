module

public import Properties.Spot.Bits
public import Spot.Slots

/-!
# An audit's members side by side, as the claims' program (`Spot.Bits.Members`)

`Bits.Members` lists an audit's members, each a circuit file over some of the partition's units, their programs side by
side, member `m`'s positions from `Bits.base ms m`. This is that program as the claims read it: the program of a
statement over several templates' circuit files, each template's units lowered from its own file.

* `shiftGate b x`: the claims' gate `x`, every gate it reads `b` later.
* `sideBySide ps`: programs side by side, each program's gates after the earlier programs', every gate it reads and
  every output it returns `offsetOf ps m` later, the earlier programs' gates.
* `memberProg c n`: a member's program, the circuit file `c`'s key program for `n` units lowered at its constants
  (`bitsProg` at `Bits.keyProg c n` and `Bits.consts c n`).
* `bitsProgs ms`: the members' programs side by side.
* `unitBase`, `memberBlocks`, `membersPart ms`: the members' proof blocks, one class whose stratum `m` is member `m`'s
  units' blocks (`Bits.blockOf`) at its positions, member `m`'s unit `u` slot `unitBase ms m + u`.
* `membersConsts ms`: each member's constants (`Bits.consts`) at its positions.
* `filesBlockOf ms ns`, `filesUnits`, `filesBlock`, `filesReplay ms ns l`: the members' replay partition over the replay
  layout `l`, each member `t` holding `n_t` units (`ns`) per request: request `ρ`'s replay block is its units of every
  member together, member `t`'s units `ρ·n_t + p` for `p < n_t`, slot `filesBlockOf ms ns ρ t p`; an input slot's block
  holds none.
* `memberProgFast`, `memberBlocksFast`, `membersBlocksFrom`, `membersPartFast`, `filesBlockAt`, `filesReplayFast`:
  `memberProg`, `memberBlocks`, `membersPart` and `filesReplay` with what each rebuilds per gate, per unit or per member
  built once, equal to them and compiled in their place (`@[csimp]`).

`Proofs.Flock.Bridge.Members` proves that member `m` sits at `Bits.base ms m`, that no member's gate reads another's,
that each unit of a member is lowered there as Bridge's lowering of the member's circuit lowers it, and where each
member's proof blocks and constants sit; and that the members' replay partition has the replay layout's shape and
request `ρ`'s cell of member `t` is `filesBlockOf ms ns ρ t` on `[0, n_t)`.
-/

@[expose] public section

namespace Properties.Spot

open _root_.Spot

/-- **The claims' gate `x`, every gate it reads `b` later.** -/
def shiftGate (b : ℕ) : Claims.Sampling.BGate → Claims.Sampling.BGate
  | .input => .input
  | .const c => .const c
  | .not i => .not (b + i)
  | .and ws => .and (ws.map (b + ·))
  | .xor ws => .xor (ws.map (b + ·))

/-- **Programs side by side**: the first program's gates and outputs, then the others' side by side, every gate they read
and every output they return the first program's gates later. -/
def sideBySide : List Claims.Sampling.Prog → Claims.Sampling.Prog
  | [] => ⟨[], []⟩
  | p :: ps =>
    ⟨p.gates ++ (sideBySide ps).gates.map (shiftGate p.gates.length),
      p.outs ++ (sideBySide ps).outs.map (p.gates.length + ·)⟩

/-- **Program `m`'s first position** side by side: the gates of the programs before it. -/
def offsetOf (ps : List Claims.Sampling.Prog) (m : ℕ) : ℕ := ((ps.take m).map (·.gates.length)).sum

/-- **A member's program**: the circuit file `c`'s key program for `n` units, lowered at its constants. -/
def memberProg (c : _root_.Flock.Circuit) (n : ℕ) : Claims.Sampling.Prog :=
  bitsProg (Bits.keyProg c n) (Bits.consts c n)

/-- `memberProg`, its constants answering from the key program's owners (`Bits.owners`), built once
(`memberProg_eq_fast`); as stated, the lowering asks `Bits.consts c n` at every input gate, and each answer rebuilds the
key program. -/
def memberProgFast (c : _root_.Flock.Circuit) (n : ℕ) : Claims.Sampling.Prog :=
  let kp := Bits.keyProg c n
  let ow := Bits.owners kp
  bitsProg kp (Bits.constsAt ow kp)

@[csimp] theorem memberProg_eq_fast : @memberProg = @memberProgFast := by
  funext c n
  simp only [memberProg, memberProgFast, Bits.constsAt_owners]

/-- **The members' programs side by side**, member `m` from `Bits.base ms m`. -/
def bitsProgs (ms : Bits.Members) : Claims.Sampling.Prog :=
  sideBySide (ms.map fun ci => memberProg ci.1 ci.2.size)

/-- **Member `m`'s first slot**: the units of the members before it. -/
def unitBase (ms : Bits.Members) (m : ℕ) : ℕ := ((ms.take m).map (·.2.size)).sum

/-- **Member `m`'s proof blocks** in `bitsProgs ms`, in unit order: each unit's (`Bits.blockOf`), `Bits.base ms m`
later; none past the members. -/
def memberBlocks (ms : Bits.Members) (m : ℕ) : List (List ℕ) :=
  match ms[m]? with
  | some ci => (List.range ci.2.size).map fun u =>
      (Bits.blockOf (Bits.keyProg ci.1 ci.2.size) (Bits.consts ci.1 ci.2.size) u).map (Bits.base ms m + ·)
  | none => []

/-- `memberBlocks`, member `m`'s key program, owners (`Bits.owners`), offsets (`Bits.offs`) and first position built
once (`memberBlocks_eq_fast`); as stated, every unit rebuilds the key program, every input gate's constant and owner
rebuild it and scan it, and every gate rebuilds the first position. -/
def memberBlocksFast (ms : Bits.Members) (m : ℕ) : List (List ℕ) :=
  match ms[m]? with
  | some ci =>
    let kp := Bits.keyProg ci.1 ci.2.size
    let ow := Bits.owners kp
    let t := Bits.offs kp
    let b := Bits.base ms m
    (List.range ci.2.size).map fun u => (Bits.blockOfAt t ow kp (Bits.constsAt ow kp) u).map (b + ·)
  | none => []

@[csimp] theorem memberBlocks_eq_fast : @memberBlocks = @memberBlocksFast := by
  funext ms m
  unfold memberBlocks memberBlocksFast
  rcases ms[m]? with _ | ci
  · rfl
  · simp only [Bits.constsAt_owners, Bits.blockOfAt_offs]

/-- **The members' proof-block partition**: one class, whose stratum `m` is member `m`'s proof blocks, so member `m`'s
unit `u` is slot `unitBase ms m + u`. -/
def membersPart (ms : Bits.Members) : Claims.Sampling.Partition :=
  ⟨[(List.range ms.length).map (memberBlocks ms)]⟩

/-- Members `ms`' proof blocks, the first member's from position `b`: each member's key program, owners and offsets
built once, its blocks `memberBlocksFast`'s, and the next member's first position this one's past its lowered gates
(`Bits.offAt` at its key program's end). -/
def membersBlocksFrom : ℕ → Bits.Members → List (List (List ℕ))
  | _, [] => []
  | b, ci :: ms =>
    let kp := Bits.keyProg ci.1 ci.2.size
    let ow := Bits.owners kp
    let t := Bits.offs kp
    ((List.range ci.2.size).map fun u => (Bits.blockOfAt t ow kp (Bits.constsAt ow kp) u).map (b + ·)) ::
      membersBlocksFrom (b + Bits.offAt t kp kp.size) ms

/-- `membersPart`, each member's first position the one before it plus that member's lowered gates
(`membersPart_eq_fast`); as stated, member `m`'s first position (`Bits.base ms m`) rebuilds the key program of every
member before it. -/
def membersPartFast (ms : Bits.Members) : Claims.Sampling.Partition := ⟨[membersBlocksFrom 0 ms]⟩

theorem bitsBase_zero (ms : Bits.Members) : Bits.base ms 0 = 0 := by
  simp [Bits.base]

theorem bitsBase_cons_succ (ci : _root_.Flock.Circuit × Array ℕ) (ms : Bits.Members) (m : ℕ) :
    Bits.base (ci :: ms) (m + 1) =
      Bits.off (Bits.keyProg ci.1 ci.2.size) (Bits.keyProg ci.1 ci.2.size).size + Bits.base ms m := by
  simp [Bits.base]

theorem memberBlocks_cons_zero (ci : _root_.Flock.Circuit × Array ℕ) (ms : Bits.Members) :
    memberBlocks (ci :: ms) 0 = (List.range ci.2.size).map fun u =>
      Bits.blockOf (Bits.keyProg ci.1 ci.2.size) (Bits.consts ci.1 ci.2.size) u := by
  simp only [memberBlocks, List.getElem?_cons_zero, bitsBase_zero, Nat.zero_add, List.map_id']

theorem memberBlocks_cons_succ (ci : _root_.Flock.Circuit × Array ℕ) (ms : Bits.Members) (m : ℕ) :
    memberBlocks (ci :: ms) (m + 1) = (memberBlocks ms m).map
      (·.map (Bits.off (Bits.keyProg ci.1 ci.2.size) (Bits.keyProg ci.1 ci.2.size).size + ·)) := by
  simp only [memberBlocks, List.getElem?_cons_succ, bitsBase_cons_succ]
  rcases ms[m]? with _ | cj
  · rfl
  · simp only [List.map_map, Function.comp_def, Nat.add_assoc]

theorem membersBlocksFrom_eq (b : ℕ) (ms : Bits.Members) :
    membersBlocksFrom b ms = (List.range ms.length).map fun m => (memberBlocks ms m).map (·.map (b + ·)) := by
  induction ms generalizing b with
  | nil => rfl
  | cons ci ms ih =>
    simp only [membersBlocksFrom, Bits.constsAt_owners, Bits.blockOfAt_offs, Bits.offAt_offs, ih, List.length_cons,
      List.range_succ_eq_map, List.map_cons, List.map_map, Function.comp_def, Nat.succ_eq_add_one,
      memberBlocks_cons_zero, memberBlocks_cons_succ, Nat.add_assoc]

@[csimp] theorem membersPart_eq_fast : @membersPart = @membersPartFast := by
  funext ms
  rw [membersPart, membersPartFast, membersBlocksFrom_eq]
  simp only [Nat.zero_add, List.map_id']

/-- **The members' constants**: member `m`'s (`Bits.consts`) at its positions, `Bits.base ms m` later; none past the
members. A key program's inputs come first, one lowered gate each, so `Bits.consts` reads a lowered position as the
input gate there. -/
def membersConsts (ms : Bits.Members) (j : ℕ) : Option Bool :=
  match (List.range ms.length).find? (fun m => decide (j < Bits.base ms (m + 1))) with
  | some m => (ms[m]?).bind fun ci => Bits.consts ci.1 ci.2.size (j - Bits.base ms m)
  | none => none

/-- **Request `u`'s unit `p` of member `t`**, member `t` holding `ns[t]` units per request: its slot in
`membersPart ms`. -/
def filesBlockOf (ms : Bits.Members) (ns : Array ℕ) (u t p : ℕ) : ℕ := unitBase ms t + (u * ns.getD t 0 + p)

/-- **Request `u`'s proof blocks**: its units of every member, member by member. -/
def filesUnits (ms : Bits.Members) (ns : Array ℕ) (u : ℕ) : List ℕ :=
  (List.range ms.length).flatMap fun t => (List.range (ns.getD t 0)).map (filesBlockOf ms ns u t)

/-- **A replay slot's block in the members' replay partition**: its request's proof blocks' gates together; none for an
input slot. -/
def filesBlock (ms : Bits.Members) (ns : Array ℕ) (b : Slots.ProofBlock) : List ℕ :=
  match b.unit with
  | some u => (filesUnits ms ns u).flatMap fun v => (membersPart ms).blocks.getD v []
  | none => []

/-- **The members' replay partition** over the replay layout `l`: the layout's classes and strata, each slot's block its
request's proof blocks' gates (`filesBlock`). -/
def filesReplay (ms : Bits.Members) (ns : Array ℕ) (l : Slots.SlotOrder) : Claims.Sampling.Partition :=
  ⟨l.classes.toList.map fun c => c.toList.map fun s => s.toList.map (filesBlock ms ns)⟩

/-- `filesBlock` reading the members' proof blocks from `bs`, `membersPart ms`'s. -/
def filesBlockAt (bs : List (List ℕ)) (ms : Bits.Members) (ns : Array ℕ) (b : Slots.ProofBlock) : List ℕ :=
  match b.unit with
  | some u => (filesUnits ms ns u).flatMap fun v => bs.getD v []
  | none => []

/-- `filesReplay`, the members' proof blocks built once (`filesReplay_eq_fast`); as stated, every unit of every slot
rebuilds them. -/
def filesReplayFast (ms : Bits.Members) (ns : Array ℕ) (l : Slots.SlotOrder) : Claims.Sampling.Partition :=
  let bs := (membersPart ms).blocks
  ⟨l.classes.toList.map fun c => c.toList.map fun s => s.toList.map (filesBlockAt bs ms ns)⟩

@[csimp] theorem filesReplay_eq_fast : @filesReplay = @filesReplayFast := by
  funext ms ns l
  have e : filesBlockAt (membersPart ms).blocks ms ns = filesBlock ms ns := by
    funext b
    unfold filesBlockAt filesBlock
    rcases b.unit with _ | u <;> rfl
  simp only [filesReplay, filesReplayFast, e]

end Properties.Spot
