import Definitions.Claims.Sampling
import Definitions.NetworkCertifier.Run
import Mathlib.Logic.Relation
import Mathlib.Order.Interval.Finset.Nat

/-!
# The sampled audit and acceptance: what an accepted audit says about the egress, at `δ`

Definitions only. Network accounting writes the whole computation as one program `C` of the Boolean profile,
each machine a subcircuit, and audits its committed trace with sampled proofs (`Claims.Sampling`). The transfer-gate
units and N2's head checks are classes drawn `all`, and the internal step units are sampled. This module says what an
accepted audit then establishes about the egress, in the network certifier's terms (`EgressPrediction`, `Accepted`), on
sampling's own vocabulary: its statement, partition, design, faulty set and `M_δ`.

**The layout** (`BoundaryGates`, public, part of the statement) says which of `C`'s gates hold each step's egress rows
(per slot, a presence gate and the frame's bit gates) and its ingress (input gates, in bit order), and which input gates
are wired: each wired input gate's writer, a gate whose value the registered descriptor's wiring carries into it. It
fits the program (`BoundaryGates.Fits`) when, before the horizon, the egress gates are gates `C` returns and not inputs,
so their values are committed and computed by `C`, the ingress gates are inputs, `C` is causal through its reads and
its wires (`DependsWired`), and the wiring fits (`BoundaryGates.WiresFit`: each wired gate an input, its writer a gate
`C` returns or an input, and the wires acyclic). An input writer is how a later copy of an input bit reads its first
copy, both opening one committed leaf. A wire may carry ingress, as long as no step's egress reaches a later step's
ingress through it. The verifier refuses a statement whose layout doesn't fit, as it
refuses one outside the sampling scope.

**The circuit's Program** (`circuitProgram`) is `C` run on the ingress up to step `t` and the declared advice, read off
at step `t`'s egress gates, each wired input gate reading what its writer computes (`resolveWires`). The advice is the
inputs other than the ingress and the wired gates, plus values held at a set of gates (`valOver`). An incorrect proof
block's committed values need not be what its gates compute, so they enter as advice.

**The incorrect set** (`incorrect`) is the faulty set and the wire-wrong proof blocks (`wireWrong`): those holding a
wired gate, or a gate that reads one, whose committed value is not its writer's committed value, or whose writer is a
constant gate (`WireRight`). With no wires it is the faulty set.

**The relation** (`RAcc`) has three parts:

* the sampling relation at the incorrect set: it is in `M_δ`;
* the incorrect set holds no proof block of a class sampled `all` (which `M_δ` implies at `δ ≥ 0`, since such a class's
  `P_a` factor is `0` on any set that meets it), so the transfer-gate proof blocks and head checks are exact, an error
  of `0`;
* some advice, holding values only at the incorrect proof blocks' committed gates, makes `Accepted` by the circuit's
  Program every observation sequence and operator that the trace's transfer gates carry (`Carries`) with every status
  complete (`Covered`).

**Sampling at the incorrect set** is `Claims.Sampling.KSBy` at the wired check (`wiredOk`): a drawn proof block is
correct and its wired rows are its writers' committed values (`WireRight`), and the proof blocks that fail it are the
incorrect set. With no wires it is `KS`.

**What this module does not model.** The certifier's signed rows and the operator's moves are tied to the transfer gates
by the transfer-gate proof blocks and, beyond them, by the certifier's trust (`trusted-host`). That tie is the condition
`Carries` picks out, inside the conclusion. The statuses are the certifier's published record, which the network check
reads (`Covered`).
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **The boundary gates**, the layout of the run's boundary in a program's gates: at step `t`, slot `σ` of egress link
`ℓ` is a presence gate and its frame's bit gates (`egress t ℓ σ`), and step `t`'s ingress is the gates `ingress t`, in
bit order. `wires g = some w` wires the input gate `g` to its writer `w`, the gate whose value the registered
descriptor's wiring carries into `g`; by default no gate is wired. -/
structure BoundaryGates (L S : ℕ) where
  egress : ℕ → Fin L → Fin S → ℕ × List ℕ
  ingress : ℕ → List ℕ
  wires : ℕ → Option ℕ := fun _ => none

/-- A slot's gates: its presence gate and its frame's bit gates. -/
def slotGates (p : ℕ × List ℕ) : List ℕ := p.1 :: p.2

/-- The row the gate values `val` hold at the slots `p`: slot `σ` holds the frame its bit gates carry when its presence
gate is set, and no frame otherwise. -/
def rowOf {S : ℕ} (val : ℕ → Bool) (p : Fin S → ℕ × List ℕ) : Row S (List Bool) :=
  fun σ => if val (p σ).1 then some ((p σ).2.map val) else none

/-- **`C` run with gates held**: a gate that `free` holds at a value has that value; every other gate computes from the
wires it reads (a wire at or past it reads `0`, as in `Prog.wires`), and an input reads `inp`. -/
def valOver (C : Prog) (inp : ℕ → Bool) (free : ℕ → Option Bool) (g : ℕ) : Bool :=
  match free g with
  | some b => b
  | none =>
    match C.gates[g]? with
    | some x =>
      (x.val (fun i => if _h : i < g then some (valOver C inp free i) else some false) (some (inp g))).getD false
    | none => false
termination_by g

/-- **The inputs with the wires resolved** in `n` rounds: in round `n + 1` a wired input gate reads what its writer
computes (`valOver`) on round `n`'s inputs, and in round `0` it reads `false`; every other input reads `inp`. Under
acyclic wires (`BoundaryGates.WiresFit`) a wired gate's value is fixed from the round past the number of wired gates
ranked below it. -/
def resolveWires (C : Prog) (wires : ℕ → Option ℕ) (inp : ℕ → Bool) (free : ℕ → Option Bool) : ℕ → ℕ → Bool
  | 0, g =>
    match wires g with
    | some _ => false
    | none => inp g
  | n + 1, g =>
    match wires g with
    | some w => valOver C (resolveWires C wires inp free n) free w
    | none => inp g

/-- **The inputs at ingress `ins`** over the horizon `k`: bit `j` of step `i`'s ingress gates, for `i < ins.length`,
reads bit `j` of `ins[i]` (`false` past its end); an ingress gate of a later step before `k` reads `false`; every other
input reads the advice `a`. -/
def BoundaryGates.inputOf {L S : ℕ} (lay : BoundaryGates L S) (k : ℕ) (ins : List (List Bool)) (a : ℕ → Bool)
    (g : ℕ) : Bool :=
  match (List.range ins.length).findSome? (fun i => ((lay.ingress i).findIdx? (· == g)).map (i, ·)) with
  | some (i, j) => (ins.getD i []).getD j false
  | none => if (List.range k).any (fun i => (lay.ingress i).contains g) then false else a g

/-- **The circuit's Program** over the horizon `k`: at step `t`, the egress rows `C` holds at the layout's egress gates,
run on the ingress up to `t` and the declared advice `(a, free)`: the inputs other than the ingress and the wired gates
(`a`), and the gates held at values (`free`). Each wired gate reads what its writer computes, the wires resolved in as
many rounds as `C` has gates (`resolveWires`). -/
def circuitProgram {L S : ℕ} (C : Prog) (lay : BoundaryGates L S) (k : ℕ) :
    EgressPrediction L S (List Bool) ((ℕ → Bool) × (ℕ → Option Bool)) where
  rows t ins adv ℓ :=
    rowOf (valOver C (resolveWires C lay.wires (lay.inputOf k ins adv.1) adv.2 C.gates.length) adv.2) (lay.egress t ℓ)

/-- Gate `g` of `C` depends on gate `i`: it is `i`, or it reads a gate that depends on `i`. -/
def Depends (C : Prog) (i g : ℕ) : Prop :=
  Relation.ReflTransGen (fun a b => a ∈ (C.gates[b]?.map BGate.reads).getD []) i g

/-- **The wires are acyclic** in `C`: some rank `r` falls along every wire, from a wired gate `g` to each wired gate its
writer depends on (`Depends` is reflexive, so a writer that is itself a wired gate ranks below `g`). With no wires it
holds (`r = 0`). -/
def WiresAcyclic (C : Prog) (wires : ℕ → Option ℕ) : Prop :=
  ∃ r : ℕ → ℕ, ∀ g w g', wires g = some w → wires g' ≠ none → Depends C g' w → r g' < r g

/-- Gate `g` of `C` depends on gate `i` through reads and wires: it is `i`, or it reads a gate that does, or it is a
wired gate whose writer does. With no wires it is `Depends`. -/
def DependsWired (C : Prog) (wires : ℕ → Option ℕ) (i g : ℕ) : Prop :=
  Relation.ReflTransGen (fun a b => a ∈ (C.gates[b]?.map BGate.reads).getD [] ∨ wires b = some a) i g

/-- **The wiring fits the program** over the horizon `k`. Each wired gate is an input of `C` and no ingress gate before
`k`, so its value is committed, and its writer is a gate `C` returns or an input, so its value is committed too, and a
wire reads that committed value (`WireRight`). A wire from a constant gate is never right, since a constant need sit in
no proof block: a constant's or a pass-through's wired leaf is written by its copy gate, a computed gate of the writer's
proof block that the block's own proof holds to the leaf's value. An input writer may itself be wired: it is how a later
copy of an input bit reads its first copy, where the layout gives each reading proof block its own copy. A writer may
depend on ingress; that no step's egress reaches a later step's ingress through a wire is the layout's causality
(`BoundaryGates.Fits`, over `DependsWired`). The wires are acyclic (`WiresAcyclic`), so the chain of writers from a
wired gate ends, at a writer that is not a wired gate: a computed gate, an ingress gate, a registered input or a free
input. -/
def BoundaryGates.WiresFit {L S : ℕ} (lay : BoundaryGates L S) (C : Prog) (k : ℕ) : Prop :=
  (∀ g w, lay.wires g = some w → C.gates[g]? = some .input ∧ (∀ t < k, g ∉ lay.ingress t) ∧
    (w ∈ C.outs ∨ C.gates[w]? = some .input)) ∧
    WiresAcyclic C lay.wires

/-- **The layout fits the program** over the horizon `k`, which the verifier checks before it samples. Before `k`: every
egress gate is one `C` returns, so its value is committed, and is not an input, so `C` computes it (the circuit's Program
would read an input egress gate from the advice, and so accept any rows the trace carries); every ingress gate is an
input of `C`; no egress gate of a step depends, through reads and wires (`DependsWired`), on an ingress gate of a later
step (`C` is causal); and the wiring fits (`BoundaryGates.WiresFit`). -/
def BoundaryGates.Fits {L S : ℕ} (lay : BoundaryGates L S) (C : Prog) (k : ℕ) : Prop :=
  (∀ t < k, ∀ ℓ σ, ∀ g ∈ slotGates (lay.egress t ℓ σ), g ∈ C.outs ∧ C.gates[g]? ≠ some .input) ∧
    (∀ t < k, ∀ g ∈ lay.ingress t, C.gates[g]? = some .input) ∧
    (∀ t < k, ∀ t' < k, t < t' → ∀ ℓ σ, ∀ g ∈ slotGates (lay.egress t ℓ σ), ∀ i ∈ lay.ingress t',
      ¬ DependsWired C lay.wires i g) ∧
    lay.WiresFit C k

/-- The trace's value at gate `g`, as a gate outside the gate's proof block reads it (`Prog.read`: a constant's own
value, every other gate's committed value), `false` when unopened. -/
def traceVal (C : Prog) (ref : ℕ → Option Bool) (g : ℕ) : Bool := (C.read ref g).getD false

/-- **What the trace's transfer gates carry**: at each step `t < k` the observation `o[t]` exists, its egress rows are
the rows the trace holds at the layout's egress gates, and the operator's move on the observations before `t` is the
ingress the trace holds at the layout's ingress gates. -/
def Carries {L K S : ℕ} (lay : BoundaryGates L S) (C : Prog) (ref : ℕ → Option Bool)
    (op : ℕ → List (Obs L K S) → List Bool) (k : ℕ) (o : List (Obs L K S)) : Prop :=
  ∀ t (_ : t < k), ∃ h : t < o.length,
    (∀ ℓ, o[t].rows ℓ = rowOf (traceVal C ref) (lay.egress t ℓ)) ∧
      op t (o.take t) = (lay.ingress t).map (traceVal C ref)

/-- Every boundary link's status is complete at each step before `k`: the certifier's published record, which the
network check reads. -/
def Covered {L K S : ℕ} (k : ℕ) (o : List (Obs L K S)) : Prop :=
  ∀ t (_ : t < k) (h : t < o.length), ∀ κ, o[t].status κ = Status.complete

/-- Gate `g` is a committed gate of an incorrect proof block: it is on the boundary of the proof unit at a slot of `F`. -/
def FaultyGate (C : Prog) (P : Partition) (F : Finset ℕ) (g : ℕ) : Prop :=
  ∃ v ∈ F, ∃ u, P.blocks[v]? = some u ∧ g ∈ u ∧ C.Boundary u g = true

/-- The slots of the classes the rules draw `all`, from slot `o`, as `classesPa` walks the classes. -/
def allSlotsFrom : List OneStage → List (List (List (List ℕ))) → ℕ → Finset ℕ
  | r :: rs, c :: cs, o =>
    (match r with
      | .all => Finset.Ico o (o + c.flatten.length)
      | _ => ∅) ∪ allSlotsFrom rs cs (o + c.flatten.length)
  | _, _, _ => ∅

/-- **The slots drawn `all`**: the proof units of the classes whose rule draws every member. -/
def allSlots (d : Design) (P : Partition) : Finset ℕ := allSlotsFrom d.rules P.classes 0

/-- **The wire into gate `g` of `C` is right** on the committed values `val`: where `g` is wired to the writer `w`, `w`
is no constant gate and `g` is opened to `w`'s committed value: the leaf a reader's session opens, not the value
`Prog.read` gives a constant. A constant's or a pass-through's wired leaf is written by its copy gate
(`BoundaryGates.WiresFit`), so a writer whose committed bit isn't the bit it copies is in an incorrect proof block. At
the deployment the wires are `SampledProofs.Bits.wiresOf`'s, and each writer is a copy row's AND or an input in the
members' lowered programs (`Properties.SampledProofs.Wires.wiresOf_gate`). -/
def WireRight (C : Prog) (wires : ℕ → Option ℕ) (val : ℕ → Option Bool) (g : ℕ) : Prop :=
  ∀ w, wires g = some w → (∀ c, C.gates[w]? ≠ some (.const c)) ∧ ∃ b, val g = some b ∧ val w = some b

/-- **A wire-wrong proof block** at slot `v` of `P` under `w`, at the wiring `wires`: the proof unit there holds a wired
gate, or a gate that reads one, whose wire is not right (`WireRight`): its writer is a constant gate, or its committed
value is not its writer's, or either is unopened. -/
def WireWrong (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (P : Partition) (wires : ℕ → Option ℕ)
    (w : Wit) (v : ℕ) : Prop :=
  ∃ u, P.blocks[v]? = some u ∧ ∃ g, (g ∈ u ∨ ∃ x ∈ u, g ∈ (s.prog.gates[x]?.map BGate.reads).getD []) ∧
    ¬ WireRight s.prog wires (w.val H Cm s) g

open Classical in
/-- **The wire-wrong set** of the trace `w`'s openings, at the wiring `wires`. With no wires it is empty. -/
noncomputable def wireWrong (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (P : Partition)
    (wires : ℕ → Option ℕ) (w : Wit) : Finset ℕ :=
  (Finset.range P.blocks.length).filter (WireWrong H Cm s P wires w)

/-- **The incorrect set** of the trace `w`'s openings, at the wiring `wires`: the faulty proof blocks and the wire-wrong
ones. With no wires it is the faulty set. -/
noncomputable def incorrect (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (P : Partition)
    (wires : ℕ → Option ℕ) (w : Wit) : Finset ℕ :=
  faulty H Cm s P w ∪ wireWrong H Cm s P wires w

/-- **The wired check** on proof block `u`, a check on each proof block (`Claims.Sampling.BlockCheck`): correct
(`correctOk`), and each wired gate it holds or reads has its wire right (`WireRight`) at the statement's wiring
`wires s`. The proof blocks that fail it are the incorrect set (`incorrect`). -/
def wiredOk (wires : AuditStatement → ℕ → Option ℕ) : BlockCheck := fun s u val ι =>
  correctOk s u val ι ∧
    ∀ g, (g ∈ u ∨ ∃ x ∈ u, g ∈ (s.prog.gates[x]?.map BGate.reads).getD []) → WireRight s.prog (wires s) val g

/-- **The sampled audit's relation** at `δ`, on the statement's bytes. The statement parses, its query evaluates, and,
at the layout's wiring: the sampling relation holds at the incorrect set (`incorrect`: it is in `M_δ`); the incorrect
set holds no slot sampled `all`; and the advice `(traceVal C ref, free)`, its inputs the trace's committed values and
`free` holding values exactly at the incorrect proof blocks' committed gates (`FaultyGate`), makes `Accepted` by the
circuit's Program, over the horizon, every observation sequence and operator the trace's transfer gates carry
(`Carries`) with every status complete (`Covered`). The inputs are pinned to the trace, so the registered ones are what
the registration commits. -/
def RAcc (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement) (partition : AuditStatement → Option Partition)
    (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧
    incorrect H Cm s P (lay s).wires w ∈ s.design.M P δ ∧
    Disjoint (incorrect H Cm s P (lay s).wires w) (allSlots s.design P) ∧
    ∃ free : ℕ → Option Bool, (∀ g b, free g = some b → FaultyGate s.prog P (incorrect H Cm s P (lay s).wires w) g) ∧
      (∀ g, FaultyGate s.prog P (incorrect H Cm s P (lay s).wires w) g → free g ≠ none) ∧
      ∀ (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)),
        Carries (lay s) s.prog (w.val H Cm s) op (k s) o → Covered (k s) o →
          Accepted (circuitProgram s.prog (lay s) (k s)) op (k s) o (traceVal s.prog (w.val H Cm s), free)

section Claim

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The sampled audit's scope**: the sampling scope (`Scope`), and the statement's layout fits its program over its
horizon (`BoundaryGates.Fits`). -/
def ScopeAcc {L S : ℕ} (D : Deploy Coin Col T) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (x : Msg) : Prop :=
  Scope D x ∧ ∃ s, D.parse x = some s ∧ (lay s).Fits s.prog (k s)

/-- **The sampled audit implies `Accepted`** at the deployment `D`, at the public layout `lay` and horizon `k` of each
statement. It holds on the statements in its scope (`ScopeAcc`), at every `δ ≥ 0`, at the sampling extractor and every
decoder that lists every trace within Flock's radius: `Pr[acc ∧ ¬RAcc δ] ≤ δ + ε_ks + ε_bind`, with `KS`'s errors
(`ε_ks = κ − δ + √(γ·Pr[B])`, Flock's error at the proof blocks' sessions; `ε_bind = Pr[B_ext]`). The selection enters only
through `M_δ`. -/
def KSAccepted (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) : Prop :=
  ∀ (δ : ℝ) decode, 0 ≤ δ → Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode →
    KSLinOn H D.V (ScopeAcc D lay k) (RAcc H D.commit D.parse D.partition L K S lay k δ) (E D decode) (κ D δ) (γ D)
      (fun _ => 1) (Bext H D.V (E D decode))

end Claim

end NetTiming
