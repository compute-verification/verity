import Definitions.Claims.Sampling
import Definitions.NetworkCertifier.Run
import Mathlib.Logic.Relation
import Mathlib.Order.Interval.Finset.Nat

/-!
# The sampled audit and acceptance: what an accepted audit says about the egress, at `δ`

Definitions only. Network accounting writes the whole computation as one program `C` of the Boolean profile,
each machine a subcircuit, and audits its committed trace with sampled proofs (`Claims.Sampling`). The transfer-gate
units and N2's head checks are classes drawn `all`, and the internal step units are sampled. This module says what an
accepted audit then establishes about the egress, in the network certifier's terms (`Program`, `Accepted`), on
sampling's own vocabulary: its statement, partition, design, faulty set and `M_δ`.

**The layout** (`Layout`, public, part of the statement) says which of `C`'s gates hold each step's egress rows (per
slot, a presence gate and the frame's bit gates) and its ingress (input gates, in bit order). It fits the program
(`Layout.Fits`) when, before the horizon, the egress gates are gates `C` returns and not inputs, so their values are
committed and computed by `C`, the ingress gates are inputs, and `C` is causal. The verifier refuses a statement whose
layout doesn't fit, as it refuses one outside the sampling scope.

**The circuit's Program** (`circuitProgram`) is `C` run on the ingress up to step `t` and the declared advice, read off
at step `t`'s egress gates. The advice is the inputs other than the ingress, plus values held at a set of gates
(`valOver`). A faulty unit's committed values need not be what its gates compute, so they enter as advice.

**The relation** (`RAcc`) has three parts:

* the sampling relation: the faulty set is in `M_δ`;
* the faulty set holds no unit of a class drawn `all` (which `M_δ` implies at `δ ≥ 0`, since such a class's `P_a` factor
  is `0` on any set that meets it), so the transfer-gate units and head checks are exact, an error of `0`;
* some advice, holding values only at the faulty units' committed gates, makes `Accepted` by the circuit's Program every
  observation sequence and operator that the trace's transfer gates carry (`Carries`) with every status complete
  (`Covered`).

**What this module does not model.** The certifier's signed rows and the operator's moves are tied to the transfer gates
by the transfer-gate units and, beyond them, by the certifier's trust (`trusted-host`). That tie is the condition
`Carries` picks out, inside the conclusion. The statuses are the certifier's published record, which the network check
reads (`Covered`).
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **The layout of the run's boundary** in a program's gates: at step `t`, slot `σ` of egress link `ℓ` is a presence
gate and its frame's bit gates (`egress t ℓ σ`), and step `t`'s ingress is the gates `ingress t`, in bit order. -/
structure Layout (L S : ℕ) where
  egress : ℕ → Fin L → Fin S → ℕ × List ℕ
  ingress : ℕ → List ℕ

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

/-- **The inputs at ingress `ins`** over the horizon `k`: bit `j` of step `i`'s ingress gates, for `i < ins.length`,
reads bit `j` of `ins[i]` (`false` past its end); an ingress gate of a later step before `k` reads `false`; every other
input reads the advice `a`. -/
def Layout.inputOf {L S : ℕ} (lay : Layout L S) (k : ℕ) (ins : List (List Bool)) (a : ℕ → Bool) (g : ℕ) : Bool :=
  match (List.range ins.length).findSome? (fun i => ((lay.ingress i).findIdx? (· == g)).map (i, ·)) with
  | some (i, j) => (ins.getD i []).getD j false
  | none => if (List.range k).any (fun i => (lay.ingress i).contains g) then false else a g

/-- **The circuit's Program** over the horizon `k`: at step `t`, the egress rows `C` holds at the layout's egress gates,
run on the ingress up to `t` and the declared advice `(a, free)`: the inputs other than the ingress (`a`), and the gates
held at values (`free`). -/
def circuitProgram {L S : ℕ} (C : Prog) (lay : Layout L S) (k : ℕ) :
    Program L S (List Bool) ((ℕ → Bool) × (ℕ → Option Bool)) where
  rows t ins adv ℓ := rowOf (valOver C (lay.inputOf k ins adv.1) adv.2) (lay.egress t ℓ)

/-- Gate `g` of `C` depends on gate `i`: it is `i`, or it reads a gate that depends on `i`. -/
def Depends (C : Prog) (i g : ℕ) : Prop :=
  Relation.ReflTransGen (fun a b => a ∈ (C.gates[b]?.map BGate.reads).getD []) i g

/-- **The layout fits the program** over the horizon `k`, which the verifier checks before it draws. Before `k`: every
egress gate is one `C` returns, so its value is committed, and is not an input, so `C` computes it (the circuit's Program
would read an input egress gate from the advice, and so accept any rows the trace carries); every ingress gate is an
input of `C`; and no egress gate of a step depends on an ingress gate of a later step (`C` is causal). -/
def Layout.Fits {L S : ℕ} (lay : Layout L S) (C : Prog) (k : ℕ) : Prop :=
  (∀ t < k, ∀ ℓ σ, ∀ g ∈ slotGates (lay.egress t ℓ σ), g ∈ C.outs ∧ C.gates[g]? ≠ some .input) ∧
    (∀ t < k, ∀ g ∈ lay.ingress t, C.gates[g]? = some .input) ∧
    ∀ t < k, ∀ t' < k, t < t' → ∀ ℓ σ, ∀ g ∈ slotGates (lay.egress t ℓ σ), ∀ i ∈ lay.ingress t', ¬ Depends C i g

/-- The trace's value at gate `g`, as a gate outside the gate's unit reads it (`Prog.read`: a constant's own value,
every other gate's committed value), `false` when unopened. -/
def traceVal (C : Prog) (ref : ℕ → Option Bool) (g : ℕ) : Bool := (C.read ref g).getD false

/-- **What the trace's transfer gates carry**: at each step `t < k` the observation `o[t]` exists, its egress rows are
the rows the trace holds at the layout's egress gates, and the operator's move on the observations before `t` is the
ingress the trace holds at the layout's ingress gates. -/
def Carries {L K S : ℕ} (lay : Layout L S) (C : Prog) (ref : ℕ → Option Bool)
    (op : ℕ → List (Obs L K S) → List Bool) (k : ℕ) (o : List (Obs L K S)) : Prop :=
  ∀ t (_ : t < k), ∃ h : t < o.length,
    (∀ ℓ, o[t].rows ℓ = rowOf (traceVal C ref) (lay.egress t ℓ)) ∧
      op t (o.take t) = (lay.ingress t).map (traceVal C ref)

/-- Every boundary link's status is complete at each step before `k`: the certifier's published record, which the
network check reads. -/
def Covered {L K S : ℕ} (k : ℕ) (o : List (Obs L K S)) : Prop :=
  ∀ t (_ : t < k) (h : t < o.length), ∀ κ, o[t].status κ = Status.complete

/-- Gate `g` is a committed gate of a faulty unit: it is on the boundary of the replay unit at a slot of `F`. -/
def FaultyGate (C : Prog) (P : Partition) (F : Finset ℕ) (g : ℕ) : Prop :=
  ∃ v ∈ F, ∃ r, P.rus[v]? = some r ∧ g ∈ RU.gates r ∧ C.Boundary (RU.gates r) g = true

/-- The slots of the classes the rules draw `all`, from slot `o`, as `classesPa` walks the classes. -/
def allSlotsFrom : List Rule → List (List (List RU)) → ℕ → Finset ℕ
  | r :: rs, c :: cs, o =>
    (match r.first with
      | .all => Finset.Ico o (o + c.flatten.length)
      | _ => ∅) ∪ allSlotsFrom rs cs (o + c.flatten.length)
  | _, _, _ => ∅

/-- **The slots drawn `all`**: the replay units of the classes whose rule draws every member. -/
def allSlots (d : Design) (P : Partition) : Finset ℕ := allSlotsFrom d.rules P.classes 0

/-- **The sampled audit's relation** at `δ`, on the statement's bytes. The statement parses, its query evaluates, and:
the sampling relation holds (`RelS`: the faulty set is in `M_δ`); the faulty set holds no slot drawn `all`; and the
advice `(traceVal C ref, free)`, its inputs the trace's committed values and `free` holding values exactly at the
faulty units' committed gates (`FaultyGate`), makes `Accepted` by the circuit's Program, over the horizon, every observation
sequence and operator the trace's transfer gates carry (`Carries`) with every status complete (`Covered`). The inputs
are pinned to the trace, so the registered ones are what the registration commits. -/
def RAcc (H : Msg → Msg) (Cm : Commitment Bool) (parse : Msg → Option Stmt) (partition : Stmt → Option Partition)
    (L K S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧ RelS H Cm δ s P w ∧
    Disjoint (faulty H Cm s P w) (allSlots s.design P) ∧
    ∃ free : ℕ → Option Bool, (∀ g b, free g = some b → FaultyGate s.prog P (faulty H Cm s P w) g) ∧
      (∀ g, FaultyGate s.prog P (faulty H Cm s P w) g → free g ≠ none) ∧
      ∀ (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)),
        Carries (lay s) s.prog (w.val H Cm s) op (k s) o → Covered (k s) o →
          Accepted (circuitProgram s.prog (lay s) (k s)) op (k s) o (traceVal s.prog (w.val H Cm s), free)

section Claim

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The sampled audit's scope**: the sampling scope (`Scope`), and the statement's layout fits its program over its
horizon (`Layout.Fits`). -/
def ScopeAcc {L S : ℕ} (D : Deploy Coin Col T) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (x : Msg) : Prop :=
  Scope D x ∧ ∃ s, D.parse x = some s ∧ (lay s).Fits s.prog (k s)

/-- **The sampled audit implies `Accepted`** at the deployment `D`, at the public layout `lay` and horizon `k` of each
statement. It holds on the statements in its scope (`ScopeAcc`), at every `δ ≥ 0`, at the sampling extractor and every
decoder that lists every trace within Flock's radius: `Pr[acc ∧ ¬RAcc δ] ≤ δ + ε_ks + ε_bind`, with `KS`'s errors
(`ε_ks = κ − δ + √(γ·Pr[B])`, Flock's error at the units' sessions; `ε_bind = Pr[B_ext]`). The selection enters only
through `M_δ`. -/
def KSAccepted (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ) : Prop :=
  ∀ (δ : ℝ) decode, 0 ≤ δ → Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode →
    KSLinOn H D.V (ScopeAcc D lay k) (RAcc H D.commit D.parse D.partition L K S lay k δ) (E D decode) (κ D δ) (γ D)
      (fun _ => 1) (Bext H D.V (E D decode))

end Claim

end NetTiming
