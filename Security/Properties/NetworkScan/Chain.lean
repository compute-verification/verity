module

public import Properties.NetworkScan
public import Definitions.NetworkCertifier.Checked

/-!
# The scan's state chain and its count: what `NetworkCheck.Period.scanOf`'s acceptance gives the checked inputs

The scan's Program is flattened with no wire between its units: bucket `t`'s ingress unit reads the state the bucket
before passed on through its `sin` parameter, and bucket `t`'s egress unit reads the view its ingress unit gives through
its `state` parameter, and both parameters are input nodes of the root (`scanOf`, the generator form). So in the
flattening those gates are free inputs (`NetTiming.freeInputs`), beside the slot entries the ingress units read through
their entry ports `prog<j>`.

They are not the prover's to choose. A run on which each `sin` holds what the unit before returned as its state, and
each egress `state` what its bucket's ingress unit returned as its view (`Chained`), has them fixed by the ingress and
the slot entries: they are checked inputs (`NetTiming.CheckedUnique`, `stateGates`), which the leak bounds with checked
inputs leave out of the count (`NetTiming.Guarantees.SampledLeakChecked`).

**The gates.** The ingress Definitions return a verdict, the state and the view, in that order, so the state a unit
passes on is component 1 of its return (`stateOut`) and the view it gives component 2 (`viewOut`), each a run of the
return's leaves (`componentRun`). A later bucket's `sin` is parameter 1 of its unit (`sinGates`) and an egress unit's
`state` its parameter 0 (`viewGates`), as `scanOf` holds the templates' parameters to their names. `Chained` equates
the values of each read with the values of what it reads as lists, so a read of another length than what it reads is
never chained: a row's commitment binds its length.

**The arguments' input nodes.** `scanOf` counts the input gates' bits against the templates' parameters' and the slot
entries' and takes each row, state and egress argument to be one input node's leaves from its first (`inputArg`). It
refuses an egress argument that is an ingress row's input node and slot entries read from one, and no other pair: two
arguments reading one input node leave the count room for an input node nothing reads. `argsApart` is the rest, the
row, state and egress arguments' input nodes pairwise distinct and none the slot entries'; the statements take it as a
condition on the Program, decided from the root's nodes.

* `ScanChainPinned`: an accepted scan's state chain pins its state gates, at every partition and design.
* `ScanCountsEntries`: an accepted scan's free inputs, less its state gates, are its slot entry gates, at most the
  slot entries' bits as `scanOf` counts them (`entryBits`).
-/

@[expose] public section

namespace Properties.NetworkScan

open NetworkCheck.Period

/-- The Definition root node `k` of `P` calls. -/
def calleeAt (P : Flock.Program.Prog) (k : ℕ) : Flock.Program.Defn :=
  P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!

/-- Bucket `t`'s ingress unit, as `scanLayout` reads it: the root node and member whose `row` is step `t`'s ingress,
the first template's one member at bucket 0 and the later ingress template's member `t - 1` at bucket `t ≥ 1`. -/
def ingressUnit (P : Flock.Program.Prog) (t : ℕ) : ℕ × ℕ :=
  if t = 0 then ((templateNodes P).headD 0, 0) else ((templateNodes P).getD 1 0, t - 1)

/-- Leaves `0 … n - 1` of parameter `i` of member `m` of root node `k`, as gates (`gateAt`). -/
def paramGates (P : Flock.Program.Prog) (k m i n : ℕ) : List ℕ :=
  (List.range n).map fun j => gateAt P (P.child P.rootScope k m) ⟨true, i, j⟩

/-- Leaves `o … o + n - 1` of the return of member `m` of root node `k`, as gates (`gateAt`). -/
def returnGates (P : Flock.Program.Prog) (k m o n : ℕ) : List ℕ :=
  (List.range n).map fun j => gateAt P P.rootScope ⟨false, k, m * (calleeAt P k).outLeaves + o + j⟩

/-- Component `i` of a tuple type as a run of its leaves: its first leaf and how many it has; no leaf where the type is
no tuple or has no component `i`. -/
def componentRun : Flock.Program.Ty → ℕ → ℕ × ℕ
  | .tuple fs, i => ((fs.extract 0 i).foldl (fun a t => a + t.leaves) 0, (fs[i]?.map (·.leaves)).getD 0)
  | _, _ => (0, 0)

/-- **The state bucket `t ≥ 1` reads** (`sin`): parameter 1 of its ingress unit, the later ingress template's member
`t - 1`, every leaf. -/
def sinGates (P : Flock.Program.Prog) (t : ℕ) : List ℕ :=
  let k := (templateNodes P).getD 1 0
  paramGates P k (t - 1) 1 (((calleeAt P k).params[1]!).2.leaves)

/-- **The state bucket `t` passes on**: component 1 of its ingress unit's return. -/
def stateOut (P : Flock.Program.Prog) (t : ℕ) : List ℕ :=
  let km := ingressUnit P t
  let r := componentRun (calleeAt P km.1).ret 1
  returnGates P km.1 km.2 r.1 r.2

/-- **The view bucket `t`'s egress unit reads** (`state`): parameter 0 of the egress template's member `t`, every
leaf. -/
def viewGates (P : Flock.Program.Prog) (t : ℕ) : List ℕ :=
  let k := (templateNodes P).getLast?.getD 0
  paramGates P k t 0 (((calleeAt P k).params[0]!).2.leaves)

/-- **The view bucket `t`'s ingress unit gives**: component 2 of its return. -/
def viewOut (P : Flock.Program.Prog) (t : ℕ) : List ℕ :=
  let km := ingressUnit P t
  let r := componentRun (calleeAt P km.1).ret 2
  returnGates P km.1 km.2 r.1 r.2

/-- **The scan's state gates** over the window's `T` buckets: each later bucket's `sin` and each bucket's egress
`state`. -/
def stateGates (P : Flock.Program.Prog) (s : Schedule) : Finset ℕ :=
  ((List.range s.window).flatMap fun t => (if t = 0 then [] else sinGates P t) ++ viewGates P t).toFinset

/-- **The scan's state chain holds on the gate values `val`**: at each bucket `t` of the window, a later bucket's
`sin` holds the state bucket `t - 1` passed on, and the egress unit's `state` holds the view bucket `t`'s ingress unit
gives, value for value and as many. -/
def Chained (P : Flock.Program.Prog) (s : Schedule) (val : ℕ → Bool) : Prop :=
  ∀ t < s.window, (0 < t → (sinGates P t).map val = (stateOut P (t - 1)).map val) ∧
    (viewGates P t).map val = (viewOut P t).map val

/-- **The scan's slot entry gates**: every gate a bucket's ingress unit reads through an entry port `prog<j>`, a lane
each, its parameters past the row at bucket 0 and past the row and the state at a later bucket. -/
def entryGates (P : Flock.Program.Prog) (s : Schedule) : Finset ℕ :=
  ((List.range s.window).flatMap fun t =>
    let km := ingressUnit P t
    let o := if t = 0 then 1 else 2
    (List.range s.lanes).flatMap fun j =>
      paramGates P km.1 km.2 (o + j) (((calleeAt P km.1).params[o + j]!).2.leaves)).toFinset

/-- **The slot entries' bits**, as `scanOf` counts them among the scan's input gates: `lanes · T / every` entries of
the entry ports' type. -/
def entryBits (P : Flock.Program.Prog) (s : Schedule) : ℕ :=
  s.lanes * (s.window / s.every) * bitsOf (((calleeAt P ((templateNodes P).headD 0)).params[1]!).2)

/-- The root node bucket 0's entry port of lane `j` reads first, where `scanOf` reads the slot entries' input node. -/
def entryNode (P : Flock.Program.Prog) (j : ℕ) : ℕ :=
  ((((P.defs[P.root]!).body.nodes[(templateNodes P).headD 0]!).args[1 + j]!).memberRef 0 0).i

/-- Whether the scan's row, state and egress arguments read input nodes of their own: the input nodes (`inputArg`) of
bucket 0's row, the later ingress template's row and state (where the window has a later bucket) and the egress
template's arguments are pairwise distinct, and none is a lane's slot entries' (`entryNode`). Decided from the root's
template nodes, at the cost of their arguments. -/
def argsApart (P : Flock.Program.Prog) (s : Schedule) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  let ts := templateNodes P
  let later := if s.window ≤ 1 then #[] else (nodes[ts.getD 1 0]!).args.extract 0 2
  let qs := ((nodes[ts.headD 0]!).args.extract 0 1 ++ later ++ (nodes[ts.getLast?.getD 0]!).args).toList.map
    (inputArg P)
  decide qs.Nodup && (List.range s.lanes).all fun j => !qs.contains (some (entryNode P j))

/-- **An accepted scan's state chain pins its state gates.** If `scanOf` accepts the scan `j` as the registered
Program `program` at the schedule `s`, its canonical text is `c` and decodes to `P`, whose row, state and egress
arguments read input nodes of their own (`argsApart`), then in every flattening `C` of `P`, at every partition and
design, the check `Chained P s` pins the state gates over the window's `T` steps (`NetTiming.CheckedUnique` at
`scanLayout`): two runs on which the chain holds, with one ingress through step `t`, the same inputs off the state
gates and the same held gates, have the same egress rows at step `t`. -/
def ScanChainPinned : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog) (Pt : Claims.Sampling.Partition) (d : Claims.Sampling.Design),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P → Flattens P C →
    argsApart P s = true →
      NetTiming.CheckedUnique C Pt d (scanLayout P s) s.window (stateGates P s) (Chained P s)

/-- **An accepted scan's free inputs, less its state gates, are its slot entries.** If `scanOf` accepts the scan `j`
as the registered Program `program` at a schedule `s` the check reads (`scheduleOf`), its canonical text is `c` and
decodes to `P`, whose row, state and egress arguments read input nodes of their own (`argsApart`), then in every
flattening `C` of `P` the free inputs over the window's `T` steps that are no state gate are exactly the slot entry
gates, and those number at most the slot entries' bits `scanOf` counts (`entryBits`). -/
def ScanCountsEntries : Prop :=
  ∀ (sj j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog),
    scheduleOf sj = .ok s → scanOf j program s = .ok sc → Flock.canon j = .ok c →
    Flock.Program.decode c.toUTF8 = .ok P → Flattens P C → argsApart P s = true →
      NetTiming.freeInputs C (scanLayout P s) s.window \ stateGates P s = entryGates P s ∧
        (entryGates P s).card ≤ entryBits P s

end Properties.NetworkScan
