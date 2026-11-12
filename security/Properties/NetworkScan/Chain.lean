module

public import Properties.NetworkScan
public import Definitions.NetworkCertifier.PeriodChecked

/-!
# The scan's two ties and its cut: what `NetworkCheck.Period.scanOf`'s acceptance gives the checked inputs

The scan's Program is flattened with no wire between its units: bucket `t`'s ingress unit reads the state the bucket
before passed on through its `sin` parameter, and bucket `t`'s egress unit reads the view its ingress unit gives through
its `state` parameter, and both parameters are input nodes of the root (`scanOf`, the generator form). So in the
flattening those gates are free inputs (`NetTiming.freeInputs`), beside the slot entries the ingress units read through
their entry ports `prog<j>`.

**Two ties**, each a property of a run's gate values:

* tie 1, the state chain (`StateTied`): each later bucket's `sin` holds what the unit before returned as its state;
* tie 2, the egress view (`ViewTied`): each egress `state` holds what its bucket's ingress unit returned as its view.

A run on which both hold has its `sin` and `state` gates (`sinSet`, `viewSet`) fixed by the ingress and the slot
entries: they are checked inputs (`NetTiming.CheckedUnique`), which the period's leak bound with checked inputs leaves
out of its count (`NetTiming.Guarantees.PeriodLeakChecked`, at `chk := tiesOf s`), so the count is the slot entries'.

**Each tie's premise is at the network certifier's run.** That an accepted run has a tie is that tie's premise, one
each (`StateTieHeld`, `ViewTieHeld`), which no statement here proves. Each is `NetTiming.CheckedWitness` at the run the
period's leak relation accepts at, the trace's run with its ingress read from the network certifier's rows
(`NetTiming.certifierRun`), which is how `PeriodLeakChecked` takes its two checks, each with its own miss in the
bound's error. A tie held only on the trace's own run (`NetTiming.traceRun`) says nothing of that run: the trace's
ingress gates may hold other values than the network certifier's rows.

With neither tie the egress rows are still a function of the `state` gates alone, so every other free input is checked
under no check at all (`ScanViewCut`, `PeriodLeakChecked` at `chk := cutOf s` and the checks that hold of every run),
and the count is the views'.

**The gates.** The ingress Definitions return a verdict, the state and the view, in that order, so the state a unit
passes on is component 1 of its return (`stateOut`) and the view it gives component 2 (`viewOut`), each a run of the
return's leaves (`componentRun`). A later bucket's `sin` is parameter 1 of its unit (`sinGates`) and an egress unit's
`state` its parameter 0 (`viewGates`), as `scanOf` holds the templates' parameters to their names. A tie equates the
values of each read with the values of what it reads as lists, so a read of another length than what it reads is
never tied: a row's commitment binds its length.

**The input nodes.** `scanOf` refuses a scan whose input gates are not read as the count reads them. Each row, state
and egress argument is all the leaves of an input node of its own, none the slot entries'
(`NetworkCheck.Period.argsApart`, `inputsRead`); the slot entries' node holds the entries and no other leaf, and the
root has no other input node; and the state and view an ingress unit returns have the leaves of the `sin` and `state`
they are tied to (`ingressReturn`). So the statements here take no condition on the Program beside `scanOf`'s
acceptance.

* `ScanViewCut`: at an accepted scan, the free inputs other than the `state` gates are pinned under no check.
* `ScanChainPinned`: an accepted scan's two ties pin its `sin` and `state` gates, at every partition and design.
* `ScanCountsEntries`: an accepted scan's free inputs, less those gates, are among its slot entry gates, at most the
  slot entries' bits `scanOf` counts (`Scan.entries`).
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

/-- **Tie 1's gates** over the window's `T` buckets: each later bucket's `sin`. -/
def sinSet (P : Flock.Program.Prog) (s : Schedule) : Finset ℕ :=
  ((List.range s.window).flatMap fun t => if t = 0 then [] else sinGates P t).toFinset

/-- **Tie 2's gates** over the window's `T` buckets: each bucket's egress `state`. -/
def viewSet (P : Flock.Program.Prog) (s : Schedule) : Finset ℕ :=
  ((List.range s.window).flatMap (viewGates P)).toFinset

/-- **The views' gates, counted**: the window's `T` buckets' egress `state` parameters' leaves, which `viewSet` holds
no more than. -/
def viewCount (P : Flock.Program.Prog) (s : Schedule) : ℕ :=
  s.window * ((calleeAt P ((templateNodes P).getLast?.getD 0)).params[0]!).2.leaves

/-- **Tie 1, the state chain, holds on the gate values `val`**: each later bucket `t` of the window reads as its `sin`
the state bucket `t - 1` passed on, value for value and as many. -/
def StateTied (P : Flock.Program.Prog) (s : Schedule) (val : ℕ → Bool) : Prop :=
  ∀ t, 0 < t → t < s.window → (sinGates P t).map val = (stateOut P (t - 1)).map val

/-- **Tie 2, the egress view, holds on the gate values `val`**: each bucket `t` of the window's egress unit reads as
its `state` the view bucket `t`'s ingress unit gives, value for value and as many. -/
def ViewTied (P : Flock.Program.Prog) (s : Schedule) (val : ℕ → Bool) : Prop :=
  ∀ t < s.window, (viewGates P t).map val = (viewOut P t).map val

/-- A statement's tie 1: its descriptor's (`StateTied`), as `layOf` reads its layout; no condition on a statement whose
descriptor doesn't decode. -/
def stateTiedAt (s : Schedule) (x : Claims.Sampling.AuditStatement) (val : ℕ → Bool) : Prop :=
  match Flock.Program.decode ⟨x.desc.toArray⟩ with
  | .ok P => StateTied P s val
  | .error _ => True

/-- A statement's tie 2: its descriptor's (`ViewTied`), as `layOf` reads its layout; no condition on a statement whose
descriptor doesn't decode. -/
def viewTiedAt (s : Schedule) (x : Claims.Sampling.AuditStatement) (val : ℕ → Bool) : Prop :=
  match Flock.Program.decode ⟨x.desc.toArray⟩ with
  | .ok P => ViewTied P s val
  | .error _ => True

/-- **A statement's tied gates**: its descriptor's `sin` and `state` gates, the inputs the two ties check; none where
the descriptor doesn't decode. -/
def tiesOf (s : Schedule) (x : Claims.Sampling.AuditStatement) : Finset ℕ :=
  match Flock.Program.decode ⟨x.desc.toArray⟩ with
  | .ok P => sinSet P s ∪ viewSet P s
  | .error _ => ∅

/-- **A statement's free inputs off its views**: every free input of its gates over the window's `T` steps but its
descriptor's `state` gates, the inputs `ScanViewCut` pins under no check; none where the descriptor doesn't decode. -/
def cutOf (s : Schedule) (x : Claims.Sampling.AuditStatement) : Finset ℕ :=
  match Flock.Program.decode ⟨x.desc.toArray⟩ with
  | .ok P => NetTiming.freeInputs x.prog (scanLayout P s) s.window \ viewSet P s
  | .error _ => ∅

section Ties

open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **Tie 1 is held at `x`**, a premise: at every prover, an accepted run whose extracted witness has no proof block
sampled `all` incorrect has the state chain on its run at the network certifier's rows (`NetTiming.certifierRun` at the
scan's layout `layOf s` over the window's `T` steps and the rows' openings `tops`), except with probability `ε x`
(`NetTiming.CheckedWitness` at `stateTiedAt`). It is `PeriodLeakChecked`'s premise for its first check at the scans'
instance. -/
def StateTieHeld (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (K : ℕ) (s : Schedule)
    (tops : Claims.Sampling.AuditStatement → List (Claims.Sampling.Opening Bool)) (ε : Claims.Msg → ℝ)
    (x : Claims.Msg) : Prop :=
  NetTiming.CheckedWitness H D
    (NetTiming.certifierRun H D.commit 1 K s.allocation (layOf s) (fun _ => s.window) tops) (stateTiedAt s) ε x

/-- **Tie 2 is held at `x`**, a premise of its own: at every prover, an accepted run whose extracted witness has no
proof block sampled `all` incorrect has the egress view on its run at the network certifier's rows
(`NetTiming.certifierRun`, as `StateTieHeld` reads it), except with probability `ε x` (`NetTiming.CheckedWitness` at
`viewTiedAt`). It is `PeriodLeakChecked`'s premise for its second check at the scans' instance. -/
def ViewTieHeld (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (K : ℕ) (s : Schedule)
    (tops : Claims.Sampling.AuditStatement → List (Claims.Sampling.Opening Bool)) (ε : Claims.Msg → ℝ)
    (x : Claims.Msg) : Prop :=
  NetTiming.CheckedWitness H D
    (NetTiming.certifierRun H D.commit 1 K s.allocation (layOf s) (fun _ => s.window) tops) (viewTiedAt s) ε x

end Ties

/-- **The scan's slot entry gates**: every gate a bucket's ingress unit reads through an entry port `prog<j>`, a lane
each, its parameters past the row at bucket 0 and past the row and the state at a later bucket. -/
def entryGates (P : Flock.Program.Prog) (s : Schedule) : Finset ℕ :=
  ((List.range s.window).flatMap fun t =>
    let km := ingressUnit P t
    let o := if t = 0 then 1 else 2
    (List.range s.lanes).flatMap fun j =>
      paramGates P km.1 km.2 (o + j) (((calleeAt P km.1).params[o + j]!).2.leaves)).toFinset

/-- **With neither tie, an accepted scan's egress views are the cut.** If `scanOf` accepts the scan `j` as the
registered Program `program` at the schedule `s`, its canonical text is `c` and decodes to `P`, then in every
flattening `C` of `P`, at every partition and design, the free inputs other than tie 2's gates are pinned under no
check (`NetTiming.CheckedUnique` at `scanLayout` with the check that holds of every run): two runs with the same
values at the egress units' `state` gates and the same held gates have the same egress rows at every step. -/
def ScanViewCut : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog) (Pt : Claims.Sampling.Partition) (d : Claims.Sampling.Design),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P → Flattens P C →
      NetTiming.CheckedUnique C Pt d (scanLayout P s) s.window
        (NetTiming.freeInputs C (scanLayout P s) s.window \ viewSet P s) (fun _ => True)

/-- **An accepted scan's two ties pin its `sin` and `state` gates.** If `scanOf` accepts the scan `j` as the
registered Program `program` at the schedule `s`, its canonical text is `c` and decodes to `P`, then in every
flattening `C` of `P`, at every partition and design, tie 1 and tie 2 together pin tie 1's and tie 2's gates over the
window's `T` steps (`NetTiming.CheckedUnique` at `scanLayout`): two runs on which both ties hold, with one ingress
through step `t`, the same inputs off those gates and the same held gates, have the same egress rows at step `t`. -/
def ScanChainPinned : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog) (Pt : Claims.Sampling.Partition) (d : Claims.Sampling.Design),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P → Flattens P C →
      NetTiming.CheckedUnique C Pt d (scanLayout P s) s.window (sinSet P s ∪ viewSet P s)
        (fun val => StateTied P s val ∧ ViewTied P s val)

/-- **An accepted scan's free inputs, less its tied gates, are slot entries.** If `scanOf` accepts the scan `j` as
the registered Program `program` at a schedule `s` the check reads (`scheduleOf`), its canonical text is `c` and
decodes to `P`, then in every flattening `C` of `P` the free inputs over the window's `T` steps that are neither tie's
gates are slot entry gates, and those number at most the slot entries' bits `scanOf` counts (`Scan.entries`). -/
def ScanCountsEntries : Prop :=
  ∀ (sj j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (C : Claims.Sampling.Prog),
    scheduleOf sj = .ok s → scanOf j program s = .ok sc → Flock.canon j = .ok c →
    Flock.Program.decode c.toUTF8 = .ok P → Flattens P C →
      NetTiming.freeInputs C (scanLayout P s) s.window \ (sinSet P s ∪ viewSet P s) ⊆ entryGates P s ∧
        (entryGates P s).card ≤ sc.entries

end Properties.NetworkScan
