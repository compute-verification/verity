module

public import Spot.Partition
public import Flock.Test.WiredPorts
meta import Spot.Partition
meta import Flock.Test.WiredPorts

/-! Scratch, not for merge: R8 (`Partition.sharedInputs`/`readAlone`) on a run Input node read by two proof blocks,
under a template query, under the empty query's fallback (`Q_word`) and with the node's gates held. Run:
`git apply security/Proofs/Scratch/lakefile.patch && verity lean try security/Proofs/Scratch/ShareEval.lean`. -/

@[expose] public section

namespace Scratch.ShareEval

open Lean (Json)
open Flock
open Flock.Test.WiredPorts (regP runP rw2 r16)

def tq : Json :=
  Json.mkObj [("name", "Q_template_instance"), ("version", Lean.toJson (0 : Nat)),
    ("params", Json.mkObj [("template", "T")])]

def withNodes (nodes : Array Program.Node) (offsets : Array Nat) : Program.Prog :=
  let root := runP.defs[2]!
  let body : Program.Body := { root.body with nodes := nodes, offsets := offsets }
  let defs := runP.defs.set! 2 { root with body := body }
  { runP with defs := defs, sha512 := ByteArray.mk (Array.replicate 64 0xab) }

def inputNode : Program.Node := { form := .batch, fn := 0, args := #[], out := .array 4 rw2, n := 4 }

/-- T's two members each read their own row of the run input (mapped, one leaf once). -/
def mappedP : Program.Prog := withNodes #[inputNode,
  { form := .batch, fn := 1, out := .array 2 r16, n := 2, axes := #[some 0],
    args := #[{ ty := .array 2 rw2, seq := Program.Seq.ofParts #[.affine false 0 0 4], dims := some (2, 2) }] }] #[0, 4]

/-- T's two members both read row 0 of the run input: a broadcast argument, no `dims`. -/
def broadcastP : Program.Prog := withNodes #[inputNode,
  { form := .batch, fn := 1, out := .array 2 r16, n := 2, axes := #[none],
    args := #[{ ty := rw2, seq := Program.Seq.ofParts #[.affine false 0 0 2] }] }] #[0, 4]

/-- Two calls of T, each a proof block, both reading row 0 of the run input. -/
def twoCallsP : Program.Prog := withNodes #[inputNode,
  { form := .call, fn := 1, out := r16, args := #[{ ty := rw2, seq := Program.Seq.ofParts #[.affine false 0 0 2] }] },
  { form := .call, fn := 1, out := r16, args := #[{ ty := rw2, seq := Program.Seq.ofParts #[.affine false 0 0 2] }] }]
  #[0, 4, 5]

def sp (P : Program.Prog) : Spot.Program := { desc := .ofProg P, prog := P }

def ranges (P : Program.Prog) (q : Json) : String :=
  match Spot.Partition.evaluate (sp P) q with
  | .ok rs => s!"ok {rs.map fun r => (r.node, r.template, r.first, r.count)}"
  | .error e => s!"error {e}"

#eval IO.println s!"SHARE mapped ranges={ranges mappedP tq} shared={Spot.Partition.shared? (sp mappedP) tq}"
#eval IO.println s!"SHARE broadcast ranges={ranges broadcastP tq} shared={Spot.Partition.shared? (sp broadcastP) tq}"
#eval IO.println s!"SHARE twoCalls ranges={ranges twoCallsP tq} shared={Spot.Partition.shared? (sp twoCallsP) tq}"
#eval IO.println s!"SHARE twoCalls held readAlone={Spot.Partition.readAlone (sp twoCallsP) tq #[(0, 1000)]}"
#eval IO.println s!"SHARE twoCalls Q_word ranges={ranges twoCallsP (Json.mkObj [])} readAlone={Spot.Partition.readAlone (sp twoCallsP) (Json.mkObj [])}"
#eval IO.println s!"SHARE broadcast Q_word ranges={ranges broadcastP (Json.mkObj [])} readAlone={Spot.Partition.readAlone (sp broadcastP) (Json.mkObj [])}"

#guard (Spot.Partition.evaluate (sp mappedP) tq).toOption.isSome
#guard Spot.Partition.readAlone (sp mappedP) tq
#guard Spot.Partition.shared? (sp broadcastP) tq == #[0]
#guard Spot.Partition.shared? (sp twoCallsP) tq == #[0]


end Scratch.ShareEval
