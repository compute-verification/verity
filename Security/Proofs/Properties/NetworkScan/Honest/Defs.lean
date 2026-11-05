import Properties.NetworkScan
import Proofs.Properties.NetworkScan.Check
import Proofs.Properties.NetworkScan.Passes

/-!
# `ScanFits`' honest instance: a scan window's Program and its templates' circuit files

The instance at which `Properties.NetworkScan.scanFits_honest` shows each of `ScanFits`' hypotheses holds:

* `prog`: a scan window's root that calls one ingress template, `IngressScanFirst_v1` (a one-byte row and the
  program's first word, to a 16-bit word), and one egress template, `EgressBucket_v1` (a state word, to a 16-bit word
  and a one-byte row), once each; `sched`, a schedule of one-byte cells and one of everything else.
* `fIngress`, `fEgress`: each template's circuit file, one template instance a VU, whose unit net holds the template's
  parameters as 16-bit ports of one input word and returns computed rows that read only columns those ports hold. The
  matrices sit in thunks, which Bits' check never forces: it reads only the unit net's header.
* `loopId`: `lastLe`'s `while` loop one step at a time, through which `cbv` reads it.
-/

namespace Properties.NetworkScan.Honest

open Flock.Program

def u16 : Ty := .array 16 (.value 1)
def rowTy : Ty := .array 1 (.array 8 (.value 1))
def ingressRet : Ty := .tuple #[u16]
def egressRet : Ty := .tuple #[u16, rowTy]
def stateTy : Ty := .array 1 u16

def input1 : Defn :=
  { key := "Input1_v1", id := "Input1_v1", prim := true, params := #[], ret := .value 1, binding := some "run",
    gates := 1, paramLeaves := 0, outLeaves := 1 }

def not1 : Defn :=
  { key := "Not_v1", id := "Not_v1", prim := true, params := #[("x", .value 1)], ret := .value 1,
    gates := 1, paramLeaves := 1, outLeaves := 1 }

/-- A batch of `n` NOTs of one bit each, over the bits `part` reads. -/
def nots (n : Nat) (part : Part) : Node :=
  { form := .batch, fn := 1, args := #[{ ty := .array n (.value 1), seq := .ofParts #[part], dims := some (n, 1) }],
    out := .array n (.value 1), axes := #[some 0], n }

/-- The ingress template: its return is the NOTs of its row's bits, each twice. -/
def ingress : Defn :=
  { key := "IngressScanFirst_v1", id := "IngressScanFirst_v1", prim := false,
    params := #[("row", rowTy), ("prog0", u16)], ret := ingressRet,
    body := { params := #[("row", rowTy), ("prog0", u16)], retType := ingressRet,
              nodes := #[nots 16 (.strided true 0 0 8 0 1 16)], ret := .ofParts #[.affine false 0 0 16],
              offsets := #[0, 16] },
    gates := 16, paramLeaves := 24, outLeaves := 16 }

/-- The egress template: its return is the NOTs of its state's 16 bits, then of its first eight again. -/
def egress : Defn :=
  { key := "EgressBucket_v1", id := "EgressBucket_v1", prim := false, params := #[("state", stateTy)],
    ret := egressRet,
    body := { params := #[("state", stateTy)], retType := egressRet,
              nodes := #[nots 24 (.strided true 0 0 16 0 1 24)], ret := .ofParts #[.affine false 0 0 24],
              offsets := #[0, 24] },
    gates := 24, paramLeaves := 16, outLeaves := 24 }

/-- `n` input bits of the run. -/
def inputs (n : Nat) : Node := { form := .batch, fn := 0, args := #[], out := .array n (.value 1), n }

/-- An argument of type `ty`: node `k`'s first `c` bits. -/
def arg (ty : Ty) (k c : Nat) : Arg := { ty, seq := .ofParts #[.affine false k 0 c] }

def root : Defn :=
  { key := "ScanWindow_v1", id := "ScanWindow_v1", prim := false, params := #[], ret := .array 1 egressRet,
    body := { params := #[], retType := .array 1 egressRet,
              nodes := #[inputs 8, inputs 16, inputs 16,
                { form := .batch, fn := 2, args := #[arg rowTy 0 8, arg u16 1 16], out := .array 1 ingressRet,
                  axes := #[none, none], n := 1 },
                { form := .batch, fn := 3, args := #[arg stateTy 2 16], out := .array 1 egressRet,
                  axes := #[none], n := 1 }],
              ret := .ofParts #[.affine false 4 0 24], offsets := #[0, 8, 24, 40, 56, 80] },
    gates := 80, paramLeaves := 0, outLeaves := 24 }

def prog : Prog :=
  { defs := #[input1, not1, ingress, egress, root], index := {}, root := 4, programFn := "ScanWindow_v1",
    sha512 := .empty }

def sched : NetworkCheck.Period.Schedule := ⟨1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1⟩

/-- A unit net of one input word holding the 16-bit ports `ins`, and `n` computed rows, row `i` the XOR of the input
columns `r i` times the constant, returned from word 1 as the 16-bit ports `outs`. -/
def unitNet (ins outs : Array Nat) (n : Nat) (r : Nat → Array Nat) : Flock.Net :=
  let c := Flock.WORD + n
  { (default : Flock.Net) with
    constPos := c, inWords := 1, inGroups := #[⟨0, 1, ins⟩], outGroups := #[⟨1, 1, outs⟩]
    aT := Thunk.mk fun _ =>
      (Flock.Sparse.ofRows (Array.replicate Flock.WORD #[] ++ (Array.range n).map r) (c + 1)).toOption.getD default
    bT := Thunk.mk fun _ =>
      (Flock.Sparse.ofRows (Array.replicate Flock.WORD #[] ++ Array.replicate n #[c]) (c + 1)).toOption.getD default }

def port (name : String) (words : Nat) : Flock.Port := { name, words, chunks := 1, nb := 1, v2 := true }

/-- The ingress template's circuit file: row `i` reads bit `i % 8` of the row's port and bit `i` of `prog0`'s. -/
def fIngress : Flock.Circuit :=
  { (default : Flock.Circuit) with
    nets := #[("unit", unitNet #[16, 16] #[16] 16 fun i => #[i % 8, 16 + i])], upv := 1,
    ports := #[port "row" 1, port "prog0" 1, port "o" 1], outRowPorts := #[1], outPorts := #[("o", 1)],
    leavesIn := #[#[0, 1]], leavesOut := #[#[0]] }

/-- The egress template's circuit file: row `i` reads bit `i % 16` of the state's port. -/
def fEgress : Flock.Circuit :=
  { (default : Flock.Circuit) with
    nets := #[("unit", unitNet #[16] #[16, 16] 32 fun i => #[i % 16])], upv := 1,
    ports := #[port "state" 1, port "o" 2], outRowPorts := #[1], outPorts := #[("o", 2)],
    leavesIn := #[#[0]], leavesOut := #[#[0, 1]] }

/-- `lastLe`'s `while` loop, one step at a time: `cbv` reads it through this, since the loop's own recursion is
opaque. -/
theorem loopId {β : Type} (b : β) (f : Unit → β → Id (ForInStep β)) :
    Lean.Loop.forIn (m := Id) ⟨⟩ b f = match (f () b).run with
      | .done v => pure v
      | .yield v => Lean.Loop.forIn (m := Id) ⟨⟩ v f := by
  rw [Lean.Loop.forIn_eq_of_monadTail]
  rfl

end Properties.NetworkScan.Honest
