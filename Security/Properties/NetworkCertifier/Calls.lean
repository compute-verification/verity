import NetworkCertifier.Calls

/-!
# The certifier's calls: what its Lean code (`NetworkCertifier.Calls`) does

The property of `NetworkCertifier.Calls.run`, the certifier program's window of calls (`calls.Calls` in
`verity/core/protocols/certifier/calls.py`, held to it by `CallsDifftest`), stated against its functions as they run:
`RefusesPastCallBits`, that a forward whose words root the certifier accepts stayed within its bits. From any call of
the forward the certifier took to its words root, the bits of every bucket it counted, with the node's release at the
boundary, are at most `cap · Σ macs / fwdMacs` over the forward's calls, compared in integers. A forward past them is
refused. `Proofs.Properties.NetworkCertifier.Calls` proves it.

The predicates below read a window's events beside the certifier's answers (`answered`): the bits of a bucket it
counted (`countedBits`, its timing bits and its real frames' content bits) and the MACs of a call it took (`callMacs`),
in the program's own terms (`timingBits`, `Shape.macs`). None restates the count.

Its two lemmas read the instances at the end: the honest one, that the certifier accepts a forward of one call at its
words root (`honestEvents`); the nontrivial one, that a certifier taking every event (`acceptsAll`) accepts a forward
past its bits (`pastEvents`).
-/

open NetworkCertifier.Calls

namespace Properties.NetworkCertifier

/-- The window's events from its start with entropy `e`, each beside the certifier's answer to it. -/
def answered (cap : Cap) (releaseBits : Nat) (e : List Nat) (evs : List Event) : List (Event × CallAnswer) :=
  evs.zip (run cap releaseBits (State.init e) evs).1

/-- Whether an answer is a refusal. -/
def refused : CallAnswer → Bool
  | .refusal => true
  | _ => false

/-- The bits of a bucket the certifier counted: its timing bits at its allocation and its real frames' content bits.
An event of another kind, or a bucket it refused, carries none. -/
def countedBits (x : Event × CallAnswer) : Nat :=
  match x.1 with
  | .bucket r frames => if refused x.2 then 0 else timingBits r + frames.sum
  | _ => 0

/-- The MACs of a call of forward `k` the certifier took: an A root of `k` it didn't refuse. Any other event brings
none. -/
def callMacs (k : Nat) (x : Event × CallAnswer) : Nat :=
  match x.1 with
  | .aRoot f _ _ s => if f = k ∧ refused x.2 = false then s.macs else 0
  | _ => 0

/-- `x` is a call of forward `k` the certifier took: an A root of `k` it didn't refuse. -/
def TookCall (k : Nat) (x : Event × CallAnswer) : Prop :=
  ∃ c root s, x.1 = .aRoot k c root s ∧ refused x.2 = false

/-- Forward `k`, whose words root is event `i` of `xs`, stayed within its bits: the certifier took a call of it before
`i`, and from each call of it the certifier took to `i`, the bits of the buckets it counted, with the node's release
`releaseBits`, are at most `cap` bits a token over `fwdMacs` MACs a token of the forward's calls before `i`:
`(bits + releaseBits) · fwdMacs ≤ cap · Σ macs`. -/
def WithinBits (cap : Cap) (releaseBits : Nat) (xs : List (Event × CallAnswer)) (i k : Nat) : Prop :=
  (∃ j < i, ∃ x, xs[j]? = some x ∧ TookCall k x) ∧
    ∀ j < i, ∀ x, xs[j]? = some x → TookCall k x →
      ((((xs.take i).drop (j + 1)).map countedBits).sum + releaseBits) * cap.fwdMacs ≤
        cap.cap * ((xs.take i).map (callMacs k)).sum

/-- **A forward the certifier accepts at its words root stayed within its bits** (`WithinBits`): a forward past them is
refused. -/
def RefusesPastCallBits : Prop :=
  ∀ (cap : Cap) (releaseBits : Nat) (e : List Nat) (evs : List Event) (i k : Nat) (root : List Nat),
    (answered cap releaseBits e evs)[i]? = some (.wordsRoot k root, .verdict .accepted) →
      WithinBits cap releaseBits (answered cap releaseBits e evs) i k

/-! ## The witnesses' instances

A forward of one call the certifier accepts, one past its bits, and a certifier that takes every event: what the
property's honest and nontrivial witnesses (`Proofs.Properties.NetworkCertifier.Calls`) are stated at. -/

/-- A call of one row, one column and an inner dimension of 1: 16 rows and 8 columns committed in one step, 120 bytes of
entropy. -/
def witnessShape : Shape := ⟨1, 1, 1⟩

/-- A 64-byte root. -/
def witnessRoot : List Nat := List.replicate 64 0

/-- A cap of 32 bits a token over the call's MACs a token. -/
def witnessCap : Cap := ⟨32, witnessShape.macs⟩

/-- A forward of one call, one bucket at allocation 1 with one 8-bit frame, and its words root: the bucket's 1 + 8 bits
and the release's 23 make 32, its cap. -/
def honestEvents : List Event :=
  [.aRoot 0 0 witnessRoot witnessShape, .bucket 1 [8], .wordsRoot 0 witnessRoot]

/-- The node's release at the boundary in the honest window. -/
def honestReleaseBits : Nat := 23

/-- The certifier's entropy in the honest window: the call's 120 bytes, each 0. -/
def honestEntropy : List Nat := List.replicate 120 0

/-- A forward of one call, one bucket at allocation 1 with a 1,000-bit frame, and its words root: 1,001 bits, past its
cap of 32. -/
def pastEvents : List Event :=
  [.aRoot 0 0 witnessRoot witnessShape, .bucket 1 [1000], .wordsRoot 0 witnessRoot]

/-- A certifier that takes every event: each A root answered with an empty release, each bucket counted, each words
root accepted, and the close with no verdicts. -/
def acceptsAll (evs : List Event) : List (Event × CallAnswer) :=
  evs.map fun ev => (ev, match ev with
    | .aRoot .. => .release []
    | .bucket .. => .counted
    | .wordsRoot .. => .verdict .accepted
    | .close => .verdicts [])

end Properties.NetworkCertifier
