module

public import Proofs.Flock.Soundness.TmplValue
public import Properties.Flock.EvalBool

/-!
# A lowered template computes its Definition (open)

`LowerEval` says that the template `Flock.Lower.lower` writes of Boolean Definition `d` computes `d`: column values
satisfying the rows of its unrolled net, with the constant at 1, put at each output port bit `k` the bit `k` of
`Prog.evalBool P d` on the values at the input port bits. It is the lowering's correctness, an open premise to be proved
(by `FlockSoundness.TmplValue.unroll_eval` per Definition, callee first), and no assumption: until then a guarantee over
the door that reads a template `lower` wrote takes it as a hypothesis.

`d` is a block's Boolean Definition: the word-level circuit's block, its gates' Boolean bodies inlined, as the block
Program gives it; its parameter bits are the block's input port bits in order, its return's bits its output port bits.
-/

@[expose] public section

namespace FlockSoundness.Assumptions

open Flock Flock.Tmpl

/-- **The template `lower` writes of Definition `d` computes `d`**: wherever the rows of its unrolled net hold with the
constant 1, output port bit `k` of every output leaf `k` of `d` is `Prog.evalBool P d`'s bit `k` on the input port bits
of `d`'s parameter leaves. -/
def LowerEval : Prop :=
  ∀ {P : Program.Prog} {d : ℕ} {inP outP : Array ℕ} {t : Tmpl} {n : Net}
    (_hl : Lower.lower P d inP outP = .ok t) (h : unroll t = .ok n) (_hu : n.unitLog ≤ 32) {z : ℕ → Bool}
    (_hs : (Rows.ofNet (TmplValue.unroll_built h)).Sat fun c => z c.val) (_hone : z t.constPos = true),
    ∀ k < (P.defs.getD d default).outLeaves,
      z (Lower.outBit t k) =
        (P.evalBool d ((List.range (P.defs.getD d default).paramLeaves).map fun i => z (Lower.inBit t i))).getD k false

end FlockSoundness.Assumptions
