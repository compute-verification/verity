module

public import Spot.Restrict
public import Spot.BlockFront
public import Spot.Edges
public import Flock.Test.Compose
public import Properties.Flock.Circuit.Chain
meta import Spot.Restrict
meta import Spot.BlockFront
meta import Spot.Edges
meta import Flock.Test.Compose
meta import Properties.Flock.Circuit.Chain

/-! Scratch, not for merge: `BlockEval`'s left side (`Prog.evalBool` of `portedOf`'s root lifted) on the port bits, at
the word-parameter case and at E. Run: `git apply security/Proofs/Scratch/lakefile.patch && verity lean try
security/Proofs/Scratch/ChainEval.lean`. -/

@[expose] public section

namespace Scratch.ChainEval

open Flock Flock.Program

def D_unused : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",1]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"fn":0,"form":"primitive","out":0},{"args":[],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"affine":["n",0,0,1]},"type":0},{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":3,"form":"call","out":1}],"params":[],"ret":{"affine":["n",2,0,1]},"ret_type":["Value",1]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Tuple",[["Value",32],["Value",1]]],["Array",2,["Value",32]],["Array",2,["Value",1]],["Array",2,["Tuple",[["Value",32],["Value",1]]]]]}"#

def D_field : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",3,["Value",1]]],["y",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,3]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Array",3,["Value",1]],["Value",1]]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":3,"out":1},{"args":[],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,3]},"type":1},{"refs":{"affine":["n",1,0,1]},"type":0}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,4]},"ret_type":["Tuple",[["Array",3,["Value",1]],["Value",1]]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",3,["Value",1]],["Tuple",[["Array",3,["Value",1]],["Value",1]]]]}"#

def bitsOf (v n : Nat) : Array Bool := Test.Lower.bits v n

/-- `BlockEval`'s left side on `d`'s units `us`: `Prog.evalBool` of `portedOf`'s root lifted (`Lift.lift`), on the input
port bits `x` (each port's value low bit first, in the front's order). -/
def chainEval (d : String) (us : Array Nat) (x : List Bool) : String :=
  match decode d.toUTF8 >>= fun P => Spot.Restrict.portedOf P us >>= fun Q => Lift.lift Q Q.root with
  | .ok (L, e) => s!"evalBool (lift (portedOf)) on {x.length} port bits, {(L.defs[e]!).paramLeaves} parameter leaves: {Prog.evalBool L e x}"
  | .error err => "REFUSED: " ++ err

-- C(a, b) = Not b at a = 2, b = 0: C's output at the front's out port is [true]
#eval chainEval D_unused #[0] ((bitsOf 2 32).toList ++ [false])

-- E(x, y) = (x, Not y) at x = 5, y = 0: C's outputs at the front's out ports are x's 3 bits then Not y, [true, false, true, true]
#eval chainEval D_field #[0] ((bitsOf 5 3).toList ++ [false])

end Scratch.ChainEval
