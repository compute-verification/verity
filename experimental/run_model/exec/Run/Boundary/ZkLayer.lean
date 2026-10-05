/-
The outer ZK layer: the outer session's masks and salts come from it, not from the GPU rack. Once the proxy hands over
V*'s witness it draws them fresh, once, and gives them to the outer prover (to prove with), the recheck (to recompute
the level-0 commitment) and the send gate (to check what the prover sends). Its salts never repeat one another.
-/
import Run.Boundary.Proxy

namespace Run.ZkLayer

structure State where
  /-- the masks and salts it drew, once -/
  drawn : Option (ByteArray × Array Salt) := none
  deriving Inhabited

def need (A : Agreement) : Nat := A.maskBytes + Proxy.saltBytes * A.maskSalts

def step (A : Agreement) (s : State) (src : Node) (b : Body) (r : ByteArray) :
    Except String (State × List (Node × Body)) := do
  if src != .party .proxy then throw "ZK layer: it takes the proxy's witness only"
  let .witness _ _ := b | throw "ZK layer: not a witness"
  if s.drawn.isSome then throw "ZK layer: masks are drawn once"
  let m := r.extract 0 A.maskBytes
  let salts := Proxy.saltsOf (r.extract A.maskBytes r.size) A.maskSalts
  if !Proxy.nodupBytes salts then throw "ZK layer: a salt repeats; refused"
  let out := Body.masks m salts
  return ({ drawn := some (m, salts) }, [(.worker .outerProver, out), (.party .recheck, out), (.party .sendGate, out)])

def core (A : Agreement) : Core where
  State := State
  init := {}
  fresh s _ b := match s.drawn, b with
    | none, .witness _ _ => need A
    | _, _ => 0
  step := step A
  halted s := s.drawn.isSome

def machine (A : Agreement) : Machine := (core A).seq (.party .zkLayer)

end Run.ZkLayer
