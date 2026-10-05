/-
The design rules, as properties of the code: each a `def … : Prop` over the machines or the world, with the theorem
that the code has it. Proofs are left to come (`sorry`); every statement here typechecks against the executable
verifier alone.

1. Each party owns its state: a step reads only its own state, the frame and its fresh bytes (the type of
   `Machine.step`), and an event changes no state but its receiver's (`OwnsState`).
2. Each party refuses an out-of-order or duplicate message (`Machine.InOrder`, one theorem about `Core.seq`).
3. Coins only after their commitment: the coin server answers a slot only in the step that took the slot's
   commitment, an inner round's coins are that step's fresh bytes, and no slot is answered twice (`CoinsAfterCommit`).
4. Salts never repeat: the proxy's and the ZK layer's salts are pairwise distinct at every state they reach
   (`SaltsFresh`).
5. Nothing leaves the boundary before its check: only the send gate sends out of the boundary (`OnlyGateLeaves`), it
   sends a frame out only in the step whose check passed on it (`GateChecks`), and under the egress premise every frame
   that left the boundary in a run came from the send gate (`Egress`).
-/
import Run.World

namespace Run.Rules

/-! ## 1. Each party owns its state -/

def OwnsState (S : Setting) : Prop :=
  ∀ (w : World S) (ev : Event) (r : ByteArray) (q : Party), w.touches ev ≠ some q → (w.step ev r).states q = w.states q

theorem ownsState (S : Setting) : OwnsState S := sorry

/-! ## 2. In order, once -/

/-- The frames a machine accepted from each sender carry `seq` 0, 1, 2, …: none out of order, none twice. -/
def _root_.Run.Machine.InOrder (M : Machine) : Prop :=
  ∀ (ins : List (Frame × ByteArray)) (src : Node),
    let acc := (M.run M.init ins).2.filter (·.1.src == src)
    acc.map (·.1.seq) = List.range acc.length

theorem seq_inOrder (C : Core) (self : Node) : Machine.InOrder (C.seq self) := sorry

/-- Every party of the honest run is a `Core.seq`, so every party has it. -/
theorem honest_inOrder (A : Agreement) (p : Party) : Machine.InOrder (Roles.honest A p) := sorry

/-! ## 3. Coins only after their commitment -/

/-- The slot a message commits to. -/
def commitsTo : Body → Option Slot
  | .commit s _ => some s
  | .outer k _ => some (.outer k)
  | _ => none

/-- The slot a message answers. -/
def answers : Body → Option Slot
  | .draw _ => some .registration
  | .coins s _ => some s
  | _ => none

def CoinsAfterCommit (M : Machine) : Prop :=
  ∀ ins : List (Frame × ByteArray),
    let tr := (M.run M.init ins).2
    (∀ x ∈ tr, ∀ o ∈ x.2.2, ∀ s, answers o.body = some s → commitsTo x.1.body = some s) ∧
    (∀ x ∈ tr, ∀ o ∈ x.2.2, ∀ b r c, o.body = .coins (.inner b r) c → c = x.2.1) ∧
    (tr.flatMap fun x => x.2.2.filterMap (answers ·.body)).Nodup

theorem coinServer_coinsAfterCommit (A : Agreement) : CoinsAfterCommit (CoinServer.machine A) := sorry

/-! ## 4. Salts never repeat -/

/-- A core's state after a run of its wired machine. -/
def _root_.Run.Core.after (C : Core) (self : Node) (ins : List (Frame × ByteArray)) : C.State :=
  let M := C.seq self
  let s : Seq C.State := (M.run M.init ins).1
  s.core

def SaltsFresh (A : Agreement) : Prop :=
  ∀ ins : List (Frame × ByteArray),
    Proxy.nodupBytes ((Proxy.core A).after (.party .proxy) ins).salts = true ∧
    ∀ m salts, ((ZkLayer.core A).after (.party .zkLayer) ins).drawn = some (m, salts) → Proxy.nodupBytes salts = true

theorem saltsFresh (A : Agreement) : SaltsFresh A := sorry

/-! ## 5. Nothing leaves the boundary before its check -/

def leavesFrom (src : Party) (dst : Node) : Bool := (Node.party src).inside && !dst.inside

def NeverLeaves (p : Party) (step : Node → Body → ByteArray → Option (List (Node × Body))) : Prop :=
  ∀ src b r outs, step src b r = some outs → ∀ o ∈ outs, leavesFrom p o.1 = false

def OnlyGateLeaves (A : Agreement) : Prop :=
  (∀ s : Proxy.State, NeverLeaves .proxy fun src b r => (Proxy.step A s src b r).toOption.map (·.2)) ∧
  (∀ s : ZkLayer.State, NeverLeaves .zkLayer fun src b r => (ZkLayer.step A s src b r).toOption.map (·.2)) ∧
  (∀ s : Recheck.State, NeverLeaves .recheck fun src b r => (Recheck.step A s src b r).toOption.map (·.2))

theorem onlyGateLeaves (A : Agreement) : OnlyGateLeaves A := sorry

/-- What the send gate's step checked before sending `out` out of the boundary, on taking `b` from `src`. -/
def Released (A : Agreement) (s : SendGate.State) (src : Node) : Body → Body → Prop
  | .commit slot hs, out =>
    src = .party .proxy ∧ out = .commit slot hs ∧ ¬ slot ∈ s.released ∧ hs.size = A.commitCount slot
  | .outer k msg, out =>
    src = .worker .outerProver ∧ out = .outer k msg ∧ k = s.outer.size ∧ SendGate.canonical A s k msg = true ∧
      k ≠ SendGate.level0Round
  | .level0 root, out =>
    src = .party .recheck ∧ ∃ k msg, s.held = some (k, msg) ∧ SendGate.level0Root msg = root ∧ out = .outer k msg
  | .finish record proofs, out =>
    src = .worker .outerProver ∧ out = .finish record proofs ∧
      Statement.verify A { s.pub with record, proofs, finished := true } = .ok ()
  | _, _ => False

def GateChecks (A : Agreement) : Prop :=
  ∀ (s : SendGate.State) (src : Node) (b : Body) (r : ByteArray) (s' : SendGate.State) (outs : List (Node × Body)),
    SendGate.step A s src b r = .ok (s', outs) → ∀ o ∈ outs, leavesFrom .sendGate o.1 = true → Released A s src b o.2

theorem gateChecks (A : Agreement) : GateChecks A := sorry

/-- In a run of the zero-knowledge setting, every frame that left the boundary came from the send gate. -/
def Egress (A : Agreement) : Prop :=
  ∀ evs : List (Event × ByteArray), ∀ x ∈ ((World.init (Setting.zk A)).runOn evs).trace,
    x.2.leaves = true → x.2.src = .party .sendGate

theorem egress (A : Agreement) : Egress A := sorry

end Run.Rules
