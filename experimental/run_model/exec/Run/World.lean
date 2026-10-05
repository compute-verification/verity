/-
The world: every party's state, the frames in flight, and the environment's events.

The environment delivers any frame in flight, in any order, or never (a delay is a frame not yet delivered); drops any;
and sends any frame from a node it plays. Which nodes it plays is the statement's: soundness trusts the auditor's four
parties and nothing else (the boundary is the developer's, so it is the adversary's too); zero knowledge trusts the
boundary's four and nothing else. A frame from a trusted party is authentic: the environment can't send one in its
name. A frame delivered to a node the environment plays goes to the environment. Every untrusted worker's output is an
`inject` from that worker: the environment's choice.

The world is executable and deterministic given the events and each step's fresh bytes (`World.runOn`); the statements
put a distribution on the bytes (`RunModel.Protocol.Run`).
-/
import Run.Auditor
import Run.Boundary

namespace Run

def Roles.honest (A : Agreement) (p : Party) : Machine :=
  match h : p.side with
  | .auditor => Auditor.machine A p h
  | .boundary => Boundary.machine A p h

structure Setting where
  roles : Party → Machine
  trusts : Party → Bool
  /-- the egress premise: a worker's frame reaches only a node inside the boundary -/
  egress : Bool

/-- Soundness trusts the auditor's parties. -/
def Setting.sound (A : Agreement) : Setting := ⟨Roles.honest A, fun p => p.side == .auditor, false⟩

/-- Zero knowledge trusts the boundary's parties, and rests on the egress premise. -/
def Setting.zk (A : Agreement) : Setting := ⟨Roles.honest A, fun p => p.side == .boundary, true⟩

def Node.trusted (S : Setting) : Node → Bool
  | .party p => S.trusts p
  | _ => false

def Node.isWorker : Node → Bool
  | .worker _ => true
  | _ => false

structure World (S : Setting) where
  states : (p : Party) → (S.roles p).State
  inflight : List Frame
  /-- every frame put in flight, with the tick it was sent at -/
  trace : List (Nat × Frame)
  tick : Nat

def World.init (S : Setting) : World S := ⟨fun p => (S.roles p).init, [], [], 0⟩

inductive Event where
  /-- deliver the `i`-th frame in flight -/
  | deliver (i : Nat)
  | drop (i : Nat)
  /-- send a frame from a node the environment plays -/
  | inject (f : Frame)
  | idle
  deriving Inhabited

variable {S : Setting}

def World.setState (w : World S) (p : Party) (s : (S.roles p).State) : (q : Party) → (S.roles q).State :=
  fun q => if h : q = p then (by subst h; exact s) else w.states q

/-- The trusted party whose state the event may change: the receiver of the frame it delivers. -/
def World.touches (w : World S) : Event → Option Party
  | .deliver i => match w.inflight[i]? with
    | some { dst := .party p, .. } => if S.trusts p then some p else none
    | _ => none
  | _ => none

/-- The fresh random bytes the event's step reads. -/
def World.need (w : World S) (ev : Event) : Nat :=
  match w.touches ev, ev with
  | some p, .deliver i => match w.inflight[i]? with
    | some f => (S.roles p).fresh (w.states p) f
    | none => 0
  | _, _ => 0

/-- An `inject` the setting allows: not in a trusted party's name, and under the egress premise not from a worker to a
node outside the boundary. -/
def Setting.admits (S : Setting) (f : Frame) : Bool :=
  !f.src.trusted S && !(S.egress && f.src.isWorker && !f.dst.inside)

def World.send (w : World S) (fs : List Frame) : World S :=
  { w with inflight := w.inflight ++ fs, trace := w.trace ++ fs.map (w.tick, ·) }

def World.step (w : World S) (ev : Event) (r : ByteArray) : World S :=
  let w' := { w with tick := w.tick + 1 }
  match ev with
  | .idle => w'
  | .drop i => { w' with inflight := w.inflight.eraseIdx i }
  | .inject f => if S.admits f then w'.send [f] else w'
  | .deliver i => match w.inflight[i]? with
    | none => w'
    | some f =>
      let w'' := { w' with inflight := w.inflight.eraseIdx i }
      match w.touches ev with
      | some p => match (S.roles p).step (w.states p) f r with
        | .ok (s', outs) => { w'' with states := w.setState p s' }.send outs
        | .error _ => w''
      | none => w''

/-- The run on given events and fresh bytes. -/
def World.runOn (w : World S) : List (Event × ByteArray) → World S
  | [] => w
  | (ev, r) :: rest => (w.step ev r).runOn rest

/-! ## What each side of the adversary sees -/

/-- Soundness: one adversary, the developer with the network, seeing every frame sent (its contents included; the
trusted parties' states, which hold the coin tree's unrevealed coins, stay hidden). -/
def World.devView (w : World S) : List (Nat × Frame) := w.trace

/-- Zero knowledge, the network's side (the auditor's parties are the adversary's here): every frame not between two
nodes inside the boundary. -/
def World.netView (w : World S) : List (Nat × Frame) := w.trace.filter fun x => !(x.2.src.inside && x.2.dst.inside)

/-- Zero knowledge, the GPU rack's side: every frame to or from a worker. -/
def World.gpuView (w : World S) : List (Nat × Frame) := w.trace.filter fun x => x.2.src.isWorker || x.2.dst.isWorker

/-- Under zero knowledge, an event is the rack's when it moves a frame between two nodes inside the boundary or sends as
a worker; every other event is the network's. -/
def World.byGpu (w : World S) : Event → Bool
  | .deliver i | .drop i => match w.inflight[i]? with
    | some f => f.src.inside && f.dst.inside
    | none => false
  | .inject f => f.src.isWorker
  | .idle => true

def Body.shape : Body → String
  | .roots _ => "roots"
  | .innerMsg b r _ => s!"innerMsg {b} {r}"
  | .commit s _ => s!"commit {repr s}"
  | .draw _ => "draw"
  | .coins s _ => s!"coins {repr s}"
  | .witness _ _ => "witness"
  | .masks _ _ => "masks"
  | .outer k _ => s!"outer {k}"
  | .level0 _ => "level0"
  | .finish _ _ => "finish"
  | .log _ => "log"
  | .closed _ => "closed"
  | .verdict _ _ _ => "verdict"
  | .profile _ => "profile"

/-- What zero knowledge concedes: when each frame left the boundary, to whom, and its shape, never its contents. The
GPU rack can modulate it (by when it sends, and by sending what the send gate refuses); the network warden bounds that
channel, not this model. -/
def World.leak (w : World S) : List (Nat × Node × Nat × String) :=
  (w.trace.filter fun x => x.2.leaves).map fun (t, f) => (t, f.dst, f.seq, f.body.shape)

/-- The profile the run issued, if it accepted. -/
def World.accepted (w : World S) : Option IntegrityProfile :=
  w.trace.findSome? fun
    | (_, { src := .party .profile, dst := .outside, body := .profile p, .. }) => p
    | _ => none

end Run
