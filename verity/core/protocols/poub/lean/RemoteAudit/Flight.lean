import RemoteAudit.Shared

/-!
# One window in flight: `audit.Pace.in_flight`

At `in_flight` f = 2, a paced session's schedule: window t's boundary message, which releases the next window's
challenge, may leave while t's last pass runs, and a root of t may follow it up to `D_t`. At f = 1 every coin leaves
after the root before it: `boundary` is refused on a clock reading before any root of the window it ends arrived, and
`receive` rejects `order` a root that `Session.receive` takes after its window's boundary, so a window's every root
arrived by its boundary, which is the next window's challenge's release.

`step f g` and `run f g` are `Shared.step g` and `Shared.run g` with that boundary and receive at f = 1, and at f = 2
they are `Shared.step g` and `Shared.run g` (`step_two`, `run_two`): every property of the code stated on
`Session.run` is about the two-in-flight session, and none yet covers f = 1. The difftest's `session` runs `run f g`,
and the registry `step f g`.
-/

namespace RemoteAudit.Flight

open Session

def boundary (s : State) (now : Nat) : Result × State :=
  match (Array.range s.windows.size).find? (s.windows[·]!.boundaryNs.isNone) with
  | some t =>
    if s.windows[t]!.arrived.any (now < ·) then (.refused "the window's boundary leaves after its last root", s)
    else Session.boundary s now
  | none => Session.boundary s now

def receive (s : State) (t i : Int) (root : ByteArray) (runs : Option (Array (Int × Int)))
    (change : Option (Array (Int × Int) × Array (Int × Int))) (step : Option Int) (now : Nat) : Result × State :=
  match Session.receive s t i root runs change step now with
  | (.ok, s') => if s'.windows[s'.filling]!.boundaryNs.isSome then s'.fail s'.filling .order else (.ok, s')
  | r => r

def step (f g : Nat) (s : State) (e : Event) : Except String (Result × State) :=
  if f == 1 then
    match e with
    | .boundary now => pure (boundary s now)
    | .receive t i root runs change st now => pure (receive s t i root runs change st now)
    | e => Shared.step g s e
  else Shared.step g s e

/-- Every event's result, in order, and the session after them. -/
def run (f g : Nat) (s : State) (events : Array Event) : Except String (Array Result × State) :=
  events.foldlM (fun (acc : Array Result × State) e => do
    let (r, s') ← step f g acc.2 e
    pure (acc.1.push r, s')) (#[], s)

theorem step_two (g : Nat) (s : State) (e : Event) : step 2 g s e = Shared.step g s e := rfl

theorem run_two (g : Nat) (s : State) (events : Array Event) : run 2 g s events = Shared.run g s events := rfl

end RemoteAudit.Flight
