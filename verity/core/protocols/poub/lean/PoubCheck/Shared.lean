import PoubCheck.Session

/-!
# One openings coin over several windows: `audit.Params.groups_per_coin`

At `groups_per_coin` g, windows cg … min((c + 1)g, W) − 1 are draw c, drawn from one openings coin released after the
last root of the draw's last window: the session's coins are its W challenges, then ⌈W/g⌉ openings coins, coin c at
W + c. `close` is `Session.close` for a draw: only the draw's last window closes, once its roots are final, which makes
every window of the draw final; each is rejected `few` in order; and the draw's `s` receipts are drawn over its passes
in order (`draws`, under the key of window c), numbered across its windows, each window keeping its own share,
(pass, position) in it, which `Session.verify` checks. A window whose share is none has nothing to open: the close
accepts it, and `Session.verify` refuses it, as it refuses any window with a verdict.

`step g` and `run g` are `Session.step` and `Session.run` with that close, and at g = 1 they are `Session.step` and
`Session.run` (`step_one`, `run_one`): the properties of the code stated on `Session.run` are about the
one-coin-a-window session, and `Properties.PoubCheck.Shared` states the draw's for every g. The difftest's `session`
and the registry run `run g` and `step g` (`Flight`, `Registry.step`), the verifier's process takes a draw's openings
coin with its last window's close only (`Verifier.withCoin`), and the line of record reads a record drawn g > 1 a coin
by its draws (`Line.ofSummaryAt`), at two windows in flight only (`Line.SHARED`).
-/

namespace PoubCheck.Shared

open Session

def close (g : Nat) (s : State) (t : Int) (now : Nat) : Except String (Result × State) := do
  if let some f := s.failed then
    return (.refused s!"the session ended: window {f.window} was rejected ({f.failure.name})", s)
  if t < 0 || t ≥ s.windows.size then return (.refused "the window is not open to draw", s)
  let t := t.toNat
  let w := s.windows[t]!
  if w.drawn.isSome then return (.refused "the window is not open to draw", s)
  let first := t / g * g
  if t + 1 ≠ min (first + g) s.params.windows then return (.refused "a draw closes at its last window", s)
  if (s.windows.extract 0 first).any (·.drawn.isNone) then return (.refused "windows are drawn in order", s)
  let d := s.deadline t
  let full := d.isSome && s.params.pace.any fun p => decide (p.k ≤ w.roots.size)
  if !w.final && !full && (match d with | none => true | some d => now ≤ d) then
    return (.refused "the window's roots are not final", s)
  let group := (List.range (t + 1 - first)).map (first + ·)
  let s := { s with windows := group.foldl (fun ws u => ws.modify u ({· with final := true})) s.windows }
  if let some u := group.find? (s.windows[·]!.roots.size < s.params.minPasses) then return s.fail u .few
  if t + 1 == s.params.windows && s.joined.size < s.appends.size then return s.fail t .coin
  let lengths := group.foldl (fun acc u => acc ++ s.windows[u]!.plans.map (·.length)) #[]
  let some drawn := draws s.digest (t / g) s.coins[s.params.windows + t / g]! s.params.samples lengths
    | throw "the openings' draw ran past its reads"
  let share := fun (base n : Nat) => drawn.filterMap fun (i, p) =>
    if base ≤ i && i < base + n then some (i - base, p) else none
  let (windows, _) := group.foldl (fun (acc : Array Win × Nat) u =>
      let n := acc.1[u]!.plans.size
      let mine := share acc.2 n
      (acc.1.modify u (fun w => { w with drawn := some mine, openedNs := some now,
                                         verdict := if mine.isEmpty then some .accept else w.verdict }),
       acc.2 + n)) (s.windows, 0)
  return (.drew drawn, { s with windows })

def step (g : Nat) (s : State) (e : Event) : Except String (Result × State) :=
  if g ≤ 1 then Session.step s e
  else match e with
    | .close t now => close g s t now
    | e => Session.step s e

/-- Every event's result, in order, and the session after them. -/
def run (g : Nat) (s : State) (events : Array Event) : Except String (Array Result × State) :=
  events.foldlM (fun (acc : Array Result × State) e => do
    let (r, s') ← step g acc.2 e
    pure (acc.1.push r, s')) (#[], s)

theorem step_one (s : State) (e : Event) : step 1 s e = Session.step s e := rfl

theorem run_one (s : State) (events : Array Event) : run 1 s events = Session.run s events := rfl

end PoubCheck.Shared
