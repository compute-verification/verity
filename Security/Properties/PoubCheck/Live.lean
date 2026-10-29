import Properties.PoubCheck
import PoubCheck.Verifier

/-!
# The verifier's process accepts what `Shared.run g` accepts (`LiveIsRun`)

The verifier's process (`PoubCheck.Verifier`, `poub-check-verifier`) doesn't run `Shared.run`: it takes a
session's events one request at a time, each with the coin the challenger released for it (`Verifier.eventOn`: the coin
set first, `withCoin`, and an event that read a coin never sent refused, `unset`), checks a window's openings on
threads of their own when the driver asks (`verify`, `Verifier.windowCheck`: the frames read by `Wire.openingsOf`, the
verdict `checkWindowChunked`'s), and applies that verdict to the session as it stands when the driver settles it
(`settle`, `Session.verifyBy`), other events between. The driver's transcript records each event as it came and each
settled window as a `verify` of its frames, read by the same `Wire.openingsOf`.

`Live` is that request loop on one session drawn `g` windows a coin, as a pure function: `step` is the process's
`event`, `verify` and `settle` on a running session, each giving its answer and the event the transcript records for it
(none for a `verify`), and `run` takes a list of requests through it.

* `LiveIsRun`: a session the process ran on its live coins, with every window it checked settled or not, is
  `Shared.run g` on the events its transcript records, from `State.init` on any coins that agree with the ones the
  challenger sent (every coin the session holds, of a coin's size, and every append coin): every result the same,
  and the same session but for its coins. So its outcome and record (`State.outcome`, `State.summary`, which read no
  coin) are `Shared.run g`'s, and a transcript that pads the coins never sent (`benchmarks/poub_session/
  lean_verifier.py`) replays to the process's verdict. At `g = 1` that is `Session.run` (`PoubCheck.Shared.run_one`).

It needs `1 ≤ g` and `1 ≤ min_passes` (`Properties.PoubCheck.Coins`: with none, an empty window drawn at its deadline
could take a root before its verdict is settled), which the process's `session` request checks (`Difftest.sessionOf`'s
`positive`). The model is the process at two windows in flight, where `Verifier.eventOn`'s `Flight.step` is
`Shared.step g` (`Flight.step_two`); a session at one in flight isn't in it, as no property covers one yet (`Flight`).
The registry's other sessions, and `abort`, which ends a session with the driver's outcome, are not in the model: the
process keeps each session's state apart (`Registry.running`, `Registry.update`) and an aborted session is not
accepted. The process's IO (reading requests and frames, threads, writing answers) is `handle`'s and `serve`'s.
-/

open PoubCheck

namespace Properties.PoubCheck.Live

/-- A request of the process's, on the one session: an event and the coin released for it (`event`), window `t`'s
check on its frames under the driver's id (`verify`, with the `openings` requests' frames in the order they came), and
that check's verdict applied (`settle`). -/
inductive Req
  | event (e : Session.Event) (coin : Option ByteArray)
  | verify (t : Int) (id : Nat) (frames : Array ByteArray)
  | settle (id : Nat)

/-- A check the process spawned and hasn't settled: its id, window and frames, and its verdict. -/
structure Check where
  id : Nat
  t : Int
  frames : Array ByteArray
  verdict : Verdict

/-- The process's state for the session: the session's, and its checks not settled. -/
structure Proc where
  session : Session.State
  checks : List Check := []

/-- An answer: an event's or a settle's result, a check spawned, or a `verify` refused. -/
inductive Answer
  | result (r : Session.Result)
  | spawned
  | refused

def Answer.result? : Answer → Option Session.Result
  | .result r => some r
  | _ => none

/-- One request through the process (`Verifier.handle`'s `event`, `verify` and `settle` on a running session): its
answer, the event the transcript records for it, and the process after it. -/
def step (chunk g : Nat) (p : Proc) : Req → Except String (Answer × Option Session.Event × Proc)
  | .event e coin => do
    let (r, s) ← Verifier.eventOn 2 g p.session e coin
    pure (.result r, some e, { p with session := s })
  | .verify t id frames => do
    if p.checks.any (·.id == id) then throw s!"check {id} is already running"
    match p.session.awaiting t with
    | none => pure (.refused, none, p)
    | some (w, drawn) =>
      pure (.spawned, none,
        { p with checks := p.checks ++
          [{ id, t, frames, verdict := (Verifier.windowCheck chunk p.session w drawn frames).1 }] })
  | .settle id => do
    let some c := p.checks.find? (·.id == id) | throw s!"no check {id}"
    let (r, s) := Session.verifyBy p.session c.t fun _ => c.verdict
    pure (.result r, some (.verify c.t (Wire.openingsOf c.frames)),
      { session := s, checks := p.checks.filter (·.id != id) })

/-- Every request's answer, in order, the events the transcript records, and the process after them. -/
def run (chunk g : Nat) (p : Proc) (qs : List Req) :
    Except String (List Answer × List Session.Event × Proc) :=
  qs.foldlM (fun (acc : List Answer × List Session.Event × Proc) q => do
    let (a, e?, p') ← step chunk g acc.2.2 q
    pure (acc.1 ++ [a], acc.2.1 ++ e?.toList, p')) ([], [], p)

end Properties.PoubCheck.Live

namespace Properties.PoubCheck

/-- The process's run of a session from `State.init` drawn `g` windows a coin, on the coins it was sent, is
`Shared.run g` on the events its transcript records, from `State.init` on coins `c` and append coins `a` that agree
with them: the same results (each event's and settle's) and the same session, with `c` and `a`. -/
def LiveIsRun : Prop :=
  ∀ (chunk g : Nat) (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray)
    (notBeforeNs : Nat) (qs : List Live.Req) (answers : List Live.Answer) (es : List Session.Event) (p : Live.Proc),
    Live.run chunk g { session := Session.State.init params reg coins appendCoins notBeforeNs } qs =
      .ok (answers, es, p) →
    ∀ (c a : Array ByteArray), 1 ≤ g → 1 ≤ params.minPasses →
      (∀ (i : Nat) (x : ByteArray), p.session.coins[i]? = some x → x.size = Codec.COIN_BYTES → c[i]? = some x) →
      (∀ (k : Nat) (x : ByteArray), p.session.appendCoins[k]? = some x → a[k]? = some x) →
      Shared.run g (Session.State.init params reg c a notBeforeNs) es.toArray =
        .ok ((answers.filterMap Live.Answer.result?).toArray, { p.session with coins := c, appendCoins := a })

end Properties.PoubCheck
