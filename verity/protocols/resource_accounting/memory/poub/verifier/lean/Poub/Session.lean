module

public import Poub.Window

/-!
# A session's bookkeeping: `audit.py`'s `Session`

The verifier's side of one session as a state machine over the events it sees, on its own clock (each event carries the
clock's reading where `Session` reads it) and with its own coins (the 2W committed coins, window t's challenge at t and
its openings coin at W + t, and the append coins in the order sampled). Each event gives what the Python call returns:

* `start`, `boundary`: the coin released (`none` at the close after the last window);
* `append(root, count)`: the first index and its coin `c_a`; rejected `append` if malformed or past the codec's
  `max_blocks` or an origin `State.originOk` refuses, `coin` if its root was sent before or the last window's roots are final;
* `release(runs, withdrawn)`: rejected `release` unless the runs are a canonical plan of the store with none released
  before, and `withdrawn` is empty or a canonical plan of appended blocks inside the runs that no plan received before
  held (memory-map/v1's withdrawn blocks, released before their encode landed);
* `receive(t, i, root, plan, step)`: rejected `malformed` (a root not 64 bytes), `order` (not the next root, a window
  already final, or past a paced window's k roots), `late` (past `D_t = b_t + A`), `step` (missing, not above the previous pass's, or 2^64 or more), `plan`
  (not a canonical plan of the store so far, whole, as runs or as a change of the previous pass's, one holding a
  released block, or one that would take its window's receipts to `RECEIPT_LIMIT`, 2^64), `lifetime` (an appended block that joined a window L or more before; with refresh, a registered block at `t ≥ L`), `probation` (with refresh, an appended block whose payload was born less than L windows before);
* `close(t)`: the window's samples, once its roots are final (the next window's first root, `D_t` passed, or a paced
  window's k roots once its boundary has left); rejected `few` under `min_passes` roots, `coin` at the last window if an append's coin went unused;
* `verify(t, openings)`: `verifyWindow`'s verdict (`checkWindow` on the samples); rejected `malformed` on a window with
  no samples, whatever its openings, so a frame that doesn't decode (`Wire.openingsOf`'s `#[]`) is refused on any
  window.

A call `Session` refuses with `SessionError` (out of turn, after the session ended, before a window's roots are final)
is `refused`. After a rejection every later call but `start` is refused, as `_live` raises.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Poub.Session

def STEP_LIMIT : Nat := 2 ^ 64

/-- A window's receipts stay below this (`receive` rejects with `plan` a pass that would take them to it), so each
sample's sampler word lies in its stream's first SHA-512 block (`Key._uniform` at `U < 2^448`). -/
def RECEIPT_LIMIT : Nat := 2 ^ 64

/-- `audit.Pace`: a paced session's windows hold at most `k` roots each, and its γ is declared per `windowNs`. -/
structure Pace where
  k : Nat
  windowNs : Nat

/-- `audit.Params`, less what the registration fixes. -/
structure Params where
  deadlineNs : Nat
  windows : Nat
  samples : Nat
  minPasses : Nat
  windowNs : Nat
  lifetime : Nat
  pace : Option Pace := none
  /-- Refresh: an append may re-key registered blocks or a fresh append's (`Origin`), a plan in window `t` holds a
  registered block only while `t < lifetime`, and an appended one only once its payload's birth `b` (`State.born`) has
  `b + lifetime ≤ t` (`probation`). Off, appends are fresh and `receive` is as without it. -/
  refresh : Bool := false

structure Win where
  challenge : ByteArray
  releasedNs : Nat
  roots : Array ByteArray := #[]
  plans : Array Plan := #[]
  post : Array Nat := #[]
  steps : Array Nat := #[]
  heads : Array (Option ByteArray) := #[]
  stale : Array Nat := #[]
  arrived : Array Nat := #[]
  boundaryNs : Option Nat := none
  final : Bool := false
  drawn : Option (Array (Nat × Nat)) := none
  openedNs : Option Nat := none
  verdict : Option Verdict := none
deriving Inhabited

/-- `U`, a window's receipts: its passes' lengths summed. -/
def Win.receipts (w : Win) : Nat := (w.plans.map (·.length)).toList.sum

structure Failed where
  window : Nat
  failure : Failure
  at? : Option (Nat × Nat)

structure State where
  params : Params
  reg : Registration
  digest : ByteArray
  coins : Array ByteArray
  appendCoins : Array ByteArray
  appendSampled : Nat := 0
  notBeforeNs : Nat
  windows : Array Win := #[]
  filling : Nat := 0
  failed : Option Failed := none
  extent : Nat
  released : Nat := 0
  appends : Array Append := #[]
  head : Option ByteArray := none
  joined : Array Nat := #[]
  gone : Array (Nat × Nat) := #[]
  planned : Array (Nat × Nat) := #[]
  plan : Plan
  step : Int := -1

inductive Event
  | start (now : Nat)
  | boundary (now : Nat)
  | append (root : ByteArray) (count : Int) (origin : Origin := .fresh)
  | release (runs withdrawn : Array (Int × Int))
  | receive (t i : Int) (root : ByteArray) (runs : Option (Array (Int × Int)))
      (change : Option (Array (Int × Int) × Array (Int × Int))) (step : Option Int) (now : Nat)
  | close (t : Int) (now : Nat)
  | verify (t : Int) (openings : Array Opening)

inductive Result
  | refused (why : String)
  | rejected (window : Nat) (failure : Failure) (at? : Option (Nat × Nat))
  | released (coin : Option Nat)
  | appended (first : Nat) (coin : ByteArray)
  | ok
  | sampled (draws : Array (Nat × Nat))
  | accepted (window : Nat)

/-- A session before its first event: `Session.__init__` on parameters and coins already checked. -/
def State.init (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat) :
    State :=
  { params, reg, digest := reg.digest, coins, appendCoins, notBeforeNs, extent := reg.blocks,
    plan := Plan.whole reg.blocks }

def State.fail (s : State) (t : Nat) (f : Failure) (at? : Option (Nat × Nat) := none) : Result × State :=
  let windows := if t < s.windows.size then s.windows.modify t ({· with verdict := some (.reject f at?)}) else s.windows
  (.rejected t f at?, { s with failed := some ⟨t, f, at?⟩, windows })

def State.deadline (s : State) (t : Nat) : Option Nat := s.windows[t]!.boundaryNs.map (· + s.params.deadlineNs)

def State.releaseCoin (s : State) (t now : Nat) : State :=
  { s with windows := s.windows.push { challenge := s.coins[t]!, releasedNs := now } }

/-- `_released(first, count)`: whether a released run overlaps `first … first + count − 1`. -/
def State.isReleased (s : State) (first count : Nat) : Bool := s.gone.any fun (a, b) => a < first + count && first < b

/-- `_stale_below(t)`: the store index below which every appended block joined a window `lifetime` or more before t. -/
def State.staleBelow (s : State) (t : Nat) : Nat :=
  let k := (s.joined.toList.takeWhile fun (x : Nat) => (x : Int) ≤ (t : Int) - s.params.lifetime).length
  match s.appends[k]? with
  | some a => a.first
  | none => s.extent

/-- The window in which append `a`'s payloads were first committed: the window its origin append joined (`joined`, or
the window being filled for one not joined yet), `none` for a re-key of registered blocks or past the appends. -/
def State.born (s : State) (a : Nat) : Option Nat :=
  match (s.appends[a]?.map (·.origin) : Option Origin) with
  | some Origin.fresh => some (s.joined.getD a s.filling)
  | some (Origin.rekey k _) => some (s.joined.getD k s.filling)
  | _ => none

/-- Whether `first … first + count − 1` holds a block of an append whose payloads were born less than `lifetime`
windows before `t`. -/
def State.probation (s : State) (t first count : Nat) : Bool :=
  (Array.range s.appends.size).any fun a =>
    let x := s.appends[a]!
    x.first < first + count && first < x.first + x.count && (s.born a).any (t < · + s.params.lifetime)

/-- What `append` holds of an origin, without an opening: `fresh`; or, with refresh, a re-key of at most `REKEY_MAX`
blocks declaring itself (`rekeyRoot` at `first`, `count`) whose origin's blocks exist: registered blocks
`a … a + count − 1` below `N`, or blocks `o … o + count − 1` of an earlier append `k` that is fresh. -/
def State.originOk (s : State) (root : ByteArray) (first count : Nat) : Origin → Bool
  | .fresh => true
  | o@(.registered a) => s.params.refresh && count ≤ REKEY_MAX && a + count ≤ s.reg.blocks
      && root == rekeyRoot s.digest first count o
  | o@(.rekey k off) => s.params.refresh && count ≤ REKEY_MAX
      && s.appends[k]?.any (fun x => x.origin == .fresh && off + count ≤ x.count)
      && root == rekeyRoot s.digest first count o

/-- The number of fresh appends before append `k`: the fresh ordinal of a fresh append `k`. -/
def State.freshOrdinal (s : State) (k : Nat) : Nat := ((s.appends.toList.take k).filter (·.origin == .fresh)).length

/-- Whether append `a`, joining window `r`, is a piece the public rota re-keys at `r`. A registered block `j` is in
class `j · L / N`, class `c` being blocks `lo … hi − 1` with `lo = ⌈c·N/L⌉`, `hi = ⌈(c+1)·N/L⌉`, and a fresh append in
class its fresh ordinal mod `L`; a chunk of base window `b` (0 registered, `joined` otherwise) and class `c` is re-keyed
at the windows `b + 1 + (c − b − 1) mod L + n·L`, in pieces of `REKEY_MAX` blocks from the run's start (`lo`, or offset
0 of the fresh append), the last piece the run's rest. A fresh append is always in it. -/
def State.inRota (s : State) (a r : Nat) : Bool :=
  let L := s.params.lifetime
  let N := s.reg.blocks
  match s.appends[a]? with
  | none => false
  | some x => match x.origin with
    | .fresh => true
    | .registered j =>
      let c := r % L
      let lo := (c * N + L - 1) / L
      let hi := ((c + 1) * N + L - 1) / L
      1 ≤ r && lo ≤ j && j < hi && (j - lo) % REKEY_MAX == 0 && x.count == min REKEY_MAX (hi - j)
    | .rekey k o =>
      let b := s.joined.getD k r
      let c : Int := (s.freshOrdinal k % L : Nat)
      let first := b + 1 + (Int.emod (c - b - 1) L).toNat
      let n := (s.appends[k]?.map (·.count)).getD 0
      b < r && first ≤ r && (r - first) % L == 0 && o % REKEY_MAX == 0 && o < n && x.count == min REKEY_MAX (n - o)

/-- The rota's order of the re-keys joining one window: registered pieces by first block, then fresh appends' pieces
by append and offset. -/
def Origin.dueBefore : Origin → Origin → Bool
  | .registered a, .registered a' => a < a'
  | .registered _, .rekey .. => true
  | .rekey k o, .rekey k' o' => k < k' || (k == k' && o < o')
  | _, _ => false

/-- What `receive` holds of the re-keys joining window `t` (those not joined yet, and those already joined at `t`):
each is a piece the rota re-keys at `t` (`inRota`), and they came in the rota's order (`Origin.dueBefore`), so a due
piece may be missing but none is sent twice or out of turn. -/
def State.rotaOk (s : State) (t : Nat) : Bool :=
  let joining := (List.range s.appends.size).filter fun a =>
    s.joined.getD a t == t && (s.appends[a]?.any (·.origin != .fresh))
  joining.all (s.inRota · t) &&
    (joining.zip joining.tail).all fun (a, b) => Origin.dueBefore s.appends[a]!.origin s.appends[b]!.origin

def State.record (s : State) (t : Nat) : WindowRecord :=
  let w := s.windows[t]!
  { index := t, challenge := w.challenge,
    previous := if t > 0 then s.windows[t - 1]!.roots.back?.getD ByteArray.empty else s.digest,
    passes := (Array.range w.roots.size).map fun i => ⟨w.roots[i]!, w.plans[i]!, w.heads[i]!⟩ }

private def live (s : State) : Option Result :=
  s.failed.map fun f => .refused s!"the session ended: window {f.window} was rejected ({f.failure.name})"

def start (s : State) (now : Nat) : Result × State :=
  if !s.windows.isEmpty then (.refused "a session starts once", s)
  else if now < s.notBeforeNs then (.refused "the session's first coin may not leave yet", s)
  else (.released (some 0), s.releaseCoin 0 now)

def boundary (s : State) (now : Nat) : Result × State := Id.run do
  if let some r := live s then return (r, s)
  let some t := (Array.range s.windows.size).find? (s.windows[·]!.boundaryNs.isNone)
    | return (.refused "every window has ended, or the session has not started", s)
  if now < s.windows[t]!.releasedNs + s.params.windowNs then return (.refused "the window is stated longer", s)
  let s := { s with windows := s.windows.modify t ({· with boundaryNs := some now}) }
  if t + 1 < s.params.windows then return (.released (some (t + 1)), s.releaseCoin (t + 1) now)
  return (.released none, s)

def append (s : State) (root : ByteArray) (count : Int) (origin : Origin := .fresh) :
    Except String (Result × State) := do
  if let some r := live s then return (r, s)
  if root.size ≠ DIGEST_BYTES || count < 1 || s.extent + count.toNat > s.reg.codec.maxBlocks
      || !s.originOk root s.extent count.toNat origin then
    return s.fail s.filling .append
  if s.appends.any (·.root == root) || (s.windows.size == s.params.windows && (s.windows.back?.map (·.final)).getD false)
  then return s.fail s.filling .coin
  let some coin := s.appendCoins[s.appendSampled]? | throw "the append coins ran out"
  let first := s.extent
  let n := count.toNat
  return (.appended first coin,
    { s with
      appendSampled := s.appendSampled + 1
      appends := s.appends.push { first, count := n, root, coin, origin }
      head := some (appendHead s.head first n root)
      extent := first + n })

private def merge (xs : Array (Nat × Nat)) : Array (Nat × Nat) :=
  (xs.qsort fun a b => a.1 < b.1 || (a.1 == b.1 && a.2 < b.2)).foldl (fun out (a, b) =>
    match out.back? with
    | some (oa, ob) => if ob == a then out.pop.push (oa, b) else out.push (a, b)
    | none => out.push (a, b)) #[]

/-- `_merged`: half-open spans `[a, b)`, overlapping or not, as the fewest disjoint ones covering the same blocks, in
order. -/
private def cover (xs : Array (Nat × Nat)) : Array (Nat × Nat) :=
  (xs.qsort fun a b => a.1 < b.1 || (a.1 == b.1 && a.2 < b.2)).foldl (fun out (a, b) =>
    match out.back? with
    | some (oa, ob) => if a ≤ ob then out.pop.push (oa, max ob b) else out.push (a, b)
    | none => out.push (a, b)) #[]

/-- Whether a release of `pl` may name `withdrawn`: none, or a canonical plan of the store whose blocks are appended
ones inside `pl` that no plan received before held (`planned`). -/
def State.withdrawable (s : State) (pl : Plan) (withdrawn : Array (Int × Int)) : Bool :=
  withdrawn.isEmpty ||
    match Plan.make? withdrawn s.extent with
    | none => false
    | some w =>
      let spans := cover (pl.runs.map fun (f, c) => (f, f + c))
      w.runs.all fun (f, c) => s.reg.blocks ≤ f && spans.any (fun (a, b) => a ≤ f && f + c ≤ b) &&
        !s.planned.any fun (a, b) => a < f + c && f < b

def release (s : State) (runs withdrawn : Array (Int × Int)) : Result × State := Id.run do
  if let some r := live s then return (r, s)
  let some pl := Plan.make? runs s.extent | return s.fail s.filling .release
  if pl.runs.any fun (f, c) => s.isReleased f c then return s.fail s.filling .release
  if !s.withdrawable pl withdrawn then return s.fail s.filling .release
  return (.ok,
    { s with
      gone := merge (s.gone ++ pl.runs.map fun (f, c) => (f, f + c))
      released := s.released + pl.runs.foldl (fun n r => n + r.2) 0 })

def receive (s : State) (t i : Int) (root : ByteArray) (runs : Option (Array (Int × Int)))
    (change : Option (Array (Int × Int) × Array (Int × Int))) (step : Option Int) (now : Nat) : Result × State := Id.run do
  if let some r := live s then return (r, s)
  if s.windows.isEmpty then return (.refused "the session has not started", s)
  if root.size ≠ DIGEST_BYTES then return s.fail s.filling .malformed
  let mut s := s
  if t == s.filling + 1 && i == 0 && t < s.windows.size then
    s := { s with windows := s.windows.modify s.filling ({· with final := true}), filling := t.toNat }
  else if t != s.filling || i != s.windows[s.filling]!.roots.size || s.windows[s.filling]!.final
      || s.params.pace.any (fun p => decide ((p.k : Int) ≤ i)) then
    return s.fail s.filling .order
  let t := s.filling
  if let some d := s.deadline t then
    if now > d then return s.fail t .late
  let some st := step | return s.fail t .step
  if !(s.step < st && st < STEP_LIMIT) then return s.fail t .step
  let N := s.reg.blocks
  let extent := s.extent
  let plan? := match change, runs with
    | some _, some _ => none
    | some (added, removed), none => s.plan.changed added removed extent
    | none, none => some (Plan.whole N)
    | none, some rs => Plan.make? rs extent
  let some plan := plan? | return s.fail t .plan
  if !s.gone.isEmpty && plan.runs.any fun (f, c) => s.isReleased f c then return s.fail t .plan
  if RECEIPT_LIMIT ≤ s.windows[t]!.receipts + plan.length then return s.fail t .plan
  s := { s with joined := s.joined ++ Array.replicate (s.appends.size - s.joined.size) t }
  if s.params.refresh && !s.rotaOk t then return s.fail t .origin
  let below := s.staleBelow t
  if below > N && plan.runs.any fun (f, c) => f < below && f + c > N then return s.fail t .lifetime
  if s.params.refresh then
    if s.params.lifetime ≤ t && plan.runs.any fun (f, _) => f < N then return s.fail t .lifetime
    if plan.runs.any fun (f, c) => f + c > N && s.probation t (max f N) (f + c - max f N) then
      return s.fail t .probation
  let post := if extent == N then 0 else plan.runs.foldl (fun n (f, c) => n + (f + c - max f N)) 0
  let head := s.head
  let add := fun (w : Win) =>
    { w with
      roots := w.roots.push root
      plans := w.plans.push plan
      post := w.post.push post
      steps := w.steps.push st.toNat
      heads := w.heads.push head
      stale := w.stale.push below
      arrived := w.arrived.push now }
  s := { s with windows := s.windows.modify t add, plan, step := st,
                planned := cover (s.planned ++ plan.runs.filterMap fun (f, c) =>
                  if f + c > N then some (max f N, f + c) else none) }
  return (.ok, s)

def close (s : State) (t : Int) (now : Nat) : Except String (Result × State) := do
  if let some r := live s then return (r, s)
  if t < 0 || t ≥ s.windows.size then return (.refused "the window is not open to draw", s)
  let t := t.toNat
  let w := s.windows[t]!
  if w.drawn.isSome then return (.refused "the window is not open to draw", s)
  if (s.windows.extract 0 t).any (·.drawn.isNone) then return (.refused "windows are drawn in order", s)
  let d := s.deadline t
  let full := d.isSome && s.params.pace.any fun p => decide (p.k ≤ w.roots.size)
  if !w.final && !full && (match d with | none => true | some d => now ≤ d) then
    return (.refused "the window's roots are not final", s)
  let s := { s with windows := s.windows.modify t ({· with final := true}) }
  if w.roots.size < s.params.minPasses then return s.fail t .few
  if t + 1 == s.params.windows && s.joined.size < s.appends.size then return s.fail t .coin
  let some sampled := receiptSample s.digest t s.coins[s.params.windows + t]! s.params.samples (w.plans.map (·.length))
    | throw "the openings' draw ran past its reads"
  return (.sampled sampled, { s with windows := s.windows.modify t ({· with drawn := some sampled, openedNs := some now}) })

def verify (s : State) (t : Int) (openings : Array Opening) : Result × State := Id.run do
  if let some r := live s then return (r, s)
  if t < 0 || t ≥ s.windows.size then return (.refused "the window has no draws awaiting openings", s)
  let t := t.toNat
  let w := s.windows[t]!
  let some sampled := w.drawn | return (.refused "the window has no draws awaiting openings", s)
  if w.verdict.isSome then return (.refused "the window has no draws awaiting openings", s)
  if sampled.isEmpty then return s.fail t .malformed
  let v := checkWindow s.reg s.digest s.appends (s.record t) sampled openings
  if v = .accept then return (.accepted t, { s with windows := s.windows.modify t ({· with verdict := some .accept}) })
  return s.fail t v.rejection.failure v.rejection.at?

/-- What window t's openings are checked against, once its samples are out and it has no verdict: the registration, its
digest, the appends, the window's record and its samples (`none` where `verify` refuses). -/
def State.awaiting (s : State) (t : Int) : Option (WindowRecord × Array (Nat × Nat)) := do
  if s.failed.isSome || t < 0 || t ≥ s.windows.size then none
  let w := s.windows[t.toNat]!
  let sampled ← w.drawn
  if w.verdict.isSome then none
  pure (s.record t.toNat, sampled)

/-- `verify` with the window's verdict `check` computed on its samples: `verify s t openings` is
`verifyBy s t (checkWindow s.reg s.digest s.appends (s.record t.toNat) · openings)` (`verify_eq_verifyBy`). -/
def verifyBy (s : State) (t : Int) (check : Array (Nat × Nat) → Verdict) : Result × State := Id.run do
  if let some r := live s then return (r, s)
  if t < 0 || t ≥ s.windows.size then return (.refused "the window has no draws awaiting openings", s)
  let t := t.toNat
  let w := s.windows[t]!
  let some sampled := w.drawn | return (.refused "the window has no draws awaiting openings", s)
  if w.verdict.isSome then return (.refused "the window has no draws awaiting openings", s)
  if sampled.isEmpty then return s.fail t .malformed
  let v := check sampled
  if v = .accept then return (.accepted t, { s with windows := s.windows.modify t ({· with verdict := some .accept}) })
  return s.fail t v.rejection.failure v.rejection.at?

theorem verify_eq_verifyBy (s : State) (t : Int) (openings : Array Opening) :
    verify s t openings = verifyBy s t (checkWindow s.reg s.digest s.appends (s.record t.toNat) · openings) := rfl

def step (s : State) : Event → Except String (Result × State)
  | .start now => pure (start s now)
  | .boundary now => pure (boundary s now)
  | .append root count origin => append s root count origin
  | .release runs withdrawn => pure (release s runs withdrawn)
  | .receive t i root runs change st now => pure (receive s t i root runs change st now)
  | .close t now => close s t now
  | .verify t openings => pure (verify s t openings)

/-- Every event's result, in order, and the session after them. -/
def run (s : State) (events : Array Event) : Except String (Array Result × State) :=
  events.foldlM (fun (acc : Array Result × State) e => do
    let (r, s') ← step acc.2 e
    pure (acc.1.push r, s')) (#[], s)

/-- `Session.outcome`: "accepted" once every window is, the failure that rejected it, or `none` while it runs. -/
def State.outcome (s : State) : Option String :=
  match s.failed with
  | some f => some f.failure.name
  | none =>
    if s.windows.size == s.params.windows && s.windows.all (·.verdict == some .accept) then some "accepted" else none

/-- What `Session.record` gives of an accepted session, which is all of it but the coin commitment's root, the block
size and the registration's fields its digest binds: each window's passes' lengths, steps, post-seed reads, heads,
stale bounds and arrivals, its challenge's release, its deadline, its openings coin's release and samples, and the
session's store size, appended and released blocks, lifetime cap, pace and chain over its appends. -/
structure WindowSummary where
  lengths : Array Nat
  steps : Array Nat
  post : Array Nat
  heads : Array (Option ByteArray)
  stale : Array Nat
  arrivedNs : Array Nat
  releasedNs : Nat
  deadlineNs : Nat
  openedNs : Nat
  drawn : Nat

structure Summary where
  windows : Array WindowSummary
  blocks : Nat
  appended : Nat
  released : Nat
  lifetime : Nat
  pace : Option Pace
  appends : Option ByteArray

def State.summary (s : State) : Option Summary :=
  if s.outcome != some "accepted" then none
  else
    let window := fun t =>
      let w := s.windows[t]!
      ({ lengths := w.plans.map (·.length)
         steps := w.steps
         post := w.post
         heads := w.heads
         stale := w.stale
         arrivedNs := w.arrived
         releasedNs := w.releasedNs
         deadlineNs := (s.deadline t).getD 0
         openedNs := w.openedNs.getD 0
         drawn := (w.drawn.map (·.size)).getD 0 } : WindowSummary)
    some
      { blocks := s.reg.blocks
        appended := s.extent - s.reg.blocks
        released := s.released
        lifetime := s.params.lifetime
        pace := s.params.pace
        appends := s.head
        windows := (Array.range s.windows.size).map window }

end Poub.Session
