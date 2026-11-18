module

public import Poub.Window

/-!
# A session's bookkeeping: `audit.py`'s `Session`

The verifier's side of one session as a state machine over the events it sees, on its own clock (each event carries the
clock's reading where `Session` reads it) and with its own coins (the 2W committed coins, window t's challenge at t and
its openings coin at W + t, and the append coins in the order sampled). Each event gives what the Python call returns:

* `start`, `boundary`: the coin released (`none` at the close after the last window);
* `append(root, count, origin, slot)`: the first index and its coin `c_a`; rejected `append` if malformed or past the
  codec's `max_blocks`, a re-key or a slot without refresh, or a re-key's root not its declaration (`rekeyRoot`);
  with refresh, `origin` unless its slot is above the last append's and holds it (`State.slotOk`); `coin` if its root
  was sent before or the last window's roots are final;
* `release(runs, withdrawn)`: rejected `release` unless the runs are a canonical plan of the store with none released
  before, and `withdrawn` is empty or a canonical plan of appended blocks inside the runs that no plan received before
  held (memory-map/v1's withdrawn blocks, released before their encode landed);
* `receive(t, i, root, plan, step)`: rejected `malformed` (a root not 64 bytes), `order` (not the next root, a window
  already final, or past a paced window's k roots), `late` (past `D_t = b_t + A`), `step` (missing, not above the previous pass's, or 2^64 or more), `plan`
  (not a canonical plan of the store so far, whole, as runs or as a change of the previous pass's, one holding a
  released block, or one that would take its window's receipts to `RECEIPT_LIMIT`, 2^64), with refresh `origin` (an
  append joining a window other than its slot's), `lifetime` (an appended block that joined a window L or more before; with refresh, a registered block at `t ≥ L`), `probation` (with refresh, an appended block whose payload was born less than L windows before);
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
  /-- Refresh: appends fill the slots of a public schedule (`Params.slotAt`), fresh or re-keys of registered blocks or
  of a fresh slot's append (`Origin`); a plan in window `t` holds a registered block only while `t < lifetime`, and an
  appended one only once its payload's birth `b` (`State.born`) has `b + lifetime ≤ t` (`probation`). Off, appends are
  fresh, carry no slot, and every check is as without it. -/
  refresh : Bool := false
  /-- With refresh, the fresh slots each window has (`Params.slotAt`); unused without. -/
  freshSlots : Nat := 0

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
  /-- With refresh, each pass's appended blocks held in probation (`State.probationHeld`); empty without. -/
  probation : Array Nat := #[]
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
  | append (root : ByteArray) (count : Int) (origin : Origin := .fresh) (slot : Option Nat := none)
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

/-- Class `c`'s registered run `[lo, hi)`: the blocks `j` with `j · L / N = c`, `lo = ⌈c·N/L⌉`, `hi = ⌈(c+1)·N/L⌉`. -/
def Params.classRun (p : Params) (N c : Nat) : Nat × Nat :=
  ((c * N + p.lifetime - 1) / p.lifetime, ((c + 1) * N + p.lifetime - 1) / p.lifetime)

/-- Whether window `w` of a session with refresh has a registered re-key slot: `w ≥ 1` and class `w mod L` holds a
block. -/
def Params.hasRegistered (p : Params) (N w : Nat) : Bool :=
  1 ≤ w && (p.classRun N (w % p.lifetime)).1 < (p.classRun N (w % p.lifetime)).2

/-- The slots of window `w`: its registered re-key slot, one re-key slot for each fresh slot of the windows `w − L`,
`w − 2L`, …, and `freshSlots` fresh slots. -/
def Params.windowSize (p : Params) (N w : Nat) : Nat :=
  (if p.hasRegistered N w then 1 else 0) + p.freshSlots * (w / p.lifetime) + p.freshSlots

/-- The first slot of window `w`: slots are laid out window by window. -/
def Params.windowStart (p : Params) (N w : Nat) : Nat := ((List.range w).map (p.windowSize N)).sum

/-- The slot of fresh slot `i` of window `b`. -/
def Params.freshSlot (p : Params) (N b i : Nat) : Nat := p.windowStart N b + p.windowSize N b - p.freshSlots + i

/-- What a slot of the schedule holds: a fresh append, the re-key of registered run `[lo, lo + count)`, or the re-key
of the fresh append in slot `k`. -/
inductive Slot
  | fresh
  | registered (lo count : Nat)
  | rekey (k : Nat)
deriving BEq, Repr, DecidableEq, Inhabited

/-- The kind of the `m`-th slot of window `w`: the registered re-key slot first, then the re-key slots of earlier fresh
slots, oldest window first and each window's in order, then the fresh slots. -/
def Params.slotIn (p : Params) (N w m : Nat) : Slot :=
  let r := if p.hasRegistered N w then 1 else 0
  if m < r then
    let (lo, hi) := p.classRun N (w % p.lifetime)
    .registered lo (hi - lo)
  else if m < r + p.freshSlots * (w / p.lifetime) then
    let q := m - r
    let n := w / p.lifetime - q / p.freshSlots
    .rekey (p.freshSlot N (w - n * p.lifetime) (q % p.freshSlots))
  else .fresh

/-- The schedule of a session with refresh: slot `k`'s window and kind, `none` past its last window's slots. A
function of `N`, `lifetime`, `windows` and `freshSlots` alone, fixed before every coin. -/
def Params.slotAt (p : Params) (N k : Nat) : Option (Nat × Slot) := Id.run do
  let mut start := 0
  for w in [0:p.windows] do
    let n := p.windowSize N w
    if k < start + n then return some (w, p.slotIn N w (k - start))
    start := start + n
  return none

/-- The window slot `k` joins (`Params.slotAt`), used or empty. -/
def State.slotWindow (s : State) (k : Nat) : Option Nat := (s.params.slotAt s.reg.blocks k).map (·.1)

/-- `_stale_below(t)`: the store index below which every appended block joined a window `lifetime` or more before t.
With refresh, `receive` refuses an append joining a window other than its slot's (`joinsAnother`), so this is the
schedule's bound over each append's slot window. -/
def State.staleBelow (s : State) (t : Nat) : Nat :=
  let k := (s.joined.toList.takeWhile fun (x : Nat) => (x : Int) ≤ (t : Int) - s.params.lifetime).length
  match s.appends[k]? with
  | some a => a.first
  | none => s.extent

/-- With refresh, the window in which append `a`'s payloads were first committed: the window of the slot of the fresh
append holding them (its own, or a re-key's origin `k`); `none` for a re-key of registered blocks or past the appends. -/
def State.born (s : State) (a : Nat) : Option Nat :=
  match (s.appends[a]?.map (·.origin) : Option Origin) with
  | some Origin.fresh => s.appends[a]?.bind fun x => s.slotWindow x.slot
  | some (Origin.rekey k) => s.slotWindow k
  | _ => none

/-- Whether `first … first + count − 1` holds a block of an append whose payloads were born less than `lifetime`
windows before `t`. -/
def State.probation (s : State) (t first count : Nat) : Bool :=
  (Array.range s.appends.size).any fun a =>
    let x := s.appends[a]!
    x.first < first + count && first < x.first + x.count && (s.born a).any (t < · + s.params.lifetime)

/-- The blocks of `first … first + count − 1` released so far. -/
def State.releasedIn (s : State) (first count : Nat) : Nat :=
  s.gone.foldl (fun n (a, b) => n + (min b (first + count) - max a first)) 0

/-- The appended blocks held in probation at window `t`: of appends whose payloads were born less than `lifetime`
windows before `t`, not released. -/
def State.probationHeld (s : State) (t : Nat) : Nat :=
  (Array.range s.appends.size).foldl (fun n a =>
    let x := s.appends[a]!
    if (s.born a).any (t < · + s.params.lifetime) then n + (x.count - s.releasedIn x.first x.count) else n) 0

/-- With refresh, whether an append not joined yet would join window `t`, another than its slot's. -/
def State.joinsAnother (s : State) (t : Nat) : Bool :=
  s.params.refresh && (List.range' s.joined.size (s.appends.size - s.joined.size)).any fun a =>
    s.slotWindow s.appends[a]!.slot != some t

/-- With refresh, what `receive` refuses of a plan in window `t` past the lifetime check: `lifetime` if it holds a
registered block at `t ≥ lifetime`, `probation` if it holds a block of an append in probation (`State.probation`). -/
def State.refreshRefusal (s : State) (t : Nat) (plan : Plan) : Option Failure :=
  let N := s.reg.blocks
  if !s.params.refresh then none
  else if s.params.lifetime ≤ t && plan.runs.any fun (f, _) => f < N then some .lifetime
  else if plan.runs.any fun (f, c) => f + c > N && s.probation t (max f N) (f + c - max f N) then some .probation
  else none

/-- A pass's probation entry pushed onto `ps`, with refresh (`probationHeld`); `ps` as it is without. -/
def State.probationPush (s : State) (t : Nat) (ps : Array Nat) : Array Nat :=
  if s.params.refresh then ps.push (s.probationHeld t) else ps

/-- `append`'s check of a slot, with refresh: above the last append's slot, in the schedule, and the append its kind
names: a fresh append of at most `REKEY_MAX` blocks, the re-key of the whole class run, or the whole re-key of the
fresh append in slot `k` (an empty slot `k` has none to re-key). -/
def State.slotOk (s : State) (slot count : Nat) (origin : Origin) : Bool :=
  s.appends.back?.all (·.slot < slot) &&
    match s.params.slotAt s.reg.blocks slot, origin with
    | some (_, .fresh), .fresh => count ≤ REKEY_MAX
    | some (_, .registered lo n), .registered a => a == lo && count == n
    | some (_, .rekey k), .rekey k' => k == k' && (s.appends.find? (·.slot == k)).any (·.count == count)
    | _, _ => false

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

def append (s : State) (root : ByteArray) (count : Int) (origin : Origin := .fresh) (slot : Option Nat := none) :
    Except String (Result × State) := do
  if let some r := live s then return (r, s)
  if root.size ≠ DIGEST_BYTES || count < 1 || s.extent + count.toNat > s.reg.codec.maxBlocks
      || (!s.params.refresh && (origin != .fresh || slot.isSome))
      || (origin != .fresh && root != rekeyRoot s.digest s.extent count.toNat origin) then
    return s.fail s.filling .append
  if s.params.refresh && !(slot.any fun k => s.slotOk k count.toNat origin) then return s.fail s.filling .origin
  if s.appends.any (·.root == root) || (s.windows.size == s.params.windows && (s.windows.back?.map (·.final)).getD false)
  then return s.fail s.filling .coin
  let some coin := s.appendCoins[s.appendSampled]? | throw "the append coins ran out"
  let first := s.extent
  let n := count.toNat
  return (.appended first coin,
    { s with
      appendSampled := s.appendSampled + 1
      appends := s.appends.push { first, count := n, root, coin, origin, slot := slot.getD s.appends.size }
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
  if s.joinsAnother t then return s.fail t .origin
  s := { s with joined := s.joined ++ Array.replicate (s.appends.size - s.joined.size) t }
  let below := s.staleBelow t
  if below > N && plan.runs.any fun (f, c) => f < below && f + c > N then return s.fail t .lifetime
  if let some f := s.refreshRefusal t plan then return s.fail t f
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
      arrived := w.arrived.push now
      probation := s.probationPush t w.probation }
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
  | .append root count origin slot => append s root count origin slot
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
  probation : Array Nat := #[]

structure Summary where
  windows : Array WindowSummary
  blocks : Nat
  appended : Nat
  released : Nat
  lifetime : Nat
  pace : Option Pace
  appends : Option ByteArray
  refresh : Bool := false

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
         drawn := (w.drawn.map (·.size)).getD 0
         probation := w.probation } : WindowSummary)
    some
      { blocks := s.reg.blocks
        appended := s.extent - s.reg.blocks
        released := s.released
        lifetime := s.params.lifetime
        pace := s.params.pace
        appends := s.head
        refresh := s.params.refresh
        windows := (Array.range s.windows.size).map window }

end Poub.Session
