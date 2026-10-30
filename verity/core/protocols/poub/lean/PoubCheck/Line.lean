import PoubCheck.Pooled
import PoubCheck.Shared

/-!
# The memory line held over time

The line of record a gate run publishes (the verifier process's `line_record` request, which `auditor_line.py` holds to
`line.py`'s): the conclusions of
`Accounting.Guarantees.Poub.HeldOverTime` (time-held) and `HeldOverTimeLinks` (link-charged),
`Security/Definitions/Accounting/Guarantees/PoubHeld.lean`, read over each stretch of a paced session's groups in exact
rationals, every refusal the Python's word for word. `lineOfRecord` reads the session's groups, its windows and its
registered blocks off the Lean verifier's record of it (`Session.Summary`), each group's times and reads beside it
(`Seen`), refused unless the record is of the registration given with it (`lineOfRegistered`); `line` takes them
given. The wrong receipts W are the pooled bound of the record's windows
(`PoubCheck.Pooled.pooled`: their receipts and draws at δ). A record drawn g > 1 windows a coin (`Shared`) has the
same groups, and its pooled bound reads its draws, each its g windows' receipts and draws summed (`ofSummaryAt`,
`lineOfRecordAt`); at one window in flight it has no line (`SHARED`). `line.py`'s docstring has the form;
`tests/test_line.py` holds the two together through `PoubCheck.Difftest`'s `line` and `line_record`. Code: Lean core
only.
-/

namespace PoubCheck.Line

/-- One group: its coin's release and its last root's arrival (ns, on one clock), its passes, the registered blocks
its least pass read, and the registered blocks its passes read in all, by kind. -/
structure Group where
  releasedNs : Nat
  lastNs : Nat
  passes : Nat
  least : Nat
  weights : Nat
  hbm : Nat
  host : Nat
  deriving Inhabited

/-- At most `rate` blocks a ns back into node memory and `burst` blocks at once beyond it (the statements' X and B). -/
structure Charge where
  rate : Rat
  burst : Rat

/-- The node's ways back into its memory: its squaring ceiling over a block's encode in squarings, its disks' and its
boundary links' read ceilings (bytes a second), and each one's burst (bytes). -/
structure Rates where
  squaringsPerS : Rat
  encodeSquarings : Nat
  diskBytesPerS : Rat
  linkBytesPerS : Rat
  rebuildBurstBytes : Rat
  diskBurstBytes : Rat
  linkBurstBytes : Rat

/-- The rates by `line.RATES`' names, in its order. -/
def Rates.named (r : Rates) : List (String × Rat) :=
  [("squarings_per_s", r.squaringsPerS), ("disk_bytes_per_s", r.diskBytesPerS),
    ("link_bytes_per_s", r.linkBytesPerS), ("rebuild_burst_bytes", r.rebuildBurstBytes),
    ("disk_burst_bytes", r.diskBurstBytes), ("link_burst_bytes", r.linkBurstBytes)]

def ns : Rat := 1000000000

/-- Each form's charge in blocks, time-held then link-charged: X = s/E + β and B = B_r + B_d, then X + β_l and
B + B_l. -/
def charges (r : Rates) (b : Nat) : Charge × Charge :=
  let x := r.squaringsPerS / (r.encodeSquarings : Rat) / ns + r.diskBytesPerS / (b : Rat) / ns
  let burst := (r.rebuildBurstBytes + r.diskBurstBytes) / (b : Rat)
  (⟨x, burst⟩, ⟨x + r.linkBytesPerS / (b : Rat) / ns, burst + r.linkBurstBytes / (b : Rat)⟩)

/-- A stretch read one way: the registered blocks it charges as absent and each kind's bytes credited. -/
structure Reading where
  lost : Rat
  weights : Rat
  hbm : Rat
  filler : Rat
  deriving Inhabited

/-- Stretch `stretch`, groups [lo, hi): its span, passes, L and its groups' L summed, each kind's mean blocks a pass,
the blocks charged back, the wrong receipts' absent block-ns, and its two readings. -/
structure Stretch where
  stretch : Nat
  lo : Nat
  hi : Nat
  spanNs : Nat
  passes : Nat
  lMaxNs : Nat
  lSumNs : Nat
  weights : Rat
  hbm : Rat
  host : Rat
  rebuild : Rat
  fill : Rat
  time : Reading
  snapshot : Reading
  deriving Inhabited

/-- Held over time (`true`) or at the passes. -/
def Stretch.reading (x : Stretch) (time : Bool) : Reading := if time then x.time else x.snapshot

def check (gs : Array Group) (per blocks blockBytes nodeBytes : Nat) (wrong : Rat) (rates : Rates) :
    Except String Unit := do
  if blockBytes < 1 then throw "a block is one or more bytes"
  if blocks < 1 then throw "the store is one or more blocks"
  if nodeBytes < 1 then throw "the node is one or more bytes"
  if rates.encodeSquarings < 1 then throw "a block's encode is one or more squarings"
  for (name, q) in rates.named do
    if q < 0 then throw s!"{name} is nonnegative"
  if wrong < 0 then throw "the wrong receipts are nonnegative"
  if per < 1 then throw "a stretch is one or more groups"
  for g in [0:gs.size] do
    let x := gs[g]!
    if x.passes < 1 then throw s!"group {g} has no pass"
    if x.lastNs < x.releasedNs then throw s!"group {g}'s last root came before its coin"
    if 0 < g ∧ x.releasedNs < gs[g - 1]!.releasedNs then throw s!"group {g}'s coin left before group {g - 1}'s"
    if x.least ≠ blocks then
      throw (s!"group {g}'s least pass read {x.least} of the {blocks} registered blocks: the paced line needs every "
        ++ "pass to read every registered block, since a block a pass leaves out can be absent across it with neither "
        ++ "a wrong read nor a rebuild")
    if x.hbm + x.host ≠ x.passes * blocks then
      throw s!"group {g}'s {x.passes} passes read {x.hbm + x.host} registered blocks in all, not {x.passes * blocks}"
    if x.hbm < x.weights then throw s!"group {g}'s passes read more weights than HBM, which holds them"
  if gs.size / per < 2 then
    throw s!"the session's {gs.size} groups at {per} a stretch leave no stretch before the closing one"

private def sumOver (xs : List Nat) (f : Nat → Nat) : Nat := xs.foldl (fun a g => a + f g) 0

/-- Every stretch of `per` groups but the last full one at charge `c`, of a session `check` passed: refused only for an
empty span. The wrong receipts fill the groups lo..hi highest rate first, ties in group order. -/
def stretches (gs : Array Group) (per N b chip : Nat) (W : Rat) (c : Charge) : Except String (Array Stretch) := do
  let G := gs.size
  let s : Array Nat := gs.map (·.releasedNs)
  let n : Array Nat := gs.map (·.passes)
  let e : Array Nat := (Array.range G).map fun g => if g + 1 < G then max gs[g]!.lastNs s[g + 1]! else gs[g]!.lastNs
  let L : Array Nat := (Array.range G).map fun g => e[g]! - s[g]!
  let gap : Array Nat := (Array.range G).map fun g => if g + 1 < G then s[g + 1]! - s[g]! else L[g]!
  let S : Rat := ((chip + b - 1) / b : Nat)
  let mut out : Array Stretch := #[]
  for m in [0:G / per - 1] do
    let lo := m * per
    let hi := (m + 1) * per
    let start := if m = 0 then e[0]! else s[lo]!
    if s[hi]! ≤ start then
      throw (s!"stretch {m}'s span is empty: group {hi}'s coin left no later than "
        ++ (if m = 0 then "group 0's end" else s!"group {lo}'s coin"))
    let span : Nat := s[hi]! - start
    let groups := List.range' lo per
    let P : Nat := sumOver groups (n[·]!)
    let lMax : Nat := (List.range' (lo - 1) (hi + 1 - (lo - 1))).foldl (fun a g => max a L[g]!) 0
    let R := c.rate * ((span : Rat) + 2 * (lMax : Rat)) + c.burst
    let v : Nat → Rat := fun g => ((if g = hi then lMax else gap[g]!) : Rat) / (n[g]! : Rat)
    let order := (List.range' lo (hi + 1 - lo)).mergeSort fun g h => decide (v h ≤ v g)
    let mut left := W
    let mut fill : Rat := 0
    for g in order do
      if left ≤ 0 then break
      let took := min left ((n[g]! * N : Nat) : Rat)
      fill := fill + took * v g
      left := left - took
    let w : Rat := (sumOver groups (gs[·]!.weights) : Rat) / (P : Rat)
    let h : Rat := (sumOver groups (gs[·]!.hbm) : Rat) / (P : Rat)
    let host : Rat := (sumOver groups (gs[·]!.host) : Rat) / (P : Rat)
    let reading (lost : Rat) : Reading :=
      ⟨lost, max 0 (w - lost) * (b : Rat), max 0 (h - lost) * (b : Rat), max 0 (h - w + host - lost) * (b : Rat)⟩
    out := out.push
      { stretch := m, lo, hi, spanNs := span, passes := P, lMaxNs := lMax, lSumNs := sumOver groups (L[·]!)
        weights := w, hbm := h, host, rebuild := R, fill
        time := reading (S + (2 * (lMax : Rat) * R + fill) / (span : Rat))
        snapshot := reading ((W + R) / (P : Rat) + S) }
  pure out

/-- The stretch whose reading credits the fewest weights bytes, the first of them on a tie. -/
def worst (xs : Array Stretch) (time : Bool) : Stretch :=
  xs.foldl (fun a x => if (x.reading time).weights < (a.reading time).weights then x else a) xs[0]!

/-- One form's name, charge and stretches. -/
structure Form where
  name : String
  charge : Charge
  stretches : Array Stretch

/-- A session's line in both forms, time-held then link-charged, against `nodeBytes` of node memory, with at most
`wrong` of its receipts wrong. -/
structure Line where
  nodeBytes : Nat
  forms : Array Form
  wrong : Rat

/-- The session's line on a node of `rates`, `nodeBytes` of memory and `chipBytes` of on-chip storage, with at most
W of its receipts wrong, the pooled bound of its record's `windows` (each window's receipts and draws) at `δ`; or why
it has none. -/
def line (gs : Array Group) (per blocks blockBytes chipBytes nodeBytes : Nat) (windows : Array (Nat × Nat)) (δ : Rat)
    (rates : Rates) : Except String Line := do
  let W := (← Pooled.pooled windows δ).wrong
  check gs per blocks blockBytes nodeBytes W rates
  let (t, l) := charges rates blockBytes
  let ts ← stretches gs per blocks blockBytes chipBytes W t
  let ls ← stretches gs per blocks blockBytes chipBytes W l
  pure ⟨nodeBytes, #[⟨"time-held", t, ts⟩, ⟨"link-charged", l, ls⟩], W⟩

/-- What the auditor saw of one window beside the Lean verifier's record of it: its coin's release and its last root's
arrival (ns, on one clock), the registered blocks its least pass read, and the registered blocks its passes read in
all, by kind. -/
structure Seen where
  releasedNs : Nat
  lastNs : Nat
  least : Nat
  weights : Nat
  hbm : Nat
  host : Nat
  deriving Inhabited

deriving instance Inhabited for Session.WindowSummary

/-- `U`, a window's receipts in the record: its passes' lengths summed. -/
def receipts (w : Session.WindowSummary) : Nat := w.lengths.toList.sum

/-- Why a record drawn more than one window a coin (`groups_per_coin`, `Shared`) at one window in flight has no line:
no property of the code covers its draws there (`Properties.PoubCheck.Live`'s `LiveIsRun` is at two). -/
def SHARED : String :=
  "a record drawn more than one window a coin (groups_per_coin) at one window in flight has no line: no property " ++
    "of the code covers its draws there"

/-- The groups and windows the line reads of an accepted session's record `sm` (`Session.State.summary`) and what the
auditor saw of each of its windows (`seen`): group `t` is window `t`, its passes the record's and its times and reads
the auditor's, and window `t`'s receipts and draws are the record's, `(Σ lengths_t, drawn_t)`. Refused unless the
auditor saw each of the record's windows and each window drew `min(s, U_t)` of its receipts at the session's `samples`
`s`, as the verifier's draw does (`Session.draws`). -/
def ofSummary (sm : Session.Summary) (samples : Nat) (seen : Array Seen) :
    Except String (Array Group × Array (Nat × Nat)) :=
  if seen.size ≠ sm.windows.size then
    .error s!"the auditor saw {seen.size} windows, not the record's {sm.windows.size}"
  else match sm.windows.findIdx? fun w => w.drawn != min samples (receipts w) with
    | some t =>
      .error (s!"window {t} drew {sm.windows[t]!.drawn} of its {receipts sm.windows[t]!} receipts, not "
        ++ s!"min({samples}, {receipts sm.windows[t]!})")
    | none =>
      .ok ((Array.range sm.windows.size).map fun t =>
          ({ releasedNs := seen[t]!.releasedNs, lastNs := seen[t]!.lastNs, passes := sm.windows[t]!.lengths.size,
             least := seen[t]!.least, weights := seen[t]!.weights, hbm := seen[t]!.hbm,
             host := seen[t]!.host } : Group),
        sm.windows.map fun w => (receipts w, w.drawn))

/-- The line of an accepted session's record `sm` on a node of `rates`, `nodeBytes` of memory and `chipBytes` of on-chip
storage, at the session's `samples`: `line` at the record's groups and windows (`ofSummary`) and its registered blocks
(`Summary.blocks`, the registration the session ran on), with W the pooled bound of those windows at `δ`; or why it
has none. -/
def lineOfRecord (sm : Session.Summary) (samples : Nat) (seen : Array Seen) (per blockBytes chipBytes nodeBytes : Nat)
    (δ : Rat) (rates : Rates) : Except String Line :=
  match ofSummary sm samples seen with
  | .ok (gs, ws) => line gs per sm.blocks blockBytes chipBytes nodeBytes ws δ rates
  | .error why => .error why

/-- The draws of a record drawn `g` windows a coin (`Shared`), as the pooled bound reads them
(`audit.pooled_windows`): draw `c` is windows `cg … min((c + 1)g, W) − 1`, its receipts `U` and its draws theirs
summed. -/
def drawsOf (g : Nat) (ws : Array Session.WindowSummary) : Array (Nat × Nat) :=
  (Array.range ((ws.size + g - 1) / g)).map fun c =>
    let xs := ws.extract (c * g) (c * g + g)
    (xs.foldl (fun a w => a + receipts w) 0, xs.foldl (fun a w => a + w.drawn) 0)

/-- `ofSummary` of a record drawn `g` windows a coin (`Shared`): `ofSummary` at g = 1; above it the same groups, one a
window, and the record's draws in place of its windows (`drawsOf`), refused unless each draw drew `min(s, U)` of its
`U` receipts at the session's `samples` `s`, as `Shared.close`'s draw does, and has fewer than `RECEIPT_LIMIT` (2^64)
receipts, as `receive` holds each window to: the memory model's draw reads at most `2^64 − 1` receipts a draw
(`Properties.PoubCheck.Line.receiptMax`), and a draw's windows' receipts, each below the limit, need not be. -/
def ofSummaryAt (g : Nat) (sm : Session.Summary) (samples : Nat) (seen : Array Seen) :
    Except String (Array Group × Array (Nat × Nat)) :=
  if g ≤ 1 then ofSummary sm samples seen
  else if seen.size ≠ sm.windows.size then
    .error s!"the auditor saw {seen.size} windows, not the record's {sm.windows.size}"
  else
    let ds := drawsOf g sm.windows
    match ds.findIdx? fun (u, d) => d != min samples u with
    | some c => .error s!"draw {c} drew {ds[c]!.2} of its {ds[c]!.1} receipts, not min({samples}, {ds[c]!.1})"
    | none =>
      match ds.findIdx? fun (u, _) => Session.RECEIPT_LIMIT ≤ u with
      | some c => .error s!"draw {c} has {ds[c]!.1} receipts, not fewer than 2^64"
      | none =>
        .ok ((Array.range sm.windows.size).map fun t =>
            ({ releasedNs := seen[t]!.releasedNs, lastNs := seen[t]!.lastNs, passes := sm.windows[t]!.lengths.size,
               least := seen[t]!.least, weights := seen[t]!.weights, hbm := seen[t]!.hbm,
               host := seen[t]!.host } : Group), ds)

/-- `lineOfRecord` of a record drawn `g` windows a coin: `line` at the groups and draws `ofSummaryAt` reads, its W the
pooled bound of the record's draws; at g = 1 `lineOfRecord`'s. -/
def lineOfRecordAt (g : Nat) (sm : Session.Summary) (samples : Nat) (seen : Array Seen)
    (per blockBytes chipBytes nodeBytes : Nat) (δ : Rat) (rates : Rates) : Except String Line :=
  match ofSummaryAt g sm samples seen with
  | .ok (gs, ws) => line gs per sm.blocks blockBytes chipBytes nodeBytes ws δ rates
  | .error why => .error why

/-- A window's last root's arrival in the record: its last pass's, or its coin's release when it has none (as
`verifier.paced_groups` reads a window seen). -/
def lastNs (w : Session.WindowSummary) : Nat := w.arrivedNs.back?.getD w.releasedNs

/-- Why the times `seen` aren't the record's own, or none: the first window whose coin's release or last root's arrival
seen isn't the one `sm` records (`releasedNs`, `lastNs`). -/
def timesOff (sm : Session.Summary) (seen : Array Seen) : Option String :=
  (List.range (min seen.size sm.windows.size)).findSome? fun t =>
    let s := seen[t]!
    let w := sm.windows[t]!
    if s.releasedNs == w.releasedNs && s.lastNs == lastNs w then none
    else some (s!"window {t} was seen released at {s.releasedNs} and last at {s.lastNs}, not at the record's "
      ++ s!"{w.releasedNs} and {lastNs w}")

/-- `lineOfRecordAt` with the times seen held to the record's where they are on its clock (`onRecord`): refused, after
`ofSummaryAt`'s refusals, where they differ (`timesOff`). Off it they are another clock's, the challenger's release
log, which the record doesn't hold, and this is `lineOfRecordAt`. -/
def lineOfRecordOn (onRecord : Bool) (g : Nat) (sm : Session.Summary) (samples : Nat) (seen : Array Seen)
    (per blockBytes chipBytes nodeBytes : Nat) (δ : Rat) (rates : Rates) : Except String Line :=
  match ofSummaryAt g sm samples seen, (if onRecord then timesOff sm seen else none) with
  | .ok _, some why => .error why
  | _, _ => lineOfRecordAt g sm samples seen per blockBytes chipBytes nodeBytes δ rates

/-- Why a record `sm` naming registration `digest` is not a record of registration `reg`: it names another, or its
registered blocks are not `reg`'s; none if it is. -/
def unregistered (reg : Registration) (digest : ByteArray) (sm : Session.Summary) : Option String :=
  if digest != reg.digest then some "the record is not of the registration given"
  else if sm.blocks != reg.blocks then
    some s!"the record's {sm.blocks} registered blocks are not the registration's {reg.blocks}"
  else none

/-- `lineOfRecordOn` of a record drawn `g` windows a coin that names registration `digest`, refused first unless it is
a record of the registration `reg` given with it (`unregistered`), so that the line credits that registration's
registered blocks. -/
def lineOfRegistered (onRecord : Bool) (g : Nat) (reg : Registration) (digest : ByteArray) (sm : Session.Summary)
    (samples : Nat) (seen : Array Seen) (per blockBytes chipBytes nodeBytes : Nat) (δ : Rat) (rates : Rates) :
    Except String Line :=
  match unregistered reg digest sm with
  | some why => .error why
  | none => lineOfRecordOn onRecord g sm samples seen per blockBytes chipBytes nodeBytes δ rates

/-- The record `Shared.run g` makes of a session drawn `g` windows a coin from `s` over `events`, none where the run
fails or doesn't accept (`Session.State.summary`): `LineHeld`'s `Record.summary` at a node of `g` windows a draw
(`Node.groupsPerCoin`), at g = 1 `Session.run`'s (`Shared.run_one`). -/
def summaryOfRun (g : Nat) (s : Session.State) (events : Array Session.Event) : Option Session.Summary :=
  match Shared.run g s events with
  | .ok (_, s') => s'.summary
  | .error _ => none

/-- `lineOfRecordOn` of a record, or `lineOf`'s refusal where there is none. -/
def lineOfSummary? (onRecord : Bool) (g : Nat) (sm? : Option Session.Summary) (samples : Nat) (seen : Array Seen)
    (per blockBytes chipBytes nodeBytes : Nat) (δ : Rat) (rates : Rates) : Except String Line :=
  match sm? with
  | some sm => lineOfRecordOn onRecord g sm samples seen per blockBytes chipBytes nodeBytes δ rates
  | none => .error "the Lean verifier gave the session no record"

/-- The line of a session drawn `g` windows a coin run from `s` over `events`, as `LineHeld`'s `lineOf` reads a record
at a node of `g` windows a draw: `lineOfRecordOn` of the run's own record (`summaryOfRun`), or why it has none. -/
def lineOfRun (onRecord : Bool) (g : Nat) (s : Session.State) (events : Array Session.Event) (samples : Nat)
    (seen : Array Seen) (per blockBytes chipBytes nodeBytes : Nat) (δ : Rat) (rates : Rates) : Except String Line :=
  lineOfSummary? onRecord g (summaryOfRun g s events) samples seen per blockBytes chipBytes nodeBytes δ rates

private def digits? (s : String) : Option Nat :=
  if !s.isEmpty && s.all Char.isDigit then s.toNat? else none

/-- A rational as the JSON form writes it, `n` or `n/d` with d positive, an optional minus first (`line.number`). -/
def ratOf? (s : String) : Option Rat := do
  let neg := s.startsWith "-"
  let body := if neg then (s.drop 1).toString else s
  let sign : Int → Int := fun x => if neg then -x else x
  match body.splitOn "/" with
  | [a] => pure (sign (← digits? a))
  | [a, d] =>
    let d ← digits? d
    if d = 0 then none else pure (mkRat (sign (← digits? a)) d)
  | _ => none

end PoubCheck.Line
