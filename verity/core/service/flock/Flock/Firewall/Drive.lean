/-
# The drive of a rep's frame (live PROTOCOL.md §10.1): its start and every request, from the statement and the coins

A firewalled rep's frame (`Frame.step`, which the session loop steps, `Frames`) takes one schedule of requests, which
this fixes:

* `start`: the frame's start, the rep's pads layout at the statement's `--zk` schedule and the statement's mask slot
  (`maskSlot`).
* `extras`: the extra claims' coordinates at the session's link points (`linkPoints`, the `Commit`'s coins), which the
  opening takes (`Stmt.regionPoint` past its six skip coordinates, `flock-circuit`'s `extra_points(..)[6..]`).
* `openStep`: the rep's opening a round at a time, the scheduled rounds after the algebraic phase's
  (`Skeleton.rounds`), each the round in whose place a device round leaves; `opening` feeds it a stream's coins.
* `reqs`: the frame's requests, **the one schedule**: rep 1's earlier rep's algebraic-phase coins (`prior`), each
  scheduled round with the device's round in its place and its coins, each of the inner proof's rounds (the firewall's
  own) with its coins, then `finish`.
-/
import Flock.Firewall.Skeleton

namespace Flock.Firewall.Drive

open Flock Transcript Finish

/-! ## The start -/

/-- **The statement's mask slot** (`flock-circuit`'s `mask_slot`): its mask range's first word, its words a slot and the
statement's blocks. -/
def maskSlot (st : Stmt) : V Frame.MaskSlot := do
  let some r := st.c.ranges.find? (·.kind == Kind.mask) | throw "drive: the statement has no mask slot"
  if r.slotLog < 7 then throw s!"drive: a mask slot of {2 ^ r.slotLog} columns, under a word"
  return ⟨r.start / 128, 2 ^ (r.slotLog - 7), 2 ^ st.nbl⟩

/-- **A rep's frame start** at the statement `st`: its pads layout at the `--zk` schedule's `k0`, and the mask slot. -/
def start (st : Stmt) : V (Zk.PadLayout × Frame.MaskSlot) := do
  let (sch, _) ← Zk.schedule st.m
  return (Zk.PadLayout.new st.m st.c.kLog sch.k0, ← maskSlot st)

/-- The link points in the `Commit`'s coins `c`: a point each `mPts` words. -/
def linkPoints (st : Stmt) (c : Array F128) : Array (Array F128) :=
  (Array.range (c.size / st.mPts)).map fun i => c.extract (i * st.mPts) ((i + 1) * st.mPts)

/-- **The extra claims' coordinates** at the link points `pts`, region-major, then point by point (`Stmt.extraClaims`'s
order): each region's point past its six skip coordinates. -/
def extras (st : Stmt) (pts : Array (Array F128)) : Array (Array F128) :=
  st.regions.flatMap fun reg => pts.map fun r => let p := st.regionPoint reg r; p.extract 6 p.size

/-! ## The scheduled rounds -/

/-- **A program that asks for coins as it goes**: it ended (`done`), stopped (`fail`), or sends a round, its framed
bytes `c` squeezing `n` coins, and goes on as `k` of the coins answered to it. -/
inductive Step (α : Type) where
  | done (a : α)
  | fail (e : String)
  | ask (c : ByteArray) (n : Nat) (k : Array F128 → Step α)

namespace Step

/-- `s`, then `f` of what it ends with. -/
def bind {α β : Type} : Step α → (α → Step β) → Step β
  | .done a, f => f a
  | .fail e, _ => .fail e
  | .ask c n k, f => .ask c n fun cs => (k cs).bind f

/-- `s`, and `h` of why it stopped where it does. -/
def tryCatch {α : Type} : Step α → (String → Step α) → Step α
  | .done a, _ => .done a
  | .fail e, h => h e
  | .ask c n k, h => .ask c n fun cs => (k cs).tryCatch h

instance instMonad : Monad Step where
  pure := .done
  bind := bind

instance instMonadExceptOfString : MonadExceptOf String Step where
  throw := .fail
  tryCatch := tryCatch

instance instMonadLiftTV : MonadLiftT V Step where
  monadLift
    | .ok a => .done a
    | .error e => .fail e

/-- **`s` fed the coins `cs`**, each answering its next round: what it goes on as, and the rounds it sent, each its bytes
and coin count. -/
def feed {α : Type} : Step α → List (Array F128) → Array (ByteArray × Nat) → Step α × Array (ByteArray × Nat)
  | .ask c n k, cs :: rest, rs => feed (k cs) rest (rs.push (c, n))
  | s, _, rs => (s, rs)

end Step

/-- A rep's opening as it runs: the rounds of coins it took. -/
abbrev OpenM := StateT (Array (Array F128)) Step

/-- **The opening's `Asks`** (`Finish.liveAsks`, its channel a `Step.ask`): round `k` answered by `coins[k]` where the
algebraic phase's coins have it, else, when `live`, by the coins the round is asked for; each must hold the round's
coin count (its squeeze header's last eight bytes). -/
def stepAsks (coins : Array (Array F128)) (live : Bool) : Asks OpenM where
  link _ _ := throw "drive: a link message inside a rep"
  round c := do
    let got ← get
    let k := got.size
    let n := (u64OfLE c (c.size - 8)).toNat
    let cs ← match coins[k]?, live with
      | some cs, _ => pure cs
      | none, false => throw s!"drive: round {k} past the algebraic phase's {coins.size} rounds of coins"
      | none, true => StateT.lift (Step.ask c n .done)
    if cs.size != n then throw s!"drive: round {k} squeezes {n} coins where the stream answered {cs.size}"
    set (got.push cs)
    return fun i => cs[i]!

/-- **A rep's opening, round by round** (`Finish.openLive` with no channel): its PIOP at zero values on the algebraic
phase's rounds of coins `coins`, every one of which it must take, then its opening on `abcClaims ++` the extra claims at
`xs`, every value and pad zero, levels ≥ 1 under `hiSalts`, each round asked for its coins (`Step.ask`). The session
loop runs it a round at a time on the coins it opens (`Frames`); `opening` runs it on a stream's. -/
def openStep (t : Tab) (rep : Nat) (dom : ByteArray) (xs : Array (Array F128)) (hiSalts : Array ByteArray)
    (coins : Array (Array F128)) : Step Unit :=
  let lay := Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0
  let commit : Level → Array F128 → (Nat → ByteArray) → OpenM Commit := fun lv g salt => pure (commitFw lv g salt)
  let salt : Nat → Nat → ByteArray := fun l => saltAt (hiSalts.getD (l - 1) ByteArray.empty)
  let prog : TMW OpenM Unit := do
    let (_, _, claims) ← repPiop (stepAsks coins false) t.st lay (flat t.cm0.cap) ByteArray.empty (claimsOf xs)
      fun _ => 0
    let k := (← getThe (Array (Array F128))).size
    if k != coins.size then throw s!"drive: the rep's PIOP took {k} of its {coins.size} rounds of coins"
    let _ ← repOpening (stepAsks coins true) commit t.st t.sch lay t.tPad rep t.g t.cm0 claims
      (Array.replicate lay.len 0) t.extra t.lanePad salt
  ((prog.run' (Transcript.init #[] dom)).run' #[])

/-- **A rep's opening's scheduled rounds** on the stream's coins `coins`: `openStep` on the algebraic phase's `nAlg`
rounds of them, fed the rest until it ends. -/
def opening (t : Tab) (rep : Nat) (dom : ByteArray) (xs : Array (Array F128)) (hiSalts : Array ByteArray)
    (coins : Array (Array F128)) (nAlg : Nat) : V (Array (ByteArray × Nat)) :=
  match (openStep t rep dom xs hiSalts (coins.extract 0 nAlg)).feed (coins.extract nAlg coins.size).toList #[] with
  | (.done _, rs) => .ok rs
  | (.fail e, _) => .error e
  | (.ask .., rs) => .error s!"drive: the opening's round {nAlg + rs.size} past the stream's {coins.size} rounds of coins"

/-! ## The requests -/

/-- **A rep's requests to its frame**, the one schedule its frame is driven on: the earlier rep's algebraic-phase coins
`prior` (rep 1's; none for rep 0), then each scheduled round `sched[k]` with the device's round `dev k` in its place and
the stream's coins `coins[k]` for it, then each of the inner proof's rounds `own[j]`, the firewall's own, with its coins
`coins[sched.size + j]`, then `finish`. -/
def reqs (prior : Array F128) (sched own : Array (ByteArray × Nat)) (dev : Nat → ByteArray)
    (coins : Array (Array F128)) : List Frame.Req :=
  let pre := if prior.isEmpty then [] else [Frame.Req.prior prior.size (Frame.wordBytes prior)]
  let at_ (rs : Array (ByteArray × Nat)) (k : Nat) : ByteArray × Nat := rs.getD k (.empty, 0)
  let rounds := (List.range sched.size).flatMap fun k =>
    [.round (at_ sched k).2 (dev k) (.sched (at_ sched k).1), .coins (.ok (coins.getD k #[]))]
  let inner := (List.range own.size).flatMap fun j =>
    [.round (at_ own j).2 (at_ own j).1 .own, .coins (.ok (coins.getD (sched.size + j) #[]))]
  pre ++ rounds ++ inner ++ [.finish]

end Flock.Firewall.Drive
