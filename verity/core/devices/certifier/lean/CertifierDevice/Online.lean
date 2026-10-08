import NetworkCertifier.Schedule
import Std.Data.HashMap

/-!
# The online prover: each window's sync chosen `lag` buckets before its first bucket

The executable form of `online.py`'s `Prover` on the one egress link a `Shaper` feeds it (link 0; the shaper never feeds
another, so a prover's other links hold nothing and change no declaration). Window w's choice (`choose`) reads the
frames ready by w·T − lag and the earlier declarations, and declares nothing; `declare` declares a choice made so.
`snapshot().choose()` is `choose` on what the prover has read by then, so this has no snapshot. The rules are
`cover-K-M[-fallback]` (`Rule`), `last` being `cover-1-0`, which ignore the earlier declarations. `online.py` caches
each window's summary and the hand time before it; both are functions of the frames seen and the declarations made by
then, so this computes them afresh. Eligibility is `NetworkCertifier.eligible`.
-/

namespace CertifierDevice

open NetworkCertifier (ClockSync SyncSet LogicalFrame eligible)

/-- `online.Rule` of `cover(k, margin, fallback)`: its depth is `k`. -/
structure Rule where
  k : Nat
  margin : Nat
  fallback : Bool

/-- `online.Past`: an earlier window as the prover sees it at a declaration. `frontier[i]`, for the i-th ρ of Σ_sync in
ascending order, is the least δ that meets the window's seen frames and their total hand time then, or none. -/
structure Past where
  window : Nat
  sync : ClockSync
  frames : Nat
  late : Bool
  frontier : Array (Option (Nat × Nat))

/-- Σ_sync's deltas, ascending. -/
def sortedDeltas (ss : SyncSet) : Array Nat := (ss.deltas.mergeSort (· ≤ ·)).toArray

/-- Σ_sync's rates, ascending, each with its membership (so a value built from it is a `ClockSync`). -/
def sortedRhos (ss : SyncSet) : Array {ρ // ρ ∈ ss.rhos} :=
  (ss.rhos.attach.mergeSort fun a b => decide (a.1 ≤ b.1)).toArray

def mkSync (ss : SyncSet) (d : Nat) (ρ : {ρ // ρ ∈ ss.rhos}) : ClockSync := ⟨d, ρ.1, ss.rhos_pos ρ.1 ρ.2⟩

/-- `ClockSync`'s equality, as Python's dataclass compares (δ, ρ). -/
def ClockSync.beq (a b : ClockSync) : Bool := a.delta == b.delta && decide (a.rho = b.rho)

/-- `sync in sync_set`. -/
def inSet (ss : SyncSet) (s : ClockSync) : Bool := decide (s.delta ∈ ss.deltas) && decide (s.rho ∈ ss.rhos)

/-- Frame k's logical release bucket, from its anchor and index (`_Link._eligible`). -/
def eligibleAt (anchor index : Nat) (s : ClockSync) : Nat := eligible ⟨[], anchor, index⟩ s

/-- `online._Link`: the egress frames as the prover has read them, by canonical position: the contiguous prefix
(`anchor`, `index`, `ready`), those read ahead of it (`early`), how many are seen, and the seen ones by anchor window. -/
structure Link where
  T : Nat
  anchor : Array Nat := #[]
  index : Array Nat := #[]
  ready : Array Int := #[]
  early : Std.HashMap Nat (Nat × Nat × Int) := {}
  seen : Nat := 0
  positions : Std.HashMap Nat (Array Nat) := {}

namespace Link

/-- `_Link.arrive`: a position already held is ignored. -/
def arrive (l : Link) (position anchor index : Nat) (ready : Int) : Link := Id.run do
  if position < l.anchor.size || l.early.contains position then return l
  let mut l := { l with early := l.early.insert position (anchor, index, ready) }
  for _ in [0:l.early.size] do
    match l.early.get? l.anchor.size with
    | none => break
    | some (a, i, r) =>
      l := { l with early := l.early.erase l.anchor.size, anchor := l.anchor.push a, index := l.index.push i,
                    ready := l.ready.push r }
  return l

/-- `_Link.see`: extend the seen prefix to the frames ready by `bucket`, and whether one of them is ready no later than
its anchor's bucket. -/
def see (l : Link) (bucket : Int) : Link × Bool := Id.run do
  let mut l := l
  let mut early := false
  for k in [l.seen:l.ready.size] do
    if l.ready[k]! ≤ bucket then
      early := early || (l.anchor[k]! : Int) ≥ l.ready[k]!
      let w := l.anchor[k]! / l.T
      l := { l with positions := l.positions.insert w ((l.positions.getD w #[]).push k), seen := k + 1 }
    else break
  return (l, early)

def count (l : Link) (u : Nat) : Nat := (l.positions.getD u #[]).size

/-- `_Link.h_before`: the most eligibility, under `syncs`, of the frames before window `u`'s first seen one. -/
def hBefore (l : Link) (u : Nat) (syncs : Array ClockSync) : Nat := Id.run do
  let first := (l.positions.getD u #[])[0]!
  let mut h := 0
  for k in [0:first] do
    match syncs[l.anchor[k]! / l.T]? with
    | some s => h := max h (eligibleAt l.anchor[k]! l.index[k]! s)
    | none => pure ()
  return h

/-- `_Link.cost`: window u's seen frames' total hand time with u at `sync` and every other window as declared, or none if
one is late. -/
def cost (l : Link) (u : Nat) (sync : ClockSync) (syncs : Array ClockSync) : Option Nat := Id.run do
  let pos := l.positions.getD u #[]
  let mut h := l.hBefore u syncs
  let mut total := 0
  for k in [pos[0]!:pos.back! + 1] do
    let a := l.anchor[k]!
    let w := a / l.T
    let s? := if w = u then some sync else syncs[w]?
    match s? with
    | none => pure ()
    | some s =>
      h := max (eligibleAt a l.index[k]! s) h
      if w = u then
        if l.ready[k]! > h then return none
        total := total + h
  return some total

end Link

/-- `online.Prover` of one spatial unit, on the shaper's link. Window w's choice sees the frames ready by w·T − `lag`.
`syncs` holds its own declarations so far, `prior` the latest `depth` windows of an earlier run. -/
structure Prover where
  syncSet : SyncSet
  T : Nat
  rule : Rule
  envelope : ClockSync
  lag : Nat
  link : Link
  syncs : Array ClockSync := #[]
  prior : Array Past := #[]

/-- `xs[-n:]`. -/
def lastN {α : Type} (xs : Array α) (n : Nat) : Array α := xs.extract (xs.size - n) xs.size

/-- `cover(k, margin, fallback).choose`: per ρ ascending, δ is the largest frontier δ over the recent windows plus the
margin, rounded up to Σ_sync; the value taken has the least estimated total hand time, ties to the smaller δ, then the
larger ρ. With no history, or (`fallback`) one of the recent windows late, the envelope. -/
def Rule.choose (rule : Rule) (past : Array Past) (ss : SyncSet) (envelope : ClockSync) : ClockSync := Id.run do
  let recent := lastN past rule.k
  if recent.isEmpty || (rule.fallback && recent.any (·.late)) then return envelope
  let deltas := sortedDeltas ss
  let rhos := sortedRhos ss
  let mut best : Option ((Nat × Nat × Nat) × ClockSync) := none
  for (ρ, i) in rhos.zipIdx do
    let xs := recent.map fun p => (p.frontier[i]?).join
    if xs.any Option.isNone then break
    let xs := xs.filterMap id
    let m := xs.foldl (fun acc x => max acc x.1) 0 + rule.margin
    match deltas.findIdx? (· ≥ m) with
    | none => break
    | some j =>
      let d := deltas[j]!
      let c := (recent.zip xs).foldl (fun acc (p, dm, c) => acc + c + p.frames * (d - dm)) 0
      let better := match best with
        | none => true
        | some ((c', d', i'), _) => c < c' || (c == c' && (d < d' || (d == d' && i > i')))
      if better then best := some ((c, d, i), mkSync ss d ρ)
  return (best.map (·.2)).getD envelope

namespace Prover

def make (ss : SyncSet) (T : Nat) (rule : Rule) (envelope : ClockSync) (prior : Array Past) (lag : Nat) : Prover :=
  { syncSet := ss, T, rule, envelope, lag, link := { T }, prior := lastN prior rule.k }

/-- `Prover.arrive` on link 0. -/
def arrive (p : Prover) (position anchor index : Nat) (ready : Int) : Prover :=
  { p with link := p.link.arrive position anchor index ready }

/-- `_summarize`: window u's frontier per ρ ascending (each ρ's search starting at the δ the last one found), and whether
u was late under its own declaration. -/
def summarize (p : Prover) (u frames : Nat) : Past := Id.run do
  let deltas := sortedDeltas p.syncSet
  let rhos := sortedRhos p.syncSet
  let cost := fun s => p.link.cost u s p.syncs
  let mut frontier := #[]
  let mut lo : Option Nat := some 0
  for ρ in rhos do
    match lo with
    | none => frontier := frontier.push none
    | some l0 =>
      match cost (mkSync p.syncSet deltas.back! ρ) with
      | none =>
        lo := none
        frontier := frontier.push none
      | some c =>
        let mut l := l0
        let mut hi := deltas.size - 1
        let mut cHi := c
        for _ in [0:deltas.size] do
          if l < hi then
            let mid := (l + hi) / 2
            match cost (mkSync p.syncSet deltas[mid]! ρ) with
            | none => l := mid + 1
            | some c => hi := mid; cHi := c
        frontier := frontier.push (some (deltas[l]!, cHi))
        lo := some l
  let own := p.syncs[u]?.getD p.envelope
  return ⟨u, own, frames, (cost own).isNone, frontier⟩

/-- `Prover.choose` (and `snapshot`, whose choice it is): the next window's sync, from the frames ready by its first
bucket less `lag` and the earlier declarations, the prover having read those frames, and whether it was refused
(`online.Refused`): a frame it reads first then is ready no later than its anchor's bucket. Python raises, and its next
call, on the frames read, is the window's: the sync returned here. It declares nothing. -/
def choose (p : Prover) : Prover × ClockSync × Bool := Id.run do
  let w := p.syncs.size
  let (link, refused) := p.link.see ((w * p.T : Nat) - (p.lag : Int))
  let p := { p with link }
  let mut seen := #[]
  for u in [w - p.rule.k:w] do
    let n := p.link.count u
    if n > 0 then seen := seen.push (p.summarize u n)
  return (p, p.rule.choose (lastN (p.prior ++ seen) p.rule.k) p.syncSet p.envelope, refused)

/-- `Prover.declare(sync)`: declare the next window's sync, a choice made elsewhere. -/
def declare (p : Prover) (sync : ClockSync) : Prover := { p with syncs := p.syncs.push sync }

end Prover

end CertifierDevice
