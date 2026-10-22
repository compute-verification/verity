/-
# The drive of a rep's frame (live PROTOCOL.md §10.1): its start and every request, from the statement and the coins

A firewalled rep's frame (`Frame.step`, folded by `Frame.run`) takes one schedule of requests, which this fixes:

* `start`: the frame's start, the rep's pads layout at the statement's `--zk` schedule and the statement's mask slot
  (`maskSlot`), which `flock-circuit` sends as `{"layout", "mask"}` (`Frame.start`).
* `extras`: the extra claims' coordinates at the session's link points (`linkPoints`, the `Commit`'s coins), which the
  opening takes (`Stmt.regionPoint` past its six skip coordinates, `flock-circuit`'s `extra_points(..)[6..]`).
* `schedule`: the scheduled rounds, the algebraic phase's (`Skeleton.rounds`) and then the opening's (`opening`, the
  Lean finish's `Finish.openLive` recorded on the stream's coins), each the round in whose place a device round leaves.
* `reqs`: the frame's requests, **the one schedule**: rep 1's earlier rep's algebraic-phase coins (`prior`), each
  scheduled round with the device's round in its place and its coins, each of the inner proof's rounds (the firewall's
  own) with its coins, then `finish`. At the scheduled rounds themselves in the device's place (`reqsAt`) they are a
  function of the statement, the table's level 0, the rep's salts and the coins alone.
* `encode`, `parse`: a request as `Frame.serveWith` reads it.
* `cmd`: `flock-firewall drive`, which `firewall_finish_agree.py` holds to what `flock-circuit` sent the rep's frame
  (`table{T}-rep{R}.start`, `.open`, `.frame`) byte for byte.
-/
import Flock.Firewall.Skeleton

namespace Flock.Firewall.Drive

open Lean (Json toJson)
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

/-- The start as `Frame.start` reads it. -/
def startJson (l : Zk.PadLayout) (m : Frame.MaskSlot) : Json :=
  Json.mkObj [("layout", Json.mkObj [("n_mlv", toJson l.nMlv), ("n_lc", toJson l.nLc), ("n_ood0", toJson l.nOod0),
                                     ("k0", toJson l.k0)]),
              ("mask", Json.mkObj [("w0", toJson m.w0), ("n", toJson m.n), ("blocks", toJson m.blocks)])]

/-- The link points in the `Commit`'s coins `c`: a point each `mPts` words. -/
def linkPoints (st : Stmt) (c : Array F128) : Array (Array F128) :=
  (Array.range (c.size / st.mPts)).map fun i => c.extract (i * st.mPts) ((i + 1) * st.mPts)

/-- **The extra claims' coordinates** at the link points `pts`, region-major, then point by point (`Stmt.extraClaims`'s
order): each region's point past its six skip coordinates. -/
def extras (st : Stmt) (pts : Array (Array F128)) : Array (Array F128) :=
  st.regions.flatMap fun reg => pts.map fun r => let p := st.regionPoint reg r; p.extract 6 p.size

/-! ## The scheduled rounds -/

/-- The rounds recorded so far, each its framed bytes and coin count, and the rounds of coins taken. -/
abbrev RecM := StateT (Array (ByteArray × Nat) × Nat) V

/-- **The recording `Asks`** on the stream's coins `coins`: round `k` answered by `coins[k]`, which must hold the round's
coin count (its squeeze header's last eight bytes); each round from `keep` on is kept with its count. -/
def recAsks (coins : Array (Array F128)) (keep : Nat) : Asks RecM where
  link _ _ := throw "drive: a link message inside a rep"
  round c := do
    let (rs, k) ← get
    let some cs := coins[k]? | throw s!"drive: round {k} past the stream's {coins.size} rounds of coins"
    let n := (u64OfLE c (c.size - 8)).toNat
    if cs.size != n then throw s!"drive: round {k} squeezes {n} coins where the stream answered {cs.size}"
    set (if k < keep then rs else rs.push (c, n), k + 1)
    return fun i => cs[i]!

/-- **A rep's opening's scheduled rounds**, `Finish.openLive`'s recorded on the stream's coins: its PIOP at zero values
on the algebraic phase's `nAlg` rounds of coins, every one of which it must take, then its opening on `abcClaims ++` the
extra claims at `xs`, every value and pad zero, levels ≥ 1 under `hiSalts`, each round answered by the stream's next
coins. -/
def opening (t : Tab) (rep : Nat) (dom : ByteArray) (xs : Array (Array F128)) (hiSalts : Array ByteArray)
    (coins : Array (Array F128)) (nAlg : Nat) : V (Array (ByteArray × Nat)) := do
  let lay := Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0
  let commit : Level → Array F128 → (Nat → ByteArray) → RecM Commit := fun lv g salt => pure (commitFw lv g salt)
  let salt : Nat → Nat → ByteArray := fun l => saltAt (hiSalts.getD (l - 1) ByteArray.empty)
  let io := recAsks coins nAlg
  let prog : TMW RecM Unit := do
    let (_, _, claims) ← repPiop io t.st lay (flat t.cm0.cap) ByteArray.empty (claimsOf xs) fun _ => 0
    let (_, k) ← getThe (Array (ByteArray × Nat) × Nat)
    if k != nAlg then throw s!"drive: the rep's PIOP took {k} of the algebraic phase's {nAlg} rounds of coins"
    let _ ← repOpening io commit t.st t.sch lay t.tPad rep t.g t.cm0 claims (Array.replicate lay.len 0) t.extra
      t.lanePad salt
  let (_, rs, _) ← (prog.run' (Transcript.init #[] dom)).run (#[], 0)
  return rs

/-- **A rep's scheduled rounds** on the table `t`, as rep `rep` of a stream whose transcript starts at `dom`: the
algebraic phase's (`Skeleton.rounds`, on level 0's cap), then the opening's (`opening`), on the stream's coins. -/
def schedule (t : Tab) (rep : Nat) (dom : ByteArray) (xs : Array (Array F128)) (hiSalts : Array ByteArray)
    (coins : Array (Array F128)) : V (Array (ByteArray × Nat)) := do
  let alg ← Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom
  return alg ++ (← opening t rep dom xs hiSalts coins alg.size)

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

/-- The requests at the scheduled rounds themselves in the device's place: an honest device's rounds but for the values
each absorbs masked, which move no frame state. -/
def reqsAt (prior : Array F128) (sched own : Array (ByteArray × Nat)) (coins : Array (Array F128)) : List Frame.Req :=
  reqs prior sched own (fun k => (sched.getD k (.empty, 0)).1) coins

/-- A request as `Frame.serveWith` reads it: its line, then its bytes. -/
def encode : Frame.Req → ByteArray
  | .round n c (.sched r) => s!"round {n} {c.size} {r.size}\n".toUTF8 ++ c ++ r
  | .round n c .fail => s!"round {n} {c.size} fail\n".toUTF8 ++ c
  | .round n c .own => s!"round {n} {c.size}\n".toUTF8 ++ c
  | .coins (.ok cs) => s!"coins {cs.size}\n".toUTF8 ++ Frame.wordBytes cs
  | .coins (.error _) => "coins 0\n".toUTF8
  | .prior k b => s!"prior {k}\n".toUTF8 ++ b
  | .finish => "finish\n".toUTF8

/-- The first `\n` of `b` at or after byte `o`. -/
def lineEnd (b : ByteArray) (o : Nat) : Option Nat := Id.run do
  for i in [o:b.size] do
    if b.get! i == 10 then return some i
  return none

/-- `n` bytes of `b` from byte `o`, and the byte after them. -/
def takeAt (b : ByteArray) (o n : Nat) : V (ByteArray × Nat) :=
  if o + n ≤ b.size then pure (b.extract o (o + n), o + n) else throw "drive: a request ends inside its bytes"

/-- **The requests in `b`**, as `Frame.serveWith` reads them, coins as the request carries them. -/
def parse (b : ByteArray) : V (Array Frame.Req) := do
  let mut o := 0
  let mut out := #[]
  for _ in [0:b.size] do
    if o ≥ b.size then break
    let some e := lineEnd b o | throw "drive: a request line without its end"
    let line := (String.fromUTF8? (b.extract o e)).getD "?"
    o := e + 1
    let nat := fun (s : String) => match s.toNat? with
      | some n => pure n
      | none => throw s!"drive: a request `{line}`"
    match line.splitOn " " with
    | ["round", n, len, "fail"] =>
      let (c, o') ← takeAt b o (← nat len)
      out := out.push (.round (← nat n) c .fail)
      o := o'
    | ["round", n, len, r] =>
      let (c, o') ← takeAt b o (← nat len)
      let (s, o'') ← takeAt b o' (← nat r)
      out := out.push (.round (← nat n) c (.sched s))
      o := o''
    | ["round", n, len] =>
      let (c, o') ← takeAt b o (← nat len)
      out := out.push (.round (← nat n) c .own)
      o := o'
    | ["coins", k] =>
      let k ← nat k
      let (w, o') ← takeAt b o (16 * k)
      out := out.push (.coins (.ok (Frame.words w 0 k)))
      o := o'
    | ["prior", k] =>
      let k ← nat k
      let (w, o') ← takeAt b o (16 * k)
      out := out.push (.prior k w)
      o := o'
    | ["finish"] => out := out.push .finish
    | _ => throw s!"drive: a request `{line}`"
  return out

/-- What a frame's requests carry: the device's round at each scheduled round, the inner proof's rounds and the coins of
each round, in order. -/
def carried (qs : Array Frame.Req) : Array ByteArray × Array (ByteArray × Nat) × Array (Array F128) :=
  qs.foldl (init := (#[], #[], #[])) fun (dev, own, cs) q => match q with
    | .round _ c (.sched _) | .round _ c .fail => (dev.push c, own, cs)
    | .round n c .own => (dev, own.push (c, n), cs)
    | .coins (.ok c) => (dev, own, cs.push c)
    | _ => (dev, own, cs)

/-! ## The difftest -/

/-- A vector's words after their `u64` count. -/
def vecBytes (v : Array F128) : ByteArray := le64 v.size.toUInt64 ++ Frame.wordBytes v

/-- An opening's request as `Finish.parseOpen` reads it: the extra claims, then the algebraic phase's coins. -/
def openBytes (xs coins : Array (Array F128)) : ByteArray :=
  (coins.foldl (· ++ vecBytes ·) (xs.foldl (· ++ vecBytes ·) (le64 xs.size.toUInt64) ++ le64 coins.size.toUInt64))

/-- The salts as `Finish.saltsBytes` writes them. -/
def parseSalts (b : ByteArray) : V (Array ByteArray) :=
  let rd : Finish.Rd (Array ByteArray) := do
    let n ← u64 "the salts"
    let mut out := #[]
    for _ in [0:n] do
      let k ← u64 "a level's salts"
      out := out.push (← take k "a level's salts")
    if (← get) != b.size then throw "drive: bytes after the salts"
    return out
  (rd.run' 0).run b

/-- Bytes after their `u64` length. -/
def blob (b : ByteArray) : ByteArray := le64 b.size.toUInt64 ++ b

/-- **Rep `rep` of table `i` driven in Lean** on what `flock-circuit` dumped in `dir` (`FC_FINISH_DUMP`): the table's
level-0 request (`table{I}.req`), the session's `Commit` coins (`link.bin`), the rep's salts for levels ≥ 1
(`table{I}-rep{R}.hi`), and the requests it sent the rep's frame (`table{I}-rep{R}.frame`, and rep 0's for rep 1's
`prior`), of which only the device's rounds, the inner proof's rounds and the coins are read. Returns the frame's start
(`startJson`), the opening's request (`openBytes`: the extra claims and the algebraic phase's coins) and the requests
(`reqs` on `schedule`'s rounds, `encode`d), each a `blob`. -/
def driveAt (st : Stmt) (t : Tab) (dom : ByteArray) (dir : System.FilePath) (i rep : Nat) : IO (V ByteArray) := do
  let link ← IO.FS.readBinFile (dir / "link.bin")
  let hi ← IO.FS.readBinFile (dir / s!"table{i}-rep{rep}.hi")
  let log ← IO.FS.readBinFile (dir / s!"table{i}-rep{rep}.frame")
  let log0 ← if rep == 0 then pure ByteArray.empty else IO.FS.readBinFile (dir / s!"table{i}-rep0.frame")
  return do
    let (lay, mask) ← start st
    let xs := extras st (linkPoints st (Link.wordsAt link 0 (link.size / 16)))
    let hiSalts ← parseSalts hi
    let (dev, own, coins) := carried (← parse log)
    let alg ← Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom
    let nAlg := alg.size
    let sched := alg ++ (← opening t rep dom xs hiSalts coins nAlg)
    let prior ← if rep == 0 then pure #[] else do
      let (_, _, c0) := carried (← parse log0)
      pure ((c0.extract 0 nAlg).foldl (· ++ ·) #[])
    let qs := reqs prior sched own (fun k => dev.getD k .empty) coins
    let enc := qs.foldl (fun b q => b ++ encode q) ByteArray.empty
    return blob (startJson lay mask).compress.toUTF8 ++ blob (openBytes xs (coins.extract 0 nAlg)) ++ blob enc

/-- **`flock-firewall drive FLAGS`** (`Skeleton.FLAGS`): on the statement `flock-firewall finish` sets up, each request on
standard input `drive I R LEN` followed by the `LEN` bytes of a dump directory's path is answered `ok N` followed by the
`N` bytes of `driveAt` on it, table `I`'s state read from its level-0 request once; anything else, or a request it
refuses, is answered `err WHY` and ends it, with exit code `1`. -/
def cmd (args : List String) : IO UInt32 := do
  let ch ← Frame.stdio
  let say := fun (s : String) => do
    ch.write (s ++ "\n").toUTF8
    ch.flush
  let refuse := fun (why : String) => do
    say s!"err {why.replace "\n" " "}"
    return (1 : UInt32)
  let (spec, sts) ← match ← Skeleton.statementOf "drive" args with
    | .ok s => pure s
    | .error e => return (← refuse e)
  let mut tabs : Array (Option Tab) := Array.replicate sts.size none
  repeat
    let line := (← ch.getLine).replace "\n" ""
    if line.isEmpty then break
    match line.splitOn " " with
    | ["drive", i, r, len] =>
      let (some i, some r, some len) := (i.toNat?, r.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
      let dir : System.FilePath := (String.fromUTF8? (← ch.readExact len)).getD ""
      let (some st, some name) := (sts[i]?, spec.tables[i]?)
        | return (← refuse s!"table {i} of the session's {sts.size}")
      let t ← match tabs[i]?.join with
        | some t => pure t
        | none =>
          match tabOf (Setup.ofCircuit st) (← IO.FS.readBinFile (dir / s!"table{i}.req")) with
          | .ok t => pure t
          | .error e => return (← refuse e)
      tabs := tabs.set! i (some t)
      match ← driveAt st t (spec.domain name r).toUTF8 dir i r with
      | .ok b =>
        say s!"ok {b.size}"
        ch.write b
        ch.flush
      | .error e => return (← refuse e)
    | _ => return (← refuse s!"a request `{line}`")
  return 0

end Flock.Firewall.Drive
