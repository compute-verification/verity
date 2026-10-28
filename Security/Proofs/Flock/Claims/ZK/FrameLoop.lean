import Proofs.Flock.Claims.ZK.SessionLean
import Proofs.Flock.Claims.ZK.FrameOwn

/-!
# The release at the session loop's own frames

`LeanRelease.ProgReads` and `LeanRelease.ProgChecks` take the transport's frame state (`prog`) as given: the state Rust's
requests leave the compiled frame in. The session loop drives each rep's frame itself, on one schedule of requests
(`Flock.Firewall.Drive.reqs`), so its state is a Lean function. This file defines it from the decoded coins alone, at
placeholder draws, and the release whose transport state it is:

* `Loop.placeholder`: a rep stream's frame at zero draws, its table at a zero level 0 of the statement's shape, and zero
  salts for levels ≥ 1, none of them the tape's.
* `Loop.schedSoFar`, `Loop.ownRounds`: the rep's scheduled rounds so far (the algebraic phase's, then the opening's fed
  the coins in) and the inner proof's own rounds (`ownRound`).
* `Loop.repState`, `Loop.progOf`: `Frame.run` over the drive's requests (`Drive.reqs`, the deployed schedule) with the
  scheduled rounds themselves in the device's place, up to the last released round, of the rep that released it, at
  the coins the history opened.
* `Loop.freshOf`: a rep's state at its first round, after rep 0's algebraic-phase coins at rep 1 (`Frame.step`'s
  `prior`, as `Frames.roundPlace` hands them), or the loop's refusal there.
* `Loop.schedOf`: the round in a scheduled round's place, the loop's own on the rep's table and salts (`Frames.placeOf`).
* `LeanRelease.ofLoop`: `LeanRelease.loop` with that state and those rounds, each round's rep its rep stream
  (`Loop.repOf`) started as the loop starts it (`Drive.start`), each round's coin count its request's (`Loop.loopNr`).
* `ofLoop_progReads`: so `ProgReads` holds by definition, at `Loop.progOf` and `Loop.freshOf`, with no salt. What `ProgReads` took as given is
  now `ProgChecks` at `ofLoop`: that the deployed frame's state is this one's but for `stopped`, the frame reading a
  round's shape alone (`roundFree_alike`, `passes_frames`, `innerOwn_ownRound`).
-/

namespace Claims.ZKServer

open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker Draws INNER_ROUNDS drawBytes)
open _root_.Flock.Firewall.Finish (Tab tabOfLevel0 flat)

namespace Loop

/-- `n` zero bytes. -/
def zeros (n : ℕ) : ByteArray := ⟨Array.replicate n 0⟩

/-- **Rep stream `k`'s frame at placeholder draws**, at the setup's statements `sts` and the `Commit`'s coins `link`: the
frame's configuration at zero draws (`Drive.start`'s layout and mask slot), the table at a zero level 0 of the statement's
shape (`Level0.ofDraws` of a zero witness and zero draws), the stream's domain, the extra claims at the link points and
zero salts for levels ≥ 1 (`hiSaltsOf`'s sizes). -/
def placeholder (spec : SessionSpec) (sts : Array Stmt) (k : ℕ) (link : Array F128) :
    V (Cfg × Tab × ByteArray × Array (Array F128) × Array ByteArray) := do
  let st := sts[k / 2]!
  let (lay, mask) ← _root_.Flock.Firewall.Drive.start st
  let (sch, tPad) ← _root_.Flock.Zk.schedule st.m
  let lv := sch.levels.getD 0 default
  let l0 ← _root_.Flock.Firewall.Level0.ofDraws st.m (Array.replicate (2 ^ (lv.lanes + lv.cols)) 0)
    (zeros (_root_.Flock.Firewall.Level0.drawSize lv tPad))
  let t ← tabOfLevel0 (_root_.Flock.Setup.ofCircuit st) l0
  let hi := (Array.range (t.sch.levels.size - 1)).map fun j => zeros (192 * 2 ^ (t.sch.levels.getD (j + 1) default).d)
  return (Cfg.ofDraws lay (Draws.ofBytes lay (zeros (drawBytes lay))) (some mask), t,
    (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8,
    _root_.Flock.Firewall.Drive.extras st (_root_.Flock.Firewall.Drive.linkPoints st link), hi)

/-- **A rep's scheduled rounds so far** at the table `t`, its released rounds' coins `cr`: the algebraic phase's
(`Skeleton.rounds`), then, once their coins are all in, the opening's (`Drive.openStep`, as `Frames.placeOf` begins it)
fed the rest, with the round it asks for next. -/
def schedSoFar (t : Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (cr : Array (Array F128)) : V (Array (ByteArray × ℕ)) := do
  let alg ← _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom
  if cr.size < alg.size then return alg
  match (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi (cr.extract 0 alg.size)).feed
      (cr.extract alg.size cr.size).toList #[] with
  | (.ask c n _, rs) => return alg ++ rs.push (c, n)
  | (.done _, rs) => return alg ++ rs
  | (.fail e, _) => throw e

/-- **The inner proof's rounds**, the firewall's own (`ownRound`), at the configuration `c`: `ε` and `β` the first
words of the coins `ci` of its first and third, `final_c'` and `τ`'s commitment zero, round `j` squeezing `ns j`. -/
def ownRounds (c : Cfg) (ci : Array (Array F128)) (ns : ℕ → ℕ) : Array (ByteArray × ℕ) :=
  let s : Masker := { eps := (ci.getD 0 #[]).getD 0 0, beta := (ci.getD 2 #[]).getD 0 0 }
  (Array.range INNER_ROUNDS).map fun j => (ownRound c s (zeros 16) (zeros 64) (ns j) j, ns j)

/-- **A rep's frame state after its first `i` rounds**, at the configuration `c` and the table `t`, its rounds' coins
`cr` (one a released round, the last owed), rep 1's after rep 0's algebraic-phase coins (of `cr0`), its round `j`
squeezing `ns j`: `Frame.run` over the drive's requests (`Drive.reqs`) with the scheduled rounds themselves in the
device's place, an honest device's rounds but for the values each absorbs masked, up to its `i`-th round's. The rounds
up to it are scheduled on the coins before the last (`cr.pop`), as `Frames.placeOf` places each round on the coins of
the rounds before it, so the last coins, which the frame has not taken, are not read. `{}` once the rep has ended, where
the frame takes no coins (`Masker.answer` owes none at `{}`). -/
def repState (c : Cfg) (t : Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) (i : ℕ) : Masker :=
  match (do
      let alg ← _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom
      return (alg.size, ← schedSoFar t rep dom xs hi cr.pop) : V (ℕ × Array (ByteArray × ℕ))) with
  | .error _ => {}
  | .ok (nAlg, sched) =>
    let prior := if rep = 0 then #[] else (cr0.extract 0 nAlg).foldl (· ++ ·) #[]
    let own := ownRounds c (cr.extract sched.size cr.size) fun j => ns (sched.size + j)
    let pre := if prior.isEmpty then 0 else 1
    let qs := _root_.Flock.Firewall.Drive.reqs prior sched own (fun k => (sched.getD k (.empty, 0)).1) cr
    let r := _root_.Flock.Firewall.Frame.run c {} (qs.take (pre + 2 * i - 1))
    if r.2 then {} else r.1.m

/-- The words of opening `q` among the decoded coins `cs`, as many as its request's count (`readCoins`'s); none past
them. -/
def wordsAt (x : Msg) (cs : List FlockDeploy.Coin) (q : ℕ) : Array F128 :=
  match cs[q]? with
  | some c => c.vals (min (loopNr x q) _root_.Flock.Firewall.Frames.COIN_MAX)
  | none => #[]

/-- **The loop's frame state after the decoded coins `cs`** at `x`: the state of the rep that released the last of the
rounds they answer, rep round `p` the `p + 2` coins answer (`CoinLayer.locate` places it at rep stream `k`'s round `i`,
as `CoinLayer.slot` places them), after its `i + 1` rounds, owing the last's coins, at placeholder draws (`repState`),
its rounds' coins and rep 0's the openings', the `Commit`'s the first. `{}` before the first rep round, or where `x`
sets up no session. -/
noncomputable def progOf (x : Msg) (cs : List FlockDeploy.Coin) : Masker :=
  match FlockDeploy.setupOf true x, cs.length with
  | some (_, spec, sts), p + 2 =>
    let rs := roundsOf spec sts
    let ki := ZKMal.CoinLayer.locate rs 0 p
    let start := p - ki.2
    let n0 := rs.getD (ki.1 - 1) 0
    let crAt := fun (s n : ℕ) => (Array.range n).map fun j => wordsAt x cs (s + j + 1)
    match placeholder spec sts ki.1 (wordsAt x cs 0) with
    | .ok (c, t, dom, xs, hi) =>
      repState c t (ki.1 % 2) dom xs hi (crAt (start - n0) n0) (crAt start (ki.2 + 1))
        (fun j => loopNr x (start + j + 1)) (ki.2 + 1)
    | .error _ => {}
  | _, _ => {}

/-- **The loop's frame state at a rep's first round** after the decoded coins `cs` at `x`, rep round `q` the `q + 1`
coins precede being the first of its rep stream `k`: rep 0's is the frame's start, and rep 1's its start after rep 0's
algebraic-phase coins, as `Frames.roundPlace` hands them (`Frame.step`'s `prior`, at the placeholder's configuration,
whose layout alone it reads), or why the loop refuses the round there: no placeholder or algebraic phase (which
`Frames.withAlg` computes at every rep's first round), rep 0's algebraic phase not ended, or the frame's refusal of the
coins. The frame's start where `cs` is empty (the `Commit`, whose coins no frame takes); refused where `x` sets up no
session. -/
noncomputable def freshOf (x : Msg) (cs : List FlockDeploy.Coin) : Except String Masker :=
  match FlockDeploy.setupOf true x, cs.length with
  | some (_, spec, sts), q + 1 =>
    let rs := roundsOf spec sts
    let k := (ZKMal.CoinLayer.locate rs 0 q).1
    let n0 := rs.getD (k - 1) 0
    match placeholder spec sts k (wordsAt x cs 0) with
    | .error e => throw e
    | .ok (c, t, dom, _, _) =>
      match _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom with
      | .error e => throw e
      | .ok alg =>
        if k % 2 = 0 then pure {}
        else if n0 < alg.size then throw "rep 1's first round before rep 0's algebraic phase ended"
        else
          let cr0 := (Array.range n0).map fun j => wordsAt x cs (q - n0 + j + 1)
          let prior := (cr0.extract 0 alg.size).foldl (· ++ ·) #[]
          match _root_.Flock.Firewall.Frame.step c {}
              (.prior prior.size (_root_.Flock.Firewall.Frame.wordBytes prior)) with
          | (.ok, st') => pure st'.m
          | _ => throw "rep 0's algebraic-phase coins"
  | some _, 0 => pure {}
  | none, _ => throw "the loop's frames: no session"

/-- **The scheduled round in the device's place after the decoded coins `cs`** at `x`, on the rep's own table and salts
for levels ≥ 1 (`real k` for rep stream `k`, as `Frames.placeOf` computes them from the session's level 0 and its finish's
salts): rep stream `k`'s scheduled round `i` (`schedSoFar`), `k` and `i` as `progOf` locates them. The empty round where
the rep's table or its schedule is refused: the loop has no round there (`Frames.placeOf` refuses it, and the session
stops), and the frame's check refuses the empty round. None past the rep's schedule (the inner proof's rounds, whose round
the firewall sends itself), or where `x` sets up no session. -/
noncomputable def schedOf (real : ℕ → V (Tab × Array ByteArray)) (x : Msg) (cs : List FlockDeploy.Coin) :
    Option ByteArray :=
  match FlockDeploy.setupOf true x, cs.length with
  | some (_, spec, sts), q + 1 =>
    let ki := ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q
    let st := sts[ki.1 / 2]!
    let cr := (Array.range ki.2).map fun j => wordsAt x cs (q - ki.2 + j + 1)
    match real ki.1 with
    | .ok (t, hi) =>
      match schedSoFar t (ki.1 % 2) (spec.domain (spec.tables[ki.1 / 2]!) (ki.1 % 2)).toUTF8
          (_root_.Flock.Firewall.Drive.extras st (_root_.Flock.Firewall.Drive.linkPoints st (wordsAt x cs 0))) hi cr with
      | .ok sched => sched[ki.2]?.map (·.1)
      | .error _ => some .empty
    | .error _ => some .empty
  | _, _ => none

/-- Round `j`'s rep at `x`: rep round `j - 2`'s stream (`CoinLayer.locate`), past `Hello` and the link. -/
noncomputable def repOf (x : Msg) (j : ℕ) : ℕ :=
  match FlockDeploy.setupOf true x with
  | some (_, spec, sts) => (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (j - 2)).1
  | none => 0

/-- Rep stream `k`'s start at `x` (`Drive.start` at table `k / 2`'s statement): its layout and mask slot. -/
noncomputable def startOf (x : Msg) (k : ℕ) : V (_root_.Flock.Zk.PadLayout × _root_.Flock.Firewall.Frame.MaskSlot) :=
  match FlockDeploy.setupOf true x with
  | some (_, _, sts) => _root_.Flock.Firewall.Drive.start sts[k / 2]!
  | none => .error "no session"

end Loop

namespace LeanRelease

/-- `L₀` with the loop's own frames: its transport's states are `Loop.progOf` and `Loop.freshOf` of the decoded coins, each round's rep is
its rep stream, started as the loop starts it, and the round in a scheduled round's place is the loop's own
(`Loop.schedOf`), on each rep's table and salts for levels ≥ 1 at `x`, the witness and the tape (`real`); at the inner
proof's rounds, `L₀`'s. At the `Hello`'s round, which the release never schedules (`kindAt`: the server's own), the
empty round, which the frame's check refuses. Each round's coin count is the one the loop hands `Frames.round`, its
request's (`Loop.loopNr`): the round after `h` takes opening `h.length - 1`'s coins. -/
noncomputable def withFrames (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray)) :
    LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand) :=
  { L₀ with
    rep := Loop.repOf
    lay := fun x k => ((Loop.startOf x k).toOption.map (·.1)).getD default
    slot := fun x k => (Loop.startOf x k).toOption.map (·.2)
    prog := fun x h => Loop.progOf x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2)
    fresh := fun x h => Loop.freshOf x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2)
    scheduled := fun x w ρ h vals =>
      match h with
      | [] => []
      | _ :: ops =>
        match Loop.schedOf (real x w ρ) x (ops.map fun p => ZKMal.CoinLayer.rd p.2) with
        | some r => r.data.toList
        | none => L₀.scheduled x w ρ h vals
    coins := fun x n => Loop.loopNr x (n - 1) }

/-- **The release at the session loop's own frames**: `loop`'s, at `withFrames`. -/
noncomputable def ofLoop (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray)) :
    LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand) :=
  loop Loop.loopNr T (withFrames T L₀ real)

/-- **The loop's frame state reads the history only through the decoded coins**, at `Loop.progOf` and
`Loop.freshOf`: by definition, on every history past `Hello`, and with no salt. -/
theorem ofLoop_progReads (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) :
    ProgReads (ZKMal.CoinLayer.session Loop.loopNr T) (ofLoop T L₀ real) Loop.progOf Loop.freshOf x :=
  fun _ _ _ _ _ _ => ⟨rfl, rfl⟩

end LeanRelease

end Claims.ZKServer
