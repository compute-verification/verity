import Proofs.Flock.Claims.ZK.SessionLean
import Proofs.Flock.Claims.ZK.FrameOwn

/-!
# The release at the session loop's own frames (Stage B)

`LeanRelease.ProgReads` and `LeanRelease.ProgChecks` take the transport's frame state (`prog`) as given: the state Rust's
requests leave the compiled frame in. The session loop drives each rep's frame itself, on one schedule of requests
(`Flock.Firewall.Drive.reqs`), so its state is a Lean function. This file defines it from the decoded coins alone, at
placeholder draws, and the release whose transport state it is (`note:20261022T0001Z-report-zkfin-stageb`):

* `Loop.placeholder`: a rep stream's frame at zero draws, its table at a zero level 0 of the statement's shape, and zero
  salts for levels ≥ 1, none of them the tape's.
* `Loop.schedSoFar`, `Loop.ownRounds`: the rep's scheduled rounds so far (the algebraic phase's, then the opening's fed
  the coins in) and the inner proof's own rounds (`ownRound`).
* `Loop.repState`, `Loop.progOf`: `Frame.run` over the drive's requests at the scheduled rounds themselves
  (`Drive.reqsAt`), up to the round's rep's last released round, at the coins the history opened.
* `LeanRelease.ofLoop`: `LeanRelease.loop` with that state, each round's rep its rep stream (`Loop.repOf`) started as the
  loop starts it (`Drive.start`).
* `ofLoop_progReads`: so `ProgReads` holds by definition, at `Loop.progOf`, with no salt. What `ProgReads` took as given is
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
squeezing `ns j`: `Frame.run` over the drive's requests at the scheduled rounds themselves (`Drive.reqsAt`), up to its
`i`-th round's. -/
def repState (c : Cfg) (t : Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray)
    (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) (i : ℕ) : Masker :=
  match (do
      let alg ← _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (flat t.cm0.cap) dom
      return (alg.size, ← schedSoFar t rep dom xs hi cr) : V (ℕ × Array (ByteArray × ℕ))) with
  | .error _ => {}
  | .ok (nAlg, sched) =>
    let prior := if rep = 0 then #[] else (cr0.extract 0 nAlg).foldl (· ++ ·) #[]
    let own := ownRounds c (cr.extract sched.size cr.size) fun j => ns (sched.size + j)
    let pre := if prior.isEmpty then 0 else 1
    (_root_.Flock.Firewall.Frame.run c {} ((_root_.Flock.Firewall.Drive.reqsAt prior sched own cr).take (pre + 2 * i - 1))).1.m

/-- The words of opening `q` among the decoded coins `cs`, as many as its request's count (`readCoins`'s); none past
them. -/
def wordsAt (x : Msg) (cs : List FlockDeploy.Coin) (q : ℕ) : Array F128 :=
  match cs[q]? with
  | some c => c.vals (min (loopNr x q) _root_.Flock.Firewall.Frames.COIN_MAX)
  | none => #[]

/-- **The loop's frame state after the decoded coins `cs`** at `x`: the state of the rep of the round after them, rep
stream `k` with `i` of its rounds released (`CoinLayer.locate`, as `CoinLayer.slot` places them), at placeholder draws
(`repState`), its rounds' coins and rep 0's the openings', the `Commit`'s the first. `{}` before the first rep round, or
where `x` sets up no session. -/
noncomputable def progOf (x : Msg) (cs : List FlockDeploy.Coin) : Masker :=
  match FlockDeploy.setupOf true x, cs.length with
  | some (_, spec, sts), q + 1 =>
    let rs := roundsOf spec sts
    let ki := ZKMal.CoinLayer.locate rs 0 q
    let start := q - ki.2
    let n0 := rs.getD (ki.1 - 1) 0
    let crAt := fun (s n : ℕ) => (Array.range n).map fun j => wordsAt x cs (s + j + 1)
    match placeholder spec sts ki.1 (wordsAt x cs 0) with
    | .ok (c, t, dom, xs, hi) =>
      repState c t (ki.1 % 2) dom xs hi (crAt (start - n0) n0) (crAt start ki.2) (fun j => loopNr x (start + j + 1)) ki.2
    | .error _ => {}
  | _, _ => {}

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

/-- `L₀` with the loop's own frames: its transport's state is `Loop.progOf` of the decoded coins, and each round's rep is
its rep stream, started as the loop starts it. -/
noncomputable def withFrames (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)) :
    LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand) :=
  { L₀ with
    rep := Loop.repOf
    lay := fun x k => ((Loop.startOf x k).toOption.map (·.1)).getD default
    slot := fun x k => (Loop.startOf x k).toOption.map (·.2)
    prog := fun x h => Loop.progOf x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) }

/-- **The release at the session loop's own frames**: `loop`'s, at `withFrames`. -/
noncomputable def ofLoop (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)) :
    LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand) :=
  loop Loop.loopNr T (withFrames T L₀)

/-- **The loop's frame state reads the history only through the decoded coins**, at `Loop.progOf`: by definition, on
every history past `Hello`, and with no salt. -/
theorem ofLoop_progReads (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (x : Msg) :
    ProgReads (ZKMal.CoinLayer.session Loop.loopNr T) (ofLoop T L₀) Loop.progOf x :=
  fun _ _ _ _ _ _ => rfl

end LeanRelease

end Claims.ZKServer
