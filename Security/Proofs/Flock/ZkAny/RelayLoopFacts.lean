module

public import Proofs.Flock.ZkAny.RelayLoopKept
public import Proofs.Flock.ZkAny.RelayLoopCommit
public import Proofs.Flock.ZkAny.RelayLoopSalts
public import Proofs.Flock.ZkAny.RelayMark
public import Proofs.Flock.ZkAny.RelayParams
public import Proofs.Flock.ZkAny.RelaySession
public import Proofs.Flock.ZkAny.RelaySlot
public import Proofs.Flock.ZkAny.RelayPlace
public import Proofs.Flock.ZkAny.RelayFollow
public import Proofs.Flock.ZkAny.RelayLoopCur
public import Proofs.Flock.ZkAny.RelayRankSched
public import Proofs.Flock.ZkAny.RelayLoopL0
public import Proofs.Flock.ZkAny.RelayLoopFirst
public import Proofs.Flock.ZkAny.RelayRoundTook
public import Proofs.Flock.SessionLoop.Statement
public import Proofs.Flock.Claims.Complete.ZkTables.Count
public import Proofs.Flock.SessionLoop.Tables
import Proofs.Flock.Claims.ZK.Reference

/-!
# What the loop's frames and the model hold at a frame round

At a frame round's slot `slotCase_round` and `loopKept_round` read `RoundFacts`. Here it is from the relay's invariant
at the deployed loop (`LoopFacts`, `LoopJ`) and what the loop read for the round (`LoopRead`'s round clause), the
model's release being the loop's own frames':

* `LoopModel.Deployed`: the model's release is `LeanRelease.ofLoop` on the statement's setup, whose rep streams' names
  are distinct and two a table statement (`helloOf_some`), with a message past its last round.
* `RealAt`: the model's tables and salts (`real`) are the loop's. **`realAt_of`**: at the loop's own tables
  (`Loop.realOf src`), the loop handed `src`'s bytes (`M.src`), it holds where the table each of the session's level 0s
  gives is `src`'s and the salts the loop keeps are the ones it was handed (`SaltsSrc`, `LoopRead`'s round clause).
  **`realAt_slot`**: it holds at a frame round's slot, from `LoopJ` (`realAt_h0`) and `SaltsSrc`.
* `roundPlace_find`: the frame `roundPlace` readies a round on is one the loop's frames hold.
  `roundPlace_prior_size`: at rep 1's first round, rep 0's frame holds the rep's algebraic phase's coins.
* `PlaceFacts`: what the model holds at the place `roundPlace` readies a round at: its scheduled round, its masking
  state but for `stopped`, and where `v` fails the crossing. `placeSt_seen`: its masking state at a round of a stream
  seen, from the rep's `SameInv`. **`placeSched_later`**: its scheduled round past a stream's first round, from
  `StreamAt` and `RealAt`, or `placeOf` places the round at the inner proof's own. **`placeSched_seen`**: the same at
  a round of a stream seen, from `LoopJ`. **`placeFacts_seen`**: `PlaceFacts` at a round of a stream seen, given the
  inner proof's own place and the crossing. **`placeSched_first`**: `PlaceFacts.sched` at a stream's first round, from
  `LoopJ`, `RealAt` and the model's answer there. **`placeSt_first0`**: `PlaceFacts.st` at a rep 0 stream's first
  round, the fresh frame's start (`roundPlace_switch0`). **`placeSt_first1`**: the same at rep 1, the frame's run on
  rep 0's algebraic-phase words (`roundPlace_switch1`), rep 0's stream being seen (`roundPlace_prior_size`).
  **`placeFacts_slot`**: `PlaceFacts` at a frame round's slot, given the crossing and, at a stream seen, the inner
  proof's own place.
* `roundPlace_scheduled_later`: past a stream's first round, on a frame keeping its schedule so far on the loop's
  table, extras and salts, which are the model's, the round's place is the model's scheduled round, or the schedule
  has no round there (the inner proof's own).
* `slot_masked`: a frame round's slot is the model's masked round below the `Finish`.
* `rep_seen`: at a frame round's slot, the stream is seen exactly where the model's last round was the same rep's.
* **`answered_of_round`**: the model answers a frame round's slot's history where the loop took the coins so far
  (`LoopModel.TookAt`), given the next rep's start at a stream's first round and, at a stream seen, the round's place
  the model's scheduled round, the empty round or the opening's first wherever the rank stop fails
  (`rank_of_release`).
* **`rank_sched_deployed`**: at a round of a stream seen of the deployed loop on a `Live` statement, wherever the rank
  stop fails at the model's coin-step state, the model's scheduled round is the empty round or the opening's first
  (`LeanRelease.ofLoop_rank_sched`): `answered_of_round`'s `hal`.
* **`roundFacts_loop`**: `RoundFacts` at a frame round's slot of the deployed loop, given the model's answer there and
  `PlaceFacts` at the round's place.
* `LeanRelease.CrossPasses`: the named hypothesis on the Rust release that its checks (`cross`) fail no frame round
  below the `Finish` that the Lean frame passes. It becomes provable once the Rust crossing check is mirrored into the
  Lean ZK server code, which would fix `cross`. `OwnSched`: the inner proof's own place at a round of a stream seen is
  the model's scheduled round, at each slot. `FreshStart`: at a stream's first round that goes out, the model's
  next-rep state is a masking state, at each slot.
* **`answered_deployed`**: the model answers each frame round's slot's history of the deployed loop where the loop
  took the coins so far, given `CrossPasses`, `OwnSched` and `FreshStart`.
* **`roundFacts_deployed`**: `RoundFacts` at each frame round's slot of the deployed loop (`loopKept_round`'s `hA`),
  given `CrossPasses`, `OwnSched` and `FreshStart`, the model's answer there by `answered_deployed` at what the loop
  took (`LoopJ.took`).
-/

@[expose] public section

namespace Claims.ZKServer.LeanRelease

/-- **Rust's checks fail no frame round the Lean frame passes**: at every frame round below the `Finish` (masked, before
`last`), at every statement, witness, tape, history and worker's value, where Rust's own checks on the worker's rep
(`cross`: the worker failing, the device's level-0 root, the constraints on the pads, the end check on the returned
proof and Rust's masking cross-checks) fail the value, the Lean frame's pass decision (`pass`) fails it too. A named
hypothesis on the Rust release: at the deployed loop's release (`ofLoop`) `cross` is the release's own, which no Lean
code fixes, and `cross`'s contract (each check can only fail a round more) does not give it. It is stated at every
history, with no guard on the rounds before it: Rust stops a session at its first failing round, so a history with an
earlier failing round is never run, and the Lean model of the release is free there. The unguarded form constrains the
real Rust release no more than a guarded one. It becomes provable once the Rust crossing check is mirrored into the
Lean ZK server code, which would fix `cross`. -/
def CrossPasses {W R : Type} (L : LeanRelease W R) : Prop :=
  ∀ (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) (v : Msg), L.kindAt x h.length = .masked →
    h.length < L.last x → L.cross x h.length w ρ h v = false → L.pass x w ρ h v = false

end Claims.ZKServer.LeanRelease

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed planOf reqAt)
open _root_.Flock (V SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Masker Place)

/-- **The model's run is the deployed loop's**: its release is the loop's own frames' (`LeanRelease.ofLoop`) at the
tables `T`, the release `L₀` and the tables' samples `real`, the statement sets up as `(s0, spec, sts)`, the setup's rep
streams' names are distinct and two a table statement (what `helloOf_some` reads off a session's `Hello`), and the run
has a message past its last round. -/
structure LoopModel.Deployed {T : ZKMal.CoinLayer.TableData}
    (M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) : Prop where
  rel : M.L = LeanRelease.ofLoop T L₀ real
  setup : FlockDeploy.setupOf true M.x = some (s0, spec, sts)
  names : (spec.repStreams.toList.map (·.2.2)).Nodup
  sizes : spec.repStreams.size = 2 * sts.size
  next : M.L.last M.x < M.nN M.x

/-- **The frame `roundPlace` readies a round on is one the loop's frames hold.** -/
theorem roundPlace_find {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pc, f, nd)) :
    ∃ j p, R.find sid i r = some (j, p) := by
  rcases hf : R.find sid i r with _ | ⟨j, p⟩
  · unfold _root_.Flock.Firewall.Frames.roundPlace at h
    simp only [hf] at h
    cases h
  · exact ⟨j, p, rfl⟩

set_option maxHeartbeats 1000000 in
/-- **At rep 1's first round rep 0's frame holds the rep's algebraic phase's coins**: `roundPlace` refuses rep 1's
first round before rep 0's algebraic phase ended. -/
theorem roundPlace_prior_size {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i j j0 : Nat}
    {dom content : ByteArray} {n : Nat} {p p0 p' q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i 1 = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i 1 dom content n = .ok (q, pl, f, nd))
    (hfst : (p.coins.isEmpty && !p.st.begun) = true) {s : _root_.Flock.Firewall.Frames.Release} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} (hs : R.sessions.find? (·.sid == sid) = some s)
    (hn : s.spec.tables[i]? = some name) (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p') (h0 : R.find sid i 0 = some (j0, p0)) :
    p'.alg.size ≤ p0.coins.size := by
  rcases Nat.lt_or_ge p0.coins.size p'.alg.size with hlt | hge
  · exfalso
    have h11 : ((1 : Nat) == 1) = true := rfl
    unfold _root_.Flock.Firewall.Frames.roundPlace at h
    simp only [hf, bind, Except.bind, pure, Except.pure] at h
    repeat' (first
      | simp only [hs, hn, hl0, hfst, hw, h11, h0, hlt, ↓reduceIte] at h
      | split at h)
    all_goals first
      | (cases h; done)
      | exact absurd (by assumption) throw_ne_ok
  · exact hge

/-- **The model's tables and salts are the loop's**, at the loop `σ` for session `sid`: at each rep `r` of each table
`i` of the session's entry, the model's samples for rep stream `2 * i + r` (`real`) are the table the loop's level 0 for
the table gives, with the salts for levels ≥ 1 the loop holds for the rep wherever it holds some. -/
def RealAt {T : ZKMal.CoinLayer.TableData}
    (M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) (sid : Nat) (σ : LoopAt) : Prop :=
  ∀ (i r : Nat) (rel : _root_.Flock.Firewall.Frames.Release) (name : String)
    (l0 : _root_.Flock.Firewall.Level0.Level0) (stm : Stmt) (t : _root_.Flock.Firewall.Finish.Tab), r < 2 →
    sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
    _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 → rel.sts[i]? = some stm →
    _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t →
    ∃ hi, real M.x M.w M.ρ (2 * i + r) = .ok (t, hi) ∧ ∀ r0 q0 hi',
      σ.R.salts.find? (fun x => x.1 == l0.root && x.2.1 == r) = some (r0, q0, hi') → hi' = hi

/-- **The loop's own tables' salts are the source's**: where `Loop.realOf src` gives a rep stream a table and salts,
the salts are the ones `src` gives it. -/
theorem realOf_hi {R : Type}
    {src : Msg → Claims.Flock.Wit → R → Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {x : Msg} {w : Claims.Flock.Wit} {ρ : R} {k : Nat} {t : _root_.Flock.Firewall.Finish.Tab}
    {hi : Array ByteArray} (h : Loop.realOf src x w ρ k = .ok (t, hi)) : ∃ g d, src x w ρ k = .ok (g, d, hi) := by
  unfold Loop.realOf at h
  split at h
  · rcases hs : src x w ρ k with e | ⟨g, d, hi'⟩
    · rw [hs] at h
      cases h
    · refine ⟨g, d, ?_⟩
      rw [hs] at h
      simp only [bind, Except.bind] at h
      split at h
      · cases h
      · split at h
        · cases h
        · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
          rw [h.2]
  · cases h

/-- **`RealAt` at the loop's own tables** (`Loop.realOf src`), where the loop's host and operating system hand it
`src`'s bytes at the model's statement, witness and tapes (`M.src`): where each table of the session's entry that its
level 0 gives is the one `Loop.realOf src` gives the table's reps (the level-0 half), and the salts the loop keeps are
the ones it was handed (`SaltsSrc`), the model's tables and salts are the loop's (`realOf_hi`). -/
theorem realAt_of {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)} {sid : Nat} {σ : LoopAt}
    (hsrc : M.src = src M.x M.w M.ρ)
    (h0 : ∀ (i r : Nat) (rel : _root_.Flock.Firewall.Frames.Release) (name : String)
      (l0 : _root_.Flock.Firewall.Level0.Level0) (stm : Stmt) (t : _root_.Flock.Firewall.Finish.Tab), r < 2 →
      sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
      _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 → rel.sts[i]? = some stm →
      _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t →
      ∃ hi, Loop.realOf src M.x M.w M.ρ (2 * i + r) = .ok (t, hi))
    (hS : SaltsSrc M sid σ) : RealAt M (Loop.realOf src) sid σ := by
  intro i r rel name l0 stm t hr hss hn hl0 hst ht
  obtain ⟨hi, hre⟩ := h0 i r rel name l0 stm t hr hss hn hl0 hst ht
  refine ⟨hi, hre, fun r0 q0 hi' hf => ?_⟩
  obtain ⟨g, d, hs⟩ := hS i r rel name l0 r0 q0 hi' hr hss hn hl0 hf
  obtain ⟨g', d', hs'⟩ := realOf_hi hre
  rw [hsrc] at hs
  rw [hs] at hs'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hs'
  exact hs'.2.2

/-- **Past a stream's first round, the round's place is the model's scheduled round**, or the schedule so far has no
round there (the inner proof's own): at the loop's own frames, after the history `e0 :: h1` whose next round is table
`i`'s rep `r`'s `j + 2`-th, where the rep's frame keeps its schedule so far (`Loop.RepInv`) on the loop's table, the
session's extras at the model's link and the salts the loop holds where `placeOf` reads them, which are the model's
(`real`), and holds the history's words for the rep's rounds so far, `roundPlace`'s place is the schedule's at the
frame's next round (`Loop.placeOf_sched`, `roundPlace_later_of`), the model's scheduled round there
(`scheduled_bytes`). -/
theorem roundPlace_scheduled_later (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Claims.Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (vals : String → Option ByteArray)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q₀ j i r : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q₀ + 1)
    (hk : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = 2 * i + r) (hr2 : r < 2)
    (hj : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = j + 1)
    {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid : Nat} {dom content : ByteArray} {n : Nat}
    {p q : Rep} {pc : Place} {f : Bool} {nd : Option String} (hfa : frameAt R sid i r = some p)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pc, f, nd))
    {s : _root_.Flock.Firewall.Frames.Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) (hspec : s.spec = spec)
    (hsts : s.sts = sts) (hlink : s.link = some (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))
    {stm : Stmt} {t : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {r0 : ByteArray} {q0 : Nat}
    (hst : sts[i]? = some stm)
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hreal : real x w ρ (2 * i + r) = .ok (t, hi))
    (hI : Loop.RepInv t (spec.domain name r).toUTF8 (_root_.Flock.Firewall.Drive.extras stm
      (_root_.Flock.Firewall.Drive.linkPoints stm (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi p)
    (hco : p.coins = (Array.range (j + 1)).map fun k =>
      Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) (q₀ - (j + 1) + k + 1))
    (hsalt : p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
      R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    {sched : Array (ByteArray × Nat)}
    (hsch : Loop.schedSoFar t r (spec.domain name r).toUTF8 (_root_.Flock.Firewall.Drive.extras stm
      (_root_.Flock.Firewall.Drive.linkPoints stm (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
      p.coins = .ok sched) :
    pc = .sched ⟨((LeanRelease.ofLoop T L₀ real).scheduled x w ρ (e0 :: h1) vals).toArray⟩ ∨
      (Loop.placeAt sched (j + 1)).1 = .own := by
  obtain ⟨_, hfd⟩ := find_of_frameAt hfa
  obtain ⟨-, htb, hrr⟩ : p.sid = sid ∧ p.table = i ∧ p.rep = r := by
    have h := find_key hfd
    simp only [keyOf, Prod.mk.injEq] at h
    exact h
  have hsz : p.coins.size = j + 1 := by rw [hco, Array.size_map, Array.size_range]
  have hne : p.coins.isEmpty = false := by simp [Array.isEmpty, hsz]
  obtain ⟨hf, -, -, -⟩ := roundPlace_next hfa hne hrp
  obtain ⟨_, hv, -⟩ := Loop.placeOf_sched R.salts l0 s p (s.spec.domain name r).toUTF8
    (by rw [hsts, htb]; exact hst) hlink hsalt htab rfl (by rw [hspec]; exact hI) (by rw [hspec, hrr]; exact hsch)
  obtain ⟨-, hpc⟩ := roundPlace_later_of hfd hrp hf hss hn hl0 hv
  simp only [hsz] at hpc
  rcases hsk : sched[j + 1]? with _ | ⟨rr, _⟩
  · right
    simp only [Loop.placeAt, hsk]
  · left
    have hpa : (Loop.placeAt sched (j + 1)).1 = .sched rr := by simp only [Loop.placeAt, hsk]
    rw [hpa] at hpc
    rcases hpc with hpc | hpc
    · have h2r : (2 * i + r) % 2 = r := by omega
      have h2i : (2 * i + r) / 2 = i := by omega
      have htn : spec.tables[i]! = name := by
        have hn' : spec.tables[i]? = some name := by
          rw [← hspec]
          exact hn
        obtain ⟨hl, he⟩ := Array.getElem?_eq_some_iff.mp hn'
        rw [getElem!_pos spec.tables i hl, he]
      have hsti : sts[i]! = stm := by
        obtain ⟨hl, he⟩ := Array.getElem?_eq_some_iff.mp hst
        rw [getElem!_pos sts i hl, he]
      have hreal' : real x w ρ (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = .ok (t, hi) := by
        rw [hk]
        exact hreal
      have hb := scheduled_bytes T L₀ real x w ρ e0 h1 vals hx hq hreal'
        (by rw [hk, hj, h2r, h2i, htn, hsti, ← hco]; exact hsch) (by rw [hj]; exact hpa)
      rw [hpc, hb]
    · cases hpc

/-- **At a stream's first round, the round's place is the model's scheduled round**, or the schedule so far has no
round there: after the history `e0 :: h1` whose next round is table `i`'s rep `r`'s first, on the rep's fresh frame,
which `withAlg` gives its algebraic phase on the loop's level 0, whose root is its table's commitment's cap, at the
loop's table, the session's extras at the model's link and the salts the loop holds where `placeOf` reads them, which
are the model's (`real`), `roundPlace`'s place is the schedule's first (`roundPlace_first_of`, `repInv_withAlg`,
`Loop.placeOf_sched`), the model's scheduled round there (`scheduled_bytes`). -/
theorem roundPlace_scheduled_first (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Claims.Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (vals : String → Option ByteArray)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q₀ i r : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q₀ + 1)
    (hk : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = 2 * i + r) (hr2 : r < 2)
    (hj : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = 0)
    {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid : Nat} {dom content : ByteArray} {n : Nat}
    {p q : Rep} {pc : Place} {f : Bool} {nd : Option String} (hfa : frameAt R sid i r = some p) (hp : FreshRep p)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pc, f, nd))
    {s : _root_.Flock.Firewall.Frames.Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) (hspec : s.spec = spec)
    (hsts : s.sts = sts) (hlink : s.link = some (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))
    {p' : Rep} (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p')
    {stm : Stmt} {t : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {r0 : ByteArray} {q0 : Nat}
    (hst : sts[i]? = some stm)
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hcap : _root_.Flock.Firewall.Finish.flat t.cm0.cap = l0.root)
    (hreal : real x w ρ (2 * i + r) = .ok (t, hi))
    (hsalt : p'.alg.size = 0 → R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    {sched : Array (ByteArray × Nat)}
    (hsch : Loop.schedSoFar t r (spec.domain name r).toUTF8 (_root_.Flock.Firewall.Drive.extras stm
      (_root_.Flock.Firewall.Drive.linkPoints stm (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
      #[] = .ok sched) :
    pc = .sched ⟨((LeanRelease.ofLoop T L₀ real).scheduled x w ρ (e0 :: h1) vals).toArray⟩ ∨
      (Loop.placeAt sched 0).1 = .own := by
  obtain ⟨_, hfd⟩ := find_of_frameAt hfa
  obtain ⟨-, htb, hrr⟩ : p.sid = sid ∧ p.table = i ∧ p.rep = r := by
    have h := find_key hfd
    simp only [keyOf, Prod.mk.injEq] at h
    exact h
  have hfst : (p.coins.isEmpty && !p.st.begun) = true := by
    rw [hp.coins, hp.st]
    rfl
  obtain ⟨p'', pl₀, _, hpo, hpc, hp''⟩ := roundPlace_first_of hfd hrp hfst hss hn hl0 hw
  obtain ⟨htb', hrep', hco', -, -, -⟩ := withAlg_keep hw
  have hst' : s.sts[p.table]? = some stm := by
    rw [hsts, htb]
    exact hst
  have hI' := repInv_withAlg hp hw hst' htab hcap (_root_.Flock.Firewall.Drive.extras stm
    (_root_.Flock.Firewall.Drive.linkPoints stm (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
  obtain ⟨htb'', hrep'', hco'', hal'', hI''⟩ : p''.table = p.table ∧ p''.rep = p.rep ∧ p''.coins = #[] ∧
      p''.alg = p'.alg ∧ Loop.RepInv t (s.spec.domain name p.rep).toUTF8 (_root_.Flock.Firewall.Drive.extras stm
        (_root_.Flock.Firewall.Drive.linkPoints stm (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
        p'' := by
    rcases hp'' with rfl | ⟨st₁, rfl⟩
    · exact ⟨htb', hrep', hco'.trans hp.coins, rfl, hI'⟩
    · exact ⟨htb', hrep', hco'.trans hp.coins, rfl, ⟨hI'.alg, hI'.toOpening, hI'.opening⟩⟩
  rw [hrr] at hI''
  have hal0 : p''.alg.size ≤ p''.coins.size → p'.alg.size = 0 := by
    intro hle
    rw [hco''] at hle
    rw [← hal'']
    simpa using hle
  obtain ⟨_, hv, -⟩ := Loop.placeOf_sched R.salts l0 s p'' (s.spec.domain name r).toUTF8
    (by rw [htb'', htb, hsts]; exact hst) hlink (fun hle _ _ => by rw [hrep'']; exact hsalt (hal0 hle))
    htab rfl hI'' (by rw [hco'', hrep'', hrr, hspec]; exact hsch)
  have hpl₀ : pl₀ = (Loop.placeAt sched 0).1 := by
    have e := hpo.symm.trans hv
    rw [hco''] at e
    simp only [Except.ok.injEq, Prod.mk.injEq] at e
    exact e.1
  rw [hpl₀] at hpc
  rcases hsk : sched[0]? with _ | ⟨rr, _⟩
  · right
    simp only [Loop.placeAt, hsk]
  · left
    have hpa : (Loop.placeAt sched 0).1 = .sched rr := by simp only [Loop.placeAt, hsk]
    rw [hpa] at hpc
    rcases hpc with hpc | hpc
    · have h2r : (2 * i + r) % 2 = r := by omega
      have h2i : (2 * i + r) / 2 = i := by omega
      have htn : spec.tables[i]! = name := by
        have hn' : spec.tables[i]? = some name := by
          rw [← hspec]
          exact hn
        obtain ⟨hl, he⟩ := Array.getElem?_eq_some_iff.mp hn'
        rw [getElem!_pos spec.tables i hl, he]
      have hsti : sts[i]! = stm := by
        obtain ⟨hl, he⟩ := Array.getElem?_eq_some_iff.mp hst
        rw [getElem!_pos sts i hl, he]
      have hreal' : real x w ρ (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = .ok (t, hi) := by
        rw [hk]
        exact hreal
      have hnil : ((Array.range 0).map fun k =>
          Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) (q₀ - 0 + k + 1)) = #[] :=
        Array.eq_empty_of_size_eq_zero (by rw [Array.size_map, Array.size_range])
      have hb := scheduled_bytes T L₀ real x w ρ e0 h1 vals hx hq hreal'
        (by rw [hk, hj, h2r, h2i, htn, hsti, hnil]; exact hsch) (by rw [hj]; exact hpa)
      rw [hpc, hb]
    · cases hpc

/-- **What the model holds at the place a frame round is readied at**, for the frame `q` and the place `pc` and
no-device reason `nd` `roundPlace` returns, the worker's value `v` at the model's history so far, the loop at `σ`: the
place is the model's scheduled round there, `q`'s masking state the model's before the rank stop (`answeredR`) but for
`stopped`, where the round takes a device round exactly where the frames haven't stopped session `sid`, `v` fails the
crossing only where it fails to pass. -/
structure PlaceFacts {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (v : Claims.Msg) (σ : LoopAt) (q : Rep)
    (pc : Place) (nd : Option String) : Prop where
  sched : pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
    (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩
  st : ∀ m, M.L.answeredR M.x M.ρ (M.hist σ.c) = .ok m → unstop q.st.m = unstop m
  cross : nd.isSome = σ.R.stopped sid → M.L.cross M.x (M.hist σ.c).length M.w M.ρ (M.hist σ.c) v = false →
    M.L.pass M.x M.w M.ρ (M.hist σ.c) v = false

/-- **`PlaceFacts.st` at a round of a stream seen**: at the slot of a round of `stream`, table `i`'s rep `r`, where
`LoopFacts` and `LoopJ` hold and `seen` has a round of `stream`, the frame `roundPlace` readies is at the model's
masking state but for `stopped`, by the rep's `SameInv` (`round_rep`, `sameInv_place`). -/
theorem placeSt_seen {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) (hseen : seen.contains stream = true)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd)) :
    ∀ m, M.L.answeredR M.x M.ρ (M.hist σ.c) = .ok m → unstop q.st.m = unstop m := by
  obtain ⟨hk, hmem⟩ := round_rep hsl hF hJ hrep hlt hnd
  obtain ⟨p, rel, name, l0, hfa, -, -, -, hA⟩ := hJ.reps i r stream hmem hseen
  obtain ⟨j, c, stm, t, xs, hi, pre, ev, -, -, -, -, -, -, hI, -⟩ := hA.1 hk.symm
  exact (sameInv_place hI hfa hrp).2.2.2.1

/-- **`PlaceFacts.sched` past a stream's first round**: at table `i`'s rep `r`'s frame `p`, on the session's entry
`rel`, its table `name` and level 0 `l0`, keeping `StreamAt` at the model's history `e0 :: h1` whose next round is the
rep's `j + 2`-th, where the entry is the model's setup at the model's link and the model's tables and salts are the
loop's (`RealAt`), the round's place is the model's scheduled round, or `placeOf` places it at the inner proof's own.
The frame keeps its schedule so far at `real`'s salts: before its opening begins only the opening's first round reads
them, where `placeOf` found the loop's (`roundPlace_salts`, `repInv_hi`); once it began they are the loop's
(`StreamAt`). `roundPlace_scheduled_later` places the round on that schedule (`sched_of_placeOf`). -/
theorem placeSched_later {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {σ : LoopAt} {i r j : Nat} (hr2 : r < 2) (hR : RealAt M real sid σ)
    {e0 : Claims.Msg × Claims.Msg} {h1 : List (Claims.Msg × Claims.Msg)} {q₀ : Nat}
    (hh : M.hist σ.c = e0 :: h1) (hq : h1.length = q₀ + 1)
    (hk : nextAt spec sts (M.hist σ.c) = (2 * i + r, j + 1))
    {p : Rep} {rel : _root_.Flock.Firewall.Frames.Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hfa : frameAt σ.R sid i r = some p) (hss : sessAt σ.R sid = some rel) (hn : rel.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0)
    (hA : StreamAt M spec sts (2 * i + r) (M.hist σ.c) rel name l0 p σ.R)
    (hspec : rel.spec = spec) (hsts : rel.sts = sts)
    (hlink : rel.link = some (Loop.wordsAt M.x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))
    {dom content : ByteArray} {n : Nat} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd)) :
    pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
        (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩ ∨
      ∃ n' q', _root_.Flock.Firewall.Frames.placeOf σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 =
        .ok (.own, n', q') := by
  have hss' : σ.R.sessions.find? (·.sid == sid) = some rel := hss
  have hlen : (M.hist σ.c).length - 2 = q₀ := by
    rw [hh, List.length_cons, hq]
    omega
  have hk1 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = 2 * i + r := by
    have := congrArg Prod.fst hk
    unfold nextAt at this
    rwa [hlen] at this
  have hk2 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = j + 1 := by
    have := congrArg Prod.snd hk
    unfold nextAt at this
    rwa [hlen] at this
  obtain ⟨j', c, stm, t, xs, hi, pre, ev, hj, hlc, hstm, htab, -, hxs, hI, hsa, hord⟩ :=
    hA.1 (by rw [hlen]; exact hk1)
  rw [hlen, hk2] at hj
  obtain rfl : j = j' := by omega
  rw [hlen, hh, List.tail_cons] at hI
  have hmod : (2 * i + r) % 2 = r := by omega
  have hdiv : (2 * i + r) / 2 = i := by omega
  rw [hmod] at hI hsa
  rw [hdiv] at hstm
  rw [hlink, Option.some.injEq] at hlc
  subst hlc hxs
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  have hpt : p.table = i := congrArg (fun x => x.2.1) (find_key hf)
  have hpr : p.rep = r := congrArg (fun x => x.2.2) (find_key hf)
  have hsz : p.coins.size = j + 1 := by rw [hI.coins, Array.size_map, Array.size_range]
  have hne' : p.coins.isEmpty = false := by simp [Array.isEmpty, hsz]
  obtain ⟨hiR, hreal, hRs⟩ := hR i r rel name l0 stm t hr2 hss hn hl0 hstm htab
  obtain ⟨r0, q0, hIR, hsalt⟩ : ∃ r0 q0, Loop.RepInv t (rel.spec.domain name r).toUTF8
      (_root_.Flock.Firewall.Drive.extras stm (_root_.Flock.Firewall.Drive.linkPoints stm
        (Loop.wordsAt M.x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hiR p ∧
      (p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
        σ.R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hiR)) := by
    by_cases hc : p.opening = none ∧ p.opened = false
    · obtain ⟨r0, q0, hiF, hs⟩ := roundPlace_salts hfa hne' hrp hss' hn hl0
      refine ⟨r0, q0, repInv_hi hI.inv hc.1 hc.2, fun hle ho hop => ?_⟩
      have hf' := hs hle ho hop
      rw [hpr] at hf' ⊢
      rw [hf', hRs r0 q0 hiF hf']
    · have hle : p.alg.size ≤ p.coins.size := by
        rcases hI.inv.opening with h1' | ⟨h2, -⟩ | ⟨h3, -⟩
        · exact absurd ⟨h1'.1, h1'.2.1⟩ hc
        · exact h2
        · exact h3
      have hlt : p.alg.size < p.coins.size := by
        rcases Nat.lt_or_eq_of_le hle with h | h
        · exact h
        · exact absurd (hord h) hc
      obtain ⟨r0, q0, hf'⟩ := find?_of_saltsAt (hsa hlt)
      have hhi := hRs r0 q0 hi hf'
      subst hhi
      exact ⟨.empty, 0, hI.inv, fun _ ho hop => absurd ⟨hop, ho⟩ hc⟩
  have hst' : rel.sts[p.table]? = some stm := by
    rw [hpt]
    exact hstm
  obtain ⟨w, hw⟩ := roundPlace_placeOf hf hrp (by rw [hne', Bool.false_and]) hss' hn hl0
  obtain ⟨sched, hsc⟩ := sched_of_placeOf hst' hlink hsalt htab rfl hIR hw
  have hsc' := hsc
  rw [hpr, hspec] at hsc'
  rcases roundPlace_scheduled_later T L₀ real M.x M.w M.ρ e0 h1
      (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c))) hD.setup hq hk1 hr2 hk2 hfa hrp hss' hn hl0 hspec hsts
      hlink (by rw [← hsts]; exact hstm) htab hreal (by rw [← hspec]; exact hIR) hI.coins hsalt hsc' with h | h
  · left
    rw [h, hD.rel, hh]
  · right
    obtain ⟨q', hq', -⟩ := Loop.placeOf_sched σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 hst' hlink hsalt
      htab rfl hIR hsc
    rw [hsz, h] at hq'
    exact ⟨_, q', hq'⟩

/-- **`PlaceFacts.sched` at a round of a stream seen**: at the slot of a round of `stream`, table `i`'s rep `r`, where
`LoopFacts` and `LoopJ` hold, `seen` has a round of `stream` and the model's tables and salts are the loop's (`RealAt`),
the round's place is the model's scheduled round, or `placeOf` places it at the inner proof's own on the rep's frame,
the session's entry, the table's name and its level 0 (`placeSched_later`): `RepJ` gives the rep's `StreamAt`,
`SessOn` the entry's setup, `LinkOn` the model's link on it past the `Commit` (`LoopJ.link`, at the rep's level 0),
and `SeenAt` the round past the stream's first. -/
theorem placeSched_seen {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hseen : seen.contains stream = true) (hR : RealAt M real sid σ)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd)) :
    pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
        (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩ ∨
      ∃ (p : Rep) (rel : _root_.Flock.Firewall.Frames.Release) (name : String)
        (l0 : _root_.Flock.Firewall.Level0.Level0),
        frameAt σ.R sid i r = some p ∧ sessAt σ.R sid = some rel ∧ rel.spec.tables[i]? = some name ∧
        _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 ∧
        ∃ n' q', _root_.Flock.Firewall.Frames.placeOf σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 =
          .ok (.own, n', q') := by
  obtain ⟨hsent, hop, -, -⟩ := slot_round hsl
  obtain ⟨hk0, hmem⟩ := round_rep hsl hF hJ hrep hlt hD.names
  have hlen : (M.hist σ.c).length = opened st + 1 := by
    rw [M.length_hist (Nat.le_of_lt hlt), hF.count hsent]
  obtain ⟨e0, h1, hh⟩ : ∃ e0 h1, M.hist σ.c = e0 :: h1 := by
    cases hc : M.hist σ.c with
    | nil =>
      rw [hc, List.length_nil] at hlen
      omega
    | cons e0 h1 => exact ⟨e0, h1, rfl⟩
  have hq : h1.length = opened st - 1 + 1 := by
    rw [hh, List.length_cons] at hlen
    omega
  have h2 : 0 < (nextAt spec sts (M.hist σ.c)).2 := by
    rcases (hJ.seenAt i r stream hmem).1 hseen with h | ⟨-, h⟩
    · omega
    · exact h
  have hk : nextAt spec sts (M.hist σ.c) = (2 * i + r, (nextAt spec sts (M.hist σ.c)).2 - 1 + 1) := by
    refine Prod.ext hk0.symm ?_
    show (nextAt spec sts (M.hist σ.c)).2 = (nextAt spec sts (M.hist σ.c)).2 - 1 + 1
    omega
  obtain ⟨p, rel, name, l0, hfa, hss, hn, hl0, hA⟩ := hJ.reps i r stream hmem hseen
  obtain ⟨hspec, hsts⟩ := (hJ.sess hsent).eq hss
  have hc2 : 2 ≤ σ.c := by
    rw [hF.count hsent]
    omega
  have hl := hJ.link hc2 name l0 hl0 rel hss
  rw [hh, List.tail_cons] at hl
  rcases placeSched_later hD (rep_lt_two hmem) hR hh hq hk hfa hss hn hl0 hA hspec hsts hl hrp with h | h
  · exact .inl h
  · exact .inr ⟨p, rel, name, l0, hfa, hss, hn, hl0, h⟩

/-- **`PlaceFacts.sched` at a stream's first round**: at the slot of a round of `stream`, table `i`'s rep `r`, where
`LoopFacts` and `LoopJ` hold, `seen` has no round of `stream`, the model's tables and salts are the loop's (`RealAt`)
and the model answers the round, the round's place is the model's scheduled round (`roundPlace_scheduled_first`). The
rep's frame is fresh (`FreshAt`); `roundPlace` read the session's entry, the table's statement, name and level 0 and
`withAlg`'s frame (`roundPlace_first_ok`); the level 0 is the one the loop drew (`BuiltAt`), so it gives a table as the
model's placeholder for the round does (`tab_of_placeholder`), whose commitment's cap is its root (`cap_of_drawn`); and
the algebraic phase has a round (`Loop.rounds_pos`), so the round reads no salts and is not the inner proof's own. -/
theorem placeSched_first {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hsn : seen.contains stream = false) (hR : RealAt M real sid σ) {m : Masker}
    (hA : M.L.answered M.x M.ρ (M.hist σ.c) = .ok m)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd)) :
    pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
      (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩ := by
  obtain ⟨hsent, hop, -, -⟩ := slot_round hsl
  obtain ⟨hk0, hmem⟩ := round_rep hsl hF hJ hrep hlt hD.names
  have hlen : (M.hist σ.c).length = opened st + 1 := by
    rw [M.length_hist (Nat.le_of_lt hlt), hF.count hsent]
  obtain ⟨e0, h1, hh⟩ : ∃ e0 h1, M.hist σ.c = e0 :: h1 := by
    cases hc : M.hist σ.c with
    | nil =>
      rw [hc, List.length_nil] at hlen
      omega
    | cons e0 h1 => exact ⟨e0, h1, rfl⟩
  obtain ⟨q₀, hq⟩ : ∃ q₀, h1.length = q₀ + 1 := by
    refine ⟨opened st - 1, ?_⟩
    rw [hh, List.length_cons] at hlen
    omega
  have hl2 : (M.hist σ.c).length - 2 = q₀ := by
    rw [hh, List.length_cons, hq]
    omega
  have h20 : (nextAt spec sts (M.hist σ.c)).2 = 0 := by
    rcases Nat.eq_zero_or_pos (nextAt spec sts (M.hist σ.c)).2 with h | h
    · exact h
    · exact absurd (hsn.symm.trans ((hJ.seenAt i r stream hmem).2 hk0 h)) Bool.false_ne_true
  have hk1 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = 2 * i + r := by
    have := hk0.symm
    unfold nextAt at this
    rwa [hl2] at this
  have hk2 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = 0 := by
    have := h20
    unfold nextAt at this
    rwa [hl2] at this
  have hr2 := rep_lt_two hmem
  have hdiv : (2 * i + r) / 2 = i := by omega
  obtain ⟨_, p, hfd⟩ := roundPlace_find hrp
  have hfa : frameAt σ.R sid i r = some p := by
    rw [← find_frameAt, hfd]
    rfl
  have hp : FreshRep p := fresh_found (hF.fresh i r stream hmem hsn) hfd
  have hpt : p.table = i := congrArg (fun x => x.2.1) (find_key hfd)
  have hpr : p.rep = r := congrArg (fun x => x.2.2) (find_key hfd)
  have hfst : (p.coins.isEmpty && !p.st.begun) = true := by
    rw [hp.coins, hp.st]
    rfl
  obtain ⟨rel, name, l0, stm, p', hss, hstm', hn, hl0, hw⟩ := roundPlace_first_ok hfd hrp hfst
  have hsa : sessAt σ.R sid = some rel := hss
  obtain ⟨hspec, hsts⟩ := (hJ.sess hsent).eq hsa
  have hc2 : 2 ≤ σ.c := by
    rw [hF.count hsent]
    omega
  have hl := hJ.link hc2 name l0 hl0 rel hsa
  rw [hh, List.tail_cons] at hl
  have hA₁ : (LeanRelease.ofLoop T L₀ real).answered M.x M.ρ (e0 :: h1) = .ok m := by
    rw [← hD.rel, ← hh]
    exact hA
  have hsw : 2 ≤ h1.length → Loop.repOf M.x (h1.length + 1) ≠ Loop.repOf M.x h1.length := by
    intro h2 he
    have e1 : Loop.repOf M.x (h1.length + 1) = (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 := by
      rw [show h1.length + 1 = q₀ + 2 by omega]
      exact LeanRelease.repOf_add_two hD.setup q₀
    have e2 : Loop.repOf M.x h1.length = (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (q₀ - 1)).1 := by
      rw [show h1.length = q₀ - 1 + 2 by omega]
      exact LeanRelease.repOf_add_two hD.setup (q₀ - 1)
    rw [e1, e2] at he
    have h' := Loop.locate_succ (Loop.roundsOf spec sts) 0 (q₀ - 1)
      (by rw [show q₀ - 1 + 1 = q₀ by omega]; exact he)
    rw [show q₀ - 1 + 1 = q₀ by omega, hk2] at h'
    omega
  have hcs : (h1.map fun e => ZKMal.CoinLayer.rd e.2).length = q₀ + 1 := by rw [List.length_map, hq]
  obtain ⟨c', t', dom', xs', hip, -, hph, -, -⟩ :=
    Loop.freshOf_ok M.x _ hD.setup hcs (ofLoop_answered_switch T L₀ real M.x M.ρ e0 h1 hsw hA₁)
  rw [hk1] at hph
  have hstmb : sts[i]! = stm := by
    obtain ⟨hli, he⟩ := Array.getElem?_eq_some_iff.mp hstm'
    rw [← hsts, getElem!_pos rel.sts i hli, he]
  obtain ⟨jb, gb, db, _, hjb, -, hdb⟩ := hJ.built hc2 name l0 hl0
  rw [tables_inj (j := i) hD.names hjb (by rw [← hspec]; exact hn), hstmb] at hdb
  obtain ⟨t, htab⟩ : ∃ t, _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t := by
    have h := tab_of_placeholder (g := gb) (d := db) (l := l0) hph (by rw [hdiv, hstmb]; exact hdb)
    rwa [hdiv, hstmb] at h
  have hcap := cap_of_drawn hdb htab
  obtain ⟨hi, hreal, -⟩ := hR i r rel name l0 stm t hr2 hsa hn hl0 hstm' htab
  have hst' : rel.sts[p.table]? = some stm := by
    rw [hpt]
    exact hstm'
  have hI' := repInv_withAlg hp hw hst' htab hcap (_root_.Flock.Firewall.Drive.extras stm
    (_root_.Flock.Firewall.Drive.linkPoints stm (Loop.wordsAt M.x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
  have hpos : 0 < p'.alg.size := Loop.rounds_pos hI'.alg
  obtain ⟨-, hrw, hcw, -, -, -⟩ := withAlg_keep hw
  have hsch := Loop.schedSoFar_alg t _ (_root_.Flock.Firewall.Drive.extras stm
    (_root_.Flock.Firewall.Drive.linkPoints stm (Loop.wordsAt M.x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi p'
    hI'.alg (by rw [hcw, hp.coins, Array.size_empty]; exact hpos)
  rw [hrw, hcw, hpr, hp.coins, hspec] at hsch
  rcases roundPlace_scheduled_first T L₀ real M.x M.w M.ρ e0 h1
      (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c))) hD.setup hq hk1 hr2 hk2 hfa hp hrp hss hn hl0 hspec hsts
      hl hw (by rw [← hsts]; exact hstm') htab hcap hreal (r0 := ByteArray.empty) (q0 := 0)
      (fun h0 => absurd h0 (by omega)) hsch with h | h
  · rw [h, hD.rel, hh]
  · obtain ⟨⟨c0, n0⟩, hc0⟩ : ∃ e, p'.alg[0]? = some e := ⟨_, Array.getElem?_eq_some_iff.mpr ⟨hpos, rfl⟩⟩
    have hpa : (Loop.placeAt p'.alg 0).1 = .sched c0 := by simp only [Loop.placeAt, hc0]
    rw [hpa] at h
    cases h

/-- **`PlaceFacts.st` at a rep 0 stream's first round**: at the slot of a round of `stream`, table `i`'s rep 0, where
`LoopFacts` and `LoopJ` hold and `seen` has no round of `stream`, the frame `roundPlace` readies is at the model's
masking state: the rep's frame is fresh (`FreshAt`), its run on no requests, and the model's state there is the next
rep's start (`ofLoop_answeredR_switch`), the frame's start at rep 0 (`roundPlace_switch0`). -/
theorem placeSt_first0 {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, 0)) (hlt : σ.c < M.τ.length)
    (hsn : seen.contains stream = false)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i 0 dom content n = .ok (q, pc, f, nd)) :
    ∀ m, M.L.answeredR M.x M.ρ (M.hist σ.c) = .ok m → unstop q.st.m = unstop m := by
  intro m hA
  obtain ⟨hsent, hop, -, -⟩ := slot_round hsl
  obtain ⟨hk0, hmem⟩ := round_rep hsl hF hJ hrep hlt hD.names
  have hlen : (M.hist σ.c).length = opened st + 1 := by
    rw [M.length_hist (Nat.le_of_lt hlt), hF.count hsent]
  obtain ⟨e0, h1, hh⟩ : ∃ e0 h1, M.hist σ.c = e0 :: h1 := by
    cases hc : M.hist σ.c with
    | nil =>
      rw [hc, List.length_nil] at hlen
      omega
    | cons e0 h1 => exact ⟨e0, h1, rfl⟩
  obtain ⟨q₀, hq⟩ : ∃ q₀, h1.length = q₀ + 1 := by
    refine ⟨opened st - 1, ?_⟩
    rw [hh, List.length_cons] at hlen
    omega
  have hl2 : (M.hist σ.c).length - 2 = q₀ := by
    rw [hh, List.length_cons, hq]
    omega
  have h20 : (nextAt spec sts (M.hist σ.c)).2 = 0 := by
    rcases Nat.eq_zero_or_pos (nextAt spec sts (M.hist σ.c)).2 with h | h
    · exact h
    · exact absurd (hsn.symm.trans ((hJ.seenAt i 0 stream hmem).2 hk0 h)) Bool.false_ne_true
  have hk1 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = 2 * i + 0 := by
    have := hk0.symm
    unfold nextAt at this
    rwa [hl2] at this
  have hk2 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = 0 := by
    have := h20
    unfold nextAt at this
    rwa [hl2] at this
  obtain ⟨_, p, hfd⟩ := roundPlace_find hrp
  have hp : FreshRep p := fresh_found (hF.fresh i 0 stream hmem hsn) hfd
  have hA₁ : (LeanRelease.ofLoop T L₀ real).answeredR M.x M.ρ (e0 :: h1) = .ok m := by
    rw [← hD.rel, ← hh]
    exact hA
  have hsw : 2 ≤ h1.length → Loop.repOf M.x (h1.length + 1) ≠ Loop.repOf M.x h1.length := by
    intro h2 he
    have e1 : Loop.repOf M.x (h1.length + 1) = (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 := by
      rw [show h1.length + 1 = q₀ + 2 by omega]
      exact LeanRelease.repOf_add_two hD.setup q₀
    have e2 : Loop.repOf M.x h1.length = (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (q₀ - 1)).1 := by
      rw [show h1.length = q₀ - 1 + 2 by omega]
      exact LeanRelease.repOf_add_two hD.setup (q₀ - 1)
    rw [e1, e2] at he
    have h' := Loop.locate_succ (Loop.roundsOf spec sts) 0 (q₀ - 1)
      (by rw [show q₀ - 1 + 1 = q₀ by omega]; exact he)
    rw [show q₀ - 1 + 1 = q₀ by omega, hk2] at h'
    omega
  have hcs : (h1.map fun e => ZKMal.CoinLayer.rd e.2).length = q₀ + 1 := by rw [List.length_map, hq]
  rw [roundPlace_switch0 hfd hrp (fresh_repRun hp) M.x _ hD.setup hcs (by rw [hk1]; omega)
    (ofLoop_answeredR_switch T L₀ real M.x M.ρ e0 h1 hsw hA₁)]

/-- **`PlaceFacts.st` at a rep 1 stream's first round**: at the slot of a round of `stream`, table `i`'s rep 1, where
`LoopFacts` and `LoopJ` hold, `seen` has no round of `stream` and the rep's frame is at the model's configuration, the
frame `roundPlace` readies is at the model's masking state: its run on rep 0's algebraic-phase words
(`roundPlace_switch1`). The algebraic phase has as many scheduled rounds as the placeholder's (`withAlg_size`), at
least one, so rep 0's frame, which holds as many coins (`roundPlace_prior_size`), is not fresh, and rep 0's stream is
seen (`LoopFacts.fresh`): its `RepJ` gives its coins as the history's words for its rounds (`StreamAt`). -/
theorem placeSt_first1 {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, 1)) (hlt : σ.c < M.τ.length)
    (hsn : seen.contains stream = false)
    (hpc : ∀ p, frameAt σ.R sid i 1 = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c))
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i 1 dom content n = .ok (q, pc, f, nd)) :
    ∀ m, M.L.answeredR M.x M.ρ (M.hist σ.c) = .ok m → unstop q.st.m = unstop m := by
  intro m hA
  obtain ⟨hsent, hop, -, -⟩ := slot_round hsl
  obtain ⟨hk0, hmem⟩ := round_rep hsl hF hJ hrep hlt hD.names
  have hlen : (M.hist σ.c).length = opened st + 1 := by
    rw [M.length_hist (Nat.le_of_lt hlt), hF.count hsent]
  obtain ⟨e0, h1, hh⟩ : ∃ e0 h1, M.hist σ.c = e0 :: h1 := by
    cases hc : M.hist σ.c with
    | nil =>
      rw [hc, List.length_nil] at hlen
      omega
    | cons e0 h1 => exact ⟨e0, h1, rfl⟩
  obtain ⟨q₀, hq⟩ : ∃ q₀, h1.length = q₀ + 1 := by
    refine ⟨opened st - 1, ?_⟩
    rw [hh, List.length_cons] at hlen
    omega
  have hl2 : (M.hist σ.c).length - 2 = q₀ := by
    rw [hh, List.length_cons, hq]
    omega
  have h20 : (nextAt spec sts (M.hist σ.c)).2 = 0 := by
    rcases Nat.eq_zero_or_pos (nextAt spec sts (M.hist σ.c)).2 with h | h
    · exact h
    · exact absurd (hsn.symm.trans ((hJ.seenAt i 1 stream hmem).2 hk0 h)) Bool.false_ne_true
  have hk1 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = 2 * i + 1 := by
    have := hk0.symm
    unfold nextAt at this
    rwa [hl2] at this
  have hk2 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = 0 := by
    have := h20
    unfold nextAt at this
    rwa [hl2] at this
  have hdiv : (2 * i + 1) / 2 = i := by omega
  obtain ⟨_, p, hfd⟩ := roundPlace_find hrp
  have hfa : frameAt σ.R sid i 1 = some p := by
    rw [← find_frameAt, hfd]
    rfl
  have hp : FreshRep p := fresh_found (hF.fresh i 1 stream hmem hsn) hfd
  have hpt : p.table = i := congrArg (fun x => x.2.1) (find_key hfd)
  have hfst : (p.coins.isEmpty && !p.st.begun) = true := by
    rw [hp.coins, hp.st]
    rfl
  obtain ⟨rel, name, l0, stm, p', hss, hstm', hn, hl0, hw⟩ := roundPlace_first_ok hfd hrp hfst
  have hsa : sessAt σ.R sid = some rel := hss
  obtain ⟨-, hsts⟩ := (hJ.sess hsent).eq hsa
  have hst' : rel.sts[p.table]? = some stm := by
    rw [hpt]
    exact hstm'
  have hstmb : sts[i]! = stm := by
    obtain ⟨hli, he⟩ := Array.getElem?_eq_some_iff.mp hstm'
    rw [← hsts, getElem!_pos rel.sts i hli, he]
  have hA₁ : (LeanRelease.ofLoop T L₀ real).answeredR M.x M.ρ (e0 :: h1) = .ok m := by
    rw [← hD.rel, ← hh]
    exact hA
  have hsw : 2 ≤ h1.length → Loop.repOf M.x (h1.length + 1) ≠ Loop.repOf M.x h1.length := by
    intro h2 he
    have e1 : Loop.repOf M.x (h1.length + 1) = (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 := by
      rw [show h1.length + 1 = q₀ + 2 by omega]
      exact LeanRelease.repOf_add_two hD.setup q₀
    have e2 : Loop.repOf M.x h1.length = (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (q₀ - 1)).1 := by
      rw [show h1.length = q₀ - 1 + 2 by omega]
      exact LeanRelease.repOf_add_two hD.setup (q₀ - 1)
    rw [e1, e2] at he
    have h' := Loop.locate_succ (Loop.roundsOf spec sts) 0 (q₀ - 1)
      (by rw [show q₀ - 1 + 1 = q₀ by omega]; exact he)
    rw [show q₀ - 1 + 1 = q₀ by omega, hk2] at h'
    omega
  have hcs : (h1.map fun e => ZKMal.CoinLayer.rd e.2).length = q₀ + 1 := by rw [List.length_map, hq]
  have hm := ofLoop_answeredR_switch T L₀ real M.x M.ρ e0 h1 hsw hA₁
  obtain ⟨c', t', dom', xs', hip, alg, hph, hal, -⟩ := Loop.freshOf_ok M.x _ hD.setup hcs hm
  have hsz : p'.alg.size = alg.size :=
    withAlg_size hw (fresh_alg hp) hst' hph (by rw [hk1, hdiv]; exact hstmb) hal
  have hpos : 0 < p'.alg.size := by
    rw [hsz]
    exact Loop.rounds_pos hal
  obtain ⟨_, p0, h0⟩ := roundPlace_rep0 hfd hrp hfst
  have hmem0 : (i, 0, s!"{spec.tables[i]!}/rep{(0 : Nat)}") ∈ spec.repStreams.toList := by
    have hs := rep_lt_size hmem
    rw [FlockDeploy.repStreams_size] at hs
    rw [FlockDeploy.repStreams_toList]
    refine List.mem_map.mpr ⟨2 * i, List.mem_range.mpr (by omega), ?_⟩
    show (2 * i / 2, 2 * i % 2, s!"{spec.tables[2 * i / 2]!}/rep{2 * i % 2}") = _
    rw [Nat.mul_div_cancel_left i (by decide : 0 < 2), Nat.mul_mod_right]
  have hsn0 : seen.contains s!"{spec.tables[i]!}/rep{(0 : Nat)}" = true := by
    cases hc : seen.contains s!"{spec.tables[i]!}/rep{(0 : Nat)}" with
    | true => rfl
    | false =>
      have hle := roundPlace_prior_size hfd hrp hfst hss hn hl0 hw h0
      rw [(fresh_found (hF.fresh i 0 _ hmem0 hc) h0).coins, Array.size_empty] at hle
      exact absurd hle (by omega)
  obtain ⟨p0', -, -, -, hfa0, -, -, -, -, hS0⟩ := hJ.reps i 0 _ hmem0 hsn0
  have hp0 : p0' = p0 := by
    have e : frameAt σ.R sid i 0 = some p0 := by
      rw [← find_frameAt, h0]
      rfl
    exact Option.some.inj (hfa0.symm.trans e)
  have hco := hS0 (by rw [hl2]; omega) (by rw [hl2]; omega)
  rw [hp0, hl2, hh, List.tail_cons, Nat.add_zero] at hco
  have hq2 : (e0 :: h1).length = q₀ + 2 := by rw [List.length_cons, hq]
  have hc : CfgAlike p.cfg c' := by
    rw [hpc p hfa, hD.rel, hh]
    exact ofLoop_cfgAlike T L₀ real M.x M.ρ (e0 :: h1) hD.setup hq2 hph
  obtain ⟨-, hqm⟩ := roundPlace_switch1 hfd hrp (fresh_repRun hp) hss hn hl0 hw h0 M.x _ hD.setup hcs
    (by rw [hk1]; omega) (n0 := (Loop.roundsOf spec sts).getD (2 * i) 0) (by rw [hk1, Nat.add_sub_cancel]) hph hal
    hc hsz hco hm
  rw [hqm]

/-- **`RealAt` at a frame round's slot of the deployed loop**: at the slot of a round, where `LoopFacts` and `LoopJ`
hold for the run of the loop's own tables on `src` (`Loop.realOf src`), the loop handed `src`'s bytes (`M.src`), and
the salts the loop keeps are the ones it was handed (`SaltsSrc`, `LoopRead`'s round clause), the model's tables and
salts are the loop's (`realAt_of`): past the `Commit` each level 0 bound to the session is the one the loop drew
(`LoopJ.built`) on the setup's entry (`LoopJ.sess`), so its table is `src`'s (`realAt_h0`). -/
theorem realAt_slot {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ (Loop.realOf src) s0 spec sts) (hsrc : M.src = src M.x M.w M.ρ)
    (hS : Loop.Level0Src src M.x M.w) (hlive : ZKMal.Masked.Live M.x)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hsal : SaltsSrc M sid σ) :
    RealAt M (Loop.realOf src) sid σ := by
  obtain ⟨hsent, hop, -, -⟩ := slot_round hsl
  have hc2 : 2 ≤ σ.c := by
    rw [hF.count hsent]
    omega
  exact realAt_of hsrc (realAt_h0 hsrc hD.setup (hJ.sess hsent) hD.names (hJ.built hc2) hS hlive) hsal

/-- **`PlaceFacts` at a round of a stream seen**: at the slot of a round of `stream`, table `i`'s rep `r`, where
`LoopFacts` and `LoopJ` hold, `seen` has a round of `stream` and the model's tables and salts are the loop's (`RealAt`),
the round's place is the model's scheduled round (`placeSched_seen`; where `placeOf` places it at the inner proof's own,
by `hown`), the frame `roundPlace` readies is at the model's masking state but for `stopped` (`placeSt_seen`), and `v`
fails the crossing only where it fails to pass (`hcr`). -/
theorem placeFacts_seen {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hseen : seen.contains stream = true) (hR : RealAt M real sid σ) {v : Claims.Msg}
    (hcr : M.L.cross M.x (M.hist σ.c).length M.w M.ρ (M.hist σ.c) v = false →
      M.L.pass M.x M.w M.ρ (M.hist σ.c) v = false)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd))
    (hown : ∀ (p : Rep) (rel : _root_.Flock.Firewall.Frames.Release) (name : String)
      (l0 : _root_.Flock.Firewall.Level0.Level0) n' q', frameAt σ.R sid i r = some p → sessAt σ.R sid = some rel →
      rel.spec.tables[i]? = some name → _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 →
      _root_.Flock.Firewall.Frames.placeOf σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 = .ok (.own, n', q') →
      pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
        (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩) :
    PlaceFacts M sid v σ q pc nd := by
  refine ⟨?_, placeSt_seen hsl hF hJ hrep hlt hD.names hseen hrp, fun _ => hcr⟩
  rcases placeSched_seen hD hsl hF hJ hrep hlt hseen hR hrp with h | ⟨p, rel, name, l0, hfa, hss, hn, hl0, n', q', ho⟩
  · exact h
  · exact hown p rel name l0 n' q' hfa hss hn hl0 ho

/-- **`PlaceFacts` at a frame round's slot of the deployed loop**: at the slot of a round of `stream`, table `i`'s rep
`r`, where `LoopFacts` and `LoopJ` hold, the model's tables and salts are the loop's (`RealAt`), the rep's frame is at
the model's configuration and the model answers the round, `PlaceFacts` holds at the place `roundPlace` readies, given
the crossing and, at a round of a stream seen, the inner proof's own place: there by `placeFacts_seen`, and at a
stream's first round by `placeSched_first`, with `placeSt_first0` at rep 0 and `placeSt_first1` at rep 1. -/
theorem placeFacts_slot {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hR : RealAt M real sid σ)
    (hpc : ∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c))
    (hans : ∃ m, M.L.answered M.x M.ρ (M.hist σ.c) = .ok m) {v : Claims.Msg}
    (hcr : M.L.cross M.x (M.hist σ.c).length M.w M.ρ (M.hist σ.c) v = false →
      M.L.pass M.x M.w M.ρ (M.hist σ.c) v = false)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd))
    (hown : seen.contains stream = true → ∀ (p : Rep) (rel : _root_.Flock.Firewall.Frames.Release) (name : String)
      (l0 : _root_.Flock.Firewall.Level0.Level0) n' q', frameAt σ.R sid i r = some p → sessAt σ.R sid = some rel →
      rel.spec.tables[i]? = some name → _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 →
      _root_.Flock.Firewall.Frames.placeOf σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 = .ok (.own, n', q') →
      pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
        (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩) :
    PlaceFacts M sid v σ q pc nd := by
  cases hsn : seen.contains stream
  · obtain ⟨m, hA⟩ := hans
    refine ⟨placeSched_first hD hsl hF hJ hrep hlt hsn hR hA hrp, ?_, fun _ => hcr⟩
    obtain ⟨-, hmem⟩ := round_rep hsl hF hJ hrep hlt hD.names
    have hr2 := rep_lt_two hmem
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · exact placeSt_first0 hD hsl hF hJ hrep hlt hsn hrp
    · exact placeSt_first1 hD hsl hF hJ hrep hlt hsn hpc hrp
  · exact placeFacts_seen hD hsl hF hJ hrep hlt hsn hR hcr hrp (hown hsn)

/-- **A frame round's slot is the model's masked round below the `Finish`**: where `LoopFacts` and `LoopJ` hold at the
deployed loop, the model's history so far is past the `Hello` and the `Commit` and its rounds fill the setup's plan. -/
theorem slot_masked {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ real s0 spec sts) {seen : List String} {st : State} {pl : Placed} {σ : LoopAt}
    {stream : String} {n : Nat} (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hlt : σ.c < M.τ.length) :
    M.L.kindAt M.x (M.hist σ.c).length = .masked ∧ (M.hist σ.c).length < M.L.last M.x := by
  obtain ⟨hsent, hop, -, hq⟩ := slot_round hsl
  have hlen : (M.hist σ.c).length = opened st + 1 := by
    rw [M.length_hist (Nat.le_of_lt hlt), hF.count hsent]
  have hq' : (planOf spec sts)[opened st - 1]? = some (stream, n) := by
    rw [← hJ.plan hsent]
    exact hq
  have hps : opened st - 1 < (planOf spec sts).size := (Array.getElem?_eq_some_iff.mp hq').1
  have hlast : M.L.last M.x = (planOf spec sts).size + 2 := by
    rw [hD.rel]
    show Loop.lastOf M.x = _
    rw [Loop.lastOf_eq, Claims.ZKMal.refK_of_flock hD.setup, Loop.rounds_eq hD.sizes, ← Loop.plan_size]
    omega
  have hlt' : (M.hist σ.c).length < M.L.last M.x := by omega
  refine ⟨?_, hlt'⟩
  have h0 : (M.hist σ.c).length ≠ 0 := by omega
  have h1 : ¬((M.hist σ.c).length = 1 ∨ (M.hist σ.c).length = M.L.last M.x) := by omega
  unfold LeanRelease.kindAt
  rw [ite_eq_right h0, ite_eq_right h1]

/-- **A stream is seen exactly where the model's last round was its rep's**: at the slot of a round of `stream`,
table `i`'s rep `r`, where `LoopFacts` and `LoopJ` hold at the deployed loop, the model's history so far is past the
`Hello` and the `Commit`, and `seen` has a round of `stream` exactly where it is past a later one whose last round was
of the same rep: the model's next round is the stream's past its first exactly there (`SeenAt`), and each round's rep
is its stream's (`LeanRelease.repOf_add_two`, `Loop.locate_pred`, `Loop.locate_first`). -/
theorem rep_seen {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ real s0 spec sts) {seen : List String} {st : State} {pl : Placed} {σ : LoopAt}
    {stream : String} {n i r : Nat} (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length) :
    2 ≤ (M.hist σ.c).length ∧ (seen.contains stream = true ↔
      3 ≤ (M.hist σ.c).length ∧ M.L.rep M.x (M.hist σ.c).length = M.L.rep M.x ((M.hist σ.c).length - 1)) := by
  obtain ⟨hsent, hop, -, -⟩ := slot_round hsl
  obtain ⟨hk0, hmem⟩ := round_rep hsl hF hJ hrep hlt hD.names
  have hlen : (M.hist σ.c).length = opened st + 1 := by
    rw [M.length_hist (Nat.le_of_lt hlt), hF.count hsent]
  have hro : ∀ k, M.L.rep M.x k = Loop.repOf M.x k := fun k => by
    rw [hD.rel]
    rfl
  obtain ⟨q₀, hq₀⟩ : ∃ q₀, (M.hist σ.c).length = q₀ + 2 := ⟨opened st - 1, by omega⟩
  have hnx : nextAt spec sts (M.hist σ.c) = ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀ := by
    unfold nextAt
    rw [hq₀, Nat.add_sub_cancel]
  have hS := hJ.seenAt i r stream hmem
  rw [hnx] at hS hk0
  refine ⟨by omega, fun hs => ?_, fun hc => ?_⟩
  · have hp : 0 < (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 := by
      rcases hS.1 hs with h | ⟨-, h⟩
      · omega
      · exact h
    obtain ⟨q₁, rfl⟩ : ∃ q₁, q₀ = q₁ + 1 := by
      rcases q₀ with _ | q₁
      · rw [Loop.locate_zero] at hp
        omega
      · exact ⟨q₁, rfl⟩
    have e := Loop.locate_pred (Loop.roundsOf spec sts) 0 q₁
      ((ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (q₁ + 1)).2 - 1) (by omega)
    refine ⟨by omega, ?_⟩
    rw [hro, hro, hq₀, show q₁ + 1 + 2 - 1 = q₁ + 2 by omega, LeanRelease.repOf_add_two hD.setup,
      LeanRelease.repOf_add_two hD.setup, e]
  · obtain ⟨h3, he⟩ := hc
    cases hs : seen.contains stream
    · have h0 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = 0 := by
        rcases Nat.eq_zero_or_pos (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 with h | h
        · exact h
        · exact absurd ((hS.2 hk0 h).symm.trans hs) (by decide)
      obtain ⟨q₁, rfl⟩ : ∃ q₁, q₀ = q₁ + 1 := ⟨q₀ - 1, by omega⟩
      rw [hro, hro, hq₀, show q₁ + 1 + 2 - 1 = q₁ + 2 by omega, LeanRelease.repOf_add_two hD.setup,
        LeanRelease.repOf_add_two hD.setup] at he
      exact absurd h0 (Loop.locate_first _ 0 q₁ he)
    · rfl

/-- **The model takes the coins after a frame round's slot's history**: at the slot of a round of `stream`, table
`i`'s rep `r`, where `LoopFacts` and `LoopJ` hold at the deployed loop, the loop took the coins so far
(`LoopModel.TookAt`) and the round goes out, the model answers its history so far, given the next rep's start at a
stream's first round (`hfr`) and, at a round of a stream seen, the round's place the model's scheduled round (`hsc`),
the empty round or the opening's first, at any count, wherever the rank stop fails at the model's coin-step state
(`hal`). The transport's check, the frame's earlier takings and its coin step are the loop's
(`LeanRelease.answered_of_took`); at a round of a stream seen the frame is at that state but for `stopped`
(`placeSt_seen`) and releases the round (`roundStep_out`), so the rank stop passes (`rank_of_release`). -/
theorem answered_of_round {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ real s0 spec sts) {seen : List String} {st : State} {pl : Placed} {e : Env}
    {σ σ' : LoopAt} {stream : String} {n i r : Nat} {dom : ByteArray} {v : Claims.Msg}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (htk : M.TookAt st σ.c) (hlt : σ.c < M.τ.length)
    (hr : RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R)
    (hpc : ∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) (ho : ∃ o, e.round = .ok o)
    (hfr : seen.contains stream = false → ∃ m, M.L.fresh M.x (M.hist σ.c) = .ok m)
    (hsc : seen.contains stream = true → ∀ q pc f nd,
      _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ n = .ok (q, pc, f, nd) →
      pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
        (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩)
    (hal : seen.contains stream = true → ∀ m, M.L.answeredR M.x M.ρ (M.hist σ.c) = .ok m →
      LeanRelease.rankFails (M.L.cfg M.x M.ρ (M.hist σ.c)) m = true →
      M.L.scheduled M.x M.w M.ρ (M.hist σ.c) (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c))) = [] ∨
        ∃ n', Flags.PcsRound (⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
          (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩, n')) :
    ∃ m, M.L.answered M.x M.ρ (M.hist σ.c) = .ok m := by
  obtain ⟨hsent, -, -, -⟩ := slot_round hsl
  obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hF.reach.sess hsent)
  obtain ⟨h2, hiff⟩ := rep_seen hD hsl hF hJ hr.rep_eq hlt
  obtain ⟨o, hro⟩ := ho
  have hout := (hr.round (request_round_slot hF.reach hF.live hsl)).symm.trans hro
  obtain ⟨q, pc, f, nd, hrp⟩ := roundPlace_of_out hout
  obtain ⟨_, p, hfd⟩ := roundPlace_find hrp
  have hfa : frameAt σ.R sid i r = some p := by
    rw [← find_frameAt, hfd]
    rfl
  refine LeanRelease.answered_of_took (htk.opens hC) htk.2.1 htk.2.2 h2 (fun hsw => ?_) (fun m hm h3 hs => ?_)
  · cases hsn : seen.contains stream
    · exact hfr hsn
    · obtain ⟨h3, he⟩ := hiff.1 hsn
      exact absurd he (hsw h3)
  · have hsn : seen.contains stream = true := hiff.2 ⟨h3, hs⟩
    obtain ⟨out, vd, hst, -⟩ := roundStep_out hrp hout
    exact rank_of_release (placeSt_seen hsl hF hJ hr.rep_eq hlt hD.names hsn hrp)
      ((roundPlace_cfg hfd hrp).trans (hpc p hfa)) (Prod.ext hst rfl)
      (fun m' hm' hrk => ⟨_, hsc hsn q pc f nd hrp,
        (hal hsn m' hm' hrk).imp_left fun he => by rw [he]; rfl⟩) m hm

/-- **Where the rank stop fails at a round of a stream seen, the model's scheduled round is the empty round or the
opening's first** (`answered_of_round`'s `hal`): at the slot of a round of `stream`, table `i`'s rep `r`, where
`LoopFacts` and `LoopJ` hold at the deployed loop on the loop's own tables (`Loop.realOf src`) at a `Live` statement,
and `seen` has a round of `stream`, wherever the coin step after the history so far leaves `m` and the rank stop fails
at `m`, the model's scheduled round is the empty round or the opening's first, at some count. The history is past a
round of the same rep (`rep_seen`), so its next round is the rep's round past its first (`Loop.locate_first`), on a
table of the setup (`rep_lt_size`), which passes `sizesOk` (`LoopJ.sizes`) and whose tables are `Live`'s
(`Loop.tabs_of_live`): `LeanRelease.ofLoop_rank_sched`. -/
theorem rank_sched_deployed {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ (Loop.realOf src) s0 spec sts) (hlive : ZKMal.Masked.Live M.x)
    {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hsn : seen.contains stream = true) {m : Masker} (hA : M.L.answeredR M.x M.ρ (M.hist σ.c) = .ok m)
    (hr : LeanRelease.rankFails (M.L.cfg M.x M.ρ (M.hist σ.c)) m = true) (vals : String → Option ByteArray) :
    M.L.scheduled M.x M.w M.ρ (M.hist σ.c) vals = [] ∨
      ∃ n', Flags.PcsRound (⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c) vals).toArray⟩, n') := by
  obtain ⟨hsent, -, -, -⟩ := slot_round hsl
  obtain ⟨hk0, hmem⟩ := round_rep hsl hF hJ hrep hlt hD.names
  obtain ⟨-, hiff⟩ := rep_seen hD hsl hF hJ hrep hlt
  obtain ⟨h3, hre⟩ := hiff.1 hsn
  have hro : ∀ k, M.L.rep M.x k = Loop.repOf M.x k := fun k => by
    rw [hD.rel]
    rfl
  obtain ⟨q₁, hq₀⟩ : ∃ q₁, (M.hist σ.c).length = q₁ + 1 + 2 := ⟨(M.hist σ.c).length - 3, by omega⟩
  rw [hro, hro, hq₀, show q₁ + 1 + 2 - 1 = q₁ + 2 by omega, LeanRelease.repOf_add_two hD.setup,
    LeanRelease.repOf_add_two hD.setup] at hre
  have hj0 := Loop.locate_first _ 0 q₁ hre
  unfold nextAt at hk0
  rw [hq₀, Nat.add_sub_cancel] at hk0
  have hki : ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (q₁ + 1) =
      (2 * i + r, (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (q₁ + 1)).2 - 1 + 1) :=
    Prod.ext hk0.symm (by omega)
  have hks : 2 * i + r < spec.repStreams.size := rep_lt_size hmem
  obtain ⟨hm14, hk6, hmk⟩ := Loop.tabs_of_live hlive hD.setup (j := (2 * i + r) / 2)
    (by have := hD.sizes; omega)
  obtain ⟨e0, h1, he⟩ : ∃ e0 h1, M.hist σ.c = e0 :: h1 := by
    cases hh : M.hist σ.c with
    | nil => rw [hh, List.length_nil] at hq₀; omega
    | cons e0 h1 => exact ⟨e0, h1, rfl⟩
  have hq : h1.length = q₁ + 1 + 1 := by
    rw [he, List.length_cons] at hq₀
    omega
  rw [hD.rel, he] at hA hr ⊢
  exact LeanRelease.ofLoop_rank_sched T L₀ (Loop.realOf src) M.x M.w M.ρ e0 h1 vals hD.setup (hJ.sizes hsent)
    (Loop.realTabs_realOf src M.x M.w M.ρ hD.setup) hq hki hks hm14 hk6 hmk hA hr

/-- **`RoundFacts` at a frame round's slot of the deployed loop**: at the slot of a round of `n` coins of `stream`,
where `LoopFacts` and `LoopJ` hold, the session's slots took fewer than the run's messages, and the loop ran the round
on table `i`'s rep `r` (`RoundRun`), the rep's frame at the model's configuration and the round out, the model's answer
at its history so far a masking state, and `PlaceFacts` at the place `roundPlace` readied the round at: the model's
coin count there is the slot's, its round masked, not its last, with a next message (`LoopModel.Deployed`); the round's
mark is whether `seen` has no round of `stream` (`FreshAt`, or `SameInv` at the rep's `RepJ`, `round_rep`); and the
frames stopped the session where the model's run has (`LoopJ.stop`). -/
theorem roundFacts_loop {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ real s0 spec sts) {seen : List String} {st : State} {pl : Placed} {e : Env}
    {σ σ' : LoopAt} {stream : String} {n i r : Nat} {dom : ByteArray} {v : Claims.Msg}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hlt : σ.c < M.τ.length)
    (hr : RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R)
    (hpc : ∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) (ho : ∃ o, e.round = .ok o)
    (hans : ∃ m, M.L.answered M.x M.ρ (M.hist σ.c) = .ok m)
    (hpl : ∀ q pc f nd, _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ n =
      .ok (q, pc, f, nd) → PlaceFacts M sid v σ q pc nd) :
    RoundFacts M sid spec seen stream n i r dom v σ := by
  obtain ⟨hsent, hop, -, hq⟩ := slot_round hsl
  have hlen : (M.hist σ.c).length = opened st + 1 := by
    rw [M.length_hist (Nat.le_of_lt hlt), hF.count hsent]
  obtain ⟨hkind, hlt'⟩ := slot_masked hD hsl hF hJ hlt
  have hcoins : M.L.coins M.x (M.hist σ.c).length = n := by
    rw [hlen, hD.rel]
    show Loop.loopNr M.x (opened st) = n
    unfold Loop.loopNr
    rw [Loop.reqOf_eq hD.setup, ← hJ.plan hsent]
    unfold reqAt
    rw [ite_eq_right hop, Array.getD_eq_getD_getElem?, hq, Option.getD_some]
  obtain ⟨hk, hmem⟩ := round_rep hsl hF hJ hr.rep_eq hlt hD.names
  obtain ⟨o, hro⟩ := ho
  obtain ⟨q, pc, f, nd, hrp⟩ := roundPlace_of_out
    ((hr.round (request_round_slot hF.reach hF.live hsl)).symm.trans hro)
  obtain ⟨_, p, hfd⟩ := roundPlace_find hrp
  have hfa : frameAt σ.R sid i r = some p := by
    rw [← find_frameAt, hfd]
    rfl
  have hP := hpl q pc f nd hrp
  obtain ⟨m, hm⟩ := hans
  have hnd := roundPlace_nd_eq hrp
  have hu := hP.st m (LeanRelease.answeredR_of_answered hm)
  refine ⟨hcoins, hkind, by have := hD.next; omega, hlt', hP.cross hnd, Option.isSome_iff_exists.mpr ⟨p, hfa⟩,
    ⟨m, q, f, nd, hm, ?_, hnd, (roundPlace_cfg hfd hrp).trans (hpc p hfa), hu, ?_⟩, ?_,
    streamRep_other hF.setup hsent (slot_round_open hsl) hr.rep_eq⟩
  · rw [← hP.sched]
    exact hrp
  · cases hsn : seen.contains stream
    · have hpF := fresh_found (hF.fresh i r stream hmem hsn) hfd
      rw [roundPlace_first hfd hrp, hpF.coins, hpF.st]
      rfl
    · obtain ⟨p', _, _, _, hf', -, -, -, hS1, -⟩ := hJ.reps i r stream hmem hsn
      obtain ⟨_, _, _, _, _, _, _, _, -, -, -, -, -, -, hI, -⟩ := hS1 hk.symm
      exact (sameInv_place hI hf' hrp).1
  · rw [hJ.stop, hsent, Bool.true_and]

/-- **The inner proof's own place at a round of a stream seen is the model's scheduled round**: at each frame round's
slot of `stream` as `loopKept_round` reads it, `seen` having a round of `stream`, where `roundPlace` readies the round
and `placeOf` places it at the inner proof's own, the place `roundPlace` readies is the model's scheduled round
(`placeFacts_slot`'s `hown`, at each slot). -/
def OwnSched {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) (stream : String)
    (n : Nat) : Prop :=
  ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt) (i r : Nat) (dom : ByteArray)
    (v : Claims.Msg), slot st pl = some (.round stream n) → LoopFacts sid spec seen st pl σ →
    LoopJ M sid spec sts seen st σ → σ.c < M.τ.length → M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
    RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
    (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → SaltsSrc M sid σ →
    InnerSrc M sid i r σ → StartOk sts sid i r σ → (∃ o, e.round = .ok o) → seen.contains stream = true →
    ∀ (q : Rep) (pc : Place) (f : Bool) (nd : Option String),
      _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ n = .ok (q, pc, f, nd) →
      ∀ (p : Rep) (rel : _root_.Flock.Firewall.Frames.Release) (name : String)
        (l0 : _root_.Flock.Firewall.Level0.Level0) n' q', frameAt σ.R sid i r = some p → sessAt σ.R sid = some rel →
        rel.spec.tables[i]? = some name → _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 →
        _root_.Flock.Firewall.Frames.placeOf σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 = .ok (.own, n', q') →
        pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
          (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩

/-- **The next rep's start at a stream's first round** (`answered_of_round`'s `hfr`): at each frame round's slot of
`stream` as `loopKept_round` reads it, the round going out and `seen` having no round of `stream`, the model's next-rep
state after its history so far (`LeanRelease.fresh`) is a masking state, not a refusal. -/
def FreshStart {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt)
    (stream : String) (n : Nat) : Prop :=
  ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt) (i r : Nat) (dom : ByteArray)
    (v : Claims.Msg), slot st pl = some (.round stream n) → LoopFacts sid spec seen st pl σ →
    LoopJ M sid spec sts seen st σ → σ.c < M.τ.length → M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
    RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
    (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → SaltsSrc M sid σ →
    InnerSrc M sid i r σ → StartOk sts sid i r σ → (∃ o, e.round = .ok o) → seen.contains stream = false →
    ∃ m, M.L.fresh M.x (M.hist σ.c) = .ok m

/-- **The model answers a frame round's slot's history of the deployed loop**, where the loop took the coins so far
(`LoopModel.TookAt`): at the loop's own tables on `src` (`Loop.realOf src`), whose bytes the loop handed (`M.src`), at
each frame round's slot as `roundFacts_deployed` reads it, `answered_of_round` with the next rep's start by
`FreshStart`, the round's place at a stream seen the model's scheduled round by `placeFacts_seen` (the tables and
salts the loop's by `realAt_slot`, the crossing by `LeanRelease.CrossPasses`, the inner proof's own place by
`OwnSched`), and the empty round or the opening's first wherever the rank stop fails by `rank_sched_deployed`. -/
theorem answered_deployed {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ (Loop.realOf src) s0 spec sts) (hsrc : M.src = src M.x M.w M.ρ)
    (hS : Loop.Level0Src src M.x M.w) (hlive : ZKMal.Masked.Live M.x) (hcp : M.L.CrossPasses)
    {sid : Nat} {stream : String} {n : Nat} (hown : OwnSched M sid spec sts stream n)
    (hfs : FreshStart M sid spec sts stream n) {seen : List String} {st : State} {pl : Placed} {e : Env}
    {σ σ' : LoopAt} {i r : Nat} {dom : ByteArray} {v : Claims.Msg} (hsl : slot st pl = some (.round stream n))
    (hF : LoopFacts sid spec seen st pl σ) (hJ : LoopJ M sid spec sts seen st σ) (htk : M.TookAt st σ.c)
    (hlt : σ.c < M.τ.length) (hv : M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false))
    (hr : RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R)
    (hpc : ∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) (hsal : SaltsSrc M sid σ)
    (hin : InnerSrc M sid i r σ) (hsk : StartOk sts sid i r σ) (ho : ∃ o, e.round = .ok o) :
    ∃ m, M.L.answered M.x M.ρ (M.hist σ.c) = .ok m := by
  obtain ⟨hk, hl⟩ := slot_masked hD hsl hF hJ hlt
  exact answered_of_round hD hsl hF hJ htk hlt hr hpc ho
    (hfs seen st pl e σ σ' i r dom v hsl hF hJ hlt hv hr hpc hsal hin hsk ho)
    (fun hsn q pc f nd hrp => (placeFacts_seen hD hsl hF hJ hr.rep_eq hlt hsn
      (realAt_slot hD hsrc hS hlive hsl hF hJ hsal) (hcp M.x M.w M.ρ (M.hist σ.c) v hk hl) hrp
      (hown seen st pl e σ σ' i r dom v hsl hF hJ hlt hv hr hpc hsal hin hsk ho hsn q pc f nd hrp)).sched)
    (fun hsn _ hm hrk => rank_sched_deployed hD hlive hsl hF hJ hr.rep_eq hlt hsn hm hrk _)

/-- **`RoundFacts` at each frame round's slot of the deployed loop** (`loopKept_round`'s `hA`): at the loop's own tables
on `src` (`Loop.realOf src`), whose bytes the loop handed (`M.src`), `roundFacts_loop` at the model's answer, which
`answered_deployed` gives from what the loop took (`LoopJ.took`) and the next rep's start (`FreshStart`), and
`PlaceFacts` by `placeFacts_slot`, at the loop's tables and salts (`realAt_slot`), the crossing at the slot's masked
round below the `Finish` (`LeanRelease.CrossPasses`, `slot_masked`) and the inner proof's own place (`OwnSched`). -/
theorem roundFacts_deployed {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ (Loop.realOf src) s0 spec sts) (hsrc : M.src = src M.x M.w M.ρ)
    (hS : Loop.Level0Src src M.x M.w) (hlive : ZKMal.Masked.Live M.x) (hcp : M.L.CrossPasses)
    {sid : Nat} {stream : String} {n : Nat} (hown : OwnSched M sid spec sts stream n)
    (hfs : FreshStart M sid spec sts stream n) :
    ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt) (i r : Nat) (dom : ByteArray)
      (v : Claims.Msg), slot st pl = some (.round stream n) → LoopFacts sid spec seen st pl σ →
      LoopJ M sid spec sts seen st σ → σ.c < M.τ.length → M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
      RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
      (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → SaltsSrc M sid σ →
      InnerSrc M sid i r σ → StartOk sts sid i r σ → (∃ o, e.round = .ok o) →
      RoundFacts M sid spec seen stream n i r dom v σ := by
  intro seen st pl e σ σ' i r dom v hsl hF hJ hlt hv hr hpc hsal hin hsk ho
  obtain ⟨hk, hl⟩ := slot_masked hD hsl hF hJ hlt
  have hans := answered_deployed hD hsrc hS hlive hcp hown hfs hsl hF hJ (hJ.took hlt) hlt hv hr hpc hsal hin hsk ho
  exact roundFacts_loop hD hsl hF hJ hlt hr hpc ho hans fun q pc f nd hrp =>
    placeFacts_slot hD hsl hF hJ hr.rep_eq hlt (realAt_slot hD hsrc hS hlive hsl hF hJ hsal) hpc hans
      (hcp M.x M.w M.ρ (M.hist σ.c) v hk hl) hrp
      (fun hsn => hown seen st pl e σ σ' i r dom v hsl hF hJ hlt hv hr hpc hsal hin hsk ho hsn q pc f nd hrp)

end Claims.ZKServer.Relay
