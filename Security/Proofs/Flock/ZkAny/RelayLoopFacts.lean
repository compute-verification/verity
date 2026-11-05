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
public import Proofs.Flock.SessionLoop.Statement

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
* `roundPlace_find`: the frame `roundPlace` readies a round on is one the loop's frames hold.
* `PlaceFacts`: what the model holds at the place `roundPlace` readies a round at: its scheduled round, its masking
  state but for `stopped`, and where `v` fails the crossing. `placeSt_seen`: its masking state at a round of a stream
  seen, from the rep's `SameInv`.
* `roundPlace_scheduled_later`: past a stream's first round, on a frame keeping its schedule so far on the loop's
  table, extras and salts, which are the model's, the round's place is the model's scheduled round, or the schedule
  has no round there (the inner proof's own).
* **`roundFacts_loop`**: `RoundFacts` at a frame round's slot of the deployed loop, given the model's answer there and
  `PlaceFacts` at the round's place.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed planOf reqAt)
open _root_.Flock (V SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep)
open _root_.Flock.Firewall.Frame (Masker Place)

/-- **The model's run is the deployed loop's**: its release is the loop's own frames' (`LeanRelease.ofLoop`) at the
tables `T`, the release `L₀` and the tables' draws `real`, the statement sets up as `(s0, spec, sts)`, the setup's rep
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

/-- **The model's tables and salts are the loop's**, at the loop `σ` for session `sid`: at each rep `r` of each table
`i` of the session's entry, the model's draws for rep stream `2 * i + r` (`real`) are the table the loop's level 0 for
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
place is the model's scheduled round there, `q`'s masking state the model's but for `stopped`, where the round takes a
device round exactly where the frames haven't stopped session `sid`, `v` fails the crossing only where it fails to
pass. -/
structure PlaceFacts {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (v : Claims.Msg) (σ : LoopAt) (q : Rep)
    (pc : Place) (nd : Option String) : Prop where
  sched : pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
    (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩
  st : ∀ m, M.L.answered M.x M.ρ (M.hist σ.c) = .ok m → unstop q.st.m = unstop m
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
    ∀ m, M.L.answered M.x M.ρ (M.hist σ.c) = .ok m → unstop q.st.m = unstop m := by
  obtain ⟨hk, hmem⟩ := round_rep hsl hF hJ hrep hlt hnd
  obtain ⟨p, rel, name, l0, hfa, -, -, -, hA⟩ := hJ.reps i r stream hmem hseen
  obtain ⟨j, c, stm, t, xs, hi, pre, ev, -, -, -, -, -, -, hI, -⟩ := hA.1 hk.symm
  exact (sameInv_place hI hfa hrp).2.2.2.1

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
  have hq' : (planOf spec sts)[opened st - 1]? = some (stream, n) := by
    rw [← hJ.plan hsent]
    exact hq
  have hps : opened st - 1 < (planOf spec sts).size := (Array.getElem?_eq_some_iff.mp hq').1
  have hlast : M.L.last M.x = (planOf spec sts).size + 2 := by
    rw [hD.rel]
    show Loop.lastOf M.x = _
    rw [Loop.lastOf_eq, Loop.refK_of_setup hD.setup, Loop.rounds_eq hD.sizes, ← Loop.plan_size]
    omega
  have hlt' : (M.hist σ.c).length < M.L.last M.x := by omega
  have hcoins : M.L.coins M.x (M.hist σ.c).length = n := by
    rw [hlen, hD.rel]
    show Loop.loopNr M.x (opened st) = n
    unfold Loop.loopNr
    rw [Loop.reqOf_eq hD.setup, ← hJ.plan hsent]
    unfold reqAt
    rw [ite_eq_right hop, Array.getD_eq_getD_getElem?, hq, Option.getD_some]
  have hkind : M.L.kindAt M.x (M.hist σ.c).length = .masked := by
    have h0 : (M.hist σ.c).length ≠ 0 := by omega
    have h1 : ¬((M.hist σ.c).length = 1 ∨ (M.hist σ.c).length = M.L.last M.x) := by omega
    unfold LeanRelease.kindAt
    rw [ite_eq_right h0, ite_eq_right h1]
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
  refine ⟨hcoins, hkind, by have := hD.next; omega, hlt', hP.cross hnd, Option.isSome_iff_exists.mpr ⟨p, hfa⟩,
    ⟨m, q, f, nd, hm, ?_, hnd, (roundPlace_cfg hfd hrp).trans (hpc p hfa), hP.st m hm, ?_⟩, ?_,
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

end Claims.ZKServer.Relay
