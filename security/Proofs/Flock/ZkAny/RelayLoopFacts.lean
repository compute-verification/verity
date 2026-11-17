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

/-! What the loop's frames and the model hold at a frame round of the deployed loop (`roundFacts_deployed`), under the named hypotheses `CrossPasses`, `OwnSched` and `FreshStart`. -/

@[expose] public section

namespace Claims.ZKServer.LeanRelease

/-- A named hypothesis on the Rust release: no Lean code fixes `cross`, and `cross`'s contract (each check can only fail a
round more) does not give it. It is stated at every history, unguarded: Rust stops a session at its first failing
round, so the unguarded form constrains the Rust release no more than a guarded one. -/
def CrossPasses {W R : Type} (L : LeanRelease W R) : Prop :=
  ∀ (x : Msg) (w : W) (ρ : R) (h : List (Msg × Msg)) (v : Msg), L.kindAt x h.length = .masked →
    h.length < L.last x → L.cross x h.length w ρ h v = false → L.pass x w ρ h v = false

end Claims.ZKServer.LeanRelease

namespace Claims.ZKServer.Relay

open _root_.Flock.ZkServer.Session (State Placed planOf reqAt)
open _root_.Flock (V SessionSpec Stmt)
open _root_.Flock.ZkServer.Frames (Sessions Rep)
open _root_.Flock.ZkServer.Frame (Masker Place)

structure LoopModel.Deployed {T : ZKMal.CoinLayer.TableData}
    (M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) : Prop where
  rel : M.L = LeanRelease.ofLoop T L₀ real
  setup : FlockDeploy.setupOf true M.x = some (s0, spec, sts)
  names : (spec.repStreams.toList.map (·.2.2)).Nodup
  sizes : spec.repStreams.size = 2 * sts.size
  next : M.L.last M.x < M.nN M.x

theorem roundPlace_find {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (h : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pc, f, nd)) :
    ∃ j p, R.find sid i r = some (j, p) := by
  rcases hf : R.find sid i r with _ | ⟨j, p⟩
  · unfold _root_.Flock.ZkServer.Frames.roundPlace at h
    simp only [hf] at h
    cases h
  · exact ⟨j, p, rfl⟩

set_option maxHeartbeats 1000000 in
theorem roundPlace_prior_size {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {sid i j j0 : Nat}
    {dom content : ByteArray} {n : Nat} {p p0 p' q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i 1 = some (j, p))
    (h : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i 1 dom content n = .ok (q, pl, f, nd))
    (hfst : (p.coins.isEmpty && !p.st.begun) = true) {s : _root_.Flock.ZkServer.Frames.Release} {name : String}
    {l0 : _root_.Flock.ZkServer.Level0.Level0} (hs : R.sessions.find? (·.sid == sid) = some s)
    (hn : s.spec.tables[i]? = some name) (hl0 : _root_.Flock.ZkServer.Level0.ofSession L0 sid name = some l0)
    (hw : _root_.Flock.ZkServer.Frames.withAlg s l0 name p = .ok p') (h0 : R.find sid i 0 = some (j0, p0)) :
    p'.alg.size ≤ p0.coins.size := by
  rcases Nat.lt_or_ge p0.coins.size p'.alg.size with hlt | hge
  · exfalso
    have h11 : ((1 : Nat) == 1) = true := rfl
    unfold _root_.Flock.ZkServer.Frames.roundPlace at h
    simp only [hf, bind, Except.bind, pure, Except.pure] at h
    repeat' (first
      | simp only [hs, hn, hl0, hfst, hw, h11, h0, hlt, ↓reduceIte] at h
      | split at h)
    all_goals first
      | (cases h; done)
      | exact absurd (by assumption) throw_ne_ok
  · exact hge

def RealAt {T : ZKMal.CoinLayer.TableData}
    (M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)) (sid : Nat) (σ : LoopAt) : Prop :=
  ∀ (i r : Nat) (rel : _root_.Flock.ZkServer.Frames.Release) (name : String)
    (l0 : _root_.Flock.ZkServer.Level0.Level0) (stm : Stmt) (t : _root_.Flock.ZkServer.Finish.Tab), r < 2 →
    sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
    _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0 → rel.sts[i]? = some stm →
    _root_.Flock.ZkServer.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t →
    ∃ hi, real M.x M.w M.ρ (2 * i + r) = .ok (t, hi) ∧ ∀ r0 q0 hi',
      σ.R.salts.find? (fun x => x.1 == l0.root && x.2.1 == r) = some (r0, q0, hi') → hi' = hi

theorem realOf_hi {R : Type}
    {src : Msg → Claims.Flock.Wit → R → Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {x : Msg} {w : Claims.Flock.Wit} {ρ : R} {k : Nat} {t : _root_.Flock.ZkServer.Finish.Tab}
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

theorem realAt_of {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)} {sid : Nat} {σ : LoopAt}
    (hsrc : M.src = src M.x M.w M.ρ)
    (h0 : ∀ (i r : Nat) (rel : _root_.Flock.ZkServer.Frames.Release) (name : String)
      (l0 : _root_.Flock.ZkServer.Level0.Level0) (stm : Stmt) (t : _root_.Flock.ZkServer.Finish.Tab), r < 2 →
      sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
      _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0 → rel.sts[i]? = some stm →
      _root_.Flock.ZkServer.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t →
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

theorem roundPlace_scheduled_later (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Claims.Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (vals : String → Option ByteArray)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q₀ j i r : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q₀ + 1)
    (hk : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = 2 * i + r) (hr2 : r < 2)
    (hj : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = j + 1)
    {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {sid : Nat} {dom content : ByteArray} {n : Nat}
    {p q : Rep} {pc : Place} {f : Bool} {nd : Option String} (hfa : frameAt R sid i r = some p)
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pc, f, nd))
    {s : _root_.Flock.ZkServer.Frames.Release} {name : String} {l0 : _root_.Flock.ZkServer.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.ZkServer.Level0.ofSession L0 sid name = some l0) (hspec : s.spec = spec)
    (hsts : s.sts = sts) (hlink : s.link = some (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))
    {stm : Stmt} {t : _root_.Flock.ZkServer.Finish.Tab} {hi : Array ByteArray} {r0 : ByteArray} {q0 : Nat}
    (hst : sts[i]? = some stm)
    (htab : _root_.Flock.ZkServer.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hreal : real x w ρ (2 * i + r) = .ok (t, hi))
    (hI : Loop.RepInv t (spec.domain name r).toUTF8 (_root_.Flock.ZkServer.Drive.extras stm
      (_root_.Flock.ZkServer.Drive.linkPoints stm (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi p)
    (hco : p.coins = (Array.range (j + 1)).map fun k =>
      Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) (q₀ - (j + 1) + k + 1))
    (hsalt : p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
      R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    {sched : Array (ByteArray × Nat)}
    (hsch : Loop.schedSoFar t r (spec.domain name r).toUTF8 (_root_.Flock.ZkServer.Drive.extras stm
      (_root_.Flock.ZkServer.Drive.linkPoints stm (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
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

theorem roundPlace_scheduled_first (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Claims.Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (vals : String → Option ByteArray)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q₀ i r : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q₀ + 1)
    (hk : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = 2 * i + r) (hr2 : r < 2)
    (hj : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = 0)
    {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {sid : Nat} {dom content : ByteArray} {n : Nat}
    {p q : Rep} {pc : Place} {f : Bool} {nd : Option String} (hfa : frameAt R sid i r = some p) (hp : FreshRep p)
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pc, f, nd))
    {s : _root_.Flock.ZkServer.Frames.Release} {name : String} {l0 : _root_.Flock.ZkServer.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.ZkServer.Level0.ofSession L0 sid name = some l0) (hspec : s.spec = spec)
    (hsts : s.sts = sts) (hlink : s.link = some (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))
    {p' : Rep} (hw : _root_.Flock.ZkServer.Frames.withAlg s l0 name p = .ok p')
    {stm : Stmt} {t : _root_.Flock.ZkServer.Finish.Tab} {hi : Array ByteArray} {r0 : ByteArray} {q0 : Nat}
    (hst : sts[i]? = some stm)
    (htab : _root_.Flock.ZkServer.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hcap : _root_.Flock.ZkServer.Finish.flat t.cm0.cap = l0.root)
    (hreal : real x w ρ (2 * i + r) = .ok (t, hi))
    (hsalt : p'.alg.size = 0 → R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    {sched : Array (ByteArray × Nat)}
    (hsch : Loop.schedSoFar t r (spec.domain name r).toUTF8 (_root_.Flock.ZkServer.Drive.extras stm
      (_root_.Flock.ZkServer.Drive.linkPoints stm (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
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
  have hI' := repInv_withAlg hp hw hst' htab hcap (_root_.Flock.ZkServer.Drive.extras stm
    (_root_.Flock.ZkServer.Drive.linkPoints stm (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
  obtain ⟨htb'', hrep'', hco'', hal'', hI''⟩ : p''.table = p.table ∧ p''.rep = p.rep ∧ p''.coins = #[] ∧
      p''.alg = p'.alg ∧ Loop.RepInv t (s.spec.domain name p.rep).toUTF8 (_root_.Flock.ZkServer.Drive.extras stm
        (_root_.Flock.ZkServer.Drive.linkPoints stm (Loop.wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
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

structure PlaceFacts {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (v : Claims.Msg) (σ : LoopAt) (q : Rep)
    (pc : Place) (nd : Option String) : Prop where
  sched : pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
    (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩
  st : ∀ m, M.L.answeredR M.x M.ρ (M.hist σ.c) = .ok m → unstop q.st.m = unstop m
  cross : nd.isSome = σ.R.stopped sid → M.L.cross M.x (M.hist σ.c).length M.w M.ρ (M.hist σ.c) v = false →
    M.L.pass M.x M.w M.ρ (M.hist σ.c) v = false

theorem placeSt_seen {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) (hseen : seen.contains stream = true)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd)) :
    ∀ m, M.L.answeredR M.x M.ρ (M.hist σ.c) = .ok m → unstop q.st.m = unstop m := by
  obtain ⟨hk, hmem⟩ := round_rep hsl hF hJ hrep hlt hnd
  obtain ⟨p, rel, name, l0, hfa, -, -, -, hA⟩ := hJ.reps i r stream hmem hseen
  obtain ⟨j, c, stm, t, xs, hi, pre, ev, -, -, -, -, -, -, hI, -⟩ := hA.1 hk.symm
  exact (sameInv_place hI hfa hrp).2.2.2.1

theorem placeSched_later {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {σ : LoopAt} {i r j : Nat} (hr2 : r < 2) (hR : RealAt M real sid σ)
    {e0 : Claims.Msg × Claims.Msg} {h1 : List (Claims.Msg × Claims.Msg)} {q₀ : Nat}
    (hh : M.hist σ.c = e0 :: h1) (hq : h1.length = q₀ + 1)
    (hk : nextAt spec sts (M.hist σ.c) = (2 * i + r, j + 1))
    {p : Rep} {rel : _root_.Flock.ZkServer.Frames.Release} {name : String} {l0 : _root_.Flock.ZkServer.Level0.Level0}
    (hfa : frameAt σ.R sid i r = some p) (hss : sessAt σ.R sid = some rel) (hn : rel.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0)
    (hA : StreamAt M spec sts (2 * i + r) (M.hist σ.c) rel name l0 p σ.R)
    (hspec : rel.spec = spec) (hsts : rel.sts = sts)
    (hlink : rel.link = some (Loop.wordsAt M.x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))
    {dom content : ByteArray} {n : Nat} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd)) :
    pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
        (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩ ∨
      ∃ n' q', _root_.Flock.ZkServer.Frames.placeOf σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 =
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
      (_root_.Flock.ZkServer.Drive.extras stm (_root_.Flock.ZkServer.Drive.linkPoints stm
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

theorem placeSched_seen {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hseen : seen.contains stream = true) (hR : RealAt M real sid σ)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd)) :
    pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
        (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩ ∨
      ∃ (p : Rep) (rel : _root_.Flock.ZkServer.Frames.Release) (name : String)
        (l0 : _root_.Flock.ZkServer.Level0.Level0),
        frameAt σ.R sid i r = some p ∧ sessAt σ.R sid = some rel ∧ rel.spec.tables[i]? = some name ∧
        _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0 ∧
        ∃ n' q', _root_.Flock.ZkServer.Frames.placeOf σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 =
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

theorem placeSched_first {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hsn : seen.contains stream = false) (hR : RealAt M real sid σ) {m : Masker}
    (hA : M.L.answered M.x M.ρ (M.hist σ.c) = .ok m)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd)) :
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
  obtain ⟨t, htab⟩ : ∃ t, _root_.Flock.ZkServer.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t := by
    have h := tab_of_placeholder (g := gb) (d := db) (l := l0) hph (by rw [hdiv, hstmb]; exact hdb)
    rwa [hdiv, hstmb] at h
  have hcap := cap_of_sampled hdb htab
  obtain ⟨hi, hreal, -⟩ := hR i r rel name l0 stm t hr2 hsa hn hl0 hstm' htab
  have hst' : rel.sts[p.table]? = some stm := by
    rw [hpt]
    exact hstm'
  have hI' := repInv_withAlg hp hw hst' htab hcap (_root_.Flock.ZkServer.Drive.extras stm
    (_root_.Flock.ZkServer.Drive.linkPoints stm (Loop.wordsAt M.x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
  have hpos : 0 < p'.alg.size := Loop.rounds_pos hI'.alg
  obtain ⟨-, hrw, hcw, -, -, -⟩ := withAlg_keep hw
  have hsch := Loop.schedSoFar_alg t _ (_root_.Flock.ZkServer.Drive.extras stm
    (_root_.Flock.ZkServer.Drive.linkPoints stm (Loop.wordsAt M.x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi p'
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

theorem placeSt_first0 {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, 0)) (hlt : σ.c < M.τ.length)
    (hsn : seen.contains stream = false)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace σ.R σ.L0 sid i 0 dom content n = .ok (q, pc, f, nd)) :
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

theorem placeSt_first1 {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, 1)) (hlt : σ.c < M.τ.length)
    (hsn : seen.contains stream = false)
    (hpc : ∀ p, frameAt σ.R sid i 1 = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c))
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace σ.R σ.L0 sid i 1 dom content n = .ok (q, pc, f, nd)) :
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

theorem placeFacts_seen {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hseen : seen.contains stream = true) (hR : RealAt M real sid σ) {v : Claims.Msg}
    (hcr : M.L.cross M.x (M.hist σ.c).length M.w M.ρ (M.hist σ.c) v = false →
      M.L.pass M.x M.w M.ρ (M.hist σ.c) v = false)
    {dom content : ByteArray} {q : Rep} {pc : Place} {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd))
    (hown : ∀ (p : Rep) (rel : _root_.Flock.ZkServer.Frames.Release) (name : String)
      (l0 : _root_.Flock.ZkServer.Level0.Level0) n' q', frameAt σ.R sid i r = some p → sessAt σ.R sid = some rel →
      rel.spec.tables[i]? = some name → _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0 →
      _root_.Flock.ZkServer.Frames.placeOf σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 = .ok (.own, n', q') →
      pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
        (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩) :
    PlaceFacts M sid v σ q pc nd := by
  refine ⟨?_, placeSt_seen hsl hF hJ hrep hlt hD.names hseen hrp, fun _ => hcr⟩
  rcases placeSched_seen hD hsl hF hJ hrep hlt hseen hR hrp with h | ⟨p, rel, name, l0, hfa, hss, hn, hl0, n', q', ho⟩
  · exact h
  · exact hown p rel name l0 n' q' hfa hss hn hl0 ho

theorem placeFacts_slot {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
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
    (hrp : _root_.Flock.ZkServer.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd))
    (hown : seen.contains stream = true → ∀ (p : Rep) (rel : _root_.Flock.ZkServer.Frames.Release) (name : String)
      (l0 : _root_.Flock.ZkServer.Level0.Level0) n' q', frameAt σ.R sid i r = some p → sessAt σ.R sid = some rel →
      rel.spec.tables[i]? = some name → _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0 →
      _root_.Flock.ZkServer.Frames.placeOf σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 = .ok (.own, n', q') →
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

theorem slot_masked {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
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

theorem rep_seen {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
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

theorem answered_of_round {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ real s0 spec sts) {seen : List String} {st : State} {pl : Placed} {e : Env}
    {σ σ' : LoopAt} {stream : String} {n i r : Nat} {dom : ByteArray} {v : Claims.Msg}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (htk : M.TookAt st σ.c) (hlt : σ.c < M.τ.length)
    (hr : RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R)
    (hpc : ∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) (ho : ∃ o, e.round = .ok o)
    (hfr : seen.contains stream = false → ∃ m, M.L.fresh M.x (M.hist σ.c) = .ok m)
    (hsc : seen.contains stream = true → ∀ q pc f nd,
      _root_.Flock.ZkServer.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ n = .ok (q, pc, f, nd) →
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

theorem roundFacts_loop {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ real s0 spec sts) {seen : List String} {st : State} {pl : Placed} {e : Env}
    {σ σ' : LoopAt} {stream : String} {n i r : Nat} {dom : ByteArray} {v : Claims.Msg}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hlt : σ.c < M.τ.length)
    (hr : RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R)
    (hpc : ∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) (ho : ∃ o, e.round = .ok o)
    (hans : ∃ m, M.L.answered M.x M.ρ (M.hist σ.c) = .ok m)
    (hpl : ∀ q pc f nd, _root_.Flock.ZkServer.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ n =
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

def OwnSched {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) (stream : String)
    (n : Nat) : Prop :=
  ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt) (i r : Nat) (dom : ByteArray)
    (v : Claims.Msg), slot st pl = some (.round stream n) → LoopFacts sid spec seen st pl σ →
    LoopJ M sid spec sts seen st σ → σ.c < M.τ.length → M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
    RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
    (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → SaltsSrc M sid σ →
    InnerSrc M sid i r σ → StartOk sts sid i r σ → (∃ o, e.round = .ok o) → seen.contains stream = true →
    ∀ (q : Rep) (pc : Place) (f : Bool) (nd : Option String),
      _root_.Flock.ZkServer.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ n = .ok (q, pc, f, nd) →
      ∀ (p : Rep) (rel : _root_.Flock.ZkServer.Frames.Release) (name : String)
        (l0 : _root_.Flock.ZkServer.Level0.Level0) n' q', frameAt σ.R sid i r = some p → sessAt σ.R sid = some rel →
        rel.spec.tables[i]? = some name → _root_.Flock.ZkServer.Level0.ofSession σ.L0 sid name = some l0 →
        _root_.Flock.ZkServer.Frames.placeOf σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 = .ok (.own, n', q') →
        pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
          (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩

def FreshStart {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt)
    (stream : String) (n : Nat) : Prop :=
  ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt) (i r : Nat) (dom : ByteArray)
    (v : Claims.Msg), slot st pl = some (.round stream n) → LoopFacts sid spec seen st pl σ →
    LoopJ M sid spec sts seen st σ → σ.c < M.τ.length → M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
    RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
    (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → SaltsSrc M sid σ →
    InnerSrc M sid i r σ → StartOk sts sid i r σ → (∃ o, e.round = .ok o) → seen.contains stream = false →
    ∃ m, M.L.fresh M.x (M.hist σ.c) = .ok m

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
