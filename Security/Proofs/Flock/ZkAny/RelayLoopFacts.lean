module

public import Proofs.Flock.ZkAny.RelayLoopKept
public import Proofs.Flock.ZkAny.RelayLoopCommit
public import Proofs.Flock.ZkAny.RelayLoopSalts
public import Proofs.Flock.ZkAny.RelayMark
public import Proofs.Flock.ZkAny.RelayParams
public import Proofs.Flock.ZkAny.RelaySession

/-!
# What the loop's frames and the model hold at a frame round

At a frame round's slot `slotCase_round` and `loopKept_round` read `RoundFacts`. Here it is from the relay's invariant
at the deployed loop (`LoopFacts`, `LoopJ`) and what the loop read for the round (`LoopRead`'s round clause), the
model's release being the loop's own frames':

* `LoopModel.Deployed`: the model's release is `LeanRelease.ofLoop` on the statement's setup, whose rep streams' names
  are distinct and two a table statement (`helloOf_some`), with a message past its last round.
* `roundPlace_find`: the frame `roundPlace` readies a round on is one the loop's frames hold.
* `PlaceFacts`: what the model holds at the place `roundPlace` readies a round at: its scheduled round, its masking
  state but for `stopped`, and where `v` fails the crossing.
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

/-- **What the model holds at the place a frame round is readied at**, for the frame `q` and the place `pc` and
no-device reason `nd` `roundPlace` returns, the worker's value `v` at the model's history so far, the loop at `σ`: the
place is the model's scheduled round there, `q`'s masking state the model's but for `stopped`, where the round takes a
device round exactly where the frames haven't stopped session `sid`, `v` fails the crossing only where it fails to
pass, and where it takes none in a session the frames haven't stopped, `v` fails the crossing. -/
structure PlaceFacts {W Rr : Type} (M : LoopModel W Rr) (sid : Nat) (v : Claims.Msg) (σ : LoopAt) (q : Rep)
    (pc : Place) (nd : Option String) : Prop where
  sched : pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
    (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩
  st : ∀ m, M.L.answered M.x M.ρ (M.hist σ.c) = .ok m → unstop q.st.m = unstop m
  cross : nd.isSome = σ.R.stopped sid → M.L.cross M.x (M.hist σ.c).length M.w M.ρ (M.hist σ.c) v = false →
    M.L.pass M.x M.w M.ρ (M.hist σ.c) v = false
  inner : nd.isSome = true → σ.R.stopped sid = false →
    M.L.cross M.x (M.hist σ.c).length M.w M.ρ (M.hist σ.c) v = false

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
    rw [if_neg hop, Array.getD_eq_getD_getElem?, hq, Option.getD_some]
  have hkind : M.L.kindAt M.x (M.hist σ.c).length = .masked := by
    have h0 : (M.hist σ.c).length ≠ 0 := by omega
    have h1 : ¬((M.hist σ.c).length = 1 ∨ (M.hist σ.c).length = M.L.last M.x) := by omega
    unfold LeanRelease.kindAt
    rw [if_neg h0, if_neg h1]
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
  refine ⟨hcoins, hkind, by have := hD.next; omega, hlt', Option.isSome_iff_exists.mpr ⟨p, hfa⟩,
    ⟨m, q, f, nd, hm, ?_, ?_, (roundPlace_cfg hfd hrp).trans (hpc p hfa), hP.st m hm, ?_⟩, ?_,
    streamRep_other hF.setup hsent (slot_round_open hsl) hr.rep_eq⟩
  · rw [← hP.sched]
    exact hrp
  · by_cases hns : nd.isSome = σ.R.stopped sid
    · exact .inl ⟨hns, hP.cross hns⟩
    · have hst : σ.R.stopped sid = false := by
        cases hs : σ.R.stopped sid
        · rfl
        · exact absurd ((roundPlace_nd hrp hs).trans hs.symm) hns
      have hnd : nd.isSome = true := by
        cases h : nd.isSome
        · rw [h, hst] at hns
          exact absurd rfl hns
        · rfl
      exact .inr ⟨hnd, hP.inner hnd hst⟩
  · cases hsn : seen.contains stream
    · have hpF := fresh_found (hF.fresh i r stream hmem hsn) hfd
      rw [roundPlace_first hfd hrp, hpF.coins, hpF.st] <;> rfl
    · obtain ⟨p', _, _, _, hf', -, -, -, hS1, -⟩ := hJ.reps i r stream hmem hsn
      obtain ⟨_, _, _, _, _, _, _, _, -, -, -, -, -, -, hI⟩ := hS1 hk.symm
      exact (sameInv_place hI hf' hrp).1
  · rw [hJ.stop, hsent, Bool.true_and]

end Claims.ZKServer.Relay
