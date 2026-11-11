module

public import Proofs.Flock.Claims.ZK.FinishBridge
public import Proofs.Flock.Claims.ZK.FinishLive
public import Proofs.Flock.Claims.ZK.FrameFree
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.ZK.FinishSrc
import Proofs.Flock.Claims.ZK.FinishTau
import Proofs.Flock.SessionLoop.FinHmkCut

/-!
# The algebraic phase's scheduled rounds are the PIOP's bodies at zero values

`Skeleton.rounds` records `Finish.repPiop`'s rounds at zero values, each answered by the next coins of its own
(`recAsks`). The cut session runs the same zerocheck and lincheck (`zerocheckFw_eq`, `lincheckFw_eq`), logging each round
and answering it by its index (`asksCut`), and its log is the PIOP's bodies whatever its coins (`zerocheckCut_bodies`,
`lincheckCut_bodies`, `piopBodies_eq`). Here one is held to the other:

* **The state map** (`skelSim fin`): a recording run from rounds `rs` that returns at rounds `rs'` only adds to them, and
  where `rs'` is a prefix of the final recording `fin`, the session's run from a log of `rs`'s bytes returns the same with
  a log of `rs'`'s, its round `j` answered by the coins `recAsks` answered `fin`'s round `j` (`skelCoin fin`).
* **The sends** (`asksRel_skel`): `recAsks` and `asksCut` at those coins send each round alike.
* **`rounds_piop`**: where `Skeleton.rounds` returns `alg`, it holds the PIOP's `piopNR` rounds, and round `jj ≥ 1` is the
  PIOP's frames at zero values (`piopFr`) closed by its squeeze, with its coin count `piopN`.
-/

@[expose] public section

namespace Claims.ZKServer.Sched

open _root_.Flock (V F128 Transcript TMW)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Skeleton (RecM recAsks)
open _root_.Flock.Firewall.Finish (Asks AsksRel repPiop)
open FlockSoundness.Refine (Frame encs)
open FlockComplete (PS asksCut)

/-- A round's coin count, read off its squeeze header as `recAsks` reads it. -/
def cntOf (c : ByteArray) : ℕ := (_root_.Flock.u64OfLE c (c.size - 8)).toNat

/-- The coins flipped over the recorded rounds `rs`. -/
def cntSum (rs : List (ByteArray × ℕ)) : ℕ := (rs.map Prod.snd).sum

/-- A recording as `recAsks` keeps it: each round with its coin count, and their total. -/
abbrev RecWF (s : Array (ByteArray × ℕ) × ℕ) : Prop := s.2 = cntSum s.1.toList ∧ ∀ p ∈ s.1.toList, p.2 = cntOf p.1

/-- The session's log of the recorded rounds `rs`. -/
def recLog (rs : Array (ByteArray × ℕ)) : List Claims.Msg := rs.toList.map fun p => Claims.toMsg p.1

/-- The session's coins at the final recording `fin`: round `j`'s are those `recAsks` answered `fin`'s round `j`. -/
def skelCoin (fin : Array (ByteArray × ℕ)) (j i : ℕ) : F128 :=
  _root_.Flock.Firewall.Skeleton.coin (cntSum (fin.toList.take j) + i + 1)

/-- **The recording's rounds as the session's**: a recording run from `s` that returns `a` at `s'` only adds rounds, and
where `s'`'s rounds are a prefix of `fin`, the session's run from a log of `s`'s rounds returns `a` with a log of
`s'`'s. -/
def skelSim (fin : Array (ByteArray × ℕ)) : _root_.Flock.With.Sim RecM PS where
  R x y := ∀ s (l : Array Claims.Msg), RecWF s → l.toList = recLog s.1 → ∀ a s', x.run s = .ok (a, s') →
    s.1.toList <+: s'.1.toList ∧ RecWF s' ∧
      (s'.1.toList <+: fin.toList → (y.go l).1 = .ok a ∧ (y.go l).2.toList = recLog s'.1)
  pure _ _ _ hs hl _ _ h := by
    cases h
    exact ⟨List.prefix_refl _, hs, fun _ => ⟨rfl, hl⟩⟩
  bind {α β x y f g} hxy hfg s l hs hl b s'' h := by
    rw [StateT.run_bind] at h
    rcases hx : x.run s with _ | ⟨a, s1⟩
    · rw [hx] at h; cases h
    · rw [hx] at h
      obtain ⟨hp1, hw1, hy1⟩ := hxy s l hs hl a s1 hx
      obtain ⟨hp2, hw2, -⟩ := hfg a s1 ⟨recLog s1.1⟩ hw1 rfl b s'' h
      refine ⟨hp1.trans hp2, hw2, fun hf => ?_⟩
      obtain ⟨hy, hyl⟩ := hy1 (hp2.trans hf)
      obtain ⟨-, -, hg⟩ := hfg a s1 (y.go l).2 hw1 hyl b s'' h
      rw [FlockComplete.go_bind]
      rcases e : y.go l with ⟨r, l1⟩
      rw [e] at hy hg
      dsimp only at hy hg
      subst hy
      exact hg hf
  throw _ _ _ _ _ _ _ h := by cases h
  lift v _ _ hs hl _ _ h := by
    cases v with
    | error e => cases h
    | ok a0 =>
      cases h
      exact ⟨List.prefix_refl _, hs, fun _ => ⟨rfl, hl⟩⟩

theorem recAsks_round_run (c : ByteArray) (rs : Array (ByteArray × ℕ)) (k : ℕ) :
    (recAsks.round c).run (rs, k) =
      .ok (fun i => _root_.Flock.Firewall.Skeleton.coin (k + i + 1), rs.push (c, cntOf c), k + cntOf c) :=
  rfl

/-- **The recording's sends are the session's** at the coins `recAsks` answered the final recording `fin`, wherever the
session's cut is past `fin`'s last round. -/
theorem asksRel_skel (fin : Array (ByteArray × ℕ)) (n : ℕ) (hn : fin.size ≤ n) :
    AsksRel (skelSim fin) recAsks.toC (asksCut (skelCoin fin) n) where
  link _ _ _ _ _ _ _ _ h := by cases h
  round c := by
    rintro ⟨rs, k⟩ l ⟨hk, hc⟩ hl a s' h
    change (recAsks.round c).run (rs, k) = _ at h
    rw [recAsks_round_run] at h
    cases h
    dsimp only at hk hc hl ⊢
    refine ⟨?_, ⟨?_, ?_⟩, fun hf => ?_⟩
    · rw [Array.toList_push]
      exact List.prefix_append _ _
    · simp [cntSum, hk]
    · intro p hp
      rw [Array.toList_push, List.mem_append, List.mem_singleton] at hp
      rcases hp with hp | rfl
      · exact hc p hp
      · rfl
    · have hpre : rs.toList <+: fin.toList := (List.prefix_append _ _).trans (by rwa [Array.toList_push] at hf)
      have hsz : l.size = rs.size := by
        have e := congrArg List.length hl
        simpa [recLog] using e
      have hlt : rs.size < fin.size := by
        have e := hf.length_le
        simp only [Array.toList_push, List.length_append, Array.length_toList, List.length_singleton] at e
        omega
      rw [_root_.Flock.Firewall.Finish.asksCut_round_fw, if_neg (show ¬ n ≤ l.size by omega)]
      refine ⟨?_, ?_⟩
      · show Except.ok (skelCoin fin l.size) = Except.ok fun i => _root_.Flock.Firewall.Skeleton.coin (k + i + 1)
        congr 1
        funext i
        have ht : fin.toList.take rs.size = rs.toList := by
          have := List.prefix_iff_eq_take.1 hpre
          rw [Array.length_toList] at this
          exact this.symm
        simp only [skelCoin, hsz, ht, ← hk]
      · simp [recLog, hl, Array.toList_push]

/-- The cut's sends as the ZK server's `Asks`. -/
def skelIO (fin : Array (ByteArray × ℕ)) (n : ℕ) : Asks PS :=
  ⟨(asksCut (skelCoin fin) n).link, (asksCut (skelCoin fin) n).round⟩

/-- A round's coin count where its bytes close with a squeeze. -/
theorem cntOf_sq (B : List UInt8) (k : UInt8) {n : ℕ} (hn : n < 2 ^ 64) :
    cntOf (ByteArray.mk (B ++ (Frame.sq k n).enc).toArray) = n := by
  have h : (ByteArray.mk B.toArray ++ header 4 k n).data.toList = B ++ (Frame.sq k n).enc := by
    simp [Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]
  have e : ByteArray.mk B.toArray ++ header 4 k n = ByteArray.mk (B ++ (Frame.sq k n).enc).toArray := by
    rw [← h]
  rw [cntOf, ← e]
  exact count_header _ k n hn

/-- The pads' root the algebraic phase's schedule is recorded on. -/
def root0 : ByteArray := ByteArray.mk (Array.replicate 64 0)

/-- The recording `Skeleton.rounds` runs, from a rep's start. -/
def skelProg (st : _root_.Flock.Setup) (k0 : ℕ) (cap : ByteArray) : TMW RecM Unit :=
  (repPiop recAsks st (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) cap root0 #[] fun _ => 0) >>= fun _ => pure ()

/-- **A schedule that returns is its recording's rounds**, round `0` replaced, from a rep's start. -/
theorem rounds_run {st : _root_.Flock.Setup} {k0 : ℕ} {cap dom : ByteArray} {alg : Array (ByteArray × ℕ)}
    (h : _root_.Flock.Firewall.Skeleton.rounds st k0 cap dom = .ok alg) :
    ∃ t' rs k, (skelProg st k0 cap).run (Flock.Transcript.init #[] dom) (#[], 0) = .ok (((), t'), rs, k) ∧
      alg.size = rs.size ∧ ∀ jj, jj ≠ 0 → alg[jj]? = rs[jj]? := by
  unfold _root_.Flock.Firewall.Skeleton.rounds at h
  dsimp only at h
  obtain ⟨⟨u, rs, k⟩, hr, h⟩ := except_bind_ok h
  rw [tmw_run'_run] at hr
  obtain ⟨⟨⟨u', t'⟩, s'⟩, hx, hp⟩ := except_bind_ok hr
  cases hp
  refine ⟨t', rs, k, hx, ?_⟩
  dsimp only at h
  split at h
  · split at h
    · cases h
    · cases h
      simp only [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds, true_and]
      intro jj hj
      rw [Array.getElem?_setIfInBounds_ne (Ne.symm hj)]
  · cases h

/-- **The algebraic phase's scheduled rounds are the PIOP's**: where `Skeleton.rounds` returns `alg`, it holds the PIOP's
`piopNR` rounds, and round `jj ≥ 1` is the PIOP's frames at zero values (`piopFr`, at the binding of `st.digest`, `cap`
and the zero root, which only round `0` reads) closed by its squeeze, with its coin count `piopN`. -/
theorem rounds_piop {st : _root_.Flock.Setup} {k0 : ℕ} {cap dom : ByteArray} {alg : Array (ByteArray × ℕ)}
    (hm : 14 ≤ st.m) (hm' : st.m < 2 ^ 64)
    (h : _root_.Flock.Firewall.Skeleton.rounds st k0 cap dom = .ok alg) :
    alg.size = piopNR st.m st.fold.kLog ∧ ∀ jj, jj ≠ 0 → jj < piopNR st.m st.fold.kLog →
      alg[jj]? = some (ByteArray.mk (encs (piopFr st.m st.fold.kLog st.digest cap root0
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) (fun _ => 0) jj) ++
        (Frame.sq (piopK jj) (piopN st.m jj)).enc).toArray, piopN st.m jj) := by
  obtain ⟨t', rs, k, hr, hsz, hset⟩ := rounds_run h
  obtain ⟨n, hn⟩ : ∃ n, n = rs.size + piopNR st.m st.fold.kLog := ⟨_, rfl⟩
  have hS := _root_.Flock.Firewall.Finish.repPiop_rel (skelSim rs) (io := recAsks) (io' := skelIO rs n)
    (asksRel_skel rs n (by omega)) st (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) cap root0 #[] (fun _ => 0)
  have hS' := ((skelSim rs).st Flock.Transcript).bind hS fun _ => ((skelSim rs).st Flock.Transcript).pure ()
  obtain ⟨-, hwf, hy⟩ := hS' (Flock.Transcript.init #[] dom) (#[], 0) #[] ⟨rfl, by simp⟩ rfl _ _ hr
  obtain ⟨hy1, hyl⟩ := hy (List.prefix_refl _)
  have hY : ((((repPiop (skelIO rs n) st (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) cap root0 #[]
      fun _ => 0) >>= fun _ => pure ()) : TMW PS Unit).run (Flock.Transcript.init #[] dom)).go #[] =
      (.ok ((), t'), ⟨recLog rs⟩) :=
    Prod.ext hy1 (Array.ext' hyl)
  obtain ⟨res, t₂, L₂, hrep, hpure⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hY
  have hL₂ : L₂ = ⟨recLog rs⟩ := congrArg Prod.snd hpure
  unfold repPiop at hrep
  obtain ⟨⟨zp, zc⟩, t₁, L₁, hz, hrest⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hrep
  obtain ⟨res₁, t₁', L₁', e₁, hL₁, hP₁⟩ :=
    zerocheckCut_bodies (skelCoin rs) n hm st.digest cap root0 (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0)
      (fun _ => 0) (Flock.Transcript.init #[] dom) #[] (by unfold piopNR at hn; simp only [Array.size_empty]; omega)
  change ((FlockComplete.zerocheckCut (asksCut (skelCoin rs) n) st.m st.digest cap root0
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) fun _ => 0).run (Flock.Transcript.init #[] dom)).go #[] = _ at hz
  rw [hz] at e₁
  simp only [Prod.mk.injEq, Except.ok.injEq] at e₁
  obtain ⟨⟨-, rfl⟩, rfl⟩ := e₁
  dsimp only at hrest
  obtain ⟨⟨lp, lc⟩, t₂', L₂', hl, hrest₂⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hrest
  change ((FlockComplete.lincheckCut (asksCut (skelCoin rs) n) st.fold st.m
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) (fun _ => 0) zc).run t₁).go L₁ = _ at hl
  have hs₁ : L₁.size = 3 + (st.m - 6) := by
    have e := congrArg List.length hL₁
    simp only [Array.length_toList, List.length_append, zcBodies, List.length_cons, List.length_map,
      List.length_range, List.length_nil] at e
    omega
  obtain ⟨res₂', t₂'', L₂'', e₂, hL₂', -⟩ := lincheckCut_bodies (skelCoin rs) n st.fold st.m
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) (fun _ => 0) zc t₁ L₁
    (by unfold piopNR at hn; rw [hs₁]; omega) (FlockComplete.RankAt.lincheckCut_fold (skelCoin rs) n hl)
  rw [hl] at e₂
  simp only [Prod.mk.injEq, Except.ok.injEq] at e₂
  obtain ⟨⟨-, -⟩, rfl⟩ := e₂
  have hL₂₂ : L₂' = L₂ := congrArg Prod.snd hrest₂
  have hlog : recLog rs = (List.range (piopNR st.m st.fold.kLog)).map
      (piopBody (Flock.Transcript.init #[] dom).pending.data.toList st.m st.fold.kLog st.digest cap root0
        (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) fun _ => 0) := by
    have e : recLog rs = L₂'.toList := by rw [hL₂₂, hL₂]
    rw [e, hL₂', hL₁, hP₁, ← piopBodies_eq]
    simp only [List.nil_append]
  have hrs : rs.size = piopNR st.m st.fold.kLog := by
    have e := congrArg List.length hlog
    simpa [recLog] using e
  refine ⟨hsz.trans hrs, fun jj hj0 hj => ?_⟩
  rw [hset jj hj0]
  have hjr : jj < rs.size := by omega
  have hb : (rs[jj]).1 = ByteArray.mk (encs (piopFr st.m st.fold.kLog st.digest cap root0
      (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) (fun _ => 0) jj) ++
      (Frame.sq (piopK jj) (piopN st.m jj)).enc).toArray := by
    have e := congrArg (·[jj]?) hlog
    simp only [recLog, List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hjr,
      List.getElem?_range hj, Option.map_some, Option.some.injEq] at e
    have hd : (rs[jj]).1.data.toList = encs (piopFr st.m st.fold.kLog st.digest cap root0
        (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) (fun _ => 0) jj) ++
        (Frame.sq (piopK jj) (piopN st.m jj)).enc := by
      rw [show (rs[jj]).1.data.toList = Claims.toMsg (rs[jj]).1 from rfl, e]
      simp only [piopBody, if_neg hj0, List.nil_append]
    rw [← hd]
  have hc : (rs[jj]).2 = piopN st.m jj := by
    rw [hwf.2 _ (Array.getElem_mem_toList hjr), hb]
    refine cntOf_sq _ _ ?_
    unfold piopN
    split_ifs <;> omega
  rw [Array.getElem?_eq_getElem hjr]
  exact congrArg some (Prod.ext hb hc)

end Claims.ZKServer.Sched
