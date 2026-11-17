module

public import Proofs.Flock.Claims.ZK.L0PinFlags
public import Proofs.Flock.Claims.ZK.FrameRun
public import Proofs.Flock.Claims.ZK.FrameInner
import all Init.Data.String.Defs
import all Init.Data.Array.Basic
public import Proofs.Flock.Claims.ZK.FrameRanked

/-! A frame that releases the opening's first round took a state the rank stop passes (`rankFails_of_step`). -/

@[expose] public section

namespace Claims.ZKServer.Flags

open _root_.Flock (V F128)
open _root_.Flock.ZkServer.Frame (Cfg Masker LABEL_PCS roundFree INNER_ROUNDS claimPoints check)
open FlockSoundness.Refine (Frame encs)

theorem rankFails_of_label {c : Cfg} {s : Masker} {q : Masker × ByteArray}
    (h : s.label c LABEL_PCS.toUTF8 = .ok q) : LeanRelease.rankFails c s = false := by
  obtain ⟨p1, p2, p3, p4⟩ := Ranked.pcs_other
  cases hon : s.on
  · simp [LeanRelease.rankFails, hon]
  by_cases hpos : s.pos = c.nPiop
  · cases hm : c.mask with
    | none => simp [LeanRelease.rankFails, hm]
    | some ms =>
      cases hcp : claimPoints c.lay s.coins with
      | error e => simp [LeanRelease.rankFails, hm, hcp]
      | ok P =>
        simp [Masker.label, hon, hm, Ranked.isLabel_self, p1, p2, p3, p4, hpos, hcp] at h
        rw [show ∀ f : Array (Array F128) → V (Masker × ByteArray), (Except.ok P >>= f) = f P from
          fun _ => rfl] at h
        split at h
        · rename_i hk
          first
            | simp [LeanRelease.rankFails, hm, hcp, Array.size_append, hk]
            | simp_all [LeanRelease.rankFails, Array.size_append]
        · cases h
  · first
      | simp [LeanRelease.rankFails, hpos]
      | (have hb : (s.pos == c.nPiop) = false := by simpa using hpos
         simp [LeanRelease.rankFails, hb])

theorem rankFails_of_roundFree {c : Cfg} {s : Masker} {r : ByteArray × ℕ} {q : ByteArray × Masker × Array ℕ}
    (hq : PcsRound r) (h : roundFree c s r.1 r.2 = .ok q) : LeanRelease.rankFails c s = false := by
  have hs := Loop.roundFree_ok_inner h
  obtain ⟨fs, k, hd, hp, hk, h0, hn⟩ := hq
  have hm : ∀ f ∈ Frame.label LABEL_PCS.toUTF8 :: fs, midFrame f := by
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · show LABEL_PCS.toUTF8.size < 2 ^ 64
      decide
    · exact (hp f hf).1
  rw [eq_mk_encs hd, roundFree_encs c s _ k r.2 hs hm hk (fun h' => h0 h'.1) hn] at h
  obtain ⟨x, hx, -⟩ := bindV_ok h
  rw [List.foldlM_cons] at hx
  obtain ⟨y, hy, -⟩ := bindV_ok hx
  simp only [stepAt] at hy
  cases hl : s.label c LABEL_PCS.toUTF8 with
  | error e => rw [hl] at hy; cases hy
  | ok q' => exact rankFails_of_label hl

theorem rankFails_stopped (c : Cfg) (s : Masker) (b : Bool) :
    LeanRelease.rankFails c { s with stopped := b } = LeanRelease.rankFails c s :=
  rfl

theorem rankFails_unstop (c : Cfg) {s s' : Masker} (h : unstop s = unstop s') :
    LeanRelease.rankFails c s = LeanRelease.rankFails c s' := by
  have e1 : LeanRelease.rankFails c s = LeanRelease.rankFails c (unstop s) := (rankFails_stopped c s false).symm
  have e2 : LeanRelease.rankFails c s' = LeanRelease.rankFails c (unstop s') := (rankFails_stopped c s' false).symm
  rw [e1, e2, h]

theorem rankFails_of_check {c : Cfg} {s : Masker} {v : ByteArray} {r : ByteArray × ℕ}
    {q : ByteArray × Masker × Option String} (hq : PcsRound r) (h : check c s v r.1 r.2 = .ok q) :
    LeanRelease.rankFails c s = false := by
  rcases q with ⟨out, s', _ | w⟩
  · obtain ⟨q', hr, -⟩ := passes_none (check_none_passes h)
    exact rankFails_of_roundFree hq hr
  · have hr := check_some h
    unfold _root_.Flock.ZkServer.Frame.round at hr
    obtain ⟨q', hq', -⟩ := bindV_ok hr
    rw [← rankFails_stopped c s true]
    exact rankFails_of_roundFree hq hq'

theorem rankFails_of_step {c : Cfg} {st st' : _root_.Flock.ZkServer.Frame.State} {r : ByteArray × ℕ}
    {v out : ByteArray} {vd : String} (hq : PcsRound r)
    (h : _root_.Flock.ZkServer.Frame.step c st (.round r.2 v (.sched r.1)) = (.round out vd, st')) :
    LeanRelease.rankFails c st.m = false := by
  obtain ⟨why, hck⟩ := _root_.Flock.ZkServer.Frame.check_of_step h
  exact rankFails_of_check hq hck

end Claims.ZKServer.Flags
