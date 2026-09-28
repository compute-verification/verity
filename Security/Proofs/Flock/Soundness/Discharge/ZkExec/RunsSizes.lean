import Proofs.Flock.Soundness.Discharge.ZkExec.RunsStatements
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes

/-!
# A decoded `--zk` proof's lengths

`zk_decode_sizes`: `Zk.decode` reads the pair with `decodeProof`'s lengths (`DecodeZL.ProofSizes`), and `Y`'s length
is a `u64`. The reader is stepped through as `DecodeZL.decodeProof_sizes` steps through `decodeProof`'s, with
`DecodeZL`'s `Post` lemmas; after the pair come level 0's hiding messages (an `Option` tag) and the inner proof, whose
`Y` is a `vecAny` (`yields_vecAny_size`, `post_inner_y`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Reader DecodeZL
open FlockSoundness.Refine (ebind_ok)

theorem yields_vecAny_size {α : Type} {minSize : ℕ} {elem : Reader.M α} {w : String} {a : Array α}
    (h : Yields (vecAny minSize elem w) a) : a.size < 2 ^ 64 := by
  unfold vecAny at h
  obtain ⟨n, -, h⟩ := yields_bind h
  obtain ⟨r, -, h⟩ := yields_bind h
  dsimp only at h
  by_cases hc : n.toNat * minSize > r.data.size - r.pos
  · rw [ite_eq_left hc] at h
    obtain ⟨_, h, -⟩ := yields_bind h
    exact (yields_throw h).elim
  · rw [ite_eq_right hc] at h
    obtain ⟨out, hout, h⟩ := yields_bind h
    rw [yields_pure h]
    obtain ⟨hs, -⟩ := yields_forIn_range (fun _ => elem) n.toNat _ out hout
    have h0 : (Array.mkEmpty n.toNat : Array α).size = 0 := rfl
    rw [hs, h0, Nat.zero_add]
    exact n.toNat_lt

theorem post_inner_y : Post Zk.innerReader (fun i => i.y.size < 2 ^ 64) := by
  unfold Zk.innerReader
  refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
  refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun y hy => ?_
  refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_pure ?_
  exact yields_vecAny_size hy

theorem zk_decode_sizes : ZkDecodeSizes := by
  intro sh bytes zp h
  unfold Zk.decode at h
  obtain ⟨⟨p', r'⟩, hrun, hp⟩ := ebind_ok h
  have hp' : p' = zp := Except.ok.inj hp
  rw [← hp']
  refine (?_ : Post (Zk.reader sh) (fun z => ProofSizes sh z.proof ∧ z.inner.y.size < 2 ^ 64)) p' ⟨_, _, hrun⟩
  unfold Zk.reader
  dsimp only
  refine post_bind_vecBytes fun cap hcap hcapE => ?_
  refine post_bind fun m _ => post_bind fun lir _ => post_bind fun lbs _ => post_bind fun profile _ => ?_
  refine post_bind fun tag _ => ?_
  split
  rotate_left 2
  · exact post_bind_throw
  all_goals
    refine post_bind fun numLanes _ => post_bind fun mh _ => ?_
    refine post_bind_vec fun r1ab hr1ab => post_bind_vec fun r1c hr1c => post_bind_vec fun rounds hrounds => ?_
    refine post_bind fun fa _ => post_bind fun fb _ => post_bind fun fc _ => post_bind fun _ _ => ?_
    refine post_bind_vec fun lr hlr => post_bind_vec fun zp hzp => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind_switches fun rs hrs hrsE => post_bind fun _ _ => ?_
    refine post_bind_vecBytes fun icap _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun n _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind_caps fun rcaps hrc hrcE => ?_
    refine post_bind fun np _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind fun rproofs _ => post_bind_vec fun yr hyr => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind_vec fun msgs hmsgs => post_bind_vec fun qn hqn => post_bind_vec fun ood hood => ?_
    refine post_bind fun _ _ => post_bind_vec fun cbn hcbn => post_bind_vec fun con hcon => ?_
    refine post_bind fun _ _ => ?_
    split
    rotate_left 1
    · exact post_bind_throw
    · exact post_bind_throw
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun inner hinner => ?_
    exact post_pure ⟨⟨hr1ab, hr1c, hrounds, hlr, hzp, hrs, hrsE, hcap, hcapE, hrc, hrcE, hyr, hmsgs, hqn, hood, hcbn,
      hcon⟩, post_inner_y inner hinner⟩

end FlockSoundness.Discharge.ZkExec
