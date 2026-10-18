import Proofs.Flock.Claims.ZK.Reference

/-!
# A reference session's tape bounds its tables

`RefSession.TapeOK` asks that the tape be read before the last coin with fibers of one size (`eP x`). Every fiber is
then nonempty, so `eP x` is onto every table's base and every pre-final leaf's salt (`eP_surjective`), and one finite
tape carries at most `log₂ |Tape|` independent 1536-bit salts. Every table has its two `τ` leaves, so at every
statement a session whose tape is read so has fewer tables than tapes (`J_lt_card_of_tapeOK`), and a session with
statements of every table count has no such tape (`not_tapeOK`).

The masked scope bounds no table count: `Claims.Flock.InScope true` asks `tables ≤ 1` only when `zk` is false. A
reference session of the whole `--zk` scope needs a tape read per statement, which `Claims.ZKmalR`'s one finite tape
type for every statement does not give.
-/

namespace Claims.ZKMal.RefSession

open FlockSoundness.Discharge

/-- **A tape read with fibers of one size is onto**: every base and every assignment of the pre-final leaves' salts is
read from some tape. -/
theorem eP_surjective (S : RefSession) (hT : S.TapeOK) (x : Msg) : Function.Surjective (S.eP x) := by
  intro p
  obtain ⟨t0⟩ := S.neTape
  have hpos : 0 < Nat.card {t // S.eP x t = S.eP x t0} := Nat.card_pos_iff.mpr ⟨⟨⟨t0, rfl⟩⟩, inferInstance⟩
  rw [← (hT x).1 p (S.eP x t0)] at hpos
  obtain ⟨⟨t, ht⟩⟩ := (Nat.card_pos_iff.mp hpos).1
  exact ⟨t, ht⟩

/-- **A good tape bounds the tables**: at every statement, a session whose tape is read with fibers of one size has
fewer tables than tapes. -/
theorem J_lt_card_of_tapeOK (S : RefSession) (hT : S.TapeOK) (x : Msg) : S.J x < Fintype.card S.Tape := by
  classical
  have hs : Function.Surjective fun t => (S.eP x t).2 := fun y => by
    obtain ⟨t, ht⟩ := S.eP_surjective hT x ((S.eP x (Classical.arbitrary _)).1, y)
    exact ⟨t, congrArg Prod.snd ht⟩
  have h1 := Fintype.card_le_of_surjective _ hs
  let ι : Fin (S.J x) → ZkView.Leaves fun j => (S.C x j).z := fun j => ⟨j, .inr (.inr 0)⟩
  have h2 := Fintype.card_le_of_injective ι fun a b h => congrArg Sigma.fst h
  rw [Fintype.card_fin] at h2
  have hS : 2 ≤ Fintype.card ZkView.Salt := Fintype.one_lt_card
  rw [Fintype.card_fun] at h1
  calc S.J x < 2 ^ S.J x := Nat.lt_two_pow_self
    _ ≤ 2 ^ Fintype.card (ZkView.Leaves fun j => (S.C x j).z) := Nat.pow_le_pow_right (by norm_num) h2
    _ ≤ Fintype.card ZkView.Salt ^ Fintype.card (ZkView.Leaves fun j => (S.C x j).z) := Nat.pow_le_pow_left hS _
    _ ≤ Fintype.card S.Tape := h1

/-- **No good tape at every table count**: a session with statements of every table count has no tape read with
fibers of one size. -/
theorem not_tapeOK (S : RefSession) (hJ : ∀ N, ∃ x, N ≤ S.J x) : ¬ S.TapeOK := fun hT => by
  obtain ⟨x, hx⟩ := hJ (Fintype.card S.Tape)
  exact absurd (S.J_lt_card_of_tapeOK hT x) (not_lt.mpr hx)

end Claims.ZKMal.RefSession
