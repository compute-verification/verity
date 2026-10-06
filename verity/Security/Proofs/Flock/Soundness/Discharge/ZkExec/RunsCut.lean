import Proofs.Flock.Soundness.Discharge.ZkExec.RunsCarries
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSizes

/-!
# A `--zk` rep's messages cut into its rounds

`zk_cut`: round 0 carries the pads root and the later rounds `zk_carries`' groups, so the rep's messages cut along
`shapeZE` are `decZ`'s groups of the rounds.

* `rootOf_encs`: a 64-byte string absorbed just before a round's closing squeeze is the 64 bytes `rootOf` reads.
* `roundGroups_succ`: from round 1 on, `decZ`'s groups are `decOf`'s, which `DecodeZL.roundGroups_eq` gives as the
  chunks' reads, and `DecodeZL.cut_of_carries` cuts `zk_carries` into the same.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128

/-- **Round 0's bytes give back the pads root**: a 64-byte string absorbed just before the closing squeeze sits 80 to
16 bytes before the end of the round's encoding. -/
theorem rootOf_encs (pre : List Frame) {root : ByteArray} (hr : root.size = 64) (k : UInt8) (n : ℕ) :
    rootOf ⟨(encs (pre ++ [.bytes root, .sq k n])).toArray⟩ = root := by
  have hp : (Transcript.padded root).data.toList = root.data.toList := by
    rw [padded_data, hr]; simp
  have he : encs (pre ++ [.bytes root, .sq k n]) = (encs pre ++ (Transcript.header 5 0 64).data.toList) ++
      (root.data.toList ++ (Transcript.header 4 k n).data.toList) := by
    simp [encs, Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay, hp, hr]
  have hH := Frame.header_length 4 k n
  have hR : root.data.toList.length = 64 := by simpa using hr
  unfold rootOf
  rw [he]
  generalize encs pre ++ (Transcript.header 5 0 64).data.toList = A
  apply ByteArray.ext
  rw [ByteArray.data_extract]
  apply Array.toList_inj.1
  rw [Array.toList_extract]
  simp only [ByteArray.size, List.size_toArray, List.length_append, hH, hR, List.toList_toArray]
  rw [List.extract_eq_take_drop, show A.length + (64 + 16) - 80 = A.length by omega, List.drop_left,
    show A.length + (64 + 16) - 16 - A.length = root.data.toList.length by omega, List.take_left]

theorem roundGroups_succ (dec : ℕ → ByteArray → List Msg) : ∀ (j : ℕ) (rs : List Round),
    PlaysCoin.roundGroups dec (j + 1) rs = PlaysCoin.roundGroups (fun i => dec (i + 1)) j rs
  | _, [] => rfl
  | j, rd :: rs => by simp [PlaysCoin.roundGroups, roundGroups_succ dec (j + 1) rs]

theorem zk_cut : ZkCut := by
  intro st sch tPad extras bytes zp T dom rds hsch h64 hdec hroot hyb htau hyl hlen hret
  obtain ⟨hs, hy⟩ := zk_decode_sizes _ _ _ hdec
  have hC := zk_carries st sch tPad extras zp T hsch h64 hs hy hyb htau hyl
  have hc₀ := chunks_single (fs := [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes st.digest,
      .bytes (zp.proof.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8,
      .label Zk.LABEL_PADS.toUTF8, .bytes zp.inner.root]) (g := .sq 2 6) (by simp [closes]) (by simp [closes])
  have hF : repFramesZ st sch dom (2 + extras.size) zp T =
      ([.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes st.digest,
        .bytes (zp.proof.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8,
        .label Zk.LABEL_PADS.toUTF8, .bytes zp.inner.root] ++ [.sq 2 6]) ++
        restFramesZ st sch (2 + extras.size) zp T := rfl
  rw [hF, chunks_append_of_closed (by rw [hc₀]), hc₀] at hlen hret
  simp only [List.singleton_append] at hlen hret
  have hW := hC.2.1
  have hlσ := carries_length hC
  have hr1 : rds.toList.length = (chunks (restFramesZ st sch (2 + extras.size) zp T)).1.length + 1 := by
    rw [Array.length_toList, ← hlen, List.length_cons]
  obtain ⟨rd₀, rds', hrds⟩ : ∃ rd₀ rds', rds.toList = rd₀ :: rds' := by
    rcases hx : rds.toList with _ | ⟨a, r⟩
    · rw [hx] at hr1; simp at hr1
    · exact ⟨a, r, rfl⟩
  have hsz : rds'.length = (chunks (restFramesZ st sch (2 + extras.size) zp T)).1.length := by
    rw [hrds, List.length_cons] at hr1; omega
  have hσ : shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len =
      .msg Dig :: .coin (Fin 6 → 𝔽) ::
        (shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len).drop 2 := rfl
  have hlS : (shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len).length =
      ((shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len).drop 2).length + 2 := by
    rw [hσ]; simp
  have hlM : (repMsgsZ st sch (2 + extras.size) tPad zp).length =
      (restMsgsZ st sch (2 + extras.size) tPad zp).length + 1 := by simp [repMsgsZ]
  rw [List.take_of_length_le (by omega)]
  rw [hσ, repMsgsZ, hrds]
  have h₀ := hret 0 (by rw [← Array.length_toList]; omega)
  rw [show rds[0]! = rd₀ by
    rw [getElem!_pos rds 0 (by rw [← Array.length_toList]; omega), ← Array.getElem_toList]; simp [hrds]] at h₀
  simp only [List.getD_cons_zero] at h₀
  have hB := bytes_of_retains h₀
  have hroot' : rootOf ⟨(encs ([.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes st.digest,
      .bytes (zp.proof.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8,
      .label Zk.LABEL_PADS.toUTF8, .bytes zp.inner.root] ++ [.sq 2 6])).toArray⟩ = zp.inner.root :=
    rootOf_encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes st.digest,
      .bytes (zp.proof.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8,
      .label Zk.LABEL_PADS.toUTF8] hroot 2 6
  have hrest := roundGroups_eq (wRestZ st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len)
    (restFramesZ st sch (2 + extras.size) zp T) rds'.toArray hW (by simp [hsz]) (by
      intro k hk
      have hk' : k + 1 < rds.size := by rw [← Array.length_toList, hrds]; simpa using hk
      have := hret (k + 1) hk'
      rw [show rds[k + 1]! = rds'.toArray[k]! by
        rw [getElem!_pos rds (k + 1) hk', getElem!_pos rds'.toArray k hk,
          ← Array.getElem_toList]
        simp [hrds]] at this
      simpa using this)
  rw [List.toList_toArray] at hrest
  show PlaysCoin.consHead _ ([] :: PlaysCoin.cut _ _) = (decZ _ 0 (rd₀.bytes.getD ByteArray.empty) ::
    PlaysCoin.roundGroups _ (0 + 1) rds') ++ [[]]
  rw [cut_of_carries hC, hB, roundGroups_succ, show (fun i => decZ (wRestZ st.m st.kLog (2 + extras.size)
    (schedOf sch) tPad (layZ st sch).len) (i + 1)) = decOf (wRestZ st.m st.kLog (2 + extras.size) (schedOf sch) tPad
      (layZ st sch).len) from rfl, hrest]
  simp only [decZ]
  rw [hroot']
  rfl

end FlockSoundness.Discharge.ZkExec
