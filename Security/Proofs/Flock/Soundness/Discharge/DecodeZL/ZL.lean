import Proofs.Flock.Soundness.Discharge.DecodeZL.Statements

/-!
# The zerocheck's and lincheck's rounds carry their messages (R11b)

`zl_carries`: from the domain frame through `z_partial`'s round, the rounds of `zcFrames ++ lcFrames`, read by `wZL`,
are `zcMsgs ++ lcMsgs`, cut at `zcShape ++ lcShape`'s coins. The frames split into rounds that each close at their last
frame (`carries_one`), the residual rounds by induction (`carries_pairs`), and the rounds glue (`Carries.append`).
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model

local notation "𝔽" => FlockLevel3.GF128

theorem chunks_single : ∀ {fs : List Frame} {g : Frame}, (∀ f ∈ fs, closes f = false) → closes g = true →
    chunks (fs ++ [g]) = ([fs ++ [g]], [])
  | [], g, _, hg => by simp [chunks, hg]
  | f :: fs, g, hfs, hg => by
      have ih := chunks_single (fun x hx => hfs x (List.mem_cons_of_mem _ hx)) hg
      have hf := hfs f List.mem_cons_self
      rw [List.cons_append, chunks, ite_eq_right (by simp [hf]), ih]

/-- **A round that closes at its last frame** carries the messages its wires read. -/
theorem carries_one {fs : List Frame} {g : Frame} {w : List Wire} {σ : List Exec.Kind} {ms : List Msg}
    (hfs : ∀ f ∈ fs, closes f = false) (hg : closes g = true) (hdec : decRound w ⟨(encs (fs ++ [g])).toArray⟩ = ms)
    (hσ : coinPat σ = List.replicate ms.length false ++ [true]) : Carries [w] σ (fs ++ [g]) ms [] := by
  have hc := chunks_single hfs hg
  have hr : rdChunks [w] (fs ++ [g]) = [ms] := by simp [rdChunks, hc, hdec]
  refine ⟨by rw [hc], by rw [hc]; rfl, by rw [hr]; simp, ?_⟩
  rw [hr, hσ]; simp [groupPat]

theorem carries_nil : Carries [] [] [] [] [] := ⟨rfl, rfl, rfl, rfl⟩

theorem wf_one (v : F128) : (Frame.one v).WF := by simp [Frame.WF, Frame.len]

theorem wf_sq (k : UInt8) {n : ℕ} (h : n < 2 ^ 64) : (Frame.sq k n).WF := h

theorem wf_vec {vs : List F128} (h : vs.length < 2 ^ 64) : (Frame.vec vs).WF := h

/-- The pair a round's two `.one` frames carry. -/
theorem read_ff (a b : F128) :
    readAll [.ff] (valsOf [.one a, .one b, .sq 1 1]) (bytesOf [.one a, .one b, .sq 1 1]) =
      [⟨𝔽 × 𝔽, pairOf (a, b)⟩] := by
  simp [readAll, Wire.ff, valsOf, bytesOf, Frame.vals, Frame.byteStr, pairOf]

theorem decRound_ff (a b : F128) :
    decRound [.ff] ⟨(encs ([.one a, .one b] ++ [.sq 1 1])).toArray⟩ = [⟨𝔽 × 𝔽, pairOf (a, b)⟩] := by
  have hwf : ∀ f ∈ [Frame.one a, .one b] ++ [.sq 1 1], f.WF := by
    simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false]
    rintro f (rfl | rfl | rfl)
    · exact wf_one _
    · exact wf_one _
    · exact wf_sq _ (by norm_num)
  rw [decRound_encs hwf]
  simpa using read_ff a b

/-- **The residual rounds** carry their pairs, one per round. -/
theorem carries_pairs (rounds : Array (F128 × F128)) : ∀ (k a : ℕ),
    Carries (List.replicate k [.ff]) (pairRounds 𝔽 k)
      ((List.range' a k).flatMap fun i => [.one (rounds[i]!).1, .one (rounds[i]!).2, .sq 1 1])
      ((List.range' a k).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩) []
  | 0, a => by simpa [pairRounds] using carries_nil
  | k + 1, a => by
      have h₁ : Carries [[.ff]] [.msg (𝔽 × 𝔽), .coin 𝔽] ([.one (rounds[a]!).1, .one (rounds[a]!).2] ++ [.sq 1 1])
          [⟨𝔽 × 𝔽, pairOf rounds[a]!⟩] [] :=
        carries_one (by simp [closes]) (by simp [closes]) (decRound_ff _ _) (by simp [coinPat, isCoin])
      have := h₁.append (carries_pairs rounds k (a + 1))
      simpa [List.range'_succ, List.replicate_succ, pairRounds] using this

theorem getElem!_append_left' {l₁ l₂ : List F128} {i : ℕ} (h : i < l₁.length) : (l₁ ++ l₂)[i]! = l₁[i]! := by
  simp only [List.getElem!_eq_getElem?_getD, List.getElem?_append_left h]

theorem getElem!_append_right' (l₁ l₂ : List F128) (i : ℕ) : (l₁ ++ l₂)[l₁.length + i]! = l₂[i]! := by
  simp only [List.getElem!_eq_getElem?_getD, List.getElem?_append_right (Nat.le_add_right _ _),
    Nat.add_sub_cancel_left]

/-- `(round1_ab, round1_c)`'s round carries the two vectors. -/
theorem read_vv (a b : Array F128) (ha : a.size = 64) (hb : b.size = 64) :
    readAll [.vv 64] (valsOf [.vec a.toList, .vec b.toList, .sq 1 1]) (bytesOf [.vec a.toList, .vec b.toList, .sq 1 1]) =
      [⟨(Fin 64 → 𝔽) × (Fin 64 → 𝔽), (vecOf 64 a, vecOf 64 b)⟩] := by
  have hl : (a.toList ++ b.toList).take (2 * 64) = a.toList ++ b.toList := List.take_of_length_le (by simp [ha, hb])
  have hp : ((fun i : Fin 64 => fOf (a.toList ++ b.toList)[i.val]!),
      (fun i : Fin 64 => fOf (a.toList ++ b.toList)[64 + i.val]!)) = (vecOf 64 a, vecOf 64 b) := by
    refine Prod.ext (funext fun i => ?_) (funext fun i => ?_)
    · simp only [vecOf, getElem!_append_left' (show i.val < a.toList.length by simp [ha]), Array.getElem!_toList]
    · have := getElem!_append_right' a.toList b.toList i.val
      rw [Array.length_toList, ha] at this
      simp only [vecOf, this, Array.getElem!_toList]
  simp only [readAll, Wire.vv, valsOf, bytesOf, Frame.vals, Frame.byteStr, List.flatMap_cons, List.flatMap_nil,
    List.append_nil, hl]
  rw [hp]

/-- The zerocheck's final pair, read from the lincheck's first round. -/
theorem read_fl (a b : F128) (l : ByteArray) :
    readAll [.ff] (valsOf [.one a, .one b, .label l, .sq 1 1]) (bytesOf [.one a, .one b, .label l, .sq 1 1]) =
      [⟨𝔽 × 𝔽, (fOf a, fOf b)⟩] := by
  simp [readAll, Wire.ff, valsOf, bytesOf, Frame.vals, Frame.byteStr]

/-- `z_partial`'s round carries it. -/
theorem read_v (a : Array F128) (ha : a.size = 64) :
    readAll [.v 64] (valsOf [.vec a.toList, .sq 1 1]) (bytesOf [.vec a.toList, .sq 1 1]) =
      [⟨Fin 64 → 𝔽, vecOf 64 a⟩] := by
  have hl : a.toList.take 64 = a.toList := List.take_of_length_le (by simp [ha])
  simp only [readAll, Wire.v, valsOf, bytesOf, Frame.vals, Frame.byteStr, List.flatMap_cons, List.flatMap_nil,
    List.append_nil, hl, Array.getElem!_toList]
  rfl

theorem zl_carries : ZLCarries := by
  intro m kLog d dig cap zp lp hm h1ab h1c hzP
  have hA : Carries [[]] [.coin (Fin 6 → 𝔽)]
      ([.dom d, .label "flock-r1cs-v0".toUTF8, .bytes dig, .bytes cap, .label "flock-zerocheck-v0".toUTF8] ++ [.sq 2 6])
      [] [] :=
    carries_one (by simp [closes]) (by simp [closes]) (decRound_nil _) (by simp [coinPat, isCoin])
  have hB : Carries [[]] [.coin (Fin (m - 13) → 𝔽)] ([] ++ [.sq 2 (m - 13)]) [] [] :=
    carries_one (by simp) (by simp [closes]; omega) (decRound_nil _) (by simp [coinPat, isCoin])
  have hC : Carries [[.vv 64]] [.msg ((Fin 64 → 𝔽) × (Fin 64 → 𝔽)), .coin 𝔽]
      ([.vec zp.round1ab.toList, .vec zp.round1c.toList] ++ [.sq 1 1])
      [⟨(Fin 64 → 𝔽) × (Fin 64 → 𝔽), (vecOf 64 zp.round1ab, vecOf 64 zp.round1c)⟩] [] := by
    refine carries_one (by simp [closes]) (by simp [closes]) ?_ (by simp [coinPat, isCoin])
    have hwf : ∀ f ∈ [Frame.vec zp.round1ab.toList, .vec zp.round1c.toList] ++ [.sq 1 1], f.WF := by
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false]
      rintro f (rfl | rfl | rfl)
      · exact wf_vec (by simp [h1ab])
      · exact wf_vec (by simp [h1c])
      · exact wf_sq _ (by norm_num)
    rw [decRound_encs hwf]
    simpa using read_vv _ _ h1ab h1c
  have hE : Carries [[.ff]] [.msg (𝔽 × 𝔽), .coin 𝔽]
      ([.one zp.finalA, .one zp.finalB, .label "flock-lincheck-v0".toUTF8] ++ [.sq 1 1])
      [⟨𝔽 × 𝔽, (fOf zp.finalA, fOf zp.finalB)⟩] [] := by
    refine carries_one (by simp [closes]) (by simp [closes]) ?_ (by simp [coinPat, isCoin])
    have hwf : ∀ f ∈ [Frame.one zp.finalA, .one zp.finalB, .label "flock-lincheck-v0".toUTF8] ++ [.sq 1 1], f.WF := by
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false]
      rintro f (rfl | rfl | rfl | rfl)
      · exact wf_one _
      · exact wf_one _
      · show ("flock-lincheck-v0".toUTF8).size < 2 ^ 64
        decide
      · exact wf_sq _ (by norm_num)
    rw [decRound_encs hwf]
    simpa using read_fl zp.finalA zp.finalB "flock-lincheck-v0".toUTF8
  have hF : Carries [[]] [.coin 𝔽] ([] ++ [.sq 1 1]) [] [] :=
    carries_one (by simp) (by simp [closes]) (decRound_nil _) (by simp [coinPat, isCoin])
  have hH : Carries [[.v 64]] [.msg (Fin 64 → 𝔽), .coin 𝔽] ([.vec lp.zPartial.toList] ++ [.sq 1 1])
      [⟨Fin 64 → 𝔽, vecOf 64 lp.zPartial⟩] [] := by
    refine carries_one (by simp [closes]) (by simp [closes]) ?_ (by simp [coinPat, isCoin])
    have hwf : ∀ f ∈ [Frame.vec lp.zPartial.toList] ++ [.sq 1 1], f.WF := by
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false]
      rintro f (rfl | rfl)
      · exact wf_vec (by simp [hzP])
      · exact wf_sq _ (by norm_num)
    rw [decRound_encs hwf]
    simpa using read_v _ hzP
  have := hA.append (hB.append (hC.append ((carries_pairs zp.rounds (m - 6) 0).append (hE.append (hF.append
    ((carries_pairs lp.rounds (kLog - 6) 0).append hH))))))
  simpa [wZL, zcShape, lcShape, zcFrames, lcFrames, zcMsgs, lcMsgs, List.range_eq_range'] using this

end FlockSoundness.Discharge.DecodeZL
