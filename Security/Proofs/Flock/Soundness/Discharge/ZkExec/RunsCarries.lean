import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSeams

/-!
# A `--zk` rep's rounds after round 0 carry its messages

`zk_carries`: the chunk lemma at `wRestZ`, glued from four phases with `CarriesP.append`.

* The zerocheck's tail and the lincheck: DecodeZL's `zl_carries` without round 0, which closes at the first coin and
  carries nothing (`carries_drop_coin`, `zl_tail_carries`).
* The opening: `carries_switch` and `carries_switches` from the ring switches' decoded lengths (`op_carries_sizes`).
* Ligerito with level 0's hiding (`zk_lig_carries`): the non-ZK session's rounds (`carriesP_f`, `carriesP_fold'`,
  `carriesP_mid`), with level 0's pair's sums (`carriesP_ev`), the second `ρ` coin's empty round and the round of `T'`,
  `ybar`, the last lane message and the level-1 cap (`carriesP_tcap`). The `--zk` schedule is fast100's with level 0
  restratified (`zk_schedule_fast100`), so the level loop is `carriesP_levels` at fast100's schedule, whose levels fit
  its bounds (`fast100_small`; level 0 at `--zk` does not).
* The inner proof's four rounds (`carriesP_inner`).

`ligPadShape_pat` is level 0 masked's coin pattern, the rounds' (`isCoin_rec` for the level loop).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

theorem zk_schedule_fast100 {m : ℕ} {sch : Flock.Schedule} {tPad : ℕ} (h : Zk.schedule m = .ok (sch, tPad)) :
    ∃ sch₀ : Flock.Schedule, Flock.fast100 m = some sch₀ ∧ sch.k0 = sch₀.k0 ∧ sch.ext = sch₀.ext ∧
      sch.levels.size = sch₀.levels.size ∧ ∀ i, 1 ≤ i → sch.levels.getD i default = sch₀.levels.getD i default := by
  unfold Zk.schedule at h
  cases hq : Zk.queries0 m with
  | error e => rw [hq] at h; cases h
  | ok q0 =>
    rw [hq] at h
    cases hf : Flock.fast100 m with
    | none => simp only [hf] at h; cases h
    | some sch₀ =>
      simp only [hf] at h
      cases h
      refine ⟨sch₀, rfl, rfl, rfl, by simp, fun i hi => ?_⟩
      simp only [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
      simp [show (0 : ℕ) ≠ i by omega]

/-- Dropping a phase's first round, when it is a coin's alone and carries nothing. -/
theorem carries_drop_coin {w₀ : List DecodeZL.Wire} {W : List (List DecodeZL.Wire)} {σ₀ : Exec.Kind}
    {σ : List Exec.Kind} {c₀ F : List Frame} {ms t : List Msg} (h : Carries (w₀ :: W) (σ₀ :: σ) (c₀ ++ F) ms t)
    (hc : chunks c₀ = ([c₀], [])) (hw : decRound w₀ ⟨(encs c₀).toArray⟩ = []) (hσ : isCoin σ₀ = true) :
    Carries W σ F ms t := by
  obtain ⟨p, l, e, c⟩ := h
  have h0 : (chunks c₀).2 = [] := by rw [hc]
  have hl0 : [w₀].length = (chunks c₀).1.length := by rw [hc]; rfl
  have hr : rdChunks [w₀] c₀ = [[]] := by simp [rdChunks, hc, hw]
  rw [show w₀ :: W = [w₀] ++ W from rfl] at l e c
  rw [chunks_append_of_closed h0] at p l
  rw [rdChunks_append h0 hl0, hr] at e c
  refine ⟨p, ?_, by simpa using e, ?_⟩
  · rw [hc] at l; simpa using l
  · simpa [coinPat, groupPat, hσ] using c

/-- **The zerocheck's tail and the lincheck carry their messages**: `zl_carries` without round 0. -/
theorem zl_tail_carries (m kLog : ℕ) (zp : ZerocheckProof) (lp : LincheckProof) (hm : 14 ≤ m)
    (h1ab : zp.round1ab.size = 64) (h1c : zp.round1c.size = 64) (hzP : lp.zPartial.size = 64) :
    Carries (wZL m kLog).tail ((zcShape 𝔽 m).tail ++ lcShape 𝔽 kLog) (zcTailZ m zp ++ lcFrames kLog lp)
      (zcMsgs m zp ++ lcMsgs kLog lp) [] := by
  have h := zl_carries m kLog ByteArray.empty ByteArray.empty ByteArray.empty zp lp hm h1ab h1c hzP
  have hc₀ : chunks ([.dom ByteArray.empty, .label "flock-r1cs-v0".toUTF8, .bytes ByteArray.empty,
      .bytes ByteArray.empty, .label "flock-zerocheck-v0".toUTF8] ++ [.sq 2 6]) =
      ([[.dom ByteArray.empty, .label "flock-r1cs-v0".toUTF8, .bytes ByteArray.empty, .bytes ByteArray.empty,
        .label "flock-zerocheck-v0".toUTF8] ++ [.sq 2 6]], []) :=
    chunks_single (by simp [closes]) (by simp [closes])
  refine carries_drop_coin (w₀ := []) (σ₀ := .coin (Fin 6 → 𝔽)) ?_ hc₀ (decRound_nil _) rfl
  simpa [wZL, zcShape, zcFrames, zcTailZ] using h

/-- **The opening's rounds carry its messages**, from the ring switches' lengths alone. -/
theorem op_carries_sizes (rs : Array (Array F128)) (n : ℕ) (hn : rs.size = n) (h1 : 1 ≤ n)
    (h : ∀ v ∈ rs, v.size = 128) : Carries (wOP n) (opShape 𝔽 n) (rsFrames n rs) (opMsgs n rs) [] := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  have h₁ := carries_switch (pre := [.label "flock-pcs-open-batch-v0".toUTF8]) (by simp [closes]) rfl
    (by simp only [List.mem_cons, List.mem_nil_iff, or_false]; rintro f rfl
        show ("flock-pcs-open-batch-v0".toUTF8).size < 2 ^ 64
        decide)
    (size_at h (show 0 < rs.size by omega))
  have h₂ := carries_switches rs h k 1 (by omega)
  have h₃ : Carries [[]] [.coin (Fin (k + 1) → 𝔽)] ([] ++ [.sq 2 (k + 1)]) [] [] :=
    carries_one (by simp) (by simp [closes]) (decRound_nil _) (by simp [coinPat, isCoin])
  have := h₁.append (h₂.append h₃)
  simpa [wOP, opShape, switches, rsFrames, opMsgs, List.range_eq_range', List.range'_succ, List.replicate_succ]
    using this

/-! ## Level 0's hiding rounds and the inner proof's -/

theorem ybar_vals (l : List F256) (acc : Array F128) :
    (l.foldl (fun acc v => (acc.push v.c0).push v.c1) acc).toList = acc.toList ++ l.flatMap fun v => [v.c0, v.c1] := by
  induction l generalizing acc with
  | nil => simp
  | cons v l ih => simp [ih]

theorem ybar_length (l : List F256) : (l.flatMap fun v => [v.c0, v.c1]).length = 2 * l.length := by
  induction l with
  | nil => simp
  | cons v l ih => simp only [List.flatMap_cons, List.length_append, ih, List.length_cons, List.length_nil]; omega

theorem ybar_get : ∀ (l : List F256) (i : ℕ), i < l.length →
    (l.flatMap fun v => [v.c0, v.c1])[2 * i]! = l[i]!.c0 ∧ (l.flatMap fun v => [v.c0, v.c1])[2 * i + 1]! = l[i]!.c1
  | [], _, h => by simp at h
  | v :: l, 0, _ => by simp
  | v :: l, i + 1, h => by
      obtain ⟨h₀, h₁⟩ := ybar_get l i (by simp at h; omega)
      have e₀ := getElem!_append_right' [v.c0, v.c1] (l.flatMap fun v => [v.c0, v.c1]) (2 * i)
      have e₁ := getElem!_append_right' [v.c0, v.c1] (l.flatMap fun v => [v.c0, v.c1]) (2 * i + 1)
      simp only [List.length_cons, List.length_nil] at e₀ e₁
      rw [List.flatMap_cons, show 2 * (i + 1) = 0 + 1 + 1 + 2 * i by omega,
        show 0 + 1 + 1 + 2 * i + 1 = 0 + 1 + 1 + (2 * i + 1) by omega, e₀, e₁, h₀, h₁]
      simp

/-- The round of the pair's sums `e`, closed by the first `ρ` coin. -/
theorem carriesP_ev (z : Zk.Level0Msgs) :
    CarriesP [[wEv]] [false, true] ([.vec [z.e0.c0, z.e0.c1], .vec [z.e1.c0, z.e1.c1]] ++ [.sq 2 2]) [evMsg z] [] := by
  have hwf : ∀ f ∈ ([.vec [z.e0.c0, z.e0.c1], .vec [z.e1.c0, z.e1.c1]] ++ [.sq 2 2] : List Frame), f.WF := by
    simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false]
    rintro f (rfl | rfl | rfl)
    · exact wf_vec (by norm_num)
    · exact wf_vec (by norm_num)
    · exact wf_sq _ (by norm_num)
  refine carriesP_one (by simp [closes]) (by simp [closes]) ?_
  rw [decRound_encs hwf]
  simp [readAll, wEv, evMsg, valsOf, bytesOf, Frame.vals, Frame.byteStr]

/-- The round after the second `ρ` coin: `LABEL_CLOSE`, `T'`, `ybar`, the last lane round's message and the level-1
cap, closed by the first out-of-domain point's coins. -/
theorem carriesP_tcap (z : Zk.Level0Msgs) {t : ℕ} (hyb : z.ybar.size = t) (ht : 2 * t < 2 ^ 64)
    (m : F256 × F256) {c : Array ByteArray} (hc : ∀ x ∈ c.toList, x.size = 64) (hcs : c.size < 2 ^ 32)
    {dim : ℕ} (hd : dim ≠ 0) (hdw : dim < 2 ^ 64) :
    CarriesP [[wK, wY t, .kk, .cap]] [false, false, false, false, true]
      (([.label Zk.LABEL_CLOSE.toUTF8, .vec [z.tPrime.c0, z.tPrime.c1],
          .vec (z.ybar.foldl (fun acc v => (acc.push v.c0).push v.c1) #[]).toList] ++ (kFr m ++ [capFr c])) ++
        [.sq 2 dim])
      [tMsg z, ybarMsg t z, kMsg m, ⟨Cap Dig, capDig c⟩] [] := by
  have hL : (z.ybar.foldl (fun acc v => (acc.push v.c0).push v.c1) #[]).toList =
      z.ybar.toList.flatMap fun v => [v.c0, v.c1] := by
    rw [← Array.foldl_toList, ybar_vals]; simp
  rw [hL]
  have hLl : (z.ybar.toList.flatMap fun v => [v.c0, v.c1]).length = 2 * t := by
    rw [← hyb, ← Array.length_toList]
    exact ybar_length _
  have hwf : ∀ f ∈ (([.label Zk.LABEL_CLOSE.toUTF8, .vec [z.tPrime.c0, z.tPrime.c1],
      .vec (z.ybar.toList.flatMap fun v => [v.c0, v.c1])] ++ (kFr m ++ [capFr c])) ++ [.sq 2 dim] : List Frame),
      f.WF := by
    intro f hf
    simp only [List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at hf
    rcases hf with ((rfl | rfl | rfl) | hf | rfl) | rfl
    · show (Zk.LABEL_CLOSE.toUTF8).size < 2 ^ 64
      decide
    · exact wf_vec (by norm_num)
    · exact wf_vec (by rw [hLl]; exact ht)
    · exact kFr_wf m f hf
    · exact wf_capFr hc hcs
    · exact wf_sq _ hdw
  refine carriesP_one ?_ (by simp [closes, hd]) ?_
  · intro f hf
    simp only [List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at hf
    rcases hf with (rfl | rfl | rfl) | hf | rfl
    · rfl
    · rfl
    · rfl
    · exact kFr_closes m f hf
    · rfl
  · rw [decRound_encs hwf]
    have hv : valsOf (([.label Zk.LABEL_CLOSE.toUTF8, .vec [z.tPrime.c0, z.tPrime.c1],
        .vec (z.ybar.toList.flatMap fun v => [v.c0, v.c1])] ++ (kFr m ++ [capFr c])) ++ [.sq 2 dim]) =
        [z.tPrime.c0, z.tPrime.c1] ++ ((z.ybar.toList.flatMap fun v => [v.c0, v.c1]) ++
          [m.1.c0, m.1.c1, m.2.c0, m.2.c1]) := by
      simp [valsOf, kFr, Frame.vals, capFr]
    have hb : bytesOf (([.label Zk.LABEL_CLOSE.toUTF8, .vec [z.tPrime.c0, z.tPrime.c1],
        .vec (z.ybar.toList.flatMap fun v => [v.c0, v.c1])] ++ (kFr m ++ [capFr c])) ++ [.sq 2 dim]) =
        [c.foldl (· ++ ·) ByteArray.empty] := by
      simp [bytesOf, kFr, Frame.byteStr, capFr]
    rw [hv, hb]
    simp only [readAll, wK, wY, Wire.kk, Wire.cap]
    have e₁ : ∀ (a b : F128) (xs : List F128), List.drop 2 ([a, b] ++ xs) = xs := fun _ _ _ => rfl
    have e₂ : ∀ (a b : F128) (xs : List F128), List.take 2 ([a, b] ++ xs) = [a, b] := fun _ _ _ => rfl
    rw [e₁, e₂, List.take_left' hLl, List.drop_left' hLl]
    simp only [tMsg, ybarMsg, kMsg, List.cons.injEq, Sigma.mk.injEq, heq_eq_eq, true_and, and_true]
    refine ⟨rfl, funext fun i => ?_, rfl, capOfRoot_eq hc rfl⟩
    have hi : i.val < z.ybar.toList.length := by simp [hyb]
    obtain ⟨g₀, g₁⟩ := ybar_get z.ybar.toList i.val hi
    rw [g₀, g₁, Array.getElem!_toList]

/-- **The inner proof's rounds carry its messages**: `final_c'`, `[ρ, σ]`, `τ`'s commitment and `Y`, each closed by
the next coin. -/
theorem carriesP_inner (pr : Zk.InnerProof) (finalC : F128) {nY : ℕ} (hy : pr.y.size = nY)
    (hyw : pr.y.size < 2 ^ 64) (htau : pr.tauCommit.size = 64) :
    CarriesP (wInner nY) [false, true, false, true, false, true, false, true] (innerFr pr finalC 10)
      (innerMsgs pr finalC nY) [] := by
  have h₁ : CarriesP [[.f]] [false, true] ([.label Zk.LABEL_INNER.toUTF8, .one finalC] ++ [.sq 1 1])
      [fMsg finalC] [] := by
    have hwf : ∀ f ∈ ([.label Zk.LABEL_INNER.toUTF8, .one finalC] ++ [.sq 1 1] : List Frame), f.WF := by
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false]
      rintro f (rfl | rfl | rfl)
      · show (Zk.LABEL_INNER.toUTF8).size < 2 ^ 64
        decide
      · exact wf_one _
      · exact wf_sq _ (by norm_num)
    refine carriesP_one (by simp [closes]) (by simp [closes]) ?_
    rw [decRound_encs hwf]
    simp [readAll, Wire.f, fMsg, valsOf, bytesOf, Frame.vals, Frame.byteStr]
  have h₂ : CarriesP [[.ff]] [false, true] ([.vec [pr.rho, pr.sigma]] ++ [.sq 2 10])
      [⟨𝔽 × 𝔽, (fOf pr.rho, fOf pr.sigma)⟩] [] := by
    have hwf : ∀ f ∈ ([.vec [pr.rho, pr.sigma]] ++ [.sq 2 10] : List Frame), f.WF := by
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false]
      rintro f (rfl | rfl)
      · exact wf_vec (by norm_num)
      · exact wf_sq _ (by norm_num)
    refine carriesP_one (by simp [closes]) (by simp [closes]) ?_
    rw [decRound_encs hwf]
    simp [readAll, Wire.ff, valsOf, bytesOf, Frame.vals, Frame.byteStr]
  have h₃ : CarriesP [[wDig]] [false, true] ([.bytes pr.tauCommit] ++ [.sq 1 1]) [⟨Dig, toDig pr.tauCommit⟩] [] := by
    have hwf : ∀ f ∈ ([.bytes pr.tauCommit] ++ [.sq 1 1] : List Frame), f.WF := by
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false]
      rintro f (rfl | rfl)
      · exact wf_bytes (by rw [htau]; norm_num)
      · exact wf_sq _ (by norm_num)
    refine carriesP_one (by simp [closes]) (by simp [closes]) ?_
    rw [decRound_encs hwf]
    simp [readAll, wDig, valsOf, bytesOf, Frame.vals, Frame.byteStr]
  have h₄ : CarriesP [[.v nY]] [false, true] ([.vec pr.y.toList] ++ [.sq 2 Zk.Q]) [⟨Fin nY → 𝔽, vecOf nY pr.y⟩] [] := by
    have hwf : ∀ f ∈ ([.vec pr.y.toList] ++ [.sq 2 Zk.Q] : List Frame), f.WF := by
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false]
      rintro f (rfl | rfl)
      · exact wf_vec (by simpa using hyw)
      · exact wf_sq _ (by simp [Zk.Q])
    refine carriesP_one (by simp [closes]) (by simp [closes, Zk.Q]) ?_
    rw [decRound_encs hwf]
    exact readAll_v hy (by simp [valsOf, Frame.vals])
  have := h₁.append (h₂.append (h₃.append h₄))
  simpa [wInner, innerFr, innerMsgs] using this

/-! ## Ligerito with level 0's hiding -/

theorem isCoin_foldZ (F K : Type) : ∀ k, (foldShapeZ F K (k + 1)).map isCoin =
    (List.replicate k [true, false]).flatten ++ [true]
  | 0 => rfl
  | k + 1 => by
      rw [show (foldShapeZ F K (k + 2)).map isCoin = true :: false :: (foldShapeZ F K (k + 1)).map isCoin from rfl,
        isCoin_foldZ F K k]
      simp [List.replicate_succ]

/-- **The coin pattern of level 0 masked**: the start's coin, the first out-of-domain value's round, the `k0` lane
rounds, the pair's sums' round, the second `ρ` coin's, the round of `T'`, `ybar`, the last message and the level-1
cap, then the non-ZK session's level 0 queries and level loop. -/
theorem ligPadShape_pat (sch : Flock.Schedule) (t : ℕ) (hk0 : 1 ≤ sch.k0) (hdim : DimSeam sch) :
    coinPat (ligPadShape 𝔽 𝕂 Dig (schedOf sch) 2 t) = [true] ++ ([false, true] ++
      ((List.replicate (sch.k0 - 1) [false, true]).flatten ++ ([false, true] ++ ([false, true] ++ ([true] ++
        ([false, false, false, false, true] ++ (pMid ++ pLev (sch.levels.size - 1)))))))) := by
  obtain ⟨-, h2, -, -, hl4, -, -⟩ := hdim
  have hlen : 2 ≤ sch.levels.toList.length := by simpa using h2
  obtain ⟨lv0, lv1, rest', hl⟩ : ∃ a b r, sch.levels.toList = a :: b :: r := by
    rcases hx : sch.levels.toList with _ | ⟨a, _ | ⟨b, r⟩⟩
    · rw [hx] at hlen; simp at hlen
    · rw [hx] at hlen; simp at hlen
    · exact ⟨a, b, r, rfl⟩
  have hsz : sch.levels.size = rest'.length + 2 := by rw [← Array.length_toList, hl]; simp
  have hl4' : ∀ x ∈ lv1 :: rest', x.lanes = 4 := by
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
    have e : sch.levels.getD (j + 1) default = (lv1 :: rest')[j] := by
      rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hl]
      simp [List.getElem?_eq_getElem hj]
    rw [← e]
    exact hl4 (j + 1) (by simp at hj; omega) (by omega)
  have hL : (schedOf sch).levels = levelOf lv0 :: (levelOf lv1 :: rest'.map levelOf) := by simp [schedOf, hl]
  have hr := isCoin_rec 𝔽 𝕂 Dig (schedOf sch).ext (levelOf lv1 :: rest'.map levelOf) (by
    intro x hx
    simp only [List.mem_cons, List.mem_map] at hx
    rcases hx with rfl | ⟨y, hy, rfl⟩
    · exact hl4' lv1 (by simp)
    · exact hl4' y (by simp [hy]))
  obtain ⟨k, hk⟩ : ∃ k, sch.k0 = k + 1 := ⟨sch.k0 - 1, by omega⟩
  have hfz := isCoin_foldZ 𝔽 𝕂 k
  have hk' : (schedOf sch).k0 = k + 1 := by simp [schedOf, hk]
  rw [show sch.levels.size - 1 = (levelOf lv1 :: rest'.map levelOf).length by simp [hsz], hk,
    show k + 1 - 1 = k by omega]
  unfold ligPadShape coinPat
  rw [hL]
  simp only [List.map_cons, List.map_append, isCoin, tailShape, oodShape, qbShape, hr, hk', hfz, List.map_nil,
    List.map_replicate]
  simp [shift_pat, pMid, isCoin]

/-- **Ligerito's rounds with level 0's hiding carry its messages**: the non-ZK session's rounds (`lig_carries`'s
pieces), with the pair's sums' round, the second `ρ` coin's and the round of `T'`, `ybar`, the last lane message and the
level-1 cap between the lane rounds and level 0's queries. The level loop is the non-ZK session's at fast100's schedule,
which differs from the `--zk` one only at level 0. -/
theorem zk_lig_carries (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (zp : Zk.ZkProof)
    (T : F128) (hsch : Zk.schedule st.m = .ok (sch, tPad)) (h64 : st.merkle.digestLen = 64)
    (hs : ProofSizes (shapeZK st sch extras) zp.proof) (hyb : zp.level0.ybar.size = tPad) :
    CarriesP (wLIGZ (schedOf sch) tPad) ([true] ++ ([false, true] ++
      ((List.replicate (sch.k0 - 1) [false, true]).flatten ++ ([false, true] ++ ([false, true] ++ ([true] ++
        ([false, false, false, false, true] ++ (pMid ++ pLev (sch.levels.size - 1)))))))))
      (ligFramesZ sch T zp.proof.cap zp.proof.ligerito zp.level0)
      (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0) [] := by
  obtain ⟨h25, h35, hsm, hk0, ht, hdim, -, -, hsmall⟩ := zk_schedule_seams _ _ _ hsch
  obtain ⟨sch₀, hf, ek0, eext, esz, eget⟩ := zk_schedule_fast100 hsch
  obtain ⟨-, -, hsm₀⟩ := fast100_small _ _ hf
  obtain ⟨hn, h2, -, -, -, -, -⟩ := hdim
  have hrc₀ : zp.proof.ligerito.recursiveCaps.size = sch₀.levels.size - 1 := by
    rw [← esz]; simpa [shapeZK] using hs.rcaps
  have hrce₀ : ∀ (i : ℕ) (hi : i < zp.proof.ligerito.recursiveCaps.size),
      zp.proof.ligerito.recursiveCaps[i].size = 2 ^ (sch₀.levels.getD (i + 1) default).cap ∧
        ∀ e ∈ zp.proof.ligerito.recursiveCaps[i], e.size = 64 := by
    intro i hi
    have := hs.rcapsEach i hi
    simp only [shapeZK, h64] at this
    rw [← eget (i + 1) (by omega)]
    exact this
  have hyr₀ : zp.proof.ligerito.finalProof.yr.size = 2 ^ (sch₀.ext + 1) := by
    rw [← eext]; simpa [shapeZK] using hs.yr
  have hlevFr : levelFr sch zp.proof.ligerito = levelFr sch₀ zp.proof.ligerito := by
    funext i
    unfold levelFr
    rw [ek0, esz, eget (i + 1) (by omega), eget (sch₀.levels.size - 1) (by omega)]
  obtain ⟨hc, hcs⟩ := caps_small hsm₀ hrc₀ hrce₀ (i := 0) (by omega)
  obtain ⟨-, hq0, hq512, -⟩ := hsmall _ (getD_mem (show 0 < sch.levels.size by omega) default)
  have hS := carriesP_none (fs := [.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, capFr zp.proof.cap,
    .label Zk.LABEL_OPEN.toUTF8]) (g := .sq 2 sch.n) (by simp [closes, capFr]) (by simp [closes]; omega)
  have hA := carriesP_f zp.proof.ligerito.oodValues[0]! zp.proof.ligerito.claimNonces[0]!
  have hK := carriesP_kk zp.proof.ligerito.messages[0 + (sch.k0 - 1)]! (xs := []) (g := .sq 2 2) (by simp) (by simp)
    (by simp [closes]) (wf_sq _ (by norm_num))
  have hE := carriesP_ev zp.level0
  have hρ := carriesP_none (fs := []) (g := .sq 2 2) (by simp) (by simp [closes])
  have hT := carriesP_tcap zp.level0 hyb (by omega) zp.proof.ligerito.messages[sch.k0]! hc hcs
    (dim := sch.n - sch.k0 + 1) (by omega) (by omega)
  have hM := carriesP_mid zp.proof.ligerito (sch.n - sch.k0 + 1) 1 2 (sch.k0 + 1) (sch.k0 + 2) (sch.k0 + 3) 1 2 3 0 0
    (sch.levels.getD 0 default) (by omega) (by omega)
  have hLv := carriesP_levels sch₀ zp.proof.ligerito hsm₀ hrc₀ hrce₀ hyr₀ (sch₀.levels.size - 1) 0 (by omega)
  rw [← hlevFr, ← eext, ← ek0, ← esz] at hLv
  have := hS.append (hA.append (carriesP_fold' zp.proof.ligerito (sch.k0 - 1) 0
    (hK.append (hE.append (hρ.append (hT.append (hM.append hLv)))))))
  have hw : wLIGZ (schedOf sch) tPad = [[]] ++ ([[.f]] ++ (List.replicate (sch.k0 - 1) [.kk] ++ ([[.kk]] ++
      ([[wEv]] ++ ([[]] ++ ([[wK, wY tPad, .kk, .cap]] ++ (wMid ++ wLev sch.ext (sch.levels.size - 1)))))))) := by
    have : List.replicate sch.k0 [DecodeZL.Wire.kk] = List.replicate (sch.k0 - 1) [.kk] ++ [[.kk]] := by
      rw [← List.replicate_succ']; congr 1; omega
    simp [wLIGZ, schedOf, this]
  rw [hw]
  simpa [ligFramesZ, ligMsgsZ, l0Fr, introFr, List.append_assoc] using this

/-! ## The chunk lemma -/

theorem carriesP_of_carries {W : List (List DecodeZL.Wire)} {σ : List Exec.Kind} {F : List Frame} {ms t : List Msg}
    (h : Carries W σ F ms t) : CarriesP W (coinPat σ) F ms t := h

theorem zk_carries : ZkCarries := by
  intro st sch tPad extras zp T hsch h64 hs hy hyb htau hyl
  obtain ⟨h25, -, -, hk0, -, hdim, -, -, -⟩ := zk_schedule_seams _ _ _ hsch
  have h1 := zl_tail_carries st.m st.kLog zp.proof.zerocheck zp.proof.lincheck (by omega) hs.r1ab hs.r1c hs.zPartial
  have h2 := op_carries_sizes zp.proof.ringSwitches (2 + extras.size) (by simpa [shapeZK] using hs.rs) (by omega)
    hs.rsEach
  have h3 := zk_lig_carries st sch tPad extras zp T hsch h64 hs hyb
  have h4 := carriesP_inner zp.inner zp.proof.zerocheck.finalC hyl hy htau
  have := (carriesP_of_carries h1).append ((carriesP_of_carries h2).append (h3.append h4))
  show CarriesP _ (coinPat _) _ _ _
  have hσ : (shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len).drop 2 =
      ((zcShape 𝔽 st.m).tail ++ lcShape 𝔽 st.kLog) ++ (opShape 𝔽 (2 + extras.size) ++
        (ligPadShape 𝔽 𝕂 Dig (schedOf sch) 2 tPad ++
          ([.msg 𝔽] ++ innerShape 𝔽 Dig 7 (layZ st sch).len Zk.K_PAD Zk.Q))) := by
    simp [shapeZE, zcShape]
  have hpat : coinPat ((shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len).drop 2) =
      coinPat ((zcShape 𝔽 st.m).tail ++ lcShape 𝔽 st.kLog) ++ (coinPat (opShape 𝔽 (2 + extras.size)) ++
        (([true] ++ ([false, true] ++ ((List.replicate (sch.k0 - 1) [false, true]).flatten ++ ([false, true] ++
          ([false, true] ++ ([true] ++ ([false, false, false, false, true] ++
            (pMid ++ pLev (sch.levels.size - 1))))))))) ++ [false, true, false, true, false, true, false, true])) := by
    rw [hσ, ← ligPadShape_pat sch tPad hk0 hdim]
    simp [coinPat, innerShape, isCoin]
  rw [hpat]
  simpa only [wRestZ, restFramesZ, restMsgsZ, List.append_assoc] using this

end FlockSoundness.Discharge.ZkExec
