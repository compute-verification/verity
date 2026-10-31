import Proofs.Flock.Soundness.Discharge.DecodeZL.ZL
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Decodes

/-!
# A rep's run, from its rounds' frames (R11b)

plays-coin's `DecodesOnOfRuns` asks each rep for a `RepRun`: a run on its rounds' coins whose messages, cut at the spine's
coins, are the decoder's groups of the rounds. This file gives it from the executable's side:

* `coinPat_eq`: `Carries`' coin pattern is plays-coin's, so `cutOfPat` cuts a `Carries` back into its rounds.
* `roundGroups_eq`: on rounds that retained the encodings of `F`'s chunks, the decoder `decOf W`'s groups are
  `rdChunks W F`.
* `repRun_at`: at a table whose statement, schedule and link layout are `stmtOf`'s, `fast100`'s and `PT_LOCAL`, the rep
  run `accepts_verdict_hm96` gives, on rounds that retained `repFrames`' chunks (`accepts_rep_rounds`), is a `RepRun`
  for the decoder `wZL ++ (wOP ++ wLIG)`, given the opening's and Ligerito's chunk lemmas.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

theorem coinPat_eq : ∀ σ : List Exec.Kind, coinPat σ = PlaysCoin.coinPat σ
  | [] => rfl
  | .msg _ :: σ => congrArg (List.cons false) (coinPat_eq σ)
  | .coin _ :: σ => congrArg (List.cons true) (coinPat_eq σ)

theorem groupPat_eq (ns : List ℕ) (t : ℕ) : groupPat ns t = PlaysCoin.groupPat ns t := rfl

theorem groupPat_length : ∀ (ns : List ℕ) (t : ℕ), (groupPat ns t).length = ns.sum + ns.length + t
  | [], t => by simp [groupPat]
  | k :: ns, t => by
      have := groupPat_length ns t
      simp only [groupPat, List.flatMap_cons, List.length_append, List.length_replicate, List.sum_cons,
        List.length_cons, List.length_nil] at this ⊢
      omega

/-- **A `Carries` cuts back into its rounds**: the messages, laid along the spine and cut at its coins, are the rounds'
messages then the tail. -/
theorem cut_of_carries {W : List (List Wire)} {σ : List Exec.Kind} {F : List Frame} {ms t : List Msg}
    (h : Carries W σ F ms t) : PlaysCoin.cut σ ms = rdChunks W F ++ [t] := by
  obtain ⟨-, -, rfl, hpat⟩ := h
  exact PlaysCoin.cutOfPat σ _ t (by rw [← coinPat_eq, hpat]; rfl)

theorem rdChunks_length (W : List (List Wire)) (F : List Frame) :
    (rdChunks W F).length = min W.length (chunks F).1.length := by
  simp [rdChunks]

/-- A `Carries`' spine is as long as its messages and rounds. -/
theorem carries_length {W : List (List Wire)} {σ : List Exec.Kind} {F : List Frame} {ms t : List Msg}
    (h : Carries W σ F ms t) : σ.length = ms.length + (chunks F).1.length := by
  obtain ⟨-, hW, rfl, hpat⟩ := h
  have e := congrArg List.length hpat
  rw [coinPat, List.length_map, groupPat_length, List.length_map, rdChunks_length, hW, Nat.min_self] at e
  rw [e, List.length_append, List.length_flatten]
  omega

theorem roundGroups_length (dec : ℕ → ByteArray → List Msg) :
    ∀ (j : ℕ) (rs : List Round), (PlaysCoin.roundGroups dec j rs).length = rs.length
  | _, [] => rfl
  | j, _ :: rs => by simp [PlaysCoin.roundGroups, roundGroups_length dec (j + 1) rs]

theorem roundGroups_getElem (dec : ℕ → ByteArray → List Msg) :
    ∀ (j : ℕ) (rs : List Round) (k : ℕ) (hk : k < rs.length),
      (PlaysCoin.roundGroups dec j rs)[k]'(by rw [roundGroups_length]; exact hk) =
        dec (j + k) ((rs[k]).bytes.getD ByteArray.empty)
  | _, [], _, hk => absurd hk (Nat.not_lt_zero _)
  | j, rd :: rs, 0, _ => by simp [PlaysCoin.roundGroups]
  | j, rd :: rs, k + 1, hk => by
      simp only [PlaysCoin.roundGroups, List.getElem_cons_succ]
      rw [roundGroups_getElem dec (j + 1) rs k (by simpa using hk), show j + (k + 1) = j + 1 + k by omega]

theorem bytes_of_retains {rd : Round} {L : List UInt8} (h : Retains rd L) :
    rd.bytes.getD ByteArray.empty = ⟨L.toArray⟩ := by
  unfold Retains at h
  cases hb : rd.bytes with
  | none => rw [hb] at h; cases h
  | some B =>
      rw [hb, Option.map_some, Option.some.injEq] at h
      subst h
      simp

/-- **The decoder's groups are the chunks' reads**: on rounds that retained the encodings of `F`'s chunks, one round
per chunk and one wire list per chunk, the groups `decOf W` reads off the rounds are `rdChunks W F`. -/
theorem roundGroups_eq (W : List (List Wire)) (F : List Frame) (rds : Array Round)
    (hW : W.length = (chunks F).1.length) (hl : (chunks F).1.length = rds.size)
    (hret : ∀ k < rds.size, Retains rds[k]! (encs ((chunks F).1.getD k []))) :
    PlaysCoin.roundGroups (decOf W) 0 rds.toList = rdChunks W F := by
  apply List.ext_getElem
  · rw [roundGroups_length, rdChunks_length, hW, Nat.min_self, hl, Array.length_toList]
  · intro k h1 h2
    have hk : k < rds.size := by rwa [roundGroups_length, Array.length_toList] at h1
    have hkc : k < (chunks F).1.length := by rw [hl]; exact hk
    have hkW : k < W.length := by rw [hW]; exact hkc
    rw [roundGroups_getElem _ _ _ _ (by rwa [Array.length_toList]), Nat.zero_add]
    have hr := hret k hk
    rw [getElem!_pos rds k hk, List.getD_eq_getElem _ _ hkc] at hr
    rw [Array.getElem_toList, bytes_of_retains hr]
    simp only [rdChunks, List.getElem_zipWith, decOf, List.getD_eq_getElem _ _ hkW]

theorem extraClaims_length (S : Model.Statement) (pl mPts : ℕ) (pts : Fin 2 → Fin mPts → 𝔽) :
    (Model.extraClaims S pl mPts pts).length = 2 * S.regions.length := by
  unfold Model.extraClaims
  generalize S.regions = rs
  induction rs with
  | nil => rfl
  | cons R rs ih => rw [List.flatMap_cons, List.length_append, ih]; simp; omega

/-- **A rep's run is a `RepRun`** for the decoder `wZL ++ (wOP ++ wLIG)`: at a table `T` whose statement is `stmtOf st`,
whose schedule is `fast100`'s and whose link layout is `PT_LOCAL`, a run of rep `r` on its rounds' coins with its proof's
messages (`accepts_verdict_hm96`), on rounds that retained the encodings of `repFrames`' chunks (`accepts_rep_rounds`),
cuts at `repSpine`'s coins into the decoder's groups of the rounds, with no tail. -/
theorem repRun_at {mPts : ℕ} {wOP : ℕ → List (List Wire)} {wLIG : Accounting.Schedule → List (List Wire)}
    (hOP : OPCarries wOP) (hLIG : LIGCarries wLIG) (sizes : DecodeSizes) (T : TabSpec mPts) (st : Flock.Stmt)
    (hwf : StmtWF st) (sch : Flock.Schedule) (hf : Flock.fast100 st.m = some sch) (hdl : st.merkle.digestLen = 64)
    (hS : T.S = stmtOf st hwf (regionsOf st)) (hsch : T.sch = schedOf sch) (hpl : T.ptLocal = PT_LOCAL)
    (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (hm : 13 ≤ (stmtOf st hwf (regionsOf st)).m) {nE : ℕ}
    (hnE : (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL mPts pts).length = nE) (bytes : ByteArray)
    (p : Proof) (hp : decodeProof (shapeOf (Setup.ofCircuit st) sch nE) bytes = .ok p) (rds : Array Round)
    (hwfr : ∀ rd ∈ rds.toList, rd.coins.size = rd.n) (pend : PendL 𝔽 Dig)
    (hplay : Plays fOf (Model.repC execArith r.val (stmtOf st hwf (regionsOf st)) (schedOf sch) hm
        (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL mPts pts))
      (repMsgs (Setup.ofCircuit st) sch (2 + nE) p capDig) (rds.toList.map (·.coins)) pend)
    (dom : ByteArray) (Tv : F128)
    (hlen : (chunks (repFrames (Setup.ofCircuit st) sch dom (2 + nE) p Tv)).1.length = rds.size)
    (hret : ∀ k < rds.size,
      Retains rds[k]! (encs ((chunks (repFrames (Setup.ofCircuit st) sch dom (2 + nE) p Tv)).1.getD k []))) :
    PlaysCoin.RepRun (repsOf (D := Dig) execArith T pts r) (PlaysCoin.repSpine Dig T pts)
      (decOf (wZL T.S.m T.S.kLog ++
        (wOP ((Model.extraClaims T.S T.ptLocal mPts (fun _ _ => (0 : 𝔽))).length + 2) ++ wLIG T.sch)))
      rds.toList [] pend := by
  obtain ⟨S', m', k0, l₀, l₁, rest, ext, hsch', hm', pl, hlay⟩ := T
  dsimp only [PlaysCoin.repSpine, repsOf, TabSpec.sch] at hS hsch hpl ⊢
  subst hS hpl
  rw [hsch]
  have hn0 : (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL mPts (fun _ _ => (0 : 𝔽))).length = nE := by
    rw [extraClaims_length, ← hnE, extraClaims_length]
  rw [hn0, hnE]
  have sz := sizes _ _ _ hp
  have h14 : 14 ≤ st.m := by have := exec_fast100_ge hf; omega
  have hZL := zl_carries st.m st.c.kLog dom (Setup.ofCircuit st).digest
    (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck p.lincheck h14 sz.r1ab sz.r1c sz.zPartial
  have hC := hZL.append ((hOP (Setup.ofCircuit st) sch nE bytes p hf hp).append
    (hLIG (Setup.ofCircuit st) sch nE bytes p Tv hf hdl hp))
  have hF : repFrames (Setup.ofCircuit st) sch dom (2 + nE) p Tv =
      (Frame.dom dom :: (zcFrames st.m (Setup.ofCircuit st).digest (p.cap.foldl (· ++ ·) ByteArray.empty)
        p.zerocheck ++ lcFrames st.c.kLog p.lincheck)) ++
      (rsFrames (2 + nE) p.ringSwitches ++ ligFrames sch Tv p.cap p.ligerito) := by
    simp only [repFrames, List.cons_append, List.append_assoc]; rfl
  have hM : repMsgs (Setup.ofCircuit st) sch (2 + nE) p capDig =
      (zcMsgs st.m p.zerocheck ++ lcMsgs st.c.kLog p.lincheck) ++
      (opMsgs (2 + nE) p.ringSwitches ++ ligMsgs p.ligerito capDig sch) := by
    simp only [repMsgs, List.append_assoc]; rfl
  have hσ : repShape 𝔽 𝕂 Dig st.m st.c.kLog nE (schedOf sch) =
      (zcShape 𝔽 st.m ++ lcShape 𝔽 st.c.kLog) ++ (opShape 𝔽 (nE + 2) ++ ligShape 𝔽 𝕂 Dig (schedOf sch)) := by
    simp only [repShape, List.append_assoc]
  rw [← hF, ← hM, ← hσ] at hC
  refine ⟨hwfr, _, hplay, ?_⟩
  have hlenσ := carries_length hC
  rw [hlen, Array.length_toList.symm] at hlenσ
  show PlaysCoin.cut ((repShape 𝔽 𝕂 Dig st.m st.c.kLog nE (schedOf sch)).take _) _ =
    PlaysCoin.roundGroups (decOf (wZL st.m st.c.kLog ++ (wOP (nE + 2) ++ wLIG (schedOf sch)))) 0 rds.toList ++ [[]]
  rw [List.take_of_length_le (le_of_eq hlenσ), cut_of_carries hC, roundGroups_eq _ _ rds hC.2.1 hlen hret]

end FlockSoundness.Discharge.DecodeZL
