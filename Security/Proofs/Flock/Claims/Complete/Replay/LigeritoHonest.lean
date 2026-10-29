import Proofs.Flock.Claims.Complete.Replay.LigeritoTrack
import Proofs.Flock.Claims.Complete.Replay.LigeritoWeight
import Proofs.Flock.Claims.Complete.Replay.LigeritoFinal

/-!
# The honest Ligerito against the model's steps

The honest prover's Ligerito carried through the model's steps with its claim and weight: binds whose rest's
postcondition is the whole's (`Lg.HRuns.bindC`, `Lg.HRuns.bindDC`, `Lg.HRuns.bindYC`); the claim kept at each of the
model's steps (`Lg.sync_chal`, `Lg.sync_chalSplit`, `Lg.sync_ood`, `Lg.sync_close`, and level 0's first claim,
`Lg.head_sync`); the weight the model's (`Lg.WOK`: at an out-of-domain claim, `Lg.wok_ood`, a consistency claim,
`Lg.wok_close`, a level's challenges, `Lg.wok_level`, level 0's, `Lg.wok_lv0`, and the lists fit for the next level,
`Lg.preok_next`); the challenge loops (`Lg.HRuns.initChals`, `Lg.HRuns.levelChals`); and the prover between levels,
`Lg.Mid`, reached after level 0 (`Lg.mid_lv0`) and kept by an inner level's step (`Lg.mid_stepN`, folding the weight by
`Lg.wok_fold`), from which the last level meets the final identity (`Lg.goal_last`).
-/

namespace FlockComplete.Replay

open Flock Transcript FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Lg

/-! ## Binds whose rest's postcondition is the whole's -/

theorem HRuns.bindDC {p : LigeritoProof} {L : ℕ} {α β : Type} {x : LP PS α} {f : α → LP PS β} {c₁ c₂ c₃ : Cur}
    {tm₁ tm₂ tm₃ : Transcript} {Qa Ya : α → Prop} {Qf Xf : α → β → Prop} {Q Xb : β → Prop} {J₁ : TW → Prop}
    {J₂ : α → TW → Prop} {J₃ : β → TW → Prop} (hx : HRuns p L x c₁ tm₁ c₂ tm₂ Qa Ya J₁ J₂)
    (hf : ∀ a, Qa a → HRuns p L (f a) c₂ tm₂ c₃ tm₃ (Qf a) (Xf a) (J₂ a) J₃) (hQ : ∀ a b, Qa a → Qf a b → Q b)
    (hX : ∀ a b, Qa a → Qf a b → Xb b → Ya a ∧ Xf a b) : HRuns p L (x >>= f) c₁ tm₁ c₃ tm₃ Q Xb J₁ J₃ :=
  (HRuns.bindD (J₃ := fun _ => J₃) hx hf hQ hX).post fun _ _ _ _ ⟨_, h⟩ => h

theorem HRuns.bindC {p : LigeritoProof} {L : ℕ} {α β : Type} {x : LP PS α} {f : α → LP PS β} {c₁ c₂ c₃ : Cur}
    {tm₁ tm₂ tm₃ : Transcript} {Qa : α → Prop} {Q Xb : β → Prop} {J₁ : TW → Prop} {J₂ : α → TW → Prop}
    {J₃ : β → TW → Prop} (hx : HRuns p L x c₁ tm₁ c₂ tm₂ Qa (fun _ => True) J₁ J₂)
    (hf : ∀ a, Qa a → HRuns p L (f a) c₂ tm₂ c₃ tm₃ Q Xb (J₂ a) J₃) : HRuns p L (x >>= f) c₁ tm₁ c₃ tm₃ Q Xb J₁ J₃ :=
  (HRuns.bind (J₃ := fun _ => J₃) hx hf).post fun _ _ _ _ ⟨_, h⟩ => h

theorem HRuns.bindYC {p : LigeritoProof} {L : ℕ} {α β : Type} {x : LP PS α} {f : α → LP PS β} {c₁ c₂ c₃ : Cur}
    {tm₁ tm₂ tm₃ : Transcript} {Qa Ya : α → Prop} {Q Xb : β → Prop} {J₁ : TW → Prop} {J₂ : α → TW → Prop}
    {J₃ : β → TW → Prop} (hx : HRuns p L x c₁ tm₁ c₂ tm₂ Qa Ya J₁ J₂)
    (hf : ∀ a, Qa a → HRuns p L (f a) c₂ tm₂ c₃ tm₃ Q Xb (J₂ a) J₃) (hY : ∀ a b, Qa a → Q b → Xb b → Ya a) :
    HRuns p L (x >>= f) c₁ tm₁ c₃ tm₃ Q Xb J₁ J₃ :=
  HRuns.bindDC (Qf := fun _ => Q) (Xf := fun _ => Xb) hx hf (fun _ _ _ h => h) fun a b ha hb h => ⟨hY a b ha hb h, h⟩

/-! ## Small facts -/

theorem fast100_k0 {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) : 1 ≤ sch.k0 := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, decide (1 ≤ s.k0) = true := by
      decide +kernel
    exact of_decide_eq_true (key ⟨m, hm⟩ sch h)
  · rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at h
    cases h

theorem foldKs_push (a : Array F256) (r : F256) (x : Array F256) :
    foldKs (a.push r).toList x = foldK r (foldKs a.toList x) := by
  simp [foldKs, Array.toList_push, List.foldl_append]

theorem embedF_c0 (g : Array F128) : (embedF g).map F256.c0 = g := by
  apply Array.ext (by simp [embedF])
  intro i h₁ h₂
  simp [embedF, F256.ofF]

theorem levelsOf_push (levels : Array (Array F256)) (ch : Array F256) :
    levelsOf (levels.push ch) = levelsOf levels ++ [ch.toList.map gOf] := by
  simp [levelsOf]

theorem levelsOf_length (levels : Array (Array F256)) : (levelsOf levels).length = levels.size := by
  simp [levelsOf]

/-! ## The claim, step by step against the model's -/

section Sync

variable {P : LigeritoProof} {cf : ℕ → ℕ → F128}

theorem sync_chal {sm : LState} {tm : Transcript} {t w : Array F256} {N : ℕ} (h : Sync sm.s sm.q t w (N + 1))
    (hm : P.messages[sm.msg]! = roundPoly (foldK (chal P cf sm tm).1 t) (foldK (chal P cf sm tm).1 w)) :
    Sync (chal P cf sm tm).2.1.s (chal P cf sm tm).2.1.q (foldK (chal P cf sm tm).1 t)
      (foldK (chal P cf sm tm).1 w) N := by
  show Sync (sm.q.eval (chal P cf sm tm).1) (Poly.ofMsg P.messages[sm.msg]! (sm.q.eval (chal P cf sm tm).1)) _ _ N
  rw [hm]
  exact h.chal _

theorem sync_chalSplit {sm : LState} {tm : Transcript} {t w : Array F256} {N : ℕ} (h : Sync sm.s sm.q t w N)
    (hm : P.messages[sm.msg]! =
      roundPoly (embedF (splitK (foldK (chal P cf sm tm).1 t))) (liftU (foldK (chal P cf sm tm).1 w))) :
    Sync (chal P cf sm tm).2.1.s (chal P cf sm tm).2.1.q (embedF (splitK (foldK (chal P cf sm tm).1 t)))
      (liftU (foldK (chal P cf sm tm).1 w)) N := by
  show Sync (sm.q.eval (chal P cf sm tm).1) (Poly.ofMsg P.messages[sm.msg]! (sm.q.eval (chal P cf sm tm).1)) _ _ N
  rw [hm]
  exact h.chalSplit _

theorem sync_ood {dim split : ℕ} {sm : LState} {tm : Transcript} {g : Array F128} {w : Array F256} {N : ℕ}
    (h : Sync sm.s sm.q (embedF g) w N) (hdim : dim = N + 1) (ho : P.oodValues[sm.ood]! = mleF g (vec cf tm dim))
    (hm : P.messages[sm.msg]! = roundPoly (embedF g) (embedF (eqTable (vec cf tm dim)))) :
    Sync (ood P cf dim split sm tm).1.s (ood P cf dim split sm tm).1.q (embedF g)
      (addScaled w (embedF (eqTable (vec cf tm dim))) (cf (tm.k + 1) 0)) N := by
  show Sync (sm.s + (F256.ofF P.oodValues[sm.ood]!).smul (cf (tm.k + 1) 0))
    (sm.q.addScaled (Poly.ofMsg P.messages[sm.msg]! (F256.ofF P.oodValues[sm.ood]!)) (cf (tm.k + 1) 0)) _ _ N
  rw [ho, hm]
  exact h.ood _ (by rw [vec_size, hdim]) _

theorem sync_close {cols start : ℕ} {sm : LState} {tm : Transcript} (lv : Level) (g : Array F128) (ch : Array F256)
    (r : Array ℕ × Array F128 × F256) {w : Array F256} {N : ℕ} (hch : ch.size = lv.lanes)
    (hg : g.size = 2 ^ (lv.lanes + lv.cols)) (hpos : ∀ j < r.1.size, r.1[j]! < 2 ^ lv.d) (hn : lv.cols = N)
    (hr : r.2.2 = consE (r.1.map fun p => (encodeRows lv g)[p]!) r.1.size r.2.1 ch)
    (h : Sync sm.s sm.q (embedF (splitK (foldKs ch.toList (embedF g)))) w N)
    (hm : P.messages[sm.msg]! =
      roundPoly (embedF (splitK (foldKs ch.toList (embedF g)))) (liftU (embedF (consW lv.cols r.1 r.2.1)))) :
    Sync (close P cf cols start r sm tm).1.s (close P cf cols start r sm tm).1.q
      (embedF (splitK (foldKs ch.toList (embedF g)))) (addScaled w (liftU (embedF (consW lv.cols r.1 r.2.1)))
        (cf tm.k 0)) N := by
  show Sync (sm.s + r.2.2.smul (cf tm.k 0)) (sm.q.addScaled (Poly.ofMsg P.messages[sm.msg]! r.2.2) (cf tm.k 0)) _ _ N
  rw [hm, hr]
  exact Sync.close lv g r.1 r.2.1 ch hch hg hpos hn h _

/-- **Level 0's claim** after its out-of-domain sample and first message: `Σ t·b = T` and the out-of-domain value. -/
theorem head_sync (n : ℕ) (T : F128) (tm : Transcript) (tb b : Array F128) (hn : n ≠ 0) (htb : tb.size = 2 ^ n)
    (hb : b.size = 2 ^ n) (hT : dot tb b = T) (ho : P.oodValues[0]! = mleF tb (vec cf tm n))
    (hm : P.messages[0]! = roundPoly (embedF tb) (addScaled (embedF b) (embedF (eqTable (vec cf tm n)))
      (cf (tm.k + 1) 0))) :
    Sync (head P cf n (init T) tm).1.s (head P cf n (init T) tm).1.q (embedF tb)
      (addScaled (embedF b) (embedF (eqTable (vec cf tm n))) (cf (tm.k + 1) 0)) (n - 1) := by
  have hn1 : n - 1 + 1 = n := by omega
  show Sync (F256.ofF T + (F256.ofF P.oodValues[0]!).smul (cf (tm.k + 1) 0))
    (Poly.ofMsg P.messages[0]! (F256.ofF T + (F256.ofF P.oodValues[0]!).smul (cf (tm.k + 1) 0))) _ _ (n - 1)
  rw [ho, hm]
  refine Sync.ofMsg (by rw [embedF_size, htb, hn1]) (by rw [addScaled_size, embedF_size, hb, hn1]) ?_
  rw [gOf_add, gOf_smul, gOf_mleF tb _ (n - 1) (by rw [htb, hn1]) (by rw [vec_size, hn1]), hn1, gOf_ofF, ← hT,
    fOf_dot_rng tb b (2 ^ n) htb, map_sum]
  have key : ∑ x ∈ Finset.range (2 ^ n), FlockLevel3.embed (fOf tb[x]! * fOf b[x]!) =
      ∑ x ∈ Finset.range (2 ^ n), gv (embedF tb) x * gv (embedF b) x := by
    refine Finset.sum_congr rfl fun x hx => ?_
    have hx' := Finset.mem_range.1 hx
    rw [gv_embedF _ (by omega), gv_embedF _ (by omega), map_mul]
  rw [key, sum_addScaled]
  refine Finset.sum_congr rfl fun x hx => ?_
  rw [gv_addScaled _ _ _ (by rw [embedF_size, hb]; exact Finset.mem_range.1 hx)]

end Sync

/-! ## The weight, step by step against the model's lists -/

section Weight

variable {P : LigeritoProof} {cf : ℕ → ℕ → F128}

/-- **The prover's weight is the model's** on a model state's lists: `preW` at `N`. -/
def WOK (n : ℕ) (b : Array F128) (initG : List 𝕂) (levels : Array (Array F256)) (st : LState) (N : ℕ)
    (w : Array F256) : Prop :=
  ∀ i < 2 ^ (N + 1), gv w i = preW execArith n (bOf b) initG (levelsOf levels) (st.oods.toList.map oodOf)
    (st.conss.toList.map consOf) N i

theorem wok_ood {n : ℕ} {b : Array F128} {initG : List 𝕂} {levels : Array (Array F256)} {sm : LState}
    {tm : Transcript} {N dim split : ℕ} {w : Array F256} (h : WOK n b initG levels sm N w) (hw : w.size = 2 ^ (N + 1))
    (hsplit : split = levels.size) (hdim : dim = N + 1) :
    WOK n b initG levels (ood P cf dim split sm tm).1 N
      (addScaled w (embedF (eqTable (vec cf tm dim))) (cf (tm.k + 1) 0)) := by
  subst hsplit hdim
  intro i hi
  rw [gv_addOod n (bOf b) initG (levelsOf levels) _ _ N w _ _ hw (by rw [vec_size]) h hi,
    (ood_fields P cf (N + 1) levels.size sm tm).2.2.2.2.2.2.2,
    (ood_fields P cf (N + 1) levels.size sm tm).2.2.2.2.2.2.1,
    Array.toList_push, List.map_append, List.map_singleton, levelsOf_length]

theorem wok_close {n : ℕ} {b : Array F128} {initG : List 𝕂} {levels : Array (Array F256)} {sm : LState}
    {tm : Transcript} {N cols start : ℕ} {w : Array F256} (r : Array ℕ × Array F128 × F256)
    (h : WOK n b initG levels sm N w) (hw : w.size = 2 ^ (N + 1)) (hstart : start = levels.size) (hcols : cols = N)
    (hpa : r.1.size ≤ 2 ^ r.2.1.size) :
    WOK n b initG levels (close P cf cols start r sm tm).1 N
      (addScaled w (liftU (embedF (consW N r.1 r.2.1))) (cf tm.k 0)) := by
  subst hstart hcols
  intro i hi
  rw [gv_addCons n (bOf b) initG (levelsOf levels) _ _ cols w r.1 r.2.1 _ hw hpa h hi,
    (close_fields P cf cols levels.size r sm tm).2.2.2.2.2.2.2,
    (close_fields P cf cols levels.size r sm tm).2.2.2.2.2.2.1,
    Array.toList_push, List.map_append, List.map_singleton, levelsOf_length]

theorem ch4 {ch : Array F256} (hch : ch.size = 4) : ∃ c cs, ch.toList.map gOf = c :: cs ∧ cs.length = 3 := by
  have hl : ch.toList.length = 4 := by simp [hch]
  match h' : ch.toList.map gOf with
  | [] => have := congrArg List.length h'; simp [hl] at this
  | c :: cs => exact ⟨c, cs, rfl, by have := congrArg List.length h'; simp [hl] at this; omega⟩

/-- **A level's four challenges** fold the weight from `preW` before the level to the model's `resAt` after it. -/
theorem wok_fold {n : ℕ} {b : Array F128} {initG : List 𝕂} {levels : Array (Array F256)} {sm : LState} {M : ℕ}
    {w : Array F256} (h : WOK n b initG levels sm (M + 3) w)
    (hok : PreOK initG (levelsOf levels) (sm.oods.toList.map oodOf) (sm.conss.toList.map consOf) (M + 3))
    (hw : w.size = 2 ^ (M + 4)) (ch : Array F256) (hch : ch.size = 4) :
    ∀ y < 2 ^ M, gv (foldKs ch.toList w) y = resAt execArith n (bOf b) initG (levelsOf (levels.push ch))
      (sm.oods.toList.map oodOf) (sm.conss.toList.map consOf) M y := by
  have hl : ch.toList.length = 4 := by simp [hch]
  obtain ⟨c, cs, hc, hcs⟩ := ch4 hch
  have h16 : 2 ^ (M + 4) = 2 ^ M * 16 := by rw [pow_add]; norm_num
  intro y hy
  have hy' : y < w.size / 2 ^ ch.toList.length := by
    rw [hw, hl, h16, (by norm_num : (2 : ℕ) ^ 4 = 16), Nat.mul_div_cancel _ (by norm_num)]
    exact hy
  rw [gv_foldKs _ _ _ hy']
  rw [foldLoL_congr _ _ (preW execArith n (bOf b) initG (levelsOf levels) (sm.oods.toList.map oodOf)
    (sm.conss.toList.map consOf) (M + 3)) y (fun x hx => h _ ?_)]
  · rw [hc, foldLoL_preW execArith n (bOf b) initG (levelsOf levels) _ _ (M + 3) c cs M (by omega) hok y hy,
      levelsOf_push, hc]
  · rw [List.length_map, hl] at hx ⊢
    rw [show M + 3 + 1 = M + 4 by omega, h16]
    norm_num at hx ⊢
    omega

/-- **A level's four challenges and split** carry the weight from `preW` before the level to `preW` after it. -/
theorem wok_level {n : ℕ} {b : Array F128} {initG : List 𝕂} {levels : Array (Array F256)} {sm : LState} {M : ℕ}
    {w : Array F256} (h : WOK n b initG levels sm (M + 3) w)
    (hok : PreOK initG (levelsOf levels) (sm.oods.toList.map oodOf) (sm.conss.toList.map consOf) (M + 3))
    (hw : w.size = 2 ^ (M + 4)) (ch : Array F256) (hch : ch.size = 4) :
    WOK n b initG (levels.push ch) sm M (liftU (foldKs ch.toList w)) ∧
      PostOK initG (levelsOf (levels.push ch)) (sm.oods.toList.map oodOf) (sm.conss.toList.map consOf) M := by
  have hl : ch.toList.length = 4 := by simp [hch]
  obtain ⟨c, cs, hc, hcs⟩ := ch4 hch
  have h16 : 2 ^ (M + 4) = 2 ^ M * 16 := by rw [pow_add]; norm_num
  have hpost := postOK_of_preOK initG (levelsOf levels) (sm.oods.toList.map oodOf) (sm.conss.toList.map consOf)
    (M + 3) c cs M (by omega) hok
  rw [← hc, ← levelsOf_push] at hpost
  refine ⟨fun i hi => gv_liftU_preW n (bOf b) initG _ _ _ M _ (by rw [foldKs_size, hw, hl, h16]; norm_num)
    (fun o ho => ?_) (wok_fold h hok hw ch hch) hi, hpost⟩
  rcases (hpost.ood o ho).1 with hs | ⟨a, ha, hs⟩
  · rw [hs]; simp
  · rw [hs]; intro he; simp only [Option.some.injEq] at he; omega

/-- **Level 0's challenges and split** carry its weight, the batched basis plus the first out-of-domain claim, to
`preW` with the challenges as the original ones. -/
theorem wok_lv0 {n : ℕ} {b : Array F128} {sm : LState} (z : Array F128) (b0 : F128) (ch : Array F256)
    (hb : b.size = 2 ^ n) (hz : z.size = n) (hch : ch.size ≤ n) (hoods : sm.oods = #[⟨z, b0, none⟩])
    (hconss : sm.conss = #[]) :
    WOK n b (ch.toList.map gOf) #[] sm (n - ch.size)
        (liftU (foldKs ch.toList (addScaled (embedF b) (embedF (eqTable z)) b0))) ∧
      PostOK (ch.toList.map gOf) (levelsOf #[]) (sm.oods.toList.map oodOf) (sm.conss.toList.map consOf)
        (n - ch.size) := by
  have hl : ch.toList.length = ch.size := by simp
  have hfold : ∀ y < 2 ^ (n - ch.size), gv (foldKs ch.toList (addScaled (embedF b) (embedF (eqTable z)) b0)) y =
      resAt execArith n (bOf b) (ch.toList.map gOf) [] [oodOf ⟨z, b0, none⟩] [] (n - ch.size) y := by
    intro y hy
    have hsz : (addScaled (embedF b) (embedF (eqTable z)) b0).size = 2 ^ n := by rw [addScaled_size, embedF_size, hb]
    have hn : 2 ^ n = 2 ^ (n - ch.size) * 2 ^ ch.size := by rw [← pow_add]; congr 1; omega
    rw [gv_foldKs _ _ _ (by rw [hsz, hl, hn, Nat.mul_div_cancel _ (by positivity)]; exact hy)]
    rw [foldLoL_congr _ _ (resAt execArith n (bOf b) [] [] [oodOf ⟨z, b0, none⟩] [] n) y (fun x hx => ?_)]
    · have := fold_init execArith n (bOf b) (ch.toList.map gOf) [oodOf ⟨z, b0, none⟩] n (by simp; omega)
        (fun o ho => by simp at ho; subst ho; simp [oodOf, hz]) y
      rw [this, List.length_map, hl]
    · refine gv_w0 b z b0 n hb hz ?_
      rw [List.length_map, hl] at hx ⊢
      rw [hn]
      have : y + 1 ≤ 2 ^ (n - ch.size) := hy
      nlinarith
  have hpost : PostOK (ch.toList.map gOf) (levelsOf #[]) [oodOf ⟨z, b0, none⟩] [] (n - ch.size) :=
    ⟨fun o ho => by
      simp at ho; subst ho
      refine ⟨Or.inl rfl, ?_⟩
      simp [oodOf, FlockSoundness.oodFixed, FlockSoundness.origOf, hz, hl]; omega,
    fun c hc => by simp at hc⟩
  have hsz : (foldKs ch.toList (addScaled (embedF b) (embedF (eqTable z)) b0)).size = 2 ^ (n - ch.size) := by
    rw [foldKs_size, addScaled_size, embedF_size, hb, hl]
    rw [show 2 ^ n = 2 ^ (n - ch.size) * 2 ^ ch.size by rw [← pow_add]; congr 1; omega]
    exact Nat.mul_div_cancel _ (by positivity)
  refine ⟨fun i hi => ?_, by simpa [hoods, hconss] using hpost⟩
  have key := gv_liftU_preW n (bOf b) _ [] [oodOf ⟨z, b0, none⟩] [] (n - ch.size) _ hsz
    (fun o ho => by simp at ho; subst ho; simp [oodOf]) hfold hi
  rw [key, hoods, hconss]
  rfl

/-- **The lists before a level's challenges**: after its two out-of-domain claims and its consistency claim. -/
theorem preok_next {initG : List 𝕂} {levels : List (List 𝕂)} {M : ℕ} {sm sm' : LState} (o₁ o₂ : OodCtx)
    (cc : ConsCtx) (hok : PostOK initG levels (sm.oods.toList.map oodOf) (sm.conss.toList.map consOf) M)
    (hoods : sm'.oods = (sm.oods.push o₁).push o₂) (hconss : sm'.conss = sm.conss.push cc)
    (h₁ : o₁.split = some levels.length) (h₂ : o₂.split = some levels.length) (hz₁ : o₁.z.size = M + 1)
    (hz₂ : o₂.z.size = M + 1) (hcc : cc.start = levels.length) :
    PreOK initG levels (sm'.oods.toList.map oodOf) (sm'.conss.toList.map consOf) M := by
  rw [hoods, hconss, Array.toList_push, Array.toList_push, List.map_append, List.map_append, List.append_assoc,
    Array.toList_push, List.map_append]
  refine preOK_of_postOK initG levels _ _ M hok _ (fun o ho => ?_) (consOf cc) hcc
  simp only [List.map_cons, List.map_nil, List.singleton_append, List.mem_cons, List.not_mem_nil, or_false] at ho
  rcases ho with rfl | rfl
  · exact ⟨h₁, by simp [oodOf, hz₁]⟩
  · exact ⟨h₂, by simp [oodOf, hz₂]⟩

end Weight

/-! ## The challenge loops -/

/-- **The table and weight after `j` of `k` challenges** from the model's `(sm, tm)`, on `A` and `W` with the claim at
`N + k - 1` variables: folded at the model's coins, the last fold split, the claim kept. -/
def JC (P : LigeritoProof) (cf : ℕ → ℕ → F128) (sm : LState) (tm : Transcript) (k N : ℕ) (A W : Array F256)
    (j : ℕ) (y : TW) : Prop :=
  (j < k → y = (foldKs (chals P cf #[] sm tm j).1.toList A, foldKs (chals P cf #[] sm tm j).1.toList W) ∧
    Sync (chals P cf #[] sm tm j).2.1.s (chals P cf #[] sm tm j).2.1.q y.1 y.2 (N + k - 1 - j)) ∧
  (j = k → y = splitTW (foldKs (chals P cf #[] sm tm j).1.toList A, foldKs (chals P cf #[] sm tm j).1.toList W) ∧
    Sync (chals P cf #[] sm tm j).2.1.s (chals P cf #[] sm tm j).2.1.q y.1 y.2 N)

theorem jc_zero {P : LigeritoProof} {cf : ℕ → ℕ → F128} {sm : LState} {tm : Transcript} {k N : ℕ} {A W : Array F256}
    (hk : 1 ≤ k) (h : Sync sm.s sm.q A W (N + k - 1)) : JC P cf sm tm k N A W 0 (A, W) :=
  ⟨fun _ => ⟨rfl, h⟩, fun h0 => absurd h0 (by omega)⟩

/-- **A challenge** keeps `JC`: a fold, split at the last. -/
theorem jc_step {P : LigeritoProof} {cf : ℕ → ℕ → F128} {sm : LState} {tm : Transcript} {k N j : ℕ}
    {A W : Array F256} {x y : TW} (hj : j < k) (hx : JC P cf sm tm k N A W j x) (split : Bool)
    (hs : split = (j + 1 == k))
    (hy : y = if split = true then splitTW (foldTW (chal P cf (chals P cf #[] sm tm j).2.1
      (chals P cf #[] sm tm j).2.2).1 x) else foldTW (chal P cf (chals P cf #[] sm tm j).2.1
        (chals P cf #[] sm tm j).2.2).1 x)
    (hm : P.messages[(chals P cf #[] sm tm j).2.1.msg]! = roundPoly y.1 y.2) : JC P cf sm tm k N A W (j + 1) y := by
  obtain ⟨rfl, hsx⟩ := hx.1 hj
  subst hs
  unfold JC
  simp only [chals, foldKs_push]
  by_cases hk : j + 1 = k
  · rw [ite_eq_left (by simp [hk])] at hy
    subst hy
    have hsx' := hsx
    rw [show N + k - 1 - j = N by omega] at hsx'
    exact ⟨fun h => absurd hk (by omega), fun _ => ⟨rfl, sync_chalSplit hsx' hm⟩⟩
  · rw [ite_eq_right (by simp [hk])] at hy
    subst hy
    have hsx' := hsx
    rw [show N + k - 1 - j = N + k - 1 - (j + 1) + 1 by omega] at hsx'
    exact ⟨fun _ => ⟨rfl, sync_chal hsx' hm⟩, fun h => absurd h hk⟩

variable {p : LigeritoProof} {L : ℕ}

/-- **Level 0's `k0` challenges**, with the table and weight. -/
theorem HRuns.initChals (resp : ℕ → Claims.Msg) (k0 N : ℕ) (sm : LState) (tm : Transcript)
    (A W : Array F256) :
    HRuns p L (forIn [:k0] PUnit.unit fun j (_ : PUnit) => do
        let _ ← challengeP (asksS resp) (j + 1 == k0)
        Pure.pure (ForInStep.yield PUnit.unit))
      (cur sm) tm (cur (chals p (cfOf resp L) #[] sm tm k0).2.1) (chals p (cfOf resp L) #[] sm tm k0).2.2
      (fun _ => True) (fun _ => True) (JC p (cfOf resp L) sm tm k0 N A W 0)
      fun _ => JC p (cfOf resp L) sm tm k0 N A W k0 :=
  (HRuns.loop _ (fun i => cur (chals p (cfOf resp L) #[] sm tm i).2.1)
    (fun i => (chals p (cfOf resp L) #[] sm tm i).2.2)
    (fun _ _ => True) (fun _ _ => True) (fun _ => trivial) (fun _ _ _ _ _ => trivial) (fun _ _ => True) (fun _ => True)
    (fun j _ => JC p (cfOf resp L) sm tm k0 N A W j) k0
    (fun _ hj _ _ => ((HRuns.challenge resp _ _ _ _).bind fun _ _ =>
      HRuns.pure (b := ForInStep.yield PUnit.unit)
        (Q := fun r => ∃ s', r = ForInStep.yield s' ∧ True ∧ True) ⟨PUnit.unit, rfl, trivial, trivial⟩ _ _ _).post
      fun _ _ _ _ ⟨_, _, hr, ⟨_, hx, hy⟩, hm⟩ => fun _ _ => jc_step hj hx _ rfl (hr ▸ hy) hm)
    (fun _ _ _ _ _ _ _ _ => trivial) _ trivial).mono (fun _ _ => trivial) (fun _ _ _ => trivial) (fun _ h => h)
    fun _ _ _ _ h => h

/-- **A level's four challenges**, with the table and weight: three folds, then a fold whose table is kept, split. The
claim is the kept table's sum against the folded weight, and in step with the split table and weight except at the
last level (`last`), whose message is the unsplit table's. -/
theorem HRuns.levelChals (resp : ℕ → Claims.Msg) (last : Bool) (sm : LState) (tm : Transcript) (A W : Array F256)
    (M : ℕ) :
    HRuns p L (forIn [:4] (#[] : Array F256) fun i (f : Array F256) =>
        if (i + 1 == 4) = true then do
          let r ← liftT (sampleKP (asksS resp))
          _root_.modify fun st : LigState =>
            { tab := foldK r st.tab, w := foldK r st.w, msgs := st.msgs, oods := st.oods, qn := st.qn, cn := st.cn,
              cbn := st.cbn }
          let d ← MonadState.get
          _root_.modify fun st : LigState =>
            { tab := embedF (splitK st.tab), w := liftU st.w, msgs := st.msgs, oods := st.oods, qn := st.qn,
              cn := st.cn, cbn := st.cbn }
          let st ← MonadState.get
          FlockComplete.sendMsg (if last = true then roundPoly d.tab d.w else roundPoly st.tab st.w)
          Pure.pure (ForInStep.yield d.tab)
        else do
          let _ ← challengeP (asksS resp) false
          Pure.pure (ForInStep.yield f))
      (cur sm) tm (cur (chals p (cfOf resp L) #[] sm tm 4).2.1) (chals p (cfOf resp L) #[] sm tm 4).2.2
      (fun _ => True) (fun _ => True) (fun y => y = (A, W) ∧ Sync sm.s sm.q A W (M + 3))
      fun f y => f = foldKs (chals p (cfOf resp L) #[] sm tm 4).1.toList A ∧
        y = splitTW (f, foldKs (chals p (cfOf resp L) #[] sm tm 4).1.toList W) ∧
        (last = false →
          Sync (chals p (cfOf resp L) #[] sm tm 4).2.1.s (chals p (cfOf resp L) #[] sm tm 4).2.1.q y.1 y.2 M) ∧
        gOf (chals p (cfOf resp L) #[] sm tm 4).2.1.s =
          ∑ x ∈ Finset.range (2 ^ M), gv f x * gv (foldKs (chals p (cfOf resp L) #[] sm tm 4).1.toList W) x := by
  refine (HRuns.loop _ (fun i => cur (chals p (cfOf resp L) #[] sm tm i).2.1)
    (fun i => (chals p (cfOf resp L) #[] sm tm i).2.2) (fun _ _ => True) (fun _ _ => True) (fun _ => trivial)
    (fun _ _ _ _ _ => trivial) (fun _ _ => True) (fun _ => True)
    (fun j f y => (j < 4 → JC p (cfOf resp L) sm tm 4 M A W j y) ∧
      (j = 4 → f = foldKs (chals p (cfOf resp L) #[] sm tm 4).1.toList A ∧
        y = splitTW (f, foldKs (chals p (cfOf resp L) #[] sm tm 4).1.toList W) ∧
        (last = false →
          Sync (chals p (cfOf resp L) #[] sm tm 4).2.1.s (chals p (cfOf resp L) #[] sm tm 4).2.1.q y.1 y.2 M) ∧
        gOf (chals p (cfOf resp L) #[] sm tm 4).2.1.s =
          ∑ x ∈ Finset.range (2 ^ M), gv f x * gv (foldKs (chals p (cfOf resp L) #[] sm tm 4).1.toList W) x))
    4 (fun j hj f _ => ?_) (fun _ _ _ _ _ _ _ _ => trivial) _ trivial).mono (fun _ _ => trivial)
    (fun _ _ _ => trivial)
    (fun y ⟨hy, h⟩ => hy ▸ ⟨fun _ => jc_zero (by omega) (by simpa using h), fun h0 => absurd h0 (by omega)⟩) ?_
  · refine HRuns.ite (fun hc => ?_) fun hc => ?_
    · have hj3 : j = 3 := by have := beq_iff_eq.mp hc; omega
      subst hj3
      apply HRuns.post
      · exact (HRuns.sampleK resp _ _ _).bind fun r _ =>
          (HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bind fun _ _ =>
          (HRuns.get _ _ _).bind fun d _ =>
          (HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bind fun _ _ =>
          (HRuns.get _ _ _).bind fun st _ => (HRuns.sendMsg _ _ _ _).bind fun _ _ =>
          HRuns.pure (b := ForInStep.yield d.tab) (Q := fun r => ∃ s', r = ForInStep.yield s' ∧ True ∧ True)
            ⟨_, rfl, trivial, trivial⟩ _ _ _
      · rintro b y - - ⟨r, -, d, -, st, -, rfl, ⟨⟨⟨s₂, ⟨⟨s₁, ⟨hJ, hr⟩, hs₁⟩, hd⟩, hy⟩, hst⟩, hm⟩⟩
        intro s' hs'
        obtain rfl := ForInStep.yield.inj hs'
        have h1 : s₂.tab = foldK r s₁.tab := congrArg Prod.fst hs₁
        have h2 : s₂.w = foldK r s₁.w := congrArg Prod.snd hs₁
        have hd' : d.tab = s₂.tab := congrArg Prod.fst hd
        have hdw : d.w = s₂.w := congrArg Prod.snd hd
        have hy' : y = splitTW (foldTW r (tw s₁)) := by
          rw [hy]
          simp only [tw, splitTW, foldTW, h1, h2]
        have hJ3 := hJ.1 (by omega)
        obtain ⟨h3, hs3⟩ := hJ3.1 (by omega)
        have hf : d.tab = foldKs (chals p (cfOf resp L) #[] sm tm 4).1.toList A := by
          rw [hd', h1, show s₁.tab = (tw s₁).1 from rfl, h3, hr]
          simp only [chals, foldKs_push]
          rfl
        have hfw : d.w = foldKs (chals p (cfOf resp L) #[] sm tm 4).1.toList W := by
          rw [hdw, h2, show s₁.w = (tw s₁).2 from rfl, h3, hr]
          simp only [chals, foldKs_push]
          rfl
        rw [show M + 4 - 1 - 3 = M by omega] at hs3
        have he : gOf ((chals p (cfOf resp L) #[] sm tm 3).2.1.q.eval r) =
            ∑ x ∈ Finset.range (2 ^ M), gv (foldK r s₁.tab) x * gv (foldK r s₁.w) x := hs3.eval r
        have hs4 : (chals p (cfOf resp L) #[] sm tm 4).2.1.s = (chals p (cfOf resp L) #[] sm tm 3).2.1.q.eval r := by
          rw [hr]
          rfl
        refine ⟨fun h => absurd h (by omega), fun _ => ⟨hf, ?_, fun hl => ?_, ?_⟩⟩
        · rw [hy', ← hfw]
          show splitTW (foldK r s₁.tab, foldK r s₁.w) = splitTW (d.tab, d.w)
          rw [hd', hdw, h1, h2]
        · simp only [hl, Bool.false_eq_true, ↓reduceIte] at hm
          exact ((jc_step (by omega) hJ3 true rfl (by rw [hy', hr]; rfl) (by rw [← hst]; exact hm)).2 rfl).2
        · rw [hs4, he, ← hfw, hd', hdw, h1, h2]
    · apply HRuns.post
      · exact (HRuns.challenge resp false _ _ _).bind fun _ _ =>
          HRuns.pure (b := ForInStep.yield f) (Q := fun r => ∃ s', r = ForInStep.yield s' ∧ True ∧ True)
            ⟨_, rfl, trivial, trivial⟩ _ _ _
      · rintro b y - - ⟨_, rfl, hr, ⟨x, hx, hy⟩, hm⟩ s' hs'
        obtain rfl := ForInStep.yield.inj hs'
        have hj3 : j + 1 ≠ 4 := by simpa using hc
        exact ⟨fun _ => jc_step hj (hx.1 hj) false (by simpa using hj3) (hr ▸ hy) hm, fun h => absurd h hj3⟩
  · rintro f y _ _ ⟨-, hf⟩
    exact hf rfl

/-! ## Between levels -/

section Mid

variable {P : LigeritoProof} {cf : ℕ → ℕ → F128}

/-- **The prover between levels**, against the model's level-loop state `x`: its table is an F-table `g` whose codeword
rows at `lv` the last commitment `cm` holds, the claim is its sum against the weight `w`, which is the model's, the
model's lists are fit for the next level's challenges, and every consistency claim's positions fit its coins. -/
def Mid (n : ℕ) (b : Array F128) (initG : List 𝕂) (x : Vars × LState × Transcript) (cm : Commit) (lv : Level)
    (y : TW) : Prop :=
  ∃ g w, y = (embedF g, w) ∧ cm.rows = encodeRows lv g ∧ Sync x.2.1.s x.2.1.q (embedF g) w (x.1.1 - 1) ∧
    WOK n b initG x.1.2.2.1 x.2.1 (x.1.1 - 1) w ∧
    PreOK initG (levelsOf x.1.2.2.1) (x.2.1.oods.toList.map oodOf) (x.2.1.conss.toList.map consOf) (x.1.1 - 1) ∧
    ∀ cs ∈ x.2.1.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size

theorem query_pos (lv : Level) (rows : Array (Array F128)) (fold : Array F256) (sm : LState) (tm : Transcript) :
    (query P cf lv rows fold sm tm).1.1 = positions lv (queryCoins P cf lv sm tm) := rfl

theorem pos_lt {lv : Level} (hl : LevelSeam lv) (coins : Array F128) :
    ∀ i < (positions lv coins).size, (positions lv coins)[i]! < 2 ^ lv.d := fun i hi => by
  rw [getElem!_pos _ i hi]
  exact positions_lt_d hl coins _ (Array.getElem_mem hi)

/-- **A level's consistency claim, closed**: from the claim and weight after its two out-of-domain claims, on the
split fold `F` of the last commitment's table `g`, the claim and weight after its query and close. -/
theorem close_step {n : ℕ} {b : Array F128} {initG : List 𝕂} {levels : Array (Array F256)} {N cols start : ℕ}
    {sm : LState} {tm : Transcript} {w : Array F256} (lv : Level) (g : Array F128) (ch : Array F256)
    (rows : Array (Array F128)) (hseam : LevelSeam lv) (hch : ch.size = lv.lanes)
    (hg : g.size = 2 ^ (lv.lanes + lv.cols)) (hn : lv.cols = N) (hcols : cols = N) (hstart : start = levels.size)
    (hrows : rows = (positions lv (queryCoins P cf lv sm tm)).map fun p => (encodeRows lv g)[p]!)
    (hs : Sync sm.s sm.q (embedF (splitK (foldKs ch.toList (embedF g)))) w N)
    (hw : WOK n b initG levels sm N w)
    (hm : P.messages[sm.msg]! = roundPoly (embedF (splitK (foldKs ch.toList (embedF g))))
      (liftU (embedF (consW lv.cols (query P cf lv rows ch sm tm).1.1 (query P cf lv rows ch sm tm).1.2.1)))) :
    Sync (close P cf cols start (query P cf lv rows ch sm tm).1 (query P cf lv rows ch sm tm).2.1
        (query P cf lv rows ch sm tm).2.2).1.s
      (close P cf cols start (query P cf lv rows ch sm tm).1 (query P cf lv rows ch sm tm).2.1
        (query P cf lv rows ch sm tm).2.2).1.q
      (embedF (splitK (foldKs ch.toList (embedF g))))
      (addScaled w (liftU (embedF (consW lv.cols (query P cf lv rows ch sm tm).1.1
        (query P cf lv rows ch sm tm).1.2.1))) (cf (query P cf lv rows ch sm tm).2.2.k 0)) N ∧
    WOK n b initG levels (close P cf cols start (query P cf lv rows ch sm tm).1 (query P cf lv rows ch sm tm).2.1
        (query P cf lv rows ch sm tm).2.2).1 N
      (addScaled w (liftU (embedF (consW lv.cols (query P cf lv rows ch sm tm).1.1
        (query P cf lv rows ch sm tm).1.2.1))) (cf (query P cf lv rows ch sm tm).2.2.k 0)) ∧
    (query P cf lv rows ch sm tm).1.1.size ≤ 2 ^ (query P cf lv rows ch sm tm).1.2.1.size := by
  have hpa : (query P cf lv rows ch sm tm).1.1.size ≤ 2 ^ (query P cf lv rows ch sm tm).1.2.1.size :=
    pos_le_alpha lv hseam _ _ (vec_size _ _ _)
  have hr : (query P cf lv rows ch sm tm).1.2.2 = consE ((query P cf lv rows ch sm tm).1.1.map fun p =>
      (encodeRows lv g)[p]!) (query P cf lv rows ch sm tm).1.1.size (query P cf lv rows ch sm tm).1.2.1 ch := by
    rw [show (query P cf lv rows ch sm tm).1.1.size = lv.q from positions_size lv hseam _, query_pos, ← hrows]
    rfl
  refine ⟨sync_close lv g ch _ hch hg (pos_lt hseam _) hn hr hs hm, ?_, hpa⟩
  have := wok_close (P := P) (cf := cf) (cols := cols) (start := start) (tm := (query P cf lv rows ch sm tm).2.2)
    (query P cf lv rows ch sm tm).1 (sm := (query P cf lv rows ch sm tm).2.1) hw hs.ws hstart hcols hpa
  rw [hn]
  exact this

/-- **An inner level's step keeps `Mid`**: from `Mid` at the level-loop state `x` (the last commitment `cm` at level
`j + 1`), the honest prover's four challenges, its commitment `cmN` of the split fold `F`, two out-of-domain claims on
`F`, its query against `cm` and its consistency claim end in `Mid` at the model's step, with `cmN` at level `j + 2`. -/
theorem mid_stepN {n : ℕ} {b : Array F128} {initG : List 𝕂} {sch : Schedule} {j M : ℕ}
    {x : Vars × LState × Transcript} {cm cmN : Commit} {g : Array F128} {w : Array F256}
    (hrows : cm.rows = encodeRows (sch.levels.getD (j + 1) default) g)
    (hsync : Sync x.2.1.s x.2.1.q (embedF g) w (x.1.1 - 1)) (hwok : WOK n b initG x.1.2.2.1 x.2.1 (x.1.1 - 1) w)
    (hpre : PreOK initG (levelsOf x.1.2.2.1) (x.2.1.oods.toList.map oodOf) (x.2.1.conss.toList.map consOf)
      (x.1.1 - 1))
    (hcs : ∀ cs ∈ x.2.1.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size)
    (hM : M = x.1.1 - 4) (h4 : 4 ≤ x.1.1) (hlvs : x.1.2.2.1.size = j)
    (hseam : LevelSeam (sch.levels.getD (j + 1) default)) (hlanes : (sch.levels.getD (j + 1) default).lanes = 4)
    (hcols : (sch.levels.getD (j + 1) default).cols = M)
    {C : Array F256 × LState × Transcript} (hC : C = chals P cf #[] x.2.1 x.2.2 4)
    {tm₁ : Transcript} (htm₁ : tm₁ = ap (pBytes (capBytes P.recursiveCaps[j + 1]!)) C.2.2)
    {o₁ o₂ : LState × Transcript} (ho₁ : o₁ = ood P cf (M + 1) (j + 1) C.2.1 tm₁)
    (ho₂ : o₂ = ood P cf (M + 1) (j + 1) o₁.1 o₁.2)
    {Q : (Array ℕ × Array F128 × F256) × LState × Transcript}
    (hQ : Q = query P cf (sch.levels.getD (j + 1) default) P.recursiveProofs[j]!.rows C.1 o₂.1 o₂.2)
    {F : Array F128} (hF : F = splitK (foldKs C.1.toList (embedF g)))
    (hsync₁ : Sync C.2.1.s C.2.1.q (embedF F) (liftU (foldKs C.1.toList w)) M)
    (hcm : cmN.rows = encodeRows (sch.levels.getD (j + 2) default) F)
    (hv₁ : P.oodValues[C.2.1.ood]! = mleF F (vec cf tm₁ (M + 1)))
    (hm₁ : P.messages[C.2.1.msg]! = roundPoly (embedF F) (embedF (eqTable (vec cf tm₁ (M + 1)))))
    (hv₂ : P.oodValues[o₁.1.ood]! = mleF F (vec cf o₁.2 (M + 1)))
    (hm₂ : P.messages[o₁.1.msg]! = roundPoly (embedF F) (embedF (eqTable (vec cf o₁.2 (M + 1)))))
    (hrp : P.recursiveProofs[j]!.rows =
      (positions (sch.levels.getD (j + 1) default) (queryCoins P cf (sch.levels.getD (j + 1) default) o₂.1 o₂.2)).map
        fun p => cm.rows[p]!)
    (hm₃ : P.messages[Q.2.1.msg]! =
      roundPoly (embedF F) (liftU (embedF (consW (sch.levels.getD (j + 1) default).cols Q.1.1 Q.1.2.1)))) :
    Mid n b initG (stepN P cf sch j x) cmN (sch.levels.getD (j + 2) default)
      (embedF F, addScaled (addScaled (addScaled (liftU (foldKs C.1.toList w)) (embedF (eqTable (vec cf tm₁ (M + 1))))
        (cf (tm₁.k + 1) 0)) (embedF (eqTable (vec cf o₁.2 (M + 1)))) (cf (o₁.2.k + 1) 0))
        (liftU (embedF (consW (sch.levels.getD (j + 1) default).cols Q.1.1 Q.1.2.1))) (cf Q.2.2.k 0)) := by
  subst hM hC htm₁ ho₁ ho₂ hQ hF
  have hN : x.1.1 - 1 = x.1.1 - 4 + 3 := by omega
  rw [hN] at hsync hwok hpre
  have hC := chals_fields P cf #[] x.2.1 x.2.2 4
  have hch : (chals P cf #[] x.2.1 x.2.2 4).1.size = (sch.levels.getD (j + 1) default).lanes := by
    rw [hC.2.2.2.2.2.2.1, hlanes]; rfl
  have hg : g.size = 2 ^ ((sch.levels.getD (j + 1) default).lanes + (sch.levels.getD (j + 1) default).cols) := by
    have := hsync.ts
    rw [embedF_size] at this
    rw [this, hlanes, hcols]
    congr 1
    omega
  have hw4 : w.size = 2 ^ (x.1.1 - 4 + 4) := hsync.ws
  have w0 : WOK n b initG x.1.2.2.1 (chals P cf #[] x.2.1 x.2.2 4).2.1 (x.1.1 - 4 + 3) w := by
    intro i hi
    rw [hC.2.2.2.2.2.2.2.1, hC.2.2.2.2.2.2.2.2]
    exact hwok i hi
  have hpre' : PreOK initG (levelsOf x.1.2.2.1) ((chals P cf #[] x.2.1 x.2.2 4).2.1.oods.toList.map oodOf)
      ((chals P cf #[] x.2.1 x.2.2 4).2.1.conss.toList.map consOf) (x.1.1 - 4 + 3) := by
    rw [hC.2.2.2.2.2.2.2.1, hC.2.2.2.2.2.2.2.2]
    exact hpre
  obtain ⟨w1, hpost⟩ := wok_level w0 hpre' hw4 _ (by rw [hC.2.2.2.2.2.2.1]; rfl)
  have s1 := sync_ood (split := j + 1) hsync₁ rfl hv₁ hm₁
  have w2 := wok_ood (P := P) (cf := cf) (split := j + 1)
    (tm := ap (pBytes (capBytes P.recursiveCaps[j + 1]!)) (chals P cf #[] x.2.1 x.2.2 4).2.2) w1 hsync₁.ws
    (by simp [hlvs]) rfl
  have s2 := sync_ood (split := j + 1) s1 rfl hv₂ hm₂
  have w3 := wok_ood (P := P) (cf := cf) (split := j + 1) (tm := (ood P cf (x.1.1 - 4 + 1) (j + 1)
    (chals P cf #[] x.2.1 x.2.2 4).2.1 (ap (pBytes (capBytes P.recursiveCaps[j + 1]!))
      (chals P cf #[] x.2.1 x.2.2 4).2.2)).2) w2 s1.ws (by simp [hlvs]) rfl
  obtain ⟨s3, w4, hpa⟩ := close_step (cols := x.1.1 - 4) (start := j + 1) (sch.levels.getD (j + 1) default) g
    (chals P cf #[] x.2.1 x.2.2 4).1 P.recursiveProofs[j]!.rows hseam hch hg hcols rfl (by simp [hlvs])
    (by rw [hrp, hrows]) s2 w3 hm₃
  unfold Mid stepN
  dsimp only
  rw [show x.1.1 - 4 + 1 - 1 = x.1.1 - 4 by omega]
  refine ⟨_, _, rfl, hcm, s3, w4, ?_, ?_⟩
  · exact preok_next _ _ _ hpost rfl rfl (by simp [hlvs]) (by simp [hlvs]) (by simp) (by simp) (by simp [hlvs])
  · refine conss_push_pos (cs := (chals P cf #[] x.2.1 x.2.2 4).2.1.conss) ?_ hpa
    rw [hC.2.2.2.2.2.2.2.2]
    exact hcs

/-- **Level 0 ends in `Mid`** at the model's level-loop start: from level 0's claim after its `k0` challenges, the
honest prover's commitment `cm1` of the split fold `F`, two out-of-domain claims on `F`, its query against the level-0
commitment `cm0` of `tb` and its consistency claim. -/
theorem mid_lv0 {sch : Schedule} {T : F128} {cap : Array ByteArray} {t : Transcript} {cm0 cm1 : Commit}
    {tb b : Array F128} (hb : b.size = 2 ^ sch.n) (hk : sch.k0 ≤ sch.n) (hseam : LevelSeam (sch.levels.getD 0 default))
    (hlanes : (sch.levels.getD 0 default).lanes = sch.k0)
    (hcols : (sch.levels.getD 0 default).cols = sch.n - sch.k0)
    (hcm0 : cm0.rows = encodeRows (sch.levels.getD 0 default) tb) (htb : tb.size = 2 ^ sch.n)
    {C : Array F256 × LState × Transcript}
    (hC : C = chals P cf #[] (head P cf sch.n (init T) (start cap T t)).1 (head P cf sch.n (init T) (start cap T t)).2
      sch.k0)
    {tm₁ : Transcript} (htm₁ : tm₁ = ap (pBytes (capBytes P.recursiveCaps[0]!)) C.2.2)
    {o₁ o₂ : LState × Transcript} (ho₁ : o₁ = ood P cf (sch.n - sch.k0 + 1) 0 C.2.1 tm₁)
    (ho₂ : o₂ = ood P cf (sch.n - sch.k0 + 1) 0 o₁.1 o₁.2)
    {Q : (Array ℕ × Array F128 × F256) × LState × Transcript}
    (hQ : Q = query P cf (sch.levels.getD 0 default) P.initialProof.rows C.1 o₂.1 o₂.2)
    {F : Array F128} (hF : F = splitK (foldKs C.1.toList (embedF tb)))
    {W : Array F256} (hW : W = addScaled (embedF b) (embedF (eqTable (vec cf (start cap T t) sch.n)))
      (cf ((start cap T t).k + 1) 0))
    (hsync₁ : Sync C.2.1.s C.2.1.q (embedF F) (liftU (foldKs C.1.toList W)) (sch.n - sch.k0))
    (hcm : cm1.rows = encodeRows (sch.levels.getD 1 default) F)
    (hv₁ : P.oodValues[C.2.1.ood]! = mleF F (vec cf tm₁ (sch.n - sch.k0 + 1)))
    (hm₁ : P.messages[C.2.1.msg]! = roundPoly (embedF F) (embedF (eqTable (vec cf tm₁ (sch.n - sch.k0 + 1)))))
    (hv₂ : P.oodValues[o₁.1.ood]! = mleF F (vec cf o₁.2 (sch.n - sch.k0 + 1)))
    (hm₂ : P.messages[o₁.1.msg]! = roundPoly (embedF F) (embedF (eqTable (vec cf o₁.2 (sch.n - sch.k0 + 1)))))
    (hrp : P.initialProof.rows =
      (positions (sch.levels.getD 0 default) (queryCoins P cf (sch.levels.getD 0 default) o₂.1 o₂.2)).map
        fun p => cm0.rows[p]!)
    (hm₃ : P.messages[Q.2.1.msg]! =
      roundPoly (embedF F) (liftU (embedF (consW (sch.levels.getD 0 default).cols Q.1.1 Q.1.2.1)))) :
    Mid sch.n b ((pre P cf sch T cap t).1.toList.map gOf) (x₀ P cf sch T cap t) cm1 (sch.levels.getD 1 default)
      (embedF F, addScaled (addScaled (addScaled (liftU (foldKs C.1.toList W))
        (embedF (eqTable (vec cf tm₁ (sch.n - sch.k0 + 1)))) (cf (tm₁.k + 1) 0))
        (embedF (eqTable (vec cf o₁.2 (sch.n - sch.k0 + 1)))) (cf (o₁.2.k + 1) 0))
        (liftU (embedF (consW (sch.levels.getD 0 default).cols Q.1.1 Q.1.2.1))) (cf Q.2.2.k 0)) := by
  subst hC htm₁ ho₁ ho₂ hQ hF hW
  have hC := chals_fields P cf #[] (head P cf sch.n (init T) (start cap T t)).1
    (head P cf sch.n (init T) (start cap T t)).2 sch.k0
  have hH := head_fields P cf sch.n (init T) (start cap T t)
  have hch : (chals P cf #[] (head P cf sch.n (init T) (start cap T t)).1 (head P cf sch.n (init T) (start cap T t)).2
      sch.k0).1.size = (sch.levels.getD 0 default).lanes := by
    rw [hC.2.2.2.2.2.2.1, hlanes]; simp
  have hg : tb.size = 2 ^ ((sch.levels.getD 0 default).lanes + (sch.levels.getD 0 default).cols) := by
    rw [htb, hlanes, hcols]; congr 1; omega
  obtain ⟨w1, hpost⟩ := wok_lv0 (n := sch.n) (b := b) (vec cf (start cap T t) sch.n)
    (cf ((start cap T t).k + 1) 0) (chals P cf #[] (head P cf sch.n (init T) (start cap T t)).1
      (head P cf sch.n (init T) (start cap T t)).2 sch.k0).1 hb (vec_size _ _ _)
    (by rw [hC.2.2.2.2.2.2.1]; simpa using hk)
    (by rw [hC.2.2.2.2.2.2.2.1, hH.2.2.2.2.2.2.2]; rfl) (by rw [hC.2.2.2.2.2.2.2.2, hH.2.2.2.2.2.2.1]; rfl)
  rw [hC.2.2.2.2.2.2.1, Array.size_empty, Nat.zero_add] at w1 hpost
  have s1 := sync_ood (split := 0) hsync₁ rfl hv₁ hm₁
  have w2 := wok_ood (P := P) (cf := cf) (split := 0) (tm := ap (pBytes (capBytes P.recursiveCaps[0]!))
    (chals P cf #[] (head P cf sch.n (init T) (start cap T t)).1 (head P cf sch.n (init T) (start cap T t)).2
      sch.k0).2.2) w1 hsync₁.ws rfl rfl
  have s2 := sync_ood (split := 0) s1 rfl hv₂ hm₂
  have w3 := wok_ood (P := P) (cf := cf) (split := 0) (tm := (ood P cf (sch.n - sch.k0 + 1) 0
    (chals P cf #[] (head P cf sch.n (init T) (start cap T t)).1 (head P cf sch.n (init T) (start cap T t)).2
      sch.k0).2.1 (ap (pBytes (capBytes P.recursiveCaps[0]!)) (chals P cf #[] (head P cf sch.n (init T)
        (start cap T t)).1 (head P cf sch.n (init T) (start cap T t)).2 sch.k0).2.2)).2) w2 s1.ws rfl rfl
  obtain ⟨s3, w4, hpa⟩ := close_step (cols := sch.n - sch.k0) (start := 0) (sch.levels.getD 0 default) tb
    (chals P cf #[] (head P cf sch.n (init T) (start cap T t)).1 (head P cf sch.n (init T) (start cap T t)).2
      sch.k0).1 P.initialProof.rows hseam hch hg hcols rfl rfl (by rw [hrp, hcm0]) s2 w3 hm₃
  unfold Mid x₀ pre lv0
  dsimp only
  rw [show sch.n - sch.k0 + 1 - 1 = sch.n - sch.k0 by omega]
  refine ⟨_, _, rfl, hcm, s3, w4, ?_, ?_⟩
  · exact preok_next _ _ _ hpost rfl rfl (by simp) (by simp) (by simp) (by simp) (by simp)
  · refine conss_push_pos (cs := (chals P cf #[] (head P cf sch.n (init T) (start cap T t)).1
      (head P cf sch.n (init T) (start cap T t)).2 sch.k0).2.1.conss) ?_ hpa
    rw [hC.2.2.2.2.2.2.2.2, hH.2.2.2.2.2.2.1]
    simp

/-- **The last level meets the final identity**: from `Mid` before the last level (the last commitment `cm` at level
`R`), the honest prover's four challenges, whose split fold is the proof's `y_r`, and its query against `cm`, whose rows
are the proof's last opening, the verifier's model ends with `lhsE y_r ext res = s`. -/
theorem goal_last {sch : Schedule} {T : F128} {cap : Array ByteArray} {t : Transcript} {b : Array F128}
    {basisAt : Array F256 → ℕ → Array F256} {j : ℕ} {x : Vars × LState × Transcript}
    (hx : x = loop P cf sch (x₀ P cf sch T cap t) j) (hjR : j + 1 = sch.levels.size - 1) (hL : 2 ≤ sch.levels.size)
    (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) {cm : Commit} {g : Array F128} {w : Array F256}
    (hrows : cm.rows = encodeRows (sch.levels.getD (sch.levels.size - 1) default) g)
    (hsync : Sync x.2.1.s x.2.1.q (embedF g) w (x.1.1 - 1))
    (hwok : WOK sch.n b ((pre P cf sch T cap t).1.toList.map gOf) x.1.2.2.1 x.2.1 (x.1.1 - 1) w)
    (hpre : PreOK ((pre P cf sch T cap t).1.toList.map gOf) (levelsOf x.1.2.2.1) (x.2.1.oods.toList.map oodOf)
      (x.2.1.conss.toList.map consOf) (x.1.1 - 1))
    (hcs : ∀ cs ∈ x.2.1.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size)
    (hseam : LevelSeam (sch.levels.getD (sch.levels.size - 1) default))
    (hlanes : (sch.levels.getD (sch.levels.size - 1) default).lanes = 4)
    (hcols : (sch.levels.getD (sch.levels.size - 1) default).cols = x.1.1 - 4)
    (hbasis : ∀ q e y, q.size + e = sch.n → y < 2 ^ e →
      gOf (basisAt q e)[y]! = Model.bHat execArith sch.n (bOf b) (q.toList.map gOf ++ Model.bitsK e y))
    {C : Array F256 × LState × Transcript} (hC : C = chals P cf #[] x.2.1 x.2.2 4)
    {F : Array F256} (hF : F = foldKs C.1.toList (embedF g))
    (hsum : gOf C.2.1.s = ∑ y ∈ Finset.range (2 ^ (x.1.1 - 4)), gv F y * gv (foldKs C.1.toList w) y)
    (hyr : P.finalProof.yr = splitK F)
    (hfin : P.finalProof.rows = (positions (sch.levels.getD (sch.levels.size - 1) default)
      (queryCoins P cf (sch.levels.getD (sch.levels.size - 1) default) C.2.1
        (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') C.2.2))).map fun p => cm.rows[p]!) :
    lhsE P.finalProof.yr (out P cf sch T cap t).1.2.2.2 (resE basisAt sch.n (pre P cf sch T cap t).1
      (out P cf sch T cap t).1.2.2.1 (out P cf sch T cap t).1.2.2.2 (out P cf sch T cap t).2.1) =
      (out P cf sch T cap t).2.1.s := by
  have hD := done_out P cf sch T cap t hL hsd
  have hI := inv_loop P cf sch T cap t hL hsd j (by omega)
  rw [← hx] at hI
  have hout : out P cf sch T cap t = stepL P cf sch x := by
    unfold out
    rw [← hjR, loop_stepL P cf sch _ (by omega), ← hx]
  rw [hout] at hD ⊢
  subst hC hF
  have hCf := chals_fields P cf #[] x.2.1 x.2.2 4
  have h4 : 4 ≤ x.1.1 := by rw [hI.sd]; omega
  have hN : x.1.1 - 1 = x.1.1 - 4 + 3 := by omega
  rw [hN] at hsync hwok hpre
  have hch : (chals P cf #[] x.2.1 x.2.2 4).1.size = (sch.levels.getD (sch.levels.size - 1) default).lanes := by
    rw [hCf.2.2.2.2.2.2.1, hlanes]; rfl
  have hg : g.size = 2 ^ ((sch.levels.getD (sch.levels.size - 1) default).lanes +
      (sch.levels.getD (sch.levels.size - 1) default).cols) := by
    have := hsync.ts
    rw [embedF_size] at this
    rw [this, hlanes, hcols]
    congr 1
    omega
  have hw4 : w.size = 2 ^ (x.1.1 - 4 + 4) := hsync.ws
  have w0 : WOK sch.n b ((pre P cf sch T cap t).1.toList.map gOf) x.1.2.2.1 (chals P cf #[] x.2.1 x.2.2 4).2.1
      (x.1.1 - 4 + 3) w := by
    intro i hi
    rw [hCf.2.2.2.2.2.2.2.1, hCf.2.2.2.2.2.2.2.2]
    exact hwok i hi
  have hpre' : PreOK ((pre P cf sch T cap t).1.toList.map gOf) (levelsOf x.1.2.2.1)
      ((chals P cf #[] x.2.1 x.2.2 4).2.1.oods.toList.map oodOf)
      ((chals P cf #[] x.2.1 x.2.2 4).2.1.conss.toList.map consOf) (x.1.1 - 4 + 3) := by
    rw [hCf.2.2.2.2.2.2.2.1, hCf.2.2.2.2.2.2.2.2]
    exact hpre
  have hfold := wok_fold w0 hpre' hw4 _ (by rw [hCf.2.2.2.2.2.2.1]; rfl)
  have h16 : ∀ a : ℕ, 2 ^ (a + 4) / 2 ^ 4 = 2 ^ a := fun a => by
    rw [pow_add, Nat.mul_div_cancel _ (by positivity)]
  have hFs : (foldKs (chals P cf #[] x.2.1 x.2.2 4).1.toList (embedF g)).size = 2 ^ (x.1.1 - 4) := by
    rw [foldKs_size, embedF_size, hg, Array.length_toList, hch, hlanes, hcols, Nat.add_comm, h16]
  have hrows' : P.finalProof.rows = (positions (sch.levels.getD (sch.levels.size - 1) default)
      (queryCoins P cf (sch.levels.getD (sch.levels.size - 1) default) (chals P cf #[] x.2.1 x.2.2 4).2.1
        (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2))).map
      fun p => (encodeRows (sch.levels.getD (sch.levels.size - 1) default) g)[p]! := by
    rw [hfin, hrows]
  have e1 : lhsE P.finalProof.yr = lhsE (splitK (foldKs (chals P cf #[] x.2.1 x.2.2 4).1.toList (embedF g))) := by
    rw [hyr]
  have hpa := pos_le_alpha (sch.levels.getD (sch.levels.size - 1) default) hseam
    (queryCoins P cf (sch.levels.getD (sch.levels.size - 1) default) (chals P cf #[] x.2.1 x.2.2 4).2.1
      (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2))
    (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows (chals P cf #[] x.2.1 x.2.2 4).1
      (chals P cf #[] x.2.1 x.2.2 4).2.1
      (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.2.1
    (vec_size _ _ _)
  have hl : ∀ l ∈ (x.1.2.2.1.push (chals P cf #[] x.2.1 x.2.2 4).1).toList, l.size = 4 := hD.lv4
  have hlvs : (x.1.2.2.1.push (chals P cf #[] x.2.1 x.2.2 4).1).size = sch.levels.size - 1 := hD.lvs
  have hext : x.1.1 - 4 = sch.n - sch.k0 - 3 * (sch.levels.size - 1) := hD.ext
  have hk0 : (pre P cf sch T cap t).1.size = sch.k0 := by simp [pre]
  have horig : (origE (pre P cf sch T cap t).1 (x.1.2.2.1.push (chals P cf #[] x.2.1 x.2.2 4).1)).size +
      (x.1.1 - 4) = sch.n := by
    rw [origE_size _ _ hl, hlvs, hext, hk0]
    omega
  unfold stepL
  dsimp only
  rw [e1]
  refine lhsE_honest _ (foldKs (chals P cf #[] x.2.1 x.2.2 4).1.toList w) (embedF (consW (x.1.1 - 4)
    (positions (sch.levels.getD (sch.levels.size - 1) default)
    (queryCoins P cf (sch.levels.getD (sch.levels.size - 1) default) (chals P cf #[] x.2.1 x.2.2 4).2.1
      (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)))
    (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows (chals P cf #[] x.2.1 x.2.2 4).1
      (chals P cf #[] x.2.1 x.2.2 4).2.1
      (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.2.1))
    _ _ _ _ _ hFs ?_ ?_ fun y hy => ?_
  · exact hsum
  · rw [show (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows
        (chals P cf #[] x.2.1 x.2.2 4).1 (chals P cf #[] x.2.1 x.2.2 4).2.1
        (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.2.2 =
        consE ((query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows
          (chals P cf #[] x.2.1 x.2.2 4).1 (chals P cf #[] x.2.1 x.2.2 4).2.1
          (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P')
            (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.1.map fun p =>
              (encodeRows (sch.levels.getD (sch.levels.size - 1) default) g)[p]!)
          (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows
            (chals P cf #[] x.2.1 x.2.2 4).1 (chals P cf #[] x.2.1 x.2.2 4).2.1
            (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P')
              (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.1.size
          (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows
            (chals P cf #[] x.2.1 x.2.2 4).1 (chals P cf #[] x.2.1 x.2.2 4).2.1
            (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P')
              (chals P cf #[] x.2.1 x.2.2 4).2.2)).1.2.1 (chals P cf #[] x.2.1 x.2.2 4).1 by
      rw [query_pos, positions_size _ hseam, ← hrows']
      rfl]
    rw [query_pos, consE_fold _ g _ _ _ hch hg (pos_lt hseam _), hcols]
  · refine res_honest basisAt sch.n (pre P cf sch T cap t).1 _ (x.1.1 - 4)
      (closeLast P cf (x.1.1 - 4) (sch.levels.size - 1)
        (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows (chals P cf #[] x.2.1 x.2.2 4).1
          (chals P cf #[] x.2.1 x.2.2 4).2.1
          (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).1
        (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows (chals P cf #[] x.2.1 x.2.2 4).1
          (chals P cf #[] x.2.1 x.2.2 4).2.1
          (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).2.1
        (query P cf (sch.levels.getD (sch.levels.size - 1) default) P.finalProof.rows (chals P cf #[] x.2.1 x.2.2 4).1
          (chals P cf #[] x.2.1 x.2.2 4).2.1
          (ap (fun P' => P.finalProof.yr.foldl (fun P'' w => pF w P'') P') (chals P cf #[] x.2.1 x.2.2 4).2.2)).2.2).1
      (bOf b) hl horig ?_ ?_ ?_ (fun y hy => hbasis _ _ y horig hy) _ _ _ _ ?_ hpa _ hfold hy
    · refine conss_push_pos ?_ hpa
      rw [(query_fields ..).2.2.2.2.2.2.1, hCf.2.2.2.2.2.2.2.2]
      exact hcs
    · intro o ho
      have := hD.oods o ho
      unfold FlockSoundness.Refine.oodFixed OodDim at *
      split at this
      · rename_i hs
        rw [hs, origE_size _ _ hl, hlvs, hext, hk0]
        omega
      · rename_i s hs
        rw [hs, oodFixed_size _ hl (by rw [hlvs]; exact this.1), hlvs, hext]
        omega
    · intro cs hc
      have := hD.conss cs hc
      unfold ConsDim at this
      rw [consFixed_size _ hl, hlvs, hext]
      omega
    · rw [hlvs]
      rfl

end Mid

end Lg

end FlockComplete.Replay
