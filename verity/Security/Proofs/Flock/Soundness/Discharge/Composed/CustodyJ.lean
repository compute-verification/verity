import Proofs.Flock.Soundness.Discharge.Composed.SoundJ
import Proofs.Flock.Soundness.Assumptions.Zk

/-!
# `zk_session_soundJ` at the coin server's custody

`zk_session_soundJ` takes `RecordsLiveZKJ` (`hRec`) and `DrawOkZKJ` at every draw of the live subset law (`hDraw`). As
at one table (`ZkLink/DrawOk.lean`), the record and its draw file are custody, and the rest of `DrawOkZKJ` is
flock-verify's own arithmetic. This file splits them:

* `drawOkZKJ_of_file`: from `DrawFileZK dj (subset kd) pop`, with `nTab I` dividing `kd`, `DrawOkZKJ` holds at
  `mPtsOf c.g (kd / nTab I)` link points at every draw `S`. `UnitDraw.ofJson` decodes a subset draw only at its `k`
  units, so a draw file it decodes is a draw of `kd` units, and part `j` (`partDraw`) of it decodes to its `kd / nTab I`
  units, which are the units `partOf` sends to table `j`. No fact about the law's runs is needed: the run that reads
  past the source's bytes draws every unit, whose draw file decodes only at `kd = n`.
* `zk_session_soundJ_custody`: `zk_session_soundJ` with `RecordCustodyZKJ` (`live-verifier`) for `hRec` and `hDraw`,
  at `mPtsOf c.g (kd / nTab I)` link points, and `nTab I ∣ kd` (flock-verify accepts a session of `count > 1` tables
  only at a draw that splits into `count` equal parts, `accepts_tables`).
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open Lean (Json toJson)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

set_option maxHeartbeats 2000000 in
/-- What `UnitDraw.ofJson` returns: its units are the `units` field's, it has no closure without a `closure` key, and a
`subset k` draw has `k` units. -/
theorem ofJson_shape {j : Json} {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson j = .ok D) :
    (∃ a, J.arr j "units" "unit_draw" = .ok a ∧ a.mapM (J.asNat · "U2: a drawn unit") = .ok D.units) ∧
    (j.getObjVal? "closure" = .error "property not found: closure" → D.closure = #[]) ∧
    (∀ k, D.law = .subset k → D.units.size = k) := by
  unfold Draw.UnitDraw.ofJson at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
  all_goals refine ⟨⟨_, ‹J.arr j "units" _ = _›, ‹Array.mapM _ _ = _›⟩, fun hcl => ?_, fun k hk => ?_⟩
  all_goals first
    | exact (Except.ok.inj ‹(pure #[] : V (Array ℕ)) = Except.ok _›).symm
    | exact absurd (‹j.getObjVal? "closure" = Except.ok _›.symm.trans hcl) (fun h => by cases h)
    | (exfalso; cases hk; done)
    | (have e : _ = k := Draw.Law.subset.inj hk
       subst e
       have hs := ‹¬(decide (_ > _) || _ != _) = true›
       simp only [Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq, not_or, not_not] at hs
       exact hs.2)

theorem mapM_asNat_toJson_list (s : String) : ∀ l : List ℕ, (l.map toJson).mapM (J.asNat · s) = .ok l
  | [] => rfl
  | a :: l => by
      rw [List.map_cons, List.mapM_cons, show J.asNat (toJson a) s = .ok a from rfl, mapM_asNat_toJson_list s l]
      rfl

theorem mapM_asNat_toJson (s : String) (a : Array ℕ) : (a.map toJson).mapM (J.asNat · s) = .ok a := by
  rw [Array.mapM_eq_mapM_toList, Array.toList_map, mapM_asNat_toJson_list]
  rfl

/-- **Part `j` of a draw decodes to its units** (`partDraw` has no closure). -/
theorem ofJson_partDraw {pop count j : ℕ} {units : Array ℕ} {D : Draw.UnitDraw}
    (h : Draw.UnitDraw.ofJson (Tables.partDraw pop units count j) = .ok D) :
    D.proved = units.extract (j * (units.size / count)) (j * (units.size / count) + units.size / count) := by
  obtain ⟨⟨a, ha, hm⟩, hcl, -⟩ := ofJson_shape h
  have h2 : J.arr (Tables.partDraw pop units count j) "units" "unit_draw" =
      .ok ((units.extract (j * (units.size / count)) (j * (units.size / count) + units.size / count)).map toJson) :=
    rfl
  rw [h2] at ha
  cases ha
  rw [mapM_asNat_toJson] at hm
  have hc : D.closure = #[] := hcl rfl
  rw [Draw.UnitDraw.proved, hc, ← Except.ok.inj hm]
  rfl


/-- **The coin server's draws are `DrawOkZKJ`.** On untyped `--statement` tags whose circuit file parses under the
`--zk` tags to `c`: when the record's draw file is the verifier's draw under the subset law `subset kd` over `[0, pop)`
(`DrawFileZK`) and the session's tables divide `kd`, every draw `S` is `DrawOkZKJ` at `mPtsOf c.g (kd / nTab I)` link
points. -/
theorem drawOkZKJ_of_file (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) pop) (S : Finset (Fin n)) :
    DrawOkZKJ I dj (mPtsOf c.g (kd / nTab I)) S := by
  intro j
  have hc' : ∀ j : ℕ, (if (tagsAt (zkTags I) I.count j).typed then HmRow.parseTyped (tagsAt (zkTags I) I.count j)
      I.circuitFile I.tables else HmRow.parse (tagsAt (zkTags I) I.count j) I.circuitFile I.tables) = .ok c := by
    intro j
    rw [Exec.tagsAt_typed, zkTags_typed, ht, parse_tagsAt]
    exact hc
  by_cases hD : ∃ D, Draw.UnitDraw.ofJson (dj S) = .ok D
  swap
  · have hj : ∀ D', Draw.UnitDraw.ofJson (drawJ I (dj S) j) ≠ .ok D' := by
      intro D' h'
      apply hD
      unfold drawJ at h'
      split_ifs at h'
      · exact ⟨_, h'⟩
      · unfold partJson at h'
        split at h'
        · exact ⟨_, ‹_›⟩
        · exact ⟨_, h'⟩
    refine ⟨fun D' h' => (hj D' h').elim, fun st hst => ?_⟩
    obtain ⟨ud, hud, -⟩ := setupH_mPts_drawn hst
    exact (hj ud hud).elim
  obtain ⟨D, hDj⟩ := hD
  have hDe := hdj S D hDj
  have hsz : D.units.size = kd := (ofJson_shape hDj).2.2 kd (by rw [hDe]; rfl)
  have hS : D.units = ((S.image Fin.val).sort (· ≤ ·)).toArray := by rw [hDe]; rfl
  by_cases h1 : I.count ≤ 1
  · have hn : nTab I = 1 := by simp [nTab, h1]
    have hdr : drawJ I (dj S) j = dj S := by simp [drawJ, h1]
    have hpart : partS I (dj S) S j = S := by
      refine Finset.filter_true_of_mem fun u _ => Fin.ext ?_
      have h1 := (partOf I (dj S) u.val).isLt
      have h2 := j.isLt
      omega
    refine ⟨fun D' h' u hu => ?_, fun st hst => ?_⟩
    · rw [hdr, hDj] at h'
      rw [← Except.ok.inj h', hDe, drawOf_proved, drawOf_mem]
      exact ⟨u, by rwa [hpart] at hu, rfl⟩
    · rw [setupH_mPts_of_proved (hc' j) (fun D' h' => ?_) hst]
      rw [hdr, hDj] at h'
      rw [← Except.ok.inj h', hn, Nat.div_one, hDe, drawOf_proved, ← hDe, hsz]
  · have hn : nTab I = I.count := by simp [nTab, h1]
    have hdr : drawJ I (dj S) j = Tables.partDraw D.population D.units I.count j := by
      simp only [drawJ, h1, ite_false, partJson, hDj]
    have hq : kd = I.count * (kd / I.count) := by
      rw [hn] at hdiv
      exact (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hdiv)).symm
    have hjq : j.val * (kd / I.count) + kd / I.count ≤ kd := by
      have hj : j.val < I.count := by have := j.isLt; omega
      have := Nat.mul_le_mul_right (kd / I.count) (Nat.succ_le_of_lt hj)
      rw [Nat.succ_mul] at this
      omega
    have hpd : ∀ D', Draw.UnitDraw.ofJson (drawJ I (dj S) j) = .ok D' →
        D'.proved = D.units.extract (j.val * (kd / I.count)) (j.val * (kd / I.count) + kd / I.count) := by
      intro D' h'
      rw [hdr] at h'
      rw [ofJson_partDraw h', hsz]
    refine ⟨fun D' h' u hu => ?_, fun st hst => ?_⟩
    · rw [hpd D' h']
      obtain ⟨huS, hpu⟩ := Finset.mem_filter.1 hu
      have hmem : u.val ∈ D.units.toList := by
        rw [hS]
        simpa [Finset.mem_sort] using ⟨u, huS, rfl⟩
      have hi : D.units.toList.idxOf u.val < D.units.toList.length := List.idxOf_lt_length_of_mem hmem
      have hget : D.units.toList[D.units.toList.idxOf u.val] = u.val := List.getElem_idxOf hi
      have hpo := congrArg Fin.val hpu
      simp only [partOf, hDj, hsz] at hpo
      generalize D.units.toList.idxOf u.val = i at hi hget hpo
      rw [Array.length_toList, hsz] at hi
      have hq0 : 0 < kd / I.count := by
        rcases Nat.eq_zero_or_pos (kd / I.count) with h0 | h0
        · rw [h0, Nat.mul_zero] at hq; omega
        · exact h0
      have hlt : i / (kd / I.count) < I.count := (Nat.div_lt_iff_lt_mul hq0).2 (by rw [← hq]; exact hi)
      have hji : i / (kd / I.count) = j.val := by omega
      have h1 := Nat.div_add_mod i (kd / I.count)
      have h2 := Nat.mod_lt i hq0
      rw [hji] at h1
      rw [Array.mem_def, Array.toList_extract, List.extract_eq_take_drop]
      refine List.mem_iff_getElem.2 ⟨i - j.val * (kd / I.count), ?_, ?_⟩
      · simp only [List.length_drop, List.length_take, Array.length_toList, hsz]
        rw [Nat.mul_comm] at h1
        omega
      · rw [List.getElem_take, List.getElem_drop]
        convert hget using 2
        rw [Nat.mul_comm] at h1
        omega
    · rw [setupH_mPts_of_proved (hc' j) (fun D' h' => ?_) hst]
      rw [hpd D' h', Array.size_extract, hsz, Nat.min_eq_left hjq, Nat.add_sub_cancel_left, hn]

/-- **One `--zk` sampled session of `J` tables is sound, at the coin server's custody.** `zk_session_soundJ` with
`RecordCustodyZKJ` (`live-verifier`: the record is the live session's, and its draw file the verifier's draw under the
subset law) in place of `hRec` and `hDraw`, at `mPtsOf c.g (kd / nTab I)` link points, the session's tables dividing
`kd`. -/
theorem zk_session_soundJ_custody
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and the coin server's custody of the record and its draw file
    (hTags : TagsOkZJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJ_live I ht dj y₀ wr _ hTags hCust.1 fun _ => drawOkZKJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
            ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
          ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope))
          k Rw M ρ t' (reg σ) (cont σ) :=
  zk_session_soundJ I ht dj y₀ wr dg N hc hU hpub hpt hscope kd hkd Lb src hA3
    (fun _ => drawOkZKJ_of_file I ht hc kd n hdiv dj hCust.2 _) hTags hCust.1 hk hRw hM hρ hr ht' K₀ τ

end FlockSoundness.Discharge.Composed
