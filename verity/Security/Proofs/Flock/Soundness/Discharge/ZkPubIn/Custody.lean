import Proofs.Flock.Soundness.Discharge.ZkPubIn.Sound

set_option Elab.async false

/-!
# The canonical session with the verifier's public inputs at the coin server's custody

`Canonical.zk_session_soundC_custody_json` and `ZkHidden.drawOkZKHJ_of_file` with `--public-inputs F` (`own` F's
bytes):

* `drawOkZKHJP_of_file`: from `DrawFileZK dj (subset kd) pop`, with `nTab I` dividing `kd`, `DrawOkZKHJP` holds at
  `mPtsOf c.g (kd / nTab I)` link points at every draw (`setupHidden_mPts_drawnP`, `setupHidden_mPts_of_provedP`);
* `zk_session_soundCP_custody_json`: `zk_session_soundCP` at the record's custody (`RecordsLiveZKJP`) and each draw file
  the JSON value `flock-verify draw` printed for the verifier's own draw under `subset kd`.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden StrictCR
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open FlockSoundness.Discharge.Canonical (formatOk)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n : ℕ} {Reg : Type}

/-- **The coin server's draws are `DrawOkZKHJP`** (`drawOkZKHJ_of_file` with `own`): when the record's draw file is
the verifier's draw under `subset kd` over `[0, pop)` (`DrawFileZK`) and the session's tables divide `kd`, every draw
`S` is `DrawOkZKHJP` at `mPtsOf c.g (kd / nTab I)` link points. -/
theorem drawOkZKHJP_of_file (I : Inputs) (own : ByteArray) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) pop) (S : Finset (Fin n)) :
    DrawOkZKHJP I own dj (mPtsOf c.g (kd / nTab I)) S := by
  intro j
  have hc' : ∀ j : ℕ, HmOut.parse (tagsAt (zkTags I) I.count j) I.circuitFile I.tables = .ok c := fun j =>
    (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc
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
    obtain ⟨ud, hud, -⟩ := setupHidden_mPts_drawnP hst
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
    · rw [setupHidden_mPts_of_provedP (hc' j) (fun D' h' => ?_) hst]
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
    · rw [setupHidden_mPts_of_provedP (hc' j) (fun D' h' => ?_) hst]
      rw [hpd D' h', Array.size_extract, hsz, Nat.min_eq_left hjq, Nat.add_sub_cancel_left, hn]

/-- **One canonical hidden-output `--zk --public-inputs F` sampled session of `J` tables is sound, at the record's
custody and the printed draw file** (`Canonical.zk_session_soundC_custody_json` with `own`): `RecordsLiveZKJP`, each
draw file the JSON value `flock-verify draw` printed for the verifier's own draw under `subset kd`, at
`mPtsOf c.g (kd / nTab I)` link points and `nTab I ∣ kd`. -/
theorem zk_session_soundCP_custody_json
    (I : Inputs) (own : ByteArray) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hfmt : formatOk c = true) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi) {pub : Public}
    (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, the coin server's custody of the record, and its draw file as `flock-verify draw`
    -- printed the verifier's draw
    (hTags : TagsOkZHJ I)
    (hRec : RecordsLiveZKJP I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr)
    (hdj : ∀ S, dj S = (drawOf (.subset kd) n S).toJson)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJA_of_runsOnZCJA execArith hs512 _ _ wr _ _ _ _ _
        (runsZCJH_liveCP I own ht hc hfmt dj y₀ wr _ hTags hRec
          fun _ => drawOkZKHJP_of_file I own ht hc kd n hdiv dj (drawFileZK_of_toJson_subset dj kd n hdj) _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZKP I own o ∧
        K₀ ≤ (((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).wrong
          (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
            (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planZHJP dj y₀)
              (tabZHJP dj y₀)
              ((rsZCP dj y₀ ht hc hfmt hpi hpub hU).bindingZ (rsZCP_computes dj y₀ ht hc hfmt hpi hpub hU))
              (reg σ) (cont σ) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJP dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planZHJP dj y₀)
          (tabZHJP dj y₀)
          ((rsZCP dj y₀ ht hc hfmt hpi hpub hU).bindingZ (rsZCP_computes dj y₀ ht hc hfmt hpi hpub hU))
          k Rw M ρ t' (reg σ) (cont σ) :=
  zk_session_soundCP I own ht dj y₀ wr dg N hc hU hfmt hpi hpub kd hkd Lb src hA3
    (fun _ => drawOkZKHJP_of_file I own ht hc kd n hdiv dj (drawFileZK_of_toJson_subset dj kd n hdj) _) hTags hRec
    hk hRw hM hρ hr ht' K₀ τ

end FlockSoundness.Discharge.ZkPubIn
