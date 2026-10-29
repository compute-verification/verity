import Proofs.Flock.Soundness.Discharge.ZkHidden.RunsJP
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundDrawOk
import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
import Proofs.Flock.Soundness.Assumptions.ZkP
import Proofs.Flock.Soundness.Discharge.ZkHidden.ParseStmt

/-!
# The coin server's draws and the decoder at the verifier's public inputs, under the record's custody

`ZkReg.drawOkZKHJ_of_file` and the decoder the `--zk` session's simulator reads, at flock-verify `--zk --public-inputs`
(`own`):

* `setupHidden_mPts_drawnP`, `setupHidden_mPts_of_provedP`: `setupHidden_mPts_drawn` and `setupHidden_mPts_of_proved`
  at `own`, from `Hidden.setupHidden_specP` and the draw over the public input ports (`HmIn.drawn`, which is
  `HmRow.drawn` at the circuit with those ports as its output ports), at the statement circuit `c`
  (`HmOut.parseStmt`, whose `g` is the committed circuit's: `unrolled_g`).
* `drawOkZKHJP_of_file`: from `DrawFileZK dj (subset kd) pop`, with `nTab I` dividing `kd`, `DrawOkZKHJP` holds at
  `mPtsOf c.g (kd / nTab I)` link points at every draw `S`, by `drawOkZKHJ_of_file`'s argument.
* `decodesZCJH_custodyP`: `decodesZCJH_liveIP` with `hRec` and `hDraw` read off `RecordCustodyZKJP` (`live-verifier` at
  `own`).
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open Lean (Json)

local notation "𝕂" => FlockLevel3.GF256

section MPts

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {partition : Option Json} {program own : Option ByteArray} {st : Flock.Stmt}

/-- **`m_pts` at a drawn hidden-output session at `own`** (`setupHidden_mPts_drawn`). -/
theorem setupHidden_mPts_drawnP {d : Json}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program own = .ok st) :
    ∃ ud, Draw.UnitDraw.ofJson d = .ok ud ∧ st.mPts = mPtsOf st.c.g ud.proved.size := by
  obtain ⟨_, s⟩ := Hidden.setupHidden_specP h
  obtain ⟨pi', pop, -, -, hd⟩ := Hidden.setupHidden_drawnP h
  unfold HmIn.drawn at hd
  obtain ⟨ud, hud, hn⟩ := drawn_n hd
  refine ⟨ud, hud, ?_⟩
  rw [s.mPts, s.nbl, s.blocksReal, s.n, hn]
  rfl

/-- **`m_pts` from the circuit and the draw's proved count at `own`** (`setupHidden_mPts_of_proved`). -/
theorem setupHidden_mPts_of_provedP {c : Flock.Circuit} {d : Json} {N₀ : ℕ} (hty : tags.typed = false)
    (hcS : HmOut.parseStmt tags cf tables = .ok c) (hD : ∀ D, Draw.UnitDraw.ofJson d = .ok D → D.proved.size = N₀)
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program own = .ok st) :
    st.mPts = mPtsOf c.g N₀ := by
  obtain ⟨ud, hud, hm⟩ := setupHidden_mPts_drawnP h
  have e := FlockSoundness.parseStmt_of_commit hty (Hidden.setupHidden_specP h).choose_spec.parse
  rw [hcS] at e
  rw [hm, ← unrolled_g st.c, ← Except.ok.inj e, hD ud hud]

end MPts

variable {n : ℕ} {Reg : Type}

/-- **The coin server's draws are `DrawOkZKHJP`** (`ZkReg.drawOkZKHJ_of_file` at `own`). -/
theorem drawOkZKHJP_of_file (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hcS : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) pop) (S : Finset (Fin n)) :
    DrawOkZKHJP I own dj (mPtsOf c.g (kd / nTab I)) S := by
  intro j
  have hty : ∀ j : ℕ, (tagsAt (zkTags I) I.count j).typed = false := fun j =>
    (Exec.tagsAt_typed _ _ _).trans ((zkTags_typed I).trans ht)
  have hc' : ∀ j : ℕ, HmOut.parseStmt (tagsAt (zkTags I) I.count j) I.circuitFile I.tables = .ok c := fun j =>
    (parseStmtH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hcS
  by_cases hD : ∃ D, Draw.UnitDraw.ofJson (dj S) = .ok D
  swap
  · have hj : ∀ D', Draw.UnitDraw.ofJson (drawJ I (dj S) j) ≠ .ok D' := by
      intro D' h'
      apply hD
      unfold drawJ at h'
      split_ifs at h'
      · exact ⟨_, h'⟩
      · unfold DecodeZLJ.partJson at h'
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
    · rw [setupHidden_mPts_of_provedP (hty j) (hc' j) (fun D' h' => ?_) hst]
      rw [hdr, hDj] at h'
      rw [← Except.ok.inj h', hn, Nat.div_one, hDe, drawOf_proved, ← hDe, hsz]
  · have hn : nTab I = I.count := by simp [nTab, h1]
    have hdr : drawJ I (dj S) j = Tables.partDraw D.population D.units I.count j := by
      simp only [drawJ, h1, ite_false, DecodeZLJ.partJson, hDj]
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
    · rw [setupHidden_mPts_of_provedP (hty j) (hc' j) (fun D' h' => ?_) hst]
      rw [hpd D' h', Array.size_extract, hsz, Nat.min_eq_left hjq, Nat.add_sub_cancel_left, hn]

/-- **The decoder at the live hidden-output tables at `own`, under `live-verifier`** (`decodesZCJH_liveIP` at
`RecordCustodyZKJP`): the record's custody at `own` gives the record's half (`hRec`), and its draw file the coin
server's draws (`drawOkZKHJP_of_file`), at `mPtsOf c.g (kd / nTab I)` link points. -/
theorem decodesZCJH_custodyP (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hcS : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJP I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr kd) :
    DecodesOnZKJA (K := 𝕂) execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr
      (opnAtZHJP (planZHJP dj y₀) I own dj) (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true)
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L :=
  decodesZCJH_liveIP I own ht dj y₀ wr L hT hCust.1 fun _ => drawOkZKHJP_of_file I own ht hcS kd n hdiv dj hCust.2 _

end FlockSoundness.Discharge.ZkHidden
