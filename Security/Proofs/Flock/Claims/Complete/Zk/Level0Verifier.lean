import Proofs.Flock.Claims.Complete.Zk.Level0HonestRun
import Proofs.Flock.Claims.Complete.Zk.Level0Opens

/-!
# Level 0's masked Ligerito: the verifier's premise, proved

`verifier : ZkTables.Level0.Verifier`: the masked verifier's Ligerito returns the masked prover's record on its proof
and ends where the prover did. It is `ligeritoWithZ_of` at the replies' coins, its premises from the prover's run:

* the prover ends at the masked model's transcript and meets its final identity (`honestZ`);
* every level's Merkle check passes (`merkleOkZ_of_opens`, `ligeritoPZk_opens`);
* the proof's lengths and `t_pad` padding coefficients (`proverShape`);
* the record the prover returns is the model's (`rc_eq_recZ`): its lane phase's (`lanes`), the model's lane coins and
  pair (`laneRun_ch`, `laneRun_rho`) and its first out-of-domain coin, which the model's state keeps (`outZ_oods0`);
* the prover's cap is the commitment's and its pair's coins independent, since it returned (`ligeritoPZk_capRho`).
-/

namespace FlockComplete.L0

open Flock Transcript FlockSoundness FlockSoundness.Refine FlockComplete Replay Replay.Lg

local instance lawfulBEqByteArrayL0 : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-- **A masked Ligerito prover that returns** names the commitment's cap and drew independent pair coins: its one
refusal is on dependent ones. -/
theorem ligeritoPZk_capRho (resp : ℕ → Claims.Msg) (hs : Hashes PS) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) {t t' : Transcript} {l l' : Array Claims.Msg} {P : LigeritoProof} {zl : Zk.Level0Msgs}
    {rc : Zk.L0Rec}
    (hrun : ((ligeritoPZk (asksS resp) hs ms sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce).run t).go l =
      (.ok ((P, zl, rc), t'), l')) :
    P.initialCap = cm0.cap ∧ Zk.rhoIndependent rc.rho = true := by
  unfold ligeritoPZk at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)]
    at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨R, stF⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  obtain ⟨hP1, hP⟩ := Prod.mk.inj hP
  obtain ⟨-, rfl⟩ := Prod.mk.inj hP
  refine ⟨by rw [← hP1], ?_⟩
  obtain ⟨tA, lA, hlA, h1⟩ := lsampleVecP_bind resp hn h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_modify _) h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_liftTM _) h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_powB nonce) h1
  obtain ⟨tB, lB, hlB, h1⟩ := lsampleP_bind resp h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_modify _) h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind lsilent_get h1
  obtain ⟨_, _, _, h1⟩ := lsilent_bind (lsilent_sendMsg _) h1
  obtain ⟨s, stC, tC, lC, hL, h1⟩ := lp_bind_ok h1
  have hI := lforIn_inv _ (fun j (s : Array F256 × (F256 × F256) × Zk.Level0Msgs) (_ : Array Claims.Msg) =>
      j = sch.k0 → Zk.rhoIndependent s.2.1 = true) sch.k0 ?step (fun h => absurd h (by omega)) hL
  case step =>
    rintro j hj s st t l₀ r st' t' l₂ - hb
    obtain ⟨t₃, l₃, hl₃, hb⟩ := lsampleKP_bind resp hb
    obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_modify _) hb
    by_cases hj1 : j + 1 = sch.k0
    · rw [ite_eq_left (by simp [hj1])] at hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨t₄, l₄, hl₄, hb⟩ := lsampleKP_bind resp hb
      obtain ⟨t₅, l₅, hl₅, hb⟩ := lsampleKP_bind resp hb
      split at hb
      · obtain ⟨_, _, _, _, h0, _⟩ := lp_bind_ok hb
        cases h0
      rename_i hρ
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_liftTM _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_modify _) hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_sendMsg _) hb
      obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
      exact ⟨_, rfl, fun _ => by simpa using hρ⟩
    · rw [ite_eq_right (by simp [hj1])] at hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind lsilent_get hb
      obtain ⟨_, _, _, hb⟩ := lsilent_bind (lsilent_sendMsg _) hb
      obtain ⟨rfl, rfl⟩ := lpure_go_ok hb
      exact ⟨_, rfl, fun h => absurd h hj1⟩
  iterate 11 (obtain ⟨_, _, _, _, -, h1⟩ := lp_bind_ok h1)
  obtain ⟨rfl, -⟩ := lpure_go_ok h1
  exact hI rfl

/-- The model's level loop only pushes out-of-domain contexts. -/
theorem loop_oods (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (x : Vars × LState × Transcript) :
    ∀ j, Pre x.2.1.oods (loop p cf sch x j).2.1.oods
  | 0 => Pre.refl _
  | j + 1 => by
    refine (loop_oods p cf sch x j).trans ?_
    by_cases hj : j + 1 < sch.levels.size - 1
    · rw [loop_stepN p cf sch x hj]
      simp only [stepN, close_fields, ood_fields, query_fields, chals_fields]
      exact (Pre.push _ _).trans (Pre.push _ _)
    · rw [loop_stepL p cf sch x hj]
      simp only [stepL, closeLast_fields, query_fields, chals_fields]
      exact Pre.refl _

/-- **The masked model keeps its first out-of-domain context** to the end: level 0's, at the head's coin. -/
theorem outZ_oods0 (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (zl : Zk.Level0) (t : Transcript) :
    ((outZ p cf sch T cap zl t).2.1.oods[0]!).beta = cf (t.k + 1) 0 := by
  have hl := (laneRun_fields p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
    (head p cf sch.n (init T) (startZ cap T t)).2 sch.k0 le_rfl).2.2.2.2.2.2.2.1
  have h0 : Pre ((head p cf sch.n (init T) (startZ cap T t)).1.oods) (x₀Z p cf sch T cap zl t).2.1.oods := by
    simp only [x₀Z, preZ, lv0Z, close_fields, query_fields, ood_fields]
    rw [hl]
    exact (Pre.push _ _).trans (Pre.push _ _)
  have h1 := h0.trans (loop_oods p cf sch _ (sch.levels.size - 1))
  rw [(head_fields p cf sch.n (init T) (startZ cap T t)).2.2.2.2.2.2.2, (init_fields T).2.2.2.2.2.1] at h1
  rw [show (outZ p cf sch T cap zl t).2.1.oods[0]! = _ from Pre.get (A := #[]) h1]
  rfl

/-- **The masked prover's record is the masked model's**: at the replies' coins from `L` sends, a prover that ran from
`L + k` sends returns the record the masked verifier returns at the model's state and lane phase. -/
theorem rc_eq_recZ {resp : ℕ → Claims.Msg} {sch : Schedule} {T : F128} {tb b : Array F128} {lay : Zk.PadLayout}
    {hv : Array F128} {rep : ℕ} {extra : ℕ → ℕ → F128} {L : ℕ} {t : Transcript} {l : Array Claims.Msg}
    {P : LigeritoProof} {zm : Zk.Level0Msgs} {rc : Zk.L0Rec} (cap : Array ByteArray) (zl : Zk.Level0)
    (hk0 : 1 ≤ sch.k0) (hl : l.size = L + t.k)
    (hf : ZkTables.Level0.LaneForm resp sch T tb b lay hv rep extra l.size P zm rc) :
    rc = recZ P T (outZ P (cfOf resp L) sch T cap zl t).2.1 (preZ P (cfOf resp L) sch T cap zl t).1 := by
  have hf0 := laneRun_fields P (cfOf resp L) sch.k0 zl.msgs (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).1
    (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).2 sch.k0 le_rfl
  have hk : (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).2.k = t.k + 2 :=
    (head_fields P (cfOf resp L) sch.n (init T) (startZ cap T t)).2.2.2.2.2.1
  rw [hf.record]
  unfold recZ
  congr 1
  · rw [outZ_oods0]
    simp only [cfOf]
    rw [show l.size + 1 = L + (t.k + 1) by omega]
  · show _ = (laneRun P (cfOf resp L) sch.k0 zl.msgs _ _ sch.k0).1.1
    refine Array.ext (by rw [hf0.2.2.2.2.2.2.1]; simp [ZkTables.Level0.laneCh]) fun i h1 h2 => ?_
    have hc := laneRun_ch P (cfOf resp L) sch.k0 zl.msgs (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).1
      (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).2 sch.k0 le_rfl i (by rwa [hf0.2.2.2.2.2.2.1] at h2)
    rw [getElem!_pos _ i h2] at hc
    rw [hc, hk]
    simp only [ZkTables.Level0.laneCh, List.getElem_toArray, List.getElem_map, List.getElem_range, coinKOf, cfOf]
    rw [show l.size + 2 + i = L + (t.k + 2 + i) by omega]
  · show _ = (laneRun P (cfOf resp L) sch.k0 zl.msgs _ _ sch.k0).1.2
    rw [laneRun_rho _ _ _ _ _ _ hk0, hk]
    simp only [ZkTables.Level0.laneRho, coinKOf, cfOf]
    congr 4 <;> omega

open ZkTables.Level0 in
/-- **`ZkTables.Level0.Verifier`**: the verifier's masked Ligerito returns the masked prover's record on its proof, and
ends where the prover did. -/
theorem verifier : Verifier := by
  intro resp h m v T cm0 tb b lay hv rep extra lanePad salt0 salt nonce basisAt L t t' l l' P zl rc R h64 h25 h35
    hv0 hrep htb hb _ hcm hbasis hbs hk hl hrun hag
  obtain ⟨sch, tPad⟩ := v
  obtain ⟨hn, hk0, hL, hsd, hq, hseam, -, -, -, -, hcap, hext⟩ := zkSch_facts h25 h35 hv0
  dsimp only at hn hk0 hL hsd hq hseam hcap hext htb hb hcm hbasis hrun ⊢
  have hnd : ∀ l r v, (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 h : Hashes V)).node l r = .ok v →
      v.size = MerkleScheme.hm96Sha512.digestLen := fun _ _ _ e => hm96_node h64 _ _ e
  have hlf : ∀ row salt v, (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 h : Hashes V)).leaf row salt = .ok v →
      v.size = MerkleScheme.hm96Sha512.digestLen := fun _ _ _ e => hm96_leaf h64 _ _ e
  have hsh := proverShape resp h m (sch, tPad)
    ⟨0, 0, 0, sch, MerkleScheme.hm96Sha512.digestLen,
      if MerkleScheme.hm96Sha512.saltBytes = 0 then none else some MerkleScheme.hm96Sha512.saltBytes, false⟩
    T cm0 tb b lay hv rep extra lanePad salt0 salt nonce t t' l l' P zl rc h64 h25 h35 hv0 rfl rfl rfl htb hb hcm hrun
  obtain ⟨⟨-, -, -, -, -, -, hyr, -, hmsg, hqn, hood, hcbn, hcn⟩, hyb⟩ := hsh
  dsimp only at hyr hmsg hqn hood hcbn hcn hyb
  obtain ⟨hcap0, hrho⟩ := ligeritoPZk_capRho resp _ _ sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce hn hk0
    hrun
  have hrc := rc_eq_recZ cm0.cap ⟨tPad, rep, zl⟩ hk0 hl
    (lanes resp h m (sch, tPad) T cm0 tb b lay hv rep extra lanePad salt nonce t t' l l' P zl rc h25 h35 hv0 htb hb
      hrun)
  obtain ⟨ht', hfin⟩ := honestZ resp h _ T cm0 tb b lay hv rep extra lanePad salt nonce basisAt L t l h25 h35 hv0
    hrep htb hb (commitRowsP_rows hcm) hbasis hl hrun
  have hmo := merkleOkZ_of_opens hnd hlf hL hcap (fun j hj coins => positions_lt_d (hseam j hj) coins)
    (commitPZk_tree hnd hlf (hcap 0 (by omega)) hcm)
    (ligeritoPZk_opens resp h _ sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce L hn hk0 hL hq hl hrun)
  subst ht' hrc
  exact ligeritoWithZ_of _ _ sch T cm0.cap P basisAt ⟨tPad, rep, zl⟩ (cfOf resp L) t hk hag
    (by rw [hcap0]; exact beq_self_eq_true _)
    (by simp [hyb, Zk.EXTRA_LANES, hrep]) hrho hn hk0 hL hsd hq (by omega) (by omega) (by omega) (by omega)
    (by omega) (by rw [hyr, hext]) hbs hmo hfin

end FlockComplete.L0
