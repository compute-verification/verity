module

public import Proofs.Flock.Claims.Complete.Zk.Level0Opens
public import Proofs.Flock.Claims.Complete.Zk.Level0HonestRun
public import Proofs.Flock.Claims.Complete.Zk.Exact
import Proofs.Flock.Claims.Complete.Json.Session

/-! Level 0's verifier premise `ZkTables.Level0.Verifier`, proved (`verifier`). -/

@[expose] public section

namespace FlockComplete.L0

open Flock Transcript FlockComplete Replay Replay.Lg

theorem ligeritoPZk_initialCap (resp : ℕ → Claims.Msg) (hs : Hashes PS) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) {t t' : Transcript}
    {l l' : Array Claims.Msg} {P : LigeritoProof} {zl : Zk.Level0Msgs} {rc : Zk.L0Rec}
    (hrun : ((ligeritoPZk (asksS resp) hs ms sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce).run t).go l =
      (.ok ((P, zl, rc), t'), l')) :
    P.initialCap = cm0.cap := by
  unfold ligeritoPZk at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)]
    at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨⟨caps, rproofs, fin, op0, zl', initCh, rho, b0', y0⟩, stF⟩, t₁⟩, l₁, -, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP3, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  obtain ⟨hP, -⟩ := Prod.mk.inj hP3
  subst hP
  rfl

theorem preZ_rho (resp : ℕ → Claims.Msg) (L : ℕ) (P : LigeritoProof) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (zl : Zk.Level0) (t : Transcript) (hk0 : 1 ≤ sch.k0) :
    (preZ P (cfOf resp L) sch T cap zl t).1.2 = ZkTables.Level0.laneRho resp sch.k0 (L + t.k) := by
  have hk : (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).2.k = t.k + 2 := by
    simp only [head_fields, startZ_k]
  show (laneRun P (cfOf resp L) sch.k0 zl.msgs (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).1
    (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).2 sch.k0).1.2 = _
  rw [laneRun_rho _ _ _ _ _ _ hk0, hk]
  show ((⟨coinOf (resp (L + (t.k + 2 + sch.k0))) 0, coinOf (resp (L + (t.k + 2 + sch.k0))) 1⟩ : F256),
      (⟨coinOf (resp (L + (t.k + 2 + sch.k0 + 1))) 0, coinOf (resp (L + (t.k + 2 + sch.k0 + 1))) 1⟩ : F256)) =
    (coinKOf (resp (L + t.k + 2 + sch.k0)), coinKOf (resp (L + t.k + 3 + sch.k0)))
  rw [show L + (t.k + 2 + sch.k0) = L + t.k + 2 + sch.k0 by omega,
    show L + (t.k + 2 + sch.k0 + 1) = L + t.k + 3 + sch.k0 by omega]
  rfl

theorem preZ_rhoIndependent (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (hn : sch.n ≠ 0)
    (hk0 : 1 ≤ sch.k0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {L : ℕ} {t t' : Transcript}
    {l l' : Array Claims.Msg} {r : LigeritoProof × Zk.Level0Msgs × Zk.L0Rec} (hl : l.size = L + t.k)
    (hrun : ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 tb b lay hv rep tPad extra lanePad salt
      nonce).run t).go l = (.ok (r, t'), l'))
    (P : LigeritoProof) (cap : Array ByteArray) (zl : Zk.Level0) :
    Zk.rhoIndependent (preZ P (cfOf resp L) sch T cap zl t).1.2 = true := by
  rw [preZ_rho resp L P sch T cap zl t hk0, ← hl]
  exact (ligeritoPZk_exact resp h ms sch T cm0 tb b lay hv rep tPad extra lanePad salt nonce hn hk0 hL hq hrun).2

theorem push2_get0 {α : Type} [Inhabited α] {a : Array α} (x y : α) (h : 0 < a.size) :
    ((a.push x).push y)[0]! = a[0]! := by
  rw [push_get_lt (a := a.push x) (j := 0) y (by simp only [Array.size_push]; omega), push_get_lt (j := 0) x h]

theorem step_oods0 (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (i : ℕ) (x : Vars × LState × Transcript)
    (h : 0 < x.2.1.oods.size) :
    0 < (step p cf sch i x).2.1.oods.size ∧ (step p cf sch i x).2.1.oods[0]! = x.2.1.oods[0]! := by
  unfold step
  by_cases hc : i + 1 < sch.levels.size - 1
  · rw [if_pos hc]
    have e : (stepN p cf sch i x).2.1.oods.size = x.2.1.oods.size + 1 + 1 := by
      simp only [stepN, close_fields, query_fields, ood_fields, chals_fields, Array.size_push]
    refine ⟨by rw [e]; omega, ?_⟩
    simp only [stepN, close_fields, query_fields, ood_fields, chals_fields]
    exact push2_get0 _ _ h
  · rw [if_neg hc]
    have e : (stepL p cf sch x).2.1.oods = x.2.1.oods := by
      simp only [stepL, closeLast_fields, query_fields, chals_fields]
    rw [e]
    exact ⟨h, rfl⟩

theorem loop_oods0 (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (x : Vars × LState × Transcript)
    (h : 0 < x.2.1.oods.size) :
    ∀ j, 0 < (loop p cf sch x j).2.1.oods.size ∧ (loop p cf sch x j).2.1.oods[0]! = x.2.1.oods[0]!
  | 0 => ⟨h, rfl⟩
  | j + 1 => by
    obtain ⟨h1, h2⟩ := loop_oods0 p cf sch x h j
    obtain ⟨h3, h4⟩ := step_oods0 p cf sch j (loop p cf sch x j) h1
    rw [show loop p cf sch x (j + 1) = step p cf sch j (loop p cf sch x j) from rfl]
    exact ⟨h3, h4.trans h2⟩

theorem x₀Z_ood0 (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (zl : Zk.Level0) (t : Transcript) :
    0 < (x₀Z p cf sch T cap zl t).2.1.oods.size ∧ ((x₀Z p cf sch T cap zl t).2.1.oods[0]!).beta = cf (t.k + 1) 0 := by
  have h8 := (laneRun_fields p cf sch.k0 zl.msgs (head p cf sch.n (init T) (startZ cap T t)).1
    (head p cf sch.n (init T) (startZ cap T t)).2 sch.k0 le_rfl).2.2.2.2.2.2.2.1
  simp only [x₀Z, preZ, lv0Z, close_fields, query_fields, ood_fields, h8, head_fields, init_fields]
  exact ⟨by simp, rfl⟩

theorem outZ_ood0 (p : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (zl : Zk.Level0) (t : Transcript) : ((outZ p cf sch T cap zl t).2.1.oods[0]!).beta = cf (t.k + 1) 0 := by
  obtain ⟨h0, hb⟩ := x₀Z_ood0 p cf sch T cap zl t
  unfold outZ
  rw [(loop_oods0 p cf sch _ h0 _).2]
  exact hb

theorem preZ_ch (resp : ℕ → Claims.Msg) (L : ℕ) (P : LigeritoProof) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (zl : Zk.Level0) (t : Transcript) :
    (preZ P (cfOf resp L) sch T cap zl t).1.1 = (ZkTables.Level0.laneCh resp sch.k0 (L + t.k)).toArray := by
  have hk : (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).2.k = t.k + 2 := by
    simp only [head_fields, startZ_k]
  have hch := laneRun_ch P (cfOf resp L) sch.k0 zl.msgs (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).1
    (head P (cfOf resp L) sch.n (init T) (startZ cap T t)).2 sch.k0 le_rfl
  rw [hk] at hch
  have hs := preZ_size P (cfOf resp L) sch T cap zl t
  apply Array.ext
  · rw [hs]
    simp [ZkTables.Level0.laneCh]
  · intro i h1 h2
    have hi : i < sch.k0 := by rw [← hs]; exact h1
    rw [← getElem!_pos _ i h1]
    refine (hch i hi).trans ?_
    simp only [ZkTables.Level0.laneCh, List.getElem_toArray, List.getElem_map, List.getElem_range]
    show (⟨coinOf (resp (L + (t.k + 2 + i))) 0, coinOf (resp (L + (t.k + 2 + i))) 1⟩ : F256) =
      coinKOf (resp (L + t.k + 2 + i))
    rw [show L + (t.k + 2 + i) = L + t.k + 2 + i by omega]
    rfl

theorem recZ_eq (resp : ℕ → Claims.Msg) (L : ℕ) (P : LigeritoProof) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (zl : Zk.Level0) (t : Transcript) (hk0 : 1 ≤ sch.k0) :
    recZ P T (outZ P (cfOf resp L) sch T cap zl t).2.1 (preZ P (cfOf resp L) sch T cap zl t).1 =
      { target := T, ood := #[(coinOf (resp (L + t.k + 1)) 0, P.oodValues[0]!)],
        challenges := (ZkTables.Level0.laneCh resp sch.k0 (L + t.k)).toArray,
        rho := ZkTables.Level0.laneRho resp sch.k0 (L + t.k) } := by
  have ho := outZ_ood0 P (cfOf resp L) sch T cap zl t
  unfold recZ
  rw [ho, preZ_ch resp L P sch T cap zl t, preZ_rho resp L P sch T cap zl t hk0]
  show ({ target := T, ood := #[(coinOf (resp (L + (t.k + 1))) 0, P.oodValues[0]!)], challenges := _, rho := _ } :
    Zk.L0Rec) = _
  rw [show L + (t.k + 1) = L + t.k + 1 by omega]

theorem commitPZk_rows {hs : Hashes V} {ms : MerkleScheme} {lv : Level} {tPad : ℕ} {g : Array F128}
    {extra lanePad : ℕ → ℕ → F128} {salt : ℕ → ByteArray} {cm : Commit}
    (h : commitPZk hs ms lv tPad g extra lanePad salt = .ok cm) : cm.rows = encodeRowsZk lv tPad g extra lanePad := by
  unfold commitPZk commitRowsP at h
  obtain ⟨leaves, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨r, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h
  rfl

local instance lawfulBEqByteArrayL0 : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

set_option maxHeartbeats 4000000 in
theorem verifier : ZkTables.Level0.Verifier := by
  intro wideLeaf resp h m v T cm0 tb b lay hv rep extra lanePad salt0 salt nonce basisAt L t t' l l' P zl rc R
    h64 h25 h35 hv0 hrep htb hb _ hcm hbasis hbs htk hl hrun hAg
  obtain ⟨hn, hk0, hL, hsd, hq, hseam, -, -, -, -, hcapd, hext⟩ := zkSch_facts h25 h35 hv0
  have hnode : ∀ l r v, (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 h : Hashes V)).node l r = .ok v →
      v.size = MerkleScheme.hm96Sha512.digestLen := fun _ _ _ e => hm96_node h64 _ _ e
  have hleaf : ∀ row s v, (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 h : Hashes V)).leaf row s = .ok v →
      v.size = MerkleScheme.hm96Sha512.digestLen := fun _ _ _ e => hm96_leaf h64 _ _ e
  obtain ⟨ht', hfin⟩ := honestZ (sch := v.1) (tPad := v.2) resp h MerkleScheme.hm96Sha512 T cm0 tb b lay hv rep extra
    lanePad salt nonce basisAt L t l h25 h35 hv0 hrep htb hb (commitPZk_rows hcm) hbasis hl hrun
  obtain ⟨hlg, hyb⟩ := L0.proverShape wideLeaf resp h m v
    { m := 0, kLog := 0, claims := 0, sched := v.1, hashLen := MerkleScheme.hm96Sha512.digestLen,
      saltLen := if MerkleScheme.hm96Sha512.saltBytes = 0 then none else some MerkleScheme.hm96Sha512.saltBytes }
    T cm0 tb b lay hv rep extra lanePad salt0 salt nonce t t' l l' P zl rc h64 h25 h35 hv0 rfl rfl rfl htb hb hcm hrun
  obtain ⟨-, -, -, -, -, -, hyr, -, hmsg, hqn, hood, hcbn, hcn, -⟩ := hlg
  dsimp only at hyr hmsg hqn hood hcbn hcn
  have hrec := (L0.lanes wideLeaf resp h m v T cm0 tb b lay hv rep extra lanePad salt nonce t t' l l' P zl rc h25 h35
    hv0 htb hb hrun).record
  rw [hl] at hrec
  have hmo := merkleOkZ_of_opens hnode hleaf hL hcapd (fun j hj coins x hx => positions_lt_d (hseam j hj) coins x hx)
    (commitPZk_tree hnode hleaf (hcapd 0 (by omega)) hcm)
    (ligeritoPZk_opens resp h MerkleScheme.hm96Sha512 v.1 T cm0 tb b lay hv rep v.2 extra lanePad salt nonce L hn hk0
      hL hq hl hrun)
  subst ht'
  obtain ⟨hret, hkf⟩ := ligeritoWithZ_of (Hashes.ofSha512 h : Hashes V) MerkleScheme.hm96Sha512 v.1 T cm0.cap P basisAt
    ⟨v.2, rep, zl⟩ (cfOf resp L) t htk hAg
    (by rw [ligeritoPZk_initialCap resp _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hrun]; exact beq_self_eq_true _)
    (by
      have h2 : rep < Zk.EXTRA_LANES / 2 := by unfold Zk.EXTRA_LANES; omega
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
      exact ⟨hyb, h2⟩)
    (preZ_rhoIndependent resp h MerkleScheme.hm96Sha512 v.1 T cm0 tb b lay hv rep v.2 extra lanePad salt nonce hn hk0 hL
      hq hl hrun P cm0.cap ⟨v.2, rep, zl⟩)
    hn hk0 hL hsd hq (by omega) (by omega) (by omega) (by omega) (by omega) (by rw [hyr, hext]) hbs hmo hfin
  rw [recZ_eq resp L P v.1 T cm0.cap ⟨v.2, rep, zl⟩ t hk0, ← hrec] at hret
  exact ⟨hret, hkf⟩

end FlockComplete.L0
