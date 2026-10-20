import Proofs.Flock.Claims.ZK.InnerTauRep

/-!
# `TauLig`: level 0's two lane rows, and `InnerTauVal`

At `InnerAt`, `cons`' last two rows are `ligeritoConstraints`' two limbs of level 0's lane claim (`Replay.lanes_eq`):
the out-of-domain claim folded through `k₀` lane rounds, plus the masked evaluations `e₀`, `e₁` under `ρ`, and `T'`.
The model's `laneRow` builds the same claim over `GF256`, so the two rows are `limbRows` of it by coefficient, once
level 0's record holds `czExe`'s coins (`ligeritoPZk_lanes`, at the rep's slots `repSlot`).

* **`Rel2A`**: a `K`-form's coefficients are a `Lin2`'s (`Rel2` without the constant), through the claim's steps
  (`rel2A_add`, `rel2A_scale`, `rel2A_masked`, `rel2A_konst`, `claim0A`).
* **`laneRunA`**, **`laneRowA`**, **`lanesA`**: `laneRun`, `laneRow` and level 0's two rows by coefficient, at coins
  that are the model's.
* **`tauLig`**: `TauLig`; **`innerTauVal`**: `InnerTauVal` (`innerTauVal_of_tauLig`).
-/

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf gOf)
namespace Masked

open FlockSoundness (execArith)

local notation "𝕂" => FlockLevel3.GF256

/-- A `K`-form's coefficients `f` are a `Lin2`'s, coordinate by coordinate (`Rel2` without the constant). -/
def Rel2A {np : ℕ} (f : Fin np → 𝕂) (x : _root_.Flock.Zk.Lin2) : Prop :=
  x.1.a.size = np ∧ x.2.a.size = np ∧
    ∀ i : Fin np, f i = execArith.ofLimbs (fOf x.1.a[i.val]!) (fOf x.2.a[i.val]!)

theorem rel2A_of {np : ℕ} {ℓ : ZkSession.LinK 𝕂 np} {x : _root_.Flock.Zk.Lin2} (h : ZkExec.Rel2 ℓ x) :
    Rel2A ℓ.1 x :=
  ⟨h.1, h.2.1, h.2.2.2⟩

theorem rel2_ofA {np : ℕ} {f : Fin np → 𝕂} {x : _root_.Flock.Zk.Lin2} (h : Rel2A f x) :
    ZkExec.Rel2 (f, execArith.ofLimbs (fOf x.1.c) (fOf x.2.c)) x :=
  ⟨h.1, h.2.1, rfl, h.2.2⟩

theorem rel2A_congr {np : ℕ} {f g : Fin np → 𝕂} {x : _root_.Flock.Zk.Lin2} (h : Rel2A f x) (he : g = f) :
    Rel2A g x :=
  he ▸ h

theorem rel2A_add {np : ℕ} {f g : Fin np → 𝕂} {x y : _root_.Flock.Zk.Lin2} (ha : Rel2A f x) (hb : Rel2A g y) :
    Rel2A (f + g) (_root_.Flock.Zk.l2add x y) :=
  rel2A_of (ZkExec.rel2_add (rel2_ofA ha) (rel2_ofA hb))

theorem rel2A_scale {np : ℕ} {f : Fin np → 𝕂} {x : _root_.Flock.Zk.Lin2} (ha : Rel2A f x)
    (k : _root_.Flock.F256) :
    Rel2A (gOf k • f) (_root_.Flock.Zk.l2scale x k) :=
  rel2A_of (ZkExec.rel2_scale (rel2_ofA ha) k)

theorem rel2A_masked {np s : ℕ} (hs : s = np) (w : _root_.Flock.F256) (i₀ i₁ : Fin np) {x₀ x₁ : ℕ}
    (h₀ : i₀.val = x₀) (h₁ : i₁.val = x₁) (hne : x₀ ≠ x₁) :
    Rel2A (ZkSession.padK execArith i₀ i₁).1 (_root_.Flock.Zk.l2masked w x₀ x₁ s) := by
  subst hs
  exact rel2A_congr (rel2A_of (ZkExec.rel2_masked w i₀ i₁ h₀ h₁ hne)) (zero_add _).symm

theorem rel2A_konst {np s : ℕ} (hs : s = np) (w : _root_.Flock.F256) :
    Rel2A (0 : Fin np → 𝕂) (_root_.Flock.Zk.l2konst w s) := by
  subst hs
  exact rel2A_of (ZkExec.rel2_konst w)

/-- Level 0's out-of-domain claim `T + β·(y + h_ood)`: its coefficients are `β` at the pad. -/
theorem claim0A {np s : ℕ} (hs : s = np) (i : Fin np) {idx : ℕ} (hi : i.val = idx) (T y β : _root_.Flock.F128) :
    Rel2A (fun j => FlockLevel3.embed (fOf β) * (Pi.single i 1 : Fin np → 𝕂) j)
      ((_root_.Flock.Zk.Lin.konst T s).add ((_root_.Flock.Zk.Lin.masked y idx s).scale β),
        _root_.Flock.Zk.Lin.konst 0 s) := by
  subst hs
  have h := ZkExec.rel2_lift (ZkExec.rel_add (ZkExec.rel_konst (np := s) T)
    (ZkExec.rel_scale (ZkExec.rel_masked y i hi) β))
  refine rel2A_congr (rel2A_of h) (funext fun j => ?_)
  simp only [ZkExec.liftK, ZkSession.cF, ZkSession.mskF, ZkSession.pF, Prod.fst_add, Prod.smul_fst, Pi.add_apply,
    Pi.smul_apply, Pi.zero_apply, smul_eq_mul, zero_add, map_mul]
  by_cases hj : j = i
  · subst hj; simp
  · simp [Pi.single_eq_of_ne hj]

section Lane

variable {mPts : ℕ} (Tz : ZkSession.TabZK FlockLevel3.GF128 mPts) (st : _root_.Flock.Stmt)
  (c : Composed.CoinsZK (Composed.RestOf Tz st)) (r : Fin 2) {nE : ℕ} (v : Fin nE → FlockLevel3.GF128)

theorem ix_val (hlen : (Composed.layOf Tz st).len = nE + 4) {k : ℕ} (hk : k < (Composed.layOf Tz st).len) :
    (Composed.ix nE k).val = k :=
  Nat.mod_eq_of_lt (hlen ▸ hk)

/-- **The model's lane rounds are the executable's** by coefficient: after `k ≤ k₀` lane rounds, `laneRun`'s
coefficients are the folded claim's (`laneStepE`), at lane coins and an out-of-domain coin that are the model's. -/
theorem laneRunA {nC : ℕ} (sh : Fin nC → Fin 128 → FlockLevel3.GF128) (hlen : (Composed.layOf Tz st).len = nE + 4)
    (lg : _root_.Flock.LigeritoProof) (rec : _root_.Flock.Zk.L0Rec)
    (hβ : ∀ i, fOf rec.ood[0]!.1 = fOf (c.rest.ood r i))
    (hch : ∀ j (hj : j < Tz.sch.k0), rec.challenges[j]! = c.rest.lane r ⟨j, hj⟩) :
    ∀ k ≤ Tz.sch.k0, Rel2A (Composed.laneRun Tz st c r v sh k).1
      ((List.range' 0 k).foldl (ZkExec.laneStepE (Composed.layOf Tz st) lg rec.challenges)
        ((_root_.Flock.Zk.Lin.konst rec.target (Composed.layOf Tz st).len).add
          ((_root_.Flock.Zk.Lin.masked lg.oodValues[0]! ((Composed.layOf Tz st).lgOod 0)
            (Composed.layOf Tz st).len).scale rec.ood[0]!.1),
          _root_.Flock.Zk.Lin.konst 0 (Composed.layOf Tz st).len)) := by
  have hb : ∀ {k : ℕ}, k < (Composed.layOf Tz st).hab → k < (Composed.layOf Tz st).len := fun hk => by
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw] at *
    omega
  intro k hk
  induction k with
  | zero =>
    have hood : (Composed.layOf Tz st).lgOod 0 < (Composed.layOf Tz st).len :=
      hb (by simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.hab]; omega)
    refine rel2A_congr (claim0A hlen _ (ix_val Tz st hlen hood) _ _ _) (funext fun j => ?_)
    have e1 : ∀ g : Fin (Composed.layOf Tz st).nOod0 → 𝕂, ∑ i, g i = g ⟨0, Nat.zero_lt_one⟩ :=
      fun g => Fin.sum_univ_one g
    simp only [Composed.laneRun, ZkSession.cK, ZkSession.padF, Prod.fst_add, Prod.fst_sum, Finset.sum_apply,
      Prod.smul_fst, Pi.smul_apply, Pi.add_apply, Pi.zero_apply, zero_add, smul_eq_mul]
    rw [e1, hβ ⟨0, Nat.zero_lt_one⟩]
    rfl
  | succ k ih =>
    have hk' : k < Tz.sch.k0 := hk
    have hm : ∀ t < 4, (Composed.layOf Tz st).lgMsg k t < (Composed.layOf Tz st).len := fun t ht =>
      hb (by
        simp only [_root_.Flock.Zk.PadLayout.lgMsg, _root_.Flock.Zk.PadLayout.hab]
        show _ < _ + _ + 4 * Tz.sch.k0 + 4
        omega)
    have hne : ∀ t t', t ≠ t' → (Composed.layOf Tz st).lgMsg k t ≠ (Composed.layOf Tz st).lgMsg k t' :=
      fun t t' h => by
        simp only [_root_.Flock.Zk.PadLayout.lgMsg]
        omega
    have M01 := rel2A_masked hlen lg.messages[k]!.1 _ _ (ix_val Tz st hlen (hm 0 (by omega)))
      (ix_val Tz st hlen (hm 1 (by omega))) (hne 0 1 (by omega))
    have M23 := rel2A_masked hlen lg.messages[k]!.2 _ _ (ix_val Tz st hlen (hm 2 (by omega)))
      (ix_val Tz st hlen (hm 3 (by omega))) (hne 2 3 (by omega))
    have h := rel2A_add (rel2A_add M01 (rel2A_scale (rel2A_add (ih (by omega)) M23) rec.challenges[k]!))
      (rel2A_scale M23 (rec.challenges[k]! * rec.challenges[k]!))
    rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul]
    refine rel2A_congr h ?_
    have hc : Composed.laneCh Tz st c r k = gOf rec.challenges[k]! := by
      rw [hch k hk']
      show gOf (Composed.seqOf (c.rest.lane r) k) = _
      rw [Composed.seqOf, dite_eq_left hk']
    simp only [Composed.laneRun, ZkSession.QuadL.eval, ZkSession.fromMsgL, Composed.mkK, ZkSession.cK, Prod.fst_add,
      Prod.smul_fst, zero_add, hc, FlockSoundness.Refine.gOf_mul, pow_two]

/-- **The model's lane row is the executable's `laneLhs`** by coefficient, at the model's coins. -/
theorem laneRowA {nC : ℕ} (sh : Fin nC → Fin 128 → FlockLevel3.GF128) (T' : 𝕂)
    (hlen : (Composed.layOf Tz st).len = nE + 4) (lg : _root_.Flock.LigeritoProof) (rec : _root_.Flock.Zk.L0Rec)
    (zl : _root_.Flock.Zk.Level0Msgs) (hβ : ∀ i, fOf rec.ood[0]!.1 = fOf (c.rest.ood r i))
    (hch : ∀ j (hj : j < Tz.sch.k0), rec.challenges[j]! = c.rest.lane r ⟨j, hj⟩) (hρ : rec.rho = c.rho r) :
    Rel2A (Composed.laneRow Tz st c r v sh T').1 (ZkExec.laneLhs (Composed.layOf Tz st) rec lg zl) := by
  have hb : ∀ {k : ℕ}, k < (Composed.layOf Tz st).hab → k < (Composed.layOf Tz st).len := fun hk => by
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw] at *
    omega
  have he : ∀ i < 2, ∀ t < 2, (Composed.layOf Tz st).lgE i t < (Composed.layOf Tz st).len := fun i hi t ht =>
    hb (by simp only [_root_.Flock.Zk.PadLayout.lgE, _root_.Flock.Zk.PadLayout.hab]; omega)
  have hne : ∀ i, (Composed.layOf Tz st).lgE i 0 ≠ (Composed.layOf Tz st).lgE i 1 := fun i => by
    simp only [_root_.Flock.Zk.PadLayout.lgE]
    omega
  have hcl := laneRunA Tz st c r v sh hlen lg rec hβ hch Tz.sch.k0 le_rfl
  have E0 := rel2A_masked hlen zl.e0 _ _ (ix_val Tz st hlen (he 0 (by omega) 0 (by omega)))
    (ix_val Tz st hlen (he 0 (by omega) 1 (by omega))) (hne 0)
  have E1 := rel2A_masked hlen zl.e1 _ _ (ix_val Tz st hlen (he 1 (by omega) 0 (by omega)))
    (ix_val Tz st hlen (he 1 (by omega) 1 (by omega))) (hne 1)
  have h := rel2A_add (rel2A_add hcl (rel2A_scale E0 rec.rho.1))
    (rel2A_add (rel2A_scale E1 rec.rho.2) (rel2A_konst hlen zl.tPrime))
  refine rel2A_congr h ?_
  rw [hρ]
  simp only [Composed.laneRow, Composed.mkK, Composed.rhoK, ZkSession.cK, Prod.fst_add, Prod.fst_sub,
    Prod.smul_fst, Fin.sum_univ_two, zero_add, sub_zero, add_zero]
  rw [ite_eq_left trivial, ite_eq_right (by decide), add_assoc]
  rfl

/-- **Level 0's two lane rows are `limbRows` of the model's lane row** by coefficient. -/
theorem lanesA {nC : ℕ} (sh : Fin nC → Fin 128 → FlockLevel3.GF128) (T' : 𝕂)
    (hlen : (Composed.layOf Tz st).len = nE + 4) (lg : _root_.Flock.LigeritoProof) (rec : _root_.Flock.Zk.L0Rec)
    (zl : _root_.Flock.Zk.Level0Msgs) (hβ : ∀ i, fOf rec.ood[0]!.1 = fOf (c.rest.ood r i))
    (hch : ∀ j (hj : j < Tz.sch.k0), rec.challenges[j]! = c.rest.lane r ⟨j, hj⟩) (hρ : rec.rho = c.rho r)
    {lgc : Array (String × _root_.Flock.Zk.Lin)}
    (hlgc : _root_.Flock.Zk.ligeritoConstraints (Composed.layOf Tz st) rec lg zl = .ok lgc) :
    lgc.size = 2 ∧ ∀ k : Fin 2, (lgc[k.val]!).2.a.size = nE + 4 ∧ ∀ i : Fin (nE + 4),
      fOf (lgc[k.val]!).2.a[i.val]! = (ZkSession.limbRows execArith (Composed.laneRow Tz st c r v sh T') k).coef i := by
  obtain ⟨h1, h2, hA⟩ := laneRowA Tz st c r v sh T' hlen lg rec zl hβ hch hρ
  rw [FlockComplete.Replay.lanes_eq hlgc rfl]
  refine ⟨rfl, fun k => ?_⟩
  fin_cases k
  · refine ⟨h1, fun i => ?_⟩
    show _ = (ZkSession.coords execArith ((Composed.laneRow Tz st c r v sh T').1 i)).1
    rw [hA i, ZkExec.coords_ofLimbs]
    rfl
  · refine ⟨h2, fun i => ?_⟩
    show _ = (ZkSession.coords execArith ((Composed.laneRow Tz st c r v sh T').1 i)).2
    rw [hA i, ZkExec.coords_ofLimbs]
    rfl

end Lane

open FlockComplete Flock.With

/-- **`TauLig`**: at `InnerAt`, `cons` has seven rows, and rows 5 and 6 are `rowsOf`'s lane limbs, each of the
layout's length. The run splits as in `tauRep`; level 0's record holds the transcript's coins at the rep's slots
(`ligeritoPZk_lanes`), which are `czExe`'s (`repSlot_eq`), so `lanesA` gives the rows. -/
theorem tauLig : TauLig := by
  intro x w τ c n j r cons L hA hn
  obtain ⟨hL, hw, hnr, s, spec, hsx, dom, cm0, ic, extras, hex, l, hl, o, t', l', hrun, rfl, -⟩ := hA
  have hj := lt_of_jOf j
  have hr := r.isLt
  unfold innerPreE ioE innerPadsE innerLayE at hrun
  have hrun' := innerPre_cutE (coinOf_respOf c) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ l _ _ hrun
  rw [tabsOf_ofCircuit x j] at hrun' hex
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hL.1 hsx
  have h25 : 25 ≤ (stAt x j).m := (hall j.val hj).1
  have hm14 : 14 ≤ (_root_.Flock.Setup.ofCircuit (stAt x j)).m := by
    show 14 ≤ (stAt x j).m
    omega
  have hmerk : (_root_.Flock.Setup.ofCircuit (stAt x j)).merkle = _root_.Flock.MerkleScheme.hm96Sha512 :=
    (hall j.val hj).2.2
  obtain ⟨full, lL, lg, zl, rec, tI, lI, rp, x', lgc, -, hlL, hlig, hre, hlgc, ho⟩ :=
    FlockComplete.ZkTables.innerPre_split (respOf c) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hm14 _ l hrun'
  rw [ho]
  rw [hmerk] at hlig
  have hz := (zkTab x j).hz
  have hLF := FlockComplete.L0.ligeritoPZk_lanes (respOf c) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    (FlockComplete.L0.zkSchedule_n hz) (FlockComplete.zkSchedule_shape hz).1 hlig
  have h3 := ZkExec.replay_cons_size _ _ _ _ hre
  have hhalf : (stsOf x)[(2 * j.val + r.val) / 2]! = stAt x j := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega, getElem!_pos (stsOf x) j.val hj]
    rfl
  have hz' : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [hhalf]
    exact hz
  have hk6 := (zkTab x j).wf.1.hk6
  have hx : extras.size = spec.points * (stAt x j).regions.size := by
    rw [FlockDeploy.extraClaims_size (st := stAt x j) hex, Nat.mul_comm]
    simp [ptsE]
  have hslot : ∀ f, repSlot x (2 * j.val + r.val) f = lL.size + f (zkTab x j).sch := by
    intro f
    rw [repSlot_eq hsx hz', hlL, hl, repBase_eq hsx, hhalf]
    unfold FlockComplete.piopRoundsZk FlockComplete.ZkTables.piopSends
    have hm' : (_root_.Flock.Setup.ofCircuit (stAt x j)).m = (stAt x j).m := rfl
    have hk' : (_root_.Flock.Setup.ofCircuit (stAt x j)).fold.kLog = (stAt x j).c.kLog := rfl
    rw [hm', hk', hx]
    omega
  have hrec := hLF.record
  have hβ : ∀ i, fOf rec.ood[0]!.1 = fOf ((czExe PE x c j).rest.ood r i) := fun i => by
    rw [hrec]
    show fOf (coinOf (respOf c (lL.size + 1)) 0) = fOf (slotAt c (repSlot x (2 * j.val + r.val) (fun _ => 1)) 0)
    rw [coinOf_respOf, hslot]
  have hch : ∀ i (hi : i < (zkTab x j).table.sch.k0),
      rec.challenges[i]! = (czExe PE x c j).rest.lane r ⟨i, hi⟩ := fun i hi => by
    rw [hrec]
    show (FlockComplete.ZkTables.Level0.laneCh (respOf c) (zkTab x j).sch.k0 lL.size).toArray[i]! =
      kAt c (repSlot x (2 * j.val + r.val) fun _ => 2 + i)
    unfold FlockComplete.ZkTables.Level0.laneCh
    rw [getElem!_pos _ i (by simp only [List.size_toArray, List.length_map, List.length_range]; exact hi)]
    simp only [List.getElem_toArray, List.getElem_map, List.getElem_range, coinKOf, coinOf_respOf, kAt, hslot,
      Nat.add_assoc]
  have hρ : rec.rho = (czExe PE x c j).rho r := by
    rw [hrec]
    show FlockComplete.ZkTables.Level0.laneRho (respOf c) (zkTab x j).sch.k0 lL.size = rhoAt c j.val r.val
    unfold FlockComplete.ZkTables.Level0.laneRho rhoAt rhoSlot
    simp only [coinKOf, coinOf_respOf, hslot]
    rw [show lL.size + 2 + (zkTab x j).sch.k0 = lL.size + ((zkTab x j).sch.k0 + 2) by omega,
      show lL.size + 3 + (zkTab x j).sch.k0 = lL.size + ((zkTab x j).sch.k0 + 2) + 1 by omega]
  have hlen : (Composed.layOf (zkTab x j).table (stAt x j)).len = (tabC PE x j).z.nE + 4 := by
    have h := innerLayE_len x j
    rw [innerLayE_eq] at h
    exact h
  obtain ⟨hsz2, hrow⟩ := lanesA (zkTab x j).table (stAt x j) (czExe PE x c j) r ((viewE x c w τ j).pre.2.1 r)
    (viewE x c w τ j).pre.1 ((viewE x c w τ j).pre.2.2.2.2.2 r).1 hlen lg rec zl hβ hch hρ hlgc
  refine ⟨?_, fun k hk => ?_⟩
  · rw [Array.size_append, Array.size_append, h3, hsz2]
    rfl
  · obtain ⟨k', rfl⟩ : ∃ k' : Fin 2, k = 5 + k'.val := ⟨⟨k - 5, by omega⟩, by simp only; omega⟩
    rw [ZkExec.append_get!_right _ _ (j := k'.val) (by rw [Array.size_append, h3]; rfl) (by rw [hsz2]; exact k'.isLt)]
    obtain ⟨hs, hc⟩ := hrow k'
    refine ⟨hs.trans (innerLayE_len x j).symm, fun i => (hc i).trans ?_⟩
    fin_cases k' <;> rfl

/-- **`τ`'s value**: the session's `τ` over the shared phases' constraints is the real view's. -/
theorem innerTauVal : InnerTauVal :=
  innerTauVal_of_tauLig tauLig

end Masked
end Claims.ZKMal
