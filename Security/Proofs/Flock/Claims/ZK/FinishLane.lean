import Proofs.Flock.Claims.ZK.InnerTauLig

/-!
# `FinishLane`: level 0's two lane rows on a view's masked values, constants included

`TauLig`'s `lanesA` gives `ligeritoConstraints`' two rows as `rowsOf`'s lane limbs by coefficient. Where level 0's
proof reads the view's masked values (the out-of-domain value, the lane rounds' messages, `e₀`, `e₁`), its target is the
opening's `T` (`openT`) and its restart is `T'`, the two rows are `rowsOf`'s rows 5 and 6 as whole forms.

* **`laneRunR`**, **`laneRowR`**: `laneRun` and `laneRow` are the executable's lane claim (`ZkExec.Rel2`, the constant
  included).
* **`lanesR`**: `ligeritoConstraints`' two rows are `limbRows` of `laneRow` (`ofZk`).
* `LaneAt`: level 0's proof read at the view, `lanesR`'s premises; `LaneCoins`: its record's coins the model's, all
  `TauLig.lanesA` reads for the rows' coefficients.
-/

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf gOf)
namespace Masked

open FlockSoundness (execArith)

local notation "𝕂" => FlockLevel3.GF256

theorem rel2_congr {np : ℕ} {f g : ZkSession.LinK 𝕂 np} {x : _root_.Flock.Zk.Lin2} (h : ZkExec.Rel2 f x)
    (he : g = f) : ZkExec.Rel2 g x :=
  he ▸ h

theorem rel2_masked' {np s : ℕ} (hs : s = np) (w : _root_.Flock.F256) (i₀ i₁ : Fin np) {x₀ x₁ : ℕ}
    (h₀ : i₀.val = x₀) (h₁ : i₁.val = x₁) (hne : x₀ ≠ x₁) :
    ZkExec.Rel2 (ZkSession.cK (gOf w) + ZkSession.padK execArith i₀ i₁) (_root_.Flock.Zk.l2masked w x₀ x₁ s) := by
  subst hs
  exact ZkExec.rel2_masked w i₀ i₁ h₀ h₁ hne

theorem rel2_konst' {np s : ℕ} (hs : s = np) (w : _root_.Flock.F256) :
    ZkExec.Rel2 (ZkSession.cK (gOf w) : ZkSession.LinK 𝕂 np) (_root_.Flock.Zk.l2konst w s) := by
  subst hs
  exact ZkExec.rel2_konst w

/-- Level 0's out-of-domain claim `T + β·(y + h_ood)` on forms, the constant included (`claim0A`'s). -/
theorem claim0R {np s : ℕ} (hs : s = np) (i : Fin np) {idx : ℕ} (hi : i.val = idx) (T y β : _root_.Flock.F128) :
    ZkExec.Rel2 (ZkSession.cK (FlockLevel3.embed (fOf T)) + FlockLevel3.embed (fOf β) •
        (ZkSession.cK (FlockLevel3.embed (fOf y)) + ZkSession.padF i))
      ((_root_.Flock.Zk.Lin.konst T s).add ((_root_.Flock.Zk.Lin.masked y idx s).scale β),
        _root_.Flock.Zk.Lin.konst 0 s) := by
  subst hs
  have h := ZkExec.rel2_lift (ZkExec.rel_add (ZkExec.rel_konst (np := s) T)
    (ZkExec.rel_scale (ZkExec.rel_masked y i hi) β))
  convert h using 1
  refine Prod.ext (funext fun j => ?_) ?_
  · simp only [ZkExec.liftK, ZkSession.cK, ZkSession.padF, ZkSession.cF, ZkSession.mskF, ZkSession.pF, Prod.fst_add,
      Prod.smul_fst, Pi.add_apply, Pi.smul_apply, Pi.zero_apply, smul_eq_mul, zero_add, map_mul]
    by_cases hj : j = i
    · subst hj; simp
    · simp [Pi.single_eq_of_ne hj]
  · simp only [ZkExec.liftK, ZkSession.cK, ZkSession.padF, ZkSession.cF, ZkSession.mskF, ZkSession.pF, Prod.snd_add,
      Prod.smul_snd, smul_eq_mul, add_zero, map_add, map_mul]

/-- **A row read from an executable constraint is it** (`ofZk`), where they agree coefficient by coefficient. -/
theorem ofZk_of_linRel {nE : ℕ} {ℓ : ZkSession.Lin FlockLevel3.GF128 (nE + 4)} {x : _root_.Flock.Zk.Lin}
    (h : ZkExec.LinRel ℓ x) : Composed.ofZk x = ℓ := by
  obtain ⟨-, hc, ha⟩ := h
  cases ℓ
  unfold Composed.ofZk
  congr 1
  · funext i
    exact (ha i).symm
  · exact hc.symm

section Lane

variable {mPts : ℕ} (Tz : ZkSession.TabZK FlockLevel3.GF128 mPts) (st : _root_.Flock.Stmt)
  (c : Composed.CoinsZK (Composed.RestOf Tz st)) (r : Fin 2) {nE : ℕ} (v : Fin nE → FlockLevel3.GF128)

/-- **The model's lane rounds are the executable's**, constants included: after `k ≤ k₀` lane rounds, `laneRun` is the
folded claim (`laneStepE`), where level 0's target is `openT`, its out-of-domain value and lane messages are the view's
masked values, and its coins are the model's. -/
theorem laneRunR {nC : ℕ} (sh : Fin nC → Fin 128 → FlockLevel3.GF128) (hlen : (Composed.layOf Tz st).len = nE + 4)
    (lg : _root_.Flock.LigeritoProof) (rec : _root_.Flock.Zk.L0Rec)
    (hT : fOf rec.target = Composed.openT Tz st c r sh)
    (hβ : ∀ i, fOf rec.ood[0]!.1 = fOf (c.rest.ood r i))
    (hch : ∀ j (hj : j < Tz.sch.k0), rec.challenges[j]! = c.rest.lane r ⟨j, hj⟩)
    (hood : fOf lg.oodValues[0]! = Composed.msgAt v ((Composed.layOf Tz st).lgOod 0))
    (hmsg : ∀ j < Tz.sch.k0,
      gOf lg.messages[j]!.1 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 0))
        (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 1)) ∧
      gOf lg.messages[j]!.2 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 2))
        (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 3))) :
    ∀ k ≤ Tz.sch.k0, ZkExec.Rel2 (Composed.laneRun Tz st c r v sh k)
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
    have hood' : (Composed.layOf Tz st).lgOod 0 < (Composed.layOf Tz st).len :=
      hb (by simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.hab]; omega)
    have h := claim0R hlen _ (ix_val Tz st hlen hood') rec.target lg.oodValues[0]! rec.ood[0]!.1
    refine rel2_congr h ?_
    simp only [Composed.laneRun]
    erw [Fin.sum_univ_one]
    rw [hT, hβ ⟨0, Nat.zero_lt_one⟩, hood]
    all_goals rfl
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
    obtain ⟨hm1, hm2⟩ := hmsg k hk'
    have M01 := rel2_masked' hlen lg.messages[k]!.1 _ _ (ix_val Tz st hlen (hm 0 (by omega)))
      (ix_val Tz st hlen (hm 1 (by omega))) (hne 0 1 (by omega))
    have M23 := rel2_masked' hlen lg.messages[k]!.2 _ _ (ix_val Tz st hlen (hm 2 (by omega)))
      (ix_val Tz st hlen (hm 3 (by omega))) (hne 2 3 (by omega))
    rw [hm1] at M01
    rw [hm2] at M23
    have h := ZkExec.rel2_add (ZkExec.rel2_add M01 (ZkExec.rel2_scale (ZkExec.rel2_add (ih (by omega)) M23)
      rec.challenges[k]!)) (ZkExec.rel2_scale M23 (rec.challenges[k]! * rec.challenges[k]!))
    rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul]
    refine rel2_congr h ?_
    have hc : Composed.laneCh Tz st c r k = gOf rec.challenges[k]! := by
      rw [hch k hk']
      show gOf (Composed.seqOf (c.rest.lane r) k) = _
      rw [Composed.seqOf, dite_eq_left hk']
    simp only [Composed.laneRun, ZkSession.QuadL.eval, ZkSession.fromMsgL, Composed.mkK, hc,
      FlockSoundness.Refine.gOf_mul, pow_two]

/-- **The model's lane row is the executable's `laneLhs`**, constants included: `laneRunR` at `k₀`, the pair's sums
`e₀, e₁` the view's masked values under the model's `ρ`, and the restart `T'` the proof's. -/
theorem laneRowR {nC : ℕ} (sh : Fin nC → Fin 128 → FlockLevel3.GF128) (T' : 𝕂)
    (hlen : (Composed.layOf Tz st).len = nE + 4) (lg : _root_.Flock.LigeritoProof) (rec : _root_.Flock.Zk.L0Rec)
    (zl : _root_.Flock.Zk.Level0Msgs)
    (hT : fOf rec.target = Composed.openT Tz st c r sh)
    (hβ : ∀ i, fOf rec.ood[0]!.1 = fOf (c.rest.ood r i))
    (hch : ∀ j (hj : j < Tz.sch.k0), rec.challenges[j]! = c.rest.lane r ⟨j, hj⟩) (hρ : rec.rho = c.rho r)
    (hood : fOf lg.oodValues[0]! = Composed.msgAt v ((Composed.layOf Tz st).lgOod 0))
    (hmsg : ∀ j < Tz.sch.k0,
      gOf lg.messages[j]!.1 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 0))
        (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 1)) ∧
      gOf lg.messages[j]!.2 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 2))
        (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 3)))
    (he0 : gOf zl.e0 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgE 0 0))
      (Composed.msgAt v ((Composed.layOf Tz st).lgE 0 1)))
    (he1 : gOf zl.e1 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgE 1 0))
      (Composed.msgAt v ((Composed.layOf Tz st).lgE 1 1)))
    (hT' : gOf zl.tPrime = T') :
    ZkExec.Rel2 (Composed.laneRow Tz st c r v sh T') (ZkExec.laneLhs (Composed.layOf Tz st) rec lg zl) := by
  have hb : ∀ {k : ℕ}, k < (Composed.layOf Tz st).hab → k < (Composed.layOf Tz st).len := fun hk => by
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw] at *
    omega
  have he : ∀ i < 2, ∀ t < 2, (Composed.layOf Tz st).lgE i t < (Composed.layOf Tz st).len := fun i hi t ht =>
    hb (by simp only [_root_.Flock.Zk.PadLayout.lgE, _root_.Flock.Zk.PadLayout.hab]; omega)
  have hne : ∀ i, (Composed.layOf Tz st).lgE i 0 ≠ (Composed.layOf Tz st).lgE i 1 := fun i => by
    simp only [_root_.Flock.Zk.PadLayout.lgE]
    omega
  have hcl := laneRunR Tz st c r v sh hlen lg rec hT hβ hch hood hmsg Tz.sch.k0 le_rfl
  have E0 : ZkExec.Rel2 (Composed.mkK v ((Composed.layOf Tz st).lgE 0 0) ((Composed.layOf Tz st).lgE 0 1))
      (_root_.Flock.Zk.l2masked zl.e0 ((Composed.layOf Tz st).lgE 0 0) ((Composed.layOf Tz st).lgE 0 1)
        (Composed.layOf Tz st).len) := by
    have h := rel2_masked' hlen zl.e0 _ _ (ix_val Tz st hlen (he 0 (by omega) 0 (by omega)))
      (ix_val Tz st hlen (he 0 (by omega) 1 (by omega))) (hne 0)
    rw [he0] at h
    exact h
  have E1 : ZkExec.Rel2 (Composed.mkK v ((Composed.layOf Tz st).lgE 1 0) ((Composed.layOf Tz st).lgE 1 1))
      (_root_.Flock.Zk.l2masked zl.e1 ((Composed.layOf Tz st).lgE 1 0) ((Composed.layOf Tz st).lgE 1 1)
        (Composed.layOf Tz st).len) := by
    have h := rel2_masked' hlen zl.e1 _ _ (ix_val Tz st hlen (he 1 (by omega) 0 (by omega)))
      (ix_val Tz st hlen (he 1 (by omega) 1 (by omega))) (hne 1)
    rw [he1] at h
    exact h
  have hK := ZkExec.rel2_neg (rel2_konst' (np := nE + 4) hlen zl.tPrime)
  rw [hT'] at hK
  have h := ZkExec.rel2_add (ZkExec.rel2_add hcl (ZkExec.rel2_scale E0 rec.rho.1))
    (ZkExec.rel2_add (ZkExec.rel2_scale E1 rec.rho.2) hK)
  refine rel2_congr h ?_
  have r0 : Composed.rhoK Tz st c r 0 = gOf rec.rho.1 := by rw [hρ]; exact if_pos rfl
  have r1 : Composed.rhoK Tz st c r 1 = gOf rec.rho.2 := by rw [hρ]; exact if_neg (by decide)
  simp only [Composed.laneRow, Fin.sum_univ_two, r0, r1, sub_eq_add_neg, Fin.val_zero, Fin.val_one, Fin.isValue]
  abel

/-- **Level 0's two lane rows are `rowsOf`'s rows 5 and 6** as whole forms: `limbRows` of the model's lane row, read
from `ligeritoConstraints`' two constraints (`ofZk`). -/
theorem lanesR {nC : ℕ} (sh : Fin nC → Fin 128 → FlockLevel3.GF128) (T' : 𝕂)
    (hlen : (Composed.layOf Tz st).len = nE + 4) (lg : _root_.Flock.LigeritoProof) (rec : _root_.Flock.Zk.L0Rec)
    (zl : _root_.Flock.Zk.Level0Msgs)
    (hT : fOf rec.target = Composed.openT Tz st c r sh)
    (hβ : ∀ i, fOf rec.ood[0]!.1 = fOf (c.rest.ood r i))
    (hch : ∀ j (hj : j < Tz.sch.k0), rec.challenges[j]! = c.rest.lane r ⟨j, hj⟩) (hρ : rec.rho = c.rho r)
    (hood : fOf lg.oodValues[0]! = Composed.msgAt v ((Composed.layOf Tz st).lgOod 0))
    (hmsg : ∀ j < Tz.sch.k0,
      gOf lg.messages[j]!.1 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 0))
        (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 1)) ∧
      gOf lg.messages[j]!.2 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 2))
        (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 3)))
    (he0 : gOf zl.e0 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgE 0 0))
      (Composed.msgAt v ((Composed.layOf Tz st).lgE 0 1)))
    (he1 : gOf zl.e1 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgE 1 0))
      (Composed.msgAt v ((Composed.layOf Tz st).lgE 1 1)))
    (hT' : gOf zl.tPrime = T')
    {lgc : Array (String × _root_.Flock.Zk.Lin)}
    (hlgc : _root_.Flock.Zk.ligeritoConstraints (Composed.layOf Tz st) rec lg zl = .ok lgc) :
    lgc.size = 2 ∧ ∀ k : Fin 2, (Composed.ofZk (lgc[k.val]!).2 : ZkSession.Lin FlockLevel3.GF128 (nE + 4)) =
      ZkSession.limbRows execArith (Composed.laneRow Tz st c r v sh T') k := by
  obtain ⟨h0, h1⟩ := ZkExec.limbRows_rel
    (laneRowR Tz st c r v sh T' hlen lg rec zl hT hβ hch hρ hood hmsg he0 he1 hT')
  rw [FlockComplete.Replay.lanes_eq hlgc rfl]
  refine ⟨rfl, fun k => ?_⟩
  fin_cases k
  · exact ofZk_of_linRel h0
  · exact ofZk_of_linRel h1

/-- **Level 0's proof read at the view** (`lanesR`'s premises): its record's target is the opening's `T`, its coins are
the model's, its out-of-domain value, lane messages and pair's sums are the view's masked values, and its restart is
`T'`. -/
structure LaneAt {nC : ℕ} (sh : Fin nC → Fin 128 → FlockLevel3.GF128) (T' : 𝕂) (lg : _root_.Flock.LigeritoProof)
    (rec : _root_.Flock.Zk.L0Rec) (zl : _root_.Flock.Zk.Level0Msgs) : Prop where
  target : fOf rec.target = Composed.openT Tz st c r sh
  ood : ∀ i, fOf rec.ood[0]!.1 = fOf (c.rest.ood r i)
  lane : ∀ j (hj : j < Tz.sch.k0), rec.challenges[j]! = c.rest.lane r ⟨j, hj⟩
  rho : rec.rho = c.rho r
  oodVal : fOf lg.oodValues[0]! = Composed.msgAt v ((Composed.layOf Tz st).lgOod 0)
  msg : ∀ j < Tz.sch.k0,
    gOf lg.messages[j]!.1 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 0))
      (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 1)) ∧
    gOf lg.messages[j]!.2 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 2))
      (Composed.msgAt v ((Composed.layOf Tz st).lgMsg j 3))
  e0 : gOf zl.e0 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgE 0 0))
    (Composed.msgAt v ((Composed.layOf Tz st).lgE 0 1))
  e1 : gOf zl.e1 = execArith.ofLimbs (Composed.msgAt v ((Composed.layOf Tz st).lgE 1 0))
    (Composed.msgAt v ((Composed.layOf Tz st).lgE 1 1))
  tPrime : gOf zl.tPrime = T'

/-- `lanesR` at `LaneAt`. -/
theorem lanesAt {nC : ℕ} (sh : Fin nC → Fin 128 → FlockLevel3.GF128) (T' : 𝕂)
    (hlen : (Composed.layOf Tz st).len = nE + 4) (lg : _root_.Flock.LigeritoProof) (rec : _root_.Flock.Zk.L0Rec)
    (zl : _root_.Flock.Zk.Level0Msgs) (hL : LaneAt Tz st c r v sh T' lg rec zl)
    {lgc : Array (String × _root_.Flock.Zk.Lin)}
    (hlgc : _root_.Flock.Zk.ligeritoConstraints (Composed.layOf Tz st) rec lg zl = .ok lgc) :
    lgc.size = 2 ∧ ∀ k : Fin 2, (Composed.ofZk (lgc[k.val]!).2 : ZkSession.Lin FlockLevel3.GF128 (nE + 4)) =
      ZkSession.limbRows execArith (Composed.laneRow Tz st c r v sh T') k :=
  lanesR Tz st c r v sh T' hlen lg rec zl hL.target hL.ood hL.lane hL.rho hL.oodVal hL.msg hL.e0 hL.e1 hL.tPrime hlgc

/-- **Level 0's record holds the model's coins**: its out-of-domain coin, lane challenges and pair coins, `LaneAt`'s
coin fields, all `lanesA`'s coefficients read. -/
structure LaneCoins (rec : _root_.Flock.Zk.L0Rec) : Prop where
  ood : ∀ i, fOf rec.ood[0]!.1 = fOf (c.rest.ood r i)
  lane : ∀ j (hj : j < Tz.sch.k0), rec.challenges[j]! = c.rest.lane r ⟨j, hj⟩
  rho : rec.rho = c.rho r

end Lane

end Masked
end Claims.ZKMal
