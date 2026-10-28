import Proofs.Flock.Claims.ZK.SplitInner

/-!
# `InnerTauVal` from its seven rows

`InZ.tau` is `dot (combo γ (cons ++ triple)).a μ` and the real view's `τ` is `Σ_i λ_i μ_i`, with
`λ = batchCoef γ (rowsOf …)` read through `padEquiv`. The two agree term by term once the ten rows do:

* **`tau_of_rows`**: the reduction, for any layout, pads, constraints and coins whose rows, `γ` and `μ` match.
* **`viewTauE_eq`**, **`gammaE`**, **`muE_eq`**: the real view's `τ` unfolds to the batch (`rfl`), the batching coins
  are `czExe`'s at the round after `ε`, and the pads' `μ` by layout index is the tape's by pad.
* **`triple_coef`**, **`tripleE`**: `Zk.triple`'s three rows are `tripleRows`' at the layout's six indices, and at the
  deployed table they are `rowsOf`'s last three on the view's `ρ_in, σ_in` (`innerTripleE`).
* **`TauRows`**: the open part, that `cons` is `rowsOf`'s first seven rows (the replay's five and the Ligerito check's
  two limbs), each of the layout's length.
* **`innerTauVal_of`**: `TauRows → InnerTauVal`.
* **`TauRep`**, **`TauLig`**, **`tauRows_of`**: `TauRows` as its first five rows (the replay's three and the two
  ring-switch rows) and its last two (level 0's lane limbs); `innerTauVal_of_rows : TauRep → TauLig → InnerTauVal`.
-/

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf)
namespace Masked

/-- **The batch, term by term**: where `all`'s ten rows have the layout's length, their coefficients are `R`'s, the
coins are `γ` and the pads' `μ` is `μ` through `padEquiv`, `τ` is `Σ_i batchCoef γ R (padEquiv i) · μ_i`. -/
theorem tau_of_rows {nE : ℕ} (lay : _root_.Flock.Zk.PadLayout) (p : FlockComplete.PadsZk)
    (cons : Array (String × _root_.Flock.Zk.Lin)) (cs : FlockComplete.Replay.InZ.Challenges)
    (γ : Fin (7 + 3) → FlockLevel3.GF128) (R : Fin (7 + 3) → ZkSession.Lin FlockLevel3.GF128 (nE + 4))
    (μ : PadIdx nE → FlockLevel3.GF128) (hlen : lay.len = nE + 4) (hcons : cons.size = 7)
    (hsz : ∀ k < 7 + 3, ((FlockComplete.Replay.InZ.all lay p cons cs)[k]!).2.a.size = lay.len)
    (hR : ∀ (k : Fin (7 + 3)) (i : Fin (nE + 4)),
      fOf ((FlockComplete.Replay.InZ.all lay p cons cs)[k.val]!).2.a[i.val]! = (R k).coef i)
    (hγ : ∀ k : Fin (7 + 3), fOf (cs.cg k.val) = γ k)
    (hμ : ∀ i : PadIdx nE, fOf p.mu[(Composed.padEquiv nE i).val]! = μ i) :
    fOf (FlockComplete.Replay.InZ.tau lay p cons cs) =
      ∑ i, ZkSession.batchCoef γ R (Composed.padEquiv nE i) * μ i := by
  set A := FlockComplete.Replay.InZ.all lay p cons cs with hA
  have hall : A.size = 7 + 3 := by
    simp [hA, FlockComplete.Replay.InZ.all, hcons, _root_.Flock.Zk.triple]
  have hws : ((Array.range A.size).map cs.cg).size = 7 + 3 := by simp [hall]
  have hls : (A.map Prod.snd).size = 7 + 3 := by simp [hall]
  have hlsz : ∀ k < 7 + 3, ((A.map Prod.snd)[k]!).a.size = lay.len := by
    intro k hk
    rw [getElem!_pos _ k (by rw [hls]; exact hk), Array.getElem_map, ← getElem!_pos A k (by rw [hall]; exact hk)]
    exact hsz k hk
  have hcoef : ∀ k : Fin (nE + 4),
      fOf (_root_.Flock.Zk.Lin.combo ((Array.range A.size).map cs.cg) (A.map Prod.snd) lay.len).a[k.val]! =
        ZkSession.batchCoef γ R k := by
    intro k
    rw [ZkExec.combo_a _ _ lay.len (7 + 3) hws hls hlsz k.val (by rw [hlen]; exact k.isLt)]
    unfold ZkSession.batchCoef
    refine Finset.sum_congr rfl fun q _ => ?_
    rw [getElem!_pos _ q.val (by rw [hws]; exact q.isLt), Array.getElem_map, Array.getElem_range, hγ,
      getElem!_pos _ q.val (by rw [hls]; exact q.isLt), Array.getElem_map,
      ← getElem!_pos A q.val (by rw [hall]; exact q.isLt), hR]
  show fOf (_root_.Flock.dot (_root_.Flock.Zk.Lin.combo ((Array.range A.size).map cs.cg) (A.map Prod.snd) lay.len).a
    p.mu) = _
  rw [FlockSoundness.Refine.fOf_dot _ _ lay.len (ZkExec.combo_size _ _ _),
    Fintype.sum_equiv (finCongr hlen)
      (fun i : Fin lay.len => fOf (_root_.Flock.Zk.Lin.combo ((Array.range A.size).map cs.cg) (A.map Prod.snd)
        lay.len).a[i.val]! * fOf p.mu[i.val]!)
      (fun k : Fin (nE + 4) => fOf (_root_.Flock.Zk.Lin.combo ((Array.range A.size).map cs.cg) (A.map Prod.snd)
        lay.len).a[k.val]! * fOf p.mu[k.val]!) (fun _ => rfl),
    Fintype.sum_equiv (Composed.padEquiv nE) (fun i => ZkSession.batchCoef γ R (Composed.padEquiv nE i) * μ i)
      (fun k => ZkSession.batchCoef γ R k * μ ((Composed.padEquiv nE).symm k)) (fun i => by simp)]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [hcoef k, ← hμ ((Composed.padEquiv nE).symm k), Equiv.apply_symm_apply]

/-- Table `j`'s real view at the opened coins `c` (`viewsAt`). -/
noncomputable abbrev viewE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) :=
  (session (fun _ _ => 0) (tableData PE QE UE)).viewsAt x c (wtOf PE x w) τ j

/-- Table `j`'s rep `r` mask `μ` at the tape `τ`, by pad. -/
noncomputable abbrev muE (x : Msg) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) :
    PadIdx (tabC PE x j).z.nE → FlockLevel3.GF128 :=
  fun i => (τ ⟨j.val, lt_of_lt_of_le j.isLt (jOf_le_cap x)⟩).1.2.2.2.2.2.1 r (padEmb' (fits_zkTab PE (zkTab x j)) i)

/-- Table `j`'s rep `r` ten constraint rows (`Composed.rowsOf`) on its real view's messages. -/
noncomputable abbrev rowsE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) : Fin (7 + 3) → ZkSession.Lin FlockLevel3.GF128 ((tabC PE x j).z.nE + 4) :=
  Composed.rowsOf (zkTab x j).table (stAt x j) (czExe PE x c j) r ((viewE x c w τ j).pre.2.1 r)
    (viewE x c w τ j).pre.1 ((viewE x c w τ j).pre.2.2.2.2.2 r).1 ((viewE x c w τ j).pre.2.2.2.2.1 r)

theorem viewTauE_eq (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) :
    fOf (viewTauE x c w τ j r) =
      ∑ i, ZkSession.batchCoef (Composed.innerGamma (zkTab x j).table (stAt x j) (czExe PE x c j) r)
        (rowsE x c w τ j r) (Composed.padEquiv _ i) * muE x τ j r i :=
  rfl

theorem innerLayE_eq (x : Msg) (j : Fin (jOf x)) :
    innerLayE x j = _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0 := by
  have hj := lt_of_jOf j
  have htab : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stAt x j) := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by simpa [FlockDeploy.tabsOf] using hj)]
    simp [FlockDeploy.tabsOf]
    rfl
  show _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
    (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0 = _
  rw [htab]
  rfl

theorem tabE_nE (x : Msg) (j : Fin (jOf x)) :
    (tabC PE x j).z.nE = (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0).hab := by
  show (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0).len - 4 = _
  simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
  omega

theorem innerLayE_len (x : Msg) (j : Fin (jOf x)) : (innerLayE x j).len = (tabC PE x j).z.nE + 4 := by
  rw [tabE_nE, innerLayE_eq]
  rfl

/-- The batching coins `γ` are `czExe`'s, at the round after the rep's `ε` slot. -/
theorem gammaE {x : Msg} {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2) (k : Fin (7 + 3)) :
    fOf ((innerCoinsE c j r).cg k.val) = Composed.innerGamma (zkTab x j).table (stAt x j) (czExe PE x c j) r k := by
  have hj := lt_of_jOf j
  have hr := r.isLt
  have hst : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
  show fOf (slotAt c (repSlot x (2 * j.val + r.val) (innerOff 0) + 1) k.val) =
    fOf (slotAt c (repSlot x (2 * j.val + r.val) (innerOff 1)) k.val)
  congr 2
  unfold repSlot innerOff
  rw [hs]
  simp only [show (2 * j.val + r.val) / 2 = j.val by omega, hst, (zkTab x j).hz]
  omega

theorem padEquiv_val {nE : ℕ} (i : PadIdx nE) : (Composed.padEquiv nE i).val = Composed.padIdx i := by
  rcases i with k | k
  · rfl
  · fin_cases k <;> rfl

/-- The pads' `μ` by layout index is the tape's `μ` by pad (`padsOf`, `padEmb'`). -/
theorem muE_eq (x : Msg) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (i : PadIdx (tabC PE x j).z.nE) :
    fOf (innerPadsE x τ j r).mu[(Composed.padEquiv _ i).val]! = muE x τ j r i := by
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hE64 : (tabC PE x j).z.nE ≤ 2 ^ 64 := (fits_zkTab PE (zkTab x j)).nE
  have hlt := (Composed.padEquiv _ i).isLt
  have hlen := innerLayE_len x j
  rw [padEquiv_val] at hlt ⊢
  have hms : (innerPadsE x τ j r).mu.size = (innerLayE x j).len := by simp [FlockComplete.padsOf]
  rw [getElem!_pos _ _ (by rw [hms, hlen]; exact hlt)]
  have hm : (innerPadsE x τ j r).mu[Composed.padIdx i]'(by rw [hms, hlen]; exact hlt) =
      (UE.tapeZk τ).mu j.val r.val (Composed.padIdx i) := by
    simp [FlockComplete.padsOf]
  rw [hm, tapeMu_eq τ hc r ⟨Composed.padIdx i, by omega⟩]
  rfl

/-- `Zk.triple`'s three rows' coefficients are `tripleRows`' at the layout's six indices. -/
theorem triple_coef {nE : ℕ} (Ly : _root_.Flock.Zk.PadLayout) (hE : Ly.hab = nE) (eps rho sigma : _root_.Flock.F128)
    (q : Fin 3) (i : Fin (nE + 4)) :
    fOf ((_root_.Flock.Zk.triple Ly eps rho sigma)[q.val]!).2.a[i.val]! =
      (ZkSession.tripleRows (Composed.trOf nE Ly) (fOf eps) (fOf rho) (fOf sigma) q).coef i := by
  subst hE
  have h1 := Composed.ZerocheckMasked.fa_lt_fb Ly
  have h2 := Composed.ZerocheckMasked.fb_lt_hab Ly
  have hlen : Ly.len = Ly.hab + 4 := rfl
  have htx : Ly.tx = Ly.hab + 1 := rfl
  have hty : Ly.ty = Ly.hab + 2 := rfl
  have htw : Ly.tw = Ly.hab + 3 := rfl
  have hix : ∀ k (hk : k < Ly.hab + 4), Composed.ix Ly.hab k = ⟨k, hk⟩ := fun k hk => Fin.ext (Nat.mod_eq_of_lt hk)
  have hi := i.isLt
  have hz : ∀ {k}, k < Ly.len → (Array.replicate Ly.len (0 : _root_.Flock.F128)).size > k := by simp
  fin_cases q
  · show fOf (((Array.replicate Ly.len 0).set! Ly.fa eps).set! Ly.tx 1)[i.val]! =
      (Pi.single (Composed.ix Ly.hab Ly.fa) (fOf eps) + Pi.single (Composed.ix Ly.hab Ly.tx) 1 :
        Fin (Ly.hab + 4) → FlockLevel3.GF128) i
    rw [ZkExec.set_get _ _ (by simp; omega) (by simp; omega), ZkExec.set_get _ _ (by simp; omega) (by simp; omega),
      ZkExec.repl_get _ (by omega), hix _ (by omega), hix _ (by omega), Pi.add_apply, Pi.single_apply,
      Pi.single_apply]
    simp only [Fin.ext_iff]
    split_ifs <;> first | omega | simp [FlockSoundness.Refine.fOf_zero, FlockSoundness.Refine.fOf_one]
  · show fOf (((Array.replicate Ly.len 0).set! Ly.fb 1).set! Ly.ty 1)[i.val]! =
      (Pi.single (Composed.ix Ly.hab Ly.fb) 1 + Pi.single (Composed.ix Ly.hab Ly.ty) 1 :
        Fin (Ly.hab + 4) → FlockLevel3.GF128) i
    rw [ZkExec.set_get _ _ (by simp; omega) (by simp; omega), ZkExec.set_get _ _ (by simp; omega) (by simp; omega),
      ZkExec.repl_get _ (by omega), hix _ (by omega), hix _ (by omega), Pi.add_apply, Pi.single_apply,
      Pi.single_apply]
    simp only [Fin.ext_iff]
    split_ifs <;> first | omega | simp [FlockSoundness.Refine.fOf_zero, FlockSoundness.Refine.fOf_one]
  · have hl : ((_root_.Flock.Zk.triple Ly eps rho sigma)[2]!).2.a =
        ((((Array.replicate Ly.len 0).set! Ly.hab eps).set! Ly.tw 1).set! Ly.tx sigma).set! Ly.ty rho := by
      simp only [_root_.Flock.Zk.triple]
      rfl
    have hr : (ZkSession.tripleRows (Composed.trOf Ly.hab Ly) (fOf eps) (fOf rho) (fOf sigma) 2).coef =
        Pi.single (Composed.ix Ly.hab Ly.hab) (fOf eps) + Pi.single (Composed.ix Ly.hab Ly.tw) 1 +
          Pi.single (Composed.ix Ly.hab Ly.tx) (fOf sigma) + Pi.single (Composed.ix Ly.hab Ly.ty) (fOf rho) := rfl
    show fOf ((_root_.Flock.Zk.triple Ly eps rho sigma)[2]!).2.a[i.val]! =
      (ZkSession.tripleRows (Composed.trOf Ly.hab Ly) (fOf eps) (fOf rho) (fOf sigma) 2).coef i
    rw [hl, hr]
    rw [ZkExec.set_get _ _ (by simp; omega) (by simp; omega), ZkExec.set_get _ _ (by simp; omega) (by simp; omega),
      ZkExec.set_get _ _ (by simp; omega) (by simp; omega), ZkExec.set_get _ _ (by simp; omega) (by simp; omega),
      ZkExec.repl_get _ (by omega), hix _ (by omega), hix _ (by omega), hix _ (by omega), hix _ (by omega),
      Pi.add_apply, Pi.add_apply, Pi.add_apply, Pi.single_apply, Pi.single_apply, Pi.single_apply, Pi.single_apply]
    simp only [Fin.ext_iff]
    split_ifs <;> first | omega | simp [FlockSoundness.Refine.fOf_zero, FlockSoundness.Refine.fOf_one]

/-- The inner proof's three triple rows are `rowsOf`'s last three, at the real view's `ρ_in, σ_in`. -/
theorem tripleE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) (q : Fin 3) (i : Fin ((tabC PE x j).z.nE + 4)) :
    fOf ((_root_.Flock.Zk.triple (innerLayE x j) ((innerCoinsE c j r).ce 0)
        (FlockComplete.Replay.InZ.rho (innerLayE x j) (innerPadsE x τ j r) (innerCoinsE c j r))
        (FlockComplete.Replay.InZ.sigma (innerLayE x j) (innerPadsE x τ j r)))[q.val]!).2.a[i.val]! =
      (rowsE x c w τ j r (Fin.natAdd 7 q)).coef i := by
  have hsrc : (srcE x c w τ j.val).rep r.val = repSrcCut PE x (czExe PE x c) (cutE x c w τ) j r := by
    simp only [srcE, tabSrcCut, j.isLt, r.isLt, ↓reduceDIte]
  have hT := innerTripleE x c w τ j r (L := innerSlotE x j r) rfl
  rw [hsrc] at hT
  have hρ : fOf (FlockComplete.Replay.InZ.rho (innerLayE x j) (innerPadsE x τ j r) (innerCoinsE c j r)) =
      ((viewE x c w τ j).pre.2.2.2.2.1 r).1 := by
    rw [← hT.1]
    rfl
  have hσ : fOf (FlockComplete.Replay.InZ.sigma (innerLayE x j) (innerPadsE x τ j r)) =
      ((viewE x c w τ j).pre.2.2.2.2.1 r).2 := by
    rw [← hT.2]
    rfl
  have hε : fOf ((innerCoinsE c j r).ce 0) = Composed.innerEps (zkTab x j).table (stAt x j) (czExe PE x c j) r := rfl
  have hhab : (innerLayE x j).hab = (tabC PE x j).z.nE := by rw [tabE_nE, innerLayE_eq]
  rw [triple_coef _ hhab, hε, hρ, hσ]
  show _ = (Fin.append _ _ (Fin.natAdd 7 q) : ZkSession.Lin FlockLevel3.GF128 _).coef i
  rw [Fin.append_right, innerLayE_eq]
  rfl

theorem triple_size (Ly : _root_.Flock.Zk.PadLayout) (eps rho sigma : _root_.Flock.F128) (q : ℕ) (hq : q < 3) :
    ((_root_.Flock.Zk.triple Ly eps rho sigma)[q]!).2.a.size = Ly.len := by
  interval_cases q <;> simp [_root_.Flock.Zk.triple]

/-- **The seven constraints before the triple**: where the shared phases reach table `j`'s rep `r`'s inner proof, its
replay's five constraints (`c_eval`, the final `a·b`, the lincheck's, the two ring-switch claims) and the Ligerito
check's two limbs are `rowsOf`'s first seven on the real view's messages, each of the layout's length. -/
def TauRows : Prop :=
  ∀ x w τ c n j r cons L, InnerAt x w τ c n j r cons L → innerSlotE x j r + 1 < n →
    cons.size = 7 ∧ (∀ k < 7, (cons[k]!).2.a.size = (innerLayE x j).len) ∧
      ∀ (k : Fin 7) (i : Fin ((tabC PE x j).z.nE + 4)),
        fOf (cons[k.val]!).2.a[i.val]! = (rowsE x c w τ j r (Fin.castAdd 3 k)).coef i

/-- `InnerTauVal` from the seven rows: the batch `dot (combo γ all).a μ` is `Σ_i λ_i μ_i` term by term, the triple's
rows by `tripleE`, `γ` by `gammaE` and `μ` by `muE_eq`. -/
theorem innerTauVal_of (h : TauRows) : InnerTauVal := by
  intro x w τ c n j r cons L hA hn
  obtain ⟨hsz, hasz, hrow⟩ := h x w τ c n j r cons L hA hn
  obtain ⟨-, -, -, s, spec, hs, -⟩ := hA
  apply FlockSoundness.Refine.fOf_injective
  rw [viewTauE_eq]
  refine tau_of_rows (innerLayE x j) (innerPadsE x τ j r) cons (innerCoinsE c j r) _ (rowsE x c w τ j r)
    (muE x τ j r) (innerLayE_len x j) hsz ?_ ?_ (gammaE hs c j r) (muE_eq x τ j r)
  · intro k hk
    unfold FlockComplete.Replay.InZ.all
    by_cases h7 : k < 7
    · rw [ZkExec.append_get!_left _ _ (by omega)]
      exact hasz k h7
    · rw [ZkExec.append_get!_right _ _ (j := k - 7) (by omega) (by simp [_root_.Flock.Zk.triple]; omega)]
      exact triple_size _ _ _ _ _ (by omega)
  · intro k i
    unfold FlockComplete.Replay.InZ.all
    induction k using Fin.addCases with
    | left k =>
      rw [Fin.val_castAdd, ZkExec.append_get!_left _ _ (by omega)]
      exact hrow k i
    | right q =>
      rw [Fin.val_natAdd, ZkExec.append_get!_right _ _ (j := q.val) (by omega)
        (by simp [_root_.Flock.Zk.triple])]
      exact tripleE x c w τ j r q i

/-- **The replay's rows** (`TauRows`' first five): `cons` begins with the replay's three constraints and the two
ring-switch rows, which are `rowsOf`'s first five by coefficient, each of the layout's length. -/
def TauRep : Prop :=
  ∀ x w τ c n j r cons L, InnerAt x w τ c n j r cons L → innerSlotE x j r + 1 < n →
    5 ≤ cons.size ∧ ∀ (k : ℕ) (hk : k < 5), (cons[k]!).2.a.size = (innerLayE x j).len ∧
      ∀ i : Fin ((tabC PE x j).z.nE + 4), fOf (cons[k]!).2.a[i.val]! = (rowsE x c w τ j r ⟨k, by omega⟩).coef i

/-- **Level 0's lane rows** (`TauRows`' last two): `cons` has seven rows, and its last two (`ligeritoConstraints`' two
limbs) are `rowsOf`'s `limbRows` of `laneRow` by coefficient, each of the layout's length. -/
def TauLig : Prop :=
  ∀ x w τ c n j r cons L, InnerAt x w τ c n j r cons L → innerSlotE x j r + 1 < n →
    cons.size = 7 ∧ ∀ (k : ℕ) (hk : 5 ≤ k ∧ k < 7), (cons[k]!).2.a.size = (innerLayE x j).len ∧
      ∀ i : Fin ((tabC PE x j).z.nE + 4), fOf (cons[k]!).2.a[i.val]! = (rowsE x c w τ j r ⟨k, by omega⟩).coef i

theorem tauRows_of (hR : TauRep) (hL : TauLig) : TauRows := by
  intro x w τ c n j r cons L hA hn
  obtain ⟨-, hR⟩ := hR x w τ c n j r cons L hA hn
  obtain ⟨hsz, hL⟩ := hL x w τ c n j r cons L hA hn
  have hrow : ∀ k (hk : k < 7), (cons[k]!).2.a.size = (innerLayE x j).len ∧
      ∀ i : Fin ((tabC PE x j).z.nE + 4), fOf (cons[k]!).2.a[i.val]! = (rowsE x c w τ j r ⟨k, by omega⟩).coef i :=
    fun k hk => if h5 : k < 5 then hR k h5 else hL k ⟨by omega, hk⟩
  exact ⟨hsz, fun k hk => (hrow k hk).1, fun k i => (hrow k.val k.isLt).2 i⟩

/-- **`InnerTauVal` from the replay's rows and level 0's lane rows.** -/
theorem innerTauVal_of_rows (hR : TauRep) (hL : TauLig) : InnerTauVal := innerTauVal_of (tauRows_of hR hL)

end Masked
end Claims.ZKMal
