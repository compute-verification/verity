module

public import Proofs.Flock.Claims.Bridge.Transport
public import Proofs.Flock.Soundness.Discharge.Tables.Sim

/-! Flock's live session as a public-coin verifier. -/

@[expose] public section

namespace Claims.Bridge.Live

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Tables FlockSoundness.Refine

abbrev LCoin (nMax : ℕ) := Fin nMax → FlockLevel3.GF128

noncomputable instance instInhabitedLCoin (nMax : ℕ) : Inhabited (LCoin nMax) := ⟨fun _ => 0⟩

def splitE (nMax n : ℕ) (h : n ≤ nMax) :
    LCoin nMax ≃ (Fin n → FlockLevel3.GF128) × (Fin (nMax - n) → FlockLevel3.GF128) :=
  (Equiv.arrowCongr ((finCongr (by omega)).trans finSumFinEquiv.symm) (Equiv.refl _)).trans
    (Equiv.sumArrowEquivProdArrow _ _ _)

def ptsE (mPts : ℕ) : (Fin (2 * mPts) → FlockLevel3.GF128) ≃ (Fin 2 → Fin mPts → FlockLevel3.GF128) :=
  (Equiv.arrowCongr finProdFinEquiv.symm (Equiv.refl _)).trans (Equiv.curry _ _ _)

structure Wire (nMax J mPts : ℕ) where
  root : Msg → Array ByteArray
  sub : Msg → Option (SubmitJ J)
  sub_le : ∀ a s, sub a = some s → s.n ≤ nMax
  proofs : Msg → Array ByteArray
  pts_le : 2 * mPts ≤ nMax
  encPts : (Fin 2 → Fin mPts → FlockLevel3.GF128) → Msg
  decPts : Msg → Fin 2 → Fin mPts → FlockLevel3.GF128
  decPts_enc : ∀ c, decPts (encPts c) = c
  encRd : (n : ℕ) → (Fin n → FlockLevel3.GF128) → Msg
  decRd : (n : ℕ) → Msg → Fin n → FlockLevel3.GF128
  decRd_enc : ∀ n c, decRd n (encRd n c) = c

def rdIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) (n : ℕ) (h : n ≤ nMax) :
    CoinIf (LCoin nMax) (Fin n → FlockLevel3.GF128) :=
  CoinIf.ofEquiv (splitE nMax n h) (w.encRd n) (w.decRd n) (w.decRd_enc n)

def ptsIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) : CoinIf (LCoin nMax) (Fin 2 → Fin mPts → FlockLevel3.GF128) :=
  CoinIf.ofEquiv ((splitE nMax (2 * mPts) w.pts_le).trans (Equiv.prodCongr (ptsE mPts) (Equiv.refl _))) w.encPts
    w.decPts w.decPts_enc

def branch {J mPts : ℕ} (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → FlockLevel3.GF128)
    (rs : Fin J → Fin 2 → List Flock.Round)
    (K : (s : SubmitJ J) → (Fin s.n → FlockLevel3.GF128) → Game (LiveOutJ J mPts)) :
    Option (SubmitJ J) → Game (LiveOutJ J mPts)
  | none => KLJ rootB pts rs
  | some s => .coin (Fin s.n → FlockLevel3.GF128) (K s)

def roundsK (dg : ByteArray → ByteArray) {J mPts : ℕ} (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) : ℕ → (Fin J → Fin 2 → List Flock.Round) → Game (LiveOutJ J mPts)
  | 0, rs => KLJ rootB pts rs
  | N + 1, rs => .send (Option (SubmitJ J)) (branch rootB pts rs fun s c =>
      roundsK dg rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c)))

def sessionK (dg : ByteArray → ByteArray) (J mPts N : ℕ) : Game (LiveOutJ J mPts) :=
  .send (Array ByteArray) fun rootB => .coin (Fin 2 → Fin mPts → FlockLevel3.GF128) fun pts =>
    roundsK dg rootB pts N fun _ _ => []

theorem bind_rounds (dg : ByteArray → ByteArray) {J mPts : ℕ} (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) :
    ∀ N rs, (liveRoundsJ dg J N rs).bind (KLJ rootB pts) = roundsK dg rootB pts N rs
  | 0, _ => rfl
  | N + 1, rs => by
    show Game.send _ _ = Game.send _ _
    congr 1
    funext o
    cases o with
    | none => rfl
    | some s =>
      show Game.coin _ _ = Game.coin _ _
      congr 1
      funext c
      exact bind_rounds dg rootB pts N _

theorem liveSession_eq (dg : ByteArray → ByteArray) (J mPts N : ℕ) :
    liveSession dg J mPts N = sessionK dg J mPts N := by
  unfold liveSession sessionK
  congr 1
  funext rootB
  congr 1
  funext pts
  exact bind_rounds dg rootB pts N _

def kIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) (rs : Fin J → Fin 2 → List Flock.Round) :
    Iface (LCoin nMax) (KLJ rootB pts rs) :=
  Iface.mkSend w.proofs fun a => Iface.mkRet (LiveOutJ.mk rootB pts rs (w.proofs a))

def subIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) (rs : Fin J → Fin 2 → List Flock.Round)
    (K : (s : SubmitJ J) → (Fin s.n → FlockLevel3.GF128) → Game (LiveOutJ J mPts))
    (next : (s : SubmitJ J) → (c : Fin s.n → FlockLevel3.GF128) → Iface (LCoin nMax) (K s c)) :
    (o : Option (SubmitJ J)) → (∀ s, o = some s → s.n ≤ nMax) → Iface (LCoin nMax) (branch rootB pts rs K o)
  | none, _ => kIf w rootB pts rs
  | some s, h => Iface.mkCoin (rdIf w s.n (h s rfl)) (next s)

def roundsIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) (dg : ByteArray → ByteArray) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) :
    (N : ℕ) → (rs : Fin J → Fin 2 → List Flock.Round) → Iface (LCoin nMax) (roundsK dg rootB pts N rs)
  | 0, rs => kIf w rootB pts rs
  | N + 1, rs => Iface.mkSend w.sub fun a =>
      subIf w rootB pts rs _ (fun _ _ => roundsIf w dg rootB pts N _) (w.sub a) (w.sub_le a)

def sessionIf {nMax J mPts : ℕ} (w : Wire nMax J mPts) (dg : ByteArray → ByteArray) (N : ℕ) :
    Iface (LCoin nMax) (sessionK dg J mPts N) :=
  Iface.mkSend w.root fun a => Iface.mkCoin (ptsIf w) fun pts => roundsIf w dg (w.root a) pts N _

theorem kLe {nMax J mPts : ℕ} (w : Wire nMax J mPts) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) (rs : Fin J → Fin 2 → List Flock.Round)
    (s : RStrat (LCoin nMax) (KLJ rootB pts rs) (kIf w rootB pts rs)) (p : Bool) (n : ℕ) :
    RoundsLe (KLJ rootB pts rs) (kIf w rootB pts rs) s p (n + 1) :=
  trivial

theorem subLe {nMax J mPts : ℕ} (w : Wire nMax J mPts) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) (rs : Fin J → Fin 2 → List Flock.Round)
    (K : (s : SubmitJ J) → (Fin s.n → FlockLevel3.GF128) → Game (LiveOutJ J mPts))
    (next : (s : SubmitJ J) → (c : Fin s.n → FlockLevel3.GF128) → Iface (LCoin nMax) (K s c)) (N : ℕ)
    (ih : ∀ s c (t : RStrat (LCoin nMax) (K s c) (next s c)), RoundsLe (K s c) (next s c) t false (N + 1)) :
    ∀ (o : Option (SubmitJ J)) (ho : ∀ s, o = some s → s.n ≤ nMax)
      (t : RStrat (LCoin nMax) (branch rootB pts rs K o) (subIf w rootB pts rs K next o ho)),
      RoundsLe (branch rootB pts rs K o) (subIf w rootB pts rs K next o ho) t true (N + 1)
  | none, _, t => kLe w rootB pts rs t true N
  | some s, _, t => fun c => ih s c (t c)

theorem roundsLe {nMax J mPts : ℕ} (w : Wire nMax J mPts) (dg : ByteArray → ByteArray) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) :
    ∀ N rs (s : RStrat (LCoin nMax) (roundsK dg rootB pts N rs) (roundsIf w dg rootB pts N rs)),
      RoundsLe (roundsK dg rootB pts N rs) (roundsIf w dg rootB pts N rs) s false (N + 1)
  | 0, rs, s => kLe w rootB pts rs s false 0
  | N + 1, rs, s =>
    subLe w rootB pts rs _ (fun _ _ => roundsIf w dg rootB pts N _) N
      (fun _ _ t => roundsLe w dg rootB pts N _ t) (w.sub s.1) (w.sub_le s.1) s.2

theorem sessionLe {nMax J mPts : ℕ} (w : Wire nMax J mPts) (dg : ByteArray → ByteArray) (N : ℕ)
    (s : RStrat (LCoin nMax) (sessionK dg J mPts N) (sessionIf w dg N)) :
    RoundsLe (sessionK dg J mPts N) (sessionIf w dg N) s false (N + 2) :=
  fun pts => roundsLe w dg (w.root s.1) pts N _ (s.2 pts)

/-- The round digest `roundOf dg` computes is the challenger's live-coin part's, not an oracle query. -/
noncomputable def liveV (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (d : (x : Msg) → LiveOutJ (J x) (mPts x) → OC Bool) :
    Verifier (LCoin nMax) :=
  ofGame (fun x => N x + 2) (fun x => sessionK dg (J x) (mPts x) (N x)) (fun x => sessionIf (w x) dg (N x)) d

noncomputable def liveStrat {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    Strategy (liveSession dg (J x) (mPts x) (N x)) :=
  (liveSession_eq dg (J x) (mPts x) (N x)).symm ▸
    stratG H P (fun x => sessionK dg (J x) (mPts x) (N x)) (fun x => sessionIf (w x) dg (N x)) x ω

noncomputable def liveOut {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (N x + 2) → LCoin nMax) : LiveOutJ (J x) (mPts x) :=
  outcome H P (fun x => N x + 2) (fun x => sessionK dg (J x) (mPts x) (N x)) (fun x => sessionIf (w x) dg (N x)) x ω r

def liveRead {nMax : ℕ} (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray) (w : (x : Msg) → Wire nMax (J x) (mPts x))
    (x : Msg) (msgs : List (Msg × Out)) : Option (LiveOutJ (J x) (mPts x)) :=
  replay ⟨sessionK dg (J x) (mPts x) (N x), sessionIf (w x) dg (N x)⟩ msgs

theorem prob_eqRec {β : Type} {g g' : Game β} (h : g = g') (p : β → Prop) (s : Strategy g) :
    prob p g' (h ▸ s) = prob p g s := by
  subst h
  rfl

open Classical in
theorem Pr_acc_live (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (d : (x : Msg) → LiveOutJ (J x) (mPts x) → OC Bool) (H : Msg → Msg)
    {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (Q : LiveOutJ (J x) (mPts x) → Prop) :
    Pr (fun r : Fin ((liveV nMax J mPts N dg w d).k x) → LCoin nMax =>
        (run H P (liveV nMax J mPts N dg w d) x ω r).out = .acc ∧ Q (liveOut J mPts N dg w H P x ω r)) =
      (prob (fun o => ((d x o).eval H).1 = true ∧ Q o) (liveSession dg (J x) (mPts x) (N x))
        (liveStrat J mPts N dg w H P x ω)).toReal := by
  rw [liveStrat, prob_eqRec, prob_toReal]
  exact Pr_acc_and H P _ _ _ d x ω (sessionLe (w x) dg (N x) _) Q

open Classical in
theorem Pr_acc_live' (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (d : (x : Msg) → LiveOutJ (J x) (mPts x) → OC Bool) (H : Msg → Msg)
    {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    Pr (fun r : Fin ((liveV nMax J mPts N dg w d).k x) → LCoin nMax =>
        (run H P (liveV nMax J mPts N dg w d) x ω r).out = .acc) =
      (prob (fun o => ((d x o).eval H).1 = true) (liveSession dg (J x) (mPts x) (N x))
        (liveStrat J mPts N dg w H P x ω)).toReal := by
  rw [liveStrat, prob_eqRec]
  exact Pr_acc H P _ _ _ d x ω (sessionLe (w x) dg (N x) _)

open Classical in
theorem Pr_acc_replay_live (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (d : (x : Msg) → LiveOutJ (J x) (mPts x) → OC Bool) (H : Msg → Msg)
    {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (Q : Option (LiveOutJ (J x) (mPts x)) → Prop) :
    Pr (fun r : Fin ((liveV nMax J mPts N dg w d).k x) → LCoin nMax =>
        (run H P (liveV nMax J mPts N dg w d) x ω r).out = .acc ∧
          Q (liveRead J mPts N dg w x (run H P (liveV nMax J mPts N dg w d) x ω r).msgs)) =
      (prob (fun o => ((d x o).eval H).1 = true ∧ Q (some o)) (liveSession dg (J x) (mPts x) (N x))
        (liveStrat J mPts N dg w H P x ω)).toReal := by
  rw [liveStrat, prob_eqRec, prob_toReal]
  exact Pr_acc_replay H P _ _ _ d x ω (sessionLe (w x) dg (N x) _) Q

theorem forkPr_live (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (d : (x : Msg) → LiveOutJ (J x) (mPts x) → OC Bool) (H : Msg → Msg)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (hk : 0 < (liveV nMax J mPts N dg w d).k x)
    (j : Fin ((liveV nMax J mPts N dg w d).k x)) :
    forkPr H (liveV nMax J mPts N dg w d) P x hk j =
      Ex fun ω : Ω => forkX (collOf H (d x)) j (sessionK dg (J x) (mPts x) (N x)) (sessionIf (w x) dg (N x))
        (stratOf H P x ω (sessionK dg (J x) (mPts x) (N x)) (sessionIf (w x) dg (N x)) [] none) none :=
  forkPr_ofGame H P _ _ _ d x (fun _ => sessionLe (w x) dg (N x) _) hk j

end Claims.Bridge.Live
