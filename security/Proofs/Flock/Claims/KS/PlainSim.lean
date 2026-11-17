module

public import Proofs.Flock.Claims.KS.PlainWire
public import Proofs.Flock.Claims.KS.PlainSample
public import Proofs.Flock.Claims.KS.ForkCoupling
public import Proofs.Flock.Claims.KS.RoundsN
public import Proofs.Flock.Claims.Recursion.PlainK

/-! The plain simulator `plainSim`, a live prover at the plain wire that runs `P` round by round on coins `fnC` of its tape and the live coins' revealed history. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge.Tables Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

noncomputable def takeE (n : ℕ) (u : LCoin 65536) : Fin n → 𝔽 :=
  fun i => if h : i.val < 65536 then u ⟨i.val, h⟩ else 0

noncomputable def padE (n : ℕ) (c : Fin n → 𝔽) : LCoin 65536 := fun e => if h : e.val < n then c ⟨e.val, h⟩ else 0

theorem padE_takeE (n : ℕ) (u : LCoin 65536) (e : Fin 65536) (he : e.val < n) : padE n (takeE n u) e = u e := by
  simp [padE, takeE, he]

theorem splitE_fst (nMax n : ℕ) (h : n ≤ nMax) (u : LCoin nMax) (i : Fin n) :
    (splitE nMax n h u).1 i = u ⟨i.val, by omega⟩ := by
  simp [splitE, Equiv.arrowCongr_apply, Equiv.sumArrowEquivProdArrow]
  rfl

theorem rdIf_proj {J mPts : ℕ} (w : Wire 65536 J mPts) (n : ℕ) (h : n ≤ 65536) (u : LCoin 65536) :
    (rdIf w n h).proj u = takeE n u := by
  funext i
  show (splitE 65536 n h u).1 i = _
  rw [splitE_fst]
  simp [takeE, show i.val < 65536 by omega]

theorem ptsIf_proj {J mPts : ℕ} (w : Wire 65536 J mPts) (u : LCoin 65536) :
    (ptsIf w).proj u = ptsE mPts (takeE (2 * mPts) u) := by
  show ptsE mPts (splitE 65536 (2 * mPts) w.pts_le u).1 = _
  congr 1
  funext i
  rw [splitE_fst]
  simp [takeE, show i.val < 65536 by have := w.pts_le; omega]

noncomputable def liveH {J mPts : ℕ} (w : Wire 65536 J mPts) :
    (ℕ → Claims.Msg) → (ℕ → SubmitJ J) → (ℕ → LCoin 65536) → ℕ → List (Claims.Msg × Claims.Msg)
  | _, _, _, 0 => []
  | b, S, r, k + 1 => (b 0, w.encRd (S 0).n (takeE (S 0).n (r 0))) ::
      liveH w (fun i => b (i + 1)) (fun i => S (i + 1)) (fun i => r (i + 1)) k

noncomputable def foldS {J : ℕ} (dg : ByteArray → ByteArray) :
    (ℕ → SubmitJ J) → (ℕ → LCoin 65536) → ℕ → (Fin J → Fin 2 → List Round) → Fin J → Fin 2 → List Round
  | _, _, 0, rs => rs
  | S, r, k + 1, rs => foldS dg (fun i => S (i + 1)) (fun i => r (i + 1)) k
      (addRoundJ rs (S 0).tab (S 0).rep (Refine.roundOf dg (S 0).bytes (S 0).n (takeE (S 0).n (r 0))))

theorem playOn_sub {J mPts : ℕ} (w : Wire 65536 J mPts) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Round) (K : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (LiveOutJ J mPts))
    (next : (s : SubmitJ J) → (c : Fin s.n → 𝔽) → Claims.Bridge.Iface (LCoin 65536) (K s c)) {Ω' : Type}
    (H : Claims.Msg → Claims.Msg) (Q : Claims.Prover Ω') (x : Claims.Msg) (ω' : Ω') (o : Option (SubmitJ J))
    (ho : ∀ s, o = some s → s.n ≤ 65536) (s₀ : SubmitJ J) (h : o = some s₀) (hist : List (Claims.Msg × Claims.Msg))
    (a : Claims.Msg) (u : LCoin 65536) (r : ℕ → LCoin 65536) :
    Claims.Bridge.playOn (branch rootB pts rs K o) (subIf w rootB pts rs K next o ho)
        (Claims.Bridge.stratOf H Q x ω' _ _ hist (some a)) (some u) r =
      Claims.Bridge.playOn (K s₀ (takeE s₀.n u)) (next s₀ (takeE s₀.n u))
        (Claims.Bridge.stratOf H Q x ω' _ _ (hist ++ [(a, w.encRd s₀.n (takeE s₀.n u))]) none) none r := by
  subst h
  show Claims.Bridge.playOn (K s₀ ((rdIf w s₀.n (ho s₀ rfl)).proj u)) (next s₀ ((rdIf w s₀.n (ho s₀ rfl)).proj u))
      (Claims.Bridge.stratOf H Q x ω' _ _ (hist ++ [(a, w.encRd s₀.n ((rdIf w s₀.n (ho s₀ rfl)).proj u))]) none)
      none r = _
  rw [rdIf_proj]

theorem playOn_roundsK {J mPts : ℕ} (w : Wire 65536 J mPts) (dg : ByteArray → ByteArray) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) {Ω' : Type} (H : Claims.Msg → Claims.Msg) (Q : Claims.Prover Ω')
    (x : Claims.Msg) (ω' : Ω') :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Round) (hist : List (Claims.Msg × Claims.Msg)) (b : ℕ → Claims.Msg)
      (S : ℕ → SubmitJ J) (r : ℕ → LCoin 65536),
      (∀ k < N, w.sub (b k) = some (S k)) →
      (∀ k ≤ N, ((Q.next x (hist ++ liveH w b S r k) ω').eval H).1 = b k) →
      Claims.Bridge.playOn (roundsK dg rootB pts N rs) (roundsIf w dg rootB pts N rs)
          (Claims.Bridge.stratOf H Q x ω' _ _ hist none) none r =
        ⟨rootB, pts, foldS dg S r N rs, w.proofs (b N)⟩
  | 0, rs, hist, b, S, r, _, hb => by
    have h0 := hb 0 le_rfl
    simp only [liveH, List.append_nil] at h0
    show (⟨rootB, pts, rs, w.proofs ((Q.next x hist ω').eval H).1⟩ : LiveOutJ J mPts) = _
    rw [h0]
    rfl
  | N + 1, rs, hist, b, S, r, hS, hb => by
    have h0 := hb 0 (Nat.zero_le _)
    simp only [liveH, List.append_nil] at h0
    have ih := playOn_roundsK w dg rootB pts H Q x ω' N
      (addRoundJ rs (S 0).tab (S 0).rep (Refine.roundOf dg (S 0).bytes (S 0).n (takeE (S 0).n (r 0))))
      (hist ++ [(b 0, w.encRd (S 0).n (takeE (S 0).n (r 0)))]) (fun i => b (i + 1)) (fun i => S (i + 1))
      (fun i => r (i + 1)) (fun k hk => hS (k + 1) (by omega)) (fun k hk => by
        have := hb (k + 1) (by omega)
        simpa [liveH] using this)
    let K : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (LiveOutJ J mPts) := fun s c =>
      roundsK dg rootB pts N (addRoundJ rs s.tab s.rep (Refine.roundOf dg s.bytes s.n c))
    let nx : (s : SubmitJ J) → (c : Fin s.n → 𝔽) → Claims.Bridge.Iface (LCoin 65536) (K s c) := fun s c =>
      roundsIf w dg rootB pts N (addRoundJ rs s.tab s.rep (Refine.roundOf dg s.bytes s.n c))
    have e : Claims.Bridge.playOn (roundsK dg rootB pts (N + 1) rs) (roundsIf w dg rootB pts (N + 1) rs)
        (Claims.Bridge.stratOf H Q x ω' _ _ hist none) none r =
        Claims.Bridge.playOn (branch rootB pts rs K (w.sub (b 0))) (subIf w rootB pts rs K nx (w.sub (b 0))
          (w.sub_le (b 0))) (Claims.Bridge.stratOf H Q x ω' (branch rootB pts rs K (w.sub (b 0)))
            (subIf w rootB pts rs K nx (w.sub (b 0)) (w.sub_le (b 0))) hist (some (b 0))) (some (r 0))
          (fun i => r (i + 1)) := by
      rw [← h0]
      rfl
    rw [e, playOn_sub w rootB pts rs K nx H Q x ω' _ _ (S 0) (hS 0 (by omega))]
    exact ih

theorem liveOut_known {mPts : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : Claims.Msg → Wire 65536 1 mPts) {Ω' : Type} (H : Claims.Msg → Claims.Msg) (Q : Claims.Prover Ω')
    (x : Claims.Msg) (ω' : Ω') (r' : Fin (N x + 2) → LCoin 65536) (b : ℕ → Claims.Msg) (S : ℕ → SubmitJ 1)
    (hS : ∀ k < N x, (w x).sub (b k) = some (S k))
    (hb : ∀ k ≤ N x, ((Q.next x ([(((Q.next x [] ω').eval H).1,
        (w x).encPts ((ptsIf (w x)).proj (Claims.coinSeq r' 0)))] ++
          liveH (w x) b S (fun i => Claims.coinSeq r' (i + 1)) k) ω').eval H).1 = b k) :
    liveOut (fun _ => 1) (fun _ => mPts) N dg w H Q x ω' r' =
      ⟨(w x).root ((Q.next x [] ω').eval H).1, (ptsIf (w x)).proj (Claims.coinSeq r' 0),
        foldS dg S (fun i => Claims.coinSeq r' (i + 1)) (N x) (fun _ _ => []), (w x).proofs (b (N x))⟩ :=
  playOn_roundsK (w x) dg _ _ H Q x ω' (N x) _ _ b S _ hS hb

def plainN (x : Claims.Msg) : ℕ := (V false).k x - 2

def plainNsOne (spec : SessionSpec) (st : Stmt) : List ℕ := plainRepNs spec st ++ plainRepNs spec st

/-- It is at least one, so one at the last message, whose coin the session never reads. -/
def plainCnt (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (i : ℕ) : ℕ :=
  max 1 (if i = 0 then 2 * spec.mPts else if i ≤ plainN x then min ((plainNsOne spec st).getD (i - 1) 0) 65536 else 0)

def plainNOf (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (i : Fin ((V false).k x)) : ℕ := plainCnt x spec st i.val

theorem plainNOf_pos (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (i : Fin ((V false).k x)) :
    1 ≤ plainNOf x spec st i :=
  le_max_left _ _

abbrev plainD (x : Claims.Msg) : Type := Fin ((V false).k x) → Fin 65536 → Option 𝔽

abbrev plainTape (x : Claims.Msg) : Type := Fin ((V false).k x) → plainD x → LCoin 65536

def plainKey (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) :
    Fin ((V false).k x) → (Fin ((V false).k x) → LCoin 65536) → plainD x :=
  Claims.Flock.FnFork.histKey (plainNOf x spec st)

noncomputable def plainΦ (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) :
    plainTape x × (Fin (plainN x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin :=
  if hL : plainN x + 2 = (V false).k x then fun q => fnAt hL (plainNOf x spec st) (plainKey x spec st) q.1 q.2
  else fun _ => default

theorem plainΦ_eq {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (hL : plainN x + 2 = (V false).k x) :
    plainΦ x spec st = fun q => fnAt hL (plainNOf x spec st) (plainKey x spec st) q.1 q.2 :=
  dite_eq_left hL

theorem Pr_plainΦ {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (hL : plainN x + 2 = (V false).k x)
    (E : (Fin ((V false).k x) → Coin) → Prop) :
    Claims.Pr (fun p => E (plainΦ x spec st p)) = Claims.Pr E := by
  rw [plainΦ_eq hL]
  exact Pr_fnAt hL (plainNOf x spec st) (plainKey x spec st) E

noncomputable def reveal (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (i : ℕ) (m : Claims.Msg) : LCoin 65536 :=
  if i = 0 then padE (2 * spec.mPts) ((ptsE spec.mPts).symm (finDecode m))
  else padE (plainCnt x spec st i) (finDecode m)

noncomputable def plainRecon (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (hl : List (Claims.Msg × Claims.Msg)) :
    Fin ((V false).k x) → LCoin 65536 :=
  fun i => if i.val < hl.length then reveal x spec st i.val (hl.getD i.val default).2 else fun _ => 0

noncomputable def simCoins (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (u : plainTape x)
    (hl : List (Claims.Msg × Claims.Msg)) : Fin ((V false).k x) → Coin :=
  fnC (plainNOf x spec st) (plainKey x spec st) u (plainRecon x spec st hl)

def depHist {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (c : ℕ → Coin) : ℕ → List (Claims.Msg × Claims.Msg)
  | 0 => []
  | j + 1 => depHist H P x ω c j ++ [(((P.next x (depHist H P x ω c j) ω).eval H).1, (c j).bytes)]

def depMsg {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (c : ℕ → Coin) (j : ℕ) : Claims.Msg :=
  ((P.next x (depHist H P x ω c j) ω).eval H).1

def plainRep (spec : SessionSpec) (st : Stmt) (i : ℕ) : ℕ := if i - 1 < (plainRepNs spec st).length then 0 else 1

def plainRound (rep n : ℕ) (a : Claims.Msg) : Claims.Msg :=
  2 :: UInt8.ofNat rep :: UInt8.ofNat (n % 256) :: UInt8.ofNat (n / 256 % 256) :: UInt8.ofNat (n / 65536) :: a.tail

noncomputable def plainMsg {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (u : plainTape x) (hl : List (Claims.Msg × Claims.Msg)) : Claims.Msg :=
  if hl.length = 0 then depMsg sha512 P x ω (Claims.coinSeq (simCoins x spec st u hl)) 0
  else if hl.length ≤ plainN x then
    plainRound (plainRep spec st hl.length) (plainCnt x spec st hl.length)
      (depMsg sha512 P x ω (Claims.coinSeq (simCoins x spec st u hl)) hl.length)
  else if (Claims.run sha512 P (V false) x ω (simCoins x spec st u hl)).out = .acc then
    depMsg sha512 P x ω (Claims.coinSeq (simCoins x spec st u hl)) hl.length
  else []

noncomputable def plainMsgK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (u : plainTape x) (hl : List (Claims.Msg × Claims.Msg)) : Claims.Msg :=
  if hl.length = 0 then depMsg sha512 P x ω (Claims.coinSeq (simCoins x spec st u hl)) 0
  else if hl.length ≤ plainN x then
    plainRound (plainRep spec st hl.length) (plainCnt x spec st hl.length)
      (depMsg sha512 P x ω (Claims.coinSeq (simCoins x spec st u hl)) hl.length)
  else if (Claims.run sha512 P (VK keep) x ω (simCoins x spec st u hl)).out = .acc then
    depMsg sha512 P x ω (Claims.coinSeq (simCoins x spec st u hl)) hl.length
  else []

noncomputable def plainSim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) :
    Claims.Prover (Ω × plainTape x) :=
  ⟨fun _ hl p => .ret (plainMsg P x spec st p.1 p.2 hl)⟩

noncomputable def plainSimK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) :
    Claims.Prover (Ω × plainTape x) :=
  ⟨fun _ hl p => .ret (plainMsgK keep P x spec st p.1 p.2 hl)⟩

def plainDg (spec : SessionSpec) : ByteArray → ByteArray := spec.dg

noncomputable def plainW (spec : SessionSpec) (h : 2 * spec.mPts ≤ 65536) : Claims.Msg → Wire 65536 1 spec.mPts :=
  fun _ => plainWire spec.mPts h

theorem plainSim_firstK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (u : plainTape x) :
    (((plainSimK keep P x spec st).next x [] (ω, u)).eval sha512).1 = ((P.next x [] ω).eval sha512).1 := by
  simp only [plainSimK, Claims.OC.eval, plainMsgK]
  rfl

theorem plainSim_rootK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (h : 2 * spec.mPts ≤ 65536) (u : plainTape x) :
    (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec h) sha512 (plainSimK keep P x spec st) x
      (ω, u)).1 = plainRoot P x ω :=
  plainRoot_of_first plainN (plainDg spec) (fun _ => rfl) (plainSim_firstK keep P x spec st ω) u

theorem plainSim_root {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (h : 2 * spec.mPts ≤ 65536) (u : plainTape x) :
    (liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec h) sha512 (plainSim P x spec st) x
      (ω, u)).1 = plainRoot P x ω :=
  plainSim_rootK id (Ω := Ω) (P := P) (x := x) (spec := spec) (st := st) (ω := ω) (h := h) (u := u)

end FlockDeploy.KS
