import Proofs.Flock.Soundness.Discharge.ZkLink.Compiled
import Proofs.Flock.Soundness.Game.Batch

/-!
# The compiled `--zk` table with its digests read as caps

A Lock's table nodes have one type of table and one type of commitment. The compiled `--zk` table commits three kinds
of object: every level by a cap (`Cap D`), and per rep the pads by a root (`D`) and `τ` by a leaf (`D`). The primed
table (`tabAfterZCP`) receives the pads root and `τ`'s commitment as caps and reads them at entry `0`
(`repZCP`, `innerGameCP`); otherwise it is the compiled table.

`expect_sessAtZ_push`: inside any session, the compiled table and the primed one have the same expectation of every
function of the session's outputs, the primed prover sending the constant cap at every digest. The proof is a Lock
(`Lock (Cap D) D`) between the two (`pushSess`) whose leaves relate equal outputs while every cap sent is constant at
its digest (`Pushed`), which the pushing chooser keeps (`expect_push`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-! ## Pushing digests to caps -/

section Push

variable {α β I : Type}

/-- Every cap fixed so far is the constant cap at its digest. -/
def Pushed (get : I → ℕ → Option (Cap D × D)) (i : I) : Prop :=
  ∀ s c d, get i s = some (c, d) → c 0 = d

/-- **The pushing chooser**: at a digest `d`, the constant cap `d`. -/
def pushC : Lock.Chooser (Cap D) D β := fun _ d _ _ _ => d

/-- **A pushed prover has the same expectations**: along a Lock whose leaves relate equal outputs while every cap is
pushed, the prover that sends the constant cap at every digest has the compiled prover's expectation of every `f`. -/
theorem expect_push {get : I → ℕ → Option (Cap D × D)} (f : α → ℝ) :
    ∀ {i : I} {g g' : Game α} (L : Lock (Cap D) D get (fun i a b => Pushed get i → a = b) i g g')
      (σ' : Strategy g'), Pushed get i → expect f g' σ' = expect f g (Lock.oracleStrat pushC L σ')
  | _, _, _, .ret h, _, hi => by
      simp only [expect]
      rw [h hi]
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi => by
      simp only [expect, Lock.oracleStrat]
      congr 1
      exact Finset.sum_congr rfl fun c _ => expect_push f (L c) (σ' c) hi
  | _, _, _, .send _ L, σ', hi => expect_push f (L σ'.1) σ'.2 hi
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s upd _ k' _ hset hkeep L, σ', hi => by
      simp only [expect, Lock.oracleStrat]
      refine expect_push f _ σ'.2 ?_
      intro s' c d h
      by_cases hs : s' = s
      · subst hs
        rw [hset] at h
        cases h
        rfl
      · rw [hkeep _ _ _ hs] at h
        exact hi s' c d h

theorem pushed_pair {I₁ I₂ : Type} {get₁ : I₁ → ℕ → Option (Cap D × D)} {get₂ : I₂ → ℕ → Option (Cap D × D)}
    (hd : Lock.Disjoint get₁ get₂) {i : I₁ × I₂} (h : Pushed (Lock.getPair get₁ get₂) i) :
    Pushed get₁ i.1 ∧ Pushed get₂ i.2 := by
  refine ⟨fun s c d hs => h s c d (by simp [Lock.getPair, hs]), fun s c d hs => h s c d ?_⟩
  have h0 := hd.right i.1 i.2 s (by rw [hs]; rfl)
  simp [Lock.getPair, h0, hs]

/-- **One table pushed in a batch**, the same game on both sides everywhere else: every output is related. -/
def batchPush {T Dd : Type} {get : I → ℕ → Option (T × Dd)} (P : I → Prop) {i : I} {g g' : Game α}
    (L : Lock T Dd get (fun i a b => P i → a = b) i g g') (post : List (Game α)) :
    (pre : List (Game α)) →
      Lock T Dd get (fun i a b => P i → a = b) i (batch (pre ++ g :: post)) (batch (pre ++ g' :: post))
  | [] => Lock.map (fun x => x.1 :: x.2) (fun x => x.1 :: x.2) (fun _ _ _ h hp => by rw [h.1 hp, h.2])
      (Lock.leftSame L (batch post))
  | h :: pre => Lock.map (fun x => x.1 :: x.2) (fun x => x.1 :: x.2) (fun _ _ _ h' hp => by rw [h'.1, h'.2 hp])
      (Lock.rightSame h (batchPush P L post pre))

end Push

/-! ## The primed table -/

/-- **The inner proof, compiled, with `τ`'s commitment a cap** read at entry `0`. -/
def innerGameCP (H : List UInt8 → D) (Ep : Merkle.Enc (F × F) D) (Lp : Merkle.Leaf (F × F) Sl D H)
    (Lτ : Merkle.Leaf F Sl D H) {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ}
    (rows : Fin nA → Lin F s) (tr : Triple s) (root : D) : Game (InnerOpens F D Sl → Prop) := do
  let ε ← draw F
  let ρσ ← recv (F × F)
  let γ ← draw (Fin (nA + 3) → F)
  let cτ ← recv (Cap D)
  let β ← draw F
  let y ← recv (Fin (s + nP) → F)
  let w ← draw (Fin q → F)
  pure (innerCheckC H Ep Lp Lτ dom pos rows tr root (cτ 0) ε ρσ γ β y w)

/-- **A compiled `--zk` rep with its digests caps**: the pads root a cap read at entry `0`, and `innerGameCP`. -/
noncomputable def repZCP (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ)
    (extra : List (Claim F T.S.m)) : Game (PendZ F D Sl) := do
  let c ← recv (Cap D)
  let o ← repMaskedZC (D := D) A T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r extra
  let v ← innerGameCP Hs.H Hs.Ep Hs.Lp Hs.Lτ T.dom T.np T.nP T.q T.pos o.1 T.tr (c 0)
  pure ⟨o.2, v⟩

/-- **The primed compiled `--zk` table** after its witness cap and the link points: `tabAfterZC` with `repZCP`. -/
noncomputable def tabAfterZCP (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D)
    (pts : Fin 2 → Fin mPts → F) : Game (OutZC F D Sl) :=
  (interleave (repZCP A Hs T 0 (extraClaims T.S T.ptLocal mPts pts))
    (repZCP A Hs T 1 (extraClaims T.S T.ptLocal mPts pts))).bind fun pend =>
    Game.send (OpensZ F D Sl) fun os => Game.ret (outZC Hs T cap₀ pend os)

/-- **A batched session after `Commit`, table `f` at position `pre.length`.** -/
noncomputable def sessAtZ (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)))
    (f : (Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)) : Game (List (OutZC F D Sl)) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts => batch (pre.map (· pts) ++ f pts :: post.map (· pts))

omit [Fintype K] in
theorem sessAfterZC_split (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts))
    (caps : Fin ts.length → Cap D) (j : Fin ts.length) :
    sessAfterZC (gamesZC A Hs ts caps) =
      sessAtZ ((gamesZC A Hs ts caps).take j) ((gamesZC A Hs ts caps).drop (j + 1))
        (tabAfterZC A Hs (ts.get j) (caps j)) := by
  have hs : gamesZC A Hs ts caps = (gamesZC A Hs ts caps).take j ++
      tabAfterZC A Hs (ts.get j) (caps j) :: (gamesZC A Hs ts caps).drop (j + 1) :=
    ofFn_split (fun i => tabAfterZC A Hs (ts.get i) (caps i)) j
  conv_lhs => rw [hs]
  simp only [sessAfterZC, sessAtZ, List.map_append, List.map_cons]

omit [Fintype K] in
theorem take_gamesZC_length (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts))
    (caps : Fin ts.length → Cap D) (j : Fin ts.length) : ((gamesZC A Hs ts caps).take j).length = j := by
  simp [gamesZC, j.isLt.le]

/-! ## The Lock -/

/-- Rep `r`'s pushed digests: the pads root at slot `r`, `τ`'s commitment at slot `2 + r`. -/
def getP (r : ℕ) (i : Option (Cap D × D) × Option (Cap D × D)) (s : ℕ) : Option (Cap D × D) :=
  if s = r then i.1 else if s = 2 + r then i.2 else none

theorem disjoint_getP : Lock.Disjoint (getP 0 : _ → ℕ → Option (Cap D × D)) (getP 1) where
  left i₁ i₂ s h := by
    unfold getP at h ⊢
    by_cases h0 : s = 0
    · subst h0; simp
    · by_cases h2 : s = 2 + 0
      · subst h2; simp
      · simp [h0, h2] at h
  right i₁ i₂ s h := by
    unfold getP at h ⊢
    by_cases h1 : s = 1
    · subst h1; simp
    · by_cases h3 : s = 2 + 1
      · subst h3; simp
      · simp [h1, h3] at h

/-- **The inner proof pushed**: `τ`'s cap is a table of slot `2 + r`. -/
def pushInner (H : List UInt8 → D) (Ep : Merkle.Enc (F × F) D) (Lp : Merkle.Leaf (F × F) Sl D H)
    (Lτ : Merkle.Leaf F Sl D H) {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ}
    (rows : Fin nA → Lin F s) (tr : Triple s) (r : ℕ) (c : Cap D) (d : D) :
    Lock (Cap D) D (getP r) (fun i a b => Pushed (getP r) i → a = b) (some (c, d), none)
      (innerGameCP H Ep Lp Lτ dom s nP q pos rows tr (c 0)) (innerGameC H Ep Lp Lτ dom s nP q pos rows tr d) :=
  .coin F fun ε => .send (F × F) fun ρσ => .coin (Fin (nA + 3) → F) fun γ =>
    .table (2 + r) (fun cτ dτ => (some (c, d), some (cτ, dτ))) (by simp [getP])
      (fun _ _ => by simp [getP])
      (fun _ _ s' hs => by unfold getP; split_ifs <;> simp_all) fun cτ dτ =>
      .coin F fun β => .send (Fin (s + nP) → F) fun y => .coin (Fin q → F) fun w => .ret fun hp => by
        have h1 : c 0 = d := hp r c d (by simp [getP])
        have h2 : cτ 0 = dτ := hp (2 + r) cτ dτ (by simp [getP])
        rw [h1, h2]

/-- **A rep pushed**: the pads root's cap is a table of slot `r`, the masked phase the same game, then the inner
proof pushed. -/
noncomputable def pushRep (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (r : ℕ)
    (extra : List (Claim F T.S.m)) :
    Lock (Cap D) D (getP r) (fun i a b => Pushed (getP r) i → a = b) (none, none)
      (repZCP A Hs T r extra) (repZC A Hs T r extra) :=
  .table r (fun c d => (some (c, d), none)) (by simp [getP]) (fun _ _ => by simp [getP])
    (fun _ _ s' hs => by unfold getP; split_ifs <;> simp_all) fun c d =>
    Lock.bind (R := fun i' a b => i' = (some (c, d), none) ∧ a = b)
      (fun i' o o' h => by
        obtain ⟨rfl, rfl⟩ := h
        exact Lock.bind (fun _ v v' hv => .ret fun hp => by rw [hv hp])
          (pushInner Hs.H Hs.Ep Hs.Lp Hs.Lτ T.dom T.np T.nP T.q T.pos o.1 T.tr r c d))
      (Lock.same (repMaskedZC (D := D) A T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r extra))

/-- **The table pushed**, below its link points: both reps pushed, interleaved, then the same openings. -/
noncomputable def pushTab [Inhabited D] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D)
    (pts : Fin 2 → Fin mPts → F) :
    Lock (Cap D) D (Lock.getPair (getP 0) (getP 1)) (fun i a b => Pushed (Lock.getPair (getP 0) (getP 1)) i → a = b)
      ((none, none), (none, none)) (tabAfterZCP A Hs T cap₀ pts) (tabAfterZC A Hs T cap₀ pts) :=
  Lock.bind (R := Lock.RPair (fun i a b => Pushed (getP 0) i → a = b) (fun i a b => Pushed (getP 1) i → a = b))
    (fun i pend pend' h => .send (OpensZ F D Sl) fun os => .ret fun hp => by
      obtain ⟨h0, h1⟩ := pushed_pair disjoint_getP hp
      rw [Prod.ext (h.1 h0) (h.2 h1)])
    (Lock.interleaveLock disjoint_getP (pushRep A Hs T 0 (extraClaims T.S T.ptLocal mPts pts))
      (pushRep A Hs T 1 (extraClaims T.S T.ptLocal mPts pts)))

/-- **The session pushed**: the table pushed at its position, the same games elsewhere. -/
noncomputable def pushSess [Inhabited D] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) :
    Lock (Cap D) D (Lock.getPair (getP 0) (getP 1)) (fun i a b => Pushed (Lock.getPair (getP 0) (getP 1)) i → a = b)
      ((none, none), (none, none)) (sessAtZ pre post (tabAfterZCP A Hs T cap₀))
      (sessAtZ pre post (tabAfterZC A Hs T cap₀)) :=
  .coin (Fin 2 → Fin mPts → F) fun pts =>
    batchPush _ (pushTab A Hs T cap₀ pts) (post.map (· pts)) (pre.map (· pts))

/-- The primed prover: the compiled prover's moves, the constant cap at every digest. -/
noncomputable def pushStrat [Inhabited D] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)))
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    Strategy (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) :=
  Lock.oracleStrat pushC (pushSess A Hs T cap₀ pre post) σ

omit [Fintype K] in
/-- **The compiled table and the primed one have the same expectations** inside any session, against the primed
prover. -/
theorem expect_sessAtZ_push [Inhabited D] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (f : List (OutZC F D Sl) → ℝ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    expect f (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ =
      expect f (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) (pushStrat A Hs T cap₀ pre post σ) :=
  expect_push f (pushSess A Hs T cap₀ pre post) σ fun s c d h => by
    simp [Lock.getPair, getP] at h

end FlockSoundness.Discharge.ZkLink
