import Proofs.Flock.Claims.ZK.PrefixLive

/-!
# A cut rep stopped at body `n` is the one stopped later, up to its stop (`CutPrefix.repCut_cut_agree`)

`repCut` answered by `asksCut coin n` and by `asksCut coin N`, `n ≤ N`, from a history of at most `n` bodies, are
coupled (`PrefixAgree.Agree n`): the same end, at no more than `n` bodies, or the run stopped at `n` threw right after
body `n` and the later run's bodies extend it. Both answer each squeeze below `n` with the same coin, so a squeeze keeps
the coupling at any size (`squeezeP_cut_agree`) and the walk needs none of `PrefixLive`'s size bounds.

* **`repCut_cut_agree`**: the coupling at a rep.
* `prefix_of_agree`, `eq_of_agree`: where the later run returns past body `n`, the run stopped at `n` threw with the
  later run's first `n + 1` bodies; where it returns by body `n`, the two runs are one.
* **`repCut_shift`**: a cut rep reads only its history's size; from two histories of one size it ends the same way after
  the same bodies (`Shift`, `go_of_shift`).
-/

namespace FlockComplete.CutPrefix

open Flock Transcript FlockComplete FlockDeploy With JsonRT PrefixAgree

/-! ## What the coupling gives -/

/-- **Where the later run returns past body `n`**, the run stopped at `n` threw, with its bodies the later run's first
`n + 1`. -/
theorem prefix_of_agree {n : ℕ} {α : Type} {x y : PS α} (h : Agree n x y) {l : Array Claims.Msg} (hl : l.size ≤ n)
    {a : α} {l' : Array Claims.Msg} (hx : x.go l = (.ok a, l')) (hn : n < l'.size) :
    (∃ e, (y.go l).1 = .error e) ∧ (y.go l).2.size = n + 1 ∧ ∀ i < n + 1, (y.go l).2[i]! = l'[i]! := by
  rcases h.2 l hl with ⟨-, hs⟩ | ⟨he, hs, bs, hbs⟩
  · rw [hx] at hs
    dsimp only at hs
    omega
  · refine ⟨he, hs, fun i hi => ?_⟩
    have hi' : i < (y.go l).2.size := by omega
    rw [hx] at hbs
    dsimp only at hbs
    subst hbs
    rw [getElem!_pos _ i hi', getElem!_pos _ i (by rw [Array.size_append]; omega), Array.getElem_append_left hi']

/-- `prefix_of_agree` with the stopped run's end named. -/
theorem prefix_of_agree' {n : ℕ} {α : Type} {x y : PS α} (h : Agree n x y) {l : Array Claims.Msg} (hl : l.size ≤ n)
    {a : α} {l' : Array Claims.Msg} (hx : x.go l = (.ok a, l')) (hn : n < l'.size) :
    ∃ e L, y.go l = (.error e, L) ∧ L.size = n + 1 ∧ ∀ i < n + 1, L[i]! = l'[i]! := by
  obtain ⟨⟨e, he⟩, hs, hi⟩ := prefix_of_agree h hl hx hn
  exact ⟨e, (y.go l).2, Prod.ext he rfl, hs, hi⟩

/-- **Where the later run returns by body `n`**, the run stopped at `n` is the same run. -/
theorem eq_of_agree {n : ℕ} {α : Type} {x y : PS α} (h : Agree n x y) {l : Array Claims.Msg} (hl : l.size ≤ n)
    {a : α} {l' : Array Claims.Msg} (hx : x.go l = (.ok a, l')) (hn : l'.size ≤ n) : y.go l = (.ok a, l') := by
  rcases h.2 l hl with ⟨he, -⟩ | ⟨-, hs, bs, hbs⟩
  · rw [← he, hx]
  · rw [hx] at hbs
    dsimp only at hbs
    have e := congrArg Array.size hbs
    rw [Array.size_append] at e
    omega

/-! ## The walk -/

section Walk

variable (coin : ℕ → ℕ → F128) {n N : ℕ} (hnN : n ≤ N)

include hnN

/-- **A squeeze keeps the coupling**: below `n` both answer with `coin`, and at `n` the stopped run throws. -/
theorem squeezeP_cut_agree {kind : UInt8} {m : ℕ} :
    ((agreeS n).st Transcript).R (squeezeP (asksCut coin N) kind m) (squeezeP (asksCut coin n) kind m) := by
  intro t
  unfold squeezeP
  dsimp only
  split
  · exact (agreeS n).pure _
  · refine ⟨fun l => ⟨#[Claims.toMsg (framed t kind m)], ?_⟩, fun l hl => ?_⟩
    · rw [go_bind, cut_round_go]
      by_cases hN : N ≤ l.size
      · simp only [hN, ↓reduceIte]
        rfl
      · simp only [hN, ↓reduceIte]
        rfl
    · rw [go_bind, go_bind, cut_round_go, cut_round_go]
      by_cases hn : n ≤ l.size
      · simp only [hn, ↓reduceIte]
        refine .inr ⟨⟨_, rfl⟩, by simp; omega, #[], ?_⟩
        by_cases hN : N ≤ l.size
        · simp [hN]
        · simp [hN, go_pure]
      · simp only [hn, ↓reduceIte, show ¬ N ≤ l.size by omega]
        exact .inl ⟨rfl, by simp [go_pure]; omega⟩

theorem sampleP_cut_agree :
    ((agreeS n).st Transcript).R (sampleP (asksCut coin N)) (sampleP (asksCut coin n)) := by
  unfold sampleP
  sim_walk [squeezeP_cut_agree coin hnN]

theorem sampleKP_cut_agree :
    ((agreeS n).st Transcript).R (sampleKP (asksCut coin N)) (sampleKP (asksCut coin n)) := by
  unfold sampleKP
  sim_walk [squeezeP_cut_agree coin hnN]

theorem challengeP_cut_agree (split : Bool) :
    (((agreeS n).st Transcript).st LigState).R (challengeP (asksCut coin N) split)
      (challengeP (asksCut coin n) split) := by
  unfold challengeP
  sim_walk [liftT_agree, sampleKP_cut_agree coin hnN, sendMsg_agree]

theorem oodIntroP_cut_agree (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    (((agreeS n).st Transcript).st LigState).R (oodIntroP (asksCut coin N) nonce g dim)
      (oodIntroP (asksCut coin n) nonce g dim) := by
  unfold oodIntroP sampleVecP
  sim_walk [liftT_agree, squeezeP_cut_agree coin hnN, sampleP_cut_agree coin hnN, sendMsg_agree, powP_agree]

theorem queryP_cut_agree (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    (((agreeS n).st Transcript).st LigState).R (queryP (asksCut coin N) nonce ms lv cm)
      (queryP (asksCut coin n) nonce ms lv cm) := by
  unfold queryP sampleVecP
  sim_walk [liftT_agree, squeezeP_cut_agree coin hnN, powP_agree]

theorem zerocheckCut_cut_agree (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout) (mk : ℕ → F128) :
    ((agreeS n).st Transcript).R (zerocheckCut (asksCut coin N) m digest cap root lay mk)
      (zerocheckCut (asksCut coin n) m digest cap root lay mk) := by
  unfold zerocheckCut sampleVecP
  sim_walk [squeezeP_cut_agree coin hnN, sampleP_cut_agree coin hnN]

theorem lincheckCut_cut_agree (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (mk : ℕ → F128) (zc : ZcOut) :
    ((agreeS n).st Transcript).R (lincheckCut (asksCut coin N) c m lay mk zc)
      (lincheckCut (asksCut coin n) c m lay mk zc) := by
  unfold lincheckCut
  sim_walk [sampleP_cut_agree coin hnN]

theorem ringSwitchCut_cut_agree (sv : ℕ → Array F128) (claims : Array Claim) :
    ((agreeS n).st Transcript).R (ringSwitchCut (asksCut coin N) sv claims)
      (ringSwitchCut (asksCut coin n) sv claims) := by
  unfold ringSwitchCut sampleVecP
  sim_walk [squeezeP_cut_agree coin hnN]

theorem piopCut_cut_agree (st : Setup) (lay : Zk.PadLayout) (src : RepSrc) (cap root : ByteArray)
    (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ) :
    ((agreeS n).st Transcript).R (piopCut (asksCut coin N) st lay src cap root extras prior words)
      (piopCut (asksCut coin n) st lay src cap root extras prior words) := by
  unfold piopCut
  sim_walk [zerocheckCut_cut_agree coin hnN, lincheckCut_cut_agree coin hnN, rankCheckP_agree,
    ringSwitchCut_cut_agree coin hnN]

set_option maxHeartbeats 1000000 in
theorem ligeritoCut_cut_agree (ms : MerkleScheme) (sch : Schedule) (T : F128) (cm0 : Commit) (b : Array F128)
    (lay : Zk.PadLayout) (src : RepSrc) :
    ((agreeS n).st Transcript).R (ligeritoCut (asksCut coin N) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 b lay src)
      (ligeritoCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 b lay src) := by
  unfold ligeritoCut sampleVecP
  sim_walk [liftT_agree, squeezeP_cut_agree coin hnN, sampleP_cut_agree coin hnN, sampleKP_cut_agree coin hnN,
    sendMsg_agree, powP_agree, challengeP_cut_agree coin hnN, oodIntroP_cut_agree coin hnN,
    queryP_cut_agree coin hnN, commitP_agree Sha512.hash, agree_pure_st]

theorem innerCut_cut_agree (lay : Zk.PadLayout) (ic : InnerCommit) (finalC : F128) (src : RepSrc)
    (cons : Array (String × Zk.Lin)) :
    ((agreeS n).st Transcript).R (innerCut (asksCut coin N) lay ic finalC src cons)
      (innerCut (asksCut coin n) lay ic finalC src cons) := by
  unfold innerCut sampleVecP
  sim_walk [sampleP_cut_agree coin hnN, squeezeP_cut_agree coin hnN]

/-- **A cut rep stopped at body `n` is coupled with the one stopped at `N ≥ n`.** -/
theorem repCut_cut_agree (st : Setup) (sch : Schedule) (rep : ℕ) (dm : ByteArray) (cm0 : Commit)
    (extras : Array Claim) (ic : InnerCommit) (src : RepSrc) (prior : Array (List F128)) (words : List ℕ) :
    ((agreeS n).st Transcript).R
      (repCut (asksCut coin N) (Hashes.ofSha512 Sha512.hash) st sch rep dm cm0 extras ic src prior words)
      (repCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch rep dm cm0 extras ic src prior words) := by
  unfold repCut
  sim_walk [piopCut_cut_agree coin hnN, ligeritoCut_cut_agree coin hnN, innerCut_cut_agree coin hnN]

end Walk

/-! ## The run from another history of the same size -/

section Shift

variable (coin : ℕ → ℕ → F128)

/-- **Two runs from histories of one size**: the same end, after the same bodies. -/
def Shift {α : Type} (x y : PS α) : Prop :=
  ∀ l l' : Array Claims.Msg, l.size = l'.size → ∃ r bs, x.go l = (r, l ++ bs) ∧ y.go l' = (r, l' ++ bs)

theorem Shift.still {α : Type} {v : Except String α} {y : PS α} (h : ∀ l, y.go l = (v, l)) : Shift y y :=
  fun l l' _ => ⟨v, #[], by rw [h l, Array.append_empty], by rw [h l', Array.append_empty]⟩

theorem Shift.bind {α β : Type} {x y : PS α} {f g : α → PS β} (hxy : Shift x y) (hfg : ∀ a, Shift (f a) (g a)) :
    Shift (x >>= f) (y >>= g) := by
  intro l l' hs
  obtain ⟨r, bs, hx, hy⟩ := hxy l l' hs
  rw [go_bind, go_bind, hx, hy]
  cases r with
  | error e => exact ⟨.error e, bs, rfl, rfl⟩
  | ok a =>
    obtain ⟨r', bs', hx', hy'⟩ := hfg a (l ++ bs) (l' ++ bs) (by simp [hs])
    refine ⟨r', bs ++ bs', ?_, ?_⟩
    · dsimp only
      rw [hx', Array.append_assoc]
    · dsimp only
      rw [hy', Array.append_assoc]

/-- **The shift's simulation**: a run of the pure session from two histories of one size. -/
def shiftS : With.Sim PS PS where
  R x y := Shift x y
  pure a := Shift.still (v := .ok a) fun _ => rfl
  bind hxy hfg := Shift.bind hxy hfg
  throw e := Shift.still (v := .error e) fun _ => rfl
  lift v := Shift.still (v := v) fun l => monadLift_go v l

/-- **A run from a history of the same size** ends the same way, after the same bodies. -/
theorem go_of_shift {α : Type} {x : PS α} (h : shiftS.R x x) {l l' L : Array Claims.Msg} (hs : l.size = l'.size)
    {r : Except String α} (hx : x.go l = (r, L)) : ∃ bs, L = l ++ bs ∧ x.go l' = (r, l' ++ bs) := by
  obtain ⟨r', bs, h1, h2⟩ := h l l' hs
  rw [hx] at h1
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj h1
  exact ⟨bs, rfl, h2⟩

theorem round_shift (N : ℕ) (c : ByteArray) : shiftS.R ((asksCut coin N).round c) ((asksCut coin N).round c) := by
  intro l l' hs
  rw [cut_round_go, cut_round_go, hs]
  by_cases h : N ≤ l'.size
  · refine ⟨.error "past the openings", #[Claims.toMsg c], ?_, ?_⟩ <;> rw [ite_eq_left h, Array.push_eq_append]
  · refine ⟨.ok (coin l'.size), #[Claims.toMsg c], ?_, ?_⟩ <;> rw [ite_eq_right h, Array.push_eq_append]

variable (N : ℕ)

theorem squeezeP_shift {kind : UInt8} {m : ℕ} :
    (shiftS.st Transcript).R (squeezeP (asksCut coin N) kind m) (squeezeP (asksCut coin N) kind m) := by
  intro t
  unfold squeezeP
  dsimp only
  split
  · exact shiftS.pure _
  · exact shiftS.bind (round_shift coin N _) fun _ => shiftS.pure _

theorem liftT_shift {α : Type} {x y : TMW PS α} (h : (shiftS.st Transcript).R x y) :
    ((shiftS.st Transcript).st LigState).R (liftT x) (liftT y) :=
  With.Sim.lift_st _ h

theorem sendMsg_shift (m : F256 × F256) :
    ((shiftS.st Transcript).st LigState).R (sendMsg m : LP PS Unit) (sendMsg m) := by
  unfold sendMsg
  sim_walk [liftT_shift]

theorem powP_shift (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    ((shiftS.st Transcript).st LigState).R (powP nonce kind : LP PS Unit) (powP nonce kind) := by
  unfold powP
  sim_walk [liftT_shift]

theorem commitP_shift (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) :
    shiftS.R (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt) (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt) :=
  Shift.still (commitP_quiet h ms lv g salt)

theorem shift_pure_st {α σ : Type} (a : α) (s : σ) :
    shiftS.R ((Pure.pure a : StateT σ PS α) s) ((Pure.pure a : StateT σ PS α) s) :=
  shiftS.pure (a, s)

theorem rankCheckP_shift (m kLog : ℕ) (prior : Array (List F128)) (words : List ℕ) :
    (shiftS.st Transcript).R (rankCheckP m kLog prior words : TMW PS Unit) (rankCheckP m kLog prior words) := by
  unfold rankCheckP
  sim_walk

theorem sampleP_shift : (shiftS.st Transcript).R (sampleP (asksCut coin N)) (sampleP (asksCut coin N)) := by
  unfold sampleP
  sim_walk [squeezeP_shift coin N]

theorem sampleKP_shift : (shiftS.st Transcript).R (sampleKP (asksCut coin N)) (sampleKP (asksCut coin N)) := by
  unfold sampleKP
  sim_walk [squeezeP_shift coin N]

theorem challengeP_shift (split : Bool) :
    ((shiftS.st Transcript).st LigState).R (challengeP (asksCut coin N) split) (challengeP (asksCut coin N) split) := by
  unfold challengeP
  sim_walk [liftT_shift, sampleKP_shift coin N, sendMsg_shift]

theorem oodIntroP_shift (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    ((shiftS.st Transcript).st LigState).R (oodIntroP (asksCut coin N) nonce g dim)
      (oodIntroP (asksCut coin N) nonce g dim) := by
  unfold oodIntroP sampleVecP
  sim_walk [liftT_shift, squeezeP_shift coin N, sampleP_shift coin N, sendMsg_shift, powP_shift]

theorem queryP_shift (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    ((shiftS.st Transcript).st LigState).R (queryP (asksCut coin N) nonce ms lv cm)
      (queryP (asksCut coin N) nonce ms lv cm) := by
  unfold queryP sampleVecP
  sim_walk [liftT_shift, squeezeP_shift coin N, powP_shift]

theorem zerocheckCut_shift (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout) (mk : ℕ → F128) :
    (shiftS.st Transcript).R (zerocheckCut (asksCut coin N) m digest cap root lay mk)
      (zerocheckCut (asksCut coin N) m digest cap root lay mk) := by
  unfold zerocheckCut sampleVecP
  sim_walk [squeezeP_shift coin N, sampleP_shift coin N]

theorem lincheckCut_shift (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (mk : ℕ → F128) (zc : ZcOut) :
    (shiftS.st Transcript).R (lincheckCut (asksCut coin N) c m lay mk zc) (lincheckCut (asksCut coin N) c m lay mk zc) := by
  unfold lincheckCut
  sim_walk [sampleP_shift coin N]

theorem ringSwitchCut_shift (sv : ℕ → Array F128) (claims : Array Claim) :
    (shiftS.st Transcript).R (ringSwitchCut (asksCut coin N) sv claims) (ringSwitchCut (asksCut coin N) sv claims) := by
  unfold ringSwitchCut sampleVecP
  sim_walk [squeezeP_shift coin N]

theorem piopCut_shift (st : Setup) (lay : Zk.PadLayout) (src : RepSrc) (cap root : ByteArray)
    (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ) :
    (shiftS.st Transcript).R (piopCut (asksCut coin N) st lay src cap root extras prior words)
      (piopCut (asksCut coin N) st lay src cap root extras prior words) := by
  unfold piopCut
  sim_walk [zerocheckCut_shift coin N, lincheckCut_shift coin N, rankCheckP_shift, ringSwitchCut_shift coin N]

set_option maxHeartbeats 1000000 in
theorem ligeritoCut_shift (ms : MerkleScheme) (sch : Schedule) (T : F128) (cm0 : Commit) (b : Array F128)
    (lay : Zk.PadLayout) (src : RepSrc) :
    (shiftS.st Transcript).R (ligeritoCut (asksCut coin N) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 b lay src)
      (ligeritoCut (asksCut coin N) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 b lay src) := by
  unfold ligeritoCut sampleVecP
  sim_walk [liftT_shift, squeezeP_shift coin N, sampleP_shift coin N, sampleKP_shift coin N, sendMsg_shift,
    powP_shift, challengeP_shift coin N, oodIntroP_shift coin N, queryP_shift coin N, commitP_shift Sha512.hash,
    shift_pure_st]

theorem innerCut_shift (lay : Zk.PadLayout) (ic : InnerCommit) (finalC : F128) (src : RepSrc)
    (cons : Array (String × Zk.Lin)) :
    (shiftS.st Transcript).R (innerCut (asksCut coin N) lay ic finalC src cons)
      (innerCut (asksCut coin N) lay ic finalC src cons) := by
  unfold innerCut sampleVecP
  sim_walk [sampleP_shift coin N, squeezeP_shift coin N]

/-- **A cut rep reads only its history's size.** -/
theorem repCut_shift (st : Setup) (sch : Schedule) (rep : ℕ) (dm : ByteArray) (cm0 : Commit)
    (extras : Array Claim) (ic : InnerCommit) (src : RepSrc) (prior : Array (List F128)) (words : List ℕ) :
    (shiftS.st Transcript).R
      (repCut (asksCut coin N) (Hashes.ofSha512 Sha512.hash) st sch rep dm cm0 extras ic src prior words)
      (repCut (asksCut coin N) (Hashes.ofSha512 Sha512.hash) st sch rep dm cm0 extras ic src prior words) := by
  unfold repCut
  sim_walk [piopCut_shift coin N, ligeritoCut_shift coin N, innerCut_shift coin N]

end Shift

end FlockComplete.CutPrefix
