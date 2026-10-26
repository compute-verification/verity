import Proofs.Flock.Claims.ZK.Split

/-!
# The cut run answers by the openings (`Masked.prefixLive`)

`PrefixLive`: the live run at openings `ops` and the deployed plan's cut run at `padOps ops` agree up to and including
send `ops.length`, and a live run that sends no more than `ops.length` bodies stopped.

* **The coupling** (`Agree n`, its simulation `agreeS n`): a run of the pure session from at most `n` sends either ends
  where the other copy ends, at no more than `n` sends, or the second copy threw at send `n + 1` and the first copy's
  sends extend it. A squeeze of at most `65536` coins keeps it (`squeezeP_agree`), since the live coin and the cut
  coin agree there (`slotAt_padOps_lt`), and so does each masked step, given its squeezes' sizes, as `RepRounds` bounds
  them (`repPZk_agree`; the recursive levels' out-of-domain sizes ride the loop, `loopR_agree`). The link coin is read
  at `points · m_pts ≤ 128` indices (`afterLinkPZk_congr`), so the whole session keeps it (`sessionPZk_agree`).
* **The returning count** (`sessionPZk_size`): a masked session that returns sent `repStartZk` of its last rep, which is
  `1 + rounds = refK x` (`repStart_last`, `refK_eq`), more than `ops.length`.
-/

namespace FlockComplete.PrefixAgree

open Flock Transcript FlockComplete FlockDeploy With JsonRT

/-- The run only appends to the sends. -/
def Grow {α : Type} (y : PS α) : Prop := ∀ l, ∃ bs : Array Claims.Msg, (y.go l).2 = l ++ bs

/-- Two ends of runs from at most `n` sends: the same end, at no more than `n` sends; or the second threw at send
`n + 1` and the first's sends extend the second's. -/
def Couple {α : Type} (n : ℕ) (a b : Except String α × Array Claims.Msg) : Prop :=
  (a = b ∧ a.2.size ≤ n) ∨ ((∃ e, b.1 = .error e) ∧ b.2.size = n + 1 ∧ ∃ bs, a.2 = b.2 ++ bs)

/-- **The coupling at `n`**: `x` only appends, and from at most `n` sends `x` and `y` end coupled. -/
def Agree (n : ℕ) {α : Type} (x y : PS α) : Prop :=
  Grow x ∧ ∀ l : Array Claims.Msg, l.size ≤ n → Couple n (x.go l) (y.go l)

theorem Agree.bind_ok {n : ℕ} {α β : Type} {x y : PS α} {f g : α → PS β} (Q : α → Prop) (hxy : Agree n x y)
    (hQ : ∀ l a l', x.go l = (.ok a, l') → Q a) (hfg : ∀ a, Q a → Agree n (f a) (g a)) :
    Agree n (x >>= f) (y >>= g) := by
  refine ⟨fun l => ?_, fun l hl => ?_⟩
  · obtain ⟨bs, hbs⟩ := hxy.1 l
    rw [go_bind]
    rcases hx : x.go l with ⟨r, l'⟩
    rw [hx] at hbs
    cases r with
    | error e => exact ⟨bs, hbs⟩
    | ok a =>
      obtain ⟨bs', h'⟩ := (hfg a (hQ l a l' hx)).1 l'
      refine ⟨bs ++ bs', ?_⟩
      dsimp only at hbs ⊢
      rw [h', hbs, Array.append_assoc]
  · rw [go_bind, go_bind]
    rcases hxy.2 l hl with ⟨he, hs⟩ | ⟨⟨e, he⟩, hs, bs, hbs⟩
    · rw [← he]
      rcases hx : x.go l with ⟨r, l'⟩
      rw [hx] at hs
      cases r with
      | error e => exact .inl ⟨rfl, hs⟩
      | ok a => exact (hfg a (hQ l a l' hx)).2 l' hs
    · rcases hy : y.go l with ⟨r', l''⟩
      rw [hy] at he hs hbs
      dsimp only at he hs hbs
      subst he
      rcases hx : x.go l with ⟨r, l'⟩
      rw [hx] at hbs
      dsimp only at hbs
      subst hbs
      cases r with
      | error e' => exact .inr ⟨⟨e, rfl⟩, hs, bs, rfl⟩
      | ok a =>
        obtain ⟨bs', h'⟩ := (hfg a (hQ l a _ hx)).1 (l'' ++ bs)
        exact .inr ⟨⟨e, rfl⟩, hs, bs ++ bs', by rw [h', Array.append_assoc]⟩

/-- A run that sends nothing and ends at `v` agrees with itself. -/
theorem Agree.still {n : ℕ} {α : Type} {v : Except String α} {y : PS α} (h : ∀ l, y.go l = (v, l)) : Agree n y y :=
  ⟨fun l => ⟨#[], by rw [h l, Array.append_empty]⟩, fun l hl => .inl ⟨rfl, by rw [h l]; exact hl⟩⟩

/-- **The coupling's simulation**: the pure session's runs at two answerers, coupled at `n`. -/
def agreeS (n : ℕ) : With.Sim PS PS where
  R x y := Agree n x y
  pure a := Agree.still (v := .ok a) fun _ => rfl
  bind hxy hfg := Agree.bind_ok (fun _ => True) hxy (fun _ _ _ _ => trivial) fun a _ => hfg a
  throw e := Agree.still (v := .error e) fun _ => rfl
  lift v := Agree.still (v := v) fun l => monadLift_go v l

theorem agree_still {n : ℕ} {α : Type} {v : Except String α} {y : PS α} (h : ∀ l, y.go l = (v, l)) :
    (agreeS n).R y y :=
  Agree.still h

/-- `Agree.bind_ok` on transcript computations. -/
theorem agreeT_bind_ok {n : ℕ} {α β : Type} {x y : TMW PS α} {f g : α → TMW PS β} (Q : α → Prop)
    (hxy : ((agreeS n).st Transcript).R x y)
    (hQ : ∀ t l a t' l', (x.run t).go l = (.ok (a, t'), l') → Q a)
    (hfg : ∀ a, Q a → ((agreeS n).st Transcript).R (f a) (g a)) :
    ((agreeS n).st Transcript).R (x >>= f) (y >>= g) := fun t =>
  Agree.bind_ok (fun p => Q p.1) (hxy t) (fun l p l' h => hQ t l p.1 p.2 l' h) fun p hp => hfg p.1 hp p.2

/-- `Agree.bind_ok` on Ligerito steps. -/
theorem agreeL_bind_ok {n : ℕ} {α β : Type} {x y : LP PS α} {f g : α → LP PS β} (Q : α → Prop)
    (hxy : (((agreeS n).st Transcript).st LigState).R x y)
    (hQ : ∀ s t l a s' t' l', ((x.run s).run t).go l = (.ok ((a, s'), t'), l') → Q a)
    (hfg : ∀ a, Q a → (((agreeS n).st Transcript).st LigState).R (f a) (g a)) :
    (((agreeS n).st Transcript).st LigState).R (x >>= f) (y >>= g) := fun s =>
  agreeT_bind_ok (fun p => Q p.1) (hxy s) (fun t l p t' l' h => hQ s t l p.1 p.2 t' l' h)
    fun p hp => hfg p.1 hp p.2

/-! ## The answerers -/

theorem go_get_bind {β : Type} (k : Array Claims.Msg → PS β) (l : Array Claims.Msg) :
    ((get : PS (Array Claims.Msg)) >>= k).go l = (k l).go l := rfl

theorem go_set_bind {β : Type} (s : Array Claims.Msg) (k : PUnit → PS β) (l : Array Claims.Msg) :
    ((set s : PS PUnit) >>= k).go l = (k ⟨⟩).go s := rfl

theorem cut_round_go (coin : ℕ → ℕ → F128) (n : ℕ) (c : ByteArray) (l : Array Claims.Msg) :
    ((asksCut coin n).round c).go l =
      if n ≤ l.size then (.error "past the openings", l.push (Claims.toMsg c))
      else (.ok (coin l.size), l.push (Claims.toMsg c)) := by
  unfold asksCut
  dsimp only
  rw [go_get_bind, go_set_bind]
  by_cases h : n ≤ l.size
  · simp only [h, ↓reduceIte]
    rfl
  · simp only [h, ↓reduceIte]
    rfl

theorem cut_link_go (coin : ℕ → ℕ → F128) (n : ℕ) (a b : Array ByteArray) (l : Array Claims.Msg) :
    ((asksCut coin n).link a b).go l =
      if n ≤ l.size then (.error "past the openings", l.push (linkBody a.toList b.toList))
      else (.ok (coin l.size), l.push (linkBody a.toList b.toList)) := by
  unfold asksCut
  dsimp only
  rw [go_get_bind, go_set_bind]
  by_cases h : n ≤ l.size
  · simp only [h, ↓reduceIte]
    rfl
  · simp only [h, ↓reduceIte]
    rfl

variable (resp : ℕ → Claims.Msg) (coin : ℕ → ℕ → F128) {n : ℕ}
  (hc : ∀ k < n, ∀ i < 65536, coinOf (resp k) i = coin k i)

include hc in
/-- **A squeeze of at most `65536` coins keeps the coupling**: the live coin and the cut coin agree there. -/
theorem squeezeP_agree (kind : UInt8) {m : ℕ} (hm : m ≤ 65536) :
    ((agreeS n).st Transcript).R (squeezeP (asksS resp) kind m) (squeezeP (asksCut coin n) kind m) := by
  intro t
  unfold squeezeP
  dsimp only
  split
  · exact (agreeS n).pure _
  · refine ⟨fun l => ⟨#[Claims.toMsg (framed t kind m)], ?_⟩, fun l hl => ?_⟩
    · rw [go_bind, round_go]
      rfl
    · rw [go_bind, go_bind, round_go, cut_round_go]
      by_cases hn : n ≤ l.size
      · simp only [hn, ↓reduceIte]
        refine .inr ⟨⟨_, rfl⟩, by simp; omega, #[], by simp [go_pure]⟩
      · simp only [hn, ↓reduceIte]
        have hv : (Array.range m).map (coinOf (resp l.size)) = (Array.range m).map (coin l.size) :=
          Array.map_congr_left fun i hi => hc _ (by omega) i (by rw [Array.mem_range] at hi; omega)
        rw [hv]
        exact .inl ⟨rfl, by simp [go_pure]; omega⟩

/-! ## Each masked step keeps the coupling, given its squeezes' sizes -/

include hc

theorem sampleP_agree :
    ((agreeS n).st Transcript).R (sampleP (asksS resp)) (sampleP (asksCut coin n)) := by
  unfold sampleP
  sim_walk [squeezeP_agree resp coin hc _ (show 1 ≤ 65536 by decide)]

theorem sampleKP_agree :
    ((agreeS n).st Transcript).R (sampleKP (asksS resp)) (sampleKP (asksCut coin n)) := by
  unfold sampleKP
  sim_walk [squeezeP_agree resp coin hc _ (show 2 ≤ 65536 by decide)]

theorem zerocheckPZk_agree (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace)
    (hm : m - 13 ≤ 65536) :
    ((agreeS n).st Transcript).R (zerocheckPZk (asksS resp) m digest cap root lay h tr)
      (zerocheckPZk (asksCut coin n) m digest cap root lay h tr) := by
  unfold zerocheckPZk sampleVecP
  sim_walk [squeezeP_agree resp coin hc _ (show 6 ≤ 65536 by decide), squeezeP_agree resp coin hc _ hm,
    sampleP_agree resp coin hc]

theorem lincheckPZk_agree (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (zc : ZcOut) :
    ((agreeS n).st Transcript).R (lincheckPZk (asksS resp) c m lay h tr zc)
      (lincheckPZk (asksCut coin n) c m lay h tr zc) := by
  unfold lincheckPZk
  sim_walk [sampleP_agree resp coin hc]

theorem ringSwitchP_agree (tr : Trace) (claims : Array Claim) (hcl : claims.size ≤ 65536) :
    ((agreeS n).st Transcript).R (ringSwitchP (asksS resp) tr claims) (ringSwitchP (asksCut coin n) tr claims) := by
  unfold ringSwitchP sampleVecP
  sim_walk [squeezeP_agree resp coin hc _ (show 7 ≤ 65536 by decide), squeezeP_agree resp coin hc _ hcl]

omit hc in
theorem liftT_agree {α : Type} {x y : TMW PS α} (h : ((agreeS n).st Transcript).R x y) :
    (((agreeS n).st Transcript).st LigState).R (liftT x) (liftT y) :=
  With.Sim.lift_st _ h

omit hc in
theorem sendMsg_agree (m : F256 × F256) :
    (((agreeS n).st Transcript).st LigState).R (sendMsg m : LP PS Unit) (sendMsg m) := by
  unfold sendMsg
  sim_walk [liftT_agree]

omit hc in
theorem powP_agree (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    (((agreeS n).st Transcript).st LigState).R (powP nonce kind : LP PS Unit) (powP nonce kind) := by
  unfold powP
  sim_walk [liftT_agree]

theorem challengeP_agree (split : Bool) :
    (((agreeS n).st Transcript).st LigState).R (challengeP (asksS resp) split) (challengeP (asksCut coin n) split) := by
  unfold challengeP
  sim_walk [liftT_agree, sampleKP_agree resp coin hc, sendMsg_agree]

theorem oodIntroP_agree (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) (hd : dim ≤ 65536) :
    (((agreeS n).st Transcript).st LigState).R (oodIntroP (asksS resp) nonce g dim)
      (oodIntroP (asksCut coin n) nonce g dim) := by
  unfold oodIntroP sampleVecP
  sim_walk [liftT_agree, squeezeP_agree resp coin hc _ hd, sampleP_agree resp coin hc, sendMsg_agree, powP_agree]

theorem queryP_agree (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) (hq : lv.q ≤ 65536)
    (hl : Nat.log2 (lv.q - 1) + 1 ≤ 65536) :
    (((agreeS n).st Transcript).st LigState).R (queryP (asksS resp) nonce ms lv cm)
      (queryP (asksCut coin n) nonce ms lv cm) := by
  unfold queryP sampleVecP
  sim_walk [liftT_agree, squeezeP_agree resp coin hc _ hq, squeezeP_agree resp coin hc _ hl, powP_agree]

theorem queryP_agree_lv (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (sch : Schedule)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ 65536) (j : ℕ) (cm : Commit) :
    (((agreeS n).st Transcript).st LigState).R (queryP (asksS resp) nonce ms (sch.levels.getD j default) cm)
      (queryP (asksCut coin n) nonce ms (sch.levels.getD j default) cm) :=
  queryP_agree resp coin hc nonce ms _ cm (by have := hq j; omega)
    (by have := hq j; have := Nat.log2_le_self ((sch.levels.getD j default).q - 1); omega)

omit hc in
theorem tau_agree (h : ByteArray → ByteArray) (tau : F128) (salt : ByteArray) :
    (agreeS n).R (Zk.tauCommitmentWith (Hashes.ofSha512 h : Hashes PS) tau salt)
      (Zk.tauCommitmentWith (Hashes.ofSha512 h : Hashes PS) tau salt) := by
  rw [Replay.tauCommitmentWith_ofSha512]
  exact (agreeS n).pure _

omit hc in
theorem commitP_agree (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) :
    (agreeS n).R (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt)
      (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt) :=
  agree_still (commitP_quiet h ms lv g salt)

omit hc in
theorem agree_pure_st {α σ : Type} (a : α) (s : σ) :
    (agreeS n).R ((Pure.pure a : StateT σ PS α) s) ((Pure.pure a : StateT σ PS α) s) :=
  (agreeS n).pure (a, s)

theorem squeezeP_agree_cons (lay : Zk.PadLayout) (cons : Array (String × Zk.Lin)) (hcs : cons.size + 3 ≤ 65536)
    (e r s : F128) :
    ((agreeS n).st Transcript).R (squeezeP (asksS resp) 2 (cons ++ Zk.triple lay e r s).size)
      (squeezeP (asksCut coin n) 2 (cons ++ Zk.triple lay e r s).size) :=
  squeezeP_agree resp coin hc 2 (by simp [Zk.triple]; omega)

theorem innerP_agree (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk) (ic : InnerCommit)
    (finalC : F128) (cons : Array (String × Zk.Lin)) (hcs : cons.size + 3 ≤ 65536) :
    ((agreeS n).st Transcript).R (innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons)
      (innerP (asksCut coin n) (Hashes.ofSha512 h) lay p ic finalC cons) := by
  unfold innerP sampleVecP
  sim_walk [sampleP_agree resp coin hc, squeezeP_agree resp coin hc _ (show Zk.Q ≤ 65536 by decide),
    squeezeP_agree_cons resp coin hc lay cons hcs, tau_agree h]

omit hc in
theorem rankCheckP_agree (m kLog : ℕ) (prior : Array (List F128)) (words : List ℕ) :
    ((agreeS n).st Transcript).R (rankCheckP m kLog prior words : TMW PS Unit) (rankCheckP m kLog prior words) := by
  unfold rankCheckP
  sim_walk

omit hc in
/-- **A loop that keeps an invariant of its live state keeps the coupling**, its steps coupled at the invariant. -/
theorem loopR_agree {β : Type} (I : β → Prop) (r : Std.Legacy.Range) (b : β) (f g : ℕ → β → LP PS (ForInStep β))
    (hb : I b) (hI : ∀ i b, I b → GrowsL 65536 (f i b) fun s => I s.value)
    (hfg : ∀ i b, I b → (((agreeS n).st Transcript).st LigState).R (f i b) (g i b)) :
    (((agreeS n).st Transcript).st LigState).R (loopR r b f) (loopR r b g) := by
  unfold loopR
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  generalize List.range' r.start r.size r.step = xs
  induction xs generalizing b with
  | nil => exact With.Sim.pure _ b
  | cons a xs ih =>
    rw [List.forIn_cons, List.forIn_cons]
    refine agreeL_bind_ok (fun s => I s.value) (hfg a b hb)
      (fun s t l st s' t' l' e => (hI a b hb s t l (st, s') t' l' e).2) fun st hst => ?_
    cases st with
    | done b' => exact With.Sim.pure _ b'
    | yield b' => exact ih b' hst

theorem oodIntroP_agree_loop (nonce : ℕ → ℕ → UInt64) (N : ℕ) (hN : N ≤ 65536) (g : Array F128)
    (b : ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof) (hb : b.1 ≤ N) :
    (((agreeS n).st Transcript).st LigState).R (oodIntroP (asksS resp) nonce g (b.1 - 4 + 1))
      (oodIntroP (asksCut coin n) nonce g (b.1 - 4 + 1)) :=
  oodIntroP_agree resp coin hc nonce g _ (by omega)

/-- **The masked Ligerito keeps the coupling**, given `n + 1` and every level's `q + 1` at most `65536`. -/
theorem ligeritoPZk_agree (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cm0 : Commit)
    (t b : Array F128) (lay : Zk.PadLayout) (hp : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (hn : sch.n + 1 ≤ 65536)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ 65536) :
    ((agreeS n).st Transcript).R
      (ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 t b lay hp rep tPad extra lanePad salt nonce)
      (ligeritoPZk (asksCut coin n) (Hashes.ofSha512 h) ms sch T cm0 t b lay hp rep tPad extra lanePad salt nonce) := by
  have h1 : 1 ≤ 65536 := by decide
  have h2 : 2 ≤ 65536 := by decide
  unfold ligeritoPZk sampleVecP
  simp (config := { zeta := false }) only [forIn_loopR]
  sim_walk [liftT_agree, squeezeP_agree resp coin hc _ (show sch.n ≤ 65536 by omega), sampleP_agree resp coin hc,
    sampleKP_agree resp coin hc, sendMsg_agree, powP_agree, challengeP_agree resp coin hc,
    oodIntroP_agree resp coin hc nonce _ (sch.n - sch.k0 + 1) (by omega),
    queryP_agree_lv resp coin hc nonce ms sch hq, commitP_agree h, agree_pure_st,
    oodIntroP_agree_loop resp coin hc nonce (sch.n + 1) hn,
    loopR_agree (fun s : ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof => s.1 ≤ sch.n + 1)]
  case hfg =>
    intro i b (hb : b.1 ≤ sch.n + 1)
    sim_walk [liftT_agree, sampleP_agree resp coin hc, sampleKP_agree resp coin hc, sendMsg_agree, powP_agree,
      challengeP_agree resp coin hc, queryP_agree_lv resp coin hc nonce ms sch hq, commitP_agree h, agree_pure_st,
      oodIntroP_agree_loop resp coin hc nonce (sch.n + 1) hn _ b hb]
  all_goals first
    | exact fun _ => ((agreeS n).st Transcript).pure _
    | (intro _ _ _
       exact agree_still fun _ => rfl)
    | (simp (config := { zetaDelta := true }) only []
       omega)
    | (intro i b hb
       repeat (first
         | extract_lets
         | (refine GrowsL.bind_keeps ?_ fun _ => ?_
            sim_walk [liftT_keeps, sampleP_keeps resp h1, sampleKP_keeps resp h2, sendMsg_keeps, powP_keeps,
              challengeP_keeps resp _ h2, oodIntroP_keeps_loop resp nonce (sch.n + 1) hn h1,
              queryP_keeps_lv resp nonce ms sch hq, commitP_keeps 65536 h, absorbF_keeps, absorbBytes_keeps])
         | (refine GrowsL.pure ?_
            simp (config := { zetaDelta := true }) only [ForInStep.value]
            omega)
         | split))

omit hc in
/-- A lifted verdict's value: what follows it is coupled at the value it returns. -/
theorem agreeT_liftV_bind {α β : Type} (v : V α) {f g : α → TMW PS β}
    (hfg : ∀ a, v = .ok a → ((agreeS n).st Transcript).R (f a) (g a)) :
    ((agreeS n).st Transcript).R (StateT.lift (monadLift v : PS α) >>= f)
      (StateT.lift (monadLift v : PS α) >>= g) := by
  refine agreeT_bind_ok (fun a => v = .ok a) (With.Sim.lift_st _ ((agreeS n).lift v)) (fun t l a t' l' e => ?_) hfg
  change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at e
  obtain ⟨b, l₁, e1, e2⟩ := go_bind_ok e
  rw [monadLift_go] at e1
  simp only [Prod.mk.injEq] at e1
  obtain ⟨hv, rfl⟩ := e1
  cases e2
  exact hv

/-- **A masked rep keeps the coupling**, given `m - 13`, its claims, `n + 1` and every level's `q + 1` at most
`65536`. -/
theorem repPZk_agree (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128))
    (words : List ℕ) (hm : st.m - 13 ≤ 65536) (hx : extras.size + 2 ≤ 65536) (hn : sch.n + 1 ≤ 65536)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ 65536) :
    ((agreeS n).st Transcript).R
      (repPZk (asksS resp) (Hashes.ofSha512 h) st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce
        prior words)
      (repPZk (asksCut coin n) (Hashes.ofSha512 h) st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt
        nonce prior words) := by
  unfold repPZk
  refine With.Sim.bind _ (zerocheckPZk_agree resp coin hc _ _ _ _ _ _ _ hm) fun ⟨zp, zc⟩ => ?_
  refine With.Sim.bind _ (lincheckPZk_agree resp coin hc _ _ _ _ _ _) fun ⟨lp, lc⟩ => ?_
  refine With.Sim.bind _ (rankCheckP_agree _ _ _ _) fun _ => ?_
  refine With.Sim.bind _ (ringSwitchP_agree resp coin hc _ _ ?_) fun ⟨ss, T, sws⟩ => ?_
  · simp only [Array.size_append, abcClaims, List.size_toArray, List.length_cons, List.length_nil]
    omega
  refine With.Sim.bind _ (ligeritoPZk_agree resp coin hc h _ sch _ _ _ _ _ _ _ _ _ _ _ _ hn hq)
    fun ⟨lg, zl, rec⟩ => ?_
  refine With.Sim.bind _ (With.Sim.get _) fun tr' => ?_
  refine agreeT_liftV_bind _ fun ⟨rp, t''⟩ hrp => ?_
  refine agreeT_liftV_bind _ fun lgc hlgc => ?_
  have h3 := FlockSoundness.Discharge.ZkExec.replay_cons_size _ _ _ _ hrp
  have h2 := FlockSoundness.Discharge.ZkExec.ligeritoConstraints_size hlgc
  refine With.Sim.bind _ (innerP_agree resp coin hc h _ _ _ _ _ ?_) fun _ => With.Sim.pure _ _
  simp only [Array.size_append, List.size_toArray, List.length_cons, List.length_nil, h3, h2]
  omega

omit hc in
/-- A `for` loop over an array whose steps at its elements are coupled. -/
theorem agree_forIn_mem {α β : Type} (xs : Array α) (b : β) {f g : α → β → PS (ForInStep β)}
    (h : ∀ a ∈ xs, ∀ b, (agreeS n).R (f a b) (g a b)) : (agreeS n).R (forIn xs b f) (forIn xs b g) := by
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  have h' : ∀ a ∈ xs.toList, ∀ b, (agreeS n).R (f a b) (g a b) := fun a ha => h a (Array.mem_toList_iff.1 ha)
  generalize xs.toList = L at h'
  induction L generalizing b with
  | nil => exact (agreeS n).pure b
  | cons a L ih =>
    rw [List.forIn_cons, List.forIn_cons]
    refine (agreeS n).bind (h' a List.mem_cons_self b) fun r => ?_
    cases r with
    | done b => exact (agreeS n).pure b
    | yield b => exact ih b fun a ha => h' a (List.mem_cons_of_mem _ ha)

omit hc in
theorem recordJson_quietZ (h : ByteArray → ByteArray) (spec : SessionSpec) (rootB : Array ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    quietS.R (recordJson (Hashes.ofSha512 h) spec rootB points streams proofs : V _)
      (recordJson (Hashes.ofSha512 h) spec rootB points streams proofs : PS _) := by
  have hh := quietS_hashes h
  unfold recordJson SessionSpec.dgWith
  sim_walk [roundJ_quiet h, With.Sim.mapM_list]

/-- **The masked session from the link coin on keeps the coupling**, at tables of `25 ≤ m ≤ 35` and at most 1,024
regions, two link points, one spec table per statement and each table's `--zk` schedule. -/
theorem afterLinkPZk_agree (h : ByteArray → ByteArray) (spec : SessionSpec) (sts : Array Stmt)
    (words : Array (List ℕ)) (schs : Array (Schedule × ℕ)) (trs : Array Trace) (tape : TapeZk)
    (tbls : Array (Array F128)) (cms : Array Commit) (c1 : ℕ → F128) (hspec : spec.tables.size = sts.size)
    (hp : spec.points = 2) (hm : ∀ j (hj : j < sts.size), 25 ≤ sts[j].m ∧ sts[j].m ≤ 35)
    (hreg : ∀ j (hj : j < sts.size), sts[j].regions.size ≤ 1024)
    (hschs : ∀ j (hj : j < sts.size), Zk.schedule sts[j].m = .ok schs[j]!) :
    (agreeS n).R (afterLinkPZk (asksS resp) (Hashes.ofSha512 h) spec (tabsOf sts) words trs tape schs tbls cms c1)
      (afterLinkPZk (asksCut coin n) (Hashes.ofSha512 h) spec (tabsOf sts) words trs tape schs tbls cms c1) := by
  unfold afterLinkPZk
  extract_lets rootB points streams proofs
  have hpts : points.size = spec.points := by simp [points]
  refine (agreeS n).bind (agree_forIn_mem _ _ fun a ha b => ?_) fun s => ?_
  · obtain ⟨i, r, nm⟩ := a
    have hi : i < sts.size := by
      rw [← Array.mem_toList_iff, FlockDeploy.repStreams_toList, List.mem_map] at ha
      obtain ⟨k, hk, he⟩ := ha
      rw [List.mem_range, hspec] at hk
      have := congrArg Prod.fst he
      simp only at this
      omega
    have ht : (tabsOf sts)[i]! = Setup.ofCircuit sts[i] := by
      rw [getElem!_pos (tabsOf sts) i (by simpa [tabsOf] using hi)]
      simp [tabsOf]
    have hst : sts[i]! = sts[i] := getElem!_pos sts i hi
    have hv := hschs i hi
    obtain ⟨hn, hq⟩ := zkSchedule_small (hm i hi).1 (hm i hi).2 hv
    have hm35 := (hm i hi).2
    have hri := hreg i hi
    extract_lets
    dsimp only
    simp only [ht]
    refine Agree.bind_ok (fun extras => (Setup.ofCircuit sts[i]).extra points = .ok extras) ((agreeS n).lift _)
      (fun l a l' e => by rw [monadLift_go] at e; exact (Prod.mk.inj e).1) fun extras he => ?_
    have hx : extras.size = spec.points * sts[i].regions.size := by
      rw [FlockDeploy.extraClaims_size (st := sts[i]) he, hpts, Nat.mul_comm]
    refine (agreeS n).bind (agree_still (innerCommitP_quiet h _ _)) fun ic => ?_
    extract_lets
    refine (agreeS n).bind (With.Sim.run _ (repPZk_agree resp coin hc h (Setup.ofCircuit sts[i]) schs[i]!.1 _ _ _ _ _
      _ extras _ _ _ _ _ _ _ _ (show sts[i].m - 13 ≤ 65536 by omega) (by rw [hx, hp]; omega) hn hq) _)
      fun _ => (agreeS n).pure _
  · try extract_lets
    exact (agreeS n).bind (agree_still (recordJson_quietZ h _ _ _ _ _)) fun _ => (agreeS n).pure _

omit hc in
/-- The session after the link reads the link coin only at the points' indices. -/
theorem afterLinkPZk_congr {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (io : Asks M)
    (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk) (schs : Array (Schedule × ℕ)) (tbls : Array (Array F128)) (cms : Array Commit)
    {c1 c1' : ℕ → F128} (h : ∀ p < spec.points, ∀ j < spec.mPts, c1 (p * spec.mPts + j) = c1' (p * spec.mPts + j)) :
    afterLinkPZk io hs spec tabs words trs tape schs tbls cms c1 =
      afterLinkPZk io hs spec tabs words trs tape schs tbls cms c1' := by
  have hpts : ((Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j => c1 (p * spec.mPts + j)) =
      (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j => c1' (p * spec.mPts + j) :=
    Array.map_congr_left fun p hp => Array.map_congr_left fun j hj =>
      h p (by simpa using hp) j (by simpa using hj)
  unfold afterLinkPZk
  rw [hpts]

/-- **The masked session keeps the coupling**, from no sends, at a cut past the link, at tables of `25 ≤ m ≤ 35` and
at most 1,024 regions, two link points of at most 64 coordinates each, and one spec table per statement. -/
theorem sessionPZk_agree (h : ByteArray → ByteArray) (spec : SessionSpec) (sts : Array Stmt)
    (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk) (hn0 : 0 < n) (hspec : spec.tables.size = sts.size)
    (hp : spec.points = 2) (hmp : spec.mPts ≤ 64) (hm : ∀ j (hj : j < sts.size), 25 ≤ sts[j].m ∧ sts[j].m ≤ 35)
    (hreg : ∀ j (hj : j < sts.size), sts[j].regions.size ≤ 1024) :
    Couple n ((sessionPZk (asksS resp) (Hashes.ofSha512 h) spec (tabsOf sts) words trs tape).go #[])
      ((sessionPZk (asksCut coin n) (Hashes.ofSha512 h) spec (tabsOf sts) words trs tape).go #[]) := by
  rw [sessionPZk_eq, sessionPZk_eq, go_bind, go_bind, schedsPZk_quiet]
  cases hs : (schedsPZk (tabsOf sts) : V _) with
  | error e => exact .inl ⟨rfl, by simp⟩
  | ok schs =>
    dsimp only
    have htabs : (tabsOf sts).size = sts.size := ZkTables.tabsOf_size sts
    obtain ⟨hssz, hsf⟩ : schs.size = (tabsOf sts).size ∧ ∀ i (h₁ : i < (tabsOf sts).size) (h₂ : i < schs.size),
        Zk.schedule (tabsOf sts)[i].m = .ok schs[i] := by
      have hA : (schedsPZk (tabsOf sts) : PS _).go #[] = (.ok schs, #[]) := by rw [schedsPZk_quiet, hs]
      exact Ret.go (Ret.mapM_array _ (fun st' v => Zk.schedule st'.m = .ok v) _ fun a _ =>
        Ret.monadLift (Zk.schedule a.m)) hA
    have hschs : ∀ i (hi : i < sts.size), Zk.schedule sts[i].m = .ok schs[i]! := fun i hi => by
      have := hsf i (by omega) (by omega)
      rw [← getElem!_pos schs i (by omega)] at this
      simp only [tabsOf, Array.getElem_map] at this
      exact this
    rw [go_bind, go_bind, commitsPZk_quiet]
    cases hc' : (commitsPZk (Hashes.ofSha512 h) (tabsOf sts) schs (tablesP (tabsOf sts) trs) tape : V _) with
    | error e => exact .inl ⟨rfl, by simp⟩
    | ok cms =>
      dsimp only
      rw [go_bind, go_bind, link_go, cut_link_go]
      simp only [Array.size_empty, show ¬ n ≤ 0 by omega, ↓reduceIte]
      rw [afterLinkPZk_congr (asksCut coin n) _ spec _ _ _ _ _ _ _ (c1 := coin 0) (c1' := coinOf (resp 0))
        fun p hp' j hj => (hc 0 hn0 _ (by rw [hp] at hp'; nlinarith)).symm]
      exact (afterLinkPZk_agree resp coin hc h spec sts words schs trs tape _ cms _ hspec hp hm hreg hschs).2 _
        (by simp; omega)

end FlockComplete.PrefixAgree

namespace FlockComplete.PrefixAgree

open Flock Transcript FlockComplete FlockDeploy JsonRT ZkTables

/-- **A masked session that returns sent through its last rep** (`repStartZk`, `repPZk_exact`), at tables of
`14 ≤ m`, one spec table per statement. -/
theorem sessionPZk_size (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec)
    (sts : Array Flock.Stmt) (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk)
    (hspec : spec.tables.size = sts.size) (hsts : ∀ j (hj : j < sts.size), 14 ≤ sts[j].m)
    {r : String × Array ByteArray} {L : Array Claims.Msg}
    (hrun : (sessionPZk (asksS resp) (Hashes.ofSha512 h) spec (tabsOf sts) words trs tape).go #[] = (.ok r, L)) :
    L.size = repStartZk spec sts (2 * sts.size) := by
  unfold sessionPZk at hrun
  obtain ⟨schs, l1, hA, hrun⟩ := go_bind_ok hrun
  obtain ⟨hssz, hsf⟩ := Ret.go (Ret.mapM_array _ (fun st' v => Zk.schedule st'.m = .ok v) _ fun a _ =>
    Ret.monadLift (Zk.schedule a.m)) hA
  have hl1 := Still.go (Still.mapM_array (fun a l => by rw [monadLift_go]) _) hA
  subst hl1
  have htabs : (tabsOf sts).size = sts.size := tabsOf_size sts
  have htab : ∀ i (hi : i < sts.size), (tabsOf sts)[i]! = Setup.ofCircuit sts[i]! := fun i hi => by
    rw [getElem!_pos (tabsOf sts) i (by omega), getElem!_pos sts i hi]
    simp [tabsOf]
  have hv : ∀ i (hi : i < sts.size), Zk.schedule sts[i]!.m = .ok schs[i]! := fun i hi => by
    have := hsf i (by omega) (by omega)
    rw [← getElem!_pos (tabsOf sts) i (by omega), htab i hi, ← getElem!_pos schs i (by omega)] at this
    exact this
  dsimp only at hrun
  obtain ⟨cms, l2, hB, hrun⟩ := go_bind_ok hrun
  have hl2 := Still.go (Still.mapM_array (fun i => by
    exact Still.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _)) _) hB
  subst hl2
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  obtain ⟨⟨ss, ps⟩, l₄, hF, hrest⟩ := go_bind_ok hrun
  have hinv := forIn_array_idx _ _ (fun k (_ : Array (Array Round) × Array ByteArray) (l : Array Claims.Msg) =>
      l.size = repStartZk spec sts k) ?step (by simp [repStartZk]) hF
  case step =>
    rintro k a ⟨ss, ps⟩ l res l' ha hl e
    obtain ⟨hk, rfl⟩ := Array.getElem?_eq_some_iff.1 ha
    have hk' : k < 2 * sts.size := by rw [repStreams_size, hspec] at hk; exact hk
    have hi : k / 2 < sts.size := by omega
    rw [repStreams_getElem _ hk] at e
    obtain ⟨extras, l₁, e1, e⟩ := go_bind_ok e
    rw [monadLift_go] at e1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨hex, rfl⟩ := e1
    obtain ⟨ic, l₀, e0, e3⟩ := go_bind_ok e
    rw [Still.go (Still.of_pureS (innerCommitP_pureS _ _ _)) e0] at e3
    obtain ⟨⟨pf, tr'⟩, l₂, e2, e⟩ := go_bind_ok e3
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    rw [htab _ hi] at e2 hex
    have hm := hsts (k / 2) hi
    rw [← getElem!_pos sts (k / 2) hi] at hm
    obtain ⟨h25, -, hsm, hk0, -, hdim, hlv, -, -⟩ :=
      FlockSoundness.Discharge.ZkExec.zk_schedule_seams sts[k / 2]!.m schs[k / 2]!.1 schs[k / 2]!.2 (hv _ hi)
    obtain ⟨hn7, h2, -, -, -, -, -⟩ := hdim
    obtain ⟨hl₂, -⟩ := repPZk_exact (sch := schs[k / 2]!.1) (hm := hm) (hn := by rw [hn7, hsm]; omega)
      (hk0 := hk0) (hL := h2) (hq := fun i hi => (hlv i hi).q_ne_zero) (h := e2)
    have hx : extras.size = spec.points * sts[k / 2]!.regions.size := by
      rw [JsonRT.extraClaims_size hex]
      simp [Nat.mul_comm]
    refine ⟨_, rfl, ?_⟩
    rw [repStart_succ, ← repRoundsT_eq (hv _ hi) hx, hl₂, hl]
    have e6 : (Setup.ofCircuit sts[k / 2]!).fold.kLog = sts[k / 2]!.c.kLog := rfl
    have e7 : (Setup.ofCircuit sts[k / 2]!).m = sts[k / 2]!.m := rfl
    unfold repRoundsT piopZk
    rw [e6, e7, hx]
    omega
  dsimp only at hrest
  obtain ⟨rec, l₅, h5, h6⟩ := go_bind_ok hrest
  rw [recordJson_quietZ h] at h5
  obtain ⟨-, rfl⟩ := Prod.mk.inj h5
  simp only [go_pure, Prod.mk.injEq] at h6
  rw [← h6.2, hinv, repStreams_size, hspec]

end FlockComplete.PrefixAgree

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

open _root_.Flock FlockSoundness.Refine in
/-- What `Stmt.setupH` returns has `PT_LOCAL + nbl` point coordinates. -/
theorem setupH_mPts {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) : st.mPts = PT_LOCAL + st.nbl := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

open _root_.Flock FlockDeploy FlockComplete in
/-- **At a `Live` statement the session links at two points of at most 64 coordinates, and every table has at most
1,024 regions** (the scope check's `checkInRange`, as `Case.points_regions`). -/
theorem live_bounds {x : Msg} (hx : Live x) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hsx : setupOf true x = some (s, spec, sts)) :
    spec.points = 2 ∧ spec.mPts ≤ 64 ∧ ∀ j (hj : j < sts.size), sts[j].regions.size ≤ 1024 := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk Readable.relation) Readable.relation (readOf true) rfl
    (fun _ _ h => readOf_zk h) x).1 hx.1
  have hi := (setupOf_some hsx).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := s.read) hp
  obtain ⟨hck, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset := Ins.setup_read (setupOf_some hsx).2
  rw [← ht] at hset
  unfold Readable.Read.setup at hset
  rw [hpi] at hset
  have hone : ∀ j (hj : j < sts.size), sts[j].regions.size ≤ 1024 ∧ sts[j].mPts ≤ 64 := fun j hj => by
    obtain ⟨d, hd, -⟩ := (HiddenExec.setupTables_setupOne hset).2 j hj
    rcases Bool.eq_false_or_eq_true (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count j).hiddenOutputs with hh | hh
    · rw [HiddenExec.setupOne_hidden hh] at hd
      have hS := FlockSoundness.Discharge.Hidden.setupHidden_spec' hd
      refine ⟨FlockVBridge.Family.spec_regions_le hS, ?_⟩
      rw [hS.mPts]
      exact ((checkInRange_ok _ _ _).1 hS.inRange).2.2
    · rw [HiddenExec.setupOne_public hh] at hd
      refine ⟨FlockVBridge.Family.setupH_regions_le hd, ?_⟩
      rw [setupH_mPts hd]
      exact ((checkInRange_ok _ _ _).1 (FlockVBridge.Family.setupH_inRange hd)).2.2
  obtain ⟨hpts, hmp⟩ := HiddenExec.setupTables_pointsAny hset
  obtain ⟨h0, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  refine ⟨hpts, ?_, fun j hj => (hone j hj).1⟩
  rw [← hmp 0 h0]
  exact (hone 0 h0).2

/-- **The cut run answers by the openings** (`PrefixLive`). -/
theorem prefixLive : PrefixLive := by
  intro x hx w _ ω ops hpos hlt s spec hsx
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  obtain ⟨hp, hmp, hreg⟩ := live_bounds hx hsx
  have hK := refK_eq hsx
  have hc : ∀ k < ops.length, ∀ i < 65536, FlockComplete.coinOf ((ops.map fun p => coinOfD p.2).getD k []) i =
      slotAt (padOps (x := x) ops) k i := fun k hk i hi => (slotAt_padOps_lt ops hk (by omega) hi).symm
  have hcpl : FlockComplete.PrefixAgree.Couple ops.length (liveRunAt x w ω ops spec)
      (runExe x w ω (padOps (x := x) ops) ops.length spec) :=
    FlockComplete.PrefixAgree.sessionPZk_agree _ _ hc _root_.Flock.Sha512.hash spec (stsOf x) _ _ ω hpos hspec hp
      hmp (fun j hj => ⟨(hall j hj).1, (hall j hj).2.1⟩) hreg
  have hret : ∀ r, (liveRunAt x w ω ops spec).1 = .ok r → ops.length < (liveRunAt x w ω ops spec).2.size := by
    intro r hr
    have hsz := FlockComplete.PrefixAgree.sessionPZk_size _ _root_.Flock.Sha512.hash spec (stsOf x) _ _ ω hspec
      (fun j hj => by have := (hall j hj).1; omega) (r := r) (L := (liveRunAt x w ω ops spec).2)
      (by rw [← hr]; rfl)
    rw [hsz, FlockComplete.ZkTables.repStart_last, ← hK]
    exact hlt
  generalize liveRunAt x w ω ops spec = L at hcpl hret ⊢
  generalize runExe x w ω (padOps (x := x) ops) ops.length spec = E at hcpl ⊢
  rcases hcpl with ⟨rfl, hs⟩ | ⟨-, hs, bs, hbs⟩
  · refine ⟨Iff.rfl, fun _ => rfl, fun _ => ?_⟩
    rcases hL : L.1 with e | r
    · exact ⟨e, rfl⟩
    · exact absurd hs (not_le.2 (hret r hL))
  · have hsz : E.2.size + bs.size = L.2.size := by rw [hbs, Array.size_append]
    refine ⟨⟨fun _ => by omega, fun _ => by omega⟩, fun _ => ?_, fun h => by omega⟩
    rw [hbs, getElem!_pos (E.2 ++ bs) ops.length (by rw [Array.size_append]; omega),
      getElem!_pos E.2 ops.length (by omega), Array.getElem_append_left]

end Masked

end Claims.ZKMal
