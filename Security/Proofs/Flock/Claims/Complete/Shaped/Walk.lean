import Proofs.Flock.Claims.Complete.Shaped.Commit

/-!
# The honest opening's lengths: the Ligerito walk

`ligeritoP_shape`: at a schedule the walk can follow (`LigSched`: a first level of at least one challenge, at least one
recursive level, `n = k0 + ext + 3R`, each level's cap within its depth, its seam, and four lanes past level 0), with a
level-0 commitment of the shape the opening reads and a table of `2^n` entries, the honest opening has every length
`decodeProof` checks of it (`LigOk`, the opening's part of `Lengths`).

The walk follows `ligeritoP` step by step on `Total.H`: `challenges0_H` is level 0's `k0` challenges; the recursive
levels' loop keeps, at level `i`, the dimension `ext + 3(R − i) + 1` still to fold, the lists' lengths `meas` (`k0 + 4 + 7i`
messages, `i + 1` query and consistency nonces, `4 + 3i` claim nonces, `3 + 2i` out-of-domain values, a table of
`2^dim`), the commitment it opens next (`CommitOk`), and the caps and openings so far; its last level sends `yr`, the
split of the folded table, of `2^(ext + 1)` entries.
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy FlockSoundness.Refine FlockSoundness.Discharge

theorem Total.lift_stQ {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {σ α : Type} {x : M α}
    {Q : α → Prop} (h : C.T x Q) : (C.st σ).T (StateT.lift x) Q :=
  fun s => C.bind h fun a ha => C.pure (a, s) ha

theorem commitP_shapeS (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) {hl : ℕ}
    (hn : ∀ l r v, (ms.opsWith (Hashes.ofSha512 h : Hashes V)).node l r = .ok v → v.size = hl)
    (hlf : ∀ row salt v, (ms.opsWith (Hashes.ofSha512 h : Hashes V)).leaf row salt = .ok v → v.size = hl)
    (hcap : lv.cap ≤ lv.d) :
    totS.T (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt) fun cm => CommitOk hl ms.saltBytes lv cm := fun l =>
  let ⟨cm, hcm⟩ := commitP_ok h ms lv g salt
  ⟨cm, l, by rw [commitP_pureS h ms lv g salt l, hcm], commitP_shape hn hlf hcap hcm⟩

theorem pow_half {k : ℕ} (hk : 1 ≤ k) : 2 ^ k / 2 = 2 ^ (k - 1) := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [Nat.pow_succ, Nat.mul_div_cancel _ (by norm_num), Nat.add_sub_cancel]

/-- The level-0 challenges: `k0` of them, the last splitting. -/
theorem challenges0_H (resp : ℕ → Claims.Msg) (k0 n : ℕ) (hk : 1 ≤ k0) (hkn : k0 ≤ n) (a b c d e : ℕ) (z : Bool) :
    totT.H (forIn [:k0] PUnit.unit fun j (_ : PUnit) => do
        let _ ← challengeP (asksS resp) (j + 1 == k0)
        pure (ForInStep.yield PUnit.unit) : LP PS PUnit)
      (fun s => meas s = ⟨a, b, c, d, e, 2 ^ n, z⟩) fun _ s => meas s = ⟨a + k0, b, c, d, e, 2 ^ (n - k0 + 1), z⟩ := by
  refine Total.H.pre (Total.H.mono (Total.H.forIn_upto k0 PUnit.unit
    (fun j _ s => meas s = ⟨a + j, b, c, d, e, if j < k0 then 2 ^ (n - j) else 2 ^ (n - k0 + 1), z⟩) ?_) ?_) ?_
  · intro j hj u
    refine Total.H.bind (challengeP_H resp _ _) fun _ => Total.H.pure _ fun s hs => ⟨_, rfl, ?_⟩
    rw [hs]
    simp only [show j < k0 from hj, ↓reduceIte, Ms.mk.injEq, true_and, and_true]
    refine ⟨by omega, ?_⟩
    rw [pow_half (by omega)]
    by_cases hj1 : j + 1 = k0
    · simp only [hj1, beq_self_eq_true, ↓reduceIte, lt_self_iff_false]
      rw [← Nat.pow_succ']
      congr 1
      omega
    · simp only [show (j + 1 == k0) = false from by simpa using hj1, Bool.false_eq_true, ↓reduceIte,
        show j + 1 < k0 by omega, Nat.sub_sub]
  · intro _ s hs
    rw [hs]
    simp
  · intro s hs
    rw [hs]
    simp [show 0 < k0 by omega]


theorem Total.H.intro {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ α : Type} {x : StateT σ M α}
    {pre : σ → Prop} {post : α → σ → Prop} (h : ∀ s, pre s → C.H x pre post) : C.H x pre post :=
  fun s hs => h s hs s hs

theorem Total.H.forIn_list_keep {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ α β : Type}
    (l : List α) (b : β) {f : α → β → StateT σ M (ForInStep β)} {pre : σ → Prop}
    (h : ∀ a b, C.H (f a b) pre fun _ s => pre s) : C.H (forIn l b f) pre fun _ s => pre s := by
  induction l generalizing b with
  | nil => exact Total.H.pure b fun _ h => h
  | cons a l ih =>
    rw [List.forIn_cons]
    refine Total.H.bind (h a b) fun r => ?_
    cases r with
    | done b => exact Total.H.pure b fun _ h => h
    | yield b => exact ih b

theorem Total.H.forIn_array_keep {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ α β : Type}
    (xs : Array α) (b : β) {f : α → β → StateT σ M (ForInStep β)} {pre : σ → Prop}
    (h : ∀ a b, C.H (f a b) pre fun _ s => pre s) : C.H (forIn xs b f) pre fun _ s => pre s := by
  rw [← Array.forIn_toList]
  exact Total.H.forIn_list_keep _ b h

theorem H_liftTQ {α : Type} {x : TMW PS α} {pre : LigState → Prop} {Q : α → Prop} (hx : totT.T x Q) :
    totT.H (liftT x) pre fun a s => pre s ∧ Q a :=
  Total.H.lift hx

theorem Total.H.fact {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ α : Type} {x : StateT σ M α}
    {pre : σ → Prop} {p : Prop} {post : α → σ → Prop} (h : p → C.H x pre post) :
    C.H x (fun s => pre s ∧ p) post :=
  fun s hs => h hs.2 s hs.1

theorem Total.H.fact_l {M : Type → Type} [Monad M] [MonadLiftT V M] {C : Total M} {σ α : Type} {x : StateT σ M α}
    {pre : σ → Prop} {p : Prop} {post : α → σ → Prop} (h : p → C.H x pre post) :
    C.H x (fun s => p ∧ pre s) post :=
  fun s hs => h hs.1 s hs.2

theorem Ms.eq {a b c d e f a' b' c' d' e' f' : ℕ} {z : Bool} (h1 : a = a') (h2 : b = b') (h3 : c = c') (h4 : d = d')
    (h5 : e = e') (h6 : f = f') : (⟨a, b, c, d, e, f, z⟩ : Ms) = ⟨a', b', c', d', e', f', z⟩ := by
  subst_vars
  rfl

theorem Ms.eq' {a b c d e f a' b' c' d' e' f' : ℕ} {z z' : Bool} (h7 : z = z') (h1 : a = a') (h2 : b = b')
    (h3 : c = c') (h4 : d = d') (h5 : e = e') (h6 : f = f') :
    (⟨a, b, c, d, e, f, z⟩ : Ms) = ⟨a', b', c', d', e', f', z'⟩ := by
  subst_vars
  rfl

theorem pow_dbl {k : ℕ} (hk : 1 ≤ k) : 2 * 2 ^ (k - 1) = 2 ^ k := by
  rw [← Nat.pow_succ']
  congr 1
  omega

/-- What the walk needs of a schedule. -/
structure LigSched (s : Schedule) : Prop where
  k0 : 1 ≤ s.k0
  two : 1 ≤ s.levels.size - 1
  dims : s.n = s.k0 + s.ext + 3 * (s.levels.size - 1)
  cap : ∀ i < s.levels.size, (s.levels.getD i default).cap ≤ (s.levels.getD i default).d
  seam : ∀ i < s.levels.size, LevelSeam (s.levels.getD i default)
  lanes : ∀ i < s.levels.size, 1 ≤ i → (s.levels.getD i default).lanes = 4

/-- The opening's part of `Lengths`. -/
def LigOk (sh : Shape) (lg : LigeritoProof) : Prop :=
  let s := sh.sched
  let R := s.levels.size - 1
  let lvl := fun (i : Nat) => s.levels.getD i default
  ZkView.CapOk sh (lvl 0) lg.initialCap ∧ ZkView.LevelOk sh (lvl 0) (2 ^ (lvl 0).lanes) lg.initialProof ∧
    lg.recursiveCaps.size = R ∧
    (∀ i (h : i < lg.recursiveCaps.size), ZkView.CapOk sh (lvl (i + 1)) lg.recursiveCaps[i]) ∧
    lg.recursiveProofs.size = R - 1 ∧
    (∀ i (h : i < lg.recursiveProofs.size), ZkView.LevelOk sh (lvl (i + 1)) 16 lg.recursiveProofs[i]) ∧
    lg.finalProof.yr.size = 2 ^ (s.ext + 1) ∧
    ZkView.LevelOk sh (lvl R) 16 ⟨lg.finalProof.rows, lg.finalProof.siblings, lg.finalProof.salts⟩ ∧
    lg.messages.size = s.k0 + 8 + 7 * (R - 1) ∧ lg.queryNonces.size = R + 1 ∧ lg.oodValues.size = 1 + 2 * R ∧
    lg.claimNonces.size = 3 * R + 2 ∧ lg.consistencyNonces.size = R + 1 ∧ ZkView.NoncesZero lg

theorem ligeritoP_shape (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) {sh : Shape} {ms : MerkleScheme}
    (hn : ∀ l r v, (ms.opsWith (Hashes.ofSha512 h : Hashes V)).node l r = .ok v → v.size = sh.hashLen)
    (hlf : ∀ row salt v, (ms.opsWith (Hashes.ofSha512 h : Hashes V)).leaf row salt = .ok v → v.size = sh.hashLen)
    (hsl : (sh.saltLen = none ∧ ms.saltBytes = 0) ∨ sh.saltLen = some ms.saltBytes)
    (hs : LigSched sh.sched) (T : F128) {cm0 : Commit}
    (hc0 : CommitOk sh.hashLen ms.saltBytes (sh.sched.levels.getD 0 default) cm0) {t : Array F128}
    (ht : t.size = 2 ^ sh.sched.n) (b : Array F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    totT.T (ligeritoP (asksS resp) (Hashes.ofSha512 h) ms sh.sched T cm0 t b salt nonce) (LigOk sh) := by
  obtain ⟨hk, hR, hdim, hcap, hseam, hlanes⟩ := hs
  unfold ligeritoP
  extract_lets n R lv sd0 rps0 fin0 f0 prog
  refine Total.bind' _ (Total.liftTM _ _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  refine Total.bind' _ (Total.liftTM _ _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  refine Total.bind' _ (Total.liftTM _ _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  have hprog : totT.H prog (fun s => meas s = ⟨0, 0, 0, 0, 0, 2 ^ n, true⟩) fun r s =>
      ZkView.LevelOk sh (lv 0) (2 ^ (lv 0).lanes) r.2.2.2 ∧ r.1.size = R ∧
      (∀ i (h : i < r.1.size), ZkView.CapOk sh (lv (i + 1)) r.1[i]) ∧ r.2.1.size = R - 1 ∧
      (∀ i (h : i < r.2.1.size), ZkView.LevelOk sh (lv (i + 1)) 16 r.2.1[i]) ∧
      r.2.2.1.yr.size = 2 ^ (sh.sched.ext + 1) ∧
      ZkView.LevelOk sh (lv R) 16 ⟨r.2.2.1.rows, r.2.2.1.siblings, r.2.2.1.salts⟩ ∧
      ∃ tb, meas s = ⟨sh.sched.k0 + 8 + 7 * (R - 1), R + 1, R + 1, 3 * R + 2, 1 + 2 * R, tb, true⟩ := by
    dsimp only [prog]
    have hc : ∀ i < sh.sched.levels.size, ∀ g sl, totS.T (commitP (Hashes.ofSha512 h : Hashes PS) ms (lv i) g sl)
        fun cm => CommitOk sh.hashLen ms.saltBytes (lv i) cm := fun i hi g sl =>
      commitP_shapeS h ms (lv i) g sl hn hlf (hcap i hi)
    have hop : ∀ i < sh.sched.levels.size, ∀ cm coins, CommitOk sh.hashLen ms.saltBytes (lv i) cm →
        ZkView.LevelOk sh (lv i) (2 ^ (lv i).lanes) (openP ms cm (lv i) (positions (lv i) coins)) :=
      fun i hi cm coins hcm =>
      openP_ok hcm (positions_facts _ (hseam i hi) coins).1 (positions_facts _ (hseam i hi) coins).2 hsl
    have hL : 2 ≤ sh.sched.levels.size := by omega
    refine Total.H.bind (H_liftT (sampleVecP_tot resp n)) fun z => ?_
    refine Total.H.bind (H_modify _ (fun v => { v with oods := v.oods + 1 }) (fun s => by simp [meas]) _) fun _ => ?_
    refine Total.H.bind (H_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
    refine Total.H.bind (powP_H2 nonce _) fun _ => ?_
    refine Total.H.bind (H_liftT (sampleP_tot resp)) fun b0 => ?_
    refine Total.H.bind (H_modify _ id (by intro s; rfl) _) fun _ => ?_
    refine Total.H.get_bind fun st => ?_
    refine Total.H.bind (sendMsg_H _ _) fun _ => ?_
    refine Total.H.bind (challenges0_H resp _ n hk (by omega) _ _ _ _ _ _) fun _ => ?_
    refine Total.H.bind Total.H.get fun dl => ?_
    refine Total.H.bind (H_liftTQ (Total.lift_stQ _ (hc 1 (by omega) _ (salt 1)))) fun cm1 => ?_
    refine Total.H.fact fun hcm1 => ?_
    refine Total.H.pre
      (pre := fun s => meas s = ⟨0 + 1 + sh.sched.k0, 0, 0, 0 + 1, 0 + 1, 2 ^ (n - sh.sched.k0 + 1), true⟩) ?_
      fun s hs => hs.2
    refine Total.H.bind (H_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
    refine Total.H.bind (oodIntroP_H resp nonce _ _ _) fun _ => ?_
    refine Total.H.bind (oodIntroP_H resp nonce _ _ _) fun _ => ?_
    refine Total.H.bind (queryP_H resp nonce ms (lv 0) cm0 _) fun x0 => ?_
    refine Total.H.fact fun ⟨c0, hx0⟩ => ?_
    have hop0 : ZkView.LevelOk sh (lv 0) (2 ^ (lv 0).lanes) x0.1 := hx0 ▸ hop 0 (by omega) cm0 c0 hc0
    refine Total.H.bind (sendMsg_H _ _) fun _ => ?_
    refine Total.H.bind (powP_H2 nonce _) fun _ => ?_
    refine Total.H.bind (H_liftT (sampleP_tot resp)) fun _ => ?_
    refine Total.H.bind (H_modify _ id (by intro s; rfl) _) fun _ => ?_
    refine Total.H.bind (Total.H.pre (Total.H.forIn_upto R _
      (fun i (q : ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof) s =>
        (i < R ∧ q.1 = sh.sched.ext + 3 * (R - i) + 1 ∧
          meas s = ⟨sh.sched.k0 + 4 + 7 * i, 1 + i, 1 + i, 4 + 3 * i, 3 + 2 * i, 2 ^ q.1, true⟩ ∧
          CommitOk sh.hashLen ms.saltBytes (lv (i + 1)) q.2.1 ∧ q.2.2.1.size = i + 1 ∧
          (∀ j (h : j < q.2.2.1.size), ZkView.CapOk sh (lv (j + 1)) q.2.2.1[j]) ∧ q.2.2.2.1.size = i ∧
          (∀ j (h : j < q.2.2.2.1.size), ZkView.LevelOk sh (lv (j + 1)) 16 q.2.2.2.1[j])) ∨
        (i = R ∧ q.2.2.1.size = R ∧ (∀ j (h : j < q.2.2.1.size), ZkView.CapOk sh (lv (j + 1)) q.2.2.1[j]) ∧
          q.2.2.2.1.size = R - 1 ∧ (∀ j (h : j < q.2.2.2.1.size), ZkView.LevelOk sh (lv (j + 1)) 16 q.2.2.2.1[j]) ∧
          q.2.2.2.2.yr.size = 2 ^ (sh.sched.ext + 1) ∧
          ZkView.LevelOk sh (lv R) 16 ⟨q.2.2.2.2.rows, q.2.2.2.2.siblings, q.2.2.2.2.salts⟩ ∧
          ∃ tb, meas s = ⟨sh.sched.k0 + 8 + 7 * (R - 1), R + 1, R + 1, 3 * R + 2, 1 + 2 * R, tb, true⟩)) ?_) ?_)
        fun q => ?_
    · intro i hi q
      refine Total.H.intro fun s0 hI => ?_
      rcases hI with ⟨-, hsd, -, hcm, hcs, hca, hrs, hrp⟩ | ⟨hiR, -⟩
      swap
      · omega
      refine Total.H.pre (pre := fun s =>
          meas s = ⟨sh.sched.k0 + 4 + 7 * i, 1 + i, 1 + i, 4 + 3 * i, 3 + 2 * i, 2 ^ q.1, true⟩) ?_ fun s hs => by
        rcases hs with h | h
        · exact h.2.2.1
        · exact absurd h.1 (by omega)
      have hsd4 : 4 ≤ q.1 := by omega
      refine Total.H.bind (Total.H.pre (Total.H.forIn_upto 4 f0
        (fun j (f : Array F256) s => (j = 4 → f.size = 2 ^ (q.1 - 4)) ∧
          meas s = ⟨sh.sched.k0 + 4 + 7 * i + j, 1 + i, 1 + i, 4 + 3 * i, 3 + 2 * i,
            if j < 4 then 2 ^ (q.1 - j) else 2 ^ (q.1 - 3), true⟩) ?_) ?_) fun f => ?_
      · intro j hj g
        refine Total.H.fact_l fun _ => ?_
        split
        · next hj3 =>
          have e3 : j = 3 := by simpa using hj3
          subst e3
          refine Total.H.bind (H_liftT (sampleKP_tot resp)) fun r => ?_
          refine Total.H.bind (H_modify _ (fun v => { v with tab := v.tab / 2 })
            (fun s => by simp [meas, foldK_size]) _) fun _ => ?_
          refine Total.H.bind Total.H.get fun dl => ?_
          refine Total.H.intro fun s hs => ?_
          have hdl : dl.tab.size = 2 ^ (q.1 - 4) := by
            obtain ⟨rfl, e⟩ := hs
            have := congrArg Ms.tab e
            simp only [meas] at this
            rw [this, pow_half (by omega), Nat.sub_sub]
          refine Total.H.pre (Total.H.bind (H_modify _ (fun v => { v with tab := 2 * v.tab })
            (fun s => by simp [meas, embedF_size, splitK_size]) _) fun _ => ?_) fun s hs => hs.2
          refine Total.H.get_bind fun st => ?_
          refine Total.H.bind (sendMsg_H _ _) fun _ => ?_
          refine Total.H.pure _ fun s hs => ⟨_, rfl, fun _ => hdl, ?_⟩
          rw [hs]
          refine Ms.eq ?_ ?_ ?_ ?_ ?_ ?_ <;> (try dsimp only)
          simp only [↓reduceIte, lt_self_iff_false]
          rw [pow_half (by omega), pow_dbl (by omega)]
        · next hj3 =>
          have hj3' : j + 1 ≠ 4 := by simpa using hj3
          refine Total.H.bind (challengeP_H resp false _) fun _ => ?_
          refine Total.H.pure _ fun s hs => ⟨_, rfl, fun h => absurd h hj3', ?_⟩
          rw [hs]
          refine Ms.eq ?_ ?_ ?_ ?_ ?_ ?_ <;> (try dsimp only) <;> (try omega)
          simp only [show j + 1 < 4 by omega, ↓reduceIte, Bool.false_eq_true]
          rw [pow_half (by omega), Nat.sub_sub]
      · intro s hs
        exact ⟨fun h => absurd h (by omega), by rw [hs]; simp⟩
      refine Total.H.fact_l fun hf => ?_
      have hf := hf rfl
      refine Total.H.pre (pre := fun s =>
          meas s = ⟨sh.sched.k0 + 8 + 7 * i, 1 + i, 1 + i, 4 + 3 * i, 3 + 2 * i, 2 ^ (q.1 - 3), true⟩) ?_ fun s hs => by
        rw [hs]
        simp only [Ms.mk.injEq, lt_self_iff_false, ↓reduceIte, and_true]
        omega
      split
      · next hlast =>
        have hiR : i + 1 ≠ R := by simpa using hlast
        refine Total.H.bind (H_liftTQ (Total.lift_stQ _ (hc (i + 2) (by omega) _ (salt (i + 2))))) fun cmN => ?_
        refine Total.H.fact fun hcmN => ?_
        refine Total.H.bind (H_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
        refine Total.H.bind (oodIntroP_H resp nonce _ _ _) fun _ => ?_
        refine Total.H.bind (oodIntroP_H resp nonce _ _ _) fun _ => ?_
        refine Total.H.bind (queryP_H resp nonce ms (lv (i + 1)) _ _) fun x => ?_
        refine Total.H.fact fun ⟨cs, hx⟩ => ?_
        have hopx : ZkView.LevelOk sh (lv (i + 1)) 16 x.1 := by
          have e : (lv (i + 1)).lanes = 4 := hlanes (i + 1) (by omega) (by omega)
          have := hop (i + 1) (by omega) q.2.1 cs hcm
          rw [e] at this
          rw [hx]
          exact this
        refine Total.H.bind (sendMsg_H _ _) fun _ => ?_
        refine Total.H.bind (powP_H2 nonce _) fun _ => ?_
        refine Total.H.bind (H_liftT (sampleP_tot resp)) fun _ => ?_
        refine Total.H.bind (H_modify _ id (by intro s; rfl) _) fun _ => ?_
        refine Total.H.pure _ fun s hs => ⟨_, rfl, Or.inl ⟨by omega, by omega, ?_, hcmN, by simp [hcs], ?_,
          by simp [hrs], ?_⟩⟩
        · rw [hs]
          refine Ms.eq ?_ ?_ ?_ ?_ ?_ ?_ <;> (try dsimp only) <;> (try omega)
          congr 1
          omega
        · intro j hj
          rw [Array.getElem_push]
          split
          · exact hca j ‹_›
          · obtain rfl : j = i + 1 := by simp at hj; omega
            exact ⟨hcmN.2.2.2.2.2.1, hcmN.2.2.2.2.2.2⟩
        · intro j hj
          rw [Array.getElem_push]
          split
          · exact hrp j ‹_›
          · obtain rfl : j = i := by simp at hj; omega
            exact hopx
      · next hlast =>
        have hiR : i + 1 = R := by simpa using hlast
        refine Total.H.bind (Total.H.forIn_array_keep _ _ fun _ _ =>
          Total.H.bind (H_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => Total.H.pure _ fun _ h => h) fun _ => ?_
        refine Total.H.bind (queryP_H resp nonce ms (lv R) _ _) fun x => ?_
        refine Total.H.fact fun ⟨cs, hx⟩ => ?_
        have hopx : ZkView.LevelOk sh (lv R) 16 x.1 := by
          have e : (lv R).lanes = 4 := hlanes R (by omega) (by omega)
          have := hop R (by omega) q.2.1 cs (by rw [← hiR]; exact hcm)
          rw [e] at this
          rw [hx]
          exact this
        refine Total.H.bind (powP_H2 nonce _) fun _ => ?_
        refine Total.H.bind (H_liftT (sampleP_tot resp)) fun _ => ?_
        refine Total.H.pure _ fun s hs => ⟨_, rfl, Or.inr ⟨by omega, by dsimp only; omega, hca,
          by dsimp only; omega, hrp, ?_, hopx,
          2 ^ (q.1 - 3), ?_⟩⟩
        · show (splitK f).size = _
          rw [splitK_size, hf, ← Nat.pow_succ']
          congr 1
          omega
        · rw [hs]
          refine Ms.eq ?_ ?_ ?_ ?_ ?_ ?_ <;> (try dsimp only) <;> omega
    · intro s hs
      refine Or.inl ⟨by omega, by simp only [sd0, n]; omega, ?_, hcm1, rfl, ?_, rfl,
        fun j hj => absurd hj (by simp [rps0])⟩
      · rw [hs]
        refine Ms.eq ?_ ?_ ?_ ?_ ?_ ?_ <;> (try dsimp only)
        omega
      · intro j hj
        obtain rfl : j = 0 := by simpa using hj
        exact ⟨hcm1.2.2.2.2.2.1, hcm1.2.2.2.2.2.2⟩
    · refine Total.H.pure _ fun s hs => ?_
      rcases hs with ⟨hlt, -⟩ | ⟨-, h1, h2, h3, h4, h5, h6, h7⟩
      · omega
      · exact ⟨hop0, h1, h2, h3, h4, h5, h6, h7⟩
  refine totT.bind (hprog _ (by simp [meas, embedF_size, ht, n, Array.all_empty])) fun p hp => ?_
  obtain ⟨⟨caps, rps, fin, op0⟩, st⟩ := p
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, tb, h7⟩ := hp
  have hz := zeros_of_meas (congrArg Ms.z h7)
  simp only [meas, Ms.mk.injEq] at h7
  obtain ⟨e1, e2, e3, e4, e5, -⟩ := h7
  exact totT.pure _ ⟨⟨hc0.2.2.2.2.2.1, hc0.2.2.2.2.2.2⟩, h0, h1, h2, h3, h4, h5, h6, e1, e2, e5, e4, e3, hz⟩

end FlockComplete.JsonRT
