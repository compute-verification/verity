import Proofs.Flock.Claims.Complete.Replay.Steps
import Proofs.Flock.VBridge.Algebra.Ligerito

/-!
# The verifier's Ligerito steps on the model's rounds

`Flock.Ligerito`, `Flock.Merkle` from the verifier's side. The verifier's Ligerito (`ligeritoWith`) reads every value the prover sends
from the proof `p` and every coin from the rounds. The model runs it over `p` and the coins `cf k` of the squeeze at
transcript position `k`: its state, and a transcript whose rounds are what the prover records at each squeeze
(`afterSq`), so that the verifier, on rounds carrying those, reads back each squeeze's coins.

* **Runs** (`Runs F k₀ y st t b st' t'`): the step `y` returns `b` from `(st, t)` at `(st', t')` on every rounds `R` that
  carry `F`'s from `k₀` on, while `t'`'s rounds begin `F`; the model's transcript only grows (`Grows`). Runs compose
  (`Runs.bind`, `Runs.loop`), so one final `F` serves every squeeze.
* **Steps**: the transcript's (`Runs.sampleVec`, `Runs.sample`, `Runs.sampleK`, `Runs.absorbF`, `Runs.absorbBytes`,
  `Runs.pow`) and the Ligerito state's (`Runs.nextMsg`, `Runs.nextOod`, `Runs.nonceFrom`, `Runs.claimPow`), when the list
  read has the entry; a challenge (`chal`, `Runs.challenge`), `n` of them (`chals`, `Runs.challenges`), an out-of-domain
  intro (`ood`, `Runs.oodIntro`), and a level's query (`query`, `Runs.queryLevel`) when its Merkle check passes.
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace Lg

/-- The transcript with its pending bytes mapped by `f`. -/
def ap (f : ByteArray → ByteArray) (t : Transcript) : Transcript := ⟨t.rounds, t.k, f t.pending⟩

/-- The pending bytes after a sumcheck message (`absorbK` of each half). -/
def pK (m : F256 × F256) (P : ByteArray) : ByteArray := pSlice #[m.2.c0, m.2.c1] (pSlice #[m.1.c0, m.1.c1] P)

/-- The pending bytes after a grinding nonce (`pow`). -/
def pN (v : UInt64) (P : ByteArray) : ByteArray := pBytes (le64 v) P

/-- The verifier's transcript on the rounds `R`, at the model's position and pending bytes. -/
def on (R : Array Round) (t : Transcript) : Transcript := ⟨R, t.k, t.pending⟩

/-- The `n` coins of the squeeze at `t`. -/
def vec (cf : ℕ → ℕ → F128) (t : Transcript) (n : ℕ) : Array F128 := (Array.range n).map (cf t.k)

theorem getElem?_of {α : Type} [Inhabited α] {xs : Array α} {i : ℕ} (h : i < xs.size) : xs[i]? = some xs[i]! := by
  rw [getElem!_pos xs i h, Array.getElem?_eq_getElem h]

/-! ## Runs -/

/-- `t'` extends `t`: its rounds begin with `t`'s, it is no earlier, and it keeps every round recorded. -/
def Grows (t t' : Transcript) : Prop :=
  Prefix t.rounds t'.rounds ∧ t.k ≤ t'.k ∧ (t.k = t.rounds.size → t'.k = t'.rounds.size)

theorem Grows.refl (t : Transcript) : Grows t t := ⟨Prefix.refl _, le_rfl, id⟩

theorem Grows.trans {a b c : Transcript} (h₁ : Grows a b) (h₂ : Grows b c) : Grows a c :=
  ⟨h₁.1.trans h₂.1, h₁.2.1.trans h₂.2.1, fun h => h₂.2.2 (h₁.2.2 h)⟩

theorem Grows.ap (f : ByteArray → ByteArray) (t : Transcript) : Grows t (ap f t) := Grows.refl t

theorem Grows.afterSq (t : Transcript) (kind : UInt8) (n : ℕ) (coin : ℕ → F128) : Grows t (afterSq t kind n coin) :=
  ⟨Prefix.push _ _, Nat.le_succ _, fun h => by simp [Replay.afterSq, h]⟩

/-- **The step `y` runs on the model's rounds**: from `(st, t)` it returns `b` at `(st', t')`, on every rounds `R` that
carry `F`'s from `k₀` on, once `t` has recorded its rounds, is past `k₀`, and `t'`'s rounds begin `F`. -/
def Runs {β : Type} (F : Array Round) (k₀ : ℕ) (y : LM β) (st : LState) (t : Transcript) (b : β) (st' : LState)
    (t' : Transcript) : Prop :=
  Grows t t' ∧ (k₀ ≤ t.k → t.k = t.rounds.size → Prefix t'.rounds F → ∀ R, Agrees R F k₀ →
    y st (on R t) = .ok ((b, st'), on R t'))

variable {F : Array Round} {k₀ : ℕ} {α β σ : Type}

theorem Runs.bind {x : LM α} {f : α → LM β} {st st₁ st₂ : LState} {t t₁ t₂ : Transcript} {a : α} {b : β}
    (hx : Runs F k₀ x st t a st₁ t₁) (hf : Runs F k₀ (f a) st₁ t₁ b st₂ t₂) :
    Runs F k₀ (x >>= f) st t b st₂ t₂ := by
  refine ⟨hx.1.trans hf.1, fun hk ht hp R hR => ?_⟩
  rw [Flock.RecAlgebra.lbind_of (hx.2 hk ht (hf.1.1.trans hp) R hR)]
  exact hf.2 (hk.trans hx.1.2.1) (hx.1.2.2 ht) hp R hR

theorem Runs.pure (b : β) (st : LState) (t : Transcript) : Runs F k₀ (Pure.pure b : LM β) st t b st t :=
  ⟨Grows.refl t, fun _ _ _ _ _ => rfl⟩

theorem Runs.cast {y : LM β} {st st₁ st₂ : LState} {t t₁ t₂ : Transcript} {b b' : β}
    (h : Runs F k₀ y st t b st₁ t₁) (hb : b = b') (hs : st₁ = st₂) (ht : t₁ = t₂) : Runs F k₀ y st t b' st₂ t₂ := by
  subst hb hs ht
  exact h

/-- A run from a start written another way. -/
theorem Runs.start {y : LM β} {st st₀ st' : LState} {t t₀ t' : Transcript} {b : β} (hs : st = st₀) (ht : t = t₀)
    (h : Runs F k₀ y st₀ t₀ b st' t') : Runs F k₀ y st t b st' t' := by
  subst hs ht
  exact h

theorem Runs.run {y : LM β} {st st' : LState} {t t' : Transcript} {b : β} (h : Runs F k₀ y st t b st' t') :
    Runs F k₀ (StateT.run y) st t b st' t' := h

/-- A step that moves no round. -/
theorem Runs.stay {y : LM β} {st st' : LState} {t t' : Transcript} {b : β} (hr : t'.rounds = t.rounds)
    (hk : t'.k = t.k) (h : ∀ R, y st (on R t) = .ok ((b, st'), on R t')) : Runs F k₀ y st t b st' t' :=
  ⟨⟨⟨#[], by rw [hr]; simp⟩, hk.ge, fun h => by rw [hr, hk]; exact h⟩, fun _ _ _ R _ => h R⟩

theorem Runs.get (st : LState) (t : Transcript) : Runs F k₀ (MonadState.get : LM LState) st t st st t :=
  Runs.pure st st t

theorem Runs.set (s st : LState) (t : Transcript) : Runs F k₀ (MonadStateOf.set s : LM PUnit) st t () s t :=
  Runs.pure () s t

theorem Runs.modify (g : LState → LState) (st : LState) (t : Transcript) :
    Runs F k₀ (_root_.modify g : LM PUnit) st t () (g st) t :=
  Runs.pure () (g st) t

theorem Runs.ite_pos {c : Prop} [Decidable c] {x y : LM β} {st st' : LState} {t t' : Transcript} {b : β} (hc : c)
    (h : Runs F k₀ x st t b st' t') : Runs F k₀ (if c then x else y) st t b st' t' := by
  rw [ite_eq_left hc]
  exact h

theorem Runs.ite_neg {c : Prop} [Decidable c] {x y : LM β} {st st' : LState} {t t' : Transcript} {b : β} (hc : ¬c)
    (h : Runs F k₀ y st t b st' t') : Runs F k₀ (if c then x else y) st t b st' t' := by
  rw [ite_eq_right hc]
  exact h

/-- A loop over `a, …, a + n − 1` whose step `j` runs from the `j`-th model value to the next and yields. -/
theorem Runs.loop (body : ℕ → σ → LM (ForInStep σ)) (S : ℕ → σ) (St : ℕ → LState) (T : ℕ → Transcript) :
    ∀ (n a : ℕ), (∀ j < n, Runs F k₀ (body (a + j) (S j)) (St j) (T j) (.yield (S (j + 1))) (St (j + 1)) (T (j + 1))) →
      Runs F k₀ (forIn (List.range' a n) (S 0) body) (St 0) (T 0) (S n) (St n) (T n)
  | 0, _, _ => Runs.pure _ _ _
  | n + 1, a, hb => by
    rw [List.range'_succ, List.forIn_cons]
    refine Runs.bind (hb 0 (Nat.succ_pos n)) ?_
    exact Runs.loop body (fun j => S (j + 1)) (fun j => St (j + 1)) (fun j => T (j + 1)) n (a + 1)
      fun j hj => by
        have := hb (j + 1) (by omega)
        rwa [show a + (j + 1) = a + 1 + j by omega] at this

/-! ## The transcript's steps -/

theorem lift_of {x : TM α} {a : α} {st : LState} {t t' : Transcript} (h : x t = .ok (a, t')) :
    (lift x : LM α) st t = .ok ((a, st), t') := by
  show (StateT.lift x st) t = _
  unfold StateT.lift
  rw [Flock.RecAlgebra.bind_of_ok h]
  rfl

/-- The round a squeeze at `t` pushes is carried at `t.k`, by rounds that carry `F` where it begins `F`. -/
theorem carries_of {R : Array Round} {t : Transcript} {kind : UInt8} {n : ℕ} {coin : ℕ → F128} (hk : k₀ ≤ t.k)
    (ht : t.k = t.rounds.size) (hp : Prefix (afterSq t kind n coin).rounds F) (hR : Agrees R F k₀) :
    Carries R t.k (sent t kind n coin) := by
  rw [ht]
  exact hR.at_push hp (ht ▸ hk)

theorem Runs.sampleVec (hs : Hashes V) (cf : ℕ → ℕ → F128) {n : ℕ} (hn : n ≠ 0) (st : LState) (t : Transcript) :
    Runs F k₀ (liftW (M := V) (sampleVecWith hs n)) st t (vec cf t n) st (afterSq t 2 n (cf t.k)) :=
  ⟨Grows.afterSq _ _ _ _, fun hk ht hp _ hR => lift_of (sampleVecWith_sent hs hn (carries_of hk ht hp hR))⟩

theorem Runs.sample (hs : Hashes V) (cf : ℕ → ℕ → F128) (st : LState) (t : Transcript) :
    Runs F k₀ (liftW (M := V) (sampleWith hs)) st t (cf t.k 0) st (afterSq t 1 1 (cf t.k)) :=
  ⟨Grows.afterSq _ _ _ _, fun hk ht hp _ hR => lift_of (sampleWith_sent hs (carries_of hk ht hp hR))⟩

theorem Runs.sampleK (hs : Hashes V) (cf : ℕ → ℕ → F128) (st : LState) (t : Transcript) :
    Runs F k₀ (liftW (M := V) (sampleKWith hs)) st t (⟨cf t.k 0, cf t.k 1⟩ : F256) st (afterSq t 2 2 (cf t.k)) :=
  ⟨Grows.afterSq _ _ _ _, fun hk ht hp _ hR => lift_of (sampleKWith_sent hs (carries_of hk ht hp hR))⟩

theorem Runs.absorbF (v : F128) (st : LState) (t : Transcript) :
    Runs F k₀ (lift (absorbF v)) st t () st (ap (pF v) t) :=
  Runs.stay rfl rfl fun _ => lift_of (Flock.RecAlgebra.absorbF_eq _ _)

theorem Runs.absorbBytes (b : ByteArray) (st : LState) (t : Transcript) :
    Runs F k₀ (lift (absorbBytes b)) st t () st (ap (pBytes b) t) :=
  Runs.stay rfl rfl fun _ => lift_of (Flock.RecAlgebra.absorbBytes_eq _ _)

theorem Runs.pow (v : UInt64) (st : LState) (t : Transcript) : Runs F k₀ (lift (pow v)) st t () st (ap (pN v) t) :=
  Runs.stay rfl rfl fun _ => lift_of (Flock.RecAlgebra.absorbBytes_eq _ _)

theorem absorbAll_of (ws : List F128) (st : LState) (R : Array Round) (k : ℕ) :
    ∀ P : ByteArray, (ForIn.forIn ws PUnit.unit fun w (_ : PUnit) => do
        lift (Transcript.absorbF w)
        Pure.pure (ForInStep.yield PUnit.unit) : LM PUnit) st ⟨R, k, P⟩ =
      .ok ((PUnit.unit, st), ⟨R, k, ws.foldl (fun P w => pF w P) P⟩) := by
  induction ws with
  | nil => intro P; rfl
  | cons w ws ih =>
    intro P
    rw [List.forIn_cons, Flock.RecAlgebra.lbind_of (Flock.RecAlgebra.lbind_of
      (lift_of (Flock.RecAlgebra.absorbF_eq w ⟨R, k, P⟩)))]
    exact ih _

/-- Each of `ws` absorbed in turn. -/
theorem Runs.absorbAll (ws : Array F128) (st : LState) (t : Transcript) :
    Runs F k₀ (ForIn.forIn ws PUnit.unit fun w (_ : PUnit) => do
        lift (Transcript.absorbF w)
        Pure.pure (ForInStep.yield PUnit.unit))
      st t PUnit.unit st (ap (fun P => ws.foldl (fun P w => pF w P) P) t) := by
  refine Runs.stay rfl rfl fun R => ?_
  rw [← Array.forIn_toList]
  refine (absorbAll_of ws.toList st R t.k t.pending).trans ?_
  rw [Array.foldl_toList]
  rfl

/-! ## The Ligerito state's steps -/

theorem Runs.nextMsg (p : LigeritoProof) (st : LState) (t : Transcript) (h : st.msg < p.messages.size) :
    Runs F k₀ (nextMsg p) st t p.messages[st.msg]! { st with msg := st.msg + 1 } (ap (pK p.messages[st.msg]!) t) := by
  refine Runs.stay rfl rfl fun _ => ?_
  unfold Flock.nextMsg
  rw [Flock.RecAlgebra.lget_bind_eq, getElem?_of h]
  rfl

theorem Runs.nextOod (p : LigeritoProof) (st : LState) (t : Transcript) (h : st.ood < p.oodValues.size) :
    Runs F k₀ (nextOod p) st t p.oodValues[st.ood]! { st with ood := st.ood + 1 } t := by
  refine Runs.stay rfl rfl fun _ => ?_
  unfold Flock.nextOod
  rw [Flock.RecAlgebra.lget_bind_eq, getElem?_of h]
  rfl

theorem Runs.nonceFrom (xs : Array UInt64) (i : ℕ) (what : String) (st : LState) (t : Transcript) (h : i < xs.size) :
    Runs F k₀ (nonceFrom xs i what) st t xs[i]! st t := by
  refine Runs.stay rfl rfl fun _ => ?_
  unfold Flock.nonceFrom
  rw [getElem?_of h]
  rfl

theorem Runs.claimPow (p : LigeritoProof) (st : LState) (t : Transcript) (h : st.cbn < p.claimNonces.size) :
    Runs F k₀ (claimPow p) st t () { st with cbn := st.cbn + 1 } (ap (pN p.claimNonces[st.cbn]!) t) := by
  unfold Flock.claimPow
  exact (Runs.get st t).bind ((Runs.nonceFrom _ _ _ st t h).bind ((Runs.set _ st t).bind (Runs.pow _ _ t)))

/-! ## A challenge -/

/-- **A challenge** from `(st, t)`: the coin `r`, the state after `s ← q(r)`, the next message `m` read and
`q ← (m, s)`, and the transcript after the squeeze and `m`. -/
def chal (p : LigeritoProof) (cf : ℕ → ℕ → F128) (st : LState) (t : Transcript) : F256 × LState × Transcript :=
  let r : F256 := ⟨cf t.k 0, cf t.k 1⟩
  let m := p.messages[st.msg]!
  (r, { { { st with s := st.q.eval r } with msg := st.msg + 1 } with q := Poly.ofMsg m (st.q.eval r) },
    ap (pK m) (afterSq t 2 2 (cf t.k)))

theorem Runs.challenge (hs : Hashes V) (p : LigeritoProof) (cf : ℕ → ℕ → F128) (st : LState) (t : Transcript)
    (hm : st.msg < p.messages.size) :
    Runs F k₀ (challengeWith hs p) st t (chal p cf st t).1 (chal p cf st t).2.1 (chal p cf st t).2.2 := by
  unfold challengeWith
  simp only [Flock.With.liftLM_V]
  exact (Runs.sampleK hs cf st t).bind ((Runs.modify _ _ _).bind ((Runs.nextMsg p _ _ (by exact hm)).bind
    ((Runs.modify _ _ _).bind (Runs.pure _ _ _))))

/-- **`n` challenges** from `(st, t)`, pushed onto `acc`. -/
def chals (p : LigeritoProof) (cf : ℕ → ℕ → F128) (acc : Array F256) (st : LState) (t : Transcript) :
    ℕ → Array F256 × LState × Transcript
  | 0 => (acc, st, t)
  | j + 1 =>
    let c := chals p cf acc st t j
    let x := chal p cf c.2.1 c.2.2
    (c.1.push x.1, x.2.1, x.2.2)

theorem chals_msg (p : LigeritoProof) (cf : ℕ → ℕ → F128) (acc : Array F256) (st : LState) (t : Transcript) :
    ∀ j, (chals p cf acc st t j).2.1.msg = st.msg + j
  | 0 => rfl
  | j + 1 => by simp only [chals, chal, chals_msg p cf acc st t j]; omega

theorem Runs.challenges (hs : Hashes V) (p : LigeritoProof) (cf : ℕ → ℕ → F128) (acc : Array F256) (st : LState)
    (t : Transcript) (n : ℕ) (hm : st.msg + n ≤ p.messages.size) :
    Runs F k₀ (ForIn.forIn (List.range' 0 n) acc fun _ chs => do
        let c ← challengeWith hs p
        Pure.pure (ForInStep.yield (chs.push c)))
      st t (chals p cf acc st t n).1 (chals p cf acc st t n).2.1 (chals p cf acc st t n).2.2 := by
  refine Runs.loop _ (fun j => (chals p cf acc st t j).1) (fun j => (chals p cf acc st t j).2.1)
    (fun j => (chals p cf acc st t j).2.2) n 0 fun j hj => ?_
  refine (Runs.challenge hs p cf _ _ (by rw [chals_msg]; omega)).bind ?_
  exact Runs.pure _ _ _

/-! ## An out-of-domain intro -/

/-- **An out-of-domain intro** of `dim` coordinates at level `split`, from `(st, t)`: the point `z`, the value `y` read,
the intro message `m`, the claim nonce, and the batching coin `b`. -/
def ood (p : LigeritoProof) (cf : ℕ → ℕ → F128) (dim split : ℕ) (st : LState) (t : Transcript) :
    LState × Transcript :=
  let z := vec cf t dim
  let t₁ := afterSq t 2 dim (cf t.k)
  let y := p.oodValues[st.ood]!
  let m := p.messages[st.msg]!
  let t₂ := ap (pN p.claimNonces[st.cbn]!) (ap (pK m) (ap (pF y) t₁))
  let b := cf t₂.k 0
  ({ { { { st with ood := st.ood + 1 } with msg := st.msg + 1 } with cbn := st.cbn + 1 } with
      q := st.q.addScaled (Poly.ofMsg m (F256.ofF y)) b, s := st.s + (F256.ofF y).smul b,
      oods := st.oods.push ⟨z, b, some split⟩ },
    afterSq t₂ 1 1 (cf t₂.k))

theorem Runs.oodIntro (hs : Hashes V) (p : LigeritoProof) (cf : ℕ → ℕ → F128) {dim : ℕ} (hdim : dim ≠ 0)
    (split : ℕ) (st : LState) (t : Transcript) (ho : st.ood < p.oodValues.size) (hm : st.msg < p.messages.size)
    (hb : st.cbn < p.claimNonces.size) :
    Runs F k₀ (oodIntroWith hs p dim split) st t () (ood p cf dim split st t).1 (ood p cf dim split st t).2 := by
  unfold oodIntroWith
  simp only [Flock.With.liftLM_V]
  exact (Runs.sampleVec hs cf hdim st t).bind ((Runs.nextOod p _ _ (by exact ho)).bind ((Runs.absorbF _ _ _).bind
    ((Runs.nextMsg p _ _ (by exact hm)).bind ((Runs.claimPow p _ _ (by exact hb)).bind ((Runs.sample hs cf _ _).bind
      (Runs.modify _ _ _))))))

/-! ## A level's query -/

/-- **A level's query** from `(st, t)` on the opened `rows`, folding with `fold`: the query and consistency nonces
read, the query coins (their positions) and the consistency coins `α`, and the consistency value `e`. -/
def query (p : LigeritoProof) (cf : ℕ → ℕ → F128) (lv : Flock.Level) (rows : Array (Array F128))
    (fold : Array F256) (st : LState) (t : Transcript) : (Array ℕ × Array F128 × F256) × LState × Transcript :=
  let t₁ := ap (pN p.queryNonces[st.qn]!) t
  let coins := vec cf t₁ lv.q
  let t₂ := afterSq t₁ 2 lv.q (cf t₁.k)
  let t₃ := ap (pN p.consistencyNonces[st.cn]!) t₂
  let al := vec cf t₃ (Nat.log2 (lv.q - 1) + 1)
  ((positions lv coins, al, FlockSoundness.Refine.consE rows lv.q al fold),
    { st with qn := st.qn + 1, cn := st.cn + 1 }, afterSq t₃ 2 (Nat.log2 (lv.q - 1) + 1) (cf t₃.k))

/-- The query coins of a level's query from `t`. -/
def queryCoins (p : LigeritoProof) (cf : ℕ → ℕ → F128) (lv : Flock.Level) (st : LState) (t : Transcript) :
    Array F128 :=
  vec cf (ap (pN p.queryNonces[st.qn]!) t) lv.q

theorem Runs.queryLevel (hs : Hashes V) (ms : MerkleScheme) (p : LigeritoProof) (cf : ℕ → ℕ → F128)
    (lv : Flock.Level) (cap : Array ByteArray) (rows : Array (Array F128)) (sibs salts : Array ByteArray)
    (fold : Array F256) (st : LState) (t : Transcript) (hq : lv.q ≠ 0) (hqn : st.qn < p.queryNonces.size)
    (hcn : st.cn < p.consistencyNonces.size)
    (hmc : merkleCheckWith (M := V) (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (queryCoins p cf lv st t)) rows sibs salts
      (2 ^ lv.lanes) = .ok ()) :
    Runs F k₀ (queryLevelWith hs ms p lv cap rows sibs salts fold) st t
      (query p cf lv rows fold st t).1 (query p cf lv rows fold st t).2.1 (query p cf lv rows fold st t).2.2 := by
  unfold queryLevelWith
  simp only [Flock.With.liftLM_V]
  refine (Runs.get st t).bind ((Runs.nonceFrom _ _ _ st t hqn).bind ((Runs.nonceFrom _ _ _ st t (by exact hcn)).bind
    ((Runs.set _ st t).bind ((Runs.pow _ _ t).bind ((Runs.sampleVec hs cf hq _ _).bind ((Runs.pow _ _ _).bind
    ((Runs.sampleVec hs cf (Nat.succ_ne_zero _) _ _).bind ?_)))))))
  have hmc' : merkleCheckWith (M := V) (ms.opsWith hs) ms cap lv.cap lv.d
      (positions lv (vec cf (ap (pN p.queryNonces[st.qn]!) t) lv.q)) rows sibs salts (2 ^ lv.lanes) = .ok () := hmc
  simp only [hmc', List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind,
    FlockSoundness.Refine.range_size, ← List.range_eq_range']
  exact Runs.pure _ _ _

end Lg

end FlockComplete.Replay
