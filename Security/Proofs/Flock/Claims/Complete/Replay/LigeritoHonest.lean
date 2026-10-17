import Proofs.Flock.Claims.Complete.Replay.LigeritoJoint
import Proofs.Flock.Claims.Complete.Replay.LigeritoProver
import Proofs.Flock.Claims.Complete.Replay.LigeritoFinal

/-!
# The honest Ligerito meets its final identity

`Lg.honest : Lg.Honest`. The honest prover's Ligerito (`ligeritoP`) is walked on its own state (`VRuns`: from a state
with `I` at transcript position `k`, every run ends at position `k'`, only pushes onto its lists, and returns a value
and state with `J`), each step's value and state exact (the squeezes' coins are the replies', `cfOf`). Along the walk
the verifier's model (`Lg.out`), on the proof the prover returns, keeps `Lg.JPre` with the prover's table and weight,
step by step (`JPre.pre0`, `JPre.oodStep`, `JPre.closeStep`, `JPre.chalsStep`), and the last level's `JPre.final` is
the final identity.
-/

namespace FlockComplete.Replay

open Flock FlockSoundness FlockSoundness.Refine Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Lg

/-! ## The relation -/

/-- **The prover's step `y` from `I` at position `k`**: every run from `L + k` bodies sent ends at position `k'` with
`L + k'` sent, only pushes onto the lists, and returns a value and state with `J`. -/
def VRuns (L : ℕ) {β : Type} (y : LP PS β) (k k' : ℕ) (I : LigState → Prop) (J : β → LigState → Prop) : Prop :=
  ∀ (st : LigState) (t : Transcript) (l : Array Claims.Msg) (b : β) (st' : LigState) (t' : Transcript)
    (l' : Array Claims.Msg), l.size = L + t.k → ((y.run st).run t).go l = (.ok ((b, st'), t'), l') → t.k = k →
    I st → l'.size = L + t'.k ∧ t'.k = k' ∧ Grow st st' ∧ J b st'

variable {L : ℕ} {α β σ : Type}

theorem VRuns.bind {x : LP PS α} {f : α → LP PS β} {k k₁ k₂ : ℕ} {I : LigState → Prop} {K : α → LigState → Prop}
    {J : β → LigState → Prop} (hx : VRuns L x k k₁ I K) (hf : ∀ a, VRuns L (f a) k₁ k₂ (K a) J) :
    VRuns L (x >>= f) k k₂ I J := by
  intro st t l b st' t' l' hl h hk hI
  obtain ⟨a, st₁, t₁, l₁, h1, h2⟩ := lp_bind_ok h
  obtain ⟨hl₁, hk₁, hG₁, hK⟩ := hx st t l a st₁ t₁ l₁ hl h1 hk hI
  obtain ⟨hl₂, hk₂, hG₂, hJ⟩ := hf a st₁ t₁ l₁ b st' t' l' hl₁ h2 hk₁ hK
  exact ⟨hl₂, hk₂, hG₁.trans hG₂, hJ⟩

theorem VRuns.pure {b : β} {k : ℕ} {I : LigState → Prop} {J : β → LigState → Prop} (h : ∀ s, I s → J b s) :
    VRuns L (Pure.pure b : LP PS β) k k I J := by
  intro st t l b' st' t' l' hl e hk hI
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok ((b, st), t), l) = _ from e)
  exact ⟨hl, hk, Grow.refl _, h st hI⟩

theorem VRuns.mono {y : LP PS β} {k k' : ℕ} {I I' : LigState → Prop} {J J' : β → LigState → Prop}
    (h : VRuns L y k k' I J) (hI : ∀ s, I' s → I s) (hJ : ∀ b s, J b s → J' b s) : VRuns L y k k' I' J' := by
  intro st t l b st' t' l' hl e hk hI'
  obtain ⟨h1, h2, h3, h4⟩ := h st t l b st' t' l' hl e hk (hI st hI')
  exact ⟨h1, h2, h3, hJ b st' h4⟩

/-- A precondition taken apart: the step from each state it allows. -/
theorem VRuns.intro {y : LP PS β} {k k' : ℕ} {I : LigState → Prop} {J : β → LigState → Prop}
    (h : ∀ S, I S → VRuns L y k k' (· = S) J) : VRuns L y k k' I J := by
  intro st t l b st' t' l' hl e hk hI
  exact h st hI st t l b st' t' l' hl e hk rfl

/-- What the lists' growth keeps: a fact `F` about a state that every state its lists grow into has. -/
theorem VRuns.frame {y : LP PS β} {k k' : ℕ} {I : LigState → Prop} {J : β → LigState → Prop}
    (h : VRuns L y k k' I J) (F : LigState → Prop) (hF : ∀ a b, Grow a b → F a → F b) :
    VRuns L y k k' (fun s => I s ∧ F s) (fun b s => J b s ∧ F s) := by
  intro st t l b st' t' l' hl e hk hI
  obtain ⟨h1, h2, h3, h4⟩ := h st t l b st' t' l' hl e hk hI.1
  exact ⟨h1, h2, h3, h4, hF st st' h3 hI.2⟩

theorem VRuns.ite {cond : Prop} [Decidable cond] {y z : LP PS β} {k k' : ℕ} {I : LigState → Prop}
    {J : β → LigState → Prop} (hy : cond → VRuns L y k k' I J) (hz : ¬cond → VRuns L z k k' I J) :
    VRuns L (if cond then y else z) k k' I J := by
  by_cases h : cond
  · rw [ite_eq_left h]; exact hy h
  · rw [ite_eq_right h]; exact hz h

theorem VRuns.cast {y : LP PS β} {k k₁ k₂ : ℕ} {I : LigState → Prop} {J : β → LigState → Prop}
    (h : VRuns L y k k₁ I J) (hk : k₁ = k₂) : VRuns L y k k₂ I J := hk ▸ h

theorem VRuns.start {y : LP PS β} {k k₀ k' : ℕ} {I : LigState → Prop} {J : β → LigState → Prop}
    (h : VRuns L y k₀ k' I J) (hk : k₀ = k) : VRuns L y k k' I J := hk ▸ h

/-- **A loop over `[0:n]`**: step `j` from position `K j`, on a loop state and prover state with `Inv j`, yields ones
with `Inv (j + 1)` at position `K (j + 1)`. -/
theorem VRuns.loop (body : ℕ → σ → LP PS (ForInStep σ)) (K : ℕ → ℕ) (Inv : ℕ → σ → LigState → Prop) (n : ℕ)
    (hbody : ∀ j < n, ∀ s, VRuns L (body j s) (K j) (K (j + 1)) (Inv j s)
      fun r st => ∃ s', r = .yield s' ∧ Inv (j + 1) s' st) (s₀ : σ) :
    VRuns L (forIn [:n] s₀ body) (K 0) (K n) (Inv 0 s₀) fun s st => Inv n s st := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices h : ∀ m a s, a + m = n → VRuns L (forIn (List.range' a m) s body) (K a) (K n) (Inv a s)
      fun s' st => Inv n s' st from h n 0 s₀ (by omega)
  intro m
  induction m with
  | zero =>
    intro a s ha
    subst ha
    exact VRuns.pure fun _ h => h
  | succ m ih =>
    intro a s ha
    rw [List.range'_succ, List.forIn_cons]
    refine VRuns.bind (hbody a (by omega) s) fun r => ?_
    refine VRuns.intro fun S hS => ?_
    obtain ⟨s', rfl, hI⟩ := hS
    exact (ih (a + 1) s' (by omega)).mono (fun st h => h ▸ hI) fun _ _ h => h

/-! ## The prover's steps, exactly -/

theorem VRuns.modify (g : LigState → LigState)
    (hg : ∀ s, (g s).msgs = s.msgs ∧ (g s).oods = s.oods ∧ (g s).cbn = s.cbn ∧ (g s).qn = s.qn ∧ (g s).cn = s.cn)
    (k : ℕ) (S : LigState) :
    VRuns L (_root_.modify g : LP PS PUnit) k k (· = S) fun _ st => st = g S := by
  intro st t l b st' t' l' hl h hk hS
  subst hS
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), g st), t), l) = _ from h)
  obtain ⟨h1, h2, h3, h4, h5⟩ := hg st
  exact ⟨hl, hk, ⟨Pre.of_eq h1.symm, Pre.of_eq h2.symm, Pre.of_eq h3.symm, Pre.of_eq h4.symm, Pre.of_eq h5.symm⟩, rfl⟩

theorem VRuns.oodPush (y : F128) (k : ℕ) (S : LigState) :
    VRuns L (_root_.modify (fun st : LigState => { st with oods := st.oods.push y })) k k (· = S)
      fun _ st => st = { S with oods := S.oods.push y } := by
  intro st t l b st' t' l' hl h hk hS
  subst hS
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with oods := st.oods.push y }), t), l) = _ from h)
  exact ⟨hl, hk, ⟨Pre.refl _, Pre.push _ _, Pre.refl _, Pre.refl _, Pre.refl _⟩, rfl⟩

theorem VRuns.get (k : ℕ) (S : LigState) :
    VRuns L (MonadState.get : LP PS LigState) k k (· = S) fun b st => b = S ∧ st = S := by
  intro st t l b st' t' l' hl h hk hS
  subst hS
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok ((st, st), t), l) = _ from h)
  exact ⟨hl, hk, Grow.refl _, rfl, rfl⟩

theorem VRuns.sendMsg (m : F256 × F256) (k : ℕ) (S : LigState) :
    VRuns L (FlockComplete.sendMsg m) k k (· = S) fun _ st => st = { S with msgs := S.msgs.push m } := by
  intro st t l b st' t' l' hl h hk hS
  subst hS
  obtain ⟨-, rfl, rfl, rfl⟩ :=
    ok_inj (show (Except.ok (((), { st with msgs := st.msgs.push m }), ap (pK m) t), l) = _ from h)
  exact ⟨hl, hk, ⟨Pre.push _ _, Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _⟩, rfl⟩

theorem VRuns.absorb (x : TM Unit) (g : ByteArray → ByteArray) (hx : ∀ t : Transcript, x t = .ok ((), ap g t))
    (k : ℕ) (S : LigState) : VRuns L (liftT (liftTM x)) k k (· = S) fun _ st => st = S := by
  intro st t l b st' t' l' hl h hk hS
  subst hS
  rw [liftT_go (liftTM_ok (hx t) l) st] at h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj h
  exact ⟨hl, hk, Grow.refl _, rfl⟩

theorem VRuns.absorbF (v : F128) (k : ℕ) (S : LigState) :
    VRuns L (liftT (liftTM (absorbF v))) k k (· = S) fun _ st => st = S :=
  VRuns.absorb _ (pF v) (fun t => Flock.RecAlgebra.absorbF_eq v t) k S

theorem VRuns.absorbBytes (bs : ByteArray) (k : ℕ) (S : LigState) :
    VRuns L (liftT (liftTM (absorbBytes bs))) k k (· = S) fun _ st => st = S :=
  VRuns.absorb _ (pBytes bs) (fun t => Flock.RecAlgebra.absorbBytes_eq bs t) k S

theorem VRuns.absorbAll (ws : Array F128) (k : ℕ) (S : LigState) :
    VRuns L (ForIn.forIn ws PUnit.unit fun w (_ : PUnit) => do
        liftT (liftTM (Transcript.absorbF w))
        Pure.pure (ForInStep.yield PUnit.unit)) k k (· = S) fun _ st => st = S := by
  have key : ∀ (xs : List F128) (st : LigState) (t : Transcript) (l : Array Claims.Msg),
      (((ForIn.forIn xs PUnit.unit fun w (_ : PUnit) => do
        liftT (liftTM (Transcript.absorbF w))
        Pure.pure (ForInStep.yield PUnit.unit) : LP PS PUnit).run st).run t).go l =
        (.ok ((PUnit.unit, st), ap (fun P => xs.foldl (fun P w => pF w P) P) t), l) := by
    intro xs
    induction xs with
    | nil => intro st t l; rfl
    | cons w ws ih =>
      intro st t l
      rw [List.forIn_cons, StateT.run_bind, StateT.run_bind, go_bind]
      exact ih st (ap (pF w) t) l
  intro st t l b st' t' l' hl h hk hS
  subst hS
  rw [← Array.forIn_toList, key] at h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj h
  exact ⟨hl, hk, Grow.refl _, rfl⟩

theorem VRuns.powQ (nonce : ℕ → ℕ → UInt64) (k : ℕ) (S : LigState) :
    VRuns L (powP nonce 0) k k (· = S) fun _ st => st = { S with qn := S.qn.push (nonce 0 S.qn.size) } := by
  intro st t l b st' t' l' hl h hk hS
  subst hS
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with qn := st.qn.push (nonce 0 st.qn.size) }),
    ap (pN (nonce 0 st.qn.size)) t), l) = _ from h)
  exact ⟨hl, hk, ⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.push _ _, Pre.refl _⟩, rfl⟩

theorem VRuns.powC (nonce : ℕ → ℕ → UInt64) (k : ℕ) (S : LigState) :
    VRuns L (powP nonce 1) k k (· = S) fun _ st => st = { S with cn := S.cn.push (nonce 1 S.cn.size) } := by
  intro st t l b st' t' l' hl h hk hS
  subst hS
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with cn := st.cn.push (nonce 1 st.cn.size) }),
    ap (pN (nonce 1 st.cn.size)) t), l) = _ from h)
  exact ⟨hl, hk, ⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _, Pre.push _ _⟩, rfl⟩

theorem VRuns.powB (nonce : ℕ → ℕ → UInt64) (k : ℕ) (S : LigState) :
    VRuns L (powP nonce 2) k k (· = S) fun _ st => st = { S with cbn := S.cbn.push (nonce 2 S.cbn.size) } := by
  intro st t l b st' t' l' hl h hk hS
  subst hS
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with cbn := st.cbn.push (nonce 2 st.cbn.size) }),
    ap (pN (nonce 2 st.cbn.size)) t), l) = _ from h)
  exact ⟨hl, hk, ⟨Pre.refl _, Pre.refl _, Pre.push _ _, Pre.refl _, Pre.refl _⟩, rfl⟩

theorem VRuns.sampleVec (resp : ℕ → Claims.Msg) {n : ℕ} (hn : n ≠ 0) (k : ℕ) (S : LigState) :
    VRuns L (liftT (sampleVecP (asksS resp) n)) k (k + 1) (· = S)
      fun b st => b = (Array.range n).map (cfOf resp L k) ∧ st = S := by
  intro st t l b st' t' l' hl h hk hS
  subst hS hk
  rw [liftT_go (sampleVecP_go resp hn t l) st] at h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h
  refine ⟨by simp [hl, afterSq]; omega, rfl, Grow.refl _, ?_, rfl⟩
  rw [hl]
  rfl

theorem VRuns.sample (resp : ℕ → Claims.Msg) (k : ℕ) (S : LigState) :
    VRuns L (liftT (sampleP (asksS resp))) k (k + 1) (· = S) fun b st => b = cfOf resp L k 0 ∧ st = S := by
  intro st t l b st' t' l' hl h hk hS
  subst hS hk
  rw [liftT_go (sampleP_go resp t l) st] at h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h
  refine ⟨by simp [hl, afterSq]; omega, rfl, Grow.refl _, ?_, rfl⟩
  rw [hl]
  rfl

theorem VRuns.sampleK (resp : ℕ → Claims.Msg) (k : ℕ) (S : LigState) :
    VRuns L (liftT (sampleKP (asksS resp))) k (k + 1) (· = S)
      fun b st => b = ⟨cfOf resp L k 0, cfOf resp L k 1⟩ ∧ st = S := by
  intro st t l b st' t' l' hl h hk hS
  subst hS hk
  rw [liftT_go (sampleKP_go resp t l) st] at h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h
  refine ⟨by simp [hl, afterSq]; omega, rfl, Grow.refl _, ?_, rfl⟩
  rw [hl]
  rfl

theorem VRuns.commit (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) (k : ℕ) (S : LigState) :
    VRuns L (liftT (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt))) k k (· = S)
      fun cm st => cm.rows = encodeRows lv g ∧ st = S := by
  intro st t l b st' t' l' hl e hk hS
  subst hS
  have hc := JsonRT.commitP_pureS h ms lv g salt
  have : (((liftT (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt)) : LP PS Commit).run st).run
      t).go l = (match commitP (Hashes.ofSha512 h : Hashes V) ms lv g salt with
        | .ok a => .ok ((a, st), t)
        | .error x => .error x, l) := by
    show (((commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt >>= fun a => Pure.pure (a, t)) >>= fun q =>
      Pure.pure ((q.1, st), q.2)) : PS _).go l = _
    rw [go_bind, go_bind, hc l]
    cases commitP (Hashes.ofSha512 h : Hashes V) ms lv g salt <;> rfl
  rw [this] at e
  split at e
  · rename_i cm hcm
    obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj e
    exact ⟨hl, hk, Grow.refl _, commitP_rows hcm, rfl⟩
  · cases e

/-- A step whose value and state are exact, then the rest from them. -/
theorem VRuns.seqV {x : LP PS α} {f : α → LP PS β} {k k₁ k₂ : ℕ} {S S₁ : LigState} {A : α}
    {J : β → LigState → Prop} (hx : VRuns L x k k₁ (· = S) fun a st => a = A ∧ st = S₁)
    (hf : VRuns L (f A) k₁ k₂ (· = S₁) J) : VRuns L (x >>= f) k k₂ (· = S) J :=
  hx.bind fun _ => VRuns.intro fun _ h => by
    obtain ⟨rfl, rfl⟩ := h
    exact hf

/-- A unit step whose state is exact, then the rest from it. -/
theorem VRuns.seq {x : LP PS PUnit} {f : PUnit → LP PS β} {k k₁ k₂ : ℕ} {S S₁ : LigState}
    {J : β → LigState → Prop} (hx : VRuns L x k k₁ (· = S) fun _ st => st = S₁)
    (hf : VRuns L (f ()) k₁ k₂ (· = S₁) J) : VRuns L (x >>= f) k k₂ (· = S) J :=
  hx.bind fun _ => VRuns.intro fun _ h => by
    subst h
    exact hf

/-! ## The prover's pieces, exactly -/

/-- The state after a challenge `r`: the table and weight folded at `r`, split when `split`, and their round
polynomial sent. -/
def chalSt (r : F256) (split : Bool) (S : LigState) : LigState :=
  let tab := if split then embedF (splitK (foldK r S.tab)) else foldK r S.tab
  let w := if split then liftU (foldK r S.w) else foldK r S.w
  { S with tab := tab, w := w, msgs := S.msgs.push (roundPoly tab w) }

theorem VRuns.challenge (resp : ℕ → Claims.Msg) (split : Bool) (k : ℕ) (S : LigState) :
    VRuns L (challengeP (asksS resp) split) k (k + 1) (· = S) fun r st =>
      r = ⟨cfOf resp L k 0, cfOf resp L k 1⟩ ∧ st = chalSt ⟨cfOf resp L k 0, cfOf resp L k 1⟩ split S := by
  unfold challengeP
  refine VRuns.seqV (VRuns.sampleK resp k S) ?_
  refine VRuns.seq (VRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _) ?_
  cases split
  · refine VRuns.ite (fun h => absurd h (by simp)) fun _ => ?_
    refine VRuns.seqV (VRuns.get _ _) ?_
    refine VRuns.seq (VRuns.sendMsg _ _ _) ?_
    exact VRuns.pure fun s h => ⟨rfl, h⟩
  · refine VRuns.ite (fun _ => ?_) fun h => absurd rfl h
    refine VRuns.seq (VRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _) ?_
    refine VRuns.seqV (VRuns.get _ _) ?_
    refine VRuns.seq (VRuns.sendMsg _ _ _) ?_
    exact VRuns.pure fun s h => ⟨rfl, h⟩

/-- The state after an out-of-domain intro on `g` at `z` with coin `b`. -/
def oodSt (nonce : ℕ → ℕ → UInt64) (g z : Array F128) (b : F128) (S : LigState) : LigState :=
  { S with
    oods := S.oods.push (mleF g z), msgs := S.msgs.push (roundPoly (embedF g) (embedF (eqTable z))),
    cbn := S.cbn.push (nonce 2 S.cbn.size), w := addScaled S.w (embedF (eqTable z)) b }

theorem VRuns.oodIntro (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (g : Array F128) {dim : ℕ} (hdim : dim ≠ 0)
    (k : ℕ) (S : LigState) :
    VRuns L (oodIntroP (asksS resp) nonce g dim) k (k + 2) (· = S) fun _ st =>
      st = oodSt nonce g ((Array.range dim).map (cfOf resp L k)) (cfOf resp L (k + 1) 0) S := by
  unfold oodIntroP
  refine VRuns.seqV (VRuns.sampleVec resp hdim k S) ?_
  refine VRuns.seq (VRuns.oodPush _ _ _) ?_
  refine VRuns.seq (VRuns.absorbF _ _ _) ?_
  refine VRuns.seq (VRuns.sendMsg _ _ _) ?_
  refine VRuns.seq (VRuns.powB _ _ _) ?_
  refine VRuns.seqV (VRuns.sample resp _ _) ?_
  exact VRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _

/-- A level's query coins' positions and consistency coins, drawn at positions `k` and `k + 1`. -/
def qPosK (cf : ℕ → ℕ → F128) (lv : Level) (k : ℕ) : Array ℕ := Flock.positions lv ((Array.range lv.q).map (cf k))

def qAlphaK (cf : ℕ → ℕ → F128) (lv : Level) (k : ℕ) : Array F128 :=
  (Array.range (Nat.log2 (lv.q - 1) + 1)).map (cf (k + 1))

theorem VRuns.query (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit)
    (hq : lv.q ≠ 0) (k : ℕ) (S : LigState) :
    VRuns L (queryP (asksS resp) nonce ms lv cm) k (k + 2) (· = S) fun r st =>
      r = (openP ms cm lv (qPosK (cfOf resp L) lv k), consW lv.cols (qPosK (cfOf resp L) lv k)
        (qAlphaK (cfOf resp L) lv k)) ∧
      st = { S with qn := S.qn.push (nonce 0 S.qn.size), cn := S.cn.push (nonce 1 S.cn.size) } := by
  unfold queryP
  refine VRuns.seq (VRuns.powQ _ _ _) ?_
  refine VRuns.seqV (VRuns.sampleVec resp hq _ _) ?_
  refine VRuns.seq (VRuns.powC _ _ _) ?_
  refine VRuns.seqV (VRuns.sampleVec resp (Nat.succ_ne_zero _) _ _) ?_
  exact VRuns.pure fun s h => ⟨rfl, h⟩

/-! ## Runs of challenges -/

/-- The coin pairs of `j` challenges from position `k`. -/
def chsK (cf : ℕ → ℕ → F128) (k j : ℕ) : List F256 := (List.range j).map fun i => ⟨cf (k + i) 0, cf (k + i) 1⟩

theorem chsK_succ (cf : ℕ → ℕ → F128) (k j : ℕ) :
    chsK cf k (j + 1) = chsK cf k j ++ [⟨cf (k + j) 0, cf (k + j) 1⟩] := by
  simp [chsK, List.range_succ]

theorem chs_eq (cf : ℕ → ℕ → F128) (t : Transcript) (j : ℕ) : chs cf t j = chsK cf t.k j := rfl

theorem push_get_size {γ : Type} [Inhabited γ] (A : Array γ) (x : γ) : (A.push x)[A.size]! = x :=
  Pre.get (Pre.refl _)

theorem Pre.get_lt {γ : Type} [Inhabited γ] {A B : Array γ} (h : Pre A B) {i : ℕ} (hi : i < A.size) : B[i]! = A[i]! := by
  obtain ⟨C, rfl⟩ := h
  rw [getElem!_pos _ i (by simp; omega), Array.getElem_append_left hi, ← getElem!_pos A i hi]

/-- **After `j` plain challenges** from `S` at position `k`: the table and weight folded at their coins, each fold's
round polynomial sent, the other lists as they were. -/
structure ChalInv (cf : ℕ → ℕ → F128) (k : ℕ) (S : LigState) (j : ℕ) (st : LigState) : Prop where
  tab : st.tab = foldKs (chsK cf k j) S.tab
  w : st.w = foldKs (chsK cf k j) S.w
  oods : st.oods = S.oods
  cbn : st.cbn = S.cbn
  qn : st.qn = S.qn
  cn : st.cn = S.cn
  pre : Pre S.msgs st.msgs
  size : st.msgs.size = S.msgs.size + j
  msg : ∀ i < j, st.msgs[S.msgs.size + i]! =
    roundPoly (foldKs (chsK cf k (i + 1)) S.tab) (foldKs (chsK cf k (i + 1)) S.w)

/-- **After `j` plain challenges and a splitting one** from `S` at position `k`. -/
structure SplitInv (cf : ℕ → ℕ → F128) (k : ℕ) (S : LigState) (j : ℕ) (st : LigState) : Prop where
  tab : st.tab = embedF (splitK (foldKs (chsK cf k (j + 1)) S.tab))
  w : st.w = liftU (foldKs (chsK cf k (j + 1)) S.w)
  oods : st.oods = S.oods
  cbn : st.cbn = S.cbn
  qn : st.qn = S.qn
  cn : st.cn = S.cn
  pre : Pre S.msgs st.msgs
  size : st.msgs.size = S.msgs.size + j + 1
  msg : ∀ i < j, st.msgs[S.msgs.size + i]! =
    roundPoly (foldKs (chsK cf k (i + 1)) S.tab) (foldKs (chsK cf k (i + 1)) S.w)
  last : st.msgs[S.msgs.size + j]! = roundPoly st.tab st.w

theorem ChalInv.zero (cf : ℕ → ℕ → F128) (k : ℕ) (S : LigState) : ChalInv cf k S 0 S :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, Pre.refl _, rfl, fun _ h => absurd h (Nat.not_lt_zero _)⟩

theorem ChalInv.step {cf : ℕ → ℕ → F128} {k : ℕ} {S : LigState} {j : ℕ} {st : LigState} (h : ChalInv cf k S j st) :
    ChalInv cf k S (j + 1) (chalSt ⟨cf (k + j) 0, cf (k + j) 1⟩ false st) where
  tab := by simp only [chalSt, Bool.false_eq_true, ↓reduceIte, h.tab, chsK_succ, foldKs_snoc]
  w := by simp only [chalSt, Bool.false_eq_true, ↓reduceIte, h.w, chsK_succ, foldKs_snoc]
  oods := h.oods
  cbn := h.cbn
  qn := h.qn
  cn := h.cn
  pre := h.pre.trans (Pre.push _ _)
  size := by simp only [chalSt, Array.size_push, h.size]; omega
  msg i hi := by
    simp only [chalSt, Bool.false_eq_true, ↓reduceIte]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
    · rw [push_get_lt _ (by rw [h.size]; omega), h.msg i hi]
    · rw [← h.size, push_get_size, h.tab, h.w, chsK_succ, foldKs_snoc, foldKs_snoc]

theorem ChalInv.split {cf : ℕ → ℕ → F128} {k : ℕ} {S : LigState} {j : ℕ} {st : LigState} (h : ChalInv cf k S j st) :
    SplitInv cf k S j (chalSt ⟨cf (k + j) 0, cf (k + j) 1⟩ true st) where
  tab := by simp only [chalSt, ↓reduceIte, h.tab, chsK_succ, foldKs_snoc]
  w := by simp only [chalSt, ↓reduceIte, h.w, chsK_succ, foldKs_snoc]
  oods := h.oods
  cbn := h.cbn
  qn := h.qn
  cn := h.cn
  pre := h.pre.trans (Pre.push _ _)
  size := by simp only [chalSt, Array.size_push, h.size]
  msg i hi := by
    simp only [chalSt, ↓reduceIte]
    rw [push_get_lt _ (by rw [h.size]; omega), h.msg i hi]
  last := by
    simp only [chalSt, ↓reduceIte]
    rw [← h.size, push_get_size]

/-- **Level 0's challenges**, exactly: `k0` of them, the last splitting. -/
theorem vChals0 (resp : ℕ → Claims.Msg) (k k0 : ℕ) (hk0 : k0 ≠ 0) (S : LigState) :
    VRuns L (forIn [:k0] PUnit.unit fun j (_ : PUnit) => do
        let _ ← challengeP (asksS resp) (j + 1 == k0)
        Pure.pure (ForInStep.yield PUnit.unit)) k (k + k0) (· = S)
      fun _ st => SplitInv (cfOf resp L) k S (k0 - 1) st := by
  have key := VRuns.loop (L := L) (σ := PUnit) (fun j (_ : PUnit) => do
      let _ ← challengeP (asksS resp) (j + 1 == k0)
      Pure.pure (ForInStep.yield PUnit.unit)) (fun j => k + j)
    (fun j _ st => (j < k0 → ChalInv (cfOf resp L) k S j st) ∧ (j = k0 → SplitInv (cfOf resp L) k S (k0 - 1) st)) k0
    (fun j hj _ => VRuns.intro fun S' hS' => by
      refine (VRuns.challenge resp _ _ S').bind fun r => VRuns.pure fun st h => ⟨(), rfl, ?_⟩
      obtain ⟨-, rfl⟩ := h
      by_cases hj1 : j + 1 = k0
      · have hb : (j + 1 == k0) = true := by simp [hj1]
        rw [hb]
        refine ⟨fun h => absurd h (by omega), fun _ => ?_⟩
        rw [show k0 - 1 = j by omega]
        exact (hS'.1 hj).split
      · have hb : (j + 1 == k0) = false := by simp [hj1]
        rw [hb]
        exact ⟨fun _ => (hS'.1 hj).step, fun h => absurd h hj1⟩) PUnit.unit
  refine (key.mono (fun st h => ?_) fun _ st h => h.2 rfl)
  subst h
  exact ⟨fun _ => ChalInv.zero _ _ _, fun h => absurd h.symm hk0⟩

/-- **A level's four challenges**, exactly: the last keeps the folded table `f`, then splits it. -/
theorem vLevelChals (resp : ℕ → Claims.Msg) (k : ℕ) (S : LigState) :
    VRuns L (forIn [:4] (#[] : Array F256) fun i (f : Array F256) =>
        if (i + 1 == 4) = true then do
          let r ← liftT (sampleKP (asksS resp))
          modify fun st : LigState =>
            { tab := foldK r st.tab, w := foldK r st.w, msgs := st.msgs, oods := st.oods, qn := st.qn, cn := st.cn,
              cbn := st.cbn }
          let d ← get
          modify fun st : LigState =>
            { tab := embedF (splitK st.tab), w := liftU st.w, msgs := st.msgs, oods := st.oods, qn := st.qn,
              cn := st.cn, cbn := st.cbn }
          let st ← get
          FlockComplete.sendMsg (roundPoly st.tab st.w)
          pure (ForInStep.yield d.tab)
        else do
          let _ ← challengeP (asksS resp) false
          pure (ForInStep.yield f)) k (k + 4) (· = S)
      fun f st => f = foldKs (chsK (cfOf resp L) k 4) S.tab ∧ SplitInv (cfOf resp L) k S 3 st := by
  refine VRuns.mono (VRuns.loop _ (fun j => k + j)
    (fun j f st => (j < 4 → ChalInv (cfOf resp L) k S j st) ∧
      (j = 4 → f = foldKs (chsK (cfOf resp L) k 4) S.tab ∧ SplitInv (cfOf resp L) k S 3 st)) 4
    (fun j hj f => VRuns.intro fun S' hS' => ?_) #[]) (fun st h => ?_) fun _ st h => h.2 rfl
  · by_cases hj3 : j = 3
    · subst hj3
      refine VRuns.ite (fun _ => ?_) fun h => absurd (by decide) h
      refine VRuns.seqV (VRuns.sampleK resp _ S') ?_
      refine VRuns.seq (VRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _) ?_
      refine VRuns.seqV (VRuns.get _ _) ?_
      refine VRuns.seq (VRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _) ?_
      refine VRuns.seqV (VRuns.get _ _) ?_
      refine VRuns.seq (VRuns.sendMsg _ _ _) ?_
      refine VRuns.pure fun st h => ⟨_, rfl, fun h' => absurd h' (by omega), fun _ => ⟨?_, ?_⟩⟩
      · show foldK _ S'.tab = _
        rw [(hS'.1 (by omega)).tab]
        exact (foldKs_snoc _ _ _).symm.trans (congrArg (foldKs · S.tab) (chsK_succ _ k 3).symm)
      · subst h
        exact (hS'.1 (by omega)).split
    · refine VRuns.ite (fun h => absurd (by simpa using h) (by omega)) fun _ => ?_
      refine (VRuns.challenge resp false _ S').bind fun r => VRuns.pure fun st h => ⟨f, rfl, fun _ => ?_,
        fun h' => absurd h' (by omega)⟩
      obtain ⟨-, rfl⟩ := h
      exact (hS'.1 hj).step
  · subst h
    exact ⟨fun _ => ChalInv.zero _ _ _, fun h => absurd h (by omega)⟩

/-! ## The schedule -/

theorem fast100_k0 {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) : sch.k0 ≠ 0 := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, decide (s.k0 ≠ 0) = true := by
      decide +kernel
    exact of_decide_eq_true (key ⟨m, hm⟩ sch h)
  · rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at h
    cases h

theorem seam_getD {sch : Schedule} (h : ∀ lv ∈ sch.levels, LevelSeam lv) {j : ℕ} (hj : j < sch.levels.size) :
    LevelSeam (sch.levels.getD j default) := by
  unfold Array.getD
  rw [dite_eq_left hj]
  exact h _ (Array.getElem_mem hj)

theorem cols_getD {sch : Schedule} (hD : DimSeam sch) :
    ∀ i < sch.levels.size, (sch.levels.getD i default).cols = sch.n - sch.k0 - 3 * i
  | 0, _ => by have := hD.2.2.2.1; omega
  | i + 1, hi => by
    have h1 := hD.2.2.2.2.2.1 i (by omega)
    have ih := cols_getD hD i (by omega)
    omega

theorem embedF_c0 (g : Array F128) : (embedF g).map F256.c0 = g := by
  rw [embedF, Array.map_map]
  conv_rhs => rw [← Array.map_id g]
  rfl

theorem chsK_length (cf : ℕ → ℕ → F128) (k j : ℕ) : (chsK cf k j).length = j := by simp [chsK]

/-! ## The level loop -/

/-- The prover's level loop's state: the split dimension, the previous commitment, the caps, the levels' openings and
the final proof. -/
abbrev LoopSt := ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof

/-- **The proof agrees with the prover so far**: its lists begin with the prover's, its level-0 opening is `op0`, and
its levels' openings begin with `rp`. -/
def Agree (P : LigeritoProof) (op0 : LevelProof) (rp : Array LevelProof) (S : LigState) : Prop :=
  Ext P S ∧ P.initialProof = op0 ∧ Pre rp P.recursiveProofs

theorem Agree.back {P : LigeritoProof} {op0 : LevelProof} {rp rp' : Array LevelProof} {S S' : LigState}
    (h : Agree P op0 rp' S') (hG : Grow S S') (hr : Pre rp rp') : Agree P op0 rp S :=
  ⟨h.1.of_grow hG, h.2.1, hr.trans h.2.2⟩

/-- **The prover before level step `j`**, against the model's level loop there: the split dimension and the openings
so far, its table the F-table `g` the previous commitment encodes, `JPre` with its table and weight once the proof
agrees with it, and its lists' lengths the model's cursors. -/
structure LvInv (P : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (t : Transcript) (b : Array F128) (op0 : LevelProof) (j : ℕ) (σ : LoopSt) (S : LigState) : Prop where
  sd : σ.1 = sch.n - sch.k0 + 1 - 3 * j
  rp : σ.2.2.2.1.size = j
  tab : ∃ g : Array F128, S.tab = embedF g ∧ g.size = 2 ^ (sch.n - sch.k0 - 3 * j + 1) ∧
    σ.2.1.rows = encodeRows (sch.levels.getD (j + 1) default) g ∧
    (Agree P op0 σ.2.2.2.1 S → JPre (fun w => fOf b[w]!) sch.n ((pre P cf sch T cap t).1.toList.map gOf)
      (levelsOf (loop P cf sch (x₀ P cf sch T cap t) j).1.2.2.1) (loop P cf sch (x₀ P cf sch T cap t) j).2.1 g S.w
      (sch.n - sch.k0 - 3 * j))
  msgs : S.msgs.size = sch.k0 + 4 + 7 * j
  oods : S.oods.size = 3 + 2 * j
  cbn : S.cbn.size = 4 + 3 * j
  qn : S.qn.size = 1 + j
  cn : S.cn.size = 1 + j

theorem pre_push₂ {γ : Type} (A : Array γ) (x y : γ) : Pre A ((A.push x).push y) := (Pre.push _ _).trans (Pre.push _ _)

theorem pre_push₃ {γ : Type} (A : Array γ) (x y z : γ) : Pre A (((A.push x).push y).push z) :=
  (pre_push₂ _ _ _).trans (Pre.push _ _)

/-- **An inner level's step keeps `LvInv`**: from `LvInv` at `j`, the four challenges (`SplitInv`), the next level's
commitment, two out-of-domain intros, the query and the closing message and coin give `LvInv` at `j + 1`. -/
theorem lvStep {P : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {T : F128} {cap : Array ByteArray}
    {t : Transcript} {b : Array F128} {op0 : LevelProof} {ms : MerkleScheme} {j kk dim : ℕ} {σ : LoopSt}
    {S S₄ S' : LigState} {cmN : Commit} {lv : Level} {g' v z₁ z₂ : Array F128}
    (hL : 2 ≤ sch.levels.size) (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) (hD : DimSeam sch)
    (hseam : LevelSeam lv) (hj : j + 1 < sch.levels.size - 1) (hlv : lv = sch.levels.getD (j + 1) default)
    (hkk : kk = t.k + sch.k0 + 9 + 11 * j) (hdim : dim = σ.1 - 4 + 1)
    (h : LvInv P cf sch T cap t b op0 j σ S) (h4 : SplitInv cf kk S 3 S₄)
    (hg' : g' = splitK (foldKs (chsK cf kk 4) S.tab))
    (hcm : cmN.rows = encodeRows (sch.levels.getD (j + 2) default) g')
    (hz₁ : z₁ = (Array.range dim).map (cf (kk + 4))) (hz₂ : z₂ = (Array.range dim).map (cf (kk + 6)))
    (hv : v = consW lv.cols (qPosK cf lv (kk + 8)) (qAlphaK cf lv (kk + 8)))
    (htab : S'.tab = S₄.tab)
    (hw : S'.w = addScaled (addScaled (addScaled S₄.w (embedF (eqTable z₁)) (cf (kk + 5) 0))
      (embedF (eqTable z₂)) (cf (kk + 7) 0)) (liftU (embedF v)) (cf (kk + 10) 0))
    (hmsgs : S'.msgs = ((S₄.msgs.push (roundPoly (embedF g') (embedF (eqTable z₁)))).push
      (roundPoly (embedF g') (embedF (eqTable z₂)))).push (roundPoly (embedF g') (liftU (embedF v))))
    (hoods : S'.oods = (S₄.oods.push (mleF g' z₁)).push (mleF g' z₂))
    (hcbn : ∃ a₁ a₂ a₃, S'.cbn = ((S₄.cbn.push a₁).push a₂).push a₃)
    (hqn : ∃ a, S'.qn = S₄.qn.push a) (hcn : ∃ a, S'.cn = S₄.cn.push a) :
    LvInv P cf sch T cap t b op0 (j + 1)
      (dim, cmN, σ.2.2.1.push cmN.cap, σ.2.2.2.1.push (openP ms σ.2.1 lv (qPosK cf lv (kk + 8))), σ.2.2.2.2) S' := by
  obtain ⟨g, hgt, hgs, hrows, hJ⟩ := h.tab
  obtain ⟨a₁, a₂, a₃, hcbn⟩ := hcbn
  obtain ⟨q₁, hqn⟩ := hqn
  obtain ⟨c₁, hcn⟩ := hcn
  have h3 : 3 * (j + 2) ≤ sch.n - sch.k0 := by omega
  have hext : sch.n - sch.k0 - 3 * j = sch.n - sch.k0 - 3 * (j + 1) + 3 := by omega
  have hdim' : dim = sch.n - sch.k0 - 3 * (j + 1) + 1 := by rw [hdim, h.sd]; omega
  have hcols : lv.cols = sch.n - sch.k0 - 3 * (j + 1) := by rw [hlv, cols_getD hD (j + 1) (by omega)]
  have hlanes : lv.lanes = 4 := by rw [hlv]; exact hD.2.2.2.2.1 (j + 1) (by omega) (by omega)
  have hf : (foldKs (chsK cf kk 4) S.tab).size = 2 ^ (sch.n - sch.k0 - 3 * (j + 1)) := by
    rw [foldKs_size, hgt, embedF_size, hgs, hext, chsK_length, show sch.n - sch.k0 - 3 * (j + 1) + 3 + 1 =
      sch.n - sch.k0 - 3 * (j + 1) + 4 by omega, pow_add, Nat.mul_div_cancel _ (Nat.two_pow_pos _)]
  have hg's : g'.size = 2 ^ (sch.n - sch.k0 - 3 * (j + 1) + 1) := by
    rw [hg', (splitK_spec _).1, hf, pow_succ]; ring
  have h4s := h4.size
  have hG4 : Grow S S₄ :=
    ⟨h4.pre, Pre.of_eq h4.oods.symm, Pre.of_eq h4.cbn.symm, Pre.of_eq h4.qn.symm, Pre.of_eq h4.cn.symm⟩
  have hGm : Pre S₄.msgs S'.msgs := by rw [hmsgs]; exact pre_push₃ _ _ _ _
  have hG' : Grow S₄ S' := ⟨hGm, by rw [hoods]; exact pre_push₂ _ _ _, by rw [hcbn]; exact pre_push₃ _ _ _ _,
    by rw [hqn]; exact Pre.push _ _, by rw [hcn]; exact Pre.push _ _⟩
  refine ⟨by rw [hdim, h.sd]; omega, by simp [h.rp], ⟨g', ?_, hg's, hcm, fun hA => ?_⟩, ?_, ?_, ?_, ?_, ?_⟩
  rotate_left 2
  · rw [hmsgs]; simp only [Array.size_push]; rw [h4s, h.msgs]; omega
  · rw [hoods]; simp only [Array.size_push]; rw [h4.oods, h.oods]; omega
  · rw [hcbn]; simp only [Array.size_push]; rw [h4.cbn, h.cbn]; omega
  · rw [hqn]; simp only [Array.size_push]; rw [h4.qn, h.qn]; omega
  · rw [hcn]; simp only [Array.size_push]; rw [h4.cn, h.cn]; omega
  · rw [htab, h4.tab, hg']
  have hA0 : Agree P op0 σ.2.2.2.1 S := hA.back (hG4.trans hG') (Pre.push _ _)
  have hJ0 := hJ hA0
  rw [hext] at hJ0
  have hI := inv_loop P cf sch T cap t hL hsd j (by omega)
  rw [loop_stepN P cf sch _ hj]
  generalize loop P cf sch (x₀ P cf sch T cap t) j = x at hI hJ0 ⊢
  have hxm : x.2.1.msg = S.msgs.size := by rw [hI.msg, h.msgs]
  have hxk : x.2.2.k = kk := by rw [hI.k, hkk]
  have hP4 : ∀ i < S₄.msgs.size, P.messages[i]! = S₄.msgs[i]! := fun i hi =>
    (hA.1.1.get_lt (by rw [hmsgs]; simp only [Array.size_push]; omega)).trans (hGm.get_lt hi)
  have hc := hJ0.chalsStep (p := P) (cf := cf) (t := x.2.2)
    (fun k hk => by rw [hxm, hP4 _ (by omega), h4.msg k hk, chs_eq, hxk, hgt])
    (by rw [hxm, hP4 _ (by omega), h4.last, h4.tab, h4.w, chs_eq, hxk, hgt])
  rw [← chals_fst P cf x.2.1 x.2.2 4] at hc
  have hx1 : x.1.1 - 4 = sch.n - sch.k0 - 3 * (j + 1) := by rw [hI.sd]; omega
  have hlen : (levelsOf x.1.2.2.1 ++ [(chals P cf #[] x.2.1 x.2.2 4).1.toList.map gOf]).length = j + 1 := by
    simp [levelsOf, hI.lvs]
  have hfst : (chals P cf #[] x.2.1 x.2.2 4).1.toList = chsK cf kk 4 := by rw [chals_fst, chs_eq, hxk]
  have hG : g' = splitK (foldKs (chals P cf #[] x.2.1 x.2.2 4).1.toList (embedF g)) := by rw [hg', hfst, hgt]
  have hck : (chals P cf #[] x.2.1 x.2.2 4).2.2.k = kk + 4 := by simp [hxk]
  have hxo : x.2.1.ood = S₄.oods.size := by rw [hI.ood, h4.oods, h.oods]
  have hco : (chals P cf #[] x.2.1 x.2.2 4).2.1.ood = S₄.oods.size := by simp [hxo]
  have hcm' : (chals P cf #[] x.2.1 x.2.2 4).2.1.msg = S₄.msgs.size := by simp [hxm, h4s]
  have hPo : Pre ((S₄.oods.push (mleF g' z₁)).push (mleF g' z₂)) P.oodValues := hoods ▸ hA.1.2.1
  have hPm : Pre (((S₄.msgs.push (roundPoly (embedF g') (embedF (eqTable z₁)))).push
      (roundPoly (embedF g') (embedF (eqTable z₂)))).push (roundPoly (embedF g') (liftU (embedF v))))
      P.messages := hmsgs ▸ hA.1.1
  unfold stepN
  simp only [hx1]
  generalize ht₁ : ap (pBytes (capBytes P.recursiveCaps[j + 1]!)) (chals P cf #[] x.2.1 x.2.2 4).2.2 = t₁
  have ht₁k : t₁.k = kk + 4 := by rw [← ht₁, ap_k, hck]
  have hv₁ : vec cf t₁ (sch.n - sch.k0 - 3 * (j + 1) + 1) = z₁ := by rw [hz₁, hdim', vec, ht₁k]
  have ho₁ := hc.oodStep (p := P) (cf := cf) (t := t₁) (split := j + 1) hlen.symm
    (by rw [hv₁, hco, ← hG]; exact ((Pre.push _ _).trans hPo).get)
    (by rw [hv₁, hcm', ← hG]; exact ((pre_push₂ _ _ _).trans hPm).get)
  have hx₁o : (ood P cf (sch.n - sch.k0 - 3 * (j + 1) + 1) (j + 1) (chals P cf #[] x.2.1 x.2.2 4).2.1 t₁).1.ood =
      S₄.oods.size + 1 := by simp; omega
  have hx₁m : (ood P cf (sch.n - sch.k0 - 3 * (j + 1) + 1) (j + 1) (chals P cf #[] x.2.1 x.2.2 4).2.1 t₁).1.msg =
      S₄.msgs.size + 1 := by simp; omega
  have hx₁k : (ood P cf (sch.n - sch.k0 - 3 * (j + 1) + 1) (j + 1) (chals P cf #[] x.2.1 x.2.2 4).2.1 t₁).2.k =
      kk + 6 := by simp [ht₁k]
  generalize hx₁ : ood P cf (sch.n - sch.k0 - 3 * (j + 1) + 1) (j + 1) (chals P cf #[] x.2.1 x.2.2 4).2.1 t₁ = x₁
    at ho₁ hx₁o hx₁m hx₁k ⊢
  have hv₂ : vec cf x₁.2 (sch.n - sch.k0 - 3 * (j + 1) + 1) = z₂ := by rw [hz₂, hdim', vec, hx₁k]
  have ho₂ := ho₁.oodStep (p := P) (cf := cf) (t := x₁.2) (split := j + 1) hlen.symm
    (by
      rw [hv₂, hx₁o, ← hG]
      have := hPo.get
      rwa [Array.size_push] at this)
    (by
      rw [hv₂, hx₁m, ← hG]
      have := ((Pre.push _ _).trans hPm).get
      rwa [Array.size_push] at this)
  have hx₂m : (ood P cf (sch.n - sch.k0 - 3 * (j + 1) + 1) (j + 1) x₁.1 x₁.2).1.msg = S₄.msgs.size + 2 := by
    simp [hx₁m]
  have hx₂k : (ood P cf (sch.n - sch.k0 - 3 * (j + 1) + 1) (j + 1) x₁.1 x₁.2).2.k = kk + 8 := by simp [hx₁k]
  generalize hx₂ : ood P cf (sch.n - sch.k0 - 3 * (j + 1) + 1) (j + 1) x₁.1 x₁.2 = x₂ at ho₂ hx₂m hx₂k ⊢
  subst hlv
  have hq : qPos cf (sch.levels.getD (j + 1) default) x₂.2 = qPosK cf (sch.levels.getD (j + 1) default) (kk + 8) := by
    rw [← hx₂k]; rfl
  have hα : qAlpha cf (sch.levels.getD (j + 1) default) x₂.2 =
      qAlphaK cf (sch.levels.getD (j + 1) default) (kk + 8) := by
    rw [← hx₂k]; rfl
  have hop : P.recursiveProofs[j]! = openP ms σ.2.1 (sch.levels.getD (j + 1) default)
      (qPosK cf (sch.levels.getD (j + 1) default) (kk + 8)) := by
    have := hA.2.2.get
    rwa [h.rp] at this
  have hcl := ho₂.closeStep (lv := sch.levels.getD (j + 1) default) (rows := P.recursiveProofs[j]!.rows)
    (cols := sch.n - sch.k0 - 3 * (j + 1)) (start := j + 1) hseam
    (by rw [hlanes]; simp) (by rw [hlanes, hcols, hgs, hext]; ring_nf) hcols rfl hlen.symm
    (by rw [hop, hq]; simp only [openP]; rw [hrows])
    (by
      rw [hx₂m, ← hG, hq, hα, ← hv]
      have := hPm.get
      simp only [Array.size_push] at this
      exact this)
  have hlev : levelsOf (x.1.2.2.1.push (chals P cf #[] x.2.1 x.2.2 4).1) =
      levelsOf x.1.2.2.1 ++ [(chals P cf #[] x.2.1 x.2.2 4).1.toList.map gOf] := by
    simp [levelsOf]
  have hv' : v = consW (sch.n - sch.k0 - 3 * (j + 1)) (qPos cf (sch.levels.getD (j + 1) default) x₂.2)
      (qAlpha cf (sch.levels.getD (j + 1) default) x₂.2) := by rw [hv, hcols, hq, hα]
  rw [hlev, hG, hw, h4.w, ← hv₁, ← hv₂, hv', (hfst.symm : chsK cf kk (3 + 1) = _)]
  rw [ht₁k, hx₁k, hx₂k] at hcl
  exact hcl

/-- **Level 0 starts the level loop**: from the out-of-domain sample and first message (`S₁`), the `k0` challenges
(`SplitInv`), level 1's commitment, two out-of-domain intros, level 0's query and the close, `LvInv` at `0`. -/
theorem lv0Inv {P : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {T : F128} {cm0 : Commit}
    {t : Transcript} {tb b : Array F128} {ms : MerkleScheme} {K dim : ℕ} {S₁ S₂ S' : LigState} {cm1 : Commit}
    {z z₁ z₂ g1 v : Array F128}
    (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) (hD : DimSeam sch) (hk0 : sch.k0 ≠ 0)
    (hseam : LevelSeam (sch.levels.getD 0 default))
    (htb : tb.size = 2 ^ sch.n) (hb : b.size = 2 ^ sch.n) (hT : dot tb b = T)
    (hcm0 : cm0.rows = encodeRows (sch.levels.getD 0 default) tb)
    (hz : z = (Array.range sch.n).map (cf t.k))
    (h1t : S₁.tab = embedF tb) (h1w : S₁.w = addScaled (embedF b) (embedF (eqTable z)) (cf (t.k + 1) 0))
    (h1m : S₁.msgs = #[roundPoly (embedF tb) (addScaled (embedF b) (embedF (eqTable z)) (cf (t.k + 1) 0))])
    (h1o : S₁.oods = #[mleF tb z]) (h1c : S₁.cbn.size = 1) (h1q : S₁.qn = #[]) (h1n : S₁.cn = #[])
    (h2 : SplitInv cf (t.k + 2) S₁ (sch.k0 - 1) S₂)
    (hK : K = t.k + 2 + sch.k0) (hdim : dim = sch.n - sch.k0 + 1)
    (hg1 : g1 = S₂.tab.map F256.c0)
    (hcm : cm1.rows = encodeRows (sch.levels.getD 1 default) g1)
    (hz₁ : z₁ = (Array.range dim).map (cf K)) (hz₂ : z₂ = (Array.range dim).map (cf (K + 2)))
    (hv : v = consW (sch.levels.getD 0 default).cols (qPosK cf (sch.levels.getD 0 default) (K + 4))
      (qAlphaK cf (sch.levels.getD 0 default) (K + 4)))
    (htab : S'.tab = S₂.tab)
    (hw : S'.w = addScaled (addScaled (addScaled S₂.w (embedF (eqTable z₁)) (cf (K + 1) 0))
      (embedF (eqTable z₂)) (cf (K + 3) 0)) (liftU (embedF v)) (cf (K + 6) 0))
    (hmsgs : S'.msgs = ((S₂.msgs.push (roundPoly (embedF g1) (embedF (eqTable z₁)))).push
      (roundPoly (embedF g1) (embedF (eqTable z₂)))).push (roundPoly (embedF g1) (liftU (embedF v))))
    (hoods : S'.oods = (S₂.oods.push (mleF g1 z₁)).push (mleF g1 z₂))
    (hcbn : ∃ a₁ a₂ a₃, S'.cbn = ((S₂.cbn.push a₁).push a₂).push a₃)
    (hqn : ∃ a, S'.qn = S₂.qn.push a) (hcn : ∃ a, S'.cn = S₂.cn.push a) :
    LvInv P cf sch T cm0.cap t b
      (openP ms cm0 (sch.levels.getD 0 default) (qPosK cf (sch.levels.getD 0 default) (K + 4))) 0
      (dim, cm1, #[cm1.cap], #[], default) S' := by
  obtain ⟨a₁, a₂, a₃, hcbn⟩ := hcbn
  obtain ⟨q₁, hqn⟩ := hqn
  obtain ⟨c₁, hcn⟩ := hcn
  have hk0n : sch.k0 ≤ sch.n := by omega
  have hlanes0 : (sch.levels.getD 0 default).lanes = sch.k0 := hD.2.2.1
  have hcols0 : (sch.levels.getD 0 default).cols = sch.n - sch.k0 := hD.2.2.2.1
  have hj1 : sch.k0 - 1 + 1 = sch.k0 := Nat.sub_add_cancel (Nat.pos_of_ne_zero hk0)
  have hX : S₂.tab = embedF (splitK (foldKs (chsK cf (t.k + 2) sch.k0) (embedF tb))) := by rw [h2.tab, hj1, h1t]
  have hg1' : g1 = splitK (foldKs (chsK cf (t.k + 2) sch.k0) (embedF tb)) := by rw [hg1, hX, embedF_c0]
  have hf : (foldKs (chsK cf (t.k + 2) sch.k0) (embedF tb)).size = 2 ^ (sch.n - sch.k0) := by
    rw [foldKs_size, embedF_size, htb, chsK_length, Nat.pow_div hk0n (by norm_num)]
  have hg1s : g1.size = 2 ^ (sch.n - sch.k0 - 3 * 0 + 1) := by
    rw [hg1', (splitK_spec _).1, hf, Nat.mul_zero, Nat.sub_zero, pow_succ]; ring
  have h2s := h2.size
  have h1ms : S₁.msgs.size = 1 := by rw [h1m]; rfl
  have h1os : S₁.oods.size = 1 := by rw [h1o]; rfl
  have hG2 : Grow S₁ S₂ :=
    ⟨h2.pre, Pre.of_eq h2.oods.symm, Pre.of_eq h2.cbn.symm, Pre.of_eq h2.qn.symm, Pre.of_eq h2.cn.symm⟩
  have hGm : Pre S₂.msgs S'.msgs := by rw [hmsgs]; exact pre_push₃ _ _ _ _
  have hG' : Grow S₂ S' := ⟨hGm, by rw [hoods]; exact pre_push₂ _ _ _, by rw [hcbn]; exact pre_push₃ _ _ _ _,
    by rw [hqn]; exact Pre.push _ _, by rw [hcn]; exact Pre.push _ _⟩
  refine ⟨by rw [hdim]; omega, rfl, ⟨g1, by rw [htab, hX, hg1'], hg1s, hcm, fun hA => ?_⟩, ?_, ?_, ?_, ?_, ?_⟩
  rotate_left
  · rw [hmsgs]; simp only [Array.size_push]; rw [h2s, h1ms]; omega
  · rw [hoods]; simp only [Array.size_push]; rw [h2.oods, h1os]
  · rw [hcbn]; simp only [Array.size_push]; rw [h2.cbn, h1c]
  · rw [hqn]; simp only [Array.size_push]; rw [h2.qn, h1q]; rfl
  · rw [hcn]; simp only [Array.size_push]; rw [h2.cn, h1n]; rfl
  have hE2 : Ext P S₂ := hA.1.of_grow hG'
  have hE1 : Ext P S₁ := hE2.of_grow hG2
  have hzv : z = vec cf t sch.n := by rw [hz]; rfl
  rw [hzv] at h1w h1m h1o
  have hP2 : ∀ i < S₂.msgs.size, P.messages[i]! = S₂.msgs[i]! := fun i hi => hE2.1.get_lt hi
  have hhk : (head P cf sch.n (init T) (start cm0.cap T t)).2.k = t.k + 2 := by simp
  have hhm : (head P cf sch.n (init T) (start cm0.cap T t)).1.msg = S₁.msgs.size := by simp [h1ms]
  have hp0 := JPre.pre0 (p := P) (cf := cf) (cap := cm0.cap) (t := t) (k0 := sch.k0) (e := sch.n - sch.k0)
    (j := sch.k0 - 1) (by omega) (by omega) htb hb hT
    (by have := hE1.2.1.get_lt (i := 0) (by rw [h1o]; simp); rw [this, h1o]; rfl)
    (by have := hE1.1.get_lt (i := 0) (by rw [h1m]; simp); rw [this, h1m]; rfl)
    (fun k hk => by rw [hhm, hP2 _ (by omega), h2.msg k hk, chs_eq, hhk, h1t, h1w])
    (by rw [hhm, hP2 _ (by omega), h2.last, h2.tab, h2.w, hj1, chs_eq, hhk, h1t, h1w])
  have hho : (head P cf sch.n (init T) (start cm0.cap T t)).1.ood = S₁.oods.size := by simp [h1os]
  show JPre _ sch.n (List.map gOf (pre P cf sch T cm0.cap t).1.toList) (levelsOf #[])
    (pre P cf sch T cm0.cap t).2.1 g1 S'.w (sch.n - sch.k0 - 3 * 0)
  simp only [pre, lv0, Nat.mul_zero, Nat.sub_zero]
  generalize hh : head P cf sch.n (init T) (start cm0.cap T t) = h₀ at hp0 hhk hhm hho ⊢
  rw [← chals_fst P cf h₀.1 h₀.2 sch.k0] at hp0
  have hfst : (chals P cf #[] h₀.1 h₀.2 sch.k0).1.toList = chsK cf (t.k + 2) sch.k0 := by
    rw [chals_fst, chs_eq, hhk]
  have hG : g1 = splitK (foldKs (chals P cf #[] h₀.1 h₀.2 sch.k0).1.toList (embedF tb)) := by rw [hg1', hfst]
  have hck : (chals P cf #[] h₀.1 h₀.2 sch.k0).2.2.k = K := by simp [hhk, hK]
  have hco : (chals P cf #[] h₀.1 h₀.2 sch.k0).2.1.ood = S₂.oods.size := by simp [hho, h2.oods]
  have hcm' : (chals P cf #[] h₀.1 h₀.2 sch.k0).2.1.msg = S₂.msgs.size := by simp [hhm, h2s]; omega
  have hPo : Pre ((S₂.oods.push (mleF g1 z₁)).push (mleF g1 z₂)) P.oodValues := hoods ▸ hA.1.2.1
  have hPm : Pre (((S₂.msgs.push (roundPoly (embedF g1) (embedF (eqTable z₁)))).push
      (roundPoly (embedF g1) (embedF (eqTable z₂)))).push (roundPoly (embedF g1) (liftU (embedF v))))
      P.messages := hmsgs ▸ hA.1.1
  generalize ht₁ : ap (pBytes (capBytes P.recursiveCaps[0]!)) (chals P cf #[] h₀.1 h₀.2 sch.k0).2.2 = t₁
  have ht₁k : t₁.k = K := by rw [← ht₁, ap_k, hck]
  have hv₁ : vec cf t₁ (sch.n - sch.k0 + 1) = z₁ := by rw [hz₁, hdim, vec, ht₁k]
  have ho₁ := hp0.oodStep (p := P) (cf := cf) (t := t₁) (split := 0) rfl
    (by rw [hv₁, hco, ← hG]; exact ((Pre.push _ _).trans hPo).get)
    (by rw [hv₁, hcm', ← hG]; exact ((pre_push₂ _ _ _).trans hPm).get)
  have hx₁o : (ood P cf (sch.n - sch.k0 + 1) 0 (chals P cf #[] h₀.1 h₀.2 sch.k0).2.1 t₁).1.ood =
      S₂.oods.size + 1 := by simp [hho, h2.oods]
  have hx₁m : (ood P cf (sch.n - sch.k0 + 1) 0 (chals P cf #[] h₀.1 h₀.2 sch.k0).2.1 t₁).1.msg =
      S₂.msgs.size + 1 := by simp; omega
  have hx₁k : (ood P cf (sch.n - sch.k0 + 1) 0 (chals P cf #[] h₀.1 h₀.2 sch.k0).2.1 t₁).2.k = K + 2 := by
    simp [ht₁k]
  generalize hx₁ : ood P cf (sch.n - sch.k0 + 1) 0 (chals P cf #[] h₀.1 h₀.2 sch.k0).2.1 t₁ = x₁
    at ho₁ hx₁o hx₁m hx₁k ⊢
  have hv₂ : vec cf x₁.2 (sch.n - sch.k0 + 1) = z₂ := by rw [hz₂, hdim, vec, hx₁k]
  have ho₂ := ho₁.oodStep (p := P) (cf := cf) (t := x₁.2) (split := 0) rfl
    (by
      rw [hv₂, hx₁o, ← hG]
      have := hPo.get
      rwa [Array.size_push] at this)
    (by
      rw [hv₂, hx₁m, ← hG]
      have := ((Pre.push _ _).trans hPm).get
      rwa [Array.size_push] at this)
  have hx₂m : (ood P cf (sch.n - sch.k0 + 1) 0 x₁.1 x₁.2).1.msg = S₂.msgs.size + 2 := by simp [hx₁m]
  have hx₂k : (ood P cf (sch.n - sch.k0 + 1) 0 x₁.1 x₁.2).2.k = K + 4 := by simp [hx₁k]
  generalize hx₂ : ood P cf (sch.n - sch.k0 + 1) 0 x₁.1 x₁.2 = x₂ at ho₂ hx₂m hx₂k ⊢
  have hq : qPos cf (sch.levels.getD 0 default) x₂.2 = qPosK cf (sch.levels.getD 0 default) (K + 4) := by
    rw [← hx₂k]; rfl
  have hα : qAlpha cf (sch.levels.getD 0 default) x₂.2 = qAlphaK cf (sch.levels.getD 0 default) (K + 4) := by
    rw [← hx₂k]; rfl
  have hcl := ho₂.closeStep (lv := sch.levels.getD 0 default) (rows := P.initialProof.rows)
    (cols := sch.n - sch.k0) (start := 0) hseam (by rw [hlanes0]; simp)
    (by rw [hlanes0, hcols0, htb, Nat.add_sub_cancel' hk0n]) hcols0 rfl rfl
    (by rw [hA.2.1, hq]; simp only [openP]; rw [hcm0])
    (by
      rw [hx₂m, ← hG, hq, hα, ← hv]
      have := hPm.get
      simp only [Array.size_push] at this
      exact this)
  have hv' : v = consW (sch.n - sch.k0) (qPos cf (sch.levels.getD 0 default) x₂.2)
      (qAlpha cf (sch.levels.getD 0 default) x₂.2) := by rw [hv, hcols0, hq, hα]
  rw [hG, hw, h2.w, hj1, h1w, ← hv₁, ← hv₂, hv', (hfst.symm : chsK cf (t.k + 2) sch.k0 = _)]
  rw [ht₁k, hx₁k, hx₂k] at hcl
  exact hcl

/-- **The final identity** at the model's end on the proof `P`: `Lg.Honest`'s conclusion. -/
def Final (P : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (t : Transcript) (basisAt : Array F256 → ℕ → Array F256) : Prop :=
  lhsE P.finalProof.yr (out P cf sch T cap t).1.2.2.2
    (resE basisAt sch.n (pre P cf sch T cap t).1 (out P cf sch T cap t).1.2.2.1 (out P cf sch T cap t).1.2.2.2
      (out P cf sch T cap t).2.1) = (out P cf sch T cap t).2.1.s

/-- **The last level gives the final identity**: from `LvInv` at the last step `j`, its four challenges (`SplitInv`),
`y_r` the split of the folded table and the final rows the opening at the model's positions, once the proof agrees
with the prover and its final proof is `fin`. -/
theorem finStep {P : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {T : F128} {cap : Array ByteArray}
    {t : Transcript} {b : Array F128} {op0 : LevelProof} {ms : MerkleScheme} {j kk : ℕ} {σ : LoopSt}
    {S S₄ S' : LigState} {basisAt : Array F256 → ℕ → Array F256} {fin : FinalProof}
    (hL : 2 ≤ sch.levels.size) (hsd : sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n) (hD : DimSeam sch)
    (hseam : LevelSeam (sch.levels.getD (sch.levels.size - 1) default)) (hj : j + 1 = sch.levels.size - 1)
    (hbasis : ∀ q e y, q.size + e = sch.n → y < 2 ^ e →
      gOf (basisAt q e)[y]! = Model.bHat execArith sch.n (fun w => fOf b[w]!) (q.toList.map gOf ++ Model.bitsK e y))
    (hkk : kk = t.k + sch.k0 + 9 + 11 * j)
    (h : LvInv P cf sch T cap t b op0 j σ S) (h4 : SplitInv cf kk S 3 S₄)
    (hyr : fin.yr = splitK (foldKs (chsK cf kk 4) S.tab))
    (hrows : fin.rows = (openP ms σ.2.1 (sch.levels.getD (sch.levels.size - 1) default)
      (qPosK cf (sch.levels.getD (sch.levels.size - 1) default) (kk + 4))).rows)
    (hG : Grow S₄ S') (hA : Agree P op0 σ.2.2.2.1 S') (hfin : P.finalProof = fin) :
    Final P cf sch T cap t basisAt := by
  obtain ⟨g, hgt, hgs, hrw, hJ⟩ := h.tab
  have hG4 : Grow S S₄ :=
    ⟨h4.pre, Pre.of_eq h4.oods.symm, Pre.of_eq h4.cbn.symm, Pre.of_eq h4.qn.symm, Pre.of_eq h4.cn.symm⟩
  have hJ0 := hJ (hA.back (hG4.trans hG) (Pre.refl _))
  have hext : sch.n - sch.k0 - 3 * j = sch.n - sch.k0 - 3 * (j + 1) + 3 := by omega
  rw [hext] at hJ0
  have hI := inv_loop P cf sch T cap t hL hsd j (by omega)
  have hDn := done_out P cf sch T cap t hL hsd
  have h4s := h4.size
  have hcols : (sch.levels.getD (sch.levels.size - 1) default).cols = sch.n - sch.k0 - 3 * (j + 1) := by
    rw [cols_getD hD _ (by omega), ← hj]
  have hlanes : (sch.levels.getD (sch.levels.size - 1) default).lanes = 4 :=
    hD.2.2.2.2.1 _ (by omega) (by omega)
  rw [hj] at hrw
  unfold Final
  unfold out at hDn ⊢
  rw [← hj, loop_stepL P cf sch _ (by omega)] at hDn ⊢
  generalize loop P cf sch (x₀ P cf sch T cap t) j = x at hI hJ0 hDn ⊢
  have hxm : x.2.1.msg = S.msgs.size := by rw [hI.msg, h.msgs]
  have hxk : x.2.2.k = kk := by rw [hI.k, hkk]
  have hx1 : x.1.1 - 4 = sch.n - sch.k0 - 3 * (j + 1) := by rw [hI.sd]; omega
  have hP4 : ∀ i < S₄.msgs.size, P.messages[i]! = S₄.msgs[i]! := fun i hi =>
    (hA.1.1.get_lt (by have := hG.1; obtain ⟨C, hC⟩ := this; rw [hC, Array.size_append]; omega)).trans
      (hG.1.get_lt hi)
  unfold stepL at hDn ⊢
  simp only [hx1] at hDn ⊢
  generalize htq : ap (fun Q => P.finalProof.yr.foldl (fun Q w => pF w Q) Q) (chals P cf #[] x.2.1 x.2.2 4).2.2 = tq
    at hDn ⊢
  have htqk : tq.k = kk + 4 := by rw [← htq, ap_k]; simp [hxk]
  have hq : qPos cf (sch.levels.getD (sch.levels.size - 1) default) tq =
      qPosK cf (sch.levels.getD (sch.levels.size - 1) default) (kk + 4) := by rw [← htqk]; rfl
  have hl4 : ∀ l ∈ (x.1.2.2.1.push (chals P cf #[] x.2.1 x.2.2 4).1).toList, l.size = 4 := by
    simp only [Array.toList_push, List.mem_append, List.mem_singleton]
    rintro l (hl | rfl)
    · exact hI.lv4 l hl
    · simp
  have horig : (origE (pre P cf sch T cap t).1 (x.1.2.2.1.push (chals P cf #[] x.2.1 x.2.2 4).1)).size +
      (sch.n - sch.k0 - 3 * (j + 1)) = sch.n := by
    rw [origE_size _ _ hl4]
    simp [pre, hI.lvs]
    omega
  refine hJ0.final (cf := cf) (t := x.2.2) (tq := tq) (start := sch.levels.size - 1)
    (fun k hk => by rw [hxm, hP4 _ (by omega), h4.msg k hk, chs_eq, hxk, hgt])
    (by rw [hxm, hP4 _ (by omega), h4.last, h4.tab, h4.w, chs_eq, hxk, hgt]) rfl rfl hseam hlanes
    (by rw [hgs, hlanes, hcols, hext]; ring_nf) hcols
    (by rw [hfin, hrows, hq]; simp only [openP]; rw [hrw])
    (by rw [hfin, hyr, chs_eq, hxk, hgt]) (by rw [hI.lvs]; omega) hl4 horig
    (fun y hy => hbasis _ _ y horig hy) ?_ ?_
  · intro o ho
    have := hDn.oods o ho
    unfold FlockSoundness.Refine.oodFixed OodDim at *
    split at this
    · rename_i hs
      rw [hs, origE_size _ _ hDn.lv4, hDn.lvs]
      simp [pre]
      omega
    · rename_i s hs
      rw [hs, oodFixed_size _ hDn.lv4 (by rw [hDn.lvs]; exact this.1), hDn.lvs]
      omega
  · intro cs hc
    have := hDn.conss cs hc
    unfold ConsDim at this
    rw [consFixed_size _ hDn.lv4, hDn.lvs]
    omega


/-! ## The honest prover's Ligerito -/

/-- **The level loop's invariant**: `LvInv` before each inner or last step, and after the last the final identity once
the proof agrees with the prover and its final proof is the loop's. -/
def LoopInv (P : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (t : Transcript) (b : Array F128) (op0 : LevelProof) (basisAt : Array F256 → ℕ → Array F256) (j : ℕ)
    (σ : LoopSt) (S : LigState) : Prop :=
  if j < sch.levels.size - 1 then LvInv P cf sch T cap t b op0 j σ S
  else Agree P op0 σ.2.2.2.1 S → P.finalProof = σ.2.2.2.2 → Final P cf sch T cap t basisAt

/-- The level loop's positions: eleven per inner level from `k + k0 + 9`, and seven for the last. -/
def loopK (sch : Schedule) (k j : ℕ) : ℕ :=
  if j < sch.levels.size - 1 then k + sch.k0 + 9 + 11 * j else k + sch.k0 + 5 + 11 * (sch.levels.size - 1)

/-- **The honest Ligerito meets its final identity** (`Lg.Honest`). -/
theorem honest : Honest := by
  intro resp h ms m sch T cm0 tb b salt nonce L basisAt t l P t' l' hsch htb hb hT hcm0 hbasis hl hrun
  obtain ⟨hn, hL, hsd, hq, -⟩ := fast100_ligerito hsch
  have hk0 := fast100_k0 hsch
  have hD := (fast100_dims m sch hsch).1
  have hS := fast100_level m sch hsch
  unfold ligeritoP at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l)] at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨⟨caps, rproofs, fin, op0⟩, stF⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  have key := (?prog : VRuns (β := Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof) L _ t.k
    (t.k + sch.k0 + 5 + 11 * (sch.levels.size - 1)) (· = { tab := embedF tb, w := embedF b }) fun r st =>
      Agree P r.2.2.2 r.2.1 st → P.finalProof = r.2.2.1 → Final P (cfOf resp L) sch T cm0.cap t basisAt)
    _ (start cm0.cap T t) l _ _ _ _ hl h1 rfl rfl
  case prog =>
    refine VRuns.seqV (VRuns.sampleVec resp hn _ _) ?_
    refine VRuns.seq (VRuns.oodPush _ _ _) ?_
    refine VRuns.seq (VRuns.absorbF _ _ _) ?_
    refine VRuns.seq (VRuns.powB _ _ _) ?_
    refine VRuns.seqV (VRuns.sample resp _ _) ?_
    refine VRuns.seq (VRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _) ?_
    refine VRuns.seqV (VRuns.get _ _) ?_
    refine VRuns.seq (VRuns.sendMsg _ _ _) ?_
    refine VRuns.bind (vChals0 resp _ sch.k0 hk0 _) fun _ => VRuns.intro fun S₂ hS₂ => ?_
    refine VRuns.seqV (VRuns.get _ _) ?_
    refine VRuns.bind (VRuns.commit h ms _ _ _ _ _) fun cm1 => VRuns.intro fun S₂' hS₂' => ?_
    obtain ⟨hcm1, rfl⟩ := hS₂'
    refine VRuns.seq (VRuns.absorbBytes _ _ _) ?_
    refine VRuns.seq (VRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _ _) ?_
    refine VRuns.seq (VRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _ _) ?_
    refine VRuns.bind (VRuns.query resp nonce ms _ cm0 (hq 0 (by omega)) _ _) fun x => VRuns.intro fun S₃ hS₃ => ?_
    obtain ⟨rfl, rfl⟩ := hS₃
    refine VRuns.seq (VRuns.sendMsg _ _ _) ?_
    refine VRuns.seq (VRuns.powB _ _ _) ?_
    refine VRuns.seqV (VRuns.sample resp _ _) ?_
    refine VRuns.seq (VRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _) ?_
    refine VRuns.bind (K := LoopInv P (cfOf resp L) sch T cm0.cap t b (openP ms cm0 (sch.levels.getD 0 default)
      (qPosK (cfOf resp L) (sch.levels.getD 0 default) (t.k + 1 + 1 + sch.k0 + 2 + 2))) basisAt
      (sch.levels.size - 1)) ?_ fun σ => VRuns.pure fun st hσ => ?_
    · refine VRuns.mono (VRuns.cast (VRuns.start (VRuns.loop _ (loopK sch t.k) _ (sch.levels.size - 1)
        (fun j hj s => ?_) _) (by rw [loopK, ite_eq_left (by omega)]; omega))
        (by rw [loopK, ite_eq_right (lt_irrefl _)])) (fun st hst => ?_) fun _ _ h => h
      · refine VRuns.intro fun S hSI => ?_
        have hLv := hSI
        unfold LoopInv at hLv
        rw [ite_eq_left hj] at hLv
        refine VRuns.start (k₀ := t.k + sch.k0 + 9 + 11 * j) ?_ (by rw [loopK, ite_eq_left hj])
        refine VRuns.bind (vLevelChals resp _ S) fun f => VRuns.intro fun S₄ hS₄ => ?_
        obtain ⟨rfl, h4⟩ := hS₄
        by_cases hj1 : j + 1 < sch.levels.size - 1
        · refine VRuns.ite (fun _ => ?_) fun hne => absurd (by simp; omega) hne
          refine VRuns.cast (k₁ := t.k + sch.k0 + 9 + 11 * j + 4 + 2 + 2 + 2 + 1) ?_
            (by rw [loopK, ite_eq_left hj1]; omega)
          refine VRuns.bind (VRuns.commit h ms _ _ _ _ _) fun cmN => VRuns.intro fun S₅ hS₅ => ?_
          obtain ⟨hcmN, rfl⟩ := hS₅
          refine VRuns.seq (VRuns.absorbBytes _ _ _) ?_
          refine VRuns.seq (VRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _ _) ?_
          refine VRuns.seq (VRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _ _) ?_
          refine VRuns.bind (VRuns.query resp nonce ms _ s.2.1 (hq _ (by omega)) _ _) fun x =>
            VRuns.intro fun S₆ hS₆ => ?_
          obtain ⟨rfl, rfl⟩ := hS₆
          refine VRuns.seq (VRuns.sendMsg _ _ _) ?_
          refine VRuns.seq (VRuns.powB _ _ _) ?_
          refine VRuns.seqV (VRuns.sample resp _ _) ?_
          refine VRuns.seq (VRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _) ?_
          refine VRuns.pure fun st hst => ⟨_, rfl, ?_⟩
          subst hst
          unfold LoopInv
          rw [ite_eq_left hj1]
          refine lvStep hL hsd hD (seam_getD hS (by omega)) hj1 rfl rfl rfl hLv h4 rfl hcmN rfl rfl rfl ?_ ?_ ?_ ?_
            ⟨_, _, _, rfl⟩ ⟨_, rfl⟩ ⟨_, rfl⟩
          all_goals rfl
        · refine VRuns.ite (fun hc => absurd hc (by simp; omega)) fun _ => ?_
          refine VRuns.cast (k₁ := t.k + sch.k0 + 9 + 11 * j + 4 + 2 + 1) ?_
            (by rw [loopK, ite_eq_right hj1]; omega)
          refine VRuns.seq (VRuns.absorbAll _ _ _) ?_
          refine VRuns.bind (VRuns.query resp nonce ms _ s.2.1 (hq _ (by omega)) _ _) fun x =>
            VRuns.intro fun S₆ hS₆ => ?_
          obtain ⟨rfl, rfl⟩ := hS₆
          refine VRuns.seq (VRuns.powB _ _ _) ?_
          refine VRuns.seqV (VRuns.sample resp _ _) ?_
          refine VRuns.pure fun st hst => ⟨_, rfl, ?_⟩
          subst hst
          unfold LoopInv
          rw [ite_eq_right hj1]
          intro hA hfin
          refine finStep (ms := ms) hL hsd hD (seam_getD hS (by omega)) (by omega) hbasis rfl hLv h4 ?_ ?_ ?_ hA hfin
          · rfl
          · rfl
          · exact ⟨Pre.refl _, Pre.refl _, Pre.push _ _, Pre.push _ _, Pre.push _ _⟩
      · subst hst
        unfold LoopInv
        rw [ite_eq_left (by omega)]
        refine lv0Inv (K := t.k + 1 + 1 + sch.k0) hsd hD hk0 (seam_getD hS (by omega)) htb hb hT hcm0 rfl ?_ ?_ ?_
          ?_ ?_ ?_ ?_ hS₂ (by omega) rfl rfl hcm1 rfl rfl rfl ?_ ?_ ?_ ?_ ⟨_, _, _, rfl⟩ ⟨_, rfl⟩ ⟨_, rfl⟩
        all_goals rfl
    · unfold LoopInv at hσ
      rwa [ite_eq_right (lt_irrefl _)] at hσ
  refine key.2.2.2 ?_ ?_ <;> subst hP
  · exact ⟨⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _⟩, rfl, Pre.refl _⟩
  · rfl

end Lg

end FlockComplete.Replay
