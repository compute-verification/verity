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

end Lg

end FlockComplete.Replay
