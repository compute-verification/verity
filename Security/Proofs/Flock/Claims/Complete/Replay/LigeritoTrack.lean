import Proofs.Flock.Claims.Complete.Replay.LigeritoProver

/-!
# The honest prover's table and weight, step by step

`HRuns`: `PRuns` (the prover's step ends where the model's does) together with what the step does to the prover's table
and weight (`tw`): from a table and weight with `J` it ends with `J'` of its value, on every run whose lists end up the
proof's first, whose value has `Xb`, from the model's cursors and transcript. The postconditions are forward: each
step's says what it did to the table and weight and what it read, so a chain of steps needs no intermediate predicate.

* **Steps**: binds (`HRuns.bindD`, `HRuns.bind`: the rest's postcondition may name the step's value), a loop
  (`HRuns.loop`, on an invariant of the loop's state and the table and weight), weakening (`HRuns.mono`, and
  `HRuns.preX`, whose precondition may read the value the run returns), and the prover's primitives.
* **Blocks**, against the model's: a challenge (`HRuns.challenge`: the table and weight folded at the model's coin,
  then split when asked, and the message their round polynomial), an out-of-domain intro (`HRuns.oodIntro`: the weight
  gains `β·eq(z, ·)` at the model's point and coin, the value is `mleF`, the message the claim's round polynomial) and a
  level's query (`HRuns.query`: the opening at the model's positions and the consistency weight at its coins).
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace Lg

/-- The prover's table and weight. -/
abbrev TW := Array F256 × Array F256

/-- The prover's table and weight. -/
def tw (s : LigState) : TW := (s.tab, s.w)

/-- **The prover's step `y` carries `J` to `J'`** on its table and weight, on every run from `L + k` bodies sent whose
lists end up `p`'s first and whose value has `Xb`, from the model's cursors `c` and transcript `tm`. -/
def JRuns (p : LigeritoProof) (L : ℕ) {β : Type} (y : LP PS β) (c : Cur) (tm : Transcript) (Xb : β → Prop)
    (J : TW → Prop) (J' : β → TW → Prop) : Prop :=
  ∀ (st : LigState) (t : Transcript) (l : Array Claims.Msg) (b : β) (st' : LigState) (t' : Transcript)
    (l' : Array Claims.Msg), l.size = L + t.k → ((y.run st).run t).go l = (.ok ((b, st'), t'), l') →
    Ext p st' → Xb b → pcur st = c → t = tm → J (tw st) → J' b (tw st')

/-- **The prover's step on the model's, with its table and weight** (`PRuns` and `JRuns`). -/
def HRuns (p : LigeritoProof) (L : ℕ) {β : Type} (y : LP PS β) (c : Cur) (tm : Transcript) (c' : Cur)
    (tm' : Transcript) (Q Xb : β → Prop) (J : TW → Prop) (J' : β → TW → Prop) : Prop :=
  PRuns p L y c tm c' tm' Q Xb ∧ JRuns p L y c tm Xb J J'

variable {p : LigeritoProof} {L : ℕ} {α β σ : Type}

/-! ## Binds, loops, weakening -/

/-- **A bind**, where the rest's value `b` may fix what the step's value `a` needs (`hX`); the rest's postcondition may
name `a`. -/
theorem HRuns.bindD {x : LP PS α} {f : α → LP PS β} {c₁ c₂ c₃ : Cur} {tm₁ tm₂ tm₃ : Transcript}
    {Qa Ya : α → Prop} {Qf Xf : α → β → Prop} {Q Xb : β → Prop} {J₁ : TW → Prop} {J₂ : α → TW → Prop}
    {J₃ : α → β → TW → Prop} (hx : HRuns p L x c₁ tm₁ c₂ tm₂ Qa Ya J₁ J₂)
    (hf : ∀ a, Qa a → HRuns p L (f a) c₂ tm₂ c₃ tm₃ (Qf a) (Xf a) (J₂ a) (J₃ a)) (hQ : ∀ a b, Qa a → Qf a b → Q b)
    (hX : ∀ a b, Qa a → Qf a b → Xb b → Ya a ∧ Xf a b) :
    HRuns p L (x >>= f) c₁ tm₁ c₃ tm₃ Q Xb J₁ fun b y => ∃ a, J₃ a b y := by
  refine ⟨PRuns.bindD hx.1 (fun a ha => (hf a ha).1) hQ hX, ?_⟩
  intro st t l b st' t' l' hl h hE hXb hc ht hJ
  obtain ⟨a, st₁, t₁, l₁, h1, h2⟩ := lp_bind_ok h
  obtain ⟨hQa, -, hl₁, hC₁⟩ := hx.1 st t l a st₁ t₁ l₁ hl h1
  obtain ⟨hQb, hG₂, -, -⟩ := (hf a hQa).1 st₁ t₁ l₁ b st' t' l' hl₁ h2
  obtain ⟨hYa, hXa⟩ := hX a b hQa hQb hXb
  obtain ⟨hc₁, ht₁⟩ := hC₁ (hE.of_grow hG₂) hYa hc ht
  exact ⟨a, (hf a hQa).2 st₁ t₁ l₁ b st' t' l' hl₁ h2 hE hXa hc₁ ht₁
    (hx.2 st t l a st₁ t₁ l₁ hl h1 (hE.of_grow hG₂) hYa hc ht hJ)⟩

/-- **A bind** whose step needs nothing of the rest. -/
theorem HRuns.bind {x : LP PS α} {f : α → LP PS β} {c₁ c₂ c₃ : Cur} {tm₁ tm₂ tm₃ : Transcript} {Qa : α → Prop}
    {Q Xb : β → Prop} {J₁ : TW → Prop} {J₂ : α → TW → Prop} {J₃ : α → β → TW → Prop}
    (hx : HRuns p L x c₁ tm₁ c₂ tm₂ Qa (fun _ => True) J₁ J₂)
    (hf : ∀ a, Qa a → HRuns p L (f a) c₂ tm₂ c₃ tm₃ Q Xb (J₂ a) (J₃ a)) :
    HRuns p L (x >>= f) c₁ tm₁ c₃ tm₃ Q Xb J₁ fun b y => ∃ a, J₃ a b y :=
  HRuns.bindD (Qf := fun _ => Q) (Xf := fun _ => Xb) hx hf (fun _ _ _ h => h) fun _ _ _ _ h => ⟨trivial, h⟩

theorem HRuns.pure {b : β} {Q Xb : β → Prop} (hb : Q b) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (Pure.pure b : LP PS β) c tm c tm Q Xb J fun b' y => b' = b ∧ J y := by
  refine ⟨PRuns.pure hb c tm, ?_⟩
  intro st t l b' st' t' l' hl h _ _ _ _ hJ
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok ((b, st), t), l) = _ from h)
  exact ⟨rfl, hJ⟩

theorem HRuns.mono {y : LP PS β} {c c' : Cur} {tm tm' : Transcript} {Q Q' Xb Xb' : β → Prop} {J J₀ : TW → Prop}
    {J' J'' : β → TW → Prop} (h : HRuns p L y c tm c' tm' Q Xb J J') (hQ : ∀ b, Q b → Q' b)
    (hX : ∀ b, Q b → Xb' b → Xb b) (hJ : ∀ x, J₀ x → J x) (hJ' : ∀ b x, Q b → Xb' b → J' b x → J'' b x) :
    HRuns p L y c tm c' tm' Q' Xb' J₀ J'' := by
  refine ⟨h.1.mono hQ hX, ?_⟩
  intro st t l b st' t' l' hl e hE hx hc ht hJ₀
  obtain ⟨hq, -, -, -⟩ := h.1 st t l b st' t' l' hl e
  exact hJ' b _ hq hx (h.2 st t l b st' t' l' hl e hE (hX b hq hx) hc ht (hJ _ hJ₀))

/-- Weakening the postcondition, which may read the value's `Q` and `Xb`. -/
theorem HRuns.post {y : LP PS β} {c c' : Cur} {tm tm' : Transcript} {Q Xb : β → Prop} {J : TW → Prop}
    {J' J'' : β → TW → Prop} (h : HRuns p L y c tm c' tm' Q Xb J J')
    (hJ' : ∀ b x, Q b → Xb b → J' b x → J'' b x) : HRuns p L y c tm c' tm' Q Xb J J'' :=
  h.mono (fun _ h => h) (fun _ _ h => h) (fun _ h => h) hJ'

/-- **Strengthening the precondition**, which may read the `Q` and `Xb` of the value the run returns. -/
theorem HRuns.preX {y : LP PS β} {c c' : Cur} {tm tm' : Transcript} {Q Xb : β → Prop} {J J₀ : TW → Prop}
    {J' : β → TW → Prop} (h : HRuns p L y c tm c' tm' Q Xb J J') (hJ : ∀ b, Q b → Xb b → ∀ x, J₀ x → J x) :
    HRuns p L y c tm c' tm' Q Xb J₀ J' := by
  refine ⟨h.1, ?_⟩
  intro st t l b st' t' l' hl e hE hx hc ht hJ₀
  obtain ⟨hq, -, -, -⟩ := h.1 st t l b st' t' l' hl e
  exact h.2 st t l b st' t' l' hl e hE hx hc ht (hJ b hq hx _ hJ₀)

theorem HRuns.cast {y : LP PS β} {c c₁ c₂ : Cur} {tm tm₁ tm₂ : Transcript} {Q Xb : β → Prop} {J : TW → Prop}
    {J' : β → TW → Prop} (h : HRuns p L y c tm c₁ tm₁ Q Xb J J') (hc : c₁ = c₂) (ht : tm₁ = tm₂) :
    HRuns p L y c tm c₂ tm₂ Q Xb J J' := by
  subst hc ht
  exact h

theorem HRuns.ite {cond : Prop} [Decidable cond] {y z : LP PS β} {c c' : Cur} {tm tm' : Transcript} {Q Xb : β → Prop}
    {J : TW → Prop} {J' : β → TW → Prop} (hy : cond → HRuns p L y c tm c' tm' Q Xb J J')
    (hz : ¬cond → HRuns p L z c tm c' tm' Q Xb J J') : HRuns p L (if cond then y else z) c tm c' tm' Q Xb J J' := by
  by_cases h : cond
  · rw [ite_eq_left h]; exact hy h
  · rw [ite_eq_right h]; exact hz h

/-- **A loop over `[0:n]`** (`PRuns.loop`), with an invariant `JS j s` of the loop's state and the table and weight. -/
theorem HRuns.loop (body : ℕ → σ → LP PS (ForInStep σ)) (C : ℕ → Cur) (Tm : ℕ → Transcript) (I : ℕ → σ → Prop)
    (G : σ → σ → Prop) (hGr : ∀ s, G s s) (hGt : ∀ a b c, G a b → G b c → G a c) (Xs : ℕ → σ → Prop) (Xf : σ → Prop)
    (JS : ℕ → σ → TW → Prop) (n : ℕ)
    (hbody : ∀ j < n, ∀ s, I j s → HRuns p L (body j s) (C j) (Tm j) (C (j + 1)) (Tm (j + 1))
      (fun r => ∃ s', r = .yield s' ∧ I (j + 1) s' ∧ G s s') (fun r => ∀ s', r = .yield s' → Xs j s') (JS j s)
      fun r y => ∀ s', r = .yield s' → JS (j + 1) s' y)
    (hX : ∀ j < n, ∀ s s', I (j + 1) s → (j + 1 = n → s' = s) → G s s' → Xf s' → Xs j s) (s₀ : σ) (h₀ : I 0 s₀) :
    HRuns p L (forIn [:n] s₀ body) (C 0) (Tm 0) (C n) (Tm n) (fun s => I n s ∧ G s₀ s) Xf (JS 0 s₀) (JS n) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices h : ∀ k a s, a + k = n → I a s → HRuns p L (forIn (List.range' a k) s body) (C a) (Tm a) (C n) (Tm n)
      (fun s' => I n s' ∧ G s s' ∧ (k = 0 → s' = s)) Xf (JS a s) (JS n) from
    (h n 0 s₀ (by omega) h₀).mono (fun _ h => ⟨h.1, h.2.1⟩) (fun _ _ h => h) (fun _ h => h) fun _ _ _ _ h => h
  intro k
  induction k with
  | zero =>
    intro a s ha hs
    subst ha
    refine (HRuns.pure ⟨hs, hGr s, fun _ => rfl⟩ _ _ _).post fun b y _ _ h => ?_
    obtain ⟨rfl, h⟩ := h
    exact h
  | succ k ih =>
    intro a s ha hs
    rw [List.range'_succ, List.forIn_cons]
    refine (HRuns.bindD (hbody a (by omega) s hs) (fun r hr => ?_)
      (Qf := fun r s' => ∃ s₁, r = .yield s₁ ∧ I n s' ∧ G s₁ s' ∧ (k = 0 → s' = s₁)) (Xf := fun _ => Xf)
      (J₃ := fun _ => JS n) ?_ ?_).post fun _ _ _ _ ⟨_, h⟩ => h
    · obtain ⟨s₁, rfl, hI, -⟩ := hr
      exact ((ih (a + 1) s₁ (by omega) hI).mono (fun s' h => ⟨s₁, rfl, h⟩) (fun _ _ h => h)
        (fun x h => h s₁ rfl) fun _ _ _ _ h => h)
    · rintro r s' ⟨s₁, rfl, -, hG₁⟩ ⟨s₂, h₂, hI, hG₂, -⟩
      cases h₂
      exact ⟨hI, hGt _ _ _ hG₁ hG₂, fun h => absurd h (Nat.succ_ne_zero k)⟩
    · rintro r s' ⟨s₁, rfl, hI₁, -⟩ ⟨s₂, h₂, -, hG₂, hk⟩ hx
      obtain rfl : s₁ = s₂ := ForInStep.yield.inj h₂
      refine ⟨fun s₃ h₃ => ?_, hx⟩
      obtain rfl : s₁ = s₃ := ForInStep.yield.inj h₃
      exact hX a (by omega) s₁ s' hI₁ (fun h => hk (by omega)) hG₂ hx

/-! ## The prover's steps -/

theorem HRuns.absorb (x : TM Unit) (g : ByteArray → ByteArray) (hx : ∀ t : Transcript, x t = .ok ((), ap g t))
    (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (liftT (liftTM x)) c tm c (ap g tm) (fun _ => True) (fun _ => True) J fun _ => J := by
  refine ⟨PRuns.absorb x g hx c tm, ?_⟩
  intro st t l b st' t' l' hl h _ _ _ _ hJ
  rw [liftT_go (liftTM_ok (hx t) l) st] at h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj h
  exact hJ

theorem HRuns.absorbF (v : F128) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (liftT (liftTM (absorbF v))) c tm c (ap (pF v) tm) (fun _ => True) (fun _ => True) J fun _ => J :=
  HRuns.absorb _ _ (fun t => Flock.RecAlgebra.absorbF_eq v t) c tm J

theorem HRuns.absorbCap (cap capM : Array ByteArray) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (liftT (liftTM (absorbBytes (flat cap)))) c tm c (ap (pBytes (capBytes capM)) tm) (fun _ => True)
      (fun _ => cap = capM) J fun _ => J :=
  ⟨PRuns.absorbCap cap capM c tm, fun st t l b st' t' l' hl h hE _ hc ht hJ =>
    (HRuns.absorb (p := p) (L := L) _ (pBytes (flat cap)) (fun t => Flock.RecAlgebra.absorbBytes_eq (flat cap) t)
      c tm J).2 st t l b st' t' l' hl h hE trivial hc ht hJ⟩

theorem HRuns.absorbAll (ws wsM : Array F128) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (ForIn.forIn ws PUnit.unit fun w (_ : PUnit) => do
        liftT (liftTM (Transcript.absorbF w))
        Pure.pure (ForInStep.yield PUnit.unit))
      c tm c (ap (fun P => wsM.foldl (fun P w => pF w P) P) tm) (fun _ => True) (fun _ => ws = wsM) J fun _ => J := by
  refine ⟨PRuns.absorbAll ws wsM c tm, ?_⟩
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
  intro st t l b st' t' l' hl h _ _ _ _ hJ
  rw [← Array.forIn_toList, key] at h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj h
  exact hJ

/-- A state update: the table and weight end as some state's with `J`, updated. -/
theorem HRuns.modify (g : LigState → LigState)
    (hg : ∀ s, (g s).msgs = s.msgs ∧ (g s).oods = s.oods ∧ (g s).cbn = s.cbn ∧ (g s).qn = s.qn ∧ (g s).cn = s.cn)
    (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (_root_.modify g : LP PS PUnit) c tm c tm (fun _ => True) (fun _ => True) J
      fun _ y => ∃ s, J (tw s) ∧ y = tw (g s) := by
  refine ⟨PRuns.modify g hg c tm, ?_⟩
  intro st t l b st' t' l' hl h _ _ _ _ hJ
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), g st), t), l) = _ from h)
  exact ⟨st, hJ, rfl⟩

theorem HRuns.get (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (MonadState.get : LP PS LigState) c tm c tm (fun _ => True) (fun _ => True) J
      fun s y => J y ∧ tw s = y := by
  refine ⟨PRuns.get c tm, ?_⟩
  intro st t l b st' t' l' hl h _ _ _ _ hJ
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok ((st, st), t), l) = _ from h)
  exact ⟨hJ, rfl⟩

/-- **A sumcheck message**: the model reads it at its cursor. -/
theorem HRuns.sendMsg (m : F256 × F256) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (FlockComplete.sendMsg m) c tm ⟨c.msg + 1, c.ood, c.cbn, c.qn, c.cn⟩ (ap (pK p.messages[c.msg]!) tm)
      (fun _ => True) (fun _ => True) J fun _ y => J y ∧ p.messages[c.msg]! = m := by
  refine ⟨PRuns.sendMsg m c tm, ?_⟩
  intro st t l b st' t' l' hl h hE _ hc _ hJ
  obtain ⟨-, rfl, rfl, rfl⟩ :=
    ok_inj (show (Except.ok (((), { st with msgs := st.msgs.push m }), ap (pK m) t), l) = _ from h)
  subst hc
  exact ⟨hJ, hE.1.get⟩

/-- **An out-of-domain value** pushed and absorbed: the model reads it at its cursor. -/
theorem HRuns.oodPush (y : F128) {k : PUnit → LP PS β} {c c' : Cur} {tm tm' : Transcript} {Q Xb : β → Prop}
    {J : TW → Prop} {J' : β → TW → Prop}
    (h : HRuns p L (k ()) ⟨c.msg, c.ood + 1, c.cbn, c.qn, c.cn⟩ (ap (pF p.oodValues[c.ood]!) tm) c' tm' Q Xb
      (fun x => J x ∧ p.oodValues[c.ood]! = y) J') :
    HRuns p L (_root_.modify (fun st : LigState => { st with oods := st.oods.push y }) >>= fun _ =>
      liftT (liftTM (Transcript.absorbF y)) >>= k) c tm c' tm' Q Xb J J' := by
  refine ⟨PRuns.oodPush y h.1, ?_⟩
  intro st t l b st' t' l' hl e hE hx hc' ht hJ
  obtain ⟨a, st₁, t₁, l₁, h1, h2⟩ := lp_bind_ok e
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with oods := st.oods.push y }), t), l) = _ from h1)
  obtain ⟨u, st₂, t₂, l₂, h3, h4⟩ := lp_bind_ok h2
  obtain ⟨rfl, rfl, rfl, rfl⟩ :=
    ok_inj (show (Except.ok (((), { st with oods := st.oods.push y }), ap (pF y) t), l) = _ from h3)
  obtain ⟨-, hg, -, -⟩ := h.1 { st with oods := st.oods.push y } (ap (pF y) t) l b st' t' l' hl h4
  subst hc' ht
  have hy : p.oodValues[st.oods.size]! = y := (hE.of_grow hg).2.1.get
  exact h.2 { st with oods := st.oods.push y } (ap (pF y) t) l b st' t' l' hl h4 hE hx (by simp [pcur])
    (by rw [← hy]; rfl) ⟨hJ, hy⟩

theorem HRuns.powQ (nonce : ℕ → ℕ → UInt64) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (powP nonce 0) c tm ⟨c.msg, c.ood, c.cbn, c.qn + 1, c.cn⟩ (ap (pN p.queryNonces[c.qn]!) tm)
      (fun _ => True) (fun _ => True) J fun _ => J := by
  refine ⟨PRuns.powQ nonce c tm, ?_⟩
  intro st t l b st' t' l' hl h _ _ _ _ hJ
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with qn := st.qn.push 0 }),
    ap (pN 0) t), l) = _ from h)
  exact hJ

theorem HRuns.powC (nonce : ℕ → ℕ → UInt64) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (powP nonce 1) c tm ⟨c.msg, c.ood, c.cbn, c.qn, c.cn + 1⟩ (ap (pN p.consistencyNonces[c.cn]!) tm)
      (fun _ => True) (fun _ => True) J fun _ => J := by
  refine ⟨PRuns.powC nonce c tm, ?_⟩
  intro st t l b st' t' l' hl h _ _ _ _ hJ
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with cn := st.cn.push 0 }),
    ap (pN 0) t), l) = _ from h)
  exact hJ

theorem HRuns.powB (nonce : ℕ → ℕ → UInt64) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (powP nonce 2) c tm ⟨c.msg, c.ood, c.cbn + 1, c.qn, c.cn⟩ (ap (pN p.claimNonces[c.cbn]!) tm)
      (fun _ => True) (fun _ => True) J fun _ => J := by
  refine ⟨PRuns.powB nonce c tm, ?_⟩
  intro st t l b st' t' l' hl h _ _ _ _ hJ
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with cbn := st.cbn.push 0 }),
    ap (pN 0) t), l) = _ from h)
  exact hJ

/-- **A vector of coins**: the model's, at its transcript's position. -/
theorem HRuns.sampleVec (resp : ℕ → Claims.Msg) {n : ℕ} (hn : n ≠ 0) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (liftT (sampleVecP (asksS resp) n)) c tm c (afterSq tm 2 n (cfOf resp L tm.k)) (fun _ => True)
      (fun _ => True) J fun z y => J y ∧ z = vec (cfOf resp L) tm n := by
  refine ⟨PRuns.sampleVec resp hn c tm, ?_⟩
  intro st t l b st' t' l' hl h _ _ _ ht hJ
  rw [liftT_go (sampleVecP_go resp hn t l) st] at h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h
  subst ht
  refine ⟨hJ, ?_⟩
  rw [hl]
  rfl

theorem HRuns.sample (resp : ℕ → Claims.Msg) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (liftT (sampleP (asksS resp))) c tm c (afterSq tm 1 1 (cfOf resp L tm.k)) (fun _ => True)
      (fun _ => True) J fun v y => J y ∧ v = cfOf resp L tm.k 0 := by
  refine ⟨PRuns.sample resp c tm, ?_⟩
  intro st t l b st' t' l' hl h _ _ _ ht hJ
  rw [liftT_go (sampleP_go resp t l) st] at h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h
  subst ht
  refine ⟨hJ, ?_⟩
  rw [hl]
  rfl

theorem HRuns.sampleK (resp : ℕ → Claims.Msg) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (liftT (sampleKP (asksS resp))) c tm c (afterSq tm 2 2 (cfOf resp L tm.k)) (fun _ => True)
      (fun _ => True) J fun r y => J y ∧ r = ⟨cfOf resp L tm.k 0, cfOf resp L tm.k 1⟩ := by
  refine ⟨PRuns.sampleK resp c tm, ?_⟩
  intro st t l b st' t' l' hl h _ _ _ ht hJ
  rw [liftT_go (sampleKP_go resp t l) st] at h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj h
  subst ht
  refine ⟨hJ, ?_⟩
  rw [hl]
  rfl

/-- **A commitment**: its value is `commitP`'s in the verifier's monad. -/
theorem HRuns.commit (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) (c : Cur) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (liftT (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt))) c tm c tm (fun _ => True)
      (fun _ => True) J fun cm y => J y ∧ commitP (Hashes.ofSha512 h : Hashes V) ms lv g salt = .ok cm := by
  refine ⟨PRuns.commit h ms lv g salt c tm, ?_⟩
  intro st t l b st' t' l' hl e _ _ _ _ hJ
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
  · rename_i a ha
    obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj e
    exact ⟨hJ, ha⟩
  · cases e

/-! ## The model's steps -/

/-- A K-table and its weight folded at `r`. -/
def foldTW (r : F256) (x : TW) : TW := (foldK r x.1, foldK r x.2)

/-- A K-table split into F lanes, and its weight lifted by `u`. -/
def splitTW (x : TW) : TW := (embedF (splitK x.1), liftU x.2)

/-- **A split when asked**, joining the rest `k`: the rest starts from the table and weight split when `split`. -/
theorem HRuns.splitJoin (split : Bool) {k : PUnit → LP PS β} {c c' : Cur} {tm tm' : Transcript} {Q Xb : β → Prop}
    {J : TW → Prop} {J' : β → TW → Prop}
    (hk : HRuns p L (k ()) c tm c' tm' Q Xb (fun y => ∃ x, J x ∧ y = if split then splitTW x else x) J') :
    HRuns p L (if split = true then
        _root_.modify (fun st : LigState => { st with tab := embedF (splitK st.tab), w := liftU st.w }) >>= k
      else k ()) c tm c' tm' Q Xb J J' := by
  refine HRuns.ite (fun hs => ?_) fun hs => ?_
  · refine ((HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bind fun u _ => ?_).post
      fun _ _ _ _ ⟨_, h⟩ => h
    refine hk.mono (fun _ h => h) (fun _ _ h => h) (fun y ⟨s, hJ, hy⟩ => ⟨_, hJ, ?_⟩) fun _ _ _ _ h => h
    rw [hy, ite_eq_left hs]
    rfl
  · refine hk.mono (fun _ h => h) (fun _ _ h => h) (fun y hJ => ⟨_, hJ, ?_⟩) fun _ _ _ _ h => h
    rw [ite_eq_right hs]

/-- **A challenge**, against the model's (`chal`): the table and weight fold at the model's coin, split when asked, and
the message is their round polynomial. -/
theorem HRuns.challenge (resp : ℕ → Claims.Msg) (split : Bool) (sm : LState) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (challengeP (asksS resp) split) (cur sm) tm (cur (chal p (cfOf resp L) sm tm).2.1)
      (chal p (cfOf resp L) sm tm).2.2 (fun _ => True) (fun _ => True) J fun r y =>
        r = (chal p (cfOf resp L) sm tm).1 ∧ (∃ x, J x ∧ y = if split then splitTW (foldTW r x) else foldTW r x) ∧
          p.messages[sm.msg]! = roundPoly y.1 y.2 := by
  unfold challengeP
  apply HRuns.post
  · exact (HRuns.sampleK resp _ _ _).bind fun r _ =>
      (HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bind fun _ _ =>
      HRuns.splitJoin split ((HRuns.get _ _ _).bind fun _ _ =>
      (HRuns.sendMsg _ _ _ _).bind fun _ _ => HRuns.pure trivial _ _ _)
  rintro b y - - ⟨r, -, s, -, rfl, ⟨⟨x, ⟨s₀, ⟨hJ, rfl⟩, rfl⟩, hy⟩, hs⟩, hm⟩
  subst hs
  exact ⟨rfl, ⟨_, hJ, hy⟩, hm⟩

/-- **An out-of-domain intro**, against the model's (`ood`): the weight gains `β·eq(z, ·)` at the model's point and
coin, the value read is `mleF g z`, and the message is the claim's round polynomial. -/
theorem HRuns.oodIntro (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (g : Array F128) {dim : ℕ} (hdim : dim ≠ 0)
    (split : ℕ) (sm : LState) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (oodIntroP (asksS resp) nonce g dim) (cur sm) tm (cur (ood p (cfOf resp L) dim split sm tm).1)
      (ood p (cfOf resp L) dim split sm tm).2 (fun _ => True) (fun _ => True) J fun _ y => ∃ x, J x ∧
        y = (x.1, addScaled x.2 (embedF (eqTable (vec (cfOf resp L) tm dim))) (cfOf resp L (tm.k + 1) 0)) ∧
        p.oodValues[sm.ood]! = mleF g (vec (cfOf resp L) tm dim) ∧
        p.messages[sm.msg]! = roundPoly (embedF g) (embedF (eqTable (vec (cfOf resp L) tm dim))) := by
  unfold oodIntroP
  apply HRuns.post
  · exact (HRuns.sampleVec resp hdim _ _ _).bind fun z _ => HRuns.oodPush (mleF g z)
      ((HRuns.sendMsg _ _ _ _).bind fun _ _ => (HRuns.powB nonce _ _ _).bind fun _ _ =>
      (HRuns.sample resp _ _ _).bind fun _ _ => HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _)
  · rintro b y - - ⟨z, -, -, v, s, ⟨⟨⟨⟨hJ, rfl⟩, ho⟩, hm⟩, rfl⟩, rfl⟩
    exact ⟨_, hJ, rfl, ho, hm⟩

/-- **A level's query**, against the model's (`query`): the opening at the model's positions, and the consistency
weight at its positions and coins. -/
theorem HRuns.query (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit)
    (hq : lv.q ≠ 0) (rows : Array (Array F128)) (fold : Array F256) (sm : LState) (tm : Transcript) (J : TW → Prop) :
    HRuns p L (queryP (asksS resp) nonce ms lv cm) (cur sm) tm (cur (Lg.query p (cfOf resp L) lv rows fold sm tm).2.1)
      (Lg.query p (cfOf resp L) lv rows fold sm tm).2.2 (fun _ => True) (fun _ => True) J fun r y => J y ∧
        r = (openP ms cm lv (Lg.query p (cfOf resp L) lv rows fold sm tm).1.1,
          consW lv.cols (Lg.query p (cfOf resp L) lv rows fold sm tm).1.1
            (Lg.query p (cfOf resp L) lv rows fold sm tm).1.2.1) := by
  unfold queryP
  apply HRuns.post
  · exact (HRuns.powQ nonce _ _ _).bind fun _ _ => (HRuns.sampleVec resp hq _ _ _).bind fun _ _ =>
      (HRuns.powC nonce _ _ _).bind fun _ _ => (HRuns.sampleVec resp (Nat.succ_ne_zero _) _ _ _).bind fun _ _ =>
      HRuns.pure trivial _ _ _
  rintro b y - - ⟨-, coins, -, alpha, rfl, ⟨hJ, rfl⟩, rfl⟩
  exact ⟨hJ, rfl⟩

end Lg

end FlockComplete.Replay
