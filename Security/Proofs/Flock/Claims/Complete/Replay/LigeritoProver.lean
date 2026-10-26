import Proofs.Flock.Claims.Complete.Replay.LigeritoRun
import Proofs.Flock.Claims.Complete.Json.Session

/-!
# The honest prover's Ligerito, on the model's rounds

`ligeritoP_go`: the honest prover's Ligerito (`ligeritoP`), run in the pure session from a transcript at which `L + k`
bodies have been sent (`k` its position), ends where the verifier's model (`Lg.out`) ends on the proof it returns, at
the coins of the replies (`cfOf resp L`: position `k`'s coin is the reply to send `L + k`), and keeps `L + k` bodies
sent. Its first cap is the commitment's.

* **The relation** (`PRuns p L y c tm c' tm' Q Xb`): every run of the prover's step `y` from `L + k` bodies sent returns a
  value with `Q`, only pushes onto its lists (`Grow`) and keeps `L + k` bodies sent; and once its lists are the first of
  `p`'s (`Ext`) and its value has `Xb`, from the model's cursors `c` (`Cur`; the prover's are its lists' lengths,
  `pcur`) and transcript `tm` it ends at `c'` and `tm'`. The model reads each value the prover sends from `p` at its
  cursor, so the two agree once the prover's lists end up as `p`'s; `Xb` carries what only the rest of the run fixes:
  the recursive caps and `y_r` the proof lists.
* **Steps**: binds (`PRuns.bind`, and `PRuns.bindD` where the rest's value fixes what a step needs), loops
  (`PRuns.loop`), the transcript's steps (`PRuns.absorb`, the squeezes), the state's (`PRuns.modify`, `PRuns.get`), a
  message (`PRuns.sendMsg`), a nonce (`PRuns.powQ`, `PRuns.powC`, `PRuns.powB`), an out-of-domain value
  (`PRuns.oodPush`) and a commitment (`PRuns.commit`); then, against the model's own steps, a challenge
  (`PRuns.challenge`), a level's four challenges (`levelChals`), an out-of-domain intro (`PRuns.oodIntro`) and a level's
  query (`PRuns.query`).
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace Lg

/-! ## Prefixes, cursors and the relation -/

/-- `A` is the first of `B`'s entries. -/
def Pre {α : Type} (A B : Array α) : Prop := ∃ C, B = A ++ C

theorem Pre.refl {α : Type} (A : Array α) : Pre A A := ⟨#[], by simp⟩

theorem Pre.trans {α : Type} {A B C : Array α} (h₁ : Pre A B) (h₂ : Pre B C) : Pre A C := by
  obtain ⟨D₁, rfl⟩ := h₁
  obtain ⟨D₂, rfl⟩ := h₂
  exact ⟨D₁ ++ D₂, by simp⟩

theorem Pre.push {α : Type} (A : Array α) (x : α) : Pre A (A.push x) := ⟨#[x], by simp⟩

theorem Pre.of_eq {α : Type} {A B : Array α} (h : A = B) : Pre A B := h ▸ Pre.refl A

theorem Pre.get {α : Type} [Inhabited α] {A B : Array α} {x : α} (h : Pre (A.push x) B) : B[A.size]! = x := by
  obtain ⟨C, rfl⟩ := h
  rw [getElem!_pos _ _ (by simp; omega), Array.getElem_append_left (by simp)]
  simp

/-- The model's cursors into the proof's lists: messages, out-of-domain values, claim, query and consistency
nonces. -/
structure Cur where
  msg : ℕ
  ood : ℕ
  cbn : ℕ
  qn : ℕ
  cn : ℕ

/-- The verifier's cursors. -/
def cur (s : LState) : Cur := ⟨s.msg, s.ood, s.cbn, s.qn, s.cn⟩

/-- The prover's cursors: its lists' lengths. -/
def pcur (s : LigState) : Cur := ⟨s.msgs.size, s.oods.size, s.cbn.size, s.qn.size, s.cn.size⟩

/-- `b`'s lists begin with `a`'s. -/
def Grow (a b : LigState) : Prop :=
  Pre a.msgs b.msgs ∧ Pre a.oods b.oods ∧ Pre a.cbn b.cbn ∧ Pre a.qn b.qn ∧ Pre a.cn b.cn

theorem Grow.refl (a : LigState) : Grow a a := ⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _⟩

theorem Grow.trans {a b c : LigState} (h₁ : Grow a b) (h₂ : Grow b c) : Grow a c :=
  ⟨h₁.1.trans h₂.1, h₁.2.1.trans h₂.2.1, h₁.2.2.1.trans h₂.2.2.1, h₁.2.2.2.1.trans h₂.2.2.2.1,
    h₁.2.2.2.2.trans h₂.2.2.2.2⟩

/-- The proof `p`'s lists begin with the prover's. -/
def Ext (p : LigeritoProof) (s : LigState) : Prop :=
  Pre s.msgs p.messages ∧ Pre s.oods p.oodValues ∧ Pre s.cbn p.claimNonces ∧ Pre s.qn p.queryNonces ∧
    Pre s.cn p.consistencyNonces

theorem Ext.of_grow {p : LigeritoProof} {a b : LigState} (h : Grow a b) (he : Ext p b) : Ext p a :=
  ⟨h.1.trans he.1, h.2.1.trans he.2.1, h.2.2.1.trans he.2.2.1, h.2.2.2.1.trans he.2.2.2.1,
    h.2.2.2.2.trans he.2.2.2.2⟩

/-- The coins of the replies to the sends from the `L`-th on, by transcript position. -/
def cfOf (resp : ℕ → Claims.Msg) (L : ℕ) : ℕ → ℕ → F128 := fun k => coinOf (resp (L + k))

/-- **The prover's step `y` on the model's**: every run from `L + k` bodies sent (`k` the position) returns a value with
`Q`, grows the lists and keeps `L + k` bodies sent; once the lists end up `p`'s first and the value has `Xb`, from the
cursors `c` and the transcript `tm` it ends at `c'` and `tm'`. -/
def PRuns (p : LigeritoProof) (L : ℕ) {β : Type} (y : LP PS β) (c : Cur) (tm : Transcript) (c' : Cur)
    (tm' : Transcript) (Q Xb : β → Prop) : Prop :=
  ∀ (st : LigState) (t : Transcript) (l : Array Claims.Msg) (b : β) (st' : LigState) (t' : Transcript)
    (l' : Array Claims.Msg), l.size = L + t.k → ((y.run st).run t).go l = (.ok ((b, st'), t'), l') →
    Q b ∧ Grow st st' ∧ l'.size = L + t'.k ∧ (Ext p st' → Xb b → pcur st = c → t = tm → pcur st' = c' ∧ t' = tm')

theorem ok_inj {β : Type} {b b' : β} {st st' : LigState} {t t' : Transcript} {l l' : Array Claims.Msg}
    (h : ((Except.ok ((b, st), t), l) : Except String ((β × LigState) × Transcript) × Array Claims.Msg) =
      (.ok ((b', st'), t'), l')) : b = b' ∧ st = st' ∧ t = t' ∧ l = l' := by
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩

theorem lp_bind_ok {α β : Type} {x : LP PS α} {f : α → LP PS β} {st : LigState} {t : Transcript}
    {l l'' : Array Claims.Msg} {r : (β × LigState) × Transcript}
    (h : (((x >>= f).run st).run t).go l = (.ok r, l'')) :
    ∃ a st₁ t₁ l₁, ((x.run st).run t).go l = (.ok ((a, st₁), t₁), l₁) ∧
      (((f a).run st₁).run t₁).go l₁ = (.ok r, l'') := by
  rw [StateT.run_bind, StateT.run_bind] at h
  obtain ⟨⟨⟨a, st₁⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok h
  exact ⟨a, st₁, t₁, l₁, h1, h2⟩

variable {p : LigeritoProof} {L : ℕ} {α β σ : Type}

/-- **A bind**, where the rest's value `b` may fix what the step's value `a` needs (`hX`). -/
theorem PRuns.bindD {x : LP PS α} {f : α → LP PS β} {c₁ c₂ c₃ : Cur} {tm₁ tm₂ tm₃ : Transcript}
    {Qa Ya : α → Prop} {Qf Xf : α → β → Prop} {Q Xb : β → Prop} (hx : PRuns p L x c₁ tm₁ c₂ tm₂ Qa Ya)
    (hf : ∀ a, Qa a → PRuns p L (f a) c₂ tm₂ c₃ tm₃ (Qf a) (Xf a)) (hQ : ∀ a b, Qa a → Qf a b → Q b)
    (hX : ∀ a b, Qa a → Qf a b → Xb b → Ya a ∧ Xf a b) : PRuns p L (x >>= f) c₁ tm₁ c₃ tm₃ Q Xb := by
  intro st t l b st' t' l' hl h
  obtain ⟨a, st₁, t₁, l₁, h1, h2⟩ := lp_bind_ok h
  obtain ⟨hQa, hG₁, hl₁, hC₁⟩ := hx st t l a st₁ t₁ l₁ hl h1
  obtain ⟨hQb, hG₂, hl₂, hC₂⟩ := hf a hQa st₁ t₁ l₁ b st' t' l' hl₁ h2
  refine ⟨hQ a b hQa hQb, hG₁.trans hG₂, hl₂, fun hE hXb hc ht => ?_⟩
  obtain ⟨hYa, hXa⟩ := hX a b hQa hQb hXb
  obtain ⟨hc₁, ht₁⟩ := hC₁ (hE.of_grow hG₂) hYa hc ht
  exact hC₂ hE hXa hc₁ ht₁

/-- A bind whose step needs `Ya`, given by the rest's value. -/
theorem PRuns.bindY {x : LP PS α} {f : α → LP PS β} {c₁ c₂ c₃ : Cur} {tm₁ tm₂ tm₃ : Transcript}
    {Qa Ya : α → Prop} {Q Xb : β → Prop} (hx : PRuns p L x c₁ tm₁ c₂ tm₂ Qa Ya)
    (hf : ∀ a, Qa a → PRuns p L (f a) c₂ tm₂ c₃ tm₃ Q Xb) (hY : ∀ a b, Qa a → Q b → Xb b → Ya a) :
    PRuns p L (x >>= f) c₁ tm₁ c₃ tm₃ Q Xb :=
  PRuns.bindD (Qf := fun _ => Q) (Xf := fun _ => Xb) hx hf (fun _ _ _ h => h) fun a b ha hb h => ⟨hY a b ha hb h, h⟩

/-- **A bind** whose step needs nothing of the rest. -/
theorem PRuns.bind {x : LP PS α} {f : α → LP PS β} {c₁ c₂ c₃ : Cur} {tm₁ tm₂ tm₃ : Transcript} {Qa : α → Prop}
    {Q Xb : β → Prop} (hx : PRuns p L x c₁ tm₁ c₂ tm₂ Qa fun _ => True)
    (hf : ∀ a, Qa a → PRuns p L (f a) c₂ tm₂ c₃ tm₃ Q Xb) : PRuns p L (x >>= f) c₁ tm₁ c₃ tm₃ Q Xb :=
  PRuns.bindY hx hf fun _ _ _ _ _ => trivial

theorem PRuns.pure {b : β} {Q Xb : β → Prop} (hb : Q b) (c : Cur) (tm : Transcript) :
    PRuns p L (Pure.pure b : LP PS β) c tm c tm Q Xb := by
  intro st t l b' st' t' l' hl h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok ((b, st), t), l) = _ from h)
  exact ⟨hb, Grow.refl _, hl, fun _ _ hc ht => ⟨hc, ht⟩⟩

theorem PRuns.mono {y : LP PS β} {c c' : Cur} {tm tm' : Transcript} {Q Q' Xb Xb' : β → Prop}
    (h : PRuns p L y c tm c' tm' Q Xb) (hQ : ∀ b, Q b → Q' b) (hX : ∀ b, Q b → Xb' b → Xb b) :
    PRuns p L y c tm c' tm' Q' Xb' := by
  intro st t l b st' t' l' hl e
  obtain ⟨hq, hg, hl', hc⟩ := h st t l b st' t' l' hl e
  exact ⟨hQ b hq, hg, hl', fun hE hx => hc hE (hX b hq hx)⟩

theorem PRuns.cast {y : LP PS β} {c c₁ c₂ : Cur} {tm tm₁ tm₂ : Transcript} {Q Xb : β → Prop}
    (h : PRuns p L y c tm c₁ tm₁ Q Xb) (hc : c₁ = c₂) (ht : tm₁ = tm₂) : PRuns p L y c tm c₂ tm₂ Q Xb := by
  subst hc ht
  exact h

theorem PRuns.ite {cond : Prop} [Decidable cond] {y z : LP PS β} {c c' : Cur} {tm tm' : Transcript} {Q Xb : β → Prop}
    (hy : cond → PRuns p L y c tm c' tm' Q Xb) (hz : ¬cond → PRuns p L z c tm c' tm' Q Xb) :
    PRuns p L (if cond then y else z) c tm c' tm' Q Xb := by
  by_cases h : cond
  · rw [ite_eq_left h]; exact hy h
  · rw [ite_eq_right h]; exact hz h

/-! ## The prover's steps -/

theorem liftT_go {y : TMW PS α} {t t' : Transcript} {l l' : Array Claims.Msg} {a : α}
    (h : (y.run t).go l = (.ok (a, t'), l')) (st : LigState) :
    (((liftT y : LP PS α).run st).run t).go l = (.ok ((a, st), t'), l') := by
  show ((y >>= fun a => Pure.pure (a, st)).run t).go l = _
  rw [bind_go h]
  rfl

/-- A transcript action that maps the pending bytes by `g`. -/
theorem PRuns.absorb (x : TM Unit) (g : ByteArray → ByteArray) (hx : ∀ t : Transcript, x t = .ok ((), ap g t))
    (c : Cur) (tm : Transcript) : PRuns p L (liftT (liftTM x)) c tm c (ap g tm) (fun _ => True) fun _ => True := by
  intro st t l b st' t' l' hl h
  rw [liftT_go (liftTM_ok (hx t) l) st] at h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj h
  exact ⟨trivial, Grow.refl _, hl, fun _ _ hc ht => ⟨hc, ht ▸ rfl⟩⟩

theorem PRuns.absorbF (v : F128) (c : Cur) (tm : Transcript) :
    PRuns p L (liftT (liftTM (absorbF v))) c tm c (ap (pF v) tm) (fun _ => True) fun _ => True :=
  PRuns.absorb _ _ (fun t => Flock.RecAlgebra.absorbF_eq v t) c tm

/-- A cap absorbed, read by the model as `capM`: they agree once the rest of the run fixes `cap = capM`. -/
theorem PRuns.absorbCap (cap capM : Array ByteArray) (c : Cur) (tm : Transcript) :
    PRuns p L (liftT (liftTM (absorbBytes (flat cap)))) c tm c (ap (pBytes (capBytes capM)) tm) (fun _ => True)
      fun _ => cap = capM := by
  intro st t l b st' t' l' hl h
  obtain ⟨hq, hg, hl', hc⟩ := PRuns.absorb (p := p) (L := L) (absorbBytes (flat cap)) (pBytes (flat cap))
    (fun t => Flock.RecAlgebra.absorbBytes_eq (flat cap) t) c tm st t l b st' t' l' hl h
  exact ⟨hq, hg, hl', fun hE hx hc' ht => by subst hx; exact hc hE trivial hc' ht⟩

/-- Each of `ws` absorbed in turn, read by the model as `wsM`'s. -/
theorem PRuns.absorbAll (ws wsM : Array F128) (c : Cur) (tm : Transcript) :
    PRuns p L (ForIn.forIn ws PUnit.unit fun w (_ : PUnit) => do
        liftT (liftTM (Transcript.absorbF w))
        Pure.pure (ForInStep.yield PUnit.unit))
      c tm c (ap (fun P => wsM.foldl (fun P w => pF w P) P) tm) (fun _ => True) fun _ => ws = wsM := by
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
  intro st t l b st' t' l' hl h
  rw [← Array.forIn_toList, key] at h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj h
  refine ⟨trivial, Grow.refl _, hl, fun _ hx hc ht => ⟨hc, ?_⟩⟩
  subst hx ht
  simp only [Array.foldl_toList]

theorem PRuns.modify (g : LigState → LigState)
    (hg : ∀ s, (g s).msgs = s.msgs ∧ (g s).oods = s.oods ∧ (g s).cbn = s.cbn ∧ (g s).qn = s.qn ∧ (g s).cn = s.cn)
    (c : Cur) (tm : Transcript) : PRuns p L (_root_.modify g : LP PS PUnit) c tm c tm (fun _ => True) fun _ => True := by
  intro st t l b st' t' l' hl h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), g st), t), l) = _ from h)
  obtain ⟨h1, h2, h3, h4, h5⟩ := hg st
  refine ⟨trivial, ⟨Pre.of_eq h1.symm, Pre.of_eq h2.symm, Pre.of_eq h3.symm, Pre.of_eq h4.symm, Pre.of_eq h5.symm⟩,
    hl, fun _ _ hc ht => ⟨?_, ht⟩⟩
  rw [← hc]
  simp only [pcur, h1, h2, h3, h4, h5]

theorem PRuns.get (c : Cur) (tm : Transcript) :
    PRuns p L (MonadState.get : LP PS LigState) c tm c tm (fun _ => True) fun _ => True := by
  intro st t l b st' t' l' hl h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok ((st, st), t), l) = _ from h)
  exact ⟨trivial, Grow.refl _, hl, fun _ _ hc ht => ⟨hc, ht⟩⟩

/-- **A sumcheck message**, read by the model at its message cursor. -/
theorem PRuns.sendMsg (m : F256 × F256) (c : Cur) (tm : Transcript) :
    PRuns p L (FlockComplete.sendMsg m) c tm ⟨c.msg + 1, c.ood, c.cbn, c.qn, c.cn⟩ (ap (pK p.messages[c.msg]!) tm)
      (fun _ => True) fun _ => True := by
  intro st t l b st' t' l' hl h
  obtain ⟨-, rfl, rfl, rfl⟩ :=
    ok_inj (show (Except.ok (((), { st with msgs := st.msgs.push m }), ap (pK m) t), l) = _ from h)
  refine ⟨trivial, ⟨Pre.push _ _, Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _⟩, hl, fun hE _ hc ht => ?_⟩
  subst hc ht
  refine ⟨by simp [pcur], ?_⟩
  show _ = ap (pK p.messages[st.msgs.size]!) t
  rw [hE.1.get]

/-- **An out-of-domain value** pushed and absorbed, read by the model at its cursor. -/
theorem PRuns.oodPush (y : F128) {k : PUnit → LP PS β} {c c' : Cur} {tm tm' : Transcript} {Q Xb : β → Prop}
    (h : PRuns p L (k ()) ⟨c.msg, c.ood + 1, c.cbn, c.qn, c.cn⟩ (ap (pF p.oodValues[c.ood]!) tm) c' tm' Q Xb) :
    PRuns p L (_root_.modify (fun st : LigState => { st with oods := st.oods.push y }) >>= fun _ =>
      liftT (liftTM (Transcript.absorbF y)) >>= k) c tm c' tm' Q Xb := by
  intro st t l b st' t' l' hl e
  obtain ⟨a, st₁, t₁, l₁, h1, h2⟩ := lp_bind_ok e
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with oods := st.oods.push y }), t), l) = _ from h1)
  obtain ⟨u, st₂, t₂, l₂, h3, h4⟩ := lp_bind_ok h2
  obtain ⟨rfl, rfl, rfl, rfl⟩ :=
    ok_inj (show (Except.ok (((), { st with oods := st.oods.push y }), ap (pF y) t), l) = _ from h3)
  obtain ⟨hq, hg, hl', hc⟩ := h { st with oods := st.oods.push y } (ap (pF y) t) l b st' t' l' hl h4
  refine ⟨hq, Grow.trans (b := { st with oods := st.oods.push y })
    ⟨Pre.refl _, Pre.push _ _, Pre.refl _, Pre.refl _, Pre.refl _⟩ hg, hl', fun hE hx hc' ht => ?_⟩
  subst hc' ht
  have hy : p.oodValues[st.oods.size]! = y := (hE.of_grow hg).2.1.get
  exact hc hE hx (by simp [pcur]) (by rw [← hy]; rfl)

theorem PRuns.powQ (nonce : ℕ → ℕ → UInt64) (c : Cur) (tm : Transcript) :
    PRuns p L (powP nonce 0) c tm ⟨c.msg, c.ood, c.cbn, c.qn + 1, c.cn⟩ (ap (pN p.queryNonces[c.qn]!) tm)
      (fun _ => True) fun _ => True := by
  intro st t l b st' t' l' hl h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with qn := st.qn.push 0 }),
    ap (pN 0) t), l) = _ from h)
  refine ⟨trivial, ⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.push _ _, Pre.refl _⟩, hl, fun hE _ hc ht => ?_⟩
  subst hc ht
  refine ⟨by simp [pcur], ?_⟩
  show _ = ap (pN p.queryNonces[st.qn.size]!) t
  rw [hE.2.2.2.1.get]

theorem PRuns.powC (nonce : ℕ → ℕ → UInt64) (c : Cur) (tm : Transcript) :
    PRuns p L (powP nonce 1) c tm ⟨c.msg, c.ood, c.cbn, c.qn, c.cn + 1⟩ (ap (pN p.consistencyNonces[c.cn]!) tm)
      (fun _ => True) fun _ => True := by
  intro st t l b st' t' l' hl h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with cn := st.cn.push 0 }),
    ap (pN 0) t), l) = _ from h)
  refine ⟨trivial, ⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _, Pre.push _ _⟩, hl, fun hE _ hc ht => ?_⟩
  subst hc ht
  refine ⟨by simp [pcur], ?_⟩
  show _ = ap (pN p.consistencyNonces[st.cn.size]!) t
  rw [hE.2.2.2.2.get]

theorem PRuns.powB (nonce : ℕ → ℕ → UInt64) (c : Cur) (tm : Transcript) :
    PRuns p L (powP nonce 2) c tm ⟨c.msg, c.ood, c.cbn + 1, c.qn, c.cn⟩ (ap (pN p.claimNonces[c.cbn]!) tm)
      (fun _ => True) fun _ => True := by
  intro st t l b st' t' l' hl h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with cbn := st.cbn.push 0 }),
    ap (pN 0) t), l) = _ from h)
  refine ⟨trivial, ⟨Pre.refl _, Pre.refl _, Pre.push _ _, Pre.refl _, Pre.refl _⟩, hl, fun hE _ hc ht => ?_⟩
  subst hc ht
  refine ⟨by simp [pcur], ?_⟩
  show _ = ap (pN p.claimNonces[st.cbn.size]!) t
  rw [hE.2.2.1.get]

theorem PRuns.sampleVec (resp : ℕ → Claims.Msg) {n : ℕ} (hn : n ≠ 0) (c : Cur) (tm : Transcript) :
    PRuns p L (liftT (sampleVecP (asksS resp) n)) c tm c (afterSq tm 2 n (cfOf resp L tm.k)) (fun _ => True)
      fun _ => True := by
  intro st t l b st' t' l' hl h
  rw [liftT_go (sampleVecP_go resp hn t l) st] at h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj h
  refine ⟨trivial, Grow.refl _, by simp [hl, afterSq]; omega, fun _ _ hc ht => ⟨hc, ?_⟩⟩
  subst ht
  rw [hl]
  rfl

theorem PRuns.sample (resp : ℕ → Claims.Msg) (c : Cur) (tm : Transcript) :
    PRuns p L (liftT (sampleP (asksS resp))) c tm c (afterSq tm 1 1 (cfOf resp L tm.k)) (fun _ => True)
      fun _ => True := by
  intro st t l b st' t' l' hl h
  rw [liftT_go (sampleP_go resp t l) st] at h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj h
  refine ⟨trivial, Grow.refl _, by simp [hl, afterSq]; omega, fun _ _ hc ht => ⟨hc, ?_⟩⟩
  subst ht
  rw [hl]
  rfl

theorem PRuns.sampleK (resp : ℕ → Claims.Msg) (c : Cur) (tm : Transcript) :
    PRuns p L (liftT (sampleKP (asksS resp))) c tm c (afterSq tm 2 2 (cfOf resp L tm.k)) (fun _ => True)
      fun _ => True := by
  intro st t l b st' t' l' hl h
  rw [liftT_go (sampleKP_go resp t l) st] at h
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj h
  refine ⟨trivial, Grow.refl _, by simp [hl, afterSq]; omega, fun _ _ hc ht => ⟨hc, ?_⟩⟩
  subst ht
  rw [hl]
  rfl

/-- **A commitment** sends nothing and touches neither the transcript nor the lists. -/
theorem PRuns.commit (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) (c : Cur) (tm : Transcript) :
    PRuns p L (liftT (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt))) c tm c tm (fun _ => True)
      fun _ => True := by
  intro st t l b st' t' l' hl e
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
  · obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj e
    exact ⟨trivial, Grow.refl _, hl, fun _ _ hc ht => ⟨hc, ht⟩⟩
  · cases e

/-- **A loop over `[0:n]`**: step `j`, from a state with `I j`, yields one with `I (j + 1)` that `G` relates to it, from
the model's `j`-th cursors and transcript to the next; it needs `Xs j` of the state it yields, which the last state's `Xf`
gives (`hX`). -/
theorem PRuns.loop (body : ℕ → σ → LP PS (ForInStep σ)) (C : ℕ → Cur) (Tm : ℕ → Transcript) (I : ℕ → σ → Prop)
    (G : σ → σ → Prop) (hGr : ∀ s, G s s) (hGt : ∀ a b c, G a b → G b c → G a c) (Xs : ℕ → σ → Prop) (Xf : σ → Prop)
    (n : ℕ)
    (hbody : ∀ j < n, ∀ s, I j s → PRuns p L (body j s) (C j) (Tm j) (C (j + 1)) (Tm (j + 1))
      (fun r => ∃ s', r = .yield s' ∧ I (j + 1) s' ∧ G s s') fun r => ∀ s', r = .yield s' → Xs j s')
    (hX : ∀ j < n, ∀ s s', I (j + 1) s → (j + 1 = n → s' = s) → G s s' → Xf s' → Xs j s) (s₀ : σ) (h₀ : I 0 s₀) :
    PRuns p L (forIn [:n] s₀ body) (C 0) (Tm 0) (C n) (Tm n) (fun s => I n s ∧ G s₀ s) Xf := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices h : ∀ k a s, a + k = n → I a s → PRuns p L (forIn (List.range' a k) s body) (C a) (Tm a) (C n) (Tm n)
      (fun s' => I n s' ∧ G s s' ∧ (k = 0 → s' = s)) Xf from
    (h n 0 s₀ (by omega) h₀).mono (fun _ h => ⟨h.1, h.2.1⟩) fun _ _ h => h
  intro k
  induction k with
  | zero =>
    intro a s ha hs
    subst ha
    exact PRuns.pure ⟨hs, hGr s, fun _ => rfl⟩ _ _
  | succ k ih =>
    intro a s ha hs
    rw [List.range'_succ, List.forIn_cons]
    refine PRuns.bindD (hbody a (by omega) s hs) (fun r hr => ?_)
      (Qf := fun r s' => ∃ s₁, r = .yield s₁ ∧ I n s' ∧ G s₁ s' ∧ (k = 0 → s' = s₁)) (Xf := fun _ => Xf) ?_ ?_
    · obtain ⟨s₁, rfl, hI, -⟩ := hr
      exact (ih (a + 1) s₁ (by omega) hI).mono (fun s' h => ⟨s₁, rfl, h⟩) fun _ _ h => h
    · rintro r s' ⟨s₁, rfl, -, hG₁⟩ ⟨s₂, h₂, hI, hG₂, -⟩
      cases h₂
      exact ⟨hI, hGt _ _ _ hG₁ hG₂, fun h => absurd h (Nat.succ_ne_zero k)⟩
    · rintro r s' ⟨s₁, rfl, hI₁, -⟩ ⟨s₂, h₂, -, hG₂, hk⟩ hx
      obtain rfl : s₁ = s₂ := ForInStep.yield.inj h₂
      refine ⟨fun s₃ h₃ => ?_, hx⟩
      obtain rfl : s₁ = s₃ := ForInStep.yield.inj h₃
      exact hX a (by omega) s₁ s' hI₁ (fun h => hk (by omega)) hG₂ hx

/-! ## The model's steps -/

/-- **A challenge**, against the model's (`chal`). -/
theorem PRuns.challenge (resp : ℕ → Claims.Msg) (split : Bool) (sm : LState) (tm : Transcript) :
    PRuns p L (challengeP (asksS resp) split) (cur sm) tm (cur (chal p (cfOf resp L) sm tm).2.1)
      (chal p (cfOf resp L) sm tm).2.2 (fun _ => True) fun _ => True := by
  unfold challengeP
  refine (PRuns.sampleK resp _ _).bind fun r _ =>
    (PRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _).bind fun _ _ => ?_
  refine PRuns.ite (fun _ => (PRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _).bind fun _ _ => ?_)
    fun _ => ?_
  all_goals exact (PRuns.get _ _).bind fun _ _ => (PRuns.sendMsg _ _ _).bind fun _ _ => PRuns.pure trivial _ _

/-- **An out-of-domain intro**, against the model's (`ood`), at any level. -/
theorem PRuns.oodIntro (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (g : Array F128) {dim : ℕ} (hdim : dim ≠ 0)
    (split : ℕ) (sm : LState) (tm : Transcript) :
    PRuns p L (oodIntroP (asksS resp) nonce g dim) (cur sm) tm (cur (ood p (cfOf resp L) dim split sm tm).1)
      (ood p (cfOf resp L) dim split sm tm).2 (fun _ => True) fun _ => True := by
  unfold oodIntroP
  refine (PRuns.sampleVec resp hdim _ _).bind fun z _ => PRuns.oodPush (mleF g z) ?_
  exact (PRuns.sendMsg _ _ _).bind fun _ _ => (PRuns.powB nonce _ _).bind fun _ _ =>
    (PRuns.sample resp _ _).bind fun _ _ => PRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _

/-- **A level's query**, against the model's (`query`), on any opened rows and fold. -/
theorem PRuns.query (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit)
    (hq : lv.q ≠ 0) (rows : Array (Array F128)) (fold : Array F256) (sm : LState) (tm : Transcript) :
    PRuns p L (queryP (asksS resp) nonce ms lv cm) (cur sm) tm (cur (Lg.query p (cfOf resp L) lv rows fold sm tm).2.1)
      (Lg.query p (cfOf resp L) lv rows fold sm tm).2.2 (fun _ => True) fun _ => True := by
  unfold queryP
  exact (PRuns.powQ nonce _ _).bind fun _ _ => (PRuns.sampleVec resp hq _ _).bind fun _ _ =>
    (PRuns.powC nonce _ _).bind fun _ _ => (PRuns.sampleVec resp (Nat.succ_ne_zero _) _ _).bind fun _ _ =>
    PRuns.pure trivial _ _

/-- **A level's four challenges**, against the model's (`chals`): the last keeps the folded table, then splits it; at
the last level (`last`) its message is the unsplit table's. -/
theorem levelChals (resp : ℕ → Claims.Msg) (last : Bool) (sm : LState) (tm : Transcript) :
    PRuns p L (forIn [:4] (#[] : Array F256) fun i (f : Array F256) =>
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
          FlockComplete.sendMsg (if last = true then roundPoly d.tab d.w else roundPoly st.tab st.w)
          pure (ForInStep.yield d.tab)
        else do
          let _ ← challengeP (asksS resp) false
          pure (ForInStep.yield f))
      (cur sm) tm (cur (chals p (cfOf resp L) #[] sm tm 4).2.1) (chals p (cfOf resp L) #[] sm tm 4).2.2
      (fun _ => True) fun _ => True :=
  (PRuns.loop _ (fun i => cur (chals p (cfOf resp L) #[] sm tm i).2.1) (fun i => (chals p (cfOf resp L) #[] sm tm i).2.2)
    (fun _ _ => True) (fun _ _ => True) (fun _ => trivial) (fun _ _ _ _ _ => trivial) (fun _ _ => True) (fun _ => True) 4
    (fun _ _ _ _ => PRuns.ite
      (fun _ => (PRuns.sampleK resp _ _).bind fun _ _ =>
        (PRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _).bind fun _ _ => (PRuns.get _ _).bind fun _ _ =>
        (PRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _).bind fun _ _ => (PRuns.get _ _).bind fun _ _ =>
        (PRuns.sendMsg _ _ _).bind fun _ _ => PRuns.pure ⟨_, rfl, trivial, trivial⟩ _ _)
      fun _ => (PRuns.challenge resp false _ _).bind fun _ _ => PRuns.pure ⟨_, rfl, trivial, trivial⟩ _ _)
    (fun _ _ _ _ _ _ _ _ => trivial) _ trivial).mono (fun _ _ => trivial) fun _ _ _ => trivial

/-! ## The prover's Ligerito -/

/-- **The honest prover's Ligerito ends where the model does** on the proof it returns: run from a transcript at which
`L + k` bodies have been sent, at the replies' coins (`cfOf resp L`), its transcript ends at `out`'s and it keeps
`L + k` bodies sent; its first cap is the commitment's. -/
theorem ligeritoP_go (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (tb b : Array F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (L : ℕ)
    (hn : sch.n ≠ 0) (hL : 2 ≤ sch.levels.size) (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0)
    {t : Transcript} {l : Array Claims.Msg} (hl : l.size = L + t.k) {P : LigeritoProof} {t' : Transcript}
    {l' : Array Claims.Msg}
    (hrun : ((ligeritoP (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 tb b salt nonce).run t).go l =
      (.ok (P, t'), l')) :
    P.initialCap = cm0.cap ∧ t' = (out P (cfOf resp L) sch T cm0.cap t).2.2 ∧ l'.size = L + t'.k := by
  unfold ligeritoP at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l)] at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨⟨caps, rproofs, fin, op0⟩, stF⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  have key := (?prog : PRuns (β := Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof) P L _
    (cur (init T)) (start cm0.cap T t) (cur (out P (cfOf resp L) sch T cm0.cap t).2.1)
    (out P (cfOf resp L) sch T cm0.cap t).2.2 (fun _ => True) fun r => r.1 = P.recursiveCaps ∧
      r.2.2.1.yr = P.finalProof.yr) _ (start cm0.cap T t) l _ _ _ _ hl h1
  case prog =>
    refine (PRuns.sampleVec resp hn _ _).bind fun z _ => PRuns.oodPush _ ?_
    refine (PRuns.powB nonce _ _).bind fun _ _ => (PRuns.sample resp _ _).bind fun _ _ =>
      (PRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _).bind fun _ _ => (PRuns.get _ _).bind fun _ _ =>
      (PRuns.sendMsg _ _ _).bind fun _ _ => ?_
    refine PRuns.bind (PRuns.loop _
      (fun j => cur (chals P (cfOf resp L) #[] (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).1
        (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).2 j).2.1)
      (fun j => (chals P (cfOf resp L) #[] (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).1
        (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).2 j).2.2)
      (fun _ _ => True) (fun _ _ => True) (fun _ => trivial) (fun _ _ _ _ _ => trivial) (fun _ _ => True) (fun _ => True)
      sch.k0 (fun j _ s _ => (PRuns.challenge resp _ _ _).bind fun _ _ => PRuns.pure ⟨(), rfl, trivial, trivial⟩ _ _)
      (fun _ _ _ _ _ _ _ _ => trivial) _ trivial) fun _ _ => ?_
    refine (PRuns.get _ _).bind fun d _ => ?_
    refine PRuns.bindD (PRuns.commit h ms _ _ _ _ _) (fun cm1 _ => ?_)
      (Qf := fun cm1 r => Pre #[cm1.cap] r.1) (Xf := fun _ r => r.1 = P.recursiveCaps ∧ r.2.2.1.yr = P.finalProof.yr)
      (fun _ _ _ _ => trivial) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
    refine PRuns.bindY (PRuns.absorbCap cm1.cap P.recursiveCaps[0]! _ _) (fun _ _ => ?_)
      (fun _ r _ hr hx => (Pre.get (A := #[]) (hr.trans (Pre.of_eq hx.1))).symm)
    refine (PRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) 0 _ _).bind fun _ _ =>
      (PRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) 0 _ _).bind fun _ _ => ?_
    refine (PRuns.query resp nonce ms _ cm0 (hq 0 (by omega)) P.initialProof.rows
      (chals P (cfOf resp L) #[] (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).1
        (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).2 sch.k0).1 _ _).bind fun x _ => ?_
    refine (PRuns.sendMsg _ _ _).bind fun _ _ => (PRuns.powB nonce _ _).bind fun _ _ =>
      (PRuns.sample resp _ _).bind fun _ _ =>
      (PRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _).bind fun _ _ => ?_
    refine PRuns.bindD (PRuns.loop (σ := ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof) _
      (fun j => cur (loop P (cfOf resp L) sch (x₀ P (cfOf resp L) sch T cm0.cap t) j).2.1)
      (fun j => (loop P (cfOf resp L) sch (x₀ P (cfOf resp L) sch T cm0.cap t) j).2.2)
      (fun j s => s.1 = (loop P (cfOf resp L) sch (x₀ P (cfOf resp L) sch T cm0.cap t) j).1.1 ∧
        (j < sch.levels.size - 1 → s.2.2.1.size = j + 1))
      (fun s s' => Pre s.2.2.1 s'.2.2.1) (fun _ => Pre.refl _) (fun _ _ _ h₁ h₂ => h₁.trans h₂)
      (fun j s => Pre s.2.2.1 P.recursiveCaps ∧ (j + 1 = sch.levels.size - 1 → s.2.2.2.2.yr = P.finalProof.yr))
      (fun s => s.2.2.1 = P.recursiveCaps ∧ s.2.2.2.2.yr = P.finalProof.yr) (sch.levels.size - 1) ?body
      (fun _ _ s s' _ he hG hx => ⟨hG.trans (Pre.of_eq hx.1), fun hj => by rw [← he hj]; exact hx.2⟩) _
      ⟨rfl, fun _ => rfl⟩)
      (fun σ _ => PRuns.pure (by rfl) _ _) (Qf := fun σ r => r = (σ.2.2.1, σ.2.2.2.1, σ.2.2.2.2, x.1))
      (Xf := fun _ _ => True) (fun σ r hσ hr => hr ▸ hσ.2) (fun σ r _ hr hx => by subst hr; exact ⟨hx, trivial⟩)
    case body =>
      intro j hj s hs
      by_cases hj1 : j + 1 < sch.levels.size - 1
      · rw [loop_stepN P (cfOf resp L) sch _ hj1]
        have hsz := hs.2 hj
        generalize loop P (cfOf resp L) sch (x₀ P (cfOf resp L) sch T cm0.cap t) j = x at hs ⊢
        rw [hs.1]
        refine PRuns.bind (levelChals resp _ x.2.1 x.2.2) fun f _ => ?_
        refine PRuns.ite (fun _ => ?_) fun hne => absurd (by simp; omega) hne
        refine PRuns.bindD (PRuns.commit h ms _ _ _ _ _) (fun cmN _ => ?_)
          (Qf := fun cmN r => (∃ s', r = .yield s' ∧ (s'.1 = (stepN P (cfOf resp L) sch j x).1.1 ∧
            (j + 1 < sch.levels.size - 1 → s'.2.2.1.size = j + 1 + 1)) ∧ Pre s.2.2.1 s'.2.2.1) ∧
            ∀ s', r = .yield s' → s'.2.2.1 = s.2.2.1.push cmN.cap)
          (fun _ _ _ hr => hr.1) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
        refine PRuns.bindY (PRuns.absorbCap cmN.cap P.recursiveCaps[j + 1]! _ _) (fun _ _ => ?_)
          (fun _ r _ hr hx => ?_)
        · refine (PRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) (j + 1) _ _).bind fun _ _ =>
            (PRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) (j + 1) _ _).bind fun _ _ => ?_
          refine (PRuns.query resp nonce ms _ s.2.1 (hq (j + 1) (by omega)) P.recursiveProofs[j]!.rows
            (chals P (cfOf resp L) #[] x.2.1 x.2.2 4).1 _ _).bind fun q _ => ?_
          refine (PRuns.sendMsg _ _ _).bind fun _ _ => (PRuns.powB nonce _ _).bind fun _ _ =>
            (PRuns.sample resp _ _).bind fun _ _ =>
            (PRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _).bind fun _ _ => PRuns.pure ?_ _ _
          refine ⟨⟨_, rfl, ⟨rfl, fun _ => by rw [Array.size_push, hsz]⟩, Pre.push _ _⟩, fun s' e => ?_⟩
          obtain rfl := ForInStep.yield.inj e
          rfl
        · obtain ⟨⟨s', rfl, -⟩, hcap⟩ := hr
          have hp := (hx s' rfl).1
          rw [hcap s' rfl] at hp
          have := Pre.get hp
          rw [hsz] at this
          exact this.symm
      · rw [loop_stepL P (cfOf resp L) sch _ hj1]
        have hjR : j + 1 = sch.levels.size - 1 := by omega
        generalize loop P (cfOf resp L) sch (x₀ P (cfOf resp L) sch T cm0.cap t) j = x at hs ⊢
        refine PRuns.bindD (levelChals resp _ x.2.1 x.2.2) (fun f _ => ?_)
          (Qf := fun f r => (∃ s', r = .yield s' ∧ (s'.1 = (stepL P (cfOf resp L) sch x).1.1 ∧
            (j + 1 < sch.levels.size - 1 → s'.2.2.1.size = j + 1 + 1)) ∧ Pre s.2.2.1 s'.2.2.1) ∧
            ∀ s', r = .yield s' → s'.2.2.2.2.yr = Array.foldl (fun acc v => (acc.push v.c0).push v.c1) #[] f)
          (Xf := fun _ r => ∀ s', r = .yield s' → Pre s'.2.2.1 P.recursiveCaps ∧
            (j + 1 = sch.levels.size - 1 → s'.2.2.2.2.yr = P.finalProof.yr))
          (fun _ _ _ hr => hr.1) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
        refine PRuns.ite (fun hc => absurd hc (by simp [hjR])) fun _ => ?_
        refine PRuns.bindY (PRuns.absorbAll _ P.finalProof.yr _ _) (fun _ _ => ?_) (fun _ r _ hr hx => ?_)
        · refine (PRuns.query resp nonce ms _ s.2.1 (hq _ (by omega)) P.finalProof.rows
            (chals P (cfOf resp L) #[] x.2.1 x.2.2 4).1 _ _).bind fun q _ => ?_
          refine (PRuns.powB nonce _ _).bind fun _ _ => (PRuns.sample resp _ _).bind fun _ _ => PRuns.pure ?_ _ _
          refine ⟨⟨_, rfl, ⟨hs.1, fun h => absurd h (by omega)⟩, Pre.refl _⟩, fun s' e => ?_⟩
          obtain rfl := ForInStep.yield.inj e
          rfl
        · obtain ⟨⟨s', rfl, -⟩, hyr⟩ := hr
          exact (hyr s' rfl).symm.trans ((hx s' rfl).2 hjR)
  obtain ⟨-, -, hl₁, hC⟩ := key
  subst hP
  exact ⟨rfl, (hC ⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _⟩ ⟨rfl, rfl⟩ rfl rfl).2, hl₁⟩

end Lg

end FlockComplete.Replay
