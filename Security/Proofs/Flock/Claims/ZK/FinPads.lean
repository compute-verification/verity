import Proofs.Flock.Claims.ZK.FinExe
import Proofs.Flock.Claims.ZK.Round1Root

/-!
# The pads' opened columns and salts read from the views are the session's (`repOpenRest_pads`)
-/

namespace FlockComplete

open Flock

/-- **What every returning run of `m` satisfies**: `P` at its value, its last transcript and its last history. -/
def RunPost {α : Type} (m : TMW PS α) (P : α → Transcript → Array Claims.Msg → Prop) : Prop :=
  ∀ s l a t l', (m.run s).go l = (.ok (a, t), l') → P a t l'

namespace RunPost

variable {α β : Type} {P : β → Transcript → Array Claims.Msg → Prop}

theorem bind {x : TMW PS α} {f : α → TMW PS β} (h : ∀ a, RunPost (f a) P) : RunPost (x >>= f) P := by
  intro s l b t l' e
  rw [StateT.run_bind] at e
  obtain ⟨⟨a, s'⟩, l₁, -, e₂⟩ := JsonRT.go_bind_ok e
  exact h a s' l₁ b t l' e₂

theorem bind_pure {x : TMW PS α} {g : α → β} (h : RunPost x fun a t l => P (g a) t l) :
    RunPost (x >>= fun a => pure (g a)) P := by
  intro s l b t l' e
  rw [StateT.run_bind] at e
  obtain ⟨⟨a, s'⟩, l₁, e₁, e₂⟩ := JsonRT.go_bind_ok e
  have e₃ : ((pure (g a) : TMW PS β).run s').go l₁ = (.ok (g a, s'), l₁) := rfl
  rw [e₃] at e₂
  simp only [Prod.mk.injEq, Except.ok.injEq] at e₂
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := e₂
  exact h s l a s' l₁ e₁

theorem ite {c : Prop} [Decidable c] {x y : TMW PS β} (hx : RunPost x P) (hy : RunPost y P) :
    RunPost (if c then x else y) P := by
  split
  · exact hx
  · exact hy

theorem throw {e : String} : RunPost (MonadExcept.throw e : TMW PS β) P := by
  intro s l a t l' h
  cases h

theorem mono {x : TMW PS β} {Q : β → Transcript → Array Claims.Msg → Prop} (h : RunPost x Q)
    (hQP : ∀ a t l, Q a t l → P a t l) : RunPost x P :=
  fun s l a t l' e => hQP _ _ _ (h s l a t l' e)

end RunPost

/-- **A draw of `m ≠ 0` coins answered by `asksCut`** returns the coin of its round, the last of its history. -/
theorem sampleVecP_asksCut (coin : ℕ → ℕ → F128) (n : ℕ) {m : ℕ} (hm : m ≠ 0) :
    RunPost (sampleVecP (asksCut coin n) m) fun v _ l => v = (Array.range m).map (coin (l.size - 1)) := by
  intro s l v t l' e
  simp only [sampleVecP, squeezeP, StateT.run, hm, ite_false] at e
  obtain ⟨a, l₁, e₁, e₂⟩ := JsonRT.go_bind_ok e
  have e₀ : ((asksCut coin n).round (Transcript.framed s 2 m)).go l =
      (if n ≤ l.size then (MonadExcept.throw "past the openings" : PS (ℕ → F128)) else pure (coin l.size)).go
        (l.push (Claims.toMsg (Transcript.framed s 2 m))) := rfl
  rw [e₀] at e₁
  by_cases hn : n ≤ l.size
  · simp only [hn, ↓reduceIte] at e₁
    cases e₁
  · simp only [hn, ↓reduceIte] at e₁
    simp only [JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at e₁
    obtain ⟨rfl, rfl⟩ := e₁
    simp only [JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at e₂
    obtain ⟨⟨rfl, -⟩, rfl⟩ := e₂
    simp

/-- **The pads positions the inner proof queries** at its last history `l`: its last round's `Q` coins, each mod `n`. -/
def padQs (coin : ℕ → ℕ → F128) (l : Array Claims.Msg) (n : ℕ) : Array ℕ :=
  ((Array.range Zk.Q).map (coin (l.size - 1))).map fun w => w.lo.toNat % n

/-- **An inner proof opens `ic` at the queried positions**: its columns and salts are `ic`'s at `padQs`. -/
def PadsAt (coin : ℕ → ℕ → F128) (ic : InnerCommit) (ip : Zk.InnerProof) (l : Array Claims.Msg) : Prop :=
  ip.opened = (padQs coin l ic.ch.size).map (fun j => (ic.ch[j]!, ic.cm[j]!)) ∧
    ip.salts = (padQs coin l ic.ch.size).map fun j => ic.salts[j]!

theorem padsAt_last (coin : ℕ → ℕ → F128) (n : ℕ) (ic : InnerCommit) {g : Array F128 → Zk.InnerProof}
    (hg : ∀ v, (g v).opened = (v.map fun w => w.lo.toNat % ic.ch.size).map (fun j => (ic.ch[j]!, ic.cm[j]!)) ∧
      (g v).salts = (v.map fun w => w.lo.toNat % ic.ch.size).map fun j => ic.salts[j]!) :
    RunPost (sampleVecP (asksCut coin n) Zk.Q >>= fun v => pure (g v)) fun ip _ l => PadsAt coin ic ip l :=
  RunPost.bind_pure (RunPost.mono (sampleVecP_asksCut coin n (by decide)) fun v _ l hv => by
    subst hv
    exact hg _)

/-- **`innerCut` answered by `asksCut` opens its commitment at the queried positions.** -/
theorem innerCut_padsAt (coin : ℕ → ℕ → F128) (n : ℕ) (lay : Zk.PadLayout) (ic : InnerCommit) (finalC : F128)
    (src : RepSrc) (cons : Array (String × Zk.Lin)) :
    RunPost (innerCut (asksCut coin n) lay ic finalC src cons) fun ip _ l => PadsAt coin ic ip l := by
  unfold innerCut
  refine RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ =>
    RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ => ?_
  dsimp only
  refine RunPost.ite (RunPost.bind fun _ => ?_) ?_ <;>
    exact RunPost.bind fun _ => padsAt_last coin n ic fun _ => ⟨rfl, rfl⟩

/-- **`innerP` answered by `asksCut` opens its commitment at the queried positions.** -/
theorem innerP_padsAt (coin : ℕ → ℕ → F128) (n : ℕ) (hs : Hashes PS) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) :
    RunPost (innerP (asksCut coin n) hs lay p ic finalC cons) fun ip _ l => PadsAt coin ic ip l := by
  unfold innerP
  refine RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ =>
    RunPost.bind fun _ => ?_
  dsimp only
  refine RunPost.ite (RunPost.bind fun _ => ?_) ?_ <;>
  · refine RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ => ?_
    refine RunPost.ite (RunPost.bind fun _ => ?_) ?_ <;>
      exact RunPost.bind fun _ => padsAt_last coin n ic fun _ => ⟨rfl, rfl⟩

/-- **The rep from the source answered by `asksCut` opens its pads commitment at the queried positions.** -/
theorem repCut_padsAt (coin : ℕ → ℕ → F128) (n : ℕ) (hs : Hashes PS) (st : Setup) (sch : Schedule) (rep : ℕ)
    (dom : ByteArray) (cm0 : Commit) (extras : Array Claim) (ic : InnerCommit) (src : RepSrc)
    (prior : Array (List F128)) (words : List ℕ) :
    RunPost (repCut (asksCut coin n) hs st sch rep dom cm0 extras ic src prior words)
      fun p _ l => PadsAt coin ic p.inner l := by
  unfold repCut
  exact RunPost.bind fun ⟨_, _, _, _, _⟩ => RunPost.bind fun ⟨_, _, _⟩ => RunPost.bind fun _ =>
    RunPost.bind fun ⟨_, _⟩ => RunPost.bind fun _ =>
      RunPost.bind_pure (RunPost.mono (innerCut_padsAt coin n _ ic _ src _) fun _ _ _ h => h)

/-- **The session's rep answered by `asksCut` opens its pads commitment at the queried positions.** -/
theorem repPZk_padsAt (coin : ℕ → ℕ → F128) (n : ℕ) (hs : Hashes PS) (st : Setup) (sch : Schedule) (tPad rep : ℕ)
    (dom : ByteArray) (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk)
    (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) :
    RunPost (repPZk (asksCut coin n) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words)
      fun p _ l => PadsAt coin ic p.inner l := by
  unfold repPZk
  exact RunPost.bind fun ⟨_, _⟩ => RunPost.bind fun ⟨_, _⟩ => RunPost.bind fun _ => RunPost.bind fun ⟨_, _, _⟩ =>
    RunPost.bind fun ⟨_, _, _⟩ => RunPost.bind fun _ => RunPost.bind fun ⟨_, _⟩ => RunPost.bind fun _ =>
      RunPost.bind_pure (RunPost.mono (innerP_padsAt coin n hs _ pads ic _ _) fun _ _ _ h => h)

end FlockComplete
