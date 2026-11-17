module

public import Proofs.Flock.Soundness.Audit.HarmBudget
import Mathlib.Algebra.Order.Algebra
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Proofs.Flock.Soundness.Game

/-! The cumulative harm budget over a lifetime of one-stage audits. -/

@[expose] public section

namespace FlockSoundness.Audit.Adaptive

open Game Finset Play
open scoped ENNReal

variable {n : ℕ}

/-- After an audit: continue with `next` if it accepted, else reject. -/
def afterAudit (next : Game Bool) (o : Bool × Finset (Fin n)) : Game Bool := cond o.1 next (.ret false)

/-- One stage of a lifetime: the prover stops (accepting) or runs the audit, then continues with
`next` if it accepted. -/
def stage (L : SamplingDesign n) (Reg : Type) (session : Finset (Fin n) → Reg → Game Bool) (next : Game Bool) :
    Game Bool :=
  .send Bool fun go => cond go ((audit L Reg session).bind (afterAudit next)) (.ret true)

/-- **A lifetime of at most `N` one-stage audits**, round `m` by sampling design `L m`. -/
def life (L : ℕ → SamplingDesign n) (Reg : ℕ → Type) (session : (m : ℕ) → Finset (Fin n) → Reg m → Game Bool) :
    ℕ → Game Bool
  | 0 => .ret true
  | N + 1 => stage (L N) (Reg N) (session N) (life L Reg session N)

/-- The run tree of `g.bind f` under a strategy, given the run tree of each continuation. -/
noncomputable def ofBindPlay {α β : Type} {N : ℕ} (f : α → Game β)
    (cont : (a : α) → Strategy (f a) → Play n N) : (g : Game α) → Strategy (g.bind f) → Play n N
  | .ret a, s => cont a s
  | .send _ k, s => ofBindPlay f cont (k s.1) s.2
  | @Game.coin _ C _ _ k, s => .coin C fun c => ofBindPlay f cont (k c) (s c)

/-- The run tree after an audit's verdict. -/
noncomputable def contPlay {next : Game Bool} {N : ℕ} (rest : Strategy next → Play n N) :
    (b : Bool) → Strategy (cond b next (.ret false)) → Play n N
  | true, s => rest s
  | false, _ => .stop false

theorem samplingDesigns_ofBindPlay {α β : Type} {N : ℕ} {S : SamplingDesign n → Prop} (f : α → Game β)
    (cont : (a : α) → Strategy (f a) → Play n N) (hc : ∀ a t, (cont a t).SamplingDesigns S) :
    ∀ (g : Game α) (s : Strategy (g.bind f)), (ofBindPlay f cont g s).SamplingDesigns S
  | .ret a, s => hc a s
  | .send _ k, s => samplingDesigns_ofBindPlay f cont hc (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => fun c => samplingDesigns_ofBindPlay f cont hc (k c) (s c)

theorem sampled_ofBindPlay {α β : Type} {N : ℕ} {err : ℕ → ℝ≥0∞} (f : α → Game β)
    (cont : (a : α) → Strategy (f a) → Play n N) (hc : ∀ a t, (cont a t).Drawn err) :
    ∀ (g : Game α) (s : Strategy (g.bind f)), (ofBindPlay f cont g s).Drawn err
  | .ret a, s => hc a s
  | .send _ k, s => sampled_ofBindPlay f cont hc (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => fun c => sampled_ofBindPlay f cont hc (k c) (s c)

theorem pass_ofBindPlay {α β : Type} {N : ℕ} (f : α → Game β) (cont : (a : α) → Strategy (f a) → Play n N)
    (P : α → Prop) (hc : ∀ a t, (cont a t).pass = prob P (.ret a) PUnit.unit) :
    ∀ (g : Game α) (s : Strategy (g.bind f)), (ofBindPlay f cont g s).pass = prob P g (Strategy.ofBind f g s)
  | .ret a, s => hc a s
  | .send _ k, s => pass_ofBindPlay f cont P hc (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => by
      show avg (fun c => (ofBindPlay f cont (k c) (s c)).pass) =
        avg fun c => prob P (k c) (Strategy.ofBind f (k c) (s c))
      exact congrArg avg (funext fun c => pass_ofBindPlay f cont P hc (k c) (s c))

theorem prEv_ofBindPlay {α β : Type} {N : ℕ} (h : Fin n → ℝ) (Ev : Bool → ℝ → Prop) (x : ℝ)
    (f : α → Game β) (cont : (a : α) → Strategy (f a) → Play n N) (p : β → Prop)
    (hc : ∀ a t, (cont a t).prEv h Ev x = prob p (f a) t) :
    ∀ (g : Game α) (s : Strategy (g.bind f)), (ofBindPlay f cont g s).prEv h Ev x = prob p (g.bind f) s
  | .ret a, s => hc a s
  | .send _ k, s => prEv_ofBindPlay h Ev x f cont p hc (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => by
      show avg (fun c => (ofBindPlay f cont (k c) (s c)).prEv h Ev x) =
        avg fun c => prob p ((k c).bind f) (s c)
      exact congrArg avg (funext fun c => prEv_ofBindPlay h Ev x f cont p hc (k c) (s c))

theorem samplingDesigns_contPlay {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N} {S : SamplingDesign n → Prop}
    (hrest : ∀ t, (rest t).SamplingDesigns S) :
    ∀ (b : Bool) (t : Strategy (cond b next (.ret false))), (contPlay rest b t).SamplingDesigns S
  | true, t => hrest t
  | false, _ => trivial

theorem sampled_contPlay {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N} {err : ℕ → ℝ≥0∞}
    (hrest : ∀ t, (rest t).Drawn err) :
    ∀ (b : Bool) (t : Strategy (cond b next (.ret false))), (contPlay rest b t).Drawn err
  | true, t => hrest t
  | false, _ => trivial

theorem pass_contPlay {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N}
    (hpass : ∀ t, (rest t).pass = 1) :
    ∀ (o : Bool × Finset (Fin n)) (t : Strategy (afterAudit next o)),
      (contPlay rest o.1 t).pass = prob (fun o : Bool × Finset (Fin n) => o.1 = true) (.ret o) PUnit.unit
  | (true, D), t => by
      show (rest t).pass = prob (fun o : Bool × Finset (Fin n) => o.1 = true) (.ret (true, D)) PUnit.unit
      rw [hpass t]
      simp [prob]
  | (false, D), t => by
      show (Play.stop false : Play n N).pass =
        prob (fun o : Bool × Finset (Fin n) => o.1 = true) (.ret (false, D)) PUnit.unit
      simp [pass_stop, prob]

theorem accept_contPlay {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N} (h : Fin n → ℝ)
    (hrest : ∀ t x, (rest t).prEv h (fun acc _ => acc = true) x = prob (· = true) next t) :
    ∀ (b : Bool) (t : Strategy (cond b next (.ret false))) (x : ℝ),
      (contPlay rest b t).prEv h (fun acc _ => acc = true) x = prob (· = true) (cond b next (.ret false)) t
  | true, t, x => hrest t x
  | false, _, _ => rfl

section Stage

variable {L : SamplingDesign n} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

/-- The run tree of one audit and what follows it: a round over the committed transcript's incorrect
proof blocks. -/
noncomputable def roundPlay (A : Analysis L Reg session) {next : Game Bool} {N : ℕ}
    (rest : Strategy next → Play n N) (t : Strategy ((audit L Reg session).bind (afterAudit next))) :
    Play n (N + 1) :=
  .round L (A.incorrect (A.committedOf (Strategy.ofBind (afterAudit next) (audit L Reg session) t))) fun ω =>
    ofBindPlay (afterAudit next) (fun o => contPlay rest o.1)
      ((session (L.draw ω) t.1).map fun ok => (ok, L.draw ω)) (t.2 ω)

/-- The run tree after the prover's choice to stop or to run the audit. -/
noncomputable def goPlay (A : Analysis L Reg session) {next : Game Bool} {N : ℕ}
    (rest : Strategy next → Play n N) :
    (go : Bool) → Strategy (cond go ((audit L Reg session).bind (afterAudit next)) (.ret true)) →
      Play n (N + 1)
  | true, t => roundPlay A rest t
  | false, _ => .stop true

/-- The run tree of one stage. -/
noncomputable def stagePlay (A : Analysis L Reg session) {next : Game Bool} {N : ℕ}
    (rest : Strategy next → Play n N) (s : Strategy (stage L Reg session next)) : Play n (N + 1) :=
  goPlay A rest s.1 s.2

theorem prob_audit_bind {next : Game Bool} (P : Bool × Finset (Fin n) → Prop)
    (t : Strategy ((audit L Reg session).bind (afterAudit next))) :
    prob P (audit L Reg session) (Strategy.ofBind (afterAudit next) (audit L Reg session) t) =
      avg fun ω => prob P ((session (L.draw ω) t.1).map fun ok => (ok, L.draw ω))
        (Strategy.ofBind (afterAudit next) ((session (L.draw ω) t.1).map fun ok => (ok, L.draw ω)) (t.2 ω)) :=
  rfl

theorem prob_audit_bind' {next : Game Bool} (p : Bool → Prop)
    (t : Strategy ((audit L Reg session).bind (afterAudit next))) :
    prob p ((audit L Reg session).bind (afterAudit next)) t =
      avg fun ω => prob p (((session (L.draw ω) t.1).map fun ok => (ok, L.draw ω)).bind (afterAudit next))
        (t.2 ω) :=
  rfl

theorem prEv_roundPlay (A : Analysis L Reg session) (h : Fin n → ℝ) (Ev : Bool → ℝ → Prop) {next : Game Bool}
    {N : ℕ} (rest : Strategy next → Play n N) (t : Strategy ((audit L Reg session).bind (afterAudit next)))
    (x : ℝ) :
    (roundPlay A rest t).prEv h Ev x =
      avg fun ω => (ofBindPlay (afterAudit next) (fun o => contPlay rest o.1)
        ((session (L.draw ω) t.1).map fun ok => (ok, L.draw ω)) (t.2 ω)).prEv h Ev
          (x + ∑ u ∈ A.incorrect (A.committedOf (Strategy.ofBind (afterAudit next) (audit L Reg session) t)), h u) :=
  rfl

theorem sampled_term {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N}
    (hpass : ∀ t, (rest t).pass = 1) (B D : Finset (Fin n)) (g : Game Bool)
    (s : Strategy ((g.map fun ok => (ok, D)).bind (afterAudit next))) :
    (if Disjoint D B then 0 else
        (ofBindPlay (afterAudit next) (fun o => contPlay rest o.1) (g.map fun ok => (ok, D)) s).pass) =
      prob (fun o => o.1 = true ∧ ¬ Disjoint B o.2) (g.map fun ok => (ok, D))
        (Strategy.ofBind (afterAudit next) (g.map fun ok => (ok, D)) s) := by
  rw [prob_map]
  by_cases hd : Disjoint D B
  · rw [ite_eq_left hd]
    exact le_antisymm zero_le
      ((prob_mono (q := fun _ => False) (fun _ ho => ho.2 hd.symm) _ _).trans (prob_false _ _).le)
  · rw [ite_eq_right hd, pass_ofBindPlay (afterAudit next) (fun o => contPlay rest o.1) (fun o => o.1 = true)
      (pass_contPlay hpass), prob_map]
    exact le_antisymm (prob_mono (fun _ ho => ⟨ho, fun h' => hd h'.symm⟩) _ _) (prob_mono (fun _ ho => ho.1) _ _)

theorem sampled_roundPlay (A : Analysis L Reg session) {εks εbind : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.BindSound εbind) {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N}
    {err : ℕ → ℝ≥0∞} (hpass : ∀ t, (rest t).pass = 1) (hrest : ∀ t, (rest t).Drawn err)
    (hN : εks + εbind ≤ err N) (t : Strategy ((audit L Reg session).bind (afterAudit next))) :
    (roundPlay A rest t).Drawn err := by
  refine ⟨?_, fun ω => ?_⟩
  · refine le_trans (le_of_eq ?_)
      ((audit_sampled A hks hlink (Strategy.ofBind (afterAudit next) (audit L Reg session) t)).trans hN)
    refine Eq.trans ?_ (prob_audit_bind _ t).symm
    exact congrArg avg (funext fun ω => sampled_term hpass _ (L.draw ω) (session (L.draw ω) t.1) (t.2 ω))
  · exact sampled_ofBindPlay (afterAudit next) (fun o => contPlay rest o.1)
      (fun o t' => sampled_contPlay hrest o.1 t') _ (t.2 ω)

theorem samplingDesigns_roundPlay (A : Analysis L Reg session) {S : SamplingDesign n → Prop} (hL : S L) {next : Game Bool} {N : ℕ}
    {rest : Strategy next → Play n N} (hrest : ∀ t, (rest t).SamplingDesigns S)
    (t : Strategy ((audit L Reg session).bind (afterAudit next))) : (roundPlay A rest t).SamplingDesigns S :=
  ⟨hL, fun ω => samplingDesigns_ofBindPlay (afterAudit next) (fun o => contPlay rest o.1)
    (fun o t' => samplingDesigns_contPlay hrest o.1 t') _ (t.2 ω)⟩

theorem accept_roundPlay (A : Analysis L Reg session) (h : Fin n → ℝ) {next : Game Bool} {N : ℕ}
    {rest : Strategy next → Play n N}
    (hrest : ∀ t x, (rest t).prEv h (fun acc _ => acc = true) x = prob (· = true) next t)
    (t : Strategy ((audit L Reg session).bind (afterAudit next))) (x : ℝ) :
    (roundPlay A rest t).prEv h (fun acc _ => acc = true) x =
      prob (· = true) ((audit L Reg session).bind (afterAudit next)) t := by
  refine (prEv_roundPlay A h _ rest t x).trans (Eq.trans ?_ (prob_audit_bind' (· = true) t).symm)
  exact congrArg avg (funext fun ω => prEv_ofBindPlay h _ _ (afterAudit next) (fun o => contPlay rest o.1)
    (· = true) (fun o t' => accept_contPlay h hrest o.1 t' _) _ (t.2 ω))

theorem samplingDesigns_goPlay (A : Analysis L Reg session) {S : SamplingDesign n → Prop} (hL : S L) {next : Game Bool} {N : ℕ}
    {rest : Strategy next → Play n N} (hrest : ∀ t, (rest t).SamplingDesigns S) :
    ∀ (go : Bool) (t : Strategy (cond go ((audit L Reg session).bind (afterAudit next)) (.ret true))),
      (goPlay A rest go t).SamplingDesigns S
  | true, t => samplingDesigns_roundPlay A hL hrest t
  | false, _ => trivial

theorem sampled_goPlay (A : Analysis L Reg session) {εks εbind : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.BindSound εbind) {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N}
    {err : ℕ → ℝ≥0∞} (hpass : ∀ t, (rest t).pass = 1) (hrest : ∀ t, (rest t).Drawn err)
    (hN : εks + εbind ≤ err N) :
    ∀ (go : Bool) (t : Strategy (cond go ((audit L Reg session).bind (afterAudit next)) (.ret true))),
      (goPlay A rest go t).Drawn err
  | true, t => sampled_roundPlay A hks hlink hpass hrest hN t
  | false, _ => trivial

theorem pass_goPlay (A : Analysis L Reg session) {next : Game Bool} {N : ℕ} (rest : Strategy next → Play n N) :
    ∀ (go : Bool) (t : Strategy (cond go ((audit L Reg session).bind (afterAudit next)) (.ret true))),
      (goPlay A rest go t).pass = 1
  | true, _ => rfl
  | false, _ => by
      show (Play.stop true : Play n (N + 1)).pass = 1
      simp [pass_stop]

theorem accept_goPlay (A : Analysis L Reg session) (h : Fin n → ℝ) {next : Game Bool} {N : ℕ}
    {rest : Strategy next → Play n N}
    (hrest : ∀ t x, (rest t).prEv h (fun acc _ => acc = true) x = prob (· = true) next t) :
    ∀ (go : Bool) (t : Strategy (cond go ((audit L Reg session).bind (afterAudit next)) (.ret true))) (x : ℝ),
      (goPlay A rest go t).prEv h (fun acc _ => acc = true) x =
        prob (· = true) (cond go ((audit L Reg session).bind (afterAudit next)) (.ret true)) t
  | true, t, x => accept_roundPlay A h hrest t x
  | false, _, _ => rfl

theorem samplingDesigns_stagePlay (A : Analysis L Reg session) {S : SamplingDesign n → Prop} (hL : S L) {next : Game Bool} {N : ℕ}
    {rest : Strategy next → Play n N} (hrest : ∀ t, (rest t).SamplingDesigns S) (s : Strategy (stage L Reg session next)) :
    (stagePlay A rest s).SamplingDesigns S :=
  samplingDesigns_goPlay A hL hrest s.1 s.2

theorem sampled_stagePlay (A : Analysis L Reg session) {εks εbind : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.BindSound εbind) {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N}
    {err : ℕ → ℝ≥0∞} (hpass : ∀ t, (rest t).pass = 1) (hrest : ∀ t, (rest t).Drawn err)
    (hN : εks + εbind ≤ err N) (s : Strategy (stage L Reg session next)) : (stagePlay A rest s).Drawn err :=
  sampled_goPlay A hks hlink hpass hrest hN s.1 s.2

theorem pass_stagePlay (A : Analysis L Reg session) {next : Game Bool} {N : ℕ} (rest : Strategy next → Play n N)
    (s : Strategy (stage L Reg session next)) : (stagePlay A rest s).pass = 1 :=
  pass_goPlay A rest s.1 s.2

theorem accept_stagePlay (A : Analysis L Reg session) (h : Fin n → ℝ) {next : Game Bool} {N : ℕ}
    {rest : Strategy next → Play n N}
    (hrest : ∀ t x, (rest t).prEv h (fun acc _ => acc = true) x = prob (· = true) next t)
    (s : Strategy (stage L Reg session next)) (x : ℝ) :
    (stagePlay A rest s).prEv h (fun acc _ => acc = true) x = prob (· = true) (stage L Reg session next) s :=
  accept_goPlay A h hrest s.1 s.2 x

end Stage

section Life

variable {L : ℕ → SamplingDesign n} {Reg : ℕ → Type} {session : (m : ℕ) → Finset (Fin n) → Reg m → Game Bool}

/-- **The run tree of a lifetime** under a prover strategy, round `m`'s incorrect proof blocks those of its
committed transcript under the analysis `A m`. -/
noncomputable def lifePlay (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) :
    (N : ℕ) → Strategy (life L Reg session N) → Play n N
  | 0, _ => .stop true
  | N + 1, s => stagePlay (A N) (lifePlay A N) s

theorem samplingDesigns_lifePlay (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) {S : SamplingDesign n → Prop}
    (hL : ∀ m, S (L m)) : ∀ (N : ℕ) (s : Strategy (life L Reg session N)), (lifePlay A N s).SamplingDesigns S
  | 0, _ => trivial
  | N + 1, s => samplingDesigns_stagePlay (A N) (hL N) (samplingDesigns_lifePlay A hL N) s

theorem pass_lifePlay (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) :
    ∀ (N : ℕ) (s : Strategy (life L Reg session N)), (lifePlay A N s).pass = 1
  | 0, _ => by
      show (Play.stop true : Play n 0).pass = 1
      simp [pass_stop]
  | N + 1, s => pass_stagePlay (A N) (lifePlay A N) s

theorem sampled_lifePlay (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) {εks εbind : ℕ → ℝ≥0∞}
    (hks : ∀ m, (A m).KnowledgeSound (εks m)) (hlink : ∀ m, (A m).BindSound (εbind m)) :
    ∀ (N : ℕ) (s : Strategy (life L Reg session N)), (lifePlay A N s).Drawn fun m => εks m + εbind m
  | 0, _ => trivial
  | N + 1, s => sampled_stagePlay (A N) (hks N) (hlink N) (pass_lifePlay A N) (sampled_lifePlay A hks hlink N) le_rfl s

theorem life_accept (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) (h : Fin n → ℝ) :
    ∀ (N : ℕ) (s : Strategy (life L Reg session N)) (x : ℝ),
      (lifePlay A N s).prEv h (fun acc _ => acc = true) x = prob (· = true) (life L Reg session N) s
  | 0, _, _ => rfl
  | N + 1, s, x => accept_stagePlay (A N) h (life_accept A h N) s x

/-- **The cumulative harm budget over a lifetime of one-stage audits.** For sampling designs meeting the design
condition with weights `π`, harms `h` with `λ h ≤ π` and `λ ≥ 0`, and every prover strategy, the
lifetime accepts with total harm `Σ_t h(incorrect(X_t))` at least `H` with probability at most
`e^{−λH} + Σ_{m<N} (ε_ks,m + ε_bind,m)`. -/
theorem life_harm_budget (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) {εks εbind : ℕ → ℝ≥0∞}
    (hks : ∀ m, (A m).KnowledgeSound (εks m)) (hlink : ∀ m, (A m).BindSound (εbind m))
    {π h : Fin n → ℝ} (hL : ∀ m, (L m).Design π) {lam : ℝ} (hlam0 : 0 ≤ lam) (hlam : ∀ u, lam * h u ≤ π u)
    (H : ℝ) (N : ℕ) (s : Strategy (life L Reg session N)) :
    (lifePlay A N s).prEv h (fun acc x => acc = true ∧ H ≤ x) 0 ≤
      ENNReal.ofReal (Real.exp (-(lam * H))) + ∑ m ∈ range N, (εks m + εbind m) :=
  harm_budget hlam0 hlam H (lifePlay A N s) (samplingDesigns_lifePlay (S := fun L' => L'.Design π) A hL N s)
    (sampled_lifePlay A hks hlink N s)

end Life

end FlockSoundness.Audit.Adaptive
