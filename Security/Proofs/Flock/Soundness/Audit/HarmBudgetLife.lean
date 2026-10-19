import Proofs.Flock.Soundness.Audit.HarmBudget

/-!
# The cumulative harm budget over a lifetime of one-stage audits

`life L Reg session N` is a lifetime of at most `N` one-stage audits (`audit`), round `m` by law
`L m`: before each, the prover chooses to stop (accepting) or to run it, and a rejected audit ends
the session. The prover's choices are adaptive, so the number of rounds is a stopping time.

For a prover strategy `s`, `lifePlay A N s` is its run tree (`Adaptive.Play`), each round's wrong
units being those of its committed transcript (`Analysis.committedOf`), which the prover's state at
registration fixes before the round's draw. It is faithful: its acceptance is the lifetime's
(`life_accept`). Each round's drawn-unit bound is `audit_drawn` (`drawn_lifePlay`), so `harm_budget`
gives the lifetime's bound (`life_harm_budget`):
$$\Pr\Big[\text{accept} \wedge \sum_{t \le T} h(\mathrm{wrong}(X_t)) \ge H\Big]
  \le e^{-\lambda H} + \sum_{m < N} (\varepsilon_{ks,m} + \delta_{link,m}).$$
-/

namespace FlockSoundness.Audit.Adaptive

open Game Finset Play
open scoped ENNReal

variable {n : ℕ}

/-- After an audit: continue with `next` if it accepted, else reject. -/
def afterAudit (next : Game Bool) (o : Bool × Finset (Fin n)) : Game Bool := cond o.1 next (.ret false)

/-- One stage of a lifetime: the prover stops (accepting) or runs the audit, then continues with
`next` if it accepted. -/
def stage (L : Law n) (Reg : Type) (session : Finset (Fin n) → Reg → Game Bool) (next : Game Bool) :
    Game Bool :=
  .send Bool fun go => cond go ((audit L Reg session).bind (afterAudit next)) (.ret true)

/-- **A lifetime of at most `N` one-stage audits**, round `m` by law `L m`. -/
def life (L : ℕ → Law n) (Reg : ℕ → Type) (session : (m : ℕ) → Finset (Fin n) → Reg m → Game Bool) :
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

theorem laws_ofBindPlay {α β : Type} {N : ℕ} {S : Law n → Prop} (f : α → Game β)
    (cont : (a : α) → Strategy (f a) → Play n N) (hc : ∀ a t, (cont a t).Laws S) :
    ∀ (g : Game α) (s : Strategy (g.bind f)), (ofBindPlay f cont g s).Laws S
  | .ret a, s => hc a s
  | .send _ k, s => laws_ofBindPlay f cont hc (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => fun c => laws_ofBindPlay f cont hc (k c) (s c)

theorem drawn_ofBindPlay {α β : Type} {N : ℕ} {err : ℕ → ℝ≥0∞} (f : α → Game β)
    (cont : (a : α) → Strategy (f a) → Play n N) (hc : ∀ a t, (cont a t).Drawn err) :
    ∀ (g : Game α) (s : Strategy (g.bind f)), (ofBindPlay f cont g s).Drawn err
  | .ret a, s => hc a s
  | .send _ k, s => drawn_ofBindPlay f cont hc (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => fun c => drawn_ofBindPlay f cont hc (k c) (s c)

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

theorem laws_contPlay {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N} {S : Law n → Prop}
    (hrest : ∀ t, (rest t).Laws S) :
    ∀ (b : Bool) (t : Strategy (cond b next (.ret false))), (contPlay rest b t).Laws S
  | true, t => hrest t
  | false, _ => trivial

theorem drawn_contPlay {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N} {err : ℕ → ℝ≥0∞}
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

variable {L : Law n} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

/-- The run tree of one audit and what follows it: a round over the committed transcript's wrong
units. -/
noncomputable def roundPlay (A : Analysis L Reg session) {next : Game Bool} {N : ℕ}
    (rest : Strategy next → Play n N) (t : Strategy ((audit L Reg session).bind (afterAudit next))) :
    Play n (N + 1) :=
  .round L (A.wrong (A.committedOf (Strategy.ofBind (afterAudit next) (audit L Reg session) t))) fun ω =>
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
          (x + ∑ u ∈ A.wrong (A.committedOf (Strategy.ofBind (afterAudit next) (audit L Reg session) t)), h u) :=
  rfl

/-- One round's drawn-unit term is the audit's drawn-unit event. -/
theorem drawn_term {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N}
    (hpass : ∀ t, (rest t).pass = 1) (B D : Finset (Fin n)) (g : Game Bool)
    (s : Strategy ((g.map fun ok => (ok, D)).bind (afterAudit next))) :
    (if Disjoint D B then 0 else
        (ofBindPlay (afterAudit next) (fun o => contPlay rest o.1) (g.map fun ok => (ok, D)) s).pass) =
      prob (fun o => o.1 = true ∧ ¬ Disjoint B o.2) (g.map fun ok => (ok, D))
        (Strategy.ofBind (afterAudit next) (g.map fun ok => (ok, D)) s) := by
  rw [prob_map]
  by_cases hd : Disjoint D B
  · rw [if_pos hd]
    exact le_antisymm zero_le
      ((prob_mono (q := fun _ => False) (fun _ ho => ho.2 hd.symm) _ _).trans (prob_false _ _).le)
  · rw [if_neg hd, pass_ofBindPlay (afterAudit next) (fun o => contPlay rest o.1) (fun o => o.1 = true)
      (pass_contPlay hpass), prob_map]
    exact le_antisymm (prob_mono (fun _ ho => ⟨ho, fun h' => hd h'.symm⟩) _ _) (prob_mono (fun _ ho => ho.1) _ _)

theorem drawn_roundPlay (A : Analysis L Reg session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N}
    {err : ℕ → ℝ≥0∞} (hpass : ∀ t, (rest t).pass = 1) (hrest : ∀ t, (rest t).Drawn err)
    (hN : εks + δlink ≤ err N) (t : Strategy ((audit L Reg session).bind (afterAudit next))) :
    (roundPlay A rest t).Drawn err := by
  refine ⟨?_, fun ω => ?_⟩
  · refine le_trans (le_of_eq ?_)
      ((audit_drawn A hks hlink (Strategy.ofBind (afterAudit next) (audit L Reg session) t)).trans hN)
    refine Eq.trans ?_ (prob_audit_bind _ t).symm
    exact congrArg avg (funext fun ω => drawn_term hpass _ (L.draw ω) (session (L.draw ω) t.1) (t.2 ω))
  · exact drawn_ofBindPlay (afterAudit next) (fun o => contPlay rest o.1)
      (fun o t' => drawn_contPlay hrest o.1 t') _ (t.2 ω)

theorem laws_roundPlay (A : Analysis L Reg session) {S : Law n → Prop} (hL : S L) {next : Game Bool} {N : ℕ}
    {rest : Strategy next → Play n N} (hrest : ∀ t, (rest t).Laws S)
    (t : Strategy ((audit L Reg session).bind (afterAudit next))) : (roundPlay A rest t).Laws S :=
  ⟨hL, fun ω => laws_ofBindPlay (afterAudit next) (fun o => contPlay rest o.1)
    (fun o t' => laws_contPlay hrest o.1 t') _ (t.2 ω)⟩

theorem accept_roundPlay (A : Analysis L Reg session) (h : Fin n → ℝ) {next : Game Bool} {N : ℕ}
    {rest : Strategy next → Play n N}
    (hrest : ∀ t x, (rest t).prEv h (fun acc _ => acc = true) x = prob (· = true) next t)
    (t : Strategy ((audit L Reg session).bind (afterAudit next))) (x : ℝ) :
    (roundPlay A rest t).prEv h (fun acc _ => acc = true) x =
      prob (· = true) ((audit L Reg session).bind (afterAudit next)) t := by
  refine (prEv_roundPlay A h _ rest t x).trans (Eq.trans ?_ (prob_audit_bind' (· = true) t).symm)
  exact congrArg avg (funext fun ω => prEv_ofBindPlay h _ _ (afterAudit next) (fun o => contPlay rest o.1)
    (· = true) (fun o t' => accept_contPlay h hrest o.1 t' _) _ (t.2 ω))

theorem laws_goPlay (A : Analysis L Reg session) {S : Law n → Prop} (hL : S L) {next : Game Bool} {N : ℕ}
    {rest : Strategy next → Play n N} (hrest : ∀ t, (rest t).Laws S) :
    ∀ (go : Bool) (t : Strategy (cond go ((audit L Reg session).bind (afterAudit next)) (.ret true))),
      (goPlay A rest go t).Laws S
  | true, t => laws_roundPlay A hL hrest t
  | false, _ => trivial

theorem drawn_goPlay (A : Analysis L Reg session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N}
    {err : ℕ → ℝ≥0∞} (hpass : ∀ t, (rest t).pass = 1) (hrest : ∀ t, (rest t).Drawn err)
    (hN : εks + δlink ≤ err N) :
    ∀ (go : Bool) (t : Strategy (cond go ((audit L Reg session).bind (afterAudit next)) (.ret true))),
      (goPlay A rest go t).Drawn err
  | true, t => drawn_roundPlay A hks hlink hpass hrest hN t
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

theorem laws_stagePlay (A : Analysis L Reg session) {S : Law n → Prop} (hL : S L) {next : Game Bool} {N : ℕ}
    {rest : Strategy next → Play n N} (hrest : ∀ t, (rest t).Laws S) (s : Strategy (stage L Reg session next)) :
    (stagePlay A rest s).Laws S :=
  laws_goPlay A hL hrest s.1 s.2

theorem drawn_stagePlay (A : Analysis L Reg session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) {next : Game Bool} {N : ℕ} {rest : Strategy next → Play n N}
    {err : ℕ → ℝ≥0∞} (hpass : ∀ t, (rest t).pass = 1) (hrest : ∀ t, (rest t).Drawn err)
    (hN : εks + δlink ≤ err N) (s : Strategy (stage L Reg session next)) : (stagePlay A rest s).Drawn err :=
  drawn_goPlay A hks hlink hpass hrest hN s.1 s.2

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

variable {L : ℕ → Law n} {Reg : ℕ → Type} {session : (m : ℕ) → Finset (Fin n) → Reg m → Game Bool}

/-- **The run tree of a lifetime** under a prover strategy, round `m`'s wrong units those of its
committed transcript under the analysis `A m`. -/
noncomputable def lifePlay (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) :
    (N : ℕ) → Strategy (life L Reg session N) → Play n N
  | 0, _ => .stop true
  | N + 1, s => stagePlay (A N) (lifePlay A N) s

theorem laws_lifePlay (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) {S : Law n → Prop}
    (hL : ∀ m, S (L m)) : ∀ (N : ℕ) (s : Strategy (life L Reg session N)), (lifePlay A N s).Laws S
  | 0, _ => trivial
  | N + 1, s => laws_stagePlay (A N) (hL N) (laws_lifePlay A hL N) s

theorem pass_lifePlay (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) :
    ∀ (N : ℕ) (s : Strategy (life L Reg session N)), (lifePlay A N s).pass = 1
  | 0, _ => by
      show (Play.stop true : Play n 0).pass = 1
      simp [pass_stop]
  | N + 1, s => pass_stagePlay (A N) (lifePlay A N) s

theorem drawn_lifePlay (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) {εks δlink : ℕ → ℝ≥0∞}
    (hks : ∀ m, (A m).KnowledgeSound (εks m)) (hlink : ∀ m, (A m).LinkSound (δlink m)) :
    ∀ (N : ℕ) (s : Strategy (life L Reg session N)), (lifePlay A N s).Drawn fun m => εks m + δlink m
  | 0, _ => trivial
  | N + 1, s => drawn_stagePlay (A N) (hks N) (hlink N) (pass_lifePlay A N) (drawn_lifePlay A hks hlink N) le_rfl s

/-- **The run tree is faithful**: it accepts exactly when the lifetime does. -/
theorem life_accept (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) (h : Fin n → ℝ) :
    ∀ (N : ℕ) (s : Strategy (life L Reg session N)) (x : ℝ),
      (lifePlay A N s).prEv h (fun acc _ => acc = true) x = prob (· = true) (life L Reg session N) s
  | 0, _, _ => rfl
  | N + 1, s, x => accept_stagePlay (A N) h (life_accept A h N) s x

/-- **The cumulative harm budget over a lifetime of one-stage audits.** For laws meeting the design
condition with weights `π`, harms `h` with `λ h ≤ π` and `λ ≥ 0`, and every prover strategy, the
lifetime accepts with total harm `Σ_t h(wrong(X_t))` at least `H` with probability at most
`e^{−λH} + Σ_{m<N} (ε_ks,m + δ_link,m)`. -/
theorem life_harm_budget (A : (m : ℕ) → Analysis (L m) (Reg m) (session m)) {εks δlink : ℕ → ℝ≥0∞}
    (hks : ∀ m, (A m).KnowledgeSound (εks m)) (hlink : ∀ m, (A m).LinkSound (δlink m))
    {π h : Fin n → ℝ} (hL : ∀ m, (L m).Design π) {lam : ℝ} (hlam0 : 0 ≤ lam) (hlam : ∀ u, lam * h u ≤ π u)
    (H : ℝ) (N : ℕ) (s : Strategy (life L Reg session N)) :
    (lifePlay A N s).prEv h (fun acc x => acc = true ∧ H ≤ x) 0 ≤
      ENNReal.ofReal (Real.exp (-(lam * H))) + ∑ m ∈ range N, (εks m + δlink m) :=
  harm_budget hlam0 hlam H (lifePlay A N s) (laws_lifePlay A hL N s) (drawn_lifePlay A hks hlink N s)

end Life

end FlockSoundness.Audit.Adaptive
