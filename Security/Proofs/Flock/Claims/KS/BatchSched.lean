import Proofs.Flock.Claims.KS.BatchSteps

/-!
# A batch's schedule read position by position

`PreSched` asks, of a strategy of a whole batch, that each table in front advance by `FirstS` and then wait by
`FollowR` against some strategy of the rest, and of the session's table that it follow some strategy by `FollowL`. The
simulator moves one table at a time, so the schedule is stated here position by position, with the followed strategies
read off the batch's by `projR` and `projL`, which leaves nothing to choose along the way:

* `WaitR`, `WaitR.followR`: a table waits at a send while the rest moves, so its rest follows `projR`;
* `FollowLP`, `FollowLP.followL`: the session's table moves, the rest waits, so the table follows `projL`;
* `Paused`, `Front`, `Sess`: the tables already paused, the tables in front still advancing, and the session's table
  leading with rep 0 (`Lock.Lead`);
* `preSched_of_front`: so the schedule `PreSched` with `TabFollowC`'s shape;
* `paused_front_zero`: a table in front whose coins are spent pauses;
* `DepAt`, `paused_front_dead`, `paused_sess_dead`: below a part of the run every outcome of which is `Dead`, where
  `Dead` reads one table, every position holds;
* `paused_sess_spent`: once the slot is behind and rep 0 has ended, every strategy holds.
-/

namespace FlockSoundness.Game.Lock

section Wait

variable {γ β : Type}

/-- **The game in front waits at a send while the rest moves**, one coin off the budget `b` at each of the rest's
coins, except below a part of the run every reached outcome of which is `Dead` (`dead`), and with any moves once the
budget is spent (`spent`). -/
inductive WaitR (Dead : γ × β → Prop) :
    Option ℕ → (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Prop
  | dead {b : Option ℕ} {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} :
      (∀ x, Reach _ σ x → Dead x) → WaitR Dead b h g' σ
  | spent {M : Type} {k : M → Game γ} {g' : Game β} {σ : Strategy (interleave (.send M k) g')} :
      WaitR Dead none (.send M k) g' σ
  | icoin {b : Option ℕ} {M : Type} {k : M → Game γ} {C₂ : Type} [Fintype C₂] [Nonempty C₂] {k₂ : C₂ → Game β}
      {σ : (c : C₂) → Strategy (interleave (.send M k) (k₂ c))} :
      (∀ c, WaitR Dead (predS b) (.send M k) (k₂ c) (σ c)) → WaitR Dead b (.send M k) (.coin C₂ k₂) ⟨false, σ⟩
  | isend {b : Option ℕ} {M : Type} {k : M → Game γ} {M₂ : Type} {k₂ : M₂ → Game β} (m₂ : M₂)
      {σ : Strategy (interleave (.send M k) (k₂ m₂))} :
      WaitR Dead b (.send M k) (k₂ m₂) σ → WaitR Dead b (.send M k) (.send M₂ k₂) ⟨false, ⟨m₂, σ⟩⟩

/-- **A waiting game's rest follows its `projR`.** -/
theorem WaitR.followR {Dead : γ × β → Prop} {Dead' : β → Prop} (hD : ∀ x : γ × β, Dead' x.2 → Dead x)
    {b : Option ℕ} {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} (hW : WaitR Dead b h g' σ) :
    FollowR Dead Dead' b h g' σ (projR Dead' h g' σ) := by
  induction hW with
  | dead hσ => exact .dead hσ
  | spent => exact followR_none hD _ _ _
  | icoin _ ih => exact .icoinS ih
  | isend m₂ _ ih => exact .isendS m₂ ih

end Wait

section Lead

variable {β γ : Type}

/-- **The game moves while the rest waits**, one coin off the budget `b` at each of its coins, until it has ended
(`gret`) or the rest has (`hret`), except below a part of the run every reached outcome of which is `Dead` (`dead`),
and with any moves once the budget is spent (`spent`). -/
inductive FollowLP (Dead : β × γ → Prop) :
    Option ℕ → (g' : Game β) → (h : Game γ) → Strategy (interleave g' h) → Prop
  | dead {b : Option ℕ} {g' : Game β} {h : Game γ} {σ : Strategy (interleave g' h)} :
      (∀ x, Reach _ σ x → Dead x) → FollowLP Dead b g' h σ
  | spent {g' : Game β} {h : Game γ} {σ : Strategy (interleave g' h)} : FollowLP Dead none g' h σ
  | gret {b : Option ℕ} {a : β} {h : Game γ} {σ : Strategy (interleave (.ret a) h)} : FollowLP Dead b (.ret a) h σ
  | hret {b : Option ℕ} {g' : Game β} {c : γ} {σ : Strategy (interleave g' (.ret c))} : FollowLP Dead b g' (.ret c) σ
  | gcoinS {b : Option ℕ} {C : Type} [Fintype C] [Nonempty C] {k : C → Game β} {M₂ : Type} {k₂ : M₂ → Game γ}
      {σ : (c : C) → Strategy (interleave (k c) (.send M₂ k₂))} :
      (∀ c, FollowLP Dead (predS b) (k c) (.send M₂ k₂) (σ c)) → FollowLP Dead b (.coin C k) (.send M₂ k₂) ⟨true, σ⟩
  | gcoinC {b : Option ℕ} {C : Type} [Fintype C] [Nonempty C] {k : C → Game β} {C₂ : Type} [Fintype C₂]
      [Nonempty C₂] {k₂ : C₂ → Game γ} {σ : (c : C) → Strategy (interleave (k c) (.coin C₂ k₂))} :
      (∀ c, FollowLP Dead (predS b) (k c) (.coin C₂ k₂) (σ c)) → FollowLP Dead b (.coin C k) (.coin C₂ k₂) ⟨true, σ⟩
  | gsendS {b : Option ℕ} {M : Type} {k : M → Game β} {M₂ : Type} {k₂ : M₂ → Game γ} (m : M)
      {σ : Strategy (interleave (k m) (.send M₂ k₂))} :
      FollowLP Dead b (k m) (.send M₂ k₂) σ → FollowLP Dead b (.send M k) (.send M₂ k₂) ⟨true, ⟨m, σ⟩⟩
  | gsendC {b : Option ℕ} {M : Type} {k : M → Game β} {C₂ : Type} [Fintype C₂] [Nonempty C₂] {k₂ : C₂ → Game γ}
      (m : M) {σ : Strategy (interleave (k m) (.coin C₂ k₂))} :
      FollowLP Dead b (k m) (.coin C₂ k₂) σ → FollowLP Dead b (.send M k) (.coin C₂ k₂) ⟨true, ⟨m, σ⟩⟩

/-- **A moving game follows its `projL`.** -/
theorem FollowLP.followL {Dead : β × γ → Prop} {Dead' : β → Prop} (hD : ∀ x : β × γ, Dead' x.1 → Dead x)
    {b : Option ℕ} {g' : Game β} {h : Game γ} {σ : Strategy (interleave g' h)} (hF : FollowLP Dead b g' h σ) :
    FollowL Dead Dead' b g' h σ (projL Dead' g' h σ) := by
  induction hF with
  | dead hσ => exact .dead hσ
  | spent => exact followL_none hD _ _ _
  | gret => exact .gret
  | @hret b g' c σ =>
      rw [FlockDeploy.KS.projL_ret Dead' c g' σ]
      exact .hret σ
  | gcoinS _ ih => exact .gcoinS ih
  | gcoinC _ ih => exact .gcoinC ih
  | gsendS m _ ih => exact .gsendS m ih
  | gsendC m _ ih => exact .gsendC m ih

end Lead

theorem FirstS.imp {γ β : Type} {Dead : γ × β → Prop}
    {Q Q' : (h : Game γ) → (g' : Game β) → Strategy (interleave h g') → Prop} (hQ : ∀ h g' σ, Q h g' σ → Q' h g' σ)
    {n : ℕ} {h : Game γ} {g' : Game β} {σ : Strategy (interleave h g')} (hS : FirstS Dead Q n h g' σ) :
    FirstS Dead Q' n h g' σ := by
  induction hS with
  | dead hσ => exact .dead hσ
  | stop hq => exact .stop (hQ _ _ _ hq)
  | last => exact .last
  | coinCoin _ ih => exact .coinCoin ih
  | coinSend _ ih => exact .coinSend ih
  | sendCoin m _ ih => exact .sendCoin m ih
  | sendSend m _ ih => exact .sendSend m ih

section Batch

variable {ζ : Type}

/-- The pairing of a batch's first output with the rest's. -/
abbrev consP : ζ × List ζ → List ζ := fun x => x.1 :: x.2

/-- A batch's strategy read at its first game and the rest's batch. -/
noncomputable abbrev headS (h : Game ζ) (l : List (Game ζ)) (σ : Strategy (batch (h :: l))) :
    Strategy (interleave h (batch l)) :=
  Strategy.ofMap consP (h.interleave (batch l)) σ

/-- `Dead` of the rest, every first output given. -/
abbrev deadT (Dead : List ζ → Prop) : List ζ → Prop := fun outs => ∀ c, Dead (c :: outs)

/-- **The first `np` games are paused**: each waits by `WaitR` while the rest moves, with the budget `b`, and `Inner`
holds at the rest's strategy read off past them (`projR`). -/
def Paused (b : Option ℕ) (Inner : (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop) :
    ℕ → (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop
  | 0, Dead, l, σ => Inner Dead l σ
  | _ + 1, _, [], _ => False
  | np + 1, Dead, h :: l, σ =>
      WaitR (fun x => Dead (consP x)) b h (batch l) (headS h l σ) ∧
        Paused b Inner np (deadT Dead) l (projR (deadT Dead) h (batch l) (headS h l σ))

/-- **The games in front still advance**: the first by `FirstS` for its `n₀` coins, then pausing by `WaitR` with the
budget of the coins still to go up to a slot `k` coins past where they all pause, and the rest's by `Front` at the
rest's strategy read off past it; `Sess` holds where they all pause. -/
def Front (k : ℕ) (Sess : (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop) :
    List ℕ → (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop
  | [], Dead, l, σ => Sess Dead l σ
  | _ :: _, _, [], _ => False
  | n₀ :: ns, Dead, h :: l, σ =>
      FirstS (fun x => Dead (consP x))
        (fun h' g'' σ'' => ∃ e : g'' = batch l, WaitR (fun x => Dead (consP x)) (some (ns.sum + k)) h' g'' σ'' ∧
          Front k Sess ns (deadT Dead) l (e ▸ projR (deadT Dead) h' g'' σ''))
        n₀ h (batch l) (headS h l σ)

/-- **The session's table follows its own strategy, rep 0 first**: it moves by `FollowLP` with the budget `b` while
the tables after it wait, and its strategy read off the batch's (`projL`) leads with rep 0 for `nL` coins
(`Lock.Lead`), the table being the reps `g₁`, `g₂` and then `f`. -/
def Sess {α₁ β₁ : Type} (b : Option ℕ) (nL : ℕ) (f : α₁ × β₁ → Game ζ) (g₁ : Game α₁) (g₂ : Game β₁) :
    (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop
  | _, [], _ => False
  | Dead, X :: B, σ =>
      FollowLP (fun x => Dead (consP x)) b X (batch B) (headS X B σ) ∧
        ∀ e : X = (interleave g₁ g₂).bind f, Lead f (fun x => ∀ rest, Dead (x :: rest)) nL g₁ g₂
          (cast (congrArg Strategy e) (projL (fun x => ∀ rest, Dead (x :: rest)) X (batch B) (headS X B σ)))

/-- **The schedule from the games in front**: `Front` over the tables in front, then `Sess`, is `PreSched` with every
paused table's rest read off by `projR`, after the sum of the tables in front's coins. -/
theorem preSched_of_front (X : Game ζ) (post : List (Game ζ)) (k : ℕ)
    {Sess : (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop}
    {Q : (List ζ → Prop) → Strategy (batch (X :: post)) → Prop} (hQ : ∀ Dead σ, Sess Dead (X :: post) σ → Q Dead σ) :
    ∀ (pre : List (Game ζ)) (ns : List ℕ), ns.length = pre.length →
      ∀ {Dead : List ζ → Prop} {σ : Strategy (batch (pre ++ X :: post))},
        Front k Sess ns Dead (pre ++ X :: post) σ → PreSched X post k pre Dead Q ns.sum σ
  | [], [], _, Dead, σ, hF => ⟨rfl, hQ Dead σ hF⟩
  | [], _ :: _, hl, _, _, _ => absurd hl (by simp)
  | _ :: _, [], hl, _, _, _ => absurd hl (by simp)
  | _ :: pre, n₀ :: ns, hl, _, _, hF =>
      ⟨n₀, ns.sum, List.sum_cons, hF.imp fun _ _ _ ⟨e, hW, hF'⟩ =>
        ⟨e, _, hW.followR fun x hx => hx x.1,
          preSched_of_front X post k hQ pre ns (by simpa using hl) hF'⟩⟩

/-- **A table in front whose coins are spent pauses**, where the budget is the coins still to go. -/
theorem paused_front_zero (k : ℕ) (Sess : (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop)
    (ns : List ℕ) : ∀ (np : ℕ) {Dead : List ζ → Prop} {l : List (Game ζ)} {σ : Strategy (batch l)},
      Paused (some (ns.sum + k)) (Front k Sess ns) (np + 1) Dead l σ →
        Paused (some (ns.sum + k)) (Front k Sess (0 :: ns)) np Dead l σ
  | _, _, [], _, h => h.elim
  | 0, _, _ :: _, _, ⟨hW, hP⟩ => .stop ⟨rfl, hW, hP⟩
  | np + 1, _, _ :: _, _, ⟨hW, hP⟩ => ⟨hW, paused_front_zero k Sess ns np hP⟩

/-! ## Below dead runs, and past the slot -/

/-- **`Dead` reads the batch's output only at position `m`.** -/
def DepAt (m : ℕ) (Dead : List ζ → Prop) : Prop := ∀ l l' : List ζ, l[m]? = l'[m]? → Dead l → Dead l'

theorem DepAt.tail {m : ℕ} {Dead : List ζ → Prop} (h : DepAt (m + 1) Dead) : DepAt m (deadT Dead) :=
  fun l l' e hd c => h (c :: l) (c :: l') (by simpa using e) (hd c)

theorem DepAt.cons {m : ℕ} {Dead : List ζ → Prop} (h : DepAt (m + 1) Dead) {c : ζ} {l : List ζ} (hd : Dead (c :: l))
    (c' : ζ) : Dead (c' :: l) :=
  h (c :: l) (c' :: l) (by simp) hd

theorem DepAt.head {Dead : List ζ → Prop} (h : DepAt 0 Dead) {c : ζ} {l : List ζ} (hd : Dead (c :: l))
    (l' : List ζ) : Dead (c :: l') :=
  h (c :: l) (c :: l') (by simp) hd


theorem reach_headS (h : Game ζ) (l : List (Game ζ)) (σ : Strategy (batch (h :: l))) {x : ζ × List ζ}
    (hx : Reach _ (headS h l σ) x) : Reach (batch (h :: l)) σ (consP x) :=
  reach_ofMap_map consP _ σ x hx

/-- **What the rest's strategy read off past the first game reaches**, the whole strategy reaches with some first
output. -/
theorem reach_projR {γ β : Type} (D' : β → Prop) :
    ∀ (h : Game γ) (g' : Game β) (σ : Strategy (interleave h g')) {y : β},
      Reach g' (projR D' h g' σ) y → ∃ x, Reach _ σ x ∧ x.2 = y
  | .ret a, g', σ, y, hy => ⟨(a, y), reach_ofMap_map (a, ·) g' σ y hy, rfl⟩
  | .send M k, g', σ, y, hy => reachWith (adv := fun g₂ => .send M fun m => interleave (k m) g₂)
      (fin := fun b => (Game.send M k).map fun a => (a, b))
      (fun g₂ σ' hy => reach_projR D' (k σ'.1) g₂ σ'.2 hy)
      (fun b σ' hy => by
        obtain ⟨x, hx⟩ := reach_exists _ σ'
        exact ⟨x, hx, (fin_reach (.send M k) b σ' x hx).trans hy⟩) g' σ hy
  | @Game.coin _ C _ iN k, g', σ, y, hy => reachWith (adv := fun g₂ => @Game.coin _ C _ iN fun c => interleave (k c) g₂)
      (fin := fun b => (@Game.coin _ C _ iN k).map fun a => (a, b))
      (fun g₂ σ' hy => by
        obtain ⟨x, hx, e⟩ := reach_projR D' (k _) g₂ (σ' _) hy
        exact ⟨x, ⟨_, hx⟩, e⟩)
      (fun b σ' hy => by
        obtain ⟨x, hx⟩ := reach_exists _ σ'
        exact ⟨x, hx, (fin_reach (@Game.coin _ C _ iN k) b σ' x hx).trans hy⟩) g' σ hy
where
  reachWith {adv : Game β → Game (γ × β)} {fin : β → Game (γ × β)}
      {advR : (g' : Game β) → Strategy (adv g') → Strategy g'} {y : β}
      (hadv : ∀ g₂ σ', Reach g₂ (advR g₂ σ') y → ∃ x, Reach (adv g₂) σ' x ∧ x.2 = y)
      (hfin : ∀ b σ', b = y → ∃ x, Reach (fin b) σ' x ∧ x.2 = y) :
      ∀ (g' : Game β) (σ : Strategy (interleaveWith adv fin g')), Reach g' (projRWith advR g' σ) y →
        ∃ x, Reach _ σ x ∧ x.2 = y
    | .ret b, σ, hy => hfin b σ hy
    | .send _ _, ⟨true, σ'⟩, hy => hadv _ σ' hy
    | .send _ k₂, ⟨false, σ'⟩, hy => reachWith hadv hfin (k₂ σ'.1) σ'.2 hy
    | @Game.coin _ _ _ _ _, ⟨true, σ'⟩, hy => hadv _ σ' hy
    | @Game.coin _ _ _ _ k₂, ⟨false, σ'⟩, ⟨c, hy⟩ => by
        obtain ⟨x, hx, e⟩ := reachWith hadv hfin (k₂ c) (σ' c) hy
        exact ⟨x, ⟨c, hx⟩, e⟩

/-- **Below dead runs the session's table holds**, where `Dead` reads it alone. -/
theorem sess_dead {α₁ β₁ : Type} {b : Option ℕ} {nL : ℕ} {f : α₁ × β₁ → Game ζ} {g₁ : Game α₁} {g₂ : Game β₁}
    {Dead : List ζ → Prop} (hd : DepAt 0 Dead) {X : Game ζ} {B : List (Game ζ)} {σ : Strategy (batch (X :: B))}
    (hσ : ∀ x, Reach _ σ x → Dead x) : Sess b nL f g₁ g₂ Dead (X :: B) σ := by
  refine ⟨.dead fun x hx => hσ _ (reach_headS X B σ hx), fun e => .dead fun c hc rest => ?_⟩
  subst e
  obtain ⟨x, hx, rfl⟩ := reach_projL _ (batch B) _ hc
  exact hd.head (hσ _ (reach_headS _ B σ hx)) rest

/-- **Below dead runs the games in front hold**, where `Dead` reads the session's table alone. -/
theorem front_dead (k : ℕ) {Sess : (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop}
    (hS : ∀ Dead l σ, DepAt 0 Dead → l ≠ [] → (∀ x, Reach (batch l) σ x → Dead x) → Sess Dead l σ) :
    ∀ (ns : List ℕ) {Dead : List ζ → Prop} {l : List (Game ζ)} {σ : Strategy (batch l)}, DepAt ns.length Dead →
      ns.length < l.length → (∀ x, Reach (batch l) σ x → Dead x) → Front k Sess ns Dead l σ
  | [], _, l, _, hd, hl, hσ => hS _ _ _ hd (List.ne_nil_of_length_pos hl) hσ
  | _ :: _, _, [], _, _, hl, _ => absurd hl (by simp)
  | _ :: _, _, h :: l, σ, _, _, hσ => .dead fun x hx => hσ _ (reach_headS h l σ hx)

/-- **Below dead runs the paused games and `Inner` hold**, where `Dead` reads a table past them alone. -/
theorem paused_dead {b : Option ℕ} {Inner : (List ζ → Prop) → (l : List (Game ζ)) → Strategy (batch l) → Prop}
    {m : ℕ} (hI : ∀ Dead l σ, DepAt m Dead → m < l.length → (∀ x, Reach (batch l) σ x → Dead x) → Inner Dead l σ) :
    ∀ (np : ℕ) {Dead : List ζ → Prop} {l : List (Game ζ)} {σ : Strategy (batch l)}, DepAt (np + m) Dead →
      np + m < l.length → (∀ x, Reach (batch l) σ x → Dead x) → Paused b Inner np Dead l σ
  | 0, _, _, _, hd, hl, hσ => hI _ _ _ (by simpa using hd) (by simpa using hl) hσ
  | _ + 1, _, [], _, _, hl, _ => absurd hl (by simp)
  | np + 1, _, h :: l, σ, hd, hl, hσ =>
      ⟨.dead fun x hx => hσ _ (reach_headS h l σ hx),
        paused_dead hI np (DepAt.tail (by rwa [Nat.add_right_comm] at hd)) (by simp at hl; omega) fun y hy => by
          obtain ⟨x, hx, rfl⟩ := reach_projR _ h (batch l) _ hy
          exact (DepAt.cons (by rwa [Nat.add_right_comm] at hd) (hσ _ (reach_headS h l σ hx)))⟩

/-- **Past the slot, with rep 0 ended, every strategy holds** of the session's table, and of the paused tables that
are at a send. -/
theorem paused_sess_spent {α₁ β₁ : Type} {f : α₁ × β₁ → Game ζ} {a : α₁} {g₂ : Game β₁} :
    ∀ (np : ℕ) {Dead : List ζ → Prop} {l : List (Game ζ)} (σ : Strategy (batch l)),
      (∀ i (_ : i < np) (hl : i < l.length), ∃ (M : Type) (k : M → Game ζ), l[i] = .send M k) → np < l.length →
        Paused none (Sess none 0 f (.ret a) g₂) np Dead l σ
  | _, _, [], _, _, hl => absurd hl (by simp)
  | 0, _, _ :: _, _, _, _ => ⟨.spent, fun _ => .done⟩
  | np + 1, _, h :: l, σ, hs, hl => by
      obtain ⟨M, k, rfl⟩ := hs 0 (by omega) (by simp)
      exact ⟨.spent, paused_sess_spent np _ (fun i hi hl' => hs (i + 1) (by omega) (by simpa using hl'))
        (by simpa using hl)⟩

end Batch

end FlockSoundness.Game.Lock
