import FlockSoundness.Discharge.Exec.SessionL
import FlockSoundness.Audit.OneStage
import FlockSoundness.Game.Union

/-!
# Weak completeness of the salted batched session

A game is decisive (`Decisive`) when every message type it asks for is inhabited (`Productive`) and two of its runs,
for some messages and some coins (`Reaches`), end differently. These are PoUW's `Pouw.Draft.Game.Decisive`,
`Productive` and `Reaches`, restated over `FlockSoundness.Game`.

The salted batched session `batchedSessionL` is decisive on every plan with a table, and so is the one-stage audit
over it (`audit_tile_decisive`, for the one-table plans of a tiling). No table needs to be satisfiable and no schedule
needs a side condition:

* **acceptance** (`batchedSessionL_reaches_true`): the all-zero run accepts every table. With zero messages and zero
  coins every running claim stays `0` (zerocheck, lincheck, every Ligerito fold, out-of-domain sample and consistency
  batch), except the region claims, whose ring-switch message puts the claim's value on the packed position where the
  claim's weight is `1` (at zero link points the region points are `0/1`). A zero residual `yr` makes the final sum
  `0`. `finalCheck`'s length conjunct `orig.length + ext = m − 7` is `Schedule.WellShaped`'s shape identity, which
  `TabSpec.hsch` gives through `fast100`. Every level's rows are zero rows under the default salt, so one value per
  level reaches every cap entry, whatever the positions;
* **rejection** (`batchedSessionL_reaches_false`): a rep whose zerocheck sends `final_a_eval = final_b_eval = 1` on
  the zero claim rejects, and with it the table and the session;
* **productivity** (`batchedSessionL_productive`): every message type of the session is inhabited.
-/

namespace FlockSoundness.Discharge.Exec.SessionLComplete

open Game Model Accounting Audit

section Defs

variable {α β : Type}

/-- Every message type `g` asks for is inhabited (coin types are, by `Game.coin`'s `Nonempty`). -/
def Productive : Game α → Prop
  | .ret _ => True
  | .send M k => Nonempty M ∧ ∀ m, Productive (k m)
  | @Game.coin _ _ _ _ k => ∀ c, Productive (k c)

/-- Some run of `g`, for some messages and some coins, ends with `a`. -/
def Reaches : Game α → α → Prop
  | .ret b, a => b = a
  | .send _ k, a => ∃ m, Reaches (k m) a
  | @Game.coin _ _ _ _ k, a => ∃ c, Reaches (k c) a

/-- **Decisive**: every message type of `g` is inhabited, and two runs of `g` end with different outputs. -/
def Decisive (g : Game α) : Prop :=
  Productive g ∧ ∃ a b, a ≠ b ∧ Reaches g a ∧ Reaches g b

theorem reaches_bind {f : α → Game β} {b : β} :
    ∀ (g : Game α) (a : α), Reaches g a → Reaches (f a) b → Reaches (g.bind f) b
  | .ret _, _, h, hf => by cases h; exact hf
  | .send _ k, a, ⟨m, h⟩, hf => ⟨m, reaches_bind (k m) a h hf⟩
  | @Game.coin _ _ _ _ k, a, ⟨c, h⟩, hf => ⟨c, reaches_bind (k c) a h hf⟩

theorem reaches_map (f : α → β) {g : Game α} {a : α} (h : Reaches g a) : Reaches (g.map f) (f a) :=
  reaches_bind (f := fun a => .ret (f a)) g a h rfl

theorem productive_bind {f : α → Game β} (hf : ∀ a, Productive (f a)) :
    ∀ g : Game α, Productive g → Productive (g.bind f)
  | .ret a, _ => hf a
  | .send _ k, ⟨hM, h⟩ => ⟨hM, fun m => productive_bind hf (k m) (h m)⟩
  | @Game.coin _ _ _ _ k, h => fun c => productive_bind hf (k c) (h c)

theorem productive_map (f : α → β) {g : Game α} (h : Productive g) : Productive (g.map f) :=
  productive_bind (f := fun a => .ret (f a)) (fun _ => trivial) g h

end Defs

/-! ## Runs and productivity, step by step -/

section Toolkit

variable {α β : Type}

theorem reaches_send_iff {M : Type} {k : M → Game α} {a : α} :
    Reaches (Game.send M k) a ↔ ∃ m, Reaches (k m) a := Iff.rfl

theorem reaches_coin_iff {C : Type} [Fintype C] [Nonempty C] {k : C → Game α} {a : α} :
    Reaches (Game.coin C k) a ↔ ∃ c, Reaches (k c) a := Iff.rfl

theorem productive_send_iff {M : Type} {k : M → Game α} :
    Productive (Game.send M k) ↔ Nonempty M ∧ ∀ m, Productive (k m) := Iff.rfl

theorem productive_coin_iff {C : Type} [Fintype C] [Nonempty C] {k : C → Game α} :
    Productive (Game.coin C k) ↔ ∀ c, Productive (k c) := Iff.rfl

theorem reaches_map_of {f : α → β} {g : Game α} {a : α} {b : β} (h : Reaches g a) (hb : f a = b) :
    Reaches (g.map f) b :=
  hb ▸ reaches_map f h

/-- A run of `g >>= f`, from a run of `g` and, for every output with `P`, a run of `f`. -/
theorem rx_bind {g : Game α} {f : α → Game β} {P : α → Prop} {Q : β → Prop}
    (hg : ∃ a, Reaches g a ∧ P a) (hf : ∀ a, P a → ∃ b, Reaches (f a) b ∧ Q b) :
    ∃ b, Reaches (g >>= f) b ∧ Q b := by
  obtain ⟨a, ha, hp⟩ := hg
  obtain ⟨b, hb, hq⟩ := hf a hp
  exact ⟨b, reaches_bind g a ha hb, hq⟩

theorem rx_recv {M : Type} {f : M → Game β} {Q : β → Prop} (m : M) (h : ∃ b, Reaches (f m) b ∧ Q b) :
    ∃ b, Reaches (recv M >>= f) b ∧ Q b :=
  rx_bind (P := (· = m)) ⟨m, reaches_send_iff.2 ⟨m, rfl⟩, rfl⟩ fun _ ha => ha ▸ h

theorem rx_draw {C : Type} [Fintype C] [Nonempty C] {f : C → Game β} {Q : β → Prop} (c : C)
    (h : ∃ b, Reaches (f c) b ∧ Q b) : ∃ b, Reaches (draw C >>= f) b ∧ Q b :=
  rx_bind (P := (· = c)) ⟨c, reaches_coin_iff.2 ⟨c, rfl⟩, rfl⟩ fun _ ha => ha ▸ h

theorem rx_pure {a : β} {Q : β → Prop} (h : Q a) : ∃ b, Reaches (pure a : Game β) b ∧ Q b :=
  ⟨a, rfl, h⟩

theorem pb {g : Game α} {f : α → Game β} (hg : Productive g) (hf : ∀ a, Productive (f a)) :
    Productive (g >>= f) :=
  productive_bind hf g hg

theorem productive_pure (a : α) : Productive (pure a : Game α) := trivial

theorem productive_recv (M : Type) [Nonempty M] : Productive (recv M) :=
  productive_send_iff.2 ⟨inferInstance, fun _ => trivial⟩

theorem productive_draw (C : Type) [Fintype C] [Nonempty C] : Productive (draw C) :=
  productive_coin_iff.2 fun _ => trivial

/-- A productive game has a run. -/
theorem productive_reaches : ∀ g : Game α, Productive g → ∃ a, Reaches g a
  | .ret a, _ => ⟨a, rfl⟩
  | .send M k, h => by
      obtain ⟨⟨m⟩, hk⟩ := productive_send_iff.1 h
      obtain ⟨a, ha⟩ := productive_reaches (k m) (hk m)
      exact ⟨a, reaches_send_iff.2 ⟨m, ha⟩⟩
  | @Game.coin _ C _ hC k, h => by
      have c : C := Classical.choice hC
      obtain ⟨a, ha⟩ := productive_reaches (k c) (productive_coin_iff.1 h c)
      exact ⟨a, reaches_coin_iff.2 ⟨c, ha⟩⟩

theorem rx_any {g : Game α} (h : Productive g) : ∃ a, Reaches g a ∧ True :=
  let ⟨a, ha⟩ := productive_reaches g h
  ⟨a, ha, trivial⟩

theorem productive_interleaveWith {adv : Game β → Game (α × β)} {fin : β → Game (α × β)}
    (hadv : ∀ g, Productive g → Productive (adv g)) (hfin : ∀ b, Productive (fin b)) :
    ∀ g : Game β, Productive g → Productive (interleaveWith adv fin g)
  | .ret b, _ => hfin b
  | .send M k, h => by
      obtain ⟨hM, hk⟩ := productive_send_iff.1 h
      refine productive_send_iff.2 ⟨inferInstance, fun first => ?_⟩
      cases first
      · exact productive_send_iff.2 ⟨hM, fun m => productive_interleaveWith hadv hfin (k m) (hk m)⟩
      · exact hadv _ h
  | @Game.coin _ C _ _ k, h => by
      refine productive_send_iff.2 ⟨inferInstance, fun first => ?_⟩
      cases first
      · exact productive_coin_iff.2 fun c =>
          productive_interleaveWith hadv hfin (k c) (productive_coin_iff.1 h c)
      · exact hadv _ h

theorem productive_interleave :
    ∀ (g₁ : Game α) (g₂ : Game β), Productive g₁ → Productive g₂ → Productive (interleave g₁ g₂)
  | .ret _, _, _, h₂ => productive_map _ h₂
  | .send M k, g₂, h₁, h₂ => by
      obtain ⟨hM, hk⟩ := productive_send_iff.1 h₁
      exact productive_interleaveWith
        (fun g hg => productive_send_iff.2 ⟨hM, fun m => productive_interleave (k m) g (hk m) hg⟩)
        (fun _ => productive_map _ h₁) g₂ h₂
  | @Game.coin _ C _ _ k, g₂, h₁, h₂ =>
      productive_interleaveWith
        (fun g hg => productive_coin_iff.2 fun c =>
          productive_interleave (k c) g (productive_coin_iff.1 h₁ c) hg)
        (fun _ => productive_map _ h₁) g₂ h₂

/-- Two runs make a run of the two games interleaved: the first runs to its end, then the second. -/
theorem reaches_interleave :
    ∀ (g₁ : Game α) (g₂ : Game β) (a : α) (b : β), Reaches g₁ a → Reaches g₂ b →
      Reaches (interleave g₁ g₂) (a, b)
  | .ret a', g₂, a, b, h, hb => by
      have h : a' = a := h
      subst h
      exact reaches_map (fun b => (a', b)) hb
  | .send M k, g₂, a, b, h, hb => by
      obtain ⟨m, hm⟩ := reaches_send_iff.1 h
      cases g₂ with
      | ret b' =>
          have hb : b' = b := hb
          subst hb
          exact reaches_map (fun x => (x, b')) h
      | send M' k' =>
          exact reaches_send_iff.2 ⟨true, reaches_send_iff.2 ⟨m, reaches_interleave (k m) _ a b hm hb⟩⟩
      | coin C k' =>
          exact reaches_send_iff.2 ⟨true, reaches_send_iff.2 ⟨m, reaches_interleave (k m) _ a b hm hb⟩⟩
  | @Game.coin _ C _ _ k, g₂, a, b, h, hb => by
      obtain ⟨c, hc⟩ := reaches_coin_iff.1 h
      cases g₂ with
      | ret b' =>
          have hb : b' = b := hb
          subst hb
          exact reaches_map (fun x => (x, b')) h
      | send M' k' =>
          exact reaches_send_iff.2 ⟨true, reaches_coin_iff.2 ⟨c, reaches_interleave (k c) _ a b hc hb⟩⟩
      | coin C' k' =>
          exact reaches_send_iff.2 ⟨true, reaches_coin_iff.2 ⟨c, reaches_interleave (k c) _ a b hc hb⟩⟩

theorem productive_batch : ∀ gs : List (Game α), (∀ g ∈ gs, Productive g) → Productive (batch gs)
  | [], _ => trivial
  | g :: gs, h => productive_map _ (productive_interleave g (batch gs) (h g (by simp))
      (productive_batch gs fun g' hg' => h g' (by simp [hg'])))

theorem reaches_batch {P : α → Prop} : ∀ gs : List (Game α), (∀ g ∈ gs, ∃ a, Reaches g a ∧ P a) →
    ∃ as, Reaches (batch gs) as ∧ as.length = gs.length ∧ ∀ a ∈ as, P a
  | [], _ => ⟨[], rfl, rfl, by simp⟩
  | g :: gs, h => by
      obtain ⟨a, ha, hpa⟩ := h g (by simp)
      obtain ⟨as, has, hl, hp⟩ := reaches_batch gs fun g' hg' => h g' (by simp [hg'])
      refine ⟨a :: as, reaches_map_of (reaches_interleave g (batch gs) a as ha has) rfl, by simp [hl],
        fun x hx => ?_⟩
      rcases List.mem_cons.1 hx with rfl | hx
      exacts [hpa, hp x hx]

theorem productive_mapM {γ : Type} (f : γ → Game β) (hf : ∀ c, Productive (f c)) :
    ∀ l : List γ, Productive (l.mapM f)
  | [] => trivial
  | c :: l => by
      rw [List.mapM_cons]
      exact pb (hf c) fun _ => pb (productive_mapM f hf l) fun _ => trivial

theorem reaches_mapM {γ : Type} (f : γ → Game β) (g : γ → β) :
    ∀ l : List γ, (∀ c ∈ l, Reaches (f c) (g c)) → Reaches (l.mapM f) (l.map g)
  | [], _ => rfl
  | c :: l, h => by
      rw [List.mapM_cons]
      exact reaches_bind _ _ (h c (by simp))
        (reaches_bind _ _ (reaches_mapM f g l fun c' hc' => h c' (by simp [hc'])) rfl)

end Toolkit

/-! ## The model's programs are productive -/

section ModelProductive

set_option linter.unusedSectionVars false

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type} [Inhabited D]

theorem residualRounds_productive : ∀ (n : ℕ) (req : Fin n → F) (c : F), Productive (residualRounds n req c)
  | 0, _, _ => trivial
  | n + 1, req, c => by
      rw [residualRounds]
      refine pb (productive_recv _) fun g => pb (productive_draw _) fun ρ =>
        pb (residualRounds_productive n _ _) fun x => ?_
      obtain ⟨_, _⟩ := x
      exact trivial

theorem zerocheck_productive (A : Arith F K) (m : ℕ) : Productive (Model.zerocheck A m) := by
  unfold Model.zerocheck
  refine pb (productive_draw _) fun _ => pb (productive_draw _) fun _ => pb (productive_recv _) fun _ =>
    pb (productive_draw _) fun _ => pb (residualRounds_productive _ _ _) fun x => ?_
  obtain ⟨_, _⟩ := x
  exact pb (productive_recv _) fun _ => trivial

theorem lincheckRounds_productive : ∀ (n : ℕ) (c : F), Productive (lincheckRounds n c)
  | 0, _ => trivial
  | n + 1, c => by
      rw [lincheckRounds]
      refine pb (productive_recv _) fun _ => pb (productive_draw _) fun _ =>
        pb (lincheckRounds_productive n _) fun x => ?_
      obtain ⟨_, _⟩ := x
      exact trivial

theorem lincheck_productive (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) :
    Productive (Model.lincheck A S zc) := by
  unfold Model.lincheck
  refine pb (productive_draw _) fun _ => pb (productive_draw _) fun _ =>
    pb (lincheckRounds_productive _ _) fun x => ?_
  obtain ⟨_, _⟩ := x
  exact pb (productive_recv _) fun _ => pb (productive_draw _) fun _ => trivial

theorem opening_productive (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (claims : List (Claim F m)) :
    Productive (Model.opening A hm claims) := by
  unfold Model.opening
  refine pb (productive_mapM _ (fun _ => ?_) _) fun _ => pb (productive_draw _) fun _ => ?_
  · exact pb (productive_recv _) fun _ => pb (productive_draw _) fun _ => trivial
  · exact trivial

theorem drawK_productive (A : Arith F K) : Productive (drawK A) := by
  unfold drawK
  exact pb (productive_draw _) fun _ => trivial

theorem foldRoundsC_productive (A : Arith F K) :
    ∀ (k : ℕ) (s : Opens F D → K) (q : Opens F D → Quad K), Productive (foldRoundsC A k s q)
  | 0, _, _ => trivial
  | k + 1, s, q => by
      rw [foldRoundsC]
      refine pb (drawK_productive A) fun _ => pb (productive_recv _) fun _ =>
        pb (foldRoundsC_productive A k _ _) fun x => ?_
      obtain ⟨_, _, _⟩ := x
      exact trivial

theorem twoOodC_productive (A : Arith F K) (dim split : ℕ) (s : Opens F D → K) (q : Opens F D → Quad K) :
    Productive (twoOodC A dim split s q) := by
  unfold twoOodC
  exact pb (productive_draw _) fun _ => pb (productive_recv _) fun _ => pb (productive_recv _) fun _ =>
    pb (productive_draw _) fun _ => pb (productive_draw _) fun _ => pb (productive_recv _) fun _ =>
    pb (productive_recv _) fun _ => pb (productive_draw _) fun _ => trivial

theorem queryAndBatchC_productive (A : Arith F K) (r : ℕ) (lv : Level) (ch : List K) (start : ℕ)
    (st : LigStateC F K D) (wm : Bool) : Productive (queryAndBatchC A r lv ch start st wm) := by
  unfold queryAndBatchC
  refine pb (productive_draw _) fun _ => pb (productive_draw _) fun _ => ?_
  dsimp only
  split
  · exact pb (productive_recv _) fun _ => pb (productive_pure _) fun _ => pb (productive_draw _) fun _ => trivial
  · exact pb (productive_pure _) fun _ => pb (productive_draw _) fun _ => trivial

theorem recursiveRoundsC_productive (A : Arith F K) (r n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ) :
    ∀ (lvs : List Level) (st : LigStateC F K D), Productive (recursiveRoundsC A r n b initial ext lvs st)
  | [], _ => trivial
  | [lv], st => by
      rw [recursiveRoundsC]
      refine pb (foldRoundsC_productive A _ _ _) fun x => ?_
      obtain ⟨_, _, _⟩ := x
      exact pb (productive_recv _) fun _ => pb (queryAndBatchC_productive A _ _ _ _ _ _) fun _ => trivial
  | lv :: lv' :: rest, st => by
      rw [recursiveRoundsC]
      refine pb (foldRoundsC_productive A _ _ _) fun x => ?_
      obtain ⟨_, _, _⟩ := x
      refine pb (productive_recv _) fun _ => pb (twoOodC_productive A _ _ _ _) fun y => ?_
      obtain ⟨_, _, _⟩ := y
      exact pb (queryAndBatchC_productive A _ _ _ _ _ _) fun _ =>
        recursiveRoundsC_productive A r n b initial ext (lv' :: rest) _

theorem ligeritoC_productive (A : Arith F K) (r : ℕ) (sch : Schedule) (T : F) (b : ℕ → F) :
    Productive (ligeritoC A r sch T b : Game (PendL F D)) := by
  unfold ligeritoC
  dsimp only
  split
  · exact trivial
  · refine pb (productive_draw _) fun _ => pb (productive_recv _) fun _ => pb (productive_draw _) fun _ =>
      pb (productive_recv _) fun _ => pb (foldRoundsC_productive A _ _ _) fun x => ?_
    obtain ⟨_, _, _⟩ := x
    split
    · exact trivial
    · refine pb (productive_recv _) fun _ => pb (twoOodC_productive A _ _ _ _) fun y => ?_
      obtain ⟨_, _, _⟩ := y
      exact pb (queryAndBatchC_productive A _ _ _ _ _ _) fun _ => recursiveRoundsC_productive A _ _ _ _ _ _ _

theorem repC_productive (A : Arith F K) (r : ℕ) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) : Productive (repC A r S sch hm extra : Game (PendL F D)) := by
  unfold repC
  refine pb (zerocheck_productive A _) fun x => ?_
  rcases x with _ | zc
  · exact trivial
  refine pb (lincheck_productive A _ _) fun y => ?_
  rcases y with _ | ab
  · exact trivial
  refine pb (opening_productive A _ _) fun z => ?_
  rcases z with _ | op
  · exact trivial
  exact ligeritoC_productive A r sch _ _

end ModelProductive

/-! ## Zero runs of the PIOP -/

section ZeroPiop

set_option linter.unusedSectionVars false

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

theorem residualStep_zero (req ρ : F) : residualStep req 0 0 0 ρ = 0 := by
  simp [residualStep]

theorem residualRounds_zero : ∀ (n : ℕ) (req : Fin n → F) (c : F), c = 0 →
    ∃ x, Reaches (residualRounds n req c) x ∧ x.2 = 0
  | 0, _, c, hc => ⟨_, rfl, hc⟩
  | n + 1, req, c, hc => by
      subst hc
      rw [residualRounds]
      refine rx_recv (0, 0) (rx_draw 0 (rx_bind (residualRounds_zero n _ _ (residualStep_zero _ _))
        fun x hx => ?_))
      obtain ⟨_, _⟩ := x
      exact rx_pure hx

/-- The zero run of the zerocheck: zero messages, zero coins, and `final_a_eval = final_b_eval = 0`. -/
theorem zerocheck_zero (A : Arith F K) (m : ℕ) :
    ∃ x, Reaches (Model.zerocheck A m) x ∧ ∃ zc, x = some zc ∧ zc.va = 0 ∧ zc.vb = 0 ∧ zc.vc = 0 := by
  unfold Model.zerocheck
  refine rx_draw 0 (rx_draw 0 (rx_recv (0, 0) (rx_draw 0 (rx_bind (residualRounds_zero _ _ _ (by simp))
    fun x hx => ?_))))
  obtain ⟨_, c⟩ := x
  refine rx_recv (0, 0) (rx_pure ?_)
  simp only at hx
  subst hx
  simp

/-- The zerocheck rejects when `final_a_eval = final_b_eval = 1` follow the zero rounds. -/
theorem zerocheck_reject (A : Arith F K) (m : ℕ) : Reaches (Model.zerocheck A m) none := by
  obtain ⟨x, hx, rfl⟩ : ∃ x, Reaches (Model.zerocheck A m) x ∧ x = none := by
    unfold Model.zerocheck
    refine rx_draw 0 (rx_draw 0 (rx_recv (0, 0) (rx_draw 0 (rx_bind (residualRounds_zero _ _ _ (by simp))
      fun x hx => ?_))))
    obtain ⟨_, c⟩ := x
    refine rx_recv (1, 1) (rx_pure ?_)
    simp only at hx
    subst hx
    simp
  exact hx

theorem lincheckRounds_zero : ∀ (n : ℕ) (c : F), c = 0 → ∃ x, Reaches (lincheckRounds n c) x ∧ x.2 = 0
  | 0, c, hc => ⟨_, rfl, hc⟩
  | n + 1, c, hc => by
      subst hc
      rw [lincheckRounds]
      refine rx_recv (0, 0) (rx_draw 0 (rx_bind (lincheckRounds_zero n _ (by simp)) fun x hx => ?_))
      obtain ⟨_, _⟩ := x
      exact rx_pure hx

/-- The zero run of the lincheck on zero claims: its `ab` claim has value `0`. -/
theorem lincheck_zero (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) (ha : zc.va = 0) (hb : zc.vb = 0) :
    ∃ x, Reaches (Model.lincheck A S zc) x ∧ ∃ ab, x = some ab ∧ ab.value = 0 := by
  unfold Model.lincheck
  refine rx_draw 0 (rx_draw 0 (rx_bind (lincheckRounds_zero _ _ (by simp [ha, hb])) fun x hx => ?_))
  obtain ⟨_, c⟩ := x
  refine rx_recv 0 (rx_draw 0 (rx_pure ?_))
  simp only at hx
  subst hx
  simp

theorem solve_of_value_zero {m : ℕ} (c : Claim F m) (hm : 7 ≤ m) (h : c.value = 0) :
    ∃ s : Fin 128 → F, ∑ v, packedWeight c hm v * s v = c.value :=
  ⟨0, by simp [h]⟩

theorem solve_of_weight {m : ℕ} (c : Claim F m) (hm : 7 ≤ m) (v : Fin 128) (h : packedWeight c hm v = 1) :
    ∃ s : Fin 128 → F, ∑ w, packedWeight c hm w * s w = c.value :=
  ⟨fun w => if w = v then c.value else 0, by simp [h]⟩

/-- The zero run of the opening, when every claim's ring-switch check has a solution: the batching coins are zero, so
the batched target is `0`. -/
theorem opening_zero (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (claims : List (Claim F m))
    (hs : ∀ c ∈ claims, ∃ s : Fin 128 → F, ∑ v, packedWeight c hm v * s v = c.value) :
    ∃ x, Reaches (Model.opening A hm claims) x ∧ ∃ op, x = some op ∧ op.T = 0 := by
  classical
  let sOf : Claim F m → Fin 128 → F := fun c =>
    if h : ∃ s : Fin 128 → F, ∑ v, packedWeight c hm v * s v = c.value then h.choose else 0
  have hsOf : ∀ c ∈ claims, ∑ v, packedWeight c hm v * sOf c v = c.value := fun c hc => by
    simp only [sOf, dite_eq_left_of_eq_true (eq_true (hs c hc))]
    exact (hs c hc).choose_spec
  unfold Model.opening
  refine rx_bind (P := (· = claims.map fun c => (c, sOf c, (0 : Fin 7 → F))))
    ⟨_, reaches_mapM _ (fun c => (c, sOf c, (0 : Fin 7 → F))) claims (fun c _ =>
      reaches_bind _ (sOf c) (reaches_send_iff.2 ⟨sOf c, rfl⟩)
        (reaches_bind _ 0 (reaches_coin_iff.2 ⟨0, rfl⟩) rfl)), rfl⟩ fun x hx => ?_
  subst hx
  refine rx_draw 0 (rx_pure ?_)
  split_ifs with hok
  · exact ⟨_, rfl, by simp⟩
  · exfalso
    apply hok
    rw [List.all_eq_true]
    intro x hx
    obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hx
    simpa using hsOf c hc

theorem exists_bits (b : Fin 6 → Bool) : ∃ s : Fin 64, ∀ j : Fin 6, s.val.testBit j.val = b j := by
  have key : ∀ b₀ b₁ b₂ b₃ b₄ b₅ : Bool,
      let s := b₀.toNat + 2 * b₁.toNat + 4 * b₂.toNat + 8 * b₃.toNat + 16 * b₄.toNat + 32 * b₅.toNat
      s < 64 ∧ s.testBit 0 = b₀ ∧ s.testBit 1 = b₁ ∧ s.testBit 2 = b₂ ∧ s.testBit 3 = b₃ ∧
        s.testBit 4 = b₄ ∧ s.testBit 5 = b₅ := by
    decide
  obtain ⟨hs, h₀, h₁, h₂, h₃, h₄, h₅⟩ := key (b 0) (b 1) (b 2) (b 3) (b 4) (b 5)
  exact ⟨⟨_, hs⟩, fun j => by fin_cases j <;> assumption⟩

theorem exists_prod_one (a : Fin 6 → F) (ha : ∀ j, a j = 0 ∨ a j = 1) :
    ∃ s : Fin 64, ∏ j : Fin 6, (if s.val.testBit j.val then a j else 1 + a j) = 1 := by
  obtain ⟨s, hs⟩ := exists_bits fun j => decide (a j = 1)
  refine ⟨s, Finset.prod_eq_one fun j _ => ?_⟩
  rw [hs j]
  rcases ha j with h | h <;> simp [h]

theorem regionPoint_zero (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ) (j : Fin S.m) :
    regionPoint S R ptLocal mPts (0 : Fin mPts → F) j = 0 ∨ regionPoint S R ptLocal mPts (0 : Fin mPts → F) j = 1 := by
  unfold regionPoint
  dsimp only
  by_cases hj : j.val < S.kLog
  · simp only [hj, ↓reduceDIte]
    cases R.free.idxOf? ⟨j.val, hj⟩ <;> dsimp only <;> split_ifs <;> simp
  · simp only [hj, ↓reduceDIte]
    split_ifs <;> simp

theorem packedWeight_one {m : ℕ} (c : Claim F m) (hm : 7 ≤ m) (s : Fin 64) (hs : c.skip s = 1)
    (hx : c.x ⟨0, by omega⟩ = 0 ∨ c.x ⟨0, by omega⟩ = 1) : ∃ v, packedWeight c hm v = 1 := by
  have hs64 := s.isLt
  rcases hx with h | h
  · refine ⟨⟨s.val, by omega⟩, ?_⟩
    have h1 : (⟨s.val % 64, Nat.mod_lt _ (by norm_num)⟩ : Fin 64) = s := Fin.ext (Nat.mod_eq_of_lt hs64)
    have h2 : s.val / 64 = 0 := Nat.div_eq_of_lt hs64
    simp only [packedWeight, h1, h2, hs, h]
    norm_num
  · refine ⟨⟨s.val + 64, by omega⟩, ?_⟩
    have h1 : (⟨(s.val + 64) % 64, Nat.mod_lt _ (by norm_num)⟩ : Fin 64) = s := Fin.ext (show (s.val + 64) % 64 = s.val by omega)
    have h2 : (s.val + 64) / 64 = 1 := by omega
    simp only [packedWeight, h1, h2, hs, h]
    norm_num

/-- At the zero link point, a region claim's weight is `1` on some packed position: its point's coordinates are
`0` or `1`, so one skip position and one half of the packed hypercube carry weight `1`. -/
theorem regionClaim_weight (S : Statement) (R : Region S.kLog S.nbl) (ptLocal mPts : ℕ) (hm : 7 ≤ S.m) :
    ∃ v, packedWeight (regionClaim S R ptLocal mPts (0 : Fin mPts → F)) hm v = 1 := by
  have h01 := regionPoint_zero (F := F) S R ptLocal mPts
  have ha : ∀ j : Fin 6, (if h : j.val < S.m then regionPoint S R ptLocal mPts (0 : Fin mPts → F) ⟨j.val, h⟩
      else 0) = 0 ∨ (if h : j.val < S.m then regionPoint S R ptLocal mPts (0 : Fin mPts → F) ⟨j.val, h⟩
      else 0) = 1 := fun j => by
    split_ifs
    exacts [h01 _, Or.inl rfl]
  obtain ⟨s, hs⟩ := exists_prod_one _ ha
  refine packedWeight_one _ hm s hs ?_
  show (if h : 0 + 6 < S.m then regionPoint S R ptLocal mPts (0 : Fin mPts → F) ⟨0 + 6, h⟩ else 0) = 0 ∨
    (if h : 0 + 6 < S.m then regionPoint S R ptLocal mPts (0 : Fin mPts → F) ⟨0 + 6, h⟩ else 0) = 1
  split_ifs
  exacts [h01 _, Or.inl rfl]

end ZeroPiop

/-! ## The session is productive -/

section SessionProductive

set_option linter.unusedSectionVars false

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type} [Inhabited D] [Inhabited Sl] {mPts : ℕ}

theorem tabAfterL_productive (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (T : TabSpec mPts) (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) :
    Productive (tabAfterL A H E Lf T cap₀ pts) := by
  unfold tabAfterL tableAfterL
  refine productive_bind (fun _ => productive_send_iff.2 ⟨inferInstance, fun _ => ?_⟩) _
    (productive_interleave _ _ (repC_productive A 0 _ _ _ _) (repC_productive A 1 _ _ _ _))
  exact trivial

theorem games_productive (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (caps : Fin ts.length → Cap D)
    (pts : Fin 2 → Fin mPts → F) : ∀ g ∈ (gamesBL A H E Lf ts caps).map (· pts), Productive g := by
  intro g hg
  simp only [gamesBL, List.mem_map, List.mem_ofFn] at hg
  obtain ⟨_, ⟨i, rfl⟩, rfl⟩ := hg
  exact tabAfterL_productive A H E Lf _ _ _

theorem sessionBL_productive (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) : Productive (sessionBL A H E Lf ts) :=
  productive_send_iff.2 ⟨inferInstance, fun caps => productive_coin_iff.2 fun pts =>
    productive_batch _ (games_productive A H E Lf ts caps pts)⟩

end SessionProductive

local notation "𝔽" => FlockLevel3.GF128

variable {D Sl : Type} [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}

/-! ## The schedule's shape -/

/-- Every `fast100` schedule folds `k0` lanes at level 0 and one fewer than each later level's lanes after it, and
these with `ext` make up the `m − 7` packed coordinates. -/
theorem fast100_orig_length : ∀ m s, fast100 m = some s →
    s.k0 + ((s.levels.drop 1).map fun lv => lv.lanes - 1).sum + s.ext = s.m - 7 := by
  intro m s h
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : fast100 m = none := by
      unfold fast100
      split <;> first | rfl | omega
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; decide +kernel)

/-- **`finalCheck`'s length condition, for a table**: the level-0 folds and the later levels' folds past their first
challenge, with the clear residual, are `m − 7` coordinates. -/
theorem TabSpec.orig_length (T : TabSpec mPts) :
    T.k0 + ((T.l₁ :: T.rest).map fun lv => lv.lanes - 1).sum + T.ext = T.m - 7 :=
  fast100_orig_length T.S.m T.sch T.hsch

/-! ## The session -/

/-- **Can accept.** For every hash, encoding, leaf scheme, plan, draw and registration, some run of the salted batched
session accepts. -/
theorem batchedSessionL_reaches_true (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (S : Finset (Fin n)) (R : Reg) :
    Reaches (batchedSessionL H E Lf plan S R) true := by
  sorry

/-- **Can reject.** When the plan has a table on the draw, some run of the salted batched session rejects. -/
theorem batchedSessionL_reaches_false (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (S : Finset (Fin n)) (R : Reg) (hne : plan S R ≠ []) :
    Reaches (batchedSessionL H E Lf plan S R) false := by
  sorry

/-- **Productive.** Every message type the salted batched session asks for is inhabited. -/
theorem batchedSessionL_productive (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (S : Finset (Fin n)) (R : Reg) :
    Productive (batchedSessionL H E Lf plan S R) :=
  productive_map _ (sessionBL_productive execArith H E Lf (plan S R))

/-- **The salted batched session is decisive** on a draw where the plan has a table. -/
theorem batchedSessionL_decisive (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (S : Finset (Fin n)) (R : Reg) (hne : plan S R ≠ []) :
    Decisive (batchedSessionL H E Lf plan S R) :=
  ⟨batchedSessionL_productive H E Lf plan S R, true, false, by decide,
    batchedSessionL_reaches_true H E Lf plan S R, batchedSessionL_reaches_false H E Lf plan S R hne⟩

/-! ## The one-stage audit over it -/

/-- **The audit can accept**, on every registration and every draw. -/
theorem audit_reaches_accept (L : Law n) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) (R : Reg) (ω : L.Ω) :
    Reaches (audit L Reg (batchedSessionL H E Lf plan)) (true, L.draw ω) :=
  ⟨R, ω, reaches_map (fun ok => (ok, L.draw ω)) (batchedSessionL_reaches_true H E Lf plan (L.draw ω) R)⟩

/-- **The audit can reject**, on every registration and every draw where the plan has a table. -/
theorem audit_reaches_reject (L : Law n) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) (R : Reg) (ω : L.Ω)
    (hne : plan (L.draw ω) R ≠ []) :
    Reaches (audit L Reg (batchedSessionL H E Lf plan)) (false, L.draw ω) :=
  ⟨R, ω, reaches_map (fun ok => (ok, L.draw ω)) (batchedSessionL_reaches_false H E Lf plan (L.draw ω) R hne)⟩

/-- **The audit is productive** when registrations exist. -/
theorem audit_productive [Nonempty Reg] (L : Law n) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) :
    Productive (audit L Reg (batchedSessionL H E Lf plan)) :=
  ⟨inferInstance, fun R ω => productive_map _ (batchedSessionL_productive H E Lf plan (L.draw ω) R)⟩

/-- **The audit is decisive** when registrations exist and the plan has a table on every draw. -/
theorem audit_decisive [Nonempty Reg] (L : Law n) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (hne : ∀ S R, plan S R ≠ []) :
    Decisive (audit L Reg (batchedSessionL H E Lf plan)) := by
  obtain ⟨R⟩ := ‹Nonempty Reg›
  obtain ⟨ω⟩ := L.nonempty
  exact ⟨audit_productive L H E Lf plan, (true, L.draw ω), (false, L.draw ω), by simp,
    audit_reaches_accept L H E Lf plan R ω, audit_reaches_reject L H E Lf plan R ω (hne _ _)⟩

/-- **A tile's audit is decisive**: the one-stage audit over the salted batched session of a one-table plan. -/
theorem audit_tile_decisive [Nonempty Reg] (L : Law n) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (T : Finset (Fin n) → Reg → TabSpec mPts) :
    Decisive (audit L Reg (batchedSessionL H E Lf fun S R => [T S R])) :=
  audit_decisive L H E Lf _ fun _ _ => List.cons_ne_nil _ _

end FlockSoundness.Discharge.Exec.SessionLComplete
