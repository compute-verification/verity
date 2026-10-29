import Proofs.Flock.Claims.Recursion.ResNextW
import Proofs.Flock.Soundness.ExecDraw

/-!
# Some algebra part reads `acc1` (`AccEndHeldW`)

V*'s algebra (`VStar.Algebra.spec`) has `acc1` as its last row port, and some part of it (`VStar.parts`) reads that
port, at every shape it accepts:

* Ligerito's final sum carries a consistency sum in both components (`ligerito_hc`): level 0's `query` adds one
  (`query_hc`), and `vx` (a union) keeps every key through the later levels;
* so `supports` has a residual with a consistency sum (`supports_cons`), and `build` refuses one outside Ligerito's
  final check, so the residual it builds adds `w (base - 2 + d)`, `d < 2` (`build_acc`);
* `spec`'s ports end `acc0, acc1` and their sizes sum to `nw = base` (`checkRepS`, `spec_acc`), so that element is in
  the last port (`portOf_acc1`);
* `touched` lists that port (`touched_w`), `parts`' groups keep it in some group's ports (`union_mem`) and each group is
  a part over its ports (`parts_acc1`).

With it, `NextHeldW` (`nextHeldW_of`), StepsW and P3 at KS's brings hold with no hypothesis (`accPorts_deployed`).
-/

namespace RecDeploy.AccEnd

open VStar VStar.Algebra


theorem forIn_count_id {α σ : Type} (Q : Nat → σ → Prop) (f : α → σ → Id (ForInStep σ))
    (hf : ∀ a r m, Q m r → ∃ r', f a r = .yield r' ∧ Q (m + 1) r') :
    ∀ (L : List α) (init : σ) (m : Nat), Q m init → Q (m + L.length) (forIn L init f)
  | [], init, m, h => h
  | a :: L, init, m, h => by
    obtain ⟨r', h1, h2⟩ := hf a init m h
    rw [List.forIn_cons, h1]
    have e : m + (a :: L).length = m + 1 + L.length := by simp only [List.length_cons]; omega
    rw [e]
    exact forIn_count_id Q f hf L r' (m + 1) h2

def unionStep {α : Type} [Ord α] [Inhabited α] (a b : Array α) (_ : Nat) (s : Array α × Nat × Nat) :
    Id (ForInStep (Array α × Nat × Nat)) :=
  if s.2.1 < a.size then
    if s.2.2 < b.size then
      match compare a[s.2.1]! b[s.2.2]! with
      | .lt => pure (.yield (s.1.push a[s.2.1]!, s.2.1 + 1, s.2.2))
      | .gt => pure (.yield (s.1.push b[s.2.2]!, s.2.1, s.2.2 + 1))
      | .eq => pure (.yield (s.1.push a[s.2.1]!, s.2.1 + 1, s.2.2 + 1))
    else pure (.yield (s.1.push a[s.2.1]!, s.2.1 + 1, s.2.2))
  else if s.2.2 < b.size then pure (.yield (s.1.push b[s.2.2]!, s.2.1, s.2.2 + 1))
  else pure (.yield (s.1, s.2.1, s.2.2))

theorem union_eq {α : Type} [Ord α] [Inhabited α] (a b : Array α) :
    union a b = (forIn (List.range' 0 (a.size + b.size)) (Array.mkEmpty (a.size + b.size), 0, 0) (unionStep a b)).1 := by
  unfold union
  simp only [Id.run]
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.add_sub_cancel, Nat.div_one, Nat.sub_zero]
  rfl

def UQ {α : Type} [Ord α] [Inhabited α] (a b : Array α) (m : Nat) (r : Array α × Nat × Nat) : Prop :=
  r.2.1 ≤ a.size ∧ r.2.2 ≤ b.size ∧ (m ≤ r.2.1 + r.2.2 ∨ (r.2.1 = a.size ∧ r.2.2 = b.size)) ∧
    (∀ k < r.2.1, a[k]! ∈ r.1) ∧ (∀ k < r.2.2, b[k]! ∈ r.1 ∨ ∃ y ∈ r.1, compare y b[k]! = .eq)

theorem uq_b {α : Type} [Ord α] [Inhabited α] {out : Array α} {x : α} {j : Nat} {b : Array α}
    (hb : ∀ k < j, b[k]! ∈ out ∨ ∃ y ∈ out, compare y b[k]! = .eq) :
    ∀ k < j, b[k]! ∈ out.push x ∨ ∃ y ∈ out.push x, compare y b[k]! = .eq := by
  intro k hk
  rcases hb k hk with h | ⟨y, hy, he⟩
  · exact .inl (Array.mem_push_of_mem _ h)
  · exact .inr ⟨y, Array.mem_push_of_mem _ hy, he⟩

theorem uq_a {α : Type} [Inhabited α] {out : Array α} {i : Nat} {a : Array α} (ha : ∀ k < i, a[k]! ∈ out) :
    ∀ k < i + 1, a[k]! ∈ out.push a[i]! := by
  intro k hk
  rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
  · exact Array.mem_push_of_mem _ (ha k hk)
  · exact Array.mem_push_self

theorem uq_a' {α : Type} [Inhabited α] {out : Array α} {x : α} {i : Nat} {a : Array α} (ha : ∀ k < i, a[k]! ∈ out) :
    ∀ k < i, a[k]! ∈ out.push x := fun k hk => Array.mem_push_of_mem _ (ha k hk)

theorem uq_b' {α : Type} [Ord α] [Inhabited α] {out : Array α} {j : Nat} {b : Array α}
    (hb : ∀ k < j, b[k]! ∈ out ∨ ∃ y ∈ out, compare y b[k]! = .eq) :
    ∀ k < j + 1, b[k]! ∈ out.push b[j]! ∨ ∃ y ∈ out.push b[j]!, compare y b[k]! = .eq := by
  intro k hk
  rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
  · exact uq_b hb k hk
  · exact .inl Array.mem_push_self

theorem unionStep_uq {α : Type} [Ord α] [Inhabited α] (a b : Array α) :
    ∀ (n : Nat) (r : Array α × Nat × Nat) (m : Nat), UQ a b m r → ∃ r', unionStep a b n r = .yield r' ∧ UQ a b (m + 1) r' := by
  rintro _ ⟨out, i, j⟩ m ⟨hi, hj, hc, ha, hb⟩
  dsimp only at hi hj hc ha hb
  unfold unionStep
  dsimp only
  split
  · split
    · split
      · exact ⟨_, rfl, by unfold UQ; dsimp only; exact ⟨by omega, by omega, by omega, uq_a ha, uq_b hb⟩⟩
      · exact ⟨_, rfl, by unfold UQ; dsimp only; exact ⟨by omega, by omega, by omega, uq_a' ha, uq_b' hb⟩⟩
      · rename_i heq
        have hb2 : ∀ k < j + 1, b[k]! ∈ out.push a[i]! ∨ ∃ y ∈ out.push a[i]!, compare y b[k]! = .eq := by
          intro k hk
          rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
          · exact uq_b hb k hk
          · exact .inr ⟨_, Array.mem_push_self, heq⟩
        exact ⟨_, rfl, by unfold UQ; dsimp only; exact ⟨by omega, by omega, by omega, uq_a ha, hb2⟩⟩
    · exact ⟨_, rfl, by unfold UQ; dsimp only; exact ⟨by omega, by omega, by omega, uq_a ha, uq_b hb⟩⟩
  · split
    · exact ⟨_, rfl, by unfold UQ; dsimp only; exact ⟨by omega, by omega, by omega, uq_a' ha, uq_b' hb⟩⟩
    · exact ⟨_, rfl, by unfold UQ; dsimp only; exact ⟨hi, hj, by omega, ha, hb⟩⟩

theorem union_mem {α : Type} [Ord α] [Inhabited α] (a b : Array α) :
    (∀ x ∈ a, x ∈ union a b) ∧ (∀ x ∈ b, x ∈ union a b ∨ ∃ y ∈ union a b, compare y x = .eq) := by
  have hQ := forIn_count_id (UQ a b) (unionStep a b) (unionStep_uq a b) (List.range' 0 (a.size + b.size))
    (Array.mkEmpty (a.size + b.size), 0, 0) 0
    ⟨Nat.zero_le _, Nat.zero_le _, .inl (Nat.zero_le _), fun k hk => absurd hk (Nat.not_lt_zero _),
      fun k hk => absurd hk (Nat.not_lt_zero _)⟩
  rw [union_eq]
  obtain ⟨hi, hj, hc, ha, hb⟩ := hQ
  simp only [List.length_range', Nat.zero_add] at hc
  refine ⟨fun x hx => ?_, fun x hx => ?_⟩
  · obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hx
    have := ha k (by omega)
    rwa [getElem!_pos a k hk] at this
  · obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hx
    have := hb k (by omega)
    rwa [getElem!_pos b k hk] at this

abbrev SM (σ : Type) := StateT σ (Except String)
variable {σ : Type}

def Tr {β : Type} (I : σ → Prop) (J : β → σ → Prop) (m : SM σ β) : Prop :=
  ∀ st, I st → ∀ b st', m.run st = .ok (b, st') → J b st'

theorem Tr.bind {β γ : Type} {m : SM σ β} {k : β → SM σ γ} {I : σ → Prop} {K : β → σ → Prop}
    {J : γ → σ → Prop} (hm : Tr I K m) (hk : ∀ b, Tr (K b) J (k b)) : Tr I J (m >>= k) := by
  intro st hI c st'' h
  rw [StateT.run_bind] at h
  cases hm' : m.run st with
  | error e => rw [hm'] at h; cases h
  | ok p =>
    rw [hm'] at h
    obtain ⟨b, st'⟩ := p
    exact hk b st' (hm st hI b st' hm') c st'' h

theorem Tr.pure {β : Type} {I : σ → Prop} {J : β → σ → Prop} {b : β} (h : ∀ st, I st → J b st) :
    Tr I J (Pure.pure b) := by
  intro st hI c st' hr
  simp only [StateT.run_pure] at hr
  cases hr
  exact h st hI

theorem Tr.throw {β : Type} {I : σ → Prop} {J : β → σ → Prop} {e : String} :
    Tr I J (MonadExcept.throw e : SM σ β) := by
  intro st _ c st' hr
  cases hr

theorem Tr.modify {I : σ → Prop} {J : PUnit → σ → Prop} {f : σ → σ}
    (h : ∀ st, I st → J ⟨⟩ (f st)) : Tr I J (modify f : SM σ PUnit) := by
  intro st hI c st' hr
  simp only [StateT.run_modify] at hr
  cases hr
  exact h st hI

theorem Tr.get {I : σ → Prop} {J : σ → σ → Prop} (h : ∀ st, I st → J st st) :
    Tr I J (MonadState.get : SM σ σ) := by
  intro st hI c st' hr
  simp only [StateT.run_get] at hr
  cases hr
  exact h st hI

theorem Tr.lift {β : Type} {I : σ → Prop} {x : Except String β} :
    Tr I (fun b st => I st ∧ x = .ok b) (liftM x : SM σ β) := by
  intro st hI c st' hr
  cases x with
  | error e => cases hr
  | ok a =>
    simp only [liftM, monadLift, MonadLift.monadLift, StateT.lift, StateT.run, bind, Except.bind, pure,
      Except.pure] at hr
    cases hr
    exact ⟨hI, rfl⟩

theorem Tr.ite {β : Type} {c : Prop} [Decidable c] {I : σ → Prop} {J : β → σ → Prop} {a b : SM σ β}
    (ha : c → Tr I J a) (hb : ¬ c → Tr I J b) : Tr I J (if c then a else b) := by
  split
  · exact ha ‹_›
  · exact hb ‹_›

theorem Tr.triv {β : Type} {I : σ → Prop} {m : SM σ β} : Tr I (fun _ _ => True) m :=
  fun _ _ _ _ _ => trivial

theorem Tr.mono {β : Type} {I I' : σ → Prop} {J J' : β → σ → Prop} {m : SM σ β} (h : Tr I J m)
    (hI : ∀ st, I' st → I st) (hJ : ∀ b st, J b st → J' b st) : Tr I' J' m :=
  fun st h1 b st' hr => hJ b st' (h st (hI st h1) b st' hr)

theorem Tr.forIn {α ρ : Type} {I : σ → Prop} (f : α → ρ → SM σ (ForInStep ρ))
    (hf : ∀ a r, Tr I (fun _ => I) (f a r)) : ∀ (L : List α) (init : ρ), Tr I (fun _ => I) (forIn L init f)
  | [], init => by
    rw [List.forIn_nil]
    exact Tr.pure fun st h => h
  | a :: L, init => by
    rw [List.forIn_cons]
    refine Tr.bind (hf a init) fun s => ?_
    cases s with
    | done b => exact Tr.pure fun st h => h
    | yield b => exact Tr.forIn f hf L b

theorem Tr.false {β : Type} {J : β → σ → Prop} {m : SM σ β} : Tr (fun _ => False) J m :=
  fun _ h => absurd h id

theorem Tr.set {I : σ → Prop} {J : PUnit → σ → Prop} {s' : σ} (h : ∀ st, I st → J ⟨⟩ s') :
    Tr I J (MonadStateOf.set s' : SM σ PUnit) := by
  intro st hI c st' hr
  simp only [StateT.run_set] at hr
  cases hr
  exact h st hI

def SQ (P : KV → KV × KV × KV → Prop) : LState → Prop := fun st => P st.s st.q

theorem nextMsg_tr (cx : Ctx) (P : KV → KV × KV × KV → Prop) : Tr (SQ P) (fun _ => SQ P) (nextMsg cx) := by
  unfold nextMsg
  refine Tr.bind (K := fun st0 st => st = st0 ∧ SQ P st) (Tr.get fun st h => ⟨rfl, h⟩) fun st0 => ?_
  refine Tr.ite (fun _ => Tr.bind (K := fun _ _ => False) Tr.throw fun _ => Tr.false) fun _ => ?_
  refine Tr.bind (K := fun _ st => SQ P st) (Tr.set fun st h => by rw [h.1] at h; exact h.2) fun _ => ?_
  refine Tr.bind Tr.lift fun _ => ?_
  refine Tr.bind Tr.lift fun _ => ?_
  refine Tr.bind Tr.lift fun _ => ?_
  refine Tr.bind Tr.lift fun _ => ?_
  exact Tr.pure fun st h => h.1.1.1.1


theorem nextOod_tr (cx : Ctx) (P : KV → KV × KV × KV → Prop) : Tr (SQ P) (fun _ => SQ P) (nextOod cx) := by
  unfold nextOod
  refine Tr.bind (K := fun st0 st => st = st0 ∧ SQ P st) (Tr.get fun st h => ⟨rfl, h⟩) fun st0 => ?_
  refine Tr.ite (fun _ => Tr.bind (K := fun _ _ => False) Tr.throw fun _ => Tr.false) fun _ => ?_
  refine Tr.bind (K := fun _ st => SQ P st) (Tr.set fun st h => by rw [h.1] at h; exact h.2) fun _ => ?_
  exact Tr.lift.mono (fun st h => h) fun _ _ h => h.1

/-- A value with a consistency sum in its support. -/
def HC (v : Val) : Prop := ∃ a, v = some a ∧ 0 < a.cons.size

theorem hc_vx_left {x y : Val} (h : HC x) : HC (vx x y) := by
  obtain ⟨a, rfl, ha⟩ := h
  cases y with
  | none => exact ⟨a, rfl, ha⟩
  | some b =>
    refine ⟨_, rfl, ?_⟩
    exact Array.size_pos_of_mem ((union_mem a.cons b.cons).1 _ (Array.getElem_mem ha))

theorem hc_vx_right {x y : Val} (h : HC y) : HC (vx x y) := by
  obtain ⟨b, rfl, hb⟩ := h
  cases x with
  | none => exact ⟨b, rfl, hb⟩
  | some a =>
    refine ⟨_, rfl, ?_⟩
    rcases (union_mem a.cons b.cons).2 _ (Array.getElem_mem hb) with h | ⟨y, hy, -⟩
    · exact Array.size_pos_of_mem h
    · exact Array.size_pos_of_mem hy

/-- Both components of `s` and of `q`'s middle carry a consistency sum. -/
def LP (s : KV) (q : KV × KV × KV) : Prop := HC s.1 ∧ HC s.2 ∧ HC q.2.1.1 ∧ HC q.2.1.2

theorem addScaled_tr (p : KV × KV × KV) : Tr (SQ LP) (fun _ => SQ LP) (addScaled p) := by
  unfold addScaled
  exact Tr.modify fun st h => ⟨h.1, h.2.1, hc_vx_left h.2.2.1, hc_vx_left h.2.2.2⟩

theorem Tr.intro_prop {β : Type} {I : σ → Prop} {J : β → σ → Prop} {m : SM σ β} {p : Prop}
    (h : p → Tr I J m) : Tr (fun st => I st ∧ p) J m :=
  fun st hI b st' hr => h hI.2 st hI.1 b st' hr

theorem Tr.const {β : Type} {p : Prop} {m : SM σ β} : Tr (fun _ => p) (fun _ _ => p) m :=
  fun _ h _ _ _ => h

theorem Tr.forInR {ρ : Type} {I : σ → Prop} (f : Nat → ρ → SM σ (ForInStep ρ))
    (hf : ∀ a r, Tr I (fun _ => I) (f a r)) (r : Std.Legacy.Range) (init : ρ) :
    Tr I (fun _ => I) (ForIn.forIn r init f) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact Tr.forIn f hf _ init

theorem hc_kc1 {x : KV} (h : HC x.1) : HC (kc x).1 := hc_vx_left h
theorem hc_kc2 {x : KV} (h : HC x.1) : HC (kc x).2 := hc_vx_left h

theorem challenge_tr (cx : Ctx) : Tr (SQ LP) (fun _ => SQ LP) (challenge cx) := by
  unfold challenge
  refine Tr.bind (K := fun st0 st => st = st0 ∧ SQ LP st) (Tr.get fun st h => ⟨rfl, h⟩) fun st0 => ?_
  generalize hq : st0.q = Q
  obtain ⟨a0, a1, a2⟩ := Q
  dsimp only
  refine Tr.bind (K := fun _ => SQ fun s _ => HC s.1 ∧ HC s.2) (Tr.modify fun st h => ?_) fun _ => ?_
  · obtain ⟨rfl, h⟩ := h
    have h1 : HC a1.1 := by have := h.2.2.1; rw [hq] at this; exact this
    exact ⟨hc_vx_left (hc_vx_right (hc_kc1 h1)), hc_vx_left (hc_vx_right (hc_kc2 h1))⟩
  · refine Tr.bind (nextMsg_tr cx _) fun mm => ?_
    exact Tr.modify fun st h => ⟨h.1, h.2, hc_vx_left h.1, hc_vx_left h.2⟩

theorem oodIntro_tr (cx : Ctx) (dim split : Nat) : Tr (SQ LP) (fun _ => SQ LP) (oodIntro cx dim split) := by
  unfold oodIntro
  refine Tr.bind (nextOod_tr cx LP) fun y => ?_
  refine Tr.bind (nextMsg_tr cx LP) fun mm => ?_
  refine Tr.bind (addScaled_tr _) fun _ => ?_
  exact Tr.modify fun st h => ⟨hc_vx_left h.1, hc_vx_left h.2.1, h.2.2.1, h.2.2.2⟩

theorem query_hc {cx : Ctx} {li foldLen : Nat} {e : KV} (h : query cx li foldLen = .ok e) : HC e.1 ∧ HC e.2 := by
  unfold VStar.Algebra.query at h
  split at h
  · rename_i lv _
    dsimp only at h
    split at h
    · cases h
    · simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      have : 0 < (Array.map (fun x => (li, x)) (Array.range (2 ^ lv.lanes))).size := by
        simp only [Array.size_map, Array.size_range]
        exact Nat.two_pow_pos _
      exact ⟨⟨_, rfl, this⟩, ⟨_, rfl, this⟩⟩
  · cases h

/-- **Ligerito's final sum carries a consistency sum in both components** (level 0's `query`, kept by every later
step). -/
theorem ligerito_hc (cx : Ctx) (tIn : Val) :
    Tr (fun _ => True) (fun s _ => HC s.1 ∧ HC s.2) (ligerito cx tIn) := by
  unfold ligerito
  dsimp only
  refine Tr.bind Tr.triv fun _ => ?_
  refine Tr.bind Tr.triv fun _ => ?_
  refine Tr.bind Tr.triv fun _ => ?_
  refine Tr.bind Tr.triv fun _ => ?_
  refine Tr.bind Tr.triv fun _ => ?_
  refine Tr.bind Tr.triv fun _ => ?_
  refine Tr.bind Tr.triv fun _ => ?_
  refine Tr.bind Tr.triv fun _ => ?_
  refine Tr.bind Tr.lift fun e => Tr.intro_prop fun he => ?_
  have hE := query_hc he
  refine Tr.bind Tr.triv fun mm => ?_
  refine Tr.bind (K := fun _ => SQ fun _ q => HC q.2.1.1 ∧ HC q.2.1.2)
    (by unfold addScaled; exact Tr.modify fun st _ => ⟨hc_vx_right (hc_vx_left hE.1), hc_vx_right (hc_vx_left hE.2)⟩)
    fun _ => ?_
  refine Tr.bind (K := fun _ => SQ LP)
    (Tr.modify fun st h => ⟨hc_vx_right hE.1, hc_vx_right hE.2, h.1, h.2⟩) fun _ => ?_
  refine Tr.bind (Tr.forInR _ (fun i r => ?_) _ _) fun r => ?_
  · try dsimp only
    refine Tr.ite (fun _ => Tr.bind (K := fun _ _ => False) Tr.throw fun _ => Tr.false) fun _ => ?_
    refine Tr.bind (Tr.forInR _ (fun _ _ => Tr.bind (challenge_tr cx) fun _ => Tr.pure fun _ h => h) _ _)
      fun _ => ?_
    refine Tr.ite (fun _ => ?_) fun _ => ?_
    · refine Tr.bind (oodIntro_tr cx _ _) fun _ => ?_
      refine Tr.bind (oodIntro_tr cx _ _) fun _ => ?_
      refine Tr.bind (Tr.lift.mono (fun st h => h) fun _ _ h => h.1) fun _ => ?_
      refine Tr.bind (nextMsg_tr cx LP) fun _ => ?_
      refine Tr.bind (addScaled_tr _) fun _ => ?_
      refine Tr.bind (K := fun _ => SQ LP) (Tr.modify fun st h => ⟨hc_vx_left h.1, hc_vx_left h.2.1, h.2.2.1, h.2.2.2⟩) fun _ => ?_
      exact Tr.pure fun _ h => h
    · refine Tr.bind (Tr.lift.mono (fun st h => h) fun _ _ h => h.1) fun _ => ?_
      refine Tr.bind (K := fun _ => SQ LP) (Tr.modify fun st h => ⟨hc_vx_left h.1, hc_vx_left h.2.1, h.2.2.1, h.2.2.2⟩) fun _ => ?_
      exact Tr.pure fun _ h => h
  · refine Tr.bind (K := fun st0 _ => SQ LP st0) (Tr.get fun st h => h) fun st0 => ?_
    repeat (first
      | exact Tr.pure fun _ h => ⟨h.1, h.2.1⟩
      | refine Tr.ite (fun _ => Tr.bind (K := fun _ _ => False) Tr.throw fun _ => Tr.false) fun _ => ?_
      | refine Tr.ite (fun _ => Tr.throw) fun _ => ?_
      | refine Tr.bind Tr.const fun _ => ?_)

theorem ebind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem ligerito_run_hc {cx : Ctx} {tIn : Val} {st : LState} {p : KV × LState}
    (h : (ligerito cx tIn).run st = .ok p) : HC p.1.1 := (ligerito_hc cx tIn st trivial p.1 p.2 h).1

theorem supports_cons {cx : Ctx} {syms : Array Sym} (h : supports cx = .ok syms) :
    ∃ f ∈ syms, 0 < f.cons.size := by
  unfold supports at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  have hl := ligerito_run_hc ‹(ligerito cx _).run _ = Except.ok _›
  obtain ⟨a, ha, hpos⟩ := hc_vx_right (x := ‹Val›) hl
  refine ⟨a, ?_, hpos⟩
  rw [Array.mem_map]
  refine ⟨_, Array.mem_append_right _ (Array.mem_def.2 List.mem_cons_self), ?_⟩
  rw [ha]; rfl

theorem buildRes_acc (cx : Ctx) (ri : Nat) (f : Sym) (hf : 0 < f.cons.size) :
    Tr (fun _ => True) (fun r _ => (ri = 2 + cx.sh.claims ∨ ri = 2 + cx.sh.claims + 1) ∧
      Term.w (cx.base - 2 + (ri - (2 + cx.sh.claims))) ∈ r.add) (buildRes cx ri f) := by
  have he : f.cons.isEmpty = false := by
    rw [Bool.eq_false_iff]; intro h; rw [Array.isEmpty_iff] at h; simp [h] at hf
  unfold buildRes
  simp only [he, Bool.not_false, ite_true]
  refine Tr.ite (fun hc => ?_) (fun _ => Tr.bind (K := fun _ _ => False) Tr.throw fun _ => Tr.false)
  have hri : ri = 2 + cx.sh.claims ∨ ri = 2 + cx.sh.claims + 1 := by simpa using hc
  refine Tr.bind Tr.triv fun _ => ?_
  refine Tr.bind Tr.triv fun _ => ?_
  refine Tr.ite (fun _ => ?_) (fun _ => Tr.pure fun _ _ => ⟨hri, by simp⟩)
  refine Tr.bind Tr.triv fun _ => ?_
  refine Tr.bind Tr.triv fun _ => ?_
  exact Tr.pure fun _ _ => ⟨hri, by simp⟩

theorem stateT_forIn_mem {ρ ε α β : Type} (g : α → StateT ρ (Except ε) β) :
    ∀ (L : List α) (out : Array β) (s : ρ) (res : Array β) (s' : ρ),
      (forIn L out (fun a r => do let v ← g a; pure (ForInStep.yield (r.push v)))).run s = .ok (res, s') →
      (∀ x ∈ out, x ∈ res) ∧ ∀ a ∈ L, ∃ s0 v s1, (g a).run s0 = .ok (v, s1) ∧ v ∈ res
  | [], out, s, res, s', h => by
    simp only [List.forIn_nil] at h
    cases h
    exact ⟨fun x hx => hx, fun a ha => by cases ha⟩
  | a :: L, out, s, res, s', h => by
    rw [List.forIn_cons] at h
    simp only [StateT.run, Bind.bind, StateT.bind, Except.bind] at h
    cases hg : g a s with
    | error e => rw [hg] at h; cases h
    | ok vs =>
      rw [hg] at h
      obtain ⟨v, s₁⟩ := vs
      obtain ⟨h1, h2⟩ := stateT_forIn_mem g L (out.push v) s₁ res s' h
      refine ⟨fun x hx => h1 x (Array.mem_push_of_mem _ hx), fun b hb => ?_⟩
      rcases List.mem_cons.1 hb with rfl | hb
      · exact ⟨s, v, s₁, hg, h1 v (Array.mem_push_self)⟩
      · exact h2 b hb

theorem build_acc {cx : Ctx} {lins : Array Sym} {gf : GfS} (h : build cx lins = .ok gf)
    (hf : ∃ f ∈ lins, 0 < f.cons.size) :
    gf.nw = cx.base ∧ ∃ r ∈ gf.res, ∃ d < 2, Term.w (cx.base - 2 + d) ∈ r.add := by
  unfold build at h
  dsimp only at h
  obtain ⟨⟨res, st⟩, hgo, h⟩ := ebind_ok h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨rfl, ?_⟩
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hgo
    simp only [bind_pure] at hgo
    obtain ⟨f, hfm, hfp⟩ := hf
    obtain ⟨ri, hri, rfl⟩ := Array.mem_iff_getElem.1 hfm
    obtain ⟨-, hm⟩ := stateT_forIn_mem _ _ _ _ _ _ hgo
    obtain ⟨s0, v, s1, hv, hvm⟩ := hm ri (by simp [Std.Legacy.Range.size]; omega)
    rw [getElem!_pos lins ri hri] at hv
    obtain ⟨hr, hw⟩ := buildRes_acc cx ri _ hfp s0 trivial v s1 hv
    exact ⟨v, by simpa using hvm, ri - (2 + cx.sh.claims), by omega, hw⟩

theorem checkRepS_sum {S : RepS} {w : Bool} (h : checkRepS S w = .ok ()) :
    (S.ports.map (·.2)).foldl (· + ·) 0 = S.gf.nw := by
  unfold checkRepS at h
  dsimp only at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hn
      simpa using hn

theorem spec_acc {sh : Shape} {S : RepS} (h : spec sh = .ok S) :
    (∃ ms, S.ports = ms ++ [("acc0", 2), ("acc1", 2)]) ∧ (S.ports.map (·.2)).foldl (· + ·) 0 = S.gf.nw ∧
      ∃ r ∈ S.gf.res, ∃ d < 2, Term.w (S.gf.nw - 2 + d) ∈ r.add := by
  unfold spec at h
  dsimp only at h
  obtain ⟨cx, -, h⟩ := ebind_ok h
  obtain ⟨syms, hsy, h⟩ := ebind_ok h
  obtain ⟨gf, hgf, h⟩ := ebind_ok h
  obtain ⟨hnw, hacc⟩ := build_acc hgf (supports_cons hsy)
  obtain ⟨ports, -, h⟩ := ebind_ok h
  obtain ⟨u, hc, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  refine ⟨⟨_, rfl⟩, checkRepS_sum hc, ?_⟩
  rw [hnw]
  exact hacc

def pstep (r : Array Nat) (p : String × Nat) : Array Nat := r.push (r.back! + p.2)

def PInv (r : Array Nat) : Prop := ∀ k (hk : k < r.size), r[k] ≤ r.back!

theorem pinv_step {r : Array Nat} (h : PInv r) (p : String × Nat) : PInv (pstep r p) := by
  intro k hk
  unfold pstep at hk ⊢
  rw [Array.back!_push]
  rw [Array.size_push] at hk
  by_cases hk' : k < r.size
  · rw [Array.getElem_push_lt hk']
    have := h k hk'
    omega
  · have : k = r.size := by omega
    subst this
    rw [Array.getElem_push_eq]

theorem pfx_gen : ∀ (l : List (String × Nat)) (init : Array Nat), PInv init →
    (l.foldl pstep init).size = init.size + l.length ∧
      (l.foldl pstep init).back! = l.foldl (fun a p => a + p.2) init.back! ∧ PInv (l.foldl pstep init)
  | [], init, h => ⟨rfl, rfl, h⟩
  | p :: l, init, h => by
    obtain ⟨h1, h2, h3⟩ := pfx_gen l (pstep init p) (pinv_step h p)
    simp only [List.foldl_cons]
    refine ⟨?_, ?_, h3⟩
    · rw [h1]; unfold pstep; simp; omega
    · rw [h2]; unfold pstep; rw [Array.back!_push]

theorem pfx (l : List (String × Nat)) :
    (l.foldl pstep #[0]).size = l.length + 1 ∧
      (l.foldl pstep #[0]).back! = (l.map (·.2)).foldl (· + ·) 0 ∧ PInv (l.foldl pstep #[0]) := by
  have h0 : PInv #[0] := fun k hk => by simp at hk; subst hk; simp
  obtain ⟨h1, h2, h3⟩ := pfx_gen l #[0] h0
  refine ⟨by rw [h1]; simp; omega, ?_, h3⟩
  rw [h2, List.foldl_map]
  rfl

theorem portOf_acc1 (ms : List (String × Nat)) (nw d : Nat) (hd : d < 2)
    (hsum : ((ms ++ [("acc0", 2), ("acc1", 2)]).map (fun p : String × Nat => p.2)).foldl (· + ·) 0 = nw) :
    ((ms ++ [("acc0", 2), ("acc1", 2)]).foldl pstep #[0]).findIdx? (· > nw - 2 + d) = some (ms.length + 2) ∧
      ((ms ++ [("acc0", 2), ("acc1", 2)]).foldl pstep #[0]).size = ms.length + 3 := by
  have hl : ms ++ [("acc0", 2), ("acc1", 2)] = (ms ++ [("acc0", 2)]) ++ [("acc1", 2)] := by simp
  rw [hl] at hsum ⊢
  rw [List.map_append, List.foldl_append] at hsum
  obtain ⟨h1, h2, h3⟩ := pfx (ms ++ [("acc0", 2)])
  simp only [List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil] at hsum
  rw [← h2] at hsum
  simp only [List.length_append, List.length_cons, List.length_nil] at h1
  rw [List.foldl_append, List.foldl_cons, List.foldl_nil]
  generalize List.foldl pstep #[0] (ms ++ [("acc0", 2)]) = R at h1 h3 hsum ⊢
  have hp : pstep R ("acc1", 2) = R.push (R.back! + 2) := rfl
  rw [hp]
  refine ⟨?_, by simp [h1]⟩
  rw [Array.findIdx?_eq_some_iff_getElem]
  have hR : ms.length + 2 = R.size := by omega
  refine ⟨by simp [h1], ?_, fun j hj => ?_⟩
  · simp only [hR, Array.getElem_push_eq, decide_eq_true_eq]
    omega
  · have hj' : j < R.size := by omega
    rw [Array.getElem_push_lt hj']
    have := h3 j hj'
    simp only [decide_eq_true_eq]
    omega

theorem id_forIn_pres {α ρ : Type} (M : ρ → Prop) (f : α → ρ → Id (ForInStep ρ))
    (hf : ∀ a r, ∃ r', f a r = .yield r' ∧ (M r → M r')) :
    ∀ (L : List α) (init : ρ), M init → M (forIn L init f : Id ρ)
  | [], init, h => h
  | a :: L, init, h => by
    obtain ⟨r', hr, hm⟩ := hf a init
    have := id_forIn_pres M f hf L r' (hm h)
    rw [List.forIn_cons, hr]
    exact this

theorem id_forIn_reach {α ρ : Type} (M : ρ → Prop) (f : α → ρ → Id (ForInStep ρ)) (a0 : α)
    (hf : ∀ a r, ∃ r', f a r = .yield r' ∧ (M r → M r') ∧ (a = a0 → M r')) :
    ∀ (L : List α) (init : ρ), (a0 ∈ L ∨ M init) → M (forIn L init f : Id ρ)
  | [], init, h => by
    rcases h with h | h
    · cases h
    · exact h
  | a :: L, init, h => by
    obtain ⟨r', hr, hm, he⟩ := hf a init
    have : a0 ∈ L ∨ M r' := by
      rcases h with h | h
      · rcases List.mem_cons.1 h with h | h
        · exact .inr (he h.symm)
        · exact .inl h
      · exact .inr (hm h)
    have := id_forIn_reach M f a0 hf L r' this
    rw [List.forIn_cons, hr]
    exact this

def ddStep (acc : Array Nat) (x : Nat) : Array Nat := if acc.back? == some x then acc else acc.push x

theorem dd_mem : ∀ (l : List Nat) (acc : Array Nat) (y : Nat), (y ∈ l ∨ y ∈ acc) → y ∈ l.foldl ddStep acc
  | [], acc, y, h => by
    rcases h with h | h
    · cases h
    · exact h
  | x :: l, acc, y, h => by
    rw [List.foldl_cons]
    apply dd_mem l
    have hk : y ∈ acc → y ∈ ddStep acc x := fun hy => by
      unfold ddStep; split
      · exact hy
      · exact Array.mem_push_of_mem _ hy
    rcases h with h | h
    · rcases List.mem_cons.1 h with rfl | h
      · refine .inr ?_
        unfold ddStep
        split
        · rename_i hb
          have hb' : acc.back? = some y := by simpa using hb
          rw [Array.back?_eq_getElem?] at hb'
          exact Array.mem_of_getElem? hb'
        · exact Array.mem_push_self
      · exact .inl h
    · exact .inr (hk h)

theorem sortDedup_mem (hq : ∀ (a : Array Nat) y, y ∈ a → y ∈ a.qsort (· < ·)) {a : Array Nat} {y : Nat}
    (h : y ∈ a) : y ∈ sortDedup a := by
  unfold sortDedup
  rw [← Array.foldl_toList]
  exact dd_mem _ _ _ (.inl (Array.mem_toList_iff.2 (hq a y h)))

theorem touched_w (hq : ∀ (a : Array Nat) y, y ∈ a → y ∈ a.qsort (· < ·)) (S : RepS) (starts : Array Nat)
    (r : Res) (i : Nat) (hi : Term.w i ∈ r.add) :
    (starts.findIdx? (· > i)).getD starts.size - 1 ∈ (touched S starts r).1 := by
  unfold touched
  simp only [Id.run]
  refine sortDedup_mem hq (id_forIn_reach (fun s : Array Nat × Array Nat => (starts.findIdx? (· > i)).getD starts.size - 1 ∈ s.1) _
    (Term.w i) ?_ _ _ (.inl (List.mem_append_left _ hi)))
  intro t st
  cases t with
  | w j =>
    refine ⟨_, rfl, fun h => Array.mem_push_of_mem _ h, fun h => ?_⟩
    cases h
    exact Array.mem_push_self
  | v j => exact ⟨_, rfl, fun h => h, fun h => by cases h⟩
  | wt j u =>
    refine ⟨_, rfl, fun h => ?_, fun h => by cases h⟩
    dsimp only
    rw [Std.Legacy.Range.forIn_eq_forIn_range']
    exact id_forIn_pres (fun s => _ ∈ s) _ (fun a r => ⟨_, rfl, fun h => Array.mem_push_of_mem _ h⟩) _ _ h

theorem ex_forIn_reach {ε α ρ : Type} (M : ρ → Prop) (f : α → ρ → Except ε (ForInStep ρ)) (a0 : α)
    (hf : ∀ a r y, f a r = .ok y → ∃ r', y = .yield r' ∧ (M r → M r') ∧ (a = a0 → M r')) :
    ∀ (L : List α) (init v : ρ), (a0 ∈ L ∨ M init) → forIn L init f = .ok v → M v
  | [], init, v, h, hv => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at hv
    subst hv
    rcases h with h | h
    · cases h
    · exact h
  | a :: L, init, v, h, hv => by
    rw [List.forIn_cons] at hv
    obtain ⟨y, hy, hv⟩ := ebind_ok hv
    obtain ⟨r', rfl, hm, he⟩ := hf a init y hy
    refine ex_forIn_reach M f a0 hf L r' v ?_ hv
    rcases h with h | h
    · rcases List.mem_cons.1 h with h | h
      · exact .inr (he h.symm)
      · exact .inl h
    · exact .inr (hm h)

theorem parts_acc1 (hq : ∀ (a : Array Nat) y, y ∈ a → y ∈ a.qsort (· < ·)) {S : RepS}
    (hports : ∃ ms, S.ports = ms ++ [("acc0", 2), ("acc1", 2)])
    (hsum : (S.ports.map (·.2)).foldl (· + ·) 0 = S.gf.nw)
    (hacc : ∃ r ∈ S.gf.res, ∃ d < 2, Term.w (S.gf.nw - 2 + d) ∈ r.add)
    {ps : Array Part} (h : parts S = .ok ps) : ∃ pt ∈ ps, ("acc1", 2) ∈ pt.S.ports := by
  obtain ⟨ms, hms⟩ := hports
  obtain ⟨r, hr, d, hd, hw⟩ := hacc
  unfold parts at h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨starts, hst, h⟩ := ebind_ok h
  obtain ⟨G, hG, h⟩ := ebind_ok h
  obtain ⟨out, hout, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rw [← Array.forIn_toList, List.forIn_pure_yield_eq_foldl] at hst
  simp only [pure, Except.pure, Except.ok.injEq, List.toList_toArray] at hst
  have hst' : starts = (ms ++ [("acc0", 2), ("acc1", 2)]).foldl pstep #[0] := by rw [← hst, hms]; rfl
  rw [hms] at hsum
  obtain ⟨hfi, -⟩ := portOf_acc1 ms S.gf.nw d hd hsum
  rw [← hst'] at hfi
  have hx : ∀ r ∈ S.gf.res, Term.w (S.gf.nw - 2 + d) ∈ r.add → ms.length + 1 ∈ (touched S starts r).1 := by
    intro r _ hw
    have := touched_w hq S starts r _ hw
    rw [hfi] at this
    simpa using this
  obtain ⟨ir, hir, hrr⟩ := List.mem_iff_getElem.1 hr
  have hP : ms.length + 1 ∈ (touched S starts S.gf.res.toArray[ir]!).1 := by
    have : S.gf.res.toArray[ir]! = r := by
      rw [getElem!_pos S.gf.res.toArray ir (by simpa using hir)]
      simpa using hrr
    rw [this]
    exact hx r hr hw
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hG
  have hGx : ∃ j, ∃ hj : j < G.size, ms.length + 1 ∈ G[j].2.1 := by
    refine ex_forIn_reach (fun G : Array (Array Nat × Array Nat × Array Nat) =>
      ∃ j, ∃ hj : j < G.size, ms.length + 1 ∈ G[j].2.1) _ ir ?_ _ _ _ (.inl ?_) hG
    · intro a G y hy
      split at hy
      · rename_i gi hgi
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        subst hy
        obtain ⟨hgs, -, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hgi
        refine ⟨_, rfl, fun ⟨j, hj, hjx⟩ => ?_, fun ha => ?_⟩
        · refine ⟨j, by simp [Array.set!_eq_setIfInBounds]; exact hj, ?_⟩
          simp only [Array.set!_eq_setIfInBounds]
          rw [Array.getElem_setIfInBounds hj]
          split
          · rename_i hij
            subst hij
            rw [getElem!_pos G _ hgs]
            exact (union_mem _ _).1 _ hjx
          · exact hjx
        · subst ha
          refine ⟨gi, by simp [Array.set!_eq_setIfInBounds]; exact hgs, ?_⟩
          simp only [Array.set!_eq_setIfInBounds]
          rw [Array.getElem_setIfInBounds hgs, if_pos rfl]
          rcases (union_mem _ _).2 _ hP with h | ⟨y, hy, hc⟩
          · exact h
          · rw [Nat.compare_eq_eq] at hc
            subst hc
            exact hy
      · split at hy
        · simp only [pure, Except.pure, Except.ok.injEq] at hy
          subst hy
          refine ⟨_, rfl, fun ⟨j, hj, hjx⟩ => ⟨j, by simp; omega, ?_⟩, fun ha => ⟨G.size, by simp, ?_⟩⟩
          · rw [Array.getElem_push_lt hj]; exact hjx
          · subst ha
            rw [Array.getElem_push_eq]
            exact hP
        · obtain ⟨_, h1, -⟩ := ebind_ok hy
          cases h1
    · simp only [Std.Legacy.Range.size, List.mem_range'_1]
      simp at hir ⊢
      omega
  obtain ⟨j, hj, hjx⟩ := hGx
  rw [← Array.forIn_toList] at hout
  refine ex_forIn_reach (fun out : Array Part => ∃ pt ∈ out, ("acc1", 2) ∈ pt.S.ports) _ G[j] ?_ _ _ _
    (.inl (Array.mem_toList_iff.2 (Array.getElem_mem hj))) hout
  intro a o y hy
  repeat' (first | (obtain ⟨_, _, hy⟩ := ebind_ok hy) | (split at hy))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    refine ⟨_, rfl, fun ⟨pt, hpt, h⟩ => ⟨pt, Array.mem_push_of_mem _ hpt, h⟩, fun ha => ⟨_, Array.mem_push_self, ?_⟩⟩
    subst ha
    rw [List.mem_map]
    refine ⟨ms.length + 1, Array.mem_toList_iff.2 hjx, ?_⟩
    simp [hms]

theorem spec_parts_acc1 (hq : ∀ (a : Array Nat) y, y ∈ a → y ∈ a.qsort (· < ·)) {sh : Shape} {S : RepS}
    {ps : Array Part} (hS : spec sh = .ok S) (hp : parts S = .ok ps) : ∃ pt ∈ ps, ("acc1", 2) ∈ pt.S.ports :=
  let ⟨h1, h2, h3⟩ := spec_acc hS
  parts_acc1 hq h1 h2 h3 hp

theorem inner_at {L j : Nat} {f : Nat → Stmt} {ps : Array Part} {g : Nat → String} {K : Array Stmt}
    (hj : j < ps.size) :
    (((List.range L).map f).toArray ++ ((List.range ps.size).map fun j => Stmt.inner j (g j) ps[j]!).toArray ++ K)[L + j]? =
      some (Stmt.inner j (g j) ps[j]!) := by
  rw [Array.getElem?_append_left (by simp; omega), Array.getElem?_append_right (by simp)]
  simp [hj]

end RecDeploy.AccEnd

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)

/-- **Some algebra part reads `acc1`**, at every dense shape the reader reads. -/
theorem accEndHeldW : AccEndHeldW := by
  intro x r _ hh hsp ss hss
  obtain ⟨A, K, n0, hA, -, rfl⟩ := OpenLower.statements_dense hss hsp
  rw [hh] at hA
  obtain ⟨sch, sp, ps, -, hsp', hps, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
  have hsp'' : VStar.Algebra.spec ⟨r.shape.m, r.shape.kLog, r.shape.claims⟩ = .ok sp := by
    unfold VStar.Algebra.specOf at hsp'
    simpa using hsp'
  obtain ⟨pt, hpt, hacc⟩ := AccEnd.spec_parts_acc1
    (fun a y h => (FlockSoundness.Audit.DrawExec.mem_qsort a _ y).2 h) hsp'' hps
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hpt
  refine ⟨_, j, _, ps[j]!, AccEnd.inner_at hj, ?_⟩
  rw [getElem!_pos ps j hj]
  exact List.mem_map.2 ⟨_, hacc, rfl⟩

/-- **`StepsWG` at `XsW … AccV` at KS's brings**, with no hypothesis. -/
theorem stepsWV_deployedA (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE w : Msg → ℝ) (b : Brings)
    (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record)
    (rest : Msg → List (Msg × Claims.Out) → List UInt8) (cmt : Msg → ℕ → Msg → Msg) :
    StepsWG DeployedR.rdR key wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest) OpenRows.elemR
      Claims.Recursion.Form.Dense recRead cmt
      (XsW DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest) OpenRows.elemR
        (AccV DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest) OpenRows.elemR
          Claims.Recursion.Form.Dense recRead cmt)) :=
  stepsWV_deployedE key wE w b aux nOf hd rest cmt accEndHeldW

/-- **P3 at KS's brings**, with no hypothesis. -/
theorem accPorts_deployed (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE w : Msg → ℝ) (b : Brings)
    (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record)
    (rest : Msg → List (Msg × Claims.Out) → List UInt8) (cmt : Msg → ℕ → Msg → Msg) :
    AccPorts DeployedR.rdR key wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest) OpenRows.elemR
      Claims.Recursion.Form.Dense recRead cmt :=
  accPorts_deployedE key wE w b aux nOf hd rest cmt accEndHeldW

end RecDeploy.ResLower
