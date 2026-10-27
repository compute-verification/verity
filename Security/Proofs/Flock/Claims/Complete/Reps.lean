import Proofs.Flock.Claims.Complete.Runs
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Arith

/-!
# The honest reps return

`proverRuns`: the hypothesis `ProverRuns` holds, so `sessionRuns_goal` proves the goal `SessionRuns` with no premise
(`sessionRuns_of`).
No coin makes a rep throw. The one refusal on the rep's path is the lincheck's fold (`Stmt.fold`), and what it asks is
the statement's: `ρ` of `k_log − 6` coins (the zerocheck's has `m − 6`, `zerocheckP_tot`), 64 weights
(`skipWeights_size`), and each range's slot type its fold's (`SlotOk`), which `HmRow.check` establishes
(`hm_check_kinds`) for the scope's untyped row circuit (`Case.table`). The level-0 and recursive commitments return
through `Hashes.ofSha512` (`commitP_ok`), and every other step is a draw, an absorb or a `pure`.

`Total M`: the computations of `M` that return, with a postcondition (`T x P`), closed under `pure`, `bind` and
returning lifts, and lifted over a state (`Total.st`). `totS` is it on `PS` from any sends; `totT` and `totL` add the
transcript and Ligerito's state. The walks apply the steps whose postcondition is `True` (`Total.bind'`, …).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy FlockSoundness.Refine

/-- **Returning** on a verdict monad's computations: `T x P`, `x` returns, with `P` of its value. -/
structure Total (M : Type → Type) [Monad M] [MonadLiftT V M] where
  T : {α : Type} → M α → (α → Prop) → Prop
  mono : ∀ {α : Type} {x : M α} {P Q : α → Prop}, T x P → (∀ a, P a → Q a) → T x Q
  pure : ∀ {α : Type} {P : α → Prop} (a : α), P a → T (Pure.pure a) P
  bind : ∀ {α β : Type} {x : M α} {f : α → M β} {Q : α → Prop} {P : β → Prop}, T x Q → (∀ a, Q a → T (f a) P) →
    T (x >>= f) P
  lift : ∀ {α : Type} {P : α → Prop} (v : V α) (a : α), v = .ok a → P a → T (monadLift v : M α) P

namespace Total

/-- `C` on the runs from each state of `σ`, on their values. -/
def st {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) (σ : Type) : Total (StateT σ M) where
  T x P := ∀ s, C.T (x s) fun p => P p.1
  mono h hq s := C.mono (h s) fun p hp => hq p.1 hp
  pure a h s := C.pure (a, s) h
  bind hx hf s := C.bind (hx s) fun p hp => hf p.1 hp p.2
  lift v a hv h s := C.bind (C.lift v a hv h) fun b hb => C.pure (b, s) hb

/-! The walk's steps, each returning with `True`. -/

theorem bind' {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {α β : Type} {x : M α} {f : α → M β}
    {P : β → Prop} (hx : C.T x fun _ => True) (hf : ∀ a, C.T (f a) P) : C.T (x >>= f) P :=
  C.bind hx fun a _ => hf a

theorem pure' {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {α : Type} (a : α) :
    C.T (Pure.pure a) fun _ => True :=
  C.pure a trivial

theorem lift_st {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {σ α : Type} {x : M α}
    (h : C.T x fun _ => True) : (C.st σ).T (StateT.lift x) fun _ => True :=
  fun s => C.bind h fun a _ => C.pure (a, s) trivial

theorem get {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {σ : Type} :
    (C.st σ).T (get : StateT σ M σ) fun _ => True :=
  fun s => C.pure (s, s) trivial

theorem set {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {σ : Type} (s' : σ) :
    (C.st σ).T (set s' : StateT σ M PUnit) fun _ => True :=
  fun _ => C.pure (PUnit.unit, s') trivial

theorem modify {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {σ : Type} (f : σ → σ) :
    (C.st σ).T (modify f : StateT σ M PUnit) fun _ => True :=
  fun s => C.pure (PUnit.unit, f s) trivial

theorem ite {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {α : Type} {c : Prop} [Decidable c]
    {x y : M α} {P : α → Prop} (hx : c → C.T x P) (hy : ¬c → C.T y P) : C.T (if c then x else y) P := by
  by_cases hc : c
  · rw [ite_eq_left hc]
    exact hx hc
  · rw [ite_eq_right hc]
    exact hy hc

theorem forIn_list {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {α β : Type} (l : List α) (b : β)
    {f : α → β → M (ForInStep β)} (h : ∀ a ∈ l, ∀ b, C.T (f a b) fun _ => True) : C.T (forIn l b f) fun _ => True := by
  induction l generalizing b with
  | nil => exact C.pure b trivial
  | cons a l ih =>
    rw [List.forIn_cons]
    refine C.bind (h a List.mem_cons_self b) fun r _ => ?_
    cases r with
    | done b => exact C.pure b trivial
    | yield b => exact ih b fun a ha => h a (List.mem_cons_of_mem _ ha)

theorem forIn_range {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {β : Type} (r : Std.Legacy.Range)
    (b : β) {f : Nat → β → M (ForInStep β)} (h : ∀ a b, C.T (f a b) fun _ => True) :
    C.T (forIn r b f) fun _ => True := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact C.forIn_list _ b fun a _ => h a

theorem forIn_array {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {α β : Type} (xs : Array α) (b : β)
    {f : α → β → M (ForInStep β)} (h : ∀ a ∈ xs, ∀ b, C.T (f a b) fun _ => True) :
    C.T (forIn xs b f) fun _ => True := by
  rw [← Array.forIn_toList]
  exact C.forIn_list xs.toList b fun a ha => h a (Array.mem_toList_iff.1 ha)

/-- A loop over `[0:n]` that keeps `I`, at the steps taken. -/
theorem forIn_upto {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {β : Type} (n : Nat) (b : β)
    {f : Nat → β → M (ForInStep β)} (I : Nat → β → Prop) (h0 : I 0 b)
    (h : ∀ i < n, ∀ b, I i b → C.T (f i b) fun r => ∃ b', r = .yield b' ∧ I (i + 1) b') :
    C.T (forIn [0:n] b f) (I n) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices ∀ j s (b : β), s + j = n → I s b → C.T (forIn (List.range' s j 1) b f) (I n) from this n 0 b (by omega) h0
  intro j
  induction j with
  | zero =>
    intro s b hs hb
    rw [List.range'_zero, List.forIn_nil]
    exact C.pure b (by rw [← hs]; exact hb)
  | succ j ih =>
    intro s b hs hb
    rw [List.range'_succ, List.forIn_cons]
    refine C.bind (h s (by omega) b hb) fun r hr => ?_
    obtain ⟨b', rfl, hb'⟩ := hr
    exact ih (s + 1) b' (by omega) hb'

/-- A loop over `[0:n]` that keeps `I`, then a computation from its end. -/
theorem forIn_bind_upto {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {β γ : Type} (n : Nat) (b : β)
    {f : Nat → β → M (ForInStep β)} {g : β → M γ} {P : γ → Prop} (I : Nat → β → Prop) (h0 : I 0 b)
    (h : ∀ i < n, ∀ b, I i b → C.T (f i b) fun r => ∃ b', r = .yield b' ∧ I (i + 1) b')
    (hg : ∀ s, I n s → C.T (g s) P) : C.T (forIn [0:n] b f >>= g) P :=
  C.bind (C.forIn_upto n b I h0 h) hg

theorem run {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {σ α : Type} {x : StateT σ M α}
    (h : (C.st σ).T x fun _ => True) (s : σ) : C.T (x.run s) fun _ => True :=
  C.mono (h s) fun _ _ => trivial

theorem liftTM {M : Type → Type} [Monad M] [MonadLiftT V M] (C : Total M) {α : Type} (x : TM α)
    (hx : ∀ t, ∃ p, x t = .ok p) : (C.st Transcript).T (Flock.liftTM x) fun _ => True :=
  fun t => let ⟨p, hp⟩ := hx t; C.lift (x t) p hp trivial

end Total

/-- **The pure session's computations that return** from any sends. -/
def totS : Total PS where
  T y P := ∀ l, ∃ a l', y.go l = (.ok a, l') ∧ P a
  mono h hq l := let ⟨a, l', e, hp⟩ := h l; ⟨a, l', e, hq a hp⟩
  pure a h l := ⟨a, l, rfl, h⟩
  bind {α β x f Q P} hx hf l := by
    obtain ⟨a, l', e, hq⟩ := hx l
    obtain ⟨b, l'', e', hp⟩ := hf a hq l'
    exact ⟨b, l'', by rw [go_bind, e]; exact e', hp⟩
  lift v a hv h l := ⟨a, l, by rw [monadLift_go, hv], h⟩

/-- Returning over the transcript. -/
abbrev totT : Total (TMW PS) := totS.st Transcript

/-- Returning over Ligerito's state too. -/
abbrev totL : Total (LP PS) := totT.st LigState

/-! ## The transcript's and Ligerito's steps return -/

theorem squeezeP_tot (resp : ℕ → Claims.Msg) (kind : UInt8) (n : ℕ) :
    totT.T (squeezeP (asksS resp) kind n) fun _ => True := by
  intro t
  unfold squeezeP
  split
  · exact totS.pure' _
  · exact totS.bind' (fun l => ⟨_, _, rfl, trivial⟩) fun _ => totS.pure' _

theorem sampleP_tot (resp : ℕ → Claims.Msg) : totT.T (sampleP (asksS resp)) fun _ => True :=
  totT.bind' (squeezeP_tot resp 1 1) fun _ => totT.pure' _

theorem sampleVecP_tot (resp : ℕ → Claims.Msg) (n : ℕ) : totT.T (sampleVecP (asksS resp) n) fun _ => True :=
  squeezeP_tot resp 2 n

theorem sampleKP_tot (resp : ℕ → Claims.Msg) : totT.T (sampleKP (asksS resp)) fun _ => True :=
  totT.bind' (squeezeP_tot resp 2 2) fun _ => totT.pure' _

theorem liftT_tot {α : Type} {x : TMW PS α} (h : totT.T x fun _ => True) : totL.T (liftT x) fun _ => True :=
  totT.lift_st h

theorem sendMsg_tot (m : F256 × F256) : totL.T (sendMsg (M := PS) m) fun _ => True := by
  unfold sendMsg
  repeat (first
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply Total.pure')
    | (with_reducible apply Total.modify)
    | (with_reducible apply liftT_tot)
    | (with_reducible apply Total.liftTM; intro _; exact ⟨_, rfl⟩))

theorem powP_tot (nonce : ℕ → ℕ → UInt64) (kind : ℕ) : totL.T (powP (M := PS) nonce kind) fun _ => True := by
  unfold powP
  repeat (first
    | extract_lets
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply Total.get)
    | (with_reducible apply Total.set)
    | (with_reducible apply liftT_tot)
    | (with_reducible apply Total.liftTM; intro _; exact ⟨_, rfl⟩))

theorem challengeP_tot (resp : ℕ → Claims.Msg) (split : Bool) :
    totL.T (challengeP (asksS resp) split) fun _ => True := by
  unfold challengeP
  repeat (first
    | extract_lets
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply Total.pure')
    | (with_reducible apply Total.get)
    | (with_reducible apply Total.modify)
    | (with_reducible apply sendMsg_tot)
    | (with_reducible apply liftT_tot)
    | (with_reducible apply sampleKP_tot)
    | (with_reducible apply Total.ite)
    | split)

theorem oodIntroP_tot (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    totL.T (oodIntroP (asksS resp) nonce g dim) fun _ => True := by
  unfold oodIntroP
  repeat (first
    | extract_lets
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply Total.modify)
    | (with_reducible apply sendMsg_tot)
    | (with_reducible apply powP_tot)
    | (with_reducible apply liftT_tot)
    | (with_reducible apply sampleP_tot)
    | (with_reducible apply sampleVecP_tot)
    | (with_reducible apply Total.liftTM; intro _; exact ⟨_, rfl⟩))

theorem queryP_tot (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    totL.T (queryP (asksS resp) nonce ms lv cm) fun _ => True := by
  unfold queryP
  repeat (first
    | extract_lets
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply Total.pure')
    | (with_reducible apply powP_tot)
    | (with_reducible apply liftT_tot)
    | (with_reducible apply sampleVecP_tot))

/-- **A level's commitment returns** in `PS` through `Hashes.ofSha512`. -/
theorem commitP_tot (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) : totS.T (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt) fun _ => True := fun l =>
  let ⟨cm, hcm⟩ := commitP_ok h ms lv g salt
  ⟨cm, l, by rw [commitP_pureS h ms lv g salt l, hcm], trivial⟩

/-- **Ligerito returns** where each level's commitment does. -/
theorem ligeritoP_tot (resp : ℕ → Claims.Msg) (hs : Hashes PS)
    (hc : ∀ ms lv g salt, totS.T (commitP hs ms lv g salt) fun _ => True) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    totT.T (ligeritoP (asksS resp) hs ms sch T cm0 t b salt nonce) fun _ => True := by
  unfold ligeritoP
  repeat (first
    | extract_lets
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply Total.pure')
    | (with_reducible apply Total.run)
    | (with_reducible apply Total.get)
    | (with_reducible apply Total.modify)
    | (with_reducible apply sendMsg_tot)
    | (with_reducible apply powP_tot)
    | (with_reducible apply challengeP_tot)
    | (with_reducible apply oodIntroP_tot)
    | (with_reducible apply queryP_tot)
    | (with_reducible apply liftT_tot)
    | (with_reducible apply Total.lift_st)
    | (with_reducible apply hc)
    | (with_reducible apply sampleP_tot)
    | (with_reducible apply sampleVecP_tot)
    | (with_reducible apply sampleKP_tot)
    | (with_reducible apply Total.forIn_range)
    | (with_reducible apply Total.forIn_array)
    | (with_reducible apply Total.ite)
    | (with_reducible apply Total.liftTM; intro _; exact ⟨_, rfl⟩)
    | split)

/-! ## The PIOP's provers return -/

/-- **The zerocheck returns**, its `ρ` one coin per residual round. -/
theorem zerocheckP_tot (resp : ℕ → Claims.Msg) (m : ℕ) (digest cap : ByteArray) (tr : Trace) :
    totT.T (zerocheckP (asksS resp) m digest cap tr) fun p => p.2.rho.size = m - 6 := by
  unfold zerocheckP
  dsimp only
  repeat (first
    | (refine Total.forIn_bind_upto _ _ _
        (fun i (s : Array F128 × Array F128 × Array (F128 × F128) × Array F128) => s.2.2.2.size = i) (by rfl)
        (fun i _ s hs => ?_) (fun s hs => ?_))
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply sampleP_tot)
    | (with_reducible apply sampleVecP_tot)
    | (with_reducible apply Total.liftTM; intro _; exact ⟨_, rfl⟩)
    | (exact totT.pure _ ⟨_, rfl, by simp [*]⟩)
    | (exact totT.pure _ ‹_›))

theorem fold_ne_error {c : CircuitFold} {lw ρ : Array F128} (hfold : ∀ α, ∃ v, c.fold α lw ρ = .ok v) {α : F128}
    {e : String} : c.fold α lw ρ ≠ .error e := by
  obtain ⟨v, hv⟩ := hfold α
  rw [hv]
  intro h
  cases h

/-- **The lincheck returns** where the statement's fold returns at the zerocheck's point. -/
theorem lincheckP_tot (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut)
    (hfold : ∀ α, ∃ v, c.fold α (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok v) :
    totT.T (lincheckP (asksS resp) c m tr zc) fun _ => True := by
  unfold lincheckP
  dsimp only
  repeat (first
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply Total.pure')
    | (with_reducible apply sampleP_tot)
    | (with_reducible apply Total.forIn_range)
    | (with_reducible apply Total.liftTM; intro _; exact ⟨_, rfl⟩)
    | (exfalso; exact fold_ne_error hfold ‹_›)
    | split)

/-- **The ring switch returns.** -/
theorem ringSwitchP_tot (resp : ℕ → Claims.Msg) (tr : Trace) (claims : Array Claim) :
    totT.T (ringSwitchP (asksS resp) tr claims) fun _ => True := by
  unfold ringSwitchP
  repeat (first
    | extract_lets
    | (with_reducible apply Total.bind')
    | (with_reducible intro _)
    | (with_reducible apply Total.pure')
    | (with_reducible apply sampleVecP_tot)
    | (with_reducible apply Total.forIn_range)
    | (with_reducible apply Total.liftTM; intro _; exact ⟨_, rfl⟩))

/-! ## The statement's fold returns

`Stmt.fold` refuses a point of the wrong size, or a range whose slot type it can't fold: a compression slot, or a net
range whose slots aren't its net's. `HmRow.check` refuses both ranges. -/

/-- What a range's slot type needs for its fold: it is no compression slot, and a net range's slots are its net's. -/
def SlotOk (c : Circuit) (r : Range) : Prop :=
  r.kind ≠ .comp ∧ ∀ i, r.kind = .net i → (c.nets.getD i default).2.unitLog = r.slotLog

/-- **A checked row circuit's ranges fold** (`HmRow.check`'s range loop). -/
theorem hm_check_kinds {c : Circuit} {tags : Tags} {ls : Lean.Json} (h : HmRow.check c tags ls = .ok ()) :
    ∀ r ∈ c.ranges.toList, SlotOk c r := by
  unfold HmRow.check at h
  dsimp only at h
  obtain ⟨_, hloop, -⟩ := ebind_ok h
  clear h
  rw [← Array.forIn_toList] at hloop
  refine list_forIn_each _ (SlotOk c) ?_ _ _ _ hloop
  clear hloop
  intro r s y hy
  repeat' (first | (obtain ⟨_, hx, hy⟩ := ebind_ok hy; try nomatch hx) | (split at hy))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    refine ⟨⟨fun hk => by simp_all, fun j hj => ?_⟩, _, rfl⟩
    simp_all

/-- The scope's untyped row circuit passed `HmRow.check`. -/
theorem circuitOf_check {tags : Tags} {file : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (ht : tags.typed = false) (h : ProvedScope.circuitOf tags file tables = .ok c) :
    ∃ ls, HmRow.check c tags ls = .ok () := by
  unfold ProvedScope.circuitOf at h
  split at h
  · unfold HmOut.parse at h
    simp only [ht, Bool.not_false, ↓reduceIte] at h
    obtain ⟨c', hc', h⟩ := ebind_ok h
    obtain ⟨_, -, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact hm_parse_check hc'
  · exact hm_parse_check h

theorem slotBase_ok (st : Stmt) (α : F128) (lw ρ : Array F128) {r : Range} (hr : SlotOk st.c r) :
    ∃ b, st.slotBase α lw ρ (r.kind, r.slotLog) = .ok b := by
  unfold Stmt.slotBase
  dsimp only
  split
  · exact absurd ‹_› hr.1
  · rename_i i hi
    rw [hr.2 i hi, bne_self_eq_false]
    exact ⟨_, rfl⟩
  · exact ⟨_, rfl⟩

theorem foldRanges_ok (st : Stmt) (α : F128) (lw ρ : Array F128) :
    ∀ (rs : List Range), (∀ r ∈ rs, SlotOk st.c r) → ∀ bases out, ∃ v, st.foldRanges α lw ρ rs bases out = .ok v
  | [], _, _, _ => ⟨_, rfl⟩
  | r :: rs, h, bases, out => by
    have ih := foldRanges_ok st α lw ρ rs (fun r' hr' => h r' (List.mem_cons_of_mem _ hr'))
    unfold Stmt.foldRanges
    dsimp only
    split
    · exact ih _ _
    · obtain ⟨b, hb⟩ := slotBase_ok st α lw ρ (h r List.mem_cons_self)
      rw [hb]
      exact ih _ _

/-- **The fold returns** at a point of the block's size, on ranges whose slot types it folds. -/
theorem fold_ok (st : Stmt) (hk : 6 ≤ st.c.kLog) (hs : ∀ r ∈ st.c.ranges.toList, SlotOk st.c r) (α : F128)
    {lw ρ : Array F128} (hlw : lw.size = 64) (hρ : ρ.size = st.c.kLog - 6) : ∃ v, st.fold α lw ρ = .ok v := by
  unfold Stmt.fold
  dsimp only
  have h1 : (ρ.size + 6 != st.c.kLog || lw.size != 64) = false := by
    simp [hρ, hlw]
    omega
  rw [h1]
  obtain ⟨v, hv⟩ := foldRanges_ok st α lw ρ _ hs #[] (Array.replicate (2 ^ st.c.kLog) 0)
  simp only [Bool.false_eq_true, ↓reduceIte, hv]
  exact ⟨_, rfl⟩

/-! ## A rep returns -/

/-- **A rep returns** on a statement whose fold returns at the zerocheck's point, on any level-0 commitment, extra
claims, salts and nonces. -/
theorem repP_tot (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Stmt) (hk : 6 ≤ st.c.kLog)
    (hkm : st.c.kLog ≤ st.m) (hs : ∀ r ∈ st.c.ranges.toList, SlotOk st.c r) (sch : Schedule) (tr : Trace)
    (t : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    totT.T (repP (asksS resp) (Hashes.ofSha512 h) (Setup.ofCircuit st) sch tr t cm0 extras salt nonce)
      fun _ => True := by
  unfold repP
  refine totT.bind (zerocheckP_tot resp _ _ _ tr) fun zz hz => ?_
  obtain ⟨zp, zc⟩ := zz
  dsimp only at hz ⊢
  have hfold : ∀ α, ∃ v, (Setup.ofCircuit st).fold.fold α (skipWeights zc.z)
      (zc.rho.extract 0 ((Setup.ofCircuit st).fold.kLog - 6)) = .ok v := fun α =>
    fold_ok st hk hs α (skipWeights_size _) (by
      simp only [Setup.ofCircuit, Stmt.circuitFold, Array.size_extract, hz]
      omega)
  refine totT.bind' (lincheckP_tot resp _ _ tr zc hfold) fun ll => ?_
  obtain ⟨lp, lc⟩ := ll
  dsimp only
  refine totT.bind' (ringSwitchP_tot resp tr _) fun rr => ?_
  obtain ⟨ss, T, sws⟩ := rr
  dsimp only
  exact totT.bind' (ligeritoP_tot resp _ (commitP_tot h) _ _ _ _ _ _ _ _) fun _ => totT.pure' _

/-- **The prover's reps return**: the hypothesis `ProverRuns`. -/
theorem proverRuns (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : ProverRuns H rel := by
  intro c st sch hst _ _ cm e _ _ salt nonce d l
  obtain ⟨st', hst', -, -, -, ⟨tags, file, tables, hty, hcirc⟩, hm, hk⟩ := c.table
  rw [hst] at hst'
  obtain rfl : st' = st := by simpa using hst'.symm
  obtain ⟨ls, hcheck⟩ := circuitOf_check hty hcirc
  obtain ⟨pt, l', e', -⟩ := repP_tot c.resp (Claims.hOf H) st' hk (by omega) (hm_check_kinds hcheck) sch _ _ cm e
    salt nonce (Transcript.init #[] d) l
  exact ⟨pt, l', e'⟩

/-- **The session runs**: the goal `SessionRuns`. -/
theorem sessionRuns_goal {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} :
    SessionRuns H rel :=
  sessionRuns_of (proverRuns H rel)

end FlockComplete.JsonRT
