import Proofs.Flock.Claims.ZK.FinishPiop

/-!
# The firewall's Ligerito is the masked prover's

`Flock.Firewall.Finish.ligeritoPZk` is `FlockComplete.ligeritoPZk` word for word on the port's own state, commitment
and sends (`LigState`, `Commit`, `Asks`, each the analysis's fields), its levels ≥ 1 committed by its argument `commit`
where the analysis's call `FlockComplete.commitP`. Its actions run on the analysis's state through `ligConj`, which
reads the port's state as the analysis's and writes it back:

* `ligConj_pure`, `ligConj_bind`, `ligConj_lift`, `ligConj_get`, `ligConj_modify`, `ligConj_set`, `ligConj_forIn_range`:
  `ligConj` goes through each step of a program;
* `ligConj_forIn_cm`: and through the levels' loop, whose carried commitment is the port's on one side and the
  analysis's on the other;
* `ligConj_sendMsg` to `ligConj_queryP`: each of the prover's actions is the analysis's under `ligConj`.
* **`ligeritoPZk_eq`**: where `commit` is `FlockComplete.commitP hs ms` through `Commit.toC`, the port's Ligerito is
  the analysis's; at SHA-512 and `hm96-sha512/v1` with `commitFw` (`ligeritoPZk_fw`), as `flock-firewall finish` runs
  it.
-/

namespace Flock.Firewall.Finish

open Transcript

/-- The port's Ligerito state as the analysis's: the same seven fields. -/
def LigState.toC (s : LigState) : FlockComplete.LigState := ⟨s.tab, s.w, s.msgs, s.oods, s.qn, s.cn, s.cbn⟩

/-- The analysis's Ligerito state as the port's. -/
def ligOfC (s : FlockComplete.LigState) : LigState := ⟨s.tab, s.w, s.msgs, s.oods, s.qn, s.cn, s.cbn⟩

/-- The analysis's commitment as the port's. -/
def cmOfC (c : FlockComplete.Commit) : Commit := ⟨c.rows, c.salts, c.layers, c.cap⟩

theorem cmOfC_toC (c : Commit) : cmOfC c.toC = c := rfl

theorem Commit.toC_cap (c : Commit) : c.toC.cap = c.cap := rfl

/-- A loop step's carried value mapped. -/
def stepMap {β β' : Type} (g : β → β') : ForInStep β → ForInStep β'
  | .done b => .done (g b)
  | .yield b => .yield (g b)

theorem map_ite' {F : Type → Type} [Functor F] {α β : Type} (f : α → β) (c : Prop) [Decidable c] (x y : F α) :
    f <$> (if c then x else y) = if c then f <$> x else f <$> y := by
  split <;> rfl

section Conj

variable {N : Type → Type} [Monad N] [LawfulMonad N]

/-- **A program on the analysis's Ligerito state, run on the port's.** -/
def ligConj {α : Type} (x : StateT FlockComplete.LigState N α) : StateT LigState N α :=
  fun s => (fun p => (p.1, ligOfC p.2)) <$> x s.toC

theorem ligConj_pure {α : Type} (a : α) : ligConj (pure a : StateT FlockComplete.LigState N α) = pure a := by
  funext s
  show (fun p => (p.1, ligOfC p.2)) <$> (pure (a, s.toC) : N _) = pure (a, s)
  rw [map_pure]; rfl

theorem ligConj_bind {α β : Type} (x : StateT FlockComplete.LigState N α)
    (f : α → StateT FlockComplete.LigState N β) : ligConj (x >>= f) = ligConj x >>= fun a => ligConj (f a) := by
  funext s
  show (fun p => (p.1, ligOfC p.2)) <$> (x s.toC >>= fun p => f p.1 p.2) =
    ((fun p => (p.1, ligOfC p.2)) <$> x s.toC) >>= fun p => (fun q => (q.1, ligOfC q.2)) <$> f p.1 p.2.toC
  rw [map_bind, bind_map_left]
  rfl

theorem ligConj_lift {α : Type} (y : N α) :
    ligConj (StateT.lift y : StateT FlockComplete.LigState N α) = StateT.lift y := by
  funext s
  show (fun p => (p.1, ligOfC p.2)) <$> (y >>= fun a => pure (a, s.toC)) = y >>= fun a => pure (a, s)
  rw [map_bind]
  simp only [map_pure]
  rfl

theorem ligConj_get :
    ligConj (get : StateT FlockComplete.LigState N FlockComplete.LigState) = get >>= fun s => pure s.toC := by
  funext s
  show (fun p => (p.1, ligOfC p.2)) <$> (pure (s.toC, s.toC) : N _) =
    (pure (s, s) : N _) >>= fun p => pure (p.1.toC, p.2)
  rw [map_pure, pure_bind]
  rfl

theorem ligConj_modify (f : FlockComplete.LigState → FlockComplete.LigState) :
    ligConj (modify f : StateT FlockComplete.LigState N PUnit) = modify fun s => ligOfC (f s.toC) := by
  funext s
  show (fun p => (p.1, ligOfC p.2)) <$> (pure (PUnit.unit, f s.toC) : N _) = pure (PUnit.unit, ligOfC (f s.toC))
  rw [map_pure]

theorem ligConj_set (t : FlockComplete.LigState) :
    ligConj (set t : StateT FlockComplete.LigState N PUnit) = set (ligOfC t) := by
  funext s
  show (fun p => (p.1, ligOfC p.2)) <$> (pure (PUnit.unit, t) : N _) = pure (PUnit.unit, ligOfC t)
  rw [map_pure]

omit [LawfulMonad N] in
theorem ligConj_ite {α : Type} (c : Prop) [Decidable c] (x y : StateT FlockComplete.LigState N α) :
    ligConj (if c then x else y) = if c then ligConj x else ligConj y := by
  split <;> rfl

theorem ligConj_throw {α : Type} [MonadExceptOf String N] (e : String) :
    ligConj (throw e : StateT FlockComplete.LigState N α) = throw e := ligConj_lift _

theorem run_toC {α : Type} (x : StateT FlockComplete.LigState N α) (s : LigState) :
    x.run s.toC = (fun p => (p.1, p.2.toC)) <$> (ligConj x).run s := by
  show x s.toC = (fun p => (p.1, p.2.toC)) <$> ((fun p => (p.1, ligOfC p.2)) <$> x s.toC)
  rw [Functor.map_map]
  exact (id_map _).symm

theorem lift_map {σ : Type} {α β : Type} (f : α → β) (x : N α) :
    (StateT.lift (f <$> x) : StateT σ N β) = f <$> StateT.lift x := by
  funext s
  show (StateT.lift (f <$> x)).run s = (f <$> StateT.lift x).run s
  rw [StateT.run_lift, StateT.run_map, StateT.run_lift, bind_map_left, map_bind]
  simp only [map_pure]

theorem ligConj_forIn_list {α β : Type} (l : List α) (b : β)
    (f : α → β → StateT FlockComplete.LigState N (ForInStep β)) :
    ligConj (forIn l b f) = forIn l b fun a b => ligConj (f a b) := by
  induction l generalizing b with
  | nil => simp only [List.forIn_nil]; exact ligConj_pure b
  | cons a l ih =>
    simp only [List.forIn_cons, ligConj_bind]
    congr 1
    funext r
    cases r with
    | done b => exact ligConj_pure b
    | yield b => exact ih b

theorem ligConj_forIn_range {β : Type} (n : ℕ) (b : β)
    (f : ℕ → β → StateT FlockComplete.LigState N (ForInStep β)) :
    ligConj (forIn [0:n] b f) = forIn [0:n] b fun a b => ligConj (f a b) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range', ligConj_forIn_list]

theorem ligConj_forIn_array {α β : Type} (xs : Array α) (b : β)
    (f : α → β → StateT FlockComplete.LigState N (ForInStep β)) :
    ligConj (forIn xs b f) = forIn xs b fun a b => ligConj (f a b) := by
  rw [← Array.forIn_toList, ← Array.forIn_toList, ligConj_forIn_list]

/-- **A loop whose carried value is mapped**: `g` with inverse `gi`, the analysis's loop from `g b` is `g` of the port's
from `b`. -/
theorem ligConj_forIn_iso {α β β' : Type} (g : β → β') (gi : β' → β) (hg : ∀ b', g (gi b') = b') (l : List α)
    (b : β) (f : α → β' → StateT FlockComplete.LigState N (ForInStep β')) :
    ligConj (forIn l (g b) f) = g <$> forIn l b (fun a r => stepMap gi <$> ligConj (f a (g r))) := by
  induction l generalizing b with
  | nil => simp only [List.forIn_nil, map_pure]; exact ligConj_pure _
  | cons a l ih =>
    simp only [List.forIn_cons, ligConj_bind, bind_map_left, map_bind]
    congr 1
    funext r
    cases r with
    | done b' => simp only [stepMap, map_pure, hg]; exact ligConj_pure b'
    | yield b' => simp only [stepMap]; rw [← ih (gi b'), hg]

/-- **The levels' loop**, which carries the commitment of the level last committed. -/
theorem ligConj_forIn_cm {γ : Type} (n k : ℕ) (c : Commit) (x : γ)
    (f : ℕ → ℕ × FlockComplete.Commit × γ → StateT FlockComplete.LigState N (ForInStep (ℕ × FlockComplete.Commit × γ))) :
    ligConj (forIn [0:n] (k, c.toC, x) f) = (fun r => (r.1, r.2.1.toC, r.2.2)) <$>
      forIn [0:n] (k, c, x) (fun a r => stepMap (fun r => (r.1, cmOfC r.2.1, r.2.2)) <$>
        ligConj (f a (r.1, r.2.1.toC, r.2.2))) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  exact ligConj_forIn_iso (fun r : ℕ × Commit × γ => (r.1, r.2.1.toC, r.2.2)) (fun r => (r.1, cmOfC r.2.1, r.2.2))
    (fun _ => rfl) _ (k, c, x) f

end Conj

section Parts

variable {M : Type → Type} [Monad M] [LawfulMonad M]

theorem ligConj_liftT [MonadExceptOf String M] [MonadLiftT V M] {α : Type} (x : TMW M α) :
    ligConj (FlockComplete.liftT x) = liftT x := ligConj_lift x

variable [MonadLiftT V M]

theorem ligConj_sendMsg [MonadExceptOf String M] (m : F256 × F256) :
    ligConj (FlockComplete.sendMsg (M := M) m) = sendMsg m := by
  simp only [FlockComplete.sendMsg, sendMsg, ligConj_bind, ligConj_modify, ligConj_liftT]
  rfl

theorem ligConj_powP [MonadExceptOf String M] (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    ligConj (FlockComplete.powP (M := M) nonce kind) = powP nonce kind := by
  rcases kind with _ | _ | kind <;>
  · simp only [FlockComplete.powP, powP, ligConj_bind, ligConj_get, ligConj_set, ligConj_liftT, bind_assoc, pure_bind]
    rfl

variable [MonadExceptOf String M]

theorem ligConj_challengeP (io : Asks M) (split : Bool) :
    ligConj (FlockComplete.challengeP io.toC split) = challengeP io split := by
  cases split <;>
  · simp only [FlockComplete.challengeP, challengeP, ligConj_bind, ligConj_liftT, ligConj_modify, ligConj_sendMsg,
      ligConj_pure, ligConj_get, bind_assoc, pure_bind, Bool.false_eq_true, ↓reduceIte]
    rfl

theorem ligConj_oodIntroP (io : Asks M) (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    ligConj (FlockComplete.oodIntroP io.toC nonce g dim) = oodIntroP io nonce g dim := by
  simp only [FlockComplete.oodIntroP, oodIntroP, ligConj_bind, ligConj_liftT, ligConj_modify, ligConj_sendMsg,
    ligConj_powP]
  rfl

theorem ligConj_queryP (io : Asks M) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    ligConj (FlockComplete.queryP io.toC nonce ms lv cm.toC) = queryP io nonce ms lv cm := by
  simp only [FlockComplete.queryP, queryP, ligConj_bind, ligConj_liftT, ligConj_powP, ligConj_pure]
  rfl

/-- **The firewall's masked Ligerito is the analysis's** where its levels ≥ 1 commit as `FlockComplete.commitP hs ms`
does. -/
theorem ligeritoPZk_eq (io : Asks M) (commit : Level → Array F128 → (ℕ → ByteArray) → M Commit) (hs : Hashes M)
    (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout)
    (h : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (hc : ∀ lv g s, Commit.toC <$> commit lv g s = FlockComplete.commitP hs ms lv g s) :
    ligeritoPZk io commit ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce =
      FlockComplete.ligeritoPZk io.toC hs ms sch T cm0.toC t b lay h rep tPad extra lanePad salt nonce := by
  have e0 : ({ tab := FlockComplete.embedF t, w := FlockComplete.embedF b } : FlockComplete.LigState) =
      ({ tab := embedF t, w := embedF b } : LigState).toC := rfl
  symm
  simp only [FlockComplete.ligeritoPZk, ← hc]
  rw [e0, run_toC]
  simp only [ligConj_bind, ligConj_modify, ligConj_get, ligConj_sendMsg, ligConj_powP, ligConj_oodIntroP,
    ligConj_queryP, ligConj_ite, ligConj_pure, ligConj_lift, bind_assoc, pure_bind, FlockComplete.liftT, lift_map,
    bind_map_left, Commit.toC_cap, ligConj_forIn_cm, map_bind, map_pure, stepMap]
  simp only [ligConj_bind, ligConj_modify, ligConj_get, ligConj_sendMsg, ligConj_challengeP, ligConj_ite,
    ligConj_throw, ligConj_pure, ligConj_lift, bind_assoc, pure_bind, ligConj_forIn_range, ligConj_forIn_array,
    map_bind, map_pure, stepMap, cmOfC_toC, map_ite']
  rfl

/-- **As `flock-firewall finish` runs it**: levels ≥ 1 committed by the FFT (`commitFw`) at SHA-512 and
`hm96-sha512/v1`. -/
theorem ligeritoPZk_fw (io : Asks M) (sch : Flock.Schedule) (T : F128) (cm0 : Commit) (t b : Array F128)
    (lay : Zk.PadLayout) (h : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) :
    ligeritoPZk io (fun lv g s => pure (commitFw lv g s)) MerkleScheme.hm96Sha512 sch T cm0 t b lay h rep tPad extra
        lanePad salt nonce =
      FlockComplete.ligeritoPZk io.toC (Hashes.ofSha512 Sha512.hash) MerkleScheme.hm96Sha512 sch T cm0.toC t b lay h
        rep tPad extra lanePad salt nonce :=
  ligeritoPZk_eq io _ _ _ sch T cm0 t b lay h rep tPad extra lanePad salt nonce fun lv g s => by
    rw [map_pure, commitP_eq]

end Parts

end Flock.Firewall.Finish
