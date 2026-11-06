import Proofs.Flock.Claims.Recursion.LogWalk

/-!
# Bounding a logged run's SHA-512 calls on every run

A run of the verifier against an oracle keeps the queries it made before a throw (`OCV`, `LogV`), so a bound on a
verdict's calls must hold on rejecting runs too. `CL x n`: a run of `x : LogV α` from any log adds at most `n` entries,
whether it returns or throws. `Calc` is what such a bound needs to compose (`pure`, `bind`, `mono`), and from it alone
come the loops (`list_forIn`, `forIn`, `range_forIn`, `list_mapM`, `mapM`, `mapIdxM`, `findM?`) and the branches.

The verifier's transcript and Ligerito layers are state monads over `LogV` (`TMW`, `LMW`). There a bound carries a
potential of the state (`CS pot`): a run that returns adds at most `n` entries plus what the potential drops, and one
that throws at most `n` plus the potential it started with. The transcript's potential is its rounds left to consume
(`potT`): a squeeze hashes only at a round it then consumes or throws at, so every squeeze costs `0`
(`cs_squeeze`), whatever loop it sits in, and a rep's transcript makes at most its stream's round count of calls.
What hashes nothing keeps the potential (`KeepT`, `KeepL`: the transcript's rounds and its round counter).
-/

namespace FlockDeploy.KS

open Flock

/-! ## The calculus -/

/-- `P x n` bounds a program's calls by `n`, and composes as calls do. -/
structure Calc (m : Type → Type) [Monad m] (P : ∀ {α : Type}, m α → ℕ → Prop) : Prop where
  pure : ∀ {α : Type} (a : α), P (Pure.pure a : m α) 0
  bind : ∀ {α β : Type} {x : m α} {f : α → m β} {n k : ℕ}, P x n → (∀ a, P (f a) k) → P (x >>= f) (n + k)
  mono : ∀ {α : Type} {x : m α} {n k : ℕ}, P x n → n ≤ k → P x k

namespace Calc

variable {m : Type → Type} [Monad m] {P : ∀ {α : Type}, m α → ℕ → Prop}

theorem pure' (C : Calc m P) {α : Type} (a : α) (n : ℕ) : P (Pure.pure a : m α) n :=
  C.mono (C.pure a) (Nat.zero_le _)

theorem bind0 (C : Calc m P) {α β : Type} {x : m α} {f : α → m β} {k : ℕ} (hx : P x 0) (hf : ∀ a, P (f a) k) :
    P (x >>= f) k :=
  C.mono (C.bind hx hf) (by omega)

theorem ite' (C : Calc m P) {α : Type} {c : Prop} [Decidable c] {t e : m α} {n₁ n₂ : ℕ} (ht : P t n₁)
    (he : P e n₂) : P (if c then t else e) (max n₁ n₂) := by
  by_cases hc : c
  · rw [if_pos hc]; exact C.mono ht (Nat.le_max_left _ _)
  · rw [if_neg hc]; exact C.mono he (Nat.le_max_right _ _)

theorem dite' (C : Calc m P) {α : Type} {c : Prop} [Decidable c] {t : c → m α} {e : ¬c → m α} {n₁ n₂ : ℕ}
    (ht : ∀ h, P (t h) n₁) (he : ∀ h, P (e h) n₂) : P (dite c t e) (max n₁ n₂) := by
  by_cases hc : c
  · rw [dif_pos hc]; exact C.mono (ht hc) (Nat.le_max_left _ _)
  · rw [dif_neg hc]; exact C.mono (he hc) (Nat.le_max_right _ _)

theorem list_forIn (C : Calc m P) {α β : Type} {body : α → β → m (ForInStep β)} {c : ℕ}
    (hb : ∀ a b, P (body a b) c) : ∀ (xs : List α) (b : β), P (forIn xs b body) (xs.length * c)
  | [], b => by rw [List.forIn_nil]; exact C.pure' b _
  | a :: xs, b => by
    rw [List.forIn_cons]
    refine C.mono (C.bind (k := xs.length * c) (hb a b) fun st => ?_)
      (by rw [List.length_cons, Nat.succ_mul]; omega)
    cases st with
    | done b' => exact C.pure' b' _
    | yield b' => exact list_forIn C hb xs b'

theorem forIn (C : Calc m P) [LawfulMonad m] {α β : Type} {body : α → β → m (ForInStep β)} {c : ℕ}
    (hb : ∀ a b, P (body a b) c) (xs : Array α) (b : β) : P (ForIn.forIn xs b body) (xs.size * c) := by
  rw [← Array.forIn_toList, ← Array.length_toList]
  exact C.list_forIn hb xs.toList b

theorem range_forIn (C : Calc m P) {β : Type} {body : ℕ → β → m (ForInStep β)} {c : ℕ}
    (hb : ∀ a b, P (body a b) c) (r : Std.Legacy.Range) (b : β) : P (ForIn.forIn r b body) (r.size * c) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have := C.list_forIn hb (List.range' r.start r.size r.step) b
  rwa [List.length_range'] at this

theorem list_mapM (C : Calc m P) [LawfulMonad m] {α β : Type} {g : α → m β} {c : α → ℕ} (hg : ∀ a, P (g a) (c a)) :
    ∀ xs : List α, P (xs.mapM g) (xs.map c).sum
  | [] => by rw [List.mapM_nil]; exact C.pure _
  | a :: xs => by
    rw [List.mapM_cons, List.map_cons, List.sum_cons]
    exact C.mono (C.bind (hg a) fun b => C.bind (list_mapM C hg xs) fun bs => C.pure (b :: bs)) (by omega)

theorem sum_map_const {α : Type} (c : ℕ) : ∀ xs : List α, (xs.map fun _ => c).sum = xs.length * c
  | [] => by simp
  | _ :: xs => by rw [List.map_cons, List.sum_cons, sum_map_const c xs, List.length_cons, Nat.succ_mul]; omega

theorem mapM (C : Calc m P) [LawfulMonad m] {α β : Type} {g : α → m β} {c : α → ℕ} (hg : ∀ a, P (g a) (c a))
    (xs : Array α) : P (xs.mapM g) (xs.toList.map c).sum := by
  rw [Array.mapM_eq_mapM_toList, map_eq_pure_bind]
  exact C.mono (C.bind (C.list_mapM hg xs.toList) fun _ => C.pure _) (by omega)

/-- `mapM` at one bound per element. -/
theorem mapM_const (C : Calc m P) [LawfulMonad m] {α β : Type} {g : α → m β} {c : ℕ} (hg : ∀ a, P (g a) c)
    (xs : Array α) : P (xs.mapM g) (xs.size * c) := by
  have h := C.mapM (c := fun _ => c) hg xs
  rwa [sum_map_const, Array.length_toList] at h

theorem list_mapM_const (C : Calc m P) [LawfulMonad m] {α β : Type} {g : α → m β} {c : ℕ} (hg : ∀ a, P (g a) c) (xs : List α) :
    P (xs.mapM g) (xs.length * c) := by
  have h := C.list_mapM (c := fun _ => c) hg xs
  rwa [sum_map_const] at h

theorem mapIdxM_go (C : Calc m P) {α β : Type} {g : ℕ → α → m β} {c : ℕ} (hg : ∀ i a, P (g i a) c) :
    ∀ (xs : List α) (acc : Array β), P (List.mapIdxM.go g xs acc) (xs.length * c)
  | [], acc => by simp only [List.mapIdxM.go]; exact C.pure' _ _
  | a :: xs, acc => by
    simp only [List.mapIdxM.go]
    exact C.mono (C.bind (hg _ a) fun b => mapIdxM_go C hg xs (acc.push b))
      (by rw [List.length_cons, Nat.succ_mul]; omega)

theorem findM? (C : Calc m P) {α : Type} {p : α → m Bool} {c : ℕ} (hp : ∀ a, P (p a) c) :
    ∀ xs : List α, P (xs.findM? p) (xs.length * c)
  | [] => by simp only [List.findM?]; exact C.pure' _ _
  | a :: xs => by
    simp only [List.findM?]
    refine C.mono (C.bind (k := xs.length * c) (hp a) fun b => ?_) (by rw [List.length_cons, Nat.succ_mul]; omega)
    cases b
    · exact findM? C hp xs
    · exact C.pure' _ _

end Calc

/-! ## Logged runs -/

/-- **A run of `x` from any log adds at most `n` entries**, whether it returns or throws. -/
def CL {α : Type} (x : LogV α) (n : ℕ) : Prop := ∀ l, (x.go l).2.size ≤ l.size + n

theorem go_throw {α : Type} (e : String) (l : Array ByteArray) : (throw e : LogV α).go l = (.error e, l) := rfl

theorem calcCL : Calc LogV CL where
  pure a l := by rw [go_pure]; exact Nat.le_add_right _ _
  bind {α β x f n k} hx hf l := by
    have h1 := hx l
    rw [go_bind]
    generalize x.go l = q at h1 ⊢
    obtain ⟨r, l₁⟩ := q
    cases r with
    | error e => dsimp only at h1 ⊢; omega
    | ok a => have h2 := hf a l₁; dsimp only at h1 ⊢; omega
  mono hx hnk l := Nat.le_trans (hx l) (by omega)

theorem cl_throw {α : Type} (e : String) : CL (throw e : LogV α) 0 := fun l => by
  rw [go_throw]; exact Nat.le_add_right _ _

theorem cl_liftM {α : Type} (v : Flock.V α) : CL (liftM v : LogV α) 0 := fun l => by
  rw [go_liftM]; exact Nat.le_add_right _ _

theorem cl_sha512 (h : ByteArray → ByteArray) (b : ByteArray) : CL ((Hashes.logged h).sha512 b) 1 := fun l => by
  show (l.push b).size ≤ l.size + 1
  simp

theorem cl_sha256 (h : ByteArray → ByteArray) (b : ByteArray) : CL ((Hashes.logged h).sha256 b) 0 := fun l => by
  show l.size ≤ l.size + 0
  omega

theorem cl_dg (h : ByteArray → ByteArray) (spec : SessionSpec) (b : ByteArray) :
    CL (spec.dgWith (Hashes.logged h) b) 1 := fun l => by
  rw [go_dg]
  unfold dgL
  split <;> simp

/-! ## State monads over logged runs, with a potential -/

/-- A run from `s` and the log `l` that returns adds at most `n` entries plus what `pot` drops; one that throws, at most
`n` plus `pot s`. -/
def Bnd {σ α : Type} (pot : σ → ℕ) (s : σ) (l : Array ByteArray) (n : ℕ) :
    Except String (α × σ) × Array ByteArray → Prop
  | (.ok p, l') => l'.size + pot p.2 ≤ l.size + n + pot s
  | (.error _, l') => l'.size ≤ l.size + n + pot s

@[simp] theorem bnd_ok {σ α : Type} {pot : σ → ℕ} {s : σ} {l l' : Array ByteArray} {n : ℕ} {p : α × σ} :
    Bnd pot s l n (.ok p, l') ↔ l'.size + pot p.2 ≤ l.size + n + pot s := Iff.rfl

@[simp] theorem bnd_error {σ α : Type} {pot : σ → ℕ} {s : σ} {l l' : Array ByteArray} {n : ℕ} {e : String} :
    Bnd (α := α) pot s l n (.error e, l') ↔ l'.size ≤ l.size + n + pot s := Iff.rfl

/-- **A run over the state `σ` adds at most `n` entries, with the potential `pot`.** -/
def CS {σ α : Type} (pot : σ → ℕ) (x : StateT σ LogV α) (n : ℕ) : Prop := ∀ s l, Bnd pot s l n ((x.run s).go l)

theorem calcCS {σ : Type} (pot : σ → ℕ) : Calc (StateT σ LogV) (CS pot) where
  pure a s l := by rw [StateT.run_pure, go_pure, bnd_ok]; dsimp only; omega
  bind {α β x f n k} hx hf s l := by
    have h1 := hx s l
    rw [StateT.run_bind, go_bind]
    generalize (x.run s).go l = q at h1 ⊢
    obtain ⟨r, l₁⟩ := q
    cases r with
    | error e => dsimp only; rw [bnd_error] at h1 ⊢; omega
    | ok p =>
      dsimp only
      rw [bnd_ok] at h1
      have h2 := hf p.1 p.2 l₁
      generalize ((f p.1).run p.2).go l₁ = q at h2 ⊢
      obtain ⟨r, l₂⟩ := q
      cases r with
      | error e => rw [bnd_error] at h2 ⊢; omega
      | ok p' => rw [bnd_ok] at h2 ⊢; omega
  mono {α x n k} hx hnk s l := by
    have h1 := hx s l
    generalize (x.run s).go l = q at h1 ⊢
    obtain ⟨r, l₁⟩ := q
    cases r with
    | error e => rw [bnd_error] at h1 ⊢; omega
    | ok p => rw [bnd_ok] at h1 ⊢; omega

theorem cs_lift {σ α : Type} (pot : σ → ℕ) {x : LogV α} {n : ℕ} (hx : CL x n) :
    CS pot (StateT.lift x : StateT σ LogV α) n := fun s l => by
  have h1 := hx l
  rw [StateT.run_lift, go_bind]
  generalize x.go l = q at h1 ⊢
  obtain ⟨r, l₁⟩ := q
  cases r with
  | error e => dsimp only at h1 ⊢; rw [bnd_error]; omega
  | ok a => dsimp only at h1 ⊢; rw [go_pure, bnd_ok]; dsimp only; omega

theorem cs_throw {σ α : Type} (pot : σ → ℕ) (e : String) : CS pot (throw e : StateT σ LogV α) 0 :=
  cs_lift pot (cl_throw e)

/-- A run over a state from a given start: its bound plus the start's potential. -/
theorem cl_run {σ α : Type} {pot : σ → ℕ} {x : StateT σ LogV α} {n : ℕ} (hx : CS pot x n) (s : σ) :
    CL (x.run s) (n + pot s) := fun l => by
  have h1 := hx s l
  generalize (x.run s).go l = q at h1 ⊢
  obtain ⟨r, l₁⟩ := q
  cases r with
  | error e => rw [bnd_error] at h1; exact Nat.le_trans h1 (by omega)
  | ok p => rw [bnd_ok] at h1; dsimp only; omega

/-- **A run over two states, with a potential of the inner one.** -/
def CS2 {σ τ α : Type} (pot : τ → ℕ) (x : StateT σ (StateT τ LogV) α) (n : ℕ) : Prop := ∀ s, CS pot (x.run s) n

theorem calcCS2 {σ τ : Type} (pot : τ → ℕ) : Calc (StateT σ (StateT τ LogV)) (CS2 pot) where
  pure a s := by rw [StateT.run_pure]; exact (calcCS pot).pure _
  bind hx hf s := by rw [StateT.run_bind]; exact (calcCS pot).bind (hx s) fun p => hf p.1 p.2
  mono hx h s := (calcCS pot).mono (hx s) h

theorem cs2_lift {σ τ α : Type} (pot : τ → ℕ) {x : StateT τ LogV α} {n : ℕ} (hx : CS pot x n) :
    CS2 pot (StateT.lift x : StateT σ (StateT τ LogV) α) n := fun s => by
  rw [StateT.run_lift]
  exact (calcCS pot).mono ((calcCS pot).bind hx fun a => (calcCS pot).pure (a, s)) (by omega)

theorem cs2_get {σ τ : Type} (pot : τ → ℕ) : CS2 pot (get : StateT σ (StateT τ LogV) σ) 0 := fun s => by
  rw [StateT.run_get]; exact (calcCS pot).pure _

theorem cs2_set {σ τ : Type} (pot : τ → ℕ) (s' : σ) : CS2 pot (set s' : StateT σ (StateT τ LogV) PUnit) 0 :=
  fun s => by rw [StateT.run_set]; exact (calcCS pot).pure _

theorem cs2_modify {σ τ : Type} (pot : τ → ℕ) (g : σ → σ) :
    CS2 pot (modify g : StateT σ (StateT τ LogV) PUnit) 0 :=
  fun s => by rw [StateT.run_modify]; exact (calcCS pot).pure _

/-! ## The transcript's potential -/

/-- The transcript's rounds left to consume. -/
def potT (t : Transcript) : ℕ := t.rounds.size - t.k

theorem answers_go (h : ByteArray → ByteArray) (r : Round) (c : ByteArray) (l : Array ByteArray) :
    ∃ v l₁, (r.answersWith (Hashes.logged h) c).go l = (.ok v, l₁) ∧ l₁.size ≤ l.size + 1 := by
  unfold Round.answersWith
  split
  · exact ⟨_, l, rfl, by omega⟩
  · split
    · exact ⟨_, l.push c, rfl, by simp⟩
    · exact ⟨_, l, rfl, by omega⟩

/-- **A squeeze costs nothing beyond the potential**: it hashes only at a round it then consumes or throws at. -/
theorem cs_squeeze (h : ByteArray → ByteArray) (kind : UInt8) (n : ℕ) :
    CS potT (Transcript.squeezeWith (Hashes.logged h) kind n) 0 := fun t l => by
  simp only [StateT.run, Transcript.squeezeWith]
  split
  · rw [go_pure, bnd_ok]; simp only [potT]; omega
  · split
    · rw [go_throw, bnd_error]; omega
    · rename_i r hr
      have hk : t.k < t.rounds.size := by
        rcases Array.getElem?_eq_some_iff.1 hr with ⟨hk, -⟩
        exact hk
      split
      · rw [go_throw, bnd_error]; omega
      · obtain ⟨v, l₁, e, hl⟩ := answers_go h r (Transcript.framed t kind n) l
        rw [go_bind, e]
        dsimp only
        split
        · rw [go_throw, bnd_error]; simp only [potT]; omega
        · rw [go_pure, bnd_ok]; simp only [potT]; omega

theorem cs_sample (h : ByteArray → ByteArray) : CS potT (Transcript.sampleWith (Hashes.logged h)) 0 :=
  (calcCS potT).bind0 (cs_squeeze h 1 1) fun _ => (calcCS potT).pure _

theorem cs_sampleVec (h : ByteArray → ByteArray) (n : ℕ) :
    CS potT (Transcript.sampleVecWith (Hashes.logged h) n) 0 :=
  cs_squeeze h 2 n

theorem cs_sampleK (h : ByteArray → ByteArray) : CS potT (Transcript.sampleKWith (Hashes.logged h)) 0 :=
  (calcCS potT).bind0 (cs_squeeze h 2 2) fun _ => (calcCS potT).pure _

/-! ## What hashes nothing keeps the potential -/

/-- A transcript action that returns keeps the potential. -/
def KeepT {α : Type} (x : Transcript.TM α) : Prop := ∀ t a t', x.run t = .ok (a, t') → potT t' ≤ potT t

theorem keepT_pure {α : Type} (a : α) : KeepT (pure a : Transcript.TM α) := fun t b t' h => by
  rw [StateT.run_pure] at h
  cases h
  exact le_rfl

theorem keepT_bind {α β : Type} {x : Transcript.TM α} {f : α → Transcript.TM β} (hx : KeepT x)
    (hf : ∀ a, KeepT (f a)) : KeepT (x >>= f) := fun t b t' h => by
  rw [StateT.run_bind] at h
  cases hxr : x.run t with
  | error e => rw [hxr] at h; cases h
  | ok p =>
    rw [hxr] at h
    exact Nat.le_trans (hf p.1 p.2 b t' h) (hx t p.1 p.2 hxr)

theorem keepT_modify (g : Transcript → Transcript) (hg : ∀ t, potT (g t) = potT t) :
    KeepT (modify g : Transcript.TM PUnit) := fun t a t' h => by
  rw [StateT.run_modify] at h
  cases h
  exact Nat.le_of_eq (hg t)

theorem keepT_lift {α : Type} (v : Flock.V α) : KeepT (StateT.lift v : Transcript.TM α) := fun t a t' h => by
  rw [StateT.run_lift] at h
  cases v with
  | error e => cases h
  | ok b => cases h; exact le_rfl

theorem keepT_label (s : String) : KeepT (Transcript.label s) := keepT_modify _ fun _ => rfl
theorem keepT_absorbF (v : F128) : KeepT (Transcript.absorbF v) := keepT_modify _ fun _ => rfl
theorem keepT_absorbSlice (vs : Array F128) : KeepT (Transcript.absorbSlice vs) := keepT_modify _ fun _ => rfl
theorem keepT_absorbBytes (b : ByteArray) : KeepT (Transcript.absorbBytes b) := keepT_modify _ fun _ => rfl
theorem keepT_absorbK (v : F256) : KeepT (Transcript.absorbK v) := keepT_absorbSlice _
theorem keepT_pow (n : UInt64) : KeepT (Transcript.pow n) := keepT_absorbBytes _

/-- **A transcript action that keeps the potential costs nothing.** -/
theorem cs_liftTM {α : Type} {x : Transcript.TM α} (hx : KeepT x) :
    CS potT (Flock.liftTM x : TMW LogV α) 0 := fun t l => by
  show Bnd potT t l 0 ((liftM (x t) : LogV (α × Transcript)).go l)
  rw [go_liftM]
  cases hxt : x t with
  | error e => rw [bnd_error]; omega
  | ok p => rw [bnd_ok]; have := hx t p.1 p.2 hxt; omega

/-- A Ligerito action that returns keeps the transcript's potential. -/
def KeepL {α : Type} (x : LM α) : Prop := ∀ s, KeepT (x.run s)

theorem keepL_pure {α : Type} (a : α) : KeepL (pure a : LM α) := fun s => by
  rw [StateT.run_pure]; exact keepT_pure _

theorem keepL_bind {α β : Type} {x : LM α} {f : α → LM β} (hx : KeepL x) (hf : ∀ a, KeepL (f a)) :
    KeepL (x >>= f) := fun s => by
  rw [StateT.run_bind]; exact keepT_bind (hx s) fun p => hf p.1 p.2

theorem keepL_get : KeepL (get : LM LState) := fun s => by rw [StateT.run_get]; exact keepT_pure _

theorem keepL_set (s' : LState) : KeepL (set s' : LM PUnit) := fun s => by rw [StateT.run_set]; exact keepT_pure _

theorem keepL_modify (g : LState → LState) : KeepL (modify g : LM PUnit) := fun s => by
  rw [StateT.run_modify]; exact keepT_pure _

theorem keepL_lift {α : Type} {x : Transcript.TM α} (hx : KeepT x) : KeepL (lift x) := fun s => by
  show KeepT ((StateT.lift x : LM α).run s)
  rw [StateT.run_lift]
  exact keepT_bind hx fun a => keepT_pure _

theorem keepL_throw {α : Type} (e : String) : KeepL (throw e : LM α) := fun s t a t' h => by cases h

/-- Walks an `LM` or `TM` program that hashes nothing to `KeepL`/`KeepT`. -/
macro "keep_walk" : tactic => `(tactic| repeat' (first
  | with_reducible refine keepL_bind ?_ fun _ => ?_
  | with_reducible refine keepT_bind ?_ fun _ => ?_
  | with_reducible exact keepL_pure _ | with_reducible exact keepT_pure _ | with_reducible exact keepL_get
  | with_reducible exact keepL_set _ | with_reducible exact keepL_modify _ | with_reducible exact keepL_throw _
  | with_reducible apply keepL_lift
  | with_reducible exact keepT_absorbK _ | with_reducible exact keepT_absorbF _
  | with_reducible exact keepT_absorbSlice _ | with_reducible exact keepT_absorbBytes _
  | with_reducible exact keepT_pow _ | with_reducible exact keepT_label _
  | split))

theorem keepL_nextMsg (p : LigeritoProof) : KeepL (nextMsg p) := by unfold nextMsg; keep_walk
theorem keepL_nextOod (p : LigeritoProof) : KeepL (nextOod p) := by unfold nextOod; keep_walk
theorem keepL_nonceFrom (xs : Array UInt64) (i : ℕ) (what : String) : KeepL (nonceFrom xs i what) := by
  unfold nonceFrom; keep_walk
theorem keepL_claimPow (p : LigeritoProof) : KeepL (claimPow p) := by
  unfold claimPow
  exact keepL_bind keepL_get fun st => keepL_bind (keepL_nonceFrom _ _ _) fun n =>
    keepL_bind (keepL_set _) fun _ => keepL_lift (keepT_pow n)

/-- **A Ligerito action that keeps the potential costs nothing.** -/
theorem cs2_liftLM {α : Type} {x : LM α} (hx : KeepL x) : CS2 potT (Flock.liftLM x : LMW LogV α) 0 := fun s t l => by
  show Bnd potT t l 0 ((liftM (x s t) : LogV ((α × LState) × Transcript)).go l)
  rw [go_liftM]
  cases hxt : x s t with
  | error e => rw [bnd_error]; omega
  | ok p => rw [bnd_ok]; have := hx s t p.1 p.2 hxt; omega

theorem cs2_liftW {α : Type} {x : TMW LogV α} {n : ℕ} (hx : CS potT x n) : CS2 potT (Flock.liftW x) n :=
  cs2_lift potT hx

end FlockDeploy.KS
