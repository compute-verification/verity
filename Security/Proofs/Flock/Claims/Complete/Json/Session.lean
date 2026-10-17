import Proofs.Flock.Claims.Complete.Json.Decode
import Proofs.Flock.Claims.Complete.Shape

/-!
# What the one-table honest session returns and sends

`sessionP_facts`: at one table, with no keyed coin tree, a run of the pure session from no sends returns the printed
honest record (`printJ` of `recordObj`) of a non-empty `root_B` under `2^64` bytes, the link's points off the first
reply's coin (`ptsOf`), and one well-formed stream per rep stream (`RoundsOk`); the table's extra claims evaluate at
those points; and its sends are the link of that `root_B` and no public words, then the streams' rounds in order
(`SessionFacts`). It rests on two facts about the prover's pieces, taken as hypotheses: the commitment's cap, at the
schedule's first level, is non-empty and under `2^64` bytes, and each rep records a round, each of at most `65536` coins
(`RepOk`).

* **Returns through `throw`, `monadLift` and `mapM`** (`Ret.throw'`, `Ret.monadLift`, `Ret.mapM_array`), on top of
  `Shape`'s `Ret`.
* **The record's JSON is `recordObj`** (`recordJson_eq`): `recordJson` through `Hashes.ofSha512 h` returns it,
  printed (`printJ`).
* **The sends**: the schedules and the commitment send nothing (`Still`; `pureS` runs a computation of `V` in `PS`, so
  `commitP` there is `commitP` in `V`, `commitP_pureS`), the link pushes its body (`link_go`), and a `for` loop keeps an
  invariant of its state and the sends (`forIn_array_inv`).
* **The rounds**: a rep's transcript records a round exactly where it sends one (`squeezeP`), so the rounds a rep adds
  are the sends it adds, in order, each with the coin at its index (`Recs`). The simulation `recS` relates a transcript
  computation of the pure session to itself when every run adds rounds and sends that way (`Rec`); every prover function
  from the squeeze to the rep is one (`squeezeP_rec`, …, `repP_rec`).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy
open Lean (Json JsonNumber)

theorem Ret.throw' {α : Type} {P : α → Prop} (e : String) : Ret (throw e : PS α) P := by
  intro l a l' h
  simp only [StV.go, Id.run] at h
  cases h

theorem Ret.monadLift {α : Type} (v : V α) : Ret (monadLift v : PS α) fun a => v = .ok a := by
  intro l a l' h
  cases v with
  | error e => cases h
  | ok b => cases h; rfl

theorem Ret.mono {α : Type} {y : PS α} {P Q : α → Prop} (h : Ret y P) (hq : ∀ a, P a → Q a) : Ret y Q :=
  fun l a l' e => hq a (h l a l' e)

theorem Ret.and {α : Type} {y : PS α} {P Q : α → Prop} (hp : Ret y P) (hq : Ret y Q) : Ret y fun a => P a ∧ Q a :=
  fun l a l' e => ⟨hp l a l' e, hq l a l' e⟩

theorem Ret.mapM_list {α β : Type} (f : α → PS β) (P : α → β → Prop) :
    ∀ (l : List α), (∀ a ∈ l, Ret (f a) (P a)) → Ret (l.mapM f) fun ys => List.Forall₂ P l ys
  | [], _ => Ret.pure .nil
  | a :: l, hf => by
    rw [List.mapM_cons]
    refine Ret.bind (hf a List.mem_cons_self) fun b hb => ?_
    refine Ret.bind (Ret.mapM_list f P l fun x hx => hf x (List.mem_cons_of_mem a hx)) fun ys hys => ?_
    exact Ret.pure (.cons hb hys)

theorem Ret.mapM_array {α β : Type} (f : α → PS β) (P : α → β → Prop) (xs : Array α)
    (hf : ∀ a ∈ xs, Ret (f a) (P a)) :
    Ret (xs.mapM f) fun ys => ys.size = xs.size ∧ ∀ (i : ℕ) (h₁ : i < xs.size) (h₂ : i < ys.size), P xs[i] ys[i] := by
  rw [Array.mapM_eq_mapM_toList]
  intro l ys l' e
  rw [show (List.toArray <$> xs.toList.mapM f) = (xs.toList.mapM f >>= fun zs => Pure.pure zs.toArray) from
    (bind_pure_comp _ _).symm] at e
  refine (Ret.bind (Ret.mapM_list f P xs.toList fun a ha => hf a (Array.mem_toList_iff.1 ha))
    fun zs hzs => Ret.pure (P := fun ys : Array β => ys.size = xs.size ∧ ∀ (i : ℕ) (h₁ : i < xs.size)
      (h₂ : i < ys.size), P xs[i] ys[i]) ⟨?_, ?_⟩) l ys l' e
  · simpa using hzs.length_eq.symm
  · intro i h₁ h₂
    have := List.forall₂_iff_get.1 hzs
    simpa using this.2 i (by simpa using h₁) (by simpa using h₂)

theorem dgWith_pure {M : Type → Type} [Monad M] [MonadExceptOf String M] (spec : SessionSpec)
    (h : ByteArray → ByteArray) (b : ByteArray) :
    spec.dgWith (Hashes.ofSha512 h : Hashes M) b = pure (dgOf spec h b) := by
  unfold SessionSpec.dgWith dgOf Hashes.ofSha512
  cases spec.sha512Digests <;> rfl

theorem foldl_push_map {α β : Type} (f : α → β) (xs : Array α) :
    xs.foldl (fun b a => b.push (f a)) #[] = xs.map f := by
  simp

/-- **The session's record JSON is `recordObj`, printed**, with no keyed coin tree. -/
theorem recordJson_eq (h : ByteArray → ByteArray) {spec : SessionSpec} (hct : spec.coinTree = none)
    (rootB : Array ByteArray) (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    recordJson (Hashes.ofSha512 h : Hashes PS) spec rootB points streams proofs =
      pure (printJ (recordObj spec h rootB points streams proofs)) := by
  unfold recordJson roundJ
  have hl : (monadLift (pure (spec.hello, ([] : List (String × Json))) : V _) : PS _) = pure (spec.hello, []) := rfl
  simp only [helloOf, hct, dgWith_pure, pure_bind, hl, List.mapM_pure, Array.mapM_pure,
    Array.forIn_pure_yield_eq_foldl, foldl_push_map, List.append_nil]
  unfold recordObj recordList linkObj linkList rootBList streamObj streamList roundObj proofObj proofList lbOf yOf
  rfl

/-- A recorded rep stream S12 accepts: it records a round, each of `1 ≤ n ≤ 65536` coins. -/
def RoundsOk (rs : Array Round) : Prop := 0 < rs.size ∧ ∀ r ∈ rs, 1 ≤ r.n ∧ r.n ≤ 65536 ∧ r.coins.size = r.n

/-- What a rep's recorded rounds need beyond being its sends (`Recs`): a round, each of at most `65536` coins. -/
def RepOk (rs : Array Round) : Prop := 0 < rs.size ∧ ∀ r ∈ rs, r.n ≤ 65536

/-- The link's points read off the link's coin. -/
def ptsOf (spec : SessionSpec) (c : ℕ → F128) : Array (Array F128) :=
  (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j => c (p * spec.mPts + j)

/-- One table's two rep streams. -/
theorem repStreams_one {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) :
    spec.repStreams = #[(0, 0, s!"{t}/rep{(0 : Nat)}"), (0, 1, s!"{t}/rep{(1 : Nat)}")] := by
  unfold SessionSpec.repStreams
  rw [ht]
  simp only [List.size_toArray, List.length_singleton]
  rw [show Array.range 1 = #[0] from rfl]
  simp

theorem eq_singleton_of {α : Type} {xs : Array α} (h : xs.size = 1) : xs = #[xs[0]] :=
  Array.ext (by simpa using h) fun i h₁ _ => by
    have : i = 0 := by omega
    subst this
    rfl

/-! ## The sends -/

/-- A computation that leaves the sends as they were. -/
def Still {α : Type} (y : PS α) : Prop := ∀ l, (y.go l).2 = l

theorem go_pure {α : Type} (a : α) (l : Array Claims.Msg) : (Pure.pure a : PS α).go l = (.ok a, l) := rfl

theorem go_bind_ok {α β : Type} {y : PS α} {g : α → PS β} {l l'' : Array Claims.Msg} {b : β}
    (h : (y >>= g).go l = (.ok b, l'')) : ∃ a l', y.go l = (.ok a, l') ∧ (g a).go l' = (.ok b, l'') := by
  rw [go_bind] at h
  rcases e : y.go l with ⟨r, l'⟩
  rw [e] at h
  cases r with
  | error _ => cases h
  | ok a => exact ⟨a, l', rfl, h⟩

theorem Still.pure {α : Type} (a : α) : Still (Pure.pure a : PS α) := fun _ => rfl

theorem Still.throw {α : Type} (e : String) : Still (throw e : PS α) := fun _ => rfl

theorem Still.bind {α β : Type} {x : PS α} {f : α → PS β} (hx : Still x) (hf : ∀ a, Still (f a)) :
    Still (x >>= f) := by
  intro l
  rw [go_bind]
  rcases e : x.go l with ⟨r, l'⟩
  have := hx l
  rw [e] at this
  subst this
  cases r with
  | error _ => rfl
  | ok a => exact hf a _

theorem Still.mapM_list {α β : Type} {f : α → PS β} (hf : ∀ a, Still (f a)) : ∀ l : List α, Still (l.mapM f)
  | [] => Still.pure _
  | a :: l => by
    rw [List.mapM_cons]
    exact Still.bind (hf a) fun _ => Still.bind (Still.mapM_list hf l) fun _ => Still.pure _

theorem Still.mapM_array {α β : Type} {f : α → PS β} (hf : ∀ a, Still (f a)) (xs : Array α) : Still (xs.mapM f) := by
  rw [Array.mapM_eq_mapM_toList, map_eq_pure_bind]
  exact Still.bind (Still.mapM_list hf _) fun _ => Still.pure _

theorem Still.go {α : Type} {y : PS α} (hy : Still y) {l l' : Array Claims.Msg} {r : Except String α}
    (h : y.go l = (r, l')) : l' = l := by
  have := hy l
  rw [h] at this
  exact this

theorem Ret.go {α : Type} {y : PS α} {P : α → Prop} (hy : Ret y P) {l l' : Array Claims.Msg} {a : α}
    (h : y.go l = (.ok a, l')) : P a := hy l a l' h

/-- The pure computations of `V`, run in `PS`: each leaves the sends as they were. -/
def pureS : With.Sim V PS where
  R x y := ∀ l, y.go l = (x, l)
  pure _ _ := rfl
  bind {α β x y f g} hxy hfg l := by
    rw [go_bind, hxy l]
    cases x with
    | error e => rfl
    | ok a => exact hfg a l
  throw _ _ := rfl
  lift v _ := by cases v <;> rfl

theorem pureS_hashes (h : ByteArray → ByteArray) :
    pureS.Hashes (Hashes.ofSha512 h : Hashes V) (Hashes.ofSha512 h : Hashes PS) where
  sha512 _ _ := rfl
  sha256 _ _ := rfl
  blake3 _ _ := rfl
  blake3Keyed _ _ _ := rfl

theorem commitP_pureS (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) :
    pureS.R (commitP (Hashes.ofSha512 h : Hashes V) ms lv g salt) (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt) := by
  have ho := With.opsWith_sim pureS (pureS_hashes h) ms
  unfold commitP
  sim_walk [ho.leaf, ho.node]

theorem link_go (resp : ℕ → Claims.Msg) (a b : Array ByteArray) (l : Array Claims.Msg) :
    ((asksS resp).link a b).go l = (.ok (coinOf (resp l.size)), l.push (linkBody a.toList b.toList)) := rfl

theorem Ret.of_pureS {α : Type} {x : V α} {y : PS α} (h : pureS.R x y) {P : α → Prop} (hp : ∀ a, x = .ok a → P a) :
    Ret y P := fun l a l' e => hp a (by rw [h l] at e; exact (Prod.mk.inj e).1)

theorem Still.of_pureS {α : Type} {x : V α} {y : PS α} (h : pureS.R x y) : Still y := fun l => by rw [h l]

theorem monadLift_go {α : Type} (v : V α) (l : Array Claims.Msg) : (monadLift v : PS α).go l = (v, l) := by
  cases v <;> rfl

/-- A `for` loop over a list in `s`, with an invariant of the steps taken, the loop's state and the sends. -/
theorem forIn_list_inv {α β : Type} (f : α → β → PS (ForInStep β)) (inv : ℕ → β → Array Claims.Msg → Prop)
    (s : List α) (hs : ∀ k a b l r l', a ∈ s → inv k b l → (f a b).go l = (.ok r, l') →
      ∃ b', r = .yield b' ∧ inv (k + 1) b' l') :
    ∀ (xs : List α), (∀ a ∈ xs, a ∈ s) → ∀ k b l b' l', inv k b l → (forIn xs b f).go l = (.ok b', l') →
      inv (k + xs.length) b' l'
  | [], _, k, b, l, b', l', h, e => by
    simp only [List.forIn_nil, go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    simpa using h
  | a :: xs, hx, k, b, l, b', l', h, e => by
    rw [List.forIn_cons] at e
    obtain ⟨r, l₁, e1, e2⟩ := go_bind_ok e
    obtain ⟨b₁, rfl, h₁⟩ := hs k a b l r l₁ (hx a List.mem_cons_self) h e1
    have := forIn_list_inv f inv s hs xs (fun x hx' => hx x (List.mem_cons_of_mem a hx')) (k + 1) b₁ l₁ b' l' h₁ e2
    rwa [List.length_cons, show k + (xs.length + 1) = k + 1 + xs.length by omega]

/-- A `for` loop over an array, with an invariant of the steps taken, the loop's state and the sends. -/
theorem forIn_array_inv {α β : Type} (xs : Array α) (f : α → β → PS (ForInStep β))
    (inv : ℕ → β → Array Claims.Msg → Prop) (hs : ∀ k a b l r l', a ∈ xs → inv k b l → (f a b).go l = (.ok r, l') →
      ∃ b', r = .yield b' ∧ inv (k + 1) b' l')
    {b : β} {l : Array Claims.Msg} {b' : β} {l' : Array Claims.Msg} (h0 : inv 0 b l)
    (e : (forIn xs b f).go l = (.ok b', l')) : inv xs.size b' l' := by
  rw [← Array.forIn_toList] at e
  have := forIn_list_inv f inv xs.toList (fun k a b l r l' ha => hs k a b l r l' (Array.mem_toList_iff.1 ha))
    xs.toList (fun _ h => h) 0 b l b' l' h0 e
  simpa using this

/-! ## The rounds -/

/-- Round `r` is the send `b` at index `i`: its retained bytes are the send's body, and its coins, at least one, the
first `n` of the coin `resp i` reads back as. -/
def RoundIs (resp : ℕ → Claims.Msg) (i : ℕ) (b : Claims.Msg) (r : Round) : Prop :=
  ∃ c, r.bytes = some c ∧ b = Claims.toMsg c ∧ 0 < r.n ∧ r.coins = (Array.range r.n).map (coinOf (resp i))

/-- The rounds `rs` are the sends `bs` from index `i` on. -/
def Recs (resp : ℕ → Claims.Msg) : ℕ → List Round → List Claims.Msg → Prop
  | _, [], [] => True
  | i, r :: rs, b :: bs => RoundIs resp i b r ∧ Recs resp (i + 1) rs bs
  | _, _, _ => False

theorem Recs.length_eq {resp : ℕ → Claims.Msg} :
    ∀ {i : ℕ} {rs : List Round} {bs : List Claims.Msg}, Recs resp i rs bs → rs.length = bs.length
  | _, [], [], _ => rfl
  | _, _ :: _, _ :: _, h => by simp [Recs.length_eq h.2]
  | _, [], _ :: _, h => h.elim
  | _, _ :: _, [], h => h.elim

theorem Recs.mem {resp : ℕ → Claims.Msg} :
    ∀ {i : ℕ} {rs : List Round} {bs : List Claims.Msg}, Recs resp i rs bs → ∀ r ∈ rs, 0 < r.n ∧ r.coins.size = r.n
  | _, [], [], _, _, hr => by simp at hr
  | _, _ :: _, _ :: _, h, r, hr => by
    rcases List.mem_cons.1 hr with rfl | hr
    · obtain ⟨_, -, -, hn, hc⟩ := h.1
      exact ⟨hn, by simp [hc]⟩
    · exact Recs.mem h.2 r hr
  | _, [], _ :: _, h, _, _ => h.elim
  | _, _ :: _, [], h, _, _ => h.elim

theorem Recs.append {resp : ℕ → Claims.Msg} :
    ∀ {i : ℕ} {rs : List Round} {bs : List Claims.Msg} {rs' : List Round} {bs' : List Claims.Msg},
      Recs resp i rs bs → Recs resp (i + rs.length) rs' bs' → Recs resp i (rs ++ rs') (bs ++ bs')
  | _, [], [], _, _, _, h => by simpa using h
  | i, _ :: rs, _ :: _, _, _, h, h' =>
    ⟨h.1, Recs.append h.2 (by rw [show i + 1 + rs.length = i + (rs.length + 1) by omega]; exact h')⟩
  | _, [], _ :: _, _, _, h, _ => h.elim
  | _, _ :: _, [], _, _, h, _ => h.elim

/-- A transcript computation of the pure session that adds, on every run, rounds that are the sends it adds. -/
def Rec (resp : ℕ → Claims.Msg) {α : Type} (y : TMW PS α) : Prop :=
  ∀ (t : Transcript) (l : Array Claims.Msg) (a : α) (t' : Transcript) (l' : Array Claims.Msg),
    (y.run t).go l = (.ok (a, t'), l') →
      ∃ rs bs, t'.rounds.toList = t.rounds.toList ++ rs ∧ l'.toList = l.toList ++ bs ∧ Recs resp l.size rs bs

theorem Rec.same {resp : ℕ → Claims.Msg} {t t' : Transcript} {l l' : Array Claims.Msg} (ht : t'.rounds = t.rounds)
    (hl : l' = l) :
    ∃ rs bs, t'.rounds.toList = t.rounds.toList ++ rs ∧ l'.toList = l.toList ++ bs ∧ Recs resp l.size rs bs :=
  ⟨[], [], by simp [ht], by simp [hl], trivial⟩

/-- **The simulation of `Rec`**: a transcript computation of the pure session is related to itself when it is `Rec`. -/
def recS (resp : ℕ → Claims.Msg) : With.Sim (TMW PS) (TMW PS) where
  R _ y := Rec resp y
  pure a t l a' t' l' h := by
    simp only [StateT.run_pure, go_pure, Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨-, rfl⟩, rfl⟩ := h
    exact Rec.same rfl rfl
  bind {α β x y f g} hxy hfg t l b t'' l'' h := by
    rw [StateT.run_bind] at h
    obtain ⟨⟨a, t'⟩, l', h1, h2⟩ := go_bind_ok h
    obtain ⟨rs, bs, e1, e2, r1⟩ := hxy t l a t' l' h1
    obtain ⟨rs', bs', e1', e2', r2⟩ := hfg a t' l' b t'' l'' h2
    refine ⟨rs ++ rs', bs ++ bs', by rw [e1', e1, List.append_assoc], by rw [e2', e2, List.append_assoc],
      Recs.append r1 ?_⟩
    have hs : l'.size = l.size + bs.length := by
      rw [← Array.length_toList, e2, List.length_append, Array.length_toList]
    rw [Recs.length_eq r1, ← hs]
    exact r2
  throw e t l a t' l' h := by
    change ((throw e : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
    obtain ⟨_, _, h1, -⟩ := go_bind_ok h
    cases h1
  lift v t l a t' l' h := by
    change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at h
    obtain ⟨b, l₁, h1, h2⟩ := go_bind_ok h
    cases v with
    | error _ => cases h1
    | ok c =>
      cases h1
      cases h2
      exact Rec.same rfl rfl

theorem round_go (resp : ℕ → Claims.Msg) (c : ByteArray) (l : Array Claims.Msg) :
    ((asksS resp).round c).go l = (.ok (coinOf (resp l.size)), l.push (Claims.toMsg c)) := rfl

theorem squeezeP_rec (resp : ℕ → Claims.Msg) (kind : UInt8) (n : ℕ) :
    (recS resp).R (squeezeP (asksS resp) kind n) (squeezeP (asksS resp) kind n) := by
  intro t l a t' l' h
  change (squeezeP (asksS resp) kind n t).go l = _ at h
  unfold squeezeP at h
  split at h
  · simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨-, rfl⟩, rfl⟩ := h
    exact Rec.same rfl rfl
  · rename_i hn
    obtain ⟨coin, l₁, h1, h2⟩ := go_bind_ok h
    rw [round_go] at h1
    simp only [Prod.mk.injEq, Except.ok.injEq] at h1
    obtain ⟨rfl, rfl⟩ := h1
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at h2
    obtain ⟨⟨-, rfl⟩, rfl⟩ := h2
    exact ⟨[⟨n, ByteArray.empty, (Array.range n).map (coinOf (resp l.size)), some (t.framed kind n)⟩],
      [Claims.toMsg (t.framed kind n)], by simp, by simp, ⟨t.framed kind n, rfl, rfl, Nat.pos_of_ne_zero hn, rfl⟩,
      trivial⟩

/-- A verifier action that keeps the rounds. -/
theorem liftTM_rec (resp : ℕ → Claims.Msg) {α : Type} (x : TM α)
    (hx : ∀ t a t', x t = .ok (a, t') → t'.rounds = t.rounds) : (recS resp).R (liftTM x : TMW PS α) (liftTM x) := by
  intro t l a t' l' h
  change (monadLift (x t) : PS _).go l = _ at h
  rcases e : x t with _ | ⟨a', t''⟩
  · rw [e] at h; cases h
  · rw [e] at h
    cases h
    exact Rec.same (hx t _ _ e) rfl

theorem label_rec (resp : ℕ → Claims.Msg) (s : String) :
    (recS resp).R (liftTM (label s) : TMW PS Unit) (liftTM (label s)) :=
  liftTM_rec resp _ fun _ _ _ h => by cases h; rfl

theorem absorbF_rec (resp : ℕ → Claims.Msg) (v : F128) :
    (recS resp).R (liftTM (absorbF v) : TMW PS Unit) (liftTM (absorbF v)) :=
  liftTM_rec resp _ fun _ _ _ h => by cases h; rfl

theorem absorbSlice_rec (resp : ℕ → Claims.Msg) (vs : Array F128) :
    (recS resp).R (liftTM (absorbSlice vs) : TMW PS Unit) (liftTM (absorbSlice vs)) :=
  liftTM_rec resp _ fun _ _ _ h => by cases h; rfl

theorem absorbK_rec (resp : ℕ → Claims.Msg) (v : F256) :
    (recS resp).R (liftTM (absorbK v) : TMW PS Unit) (liftTM (absorbK v)) :=
  liftTM_rec resp _ fun _ _ _ h => by cases h; rfl

theorem absorbBytes_rec (resp : ℕ → Claims.Msg) (b : ByteArray) :
    (recS resp).R (liftTM (absorbBytes b) : TMW PS Unit) (liftTM (absorbBytes b)) :=
  liftTM_rec resp _ fun _ _ _ h => by cases h; rfl

theorem pow_rec (resp : ℕ → Claims.Msg) (v : UInt64) :
    (recS resp).R (liftTM (pow v) : TMW PS Unit) (liftTM (pow v)) :=
  liftTM_rec resp _ fun _ _ _ h => by cases h; rfl

theorem commitP_rec (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level)
    (g : Array F128) (salt : ℕ → ByteArray) :
    (recS resp).R (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt) : TMW PS Commit)
      (StateT.lift (commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt)) := by
  intro t l a t' l' e
  change ((commitP (Hashes.ofSha512 h : Hashes PS) ms lv g salt) >>= fun a => Pure.pure (a, t)).go l = _ at e
  obtain ⟨b, l₁, e1, e2⟩ := go_bind_ok e
  have := Still.go (Still.of_pureS (commitP_pureS h ms lv g salt)) e1
  subst this
  cases e2
  exact Rec.same rfl rfl

theorem liftT_rec (resp : ℕ → Claims.Msg) {α : Type} {x y : TMW PS α} (h : (recS resp).R x y) :
    ((recS resp).st LigState).R (liftT x) (liftT y) :=
  With.Sim.lift_st _ h

theorem sampleP_rec (resp : ℕ → Claims.Msg) : (recS resp).R (sampleP (asksS resp)) (sampleP (asksS resp)) := by
  unfold sampleP
  sim_walk [squeezeP_rec resp]

theorem sampleKP_rec (resp : ℕ → Claims.Msg) : (recS resp).R (sampleKP (asksS resp)) (sampleKP (asksS resp)) := by
  unfold sampleKP
  sim_walk [squeezeP_rec resp]

theorem zerocheckP_rec (resp : ℕ → Claims.Msg) (m : ℕ) (digest cap : ByteArray) (tr : Trace) :
    (recS resp).R (zerocheckP (asksS resp) m digest cap tr) (zerocheckP (asksS resp) m digest cap tr) := by
  unfold zerocheckP sampleVecP
  sim_walk [squeezeP_rec resp, sampleP_rec resp, label_rec resp, absorbBytes_rec resp, absorbSlice_rec resp,
    absorbF_rec resp]

theorem lincheckP_rec (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) :
    (recS resp).R (lincheckP (asksS resp) c m tr zc) (lincheckP (asksS resp) c m tr zc) := by
  unfold lincheckP
  sim_walk [sampleP_rec resp, label_rec resp, absorbSlice_rec resp, absorbF_rec resp]

theorem ringSwitchP_rec (resp : ℕ → Claims.Msg) (tr : Trace) (claims : Array Claim) :
    (recS resp).R (ringSwitchP (asksS resp) tr claims) (ringSwitchP (asksS resp) tr claims) := by
  unfold ringSwitchP sampleVecP
  sim_walk [squeezeP_rec resp, label_rec resp, absorbSlice_rec resp]

theorem sendMsg_rec (resp : ℕ → Claims.Msg) (m : F256 × F256) :
    ((recS resp).st LigState).R (sendMsg m : LP PS Unit) (sendMsg m) := by
  unfold sendMsg
  sim_walk [liftT_rec resp, absorbK_rec resp]

theorem powP_rec (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    ((recS resp).st LigState).R (powP nonce kind : LP PS Unit) (powP nonce kind) := by
  unfold powP
  sim_walk [liftT_rec resp, pow_rec resp]

theorem challengeP_rec (resp : ℕ → Claims.Msg) (split : Bool) :
    ((recS resp).st LigState).R (challengeP (asksS resp) split) (challengeP (asksS resp) split) := by
  unfold challengeP
  sim_walk [liftT_rec resp, sampleKP_rec resp, sendMsg_rec resp]

theorem oodIntroP_rec (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    ((recS resp).st LigState).R (oodIntroP (asksS resp) nonce g dim) (oodIntroP (asksS resp) nonce g dim) := by
  unfold oodIntroP sampleVecP
  sim_walk [liftT_rec resp, squeezeP_rec resp, sampleP_rec resp, sendMsg_rec resp, powP_rec resp, absorbF_rec resp]

theorem queryP_rec (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    ((recS resp).st LigState).R (queryP (asksS resp) nonce ms lv cm) (queryP (asksS resp) nonce ms lv cm) := by
  unfold queryP sampleVecP
  sim_walk [liftT_rec resp, squeezeP_rec resp, powP_rec resp]

theorem ligeritoP_rec (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (recS resp).R (ligeritoP (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 t b salt nonce)
      (ligeritoP (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 t b salt nonce) := by
  unfold ligeritoP sampleVecP
  sim_walk [liftT_rec resp, squeezeP_rec resp, sampleP_rec resp, sampleKP_rec resp, sendMsg_rec resp, powP_rec resp,
    challengeP_rec resp, oodIntroP_rec resp, queryP_rec resp, commitP_rec resp h, label_rec resp, absorbF_rec resp,
    absorbBytes_rec resp]

/-- **A rep's new rounds are its new sends.** -/
theorem repP_rec (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tr : Trace)
    (t : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (recS resp).R (repP (asksS resp) (Hashes.ofSha512 h) st sch tr t cm0 extras salt nonce)
      (repP (asksS resp) (Hashes.ofSha512 h) st sch tr t cm0 extras salt nonce) := by
  unfold repP
  sim_walk [zerocheckP_rec resp, lincheckP_rec resp, ringSwitchP_rec resp, ligeritoP_rec resp h]

theorem Rec.go {resp : ℕ → Claims.Msg} {α : Type} {y : TMW PS α} (hy : (recS resp).R y y) {t : Transcript}
    {l : Array Claims.Msg} {a : α} {t' : Transcript} {l' : Array Claims.Msg} (h : (y.run t).go l = (.ok (a, t'), l')) :
    ∃ rs bs, t'.rounds.toList = t.rounds.toList ++ rs ∧ l'.toList = l.toList ++ bs ∧ Recs resp l.size rs bs :=
  hy t l a t' l' h

/-- **What the one-table session returns and sends**: its record is the honest record of a non-empty `root_B`, the
link's points off the first reply's coin, and one well-formed stream per rep stream; the table's extra claims evaluate
at those points; and its sends are the link of that `root_B` and no public words, then the streams' rounds in order,
each from index `1` on (`Recs`). -/
def SessionFacts (h : ByteArray → ByteArray) (resp : ℕ → Claims.Msg) (spec : SessionSpec) (st : Setup)
    (r : String × Array ByteArray) (bodies : Array Claims.Msg) : Prop :=
  ∃ rb streams, r.1 = printJ (recordObj spec h #[rb] (ptsOf spec (coinOf (resp 0))) streams r.2) ∧
    0 < rb.size ∧ rb.size < 2 ^ 64 ∧ streams.size = spec.repStreams.size ∧ (∀ rs ∈ streams, RoundsOk rs) ∧
    (∃ e, st.extra (ptsOf spec (coinOf (resp 0))) = .ok e) ∧
    ∃ bs, bodies.toList = linkBody [rb] [ByteArray.empty] :: bs ∧ Recs resp 1 (streams.toList.flatMap Array.toList) bs

/-- **The one-table session's facts**, given a non-empty commitment cap under `2^64` bytes and well-formed rep rounds. -/
theorem sessionP_facts (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) (hct : spec.coinTree = none) (st : Setup) (trs : Array Trace) (tape : Tape)
    (hC : ∀ sch, fast100 st.m = some sch → ∀ g salt cm,
      commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle (sch.levels.getD 0 default) g salt = .ok cm →
        0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64)
    (hR : ∀ sch, fast100 st.m = some sch → ∀ tr tb cm₀ extras salt nonce d,
      st.extra (ptsOf spec (coinOf (resp 0))) = .ok extras →
      Ret ((repP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) st sch tr tb cm₀ extras salt nonce).run
        (Transcript.init #[] d)) fun pt => RepOk pt.2.rounds)
    {r : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec #[st] trs tape).go #[] = (.ok r, bodies)) :
    SessionFacts (Claims.hOf H) resp spec st r bodies := by
  unfold sessionP at hrun
  obtain ⟨schs, l1, hA, hrun⟩ := go_bind_ok hrun
  have hschs := Ret.go (Ret.mapM_array _ (fun st' sch => fast100 st'.m = some sch) _ fun a _ => by
    cases hf : fast100 a.m with
    | none => exact Ret.throw' _
    | some sch => exact Ret.pure rfl) hA
  have hl1 := Still.go (Still.mapM_array (fun a => by
    split
    · exact Still.pure _
    · exact Still.throw _) _) hA
  subst hl1
  obtain ⟨sch, rfl, hsch⟩ : ∃ sch, schs = #[sch] ∧ fast100 st.m = some sch := by
    obtain ⟨hsz, hf⟩ := hschs
    have h1 : schs.size = 1 := by simpa using hsz
    exact ⟨schs[0], eq_singleton_of h1, by simpa using hf 0 (by simp) (by omega)⟩
  dsimp only at hrun
  obtain ⟨cms, l2, hB, hrun⟩ := go_bind_ok hrun
  have hcms := Ret.go (Ret.mapM_array _ (fun _ cm => 0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64) _
    fun i hi => by
      have : i = 0 := by simpa using hi
      subst this
      exact Ret.of_pureS (commitP_pureS _ _ _ _ _) fun cm e => hC sch hsch _ _ cm (by simpa using e)) hB
  have hl2 := Still.go (Still.mapM_array (fun i => by
    exact Still.of_pureS (commitP_pureS _ _ _ _ _)) _) hB
  subst hl2
  obtain ⟨cm, rfl, hcm⟩ : ∃ cm, cms = #[cm] ∧ 0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64 := by
    obtain ⟨hsz, hf⟩ := hcms
    have h1 : cms.size = 1 := by simpa using hsz
    exact ⟨cms[0], eq_singleton_of h1, hf 0 (by simp) (by omega)⟩
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  have hrs1 := repStreams_one ht
  obtain ⟨⟨ss, ps⟩, l₄, hF, hrun⟩ := go_bind_ok hrun
  have hinv := forIn_array_inv _ _ (fun k (b : Array (Array Round) × Array ByteArray) (l : Array Claims.Msg) =>
      b.1.size = k ∧ b.2.size = k ∧ (∀ rs ∈ b.1, RoundsOk rs) ∧
        (0 < k → ∃ e, st.extra (ptsOf spec (coinOf (resp 0))) = .ok e) ∧
        ∃ bs, l.toList = linkBody [flat cm.cap] [ByteArray.empty] :: bs ∧
          Recs resp 1 (b.1.toList.flatMap Array.toList) bs) ?_
    (by exact ⟨rfl, rfl, by simp, by simp, [], by simp, trivial⟩) hF
  · obtain ⟨h₁, h₂, h₃, h₄, bs, hbs, hrec⟩ := hinv
    dsimp only at hrun
    rw [recordJson_eq _ hct] at hrun
    obtain ⟨rec, l₅, hJ, hrun⟩ := go_bind_ok hrun
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at hJ hrun
    obtain ⟨rfl, rfl⟩ := hJ
    obtain ⟨rfl, rfl⟩ := hrun
    refine ⟨flat cm.cap, ss, ?_, hcm.1, hcm.2, h₁, h₃, h₄ (by rw [hrs1]; simp), bs, hbs, hrec⟩
    simp only [List.map_toArray, List.map_cons, List.map_nil]
    rfl
  · rintro k ⟨i, r, name⟩ ⟨ss, ps⟩ l res l' hmem ⟨h₁, h₂, h₃, -, bs, hl, hrec⟩ e
    have hi : i = 0 := by
      rw [hrs1] at hmem
      simp at hmem
      omega
    subst hi
    obtain ⟨extras, l₁, e1, e⟩ := go_bind_ok e
    rw [monadLift_go] at e1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨hex, rfl⟩ := e1
    obtain ⟨⟨pf, tr'⟩, l₂, e2, e⟩ := go_bind_ok e
    have hok := Ret.go (hR sch hsch _ _ _ _ _ _ _ hex) e2
    obtain ⟨rs, bs', hrs, hl₂, hrec'⟩ := Rec.go (repP_rec resp (Claims.hOf H) _ _ _ _ _ _ _ _) e2
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    have hsz : l.size = 1 + (ss.toList.flatMap Array.toList).length := by
      rw [← Array.length_toList, hl, Recs.length_eq hrec]
      simp only [List.length_cons]
      omega
    refine ⟨_, rfl, by simp [h₁], by simp [h₂], fun q hq => ?_, fun _ => ⟨extras, by simpa [ptsOf] using hex⟩,
      bs ++ bs', by rw [hl₂, hl]; rfl, ?_⟩
    · rcases Array.mem_push.1 hq with hq | rfl
      · exact h₃ q hq
      · have hrs' : tr'.rounds.toList = rs := by simpa [Transcript.init] using hrs
        refine ⟨hok.1, fun x hx => ?_⟩
        have := Recs.mem hrec' x (by rw [← hrs']; exact Array.mem_toList_iff.2 hx)
        exact ⟨this.1, hok.2 x hx, this.2⟩
    · simp only [Transcript.init, List.nil_append] at hrs
      simp only [Array.toList_push, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, hrs]
      exact Recs.append hrec (by rw [← hsz]; exact hrec')

end FlockComplete.JsonRT
