import Proofs.Flock.Claims.Complete.Json.Decode
import Proofs.Flock.Claims.Complete.Shape

/-!
# What the one-table honest session returns and sends first

`sessionP_facts`: at one table, with no keyed coin tree, a run of the pure session from no sends returns the printed
honest record (`recordObj`'s `compress`) of a non-empty `root_B` under `2^64` bytes, the link's points off the first
reply's coin (`ptsOf`), and one well-formed stream per rep stream (`RoundsOk`); the table's extra claims evaluate at
those points; and its first send is the link of that `root_B` and no public words (`SessionFacts`). It rests on two facts
about the prover's pieces, taken as hypotheses: the commitment's cap, at the schedule's first level, is non-empty and
under `2^64` bytes, and each rep records rounds of `1 ≤ n ≤ 65536` coins, at least one.

* **Returns through `throw`, `monadLift`, `mapM` and a `for` loop with membership** (`Ret.throw'`, `Ret.monadLift`,
  `Ret.mapM_array`, `Ret.forIn_array_mem`), on top of `Shape`'s `Ret`.
* **The record's JSON is `recordObj`** (`recordJson_eq`): `recordJson` through `Hashes.ofSha512 h` returns its
  `compress`.
* **The sends**: the schedules and the commitment send nothing (`Still`; `pureS` runs a computation of `V` in `PS`, so
  `commitP` there is `commitP` in `V`, `commitP_pureS`), the link pushes its body (`link_go`), and the reps and the
  record only append (`Grows`, from `Sim`'s `repP_sim`).
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

theorem Ret.forIn_list_mem {α β : Type} (f : α → β → PS (ForInStep β)) (inv : ℕ → β → Prop) (s : List α)
    (hs : ∀ k a b, a ∈ s → inv k b → Ret (f a b) fun r => ∃ b', r = .yield b' ∧ inv (k + 1) b') :
    ∀ (l : List α), (∀ a ∈ l, a ∈ s) → ∀ (k : ℕ) (b : β), inv k b → Ret (forIn l b f) (inv (k + l.length)) := by
  intro l
  induction l with
  | nil => intro _ k b h; exact Ret.pure (by simpa using h)
  | cons a as ih =>
    intro hl k b h
    rw [List.forIn_cons]
    refine Ret.bind (hs k a b (hl a List.mem_cons_self) h) ?_
    rintro r ⟨b', rfl, hb'⟩
    have := ih (fun x hx => hl x (List.mem_cons_of_mem a hx)) (k + 1) b' hb'
    rw [List.length_cons, show k + (as.length + 1) = k + 1 + as.length by omega]
    exact this

/-- `Shape`'s `Ret.forIn_array`, with each step's element in the array. -/
theorem Ret.forIn_array_mem {α β : Type} (xs : Array α) (b : β) (f : α → β → PS (ForInStep β)) (inv : ℕ → β → Prop)
    (h0 : inv 0 b) (hs : ∀ k a b, a ∈ xs → inv k b → Ret (f a b) fun r => ∃ b', r = .yield b' ∧ inv (k + 1) b') :
    Ret (forIn xs b f) (inv xs.size) := by
  rw [← Array.forIn_toList]
  have := Ret.forIn_list_mem f inv xs.toList (fun k a b ha => hs k a b (Array.mem_toList_iff.1 ha)) xs.toList
    (fun _ h => h) 0 b h0
  simpa using this

theorem dgWith_pure {M : Type → Type} [Monad M] [MonadExceptOf String M] (spec : SessionSpec)
    (h : ByteArray → ByteArray) (b : ByteArray) :
    spec.dgWith (Hashes.ofSha512 h : Hashes M) b = pure (dgOf spec h b) := by
  unfold SessionSpec.dgWith dgOf Hashes.ofSha512
  cases spec.sha512Digests <;> rfl

theorem foldl_push_map {α β : Type} (f : α → β) (xs : Array α) :
    xs.foldl (fun b a => b.push (f a)) #[] = xs.map f := by
  simp

/-- **The session's record JSON is `recordObj`'s `compress`**, with no keyed coin tree. -/
theorem recordJson_eq (h : ByteArray → ByteArray) {spec : SessionSpec} (hct : spec.coinTree = none)
    (rootB : Array ByteArray) (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    recordJson (Hashes.ofSha512 h : Hashes PS) spec rootB points streams proofs =
      pure (recordObj spec h rootB points streams proofs).compress := by
  unfold recordJson roundJ
  have hl : (monadLift (pure (spec.hello, ([] : List (String × Json))) : V _) : PS _) = pure (spec.hello, []) := rfl
  simp only [helloOf, hct, dgWith_pure, pure_bind, hl, List.mapM_pure, Array.mapM_pure,
    Array.forIn_pure_yield_eq_foldl, foldl_push_map, List.append_nil]
  unfold recordObj recordList linkObj linkList rootBList streamObj streamList roundObj proofObj proofList lbOf yOf
  rfl

/-- A recorded rep stream S12 accepts: it records a round, each of `1 ≤ n ≤ 65536` coins. -/
def RoundsOk (rs : Array Round) : Prop := 0 < rs.size ∧ ∀ r ∈ rs, 1 ≤ r.n ∧ r.n ≤ 65536 ∧ r.coins.size = r.n

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

/-- A computation that only appends sends. -/
def Grows {α : Type} (y : PS α) : Prop := ∀ l, ∃ bs : List Claims.Msg, (y.go l).2 = l ++ bs.toArray

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

theorem Grows.pure {α : Type} (a : α) : Grows (Pure.pure a : PS α) := fun l => ⟨[], by simp [go_pure]⟩

theorem Grows.bind {α β : Type} {x : PS α} {f : α → PS β} (hx : Grows x) (hf : ∀ a, Grows (f a)) :
    Grows (x >>= f) := by
  intro l
  rw [go_bind]
  rcases e : x.go l with ⟨r, l'⟩
  obtain ⟨bs, hbs⟩ := hx l
  rw [e] at hbs
  dsimp only at hbs
  subst hbs
  cases r with
  | error _ => exact ⟨bs, rfl⟩
  | ok a =>
    obtain ⟨bs', hbs'⟩ := hf a (l ++ bs.toArray)
    exact ⟨bs ++ bs', by rw [hbs']; simp⟩

theorem Grows.monadLift {α : Type} (v : V α) : Grows (monadLift v : PS α) := fun l => ⟨[], by
  cases v <;> rfl⟩

theorem Grows.forIn_list {α β : Type} {f : α → β → PS (ForInStep β)} (hf : ∀ a b, Grows (f a b)) :
    ∀ (l : List α) (b : β), Grows (forIn l b f)
  | [], b => Grows.pure b
  | a :: l, b => by
    rw [List.forIn_cons]
    refine Grows.bind (hf a b) fun r => ?_
    cases r with
    | done b => exact Grows.pure b
    | yield b => exact Grows.forIn_list hf l b

theorem Grows.forIn_array {α β : Type} {f : α → β → PS (ForInStep β)} (hf : ∀ a b, Grows (f a b)) (xs : Array α)
    (b : β) : Grows (forIn xs b f) := by
  rw [← Array.forIn_toList]
  exact Grows.forIn_list hf _ b

theorem Grows.of_simS {H : Claims.Msg → Claims.Msg} {resp : ℕ → Claims.Msg} {α : Type} {x : PV α} {y : PS α}
    (h : (simS H resp).R x y) : Grows y := fun l => by
  obtain ⟨bs, -, hl⟩ := h l
  exact ⟨bs, hl⟩

theorem Grows.go {α : Type} {y : PS α} (hy : Grows y) {l l' : Array Claims.Msg} {r : Except String α}
    (h : y.go l = (r, l')) : ∃ bs : List Claims.Msg, l' = l ++ bs.toArray := by
  obtain ⟨bs, hbs⟩ := hy l
  rw [h] at hbs
  exact ⟨bs, hbs⟩

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

/-- **What the one-table session returns and sends first**: its record is the honest record of a non-empty `root_B`,
the link's points off the first reply's coin, and one well-formed stream per rep stream; the table's extra claims
evaluate at those points; and its first send is the link of that `root_B` and no public words. -/
def SessionFacts (h : ByteArray → ByteArray) (resp : ℕ → Claims.Msg) (spec : SessionSpec) (st : Setup)
    (r : String × Array ByteArray) (bodies : Array Claims.Msg) : Prop :=
  ∃ rb streams, r.1 = (recordObj spec h #[rb] (ptsOf spec (coinOf (resp 0))) streams r.2).compress ∧
    0 < rb.size ∧ rb.size < 2 ^ 64 ∧ streams.size = spec.repStreams.size ∧ (∀ rs ∈ streams, RoundsOk rs) ∧
    (∃ e, st.extra (ptsOf spec (coinOf (resp 0))) = .ok e) ∧ bodies[0]? = some (linkBody [rb] [ByteArray.empty])

/-- **The one-table session's facts**, given a non-empty commitment cap under `2^64` bytes and well-formed rep rounds. -/
theorem sessionP_facts (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) (hct : spec.coinTree = none) (st : Setup) (trs : Array Trace) (tape : Tape)
    (hC : ∀ sch, fast100 st.m = some sch → ∀ g salt cm,
      commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle (sch.levels.getD 0 default) g salt = .ok cm →
        0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64)
    (hR : ∀ sch, fast100 st.m = some sch → ∀ tr tb cm₀ extras salt nonce d,
      Ret ((repP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) st sch tr tb cm₀ extras salt nonce).run
        (Transcript.init #[] d)) fun pt => RoundsOk pt.2.rounds)
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
  obtain ⟨bs, hbs⟩ := Grows.go (by
    refine Grows.bind (Grows.forIn_array (fun a b => ?_) _ _) fun b => ?_
    · obtain ⟨i, r, name⟩ := a
      obtain ⟨ss, ps⟩ := b
      exact Grows.bind (Grows.monadLift _) fun extras => Grows.bind
        (Grows.of_simS (With.Sim.run _ (repP_sim H resp _ _ _ _ _ _ _ _) _)) fun _ => Grows.pure _
    · obtain ⟨ss, ps⟩ := b
      dsimp only
      rw [recordJson_eq _ hct]
      exact Grows.bind (Grows.pure _) fun _ => Grows.pure _) hrun
  have hv := Ret.go (P := fun r : String × Array ByteArray => ∃ streams,
      r.1 = (recordObj spec (Claims.hOf H) #[flat cm.cap] (ptsOf spec (coinOf (resp 0))) streams r.2).compress ∧
      streams.size = spec.repStreams.size ∧ (∀ rs ∈ streams, RoundsOk rs) ∧
      ∃ e, st.extra (ptsOf spec (coinOf (resp 0))) = .ok e) (by
    refine Ret.bind (Ret.forIn_array_mem _ _ _ (fun k (b : Array (Array Round) × Array ByteArray) =>
      b.1.size = k ∧ b.2.size = k ∧ (∀ rs ∈ b.1, RoundsOk rs) ∧
        (0 < k → ∃ e, st.extra (ptsOf spec (coinOf (resp 0))) = .ok e))
      ⟨rfl, rfl, by simp, by simp⟩ ?_) ?_
    · rintro k ⟨i, r, name⟩ ⟨ss, ps⟩ hmem ⟨h₁, h₂, h₃, -⟩
      have hi : i = 0 := by
        rw [hrs1] at hmem
        simp at hmem
        omega
      subst hi
      refine Ret.bind (Ret.monadLift _) fun extras hex => ?_
      refine Ret.bind (hR sch hsch _ _ _ _ _ _ _) fun ⟨pf, tr'⟩ hpt => ?_
      refine Ret.pure ⟨_, rfl, by simp [h₁], by simp [h₂], fun rs hrs => ?_,
        fun _ => ⟨extras, by simpa [ptsOf] using hex⟩⟩
      rcases Array.mem_push.1 hrs with hrs | rfl
      · exact h₃ rs hrs
      · exact hpt
    · rintro ⟨ss, ps⟩ ⟨h₁, h₂, h₃, h₄⟩
      dsimp only
      rw [recordJson_eq _ hct]
      refine Ret.bind (Ret.pure rfl) fun rec hrec => Ret.pure ⟨ss, ?_, h₁, h₃, h₄ (by rw [hrs1]; simp)⟩
      rw [← hrec]
      simp only [List.map_toArray, List.map_cons, List.map_nil]
      rfl) hrun
  obtain ⟨streams, h₁, h₂, h₃, h₄⟩ := hv
  refine ⟨flat cm.cap, streams, h₁, hcm.1, hcm.2, h₂, h₃, h₄, ?_⟩
  rw [hbs]
  simp

end FlockComplete.JsonRT
