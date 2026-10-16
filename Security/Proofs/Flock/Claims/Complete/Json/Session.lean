import Proofs.Flock.Claims.Complete.Json.Decode
import Proofs.Flock.Claims.Complete.Shape

/-!
# What the one-table honest session returns

`sessionP_record`: at one table, with no keyed coin tree, the pure session's record is the printed honest record
(`recordObj`'s `compress`) of a non-empty `root_B`, the link's points off its coin (`ptsOf`), and one well-formed stream
per rep stream (`RoundsOk`), and the table's extra claims evaluate at those points (`RecordFacts`). It rests on two
facts about the prover's pieces, taken as hypotheses: the commitment's cap is non-empty, and each rep records rounds of
`1 ≤ n ≤ 65536` coins, at least one.

* **Returns through `throw`, `monadLift`, `mapM` and a `for` loop with membership** (`Ret.throw'`, `Ret.monadLift`,
  `Ret.mapM_array`, `Ret.forIn_array_mem`), on top of `Shape`'s `Ret`.
* **The record's JSON is `recordObj`** (`recordJson_eq`): `recordJson` through `Hashes.ofSha512 h` returns its
  `compress`.
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

/-- **What the one-table session returns**: its record is the honest record of a non-empty `root_B`, the link's points
off some coin, and one well-formed stream per rep stream, and the table's extra claims evaluate at those points. -/
def RecordFacts (h : ByteArray → ByteArray) (spec : SessionSpec) (st : Setup) (r : String × Array ByteArray) : Prop :=
  ∃ rb c₁ streams, r.1 = (recordObj spec h #[rb] (ptsOf spec c₁) streams r.2).compress ∧ 0 < rb.size ∧
    streams.size = spec.repStreams.size ∧ (∀ rs ∈ streams, RoundsOk rs) ∧ ∃ e, st.extra (ptsOf spec c₁) = .ok e

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

/-- **The one-table session returns `RecordFacts`**, given a non-empty commitment cap and well-formed rep rounds. -/
theorem sessionP_record (h : ByteArray → ByteArray) (resp : ℕ → Claims.Msg) {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) (hct : spec.coinTree = none) (st : Setup) (trs : Array Trace) (tape : Tape)
    (hC : ∀ lv g salt, Ret (commitP (Hashes.ofSha512 h : Hashes PS) st.merkle lv g salt)
      fun cm => 0 < (flat cm.cap).size)
    (hR : ∀ sch, fast100 st.m = some sch → ∀ tr tb cm₀ extras salt nonce d,
      Ret ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr tb cm₀ extras salt nonce).run (Transcript.init #[] d))
        fun pt => RoundsOk pt.2.rounds) :
    Ret (sessionP (asksS resp) (Hashes.ofSha512 h) spec #[st] trs tape) (RecordFacts h spec st) := by
  unfold sessionP
  refine Ret.bind (Ret.mapM_array _ (fun st' sch => fast100 st'.m = some sch) _ fun a _ => ?_) fun schs hschs => ?_
  · cases hf : fast100 a.m with
    | none => exact Ret.throw' _
    | some sch => exact Ret.pure rfl
  obtain ⟨sch, rfl, hsch⟩ : ∃ sch, schs = #[sch] ∧ fast100 st.m = some sch := by
    obtain ⟨hsz, hf⟩ := hschs
    have h1 : schs.size = 1 := by simpa using hsz
    exact ⟨schs[0], eq_singleton_of h1, by simpa using hf 0 (by simp) (by omega)⟩
  have hrs1 := repStreams_one ht
  refine Ret.bind (Ret.mapM_array _ (fun _ cm => 0 < (flat cm.cap).size) _ fun i hi => ?_) fun cms hcms => ?_
  · have : i = 0 := by simpa using hi
    subst this
    exact hC _ _ _
  refine Ret.bind (Ret.true _) fun c₁ _ => ?_
  refine Ret.bind (Ret.forIn_array_mem _ _ _ (fun k (b : Array (Array Round) × Array ByteArray) =>
    b.1.size = k ∧ b.2.size = k ∧ (∀ rs ∈ b.1, RoundsOk rs) ∧ (0 < k → ∃ e, st.extra (ptsOf spec c₁) = .ok e))
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
    obtain ⟨cm, rfl, hcm⟩ : ∃ cm, cms = #[cm] ∧ 0 < (flat cm.cap).size := by
      obtain ⟨hsz, hf⟩ := hcms
      have h1 : cms.size = 1 := by simpa using hsz
      exact ⟨cms[0], eq_singleton_of h1, hf 0 (by simp) (by omega)⟩
    dsimp only
    rw [recordJson_eq h hct]
    refine Ret.bind (Ret.pure rfl) fun rec hrec => Ret.pure ⟨flat cm.cap, c₁, ss, ?_, hcm, h₁, h₃, ?_⟩
    · rw [← hrec]
      simp only [List.map_toArray, List.map_cons, List.map_nil]
      rfl
    · exact h₄ (by rw [hrs1]; simp)

end FlockComplete.JsonRT
