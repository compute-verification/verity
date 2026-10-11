import Proofs.Flock.VBridge.Algebra.Sizes

/-!
# The rounds `verifierRows` asks for

At the zero coins `run` (RecAlgebra.lean:532–539) draws some list `ns` of round sizes. At a record of zero rounds of
those sizes it draws the same coins and returns the same residuals, and at any record of zero coins where it succeeds
it has drawn rounds of those sizes, and every round of the record (`run_z`, `run_e`). With `residualLins_sizes`,
`verifierRows` returns, whenever `structureOf` does, exactly at the records whose rounds have the sizes `ns`, with the
statement's number of extra claims (`verifierRows_schedule`).
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine

/-- A record whose rounds from `k` are zero rounds of the sizes `ns`. -/
def Fits0 (rounds : Array (Array F128)) (k : Nat) (ns : List Nat) : Prop :=
  ∀ i (h : i < ns.length), rounds[k + i]? = some (Array.replicate ns[i] 0)

theorem Fits0.left {rounds : Array (Array F128)} {k : Nat} {ns₁ ns₂ : List Nat} (h : Fits0 rounds k (ns₁ ++ ns₂)) :
    Fits0 rounds k ns₁ := by
  intro i hi
  have := h i (by simp; omega)
  rwa [List.getElem_append_left hi] at this

theorem Fits0.right {rounds : Array (Array F128)} {k : Nat} {ns₁ ns₂ : List Nat} (h : Fits0 rounds k (ns₁ ++ ns₂)) :
    Fits0 rounds (k + ns₁.length) ns₂ := by
  intro i hi
  have := h (ns₁.length + i) (by simp; omega)
  rw [List.getElem_append_right (by omega)] at this
  simpa [Nat.add_assoc] using this

/-- A record whose rounds from `k` have the sizes `ns`. -/
def FitsSz (rounds : Array (Array F128)) (k : Nat) (ns : List Nat) : Prop :=
  ∀ i (h : i < ns.length), rounds[k + i]?.map Array.size = some ns[i]

theorem FitsSz.nil {rounds : Array (Array F128)} {k : Nat} : FitsSz rounds k [] :=
  fun i h => absurd h (Nat.not_lt_zero i)

theorem FitsSz.append {rounds : Array (Array F128)} {k : Nat} {ns₁ ns₂ : List Nat} (h₁ : FitsSz rounds k ns₁)
    (h₂ : FitsSz rounds (k + ns₁.length) ns₂) : FitsSz rounds k (ns₁ ++ ns₂) := by
  intro i hi
  rw [List.getElem_append]
  split
  · rename_i h; exact h₁ i h
  · rename_i h
    have := h₂ (i - ns₁.length) (by simp at hi; omega)
    rwa [show k + ns₁.length + (i - ns₁.length) = k + i by omega] at this

/-- A record of zero coins. -/
def Zero (rounds : Array (Array F128)) : Prop := ∀ i (h : i < rounds.size), rounds[i] = Array.replicate rounds[i].size 0

theorem Zero.get {rounds : Array (Array F128)} (hz : Zero rounds) {k : Nat} {r : Array F128}
    (h : rounds[k]? = some r) : r = Array.replicate r.size 0 := by
  obtain ⟨hk, rfl⟩ := Array.getElem?_eq_some_iff.mp h
  exact hz k hk

theorem ofRounds_get? {rounds : Array (Array F128)} (hne : NoEmpty rounds) {k n : Nat} {v : Array F128}
    (h : (Coins.ofRounds rounds).round k n = .ok v) : rounds[k]? = some v ∧ v.size = n := by
  simp only [Coins.ofRounds, filter_noEmpty hne] at h
  cases hr : rounds[k]? with
  | none => rw [hr] at h; cases h
  | some r =>
    rw [hr] at h
    by_cases hn : r.size = n
    · simp only [hn, ↓reduceIte] at h
      cases h
      exact ⟨rfl, hn⟩
    · simp only [hn, ↓reduceIte] at h
      cases h

/-- `x` at the zero coins and at records of zero coins: whenever it succeeds at the zero coins, drawing the sizes `ns`
(none of them 0) from round `k`, it succeeds alike at every record of zero rounds of those sizes holding `t` rounds
more, and wherever it succeeds at a record of zero coins it returns alike, having drawn rounds of the sizes `ns`; both
at records with no round of no coins (`NoEmpty`). -/
def Z {A : Type} (t : Nat) (x : CM A) : Prop :=
  ∀ k a j, x k Coins.zeros = .ok (a, j) → ∃ ns : List Nat, j = k + ns.length ∧ (∀ n ∈ ns, n ≠ 0) ∧
    (∀ rounds, NoEmpty rounds → Fits0 rounds k ns → rounds.size = j + t →
      x k (Coins.ofRounds rounds) = .ok (a, j)) ∧
    ∀ rounds b j', NoEmpty rounds → Zero rounds → x k (Coins.ofRounds rounds) = .ok (b, j') →
      b = a ∧ j' = j ∧ FitsSz rounds k ns

/-- `x` draws no coin and reads no source. -/
def NZ {A : Type} (x : CM A) : Prop :=
  ∀ k a j, x k Coins.zeros = .ok (a, j) → j = k ∧ ∀ src, x k src = .ok (a, j)

namespace NZ

variable {A B : Type}

theorem pure' {a : A} : NZ (Pure.pure a : CM A) := by
  intro k a' j h
  cases cm_pure h
  exact ⟨rfl, fun _ => rfl⟩

theorem throw' (e : String) : NZ (throw e : CM A) := by
  intro _ _ _ h
  cases h

theorem liftE' {e : Except String A} : NZ (liftE e) := by
  intro k a j h
  obtain ⟨a₁, rfl, hr⟩ := liftE_ok h
  cases hr
  exact ⟨rfl, fun _ => rfl⟩

theorem bind {x : CM A} {f : A → CM B} (hx : NZ x) (hf : ∀ a, NZ (f a)) : NZ (x >>= f) := by
  intro k c j h
  obtain ⟨a, k₁, h₁, h₂⟩ := cm_bind h
  obtain ⟨rfl, H₁⟩ := hx k a k₁ h₁
  obtain ⟨rfl, H₂⟩ := hf a _ c j h₂
  exact ⟨rfl, fun src => by rw [cm_bind_of (H₁ src)]; exact H₂ src⟩

theorem ite' {c : Prop} [Decidable c] {x₁ x₂ : CM A} (h₁ : c → NZ x₁) (h₂ : ¬c → NZ x₂) :
    NZ (if c then x₁ else x₂) := by
  split
  · rename_i hc; exact h₁ hc
  · rename_i hc; exact h₂ hc

end NZ

/-- `x` draws no coin, at any source. -/
def ND {A : Type} (x : CM A) : Prop := ∀ k src b j, x k src = .ok (b, j) → j = k

namespace ND

variable {A B : Type}

theorem pure' {a : A} : ND (Pure.pure a : CM A) := by
  intro k src b j h
  cases h
  rfl

theorem throw' (e : String) : ND (throw e : CM A) := by
  intro _ _ _ _ h
  cases h

theorem liftE' {e : Except String A} : ND (liftE e) := by
  intro k src b j h
  obtain ⟨a, -, hr⟩ := liftE_ok h
  cases hr
  rfl

theorem bind {x : CM A} {f : A → CM B} (hx : ND x) (hf : ∀ a, ND (f a)) : ND (x >>= f) := by
  intro k src c j h
  obtain ⟨a, k₁, h₁, h₂⟩ := cm_bind h
  rw [hf a _ _ _ _ h₂, hx _ _ _ _ h₁]

theorem ite' {c : Prop} [Decidable c] {x₁ x₂ : CM A} (h₁ : c → ND x₁) (h₂ : ¬c → ND x₂) :
    ND (if c then x₁ else x₂) := by
  split
  · rename_i hc; exact h₁ hc
  · rename_i hc; exact h₂ hc

end ND

theorem guard_eq {A : Type} {err : String} {rest : CM A} (k : Nat) (src : Coins) :
    (read >>= fun s : Coins => get >>= fun k : Nat =>
      if (!s.drawnAll k) = true then (throw err : CM A) else rest) k src =
      (if (!src.drawnAll k) = true then (throw err : CM A) else rest) k src := rfl

/-- Whenever `x` succeeds at a record with no round of no coins, it has drawn every round of the record. -/
def E {A : Type} (x : CM A) : Prop :=
  ∀ rounds k b j, NoEmpty rounds → x k (Coins.ofRounds rounds) = .ok (b, j) → rounds.size = j

namespace E

variable {A B : Type}

theorem bind {x : CM A} {f : A → CM B} (hf : ∀ a, E (f a)) : E (x >>= f) := by
  intro rounds k c j hne h
  obtain ⟨a, k₁, -, h₂⟩ := cm_bind h
  exact hf a rounds k₁ c j hne h₂

theorem bind_nd {x : CM A} {f : A → CM B} (hx : E x) (hf : ∀ a, ND (f a)) : E (x >>= f) := by
  intro rounds k c j hne h
  obtain ⟨a, k₁, h₁, h₂⟩ := cm_bind h
  rw [hf a _ _ _ _ h₂]
  exact hx _ _ _ _ hne h₁

theorem throw' (e : String) : E (throw e : CM A) := by
  intro _ _ _ _ _ h
  cases h

theorem ite' {c : Prop} [Decidable c] {x₁ x₂ : CM A} (h₁ : c → E x₁) (h₂ : ¬c → E x₂) :
    E (if c then x₁ else x₂) := by
  split
  · rename_i hc; exact h₁ hc
  · rename_i hc; exact h₂ hc

/-- The check that every recorded round was drawn (RecAlgebra.lean:513), before a tail that draws nothing. -/
theorem guardDrawn {err : String} {rest : CM A} (h : ND rest) :
    E (read >>= fun s : Coins => get >>= fun k : Nat =>
      if (!s.drawnAll k) = true then (throw err : CM A) else rest) := by
  intro rounds k b j hne hx
  rw [guard_eq] at hx
  split at hx
  · cases hx
  · rename_i hc
    rw [h _ _ _ _ hx]
    have e : (Coins.ofRounds rounds).drawnAll k = (k == rounds.size) := by
      show (k == (rounds.filter (·.size ≠ 0)).size) = _
      rw [filter_noEmpty hne]
    rw [e] at hc
    simp at hc
    omega

end E

namespace Z

variable {A B : Type} {t : Nat}

theorem bind {x : CM A} {f : A → CM B} (hx : ∀ {t}, Z t x) (hf : ∀ a, Z t (f a)) : Z t (x >>= f) := by
  intro k c j h
  obtain ⟨a, k₁, h₁, h₂⟩ := cm_bind h
  obtain ⟨ns₂, rfl, N₂, H₂, C₂⟩ := hf a k₁ c j h₂
  obtain ⟨ns₁, rfl, N₁, H₁, C₁⟩ := hx (t := ns₂.length + t) k a k₁ h₁
  refine ⟨ns₁ ++ ns₂, by simp [Nat.add_assoc], fun n hn => (List.mem_append.mp hn).elim (N₁ n) (N₂ n),
    fun rounds hne hr hs => ?_, fun rounds c' j' hne hz h' => ?_⟩
  · rw [cm_bind_of (H₁ rounds hne hr.left (by omega))]
    exact H₂ rounds hne hr.right hs
  · obtain ⟨b, k₁', h₁', h₂'⟩ := cm_bind h'
    obtain ⟨rfl, rfl, F₁⟩ := C₁ rounds b k₁' hne hz h₁'
    obtain ⟨rfl, rfl, F₂⟩ := C₂ rounds c' j' hne hz h₂'
    exact ⟨rfl, by simp [Nat.add_assoc], F₁.append F₂⟩

theorem bind_nz {x : CM A} {f : A → CM B} (hx : Z t x) (hf : ∀ a, NZ (f a)) : Z t (x >>= f) := by
  intro k c j h
  obtain ⟨a, k₁, h₁, h₂⟩ := cm_bind h
  obtain ⟨hj, H₂⟩ := hf a k₁ c j h₂
  subst hj
  obtain ⟨ns, rfl, N, H₁, C₁⟩ := hx k a _ h₁
  refine ⟨ns, rfl, N, fun rounds hne hr hs => by rw [cm_bind_of (H₁ rounds hne hr hs)]; exact H₂ _,
    fun rounds c' j' hne hz h' => ?_⟩
  obtain ⟨b, k₁', h₁', h₂'⟩ := cm_bind h'
  obtain ⟨rfl, rfl, F⟩ := C₁ rounds b k₁' hne hz h₁'
  rw [H₂] at h₂'
  cases h₂'
  exact ⟨rfl, rfl, F⟩

theorem pure' {a : A} : Z t (Pure.pure a : CM A) := by
  intro k a' j h
  cases cm_pure h
  exact ⟨[], rfl, (fun _ h => by simp at h), fun _ _ _ _ => rfl,
    fun _ b j' _ _ h' => by cases h'; exact ⟨rfl, rfl, FitsSz.nil⟩⟩

theorem throw' (e : String) : Z t (throw e : CM A) := by
  intro _ _ _ h
  cases h

theorem liftE' {e : Except String A} : Z t (liftE e) := by
  intro k a j h
  obtain ⟨a₁, rfl, hr⟩ := liftE_ok h
  cases hr
  exact ⟨[], rfl, (fun _ h => by simp at h), fun _ _ _ _ => rfl,
    fun _ b j' _ _ h' => by cases h'; exact ⟨rfl, rfl, FitsSz.nil⟩⟩

theorem ite' {c : Prop} [Decidable c] {x₁ x₂ : CM A} (h₁ : c → Z t x₁) (h₂ : ¬c → Z t x₂) :
    Z t (if c then x₁ else x₂) := by
  split
  · rename_i hc; exact h₁ hc
  · rename_i hc; exact h₂ hc

/-- The check that every recorded round was drawn (RecAlgebra.lean:513), before a tail that draws nothing. -/
theorem guardDrawn {err : String} {rest : CM A} (h : NZ rest) :
    Z 0 (read >>= fun s : Coins => get >>= fun k : Nat =>
      if (!s.drawnAll k) = true then (throw err : CM A) else rest) := by
  intro k a j hx
  rw [guard_eq] at hx
  have hz : Coins.zeros.drawnAll k = true := rfl
  rw [hz] at hx
  simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hx
  obtain ⟨hj, H⟩ := h k a j hx
  subst j
  refine ⟨[], rfl, (fun _ h => by simp at h), fun rounds hne _ hs => ?_, fun rounds b j' _ _ h' => ?_⟩
  · rw [guard_eq]
    have hd : (Coins.ofRounds rounds).drawnAll k = true := by
      show (k == (rounds.filter (·.size ≠ 0)).size) = true
      rw [filter_noEmpty hne, hs]
      simp
    rw [hd]
    simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]
    exact H _
  · rw [guard_eq] at h'
    split at h'
    · cases h'
    · rw [H] at h'
      cases h'
      exact ⟨rfl, rfl, FitsSz.nil⟩

end Z

theorem coinVec_z {t : Nat} (n : Nat) : Z t (coinVec n) := by
  intro k v j h
  rcases coinVec_ok h with ⟨rfl, rfl, rfl⟩ | ⟨hn, hv, rfl⟩
  · refine ⟨[], rfl, (fun _ h => by simp at h), fun _ _ _ _ => rfl, fun rounds b j' _ _ h' => ?_⟩
    rcases coinVec_ok h' with ⟨-, rfl, rfl⟩ | ⟨hn, -, -⟩
    · exact ⟨rfl, rfl, FitsSz.nil⟩
    · exact absurd rfl hn
  · cases hv
    refine ⟨[n], rfl, fun n' h => by rw [List.mem_singleton.mp h]; exact hn,
      fun rounds hne hr _ => coinVec_of hn ?_, fun rounds b j' hne hz h' => ?_⟩
    · have := hr 0 (by simp)
      simp only [Nat.add_zero, List.getElem_cons_zero] at this
      simp only [Coins.ofRounds, filter_noEmpty hne]
      rw [this]
      simp
    · rcases coinVec_ok h' with ⟨hn', -, -⟩ | ⟨-, hb, rfl⟩
      · exact absurd hn' hn
      · obtain ⟨hk, hsz⟩ := ofRounds_get? hne hb
        refine ⟨?_, rfl, fun i hi => ?_⟩
        · rw [hz.get hk, hsz]
        · have hi0 : i = 0 := by simp at hi; omega
          subst hi0
          simp [hk, hsz]

theorem coinOne_z {t : Nat} : Z t coinOne := by
  unfold coinOne
  exact Z.bind (coinVec_z 1) fun _ => Z.pure'

theorem coinK2_z {t : Nat} : Z t coinK2 := by
  unfold coinK2
  exact Z.bind (coinVec_z 2) fun _ => Z.pure'

/-- `Z` from every Ligerito state. -/
def ZL {A : Type} (t : Nat) (x : LM Lin A) : Prop := ∀ st, Z t (x st)

namespace ZL

variable {A B : Type} {t : Nat}

theorem bind {x : LM Lin A} {f : A → LM Lin B} (hs : ∀ {t}, ZL t x) (h : ∀ a, ZL t (f a)) : ZL t (x >>= f) :=
  fun st => Z.bind (f := fun p : A × LSt Lin => f p.1 p.2) (hs st) fun p => h p.1 p.2

theorem lift {x : CM A} (hs : ∀ {t}, Z t x) : ZL t (StateT.lift x : LM Lin A) :=
  fun _ => Z.bind hs fun _ => Z.pure'

theorem pure' {a : A} : ZL t (Pure.pure a : LM Lin A) := fun _ => Z.pure'

theorem throw' (e : String) : ZL t (throw e : LM Lin A) := by
  intro st k a j hx
  cases hx

theorem ite' {c : Prop} [Decidable c] {x₁ x₂ : LM Lin A} (h₁ : c → ZL t x₁) (h₂ : ¬c → ZL t x₂) :
    ZL t (if c then x₁ else x₂) := by
  split
  · rename_i hc; exact h₁ hc
  · rename_i hc; exact h₂ hc

theorem get' : ZL t (get : LM Lin (LSt Lin)) := fun _ => Z.pure'

theorem set' {s : LSt Lin} : ZL t (set s : LM Lin PUnit) := fun _ => Z.pure'

theorem modify' {m : LSt Lin → LSt Lin} : ZL t (modify m : LM Lin PUnit) := fun _ => Z.pure'

end ZL

section Run

variable {M : Messages Lin} {t : Nat}

theorem zcRounds_z (r : Array F128) :
    ∀ (l : List (Lin × Lin)) (i : Nat) (claim : Lin) (rho : Array F128), Z t (zcRounds r l i claim rho)
  | [], _, _, _ => Z.pure'
  | (g1, ginf) :: rest, i, claim, rho => by
    simp only [zcRounds]
    exact Z.bind coinOne_z fun _ => zcRounds_z r rest _ _ _

theorem zerocheck_z (m : Nat) : Z t (zerocheck m M) := by
  unfold zerocheck
  refine Z.bind (coinVec_z _) fun _ => Z.bind (coinVec_z _) fun _ => Z.bind coinOne_z fun _ => ?_
  exact Z.bind (zcRounds_z _ M.zc.toList 0 _ #[]) fun ⟨claim, rho⟩ => Z.bind Z.liftE' fun _ => Z.pure'

theorem lcRounds_z : ∀ (l : List (Lin × Lin)) (claim : Lin) (ts : Array F128), Z t (lcRounds l claim ts)
  | [], _, _ => Z.pure'
  | (q1, qinf) :: rest, claim, ts => by
    simp only [lcRounds]
    exact Z.bind coinOne_z fun _ => lcRounds_z rest _ _

theorem lincheck_z (kLog : Nat) (zc : ZcPts Lin) : Z t (lincheck kLog M zc) := by
  unfold lincheck
  refine Z.bind coinOne_z fun _ => Z.bind coinOne_z fun _ => ?_
  exact Z.bind (lcRounds_z M.lc.toList _ #[]) fun ⟨claim, ts⟩ => Z.bind Z.liftE' fun _ =>
    Z.bind coinOne_z fun _ => Z.pure'

theorem rsClaim_z (k : Nat) (cl : AClaim Lin) : Z t (rsClaim M k cl) := by
  unfold rsClaim
  exact Z.bind (coinVec_z 7) fun _ => Z.pure'

theorem rsClaims_z : ∀ (cls : List (AClaim Lin)) (k : Nat) {t : Nat}, Z t (rsClaims M cls k)
  | [], _, _ => Z.pure'
  | cl :: rest, k, _ => by
    simp only [rsClaims]
    exact Z.bind (rsClaim_z k cl) fun ⟨out, beta, rpp, z⟩ => Z.bind (rsClaims_z rest (k + 1))
      fun ⟨outs, betas, pts⟩ => Z.pure'

theorem ringSwitch_z (claims : Array (AClaim Lin)) : Z t (ringSwitch M claims) := by
  unfold ringSwitch
  exact Z.bind (rsClaims_z _ 0) fun ⟨outs, betas, pts⟩ => Z.bind (coinVec_z _) fun _ => Z.pure'

theorem nextMsg_z : ZL t (nextMsg M) := by
  unfold nextMsg
  refine ZL.bind ZL.get' fun st => ?_
  cases M.msgs[st.msg]? with
  | none => exact ZL.throw' _
  | some mm => exact ZL.bind ZL.set' fun _ => ZL.pure'

theorem nextOod_z : ZL t (nextOod M) := by
  unfold nextOod
  refine ZL.bind ZL.get' fun st => ?_
  cases M.ood[st.ood]? with
  | none => exact ZL.throw' _
  | some y => exact ZL.bind ZL.set' fun _ => ZL.pure'

theorem challenge_z : ZL t (challenge M) := by
  unfold challenge
  refine ZL.bind (ZL.lift coinK2_z) fun rk => ZL.bind ZL.get' fun st => ?_
  obtain ⟨s, ⟨a0, a1, a2⟩, msg, ood, oods, conss⟩ := st
  exact ZL.bind ZL.set' fun _ => ZL.bind nextMsg_z fun _ => ZL.bind ZL.modify' fun _ => ZL.pure'

theorem challenges_z : ∀ n {t : Nat}, ZL t (challenges M n)
  | 0, _ => ZL.pure'
  | n + 1, _ => by
    simp only [challenges]
    exact ZL.bind challenge_z fun _ => ZL.bind (challenges_z n) fun _ => ZL.pure'

theorem oodIntro_z (dim split : Nat) : ZL t (oodIntro M dim split) := by
  unfold oodIntro
  exact ZL.bind (ZL.lift (coinVec_z dim)) fun _ => ZL.bind nextOod_z fun _ => ZL.bind nextMsg_z fun _ =>
    ZL.bind (ZL.lift coinOne_z) fun _ => ZL.bind (x := addScaled _ _) ZL.modify' fun _ => ZL.modify'

theorem query_z (sch : Schedule) (i : Nat) (fold : Array F256) : ZL t (query sch M i fold) := by
  unfold query
  exact ZL.bind (ZL.lift (coinVec_z _)) fun _ => ZL.bind (ZL.lift (coinVec_z _)) fun _ => ZL.pure'

theorem queryFold_z (sch : Schedule) (i cols : Nat) (fold : Array F256) : ZL t (queryFold sch M i cols fold) := by
  unfold queryFold
  exact ZL.bind (query_z sch i fold) fun ⟨pos, al, e⟩ => ZL.bind nextMsg_z fun _ =>
    ZL.bind (ZL.lift coinOne_z) fun _ => ZL.bind (x := addScaled _ _) ZL.modify' fun _ => ZL.modify'

theorem levelsLoop_z (sch : Schedule) (R : Nat) :
    ∀ fuel i split (levels : Array (Array F256)) ext, ZL t (levelsLoop sch M R fuel i split levels ext)
  | 0, _, _, _, _ => ZL.pure'
  | fuel + 1, i, split, levels, ext => by
    simp only [levelsLoop]
    refine ZL.ite' (fun _ => ZL.throw' _) fun _ => ?_
    refine ZL.bind (challenges_z 4) fun chs => ZL.ite' (fun _ => ?_) fun _ => ?_
    · exact ZL.bind (oodIntro_z _ _) fun _ => ZL.bind (oodIntro_z _ _) fun _ =>
        ZL.bind (queryFold_z sch _ _ _) fun _ => levelsLoop_z sch R fuel _ _ _ _
    · exact ZL.bind (query_z sch R _) fun ⟨pos, al, e⟩ => ZL.bind (ZL.lift coinOne_z) fun _ =>
        ZL.bind ZL.modify' fun _ => levelsLoop_z sch R fuel _ _ _ _

theorem ligeritoRun_z (sch : Schedule) : ZL t (ligeritoRun sch M) := by
  unfold ligeritoRun
  refine ZL.bind (ZL.lift (coinVec_z _)) fun _ => ZL.bind nextOod_z fun _ => ZL.bind (ZL.lift coinOne_z) fun _ => ?_
  refine ZL.bind ZL.modify' fun _ => ZL.bind nextMsg_z fun _ => ZL.bind ZL.modify' fun _ =>
    ZL.bind (challenges_z _) fun _ => ?_
  exact ZL.bind (oodIntro_z _ _) fun _ => ZL.bind (oodIntro_z _ _) fun _ => ZL.bind (queryFold_z sch _ _ _) fun _ =>
    ZL.bind (levelsLoop_z sch _ _ _ _ _ _) fun ⟨levels, ext⟩ => ZL.pure'

theorem ligerito_z (sch : Schedule) (tIn : Lin) (sws : Array Switched) : Z 0 (ligerito sch M tIn sws) := by
  unfold ligerito
  refine Z.bind (ligeritoRun_z sch _) fun ⟨⟨initCh, levels, ext⟩, st⟩ => Z.guardDrawn ?_
  exact NZ.ite' (fun _ => NZ.throw' _) fun _ => NZ.bind NZ.liftE' fun _ =>
    NZ.ite' (fun _ => NZ.throw' _) fun _ => NZ.pure'

theorem run_z (sh : Shape) (extras : Array Claim) : Z 0 (run sh M extras) := by
  unfold run
  refine Z.ite' (fun _ => Z.throw' _) fun _ => ?_
  refine Z.bind (zerocheck_z _) fun ⟨r0, zc⟩ => Z.bind (lincheck_z _ zc) fun ⟨r1, lc⟩ =>
    Z.bind (ringSwitch_z _) fun ⟨rsr, rs⟩ => ?_
  exact Z.bind_nz (ligerito_z _ _ _) fun ⟨lgr, lg⟩ => NZ.pure'

theorem ligerito_e (sch : Schedule) (tIn : Lin) (sws : Array Switched) : E (ligerito sch M tIn sws) := by
  unfold ligerito
  exact E.bind fun ⟨⟨initCh, levels, ext⟩, st⟩ => E.guardDrawn (ND.ite' (fun _ => ND.throw' _) fun _ =>
    ND.bind ND.liftE' fun _ => ND.ite' (fun _ => ND.throw' _) fun _ => ND.pure')

theorem run_e (sh : Shape) (extras : Array Claim) : E (run sh M extras) := by
  unfold run
  refine E.ite' (fun _ => E.throw' _) fun _ => ?_
  exact E.bind fun ⟨r0, zc⟩ => E.bind fun ⟨r1, lc⟩ => E.bind fun ⟨rsr, rs⟩ =>
    E.bind_nd (ligerito_e _ _ _) fun ⟨lgr, lg⟩ => ND.pure'

end Run

theorem run_claims {sh : Shape} {M : Messages Lin} {extras : Array Claim} {k : Nat} {src : Coins}
    {p : Array Lin × Info Lin} {j : Nat} (h : run sh M extras k src = .ok (p, j)) : sh.claims = 2 + extras.size := by
  unfold run at h
  split at h
  · cases h
  · rename_i hc
    exact not_not.mp hc

/-- A record of zero rounds of the sizes `ns`. -/
def zeroRounds (ns : List Nat) : Array (Array F128) := (ns.map fun n => Array.replicate n 0).toArray

theorem sameSizes_zeroRounds {ns : List Nat} {rounds : Array (Array F128)} (h : rounds.toList.map Array.size = ns) :
    SameSizes (zeroRounds ns) rounds := by
  unfold SameSizes zeroRounds
  rw [h]
  simp [Function.comp_def]

theorem zero_zeroRounds (ns : List Nat) : Zero (zeroRounds ns) := fun i h => by simp [zeroRounds]

theorem eq_of_fitsSz {L ns : List Nat} (hF : FitsSz (zeroRounds L) 0 ns) (hs : (zeroRounds L).size = ns.length) :
    L = ns := by
  have hl : L.length = ns.length := by simpa [zeroRounds] using hs
  refine List.ext_getElem hl fun i h1 h2 => ?_
  have := hF i h2
  rw [Nat.zero_add] at this
  have e : (zeroRounds L)[i]? = some (Array.replicate L[i] 0) := by simp [zeroRounds, h1]
  rw [e] at this
  simpa using this

theorem residualLins_of_run {sh : Shape} {src : Coins} {extras : Array Claim} {p : Array Lin × Info Lin} {j : Nat}
    (h : run sh (symbolic sh) extras 0 src = .ok (p, j)) : residualLins sh src extras = .ok p := by
  unfold residualLins
  change (Prod.fst <$> run sh (symbolic sh) extras 0 src) = _
  rw [h]
  rfl

/-- `verifierRows` returns only with the statement's number of extra claims (RecAlgebra.lean:533). -/
theorem verifierRows_claims {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb : Array F128}
    {x : Array F128 × Info Lin × Array (Array (Array F128))} (h : verifierRows sh rounds extras comb = .ok x) :
    sh.claims = 2 + extras.size := by
  obtain ⟨-, p, hp⟩ := verifierRows_ok_iff.mp ⟨x, h⟩
  obtain ⟨j, hj⟩ := residualLins_run hp
  exact run_claims hj

theorem noEmpty_zeroRounds {ns : List Nat} (hn : ∀ n ∈ ns, n ≠ 0) : NoEmpty (zeroRounds ns) := fun r hr => by
  simp only [zeroRounds, List.mem_toArray, List.mem_map] at hr
  obtain ⟨n, hn', rfl⟩ := hr
  simpa using hn n hn'

theorem sizes_filter_ne {rounds : Array (Array F128)} :
    ∀ n ∈ (rounds.filter (·.size ≠ 0)).toList.map Array.size, n ≠ 0 := fun n hn => by
  obtain ⟨r, hr, rfl⟩ := List.mem_map.mp hn
  exact noEmpty_filter rounds r (Array.mem_toList_iff.mp hr)

/-- **The rounds `verifierRows` returns at** (RecAlgebra.lean:824–829, 915–921): whenever `structureOf` does, there is
one list `ns` of round sizes, none of them 0, those `run` draws at the zero coins, such that `verifierRows` returns at a
record, extra claims and comb exactly when the record's rounds of some coins have the sizes `ns` (`Coins.ofRounds`
skips a round of none) and the extra claims are the statement's number. -/
theorem verifierRows_schedule {sh : Shape} {S : GfS} (hS : structureOf sh = .ok S) :
    ∃ ns : List Nat, (∀ n ∈ ns, n ≠ 0) ∧ ∀ (rounds : Array (Array F128)) (extras : Array Claim) (comb : Array F128),
      (∃ x, verifierRows sh rounds extras comb = .ok x) ↔
        (rounds.filter (·.size ≠ 0)).toList.map Array.size = ns ∧ sh.claims = 2 + extras.size := by
  have hS' := hS
  unfold structureOf at hS'
  cases hz : residualLins sh Coins.zeros (extrasLike sh) with
  | error e => rw [hz] at hS'; cases hS'
  | ok p =>
    obtain ⟨j, hj⟩ := residualLins_run hz
    obtain ⟨ns, rfl, N, H, C⟩ := run_z (M := symbolic sh) sh (extrasLike sh) 0 p j hj
    refine ⟨ns, N, fun rounds extras comb => ⟨fun ⟨x, hx⟩ => ?_, fun ⟨hr, hx⟩ => ?_⟩⟩
    · have hc := verifierRows_claims hx
      obtain ⟨-, p', hp'⟩ := verifierRows_ok_iff.mp ⟨x, hx⟩
      rw [← ofRounds_filter] at hp'
      have hsz : extras.size = (extrasLike sh).size := by simp [extrasLike]; omega
      obtain ⟨p'', hp''⟩ := residualLins_sizes (sameSizes_zeroRounds rfl).symm hsz hp'
      obtain ⟨j', hj'⟩ := residualLins_run hp''
      have hneZ := noEmpty_zeroRounds (sizes_filter_ne (rounds := rounds))
      obtain ⟨-, rfl, hF⟩ := C _ _ j' hneZ (zero_zeroRounds _) hj'
      exact ⟨eq_of_fitsSz hF (by simpa using run_e (M := symbolic sh) sh _ _ _ _ _ hneZ hj'), hc⟩
    · have hns := residualLins_of_run (H (zeroRounds ns) (noEmpty_zeroRounds N)
        (fun i hi => by simp [zeroRounds, hi]) (by simp [zeroRounds]))
      have hsz : (extrasLike sh).size = extras.size := by simp [extrasLike]; omega
      obtain ⟨p', hp'⟩ := residualLins_sizes (sameSizes_zeroRounds hr) hsz hns
      rw [ofRounds_filter] at hp'
      exact verifierRows_ok_iff.mpr ⟨⟨S, hS⟩, p', hp'⟩

end Flock.RecAlgebra
