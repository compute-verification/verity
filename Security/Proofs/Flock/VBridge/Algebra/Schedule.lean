import Proofs.Flock.VBridge.Algebra.Sizes

/-!
# The rounds `verifierRows` asks for

At the zero coins `run` (RecAlgebra.lean:532–539) draws some list `ns` of round sizes; at the record of zero rounds of
those sizes it draws the same coins, so it returns the same residuals (`residualLins_zeros`). With
`residualLins_sizes`, `verifierRows` returns whenever `structureOf` does, at every record whose rounds have the sizes
`ns` and with the statement's number of extra claims (`verifierRows_schedule`); it never returns with another number
of extra claims (`verifierRows_claims`).
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

/-- `x` at the zero coins and at the records of zero rounds of the sizes it draws: whenever it succeeds at the zero
coins, drawing the sizes `ns` from round `k`, it succeeds alike at every such record holding `t` rounds more. -/
def Z {A : Type} (t : Nat) (x : CM A) : Prop :=
  ∀ k a j, x k Coins.zeros = .ok (a, j) → ∃ ns : List Nat, j = k + ns.length ∧
    ∀ rounds, Fits0 rounds k ns → rounds.size = j + t → x k (Coins.ofRounds rounds) = .ok (a, j)

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

namespace Z

variable {A B : Type} {t : Nat}

theorem bind {x : CM A} {f : A → CM B} (hx : ∀ {t}, Z t x) (hf : ∀ a, Z t (f a)) : Z t (x >>= f) := by
  intro k c j h
  obtain ⟨a, k₁, h₁, h₂⟩ := cm_bind h
  obtain ⟨ns₂, rfl, H₂⟩ := hf a k₁ c j h₂
  obtain ⟨ns₁, rfl, H₁⟩ := hx (t := ns₂.length + t) k a k₁ h₁
  refine ⟨ns₁ ++ ns₂, by simp [Nat.add_assoc], fun rounds hr hs => ?_⟩
  rw [cm_bind_of (H₁ rounds hr.left (by omega))]
  exact H₂ rounds hr.right hs

theorem bind_nz {x : CM A} {f : A → CM B} (hx : Z t x) (hf : ∀ a, NZ (f a)) : Z t (x >>= f) := by
  intro k c j h
  obtain ⟨a, k₁, h₁, h₂⟩ := cm_bind h
  obtain ⟨rfl, H₂⟩ := hf a k₁ c j h₂
  obtain ⟨ns, rfl, H₁⟩ := hx k a _ h₁
  exact ⟨ns, rfl, fun rounds hr hs => by rw [cm_bind_of (H₁ rounds hr hs)]; exact H₂ _⟩

theorem pure' {a : A} : Z t (Pure.pure a : CM A) := by
  intro k a' j h
  cases cm_pure h
  exact ⟨[], rfl, fun _ _ _ => rfl⟩

theorem throw' (e : String) : Z t (throw e : CM A) := by
  intro _ _ _ h
  cases h

theorem liftE' {e : Except String A} : Z t (liftE e) := by
  intro k a j h
  obtain ⟨a₁, rfl, hr⟩ := liftE_ok h
  cases hr
  exact ⟨[], rfl, fun _ _ _ => rfl⟩

theorem ite' {c : Prop} [Decidable c] {x₁ x₂ : CM A} (h₁ : c → Z t x₁) (h₂ : ¬c → Z t x₂) :
    Z t (if c then x₁ else x₂) := by
  split
  · rename_i hc; exact h₁ hc
  · rename_i hc; exact h₂ hc

/-- The check that every recorded round was drawn (RecAlgebra.lean:513), before a tail that draws nothing. -/
theorem guardDrawn {err : String} {rest : CM A} (h : NZ rest) :
    Z 0 (read >>= fun s : Coins => get >>= fun k : Nat =>
      if (!s.drawnAll k) = true then (throw err : CM A) else rest) := by
  have e : ∀ (k : Nat) (src : Coins), (read >>= fun s : Coins => get >>= fun k : Nat =>
      if (!s.drawnAll k) = true then (throw err : CM A) else rest) k src =
      (if (!src.drawnAll k) = true then (throw err : CM A) else rest) k src := fun _ _ => rfl
  intro k a j hx
  rw [e] at hx
  have hz : Coins.zeros.drawnAll k = true := rfl
  rw [hz] at hx
  simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hx
  obtain ⟨hj, H⟩ := h k a j hx
  subst j
  refine ⟨[], rfl, fun rounds _ hs => ?_⟩
  rw [e]
  have hd : (Coins.ofRounds rounds).drawnAll k = true := by simp [Coins.ofRounds, hs]
  rw [hd]
  simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  exact H _

end Z

theorem coinVec_z {t : Nat} (n : Nat) : Z t (coinVec n) := by
  intro k v j h
  rcases coinVec_ok h with ⟨rfl, rfl, rfl⟩ | ⟨hn, hv, rfl⟩
  · exact ⟨[], rfl, fun _ _ _ => rfl⟩
  · cases hv
    refine ⟨[n], rfl, fun rounds hr _ => coinVec_of hn ?_⟩
    have := hr 0 (by simp)
    simp only [Nat.add_zero, List.getElem_cons_zero] at this
    simp only [Coins.ofRounds]
    rw [this]
    simp

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

/-- `run` at the zero coins and at the record of zero rounds of the sizes it draws there returns the same residuals
(RecAlgebra.lean:810–811, 532–539). -/
theorem residualLins_zeros {sh : Shape} {extras : Array Claim} {p : Array Lin × Info Lin}
    (h : residualLins sh Coins.zeros extras = .ok p) :
    ∃ ns : List Nat, residualLins sh (Coins.ofRounds (zeroRounds ns)) extras = .ok p := by
  obtain ⟨j, hj⟩ := residualLins_run h
  obtain ⟨ns, rfl, H⟩ := run_z (M := symbolic sh) sh extras 0 p j hj
  refine ⟨ns, ?_⟩
  have hr := H (zeroRounds ns) (fun i hi => by simp [zeroRounds, hi]) (by simp [zeroRounds])
  unfold residualLins
  change (Prod.fst <$> run sh (symbolic sh) extras 0 (Coins.ofRounds (zeroRounds ns))) = _
  rw [hr]
  rfl

/-- **`verifierRows` returns at the rounds it asks for** (RecAlgebra.lean:824–829, 915–921): whenever `structureOf`
does, there is a list `ns` of round sizes, those `run` draws at the zero coins, such that `verifierRows` returns at
every record whose rounds have the sizes `ns`, with the statement's number of extra claims and any comb. -/
theorem verifierRows_schedule {sh : Shape} {S : GfS} (hS : structureOf sh = .ok S) :
    ∃ ns : List Nat, ∀ (rounds : Array (Array F128)) (extras : Array Claim) (comb : Array F128),
      rounds.toList.map Array.size = ns → sh.claims = 2 + extras.size →
        ∃ x, verifierRows sh rounds extras comb = .ok x := by
  have hS' := hS
  unfold structureOf at hS'
  cases hz : residualLins sh Coins.zeros (extrasLike sh) with
  | error e => rw [hz] at hS'; cases hS'
  | ok p =>
    obtain ⟨ns, hns⟩ := residualLins_zeros hz
    refine ⟨ns, fun rounds extras comb hr hx => ?_⟩
    have hsz : (extrasLike sh).size = extras.size := by simp [extrasLike]; omega
    obtain ⟨p', hp'⟩ := residualLins_sizes (sameSizes_zeroRounds hr) hsz hns
    exact verifierRows_ok_iff.mpr ⟨⟨S, hS⟩, p', hp'⟩

/-- `verifierRows` returns only with the statement's number of extra claims (RecAlgebra.lean:533). -/
theorem verifierRows_claims {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb : Array F128}
    {x : Array F128 × Info Lin × Array (Array (Array F128))} (h : verifierRows sh rounds extras comb = .ok x) :
    sh.claims = 2 + extras.size := by
  obtain ⟨-, p, hp⟩ := verifierRows_ok_iff.mp ⟨x, h⟩
  obtain ⟨j, hj⟩ := residualLins_run hp
  exact run_claims hj

end Flock.RecAlgebra
