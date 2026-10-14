import Proofs.Flock.C3.Compose.Tied

/-!
# The guarantee's message types hold the stages' shapes

`shapeTys_holo`: `ShapeTys` at the guarantee's message types `holoTys st hp` (`Recursive/AuditC3.lean`), at every run
of `holoAt st hp`: every message kind of the reps' shape (`holoRepShape`), the sparse phases' (`sparseKinds`), R's
opening's (`opShape` at two claims) and R's Ligerito's (`ligShape`) is one of `holoTys`'s sixteen types. Each shape's
lemma takes, for each type its messages have, an index of `T` naming it.
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled

section kinds

variable {nT : ℕ} {T : Fin nT → Type} {F K D : Type}

theorem nil_msg {M : Type} (h : Exec.Kind.msg M ∈ ([] : List Exec.Kind)) : ∃ j, T j = M := by
  simp at h

theorem cons_msg {M A : Type} {l : List Exec.Kind} (hA : ∃ j, T j = A) (hl : Exec.Kind.msg M ∈ l → ∃ j, T j = M)
    (h : Exec.Kind.msg M ∈ Exec.Kind.msg A :: l) : ∃ j, T j = M := by
  rcases List.mem_cons.1 h with h | h
  · cases Exec.Kind.msg.inj h
    exact hA
  · exact hl h

theorem cons_coin {M C : Type} {l : List Exec.Kind} (hl : Exec.Kind.msg M ∈ l → ∃ j, T j = M)
    (h : Exec.Kind.msg M ∈ Exec.Kind.coin C :: l) : ∃ j, T j = M := by
  rcases List.mem_cons.1 h with h | h
  · cases h
  · exact hl h

theorem append_msg {M : Type} {l₁ l₂ : List Exec.Kind} (h₁ : Exec.Kind.msg M ∈ l₁ → ∃ j, T j = M)
    (h₂ : Exec.Kind.msg M ∈ l₂ → ∃ j, T j = M) (h : Exec.Kind.msg M ∈ l₁ ++ l₂) : ∃ j, T j = M :=
  (List.mem_append.1 h).elim h₁ h₂

theorem replicate_msg {M C : Type} {n : ℕ} (h : Exec.Kind.msg M ∈ List.replicate n (Exec.Kind.coin C)) :
    ∃ j, T j = M := by
  cases (List.mem_replicate.1 h).2

theorem pairRounds_msg (hF : ∃ j, T j = (F × F)) (n : ℕ) {M : Type} (h : Exec.Kind.msg M ∈ Exec.pairRounds F n) :
    ∃ j, T j = M := by
  induction n with
  | zero => exact nil_msg h
  | succ n ih => exact cons_msg hF (cons_coin ih) h

theorem foldShape_msg (hK : ∃ j, T j = (K × K)) (k : ℕ) {M : Type} (h : Exec.Kind.msg M ∈ Exec.foldShape F K k) :
    ∃ j, T j = M := by
  induction k with
  | zero => exact nil_msg h
  | succ k ih => exact cons_coin (cons_msg hK ih) h

theorem switches_msg (hS : ∃ j, T j = (Fin 128 → F)) (n : ℕ) {M : Type} (h : Exec.Kind.msg M ∈ Exec.switches F n) :
    ∃ j, T j = M := by
  induction n with
  | zero => exact nil_msg h
  | succ n ih => exact cons_msg hS (cons_coin ih) h

theorem degKinds_msg {k : ℕ} (hD : ∃ j, T j = (F × (Fin k → F))) (n : ℕ) {M : Type}
    (h : Exec.Kind.msg M ∈ FlockC3.Sparse.degKinds F k n) : ∃ j, T j = M := by
  induction n with
  | zero => exact nil_msg h
  | succ n ih => exact cons_msg hD (cons_coin ih) h

theorem oodShape_msg (hF : ∃ j, T j = F) (hK : ∃ j, T j = (K × K)) (dim : ℕ) {M : Type}
    (h : Exec.Kind.msg M ∈ Exec.oodShape F K dim) : ∃ j, T j = M := by
  unfold Exec.oodShape at h
  exact cons_coin (cons_msg hF (cons_msg hK (cons_coin (cons_coin (cons_msg hF (cons_msg hK (cons_coin nil_msg)))))))
    h

theorem qbShape_msg (hK : ∃ j, T j = (K × K)) (lv : FlockSoundness.Accounting.Level) (b : Bool) {M : Type}
    (h : Exec.Kind.msg M ∈ Exec.qbShape F K lv b) : ∃ j, T j = M := by
  unfold Exec.qbShape at h
  refine cons_coin (cons_coin (append_msg ?_ (cons_coin nil_msg))) h
  intro h'
  split at h'
  · exact cons_msg hK nil_msg h'
  · exact nil_msg h'

theorem recShape_msg {ext : ℕ} (hF : ∃ j, T j = F) (hK : ∃ j, T j = (K × K))
    (hC : ∃ j, T j = FlockSoundness.Model.Cap D) (hE : ∃ j, T j = (Fin (2 ^ (ext + 1)) → F)) {M : Type} :
    ∀ levels : List FlockSoundness.Accounting.Level, Exec.Kind.msg M ∈ Exec.recShape F K D ext levels →
      ∃ j, T j = M
  | [], h => nil_msg h
  | [lv], h => append_msg (foldShape_msg hK _) (cons_msg hE (qbShape_msg hK lv false)) h
  | lv :: lv' :: rest, h => append_msg (foldShape_msg hK _) (cons_msg hC (append_msg (oodShape_msg hF hK _)
      (append_msg (qbShape_msg hK lv true) (recShape_msg hF hK hC hE (lv' :: rest))))) h

theorem zcShape_msg (hZ : ∃ j, T j = ((Fin 64 → F) × (Fin 64 → F))) (hP : ∃ j, T j = (F × F)) (m : ℕ) {M : Type}
    (h : Exec.Kind.msg M ∈ Exec.zcShape F m) : ∃ j, T j = M := by
  unfold Exec.zcShape at h
  exact cons_coin (cons_coin (cons_msg hZ (cons_coin (append_msg (pairRounds_msg hP _) (cons_msg hP nil_msg))))) h

theorem holoLcShape_msg (hP : ∃ j, T j = (F × F)) (hV : ∃ j, T j = (Fin 64 → F)) (kLog : ℕ) {M : Type}
    (h : Exec.Kind.msg M ∈ FlockC3.HoloValues.holoLcShape F kLog) : ∃ j, T j = M := by
  unfold FlockC3.HoloValues.holoLcShape at h
  exact cons_coin (cons_coin (append_msg (pairRounds_msg hP _) (cons_msg hV (cons_msg hV (cons_coin nil_msg))))) h

theorem opShape_msg (hS : ∃ j, T j = (Fin 128 → F)) (n : ℕ) {M : Type} (h : Exec.Kind.msg M ∈ Exec.opShape F n) :
    ∃ j, T j = M := by
  unfold Exec.opShape at h
  exact append_msg (switches_msg hS _) (cons_coin nil_msg) h

theorem ligShape_msg {sch : FlockSoundness.Accounting.Schedule} (hF : ∃ j, T j = F) (hK : ∃ j, T j = (K × K))
    (hC : ∃ j, T j = FlockSoundness.Model.Cap D)
    (hE : ∃ j, T j = (Fin (2 ^ (sch.ext + 1)) → F)) {M : Type} (h : Exec.Kind.msg M ∈ Exec.ligShape F K D sch) :
    ∃ j, T j = M := by
  unfold Exec.ligShape at h
  split at h
  · exact nil_msg h
  · refine cons_coin (cons_msg hF (cons_coin (cons_msg hK (append_msg (foldShape_msg hK _) ?_)))) h
    intro h'
    split at h'
    · exact nil_msg h'
    · exact cons_msg hC (append_msg (oodShape_msg hF hK _) (append_msg (qbShape_msg hK _ true)
        (recShape_msg hF hK hC hE _))) h'

theorem sparseKinds_msg {sh : FlockSoundness.Model.SparseShape} (h1 : ∃ j, T j = (F × (Fin 1 → F))) (hF : ∃ j, T j = F)
    (h2 : ∃ j, T j = (F × (Fin (2 * sh.ell) → F))) (h3 : ∃ j, T j = (F × (Fin (2 * sh.ell + 1) → F)))
    (hp : ∃ j, T j = (Fin (2 ^ sh.pLog) → F)) (ht : ∃ j, T j = (Fin (2 ^ sh.tLog) → F)) {M : Type}
    (h : Exec.Kind.msg M ∈ FlockC3.Sparse.sparseKinds F sh) : ∃ j, T j = M := by
  unfold FlockC3.Sparse.sparseKinds at h
  refine append_msg replicate_msg (append_msg (degKinds_msg h1 _) (append_msg (cons_msg hF nil_msg)
    (append_msg (degKinds_msg h2 _) (append_msg (degKinds_msg h3 _) (append_msg (cons_msg hp nil_msg)
      (append_msg replicate_msg (append_msg ?_ replicate_msg))))))) h
  intro h'
  split at h'
  · exact nil_msg h'
  · exact cons_msg ht nil_msg h'

end kinds

/-- **The guarantee's message types hold the stages' shapes** (`ShapeTys`) at every run of candidate 3's table at
`holoAt st hp`: each shape's message types are among `holoTys st hp`'s. -/
theorem shapeTys_holo (st : Flock.Stmt) (hp : HoloParams) (u : C3Run (holoAt st hp)) :
    ShapeTys (Flock.Guarantees.holoTys st hp) (holoAt st hp) u := by
  unfold ShapeTys
  have h0 : ∃ j, Flock.Guarantees.holoTys st hp j = FlockSoundness.Model.Cap Refine.Dig := ⟨0, rfl⟩
  have h3 : ∃ j, Flock.Guarantees.holoTys st hp j = ((Fin 64 → GF128) × (Fin 64 → GF128)) := ⟨3, rfl⟩
  have h4 : ∃ j, Flock.Guarantees.holoTys st hp j = (GF128 × GF128) := ⟨4, rfl⟩
  have h5 : ∃ j, Flock.Guarantees.holoTys st hp j = (Fin 64 → GF128) := ⟨5, rfl⟩
  have h6 : ∃ j, Flock.Guarantees.holoTys st hp j = (Fin 128 → GF128) := ⟨6, rfl⟩
  have h7 : ∃ j, Flock.Guarantees.holoTys st hp j = GF128 := ⟨7, rfl⟩
  have h8 : ∃ j, Flock.Guarantees.holoTys st hp j = (GF256 × GF256) := ⟨8, rfl⟩
  have h9 : ∃ j, Flock.Guarantees.holoTys st hp j = (Fin (2 ^ ((holoAt st hp).sch.ext + 1)) → GF128) := ⟨9, rfl⟩
  have h10 : ∃ j, Flock.Guarantees.holoTys st hp j = (Fin (2 ^ ((holoAt st hp).rsch.sch.ext + 1)) → GF128) :=
    ⟨10, rfl⟩
  have h11 : ∃ j, Flock.Guarantees.holoTys st hp j = (GF128 × (Fin 1 → GF128)) := ⟨11, rfl⟩
  have h12 : ∃ j, Flock.Guarantees.holoTys st hp j = (GF128 × (Fin (2 * (holoAt st hp).cls.shape.ell) → GF128)) :=
    ⟨12, rfl⟩
  have h13 :
      ∃ j, Flock.Guarantees.holoTys st hp j = (GF128 × (Fin (2 * (holoAt st hp).cls.shape.ell + 1) → GF128)) :=
    ⟨13, rfl⟩
  have h14 : ∃ j, Flock.Guarantees.holoTys st hp j = (Fin (2 ^ (holoAt st hp).cls.shape.pLog) → GF128) := ⟨14, rfl⟩
  have h15 : ∃ j, Flock.Guarantees.holoTys st hp j = (Fin (2 ^ (holoAt st hp).cls.shape.tLog) → GF128) := ⟨15, rfl⟩
  refine ⟨fun M h => ?_, fun M h => sparseKinds_msg h11 h7 h12 h13 h14 h15 h, fun M h => opShape_msg h6 _ h,
    fun M h => ligShape_msg h7 h8 h0 h10 h⟩
  unfold FlockC3.HoloValues.holoRepShape at h
  exact append_msg (zcShape_msg h3 h4 _) (append_msg (holoLcShape_msg h4 h5 _)
    (append_msg (opShape_msg h6 _) (ligShape_msg h7 h8 h0 h9))) h

end FlockC3.Compose
