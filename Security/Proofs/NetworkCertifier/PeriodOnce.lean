import Definitions.NetworkCertifier.Guarantees.PeriodOnce
import Proofs.NetworkCertifier.Period
import Proofs.NetworkCertifier.RecordRows

/-!
# The network certifier's rows read back its lanes from `(r, D)` alone

`Definitions.NetworkCertifier.Guarantees.PeriodOnce`. Connection `c`'s frames over the monitoring period's buckets, in
bucket order (`RowsFormat.framesOf`), are its padding frames, `r` a bucket from its admission to `adm c + D`, then its
lane's cells, which are in buckets from `adm c + D` on (`flatMap_range_append`). The padding numbers `r·D` whenever the
lane has a cell in the monitoring period, and at most that otherwise (`length_flatMap_pad`), and a lane's cells in
bucket order are the lane (`flatMap_range_filter`). So the lane is `c`'s frames past the first `r·D`
(`RowsFormat.readFree_record`), whatever its admission and its number of cells.
-/

namespace NetTiming

variable {κ Cell : Type}

/-- Bucket by bucket, `P`'s lists then `Q`'s, when no bucket's `P` list follows an earlier bucket's `Q` list. -/
theorem flatMap_range_append {β : Type} (P Q : ℕ → List β) (h : ∀ b b', b' < b → P b ≠ [] → Q b' = []) :
    ∀ N, (List.range N).flatMap (fun b => P b ++ Q b) = (List.range N).flatMap P ++ (List.range N).flatMap Q
  | 0 => by simp
  | N + 1 => by
    rw [List.range_succ, List.flatMap_append, List.flatMap_append, List.flatMap_append, flatMap_range_append P Q h N]
    simp only [List.flatMap_singleton]
    by_cases hP : P N = []
    · simp [hP]
    · have hQ : (List.range N).flatMap Q = [] :=
        List.flatMap_eq_nil_iff.2 fun b hb => h N b (List.mem_range.1 hb) hP
      simp [hQ]

theorem filter_fst_lt_succ {α : Type} : ∀ {l : List (ℕ × α)}, l.Pairwise (fun x y => x.1 ≤ y.1) → ∀ N,
    (l.filter fun x => x.1 < N) ++ (l.filter fun x => x.1 == N) = l.filter fun x => x.1 < N + 1
  | [], _, _ => by simp
  | x :: xs, hl, N => by
    obtain ⟨hx, hxs⟩ := List.pairwise_cons.1 hl
    have ih := filter_fst_lt_succ hxs N
    rcases Nat.lt_trichotomy x.1 N with h | h | h
    · have e1 : decide (x.1 < N) = true := by simpa using h
      have e2 : (x.1 == N) = false := by simp; omega
      have e3 : decide (x.1 < N + 1) = true := by simp; omega
      simp only [List.filter_cons, e1, e2, e3, ↓reduceIte, Bool.false_eq_true, List.cons_append, ih]
    · have hnil : xs.filter (fun y => y.1 < N) = [] :=
        List.filter_eq_nil_iff.2 fun y hy => by have := hx y hy; simp; omega
      rw [hnil, List.nil_append] at ih
      have e1 : decide (x.1 < N) = false := by simp; omega
      have e2 : (x.1 == N) = true := by simp; omega
      have e3 : decide (x.1 < N + 1) = true := by simp; omega
      simp only [List.filter_cons, e1, e2, e3, ↓reduceIte, Bool.false_eq_true, hnil, List.nil_append, ih]
    · have e1 : decide (x.1 < N) = false := by simp; omega
      have e2 : (x.1 == N) = false := by simp; omega
      have e3 : decide (x.1 < N + 1) = false := by simp; omega
      simp only [List.filter_cons, e1, e2, e3, ↓reduceIte, Bool.false_eq_true, ih]

/-- **A list sorted by its first components, read bucket by bucket**: its elements below `N`, in order. -/
theorem flatMap_range_filter {α : Type} {l : List (ℕ × α)} (hl : l.Pairwise fun x y => x.1 ≤ y.1) :
    ∀ N, (List.range N).flatMap (fun b => l.filter fun x => x.1 == b) = l.filter fun x => x.1 < N
  | 0 => by simp
  | N + 1 => by
    rw [List.range_succ, List.flatMap_append, flatMap_range_filter hl N, List.flatMap_singleton]
    exact filter_fst_lt_succ hl N

/-- The padding of a lane admitted at `A`, `r` frames a bucket for `D` buckets, over the first `N` buckets. -/
theorem length_flatMap_pad {β γ : Type} (r A D : ℕ) (x : β) (g : ℕ → β → γ) : ∀ N,
    ((List.range N).flatMap fun b => (if A ≤ b ∧ b < A + D then List.replicate r x else []).map (g b)).length =
      r * (min N (A + D) - A)
  | 0 => by simp
  | N + 1 => by
    rw [List.range_succ, List.flatMap_append, List.length_append, length_flatMap_pad r A D x g N]
    simp only [List.flatMap_singleton, List.length_map]
    by_cases h : A ≤ N ∧ N < A + D
    · rw [ite_eq_left h, List.length_replicate, ← Nat.mul_succ]
      congr 1
      omega
    · rw [ite_eq_right h, List.length_nil, Nat.add_zero]
      congr 1
      omega

/-- A lane's cells are in the order of their buckets. -/
theorem wardenCode_pairwise (s : WardenSchedule κ) (a : κ → Arrivals Cell) (c : κ) :
    (wardenCode s a c).Pairwise fun x y => x.1 ≤ y.1 := by
  have h : ((wardenCode s a c).map Prod.fst).Pairwise (· ≤ ·) := by
    rw [wardenCode_fst, List.pairwise_map]
    exact List.pairwise_lt_range.imp fun hij => Nat.add_le_add_left (Nat.div_le_div_right hij.le) _
  exact List.pairwise_map.1 h

theorem wardenCode_mem_fst (s : WardenSchedule κ) (a : κ → Arrivals Cell) (c : κ) {x : ℕ × CellView Cell}
    (hx : x ∈ wardenCode s a c) : ∃ j < s.cells c, x.1 = s.bucket c j := by
  have hm : x.1 ∈ (wardenCode s a c).map Prod.fst := List.mem_map_of_mem hx
  rw [wardenCode_fst] at hm
  obtain ⟨j, hj, hjx⟩ := List.mem_map.1 hm
  exact ⟨j, List.mem_range.1 hj, hjx.symm⟩

namespace RowsFormat

variable {L K : ℕ}

/-- Connection `c`'s frames over the monitoring period's `n·T` buckets, in bucket order, each with its bucket. -/
def framesOf [DecidableEq κ] {S n : ℕ} (F : RowsFormat κ Cell L K) (o : Fin n → List (Obs L K S)) (c : κ) :
    List (ℕ × CellView Cell) :=
  (List.range (n * F.T)).flatMap fun b =>
    ((F.frames o b).filterMap fun f => if f.1 = c then some f.2 else none).map (Prod.mk b)

/-- **The lanes read back from the record and `(r, D)` alone**: connection `c`'s frames past the first `r·D`. -/
def readFree [DecidableEq κ] {S n : ℕ} (F : RowsFormat κ Cell L K) (θ : ℕ × ℕ) (o : Fin n → List (Obs L K S))
    (c : κ) : List (ℕ × CellView Cell) :=
  (F.framesOf o c).drop (θ.1 * θ.2)

/-- **The rows format reads back the code's cells from the record and the schedule's `(r, D)`** on every schedule
that fits it. -/
theorem readFree_record [DecidableEq κ] (F : RowsFormat κ Cell L K) (hF : F.Wf) (S n : ℕ) (s : WardenSchedule κ)
    (hs : F.Fits S n s) (a : κ → Arrivals Cell) (c : κ) :
    F.readFree (s.r, s.D) (F.record S n s a (wardenCode s a)) c = wardenCode s a c := by
  obtain ⟨hnd, hall, hdec⟩ := hF
  have hin : ∀ x ∈ wardenCode s a c, s.adm c + s.D ≤ x.1 ∧ x.1 < n * F.T := fun x hx => by
    obtain ⟨j, hj, hjx⟩ := wardenCode_mem_fst s a c hx
    rw [hjx]
    exact ⟨by rw [WardenSchedule.bucket]; exact Nat.le_add_right _ _, hs.1 c j hj⟩
  let P : ℕ → List (ℕ × CellView Cell) := fun b =>
    (if s.adm c ≤ b ∧ b < s.adm c + s.D then List.replicate s.r CellView.pad else []).map (Prod.mk b)
  let Q : ℕ → List (ℕ × CellView Cell) := fun b => (wardenCode s a c).filter fun x => x.1 == b
  have hbucket : ∀ b ∈ List.range (n * F.T),
      ((F.frames (F.record S n s a (wardenCode s a)) b).filterMap fun f => if f.1 = c then some f.2 else none).map
        (Prod.mk b) = P b ++ Q b := by
    intro b hb
    rw [frames_record F hdec S n s a _ (List.mem_range.1 hb) ((length_bucketFrames _ _ _ _).trans_le (hs.2 _)),
      filterMap_bucketFrames hnd (hall c), laneFrames, List.map_append, List.map_map]
    show _ ++ _ = (if s.adm c ≤ b ∧ b < s.adm c + s.D then List.replicate s.r CellView.pad else []).map (Prod.mk b) ++
      (wardenCode s a c).filter fun x => x.1 == b
    congr 1
    refine (List.map_congr_left (g := id) fun x hx => ?_).trans (List.map_id _)
    have hxb : x.1 = b := by simpa using (List.mem_filter.1 hx).2
    exact Prod.ext hxb.symm rfl
  have hsplit : F.framesOf (F.record S n s a (wardenCode s a)) c =
      (List.range (n * F.T)).flatMap P ++ (List.range (n * F.T)).flatMap Q := by
    rw [framesOf, List.flatMap_congr hbucket]
    refine flatMap_range_append P Q (fun b b' hbb' hP => ?_) _
    have hb : b < s.adm c + s.D := by
      by_contra h
      exact hP (by simp only [P, h, and_false, ↓reduceIte, List.map_nil])
    show (wardenCode s a c).filter (fun x => x.1 == b') = []
    refine List.filter_eq_nil_iff.2 fun x hx => ?_
    have := (hin x hx).1
    simp only [beq_iff_eq]
    omega
  have hQ : (List.range (n * F.T)).flatMap Q = wardenCode s a c := by
    show (List.range (n * F.T)).flatMap (fun b => (wardenCode s a c).filter fun x => x.1 == b) = _
    rw [flatMap_range_filter (wardenCode_pairwise s a c)]
    exact List.filter_eq_self.2 fun x hx => by simpa using (hin x hx).2
  have hP : ((List.range (n * F.T)).flatMap P).length = s.r * (min (n * F.T) (s.adm c + s.D) - s.adm c) :=
    length_flatMap_pad s.r (s.adm c) s.D CellView.pad Prod.mk _
  rw [readFree, hsplit, hQ]
  show List.drop (s.r * s.D) _ = _
  rcases (wardenCode s a c).eq_nil_or_concat with h | ⟨l, x, hx⟩
  · rw [h, List.append_nil]
    refine List.drop_eq_nil_of_le ?_
    rw [hP]
    exact Nat.mul_le_mul_left _ (by omega)
  · have hxin := hin x (by rw [hx]; simp)
    refine List.drop_left' ?_
    rw [hP]
    congr 1
    omega

end RowsFormat

open Claims Claims.Sampling Finset

section Deploy

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- A window that meets `WindowGood` has its partition at its registration. -/
theorem partition_of_windowGood {H : Msg → Msg} {D : Deploy Coin Col T} {L K S : ℕ}
    {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ} {reg : AuditStatement → Finset ℕ}
    {Dj : ℕ} {s₀ : AuditStatement} {ρ : ℕ → Bool} {tops : AuditStatement → List (Opening Bool)} {x : Msg} {w : Wit}
    {o : List (Obs L K S)} (h : WindowGood H D L K S lay k δ reg Dj s₀ ρ tops x w o) :
    D.partition s₀ = some (D.part s₀) := by
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, -⟩, -, s, hs, ⟨-, -, -, -, -, hpart⟩, -⟩ := h
  rw [Option.some_injective _ (hs₁.symm.trans hs)] at hP₁
  rw [Deploy.part, ← hpart, hP₁]
  rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem encard_periodOnce_le (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (reg : AuditStatement → Finset ℕ)
    (Dj : ℕ) {n : ℕ} (sw : Fin n → AuditStatement) (ρ : Fin n → ℕ → Bool)
    (op : Fin n → ℕ → List (Obs L K S) → List Bool)
    {Conn Cell Srv Dev Z : Type} (arr : Srv → Conn → Arrivals Cell) (F : RowsFormat Conn Cell L K) (hF : F.Wf)
    (tap : Dev)
    (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p =>
      (wardenCode p.1 (arr p.2), F.record S n p.1 (arr p.2) (wardenCode p.1 (arr p.2))))
    (Θ : Set (ℕ × ℕ)) (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z)
    (x : Z → Fin n → Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool))
    (htap : ∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → TapOpens H D.commit (lay s) (k s) (op i) s (tops s)
      ((run tap (schedOf z, srvOf z)).2 i)) :
    {z | z ∈ Zs ∧ ((schedOf z).r, (schedOf z).D) ∈ Θ ∧ F.Fits S n (schedOf z) ∧
        dec (run tap (schedOf z, srvOf z)).1 = z ∧
        ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
          ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
      Θ.encard * ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
        (faultySets (sw i).design (D.part (sw i)) δ).card *
        2 ^ ((faultySets (sw i).design (D.part (sw i)) δ).sup (faultyBits (sw i).prog (D.part (sw i)))) *
        Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
  classical
  set G := {z | z ∈ Zs ∧ ((schedOf z).r, (schedOf z).D) ∈ Θ ∧ F.Fits S n (schedOf z) ∧
    dec (run tap (schedOf z, srvOf z)).1 = z ∧
    ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
      ((run tap (schedOf z, srvOf z)).2 i)} with hG
  rcases G.eq_empty_or_nonempty with he | ⟨z₀, hz₀⟩
  · rw [he, Set.encard_empty]
    exact zero_le
  have hP : ∀ i, D.partition (sw i) = some (D.part (sw i)) := fun i => partition_of_windowGood (hz₀.2.2.2.2 i)
  let A : ∀ i : Fin n, Set (List (Obs L K S)) := fun i =>
    AccSeqsSampled (sw i).prog (D.part (sw i)) (sw i).design (lay (sw i)) (k (sw i))
      (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) δ (op i) Dj
  have hgood : ∀ z ∈ G, ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈ A i := fun z hz i =>
    (mem_accSeqsSampled_of_windowGood (hP i) (htap z hz.1 i) (hz.2.2.2.2 i)).2
  have hfits : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)) := fun i =>
    (mem_accSeqsSampled_of_windowGood (hP i) (htap z₀ hz₀.1 i) (hz₀.2.2.2.2 i)).1
  have hview : ∀ z ∈ G, (run tap (schedOf z, srvOf z)).1 =
      F.readFree ((schedOf z).r, (schedOf z).D) (run tap (schedOf z, srvOf z)).2 := fun z hz => by
    rw [htrust (schedOf z, srvOf z)]
    funext c
    exact (F.readFree_record hF S n (schedOf z) hz.2.2.1 (arr (srvOf z)) c).symm
  let V : Set (Conn → List (ℕ × CellView Cell)) :=
    (fun p : (ℕ × ℕ) × (Fin n → List (Obs L K S)) => F.readFree p.1 p.2) '' (Θ ×ˢ Set.pi Set.univ A)
  have hsub : G ⊆ dec '' V := fun z hz =>
    ⟨(run tap (schedOf z, srvOf z)).1, ⟨(((schedOf z).r, (schedOf z).D), (run tap (schedOf z, srvOf z)).2),
      ⟨hz.2.1, fun i _ => hgood z hz i⟩, (hview z hz).symm⟩, hz.2.2.2.1⟩
  have hA : (Set.pi Set.univ A).encard ≤ ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
      reg (sw i)).card * (faultySets (sw i).design (D.part (sw i)) δ).card *
      2 ^ ((faultySets (sw i).design (D.part (sw i)) δ).sup (faultyBits (sw i).prog (D.part (sw i)))) *
      Dj ^ k (sw i)) : ℕ) : ℕ∞) := by
    calc (Set.pi Set.univ A).encard = ∏ i, (A i).encard := Set.encard_pi_eq_prod_encard
      _ ≤ ∏ i, ((2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card *
          (faultySets (sw i).design (D.part (sw i)) δ).card *
          2 ^ ((faultySets (sw i).design (D.part (sw i)) δ).sup (faultyBits (sw i).prog (D.part (sw i)))) *
          Dj ^ k (sw i) : ℕ) : ℕ∞) :=
        Finset.prod_le_prod fun i _ => encard_accSeqsSampled_le _ _ _ _ _ (hfits i) δ _ _
          (encard_asReg_le _ _ _ _ _) _ (fun _ hF => Finset.le_sup hF) (op i) Dj
      _ = _ := (Nat.cast_prod _ _).symm
  calc G.encard ≤ (dec '' V).encard := Set.encard_le_encard hsub
    _ ≤ V.encard := Set.encard_image_le _ _
    _ ≤ (Θ ×ˢ Set.pi Set.univ A).encard := Set.encard_image_le _ _
    _ = Θ.encard * (Set.pi Set.univ A).encard := Set.encard_prod
    _ ≤ _ := by gcongr

/-- **The leak through the network certifier over a monitoring period, each bit charged once**:
`Guarantees.PeriodLeakOnce`, from `sampledLeak_tap` with the registered inputs pinned (`adviceIn_asReg`), and
`encard_periodOnce_le`. -/
theorem periodLeakOnce : Guarantees.PeriodLeakOnce := by
  intro Coin Col T _ _ _ _ _ H D hks L K S lay k δ decode hδ hdec reg Dj n sw ρ op Conn Cell Srv Dev Z arr F hF tap
    run htrust Θ Zs schedOf srvOf dec x wit tops hb
  exact ⟨fun m hm Ω _ _ P => TapProof.sampledLeak_tap H D hks hb L K S lay k δ decode hδ hdec _ Dj
    (fun _ _ _ _ => adviceIn_asReg K _ _ _ _ _ _ _ _) tops m hm Ω P,
    encard_periodOnce_le H D L K S lay k δ reg Dj sw ρ op arr F hF tap run htrust Θ Zs schedOf srvOf dec x wit tops⟩

end Deploy

end NetTiming
