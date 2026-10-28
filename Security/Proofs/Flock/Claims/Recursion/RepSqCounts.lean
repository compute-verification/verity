import Proofs.Flock.Claims.Recursion.NextRFits
import Proofs.Flock.Soundness.Refine.Seam

/-!
# A rep's coin counts: the frames' squeezes are the schedule's

`FramedAt` asks of each round that it count its chunk's closing squeeze's coins (`sqCount`). The reader's rounds count
their schedule's (`recStreams`: `RecDeploy.repSchedAt`, which is `RecVStar.repRounds`). The two agree round by round
at a `fast100` schedule (`sqCount_repFrames`):

* **The frames' side** (`sqs`): a chunk ends in the squeeze that closes it, so the chunks' counts are the closing
  squeezes' counts, in order (`sqCount_chunks`), and they compose over `++`. Each part of a rep's frames has its
  closed list (`sqs_zc`, `sqs_lc`, `sqs_rs`, `sqs_levelFr`, `sqs_lig`), whatever the proof's values.
* **The schedule's side** (`schedN`): `RecVStar.repRounds`' coin counts, round by round (`repRounds_sqs`), at a
  schedule whose levels' columns shrink by three a level (`DimSeam`, as every `fast100` schedule's do).
* **The queries**: `RecVStar.bitLength (q − 1)` is the frames' `log₂ (q − 1) + 1` at `q ≥ 2`, which every `fast100`
  level has (`fast100_q_two`).
-/

namespace FlockDeploy

open Flock
open FlockSoundness.Refine (Frame DimSeam)
open FlockSoundness.Discharge.Exec (chunks closes repFrames zcFrames lcFrames rsFrames ligFrames levelFr introFr
  queryFr chFr kFr capFr)
open FlockVBridge.Family (sqCount)

/-! ## The frames' side -/

/-- A frame's coin count where it closes a round. -/
def sqOf : Frame → Option ℕ
  | .sq _ n => if n = 0 then none else some n
  | _ => none

/-- The coin counts of the squeezes that close a frame list's rounds, in order. -/
def sqs (fs : List Frame) : List ℕ := fs.filterMap sqOf

theorem sqs_nil : sqs [] = [] := rfl

theorem sqs_append (a b : List Frame) : sqs (a ++ b) = sqs a ++ sqs b := List.filterMap_append

theorem sqs_cons (f : Frame) (fs : List Frame) : sqs (f :: fs) = (sqOf f).toList ++ sqs fs := by
  unfold sqs
  cases h : sqOf f <;> simp [h]

theorem chunks_ne_nil : ∀ (fs : List Frame), ∀ c ∈ (chunks fs).1, c ≠ []
  | [] => by simp [chunks]
  | f :: fs => by
    have ih := chunks_ne_nil fs
    intro c hc
    cases hf : closes f
    · simp only [chunks, hf, Bool.false_eq_true, ↓reduceIte] at hc
      generalize chunks fs = x at hc ih
      rcases x with ⟨_ | ⟨c', cs⟩, t⟩
      · change c ∈ ([] : List (List Frame)) at hc
        simp at hc
      · change c ∈ (f :: c') :: cs at hc
        rcases List.mem_cons.1 hc with rfl | hc
        · exact List.cons_ne_nil _ _
        · exact ih c (List.mem_cons_of_mem _ hc)
    · simp only [chunks, hf, ↓reduceIte] at hc
      rcases List.mem_cons.1 hc with rfl | hc
      · exact List.cons_ne_nil _ _
      · exact ih c hc

theorem sqCount_cons (f : Frame) : ∀ {c : List Frame}, c ≠ [] → sqCount (f :: c) = sqCount c
  | [], h => absurd rfl h
  | _ :: _, _ => by simp only [sqCount, List.getLast?_cons_cons]

theorem sqOf_none {f : Frame} (h : closes f = false) : sqOf f = none := by
  cases f with
  | sq k n =>
    have : n = 0 := by simpa [closes] using h
    simp [sqOf, this]
  | _ => rfl

/-- **The chunks' coin counts are the closing squeezes'.** -/
theorem sqCount_chunks : ∀ (fs : List Frame), (chunks fs).1.map sqCount = sqs fs
  | [] => rfl
  | f :: fs => by
    have ih := sqCount_chunks fs
    have hne := chunks_ne_nil fs
    rw [sqs_cons]
    cases hf : closes f
    · simp only [chunks, hf, Bool.false_eq_true, ↓reduceIte]
      rw [sqOf_none hf, Option.toList_none, List.nil_append, ← ih]
      generalize chunks fs = x at hne ⊢
      rcases x with ⟨_ | ⟨c, cs⟩, t⟩
      · rfl
      · show sqCount (f :: c) :: cs.map sqCount = sqCount c :: cs.map sqCount
        rw [sqCount_cons f (hne c List.mem_cons_self)]
    · cases f with
      | sq k n =>
        have hn : n ≠ 0 := by simpa [closes] using hf
        simp only [chunks, hf, ↓reduceIte, List.map_cons, ih]
        simp [sqOf, hn, sqCount]
      | _ => simp [closes] at hf

theorem sqs_flatMap {α : Type} (F : α → List Frame) : ∀ (L : List α), sqs (L.flatMap F) = L.flatMap fun a => sqs (F a)
  | [] => rfl
  | a :: L => by rw [List.flatMap_cons, sqs_append, sqs_flatMap F L, List.flatMap_cons]

private theorem flatMap_range'_congr {β : Type} {F G : ℕ → List β} :
    ∀ a k, (∀ i, a ≤ i → i < a + k → F i = G i) → (List.range' a k).flatMap F = (List.range' a k).flatMap G
  | _, 0, _ => rfl
  | a, k + 1, h => by
    rw [List.range'_succ, List.flatMap_cons, List.flatMap_cons, h a (Nat.le_refl a) (by omega),
      flatMap_range'_congr (a + 1) k fun i h1 h2 => h i (by omega) (by omega)]

private theorem flatMap_range'_const {β : Type} (b : β) : ∀ a k, (List.range' a k).flatMap (fun _ => [b]) = List.replicate k b
  | _, 0 => rfl
  | a, k + 1 => by rw [List.range'_succ, List.flatMap_cons, flatMap_range'_const b (a + 1) k]; rfl

theorem sqs_map_one : ∀ l : List F128, sqs (l.map Frame.one) = []
  | [] => rfl
  | v :: l => by rw [List.map_cons, sqs_cons, sqs_map_one l]; rfl

theorem sqs_kFr (m : F256 × F256) : sqs (kFr m) = [] := rfl

theorem sqs_chFr (p : Flock.LigeritoProof) : ∀ (k msg : ℕ), sqs (chFr p msg k) = List.replicate k 2
  | 0, _ => rfl
  | k + 1, msg => by
    rw [chFr, List.cons_append, sqs_cons, sqs_append, sqs_kFr, sqs_chFr p k (msg + 1)]
    rfl

theorem sqs_introFr (p : Flock.LigeritoProof) (dim ood msg cbn : ℕ) :
    sqs (introFr p (dim + 1) ood msg cbn) = [dim + 1, 1] := by
  unfold introFr
  simp [sqs_cons, sqs_append, sqs_kFr, sqOf, sqs_nil]

theorem sqs_queryFr (p : Flock.LigeritoProof) {lv : Flock.Level} (hq : lv.q ≠ 0) (qn cn : ℕ) :
    sqs (queryFr p lv qn cn) = [lv.q, (lv.q - 1).log2 + 1] := by
  unfold queryFr
  simp [sqs_cons, sqOf, hq, sqs_nil]

theorem sqs_zc {m : ℕ} (hm : 14 ≤ m) (digest cap : ByteArray) (p : Flock.ZerocheckProof) :
    sqs (zcFrames m digest cap p) = [6, m - 13, 1] ++ List.replicate (m - 6) 1 := by
  have h13 : m - 13 ≠ 0 := by omega
  unfold zcFrames
  rw [sqs_append, sqs_append, sqs_flatMap]
  rw [flatMap_range'_congr (G := fun _ => [1]) 0 (m - 6) (fun i _ _ => by simp [sqs_cons, sqOf, sqs_nil]),
    flatMap_range'_const]
  simp [sqs_cons, sqOf, h13, sqs_nil]

theorem sqs_lc (k : ℕ) (p : Flock.LincheckProof) :
    sqs (lcFrames k p) = [1, 1] ++ List.replicate (k - 6) 1 ++ [1] := by
  unfold lcFrames
  rw [sqs_append, sqs_append, sqs_flatMap]
  rw [flatMap_range'_congr (G := fun _ => [1]) 0 (k - 6) (fun i _ _ => by simp [sqs_cons, sqOf, sqs_nil]),
    flatMap_range'_const]
  simp [sqs_cons, sqOf, sqs_nil]

theorem sqs_rs {n : ℕ} (hn : n ≠ 0) (rs : Array (Array F128)) :
    sqs (rsFrames n rs) = List.replicate n 7 ++ [n] := by
  unfold rsFrames
  rw [sqs_cons, sqs_append, sqs_flatMap]
  rw [flatMap_range'_congr (G := fun _ => [7]) 0 n (fun i _ _ => by simp [sqs_cons, sqOf, sqs_nil]),
    flatMap_range'_const]
  simp [sqOf, sqs_cons, hn, sqs_nil]

/-! ## The schedule's side -/

/-- Level `i`'s seven rounds' coin counts at `split` columns. -/
def lvN (sch : Flock.Schedule) (i split : ℕ) : List ℕ :=
  [split, 1, split, 1, (sch.levels.getD i default).q, RecVStar.bitLength ((sch.levels.getD i default).q - 1), 1]

/-- Pass `i` of the levels' loop, of `r` passes: four folding rounds, then level `i + 1`'s or the last level's. -/
def passN (sch : Flock.Schedule) (r i : ℕ) : List ℕ :=
  [2, 2, 2, 2] ++
    if i + 1 < r then lvN sch (i + 1) ((sch.levels.getD (i + 1) default).cols + 1)
    else [(sch.levels.getD r default).q, RecVStar.bitLength ((sch.levels.getD r default).q - 1), 1]

/-- **`RecVStar.repRounds`' coin counts**, round by round. -/
def schedN (sch : Flock.Schedule) (m kLog claims : ℕ) : List ℕ :=
  [6, m - 13, 1] ++ List.replicate (m - 5) 1 ++ [1] ++ List.replicate (kLog - 6) 1 ++ [1] ++
    List.replicate claims 7 ++ [claims] ++ [sch.n, 1] ++ List.replicate sch.k0 2 ++
    lvN sch 0 (sch.n - sch.k0 + 1) ++ (List.range' 0 (sch.levels.size - 1)).flatMap (passN sch (sch.levels.size - 1))

/-- A loop over `range' a k` whose steps keep an invariant and each append `g i`. -/
theorem forIn_range'_inv {σ β : Type} (f : ℕ → σ → Option (ForInStep σ)) (π : σ → List β) (g : ℕ → List β)
    (I : ℕ → σ → Prop)
    (hf : ∀ i s t, I i s → f i s = some t → ∃ u, t = .yield u ∧ π u = π s ++ g i ∧ I (i + 1) u) :
    ∀ (k a : ℕ) (s0 s : σ), I a s0 → forIn (List.range' a k) s0 f = some s →
      π s = π s0 ++ (List.range' a k).flatMap g
  | 0, a, s0, s, _, h => by simp at h; subst h; simp
  | k + 1, a, s0, s, hI, h => by
    rw [List.range'_succ, List.forIn_cons] at h
    cases hfa : f a s0 with
    | none => rw [hfa] at h; cases h
    | some t =>
      rw [hfa] at h
      obtain ⟨u, rfl, hu, hI'⟩ := hf a s0 t hI hfa
      rw [forIn_range'_inv f π g I hf k (a + 1) u s hI' h, hu, List.range'_succ, List.flatMap_cons,
        List.append_assoc]

/-- **`RecVStar.repRounds`' coin counts are `schedN`'s**, at a schedule whose columns shrink by three a level. -/
theorem repRounds_sqs {m kLog claims : ℕ} {sch : Flock.Schedule} {rs : Array RecVStar.SchedRound}
    (hs : fast100 m = some sch) (hd : DimSeam sch) (h : RecVStar.repRounds m kLog claims = some rs) :
    rs.toList.map (·.1) = schedN sch m kLog claims := by
  obtain ⟨-, -, -, hc0, -, hcs, -⟩ := hd
  simp only [Array.getD_eq_getD_getElem?] at hc0 hcs
  unfold RecVStar.repRounds at h
  simp only [hs, Std.Legacy.Range.forIn_eq_forIn_range'] at h
  simp only [Option.bind_eq_bind, Option.pure_def, Option.bind_some] at h
  simp only [Option.bind_eq_some_iff] at h
  obtain ⟨s, hl, he⟩ := h
  cases he
  have key := forIn_range'_inv _ (fun st : Array RecVStar.SchedRound × ℕ => st.1.toList.map (·.1))
    (passN sch (sch.levels.size - 1))
    (fun i st => i < sch.levels.size - 1 → st.2 = (sch.levels.getD i default).cols + 1)
    (fun i s t hI ht => by
      by_cases h4 : s.2 < 4
      · simp [h4] at ht
      · by_cases hi : i + 1 < sch.levels.size - 1
        · simp [h4, hi] at ht
          have hs2 := hI (by omega)
          simp only [Array.getD_eq_getD_getElem?] at hs2
          have hc := hcs i (by omega)
          have hsp : s.2 - 4 = (sch.levels[i + 1]?.getD default).cols := by omega
          refine ⟨_, ht.symm, ?_, fun _ => ?_⟩
          · simp [passN, lvN, hi, rep3, hsp]
          · simp only [Array.getD_eq_getD_getElem?]
            omega
        · simp [h4, hi] at ht
          refine ⟨_, ht.symm, ?_, fun h' => absurd h' (by omega)⟩
          simp [passN, hi, rep3]) _ _ _ _ (fun _ => by simp only [Array.getD_eq_getD_getElem?]; omega) hl
  refine key.trans ?_
  simp [schedN, lvN]

/-! ## The two sides agree -/

theorem bitLength_pred {q : ℕ} (hq : 2 ≤ q) : RecVStar.bitLength (q - 1) = (q - 1).log2 + 1 := by
  unfold RecVStar.bitLength
  have : (q - 1 == 0) = false := by simp; omega
  rw [this]
  rfl

/-- **Every `fast100` level queries at least twice.** -/
theorem fast100_q_two {m : ℕ} {sch : Flock.Schedule} (hs : fast100 m = some sch) :
    ∀ i < sch.levels.size, 2 ≤ (sch.levels.getD i default).q := by
  have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, (s.levels.all fun lv => decide (2 ≤ lv.q)) = true := by
    decide +kernel
  have hm : m < 40 := by
    by_contra h
    rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at hs
    cases hs
  intro i hi
  have h := Array.all_eq_true.1 (key ⟨m, hm⟩ sch hs) i hi
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
  simpa using h

theorem sqs_levelFr {sch : Flock.Schedule} (p : Flock.LigeritoProof)
    (hq : ∀ i < sch.levels.size, 2 ≤ (sch.levels.getD i default).q) {i : ℕ} (hi : i < sch.levels.size - 1) :
    sqs (levelFr sch p i) = passN sch (sch.levels.size - 1) i := by
  unfold levelFr passN
  rw [sqs_append, sqs_chFr]
  by_cases hc : i + 1 < sch.levels.size - 1
  · have hq1 := hq (i + 1) (by omega)
    simp only [hc, ↓reduceIte, sqs_cons, sqs_append, sqs_introFr,
      sqs_queryFr p (lv := sch.levels.getD (i + 1) default) (by omega), sqs_kFr, lvN, bitLength_pred hq1]
    simp [sqOf, capFr, sqs_nil]
  · have hq1 := hq (sch.levels.size - 1) (by omega)
    simp only [hc, ↓reduceIte, sqs_append, sqs_map_one,
      sqs_queryFr p (lv := sch.levels.getD (sch.levels.size - 1) default) (by omega), bitLength_pred hq1]
    simp [sqs_cons, sqOf, sqs_nil]

theorem sqs_lig {sch : Flock.Schedule} (T : F128) (cap : Array ByteArray) (p : Flock.LigeritoProof)
    (hq : ∀ i < sch.levels.size, 2 ≤ (sch.levels.getD i default).q) (hn : sch.n ≠ 0) (hL : 1 ≤ sch.levels.size) :
    sqs (ligFrames sch T cap p) = [sch.n, 1] ++ List.replicate sch.k0 2 ++ lvN sch 0 (sch.n - sch.k0 + 1) ++
      (List.range' 0 (sch.levels.size - 1)).flatMap (passN sch (sch.levels.size - 1)) := by
  have hq0 := hq 0 (by omega)
  unfold ligFrames
  simp only [sqs_append, sqs_cons, List.cons_append, sqs_kFr, sqs_chFr, sqs_flatMap]
  rw [flatMap_range'_congr (G := passN sch (sch.levels.size - 1)) 0 (sch.levels.size - 1)
    (fun i _ h => sqs_levelFr p hq (by omega))]
  simp only [sqs_introFr p (sch.n - sch.k0), sqs_queryFr p (lv := sch.levels.getD 0 default) (by omega), lvN,
    bitLength_pred hq0]
  simp [sqOf, hn, capFr, sqs_nil]

/-- **A rep's frames' coin counts are its schedule's**, at a `fast100` schedule of the setup's `m`. -/
theorem sqs_repFrames {st : Flock.Setup} {sch : Flock.Schedule} (d : ByteArray) (e : ℕ) (p : Flock.Proof) (T : F128)
    (hs : fast100 st.m = some sch) :
    sqs (repFrames st sch d (2 + e) p T) = schedN sch st.m st.fold.kLog (2 + e) := by
  obtain ⟨hd, hm⟩ := FlockSoundness.Refine.fast100_dims _ _ hs
  have hsm := (FlockSoundness.Discharge.DecodeZL.fast100_small _ _ hs).1
  have hq := fast100_q_two hs
  have hn : sch.n ≠ 0 := by rw [hd.1]; omega
  have hL : 1 ≤ sch.levels.size := by have := hd.2.1; omega
  have h5 : List.replicate (st.m - 5) 1 = List.replicate (st.m - 6) 1 ++ [1] := by
    rw [show st.m - 5 = st.m - 6 + 1 by omega, List.replicate_succ']
  unfold repFrames schedN
  rw [sqs_cons, sqs_append, sqs_append, sqs_append, sqs_zc (by omega), sqs_lc, sqs_rs (by omega),
    sqs_lig T p.cap p.ligerito hq hn hL, h5]
  simp [sqOf]

/-- **Round `j` of a rep counts its chunk's closing squeeze's coins**: at a `fast100` schedule of the setup's `m` and a
schedule `rs` that is `RecVStar.repRounds`' at the frames' `m`, `k_log` and claim count. -/
theorem sqCount_repFrames {st : Flock.Setup} {sch : Flock.Schedule} (d : ByteArray) (e : ℕ) (p : Flock.Proof)
    (T : F128) {rs : Array RecVStar.SchedRound} (hs : fast100 st.m = some sch)
    (hr : RecVStar.repRounds st.m st.fold.kLog (2 + e) = some rs) {j : ℕ}
    (hj : j < (chunks (repFrames st sch d (2 + e) p T)).1.length) :
    sqCount ((chunks (repFrames st sch d (2 + e) p T)).1.getD j []) = (rs.toList.getD j default).1 := by
  have h1 := sqCount_chunks (repFrames st sch d (2 + e) p T)
  rw [sqs_repFrames d e p T hs, ← repRounds_sqs hs (FlockSoundness.Refine.fast100_dims _ _ hs).1 hr] at h1
  have hl := congrArg List.length h1
  simp only [List.length_map] at hl
  have h3 := List.getElem_of_eq h1 (i := j) (by rw [List.length_map]; exact hj)
  simp only [List.getElem_map] at h3
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, Option.getD_some,
    List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega), Option.getD_some]
  exact h3

end FlockDeploy
