import Proofs.Flock.Claims.Recursion.NextRFits
import Proofs.Flock.VBridge.Family.Runs

/-!
# A rep's rounds squeeze as its schedule's: `chunksOf_counts`

Each round of a rep's frames closes at a squeeze, and its coin count (`sqCount`, what `FramedAt` asks each round to
issue) is the count `RecVStar.repRounds` schedules for that round (`chunksOf_counts`). This is `NextRFits`' `roundPat`
for the counts:

* **The frames' side** (`closeNs`): the counts of a frame list's closing squeezes, in order, are its chunks' counts
  (`chunks_counts`), and they compose over `++` and `flatMap` with no glue. Each part of a rep's frames has closed
  counts (`closeNs_zc`, `closeNs_lc`, `closeNs_rs`, `closeNs_levelFr`, `closeNs_lig`), so a rep's frames have
  `schedCnt`'s (`closeNs_repFrames`).
* **The schedule's side** (`repRounds_counts`): `RecVStar.repRounds`' counts are `schedCnt`'s. Its loop's running
  split is each level's columns plus one, which `fast100` gives (`fast100_cols`, with every level's `q ≥ 2`, where
  `bitLength (q - 1)` is the frames' `(q - 1).log2 + 1`).
-/

namespace FlockDeploy

open Flock
open FlockSoundness.Refine (Frame)
open FlockSoundness.Discharge.Exec (chunks closes repFrames zcFrames lcFrames rsFrames ligFrames levelFr introFr
  queryFr chFr kFr capFr)
open FlockVBridge.Family (sqCount)

/-! ## The frames' side -/

/-- The coins a frame closes a round with: `[n]` at a squeeze of `n ≥ 1` coins, none otherwise. -/
def sqN : Frame → List ℕ
  | .sq _ n => if n = 0 then [] else [n]
  | _ => []

/-- The coin counts of a frame list's closing squeezes, in order. -/
def closeNs (F : List Frame) : List ℕ := F.flatMap sqN

@[simp] theorem closeNs_nil : closeNs [] = [] := rfl

@[simp] theorem closeNs_cons (f : Frame) (F : List Frame) : closeNs (f :: F) = sqN f ++ closeNs F := by
  simp only [closeNs, List.flatMap_cons]

@[simp] theorem closeNs_append (F G : List Frame) : closeNs (F ++ G) = closeNs F ++ closeNs G := by
  simp only [closeNs, List.flatMap_append]

@[simp] theorem sqN_one (v : F128) : sqN (.one v) = [] := rfl
@[simp] theorem sqN_vec (vs : List F128) : sqN (.vec vs) = [] := rfl
@[simp] theorem sqN_dom (b : ByteArray) : sqN (.dom b) = [] := rfl
@[simp] theorem sqN_label (b : ByteArray) : sqN (.label b) = [] := rfl
@[simp] theorem sqN_bytes (b : ByteArray) : sqN (.bytes b) = [] := rfl
@[simp] theorem sqN_capFr (cap : Array ByteArray) : sqN (capFr cap) = [] := rfl
@[simp] theorem closeNs_kFr (m : F256 × F256) : closeNs (kFr m) = [] := rfl

theorem sqN_sq (k : UInt8) {n : ℕ} (hn : n ≠ 0) : sqN (.sq k n) = [n] := by
  simp [sqN, hn]

theorem sqN_open {f : Frame} (hf : closes f = false) : sqN f = [] := by
  cases f <;> simp_all [sqN, closes]

theorem sqN_closed {f : Frame} (hf : closes f = true) : sqN f = [sqCount [f]] := by
  cases f <;> simp_all [sqN, sqCount, closes]

theorem sqCount_open {f : Frame} (hf : closes f = false) (c : List Frame) : sqCount (f :: c) = sqCount c := by
  cases c with
  | nil => cases f <;> simp_all [sqCount, closes]
  | cons g c => simp only [sqCount, List.getLast?_cons_cons]

/-- **A frame list's rounds close at its closing squeezes**: its chunks' counts are its closing squeezes', in order. -/
theorem chunks_counts : ∀ F : List Frame, (chunks F).1.map sqCount = closeNs F
  | [] => rfl
  | f :: F => by
    have ih := chunks_counts F
    rw [closeNs_cons]
    by_cases hf : closes f = true
    · simp only [chunks, hf, ↓reduceIte, List.map_cons, ih, sqN_closed hf, List.singleton_append]
    · have hf' : closes f = false := by simpa using hf
      rw [sqN_open hf', List.nil_append, ← ih]
      simp only [chunks, hf', Bool.false_eq_true, ↓reduceIte]
      generalize (chunks F).1 = x
      rcases x with _ | ⟨c, cs⟩
      · rfl
      · simp only [List.map_cons, sqCount_open hf' c]

theorem closeNs_flatMap {α : Type} (F : α → List Frame) :
    ∀ l : List α, closeNs (l.flatMap F) = l.flatMap fun i => closeNs (F i)
  | [] => rfl
  | a :: l => by rw [List.flatMap_cons, List.flatMap_cons, closeNs_append, closeNs_flatMap F l]

theorem flatMap_range'_const (x : ℕ) : ∀ a k, (List.range' a k).flatMap (fun _ => [x]) = List.replicate k x
  | _, 0 => rfl
  | a, k + 1 => by rw [List.range'_succ, List.flatMap_cons, flatMap_range'_const x (a + 1) k]; rfl

theorem flatMap_range'_congr {β : Type} {f g : ℕ → List β} :
    ∀ a k, (∀ i, a ≤ i → i < a + k → f i = g i) → (List.range' a k).flatMap f = (List.range' a k).flatMap g
  | _, 0, _ => rfl
  | a, k + 1, h => by
    rw [List.range'_succ, List.flatMap_cons, List.flatMap_cons, h a (Nat.le_refl a) (by omega),
      flatMap_range'_congr (a + 1) k fun i h1 h2 => h i (by omega) (by omega)]

/-- **Blocks that each close one round of `x` coins.** -/
theorem closeNs_blocks {F : ℕ → List Frame} {x : ℕ} (hF : ∀ i, closeNs (F i) = [x]) (a k : ℕ) :
    closeNs ((List.range' a k).flatMap F) = List.replicate k x := by
  rw [closeNs_flatMap, show (fun i => closeNs (F i)) = fun _ => [x] from funext hF, flatMap_range'_const]

theorem closeNs_ones : ∀ l : List F128, closeNs (l.map Frame.one) = []
  | [] => rfl
  | v :: l => by rw [List.map_cons, closeNs_cons, sqN_one, closeNs_ones l]; rfl

theorem closeNs_chFr (p : Flock.LigeritoProof) : ∀ msg k, closeNs (chFr p msg k) = List.replicate k 2
  | _, 0 => rfl
  | msg, k + 1 => by
    rw [chFr, closeNs_cons, closeNs_append, closeNs_kFr, closeNs_chFr p (msg + 1) k, sqN_sq _ (by decide)]
    rfl

theorem closeNs_introFr (p : Flock.LigeritoProof) {dim : ℕ} (hd : dim ≠ 0) (ood msg cbn : ℕ) :
    closeNs (introFr p dim ood msg cbn) = [dim, 1] := by
  unfold introFr
  simp only [closeNs_cons, closeNs_append, closeNs_kFr, closeNs_nil, sqN_one, sqN_bytes, sqN_sq _ hd,
    sqN_sq _ Nat.one_ne_zero, List.nil_append, List.cons_append, List.append_nil]

theorem closeNs_queryFr (p : Flock.LigeritoProof) {lv : Flock.Level} (hq : 2 ≤ lv.q) (qn cn : ℕ) :
    closeNs (queryFr p lv qn cn) = [lv.q, RecVStar.bitLength (lv.q - 1)] := by
  have h1 : lv.q ≠ 0 := by omega
  have h2 : (lv.q - 1 == 0) = false := by simp; omega
  unfold queryFr RecVStar.bitLength
  simp only [closeNs_cons, closeNs_nil, sqN_bytes, sqN_sq _ h1, sqN_sq _ (Nat.add_one_ne_zero _), h2,
    Bool.false_eq_true, ↓reduceIte, List.nil_append, List.cons_append]

theorem closeNs_zc {m : ℕ} (hm : 14 ≤ m) (digest cap : ByteArray) (p : Flock.ZerocheckProof) :
    closeNs (zcFrames m digest cap p) = 6 :: (m - 13) :: 1 :: List.replicate (m - 6) 1 := by
  have hB := closeNs_blocks (F := fun i => [Frame.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1]) (x := 1)
    (fun i => by simp only [closeNs_cons, closeNs_nil, sqN_one, sqN_sq _ Nat.one_ne_zero, List.nil_append]) 0 (m - 6)
  unfold zcFrames
  simp only [closeNs_append, hB, closeNs_cons, closeNs_nil, sqN_label, sqN_bytes, sqN_vec, sqN_one,
    sqN_sq _ (by decide : (6 : ℕ) ≠ 0), sqN_sq _ (show m - 13 ≠ 0 by omega), sqN_sq _ Nat.one_ne_zero,
    List.nil_append, List.append_nil, List.cons_append]

theorem closeNs_lc (k : ℕ) (p : Flock.LincheckProof) :
    closeNs (lcFrames k p) = 1 :: 1 :: (List.replicate (k - 6) 1 ++ [1]) := by
  have hB := closeNs_blocks (F := fun j => [Frame.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1]) (x := 1)
    (fun i => by simp only [closeNs_cons, closeNs_nil, sqN_one, sqN_sq _ Nat.one_ne_zero, List.nil_append]) 0 (k - 6)
  unfold lcFrames
  simp only [closeNs_append, hB, closeNs_cons, closeNs_nil, sqN_label, sqN_vec, sqN_sq _ Nat.one_ne_zero,
    List.nil_append, List.cons_append]

theorem closeNs_rs {n : ℕ} (hn : n ≠ 0) (rs : Array (Array F128)) :
    closeNs (rsFrames n rs) = List.replicate n 7 ++ [n] := by
  have hB := closeNs_blocks
    (F := fun k => [Frame.label "flock-ring-switch-v0".toUTF8, .vec (rs[k]!).toList, .sq 2 7]) (x := 7)
    (fun i => by simp only [closeNs_cons, closeNs_nil, sqN_label, sqN_vec, sqN_sq _ (by decide : (7 : ℕ) ≠ 0),
      List.nil_append]) 0 n
  unfold rsFrames
  simp only [closeNs_cons, closeNs_append, hB, closeNs_nil, sqN_label, sqN_sq _ hn, List.nil_append]

/-! ## The counts a rep's rounds squeeze -/

/-- Level `i`'s seven rounds' counts, its two intros at `dim` coins. -/
def lvC (sch : Flock.Schedule) (i dim : ℕ) : List ℕ :=
  [dim, 1, dim, 1, (sch.levels.getD i default).q, RecVStar.bitLength ((sch.levels.getD i default).q - 1), 1]

/-- Pass `i` of the levels' loop, of `r` passes: four folding rounds, then level `i + 1`'s or the last level's. -/
def passC (sch : Flock.Schedule) (r i : ℕ) : List ℕ :=
  [2, 2, 2, 2] ++
    if i + 1 < r then lvC sch (i + 1) ((sch.levels.getD (i + 1) default).cols + 1)
    else [(sch.levels.getD r default).q, RecVStar.bitLength ((sch.levels.getD r default).q - 1), 1]

/-- **`RecVStar.repRounds`' coin counts**, round by round. -/
def schedCnt (sch : Flock.Schedule) (m kLog claims : ℕ) : List ℕ :=
  6 :: (m - 13) :: 1 :: (List.replicate (m - 5) 1 ++ 1 :: (List.replicate (kLog - 6) 1 ++ 1 ::
    (List.replicate claims 7 ++ claims :: sch.n :: 1 :: (List.replicate sch.k0 2 ++
      (lvC sch 0 (sch.n - sch.k0 + 1) ++
        (List.range' 0 (sch.levels.size - 1)).flatMap (passC sch (sch.levels.size - 1)))))))

theorem closeNs_levelFr {sch : Flock.Schedule} (p : Flock.LigeritoProof)
    (hq : ∀ i < sch.levels.size, 2 ≤ (sch.levels.getD i default).q) {i : ℕ} (hi : i < sch.levels.size - 1) :
    closeNs (levelFr sch p i) = passC sch (sch.levels.size - 1) i := by
  unfold levelFr passC
  by_cases hc : i + 1 < sch.levels.size - 1
  · simp only [hc, ↓reduceIte, closeNs_append, closeNs_cons, closeNs_chFr, sqN_capFr,
      closeNs_introFr p (Nat.add_one_ne_zero _), closeNs_queryFr p (hq (i + 1) (by omega)), closeNs_kFr,
      sqN_bytes, sqN_sq _ Nat.one_ne_zero, closeNs_nil, lvC, List.nil_append, List.cons_append, List.append_nil]
    rfl
  · simp only [hc, ↓reduceIte, closeNs_append, closeNs_chFr, closeNs_ones,
      closeNs_queryFr p (hq (sch.levels.size - 1) (by omega)), closeNs_cons, sqN_bytes, sqN_sq _ Nat.one_ne_zero,
      closeNs_nil, List.nil_append, List.cons_append]
    rfl

theorem closeNs_loop {sch : Flock.Schedule} (p : Flock.LigeritoProof)
    (hq : ∀ i < sch.levels.size, 2 ≤ (sch.levels.getD i default).q) :
    closeNs ((List.range' 0 (sch.levels.size - 1)).flatMap (levelFr sch p)) =
      (List.range' 0 (sch.levels.size - 1)).flatMap (passC sch (sch.levels.size - 1)) := by
  rw [closeNs_flatMap]
  exact flatMap_range'_congr 0 _ fun i _ h => closeNs_levelFr p hq (by omega)

theorem closeNs_lig {sch : Flock.Schedule} (T : F128) (cap : Array ByteArray) (p : Flock.LigeritoProof)
    (hn : sch.n ≠ 0) (hq : ∀ i < sch.levels.size, 2 ≤ (sch.levels.getD i default).q) (hL : 1 ≤ sch.levels.size) :
    closeNs (ligFrames sch T cap p) = sch.n :: 1 :: (List.replicate sch.k0 2 ++
      (lvC sch 0 (sch.n - sch.k0 + 1) ++
        (List.range' 0 (sch.levels.size - 1)).flatMap (passC sch (sch.levels.size - 1)))) := by
  unfold ligFrames
  simp only [closeNs_append, closeNs_cons, closeNs_nil, closeNs_chFr, closeNs_kFr, closeNs_loop p hq,
    closeNs_introFr p (Nat.add_one_ne_zero _), closeNs_queryFr p (hq 0 (by omega)), sqN_label, sqN_one, sqN_bytes,
    sqN_capFr, sqN_sq _ hn, sqN_sq _ Nat.one_ne_zero, lvC, List.nil_append, List.cons_append, List.append_assoc,
    List.append_nil]

/-- **A rep's frames close their rounds at `schedCnt`'s counts.** -/
theorem closeNs_repFrames {st : Flock.Setup} {sch : Flock.Schedule} (d : ByteArray) (n : ℕ) (p : Flock.Proof)
    (T : F128) (hm : 14 ≤ st.m) (hn : n ≠ 0) (hsn : sch.n ≠ 0)
    (hq : ∀ i < sch.levels.size, 2 ≤ (sch.levels.getD i default).q) (hL : 1 ≤ sch.levels.size) :
    closeNs (repFrames st sch d n p T) = schedCnt sch st.m st.fold.kLog n := by
  have h5 : List.replicate (st.m - 5) 1 = List.replicate (st.m - 6) 1 ++ [1] := by
    rw [show st.m - 5 = st.m - 6 + 1 by omega, List.replicate_succ']
  unfold repFrames schedCnt
  simp only [closeNs_cons, sqN_dom, closeNs_append, closeNs_zc hm, closeNs_lc, closeNs_rs hn,
    closeNs_lig T p.cap p.ligerito hsn hq hL, h5, List.nil_append, List.append_assoc, List.cons_append,
    List.singleton_append]

/-! ## The schedule's side -/

/-- **`fast100`'s levels**: level 0 has `n − k₀` columns, each later level three fewer than the one before, and every
level queries at least twice. -/
theorem fast100_cols {m : ℕ} {sch : Flock.Schedule} (hs : fast100 m = some sch) :
    (sch.levels.getD 0 default).cols = sch.n - sch.k0 ∧
      (∀ i, i + 1 < sch.levels.size →
        (sch.levels.getD (i + 1) default).cols = (sch.levels.getD i default).cols - 3) ∧
      ∀ i < sch.levels.size, 2 ≤ (sch.levels.getD i default).q := by
  have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, decide ((s.levels.getD 0 default).cols = s.n - s.k0 ∧
      (∀ i < s.levels.size - 1, (s.levels.getD (i + 1) default).cols = (s.levels.getD i default).cols - 3) ∧
      ∀ i < s.levels.size, 2 ≤ (s.levels.getD i default).q) = true := by
    decide +kernel
  obtain ⟨-, h35, -⟩ := RecDeploy.fast100_sizes hs
  obtain ⟨h0, h1, h2⟩ := of_decide_eq_true (key ⟨m, by omega⟩ sch hs)
  exact ⟨h0, fun i hi => h1 i (by omega), h2⟩

theorem forIn_opt_inv {σ β : Type} (f : ℕ → σ → Option (ForInStep σ)) (π : σ → List β) (g : ℕ → List β)
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
      rw [forIn_opt_inv f π g I hf k (a + 1) u s hI' h, hu, List.range'_succ, List.flatMap_cons, List.append_assoc]

/-- **`RecVStar.repRounds`' counts are `schedCnt`'s.** -/
theorem repRounds_counts {m kLog claims : ℕ} {sch : Flock.Schedule} {rs : Array RecVStar.SchedRound}
    (hs : fast100 m = some sch) (h : RecVStar.repRounds m kLog claims = some rs) :
    rs.toList.map (·.1) = schedCnt sch m kLog claims := by
  obtain ⟨hc0, hcs, -⟩ := fast100_cols hs
  unfold RecVStar.repRounds at h
  simp only [hs, Std.Legacy.Range.forIn_eq_forIn_range'] at h
  simp only [Option.bind_eq_bind, Option.pure_def, Option.bind_some] at h
  simp only [Option.bind_eq_some_iff] at h
  obtain ⟨s, hl, he⟩ := h
  cases he
  have key := forIn_opt_inv _ (fun st : Array RecVStar.SchedRound × ℕ => st.1.toList.map (·.1))
    (passC sch (sch.levels.size - 1))
    (fun i st => i < sch.levels.size - 1 → st.2 = (sch.levels.getD i default).cols + 1)
    (fun i s t hI ht => by
      by_cases h4 : s.2 < 4
      · simp [h4] at ht
      · by_cases hi : i + 1 < sch.levels.size - 1
        · have e : s.2 - 4 + 1 = (sch.levels.getD (i + 1) default).cols + 1 := by
            have := hI (by omega)
            have := hcs i (by omega)
            omega
          simp [h4, hi] at ht
          refine ⟨_, ht.symm, ?_, fun _ => e⟩
          simp [passC, lvC, hi, rep3, e]
        · simp [h4, hi] at ht
          refine ⟨_, ht.symm, ?_, fun h' => absurd h' (by omega)⟩
          simp [passC, hi, rep3]) _ _ _ _ (fun _ => by simp [hc0]) hl
  refine key.trans ?_
  simp [schedCnt, lvC]

/-! ## The statement -/

/-- **A rep's rounds squeeze as its schedule's**: at a rep whose schedule is its frames' (`SchedOf`), with a claim, the
coins each chunk of its frames closes with are its schedule's round's, round by round. -/
theorem chunksOf_counts {c : RepCtx} (hs : SchedOf c) (hn : c.n ≠ 0) (p : Flock.Proof) (T : F128) :
    (chunksOf c p T).map sqCount = c.sched.map (·.1) := by
  obtain ⟨hf, -, rs, hrs, hsc⟩ := hs
  obtain ⟨h22, -, hn7, -, hL⟩ := RecDeploy.fast100_sizes hf
  obtain ⟨-, -, hq⟩ := fast100_cols hf
  unfold chunksOf
  rw [chunks_counts, closeNs_repFrames c.dom c.n p T (by omega) hn (by omega) hq (by rw [hL]; split_ifs <;> omega),
    hsc, repRounds_counts hf hrs]

end FlockDeploy
