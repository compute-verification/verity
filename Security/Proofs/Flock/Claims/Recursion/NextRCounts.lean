import Proofs.Flock.Claims.Recursion.NextRFits
import Proofs.Flock.VBridge.Family.Runs
import Proofs.Flock.VBridge.Algebra.RingSwitch
import Proofs.Flock.C3.Reps.LigNames

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
* **`T`'s slot** (`valOf_T`): the witness a rep's chunks' values start reads its `T` slot (`wi_T`, right after the
  prefix rounds and the claims' ring-switch blocks) at the `T` its frames were built at, given the proof's sizes.
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
    simp only [chFr, closeNs_append, closeNs_cons, closeNs_kFr, closeNs_chFr p (msg + 1) k,
      sqN_sq _ (show (2 : ℕ) ≠ 0 by decide), List.replicate_succ, List.cons_append, List.nil_append, List.append_nil]

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
    (fun i => by
      simp only [closeNs_cons, closeNs_nil, sqN_one, sqN_sq _ Nat.one_ne_zero, List.nil_append, List.append_nil]) 0 (m - 6)
  unfold zcFrames
  simp only [closeNs_append, hB, closeNs_cons, closeNs_nil, sqN_label, sqN_bytes, sqN_vec, sqN_one,
    sqN_sq _ (by decide : (6 : ℕ) ≠ 0), sqN_sq _ (show m - 13 ≠ 0 by omega), sqN_sq _ Nat.one_ne_zero,
    List.nil_append, List.append_nil, List.cons_append]

theorem closeNs_lc (k : ℕ) (p : Flock.LincheckProof) :
    closeNs (lcFrames k p) = 1 :: 1 :: (List.replicate (k - 6) 1 ++ [1]) := by
  have hB := closeNs_blocks (F := fun j => [Frame.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1]) (x := 1)
    (fun i => by
      simp only [closeNs_cons, closeNs_nil, sqN_one, sqN_sq _ Nat.one_ne_zero, List.nil_append, List.append_nil]) 0 (k - 6)
  unfold lcFrames
  simp only [closeNs_append, hB, closeNs_cons, closeNs_nil, sqN_label, sqN_vec, sqN_sq _ Nat.one_ne_zero,
    List.nil_append, List.cons_append]

theorem closeNs_rs {n : ℕ} (hn : n ≠ 0) (rs : Array (Array F128)) :
    closeNs (rsFrames n rs) = List.replicate n 7 ++ [n] := by
  have hB := closeNs_blocks
    (F := fun k => [Frame.label "flock-ring-switch-v0".toUTF8, .vec (rs[k]!).toList, .sq 2 7]) (x := 7)
    (fun i => by simp only [closeNs_cons, closeNs_nil, sqN_label, sqN_vec, sqN_sq _ (by decide : (7 : ℕ) ≠ 0),
      List.nil_append, List.append_nil]) 0 n
  unfold rsFrames
  simp only [closeNs_cons, closeNs_append, hB, closeNs_nil, sqN_label, sqN_sq _ hn, List.nil_append, List.append_nil]

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
          simp [passC, hi, rep3]) _ _ _ _ (fun _ => by rw [hc0]) hl
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
  have hL1 : 1 ≤ c.sch.levels.size := by rw [hL]; split_ifs <;> omega
  rw [chunks_counts, closeNs_repFrames (st := c.st) (sch := c.sch) c.dom c.n p T (by omega) hn (by omega) hq hL1,
    hsc, repRounds_counts hf hrs]

/-! ## `T`'s slot in w -/

open Flock.RecAlgebra (Key wi wNames ligRounds preRounds rounds_split idxOf?_of blocks blocks_spec valOf hRound)
open FlockC3.Reps (ligF1 ligF2 lig0 ligMid ligLast ligLevels ligRounds_eq)
open FlockSoundness.Discharge.Exec (chunks_flatten repFrames_chunks_snd)
open FlockSoundness.Discharge.DecodeZL (ProofSizes)

/-- Rounds whose first is `T`'s alone. -/
def TFirst (out : Array (Array Key)) : Prop := ∃ rest, out.toList = #[Key.T] :: rest

theorem TFirst.push {out : Array (Array Key)} (h : TFirst out) (r : Array Key) : TFirst (out.push r) := by
  obtain ⟨rest, h⟩ := h
  exact ⟨rest ++ [r], by rw [Array.toList_push, h, List.cons_append]⟩

theorem ligF1_first : ∀ (l : List ℕ) (mi : ℕ) (out : Array (Array Key)), TFirst out →
    TFirst (l.foldl ligF1 (mi, out)).2
  | [], _, _, h => h
  | _ :: l, mi, out, h => ligF1_first l (mi + 1) (out.push (Flock.RecAlgebra.msgNames mi)) (h.push _)

theorem ligF2_first : ∀ (l : List ℕ) (mi oi : ℕ) (out : Array (Array Key)), TFirst out →
    TFirst (l.foldl ligF2 (mi, oi, out)).2.2
  | [], _, _, _, h => h
  | _ :: l, mi, oi, out, h =>
    ligF2_first l (mi + 1) (oi + 1) (out.push (#[Key.ood oi] ++ Flock.RecAlgebra.msgNames mi)) (h.push _)

theorem lig0_first (sch : Flock.Schedule) : TFirst (lig0 sch).2.2 := by
  simp only [lig0]
  exact (ligF2_first _ _ _ _
    (ligF1_first (List.range' 0 (sch.k0 + 1)) 0 #[#[Key.T], #[Key.ood 0]] ⟨[#[Key.ood 0]], rfl⟩)).push _

theorem ligMid_first (s : ℕ × ℕ × Array (Array Key)) (h : TFirst s.2.2) : TFirst (ligMid s).2.2 := by
  simp only [ligMid]
  exact (ligF2_first _ _ _ _ (ligF1_first _ _ _ h)).push _

theorem ligLast_first (ext : ℕ) (s : ℕ × ℕ × Array (Array Key)) (h : TFirst s.2.2) :
    TFirst (ligLast ext s).2.2 := by
  simp only [ligLast]
  exact (ligF1_first _ _ _ h).push _

theorem ligLevels_first {R : ℕ} {G1 G2 : ℕ × ℕ × Array (Array Key) → ℕ × ℕ × Array (Array Key)}
    (h1 : ∀ s, TFirst s.2.2 → TFirst (G1 s).2.2) (h2 : ∀ s, TFirst s.2.2 → TFirst (G2 s).2.2) :
    ∀ (l : List ℕ) (init : ℕ × ℕ × Array (Array Key)), TFirst init.2.2 → TFirst (ligLevels R G1 G2 l init).2.2
  | [], _, h => by
    simp only [ligLevels, List.forIn_nil]
    exact h
  | a :: l, init, h => by
    unfold ligLevels
    rw [List.forIn_cons]
    by_cases hn : a + 1 < R
    · simp only [hn, ↓reduceIte, pure_bind]
      exact ligLevels_first h1 h2 l _ (h1 init h)
    · simp only [hn, ↓reduceIte, pure_bind]
      exact ligLevels_first h1 h2 l _ (h2 init h)

/-- **Ligerito's first round is `T`'s alone.** -/
theorem ligRounds_first (sch : Flock.Schedule) : TFirst (ligRounds sch) := by
  rw [ligRounds_eq]
  exact ligLevels_first ligMid_first (ligLast_first sch.ext) _ _ (lig0_first sch)

theorem T_not_pre (sh : Shape) : Key.T ∉ (preRounds sh).flatten := by
  cases h : sh.holo <;> simp [preRounds, hRound, h, Array.mem_flatten]

/-- **Where `T` is in w**: first after the ring switches. -/
theorem wi_T (sh : Shape) : wi sh .T = some ((preRounds sh).flatten.size + 128 * sh.claims) := by
  obtain ⟨rest, hr⟩ := ligRounds_first sh.sched
  have hL : (ligRounds sh.sched).flatten[0]? = some Key.T := by
    rw [← Array.getElem?_toList, Array.toList_flatten, hr]
    simp
  have hlen : 0 < (ligRounds sh.sched).flatten.size := (Array.getElem?_eq_some_iff.1 hL).1
  unfold wi wNames
  rw [rounds_split, Array.flatten_append, Array.flatten_append]
  obtain ⟨hsz, hget⟩ := blocks_spec Key.rs sh.claims
  have hB : (((Array.range sh.claims).map fun k => (Array.range 128).map (Key.rs k)).flatten) =
      blocks Key.rs sh.claims := rfl
  rw [hB]
  have s1 : ((preRounds sh).flatten ++ blocks Key.rs sh.claims).size =
      (preRounds sh).flatten.size + 128 * sh.claims := by rw [Array.size_append, hsz]
  have s2 : ((preRounds sh).flatten ++ blocks Key.rs sh.claims ++ (ligRounds sh.sched).flatten).size =
      (preRounds sh).flatten.size + 128 * sh.claims + (ligRounds sh.sched).flatten.size := by
    rw [Array.size_append, s1]
  apply idxOf?_of
  · rw [Array.getElem?_append_left (show (preRounds sh).flatten.size + 128 * sh.claims <
        ((preRounds sh).flatten ++ blocks Key.rs sh.claims ++ (ligRounds sh.sched).flatten).size by rw [s2]; omega),
      Array.getElem?_append_right (show ((preRounds sh).flatten ++ blocks Key.rs sh.claims).size ≤
        (preRounds sh).flatten.size + 128 * sh.claims by rw [s1]), s1, Nat.sub_self]
    exact hL
  · intro j hj he
    rw [Array.getElem?_append_left (show j <
        ((preRounds sh).flatten ++ blocks Key.rs sh.claims ++ (ligRounds sh.sched).flatten).size by rw [s2]; omega),
      Array.getElem?_append_left (show j < ((preRounds sh).flatten ++ blocks Key.rs sh.claims).size by
        rw [s1]; exact hj)] at he
    by_cases hjp : j < (preRounds sh).flatten.size
    · rw [Array.getElem?_append_left hjp] at he
      exact T_not_pre sh (Array.mem_of_getElem? he)
    · rw [Array.getElem?_append_right (by omega), hget _ (by omega)] at he
      simp at he

theorem pairs_flat_size (f g : ℕ → Key) : ∀ n, ((Array.range n).map fun i => #[f i, g i]).flatten.size = 2 * n
  | 0 => by simp [Array.range]
  | n + 1 => by
    have h1 : (((#[n] : Array ℕ).map fun i => #[f i, g i]).flatten).size = 2 := by simp
    rw [Array.range_succ, Array.map_append, Array.flatten_append, Array.size_append, pairs_flat_size f g n, h1]
    omega

theorem preRounds_size {sh : Shape} (hh : sh.holo = false) :
    (preRounds sh).flatten.size = 128 + 2 * (sh.m - 6) + 2 + 2 * (sh.kLog - 6) + 64 := by
  have h1 : (#[(Array.range 64).map Key.ab ++ (Array.range 64).map Key.c] : Array (Array Key)).flatten.size = 128 := by
    simp
  have h2 : (#[#[Key.fa, Key.fb]] : Array (Array Key)).flatten.size = 2 := by simp
  have h3 : (#[(Array.range 64).map Key.zp] : Array (Array Key)).flatten.size = 64 := by simp
  have h4 : (hRound sh).flatten.size = 0 := by simp [hRound, hh]
  simp only [preRounds, Array.flatten_append, Array.size_append, pairs_flat_size, h1, h2, h3, h4] <;> omega

/-! ### The frames' values -/

theorem vals_nil : vals ([] : List Frame) = [] := rfl
theorem vals_cons_one (v : F128) (fs : List Frame) : vals (.one v :: fs) = v :: vals fs := rfl
theorem vals_cons_vec (vs : List F128) (fs : List Frame) : vals (.vec vs :: fs) = vs ++ vals fs := rfl
theorem vals_cons_label (b : ByteArray) (fs : List Frame) : vals (.label b :: fs) = vals fs := rfl
theorem vals_cons_bytes (b : ByteArray) (fs : List Frame) : vals (.bytes b :: fs) = vals fs := rfl
theorem vals_cons_dom (b : ByteArray) (fs : List Frame) : vals (.dom b :: fs) = vals fs := rfl
theorem vals_cons_sq (k : UInt8) (n : ℕ) (fs : List Frame) : vals (.sq k n :: fs) = vals fs := rfl

theorem vals_pairs_length {x y : ℕ → F128} : ∀ a k,
    (vals ((List.range' a k).flatMap fun i => [Frame.one (x i), .one (y i), .sq 1 1])).length = 2 * k
  | _, 0 => rfl
  | a, k + 1 => by
    rw [List.range'_succ, List.flatMap_cons, vals_append, List.length_append, vals_pairs_length (a + 1) k]
    have e : vals [Frame.one (x a), .one (y a), .sq 1 1] = [x a, y a] := rfl
    rw [e, List.length_cons, List.length_cons, List.length_nil]
    omega

theorem vals_rsBlocks_length {rs : Array (Array F128)} : ∀ a k, (∀ j, a ≤ j → j < a + k → (rs[j]!).size = 128) →
    (vals ((List.range' a k).flatMap fun j =>
      [Frame.label "flock-ring-switch-v0".toUTF8, .vec (rs[j]!).toList, .sq 2 7])).length = 128 * k
  | _, 0, _ => rfl
  | a, k + 1, h => by
    rw [List.range'_succ, List.flatMap_cons, vals_append, List.length_append,
      vals_rsBlocks_length (a + 1) k fun j h1 h2 => h j (by omega) (by omega)]
    have e : vals [Frame.label "flock-ring-switch-v0".toUTF8, .vec (rs[a]!).toList, .sq 2 7] =
        (rs[a]!).toList ++ [] := rfl
    rw [e, List.append_nil, Array.length_toList, h a (Nat.le_refl a) (by omega)]
    omega

theorem vals_zc_length {m : ℕ} (digest cap : ByteArray) {p : Flock.ZerocheckProof} (h1 : p.round1ab.size = 64)
    (h2 : p.round1c.size = 64) : (vals (zcFrames m digest cap p)).length = 128 + 2 * (m - 6) + 2 := by
  unfold zcFrames
  simp only [vals_append, List.length_append, vals_pairs_length, vals_cons_one, vals_cons_vec, vals_cons_label,
    vals_cons_bytes, vals_cons_sq, vals_nil, List.length_cons, List.length_nil, Array.length_toList, h1, h2] <;> omega

theorem vals_lc_length (k : ℕ) {p : Flock.LincheckProof} (h : p.zPartial.size = 64) :
    (vals (lcFrames k p)).length = 2 * (k - 6) + 64 := by
  unfold lcFrames
  simp only [vals_append, List.length_append, vals_pairs_length, vals_cons_vec, vals_cons_label, vals_cons_sq,
    vals_nil, List.length_nil, Array.length_toList, h]
  omega

theorem vals_rs_length {n : ℕ} {rs : Array (Array F128)} (hs : rs.size = n) (he : ∀ v ∈ rs, v.size = 128) :
    (vals (rsFrames n rs)).length = 128 * n := by
  unfold rsFrames
  rw [vals_cons_label, vals_append, List.length_append,
    vals_rsBlocks_length (rs := rs) 0 n fun j _ hj => he _ (by
      rw [getElem!_pos rs j (by omega)]; exact Array.getElem_mem _)]
  rfl

theorem vals_repFrames (st : Flock.Setup) (sch : Flock.Schedule) (d : ByteArray) (n : ℕ) (p : Flock.Proof)
    (T : F128) : ∃ ys, vals (repFrames st sch d n p T) =
      vals (zcFrames st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck) ++
        vals (lcFrames st.fold.kLog p.lincheck) ++ vals (rsFrames n p.ringSwitches) ++ T :: ys := by
  have hY : ∃ ys, vals (ligFrames sch T p.cap p.ligerito) = T :: ys := ⟨_, by unfold ligFrames; rw [vals_append]; rfl⟩
  obtain ⟨ys, hY⟩ := hY
  refine ⟨ys, ?_⟩
  unfold repFrames
  rw [vals_cons_dom, vals_append, vals_append, vals_append, hY]

theorem getElem!_mid (xs ys : List F128) (T : F128) (acc : Array F128) : ((xs ++ T :: ys).toArray ++ acc)[xs.length]! = T := by
  rw [getElem!_def, Array.getElem?_append_left (show xs.length < (xs ++ T :: ys).toArray.size by simp),
    List.getElem?_toArray,
    List.getElem?_append_right (Nat.le_refl _), Nat.sub_self, List.getElem?_cons_zero]

/-- **w holds `T` at `T`'s slot**: at a proof of the lengths its shape fixes (`ProofSizes`), at a setup that folds at its
own `k_log`, the value at `wi sh .T` of the values a rep's frames absorb, round by round, is `T`. -/
theorem valOf_T {st : Flock.Setup} {sch : Flock.Schedule} {e : ℕ} (d : ByteArray) (p : Flock.Proof) (T : F128)
    (acc comb : Array F128) (cons : ℕ → ℕ → F128)
    (hs : ProofSizes (FlockSoundness.Refine.shapeOf st sch e) p) (hk : st.fold.kLog = st.kLog) :
    valOf (FlockSoundness.Refine.shapeOf st sch e)
      (((chunks (repFrames st sch d (2 + e) p T)).1.flatMap vals).toArray ++ acc) comb cons .T = T := by
  have hw : (chunks (repFrames st sch d (2 + e) p T)).1.flatMap vals = vals (repFrames st sch d (2 + e) p T) := by
    have h := chunks_flatten (repFrames st sch d (2 + e) p T)
    rw [repFrames_chunks_snd, List.append_nil] at h
    rw [List.flatMap_def, ← vals_flatten, h]
  obtain ⟨ys, hY⟩ := vals_repFrames st sch d (2 + e) p T
  have hz := vals_zc_length (m := st.m) st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) hs.r1ab hs.r1c
  have hl := vals_lc_length st.fold.kLog hs.zPartial
  have hr := vals_rs_length (n := 2 + e) hs.rs hs.rsEach
  have hp := preRounds_size (sh := FlockSoundness.Refine.shapeOf st sch e) rfl
  simp only [valOf, wi_T]
  rw [hw, hY]
  have hi : (preRounds (FlockSoundness.Refine.shapeOf st sch e)).flatten.size +
      128 * (FlockSoundness.Refine.shapeOf st sch e).claims =
      (vals (zcFrames st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck) ++
        vals (lcFrames st.fold.kLog p.lincheck) ++ vals (rsFrames (2 + e) p.ringSwitches)).length := by
    rw [hp, List.length_append, List.length_append, hz, hl, hr, hk]
    simp only [FlockSoundness.Refine.shapeOf]
    omega
  rw [hi]
  exact getElem!_mid _ ys T acc

end FlockDeploy
