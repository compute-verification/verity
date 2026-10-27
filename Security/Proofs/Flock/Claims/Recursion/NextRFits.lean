import Proofs.Flock.Claims.Recursion.NextRHonest
import Proofs.Flock.Claims.Recursion.SchedCount
import Proofs.Flock.Soundness.Discharge.DecodeZL.LIG
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.C3.Compose.HoloStreamSizes

/-!
# The honest rows fit: `RepFits`

An honest rep's proof fits the rows the honest prover sends (`repFits`), so one honest rep reads back (`repReadsBack`)
and an honest prover's rounds read back at `nextR` (`readsBack`). The rows fit where the frames' chunks and the
schedule's rounds agree round by round, on which rounds absorb element values and which tops each commits:

* **The frames' side** (`roundPat`): which of a frame list's chunks carry element values (a `one` or `vec` frame), and
  whether its pending frames do. It composes over `++` (`roundPat_append`, as `chunks_append`), and each part of a rep's
  frames has a closed pattern: the zerocheck's, the lincheck's, the ring switch's, and Ligerito's, level by level
  (`roundPat_zc`, `roundPat_lc`, `roundPat_rs`, `roundPat_levelFr`, `roundPat_lig`), so a rep's frames have
  `schedPat`'s flags (`roundPat_repFrames`). This reads only the frames' constructors: any values, at a `fast100`
  schedule.
* **The schedule's side** (`schedPat`): `RecVStar.repRounds`' absorb flags and tops, round by round
  (`repRounds_flagTop`), whose tops are the levels' caps after level 0, in order (`schedPat_tops`): the recursive caps'
  node counts at a proof of the rep's shape (`tops_caps`).
* **The rows** (`honestRows_fit`): where chunks absorb as rounds do and the rounds' tops are the caps' node counts, the
  honest rows are one per round, their tops cut back into the caps, and each holds its chunk's values (`RowsAt`): a
  round that absorbs nothing has a chunk with no values.

`RepReadsBack`'s hypotheses hold together (`honest₀`): at `m = 22`, `k_log = 6`, two claims, a proof with every length
its shape fixes and zero nonces (`sizedProof`), so one honest rep reads back on a run that exists (`readRep₀`).
`ReadsBack`'s `SchedOf` holds at every rep whose table has a `fast100` schedule and extra claims (`schedOf_repCtx`).
-/

namespace FlockDeploy

open Flock
open FlockSoundness.Refine (Frame colBytes)
open FlockSoundness.Discharge.Exec (chunks closes glue chunks_append repFrames zcFrames lcFrames rsFrames ligFrames
  levelFr introFr queryFr chFr kFr capFr)

/-! ## Which rounds absorb: the frames' side -/

/-- A frame that carries element values. -/
def carriesV : Frame → Bool
  | .one _ => true
  | .vec _ => true
  | .dom _ => false
  | .label _ => false
  | .bytes _ => false
  | .sq _ _ => false

/-- Whether frames carry element values. -/
def anyV (fs : List Frame) : Bool := fs.any carriesV

/-- Chunks' pattern: which carry element values, and whether the frames after the last do. -/
def patOf (x : List (List Frame) × List Frame) : List Bool × Bool := (x.1.map anyV, anyV x.2)

/-- `glue` on patterns. -/
def glueP (a b : List Bool × Bool) : List Bool × Bool :=
  match b.1 with
  | [] => (a.1, a.2 || b.2)
  | c :: cs => (a.1 ++ (a.2 || c) :: cs, b.2)

/-- Frames' round pattern: which of their chunks carry element values, and whether their pending frames do. -/
def roundPat (F : List Frame) : List Bool × Bool := patOf (chunks F)

theorem patOf_glue (a b : List (List Frame) × List Frame) : patOf (glue a b) = glueP (patOf a) (patOf b) := by
  rcases b with ⟨_ | ⟨c, cs⟩, t⟩ <;> simp [glue, patOf, glueP, anyV, List.any_append]

theorem roundPat_append (F G : List Frame) : roundPat (F ++ G) = glueP (roundPat F) (roundPat G) := by
  unfold roundPat
  rw [chunks_append, patOf_glue]

theorem roundPat_nil : roundPat [] = ([], false) := rfl

theorem glueP_closed (la lb : List Bool) (pb : Bool) : glueP (la, false) (lb, pb) = (la ++ lb, pb) := by
  cases lb <;> simp [glueP]

theorem glueP_cons (a : List Bool) (x c : Bool) (cs : List Bool) (y : Bool) :
    glueP (a, x) (c :: cs, y) = (a ++ (x || c) :: cs, y) := rfl

theorem glueP_nil (a : List Bool) (x y : Bool) : glueP (a, x) ([], y) = (a, x || y) := rfl

theorem glueP_false_nil (x : List Bool × Bool) : glueP ([], false) x = x := by
  rcases x with ⟨_ | _, _⟩ <;> rfl

theorem roundPat_open {f : Frame} (hf : closes f = false) (F : List Frame) :
    roundPat (f :: F) = glueP ([], carriesV f) (roundPat F) := by
  unfold roundPat
  simp only [chunks, hf, Bool.false_eq_true, ↓reduceIte]
  generalize chunks F = x
  rcases x with ⟨_ | ⟨c, cs⟩, t⟩ <;> simp [patOf, glueP, anyV]

theorem roundPat_sq (k : UInt8) {n : ℕ} (hn : n ≠ 0) (F : List Frame) :
    roundPat (.sq k n :: F) = (false :: (roundPat F).1, (roundPat F).2) := by
  have hc : closes (.sq k n) = true := by simp [closes, hn]
  unfold roundPat
  simp only [chunks, hc, ↓reduceIte]
  rfl

theorem roundPat_dom (b : ByteArray) (F : List Frame) : roundPat (.dom b :: F) = roundPat F :=
  (roundPat_open (f := .dom b) rfl F).trans (glueP_false_nil _)

theorem roundPat_label (b : ByteArray) (F : List Frame) : roundPat (.label b :: F) = roundPat F :=
  (roundPat_open (f := .label b) rfl F).trans (glueP_false_nil _)

theorem roundPat_bytes (b : ByteArray) (F : List Frame) : roundPat (.bytes b :: F) = roundPat F :=
  (roundPat_open (f := .bytes b) rfl F).trans (glueP_false_nil _)

theorem roundPat_capFr (cap : Array ByteArray) (F : List Frame) : roundPat (capFr cap :: F) = roundPat F :=
  roundPat_bytes _ F

theorem roundPat_one (v : F128) (F : List Frame) : roundPat (.one v :: F) = glueP ([], true) (roundPat F) :=
  roundPat_open rfl F

theorem roundPat_vec (vs : List F128) (F : List Frame) : roundPat (.vec vs :: F) = glueP ([], true) (roundPat F) :=
  roundPat_open rfl F

theorem roundPat_kFr (m : F256 × F256) : roundPat (kFr m) = ([], true) := rfl

theorem roundPat_ones : ∀ l : List F128, roundPat (l.map Frame.one) = ([], !l.isEmpty)
  | [] => rfl
  | v :: l => by rw [List.map_cons, roundPat_one, roundPat_ones l]; rfl

/-- **Blocks that each close one round carrying values.** -/
theorem roundPat_blocks {F : ℕ → List Frame} (hF : ∀ i, roundPat (F i) = ([true], false)) :
    ∀ a k, roundPat ((List.range' a k).flatMap F) = (List.replicate k true, false)
  | _, 0 => rfl
  | a, k + 1 => by
    rw [List.range'_succ, List.flatMap_cons, roundPat_append, hF, roundPat_blocks hF (a + 1) k, glueP_closed]
    rfl

/-- **Blocks that each close their rounds.** -/
theorem roundPat_flatMap {F : ℕ → List Frame} {G : ℕ → List Bool} :
    ∀ a k, (∀ i, a ≤ i → i < a + k → roundPat (F i) = (G i, false)) →
      roundPat ((List.range' a k).flatMap F) = ((List.range' a k).flatMap G, false)
  | _, 0, _ => rfl
  | a, k + 1, h => by
    rw [List.range'_succ, List.flatMap_cons, List.flatMap_cons, roundPat_append, h a (Nat.le_refl a) (by omega),
      roundPat_flatMap (a + 1) k fun i h1 h2 => h i (by omega) (by omega), glueP_closed]

theorem roundPat_chFr (p : Flock.LigeritoProof) :
    ∀ k msg, roundPat (chFr p msg (k + 1)) = (false :: List.replicate k true, true)
  | 0, msg => by
    simp only [chFr, List.append_nil, roundPat_sq 2 (by decide : (2 : ℕ) ≠ 0), roundPat_kFr, List.replicate_zero]
  | k + 1, msg => by
    rw [chFr, roundPat_append, roundPat_chFr p k (msg + 1)]
    simp only [roundPat_sq 2 (by decide : (2 : ℕ) ≠ 0), roundPat_kFr, glueP_cons, Bool.true_or, List.nil_append,
      List.cons_append, List.replicate_succ]

theorem roundPat_chFr4 (p : Flock.LigeritoProof) (msg : ℕ) :
    roundPat (chFr p msg 4) = ([false, true, true, true], true) :=
  roundPat_chFr p 3 msg

theorem roundPat_introFr (p : Flock.LigeritoProof) {dim : ℕ} (hd : dim ≠ 0) (ood msg cbn : ℕ) :
    roundPat (introFr p dim ood msg cbn) = ([false, true], false) := by
  unfold introFr
  simp only [roundPat_append, roundPat_sq _ hd, roundPat_one, roundPat_kFr, roundPat_bytes, roundPat_nil,
    roundPat_sq _ Nat.one_ne_zero, glueP_cons, Bool.or_self, Bool.true_or, List.nil_append, List.cons_append]

theorem roundPat_queryFr (p : Flock.LigeritoProof) {lv : Flock.Level} (hq : lv.q ≠ 0) (qn cn : ℕ) :
    roundPat (queryFr p lv qn cn) = ([false, false], false) := by
  unfold queryFr
  simp only [roundPat_bytes, roundPat_sq _ hq, roundPat_sq _ (Nat.add_one_ne_zero _), roundPat_nil]

theorem roundPat_zc {m : ℕ} (hm : 14 ≤ m) (digest cap : ByteArray) (p : Flock.ZerocheckProof) :
    roundPat (zcFrames m digest cap p) = (false :: false :: true :: List.replicate (m - 6) true, true) := by
  have hB := roundPat_blocks (F := fun i => [Frame.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1])
    (fun i => by simp only [roundPat_one, roundPat_sq _ Nat.one_ne_zero, roundPat_nil, glueP_cons, Bool.true_or,
      List.nil_append]) 0 (m - 6)
  have h13 : m - 13 ≠ 0 := by omega
  unfold zcFrames
  simp only [roundPat_append, hB, roundPat_one, roundPat_nil, roundPat_label, roundPat_bytes,
    roundPat_sq _ (by decide : (6 : ℕ) ≠ 0), roundPat_sq _ h13, roundPat_vec, roundPat_sq _ Nat.one_ne_zero,
    glueP_nil, glueP_cons, Bool.or_self, Bool.or_false, Bool.false_or, List.nil_append, List.cons_append]

theorem roundPat_lc (k : ℕ) (p : Flock.LincheckProof) :
    roundPat (lcFrames k p) = (false :: false :: (List.replicate (k - 6) true ++ [true]), false) := by
  have hB := roundPat_blocks (F := fun j => [Frame.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1])
    (fun i => by simp only [roundPat_one, roundPat_sq _ Nat.one_ne_zero, roundPat_nil, glueP_cons, Bool.true_or,
      List.nil_append]) 0 (k - 6)
  unfold lcFrames
  simp only [roundPat_append, hB, roundPat_vec, roundPat_nil, roundPat_label, roundPat_sq _ Nat.one_ne_zero,
    glueP_cons, Bool.true_or, Bool.false_or, List.nil_append, List.cons_append]

theorem roundPat_rs {n : ℕ} (hn : n ≠ 0) (rs : Array (Array F128)) :
    roundPat (rsFrames n rs) = (List.replicate n true ++ [false], false) := by
  have hB := roundPat_blocks
    (F := fun k => [Frame.label "flock-ring-switch-v0".toUTF8, .vec (rs[k]!).toList, .sq 2 7])
    (fun i => by simp only [roundPat_label, roundPat_vec, roundPat_sq _ (by decide : (7 : ℕ) ≠ 0), roundPat_nil,
      glueP_cons, Bool.true_or, List.nil_append]) 0 n
  unfold rsFrames
  simp only [roundPat_label, roundPat_append, hB, roundPat_sq _ hn, roundPat_nil, glueP_closed]

/-! ## Which rounds absorb and what they commit: the schedule's side -/

/-- A round's absorb flag and its tops' node counts. -/
def flagTop (s : RecVStar.SchedRound) : Bool × List ℕ := (s.2.2.2, s.2.1.toList)

/-- Level `l`'s top, as its node count. -/
def topsAt (sch : Flock.Schedule) (l : ℕ) : List ℕ := [2 ^ (sch.levels.getD l default).cap]

/-- Level `i`'s seven rounds: the first commits level `i + 1`'s top. -/
def lvT (sch : Flock.Schedule) (i : ℕ) : List (Bool × List ℕ) :=
  [(true, topsAt sch (i + 1)), (true, []), (false, []), (true, []), (false, []), (false, []), (true, [])]

/-- Pass `i` of the levels' loop, of `r` passes: four folding rounds, then level `i + 1`'s or the last level's. -/
def passT (sch : Flock.Schedule) (r i : ℕ) : List (Bool × List ℕ) :=
  [(false, []), (true, []), (true, []), (true, [])] ++
    if i + 1 < r then lvT sch (i + 1) else [(true, []), (false, []), (false, [])]

/-- **`RecVStar.repRounds`' absorb flags and tops**, round by round. -/
def schedPat (sch : Flock.Schedule) (m kLog claims : ℕ) : List (Bool × List ℕ) :=
  [(false, []), (false, []), (true, [])] ++ List.replicate (m - 5) (true, []) ++ [(false, [])] ++
    List.replicate (kLog - 6) (true, []) ++ [(true, [])] ++ List.replicate claims (true, []) ++ [(false, [])] ++
    [(true, []), (true, [])] ++ List.replicate sch.k0 (true, []) ++ lvT sch 0 ++
    (List.range' 0 (sch.levels.size - 1)).flatMap (passT sch (sch.levels.size - 1))

theorem roundPat_levelFr {sch : Flock.Schedule} (p : Flock.LigeritoProof)
    (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) {i : ℕ} (hi : i < sch.levels.size - 1) :
    roundPat (levelFr sch p i) = ((passT sch (sch.levels.size - 1) i).map Prod.fst, false) := by
  unfold levelFr passT
  by_cases hc : i + 1 < sch.levels.size - 1
  · simp only [hc, ↓reduceIte, roundPat_append, roundPat_chFr4, roundPat_capFr,
      roundPat_introFr p (Nat.add_one_ne_zero _), roundPat_queryFr p (hq (i + 1) (by omega)), roundPat_kFr,
      roundPat_bytes, roundPat_sq _ Nat.one_ne_zero, roundPat_nil, glueP_closed, glueP_cons, Bool.true_or,
      Bool.false_or, List.nil_append, List.cons_append, lvT, List.map_cons, List.map_nil]
  · simp only [hc, ↓reduceIte, roundPat_append, roundPat_chFr4, roundPat_ones,
      roundPat_queryFr p (hq (sch.levels.size - 1) (by omega)), roundPat_bytes, roundPat_sq _ Nat.one_ne_zero,
      roundPat_nil, glueP_cons, Bool.true_or, Bool.or_false, List.nil_append, List.cons_append, List.map_cons,
      List.map_nil]

theorem roundPat_lig {sch : Flock.Schedule} (T : F128) (cap : Array ByteArray) (p : Flock.LigeritoProof)
    (hk : 0 < sch.k0) (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) (hn : sch.n ≠ 0)
    (hL : 1 ≤ sch.levels.size) :
    roundPat (ligFrames sch T cap p) = (true :: true :: (List.replicate sch.k0 true ++ ((lvT sch 0).map Prod.fst ++
      (List.range' 0 (sch.levels.size - 1)).flatMap fun i => (passT sch (sch.levels.size - 1) i).map Prod.fst)),
      false) := by
  have hloop := roundPat_flatMap (F := levelFr sch p)
    (G := fun i => (passT sch (sch.levels.size - 1) i).map Prod.fst) 0 (sch.levels.size - 1)
    (fun i _ h => roundPat_levelFr p hq (by omega))
  obtain ⟨j, hj⟩ : ∃ j, sch.k0 = j + 1 := ⟨sch.k0 - 1, by omega⟩
  unfold ligFrames
  rw [hj]
  simp only [roundPat_append, hloop, roundPat_label, roundPat_one, roundPat_capFr, roundPat_sq _ hn, roundPat_bytes,
    roundPat_sq _ Nat.one_ne_zero, roundPat_kFr, roundPat_chFr, roundPat_introFr p (Nat.add_one_ne_zero _),
    roundPat_queryFr p (hq 0 (by omega)), glueP_cons, Bool.or_false, Bool.false_or, List.nil_append,
    List.cons_append, List.append_assoc, List.replicate_succ, lvT, List.map_cons, List.map_nil]

theorem roundPat_repFrames {st : Flock.Setup} {sch : Flock.Schedule} (d : ByteArray) (e : ℕ) (p : Flock.Proof)
    (T : F128) (hm : 14 ≤ st.m) (hk : 0 < sch.k0) (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0)
    (hn : sch.n ≠ 0) (hL : 1 ≤ sch.levels.size) :
    roundPat (repFrames st sch d (2 + e) p T) = ((schedPat sch st.m st.fold.kLog (2 + e)).map Prod.fst, false) := by
  have h5 : List.replicate (st.m - 5) true = List.replicate (st.m - 6) true ++ [true] := by
    rw [show st.m - 5 = st.m - 6 + 1 by omega, List.replicate_succ']
  have h2 : 2 + e ≠ 0 := by omega
  unfold repFrames schedPat
  simp only [roundPat_dom, roundPat_append, roundPat_zc hm, roundPat_lc, roundPat_rs h2,
    roundPat_lig T p.cap p.ligerito hk hq hn hL, glueP_cons, glueP_closed, Bool.true_or, Bool.false_or,
    List.map_append, List.map_cons, List.map_replicate, List.map_flatMap, h5, List.nil_append,
    List.cons_append, List.append_assoc]

theorem forIn_opt_map {σ β : Type} (f : ℕ → σ → Option (ForInStep σ)) (π : σ → List β) (g : ℕ → List β)
    (hf : ∀ i s t, f i s = some t → ∃ u, t = .yield u ∧ π u = π s ++ g i) :
    ∀ (l : List ℕ) (s0 s : σ), forIn l s0 f = some s → π s = π s0 ++ l.flatMap g := by
  intro l
  induction l with
  | nil => intro s0 s h; simp at h; subst h; simp
  | cons a l ih =>
    intro s0 s h
    simp only [List.forIn_cons] at h
    cases hfa : f a s0 with
    | none => rw [hfa] at h; cases h
    | some t =>
      rw [hfa] at h
      obtain ⟨u, rfl, hu⟩ := hf a s0 t hfa
      rw [ih u s h, hu, List.flatMap_cons, List.append_assoc]

theorem rep3 {α : Type} (a : α) : List.replicate 3 a = [a, a, a] := rfl

/-- **`RecVStar.repRounds`' flags and tops are `schedPat`'s.** -/
theorem repRounds_flagTop {m kLog claims : ℕ} {sch : Flock.Schedule} {rs : Array RecVStar.SchedRound}
    (hs : fast100 m = some sch) (h : RecVStar.repRounds m kLog claims = some rs) :
    rs.toList.map flagTop = schedPat sch m kLog claims := by
  unfold RecVStar.repRounds at h
  simp only [hs, Std.Legacy.Range.forIn_eq_forIn_range'] at h
  simp only [Option.bind_eq_bind, Option.pure_def, Option.bind_some] at h
  simp only [Option.bind_eq_some_iff] at h
  obtain ⟨s, hl, he⟩ := h
  cases he
  have key := forIn_opt_map _ (fun st : Array RecVStar.SchedRound × ℕ => st.1.toList.map flagTop)
    (passT sch (sch.levels.size - 1)) (fun i s t ht => by
      by_cases h4 : s.2 < 4
      · simp [h4] at ht
      · by_cases hi : i + 1 < sch.levels.size - 1
        · simp [h4, hi] at ht
          refine ⟨_, ht.symm, ?_⟩
          simp [flagTop, passT, lvT, topsAt, hi, rep3]
        · simp [h4, hi] at ht
          refine ⟨_, ht.symm, ?_⟩
          simp [flagTop, passT, hi, rep3]) _ _ _ hl
  refine key.trans ?_
  simp [schedPat, flagTop, lvT, topsAt]

theorem tops_loop (sch : Flock.Schedule) (r : ℕ) : ∀ (k a : ℕ), a + k + 1 = r →
    (((List.range' a (k + 1)).flatMap (passT sch r)).map Prod.snd).flatten =
      (List.range' a k).map fun i => 2 ^ (sch.levels.getD (i + 1 + 1) default).cap
  | 0, a, h => by
    have hc : ¬ a + 1 < r := by omega
    simp [passT, hc, List.range'_succ]
  | k + 1, a, h => by
    have hc : a + 1 < r := by omega
    have e1 : List.range' a (k + 1 + 1) = a :: List.range' (a + 1) (k + 1) := List.range'_succ
    have e2 : List.range' a (k + 1) = a :: List.range' (a + 1) k := List.range'_succ
    rw [e1, e2, List.flatMap_cons, List.map_append, List.flatten_append, tops_loop sch r k (a + 1) (by omega),
      List.map_cons]
    simp [passT, lvT, topsAt, hc]

/-- **The tops `RecVStar.repRounds` commits**: level 1's, then each later level's but the last's. -/
theorem schedPat_tops {sch : Flock.Schedule} {k : ℕ} (hk : sch.levels.size - 1 = k + 1) (m kLog claims : ℕ) :
    ((schedPat sch m kLog claims).map Prod.snd).flatten =
      2 ^ (sch.levels.getD 1 default).cap ::
        (List.range' 0 k).map fun i => 2 ^ (sch.levels.getD (i + 1 + 1) default).cap := by
  unfold schedPat
  rw [hk, List.map_append, List.flatten_append, tops_loop sch (k + 1) k 0 (by omega)]
  simp [lvT, topsAt]

theorem tops_caps {sch : Flock.Schedule} {rc : Array (Array ByteArray)} {k : ℕ} (hk : sch.levels.size - 1 = k + 1)
    (hsz : rc.size = sch.levels.size - 1)
    (he : ∀ (i : ℕ) (hi : i < rc.size), rc[i].size = 2 ^ (sch.levels.getD (i + 1) default).cap) :
    2 ^ (sch.levels.getD 1 default).cap ::
        (List.range' 0 k).map (fun i => 2 ^ (sch.levels.getD (i + 1 + 1) default).cap) = rc.toList.map Array.size := by
  refine List.ext_getElem (by simp [hsz, hk]) fun i h₁ h₂ => ?_
  have hi : i < rc.size := by simpa using h₂
  simp only [List.getElem_map, Array.getElem_toList]
  rw [he i hi]
  cases i with
  | zero => rfl
  | succ j => simp

/-! ## The honest rows fit -/

theorem vals_nil_of_anyV : ∀ {fs : List Frame}, anyV fs = false → vals fs = []
  | [], _ => rfl
  | .dom _ :: fs, h => vals_nil_of_anyV (fs := fs) h
  | .label _ :: fs, h => vals_nil_of_anyV (fs := fs) h
  | .one _ :: _, h => by simp [anyV, carriesV] at h
  | .vec _ :: _, h => by simp [anyV, carriesV] at h
  | .bytes _ :: fs, h => vals_nil_of_anyV (fs := fs) h
  | .sq _ _ :: fs, h => vals_nil_of_anyV (fs := fs) h

theorem cutLens_caps (f : ByteArray → ByteArray) : ∀ (caps : List (Array ByteArray)) (x : List ByteArray),
    cutLens (caps.map Array.size) (caps.flatMap (fun a => a.toList.map f) ++ x) = caps.map fun a => a.toList.map f
  | [], _ => rfl
  | c :: cs, x => by
    have hl : (c.toList.map f).length = c.size := by rw [List.length_map, Array.length_toList]
    rw [List.map_cons, List.flatMap_cons, List.append_assoc, cutLens, List.take_left' hl, List.drop_left' hl,
      cutLens_caps f cs x, List.map_cons]

theorem map_nodeOf_caps (l : List (Array ByteArray)) :
    (l.map fun a => a.toList.map nodeRow).map (List.map nodeOf) = l.map Array.toList := by
  rw [List.map_map]
  exact List.map_congr_left fun a _ => map_nodeOf_nodeRow a.toList

/-- **Rows the honest prover sends fit**: where the frames' chunks absorb as the schedule's rounds do, and the rounds'
tops are the caps' node counts, the honest rows are one per round, cut at the tops into the caps, and hold each chunk's
values. -/
theorem honestRows_fit : ∀ (sched : List RecVStar.SchedRound) (chs : List (List Frame)) (caps : List (Array ByteArray)),
    chs.map anyV = sched.map (·.2.2.2) → (sched.map (·.2.1.toList)).flatten = caps.map Array.size →
    (honestRows sched chs caps).length = sched.length ∧ chs.length = sched.length ∧
      topGroups sched (honestRows sched chs caps) = caps.map Array.toList ∧
      ∀ j < sched.length, RowsAt (sched.getD j default) ((honestRows sched chs caps).getD j []) (vals (chs.getD j []))
  | [], chs, caps, h1, h2 => by
    obtain rfl : chs = [] := List.map_eq_nil_iff.1 h1
    obtain rfl : caps = [] := List.map_eq_nil_iff.1 h2.symm
    exact ⟨rfl, rfl, rfl, fun j hj => absurd hj (Nat.not_lt_zero j)⟩
  | _ :: _, [], _, h1, _ => by simp at h1
  | ⟨n, tops, c, b⟩ :: ss, ch :: chs, caps, h1, h2 => by
    simp only [List.map_cons, List.cons.injEq] at h1
    simp only [List.map_cons, List.flatten_cons] at h2
    have hk : tops.toList.length = tops.size := Array.length_toList
    have ht : (caps.take tops.size).map Array.size = tops.toList := by
      rw [List.map_take, ← h2, List.take_left' hk]
    have hd : (ss.map (·.2.1.toList)).flatten = (caps.drop tops.size).map Array.size := by
      rw [List.map_drop, ← h2, List.drop_left' hk]
    obtain ⟨l1, l2, l3, l4⟩ := honestRows_fit ss chs (caps.drop tops.size) h1.2 hd
    have hsum : (((caps.take tops.size).flatMap Array.toList).map nodeRow).length = tops.toList.sum := by
      rw [List.length_map, ← ht, List.length_flatMap]
      rfl
    simp only [honestRows]
    refine ⟨by simp [l1], by simp [l2], ?_, ?_⟩
    · show (cutLens tops.toList _).map (List.map nodeOf) ++ topGroups ss _ = _
      rw [l3, ← ht, List.map_flatMap, cutLens_caps, map_nodeOf_caps, ← List.map_append, List.take_append_drop]
    · intro j hj
      cases j with
      | zero =>
        refine rowsAt_padRow ?_ (List.drop_left' hsum) fun hb => vals_nil_of_anyV (h1.1.trans hb)
        simp only [List.getD_cons_zero, List.length_append, hsum]
        cases b <;> rfl
      | succ j => exact l4 j (by simpa using hj)

theorem map_flag (l : List RecVStar.SchedRound) : l.map (·.2.2.2) = (l.map flagTop).map Prod.fst := by
  rw [List.map_map]
  rfl

theorem map_tops (l : List RecVStar.SchedRound) : l.map (·.2.1.toList) = (l.map flagTop).map Prod.snd := by
  rw [List.map_map]
  rfl

/-- A `fast100` schedule's levels each query. -/
theorem q_ne_zero {m : ℕ} {sch : Flock.Schedule} (hs : fast100 m = some sch) :
    ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0 := fun i hi => by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
  exact Nat.pos_iff_ne_zero.1
    ((FlockSoundness.Discharge.DecodeZL.fast100_small m sch hs).2.2 _ (Array.getElem_mem hi)).2.1

/-! ## The statements -/

/-- **The honest proof fits the honest rows**: its frames' chunks absorb as the rep's schedule's rounds do
(`roundPat_repFrames`, `repRounds_flagTop`), and the rounds' tops are its recursive caps' node counts (`schedPat_tops`,
`tops_caps`), so the honest rows fit (`honestRows_fit`). -/
theorem repFits : RepFits := by
  intro c p T hsch hp hcap
  obtain ⟨hs, -, rs, hrs, hsc⟩ := hsch
  obtain ⟨e, hn, hps⟩ := hp.sizes
  obtain ⟨h22, -, hn7, hk0, -⟩ := RecDeploy.fast100_sizes hs
  have hk : 0 < c.sch.k0 := by rw [hk0]; split_ifs <;> omega
  have hL := hp.levels
  obtain ⟨k, hk1⟩ : ∃ k, c.sch.levels.size - 1 = k + 1 := ⟨c.sch.levels.size - 2, by omega⟩
  have hft := repRounds_flagTop hs hrs
  rw [hn] at hft
  have hch : chunksOf c p T = (chunks (repFrames c.st c.sch c.dom (2 + e) p T)).1 := by
    unfold chunksOf
    rw [hn]
  have h1 : (chunksOf c p T).map anyV = c.sched.map (·.2.2.2) := by
    have := congrArg Prod.fst
      (roundPat_repFrames (st := c.st) (sch := c.sch) c.dom e p T (by omega) hk (q_ne_zero hs) (by omega) (by omega))
    rw [hch, hsc, map_flag, hft]
    exact this
  have h2 : (c.sched.map (·.2.1.toList)).flatten = p.ligerito.recursiveCaps.toList.map Array.size := by
    rw [hsc, map_tops, hft, schedPat_tops hk1]
    exact tops_caps hk1 hps.rcaps fun i hi => (hps.rcapsEach i hi).1
  obtain ⟨l1, l2, l3, l4⟩ := honestRows_fit c.sched (chunksOf c p T) p.ligerito.recursiveCaps.toList h1 h2
  show FitsRep c (honestRows c.sched (chunksOf c p T) p.ligerito.recursiveCaps.toList) p T
  exact
    { rounds := l2.trans l1.symm
      sched := l1.symm
      cap := hcap
      tops := l3.symm
      nonces := hp.nonces
      sizes := hp.sizes
      levels := hp.levels
      elems := fun j hj => l4 j (by rw [← l1]; exact hj) }

/-- **One honest rep reads back** (`repFits`, `repReadsBack_of`). -/
theorem repReadsBack : RepReadsBack := repReadsBack_of repFits

/-- **An honest prover's rounds read back** (`repReadsBack`, `readsBack_of_rep`): on the honest prover's messages `decR`
decides, and reads exactly the honest proofs. -/
theorem readsBack : ReadsBack := readsBack_of_rep repReadsBack

/-! ## Non-vacuity: an honest rep at `m = 22` -/

/-- `n` zero bytes. -/
def zeroBytes (n : ℕ) : ByteArray := ⟨Array.replicate n 0⟩

theorem zeroBytes_size (n : ℕ) : (zeroBytes n).size = n := Array.size_replicate

/-- **A proof with the lengths `sh` fixes** (`ProofSizes`): every entry zero or `default`, every nonce 0, and the
unabsorbed fields as V\* takes them at `st`, `sch` and the point `z` (`sizedProof_unabsorbed`). -/
def sizedProof (st : Flock.Setup) (sch : Flock.Schedule) (z : F128) (sh : Flock.Shape) : Flock.Proof where
  cap := .replicate (2 ^ (sh.sched.levels.getD 0 default).cap) (zeroBytes sh.hashLen)
  params := pcsParams st.merkle sch
  zerocheck :=
    { round1ab := .replicate 64 default, round1c := .replicate 64 default, rounds := .replicate (sh.m - 6) default
      finalA := default, finalB := default, finalC := dot (lagrange nodesL den6Inv z) (.replicate 64 default) }
  lincheck := { rounds := .replicate (sh.kLog - 6) default, zPartial := .replicate 64 default }
  ringSwitches := .replicate sh.claims (.replicate 128 default)
  ligerito :=
    { initialCap := .replicate (2 ^ (sh.sched.levels.getD 0 default).cap) (zeroBytes sh.hashLen)
      initialProof := default
      recursiveCaps := (Array.range (sh.sched.levels.size - 1)).map fun i =>
        .replicate (2 ^ (sh.sched.levels.getD (i + 1) default).cap) (zeroBytes sh.hashLen)
      recursiveProofs := #[]
      finalProof := { yr := .replicate (2 ^ (sh.sched.ext + 1)) default, rows := #[], siblings := #[] }
      messages := .replicate (sh.sched.k0 + 8 + 7 * (sh.sched.levels.size - 1 - 1)) default
      queryNonces := .replicate (sh.sched.levels.size - 1 + 1) 0
      oodValues := .replicate (1 + 2 * (sh.sched.levels.size - 1)) default
      claimNonces := .replicate (3 * (sh.sched.levels.size - 1) + 2) 0
      consistencyNonces := .replicate (sh.sched.levels.size - 1 + 1) 0 }

/-- `sizedProof st sch z sh` has `sh`'s lengths. -/
theorem sizedProof_sizes (st : Flock.Setup) (sch : Flock.Schedule) (z : F128) (sh : Flock.Shape) :
    FlockSoundness.Discharge.DecodeZL.ProofSizes sh (sizedProof st sch z sh) where
  r1ab := Array.size_replicate
  r1c := Array.size_replicate
  zcRounds := Array.size_replicate
  lcRounds := Array.size_replicate
  zPartial := Array.size_replicate
  rs := Array.size_replicate
  rsEach _ h := by rw [(Array.mem_replicate.1 h).2]; exact Array.size_replicate
  cap := Array.size_replicate
  capEach _ h := by rw [(Array.mem_replicate.1 h).2]; exact zeroBytes_size _
  rcaps := by simp only [sizedProof, Array.size_map, Array.size_range]
  rcapsEach i hi := by
    simp only [sizedProof, Array.getElem_map, Array.getElem_range, Array.size_replicate, true_and]
    intro e he
    rw [(Array.mem_replicate.1 he).2]
    exact zeroBytes_size _
  yr := Array.size_replicate
  messages := Array.size_replicate
  queryNonces := Array.size_replicate
  oodValues := Array.size_replicate
  claimNonces := Array.size_replicate
  consistencyNonces := Array.size_replicate

/-- `sizedProof`'s nonces are 0. -/
theorem sizedProof_nonces (st : Flock.Setup) (sch : Flock.Schedule) (z : F128) (sh : Flock.Shape) :
    (∀ n ∈ (sizedProof st sch z sh).ligerito.queryNonces, n = 0) ∧
      (∀ n ∈ (sizedProof st sch z sh).ligerito.claimNonces, n = 0) ∧
      ∀ n ∈ (sizedProof st sch z sh).ligerito.consistencyNonces, n = 0 :=
  ⟨fun _ h => (Array.mem_replicate.1 h).2, fun _ h => (Array.mem_replicate.1 h).2,
    fun _ h => (Array.mem_replicate.1 h).2⟩

/-- `sizedProof`'s unabsorbed fields are V\*'s: the pinned `params`, the cap as `initialCap`, and `finalC` at `z`. -/
theorem sizedProof_unabsorbed (st : Flock.Setup) (sch : Flock.Schedule) (z : F128) (sh : Flock.Shape) :
    (sizedProof st sch z sh).params = pcsParams st.merkle sch ∧
      (sizedProof st sch z sh).ligerito.initialCap = (sizedProof st sch z sh).cap ∧
      (sizedProof st sch z sh).zerocheck.finalC =
        dot (lagrange nodesL den6Inv z) (sizedProof st sch z sh).zerocheck.round1c :=
  ⟨rfl, rfl, rfl⟩

/-- A setup at `m = 22`, `k_log = 6`, folding at its own `k_log`; everything else `default`. -/
def st₀ : Flock.Setup :=
  { (default : Flock.Setup) with m := 22, kLog := 6, fold := { (default : CircuitFold) with kLog := 6 } }

/-- `fast100`'s schedule at `m = 22`. -/
def sch₀ : Flock.Schedule := (fast100 22).getD default

/-- The honest proof at `st₀`, no extra claims: `sizedProof` at its shape. -/
def p₀ : Flock.Proof := sizedProof st₀ sch₀ default (FlockSoundness.Refine.shapeOf st₀ sch₀ 0)

/-- A rep at `st₀` with two claims, `repRounds`' schedule and `p₀`'s cap. -/
def c₀ : RepCtx where
  st := st₀
  sch := sch₀
  dom := ByteArray.empty
  n := 2
  sched := ((RecVStar.repRounds 22 6 2).getD #[]).toList
  cap0 := p₀.cap.toList
  z := default

theorem fast100_22 : fast100 22 = some sch₀ := by
  obtain ⟨s, hs⟩ := Option.isSome_iff_exists.1 (by decide +kernel : (fast100 22).isSome)
  unfold sch₀
  rw [hs, Option.getD_some]

/-- **`RepReadsBack`'s hypotheses hold together** at `c₀` and `p₀`. -/
theorem honest₀ : SchedOf c₀ ∧ HonestRep c₀ p₀ ∧ p₀.cap.toList = c₀.cap0 := by
  obtain ⟨rs, hrs⟩ := Option.isSome_iff_exists.1 (by decide +kernel : (RecVStar.repRounds 22 6 2).isSome)
  have hs : SchedOf c₀ := by
    unfold SchedOf
    refine ⟨fast100_22, rfl, rs, hrs, ?_⟩
    show ((RecVStar.repRounds 22 6 2).getD #[]).toList = rs.toList
    rw [hrs, Option.getD_some]
  have hL : 2 ≤ sch₀.levels.size := by
    show 2 ≤ ((fast100 22).getD default).levels.size
    decide +kernel
  obtain ⟨h1, h2, h3⟩ := sizedProof_unabsorbed st₀ sch₀ default (FlockSoundness.Refine.shapeOf st₀ sch₀ 0)
  exact ⟨hs, ⟨⟨0, rfl, sizedProof_sizes _ _ _ (FlockSoundness.Refine.shapeOf st₀ sch₀ 0)⟩, hL,
    sizedProof_nonces _ _ _ (FlockSoundness.Refine.shapeOf st₀ sch₀ 0), h1, h2, h3⟩, rfl⟩

/-- **An honest rep reads back** at `c₀`, at every `T`: `repReadsBack`'s conclusion on a run that exists. -/
theorem readRep₀ (T : F128) : readRep c₀ (repRows c₀ p₀ T) p₀ = some (p₀, framesOf c₀ p₀ T) :=
  repReadsBack c₀ p₀ T honest₀.1 honest₀.2.1 honest₀.2.2

/-- **`ReadsBack`'s `SchedOf` at a set-up rep**: rep `k`'s schedule is its frames' wherever its table has a `fast100`
schedule and extra claims at the first coin's link points. -/
theorem schedOf_repCtx {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray} {cs : List Coin}
    {k : ℕ} {extras : Array Flock.Claim} (hk : k / 2 < sts.size) (hf : (fast100 sts[k / 2].m).isSome)
    (he : sts[k / 2].extraClaims (pointsOf spec c₁) = .ok extras) : SchedOf (repCtx spec sts c₁ r₀ cs k) := by
  have hk' : k / 2 < (tabsOf sts).size := by
    unfold tabsOf
    rw [Array.size_map]
    exact hk
  have ht : (tabsOf sts)[k / 2]! = Setup.ofCircuit sts[k / 2] := by
    rw [getElem!_pos (tabsOf sts) (k / 2) hk']
    exact Array.getElem_map ..
  obtain ⟨sch, hsch⟩ := Option.isSome_iff_exists.1 hf
  obtain ⟨-, h35, -⟩ := RecDeploy.fast100_sizes hsch
  have key : ∀ m < 36, (fast100 m).isSome → (RecVStar.repRounds m 6 1).isSome := by decide +kernel
  obtain ⟨out, hout⟩ := Option.isSome_iff_exists.1 (key _ (by omega) hf)
  obtain ⟨T, -, hall⟩ := FlockC3.Compose.repRounds_tail hout
  have h2 := hall sts[k / 2].c.kLog (2 + spec.points * sts[k / 2].regions.size)
  have hst : (repCtx spec sts c₁ r₀ cs k).st = Setup.ofCircuit sts[k / 2] := ht
  have hsch' : (repCtx spec sts c₁ r₀ cs k).sch = sch := by
    show (fast100 (tabsOf sts)[k / 2]!.m).getD default = sch
    rw [ht]
    show (fast100 sts[k / 2].m).getD default = sch
    rw [hsch, Option.getD_some]
  have hn : (repCtx spec sts c₁ r₀ cs k).n = 2 + spec.points * sts[k / 2].regions.size := by
    show 2 + (match (tabsOf sts)[k / 2]!.extra (pointsOf spec c₁) with
      | .ok e => e.size
      | .error _ => 0) = _
    rw [ht]
    show 2 + (match sts[k / 2].extraClaims (pointsOf spec c₁) with
      | .ok e => e.size
      | .error _ => 0) = _
    have hpts : (pointsOf spec c₁).size = spec.points := by
      unfold pointsOf
      rw [Array.size_map, Array.size_range]
    have hsz : extras.size = spec.points * sts[k / 2].regions.size := by
      rw [extraClaims_size he, hpts, Nat.mul_comm]
    rw [he]
    exact congrArg (2 + ·) hsz
  have hsc : (repCtx spec sts c₁ r₀ cs k).sched = RecDeploy.repSchedAt spec sts[k / 2] := by
    show RecDeploy.repSchedAt spec sts[k / 2]! = _
    rw [getElem!_pos sts (k / 2) hk]
  unfold SchedOf
  rw [hst, hsch', hn, hsc]
  refine ⟨hsch, rfl, _, h2, ?_⟩
  simp only [RecDeploy.repSchedAt, h2, Option.map_some, Option.getD_some]

end FlockDeploy
