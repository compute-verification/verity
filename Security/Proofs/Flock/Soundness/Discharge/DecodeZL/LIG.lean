import Proofs.Flock.Soundness.Discharge.DecodeZL.ZL
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes

/-!
# Ligerito's rounds carry its messages (R11b)

`lig_carries`: Ligerito's frames `ligFrames`, read by `wLIG`, are `ligMsgs`, cut at `ligShape`'s coins. `Carries` reads
the spine only through its coin pattern, so the rounds are proved at a pattern (`CarriesP`) and glued with
`CarriesP.append`: the start, the first out-of-domain value (`carriesP_f`), the `k0` folding rounds (`carriesP_fold`),
the last one with the level-1 cap (`carriesP_cap`), level 0's out-of-domain samples and queries (`carriesP_mid`), and
the level loop (`carriesP_levels`, by recursion on the levels left, ending at `yr` with `carriesP_yr`). `fast100`'s
schedules fold 4 lanes above level 0, so `ligShape`'s pattern is the rounds' (`isCoin_rec`), and their caps and query
counts fit the frames' bounds (`fast100_small`).
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

theorem le64_size (x : UInt64) : (le64 x).size = 8 := by
  simp [le64, Std.Legacy.Range.forIn_eq_forIn_range', List.range', ByteArray.size_push]
  rfl

/-- `fast100`'s schedules: `m` from 22 to 35, and per level fewer than 64 columns, 1 to 255 queries and a cap below
8. -/
theorem fast100_small (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100 m = some s) :
    22 ≤ s.m ∧ s.m ≤ 35 ∧ ∀ lv ∈ s.levels, lv.cols < 64 ∧ 0 < lv.q ∧ lv.q < 256 ∧ lv.cap < 8 := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100 m.val, (decide (22 ≤ s.m ∧ s.m ≤ 35) &&
        s.levels.all fun lv => decide (lv.cols < 64 ∧ 0 < lv.q ∧ lv.q < 256 ∧ lv.cap < 8)) = true := by
      decide +kernel
    have h := key ⟨m, hm⟩ s hs
    rw [Bool.and_eq_true, decide_eq_true_iff] at h
    refine ⟨h.1.1, h.1.2, fun lv hlv => ?_⟩
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hlv
    simpa using Array.all_eq_true.1 h.2 i hi
  · rw [Refine.exec_fast100_none m (by omega)] at hs; cases hs

/-! ## Carrying at a coin pattern -/

/-- `Carries` at a coin pattern: `Carries W σ` is `CarriesP W (coinPat σ)`. -/
def CarriesP (W : List (List Wire)) (P : List Bool) (F : List Frame) (ms tail : List Msg) : Prop :=
  (chunks F).2 = [] ∧ W.length = (chunks F).1.length ∧ ms = (rdChunks W F).flatten ++ tail ∧
    P = groupPat ((rdChunks W F).map List.length) tail.length

theorem CarriesP.append {W₁ W₂ : List (List Wire)} {P₁ P₂ : List Bool} {F₁ F₂ : List Frame} {ms₁ ms₂ t : List Msg}
    (h₁ : CarriesP W₁ P₁ F₁ ms₁ []) (h₂ : CarriesP W₂ P₂ F₂ ms₂ t) :
    CarriesP (W₁ ++ W₂) (P₁ ++ P₂) (F₁ ++ F₂) (ms₁ ++ ms₂) t := by
  obtain ⟨p₁, l₁, e₁, c₁⟩ := h₁
  obtain ⟨p₂, l₂, e₂, c₂⟩ := h₂
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [chunks_append_of_closed p₁]; exact p₂
  · rw [chunks_append_of_closed p₁, List.length_append, List.length_append, l₁, l₂]
  · rw [rdChunks_append p₁ l₁, e₁, e₂]; simp
  · rw [rdChunks_append p₁ l₁, c₁, c₂]
    simp [groupPat]

theorem carriesP_nil : CarriesP [] [] [] [] [] := ⟨rfl, rfl, rfl, rfl⟩

theorem carriesP_one {fs : List Frame} {g : Frame} {w : List Wire} {ms : List Msg}
    (hfs : ∀ f ∈ fs, closes f = false) (hg : closes g = true) (hdec : decRound w ⟨(encs (fs ++ [g])).toArray⟩ = ms) :
    CarriesP [w] (List.replicate ms.length false ++ [true]) (fs ++ [g]) ms [] := by
  have hc := chunks_single hfs hg
  have hr : rdChunks [w] (fs ++ [g]) = [ms] := by simp [rdChunks, hc, hdec]
  refine ⟨by rw [hc], by rw [hc]; rfl, by rw [hr]; simp, ?_⟩
  rw [hr]; simp [groupPat]

/-- A round that carries nothing. -/
theorem carriesP_none {fs : List Frame} {g : Frame} (hfs : ∀ f ∈ fs, closes f = false) (hg : closes g = true) :
    CarriesP [[]] [true] (fs ++ [g]) [] [] := by
  simpa using carriesP_one (w := []) hfs hg (decRound_nil _)

/-! ## Ligerito's rounds -/

theorem wf_bytes {b : ByteArray} (h : b.size < 2 ^ 64) : (Frame.bytes b).WF := h

theorem wf_le64 (x : UInt64) : (Frame.bytes (le64 x)).WF := wf_bytes (by rw [le64_size]; norm_num)

theorem kFr_closes (m : F256 × F256) : ∀ f ∈ kFr m, closes f = false := by simp [kFr, closes]

theorem kFr_wf (m : F256 × F256) : ∀ f ∈ kFr m, f.WF := by
  simp only [kFr, List.mem_cons, List.mem_nil_iff, or_false]
  rintro f (rfl | rfl) <;> exact wf_vec (by norm_num)

theorem valsOf_kFr (m : F256 × F256) (xs : List Frame) :
    valsOf (kFr m ++ xs) = [m.1.c0, m.1.c1, m.2.c0, m.2.c1] ++ valsOf xs := by
  simp [valsOf, kFr, Frame.vals]

theorem bytesOf_kFr (m : F256 × F256) (xs : List Frame) : bytesOf (kFr m ++ xs) = bytesOf xs := by
  simp [bytesOf, kFr, Frame.byteStr]

theorem foldl_append_size : ∀ (l : List ByteArray) (acc : ByteArray), (∀ x ∈ l, x.size = 64) →
    (l.foldl (· ++ ·) acc).size = acc.size + 64 * l.length
  | [], acc, _ => by simp
  | x :: l, acc, h => by
      rw [List.foldl_cons, foldl_append_size l _ (fun y hy => h y (List.mem_cons_of_mem _ hy)), ByteArray.size_append,
        h x List.mem_cons_self, List.length_cons]
      ring

theorem wf_capFr {c : Array ByteArray} (h : ∀ x ∈ c.toList, x.size = 64) (hs : c.size < 2 ^ 32) : (capFr c).WF := by
  show (c.foldl (· ++ ·) ByteArray.empty).size < 2 ^ 64
  rw [← Array.foldl_toList, foldl_append_size _ _ h, Array.length_toList]
  have : (2 : ℕ) ^ 64 = 64 * 2 ^ 58 := by norm_num
  simp only [ByteArray.size_empty, Nat.zero_add, this]
  have : (2 : ℕ) ^ 32 < 2 ^ 58 := by norm_num
  omega

/-- A round of one sumcheck message. -/
theorem carriesP_kk (m : F256 × F256) {xs : List Frame} {g : Frame} (hxs : ∀ f ∈ xs, closes f = false)
    (hwxs : ∀ f ∈ xs, f.WF) (hg : closes g = true) (hwg : g.WF) :
    CarriesP [[.kk]] [false, true] ((kFr m ++ xs) ++ [g]) [kMsg m] [] := by
  have hwf : ∀ f ∈ (kFr m ++ xs) ++ [g], f.WF := by
    intro f hf
    simp only [List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at hf
    rcases hf with (hf | hf) | rfl
    · exact kFr_wf m f hf
    · exact hwxs f hf
    · exact hwg
  refine carriesP_one ?_ hg ?_
  · intro f hf
    rcases List.mem_append.1 hf with hf | hf
    · exact kFr_closes m f hf
    · exact hxs f hf
  · rw [decRound_encs hwf, List.append_assoc, valsOf_kFr]
    simp [readAll, Wire.kk, kMsg]

/-- A round of an out-of-domain value and its intro message. -/
theorem carriesP_fkk (o : F128) (m : F256 × F256) (x : UInt64) :
    CarriesP [[.f, .kk]] [false, false, true] ((.one o :: kFr m ++ [.bytes (le64 x)]) ++ [.sq 1 1])
      [fMsg o, kMsg m] [] := by
  have hwf : ∀ f ∈ (.one o :: kFr m ++ [.bytes (le64 x)]) ++ [.sq 1 1], f.WF := by
    intro f hf
    simp only [List.cons_append, List.mem_cons, List.mem_append, List.mem_nil_iff, or_false] at hf
    rcases hf with rfl | (hf | rfl) | rfl
    · exact wf_one _
    · exact kFr_wf m f hf
    · exact wf_le64 x
    · exact wf_sq _ (by norm_num)
  refine carriesP_one ?_ (by simp [closes]) ?_
  · intro f hf
    simp only [List.cons_append, List.mem_cons, List.mem_append, List.mem_nil_iff, or_false] at hf
    rcases hf with rfl | (hf | rfl)
    · rfl
    · exact kFr_closes m f hf
    · rfl
  · rw [decRound_encs hwf]
    simp [readAll, Wire.f, Wire.kk, kMsg, fMsg, valsOf, bytesOf, kFr, Frame.vals, Frame.byteStr]

/-- The round of the first out-of-domain value. -/
theorem carriesP_f (o : F128) (x : UInt64) :
    CarriesP [[.f]] [false, true] ([.one o, .bytes (le64 x)] ++ [.sq 1 1]) [fMsg o] [] := by
  have hwf : ∀ f ∈ [Frame.one o, .bytes (le64 x)] ++ [.sq 1 1], f.WF := by
    simp only [List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false]
    rintro f (rfl | rfl | rfl)
    · exact wf_one _
    · exact wf_le64 x
    · exact wf_sq _ (by norm_num)
  refine carriesP_one (by simp [closes]) (by simp [closes]) ?_
  rw [decRound_encs hwf]
  simp [readAll, Wire.f, fMsg, valsOf, bytesOf, Frame.vals, Frame.byteStr]

/-- The last fold round of a level below the last: its message and the next level's cap. -/
theorem carriesP_cap (m : F256 × F256) {c : Array ByteArray} (hc : ∀ x ∈ c.toList, x.size = 64)
    (hcs : c.size < 2 ^ 32) {dim : ℕ} (hd : dim ≠ 0) (hdw : dim < 2 ^ 64) :
    CarriesP [[.kk, .cap]] [false, false, true] ((kFr m ++ [capFr c]) ++ [.sq 2 dim])
      [kMsg m, ⟨Cap Dig, capDig c⟩] [] := by
  have hwf : ∀ f ∈ (kFr m ++ [capFr c]) ++ [.sq 2 dim], f.WF := by
    intro f hf
    simp only [List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at hf
    rcases hf with (hf | rfl) | rfl
    · exact kFr_wf m f hf
    · exact wf_capFr hc hcs
    · exact wf_sq _ hdw
  refine carriesP_one ?_ (by simp [closes, hd]) ?_
  · intro f hf
    rcases List.mem_append.1 hf with hf | hf
    · exact kFr_closes m f hf
    · simp only [List.mem_cons, List.mem_nil_iff, or_false] at hf; subst hf; rfl
  · rw [decRound_encs hwf, List.append_assoc, valsOf_kFr, bytesOf_kFr]
    simp [readAll, Wire.kk, Wire.cap, kMsg, bytesOf, Frame.byteStr, capFr, capOfRoot_eq hc rfl]

/-- The last level's last fold round: its message and the clear residual `yr`. -/
theorem carriesP_yr (m : F256 × F256) {yr : Array F128} {N : ℕ} (hN : yr.size = N) (x : UInt64) {q : ℕ}
    (hq : q ≠ 0) (hqw : q < 2 ^ 64) :
    CarriesP [[.kk, .v N]] [false, false, true] ((kFr m ++ (yr.toList.map .one ++ [.bytes (le64 x)])) ++ [.sq 2 q])
      [kMsg m, ⟨Fin N → 𝔽, fun i => fOf yr[i.val]!⟩] [] := by
  have hwf : ∀ f ∈ (kFr m ++ (yr.toList.map .one ++ [.bytes (le64 x)])) ++ [.sq 2 q], f.WF := by
    intro f hf
    simp only [List.mem_append, List.mem_map, List.mem_cons, List.mem_nil_iff, or_false] at hf
    rcases hf with (hf | ⟨v, -, rfl⟩ | rfl) | rfl
    · exact kFr_wf m f hf
    · exact wf_one _
    · exact wf_le64 x
    · exact wf_sq _ hqw
  refine carriesP_one ?_ (by simp [closes, hq]) ?_
  · intro f hf
    simp only [List.mem_append, List.mem_map, List.mem_cons, List.mem_nil_iff, or_false] at hf
    rcases hf with hf | ⟨v, -, rfl⟩ | rfl
    · exact kFr_closes m f hf
    · rfl
    · rfl
  · have hv : valsOf ((yr.toList.map .one ++ [.bytes (le64 x)]) ++ [.sq 2 q]) = yr.toList := by
      simp [valsOf, Frame.vals, List.flatMap_map]
    have hl : yr.toList.take N = yr.toList := List.take_of_length_le (by simp [hN])
    rw [decRound_encs hwf, List.append_assoc, valsOf_kFr, hv]
    simp only [readAll, Wire.kk, Wire.v, kMsg, List.cons_append, List.nil_append, List.drop_succ_cons, List.drop_zero,
      hl, Array.getElem!_toList]
    rfl

theorem kMsgs_succ (lp : LigeritoProof) (a k : ℕ) :
    kMsgs lp a (k + 1) = kMsg lp.messages[a]! :: kMsgs lp (a + 1) k := by
  simp [kMsgs, List.range'_succ]

/-- **The folding rounds** from message `a`: `k` rounds of one message, each closed by its fold coin, then the round of
message `a + k`, which carries on into `ys`. -/
theorem carriesP_fold (lp : LigeritoProof) {W : List (List Wire)} {P : List Bool} {ys : List Frame} {msY : List Msg} :
    ∀ (k a : ℕ), CarriesP W P (kFr lp.messages[a + k]! ++ ys) (kMsg lp.messages[a + k]! :: msY) [] →
      CarriesP (List.replicate k [.kk] ++ W) ((List.replicate k [false, true]).flatten ++ P)
        (kFr lp.messages[a]! ++ (chFr lp (a + 1) k ++ ys)) (kMsg lp.messages[a]! :: (kMsgs lp (a + 1) k ++ msY)) []
  | 0, a, h => by simpa [chFr, kMsgs] using h
  | k + 1, a, h => by
      have h₁ := carriesP_kk lp.messages[a]! (xs := []) (g := .sq 2 2) (by simp) (by simp) (by simp [closes])
        (wf_sq _ (by norm_num))
      have := h₁.append (carriesP_fold lp k (a + 1) (by rwa [show a + 1 + k = a + (k + 1) by omega]))
      rw [show chFr lp (a + 1) (k + 1) = .sq 2 2 :: (kFr lp.messages[a + 1]! ++ chFr lp (a + 1 + 1) k) from rfl,
        kMsgs_succ]
      simpa [List.replicate_succ, List.append_assoc] using this

/-- `carriesP_fold`, with the round of message `a + k` given as its frames and messages. -/
theorem carriesP_fold' (lp : LigeritoProof) {W : List (List Wire)} {P : List Bool} {xs rest : List Frame} {g : Frame}
    {ms₁ ms₂ : List Msg} (k a : ℕ)
    (h : CarriesP W P (((kFr lp.messages[a + k]! ++ xs) ++ [g]) ++ rest) ((kMsg lp.messages[a + k]! :: ms₁) ++ ms₂)
      []) :
    CarriesP (List.replicate k [.kk] ++ W) ((List.replicate k [false, true]).flatten ++ P)
      (kFr lp.messages[a]! ++ (chFr lp (a + 1) k ++ (xs ++ ([g] ++ rest))))
      (kMsg lp.messages[a]! :: (kMsgs lp (a + 1) k ++ (ms₁ ++ ms₂))) [] :=
  carriesP_fold lp k a (by simpa only [List.append_assoc, List.cons_append] using h)

/-- The wires of a level's two out-of-domain rounds (the second point's coins carry nothing), its two query rounds and
its query message. -/
def wMid : List (List Wire) := [[.f, .kk], [], [.f, .kk], [], [], [.kk]]

/-- `wMid`'s coin pattern. -/
def pMid : List Bool := [false, false, true, true, false, false, true, true, true, false, true]

/-- **A level's out-of-domain samples and queries**: two out-of-domain values with their intro messages, the queries'
coins, and the query message. -/
theorem carriesP_mid (lp : LigeritoProof) (dim o₁ o₂ x₁ x₂ x₃ c₁ c₂ c₃ qn cn : ℕ) (lv : Flock.Level) (hd : dim ≠ 0)
    (hq : lv.q ≠ 0) :
    CarriesP wMid pMid
      ((.one lp.oodValues[o₁]! :: kFr lp.messages[x₁]! ++ [.bytes (le64 lp.claimNonces[c₁]!), .sq 1 1]) ++
        (introFr lp dim o₂ x₂ c₂ ++
          (queryFr lp lv qn cn ++ (kFr lp.messages[x₃]! ++ [.bytes (le64 lp.claimNonces[c₃]!), .sq 1 1]))))
      [fMsg lp.oodValues[o₁]!, kMsg lp.messages[x₁]!, fMsg lp.oodValues[o₂]!, kMsg lp.messages[x₂]!,
        kMsg lp.messages[x₃]!] [] := by
  have h₁ := carriesP_fkk lp.oodValues[o₁]! lp.messages[x₁]! lp.claimNonces[c₁]!
  have h₂ := carriesP_none (fs := []) (g := .sq 2 dim) (by simp) (by simp [closes, hd])
  have h₃ := carriesP_fkk lp.oodValues[o₂]! lp.messages[x₂]! lp.claimNonces[c₂]!
  have h₄ := carriesP_none (fs := [.bytes (le64 lp.queryNonces[qn]!)]) (g := .sq 2 lv.q) (by simp [closes])
    (by simp [closes, hq])
  have h₅ := carriesP_none (fs := [.bytes (le64 lp.consistencyNonces[cn]!)]) (g := .sq 2 ((lv.q - 1).log2 + 1))
    (by simp [closes]) (by simp [closes])
  have h₆ := carriesP_kk lp.messages[x₃]! (xs := [.bytes (le64 lp.claimNonces[c₃]!)]) (g := .sq 1 1)
    (by simp [closes]) (by simp [wf_le64]) (by simp [closes]) (wf_sq _ (by norm_num))
  have := h₁.append (h₂.append (h₃.append (h₄.append (h₅.append h₆))))
  simpa [wMid, pMid, introFr, queryFr, List.append_assoc] using this

/-- **The level loop's wires**, `j` levels left: the first fold coin's round carries nothing, three fold rounds their
message, the last fold round its message and the next level's cap, then `wMid`; at the last level, the last fold round's
message and `yr`, then the two query rounds and the batching coin's carry nothing. -/
noncomputable def wLev (ext : ℕ) : ℕ → List (List Wire)
  | 0 => []
  | 1 => [[]] ++ (List.replicate 3 [.kk] ++ ([[.kk, .v (2 ^ (ext + 1))]] ++ ([[]] ++ [[]])))
  | j + 2 => [[]] ++ (List.replicate 3 [.kk] ++ ([[.kk, .cap]] ++ (wMid ++ wLev ext (j + 1))))

/-- `wLev`'s coin pattern. -/
def pLev : ℕ → List Bool
  | 0 => []
  | 1 => [true] ++ ((List.replicate 3 [false, true]).flatten ++ ([false, false, true] ++ ([true] ++ [true])))
  | j + 2 => [true] ++ ((List.replicate 3 [false, true]).flatten ++ ([false, false, true] ++ (pMid ++ pLev (j + 1))))

theorem getD_mem {α : Type} {a : Array α} {k : ℕ} (h : k < a.size) (d : α) : a.getD k d ∈ a := by
  simp [Array.getD_eq_getD_getElem?, h]

theorem caps_small {lp : LigeritoProof} {sch : Flock.Schedule}
    (hsm : ∀ lv ∈ sch.levels, lv.cols < 64 ∧ 0 < lv.q ∧ lv.q < 256 ∧ lv.cap < 8)
    (hrc : lp.recursiveCaps.size = sch.levels.size - 1)
    (hrce : ∀ (i : ℕ) (hi : i < lp.recursiveCaps.size),
      lp.recursiveCaps[i].size = 2 ^ (sch.levels.getD (i + 1) default).cap ∧ ∀ e ∈ lp.recursiveCaps[i], e.size = 64)
    {i : ℕ} (hi : i < lp.recursiveCaps.size) :
    (∀ x ∈ (lp.recursiveCaps[i]!).toList, x.size = 64) ∧ (lp.recursiveCaps[i]!).size < 2 ^ 32 := by
  rw [getElem!_pos lp.recursiveCaps i hi]
  obtain ⟨hs, he⟩ := hrce i hi
  obtain ⟨-, -, -, hcap⟩ := hsm _ (getD_mem (show i + 1 < sch.levels.size by omega) default)
  refine ⟨fun x hx => he x (Array.mem_toList_iff.1 hx), ?_⟩
  rw [hs]
  calc 2 ^ (sch.levels.getD (i + 1) default).cap < 2 ^ 8 := Nat.pow_lt_pow_right (by norm_num) hcap
    _ ≤ 2 ^ 32 := by norm_num

theorem lt_two64 {x : ℕ} (h : x < 256) : x < 2 ^ 64 := lt_of_lt_of_le h (by norm_num)

/-- **The level loop**: with `j` levels left from level `i + 1`, its rounds carry `recMsgs`. -/
theorem carriesP_levels (sch : Flock.Schedule) (lp : LigeritoProof)
    (hsm : ∀ lv ∈ sch.levels, lv.cols < 64 ∧ 0 < lv.q ∧ lv.q < 256 ∧ lv.cap < 8)
    (hrc : lp.recursiveCaps.size = sch.levels.size - 1)
    (hrce : ∀ (i : ℕ) (hi : i < lp.recursiveCaps.size),
      lp.recursiveCaps[i].size = 2 ^ (sch.levels.getD (i + 1) default).cap ∧ ∀ e ∈ lp.recursiveCaps[i], e.size = 64)
    (hyr : lp.finalProof.yr.size = 2 ^ (sch.ext + 1)) :
    ∀ (j i : ℕ), i + j = sch.levels.size - 1 →
      CarriesP (wLev sch.ext j) (pLev j) ((List.range' i j).flatMap (levelFr sch lp))
        (recMsgs lp capDig sch.ext j i (sch.k0 + 4 + 7 * i) (3 + 2 * i)) []
  | 0, i, _ => by simpa [wLev, pLev, recMsgs] using carriesP_nil
  | 1, i, hij => by
      have hlv := getD_mem (show sch.levels.size - 1 < sch.levels.size by omega) default
      obtain ⟨-, hq0, hq, -⟩ := hsm _ hlv
      have hS := carriesP_none (fs := []) (g := .sq 2 2) (by simp) (by simp [closes])
      have hT := carriesP_yr lp.messages[sch.k0 + 4 + 7 * i + 3]! hyr lp.queryNonces[1 + i]!
        (q := (sch.levels.getD (sch.levels.size - 1) default).q) (by omega) (lt_two64 hq)
      have hQ := carriesP_none (fs := [.bytes (le64 lp.consistencyNonces[1 + i]!)])
        (g := .sq 2 (((sch.levels.getD (sch.levels.size - 1) default).q - 1).log2 + 1)) (by simp [closes])
        (by simp [closes])
      have hC := carriesP_none (fs := [.bytes (le64 lp.claimNonces[4 + 3 * i]!)]) (g := .sq 1 1) (by simp [closes])
        (by simp [closes])
      have := hS.append (carriesP_fold' lp 3 (sch.k0 + 4 + 7 * i) (hT.append (hQ.append hC)))
      rw [show List.range' i 1 = [i] from rfl, List.flatMap_cons, List.flatMap_nil, List.append_nil, levelFr,
        ite_eq_right (show ¬(i + 1 < sch.levels.size - 1) by omega),
        show chFr lp (sch.k0 + 4 + 7 * i) 4 =
          .sq 2 2 :: (kFr lp.messages[sch.k0 + 4 + 7 * i]! ++ chFr lp (sch.k0 + 4 + 7 * i + 1) 3) from rfl,
        recMsgs, kMsgs_succ, show yrOf lp sch.ext = fun i => fOf lp.finalProof.yr[i.val]! from rfl]
      simpa [wLev, pLev, queryFr, List.append_assoc] using this
  | j + 2, i, hij => by
      have hi1 : i + 1 < sch.levels.size - 1 := by omega
      have hlv := getD_mem (show i + 1 < sch.levels.size by omega) default
      obtain ⟨hcols, hq0, -, -⟩ := hsm _ hlv
      obtain ⟨hc, hcs⟩ := caps_small hsm hrc hrce (i := i + 1) (by omega)
      have hS := carriesP_none (fs := []) (g := .sq 2 2) (by simp) (by simp [closes])
      have hC := carriesP_cap lp.messages[sch.k0 + 4 + 7 * i + 3]! hc hcs
        (dim := (sch.levels.getD (i + 1) default).cols + 1) (by omega) (lt_two64 (by omega))
      have hM := carriesP_mid lp ((sch.levels.getD (i + 1) default).cols + 1) (3 + 2 * i) (4 + 2 * i)
        (sch.k0 + 8 + 7 * i) (sch.k0 + 9 + 7 * i) (sch.k0 + 10 + 7 * i) (4 + 3 * i) (5 + 3 * i) (6 + 3 * i) (1 + i)
        (1 + i) (sch.levels.getD (i + 1) default) (by omega) (by omega)
      have hL := carriesP_levels sch lp hsm hrc hrce hyr (j + 1) (i + 1) (by omega)
      have := hS.append (carriesP_fold' lp 3 (sch.k0 + 4 + 7 * i) (hC.append (hM.append hL)))
      rw [List.range'_succ, List.flatMap_cons, levelFr, ite_eq_left hi1,
        show chFr lp (sch.k0 + 4 + 7 * i) 4 =
          .sq 2 2 :: (kFr lp.messages[sch.k0 + 4 + 7 * i]! ++ chFr lp (sch.k0 + 4 + 7 * i + 1) 3) from rfl,
        recMsgs, kMsgs_succ, show sch.k0 + 4 + 7 * i + 4 = sch.k0 + 8 + 7 * i by omega,
        show sch.k0 + 4 + 7 * i + 5 = sch.k0 + 9 + 7 * i by omega,
        show sch.k0 + 4 + 7 * i + 6 = sch.k0 + 10 + 7 * i by omega,
        show sch.k0 + 4 + 7 * i + 7 = sch.k0 + 4 + 7 * (i + 1) by omega, show 3 + 2 * i + 1 = 4 + 2 * i by omega,
        show 3 + 2 * i + 2 = 3 + 2 * (i + 1) by omega]
      simpa [wLev, pLev, introFr, List.append_assoc] using this

/-! ## The coin pattern of Ligerito's spine -/

theorem isCoin_fold (F K : Type) : ∀ k, (foldShape F K k).map isCoin = (List.replicate k [true, false]).flatten
  | 0 => rfl
  | k + 1 => by
      rw [show (foldShape F K (k + 1)).map isCoin = true :: false :: (foldShape F K k).map isCoin from rfl,
        isCoin_fold F K k]
      simp [List.replicate_succ]

theorem shift_pat : ∀ (k : ℕ) (l : List Bool),
    false :: ((List.replicate k [true, false]).flatten ++ l) = (List.replicate k [false, true]).flatten ++ false :: l
  | 0, l => rfl
  | k + 1, l => by simp [List.replicate_succ, shift_pat k]

/-- The level loop's spine has pattern `pLev`, when every level folds 4 lanes. -/
theorem isCoin_rec (F K D : Type) (ext : ℕ) : ∀ lvs : List Accounting.Level, (∀ lv ∈ lvs, lv.lanes = 4) →
    (recShape F K D ext lvs).map isCoin = pLev lvs.length
  | [], _ => rfl
  | [lv], h => by
      have h4 : lv.lanes = 4 := h lv (by simp)
      simp only [recShape, List.map_append, List.map_cons, isCoin_fold, h4, qbShape, isCoin, List.map_nil]
      rfl
  | lv :: lv' :: rest, h => by
      have h4 : lv.lanes = 4 := h lv (by simp)
      have ih := isCoin_rec F K D ext (lv' :: rest) (fun x hx => h x (List.mem_cons_of_mem _ hx))
      simp only [recShape, List.map_append, List.map_cons, isCoin_fold, h4, qbShape, oodShape, isCoin, List.map_nil, ih]
      rfl

/-- **The Ligerito wires**: the start's round carries nothing, the first out-of-domain round its value, the `k0` fold
rounds their messages, the last one also the level-1 cap, then level 0's `wMid` and the level loop's `wLev`. -/
noncomputable def wLIG (s : Accounting.Schedule) : List (List Wire) :=
  [[]] ++ ([[.f]] ++ (List.replicate s.k0 [.kk] ++ ([[.kk, .cap]] ++ (wMid ++ wLev s.ext (s.levels.length - 1)))))

theorem lig_carries : LIGCarries wLIG := by
  intro st sch nExtra bytes p T hf h64 hp
  have hs := decodeProof_sizes _ _ _ hp
  obtain ⟨hdim, -⟩ := fast100_dims _ _ hf
  obtain ⟨hm22, hm35, hsm⟩ := fast100_small _ _ hf
  obtain ⟨hn, h2, -, -, hl4, -, -⟩ := hdim
  have hrc : p.ligerito.recursiveCaps.size = sch.levels.size - 1 := by simpa [shapeOf] using hs.rcaps
  have hrce : ∀ (i : ℕ) (hi : i < p.ligerito.recursiveCaps.size),
      p.ligerito.recursiveCaps[i].size = 2 ^ (sch.levels.getD (i + 1) default).cap ∧
        ∀ e ∈ p.ligerito.recursiveCaps[i], e.size = 64 := by
    intro i hi
    have := hs.rcapsEach i hi
    simp only [shapeOf, h64] at this
    exact this
  have hyr : p.ligerito.finalProof.yr.size = 2 ^ (sch.ext + 1) := by simpa [shapeOf] using hs.yr
  have hlen : 2 ≤ sch.levels.toList.length := by simpa using h2
  obtain ⟨lv0, lv1, rest', hl⟩ : ∃ a b r, sch.levels.toList = a :: b :: r := by
    rcases hx : sch.levels.toList with _ | ⟨a, _ | ⟨b, r⟩⟩
    · rw [hx] at hlen; simp at hlen
    · rw [hx] at hlen; simp at hlen
    · exact ⟨a, b, r, rfl⟩
  have hsz : sch.levels.size = rest'.length + 2 := by rw [← Array.length_toList, hl]; simp
  have hl4' : ∀ x ∈ lv1 :: rest', x.lanes = 4 := by
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
    have e : sch.levels.getD (j + 1) default = (lv1 :: rest')[j] := by
      rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hl]
      simp [List.getElem?_eq_getElem hj]
    rw [← e]
    exact hl4 (j + 1) (by simp at hj; omega) (by omega)
  have hpat : coinPat (ligShape 𝔽 𝕂 Dig (schedOf sch)) = [true] ++ ([false, true] ++
      ((List.replicate sch.k0 [false, true]).flatten ++
        ([false, false, true] ++ (pMid ++ pLev (sch.levels.size - 1))))) := by
    have hL : (schedOf sch).levels = levelOf lv0 :: (levelOf lv1 :: rest'.map levelOf) := by simp [schedOf, hl]
    have hr := isCoin_rec 𝔽 𝕂 Dig (schedOf sch).ext (levelOf lv1 :: rest'.map levelOf) (by
      intro x hx
      simp only [List.mem_cons, List.mem_map] at hx
      rcases hx with rfl | ⟨y, hy, rfl⟩
      · exact hl4' lv1 (by simp)
      · exact hl4' y (by simp [hy]))
    rw [show sch.levels.size - 1 = (levelOf lv1 :: rest'.map levelOf).length by simp [hsz]]
    unfold ligShape coinPat
    rw [hL]
    simp only [List.map_cons, List.map_append, isCoin, isCoin_fold, oodShape, qbShape, hr, List.map_nil]
    simp [schedOf, shift_pat, pMid, isCoin]
  obtain ⟨hc, hcs⟩ := caps_small hsm hrc hrce (i := 0) (by omega)
  have hlv0 := hsm _ (getD_mem (show 0 < sch.levels.size by omega) default)
  have hS := carriesP_none (fs := [.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, capFr p.cap])
    (g := .sq 2 sch.n) (by simp [closes, capFr]) (by simp [closes]; omega)
  have hA := carriesP_f p.ligerito.oodValues[0]! p.ligerito.claimNonces[0]!
  have hC := carriesP_cap p.ligerito.messages[0 + sch.k0]! hc hcs (dim := sch.n - sch.k0 + 1) (by omega)
    (lt_two64 (by omega))
  have hM := carriesP_mid p.ligerito (sch.n - sch.k0 + 1) 1 2 (sch.k0 + 1) (sch.k0 + 2) (sch.k0 + 3) 1 2 3 0 0
    (sch.levels.getD 0 default) (by omega) (by omega)
  have hLv := carriesP_levels sch p.ligerito hsm hrc hrce hyr (sch.levels.size - 1) 0 (by omega)
  have := hS.append (hA.append (carriesP_fold' p.ligerito sch.k0 0 (hC.append (hM.append hLv))))
  show CarriesP _ (coinPat _) _ _ _
  rw [hpat]
  simpa [wLIG, schedOf, ligFrames, ligMsgs, introFr, List.append_assoc] using this

end FlockSoundness.Discharge.DecodeZL
