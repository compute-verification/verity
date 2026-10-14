import Proofs.Flock.C3.Sparse.Read

/-!
# A play of a rep's sparse phase is a RecSparse unit

`unit_of_playsT`: a play of the sparse phase to a claim, on steps `l`, is the steps of a RecSparse unit its verifier row
accepts at the phase's public values, slots' types and `h` (`SparseUnit`, `stepsOf … = l`). The play takes the phase's
shape (`sparsePhase_exact`), so its messages are typed as the phase receives them (`PMsgs`), which a witness lays at the
message ports' offsets (`swOf`), and its coins are words (`coins_ofFn`). The unit's row plays the phase on the same
steps (`sparse_refines`, `playsT_stepsOf`), and a game plays on at most one prefix of a tape to one output
(`playsT_prefix`), so that row ends at the play's claim, which is the registered one at the `claim` port.
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Refine FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockC3.Compose (playsT_prefix)
open FlockVBridge.Sparse (SCoins nw nc nR deg degSum tw wPre wV2 wSc wU wVt wClaim roundOf roundOf_degSum
  degSum_add_lt msg2 msg3 uOf vtTape phaseMsgs coinTape msgTape pubOf verdictOf sparseClaim TypesIn sparse_refines)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The play's messages and coins, by type -/

/-- A kind's message type. -/
def kMsgTy : Kind → Option Type
  | .msg M => some M
  | .coin _ => none

/-- A kind's coin type. -/
def kCoinTy : Kind → Option Type
  | .msg _ => none
  | .coin C => some C

/-- The messages' types are the message kinds'. -/
theorem msgs_fst : ∀ l : List Step, (l.filterMap Sum.getLeft?).map Sigma.fst = (l.map kindOf).filterMap kMsgTy
  | [] => rfl
  | .inl x :: l => congrArg (x.1 :: ·) (msgs_fst l)
  | .inr _ :: l => msgs_fst l

theorem coins_fst : ∀ l : List Step, (l.filterMap Sum.getRight?).map Sigma.fst = (l.map kindOf).filterMap kCoinTy
  | [] => rfl
  | .inl _ :: l => coins_fst l
  | .inr x :: l => congrArg (x.1 :: ·) (coins_fst l)

/-- **Messages of one type `M` are a tape** `j ↦ ⟨M, g j⟩`. -/
theorem tape_of_fst {M : Type} (d : M) : ∀ (l : List Msg) (a n : ℕ), l.map Sigma.fst = List.replicate n M →
    ∃ g : ℕ → M, l = (List.range' a n).map fun j => (⟨M, g j⟩ : Msg)
  | [], _, 0, _ => ⟨fun _ => d, rfl⟩
  | [], _, _ + 1, h => by simp at h
  | _ :: _, _, 0, h => by simp at h
  | ⟨M', v⟩ :: l, a, n + 1, h => by
      rw [List.map_cons, List.replicate_succ, List.cons.injEq] at h
      obtain ⟨g, rfl⟩ := tape_of_fst d l (a + 1) n h.2
      obtain rfl : M' = M := h.1
      refine ⟨fun j => if j = a then v else g j, ?_⟩
      rw [List.range'_succ, List.map_cons]
      refine congr (congrArg List.cons (by simp)) (List.map_congr_left fun j hj => ?_)
      rw [List.mem_range'_1] at hj
      simp [show j ≠ a by omega]

theorem single_of_fst {M : Type} {l : List Msg} (h : l.map Sigma.fst = [M]) : ∃ v : M, l = [⟨M, v⟩] := by
  obtain ⟨⟨M', v⟩, rfl, hM⟩ := List.map_eq_singleton_iff.1 h
  obtain rfl : M' = M := hM
  exact ⟨v, rfl⟩

theorem degKinds_msgs (k : ℕ) : ∀ n, (degKinds 𝔽 k n).filterMap kMsgTy = List.replicate n (𝔽 × (Fin k → 𝔽))
  | 0 => rfl
  | n + 1 => congrArg (_ :: ·) (degKinds_msgs k n)

theorem degKinds_coins (k : ℕ) : ∀ n, (degKinds 𝔽 k n).filterMap kCoinTy = List.replicate n 𝔽
  | 0 => rfl
  | n + 1 => congrArg (_ :: ·) (degKinds_coins k n)

theorem rep_msgs (C : Type) : ∀ n, (List.replicate n (Kind.coin C)).filterMap kMsgTy = []
  | 0 => rfl
  | n + 1 => rep_msgs C n

theorem rep_coins (C : Type) : ∀ n, (List.replicate n (Kind.coin C)).filterMap kCoinTy = List.replicate n C
  | 0 => rfl
  | n + 1 => congrArg (C :: ·) (rep_coins C n)

theorem msg_one (M : Type) : [Kind.msg M].filterMap kMsgTy = [M] := rfl

theorem msgc_one (M : Type) : [Kind.msg M].filterMap kCoinTy = [] := rfl

/-- **The phase's claim is at its coins** `(r_k, ζ_p, ρ_t)`, whatever its value. -/
theorem sparseVerdict_shape {K : Type} [Field K] {A : Arith 𝔽 K} {sh : SparseShape} {pub : SparsePub 𝔽 sh}
    {τ : Fin (2 ^ sh.sLog) → ℕ} {ζc ζr : Fin 6 → 𝔽} {c₂ v₂ c₄ : 𝔽} {rk : Fin sh.nu → 𝔽} {rs : Fin sh.sLog → 𝔽}
    {u : Fin (2 ^ sh.pLog) → 𝔽} {ζp : Fin sh.pLog → 𝔽} {vt : Fin (2 ^ sh.tLog) → 𝔽} {ρ : Fin sh.tLog → 𝔽}
    {y : RegClaim 𝔽 sh} (h : sparseVerdict A sh pub τ ζc ζr c₂ v₂ rk rs c₄ u ζp vt ρ = some y) :
    y = ⟨y.value, rk, ζp, ρ⟩ := by
  simp only [sparseVerdict] at h
  split at h
  · split at h
    · cases h; rfl
    · cases h
  · split at h
    · cases h; rfl
    · cases h

variable (sh : SparseShape)

/-- The sparse phase's message types, in order. -/
def msgTys : List Type :=
  List.replicate 6 (𝔽 × (Fin 1 → 𝔽)) ++ ([𝔽] ++ (List.replicate sh.nu (𝔽 × (Fin (2 * sh.ell) → 𝔽)) ++
    (List.replicate sh.sLog (𝔽 × (Fin (2 * sh.ell + 1) → 𝔽)) ++ ([Fin (2 ^ sh.pLog) → 𝔽] ++
      if sh.tLog = 0 then [] else [Fin (2 ^ sh.tLog) → 𝔽]))))

theorem sparseKinds_msgs : (sparseKinds 𝔽 sh).filterMap kMsgTy = msgTys sh := by
  unfold sparseKinds msgTys
  by_cases ht : sh.tLog = 0
  · rw [ite_eq_left ht, ite_eq_left ht]
    simp only [List.filterMap_append, rep_msgs, degKinds_msgs, msg_one, List.nil_append, List.append_nil]
  · rw [ite_eq_right ht, ite_eq_right ht]
    simp only [List.filterMap_append, rep_msgs, degKinds_msgs, msg_one, List.nil_append, List.append_nil]

theorem sparseKinds_coins : (sparseKinds 𝔽 sh).filterMap kCoinTy = List.replicate (nc sh) 𝔽 := by
  have e : (if sh.tLog = 0 then [] else [Kind.msg (Fin (2 ^ sh.tLog) → 𝔽)]).filterMap kCoinTy = [] := by
    split <;> rfl
  rw [show nc sh = 6 + (6 + (sh.nu + (sh.sLog + (sh.pLog + sh.tLog)))) by unfold nc nR; omega]
  unfold sparseKinds
  simp only [List.filterMap_append, rep_coins, degKinds_coins, msgc_one, e, List.nil_append,
    List.replicate_append_replicate]

/-- A play's messages, by type: H2's rounds', `v₂`, H3's rounds' over the entries and over the slots, the planes
`u`, and the types' values `vt` (unused with `T = 1`). -/
structure PMsgs where
  m2 : ℕ → 𝔽 × (Fin 1 → 𝔽)
  v2 : 𝔽
  m3 : ℕ → 𝔽 × (Fin (2 * sh.ell) → 𝔽)
  m4 : ℕ → 𝔽 × (Fin (2 * sh.ell + 1) → 𝔽)
  u : Fin (2 ^ sh.pLog) → 𝔽
  vt : Fin (2 ^ sh.tLog) → 𝔽

/-- The types' values' message, sent only with `T > 1`. -/
def vtList (vt : Fin (2 ^ sh.tLog) → 𝔽) : List Msg := if sh.tLog = 0 then [] else [⟨Fin (2 ^ sh.tLog) → 𝔽, vt⟩]

/-- The messages, in the order the phase receives them. -/
def pList (p : PMsgs sh) : List Msg :=
  msgTape p.m2 0 6 ++ ([⟨𝔽, p.v2⟩] ++ (msgTape p.m3 0 sh.nu ++ (msgTape p.m4 sh.nu sh.sLog ++
    ([⟨Fin (2 ^ sh.pLog) → 𝔽, p.u⟩] ++ vtList sh p.vt))))

theorem pmsgs_of_tys {ms : List Msg} (h : ms.map Sigma.fst = msgTys sh) : ∃ p : PMsgs sh, ms = pList sh p := by
  unfold msgTys at h
  obtain ⟨l₁, r₁, rfl, h₁, h⟩ := List.map_eq_append_iff.1 h
  obtain ⟨l₂, r₂, rfl, h₂, h⟩ := List.map_eq_append_iff.1 h
  obtain ⟨l₃, r₃, rfl, h₃, h⟩ := List.map_eq_append_iff.1 h
  obtain ⟨l₄, r₄, rfl, h₄, h⟩ := List.map_eq_append_iff.1 h
  obtain ⟨l₅, l₆, rfl, h₅, h₆⟩ := List.map_eq_append_iff.1 h
  obtain ⟨m2, rfl⟩ := tape_of_fst (0, fun _ => 0) l₁ 0 6 h₁
  obtain ⟨v2, rfl⟩ := single_of_fst h₂
  obtain ⟨m3, rfl⟩ := tape_of_fst (0, fun _ => 0) l₃ 0 sh.nu h₃
  obtain ⟨m4, rfl⟩ := tape_of_fst (0, fun _ => 0) l₄ sh.nu sh.sLog h₄
  obtain ⟨u, rfl⟩ := single_of_fst h₅
  by_cases ht : sh.tLog = 0
  · rw [ite_eq_left ht, List.map_eq_nil_iff] at h₆
    subst h₆
    refine ⟨⟨m2, v2, m3, m4, u, fun _ => 0⟩, ?_⟩
    unfold pList vtList
    rw [ite_eq_left ht]
    rfl
  · rw [ite_eq_right ht] at h₆
    obtain ⟨vt, rfl⟩ := single_of_fst h₆
    refine ⟨⟨m2, v2, m3, m4, u, vt⟩, ?_⟩
    unfold pList vtList
    rw [ite_eq_right ht]
    rfl

/-! ## The witness at the message ports' offsets -/

theorem arr_ofFn_get {α : Type} [Inhabited α] {n : ℕ} (f : Fin n → α) {i : ℕ} (hi : i < n) :
    (Array.ofFn f)[i]! = f ⟨i, hi⟩ := by
  rw [getElem!_pos _ _ (by simpa using hi), Array.getElem_ofFn]

theorem vget_fin {n : ℕ} (u : Fin n → 𝔽) (i : Fin n) : vget u i.val = u i := by
  simp [vget]

/-- A round's message's element `o`: `c₀`, then `c₂ …`. -/
def pv {k : ℕ} (m : 𝔽 × (Fin k → 𝔽)) (o : ℕ) : 𝔽 := if o = 0 then m.1 else vget m.2 (o - 1)

theorem pv_zero {k : ℕ} (m : 𝔽 × (Fin k → 𝔽)) : pv m 0 = m.1 := rfl

theorem pv_succ {k : ℕ} (m : 𝔽 × (Fin k → 𝔽)) (i : Fin k) : pv m (i.val + 1) = m.2 i := by
  rw [pv, ite_eq_right (by omega), Nat.add_sub_cancel, vget_fin]

/-- w's element `i`, laid as `message_ports` lays it: `h`, `pre<j>`, `v2`, `sc<j>`, `u`, `vt`, then `claim`. -/
def swEl (p : PMsgs sh) (h : Fin 64 → 𝔽) (cl : 𝔽) (i : ℕ) : 𝔽 :=
  if i < 64 then vget h i
  else if i < 76 then (if (i - 64) % 2 = 0 then (p.m2 ((i - 64) / 2)).1 else (p.m2 ((i - 64) / 2)).2 0)
  else if i = 76 then p.v2
  else if i < wU sh then
    if roundOf sh (i - 77) < sh.nu then pv (p.m3 (roundOf sh (i - 77))) (i - 77 - degSum sh (roundOf sh (i - 77)))
    else pv (p.m4 (roundOf sh (i - 77))) (i - 77 - degSum sh (roundOf sh (i - 77)))
  else if i < wVt sh then vget p.u (i - wU sh)
  else if i < wClaim sh then vget p.vt (i - wVt sh)
  else cl

/-- **The witness of a play's messages**, with `h` first and `cl` at `claim`. -/
def swOf (p : PMsgs sh) (h : Fin 64 → 𝔽) (cl : 𝔽) : Array Flock.F128 :=
  Array.ofFn (n := nw sh) fun i => swEl sh p h cl i.val

section Layout

variable (p : PMsgs sh) (h : Fin 64 → 𝔽) (cl : 𝔽)

/-- The offsets, for `omega`, which takes `2 ^ k` as an atom it doesn't know is positive. -/
theorem woffs : wU sh = 77 + degSum sh (nR sh) ∧ wVt sh = wU sh + 2 ^ sh.pLog ∧ wClaim sh = wVt sh + tw sh ∧
    nw sh = wClaim sh + 1 ∧ 0 < 2 ^ sh.pLog := ⟨rfl, rfl, rfl, rfl, Nat.two_pow_pos _⟩

theorem swOf_size : (swOf sh p h cl).size = nw sh := Array.size_ofFn

theorem swOf_get {i : ℕ} (hi : i < nw sh) : fOf (swOf sh p h cl)[i]! = swEl sh p h cl i :=
  congrArg fOf (arr_ofFn_get _ hi)

theorem w_h (s : Fin 64) : fOf (swOf sh p h cl)[s.val]! = h s := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := woffs sh
  have := s.isLt
  rw [swOf_get sh p h cl (by omega), swEl, ite_eq_left s.isLt, vget_fin]

theorem w_pre0 {j : ℕ} (hj : j < 6) : fOf (swOf sh p h cl)[wPre j]! = (p.m2 j).1 := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := woffs sh
  unfold wPre
  rw [swOf_get sh p h cl (by omega), swEl, ite_eq_right (by omega), ite_eq_left (by omega), ite_eq_left (by omega),
    show (64 + 2 * j - 64) / 2 = j by omega]

theorem w_pre1 {j : ℕ} (hj : j < 6) : fOf (swOf sh p h cl)[wPre j + 1]! = (p.m2 j).2 0 := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := woffs sh
  unfold wPre
  rw [swOf_get sh p h cl (by omega), swEl, ite_eq_right (by omega), ite_eq_left (by omega), ite_eq_right (by omega),
    show (64 + 2 * j + 1 - 64) / 2 = j by omega]

theorem w_v2 : fOf (swOf sh p h cl)[wV2]! = p.v2 := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := woffs sh
  unfold wV2
  rw [swOf_get sh p h cl (by omega), swEl, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left rfl]

theorem w_sc {j k : ℕ} (hj : j < nR sh) (hk : k < deg sh j) :
    fOf (swOf sh p h cl)[wSc sh j + k]! = if j < sh.nu then pv (p.m3 j) k else pv (p.m4 j) k := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := woffs sh
  have hlt := degSum_add_lt sh hj hk
  have hr := roundOf_degSum sh hk
  unfold wSc
  rw [swOf_get sh p h cl (by omega), swEl, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_left (by omega), show 77 + degSum sh j + k - 77 = degSum sh j + k by omega, hr,
    show degSum sh j + k - degSum sh j = k by omega]

theorem w_u {q : ℕ} (hq : q < 2 ^ sh.pLog) : fOf (swOf sh p h cl)[wU sh + q]! = vget p.u q := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := woffs sh
  rw [swOf_get sh p h cl (by omega), swEl, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_left (by omega), Nat.add_sub_cancel_left]

theorem w_vt {t : ℕ} (ht : t < tw sh) : fOf (swOf sh p h cl)[wVt sh + t]! = vget p.vt t := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := woffs sh
  rw [swOf_get sh p h cl (by omega), swEl, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega), Nat.add_sub_cancel_left]

theorem w_claim : fOf (swOf sh p h cl)[wClaim sh]! = cl := by
  obtain ⟨o1, o2, o3, o4, o5⟩ := woffs sh
  rw [swOf_get sh p h cl (by omega), swEl, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]

/-! ## The witness's messages are the play's -/

theorem msgTape_congr {k : ℕ} {f g : ℕ → 𝔽 × (Fin k → 𝔽)} {a n : ℕ}
    (hfg : ∀ j, a ≤ j → j < a + n → f j = g j) : msgTape f a n = msgTape g a n := by
  unfold msgTape
  exact List.map_congr_left fun j hj => by rw [List.mem_range'_1] at hj; rw [hfg j hj.1 hj.2]

theorem msg2_swOf {j : ℕ} (hj : j < 6) : msg2 (swOf sh p h cl) j = p.m2 j :=
  Prod.ext (w_pre0 sh p h cl hj) (funext fun i => by
    obtain rfl : i = 0 := Subsingleton.elim i 0
    exact w_pre1 sh p h cl hj)

theorem msg3_lo {j : ℕ} (hj : j < sh.nu) : msg3 sh (swOf sh p h cl) (2 * sh.ell) j = p.m3 j := by
  have hj' : j < nR sh := by unfold nR; omega
  have hd : deg sh j = 2 * sh.ell + 1 := by unfold deg; exact ite_eq_left hj
  refine Prod.ext ?_ (funext fun i => ?_)
  · have e := w_sc sh p h cl hj' (k := 0) (by omega)
    rw [Nat.add_zero, ite_eq_left hj, pv_zero] at e
    exact e
  · have := i.isLt
    have e := w_sc sh p h cl hj' (k := i.val + 1) (by omega)
    rw [ite_eq_left hj, pv_succ] at e
    exact e

theorem msg3_hi {j : ℕ} (h₁ : sh.nu ≤ j) (h₂ : j < sh.nu + sh.sLog) :
    msg3 sh (swOf sh p h cl) (2 * sh.ell + 1) j = p.m4 j := by
  have hj' : j < nR sh := by unfold nR; omega
  have hd : deg sh j = 2 * sh.ell + 2 := by unfold deg; exact ite_eq_right (by omega)
  refine Prod.ext ?_ (funext fun i => ?_)
  · have e := w_sc sh p h cl hj' (k := 0) (by omega)
    rw [Nat.add_zero, ite_eq_right (by omega), pv_zero] at e
    exact e
  · have := i.isLt
    have e := w_sc sh p h cl hj' (k := i.val + 1) (by omega)
    rw [ite_eq_right (by omega), pv_succ] at e
    exact e

theorem uOf_swOf : uOf sh (swOf sh p h cl) = p.u :=
  funext fun q => (w_u sh p h cl q.isLt).trans (vget_fin _ _)

theorem vtTape_swOf : vtTape sh (swOf sh p h cl) = vtList sh p.vt := by
  unfold vtTape vtList
  by_cases ht : sh.tLog = 0
  · rw [ite_eq_left ht, ite_eq_left ht]
  · rw [ite_eq_right ht, ite_eq_right ht]
    have htw : tw sh = 2 ^ sh.tLog := by unfold tw; exact ite_eq_right ht
    have e : (fun t : Fin (2 ^ sh.tLog) => fOf (swOf sh p h cl)[wVt sh + t.val]!) = p.vt :=
      funext fun t => (w_vt sh p h cl (by rw [htw]; exact t.isLt)).trans (vget_fin _ _)
    rw [e]

theorem phaseMsgs_swOf : phaseMsgs sh (swOf sh p h cl) = pList sh p := by
  have e2 : msgTape (msg2 (swOf sh p h cl)) 0 6 = msgTape p.m2 0 6 :=
    msgTape_congr fun j _ hj => msg2_swOf sh p h cl (by omega)
  have e3 : msgTape (msg3 sh (swOf sh p h cl) (2 * sh.ell)) 0 sh.nu = msgTape p.m3 0 sh.nu :=
    msgTape_congr fun j _ hj => msg3_lo sh p h cl (by omega)
  have e4 : msgTape (msg3 sh (swOf sh p h cl) (2 * sh.ell + 1)) sh.nu sh.sLog = msgTape p.m4 sh.nu sh.sLog :=
    msgTape_congr fun j h₁ h₂ => msg3_hi sh p h cl h₁ h₂
  rw [phaseMsgs, pList, e2, e3, e4, uOf_swOf, vtTape_swOf, w_v2]

/-! ## The coins -/

theorem coinArr_one (x : Flock.F128) : coinArr fOf (#[x] : Array Flock.F128) = (⟨𝔽, fOf x⟩ : Msg) := by
  simp [coinArr]

theorem coins_ofFn (g : ℕ → 𝔽) (n : ℕ) :
    (coinTape (Array.ofFn (n := n) fun j => (g j.val : Flock.F128)) 0 n).map (coinArr fOf) =
      (List.range' 0 n).map fun j => (⟨𝔽, g j⟩ : Msg) := by
  unfold coinTape
  rw [List.map_map]
  refine List.map_congr_left fun j hj => ?_
  rw [List.mem_range'_1] at hj
  rw [Function.comp_apply, coinArr_one]
  exact congrArg (fun x => (⟨𝔽, fOf x⟩ : Msg))
    (arr_ofFn_get (α := Flock.F128) (fun j : Fin n => g j.val) (show j < n by omega))

/-- **Steps of the sparse phase's kinds are a unit's**: a witness of the message ports' size with `h` first and `cl`
at `claim`, and the coins one word each, whose `stepsOf` they are. -/
theorem stepsOf_of_kinds {l : List Step} (hk : l.map kindOf = sparseKinds 𝔽 sh) :
    ∃ w cs : Array Flock.F128, w.size = nw sh ∧ cs.size = nc sh ∧ (∀ s : Fin 64, fOf w[s.val]! = h s) ∧
      fOf w[wClaim sh]! = cl ∧ stepsOf sh w cs = l := by
  obtain ⟨p, hm⟩ := pmsgs_of_tys sh (ms := l.filterMap Sum.getLeft?) (by rw [msgs_fst, hk, sparseKinds_msgs])
  obtain ⟨g, hc⟩ := tape_of_fst (M := 𝔽) 0 (l.filterMap Sum.getRight?) 0 (nc sh)
    (by rw [coins_fst, hk, sparseKinds_coins])
  refine ⟨swOf sh p h cl, Array.ofFn (n := nc sh) fun j => (g j.val : Flock.F128), swOf_size sh p h cl,
    Array.size_ofFn, w_h sh p h cl, w_claim sh p h cl, ?_⟩
  rw [stepsOf, phaseMsgs_swOf, coins_ofFn, ← hm, ← hc, ← hk]
  exact weave_self l

end Layout

/-- Coins with the rep's public values `pub` and the phase's coins `cs`. -/
def scoinsOf (pub : SparsePub 𝔽 sh) (cs : Array Flock.F128) : SCoins :=
  ⟨pub.alpha, pub.z, Array.ofFn pub.xRest, Array.ofFn pub.yRest, Array.ofFn pub.U, cs⟩

theorem pubOf_scoinsOf (pub : SparsePub 𝔽 sh) (cs : Array Flock.F128) : pubOf sh (scoinsOf sh pub cs) = pub := by
  have e : ∀ {m : ℕ} (f : Fin m → 𝔽), (fun i : Fin m => fOf (Array.ofFn (α := Flock.F128) f)[i.val]!) = f :=
    fun f => funext fun i => congrArg fOf (arr_ofFn_get (α := Flock.F128) f i.isLt)
  obtain ⟨z, al, xr, yr, U⟩ := pub
  exact congr (congr (congrArg (SparsePub.mk z al) (e xr)) (e yr)) (e U)

/-! ## The unit -/

/-- **A play of the sparse phase to a claim is a RecSparse unit's steps**, with the phase's public values, slots'
types (below `T`) and `h`, at which the unit's verifier row accepts. -/
theorem unit_of_playsT {pub : SparsePub 𝔽 sh} {τ' : Fin (2 ^ sh.sLog) → ℕ} {h : Fin 64 → 𝔽} {l : List Step}
    {x : RegClaim 𝔽 sh} (hp : PlaysT (sparsePhase execArith sh pub τ' h) l (some x))
    (hτ : sh.tLog ≠ 0 → ∀ σ, τ' σ < 2 ^ sh.tLog) :
    ∃ w τ c, SparseUnit sh pub τ' h w τ c ∧ stepsOf sh w c.cs = l := by
  obtain ⟨w, cs, hw, hcs, hh, hcl, rfl⟩ :=
    stepsOf_of_kinds sh h x.value (Exact.kinds hp (sparsePhase_exact execArith sh pub τ' h))
  have hc : nc sh ≤ (scoinsOf sh pub cs).cs.size := Nat.le_of_eq hcs.symm
  have hτf : (fun σ : Fin (2 ^ sh.sLog) => (Array.ofFn τ')[σ.val]!) = τ' :=
    funext fun σ => arr_ofFn_get τ' σ.isLt
  have hhf : (fun s : Fin 64 => fOf w[s.val]!) = h := funext hh
  obtain ⟨o, ho, hpl, hiff⟩ := sparse_refines sh w (scoinsOf sh pub cs) (Array.ofFn τ') hw hc
  have hpl' := playsT_stepsOf hpl
  rw [pubOf_scoinsOf, hτf, hhf] at hpl'
  have hv := (playsT_prefix hp hpl' (List.prefix_refl _) (List.prefix_refl _)).2
  refine ⟨w, Array.ofFn τ', scoinsOf sh pub cs,
    ⟨hw, hc, pubOf_scoinsOf sh pub cs, fun σ => arr_ofFn_get τ' σ.isLt, hh, o, ho, hiff.2 ⟨?_, ?_⟩⟩, rfl⟩
  · intro ht σ hσ
    rw [arr_ofFn_get τ' hσ]
    exact hτ ht _
  · rw [hv, sparseClaim, hcl]
    exact congrArg some (sparseVerdict_shape hv)

end

end FlockC3.Sparse
