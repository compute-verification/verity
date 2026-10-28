import Proofs.Flock.Claims.Recursion.VStarBuild
import Proofs.Flock.Claims.Recursion.NextRCounts
import Proofs.Flock.Recursive.HoloSchedCheck
import Proofs.Flock.C3.Reps.RoundOfHolo
import Proofs.Flock.C3.Reps.RoundLinks
import Proofs.Flock.C3.Reps.AtList
import Proofs.Flock.VBridge.Family.Structure

/-!
# V\*'s inner parts' port names (`innerPorts_named`)

Every port of an inner part at the compiled statement (`partAt x s = some (.inner j nm pt, c)`) is `acc0`, `acc1`, or
`m<i>` at one of the rep schedule's message rounds (`i < (commitRounds rs).length`, `rs` the inner shape's `repSched`):

* **The part is one of V\*'s** (`inner_mem`): the statement list is `statementsOf`'s, whose inner statements are the parts
  of `parts (specOf vsh holo)`.
* **Its ports are the spec's** (`parts_ports`, `innerUnit_sizes`): `parts` gives each part ports `S.ports[p]!`, and the
  part's own `checkRepS` (in `checkParsed`'s unit, which `readX` ran) holds each port to at least one element, so `p` is in
  range (the default port has none).
* **The spec's names** (`specOf_names`): `spec`'s are `m0 … m(N−1)` then `acc0, acc1`, `N` the size of
  `Algebra.rounds`; `specHolo`'s are `m0 … mN` then the same two.
* **The count** (`vrounds_size`, `repRounds_countP`, `holoRepRounds_countP`): `repRounds` absorbs at `N` of its rounds
  (`m ≥ 22`), and `holoRepRounds` at one more.
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)
open FlockSoundness.Refine (ebind_ok)
open Flock.Guarantees.HoloRepCoins (obind_some)
open _root_.VStar.Algebra (Name)

/-! ## The size of `Algebra.rounds` -/

/-- `rounds`' state in its loops: the rounds so far, the message counter and the out-of-domain counter. -/
abbrev VSt := Array (Array Name) × ℕ × ℕ

/-- `rounds`' push of a message, at the message counter. -/
def vF1 (b : Array (Array Name) × ℕ) (_ : ℕ) : Array (Array Name) × ℕ :=
  (b.1.push (_root_.VStar.Algebra.msgNames b.2), b.2 + 1)

/-- `rounds`' push of an out-of-domain value with a message, at the message and out-of-domain counters. -/
def vF2 (b : VSt) (_ : ℕ) : VSt :=
  (b.1.push (#[Name.ood b.2.2] ++ _root_.VStar.Algebra.msgNames b.2.1), b.2.1 + 1, b.2.2 + 1)

/-- `rounds` before `T`: the zerocheck's, the lincheck's and the ring switches' rounds. -/
def vPre (sh : _root_.VStar.Algebra.Shape) : Array (Array Name) :=
  (List.range' 0 sh.claims).foldl (fun out k => out.push ((Array.range 128).map (Name.rs k)))
    ((((List.range' 0 (sh.kLog - 6)).foldl (fun out j => out.push #[Name.q1 j, Name.qinf j])
      (((List.range' 0 (sh.m - 6)).foldl (fun out i => out.push #[Name.g1 i, Name.ginf i])
        #[(Array.range 64).map Name.ab ++ (Array.range 64).map Name.c]).push #[Name.fa, Name.fb])).push
      ((Array.range 64).map Name.zp)))

/-- `rounds` before its level loop. -/
def v0 (s : Schedule) (pre : Array (Array Name)) : VSt :=
  let a1 := List.foldl vF1 ((pre.push #[Name.T]).push #[Name.ood 0], 0) (List.range' 0 (s.k0 + 1))
  let a := List.foldl vF2 (a1.1, a1.2, 1) (List.range' 0 2)
  (a.1.push (_root_.VStar.Algebra.msgNames a.2.1), a.2.1 + 1, a.2.2)

/-- A level of `rounds`' loop but the last. -/
def vMid (s : VSt) : VSt :=
  let a1 := List.foldl vF1 (s.1, s.2.1) (List.range' 0 4)
  let a := List.foldl vF2 (a1.1, a1.2, s.2.2) (List.range' 0 2)
  (a.1.push (_root_.VStar.Algebra.msgNames a.2.1), a.2.1 + 1, a.2.2)

/-- The last level of `rounds`' loop, with the final `yr`. -/
def vLast (ext : ℕ) (s : VSt) : VSt :=
  let a := List.foldl vF1 (s.1, s.2.1) (List.range' 0 3)
  (a.1.push (_root_.VStar.Algebra.msgNames a.2 ++ Array.map Name.yr (Array.range (2 <<< ext))), a.2 + 1, s.2.2)

/-- `rounds`' level loop over the levels `l`, of `R` after the first: `G1` at each but the last, `G2` there. -/
def vLevels (R : ℕ) (G1 G2 : VSt → VSt) (l : List ℕ) (init : VSt) : VSt :=
  Id.run (forIn l init fun i s => if i + 1 < R then pure (ForInStep.yield (G1 s)) else pure (ForInStep.yield (G2 s)))

theorem rounds_eq (sh : _root_.VStar.Algebra.Shape) (s : Schedule) : _root_.VStar.Algebra.rounds sh s =
    (vLevels (s.levels.size - 1) vMid (vLast s.ext) (List.range' 0 (s.levels.size - 1)) (v0 s (vPre sh))).1 := by
  unfold _root_.VStar.Algebra.rounds
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, List.forIn_pure_yield_eq_foldl,
    bind_pure_comp, map_pure, pure_bind, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  rfl

theorem vF1_size : ∀ (l : List ℕ) (b : Array (Array Name) × ℕ), (l.foldl vF1 b).1.size = b.1.size + l.length
  | [], _ => rfl
  | a :: l, b => by
    rw [List.foldl_cons, vF1_size l]
    simp only [vF1, Array.size_push, List.length_cons]
    omega

theorem vF2_size : ∀ (l : List ℕ) (b : VSt), (l.foldl vF2 b).1.size = b.1.size + l.length
  | [], _ => rfl
  | a :: l, b => by
    rw [List.foldl_cons, vF2_size l]
    simp only [vF2, Array.size_push, List.length_cons]
    omega

theorem foldl_push_size {α β : Type} (f : β → α) (l : List β) (init : Array α) :
    (l.foldl (fun out b => out.push (f b)) init).size = init.size + l.length := by
  rw [FlockVBridge.Family.foldl_push_map]
  simp

theorem vPre_size (sh : _root_.VStar.Algebra.Shape) :
    (vPre sh).size = (sh.m - 6) + (sh.kLog - 6) + sh.claims + 3 := by
  simp only [vPre, foldl_push_size, Array.size_push, List.length_range']
  simp
  omega

theorem v0_size (s : Schedule) (pre : Array (Array Name)) : (v0 s pre).1.size = pre.size + s.k0 + 6 := by
  simp only [v0, Array.size_push, vF2_size, vF1_size, List.length_range']
  omega

theorem vMid_size (s : VSt) : (vMid s).1.size = s.1.size + 7 := by
  simp only [vMid, Array.size_push, vF2_size, vF1_size, List.length_range']

theorem vLast_size (ext : ℕ) (s : VSt) : (vLast ext s).1.size = s.1.size + 4 := by
  simp only [vLast, Array.size_push, vF1_size, List.length_range']

theorem vLevels_foldl (R : ℕ) (G1 G2 : VSt → VSt) (l : List ℕ) (init : VSt) :
    vLevels R G1 G2 l init = l.foldl (fun s i => if i + 1 < R then G1 s else G2 s) init := by
  unfold vLevels
  have e : (fun i s => if i + 1 < R then pure (ForInStep.yield (G1 s)) else pure (ForInStep.yield (G2 s)) :
      ℕ → VSt → Id (ForInStep VSt)) = fun i s => pure (ForInStep.yield (if i + 1 < R then G1 s else G2 s)) := by
    funext i s
    split <;> rfl
  rw [e, List.forIn_pure_yield_eq_foldl]
  rfl

/-- A level's message rounds: seven at each but the last, four there. -/
def levelAbs (R : ℕ) : ℕ := ((List.range' 0 R).map fun i => if i + 1 < R then 7 else 4).sum

theorem foldl_levels_size (R : ℕ) (G1 G2 : VSt → VSt)
    (h1 : ∀ s, (G1 s).1.size = s.1.size + 7) (h2 : ∀ s, (G2 s).1.size = s.1.size + 4) :
    ∀ (l : List ℕ) (init : VSt), (l.foldl (fun s i => if i + 1 < R then G1 s else G2 s) init).1.size =
        init.1.size + (l.map fun i => if i + 1 < R then 7 else 4).sum
  | [], _ => by simp
  | i :: l, init => by
    rw [List.foldl_cons, foldl_levels_size R G1 G2 h1 h2 l, List.map_cons, List.sum_cons]
    by_cases hi : i + 1 < R
    · simp only [hi, ↓reduceIte, h1]
      omega
    · simp only [hi, ↓reduceIte, h2]
      omega

/-- **`Algebra.rounds`' size**: the zerocheck's `1 + (m − 6)`, the lincheck's `1 + (kLog − 6)` and `z_partial`'s, the ring
switches' `claims`, `T`'s and the first out-of-domain value's two, Ligerito's first level's `k₀ + 4`, and its later
levels'. -/
theorem vrounds_size (sh : _root_.VStar.Algebra.Shape) (s : Schedule) :
    (_root_.VStar.Algebra.rounds sh s).size =
      (sh.m - 6) + (sh.kLog - 6) + sh.claims + s.k0 + 9 + levelAbs (s.levels.size - 1) := by
  rw [rounds_eq, vLevels_foldl, foldl_levels_size _ vMid (vLast s.ext) vMid_size (vLast_size s.ext), v0_size,
    vPre_size, levelAbs]
  omega

/-! ## The rep schedule's message rounds -/

/-- A later level's rounds' absorbing flags past `repRounds`' level 0. -/
def lvAbs : List Bool := [true, true, false, true, false, false, true]

/-- Pass `i` of `repRounds`' level loop, of `r`: its four rounds, then the next level's or the last's three. -/
def passAbs (r i : ℕ) : List Bool := [false, true, true, true] ++ if i + 1 < r then lvAbs else [true, false, false]

theorem count_passAbs (r i : ℕ) : (passAbs r i).count true = if i + 1 < r then 7 else 4 := by
  unfold passAbs lvAbs
  split <;> rfl

theorem count_flatMap_passAbs (R : ℕ) : ∀ (n a : ℕ),
    ((List.range' a n).flatMap (passAbs R)).count true = ((List.range' a n).map fun i => if i + 1 < R then 7 else 4).sum
  | 0, _ => rfl
  | n + 1, a => by
    rw [List.range'_succ, List.flatMap_cons, List.count_append, count_passAbs, count_flatMap_passAbs R n (a + 1),
      List.map_cons, List.sum_cons]

theorem count_map_true {α : Type} (f : α → Bool) : ∀ l : List α, (l.map f).count true = l.countP f
  | [] => rfl
  | a :: l => by
    cases h : f a <;> simp [List.count_cons, List.countP_cons, h, count_map_true f l]

/-- **`repRounds` absorbs at `(m − 5) + (kLog − 6) + claims + k₀ + 8` rounds and its later levels'.** -/
theorem repRounds_countP {m kLog claims : ℕ} {sch : Schedule} {rs : Array RecVStar.SchedRound}
    (hs : fast100 m = some sch) (h : RecVStar.repRounds m kLog claims = some rs) :
    rs.toList.countP (·.2.2.2) = (m - 5) + (kLog - 6) + claims + sch.k0 + 8 + levelAbs (sch.levels.size - 1) := by
  unfold RecVStar.repRounds at h
  simp only [hs, Std.Legacy.Range.forIn_eq_forIn_range'] at h
  simp only [Option.bind_eq_bind, Option.pure_def, Option.bind_some] at h
  simp only [Option.bind_eq_some_iff] at h
  obtain ⟨s, hl, he⟩ := h
  cases he
  have key := FlockDeploy.forIn_opt_inv _ (fun st : Array RecVStar.SchedRound × ℕ => st.1.toList.map (·.2.2.2))
    (passAbs (sch.levels.size - 1)) (fun _ _ => True)
    (fun i s t _ ht => by
      by_cases h4 : s.2 < 4
      · simp [h4] at ht
      · by_cases hi : i + 1 < sch.levels.size - 1
        · simp [h4, hi] at ht
          exact ⟨_, ht.symm, by simp [passAbs, lvAbs, hi, FlockDeploy.rep3], trivial⟩
        · simp [h4, hi] at ht
          exact ⟨_, ht.symm, by simp [passAbs, hi, FlockDeploy.rep3], trivial⟩) _ _ _ _ trivial hl
  rw [← count_map_true, key, List.count_append]
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  rw [show ((List.range' 0 (sch.levels.size - 1)).flatMap (passAbs (sch.levels.size - 1))).count true =
      levelAbs (sch.levels.size - 1) from count_flatMap_passAbs _ _ 0]
  simp [List.count_replicate] <;> omega

theorem countP_splice {α : Type} (p : α → Bool) (l : List α) (k : ℕ) (x a b : α) (hx : l[k]? = some x)
    (hpx : p x = true) (ha : p a = true) (hb : p b = true) :
    (l.take k ++ [a, b] ++ l.drop (k + 1)).countP p = l.countP p + 1 := by
  obtain ⟨hk, rfl⟩ := List.getElem?_eq_some_iff.1 hx
  conv_rhs => rw [← List.take_append_drop k l, List.drop_eq_getElem_cons hk]
  simp only [List.countP_append, List.countP_cons, List.countP_nil, hpx, ha, hb, ↓reduceIte]
  omega

/-- **`holoRepRounds` absorbs at one round more than `repRounds`**: it puts two absorbing rounds for `z_partial`'s. -/
theorem holoRepRounds_countP {m kLog claims : ℕ} {rs : Array RecVStar.SchedRound}
    (h : RecVStar.holoRepRounds m kLog claims = some rs) :
    ∃ out, RecVStar.repRounds m kLog claims = some out ∧
      rs.toList.countP (·.2.2.2) = out.toList.countP (·.2.2.2) + 1 := by
  unfold RecVStar.holoRepRounds at h
  obtain ⟨out, hout, h⟩ := obind_some h
  refine ⟨out, hout, ?_⟩
  try dsimp only at h
  split at h
  · cases h
  · rename_i hcond
    have hk : out.toList[m + kLog - 7]? = some ((1, #[], 0, true) : RecVStar.SchedRound) := by
      rw [Array.getElem?_toList]
      simpa using hcond
    cases h
    simp only [Array.toList_append, Array.toList_extract, List.extract_eq_take_drop, List.drop_zero, Nat.sub_zero]
    rw [List.take_of_length_le (l := out.toList.drop (m + kLog - 7 + 1)) (by simp)]
    exact countP_splice (fun q : RecVStar.SchedRound => q.2.2.2) out.toList (m + kLog - 7) _ (0, #[], 0, true)
      (1, #[], 0, true) hk rfl rfl rfl

theorem commitRounds_length (rs : Array RecVStar.SchedRound) :
    (RecVStar.commitRounds rs).length = rs.toList.countP (·.2.2.2) := by
  rcases rs with ⟨l⟩
  rw [← Flock.Guarantees.HoloSchedCheck.msgRounds_eq]
  exact (congrArg List.length (FlockC3.Reps.RoundOfHolo.msgRounds_toArray l)).trans
    (FlockC3.Reps.RoundOfHolo.msgL_length l)

/-! ## The spec's port names -/

theorem ctx_rounds {vsh : _root_.VStar.Algebra.Shape} {cx : _root_.VStar.Algebra.Ctx}
    (h : _root_.VStar.Algebra.Ctx.make vsh = .ok cx) :
    ∃ s, vsh.sched = .ok s ∧ cx.rounds = _root_.VStar.Algebra.rounds vsh s := by
  unfold _root_.VStar.Algebra.Ctx.make at h
  obtain ⟨s, hs, h⟩ := ebind_ok h
  refine ⟨s, hs, ?_⟩
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rfl

/-- **`spec`'s port names**: `m<i>` for each of `rounds`' rounds, then the running sums' two. -/
theorem spec_names {vsh : _root_.VStar.Algebra.Shape} {sp : _root_.VStar.RepS}
    (h : _root_.VStar.Algebra.spec vsh = .ok sp) :
    ∃ cx, _root_.VStar.Algebra.Ctx.make vsh = .ok cx ∧
      sp.ports.map Prod.fst = (List.range' 0 cx.rounds.size).map (fun i => s!"m{i}") ++ ["acc0", "acc1"] := by
  unfold _root_.VStar.Algebra.spec at h
  dsimp only at h
  obtain ⟨cx, hcx, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨ports, hports, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  refine ⟨cx, hcx, ?_⟩
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind, bind_pure] at hports
  simp only [pure, Except.pure, Except.ok.injEq] at hports
  subst hports
  rw [FlockVBridge.Family.foldl_push_map]
  simp [Std.Legacy.Range.size]

/-- **`specHolo`'s port names**: `m<i>` for `i` up to the plain spec's round count, then the plain spec's last two. -/
theorem specHolo_names {vsh : _root_.VStar.Algebra.Shape} {sp : _root_.VStar.RepS}
    (h : _root_.VStar.Algebra.specHolo vsh = .ok sp) :
    ∃ S, _root_.VStar.Algebra.spec vsh = .ok S ∧ ∀ n ∈ sp.ports.map Prod.fst,
      (∃ i, i < S.ports.length - 2 + 1 ∧ n = s!"m{i}") ∨
        n ∈ (S.ports.drop (S.ports.length - 2)).map Prod.fst := by
  unfold _root_.VStar.Algebra.specHolo at h
  obtain ⟨S, hS, h⟩ := ebind_ok h
  refine ⟨S, hS, ?_⟩
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h <;>
    · intro n hn
      rw [List.map_append, List.mem_append] at hn
      rcases hn with hn | hn
      · left
        rw [List.map_map, List.mem_map] at hn
        obtain ⟨⟨a, i⟩, hai, rfl⟩ := hn
        refine ⟨i, ?_, rfl⟩
        have := (List.mem_zipIdx hai).2.1
        simp only [List.length_append, List.length_map, List.length_take, List.length_drop, List.length_singleton,
          Nat.zero_add] at this
        omega
      · exact .inr hn

/-- **`specOf`'s port names**: `acc0`, `acc1`, and `m<i>` below `rounds`' size, one more over a holo session. -/
theorem specOf_names {vsh : _root_.VStar.Algebra.Shape} {holo : Bool} {sp : _root_.VStar.RepS}
    (h : _root_.VStar.Algebra.specOf vsh holo = .ok sp) :
    ∃ s, vsh.sched = .ok s ∧ ∀ n ∈ sp.ports.map Prod.fst, n = "acc0" ∨ n = "acc1" ∨
      ∃ i, n = s!"m{i}" ∧ i < (_root_.VStar.Algebra.rounds vsh s).size + (if holo then 1 else 0) := by
  cases holo
  · change _root_.VStar.Algebra.spec vsh = .ok sp at h
    obtain ⟨cx, hcx, hn⟩ := spec_names h
    obtain ⟨s, hs, hr⟩ := ctx_rounds hcx
    refine ⟨s, hs, fun n hn' => ?_⟩
    rw [hn] at hn'
    rcases List.mem_append.1 hn' with h1 | h1
    · obtain ⟨i, hi, rfl⟩ := List.mem_map.1 h1
      refine .inr (.inr ⟨i, rfl, ?_⟩)
      rw [← hr, show (if false = true then 1 else 0) = 0 from rfl]
      simp only [List.mem_range'_1] at hi
      omega
    · simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at h1
      rcases h1 with h1 | h1
      · exact .inl h1
      · exact .inr (.inl h1)
  · change _root_.VStar.Algebra.specHolo vsh = .ok sp at h
    obtain ⟨S, hS, hn⟩ := specHolo_names h
    obtain ⟨cx, hcx, hSn⟩ := spec_names hS
    obtain ⟨s, hs, hr⟩ := ctx_rounds hcx
    have hlen : S.ports.length = cx.rounds.size + 2 := by
      have := congrArg List.length hSn
      simpa using this
    refine ⟨s, hs, fun n hn' => ?_⟩
    rcases hn n hn' with ⟨i, hi, rfl⟩ | h1
    · refine .inr (.inr ⟨i, rfl, ?_⟩)
      rw [← hr, show (if true = true then 1 else 0) = 1 from rfl]
      omega
    · rw [List.map_drop, hSn, hlen,
        show cx.rounds.size + 2 - 2 = ((List.range' 0 cx.rounds.size).map fun i => s!"m{i}").length by simp,
        List.drop_left] at h1
      simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at h1
      rcases h1 with h1 | h1
      · exact .inl h1
      · exact .inr (.inl h1)

/-! ## A part's ports are its spec's -/

theorem forIn_pushes_all {ε α β : Type} (P : β → Prop) (f : α → Array β → Except ε (ForInStep (Array β)))
    (hf : ∀ a r y, f a r = .ok y → ∃ x, y = .yield (r.push x) ∧ P x) :
    ∀ (L : List α) (init v : Array β), (∀ x ∈ init, P x) → forIn L init f = .ok v → ∀ x ∈ v, P x
  | [], init, v, hi, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact hi
  | a :: L, init, v, hi, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := ebind_ok h
    obtain ⟨x, rfl, hx⟩ := hf a init y hy
    refine forIn_pushes_all P f hf L _ v (fun z hz => ?_) h
    rcases Array.mem_push.1 hz with hz | rfl
    · exact hi z hz
    · exact hx

/-- **Each part's ports are its S's** (`parts`, `Parts.lean:100`): part `(R, P, V)`'s are `S.ports[p]!` for `p ∈ P`. -/
theorem parts_ports {S : _root_.VStar.RepS} {ps : Array _root_.VStar.Part} (h : _root_.VStar.parts S = .ok ps) :
    ∀ pt ∈ ps, ∀ q ∈ pt.S.ports, ∃ p : ℕ, q = S.ports.toArray[p]! := by
  unfold _root_.VStar.parts at h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨out, hout, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rw [← Array.forIn_toList] at hout
  refine forIn_pushes_all (fun pt : _root_.VStar.Part => ∀ q ∈ pt.S.ports, ∃ p : ℕ, q = S.ports.toArray[p]!) _
    (fun a r y hy => ?_) _ _ _ (by simp) hout
  repeat' (first | (obtain ⟨_, _, hy⟩ := ebind_ok hy) | (split at hy))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    exact ⟨_, hy.symm, fun q hq => by
      obtain ⟨p, -, rfl⟩ := List.mem_map.1 hq
      exact ⟨p, rfl⟩⟩

theorem checkRepS_sizes {S : _root_.VStar.RepS} {whole : Bool} {u : Unit}
    (h : _root_.VStar.checkRepS S whole = .ok u) : ∀ q ∈ S.ports, 1 ≤ q.2 := by
  intro q hq
  by_contra hlt
  have hany : S.ports.any (fun p => decide (p.2 < 1)) = true := List.any_eq_true.2 ⟨q, hq, by simpa using hlt⟩
  unfold _root_.VStar.checkRepS at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := ebind_ok h
    first | cases h1 | simp [throw, throwThe, MonadExceptOf.throw] at h1
  · rename_i hc
    exact hc (by rw [hany]; simp)

theorem innerUnit_sizes {S : _root_.VStar.RepS} {name : String} {u} (h : _root_.VStar.innerUnit S name = .ok u) :
    ∀ q ∈ S.ports, 1 ≤ q.2 := by
  unfold _root_.VStar.innerUnit at h
  obtain ⟨_, h1, -⟩ := ebind_ok h
  exact checkRepS_sizes h1

theorem checkParsed_unit {s : _root_.VStar.Stmt} {c : Circuit} (h : (RecVStar.checkParsed s c).isOk = true) :
    ∃ u, s.unit = .ok u := by
  cases hu : s.unit with
  | ok u => exact ⟨u, rfl⟩
  | error e =>
    unfold RecVStar.checkParsed at h
    rw [hu] at h
    cases h

/-- **A statement of the list tagged `.inner` holds one of V\*'s parts**: one of `parts (specOf vsh holo)`. -/
theorem inner_mem {i : RecVStar.Inner} {vs : Array _root_.VStar.Stmt} (h : RecVStar.statements i = .ok vs) {s j : ℕ}
    {nm : String} {pt : _root_.VStar.Part} (hs : vs[s]? = some (.inner j nm pt)) :
    ∃ sp ps, _root_.VStar.Algebra.specOf (FlockC3.Reps.vshOf i) i.holo = .ok sp ∧ _root_.VStar.parts sp = .ok ps ∧
      pt ∈ ps := by
  obtain ⟨-, -, -, -, -, -, A, C, -, -, -, -, -, hA, hC, K, n, hK, -, rfl⟩ := FlockC3.Reps.list_ok h
  obtain ⟨sch, sp, ps, -, hsp, hps, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
  refine ⟨sp, ps, hsp, hps, ?_⟩
  have hm := Array.mem_of_getElem? hs
  rcases Array.mem_append.1 hm with hm | hm
  · rcases Array.mem_append.1 hm with hm | hm
    · rcases Array.mem_append.1 hm with hm | hm
      · simp at hm
      · simp only [List.mem_toArray, List.mem_map, List.mem_range] at hm
        obtain ⟨j', hj', he⟩ := hm
        simp only [_root_.VStar.Stmt.inner.injEq] at he
        rw [← he.2.2, getElem!_pos ps j' hj']
        exact Array.getElem_mem hj'
    · rcases FlockC3.Reps.candidate3_kinds hC _ hm with ⟨_, _, he⟩ | ⟨_, _, _, _, he⟩ |
        ⟨_, _, _, _, _, _, he⟩ <;> cases he
  · obtain ⟨_, _, he⟩ := FlockC3.VStarList.caps_cap hK n _ hm
    cases he

/-! ## The names -/

/-- **Every port of an inner part at the compiled statement is `acc0`, `acc1`, or a message round's `m<i>`**: `i` below
the count of the rep schedule's message rounds (`commitRounds`), at the inner shape's `repSched`. -/
theorem innerPorts_named {x : Msg} {s j : ℕ} {nm : String} {pt : _root_.VStar.Part} {c : Circuit}
    (hp : partAt x s = some (.inner j nm pt, c)) {r : XR} (hr : readX x = some r)
    {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) :
    ∀ n ∈ pt.S.ports.map Prod.fst, n = "acc0" ∨ n = "acc1" ∨
      ∃ i, RecVStar.mIndex n = some i ∧ i < (RecVStar.commitRounds rs).length := by
  obtain ⟨vs, hvs, hpv⟩ := readX_parts hr
  have hst : vs[s]? = some (.inner j nm pt) := by
    unfold partAt at hp
    obtain ⟨r', hr', hp⟩ := obind_some hp
    rw [hr] at hr'
    cases hr'
    obtain ⟨st, hst, hp⟩ := obind_some hp
    obtain ⟨c', -, hp⟩ := obind_some hp
    cases hp
    rw [hvs] at hst
    exact hst
  obtain ⟨hlt, -⟩ := Array.getElem?_eq_some_iff.1 hst
  unfold partsVStar at hpv
  have hall := List.all_eq_true.1 hpv s (List.mem_range.2 hlt)
  obtain ⟨u, hu⟩ : ∃ u, (_root_.VStar.Stmt.inner j nm pt).unit = .ok u := by
    cases hq : r.w.parts[s]? with
    | none => simp [hst, hq] at hall
    | some p =>
      cases hb : ownBindings p.bindings with
      | none => simp [hst, hq, hb] at hall
      | some f =>
        simp [hst, hq, hb] at hall
        unfold partVStar at hall
        split at hall
        · simp only [Bool.and_eq_true] at hall
          exact checkParsed_unit hall.1.1.1.2
        · cases hall
  have hsz : ∀ q ∈ pt.S.ports, 1 ≤ q.2 := innerUnit_sizes (show _root_.VStar.innerUnit pt.S nm = .ok u from hu)
  obtain ⟨sp, ps, hsp, hps, hpt⟩ := inner_mem hvs hst
  obtain ⟨sch, hsch, hnames⟩ := specOf_names hsp
  have hf : fast100 r.shape.m = some sch := FlockVBridge.Family.sched_fast100 hsch
  have h22 := (FlockVBridge.Family.fast100_range hf).1
  intro n hn
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hn
  obtain ⟨p, rfl⟩ := parts_ports hps pt hpt q hq
  have h1 := hsz _ hq
  have hpl : p < sp.ports.length := by
    by_contra hge
    rw [getElem!_neg _ _ (by simpa using hge)] at h1
    exact absurd h1 (by decide)
  have hmem : (sp.ports.toArray[p]!).1 ∈ sp.ports.map Prod.fst := by
    rw [getElem!_pos _ _ (by simpa using hpl)]
    exact List.mem_map.2 ⟨_, by simp, rfl⟩
  rcases hnames _ hmem with e | e | ⟨i, e, hi⟩
  · exact .inl e
  · exact .inr (.inl e)
  · refine .inr (.inr ⟨i, by rw [e]; exact FlockC3.Reps.mIndex_m i, ?_⟩)
    rw [vrounds_size] at hi
    dsimp only [FlockC3.Reps.vshOf] at hi
    rw [commitRounds_length]
    unfold repSched at hrs
    cases hh : r.shape.holo
    · rw [hh, show (if false = true then 1 else 0) = 0 from rfl] at hi
      rw [hh] at hrs
      change RecVStar.repRounds _ _ _ = some rs at hrs
      rw [repRounds_countP hf hrs]
      omega
    · rw [hh, show (if true = true then 1 else 0) = 1 from rfl] at hi
      rw [hh] at hrs
      change RecVStar.holoRepRounds _ _ _ = some rs at hrs
      obtain ⟨out, hout, hc⟩ := holoRepRounds_countP hrs
      rw [hc, repRounds_countP hf hout]
      omega

end RecDeploy.VStarBuild
