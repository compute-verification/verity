import Proofs.Flock.Claims.Recursion.RestRead
import Proofs.Flock.Claims.Recursion.TopsOpens
import Proofs.Flock.Claims.Recursion.RowsCover
import Proofs.Flock.Claims.Recursion.CapPorts

/-!
# Each round before the last has its length at every witnesses (`RoundLens`)

`hrest_len` takes `RoundLens fin`: at a dense compiled statement and witnesses one a part, round `j < k - 1`'s message
off the witnesses (`(layAt fin).msgs`) has `128 · nAtSched j` bytes. It holds at every `fin` (`roundLens`), since
every round before the last is fixed, held or assembled at lengths the statement fixes:

* **Round 0** is RecCap's part's, its `N` instances' commit strings (`capHold`), `N = 2^cap₀` (`capStatement`), which
  is `level0Nodes` of the one inner table (`capAt_level0`).
* **A stream round with no tops** is the empty message where it absorbs none, and where it absorbs one, a part's port's
  hm96 output (one 128-byte string), and some part holds it (`rowsCoverAll`).
* **A stream round with tops** is assembled from its tops at their caps (`128 · n` bytes each, `pieceLen_topPiece`) and
  its row, which is there exactly where the round absorbs a message (`rowsCoverAll`, `heldRows_absorbs`).

Each count is the schedule's (`nAtSched`): the deployed plain verifier sets the inner statement up as the one table
whose rep schedule is the layout's (`schedBridge`).
-/

namespace RecDeploy.RestRead

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB

/-! ## The deployed verifier's schedule is the layout's -/

theorem sched_fast100' {vsh : VStar.Algebra.Shape} {s : _root_.Flock.Schedule} (h : vsh.sched = .ok s) :
    _root_.Flock.fast100 vsh.m = some s := by
  unfold VStar.Algebra.Shape.sched at h
  cases hf : _root_.Flock.fast100 vsh.m with
  | none => simp [hf] at h
  | some s' =>
    simp only [hf] at h
    split at h
    · cases h
    · split at h
      · cases h
      · cases h; rfl

/-- **RecCap's statement counts `2^cap₀` nodes**, `cap₀` the cap of `fast100(m)`'s level 0. -/
theorem capStatement_eq {vsh : VStar.Algebra.Shape} {K : VStar.Stmt} (h : VStar.capStatement vsh = .ok K) :
    K = .cap 0 (2 ^ ((_root_.Flock.fast100 vsh.m).map fun s => (s.levels.getD 0 default).cap).getD 0) := by
  unfold VStar.capStatement at h
  obtain ⟨s, hs, h⟩ := FlockC3.VStarList.eok_bind h
  rw [sched_fast100' hs]
  split at h
  · rename_i lv hlv
    cases h
    simp [hlv]
  · cases h

/-- **V\*'s list ends with RecCap over `2^cap₀` nodes** at the inner shape's `m`. -/
theorem statements_cap {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    ss.back? = some (.cap 0 (2 ^ ((_root_.Flock.fast100 i.m).map fun s => (s.levels.getD 0 default).cap).getD 0)) := by
  unfold RecVStar.statements at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  split at h
  · cases h
  obtain ⟨ps, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨gs, -, h⟩ := FlockC3.VStarList.eok_bind h
  split at h
  · cases h
  split at h
  · cases h
  obtain ⟨K, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨t, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨A, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨C, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨K0, hK0, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u', -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u'', -, h⟩ := FlockC3.VStarList.eok_bind h
  cases h
  rw [capStatement_eq hK0]
  simp

/-- **The schedule at a dense compiled statement**: `readX` reads it at a shape with rep schedule `rs`, the deployed
plain verifier sets the inner statement up as one table `st` at the shape's `m` whose rep schedule is `rs`
(`repSchedAt`), and it takes `2 + 2R` rounds (as `schedMatches_dense`). -/
theorem schedBridge {x : Msg} (hx : DenseX x) :
    ∃ r rs s spec st, readX x = some r ∧ repSched r.shape = some rs ∧
      FlockDeploy.setupOf false (xIOf x) = some (s, spec, #[st]) ∧ st.m = r.shape.m ∧
      repSchedAt spec st = rs.toList ∧ kI x = 2 + 2 * rs.size := by
  obtain ⟨st0, hst, hf⟩ := hx
  obtain ⟨r, hr, ha, rfl⟩ := parse_read hst
  have hh : r.shape.holo = false := hf.1
  obtain ⟨f, spec', st', s, spec, st, hfs, hstm, hset, hm, hk, hreg, hpts, -, -⟩ := innerSetup_files hr ha
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, ⟨u, hsh⟩⟩, -⟩ := ha
  have hss : RecVStar.statements r.shape = .ok ss := hss
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  rw [hfs] at hsh
  obtain ⟨spec'', st'', hstm', e1, e2, -, e4, -⟩ := checkShape_files hsh
  rw [hstm] at hstm'
  cases hstm'
  obtain ⟨-, -, hk6, -, -⟩ := RecScope.statements_ok hss
  obtain ⟨rs, hrs⟩ := repSched_of_statements hss
  have hrs' : RecVStar.repRounds r.shape.m r.shape.kLog r.shape.claims = some rs := by
    unfold repSched at hrs
    rw [hh] at hrs
    exact hrs
  have hx : xIOf x = r.w.inner := by unfold xIOf; rw [hst]; rfl
  have hm' : r.shape.m = st.m := e1.trans hm.symm
  have hk' : r.shape.kLog = st.c.kLog := e2.trans hk.symm
  have hc' : r.shape.claims = 2 + spec.points * st.regions.size := by rw [e4, hpts, hreg]
  refine ⟨r, rs, s, spec, st, hr, hrs, by rw [hx]; exact hset, hm'.symm, ?_, ?_⟩
  · rw [hm', hk', hc'] at hrs'
    simp only [repSchedAt, hrs', Option.map_some, Option.getD_some]
  · show (FlockDeploy.V false).k (xIOf x) = _
    rw [hx]
    exact k_plain hset hm' hk' hc' (by omega) hrs'

theorem getElem?_twice {α : Type} [Inhabited α] (a : Array α) {i : ℕ} (hi : i < 2 * a.size) :
    (a.toList ++ a.toList)[i]? = some (a.getD (i % a.size) default) := by
  have hd : ∀ j < a.size, a[j]? = some (a.getD j default) := fun j hj => by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some]
  by_cases h : i < a.size
  · rw [List.getElem?_append_left (by simpa using h), Nat.mod_eq_of_lt h, Array.getElem?_toList, hd i h]
  · rw [List.getElem?_append_right (by simp only [Array.length_toList]; omega), Array.length_toList,
      Array.getElem?_toList, Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega : i - a.size < a.size),
      hd _ (by omega)]

/-- Round 0's count: the one table's level-0 nodes. -/
theorem nAtSched_zero {xI : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {st : _root_.Flock.Stmt}
    (hset : FlockDeploy.setupOf false xI = some (s, spec, #[st])) : nAtSched xI 0 = level0Nodes st := by
  simp [nAtSched, hset]

/-- Stream round `i`'s count: the rep schedule's round `i % R`'s tops' nodes, and one where it absorbs a message. -/
theorem nAtSched_succ {xI : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {st : _root_.Flock.Stmt}
    {rs : Array RecVStar.SchedRound} (hset : FlockDeploy.setupOf false xI = some (s, spec, #[st]))
    (hsc : repSchedAt spec st = rs.toList) {i : ℕ} (hi : i < 2 * rs.size) :
    nAtSched xI (i + 1) = (rs.getD (i % rs.size) default).2.1.toList.sum +
      if (rs.getD (i % rs.size) default).2.2.2 then 1 else 0 := by
  have e : (#[st].toList.flatMap fun st => repSchedAt spec st ++ repSchedAt spec st)[i]? =
      some (rs.getD (i % rs.size) default) := by
    show (repSchedAt spec st ++ repSchedAt spec st ++ [])[i]? = _
    rw [List.append_nil, hsc]
    exact getElem?_twice rs hi
  unfold nAtSched
  simp only [hset, Nat.add_one_ne_zero, ↓reduceIte, Nat.add_sub_cancel, e]

/-- **RecCap's part is the last, over the one table's level-0 nodes.** -/
theorem capAt_level0 {x : Msg} {r : XR} (hr : readX x = some r) {st : _root_.Flock.Stmt} (hm : st.m = r.shape.m) :
    capAt r = some (nS x - 1, level0Nodes st) := by
  obtain ⟨vs, hvs, hlen⟩ := readX_statements hr
  have hb := statements_cap hvs
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  unfold capAt
  rw [hvs]
  simp only [Except.toOption, Option.bind_some, hb, hn, hlen, level0Nodes, hm]

/-! ## The pieces' lengths -/

theorem sum_map_mul128 (l : List ℕ) (f : ℕ → ℕ) : (l.map fun e => 128 * f e).sum = 128 * (l.map f).sum := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [List.sum_cons, ih, Nat.mul_add]

theorem range_map_getD (a : Array ℕ) : (List.range a.size).map (fun e => a.getD e 0) = a.toList := by
  apply List.ext_getElem
  · simp
  · intro n _ h2
    simp only [Array.length_toList] at h2
    simp [h2]

theorem pieceLen_commits (c : Circuit) (q N : ℕ) :
    pieceLen ((List.range N).map fun n => (([] : Msg), hmCommit c q n)) = 128 * N := by
  simp only [pieceLen, List.map_map, Function.comp_def, hmCommit_length, List.length_nil]
  rw [sum_map_const, List.length_range]
  omega

theorem capHold_len {r : XR} {s i N : ℕ} {segs : List (Msg × List HmNets.Lin)} (hc : capAt r = some (i, N))
    (h : capHold r s = some segs) : pieceLen segs = 128 * N := by
  unfold capHold at h
  rw [hc, Option.bind_some] at h
  dsimp only at h
  split_ifs at h
  · obtain ⟨c, -, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨q, -, rfl⟩ := Option.map_eq_some_iff.1 h
    exact pieceLen_commits c q N

theorem capHold_at {x : Msg} {r : XR} (hr : readX x = some r) (hx : DenseX x) {N : ℕ}
    (hc : capAt r = some (nS x - 1, N)) : ∃ segs, capHold r (nS x - 1) = some segs := by
  obtain ⟨st, hst, hd⟩ := hx
  obtain ⟨c, hc', hq⟩ := capPorts Claims.Recursion.Form.Dense x st hst hd r hr
  obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 hq
  simp [capHold, hc, hc', hq]

theorem findSome_getD_length {α : Type} {f : α → Option Msg} {L : ℕ} (hf : ∀ a b, f a = some b → b.length = L)
    (l : List α) (h : ∃ a ∈ l, (f a).isSome) : ((l.findSome? f).getD []).length = L := by
  induction l with
  | nil =>
    obtain ⟨_, ha, _⟩ := h
    simp at ha
  | cons a l ih =>
    cases hfa : f a with
    | some v =>
      simp only [List.findSome?_cons, hfa, Option.getD_some]
      exact hf a v hfa
    | none =>
      simp only [List.findSome?_cons, hfa]
      obtain ⟨b, hb, hfb⟩ := h
      rcases List.mem_cons.1 hb with rfl | hb
      · simp [hfa] at hfb
      · exact ih ⟨b, hb, hfb⟩

theorem assembled_length (L : LayoutB) (x : Msg) (ws : List Claims.Flock.Wit) (i : ℕ) :
    (L.assembled x ws i).length = ((L.pieceAt x i).map fun p => pieceLen p.2).sum := by
  unfold LayoutB.assembled
  rw [List.length_flatMap]
  congr 1
  apply List.map_congr_left
  intro q _
  cases ws[q.1]? with
  | none => simp
  | some w => simp [readSegs_length]

/-- A held row's message is one hm96 output, 128 bytes. -/
theorem msgAt_succ_len {fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {x : Msg}
    {s i : ℕ} {segs : List (Msg × List HmNets.Lin)} (h : (layAt fin).msgAt x s (i + 1) = some segs) :
    pieceLen segs = 128 := by
  change ((streamRound x i).bind fun sr => if sr.2.1.isEmpty then if sr.2.2.2 then
    (readX x).bind fun r => (r.w.parts[s]?.bind partCircuit).bind fun c =>
      ((heldRows r s).find? (·.1 == i)).map fun h => [(([] : Msg), hmCommit c h.2.2.1 h.2.2.2)]
    else none else none) = some segs at h
  obtain ⟨sr, -, h⟩ := Option.bind_eq_some_iff.1 h
  split_ifs at h
  all_goals try (exact absurd h (by simp))
  obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨c, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨hh, -, rfl⟩ := Option.map_eq_some_iff.1 h
  simp [pieceLen, hmCommit_length]

/-! ## The rows -/

theorem heldRows_circuit {r : XR} {s : ℕ} {h : ℕ × String × ℕ × ℕ} (hh : h ∈ heldRows r s) :
    ∃ c, r.w.parts[s]?.bind partCircuit = some c := by
  cases hc : r.w.parts[s]?.bind partCircuit with
  | some c => exact ⟨c, rfl⟩
  | none =>
    unfold heldRows at hh
    rw [hc] at hh
    split at hh <;> simp_all

/-- **A held row is of a round that absorbs a message**: its stream round `j` is rep `j / R`'s round `j % R`, among
the schedule's message rounds (`RecVStar.commitRounds`). -/
theorem heldRows_absorbs {r : XR} {s : ℕ} {h : ℕ × String × ℕ × ℕ} (hh : h ∈ heldRows r s)
    {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) :
    (rs.getD (h.1 % rs.size) default).2.2.2 = true := by
  obtain ⟨c, hc⟩ := heldRows_circuit hh
  unfold heldRows at hh
  rw [hrs, hc] at hh
  cases hv : (RecVStar.statements r.shape).toOption.bind (fun vs => vs[s]?) with
  | none => simp [hv] at hh
  | some st =>
    simp only [hv] at hh
    obtain ⟨q, -, hq⟩ := List.mem_flatMap.1 hh
    split at hq
    · obtain ⟨row, -, hrow⟩ := List.mem_filterMap.1 hq
      simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hrow
      obtain ⟨rep, -, j, hj, qi, -, rfl⟩ := hrow
      unfold msgRound at hj
      obtain ⟨t, ht, rfl⟩ := Option.map_eq_some_iff.1 hj
      have hmem := List.mem_of_getElem? ht
      simp only [RecVStar.commitRounds, List.mem_filter, List.mem_range] at hmem
      obtain ⟨hlt, hab⟩ := hmem
      show (rs.getD ((rep * rs.size + t) % rs.size) default).2.2.2 = true
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hlt]
      exact hab
    · simp at hq

/-- **Every absorbing stream round is a part's row** (`rowsCover`, at a round with or without tops). -/
theorem rowsCoverAll {x : Msg} {r : XR} {j : ℕ} {sr : RecVStar.SchedRound} (hr : readX x = some r)
    (hsr : streamRound x j = some sr) (hA : sr.2.2.2 = true) : ∃ s < nS x, ∃ h ∈ heldRows r s, h.1 = j := by
  obtain ⟨vs, hvs, hlen, hpn, -⟩ := readX_listed hr
  unfold streamRound at hsr
  rw [hr, Option.bind_some] at hsr
  cases hrs : repSched r.shape with
  | none =>
    rw [hrs] at hsr
    cases hsr
  | some rs =>
    rw [hrs, Option.bind_some] at hsr
    by_cases hj : j < 2 * rs.size
    swap
    · rw [if_neg hj] at hsr
      cases hsr
    rw [if_pos hj] at hsr
    obtain rfl := Option.some.inj hsr
    have hpos : 0 < rs.size := by omega
    have ht : j % rs.size ∈ RecVStar.commitRounds rs := by
      simp only [RecVStar.commitRounds, List.mem_filter, List.mem_range]
      exact ⟨Nat.mod_lt _ hpos, by simpa using hA⟩
    obtain ⟨k, hk⟩ := List.mem_iff_getElem?.1 ht
    have hkl : k < (RecVStar.commitRounds rs).length := by
      by_contra hc
      have h0 : (RecVStar.commitRounds rs)[k]? = none := List.getElem?_eq_none_iff.2 (by omega)
      rw [h0] at hk
      cases hk
    have hl := roundListed_of_check hrs (roundsListed_of_statements hvs) hkl
    unfold RecVStar.roundListed at hl
    obtain ⟨s, hs, hl⟩ := Array.any_eq_true.1 hl
    obtain ⟨q, hq, hql⟩ := List.any_eq_true.1 hl
    simp only [Bool.and_eq_true] at hql
    have hqv : q.2.1 = s!"rec-c{k}" := beq_iff_eq.1 hql.1
    have hqr : q.2.2 = #[some 0, some 1] := beq_iff_eq.1 hql.2
    have hck : RecVStar.cIndex q.2.1 = some k := by rw [hqv]; exact FlockC3.Reps.cIndex_c k
    obtain ⟨c, hc, qi, hqi⟩ := port_of_portsNamed hpn hs hq hck
    have hsv : ((RecVStar.statements r.shape).toOption.bind fun vs => vs[s]?) = some vs[s] := by
      rw [hvs]
      exact Array.getElem?_eq_getElem hs
    have hrep : j / rs.size = 0 ∨ j / rs.size = 1 := by
      have : j / rs.size < 2 := (Nat.div_lt_iff_lt_mul hpos).2 (by omega)
      generalize j / rs.size = d at this ⊢
      omega
    have hrow : some (j / rs.size) ∈ q.2.2.toList := by
      rw [hqr]
      rcases hrep with h | h <;> simp [h]
    have hmr : msgRound rs (j / rs.size) k = some j := by
      unfold msgRound
      rw [hk]
      show some (j / rs.size * rs.size + j % rs.size) = some j
      rw [Nat.div_add_mod']
    refine ⟨s, by simp only [nS, hr]; omega, (j, q.1, qi, j / rs.size), ?_, rfl⟩
    unfold heldRows
    rw [hsv, hc, hrs]
    refine List.mem_flatMap.2 ⟨q, hq, ?_⟩
    simp only [hck]
    exact List.mem_filterMap.2 ⟨some (j / rs.size), hrow, by simp [hmr, hqi]⟩

theorem rowPiece_nil {r : XR} {i : ℕ} (hn : ∀ s, ∀ h ∈ heldRows r s, h.1 ≠ i) : rowPiece r i = [] := by
  cases hp : rowPiece r i with
  | nil => rfl
  | cons p ps =>
    obtain ⟨h, hh, hi, -⟩ := rowPiece_mem (p := p) (by rw [hp]; exact List.mem_cons_self)
    exact absurd hi (hn _ h hh)

theorem rowPiece_length_one {r : XR} {i s : ℕ} {h : ℕ × String × ℕ × ℕ} (hs : s < r.w.parts.length)
    (hh : h ∈ heldRows r s) (hi : h.1 = i) : (rowPiece r i).length = 1 := by
  obtain ⟨c, hc⟩ := heldRows_circuit hh
  have hf : ((heldRows r s).find? (·.1 == i)).isSome := List.find?_isSome.2 ⟨h, hh, by simp [hi]⟩
  obtain ⟨h', hh'⟩ := Option.isSome_iff_exists.1 hf
  have hne : rowPiece r i ≠ [] := by
    unfold rowPiece
    intro he
    cases hq : (List.range r.w.parts.length).findSome? (fun s => ((heldRows r s).find? (·.1 == i)).bind fun h =>
        (r.w.parts[s]?.bind partCircuit).map fun c => (s, [(([] : Msg), hmCommit c h.2.2.1 h.2.2.2)])) with
    | some p => rw [hq] at he; cases he
    | none =>
      have := List.findSome?_eq_none_iff.1 hq s (List.mem_range.2 hs)
      simp [hh', hc] at this
  have := rowPiece_length r i
  have h0 : (rowPiece r i).length ≠ 0 := fun h0 => hne (List.length_eq_zero_iff.1 h0)
  omega

theorem rowPiece_sum (r : XR) (i : ℕ) :
    ((rowPiece r i).map fun p => pieceLen p.2).sum = 128 * (rowPiece r i).length := by
  have e : (rowPiece r i).map (fun p => pieceLen p.2) = (rowPiece r i).map fun _ => 128 := by
    apply List.map_congr_left
    intro p hp
    obtain ⟨_h, _hh, _hi, _c, _hc, hpe⟩ := rowPiece_mem hp
    rw [hpe]
    simp [pieceLen, hmCommit_length]
  rw [e, sum_map_const]
  omega

theorem capPieces_sum (r : XR) (rs : Array RecVStar.SchedRound) (i : ℕ) :
    (((List.range ((rs.getD (i % rs.size) default).2.1).size).map (capPieceAt r rs i)).map
      fun p => pieceLen p.2).sum = 128 * ((rs.getD (i % rs.size) default).2.1).toList.sum := by
  rw [List.map_map]
  have e : ((fun p : ℕ × List (Msg × List HmNets.Lin) => pieceLen p.2) ∘ capPieceAt r rs i) =
      fun e => 128 * ((rs.getD (i % rs.size) default).2.1).getD e 0 := by
    funext e
    simp only [Function.comp_apply, capPieceAt, pieceLen_topPiece]
  rw [e, sum_map_mul128, range_map_getD]

/-! ## Each round's length -/

/-- **Each round before the last has `128 · nAtSched j` bytes at every witnesses**, at every last round's pieces. -/
theorem roundLens (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    RoundLens fin := by
  intro x hx ws hl j hj
  obtain ⟨r, rs, s0, spec, st, hr, hrs, hset, hm, hsc, hk⟩ := schedBridge hx
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  have hR : layoutOf.rounds x = 2 + 2 * rs.size := hk
  have hjr : j < (layAt fin).rounds x := by show j < kI x; omega
  rw [(layAt fin).msgs_getD x ws hjr]
  cases j with
  | zero =>
    obtain ⟨N, hc, hpos⟩ := capAt_readX hr
    have hN : N = level0Nodes st := by
      rw [capAt_level0 hr hm] at hc
      exact (Prod.mk.inj (Option.some.inj hc)).2.symm
    rw [nAtSched_zero hset, ← hN]
    have hp0 : (layAt fin).pieceAt x 0 = [] := by
      show stepPieces (finOpen fin) x 0 = []
      unfold stepPieces
      rw [hR, if_neg (by omega), if_pos rfl]
    have hf0 : (layAt fin).fixedAt x 0 = none := rfl
    have hm0 : ∀ s, (layAt fin).msgAt x s 0 = capHold r s := fun s => by
      show (readX x).bind (fun r => capHold r s) = _
      rw [hr, Option.bind_some]
    simp only [LayoutB.msg, hf0, hp0, List.isEmpty_nil, ↓reduceIte]
    refine findSome_getD_length (L := 128 * N) ?_ _ ?_
    · intro s b hb
      obtain ⟨w, -, hb⟩ := Option.bind_eq_some_iff.1 hb
      simp only [LayoutB.held, hf0, Option.isSome_none, Bool.false_eq_true, ↓reduceIte, hm0] at hb
      obtain ⟨segs, hsegs, rfl⟩ := Option.map_eq_some_iff.1 hb
      rw [readSegs_length]
      exact capHold_len hc hsegs
    · obtain ⟨segs, hsegs⟩ := capHold_at hr hx hc
      refine ⟨nS x - 1, List.mem_range.2 (by omega), ?_⟩
      rw [List.getElem?_eq_getElem (by omega : nS x - 1 < ws.length)]
      simp [LayoutB.held, hf0, hm0, hsegs]
  | succ i =>
    have hi2 : i < 2 * rs.size := by omega
    have hsr : streamRound x i = some (rs.getD (i % rs.size) default) := by
      unfold streamRound
      rw [hr, Option.bind_some, hrs, Option.bind_some, if_pos hi2]
    rw [nAtSched_succ hset hsc hi2]
    have hpi : (layAt fin).pieceAt x (i + 1) = topsPiecesAt r rs i := by
      show stepPieces (finOpen fin) x (i + 1) = _
      unfold stepPieces
      rw [hR, if_neg (by omega), if_neg (by omega), Nat.add_sub_cancel]
      simp [topsPieces, hr, hrs]
    have hfi : (layAt fin).fixedAt x (i + 1) = if (rs.getD (i % rs.size) default).2.1.isEmpty &&
        !(rs.getD (i % rs.size) default).2.2.2 then some [] else none := by
      show (streamRound x i).bind _ = _
      rw [hsr, Option.bind_some]
    by_cases he : (rs.getD (i % rs.size) default).2.1.isEmpty = true
    · have hpe : topsPiecesAt r rs i = [] := by
        unfold topsPiecesAt
        rw [if_pos he]
      have hsum0 : (rs.getD (i % rs.size) default).2.1.toList.sum = 0 := by
        rw [Array.isEmpty_iff] at he
        rw [he]
        rfl
      rw [hsum0, Nat.zero_add]
      by_cases hA : (rs.getD (i % rs.size) default).2.2.2 = true
      · have hf : (layAt fin).fixedAt x (i + 1) = none := by rw [hfi, he, hA] <;> rfl
        simp only [LayoutB.msg, hf, hpi, hpe, List.isEmpty_nil, ↓reduceIte, hA, Nat.mul_one]
        refine findSome_getD_length (L := 128) ?_ _ ?_
        · intro s b hb
          obtain ⟨w, -, hb⟩ := Option.bind_eq_some_iff.1 hb
          simp only [LayoutB.held, hf, Option.isSome_none, Bool.false_eq_true, ↓reduceIte] at hb
          obtain ⟨segs, hsegs, rfl⟩ := Option.map_eq_some_iff.1 hb
          rw [readSegs_length]
          exact msgAt_succ_len hsegs
        · obtain ⟨s, hs, h, hh, hi⟩ := rowsCoverAll hr hsr hA
          obtain ⟨c, hc⟩ := heldRows_circuit hh
          have hf' : ((heldRows r s).find? (·.1 == i)).isSome := List.find?_isSome.2 ⟨h, hh, by simp [hi]⟩
          obtain ⟨h', hh'⟩ := Option.isSome_iff_exists.1 hf'
          refine ⟨s, List.mem_range.2 (by omega), ?_⟩
          rw [List.getElem?_eq_getElem (by omega : s < ws.length)]
          have hms : (layAt fin).msgAt x s (i + 1) = some [(([] : Msg), hmCommit c h'.2.2.1 h'.2.2.2)] := by
            show (streamRound x i).bind _ = _
            rw [hsr, Option.bind_some]
            simp only [he, hA, ↓reduceIte, hr, Option.bind_some, hc, hh', Option.map_some]
          simp [LayoutB.held, hf, hms]
      · have hA' : (rs.getD (i % rs.size) default).2.2.2 = false := by simpa using hA
        have hf : (layAt fin).fixedAt x (i + 1) = some [] := by rw [hfi, he, hA'] <;> rfl
        simp only [LayoutB.msg, hf, hA', Bool.false_eq_true, ↓reduceIte]
        rfl
    · have he' : (rs.getD (i % rs.size) default).2.1.isEmpty = false := by simpa using he
      have hf : (layAt fin).fixedAt x (i + 1) = none := by rw [hfi, he'] <;> rfl
      have hpt : topsPiecesAt r rs i = (List.range (rs.getD (i % rs.size) default).2.1.size).map
          (capPieceAt r rs i) ++ rowPiece r i := by
        unfold topsPiecesAt
        rw [if_neg he]
      have hsz : 0 < (rs.getD (i % rs.size) default).2.1.size := by
        rcases Nat.eq_zero_or_pos (rs.getD (i % rs.size) default).2.1.size with h0 | h0
        · exact absurd (Array.isEmpty_iff_size_eq_zero.2 h0) he
        · exact h0
      have hne : (layAt fin).pieceAt x (i + 1) ≠ [] := by
        rw [hpi, hpt]
        exact List.ne_nil_of_mem (List.mem_append_left _ (List.mem_map.2 ⟨0, List.mem_range.2 hsz, rfl⟩))
      rw [LayoutB.msg_of_pieces hf hne, assembled_length, hpi, hpt, List.map_append, List.sum_append, capPieces_sum,
        rowPiece_sum]
      by_cases hA : (rs.getD (i % rs.size) default).2.2.2 = true
      · obtain ⟨s, hs, h, hh, hi⟩ := rowsCoverAll hr hsr hA
        rw [rowPiece_length_one (by omega) hh hi, if_pos hA]
        omega
      · have hnr : ∀ s, ∀ h ∈ heldRows r s, h.1 ≠ i := by
          intro s h hh hi
          have := heldRows_absorbs hh hrs
          rw [hi] at this
          exact hA this
        rw [rowPiece_nil hnr, if_neg hA]
        simp

end RecDeploy.RestRead
