import Proofs.Flock.Claims.Recursion.ResStepsWD
import Proofs.Flock.Claims.Recursion.ResEndsW
import Proofs.Flock.Claims.Recursion.AccFits

/-!
# The next rows' owners (`NextHeldW`)

Where a level part's unit holds `acc` at row `p` of the chain (`RecVStar.accReads`: unit `u` of level `l` reads
`u / q · n + opensBelow l + u % q`), row `p + 1` has an owner:

* within the level, the next unit holds it (`acc_holding`);
* at the level's last unit, the next level with openings holds it at its rep's first unit (`next_open`: the levels
  between have none, so `opensBelow` stands still across them);
* past the last level (the chain's end, `p + 1 = rep · n + n`), an algebra part holds it at `acc1` (`acc1_held`),
  wherever some algebra part reads `acc1` (`AccEndHeldW`).

`AccEndHeldW` is a statement about the verifier's reader alone. It holds because Ligerito's final check carries a
consistency sum in both of its residuals (level 0's `queryFold` always adds one, and `Lin`'s operations keep every key),
so `RecAlgebra.accTerms` adds `acc1` to them and `RecResiduals.groups` puts `acc1`'s port in a part's `P`.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-- **Some algebra part reads `acc1`**: at every dense shape the reader reads, some `.inner` statement's part has an
`acc1` port (proved from V*'s algebra and parts: `ResAccEnd.accEndHeldW`). -/
def AccEndHeldW : Prop :=
  ∀ (x : Msg) (r : XR), readX x = some r → r.shape.holo = false → r.shape.sparse = none →
    ∀ (ss : Array VStar.Stmt), _root_.Flock.RecVStar.statements r.shape = .ok ss →
    ∃ s : ℕ, ∃ j nm pt, ss[s]? = some (.inner j nm pt) ∧ "acc1" ∈ pt.S.ports.map (·.1)

/-! ## The levels' rows -/

theorem opensBelow_succ (sch : Schedule) (l : ℕ) :
    RecVStar.opensBelow sch (l + 1) = RecVStar.opensBelow sch l + (sch.levels[l]!).q := by
  simp only [RecVStar.opensBelow, List.range_succ, List.map_append, List.sum_append, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero]

/-- From `a`, either a level `l'` with openings comes within `d` levels and the sums before it and before `a` agree, or
none does and the sum stands still across all `d`. -/
theorem next_open (f : ℕ → ℕ) (a : ℕ) : ∀ d : ℕ,
    (∃ l', a ≤ l' ∧ l' < a + d ∧ 0 < f l' ∧ ((List.range l').map f).sum = ((List.range a).map f).sum) ∨
      ((List.range (a + d)).map f).sum = ((List.range a).map f).sum
  | 0 => Or.inr rfl
  | d + 1 => by
    rcases next_open f a d with ⟨l', h1, h2, h3, h4⟩ | h
    · exact Or.inl ⟨l', h1, by omega, h3, h4⟩
    · by_cases hq : 0 < f (a + d)
      · exact Or.inl ⟨a + d, by omega, by omega, hq, h⟩
      · right
        rw [← Nat.add_assoc, List.range_succ, List.map_append, List.sum_append, h]
        simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
        omega

/-- A dense statement's read shape has no holo flag and no sparse phase. -/
theorem dense_shape {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st)
    (hd : Claims.Recursion.Form.Dense st.form) {r : XR} (hr : readX x = some r) :
    r.shape.holo = false ∧ r.shape.sparse = none := by
  obtain ⟨r', hr', -, rfl⟩ := parse_read hst
  rw [hr] at hr'
  cases hr'
  refine ⟨hd.1, ?_⟩
  have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hd.2
  cases hs : r.shape.sparse with
  | none => rfl
  | some _ => simp [hs] at h2

/-- **Level `l`'s statement is the `l`-th**, at a dense shape: `statementsOf` lists the levels first. -/
theorem recOpen_at_dense {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    (hsp : i.sparse = none) (hh : i.holo = false) {sch : Schedule} (hf : fast100 i.m = some sch) {l : ℕ}
    (hl : l < sch.levels.size) :
    ss[l]? = some (.recOpen l (2 ^ (sch.levels[l]!).lanes) ((sch.levels[l]!).d - (sch.levels[l]!).cap)
      (2 * (sch.levels[l]!).q)) := by
  obtain ⟨A, K, n0, hA, hK, rfl⟩ := OpenLower.statements_dense h hsp
  rw [hh] at hA
  obtain ⟨sch', sp, ps, hsch, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
  obtain rfl : sch' = sch := Option.some.inj ((FlockVBridge.Family.sched_fast100 hsch).symm.trans hf)
  rw [Array.append_assoc, Array.getElem?_append_left (by simpa using hl)]
  simp only [List.getElem?_toArray, List.getElem?_map, List.getElem?_range hl, Option.map_some]

/-- A level's statement sits below the levels' count. -/
theorem recOpen_lt {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    (hsp : i.sparse = none) (hh : i.holo = false) {s lv LANES HH n : ℕ} (hsv : ss[s]? = some (.recOpen lv LANES HH n))
    {sch : Schedule} (hf : fast100 i.m = some sch) : s < sch.levels.size := by
  obtain ⟨A, K, n0, hA, hK, rfl⟩ := OpenLower.statements_dense h hsp
  rw [hh] at hA
  obtain ⟨sch', sp, ps, hsch, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
  obtain rfl : sch' = sch := Option.some.inj ((FlockVBridge.Family.sched_fast100 hsch).symm.trans hf)
  apply Classical.byContradiction
  intro h1
  rw [Array.append_assoc, Array.getElem?_append_right (by simp; omega)] at hsv
  rcases Array.mem_append.1 (Array.mem_of_getElem? hsv) with hm | hm
  · simp at hm
  · obtain ⟨l, N, hlN⟩ := FlockC3.VStarList.caps_cap hK n0 _ hm
    cases hlN

/-- **A level unit's `acc` row** (`accReads`): where unit `u` of level `lv`'s statement holds `acc` at row `p`,
`u < 2q` and `p = u / q · n + opensBelow lv + u % q`. -/
theorem acc_of_holding {r : XR} {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {s lv LANES HH n : ℕ} (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {sch : Schedule}
    (hf : fast100 r.shape.m = some sch) {u p : ℕ} (hh : ("acc", u) ∈ accHoldings r s p) :
    u < 2 * (sch.levels[lv]!).q ∧
      p = u / (sch.levels[lv]!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch lv +
        u % (sch.levels[lv]!).q := by
  have hA : RecVStar.accReads r.shape (.recOpen lv LANES HH n) =
      .ok (some ((Array.range (2 * (sch.levels[lv]!).q)).map fun u =>
        some (u / (sch.levels[lv]!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch lv +
          u % (sch.levels[lv]!).q))) := by
    unfold RecVStar.accReads
    simp only [hf]
    try rfl
  have hst : (_root_.Flock.RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.recOpen lv LANES HH n) := by
    rw [hss]
    exact hsv
  unfold accHoldings at hh
  rw [hst] at hh
  dsimp only at hh
  have hR : accRead r (.recOpen lv LANES HH n) "acc" u = some p := by
    simpa using (List.mem_filter.1 hh).2
  have hu : u < 2 * (sch.levels[lv]!).q := by
    have := MoreSlots.lt_of_accRead (u := u) hA (fun _ _ _ h => by cases h) (by simp [hR])
    simpa using this
  refine ⟨hu, ?_⟩
  unfold accRead at hR
  simp [hA, Except.toOption, hu] at hR
  first
    | exact hR.symm
    | exact hR
    | omega

/-- A holding at a statement of the reader's list gives the row an owner (`accCanon_some`). -/
theorem canon_of_holding {x : Msg} {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt}
    (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss) {s : ℕ} (hs : s < ss.size) {h : String × ℕ} {p : ℕ}
    (hh : h ∈ accHoldings r s p) : (accCanon r p).isSome := by
  obtain ⟨vs, hvs, hlen, hall⟩ := readX_partVStar hr
  obtain rfl : vs = ss := Except.ok.inj (hvs.symm.trans hss)
  obtain ⟨-, pt, cc, -, hpt, hpc, -⟩ := hall s hs
  have hc : r.w.parts[s]?.bind partCircuit = some cc := by rw [hpt, Option.bind_some, hpc]
  obtain ⟨e, he⟩ := accCanon_some (by omega) hc hh
  simp [he]

/-! ## `NextHeldW` -/

/-- **The next rows' owners**, wherever some algebra part reads `acc1` (`hE`): within a level the next unit, at a
level's last unit the next level with openings, and at the chain's end the algebra part's `acc1`. -/
theorem nextHeldW_of (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (hE : AccEndHeldW) : NextHeldW rd wE final w b elem Claims.Recursion.Form.Dense := by
  intro x st hx hf r hr ss hss s lv LANES HH n hsv u p hh
  obtain ⟨hho, hsp⟩ := dense_shape hx hf hr
  obtain ⟨sch, hfs, hls, -, -, -⟩ := OpenLower.recOpen_dense hss hsp hho hsv
  have hlt : lv < sch.levels.size := by
    rw [hls]
    exact recOpen_lt hss hsp hho hsv hfs
  have hs : s < ss.size := (Array.getElem?_eq_some_iff.1 hsv).1
  obtain ⟨hu, hp⟩ := acc_of_holding hss hsv hfs hh
  have hq : 0 < (sch.levels[lv]!).q := by omega
  have hrep : u / (sch.levels[lv]!).q < 2 := by
    rw [Nat.div_lt_iff_lt_mul hq]
    omega
  have hj : u % (sch.levels[lv]!).q < (sch.levels[lv]!).q := Nat.mod_lt _ hq
  by_cases hA : u % (sch.levels[lv]!).q + 1 < (sch.levels[lv]!).q
  · -- the level's next unit
    have hh' := acc_holding hss hsv hfs hrep hA
    have e : u / (sch.levels[lv]!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch lv +
        (u % (sch.levels[lv]!).q + 1) = p + 1 := by
      rw [hp]
      omega
    rw [e] at hh'
    exact canon_of_holding hr hss hs hh'
  · have hlast : u % (sch.levels[lv]!).q + 1 = (sch.levels[lv]!).q := by omega
    have hsucc := opensBelow_succ sch lv
    have hN : lv + 1 + (sch.levels.size - (lv + 1)) = sch.levels.size := by omega
    rcases next_open (fun k => (sch.levels[k]!).q) (lv + 1) (sch.levels.size - (lv + 1)) with
      ⟨l', h1, h2, h3, h4⟩ | hend
    · -- the next level with openings, at the rep's first unit
      have hl' : l' < sch.levels.size := by omega
      have h4' : RecVStar.opensBelow sch l' = RecVStar.opensBelow sch (lv + 1) := h4
      have hsv' := recOpen_at_dense hss hsp hho hfs hl'
      have hh' := acc_holding hss hsv' hfs (j := 0) hrep h3
      have e : u / (sch.levels[lv]!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch l' + 0 =
          p + 1 := by
        rw [h4', hsucc, hp]
        omega
      rw [e] at hh'
      exact canon_of_holding hr hss (Array.getElem?_eq_some_iff.1 hsv').1 hh'
    · -- the chain's end: an algebra part's `acc1`
      rw [hN] at hend
      have hend' : RecVStar.opensBelow sch sch.levels.size = RecVStar.opensBelow sch (lv + 1) := hend
      obtain ⟨s', j', nm, pt, hsv', ha⟩ := hE x r hr hho hsp ss hss
      have hsvO : (_root_.Flock.RecVStar.statements r.shape).toOption.bind (·[s']?) = some (.inner j' nm pt) := by
        rw [hss]
        exact hsv'
      have hh' := acc1_held hsvO hrep ha
      have hR : repOpens r = RecVStar.opensBelow sch sch.levels.size := by
        simp [repOpens, hfs]
      have e : u / (sch.levels[lv]!).q * repOpens r + repOpens r = p + 1 := by
        rw [hR, hp]
        have : RecVStar.opensBelow sch sch.levels.size = RecVStar.opensBelow sch lv + (sch.levels[lv]!).q := by
          rw [hend', hsucc]
        omega
      rw [e] at hh'
      exact canon_of_holding hr hss (Array.getElem?_eq_some_iff.1 hsv').1 hh'

/-! ## At the deployed readers and brings -/

/-- **`StepsWG` at `XsW … AccV` at KS's brings**, wherever some algebra part reads `acc1` (`hE`). -/
theorem stepsWV_deployedE (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE w : Msg → ℝ) (b : Brings)
    (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record)
    (rest : Msg → List (Msg × Claims.Out) → List UInt8) (cmt : Msg → ℕ → Msg → Msg) (hE : AccEndHeldW) :
    StepsWG DeployedR.rdR key wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest) OpenRows.elemR
      Claims.Recursion.Form.Dense recRead cmt
      (XsW DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest) OpenRows.elemR
        (AccV DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest) OpenRows.elemR
          Claims.Recursion.Form.Dense recRead cmt)) :=
  stepsWV_deployed key wE w b aux nOf hd rest cmt (nextHeldW_of _ _ _ _ _ _ hE)

/-- **P3 at KS's brings** (`accPorts_of_G`), wherever some algebra part reads `acc1` (`hE`). -/
theorem accPorts_deployedE (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE w : Msg → ℝ) (b : Brings)
    (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record)
    (rest : Msg → List (Msg × Claims.Out) → List UInt8) (cmt : Msg → ℕ → Msg → Msg) (hE : AccEndHeldW) :
    AccPorts DeployedR.rdR key wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd rest) OpenRows.elemR
      Claims.Recursion.Form.Dense recRead cmt :=
  accPorts_of_G _ key wE _ w _ _ recRead cmt (stepsWV_deployedE key wE w b aux nOf hd rest cmt hE)
    (regSetup_with_tr (regSetup_deployedB b) _)

end RecDeploy.ResLower
