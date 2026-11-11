module

public import Definitions.Claims.SamplingTwo
public import Properties.Spot.Members
public import Proofs.Properties.Spot.Bits
public import Proofs.Flock.Bridge.BitsKeyProg
public import Proofs.Flock.Bridge.Lowers
public import Proofs.Flock.Claims.Bridge.WrongSet
public import Proofs.Flock.SamplingGeneric.ReplayPartition
import Proofs.Flock.SamplingBoundary

/-!
# Members side by side lower each member's units as its own program does

`Properties.Spot.bitsProgs ms` puts an audit's members' programs side by side. A program `P` sits in a program
`Q` from `b` (`SitsAt Q b P`) when `Q`'s gates from `b` are `P`'s reading `b` later, no other gate of `Q` reads one of
them, and `Q` returns one of them exactly when `P` returns it. Then:

* `SitsAt.read`, `SitsAt.runIn`, `SitsAt.boundary`, `SitsAt.need`: `Q` reads, computes, commits and needs a set of
  `P`'s gates `b` later as `P` does them.
* `SitsAt.correct`, `SitsAt.lowers`: a set of `P`'s gates correct in `P` is correct `b` later in `Q`, so
  `WrongSet.Lowers` moves from `P` to `Q`.
* `sitsAt_sideBySide`: each of several topologically ordered programs sits in them side by side, from the gates of the
  ones before it (`offsetOf`).
* `sitsAt_bitsProgs`, `lowers_bitsProgs`: member `m`'s program sits in `bitsProgs ms` from `Bits.base ms m`, so each
  unit of a member's circuit is lowered there, `Bits.base ms m` later, as Bridge's `lowers` lowers it in the member's
  own program (`topological_memberProg`, through `progOf_keyProg`).
* `getElem?_blocks_membersPart`, `stratumOf_membersPart`, `getElem?_blocks_membersPart_bridge`: member `m`'s unit `u`
  is slot `unitBase ms m + u` of `membersPart ms`, in stratum `m`, and its proof block is the member's own
  (`Bits.blockOf`, Bridge's `blockOf` of its key program's partition), `Bits.base ms m` later.
* `membersConsts_base`: the members' constants at `Bits.base ms m + p` are member `m`'s at `p`.
* `getElem?_blocks_filesReplay`: slot `ρ` of the members' replay partition (`filesReplay`) is its request's proof
  blocks' gates together (`slotFiles`).
* `cell_filesReplay`, `cell_filesReplay_input`: where the layout fits the members' files (`FilesFit`: request slot `ρ`
  is unit `ρ`, with room in each member for its units), request `ρ`'s cell of member `t` is `filesBlockOf ms ns ρ t`
  on `[0, n_t)`, injectively (`injective_filesBlockOf`), and an input slot's cells are empty
  (`Claims.Sampling.cell_eq_union`).
* `replayOf_filesReplay`, `replayOf_filesReplay_iff_filesBlockOf`: there, proof block `v` replays in request `ρ`
  exactly when it is one of `ρ`'s units, `filesBlockOf ms ns ρ t p` (`Claims.Sampling.replayOf_eq_some_iff_union`).
-/

@[expose] public section

namespace Flock.Bridge.Members

open Claims.Sampling Properties.Spot FlockSoundness.Discharge.ZkLink

/-! ## A gate shifted -/

theorem reads_shiftGate (b : ℕ) (x : BGate) : (shiftGate b x).reads = x.reads.map (b + ·) := by
  cases x <;> simp [shiftGate, BGate.reads]

theorem shiftGate_zero : shiftGate 0 = id := by
  funext x
  cases x <;> simp [shiftGate]

theorem shiftGate_comp (a b : ℕ) : shiftGate a ∘ shiftGate b = shiftGate (a + b) := by
  funext x
  cases x <;> simp [shiftGate, Nat.add_assoc]

theorem shiftGate_eq_const {b : ℕ} {x : BGate} {c : Bool} : shiftGate b x = .const c ↔ x = .const c := by
  cases x <;> simp [shiftGate]

theorem shiftGate_eq_input {b : ℕ} {x : BGate} : shiftGate b x = .input ↔ x = .input := by
  cases x <;> simp [shiftGate]

theorem mapM_map_add (b : ℕ) (rd : ℕ → Option Bool) :
    ∀ ws : List ℕ, (ws.map (b + ·)).mapM rd = ws.mapM fun i => rd (b + i)
  | [] => rfl
  | w :: ws => by simp only [List.map_cons, List.mapM_cons, mapM_map_add b rd ws]

theorem val_shiftGate (b : ℕ) (x : BGate) (rd : ℕ → Option Bool) (own : Option Bool) :
    (shiftGate b x).val rd own = x.val (fun i => rd (b + i)) own := by
  cases x <;> simp only [shiftGate, BGate.val, mapM_map_add]

theorem mem_map_add {b i : ℕ} {u : List ℕ} : b + i ∈ u.map (b + ·) ↔ i ∈ u := by
  simp

/-! ## A program inside another -/

/-- **`P` sits in `Q` from `b`**: `Q`'s gate `b + p` is `P`'s gate `p` reading `b` later, a gate of `Q` outside them
reads none of them, and `Q` returns `b + p` exactly when `P` returns `p`. -/
structure SitsAt (Q : Claims.Sampling.Prog) (b : ℕ) (P : Claims.Sampling.Prog) : Prop where
  gates : ∀ p < P.gates.length, Q.gates[b + p]? = P.gates[p]?.map (shiftGate b)
  closed : ∀ g (x : BGate), Q.gates[g]? = some x → g < b ∨ b + P.gates.length ≤ g →
    ∀ i ∈ x.reads, i < b ∨ b + P.gates.length ≤ i
  outs : ∀ p < P.gates.length, (b + p ∈ Q.outs ↔ p ∈ P.outs)

namespace SitsAt

variable {Q P : Claims.Sampling.Prog} {b : ℕ}

theorem gates_eq (hA : SitsAt Q b P) {p : ℕ} (hp : p < P.gates.length) :
    Q.gates[b + p]? = some (shiftGate b P.gates[p]) := by
  rw [hA.gates p hp, List.getElem?_eq_getElem hp, Option.map_some]

theorem lt_length (hA : SitsAt Q b P) {p : ℕ} (hp : p < P.gates.length) : b + p < Q.gates.length :=
  (List.getElem?_eq_some_iff.1 (hA.gates_eq hp)).1

theorem readsAt (hA : SitsAt Q b P) {p : ℕ} (hp : p < P.gates.length) :
    (Q.gates[b + p]?.map BGate.reads).getD [] = ((P.gates[p]?.map BGate.reads).getD []).map (b + ·) := by
  rw [hA.gates_eq hp, List.getElem?_eq_getElem hp]
  simp [reads_shiftGate]

/-- `Q` reads gate `b + i` as `P` reads gate `i`, on committed values `b` later. -/
theorem read (hA : SitsAt Q b P) {i : ℕ} (hi : i < P.gates.length) (ref : ℕ → Option Bool) :
    Q.read ref (b + i) = P.read (fun j => ref (b + j)) i := by
  by_cases hc : ∃ c, P.gates[i]? = some (.const c)
  · obtain ⟨c, hc⟩ := hc
    have hq : Q.gates[b + i]? = some (.const c) := by simp [hA.gates i hi, hc, shiftGate]
    rw [Boundary.read_const Q hq, Boundary.read_const P hc]
  · push Not at hc
    have hq : ∀ c, Q.gates[b + i]? ≠ some (.const c) := fun c h => by
      rw [hA.gates_eq hi, Option.some.injEq, shiftGate_eq_const] at h
      exact hc c (by rw [List.getElem?_eq_getElem hi, h])
    rw [Boundary.read_eq Q hq, Boundary.read_eq P hc]

/-- **`Q` computes gates `u` `b` later as `P` computes `u`**, in a topologically ordered `P`. -/
theorem runIn (hA : SitsAt Q b P) (htop : P.Topological) (ref : ℕ → Option Bool) (u : List ℕ) :
    ∀ g < P.gates.length, Q.runIn ref (u.map (b + ·)) (b + g) = P.runIn (fun j => ref (b + j)) u g := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    intro hg
    rw [Boundary.runIn_eq Q, Boundary.runIn_eq P, hA.gates_eq hg, List.getElem?_eq_getElem hg, Option.bind_some,
      Option.bind_some, val_shiftGate]
    refine Boundary.val_congr _ (fun i hi => ?_) (fun _ => rfl)
    have hig : i < g := htop.1 g _ (List.getElem?_eq_getElem hg) i hi
    have hiP : i < P.gates.length := by omega
    try dsimp only
    by_cases hiu : i ∈ u
    · rw [ite_eq_left ((mem_map_add (b := b)).2 hiu), ite_eq_left hiu, ite_eq_left (show b + i < b + g by omega), ite_eq_left hig,
        ih i hig hiP]
    · rw [ite_eq_right (mt (mem_map_add (b := b)).1 hiu), ite_eq_right hiu, hA.read hiP]

/-- `Q` commits gate `b + g` for gates `u` `b` later exactly when `P` commits `g` for `u`. -/
theorem boundary (hA : SitsAt Q b P) (u : List ℕ) {g : ℕ} (hg : g < P.gates.length) :
    Q.Boundary (u.map (b + ·)) (b + g) = P.Boundary u g := by
  refine Bool.eq_iff_iff.2 ?_
  simp only [Prog.Boundary, Bool.or_eq_true, List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq,
    List.contains_iff_mem, List.mem_range]
  refine or_congr (or_congr (hA.outs g hg) ?_) ⟨?_, ?_⟩
  · simp [hA.gates_eq hg, List.getElem?_eq_getElem hg, shiftGate_eq_input]
  · rintro ⟨x, -, hxu, hr⟩
    rcases Nat.lt_or_ge x b with hxb | hxb
    · exfalso
      cases hy : Q.gates[x]? with
      | none => rw [hy] at hr; simp at hr
      | some y =>
        rw [hy] at hr
        rcases hA.closed x y hy (Or.inl hxb) (b + g) (by simpa using hr) with h | h <;> omega
    · rcases Nat.lt_or_ge x (b + P.gates.length) with hxe | hxe
      · obtain ⟨p, rfl⟩ : ∃ p, x = b + p := ⟨x - b, by omega⟩
        have hp : p < P.gates.length := by omega
        rw [hA.readsAt hp, mem_map_add] at hr
        exact ⟨p, hp, fun hpu => hxu (mem_map_add.2 hpu), hr⟩
      · exfalso
        cases hy : Q.gates[x]? with
        | none => rw [hy] at hr; simp at hr
        | some y =>
          rw [hy] at hr
          rcases hA.closed x y hy (Or.inr hxe) (b + g) (by simpa using hr) with h | h <;> omega
  · rintro ⟨p, hp, hpu, hr⟩
    refine ⟨b + p, hA.lt_length hp, fun h => hpu (mem_map_add.1 h), ?_⟩
    rw [hA.readsAt hp, mem_map_add]
    exact hr

/-- What `Q` needs of gates `u` `b` later is what `P` needs of `u`, `b` later. -/
theorem need (hA : SitsAt Q b P) {u : List ℕ} (hu : ∀ v ∈ u, v < P.gates.length) :
    WrongSet.need Q (u.map (b + ·)) = (WrongSet.need P u).map (b + ·) := by
  simp only [WrongSet.need, List.map_append, List.flatMap_map, List.map_flatMap]
  congr 1
  exact List.flatMap_congr fun v hv => hA.readsAt (hu v hv)

theorem lt_of_mem_need (htop : P.Topological) {u : List ℕ} (hu : ∀ v ∈ u, v < P.gates.length) {i : ℕ}
    (hi : i ∈ WrongSet.need P u) : i < P.gates.length := by
  simp only [WrongSet.need, List.mem_append, List.mem_flatMap] at hi
  rcases hi with hi | ⟨v, hv, hi⟩
  · exact hu i hi
  · have hvP := hu v hv
    rw [List.getElem?_eq_getElem hvP] at hi
    simp only [Option.map_some, Option.getD_some] at hi
    exact lt_trans (htop.1 v _ (List.getElem?_eq_getElem hvP) i hi) hvP

/-- **Gates `u` correct in `P` are correct `b` later in `Q`**, on committed values `b` later. -/
theorem correct (hA : SitsAt Q b P) (htop : P.Topological) {ref : ℕ → Option Bool} {u : List ℕ}
    (hu : ∀ v ∈ u, v < P.gates.length) (h : P.Correct (fun j => ref (b + j)) u) :
    Q.Correct ref (u.map (b + ·)) := by
  intro g' hg'
  obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hg'
  have hgP := hu g hg
  obtain ⟨h1, h2⟩ := h g hg
  refine ⟨fun x hx i hi hiu => ?_, fun hB => ?_⟩
  · rw [hA.gates_eq hgP, Option.some.injEq] at hx
    subst hx
    rw [reads_shiftGate] at hi
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hi
    have hjP : j < P.gates.length := lt_trans (htop.1 g _ (List.getElem?_eq_getElem hgP) j hj) hgP
    rw [hA.read hjP]
    exact h1 _ (List.getElem?_eq_getElem hgP) j hj fun hju => hiu (mem_map_add.2 hju)
  · rw [hA.boundary u hgP] at hB
    obtain ⟨v, hv, hrun⟩ := h2 hB
    exact ⟨v, hv, by rw [hA.runIn htop ref u g hgP]; exact hrun⟩

/-- **`Lowers` moves from `P` to `Q`**: gates `u` that lower to the circuit in `P` lower to it `b` later in `Q`, each
position read `b` later. -/
theorem lowers {C : Verity.Boolean.Circuit} (hA : SitsAt Q b P) (htop : P.Topological) {u : List ℕ}
    (hu : ∀ v ∈ u, v < P.gates.length) {φ : ℕ → Fin C.N} {Xv : Fin C.N → Bool}
    (hL : WrongSet.Lowers P u φ Xv) : WrongSet.Lowers Q (u.map (b + ·)) (fun j => φ (j - b)) Xv := by
  intro ref href
  refine hA.correct htop hu (hL _ fun g hg hnc => ?_)
  have hgP := lt_of_mem_need htop hu hg
  have hnc' : ∀ c, Q.gates[b + g]? ≠ some (.const c) := fun c h => by
    rw [hA.gates_eq hgP, Option.some.injEq, shiftGate_eq_const] at h
    exact hnc c (by rw [List.getElem?_eq_getElem hgP, h])
  have := href (b + g) (by rw [hA.need hu]; exact List.mem_map.2 ⟨g, hg, rfl⟩) hnc'
  simpa using this

end SitsAt

/-! ## Programs side by side -/

theorem sideBySide_gates (p : Claims.Sampling.Prog) (ps : List Claims.Sampling.Prog) :
    (sideBySide (p :: ps)).gates = p.gates ++ (sideBySide ps).gates.map (shiftGate p.gates.length) := rfl

theorem sideBySide_outs (p : Claims.Sampling.Prog) (ps : List Claims.Sampling.Prog) :
    (sideBySide (p :: ps)).outs = p.outs ++ (sideBySide ps).outs.map (p.gates.length + ·) := rfl

theorem offsetOf_zero (ps : List Claims.Sampling.Prog) : offsetOf ps 0 = 0 := by
  simp [offsetOf]

theorem offsetOf_succ (p : Claims.Sampling.Prog) (ps : List Claims.Sampling.Prog) (m : ℕ) :
    offsetOf (p :: ps) (m + 1) = p.gates.length + offsetOf ps m := by
  simp [offsetOf]

theorem sitsAt_sideBySide_zero (p : Claims.Sampling.Prog) (ps : List Claims.Sampling.Prog) :
    SitsAt (sideBySide (p :: ps)) 0 p where
  gates i hi := by
    simp [sideBySide_gates, hi, List.getElem?_append_left hi, shiftGate_zero]
  closed g x hx hg i hi := by
    rw [sideBySide_gates] at hx
    rcases hg with hg | hg
    · omega
    · rw [Nat.zero_add] at hg
      rw [List.getElem?_append_right hg, List.getElem?_map] at hx
      cases hy : (sideBySide ps).gates[g - p.gates.length]? with
      | none => rw [hy] at hx; simp at hx
      | some y =>
        rw [hy, Option.map_some, Option.some.injEq] at hx
        subst hx
        rw [reads_shiftGate] at hi
        obtain ⟨j, -, rfl⟩ := List.mem_map.1 hi
        omega
  outs i hi := by
    rw [sideBySide_outs, Nat.zero_add, List.mem_append]
    refine or_iff_left fun h => ?_
    obtain ⟨j, -, hj⟩ := List.mem_map.1 h
    omega

theorem sitsAt_sideBySide_succ {p : Claims.Sampling.Prog} {ps : List Claims.Sampling.Prog} (hp : p.Topological)
    {o : ℕ} {Pm : Claims.Sampling.Prog} (hA : SitsAt (sideBySide ps) o Pm) :
    SitsAt (sideBySide (p :: ps)) (p.gates.length + o) Pm where
  gates i hi := by
    rw [sideBySide_gates, List.getElem?_append_right (show p.gates.length ≤ p.gates.length + o + i by omega),
      List.getElem?_map, show p.gates.length + o + i - p.gates.length = o + i by omega, hA.gates i hi,
      Option.map_map, shiftGate_comp]
  closed g x hx hg i hi := by
    rw [sideBySide_gates] at hx
    by_cases hgp : g < p.gates.length
    · rw [List.getElem?_append_left hgp] at hx
      have := hp.1 g x hx i hi
      omega
    · push Not at hgp
      rw [List.getElem?_append_right hgp, List.getElem?_map] at hx
      cases hy : (sideBySide ps).gates[g - p.gates.length]? with
      | none => rw [hy] at hx; simp at hx
      | some y =>
        rw [hy, Option.map_some, Option.some.injEq] at hx
        subst hx
        rw [reads_shiftGate] at hi
        obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hi
        have := hA.closed _ y hy (by omega) j hj
        omega
  outs i hi := by
    rw [sideBySide_outs, List.mem_append]
    constructor
    · rintro (h | h)
      · have := hp.2 _ h
        omega
      · obtain ⟨j, hj, he⟩ := List.mem_map.1 h
        obtain rfl : j = o + i := by omega
        exact (hA.outs i hi).1 hj
    · intro h
      exact Or.inr (List.mem_map.2 ⟨o + i, (hA.outs i hi).2 h, (Nat.add_assoc _ _ _).symm⟩)

/-- **Each of several topologically ordered programs sits in them side by side**, from the gates of the ones before it. -/
theorem sitsAt_sideBySide : ∀ (ps : List Claims.Sampling.Prog), (∀ p ∈ ps, p.Topological) →
    ∀ (m : ℕ) (hm : m < ps.length), SitsAt (sideBySide ps) (offsetOf ps m) ps[m]
  | [], _, _, hm => absurd hm (Nat.not_lt_zero _)
  | p :: ps, _, 0, _ => by
    rw [offsetOf_zero]
    exact sitsAt_sideBySide_zero p ps
  | p :: ps, htop, m + 1, hm => by
    rw [offsetOf_succ]
    exact sitsAt_sideBySide_succ (htop p List.mem_cons_self)
      (sitsAt_sideBySide ps (fun q hq => htop q (List.mem_cons_of_mem _ hq)) m (by simpa using hm))

/-! ## The members side by side -/

section lowered

variable {C : Verity.Boolean.Circuit} {κ : Fin C.N → Option Bool}

theorem posOf_lt_length (g : Fin C.N) : posOf C g < (progOf C κ).gates.length := by
  have hs := one_le_gateSize (C.op g)
  have h := gates_off (κ := κ) g (j := gateSize (C.op g) - 1) (by omega)
  rw [← posOf_eq, List.getElem?_eq_getElem
    (show gateSize (C.op g) - 1 < (lower C κ g).length by rw [length_lower]; omega)] at h
  exact (List.getElem?_eq_some_iff.1 h).1

theorem lt_length_of_mem_unitGates {n : ℕ} {P : FlockSoundness.Audit.Partition C n} {u : Fin n} {i : ℕ}
    (hi : i ∈ unitGates P u) : i < (progOf C κ).gates.length := by
  obtain ⟨g, -, j, hj, rfl⟩ := mem_unitGates.1 hi
  have h := gates_off (κ := κ) g hj
  rw [List.getElem?_eq_getElem (show j < (lower C κ g).length by rw [length_lower]; exact hj)] at h
  exact (List.getElem?_eq_some_iff.1 h).1

/-- **Bridge's lowering is topologically ordered.** -/
theorem topological_progOf : (progOf C κ).Topological := by
  refine And.intro (fun _ _ hx _ hi => reads_lt hx hi) fun o ho => ?_
  obtain ⟨g, -, rfl⟩ := List.mem_map.1 ho
  exact posOf_lt_length g

end lowered

section keyProg

open FlockSoundness FlockSoundness.Discharge

/-- **A member's program is topologically ordered**: it is Bridge's lowering of the member's circuit. -/
theorem topological_memberProg (c : _root_.Flock.Circuit) {hU : Built c.unit} (hu : c.unit.unitLog ≤ 32) (n : ℕ) :
    (memberProg c n).Topological := by
  have h := topological_progOf (C := (Placed.keyProg c hU n).circuit (Layout.outsOf c hU n)) (κ := constsOf c hU n)
  rw [progOf_keyProg hu] at h
  exact h

end keyProg

theorem offsetOf_bitsProgs (ms : Spot.Bits.Members) (m : ℕ) :
    offsetOf (ms.map fun ci => memberProg ci.1 ci.2.size) m = Spot.Bits.base ms m := by
  simp only [offsetOf, Spot.Bits.base, ← List.map_take, List.map_map]
  congr 1
  exact List.map_congr_left fun ci _ => Properties.Spot.SecurityProofs.bitsProg_length _ _

/-- **Member `m`'s program sits in `bitsProgs ms` from `Bits.base ms m`.** -/
theorem sitsAt_bitsProgs (ms : Spot.Bits.Members)
    (htop : ∀ ci ∈ ms, (memberProg ci.1 ci.2.size).Topological) (m : ℕ) (hm : m < ms.length) :
    SitsAt (bitsProgs ms) (Spot.Bits.base ms m) (memberProg ms[m].1 ms[m].2.size) := by
  have h := sitsAt_sideBySide (ms.map fun ci => memberProg ci.1 ci.2.size)
    (fun p hp => by obtain ⟨ci, hci, rfl⟩ := List.mem_map.1 hp; exact htop ci hci) m (by simpa using hm)
  rw [offsetOf_bitsProgs, List.getElem_map] at h
  exact h

/-- **Each unit of a member is lowered side by side**: in `bitsProgs ms`, unit `u`'s gates of a circuit `C` whose
lowering is member `m`'s program, `Bits.base ms m` later, are correct on every committed values that carry `C`'s
evaluation at the wires `φ` maps their positions to, `Bits.base ms m` earlier, as Bridge's `lowers` has them in the
member's own program. -/
theorem lowers_bitsProgs (ms : Spot.Bits.Members)
    (htop : ∀ ci ∈ ms, (memberProg ci.1 ci.2.size).Topological) (m : ℕ) (hm : m < ms.length)
    {C : Verity.Boolean.Circuit} {κ : Fin C.N → Option Bool}
    (hC : progOf C κ = memberProg ms[m].1 ms[m].2.size) {k : ℕ} (P : FlockSoundness.Audit.Partition C k)
    (u : Fin k) (φ : ℕ → Fin C.N) (hφ : ∀ g, φ (posOf C g) = g) (Xv : Fin C.N → Bool)
    (hκ : ∀ a b, C.op a = .input → κ a = some b → (∃ h, P.unit h = some u ∧ a ∈ (C.op h).args) →
      C.eval Xv a = b) :
    WrongSet.Lowers (bitsProgs ms) ((unitGates P u).map (Spot.Bits.base ms m + ·))
      (fun j => φ (j - Spot.Bits.base ms m)) Xv := by
  have hL := Flock.Bridge.lowers (κ := κ) P u φ hφ Xv hκ
  rw [hC] at hL
  exact (sitsAt_bitsProgs ms htop m hm).lowers (htop _ (List.getElem_mem hm))
    (fun v hv => by rw [← hC]; exact lt_length_of_mem_unitGates hv) hL

/-! ## The members' proof blocks and constants -/

theorem getElem?_flatMap_range {α : Type _} (f : ℕ → List α) {N m u : ℕ} (hm : m < N) (hu : u < (f m).length) :
    ((List.range N).flatMap f)[((List.range m).map fun i => (f i).length).sum + u]? = (f m)[u]? := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_lt hm
  rw [show m + k + 1 = m + (k + 1) by omega, List.range_add, List.flatMap_append,
    List.getElem?_append_right (by rw [List.length_flatMap]; omega), List.length_flatMap, Nat.add_sub_cancel_left,
    List.range_succ_eq_map, List.map_cons, List.flatMap_cons, Nat.add_zero, List.getElem?_append_left hu]

variable (ms : Spot.Bits.Members)

theorem length_memberBlocks {m : ℕ} (hm : m < ms.length) : (memberBlocks ms m).length = ms[m].2.size := by
  simp [memberBlocks, List.getElem?_eq_getElem hm]

theorem getElem?_memberBlocks {m u : ℕ} (hm : m < ms.length) (hu : u < ms[m].2.size) :
    (memberBlocks ms m)[u]? = some ((Spot.Bits.blockOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size)
      (Spot.Bits.consts ms[m].1 ms[m].2.size) u).map (Spot.Bits.base ms m + ·)) := by
  simp [memberBlocks, List.getElem?_eq_getElem hm, hu]

theorem unitBase_eq {m : ℕ} (hm : m ≤ ms.length) :
    unitBase ms m = ((List.range m).map fun i => (memberBlocks ms i).length).sum := by
  unfold unitBase
  congr 1
  refine List.ext_getElem (by simp; omega) fun i h1 h2 => ?_
  simp only [List.length_map, List.length_range] at h2
  simp only [List.getElem_map, List.getElem_take, List.getElem_range]
  rw [length_memberBlocks ms (by omega)]

theorem strata_membersPart : (membersPart ms).strata = (List.range ms.length).map (memberBlocks ms) := by
  simp [membersPart, Partition.strata]

theorem blocks_membersPart : (membersPart ms).blocks = (List.range ms.length).flatMap (memberBlocks ms) := by
  simp [membersPart, Partition.blocks, List.flatMap]

/-- **Member `m`'s unit `u` is slot `unitBase ms m + u`** of the members' partition, its proof block `Bits.blockOf`'s,
`Bits.base ms m` later. -/
theorem getElem?_blocks_membersPart {m u : ℕ} (hm : m < ms.length) (hu : u < ms[m].2.size) :
    (membersPart ms).blocks[unitBase ms m + u]? =
      some ((Spot.Bits.blockOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size)
        (Spot.Bits.consts ms[m].1 ms[m].2.size) u).map (Spot.Bits.base ms m + ·)) := by
  rw [blocks_membersPart, unitBase_eq ms hm.le,
    getElem?_flatMap_range _ hm (by rw [length_memberBlocks ms hm]; exact hu), getElem?_memberBlocks ms hm hu]

theorem stratumIdx_sum_take_add : ∀ (ns : List ℕ) (m u : ℕ) (hm : m < ns.length), u < ns[m] →
    stratumIdx ns ((ns.take m).sum + u) = m
  | n :: _, 0, u, _, hu => by
    have hu' : u < n := by simpa using hu
    simp [stratumIdx, hu']
  | n :: ns, m + 1, u, hm, hu => by
    have hm' : m < ns.length := by simpa using hm
    have hu' : u < ns[m] := by simpa using hu
    rw [List.take_succ_cons, List.sum_cons, stratumIdx, ite_eq_right (by omega),
      show n + (ns.take m).sum + u - n = (ns.take m).sum + u by omega, stratumIdx_sum_take_add ns m u hm' hu']

/-- **Member `m`'s units are stratum `m`** of the members' partition. -/
theorem stratumOf_membersPart {m u : ℕ} (hm : m < ms.length) (hu : u < ms[m].2.size) :
    (membersPart ms).stratumOf (unitBase ms m + u) = m := by
  have hs : (membersPart ms).strata.map List.length = ms.map (·.2.size) := by
    rw [strata_membersPart, List.map_map]
    refine List.ext_getElem (by simp) fun i h1 _ => ?_
    simp only [List.length_map, List.length_range] at h1
    simp only [Function.comp_apply, List.getElem_map, List.getElem_range]
    exact length_memberBlocks ms h1
  unfold Partition.stratumOf
  rw [hs, show unitBase ms m = ((ms.map (·.2.size)).take m).sum by simp [unitBase, List.map_take]]
  exact stratumIdx_sum_take_add _ m u (by simpa using hm) (by simpa using hu)

theorem base_succ {m : ℕ} (hm : m < ms.length) :
    Spot.Bits.base ms (m + 1) =
      Spot.Bits.base ms m + (memberProg ms[m].1 ms[m].2.size).gates.length := by
  rw [← offsetOf_bitsProgs, ← offsetOf_bitsProgs]
  simp [offsetOf, List.take_add_one, List.getElem?_eq_getElem hm]

theorem base_mono {m m' : ℕ} (h : m ≤ m') : Spot.Bits.base ms m ≤ Spot.Bits.base ms m' := by
  rw [← offsetOf_bitsProgs, ← offsetOf_bitsProgs]
  unfold offsetOf
  rw [← List.take_append_drop m ((ms.map fun ci => memberProg ci.1 ci.2.size).take m'), List.take_take,
    Nat.min_eq_left h, List.map_append, List.sum_append]
  exact Nat.le_add_right _ _

/-- **Member `m`'s constants sit at its positions**: the members' constants at `Bits.base ms m + p` are member `m`'s at
`p`. -/
theorem membersConsts_base {m p : ℕ} (hm : m < ms.length)
    (hp : p < (memberProg ms[m].1 ms[m].2.size).gates.length) :
    membersConsts ms (Spot.Bits.base ms m + p) = Spot.Bits.consts ms[m].1 ms[m].2.size p := by
  have hf : (List.range ms.length).find? (fun m' => decide (Spot.Bits.base ms m + p <
      Spot.Bits.base ms (m' + 1))) = some m := by
    refine List.find?_eq_some_iff_getElem.2 ⟨by simp [base_succ ms hm, hp], m, by simpa using hm, by simp,
      fun j hj => ?_⟩
    have := base_mono ms (show j + 1 ≤ m by omega)
    simp at hj ⊢
    omega
  simp [membersConsts, hf, List.getElem?_eq_getElem hm]

section keyProg

open FlockSoundness FlockSoundness.Discharge

/-- **Member `m`'s unit `u`'s slot, in Bridge's terms**: its circuit file's proof block `u` of the key program's
partition at the circuit's constants (`blockOf_bitsKeyProg`), `Bits.base ms m` later. -/
theorem getElem?_blocks_membersPart_bridge {m : ℕ} (hm : m < ms.length) {hU : Built ms[m].1.unit}
    (hu : ms[m].1.unit.unitLog ≤ 32) (u : Fin ms[m].2.size) :
    (membersPart ms).blocks[unitBase ms m + u]? =
      some ((blockOf ((Placed.keyProg ms[m].1 hU ms[m].2.size).partition (Layout.outsOf ms[m].1 hU ms[m].2.size))
        (constsOf ms[m].1 hU ms[m].2.size) u).map (Spot.Bits.base ms m + ·)) := by
  rw [getElem?_blocks_membersPart ms hm u.isLt, blockOf_bitsKeyProg hu u]

end keyProg

/-! ## The members' replay partition and its cells -/

theorem unitBase_succ {m : ℕ} (hm : m < ms.length) : unitBase ms (m + 1) = unitBase ms m + ms[m].2.size := by
  unfold unitBase
  rw [List.take_add_one, List.getElem?_eq_getElem hm, Option.toList_some, List.map_append, List.sum_append]
  rfl

theorem unitBase_mono {m m' : ℕ} (h : m ≤ m') : unitBase ms m ≤ unitBase ms m' := by
  unfold unitBase
  rw [← List.take_append_drop m (ms.take m'), List.take_take, Nat.min_eq_left h, List.map_append, List.sum_append]
  exact Nat.le_add_right _ _

theorem length_blocks_membersPart : (membersPart ms).blocks.length = unitBase ms ms.length := by
  rw [blocks_membersPart, List.length_flatMap, unitBase_eq ms le_rfl]

theorem unitBase_add_lt {m u : ℕ} (hm : m < ms.length) (hu : u < ms[m].2.size) :
    unitBase ms m + u < (membersPart ms).blocks.length := by
  have := unitBase_mono ms (show m + 1 ≤ ms.length by omega)
  rw [unitBase_succ ms hm] at this
  rw [length_blocks_membersPart]
  omega

theorem rank_mul_add_lt {u N n p : ℕ} (hu : u < N) (hp : p < n) : u * n + p < N * n := by
  have := Nat.mul_le_mul_right n (show u + 1 ≤ N by omega)
  rw [Nat.add_one_mul] at this
  omega

theorem injective_filesBlockOf (ns : Array ℕ) (u t : ℕ) : Function.Injective (filesBlockOf ms ns u t) := by
  intro p p' h
  simp only [filesBlockOf] at h
  omega

theorem mem_filesUnits {ns : Array ℕ} {u v : ℕ} :
    v ∈ filesUnits ms ns u ↔ ∃ t, t < ms.length ∧ ∃ p, p < ns.getD t 0 ∧ unitBase ms t + (u * ns.getD t 0 + p) = v := by
  simp [filesUnits, filesBlockOf]

theorem blocks_filesReplay (ns : Array ℕ) (l : Spot.Slots.Layout) :
    (filesReplay ms ns l).blocks = l.blocks.toList.map (filesBlock ms ns) := by
  simp [filesReplay, Partition.blocks, Spot.Slots.Layout.blocks, List.map_flatten, Function.comp_def]

theorem length_blocks_filesReplay (ns : Array ℕ) (l : Spot.Slots.Layout) :
    (filesReplay ms ns l).blocks.length = l.blocks.size := by
  simp [blocks_filesReplay]

/-- Slot `ρ`'s proof blocks in the members' replay partition: its request's (`filesUnits`); none for an input slot or
past the layout. -/
def slotFiles (ns : Array ℕ) (l : Spot.Slots.Layout) (ρ : ℕ) : List ℕ :=
  match (l.blocks[ρ]?).bind (·.unit) with
  | some u => filesUnits ms ns u
  | none => []

/-- **Slot `ρ`'s block is its proof blocks' gates together.** -/
theorem getElem?_blocks_filesReplay (ns : Array ℕ) (l : Spot.Slots.Layout) {ρ : ℕ}
    (hρ : ρ < (filesReplay ms ns l).blocks.length) :
    (filesReplay ms ns l).blocks[ρ]? =
      some ((slotFiles ms ns l ρ).flatMap fun w => (membersPart ms).blocks.getD w []) := by
  rw [length_blocks_filesReplay] at hρ
  simp only [blocks_filesReplay, List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hρ,
    Option.map_some, slotFiles, filesBlock, Option.bind_some]
  cases l.blocks[ρ].unit <;> rfl

/-- **The replay layout fits the members' files**: each request slot `ρ` is unit `ρ`, and each member `t` has room for
its `n_t` units, `ρ·n_t` to `(ρ + 1)·n_t`. -/
def FilesFit (ns : Array ℕ) (l : Spot.Slots.Layout) : Prop :=
  ∀ ρ b, l.blocks[ρ]? = some b → ∀ u, b.unit = some u →
    u = ρ ∧ ∀ t (ht : t < ms.length), (ρ + 1) * ns.getD t 0 ≤ ms[t].2.size

section files

variable {ns : Array ℕ} {l : Spot.Slots.Layout}

/-- A proof block of slot `ρ` is member `t`'s unit `ρ·n_t + p`, with room for it. -/
theorem exists_of_mem_slotFiles (hfit : FilesFit ms ns l) {ρ w : ℕ} (hw : w ∈ slotFiles ms ns l ρ) :
    (∀ t (ht : t < ms.length), (ρ + 1) * ns.getD t 0 ≤ ms[t].2.size) ∧
      ∃ t, t < ms.length ∧ ∃ p, p < ns.getD t 0 ∧ unitBase ms t + (ρ * ns.getD t 0 + p) = w := by
  unfold slotFiles at hw
  rcases hb : l.blocks[ρ]? with _ | b
  · simp [hb] at hw
  · rcases hu : b.unit with _ | u
    · simp [hb, hu] at hw
    · simp only [hb, hu, Option.bind_some] at hw
      obtain ⟨rfl, hroom⟩ := hfit ρ b hb u hu
      exact ⟨hroom, (mem_filesUnits ms).1 hw⟩

/-- **A slot's proof blocks are proof blocks** of the members' partition. -/
theorem lt_of_mem_slotFiles (hfit : FilesFit ms ns l) {ρ w : ℕ} (hw : w ∈ slotFiles ms ns l ρ) :
    w < (membersPart ms).blocks.length := by
  obtain ⟨hroom, t, ht, p, hp, rfl⟩ := exists_of_mem_slotFiles ms hfit hw
  exact unitBase_add_lt ms ht (lt_of_lt_of_le (rank_mul_add_lt (Nat.lt_add_one ρ) hp) (hroom t ht))

/-- **Two slots share no proof block.** -/
theorem not_mem_slotFiles (hfit : FilesFit ms ns l) {ρ₁ ρ₂ : ℕ} (h12 : ρ₁ ≠ ρ₂) {w : ℕ}
    (hw₁ : w ∈ slotFiles ms ns l ρ₁) : w ∉ slotFiles ms ns l ρ₂ := by
  intro hw₂
  obtain ⟨hroom₁, t₁, ht₁, p₁, hp₁, rfl⟩ := exists_of_mem_slotFiles ms hfit hw₁
  obtain ⟨hroom₂, t₂, ht₂, p₂, hp₂, heq⟩ := exists_of_mem_slotFiles ms hfit hw₂
  have hx₁ : ρ₁ * ns.getD t₁ 0 + p₁ < ms[t₁].2.size :=
    lt_of_lt_of_le (rank_mul_add_lt (Nat.lt_add_one ρ₁) hp₁) (hroom₁ t₁ ht₁)
  have hx₂ : ρ₂ * ns.getD t₂ 0 + p₂ < ms[t₂].2.size :=
    lt_of_lt_of_le (rank_mul_add_lt (Nat.lt_add_one ρ₂) hp₂) (hroom₂ t₂ ht₂)
  have e := stratumOf_membersPart ms ht₂ hx₂
  rw [heq, stratumOf_membersPart ms ht₁ hx₁] at e
  subst e
  have h : ρ₁ * ns.getD t₁ 0 + p₁ = ρ₂ * ns.getD t₁ 0 + p₂ := by omega
  rcases Nat.lt_or_gt_of_ne h12 with hlt | hlt
  · have := rank_mul_add_lt hlt hp₁
    omega
  · have := rank_mul_add_lt hlt hp₂
    omega

/-- **Cell `(ρ, t)` of the members' replay partition** is slot `ρ`'s proof blocks in stratum `t`. -/
theorem cell_filesReplay_eq (hnd : (membersPart ms).gates.Nodup) (hne : ∀ u ∈ (membersPart ms).blocks, u ≠ [])
    (hfit : FilesFit ms ns l) {ρ t : ℕ} (hρ : ρ < l.blocks.size) :
    cell (membersPart ms) (filesReplay ms ns l) ρ t =
      ((slotFiles ms ns l ρ).filter fun v => (membersPart ms).stratumOf v = t).toFinset :=
  cell_eq_union (I := slotFiles ms ns l) hnd hne (fun _ h => getElem?_blocks_filesReplay ms ns l h)
    (fun _ _ _ _ h12 _ hw => not_mem_slotFiles ms hfit h12 hw)
    (fun _ _ _ hw => lt_of_mem_slotFiles ms hfit hw) (by rw [length_blocks_filesReplay]; exact hρ)

/-- **Request `ρ`'s cell of member `t` is member `t`'s units `ρ·n_t + p`**, `p < n_t`: `filesBlockOf ms ns ρ t` on
`[0, n_t)`. -/
theorem cell_filesReplay (hnd : (membersPart ms).gates.Nodup) (hne : ∀ u ∈ (membersPart ms).blocks, u ≠ [])
    (hfit : FilesFit ms ns l) {ρ t : ℕ} {b : Spot.Slots.ProofBlock} (hb : l.blocks[ρ]? = some b)
    (hbu : b.unit = some ρ) (ht : t < ms.length) :
    cell (membersPart ms) (filesReplay ms ns l) ρ t = (Finset.range (ns.getD t 0)).image (filesBlockOf ms ns ρ t) := by
  have hroom := (hfit ρ b hb ρ hbu).2
  rw [cell_filesReplay_eq ms hnd hne hfit (Array.getElem?_eq_some_iff.1 hb).1,
    show slotFiles ms ns l ρ = filesUnits ms ns ρ by simp [slotFiles, hb, hbu]]
  ext v
  simp only [List.mem_toFinset, List.mem_filter, decide_eq_true_eq, mem_filesUnits, Finset.mem_image,
    Finset.mem_range, filesBlockOf]
  constructor
  · rintro ⟨⟨t', ht', p, hp, rfl⟩, hs⟩
    rw [stratumOf_membersPart ms ht' (lt_of_lt_of_le (rank_mul_add_lt (Nat.lt_add_one ρ) hp) (hroom t' ht'))] at hs
    subst hs
    exact ⟨p, hp, rfl⟩
  · rintro ⟨p, hp, rfl⟩
    exact ⟨⟨t, ht, p, hp, rfl⟩,
      stratumOf_membersPart ms ht (lt_of_lt_of_le (rank_mul_add_lt (Nat.lt_add_one ρ) hp) (hroom t ht))⟩

/-- **An input slot's cells are empty.** -/
theorem cell_filesReplay_input (hnd : (membersPart ms).gates.Nodup) (hne : ∀ u ∈ (membersPart ms).blocks, u ≠ [])
    (hfit : FilesFit ms ns l) {ρ t : ℕ} {b : Spot.Slots.ProofBlock} (hb : l.blocks[ρ]? = some b)
    (hbu : b.unit = none) : cell (membersPart ms) (filesReplay ms ns l) ρ t = ∅ := by
  rw [cell_filesReplay_eq ms hnd hne hfit (Array.getElem?_eq_some_iff.1 hb).1]
  simp [slotFiles, hb, hbu]

/-- **Proof block `v` replays in request `ρ`** exactly when it is one of slot `ρ`'s proof blocks. -/
theorem replayOf_filesReplay (hnd : (membersPart ms).gates.Nodup) (hne : ∀ u ∈ (membersPart ms).blocks, u ≠ [])
    (hfit : FilesFit ms ns l) {v ρ : ℕ} (hv : v < (membersPart ms).blocks.length) (hρ : ρ < l.blocks.size) :
    replayOf (membersPart ms) (filesReplay ms ns l) v = some ρ ↔ v ∈ slotFiles ms ns l ρ :=
  replayOf_eq_some_iff_union (I := slotFiles ms ns l) hnd hne (fun _ h => getElem?_blocks_filesReplay ms ns l h)
    (fun _ _ _ _ h12 _ hw => not_mem_slotFiles ms hfit h12 hw) hv (by rw [length_blocks_filesReplay]; exact hρ)

/-- **Proof block `v` replays in request `ρ`'s unit slot** exactly when it is unit `p < n_t` of some member `t`:
`v = filesBlockOf ms ns ρ t p`. -/
theorem replayOf_filesReplay_iff_filesBlockOf (hnd : (membersPart ms).gates.Nodup) (hne : ∀ u ∈ (membersPart ms).blocks, u ≠ [])
    (hfit : FilesFit ms ns l) {v ρ : ℕ} {b : Spot.Slots.ProofBlock} (hb : l.blocks[ρ]? = some b)
    (hbu : b.unit = some ρ) (hv : v < (membersPart ms).blocks.length) :
    replayOf (membersPart ms) (filesReplay ms ns l) v = some ρ ↔
      ∃ t, t < ms.length ∧ ∃ p, p < ns.getD t 0 ∧ filesBlockOf ms ns ρ t p = v := by
  rw [replayOf_filesReplay ms hnd hne hfit hv (Array.getElem?_eq_some_iff.1 hb).1,
    show slotFiles ms ns l ρ = filesUnits ms ns ρ by simp [slotFiles, hb, hbu]]
  exact mem_filesUnits ms

end files

end Flock.Bridge.Members
