module

public import Properties.SampledProofs.Members
public import Proofs.Properties.SampledProofs.Bits
public import Proofs.Flock.Bridge.BitsKeyProg
public import Proofs.Flock.Bridge.Lowers
public import Proofs.Flock.Claims.Bridge.WrongSet
import Proofs.Flock.SamplingBoundary

/-!
# Members side by side lower each member's units as its own program does

`Properties.SampledProofs.bitsProgs ms` puts an audit's members' programs side by side. A program `P` sits in a program
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
-/

@[expose] public section

namespace Flock.Bridge.Members

open Claims.Sampling Properties.SampledProofs FlockSoundness.Discharge.ZkLink

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
    · rw [if_pos ((mem_map_add (b := b)).2 hiu), if_pos hiu, if_pos (show b + i < b + g by omega), if_pos hig,
        ih i hig hiP]
    · rw [if_neg (mt (mem_map_add (b := b)).1 hiu), if_neg hiu, hA.read hiP]

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
theorem lowers {C : FlockSoundness.Audit.Circuit} (hA : SitsAt Q b P) (htop : P.Topological) {u : List ℕ}
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
    {o : ℕ} {X : Claims.Sampling.Prog} (hA : SitsAt (sideBySide ps) o X) :
    SitsAt (sideBySide (p :: ps)) (p.gates.length + o) X where
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

variable {C : FlockSoundness.Audit.Circuit} {κ : Fin C.N → Option Bool}

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

theorem offsetOf_bitsProgs (ms : SampledProofs.Bits.Members) (m : ℕ) :
    offsetOf (ms.map fun ci => memberProg ci.1 ci.2.size) m = SampledProofs.Bits.base ms m := by
  simp only [offsetOf, SampledProofs.Bits.base, ← List.map_take, List.map_map]
  congr 1
  exact List.map_congr_left fun ci _ => Properties.SampledProofs.SecurityProofs.bitsProg_length _ _

/-- **Member `m`'s program sits in `bitsProgs ms` from `Bits.base ms m`.** -/
theorem sitsAt_bitsProgs (ms : SampledProofs.Bits.Members)
    (htop : ∀ ci ∈ ms, (memberProg ci.1 ci.2.size).Topological) (m : ℕ) (hm : m < ms.length) :
    SitsAt (bitsProgs ms) (SampledProofs.Bits.base ms m) (memberProg ms[m].1 ms[m].2.size) := by
  have h := sitsAt_sideBySide (ms.map fun ci => memberProg ci.1 ci.2.size)
    (fun p hp => by obtain ⟨ci, hci, rfl⟩ := List.mem_map.1 hp; exact htop ci hci) m (by simpa using hm)
  rw [offsetOf_bitsProgs, List.getElem_map] at h
  exact h

/-- **Each unit of a member is lowered side by side**: in `bitsProgs ms`, unit `u`'s gates of a circuit `C` whose
lowering is member `m`'s program, `Bits.base ms m` later, are correct on every committed values that carry `C`'s
evaluation at the wires `φ` maps their positions to, `Bits.base ms m` earlier, as Bridge's `lowers` has them in the
member's own program. -/
theorem lowers_bitsProgs (ms : SampledProofs.Bits.Members)
    (htop : ∀ ci ∈ ms, (memberProg ci.1 ci.2.size).Topological) (m : ℕ) (hm : m < ms.length)
    {C : FlockSoundness.Audit.Circuit} {κ : Fin C.N → Option Bool}
    (hC : progOf C κ = memberProg ms[m].1 ms[m].2.size) {k : ℕ} (P : FlockSoundness.Audit.Partition C k)
    (u : Fin k) (φ : ℕ → Fin C.N) (hφ : ∀ g, φ (posOf C g) = g) (Xv : Fin C.N → Bool)
    (hκ : ∀ a b, C.op a = .input → κ a = some b → (∃ h, P.unit h = some u ∧ a ∈ (C.op h).args) →
      C.eval Xv a = b) :
    WrongSet.Lowers (bitsProgs ms) ((unitGates P u).map (SampledProofs.Bits.base ms m + ·))
      (fun j => φ (j - SampledProofs.Bits.base ms m)) Xv := by
  have hL := Flock.Bridge.lowers (κ := κ) P u φ hφ Xv hκ
  rw [hC] at hL
  exact (sitsAt_bitsProgs ms htop m hm).lowers (htop _ (List.getElem_mem hm))
    (fun v hv => by rw [← hC]; exact lt_length_of_mem_unitGates hv) hL

end Flock.Bridge.Members
