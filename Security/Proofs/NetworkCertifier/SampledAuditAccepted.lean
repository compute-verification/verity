import Definitions.NetworkCertifier.Guarantees.SampledAudit
import Proofs.Flock.Claims.Prob

/-!
# The sampled audit implies `Accepted`

`NetTiming.SecurityProofs.SampledAuditAccepted : NetTiming.Guarantees.SampledAuditAccepted`. `KSAccepted` is `KS`'s
bound with a stronger relation on a smaller scope, so it is `KS` once `R δ x w` gives `RAcc δ x w` at every statement in
the audit's scope (`rAcc_of_r`), `PrBad` monotone in the relation (`Claims.Pr_mono`). On that scope:

* **No faulty unit is drawn `all`** (`disjoint_allSlots`): a set that meets a class drawn `all` has that class's
  `P_a` factor `0` (`classesPa_eq_zero`), so it is in no `M_δ` at `δ ≥ 0`.
* **The advice** (`heldAdv`): every faulty unit's committed gate (`FaultyGate`) held at the trace's value there, and the
  other inputs the trace's values.
* **`Accepted`** (`accepted_of`): at each step before the horizon, `C` run on the ingress up to the step and the advice
  holds the trace's value at every egress gate (`valOver_out`). A gate of a unit that isn't faulty computes what its
  proof unit computes on its committed values and interior (`valOver_unit`, by strong induction on the gate, every unit
  correct there, `runIn_ne_none`), and a value it reads from outside its proof unit is a committed one: the reading
  gate's own replay unit's, a held gate's or another correct unit's boundary value, each the trace's. An input reads
  the ingress the trace carries (`inputOf_eq`): the layout fits the program, so no egress gate of a step depends on a
  later step's ingress (`Good`).
-/

namespace NetTiming.SampledAuditProof

open Claims Claims.Sampling

/-! ## A gate's unit is unique -/

theorem idx_eq_of_mem_flatten {α : Type} {L : List (List α)} (h : L.flatten.Nodup) {i j : ℕ} (hi : i < L.length)
    (hj : j < L.length) {a : α} (ha : a ∈ L[i]) (hb : a ∈ L[j]) : i = j := by
  have hp := List.pairwise_iff_getElem.1 (List.pairwise_flatten.1 h).2
  rcases lt_trichotomy i j with hij | hij | hij
  · exact (hp i j hi hj hij a ha a hb rfl).elim
  · exact hij
  · exact (hp j i hj hi hij a hb a ha rfl).elim

theorem gates_eq (P : Partition) : P.gates = (P.rus.map RU.gates).flatten := by
  rw [Partition.gates, List.flatten_flatten]
  rfl

theorem mem_rus {P : Partition} {v : ℕ} {r : RU} (hr : P.rus[v]? = some r) :
    RU.gates r ∈ P.rus.map RU.gates :=
  List.mem_map.2 ⟨r, List.mem_of_getElem? hr, rfl⟩

theorem mem_ru_gates {r : RU} {u : List ℕ} (hu : u ∈ r) {g : ℕ} (hg : g ∈ u) : g ∈ RU.gates r :=
  List.mem_flatten.2 ⟨u, hu, hg⟩

theorem mem_gates {P : Partition} {v : ℕ} {r : RU} (hr : P.rus[v]? = some r) {u : List ℕ} (hu : u ∈ r) {g : ℕ}
    (hg : g ∈ u) : g ∈ P.gates := by
  rw [gates_eq]
  exact List.mem_flatten.2 ⟨_, mem_rus hr, mem_ru_gates hu hg⟩

theorem exists_of_mem_gates {P : Partition} {g : ℕ} (h : g ∈ P.gates) :
    ∃ (v : ℕ) (r : RU) (u : List ℕ), P.rus[v]? = some r ∧ u ∈ r ∧ g ∈ u := by
  rw [gates_eq] at h
  obtain ⟨l, hl, hg⟩ := List.mem_flatten.1 h
  obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hl
  obtain ⟨u, hu, hgu⟩ := List.mem_flatten.1 hg
  obtain ⟨v, hv, rfl⟩ := List.getElem_of_mem hr
  exact ⟨v, _, u, List.getElem?_eq_getElem hv, hu, hgu⟩

/-- **A gate is in one replay unit**: two slots whose replay units hold it are one. -/
theorem slot_unique {P : Partition} (hN : P.gates.Nodup) {v v' : ℕ} {r r' : RU} (hr : P.rus[v]? = some r)
    (hr' : P.rus[v']? = some r') {g : ℕ} (hg : g ∈ RU.gates r) (hg' : g ∈ RU.gates r') : v = v' := by
  rw [gates_eq] at hN
  obtain ⟨hv, rfl⟩ := List.getElem?_eq_some_iff.1 hr
  obtain ⟨hv', rfl⟩ := List.getElem?_eq_some_iff.1 hr'
  have h1 : v < (P.rus.map RU.gates).length := by simpa using hv
  have h2 : v' < (P.rus.map RU.gates).length := by simpa using hv'
  exact idx_eq_of_mem_flatten hN h1 h2 (by simpa using hg) (by simpa using hg')

/-- **A gate is in one proof unit of its replay unit.** -/
theorem unit_unique {P : Partition} (hN : P.gates.Nodup) {v : ℕ} {r : RU} (hr : P.rus[v]? = some r) {u u' : List ℕ}
    (hu : u ∈ r) (hu' : u' ∈ r) {g : ℕ} (hg : g ∈ u) (hg' : g ∈ u') : u = u' := by
  rw [gates_eq] at hN
  have hrN : r.flatten.Nodup := (List.pairwise_flatten.1 hN).1 _ (mem_rus hr)
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hu
  obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hu'
  have := idx_eq_of_mem_flatten hrN hi hj hg hg'
  subst this
  rfl

/-! ## Boundaries, reads and the replay unit's values -/

theorem boundary_of_read {C : Prog} {u : List ℕ} {g' i : ℕ} {x : BGate} (hx : C.gates[g']? = some x)
    (hi : i ∈ x.reads) (hg' : g' ∉ u) : C.Boundary u i = true := by
  have hlt : g' < C.gates.length := (List.getElem?_eq_some_iff.1 hx).1
  unfold Prog.Boundary
  simp only [Bool.or_eq_true, List.any_eq_true, List.mem_range, Bool.and_eq_true, decide_eq_true_eq]
  exact Or.inr ⟨g', hlt, hg', by simp [hx, hi]⟩

theorem boundary_of_input {C : Prog} {u : List ℕ} {g : ℕ} (h : C.gates[g]? = some .input) : C.Boundary u g = true := by
  unfold Prog.Boundary
  simp [h]

theorem boundary_of_out {C : Prog} {u : List ℕ} {g : ℕ} (h : g ∈ C.outs) : C.Boundary u g = true := by
  unfold Prog.Boundary
  simp [h]

theorem read_const {C : Prog} {ref : ℕ → Option Bool} {i : ℕ} {b : Bool} (h : C.gates[i]? = some (.const b)) :
    C.read ref i = some b := by
  simp [Prog.read, h]

theorem read_of_not_const {C : Prog} {ref : ℕ → Option Bool} {i : ℕ} (h : ∀ b, C.gates[i]? ≠ some (.const b)) :
    C.read ref i = ref i := by
  unfold Prog.read
  split
  · rename_i b hb
    exact absurd hb (h b)
  · rfl

theorem ref_of_boundary {C : Prog} {r : RU} {ref : ℕ → Option Bool} {ι : ℕ → Bool} {g : ℕ}
    (h : C.Boundary (RU.gates r) g = true) : RU.ref C r ref ι g = ref g := by
  simp [RU.ref, h]

theorem ref_of_not_mem {C : Prog} {r : RU} {ref : ℕ → Option Bool} {ι : ℕ → Bool} {g : ℕ} (h : g ∉ RU.gates r) :
    RU.ref C r ref ι g = ref g := by
  simp [RU.ref, h]

theorem read_ref_of_not_mem {C : Prog} {r : RU} {ref : ℕ → Option Bool} {ι : ℕ → Bool} {i : ℕ}
    (h : i ∉ RU.gates r) : C.read (RU.ref C r ref ι) i = C.read ref i := by
  unfold Prog.read
  split
  · rfl
  · exact ref_of_not_mem h

/-! ## A gate's value from the wires it reads -/

theorem mapM_ne_none {rd : ℕ → Option Bool} : ∀ {ws : List ℕ}, (∀ i ∈ ws, rd i ≠ none) → ws.mapM rd ≠ none
  | [], _ => by simp
  | i :: ws, h => by
    obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.1 (h i (by simp))
    obtain ⟨bs, hbs⟩ := Option.ne_none_iff_exists'.1 (mapM_ne_none (ws := ws) fun j hj => h j (by simp [hj]))
    simp [List.mapM_cons, hb, hbs]

theorem mapM_congr {rd rd' : ℕ → Option Bool} : ∀ {ws : List ℕ}, (∀ i ∈ ws, rd i = rd' i) → ws.mapM rd = ws.mapM rd'
  | [], _ => rfl
  | i :: ws, h => by
    simp only [List.mapM_cons, h i (by simp), mapM_congr (ws := ws) fun j hj => h j (by simp [hj])]

theorem val_congr {rd rd' : ℕ → Option Bool} {o o' : Option Bool} {x : BGate} (hx : x ≠ .input)
    (h : ∀ i ∈ x.reads, rd i = rd' i) : x.val rd o = x.val rd' o' := by
  cases x with
  | input => exact absurd rfl hx
  | const b => rfl
  | not i => simp only [BGate.val, h i (by simp [BGate.reads])]
  | and ws => simp only [BGate.val, mapM_congr (ws := ws) h]
  | xor ws => simp only [BGate.val, mapM_congr (ws := ws) h]

theorem val_ne_none {rd : ℕ → Option Bool} {o : Option Bool} {x : BGate} (hx : x ≠ .input)
    (h : ∀ i ∈ x.reads, rd i ≠ none) : x.val rd o ≠ none := by
  cases x with
  | input => exact absurd rfl hx
  | const b => simp [BGate.val]
  | not i => simpa [BGate.val] using h i (by simp [BGate.reads])
  | and ws => simpa [BGate.val] using mapM_ne_none (ws := ws) h
  | xor ws => simpa [BGate.val] using mapM_ne_none (ws := ws) h

/-- **A correct proof unit computes every gate of it**: at a topological program, each gate of a correct unit `u` has a
value computed in `u`. -/
theorem runIn_ne_none {C : Prog} (hT : C.Topological) {ref : ℕ → Option Bool} {u : List ℕ} (hc : C.Correct ref u)
    (hlen : ∀ g ∈ u, g < C.gates.length) : ∀ g ∈ u, C.runIn ref u g ≠ none := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
  intro hg
  obtain ⟨x, hx⟩ : ∃ x, C.gates[g]? = some x := ⟨_, List.getElem?_eq_getElem (hlen g hg)⟩
  rw [Prog.runIn]
  simp only [hx, Option.bind_some]
  by_cases hxi : x = .input
  · subst hxi
    obtain ⟨b, hb, -⟩ := (hc g hg).2 (boundary_of_input hx)
    simp [BGate.val, hb]
  · refine val_ne_none hxi fun i hi => ?_
    have hlt := hT.1 g x hx i hi
    by_cases hiu : i ∈ u
    · simp only [hiu, ite_true, dite_eq_left hlt]
      exact ih i hlt hiu
    · simp only [hiu, ite_false]
      exact (hc g hg).1 x hx i hi hiu

/-! ## `C` run with the faulty units' gates held -/

theorem valOver_held {C : Prog} {inp : ℕ → Bool} {free : ℕ → Option Bool} {g : ℕ} {b : Bool} (h : free g = some b) :
    valOver C inp free g = b := by
  rw [valOver, h]

theorem valOver_gate {C : Prog} {inp : ℕ → Bool} {free : ℕ → Option Bool} {g : ℕ} {x : BGate} (h : free g = none)
    (hx : C.gates[g]? = some x) :
    valOver C inp free g =
      (x.val (fun i => if _h : i < g then some (valOver C inp free i) else some false) (some (inp g))).getD false := by
  rw [valOver]
  simp only [h, hx]

theorem valOver_const {C : Prog} {inp : ℕ → Bool} {free : ℕ → Option Bool} {g : ℕ} {b : Bool} (h : free g = none)
    (hx : C.gates[g]? = some (.const b)) : valOver C inp free g = b := by
  rw [valOver_gate h hx]
  rfl

open Classical in
/-- **The advice's held gates**: every faulty unit's committed gate, at the trace's value. -/
noncomputable def heldAdv (C : Prog) (P : Partition) (F : Finset ℕ) (ref : ℕ → Option Bool) (g : ℕ) : Option Bool :=
  if FaultyGate C P F g then some (traceVal C ref g) else none

theorem heldAdv_of {C : Prog} {P : Partition} {F : Finset ℕ} {ref : ℕ → Option Bool} {g : ℕ} (h : FaultyGate C P F g) :
    heldAdv C P F ref g = some (traceVal C ref g) := by
  simp [heldAdv, h]

theorem heldAdv_of_not {C : Prog} {P : Partition} {F : Finset ℕ} {ref : ℕ → Option Bool} {g : ℕ}
    (h : ¬ FaultyGate C P F g) : heldAdv C P F ref g = none := by
  simp [heldAdv, h]

theorem faultyGate_of_heldAdv {C : Prog} {P : Partition} {F : Finset ℕ} {ref : ℕ → Option Bool} {g : ℕ} {b : Bool}
    (h : heldAdv C P F ref g = some b) : FaultyGate C P F g := by
  by_contra hf
  rw [heldAdv_of_not hf] at h
  cases h

section Run

variable {C : Prog} {P : Partition} {ref : ℕ → Option Bool} {F : Finset ℕ} {inp : ℕ → Bool}
  (hT : C.Topological) (hP : P.Partitions C) (ι : ℕ → ℕ → Bool)
  (hι : ∀ v r, P.rus[v]? = some r → v ∉ F → ∀ u ∈ r, C.Correct (RU.ref C r ref (ι v)) u)
  (Good : ℕ → Prop) (hgood : ∀ g x, C.gates[g]? = some x → Good g → ∀ i ∈ x.reads, Good i)
  (hinp : ∀ g, C.gates[g]? = some .input → Good g → inp g = traceVal C ref g)
include hT hP hι hgood hinp

/-- **A unit that isn't faulty computes what its proof unit does**: at a gate whose inputs read the trace, `C` run with
the faulty units' gates held has the value the gate's proof unit computes on its replay unit's committed values and
interior. -/
theorem valOver_unit : ∀ g, Good g → ∀ v r u, P.rus[v]? = some r → v ∉ F → u ∈ r → g ∈ u →
    valOver C inp (heldAdv C P F ref) g = (C.runIn (RU.ref C r ref (ι v)) u g).getD false := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
  intro hgd v r u hr hv hu hg
  obtain ⟨hN, hlen, hcover, -⟩ := hP
  have hglt : g < C.gates.length := hlen g (mem_gates hr hu hg)
  obtain ⟨x, hx⟩ : ∃ x, C.gates[g]? = some x := ⟨_, List.getElem?_eq_getElem hglt⟩
  have hnf : ¬ FaultyGate C P F g := by
    rintro ⟨v', hv', r', hr', hg', -⟩
    rw [slot_unique hN hr hr' (mem_ru_gates hu hg) hg'] at hv
    exact hv hv'
  rw [valOver_gate (heldAdv_of_not hnf) hx, Prog.runIn]
  simp only [hx, Option.bind_some]
  by_cases hxi : x = .input
  · subst hxi
    simp only [BGate.val, Option.getD_some]
    rw [hinp g hx hgd, ref_of_boundary (boundary_of_input hx), traceVal,
      read_of_not_const fun b h => by rw [hx] at h; cases h]
  congr 1
  refine val_congr hxi fun i hi => ?_
  have hlt := hT.1 g x hx i hi
  have hgi := hgood g x hx hgd i hi
  rw [dite_eq_left hlt]
  by_cases hiu : i ∈ u
  · rw [ite_eq_left hiu, dite_eq_left hlt, ih i hlt hgi v r u hr hv hu hiu]
    obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.1
      (runIn_ne_none hT (hι v r hr hv u hu) (fun g' hg' => hlen g' (mem_gates hr hu hg')) i hiu)
    rw [hb]
    rfl
  rw [ite_eq_right hiu]
  obtain ⟨c, hc⟩ := Option.ne_none_iff_exists'.1 ((hι v r hr hv u hu g hg).1 x hx i hi hiu)
  rw [hc]
  congr 1
  by_cases hfi : FaultyGate C P F i
  · rw [valOver_held (heldAdv_of hfi)]
    obtain ⟨v', hv', r', hr', hi', -⟩ := hfi
    have hnot : i ∉ RU.gates r := fun hir => by
      rw [slot_unique hN hr hr' hir hi'] at hv
      exact hv hv'
    rw [traceVal, ← read_ref_of_not_mem (ι := ι v) hnot, hc]
    rfl
  have hfree := heldAdv_of_not (ref := ref) hfi
  by_cases hconst : ∃ b, C.gates[i]? = some (.const b)
  · obtain ⟨b, hb⟩ := hconst
    rw [valOver_const hfree hb]
    rw [read_const hb] at hc
    exact Option.some.inj hc
  push Not at hconst
  have hilt : i < C.gates.length := hlt.trans hglt
  obtain ⟨v', r', u', hr', hu', hiu'⟩ := exists_of_mem_gates (hcover i hilt hconst)
  rw [read_of_not_const hconst] at hc
  by_cases hvv : v' = v
  · subst hvv
    rw [hr] at hr'
    cases hr'
    have hgu' : g ∉ u' := fun hgu' => hiu (unit_unique hN hr hu hu' hg hgu' ▸ hiu')
    obtain ⟨b, hb, hrun⟩ := (hι v' r hr hv u' hu' i hiu').2 (boundary_of_read hx hi hgu')
    rw [ih i hlt hgi v' r u' hr hv hu' hiu', hrun]
    rw [hb] at hc
    exact Option.some.inj hc
  · have hnot : i ∉ RU.gates r := fun hir => hvv (slot_unique hN hr' hr (mem_ru_gates hu' hiu') hir)
    have hgr' : g ∉ RU.gates r' := fun hgr => hvv (slot_unique hN hr hr' (mem_ru_gates hu hg) hgr).symm
    have hbd : C.Boundary (RU.gates r') i = true := boundary_of_read hx hi hgr'
    have hv' : v' ∉ F := fun hv' => hfi ⟨v', hv', r', hr', mem_ru_gates hu' hiu', hbd⟩
    have hgu' : g ∉ u' := fun hgu' => hgr' (mem_ru_gates hu' hgu')
    obtain ⟨b, hb, hrun⟩ := (hι v' r' hr' hv' u' hu' i hiu').2 (boundary_of_read hx hi hgu')
    rw [ih i hlt hgi v' r' u' hr' hv' hu' hiu', hrun]
    rw [ref_of_not_mem hnot] at hc
    rw [ref_of_boundary hbd] at hb
    rw [hb] at hc
    exact Option.some.inj hc

/-- **At a returned gate, `C` run with the faulty units' gates held has the trace's value.** -/
theorem valOver_out {g : ℕ} (hout : g ∈ C.outs) (hgd : Good g) :
    valOver C inp (heldAdv C P F ref) g = traceVal C ref g := by
  by_cases hfg : FaultyGate C P F g
  · exact valOver_held (heldAdv_of hfg)
  have hfree := heldAdv_of_not (ref := ref) hfg
  have hglt : g < C.gates.length := hT.2 g hout
  by_cases hconst : ∃ b, C.gates[g]? = some (.const b)
  · obtain ⟨b, hb⟩ := hconst
    rw [valOver_const hfree hb, traceVal, read_const hb]
    rfl
  push Not at hconst
  obtain ⟨v, r, u, hr, hu, hg⟩ := exists_of_mem_gates (hP.2.2.1 g hglt hconst)
  have hv : v ∉ F := fun hv => hfg ⟨v, hv, r, hr, mem_ru_gates hu hg, boundary_of_out hout⟩
  obtain ⟨b, hb, hrun⟩ := (hι v r hr hv u hu g hg).2 (boundary_of_out hout)
  rw [valOver_unit hT hP ι hι Good hgood hinp g hgd v r u hr hv hu hg, hrun, traceVal, read_of_not_const hconst,
    ← ref_of_boundary (r := r) (ref := ref) (ι := ι v) (boundary_of_out hout), hb]

end Run

/-! ## The ingress -/

/-- **An input reads the ingress the trace carries**: at step `t` before the horizon, on the ingress up to `t` and the
trace's values as the other inputs, a gate in no later step's ingress reads the trace's value. -/
theorem inputOf_eq {L K S : ℕ} {lay : Layout L S} {C : Prog} {ref : ℕ → Option Bool}
    {op : ℕ → List (Obs L K S) → List Bool} {k : ℕ} {o : List (Obs L K S)} {t : ℕ} (hcar : Carries lay C ref op k o)
    (ht : t < k) {g : ℕ} (hgd : ∀ t', t < t' → t' < k → g ∉ lay.ingress t') :
    lay.inputOf k (ingressUpTo op o t) (traceVal C ref) g = traceVal C ref g := by
  have hlen : (ingressUpTo op o t).length = t + 1 := by simp [ingressUpTo]
  unfold Layout.inputOf
  split
  · rename_i i j h
    obtain ⟨i', hi', he⟩ := List.exists_of_findSome?_eq_some h
    obtain ⟨j', hj', hij⟩ := Option.map_eq_some_iff.1 he
    cases hij
    have hi : i < t + 1 := by simpa [hlen] using hi'
    obtain ⟨hjl, hjg, -⟩ := List.findIdx?_eq_some_iff_getElem.1 hj'
    obtain ⟨-, -, hop⟩ := hcar i (by omega)
    have hins : (ingressUpTo op o t).getD i [] = (lay.ingress i).map (traceVal C ref) := by
      rw [List.getD_eq_getElem?_getD]
      simp only [ingressUpTo, List.getElem?_map, List.getElem?_range hi, Option.map_some, Option.getD_some]
      exact hop
    rw [hins, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hjl]
    simp only [Option.map_some, Option.getD_some]
    rw [beq_iff_eq.1 hjg]
  · rename_i h
    have hno : ∀ i < t + 1, g ∉ lay.ingress i := by
      intro i hi hg
      have := List.findSome?_eq_none_iff.1 h i (by simp [hlen, hi])
      rw [Option.map_eq_none_iff, List.findIdx?_eq_none_iff] at this
      simpa using this g hg
    rw [ite_eq_right]
    simp only [List.any_eq_true, List.mem_range, List.contains_iff_mem, not_exists, not_and]
    intro i hik hg
    by_cases hit : i < t + 1
    · exact hno i hit hg
    · exact hgd i (by omega) hik hg

/-! ## No faulty unit is drawn `all` -/

/-- **A set that meets a class drawn `all` has acceptance probability `0`**: that class's factor is `0`. -/
theorem classesPa_eq_zero : ∀ (rs : List Rule) (cs : List (List (List RU))) (o : ℕ) (B : Finset ℕ),
    ¬ Disjoint B (allSlotsFrom rs cs o) → classesPa rs cs o B = 0
  | [], _, _, _, h => by simp [allSlotsFrom] at h
  | _ :: _, [], _, _, h => by simp [allSlotsFrom] at h
  | r :: rs, c :: cs, o, B, h => by
    simp only [classesPa]
    by_cases h2 : Disjoint B (allSlotsFrom rs cs (o + c.flatten.length))
    · rw [allSlotsFrom, Finset.disjoint_union_right] at h
      obtain ⟨first, check⟩ := r
      cases first
      case all =>
        have h1 : ¬ Disjoint B (Finset.Ico o (o + c.flatten.length)) := fun h1 => h ⟨h1, h2⟩
        obtain ⟨v, hvB, hvI⟩ := Finset.not_disjoint_iff.1 h1
        have hw : window o c.flatten.length B ≠ ∅ := by
          intro he
          have hm : v - o ∈ window o c.flatten.length B := by
            simp only [window, Finset.mem_image, Finset.mem_filter]
            exact ⟨v, ⟨hvB, Finset.mem_Ico.1 hvI⟩, rfl⟩
          rw [he] at hm
          simp at hm
        have h0 : Rule.Pa ⟨.all, check⟩ (strataSizes c) (ruSizes c) (window o c.flatten.length B) = 0 := by
          cases check <;> simpa [Rule.Pa, OneStage.Pa] using hw
        rw [h0, zero_mul]
      all_goals exact absurd ⟨Finset.disjoint_empty_right B, h2⟩ h
    · rw [classesPa_eq_zero rs cs _ B h2, mul_zero]

theorem disjoint_allSlots {d : Design} {P : Partition} {δ : ℝ} {B : Finset ℕ} (hδ : 0 ≤ δ) (hM : B ∈ d.M P δ) :
    Disjoint B (allSlots d P) := by
  by_contra h
  have hz := classesPa_eq_zero d.rules P.classes 0 B h
  have hM' : δ < classesPa d.rules P.classes 0 B := hM
  rw [hz] at hM'
  linarith

/-! ## The relation -/

open Classical in
/-- **Every unit that isn't faulty has interior values making its proof units correct.** -/
theorem correct_of_not_faulty (H : Msg → Msg) (Cm : Commitment Bool) (s : Stmt) (P : Partition) (w : Wit) :
    ∃ ι : ℕ → ℕ → Bool, ∀ v r, P.rus[v]? = some r → v ∉ faulty H Cm s P w →
      ∀ u ∈ r, s.prog.Correct (RU.ref s.prog r (w.val H Cm s) (ι v)) u := by
  have : ∀ v, ∃ ι : ℕ → Bool, ∀ r, P.rus[v]? = some r → v ∉ faulty H Cm s P w →
      ∀ u ∈ r, s.prog.Correct (RU.ref s.prog r (w.val H Cm s) ι) u := by
    intro v
    by_cases h : ∃ r, P.rus[v]? = some r ∧ v ∉ faulty H Cm s P w
    · obtain ⟨r, hr, hv⟩ := h
      have hlt : v < P.rus.length := (List.getElem?_eq_some_iff.1 hr).1
      have hnf : ¬ Faulty H Cm s P w v := fun hf => hv (by
        unfold faulty
        exact Finset.mem_filter.2 ⟨Finset.mem_range.2 hlt, hf⟩)
      obtain ⟨r', hr', ι, hι⟩ := not_not.1 hnf
      rw [hr] at hr'
      cases hr'
      exact ⟨ι, fun r'' hr'' _ => by rw [hr] at hr''; cases hr''; exact hι⟩
    · exact ⟨fun _ => false, fun r hr hv => absurd ⟨r, hr, hv⟩ h⟩
  choose ι hι using this
  exact ⟨ι, hι⟩

/-- **The trace's transfer gates carry what the Program accepts**: at a topological program that the partition
partitions and the layout fits, with every unit outside `F` correct at some interior, the advice holding `F`'s
committed gates at the trace's values and the other inputs at the trace's makes `Accepted` every observation sequence
and operator the trace carries with every status complete. -/
theorem accepted_of {L K S : ℕ} {C : Prog} {P : Partition} {lay : Layout L S} {k : ℕ} (hT : C.Topological)
    (hP : P.Partitions C) (hfit : lay.Fits C k) {ref : ℕ → Option Bool} {F : Finset ℕ} (ι : ℕ → ℕ → Bool)
    (hι : ∀ v r, P.rus[v]? = some r → v ∉ F → ∀ u ∈ r, C.Correct (RU.ref C r ref (ι v)) u)
    {op : ℕ → List (Obs L K S) → List Bool} {o : List (Obs L K S)} (hcar : Carries lay C ref op k o)
    (hcov : Covered k o) : Accepted (circuitProgram C lay k) op k o (traceVal C ref, heldAdv C P F ref) := by
  intro t ht
  obtain ⟨ho, hrows, -⟩ := hcar t ht
  refine ⟨ho, ?_, hcov t ht ho⟩
  funext ℓ
  rw [hrows ℓ]
  let Good : ℕ → Prop := fun g => ∀ t', t < t' → t' < k → ∀ i ∈ lay.ingress t', ¬ Depends C i g
  have hgood : ∀ g x, C.gates[g]? = some x → Good g → ∀ i ∈ x.reads, Good i := by
    intro g x hx hg i hi t' h1 h2 j hj hdep
    exact hg t' h1 h2 j hj (hdep.tail (by simp [hx, hi]))
  have hinp : ∀ g, C.gates[g]? = some .input → Good g →
      lay.inputOf k (ingressUpTo op o t) (traceVal C ref) g = traceVal C ref g := fun g _ hg =>
    inputOf_eq hcar ht fun t' h1 h2 hgi => hg t' h1 h2 g hgi Relation.ReflTransGen.refl
  have heq : ∀ σ, ∀ g ∈ slotGates (lay.egress t ℓ σ),
      valOver C (lay.inputOf k (ingressUpTo op o t) (traceVal C ref)) (heldAdv C P F ref) g = traceVal C ref g :=
    fun σ g hg => valOver_out hT hP ι hι Good hgood hinp (hfit.1 t ht ℓ σ g hg).1
      fun t' h1 h2 i hi => hfit.2.2 t ht t' h2 h1 ℓ σ g hg i hi
  funext σ
  show rowOf (traceVal C ref) (lay.egress t ℓ) σ =
    rowOf (valOver C (lay.inputOf k (ingressUpTo op o t) (traceVal C ref)) (heldAdv C P F ref)) (lay.egress t ℓ) σ
  symm
  simp only [rowOf]
  rw [heq σ _ (by simp [slotGates]),
    List.map_congr_left fun g hg => heq σ g (by simp [slotGates, hg])]

variable {Coin Col T : Type}

/-- **On the audit's scope the sampling relation gives the audit's**: at `δ ≥ 0`, `R δ x w` gives `RAcc δ x w`. -/
theorem rAcc_of_r (H : Msg → Msg) (D : Deploy Coin Col T) {L K S : ℕ} (lay : Stmt → Layout L S) (k : Stmt → ℕ)
    {δ : ℝ} (hδ : 0 ≤ δ) {x : Msg} (hx : ScopeAcc D lay k x) {w : Wit}
    (hR : R H D.commit D.parse D.partition δ x w) : RAcc H D.commit D.parse D.partition L K S lay k δ x w := by
  obtain ⟨s, P, hs, hP, hM⟩ := hR
  obtain ⟨⟨s₁, P₁, hs₁, hP₁, hin⟩, s₂, hs₂, hfit⟩ := hx
  rw [hs] at hs₁ hs₂
  obtain rfl := Option.some.inj hs₁
  obtain rfl := Option.some.inj hs₂
  rw [hP] at hP₁
  obtain rfl := Option.some.inj hP₁
  obtain ⟨ι, hι⟩ := correct_of_not_faulty H D.commit s P w
  refine ⟨s, P, hs, hP, hM, disjoint_allSlots hδ hM,
    heldAdv s.prog P (faulty H D.commit s P w) (w.val H D.commit s),
    fun g b h => faultyGate_of_heldAdv h, fun g h => by
      rw [heldAdv_of (ref := w.val H D.commit s) h]; exact Option.some_ne_none _, fun op o hcar hcov => ?_⟩
  exact accepted_of hin.1 hin.2.1 hfit ι hι hcar hcov

end NetTiming.SampledAuditProof

open NetTiming.SampledAuditProof in
/-- **The sampled audit implies `Accepted`, at `ε` at `δ` over the selection**
(`NetTiming.Guarantees.SampledAuditAccepted`): `KS`'s bound at the stronger relation on the smaller scope, `R δ` giving
`RAcc δ` there (`rAcc_of_r`). -/
theorem NetTiming.SecurityProofs.SampledAuditAccepted : NetTiming.Guarantees.SampledAuditAccepted := by
  intro Coin Col T _ _ _ _ _ H D L K S lay k hKS δ decode hδ hdec x hx Ω _ _ P
  have h := hKS δ decode hδ hdec x hx.1 Ω P
  unfold Claims.PrBad at h ⊢
  refine le_trans (Claims.Pr_mono fun s hs => ⟨hs.1, fun hR => hs.2 ?_⟩) h
  revert hR
  cases ((Claims.Sampling.E D decode).eval H P x s.1 s.2.1 s.2.2).1 with
  | none => exact id
  | some w => exact rAcc_of_r H D lay k hδ hx
