module

public import Proofs.Flock.Claims.Sampling.Rewire
public import Proofs.Flock.SamplingGeneric.ReplayPartition

/-!
# The two-stage scope at the wired program

Where a statement's program, its proof-block partition and its two-stage design are in the two-stage scope
(`InScopeTwo`), so are the wired program (`Prog.wired`), the cut partition (`Partition.cut`) and the design with its
replay partition cut (`inScopeTwo_cut`), under conditions on the map `ω` that are each decidable over the program and
the map: `ω` moves only input gates, each to an earlier gate the program returns; it moves no gate it sends a wire to;
the wired program returns no gate `ω` moves; and every proof block has a gate `ω` leaves alone.

* `Prog.unread_wired_of_unread`, `Prog.unread_wired_of_moved`: an unread input of the program is one of the wired
  program, and so is every gate `ω` moves.
* `committed_wired_le`, `sizeClass_of_le`: the wired program commits, of a cut proof block, no more than the program
  commits of the block, so the cut block has a public size class that holds what it commits.
* `Partition.partitions_cut`: the cut partition partitions the wired program.
* `replayOf_cut`, `replayOf_cut_eq`, `Partition.stratumOf_cut`, `cell_cut`: a cut proof block lies in the replay block
  and the stratum the block lay in, so each cell holds the proof blocks it held.
* `Prog.Rewires`: every condition on the map as one proposition, with the scope (`Prog.Rewires.inScopeTwo`) and a cut
  proof block's correctness (`Prog.Rewires.exists_correct`) under it, and `Prog.Rewires.of_earlier`: it holds of a map
  that moves only input gates, each to a returned gate of an earlier member, no two that one proof block holds or
  reads to one gate.
-/

@[expose] public section

namespace Claims.Sampling

/-- **An unread input of `C` is one of the wired program**, where every gate `ω` sends a moved wire to is returned in
`C`. -/
theorem Prog.unread_wired_of_unread {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} {g : ℕ}
    (hw : ∀ i, ω i ≠ i → ω i ∈ C.outs) (hg : C.Unread g) : (C.wired ω keep).Unread g := by
  unfold Prog.Unread at hg
  obtain ⟨hin, ho, hr⟩ := hg
  unfold Prog.wired
  refine Prog.unread_rewire hin (fun (h : g ∈ C.outs.filter keep) => ho (List.mem_filter.1 h).1)
    fun x hx i hi h => ?_
  by_cases hωi : ω i = i
  · exact hr x hx (by rw [← h, hωi]; exact hi)
  · exact ho (by rw [← h]; exact hw i hωi)

/-- **A gate `ω` moves is an unread input of the wired program**, where `ω` moves only input gates, moves no gate it
sends a wire to, and the wired program returns no gate `ω` moves. -/
theorem Prog.unread_wired_of_moved {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} {g : ℕ}
    (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input) (hidem : ∀ i, ω (ω i) = ω i)
    (hkeep : ∀ i, ω i ≠ i → keep i = false) (hg : ω g ≠ g) : (C.wired ω keep).Unread g := by
  unfold Prog.wired
  refine Prog.unread_rewire (hin g hg) (fun (h : g ∈ C.outs.filter keep) => ?_) fun x _ i _ h => hg ?_
  · have hk : keep g = true := (List.mem_filter.1 h).2
    rw [hkeep g hg] at hk
    exact Bool.false_ne_true hk
  · rw [← h]
    exact hidem i

/-- **The wired program commits, of a cut proof block, no more than `C` commits of the block.** -/
theorem committed_wired_le {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} (u : List ℕ)
    (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input) (hw : ∀ i, ω i ≠ i → ω i ∈ C.outs) :
    committed (C.wired ω keep) (cutBlock ω u) ≤ committed C u := by
  have hb : ∀ g, (C.wired ω keep).Boundary (cutBlock ω u) g = true → C.Boundary u g = true := fun g hg =>
    Prog.boundary_of_boundary_rewire (outs := C.outs.filter keep) (u' := cutBlock ω u)
      (fun o ho => (List.mem_filter.1 ho).1) hw
      (fun g' hg'u hg'u' => hin g' fun h => hg'u' (mem_cutBlock.2 ⟨hg'u, h⟩)) hg
  have hcut : ((cutBlock ω u).filter fun g => (C.wired ω keep).Boundary (cutBlock ω u) g) =
      u.filter fun g => (C.wired ω keep).Boundary (cutBlock ω u) g && decide (ω g = g) := by
    show ((u.filter fun g => decide (ω g = g)).filter fun g => (C.wired ω keep).Boundary (cutBlock ω u) g) = _
    exact List.filter_filter
  show ((cutBlock ω u).filter fun g => (C.wired ω keep).Boundary (cutBlock ω u) g).length ≤
    (u.filter fun g => C.Boundary u g).length
  rw [hcut, ← List.countP_eq_length_filter, ← List.countP_eq_length_filter]
  refine List.countP_mono_left fun g _ hg => ?_
  simp only [Bool.and_eq_true] at hg
  exact hb g hg.1

/-- **A count no more than one with a size class has one**: where a list of sizes has a first entry at least `a`, it
has a first entry at least `a' ≤ a`. At the public size classes, a set that commits no more than one with a public size
class has one that holds what it commits. -/
theorem sizeClass_of_le {l : List ℕ} {a a' : ℕ} (hle : a' ≤ a)
    (h : ∃ w, (l.find? fun w => decide (a ≤ w)) = some w ∧ a ≤ w) :
    ∃ w, (l.find? fun w => decide (a' ≤ w)) = some w ∧ a' ≤ w := by
  obtain ⟨w, hw, haw⟩ := h
  have hsome : (l.find? fun w => decide (a' ≤ w)).isSome :=
    List.find?_isSome.2 ⟨w, List.mem_of_find?_eq_some hw, by simpa using hle.trans haw⟩
  obtain ⟨w', hw'⟩ := Option.isSome_iff_exists.1 hsome
  exact ⟨w', hw', by simpa using List.find?_some hw'⟩

/-- **The cut partition partitions the wired program** where the partition partitions `C`: `ω` moves only input gates,
each to a gate `C` returns, moves no gate it sends a wire to, the wired program returns no gate `ω` moves, and every
proof block has a gate `ω` leaves alone. A gate `ω` moves is an unread input there, in no proof block. -/
theorem Partition.partitions_cut {P : Partition} {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool}
    (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input) (hw : ∀ i, ω i ≠ i → ω i ∈ C.outs)
    (hidem : ∀ i, ω (ω i) = ω i) (hkeep : ∀ i, ω i ≠ i → keep i = false)
    (hne : ∀ u ∈ P.blocks, ∃ g ∈ u, ω g = g) (hP : P.Partitions C) :
    (P.cut ω).Partitions (C.wired ω keep) := by
  unfold Partition.Partitions at hP ⊢
  obtain ⟨hnd, hlt, hcov, -, hcls⟩ := hP
  have hlen : (C.wired ω keep).gates.length = C.gates.length := by simp [Prog.wired, Prog.rewire]
  refine ⟨Partition.nodup_gates_cut ω hnd, fun g hg => ?_, fun g hg hnc hnu => ?_, fun u hu => ?_, fun u hu => ?_⟩
  · rw [Partition.gates_cut] at hg
    rw [hlen]
    exact hlt g (mem_cutBlock.1 hg).1
  · rw [hlen] at hg
    have hωg : ω g = g := by
      by_contra hmoved
      exact hnu (Prog.unread_wired_of_moved hin hidem hkeep hmoved)
    have hncC : ∀ b, C.gates[g]? ≠ some (.const b) := fun b hb => hnc b (by
      show (C.rewire ω (C.outs.filter keep)).gates[g]? = some (.const b)
      exact Prog.rewire_const.2 hb)
    rw [Partition.gates_cut]
    exact mem_cutBlock.2 ⟨hcov g hg hncC fun hun => hnu (Prog.unread_wired_of_unread hw hun), hωg⟩
  · rw [Partition.blocks_cut] at hu
    obtain ⟨u₀, hu₀, rfl⟩ := List.mem_map.1 hu
    obtain ⟨g, hg, hωg⟩ := hne u₀ hu₀
    exact List.ne_nil_of_mem (mem_cutBlock.2 ⟨hg, hωg⟩)
  · rw [Partition.blocks_cut] at hu
    obtain ⟨u₀, hu₀, rfl⟩ := List.mem_map.1 hu
    exact sizeClass_of_le (l := sizeClasses) (committed_wired_le u₀ hin hw) (hcls u₀ hu₀)

/-- **A cut proof block lies in the replay block the block lay in**, where no gate is in two replay blocks and the
proof block has a gate `ω` leaves alone: that gate is in no earlier replay block. -/
theorem replayOf_cut {Pt Pρ : Partition} (ω : ℕ → ℕ) (hnd : Pρ.gates.Nodup)
    (hne : ∀ u ∈ Pt.blocks, ∃ g ∈ u, ω g = g) {v ρ : ℕ} (h : replayOf Pt Pρ v = some ρ) :
    replayOf (Pt.cut ω) (Pρ.cut ω) v = some ρ := by
  unfold replayOf at h ⊢
  rw [List.find?_range_eq_some] at h ⊢
  obtain ⟨hp, hρ, -⟩ := h
  cases hu : Pt.blocks[v]? with
  | none => simp [hu] at hp
  | some u =>
    cases hr : Pρ.blocks[ρ]? with
    | none => simp [hu, hr] at hp
    | some r =>
      simp only [hu, hr, List.all_eq_true, List.contains_iff_mem] at hp
      obtain ⟨g, hg, hωg⟩ := hne u (List.mem_of_getElem? hu)
      refine ⟨?_, ?_, fun j hj => ?_⟩
      · simp only [Partition.cut_getElem?, hu, hr, Option.map_some, List.all_eq_true, List.contains_iff_mem]
        intro a ha
        exact mem_cutBlock.2 ⟨hp a (mem_cutBlock.1 ha).1, (mem_cutBlock.1 ha).2⟩
      · rw [Partition.length_blocks_cut]
        exact hρ
      · cases hr' : Pρ.blocks[j]? with
        | none => simp [Partition.cut_getElem?, hu, hr']
        | some r' =>
          simp only [Partition.cut_getElem?, hu, hr', Option.map_some, Bool.not_eq_eq_eq_not, Bool.not_true,
            List.all_eq_false, List.contains_iff_mem]
          exact ⟨g, mem_cutBlock.2 ⟨hg, hωg⟩, fun hgr' =>
            Partition.disjoint_blocks hnd (Nat.ne_of_lt hj) hr' hr (mem_cutBlock.1 hgr').1 (hp g hg)⟩

/-- A cut proof block's replay block is the block's, where the block has one. -/
theorem replayOf_cut_eq {Pt Pρ : Partition} (ω : ℕ → ℕ) (hnd : Pρ.gates.Nodup)
    (hne : ∀ u ∈ Pt.blocks, ∃ g ∈ u, ω g = g) {v : ℕ} (hv : (replayOf Pt Pρ v).isSome) :
    replayOf (Pt.cut ω) (Pρ.cut ω) v = replayOf Pt Pρ v := by
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 hv
  rw [hρ]
  exact replayOf_cut ω hnd hne hρ

/-- The cut partition's strata hold as many proof blocks each as the partition's. -/
theorem Partition.strata_cut (P : Partition) (ω : ℕ → ℕ) :
    (P.cut ω).strata.map List.length = P.strata.map List.length := by
  show ((P.classes.map (List.map (List.map (cutBlock ω)))).flatten).map List.length =
    P.classes.flatten.map List.length
  rw [← List.map_flatten, List.map_map]
  exact List.map_congr_left fun s _ => by simp

theorem Partition.length_strata_cut (P : Partition) (ω : ℕ → ℕ) : (P.cut ω).strata.length = P.strata.length := by
  simpa using congrArg List.length (Partition.strata_cut P ω)

/-- A slot's stratum in the cut partition is its stratum in the partition. -/
theorem Partition.stratumOf_cut (P : Partition) (ω : ℕ → ℕ) (v : ℕ) : (P.cut ω).stratumOf v = P.stratumOf v := by
  unfold Partition.stratumOf
  rw [Partition.strata_cut]

/-- **Each cell holds the proof blocks it held**, cut: where every proof block has a replay block, no gate is in two
replay blocks and every proof block has a gate `ω` leaves alone. -/
theorem cell_cut {Pt Pρ : Partition} (ω : ℕ → ℕ) (hnd : Pρ.gates.Nodup)
    (hne : ∀ u ∈ Pt.blocks, ∃ g ∈ u, ω g = g) (hall : ∀ v < Pt.blocks.length, (replayOf Pt Pρ v).isSome)
    (ρ t : ℕ) : cell (Pt.cut ω) (Pρ.cut ω) ρ t = cell Pt Pρ ρ t := by
  ext v
  simp only [cell, Finset.mem_filter, Finset.mem_range, Partition.length_blocks_cut, Partition.stratumOf_cut]
  exact and_congr_right fun hv => by rw [replayOf_cut_eq ω hnd hne (hall v hv)]

/-- **The two-stage scope at the wired program**: where a statement, its proof-block partition `Pt` and its two-stage
design `d` are in the two-stage scope, so are the statement at the wired program, `Pt` cut and `d` with its replay
partition cut. `ω` moves only input gates, each to an earlier gate the program returns; it moves no gate it sends a
wire to; the wired program returns no gate `ω` moves; and every proof block has a gate `ω` leaves alone. -/
theorem inScopeTwo_cut {Pt : Partition} {d : Design.Two} {s : AuditStatement} {ω : ℕ → ℕ} {keep : ℕ → Bool}
    (hin : ∀ i, ω i ≠ i → s.prog.gates[i]? = some .input) (hw : ∀ i, ω i ≠ i → ω i ∈ s.prog.outs)
    (hle : ∀ i, ω i ≤ i) (hidem : ∀ i, ω (ω i) = ω i) (hkeep : ∀ i, ω i ≠ i → keep i = false)
    (hne : ∀ u ∈ Pt.blocks, ∃ g ∈ u, ω g = g) (h : InScopeTwo Pt d s) :
    InScopeTwo (Pt.cut ω) { d with part := d.part.cut ω } { s with prog := s.prog.wired ω keep } := by
  unfold InScopeTwo at h ⊢
  obtain ⟨htop, hP, hnd, hall, hwf⟩ := h
  refine ⟨Prog.topological_wired htop hle, Partition.partitions_cut hin hw hidem hkeep hne hP,
    Partition.nodup_gates_cut ω hnd, fun v hv => ?_, ?_⟩
  · rw [Partition.length_blocks_cut] at hv
    have e := replayOf_cut_eq ω hnd hne (hall v hv)
    show (replayOf (Pt.cut ω) (d.part.cut ω) v).isSome = true
    rw [e]
    exact hall v hv
  · unfold Design.Two.WellFormed at hwf ⊢
    obtain ⟨h1, h2, h3⟩ := hwf
    exact ⟨(Design.wellFormed_shape d.replay (Partition.shape_cut d.part ω)).2 h1,
      h2.trans (Partition.length_strata_cut Pt ω).symm, h3⟩

/-! ## What the wired program asks of a map, as one proposition -/

/-- **A gate proof block `u` holds or reads** in `C`. -/
def Prog.Touches (C : Prog) (u : List ℕ) (i : ℕ) : Prop :=
  i ∈ u ∨ ∃ g ∈ u, ∃ x : BGate, C.gates[g]? = some x ∧ i ∈ x.reads

/-- **`ω` rewires `C` over partition `P`, returning `keep`**: every condition the wired program's scope
(`Prog.Rewires.inScopeTwo`) and a cut proof block's correctness there (`Prog.Rewires.exists_correct`) read, each
decidable over a program, a partition and a map given as tables. -/
structure Prog.Rewires (C : Prog) (P : Partition) (ω : ℕ → ℕ) (keep : ℕ → Bool) : Prop where
  /-- `ω` moves only input gates -/
  input : ∀ i, ω i ≠ i → C.gates[i]? = some .input
  /-- each to a gate that is no constant -/
  noConst : ∀ i, ω i ≠ i → ∀ c, C.gates[ω i]? ≠ some (.const c)
  /-- which `C` returns -/
  out : ∀ i, ω i ≠ i → ω i ∈ C.outs
  /-- and is no later -/
  le : ∀ i, ω i ≤ i
  /-- `ω` moves no gate it sends a wire to -/
  idem : ∀ i, ω (ω i) = ω i
  /-- the wired program returns no gate `ω` moves -/
  kept : ∀ i, ω i ≠ i → keep i = false
  /-- every proof block has a gate `ω` leaves alone -/
  stays : ∀ u ∈ P.blocks, ∃ g ∈ u, ω g = g
  /-- no proof block reads one of its own gates through `ω` -/
  off : ∀ u ∈ P.blocks, ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, ω i ≠ i → ω i ∉ u
  /-- `ω` sends no two gates one proof block holds or reads to one gate -/
  inj : ∀ u ∈ P.blocks, ∀ i j, C.Touches u i → C.Touches u j → ω i = ω j → i = j

/-- **The two-stage scope at the wired program** (`inScopeTwo_cut`), under `Prog.Rewires`. -/
theorem Prog.Rewires.inScopeTwo {Pt : Partition} {d : Design.Two} {s : AuditStatement} {ω : ℕ → ℕ} {keep : ℕ → Bool}
    (hW : s.prog.Rewires Pt ω keep) (h : InScopeTwo Pt d s) :
    InScopeTwo (Pt.cut ω) { d with part := d.part.cut ω } { s with prog := s.prog.wired ω keep } :=
  inScopeTwo_cut hW.input hW.out hW.le hW.idem hW.kept hW.stays h

/-- **A proof block correct in `C` is correct, cut, at the wired program** (`Prog.exists_correct_wired`), under
`Prog.Rewires`: on values that are, through `ω`, the block's own at every gate it holds or reads. -/
theorem Prog.Rewires.exists_correct {C : Prog} {P : Partition} {ω : ℕ → ℕ} {keep : ℕ → Bool}
    (hW : C.Rewires P ω keep) {u : List ℕ} (hu : u ∈ P.blocks) {v₀ : ℕ → Bool}
    (hc : C.Correct (fun g => some (v₀ g)) u) :
    ∃ v : ℕ → Bool, (∀ i, C.Touches u i → v (ω i) = v₀ i) ∧
      (C.wired ω keep).Correct (fun g => some (v g)) (cutBlock ω u) :=
  Prog.exists_correct_wired hW.input hW.noConst hW.out (hW.off u hu) (C.Touches u) (fun _ hg => Or.inl hg)
    (fun g hg x hx _ hi => Or.inr ⟨g, hg, x, hx, hi⟩) (hW.inj u hu) hc

/-- **A map into earlier members rewires `C`**: `ω` moves only input gates, each to a returned gate of an earlier
member (`mem`, a member's gates before every later member's) that is no input and no constant; the wired program
returns no gate `ω` moves; each proof block has a gate that is no input, and the gates it holds or reads are one
member's; and `ω` sends no two of them that it moves to one gate. -/
theorem Prog.Rewires.of_earlier {C : Prog} {P : Partition} {ω mem : ℕ → ℕ} {keep : ℕ → Bool}
    (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input)
    (hgate : ∀ i, ω i ≠ i → C.gates[ω i]? ≠ some .input ∧ ∀ c, C.gates[ω i]? ≠ some (.const c))
    (hout : ∀ i, ω i ≠ i → ω i ∈ C.outs)
    (hmem : ∀ i j, mem i < mem j → i < j) (hω : ∀ i, ω i ≠ i → mem (ω i) < mem i)
    (hkeep : ∀ i, ω i ≠ i → keep i = false)
    (hblk : ∀ u ∈ P.blocks, (∃ g ∈ u, C.gates[g]? ≠ some .input) ∧ ∃ t, ∀ i, C.Touches u i → mem i = t)
    (hinj : ∀ u ∈ P.blocks, ∀ i j, C.Touches u i → C.Touches u j → ω i ≠ i → ω j ≠ j → ω i = ω j → i = j) :
    C.Rewires P ω keep where
  input := hin
  noConst := fun i h => (hgate i h).2
  out := hout
  le := le_of_earlier hmem hω
  idem := idem_of_not_input hin fun i h => (hgate i h).1
  kept := hkeep
  stays := fun u hu => stays_of_not_input hin (hblk u hu).1
  off := fun u hu => by
    obtain ⟨t, ht⟩ := (hblk u hu).2
    exact off_of_earlier (fun g hg => ht g (Or.inl hg)) (fun g hg x hx i hi => ht i (Or.inr ⟨g, hg, x, hx, hi⟩)) hω
  inj := fun u hu i j hi hj hij => by
    obtain ⟨t, ht⟩ := (hblk u hu).2
    by_cases h1 : ω i = i
    · by_cases h2 : ω j = j
      · rw [h1, h2] at hij
        exact hij
      · exfalso
        have hlt := hω j h2
        rw [← hij, h1, ht i hi, ht j hj] at hlt
        exact Nat.lt_irrefl t hlt
    · by_cases h2 : ω j = j
      · exfalso
        have hlt := hω i h1
        rw [hij, h2, ht j hj, ht i hi] at hlt
        exact Nat.lt_irrefl t hlt
      · exact hinj u hu i j hi hj h1 h2 hij

/-- `Prog.Rewires.of_earlier`'s conditions are met by a map that moves a gate: two members of an input and its
negation each, each a proof block, the second's input wired to the first's output. -/
example : ∃ (C : Prog) (P : Partition) (ω : ℕ → ℕ) (keep : ℕ → Bool), C.Rewires P ω keep ∧ ω 2 ≠ 2 := by
  refine ⟨⟨[.input, .not 0, .input, .not 2], [1, 3]⟩, ⟨[[[[0, 1], [2, 3]]]]⟩, fun i => if i = 2 then 1 else i,
    fun g => g == 3, ?_, by decide⟩
  have hω : ∀ i : ℕ, (if i = 2 then 1 else i) ≠ i → i = 2 := fun i h => by
    by_cases hi : i = 2
    · exact hi
    · exact (h (ite_eq_right hi)).elim
  refine Prog.Rewires.of_earlier (mem := fun i => i / 2) ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · intro i h
    obtain rfl := hω i h
    decide
  · intro i h
    obtain rfl := hω i h
    exact ⟨by decide, fun c => by cases c <;> decide⟩
  · intro i h
    obtain rfl := hω i h
    decide
  · intro i j h
    have h' : i / 2 < j / 2 := h
    omega
  · intro i h
    obtain rfl := hω i h
    decide
  · intro i h
    obtain rfl := hω i h
    decide
  · intro u hu
    have hu' : u ∈ ([[0, 1], [2, 3]] : List (List ℕ)) := hu
    rcases List.mem_cons.1 hu' with rfl | hu''
    · refine ⟨⟨1, by decide, by decide⟩, 0, fun i hi => ?_⟩
      rcases hi with hi | ⟨g, hg, x, hx, hi⟩
      · exact (by decide : ∀ i ∈ [0, 1], i / 2 = 0) i hi
      · rcases List.mem_cons.1 hg with rfl | hg'
        · have hx' : (some BGate.input : Option BGate) = some x := hx
          cases hx'
          exact absurd hi List.not_mem_nil
        · obtain rfl := List.mem_singleton.1 hg'
          have hx' : (some (BGate.not 0) : Option BGate) = some x := hx
          cases hx'
          obtain rfl := List.mem_singleton.1 hi
          rfl
    · obtain rfl := List.mem_singleton.1 hu''
      refine ⟨⟨3, by decide, by decide⟩, 1, fun i hi => ?_⟩
      rcases hi with hi | ⟨g, hg, x, hx, hi⟩
      · exact (by decide : ∀ i ∈ [2, 3], i / 2 = 1) i hi
      · rcases List.mem_cons.1 hg with rfl | hg'
        · have hx' : (some BGate.input : Option BGate) = some x := hx
          cases hx'
          exact absurd hi List.not_mem_nil
        · obtain rfl := List.mem_singleton.1 hg'
          have hx' : (some (BGate.not 2) : Option BGate) = some x := hx
          cases hx'
          obtain rfl := List.mem_singleton.1 hi
          rfl
  · intro u _ i j _ _ hi hj _
    exact (hω i hi).trans (hω j hj).symm

end Claims.Sampling
