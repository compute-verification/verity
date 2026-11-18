module

public import Proofs.Properties.InferenceOnly.RecordSharedHeld

/-!
# A strided part's rows lie apart

`rowRuns` reads every part of every argument of a unit (`unitParts`): a part that is a strided part's rows
(`gathered`) as its rows (`gatherRuns`), each its member's own, and the others as runs (`runsOf` of `rowParts`).
On a root it reads, every part is a run or rows, and where one is rows, `apart` passed over the rows and the runs
(`rowRuns_parts`). Sorted by node and first leaf, what `apart` asks of each two neighbours holds of every two
(`apart_sorted`), so no two of them share a leaf but two runs that are one (`apart_pairwise`): a row shares no leaf
with any run (`row_run_apart`), nor with another row (`rows_apart`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Flock.Program (Prog Part)

/-- The parts `rowRuns` reads: every part of an argument of a root node that is no input node, but an edge's. -/
def unitParts (P : Prog) : List Part :=
  let nodes := (P.defs[P.root]!).body.nodes
  let units := nodes.toList.filter fun n => !InferenceOnlyVerifier.Desc.isInputNode P n
  let edge := fun (q : Part) => match q with
    | .affine false i _ _ | .strided false i _ _ _ _ _ =>
      (nodes[i]?.map fun n => !InferenceOnlyVerifier.Desc.isInputNode P n).getD false
    | .explicit it => it.any fun r => !r.p &&
      (nodes[r.i]?.map fun n => !InferenceOnlyVerifier.Desc.isInputNode P n).getD false
    | .grid _ lanes => lanes.all fun l => !l.p &&
      (nodes[l.i]?.map fun n => !InferenceOnlyVerifier.Desc.isInputNode P n).getD false
    | _ => false
  units.flatMap fun n => n.args.toList.flatMap fun a => a.seq.parts.toList.filter (!edge ·)

theorem rowParts_eq (P : Prog) : rowParts P = (unitParts P).filter fun q => !gathered P q := rfl

/-- The rows of the parts that are a strided part's rows, each its member's own. -/
def gatheredRows (P : Prog) : List (Nat × Nat × Nat) :=
  ((unitParts P).filterMap fun q => if gathered P q then gatherRuns P q else none).flatten

/-- What `apart` reads: the rows, marked, and the other parts' runs. -/
def apartList (P : Prog) : List ((Nat × Nat × Nat) × Bool) :=
  (gatheredRows P).map (·, true) ++ ((rowParts P).filterMap (partRun P)).map (·.1, false)

/-- **What `rowRuns` establishes of the parts it reads**: the parts that are no rows are runs `runsOf` admits, with
the root's runs; every part is a run or rows; and where some part is rows, they and the runs lie apart. -/
theorem rowRuns_parts {P : Prog} {R : Runs} (h : rowRuns P = some R) :
    runsOf P (rowParts P) = some R ∧
      (∀ q ∈ unitParts P, (∃ y, partRun P q = some y) ∨ (gathered P q = true ∧ ∃ rs, gatherRuns P q = some rs)) ∧
      ((unitParts P).any (gathered P) = true → apart (apartList P) = true) := by
  have hr := rowRuns_runsOf h
  refine ⟨hr, fun q hq => ?_, fun hg => ?_⟩
  · cases hgq : gathered P q
    · left
      have hq' : q ∈ rowParts P := by
        rw [rowParts_eq]
        exact List.mem_filter.2 ⟨hq, by simp [hgq]⟩
      exact filterMap_all (partRun P) _ (runsOf_some hr).1 q hq'
    · right
      refine ⟨rfl, ?_⟩
      unfold gathered at hgq
      rw [Bool.and_eq_true] at hgq
      exact Option.isSome_iff_exists.1 hgq.2
  · unfold rowRuns at h
    dsimp only at h
    split at h
    · split at h
      · unfold gatherRows at h
        dsimp only at h
        change Option.bind (runsOf P (rowParts P)) _ = some R at h
        rw [hr] at h
        dsimp only [Option.bind] at h
        split at h
        · rename_i ha
          exact ha
        · cases h
      · rename_i hng
        exact absurd hg hng
    · cases h

/-- Run `a`, sorted before `b`: of another node, ending where `b` starts or before, or one unmarked run with it. -/
def ApartBefore (a b : (Nat × Nat × Nat) × Bool) : Prop :=
  a.1.1 ≠ b.1.1 ∨ a.1.2.1 + a.1.2.2 ≤ b.1.2.1 ∨ (a.1 = b.1 ∧ a.2 = false ∧ b.2 = false)

/-- Runs `a` and `b`: of two nodes, sharing no leaf, or one unmarked run. -/
def ApartPair (a b : (Nat × Nat × Nat) × Bool) : Prop :=
  a.1.1 ≠ b.1.1 ∨ a.1.2.1 + a.1.2.2 ≤ b.1.2.1 ∨ b.1.2.1 + b.1.2.2 ≤ a.1.2.1 ∨ (a.1 = b.1 ∧ a.2 = false ∧ b.2 = false)

theorem apartPair_symm {a b : (Nat × Nat × Nat) × Bool} (h : ApartPair a b) : ApartPair b a := by
  rcases h with h | h | h | ⟨h1, h2, h3⟩
  · exact Or.inl (Ne.symm h)
  · exact Or.inr (Or.inr (Or.inl h))
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr (Or.inr ⟨h1.symm, h3, h2⟩))

theorem apartPair_of_before {a b : (Nat × Nat × Nat) × Bool} (h : ApartBefore a b) : ApartPair a b := by
  rcases h with h | h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr (Or.inr h))

/-- Sorted by node and first leaf, runs `apart` passes each with the next are each apart from every later one. -/
theorem apart_sorted : ∀ (l : List ((Nat × Nat × Nat) × Bool)), l.Pairwise (fun a b => runOrd a b = true) →
    ((l.zip l.tail).all fun (a, b) =>
      a.1.1 != b.1.1 || a.1.2.1 + a.1.2.2 ≤ b.1.2.1 || (a.1 == b.1 && !a.2 && !b.2)) = true →
    l.Pairwise ApartBefore
  | [], _, _ => List.Pairwise.nil
  | [_], _, _ => List.pairwise_singleton _ _
  | x :: y :: rest, hs, ha => by
    rw [List.pairwise_cons] at hs
    obtain ⟨hx, hs'⟩ := hs
    simp only [List.tail_cons, List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at ha
    obtain ⟨hxy, ha'⟩ := ha
    have ih := apart_sorted (y :: rest) hs' (by simpa using ha')
    refine List.pairwise_cons.2 ⟨fun z hz => ?_, ih⟩
    have hxy' : ApartBefore x y := by
      simp only [Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq, beq_iff_eq,
        Bool.not_eq_true'] at hxy
      unfold ApartBefore
      rcases hxy with (h | h) | ⟨⟨h1, h2⟩, h3⟩
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr ⟨h1, h2, h3⟩)
    rcases List.mem_cons.1 hz with rfl | hz'
    · exact hxy'
    · have hyz : ApartBefore y z := (List.pairwise_cons.1 ih).1 z hz'
      have hlxy := hx y List.mem_cons_self
      have hlyz := (List.pairwise_cons.1 hs').1 z hz'
      simp only [runOrd, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hlxy hlyz
      unfold ApartBefore at hxy' hyz ⊢
      rcases hxy' with h1 | h1 | ⟨e1, f1, -⟩
      · left
        omega
      · rcases hyz with h2 | h2 | ⟨e2, -, -⟩
        · left
          omega
        · right
          left
          omega
        · right
          left
          rw [← e2]
          exact h1
      · rw [e1]
        rcases hyz with h2 | h2 | ⟨e2, -, g2⟩
        · exact Or.inl h2
        · exact Or.inr (Or.inl h2)
        · exact Or.inr (Or.inr ⟨e2, f1, g2⟩)

/-- **The spec of `apart`**: of the runs it passes, every two that each hold a leaf, at two places of the list, are
of two nodes, share no leaf, or are one unmarked run. -/
theorem apart_pairwise {L : List ((Nat × Nat × Nat) × Bool)} (h : apart L = true) :
    (L.filter fun r => r.1.2.2 > 0).Pairwise ApartPair := by
  unfold apart at h
  dsimp only at h
  have hs := apart_sorted _ (List.pairwise_mergeSort (le := runOrd) runOrd_trans runOrd_total _) h
  exact (List.Perm.pairwise_iff (fun h => apartPair_symm h) (List.mergeSort_perm _ _)).1
    (hs.imp fun {_ _} hab => apartPair_of_before hab)

/-- **A row shares no leaf with any run**: on a root `rowRuns` reads with some part a strided part's rows, a row of
one, holding a leaf, and the run of a part that is no rows, holding one, are of two nodes or share no leaf. -/
theorem row_run_apart {P : Prog} {R : Runs} (h : rowRuns P = some R) (hg : (unitParts P).any (gathered P) = true)
    {q : Part} (hq : q ∈ unitParts P) (hgq : gathered P q = true) {rs : List (Nat × Nat × Nat)}
    (hrs : gatherRuns P q = some rs) {r : Nat × Nat × Nat} (hr : r ∈ rs) (hr0 : 0 < r.2.2) {q' : Part}
    (hq' : q' ∈ rowParts P) {y : Nat × Nat × Nat} {b : Bool} (hy : partRun P q' = some (y, b)) (hy0 : 0 < y.2.2) :
    r.1 ≠ y.1 ∨ r.2.1 + r.2.2 ≤ y.2.1 ∨ y.2.1 + y.2.2 ≤ r.2.1 := by
  obtain ⟨-, -, ha⟩ := rowRuns_parts h
  have hp := apart_pairwise (ha hg)
  have hmr : (r, true) ∈ (apartList P).filter fun r => r.1.2.2 > 0 := by
    refine List.mem_filter.2 ⟨?_, by simpa using hr0⟩
    unfold apartList gatheredRows
    exact List.mem_append_left _ (List.mem_map.2 ⟨r, List.mem_flatten.2 ⟨rs, List.mem_filterMap.2
      ⟨q, hq, by simp [hgq, hrs]⟩, hr⟩, rfl⟩)
  have hmy : (y, false) ∈ (apartList P).filter fun r => r.1.2.2 > 0 := by
    refine List.mem_filter.2 ⟨?_, by simpa using hy0⟩
    unfold apartList
    exact List.mem_append_right _ (List.mem_map.2 ⟨(y, b), List.mem_filterMap.2 ⟨q', hq', hy⟩, rfl⟩)
  rcases pairwise_mem (fun _ _ h => apartPair_symm h) hp _ hmr _ hmy (by simp) with h1 | h1 | h1 | ⟨-, h2, -⟩
  · exact Or.inl h1
  · exact Or.inr (Or.inl h1)
  · exact Or.inr (Or.inr h1)
  · simp at h2

/-- **No two rows share a leaf**: on a root `rowRuns` reads with some part a strided part's rows, two rows, each
holding a leaf, at two places of the rows, are of two nodes or share no leaf. -/
theorem rows_apart {P : Prog} {R : Runs} (h : rowRuns P = some R) (hg : (unitParts P).any (gathered P) = true) :
    ((gatheredRows P).filter fun r => r.2.2 > 0).Pairwise fun r r' =>
      r.1 ≠ r'.1 ∨ r.2.1 + r.2.2 ≤ r'.2.1 ∨ r'.2.1 + r'.2.2 ≤ r.2.1 := by
  obtain ⟨-, -, ha⟩ := rowRuns_parts h
  have hp := (apart_pairwise (ha hg)).sublist ((List.sublist_append_left _ _).filter _)
  rw [List.filter_map, List.pairwise_map] at hp
  refine hp.imp fun {a b} hab => ?_
  unfold ApartPair at hab
  rcases hab with h1 | h1 | h1 | ⟨-, h2, -⟩
  · exact Or.inl h1
  · exact Or.inr (Or.inl h1)
  · exact Or.inr (Or.inr h1)
  · simp at h2

end InferenceOnlyVerifier.Record.SecurityProofs
