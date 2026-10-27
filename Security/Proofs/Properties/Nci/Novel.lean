import Proofs.Properties.Nci.Resolve
import Proofs.Properties.Nci.Width

/-!
# The novel reads, from what `Desc.analyze` and `Desc.novelWidth` compute

Toward `NovelExact`. A segment names gates with their widths (`Width.segGates`). `Desc.intoOf`
collects, per IU, the segments its root nodes read whose first piece is in a root node of another IU and outside θ's
nodes (`intoOf_spec`), and `NciCheck.ius` sums the widths of the gates outside an IU that its gates read, θ's taken out
(`ius_snd`, `hashSet_fold_sum'`), and `Desc.novelWidth` sums the widths of the distinct gates of those segments
(`Width.novelWidthExact`). So `NovelExact` holds wherever one smaller statement does, a named `Prop` here
(`novelExact_of`): `ReadsExact`, that on a Program `Desc.evaluate` reads, the segments `Desc.analyze` gives a root node
name, as gates with their widths, exactly the gates outside the node that its gates read, each in the root node of its
segment's first piece. `readsExact` (`Proofs.Properties.Nci.Agrees`) proves it.

`descAgrees_of_open` gives `DescAgrees` from it, `Walk.OperandsResolve` and `Resolve.BodiesOk`, and
`descAgrees_of_readsExact` (`Proofs.Properties.Nci.Agrees`) from it alone, the other two proved.
-/

namespace NciCheck.Properties.SecurityProofs.Novel

open DescAgree
open Flock.Program (Prog)
open Width (pieceGates piecesGates segGates segsGates dedup_sum)

/-- **`analyze`'s reads are the unrolled operands outside each root node.** On a Program `Desc.evaluate` reads,
with `NciCheck.circuitOf`'s circuit `c`: every gate a segment of root node `k`'s reads names (one of positive count) is a
gate `a` of width `c.w[a]` in the root node `Desc.nodeOf` finds for the segment's first piece; and those gates are
exactly the gates outside node `k` that its gates read. -/
def ReadsExact : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (reads : Array (Array Desc.Seg)) (c : Circuit),
    Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.rootLaidOut P = true →
    Desc.inputsAtRoot P = true → Desc.walkable P = true → Desc.analyze P = .ok reads → circuitOf P = .ok c →
    ∀ k < nodeCount P,
      (∀ s ∈ reads[k]!, s.count ≠ 0 → ∀ q, s.ps[0]? = some q →
        Desc.nodeOf (rootOffs P) q.base.toNat < nodeCount P ∧
        ∀ x ∈ segGates s, ∃ a : ℕ, x = ((a : ℤ), c.w[a]!) ∧ a ∈ nodeGates P (Desc.nodeOf (rootOffs P) q.base.toNat)) ∧
      ∀ a : ℕ, (∃ s ∈ reads[k]!, s.count ≠ 0 ∧ ∃ w, ((a : ℤ), w) ∈ segGates s) ↔
        ∃ g ∈ nodeGates P k, a ∈ c.reads[g]! ∧ a ∉ nodeGates P k

/-! ## What `intoOf` collects -/

/-- `Desc.intoOf`'s step over one segment `s` of a root node of IU `u`. -/
def collStep (offs : Array ℕ) (own : Array (Option ℕ)) (theta : Array Bool) (u : ℕ)
    (into : Array (Array Desc.Seg)) (s : Desc.Seg) : Array (Array Desc.Seg) :=
  match s.ps[0]? with
  | none => into
  | some q =>
    let k' := Desc.nodeOf offs q.base.toNat
    if s.count == 0 || own[k']! == some u || theta[k']! then into else into.modify u (·.push s)

/-- `Desc.intoOf`'s step over root node `k`. -/
def intoStep (offs : Array ℕ) (own : Array (Option ℕ)) (theta : Array Bool) (reads : Array (Array Desc.Seg))
    (into : Array (Array Desc.Seg)) (k : ℕ) : Array (Array Desc.Seg) :=
  match own[k]! with
  | none => into
  | some u => reads[k]!.foldl (collStep offs own theta u) into

theorem intoOf_eq (offs : Array ℕ) (own : Array (Option ℕ)) (theta : Array Bool) (reads : Array (Array Desc.Seg))
    (n : ℕ) :
    Desc.intoOf offs own theta reads n =
      (List.range own.size).foldl (intoStep offs own theta reads) (Array.replicate n #[]) := rfl

/-- What `collStep` keeps for IU `u`: a segment of positive count whose first piece is in a root node of another IU and
outside θ's nodes. -/
abbrev Coll (offs : Array ℕ) (own : Array (Option ℕ)) (theta : Array Bool) (u : ℕ) (s : Desc.Seg) : Prop :=
  ∃ q, s.ps[0]? = some q ∧ s.count ≠ 0 ∧ own[Desc.nodeOf offs q.base.toNat]! ≠ some u ∧
    theta[Desc.nodeOf offs q.base.toNat]! = false

theorem collStep_spec (offs : Array ℕ) (own : Array (Option ℕ)) (theta : Array Bool) (u : ℕ)
    (into : Array (Array Desc.Seg)) (s : Desc.Seg) :
    (collStep offs own theta u into s).size = into.size ∧ ∀ v < into.size, ∀ t,
      t ∈ (collStep offs own theta u into s)[v]! ↔ t ∈ into[v]! ∨ (v = u ∧ t = s ∧ Coll offs own theta u s) := by
  unfold collStep
  rcases hq : s.ps[0]? with _ | q
  · refine ⟨rfl, fun v _ t => ⟨Or.inl, ?_⟩⟩
    rintro (h | ⟨-, -, q', hq', -⟩)
    · exact h
    · rw [hq] at hq'; cases hq'
  · dsimp only
    split
    · next hc =>
      refine ⟨rfl, fun v _ t => ⟨Or.inl, ?_⟩⟩
      rintro (h | ⟨-, -, q', hq', hcnt, ho, hθ⟩)
      · exact h
      · rw [hq] at hq'
        obtain rfl := Option.some.inj hq'
        simp only [Bool.or_eq_true, beq_iff_eq] at hc
        rcases hc with (hc | hc) | hc
        · exact absurd hc hcnt
        · exact absurd hc ho
        · rw [hθ] at hc; cases hc
    · next hc =>
      refine ⟨by simp, fun v hv t => ?_⟩
      rw [getElem!_modify _ _ _ hv]
      by_cases huv : u = v
      · subst huv
        rw [if_pos rfl, Array.mem_push]
        simp only [Bool.or_eq_true, beq_iff_eq, not_or] at hc
        exact or_congr_right ⟨fun h => ⟨rfl, h, q, hq, hc.1.1, hc.1.2, by simpa using hc.2⟩, fun h => h.2.1⟩
      · rw [if_neg huv]
        exact ⟨Or.inl, fun h => h.elim id fun h' => absurd h'.1.symm huv⟩

theorem collFold_spec (offs : Array ℕ) (own : Array (Option ℕ)) (theta : Array Bool) (u : ℕ)
    (ss : Array Desc.Seg) (A : Array (Array Desc.Seg)) :
    (ss.foldl (collStep offs own theta u) A).size = A.size ∧ ∀ v < A.size, ∀ t,
      t ∈ (ss.foldl (collStep offs own theta u) A)[v]! ↔ t ∈ A[v]! ∨ (v = u ∧ t ∈ ss ∧ Coll offs own theta u t) := by
  have H := Array.foldl_induction (as := ss) (f := collStep offs own theta u) (init := A)
    (fun i B => B.size = A.size ∧ ∀ v < A.size, ∀ t,
      t ∈ B[v]! ↔ t ∈ A[v]! ∨ (v = u ∧ (∃ j < i, ss[j]! = t) ∧ Coll offs own theta u t))
    ⟨rfl, fun v _ t => ⟨Or.inl, fun h => h.elim id fun ⟨_, ⟨j, hj, _⟩, _⟩ => absurd hj (Nat.not_lt_zero j)⟩⟩
    (by
      intro i B ⟨hs, hB⟩
      simp only [getElem!_fin]
      obtain ⟨hs', hB'⟩ := collStep_spec offs own theta u B ss[i.1]!
      refine ⟨hs'.trans hs, fun v hv t => ?_⟩
      rw [hB' v (by rw [hs]; exact hv) t, hB v hv t]
      constructor
      · rintro ((h | ⟨h1, ⟨j, hj, h2⟩, h3⟩) | ⟨h1, rfl, h3⟩)
        · exact Or.inl h
        · exact Or.inr ⟨h1, ⟨j, by omega, h2⟩, h3⟩
        · exact Or.inr ⟨h1, ⟨i.1, by omega, rfl⟩, h3⟩
      · rintro (h | ⟨h1, ⟨j, hj, h2⟩, h3⟩)
        · exact Or.inl (Or.inl h)
        · rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | hj
          · exact Or.inl (Or.inr ⟨h1, ⟨j, hj, h2⟩, h3⟩)
          · subst hj
            subst h2
            exact Or.inr ⟨h1, rfl, h3⟩)
  refine ⟨H.1, fun v hv t => (H.2 v hv t).trans (or_congr_right (and_congr_right fun _ => and_congr_left fun _ => ?_))⟩
  constructor
  · rintro ⟨j, hj, rfl⟩
    rw [getElem!_pos ss j hj]; exact Array.getElem_mem hj
  · intro hx
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hx
    exact ⟨j, hj, by rw [getElem!_pos ss j hj]⟩

theorem intoStep_none (offs : Array ℕ) (own : Array (Option ℕ)) (theta : Array Bool)
    (reads : Array (Array Desc.Seg)) (A : Array (Array Desc.Seg)) {k : ℕ} (h : own[k]! = none) :
    intoStep offs own theta reads A k = A := by
  unfold intoStep; rw [h]

theorem intoStep_some (offs : Array ℕ) (own : Array (Option ℕ)) (theta : Array Bool)
    (reads : Array (Array Desc.Seg)) (A : Array (Array Desc.Seg)) {k u : ℕ} (h : own[k]! = some u) :
    intoStep offs own theta reads A k = reads[k]!.foldl (collStep offs own theta u) A := by
  unfold intoStep; rw [h]

/-- `intoOf`'s first `m` root nodes: per IU, the segments its nodes read that `collStep` keeps. -/
theorem intoOf_inv (offs : Array ℕ) (own : Array (Option ℕ)) (theta : Array Bool) (reads : Array (Array Desc.Seg))
    (n : ℕ) : ∀ m,
    let T := (List.range m).foldl (intoStep offs own theta reads) (Array.replicate n #[])
    T.size = n ∧ ∀ u < n, ∀ s, s ∈ T[u]! ↔ ∃ k < m, own[k]! = some u ∧ s ∈ reads[k]! ∧ Coll offs own theta u s
  | 0 => by
    refine ⟨by simp, fun u hu s => ?_⟩
    rw [List.range_zero, List.foldl_nil, getElem!_pos _ u (by simpa using hu), Array.getElem_replicate]
    simp
  | m + 1 => by
    obtain ⟨hs, hT⟩ := intoOf_inv offs own theta reads n m
    intro T
    simp only [T, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    generalize (List.range m).foldl _ _ = A at hs hT
    rcases ho : own[m]! with _ | v
    · rw [intoStep_none offs own theta reads A ho]
      refine ⟨hs, fun u hu s => ?_⟩
      rw [hT u hu s]
      constructor
      · rintro ⟨k, hk, h⟩; exact ⟨k, by omega, h⟩
      · rintro ⟨k, hk, h1, h⟩
        rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | hk
        · exact ⟨k, hk, h1, h⟩
        · subst hk; rw [ho] at h1; cases h1
    · rw [intoStep_some offs own theta reads A ho]
      obtain ⟨hs', hB⟩ := collFold_spec offs own theta v reads[m]! A
      refine ⟨hs'.trans hs, fun u hu s => ?_⟩
      rw [hB u (by rw [hs]; exact hu) s, hT u hu s]
      constructor
      · rintro (⟨k, hk, h⟩ | ⟨rfl, h1, h2⟩)
        · exact ⟨k, by omega, h⟩
        · exact ⟨m, by omega, ho, h1, h2⟩
      · rintro ⟨k, hk, h1, h2, h3⟩
        rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | hk
        · exact Or.inl ⟨k, hk, h1, h2, h3⟩
        · subst hk
          rw [ho] at h1
          obtain rfl := Option.some.inj h1
          exact Or.inr ⟨rfl, h2, h3⟩

theorem intoOf_spec (offs : Array ℕ) (own : Array (Option ℕ)) (theta : Array Bool) (reads : Array (Array Desc.Seg))
    (n : ℕ) :
    (Desc.intoOf offs own theta reads n).size = n ∧ ∀ u < n, ∀ s, s ∈ (Desc.intoOf offs own theta reads n)[u]! ↔
      ∃ k < own.size, own[k]! = some u ∧ s ∈ reads[k]! ∧ Coll offs own theta u s := by
  rw [intoOf_eq]
  exact intoOf_inv offs own theta reads n own.size

/-! ## The sums -/

/-- A sum over a hash set of naturals, as a sum over the finite set it holds. -/
theorem hashSet_fold_sum' (S : Std.HashSet ℕ) (p : ℕ → Bool) (f : ℕ → ℕ) :
    S.fold (fun x a => if p a = true then x else x + f a) 0 =
      ∑ a ∈ S.toList.toFinset, if p a = true then 0 else f a := by
  rw [Std.HashSet.fold_eq_foldl_toList, foldl_ite_add, zero_add]
  have hnd : S.toList.Nodup := Std.HashSet.distinct_toList.imp fun h => by simpa using h
  rw [← List.sum_toFinset _ hnd]

theorem ius_snd (c : Circuit) (dv : Division) {u : ℕ} (hu : u < dv.n) :
    (ius c dv)[u]!.2 = (tally c dv)[u]!.2.fold (fun x a => if dv.theta.contains a then x else x + c.w[a]!) 0 := by
  have hs := (tally_spec c dv).1
  rw [getElem!_pos (ius c dv) u (by simp [ius, hs, hu]), getElem!_pos (tally c dv) u (by rw [hs]; exact hu)]
  simp [ius]

/-- A segment that names a gate has a first piece. -/
theorem segGates_ps {s : Desc.Seg} {x : ℤ × ℕ} (h : x ∈ segGates s) : ∃ q, s.ps[0]? = some q := by
  rcases hps : s.ps[0]? with _ | q
  · exfalso
    have h0 : s.ps.toList = [] := by
      rcases hl : s.ps.toList with _ | ⟨q, l⟩
      · rfl
      · exfalso
        have : s.ps[0]? = some q := by rw [← Array.getElem?_toList, hl]; try rfl
        rw [hps] at this; cases this
    unfold segGates at h
    split_ifs at h
    · simp [piecesGates, h0] at h
    · simp [piecesGates, Desc.Seg.copy, Desc.shiftGates, Array.toList_map, h0] at h
  · exact ⟨q, rfl⟩

/-! ## `NovelExact` -/

/-- **`NovelExact`, from `analyze`'s reads and `novelWidth`'s union**: the segments `intoOf` collects for an IU name, with
their widths, the gates outside it that its gates read, θ's taken out, each once; `novelWidth` sums their widths, and so
does `NciCheck.ius`. -/
theorem novelExact_of (hR : ReadsExact) : NovelExact := by
  intro bytes P reads mask own c dv hdec hov hlo hin hwk han hc hms hos hdsz hmask hown hdiv hn u hu
  obtain ⟨-, h0, hK, hmono⟩ := rootLaidOut_spec P hlo
  obtain ⟨hcN, -, -, -, -⟩ := circuitOf_ok P c hc
  have hng : ∀ k g, g ∈ nodeGates P k ↔ (rootOffs P)[k]! ≤ g ∧ g < (rootOffs P)[k + 1]! := mem_nodeGates P
  have hlt : ∀ k < nodeCount P, ∀ g ∈ nodeGates P k, g < gates P := by
    intro k hk g hg
    have hg' := (hng k g).1 hg
    have hle := offs_mono (rootOffs P) (nodeCount P) hmono (k + 1) (nodeCount P) (by omega) (by omega)
    rw [← hK]; omega
  have hnode : ∀ g < gates P, ∃ k < nodeCount P, g ∈ nodeGates P k := by
    intro g hg
    obtain ⟨k, hk, h'⟩ := node_exists (rootOffs P) (nodeCount P) h0 g (by rw [hK]; exact hg)
    exact ⟨k, hk, (hng k g).2 h'⟩
  have hRk := hR bytes P reads c hdec hov hlo hin hwk han hc
  obtain ⟨-, hI⟩ := intoOf_spec (rootOffs P) own mask reads dv.n
  set segs := (Desc.intoOf (rootOffs P) own mask reads dv.n)[u]! with hsegs
  -- every gate a collected segment names, with its width, its segment and their root nodes
  have hel : ∀ x ∈ segsGates segs, ∃ a : ℕ, x = ((a : ℤ), c.w[a]!) ∧ ∃ k < nodeCount P, ∃ k' < nodeCount P,
      own[k]! = some u ∧ own[k']! ≠ some u ∧ mask[k']! = false ∧ a ∈ nodeGates P k' ∧
      ∃ s ∈ reads[k]!, s.count ≠ 0 ∧ x ∈ segGates s := by
    intro x hx
    obtain ⟨s, hs, hxs⟩ := List.mem_flatMap.1 hx
    obtain ⟨k, hk, hku, hsk, q, hq, hcnt, hown', hmask'⟩ := (hI u hu s).1 (Array.mem_def.2 hs)
    rw [hos] at hk
    obtain ⟨hk'lt, hall⟩ := (hRk k hk).1 s hsk hcnt q hq
    obtain ⟨a, rfl, ha⟩ := hall x hxs
    exact ⟨a, rfl, k, hk, _, hk'lt, hku, hown', hmask', ha, s, hsk, hcnt, hxs⟩
  have hnn : ∀ x ∈ segsGates segs, 0 ≤ x.1 := by
    intro x hx
    obtain ⟨a, rfl, -⟩ := hel x hx
    exact Int.natCast_nonneg a
  have hcons : ∀ x ∈ segsGates segs, ∀ y ∈ segsGates segs, x.1 = y.1 → x.2 = y.2 := by
    intro x hx y hy hxy
    obtain ⟨a, rfl, -⟩ := hel x hx
    obtain ⟨b, rfl, -⟩ := hel y hy
    have hab : (a : ℤ) = b := hxy
    have : a = b := by exact_mod_cast hab
    subst this; rfl
  have e1 : Desc.novelWidth segs = ∑ x ∈ (segsGates segs).toFinset, x.2 := by
    rw [Width.novelWidthExact segs hnn hcons, dedup_sum]
  have e2 : (ius c dv)[u]!.2 =
      ∑ a ∈ (tally c dv)[u]!.2.toList.toFinset, if dv.theta.contains a = true then 0 else c.w[a]! := by
    rw [ius_snd c dv hu]; exact hashSet_fold_sum' _ _ _
  have hL : ∑ x ∈ (segsGates segs).toFinset, x.2 =
      ∑ a ∈ (segsGates segs).toFinset.image (fun x => x.1.toNat), c.w[a]! := by
    rw [Finset.sum_image]
    · refine Finset.sum_congr rfl fun x hx => ?_
      obtain ⟨a, rfl, -⟩ := hel x (by simpa using hx)
      simp
    · intro x hx y hy hxy
      obtain ⟨a, rfl, -⟩ := hel x (by simpa using hx)
      obtain ⟨b, rfl, -⟩ := hel y (by simpa using hy)
      have hab : a = b := by simpa using hxy
      subst hab; rfl
  have hset : (segsGates segs).toFinset.image (fun x => x.1.toNat) =
      (tally c dv)[u]!.2.toList.toFinset.filter (fun a => dv.theta.contains a = false) := by
    ext a
    simp only [Finset.mem_image, List.mem_toFinset, Finset.mem_filter, Std.HashSet.mem_toList]
    rw [((tally_spec c dv).2 u hu).2 a]
    constructor
    · rintro ⟨x, hx, hxa⟩
      obtain ⟨b, rfl, k, hk, k', hk', hku, hk'u, hmk', hbk', s, hs, hcnt, hxs⟩ := hel x hx
      have hba : b = a := by simpa using hxa
      subst hba
      obtain ⟨g, hg, hbg, -⟩ := ((hRk k hk).2 b).1 ⟨s, hs, hcnt, _, hxs⟩
      refine ⟨⟨g, by rw [hcN]; exact hlt k hk g hg, by rw [(hdiv k hk g hg).1]; exact hku, hbg, ?_⟩, ?_⟩
      · rw [(hdiv k' hk' b hbk').1]; exact hk'u
      · rw [Bool.eq_false_iff]
        intro h
        have := (hdiv k' hk' b hbk').2.1 (Std.HashSet.mem_iff_contains.2 h)
        rw [hmk'] at this; cases this
    · rintro ⟨⟨g, hg, hgu, hag, hau⟩, hθ⟩
      rw [hcN] at hg
      obtain ⟨k, hk, hgk⟩ := hnode g hg
      have hku : own[k]! = some u := by rw [← (hdiv k hk g hgk).1]; exact hgu
      have hak : a ∉ nodeGates P k := fun h => hau (by rw [(hdiv k hk a h).1]; exact hku)
      obtain ⟨s, hs, hcnt, w, haw⟩ := ((hRk k hk).2 a).2 ⟨g, hgk, hag, hak⟩
      obtain ⟨q, hq⟩ := segGates_ps haw
      obtain ⟨hk'lt, hall⟩ := (hRk k hk).1 s hs hcnt q hq
      obtain ⟨a', ha', hak'⟩ := hall _ haw
      have haa : (a : ℤ) = a' := congrArg Prod.fst ha'
      have haa' : a = a' := by exact_mod_cast haa
      subst haa'
      have hown' : own[Desc.nodeOf (rootOffs P) q.base.toNat]! ≠ some u := fun h =>
        hau (by rw [(hdiv _ hk'lt a hak').1]; exact h)
      have hmask' : mask[Desc.nodeOf (rootOffs P) q.base.toNat]! = false := by
        rw [Bool.eq_false_iff]
        intro hm
        have := Std.HashSet.mem_iff_contains.1 ((hdiv _ hk'lt a hak').2.2 hm)
        rw [hθ] at this; cases this
      refine ⟨((a : ℤ), w), ?_, by simp⟩
      exact List.mem_flatMap.2 ⟨s, Array.mem_def.1
        ((hI u hu s).2 ⟨k, by rw [hos]; exact hk, hku, hs, q, hq, hcnt, hown', hmask'⟩), haw⟩
  rw [e1, e2, hL, hset, Finset.sum_filter]
  refine Finset.sum_congr rfl fun a _ => ?_
  cases dv.theta.contains a <;> simp

/-- **`DescAgrees`, from three statements** (`Resolve.bodiesOk` proves the second, and
`OperandsResolve.operandsResolve_of` the first from it):
`Walk.OperandsResolve` and `Resolve.BodiesOk` give `GatesLocated`
(`Resolve.operandsEarlier_of`, `Walk.gatesLocated_of`), and `ReadsExact` gives `NovelExact` (`novelExact_of`). -/
theorem descAgrees_of_open (hO : Walk.OperandsResolve) (hB : Resolve.BodiesOk) (hR : ReadsExact) :
    Properties.DescAgrees :=
  Walk.descAgrees_of_gaps hO (Resolve.operandsEarlier_of hB) (novelExact_of hR)

end NciCheck.Properties.SecurityProofs.Novel
