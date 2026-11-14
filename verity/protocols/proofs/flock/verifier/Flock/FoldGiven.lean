/-
The statement's lincheck fold with a unit net's walks given (`Stmt.foldedGiven`): `Stmt.folded` (`foldedGiven_eq`) for walks
that are the statement's at their points (`WalksOk`), which those of one pass over each net for two points are
(`walks2_ok`). The walks are `Aᵀe_lo` and `Bᵀe_lo`, all of a slot type's fold that reads the net's rows; a point with no
walks is folded as `folded` folds it.
-/
module

public import Flock.Statement
public import Flock.FoldTwo

@[expose] public section

namespace Flock

namespace Stmt

/-- `Bᵀe_lo` with the product rows' B side read from the table for a lookup slot, as `baseOf` reads it. -/
def withLookup (lk : Option LookupTable) (eLo fb : Array F128) : Array F128 :=
  match lk with
  | some t => Lookup.foldB t eLo fb
  | none => fb

/-- `baseOf` from its two walks (`baseOf_eq_walks`), `fa = Aᵀe_lo` and `fb = Bᵀe_lo` (`withLookup`), as `baseOfFast`
computes it. -/
def baseOfWalks (alpha : F128) (fa fb : Array F128) (sl : Nat) : Array F128 :=
  if 2 ^ sl < 2048 then (Array.range (2 ^ sl)).map fun i => alpha * fa[i]! + fb[i]!
  else
    let ta := F128.mulTableL alpha
    (Array.range (2 ^ sl)).map fun i => F128.mulBy ta fa[i]! + fb[i]!

theorem baseOf_eq_walks (alpha : F128) (eLo : Array F128) (a b : Sparse) (lk : Option LookupTable) (sl : Nat) :
    baseOf alpha eLo a b lk sl = baseOfWalks alpha (a.foldT eLo) (withLookup lk eLo (b.foldT eLo)) sl := by
  rw [baseOf_eq_fast]
  cases lk <;> rfl

/-- `f` of a mapped array's entry, read past its end too, for an `f` that keeps the default. -/
private theorem getElem!_map_fix {α β : Type} [Inhabited α] [Inhabited β] (f : α → β) (hf : f default = default)
    (a : Array α) (i : Nat) : (a.map f)[i]! = f a[i]! := by
  by_cases h : i < a.size
  · rw [getElem!_pos (a.map f) i (by simpa using h), getElem!_pos a i h, Array.getElem_map]
  · rw [getElem!_neg (a.map f) i (by simpa using h), getElem!_neg a i h, hf]

/-- `baseOfWalks` on the walks the cells `ca` and `cb` hold under `f` (`baseOfBy_eq`), read from the cells: no array of
either walk is made. -/
@[inline] def baseOfBy (f : Sparse.Cell2 → F128) (alpha : F128) (ca cb : Array Sparse.Cell2) (sl : Nat) : Array F128 :=
  if 2 ^ sl < 2048 then (Array.range (2 ^ sl)).map fun i => alpha * f ca[i]! + f cb[i]!
  else
    let ta := F128.mulTableL alpha
    (Array.range (2 ^ sl)).map fun i => F128.mulBy ta (f ca[i]!) + f cb[i]!

theorem baseOfBy_eq (f : Sparse.Cell2 → F128) (hf : f default = default) (alpha : F128) (ca cb : Array Sparse.Cell2)
    (sl : Nat) : baseOfBy f alpha ca cb sl = baseOfWalks alpha (ca.map f) (cb.map f) sl := by
  simp only [baseOfBy, baseOfWalks, getElem!_map_fix f hf]

/-- Every base of `bases` is its slot type's (`slotBase`) at `(α, lw, ρ)`: what `partialRangesT` keeps of the bases it
has folded. -/
def BasesOk (st : Stmt) (alpha : F128) (lw rho : Array F128) (bases : Array ((Kind × Nat) × Array F128)) : Prop :=
  ∀ kb ∈ bases, st.slotBase alpha lw rho kb.1 = .ok kb.2

theorem basesOk_empty (st : Stmt) (alpha : F128) (lw rho : Array F128) : st.BasesOk alpha lw rho #[] := by
  intro kb h
  simp at h

theorem basesOk_push (st : Stmt) (alpha : F128) (lw rho : Array F128) {bases : Array ((Kind × Nat) × Array F128)}
    (h : st.BasesOk alpha lw rho bases) {key : Kind × Nat} {b : Array F128} (hb : st.slotBase alpha lw rho key = .ok b) :
    st.BasesOk alpha lw rho (bases.push (key, b)) := by
  intro kb hm
  rcases Array.mem_push.mp hm with hm | rfl
  · exact h kb hm
  · exact hb

/-- A slot type's base from `bases`, folded and kept where `bases` has none: `partialRangesT`'s step
(`partialRangesT_cons`). -/
def baseAt (st : Stmt) (alpha : F128) (lw rho : Array F128) (bases : Array ((Kind × Nat) × Array F128))
    (key : Kind × Nat) : V (Array F128 × Array ((Kind × Nat) × Array F128)) :=
  match bases.find? (fun kb => decide (kb.1 = key)) with
  | some (_, b) => pure (b, bases)
  | none => do
    let b ← st.slotBase alpha lw rho key
    pure (b, bases.push (key, b))

/-- From bases that are their slot types', `baseAt` gives `slotBase`'s answer, beside bases that are still theirs. -/
theorem baseAt_ok (st : Stmt) (alpha : F128) (lw rho : Array F128) {bases : Array ((Kind × Nat) × Array F128)}
    (h : st.BasesOk alpha lw rho bases) (key : Kind × Nat) :
    (∃ e, st.baseAt alpha lw rho bases key = .error e ∧ st.slotBase alpha lw rho key = .error e) ∨
    (∃ b bases', st.baseAt alpha lw rho bases key = .ok (b, bases') ∧ st.slotBase alpha lw rho key = .ok b ∧
      st.BasesOk alpha lw rho bases') := by
  unfold baseAt
  cases hf : bases.find? (fun kb => decide (kb.1 = key)) with
  | some kb =>
    obtain ⟨k, b⟩ := kb
    have hk : k = key := of_decide_eq_true
      (Array.find?_some (p := fun (kb : (Kind × Nat) × Array F128) => decide (kb.1 = key)) hf)
    have hm := h _ (Array.mem_of_find?_eq_some hf)
    exact .inr ⟨b, bases, rfl, hk ▸ hm, h⟩
  | none =>
    cases hs : st.slotBase alpha lw rho key with
    | error e => exact .inl ⟨e, rfl, rfl⟩
    | ok b => exact .inr ⟨b, _, rfl, rfl, st.basesOk_push alpha lw rho h hs⟩

theorem partialRangesT_cons (st : Stmt) (alpha : F128) (lw rho rr : Array F128) (r : Range) (rs : List Range)
    (bases : Array ((Kind × Nat) × Array F128)) (out : Array F128) :
    st.partialRangesT alpha lw rho rr (r :: rs) bases out =
      (match st.unitShare alpha lw rho rr r with
        | some p => pure (p, bases)
        | none => do
          let (base, bases') ← st.baseAt alpha lw rho bases (r.kind, r.slotLog)
          pure (project base (eqTable (rr.extract 0 (r.slotLog - 6))), bases') :
        V (Array F128 × Array ((Kind × Nat) × Array F128))) >>= fun pb =>
      st.partialRangesT alpha lw rho rr rs pb.2 (Array.ofFn (n := 64) fun s => out[s.val]! +
        (sumUpTo r.count fun k => (eqTable (rho.extract (r.slotLog - 6) rho.size))[r.pos0 + k]! *
          (eqTable (rr.extract (r.slotLog - 6) rr.size))[r.pos0 + k]!) * pb.1[s.val]!) := by
  rw [partialRangesT]
  unfold baseAt
  cases st.unitShare alpha lw rho rr r with
  | some p => rfl
  | none =>
    dsimp only
    cases bases.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
    | some kb =>
      obtain ⟨k, b⟩ := kb
      simp only [pure_bind]
    | none => simp only [pure_bind, bind_assoc]

/-- `partialRangesT` gives one answer from any bases that are their slot types'. -/
theorem partialRangesT_bases (st : Stmt) (alpha : F128) (lw rho rr : Array F128) (rs : List Range)
    (b1 b2 : Array ((Kind × Nat) × Array F128)) (out : Array F128) (h1 : st.BasesOk alpha lw rho b1)
    (h2 : st.BasesOk alpha lw rho b2) :
    st.partialRangesT alpha lw rho rr rs b1 out = st.partialRangesT alpha lw rho rr rs b2 out := by
  induction rs generalizing b1 b2 out with
  | nil => rfl
  | cons r rs ih =>
    rw [partialRangesT_cons, partialRangesT_cons]
    cases st.unitShare alpha lw rho rr r with
    | some p => exact ih b1 b2 _ h1 h2
    | none =>
      rcases st.baseAt_ok alpha lw rho h1 (r.kind, r.slotLog) with ⟨e, he, hs⟩ | ⟨b, b1', hb, hs, hok1⟩
      · rcases st.baseAt_ok alpha lw rho h2 (r.kind, r.slotLog) with ⟨e', he', hs'⟩ | ⟨b', b2', hb', hs', _⟩
        · rw [hs] at hs'
          cases hs'
          simp only [he, he']
        · rw [hs] at hs'
          cases hs'
      · rcases st.baseAt_ok alpha lw rho h2 (r.kind, r.slotLog) with ⟨e', he', hs'⟩ | ⟨b', b2', hb', hs', hok2⟩
        · rw [hs] at hs'
          cases hs'
        · rw [hs] at hs'
          cases hs'
          simp only [hb, hb']
          exact ih b1' b2' _ hok1 hok2

/-- `foldedPartial` (`foldedPartialFrom_eq`) with some slot types' bases given; `foldedPartial` starts from none. -/
def foldedPartialFrom (st : Stmt) (bases : Array ((Kind × Nat) × Array F128)) (alpha beta : F128)
    (lw rho ts : Array F128) : V (Array F128) := do
  let kl := st.c.kLog
  if rho.size + 6 != kl || lw.size != 64 then throw "lincheck fold: the point is not the block's"
  let rr := ts.reverse
  let out ← st.partialRangesT alpha lw rho rr st.c.ranges.toList bases (Array.replicate 64 0)
  let n := 2 ^ kl
  let eIn := eqTable rho
  let eR := eqTable rr
  let e := fun (i : Nat) => lw[i % 64]! * eIn[i / 64]!
  let add := fun (w : Nat → F128) (o : Array F128) (ic : Nat × Nat) =>
    if ic.2 < n then o.modify (ic.2 % 64) (· + w ic.1 * eR[ic.2 / 64]!) else o
  let out := st.db.toList.foldl (add e) (st.da.toList.foldl (add fun i => alpha * e i) out)
  return if st.pin < n then out.modify (st.pin % 64) (· + beta * eR[st.pin / 64]!) else out

theorem foldedPartialFrom_eq (st : Stmt) {bases : Array ((Kind × Nat) × Array F128)} (alpha beta : F128)
    (lw rho ts : Array F128) (h : st.BasesOk alpha lw rho bases) :
    st.foldedPartialFrom bases alpha beta lw rho ts = st.foldedPartial alpha beta lw rho ts := by
  have key := fun rr out => st.partialRangesT_bases alpha lw rho rr st.c.ranges.toList bases #[] out h
    (st.basesOk_empty alpha lw rho)
  simp only [foldedPartialFrom, foldedPartial, key]

/-- A net's slot type's base at one point, as a function of `α`: `baseOfWalks α` on the net's two walks there
(`Walked.Ok`), which a pass has finished and `fold` holds. -/
structure Walked where
  net : Nat
  fold : F128 → Array F128

/-- `w` folds the walks of the statement's net `w.net` on the local tensor of the point `(lw, ρ)`: `Aᵀe_lo`, and `Bᵀe_lo`
as `baseOf` reads it (`withLookup`). -/
def Walked.Ok (st : Stmt) (lw rho : Array F128) (w : Walked) : Prop :=
  let n := (st.c.nets.getD w.net default).2
  let eLo := tensor lw (rho.extract 0 (n.unitLog - 6))
  ∀ alpha, w.fold alpha = baseOfWalks alpha (n.a.foldT eLo) (withLookup n.lookup eLo (n.b.foldT eLo)) n.unitLog

/-- The net's slot type and its base, from the walks. -/
def Walked.base (st : Stmt) (alpha : F128) (w : Walked) : (Kind × Nat) × Array F128 :=
  ((.net w.net, (st.c.nets.getD w.net default).2.unitLog), w.fold alpha)

theorem Walked.base_ok (st : Stmt) (alpha : F128) (lw rho : Array F128) {w : Walked} (h : w.Ok st lw rho) :
    st.slotBase alpha lw rho (w.base st alpha).1 = .ok (w.base st alpha).2 := by
  simp only [Walked.base, slotBase, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, baseOf_eq_walks, h alpha]
  rfl

theorem basesOk_walked (st : Stmt) (alpha : F128) (lw rho : Array F128) {ws : Array Walked}
    (h : ∀ w ∈ ws, w.Ok st lw rho) : st.BasesOk alpha lw rho (ws.map (Walked.base st alpha)) := by
  intro kb hm
  obtain ⟨w, hw, rfl⟩ := Array.mem_map.mp hm
  exact Walked.base_ok st alpha lw rho (h w hw)

/-- Walks finished ahead of the fold: a point `(lw, ρ)` the fold may be asked at, and the nets' walks there. -/
abbrev Walks := Array ((Array F128 × Array F128) × Array Walked)

/-- Every entry of `ws` holds the statement's walks at its point. -/
def WalksOk (st : Stmt) (ws : Walks) : Prop := ∀ e ∈ ws, ∀ w ∈ e.2, w.Ok st e.1.1 e.1.2

/-- Whether two vectors are equal, entry by entry. -/
def sameVec (a b : Array F128) : Bool :=
  a.size == b.size && (List.range a.size).all fun i => decide (a[i]! = b[i]!)

theorem eq_of_sameVec {a b : Array F128} (h : sameVec a b = true) : a = b := by
  simp only [sameVec, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range, decide_eq_true_eq] at h
  apply Array.ext h.1
  intro i h1 h2
  have := h.2 i h1
  rwa [getElem!_pos a i h1, getElem!_pos b i h2] at this

/-- `folded` (`foldedGiven_eq`), a structured block's slot types starting from the bases of the walks `ws` holds at the
point. The walks are a `Thunk`, read only when a structured block is folded; a point `ws` does not hold is folded as
`folded` folds it, so what fills `ws` chooses its points freely. -/
def foldedGiven (st : Stmt) (ws : Thunk Walks) (alpha beta : F128) (lw rho ts : Array F128) : V (Array F128) :=
  if st.structured && ts.size + 6 == st.c.kLog then
    -- No theorem says `ws` holds the points the replay asks at, and none is needed: a hit is the walk stored under
    -- this very point (`WalksOk`), and a miss runs the plain walk, so a wrong guess of the points costs time only.
    match ws.get.find? (fun e => sameVec e.1.1 lw && sameVec e.1.2 rho) with
    | some e => st.foldedPartialFrom (e.2.map (Walked.base st alpha)) alpha beta lw rho ts
    | none => st.foldedPartial alpha beta lw rho ts
  else do
    let v ← st.fold alpha lw rho
    pure (ts.foldl (fun c t => halve t c) (v.modify st.pin (· + beta)))

theorem foldedGiven_eq (st : Stmt) {ws : Thunk Walks} (h : st.WalksOk ws.get) : st.foldedGiven ws = st.folded := by
  funext alpha beta lw rho ts
  unfold foldedGiven folded
  split
  · split
    · rename_i e he
      have hp := Array.find?_some
        (p := fun (e : (Array F128 × Array F128) × Array Walked) => sameVec e.1.1 lw && sameVec e.1.2 rho) he
      simp only [Bool.and_eq_true] at hp
      have hw := h e (Array.mem_of_find?_eq_some he)
      rw [eq_of_sameVec hp.1, eq_of_sameVec hp.2] at hw
      exact st.foldedPartialFrom_eq alpha beta lw rho ts (st.basesOk_walked alpha lw rho hw)
    · rfl
  · rfl

/-- `circuitFold` (`circuitFoldGiven_eq`) with `folded` reading the walks `ws`. -/
def circuitFoldGiven (st : Stmt) (ws : Thunk Walks) : CircuitFold :=
  { kLog := st.c.kLog, pin := st.pin, fold := st.fold, folded := st.foldedGiven ws }

theorem circuitFoldGiven_eq (st : Stmt) {ws : Thunk Walks} (h : st.WalksOk ws.get) :
    st.circuitFoldGiven ws = st.circuitFold := by
  unfold circuitFoldGiven circuitFold
  rw [st.foldedGiven_eq h]

/-- The nets a range's slot type folds by its rows (`slotBase`): each net a range names at its own slot size, once. None
for a templated statement, whose unit is folded from its template (`unitShare`) and never walked. -/
def walkNets (st : Stmt) : Array Nat :=
  if st.c.tmpl.isSome then #[]
  else st.c.ranges.foldl (fun acc r => match r.kind with
    | .net i => if (st.c.nets.getD i default).2.unitLog == r.slotLog && !acc.contains i then acc.push i else acc
    | _ => acc) #[]

/-- Net `i`'s bases at the points `p0` and `p1`, each matrix read in one pass for both. Where the net has no lookup the
bases are read from the pass's cells (`Sparse.foldC2`, `baseOfBy`); a lookup's table reads `Bᵀe_lo` as an array, so there
the cells are split first (`Sparse.foldT2`). -/
def walkPair (st : Stmt) (p0 p1 : Array F128 × Array F128) (i : Nat) : Walked × Walked :=
  let n := (st.c.nets.getD i default).2
  let e0 := tensor p0.1 (p0.2.extract 0 (n.unitLog - 6))
  let e1 := tensor p1.1 (p1.2.extract 0 (n.unitLog - 6))
  if e0.size = e1.size ∧ n.lookup = none then
    let ca := n.a.foldC2 e0 e1
    let cb := n.b.foldC2 e0 e1
    (⟨i, fun alpha => baseOfBy Sparse.Cell2.fst alpha ca cb n.unitLog⟩,
      ⟨i, fun alpha => baseOfBy Sparse.Cell2.snd alpha ca cb n.unitLog⟩)
  else
    let fa := n.a.foldT2 e0 e1
    let fb := n.b.foldT2 e0 e1
    let fb0 := withLookup n.lookup e0 fb.1
    let fb1 := withLookup n.lookup e1 fb.2
    (⟨i, fun alpha => baseOfWalks alpha fa.1 fb0 n.unitLog⟩, ⟨i, fun alpha => baseOfWalks alpha fa.2 fb1 n.unitLog⟩)

theorem walkPair_ok (st : Stmt) (p0 p1 : Array F128 × Array F128) (i : Nat) :
    (st.walkPair p0 p1 i).1.Ok st p0.1 p0.2 ∧ (st.walkPair p0 p1 i).2.Ok st p1.1 p1.2 := by
  unfold walkPair
  dsimp only
  split
  · rename_i h
    obtain ⟨hs, hl⟩ := h
    constructor
    · intro alpha
      simp only [baseOfBy_eq Sparse.Cell2.fst rfl, Sparse.foldC2_fst, hl, withLookup]
    · intro alpha
      simp only [baseOfBy_eq Sparse.Cell2.snd rfl, Sparse.foldC2_snd _ hs, hl, withLookup]
  · constructor
    · intro alpha
      simp only [Sparse.foldT2_eq]
    · intro alpha
      simp only [Sparse.foldT2_eq]

/-- The walks of the statement's nets (`walkNets`) at two points, one pass over each matrix. -/
def walks2 (st : Stmt) (p0 p1 : Array F128 × Array F128) : Walks :=
  let ps := st.walkNets.map (st.walkPair p0 p1)
  #[(p0, ps.map (·.1)), (p1, ps.map (·.2))]

theorem walks2_ok (st : Stmt) (p0 p1 : Array F128 × Array F128) : st.WalksOk (st.walks2 p0 p1) := by
  intro e he w hw
  simp only [walks2, List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl
  · simp only [Array.mem_map] at hw
    obtain ⟨pr, ⟨i, _, rfl⟩, rfl⟩ := hw
    exact (st.walkPair_ok p0 p1 i).1
  · simp only [Array.mem_map] at hw
    obtain ⟨pr, ⟨i, _, rfl⟩, rfl⟩ := hw
    exact (st.walkPair_ok p0 p1 i).2

end Stmt

end Flock
