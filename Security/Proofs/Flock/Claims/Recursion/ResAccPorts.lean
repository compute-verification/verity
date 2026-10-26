import Proofs.Flock.Claims.Recursion.ResLowerD
import Proofs.Flock.C3.PubRows.Check

/-!
# P3's halves that read no other part: a part's end reads, and two chains with one step

`AccPorts` (P3) asks that a piece's copy read the accumulators `elemReadC (Top.accRC rd key)` reads, up to one offset
per component. The copy's `acc0` and `acc1` are rows of the outer witness's own `rec-acc` chain; `accRC` reads
`Top.chainC`'s. Two facts need only the part's own checks or the chain's steps:

* **Where the copy reads** (`ends_of_checkAccReads`, `endReads_inner`): at a statement that is no level,
  `checkAccReads` holds its `acc0` and `acc1` to `endReads`' rows (`RecVStar.checkEndReads`); at a part of the table's
  algebra, `acc0` reads rows `0` and `n` and `acc1` rows `n` and `2 · n` (`n` openings a rep): rep `r`'s ends.
* **Two chains with one step** (`chainHolds_off`): two chains that carry row D's running sums (`Top.ChainHolds`) differ
  along each rep's run of table `t`'s chain by one constant per component, since each step (`consTerm`) adds one term
  to both (`off_of_steps`), and the rep's pieces step every position of its run (`pos_split`).
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)
open FlockSoundness.Refine (fOf fOf_add fOf_injective)

/-! ## Where a part's copy reads the chain -/

/-- **`endReads` at a part of the table's algebra**: `acc0` reads rows `0` and `n`, `acc1` rows `n` and `2 · n`. -/
theorem endReads_inner {i : RecVStar.Inner} {sch : Schedule} (hsch : fast100 i.m = some sch) (j : ℕ) (nm : String)
    (pt : VStar.Part) : RecVStar.endReads i (.inner j nm pt) =
      .ok (some (#[some 0, some (RecVStar.opensBelow sch sch.levels.size)],
        #[some (RecVStar.opensBelow sch sch.levels.size), some (2 * RecVStar.opensBelow sch sch.levels.size)])) := by
  simp only [RecVStar.endReads, hsch]
  rfl

/-- A part of the table's algebra is no level: `accReads` has no rows for it. -/
theorem accReads_inner (i : RecVStar.Inner) (j : ℕ) (nm : String) (pt : VStar.Part) :
    RecVStar.accReads i (.inner j nm pt) = .ok none := rfl

/-- **`checkAccReads` at a statement that is no level**: each of its `accPorts` is `own`'s first entry of its name, and
reads `endReads`' rows of the name (`endWant`). -/
theorem ends_of_checkAccReads {i : RecVStar.Inner} {s : VStar.Stmt} {own : Array Registered.Port}
    (h : RecVStar.checkAccReads i s own = .ok ()) (hn : RecVStar.accReads i s = .ok none)
    {w : Array (Option ℕ) × Array (Option ℕ)} (hw : RecVStar.endReads i s = .ok (some w)) :
    ∀ nm ∈ RecVStar.accPorts s, ∃ p, own.find? (·.name == nm) = some p ∧ p.reads = RecVStar.endWant w nm := by
  unfold RecVStar.checkAccReads at h
  rw [hn] at h
  obtain ⟨_, h0, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h0
  dsimp only at h
  unfold RecVStar.checkEndReads at h
  split at h
  · rename_i he
    intro nm hnm
    rw [List.isEmpty_iff.mp he] at hnm
    cases hnm
  · rw [hw] at h
    obtain ⟨_, h1, h⟩ := FlockSoundness.Refine.ebind_ok h
    cases h1
    dsimp only at h
    obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
    refine FlockC3.PubRows.rows_forIn_each _ ?_ _ h
    intro nm r hr
    dsimp only at hr
    split at hr
    · rename_i p hf
      split at hr
      · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
        cases h2
      · rename_i hne
        refine ⟨⟨p, hf, by simpa using hne⟩, ?_⟩
        cases hr
        rfl
    · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
      cases h2

/-! ## Two chains with one step -/

/-- **A step adds one term**: `consTerm`'s component `k` is the running sum's plus the opening's terms. -/
theorem fOf_consTerm (LANES : ℕ) (r a c : Array F128) (k : ℕ) :
    fOf (FlockVBridge.consTerm LANES r a c k) =
      fOf a[k]! + ∑ l ∈ Finset.range LANES, fOf (c[k * LANES + l]! * r[l]!) := by
  simp only [FlockVBridge.consTerm]
  exact FlockVBridge.fOf_foldl_add (fun l => c[k * LANES + l]! * r[l]!) LANES a[k]!

/-- **Two chains stepped alike from `base` differ by one constant**: wherever both take each of the `N` steps after
`base` by one `consTerm` (one row, one `coef`), each of their first `N + 1` rows' component `k` is the other's plus
their difference at `base`. -/
theorem off_of_steps {A B : ℕ → Array F128} {k base N : ℕ}
    (hstep : ∀ j < N, ∃ (LANES : ℕ) (r c : Array F128),
      (A (base + j + 1))[k]! = FlockVBridge.consTerm LANES r (A (base + j)) c k ∧
      (B (base + j + 1))[k]! = FlockVBridge.consTerm LANES r (B (base + j)) c k) :
    ∀ j ≤ N, (A (base + j))[k]! = (B (base + j))[k]! + ((A base)[k]! + (B base)[k]!)
  | 0, _ => by
    rw [Nat.add_zero]
    apply fOf_injective
    rw [fOf_add, fOf_add, ← add_assoc, add_comm (fOf (B base)[k]!) (fOf (A base)[k]!), add_assoc,
      CharTwo.add_self_eq_zero, add_zero]
  | j + 1, hj => by
    obtain ⟨L, r, c, hA, hB⟩ := hstep j (by omega)
    have ih := off_of_steps hstep j (by omega)
    rw [← Nat.add_assoc, hA, hB]
    apply fOf_injective
    simp only [fOf_add, fOf_consTerm]
    rw [ih]
    simp only [fOf_add]
    ring

/-- **A position of blocks `f 0, f 1, …` laid end to end** is one block's and one offset in it. -/
theorem pos_split (f : ℕ → ℕ) : ∀ (L p : ℕ), p < ((List.range L).map f).sum →
    ∃ ℓ < L, ∃ j < f ℓ, ((List.range ℓ).map f).sum + j = p
  | 0, p, h => by simp at h
  | L + 1, p, h => by
    rw [List.range_succ, List.map_append, List.sum_append] at h
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero] at h
    by_cases hp : p < ((List.range L).map f).sum
    · obtain ⟨ℓ, hℓ, j, hj, he⟩ := pos_split f L p hp
      exact ⟨ℓ, by omega, j, hj, he⟩
    · exact ⟨L, by omega, p - ((List.range L).map f).sum, by omega, by omega⟩

/-- **Two chains that carry row D's running sums differ by one constant along a rep's run**: at a setup, along rep
`r`'s run of table `t`'s chain (rows `r · N` to `r · N + N`, `N` the rep's openings at the table's schedule), each
row's component `k` of `A` is `B`'s plus their difference at the run's start. Each position of the run is a piece's
(`pos_split`, `mem_rowDPiecesOf`), whose `accHolds` steps both chains by one row and one `coef`. -/
theorem chainHolds_off {rd : Msg → Option FinalP} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ} {x : Msg}
    {ms : List Msg} {cs : List Coin} {A B : ℕ → ℕ → Array F128} (hA : Top.ChainHolds rd key x ms cs A)
    (hB : Top.ChainHolds rd key x ms cs B) {v} (hv : setupOf false x = some v)
    {t : ℕ} (ht : t < v.2.2.size) {r : ℕ} (hr : r < 2) {k : ℕ} (hk : k < 2) :
    ∀ p ≤ Rows.nOpen (Rows.schOf x t),
      (A t (r * Rows.nOpen (Rows.schOf x t) + p))[k]! =
        (B t (r * Rows.nOpen (Rows.schOf x t) + p))[k]! +
          ((A t (r * Rows.nOpen (Rows.schOf x t)))[k]! + (B t (r * Rows.nOpen (Rows.schOf x t)))[k]!) := by
  have hsch : Rows.schOf x t = Rows.schAt v.2.2 t := by simp only [Rows.schOf, Rows.schAt, hv]
  refine off_of_steps (A := A t) (B := B t) fun p hp => ?_
  obtain ⟨ℓ, hℓ, j, hj, he⟩ := pos_split (fun ℓ => ((Rows.schOf x t).levels.getD ℓ default).q)
    (Rows.schOf x t).levels.size p hp
  dsimp only at hj he
  have hq : (⟨t, r, ℓ, j, key t r ℓ j⟩ : OpenPiece) ∈ Rows.rowDPieces key x := by
    simp only [Rows.rowDPieces, hv]
    rw [Rows.mem_rowDPiecesOf]
    rw [hsch] at hℓ hj
    refine ⟨ht, hr, hℓ, ?_, rfl⟩
    rwa [← Array.getElem!_eq_getD] at hj
  have hpos : Rows.chainPos (Rows.schOf x t) r ℓ j = r * Rows.nOpen (Rows.schOf x t) + p := by
    simp only [Rows.chainPos]
    rw [Nat.add_assoc, he]
  have hA' := hA _ hq k hk
  have hB' := hB _ hq k hk
  have hs : (Rows.sideAt rd x ms ⟨t, r, ℓ, j, key t r ℓ j⟩).sch = Rows.schOf x t := rfl
  dsimp only at hA' hB'
  rw [hs, hpos] at hA' hB'
  exact ⟨_, _, _, hA', hB'⟩

end RecDeploy.ResLower
