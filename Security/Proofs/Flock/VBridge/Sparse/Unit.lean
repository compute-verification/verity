import Flock.RecSparse

/-!
# The verifier's `rec-sparse/…` unit

`RecSparse.check` accepts a `rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>` unit only if it is the net the verifier builds
itself at the shape its name gives (`RecSparse.unitNet`, `rec_sparse.lowering` as `Flock.RecSparse` states it), and
`HmRow.parse` calls it on every circuit (`ParseFacts.recSparse`). `check_sparse` states what it accepted.
-/

namespace FlockVBridge.Sparse

open Flock

/-- `params` gives `sh` only for the name `unitName sh`. -/
theorem unitName_params {name : String} {sh : RecSparse.Shape} (h : RecSparse.params name = some sh) :
    RecSparse.unitName sh = name := by
  unfold RecSparse.params at h
  repeat' (first | split at h | dsimp only at h)
  all_goals first
    | (simp only [Option.some.injEq] at h; subst h; assumption)
    | cases h

theorem check_sparse_all {c : Circuit} (h : RecSparse.check c = .ok ()) {sh : RecSparse.Shape}
    (hp : RecSparse.params c.unit.name = some sh) :
    sh.valid = true ∧ RecOpen.inBits c.unit = 16 * RecSparse.inWords sh ∧
      RecOpen.outBits c.unit = 16 * RecSparse.outWords sh ∧ RecSparse.rowsOk c sh = true ∧
      RecSparse.leavesOk c sh = true ∧ RecSparse.unitNet sh = .ok c.unit := by
  unfold RecSparse.check at h
  rw [hp] at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hv
    split at h
    · cases h
    · split at h
      · cases h
      · rename_i hb
        split at h
        · cases h
        · rename_i hr
          split at h
          · cases h
          · rename_i hl
            split at h
            · cases h
            · rename_i g hg
              split at h
              · rename_i hcg
                simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hb
                exact ⟨by simpa using hv, hb.1, hb.2, by simpa using hr, by simpa using hl, by rw [hcg]; exact hg⟩
              · cases h

/-- **What `RecSparse.check` accepted**: a unit whose name gives the shape `sh` is at a valid shape, has the shape's
input and output bits, and is the net the verifier builds at `sh`. -/
theorem check_sparse {c : Circuit} (h : RecSparse.check c = .ok ()) {sh : RecSparse.Shape}
    (hp : RecSparse.params c.unit.name = some sh) :
    sh.valid = true ∧ RecOpen.inBits c.unit = 16 * RecSparse.inWords sh ∧
      RecOpen.outBits c.unit = 16 * RecSparse.outWords sh ∧ RecSparse.unitNet sh = .ok c.unit :=
  let ⟨hv, hi, ho, _, _, hn⟩ := check_sparse_all h hp
  ⟨hv, hi, ho, hn⟩

/-- **The statement's rows are the unit's inputs** (`RecSparse.check`'s `rowsOk` and `leavesOk`): its first rows are
`ports sh` at their widths, and every copy reads their words in order. -/
theorem check_sparse_rows {c : Circuit} (h : RecSparse.check c = .ok ()) {sh : RecSparse.Shape}
    (hp : RecSparse.params c.unit.name = some sh) :
    RecSparse.rowsOk c sh = true ∧ RecSparse.leavesOk c sh = true :=
  let ⟨_, _, _, hr, hl, _⟩ := check_sparse_all h hp
  ⟨hr, hl⟩

theorem messagePorts_head (sh : RecSparse.Shape) : ∃ rest, RecSparse.messagePorts sh = ("h", 64) :: rest :=
  ⟨_, by simp [RecSparse.messagePorts, RecSparse.SKIP]; rfl⟩

/-- `h` is the first port, at 512 words. -/
theorem ports_head (sh : RecSparse.Shape) : ∃ rest, RecSparse.ports sh = ("h", 512) :: rest :=
  ⟨_, by simp [RecSparse.ports, RecSparse.messagePorts, RecSparse.portWords, RecSparse.SKIP, RecSparse.F_WORDS,
    RecSparse.ROW_BLOCK]; rfl⟩

theorem h_le_inWords (sh : RecSparse.Shape) : 512 ≤ RecSparse.inWords sh := by
  obtain ⟨rest, hr⟩ := ports_head sh
  have h : ∀ (l : List (String × Nat)) (a : Nat), a ≤ l.foldl (fun a p => a + p.2) a := by
    intro l
    induction l with
    | nil => intro a; exact Nat.le_refl a
    | cons p l ih => intro a; exact Nat.le_trans (Nat.le_add_right a p.2) (ih _)
  unfold RecSparse.inWords
  rw [hr, List.foldl_cons]
  exact h rest _

/-- **`rowsOk`'s row 0**: the statement's first row is `h`, 512 words and as many bits as its words hold. -/
theorem rowsOk_h {c : Circuit} {sh : RecSparse.Shape} (h : RecSparse.rowsOk c sh = true) :
    ∃ h0 : 0 < c.ports.size, c.ports[0].name = "h" ∧ c.ports[0].words = 512 ∧ c.ports[0].bits = 16 * 512 := by
  obtain ⟨rest, hr⟩ := ports_head sh
  unfold RecSparse.rowsOk at h
  have e := congrArg (·[0]?) (of_decide_eq_true h)
  simp only [hr] at e
  by_cases h0 : 0 < c.ports.size
  · refine ⟨h0, ?_⟩
    rw [Array.getElem?_map, Array.getElem?_extract] at e
    simp [h0] at e
    exact ⟨e.2.1, by omega, e.2.2.2⟩
  · simp [Nat.pos_iff_ne_zero] at h0
    simp [h0] at e

theorem leavesOk_get {c : Circuit} {sh : RecSparse.Shape} (h : RecSparse.leavesOk c sh = true) {j : Nat}
    (hj : j < c.upv) : c.leavesIn.getD j #[] = (Array.range (RecSparse.inWords sh)).map Int.ofNat := by
  unfold RecSparse.leavesOk at h
  exact of_decide_eq_true (List.all_eq_true.1 h j (List.mem_range.2 hj))

/-- The verifier's unit's inputs: one group of `inWords` 16-bit ports from column 0. -/
theorem inGroups_sparse {name : String} {sh : RecSparse.Shape} {net : Net}
    (hl : HmNets.netOf name (RecSparse.unitLaid sh) = .ok net) :
    net.inGroups = #[⟨0, HmNets.groupWords ((Array.replicate (RecSparse.inWords sh) 16).foldl (· + ·) 0),
      Array.replicate (RecSparse.inWords sh) 16⟩] := by
  unfold HmNets.netOf at hl
  rw [(Net.ofRows_ok hl).2.2.2.2.2.1]
  rfl

end FlockVBridge.Sparse
