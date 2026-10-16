import Proofs.Flock.Claims.Scope.Fit

/-!
# The deployed relation splits by table

Each table of a session is its own statement with its own units: the verifier sets up one `Stmt` per table
(`Stmt.setupTables`), and no hidden value of one table is read by another's blocks. In the readable relation
(`relation`) table `c` reads only its own share of the hidden values (`zOf`, the bits `a·J + c`).

* `tableWit J w j`: table `j`'s share of `w`'s hidden values, as a witness of its own.
* `relS_congr`: `RelS` reads a witness only through its hidden values' `getD`.
* `relS_only`: at a statement the verifier parses (untyped, without public input ports), `RelS` of the readable
  statement holds exactly when, for every table `j`, `RelS` of table `j` alone (`Parsed.only`) holds at its share.
  Every table fits the first table's blocks (`tables_fit`), so a table fits its own blocks exactly when it fits the
  session's.
-/

namespace Flock.Readable

/-- **Table `j`'s share** of the hidden values of `J` tables (its bit `a` is hidden value `a·J + j`), as a witness of its
own. -/
def tableWit (J : ℕ) (w : Claims.Flock.Wit) (j : ℕ) : Claims.Flock.Wit :=
  ⟨(List.range w.hid.length).map fun a => w.hid.getD (a * J + j) false⟩

theorem tableWit_getD {J j : ℕ} (hj : j < J) (w : Claims.Flock.Wit) (a : ℕ) :
    (tableWit J w j).hid.getD a false = w.hid.getD (a * J + j) false := by
  by_cases ha : a < w.hid.length
  · simp only [tableWit, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range ha, Option.map_some,
      Option.getD_some]
  · have hJ : a ≤ a * J := Nat.le_mul_of_pos_right a (by omega)
    rw [List.getD_eq_default _ _ (by simp only [tableWit, List.length_map, List.length_range]; omega),
      List.getD_eq_default _ _ (by omega)]

/-- **Table `j`'s model witness is its share's**: as the one table of its own statement, at its share of `w`. -/
theorem zOf_tableWit {J j : ℕ} (hj : j < J) (w : Claims.Flock.Wit) (m : ℕ) :
    zOf 1 (tableWit J w j) 0 m = zOf J w j m := by
  funext a
  simp only [zOf, Nat.mul_one, Nat.add_zero, tableWit_getD hj]

/-- **`RelS` reads only the hidden values' `getD`.** -/
theorem relS_congr (s : Claims.Flock.Stmt) {w w' : Claims.Flock.Wit}
    (h : ∀ i, w.hid.getD i false = w'.hid.getD i false) : Claims.Flock.RelS s w ↔ Claims.Flock.RelS s w' := by
  have hv : Claims.Flock.val s w = Claims.Flock.val s w' := by
    funext src
    cases src <;> simp only [Claims.Flock.val, h]
  unfold Claims.Flock.RelS
  rw [hv]

/-- **The relation splits by table**: at a statement the verifier parses (untyped, without public input ports), a
witness satisfies the readable statement exactly when every table's share of it satisfies that table alone. -/
theorem relS_only {p : Parsed} (hp : Parses p) (ht : p.tags.typed = false) (hpi : p.publicInputs = none)
    (w : Claims.Flock.Wit) :
    Claims.Flock.RelS (toReadable relation p) w ↔
      ∀ j < p.sts.size, Claims.Flock.RelS (toReadable relation (p.only j)) (tableWit p.sts.size w j) := by
  have hs := tablesOf_of_parses hp
  rw [relS_relation]
  refine forall₂_congr fun j hj => ?_
  rw [relS_relation]
  have h0 : 0 < p.sts.size := by omega
  have hfit := tables_fit hs ht hpi hj
  have hm : (toReadable relation p).m = p.sts[0].m := by
    simp only [toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some]
  have hk : (toReadable relation p).kLog = p.sts[0].c.kLog := by
    simp only [toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some]
  have hj' : (p.only j).sts = #[p.sts[j]] := by
    show #[p.sts[j]!] = _
    rw [getElem!_pos p.sts j hj]
  have hm1 : (toReadable relation (p.only j)).m = p.sts[j].m := by simp [toReadable, hj']
  have hk1 : (toReadable relation (p.only j)).kLog = p.sts[j].c.kLog := by simp [toReadable, hj']
  have h00 : (p.only j).sts[0]! = p.sts[j] := by
    rw [hj']
    rfl
  have hsz : (p.only j).sts.size = 1 := by rw [hj']; rfl
  rw [hsz, getElem!_pos p.sts j hj, hm, hk, hm1, hk1]
  constructor
  · rintro ⟨_, hz⟩ c hc
    obtain rfl : c = 0 := by omega
    rw [h00, zOf_tableWit hj]
    exact ⟨⟨rfl, rfl, hfit.2.2⟩, hz⟩
  · intro H
    have H0 := H 0 Nat.one_pos
    rw [h00, zOf_tableWit hj] at H0
    obtain ⟨_, hz⟩ := H0
    exact ⟨hfit, hz⟩

end Flock.Readable
