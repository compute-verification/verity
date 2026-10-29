import Proofs.Flock.Soundness.Lowering
import Proofs.Flock.Soundness.GateRows

/-!
# A unit's certificate: rows that are a gate circuit's rows compute it

The audit certifies a unit against its rows: a correct unit's committed values satisfy them (`lowering_sound`, through
`Partition.Correct`). What those values compute is the certificate's. When the unit's rows are the rows `GateRows.rows`
derives from a gate circuit `G` at a layout `L` (`RowsCert`), the committed values on the unit's output columns are `G`'s
outputs on its input columns (`IsRowsUnit.computes_of_cert`, from `GateRows.rows_sound`). And when `G` is the lowering of
a definition `D`, its Boolean export, they are `D`'s.

So nothing about the rows is assumed: the circuit is the rows the untrusted side chose, and the lowering does no
optimizing. "The rows compute `D`" takes two premises rather than a named hypothesis. The certificate's
rows against `GateRows.rows G L` (`RowsCert`) is a fact about finite data, which a kernel computation checks. `G` against
`D` (`∀ x, G.eval x = D x`) ranges over every input, and nothing in Lean ties `D` to a Definition until the Definition's
Boolean semantics is stated there.

* `RowsCert R G L`: `R`'s input columns are `L`'s input region, its computed rows are the AND and output rows of
  `rows G L` as masks, its constant is `rows G L`'s, and the side conditions of `rows_sound` hold. Its rows read, among
  the input columns, only `G`'s inputs' (`reads`): the input region's other columns are padding, which `rows G L` forces to
  0 and `R` leaves free, so `RowsCert.sound` zeroes them.
* `RowsCert.sound`: column values satisfying `R` with the constant 1 carry on each output column `G`'s output on the input
  columns.
* `IsRowsUnit.computes_of_cert`: at a unit that is not wrong, with its constant 1, each committed output wire carries
  `D`'s output on the unit's input wires.
-/

namespace FlockSoundness

open Audit

namespace GateRows

theorem run_congr : ∀ (gs : List Gate) (w : ℕ) {v v' : ℕ → Bool}, (∀ j < w, v j = v' j) → gatesWF w gs = true →
    ∀ j < w + gs.length, run v w gs j = run v' w gs j
  | [], _, _, _, h, _, j, hj => h j (by simpa using hj)
  | g :: gs, w, v, v', h, hwf, j, hj => by
      simp only [gatesWF, Bool.and_eq_true] at hwf
      simp only [run]
      refine run_congr gs (w + 1) (fun i hi => ?_) hwf.2 j (by simp only [List.length_cons] at hj; omega)
      by_cases hiw : i = w
      · rw [ite_eq_left hiw, ite_eq_left hiw]; exact Gate.val_congr g hwf.1 h
      · rw [ite_eq_right hiw, ite_eq_right hiw]; exact h i (by omega)

/-- A gate circuit's outputs read only its inputs. -/
theorem Circ.eval_congr (C : Circ) (hwf : gatesWF C.nIn C.gates = true)
    (houts : ∀ o ∈ C.outs, o.below (C.nIn + C.gates.length) = true) {x x' : ℕ → Bool}
    (h : ∀ k < C.nIn, x k = x' k) : C.eval x = C.eval x' := by
  unfold Circ.eval
  exact List.map_congr_left fun o ho => Src.val_congr o (houts o ho) (run_congr C.gates C.nIn h hwf)

theorem rows_in (C : Circ) (L : Lay) (r : ℕ) (hr : r < L.andBase) : (rows C L)[r]? = some (inRow C L r) := by
  have h1 : ((List.range L.andBase).map (inRow C L)).length = L.andBase := by simp
  unfold rows
  simp only
  rw [List.getElem?_append_left (by simp only [List.length_append]; omega),
    List.getElem?_append_left (by simp only [List.length_append]; omega),
    List.getElem?_append_left (by rw [h1]; exact hr), List.getElem?_map, List.getElem?_range hr, Option.map_some]

end GateRows

open GateRows

/-- **A unit's certificate**: rows `R` are the rows `GateRows.rows` derives from the gate circuit `G` at the layout `L`.
`R`'s input columns are `L`'s input region and its constant `rows G L`'s; computed row `i` is row `nIn + i` of
`rows G L`, its columns as masks; among the input columns it reads only `G`'s inputs'; and `rows_sound`'s side conditions
hold. Each field is a decidable fact about `R`, `G` and `L`. -/
structure RowsCert (R : Rows) (G : Circ) (L : Lay) : Prop where
  nIn_eq : R.nIn = L.andBase
  w_eq : R.nIn + R.nComp = constCol G L
  row : ∀ i : Fin R.nComp,
    (rows G L)[R.nIn + i.val]? = some (listMask ((R.a i).map Fin.val), listMask ((R.b i).map Fin.val))
  reads : ∀ (i : Fin R.nComp), ∀ c ∈ R.a i ++ R.b i, c.val < R.nIn → ∃ k < G.nIn, L.inCol k = c.val
  wf : gatesWF G.nIn G.gates = true
  bound : G.nIn + G.gates.length ≤ 2 ^ depth
  inCol_lt : ∀ k < G.nIn, L.inCol k < L.andBase
  outs_below : ∀ o ∈ G.outs, o.below (G.nIn + G.gates.length) = true

namespace RowsCert

variable {R : Rows} {G : Circ} {L : Lay} (cert : RowsCert R G L)
include cert

theorem out_lt {i : ℕ} (hi : i < L.outLen) : L.andBase + nAnd G + i < R.w := by
  have := cert.w_eq
  simp only [constCol] at this
  show _ < R.nIn + R.nComp + 1
  omega

theorem inCol_lt_w {k : ℕ} (hk : k < G.nIn) : L.inCol k < R.w := by
  have := cert.inCol_lt k hk
  have := cert.nIn_eq
  show _ < R.nIn + R.nComp + 1
  omega

open Classical in
/-- **Satisfied certified rows compute the gate circuit.** Column values satisfying `R` with the constant 1 carry on
each output column `G`'s output on the input columns. The input region's padding columns, which `rows G L` forces to 0
and `R` leaves free, are zeroed first: `R`'s rows don't read them (`reads`). -/
theorem sound (zR : Fin R.w → Bool) (hz : R.Sat zR) (h1 : zR R.one = true)
    (i : ℕ) (hi : i < L.outLen) (o : ℕ) (ho : L.outAt (L.andBase + nAnd G + i) = some o)
    (hol : o < G.outs.length) :
    zR ⟨L.andBase + nAnd G + i, cert.out_lt hi⟩ =
      (G.eval fun k => if h : L.inCol k < R.w then zR ⟨L.inCol k, h⟩ else false)[o]'(by
        simpa [Circ.eval] using hol) := by
  have hlen : (rows G L).length = R.w := by
    rw [rows_length, ← cert.w_eq]
  have hw : R.w = constCol G L + 1 := by rw [← hlen, rows_length]
  have hcw : constCol G L < R.w := by rw [← cert.w_eq]; show _ < R.nIn + R.nComp + 1; omega
  have hab : L.andBase ≤ constCol G L := by unfold constCol; omega
  set z : ℕ → Bool := fun c => if h : c < R.w then zR ⟨c, h⟩ else false with hzdef
  set pad : ℕ → Prop := fun c => c < L.andBase ∧ ¬ ∃ k < G.nIn, L.inCol k = c with hpad
  set z' : ℕ → Bool := fun c => if pad c then false else z c with hz'def
  have hz'z : ∀ c, ¬ pad c → z' c = z c := fun c hc => ite_eq_right hc
  have hnpad : ∀ c, L.andBase ≤ c → ¬ pad c := fun c hc hp => by have := hp.1; omega
  have hzR : ∀ c : Fin R.w, z c.val = zR c := fun c => by simp only [hzdef, dite_eq_left c.isLt]
  have hlist : ∀ l : List (Fin R.w), (∀ c ∈ l, c.val < R.nIn → ∃ k < G.nIn, L.inCol k = c.val) →
      psum z' (listMask (l.map Fin.val)) (rows G L).length = xorSum l zR := by
    intro l hl
    rw [← xorSum_eq_psum z' _ _ (fun c hc => by
      obtain ⟨c', -, rfl⟩ := List.mem_map.1 hc; rw [hlen]; exact c'.isLt), xorSum_map]
    refine xorSum_congr fun c hc => ?_
    show z' c.val = zR c
    rw [hz'z c.val ?_, hzR]
    intro hp
    by_cases hcn : c.val < R.nIn
    · exact hp.2 (hl c hc hcn)
    · have := hp.1; have := cert.nIn_eq; omega
  have hsat : Sat (rows G L) z' := by
    intro r p hp
    by_cases hr : r < L.andBase
    · rw [rows_in G L r hr] at hp
      cases hp
      unfold inRow
      by_cases hin : ∃ k < G.nIn, L.inCol k = r
      · have hany : ((List.range G.nIn).any fun k => L.inCol k == r) = true := by
          obtain ⟨k, hk, hkr⟩ := hin
          exact List.any_eq_true.2 ⟨k, List.mem_range.2 hk, by simp [hkr]⟩
        rw [ite_eq_left hany]
        simp only [psum_two_pow, show r < (rows G L).length by omega, decide_true, Bool.true_and, Bool.and_self]
      · have hany : ((List.range G.nIn).any fun k => L.inCol k == r) = false := by
          rw [Bool.eq_false_iff]
          intro h
          obtain ⟨k, hk, hkr⟩ := List.any_eq_true.1 h
          exact hin ⟨k, List.mem_range.1 hk, by simpa using hkr⟩
        rw [ite_eq_right (by simp [hany])]
        simp only [psum_zero, Bool.and_self]
        exact ite_eq_left ⟨hr, hin⟩
    · by_cases hrc : r < constCol G L
      · have hi' : r - R.nIn < R.nComp := by have := cert.w_eq; have := cert.nIn_eq; omega
        have hrow := cert.row ⟨r - R.nIn, hi'⟩
        rw [show R.nIn + (r - R.nIn) = r by have := cert.nIn_eq; omega, hp] at hrow
        cases hrow
        have hcomp := hz ⟨r - R.nIn, hi'⟩
        rw [hlist _ fun c hc => cert.reads _ c (List.mem_append_left _ hc),
          hlist _ fun c hc => cert.reads _ c (List.mem_append_right _ hc), ← hcomp,
          hz'z r (hnpad r (by omega))]
        rw [← hzR]
        simp only [Rows.comp]
        congr 1
        have := cert.nIn_eq
        omega
      · by_cases hrk : r = constCol G L
        · subst hrk
          rw [rows_const] at hp
          cases hp
          simp only [psum_two_pow, show constCol G L < (rows G L).length by omega, decide_true, Bool.true_and,
            Bool.and_self]
        · rw [List.getElem?_eq_none (by omega)] at hp
          cases hp
  have h1' : z' (constCol G L) = true := by
    rw [hz'z _ (hnpad _ hab)]
    refine (hzR ⟨constCol G L, hcw⟩).trans ?_
    rw [← h1]
    congr 1
    exact Fin.ext cert.w_eq.symm
  have key := rows_sound G L cert.wf cert.bound cert.inCol_lt cert.outs_below z' hsat h1' i hi o ho hol
  rw [hz'z _ (hnpad _ (by omega))] at key
  refine (hzR ⟨_, cert.out_lt hi⟩).symm.trans
    (key.trans (List.getElem_of_eq (Circ.eval_congr G cert.wf cert.outs_below fun k hk => ?_) _))
  rw [hz'z _ fun hp => hp.2 ⟨k, hk, rfl⟩]

end RowsCert

namespace Audit.Partition.IsRowsUnit

variable {C : Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {R : Rows} (I : P.IsRowsUnit u R)

theorem one_ne_comp (i : Fin R.nComp) : R.one ≠ R.comp i := fun h => by
  have := congrArg Fin.val h
  simp only [Rows.one, Rows.comp] at this
  omega

/-- **A certified unit that is not wrong computes its definition.** Let unit `u`'s rows be certified against the gate
circuit `G` at the layout `L` (`RowsCert`), and let `G` compute `D` (the lowering's Boolean export of a definition `D`).
At committed values `X` on which `u` is not wrong, with its constant 1, each committed output wire of `u` carries `D`'s
output on `u`'s input wires: output `o`, held by the copy row `L.andBase + nAnd G + i`, is `D`'s output `o` on the values
at the input columns `L.inCol k`. -/
theorem computes_of_cert {G : Circ} {L : Lay} (cert : RowsCert R G L) (D : (ℕ → Bool) → List Bool)
    (hD : ∀ x, G.eval x = D x) {X : Fin C.N → Bool} (hX : u ∉ P.wrong X) (h1 : X (I.wire R.one) = true)
    (i : ℕ) (hi : i < L.outLen) (o : ℕ) (ho : L.outAt (L.andBase + nAnd G + i) = some o) (hol : o < G.outs.length)
    (hcom : I.wire ⟨L.andBase + nAnd G + i, cert.out_lt hi⟩ ∈ P.committed) :
    X (I.wire ⟨L.andBase + nAnd G + i, cert.out_lt hi⟩) =
      (D fun k => if h : L.inCol k < R.w then X (I.wire ⟨L.inCol k, h⟩) else false)[o]'(by
        rw [← hD]; simpa [Circ.eval] using hol) := by
  have hcor : P.Correct X u := fun g hg hgu => by
    by_contra hne
    exact hX (P.mem_wrong.2 fun hc => hne (hc g hg hgu))
  set zR : Fin R.w → Bool := fun c => P.localEval X u (I.wire c) with hzR
  have hsat : R.Sat zR := by
    intro j
    show P.localEval X u (I.wire (R.comp j)) = _
    rw [P.localEval_of_mem X (I.mem j), I.op j]
    simp only [Op.apply, xorSum_map, Function.comp_def, hzR]
  have h1' : zR R.one = true := by
    show P.localEval X u (I.wire R.one) = true
    rw [P.localEval_of_not_mem X (I.outside R.one (one_ne_comp)), h1]
  have hnc : ∀ c : Fin R.w, c.val < R.nIn → ∀ j, c ≠ R.comp j := fun c hc j h => by
    have := congrArg Fin.val h
    simp only [Rows.comp] at this
    omega
  have hout : R.comp ⟨nAnd G + i, by have := cert.w_eq; have := cert.nIn_eq; simp only [constCol] at *; omega⟩ =
      ⟨L.andBase + nAnd G + i, cert.out_lt hi⟩ := Fin.ext (by
    simp only [Rows.comp]; have := cert.nIn_eq; omega)
  have key := cert.sound zR hsat h1' i hi o ho hol
  rw [hcor _ hcom (by rw [← hout]; exact I.mem _)]
  refine key.trans ((List.getElem_of_eq (Circ.eval_congr G cert.wf cert.outs_below fun k hk => ?_) _).trans
    (List.getElem_of_eq (hD _) _))
  have hlt := cert.inCol_lt_w hk
  have hin : L.inCol k < R.nIn := by have := cert.inCol_lt k hk; have := cert.nIn_eq; omega
  rw [dite_eq_left hlt, dite_eq_left hlt, hzR]
  exact P.localEval_of_not_mem X (I.outside _ (hnc ⟨L.inCol k, hlt⟩ hin))

end Audit.Partition.IsRowsUnit

end FlockSoundness
