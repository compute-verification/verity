import Proofs.Flock.C3.Reps.ResRead
import Proofs.Flock.C3.Reps.IoCols
import Proofs.Flock.C3.Sparse.SrcLayer
import Proofs.Flock.Recursive.HPortTie

/-!
# The RecSparse unit reads `h`'s columns (`hCols_of_check`)

`HCols` (`HPortTie`), proved at every unit `RecSparse.check` accepts at the shape its name gives: the unit is
`RecSparse.unitLaid sh`, `steps` on its ports by name laid out (`unitNet'_run`).

* The builder (`rowReads_steps`): `steps` computes the residuals first, and residual `h1`'s product `l` multiplies `v`'s
  element `l` (`eqc`) by `w`'s element `l` (`h`'s; `structureOf_spec`). Karatsuba's leaf `e` ANDs their bits `e`, two
  input bits, `v`'s after the message ports and `h`'s at `rowCol` (`reads_residualForms_of`), and every builder after
  the residuals keeps that row (`reads_steps`, the `keeps_*` lemmas).
* The layout and the net (`laidSparse_reads`, `hCols_of_check`): the builder's `k`-th row is the laid-out row
  `inRows + k`, a computed row of the net at `unitLog ≤ 32` (`readCol_of_laid`).
-/

namespace FlockC3.Sparse

section Builder

open Flock.HmNets hiding and
open FlockC3.Reps.RowRead
open FlockSoundness.Discharge.Hm (vars length_vars inRowsOf)
open FlockVBridge.OfVStar (rowForms rowCol)

/-! ## The builders after the residuals keep their rows -/

theorem keeps_forIn {σ : Type} (f : ℕ → σ → M (ForInStep σ)) (hf : ∀ j s, Keeps (f j s)) :
    ∀ (l : List ℕ) (s : σ), Keeps (forIn l s f)
  | [], s => by rw [List.forIn_nil]; exact Keeps.pure' _
  | j :: l, s => by
      rw [List.forIn_cons]
      refine Keeps.bind' (hf j s) fun y => ?_
      cases y with
      | done b => exact Keeps.pure' _
      | yield b => exact keeps_forIn f hf l b

theorem keeps_forIn_range {σ : Type} (n : ℕ) (init : σ) (f : ℕ → σ → M (ForInStep σ)) (hf : ∀ j s, Keeps (f j s)) :
    Keeps (forIn [0:n] init f) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact keeps_forIn f hf _ _

theorem keeps_mulF (a b : List Lin) : Keeps (Flock.RecSparse.mulF a b) := by
  unfold Flock.RecSparse.mulF
  refine Keeps.bind' (keeps_commitW a) fun _ => Keeps.bind' (keeps_commitW b) fun _ => ?_
  exact Keeps.bind' (Keeps.mapM' _ (fun _ => keeps_and _ _) _) fun _ => Keeps.pure' _

theorem keeps_pairsMul : ∀ fac : List (List Lin), Keeps (Flock.RecSparse.pairsMul fac)
  | [] => Keeps.pure' _
  | [_] => Keeps.pure' _
  | a :: b :: rest => by
      simp only [Flock.RecSparse.pairsMul]
      exact Keeps.bind' (keeps_mulF a b) fun _ => Keeps.bind' (keeps_pairsMul rest) fun _ => Keeps.pure' _

theorem keeps_tree : ∀ (n : ℕ) (fac : List (List Lin)), Keeps (Flock.RecSparse.tree n fac)
  | 0, _ => Keeps.pure' _
  | _ + 1, [] => Keeps.pure' _
  | _ + 1, [_] => Keeps.pure' _
  | n + 1, a :: b :: rest => by
      simp only [Flock.RecSparse.tree]
      exact Keeps.bind' (keeps_pairsMul _) fun ps => keeps_tree n ps

theorem keeps_demuxLevel (tau : Array Lin) (j : ℕ) (lists : List (List (List Lin))) :
    Keeps (Flock.RecSparse.demuxLevel tau j lists) := by
  unfold Flock.RecSparse.demuxLevel
  refine Keeps.mapM' _ (fun p => ?_) _
  obtain ⟨xs, s⟩ := p
  refine Keeps.bind' (Keeps.mapM' _ (fun x => ?_) _) fun _ => Keeps.pure' _
  exact Keeps.bind' (Keeps.mapM' _ (fun b => keeps_and _ b) _) fun _ => Keeps.pure' _

theorem keeps_demux (tau : Array Lin) : ∀ (js : List ℕ) (lists : List (List (List Lin))),
    Keeps (Flock.RecSparse.demux tau js lists)
  | [], lists => by
      simp only [Flock.RecSparse.demux]
      exact Keeps.pure' _
  | j :: js, lists => by
      simp only [Flock.RecSparse.demux]
      exact Keeps.bind' (keeps_demuxLevel tau j lists) fun l => keeps_demux tau js l

/-- **`steps` keeps every row its residuals leave**: a row of the residuals' computation over the message ports' first
bits and `v`'s stays through the rest of `steps`. -/
theorem reads_steps {sh : Flock.RecSparse.Shape} {P : String → List Lin} {u : Var}
    (h : Reads (Flock.RecOpen.residualForms (Flock.RecSparse.structureOf sh).1
      ((Flock.RecSparse.messagePorts sh).flatMap fun p => (P p.1).take (128 * p.2))
      ((P "v").take (128 * Flock.RecSparse.nv sh))) u) :
    Reads (Flock.RecSparse.steps sh P) u := by
  unfold Flock.RecSparse.steps
  revert h
  generalize Flock.RecSparse.structureOf sh = p
  obtain ⟨S, names⟩ := p
  intro h
  dsimp only
  rw [Flock.RecResiduals.residualForms_eq]
  refine Reads.bind_left h fun out => ?_
  refine Keeps.bind' (keeps_forIn_range _ _ _ fun _ _ => Keeps.bind' (keeps_mulF _ _) fun _ => Keeps.pure' _)
    fun c2 => ?_
  refine Keeps.bind' (keeps_forIn_range _ _ _ fun _ _ => Keeps.bind' (keeps_mulF _ _) fun _ => Keeps.pure' _)
    fun c4 => ?_
  refine Keeps.bind' (keeps_tree _ _) fun phi => ?_
  split
  · simp only [pure_bind]
    refine Keeps.bind' (keeps_demux _ _ _) fun lists => ?_
    refine Keeps.bind' (Keeps.mapM' _ (fun x => ?_) _) fun sel => Keeps.pure' _
    obtain ⟨a, b⟩ := x
    exact keeps_mulF a b
  · simp only [pure_bind]
    exact Keeps.pure' _

/-! ## Residual `h1` reads `h` -/

theorem term_of_kTerm_v {k : Flock.RecSparse.Key} {i : ℕ} (h : FlockVBridge.Sparse.kTerm k = .v i) :
    Flock.RecSparse.term k = .v i := by
  unfold FlockVBridge.Sparse.kTerm at h
  unfold Flock.RecSparse.term
  split at h
  · cases h
  · rename_i hk
    simp only [hk, ↓reduceIte]
    exact congrArg Flock.RecOpen.Term.v (Flock.RecAlgebra.Term.v.inj h)

theorem term_of_kTerm_w {k : Flock.RecSparse.Key} {i : ℕ} (h : FlockVBridge.Sparse.kTerm k = .w i) :
    Flock.RecSparse.term k = .w i := by
  unfold FlockVBridge.Sparse.kTerm at h
  unfold Flock.RecSparse.term
  split at h
  · rename_i hk
    simp only [hk, ↓reduceIte]
    exact congrArg Flock.RecOpen.Term.w (Flock.RecAlgebra.Term.w.inj h)
  · cases h

theorem getD_ops {F : Array Flock.RecSparse.Key} {i : ℕ} {k : Flock.RecSparse.Key} (h : F[i]? = some k) :
    (F.toList.map fun k => [Flock.RecSparse.term k]).getD i [] = [Flock.RecSparse.term k] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList, h]
  rfl

/-- **Residual `h1`'s product `l` is `v`'s element `l` times `w`'s**: at the builder's structure, its `l`-th pair of
operands are the one-term operands `v l` and `w l`. -/
theorem h1_pair (sh : Flock.RecSparse.Shape) {l : ℕ} (hl : l < 64) :
    ∃ r ∈ (Flock.RecSparse.structureOf sh).1.res, ∃ ab ∈ r.mul,
      (Flock.RecSparse.structureOf sh).1.ops.getD ab.1 [] = [.v l] ∧
      (Flock.RecSparse.structureOf sh).1.ops.getD ab.2 [] = [.w l] := by
  obtain ⟨F, rs, hs, hrel⟩ := FlockVBridge.Sparse.structureOf_spec sh
  rw [hs]
  obtain ⟨rest, hn⟩ : ∃ rest, FlockVBridge.Sparse.names (FlockVBridge.Sparse.shOf sh) = .h1 :: rest := ⟨_, rfl⟩
  rw [hn] at hrel
  obtain ⟨p, rs', rfl⟩ : ∃ p rs', rs = p :: rs' := by
    cases hrel with
    | cons h _ => exact ⟨_, _, rfl⟩
  obtain ⟨-, mul, add, hsp, -, hm⟩ := (List.forall₂_cons.1 hrel).1
  have hsp' := congrArg Prod.fst hsp
  simp only [FlockVBridge.Sparse.spec, FlockVBridge.Sparse.kSpec] at hsp'
  have hlen : mul.length = 64 := by simpa using (congrArg List.length hsp').symm
  have hml : l < mul.length := by omega
  have hpl : l < p.2.mul.length := by rw [← hm.length_eq]; exact hml
  have hk := congrArg (·[l]?) hsp'
  simp only [List.getElem?_map, List.getElem?_range hl, List.getElem?_eq_getElem hml, Option.map_some,
    Option.some.injEq, Prod.mk.injEq] at hk
  have h12 := hm.get hml hpl
  simp only [List.get_eq_getElem] at h12
  obtain ⟨h1, h2⟩ := h12
  refine ⟨p.2, List.mem_map.2 ⟨p, List.mem_cons_self, rfl⟩, p.2.mul[l], List.getElem_mem hpl, ?_, ?_⟩
  · show (F.toList.map fun k => [Flock.RecSparse.term k]).getD _ [] = _
    rw [getD_ops h1, term_of_kTerm_v hk.1.symm]
  · show (F.toList.map fun k => [Flock.RecSparse.term k]).getD _ [] = _
    rw [getD_ops h2, term_of_kTerm_w hk.2.symm]

/-- `h`'s bits are on the message ports, below `v`'s port. -/
theorem rowCol_lt_vBaseS (sh : Flock.RecSparse.Shape) {q : ℕ}
    (hq : q < 128 * ((Flock.RecSparse.messagePorts sh).map (·.2)).sum) :
    rowCol (Flock.RecSparse.messagePorts sh) 0 q < FlockVBridge.Sparse.vBaseS sh := by
  have hlt := FlockVBridge.OfVStar.rowCol_lt (Flock.RecSparse.messagePorts sh) 0 q hq
  have htb : FlockVBridge.Sparse.tauBase sh =
      16 * ((Flock.RecSparse.messagePorts sh).map fun p => VStar.portWordsOf p.2).sum := rfl
  have hvb : FlockVBridge.Sparse.tauBase sh ≤ FlockVBridge.Sparse.vBaseS sh := Nat.le_add_right _ _
  omega

/-- **The sparse unit's builder leaves a row reading each bit of `h`**: residual `h1`'s product `l` ANDs bit `e` of
`v`'s element `l` with bit `e` of `h`'s (`h1_pair`, `reads_residualForms_of`). -/
theorem rowReads_steps (sh : Flock.RecSparse.Shape) {l e : ℕ} (hl : l < 64) (he : e < 128) :
    RowReads ((Flock.RecSparse.steps sh (FlockVBridge.Sparse.portOf sh)).run
      ⟨16 * Flock.RecSparse.inWords sh, []⟩).2 (rowCol (Flock.RecSparse.messagePorts sh) 0 (128 * l + e)) := by
  refine reads_steps ?_ _
  rw [FlockVBridge.Sparse.flatMap_portOf, FlockVBridge.Sparse.portOf_v,
    FlockVBridge.OfVStar.take_vars _ _ _ (FlockVBridge.OfVStar.le_portWords _)]
  obtain ⟨r, hr, ab, hab, h₁, h₂⟩ := h1_pair sh hl
  have hq := FlockC3.HPort.h_lt_ports sh hl he
  have hnv := FlockVBridge.Sparse.nv_eq sh
  have hlt := rowCol_lt_vBaseS sh hq
  have hne : FlockVBridge.Sparse.vBaseS sh + (128 * l + e) ≠ rowCol (Flock.RecSparse.messagePorts sh) 0 (128 * l + e) :=
    by omega
  refine reads_residualForms_of hr hab he (u₁ := FlockVBridge.Sparse.vBaseS sh + (128 * l + e))
    (u₂ := rowCol (Flock.RecSparse.messagePorts sh) 0 (128 * l + e)) ?_ ?_ hne (Or.inr rfl)
  · rw [getD_opForms h₁ he, getD_termForms_v _ _ _ he]
    exact congrArg _ (getD_inputs _ _ _ (by omega))
  · rw [getD_opForms h₂ he, getD_termForms_w _ _ _ he, FlockC3.Reps.RowRead.getD_rowForms _ _ _ hq]

/-- **The laid-out sparse unit has a computed row for each input bit a builder row reads**: the `k`-th AND, at row
`inRows + k`, its forms' columns those of their variables. -/
theorem laidSparse_reads (sh : Flock.RecSparse.Shape) {c : ℕ}
    (hr : RowReads ((Flock.RecSparse.unitNet' sh).run St.empty).2 c) (hlt : c < 16 * Flock.RecSparse.inWords sh) :
    ∃ k, inRowsOf (FlockVBridge.Sparse.sparseInBits sh) + k < (Flock.RecSparse.unitLaid sh).constPos ∧
      c ∈ ((Flock.RecSparse.unitLaid sh).ra.getD (inRowsOf (FlockVBridge.Sparse.sparseInBits sh) + k) #[]).toList ++
        ((Flock.RecSparse.unitLaid sh).rb.getD (inRowsOf (FlockVBridge.Sparse.sparseInBits sh) + k) #[]).toList := by
  have hN : (FlockVBridge.Sparse.sparseInBits sh).foldl (· + ·) 0 = 16 * Flock.RecSparse.inWords sh := by
    rw [FlockVBridge.Sparse.sparseInBits, FlockVBridge.OfVStar.foldl_replicate]
    exact Nat.mul_comm _ _
  have e : Flock.RecSparse.unitLaid sh = layout ((Flock.RecSparse.unitNet' sh).run St.empty).2
      (FlockVBridge.Sparse.sparseInBits sh) (Array.replicate (Flock.RecSparse.outWords sh) 16)
      ((Flock.RecSparse.unitNet' sh).run St.empty).1.toArray := rfl
  obtain ⟨r, hr, hv⟩ := hr
  obtain ⟨k, hk, hrk⟩ := List.mem_iff_getElem.1 (List.mem_reverse.2 hr)
  obtain ⟨h1, -, h3, -⟩ := layout_rows ((Flock.RecSparse.unitNet' sh).run St.empty).2
    (FlockVBridge.Sparse.sparseInBits sh) (Array.replicate (Flock.RecSparse.outWords sh) 16)
    ((Flock.RecSparse.unitNet' sh).run St.empty).1.toArray
  have hspan := groupWords_pos ((Array.replicate (Flock.RecSparse.outWords sh) 16).foldl (· + ·) 0)
  have hro := le_roundUp (groupWords ((FlockVBridge.Sparse.sparseInBits sh).foldl (· + ·) 0) * Flock.WORD +
    ((Flock.RecSparse.unitNet' sh).run St.empty).2.rows.length) _ hspan
  have hk' : k < ((Flock.RecSparse.unitNet' sh).run St.empty).2.rows.length := by
    rw [← List.length_reverse]; exact hk
  rw [e]
  refine ⟨k, ?_, ?_⟩
  · rw [h1]
    show groupWords _ * Flock.WORD + k < _
    omega
  · obtain ⟨ha, hb⟩ := h3 k hk
    rw [ha, hb, hrk]
    rcases List.mem_append.1 hv with hv | hv
    · exact List.mem_append_left _ (mem_cols hv (by rw [hN]; exact hlt))
    · exact List.mem_append_right _ (mem_cols hv (by rw [hN]; exact hlt))

end Builder

section Net

open FlockSoundness FlockSoundness.Discharge.Recursion
open FlockVBridge.OfVStar (rowCol)

/-- **`HCols` from the verifier's check**: at a unit `RecSparse.check` accepted at the shape its name gives, the unit
is `RecSparse.unitLaid sh` (`check_sparse_all`), whose builder has a row reading each column of `h`'s element `l`, bit
`e` (`rowReads_steps`), a computed row of the net (`laidSparse_reads`, `readCol_of_laid`; `unitLog ≤ 32` from the
statement's scope). -/
theorem hCols_of_check {V : VStmt} (hck : Flock.RecSparse.check V.c = .ok ()) {sh : Flock.RecSparse.Shape}
    (hp : Flock.RecSparse.params V.c.unit.name = some sh) :
    FlockC3.HPort.HCols V (Flock.RecSparse.messagePorts sh) := by
  intro l e he
  obtain ⟨-, -, -, -, -, hn⟩ := FlockVBridge.Sparse.check_sparse_all hck hp
  have hn' : Flock.HmNets.netOf (Flock.RecSparse.unitName sh) (Flock.RecSparse.unitLaid sh) = .ok V.c.unit := hn
  have hq := FlockC3.HPort.h_lt_ports sh l.isLt he
  have hlt := FlockC3.Sparse.rowCol_lt_vBaseS sh hq
  have h16 := FlockVBridge.Sparse.inWords_16 sh
  have hr : FlockC3.Reps.RowRead.RowReads ((Flock.RecSparse.unitNet' sh).run Flock.HmNets.St.empty).2
      (rowCol (Flock.RecSparse.messagePorts sh) 0 (128 * l.val + e)) := by
    rw [FlockVBridge.Sparse.unitNet'_run]
    exact rowReads_steps sh l.isLt he
  obtain ⟨k, hk, hmem⟩ := laidSparse_reads sh hr (by omega)
  rw [← FlockVBridge.Sparse.inWords_sparse hn'] at hk hmem
  exact FlockC3.Reps.RowRead.readCol_of_laid V.hU hn' (FlockC3.Reps.unitLog_of_scope V.hscope) hk hmem

end Net

end FlockC3.Sparse
