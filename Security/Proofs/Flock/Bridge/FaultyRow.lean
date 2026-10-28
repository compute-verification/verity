import Proofs.Flock.Bridge.PublicWitnesses
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Bridge.Transports

/-!
# A returned row whose first list is one input, opened against it, is faulty

`faulty_row`: a returned gate `g = row [i] b` of unit `u`, on an input `i` the constants don't fix, under openings
`witRows v` with `v i = false` and `v g = true` wherever `g`'s position opens, makes `u` faulty. Its replay reads `i` from
the commitment (a value it reads must be opened), so it computes `g` as `false && …`, while the opening at `g`'s
position, which its boundary commits, says `true` or is missing. `not_relS_row` is it at a draw of every unit, and
`commitDep_rows_of_opens` says where `commitDep` opens: only at a wire with a row.
-/

namespace Flock.Bridge

open FlockSoundness.Audit Claims.Sampling

variable {C : FlockSoundness.Audit.Circuit} {κ : Fin C.N → Option Bool}

/-- **What `witRows` opens at a wire's position is its value**, through an opening that passes there. -/
theorem witRows_val_some {Cm : CommitmentScheme Bool} {H : Claims.Msg → Claims.Msg} {roots : List Claims.Msg}
    {v : Fin C.N → Bool} {s : Claims.Sampling.AuditStatement} (hs : s.roots = roots) {g : Fin C.N} {b : Bool}
    (h : (witRows C Cm H roots v).val H Cm s (posOf C g) = some b) :
    b = v g ∧ ∃ d, Cm.Opens H roots ⟨posOf C g, v g, d⟩ = true := by
  unfold Wit.val at h
  rw [hs] at h
  obtain ⟨o, ho, rfl⟩ := Option.map_eq_some_iff.1 h
  have hpo := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hpo
  obtain ⟨g', hp, hv⟩ := witRows_mem (List.mem_of_find?_eq_some ho)
  have hg : g' = g := posOf_injective (hp.symm.trans hpo.1)
  subst hg
  obtain ⟨op, ov, od⟩ := o
  simp only at hp hv hpo
  subst hp hv
  exact ⟨rfl, od, hpo.2⟩

/-- **A returned row on one unfixed input, opened against it, is faulty**: at a returned gate `g = row [i] b` of unit `u`,
with `i` an input the constants don't fix, `v i = false`, and `v g = true` wherever `g`'s position opens at `v g`. -/
theorem faulty_row {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (kd : ℕ) (desc : Claims.Msg) (q : Query)
    (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : CommitmentScheme Bool) (v : Fin C.N → Bool)
    {g i : Fin C.N} {b : List (Fin C.N)} {u : Fin n} (hg : g ∈ C.outputs) (hu : P.unit g = some u)
    (hop : C.op g = .row [i] b) (hi : C.op i = .input) (hκ : κ i = none) (hvi : v i = false)
    (hvg : ∀ d, Cm.Opens H roots ⟨posOf C g, v g, d⟩ = true → v g = true) :
    u.val ∈ faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) (witRows C Cm H roots v) := by
  classical
  have hr0 := rus_partOf_getElem? (κ := κ) P u
  have hlen : u.val < (partOf P κ).rus.length := (List.getElem?_eq_some_iff.1 hr0).1
  unfold Claims.Sampling.faulty
  refine Finset.mem_filter.2 ⟨Finset.mem_range.2 hlen, ?_⟩
  rintro ⟨r, hr, ι, hc⟩
  rw [hr0] at hr
  cases hr
  have hc' : (progOf C κ).Correct (RU.ref (progOf C κ) [unitGates P u]
      ((witRows C Cm H roots v).val H Cm (stmtOf P κ kd desc q roots)) ι) (unitGates P u) := by
    have h := hc (unitGates P u) (List.mem_append_left _ (List.mem_singleton_self _))
    change (progOf C κ).Correct (RU.ref (progOf C κ) (slotOf P κ u)
      ((witRows C Cm H roots v).val H Cm (stmtOf P κ kd desc q roots)) ι) (unitGates P u) at h
    rwa [ref_slotOf] at h
  have hgates : RU.gates [unitGates P u] = unitGates P u := by simp [RU.gates]
  have hs : (stmtOf P κ kd desc q roots).roots = roots := rfl
  have href : ∀ p, ((progOf C κ).Boundary (unitGates P u) p = true ∨ p ∉ unitGates P u) →
      RU.ref (progOf C κ) [unitGates P u] ((witRows C Cm H roots v).val H Cm (stmtOf P κ kd desc q roots)) ι p =
        (witRows C Cm H roots v).val H Cm (stmtOf P κ kd desc q roots) p := by
    intro p hp
    unfold RU.ref
    rw [hgates, if_neg]
    rintro ⟨h1, h2⟩
    rcases hp with hp | hp
    · rw [hp] at h2; cases h2
    · exact hp h1
  have hiu : P.unit i = none := (P.input_iff i).2 (by simp [hi, Op.isInput])
  have hiU : posOf C i ∉ unitGates P u := fun h => by
    rw [posOf_mem_unitGates.1 h] at hiu; cases hiu
  have hpg : posOf C g ∈ unitGates P u := posOf_mem_unitGates.2 hu
  have hout : posOf C g ∈ (progOf C κ).outs :=
    List.mem_map.2 ⟨g, List.mem_filter.2 ⟨List.mem_finRange g, decide_eq_true hg⟩, rfl⟩
  have hb : (progOf C κ).Boundary (unitGates P u) (posOf C g) = true := by
    unfold Claims.Sampling.Prog.Boundary
    simp only [Bool.or_eq_true]
    exact Or.inl (Or.inl (List.elem_eq_true_of_mem hout))
  have F1 := href _ (Or.inr hiU)
  have F2 := href _ (Or.inl hb)
  generalize RU.ref (progOf C κ) [unitGates P u]
    ((witRows C Cm H roots v).val H Cm (stmtOf P κ kd desc q roots)) ι = ref at hc' F1 F2
  have h3 : gateSize (C.op g) = 3 := by simp [hop, gateSize]
  have hp0 : off C g.val + 0 ∈ unitGates P u := mem_unitGates.2 ⟨g, hu, 0, by omega, rfl⟩
  have hx0 : (progOf C κ).gates[off C g.val + 0]? = some (.xor [posOf C i]) := by
    rw [gates_off g (by omega)]
    simp [lower, hop]
  have hrd : (progOf C κ).read ref (posOf C i) ≠ none :=
    (hc' _ hp0).1 _ hx0 (posOf C i) (by simp [BGate.reads]) hiU
  have hgi : (progOf C κ).gates[posOf C i]? = some .input := by
    have hp : posOf C i = off C i.val + 0 := by rw [posOf_eq, hi]; rfl
    rw [hp, gates_off i (by simp [hi, gateSize])]
    simp [lower, hi, hκ]
  have hread_i : (progOf C κ).read ref (posOf C i) = ref (posOf C i) := by
    simp [Prog.read, hgi]
  rw [hread_i, F1] at hrd
  obtain ⟨bi, hbi⟩ := Option.ne_none_iff_exists'.1 hrd
  have hbi' := (witRows_val_some hs hbi).1
  rw [hvi] at hbi'
  subst hbi'
  let Xv : Fin C.N → Bool := fun a => ((progOf C κ).read ref (posOf C a)).getD false
  have hread : ∀ p ∈ unitGates P u, ∀ x, (progOf C κ).gates[p]? = some x → ∀ a : Fin C.N, posOf C a ∈ x.reads →
      posOf C a ∉ unitGates P u → (progOf C κ).read ref (posOf C a) = some (P.localEval Xv u a) := by
    intro p hp x hx a ha hn
    have hne := (hc' p hp).1 x hx (posOf C a) ha hn
    obtain ⟨y, hy⟩ := Option.ne_none_iff_exists'.1 hne
    have hau : P.unit a ≠ some u := fun h => hn (posOf_mem_unitGates.2 h)
    rw [Partition.localEval_of_not_mem P Xv hau]
    simp only [Xv, hy, Option.getD_some]
  have hrun := runIn_posOf_of (κ := κ) P u (P.localEval Xv u) (fun _ h => Partition.localEval_of_mem P Xv h) ref
    hread g.val g rfl hu
  have hVi : P.localEval Xv u i = false := by
    have hne : P.unit i ≠ some u := by rw [hiu]; simp
    rw [Partition.localEval_of_not_mem P Xv hne]
    simp only [Xv, hread_i, F1, hbi, Option.getD_some]
  have hVg : P.localEval Xv u g = false := by
    rw [Partition.localEval_of_mem P Xv hu, hop]
    simp [Op.apply, xorSum, hVi]
  rw [hVg] at hrun
  obtain ⟨bb, hbb, hrb⟩ := (hc' _ hpg).2 hb
  rw [hrun] at hrb
  cases hrb
  rw [F2] at hbb
  obtain ⟨hvg0, d, hd⟩ := witRows_val_some hs hbb
  have := hvg d hd
  rw [← hvg0] at this
  cases this

/-- **`faulty_row`, as the claim failing**: at a draw of every unit (`k_d = n`), at any `δ ≥ 0`. -/
theorem not_relS_row {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (kd : ℕ) (hkd : kd = n) (desc : Claims.Msg)
    (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : CommitmentScheme Bool) (v : Fin C.N → Bool)
    {δ : ℝ} (hδ : 0 ≤ δ) {g i : Fin C.N} {b : List (Fin C.N)} {u : Fin n} (hg : g ∈ C.outputs)
    (hu : P.unit g = some u) (hop : C.op g = .row [i] b) (hi : C.op i = .input) (hκ : κ i = none) (hvi : v i = false)
    (hvg : ∀ d, Cm.Opens H roots ⟨posOf C g, v g, d⟩ = true → v g = true) :
    ¬ RelS H Cm δ (stmtOf P κ kd desc q roots) (partOf P κ) (witRows C Cm H roots v) := by
  subst hkd
  intro h
  have h' : δ < (designOf kd).Pa (partOf P κ)
      (faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) (witRows C Cm H roots v)) := h
  rw [pa_zero P (κ := κ) (faulty_row P kd desc q roots H Cm v hg hu hop hi hκ hvi hvg)] at h'
  linarith

/-- **`commitDep` opens only at a wire with a row.** -/
theorem commitDep_rows_of_opens (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (nl : ℕ)
    (H : Claims.Msg → Claims.Msg) (roots : List Claims.Msg) (g : Fin C.N) (b : Bool) (d : Claims.Msg)
    (h : (commitDep C rows c nl roots).Opens H roots ⟨posOf C g, b, d⟩ = true) : rows g ≠ none := by
  intro hg
  have ha : atOf C rows roots (posOf C g) = none := by
    unfold atOf
    rw [rowOfPos_posOf, hg]
    rfl
  have : (commitDep C rows c nl roots).Opens H roots ⟨posOf C g, b, d⟩ = false := by
    simp only [Claims.Sampling.CommitmentScheme.Opens, commitDep, pathCommit, ha] <;> rfl
  rw [this] at h
  cases h

end Flock.Bridge
