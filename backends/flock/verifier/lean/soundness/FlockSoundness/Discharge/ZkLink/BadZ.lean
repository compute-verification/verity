import FlockSoundness.Discharge.ZkLink.LockRep
import FlockSoundness.Discharge.Exec.CompiledL
import FlockSoundness.Protocol.Discharge.ZkLink.BadZ

/-!
# Where a bad compiled `--zk` run goes wrong

`CompiledL`'s bad split for the compiled `--zk` table (`TopRelZ`): an accepted run whose openings disagree with the
fixed tables (`AgreeAllZ`) either clashes within the run (`SelfClashZ`), or opens a position of one fixed table to
something else. Each committed kind has its openings by position:

* the level-0 oracle at the witness cap, its rows `width₀` lanes wide (`op₀Z`, `opOfW`), and each level's table at its
  cap (`opRZ`, `Exec.opOfL` on `slotOpensZ`);
* rep `r`'s pads oracle at its root (`opPadsZ`): at each pads position, the pair of the first query whose opening
  verifies there;
* rep `r`'s `τ` oracle at its commitment (`opTauZ`): `τ`, if it opens the commitment.

Every clash, and every pair of runs opening one position of one cap, root or commitment two ways, is a collision of `H`
(`SelfClashZ.collision`, `opOfW_conflict_collision`, `opPadsZ_conflict_collision`, `opTauZ_conflict_collision`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-! ## Level 0's wide rows -/

/-- An oracle's `w`-lane rows on a level's domain. -/
def rowTW (w d : ℕ) (t : Oracle F) : Fin (2 ^ d) → Fin w → F := fun p lane => t lane.val p.val

omit [Fintype F] [DecidableEq F] in
theorem rowTW_tabOfW (w d : ℕ) (v : Fin (2 ^ d) → Fin w → F) : rowTW w d (tabOfW w d v) = v := by
  funext p lane
  simp [rowTW, tabOfW, p.isLt, lane.isLt]

theorem op₀Z_eq (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (T : TabZK F mPts) (cap : Cap D) (out : OutZC F D Sl) :
    op₀Z H E Lf T cap out =
      opOfW H E Lf (width₀ T) (pathLen (lvOf T.sch 0)) (lvOf T.sch 0).logLen (opens₀Z T.sch out) cap :=
  rfl

omit [Fintype F] [DecidableEq F] in
open Classical in
/-- A run opens at most as many positions as it has openings. -/
theorem opOfW_count (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (w depth d : ℕ) (xs : List (ℕ × List F × Sl × List D)) (cap : Cap D) :
    (Finset.univ.filter fun p => (opOfW H E Lf w depth d xs cap p).isSome).card ≤ xs.length := by
  calc (Finset.univ.filter fun p => (opOfW H E Lf w depth d xs cap p).isSome).card
      ≤ (xs.map fun x => x.1).toFinset.card := by
        refine Finset.card_le_card_of_injOn (fun p => p.val) (fun p hp => ?_)
          (fun p _ q _ h => Fin.ext h)
        simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hp
        unfold opOfW at hp
        rw [Option.isSome_map] at hp
        obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hp
        have hmem := List.mem_of_mem_head? hx
        rw [List.mem_filter, decide_eq_true_eq] at hmem
        simp only [List.coe_toFinset, List.mem_map, Set.mem_ofPred_eq]
        exact ⟨x, hmem.1, hmem.2.1⟩
    _ ≤ (xs.map fun x => x.1).length := List.toFinset_card_le _
    _ = xs.length := List.length_map _

omit [Fintype F] in
/-- **Off or clash.** A verifying `w`-lane opening of `p` whose row is not `v`: the run opens `p` to a row other than
`v`, or it opens `p` two ways. -/
theorem off_or_clashW (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (w depth d : ℕ) (xs : List (ℕ × List F × Sl × List D)) (cap : Cap D) (x : ℕ × List F × Sl × List D)
    (hx : x ∈ xs) (p : Fin (2 ^ d)) (hxp : x.1 = p.val) (hv : VerifiedZ H E Lf cap w depth x)
    (v : Fin w → F) (hne : rowW w x.2.1 ≠ v) :
    ((opOfW H E Lf w depth d xs cap p).isSome ∧ opOfW H E Lf w depth d xs cap p ≠ some v) ∨
      ClashW H E Lf cap w depth xs := by
  classical
  unfold opOfW
  set ys := xs.filter fun x => decide (x.1 = p.val ∧ VerifiedZ H E Lf cap w depth x)
  have hxy : x ∈ ys := List.mem_filter.2 ⟨hx, decide_eq_true ⟨hxp, hv⟩⟩
  obtain ⟨h, hh⟩ : ∃ h, ys.head? = some h := by
    cases hys : ys with
    | nil => rw [hys] at hxy; simp at hxy
    | cons h _ => exact ⟨h, rfl⟩
  have hmem := List.mem_of_mem_head? hh
  rw [List.mem_filter, decide_eq_true_eq] at hmem
  rw [hh, Option.map_some]
  by_cases hrow : h.2.1 = x.2.1
  · left
    refine ⟨rfl, fun heq => hne ?_⟩
    rw [← hrow]
    exact Option.some.inj heq
  · right
    exact ⟨h, hmem.1, x, hx, hmem.2.1.trans hxp.symm, hmem.2.2, hv, hrow⟩

/-! ## The levels' caps -/

omit [Field F] [Fintype F] [DecidableEq F] in
theorem slotOpensZ_length (out : OutZC F D Sl) (r ℓ Q : ℕ) : (slotOpensZ out r ℓ Q).length ≤ Q := by
  unfold slotOpensZ
  split
  · simp
  · simp only [List.length_map, List.length_zipIdx, List.length_take]
    exact min_le_left _ _

omit [Field F] [Fintype F] [DecidableEq F] in
theorem mem_slotOpensZ {out : OutZC F D Sl} {r : ℕ} {req : Req}
    (hfind : (if r = 0 then out.pend.1 else out.pend.2).lig.reqs.find? (fun q => q.level == req.level) =
      some req)
    {Q : ℕ} (hQ : req.positions.length = Q) {j : ℕ} (hj : j < req.positions.length) :
    (req.positions.getD j 0, (out.opens.rows r req.level j).1, out.opens.salts r req.level j,
      (out.opens.rows r req.level j).2) ∈ slotOpensZ out r req.level Q := by
  unfold slotOpensZ
  rw [hfind]
  simp only [List.mem_map]
  refine ⟨(req.positions.getD j 0, j), ?_, rfl⟩
  rw [List.take_of_length_le hQ.le, List.mem_iff_getElem]
  refine ⟨j, by simpa using hj, ?_⟩
  rw [List.getElem_zipIdx]
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]

/-! ## The pads roots -/

/-- A pads oracle's pairs, lanes `0` and `1` at each position. -/
def rowPads (N : ℕ) (t : Oracle F) (p : Fin N) : F × F := (t 0 p.val, t 1 p.val)

omit [Fintype F] [DecidableEq F] in
theorem rowPads_padsTab (N : ℕ) (v : Fin N → F × F) : rowPads N (padsTab N v) = v := by
  funext p
  simp [rowPads, padsTab, p.isLt]

omit [Field F] [Fintype F] [DecidableEq F] in
open Classical in
theorem opPadsZ_isSome {Hs : Hashes F D Sl} {N q r : ℕ} {c : Cap D} {out : OutZC F D Sl} {p : Fin N} {jj : ℕ}
    (hjj : jj < q)
    (hv : Merkle.VerifiesL Hs.H Hs.Ep Hs.Lp (fun _ => c 0) (Nat.log 2 N) p.val ((out.opens.inner r).pads jj).1
      ((out.opens.inner r).pads jj).2.1 ((out.opens.inner r).pads jj).2.2) :
    (opPadsZ Hs N q r c out p).isSome := by
  unfold opPadsZ
  rw [Option.isSome_map, List.isSome_head?]
  exact List.ne_nil_of_mem (List.mem_filter.2 ⟨List.mem_range.2 hjj, decide_eq_true hv⟩)

omit [Field F] [Fintype F] [DecidableEq F] in
/-- **A rep's disagreeing pads opening**: a pads clash within the run, or the run opens a pads position of the fixed
pads oracle to another pair. -/
theorem pads_badZ (Hs : Hashes F D Sl) (N q r : ℕ) (hr : r < 2) (j : IZ F D) (out : OutZC F D Sl)
    (hnot : ¬ PadsAgree Hs N q j (out.opens.inner r)) :
    PadsClashZ Hs N q out ∨
      ∃ t d, j.pads = some (t, d) ∧ ∃ p, Rewinding.Off (opPadsZ Hs N q r d) (rowPads N t) out p := by
  unfold PadsAgree at hnot
  push Not at hnot
  obtain ⟨O, c, hpads, jj, hjj, p, hver, hne⟩ := hnot
  obtain ⟨h, hh, hhv, hop⟩ := opPadsZ_some (opPadsZ_isSome (r := r) (out := out) hjj hver)
  by_cases heq : ((out.opens.inner r).pads h).1 = rowPads N O p
  · exact Or.inl ⟨r, hr, c 0, h, hh, jj, hjj, p, hhv, hver, fun e => hne (e.symm.trans heq)⟩
  · have hoff : (opPadsZ Hs N q r c out p).isSome ∧ opPadsZ Hs N q r c out p ≠ some (rowPads N O p) :=
      ⟨by rw [hop]; rfl, by rw [hop]; exact fun e => heq (Option.some.inj e)⟩
    exact Or.inr ⟨O, c, hpads, p, hoff⟩

/-! ## `τ`'s commitments -/

/-- A `τ` oracle's value, at `(0, 0)`. -/
def rowTau (t : Oracle F) (_ : Unit) : F := t 0 0

omit [Field F] [Fintype F] [DecidableEq F] in
theorem rowTau_tauTab (v : Unit → F) : rowTau (tauTab v) = v := funext fun _ => rfl

omit [Field F] [Fintype F] [DecidableEq F] in
/-- **A rep's disagreeing `τ` opening**: the run opens the fixed `τ` oracle's commitment to another value. -/
theorem tau_badZ (Hs : Hashes F D Sl) (r : ℕ) (j : IZ F D) (out : OutZC F D Sl)
    (hnot : ¬ TauAgree Hs j (out.opens.inner r)) :
    ∃ t d, j.tau = some (t, d) ∧ ∃ p, Rewinding.Off (opTauZ Hs r d) (rowTau t) out p := by
  classical
  unfold TauAgree at hnot
  push Not at hnot
  obtain ⟨Oτ, cτ, htau, hleaf, hne⟩ := hnot
  have hop : opTauZ Hs r cτ out () = some (out.opens.inner r).tau.1 := by
    simp [opTauZ, hleaf]
  have hoff : (opTauZ Hs r cτ out ()).isSome ∧ opTauZ Hs r cτ out () ≠ some (rowTau Oτ ()) :=
    ⟨by rw [hop]; rfl, by rw [hop]; exact fun e => hne (Option.some.inj e)⟩
  exact ⟨Oτ, cτ, htau, (), hoff⟩

/-! ## Self-clashes -/

/-! ## Where a bad run goes wrong -/

/-- **A rep's disagreeing row opening**, on an accepted run: at level 0, off `C₀` or a clash at the witness cap; at a
level `ℓ ≥ 1`, off the table fixed there or a clash at its recorded cap. -/
theorem rep_badZ (Hs : Hashes F D Sl) (T : TabZK F mPts) (C₀ : Oracle F) (cap₀ : Cap D) (out : OutZC F D Sl)
    (r : ℕ) (hr : r < 2) (ir : IRep F D)
    (hs : RecStruct T.sch ir (if r = 0 then out.pend.1 else out.pend.2).lig)
    (hok : OpensOKZ Hs.H Hs.E Hs.Lf (width₀ T) cap₀ r (if r = 0 then out.pend.1 else out.pend.2).lig out.opens.rows
      out.opens.salts)
    (hnot : ¬ AgreeZ r C₀ (width₀ T) ir (if r = 0 then out.pend.1 else out.pend.2).lig.reqs out.opens.rows) :
    ClashW Hs.H Hs.E Hs.Lf cap₀ (width₀ T) (pathLen (lvOf T.sch 0)) (opens₀Z T.sch out) ∨
      (∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C₀) out p) ∨
      ∃ ℓ ∈ Finset.Icc 1 T.sch.levels.length, ∃ t d, ir ℓ = some (t, d) ∧
        (if r = 0 then out.pend.1 else out.pend.2).lig.caps[ℓ - 1]? = some d ∧
        ((∃ p, Rewinding.Off (opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d) (rowT _ _ t) out p) ∨
          Exec.ClashL Hs.H Hs.E Hs.Lf d (lvOf T.sch ℓ).lanes (pathLen (lvOf T.sch ℓ))
            (slotOpensZ out r ℓ (lvOf T.sch ℓ).queries)) := by
  classical
  unfold AgreeZ at hnot
  push Not at hnot
  obtain ⟨req, hreq, t, ht, j, hj, lane, hlane, hne⟩ := hnot
  obtain ⟨hlev, hshape, hcaps, hlen⟩ := hs
  obtain ⟨hQ, hlt, hlanes, hdepth⟩ := hshape req hreq
  obtain ⟨cap, hcapOf, hver⟩ := hok req hreq
  obtain ⟨hrowlen, hpath⟩ := hver j hj
  have hfind := find_level hlev hreq
  have hxmem := mem_slotOpensZ hfind hQ hj
  have hposmem : req.positions.getD j 0 ∈ req.positions := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]
    exact List.getElem_mem hj
  have hℓlt : req.level < (if r = 0 then out.pend.1 else out.pend.2).lig.reqs.length := by
    have := List.mem_map_of_mem (f := Req.level) hreq
    rw [hlev, List.mem_range] at this
    exact this
  have hposlt : req.positions.getD j 0 < 2 ^ (lvOf T.sch req.level).logLen := hlt _ hposmem
  have hrowlen' : (out.opens.rows r req.level j).1.length =
      if req.level = 0 then width₀ T else 2 ^ (lvOf T.sch req.level).lanes := by
    rw [hrowlen, hlanes]
  have hlane' : lane < if req.level = 0 then width₀ T else 2 ^ (lvOf T.sch req.level).lanes := by
    rw [← hlanes]; exact hlane
  have hpath' : Merkle.VerifiesL Hs.H Hs.E Hs.Lf cap (pathLen (lvOf T.sch req.level)) (req.positions.getD j 0)
      (out.opens.rows r req.level j).1 (out.opens.salts r req.level j) (out.opens.rows r req.level j).2 := by
    rw [← hdepth]; exact hpath
  generalize hL : req.level = L at hfind hxmem hℓlt hposlt hrowlen' hlane' hpath' hcapOf ht hne
  cases L with
  | zero =>
    rw [ite_eq_left_of_eq_true _ _ (eq_true rfl)] at hrowlen' hlane'
    have hcap : cap = cap₀ := by
      simp only [capOf, ite_true, Option.some.injEq] at hcapOf; exact hcapOf.symm
    have htC : t = C₀ := by simp only [tabAt, ite_true, Option.some.injEq] at ht; exact ht.symm
    rw [hcap] at hpath'
    rw [htC] at hne
    have hx₀ : (req.positions.getD j 0, (out.opens.rows r 0 j).1, out.opens.salts r 0 j,
        (out.opens.rows r 0 j).2) ∈ opens₀Z T.sch out := by
      unfold opens₀Z
      rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
      · exact List.mem_append_left _ hxmem
      · exact List.mem_append_right _ hxmem
    have hv : VerifiedZ Hs.H Hs.E Hs.Lf cap₀ (width₀ T) (pathLen (lvOf T.sch 0))
        (req.positions.getD j 0, (out.opens.rows r 0 j).1, out.opens.salts r 0 j, (out.opens.rows r 0 j).2) :=
      ⟨hrowlen', hpath'⟩
    have hne' : rowW (width₀ T) (out.opens.rows r 0 j).1 ≠
        rowTW (width₀ T) (lvOf T.sch 0).logLen C₀ ⟨req.positions.getD j 0, hposlt⟩ := fun h =>
      hne (congrFun h ⟨lane, hlane'⟩)
    rcases off_or_clashW Hs.H Hs.E Hs.Lf _ _ _ (opens₀Z T.sch out) cap₀ _ hx₀ ⟨req.positions.getD j 0, hposlt⟩
        rfl hv _ hne' with hoff | hcl
    · exact Or.inr (Or.inl ⟨_, hoff⟩)
    · exact Or.inl hcl
  | succ L =>
    rw [ite_eq_right_of_eq_false _ _ (eq_false (Nat.succ_ne_zero L))] at hrowlen' hlane'
    have hcapS : (if r = 0 then out.pend.1 else out.pend.2).lig.caps[L + 1 - 1]? = some cap := by
      simpa [capOf] using hcapOf
    obtain ⟨t', d', hi, hcd⟩ := hcaps (L + 1) (by omega) hℓlt
    have htd : t' = t := by simpa [tabAt, hi] using ht
    have hdc : d' = cap := Option.some.inj (hcd.symm.trans hcapS)
    subst htd hdc
    have hv : Exec.VerifiedL Hs.H Hs.E Hs.Lf d' (lvOf T.sch (L + 1)).lanes (pathLen (lvOf T.sch (L + 1)))
        (req.positions.getD j 0, (out.opens.rows r (L + 1) j).1, out.opens.salts r (L + 1) j,
          (out.opens.rows r (L + 1) j).2) :=
      ⟨hrowlen', hpath'⟩
    have hne' : rowFin (lvOf T.sch (L + 1)).lanes (out.opens.rows r (L + 1) j).1 ≠
        rowT (lvOf T.sch (L + 1)).lanes (lvOf T.sch (L + 1)).logLen t' ⟨req.positions.getD j 0, hposlt⟩ :=
      fun h => hne (congrFun h ⟨lane, hlane'⟩)
    refine Or.inr (Or.inr ⟨L + 1, Finset.mem_Icc.2 ⟨by omega, by omega⟩, t', d', hi, hcd, ?_⟩)
    rcases Exec.off_or_clashL Hs.H Hs.E Hs.Lf _ _ _ _ d' _ hxmem ⟨req.positions.getD j 0, hposlt⟩ rfl hv _ hne' with
      hoff | hcl
    · exact Or.inl ⟨_, hoff⟩
    · exact Or.inr hcl

theorem getPairZ_lev (r : ℕ) (hr : r < 2) (i : IZ F D × IZ F D) (ℓ : ℕ) (hℓ : 1 ≤ ℓ) :
    Lock.getPair (getZ 0) (getZ 1) i (slotOf r ℓ + 4) = (if r = 0 then i.1 else i.2).lev ℓ := by
  rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
  · have h1 : getRep 1 i.2.lev (slotOf 0 ℓ) = none := by
      unfold getRep slotOf
      split_ifs with h
      · omega
      · rfl
    show (getZ 0 i.1 (slotOf 0 ℓ + 4)).orElse (fun _ => getZ 1 i.2 (slotOf 0 ℓ + 4)) = i.1.lev ℓ
    rw [getZ_lev, getZ_lev, getRep_slot 0 (by norm_num) i.1.lev ℓ hℓ, h1]
    cases i.1.lev ℓ <;> rfl
  · have h0 : getRep 0 i.1.lev (slotOf 1 ℓ) = none := by
      unfold getRep slotOf
      split_ifs with h
      · omega
      · rfl
    show (getZ 0 i.1 (slotOf 1 ℓ + 4)).orElse (fun _ => getZ 1 i.2 (slotOf 1 ℓ + 4)) = i.2.lev ℓ
    rw [getZ_lev, getZ_lev, getRep_slot 1 (by norm_num) i.2.lev ℓ hℓ, h0]
    rfl

omit [Field F] [Fintype F] [DecidableEq F] in
theorem getPairZ_pads (r : ℕ) (hr : r < 2) (i : IZ F D × IZ F D) :
    Lock.getPair (getZ 0) (getZ 1) i r = (if r = 0 then i.1 else i.2).pads := by
  rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
  · show (getZ 0 i.1 0).orElse (fun _ => getZ 1 i.2 0) = i.1.pads
    rw [show getZ 0 i.1 0 = i.1.pads by simp [getZ], show getZ 1 i.2 0 = none by simp [getZ]]
    cases i.1.pads <;> rfl
  · show (getZ 0 i.1 1).orElse (fun _ => getZ 1 i.2 1) = i.2.pads
    rw [show getZ 0 i.1 1 = none by simp [getZ], show getZ 1 i.2 1 = i.2.pads by simp [getZ]]
    rfl

omit [Field F] [Fintype F] [DecidableEq F] in
theorem getPairZ_tau (r : ℕ) (hr : r < 2) (i : IZ F D × IZ F D) :
    Lock.getPair (getZ 0) (getZ 1) i (2 + r) = (if r = 0 then i.1 else i.2).tau := by
  rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
  · show (getZ 0 i.1 (2 + 0)).orElse (fun _ => getZ 1 i.2 (2 + 0)) = i.1.tau
    rw [show getZ 0 i.1 (2 + 0) = i.1.tau by simp [getZ], show getZ 1 i.2 (2 + 0) = none by simp [getZ]]
    cases i.1.tau <;> rfl
  · show (getZ 0 i.1 (2 + 1)).orElse (fun _ => getZ 1 i.2 (2 + 1)) = i.2.tau
    rw [show getZ 0 i.1 (2 + 1) = none by simp [getZ], show getZ 1 i.2 (2 + 1) = i.2.tau by simp [getZ]]
    rfl

/-- **The openings agree with every fixed table**: both reps' rows (`AgreeZ`), pads pairs (`PadsAgree`) and `τ`
(`TauAgree`). -/
def AgreeAllZ (Hs : Hashes F D Sl) (T : TabZK F mPts) (C₀ : Oracle F) (i : IZ F D × IZ F D) (out : OutZC F D Sl) :
    Prop :=
  AgreeZ 0 C₀ (width₀ T) i.1.lev out.pend.1.lig.reqs out.opens.rows ∧
    AgreeZ 1 C₀ (width₀ T) i.2.lev out.pend.2.lig.reqs out.opens.rows ∧
    PadsAgree Hs T.N T.q i.1 (out.opens.inner 0) ∧ PadsAgree Hs T.N T.q i.2 (out.opens.inner 1) ∧
    TauAgree Hs i.1 (out.opens.inner 0) ∧ TauAgree Hs i.2 (out.opens.inner 1)

/-- **Where a bad run goes wrong.** An accepted run that disagrees with the fixed tables either clashes within the run,
or opens a position off `C₀`, off the table fixed at one of the reps' caps, off a rep's fixed pads oracle, or off a
rep's fixed `τ` oracle. -/
theorem bad_splitZ (Hs : Hashes F D Sl) (T : TabZK F mPts) (C₀ : Oracle F) (cap₀ : Cap D) (i : IZ F D × IZ F D)
    (a : Bool × Oracle F) (out : OutZC F D Sl) (hR : TopRelZ Hs T C₀ cap₀ i a out) (hacc : out.accepted = true)
    (hbad : ¬ AgreeAllZ Hs T C₀ i out) :
    SelfClashZ Hs T out ∨ (∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C₀) out p) ∨
      (∃ r ∈ Finset.range 2, ∃ ℓ ∈ Finset.Icc 1 T.sch.levels.length, ∃ t d,
        Lock.getPair (getZ 0) (getZ 1) i (slotOf r ℓ + 4) = some (t, d) ∧
          ∃ p, Rewinding.Off (opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d) (rowT _ _ t) out p) ∨
      (∃ r ∈ Finset.range 2, ∃ t d, Lock.getPair (getZ 0) (getZ 1) i r = some (t, d) ∧
        ∃ p, Rewinding.Off (opPadsZ Hs T.N T.q r d) (rowPads T.N t) out p) ∨
      ∃ r ∈ Finset.range 2, ∃ t d, Lock.getPair (getZ 0) (getZ 1) i (2 + r) = some (t, d) ∧
        ∃ p, Rewinding.Off (opTauZ Hs r d) (rowTau t) out p := by
  obtain ⟨-, hcap, -, hok, hs₀, hs₁⟩ := hR
  obtain ⟨hok₀, hok₁⟩ := hok hacc
  have hcase : (∃ r < 2, RecStruct T.sch (if r = 0 then i.1 else i.2).lev
        (if r = 0 then out.pend.1 else out.pend.2).lig ∧
      OpensOKZ Hs.H Hs.E Hs.Lf (width₀ T) cap₀ r (if r = 0 then out.pend.1 else out.pend.2).lig out.opens.rows
        out.opens.salts ∧
      ¬ AgreeZ r C₀ (width₀ T) (if r = 0 then i.1 else i.2).lev (if r = 0 then out.pend.1 else out.pend.2).lig.reqs
        out.opens.rows) ∨
      (∃ r < 2, ¬ PadsAgree Hs T.N T.q (if r = 0 then i.1 else i.2) (out.opens.inner r)) ∨
      ∃ r < 2, ¬ TauAgree Hs (if r = 0 then i.1 else i.2) (out.opens.inner r) := by
    unfold AgreeAllZ at hbad
    by_cases h0 : AgreeZ 0 C₀ (width₀ T) i.1.lev out.pend.1.lig.reqs out.opens.rows
    · by_cases h1 : AgreeZ 1 C₀ (width₀ T) i.2.lev out.pend.2.lig.reqs out.opens.rows
      · by_cases p0 : PadsAgree Hs T.N T.q i.1 (out.opens.inner 0)
        · by_cases p1 : PadsAgree Hs T.N T.q i.2 (out.opens.inner 1)
          · by_cases t0 : TauAgree Hs i.1 (out.opens.inner 0)
            · have t1 : ¬ TauAgree Hs i.2 (out.opens.inner 1) := fun t1 => hbad ⟨h0, h1, p0, p1, t0, t1⟩
              exact Or.inr (Or.inr ⟨1, by norm_num, by simpa using t1⟩)
            · exact Or.inr (Or.inr ⟨0, by norm_num, by simpa using t0⟩)
          · exact Or.inr (Or.inl ⟨1, by norm_num, by simpa using p1⟩)
        · exact Or.inr (Or.inl ⟨0, by norm_num, by simpa using p0⟩)
      · exact Or.inl ⟨1, by norm_num, by simpa using hs₁, by simpa using hok₁, by simpa using h1⟩
    · exact Or.inl ⟨0, by norm_num, by simpa using hs₀, by simpa using hok₀, by simpa using h0⟩
  rcases hcase with ⟨r, hr, hs, hok, hnot⟩ | ⟨r, hr, hnot⟩ | ⟨r, hr, hnot⟩
  · rcases rep_badZ Hs T C₀ cap₀ out r hr _ hs hok hnot with hcl | hoff | ⟨ℓ, hℓ, t, d, hi, hc, h⟩
    · exact Or.inl (Or.inl (hcap ▸ hcl))
    · exact Or.inr (Or.inl hoff)
    · rcases h with hoff | hcl
      · exact Or.inr (Or.inr (Or.inl ⟨r, Finset.mem_range.2 hr, ℓ, hℓ, t, d,
          by rw [getPairZ_lev r hr i ℓ (Finset.mem_Icc.1 hℓ).1, hi], hoff⟩))
      · exact Or.inl (Or.inr (Or.inl ⟨r, hr, ℓ, (Finset.mem_Icc.1 hℓ).1, d, hc, hcl⟩))
  · rcases pads_badZ Hs T.N T.q r hr _ out hnot with hcl | ⟨t, d, hp, h⟩
    · exact Or.inl (Or.inr (Or.inr hcl))
    · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨r, Finset.mem_range.2 hr, t, d, by rw [getPairZ_pads r hr i, hp], h⟩)))
  · obtain ⟨t, d, hp, h⟩ := tau_badZ Hs r _ out hnot
    exact Or.inr (Or.inr (Or.inr (Or.inr ⟨r, Finset.mem_range.2 hr, t, d, by rw [getPairZ_tau r hr i, hp], h⟩)))

end FlockSoundness.Discharge.ZkLink
