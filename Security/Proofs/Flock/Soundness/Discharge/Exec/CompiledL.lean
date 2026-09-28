import Proofs.Flock.Soundness.CompiledSound

/-!
# Compiled soundness of one table under a leaf scheme (README.md §1.2, PROTOCOL.md §14)

`table_sound_compiled` (`CompiledSound.lean`) bounds `tableC`, whose openings verify from the unsalted leaf
`H(encode row)`. This file proves the same bound for `Model.tableCL A H E Lf`, whose openings carry salts and verify
from the row's leaf under its salt (`OpensOKL`), for any leaf scheme `Lf` (`Merkle.Leaf`). At `Refine.leaf512` that
is the `hm96-sha512/v1` table flock-verify checks (`accepts_tables_hm96`).

The argument is `CompiledSound`'s, and only the openings change. A salted opening is `(position, row, salt, path)`. A
clash is two verifying openings of one position to different rows, under any salts, and it is a collision of `H` by
`Merkle.opening_bindingL`: the leaf scheme's binding (`Leaf.bind`) does what the encoding's injectivity does for the
unsalted leaf. The lockstep with the oracle table (`lockTableL`) is `lockTable` with the salts in the openings'
message; the reps' locksteps (`lockRep`) are unchanged.

* `table_sound_compiledL`: `Pr[accepted ∧ T*(P*) uncommitted] ≤ tableError + √(N₀·Q₀·Adv₀) + Σ √(N_ℓ·Q_ℓ·Adv_{r,ℓ})
  + Pr[self-clash]`, with each `Adv` and the self-clash a collision of `H` (`op₀L_conflict_collision`,
  `opRL_conflict_collision`, `SelfClashL.collision`).
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type}

/-! ## The lockstep -/

/-- The oracle table's continuation after the level-0 table `C₀`, with the salted openings' message as its dummy last
message. -/
def tableOrcL (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (C₀ : Oracle F) : Game (Bool × Oracle F) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    (interleave (rep A S sch hm C₀ (extraClaims S ptLocal mPts pts))
      (rep A S sch hm C₀ (extraClaims S ptLocal mPts pts))).bind
      fun acc => Game.send (Opens F D × Salts Sl) fun _ => Game.ret (acc.1 && acc.2, C₀)

omit [Fintype K] in
theorem withDummy_tableL (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) :
    Lock.withDummy (Opens F D × Salts Sl) (table A S sch hm ptLocal mPts) =
      Game.send (Oracle F) (tableOrcL (D := D) (Sl := Sl) A S sch hm ptLocal mPts) := by
  refine congrArg (Game.send (Oracle F)) (funext fun C₀ =>
    congrArg (@Game.coin _ (Fin 2 → Fin mPts → F) _ _) (funext fun pts => ?_))
  simp only [Game.bind_ret, Game.bind_eq, Game.pure_eq, Game.bind_assoc]

open Classical in
/-- The salted compiled table's continuation after the level-0 cap. -/
noncomputable def tableCmpL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (cap₀ : Cap D) : Game (OutCL F D Sl) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    (interleave (repC A 0 S sch hm (extraClaims S ptLocal mPts pts))
      (repC A 1 S sch hm (extraClaims S ptLocal mPts pts))).bind
      fun pend => Game.send (Opens F D × Salts Sl) fun os => Game.ret
        ⟨pend.1.verdict os.1 && pend.2.verdict os.1 &&
          decide (OpensOKL H E Lf cap₀ 0 pend.1 os.1 os.2 ∧ OpensOKL H E Lf cap₀ 1 pend.2 os.1 os.2), cap₀, pend,
          os.1, os.2⟩

omit [Fintype K] in
theorem tableCL_send (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) :
    tableCL A H E Lf S sch hm ptLocal mPts = Game.send (Cap D) (tableCmpL A H E Lf S sch hm ptLocal mPts) :=
  rfl

/-- **The salted table's leaf relation** below the level-0 cap. -/
def TopRelL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (sch : Schedule)
    (C₀ : Oracle F) (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F) (out : OutCL F D Sl) : Prop :=
  a.2 = C₀ ∧ out.cap₀ = cap₀ ∧
    (out.accepted = true → AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens →
      AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens → a.1 = true) ∧
    (out.accepted = true →
      OpensOKL H E Lf cap₀ 0 out.pend.1 out.opens out.salts ∧ OpensOKL H E Lf cap₀ 1 out.pend.2 out.opens out.salts) ∧
    RecStruct sch i.1 out.pend.1 ∧ RecStruct sch i.2 out.pend.2

/-- **The salted table in lockstep** below the level-0 cap: the link points, both reps interleaved, the openings and
their salts. -/
noncomputable def lockTableL [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₀ : Oracle F)
    (cap₀ : Cap D) :
    Lock (Oracle F) (Cap D) (Lock.getPair (getRep 0) (getRep 1))
      (TopRelL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀ cap₀) ((fun _ => none), (fun _ => none))
      (tableOrcL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts C₀)
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀) :=
  .coin (Fin 2 → Fin mPts → F) fun pts =>
    Lock.bind (R := Lock.RPair (RecRel 0 C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (RecRel 1 C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (fun i acc pend h => .send (Opens F D × Salts Sl) fun os => .ret (by
        obtain ⟨⟨hv₀, hs₀⟩, ⟨hv₁, hs₁⟩⟩ := h
        refine ⟨rfl, rfl, fun hacc ha₀ ha₁ => ?_, fun hacc => ?_, hs₀, hs₁⟩
        · simp only [Bool.and_eq_true, decide_eq_true_eq] at hacc
          rw [← hv₀ os.1 ha₀, ← hv₁ os.1 ha₁]
          simp [hacc.1.1, hacc.1.2]
        · simp only [Bool.and_eq_true, decide_eq_true_eq] at hacc
          exact hacc.2))
      (Lock.interleaveLock disjoint_reps
        (lockRep A 0 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₀ (extraClaims S ptLocal mPts pts))
        (lockRep A 1 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₀ (extraClaims S ptLocal mPts pts)))

/-! ## Salted openings -/

/-- The salted openings of rep `r`'s first level-`ℓ` request, `(position, row, salt, path)` for its first `Q`
queries. -/
def slotOpensL (out : OutCL F D Sl) (r ℓ Q : ℕ) : List (ℕ × List F × Sl × List D) :=
  match ((if r = 0 then out.pend.1 else out.pend.2).reqs.find? fun req => req.level == ℓ) with
  | none => []
  | some req => ((req.positions.take Q).zipIdx).map fun pj =>
      (pj.1, (out.opens r ℓ pj.2).1, out.salts r ℓ pj.2, (out.opens r ℓ pj.2).2)

omit [Field F] [Fintype F] [DecidableEq F] in
theorem slotOpensL_length (out : OutCL F D Sl) (r ℓ Q : ℕ) : (slotOpensL out r ℓ Q).length ≤ Q := by
  unfold slotOpensL
  split
  · simp
  · simp only [List.length_map, List.length_zipIdx, List.length_take]
    exact min_le_left _ _

/-- A salted opening verifies against `cap`: a row of `2^lanes` elements whose path verifies from its leaf under its
salt. -/
def VerifiedL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cap : Cap D)
    (lanes depth : ℕ) (x : ℕ × List F × Sl × List D) : Prop :=
  x.2.1.length = 2 ^ lanes ∧ Merkle.VerifiesL H E Lf cap depth x.1 x.2.1 x.2.2.1 x.2.2.2

open Classical in
/-- The row a run opens at `p`: its first verifying opening there. -/
noncomputable def opOfL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (lanes depth d : ℕ) (xs : List (ℕ × List F × Sl × List D)) (cap : Cap D) (p : Fin (2 ^ d)) :
    Option (Fin (2 ^ lanes) → F) :=
  ((xs.filter fun x => decide (x.1 = p.val ∧ VerifiedL H E Lf cap lanes depth x)).head?).map
    fun x => rowFin lanes x.2.1

/-- Two verifying openings of one position to different rows, under any salts. -/
def ClashL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cap : Cap D)
    (lanes depth : ℕ) (xs : List (ℕ × List F × Sl × List D)) : Prop :=
  ∃ x ∈ xs, ∃ y ∈ xs, x.1 = y.1 ∧ VerifiedL H E Lf cap lanes depth x ∧
    VerifiedL H E Lf cap lanes depth y ∧ x.2.1 ≠ y.2.1

omit [Fintype F] [DecidableEq F] in
open Classical in
/-- A run opens at most as many positions as it has openings. -/
theorem opOfL_count (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (lanes depth d : ℕ) (xs : List (ℕ × List F × Sl × List D)) (cap : Cap D) :
    (Finset.univ.filter fun p => (opOfL H E Lf lanes depth d xs cap p).isSome).card ≤ xs.length := by
  calc (Finset.univ.filter fun p => (opOfL H E Lf lanes depth d xs cap p).isSome).card
      ≤ (xs.map fun x => x.1).toFinset.card := by
        refine Finset.card_le_card_of_injOn (fun p => p.val) (fun p hp => ?_)
          (fun p _ q _ h => Fin.ext h)
        simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq] at hp
        unfold opOfL at hp
        rw [Option.isSome_map] at hp
        obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hp
        have hmem := List.mem_of_mem_head? hx
        rw [List.mem_filter, decide_eq_true_eq] at hmem
        simp only [List.coe_toFinset, List.mem_map, Set.mem_setOf_eq]
        exact ⟨x, hmem.1, hmem.2.1⟩
    _ ≤ (xs.map fun x => x.1).length := List.toFinset_card_le _
    _ = xs.length := List.length_map _

omit [Fintype F] in
/-- **Off or clash.** A verifying opening of `p` whose row is not `v`: the run opens `p` to a row other than `v`, or it
opens `p` two ways. -/
theorem off_or_clashL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (lanes depth d : ℕ) (xs : List (ℕ × List F × Sl × List D)) (cap : Cap D) (x : ℕ × List F × Sl × List D)
    (hx : x ∈ xs) (p : Fin (2 ^ d)) (hxp : x.1 = p.val) (hv : VerifiedL H E Lf cap lanes depth x)
    (v : Fin (2 ^ lanes) → F) (hne : rowFin lanes x.2.1 ≠ v) :
    ((opOfL H E Lf lanes depth d xs cap p).isSome ∧ opOfL H E Lf lanes depth d xs cap p ≠ some v) ∨
      ClashL H E Lf cap lanes depth xs := by
  classical
  unfold opOfL
  set ys := xs.filter fun x => decide (x.1 = p.val ∧ VerifiedL H E Lf cap lanes depth x)
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

/-! ## Per-cap openings -/

/-- Both reps' level-0 salted openings. -/
def opens₀L (sch : Schedule) (out : OutCL F D Sl) : List (ℕ × List F × Sl × List D) :=
  slotOpensL out 0 0 (lvOf sch 0).queries ++ slotOpensL out 1 0 (lvOf sch 0).queries

/-- The level-0 cap's openings, by position. -/
noncomputable def op₀L (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (cap : Cap D) (out : OutCL F D Sl) :
    Fin (2 ^ (lvOf sch 0).logLen) → Option (Fin (2 ^ (lvOf sch 0).lanes) → F) :=
  opOfL H E Lf (lvOf sch 0).lanes (pathLen (lvOf sch 0)) (lvOf sch 0).logLen (opens₀L sch out) cap

/-- Rep `r`'s level-`ℓ` cap's openings, by position. -/
noncomputable def opRL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (r ℓ : ℕ) (cap : Cap D) (out : OutCL F D Sl) :
    Fin (2 ^ (lvOf sch ℓ).logLen) → Option (Fin (2 ^ (lvOf sch ℓ).lanes) → F) :=
  opOfL H E Lf (lvOf sch ℓ).lanes (pathLen (lvOf sch ℓ)) (lvOf sch ℓ).logLen
    (slotOpensL out r ℓ (lvOf sch ℓ).queries) cap

/-- **A self-clash**: within one run, two verifying openings of a position to different rows, at the level-0 cap or
at a rep's recorded cap. -/
def SelfClashL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (out : OutCL F D Sl) : Prop :=
  ClashL H E Lf out.cap₀ (lvOf sch 0).lanes (pathLen (lvOf sch 0)) (opens₀L sch out) ∨
    ∃ r < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap,
      (if r = 0 then out.pend.1 else out.pend.2).caps[ℓ - 1]? = some cap ∧
        ClashL H E Lf cap (lvOf sch ℓ).lanes (pathLen (lvOf sch ℓ))
          (slotOpensL out r ℓ (lvOf sch ℓ).queries)

/-! ## Where a bad run goes wrong -/

omit [Field F] [Fintype F] [DecidableEq F] in
theorem mem_slotOpensL {out : OutCL F D Sl} {r : ℕ} {req : Req}
    (hfind : (if r = 0 then out.pend.1 else out.pend.2).reqs.find? (fun q => q.level == req.level) =
      some req)
    {Q : ℕ} (hQ : req.positions.length = Q) {j : ℕ} (hj : j < req.positions.length) :
    (req.positions.getD j 0, (out.opens r req.level j).1, out.salts r req.level j, (out.opens r req.level j).2) ∈
      slotOpensL out r req.level Q := by
  unfold slotOpensL
  rw [hfind]
  simp only [List.mem_map]
  refine ⟨(req.positions.getD j 0, j), ?_, rfl⟩
  rw [List.take_of_length_le hQ.le, List.mem_iff_getElem]
  refine ⟨j, by simpa using hj, ?_⟩
  rw [List.getElem_zipIdx]
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]

omit [Fintype F] in
/-- **A rep's disagreeing opening**, on an accepted run: at level 0, off `C₀` or a clash at the level-0 cap; at a
level `ℓ ≥ 1`, off the table fixed there or a clash at its recorded cap. -/
theorem rep_badL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (C₀ : Oracle F) (cap₀ : Cap D) (out : OutCL F D Sl) (r : ℕ) (hr : r < 2) (ir : IRep F D)
    (hs : RecStruct sch ir (if r = 0 then out.pend.1 else out.pend.2))
    (hok : OpensOKL H E Lf cap₀ r (if r = 0 then out.pend.1 else out.pend.2) out.opens out.salts)
    (hnot : ¬ AgreeRep r C₀ ir (if r = 0 then out.pend.1 else out.pend.2).reqs out.opens) :
    ClashL H E Lf cap₀ (lvOf sch 0).lanes (pathLen (lvOf sch 0)) (opens₀L sch out) ∨
      (∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) out p) ∨
      ∃ ℓ ∈ Finset.Icc 1 sch.levels.length, ∃ t d, ir ℓ = some (t, d) ∧
        (if r = 0 then out.pend.1 else out.pend.2).caps[ℓ - 1]? = some d ∧
        ((∃ p, Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ t) out p) ∨
          ClashL H E Lf d (lvOf sch ℓ).lanes (pathLen (lvOf sch ℓ))
            (slotOpensL out r ℓ (lvOf sch ℓ).queries)) := by
  classical
  unfold AgreeRep at hnot
  push_neg at hnot
  obtain ⟨req, hreq, t, ht, j, hj, lane, hlane, hne⟩ := hnot
  obtain ⟨hlev, hshape, hcaps, hlen⟩ := hs
  obtain ⟨hQ, hlt, hlanes, hdepth⟩ := hshape req hreq
  obtain ⟨cap, hcapOf, hver⟩ := hok req hreq
  obtain ⟨hrowlen, hpath⟩ := hver j hj
  have hfind := find_level hlev hreq
  have hxmem := mem_slotOpensL hfind hQ hj
  have hposmem : req.positions.getD j 0 ∈ req.positions := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]
    exact List.getElem_mem hj
  have hℓlt : req.level < (if r = 0 then out.pend.1 else out.pend.2).reqs.length := by
    have := List.mem_map_of_mem (f := Req.level) hreq
    rw [hlev, List.mem_range] at this
    exact this
  have hv : VerifiedL H E Lf cap (lvOf sch req.level).lanes (pathLen (lvOf sch req.level))
      (req.positions.getD j 0, (out.opens r req.level j).1, out.salts r req.level j,
        (out.opens r req.level j).2) :=
    ⟨by rw [hrowlen, hlanes], by rw [← hdepth]; exact hpath⟩
  have hposlt : req.positions.getD j 0 < 2 ^ (lvOf sch req.level).logLen := hlt _ hposmem
  have hlane' : lane < 2 ^ (lvOf sch req.level).lanes := by rw [← hlanes]; exact hlane
  generalize hL : req.level = L at hfind hxmem hℓlt hv hposlt hlane' hcapOf ht hne
  have hne' : rowFin (lvOf sch L).lanes (out.opens r L j).1 ≠
      rowT (lvOf sch L).lanes (lvOf sch L).logLen t ⟨req.positions.getD j 0, hposlt⟩ := fun h =>
    hne (congrFun h ⟨lane, hlane'⟩)
  cases L with
  | zero =>
    have hcap : cap = cap₀ := by
      simp only [capOf, if_true, Option.some.injEq] at hcapOf; exact hcapOf.symm
    have htC : t = C₀ := by simp only [tabAt, if_true, Option.some.injEq] at ht; exact ht.symm
    subst hcap htC
    have hx₀ : (req.positions.getD j 0, (out.opens r 0 j).1, out.salts r 0 j, (out.opens r 0 j).2) ∈
        opens₀L sch out := by
      unfold opens₀L
      rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
      · exact List.mem_append_left _ hxmem
      · exact List.mem_append_right _ hxmem
    rcases off_or_clashL H E Lf _ _ _ (opens₀L sch out) cap _ hx₀ ⟨req.positions.getD j 0, hposlt⟩ rfl
        hv _ hne' with hoff | hcl
    · exact Or.inr (Or.inl ⟨_, hoff⟩)
    · exact Or.inl hcl
  | succ L =>
    have hcapS : (if r = 0 then out.pend.1 else out.pend.2).caps[L + 1 - 1]? = some cap := by
      simpa [capOf] using hcapOf
    obtain ⟨t', d', hi, hcd⟩ := hcaps (L + 1) (by omega) hℓlt
    have htd : t' = t := by simpa [tabAt, hi] using ht
    have hdc : d' = cap := Option.some.inj (hcd.symm.trans hcapS)
    subst htd hdc
    refine Or.inr (Or.inr ⟨L + 1, Finset.mem_Icc.2 ⟨by omega, by omega⟩, t', d', hi, hcd, ?_⟩)
    rcases off_or_clashL H E Lf _ _ _ _ d' _ hxmem ⟨req.positions.getD j 0, hposlt⟩ rfl hv _ hne' with
      hoff | hcl
    · exact Or.inl ⟨_, hoff⟩
    · exact Or.inr hcl

/-- **Where a bad run goes wrong.** An accepted run that disagrees with the fixed tables either clashes within the run,
or opens a position off `C₀`, or opens a position off the table fixed at one of the reps' caps. -/
theorem bad_splitL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (C₀ : Oracle F) (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F)
    (out : OutCL F D Sl) (hR : TopRelL H E Lf sch C₀ cap₀ i a out) (hacc : out.accepted = true)
    (hbad : ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
      AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) :
    SelfClashL H E Lf sch out ∨ (∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) out p) ∨
      ∃ r ∈ Finset.range 2, ∃ ℓ ∈ Finset.Icc 1 sch.levels.length, ∃ t d,
        Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ t) out p := by
  obtain ⟨-, hcap, -, hok, hs₀, hs₁⟩ := hR
  obtain ⟨hok₀, hok₁⟩ := hok hacc
  have key : ∀ r (hr : r < 2) (ir : IRep F D),
      (∀ ℓ, 1 ≤ ℓ → Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = ir ℓ) →
      RecStruct sch ir (if r = 0 then out.pend.1 else out.pend.2) →
      OpensOKL H E Lf cap₀ r (if r = 0 then out.pend.1 else out.pend.2) out.opens out.salts →
      ¬ AgreeRep r C₀ ir (if r = 0 then out.pend.1 else out.pend.2).reqs out.opens →
      SelfClashL H E Lf sch out ∨ (∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) out p) ∨
        ∃ r ∈ Finset.range 2, ∃ ℓ ∈ Finset.Icc 1 sch.levels.length, ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ t) out p := by
    intro r hr ir hget hs hok hnot
    rcases rep_badL H E Lf sch C₀ cap₀ out r hr ir hs hok hnot with hcl | hoff | ⟨ℓ, hℓ, t, d, hi, hc, h⟩
    · exact Or.inl (Or.inl (hcap ▸ hcl))
    · exact Or.inr (Or.inl hoff)
    · rcases h with hoff | hcl
      · exact Or.inr (Or.inr ⟨r, Finset.mem_range.2 hr, ℓ, hℓ, t, d,
          by rw [hget ℓ (Finset.mem_Icc.1 hℓ).1, hi], hoff⟩)
      · exact Or.inl (Or.inr ⟨r, hr, ℓ, (Finset.mem_Icc.1 hℓ).1, d, hc, hcl⟩)
  by_cases h₀ : AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens
  · have h₁ : ¬ AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens := fun h₁ => hbad ⟨h₀, h₁⟩
    refine key 1 (by norm_num) i.2 (fun ℓ hℓ => ?_) (by simpa using hs₁) (by simpa using hok₁)
      (by simpa using h₁)
    have h0 : getRep 0 i.1 (slotOf 1 ℓ) = none := by
      unfold getRep slotOf; rw [if_neg (by omega)]
    simp only [Lock.getPair, h0, Option.orElse_none, getRep_slot 1 (by norm_num) i.2 ℓ hℓ]
  · refine key 0 (by norm_num) i.1 (fun ℓ hℓ => ?_) (by simpa using hs₀) (by simpa using hok₀)
      (by simpa using h₀)
    unfold Lock.getPair
    rw [getRep_slot 0 (by norm_num) i.1 ℓ hℓ]
    have h1 : getRep 1 i.2 (slotOf 0 ℓ) = none := by
      unfold getRep slotOf; rw [if_neg (by omega)]
    cases i.1 ℓ <;> simp [h1]

/-! ## Clashes and conflicts are collisions -/

omit [Fintype F] [DecidableEq F] in
open Classical in
theorem opOfL_some {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {lanes depth d : ℕ} {xs : List (ℕ × List F × Sl × List D)} {cap : Cap D} {p : Fin (2 ^ d)}
    (h : (opOfL H E Lf lanes depth d xs cap p).isSome) :
    ∃ x ∈ xs, x.1 = p.val ∧ VerifiedL H E Lf cap lanes depth x ∧
      opOfL H E Lf lanes depth d xs cap p = some (rowFin lanes x.2.1) := by
  unfold opOfL at h ⊢
  rw [Option.isSome_map] at h
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 h
  have hmem := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq] at hmem
  exact ⟨x, hmem.1, hmem.2.1, hmem.2.2, by rw [hx]; rfl⟩

omit [Field F] [Fintype F] [DecidableEq F] in
/-- **A clash is a collision**, under any salts (`Merkle.opening_bindingL`). -/
theorem ClashL.collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {cap : Cap D} {lanes depth : ℕ} {xs : List (ℕ × List F × Sl × List D)} (h : ClashL H E Lf cap lanes depth xs) :
    Merkle.Collision H := by
  obtain ⟨x, -, y, -, hxy, hx, hy, hne⟩ := h
  exact Merkle.opening_bindingL H E Lf cap depth x.1 x.2.1 y.2.1 x.2.2.1 y.2.2.1 x.2.2.2 y.2.2.2 hne hx.2
    (hxy ▸ hy.2)

omit [Field F] [Fintype F] [DecidableEq F] in
/-- **A self-clash is a collision.** -/
theorem SelfClashL.collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {sch : Schedule} {out : OutCL F D Sl} (h : SelfClashL H E Lf sch out) : Merkle.Collision H := by
  rcases h with h | ⟨_, _, _, _, _, _, h⟩
  · exact h.collision
  · exact h.collision

omit [Fintype F] [DecidableEq F] in
/-- Two runs that open a position to different rows under one cap give a collision. -/
theorem opOfL_conflict_collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D}
    {Lf : Merkle.Leaf (List F) Sl D H} {lanes depth d : ℕ} {cap : Cap D} {xs xs' : List (ℕ × List F × Sl × List D)}
    {p : Fin (2 ^ d)} (h1 : (opOfL H E Lf lanes depth d xs cap p).isSome)
    (h2 : (opOfL H E Lf lanes depth d xs' cap p).isSome)
    (hne : opOfL H E Lf lanes depth d xs cap p ≠ opOfL H E Lf lanes depth d xs' cap p) :
    Merkle.Collision H := by
  obtain ⟨x, -, hxp, hx, hox⟩ := opOfL_some h1
  obtain ⟨y, -, hyp, hy, hoy⟩ := opOfL_some h2
  refine Merkle.opening_bindingL H E Lf cap depth p.val x.2.1 y.2.1 x.2.2.1 y.2.2.1 x.2.2.2 y.2.2.2
    (fun hxy => hne ?_) (hxp ▸ hx.2) (hyp ▸ hy.2)
  rw [hox, hoy, hxy]

omit [Fintype F] [DecidableEq F] in
/-- **Every pair the level-0 collision finder counts is a collision.** -/
theorem op₀L_conflict_collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D}
    {Lf : Merkle.Leaf (List F) Sl D H} {sch : Schedule} {cap : Cap D} {x : OutCL F D Sl × OutCL F D Sl}
    {p : Fin (2 ^ (lvOf sch 0).logLen)} (h : Rewinding.ConflictAt (op₀L H E Lf sch cap) x p) :
    Merkle.Collision H :=
  opOfL_conflict_collision h.1 h.2.1 h.2.2

omit [Fintype F] [DecidableEq F] in
/-- **Every pair a rep's collision finder counts is a collision.** -/
theorem opRL_conflict_collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D}
    {Lf : Merkle.Leaf (List F) Sl D H} {sch : Schedule} {r ℓ : ℕ} {cap : Cap D} {x : OutCL F D Sl × OutCL F D Sl}
    {p : Fin (2 ^ (lvOf sch ℓ).logLen)} (h : Rewinding.ConflictAt (opRL H E Lf sch r ℓ cap) x p) :
    Merkle.Collision H :=
  opOfL_conflict_collision h.1 h.2.1 h.2.2

/-! ## The plurality prover and the collision finders -/

open Classical in
/-- The table `P̃` sends at a rep's cap: the plurality of the openings there. -/
noncomputable def plurCL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) : Lock.Chooser (Oracle F) (Cap D) (OutCL F D Sl) :=
  fun s d g σ => tabOf (lvOf sch (s / 2)).lanes (lvOf sch (s / 2)).logLen
    (Classical.choose (Rewinding.exists_pluralityE (opRL H E Lf sch (s % 2) (s / 2) d) g σ))

theorem plurCL_plurality (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (r ℓ : ℕ) (hr : r < 2) (d : Cap D) (g : Game (OutCL F D Sl)) (σ : Strategy g) :
    Rewinding.IsPluralityE (opRL H E Lf sch r ℓ d) g σ
      (rowT (lvOf sch ℓ).lanes (lvOf sch ℓ).logLen (plurCL H E Lf sch (slotOf r ℓ) d g σ)) := by
  have key : ∀ s, Rewinding.IsPluralityE (opRL H E Lf sch (s % 2) (s / 2) d) g σ
      (rowT (lvOf sch (s / 2)).lanes (lvOf sch (s / 2)).logLen (plurCL H E Lf sch s d g σ)) := by
    intro s
    unfold plurCL
    rw [rowT_tabOf]
    exact Classical.choose_spec (Rewinding.exists_pluralityE (opRL H E Lf sch (s % 2) (s / 2) d) g σ)
  have h1 : slotOf r ℓ % 2 = r := by unfold slotOf; omega
  have h2 : slotOf r ℓ / 2 = ℓ := by unfold slotOf; omega
  have := key (slotOf r ℓ)
  rw [h1, h2] at this
  exact this

open Classical in
/-- **`T*(P*)`**: the plurality of the level-0 openings over the coins after the level-0 cap. -/
noncomputable def C0starL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (cap₀ : Cap D) (σ : Strategy (tableCmpL A H E Lf S sch hm ptLocal mPts cap₀)) : Oracle F :=
  tabOf (lvOf sch 0).lanes (lvOf sch 0).logLen
    (Classical.choose (Rewinding.exists_pluralityE (op₀L H E Lf sch cap₀) _ σ))

/-- **`Adv₀`**: the collision finder at the level-0 cap. Two continuations after `P*`'s level-0 cap open a level-0
position to different rows. -/
noncomputable def adv₀L (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (σ : Strategy (tableCL A H E Lf S sch hm ptLocal mPts)) : ℝ :=
  expect₂ (fun b b' => ind (fun x : OutCL F D Sl × OutCL F D Sl =>
      ∃ p, Rewinding.ConflictAt (op₀L H E Lf sch σ.1) x p) (b, b'))
    (tableCmpL A H E Lf S sch hm ptLocal mPts σ.1) σ.2

/-- **`Adv_{r,ℓ}`**: the collision finder at rep `r`'s level-`ℓ` cap. Along the run, at that cap, two continuations
open a level-`ℓ` position to different rows. -/
noncomputable def advRL [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (r ℓ : ℕ)
    (σ : Strategy (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) : ℝ :=
  Lock.forkE (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (slotOf r ℓ)
    (fun d b b' => ind (fun x : OutCL F D Sl × OutCL F D Sl =>
      ∃ p, Rewinding.ConflictAt (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) x p) (b, b'))
    (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
      (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2

/-! ## The theorem -/

/-- **The coupling and `table_sound`.** Whatever table `C₀` the oracle prover sends at the level-0 cap, a salted
compiled run accepts with `C₀` uncommitted with probability at most `tableError` plus the probability of a bad run. -/
theorem compiled_le_badL (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts) (C₀ : Oracle F) (cap₀ : Cap D)
    (σ₁ : Strategy (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀)) :
    expect (ind fun out : OutCL F D Sl => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀)
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀) σ₁ ≤
    tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
      Lock.leafE (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) (out : OutCL F D Sl) => ind (fun out : OutCL F D Sl =>
          out.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
            AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) out)
        (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁ := by
  have hcpl := Lock.expect_le_coupled (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (fun out : OutCL F D Sl => out.accepted = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀)
    (fun a : Bool × Oracle F => a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
    (fun (i : IRep F D × IRep F D) (out : OutCL F D Sl) => out.accepted = true ∧
      ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧ AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens))
    (fun i a b hR hE hB => by
      obtain ⟨ha2, -, hacc, -, -, -⟩ := hR
      refine ⟨?_, by rw [ha2]; exact hE.2⟩
      by_contra hn
      exact hB ⟨hE.1, fun h => hn (hacc hE.1 h.1 h.2)⟩)
    (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁
  have horc : expect (ind fun a : Bool × Oracle F =>
        a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
      (tableOrcL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts C₀)
      (Lock.oracleStrat (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁) ≤
        tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) := by
    have h1 := expect_cast (ind fun a : Bool × Oracle F =>
        a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
      (withDummy_tableL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)
      (⟨C₀, Lock.oracleStrat (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁⟩ :
        Strategy (Game.send (Oracle F)
          (tableOrcL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)))
    calc _ = expect (ind fun a : Bool × Oracle F =>
            a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
          (Game.send (Oracle F)
            (tableOrcL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))
          ⟨C₀, Lock.oracleStrat (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
            (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁⟩ := rfl
      _ = _ := h1.symm
      _ = _ := Lock.expect_withDummy _ _ _ _
      _ = _ := (prob_toReal _ _ _).symm
      _ ≤ _ := ENNReal.toReal_le_of_le_ofReal (tableError_nonneg _ _)
        (table_sound A hA S _ hsch hm ptLocal mPts hlay _)
  linarith

open Classical in
/-- A bad leaf, cap by cap (`bad_splitL` as indicators). The bad event requires acceptance, so the level-0 term counts
only accepted runs. -/
theorem bad_pointwise_accL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (C₀ : Oracle F) (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F)
    (b : OutCL F D Sl) (hR : TopRelL H E Lf sch C₀ cap₀ i a b) :
    ind (fun out : OutCL F D Sl => out.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
      AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) b ≤
    ind (SelfClashL H E Lf sch) b +
      ind (fun b : OutCL F D Sl => b.accepted = true ∧
        ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) b p) b +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ t) b p) b := by
  have hS := ind_nonneg (SelfClashL H E Lf sch) b
  have h0 := ind_nonneg (fun b : OutCL F D Sl => b.accepted = true ∧
    ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) b p) b
  have hsum : 0 ≤ ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
      ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
        ∃ p, Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ t) b p) b :=
    Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => ind_nonneg _ _
  by_cases hB : b.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 b.pend.1.reqs b.opens ∧
      AgreeRep 1 C₀ i.2 b.pend.2.reqs b.opens)
  · rw [ind, if_pos hB]
    rcases bad_splitL H E Lf sch C₀ cap₀ i a b hR hB.1 hB.2 with hc | ho | ⟨r, hr, ℓ, hℓ, t, d, hget, p, hp⟩
    · have : ind (SelfClashL H E Lf sch) b = 1 := if_pos hc
      linarith
    · have : ind (fun b : OutCL F D Sl => b.accepted = true ∧
          ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) b p) b = 1 := if_pos ⟨hB.1, ho⟩
      linarith
    · have h1 : ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ t) b p) b = 1 := if_pos ⟨t, d, hget, p, hp⟩
      have h2 := Finset.single_le_sum (f := fun ℓ' => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ') = some (t, d) ∧
            ∃ p, Rewinding.Off (opRL H E Lf sch r ℓ' d) (rowT _ _ t) b p) b)
        (fun _ _ => ind_nonneg _ _) hℓ
      have h3 := Finset.single_le_sum (f := fun r' => ∑ ℓ' ∈ Finset.Icc 1 sch.levels.length,
          ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r' ℓ') = some (t, d) ∧
            ∃ p, Rewinding.Off (opRL H E Lf sch r' ℓ' d) (rowT _ _ t) b p) b)
        (fun _ _ => Finset.sum_nonneg fun _ _ => ind_nonneg _ _) hr
      linarith
  · rw [ind, if_neg hB]
    linarith

open Classical in
/-- A bad leaf, cap by cap (`bad_splitL` as indicators). -/
theorem bad_pointwiseL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (C₀ : Oracle F) (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F)
    (b : OutCL F D Sl) (hR : TopRelL H E Lf sch C₀ cap₀ i a b) :
    ind (fun out : OutCL F D Sl => out.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
      AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) b ≤
    ind (SelfClashL H E Lf sch) b +
      ind (fun b => ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) b p) b +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ t) b p) b := by
  have h := bad_pointwise_accL H E Lf sch C₀ cap₀ i a b hR
  have h0 := ind_mono (fun (b : OutCL F D Sl)
    (h : b.accepted = true ∧ ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) b p) => h.2) b
  linarith

open Classical in
/-- **The bad runs, cap by cap**: rewinding at the level-0 cap (`expect_bad_le`) and at every rep's cap
(`Lock.leafE_off_le`). -/
theorem bad_leL [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (σ : Strategy (Game.send (Cap D)
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))) :
    Lock.leafE (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) (out : OutCL F D Sl) => ind (fun out : OutCL F D Sl =>
          out.accepted = true ∧
            ¬ (AgreeRep 0 (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) i.1
                out.pend.1.reqs out.opens ∧
              AgreeRep 1 (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) i.2
                out.pend.2.reqs out.opens)) out)
        (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
          (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2 ≤
      expect (ind (SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
          (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ +
        Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) *
          adv₀L A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ) +
        ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
          Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
            (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
            advRL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ) := by
  have hsplit := Lock.leafE_mono_rel (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (fun i a b hR => bad_pointwiseL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ _ σ.1 i a b hR)
    (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
      (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2
  rw [Lock.leafE_add, Lock.leafE_add, Lock.leafE_const, Lock.leafE_const] at hsplit
  simp only [Lock.leafE_sum] at hsplit
  have eS : expect (ind (SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ =
    expect (ind (SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := rfl
  have h0 : expect (ind fun b => ∃ p, Rewinding.Off (op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (rowT _ _ (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2)) b p)
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 ≤
      Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) *
        adv₀L A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ) := by
    have hpl : Rewinding.IsPluralityE (op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2
        (rowT _ _ (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2)) := by
      unfold C0starL
      rw [rowT_tabOf]
      exact Classical.choose_spec _
    have hQ : ∀ b, (Finset.univ.filter fun p =>
        (op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 b p).isSome).card ≤ 2 * l₀.queries := by
      intro b
      refine (opOfL_count _ _ _ _ _ _ _ _).trans ?_
      unfold opens₀L
      rw [List.length_append]
      have h1 := slotOpensL_length b 0 0 (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0).queries
      have h2 := slotOpensL_length b 1 0 (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0).queries
      simp only [lvOf, List.getD_cons_zero] at h1 h2 ⊢
      omega
    have := Rewinding.expect_bad_le (op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1) _ σ.2 _ hpl _ hQ
    rw [Fintype.card_fin] at this
    unfold adv₀L
    simpa [lvOf, Nat.cast_pow, Nat.cast_mul] using this
  have hR : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Lock.leafE (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) b => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) (rowT _ _ t) b p) b)
        (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
          (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2 ≤
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ) := by
    intro r hr ℓ _
    have := Lock.leafE_off_le (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ)
      (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
      (slotOf r ℓ) (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries
      (fun d b => (opOfL_count _ _ _ _ _ _ _ _).trans (slotOpensL_length _ _ _ _))
      (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
      (fun d g σ' => plurCL_plurality H E Lf _ r ℓ (Finset.mem_range.1 hr) d g σ')
      (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
        (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2
      (getPair_init_none _)
    rw [Fintype.card_fin] at this
    unfold advRL
    simpa [Nat.cast_pow] using this
  have hsum := Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => hR r hr ℓ hℓ
  rw [eS]
  linarith

open Classical in
/-- **Compiled soundness of a salted table under collision resistance** (README.md §1.2). For every prover `P*`
(strategy `σ`) of the salted compiled table `tableCL A H E Lf`, with `T*(P*)` the plurality table of its level-0
openings: `Pr[accepted ∧ no codeword near T*(P*) packs a satisfying witness]` is at most `tableError`, plus
`√(N·Q·Adv)` for the level-0 cap and for every rep's cap, plus the probability of a self-clash. Each `Adv` is the
success probability of an explicit collision finder, and every pair it counts, like every self-clash, is two verifying
openings of one position to different rows under one cap, under any salts: a collision of `H`
(`Merkle.opening_bindingL`). At `Leaf.plain` it is `table_sound_compiled`'s bound; at `Refine.leaf512`,
`hm96-sha512/v1`'s. -/
theorem table_sound_compiledL (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (σ : Strategy (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) :
    expect (ind fun out : OutCL F D Sl => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2))
      (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ ≤
    tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
      Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) *
        adv₀L A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ) +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
          (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
          advRL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts r ℓ σ) +
      expect (ind (SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
        (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ := by
  revert σ
  show ∀ σ : Strategy (Game.send (Cap D)
    (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)), _
  intro σ
  have h1 := compiled_le_badL A hA H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hsch
    hm ptLocal mPts hlay (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1 σ.2
  have h2 := bad_leL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ
  have e : expect (ind fun out : OutCL F D Sl => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2))
      (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ =
    expect (ind fun out : OutCL F D Sl => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2))
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := rfl
  rw [e]
  linarith

end FlockSoundness.Discharge.Exec
