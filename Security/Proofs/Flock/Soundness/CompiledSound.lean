import Proofs.Flock.Soundness.CompiledLock
import Proofs.Flock.Soundness.Soundness

/-!
# Compiled soundness under collision resistance (README.md §1.2)

The oracle prover `P̃` sends, at each commitment, the plurality table of the openings the
compiled prover `P*` makes there (`Lock.oracleStrat`). A compiled run that accepts while some
opening disagrees with the fixed tables either opens a position two ways within the run
(`SelfClash`) or opens a position off the table fixed at some cap (`bad_split`). By rewinding at
every cap (`Lock.leafE_off_le`), the second costs `√(N · Q · Adv)` per cap, where `Adv` is the
probability that two continuations from the cap open a position differently; each such pair,
like each self-clash, is a SHA-512 collision (`Merkle.opening_binding`).
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-- A level's parameters. -/
abbrev lvOf (sch : Schedule) (ℓ : ℕ) : Level := sch.levels.getD ℓ default

/-- The openings of rep `r`'s first level-`ℓ` request, `(position, row, path)` for its first `Q`
queries. -/
def slotOpens (out : OutC F D) (r ℓ Q : ℕ) : List (ℕ × List F × List D) :=
  match ((if r = 0 then out.pend.1 else out.pend.2).reqs.find? fun req => req.level == ℓ) with
  | none => []
  | some req => ((req.positions.take Q).zipIdx).map fun pj =>
      (pj.1, (out.opens r ℓ pj.2).1, (out.opens r ℓ pj.2).2)

theorem slotOpens_length (out : OutC F D) (r ℓ Q : ℕ) : (slotOpens out r ℓ Q).length ≤ Q := by
  unfold slotOpens
  split
  · simp
  · simp only [List.length_map, List.length_zipIdx, List.length_take]
    exact min_le_left _ _

/-- An opening verifies against `cap`: a row of `2^lanes` elements whose path verifies. -/
def Verified (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (cap : Cap D) (lanes depth : ℕ)
    (x : ℕ × List F × List D) : Prop :=
  x.2.1.length = 2 ^ lanes ∧ Merkle.Verifies H E cap depth x.1 x.2.1 x.2.2

/-- A row as a function of its lanes. -/
def rowFin (lanes : ℕ) (row : List F) : Fin (2 ^ lanes) → F := fun lane => row.getD lane.val 0

open Classical in
/-- The row a run opens at `p`: its first verifying opening there. -/
noncomputable def opOf (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (lanes depth d : ℕ)
    (xs : List (ℕ × List F × List D)) (cap : Cap D) (p : Fin (2 ^ d)) :
    Option (Fin (2 ^ lanes) → F) :=
  ((xs.filter fun x => decide (x.1 = p.val ∧ Verified H E cap lanes depth x)).head?).map
    fun x => rowFin lanes x.2.1

/-- Two verifying openings of one position to different rows. -/
def Clash (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (cap : Cap D) (lanes depth : ℕ)
    (xs : List (ℕ × List F × List D)) : Prop :=
  ∃ x ∈ xs, ∃ y ∈ xs, x.1 = y.1 ∧ Verified H E cap lanes depth x ∧
    Verified H E cap lanes depth y ∧ x.2.1 ≠ y.2.1

open Classical in
/-- A run opens at most as many positions as it has openings. -/
theorem opOf_count (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (lanes depth d : ℕ)
    (xs : List (ℕ × List F × List D)) (cap : Cap D) :
    (Finset.univ.filter fun p => (opOf H E lanes depth d xs cap p).isSome).card ≤ xs.length := by
  calc (Finset.univ.filter fun p => (opOf H E lanes depth d xs cap p).isSome).card
      ≤ (xs.map fun x => x.1).toFinset.card := by
        refine Finset.card_le_card_of_injOn (fun p => p.val) (fun p hp => ?_)
          (fun p _ q _ h => Fin.ext h)
        simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq] at hp
        unfold opOf at hp
        rw [Option.isSome_map] at hp
        obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hp
        have hmem := List.mem_of_mem_head? hx
        rw [List.mem_filter, decide_eq_true_eq] at hmem
        simp only [List.coe_toFinset, List.mem_map, Set.mem_setOf_eq]
        exact ⟨x, hmem.1, hmem.2.1⟩
    _ ≤ (xs.map fun x => x.1).length := List.toFinset_card_le _
    _ = xs.length := List.length_map _

/-- **Off or clash.** A verifying opening of `p` whose row is not `v`: the run opens `p` to a row
other than `v`, or it opens `p` two ways. -/
theorem off_or_clash (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (lanes depth d : ℕ)
    (xs : List (ℕ × List F × List D)) (cap : Cap D) (x : ℕ × List F × List D) (hx : x ∈ xs)
    (p : Fin (2 ^ d)) (hxp : x.1 = p.val) (hv : Verified H E cap lanes depth x)
    (v : Fin (2 ^ lanes) → F) (hne : rowFin lanes x.2.1 ≠ v) :
    ((opOf H E lanes depth d xs cap p).isSome ∧ opOf H E lanes depth d xs cap p ≠ some v) ∨
      Clash H E cap lanes depth xs := by
  classical
  unfold opOf
  set ys := xs.filter fun x => decide (x.1 = p.val ∧ Verified H E cap lanes depth x)
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

/-- The unique request at its level is the one found. -/
theorem find_level {reqs : List Req} (h : reqs.map Req.level = List.range reqs.length) {req : Req}
    (hr : req ∈ reqs) : reqs.find? (fun q => q.level == req.level) = some req := by
  have hnd : (reqs.map Req.level).Nodup := by rw [h]; exact List.nodup_range
  obtain ⟨q, hq⟩ : ∃ q, reqs.find? (fun q => q.level == req.level) = some q :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨req, hr, by simp⟩)
  rw [hq]
  have hqm := List.mem_of_find?_eq_some hq
  have hql : q.level = req.level := by simpa using List.find?_some hq
  exact congrArg some (List.inj_on_of_nodup_map hnd hqm hr hql)

/-! ## Per-cap openings -/

/-- Both reps' level-0 openings. -/
def opens₀ (sch : Schedule) (out : OutC F D) : List (ℕ × List F × List D) :=
  slotOpens out 0 0 (lvOf sch 0).queries ++ slotOpens out 1 0 (lvOf sch 0).queries

/-- The level-0 cap's openings, by position. -/
noncomputable def op₀ (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule)
    (cap : Cap D) (out : OutC F D) :
    Fin (2 ^ (lvOf sch 0).logLen) → Option (Fin (2 ^ (lvOf sch 0).lanes) → F) :=
  opOf H E (lvOf sch 0).lanes (pathLen (lvOf sch 0)) (lvOf sch 0).logLen (opens₀ sch out) cap

/-- Rep `r`'s level-`ℓ` cap's openings, by position. -/
noncomputable def opR (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule) (r ℓ : ℕ)
    (cap : Cap D) (out : OutC F D) :
    Fin (2 ^ (lvOf sch ℓ).logLen) → Option (Fin (2 ^ (lvOf sch ℓ).lanes) → F) :=
  opOf H E (lvOf sch ℓ).lanes (pathLen (lvOf sch ℓ)) (lvOf sch ℓ).logLen
    (slotOpens out r ℓ (lvOf sch ℓ).queries) cap

/-- A table's rows on a level's domain. -/
def rowT (lanes d : ℕ) (t : Oracle F) : Fin (2 ^ d) → Fin (2 ^ lanes) → F :=
  fun p lane => t lane.val p.val

/-- The table with the given rows on a level's domain. -/
def tabOf (lanes d : ℕ) (v : Fin (2 ^ d) → Fin (2 ^ lanes) → F) : Oracle F :=
  fun lane q => if h : q < 2 ^ d ∧ lane < 2 ^ lanes then v ⟨q, h.1⟩ ⟨lane, h.2⟩ else 0

theorem rowT_tabOf (lanes d : ℕ) (v : Fin (2 ^ d) → Fin (2 ^ lanes) → F) :
    rowT lanes d (tabOf lanes d v) = v := by
  funext p lane
  simp [rowT, tabOf, p.isLt, lane.isLt]

/-- **A self-clash**: within one run, two verifying openings of a position to different rows, at
the level-0 cap or at a rep's recorded cap. -/
def SelfClash (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule)
    (out : OutC F D) : Prop :=
  Clash H E out.cap₀ (lvOf sch 0).lanes (pathLen (lvOf sch 0)) (opens₀ sch out) ∨
    ∃ r < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap,
      (if r = 0 then out.pend.1 else out.pend.2).caps[ℓ - 1]? = some cap ∧
        Clash H E cap (lvOf sch ℓ).lanes (pathLen (lvOf sch ℓ))
          (slotOpens out r ℓ (lvOf sch ℓ).queries)

/-! ## Where a bad run goes wrong -/

theorem mem_slotOpens {out : OutC F D} {r : ℕ} {req : Req}
    (hfind : (if r = 0 then out.pend.1 else out.pend.2).reqs.find? (fun q => q.level == req.level) =
      some req)
    {Q : ℕ} (hQ : req.positions.length = Q) {j : ℕ} (hj : j < req.positions.length) :
    (req.positions.getD j 0, (out.opens r req.level j).1, (out.opens r req.level j).2) ∈
      slotOpens out r req.level Q := by
  unfold slotOpens
  rw [hfind]
  simp only [List.mem_map]
  refine ⟨(req.positions.getD j 0, j), ?_, rfl⟩
  rw [List.take_of_length_le hQ.le, List.mem_iff_getElem]
  refine ⟨j, by simpa using hj, ?_⟩
  rw [List.getElem_zipIdx]
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]

/-- **A rep's disagreeing opening**, on an accepted run: at level 0, off `C₀` or a clash at the
level-0 cap; at a level `ℓ ≥ 1`, off the table fixed there or a clash at its recorded cap. -/
theorem rep_bad (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule) (C₀ : Oracle F)
    (cap₀ : Cap D) (out : OutC F D) (r : ℕ) (hr : r < 2) (ir : IRep F D)
    (hs : RecStruct sch ir (if r = 0 then out.pend.1 else out.pend.2))
    (hok : OpensOK H E cap₀ r (if r = 0 then out.pend.1 else out.pend.2) out.opens)
    (hnot : ¬ AgreeRep r C₀ ir (if r = 0 then out.pend.1 else out.pend.2).reqs out.opens) :
    Clash H E cap₀ (lvOf sch 0).lanes (pathLen (lvOf sch 0)) (opens₀ sch out) ∨
      (∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) out p) ∨
      ∃ ℓ ∈ Finset.Icc 1 sch.levels.length, ∃ t d, ir ℓ = some (t, d) ∧
        (if r = 0 then out.pend.1 else out.pend.2).caps[ℓ - 1]? = some d ∧
        ((∃ p, Rewinding.Off (opR H E sch r ℓ d) (rowT _ _ t) out p) ∨
          Clash H E d (lvOf sch ℓ).lanes (pathLen (lvOf sch ℓ))
            (slotOpens out r ℓ (lvOf sch ℓ).queries)) := by
  classical
  unfold AgreeRep at hnot
  push_neg at hnot
  obtain ⟨req, hreq, t, ht, j, hj, lane, hlane, hne⟩ := hnot
  obtain ⟨hlev, hshape, hcaps, hlen⟩ := hs
  obtain ⟨hQ, hlt, hlanes, hdepth⟩ := hshape req hreq
  obtain ⟨cap, hcapOf, hver⟩ := hok req hreq
  obtain ⟨hrowlen, hpath⟩ := hver j hj
  have hfind := find_level hlev hreq
  have hxmem := mem_slotOpens hfind hQ hj
  have hposmem : req.positions.getD j 0 ∈ req.positions := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]
    exact List.getElem_mem hj
  have hℓlt : req.level < (if r = 0 then out.pend.1 else out.pend.2).reqs.length := by
    have := List.mem_map_of_mem (f := Req.level) hreq
    rw [hlev, List.mem_range] at this
    exact this
  have hv : Verified H E cap (lvOf sch req.level).lanes (pathLen (lvOf sch req.level))
      (req.positions.getD j 0, (out.opens r req.level j).1, (out.opens r req.level j).2) :=
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
    have hx₀ : (req.positions.getD j 0, (out.opens r 0 j).1, (out.opens r 0 j).2) ∈
        opens₀ sch out := by
      unfold opens₀
      rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
      · exact List.mem_append_left _ hxmem
      · exact List.mem_append_right _ hxmem
    rcases off_or_clash H E _ _ _ (opens₀ sch out) cap _ hx₀ ⟨req.positions.getD j 0, hposlt⟩ rfl
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
    rcases off_or_clash H E _ _ _ _ d' _ hxmem ⟨req.positions.getD j 0, hposlt⟩ rfl hv _ hne' with
      hoff | hcl
    · exact Or.inl ⟨_, hoff⟩
    · exact Or.inr hcl

/-- **Where a bad run goes wrong.** An accepted run that disagrees with the fixed tables either
clashes within the run, or opens a position off `C₀`, or opens a position off the table fixed at
one of the reps' caps. -/
theorem bad_split (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule)
    (C₀ : Oracle F) (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F)
    (out : OutC F D) (hR : TopRel H E sch C₀ cap₀ i a out) (hacc : out.accepted = true)
    (hbad : ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
      AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) :
    SelfClash H E sch out ∨ (∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) out p) ∨
      ∃ r ∈ Finset.range 2, ∃ ℓ ∈ Finset.Icc 1 sch.levels.length, ∃ t d,
        Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opR H E sch r ℓ d) (rowT _ _ t) out p := by
  obtain ⟨-, hcap, -, hok, hs₀, hs₁⟩ := hR
  obtain ⟨hok₀, hok₁⟩ := hok hacc
  have key : ∀ r (hr : r < 2) (ir : IRep F D),
      (∀ ℓ, 1 ≤ ℓ → Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = ir ℓ) →
      RecStruct sch ir (if r = 0 then out.pend.1 else out.pend.2) →
      OpensOK H E cap₀ r (if r = 0 then out.pend.1 else out.pend.2) out.opens →
      ¬ AgreeRep r C₀ ir (if r = 0 then out.pend.1 else out.pend.2).reqs out.opens →
      SelfClash H E sch out ∨ (∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) out p) ∨
        ∃ r ∈ Finset.range 2, ∃ ℓ ∈ Finset.Icc 1 sch.levels.length, ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (opR H E sch r ℓ d) (rowT _ _ t) out p := by
    intro r hr ir hget hs hok hnot
    rcases rep_bad H E sch C₀ cap₀ out r hr ir hs hok hnot with hcl | hoff | ⟨ℓ, hℓ, t, d, hi, hc, h⟩
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

open Classical in
theorem opOf_some {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {lanes depth d : ℕ}
    {xs : List (ℕ × List F × List D)} {cap : Cap D} {p : Fin (2 ^ d)}
    (h : (opOf H E lanes depth d xs cap p).isSome) :
    ∃ x ∈ xs, x.1 = p.val ∧ Verified H E cap lanes depth x ∧
      opOf H E lanes depth d xs cap p = some (rowFin lanes x.2.1) := by
  unfold opOf at h ⊢
  rw [Option.isSome_map] at h
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 h
  have hmem := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq] at hmem
  exact ⟨x, hmem.1, hmem.2.1, hmem.2.2, by rw [hx]; rfl⟩

/-- **A clash is a collision.** -/
theorem Clash.collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {cap : Cap D}
    {lanes depth : ℕ} {xs : List (ℕ × List F × List D)} (h : Clash H E cap lanes depth xs) :
    Merkle.Collision H := by
  obtain ⟨x, -, y, -, hxy, hx, hy, hne⟩ := h
  exact Merkle.opening_binding H E cap depth x.1 x.2.1 y.2.1 x.2.2 y.2.2 hne hx.2 (hxy ▸ hy.2)

/-- **A self-clash is a collision.** -/
theorem SelfClash.collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {sch : Schedule}
    {out : OutC F D} (h : SelfClash H E sch out) : Merkle.Collision H := by
  rcases h with h | ⟨_, _, _, _, _, _, h⟩
  · exact h.collision
  · exact h.collision

/-- Two runs that open a position to different rows under one cap give a collision. -/
theorem opOf_conflict_collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D}
    {lanes depth d : ℕ} {cap : Cap D} {xs xs' : List (ℕ × List F × List D)} {p : Fin (2 ^ d)}
    (h1 : (opOf H E lanes depth d xs cap p).isSome) (h2 : (opOf H E lanes depth d xs' cap p).isSome)
    (hne : opOf H E lanes depth d xs cap p ≠ opOf H E lanes depth d xs' cap p) :
    Merkle.Collision H := by
  obtain ⟨x, -, hxp, hx, hox⟩ := opOf_some h1
  obtain ⟨y, -, hyp, hy, hoy⟩ := opOf_some h2
  refine Merkle.opening_binding H E cap depth p.val x.2.1 y.2.1 x.2.2 y.2.2 (fun hxy => hne ?_)
    (hxp ▸ hx.2) (hyp ▸ hy.2)
  rw [hox, hoy, hxy]

/-- **Every pair the level-0 collision finder counts is a collision.** -/
theorem op₀_conflict_collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {sch : Schedule}
    {cap : Cap D} {x : OutC F D × OutC F D} {p : Fin (2 ^ (lvOf sch 0).logLen)}
    (h : Rewinding.ConflictAt (op₀ H E sch cap) x p) : Merkle.Collision H :=
  opOf_conflict_collision h.1 h.2.1 h.2.2

/-- **Every pair a rep's collision finder counts is a collision.** -/
theorem opR_conflict_collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {sch : Schedule}
    {r ℓ : ℕ} {cap : Cap D} {x : OutC F D × OutC F D} {p : Fin (2 ^ (lvOf sch ℓ).logLen)}
    (h : Rewinding.ConflictAt (opR H E sch r ℓ cap) x p) : Merkle.Collision H :=
  opOf_conflict_collision h.1 h.2.1 h.2.2

/-! ## The plurality prover and the collision finders -/

theorem tableError_nonneg (s : Schedule) (st : Stmt) : 0 ≤ tableError s st := by
  unfold tableError linkError; positivity

theorem expect_cast {α : Type} (f : α → ℝ) : ∀ {g₁ g₂ : Game α} (h : g₁ = g₂) (s : Strategy g₂),
    expect f g₁ (cast (congrArg Strategy h.symm) s) = expect f g₂ s
  | _, _, rfl, _ => rfl

open Classical in
/-- The table `P̃` sends at a rep's cap: the plurality of the openings there. -/
noncomputable def plurC (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule) :
    Lock.Chooser (Oracle F) (Cap D) (OutC F D) :=
  fun s d g σ => tabOf (lvOf sch (s / 2)).lanes (lvOf sch (s / 2)).logLen
    (Classical.choose (Rewinding.exists_pluralityE (opR H E sch (s % 2) (s / 2) d) g σ))

theorem plurC_plurality (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule)
    (r ℓ : ℕ) (hr : r < 2) (d : Cap D) (g : Game (OutC F D)) (σ : Strategy g) :
    Rewinding.IsPluralityE (opR H E sch r ℓ d) g σ
      (rowT (lvOf sch ℓ).lanes (lvOf sch ℓ).logLen (plurC H E sch (slotOf r ℓ) d g σ)) := by
  have key : ∀ s, Rewinding.IsPluralityE (opR H E sch (s % 2) (s / 2) d) g σ
      (rowT (lvOf sch (s / 2)).lanes (lvOf sch (s / 2)).logLen (plurC H E sch s d g σ)) := by
    intro s
    unfold plurC
    rw [rowT_tabOf]
    exact Classical.choose_spec (Rewinding.exists_pluralityE (opR H E sch (s % 2) (s / 2) d) g σ)
  have h1 : slotOf r ℓ % 2 = r := by unfold slotOf; omega
  have h2 : slotOf r ℓ / 2 = ℓ := by unfold slotOf; omega
  have := key (slotOf r ℓ)
  rw [h1, h2] at this
  exact this

open Classical in
/-- **`T*(P*)`**: the plurality of the level-0 openings over the coins after the level-0 cap. -/
noncomputable def C0star (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (cap₀ : Cap D)
    (σ : Strategy (tableCmp A H E S sch hm ptLocal mPts cap₀)) : Oracle F :=
  tabOf (lvOf sch 0).lanes (lvOf sch 0).logLen
    (Classical.choose (Rewinding.exists_pluralityE (op₀ H E sch cap₀) _ σ))

/-- **`Adv₀`**: the collision finder at the level-0 cap. Two continuations after `P*`'s level-0
cap open a level-0 position to different rows. -/
noncomputable def adv₀ (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (σ : Strategy (tableC A H E S sch hm ptLocal mPts)) : ℝ :=
  expect₂ (fun b b' => ind (fun x : OutC F D × OutC F D =>
      ∃ p, Rewinding.ConflictAt (op₀ H E sch σ.1) x p) (b, b'))
    (tableCmp A H E S sch hm ptLocal mPts σ.1) σ.2

/-- **`Adv_{r,ℓ}`**: the collision finder at rep `r`'s level-`ℓ` cap. Along the run, at that
cap, two continuations open a level-`ℓ` position to different rows. -/
noncomputable def advR [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (r ℓ : ℕ)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) : ℝ :=
  Lock.forkE (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (slotOf r ℓ)
    (fun d b b' => ind (fun x : OutC F D × OutC F D =>
      ∃ p, Rewinding.ConflictAt (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) x p) (b, b'))
    (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
      (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2

/-! ## The theorem -/

theorem getPair_init_none (s : ℕ) :
    Lock.getPair (getRep 0) (getRep 1) ((fun _ => none, fun _ => none) : IRep F D × IRep F D) s =
      none := by
  simp [Lock.getPair, getRep]

/-- **The coupling and `table_sound`.** Whatever table `C₀` the oracle prover sends at the level-0
cap, a compiled run accepts with `C₀` uncommitted with probability at most `tableError` plus the
probability of a bad run. -/
theorem compiled_le_bad (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts) (C₀ : Oracle F) (cap₀ : Cap D)
    (σ₁ : Strategy (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀)) :
    expect (ind fun out : OutC F D => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀) σ₁ ≤
    tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
      Lock.leafE (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) (out : OutC F D) => ind (fun out : OutC F D =>
          out.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
            AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) out)
        (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁ := by
  have hcpl := Lock.expect_le_coupled (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (fun out : OutC F D => out.accepted = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀)
    (fun a : Bool × Oracle F => a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
    (fun (i : IRep F D × IRep F D) (out : OutC F D) => out.accepted = true ∧
      ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧ AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens))
    (fun i a b hR hE hB => by
      obtain ⟨ha2, -, hacc, -, -, -⟩ := hR
      refine ⟨?_, by rw [ha2]; exact hE.2⟩
      by_contra hn
      exact hB ⟨hE.1, fun h => hn (hacc hE.1 h.1 h.2)⟩)
    (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁
  have horc : expect (ind fun a : Bool × Oracle F =>
        a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
      (tableOrc (D := D) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts C₀)
      (Lock.oracleStrat (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁) ≤
        tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) := by
    have h1 := expect_cast (ind fun a : Bool × Oracle F =>
        a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
      (withDummy_table (D := D) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)
      (⟨C₀, Lock.oracleStrat (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁⟩ :
        Strategy (Game.send (Oracle F)
          (tableOrc (D := D) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)))
    calc _ = expect (ind fun a : Bool × Oracle F =>
            a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
          (Game.send (Oracle F) (tableOrc (D := D) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))
          ⟨C₀, Lock.oracleStrat (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
            (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁⟩ := rfl
      _ = _ := h1.symm
      _ = _ := Lock.expect_withDummy _ _ _ _
      _ = _ := (prob_toReal _ _ _).symm
      _ ≤ _ := ENNReal.toReal_le_of_le_ofReal (tableError_nonneg _ _)
        (table_sound A hA S _ hsch hm ptLocal mPts hlay _)
  linarith

open Classical in
/-- A bad leaf, cap by cap (`bad_split` as indicators). The bad event requires acceptance, so the
level-0 term counts only accepted runs. -/
theorem bad_pointwise_acc (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule)
    (C₀ : Oracle F) (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F) (b : OutC F D)
    (hR : TopRel H E sch C₀ cap₀ i a b) :
    ind (fun out : OutC F D => out.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
      AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) b ≤
    ind (SelfClash H E sch) b +
      ind (fun b : OutC F D => b.accepted = true ∧
        ∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) b p) b +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opR H E sch r ℓ d) (rowT _ _ t) b p) b := by
  have hS := ind_nonneg (SelfClash H E sch) b
  have h0 := ind_nonneg (fun b : OutC F D => b.accepted = true ∧
    ∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) b p) b
  have hsum : 0 ≤ ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
      ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
        ∃ p, Rewinding.Off (opR H E sch r ℓ d) (rowT _ _ t) b p) b :=
    Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => ind_nonneg _ _
  by_cases hB : b.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 b.pend.1.reqs b.opens ∧
      AgreeRep 1 C₀ i.2 b.pend.2.reqs b.opens)
  · rw [ind, if_pos hB]
    rcases bad_split H E sch C₀ cap₀ i a b hR hB.1 hB.2 with hc | ho | ⟨r, hr, ℓ, hℓ, t, d, hget, p, hp⟩
    · have : ind (SelfClash H E sch) b = 1 := if_pos hc
      linarith
    · have : ind (fun b : OutC F D => b.accepted = true ∧
          ∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) b p) b = 1 := if_pos ⟨hB.1, ho⟩
      linarith
    · have h1 : ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opR H E sch r ℓ d) (rowT _ _ t) b p) b = 1 := if_pos ⟨t, d, hget, p, hp⟩
      have h2 := Finset.single_le_sum (f := fun ℓ' => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ') = some (t, d) ∧
            ∃ p, Rewinding.Off (opR H E sch r ℓ' d) (rowT _ _ t) b p) b)
        (fun _ _ => ind_nonneg _ _) hℓ
      have h3 := Finset.single_le_sum (f := fun r' => ∑ ℓ' ∈ Finset.Icc 1 sch.levels.length,
          ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r' ℓ') = some (t, d) ∧
            ∃ p, Rewinding.Off (opR H E sch r' ℓ' d) (rowT _ _ t) b p) b)
        (fun _ _ => Finset.sum_nonneg fun _ _ => ind_nonneg _ _) hr
      linarith
  · rw [ind, if_neg hB]
    linarith

open Classical in
/-- A bad leaf, cap by cap (`bad_split` as indicators). -/
theorem bad_pointwise (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule)
    (C₀ : Oracle F) (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F) (b : OutC F D)
    (hR : TopRel H E sch C₀ cap₀ i a b) :
    ind (fun out : OutC F D => out.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
      AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) b ≤
    ind (SelfClash H E sch) b +
      ind (fun b => ∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) b p) b +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opR H E sch r ℓ d) (rowT _ _ t) b p) b := by
  have h := bad_pointwise_acc H E sch C₀ cap₀ i a b hR
  have h0 := ind_mono (fun (b : OutC F D)
    (h : b.accepted = true ∧ ∃ p, Rewinding.Off (op₀ H E sch cap₀) (rowT _ _ C₀) b p) => h.2) b
  linarith

open Classical in
/-- **The bad runs, cap by cap**: rewinding at the level-0 cap (`expect_bad_le`) and at every
rep's cap (`Lock.leafE_off_le`). -/
theorem bad_le [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (σ : Strategy (Game.send (Cap D)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))) :
    Lock.leafE (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) (out : OutC F D) => ind (fun out : OutC F D =>
          out.accepted = true ∧
            ¬ (AgreeRep 0 (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) i.1
                out.pend.1.reqs out.opens ∧
              AgreeRep 1 (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) i.2
                out.pend.2.reqs out.opens)) out)
        (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
          (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2 ≤
      expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
          (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ +
        Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) *
          adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ) +
        ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
          Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
            (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
            advR A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ) := by
  have hsplit := Lock.leafE_mono_rel (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (fun i a b hR => bad_pointwise H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ _ σ.1 i a b hR)
    (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
      (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2
  rw [Lock.leafE_add, Lock.leafE_add, Lock.leafE_const, Lock.leafE_const] at hsplit
  simp only [Lock.leafE_sum] at hsplit
  have eS : expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ =
    expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := rfl
  have h0 : expect (ind fun b => ∃ p, Rewinding.Off (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (rowT _ _ (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2)) b p)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 ≤
      Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) *
        adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ) := by
    have hpl : Rewinding.IsPluralityE (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2
        (rowT _ _ (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2)) := by
      unfold C0star
      rw [rowT_tabOf]
      exact Classical.choose_spec _
    have hQ : ∀ b, (Finset.univ.filter fun p =>
        (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 b p).isSome).card ≤ 2 * l₀.queries := by
      intro b
      refine (opOf_count _ _ _ _ _ _ _).trans ?_
      unfold opens₀
      rw [List.length_append]
      have h1 := slotOpens_length b 0 0 (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0).queries
      have h2 := slotOpens_length b 1 0 (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0).queries
      simp only [lvOf, List.getD_cons_zero] at h1 h2 ⊢
      omega
    have := Rewinding.expect_bad_le (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1) _ σ.2 _ hpl _ hQ
    rw [Fintype.card_fin] at this
    unfold adv₀
    simpa [lvOf, Nat.cast_pow, Nat.cast_mul] using this
  have hR : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Lock.leafE (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) b => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) (rowT _ _ t) b p) b)
        (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
          (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2 ≤
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advR A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ) := by
    intro r hr ℓ _
    have := Lock.leafE_off_le (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ)
      (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
      (slotOf r ℓ) (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries
      (fun d b => (opOf_count _ _ _ _ _ _ _).trans (slotOpens_length _ _ _ _))
      (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
      (fun d g σ' => plurC_plurality H E _ r ℓ (Finset.mem_range.1 hr) d g σ')
      (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
        (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2
      (getPair_init_none _)
    rw [Fintype.card_fin] at this
    unfold advR
    simpa [Nat.cast_pow] using this
  have hsum := Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => hR r hr ℓ hℓ
  rw [eS]
  linarith

open Classical in
/-- **Compiled soundness under collision resistance** (README.md §1.2). For every compiled
prover `P*` (strategy `σ`), with `T*(P*)` the plurality table of its level-0 openings:
`Pr[accepted ∧ no codeword near T*(P*) packs a satisfying witness]` is at most `tableError`,
plus `√(N·Q·Adv)` for the level-0 cap and for every rep's cap, plus the probability of a
self-clash. Each `Adv` is the success probability of an explicit collision finder, and every
pair it counts, like every self-clash, is two verifying openings of one position to different
rows under one cap: a SHA-512 collision (`Merkle.opening_binding`). -/
theorem table_sound_compiled (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) :
    expect (ind fun out : OutC F D => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2))
      (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ ≤
    tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
      Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) *
        adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ) +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
          (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
          advR A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts r ℓ σ) +
      expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
        (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ := by
  revert σ
  show ∀ σ : Strategy (Game.send (Cap D)
    (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)), _
  intro σ
  have h1 := compiled_le_bad A hA H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hsch
    hm ptLocal mPts hlay (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1 σ.2
  have h2 := bad_le A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ
  have e : expect (ind fun out : OutC F D => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2))
      (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ =
    expect (ind fun out : OutC F D => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2))
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := rfl
  rw [e]
  linarith

end FlockSoundness
