import FlockSoundness.Model.Compiled
import FlockSoundness.Game.Lock
import FlockSoundness.LigeritoMid
import FlockSoundness.Protocol.CompiledLock

/-!
# The compiled Flock verifier runs in lockstep with the oracle verifier

`Game.Lock` between `Model.table` (with a dummy last message) and `Model.tableC`: the same coins
and messages, caps for tables. Along the lockstep, the compiled claim evaluated at any openings
that agree with the fixed tables at every requested position equals the oracle claim
(`StRel`), so at the end the compiled verdict equals the oracle verdict on such openings.

Slots: `0` is the level-0 cap; rep `r`'s level-`ℓ` cap (`ℓ ≥ 1`) is `slotOf r ℓ = 2ℓ + r`.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

theorem AgreeRep.of_append {r : ℕ} {C₀ : Oracle F} {i : IRep F D} {reqs reqs' : List Req}
    {o : Opens F D} (h : AgreeRep r C₀ i (reqs ++ reqs') o) : AgreeRep r C₀ i reqs o :=
  fun req hr => h req (List.mem_append_left _ hr)

/-- On agreeing openings, the consistency value from the opened rows is the oracle's. -/
theorem consValueO_eq (A : Arith F K) (α : List F) (ch : List K) (C : Oracle F) (rows : ℕ → List F)
    (pos : List ℕ)
    (h : ∀ j < pos.length, ∀ lane < 2 ^ ch.length, (rows j).getD lane 0 = C lane (pos.getD j 0)) :
    consValueO A α ch rows pos = consValue A α ch C pos := by
  unfold consValueO consValue
  refine congrArg List.sum (List.map_congr_left fun pj hpj => ?_)
  obtain ⟨p, j⟩ := pj
  have hj : j < pos.length ∧ pos.getD j 0 = p := by
    rw [List.mem_iff_getElem] at hpj
    obtain ⟨n, hn, hpn⟩ := hpj
    rw [List.getElem_zipIdx] at hpn
    simp only [Prod.mk.injEq, zero_add] at hpn
    obtain ⟨h1, h2⟩ := hpn
    subst h2
    simp only [List.length_zipIdx] at hn
    exact ⟨hn, by rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hn]; simp [h1]⟩
  simp only
  congr 1
  unfold foldColumn
  refine Finset.sum_congr rfl fun lane hl => ?_
  show eqList ch lane * A.embed ((rows j).getD lane 0) = eqList ch lane * A.embed (C lane p)
  rw [h j hj.1 lane (Finset.mem_range.1 hl), hj.2]

theorem mem_querySlots {Q d c t : ℕ} (h : (c, t) ∈ querySlots Q d) : c ≤ d ∧ t < 2 ^ c := by
  simp only [querySlots, List.mem_flatMap, List.mem_map, List.mem_range, Prod.mk.injEq] at h
  obtain ⟨c', hc', t', ht', rfl, rfl⟩ := h
  exact ⟨strata_le Q d c' hc', ht'⟩

/-- Query positions lie on the level's domain. -/
theorem positions_lt (A : Arith F K) (Q d : ℕ) (w : Fin Q → F) :
    ∀ p ∈ positions A Q d w, p < 2 ^ d := by
  intro p hp
  rw [positions_eq, List.mem_ofFn] at hp
  obtain ⟨j, rfl⟩ := hp
  set cs := (querySlots Q d)[j.val]'(by rw [querySlots_length]; exact j.isLt)
  have hmem : cs ∈ querySlots Q d := List.getElem_mem _
  obtain ⟨hc, ht⟩ := mem_querySlots (c := cs.1) (t := cs.2) (by simpa using hmem)
  unfold slotPos
  have hpow : 2 ^ cs.1 * 2 ^ (d - cs.1) = 2 ^ d := by rw [← pow_add]; congr 1; omega
  have hmod := Nat.mod_lt (A.lo (w j)) (Nat.two_pow_pos (d - cs.1))
  calc cs.2 * 2 ^ (d - cs.1) + A.lo (w j) % 2 ^ (d - cs.1)
      < cs.2 * 2 ^ (d - cs.1) + 2 ^ (d - cs.1) := by omega
    _ = (cs.2 + 1) * 2 ^ (d - cs.1) := by ring
    _ ≤ 2 ^ cs.1 * 2 ^ (d - cs.1) := Nat.mul_le_mul_right _ ht
    _ = 2 ^ d := hpow

theorem queryRel_leaf (A : Arith F K) (r : ℕ) (C₀ : Oracle F) (i : IRep F D) (lv : Level)
    (ch : List K) (start : ℕ) (st : LigState F K) (stC : LigStateC F K D) (h : StRel r C₀ i st stC)
    (w : Fin lv.queries → F) (α : Fin (ceilLog2 lv.queries) → F) (β : F) (qO : Quad K)
    (qC : Opens F D → Quad K)
    (hq : ∀ o, AgreeRep r C₀ i (stC.reqs ++ [⟨stC.level, positions A lv.queries lv.logLen w,
        ch.length, pathLen lv⟩]) o → qC o = qO) :
    QueryRel r C₀ lv ch stC i i
      { st with
        s := (st.s + A.embed β *
          consValue A (List.ofFn α) ch st.prev (positions A lv.queries lv.logLen w)),
        q := qO,
        cons := (st.cons ++ [⟨lv.logCols, positions A lv.queries lv.logLen w, List.ofFn α, β,
          start⟩]) }
      { stC with
        s := (fun o => stC.s o + A.embed β * consValueO A (List.ofFn α) ch
          (fun j => (o r stC.level j).1) (positions A lv.queries lv.logLen w)),
        q := qC,
        cons := (stC.cons ++ [⟨lv.logCols, positions A lv.queries lv.logLen w, List.ofFn α, β,
          start⟩]),
        reqs := (stC.reqs ++ [⟨stC.level, positions A lv.queries lv.logLen w, ch.length,
          pathLen lv⟩]) } := by
  refine ⟨rfl, ⟨h.levels, h.oods, by simp [h.cons], h.prevLevel, h.prev, fun o ho => ⟨?_, hq o ho⟩⟩,
    rfl, rfl, rfl, _, rfl, positions_length A _ _ w, positions_lt A _ _ w⟩
  simp only
  rw [(h.sq o ho.of_append).1]
  congr 2
  refine consValueO_eq A _ ch st.prev _ _ fun j hj lane hl => ?_
  exact ho _ (List.mem_append_right _ (List.mem_singleton_self _)) st.prev h.prev j hj lane hl

/-! ## The recursive rounds -/

theorem getRep_slot (r : ℕ) (hr : r < 2) (i : IRep F D) (ℓ : ℕ) (hℓ : 1 ≤ ℓ) :
    getRep r i (slotOf r ℓ) = i ℓ := by
  unfold getRep slotOf
  rw [if_pos ⟨by omega, by omega⟩]
  congr 1
  omega

theorem getRep_update_ne (r : ℕ) (hr : r < 2) (i : IRep F D) (ℓ : ℕ) (x : Option (Oracle F × Cap D))
    (s : ℕ) (hs : s ≠ slotOf r ℓ) : getRep r (Function.update i ℓ x) s = getRep r i s := by
  unfold getRep
  split_ifs with h
  · rw [Function.update_of_ne]
    intro heq
    apply hs
    unfold slotOf
    omega
  · rfl

theorem tabAt_update (C₀ : Oracle F) (i : IRep F D) (ℓ m : ℕ) (x : Option (Oracle F × Cap D))
    (h : m ≠ ℓ) : tabAt C₀ (Function.update i ℓ x) m = tabAt C₀ i m := by
  unfold tabAt
  rw [Function.update_of_ne h]

theorem StRel.update {r : ℕ} {C₀ : Oracle F} {i : IRep F D} {st : LigState F K}
    {stC : LigStateC F K D} (h : StRel r C₀ i st stC) (ℓ : ℕ) (x : Option (Oracle F × Cap D))
    (hℓ : stC.level < ℓ) (hreq : ∀ req ∈ stC.reqs, req.level ≤ stC.level) :
    StRel r C₀ (Function.update i ℓ x) st stC where
  levels := h.levels
  oods := h.oods
  cons := h.cons
  prevLevel := h.prevLevel
  prev := by rw [tabAt_update C₀ i ℓ _ x (by omega)]; exact h.prev
  sq o ho := h.sq o fun req hr t ht => ho req hr t (by
    rw [tabAt_update C₀ i ℓ _ x (by have := hreq req hr; omega)]; exact ht)

theorem getD_of_drop {l : List Level} {L : ℕ} {lv : Level} {rest : List Level}
    (h : lv :: rest = l.drop L) : l.getD L default = lv := by
  have h1 : l[L]? = some lv := by rw [← List.head?_drop, ← h]; rfl
  simp [List.getD_eq_getElem?_getD, h1]

theorem reqs_level_le {reqs : List Req} {L : ℕ} (h : reqs.map Req.level = List.range L) :
    ∀ req ∈ reqs, req.level < L := by
  intro req hr
  have : req.level ∈ reqs.map Req.level := List.mem_map_of_mem hr
  rw [h, List.mem_range] at this
  exact this

theorem reqs_length {reqs : List Req} {L : ℕ} (h : reqs.map Req.level = List.range L) :
    reqs.length = L := by
  have := congrArg List.length h
  simpa using this

theorem RecRel.reject (r : ℕ) (C₀ : Oracle F) (sch : Schedule) (i : IRep F D) :
    RecRel r C₀ sch i false (PendL.reject : PendL F D) :=
  ⟨fun _ _ => rfl, rfl, by simp [PendL.reject], fun ℓ _ h => by simp [PendL.reject] at h,
    by simp [PendL.reject]⟩

/-! ## Ligerito, a rep, a table -/

/-- The oracle table's continuation after the level-0 table `C₀`, with the dummy last message. -/
def tableOrc (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (C₀ : Oracle F) : Game (Bool × Oracle F) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    (interleave (rep A S sch hm C₀ (extraClaims S ptLocal mPts pts))
      (rep A S sch hm C₀ (extraClaims S ptLocal mPts pts))).bind
      fun acc => Game.send (Opens F D) fun _ => Game.ret (acc.1 && acc.2, C₀)

theorem withDummy_table (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) :
    Lock.withDummy (Opens F D) (table A S sch hm ptLocal mPts) =
      Game.send (Oracle F) (tableOrc (D := D) A S sch hm ptLocal mPts) := by
  refine congrArg (Game.send (Oracle F)) (funext fun C₀ =>
    congrArg (@Game.coin _ (Fin 2 → Fin mPts → F) _ _) (funext fun pts => ?_))
  simp only [Game.bind_ret, Game.bind_eq, Game.pure_eq, Game.bind_assoc]

open Classical in
/-- The compiled table's continuation after the level-0 cap. -/
noncomputable def tableCmp (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (cap₀ : Cap D) :
    Game (OutC F D) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts =>
    (interleave (repC A 0 S sch hm (extraClaims S ptLocal mPts pts))
      (repC A 1 S sch hm (extraClaims S ptLocal mPts pts))).bind
      fun pend => Game.send (Opens F D) fun o => Game.ret
        ⟨pend.1.verdict o && pend.2.verdict o &&
          decide (OpensOK H E cap₀ 0 pend.1 o ∧ OpensOK H E cap₀ 1 pend.2 o), cap₀, pend, o⟩

theorem tableC_eq (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) :
    tableC A H E S sch hm ptLocal mPts = Game.send (Cap D) (tableCmp A H E S sch hm ptLocal mPts) :=
  rfl

/-- **The table's leaf relation** below the level-0 cap. -/
def TopRel (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (sch : Schedule) (C₀ : Oracle F)
    (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F) (out : OutC F D) : Prop :=
  a.2 = C₀ ∧ out.cap₀ = cap₀ ∧
    (out.accepted = true → AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens →
      AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens → a.1 = true) ∧
    (out.accepted = true →
      OpensOK H E cap₀ 0 out.pend.1 out.opens ∧ OpensOK H E cap₀ 1 out.pend.2 out.opens) ∧
    RecStruct sch i.1 out.pend.1 ∧ RecStruct sch i.2 out.pend.2

/-- **The table in lockstep** below the level-0 cap: the link points, both reps interleaved, the
openings. -/
noncomputable def lockTable [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₀ : Oracle F)
    (cap₀ : Cap D) :
    Lock (Oracle F) (Cap D) (Lock.getPair (getRep 0) (getRep 1))
      (TopRel H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀ cap₀) ((fun _ => none), (fun _ => none))
      (tableOrc (D := D) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts C₀)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀) :=
  .coin (Fin 2 → Fin mPts → F) fun pts =>
    Lock.bind (R := Lock.RPair (RecRel 0 C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (RecRel 1 C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (fun i acc pend h => .send (Opens F D) fun o => .ret (by
        obtain ⟨⟨hv₀, hs₀⟩, ⟨hv₁, hs₁⟩⟩ := h
        refine ⟨rfl, rfl, fun hacc ha₀ ha₁ => ?_, fun hacc => ?_, hs₀, hs₁⟩
        · simp only [Bool.and_eq_true, decide_eq_true_eq] at hacc
          rw [← hv₀ o ha₀, ← hv₁ o ha₁]
          simp [hacc.1.1, hacc.1.2]
        · simp only [Bool.and_eq_true, decide_eq_true_eq] at hacc
          exact hacc.2))
      (Lock.interleaveLock disjoint_reps
        (lockRep A 0 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₀ (extraClaims S ptLocal mPts pts))
        (lockRep A 1 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₀ (extraClaims S ptLocal mPts pts)))

end FlockSoundness
