import Proofs.Flock.Soundness.Model.Compiled
import Proofs.Flock.Soundness.Game.Lock
import Proofs.Flock.Soundness.LigeritoMid

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

/-- Rep `r`'s slot for its level-`ℓ` cap (`ℓ ≥ 1`). -/
def slotOf (r ℓ : ℕ) : ℕ := 2 * ℓ + r

/-- The tables and caps a rep has fixed, by level. -/
abbrev IRep (F D : Type) := ℕ → Option (Oracle F × Cap D)

/-- Rep `r`'s fixed tables, by slot. -/
def getRep (r : ℕ) (i : IRep F D) (s : ℕ) : Option (Oracle F × Cap D) :=
  if s % 2 = r ∧ 2 ≤ s then i (s / 2) else none

/-- The relation of the folds' outputs: equal challenges (`k` of them), claims equal on openings
in `P`. -/
def FoldRel (P : Opens F D → Prop) (k : ℕ) (i₀ i : IRep F D) (x : List K × K × Quad K)
    (y : List K × (Opens F D → K) × (Opens F D → Quad K)) : Prop :=
  i = i₀ ∧ x.1 = y.1 ∧ x.1.length = k ∧ ∀ o, P o → y.2.1 o = x.2.1 ∧ y.2.2 o = x.2.2

/-- **The folds in lockstep.** -/
def lockFold (A : Arith F K) (r : ℕ) (i : IRep F D) (P : Opens F D → Prop) :
    (k : ℕ) → (s : K) → (q : Quad K) → (sF : Opens F D → K) → (qF : Opens F D → Quad K) →
      (∀ o, P o → sF o = s ∧ qF o = q) →
      Lock (Oracle F) (Cap D) (getRep r) (FoldRel P k i) i (foldRounds A k s q)
        (foldRoundsC A k sF qF)
  | 0, _, _, _, _, h => .ret ⟨rfl, rfl, rfl, h⟩
  | k + 1, s, q, sF, qF, h =>
      .coin (Fin 2 → F) fun w => .send (K × K) fun msg =>
        Lock.bind (g := foldRounds A k (q.eval (A.ofLimbs (w 0) (w 1)))
            (fromMsg msg (q.eval (A.ofLimbs (w 0) (w 1)))))
          (g' := foldRoundsC A k (fun o => (qF o).eval (A.ofLimbs (w 0) (w 1)))
            (fun o => fromMsg msg ((qF o).eval (A.ofLimbs (w 0) (w 1)))))
          (fun _ x y hxy => by
            rcases x with ⟨rs, x₂⟩
            rcases x₂ with ⟨s'', q'⟩
            rcases y with ⟨rs', y₂⟩
            rcases y₂ with ⟨sF'', qF'⟩
            obtain ⟨hi, h1, hl, h2⟩ := hxy
            exact .ret ⟨hi, by simp only at h1 ⊢; rw [h1], by simp only at hl ⊢; simp [hl], h2⟩)
          (lockFold A r i P k _ _ _ _ (fun o ho => by rw [(h o ho).2]; exact ⟨rfl, rfl⟩))

/-- The relation of the two OOD samples' outputs. -/
def OodRel (P : Opens F D → Prop) (i₀ i : IRep F D) (x : K × Quad K × List (OodCtx F))
    (y : (Opens F D → K) × (Opens F D → Quad K) × List (OodCtx F)) : Prop :=
  i = i₀ ∧ x.2.2 = y.2.2 ∧ ∀ o, P o → y.1 o = x.1 ∧ y.2.1 o = x.2.1

/-- **The two OOD samples in lockstep.** -/
def lockTwoOod (A : Arith F K) (r : ℕ) (i : IRep F D) (P : Opens F D → Prop) (dim split split' : ℕ)
    (hsplit : split = split') (s : K) (q : Quad K) (sF : Opens F D → K) (qF : Opens F D → Quad K)
    (h : ∀ o, P o → sF o = s ∧ qF o = q) :
    Lock (Oracle F) (Cap D) (getRep r) (OodRel P i) i (twoOod A dim split s q)
      (twoOodC A dim split' sF qF) := by
  subst hsplit
  exact .coin (Fin dim → F) fun z₁ => .send F fun y₁ => .send (K × K) fun m₁ => .coin F fun β₁ =>
    .coin (Fin dim → F) fun z₂ => .send F fun y₂ => .send (K × K) fun m₂ => .coin F fun β₂ =>
      .ret ⟨rfl, by simp, fun o ho => by
        refine ⟨?_, ?_⟩ <;> simp only [(h o ho).1, (h o ho).2]⟩

/-- The table a rep reads at a level: `C₀` at level 0, else the one fixed at the level. -/
def tabAt (C₀ : Oracle F) (i : IRep F D) (ℓ : ℕ) : Option (Oracle F) :=
  if ℓ = 0 then some C₀ else (i ℓ).map Prod.fst

/-- The openings agree with the fixed tables at every position rep `r` requested. -/
def AgreeRep (r : ℕ) (C₀ : Oracle F) (i : IRep F D) (reqs : List Req) (o : Opens F D) : Prop :=
  ∀ req ∈ reqs, ∀ t, tabAt C₀ i req.level = some t → ∀ j < req.positions.length,
    ∀ lane < 2 ^ req.lanes, (o r req.level j).1.getD lane 0 = t lane (req.positions.getD j 0)

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

/-- The oracle and compiled Ligerito states correspond (rep `r`, level-0 table `C₀`, fixed
tables `i`): equal records, the queried table fixed, and equal claims on agreeing openings. -/
structure StRel (r : ℕ) (C₀ : Oracle F) (i : IRep F D) (st : LigState F K)
    (stC : LigStateC F K D) : Prop where
  levels : st.levels = stC.levels
  oods : st.oods = stC.oods
  cons : st.cons = stC.cons
  prevLevel : st.prevLevel = stC.prevLevel
  prev : tabAt C₀ i stC.level = some st.prev
  sq : ∀ o, AgreeRep r C₀ i stC.reqs o → stC.s o = st.s ∧ stC.q o = st.q

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

/-- The queries' output relation: the states correspond, and one request of the level's shape
was appended. -/
def QueryRel (r : ℕ) (C₀ : Oracle F) (lv : Level) (ch : List K) (stC : LigStateC F K D)
    (i₀ i : IRep F D) (st' : LigState F K) (stC' : LigStateC F K D) : Prop :=
  i = i₀ ∧ StRel r C₀ i st' stC' ∧ stC'.level = stC.level ∧ stC'.caps = stC.caps ∧
    stC'.splitDim = stC.splitDim ∧
    ∃ pos : List ℕ, stC'.reqs = stC.reqs ++ [⟨stC.level, pos, ch.length, pathLen lv⟩] ∧
      pos.length = lv.queries ∧ ∀ p ∈ pos, p < 2 ^ lv.logLen

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

/-- **The queries in lockstep.** -/
def lockQuery (A : Arith F K) (r : ℕ) (C₀ : Oracle F) (i : IRep F D) (lv : Level) (ch : List K)
    (start start' : ℕ) (hstart : start = start') (st : LigState F K) (stC : LigStateC F K D)
    (h : StRel r C₀ i st stC) (wm : Bool) :
    Lock (Oracle F) (Cap D) (getRep r) (QueryRel r C₀ lv ch stC i) i
      (queryAndBatch A lv st.prev ch start st wm) (queryAndBatchC A r lv ch start' stC wm) := by
  subst hstart
  exact lockQuery' A r C₀ i lv ch start st stC h wm
where
  lockQuery' (A : Arith F K) (r : ℕ) (C₀ : Oracle F) (i : IRep F D) (lv : Level) (ch : List K)
      (start : ℕ) (st : LigState F K) (stC : LigStateC F K D) (h : StRel r C₀ i st stC) :
      (wm : Bool) → Lock (Oracle F) (Cap D) (getRep r) (QueryRel r C₀ lv ch stC i) i
        (queryAndBatch A lv st.prev ch start st wm) (queryAndBatchC A r lv ch start stC wm)
  | true => .coin (Fin lv.queries → F) fun w => .coin (Fin (ceilLog2 lv.queries) → F) fun α =>
      .send (K × K) fun msg => .coin F fun β =>
        .ret (queryRel_leaf A r C₀ i lv ch start st stC h w α β _ _ fun o ho => by
          simp only
          rw [(h.sq o ho.of_append).2]
          congr 2
          refine consValueO_eq A _ ch st.prev _ _ fun j hj lane hl => ?_
          exact ho _ (List.mem_append_right _ (List.mem_singleton_self _)) st.prev h.prev j hj lane hl)
  | false => .coin (Fin lv.queries → F) fun w => .coin (Fin (ceilLog2 lv.queries) → F) fun α =>
      .coin F fun β =>
        .ret (queryRel_leaf A r C₀ i lv ch start st stC h w α β _ _ fun o ho =>
          (h.sq o ho.of_append).2)

/-! ## The recursive rounds -/

/-- A request's shape: its level's queries, domain, lanes and path length. -/
def ReqShape (sch : Schedule) (req : Req) : Prop :=
  req.positions.length = (sch.levels.getD req.level default).queries ∧
    (∀ p ∈ req.positions, p < 2 ^ (sch.levels.getD req.level default).logLen) ∧
    req.lanes = (sch.levels.getD req.level default).lanes ∧
    req.depth = pathLen (sch.levels.getD req.level default)

/-- The requests' structure a rep's leaf records: one per level of the schedule, in order, of
the level's shape, against the fixed caps. -/
def RecStruct (sch : Schedule) (i : IRep F D) (p : PendL F D) : Prop :=
  p.reqs.map Req.level = List.range p.reqs.length ∧ (∀ req ∈ p.reqs, ReqShape sch req) ∧
    (∀ ℓ, 1 ≤ ℓ → ℓ < p.reqs.length → ∃ t d, i ℓ = some (t, d) ∧ p.caps[ℓ - 1]? = some d) ∧
    p.reqs.length ≤ sch.levels.length

/-- **A rep's leaf relation**: on agreeing openings the pending verdict is the oracle's, and the
requests have their structure. -/
def RecRel (r : ℕ) (C₀ : Oracle F) (sch : Schedule) (i : IRep F D) (v : Bool) (p : PendL F D) :
    Prop :=
  (∀ o, AgreeRep r C₀ i p.reqs o → p.verdict o = v) ∧ RecStruct sch i p

/-- The recursion's invariant when it is called on level `stC.level`. -/
structure RecInv (r : ℕ) (C₀ : Oracle F) (sch : Schedule) (levs : List Level) (i : IRep F D)
    (st : LigState F K) (stC : LigStateC F K D) : Prop where
  rel : StRel r C₀ i st stC
  levs : levs = sch.levels.drop stC.level
  reqs : stC.reqs.map Req.level = List.range stC.level
  shape : ∀ req ∈ stC.reqs, ReqShape sch req
  fresh : ∀ ℓ, stC.level < ℓ → i ℓ = none
  capsLen : stC.caps.length = stC.level
  capsRec : ∀ ℓ, 1 ≤ ℓ → ℓ ≤ stC.level → ∃ t d, i ℓ = some (t, d) ∧ stC.caps[ℓ - 1]? = some d

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

/-- **The recursive rounds in lockstep.** -/
def lockRec (A : Arith F K) (r : ℕ) (hr : r < 2) (n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ)
    (C₀ : Oracle F) (sch : Schedule) :
    (levs : List Level) → (i : IRep F D) → (st : LigState F K) → (stC : LigStateC F K D) →
      RecInv r C₀ sch levs i st stC →
        Lock (Oracle F) (Cap D) (getRep r) (RecRel r C₀ sch) i
          (recursiveRounds A n b initial ext levs st) (recursiveRoundsC A r n b initial ext levs stC)
  | [], _, _, _, _ => .ret ⟨fun _ _ => rfl, rfl, by simp [PendL.reject], fun ℓ _ h => by
      simp [PendL.reject] at h, by simp [PendL.reject]⟩
  | [lv], i, st, stC, h =>
      Lock.bind (g := foldRounds A lv.lanes st.s st.q) (g' := foldRoundsC A lv.lanes stC.s stC.q)
        (fun i' x y hxy => by
          rcases x with ⟨ch, x₂⟩
          rcases x₂ with ⟨s, q⟩
          rcases y with ⟨ch', y₂⟩
          rcases y₂ with ⟨sF, qF⟩
          obtain ⟨hi, hch, hlen, hsq⟩ := hxy
          simp only at hch hlen hsq
          subst i'
          subst hch
          have h1 : StRel r C₀ i { st with s := s, q := q, levels := st.levels ++ [ch] }
              { stC with s := sF, q := qF, levels := stC.levels ++ [ch] } :=
            ⟨by simp [h.rel.levels], h.rel.oods, h.rel.cons, h.rel.prevLevel, h.rel.prev,
              fun o ho => hsq o ho⟩
          exact .send (Fin (2 ^ (ext + 1)) → F) fun yr =>
            Lock.bind (g := queryAndBatch A lv st.prev ch (st.levels ++ [ch]).length
                { st with s := s, q := q, levels := st.levels ++ [ch] } false)
              (g' := queryAndBatchC A r lv ch (stC.levels ++ [ch]).length
                { stC with s := sF, q := qF, levels := stC.levels ++ [ch] } false)
              (R := QueryRel r C₀ lv ch { stC with s := sF, q := qF, levels := stC.levels ++ [ch] } i)
              (fun i3 st3 stC3 hq => by
                obtain ⟨hi3, hq'⟩ := hq
                subst i3
                exact .ret (by
                obtain ⟨hrel, hlev, hcaps, -, pos, hreqs, hposlen, hposlt⟩ := hq'
                simp only at hlev hcaps hreqs
                have hlv := getD_of_drop h.levs
                have hlenR := reqs_length h.reqs
                refine ⟨fun o ho => ?_, ?_, ?_, ?_, ?_⟩
                · simp only
                  rw [(hrel.sq o ho).1, hrel.levels, hrel.oods, hrel.cons]
                · rw [hreqs, List.map_append, h.reqs, List.length_append, hlenR]
                  simp [List.range_succ]
                · intro req hreq
                  rw [hreqs] at hreq
                  rcases List.mem_append.1 hreq with hreq | hreq
                  · exact h.shape req hreq
                  · rw [List.mem_singleton] at hreq
                    subst hreq
                    unfold ReqShape
                    simp only
                    rw [hlv]
                    exact ⟨hposlen, hposlt, hlen, rfl⟩
                · intro ℓ hℓ1 hℓ2
                  rw [hreqs, List.length_append, hlenR] at hℓ2
                  show ∃ t d, _ ∧ stC3.caps[ℓ - 1]? = some d
                  rw [hcaps]
                  exact h.capsRec ℓ hℓ1 (by simp at hℓ2; omega)
                · have hd := congrArg List.length h.levs
                  rw [List.length_drop] at hd
                  show stC3.reqs.length ≤ _
                  rw [hreqs, List.length_append, hlenR]
                  simp only [List.length_singleton, List.length_cons, List.length_nil] at hd ⊢
                  omega))
              (lockQuery A r C₀ i lv ch (st.levels ++ [ch]).length (stC.levels ++ [ch]).length
                (by rw [h.rel.levels]) _ _ h1 false))
        (lockFold A r i (AgreeRep r C₀ i stC.reqs) lv.lanes st.s st.q stC.s stC.q h.rel.sq)
  | lv :: lv' :: rest, i, st, stC, h =>
      Lock.bind (g := foldRounds A lv.lanes st.s st.q) (g' := foldRoundsC A lv.lanes stC.s stC.q)
        (fun i' x y hxy => by
          rcases x with ⟨ch, x₂⟩
          rcases x₂ with ⟨s, q⟩
          rcases y with ⟨ch', y₂⟩
          rcases y₂ with ⟨sF, qF⟩
          obtain ⟨hi, hch, hlen, hsq⟩ := hxy
          simp only at hch hlen hsq
          subst i'
          subst hch
          have hreqle : ∀ req ∈ stC.reqs, req.level ≤ stC.level :=
            fun req hr => (reqs_level_le h.reqs req hr).le
          exact .table (slotOf r (stC.level + 1))
            (fun t d => Function.update i (stC.level + 1) (some (t, d)))
            (by rw [getRep_slot r hr i _ (by omega)]; exact h.fresh _ (by omega))
            (fun t d => by rw [getRep_slot r hr _ _ (by omega)]; simp)
            (fun t d s hs => getRep_update_ne r hr i _ _ s hs)
            fun C' cap => by
              have h1₀ : StRel r C₀ i { st with s := s, q := q, levels := st.levels ++ [ch] }
                  { stC with s := sF, q := qF, levels := stC.levels ++ [ch] } :=
                ⟨by simp [h.rel.levels], h.rel.oods, h.rel.cons, h.rel.prevLevel, h.rel.prev,
                  fun o ho => hsq o ho⟩
              have h1 : StRel r C₀ (Function.update i (stC.level + 1) (some (C', cap)))
                  { st with s := s, q := q, levels := st.levels ++ [ch] }
                  { stC with s := sF, q := qF, levels := stC.levels ++ [ch] } :=
                h1₀.update _ _ (by show stC.level < stC.level + 1; omega) hreqle
              exact Lock.bind (g := twoOod A (lv'.logCols + lv'.lanes) (st.levels ++ [ch]).length s q)
                (g' := twoOodC A (lv'.logCols + lv'.lanes) (stC.levels ++ [ch]).length sF qF)
                (R := OodRel (AgreeRep r C₀ (Function.update i (stC.level + 1) (some (C', cap)))
                  stC.reqs) (Function.update i (stC.level + 1) (some (C', cap))))
                (fun i₂ x y hxy => by
                  rcases x with ⟨s2, x₂⟩
                  rcases x₂ with ⟨q2, oods2⟩
                  rcases y with ⟨sF2, y₂⟩
                  rcases y₂ with ⟨qF2, oodsC2⟩
                  obtain ⟨hi2, hoods, hsq2⟩ := hxy
                  simp only at hoods hsq2
                  subst i₂
                  subst hoods
                  have h2 : StRel r C₀ (Function.update i (stC.level + 1) (some (C', cap)))
                      { st with s := s2, q := q2, levels := st.levels ++ [ch], oods := st.oods ++ oods2 }
                      { stC with s := sF2, q := qF2, levels := stC.levels ++ [ch], oods := stC.oods ++ oods2 } :=
                    ⟨by simp [h.rel.levels], by simp [h.rel.oods], h.rel.cons, h.rel.prevLevel,
                      h1.prev, fun o ho => hsq2 o ho⟩
                  exact Lock.bind
                    (g := queryAndBatch A lv st.prev ch (st.levels ++ [ch]).length
                      { st with s := s2, q := q2, levels := st.levels ++ [ch], oods := st.oods ++ oods2 } true)
                    (g' := queryAndBatchC A r lv ch (stC.levels ++ [ch]).length
                      { stC with s := sF2, q := qF2, levels := stC.levels ++ [ch], oods := stC.oods ++ oods2 } true)
                    (R := QueryRel r C₀ lv ch
                      { stC with s := sF2, q := qF2, levels := stC.levels ++ [ch], oods := stC.oods ++ oods2 }
                      (Function.update i (stC.level + 1) (some (C', cap))))
                    (fun i₃ st3 stC3 hq => by
                      obtain ⟨hi3, hq'⟩ := hq
                      subst i₃
                      exact lockRec A r hr n b initial ext C₀ sch (lv' :: rest) _
                        { st3 with prev := C', prevLevel := lv' }
                        { stC3 with level := stC3.level + 1, caps := stC3.caps ++ [cap], prevLevel := lv' }
                        (by
                        obtain ⟨hrel, hlev, hcaps, -, pos, hreqs, hposlen, hposlt⟩ := hq'
                        simp only at hlev hcaps hreqs
                        have hlv := getD_of_drop h.levs
                        have hlenR := reqs_length h.reqs
                        refine ⟨⟨hrel.levels, hrel.oods, hrel.cons, rfl, ?_, hrel.sq⟩, ?_, ?_, ?_, ?_,
                          ?_, ?_⟩
                        · simp only [tabAt, hlev, Function.update_self, Option.map_some]
                          simp
                        · simp only [hlev]
                          rw [← List.tail_drop, ← h.levs]
                          rfl
                        · simp only [hlev, hreqs, List.map_append, h.reqs]
                          simp [List.range_succ]
                        · intro req hreq
                          simp only [hreqs] at hreq
                          rcases List.mem_append.1 hreq with hreq | hreq
                          · exact h.shape req hreq
                          · rw [List.mem_singleton] at hreq
                            subst hreq
                            unfold ReqShape
                            simp only
                            rw [hlv]
                            exact ⟨hposlen, hposlt, hlen, rfl⟩
                        · intro ℓ hℓ
                          simp only [hlev] at hℓ
                          rw [Function.update_of_ne (by omega)]
                          exact h.fresh ℓ (by omega)
                        · simp only [hlev, hcaps, List.length_append, List.length_singleton, h.capsLen]
                        · intro ℓ hℓ1 hℓ2
                          simp only [hlev] at hℓ2
                          simp only [hcaps]
                          by_cases hℓ : ℓ = stC.level + 1
                          · subst hℓ
                            refine ⟨C', cap, by simp, ?_⟩
                            rw [List.getElem?_append_right (by rw [h.capsLen]; omega), h.capsLen]
                            simp
                          · obtain ⟨t, d, htd, hcap⟩ := h.capsRec ℓ hℓ1 (by omega)
                            refine ⟨t, d, by rw [Function.update_of_ne hℓ]; exact htd, ?_⟩
                            rw [List.getElem?_append_left (by rw [h.capsLen]; omega)]
                            exact hcap))
                    (lockQuery A r C₀ _ lv ch (st.levels ++ [ch]).length (stC.levels ++ [ch]).length
                      (by rw [h.rel.levels]) _ _ h2 true))
                (lockTwoOod A r _ (AgreeRep r C₀ (Function.update i (stC.level + 1) (some (C', cap)))
                  stC.reqs) _ _ _ (by rw [h.rel.levels]) s q sF qF (fun o ho => h1.sq o ho)))
        (lockFold A r i (AgreeRep r C₀ i stC.reqs) lv.lanes st.s st.q stC.s stC.q h.rel.sq)

theorem RecRel.reject (r : ℕ) (C₀ : Oracle F) (sch : Schedule) (i : IRep F D) :
    RecRel r C₀ sch i false (PendL.reject : PendL F D) :=
  ⟨fun _ _ => rfl, rfl, by simp [PendL.reject], fun ℓ _ h => by simp [PendL.reject] at h,
    by simp [PendL.reject]⟩

/-! ## Ligerito, a rep, a table -/

/-- **Ligerito in lockstep**, for a schedule of at least two levels whose level 0 folds `k0`
lanes. -/
def lockLig (A : Arith F K) (r : ℕ) (hr : r < 2) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (T : F) (b : ℕ → F) (C₀ : Oracle F) :
    Lock (Oracle F) (Cap D) (getRep r) (RecRel r C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (fun _ => none)
      (ligerito A ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T b C₀)
      (ligeritoC A r ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T b) :=
  .coin (Fin (m - 7) → F) fun z₀ => .send F fun y₀ => .coin F fun β₀ => .send (K × K) fun msg =>
    Lock.bind (g := foldRounds A k0 (A.embed T + A.embed β₀ * A.embed y₀)
        (fromMsg msg (A.embed T + A.embed β₀ * A.embed y₀)))
      (g' := foldRoundsC A k0 (fun _ => A.embed T + A.embed β₀ * A.embed y₀)
        (fun _ => fromMsg msg (A.embed T + A.embed β₀ * A.embed y₀)))
      (R := FoldRel (fun _ => True) k0 (fun _ => none))
      (fun i' x y hxy => by
        rcases x with ⟨initial, x₂⟩
        rcases x₂ with ⟨s, q⟩
        rcases y with ⟨initial', y₂⟩
        rcases y₂ with ⟨sF, qF⟩
        obtain ⟨hi, hch, hlen, hsq⟩ := hxy
        simp only at hch hlen hsq
        subst i'
        subst hch
        exact .table (slotOf r 1) (fun t d => Function.update (fun _ => none : IRep F D) 1 (some (t, d)))
          (by rw [getRep_slot r hr _ _ le_rfl])
          (fun t d => by rw [getRep_slot r hr _ _ le_rfl]; simp)
          (fun t d s hs => getRep_update_ne r hr _ _ _ s hs)
          fun C₁ cap₁ =>
            Lock.bind (g := twoOod A (m - 7 - k0 + 1) 0 s q)
              (g' := twoOodC A (m - 7 - k0 + 1) 0 sF qF)
              (R := OodRel (fun _ => True) (Function.update (fun _ => none : IRep F D) 1 (some (C₁, cap₁))))
              (fun i₂ x y hxy => by
                rcases x with ⟨s2, x₂⟩
                rcases x₂ with ⟨q2, oods⟩
                rcases y with ⟨sF2, y₂⟩
                rcases y₂ with ⟨qF2, oodsC⟩
                obtain ⟨hi2, hoods, hsq2⟩ := hxy
                simp only at hoods hsq2
                subst i₂
                subst hoods
                have h2 : StRel r C₀ (Function.update (fun _ => none : IRep F D) 1 (some (C₁, cap₁)))
                    (⟨s2, q2, m - 7 - k0 + 1, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], C₀, l₀⟩ :
                      LigState F K)
                    (⟨sF2, qF2, m - 7 - k0 + 1, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], 0, [], [],
                      l₀⟩ : LigStateC F K D) :=
                  ⟨rfl, rfl, rfl, rfl, by simp [tabAt], fun o _ => hsq2 o trivial⟩
                exact Lock.bind
                  (g := queryAndBatch A l₀ C₀ initial 0
                    ⟨s2, q2, m - 7 - k0 + 1, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], C₀, l₀⟩ true)
                  (g' := queryAndBatchC A r l₀ initial 0
                    ⟨sF2, qF2, m - 7 - k0 + 1, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], 0, [], [], l₀⟩
                    true)
                  (R := QueryRel r C₀ l₀ initial
                    ⟨sF2, qF2, m - 7 - k0 + 1, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], 0, [], [], l₀⟩
                    (Function.update (fun _ => none : IRep F D) 1 (some (C₁, cap₁))))
                  (fun i₃ st3 stC3 hq => by
                    obtain ⟨hi3, hq'⟩ := hq
                    subst i₃
                    exact lockRec A r hr (m - 7) b initial ext C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
                      (l₁ :: rest) _ { st3 with prev := C₁, prevLevel := l₁ }
                      { stC3 with level := 1, caps := [cap₁], prevLevel := l₁ } (by
                      obtain ⟨hrel, hlev, hcaps, -, pos, hreqs, hposlen, hposlt⟩ := hq'
                      simp only at hlev hcaps hreqs
                      refine ⟨⟨hrel.levels, hrel.oods, hrel.cons, rfl, by simp [tabAt], hrel.sq⟩,
                        rfl, ?_, ?_, ?_, rfl, ?_⟩
                      · simp [hreqs]
                      · intro req hreq
                        simp only [hreqs, List.nil_append, List.mem_singleton] at hreq
                        subst hreq
                        exact ⟨hposlen, hposlt, by simp [hlen, hk0], rfl⟩
                      · intro ℓ hℓ
                        dsimp only at hℓ
                        rw [Function.update_of_ne (by omega)]
                      · intro ℓ hℓ1 hℓ2
                        dsimp only at hℓ2
                        obtain rfl : ℓ = 1 := by omega
                        exact ⟨C₁, cap₁, by simp, rfl⟩))
                  (lockQuery A r C₀ _ l₀ initial 0 0 rfl _ _ h2 true))
              (lockTwoOod A r _ (fun _ => True) _ 0 0 rfl s q sF qF fun o _ => hsq o trivial))
      (lockFold A r (fun _ => none) (fun _ => True) k0 _ _ _ _ fun _ _ => ⟨rfl, rfl⟩)

/-- **A rep in lockstep**: the PIOP phases are the same games; Ligerito is `lockLig`. -/
def lockRep (A : Arith F K) (r : ℕ) (hr : r < 2) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) :
    Lock (Oracle F) (Cap D) (getRep r) (RecRel r C₀ ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (fun _ => none)
      (rep A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm C₀ extra)
      (repC A r S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm extra) :=
  Lock.bind (R := fun i' a b => i' = (fun _ => none) ∧ a = b)
    (fun i₁ zc zc' hzc => by
      obtain ⟨hi, rfl⟩ := hzc
      subst i₁
      match zc with
      | none => exact .ret (RecRel.reject r C₀ _ _)
      | some zc =>
        exact Lock.bind (R := fun i' a b => i' = (fun _ => none) ∧ a = b)
          (fun i₂ ab ab' hab => by
            obtain ⟨hi, rfl⟩ := hab
            subst i₂
            match ab with
            | none => exact .ret (RecRel.reject r C₀ _ _)
            | some ab =>
              exact Lock.bind (R := fun i' a b => i' = (fun _ => none) ∧ a = b)
                (fun i₃ op op' hop => by
                  obtain ⟨hi, rfl⟩ := hop
                  subst i₃
                  match op with
                  | none => exact .ret (RecRel.reject r C₀ _ _)
                  | some op => exact lockLig A r hr m k0 l₀ l₁ rest ext hk0 op.T op.b C₀)
                (Lock.same (opening A (by omega) (ab :: cClaim A zc :: extra))))
          (Lock.same (lincheck A S zc)))
    (Lock.same (zerocheck A S.m))

theorem disjoint_reps : Lock.Disjoint (getRep 0 : IRep F D → ℕ → Option (Oracle F × Cap D))
    (getRep 1) where
  left i₁ i₂ s h := by
    unfold getRep at h ⊢
    by_cases hs : s % 2 = 0 ∧ 2 ≤ s
    · rw [if_neg fun h2 => by omega]
    · rw [if_neg hs] at h; simp at h
  right i₁ i₂ s h := by
    unfold getRep at h ⊢
    by_cases hs : s % 2 = 1 ∧ 2 ≤ s
    · rw [if_neg fun h2 => by omega]
    · rw [if_neg hs] at h; simp at h

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

instance instInhabitedOracle : Inhabited (Oracle F) := ⟨fun _ _ => 0⟩

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
