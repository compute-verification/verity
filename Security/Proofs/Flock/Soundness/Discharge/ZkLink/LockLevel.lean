import Proofs.Flock.Soundness.Discharge.ZkLink.OracleP
import Proofs.Flock.Soundness.CompiledAlign

/-!
# The compiled masked level 0 in lockstep with the oracle one

`CompiledLock`'s lockstep with level 0's rows `w₀` lanes wide (`AgreeZ`: `AgreeRep` with level 0's lanes `< w₀`, the
witness lanes and both reps' extra lanes of one tree). `lockRecZ` is `lockRec` over `AgreeZ`; `lockLigPadZ` runs C-Flock
ZK's masked level 0 (`ligeritoPadZK`) beside its compiled form (`ligeritoPadZC`), whose level-0 consistency value is
computed from the opened rows (`padColumnO_eq`). The folds and the OOD samples are `CompiledLock`'s (`lockFold`,
`lockTwoOod`), the masked lane rounds and `ρ` the same game on both sides (`lockFoldZK`, `Lock.same`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-! ## Agreement with level 0's wide rows -/

/-- The openings agree with the fixed tables at every position rep `r` requested, level 0's rows on their `w₀` lanes
and level `ℓ ≥ 1`'s on `2^lanes`. -/
def AgreeZ (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (i : IRep F D) (reqs : List Req) (o : Opens F D) : Prop :=
  ∀ req ∈ reqs, ∀ t, tabAt C₀ i req.level = some t → ∀ j < req.positions.length,
    ∀ lane < (if req.level = 0 then w₀ else 2 ^ req.lanes),
      (o r req.level j).1.getD lane 0 = t lane (req.positions.getD j 0)

omit [Fintype F] [DecidableEq F] in
theorem AgreeZ.of_append {r : ℕ} {C₀ : Oracle F} {w₀ : ℕ} {i : IRep F D} {reqs reqs' : List Req}
    {o : Opens F D} (h : AgreeZ r C₀ w₀ i (reqs ++ reqs') o) : AgreeZ r C₀ w₀ i reqs o :=
  fun req hr => h req (List.mem_append_left _ hr)

omit [Fintype F] [DecidableEq F] in
theorem AgreeZ.last {r : ℕ} {C₀ : Oracle F} {w₀ : ℕ} {i : IRep F D} {reqs : List Req} {req : Req}
    {o : Opens F D} (h : AgreeZ r C₀ w₀ i (reqs ++ [req]) o) {t : Oracle F} (ht : tabAt C₀ i req.level = some t)
    (j : ℕ) (hj : j < req.positions.length) (lane : ℕ) (hl : lane < if req.level = 0 then w₀ else 2 ^ req.lanes) :
    (o r req.level j).1.getD lane 0 = t lane (req.positions.getD j 0) :=
  h req (List.mem_append_right _ (List.mem_singleton_self _)) t ht j hj lane hl

/-- `StRel` over `AgreeZ`. -/
structure StRelZ (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (i : IRep F D) (st : LigState F K)
    (stC : LigStateC F K D) : Prop where
  levels : st.levels = stC.levels
  oods : st.oods = stC.oods
  cons : st.cons = stC.cons
  prevLevel : st.prevLevel = stC.prevLevel
  prev : tabAt C₀ i stC.level = some st.prev
  sq : ∀ o, AgreeZ r C₀ w₀ i stC.reqs o → stC.s o = st.s ∧ stC.q o = st.q

/-- `QueryRel` over `AgreeZ`, the appended request `lanes` lanes wide. -/
def QueryRelZ (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (lv : Level) (lanes : ℕ) (stC : LigStateC F K D)
    (i₀ i : IRep F D) (st' : LigState F K) (stC' : LigStateC F K D) : Prop :=
  i = i₀ ∧ StRelZ r C₀ w₀ i st' stC' ∧ stC'.level = stC.level ∧ stC'.caps = stC.caps ∧
    stC'.splitDim = stC.splitDim ∧
    ∃ pos : List ℕ, stC'.reqs = stC.reqs ++ [⟨stC.level, pos, lanes, pathLen lv⟩] ∧
      pos.length = lv.queries ∧ ∀ p ∈ pos, p < 2 ^ lv.logLen

theorem queryRel_leafZ (A : Arith F K) (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (i : IRep F D) (lv : Level)
    (ch : List K) (start : ℕ) (st : LigState F K) (stC : LigStateC F K D) (h : StRelZ r C₀ w₀ i st stC)
    (hlev : stC.level ≠ 0) (w : Fin lv.queries → F) (α : Fin (ceilLog2 lv.queries) → F) (β : F) (qO : Quad K)
    (qC : Opens F D → Quad K)
    (hq : ∀ o, AgreeZ r C₀ w₀ i (stC.reqs ++ [⟨stC.level, positions A lv.queries lv.logLen w,
        ch.length, pathLen lv⟩]) o → qC o = qO) :
    QueryRelZ r C₀ w₀ lv ch.length stC i i
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
  exact AgreeZ.last ho h.prev j hj lane (by simpa [hlev] using hl)

/-- **The queries of a level `ℓ ≥ 1` in lockstep**, over `AgreeZ`. -/
def lockQueryZ (A : Arith F K) (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (i : IRep F D) (lv : Level) (ch : List K)
    (start start' : ℕ) (hstart : start = start') (st : LigState F K) (stC : LigStateC F K D)
    (h : StRelZ r C₀ w₀ i st stC) (hlev : stC.level ≠ 0) (wm : Bool) :
    Lock (Oracle F) (Cap D) (getRep r) (QueryRelZ r C₀ w₀ lv ch.length stC i) i
      (queryAndBatch A lv st.prev ch start st wm) (queryAndBatchC A r lv ch start' stC wm) := by
  subst hstart
  exact lockQueryZ' A r C₀ w₀ i lv ch start st stC h hlev wm
where
  lockQueryZ' (A : Arith F K) (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (i : IRep F D) (lv : Level) (ch : List K)
      (start : ℕ) (st : LigState F K) (stC : LigStateC F K D) (h : StRelZ r C₀ w₀ i st stC) (hlev : stC.level ≠ 0) :
      (wm : Bool) → Lock (Oracle F) (Cap D) (getRep r) (QueryRelZ r C₀ w₀ lv ch.length stC i) i
        (queryAndBatch A lv st.prev ch start st wm) (queryAndBatchC A r lv ch start stC wm)
  | true => .coin (Fin lv.queries → F) fun w => .coin (Fin (ceilLog2 lv.queries) → F) fun α =>
      .send (K × K) fun msg => .coin F fun β =>
        .ret (queryRel_leafZ A r C₀ w₀ i lv ch start st stC h hlev w α β _ _ fun o ho => by
          simp only
          rw [(h.sq o ho.of_append).2]
          congr 2
          refine consValueO_eq A _ ch st.prev _ _ fun j hj lane hl => ?_
          exact AgreeZ.last ho h.prev j hj lane (by simpa [hlev] using hl))
  | false => .coin (Fin lv.queries → F) fun w => .coin (Fin (ceilLog2 lv.queries) → F) fun α =>
      .coin F fun β =>
        .ret (queryRel_leafZ A r C₀ w₀ i lv ch start st stC h hlev w α β _ _ fun o ho =>
          (h.sq o ho.of_append).2)

/-! ## The recursive rounds -/

/-- `RecRel` over `AgreeZ`. -/
def RecRelZ (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (sch : Schedule) (i : IRep F D) (v : Bool) (p : PendL F D) : Prop :=
  (∀ o, AgreeZ r C₀ w₀ i p.reqs o → p.verdict o = v) ∧ RecStruct sch i p

/-- `RecInv` over `AgreeZ`, called on a level `≥ 1`. -/
structure RecInvZ (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (sch : Schedule) (levs : List Level) (i : IRep F D)
    (st : LigState F K) (stC : LigStateC F K D) : Prop where
  rel : StRelZ r C₀ w₀ i st stC
  levs : levs = sch.levels.drop stC.level
  reqs : stC.reqs.map Req.level = List.range stC.level
  shape : ∀ req ∈ stC.reqs, ReqShape sch req
  fresh : ∀ ℓ, stC.level < ℓ → i ℓ = none
  capsLen : stC.caps.length = stC.level
  capsRec : ∀ ℓ, 1 ≤ ℓ → ℓ ≤ stC.level → ∃ t d, i ℓ = some (t, d) ∧ stC.caps[ℓ - 1]? = some d
  lev1 : stC.level ≠ 0

omit [Field K] [Fintype K] [DecidableEq K] in
theorem StRelZ.update {r : ℕ} {C₀ : Oracle F} {w₀ : ℕ} {i : IRep F D} {st : LigState F K}
    {stC : LigStateC F K D} (h : StRelZ r C₀ w₀ i st stC) (ℓ : ℕ) (x : Option (Oracle F × Cap D))
    (hℓ : stC.level < ℓ) (hreq : ∀ req ∈ stC.reqs, req.level ≤ stC.level) :
    StRelZ r C₀ w₀ (Function.update i ℓ x) st stC where
  levels := h.levels
  oods := h.oods
  cons := h.cons
  prevLevel := h.prevLevel
  prev := by rw [tabAt_update C₀ i ℓ _ x (by omega)]; exact h.prev
  sq o ho := h.sq o fun req hr t ht => ho req hr t (by
    rw [tabAt_update C₀ i ℓ _ x (by have := hreq req hr; omega)]; exact ht)

/-- **The recursive rounds in lockstep**, over `AgreeZ`. -/
def lockRecZ (A : Arith F K) (r : ℕ) (hr : r < 2) (n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ)
    (C₀ : Oracle F) (w₀ : ℕ) (sch : Schedule) :
    (levs : List Level) → (i : IRep F D) → (st : LigState F K) → (stC : LigStateC F K D) →
      RecInvZ r C₀ w₀ sch levs i st stC →
        Lock (Oracle F) (Cap D) (getRep r) (RecRelZ r C₀ w₀ sch) i
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
          have h1 : StRelZ r C₀ w₀ i { st with s := s, q := q, levels := st.levels ++ [ch] }
              { stC with s := sF, q := qF, levels := stC.levels ++ [ch] } :=
            ⟨by simp [h.rel.levels], h.rel.oods, h.rel.cons, h.rel.prevLevel, h.rel.prev,
              fun o ho => hsq o ho⟩
          exact .send (Fin (2 ^ (ext + 1)) → F) fun yr =>
            Lock.bind (g := queryAndBatch A lv st.prev ch (st.levels ++ [ch]).length
                { st with s := s, q := q, levels := st.levels ++ [ch] } false)
              (g' := queryAndBatchC A r lv ch (stC.levels ++ [ch]).length
                { stC with s := sF, q := qF, levels := stC.levels ++ [ch] } false)
              (R := QueryRelZ r C₀ w₀ lv ch.length { stC with s := sF, q := qF, levels := stC.levels ++ [ch] } i)
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
                  simp only [List.length_cons, List.length_nil] at hd ⊢
                  omega))
              (lockQueryZ A r C₀ w₀ i lv ch (st.levels ++ [ch]).length (stC.levels ++ [ch]).length
                (by rw [h.rel.levels]) _ _ h1 h.lev1 false))
        (lockFold A r i (AgreeZ r C₀ w₀ i stC.reqs) lv.lanes st.s st.q stC.s stC.q h.rel.sq)
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
              have h1₀ : StRelZ r C₀ w₀ i { st with s := s, q := q, levels := st.levels ++ [ch] }
                  { stC with s := sF, q := qF, levels := stC.levels ++ [ch] } :=
                ⟨by simp [h.rel.levels], h.rel.oods, h.rel.cons, h.rel.prevLevel, h.rel.prev,
                  fun o ho => hsq o ho⟩
              have h1 : StRelZ r C₀ w₀ (Function.update i (stC.level + 1) (some (C', cap)))
                  { st with s := s, q := q, levels := st.levels ++ [ch] }
                  { stC with s := sF, q := qF, levels := stC.levels ++ [ch] } :=
                h1₀.update _ _ (by show stC.level < stC.level + 1; omega) hreqle
              exact Lock.bind (g := twoOod A (lv'.logCols + lv'.lanes) (st.levels ++ [ch]).length s q)
                (g' := twoOodC A (lv'.logCols + lv'.lanes) (stC.levels ++ [ch]).length sF qF)
                (R := OodRel (AgreeZ r C₀ w₀ (Function.update i (stC.level + 1) (some (C', cap)))
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
                  have h2 : StRelZ r C₀ w₀ (Function.update i (stC.level + 1) (some (C', cap)))
                      { st with s := s2, q := q2, levels := st.levels ++ [ch], oods := st.oods ++ oods2 }
                      { stC with s := sF2, q := qF2, levels := stC.levels ++ [ch], oods := stC.oods ++ oods2 } :=
                    ⟨by simp [h.rel.levels], by simp [h.rel.oods], h.rel.cons, h.rel.prevLevel,
                      h1.prev, fun o ho => hsq2 o ho⟩
                  exact Lock.bind
                    (g := queryAndBatch A lv st.prev ch (st.levels ++ [ch]).length
                      { st with s := s2, q := q2, levels := st.levels ++ [ch], oods := st.oods ++ oods2 } true)
                    (g' := queryAndBatchC A r lv ch (stC.levels ++ [ch]).length
                      { stC with s := sF2, q := qF2, levels := stC.levels ++ [ch], oods := stC.oods ++ oods2 } true)
                    (R := QueryRelZ r C₀ w₀ lv ch.length
                      { stC with s := sF2, q := qF2, levels := stC.levels ++ [ch], oods := stC.oods ++ oods2 }
                      (Function.update i (stC.level + 1) (some (C', cap))))
                    (fun i₃ st3 stC3 hq => by
                      obtain ⟨hi3, hq'⟩ := hq
                      subst i₃
                      exact lockRecZ A r hr n b initial ext C₀ w₀ sch (lv' :: rest) _
                        { st3 with prev := C', prevLevel := lv' }
                        { stC3 with level := stC3.level + 1, caps := stC3.caps ++ [cap], prevLevel := lv' }
                        (by
                        obtain ⟨hrel, hlev, hcaps, -, pos, hreqs, hposlen, hposlt⟩ := hq'
                        simp only at hlev hcaps hreqs
                        have hlv := getD_of_drop h.levs
                        have hlenR := reqs_length h.reqs
                        refine ⟨⟨hrel.levels, hrel.oods, hrel.cons, rfl, ?_, hrel.sq⟩, ?_, ?_, ?_, ?_,
                          ?_, ?_, Nat.succ_ne_zero _⟩
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
                    (lockQueryZ A r C₀ w₀ _ lv ch (st.levels ++ [ch]).length (stC.levels ++ [ch]).length
                      (by rw [h.rel.levels]) _ _ h2 h.lev1 true))
                (lockTwoOod A r _ (AgreeZ r C₀ w₀ (Function.update i (stC.level + 1) (some (C', cap)))
                  stC.reqs) _ _ _ (by rw [h.rel.levels]) s q sF qF (fun o ho => h1.sq o ho)))
        (lockFold A r i (AgreeZ r C₀ w₀ i stC.reqs) lv.lanes st.s st.q stC.s stC.q h.rel.sq)

omit [Fintype F] [DecidableEq F] in
theorem RecRelZ.reject (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (sch : Schedule) (i : IRep F D) :
    RecRelZ r C₀ w₀ sch i false (PendL.reject : PendL F D) :=
  ⟨fun _ _ => rfl, rfl, by simp [PendL.reject], fun ℓ _ h => by simp [PendL.reject] at h,
    by simp [PendL.reject]⟩

/-! ## Level 0 -/

/-- **The masked lane rounds**, the same game on both sides, `k` challenges. -/
def lockFoldZK (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (r : ℕ) (i : IRep F D) :
    (k j : ℕ) → (s : LinK K np) → (q : QuadL K np) →
      Lock (Oracle F) (Cap D) (getRep r)
        (fun i' (a b : List K × LinK K np) => i' = i ∧ a = b ∧ a.1.length = k) i
        (foldRoundsZK A P k j s q) (foldRoundsZK A P k j s q)
  | 0, _, _, _ => .ret ⟨rfl, rfl, rfl⟩
  | 1, _, _, _ => .coin (Fin 2 → F) fun _ => .ret ⟨rfl, rfl, rfl⟩
  | k + 2, j, _, q => .coin (Fin 2 → F) fun w => .send (K × K) fun msg =>
      Lock.bind (R := fun i' (a b : List K × LinK K np) => i' = i ∧ a = b ∧ a.1.length = k + 1)
        (fun i' x y hxy => by
          obtain ⟨rfl, rfl, hl⟩ := hxy
          rcases x with ⟨rs, s''⟩
          exact .ret ⟨rfl, rfl, by simp only [List.length_cons] at hl ⊢; omega⟩)
        (lockFoldZK A P r i (k + 1) (j + 1) _ _)

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- On a row agreeing with `C` at `p` on its `w₀` lanes, the padded column from the row is the oracle's. -/
theorem padColumnO_eq (A : Arith F K) (d cols t : ℕ) (c₀ : F) (ch : List K) {e : ℕ} (ρ : Fin e → K)
    (xl : Fin e → ℕ) (ybar : Fin t → K) (row : List F) (C : Oracle F) (p w₀ : ℕ)
    (h : ∀ lane < w₀, row.getD lane 0 = C lane p) (hch : 2 ^ ch.length ≤ w₀) (hxl : ∀ i, xl i < w₀) :
    padColumnO A d cols t c₀ ch ρ xl ybar row p = padColumn A d cols t c₀ ch ρ xl ybar C p := by
  unfold padColumnO padColumn foldColumn
  congr 2
  · exact Finset.sum_congr rfl fun lane hl => by
      dsimp only
      rw [h lane ((Finset.mem_range.1 hl).trans_le hch)]
  · exact Finset.sum_congr rfl fun i _ => by rw [h _ (hxl i)]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- On agreeing rows, the batched value of a column function of the rows is the oracle column's. -/
theorem consValuePad_eq (A : Arith F K) (α : List F) (col : List F → ℕ → K) (v : ℕ → K) (rows : ℕ → List F)
    (pos : List ℕ) (h : ∀ j < pos.length, col (rows j) (pos.getD j 0) = v (pos.getD j 0)) :
    ((pos.zipIdx).map fun pj => A.embed (eqList α pj.2) * col (rows pj.2) pj.1).sum = consValueV A α v pos := by
  unfold consValueV
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
  rw [← hj.2, h j hj.1]

/-- **Level 0's queries in lockstep** (`queryAndBatchV` beside `queryAndBatchPadC`): on openings agreeing with `C₀` on
the level-0 request's `w₀` lanes, the compiled consistency value is the oracle's. -/
def lockQuery0Z (A : Arith F K) (r : ℕ) (C₀ : Oracle F) (w₀ : ℕ) (i : IRep F D) (lv : Level) (v : ℕ → K)
    (col : List F → ℕ → K) (hcol : ∀ (row : List F) (p : ℕ), (∀ lane < w₀, row.getD lane 0 = C₀ lane p) →
      col row p = v p)
    (st : LigState F K) (stC : LigStateC F K D) (h : StRelZ r C₀ w₀ i st stC) (hlev : stC.level = 0) :
    Lock (Oracle F) (Cap D) (getRep r) (QueryRelZ r C₀ w₀ lv lv.lanes stC i) i
      (queryAndBatchV A lv v 0 st) (queryAndBatchPadC A r lv col 0 stC) :=
  .coin (Fin lv.queries → F) fun w => .coin (Fin (ceilLog2 lv.queries) → F) fun α =>
    .send (K × K) fun msg => .coin F fun β => .ret (by
      have he : ∀ o, AgreeZ r C₀ w₀ i (stC.reqs ++ [⟨stC.level, positions A lv.queries lv.logLen w, lv.lanes,
          pathLen lv⟩]) o →
          ((positions A lv.queries lv.logLen w).zipIdx.map fun pj =>
            A.embed (eqList (List.ofFn α) pj.2) * col (o r stC.level pj.2).1 pj.1).sum =
          consValueV A (List.ofFn α) v (positions A lv.queries lv.logLen w) := fun o ho =>
        consValuePad_eq A _ col v _ _ fun j hj => hcol _ _ fun lane hl =>
          AgreeZ.last ho (by rw [hlev]; simp [tabAt]) j hj lane (by simpa [hlev] using hl)
      refine ⟨rfl, ⟨h.levels, h.oods, by simp [h.cons], h.prevLevel, h.prev, fun o ho => ?_⟩, rfl, rfl, rfl, _,
        rfl, positions_length A _ _ w, positions_lt A _ _ w⟩
      simp only
      rw [(h.sq o ho.of_append).1, (h.sq o ho.of_append).2, he o ho]
      exact ⟨rfl, rfl⟩)

/-- Rep `r`'s extra lanes lie in level 0's rows. -/
theorem extraLane_lt (k0 e r : ℕ) (hr : r < 2) (i : Fin e) : extraLane k0 e r i < 2 ^ k0 + 2 * e := by
  unfold extraLane
  have : e * r ≤ e := by
    calc e * r ≤ e * 1 := Nat.mul_le_mul_left e (by omega)
      _ = e := mul_one e
  have := i.isLt
  rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl <;> simp at * <;> omega

/-- **Levels 1+ of the masked level 0 in lockstep** (`ligTail` beside `ligTailC`). -/
def lockLigTailZ (A : Arith F K) (r : ℕ) (hr : r < 2) (sch : Schedule) (t : ℕ) (c₀ : F) (e : ℕ) (b : ℕ → F)
    (C₀ : Oracle F) (lv₀ : Level) (z₀ : Fin (sch.m - 7) → F) (β₀ : F) (initial : List K)
    (hinit : initial.length = sch.k0) (ρ : Fin e → K) (ybar : Fin t → K) (s : K) :
    (rest : List Level) → sch.levels = lv₀ :: rest →
      Lock (Oracle F) (Cap D) (getRep r) (RecRelZ r C₀ (2 ^ sch.k0 + 2 * e) sch) (fun _ => none)
        (ligTail A sch t c₀ e r b C₀ lv₀ rest z₀ β₀ initial ρ ybar s)
        (ligTailC A r sch t c₀ e b lv₀ rest z₀ β₀ initial ρ ybar s)
  | [], _ => .send (K × K) fun _ => .ret (RecRelZ.reject r C₀ _ sch _)
  | lv₁ :: rest, hlevs => .send (K × K) fun msg' =>
      .table (slotOf r 1) (fun t d => Function.update (fun _ => none : IRep F D) 1 (some (t, d)))
        (by rw [getRep_slot r hr _ _ le_rfl])
        (fun t d => by rw [getRep_slot r hr _ _ le_rfl]; simp)
        (fun t d s hs => getRep_update_ne r hr _ _ _ s hs)
        fun C₁ cap₁ =>
          Lock.bind (g := twoOod A (sch.m - 7 - sch.k0 + 1) 0 s (fromMsg msg' s))
            (g' := twoOodC A (sch.m - 7 - sch.k0 + 1) 0 (fun _ => s) (fun _ => fromMsg msg' s))
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
              have h2 : StRelZ r C₀ (2 ^ sch.k0 + 2 * e)
                  (Function.update (fun _ => none : IRep F D) 1 (some (C₁, cap₁)))
                  (⟨s2, q2, sch.m - 7 - sch.k0 + 1, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], C₀, lv₀⟩ :
                    LigState F K)
                  (⟨sF2, qF2, sch.m - 7 - sch.k0 + 1, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], 0, [], [],
                    lv₀⟩ : LigStateC F K D) :=
                ⟨rfl, rfl, rfl, rfl, by simp [tabAt], fun o _ => hsq2 o trivial⟩
              exact Lock.bind
                (g := queryAndBatchV A lv₀
                  (padColumn A lv₀.logLen lv₀.logCols t c₀ initial ρ (extraLane sch.k0 e r) ybar C₀) 0
                  ⟨s2, q2, sch.m - 7 - sch.k0 + 1, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], C₀, lv₀⟩)
                (g' := queryAndBatchPadC A r lv₀
                  (padColumnO A lv₀.logLen lv₀.logCols t c₀ initial ρ (extraLane sch.k0 e r) ybar) 0
                  ⟨sF2, qF2, sch.m - 7 - sch.k0 + 1, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], 0, [], [], lv₀⟩)
                (R := QueryRelZ r C₀ (2 ^ sch.k0 + 2 * e) lv₀ lv₀.lanes
                  ⟨sF2, qF2, sch.m - 7 - sch.k0 + 1, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], 0, [], [], lv₀⟩
                  (Function.update (fun _ => none : IRep F D) 1 (some (C₁, cap₁))))
                (fun i₃ st3 stC3 hq => by
                  obtain ⟨hi3, hq'⟩ := hq
                  subst i₃
                  exact lockRecZ A r hr (sch.m - 7) b initial sch.ext C₀ (2 ^ sch.k0 + 2 * e) sch
                    (lv₁ :: rest) _ { st3 with prev := C₁, prevLevel := lv₁ }
                    { stC3 with level := 1, caps := [cap₁], prevLevel := lv₁ } (by
                    obtain ⟨hrel, hlev, hcaps, -, pos, hreqs, hposlen, hposlt⟩ := hq'
                    simp only at hlev hcaps hreqs
                    refine ⟨⟨hrel.levels, hrel.oods, hrel.cons, rfl, by simp [tabAt], hrel.sq⟩,
                      by rw [hlevs]; rfl, ?_, ?_, ?_, rfl, ?_, one_ne_zero⟩
                    · simp [hreqs]
                    · intro req hreq
                      simp only [hreqs, List.nil_append, List.mem_singleton] at hreq
                      subst hreq
                      unfold ReqShape
                      exact ⟨by simpa [hlevs] using hposlen, by simpa [hlevs] using hposlt, by simp [hlevs],
                        by simp [hlevs]⟩
                    · intro ℓ hℓ
                      dsimp only at hℓ
                      rw [Function.update_of_ne (by omega)]
                    · intro ℓ hℓ1 hℓ2
                      dsimp only at hℓ2
                      obtain rfl : ℓ = 1 := by omega
                      exact ⟨C₁, cap₁, by simp, rfl⟩))
                (lockQuery0Z A r C₀ (2 ^ sch.k0 + 2 * e) _ lv₀ _ _
                  (fun row p hrow => padColumnO_eq A _ _ t c₀ initial ρ _ ybar row C₀ p _ hrow
                    (by rw [hinit]; omega) fun i => extraLane_lt sch.k0 e r hr i) _ _ h2 rfl))
            (lockTwoOod A r _ (fun _ => True) _ 0 0 rfl s (fromMsg msg' s) (fun _ => s) (fun _ => fromMsg msg' s)
              fun _ _ => ⟨rfl, rfl⟩)

/-- **C-Flock ZK's masked level 0 in lockstep** (`ligeritoPadZK` beside `ligeritoPadZC`): equal lane-phase rows, and on
openings agreeing with `C₀` and the fixed tables the compiled verdict is the oracle's. -/
def lockLigPadZ (A : Arith F K) (r : ℕ) (hr : r < 2) {np e : ℕ} (P : Pads0 np e) (t : ℕ) (c₀ : F) (T : F)
    (b : ℕ → F) (C₀ : Oracle F) :
    (sch : Schedule) → Lock (Oracle F) (Cap D) (getRep r)
      (fun i a b => a.1 = b.1 ∧ RecRelZ r C₀ (2 ^ sch.k0 + 2 * e) sch i a.2 b.2) (fun _ => none)
      (ligeritoPadZK A P sch t c₀ r T b C₀) (ligeritoPadZC A P sch t c₀ r T b)
  | ⟨_, _, [], _⟩ => .ret ⟨rfl, RecRelZ.reject r C₀ _ _ _⟩
  | ⟨m, k0, lv₀ :: rest, ext⟩ =>
      .coin (Fin (m - 7) → F) fun z₀ => .send F fun y₀ => .coin F fun β₀ => .send (K × K) fun msg =>
        Lock.bind (R := fun i' (a b : List K × LinK K np) => i' = (fun _ => none : IRep F D) ∧ a = b ∧
            a.1.length = k0)
          (fun i' x y hxy => by
            obtain ⟨rfl, rfl, hlen⟩ := hxy
            rcases x with ⟨initial, s⟩
            exact .send (Fin e → K) fun ev =>
              Lock.bind (g := drawKs A e) (g' := drawKs A e)
                (R := fun i' (a b : Fin e → K) => i' = (fun _ => none : IRep F D) ∧ a = b)
                (fun i'' ρ ρ' hρ => by
                  obtain ⟨rfl, rfl⟩ := hρ
                  exact .send K fun T' => .send (Fin t → K) fun ybar =>
                    Lock.bind
                      (g := ligTail A ⟨m, k0, lv₀ :: rest, ext⟩ t c₀ e r b C₀ lv₀ rest z₀ β₀ initial ρ ybar T')
                      (g' := ligTailC (D := D) A r ⟨m, k0, lv₀ :: rest, ext⟩ t c₀ e b lv₀ rest z₀ β₀ initial ρ ybar T')
                      (fun i3 acc pend h => .ret ⟨rfl, h⟩)
                      (lockLigTailZ A r hr ⟨m, k0, lv₀ :: rest, ext⟩ t c₀ e b C₀ lv₀ z₀ β₀ initial hlen ρ ybar T'
                        rest rfl))
                (Lock.same (drawKs A e)))
          (lockFoldZK A P r (fun _ => none) k0 0 _ _)

end FlockSoundness.Discharge.ZkLink
