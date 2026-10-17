import Proofs.Flock.Claims.Complete.Replay.LigeritoResidual
import Proofs.Flock.Claims.Complete.Replay.LigeritoRun

/-!
# The verifier's model steps on the honest prover's values

The verifier's model (`Lg.out`) reads every value from the proof at its cursors. When the value at a cursor is the one
the honest prover sends from its table and weight there, each model step keeps the model's claim the prover's sum
(`Sync`) and, at a recursive level, the prover's weight the model's residual weight at the model's own contexts
(`JPre`, from `WPre`):

* **challenges** (`chal_sync`, `chals_sync`, `chals_split_sync`): the model's challenges are the coin pairs of their
  rounds (`chs`, `chals_fst`), and each message the folded tables' round polynomial;
* **an out-of-domain intro** (`ood_sync`, `JPre.ood`) and **a level's query and close** (`close_sync`, `JPre.close`);
* **a level's four challenges** (`JPre.chals`): the fold, then the split (`WPre.fold`, `WAt.lift`).
-/

namespace FlockComplete.Replay

open Flock FlockSoundness FlockSoundness.Refine Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Lg

/-! ## Challenges -/

/-- The coin pairs of `j` challenges from `t`: rounds `t.k, …, t.k + j − 1`. -/
def chs (cf : ℕ → ℕ → F128) (t : Transcript) (j : ℕ) : List F256 :=
  (List.range j).map fun k => ⟨cf (t.k + k) 0, cf (t.k + k) 1⟩

theorem chs_succ (cf : ℕ → ℕ → F128) (t : Transcript) (j : ℕ) :
    chs cf t (j + 1) = chs cf t j ++ [⟨cf (t.k + j) 0, cf (t.k + j) 1⟩] := by
  simp [chs, List.range_succ]

theorem chs_length (cf : ℕ → ℕ → F128) (t : Transcript) (j : ℕ) : (chs cf t j).length = j := by simp [chs]

/-- The model's challenges are the coin pairs of their rounds. -/
theorem chals_fst (p : LigeritoProof) (cf : ℕ → ℕ → F128) (st : LState) (t : Transcript) :
    ∀ j, (chals p cf #[] st t j).1.toList = chs cf t j
  | 0 => rfl
  | j + 1 => by
    have hk := (chals_fields p cf #[] st t j).2.2.2.2.2.1
    simp only [chals, chal, Array.toList_push, chals_fst p cf st t j, chs_succ, hk]

theorem foldKs_snoc (l : List F256) (r : F256) (a : Array F256) : foldKs (l ++ [r]) a = foldK r (foldKs l a) := by
  simp [foldKs, List.foldl_append]

/-- **A challenge**: when the message at the model's cursor is the folded tables' round polynomial. -/
theorem chal_sync {p : LigeritoProof} {cf : ℕ → ℕ → F128} {st : LState} {t : Transcript} {tab w : Array F256}
    {e : ℕ} (h : Sync st.s st.q tab w (e + 1))
    (hm : p.messages[st.msg]! = roundPoly (foldK ⟨cf t.k 0, cf t.k 1⟩ tab) (foldK ⟨cf t.k 0, cf t.k 1⟩ w)) :
    Sync (chal p cf st t).2.1.s (chal p cf st t).2.1.q (foldK ⟨cf t.k 0, cf t.k 1⟩ tab)
      (foldK ⟨cf t.k 0, cf t.k 1⟩ w) e := by
  show Sync (st.q.eval _) (Poly.ofMsg p.messages[st.msg]! (st.q.eval _)) _ _ e
  rw [hm]
  exact h.chal _

/-- **A challenge that splits**: when the message is the split table's round polynomial against the lifted weight. -/
theorem chal_split_sync {p : LigeritoProof} {cf : ℕ → ℕ → F128} {st : LState} {t : Transcript} {tab w : Array F256}
    {e : ℕ} (h : Sync st.s st.q tab w e)
    (hm : p.messages[st.msg]! = roundPoly (embedF (splitK (foldK ⟨cf t.k 0, cf t.k 1⟩ tab)))
      (liftU (foldK ⟨cf t.k 0, cf t.k 1⟩ w))) :
    Sync (chal p cf st t).2.1.s (chal p cf st t).2.1.q (embedF (splitK (foldK ⟨cf t.k 0, cf t.k 1⟩ tab)))
      (liftU (foldK ⟨cf t.k 0, cf t.k 1⟩ w)) e := by
  show Sync (st.q.eval _) (Poly.ofMsg p.messages[st.msg]! (st.q.eval _)) _ _ e
  rw [hm]
  exact h.chalSplit _

/-- **`j` challenges**, each message the tables folded so far. -/
theorem chals_sync (p : LigeritoProof) (cf : ℕ → ℕ → F128) (st : LState) (t : Transcript) (tab w : Array F256) :
    ∀ j e : ℕ, Sync st.s st.q tab w (e + j) →
      (∀ k < j, p.messages[st.msg + k]! =
        roundPoly (foldKs (chs cf t (k + 1)) tab) (foldKs (chs cf t (k + 1)) w)) →
      Sync (chals p cf #[] st t j).2.1.s (chals p cf #[] st t j).2.1.q (foldKs (chs cf t j) tab)
        (foldKs (chs cf t j) w) e
  | 0, _, h, _ => h
  | j + 1, e, h, hm => by
    have ih := chals_sync p cf st t tab w j (e + 1) (by rwa [show e + 1 + j = e + (j + 1) by ring])
      fun k hk => hm k (by omega)
    obtain ⟨hmsg, -, -, -, -, hk, -⟩ := chals_fields p cf #[] st t j
    have := chal_sync (p := p) (cf := cf) (st := (chals p cf #[] st t j).2.1) (t := (chals p cf #[] st t j).2.2) ih
      (by rw [hmsg, hm j (by omega), hk, chs_succ, foldKs_snoc, foldKs_snoc])
    rw [hk] at this
    rw [chs_succ, foldKs_snoc, foldKs_snoc]
    exact this

/-- **`j` challenges, the last splitting**: the folded table split, the folded weight lifted. -/
theorem chals_split_sync (p : LigeritoProof) (cf : ℕ → ℕ → F128) (st : LState) (t : Transcript) (tab w : Array F256)
    (j e : ℕ) (h : Sync st.s st.q tab w (e + j))
    (hm : ∀ k < j, p.messages[st.msg + k]! =
      roundPoly (foldKs (chs cf t (k + 1)) tab) (foldKs (chs cf t (k + 1)) w))
    (hs : p.messages[st.msg + j]! =
      roundPoly (embedF (splitK (foldKs (chs cf t (j + 1)) tab))) (liftU (foldKs (chs cf t (j + 1)) w))) :
    Sync (chals p cf #[] st t (j + 1)).2.1.s (chals p cf #[] st t (j + 1)).2.1.q
      (embedF (splitK (foldKs (chs cf t (j + 1)) tab))) (liftU (foldKs (chs cf t (j + 1)) w)) e := by
  have ih := chals_sync p cf st t tab w j e h hm
  obtain ⟨hmsg, -, -, -, -, hk, -⟩ := chals_fields p cf #[] st t j
  have := chal_split_sync (p := p) (cf := cf) (st := (chals p cf #[] st t j).2.1)
    (t := (chals p cf #[] st t j).2.2) ih (by rw [hmsg, hs, hk, chs_succ, foldKs_snoc, foldKs_snoc])
  rw [hk] at this
  rw [chs_succ, foldKs_snoc, foldKs_snoc]
  exact this

/-! ## Out-of-domain intros, queries and closes -/

/-- **An out-of-domain intro** on the F-table `g`: when the value at the model's cursor is `g`'s extension at the
model's point and the message the intro's round polynomial. -/
theorem ood_sync {p : LigeritoProof} {cf : ℕ → ℕ → F128} {e split : ℕ} {st : LState} {t : Transcript}
    {g : Array F128} {w : Array F256} (h : Sync st.s st.q (embedF g) w e)
    (hy : p.oodValues[st.ood]! = mleF g (vec cf t (e + 1)))
    (hm : p.messages[st.msg]! = roundPoly (embedF g) (embedF (eqTable (vec cf t (e + 1))))) :
    Sync (ood p cf (e + 1) split st t).1.s (ood p cf (e + 1) split st t).1.q (embedF g)
      (addScaled w (embedF (eqTable (vec cf t (e + 1)))) (cf (t.k + 1) 0)) e := by
  show Sync (st.s + (F256.ofF p.oodValues[st.ood]!).smul (cf (t.k + 1) 0))
    (st.q.addScaled (Poly.ofMsg p.messages[st.msg]! (F256.ofF p.oodValues[st.ood]!)) (cf (t.k + 1) 0)) _ _ e
  rw [hy, hm]
  exact h.ood _ (by simp [vec]) _

/-- A level's query positions from `t`: its query coins are round `t.k`'s. -/
def qPos (cf : ℕ → ℕ → F128) (lv : Level) (t : Transcript) : Array ℕ :=
  Flock.positions lv ((Array.range lv.q).map (cf t.k))

/-- A level's consistency coins from `t`: round `t.k + 1`'s. -/
def qAlpha (cf : ℕ → ℕ → F128) (lv : Level) (t : Transcript) : Array F128 :=
  (Array.range (Nat.log2 (lv.q - 1) + 1)).map (cf (t.k + 1))

/-- **A level's query and close** on the split of the prover's fold of the level's F-table `g` at its challenges `ch`:
when the opened rows are `g`'s codeword rows at the model's positions and the message is the consistency claim's round
polynomial. -/
theorem close_sync {p : LigeritoProof} {cf : ℕ → ℕ → F128} {lv : Level} {rows : Array (Array F128)}
    {ch : Array F256} {cols start : ℕ} {st : LState} {t : Transcript} {g : Array F128} {w : Array F256} {e : ℕ}
    (hl : LevelSeam lv) (hch : ch.size = lv.lanes) (hg : g.size = 2 ^ (lv.lanes + lv.cols)) (he : lv.cols = e)
    (h : Sync st.s st.q (embedF (splitK (foldKs ch.toList (embedF g)))) w e)
    (hrows : rows = (qPos cf lv t).map fun q => (encodeRows lv g)[q]!)
    (hm : p.messages[st.msg]! = roundPoly (embedF (splitK (foldKs ch.toList (embedF g))))
      (liftU (embedF (consW lv.cols (qPos cf lv t) (qAlpha cf lv t))))) :
    Sync (close p cf cols start (query p cf lv rows ch st t).1 (query p cf lv rows ch st t).2.1
        (query p cf lv rows ch st t).2.2).1.s
      (close p cf cols start (query p cf lv rows ch st t).1 (query p cf lv rows ch st t).2.1
        (query p cf lv rows ch st t).2.2).1.q
      (embedF (splitK (foldKs ch.toList (embedF g))))
      (addScaled w (liftU (embedF (consW lv.cols (qPos cf lv t) (qAlpha cf lv t)))) (cf (t.k + 2) 0)) e := by
  have hq : lv.q = (qPos cf lv t).size := (positions_size lv hl _).symm
  show Sync (st.s + (consE rows lv.q (qAlpha cf lv t) ch).smul (cf (t.k + 2) 0))
    (st.q.addScaled (Poly.ofMsg p.messages[st.msg]! (consE rows lv.q (qAlpha cf lv t) ch)) (cf (t.k + 2) 0)) _ _ e
  subst hrows
  rw [hm, hq]
  exact Sync.close lv g _ _ ch hch hg (fun j hj => positions_lt_d hl ((Array.range lv.q).map (cf t.k)) _
    (by rw [getElem!_pos (qPos cf lv t) j hj]; exact Array.getElem_mem hj)) he h _

/-! ## The head -/

theorem sum_addScaled_gv (t w v : Array F256) (b : F128) {N : ℕ} (hw : w.size = N) :
    ∑ x ∈ Finset.range N, gv t x * gv (addScaled w v b) x =
      ∑ x ∈ Finset.range N, gv t x * gv w x +
        (∑ x ∈ Finset.range N, gv t x * gv v x) * FlockLevel3.embed (fOf b) := by
  rw [sum_addScaled]
  refine Finset.sum_congr rfl fun y hy => ?_
  rw [gv_addScaled _ _ _ (by rw [hw]; exact Finset.mem_range.1 hy)]

theorem gOf_dot (a b : Array F128) {N : ℕ} (ha : a.size = N) (hb : b.size = N) :
    gOf (F256.ofF (dot a b)) = ∑ x ∈ Finset.range N, gv (embedF a) x * gv (embedF b) x := by
  rw [gOf_ofF, fOf_dot a b N ha, map_sum,
    ← Fin.sum_univ_eq_sum_range (fun x => gv (embedF a) x * gv (embedF b) x) N]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [gv_embedF _ (by omega), gv_embedF _ (by omega), map_mul]

/-- **Level 0's out-of-domain sample and first message**: on the claim `Σ tb·b = T`, the claim after the sample is the
prover's on its table and weight, when the value read is `tb`'s extension at the model's point and the message the
round polynomial of the table against the weight. -/
theorem head_sync {p : LigeritoProof} {cf : ℕ → ℕ → F128} {e : ℕ} {T : F128} {cap : Array ByteArray}
    {t : Transcript} {tb b : Array F128} (htb : tb.size = 2 ^ (e + 1)) (hb : b.size = 2 ^ (e + 1))
    (hT : dot tb b = T) (hy : p.oodValues[0]! = mleF tb (vec cf t (e + 1)))
    (hm : p.messages[0]! = roundPoly (embedF tb)
      (addScaled (embedF b) (embedF (eqTable (vec cf t (e + 1)))) (cf (t.k + 1) 0))) :
    Sync (head p cf (e + 1) (init T) (start cap T t)).1.s (head p cf (e + 1) (init T) (start cap T t)).1.q
      (embedF tb) (addScaled (embedF b) (embedF (eqTable (vec cf t (e + 1)))) (cf (t.k + 1) 0)) e := by
  show Sync (F256.ofF T + (F256.ofF p.oodValues[0]!).smul (cf (t.k + 1) 0))
    (Poly.ofMsg p.messages[0]! (F256.ofF T + (F256.ofF p.oodValues[0]!).smul (cf (t.k + 1) 0))) _ _ e
  rw [hm, hy]
  refine Sync.ofMsg (by rw [embedF_size, htb]) (by rw [addScaled_size, embedF_size, hb]) ?_
  rw [gOf_add, gOf_smul, sum_addScaled_gv _ _ _ _ (by rw [embedF_size, hb]), ← hT, gOf_dot tb b htb hb,
    gOf_mleF tb _ e htb (by simp [vec])]

/-! ## The contexts' shape -/

theorem preOK_snocOod {initial : List 𝕂} {levels : List (List 𝕂)} {oods : List (Model.OodCtx 𝔽)}
    {cons : List (Model.ConsCtx 𝔽)} {ext : ℕ} (h : PreOK initial levels oods cons ext) (o : Model.OodCtx 𝔽)
    (hs : o.split = some levels.length) (hz : o.z.length = ext + 1) : PreOK initial levels (oods ++ [o]) cons ext :=
  ⟨fun o' ho => by
    rcases List.mem_append.1 ho with h' | h'
    · exact h.ood o' h'
    · rw [List.mem_singleton.1 h']
      exact Or.inl ⟨hs, hz⟩, h.cons⟩

theorem preOK_snocCons {initial : List 𝕂} {levels : List (List 𝕂)} {oods : List (Model.OodCtx 𝔽)}
    {cons : List (Model.ConsCtx 𝔽)} {ext : ℕ} (h : PreOK initial levels oods cons ext) (c : Model.ConsCtx 𝔽)
    (hc : c.start = levels.length) : PreOK initial levels oods (cons ++ [c]) ext :=
  ⟨h.ood, fun c' hc' => by
    rcases List.mem_append.1 hc' with h' | h'
    · exact h.cons c' h'
    · rw [List.mem_singleton.1 h', hc]⟩

theorem preOK_of_postOK' {initial : List 𝕂} {levels : List (List 𝕂)} {oods : List (Model.OodCtx 𝔽)}
    {cons : List (Model.ConsCtx 𝔽)} {ext : ℕ} (h : PostOK initial levels oods cons ext) :
    PreOK initial levels oods cons ext :=
  ⟨fun o ho => Or.inr (h.ood o ho), h.cons⟩

theorem cons_of_length {α : Type} {k : ℕ} : ∀ l : List α, l.length = k + 1 → ∃ c cs, l = c :: cs ∧ cs.length = k
  | [], h => absurd h (by simp)
  | c :: cs, h => ⟨c, cs, rfl, by simpa using h⟩

/-! ## The joint invariant -/

variable (bF : ℕ → 𝔽) (n : ℕ)

/-- **At a split table** of `2^(ext+1)` entries, the prover's F-table `g` and weight `w` against the model's state `st`
at the challenges `initial` and `levels`: the claim is the prover's (`Sync`), the weight the residual weight before the
next fold (`WPre`), the contexts have their shape (`PreOK`), and every consistency context has at most `2^|α|`
positions. -/
structure JPre (initial : List 𝕂) (levels : List (List 𝕂)) (st : LState) (g : Array F128) (w : Array F256)
    (ext : ℕ) : Prop where
  sync : Sync st.s st.q (embedF g) w ext
  wpre : WPre bF n w initial levels (st.oods.toList.map oodOf) (st.conss.toList.map consOf) ext
  ok : PreOK initial levels (st.oods.toList.map oodOf) (st.conss.toList.map consOf) ext
  pos : ∀ cs ∈ st.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size

variable {bF n}

/-- **An out-of-domain intro at this level** keeps `JPre`. -/
theorem JPre.oodStep {p : LigeritoProof} {cf : ℕ → ℕ → F128} {initial : List 𝕂} {levels : List (List 𝕂)}
    {st : LState} {t : Transcript} {g : Array F128} {w : Array F256} {ext split : ℕ}
    (h : JPre bF n initial levels st g w ext) (hs : split = levels.length)
    (hy : p.oodValues[st.ood]! = mleF g (vec cf t (ext + 1)))
    (hm : p.messages[st.msg]! = roundPoly (embedF g) (embedF (eqTable (vec cf t (ext + 1))))) :
    JPre bF n initial levels (ood p cf (ext + 1) split st t).1 g
      (addScaled w (embedF (eqTable (vec cf t (ext + 1)))) (cf (t.k + 1) 0)) ext := by
  obtain ⟨-, -, -, -, -, -, hc, ho⟩ := ood_fields p cf (ext + 1) split st t
  subst hs
  have hnew : (st.oods.push ⟨vec cf t (ext + 1), cf (t.k + 1) 0, some levels.length⟩).toList.map oodOf =
      st.oods.toList.map oodOf ++
        [⟨(vec cf t (ext + 1)).toList.map fOf, fOf (cf (t.k + 1) 0), some levels.length⟩] := by
    simp [oodOf]
  refine ⟨ood_sync h.sync hy hm, ?_, ?_, ?_⟩
  · rw [ho, hc, hnew]
    exact h.wpre.ood _ (by simp [vec]) _
  · rw [ho, hc, hnew]
    exact preOK_snocOod h.ok _ rfl (by simp [vec])
  · rw [hc]
    exact h.pos

/-- **A level's query and close** keep `JPre`. -/
theorem JPre.closeStep {p : LigeritoProof} {cf : ℕ → ℕ → F128} {initial : List 𝕂} {levels : List (List 𝕂)}
    {st : LState} {t : Transcript} {lv : Level} {rows : Array (Array F128)} {ch : Array F256} {gp : Array F128}
    {w : Array F256} {ext cols start : ℕ} (h : JPre bF n initial levels st (splitK (foldKs ch.toList (embedF gp))) w ext)
    (hl : LevelSeam lv) (hch : ch.size = lv.lanes) (hg : gp.size = 2 ^ (lv.lanes + lv.cols)) (he : lv.cols = ext)
    (hcols : cols = ext) (hs : start = levels.length)
    (hrows : rows = (qPos cf lv t).map fun q => (encodeRows lv gp)[q]!)
    (hm : p.messages[st.msg]! = roundPoly (embedF (splitK (foldKs ch.toList (embedF gp))))
      (liftU (embedF (consW lv.cols (qPos cf lv t) (qAlpha cf lv t))))) :
    JPre bF n initial levels (close p cf cols start (query p cf lv rows ch st t).1 (query p cf lv rows ch st t).2.1
        (query p cf lv rows ch st t).2.2).1 (splitK (foldKs ch.toList (embedF gp)))
      (addScaled w (liftU (embedF (consW ext (qPos cf lv t) (qAlpha cf lv t)))) (cf (t.k + 2) 0)) ext := by
  have hpa : (qPos cf lv t).size ≤ 2 ^ (qAlpha cf lv t).size := pos_le_alpha lv hl _ _ (by simp [qAlpha])
  have hcs : (close p cf cols start (query p cf lv rows ch st t).1 (query p cf lv rows ch st t).2.1
      (query p cf lv rows ch st t).2.2).1.conss =
        st.conss.push ⟨cols, qPos cf lv t, qAlpha cf lv t, cf (t.k + 2) 0, start⟩ := rfl
  have hos : (close p cf cols start (query p cf lv rows ch st t).1 (query p cf lv rows ch st t).2.1
      (query p cf lv rows ch st t).2.2).1.oods = st.oods := rfl
  have hsy := close_sync (rows := rows) (cols := cols) (start := start) hl hch hg he h.sync hrows hm
  rw [he] at hsy
  subst hcols hs
  refine ⟨hsy, ?_, ?_, ?_⟩
  · rw [hcs, hos, Array.toList_push, List.map_append, List.map_singleton]
    exact h.wpre.cons _ _ hpa _ rfl
  · rw [hcs, hos, Array.toList_push, List.map_append, List.map_singleton]
    exact preOK_snocCons h.ok _ rfl
  · rw [hcs, Array.toList_push]
    intro cs hc
    rcases List.mem_append.1 hc with hc | hc
    · exact h.pos cs hc
    · rw [List.mem_singleton.1 hc]
      exact hpa

/-- **A level's four challenges fold the weight** to the next stage's `resAt`, and the contexts keep their shape. -/
theorem JPre.foldAt {cf : ℕ → ℕ → F128} {initial : List 𝕂} {levels : List (List 𝕂)} {st : LState}
    {t : Transcript} {g : Array F128} {w : Array F256} {ext : ℕ} (h : JPre bF n initial levels st g w (ext + 3)) :
    WAt bF n (foldKs (chs cf t 4) w) initial (levels ++ [(chs cf t 4).map gOf]) (st.oods.toList.map oodOf)
        (st.conss.toList.map consOf) ext ∧
      PostOK initial (levels ++ [(chs cf t 4).map gOf]) (st.oods.toList.map oodOf) (st.conss.toList.map consOf)
        ext := by
  have hw := h.wpre.fold h.ok (chs cf t 4).toArray (by simp [chs_length])
  rw [List.toList_toArray] at hw
  obtain ⟨c, cs, hcs, hlen⟩ := cons_of_length (k := 3) ((chs cf t 4).map gOf) (by simp [chs_length])
  rw [hcs] at hw ⊢
  exact ⟨hw, postOK_of_preOK _ _ _ _ _ c cs ext (by omega) h.ok⟩

/-- **A level's four challenges**, the last splitting, keep `JPre` at the next level. -/
theorem JPre.chalsStep {p : LigeritoProof} {cf : ℕ → ℕ → F128} {initial : List 𝕂} {levels : List (List 𝕂)}
    {st : LState} {t : Transcript} {g : Array F128} {w : Array F256} {ext : ℕ}
    (h : JPre bF n initial levels st g w (ext + 3))
    (hm : ∀ k < 3, p.messages[st.msg + k]! =
      roundPoly (foldKs (chs cf t (k + 1)) (embedF g)) (foldKs (chs cf t (k + 1)) w))
    (hs : p.messages[st.msg + 3]! =
      roundPoly (embedF (splitK (foldKs (chs cf t 4) (embedF g)))) (liftU (foldKs (chs cf t 4) w))) :
    JPre bF n initial (levels ++ [(chs cf t 4).map gOf]) (chals p cf #[] st t 4).2.1
      (splitK (foldKs (chs cf t 4) (embedF g))) (liftU (foldKs (chs cf t 4) w)) ext := by
  obtain ⟨hw, hpost⟩ := h.foldAt (cf := cf) (t := t)
  obtain ⟨-, -, -, -, -, -, -, ho, hc⟩ := chals_fields p cf #[] st t 4
  refine ⟨chals_split_sync p cf st t _ _ 3 ext h.sync hm hs, ?_, ?_, ?_⟩
  · rw [ho, hc]
    refine hw.lift fun o hmem => ?_
    rcases (hpost.ood o hmem).1 with h' | ⟨a, ha, h'⟩
    · simp [h']
    · rw [h']
      intro he
      simp only [Option.some.injEq] at he
      omega
  · rw [ho, hc]
    exact preOK_of_postOK' hpost
  · rw [hc]
    exact h.pos

/-! ## Level 0 -/

/-- **Level 0's challenges** append their coin pairs to the original point. -/
theorem WAt.chals0 {w : Array F256} {initial : List 𝕂} {oods : List (Model.OodCtx 𝔽)} (cf : ℕ → ℕ → F128)
    (t : Transcript) : ∀ j ext, WAt bF n w initial [] oods [] (ext + j) → Orig initial (ext + j) oods →
      WAt bF n (foldKs (chs cf t j) w) (initial ++ (chs cf t j).map gOf) [] oods [] ext ∧
        Orig (initial ++ (chs cf t j).map gOf) ext oods
  | 0, ext, h, ho => by simpa [chs, foldKs] using And.intro h ho
  | j + 1, ext, h, ho => by
    obtain ⟨h1, ho1⟩ := WAt.chals0 cf t j (ext + 1) (by rwa [show ext + 1 + j = ext + (j + 1) by ring])
      (by rwa [show ext + 1 + j = ext + (j + 1) by ring])
    obtain ⟨h2, ho2⟩ := h1.chal0 ho1 ⟨cf (t.k + j) 0, cf (t.k + j) 1⟩
    rw [chs_succ, foldKs_snoc, List.map_append, List.map_singleton, ← List.append_assoc]
    exact ⟨h2, ho2⟩

/-- **Level 0 up to its split**: after the out-of-domain sample and the `k0 = j + 1` challenges, the last splitting,
`JPre` holds at the split table, with the challenges as the original point and no level yet. -/
theorem JPre.pre0 {p : LigeritoProof} {cf : ℕ → ℕ → F128} {T : F128} {cap : Array ByteArray} {t : Transcript}
    {tb b : Array F128} {nn k0 e j : ℕ} (hn : nn = e + j + 1) (hk : k0 = j + 1) (htb : tb.size = 2 ^ nn)
    (hb : b.size = 2 ^ nn) (hT : dot tb b = T) (hy : p.oodValues[0]! = mleF tb (vec cf t nn))
    (hm0 : p.messages[0]! = roundPoly (embedF tb)
      (addScaled (embedF b) (embedF (eqTable (vec cf t nn))) (cf (t.k + 1) 0)))
    (hm : ∀ k < j, p.messages[(head p cf nn (init T) (start cap T t)).1.msg + k]! =
      roundPoly (foldKs (chs cf (head p cf nn (init T) (start cap T t)).2 (k + 1)) (embedF tb))
        (foldKs (chs cf (head p cf nn (init T) (start cap T t)).2 (k + 1))
          (addScaled (embedF b) (embedF (eqTable (vec cf t nn))) (cf (t.k + 1) 0))))
    (hs : p.messages[(head p cf nn (init T) (start cap T t)).1.msg + j]! =
      roundPoly (embedF (splitK (foldKs (chs cf (head p cf nn (init T) (start cap T t)).2 k0) (embedF tb))))
        (liftU (foldKs (chs cf (head p cf nn (init T) (start cap T t)).2 k0)
          (addScaled (embedF b) (embedF (eqTable (vec cf t nn))) (cf (t.k + 1) 0))))) :
    JPre (fun w => fOf b[w]!) nn ((chs cf (head p cf nn (init T) (start cap T t)).2 k0).map gOf) []
      (chals p cf #[] (head p cf nn (init T) (start cap T t)).1 (head p cf nn (init T) (start cap T t)).2 k0).2.1
      (splitK (foldKs (chs cf (head p cf nn (init T) (start cap T t)).2 k0) (embedF tb)))
      (liftU (foldKs (chs cf (head p cf nn (init T) (start cap T t)).2 k0)
        (addScaled (embedF b) (embedF (eqTable (vec cf t nn))) (cf (t.k + 1) 0)))) e := by
  subst hn hk
  have hS := head_sync (p := p) (cf := cf) (cap := cap) (e := e + j) htb hb hT hy hm0
  obtain ⟨hW, ho⟩ := (WAt.start b hb).oodHead (fun o h => by simp at h) rfl (vec cf t (e + j + 1)) (by simp [vec])
    (cf (t.k + 1) 0)
  rw [List.nil_append] at hW ho
  obtain ⟨hW2, ho2⟩ := WAt.chals0 cf (head p cf (e + j + 1) (init T) (start cap T t)).2 (j + 1) e
    (by rwa [show e + (j + 1) = e + j + 1 by ring]) (by rwa [show e + (j + 1) = e + j + 1 by ring])
  rw [List.nil_append] at hW2 ho2
  obtain ⟨-, -, -, -, -, -, -, hco, hcc⟩ := chals_fields p cf #[] (head p cf (e + j + 1) (init T) (start cap T t)).1
    (head p cf (e + j + 1) (init T) (start cap T t)).2 (j + 1)
  have hoods : (head p cf (e + j + 1) (init T) (start cap T t)).1.oods.toList.map oodOf =
      [⟨(vec cf t (e + j + 1)).toList.map fOf, fOf (cf (t.k + 1) 0), none⟩] := by
    simp [head, oodOf, init, vec, start]
  have hconss : (head p cf (e + j + 1) (init T) (start cap T t)).1.conss = #[] := rfl
  refine ⟨chals_split_sync p cf _ _ _ _ j e hS hm hs, ?_, ?_, ?_⟩
  · rw [hco, hcc, hoods, hconss]
    exact hW2.lift fun o hmem => by simp at hmem; simp [hmem]
  · rw [hco, hcc, hoods, hconss]
    refine ⟨fun o hmem => Or.inr ⟨Or.inl ?_, ?_⟩, fun c hc => by simp at hc⟩
    · simp at hmem; simp [hmem]
    · obtain ⟨-, hz⟩ := ho2 o hmem
      simp at hmem
      rw [hz, hmem]
      simp [FlockSoundness.oodFixed, origOf, chs_length]
  · rw [hcc, hconss]
    simp

/-! ## The last level -/

/-- **The last level gives the final identity**: from `JPre` before its four challenges, when the messages are the
folded tables' round polynomials, the last query opens the codeword rows of the level's F-table `g`, `y_r` is the split
of `g`'s fold, and the final state has the final check's dimensions, the final check holds at the model's final state
`F` and levels. -/
theorem JPre.final {p : LigeritoProof} {cf : ℕ → ℕ → F128} {basisAt : Array F256 → ℕ → Array F256}
    {initCh : Array F256} {levels : Array (Array F256)} {st : LState} {t tq : Transcript} {lv : Level}
    {g : Array F128} {w : Array F256} {ext start : ℕ} {c : Array F256 × LState × Transcript} {F : LState}
    (h : JPre bF n (initCh.toList.map gOf) (levelsOf levels) st g w (ext + 3))
    (hm : ∀ k < 3, p.messages[st.msg + k]! =
      roundPoly (foldKs (chs cf t (k + 1)) (embedF g)) (foldKs (chs cf t (k + 1)) w))
    (hs : p.messages[st.msg + 3]! =
      roundPoly (embedF (splitK (foldKs (chs cf t 4) (embedF g)))) (liftU (foldKs (chs cf t 4) w)))
    (hc : c = chals p cf #[] st t 4)
    (hF : F = (closeLast p cf ext start (query p cf lv p.finalProof.rows c.1 c.2.1 tq).1
      (query p cf lv p.finalProof.rows c.1 c.2.1 tq).2.1 (query p cf lv p.finalProof.rows c.1 c.2.1 tq).2.2).1)
    (hl : LevelSeam lv) (hlanes : lv.lanes = 4) (hg : g.size = 2 ^ (lv.lanes + lv.cols)) (he : lv.cols = ext)
    (hrows : p.finalProof.rows = (qPos cf lv tq).map fun q => (encodeRows lv g)[q]!)
    (hyr : p.finalProof.yr = splitK (foldKs (chs cf t 4) (embedF g))) (hstart : start = levels.size + 1)
    (hl4 : ∀ l ∈ (levels.push c.1).toList, l.size = 4) (horig : (origE initCh (levels.push c.1)).size + ext = n)
    (hbasis : ∀ y < 2 ^ ext, gOf (basisAt (origE initCh (levels.push c.1)) ext)[y]! =
      Model.bHat execArith n bF ((origE initCh (levels.push c.1)).toList.map gOf ++ Model.bitsK ext y))
    (hood : ∀ o ∈ F.oods.toList,
      o.z.size = (Refine.oodFixed (levels.push c.1) (origE initCh (levels.push c.1)) o).1.size + ext)
    (hcons : ∀ cs ∈ F.conss.toList, (Refine.consFixed (levels.push c.1) cs.start).size + ext = cs.cols) :
    lhsE p.finalProof.yr ext (resE basisAt n initCh (levels.push c.1) ext F) = F.s := by
  subst hc
  obtain ⟨hw, -⟩ := h.foldAt (cf := cf) (t := t)
  have hc1 : (chals p cf #[] st t 4).1.toList = chs cf t 4 := chals_fst p cf st t 4
  have hch : (chals p cf #[] st t 4).1.size = lv.lanes := by
    rw [hlanes, ← Array.length_toList, hc1, chs_length]
  have hlev : levelsOf (levels.push (chals p cf #[] st t 4).1) = levelsOf levels ++ [(chs cf t 4).map gOf] := by
    simp [levelsOf, hc1]
  rw [← hlev] at hw
  have hS : Sync (chals p cf #[] st t 4).2.1.s (chals p cf #[] st t 4).2.1.q
      (embedF (splitK (foldKs (chs cf t 4) (embedF g)))) (liftU (foldKs (chs cf t 4) w)) ext :=
    chals_split_sync p cf st t (embedF g) w 3 ext h.sync hm hs
  have hf : (foldKs (chs cf t 4) (embedF g)).size = 2 ^ ext := by
    rw [foldKs_size, embedF_size, hg, chs_length, hlanes, he, pow_add, Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]
  obtain ⟨-, -, -, -, -, -, -, ho, hcs⟩ := chals_fields p cf #[] st t 4
  have hFc : F.conss = st.conss.push ⟨ext, qPos cf lv tq, qAlpha cf lv tq, cf (tq.k + 2) 0, start⟩ := by
    rw [hF, ← hcs]
    rfl
  have hFo : F.oods = st.oods := by
    rw [hF, ← ho]
    rfl
  have hpa : (qPos cf lv tq).size ≤ 2 ^ (qAlpha cf lv tq).size := pos_le_alpha lv hl _ _ (by simp [qAlpha])
  have hpos : ∀ cs ∈ F.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size := by
    rw [hFc, Array.toList_push]
    intro cs hc
    rcases List.mem_append.1 hc with hc | hc
    · exact h.pos cs hc
    · rw [List.mem_singleton.1 hc]
      exact hpa
  have hsum : gOf (chals p cf #[] st t 4).2.1.s =
      ∑ x ∈ Finset.range (2 ^ ext), gv (foldKs (chs cf t 4) (embedF g)) x * gv (foldKs (chs cf t 4) w) x := by
    rw [hS.sum, sum_split _ _ ext hf hw.size]
  have hval : gOf (consE p.finalProof.rows lv.q (qAlpha cf lv tq) (chals p cf #[] st t 4).1) =
      ∑ x ∈ Finset.range (2 ^ ext), gv (foldKs (chs cf t 4) (embedF g)) x *
        gv (embedF (consW ext (qPos cf lv tq) (qAlpha cf lv tq))) x := by
    rw [hrows, show lv.q = (qPos cf lv tq).size from (positions_size lv hl _).symm,
      consE_fold lv g _ _ _ hch hg (fun j hj => positions_lt_d hl ((Array.range lv.q).map (cf tq.k)) _
        (by rw [getElem!_pos (qPos cf lv tq) j hj]; exact Array.getElem_mem hj)), hc1, he]
  have hF' : F.s = (chals p cf #[] st t 4).2.1.s +
      (consE p.finalProof.rows lv.q (qAlpha cf lv tq) (chals p cf #[] st t 4).1).smul (cf (tq.k + 2) 0) := by
    rw [hF]
    rfl
  rw [hyr, hF']
  exact lhsE_resE (cs₀ := st.conss.toList)
    (cc := ⟨ext, qPos cf lv tq, qAlpha cf lv tq, cf (tq.k + 2) 0, start⟩) hl4 horig hpos hbasis hood hcons hw
    (by rw [hFo]) (by rw [hFc, Array.toList_push]) rfl rfl (by simp [hstart]) hf hsum hval

end Lg

end FlockComplete.Replay
