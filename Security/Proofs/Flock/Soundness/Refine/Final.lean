import Proofs.Flock.Soundness.Refine.LigeritoRun

/-!
# The final check refines the model's (PROTOCOL.md §13.6)

`final_refines`: when the executable's final check (`finalProg`, the verifier's last step) passes, the model's
`finalCheck` holds on the images of its state, given that the executable's basis at the original point is the model's
`bHat` there (`hbasis`); and the final check touches no rounds.

The two compute the same sum term by term:

* the subspace polynomials: the executable's `whats` is the model's `What` at `omega` (`whats_get`);
* the executable's out-of-domain and consistency loops are folds (`forIn_yield_fold`), whose words are the model's
  sums (`oodFold_get`, `consFold_get`).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## The subspace polynomials (§13.4) -/

theorem whatsGo_get (sv : Array F128) (ω : F128) : ∀ (fuel i : ℕ) (s : F128) (out : Array F128),
    (∀ j < i + fuel, fOf sv[j]! = (sPoly j).eval (beta j)) → fOf s = (sPoly i).eval (fOf ω) →
    (whatsGo sv fuel i s out).size = out.size + fuel ∧
    (∀ k < out.size, (whatsGo sv fuel i s out)[k]! = out[k]!) ∧
    ∀ k < fuel, fOf (whatsGo sv fuel i s out)[out.size + k]! = What (i + k) (fOf ω)
  | 0, i, s, out, _, _ => ⟨rfl, fun _ _ => rfl, fun _ hk => absurd hk (Nat.not_lt_zero _)⟩
  | fuel + 1, i, s, out, hsv, hs => by
      have hs' : fOf (s * s + sv[i]! * s) = (sPoly (i + 1)).eval (fOf ω) := by
        rw [fOf_add, fOf_mul, fOf_mul, hs, hsv i (by omega)]
        simp only [sPoly, Polynomial.eval_add, Polynomial.eval_pow, Polynomial.eval_mul, Polynomial.eval_C]
        ring
      obtain ⟨h1, h2, h3⟩ := whatsGo_get sv ω fuel (i + 1) (s * s + sv[i]! * s) (out.push (s * sv[i]!.inv))
        (fun j hj => hsv j (by omega)) hs'
      simp only [whatsGo]
      refine ⟨by rw [h1, Array.size_push]; omega, fun k hk => ?_, fun k hk => ?_⟩
      · rw [h2 k (by simp; omega), getElem!_pos _ k (by simp; omega), getElem!_pos out k hk, Array.getElem_push_lt]
      · rcases k with _ | k
        · simp only [Nat.add_zero]
          rw [h2 out.size (by simp), getElem!_pos _ out.size (by simp), Array.getElem_push_eq,
            fOf_mul, fOf_inv, hs, hsv i (by omega)]
          rfl
        · have := h3 k (by omega)
          rw [Array.size_push, show out.size + 1 + k = out.size + (k + 1) by omega,
            show i + 1 + k = i + (k + 1) by omega] at this
          exact this

/-- **The subspace polynomials** (§13.4): the executable's `whats (svTable m) q n` at `k < n ≤ m` is the model's
`Ŵ_k(ω_q)`. -/
theorem whats_get (m q n k : ℕ) (hn : n ≤ m) (hk : k < n) :
    fOf (whats (svTable m) q n)[k]! = What k (FlockLevel3.omega q) := by
  have := (whatsGo_get (svTable m) ⟨q.toUInt64, 0⟩ n 0 ⟨q.toUInt64, 0⟩ #[]
    (fun j hj => svTable_eval m j (by omega)) (by simp [sPoly])).2.2 k hk
  rw [Nat.zero_add, Array.size_empty, Nat.zero_add] at this
  exact this

/-! ## The executable's final check, as folds -/

/-- The original point (§13.6): the level-0 challenges, then each later level's challenges after its first. -/
def origE (initCh : Array F256) (levels : Array (Array F256)) : Array F256 :=
  levels.foldl (fun acc l => acc ++ l.extract 1 l.size) initCh

/-- `Π_{s ≥ a} cff(levels[s][0])`, as the executable folds it. -/
def prodFromE (levels : Array (Array F256)) (a : ℕ) : F256 :=
  ((levels.map fun l => l[0]!).extract a (levels.map fun l => l[0]!).size).foldl (fun acc c => acc * Flock.cff c) 1

/-- An out-of-domain sample's fixed coordinates and scale. -/
def oodFixed (levels : Array (Array F256)) (orig : Array F256) (o : Flock.OodCtx) : Array F256 × F256 :=
  match o.split with
  | none => (orig, prodFromE levels 0)
  | some s => ((levels.extract s levels.size).foldl (fun acc l => if acc.isEmpty then l else acc ++ l.extract 1 l.size)
      #[], prodFromE levels (s + 1))

/-- One out-of-domain sample's contribution to the residual weights. -/
def oodStep (ext : ℕ) (fs : Array F256 × F256) (o : Flock.OodCtx) (res : Array F256) : Array F256 :=
  (Array.range (2 ^ ext)).map fun y =>
    res[y]! + F256.smul (F256.smul ((Array.range fs.1.size).foldl
        (fun acc t => acc * ((1 : F256) + F256.ofF o.z[t]! + fs.1[t]!)) (1 : F256) * fs.2) o.beta)
      ((Array.range ext).foldl (fun acc j =>
        acc * (if (y >>> j) % 2 == 1 then o.z[fs.1.size + j]! else (1 : F128) + o.z[fs.1.size + j]!)) (1 : F128))

/-- A consistency context's fixed coordinates. -/
def consFixed (levels : Array (Array F256)) (start : ℕ) : Array F256 :=
  (levels.extract start levels.size).foldl (fun acc l => acc ++ l.extract 1 l.size) #[]

/-- A consistency context's per-residual sum over its queries. -/
def consAdd (sv : Array F128) (ext : ℕ) (fixed : Array F256) (cs : Flock.ConsCtx) : Array F256 :=
  (List.range' 0 cs.pos.size).foldl (fun add j =>
    (Array.range (2 ^ ext)).map fun y =>
      add[y]! + F256.smul (F256.smul ((Array.range fixed.size).foldl (fun acc i =>
          acc * ((1 : F256) + fixed[i]! * F256.ofF ((1 : F128) + (Flock.whats sv cs.pos[j]! cs.cols)[i]!))) (1 : F256))
          ((Flock.eqTable cs.alpha)[j]!)) ((Array.range ext).foldl (fun acc i =>
            if (y >>> i) % 2 == 1 then acc * (Flock.whats sv cs.pos[j]! cs.cols)[fixed.size + i]! else acc) (1 : F128)))
    (Array.replicate (2 ^ ext) 0)

/-- One consistency context's contribution to the residual weights. -/
def consStep (sv : Array F128) (ext : ℕ) (levels : Array (Array F256)) (res : Array F256) (cs : Flock.ConsCtx) :
    Array F256 :=
  (Array.range (2 ^ ext)).map fun y =>
    res[y]! + ((consAdd sv ext (consFixed levels cs.start) cs)[y]! * prodFromE levels cs.start).smul cs.beta

/-- The residual weights after both loops. -/
def resE (basisAt : Array F256 → ℕ → Array F256) (n : ℕ) (initCh : Array F256) (levels : Array (Array F256))
    (ext : ℕ) (st : LState) : Array F256 :=
  st.conss.toList.foldl (consStep (svTable (n + 1)) ext levels)
    (st.oods.toList.foldl (fun res o => oodStep ext (oodFixed levels (origE initCh levels) o) o res)
      ((basisAt (origE initCh levels) ext).map (· * prodFromE levels 0)))

/-- The final check's left-hand side, `Σ_y res[y]·(yr[2y] + yr[2y + 1]·u)`. -/
def lhsE (yr : Array F128) (ext : ℕ) (res : Array F256) : F256 :=
  (Array.range (2 ^ ext)).foldl (fun acc y => acc + res[y]! * ⟨yr[2 * y]!, yr[2 * y + 1]!⟩) (0 : F256)

/-- A loop whose every step yields a function of the step, touching neither state, is a fold. -/
theorem forIn_yield_fold {α σ : Type} (body : α → σ → LM (ForInStep σ)) (g : σ → α → σ) (P : α → Prop)
    (hb : ∀ a s st tr b st' tr', body a s st tr = .ok ((b, st'), tr') →
      b = .yield (g s a) ∧ st' = st ∧ tr' = tr ∧ P a) :
    ∀ (l : List α) (s : σ) (st st' : LState) (tr tr' : Transcript) (out : σ),
      forIn l s body st tr = .ok ((out, st'), tr') → out = l.foldl g s ∧ st' = st ∧ tr' = tr ∧ ∀ a ∈ l, P a
  | [], s, st, st', tr, tr', out, h => by
      simp only [List.forIn_nil] at h
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
      exact ⟨rfl, rfl, rfl, fun _ h => absurd h (List.not_mem_nil)⟩
  | a :: l, s, st, st', tr, tr', out, h => by
      rw [List.forIn_cons] at h
      obtain ⟨b, st₁, tr₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨rfl, rfl, rfl, hP⟩ := hb a s st tr b st₁ tr₁ h₁
      obtain ⟨rfl, rfl, rfl, hl⟩ := forIn_yield_fold body g P hb l (g s a) _ _ _ _ out h₂
      exact ⟨rfl, rfl, rfl, fun x hx => (List.mem_cons.1 hx).elim (fun e => e ▸ hP) (hl x)⟩


theorem finalProg_ok {p : LigeritoProof} {basisAt : Array F256 → ℕ → Array F256} {n : ℕ} {initCh : Array F256}
    {levels : Array (Array F256)} {ext : ℕ} {st st' : LState} {tr tr' : Transcript}
    (h : finalProg p basisAt n initCh levels ext st tr = .ok (((), st'), tr')) :
    st' = st ∧ tr' = tr ∧ (origE initCh levels).size + ext = n ∧
      (basisAt (origE initCh levels) ext).size = 2 ^ ext ∧
      (∀ o ∈ st.oods.toList, o.z.size = (oodFixed levels (origE initCh levels) o).1.size + ext) ∧
      (∀ cs ∈ st.conss.toList, (consFixed levels cs.start).size + ext = cs.cols) ∧
      lhsE p.finalProof.yr ext (resE basisAt n initCh levels ext st) = st.s := by
  unfold finalProg at h
  have h1 := lget_bind h
  clear h
  dsimp only at h1
  change ((if ((origE initCh levels).size + ext != n) = true then _ else _ : LM Unit) st tr) = _ at h1
  by_cases c1 : ((origE initCh levels).size + ext != n) = true
  · rw [ite_eq_left c1] at h1; obtain ⟨_, _, _, h0, _⟩ := lbind_ok h1; cases h0
  rw [ite_eq_right c1] at h1
  change ((if ((basisAt (origE initCh levels) ext).size != 2 ^ ext) = true then _ else _ : LM Unit) st tr) = _ at h1
  by_cases c2 : ((basisAt (origE initCh levels) ext).size != 2 ^ ext) = true
  · rw [ite_eq_left c2] at h1; obtain ⟨_, _, _, h0, _⟩ := lbind_ok h1; cases h0
  rw [ite_eq_right c2] at h1
  obtain ⟨res₁, st₁, tr₁, hl₁, h2⟩ := lbind_ok h1
  clear h1
  rw [← Array.forIn_toList] at hl₁
  obtain ⟨hres₁, rfl, rfl, hood⟩ := forIn_yield_fold _
    (fun res o => oodStep ext (oodFixed levels (origE initCh levels) o) o res)
    (fun o => o.z.size = (oodFixed levels (origE initCh levels) o).1.size + ext) (by
      intro o s st tr b st' tr' hb
      change ((if (o.z.size != (oodFixed levels (origE initCh levels) o).1.size + ext) = true then _ else _ :
        LM (ForInStep (Array F256))) st tr) = _ at hb
      by_cases cz : (o.z.size != (oodFixed levels (origE initCh levels) o).1.size + ext) = true
      · rw [ite_eq_left cz] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
      rw [ite_eq_right cz] at hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      simp only [bne_iff_ne, ne_eq, not_not] at cz
      exact ⟨rfl, rfl, rfl, cz⟩) _ _ _ _ _ _ _ hl₁
  obtain ⟨res₂, st₂, tr₂, hl₂, h3⟩ := lbind_ok h2
  clear h2
  rw [← Array.forIn_toList] at hl₂
  obtain ⟨hres₂, rfl, rfl, hcons⟩ := forIn_yield_fold _ (consStep (svTable (n + 1)) ext levels)
    (fun cs => (consFixed levels cs.start).size + ext = cs.cols) (by
      intro cs s st tr b st' tr' hb
      change ((if ((consFixed levels cs.start).size + ext != cs.cols) = true then _ else _ :
        LM (ForInStep (Array F256))) st tr) = _ at hb
      by_cases cz : ((consFixed levels cs.start).size + ext != cs.cols) = true
      · rw [ite_eq_left cz] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
      rw [ite_eq_right cz] at hb
      obtain ⟨add, st₃, tr₃, hadd, hb'⟩ := lbind_ok hb
      clear hb
      rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hadd
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hadd
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb'
      simp only [bne_iff_ne, ne_eq, not_not] at cz
      exact ⟨rfl, rfl, rfl, cz⟩) _ _ _ _ _ _ _ hl₂
  change ((if (lhsE p.finalProof.yr ext res₂ != st₂.s) = true then _ else _ : LM Unit) st₂ tr₂) = _ at h3
  by_cases c3 : (lhsE p.finalProof.yr ext res₂ != st₂.s) = true
  · rw [ite_eq_left c3] at h3; obtain ⟨_, _, _, h0, _⟩ := lbind_ok h3; cases h0
  rw [ite_eq_right c3] at h3
  by_cases c4 : (st₂.ood != p.oodValues.size || st₂.qn != p.queryNonces.size ||
      st₂.cn != p.consistencyNonces.size || st₂.cbn != p.claimNonces.size) = true
  · rw [ite_eq_left c4] at h3; cases h3
  rw [ite_eq_right c4] at h3
  obtain ⟨-, rfl, rfl⟩ := lpure_ok h3
  simp only [bne_iff_ne, ne_eq, not_not] at c1 c2 c3
  refine ⟨rfl, rfl, c1, c2, hood, hcons, ?_⟩
  rw [← c3, hres₂, hres₁]
  rfl

/-! ## The final check's words, as the model's -/

theorem getElem!_range_map {α : Type} [Inhabited α] (f : ℕ → α) {N y : ℕ} (hy : y < N) :
    ((Array.range N).map f)[y]! = f y := by
  rw [getElem!_pos _ y (by simpa using hy), Array.getElem_map, Array.getElem_range]

/-- One out-of-domain sample's term in residual word `y`. -/
def oodTerm (ext : ℕ) (fs : Array F256 × F256) (o : Flock.OodCtx) (y : ℕ) : F256 :=
  F256.smul (F256.smul ((Array.range fs.1.size).foldl
      (fun acc t => acc * ((1 : F256) + F256.ofF o.z[t]! + fs.1[t]!)) (1 : F256) * fs.2) o.beta)
    ((Array.range ext).foldl (fun acc j =>
      acc * (if (y >>> j) % 2 == 1 then o.z[fs.1.size + j]! else (1 : F128) + o.z[fs.1.size + j]!)) (1 : F128))

/-- One consistency context's term in residual word `y`. -/
def consTerm (sv : Array F128) (ext : ℕ) (levels : Array (Array F256)) (cs : Flock.ConsCtx) (y : ℕ) : F256 :=
  ((consAdd sv ext (consFixed levels cs.start) cs)[y]! * prodFromE levels cs.start).smul cs.beta

theorem oodFold_get (ext : ℕ) (F : Flock.OodCtx → Array F256 × F256) {y : ℕ} (hy : y < 2 ^ ext) :
    ∀ (l : List Flock.OodCtx) (init : Array F256),
      (l.foldl (fun res o => oodStep ext (F o) o res) init)[y]! =
        l.foldl (fun acc o => acc + oodTerm ext (F o) o y) init[y]!
  | [], _ => rfl
  | o :: l, init => by
      rw [List.foldl_cons, List.foldl_cons, oodFold_get ext F hy l]
      unfold oodStep
      rw [getElem!_range_map _ hy]
      rfl

theorem consFold_get (sv : Array F128) (ext : ℕ) (levels : Array (Array F256)) {y : ℕ} (hy : y < 2 ^ ext) :
    ∀ (l : List Flock.ConsCtx) (init : Array F256),
      (l.foldl (consStep sv ext levels) init)[y]! = l.foldl (fun acc cs => acc + consTerm sv ext levels cs y) init[y]!
  | [], _ => rfl
  | cs :: l, init => by
      rw [List.foldl_cons, List.foldl_cons, consFold_get sv ext levels hy l]
      unfold consStep
      rw [getElem!_range_map _ hy]
      rfl

theorem gOf_foldl_add {α : Type} (f : α → F256) : ∀ (l : List α) (x : F256),
    gOf (l.foldl (fun acc a => acc + f a) x) = gOf x + (l.map fun a => gOf (f a)).sum
  | [], x => by simp
  | a :: l, x => by rw [List.foldl_cons, gOf_foldl_add f l, gOf_add, List.map_cons, List.sum_cons, add_assoc]

theorem gOf_foldl_mul {α : Type} (f : α → F256) : ∀ (l : List α) (x : F256),
    gOf (l.foldl (fun acc a => acc * f a) x) = gOf x * (l.map fun a => gOf (f a)).prod
  | [], x => by simp
  | a :: l, x => by rw [List.foldl_cons, gOf_foldl_mul f l, gOf_mul, List.map_cons, List.prod_cons, mul_assoc]

theorem gOf_cff (c : F256) : gOf (Flock.cff c) = Model.cff execArith (gOf c) := by
  unfold Flock.cff Model.cff
  rw [gOf_add, gOf_one, gOf_mul, gOf_add, gOf_one, gOf_u]
  rfl

/-- The model's levels, from the executable's. -/
abbrev levelsOf (levels : Array (Array F256)) : List (List 𝕂) := levels.toList.map fun l => l.toList.map gOf

theorem tail_map_gOf (l : List F256) : (l.map gOf).tail = l.tail.map gOf := by cases l <;> rfl

theorem extract_toList {α : Type} (a : Array α) (i : ℕ) : (a.extract i a.size).toList = a.toList.drop i := by
  rw [Array.toList_extract, List.extract_eq_take_drop, List.take_of_length_le (by simp)]

theorem prodFromE_eq (levels : Array (Array F256)) (hl : ∀ l ∈ levels.toList, l.size = 4) (a : ℕ) :
    gOf (prodFromE levels a) = Model.cffFrom execArith (levelsOf levels) a := by
  unfold prodFromE Model.cffFrom
  rw [← Array.foldl_toList, extract_toList, gOf_foldl_mul, gOf_one, one_mul, Array.toList_map, List.map_drop,
    List.map_drop, List.map_map, List.map_map]
  refine congrArg List.prod (congrArg (List.drop a) (List.map_congr_left fun l hm => ?_))
  have hl4 := hl l hm
  simp only [Function.comp, gOf_cff]
  congr 1
  rw [getElem!_pos l 0 (by omega)]
  cases hlist : l.toList with
  | nil => rw [← Array.length_toList, hlist] at hl4; cases hl4
  | cons x xs =>
      rw [List.map_cons, List.headD_cons]
      congr 1
      have h0 : l.toList[0]'(by simp [hlist]) = x := by simp [hlist]
      rw [← h0, Array.getElem_toList]

theorem foldl_tails (L : List (Array F256)) : ∀ acc : Array F256,
    (L.foldl (fun acc l => acc ++ l.extract 1 l.size) acc).toList = acc.toList ++ (L.map fun l => l.toList.tail).flatten := by
  induction L with
  | nil => intro acc; simp
  | cons l L ih =>
      intro acc
      rw [List.foldl_cons, ih, Array.toList_append, extract_toList]
      simp [List.drop_one]

theorem map_tails (L : List (Array F256)) :
    (L.map fun l => l.toList.tail).flatten.map gOf = ((L.map fun l => l.toList.map gOf).map List.tail).flatten := by
  rw [List.map_flatten, List.map_map, List.map_map]
  congr 1
  exact List.map_congr_left fun l _ => (tail_map_gOf l.toList).symm

theorem origE_toList (initCh : Array F256) (levels : Array (Array F256)) :
    (origE initCh levels).toList.map gOf =
      initCh.toList.map gOf ++ ((levelsOf levels).map List.tail).flatten := by
  unfold origE
  rw [← Array.foldl_toList, foldl_tails, List.map_append, map_tails]

theorem consFixed_toList (levels : Array (Array F256)) (start : ℕ) :
    (consFixed levels start).toList.map gOf = (((levelsOf levels).drop start).map List.tail).flatten := by
  unfold consFixed
  rw [← Array.foldl_toList, foldl_tails, extract_toList, Array.toList_empty, List.nil_append, map_tails, List.map_drop]

theorem foldl_tails' (L : List (Array F256)) : ∀ acc : Array F256, acc.isEmpty = false →
    (L.foldl (fun acc l => if acc.isEmpty then l else acc ++ l.extract 1 l.size) acc).toList =
      acc.toList ++ (L.map fun l => l.toList.tail).flatten := by
  induction L with
  | nil => intro acc _; simp
  | cons l L ih =>
      intro acc hacc
      have hne : (acc ++ l.extract 1 l.size).isEmpty = false := by
        unfold Array.isEmpty at hacc ⊢
        simp only [Array.size_append, decide_eq_false_iff_not] at hacc ⊢
        omega
      rw [List.foldl_cons, hacc, ite_eq_right Bool.false_ne_true, ih _ hne, Array.toList_append, extract_toList]
      simp only [List.drop_one, List.map_cons, List.flatten_cons, List.append_assoc]

theorem oodFixed_some (levels : Array (Array F256)) (hl : ∀ l ∈ levels.toList, l.size = 4) (s : ℕ) :
    ((levels.extract s levels.size).foldl (fun acc l => if acc.isEmpty then l else acc ++ l.extract 1 l.size)
      #[]).toList.map gOf = (levelsOf levels).getD s [] ++ (((levelsOf levels).drop (s + 1)).map List.tail).flatten := by
  rw [← Array.foldl_toList, extract_toList]
  by_cases hs : s < levels.size
  · have hs' : s < levels.toList.length := by simpa using hs
    rw [List.drop_eq_getElem_cons hs', List.foldl_cons]
    simp only [Array.isEmpty_empty, ite_true]
    have h4 := hl _ (List.getElem_mem hs')
    have h4' : levels[s].size = 4 := by rw [← Array.getElem_toList]; exact h4
    rw [foldl_tails' _ _ (by unfold Array.isEmpty; simp [h4']), List.map_append, map_tails,
      List.getD_eq_getElem _ _ (by simpa using hs), List.map_drop]
    simp
  · have hs' : levels.toList.length ≤ s := by simpa using Nat.le_of_not_lt hs
    rw [List.drop_eq_nil_of_le hs', List.getD_eq_default _ _ (by simpa using hs'),
      List.drop_eq_nil_of_le (by simp [levelsOf]; omega)]
    rfl

theorem bit_eq (y j : ℕ) : ((y >>> j) % 2 == 1) = y.testBit j := by
  rw [Nat.testBit_eq_decide_div_mod_eq, Nat.shiftRight_eq_div_pow]
  cases h : y / 2 ^ j % 2 == 1 <;> simp_all

theorem fOf_foldl_mul' {α : Type} (f : α → F128) : ∀ (l : List α) (x : F128),
    fOf (l.foldl (fun acc a => acc * f a) x) = fOf x * (l.map fun a => fOf (f a)).prod
  | [], x => by simp
  | a :: l, x => by rw [List.foldl_cons, fOf_foldl_mul' f l, fOf_mul, List.map_cons, List.prod_cons, mul_assoc]

theorem fOf_foldl_ite {α : Type} (c : α → Bool) (f : α → F128) : ∀ (l : List α) (x : F128),
    fOf (l.foldl (fun acc a => if c a = true then acc * f a else acc) x) =
      fOf x * (l.map fun a => if c a = true then fOf (f a) else 1).prod
  | [], x => by simp
  | a :: l, x => by
      rw [List.foldl_cons, fOf_foldl_ite c f l, List.map_cons, List.prod_cons]
      cases c a <;> simp [fOf_mul, mul_assoc]

theorem zip_map_eq {R : Type} [Zero R] (a b : List R) (f : R × R → R) (h : b.length ≤ a.length) :
    (a.zip b).map f = (List.range b.length).map fun t => f (a.getD t 0, b.getD t 0) := by
  refine List.ext_getElem (by simp [Nat.min_eq_right h]) fun i h₁ h₂ => ?_
  simp only [List.length_map, List.length_zip, Nat.min_eq_right h] at h₁
  simp [List.getElem_zip, List.getElem?_eq_getElem h₁, List.getElem?_eq_getElem (show i < a.length by omega)]

theorem zipIdx_map_eq {α β : Type} (l : List α) (d : α) (f : α × ℕ → β) :
    l.zipIdx.map f = (List.range l.length).map fun i => f (l.getD i d, i) := by
  refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
  simp only [List.length_map, List.length_zipIdx] at h₁
  simp [List.getElem_zipIdx, List.getElem?_eq_getElem h₁]

theorem range_foldl {α : Type} (N : ℕ) (f : α → ℕ → α) (x : α) :
    (Array.range N).foldl f x = (List.range N).foldl f x := by
  rw [← Array.foldl_toList, Array.toList_range]

theorem getD_map_fOf (a : Array F128) (i : ℕ) (hi : i < a.size) : (a.toList.map fOf).getD i 0 = fOf a[i]! := by
  rw [List.getD_eq_getElem _ _ (by simpa using hi), List.getElem_map, Array.getElem_toList, getElem!_pos a i hi]

theorem getD_map_gOf (a : Array F256) (i : ℕ) (hi : i < a.size) : (a.toList.map gOf).getD i 0 = gOf a[i]! := by
  rw [List.getD_eq_getElem _ _ (by simpa using hi), List.getElem_map, Array.getElem_toList, getElem!_pos a i hi]

/-- **One out-of-domain term** (§13.6): the executable's, in residual word `y`, is the model's. -/
theorem gOf_oodTerm (ext : ℕ) (fs : Array F256 × F256) (o : Flock.OodCtx) (y : ℕ)
    (hz : o.z.size = fs.1.size + ext) :
    gOf (oodTerm ext fs o y) = execArith.embed (fOf o.beta) * gOf fs.2 *
      (List.map (fun x => 1 + x.1 + x.2) ((List.map execArith.embed (o.z.toList.map fOf)).zip
        (fs.1.toList.map gOf))).prod *
      (List.map (fun j => if y.testBit j = true then
          execArith.embed ((o.z.toList.map fOf).getD ((fs.1.toList.map gOf).length + j) 0)
        else 1 + execArith.embed ((o.z.toList.map fOf).getD ((fs.1.toList.map gOf).length + j) 0))
        (List.range ext)).prod := by
  have hemb : execArith.embed = FlockLevel3.embed := rfl
  unfold oodTerm
  rw [hemb, gOf_smul, gOf_smul, gOf_mul, range_foldl, gOf_foldl_mul, gOf_one, one_mul, range_foldl, fOf_foldl_mul',
    fOf_one, one_mul, map_list_prod, zip_map_eq _ _ _ (by simp; omega)]
  have hhead : (List.map (fun t => gOf ((1 : F256) + F256.ofF o.z[t]! + fs.1[t]!)) (List.range fs.1.size)) =
      List.map (fun t => 1 + ((o.z.toList.map fOf).map FlockLevel3.embed).getD t 0 + (fs.1.toList.map gOf).getD t 0)
        (List.range (fs.1.toList.map gOf).length) := by
    rw [List.length_map, Array.length_toList]
    refine List.map_congr_left fun t ht => ?_
    rw [List.mem_range] at ht
    rw [List.map_map, List.getD_eq_getElem _ _ (by simp; omega), getD_map_gOf _ _ ht, gOf_add, gOf_add, gOf_one,
      gOf_ofF]
    simp only [List.getElem_map, Array.getElem_toList, Function.comp]
    rw [getElem!_pos o.z t (by omega)]
  have htail : List.map FlockLevel3.embed (List.map (fun j => fOf (if ((y >>> j) % 2 == 1) = true then
      o.z[fs.1.size + j]! else 1 + o.z[fs.1.size + j]!)) (List.range ext)) =
      List.map (fun j => if y.testBit j = true then
          FlockLevel3.embed ((o.z.toList.map fOf).getD ((fs.1.toList.map gOf).length + j) 0)
        else 1 + FlockLevel3.embed ((o.z.toList.map fOf).getD ((fs.1.toList.map gOf).length + j) 0))
        (List.range ext) := by
    rw [List.map_map, List.length_map, Array.length_toList]
    refine List.map_congr_left fun j hj => ?_
    rw [List.mem_range] at hj
    simp only [Function.comp]
    rw [bit_eq, getD_map_fOf _ _ (by omega)]
    split_ifs
    · rfl
    · rw [fOf_add, fOf_one, map_add, map_one]
  rw [hhead, htail]
  ring


/-- One query's term in a consistency context's sum. -/
def consAddTerm (sv : Array F128) (ext : ℕ) (fixed : Array F256) (cs : Flock.ConsCtx) (y j : ℕ) : F256 :=
  F256.smul (F256.smul ((Array.range fixed.size).foldl (fun acc i =>
      acc * ((1 : F256) + fixed[i]! * F256.ofF ((1 : F128) + (Flock.whats sv cs.pos[j]! cs.cols)[i]!))) (1 : F256))
      ((Flock.eqTable cs.alpha)[j]!)) ((Array.range ext).foldl (fun acc i =>
        if (y >>> i) % 2 == 1 then acc * (Flock.whats sv cs.pos[j]! cs.cols)[fixed.size + i]! else acc) (1 : F128))

theorem consAdd_get (sv : Array F128) (ext : ℕ) (fixed : Array F256) (cs : Flock.ConsCtx) {y : ℕ}
    (hy : y < 2 ^ ext) :
    (consAdd sv ext fixed cs)[y]! = (List.range cs.pos.size).foldl (fun acc j => acc + consAddTerm sv ext fixed cs y j) 0 := by
  have hgen : ∀ (L : List ℕ) (init : Array F256), (L.foldl (fun add j => (Array.range (2 ^ ext)).map fun y =>
      add[y]! + consAddTerm sv ext fixed cs y j) init)[y]! =
      L.foldl (fun acc j => acc + consAddTerm sv ext fixed cs y j) init[y]! := by
    intro L
    induction L with
    | nil => intro init; rfl
    | cons j L ih => intro init; rw [List.foldl_cons, List.foldl_cons, ih, getElem!_range_map _ hy]
  unfold consAdd
  rw [← List.range_eq_range']
  refine (hgen _ _).trans ?_
  rw [getElem!_pos _ y (by simpa using hy), Array.getElem_replicate]

/-- **One consistency term** (§13.6): the executable's, in residual word `y`, is the model's, when the context's
columns are its fixed coordinates and the residual (`hcols`), within the subspace table (`hm`), and its queries within
its batching table (`hpos`). -/
theorem gOf_consTerm (m ext : ℕ) (levels : Array (Array F256)) (hl : ∀ l ∈ levels.toList, l.size = 4)
    (cs : Flock.ConsCtx) (y : ℕ) (hy : y < 2 ^ ext) (hcols : (consFixed levels cs.start).size + ext = cs.cols)
    (hm : cs.cols ≤ m) (hpos : cs.pos.size ≤ 2 ^ cs.alpha.size) :
    gOf (consTerm (svTable m) ext levels cs y) =
      execArith.embed (fOf cs.beta) * Model.cffFrom execArith (levelsOf levels) cs.start *
        (List.map (fun x_1 => execArith.embed (Model.eqList (cs.alpha.toList.map fOf) x_1.2) *
            (List.map (fun x => 1 + x.1 * (1 + execArith.embed (execArith.What x.2 (execArith.omega x_1.1))))
              (List.map List.tail (List.drop cs.start (levelsOf levels))).flatten.zipIdx).prod *
            (List.map (fun i => if y.testBit i = true then
                execArith.embed (execArith.What
                  ((List.map List.tail (List.drop cs.start (levelsOf levels))).flatten.length + i)
                  (execArith.omega x_1.1)) else 1) (List.range ext)).prod)
          cs.pos.toList.zipIdx).sum := by
  have hemb : execArith.embed = FlockLevel3.embed := rfl
  have hW : execArith.What = FlockLevel3.What := rfl
  have hom : execArith.omega = FlockLevel3.omega := rfl
  rw [← consFixed_toList, hemb, hW, hom]
  unfold consTerm
  rw [gOf_smul, gOf_mul, prodFromE_eq levels hl, consAdd_get _ _ _ _ hy, gOf_foldl_add, gOf_zero, zero_add,
    zipIdx_map_eq cs.pos.toList 0, show cs.pos.toList.length = cs.pos.size from Array.length_toList]
  have hs : ∀ j ∈ List.range cs.pos.size, gOf (consAddTerm (svTable m) ext (consFixed levels cs.start) cs y j) =
      FlockLevel3.embed (Model.eqList (cs.alpha.toList.map fOf) j) *
        (List.map (fun x => 1 + x.1 * (1 + FlockLevel3.embed (FlockLevel3.What x.2
            (FlockLevel3.omega (cs.pos.toList.getD j 0)))))
          ((consFixed levels cs.start).toList.map gOf).zipIdx).prod *
        (List.map (fun i => if y.testBit i = true then FlockLevel3.embed (FlockLevel3.What
            (((consFixed levels cs.start).toList.map gOf).length + i) (FlockLevel3.omega (cs.pos.toList.getD j 0)))
          else 1) (List.range ext)).prod := by
    intro j hj
    rw [List.mem_range] at hj
    have hp : cs.pos.toList.getD j 0 = cs.pos[j]! := by
      rw [List.getD_eq_getElem _ _ (by simpa using hj), Array.getElem_toList, getElem!_pos cs.pos j hj]
    rw [hp]
    unfold consAddTerm
    rw [gOf_smul, gOf_smul, range_foldl, gOf_foldl_mul, gOf_one, one_mul, range_foldl, fOf_foldl_ite, fOf_one,
      one_mul, map_list_prod, eqTable_eqList _ _ (by omega), zipIdx_map_eq _ 0, List.length_map, Array.length_toList]
    have hhead : List.map (fun i => gOf ((1 : F256) + (consFixed levels cs.start)[i]! *
        F256.ofF ((1 : F128) + (Flock.whats (svTable m) cs.pos[j]! cs.cols)[i]!))) (List.range (consFixed levels cs.start).size) =
        List.map (fun i => 1 + ((consFixed levels cs.start).toList.map gOf).getD i 0 *
          (1 + FlockLevel3.embed (FlockLevel3.What i (FlockLevel3.omega cs.pos[j]!))))
          (List.range (consFixed levels cs.start).size) := by
      refine List.map_congr_left fun i hi => ?_
      rw [List.mem_range] at hi
      rw [getD_map_gOf _ _ hi, gOf_add, gOf_one, gOf_mul, gOf_ofF, fOf_add, fOf_one, map_add, map_one,
        whats_get m _ _ i hm (by omega)]
    have htail : List.map FlockLevel3.embed (List.map (fun i => if ((y >>> i) % 2 == 1) = true then
        fOf (Flock.whats (svTable m) cs.pos[j]! cs.cols)[(consFixed levels cs.start).size + i]! else 1)
          (List.range ext)) =
        List.map (fun i => if y.testBit i = true then FlockLevel3.embed (FlockLevel3.What
          ((consFixed levels cs.start).size + i) (FlockLevel3.omega cs.pos[j]!)) else 1) (List.range ext) := by
      rw [List.map_map]
      refine List.map_congr_left fun i hi => ?_
      rw [List.mem_range] at hi
      simp only [Function.comp, bit_eq]
      split_ifs
      · rw [whats_get m _ _ _ hm (by omega)]
      · exact map_one _
    rw [hhead, htail]
    ring
  rw [List.map_congr_left hs]
  ring


theorem consFixed_size_le (initCh : Array F256) (levels : Array (Array F256)) (start : ℕ) :
    (consFixed levels start).size ≤ (origE initCh levels).size := by
  rw [← Array.length_toList, ← Array.length_toList]
  unfold consFixed origE
  rw [← Array.foldl_toList, ← Array.foldl_toList, foldl_tails, foldl_tails, extract_toList]
  simp only [Array.toList_empty, List.nil_append, List.length_append]
  have e := congrArg (fun L : List (Array F256) => (L.map fun l => l.toList.tail).flatten.length)
    (List.take_append_drop start levels.toList)
  simp only [List.map_append, List.flatten_append, List.length_append] at e
  omega

/-- The final check's left-hand side, as the model's sum. -/
theorem gOf_lhsE (yr : Array F128) (ext : ℕ) (res : Array F256) :
    gOf (lhsE yr ext res) = ∑ y ∈ Finset.range (2 ^ ext),
      gOf res[y]! * (execArith.embed (fOf yr[2 * y]!) + execArith.embed (fOf yr[2 * y + 1]!) * execArith.u) := by
  unfold lhsE
  rw [range_foldl, gOf_foldl_add, gOf_zero, zero_add, FlockLevel3.list_sum_range]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [gOf_mul, gOf_limbs (⟨yr[2 * y]!, yr[2 * y + 1]!⟩ : F256)]
  rfl


theorem final_refines {p : LigeritoProof} {basisAt : Array F256 → ℕ → Array F256} {n : ℕ} {initCh : Array F256}
    {levels : Array (Array F256)} {ext : ℕ} {st st' : LState} {tr tr' : Transcript} (b : ℕ → 𝔽)
    (hl : ∀ l ∈ levels.toList, l.size = 4) (hyr : p.finalProof.yr.size = 2 ^ (ext + 1))
    (hpos : ∀ cs ∈ st.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size)
    (hbasis : ∀ y < 2 ^ ext, gOf (basisAt (origE initCh levels) ext)[y]! =
      Model.bHat execArith n b ((origE initCh levels).toList.map gOf ++ bitsK ext y))
    (h : finalProg p basisAt n initCh levels ext st tr = .ok (((), st'), tr')) :
    tr' = tr ∧ Model.finalCheck execArith n b (initCh.toList.map gOf) (levelsOf levels) (st.oods.toList.map oodOf)
      (st.conss.toList.map consOf) ext (yrOf p ext) (gOf st.s) = true := by
  obtain ⟨-, rfl, horig, hbres, hood, hcons, hlhs⟩ := finalProg_ok h
  refine ⟨rfl, ?_⟩
  unfold Model.finalCheck
  simp only [Bool.and_eq_true, decide_eq_true_eq, ← origE_toList, List.length_map, Array.length_toList]
  refine ⟨horig, ?_⟩
  rw [← hlhs, gOf_lhsE]
  refine Finset.sum_congr rfl fun y hy => ?_
  rw [Finset.mem_range] at hy
  have hy2 : 2 * y < 2 ^ (ext + 1) := by rw [pow_succ]; omega
  have hy3 : 2 * y + 1 < 2 ^ (ext + 1) := by rw [pow_succ]; omega
  have hyr0 : yrOf p ext ⟨2 * y % 2 ^ (ext + 1), Nat.mod_lt _ (Nat.two_pow_pos _)⟩ = fOf p.finalProof.yr[2 * y]! := by
    simp only [yrOf, Nat.mod_eq_of_lt hy2]
  have hyr1 : yrOf p ext ⟨(2 * y + 1) % 2 ^ (ext + 1), Nat.mod_lt _ (Nat.two_pow_pos _)⟩ =
      fOf p.finalProof.yr[2 * y + 1]! := by
    simp only [yrOf, Nat.mod_eq_of_lt hy3]
  rw [hyr0, hyr1, mul_comm execArith.u]
  refine congrArg (· * _) ?_
  symm
  unfold resE
  rw [consFold_get _ _ _ hy, gOf_foldl_add, oodFold_get _ _ hy, gOf_foldl_add,
    getElem!_pos _ y (by simpa [hbres] using hy), Array.getElem_map, gOf_mul,
    ← getElem!_pos _ y (by simpa [hbres] using hy), hbasis y hy, prodFromE_eq levels hl]
  refine congrArg₂ (· + ·) (congrArg₂ (· + ·) rfl ?_) ?_
  · rw [List.map_map]
    refine congrArg List.sum (List.map_congr_left fun o ho => ?_)
    have hz := hood o ho
    cases hsp : o.split with
    | none =>
        have hfs : oodFixed levels (origE initCh levels) o = (origE initCh levels, prodFromE levels 0) := by
          simp [oodFixed, hsp]
        rw [hfs] at hz
        simp only [Function.comp, hfs]
        rw [gOf_oodTerm ext _ o y hz, prodFromE_eq levels hl]
        simp only [oodOf, hsp]
    | some a =>
        have hfs : oodFixed levels (origE initCh levels) o = ((levels.extract a levels.size).foldl
            (fun acc l => if acc.isEmpty then l else acc ++ l.extract 1 l.size) #[], prodFromE levels (a + 1)) := by
          simp [oodFixed, hsp]
        rw [hfs] at hz
        simp only [Function.comp, hfs]
        rw [gOf_oodTerm ext _ o y hz, prodFromE_eq levels hl, oodFixed_some levels hl a]
        simp only [oodOf, hsp]
  · rw [List.map_map]
    refine congrArg List.sum (List.map_congr_left fun cs hcs' => ?_)
    have hc := hcons cs hcs'
    have hle := consFixed_size_le initCh levels cs.start
    simp only [Function.comp]
    rw [gOf_consTerm (n + 1) ext levels hl cs y hy hc (by omega) (hpos cs hcs')]
    simp only [consOf]

/-- **Ligerito refines the model's** (§13.3–§13.6). When the executable's Ligerito verifier accepts, and its basis
`basisAt` is the model's `bHat` of `b` at every original point with the schedule's residual (`hbasis`), the model's
compiled Ligerito, played with the proof's messages and the rounds consumed, accepts at the openings `o`: its verdict
there is true, its caps are the proof's, and every request passed the executable's Merkle check. -/
theorem ligerito_accepts {D : Type} (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (r : ℕ) (b : ℕ → 𝔽)
    (capD : Array ByteArray → Cap D) (o : Opens 𝔽 D) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1)
    (ho : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q,
      (o r ℓ j).1 = ((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf)
    (hbasis : ∀ (orig : Array F256) (y : ℕ), orig.size + sch.ext = sch.m - 7 → y < 2 ^ sch.ext →
      gOf (basisAt orig sch.ext)[y]! = Model.bHat execArith (sch.m - 7) b (orig.toList.map gOf ++ bitsK sch.ext y))
    {t t' : Transcript} (hwf : RoundsWF t.rounds) (h : Flock.ligerito ms sch T cap p basisAt t = .ok ((), t')) :
    ∃ pend : PendL 𝔽 D, t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
      Plays fOf (Model.ligeritoC execArith r (schedOf sch) (fOf T) b) (ligMsgs p capD sch)
        (consumed t.rounds t.k t'.k) pend ∧
      pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capD p.recursiveCaps[j]!) ∧
      (∀ req ∈ pend.reqs, ReqOK ms sch cap p req) ∧ pend.verdict o = true := by
  obtain ⟨pend, initCh, levels, stF, stF', tF, hr, hk, hpl, hcaps, hreqs, -, -, hlanes, hyr, hpos, hfin, hver⟩ :=
    ligerito_refines ms sch T cap p basisAt r b capD o hdim hlv hα1 ho hwf h
  have hn7 := hdim.1
  obtain ⟨-, -, horig, -⟩ := finalProg_ok hfin
  obtain ⟨rfl, hfc⟩ := final_refines (n := sch.n) b hlanes hyr hpos
    (fun y hy => by rw [hn7]; exact hbasis _ y (by omega) hy) hfin
  refine ⟨pend, hr, hk, hpl, hcaps, hreqs, ?_⟩
  rw [hver, ← hn7]
  exact hfc

end FlockSoundness.Refine
