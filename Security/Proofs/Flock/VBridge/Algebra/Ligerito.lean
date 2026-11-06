import Proofs.Flock.VBridge.Algebra.RingSwitch
import Proofs.Flock.Soundness.Refine.Final

/-!
# V*'s Ligerito residuals are zero only when Ligerito's final check passes

V*'s Ligerito (`ligerito`, rec_algebra.py:637–721) computes over F the K values the verifier's (`Flock.ligerito`,
Ligerito.lean:165–277) computes over `F256`: a K value `(c0, c1)` is the word `⟨c0, c1⟩` (`kv`), a round polynomial
`(a0, a1, a2)` the verifier's `Poly` (`polyOf`).

* Each update of the running sum and the round polynomial is the verifier's: a challenge's (`kv_challenge`,
  Ligerito.lean:120–122), an OOD sample's (`kv_ood`, `polyOf_addScaled`, Ligerito.lean:131–134) and a level's queries'
  (`kv_query`, Ligerito.lean:196–197).
* A level's consistency sum `e = Σ_l eq_K(fold)[l]·cons[l]` is the verifier's
  `e = Σ_j eq(α)[j]·Σ_l eq_K(fold)[l]·row_j[l]` (`Refine.consE`, Ligerito.lean:151–159) when each `cons[l]` is
  `Σ_j eq(α)[j]·row_j[l]` (`kv_e`).
* `ligerito_checks`: when V*'s two final-check residuals (`c0`, `c1`) are zero, the final check
  (Ligerito.lean:235–273, `Refine.finalProg`) holds on V*'s final state: the shape checks, and `lhsE yr ext res = s`,
  with `res` the verifier's residual weights (`Refine.resE`) at V*'s challenges and contexts.
* `ligerito_of_residuals` is the stage at a rep.
* `finalProg_of` is `Refine.finalProg_ok`'s converse: on a state that meets the final check's conditions, the verifier's
  final check passes; `finalProg_of_finalOK` applies it to `FinalOK`.
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine

/-! ## A record's rounds of some coins -/

/-- A record with no round of no coins, which `CoinSource.ofRounds` reads as recorded: it skips a round of no coins
(RecAlgebra.lean, `CoinSource.ofRounds`), and a record whose rounds have `run`'s sizes has none (`verifierRows_schedule`). -/
def NoEmpty (rounds : Array (Array F128)) : Prop := ∀ r ∈ rounds, r.size ≠ 0

theorem filter_noEmpty {rounds : Array (Array F128)} (h : NoEmpty rounds) : rounds.filter (·.size ≠ 0) = rounds :=
  Array.filter_eq_self.mpr fun r hr => decide_eq_true (h r hr)

theorem noEmpty_filter (rounds : Array (Array F128)) : NoEmpty (rounds.filter (·.size ≠ 0)) := fun r hr => by
  simpa using (Array.mem_filter.mp hr).2

theorem noEmpty_of_sizes {rounds : Array (Array F128)} {ns : List Nat} (h : rounds.toList.map Array.size = ns)
    (hn : ∀ n ∈ ns, n ≠ 0) : NoEmpty rounds := fun r hr => by
  subst h
  exact hn _ (List.mem_map_of_mem (Array.mem_toList_iff.mpr hr))

theorem ofRounds_filter (rounds : Array (Array F128)) :
    CoinSource.ofRounds (rounds.filter (·.size ≠ 0)) = CoinSource.ofRounds rounds := by
  unfold CoinSource.ofRounds
  rw [filter_noEmpty (noEmpty_filter rounds)]

/-! ## K values as the verifier's words -/

/-- A K value over F as the verifier's word `c0 + c1·u`. -/
def kv (x : KV F128) : F256 := ⟨x.1, x.2⟩

/-- A round polynomial over F as the verifier's. -/
def polyOf (q : KV F128 × KV F128 × KV F128) : Poly := ⟨kv q.1, kv q.2.1, kv q.2.2⟩

theorem gOf_inj {a b : F256} (h : gOf a = gOf b) : a = b := h

theorem f128_mul_comm (a b : F128) : a * b = b * a := by
  (apply fOf_injective; simp only [fOf_mul]; ring)

theorem f128_zero_mul (b : F128) : 0 * b = 0 := by
  (apply fOf_injective; simp only [fOf_mul, fOf_zero]; ring)

theorem kv_kadd (x y : KV F128) : kv (kadd x y) = kv x + kv y := rfl

theorem kv_kf (c : F256) (x : F128) : kv (kf c x) = c.smul x := rfl

theorem kv_kofF (y : F128) : kv (kofF y) = F256.ofF y := rfl

theorem kv_fk (b : F128) (x : KV F128) : kv (fk b x) = (kv x).smul b := by
  simp only [kv, fk, F256.smul, Val.sm, f128_mul_comm b]

theorem kv_kc (c : F256) (x : KV F128) : kv (kc c x) = c * kv x := by
  rw [FlockLevel3.F256.mul_def]
  simp only [kv, kc, Val.add, Val.sm, F256.mk.injEq]
  constructor
  · (apply fOf_injective; simp only [fOf_add, fOf_mul]; ring)
  · apply fOf_injective
    simp only [fOf_add, fOf_mul]
    linear_combination (-(fOf c.c0 * fOf x.1)) * FlockLevel3.GF128.two_eq_zero

theorem kv_foldl_kadd : ∀ (l : List (KV F128)) (acc : KV F128),
    kv (l.foldl kadd acc) = (l.map kv).foldl (· + ·) (kv acc)
  | [], _ => rfl
  | a :: l, acc => by
    rw [List.foldl_cons, kv_foldl_kadd l, List.map_cons, List.foldl_cons, kv_kadd]

/-! ## The updates of the running sum and the round polynomial -/

/-- A challenge's running sum (Ligerito.lean:120): `q(r) = a0 + a1·r + a2·r²`. -/
theorem kv_challenge (a0 a1 a2 : KV F128) (r : F256) :
    kv (kadd (kadd a0 (kc r a1)) (kc (r * r) a2)) = (polyOf (a0, a1, a2)).eval r := by
  simp only [kv_kadd, kv_kc, polyOf, Poly.eval]
  (apply gOf_inj; simp only [gOf_add, gOf_mul]; ring)

/-- A message's round polynomial (`Poly.ofMsg`). -/
theorem polyOf_ofMsg (mm : KV F128 × KV F128) (s : KV F128) :
    polyOf (ofMsg mm s) = Poly.ofMsg (kv mm.1, kv mm.2) (kv s) := rfl

/-- A batched round polynomial (`Poly.addScaled`, Ligerito.lean:134, 197). -/
theorem polyOf_addScaled (q p : KV F128 × KV F128 × KV F128) (b : F128) :
    polyOf (kadd q.1 (fk b p.1), kadd q.2.1 (fk b p.2.1), kadd q.2.2 (fk b p.2.2)) =
      (polyOf q).addScaled (polyOf p) b := by
  simp only [polyOf, kv_kadd, kv_fk, Poly.addScaled]

/-- An OOD sample's running sum (Ligerito.lean:134, 182): `s + y·β`. -/
theorem kv_ood (s : KV F128) (b y : F128) : kv (kadd s (kofF (Val.sm b y))) = kv s + (F256.ofF y).smul b := by
  rw [kv_kadd, kv_kofF]
  show kv s + F256.ofF (b * y) = kv s + ⟨y * b, 0 * b⟩
  rw [f128_zero_mul, f128_mul_comm b y]
  rfl

/-- A level's queries' running sum (Ligerito.lean:197, 233): `s + e·β`. -/
theorem kv_query (s e : KV F128) (b : F128) : kv (kadd s (fk b e)) = kv s + (kv e).smul b := by
  rw [kv_kadd, kv_fk]

/-! ## A level's consistency sum -/

/-- The verifier's consistency sum on lane `l`: `Σ_j eq(α)[j]·row_j[l]`, the value the decoded messages' `cons`
must hold. -/
def consAt (rows : Array (Array F128)) (q : Nat) (alpha : Array F128) (l : Nat) : F128 :=
  rangeSum q fun j => (eqTable alpha)[j]! * (rows[j]!)[l]!

theorem fOf_rangeSum (n : Nat) (f : Nat → F128) : fOf (rangeSum n f) = ∑ j ∈ Finset.range n, fOf (f j) := by
  unfold rangeSum
  suffices ∀ acc, fOf ((List.range n).foldl (fun acc i => acc + f i) acc) =
      fOf acc + ∑ j ∈ Finset.range n, fOf (f j) by rw [this, fOf_zero, zero_add]
  induction n with
  | zero => intro acc; simp
  | succ n ih =>
    intro acc
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, fOf_add, ih, Finset.sum_range_succ,
      add_assoc]

/-- **A level's consistency sum.** V*'s `e` (`query`, rec_algebra.py:666–673), on consistency sums `cons[l]` that are
the verifier's `Σ_j eq(α)[j]·row_j[l]` over the level's `N = 2^|fold|` lanes, is the verifier's `e`. -/
theorem kv_e (rows : Array (Array F128)) (q : Nat) (alpha : Array F128) (fold : Array F256) (N : Nat)
    (hN : (eqTableK fold).size = N) (hrows : ∀ j < q, rows[j]!.size = N) (cs : Nat → F128)
    (hcs : ∀ l < N, cs l = consAt rows q alpha l) :
    kv (kaddAll (((eqTableK fold).zip ((Array.range N).map cs)).toList.map fun fa => kf fa.1 fa.2)) =
      consE rows q alpha fold := by
  have hz : ((eqTableK fold).zip ((Array.range N).map cs)).toList =
      (List.range N).map fun l => ((eqTableK fold)[l]!, ((Array.range N).map cs)[l]!) := by
    rw [Array.toList_zip]
    refine List.ext_getElem (by simp [hN]) fun i h₁ h₂ => ?_
    have hi : i < N := by simpa using h₂
    simp only [List.getElem_zip, List.getElem_map, List.getElem_range, Array.getElem_toList]
    rw [getElem!_pos (eqTableK fold) i (by omega), getElem!_pos ((Array.range N).map cs) i (by simpa using hi)]
  unfold kaddAll
  rw [hz, kv_foldl_kadd, List.map_map, List.map_map, show kv (Val.zero, Val.zero) = 0 from rfl, List.foldl_map]
  apply gOf_inj
  change gOf ((List.range N).foldl (fun acc l => acc + F256.smul (eqTableK fold)[l]! ((Array.range N).map cs)[l]!)
    0) = _
  rw [gOf_foldl_row, gOf_zero, zero_add]
  unfold consE
  rw [gOf_foldl_cons, gOf_zero, zero_add]
  have e1 : ∀ l ∈ Finset.range N, gOf (eqTableK fold)[l]! * FlockSoundness.execArith.embed
      (fOf ((Array.range N).map cs)[l]!) = ∑ j ∈ Finset.range q, gOf (eqTableK fold)[l]! *
        FlockSoundness.execArith.embed (fOf (rows[j]!)[l]!) *
          FlockSoundness.execArith.embed (fOf (eqTable alpha)[j]!) := by
    intro l hl
    rw [Finset.mem_range] at hl
    rw [getElem!_pos ((Array.range N).map cs) l (by simpa using hl), Array.getElem_map, Array.getElem_range, hcs l hl,
      consAt,
      fOf_rangeSum, map_sum]
    refine (Finset.mul_sum _ _ _).trans (Finset.sum_congr rfl fun j _ => ?_)
    rw [fOf_mul, map_mul]
    ring
  rw [Finset.sum_congr rfl e1, Finset.sum_comm]
  refine Finset.sum_congr rfl fun j hj => ?_
  rw [Finset.mem_range] at hj
  rw [gOf_foldl_row, gOf_zero, zero_add, hrows j hj]
  exact (Finset.sum_mul _ _ _).symm

/-! ## The final weights -/

theorem except_bind_ok {A B : Type} {x : Except String A} {f : A → Except String B} {r : B}
    (h : (x >>= f) = .ok r) : ∃ a, x = .ok a ∧ f a = .ok r := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem foldlM_ok {α β : Type} {f : β → α → Except String β} {g : β → α → β} {P : α → Prop}
    (hf : ∀ b a r, f b a = .ok r → r = g b a ∧ P a) :
    ∀ (l : List α) (b r : β), l.foldlM f b = .ok r → r = l.foldl g b ∧ ∀ a ∈ l, P a
  | [], b, r, h => by
    rw [List.foldlM_nil] at h
    cases h
    exact ⟨rfl, fun _ h => absurd h List.not_mem_nil⟩
  | a :: l, b, r, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨b', hb, h⟩ := except_bind_ok h
    obtain ⟨rfl, hP⟩ := hf b a b' hb
    obtain ⟨rfl, hl⟩ := foldlM_ok hf l _ r h
    exact ⟨rfl, fun x hx => (List.mem_cons.1 hx).elim (fun e => e ▸ hP) (hl x)⟩

theorem foldl_size {α : Type} {g : Array F256 → α → Array F256} {N : Nat} (hg : ∀ r a, (g r a).size = N) :
    ∀ (l : List α) (r : Array F256), r.size = N → (l.foldl g r).size = N
  | [], _, h => h
  | a :: l, r, _ => foldl_size hg l (g r a) (hg r a)

theorem arr_range_foldl {β : Type} (f : β → Nat → β) (init : β) (n : Nat) :
    Array.foldl f init (Array.range n) = (List.range' 0 n).foldl f init := by
  rw [← Array.foldl_toList, Array.toList_range, List.range_eq_range']

/-- **The final weights.** `finalWeights` (rec_algebra.py:395–455) passes the final check's shape checks
(Ligerito.lean:237, 242, 249, 259) and is the verifier's residual weights `resE` (Ligerito.lean:236–270) on any
verifier state with its OOD and consistency contexts. -/
theorem finalWeights_ok {sch : Schedule} {ext : Nat} {initCh : Array F256} {levels : Array (Array F256)}
    {oods : Array OodCtx} {conss : Array ConsCtx} {sws : Array Switched} {res : Array F256}
    (h : finalWeights sch ext initCh levels oods conss sws = .ok res) :
    (origE initCh levels).size + ext = sch.n ∧ (basis sws (origE initCh levels) ext).size = 2 ^ ext ∧
      (∀ o ∈ oods.toList, o.z.size = (oodFixed levels (origE initCh levels) o).1.size + ext) ∧
      (∀ cs ∈ conss.toList, (consFixed levels cs.start).size + ext = cs.cols) ∧ res.size = 2 ^ ext ∧
      ∀ vst : LState, vst.oods = oods → vst.conss = conss → res = resE (basis sws) sch.n initCh levels ext vst := by
  unfold finalWeights at h
  dsimp only at h
  by_cases c1 : (levels.foldl (fun acc l => acc ++ l.extract 1 l.size) initCh).size + ext = sch.n
  swap
  · rw [ite_eq_left c1] at h; cases h
  rw [ite_eq_right (not_not.mpr c1)] at h
  by_cases c2 : (basis sws (levels.foldl (fun acc l => acc ++ l.extract 1 l.size) initCh) ext).size = 2 ^ ext
  swap
  · rw [ite_eq_left c2] at h; cases h
  rw [ite_eq_right (not_not.mpr c2)] at h
  obtain ⟨res₁, h₁, h⟩ := except_bind_ok h
  rw [← Array.foldlM_toList] at h₁ h
  obtain ⟨hres₁, hood⟩ := foldlM_ok (g := fun res o => oodStep ext (oodFixed levels (origE initCh levels) o) o res)
    (P := fun o => o.z.size = (oodFixed levels (origE initCh levels) o).1.size + ext) (by
      intro b o r hb
      obtain ⟨z, beta, sp⟩ := o
      cases sp <;>
      · dsimp only at hb
        split at hb
        · cases hb
        rename_i cz
        cases hb
        exact ⟨rfl, by simpa [oodFixed, origE] using cz⟩) _ _ _ h₁
  obtain ⟨hres, hcons⟩ := foldlM_ok (g := consStep (svTable (sch.n + 1)) ext levels)
    (P := fun cs => (consFixed levels cs.start).size + ext = cs.cols) (by
      intro b cs r hb
      split at hb
      · cases hb
      rename_i cz
      cases hb
      refine ⟨?_, by simpa [consFixed] using cz⟩
      unfold consStep consAdd
      rw [← arr_range_foldl]
      rfl) _ _ _ h
  have hsz : res.size = 2 ^ ext := by
    rw [hres]
    refine foldl_size (fun _ _ => by simp [consStep]) _ _ ?_
    rw [hres₁]
    refine foldl_size (fun _ _ => by simp [oodStep]) _ _ ?_
    simpa using c2
  refine ⟨c1, c2, hood, hcons, hsz, fun vst ho hc => ?_⟩
  rw [hres, hres₁, resE, ho, hc]
  rfl

/-! ## The final check -/

theorem getD_zero (a : Array F128) (i : Nat) : a.getD i Val.zero = a[i]! := by
  rw [Array.getD_eq_getD_getElem?, getElem!_def]
  cases a[i]? <;> rfl

/-- The final check's left-hand side (rec_algebra.py:720–721) is the verifier's (`lhsE`, Ligerito.lean:272). -/
theorem kv_lhs (yr : Array F128) (ext : Nat) (res : Array F256) (hres : res.size = 2 ^ ext) :
    kv (kaddAll ((Array.range res.size).toList.map fun j =>
        kc res[j]! (yr.getD (2 * j) Val.zero, yr.getD (2 * j + 1) Val.zero))) = lhsE yr ext res := by
  unfold kaddAll lhsE
  rw [kv_foldl_kadd, List.map_map, show kv (Val.zero, Val.zero) = 0 from rfl, List.foldl_map, hres,
    ← Array.foldl_toList]
  simp only [Function.comp, kv_kc, getD_zero]
  rfl

/-- What the verifier's final check (Ligerito.lean:235–273; `Refine.finalProg_ok`'s conclusion) asks of a final
state, on V*'s challenges `initCh`, `levels`, `ext` and its running sum and contexts `st`, read in K. -/
structure FinalOK (sch : Schedule) (sws : Array Switched) (yr : Array F128) (initCh : Array F256)
    (levels : Array (Array F256)) (ext : Nat) (st : LSt F128) : Prop where
  orig_size : (origE initCh levels).size + ext = sch.n
  basis_size : (basis sws (origE initCh levels) ext).size = 2 ^ ext
  ood_dim : ∀ o ∈ st.oods.toList, o.z.size = (oodFixed levels (origE initCh levels) o).1.size + ext
  cons_dim : ∀ cs ∈ st.conss.toList, (consFixed levels cs.start).size + ext = cs.cols
  yr_size : yr.size = 2 ^ (ext + 1)
  check : ∀ vst : LState, vst.oods = st.oods → vst.conss = st.conss →
    lhsE yr ext (resE (basis sws) sch.n initCh levels ext vst) = kv st.s

/-! ## The verifier's final check passes -/

theorem lbind_of {α β : Type} {x : Flock.LM α} {f : α → Flock.LM β} {st st₁ : LState} {t t₁ : Transcript}
    {a : α} (hx : x st t = .ok ((a, st₁), t₁)) : (x >>= f) st t = f a st₁ t₁ := by
  change (StateT.bind x f) st t = _
  unfold StateT.bind
  change (StateT.bind (x st) fun p => f p.1 p.2) t = _
  unfold StateT.bind
  rw [hx]
  rfl

theorem lget_bind_eq {β : Type} {f : LState → Flock.LM β} {st : LState} {t : Transcript} :
    (get >>= f) st t = f st st t := rfl

/-- A loop whose every step yields a function of the step, touching neither state, runs as that fold. -/
theorem forIn_yield_of {α σ : Type} (body : α → σ → Flock.LM (ForInStep σ)) (g : σ → α → σ)
    (P : α → Prop) (hb : ∀ a s st tr, P a → body a s st tr = .ok ((.yield (g s a), st), tr)) :
    ∀ (l : List α) (s : σ) (st : LState) (tr : Transcript), (∀ a ∈ l, P a) →
      forIn l s body st tr = .ok ((l.foldl g s, st), tr)
  | [], _, _, _, _ => rfl
  | a :: l, s, st, tr, hP => by
    rw [List.forIn_cons, lbind_of (hb a s st tr (hP a (List.mem_cons_self ..)))]
    exact forIn_yield_of body g P hb l (g s a) st tr fun x hx => hP x (List.mem_cons_of_mem _ hx)

theorem lbind_forIn {α σ β : Type} {body : α → σ → Flock.LM (ForInStep σ)} {f : σ → Flock.LM β}
    (g : σ → α → σ) (P : α → Prop) {l : List α} {s : σ} {st : LState} {tr : Transcript}
    (hb : ∀ a s st tr, P a → body a s st tr = .ok ((.yield (g s a), st), tr)) (hP : ∀ a ∈ l, P a) :
    (forIn l s body >>= f) st tr = f (l.foldl g s) st tr :=
  lbind_of (forIn_yield_of body g P hb l s st tr hP)

/-- **The verifier's final check passes** (`Refine.finalProg_ok`'s converse): on a state that meets its shape checks,
its value check `lhs = s` and its leftover-list check (Ligerito.lean:274), `finalProg` succeeds and changes nothing. -/
theorem finalProg_of {p : LigeritoProof} {basisAt : Array F256 → Nat → Array F256} {n : Nat} {initCh : Array F256}
    {levels : Array (Array F256)} {ext : Nat} {st : LState} {tr : Transcript}
    (h1 : (origE initCh levels).size + ext = n) (h2 : (basisAt (origE initCh levels) ext).size = 2 ^ ext)
    (hood : ∀ o ∈ st.oods.toList, o.z.size = (oodFixed levels (origE initCh levels) o).1.size + ext)
    (hcons : ∀ cs ∈ st.conss.toList, (consFixed levels cs.start).size + ext = cs.cols)
    (hlhs : lhsE p.finalProof.yr ext (resE basisAt n initCh levels ext st) = st.s)
    (hleft : st.ood = p.oodValues.size ∧ st.qn = p.queryNonces.size ∧ st.cn = p.consistencyNonces.size ∧
      st.cbn = p.claimNonces.size) :
    finalProg p basisAt n initCh levels ext st tr = .ok (((), st), tr) := by
  unfold finalProg
  rw [lget_bind_eq]
  dsimp only
  change ((if ((origE initCh levels).size + ext != n) = true then _ else _ : Flock.LM Unit) st tr) = _
  rw [ite_eq_right (by simp [h1])]
  change ((if ((basisAt (origE initCh levels) ext).size != 2 ^ ext) = true then _ else _ : Flock.LM Unit) st tr) = _
  rw [ite_eq_right (by simp [h2]), ← Array.forIn_toList]
  rw [lbind_forIn (fun res o => oodStep ext (oodFixed levels (origE initCh levels) o) o res)
    (fun o => o.z.size = (oodFixed levels (origE initCh levels) o).1.size + ext) ?hb1 hood]
  case hb1 =>
    intro o s st tr hz
    change ((if (o.z.size != (oodFixed levels (origE initCh levels) o).1.size + ext) = true then _ else _ :
      Flock.LM (ForInStep (Array F256))) st tr) = _
    rw [ite_eq_right (by simp [hz])]
    rfl
  rw [← Array.forIn_toList]
  rw [lbind_forIn (consStep (svTable (n + 1)) ext levels)
    (fun cs => (consFixed levels cs.start).size + ext = cs.cols) ?hb2 hcons]
  case hb2 =>
    intro cs s st tr hz
    change ((if ((consFixed levels cs.start).size + ext != cs.cols) = true then _ else _ :
      Flock.LM (ForInStep (Array F256))) st tr) = _
    rw [ite_eq_right (by simp [hz]), lforIn_range_eq, List.forIn_pure_yield_eq_foldl]
    rfl
  change ((if (lhsE p.finalProof.yr ext (resE basisAt n initCh levels ext st) != st.s) = true then _ else _ :
    Flock.LM Unit) st tr) = _
  obtain ⟨e1, e2, e3, e4⟩ := hleft
  rw [ite_eq_right (by simp [hlhs]), ite_eq_right (by simp [e1, e2, e3, e4])]
  rfl

/-- **`FinalOK` is the verifier's final check.** On a verifier state with V*'s OOD and consistency contexts and V*'s
running sum read in K, and with no proof list left over, `finalProg` passes at V*'s challenges, for a proof whose
final `y_r` is V*'s. -/
theorem finalProg_of_finalOK {sch : Schedule} {sws : Array Switched} {yr : Array F128} {initCh : Array F256}
    {levels : Array (Array F256)} {ext : Nat} {st : LSt F128} (hf : FinalOK sch sws yr initCh levels ext st)
    {p : LigeritoProof} (hyr : p.finalProof.yr = yr) {vst : LState} {tr : Transcript} (ho : vst.oods = st.oods)
    (hc : vst.conss = st.conss) (hs : vst.s = kv st.s)
    (hleft : vst.ood = p.oodValues.size ∧ vst.qn = p.queryNonces.size ∧ vst.cn = p.consistencyNonces.size ∧
      vst.cbn = p.claimNonces.size) :
    finalProg p (basis sws) sch.n initCh levels ext vst tr = .ok (((), vst), tr) :=
  finalProg_of hf.orig_size hf.basis_size (ho ▸ hf.ood_dim) (hc ▸ hf.cons_dim)
    (by rw [hyr, hf.check vst ho hc, hs]) hleft

/-- **Ligerito's final check.** When `ligerito` over F returns residuals all zero, its run up to the final check
(`ligeritoRun`) ends in a state on which the verifier's final check passes (`FinalOK`), having read every message
and OOD value (Ligerito.lean:227, 274), and the final weights it returns are the verifier's. -/
theorem ligerito_checks {sch : Schedule} {M : Messages F128} {tIn : F128} {sws : Array Switched} {k k' : Nat}
    {src : CoinSource} {lgr : Array F128} {lg : LgPts}
    (h : ligerito sch M tIn sws k src = .ok ((lgr, lg), k')) (hz : ∀ x ∈ lgr, x = 0) :
    ∃ (ext : Nat) (st : LSt F128),
      (ligeritoRun sch M).run
          { s := kofF tIn, q := ((Val.zero, Val.zero), (Val.zero, Val.zero), (Val.zero, Val.zero)) } k src =
        .ok (((lg.initCh, lg.levels, ext), st), k') ∧
      src.drawnAll k' = true ∧ st.msg = M.msgs.size ∧ st.ood = M.ood.size ∧ lg.oods = st.oods ∧
      lg.conss = st.conss ∧ FinalOK sch sws M.yr lg.initCh lg.levels ext st ∧
      ∀ vst : LState, vst.oods = st.oods → vst.conss = st.conss →
        lg.res = resE (basis sws) sch.n lg.initCh lg.levels ext vst := by
  unfold ligerito at h
  obtain ⟨⟨⟨initCh, levels, ext⟩, st⟩, k₁, h1, h⟩ := cm_bind h
  obtain ⟨a, k₂, ha, h⟩ := cm_bind h
  cases ha
  obtain ⟨b, k₃, hb, h⟩ := cm_bind h
  cases hb
  by_cases cd : src.drawnAll k₁ = true
  swap
  · rw [ite_eq_left (by simpa using cd)] at h
    obtain ⟨_, _, h0, _⟩ := cm_bind h
    cases h0
  rw [ite_eq_right (by simpa using cd)] at h
  dsimp only at h
  by_cases cm : st.msg = M.msgs.size ∧ st.ood = M.ood.size
  swap
  · rw [ite_eq_left (by omega)] at h
    obtain ⟨_, _, h0, _⟩ := cm_bind h
    cases h0
  rw [ite_eq_right (by omega)] at h
  obtain ⟨res, k₆, h6, h⟩ := cm_bind h
  obtain ⟨res', hfw, h6⟩ := liftE_ok h6
  cases h6
  by_cases cy : M.yr.size = 2 <<< ext
  swap
  · rw [ite_eq_left cy] at h
    obtain ⟨_, _, h0, _⟩ := cm_bind h
    cases h0
  rw [ite_eq_right (not_not.mpr cy)] at h
  cases cm_pure h
  obtain ⟨c1, c2, hood, hcons, hsz, hres⟩ := finalWeights_ok hfw
  have h0 : ∀ i, i < 2 → (#[(kadd (kaddAll ((Array.range res.size).toList.map fun j =>
      kc res[j]! (M.yr.getD (2 * j) Val.zero, M.yr.getD (2 * j + 1) Val.zero))) st.s).1,
      (kadd (kaddAll ((Array.range res.size).toList.map fun j =>
      kc res[j]! (M.yr.getD (2 * j) Val.zero, M.yr.getD (2 * j + 1) Val.zero))) st.s).2] : Array F128)[i]! = 0 :=
    fun i hi => hz _ (by rw [getElem!_pos _ i (by simpa using hi)]; exact Array.getElem_mem _)
  have e1 := f128_eq_of_add_eq_zero (h0 0 (by omega))
  have e2 := f128_eq_of_add_eq_zero (h0 1 (by omega))
  refine ⟨ext, st, h1, cd, cm.1, cm.2, rfl, rfl, ⟨c1, c2, hood, hcons, ?_, fun vst ho hc => ?_⟩, fun vst ho hc =>
    hres vst ho hc⟩
  · rw [cy, Nat.shiftLeft_eq, pow_succ, mul_comm]
  · rw [← hres vst ho hc, ← kv_lhs M.yr ext res hsz]
    exact congrArg₂ F256.mk e1 e2

/-! ## Ligerito at a rep -/

/-- **Ligerito at a rep.** When `GfResiduals`' residuals at w, on the S `structureOf` builds and the v `verifierRows`
gives at the rep's coins, are all zero, Ligerito's run from the `T` w holds ends in a state on which the verifier's
final check passes, on the messages w holds (`decoded`), the ring switches' switched claims and the challenges and
contexts at the rep's coins (`info.lg`). -/
theorem ligerito_of_residuals {sh : Shape} {w comb : Array F128} {cons : Nat → Nat → F128}
    {rounds : Array (Array F128)} {extras : Array Claim} {v : Array F128} {info : Info Lin}
    {kappa : Array (Array (Array F128))} {S : GfS} {rs : Array F128}
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : w.size = S.nw) (hc : comb.size = 64)
    (hacc : ∀ r < 2, valOf sh w comb cons (.acc1 r) + valOf sh w comb cons (.acc0 r) = kapSum kappa r cons)
    (hres : residuals S w v = .ok rs) (hzero : ∀ x ∈ rs, x = 0) :
    ∃ (k₀ ext : Nat) (st : LSt F128) (k' : Nat),
      (ligeritoRun sh.sched (decoded sh w comb cons)).run
          { s := kofF (valOf sh w comb cons .T),
            q := ((Val.zero, Val.zero), (Val.zero, Val.zero), (Val.zero, Val.zero)) } k₀ (CoinSource.ofRounds rounds) =
        .ok (((info.lg.initCh, info.lg.levels, ext), st), k') ∧
      (CoinSource.ofRounds rounds).drawnAll k' = true ∧ st.msg = (decoded sh w comb cons).msgs.size ∧
      st.ood = (decoded sh w comb cons).ood.size ∧ info.lg.oods = st.oods ∧ info.lg.conss = st.conss ∧
      FinalOK sh.sched info.rs.sws (decoded sh w comb cons).yr info.lg.initCh info.lg.levels ext st := by
  obtain ⟨rs', k, hrun, hres'⟩ := residuals_eq_run w comb cons hrows hS hw hc hacc
  rw [hres] at hres'
  cases hres'
  obtain ⟨-, r0, r1, rsr, lgr, k1, k2, k3, -, -, -, h4, hrs⟩ := run_stages hrun
  have hT : (decoded sh w comb cons).tAtom.getD (info.map (Lin.eval (valOf sh w comb cons))).rs.T =
      valOf sh w comb cons .T := by
    simp [decoded, Messages.map, symbolic, Lin.eval_var]
  rw [hT] at h4
  obtain ⟨ext, st, hl, hd, hm, ho, hlo, hlc, hf, -⟩ :=
    ligerito_checks h4 (fun x hx => hzero x (by rw [hrs]; simp [hx]))
  exact ⟨k3, ext, st, k, hl, hd, hm, ho, hlo, hlc, hf⟩

end Flock.RecAlgebra
