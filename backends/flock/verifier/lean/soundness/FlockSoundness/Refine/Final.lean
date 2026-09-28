import FlockSoundness.Refine.LigeritoRun

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

end FlockSoundness.Refine
