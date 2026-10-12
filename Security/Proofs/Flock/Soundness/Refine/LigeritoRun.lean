import Proofs.Flock.Soundness.Refine.Ligerito

/-!
# A run of the executable's Ligerito is a run of the model's (PROTOCOL.md §13.3)

`ligerito_refines`: when the executable's Ligerito verifier accepts, the model's compiled Ligerito (`ligeritoC`), played
with the proof's messages (`ligMsgs`) and the rounds the executable consumed, ends in a result

* whose caps are the proof's recursive caps, decoded by `capD`;
* whose every request is a query level whose Merkle check the executable passed (`ReqOK`);
* whose verdict at the openings `o` is the model's final check (§13.6) on the executable's final state, where the
  executable's own final check (`finalProg`, the verifier's last step, verbatim) passed.

That the executable's final check makes the model's hold, and that a passed Merkle check is the model's path check
(§14), are separate steps.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {D : Type}

/-! ## The executable's final check, verbatim -/

section
open Flock
/-- The executable Ligerito verifier's final check (PROTOCOL.md §13.6), as `Flock.ligerito` runs it after the level
loop: its text is the verifier's, so a run of the verifier's last step is a run of this. -/
def finalProg (p : LigeritoProof) (basisAt : Array F256 → Nat → Array F256) (n : Nat) (initCh : Array F256)
    (levels : Array (Array F256)) (ext : Nat) : LM Unit := do
    let st ← get
    let orig := levels.foldl (fun acc l => acc ++ l.extract 1 l.size) initCh
    if orig.size + ext != n then throw "ligerito: |orig| + ext ≠ n"
    let heads := levels.map (fun l => l[0]!)
    let prodFrom := fun (a : Nat) => (heads.extract a heads.size).foldl (fun acc c => acc * Flock.cff c) (1 : F256)
    let pi0 := prodFrom 0
    let bres := basisAt orig ext
    if bres.size != 2 ^ ext then throw "ligerito: basis residual length"
    let mut res : Array F256 := bres.map (· * pi0)
    let sv := Flock.svTable (n + 1)
    for o in st.oods do
      let (fixed, scale) := match o.split with
        | none => (orig, pi0)
        | some s => (levels.extract s levels.size |>.foldl (fun acc l => if acc.isEmpty then l else acc ++ l.extract 1 l.size) #[], prodFrom (s + 1))
      if o.z.size != fixed.size + ext then throw "ligerito: ood dimension"
      let head := (Array.range fixed.size).foldl (fun acc t => acc * ((1 : F256) + F256.ofF o.z[t]! + fixed[t]!)) (1 : F256)
      let c := (head * scale).smul o.beta
      res := (Array.range (2 ^ ext)).map fun y =>
        let tail := (Array.range ext).foldl (fun acc j =>
          let zj := o.z[fixed.size + j]!
          acc * (if (y >>> j) % 2 == 1 then zj else (1 : F128) + zj)) (1 : F128)
        res[y]! + c.smul tail
    for cs in st.conss do
      let fixed := (levels.extract cs.start levels.size).foldl (fun acc l => acc ++ l.extract 1 l.size) #[]
      if fixed.size + ext != cs.cols then throw "ligerito: consistency dimension"
      let ea := Flock.eqTable cs.alpha
      let scale := prodFrom cs.start
      let mut add : Array F256 := Array.replicate (2 ^ ext) 0
      for j in [0:cs.pos.size] do
        let wh := Flock.whats sv cs.pos[j]! cs.cols
        let pj := (Array.range fixed.size).foldl (fun acc i => acc * ((1 : F256) + fixed[i]! * F256.ofF ((1 : F128) + wh[i]!))) (1 : F256)
        let coef := pj.smul (ea[j]!)
        add := (Array.range (2 ^ ext)).map fun y =>
          let tj := (Array.range ext).foldl (fun acc i => if (y >>> i) % 2 == 1 then acc * wh[fixed.size + i]! else acc) (1 : F128)
          add[y]! + coef.smul tj
      res := (Array.range (2 ^ ext)).map fun y => res[y]! + (add[y]! * scale).smul cs.beta
    let yr := p.finalProof.yr
    let lhs := (Array.range (2 ^ ext)).foldl (fun acc y => acc + res[y]! * ⟨yr[2 * y]!, yr[2 * y + 1]!⟩) (0 : F256)
    if lhs != st.s then throw "ligerito: final check failed"
    if st.ood != p.oodValues.size || st.qn != p.queryNonces.size || st.cn != p.consistencyNonces.size
        || st.cbn != p.claimNonces.size then throw "ligerito: leftover proof lists"
end

/-! ## Tapes, requests and the loop's invariant -/

/-- An executable consistency context as the model's. -/
def consOf (c : Flock.ConsCtx) : Model.ConsCtx 𝔽 := ⟨c.cols, c.pos.toList, c.alpha.toList.map fOf, fOf c.beta, c.start⟩

/-- The proof's clear residual as the model's last message (§13.6). -/
def yrOf (p : LigeritoProof) (ext : ℕ) : Fin (2 ^ (ext + 1)) → 𝔽 := fun i => fOf p.finalProof.yr[i.val]!

/-- The messages of the level loop's iterations `i, …, i + left − 1` (§13.3): each folds with four messages; a non-last
iteration then sends the next level's cap, two out-of-domain values with their intro messages, and the queries'
message; the last sends the clear residual. `msg` and `ood` are the cursors into the proof's lists. -/
def recMsgs (p : LigeritoProof) (capD : Array ByteArray → Cap D) (ext : ℕ) : ℕ → ℕ → ℕ → ℕ → List Msg
  | 0, _, _, _ => []
  | 1, _, msg, _ => kMsgs p msg 4 ++ [⟨Fin (2 ^ (ext + 1)) → 𝔽, yrOf p ext⟩]
  | left + 2, i, msg, ood => kMsgs p msg 4 ++ ⟨Cap D, capD p.recursiveCaps[i + 1]!⟩ ::
      [fMsg p.oodValues[ood]!, kMsg p.messages[msg + 4]!, fMsg p.oodValues[ood + 1]!, kMsg p.messages[msg + 5]!] ++
      [kMsg p.messages[msg + 6]!] ++ recMsgs p capD ext (left + 1) (i + 1) (msg + 7) (ood + 2)

/-- **The messages of a Ligerito run** (§13.3): the level-0 out-of-domain value, the first message and the `k0` folding
messages, the level-1 cap, two out-of-domain values with their intro messages, the level-0 queries' message, and the
level loop's. -/
def ligMsgs (p : LigeritoProof) (capD : Array ByteArray → Cap D) (sch : Flock.Schedule) : List Msg :=
  fMsg p.oodValues[0]! :: kMsg p.messages[0]! :: kMsgs p 1 sch.k0 ++ ⟨Cap D, capD p.recursiveCaps[0]!⟩ ::
    [fMsg p.oodValues[1]!, kMsg p.messages[sch.k0 + 1]!, fMsg p.oodValues[2]!, kMsg p.messages[sch.k0 + 2]!] ++
    [kMsg p.messages[sch.k0 + 3]!] ++ recMsgs p capD sch.ext (sch.levels.size - 1) 0 (sch.k0 + 4) 3

/-- Level `ℓ`'s cap in the executable: the table's at level 0, the proof's `(ℓ − 1)`-th recursive cap above. -/
def capAt (cap : Array ByteArray) (p : LigeritoProof) (ℓ : ℕ) : Array ByteArray :=
  if ℓ = 0 then cap else p.recursiveCaps[ℓ - 1]!

/-- Level `ℓ`'s opened rows, siblings and salts in the proof, `R` the last level. -/
def openAt (p : LigeritoProof) (R ℓ : ℕ) : Array (Array F128) × Array ByteArray × Array ByteArray :=
  if ℓ = 0 then (p.initialProof.rows, p.initialProof.siblings, p.initialProof.salts)
  else if ℓ < R then
    ((p.recursiveProofs[ℓ - 1]!).rows, (p.recursiveProofs[ℓ - 1]!).siblings, (p.recursiveProofs[ℓ - 1]!).salts)
  else (p.finalProof.rows, p.finalProof.siblings, p.finalProof.salts)

/-- A model request whose Merkle check the executable passed (§14): the request is level `ℓ`'s queries, it folds the
level's lanes, its paths have the level's length, and the rows the proof opens there verify against the level's cap. -/
def ReqOK (ms : MerkleScheme) (sch : Flock.Schedule) (cap : Array ByteArray) (p : LigeritoProof) (req : Req) : Prop :=
  req.level < sch.levels.size ∧ req.lanes = (sch.levels.getD req.level default).lanes ∧
    req.depth = pathLen (levelOf (sch.levels.getD req.level default)) ∧
    merkleCheck ms (capAt cap p req.level) (sch.levels.getD req.level default).cap
      (sch.levels.getD req.level default).d req.positions.toArray (openAt p (sch.levels.size - 1) req.level).1
      (openAt p (sch.levels.size - 1) req.level).2.1 (openAt p (sch.levels.size - 1) req.level).2.2
      (2 ^ req.lanes) = .ok ()

/-- **The level loop's invariant** before iteration `i` (§13.3): the loop's variables `v = (splitDim, prevCap, levels,
ext)`, the executable's state `stE` and transcript `t`, and the model's state `stM`, which at the openings `o` has the
executable's claim and polynomial. -/
structure LoopRel (ms : MerkleScheme) (sch : Flock.Schedule) (cap : Array ByteArray) (p : LigeritoProof)
    (capD : Array ByteArray → Cap D) (o : Opens 𝔽 D) (i : ℕ) (v : ℕ × Array ByteArray × Array (Array F256) × ℕ)
    (stE : LState) (t : Transcript) (stM : LigStateC 𝔽 𝕂 D) : Prop where
  wf : RoundsWF t.rounds
  s : stM.s o = gOf stE.s
  q : stM.q o = quadOf stE.q
  split : v.1 = (sch.levels.getD (i + 1) default).cols + 4
  prev : v.2.1 = p.recursiveCaps[i]!
  size : v.2.2.1.size = i
  lanes : ∀ l ∈ v.2.2.1.toList, l.size = 4
  levels : stM.levels = v.2.2.1.toList.map fun l => l.toList.map gOf
  oods : stM.oods = stE.oods.toList.map oodOf
  cons : stM.cons = stE.conss.toList.map consOf
  pos : ∀ cs ∈ stE.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size
  level : stM.level = i + 1
  caps : stM.caps = (List.range (i + 1)).map fun j => capD p.recursiveCaps[j]!
  reqs : ∀ req ∈ stM.reqs, ReqOK ms sch cap p req

/-- What the level loop ends with (§13.3, §13.6): `R` levels of four challenges, `ext` the schedule's, the clear
residual's length checked, and the model's result: the final check on the executable's final state, the proof's caps,
and requests that passed. -/
structure LoopFinal (ms : MerkleScheme) (sch : Flock.Schedule) (cap : Array ByteArray) (p : LigeritoProof)
    (capD : Array ByteArray → Cap D) (o : Opens 𝔽 D) (b : ℕ → 𝔽) (ini : List 𝕂)
    (v : ℕ × Array ByteArray × Array (Array F256) × ℕ) (stE : LState) (pend : PendL 𝔽 D) : Prop where
  size : v.2.2.1.size = sch.levels.size - 1
  lanes : ∀ l ∈ v.2.2.1.toList, l.size = 4
  ext : v.2.2.2 = sch.ext
  yr : p.finalProof.yr.size = 2 ^ (sch.ext + 1)
  verdict : pend.verdict o = Model.finalCheck execArith (sch.m - 7) b ini
    (v.2.2.1.toList.map fun l => l.toList.map gOf) (stE.oods.toList.map oodOf) (stE.conss.toList.map consOf) sch.ext
    (yrOf p sch.ext) (gOf stE.s)
  caps : pend.caps = (List.range (sch.levels.size - 1)).map fun j => capD p.recursiveCaps[j]!
  reqs : ∀ req ∈ pend.reqs, ReqOK ms sch cap p req
  pos : ∀ cs ∈ stE.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size

/-! ## Helpers -/

theorem sampleVec_ok {n : ℕ} (hn : n ≠ 0) {t t₁ : Transcript} {cs : Array F128} (h : sampleVec n t = .ok (cs, t₁)) :
    (t.rounds[t.k]!).n = n ∧ cs = (t.rounds[t.k]!).coins ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 := by
  obtain ⟨cs', t₂, h₁, h₂⟩ := bind_ok (show (sampleVec n >>= pure) t = _ by rw [bind_pure]; exact h)
  obtain ⟨-, hn', hcs, hr, hk⟩ := squeeze_ok hn h₁
  obtain ⟨rfl, rfl⟩ := pure_ok h₂
  exact ⟨hn', hcs, hr, hk⟩

theorem absorbF_ok {v : F128} {t t₁ : Transcript} {u : Unit} (h : absorbF v t = .ok (u, t₁)) : Keeps t t₁ := by
  obtain ⟨t₂, k, h'⟩ := absorbF_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h)
  obtain ⟨-, rfl⟩ := pure_ok h'
  exact k

theorem absorbBytes_ok {bs : ByteArray} {t t₁ : Transcript} {u : Unit} (h : absorbBytes bs t = .ok (u, t₁)) :
    Keeps t t₁ := by
  obtain ⟨t₂, k, h'⟩ := absorbBytes_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h)
  obtain ⟨-, rfl⟩ := pure_ok h'
  exact k

theorem lforIn_range_eq {σ : Type} (n : ℕ) (s : σ) (g : ℕ → σ → LM (ForInStep σ)) :
    (forIn [:n] s g : LM σ) = forIn (List.range' 0 n) s g := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', range_size]

/-- A passed Merkle check opened one row per position, of the level's lane count. -/
theorem merkleCheck_rows {ms : MerkleScheme} {cap : Array ByteArray} {c d : ℕ} {pos : Array ℕ}
    {rows : Array (Array F128)} {sibs salts : Array ByteArray} {lanes : ℕ}
    (h : merkleCheck ms cap c d pos rows sibs salts lanes = .ok ()) :
    rows.size = pos.size ∧ ∀ j < pos.size, rows[j]!.size = lanes := by
  unfold merkleCheck at h
  split at h
  · cases h
  rename_i h₁
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at h₁
  split at h
  · cases h
  split at h
  · cases h
  rename_i hf
  refine ⟨h₁.1.1.2, fun j hj => ?_⟩
  have := List.find?_eq_none.1 hf j (List.mem_range.2 hj)
  have h2 : opens ms cap c d lanes pos[j]! rows[j]! (salts.getD j ByteArray.empty) (sibsOf sibs (d - c) j) = true := by
    cases h3 : opens ms cap c d lanes pos[j]! rows[j]! (salts.getD j ByteArray.empty) (sibsOf sibs (d - c) j)
    · rw [h3] at this; exact absurd rfl this
    · rfl
  simp only [opens, Bool.and_eq_true, decide_eq_true_eq] at h2
  exact h2.1.1.1

theorem LevelSeam.q_ne_zero {lv : Flock.Level} (hl : LevelSeam lv) : lv.q ≠ 0 := by
  intro h0
  have h := hl.2.2.2.1
  rw [h0] at h
  exact absurd h (by decide)

theorem LevelSeam.q_ne_one {lv : Flock.Level} (hl : LevelSeam lv) : lv.q ≠ 1 := by
  intro h1
  have h := hl.2.2.2.1
  rw [h1] at h
  exact absurd h (by decide)

theorem positions_size (lv : Flock.Level) (hl : LevelSeam lv) (coins : Array F128) :
    (Flock.positions lv coins).size = lv.q := by
  rw [← Array.length_toList, ← positions_eq lv hl coins]
  unfold Model.positions
  rw [List.length_map, List.length_zipIdx]
  have := slotsOf_length (strata (levelOf lv).queries (levelOf lv).logLen)
  simp only [slotsOf] at this
  rw [this, strata_sum]
  rfl

theorem consValueO_congr (α : List 𝔽) (ch : List 𝕂) (rows rows' : ℕ → List 𝔽) (pos : List ℕ)
    (h : ∀ j < pos.length, rows j = rows' j) :
    Model.consValueO execArith α ch rows pos = Model.consValueO execArith α ch rows' pos := by
  unfold Model.consValueO
  refine congrArg List.sum (List.map_congr_left fun pj hm => ?_)
  obtain ⟨q, j⟩ := pj
  have := (List.mem_zipIdx hm).2.1
  simp only at this ⊢
  rw [h j (by omega)]

/-- A level's queries fit its consistency coins' table: `q ≤ 2^(⌊log2 (q − 1)⌋ + 1)`. -/
theorem pos_le_alpha (lv : Flock.Level) (hl : LevelSeam lv) (cw cα : Array F128)
    (hcα : cα.size = Nat.log2 (lv.q - 1) + 1) : (Flock.positions lv cw).size ≤ 2 ^ cα.size := by
  rw [positions_size lv hl cw, hcα]
  have := @Nat.lt_log2_self (lv.q - 1)
  have := hl.q_ne_zero
  omega

theorem conss_push_pos {cs : Array Flock.ConsCtx} {c : Flock.ConsCtx}
    (h : ∀ x ∈ cs.toList, x.pos.size ≤ 2 ^ x.alpha.size) (hc : c.pos.size ≤ 2 ^ c.alpha.size) :
    ∀ x ∈ (cs.push c).toList, x.pos.size ≤ 2 ^ x.alpha.size := by
  intro x hx
  rw [Array.toList_push, List.mem_append, List.mem_singleton] at hx
  rcases hx with hx | rfl
  · exact h x hx
  · exact hc

/-- **The consistency value at the openings** (§13.3 step 10): the model's, reading the openings of level `ℓ`, is the
executable's `consE` on the proof's rows, when the openings are those rows and the Merkle check passed. -/
theorem qbE_eq (r ℓ : ℕ) (lv : Flock.Level) (hl : LevelSeam lv) (ch : Array F256) (hch : ch.size = lv.lanes)
    (o : Opens 𝔽 D) (rows : Array (Array F128)) (hrows : ∀ j < lv.q, rows[j]!.size = 2 ^ lv.lanes)
    (ho : ∀ j < lv.q, (o r ℓ j).1 = rows[j]!.toList.map fOf) (cw cα : Array F128)
    (hcα : cα.size = Nat.log2 (lv.q - 1) + 1) :
    Model.consValueO execArith (List.ofFn (vecOf (ceilLog2 (levelOf lv).queries) cα)) (ch.toList.map gOf)
        (fun j => (o r ℓ j).1)
        (Model.positions execArith (levelOf lv).queries (levelOf lv).logLen (vecOf (levelOf lv).queries cw)) =
      gOf (consE rows lv.q cα ch) := by
  have hq := hl.q_ne_zero
  have hc : cα.size = ceilLog2 (levelOf lv).queries := hcα.trans hl.2.2.2.1
  rw [show List.ofFn (vecOf (ceilLog2 (levelOf lv).queries) cα) = cα.toList.map fOf from
    ofFn_vecOf _ cα hc, show Model.positions execArith (levelOf lv).queries (levelOf lv).logLen
      (vecOf (levelOf lv).queries cw) = (Flock.positions lv cw).toList from positions_eq lv hl cw]
  have hlen : (Flock.positions lv cw).toList.length = lv.q := by
    rw [Array.length_toList, positions_size lv hl cw]
  rw [consValueO_congr _ _ _ (fun j => rows[j]!.toList.map fOf) _ fun j hj => ho j (hlen ▸ hj)]
  refine consValue_eq rows lv.q cα ch _ hlen (fun j hj => by rw [hrows j hj, hch]) ?_
  rw [hcα]
  have := @Nat.lt_log2_self (lv.q - 1)
  omega

theorem lmodify_bind' {β : Type} {g : LState → LState} {f : PUnit → LM β} {st st' : LState} {t t' : Transcript}
    {b : β} (h : (modify g >>= f) st t = .ok ((b, st'), t')) : ∃ st₁, st₁ = g st ∧ f () st₁ t = .ok ((b, st'), t') :=
  ⟨_, rfl, lmodify_bind h⟩

theorem consumed_cons (rs : Array Round) {k k' : ℕ} (h : k < k') :
    consumed rs k k' = (rs[k]!).coins :: consumed rs (k + 1) k' := by
  rw [consumed_append rs (Nat.le_succ k) h, consumed_succ]; rfl

theorem consumed_add (rs : Array Round) (k n : ℕ) :
    consumed rs k (k + n) = (List.range n).map fun j => (rs[k + j]!).coins := by
  unfold consumed
  rw [show k + n - k = n by omega, List.range'_eq_map_range, List.map_map]
  rfl

/-- Absorbing words keeps the rounds and the Ligerito state. -/
theorem forIn_absorbF_keeps : ∀ (ws : List F128) {st st' : LState} {t t' : Transcript} {u : PUnit},
    forIn ws PUnit.unit (fun w (_ : PUnit) => lift (absorbF w) >>= fun _ => pure (ForInStep.yield PUnit.unit)) st t =
      .ok ((u, st'), t') → st' = st ∧ Keeps t t'
  | [], st, st', t, t', u, h => by
      simp only [List.forIn_nil] at h
      obtain ⟨-, rfl, rfl⟩ := lpure_ok h
      exact ⟨rfl, rfl, rfl⟩
  | w :: ws, st, st', t, t', u, h => by
      rw [List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨_, t₂, h₃, h₁⟩ := lift_bind h₁
      have k₃ := absorbF_ok h₃
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
      obtain ⟨rfl, k⟩ := forIn_absorbF_keeps ws h₂
      exact ⟨rfl, k.1.trans k₃.1, k.2.trans k₃.2⟩

theorem drop_levels (sch : Flock.Schedule) (i : ℕ) (hi : i < sch.levels.size) :
    (sch.levels.toList.map levelOf).drop i =
      levelOf (sch.levels.getD i default) :: (sch.levels.toList.map levelOf).drop (i + 1) := by
  rw [List.drop_eq_getElem_cons (by simpa using hi)]
  simp [Array.getD, hi]

/-- **One level's queries** (§13.3 steps 7–11 and the recursive rounds): the model's `qbOut` on the executable's coins
has, at the openings, the executable's claim and polynomial updates; it records the executable's consistency context,
and a request that passed the executable's Merkle check. -/
theorem qbOut_rel (ms : MerkleScheme) (sch : Flock.Schedule) (cap : Array ByteArray) (p : LigeritoProof) (r ℓ : ℕ)
    (hℓ : ℓ < sch.levels.size) (hl : LevelSeam (sch.levels.getD ℓ default)) (ch : Array F256)
    (hch : ch.size = (sch.levels.getD ℓ default).lanes) (start : ℕ) (stM : LigStateC 𝔽 𝕂 D)
    (hlevel : stM.level = ℓ) (o : Opens 𝔽 D)
    (ho : ∀ j < (sch.levels.getD ℓ default).q,
      (o r ℓ j).1 = ((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf)
    (cw cα cβ : Array F128) (hcα : cα.size = Nat.log2 ((sch.levels.getD ℓ default).q - 1) + 1)
    (hmc : merkleCheck ms (capAt cap p ℓ) (sch.levels.getD ℓ default).cap (sch.levels.getD ℓ default).d
      (Flock.positions (sch.levels.getD ℓ default) cw) (openAt p (sch.levels.size - 1) ℓ).1
      (openAt p (sch.levels.size - 1) ℓ).2.1 (openAt p (sch.levels.size - 1) ℓ).2.2
      (2 ^ (sch.levels.getD ℓ default).lanes) = .ok ())
    (withMessage : Bool) (m : F256 × F256) :
    (qbOut r (levelOf (sch.levels.getD ℓ default)) (ch.toList.map gOf) start stM cw cα cβ withMessage m).s o =
        stM.s o + gOf (consE (openAt p (sch.levels.size - 1) ℓ).1 (sch.levels.getD ℓ default).q cα ch) *
          execArith.embed (fOf (cβ.getD 0 0)) ∧
      (qbOut r (levelOf (sch.levels.getD ℓ default)) (ch.toList.map gOf) start stM cw cα cβ withMessage m).q o =
        (if withMessage then (stM.q o).addScaled (Model.fromMsg (gOf m.1, gOf m.2)
          (gOf (consE (openAt p (sch.levels.size - 1) ℓ).1 (sch.levels.getD ℓ default).q cα ch)))
          (execArith.embed (fOf (cβ.getD 0 0))) else stM.q o) ∧
      (qbOut r (levelOf (sch.levels.getD ℓ default)) (ch.toList.map gOf) start stM cw cα cβ withMessage m).cons =
        stM.cons ++ [consOf ⟨(sch.levels.getD ℓ default).cols, Flock.positions (sch.levels.getD ℓ default) cw, cα,
          cβ.getD 0 0, start⟩] ∧
      (∀ req ∈ (qbOut r (levelOf (sch.levels.getD ℓ default)) (ch.toList.map gOf) start stM cw cα cβ withMessage
        m).reqs, req ∈ stM.reqs ∨ ReqOK ms sch cap p req) ∧
      (qbOut r (levelOf (sch.levels.getD ℓ default)) (ch.toList.map gOf) start stM cw cα cβ withMessage m).levels =
        stM.levels ∧
      (qbOut r (levelOf (sch.levels.getD ℓ default)) (ch.toList.map gOf) start stM cw cα cβ withMessage m).oods =
        stM.oods ∧
      (qbOut r (levelOf (sch.levels.getD ℓ default)) (ch.toList.map gOf) start stM cw cα cβ withMessage m).level =
        stM.level ∧
      (qbOut r (levelOf (sch.levels.getD ℓ default)) (ch.toList.map gOf) start stM cw cα cβ withMessage m).caps =
        stM.caps := by
  have hq := hl.q_ne_zero
  obtain ⟨-, hrows'⟩ := merkleCheck_rows hmc
  have hpos := positions_size _ hl cw
  have hE : Model.consValueO execArith (List.ofFn (vecOf (ceilLog2 (levelOf (sch.levels.getD ℓ default)).queries) cα))
      (ch.toList.map gOf) (fun j => (o r stM.level j).1)
      (Model.positions execArith (levelOf (sch.levels.getD ℓ default)).queries
        (levelOf (sch.levels.getD ℓ default)).logLen (vecOf (levelOf (sch.levels.getD ℓ default)).queries cw)) =
      gOf (consE (openAt p (sch.levels.size - 1) ℓ).1 (sch.levels.getD ℓ default).q cα ch) := by
    rw [hlevel]
    exact qbE_eq r ℓ _ hl ch hch o _ (fun j hj => hrows' j (hpos ▸ hj)) ho cw cα hcα
  have hc : cα.size = ceilLog2 (levelOf (sch.levels.getD ℓ default)).queries := hcα.trans hl.2.2.2.1
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl, rfl, rfl⟩
  · simp only [qbOut]
    rw [hE, getElem!_eq_getD, mul_comm]
  · cases withMessage
    · rfl
    · simp only [qbOut, ite_true]
      rw [hE, getElem!_eq_getD]
  · simp only [qbOut]
    rw [show List.ofFn (vecOf (ceilLog2 (levelOf (sch.levels.getD ℓ default)).queries) cα) = cα.toList.map fOf from
      ofFn_vecOf _ cα hc, show Model.positions execArith (levelOf (sch.levels.getD ℓ default)).queries
        (levelOf (sch.levels.getD ℓ default)).logLen (vecOf (levelOf (sch.levels.getD ℓ default)).queries cw) =
        (Flock.positions (sch.levels.getD ℓ default) cw).toList from positions_eq _ hl cw]
    simp only [consOf, getElem!_eq_getD]
    rfl
  · intro req hreq
    simp only [qbOut, List.mem_append, List.mem_singleton] at hreq
    rcases hreq with h | rfl
    · exact .inl h
    · right
      rw [show Model.positions execArith (levelOf (sch.levels.getD ℓ default)).queries
        (levelOf (sch.levels.getD ℓ default)).logLen (vecOf (levelOf (sch.levels.getD ℓ default)).queries cw) =
        (Flock.positions (sch.levels.getD ℓ default) cw).toList from positions_eq _ hl cw]
      refine ⟨by rw [hlevel]; exact hℓ, by simp [hlevel, hch], by simp [hlevel], ?_⟩
      simp only [hlevel, List.length_map, Array.length_toList, hch, Array.toArray_toList]
      exact hmc

/-! ## The model's prefix -/

/-- **The model's Ligerito before the level loop** (§13.3 steps 2–11), composed from its parts: the level-0
out-of-domain sample, the first message and the `k0` folds (`hfold`), the level-1 cap, its two out-of-domain samples
(`htwo`), the level-0 queries with their message, and any run of the recursive rounds from the resulting state
(`hrec`). -/
theorem ligeritoC_model (r : ℕ) (sch : Accounting.Schedule) (T : 𝔽) (b : ℕ → 𝔽) (lv₀ lv₁ : Accounting.Level)
    (rest : List Accounting.Level) (hl : sch.levels = lv₀ :: lv₁ :: rest)
    (z₀ c₀ : Array F128) (hz₀ : z₀.size = sch.m - 7) (hm7 : sch.m - 7 ≠ 1) (hc₀ : c₀.size = 1) (y₀ : F128)
    (m₀ : F256 × F256)
    {msF : List Msg} {csF : List (Array F128)} {ini : List 𝕂} {sF : Opens 𝔽 D → 𝕂} {qF : Opens 𝔽 D → Model.Quad 𝕂}
    (hfold : Plays fOf (Model.foldRoundsC execArith sch.k0
      (fun _ => execArith.embed T + execArith.embed (fOf c₀[0]!) * execArith.embed (fOf y₀))
      (fun _ => Model.fromMsg (gOf m₀.1, gOf m₀.2)
        (execArith.embed T + execArith.embed (fOf c₀[0]!) * execArith.embed (fOf y₀)))) msF csF (ini, sF, qF))
    (C₁ : Cap D) {msO : List Msg} {csO : List (Array F128)} {sT : Opens 𝔽 D → 𝕂} {qT : Opens 𝔽 D → Model.Quad 𝕂}
    {ctxs : List (Model.OodCtx 𝔽)}
    (htwo : Plays fOf (Model.twoOodC execArith (sch.m - 7 - sch.k0 + 1) 0 sF qF) msO csO (sT, qT, ctxs))
    (cw cα cβ : Array F128) (hcw : cw.size = lv₀.queries) (hcα : cα.size = ceilLog2 lv₀.queries)
    (hcβ : cβ.size = 1) (hq1 : lv₀.queries ≠ 1) (hα1 : ceilLog2 lv₀.queries ≠ 1) (mq : F256 × F256)
    {msR : List Msg} {csR : List (Array F128)} {pend : PendL 𝔽 D}
    (hrec : Plays fOf (Model.recursiveRoundsC execArith r (sch.m - 7) b ini sch.ext (lv₁ :: rest)
      { qbOut r lv₀ ini 0 ⟨sT, qT, sch.m - 7 - sch.k0 + 1, [],
          ⟨List.ofFn (vecOf (sch.m - 7) z₀), fOf c₀[0]!, none⟩ :: ctxs, [], 0, [], [], lv₀⟩ cw cα cβ true mq with
        level := 1, caps := [C₁], prevLevel := lv₁ }) msR csR pend) :
    Plays fOf (Model.ligeritoC execArith r sch T b)
      (fMsg y₀ :: kMsg m₀ :: msF ++ ⟨Cap D, C₁⟩ :: msO ++ [kMsg mq] ++ msR)
      (z₀ :: c₀ :: csF ++ csO ++ [cw, cα, cβ] ++ csR) pend := by
  unfold Model.ligeritoC
  rw [hl]
  have hqb := queryAndBatchC_plays r lv₀ ini 0
    (⟨sT, qT, sch.m - 7 - sch.k0 + 1, [], ⟨List.ofFn (vecOf (sch.m - 7) z₀), fOf c₀[0]!, none⟩ :: ctxs, [], 0, [], [],
      lv₀⟩ : LigStateC 𝔽 𝕂 D) cw cα cβ hcw hcα hcβ hq1 hα1 true mq
  simp only [ite_true] at hqb
  refine Plays.bind' (ms₂ := fMsg y₀ :: kMsg m₀ :: msF ++ ⟨Cap D, C₁⟩ :: msO ++ [kMsg mq] ++ msR)
    (cs₂ := c₀ :: csF ++ csO ++ [cw, cα, cβ] ++ csR) (Plays.drawVec z₀ hz₀ hm7) ?_ (by simp) (by simp)
  refine Plays.bind' (ms₂ := kMsg m₀ :: msF ++ ⟨Cap D, C₁⟩ :: msO ++ [kMsg mq] ++ msR)
    (cs₂ := c₀ :: csF ++ csO ++ [cw, cα, cβ] ++ csR) (Plays.recv _ (fOf y₀)) ?_ (by simp [fMsg]) (by simp)
  refine Plays.bind' (ms₂ := kMsg m₀ :: msF ++ ⟨Cap D, C₁⟩ :: msO ++ [kMsg mq] ++ msR)
    (cs₂ := csF ++ csO ++ [cw, cα, cβ] ++ csR) (Plays.drawOne c₀ hc₀) ?_ (by simp) (by simp)
  refine Plays.bind' (ms₂ := msF ++ ⟨Cap D, C₁⟩ :: msO ++ [kMsg mq] ++ msR)
    (cs₂ := csF ++ csO ++ [cw, cα, cβ] ++ csR) (Plays.recv _ (gOf m₀.1, gOf m₀.2)) ?_ (by simp [kMsg]) (by simp)
  refine Plays.bind' (ms₂ := ⟨Cap D, C₁⟩ :: msO ++ [kMsg mq] ++ msR) (cs₂ := csO ++ [cw, cα, cβ] ++ csR) hfold ?_
    (by simp) (by simp)
  refine Plays.bind' (ms₂ := msO ++ [kMsg mq] ++ msR) (cs₂ := csO ++ [cw, cα, cβ] ++ csR) (Plays.recv _ C₁) ?_
    (by simp) (by simp)
  refine Plays.bind' (ms₂ := [kMsg mq] ++ msR) (cs₂ := [cw, cα, cβ] ++ csR) htwo ?_ (by simp) (by simp)
  exact Plays.bind' hqb hrec rfl rfl

/-! ## The run -/

set_option maxHeartbeats 4000000 in
/-- **Ligerito's rounds refine the model's** (§13.3). When the executable's Ligerito verifier accepts, from a transcript
whose rounds are well formed, on a schedule whose dimensions and levels are the model's (`DimSeam`, `LevelSeam`) and
whose levels' consistency coins are not one word (`ceilLog2 q ≠ 1`, which `fast100_alpha` gives), the model's compiled
Ligerito, played with the proof's messages and the rounds consumed, ends in a result whose caps are the proof's, whose
requests passed their Merkle checks, and whose verdict at the openings `o` (the proof's opened rows, `ho`) is the
model's final check on the executable's final state; and the executable's final check passed on that state. -/
theorem ligerito_refines (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (r : ℕ) (b : ℕ → 𝔽)
    (capD : Array ByteArray → Cap D) (o : Opens 𝔽 D) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1)
    (ho : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q,
      (o r ℓ j).1 = ((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf)
    {t t' : Transcript} (hwf : RoundsWF t.rounds) (h : Flock.ligerito ms sch T cap p basisAt t = .ok ((), t')) :
    ∃ (pend : PendL 𝔽 D) (initCh : Array F256) (levels : Array (Array F256)) (stF stF' : LState) (tF : Transcript),
      tF.rounds = t.rounds ∧ t.k ≤ tF.k ∧
      Plays fOf (Model.ligeritoC execArith r (schedOf sch) (fOf T) b) (ligMsgs p capD sch)
        (consumed t.rounds t.k tF.k) pend ∧
      pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capD p.recursiveCaps[j]!) ∧
      (∀ req ∈ pend.reqs, ReqOK ms sch cap p req) ∧
      initCh.size = sch.k0 ∧ levels.size = sch.levels.size - 1 ∧ (∀ l ∈ levels.toList, l.size = 4) ∧
      p.finalProof.yr.size = 2 ^ (sch.ext + 1) ∧ (∀ cs ∈ stF.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size) ∧
      finalProg p basisAt sch.n initCh levels sch.ext stF tF = .ok (((), stF'), t') ∧
      pend.verdict o = Model.finalCheck execArith (sch.m - 7) b (initCh.toList.map gOf)
        (levels.toList.map fun l => l.toList.map gOf) (stF.oods.toList.map oodOf) (stF.conss.toList.map consOf)
        sch.ext (yrOf p sch.ext) (gOf stF.s) := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  unfold Flock.ligerito at h
  obtain ⟨hcap, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨⟨u, stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  have hc01 := hcols 0 (by omega)
  have hn3 : 3 ≤ sch.n := by omega
  obtain ⟨z, t₅, h₅, h₄⟩ := lift_bind h₄
  obtain ⟨hn₅, rfl, hr₅, hk₅⟩ := sampleVec_ok (by omega) h₅
  obtain ⟨y₀, hy₀, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have k₆ := absorbF_ok h₆
  obtain ⟨t₇, k₇, h₄⟩ := claimPow_bind h₄
  obtain ⟨bc, t₈, h₈, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, hm₀, k₉, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨initCh, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  have er₉ : t₉.rounds = t.rounds := by rw [k₉.2, hr₈, k₇.2, k₆.2, hr₅, k₃.2, k₂.2, k₁.2]
  have ek₉ : t₉.k = t.k + 2 := by rw [k₉.1, hk₈, k₇.1, k₆.1, hk₅, k₃.1, k₂.1, k₁.1]
  have hs₀ : execArith.embed (fOf T) + execArith.embed (fOf (t₇.rounds[t₇.k]!).coins[0]!) * execArith.embed (fOf y₀) =
      gOf (F256.ofF T + (F256.ofF y₀).smul ((t₇.rounds[t₇.k]!).coins.getD 0 0)) := by
    rw [gOf_add, gOf_smul, gOf_ofF, gOf_ofF, getElem!_eq_getD, mul_comm]; rfl
  have hq₀ : Model.fromMsg (gOf m₀.1, gOf m₀.2) (execArith.embed (fOf T) +
      execArith.embed (fOf (t₇.rounds[t₇.k]!).coins[0]!) * execArith.embed (fOf y₀)) =
      quadOf (Poly.ofMsg m₀ (F256.ofF T + (F256.ofF y₀).smul ((t₇.rounds[t₇.k]!).coins.getD 0 0))) := by
    rw [quadOf_ofMsg, hs₀]
  obtain ⟨hr₁₀, hk₁₀, hmsg₁₀, hood₁₀, -, -, -, hoods₁₀, hconss₁₀, hsz₁₀, -, sF, qF, hfold, hsF, hqF⟩ :=
    fold_refines p o sch.k0 0 #[] initCh _ st₁₀ t₉ t₁₀
      (fun _ => execArith.embed (fOf T) + execArith.embed (fOf (t₇.rounds[t₇.k]!).coins[0]!) * execArith.embed (fOf y₀))
      (fun _ => Model.fromMsg (gOf m₀.1, gOf m₀.2) (execArith.embed (fOf T) +
        execArith.embed (fOf (t₇.rounds[t₇.k]!).coins[0]!) * execArith.embed (fOf y₀))) hs₀ hq₀ (er₉ ▸ hwf) h₁₀
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨hnA, hnB, yA, mA, t₁₂, hyA, hmA, hr₁₂, hk₁₂, h₄⟩ := oodIntro_bind (by omega) h₄
  obtain ⟨hnC, hnD, yB, mB, t₁₃, hyB, hmB, hr₁₃, hk₁₃, h₄⟩ := oodIntro_bind (by omega) h₄
  have hlv0 := hlv 0 (by omega)
  obtain ⟨hnQ, hnQ', hmc, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevel_bind hlv0.q_ne_zero h₄
  obtain ⟨mq, t₁₅, hmq, k₁₅, h₄⟩ := nextMsg_bind h₄
  obtain ⟨t₁₆, k₁₆, h₄⟩ := claimPow_bind h₄
  obtain ⟨bq, t₁₇, h₁₇, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₁₇, rfl, hr₁₇, hk₁₇⟩ := sample_ok h₁₇
  obtain ⟨stL, hstL, h₄⟩ := lmodify_bind' h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have hfin : finalProg p basisAt sch.n initCh out.2.2.1 out.2.2.2 st₁₈ t₁₈ = .ok ((u, stF), t') := hcont
  simp only [Array.size_empty, Array.extract_size] at hfold
  have er₁₁ : t₁₁.rounds = t.rounds := by rw [k₁₁.2, hr₁₀, er₉]
  have ek₁₁ : t₁₁.k = t.k + 2 + sch.k0 := by rw [k₁₁.1, hk₁₀, ek₉]
  have er₁₂ : t₁₂.rounds = t.rounds := hr₁₂.trans er₁₁
  have ek₁₂ : t₁₂.k = t.k + 4 + sch.k0 := by rw [hk₁₂, ek₁₁]; omega
  have er₁₃ : t₁₃.rounds = t.rounds := hr₁₃.trans er₁₂
  have ek₁₃ : t₁₃.k = t.k + 6 + sch.k0 := by rw [hk₁₃, ek₁₂]; omega
  have er₁₆ : t₁₆.rounds = t.rounds := by rw [k₁₆.2, k₁₅.2, hr₁₄, er₁₃]
  have ek₁₆ : t₁₆.k = t.k + 8 + sch.k0 := by rw [k₁₆.1, k₁₅.1, hk₁₄, ek₁₃]; omega
  have er₁₇ : t₁₇.rounds = t.rounds := hr₁₇.trans er₁₆
  have ek₁₇ : t₁₇.k = t.k + 9 + sch.k0 := by rw [hk₁₇, ek₁₆]; omega
  obtain ⟨sT, qT, htwo, hsT, hqT⟩ := twoOodC_plays o (sch.m - 7 - sch.k0 + 1) 0 sF qF _ _ hsF hqF
    (t₁₁.rounds[t₁₁.k]!).coins (t₁₁.rounds[t₁₁.k + 1]!).coins (t₁₂.rounds[t₁₂.k]!).coins
    (t₁₂.rounds[t₁₂.k + 1]!).coins (by rw [RoundsWF.get (er₁₁ ▸ hwf), hnA, hn7])
    (by rw [RoundsWF.get (er₁₁ ▸ hwf), hnB]) (by rw [RoundsWF.get (er₁₂ ▸ hwf), hnC, hn7])
    (by rw [RoundsWF.get (er₁₂ ▸ hwf), hnD]) (by omega) yA yB mA mB
  obtain ⟨stM₀, hstM₀⟩ : ∃ x : LigStateC 𝔽 𝕂 D, x =
      { qbOut r (levelOf (sch.levels.getD 0 default)) (initCh.toList.map gOf) 0
          ⟨sT, qT, sch.m - 7 - sch.k0 + 1, [],
            ⟨List.ofFn (vecOf (sch.m - 7) (t₃.rounds[t₃.k]!).coins), fOf (t₇.rounds[t₇.k]!).coins[0]!, none⟩ ::
              [oodOf ⟨(t₁₁.rounds[t₁₁.k]!).coins, (t₁₁.rounds[t₁₁.k + 1]!).coins.getD 0 0, some 0⟩,
                oodOf ⟨(t₁₂.rounds[t₁₂.k]!).coins, (t₁₂.rounds[t₁₂.k + 1]!).coins.getD 0 0, some 0⟩],
            [], 0, [], [], levelOf (sch.levels.getD 0 default)⟩
          (t₁₃.rounds[t₁₃.k]!).coins (t₁₃.rounds[t₁₃.k + 1]!).coins (t₁₆.rounds[t₁₆.k]!).coins true mq with
        level := 1, caps := [capD p.recursiveCaps[0]!], prevLevel := levelOf (sch.levels.getD 1 default) } :=
    ⟨_, rfl⟩
  simp only at hoods₁₀ hconss₁₀ hood₁₀ hmsg₁₀
  have h0 : 0 < sch.levels.size := Nat.lt_of_lt_of_le (by decide) h2
  have hz₀ : (t₃.rounds[t₃.k]!).coins.size = sch.m - 7 := by
    rw [RoundsWF.get ((k₃.2.trans (k₂.2.trans k₁.2)) ▸ hwf), hn₅, hn7]
  have hopen0 : openAt p (sch.levels.size - 1) 0 =
      (p.initialProof.rows, p.initialProof.siblings, p.initialProof.salts) := by simp [openAt]
  have hmc0 : merkleCheck ms (capAt cap p 0) (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d
      (Flock.positions (sch.levels.getD 0 default) (t₁₃.rounds[t₁₃.k]!).coins) (openAt p (sch.levels.size - 1) 0).1
      (openAt p (sch.levels.size - 1) 0).2.1 (openAt p (sch.levels.size - 1) 0).2.2
      (2 ^ (sch.levels.getD 0 default).lanes) = .ok () := by
    rw [show capAt cap p 0 = cap by simp [capAt], hopen0]; exact hmc
  obtain ⟨stQ, hstQ⟩ : ∃ x : LigStateC 𝔽 𝕂 D, x = ⟨sT, qT, sch.m - 7 - sch.k0 + 1, [],
      ⟨List.ofFn (vecOf (sch.m - 7) (t₃.rounds[t₃.k]!).coins), fOf (t₇.rounds[t₇.k]!).coins[0]!, none⟩ ::
        [oodOf ⟨(t₁₁.rounds[t₁₁.k]!).coins, (t₁₁.rounds[t₁₁.k + 1]!).coins.getD 0 0, some 0⟩,
          oodOf ⟨(t₁₂.rounds[t₁₂.k]!).coins, (t₁₂.rounds[t₁₂.k + 1]!).coins.getD 0 0, some 0⟩],
      [], 0, [], [], levelOf (sch.levels.getD 0 default)⟩ := ⟨_, rfl⟩
  rw [← hstQ] at hstM₀
  obtain ⟨hqs, hqq, hqc, hqr, hql, hqo, hqlev, hqcaps⟩ := qbOut_rel ms sch cap p r 0 h0 hlv0 initCh
    (by rw [hsz₁₀, hl0]; simp) 0 stQ (by rw [hstQ]) o (ho 0 h0) (t₁₃.rounds[t₁₃.k]!).coins
    (t₁₃.rounds[t₁₃.k + 1]!).coins (t₁₆.rounds[t₁₆.k]!).coins (by rw [RoundsWF.get (er₁₃ ▸ hwf), hnQ']) hmc0 true mq
  rw [hopen0] at hqs hqq
  have hrel₀ : LoopRel ms sch cap p capD o 0 (sch.n - sch.k0 + 1, p.recursiveCaps[0]!, #[], 0) stL t₁₇ stM₀ := by
    rw [hstM₀]
    refine ⟨er₁₇ ▸ hwf, ?_, ?_, ?_, rfl, rfl, by simp, ?_, ?_, ?_, ?_, rfl, rfl, ?_⟩
    · dsimp only
      rw [hqs, hstL, hstQ]
      dsimp only
      rw [hsT]
      simp only [gOf_add, gOf_smul]
      rfl
    · dsimp only
      rw [hqq, hstL, hstQ]
      dsimp only
      rw [hqT]
      simp only [ite_true, quadOf_addScaled, quadOf_ofMsg]
      rfl
    · show sch.n - sch.k0 + 1 = _
      rw [← hc0, ← hc01]
    · dsimp only
      rw [hql, hstQ]
      rfl
    · dsimp only
      rw [hqo, hstQ, hstL]
      dsimp only
      rw [hoods₁₀, show List.ofFn (vecOf (sch.m - 7) (t₃.rounds[t₃.k]!).coins) =
        (t₃.rounds[t₃.k]!).coins.toList.map fOf from ofFn_vecOf _ _ hz₀]
      simp [oodOf, getElem!_eq_getD]
    · dsimp only
      rw [hqc, hstQ, hstL]
      dsimp only
      rw [hconss₁₀, hc0]
      simp
    · rw [hstL]
      dsimp only
      rw [hconss₁₀]
      exact conss_push_pos (fun x hx => by simp at hx) (pos_le_alpha _ hlv0 _ _
        (by rw [RoundsWF.get (er₁₃ ▸ hwf), hnQ']))
    · intro req hreq
      rcases hqr req hreq with h | h
      · rw [hstQ] at h; cases h
      · exact h
  have hL := loop_compose _ (sch.levels.size - 1)
    (fun i stM => Model.recursiveRoundsC execArith r (sch.m - 7) b (initCh.toList.map gOf) sch.ext
      ((sch.levels.toList.map levelOf).drop (i + 1)) stM)
    (fun i v stE tE stM => LoopRel ms sch cap p capD o i v stE tE stM)
    (fun i stE => recMsgs p capD sch.ext (sch.levels.size - 1 - i) i stE.msg stE.ood)
    (fun v stE _ pend => LoopFinal ms sch cap p capD o b (initCh.toList.map gOf) v stE pend)
    ?hstep ?hlast (sch.levels.size - 1) 0 _ _ _ out st₁₈ t₁₈ stM₀ (by omega) (by omega) hrel₀ hloop
  case hstep =>
    clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1 ho
    intro i v stE tE bv stE' tE' stM hi hrel hb
    obtain ⟨hwfE, hsE, hqE, hsplit, hprev, hsize, hlanes, hlevels, hoods, hcons, hposE, hlevel, hcaps, hreqs⟩ := hrel
    have hc1 := hcols (i + 1) (by omega)
    have hl1 := hl4 (i + 1) (by omega) (by omega)
    have hl2 := hl4 (i + 1 + 1) (by omega) (by omega)
    dsimp only at hb
    rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
    obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
    rw [lforIn_range_eq] at hf
    obtain ⟨hr₁, hk₁, hmsg₁, hood₁, -, -, -, hoods₁, hconss₁, hsz₁, -, sM₁, qM₁, hfoldM, hsM₁, hqM₁⟩ :=
      fold_refines p o 4 0 #[] chs stE st₁ tE t₁ stM.s stM.q hsE hqE hwfE hf
    simp only [Array.size_empty, Array.extract_size] at hfoldM
    rw [ite_eq_left hi] at hb
    obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
    have k₂ := absorbBytes_ok h₂
    obtain ⟨hnA, hnB, yA, mA, t₃, hyA, hmA, hr₃, hk₃, hb⟩ := oodIntro_bind (by omega) hb
    obtain ⟨hnC, hnD, yB, mB, t₄, hyB, hmB, hr₄, hk₄, hb⟩ := oodIntro_bind (by omega) hb
    have hlvi := hlv (i + 1) (by omega)
    obtain ⟨hnQ, hnQ', hmc, t₅, hr₅, hk₅, hb⟩ := queryLevel_bind hlvi.q_ne_zero hb
    obtain ⟨mq, t₆, hmq, k₆, hb⟩ := nextMsg_bind hb
    obtain ⟨t₇, k₇, hb⟩ := claimPow_bind hb
    obtain ⟨bq, t₈, h₈, hb⟩ := lift_bind hb
    obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
    obtain ⟨stN, hstN, hb⟩ := lmodify_bind' hb
    rw [ite_eq_right (show ¬ v.1 - 4 + 1 < 4 by omega)] at hb
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
    have er₂ : t₂.rounds = tE.rounds := k₂.2.trans hr₁
    have ek₂ : t₂.k = tE.k + 4 := by rw [k₂.1, hk₁]
    have er₃ : t₃.rounds = tE.rounds := hr₃.trans er₂
    have ek₃ : t₃.k = tE.k + 6 := by rw [hk₃, ek₂]
    have er₄ : t₄.rounds = tE.rounds := hr₄.trans er₃
    have ek₄ : t₄.k = tE.k + 8 := by rw [hk₄, ek₃]
    have er₇ : t₇.rounds = tE.rounds := by rw [k₇.2, k₆.2, hr₅, er₄]
    have ek₇ : t₇.k = tE.k + 10 := by rw [k₇.1, k₆.1, hk₅, ek₄]
    have er₈ : tE'.rounds = tE.rounds := hr₈.trans er₇
    have ek₈ : tE'.k = tE.k + 11 := by rw [hk₈, ek₇]
    have hsz : chs.size = 4 := by simpa using hsz₁
    have hlen : (stM.levels ++ [chs.toList.map gOf]).length = i + 1 := by simp [hlevels, hsize]
    have hdim : (levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
        (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes = v.1 - 4 + 1 := by
      show (sch.levels.getD (i + 1 + 1) default).cols + (sch.levels.getD (i + 1 + 1) default).lanes = _
      omega
    obtain ⟨sT, qT, htwo, hsT, hqT⟩ := twoOodC_plays o (v.1 - 4 + 1) (i + 1) sM₁ qM₁ _ _ hsM₁ hqM₁
      (t₂.rounds[t₂.k]!).coins (t₂.rounds[t₂.k + 1]!).coins (t₃.rounds[t₃.k]!).coins (t₃.rounds[t₃.k + 1]!).coins
      (by rw [RoundsWF.get (er₂ ▸ hwfE), hnA]) (by rw [RoundsWF.get (er₂ ▸ hwfE), hnB])
      (by rw [RoundsWF.get (er₃ ▸ hwfE), hnC]) (by rw [RoundsWF.get (er₃ ▸ hwfE), hnD]) (by omega) yA yB mA mB
    have hcapAt : capAt cap p (i + 1) = v.2.1 := by rw [hprev]; simp [capAt]
    have hopen : openAt p (sch.levels.size - 1) (i + 1) =
        (p.recursiveProofs[i]!.rows, p.recursiveProofs[i]!.siblings, p.recursiveProofs[i]!.salts) := by
      simp [openAt, hi]
    have hmc' : merkleCheck ms (capAt cap p (i + 1)) (sch.levels.getD (i + 1) default).cap
        (sch.levels.getD (i + 1) default).d (Flock.positions (sch.levels.getD (i + 1) default) (t₄.rounds[t₄.k]!).coins)
        (openAt p (sch.levels.size - 1) (i + 1)).1 (openAt p (sch.levels.size - 1) (i + 1)).2.1
        (openAt p (sch.levels.size - 1) (i + 1)).2.2 (2 ^ (sch.levels.getD (i + 1) default).lanes) = .ok () := by
      rw [hcapAt, hopen]; exact hmc
    set stQ : LigStateC 𝔽 𝕂 D := { stM with
        s := sT, q := qT, levels := stM.levels ++ [chs.toList.map gOf],
        oods := stM.oods ++ [oodOf ⟨(t₂.rounds[t₂.k]!).coins, (t₂.rounds[t₂.k + 1]!).coins.getD 0 0, some (i + 1)⟩,
          oodOf ⟨(t₃.rounds[t₃.k]!).coins, (t₃.rounds[t₃.k + 1]!).coins.getD 0 0, some (i + 1)⟩] } with hstQ
    have hq := qbOut_rel ms sch cap p r (i + 1) (by omega) hlvi chs (by rw [hsz, hl1])
      (stM.levels ++ [chs.toList.map gOf]).length stQ hlevel o (ho (i + 1) (by omega)) (t₄.rounds[t₄.k]!).coins
      (t₄.rounds[t₄.k + 1]!).coins (t₇.rounds[t₇.k]!).coins (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ']) hmc' true mq
    rw [hopen] at hq
    obtain ⟨hqs, hqq, hqc, hqr, hql, hqo, hqlev, hqcaps⟩ := hq
    set qb := qbOut r (levelOf (sch.levels.getD (i + 1) default)) (chs.toList.map gOf)
      (stM.levels ++ [chs.toList.map gOf]).length stQ (t₄.rounds[t₄.k]!).coins (t₄.rounds[t₄.k + 1]!).coins
      (t₇.rounds[t₇.k]!).coins true mq with hqb
    have hmsgA := getElem!_of_getElem? hmA
    have hoodA := getElem!_of_getElem? hyA
    have hmsgB := getElem!_of_getElem? hmB
    have hoodB := getElem!_of_getElem? hyB
    have hmsgQ := getElem!_of_getElem? hmq
    simp only [hmsg₁, hood₁, Nat.add_assoc, Nat.reduceAdd] at hmsgA hoodA hmsgB hoodB hmsgQ
    subst hstN
    refine ⟨_, { qb with
        level := stM.level + 1, caps := stM.caps ++ [capD p.recursiveCaps[i + 1]!],
        prevLevel := levelOf (sch.levels.getD (i + 1 + 1) default) },
      kMsgs p stE.msg 4 ++ ⟨Cap D, capD p.recursiveCaps[i + 1]!⟩ ::
        [fMsg p.oodValues[stE.ood]!, kMsg p.messages[stE.msg + 4]!, fMsg p.oodValues[stE.ood + 1]!,
          kMsg p.messages[stE.msg + 5]!] ++ [kMsg p.messages[stE.msg + 6]!],
      rfl, ?rel, er₈, by omega, ?tape, ?plays⟩
    case rel =>
      refine ⟨er₈ ▸ hwfE, ?_, ?_, ?_, rfl, by simp [hsize], ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · show qb.s o = _
        rw [hqs]
        simp only [hstQ, hsT, gOf_add, gOf_smul]
        rfl
      · show qb.q o = _
        rw [hqq]
        simp only [ite_true, hstQ, hqT, quadOf_addScaled, quadOf_ofMsg]
        rfl
      · show v.1 - 4 + 1 = _
        omega
      · intro l hl
        simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hl
        rcases hl with hl | rfl
        · exact hlanes l hl
        · exact hsz
      · show qb.levels = _
        rw [hql, hstQ]
        simp [hlevels]
      · show qb.oods = _
        rw [hqo, hstQ]
        simp [hoods, hoods₁]
      · show qb.cons = _
        rw [hqc, hstQ, hlen]
        simp [hcons, hconss₁, hsplit]
      · dsimp only
        rw [hconss₁]
        exact conss_push_pos hposE (pos_le_alpha _ hlvi _ _ (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ']))
      · show stM.level + 1 = _
        rw [hlevel]
      · show stM.caps ++ _ = _
        rw [hcaps, List.range_succ (n := i + 1), List.map_append]
        rfl
      · intro req hreq
        rcases hqr req hreq with h | h
        · exact hreqs req h
        · exact h
    case tape =>
      obtain ⟨j, hj⟩ : ∃ j, sch.levels.size - 1 - i = j + 2 := ⟨sch.levels.size - 1 - i - 2, by omega⟩
      rw [hj, show sch.levels.size - 1 - (i + 1) = j + 1 by omega]
      simp only [recMsgs, hmsg₁, hood₁, Nat.add_assoc, Nat.reduceAdd, List.append_assoc, List.cons_append]
    case plays =>
      intro msR csR pend hrec
      rw [drop_levels sch (i + 1) (by omega), drop_levels sch (i + 1 + 1) (by omega)]
      rw [drop_levels sch (i + 1 + 1) (by omega)] at hrec
      have hfold' : Plays fOf (Model.foldRoundsC execArith (levelOf (sch.levels.getD (i + 1) default)).lanes stM.s stM.q)
          (kMsgs p stE.msg 4) (consumed tE.rounds tE.k t₁.k) (List.map gOf chs.toList, sM₁, qM₁) := by
        rw [show (levelOf (sch.levels.getD (i + 1) default)).lanes = 4 from hl1]; exact hfoldM
      have htwo' : Plays fOf (Model.twoOodC execArith ((levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
          (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes) (stM.levels ++ [chs.toList.map gOf]).length sM₁ qM₁)
          [fMsg yA, kMsg mA, fMsg yB, kMsg mB]
          [t₂.rounds[t₂.k]!.coins, t₂.rounds[t₂.k + 1]!.coins, t₃.rounds[t₃.k]!.coins, t₃.rounds[t₃.k + 1]!.coins]
          (sT, qT, [oodOf ⟨(t₂.rounds[t₂.k]!).coins, (t₂.rounds[t₂.k + 1]!).coins.getD 0 0, some (i + 1)⟩,
            oodOf ⟨(t₃.rounds[t₃.k]!).coins, (t₃.rounds[t₃.k + 1]!).coins.getD 0 0, some (i + 1)⟩]) := by
        rw [hdim, hlen]; exact htwo
      have hmain := iterNL_model r (sch.m - 7) b (initCh.toList.map gOf) sch.ext
        (levelOf (sch.levels.getD (i + 1) default)) (levelOf (sch.levels.getD (i + 1 + 1) default))
        ((sch.levels.toList.map levelOf).drop (i + 1 + 1 + 1)) stM hfold' (capD p.recursiveCaps[i + 1]!) htwo'
        (t₄.rounds[t₄.k]!).coins (t₄.rounds[t₄.k + 1]!).coins (t₇.rounds[t₇.k]!).coins
        (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ]; rfl)
        (by rw [RoundsWF.get (er₄ ▸ hwfE), hnQ']; exact hlvi.2.2.2.1)
        (by rw [RoundsWF.get (er₇ ▸ hwfE), hn₈]) hlvi.q_ne_one (hα1 (i + 1) (by omega)) mq hrec
      have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k t₁.k ++
          [t₂.rounds[t₂.k]!.coins, t₂.rounds[t₂.k + 1]!.coins, t₃.rounds[t₃.k]!.coins, t₃.rounds[t₃.k + 1]!.coins] ++
          [t₄.rounds[t₄.k]!.coins, t₄.rounds[t₄.k + 1]!.coins, t₇.rounds[t₇.k]!.coins] := by
        rw [er₂, ek₂, er₃, ek₃, er₄, ek₄, er₇, ek₇, ek₈, hk₁,
          consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 11 by omega),
          show tE.k + 11 = tE.k + 4 + 7 by omega, consumed_add tE.rounds (tE.k + 4) 7]
        simp [List.range_succ, Nat.add_assoc]
      rw [hcs, hmsgA, hoodA, hmsgB, hoodB, hmsgQ]
      simp only [List.append_assoc, List.cons_append, List.nil_append] at hmain ⊢
      exact hmain
  case hlast =>
    clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1 ho
    intro i v stE tE bv stE' tE' stM hi hrel hb
    obtain ⟨hwfE, hsE, hqE, hsplit, hprev, hsize, hlanes, hlevels, hoods, hcons, hposE, hlevel, hcaps, hreqs⟩ := hrel
    have hl1 := hl4 (i + 1) (by omega) (by omega)
    dsimp only at hb
    rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
    obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
    rw [lforIn_range_eq] at hf
    obtain ⟨hr₁, hk₁, hmsg₁, hood₁, -, -, -, hoods₁, hconss₁, hsz₁, -, sM₁, qM₁, hfoldM, hsM₁, hqM₁⟩ :=
      fold_refines p o 4 0 #[] chs stE st₁ tE t₁ stM.s stM.q hsE hqE hwfE hf
    simp only [Array.size_empty, Array.extract_size] at hfoldM
    rw [ite_eq_right (show ¬ i + 1 < sch.levels.size - 1 by omega)] at hb
    have hb := lget_bind hb
    by_cases hmc₀ : (st₁.msg != p.messages.size) = true
    · rw [ite_eq_left hmc₀] at hb
      obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
      cases h0
    rw [ite_eq_right hmc₀] at hb
    by_cases hyr : (p.finalProof.yr.size != 2 ^ (v.1 - 4 + 1)) = true
    · rw [ite_eq_left hyr] at hb
      obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
      cases h0
    rw [ite_eq_right hyr] at hb
    obtain ⟨_, st₂, t₂, h₂, hb₂⟩ := lbind_ok hb
    clear hb
    rw [← Array.forIn_toList] at h₂
    obtain ⟨rfl, k₂⟩ := forIn_absorbF_keeps _ h₂
    have hb := hb₂
    clear hb₂
    have hlvR := hlv (sch.levels.size - 1) (by omega)
    obtain ⟨hnQ, hnQ', hmc, t₃, hr₃, hk₃, hb⟩ := queryLevel_bind hlvR.q_ne_zero hb
    obtain ⟨t₄, k₄, hb⟩ := claimPow_bind hb
    obtain ⟨bq, t₅, h₅, hb⟩ := lift_bind hb
    obtain ⟨-, hn₅, rfl, hr₅, hk₅⟩ := sample_ok h₅
    obtain ⟨stN, hstN, hb⟩ := lmodify_bind' hb
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
    have er₂ : t₂.rounds = tE.rounds := k₂.2.trans hr₁
    have ek₂ : t₂.k = tE.k + 4 := by rw [k₂.1, hk₁]
    have er₄ : t₄.rounds = tE.rounds := by rw [k₄.2, hr₃, er₂]
    have ek₄ : t₄.k = tE.k + 6 := by rw [k₄.1, hk₃, ek₂]
    have er₅ : tE'.rounds = tE.rounds := hr₅.trans er₄
    have ek₅ : tE'.k = tE.k + 7 := by rw [hk₅, ek₄]
    have hsz : chs.size = 4 := by simpa using hsz₁
    rw [hi] at hsplit hl1
    have hlen : (stM.levels ++ [chs.toList.map gOf]).length = sch.levels.size - 1 := by
      simp [hlevels, hsize]; omega
    have hyr' : p.finalProof.yr.size = 2 ^ (sch.ext + 1) := by
      simp only [bne_iff_ne, ne_eq, not_not] at hyr
      rw [hyr, hext, hsplit, Nat.add_sub_cancel]
    have hcapAt : capAt cap p (sch.levels.size - 1) = v.2.1 := by
      rw [hprev]; simp [capAt, show sch.levels.size - 1 ≠ 0 by omega, show sch.levels.size - 1 - 1 = i by omega]
    have hopen : openAt p (sch.levels.size - 1) (sch.levels.size - 1) =
        (p.finalProof.rows, p.finalProof.siblings, p.finalProof.salts) := by
      simp [openAt, show sch.levels.size - 1 ≠ 0 by omega]
    have hmc' : merkleCheck ms (capAt cap p (sch.levels.size - 1)) (sch.levels.getD (sch.levels.size - 1) default).cap
        (sch.levels.getD (sch.levels.size - 1) default).d
        (Flock.positions (sch.levels.getD (sch.levels.size - 1) default) (t₂.rounds[t₂.k]!).coins)
        (openAt p (sch.levels.size - 1) (sch.levels.size - 1)).1 (openAt p (sch.levels.size - 1) (sch.levels.size - 1)).2.1
        (openAt p (sch.levels.size - 1) (sch.levels.size - 1)).2.2
        (2 ^ (sch.levels.getD (sch.levels.size - 1) default).lanes) = .ok () := by
      rw [hcapAt, hopen]; exact hmc
    set stQ : LigStateC 𝔽 𝕂 D := { stM with s := sM₁, q := qM₁, levels := stM.levels ++ [chs.toList.map gOf] }
      with hstQ
    have hq := qbOut_rel ms sch cap p r (sch.levels.size - 1) (by omega) hlvR chs (by rw [hsz, hl1])
      (stM.levels ++ [chs.toList.map gOf]).length stQ (by rw [hstQ]; exact hlevel.trans hi) o
      (ho (sch.levels.size - 1) (by omega)) (t₂.rounds[t₂.k]!).coins (t₂.rounds[t₂.k + 1]!).coins
      (t₄.rounds[t₄.k]!).coins (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ']) hmc' false (0, 0)
    rw [hopen] at hq
    obtain ⟨hqs, hqq, hqc, hqr, hql, hqo, hqlev, hqcaps⟩ := hq
    have hfold' : Plays fOf (Model.foldRoundsC execArith (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes
        stM.s stM.q) (kMsgs p stE.msg 4) (consumed tE.rounds tE.k t₁.k) (List.map gOf chs.toList, sM₁, qM₁) := by
      rw [show (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes = 4 from hl1]; exact hfoldM
    have hmain := iterLast_model r (sch.m - 7) b (initCh.toList.map gOf) sch.ext
      (levelOf (sch.levels.getD (sch.levels.size - 1) default)) stM hfold' (yrOf p sch.ext)
      (t₂.rounds[t₂.k]!).coins (t₂.rounds[t₂.k + 1]!).coins (t₄.rounds[t₄.k]!).coins
      (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ]; rfl)
      (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ']; exact hlvR.2.2.2.1)
      (by rw [RoundsWF.get (er₄ ▸ hwfE), hn₅]) hlvR.q_ne_one (hα1 (sch.levels.size - 1) (by omega))
    have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k t₁.k ++
        [t₂.rounds[t₂.k]!.coins, t₂.rounds[t₂.k + 1]!.coins, t₄.rounds[t₄.k]!.coins] := by
      rw [er₂, ek₂, er₄, ek₄, ek₅, hk₁,
        consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 7 by omega),
        show tE.k + 7 = tE.k + 4 + 3 by omega, consumed_add tE.rounds (tE.k + 4) 3]
      simp [List.range_succ, Nat.add_assoc]
    have hdrop : (sch.levels.toList.map levelOf).drop (i + 1) =
        [levelOf (sch.levels.getD (sch.levels.size - 1) default)] := by
      rw [hi, drop_levels sch (sch.levels.size - 1) (by omega), show sch.levels.size - 1 + 1 = sch.levels.size by omega]
      simp
    rw [← hstQ] at hmain
    rw [hdrop, show sch.levels.size - 1 - i = 1 by omega, hcs]
    simp only [recMsgs]
    refine ⟨_, _, rfl, er₅, by omega, hmain, ?_⟩
    subst hstN
    refine ⟨by simp [hsize]; omega, ?_, by simp only; omega, hyr', ?_, ?_, ?_, ?_⟩
    · intro l hl
      simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hl
      rcases hl with hl | rfl
      · exact hlanes l hl
      · exact hsz
    · dsimp only
      rw [hql, hqo, hqc, hqs, hstQ, hlen]
      dsimp only
      rw [hlevels, hoods, hcons, hsM₁, hoods₁, hconss₁,
        show v.1 - 4 = (sch.levels.getD (sch.levels.size - 1) default).cols by omega, gOf_add, gOf_smul]
      simp only [Array.toList_push, List.map_append, List.map_cons, List.map_nil]
      rfl
    · dsimp only
      rw [hqcaps, hstQ]
      simp only [hcaps, hi]
    · intro req hreq
      rcases hqr req hreq with h | h
      · exact hreqs req h
      · exact h
    · dsimp only
      rw [hconss₁]
      exact conss_push_pos hposE (pos_le_alpha _ hlvR _ _ (by rw [RoundsWF.get (er₂ ▸ hwfE), hnQ']))
  obtain ⟨pend, hrL, hkL, hplL, hsizeL, hlanesL, hextL, hyrL, hverL, hcapsL, hreqsL, hposL⟩ := hL
  clear hcont h₄ hloop
  have er₃ : t₃.rounds = t.rounds := by rw [k₃.2, k₂.2, k₁.2]
  have ek₃ : t₃.k = t.k := by rw [k₃.1, k₂.1, k₁.1]
  have er₇ : t₇.rounds = t.rounds := by rw [k₇.2, k₆.2, hr₅, er₃]
  have ek₇ : t₇.k = t.k + 1 := by rw [k₇.1, k₆.1, hk₅, ek₃]
  have er₁₀ : t₁₀.rounds = t.rounds := hr₁₀.trans er₉
  have ek₁₀ : t₁₀.k = t.k + 2 + sch.k0 := by rw [hk₁₀, ek₉]
  have ek₁₂' : t₁₂.k = t.k + 2 + sch.k0 + 2 := by rw [ek₁₂]; omega
  have ek₁₃' : t₁₃.k = t.k + 2 + sch.k0 + 4 := by rw [ek₁₃]; omega
  have ek₁₆' : t₁₆.k = t.k + 2 + sch.k0 + 6 := by rw [ek₁₆]; omega
  have ek₁₇' : t₁₇.k = t.k + 2 + sch.k0 + 7 := by rw [ek₁₇]; omega
  have hcs : consumed t.rounds t.k t₁₈.k = (t₃.rounds[t₃.k]!).coins :: (t₇.rounds[t₇.k]!).coins ::
      (consumed t₉.rounds t₉.k t₁₀.k ++
        [(t₁₁.rounds[t₁₁.k]!).coins, (t₁₁.rounds[t₁₁.k + 1]!).coins, (t₁₂.rounds[t₁₂.k]!).coins,
          (t₁₂.rounds[t₁₂.k + 1]!).coins] ++
        [(t₁₃.rounds[t₁₃.k]!).coins, (t₁₃.rounds[t₁₃.k + 1]!).coins, (t₁₆.rounds[t₁₆.k]!).coins] ++
        consumed t₁₇.rounds t₁₇.k t₁₈.k) := by
    rw [er₃, ek₃, er₇, ek₇, er₉, ek₉, ek₁₀, er₁₁, ek₁₁, er₁₂, ek₁₂', er₁₃, ek₁₃', er₁₆, ek₁₆', er₁₇, ek₁₇',
      consumed_append t.rounds (show t.k ≤ t.k + 2 + sch.k0 + 7 by omega) (by rw [← ek₁₇']; exact hkL),
      consumed_cons _ (show t.k < t.k + 2 + sch.k0 + 7 by omega),
      consumed_cons _ (show t.k + 1 < t.k + 2 + sch.k0 + 7 by omega),
      consumed_append t.rounds (show t.k + 1 + 1 ≤ t.k + 2 + sch.k0 by omega)
        (show t.k + 2 + sch.k0 ≤ t.k + 2 + sch.k0 + 7 by omega),
      consumed_add t.rounds (t.k + 2 + sch.k0) 7]
    simp [List.range_succ, Nat.add_assoc]
  have hlv01 : (schedOf sch).levels = levelOf (sch.levels.getD 0 default) ::
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := by
    have e0 := drop_levels sch 0 h0
    have e1 := drop_levels sch 1 (by omega)
    simp only [List.drop_zero, Nat.zero_add, Nat.reduceAdd] at e0 e1
    rw [e1] at e0
    exact e0
  have hd1 : (sch.levels.toList.map levelOf).drop (0 + 1) =
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := drop_levels sch 1 (by omega)
  rw [hd1, hstM₀, hstQ] at hplL
  have hmain := ligeritoC_model r (schedOf sch) (fOf T) b _ _ _ hlv01 (t₃.rounds[t₃.k]!).coins
    (t₇.rounds[t₇.k]!).coins hz₀ (by show sch.m - 7 ≠ 1; omega) (by rw [RoundsWF.get (er₇ ▸ hwf), hn₈]) y₀ m₀ hfold
    (capD p.recursiveCaps[0]!) htwo
    (t₁₃.rounds[t₁₃.k]!).coins (t₁₃.rounds[t₁₃.k + 1]!).coins (t₁₆.rounds[t₁₆.k]!).coins
    (by rw [RoundsWF.get (er₁₃ ▸ hwf), hnQ]; rfl) (by rw [RoundsWF.get (er₁₃ ▸ hwf), hnQ']; exact hlv0.2.2.2.1)
    (by rw [RoundsWF.get (er₁₆ ▸ hwf), hn₁₇]) hlv0.q_ne_one (hα1 0 (by omega)) mq hplL
  have hy0 := getElem!_of_getElem? hy₀
  have hm0 := getElem!_of_getElem? hm₀
  have hyA' := getElem!_of_getElem? hyA
  have hmA' := getElem!_of_getElem? hmA
  have hyB' := getElem!_of_getElem? hyB
  have hmB' := getElem!_of_getElem? hmB
  have hmq' := getElem!_of_getElem? hmq
  simp only [hood₁₀, hmsg₁₀] at hy0 hm0 hyA' hmA' hyB' hmB' hmq'
  have hLmsg : stL.msg = sch.k0 + 4 := by rw [hstL]; simp only [hmsg₁₀]; omega
  have hLood : stL.ood = 3 := by rw [hstL]; simp only [hood₁₀]
  refine ⟨pend, initCh, out.2.2.1, st₁₈, stF, t₁₈, hrL.trans er₁₇, by rw [er₁₇] at hrL; omega, ?_, hcapsL,
    hreqsL, by rw [hsz₁₀]; simp, hsizeL, hlanesL, hyrL, hposL, by rw [← hextL]; exact hfin, hverL⟩
  rw [hcs]
  unfold ligMsgs
  rw [hLmsg, hLood, Nat.sub_zero] at hmain
  rw [show 0 + 1 + sch.k0 = sch.k0 + 1 by rw [Nat.zero_add, Nat.add_comm]] at hmA'
  rw [show 0 + 1 + sch.k0 + 1 = sch.k0 + 2 by rw [Nat.zero_add, Nat.add_comm 1 sch.k0]] at hmB'
  rw [show 0 + 1 + sch.k0 + 1 + 1 = sch.k0 + 3 by rw [Nat.zero_add, Nat.add_comm 1 sch.k0]] at hmq'
  rw [show (0 + 1 + 1 : ℕ) = 2 from rfl] at hyB'
  rw [show (0 + 1 : ℕ) = 1 from rfl] at hyA' hmain
  rw [hy0, hm0, hyA', hmA', hyB', hmB', hmq']
  simp only [List.append_assoc, List.cons_append, List.nil_append] at hmain ⊢
  exact hmain

end FlockSoundness.Refine
