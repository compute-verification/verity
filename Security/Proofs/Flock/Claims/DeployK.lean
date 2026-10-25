import Proofs.Flock.Claims.DeployRoundsZk
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Soundness.Discharge.Exec.Tables
import Flock.With.VerifyE

/-!
# `Deploy.V`'s round count is the rounds the shipped verifier reads

The plain verdict's facts are stated at `verifyWithMK keep` (the shipped text with its digests of encodings read
through `keep`) wherever their proofs read none of those digests, so they hold at the element-level verdict
(`keep := fun _ => true`) too. The shipped verdict's (`verifyWithM`, `keep := id`) follow by `rfl`.
-/

namespace FlockDeploy

open Flock Transcript FlockSoundness.Refine FlockSoundness.Discharge.Exec

/-! ## `fast100`'s schedules -/

/-- Every `fast100` schedule is for `22 ≤ m`, draws a nonempty first point, has at least two levels, and queries each. -/
theorem fast100_rounds_ok {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    22 ≤ m ∧ sch.n ≠ 0 ∧ 2 ≤ sch.levels.size ∧ ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0 := by
  have hm : m < 40 := by
    by_contra hc
    rw [exec_fast100_none m (by omega)] at h; cases h
  have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, (s.n != 0 && decide (2 ≤ s.levels.size) &&
      s.levels.all fun lv => lv.q != 0) = true := by
    decide +kernel
  have hk := key ⟨m, hm⟩ sch h
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq, Array.all_eq_true] at hk
  refine ⟨exec_fast100_ge h, hk.1.1, hk.1.2, fun i hi => ?_⟩
  simp only [Array.getD, hi, ↓reduceDIte]
  exact hk.2 i hi

/-! ## One rep -/

/-- **A plain rep reads every recorded round of its stream**: when `verifyRepWith hs` accepts rep `r` of table `i`, at
the table's `fast100` schedule, its stream has `(3 + (m − 6)) + (3 + (kLog − 6)) + (claims + 1) + (k0 + 5 + 11 R)`
rounds (zerocheck, lincheck, the ring switch over `claims = 2 + |extras|`, Ligerito over `R + 1` levels), whatever the
hashes `hs` are. Level 0's query coins are the round after zerocheck, lincheck, the ring switch and Ligerito's first
`k0 + 6`: that round issued level 0's `q` coins, and level 0's Merkle check passes at their positions. -/
theorem verifyRepWith_rounds (hs : Hashes Flock.V) {spec : SessionSpec} {st : Setup} {sch : Schedule} {sess : Session}
    {i r : ℕ} {bytes c : ByteArray} (hk : st.fold.kLog = st.kLog) (hf : fast100 st.m = some sch)
    (h : verifyRepWith hs spec st sch sess i r bytes = .ok c) :
    ∃ (stream : StreamRec) (extras : Array Claim) (p : Proof),
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream ∧
      st.extra sess.points = .ok extras ∧ decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧
      stream.rounds.size =
        (3 + (st.m - 6)) + (3 + (st.kLog - 6)) + ((2 + extras.size) + 1) +
          (sch.k0 + 5 + 11 * (sch.levels.size - 1)) ∧
      (stream.rounds[(3 + (st.m - 6)) + (3 + (st.kLog - 6)) + ((2 + extras.size) + 1) + (sch.k0 + 6)]!).n =
        (sch.levels.getD 0 default).q ∧
      merkleCheckWith (st.merkle.opsWith hs) st.merkle p.cap (sch.levels.getD 0 default).cap
        (sch.levels.getD 0 default).d
        (positions (sch.levels.getD 0 default)
          (stream.rounds[(3 + (st.m - 6)) + (3 + (st.kLog - 6)) + ((2 + extras.size) + 1) + (sch.k0 + 6)]!).coins)
        p.ligerito.initialProof.rows p.ligerito.initialProof.siblings p.ligerito.initialProof.salts
        (2 ^ (sch.levels.getD 0 default).lanes) = .ok () := by
  obtain ⟨h22, hn, h2, hq⟩ := fast100_rounds_ok hf
  unfold verifyRepWith at h
  dsimp only at h
  cases hs' : Array.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r)
      sess.streams with
  | none => rw [hs'] at h; cases h
  | some stream =>
  rw [hs'] at h
  obtain ⟨extras, hex, h⟩ := ebind_ok h
  obtain ⟨p, hp, h⟩ := ebind_ok h
  by_cases c1 : (p.params != pcsParams st.merkle sch) = true
  · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1] at h
  obtain ⟨⟨u, tr⟩, hrun, h⟩ := ebind_ok h
  change (if (tr.k != stream.rounds.size) = true then _ else _) = _ at h
  by_cases c2 : (tr.k != stream.rounds.size) = true
  · rw [ite_eq_left c2] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  simp only [bne_iff_ne, ne_eq, not_not] at c2
  refine ⟨stream, extras, p, rfl, hex, hp, ?_⟩
  rw [StateT.run] at hrun
  obtain ⟨zc, t₁, h₁, hrun⟩ := bind_ok hrun
  obtain ⟨hr₁, hk₁⟩ := zerocheckWith_rounds hs (by omega) h₁
  obtain ⟨lc, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨hr₂, hk₂⟩ := lincheckWith_rounds hs h₂
  obtain ⟨⟨T, sws⟩, t₃, h₃, hrun⟩ := bind_ok hrun
  obtain ⟨hr₃, hk₃⟩ := ringSwitchWith_rounds hs (by simp [abcClaims]) h₃
  obtain ⟨hr₄, hk₄, hn₄, hmc₄⟩ := ligeritoWith_rounds hs st.merkle sch T p.cap p.ligerito (basis sws) hn h2 hq hrun
  have e₃ : t₃.rounds = stream.rounds := by rw [hr₃, hr₂, hr₁]; rfl
  have k₃ : t₃.k = (3 + (st.m - 6)) + (3 + (st.kLog - 6)) + ((2 + extras.size) + 1) := by
    rw [hk₃, hk₂, hk₁, hk]
    simp only [Transcript.init, abcClaims, Array.size_append, List.size_toArray, List.length_cons,
      List.length_nil]
    omega
  refine ⟨?_, ?_, ?_⟩
  · rw [← c2, hk₄, k₃]
  · rw [← e₃, ← k₃]; exact hn₄
  · rw [← e₃, ← k₃]; exact hmc₄

/-! ## The session -/

/-- **The plain verdict accepts each table's reps**: when `verifyWithMK keep hs` accepts, whatever the hashes `hs` and
`keep`, the record decodes, there is one setup per table, and each table `i` has its `fast100` schedule and both reps'
checks accept on the proof files `proofs[2i]` and `proofs[2i+1]`. -/
theorem verifyWithMK_tables (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup}
    {record : String} {proofs : Array ByteArray} (h : verifyWithMK keep hs spec tabs record proofs = .ok ()) :
    ∃ (sess : Session) (schs : Array Schedule), Record.decodeWithK keep hs spec record = .ok sess ∧
      tabs.size = spec.tables.size ∧ schs.size = tabs.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (hs' : i < schs.size) (c₀ c₁ : ByteArray),
        fast100 tabs[i].m = some schs[i] ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 0 proofs[2 * i]! = .ok c₀ ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 1 proofs[2 * i + 1]! = .ok c₁ := by
  unfold verifyWithMK at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  obtain ⟨sess, hsess, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨schs, hschs, h⟩ := ebind_ok h
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, hloop, _⟩ := ebind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one] at hloop
  have c0' : tabs.size = spec.tables.size := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, not_not] at c0
    exact c0.1
  obtain ⟨hsz, hsch⟩ := array_mapM_ok_getElem hschs
  have hP := range'_forIn_each tabs.size _
    (fun i => ∃ c₀ c₁, verifyRepWith hs spec tabs[i]! schs[i]! sess i 0 proofs[2 * i]! = .ok c₀ ∧
      verifyRepWith hs spec tabs[i]! schs[i]! sess i 1 proofs[2 * i + 1]! = .ok c₁)
    (fun a s y hy => by
      obtain ⟨c₀, h₀, hy⟩ := ebind_ok hy
      obtain ⟨c₁, h₁, hy⟩ := ebind_ok hy
      by_cases c3 : ((c₀ : ByteArray) != c₁) = true
      · rw [ite_eq_left c3] at hy; obtain ⟨_, h0, _⟩ := ebind_ok hy; cases h0
      rw [ite_eq_right c3] at hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      exact ⟨⟨c₀, c₁, h₀, h₁⟩, _, hy.symm⟩) _ _ hloop
  refine ⟨sess, schs, hsess, c0', hsz, fun i hi => ?_⟩
  have hs' : i < schs.size := hsz ▸ hi
  obtain ⟨c₀, c₁, h₀, h₁⟩ := hP i hi
  rw [getElem!_pos tabs i hi, getElem!_pos schs i hs'] at h₀ h₁
  refine ⟨hs', c₀, c₁, ?_, h₀, h₁⟩
  have hf := hsch i hi hs'
  cases hm : fast100 tabs[i].m with
  | none => rw [hm] at hf; cases hf
  | some s => rw [hm] at hf; cases hf; rfl

/-- **The shipped plain verdict accepts each table's reps** (`verifyWithMK_tables` at `keep := id`). -/
theorem verifyWithM_tables (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} (h : verifyWithM hs spec tabs record proofs = .ok ()) :
    ∃ (sess : Session) (schs : Array Schedule), Record.decodeWith hs spec record = .ok sess ∧
      tabs.size = spec.tables.size ∧ schs.size = tabs.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (hs' : i < schs.size) (c₀ c₁ : ByteArray),
        fast100 tabs[i].m = some schs[i] ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 0 proofs[2 * i]! = .ok c₀ ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 1 proofs[2 * i + 1]! = .ok c₁ :=
  verifyWithMK_tables id hs h

/-- **The masked verdict accepts each table's reps**: when `Zk.verifyWithM hs` accepts, whatever the hashes `hs`, the
record decodes, there is one setup per table, and each table `i` has its `--zk` schedule and both reps' checks accept on
the proof files `proofs[2i]` and `proofs[2i+1]`. -/
theorem zkVerifyWithM_tables (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} (h : Zk.verifyWithM hs spec tabs record proofs = .ok ()) :
    ∃ (sess : Session) (schs : Array (Schedule × ℕ)), Record.decodeWith hs spec record = .ok sess ∧
      tabs.size = spec.tables.size ∧ schs.size = tabs.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (hs' : i < schs.size) (c₀ c₁ : ByteArray),
        Zk.schedule tabs[i].m = .ok schs[i] ∧
        Zk.verifyRepWith hs spec tabs[i] schs[i].1 schs[i].2 sess i 0 proofs[2 * i]! = .ok c₀ ∧
        Zk.verifyRepWith hs spec tabs[i] schs[i].1 schs[i].2 sess i 1 proofs[2 * i + 1]! = .ok c₁ := by
  unfold Zk.verifyWithM at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  by_cases c0' : spec.coinTree.isNone = true
  · rw [ite_eq_left c0'] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0'] at h
  obtain ⟨sess, hsess, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨schs, hschs, h⟩ := ebind_ok h
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, hloop, _⟩ := ebind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one] at hloop
  have c0'' : tabs.size = spec.tables.size := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, not_not] at c0
    exact c0.1
  obtain ⟨hsz, hsch⟩ := array_mapM_ok_getElem hschs
  have hP := range'_forIn_each tabs.size _
    (fun i => ∃ c₀ c₁, Zk.verifyRepWith hs spec tabs[i]! schs[i]!.1 schs[i]!.2 sess i 0 proofs[2 * i]! = .ok c₀ ∧
      Zk.verifyRepWith hs spec tabs[i]! schs[i]!.1 schs[i]!.2 sess i 1 proofs[2 * i + 1]! = .ok c₁)
    (fun a s y hy => by
      obtain ⟨c₀, h₀, hy⟩ := ebind_ok hy
      obtain ⟨c₁, h₁, hy⟩ := ebind_ok hy
      by_cases c3 : ((c₀ : ByteArray) != c₁) = true
      · rw [ite_eq_left c3] at hy; obtain ⟨_, h0, _⟩ := ebind_ok hy; cases h0
      rw [ite_eq_right c3] at hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      exact ⟨⟨c₀, c₁, h₀, h₁⟩, _, hy.symm⟩) _ _ hloop
  refine ⟨sess, schs, hsess, c0'', hsz, fun i hi => ?_⟩
  have hs' : i < schs.size := hsz ▸ hi
  obtain ⟨c₀, c₁, h₀, h₁⟩ := hP i hi
  rw [getElem!_pos tabs i hi, getElem!_pos schs i hs'] at h₀ h₁
  exact ⟨hs', c₀, c₁, hsch i hi hs', h₀, h₁⟩

/-! ## The record's streams, read twice -/

section Lists

variable {α β γ ε : Type}

theorem forall₂_of_mapM {f : α → Except ε β} {P : α → β → Prop} (hf : ∀ a b, f a = .ok b → P a b) :
    ∀ {xs : List α} {ys : List β}, xs.mapM f = .ok ys → List.Forall₂ P xs ys
  | [], ys, h => by
    rw [List.mapM_nil] at h
    cases h
    exact .nil
  | x :: xs, ys, h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := ebind_ok h
    obtain ⟨bs, hbs, h⟩ := ebind_ok h
    cases h
    exact .cons (hf _ _ hb) (forall₂_of_mapM hf hbs)

theorem forall₂_of_array_mapM {f : α → Except ε β} {P : α → β → Prop} (hf : ∀ a b, f a = .ok b → P a b)
    {xs : Array α} {ys : Array β} (h : xs.mapM f = .ok ys) : List.Forall₂ P xs.toList ys.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    simp only [Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    simpa using forall₂_of_mapM hf hl

theorem forall₂_of_mapIdxM_go {f : ℕ → α → Except ε β} {P : α → β → Prop} (hf : ∀ k a b, f k a = .ok b → P a b) :
    ∀ {l : List α} {acc : Array β} {l' : List β}, List.mapIdxM.go f l acc = .ok l' →
      ∃ pre, l' = acc.toList ++ pre ∧ List.Forall₂ P l pre
  | [], acc, l', h => by
    simp only [List.mapIdxM.go, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨[], by simp, .nil⟩
  | a :: l, acc, l', h => by
    simp only [List.mapIdxM.go] at h
    obtain ⟨b, hb, h⟩ := ebind_ok h
    obtain ⟨pre, rfl, hpre⟩ := forall₂_of_mapIdxM_go hf h
    exact ⟨b :: pre, by simp, .cons (hf _ _ _ hb) hpre⟩

theorem forall₂_of_mapIdxM {f : ℕ → α → Except ε β} {P : α → β → Prop} (hf : ∀ k a b, f k a = .ok b → P a b)
    {xs : Array α} {ys : Array β} (h : xs.mapIdxM f = .ok ys) : List.Forall₂ P xs.toList ys.toList := by
  have e := Array.toList_mapIdxM (xs := xs) (f := f)
  rw [h] at e
  simp only [Functor.map, Except.map] at e
  obtain ⟨pre, hpre, hP⟩ := forall₂_of_mapIdxM_go hf (l := xs.toList) (acc := #[])
    (by simpa [List.mapIdxM] using e.symm)
  simpa [hpre] using hP

theorem find?_forall₂ {R : α → β → Prop} {p : α → Bool} {q : β → Bool} (hpq : ∀ a b, R a b → p a = q b) :
    ∀ {l : List α} {l' : List β} {a : α}, List.Forall₂ R l l' → l.find? p = some a → ∃ b, l'.find? q = some b ∧ R a b
  | _, _, _, .nil, h => by cases h
  | a₀ :: _, b₀ :: _, a, .cons hab t, h => by
    rw [List.find?_cons] at h ⊢
    rw [← hpq _ _ hab]
    cases hp : p a₀
    · rw [hp] at h; exact find?_forall₂ hpq t h
    · rw [hp] at h; cases h; exact ⟨b₀, rfl, hab⟩

theorem forall₂_meet {P : γ → α → Prop} {Q : γ → β → Prop} {S : α → β → Prop}
    (hS : ∀ c a b, P c a → Q c b → S a b) :
    ∀ {l : List γ} {l₁ : List α} {l₂ : List β}, List.Forall₂ P l l₁ → List.Forall₂ Q l l₂ → List.Forall₂ S l₁ l₂
  | _, _, _, .nil, .nil => .nil
  | _, _, _, .cons h₁ t₁, .cons h₂ t₂ => .cons (hS _ _ _ h₁ h₂) (forall₂_meet hS t₁ t₂)

end Lists

theorem str_getObjValAs {s : Lean.Json} {k n : String} (h : Record.str s k = .ok n) :
    (s.getObjValAs? String k).toOption = some n := by
  unfold Record.str Record.fld at h
  cases hv : s.getObjVal? k with
  | error e => rw [hv] at h; cases h
  | ok v =>
    have h' : v.getStr? = .ok n := by
      rw [hv] at h
      cases hg : v.getStr? with
      | error e => simp [hg, bind, Except.bind] at h
      | ok s' => simp [hg, bind, Except.bind] at h; rw [h]
    unfold Lean.Json.getObjValAs? Lean.Json.getObjValD
    rw [hv]
    show (v.getStr?).toOption = some n
    rw [h']; rfl

/-- A round as both readers read it: its coin count and its coins. -/
def RoundR (r : Lean.Json) (rd : Round) : Prop :=
  Record.nat r "n" = .ok rd.n ∧ (Record.arr r "coins" >>= Record.f128s) = .ok rd.coins

/-- A stream as the verifier's decoder reads it: its name, and its rounds, one per recorded round. -/
def StreamR (s : Lean.Json) (x : StreamRec) : Prop :=
  Record.str s "stream" = .ok x.name ∧ ∃ rs, Record.arr s "rounds" = .ok rs ∧ List.Forall₂ RoundR rs.toList x.rounds.toList

/-- **What the verifier's decoder reads of the record**, whatever the hashes and `keep`: the JSON, the link's points
(S9: the configured number of them) and the streams, each with its name and its rounds' coin counts and coins. -/
theorem decodeWithK_parts (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : Session} (h : Record.decodeWithK keep hs spec text = .ok sess) :
    ∃ (j l : Lean.Json) (ps recs : Array Lean.Json) (SF : Lean.Json → Flock.V StreamRec),
      Lean.Json.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.arr l "points" = .ok ps ∧
      ps.mapM (fun p => (match p.getArr? with
        | .ok a => Record.f128s a
        | .error _ => .error "record: a link point is not a list" : Flock.V (Array F128))) = .ok sess.points ∧
      sess.points.size = spec.points ∧
      Record.arr j "streams" = .ok recs ∧ recs.mapM SF = .ok sess.streams ∧ ∀ s x, SF s = .ok x → StreamR s x := by
  unfold Record.decodeWithK Record.decodeJWithK at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i j hparse
  obtain ⟨j', hj, h⟩ := ebind_ok h
  cases hj
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine ⟨_, _, _, _, _, by assumption, by assumption, by assumption, by assumption,
      (check_pts (by assumption)).1, by assumption, by assumption, fun s x hx => ?_⟩
    try dsimp only at hx
    obtain ⟨name, hname, hx⟩ := ebind_ok hx
    obtain ⟨rs, hrs, hx⟩ := ebind_ok hx
    obtain ⟨rounds, hrounds, hx⟩ := ebind_ok hx
    repeat' obtain ⟨_, _, hx⟩ := ebind_ok hx
    cases hx
    refine ⟨hname, rs, hrs, forall₂_of_mapIdxM (fun k r rd hr => ?_) hrounds⟩
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨n, hn, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨_, _, hr⟩ := ebind_ok hr
    obtain ⟨cs, hcs, hr⟩ := ebind_ok hr
    obtain ⟨coins, hcoins, hr⟩ := ebind_ok hr
    repeat' (first | (obtain ⟨_, _, hr⟩ := ebind_ok hr) | (split at hr))
    all_goals
      cases hr
      exact ⟨hn, by rw [show Record.arr r "coins" = .ok cs from hcs]; exact hcoins⟩

/-- **What the shipped decoder reads of the record** (`decodeWithK_parts` at `keep := id`). -/
theorem decodeWith_parts (hs : Hashes Flock.V) {spec : SessionSpec} {text : String} {sess : Session}
    (h : Record.decodeWith hs spec text = .ok sess) :
    ∃ (j l : Lean.Json) (ps recs : Array Lean.Json) (SF : Lean.Json → Flock.V StreamRec),
      Lean.Json.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.arr l "points" = .ok ps ∧
      ps.mapM (fun p => (match p.getArr? with
        | .ok a => Record.f128s a
        | .error _ => .error "record: a link point is not a list" : Flock.V (Array F128))) = .ok sess.points ∧
      sess.points.size = spec.points ∧
      Record.arr j "streams" = .ok recs ∧ recs.mapM SF = .ok sess.streams ∧ ∀ s x, SF s = .ok x → StreamR s x :=
  decodeWithK_parts id hs h

/-- **What `liveOf` reads of the record**: the JSON, the link's points, and for each rep stream in `spec.repStreams`
order, the first stream record of its name and that record's rounds' coin counts and coins. -/
theorem liveOf_parts {spec : SessionSpec} {text : String} {lv : Live} (h : liveOf spec text = .ok lv) :
    ∃ (j l : Lean.Json) (ps recs : Array Lean.Json),
      Lean.Json.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.arr l "points" = .ok ps ∧
      ps.mapM (fun p => (match p.getArr? with
        | .ok a => Record.f128s a
        | .error _ => .error "record: a link point is not a list" : Flock.V (Array F128))) = .ok lv.points ∧
      Record.arr j "streams" = .ok recs ∧ lv.streams.size = spec.repStreams.size ∧
      ∀ (k : ℕ) (hk : k < spec.repStreams.size) (hk' : k < lv.streams.size), ∃ (s : Lean.Json) (rs : Array Lean.Json),
        recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some spec.repStreams[k].2.2) = some s ∧
        Record.arr s "rounds" = .ok rs ∧ List.Forall₂ RoundR rs.toList lv.streams[k].toList := by
  unfold liveOf at h
  split at h
  rotate_left
  · cases h
  rename_i j hparse
  obtain ⟨j', hj, h⟩ := ebind_ok h
  cases hj
  obtain ⟨l, hl, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨ps, hps, h⟩ := ebind_ok h
  obtain ⟨pts, hpts, h⟩ := ebind_ok h
  obtain ⟨recs, hrecs, h⟩ := ebind_ok h
  obtain ⟨streams, hstreams, h⟩ := ebind_ok h
  cases h
  obtain ⟨hsz, hall⟩ := array_mapM_ok_getElem hstreams
  refine ⟨j, l, ps, recs, hparse, hl, hps, hpts, hrecs, hsz, fun k hk hk' => ?_⟩
  have hg := hall k hk hk'
  rcases he : spec.repStreams[k] with ⟨i, r, name⟩
  rw [he] at hg
  dsimp only at hg ⊢
  split at hg
  rotate_left
  · cases hg
  rename_i s hfind
  obtain ⟨rs, hrs, hg⟩ := ebind_ok hg
  refine ⟨s, rs, hfind, hrs, forall₂_of_array_mapM (fun r rd hr => ?_) hg⟩
  obtain ⟨n, hn, hr⟩ := ebind_ok hr
  obtain ⟨cs, hcs, hr⟩ := ebind_ok hr
  obtain ⟨coins, hcoins, hr⟩ := ebind_ok hr
  repeat' (first | (obtain ⟨_, _, hr⟩ := ebind_ok hr) | (split at hr))
  all_goals
    cases hr
    exact ⟨hn, by rw [hcs]; exact hcoins⟩

/-- **The two readers agree**: when `liveOf` and the verifier's decoder (at any hashes and `keep`) both read the
record, they read the same link points, the configured number of them, and for each rep stream in `spec.repStreams`
order the verifier's stream of that name has the rounds `liveOf` read, with the same coin counts and coins. -/
theorem liveOf_decodeWithK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {lv : Live} {sess : Session} (hl : liveOf spec text = .ok lv)
    (hd : Record.decodeWithK keep hs spec text = .ok sess) :
    lv.points = sess.points ∧ sess.points.size = spec.points ∧ lv.streams.size = spec.repStreams.size ∧
      ∀ (k : ℕ) (hk : k < spec.repStreams.size) (hk' : k < lv.streams.size), ∃ stream : StreamRec,
        sess.streams.find? (fun x => x.name == spec.repStreams[k].2.2) = some stream ∧
        List.Forall₂ (fun a b => a.n = b.n ∧ a.coins = b.coins) lv.streams[k].toList stream.rounds.toList := by
  obtain ⟨j, l, ps, recs, hj, hl', hps, hpts, hrecs, hsz, hall⟩ := liveOf_parts hl
  obtain ⟨j₂, l₂, ps₂, recs₂, SF, hj₂, hl₂, hps₂, hpts₂, hpn, hrecs₂, hSF, hR⟩ := decodeWithK_parts keep hs hd
  rw [hj] at hj₂; cases hj₂
  rw [hl'] at hl₂; cases hl₂
  rw [hps] at hps₂; cases hps₂
  rw [hpts] at hpts₂
  rw [hrecs] at hrecs₂; cases hrecs₂
  refine ⟨Except.ok.inj hpts₂, hpn, hsz, fun k hk hk' => ?_⟩
  obtain ⟨s, rs, hfind, hrs, hrd⟩ := hall k hk hk'
  have h₂ := forall₂_of_array_mapM hR hSF
  rw [← Array.find?_toList] at hfind
  obtain ⟨x, hx, hsx⟩ := find?_forall₂ (q := fun x => x.name == spec.repStreams[k].2.2)
    (fun a b hab => by rw [str_getObjValAs hab.1]; simp) h₂ hfind
  rw [Array.find?_toList] at hx
  obtain ⟨-, rs₂, hrs₂, hrd₂⟩ := hsx
  rw [hrs] at hrs₂; cases hrs₂
  refine ⟨x, hx, forall₂_meet (fun r a b ha hb => ⟨?_, ?_⟩) hrd hrd₂⟩
  · exact Except.ok.inj (ha.1.symm.trans hb.1)
  · exact Except.ok.inj (ha.2.symm.trans hb.2)

/-- **`liveOf` and the shipped decoder agree** (`liveOf_decodeWithK` at `keep := id`). -/
theorem liveOf_decodeWith (hs : Hashes Flock.V) {spec : SessionSpec} {text : String} {lv : Live} {sess : Session}
    (hl : liveOf spec text = .ok lv) (hd : Record.decodeWith hs spec text = .ok sess) :
    lv.points = sess.points ∧ sess.points.size = spec.points ∧ lv.streams.size = spec.repStreams.size ∧
      ∀ (k : ℕ) (hk : k < spec.repStreams.size) (hk' : k < lv.streams.size), ∃ stream : StreamRec,
        sess.streams.find? (fun x => x.name == spec.repStreams[k].2.2) = some stream ∧
        List.Forall₂ (fun a b => a.n = b.n ∧ a.coins = b.coins) lv.streams[k].toList stream.rounds.toList :=
  liveOf_decodeWithK id hs hl hd

/-! ## Counting -/

section Count

variable {α β : Type}

theorem flatMap_pairs (g : ℕ → ℕ → β) :
    ∀ n, (List.range n).flatMap (fun i => [g i 0, g i 1]) = (List.range (2 * n)).map fun k => g (k / 2) (k % 2)
  | 0 => rfl
  | n + 1 => by
    have h0 : 2 * n / 2 = n := by omega
    have h1 : 2 * n % 2 = 0 := by omega
    have h2 : (2 * n + 1) / 2 = n := by omega
    have h3 : (2 * n + 1) % 2 = 1 := by omega
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by omega]
    simp only [List.range_succ, List.flatMap_append, flatMap_pairs g n, List.map_append, List.append_assoc,
      List.flatMap_cons, List.flatMap_nil, List.append_nil, List.map_cons, List.map_nil, List.cons_append,
      List.nil_append, h0, h1, h2, h3]

theorem sum_pairs (G : ℕ → ℕ) :
    ∀ n, ((List.range (2 * n)).map fun k => G (k / 2)).sum = ((List.range n).map fun i => 2 * G i).sum
  | 0 => rfl
  | n + 1 => by
    have h0 : 2 * n / 2 = n := by omega
    have h2 : (2 * n + 1) / 2 = n := by omega
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by omega]
    simp only [List.range_succ, List.map_append, List.sum_append, sum_pairs G n, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, h0, h2]
    omega

theorem foldl_add_sum (f : α → ℕ) :
    ∀ (l : List α) (acc : ℕ), l.foldl (fun acc a => acc + f a) acc = acc + (l.map f).sum
  | [], acc => by simp
  | a :: l, acc => by rw [List.foldl_cons, foldl_add_sum f l, List.map_cons, List.sum_cons, Nat.add_assoc]

end Count

/-- The session's rep streams are table `k / 2`'s rep `k % 2`, for `k < 2 · tables`. -/
theorem repStreams_toList (spec : SessionSpec) :
    spec.repStreams.toList =
      (List.range (2 * spec.tables.size)).map fun k => (k / 2, k % 2, s!"{spec.tables[k / 2]!}/rep{k % 2}") := by
  unfold SessionSpec.repStreams
  rw [Array.toList_flatMap, Array.toList_range]
  simp only [Array.toList_map]
  exact flatMap_pairs (fun i r => (i, r, s!"{spec.tables[i]!}/rep{r}")) _

theorem repStreams_size (spec : SessionSpec) : spec.repStreams.size = 2 * spec.tables.size := by
  rw [← Array.length_toList, repStreams_toList, List.length_map, List.length_range]

theorem repStreams_getElem (spec : SessionSpec) {k : ℕ} (hk : k < spec.repStreams.size) :
    spec.repStreams[k] = (k / 2, k % 2, s!"{spec.tables[k / 2]!}/rep{k % 2}") := by
  rw [← Array.getElem_toList]
  simp only [repStreams_toList, List.getElem_map, List.getElem_range]

/-- One extra claim per region and link point. -/
theorem extraClaims_size {st : Stmt} {pts : Array (Array F128)} {extras : Array Flock.Claim}
    (h : st.extraClaims pts = .ok extras) : extras.size = st.regions.size * pts.size := by
  obtain ⟨ess, hess, heq⟩ := extraClaims_ok st pts extras h
  rw [← Array.length_toList, heq, List.length_flatten, ← Array.length_toList (xs := st.regions),
    ← Array.length_toList (xs := pts)]
  clear heq
  generalize st.regions.toList = rs at hess ⊢
  generalize pts.toList = ps at hess ⊢
  induction hess with
  | nil => simp
  | cons hes _ ih => rw [List.map_cons, List.sum_cons, ih, List.length_cons, ← hes.length_eq, Nat.succ_mul]; omega

/-! ## The decision's messages -/

theorem finalOf_head {a : Claims.Msg} {f : Final} (h : finalOf a = some f) : a.head? = some 1 := by
  unfold finalOf at h
  split at h
  · rfl
  · cases h

theorem linkOf_head {a : Claims.Msg} {link : List ByteArray × List ByteArray} (h : linkOf a = some link) :
    a.head? = some 0 := by
  unfold linkOf at h
  split at h
  · rfl
  · cases h

namespace KS.Gen

variable [S : ZkSetup]

/-- **A decision's round messages are all the messages but the link message and the last**, with one coin per
message. -/
theorem act_decide_length {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (hcs : cs.length = as.length)
    (h : act zk x as cs = .decide s spec sts link c₁ mids f) : as.length = mids.length + 2 := by
  unfold act at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  rename_i a _ _ _ _ a₁ rest _ crest hfin hlast _ _ _ hlink
  cases h
  cases rest with
  | nil =>
    simp only [List.getLast?_singleton, Option.some.injEq] at hlast
    subst hlast
    have h1 := finalOf_head hfin
    rw [linkOf_head hlink] at h1
    cases h1
  | cons b rest =>
    simp only [List.length_cons] at hcs
    simp only [List.length_cons, List.length_zip, List.length_map, List.length_dropLast]
    omega

end KS.Gen

/-- **A decision's round messages are all the messages but the link message and the last**, with one coin per
message. -/
theorem act_decide_length {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (hcs : cs.length = as.length)
    (h : act zk x as cs = .decide s spec sts link c₁ mids f) : as.length = mids.length + 2 :=
  KS.Gen.act_decide_length (S := KS.S₀) (zk := zk) (x := x) (as := as) (cs := cs) (s := s) (spec := spec) (sts := sts)
    (link := link) (c₁ := c₁) (mids := mids) (f := f) (hcs := hcs) (h := h)

/-- The checks pass only on the live session's custody. -/
theorem checks_custody {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) :
    ∃ lv, liveOf spec f.record = .ok lv ∧ custody spec lv link c₁ mids = .ok () := by
  unfold checks at h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨lv, hlv, h⟩ := ebind_ok h
  obtain ⟨_, hc, _⟩ := ebind_ok h
  exact ⟨lv, hlv, hc⟩


namespace KS.Gen

variable [S : ZkSetup]

/-- The checks pass only on the live session's custody. -/
theorem checks_custody {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) :
    ∃ lv, liveOf spec f.record = .ok lv ∧ custody spec lv link c₁ mids = .ok () :=
  _root_.FlockDeploy.checks_custody (checks_base h)

end KS.Gen

/-- Custody holds only when the record's rounds are the round messages, one each. -/
theorem custody_length {spec : SessionSpec} {lv : Live} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} (h : custody spec lv link c₁ mids = .ok ()) :
    (lv.streams.toList.flatMap (·.toList)).length = mids.length := by
  unfold custody at h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, hc, _⟩ := ebind_ok h
  simpa using check_ok_iff.1 hc

/-! ## The plain round count -/

/-- **Rep stream `k` of an accepted plain record** is table `k / 2`'s rep `k % 2`: when the plain verdict accepts the
record at any hashes and `liveOf` reads it, there are two rep streams per table, and for stream `k` the verifier's rep
check passes on proof file `k` at the table's `fast100` schedule, over the link points `liveOf` read, and its stream of
that name has the rounds `liveOf` read, with the same coin counts and coins. At any `keep` (`verifyWithMK`). -/
theorem rep_plainK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {record : String} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyWithMK keep hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧ ∀ (k : ℕ) (hk : k < lv.streams.size),
      ∃ (sess : Session) (sch : Schedule) (stream : StreamRec) (c : ByteArray),
        sess.points = lv.points ∧ sess.points.size = spec.points ∧ fast100 sts[k / 2]!.m = some sch ∧
        verifyRepWith hs spec (Setup.ofCircuit sts[k / 2]!) sch sess (k / 2) (k % 2) proofs[k]! = .ok c ∧
        sess.streams.find? (fun x => x.name == s!"{spec.tables[k / 2]!}/rep{k % 2}") = some stream ∧
        List.Forall₂ (fun a b => a.n = b.n ∧ a.coins = b.coins) lv.streams[k].toList stream.rounds.toList := by
  obtain ⟨sess, schs, hdec, hsz, -, hall⟩ := verifyWithMK_tables keep hs hv
  obtain ⟨hpts, hpn, hlsz, hstr⟩ := liveOf_decodeWithK keep hs hl hdec
  have hn : sts.size = spec.tables.size := by simpa [tabsOf] using hsz
  have hk₂ : ∀ k, k < lv.streams.size → k < spec.repStreams.size := fun k hk => hlsz ▸ hk
  rw [repStreams_size, ← hn] at hlsz
  refine ⟨hlsz, fun k hk => ?_⟩
  obtain ⟨stream, hfind, hfa⟩ := hstr k (hk₂ k hk) hk
  have hi : k / 2 < (tabsOf sts).size := by simp only [tabsOf, Array.size_map]; omega
  have hi' : k / 2 < sts.size := by omega
  obtain ⟨_, c₀, c₁, hf, h₀, h₁⟩ := hall (k / 2) hi
  have htab : (tabsOf sts)[k / 2] = Setup.ofCircuit sts[k / 2]! := by
    rw [getElem!_pos sts (k / 2) hi']; exact Array.getElem_map ..
  rw [htab] at h₀ h₁ hf
  simp only [repStreams_getElem spec (hk₂ k hk)] at hfind
  refine ⟨sess, schs[k / 2], stream, ?_⟩
  rcases Nat.mod_two_eq_zero_or_one k with h | h
  · rw [show 2 * (k / 2) = k by omega] at h₀
    rw [h] at hfind ⊢
    exact ⟨c₀, hpts.symm, hpn, hf, h₀, hfind, hfa⟩
  · rw [show 2 * (k / 2) + 1 = k by omega] at h₁
    rw [h] at hfind ⊢
    exact ⟨c₁, hpts.symm, hpn, hf, h₁, hfind, hfa⟩

/-- **Rep stream `k` of a record the shipped plain verdict accepts** (`rep_plainK` at `keep := id`). -/
theorem rep_plain (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧ ∀ (k : ℕ) (hk : k < lv.streams.size),
      ∃ (sess : Session) (sch : Schedule) (stream : StreamRec) (c : ByteArray),
        sess.points = lv.points ∧ sess.points.size = spec.points ∧ fast100 sts[k / 2]!.m = some sch ∧
        verifyRepWith hs spec (Setup.ofCircuit sts[k / 2]!) sch sess (k / 2) (k % 2) proofs[k]! = .ok c ∧
        sess.streams.find? (fun x => x.name == s!"{spec.tables[k / 2]!}/rep{k % 2}") = some stream ∧
        List.Forall₂ (fun a b => a.n = b.n ∧ a.coins = b.coins) lv.streams[k].toList stream.rounds.toList :=
  rep_plainK id hs hv hl

/-- **Each plain rep stream has its table's `repRounds`**: when the plain verdict accepts the record at any hashes and
`liveOf` reads it, there are two rep streams per table, and stream `k` (table `k / 2`'s rep `k % 2`) has
`repRounds false spec sts[k / 2]` rounds. At any `keep` (`verifyWithMK`). -/
theorem streams_size_plainK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {record : String} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyWithMK keep hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].size = repRounds false spec sts[k / 2]! := by
  obtain ⟨hsz, hall⟩ := rep_plainK keep hs hv hl
  refine ⟨hsz, fun k hk => ?_⟩
  obtain ⟨sess, sch, stream, c, -, hpn, hf, hrep, hfind, hfa⟩ := hall k hk
  obtain ⟨stream', extras, p, hfind', hex, -, hsize, -⟩ := verifyRepWith_rounds hs rfl hf hrep
  rw [hfind] at hfind'
  cases hfind'
  have hx : extras.size = sts[k / 2]!.regions.size * sess.points.size := extraClaims_size hex
  rw [← Array.length_toList, hfa.length_eq, Array.length_toList, hsize]
  unfold repRounds
  simp only [Bool.false_eq_true, ↓reduceIte, hf]
  rw [hx, hpn, Nat.mul_comm]
  rfl

/-- **Each rep stream the shipped plain verdict accepts has its table's `repRounds`** (`streams_size_plainK` at
`keep := id`). -/
theorem streams_size_plain (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].size = repRounds false spec sts[k / 2]! :=
  streams_size_plainK id hs hv hl

/-- The column reader's plain proof shape is the verifier's (`shapeOf`) at the table's `fast100` schedule. -/
theorem plainProofShape_eq {tab : Setup} {sch : Schedule} (h : Claims.Flock.plainSched tab.m = sch) (n : ℕ) :
    Claims.Flock.plainProofShape tab.m tab.kLog (2 + n) tab.merkle.digestLen (saltLenOf false tab) =
      shapeOf tab sch n := by
  subst h
  rfl

/-- **The plain query round is the round whose coins the shipped verifier queries level 0 at**: when the plain verdict
accepts the record at any hashes and `liveOf` reads it, rep stream `k` (table `k / 2`'s rep `k % 2`, at its setup
`tab`) has, at `plainQueryRound`, a round of level 0's `q₀` coins, and proof file `k` decodes at `plainProofShape` to
the proof whose level-0 Merkle check the verifier passes at those coins' positions. At any `keep` (`verifyWithMK`). -/
theorem queryRound_plainK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {record : String} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyWithMK keep hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) {k : ℕ} (hk : k < lv.streams.size) {tab : Setup}
    (htab : tab = Setup.ofCircuit sts[k / 2]!) :
    ∃ (extras : Array Flock.Claim) (p : Proof) (rd : Round),
      tab.extra lv.points = .ok extras ∧
      decodeProof (Claims.Flock.plainProofShape tab.m tab.kLog (2 + extras.size) tab.merkle.digestLen
        (saltLenOf false tab)) proofs[k]! = .ok p ∧
      lv.streams[k][Claims.Flock.plainQueryRound tab.m tab.kLog (Claims.Flock.plainSched tab.m).k0
        (2 + extras.size)]? = some rd ∧
      rd.n = (Claims.Flock.plainLevel0 tab.m).q ∧
      merkleCheckWith (tab.merkle.opsWith hs) tab.merkle p.cap (Claims.Flock.plainLevel0 tab.m).cap
        (Claims.Flock.plainLevel0 tab.m).d (positions (Claims.Flock.plainLevel0 tab.m) rd.coins)
        p.ligerito.initialProof.rows p.ligerito.initialProof.siblings p.ligerito.initialProof.salts
        (2 ^ (Claims.Flock.plainLevel0 tab.m).lanes) = .ok () := by
  obtain ⟨-, hall⟩ := rep_plainK keep hs hv hl
  obtain ⟨sess, sch, stream, c, hpts, -, hf, hrep, hfind, hfa⟩ := hall k hk
  obtain ⟨-, -, h2, -⟩ := fast100_rounds_ok hf
  subst htab
  obtain ⟨stream', extras, p, hfind', hex, hp, hsize, hn₀, hmc⟩ := verifyRepWith_rounds hs rfl hf hrep
  rw [hfind] at hfind'
  cases hfind'
  have hsched : Claims.Flock.plainSched (Setup.ofCircuit sts[k / 2]!).m = sch := by
    simp only [Claims.Flock.plainSched]
    rw [show (Setup.ofCircuit sts[k / 2]!).m = sts[k / 2]!.m from rfl, hf]
    rfl
  have hlv0 : Claims.Flock.plainLevel0 (Setup.ofCircuit sts[k / 2]!).m = sch.levels.getD 0 default := by
    rw [Claims.Flock.plainLevel0, hsched]
  have hq : Claims.Flock.plainQueryRound (Setup.ofCircuit sts[k / 2]!).m (Setup.ofCircuit sts[k / 2]!).kLog
      (Claims.Flock.plainSched (Setup.ofCircuit sts[k / 2]!).m).k0 (2 + extras.size) =
      (3 + ((Setup.ofCircuit sts[k / 2]!).m - 6)) + (3 + ((Setup.ofCircuit sts[k / 2]!).kLog - 6)) +
        ((2 + extras.size) + 1) + (sch.k0 + 6) := by
    rw [Claims.Flock.plainQueryRound, hsched]
  have hI : (3 + ((Setup.ofCircuit sts[k / 2]!).m - 6)) + (3 + ((Setup.ofCircuit sts[k / 2]!).kLog - 6)) +
      ((2 + extras.size) + 1) + (sch.k0 + 6) < stream.rounds.size := by
    rw [hsize]; omega
  have hI' : (3 + ((Setup.ofCircuit sts[k / 2]!).m - 6)) + (3 + ((Setup.ofCircuit sts[k / 2]!).kLog - 6)) +
      ((2 + extras.size) + 1) + (sch.k0 + 6) < lv.streams[k].size := by
    rw [← Array.length_toList (xs := lv.streams[k]), hfa.length_eq, Array.length_toList]; exact hI
  obtain ⟨hrn, hrc⟩ := hfa.get (by simpa using hI') (by simpa using hI)
  simp only [List.get_eq_getElem, Array.getElem_toList] at hrn hrc
  rw [getElem!_pos stream.rounds _ hI] at hn₀ hmc
  refine ⟨extras, p, _, by rw [← hpts]; exact hex, ?_, by rw [hq]; exact Array.getElem?_eq_getElem hI', ?_, ?_⟩
  · rw [plainProofShape_eq hsched]
    exact hp
  · rw [hlv0, hrn]; exact hn₀
  · rw [hlv0, hrc]; exact hmc

/-- **The plain query round of a record the shipped plain verdict accepts** (`queryRound_plainK` at `keep := id`). -/
theorem queryRound_plain (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) {k : ℕ} (hk : k < lv.streams.size) {tab : Setup}
    (htab : tab = Setup.ofCircuit sts[k / 2]!) :
    ∃ (extras : Array Flock.Claim) (p : Proof) (rd : Round),
      tab.extra lv.points = .ok extras ∧
      decodeProof (Claims.Flock.plainProofShape tab.m tab.kLog (2 + extras.size) tab.merkle.digestLen
        (saltLenOf false tab)) proofs[k]! = .ok p ∧
      lv.streams[k][Claims.Flock.plainQueryRound tab.m tab.kLog (Claims.Flock.plainSched tab.m).k0
        (2 + extras.size)]? = some rd ∧
      rd.n = (Claims.Flock.plainLevel0 tab.m).q ∧
      merkleCheckWith (tab.merkle.opsWith hs) tab.merkle p.cap (Claims.Flock.plainLevel0 tab.m).cap
        (Claims.Flock.plainLevel0 tab.m).d (positions (Claims.Flock.plainLevel0 tab.m) rd.coins)
        p.ligerito.initialProof.rows p.ligerito.initialProof.siblings p.ligerito.initialProof.salts
        (2 ^ (Claims.Flock.plainLevel0 tab.m).lanes) = .ok () :=
  queryRound_plainK id hs hv hl hk htab

theorem getD_empty (a : Array ByteArray) (j : ℕ) : a.getD j ByteArray.empty = a[j]! := by
  simp [Array.getD, getElem!_def]
  split <;> simp_all
  rfl

/-- **The column reader reads the verifier's level-0 query round**: where the run's last message `f` sets up and the
plain verdict accepts its record and proof files at any hashes, each of table 0's two reps that `runOf … 0` takes has,
at `plainQueryRound`, a round of level 0's `q₀` coins; its proof file decodes at the run's `plainShape` to the proof
whose level-0 Merkle check the verifier passes at those coins' positions; and `plainRepColsOf` is that proof's level-0
rows at those coins. At any `keep` (`verifyWithMK`). -/
theorem runOf_queryRound_plainK (keep : Bool → Bool) (hs : Hashes Flock.V) {s : Claims.Seen} {f : Final} {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {r : Claims.Flock.ZkRun} (hf : lastFinal s = some f)
    (hset : setupOf false f.stmt = some (ins, spec, sts))
    (hv : verifyWithMK keep hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hr : runOf false s 0 = some r) :
    ∀ x ∈ r.reps, ∃ (p : Proof) (rd : Round),
      decodeProof r.plainShape x.2 = .ok p ∧
      x.1[Claims.Flock.plainQueryRound r.m r.kLog (Claims.Flock.plainSched r.m).k0 r.claims]? = some rd ∧
      rd.n = (Claims.Flock.plainLevel0 r.m).q ∧
      merkleCheckWith ((Setup.ofCircuit sts[0]!).merkle.opsWith hs) (Setup.ofCircuit sts[0]!).merkle p.cap
        (Claims.Flock.plainLevel0 r.m).cap (Claims.Flock.plainLevel0 r.m).d
        (positions (Claims.Flock.plainLevel0 r.m) rd.coins) p.ligerito.initialProof.rows
        p.ligerito.initialProof.siblings p.ligerito.initialProof.salts (2 ^ (Claims.Flock.plainLevel0 r.m).lanes) =
          .ok () ∧
      Claims.Flock.plainRepColsOf r.plainShape x.1 x.2 =
        Claims.Flock.zkRepCols (Claims.Flock.plainLevel0 r.m) rd.coins p.ligerito.initialProof.rows := by
  unfold runOf at hr
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hr
  obtain ⟨f', hf', ⟨ins', spec', sts'⟩, hset', st, hst, lv, hlv, extras, hex, rfl⟩ := hr
  rw [hf] at hf'
  cases hf'
  rw [hset] at hset'
  cases hset'
  have hl : liveOf spec f.record = .ok lv := by
    cases h : liveOf spec f.record <;> simp [h, Except.toOption] at hlv
    rw [hlv]
  have hex' : (Setup.ofCircuit st).extra lv.points = .ok extras := by
    cases h : (Setup.ofCircuit st).extra lv.points <;> simp [h, Except.toOption] at hex
    rw [hex]
  have h0 : 0 < sts.size ∧ sts[0]! = st := by
    rw [Array.getElem?_eq_some_iff] at hst
    obtain ⟨h0, rfl⟩ := hst
    exact ⟨h0, getElem!_pos sts 0 h0⟩
  obtain ⟨hsz, -⟩ := rep_plainK keep hs hv hl
  intro x hx
  have hj : ∃ j, j < 2 ∧ x = (lv.streams.getD j #[], f.proofs.getD j ByteArray.empty) := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl
    · exact ⟨0, by omega, rfl⟩
    · exact ⟨1, by omega, rfl⟩
  obtain ⟨j, hj2, rfl⟩ := hj
  have hjk : j < lv.streams.size := by omega
  have hj0 : j / 2 = 0 := by omega
  obtain ⟨extras', p, rd, hex₂, hp, hrd, hn, hmc⟩ :=
    queryRound_plainK keep hs hv hl hjk (tab := Setup.ofCircuit st) (by rw [hj0, h0.2])
  rw [hex'] at hex₂
  cases hex₂
  have hget : lv.streams.getD j #[] = lv.streams[j] := by simp [Array.getD, hjk]
  rw [h0.2]
  refine ⟨p, rd, ?_, ?_, hn, hmc, ?_⟩
  · rw [getD_empty]; exact hp
  · rw [hget]; exact hrd
  · show Claims.Flock.plainRepColsOf
        (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
          (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
        (lv.streams.getD j #[]) (f.proofs.getD j ByteArray.empty) =
      Claims.Flock.zkRepCols (Claims.Flock.plainLevel0 (Setup.ofCircuit st).m) rd.coins p.ligerito.initialProof.rows
    rw [hget, getD_empty f.proofs j]
    unfold Claims.Flock.plainRepColsOf
    rw [hp]
    show Claims.Flock.zkRepCols (Claims.Flock.plainLevel0 (Setup.ofCircuit st).m)
        ((lv.streams[j][Claims.Flock.plainQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
          (Claims.Flock.plainSched (Setup.ofCircuit st).m).k0 (2 + extras.size)]?.map (·.coins)).getD #[])
        p.ligerito.initialProof.rows = _
    rw [hrd]
    rfl

/-- **The column reader reads the shipped verifier's level-0 query round** (`runOf_queryRound_plainK` at
`keep := id`). -/
theorem runOf_queryRound_plain (hs : Hashes Flock.V) {s : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {r : Claims.Flock.ZkRun} (hf : lastFinal s = some f)
    (hset : setupOf false f.stmt = some (ins, spec, sts))
    (hv : verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hr : runOf false s 0 = some r) :
    ∀ x ∈ r.reps, ∃ (p : Proof) (rd : Round),
      decodeProof r.plainShape x.2 = .ok p ∧
      x.1[Claims.Flock.plainQueryRound r.m r.kLog (Claims.Flock.plainSched r.m).k0 r.claims]? = some rd ∧
      rd.n = (Claims.Flock.plainLevel0 r.m).q ∧
      merkleCheckWith ((Setup.ofCircuit sts[0]!).merkle.opsWith hs) (Setup.ofCircuit sts[0]!).merkle p.cap
        (Claims.Flock.plainLevel0 r.m).cap (Claims.Flock.plainLevel0 r.m).d
        (positions (Claims.Flock.plainLevel0 r.m) rd.coins) p.ligerito.initialProof.rows
        p.ligerito.initialProof.siblings p.ligerito.initialProof.salts (2 ^ (Claims.Flock.plainLevel0 r.m).lanes) =
          .ok () ∧
      Claims.Flock.plainRepColsOf r.plainShape x.1 x.2 =
        Claims.Flock.zkRepCols (Claims.Flock.plainLevel0 r.m) rd.coins p.ligerito.initialProof.rows :=
  runOf_queryRound_plainK id hs hf hset hv hr

/-! ## The masked round count -/

/-- **Rep stream `k` of an accepted masked record** is table `k / 2`'s rep `k % 2`: when the masked verdict accepts the
record at any hashes and `liveOf` reads it, there are two rep streams per table, and for stream `k` the verifier's rep
check passes on proof file `k` at the table's `--zk` schedule, over the link points `liveOf` read, and its stream of
that name has the rounds `liveOf` read, with the same coin counts and coins. -/
theorem rep_zk (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : Zk.verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧ ∀ (k : ℕ) (hk : k < lv.streams.size),
      ∃ (sess : Session) (sch : Schedule) (tPad : ℕ) (stream : StreamRec) (c : ByteArray),
        sess.points = lv.points ∧ sess.points.size = spec.points ∧ Zk.schedule sts[k / 2]!.m = .ok (sch, tPad) ∧
        Zk.verifyRepWith hs spec (Setup.ofCircuit sts[k / 2]!) sch tPad sess (k / 2) (k % 2) proofs[k]! = .ok c ∧
        sess.streams.find? (fun x => x.name == s!"{spec.tables[k / 2]!}/rep{k % 2}") = some stream ∧
        List.Forall₂ (fun a b => a.n = b.n ∧ a.coins = b.coins) lv.streams[k].toList stream.rounds.toList := by
  obtain ⟨sess, schs, hdec, hsz, -, hall⟩ := zkVerifyWithM_tables hs hv
  obtain ⟨hpts, hpn, hlsz, hstr⟩ := liveOf_decodeWith hs hl hdec
  have hn : sts.size = spec.tables.size := by simpa [tabsOf] using hsz
  have hk₂ : ∀ k, k < lv.streams.size → k < spec.repStreams.size := fun k hk => hlsz ▸ hk
  rw [repStreams_size, ← hn] at hlsz
  refine ⟨hlsz, fun k hk => ?_⟩
  obtain ⟨stream, hfind, hfa⟩ := hstr k (hk₂ k hk) hk
  have hi : k / 2 < (tabsOf sts).size := by simp only [tabsOf, Array.size_map]; omega
  have hi' : k / 2 < sts.size := by omega
  obtain ⟨_, c₀, c₁, hf, h₀, h₁⟩ := hall (k / 2) hi
  have htab : (tabsOf sts)[k / 2] = Setup.ofCircuit sts[k / 2]! := by
    rw [getElem!_pos sts (k / 2) hi']; exact Array.getElem_map ..
  rw [htab] at h₀ h₁ hf
  simp only [repStreams_getElem spec (hk₂ k hk)] at hfind
  refine ⟨sess, schs[k / 2].1, schs[k / 2].2, stream, ?_⟩
  rcases Nat.mod_two_eq_zero_or_one k with h | h
  · rw [show 2 * (k / 2) = k by omega] at h₀
    rw [h] at hfind ⊢
    exact ⟨c₀, hpts.symm, hpn, hf, h₀, hfind, hfa⟩
  · rw [show 2 * (k / 2) + 1 = k by omega] at h₁
    rw [h] at hfind ⊢
    exact ⟨c₁, hpts.symm, hpn, hf, h₁, hfind, hfa⟩

/-- **Each masked rep stream has its table's `repRounds`**: when the masked verdict accepts the record at any hashes and
`liveOf` reads it, there are two rep streams per table, and stream `k` (table `k / 2`'s rep `k % 2`) has
`repRounds true spec sts[k / 2]` rounds. -/
theorem streams_size_zk (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : Zk.verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].size = repRounds true spec sts[k / 2]! := by
  obtain ⟨hsz, hall⟩ := rep_zk hs hv hl
  refine ⟨hsz, fun k hk => ?_⟩
  obtain ⟨sess, sch, tPad, stream, c, -, hpn, hf, hrep, hfind, hfa⟩ := hall k hk
  obtain ⟨stream', extras, hfind', hex, hsize⟩ := zkVerifyRepWith_rounds hs hf hrep
  rw [hfind] at hfind'
  cases hfind'
  have hx : extras.size = sts[k / 2]!.regions.size * sess.points.size := extraClaims_size hex
  rw [← Array.length_toList, hfa.length_eq, Array.length_toList, hsize]
  unfold repRounds
  simp only [↓reduceIte, hf]
  rw [hx, hpn, Nat.mul_comm]
  rfl

/-- **Each rep stream has its table's `repRounds`**, plain or masked: when the shipped verdict accepts the record at any
hashes and `liveOf` reads it. -/
theorem streams_size (zk : Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : verdictM zk hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].size = repRounds zk spec sts[k / 2]! := by
  cases zk
  · exact streams_size_plain hs hv hl
  · exact streams_size_zk hs hv hl

/-- The session's recorded rounds, table by table. -/
theorem rounds_sum (zk : Bool) (spec : SessionSpec) (sts : Array Stmt) :
    rounds zk spec sts = ((List.range sts.size).map fun i => 2 * repRounds zk spec sts[i]!).sum := by
  unfold rounds
  rw [← Array.foldl_toList, foldl_add_sum, Nat.zero_add]
  congr 1
  apply List.ext_getElem
  · simp
  · intro i h₁ h₂
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
    rw [getElem!_pos sts i (by simpa using h₁)]

/-- **Wherever `Deploy.V` accepts, the prover sent the rounds of the session the statement sets up against `H`**,
plain or masked: against any oracle `H` and with one coin per message, the link message, one message per round the
shipped verifier's reps read from the record (`verifyRepWith_rounds`, `zkVerifyRepWith_rounds`: each table's two reps,
each `repRounds zk` rounds, at the setup with `hOf H`), and the last message. -/
theorem next_acc_length_setup (zk : Bool) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} (hcs : cs.length = as.length) (h : ((next zk x as cs).eval H).1 = .acc) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (spec' : SessionSpec) (sts' : Array Stmt),
      setupOf zk x = some (s, spec, sts) ∧ setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        as.length = 2 + rounds zk spec' sts' := by
  cases ha : act zk x as cs with
  | rej => rw [next_rej H ha] at h; cases h
  | coin c => rw [next_coin H ha] at h; cases h
  | decide s spec sts link c₁ mids f =>
  obtain ⟨spec', sts', hsu, hv, hc⟩ := (next_decide H ha).1 h
  refine ⟨s, spec, sts, spec', sts', act_decide ha, hsu, ?_⟩
  obtain ⟨lv, hlv, hcus⟩ := checks_custody hc
  have hv' : verdictM (M := Flock.V) zk (Hashes.ofSha512 (Claims.hOf H)) spec' (tabsOf sts') f.record f.proofs =
      .ok () := by
    cases zk <;> exact hv
  obtain ⟨hsz, hk⟩ := streams_size zk (Hashes.ofSha512 (Claims.hOf H)) hv' hlv
  rw [act_decide_length hcs ha, ← custody_length hcus, rounds_sum, List.length_flatMap,
    Nat.add_comm _ 2, ← sum_pairs (fun i => repRounds zk spec' sts'[i]!)]
  congr 2
  apply List.ext_getElem
  · simp [hsz]
  · intro k h₁ h₂
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList, Array.length_toList]
    exact hk k _


namespace KS.Gen

variable [S : ZkSetup]

/-- **Wherever `Deploy.V` accepts, the prover sent the rounds of the session the statement sets up against `H`**,
plain or masked: against any oracle `H` and with one coin per message, the link message, one message per round the
shipped verifier's reps read from the record (`verifyRepWith_rounds`, `zkVerifyRepWith_rounds`: each table's two reps,
each `repRounds zk` rounds, at the setup with `hOf H`), and the last message. -/
theorem next_acc_length_setup (zk : Bool) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} (hcs : cs.length = as.length) (h : ((next zk x as cs).eval H).1 = .acc) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (spec' : SessionSpec) (sts' : Array Stmt),
      setupOf zk x = some (s, spec, sts) ∧ setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        as.length = 2 + rounds zk spec' sts' := by
  cases ha : act zk x as cs with
  | rej => rw [next_rej H ha] at h; cases h
  | coin c => rw [next_coin H ha] at h; cases h
  | decide s spec sts link c₁ mids f =>
  obtain ⟨spec', sts', hsu, hv, hc⟩ := (next_decide H ha).1 h
  refine ⟨s, spec, sts, spec', sts', act_decide ha, hsu, ?_⟩
  obtain ⟨lv, hlv, hcus⟩ := checks_custody hc
  have hv' : verdictM (M := Flock.V) zk (Hashes.ofSha512 (Claims.hOf H)) spec' (tabsOf sts') f.record f.proofs =
      .ok () := by
    cases zk <;> exact hv
  obtain ⟨hsz, hk⟩ := streams_size zk (Hashes.ofSha512 (Claims.hOf H)) hv' hlv
  rw [act_decide_length hcs ha, ← custody_length hcus, rounds_sum, List.length_flatMap,
    Nat.add_comm _ 2, ← sum_pairs (fun i => repRounds zk spec' sts'[i]!)]
  congr 2
  apply List.ext_getElem
  · simp [hsz]
  · intro k h₁ h₂
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList, Array.length_toList]
    exact hk k _

end KS.Gen

/-- **`Deploy.V`'s round count is the rounds the shipped verifier reads**, plain or masked: wherever the verifier
accepts, against an oracle `H` at which the statement's setup has the rounds of its setup at SHA-512 (`hr`), and with one
coin per message, the prover sent `(V zk).k x` messages (`next_acc_length_setup`). -/
theorem next_acc_length (zk : Bool) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} (hcs : cs.length = as.length)
    (hr : ∀ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (spec' : SessionSpec) (sts' : Array Stmt),
      setupOf zk x = some (s, spec, sts) → setupWith (Claims.hOf H) s.read = .ok (spec', sts') →
        rounds zk spec' sts' = rounds zk spec sts)
    (h : ((next zk x as cs).eval H).1 = .acc) : as.length = (V zk).k x := by
  obtain ⟨s, spec, sts, spec', sts', hso, hsu, hl⟩ := next_acc_length_setup zk H hcs h
  show as.length = (match setupOf zk x with | some (_, spec, sts) => 2 + rounds zk spec sts | none => 1)
  rw [hso]
  dsimp only
  rw [hl, hr s spec sts spec' sts' hso hsu]


namespace KS.Gen

variable [S : ZkSetup]

/-- **`Deploy.V`'s round count is the rounds the shipped verifier reads**, plain or masked: wherever the verifier
accepts, against an oracle `H` at which the statement's setup has the rounds of its setup at SHA-512 (`hr`), and with one
coin per message, the prover sent `(V zk).k x` messages (`next_acc_length_setup`). -/
theorem next_acc_length (zk : Bool) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} (hcs : cs.length = as.length)
    (hr : ∀ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (spec' : SessionSpec) (sts' : Array Stmt),
      setupOf zk x = some (s, spec, sts) → setupWith (Claims.hOf H) s.read = .ok (spec', sts') →
        rounds zk spec' sts' = rounds zk spec sts)
    (h : ((next zk x as cs).eval H).1 = .acc) : as.length = (V zk).k x := by
  obtain ⟨s, spec, sts, spec', sts', hso, hsu, hl⟩ := next_acc_length_setup zk H hcs h
  show as.length = (match setupOf zk x with | some (_, spec, sts) => 2 + rounds zk spec sts | none => 1)
  rw [hso]
  dsimp only
  rw [hl, hr s spec sts spec' sts' hso hsu]

end KS.Gen

namespace KS.Gen

variable [S : ZkSetup]

/-- **At the SHA-512 oracle, `Deploy.V`'s round count is the rounds the shipped verifier reads**: wherever the verifier
accepts against `sha512`, with one coin per message, the prover sent `(V zk).k x` messages. -/
theorem next_acc_length_sha512 (zk : Bool) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (hcs : cs.length = as.length) (h : ((next zk x as cs).eval sha512).1 = .acc) : as.length = (V zk).k x :=
  next_acc_length zk sha512 hcs (fun s spec sts spec' sts' hso hsu => by
    rw [hOf_sha512, setupWith_sha512, Ins.setup_read (setupOf_some hso).2] at hsu
    cases hsu
    rfl) h

end KS.Gen

/-- **At the SHA-512 oracle, `Deploy.V`'s round count is the rounds the shipped verifier reads**: wherever the verifier
accepts against `sha512`, with one coin per message, the prover sent `(V zk).k x` messages. -/
theorem next_acc_length_sha512 (zk : Bool) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (hcs : cs.length = as.length) (h : ((next zk x as cs).eval sha512).1 = .acc) : as.length = (V zk).k x :=
  KS.Gen.next_acc_length_sha512 (S := KS.S₀) (zk := zk) (x := x) (as := as) (cs := cs) (hcs := hcs) (h := h)

end FlockDeploy
