import Proofs.Flock.Claims.DeployRounds
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Soundness.Discharge.Exec.Tables

/-!
# `Deploy.V`'s round count is the rounds the shipped verifier reads
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

/-- **The plain verdict accepts each table's reps**: when `verifyWithM hs` accepts, whatever the hashes `hs`, the record
decodes, there is one setup per table, and each table `i` has its `fast100` schedule and both reps' checks accept on
the proof files `proofs[2i]` and `proofs[2i+1]`. -/
theorem verifyWithM_tables (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} (h : verifyWithM hs spec tabs record proofs = .ok ()) :
    ∃ (sess : Session) (schs : Array Schedule), Record.decodeWith hs spec record = .ok sess ∧
      tabs.size = spec.tables.size ∧ schs.size = tabs.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (hs' : i < schs.size) (c₀ c₁ : ByteArray),
        fast100 tabs[i].m = some schs[i] ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 0 proofs[2 * i]! = .ok c₀ ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 1 proofs[2 * i + 1]! = .ok c₁ := by
  unfold verifyWithM at h
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

/-- **What the verifier's decoder reads of the record**, whatever the hashes: the JSON, the link's points (S9: the
configured number of them) and the streams, each with its name and its rounds' coin counts and coins. -/
theorem decodeWith_parts (hs : Hashes Flock.V) {spec : SessionSpec} {text : String} {sess : Session}
    (h : Record.decodeWith hs spec text = .ok sess) :
    ∃ (j l : Lean.Json) (ps recs : Array Lean.Json) (SF : Lean.Json → Flock.V StreamRec),
      Lean.Json.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.arr l "points" = .ok ps ∧
      ps.mapM (fun p => (match p.getArr? with
        | .ok a => Record.f128s a
        | .error _ => .error "record: a link point is not a list" : Flock.V (Array F128))) = .ok sess.points ∧
      sess.points.size = spec.points ∧
      Record.arr j "streams" = .ok recs ∧ recs.mapM SF = .ok sess.streams ∧ ∀ s x, SF s = .ok x → StreamR s x := by
  unfold Record.decodeWith at h
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

/-- **The two readers agree**: when `liveOf` and the verifier's decoder (at any hashes) both read the record, they read
the same link points, the configured number of them, and for each rep stream in `spec.repStreams` order the verifier's
stream of that name has the rounds `liveOf` read, with the same coin counts and coins. -/
theorem liveOf_decodeWith (hs : Hashes Flock.V) {spec : SessionSpec} {text : String} {lv : Live} {sess : Session}
    (hl : liveOf spec text = .ok lv) (hd : Record.decodeWith hs spec text = .ok sess) :
    lv.points = sess.points ∧ sess.points.size = spec.points ∧ lv.streams.size = spec.repStreams.size ∧
      ∀ (k : ℕ) (hk : k < spec.repStreams.size) (hk' : k < lv.streams.size), ∃ stream : StreamRec,
        sess.streams.find? (fun x => x.name == spec.repStreams[k].2.2) = some stream ∧
        List.Forall₂ (fun a b => a.n = b.n ∧ a.coins = b.coins) lv.streams[k].toList stream.rounds.toList := by
  obtain ⟨j, l, ps, recs, hj, hl', hps, hpts, hrecs, hsz, hall⟩ := liveOf_parts hl
  obtain ⟨j₂, l₂, ps₂, recs₂, SF, hj₂, hl₂, hps₂, hpts₂, hpn, hrecs₂, hSF, hR⟩ := decodeWith_parts hs hd
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

end FlockDeploy
