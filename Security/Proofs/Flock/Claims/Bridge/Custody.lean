import Proofs.Flock.Claims.Deploy
import Proofs.Flock.Soundness.Assumptions.Zk
import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
import Proofs.Flock.Soundness.Refine.Walk
import Proofs.Flock.Soundness.Refine.Table

/-!
# `Deploy.V`'s custody checks, as the old theorems' `RecordsLiveZKJ`

The old `--zk` theorems bound flock-verify `--zk`'s acceptance of the transcript a writer `wr` makes of the live
session's outcome, under `RecordsLiveZKJ I dj Jf wr` (`DefsJ.lean`, the record half of the `live-verifier` assumption
`RecordCustodyZKJ`): the record `verify --zk` accepts carries the draw, the outcome's proof files, link points, roots
and rounds. `Deploy.V` takes the record in its last message (`Final`) and checks its custody (`checks`, `custody`):
the record's roots, link points and rounds are the run's link message, first coin, round messages and their coins.

* `liveOf_ok`, `decode_ok`, `decode_live`: what `liveOf` and `Record.decode` read of the record, field by field, and
  that the decoded session's roots, points and each rep stream's rounds are `liveOf`'s, with a retained round's recorded
  digest that of its bytes (S12).
* `checks_ok`, `custody_ok`, `checks_draw`: what the checks establish, the record's draw the statement's whatever its
  key order among it (`drawEq`, `JsonEquiv`).
* `recordDraw_of_checks`, `recordDraw_of_decide`: and so the record's draw is the statement's as a value, since the
  verifier rebuilds both (`Flock.norm`, `JsonNorm`): the `hdraw` of `setupOf_record` and `recordsLiveAt_of_checks`.
* `viewOf`: the run's live outcome (`LiveOutJ`), its rounds the record's (`viewOf_rounds`: the round messages with
  their coins, in `spec.repStreams` order). Each round's coin count and each stream's share of the rounds are the
  record's: a round message carries no count.
* `publicInputs_none_of_checks`: the checks run the scope check, which refuses `--public-inputs` (which `Exec.Inputs`
  does not carry); `publicInputs_none_of_inScope`: so does `InScope`, at the verifier's reading of the statement.
* `setupOf_record`: where the checks pass and the record's draw is the statement's as a value, the setup the shipped
  verifier makes of the record, at its own draw, is `Deploy.V`'s at the statement's.
* `recordsLiveAt_of_checks`: where the checks pass and the record's draw is the statement's, the last message's record
  and proof files record `viewOf` (`RecordsLiveAt`, `RecordsLiveZKJ` at one outcome).
* `wrD`, `recordsLiveZKJ_wrD`, `acceptsZK_wrD`: the writer that, at an outcome, takes a transcript `verify --zk`
  accepts that records it, where one exists. It satisfies `RecordsLiveZKJ` (so `RecordCustodyZKJ` is its draw file
  half alone, `recordCustodyZKJ_wrD`), and on every run whose decision accepts at SHA-512, `verify --zk` accepts what it
  writes for the run's outcome (`AcceptsZK`), the event the old theorems bound.
* `decodesOnZKJ_wrD`, `decodesOnZKJ_wrD_file`: at `wrD` the deployed `--zk` decoders are right (`DecodesOnZKJ`,
  `LiveZk.Pr_acc_zk_le`'s `hdec`) on untyped inputs under `TagsOkZJ` where every draw sets each table up at `mPts`
  link points (`DrawOkZKJ`), as the draw file half of `RecordCustodyZKJ` gives.
-/

namespace Claims.Bridge.Custody

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge
open FlockSoundness.Refine (ebind_ok check_ok_iff bytes_eq_of_beq Ret ptsOf)
open Lean (Json)

local instance lawfulBEqByteArrayCustody : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-! ## Lists -/

theorem list_mapM_forall₂ {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → List.Forall₂ (fun a b => f a = .ok b) l l'
  | [], l', h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact .nil
  | a :: l, l', h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := ebind_ok h
    obtain ⟨bs, hbs, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact .cons hb (list_mapM_forall₂ hbs)

theorem mapM_forall₂ {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β} (h : xs.mapM f = .ok ys) :
    List.Forall₂ (fun a b => f a = .ok b) xs.toList ys.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    simp only [Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    simpa using list_mapM_forall₂ hl

theorem mapIdxM_go_forall₂ {α β ε : Type} {f : ℕ → α → Except ε β} :
    ∀ {l : List α} {acc : Array β} {l' : List β}, List.mapIdxM.go f l acc = .ok l' →
      ∃ l'', l' = acc.toList ++ l'' ∧ List.Forall₂ (fun a b => ∃ i, f i a = .ok b) l l''
  | [], acc, l', h => by
    simp only [List.mapIdxM.go, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨[], by simp [h], .nil⟩
  | a :: l, acc, l', h => by
    simp only [List.mapIdxM.go] at h
    obtain ⟨b, hb, h⟩ := ebind_ok h
    obtain ⟨l'', rfl, h2⟩ := mapIdxM_go_forall₂ h
    exact ⟨b :: l'', by simp, .cons ⟨_, hb⟩ h2⟩

theorem mapIdxM_forall₂ {α β ε : Type} {f : ℕ → α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapIdxM f = .ok ys) : List.Forall₂ (fun a b => ∃ i, f i a = .ok b) xs.toList ys.toList := by
  have e := Array.toList_mapIdxM (xs := xs) (f := f)
  rw [h] at e
  simp only [Functor.map, Except.map] at e
  obtain ⟨l'', hl, h2⟩ := mapIdxM_go_forall₂ (f := f) (l := xs.toList) (acc := #[])
    (by simpa [List.mapIdxM] using e.symm)
  simpa [hl] using h2

theorem forall₂_unique {α β : Type} {R : α → β → Prop} (hR : ∀ a b c, R a b → R a c → b = c) :
    ∀ {l : List α} {l₁ l₂ : List β}, List.Forall₂ R l l₁ → List.Forall₂ R l l₂ → l₁ = l₂
  | _, _, _, .nil, .nil => rfl
  | _, _, _, .cons h₁ t₁, .cons h₂ t₂ => by rw [hR _ _ _ h₁ h₂, forall₂_unique hR t₁ t₂]

theorem forall₂_find? {α β : Type} {R : α → β → Prop} {p : α → Bool} {q : β → Bool}
    (hpq : ∀ a b, R a b → q b = p a) :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → ∀ {a : α}, l.find? p = some a →
      ∃ b, R a b ∧ l'.find? q = some b
  | _, _, .nil, _, h => by cases h
  | a :: _, b :: _, .cons hab hl, a', h => by
    by_cases hp : p a = true
    · rw [List.find?_cons_of_pos hp] at h
      cases h
      exact ⟨b, hab, List.find?_cons_of_pos (by rw [hpq a b hab]; exact hp)⟩
    · rw [List.find?_cons_of_neg hp] at h
      obtain ⟨b', h1, h2⟩ := forall₂_find? hpq hl h
      exact ⟨b', h1, by rw [List.find?_cons_of_neg (by rw [hpq a b hab]; exact hp)]; exact h2⟩

theorem forall₂_mem_left {α β : Type} {R : α → β → Prop} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → ∀ a ∈ l, ∃ b ∈ l', R a b
  | _, _, .nil, _, h => by cases h
  | _, _, .cons hab hl, a, h => by
    rcases List.mem_cons.1 h with rfl | h
    · exact ⟨_, by simp, hab⟩
    · obtain ⟨b, hb, h⟩ := forall₂_mem_left hl a h
      exact ⟨b, List.mem_cons_of_mem _ hb, h⟩

theorem forall₂_left {α β γ : Type} {R : α → β → Prop} {S : α → γ → Prop} :
    ∀ {l : List α} {l₁ : List β} {l₂ : List γ}, List.Forall₂ R l l₁ → List.Forall₂ S l l₂ →
      List.Forall₂ (fun b c => ∃ a, R a b ∧ S a c) l₁ l₂
  | _, _, _, .nil, .nil => .nil
  | _, _, _, .cons h₁ t₁, .cons h₂ t₂ => .cons ⟨_, h₁, h₂⟩ (forall₂_left t₁ t₂)

theorem forall₂_map_eq {α β : Type} {R : α → β → Prop} {f : β → α} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → (∀ a b, b ∈ l' → R a b → a = f b) → l = l'.map f
  | _, _, .nil, _ => rfl
  | _, _, .cons hab hl, H => by
    rw [List.map_cons, H _ _ (by simp) hab,
      forall₂_map_eq hl fun a b hb h => H a b (List.mem_cons_of_mem _ hb) h]

theorem forall₂_getElem {α β : Type} [Inhabited β] {R : α → β → Prop} {xs : Array α} {ys : Array β}
    (h : List.Forall₂ R xs.toList ys.toList) {i : ℕ} (hi : i < xs.size) : R xs[i] ys[i]! := by
  have hl := h.length_eq
  simp only [Array.length_toList] at hl
  rw [getElem!_pos ys i (hl ▸ hi)]
  have := (List.forall₂_iff_get.1 h).2 i (by simpa using hi) (by simpa [hl] using hi)
  simpa using this

/-! ## The record's fields -/

/-- Table `t`'s `root_B` as the record's link has it. -/
def RootField (l : Json) (t : String) (rb : ByteArray) : Prop :=
  ∃ v s, Record.fld l "root_b" = .ok v ∧ Record.str v t = .ok s ∧ unhex s = .ok rb

/-- A link point as the record has it. -/
def PointField (p : Json) (pt : Array F128) : Prop := ∃ a, p.getArr? = .ok a ∧ Record.f128s a = .ok pt

/-- A round's coin count, coins and retained bytes as the record has them. -/
def RoundFields (r : Json) (n : ℕ) (coins : Array F128) (bytes : Option ByteArray) : Prop :=
  Record.nat r "n" = .ok n ∧ (∃ cs, Record.arr r "coins" = .ok cs ∧ Record.f128s cs = .ok coins) ∧
    (Record.isNull r "msg" = true → bytes = none) ∧
    (Record.isNull r "msg" = false → ∃ s b, Record.str r "msg" = .ok s ∧ unhex s = .ok b ∧ bytes = some b)

/-- A stream as `Record.decode` reads it: its name, and its rounds with each retained round's digest. -/
def DecStream (spec : SessionSpec) (x : Json) (y : StreamRec) : Prop :=
  Record.str x "stream" = .ok y.name ∧ ∃ rs, Record.arr x "rounds" = .ok rs ∧
    List.Forall₂ (fun r rd => RoundFields r rd.n rd.coins rd.bytes ∧ ∀ b, rd.bytes = some b → rd.sha = spec.dg b)
      rs.toList y.rounds.toList

/-- The rep stream `w` as `liveOf` reads it from the record's streams `recs`. -/
def LiveStream (recs : Array Json) (w : ℕ × ℕ × String) (rds : Array Round) : Prop :=
  ∃ x, recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some w.2.2) = some x ∧
    ∃ rs, Record.arr x "rounds" = .ok rs ∧
      List.Forall₂ (fun r rd => RoundFields r rd.n rd.coins rd.bytes ∧ rd.sha = ByteArray.empty) rs.toList rds.toList

theorem rootField_unique {l : Json} {t : String} {a b : ByteArray} (ha : RootField l t a) (hb : RootField l t b) :
    a = b := by
  obtain ⟨v, s, h1, h2, h3⟩ := ha
  obtain ⟨v', s', h1', h2', h3'⟩ := hb
  rw [h1] at h1'
  cases h1'
  rw [h2] at h2'
  cases h2'
  rw [h3] at h3'
  cases h3'
  rfl

theorem pointField_unique {p : Json} {a b : Array F128} (ha : PointField p a) (hb : PointField p b) : a = b := by
  obtain ⟨c, h1, h2⟩ := ha
  obtain ⟨c', h1', h2'⟩ := hb
  rw [h1] at h1'
  cases h1'
  rw [h2] at h2'
  cases h2'
  rfl

theorem roundFields_unique {r : Json} {n n' : ℕ} {c c' : Array F128} {b b' : Option ByteArray}
    (h : RoundFields r n c b) (h' : RoundFields r n' c' b') : n = n' ∧ c = c' ∧ b = b' := by
  obtain ⟨hn, ⟨cs, hcs, hc⟩, hb0, hb1⟩ := h
  obtain ⟨hn', ⟨cs', hcs', hc'⟩, hb0', hb1'⟩ := h'
  refine ⟨Except.ok.inj (hn.symm.trans hn'), ?_, ?_⟩
  · rw [hcs] at hcs'
    cases hcs'
    exact Except.ok.inj (hc.symm.trans hc')
  · cases hnull : Record.isNull r "msg"
    · obtain ⟨s, b₁, hs, hb₁, rfl⟩ := hb1 hnull
      obtain ⟨s', b₂, hs', hb₂, rfl⟩ := hb1' hnull
      rw [hs] at hs'
      cases hs'
      rw [hb₁] at hb₂
      cases hb₂
      rfl
    · rw [hb0 hnull, hb0' hnull]

/-- `liveOf`'s test of a stream's name is `Record.decode`'s, on a stream `decode` reads. -/
theorem stream_pred {x : Json} {name w : String} (h : Record.str x "stream" = .ok name) :
    ((x.getObjValAs? String "stream").toOption == some w) = (name == w) := by
  unfold Record.str Record.fld at h
  cases hv : x.getObjVal? "stream" with
  | error e => rw [hv] at h; cases h
  | ok v =>
    rw [hv] at h
    simp only [bind, Except.bind] at h
    have hs : v.getStr? = .ok name := by
      split at h
      · rename_i s hs; cases h; exact hs
      · cases h
    simp only [Json.getObjValAs?, Json.getObjValD, hv, Except.toOption]
    show ((Json.getStr? v).toOption == some w) = (name == w)
    rw [hs]
    rfl

/-- **What `liveOf` reads**: the record's JSON, its link, its streams, each table's `root_B`, the link points and each
rep stream's rounds, field by field. -/
theorem liveOf_ok {spec : SessionSpec} {record : String} {lv : Live} (h : liveOf spec record = .ok lv) :
    ∃ j l recs, Flock.JsonParse.parse record = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.arr j "streams" = .ok recs ∧
      List.Forall₂ (RootField l) spec.tables.toList lv.rootB.toList ∧
      (∃ ps, Record.arr l "points" = .ok ps ∧ List.Forall₂ PointField ps.toList lv.points.toList) ∧
      List.Forall₂ (LiveStream recs) spec.repStreams.toList lv.streams.toList := by
  revert lv
  show Ret (liveOf spec record) _
  rw [liveOf.eq_def]
  cases hp : Flock.JsonParse.parse record with
  | error e => intro lv h; cases h
  | ok j =>
    dsimp only
    rw [pure_bind]
    ret_walk [Record.fld, Record.arr, Array.mapM]
    refine ⟨j, _, _, rfl, by assumption, by assumption, ?_, ⟨_, by assumption, ?_⟩, ?_⟩
    · refine List.Forall₂.imp (fun t rb h => ?_) (mapM_forall₂ (by assumption))
      beta_reduce at h
      obtain ⟨v, h1, h⟩ := ebind_ok h
      obtain ⟨s, h2, h3⟩ := ebind_ok h
      exact ⟨v, s, h1, h2, h3⟩
    · refine List.Forall₂.imp (fun p pt h => ?_) (mapM_forall₂ (by assumption))
      beta_reduce at h
      split at h
      · rename_i a ha
        exact ⟨a, ha, h⟩
      · cases h
    · refine List.Forall₂.imp (fun w rds h => ?_) (mapM_forall₂ (by assumption))
      beta_reduce at h
      split at h
      · rename_i s hs
        obtain ⟨rs, h1, h⟩ := ebind_ok h
        refine ⟨s, hs, rs, h1, List.Forall₂.imp (fun r rd h => ?_) (mapM_forall₂ h)⟩
        beta_reduce at h
        obtain ⟨n, hn, h⟩ := ebind_ok h
        obtain ⟨cs, hcs, h⟩ := ebind_ok h
        obtain ⟨coins, hcoins, h⟩ := ebind_ok h
        by_cases hnull : Record.isNull r "msg" = true
        · rw [ite_eq_left hnull] at h
          obtain ⟨b, hb, h⟩ := ebind_ok h
          simp only [pure, Except.pure, Except.ok.injEq] at hb h
          subst hb h
          exact ⟨⟨hn, ⟨cs, hcs, hcoins⟩, fun _ => rfl, fun h' => by simp [hnull] at h'⟩, rfl⟩
        · rw [ite_eq_right hnull] at h
          obtain ⟨s', hs', h⟩ := ebind_ok h
          obtain ⟨b, hb, h⟩ := ebind_ok h
          cases hu : unhex s' with
          | error e => rw [hu] at hb; cases hb
          | ok u =>
            rw [hu] at hb
            simp only [Functor.map, Except.map, pure, Except.pure, Except.ok.injEq] at hb h
            subst hb h
            exact ⟨⟨hn, ⟨cs, hcs, hcoins⟩, fun h' => absurd h' hnull,
              fun _ => ⟨s', u, hs', hu, rfl⟩⟩, rfl⟩
      · cases h

/-- **What `Record.decode` reads**, of what `liveOf` reads: the record's JSON, its link, its streams, each table's
`root_B`, the link points, and each stream's name and rounds, with a retained round's recorded digest that of its bytes
(S12). -/
theorem decode_ok {spec : SessionSpec} {text : String} {sess : RunRecord} (h : Record.decode spec text = .ok sess) :
    ∃ j l recs, Flock.JsonParse.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.arr j "streams" = .ok recs ∧
      List.Forall₂ (RootField l) spec.tables.toList sess.rootB.toList ∧
      (∃ ps, Record.arr l "points" = .ok ps ∧ List.Forall₂ PointField ps.toList sess.points.toList) ∧
      List.Forall₂ (DecStream spec) recs.toList sess.streams.toList := by
  revert sess
  show Ret (Record.decode spec text) _
  rw [Record.decode.eq_def]
  cases hp : Flock.JsonParse.parse text with
  | error e => intro sess h; cases h
  | ok j =>
    simp only []
    rw [pure_bind]
    ret_walk [Record.fld, Record.arr, Array.mapM]
    all_goals
      refine ⟨j, _, _, rfl, by assumption, by assumption, ?_, ⟨_, by assumption, ?_⟩, ?_⟩
      · refine List.Forall₂.imp (fun t rb h => ?_) (mapM_forall₂ (by assumption))
        beta_reduce at h
        obtain ⟨v, h1, h⟩ := ebind_ok h
        obtain ⟨s, h2, h⟩ := ebind_ok h
        obtain ⟨rb', h3, h⟩ := ebind_ok h
        repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (simp only [] at h))
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact ⟨v, s, h1, h2, h3⟩
      · refine List.Forall₂.imp (fun p pt h => ?_) (mapM_forall₂ (by assumption))
        beta_reduce at h
        split at h
        · rename_i a ha
          exact ⟨a, ha, h⟩
        · cases h
      · refine List.Forall₂.imp (fun x y h => ?_) (mapM_forall₂ (by assumption))
        beta_reduce at h
        repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (simp only [] at h))
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        refine ⟨by assumption, _, by assumption, List.Forall₂.imp (fun r rd h => ?_) (mapIdxM_forall₂ (by assumption))⟩
        obtain ⟨k, h⟩ := h
        repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (simp only [] at h))
        by_cases hnull : Record.isNull r "msg" = true
        · rw [ite_eq_left hnull] at h
          obtain ⟨b, hb, h⟩ := ebind_ok h
          simp only [pure, Except.pure, Except.ok.injEq] at hb h
          subst hb h
          exact ⟨⟨by assumption, ⟨_, by assumption, by assumption⟩, fun _ => rfl, fun h' => by simp [hnull] at h'⟩,
            fun b hb => by cases hb⟩
        · rw [ite_eq_right hnull] at h
          obtain ⟨s', hs', h⟩ := ebind_ok h
          obtain ⟨b, hb, h⟩ := ebind_ok h
          obtain ⟨len, hlen, h⟩ := ebind_ok h
          obtain ⟨u, hck, h⟩ := ebind_ok h
          obtain ⟨b', hb', h⟩ := ebind_ok h
          simp only [pure, Except.pure, Except.ok.injEq] at hb' h
          subst hb' h
          have hdg := check_ok_iff.1 hck
          simp only [Bool.and_eq_true] at hdg
          exact ⟨⟨by assumption, ⟨_, by assumption, by assumption⟩, fun h' => absurd h' hnull,
            fun _ => ⟨s', b, hs', hb, rfl⟩⟩, fun b₀ hb₀ => by cases hb₀; exact (bytes_eq_of_beq hdg.1).symm⟩

/-- **The decoded session's live part is `liveOf`'s**: the same `root_B` and link points, and at each rep stream the
stream `Record.decode` finds by its name, whose rounds are `liveOf`'s with each retained round's recorded digest that
of its bytes. -/
theorem decode_live {spec : SessionSpec} {text : String} {sess : RunRecord} {lv : Live}
    (hd : Record.decode spec text = .ok sess) (hl : liveOf spec text = .ok lv) :
    sess.rootB = lv.rootB ∧ sess.points = lv.points ∧ lv.streams.size = spec.repStreams.size ∧
      ∀ q (hq : q < spec.repStreams.size), ∃ y, sess.streams.find? (·.name == spec.repStreams[q].2.2) = some y ∧
        List.Forall₂ (fun rd rd' => rd.n = rd'.n ∧ rd.coins = rd'.coins ∧ rd.bytes = rd'.bytes ∧
          ∀ b, rd.bytes = some b → rd.sha = spec.dg b) y.rounds.toList (lv.streams[q]!).toList := by
  obtain ⟨j, l, recs, hp, hl1, hr, hroot, ⟨ps, hps, hpts⟩, hstr⟩ := decode_ok hd
  obtain ⟨j', l', recs', hp', hl1', hr', hroot', ⟨ps', hps', hpts'⟩, hstr'⟩ := liveOf_ok hl
  rw [hp] at hp'
  cases hp'
  rw [hl1] at hl1'
  cases hl1'
  rw [hr] at hr'
  cases hr'
  rw [hps] at hps'
  cases hps'
  refine ⟨Array.ext' (forall₂_unique (R := RootField l) (fun _ _ _ => rootField_unique) hroot hroot'),
    Array.ext' (forall₂_unique (R := PointField) (fun _ _ _ => pointField_unique) hpts hpts'),
    by simpa using hstr'.length_eq.symm,
    fun q hq => ?_⟩
  obtain ⟨x, hx, rs, hrs, hrds⟩ := forall₂_getElem hstr' hq
  rw [← Array.find?_toList] at hx
  obtain ⟨y, ⟨-, rs', hrs', hdr⟩, hy⟩ := forall₂_find? (q := (·.name == spec.repStreams[q].2.2))
    (fun a b hab => (stream_pred hab.1).symm) hstr hx
  rw [hrs] at hrs'
  cases hrs'
  refine ⟨y, by rw [← Array.find?_toList]; exact hy, (forall₂_left hdr hrds).imp fun rd rd' h => ?_⟩
  obtain ⟨r, ⟨h1, h2⟩, h3, -⟩ := h
  obtain ⟨hn, hc, hb⟩ := roundFields_unique h1 h3
  exact ⟨hn, hc, hb, h2⟩

/-! ## The custody checks -/

/-- **What `checks` establishes**: the last message's statement is `x`, the record's live session reads (`liveOf`),
and its custody passes; the record's draw is the statement's up to key order (`checks_draw`). -/
theorem checks_ok {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final} (h : checks zk s spec x link c₁ mids f = .ok ()) :
    f.stmt = x ∧ ∃ lv, liveOf spec f.record = .ok lv ∧ custody spec lv link c₁ mids = .ok () := by
  unfold checks at h
  obtain ⟨_, h1, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨lv, h3, h⟩ := ebind_ok h
  obtain ⟨_, h4, -⟩ := ebind_ok h
  exact ⟨beq_iff_eq.1 (check_ok_iff.1 h1), lv, h3, h4⟩

/-- **The checks hold the record to its canonical text** (S0, last): it is what `printT` prints of what it parses to
(`Flock.RecordText.check_ok`). -/
theorem checks_canonical {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) : Flock.RecordText.check f.record = .ok () := by
  unfold checks at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  exact h

/-- **The checks pin the record's draw to the statement's, whatever its key order**: `drawEq` passes exactly on
equivalent values (`drawEq_iff`). -/
theorem checks_draw {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) :
    Option.Rel JsonEquiv (Exec.recordDraw f.record) s.read.draw := by
  unfold checks at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, h2, -⟩ := ebind_ok h
  exact (drawEq_iff _ _).1 (check_ok_iff.1 h2)

/-- **A record's draw is rebuilt** (`Flock.norm`, as `Main.recordDraw` keeps it). -/
theorem recordDraw_norm {record : String} {d : Json} (h : Exec.recordDraw record = some d) : Flock.norm d = d := by
  unfold Exec.recordDraw at h
  split at h
  · split at h
    · cases h
    · cases h
      exact norm_norm _
    · cases h
  · cases h

/-- **The record's draw is the statement's, as a value**, wherever the checks pass on a statement the verifier read
(`insOf`): both draws are rebuilt (`recordDraw_norm`, `insOf_draw_norm`), and two rebuilt values the draw check passes
on are equal (`eq_of_rel_norm`). This is the `hdraw` of `setupOf_record` and `recordsLiveAt_of_checks`, with no premise
on the JSON parser. -/
theorem recordDraw_of_checks {zk zk' : Bool} {s : Ins} {spec : SessionSpec} {x x' : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (hs : insOf zk' x' = some s) (h : checks zk s spec x link c₁ mids f = .ok ()) :
    Exec.recordDraw f.record = s.read.draw :=
  eq_of_rel_norm (checks_draw h) (fun _ hd => recordDraw_norm hd) (fun _ hd => (insOf_draw_norm hs hd).1)

/-- **An accepting decision's record carries the statement's draw, as a value** (`recordDraw_of_checks`). -/
theorem recordDraw_of_decide {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (ha : act zk x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next zk x as cs).eval sha512).1 = .acc) : Exec.recordDraw f.record = s.read.draw :=
  recordDraw_of_checks (setupOf_some (act_decide ha)).1 ((next_decide_sha512 ha).1 hacc).2

/-- **What the custody check establishes**: the record's `root_B` and public words are the link message's, its link
points the first coin's, and its rounds, flattened in `spec.repStreams` order, the round messages with their coins: each
round's retained bytes are its message and its coins the first `n` of its coin. -/
theorem custody_ok {spec : SessionSpec} {lv : Live} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} (h : custody spec lv link c₁ mids = .ok ()) :
    lv.rootB.toList = link.1 ∧ lv.publics.toList = link.2 ∧ lv.points = pointsOf spec c₁ ∧
      List.Forall₂ (fun rd am => rd.bytes = some ⟨am.1.toArray⟩ ∧ rd.coins = am.2.vals rd.n)
        (lv.streams.toList.flatMap (·.toList)) mids := by
  unfold custody at h
  obtain ⟨_, h1, h⟩ := ebind_ok h
  obtain ⟨_, h2, h⟩ := ebind_ok h
  obtain ⟨_, h3, h⟩ := ebind_ok h
  obtain ⟨_, h4, h5⟩ := ebind_ok h
  refine ⟨beq_iff_eq.1 (check_ok_iff.1 h1), beq_iff_eq.1 (check_ok_iff.1 h2), beq_iff_eq.1 (check_ok_iff.1 h3),
    List.forall₂_iff_zip.2 ⟨beq_iff_eq.1 (check_ok_iff.1 h4), fun {rd am} hm => ?_⟩⟩
  have := List.all_eq_true.1 (check_ok_iff.1 h5) (rd, am) hm
  obtain ⟨a, c⟩ := am
  simpa only [Bool.and_eq_true, beq_iff_eq] using this

/-! ## No `--public-inputs` -/

/-- **The checks run the verifier's scope check** (`Read.check`, `ProvedScope.check`). -/
theorem checks_scope {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) : s.read.check = .ok () := by
  unfold checks at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, h5, -⟩ := ebind_ok h
  exact h5

/-- **The scope check refuses `--public-inputs`** (`statementRules`' last rule). -/
theorem publicInputs_none_of_check {r : Readable.Read} (h : r.check = .ok ()) : r.publicInputs = none := by
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 h).1
  exact Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])

/-- **A statement `Deploy.V`'s checks pass carries no `--public-inputs`**: the checks run the scope check. -/
theorem publicInputs_none_of_checks {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) : s.read.publicInputs = none :=
  publicInputs_none_of_check (checks_scope h)

/-- **`InScope` excludes `--public-inputs`** at the verifier's reading of the statement: where the claims' parser reads
`x` as `st` and `st` is in scope, the inputs the verifier reads of `x` (at a decision, `setupOf_some`) carry none. -/
theorem publicInputs_none_of_inScope {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {st : Claims.Flock.Stmt} {s : Ins} (hp : FlockDeploy.parse zk rel x = some st)
    (hin : Claims.Flock.InScope zk st) (hs : insOf zk x = some s) : s.read.publicInputs = none := by
  obtain ⟨p, hpb, rfl⟩ := Option.map_eq_some_iff.1 hp
  obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hpb
  have hr' : r = s.read := by
    simp only [readOf, hs, Option.map_some] at hr
    exact (Option.some.inj hr).symm
  obtain ⟨-, -, -, -, -, -, -, -, -, hpi, -⟩ := hin
  simp only [Readable.toReadable, Readable.Read.form, Bool.not_eq_true] at hpi
  rw [← hr', ← Readable.toRead_of_parse hrp]
  exact Option.not_isSome_iff_eq_none.1 (by simp [hpi])

/-- The session's rep streams: table `i`'s rep `r` is stream `2i + r`, named `<table>/rep<r>`. -/
theorem repStreams_toList (spec : SessionSpec) :
    spec.repStreams.toList = (List.range spec.tables.size).flatMap fun i =>
      [(i, 0, s!"{spec.tables[i]!}/rep{0}"), (i, 1, s!"{spec.tables[i]!}/rep{1}")] := by
  simp [SessionSpec.repStreams, Array.toList_flatMap]

theorem flatMap_two_getElem? {β : Type} (g : ℕ → List β) (hg : ∀ i, (g i).length = 2) :
    ∀ (N i r : ℕ), i < N → r < 2 → ((List.range N).flatMap g)[2 * i + r]? = (g i)[r]?
  | 0, _, _, hi, _ => absurd hi (Nat.not_lt_zero _)
  | N + 1, i, r, hi, hr => by
    have hlen : ((List.range N).flatMap g).length = 2 * N := by
      induction N with
      | zero => rfl
      | succ N ih => simp [List.range_succ, List.flatMap_append, hg] at ih ⊢; omega
    rw [List.range_succ, List.flatMap_append]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
    · rw [List.getElem?_append_left (by omega), flatMap_two_getElem? g hg N i r hi hr]
    · rw [List.getElem?_append_right (by omega), hlen]
      simp

theorem repStreams_size (spec : SessionSpec) : spec.repStreams.size = 2 * spec.tables.size := by
  rw [← Array.length_toList, repStreams_toList]
  simp [List.length_flatMap, List.sum_replicate, Nat.mul_comm]

theorem repStreams_getElem (spec : SessionSpec) {i r : ℕ} (hi : i < spec.tables.size) (hr : r < 2) :
    ∃ h : 2 * i + r < spec.repStreams.size,
      spec.repStreams[2 * i + r] = (i, r, toString spec.tables[i]! ++ toString "/rep" ++ toString r) := by
  have hlt : 2 * i + r < spec.repStreams.size := by rw [repStreams_size]; omega
  refine ⟨hlt, ?_⟩
  have e := flatMap_two_getElem? (fun i => [(i, 0, s!"{spec.tables[i]!}/rep{0}"),
    (i, 1, s!"{spec.tables[i]!}/rep{1}")]) (fun _ => rfl) spec.tables.size i r hi hr
  rw [← repStreams_toList, Array.getElem?_toList, Array.getElem?_eq_getElem hlt] at e
  interval_cases r <;> simpa using e

/-! ## The setup -/

/-- Under `--zk` the setup passes every table's `Zk.shapeOk`. -/
theorem setup_shapeOk {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts))
    (hz : s.read.zk = true) : sts.forM Zk.shapeOk = .ok () := by
  unfold Ins.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

/-- **`verify --zk`'s setup of a record whose draw is the statement's** is `Deploy.V`'s, where the statement carries no
`--public-inputs` (`Exec.setupOf` reads none). -/
theorem setupOf_zkInputs {r : Readable.Read} {spec : SessionSpec} {sts : Array Stmt} {record : String}
    (hs : r.setup = .ok (spec, sts)) (hzk : r.zk = true) (hpi : r.publicInputs = none)
    (hdraw : Exec.recordDraw record = r.draw) :
    Exec.setupOf (ZkExec.zkInputs r.I) record = .ok (spec, sts) := by
  rw [← hs]
  simp only [Exec.setupOf, Readable.Read.setup, Readable.Read.tags, Readable.Read.coins, ZkExec.zkInputs, hzk, hpi,
    hdraw, ite_true]

/-- **`Deploy.V`'s setup is the shipped verifier's**: `verifyCmd` sets a record up at the record's own draw
(`Exec.setupOf`, at `zkInputs` under `--zk`), and where the checks pass and that draw is the statement's as a value
(`hdraw`), that is the session and tables `Deploy.V` set up at the statement's. The checks give the draw only up to key
order (`checks_draw`), and the setup keeps the draw itself: `Stmt.setupTables` makes it the session's `unitDraw`, and
`HmRow.drawn` writes it into the public header (`setObjVal!`), whose `canon` the public digest hashes. -/
theorem setupOf_record {zk : Bool} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (hs : s.setup = .ok (spec, sts)) (hc : checks zk s spec x link c₁ mids f = .ok ())
    (hdraw : Exec.recordDraw f.record = s.read.draw) :
    Exec.setupOf (if s.read.zk then ZkExec.zkInputs s.read.I else s.read.I) f.record = .ok (spec, sts) := by
  have hr := Ins.setup_read hs
  have hpi := publicInputs_none_of_checks hc
  cases hz : s.read.zk
  · rw [← hr]
    simp only [Exec.setupOf, Readable.Read.setup, Readable.Read.tags, Readable.Read.coins, hz, hpi, hdraw,
      Bool.false_eq_true, ite_false]
  · simp only [↓reduceIte]
    exact setupOf_zkInputs hr hz hpi hdraw

/-- **`verify --zk` refuses a session without proof files**: it expects one per rep stream, two per table, and at
least one table. -/
theorem verify_nil (spec : SessionSpec) (tabs : Array Setup) (record : String) :
    Zk.verify spec tabs record #[] ≠ .ok () := by
  intro h
  unfold Zk.verify at h
  simp only [] at h
  by_cases c1 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c1] at h
    obtain ⟨_, h', _⟩ := ebind_ok h
    cases h'
  rw [ite_eq_right c1] at h
  by_cases c2 : spec.coinTree.isNone = true
  · rw [ite_eq_left c2] at h
    obtain ⟨_, h', _⟩ := ebind_ok h
    cases h'
  rw [ite_eq_right c2] at h
  obtain ⟨sess, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨schs, -, h⟩ := ebind_ok h
  have c3 : ((#[] : Array ByteArray).size != spec.repStreams.size) = true := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, Decidable.not_not] at c1
    simp only [List.size_toArray, List.length_nil, repStreams_size, bne_iff_ne, ne_eq]
    omega
  rw [ite_eq_left c3] at h
  obtain ⟨_, h', _⟩ := ebind_ok h
  cases h'

/-! ## The live outcome a decided run records -/

/-- A round with its retained bytes' digest (`spec.dg`, the record's `msg_<dk>`). -/
def withSha (spec : SessionSpec) (rd : Round) : Round := { rd with sha := spec.dg (rd.bytes.getD ByteArray.empty) }

/-- The record's live session (`liveOf`), empty where the record does not read. -/
def liveD (spec : SessionSpec) (record : String) : Live :=
  match liveOf spec record with
  | .ok lv => lv
  | .error _ => ⟨#[], #[], #[], #[]⟩

/-- **The live outcome of a decision of `Deploy.V`**, at `J` tables and `mPts` link coordinates: the link message's
`root_B`, the first coin's two points, each rep stream's rounds as the last message's record lists them, with their
bytes' digests (table `j`'s rep `r` is stream `2j + r`), and the last message's proof files. -/
def viewOf (J mPts : ℕ) (spec : SessionSpec) (link : List ByteArray × List ByteArray) (c₁ : Coin) (f : Final) :
    Tables.LiveOutJ J mPts where
  rootB := link.1.toArray
  pts := ptsOf mPts (pointsOf spec c₁)
  reps j r := ((liveD spec f.record).streams[2 * j.val + r.val]!).toList.map (withSha spec)
  proofs := f.proofs

/-- **The outcome's rounds are the round messages with their coins**, where the custody checks pass: its streams, one per
rep stream, flatten in `spec.repStreams` order to rounds whose retained bytes are the round messages and whose coins
are the first `n` of each message's coin. -/
theorem viewOf_rounds {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (hc : checks zk s spec x link c₁ mids f = .ok ()) :
    (liveD spec f.record).streams.size = spec.repStreams.size ∧
      List.Forall₂ (fun rd am => rd.bytes = some ⟨am.1.toArray⟩ ∧ rd.coins = am.2.vals rd.n)
        ((liveD spec f.record).streams.toList.flatMap (·.toList)) mids := by
  obtain ⟨-, lv, hlv, hcus⟩ := checks_ok hc
  obtain ⟨_, _, _, -, -, -, -, -, hstr⟩ := liveOf_ok hlv
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  rw [hD]
  exact ⟨by simpa using hstr.length_eq.symm, (custody_ok hcus).2.2.2⟩

/-! ## The record is the live session's -/

/-- **`RecordsLiveZKJ` at one outcome**: where `verify --zk` accepts the transcript `t`, its record carries the draw
`d`, its proof files are the outcome `o`'s, and the session it decodes to has `o`'s link points and, at each of `o`'s
tables, `o`'s `root_B` and rep streams' rounds. -/
def RecordsLiveAt (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts) (t : Exec.Transcript) :
    Prop :=
  ZkExec.AcceptsZK I t →
    Exec.recordDraw t.record = some d ∧ t.proofs = o.proofs ∧
    ∀ spec sts sess, Exec.setupOf (ZkExec.zkInputs I) t.record = .ok (spec, sts) →
      Record.decode spec t.record = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin J, sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = o.reps j r

theorem recordsLiveZKJ_iff {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Tables.LiveOutJ (Jf S R) mPts → Exec.Transcript) :
    Composed.RecordsLiveZKJ I dj Jf wr ↔ ∀ S R o, RecordsLiveAt I (dj S) o (wr S R o) := Iff.rfl

/-- **Custody gives `RecordsLiveAt`**: where `Deploy.V --zk`'s checks pass on a statement that sets up at its draw `d`
(so with no `--public-inputs`, `publicInputs_none_of_checks`), and the record's draw is the statement's (`hdraw`), the
last message's record and proof files are a record of the run's live outcome (`viewOf`), at any `J` of the session's
tables. -/
theorem recordsLiveAt_of_checks {zk : Bool} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {J mPts : ℕ} {d : Json} (hs : s.setup = .ok (spec, sts)) (hzk : s.read.zk = true)
    (hdraw : Exec.recordDraw f.record = s.read.draw) (hd : s.read.draw = some d)
    (hJ : J ≤ spec.tables.size) (hc : checks zk s spec x link c₁ mids f = .ok ()) :
    RecordsLiveAt s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  intro _
  refine ⟨hdraw.trans hd, rfl, fun spec' sts' sess hset hdec => ?_⟩
  rw [setupOf_zkInputs (Ins.setup_read hs) hzk (publicInputs_none_of_checks hc) hdraw] at hset
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hset)
  obtain ⟨-, lv, hlv, hcus⟩ := checks_ok hc
  obtain ⟨hroot, -, hpts, hrounds⟩ := custody_ok hcus
  obtain ⟨eroot, epts, esize, estr⟩ := decode_live hdec hlv
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  refine ⟨by simp only [viewOf]; rw [epts, hpts], fun j => ⟨?_, fun r => ?_⟩⟩
  · show sess.rootB[j.val]! = link.1.toArray[j.val]!
    rw [eroot, ← hroot]
  · obtain ⟨hq, hw⟩ := repStreams_getElem spec (i := j.val) (r := r.val) (by omega) r.isLt
    obtain ⟨y, hy, hyr⟩ := estr _ hq
    rw [hw] at hy
    refine ⟨y, hy, ?_⟩
    show y.rounds.toList = ((liveD spec f.record).streams[2 * j.val + r.val]!).toList.map (withSha spec)
    rw [hD]
    have hq' : 2 * j.val + r.val < lv.streams.size := esize ▸ hq
    refine forall₂_map_eq hyr fun rd rd' hmem ⟨hn, hcn, hb, hsha⟩ => ?_
    obtain ⟨am, -, hbytes, -⟩ := forall₂_mem_left hrounds rd' (List.mem_flatMap.2
      ⟨lv.streams[2 * j.val + r.val]!, by rw [getElem!_pos lv.streams _ hq']; exact Array.getElem_mem_toList hq',
        hmem⟩)
    rcases rd with ⟨n, sha, coins, bytes⟩
    rcases rd' with ⟨n', sha', coins', bytes'⟩
    simp only at hn hcn hb hsha hbytes
    subst hn hcn hb hbytes
    simp [withSha, hsha _ rfl]

/-! ## The writer that reads the record out of the last message -/

open Classical in
/-- **The record writer of `Deploy.V`'s runs**: at the live outcome `o` of draw `S`, a transcript `verify --zk`
accepts that records `o` (`RecordsLiveAt`), where one exists, else a session without proof files, which `verify --zk`
refuses (`verify_nil`). A decided run whose last message's record and proof files `verify --zk` accepts is such a
transcript for its outcome (`acceptsZK_wrD`). -/
noncomputable def wrD {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) (S : Finset (Fin n)) (R : Reg) (o : Tables.LiveOutJ (Jf S R) mPts) :
    Exec.Transcript :=
  if h : ∃ t, ZkExec.AcceptsZK I t ∧ RecordsLiveAt I (dj S) o t then h.choose else ⟨"", #[]⟩

theorem not_acceptsZK_nil (I : Exec.Inputs) (record : String) : ¬ ZkExec.AcceptsZK I ⟨record, #[]⟩ := by
  rintro ⟨spec, sts, -, -, h⟩
  exact verify_nil _ _ _ h

/-- **`wrD` records the live session** (`RecordsLiveZKJ`, the record half of the old `live-verifier` assumption
`RecordCustodyZKJ`). -/
theorem recordsLiveZKJ_wrD {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) : Composed.RecordsLiveZKJ (mPts := mPts) I dj Jf (wrD I dj Jf) := by
  refine (recordsLiveZKJ_iff I dj Jf _).2 fun S R o => ?_
  unfold wrD
  split
  · rename_i h
    exact (Classical.choose_spec h).2
  · exact fun h => absurd h (not_acceptsZK_nil _ _)

/-- **`RecordCustodyZKJ` at `wrD`** is its draw file half (`DrawFileZK`) alone. -/
theorem recordCustodyZKJ_wrD {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) (kd : ℕ) (hdf : ZkLink.DrawFileZK dj (.subset kd) n) :
    Assumptions.Zk.RecordCustodyZKJ (mPts := mPts) I dj Jf (wrD I dj Jf) kd :=
  ⟨recordsLiveZKJ_wrD I dj Jf, hdf⟩

/-- **An accepting decision of `Deploy.V --zk` at SHA-512 is an accepting `verify --zk` on a record of its live
outcome**: on a statement of draw `d`, the last message's record and proof files are accepted by `verify --zk`
(`AcceptsZK`) at the statement's inputs, and record the run's live outcome `viewOf` (`RecordsLiveAt`): the record's
draw is the statement's as a value (`recordDraw_of_decide`). -/
theorem accepts_of_decide {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : Final} {J mPts : ℕ} {d : Json} (ha : act true x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next true x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size) :
    ZkExec.AcceptsZK s.read.I ⟨f.record, f.proofs⟩ ∧
      RecordsLiveAt s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  have hdraw := recordDraw_of_decide ha hacc
  obtain ⟨hv, hc⟩ := (next_decide_sha512 ha).1 hacc
  simp only [ite_true] at hv
  obtain ⟨hins, hs⟩ := setupOf_some (act_decide ha)
  have hzk : s.read.zk = true := readOf_zk (x := x) (by simp [readOf, hins])
  exact ⟨⟨spec, sts, setupOf_zkInputs (Ins.setup_read hs) hzk (publicInputs_none_of_checks hc) hdraw,
    setup_shapeOk hs hzk, hv⟩, recordsLiveAt_of_checks hs hzk hdraw hd hJ hc⟩

/-- **The old event holds on `Deploy.V --zk`'s accepting runs**: where its decision accepts at SHA-512, on a statement
of draw `dj S`, `verify --zk` accepts what `wrD` writes for the run's live outcome at `Jf S R ≤` the session's
tables, the event the old theorems bound under `RecordsLiveZKJ`. -/
theorem acceptsZK_wrD {n mPts : ℕ} {Reg : Type} (dj : Finset (Fin n) → Json) (Jf : Finset (Fin n) → Reg → ℕ)
    {S : Finset (Fin n)} {R : Reg} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (ha : act true x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next true x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some (dj S))
    (hJ : Jf S R ≤ spec.tables.size) :
    ZkExec.AcceptsZK s.read.I (wrD s.read.I dj Jf S R (viewOf (Jf S R) mPts spec link c₁ f)) := by
  have hex : ∃ t, ZkExec.AcceptsZK s.read.I t ∧
      RecordsLiveAt s.read.I (dj S) (viewOf (Jf S R) mPts spec link c₁ f) t :=
    ⟨_, accepts_of_decide ha hacc hd hJ⟩
  unfold wrD
  rw [dite_eq_left_of_eq_true (eq_true hex)]
  exact (Classical.choose_spec hex).1

/-! ## The deployed decoders at `wrD` -/

/-- **The deployed `--zk` decoders are right at `wrD`** (`DecodesOnZKJ`: at each draw and registration, `LiveZk`'s
`hdec`): `wrD` records the live session (`recordsLiveZKJ_wrD`), so the old discharge (`runsZCJ_live`) needs only the
inputs' tags (`TagsOkZJ`, untyped) and every draw setting each table up at `mPts` link points (`DrawOkZKJ`). -/
theorem decodesOnZKJ_wrD {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} {d₀ : Json}
    (y₀ : Composed.DrawSetupZJ (Composed.inputsJ I 0) mPts S₀ d₀) (L : Audit.Law n) (hT : Composed.TagsOkZJ I)
    (hDraw : ∀ ω, Composed.DrawOkZKJ I dj mPts (L.draw ω)) :
    Composed.DecodesOnZKJ execArith ZkExec.hs512 (Composed.planZJ (Reg := Reg) dj y₀) I
      (wrD I dj fun S R => (Composed.planZJ (Reg := Reg) dj y₀ S R).length)
      (Composed.opnAtZJ (Composed.planZJ dj y₀) I dj) (Composed.decAtZJ (Composed.planZJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => Exec.capOfRoot rs[j.val]!) L :=
  Composed.decodesOnZKJ_of_runsOnZCJ execArith ZkExec.hs512 _ I _ _ _ _ _ _
    (Composed.runsZCJ_live I ht dj y₀ _ L hT (recordsLiveZKJ_wrD I dj _) hDraw)

/-- **`decodesOnZKJ_wrD` from the draw file half**: where the circuit parses and the draw file is the coin server's
subset draw of `kd` units (`DrawFileZK`, the half of `RecordCustodyZKJ` that `wrD` leaves, `recordCustodyZKJ_wrD`),
every draw sets each table up at `mPtsOf c.g (kd / nTab I)` link points (`drawOkZKJ_of_file`). -/
theorem decodesOnZKJ_wrD_file {n : ℕ} {Reg : Type} (I : Exec.Inputs) (ht : I.tags.typed = false) {c : _root_.Flock.Circuit}
    (hc : HmRow.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ) (hdiv : kd % Composed.nTab I = 0)
    (dj : Finset (Fin n) → Json) (hdf : ZkLink.DrawFileZK dj (.subset kd) n) {S₀ : Finset (Fin n)} {d₀ : Json}
    (y₀ : Composed.DrawSetupZJ (Composed.inputsJ I 0) (Exec.mPtsOf c.g (kd / Composed.nTab I)) S₀ d₀)
    (L : Audit.Law n) (hT : Composed.TagsOkZJ I) :
    Composed.DecodesOnZKJ execArith ZkExec.hs512 (Composed.planZJ (Reg := Reg) dj y₀) I
      (wrD I dj fun S R => (Composed.planZJ (Reg := Reg) dj y₀ S R).length)
      (Composed.opnAtZJ (Composed.planZJ dj y₀) I dj) (Composed.decAtZJ (Composed.planZJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => Exec.capOfRoot rs[j.val]!) L :=
  decodesOnZKJ_wrD I ht dj y₀ L hT fun _ => Composed.drawOkZKJ_of_file I ht hc kd n hdiv dj hdf _

end Claims.Bridge.Custody
