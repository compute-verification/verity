import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Flock.Soundness.ExecDraw

/-!
# The coins a staged part's check 5 computes `v` at are the coins the coin server's transcript shows

Check 5 (`RecCoins.checkResiduals`) computes each rep's `v` at its rounds' coins as the inner session's record holds them:
`Inner.load` decodes that record (`Record.decode`), and `Inner.rounds r` is its stream `<table>/rep<r>`'s coins, round by
round (`load_rounds`). `PartComputes` evaluates at the coins the V\* statement's transcript shows (`showsTr recRead`):
the coin server's record read in its own format (`RecRecord.readFull`). This file ties the two, from the verifier's code
alone, with no premise on JSON.

The tie runs through the transcript's JSON, which the recursive setup reads twice (`RecSetup.commits`):

* `RecVStar.innerCommits` holds the inner session record's coin strings to the transcript JSON's, round for round on
  each rep stream (`repCoins`, which places a holo session's split z_partial round), and `Record.decode` parses those
  same strings of the same record text (`decode_streams`, `coins_hex`), since both read `Json.parse record`;
* `RecSetup.checkReads` holds `readFull`'s rounds to the transcript JSON's, stream, round and hex-parsed coins
  (`jsonRounds`, `checkReads_rounds`).

`Json.parse` is opaque to the kernel, and the proof never looks inside it: each side reads one parse of one text.

`coinTie_core` is the tie in the verifier's code: rep `r`'s round `m` has the coins of `readFull`'s round of that stream
at round `sp m`, where `sp` is the identity or skips the split round. `staged_coinTie` takes it to a staged part, at the
record `shownOf recRead` reads, and `staged_coinTie_drawn` to `showsTr`: each such round's coins are the values its coin
drew (`Drawn`).
-/

namespace RecDeploy.CoinTie

open Flock Lean
open Claims (Msg Out)
open FlockDeploy (Coin)

instance instLawfulBEqF128_coinTie : LawfulBEq F128 where
  eq_of_beq {a b} h := by
    cases a with | mk alo ahi => cases b with | mk blo bhi =>
    have h' : ((alo == blo) && (ahi == bhi)) = true := h
    rw [Bool.and_eq_true, beq_iff_eq, beq_iff_eq] at h'
    rw [h'.1, h'.2]
  rfl {a} := by
    cases a with | mk alo ahi =>
    show ((alo == alo) && (ahi == ahi)) = true
    simp

theorem tie_bind {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases hx : x with
  | error e => rw [hx] at h; cases h
  | ok a => rw [hx] at h; exact ⟨a, rfl, h⟩

theorem checkReads_rounds {ish : RecVStar.Inner} {tt : String} {tr : Json} {record : String}
    {cs : RecVStar.InnerCommits} (h : RecSetup.checkReads ish tt tr record cs = .ok ()) :
    ∃ R, RecRecord.readFull tt.toUTF8.toList = some R ∧
      RecSetup.jsonRounds tr = .ok (R.rounds.map fun x => (some x.stream, some x.k, x.coins)) := by
  unfold RecSetup.checkReads at h
  obtain ⟨lk, -, h⟩ := tie_bind h
  split at h
  · rename_i R hR
    refine ⟨R, hR, ?_⟩
    simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    split at h; · cases h
    split at h; · cases h
    split at h; · cases h
    split at h
    · split at h; · cases h
      split at h; · cases h
      cases hj : RecSetup.jsonRounds tr with
      | error e => rw [hj] at h; cases h
      | ok v =>
        rw [hj] at h
        simp only [Except.mapError] at h
        split at h
        · cases h
        · rename_i hne
          simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hne
          rw [hne]
    · cases h
  · cases h

/-- `R` holds of the two lists' elements, pair by pair. -/
inductive Pairs {α β : Type} (R : α → β → Prop) : List α → List β → Prop
  | nil : Pairs R [] []
  | cons {a b l l'} : R a b → Pairs R l l' → Pairs R (a :: l) (b :: l')

theorem Pairs.imp {α β : Type} {R S : α → β → Prop} (h : ∀ a b, R a b → S a b) :
    ∀ {l l'}, Pairs R l l' → Pairs S l l'
  | _, _, .nil => .nil
  | _, _, .cons hab hl => .cons (h _ _ hab) (Pairs.imp h hl)

theorem lmapM_ok_id {ε α : Type} : ∀ l : List α, l.mapM (m := Except ε) Except.ok = .ok l
  | [] => rfl
  | a :: l => by
    simp only [List.mapM_cons]
    rw [show (Except.ok a : Except ε α) = pure a from rfl, lmapM_ok_id l]
    rfl

theorem amapM_list {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : xs.toList.mapM f = .ok ys.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    simp only [Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    simp

theorem lmapM_forall₂ {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → Pairs (fun a b => f a = .ok b) l l'
  | [], l', h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; exact .nil
  | a :: l, l', h => by
    simp only [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := tie_bind h
    obtain ⟨bs, hbs, h⟩ := tie_bind h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact .cons hb (lmapM_forall₂ hbs)

theorem amapM_forall₂ {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : Pairs (fun a b => f a = .ok b) xs.toList ys.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    simp only [Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    simpa using lmapM_forall₂ hl

theorem mapIdxM_go_forall₂ {α β ε : Type} {f : Nat → α → Except ε β} :
    ∀ {l : List α} {acc : Array β} {l' : List β}, List.mapIdxM.go f l acc = .ok l' →
      ∃ pre, l' = acc.toList ++ pre ∧ Pairs (fun a b => ∃ i, f i a = .ok b) l pre
  | [], acc, l', h => by
    simp only [List.mapIdxM.go, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨[], by simp [h], .nil⟩
  | a :: l, acc, l', h => by
    simp only [List.mapIdxM.go] at h
    obtain ⟨b, hb, h⟩ := tie_bind h
    obtain ⟨pre, rfl, hp⟩ := mapIdxM_go_forall₂ h
    exact ⟨b :: pre, by simp, .cons ⟨_, hb⟩ hp⟩

theorem amapIdxM_forall₂ {α β ε : Type} {f : Nat → α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapIdxM f = .ok ys) : Pairs (fun a b => ∃ i, f i a = .ok b) xs.toList ys.toList := by
  have e := Array.toList_mapIdxM (xs := xs) (f := f)
  rw [h] at e
  simp only [Functor.map, Except.map] at e
  obtain ⟨pre, hpre, hf⟩ := mapIdxM_go_forall₂ (f := f) (l := xs.toList) (acc := #[])
    (by simpa [List.mapIdxM] using e.symm)
  simpa [hpre] using hf

theorem find?_forall₂ {α β : Type} {R : α → β → Prop} {p : α → Bool} {q : β → Bool}
    (hpq : ∀ a b, R a b → (p a = q b)) :
    ∀ {l : List α} {l' : List β}, Pairs R l l' → ∀ {b}, l'.find? q = some b →
      ∃ a, l.find? p = some a ∧ R a b
  | _, _, .nil, _, h => by cases h
  | a :: l, b' :: l', .cons hab hl, b, h => by
    rw [List.find?_cons] at h ⊢
    rw [hpq a b' hab]
    cases hq : q b' with
    | true => rw [hq] at h; cases h; exact ⟨a, rfl, hab⟩
    | false => rw [hq] at h; exact find?_forall₂ hpq hl h

theorem objAs_eq {α : Type} [FromJson α] {j v : Json} {k : String} (h : j.getObjVal? k = .ok v) :
    j.getObjValAs? α k = fromJson? v := by
  simp [Json.getObjValAs?, Json.getObjValD, h, Except.toOption]

theorem arr_as {j : Json} {k : String} {a : Array Json} (h : Record.arr j k = .ok a) :
    j.getObjValAs? (Array Json) k = .ok a := by
  unfold Record.arr Record.fld at h
  cases hv : j.getObjVal? k with
  | error e => simp [hv, bind, Except.bind] at h
  | ok v =>
    simp only [hv, bind, Except.bind] at h
    rw [objAs_eq hv]
    cases v <;> simp [Json.getArr?, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    subst h
    show Array.fromJson? _ = _
    simp only [Array.fromJson?, Array.mapM_eq_mapM_toList]
    show List.toArray <$> List.mapM (m := Except String) Except.ok _ = _
    rw [lmapM_ok_id]
    rfl

theorem str_as {j : Json} {k : String} {s : String} (h : Record.str j k = .ok s) :
    j.getObjValAs? String k = .ok s := by
  unfold Record.str Record.fld at h
  cases hv : j.getObjVal? k with
  | error e => simp [hv, bind, Except.bind] at h
  | ok v =>
    simp only [hv, bind, Except.bind] at h
    rw [objAs_eq hv]
    cases v <;> simp [Json.getStr?, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    subst h
    rfl

theorem nat_as {j : Json} {k : String} {n : Nat} (h : Record.nat j k = .ok n) :
    j.getObjValAs? Nat k = .ok n := by
  unfold Record.nat Record.fld at h
  cases hv : j.getObjVal? k with
  | error e => simp [hv, bind, Except.bind] at h
  | ok v =>
    simp only [hv, bind, Except.bind] at h
    rw [objAs_eq hv]
    show v.getNat? = _
    split at h
    · rename_i a ha; rw [ha]; exact h
    · cases h

theorem strs_hex :
    ∀ {l : List Json} {out : List F128}, l.mapM (fun x => match x.getStr? with
        | .ok s => f128OfHex s
        | .error _ => .error "record: an F128 is not a hex string") = .ok out →
      ∃ ss, l.mapM Json.getStr? = .ok ss ∧ ss.mapM f128OfHex = .ok out
  | [], out, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; exact ⟨[], rfl, rfl⟩
  | a :: l, out, h => by
    simp only [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := tie_bind h
    obtain ⟨bs, hbs, h⟩ := tie_bind h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨ss, h1, h2⟩ := strs_hex hbs
    split at hb
    · rename_i s hs
      refine ⟨s :: ss, ?_, ?_⟩
      · simp [List.mapM_cons, hs, h1, bind, Except.bind, pure, Except.pure]
      · simp [List.mapM_cons, hb, h2, bind, Except.bind, pure, Except.pure]
    · cases hb

theorem coins_hex {r : Json} {cj : Array Json} {out : Array F128} (ha : Record.arr r "coins" = .ok cj)
    (h : Record.f128s cj = .ok out) : (RecVStar.coinsOf r).toList.mapM f128OfHex = .ok out.toList := by
  unfold Record.arr Record.fld at ha
  cases hv : r.getObjVal? "coins" with
  | error e => simp [hv, bind, Except.bind] at ha
  | ok v =>
    simp only [hv, bind, Except.bind] at ha
    cases v <;> simp [Json.getArr?, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at ha
    subst ha
    unfold Record.f128s at h
    have hl := amapM_list h
    obtain ⟨ss, h1, h2⟩ := strs_hex hl
    have hc : RecVStar.coinsOf r = ss.toArray := by
      unfold RecVStar.coinsOf
      rw [objAs_eq hv]
      show (Array.fromJson? (Json.arr _)).toOption.getD #[] = _
      simp only [Array.fromJson?]
      rw [Array.mapM_eq_mapM_toList]
      show (List.toArray <$> List.mapM Json.getStr? _).toOption.getD #[] = _
      rw [h1]
      rfl
    rw [hc]
    simpa using h2

/-- **A stream of the record's JSON, as `Record.decode` decoded it**: its name the JSON's, and each round's coins the
hex of its JSON's `coins`, round for round. -/
def StreamOk (x : Json) (st : StreamRec) : Prop :=
  Record.str x "stream" = .ok st.name ∧ ∃ rj, Record.arr x "rounds" = .ok rj ∧
    Pairs (fun r rd => ∃ cj, Record.arr r "coins" = .ok cj ∧ Record.f128s cj = .ok rd.coins) rj.toList st.rounds.toList

set_option maxHeartbeats 1000000 in
theorem decode_streams {spec : SessionSpec} {text : String} {sess : Session}
    (h : Record.decode spec text = .ok sess) :
    ∃ j sj, Json.parse text = .ok j ∧ Record.arr j "streams" = .ok sj ∧ Pairs StreamOk sj.toList sess.streams.toList := by
  unfold Record.decode at h
  cases hp : Json.parse text with
  | error e => rw [hp] at h; obtain ⟨w, hw, -⟩ := tie_bind h; cases hw
  | ok j =>
  rw [hp] at h
  obtain ⟨w, hw, h⟩ := tie_bind h
  cases hw
  refine ⟨j, ?_⟩
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := tie_bind h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨_, rfl, by assumption, (amapM_forall₂ (by assumption)).imp ?_⟩
    intro x st hx
    try dsimp only at hx
    repeat' (first | (obtain ⟨_, _, hx⟩ := tie_bind hx) | (split at hx))
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at hx
      subst hx
      refine ⟨by assumption, _, by assumption, (amapIdxM_forall₂ (by assumption)).imp ?_⟩
      intro r rd hr
      obtain ⟨_, hr⟩ := hr
      try dsimp only at hr
      repeat' (first | (obtain ⟨_, _, hr⟩ := tie_bind hr) | (split at hr))
      all_goals
        simp only [pure, Except.pure, Except.ok.injEq] at hr
        subst hr
        exact ⟨_, by assumption, by assumption⟩

/-- **Rep `r`'s rounds are its stream's coins in the decoded record**: `Inner.load` decodes the inner session's record
(`Record.decode`), and `Inner.rounds r` is the coins, round by round, of its stream `<table>/rep<r>`. -/
theorem load_rounds {unit : String} {f : RecCoins.Files} {inn : RecCoins.Inner} {r : Nat}
    {rounds : Array (Array F128)} (h : RecCoins.Inner.load unit f = .ok inn) (hr : inn.rounds r = .ok rounds) :
    ∃ spec sess st, Record.decode spec f.record = .ok sess ∧
      sess.streams.find? (·.name == s!"{spec.tables[0]!}/rep{r}") = some st ∧ rounds = st.rounds.map (·.coins) := by
  unfold RecCoins.Inner.load at h
  repeat' (first | (obtain ⟨_, _, h⟩ := tie_bind h) | (split at h))
  all_goals (try (simp only [throw, throwThe, MonadExceptOf.throw] at *; contradiction))
  all_goals
    dsimp only at h
    revert h
    split
    · rename_i sess hd
      intro h
      obtain ⟨_, hx, h⟩ := tie_bind h
      cases hx
      cases h
      unfold RecCoins.Inner.rounds at hr
      dsimp only at hr
      revert hr
      split
      · rename_i st hf
        intro hr
        cases hr
        exact ⟨_, sess, st, hd, hf, rfl⟩
      · intro hr; cases hr
    · intro h
      obtain ⟨_, hx, -⟩ := tie_bind h
      cases hx

-- `repCoins` matches on array literals, which get no equation lemmas, so the proof unfolds its matcher.
set_option linter.auxLemma false in
/-- **`repCoins` places each of the record's rounds at a transcript round**: the transcript's rounds of the stream are
its rounds `0 … n-1`, `repCoins` gives one entry each, and the record's round `m` is the transcript's round `sp m`,
which is `m` (no split) or skips the one round of no coins `j` (a holo session's z_partial split). -/
theorem repCoins_places {s : String} {rs : Array Json} {srec : Json} {cc : Array (Option (Array String))}
    (h : RecVStar.repCoins s rs srec = .ok cc) :
    rs.toList.map (fun r => (r.getObjValAs? Nat "k").toOption) = (List.range rs.size).map some ∧
      cc.size = rs.size ∧
      ∃ sp : Nat → Nat, (sp = id ∨ ∃ j, sp = fun m => if m < j then m else m + 1) ∧
        ∀ m (hm : m < ((srec.getObjValAs? (Array Json) "rounds").toOption.getD #[]).size),
          sp m < rs.size ∧ cc[sp m]? = some (some (RecVStar.coinsOf
            ((srec.getObjValAs? (Array Json) "rounds").toOption.getD #[])[m])) := by
  unfold RecVStar.repCoins at h
  generalize Array.map (fun x : Json × Nat => x.snd) _ = js at h
  have hb : ∀ {α : Type} [BEq α] [LawfulBEq α] {a b : α}, ¬(a != b) = true → a = b := by
    intro α _ _ a b h; simpa using h
  rcases js with ⟨_ | ⟨j, _ | ⟨j2, l⟩⟩⟩
  · delta RecVStar.repCoins.match_3 at h
    simp only [List.size_toArray, List.length_nil, dite_true] at h
    simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    split at h
    · cases h
    rename_i hk
    split at h
    · cases h
    rename_i hn
    cases h
    have hn' := hb hn
    refine ⟨hb hk, by simp [hn'], id, Or.inl rfl, fun m hm => ⟨by simpa [hn'] using hm, ?_⟩⟩
    simp [hm]
  · delta RecVStar.repCoins.match_3 at h
    simp only [List.size_toArray, List.length_cons, List.length_nil] at h
    simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    simp only [Nat.zero_add, Nat.one_ne_zero, ↓reduceDIte, Array.getLit, List.getElem_toArray,
      List.getElem_cons_zero] at h
    split at h
    · cases h
    rename_i hk
    split at h
    · cases h
    rename_i hj
    split at h
    · cases h
    rename_i hn
    cases h
    have hn' : _ + 1 = rs.size := hb hn
    refine ⟨hb hk, by simp, fun m => if m < j then m else m + 1, Or.inr ⟨j, rfl⟩, fun m hm => ?_⟩
    have hlt : (if m < j then m else m + 1) < rs.size := by split <;> omega
    refine ⟨hlt, ?_⟩
    rw [Array.getElem?_map, Array.getElem?_range, ite_eq_left_of_eq_true _ _ (eq_true hlt)]
    simp only [Option.map_some, Option.some.injEq]
    by_cases hmj : m < j
    · simp [hmj, Nat.ne_of_lt hmj, hm]
    · have : m + 1 ≠ j := by omega
      have h2 : ¬ m + 1 < j := by omega
      simp [hmj, this, h2, hm]
  · delta RecVStar.repCoins.match_3 at h
    simp only [List.size_toArray, List.length_cons] at h
    simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    try (split at h <;> cases h)

/-- **A loop over a list beside a range stream sees every element**: where the range, of step 1, is no shorter than the
list, the stream never runs out, so a loop that throws at an element failing `P` and passes otherwise passes only if every
element satisfies `P`. -/
theorem zip_loop {α : Type} {msg : Nat → String} {P : α → Bool} :
    ∀ (L : List α) (st : Std.Legacy.Range) {u : Std.Legacy.Range}, st.step = 1 → L.length ≤ st.stop - st.start →
      forIn (m := Except String) L st (fun x __s =>
        match Std.Stream.next? __s with
        | none => pure (ForInStep.done __s)
        | some (k, s') =>
          if P x = true then do
            throw (msg k)
            pure (ForInStep.yield s')
          else pure (ForInStep.yield s')) = .ok u →
      ∀ p ∈ L, P p = false
  | [], _, _, _, _, _, p, hp => by cases hp
  | a :: L, st, u, h1, hl, h, p, hp => by
    rw [List.forIn_cons] at h
    have hnx : Std.Stream.next? st = some (st.start, { st with start := st.start + st.step }) := by
      show (if st.start < st.stop then _ else _) = _
      rw [ite_eq_left_of_eq_true _ _ (eq_true (by simp at hl; omega))]
    simp only [hnx] at h
    cases hP : P a with
    | true => rw [hP] at h; simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    | false =>
      rw [hP] at h
      simp only [Bool.false_eq_true, ↓reduceIte, pure, Except.pure, bind, Except.bind] at h
      rcases List.mem_cons.1 hp with rfl | hp
      · exact hP
      · exact zip_loop L { st with start := st.start + st.step } h1 (by simp at hl ⊢; omega) h p hp

theorem forIn_units {α ε : Type} (P : α → Prop) {f : α → PUnit.{1} → Except ε (ForInStep PUnit.{1})}
    (hf : ∀ a r, f a PUnit.unit = .ok r → P a ∧ r = .yield PUnit.unit) :
    ∀ (l : List α) {u : PUnit.{1}}, forIn l PUnit.unit f = .ok u → ∀ a ∈ l, P a
  | [], _, _, a, ha => by cases ha
  | b :: l, u, h, a, ha => by
    rw [List.forIn_cons] at h
    obtain ⟨r, hr, h⟩ := tie_bind h
    obtain ⟨hP, rfl⟩ := hf b r hr
    rcases List.mem_cons.1 ha with rfl | ha
    · exact hP
    · exact forIn_units P hf l h a ha

/-- Stream `s`'s rounds in the transcript's JSON, in round order, as `innerCommits` reads them. -/
def transcriptRounds (s : String) (rounds : Array Json) : Array Json :=
  (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some s).qsort fun a b =>
    (a.getObjValAs? Nat "k").toOption.getD 0 < (b.getObjValAs? Nat "k").toOption.getD 0

/-- Stream `s` in the inner session record's JSON, as `innerCommits` finds it. -/
def recStream (s : String) (streams : Array Json) : Json :=
  (streams.find? fun x => (x.getObjValAs? String "stream").toOption == some s).getD Json.null

/-- **What `innerCommits` checks of a rep's coins**: `repCoins` places the record's rounds of stream `s` at the
transcript's, and each transcript round's coins are the record's there. -/
def CoinsAgree (s : String) (rounds streams : Array Json) : Prop :=
  ∃ cc, RecVStar.repCoins s (transcriptRounds s rounds) (recStream s streams) = .ok cc ∧
    ∀ p ∈ ((transcriptRounds s rounds).zip cc).toList, (p.2.any (· != RecVStar.coinsOf p.1)) = false

open RecVStar in
/-- **`innerCommits` ties each stream of the inner session's record to the transcript**: the transcript's rounds and
the record's streams, both JSON, and at every stream the record names (`recNames`), its coins round for round
(`CoinsAgree`). -/
theorem innerCommits_reps {tr : Json} {record : String} {reg : Option Json} {bound : Option String}
    {shape : Option StackedShape} {cs : InnerCommits}
    (h : innerCommits tr record reg bound shape = .ok cs) :
    ∃ rounds rj streams, tr.getObjValAs? (Array Json) "rounds" = .ok rounds ∧ Json.parse record = .ok rj ∧
      rj.getObjValAs? (Array Json) "streams" = .ok streams ∧
      ∀ s ∈ streams.toList.filterMap (fun x => (x.getObjValAs? String "stream").toOption),
        CoinsAgree s rounds streams := by
  unfold innerCommits at h
  extract_lets msgs0 extra stackedS declaredR jpMain at h
  split at h
  · cases h
  simp (config := { zeta := false }) only [jpMain] at h
  obtain ⟨_, -, h⟩ := tie_bind h
  obtain ⟨held, hheld, h⟩ := tie_bind h
  split at h
  rotate_left
  · cases h
  rename_i rounds hrounds
  extract_lets regR jpRec at h
  split at h
  rotate_left
  · cases h
  rename_i rj hrj
  obtain ⟨j, hj, h⟩ := tie_bind h
  cases hj
  simp (config := { zeta := false }) only [jpRec] at h
  split at h
  rotate_left
  · cases h
  rename_i streams hstreams
  obtain ⟨msgs, -, h⟩ := tie_bind h
  extract_lets msgs' names c3 recNames jpT at h
  have key : ∀ t', jpT t' = .ok cs → ∀ s ∈ recNames, CoinsAgree s rounds streams := by
    intro t' hT
    simp (config := { zeta := false }) only [jpT] at hT
    extract_lets repNames reps jpC3 jpShape jpLoop at hT
    split at hT
    · cases hT
    rename_i hnames
    have hn : recNames = repNames := by simpa using hnames
    obtain ⟨u, hloop, -⟩ := tie_bind hT
    rw [hn]
    refine forIn_units _ ?_ repNames hloop
    intro s r hr
    dsimp only at hr
    obtain ⟨cc, hcc, hr⟩ := tie_bind hr
    obtain ⟨v, hv, hr⟩ := tie_bind hr
    cases hr
    refine ⟨⟨cc, hcc, ?_⟩, rfl⟩
    rw [← Array.forIn_toList] at hv
    refine zip_loop _ _ rfl ?_ hv
    simp only [Array.toList_zip, List.length_zip, Array.length_toList]
    show _ ≤ (transcriptRounds s rounds).size - 0
    omega
  refine ⟨rounds, rj, streams, hrounds, ‹_›, hstreams, ?_⟩
  split at h
  rotate_left
  · cases h
  split at h
  · split at h
    · obtain ⟨w, -, h⟩ := tie_bind h
      exact key _ h
    · cases h
  · split at h
    · cases h
    · obtain ⟨_, -, h⟩ := tie_bind h
      exact key _ h

theorem Pairs.length_eq {α β : Type} {R : α → β → Prop} : ∀ {l l'}, Pairs R l l' → l.length = l'.length
  | _, _, .nil => rfl
  | _, _, .cons _ hl => by simp [Pairs.length_eq hl]

theorem Pairs.get {α β : Type} {R : α → β → Prop} :
    ∀ {l l'}, Pairs R l l' → ∀ i (h1 : i < l.length) (h2 : i < l'.length), R l[i] l'[i]
  | _, _, .nil, i, h1, _ => by simp at h1
  | _, _, .cons hab hl, 0, _, _ => hab
  | _, _, .cons _ hl, i + 1, h1, h2 => Pairs.get hl i (by simpa using h1) (by simpa using h2)

/-- **The transcript JSON's rounds as `jsonRounds` reads them**: round for round, its stream, round number and coins. -/
theorem jsonRounds_both {tr : Json} {rounds : Array Json} {L : List (Option (List UInt8) × Option Nat × List F128)}
    (hr : tr.getObjValAs? (Array Json) "rounds" = .ok rounds) (h : RecSetup.jsonRounds tr = .ok L) :
    Pairs (fun x y => ∃ coins, (RecVStar.coinsOf x).toList.mapM f128OfHex = .ok coins ∧
      y = ((x.getObjValAs? String "stream").toOption.map (·.toUTF8.toList), (x.getObjValAs? Nat "k").toOption, coins))
      rounds.toList L := by
  unfold RecSetup.jsonRounds at h
  rw [hr] at h
  refine (lmapM_forall₂ h).imp fun x y hxy => ?_
  obtain ⟨coins, hc, hxy⟩ := tie_bind hxy
  simp only [pure, Except.pure, Except.ok.injEq] at hxy
  exact ⟨coins, hc, hxy.symm⟩

theorem some_beq_some {α : Type} [BEq α] [LawfulBEq α] {a b : α} : ((some a == some b) = true) ↔ a = b := by
  simp

/-- **The coin tie, in the verifier's code**: where `innerCommits` and `checkReads` pass on the coin server's transcript
(`tt`, its JSON `tr`) and the inner session's record (`record`), and `Record.decode` decodes that record, each round
`m` of the record's stream `name` has the coins of a round of the record `checkReads` reads in its own format
(`readFull`): of that stream, at round `sp m`, where `sp` is the identity or skips the one split round of a holo
session's z_partial. `hq` is `qsort`'s membership (`FlockSoundness.Audit.DrawExec.mem_qsort`). -/
theorem coinTie_core (hq : ∀ (a : Array Json) (lt : Json → Json → Bool) x, x ∈ a.qsort lt → x ∈ a)
    {tr : Json} {record : String} {reg : Option Json} {bound : Option String}
    {shape : Option RecVStar.StackedShape} {cs : RecVStar.InnerCommits} {ish : RecVStar.Inner} {tt : String}
    {spec : SessionSpec} {sess : Session} {name : String} {st : StreamRec}
    (hic : RecVStar.innerCommits tr record reg bound shape = .ok cs)
    (hR : RecSetup.checkReads ish tt tr record cs = .ok ())
    (hdec : Record.decode spec record = .ok sess) (hst : sess.streams.find? (·.name == name) = some st) :
    ∃ R, RecRecord.readFull tt.toUTF8.toList = some R ∧
      ∃ sp : Nat → Nat, (sp = id ∨ ∃ j, sp = fun m => if m < j then m else m + 1) ∧
        ∀ m (hm : m < st.rounds.size), ∃ g, ∃ hg : g < R.rounds.length,
          R.rounds[g].stream = name.toUTF8.toList ∧ R.rounds[g].k = sp m ∧
          R.rounds[g].coins = st.rounds[m].coins.toList := by
  obtain ⟨R, hRead, hJR⟩ := checkReads_rounds hR
  obtain ⟨rounds, rj, streams, hrounds, hparse, hstreams, hall⟩ := innerCommits_reps hic
  obtain ⟨j, sj, hparse', harr, hboth⟩ := decode_streams hdec
  rw [hparse] at hparse'
  cases hparse'
  rw [arr_as harr] at hstreams
  have hss := (Except.ok.inj hstreams).symm
  subst hss
  have hst' : sess.streams.toList.find? (·.name == name) = some st := by
    rw [Array.find?_toList]; exact hst
  obtain ⟨x, hx, hxs⟩ := find?_forall₂ (p := fun x => (x.getObjValAs? String "stream").toOption == some name)
    (fun x st hxs => by simp only [str_as hxs.1, Except.toOption]; simp) hboth hst'
  have hname : st.name = name := by simpa using List.find?_some hst'
  have hrec : recStream name streams = x := by
    unfold recStream; rw [← Array.find?_toList, hx]; rfl
  have hmem : name ∈ streams.toList.filterMap (fun x => (x.getObjValAs? String "stream").toOption) :=
    List.mem_filterMap.2 ⟨x, List.mem_of_find?_eq_some hx,
      (by rw [str_as hxs.1, hname]; rfl : (x.getObjValAs? String "stream").toOption = some name)⟩
  obtain ⟨cc, hcc, hloop⟩ := hall name hmem
  obtain ⟨hk, hsz, sp, hsp, hmap⟩ := repCoins_places hcc
  obtain ⟨rjr, hrjr, hrb⟩ := hxs.2
  rw [hrec, arr_as hrjr] at hmap
  have hgetD : (Except.ok rjr : Except String (Array Json)).toOption.getD #[] = rjr := rfl
  simp only [hgetD] at hmap
  have hBoth := jsonRounds_both hrounds hJR
  refine ⟨R, hRead, sp, hsp, fun m hm => ?_⟩
  have hlen := hrb.length_eq
  simp only [Array.length_toList] at hlen
  have hm' : m < rjr.size := by omega
  obtain ⟨hspm, hcm⟩ := hmap m hm'
  obtain ⟨cj, hcj, hfc⟩ := hrb.get m (by simpa using hm') (by simpa using hm)
  have hhex := coins_hex hcj hfc
  simp only [Array.getElem_toList] at hhex
  have hi : sp m < ((transcriptRounds name rounds).zip cc).size := by simp [hsz, hspm]
  have hpair := hloop _ (Array.getElem_mem_toList hi)
  have hccm : cc[sp m] = some (RecVStar.coinsOf rjr[m]) := by
    have := hcm; rw [Array.getElem?_eq_getElem (by omega)] at this; simpa using this
  simp only [Array.getElem_zip, hccm, Option.any_some, bne_eq_false_iff_eq] at hpair
  have hin : (transcriptRounds name rounds)[sp m] ∈ rounds := by
    have := hq _ _ _ (Array.getElem_mem (h := hspm))
    exact (Array.mem_filter.1 this).1
  have hsel : (((transcriptRounds name rounds)[sp m]).getObjValAs? String "stream").toOption = some name := by
    have := hq _ _ _ (Array.getElem_mem (h := hspm))
    exact beq_iff_eq.1 (Array.mem_filter.1 this).2
  have hkm : (((transcriptRounds name rounds)[sp m]).getObjValAs? Nat "k").toOption = some (sp m) := by
    have := congrArg (fun l => l[sp m]?) hk
    simp only [List.getElem?_map, Array.getElem?_toList, List.getElem?_range hspm] at this
    rw [Array.getElem?_eq_getElem hspm] at this
    simpa using this
  obtain ⟨g, hg, hgx⟩ := List.mem_iff_getElem.1 (Array.mem_toList_iff.2 hin)
  have hgl : g < (R.rounds.map fun x => (some x.stream, some x.k, x.coins)).length := by
    rw [← hBoth.length_eq]; exact hg
  obtain ⟨coins, hcoins, hy⟩ := hBoth.get g hg hgl
  rw [List.getElem_map, hgx, hsel, hkm, Option.map_some] at hy
  simp only [Prod.mk.injEq, Option.some.injEq] at hy
  obtain ⟨h1, h2, h3⟩ := hy
  rw [hgx, ← hpair, hhex] at hcoins
  simp only [Except.ok.injEq] at hcoins
  refine ⟨g, by simpa using hgl, h1, h2, ?_⟩
  rw [h3, ← hcoins]


/-! ## At a staged part -/

theorem commits_inner {sh : RecVStar.Inner} {tt : String} {tr : Json} {r : Except String (Option String)}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} {record : String} {cs : RecVStar.InnerCommits}
    (htr : Json.parse tt = .ok tr) (h : RecSetup.commits (some sh) (.ok tt) r bound record = .ok cs) :
    ∃ reg d shp, RecVStar.innerCommits tr record reg d shp = .ok cs := by
  unfold RecSetup.commits at h
  simp only [bind, Except.bind, pure, Except.pure, htr] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨_, _, _, by assumption⟩)

/-- **A transcript's commits at the verifier's inner shape, with the record they read**: `commitsOf_reads`, and
`innerCommits` passed on the transcript's JSON and the source's record. -/
theorem commitsOf_tie {w : RecRaw} {src : Option RecCoins.Source}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} {sh : RecVStar.Inner}
    {c : Option RecVStar.InnerCommits} {t : ByteArray} (ht : w.transcript = some t)
    (h : commitsOf w src bound (some sh) = .ok c) :
    ∃ cs tt tr record, c = some cs ∧ String.fromUTF8? t = some tt ∧ Json.parse tt = .ok tr ∧
      src.map (·.record) = some record ∧ RecSetup.checkReads sh tt tr record cs = .ok () ∧
      ∃ reg d shp, RecVStar.innerCommits tr record reg d shp = .ok cs := by
  unfold commitsOf at h
  rw [ht] at h
  cases hsrc : src.map (·.record) with
  | none => rw [hsrc] at h; cases h
  | some record =>
    rw [hsrc] at h
    dsimp only at h
    split at h
    · cases h
    cases hX : RecSetup.commits (some sh) (textOf "rec inner transcript" t)
        (w.registration.mapM (textOf "rec inner registration")) bound record with
    | error e => rw [hX] at h; cases h
    | ok cs =>
      rw [hX] at h
      cases h
      cases htx : String.fromUTF8? t with
      | none =>
        rw [textOf, htx] at hX
        exact absurd hX commits_text_error
      | some tt =>
        rw [textOf, htx] at hX
        cases hp : Json.parse tt with
        | error e => exact absurd hX (commits_parse_error hp)
        | ok tr =>
          exact ⟨cs, tt, tr, record, rfl, rfl, hp, rfl, FlockC3.VStarList.reads_of_commits hp hX,
            commits_inner hp hX⟩

/-- Check 5's source at candidate 3's streams holds the source's record. -/
theorem atC3_record {src : Option RecCoins.Source} {inner : Option RecVStar.Inner}
    {commits : Option RecVStar.InnerCommits} {f : RecCoins.Files}
    (h : RecCoins.Source.atC3 src inner commits = some (.files f)) : src.map (·.record) = some f.record := by
  unfold RecCoins.Source.atC3 at h
  split at h
  · simp only [Option.some.injEq, RecCoins.Source.files.injEq] at h
    subst h
    split <;> rfl
  · subst h; rfl

/-- **The coin tie at a staged part**: the coin server's record `R` that `recRead` reads off the V\* statement
(`shownOf`), and wherever check 5 loads the inner statement from its source (`Source.atC3`, `Inner.load`), each round
`m` of rep `r`'s coins is the coins of `R`'s round of rep `r`'s stream at round `sp m`, `sp` the identity or skipping
the one split round. -/
theorem staged_coinTie {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) :
    ∃ (i : RecIns) (R : Flock.RecRecord.Record), insO xO = some i ∧
      shownOf recRead x xO = some (R.msgs.map fun mc => (mc.1, mc.2.toArray)) ∧
      ∀ f : RecCoins.Files, RecCoins.Source.atC3 i.src i.inner i.commits = some (.files f) →
      ∀ (unit : String) (inn : RecCoins.Inner), RecCoins.Inner.load unit f = .ok inn →
      ∀ (r : ℕ) (rounds : Array (Array F128)), inn.rounds r = .ok rounds →
        ∃ t : String, ∃ sp : ℕ → ℕ, (sp = id ∨ ∃ j, sp = fun m => if m < j then m else m + 1) ∧
          ∀ m (hm : m < rounds.size), ∃ g, ∃ hg : g < R.rounds.length,
            R.rounds[g].stream = (s!"{t}/rep{r}").toUTF8.toList ∧ R.rounds[g].k = sp m ∧
            R.rounds[g].coins = rounds[m].toList := by
  obtain ⟨ms, i, _spec, _sts, rfl, hsu, -⟩ := hS
  obtain ⟨hins, -⟩ := setupO_some hsu
  obtain ⟨xr, -, hinner, -⟩ := insO_staged hins
  obtain ⟨wr, hd, htr, -, -⟩ := insO_staged_link hins
  obtain ⟨wr', hd', hw⟩ := ins_of_insO hins
  rw [hd] at hd'
  cases hd'
  have hc := RecRaw.ins_commits hw
  rw [hinner] at hc
  obtain ⟨cs, tt, tr, record, hcs, htx, -, hsrc, hR, reg, d, shp, hic⟩ := commitsOf_tie htr hc
  obtain ⟨R, hRead, -⟩ := checkReads_rounds hR
  have hb : (st.transcript x ms).data.toList = tt.toUTF8.toList := by
    rw [byteArray_toList_eq, toUTF8_of_fromUTF8? htx]
  refine ⟨i, R, hins, ?_, fun f hf unit inn hload r rounds hr => ?_⟩
  · simp only [shownOf, hd, htr, Option.bind_some, recRead, hb, hRead, Option.map_some]
  obtain ⟨spec', sess, st', hdec, hfind, rfl⟩ := load_rounds hload hr
  have hrec : f.record = record := by
    rw [atC3_record hf] at hsrc
    exact Option.some.inj hsrc
  rw [hrec] at hdec
  obtain ⟨R', hRead', sp, hsp, hall⟩ := coinTie_core
    (fun a lt x hx => (FlockSoundness.Audit.DrawExec.mem_qsort a lt x).1 hx) hic hR hdec hfind
  rw [hRead] at hRead'
  cases hRead'
  refine ⟨spec'.tables[0]!, sp, hsp, fun m hm => ?_⟩
  obtain ⟨g, hg, h1, h2, h3⟩ := hall m (by simpa using hm)
  exact ⟨g, hg, h1, h2, by simpa using h3⟩

/-- **The coins check 5 computes `v` at are the coins the transcript shows** (`showsTr recRead`): at a staged part
whose statement shows coins `c`, each round `m` of rep `r`'s coins, as check 5 loads them, is the values its coin
drew at the inner round `g + 1` that the record holds it at (`Record.msgs`: the link, then the record's rounds), a
round of rep `r`'s stream at round `sp m`. -/
theorem staged_coinTie_drawn {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {c : ℕ → Coin}
    (hc : showsTr recRead s x xO c) :
    ∃ (i : RecIns) (R : Flock.RecRecord.Record), insO xO = some i ∧
      shownOf recRead x xO = some (R.msgs.map fun mc => (mc.1, mc.2.toArray)) ∧
      ∀ f : RecCoins.Files, RecCoins.Source.atC3 i.src i.inner i.commits = some (.files f) →
      ∀ (unit : String) (inn : RecCoins.Inner), RecCoins.Inner.load unit f = .ok inn →
      ∀ (r : ℕ) (rounds : Array (Array F128)), inn.rounds r = .ok rounds →
        ∃ t : String, ∃ sp : ℕ → ℕ, (sp = id ∨ ∃ j, sp = fun m => if m < j then m else m + 1) ∧
          ∀ m (hm : m < rounds.size), ∃ g, ∃ hg : g < R.rounds.length,
            R.rounds[g].stream = (s!"{t}/rep{r}").toUTF8.toList ∧ R.rounds[g].k = sp m ∧
            Drawn (c (g + 1)) rounds[m] := by
  obtain ⟨i, R, hins, hshow, hall⟩ := staged_coinTie hS
  obtain ⟨ps, hps, hdrawn⟩ := hc
  rw [hshow] at hps
  cases hps
  refine ⟨i, R, hins, hshow, fun f hf unit inn hload r rounds hr => ?_⟩
  obtain ⟨t, sp, hsp, hm⟩ := hall f hf unit inn hload r rounds hr
  refine ⟨t, sp, hsp, fun m hmm => ?_⟩
  obtain ⟨g, hg, h1, h2, h3⟩ := hm m hmm
  refine ⟨g, hg, h1, h2, ?_⟩
  have hlen : g + 1 < (R.msgs.map fun mc => (mc.1, mc.2.toArray)).length := by
    simp [Flock.RecRecord.Record.msgs, hg]
  have hd := hdrawn (g + 1) hlen
  have hget : ((R.msgs.map fun mc => (mc.1, mc.2.toArray)).getD (g + 1) ([], #[])).2 = rounds[m] := by
    rw [List.getD_eq_getElem _ _ hlen]
    simp [Flock.RecRecord.Record.msgs, h3]
  rwa [hget] at hd

end RecDeploy.CoinTie
