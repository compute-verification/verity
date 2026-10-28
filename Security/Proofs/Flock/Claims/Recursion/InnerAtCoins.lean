import Proofs.Flock.Claims.Recursion.VerdictRowDTie
import Proofs.Flock.Claims.Recursion.RestLens

/-!
# Check 5's inner session at a staged part is the inner statement at the coins

At a V\* statement `xO` part `s` stages at a dense compiled statement `x` and shows the coins `c` at, wherever check 5
loads the inner session from its source (`Source.atC3`, `Source.load`), `InnerAt` says what it loads, as row D's
readers take it off `x` and the coins alone:

* its `m` and `k_log` are the inner statement's table 0's (`Files.stmt` at the files `x` carries, which the deployed
  plain verifier sets up at the same numbers, `innerSetup_files`);
* each rep `r`'s rounds, where its stream has any, are `canonRounds` of the coins after the link, up to rounds of no
  coins (`CoinSource.ofRounds` drops them, so `verifierRows` reads only the rest: `verifierRows_filter`).

`staged_innerAt` proves it from the verifier's checks. The rounds are the coin server's record's (`coinTie_sized`, the
coin tie with `repCoins`' round counts); `checkReads` holds that record to the schedule of the verifier's inner shape,
which at the dense form is the table's two reps at `repRounds` (`scheduleOf_dense`, `two_reps`), so the record's
round of rep `r`'s stream at round `j` is its round `1 + r R + j` after the link, drawn by that coin (`showsTr`); the
record's rounds are the transcript's, but for the one round of no coins `repCoins` lets it skip
(`filter_of_places`).

`InnerCoinsR` (check 5's inner session at the coins, which `posTieR_of` reads) is `InnerAt` with `ExtrasR`, the extra
claims at the first coin's link points (`innerCoinsR_of`), and `rep_canon` gives a rep's verifier rows at
`canonRounds`, the coin side of `VVTie`.
-/

namespace RecDeploy.InnerAt

open Flock Lean
open Claims (Msg Out)
open FlockDeploy (Coin setupOf tabsOf pointsOf)
open RecDeploy.CoinTie
open RecDeploy.Rows (canonRounds)

/-! ## Lists -/

theorem Pairs.countP_eq {α β : Type} {R : α → β → Prop} {p : α → Bool} {q : β → Bool}
    (h : ∀ a b, R a b → p a = q b) : ∀ {l l'}, Pairs R l l' → l.countP p = l'.countP q
  | _, _, .nil => rfl
  | _, _, .cons hab hl => by simp only [List.countP_cons, h _ _ hab, Pairs.countP_eq h hl]

/-- **Two reps' rounds, one after the other**: a list whose rounds are the schedule `rs` at stream `b0`, then at
`b1`, each its stream's round `k` with the schedule's round `k`. -/
theorem two_reps {β δ : Type} [Inhabited δ] {L : List β} {F : β → List UInt8 × ℕ × ℕ × (ℕ × δ)}
    {b0 b1 : List UInt8} {rs : Array (ℕ × δ)}
    (h : L.map F = rs.toList.mapIdx (fun k w => (b0, k, w.1, w)) ++ rs.toList.mapIdx (fun k w => (b1, k, w.1, w))) :
    L.length = 2 * rs.size ∧ ∀ g (hg : g < L.length),
      F L[g] = (if g < rs.size then b0 else b1, g % rs.size, (rs.getD (g % rs.size) default).1,
        rs.getD (g % rs.size) default) := by
  have hlen : L.length = 2 * rs.size := by
    have := congrArg List.length h
    simp at this; omega
  refine ⟨hlen, fun g hg => ?_⟩
  have e := congrArg (fun l => l[g]?) h
  simp only [List.getElem?_map, List.getElem?_eq_getElem hg, Option.map_some] at e
  by_cases hlt : g < rs.size
  · rw [List.getElem?_append_left (by simpa using hlt), List.getElem?_mapIdx, Array.getElem?_toList,
      Array.getElem?_eq_getElem hlt] at e
    simp only [Option.map_some, Option.some.injEq] at e
    rw [e, Nat.mod_eq_of_lt hlt, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
    simp [hlt]
  · have hlt' : g - rs.size < rs.size := by omega
    rw [List.getElem?_append_right (by simp; omega), List.getElem?_mapIdx, List.length_mapIdx,
      Array.length_toList, Array.getElem?_toList, Array.getElem?_eq_getElem hlt'] at e
    simp only [Option.map_some, Option.some.injEq] at e
    have hm : g % rs.size = g - rs.size := by
      rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt hlt']
    rw [e, hm, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt', Option.getD_some]
    simp [hlt]

/-- **Rep `r`'s stream's count**: in a list of two reps of `n` rounds each, the rounds of rep `r`'s stream are `n`. -/
theorem countP_two {β : Type} {L : List β} {p : β → Bool} {n r : ℕ} (hl : L.length = 2 * n) (hr : r < 2)
    (hp : ∀ g (hg : g < L.length), (p L[g] = true ↔ (g < n ↔ r = 0))) : L.countP p = n := by
  rw [← List.take_append_drop n L, List.countP_append]
  have ht : ∀ a ∈ L.take n, (p a = true ↔ r = 0) := by
    intro a ha
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
    simp only [List.length_take] at hi
    rw [List.getElem_take, hp i (by omega)]
    exact ⟨fun h => h.1 (by omega), fun h => ⟨fun _ => h, fun _ => by omega⟩⟩
  have hd : ∀ a ∈ L.drop n, (p a = true ↔ r ≠ 0) := by
    intro a ha
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
    simp only [List.length_drop] at hi
    rw [List.getElem_drop, hp (n + i) (by omega)]
    constructor
    · intro h h0; exact absurd (h.2 h0) (by omega)
    · intro h; exact ⟨fun h' => absurd h' (by omega), fun h0 => absurd h0 h⟩
  have htl : (L.take n).length = n := by simp; omega
  have hdl : (L.drop n).length = n := by simp; omega
  obtain rfl | rfl : r = 0 ∨ r = 1 := by omega
  · rw [List.countP_eq_length.2 (fun a ha => (ht a ha).2 rfl),
      List.countP_eq_zero.2 (fun a ha h => (hd a ha).1 h rfl), htl]
    omega
  · rw [List.countP_eq_zero.2 (fun a ha h => absurd ((ht a ha).1 h) (by decide)),
      List.countP_eq_length.2 (fun a ha => (hd a ha).2 (by decide)), hdl]
    omega

theorem eq_eraseIdx {α : Type} {a b : List α} {j : ℕ} (hj : j < b.length) (hs : a.length + 1 = b.length)
    (h : ∀ m (hm : m < a.length) (hb : (if m < j then m else m + 1) < b.length),
      a[m] = b[if m < j then m else m + 1]) : a = b.eraseIdx j := by
  apply List.ext_getElem (by rw [List.length_eraseIdx_of_lt hj]; omega)
  intro m h1 h2
  rw [List.getElem_eraseIdx, h m h1 (by split <;> omega)]
  by_cases hm : m < j <;> simp [hm]

theorem filter_eraseIdx {α : Type} (p : α → Bool) {l : List α} {j : ℕ} (hj : j < l.length)
    (hp : p l[j] = false) : (l.eraseIdx j).filter p = l.filter p := by
  have e : l.filter p = (l.take j ++ l[j] :: l.drop (j + 1)).filter p := by
    rw [← List.drop_eq_getElem_cons hj, List.take_append_drop]
  rw [List.eraseIdx_eq_take_drop_succ, e, List.filter_append, List.filter_append, List.filter_cons, hp]
  rfl

/-- **Rounds placed at a schedule's, the one round they skip failing `p`, filter to its rounds**: `a`'s round `m` is
`b`'s round `sp m`, where `sp` is the identity at `a` of `b`'s size or skips `b`'s round `j`. -/
theorem filter_of_places {α : Type} {p : α → Bool} {a b : Array α} {sp : ℕ → ℕ}
    (hcase : (sp = id ∧ a.size = b.size) ∨
      ∃ j, sp = (fun m => if m < j then m else m + 1) ∧ a.size + 1 = b.size ∧ ∃ hj : j < b.size, p b[j] = false)
    (hget : ∀ m (hm : m < a.size), ∃ h : sp m < b.size, a[m] = b[sp m]) :
    a.filter p = b.filter p := by
  rcases hcase with ⟨rfl, hs⟩ | ⟨j, rfl, hs, hj, hpj⟩
  · congr 1
    apply Array.ext hs
    intro m h1 h2
    obtain ⟨hh, e⟩ := hget m h1
    simpa using e
  · have hl : a.toList = b.toList.eraseIdx j := by
      apply eq_eraseIdx (by simpa using hj) (by simpa using hs)
      intro m hm hb
      obtain ⟨hh, e⟩ := hget m (by simpa using hm)
      simpa using e
    apply Array.toList_inj.1
    rw [Array.toList_filter, Array.toList_filter, hl, filter_eraseIdx p (by simpa using hj) (by simpa using hpj)]

theorem idx_of {g n k : ℕ} (hk : k < 2) (hg : g < 2 * n) (hc : g < n ↔ k = 0) :
    g = k * n + g % n ∧ g % n < n := by
  obtain rfl | rfl : k = 0 ∨ k = 1 := by omega
  · have h := hc.2 rfl
    rw [Nat.mod_eq_of_lt h]; omega
  · have h : ¬ g < n := fun h => absurd (hc.1 h) (by decide)
    rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]; omega

theorem getD_mid {α : Type} [Inhabited α] (c : ℕ → α) {B i : ℕ} (hi : i + 3 ≤ B) :
    ((List.range B).map c).tail.dropLast.getD i default = c (i + 1) := by
  have h1 : i < ((List.range B).map c).tail.length - 1 := by
    simp only [List.length_tail, List.length_map, List.length_range]; omega
  rw [List.getD_eq_getElem?_getD, List.dropLast_eq_take, List.getElem?_take_of_lt h1,
    List.getElem?_tail, List.getElem?_map, List.getElem?_range (by omega)]
  rfl

/-! ## Rep streams' names -/

/-- **A rep stream's name fixes its rep**: `t/rep k` and `T/rep x` (`k, x < 2`) are one name only at `k = x`, their
last bytes being the digits. -/
theorem rep_digit {t T : String} {k x : ℕ} (hk : k < 2) (hx : x < 2)
    (h : (s!"{t}/rep{k}").toUTF8.toList = (s!"{T}/rep{x}").toUTF8.toList) : k = x := by
  have e := congrArg List.getLast? h
  rw [byteArray_toList_eq, byteArray_toList_eq] at e
  have h0 : (toString (0 : ℕ)).toByteArray.data.toList.getLast? = some 48 := by decide
  have h1 : (toString (1 : ℕ)).toByteArray.data.toList.getLast? = some 49 := by decide
  obtain rfl | rfl : k = 0 ∨ k = 1 := by omega
  all_goals obtain rfl | rfl : x = 0 ∨ x = 1 := by omega
  all_goals first | rfl | (simp only [String.toUTF8, String.toByteArray_append, ByteArray.data_append,
    Array.toList_append, List.getLast?_append, h0, h1, Option.some_or] at e; cases e)

/-! ## The coin tie, with `repCoins`' counts -/

-- `repCoins` matches on array literals, which get no equation lemmas, so the proof unfolds its matcher.
set_option linter.auxLemma false in
/-- **`repCoins_places`, with its counts**: with no round of no coins the record has the transcript's rounds; with one,
at `j`, it has one fewer, skipping `j`. -/
theorem repCoins_sized {s : String} {rs : Array Json} {srec : Json} {cc : Array (Option (Array String))}
    (h : RecVStar.repCoins s rs srec = .ok cc) :
    rs.toList.map (fun r => (r.getObjValAs? Nat "k").toOption) = (List.range rs.size).map some ∧
      cc.size = rs.size ∧
      ∃ sp : Nat → Nat,
        ((sp = id ∧ ((srec.getObjValAs? (Array Json) "rounds").toOption.getD #[]).size = rs.size) ∨
          ∃ j, sp = (fun m => if m < j then m else m + 1) ∧
            ((srec.getObjValAs? (Array Json) "rounds").toOption.getD #[]).size + 1 = rs.size ∧
            ∃ hj : j < rs.size, (RecVStar.coinsOf rs[j]).isEmpty = true) ∧
        ∀ m (hm : m < ((srec.getObjValAs? (Array Json) "rounds").toOption.getD #[]).size),
          sp m < rs.size ∧ cc[sp m]? = some (some (RecVStar.coinsOf
            ((srec.getObjValAs? (Array Json) "rounds").toOption.getD #[])[m])) := by
  unfold RecVStar.repCoins at h
  generalize hjs : Array.map (fun x : Json × Nat => x.snd) _ = js at h
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
    refine ⟨hb hk, by simp [hn'], id, Or.inl ⟨rfl, by simpa using hn'⟩, fun m hm => ⟨by simpa [hn'] using hm, ?_⟩⟩
    simp [hm]
  · have hjf : ∃ hj : j < rs.size, (RecVStar.coinsOf rs[j]).isEmpty = true := by
      have hmem : j ∈ ({ toList := [j] } : Array Nat) := by simp
      rw [← hjs] at hmem
      obtain ⟨⟨y, i⟩, hy, hij⟩ := Array.mem_map.1 hmem
      simp only at hij
      subst hij
      obtain ⟨hz, hy'⟩ := Array.mem_filter.1 hy
      obtain ⟨hi, hyi⟩ := Array.getElem?_eq_some_iff.1 (Array.mem_zipIdx_iff_getElem?.1 hz)
      exact ⟨hi, by rw [hyi]; simpa using hy'⟩
    delta RecVStar.repCoins.match_3 at h
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
    refine ⟨hb hk, by simp, fun m => if m < j then m else m + 1, Or.inr ⟨j, rfl, by simpa using hn', hjf⟩,
      fun m hm => ?_⟩
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

/-- **`coinTie_core`, with the counts**: the decoded stream `name` has as many rounds as the record `R` has of its
stream, or one fewer, skipping the round `j` of `R`'s stream that has no coins. -/
theorem coinTie_sized (hq : ∀ (a : Array Json) (lt : Json → Json → Bool) x, x ∈ a.qsort lt → x ∈ a)
    {tr : Json} {record : String} {reg : Option Json} {bound : Option String}
    {shape : Option RecVStar.StackedShape} {cs : RecVStar.InnerCommits} {ish : RecVStar.Inner} {tt : String}
    {spec : SessionSpec} {sess : Session} {name : String} {st : StreamRec}
    (hic : RecVStar.innerCommits tr record reg bound shape = .ok cs)
    (hR : RecSetup.checkReads ish tt tr record cs = .ok ())
    (hdec : Record.decode spec record = .ok sess) (hst : sess.streams.find? (·.name == name) = some st) :
    ∃ R, RecRecord.readFull tt.toUTF8.toList = some R ∧
      ∃ sp : Nat → Nat,
        ((sp = id ∧ st.rounds.size = R.rounds.countP (·.stream == name.toUTF8.toList)) ∨
          ∃ j, sp = (fun m => if m < j then m else m + 1) ∧
            st.rounds.size + 1 = R.rounds.countP (·.stream == name.toUTF8.toList) ∧
            ∃ g, ∃ hg : g < R.rounds.length, R.rounds[g].stream = name.toUTF8.toList ∧ R.rounds[g].k = j ∧
              R.rounds[g].coins = []) ∧
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
  obtain ⟨hk, hsz, sp, hsized, hmap⟩ := repCoins_sized hcc
  obtain ⟨rjr, hrjr, hrb⟩ := hxs.2
  rw [hrec, arr_as hrjr] at hmap hsized
  have hgetD : (Except.ok rjr : Except String (Array Json)).toOption.getD #[] = rjr := rfl
  simp only [hgetD] at hmap hsized
  have hBoth := jsonRounds_both hrounds hJR
  have hlen := hrb.length_eq
  simp only [Array.length_toList] at hlen
  -- each transcript round of the stream is a round of `R`
  have hat : ∀ i (hi : i < (transcriptRounds name rounds).size), ∃ g, ∃ hg : g < R.rounds.length,
      R.rounds[g].stream = name.toUTF8.toList ∧ R.rounds[g].k = i ∧
      (RecVStar.coinsOf (transcriptRounds name rounds)[i]).toList.mapM f128OfHex = .ok R.rounds[g].coins := by
    intro i hi
    have hin : (transcriptRounds name rounds)[i] ∈ rounds := by
      have := hq _ _ _ (Array.getElem_mem (h := hi))
      exact (Array.mem_filter.1 this).1
    have hsel : (((transcriptRounds name rounds)[i]).getObjValAs? String "stream").toOption = some name := by
      have := hq _ _ _ (Array.getElem_mem (h := hi))
      exact beq_iff_eq.1 (Array.mem_filter.1 this).2
    have hki : (((transcriptRounds name rounds)[i]).getObjValAs? Nat "k").toOption = some i := by
      have := congrArg (fun l => l[i]?) hk
      simp only [List.getElem?_map, Array.getElem?_toList, List.getElem?_range hi] at this
      rw [Array.getElem?_eq_getElem hi] at this
      simpa using this
    obtain ⟨g, hg, hgx⟩ := List.mem_iff_getElem.1 (Array.mem_toList_iff.2 hin)
    have hgl : g < (R.rounds.map fun x => (some x.stream, some x.k, x.coins)).length := by
      rw [← hBoth.length_eq]; exact hg
    obtain ⟨coins, hcoins, hy⟩ := hBoth.get g hg hgl
    rw [List.getElem_map, hgx, hsel, hki, Option.map_some] at hy
    simp only [Prod.mk.injEq, Option.some.injEq] at hy
    obtain ⟨h1, h2, h3⟩ := hy
    rw [hgx] at hcoins
    refine ⟨g, by simpa using hgl, h1, h2, ?_⟩
    rw [hcoins, h3]
  -- the stream's transcript rounds are `R`'s rounds of it
  have hcount : (transcriptRounds name rounds).size = R.rounds.countP (·.stream == name.toUTF8.toList) := by
    have hp : (transcriptRounds name rounds).size =
        (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some name).size := by
      unfold transcriptRounds
      exact (CoinPos.qsort_perm _ _).size_eq
    rw [hp, ← Array.length_toList, Array.toList_filter, ← List.countP_eq_length_filter]
    rw [Pairs.countP_eq (q := fun y => y.1 == some name.toUTF8.toList) ?_ hBoth, List.countP_map]
    · apply List.countP_congr
      intro y _
      simp
    · intro a b ⟨_, _, hb⟩
      subst hb
      cases (a.getObjValAs? String "stream").toOption with
      | none => rfl
      | some s =>
        simp only [Option.map_some]
        by_cases hs : s = name
        · subst hs; simp
        · have : s.toUTF8.toList ≠ name.toUTF8.toList := fun e => hs (toUTF8_toList_inj e)
          simpa [hs] using this
  refine ⟨R, hRead, sp, ?_, fun m hm => ?_⟩
  · rcases hsized with ⟨hid, hs⟩ | ⟨j, hj, hs, hjn, hemp⟩
    · exact Or.inl ⟨hid, by omega⟩
    · refine Or.inr ⟨j, hj, by omega, ?_⟩
      obtain ⟨g, hg, h1, h2, h3⟩ := hat j hjn
      have he : RecVStar.coinsOf (transcriptRounds name rounds)[j] = #[] := Array.isEmpty_iff.1 hemp
      rw [he] at h3
      have h3' : ([] : List F128) = R.rounds[g].coins := by
        simpa [pure, Except.pure] using h3
      exact ⟨g, hg, h1, h2, h3'.symm⟩
  · have hm' : m < rjr.size := by omega
    obtain ⟨hspm, hcm⟩ := hmap m hm'
    obtain ⟨cj, hcj, hfc⟩ := hrb.get m (by simpa using hm') (by simpa using hm)
    have hhex := coins_hex hcj hfc
    simp only [Array.getElem_toList] at hhex
    have hi : sp m < ((transcriptRounds name rounds).zip cc).size := by simp [hsz, hspm]
    have hpair := hloop _ (Array.getElem_mem_toList hi)
    have hccm : cc[sp m] = some (RecVStar.coinsOf rjr[m]) := by
      have := hcm; rw [Array.getElem?_eq_getElem (by omega)] at this; simpa using this
    simp only [Array.getElem_zip, hccm, Option.any_some, bne_eq_false_iff_eq] at hpair
    obtain ⟨g, hg, h1, h2, h3⟩ := hat (sp m) hspm
    rw [← hpair, hhex] at h3
    exact ⟨g, hg, h1, h2, (Except.ok.inj h3).symm⟩

/-! ## The schedule at the dense form -/

/-- **The dense form's schedule**: with no RecSparse phase and no R, at public circuits, the streams are the table's
two reps at `repRounds`. -/
theorem scheduleOf_dense {i : RecVStar.Inner} {t : String}
    {want : Array (String × String × ℕ × Array RecVStar.SchedRound)} {rs : Array RecVStar.SchedRound}
    (hh : i.holo = false) (hs : i.sparse = none) (hk : i.stacked = none)
    (hrs : RecVStar.repRounds i.m i.kLog i.claims = some rs) (h : RecVStar.scheduleOf i t = some want) :
    want.toList = [(s!"{t}/rep{0}", "", 0, rs), (s!"{t}/rep{1}", "", 0, rs)] := by
  unfold RecVStar.scheduleOf at h
  simp [hh, hs, hk, hrs] at h
  subst h
  rfl

/-- Rep `k`'s rounds as the coins give them: round `j` the first `n_j` values of the coin `1 + k R + j`. -/
def canonAt (c : ℕ → Coin) (rs : Array RecVStar.SchedRound) (k : ℕ) : Array (Array F128) :=
  (Array.range rs.size).map fun j => (c (1 + k * rs.size + j)).vals (rs.getD j default).1

theorem canonAt_size (c : ℕ → Coin) (rs : Array RecVStar.SchedRound) (k : ℕ) : (canonAt c rs k).size = rs.size := by
  simp [canonAt]

theorem canonAt_get (c : ℕ → Coin) (rs : Array RecVStar.SchedRound) (k : ℕ) {j : ℕ}
    (hj : j < (canonAt c rs k).size) :
    (canonAt c rs k)[j] = (c (1 + k * rs.size + j)).vals (rs.getD j default).1 := by
  simp [canonAt]

/-- **`canonRounds` at the one table**: rep `k`'s rounds of the coins after the link, at a table whose rep schedule
is `rs`, among at least `2 + 2R` coins. -/
theorem canon_eq {spec : SessionSpec} {st : Stmt} {rs : Array RecVStar.SchedRound} {c : ℕ → Coin} {B k : ℕ}
    (hsc : repSchedAt spec st = rs.toList) (hB : 2 + 2 * rs.size ≤ B) (hk : k < 2) :
    canonRounds spec #[st] ((List.range B).map c).tail.dropLast k = canonAt c rs k := by
  have h0 : (#[st] : Array Stmt)[0 / 2]! = st := rfl
  have h1 : (#[st] : Array Stmt)[1 / 2]! = st := rfl
  have hl : FlockDeploy.repLens spec #[st] = [rs.size, rs.size] := by
    show (List.range 2).map (fun k => (repSchedAt spec (#[st] : Array Stmt)[k / 2]!).length) = _
    rw [show List.range 2 = [0, 1] from rfl, List.map_cons, List.map_cons, List.map_nil]
    simp only [h0, hsc, Array.length_toList]
  have hget : (#[st] : Array Stmt)[k / 2]! = st := by
    obtain rfl | rfl : k = 0 ∨ k = 1 := by omega
    · exact h0
    · exact h1
  have hsz : [rs.size, rs.size].getD k 0 = rs.size := by
    obtain rfl | rfl : k = 0 ∨ k = 1 := by omega
    all_goals simp
  have hsum : ([rs.size, rs.size].take k).sum = k * rs.size := by
    obtain rfl | rfl : k = 0 ∨ k = 1 := by omega
    all_goals simp
  unfold canonRounds canonAt
  rw [hl, hget, hsc, hsz, hsum]
  apply Array.ext (by simp)
  intro j h₁ h₂
  have hj : j < rs.size := by simpa using h₁
  simp only [Array.getElem_map, Array.getElem_range]
  have hi : k * rs.size + j + 3 ≤ B := by
    obtain rfl | rfl : k = 0 ∨ k = 1 := by omega
    all_goals omega
  rw [getD_mid c hi]
  have hr : rs.toList.getD j default = rs.getD j default := by
    simp [List.getD_eq_getElem?_getD, Array.getD_eq_getD_getElem?]
  rw [hr, show k * rs.size + j + 1 = 1 + k * rs.size + j by omega]

/-! ## The rounds at a staged part -/

/-- **Check 5's rounds at a staged part are the coins'**: at a V\* statement part `s` stages at `x`, read at a shape of
public circuits with no RecSparse phase and no R whose rep schedule is `rs`, and which shows the coins `c`, each rep
`k`'s rounds, as check 5 loads them, where its stream has any, are `canonAt c rs k` up to rounds of no coins. -/
theorem staged_rounds {b : Brings} {s : ℕ} {x xO : Msg} {c : ℕ → Coin} {r : XR}
    {rs : Array RecVStar.SchedRound} (hr : readX x = some r) (hh : r.shape.holo = false)
    (hsp : r.shape.sparse = none) (hstk : r.shape.stacked = none) (hrs : repSched r.shape = some rs)
    (hsh : showsPart recRead b s x xO c) :
    ∃ i : RecIns, insO xO = some i ∧
      ∀ f : RecCoins.Files, RecCoins.Source.atC3 i.src i.inner i.commits = some (.files f) →
      ∀ (unit : String) (inn : RecCoins.Inner), RecCoins.Inner.load unit f = .ok inn →
      ∀ k < 2, ∀ rounds : Array (Array F128), inn.rounds k = .ok rounds → 0 < rounds.size →
        rounds.filter (·.size ≠ 0) = (canonAt c rs k).filter (·.size ≠ 0) := by
  obtain ⟨hc, ms, i, _spec, _sts, rfl, hsu, -⟩ := hsh
  obtain ⟨hins, -⟩ := setupO_some hsu
  obtain ⟨xr, hxr, hinner, -⟩ := insO_staged hins
  rw [hr] at hxr
  cases hxr
  obtain ⟨wr, hdw, htr, -, -⟩ := insO_staged_link hins
  obtain ⟨wr', hd', hw⟩ := ins_of_insO hins
  rw [hdw] at hd'
  cases hd'
  have hcm := RecRaw.ins_commits hw
  rw [hinner] at hcm
  obtain ⟨cs, tt, tr, record, hcs, htx, -, hsrc, hR, reg, d, shp, hic⟩ := commitsOf_tie htr hcm
  obtain ⟨lk, R, want, -, hRead, -, -, -, hwant, hflat, -, -⟩ := FlockC3.VStarList.checkReads_ok hR
  have hb : (b.st.transcript x ms).data.toList = tt.toUTF8.toList := by
    rw [byteArray_toList_eq, toUTF8_of_fromUTF8? htx]
  have hshow : shownOf recRead x (staged b.st x ms s) = some (R.msgs.map fun mc => (mc.1, mc.2.toArray)) := by
    simp only [shownOf, hdw, htr, Option.bind_some, recRead, hb, hRead, Option.map_some]
  obtain ⟨ps, hps, hdrawn⟩ := hc
  rw [hshow] at hps
  cases hps
  -- each round of `R` drew its coins from its coin
  have hdr : ∀ g (hg : g < R.rounds.length), Drawn (c (g + 1)) R.rounds[g].coins.toArray := by
    intro g hg
    have hlen : g + 1 < (R.msgs.map fun mc => (mc.1, mc.2.toArray)).length := by
      simp [Flock.RecRecord.Record.msgs, hg]
    have hd := hdrawn (g + 1) hlen
    have hget : ((R.msgs.map fun mc => (mc.1, mc.2.toArray)).getD (g + 1) ([], #[])).2 =
        R.rounds[g].coins.toArray := by
      rw [List.getD_eq_getElem _ _ hlen]
      simp [Flock.RecRecord.Record.msgs]
    rwa [hget] at hd
  -- `R`'s rounds are the two reps' at `rs`
  have hrs' : RecVStar.repRounds r.shape.m r.shape.kLog r.shape.claims = some rs := by
    unfold repSched at hrs
    rw [hh] at hrs
    exact hrs
  rw [scheduleOf_dense hh hsp hstk hrs' hwant] at hflat
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil] at hflat
  obtain ⟨hlen2, hF⟩ := two_reps hflat
  have hG : ∀ g (hg : g < R.rounds.length), ∃ xg < 2, (g < rs.size ↔ xg = 0) ∧
      R.rounds[g].stream = (s!"{lk.table}/rep{xg}").toUTF8.toList ∧ R.rounds[g].k = g % rs.size ∧
      R.rounds[g].coins.length = (rs.getD (g % rs.size) default).1 := by
    intro g hg
    have e := hF g hg
    simp only [Prod.mk.injEq] at e
    obtain ⟨e1, e2, -, e4⟩ := e
    have e4' : R.rounds[g].coins.length = (rs.getD (g % rs.size) default).1 := by
      rw [← e4]; rfl
    by_cases hlt : g < rs.size
    · exact ⟨0, by omega, iff_of_true hlt rfl, by rw [e1, if_pos hlt], e2, e4'⟩
    · exact ⟨1, by omega, iff_of_false hlt (by decide), by rw [e1, if_neg hlt], e2, e4'⟩
  refine ⟨i, hins, fun f hf unit inn hload k hk rounds hrd hpos => ?_⟩
  obtain ⟨spec', sess, st', hdec, hfind, rfl⟩ := load_rounds hload hrd
  have hrec : f.record = record := by
    rw [atC3_record hf] at hsrc
    exact Option.some.inj hsrc
  rw [hrec] at hdec
  obtain ⟨R', hRead', sp, hsized, hall⟩ := coinTie_sized
    (fun a lt x hx => (FlockSoundness.Audit.DrawExec.mem_qsort a lt x).1 hx) hic hR hdec hfind
  rw [hRead] at hRead'
  cases hRead'
  have hsz : (st'.rounds.map (·.coins)).size = st'.rounds.size := Array.size_map ..
  -- rep `k`'s stream is `lk.table`'s rep `k`
  have hnb : (s!"{spec'.tables[0]!}/rep{k}").toUTF8.toList = (s!"{lk.table}/rep{k}").toUTF8.toList := by
    obtain ⟨g0, hg0, h1, -, -⟩ := hall 0 (by omega)
    obtain ⟨x0, hx0, -, hs0, -, -⟩ := hG g0 hg0
    have e := h1.symm.trans hs0
    rw [← rep_digit hk hx0 e] at e
    exact e
  have hcnt : R.rounds.countP (·.stream == (s!"{spec'.tables[0]!}/rep{k}").toUTF8.toList) = rs.size := by
    refine countP_two hlen2 hk fun g hg => ?_
    obtain ⟨xg, hxg, hxc, hsg, -, -⟩ := hG g hg
    rw [beq_iff_eq, hsg, hnb]
    constructor
    · intro h; rw [← rep_digit hxg hk h]; exact hxc
    · intro h
      have e : xg = k := by
        by_cases hgl : g < rs.size
        · have := hxc.1 hgl; have := h.1 hgl; omega
        · have h1 : xg ≠ 0 := fun h0 => hgl (hxc.2 h0)
          have h2 : k ≠ 0 := fun h0 => hgl (h.2 h0)
          omega
      rw [e]
  apply filter_of_places
  · rcases hsized with ⟨hid, hs⟩ | ⟨j, hj, hs, gj, hgj, hs1, hk1, hc1⟩
    · exact Or.inl ⟨hid, by rw [canonAt_size]; omega⟩
    · obtain ⟨xg, -, -, -, hkg, hcg⟩ := hG gj hgj
      have hjg : j = gj % rs.size := hk1.symm.trans hkg
      have hjn : j < rs.size := by rw [hjg]; exact Nat.mod_lt _ (by omega)
      have h0 : (rs.getD j default).1 = 0 := by
        rw [hjg, ← hcg, hc1]; rfl
      refine Or.inr ⟨j, hj, by rw [canonAt_size]; omega, by rw [canonAt_size]; exact hjn, ?_⟩
      rw [canonAt_get, h0]
      simp [FlockDeploy.Coin.vals]
  · intro m hm
    have hm' : m < st'.rounds.size := by omega
    obtain ⟨g, hg, h1, h2, h3⟩ := hall m hm'
    obtain ⟨xg, hxg, hxc, hsg, hkg, hcg⟩ := hG g hg
    have hxk : k = xg := rep_digit hk hxg (h1.symm.trans hsg)
    subst hxk
    obtain ⟨hgi, hmod⟩ := idx_of hk (by omega) hxc
    rw [hkg] at h2
    have hspm : sp m < (canonAt c rs k).size := by rw [canonAt_size, ← h2]; exact hmod
    refine ⟨hspm, ?_⟩
    have hd := hdr g hg
    unfold Drawn at hd
    rw [h3, Array.toArray_toList] at hd
    have hcs : st'.rounds[m].coins.size = (rs.getD (sp m) default).1 := by
      rw [← h2, ← hcg, h3]; simp
    rw [canonAt_get, Array.getElem_map, hd, hcs, show g + 1 = 1 + k * rs.size + sp m by omega]

/-! ## The source and the statement check 5 loads -/

/-- **A staged statement's source is the files `x` carries**, with the stage's record. -/
theorem src_staged {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} {i : RecIns}
    (h : insO (staged st x ms s) = some i) :
    ∃ r rec, readX x = some r ∧ ∃ pj, r.partition = some pj ∧
      i.src = some (.files ⟨r.innerRaw.circuit, r.innerRaw.publicFile, pj, rec, none,
        some (innerFormOf r.innerRaw)⟩) := by
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at h; cases h
  obtain ⟨r, p, hr, -, hd⟩ := staged_decode hne
  obtain ⟨-, hpa, -⟩ := readX_spec hr
  unfold insO at h
  rw [hd] at h
  simp only [Option.bind_eq_bind, Option.bind_some] at h
  split_ifs at h
  · simp at h
  have h' : (partRaw r st x ms s p).ins.toOption = some i := by simpa using h
  obtain ⟨-, hsrc⟩ := RecRaw.ins_inner_src (toOption_eq_some h')
  unfold sourceOf at hsrc
  rw [formOf_partRaw] at hsrc
  dsimp only [partRaw] at hsrc
  cases hq : r.innerRaw.partition with
  | none => rw [hq] at hsrc; simp at hsrc
  | some q =>
    rw [hq] at hsrc hpa
    simp only [Option.mapM_some] at hpa
    simp only [bind, Except.bind] at hsrc
    cases hj : FlockDeploy.jsonOf "--rec-inner-partition" q with
    | error e => rw [hj] at hsrc; cases hsrc
    | ok pj =>
      rw [hj] at hsrc
      have hpj : r.partition = some pj := by
        have := jsonOf_toOption "partition" "--rec-inner-partition" q
        rw [hj] at this
        cases hj' : FlockDeploy.jsonOf "partition" q with
        | error e => rw [hj'] at this; cases this
        | ok pj' =>
          rw [hj'] at this hpa
          simp only [Except.toOption, Option.some.injEq] at this
          simp only [Functor.map, Except.map, Except.toOption, Option.some.injEq] at hpa
          rw [← hpa, this]
      cases ht : textOf "rec inner session" (st.record x ms) with
      | error e => rw [ht] at hsrc; cases hsrc
      | ok rec =>
        rw [ht] at hsrc
        simp only [pure, Except.pure, Except.ok.injEq] at hsrc
        exact ⟨r, rec, hr, pj, hpj, hsrc.symm⟩

/-- **Check 5's source at candidate 3's streams is the files' statement**: whatever the shape and commits, the files
`atC3` gives set the statement up as `g` does. -/
theorem atC3_files {g : RecCoins.Files} {inner : Option RecVStar.Inner} {commits : Option RecVStar.InnerCommits}
    {f : RecCoins.Source} (h : RecCoins.Source.atC3 (some (.files g)) inner commits = some f) :
    ∃ f', f = .files f' ∧ ∀ u, f'.stmt u = g.stmt u := by
  unfold RecCoins.Source.atC3 at h
  rcases inner with _ | i <;> rcases commits with _ | cs <;> dsimp only at h <;> cases h
  all_goals first | exact ⟨g, rfl, fun _ => rfl⟩ | exact ⟨_, rfl, fun u => by split <;> rfl⟩

/-- **The files' statement doesn't read the unit**: it names it only in its errors. -/
theorem stmt_unit {f : RecCoins.Files} {u u' : String} {p : SessionSpec × Stmt} (h : f.stmt u = .ok p) :
    f.stmt u' = .ok p := by
  unfold RecCoins.Files.stmt at h ⊢
  revert h
  cases Stmt.setupTables RecCoins.innerTags f.circuit f.pub Coins.os 1 (partition := some f.partition) with
  | error e => intro h; cases h
  | ok q =>
    obtain ⟨spec, sts⟩ := q
    intro h
    cases hs : sts[0]? with
    | none => simp [hs, bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    | some st => simpa [hs, bind, Except.bind, pure, Except.pure] using h

/-- **What `Inner.load` sets up**: the files' statement, whose `m` and `k_log` the session has, and whose circuit fold
gives its comb. -/
theorem load_fields {unit : String} {f : RecCoins.Files} {inn : RecCoins.Inner}
    (h : RecCoins.Inner.load unit f = .ok inn) :
    ∃ spec st, f.stmt unit = .ok (spec, st) ∧ inn.m = st.m ∧ inn.kLog = st.c.kLog ∧
      inn.combOf = fun _ info => (Setup.ofCircuit st).fold.folded info.lc.fold.alpha info.lc.fold.beta
        info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts := by
  unfold RecCoins.Inner.load at h
  obtain ⟨⟨spec, st⟩, hs, h⟩ := tie_bind h
  refine ⟨spec, st, hs, ?_⟩
  dsimp only at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | skip
  all_goals obtain ⟨spec', -, h⟩ := tie_bind h
  all_goals split at h
  all_goals first
    | (cases h; done)
    | skip
  all_goals obtain ⟨sess, -, h⟩ := tie_bind h
  all_goals cases h
  all_goals exact ⟨rfl, rfl, rfl⟩

end RecDeploy.InnerAt

namespace RecDeploy.OpenLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin setupOf tabsOf pointsOf)
open Claims.Recursion.VStarB
open RecDeploy.Rows (canonRounds posAt)
open RecDeploy.InnerAt

/-- **Check 5's inner session at the coins, but its extra claims** (`InnerAt`): at a V\* statement `xO` part `s`
stages at `x` and shows the coins `c` at, wherever check 5 loads the inner session from its source, its `m` and `k_log`
are the inner statement's table 0's, and each rep's rounds, where its stream has any, are `canonRounds` of the coins
after the link, up to rounds of no coins (which `verifierRows` doesn't read, `verifierRows_filter`). -/
def InnerAt (untr : TrReader) (b : Brings) (Bk : Msg → ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin) : Prop :=
  ∀ (xO : Msg), showsPart untr b s x xO c → ∀ (i : RecIns), insO xO = some i →
    ∀ (f : RecCoins.Source) (unit : String) (inn : RecCoins.Inner),
      RecCoins.Source.atC3 i.src i.inner i.commits = some f → f.load unit = .ok inn →
      ∃ v, setupOf false (xIOf x) = some v ∧ inn.m = (tabsOf v.2.2)[0]!.m ∧ inn.kLog = (tabsOf v.2.2)[0]!.kLog ∧
        ∀ r < 2, ∀ rounds, inn.rounds r = .ok rounds → 0 < rounds.size →
          rounds.filter (·.size ≠ 0) =
            (canonRounds v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c).tail.dropLast r).filter (·.size ≠ 0)

/-- **Check 5's inner session's extra claims are the table's at the first coin's link points** (`ExtrasR`, the part of
`InnerCoinsR` `InnerAt` leaves): its extra claims are table 0's `extra` at `pointsOf` the first coin. -/
def ExtrasR (untr : TrReader) (b : Brings) (Bk : Msg → ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin) : Prop :=
  ∀ (xO : Msg), showsPart untr b s x xO c → ∀ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt),
    setupO xO = some (i, spec, sts) → ∀ (_ : 0 < sts.size) (f : RecCoins.Source) (inn : RecCoins.Inner),
      RecCoins.Source.atC3 i.src i.inner i.commits = some f → f.load sts[0].c.unit.name = .ok inn →
      ∀ v, setupOf false (xIOf x) = some v →
        inn.extras = (tabsOf v.2.2)[0]!.extra (pointsOf v.2.1 (((List.range (Bk (xIOf x))).map c).headD default))

/-- **`InnerAt` at a dense compiled statement**, from the verifier's checks, wherever the coins `c` run to at least the
deployed plain verifier's inner rounds. -/
theorem staged_innerAt {b : Brings} {Bk : Msg → ℕ} {s : ℕ} {x : Msg} {c : ℕ → Coin} {st0 : Claims.Recursion.Stmt}
    (hst : parse x = some st0) (hd : Claims.Recursion.Form.Dense st0.form) (hB : RestRead.kI x ≤ Bk (xIOf x)) :
    InnerAt recRead b Bk s x c := by
  intro xO hsh i hins f unit inn hf hload
  obtain ⟨r, rs, s', spec, st, hr, hrs, hset, -, hsched, hk⟩ := RestRead.schedBridge ⟨st0, hst, hd⟩
  obtain ⟨r', hr', ha, rfl⟩ := parse_read hst
  rw [hr] at hr'
  cases hr'
  have hh : r.shape.holo = false := hd.1
  have hsp : r.shape.sparse = none := by
    have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hd.2
    cases hs : r.shape.sparse with
    | none => rfl
    | some _ => simp [hs] at h2
  have hstk : r.shape.stacked = none := by
    have ha' := ha
    unfold RecScope.accepts at ha'
    simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha'
    obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, -⟩, -⟩ := ha'
    have hss : RecVStar.statements r.shape = .ok ss := hss
    obtain ⟨-, -, -, -, ss', hc3⟩ := RecScope.statements_ok hss
    rcases RecScope.candidate3_ok hc3 with ⟨-, h⟩ | ⟨sp, -, hs, -⟩
    · exact h
    · rw [hsp] at hs; cases hs
  -- the source is the files `x` carries
  obtain ⟨ms, i', _, _, hxO, hsu, -⟩ := hsh.2
  subst hxO
  obtain ⟨hins', -⟩ := setupO_some hsu
  rw [hins] at hins'
  cases hins'
  obtain ⟨r2, rec, hr2, pj, hpj, hsrc⟩ := src_staged hins
  rw [hr] at hr2
  cases hr2
  have hf' := hf
  rw [hsrc] at hf'
  obtain ⟨f', rfl, hfs⟩ := atC3_files hf'
  have hload' : RecCoins.Inner.load unit f' = .ok inn := hload
  obtain ⟨specL, stL, hstL, hm, hkl, -⟩ := load_fields hload'
  obtain ⟨f0, spec', st', s'', spec'', st'', hfs0, hstm, hset', hmm, hkk, -⟩ := innerSetup_files hr ha
  have hf0 : f0 = ⟨r.innerRaw.circuit, r.innerRaw.publicFile, pj, "", none, some (innerFormOf r.innerRaw)⟩ := by
    rw [filesOf, hpj] at hfs0
    simp only [Option.map_some, Option.some.injEq, RecCoins.Source.files.injEq] at hfs0
    exact hfs0.symm
  subst hf0
  have hstL' : RecCoins.Files.stmt "the inner session"
      ⟨r.innerRaw.circuit, r.innerRaw.publicFile, pj, "", none, some (innerFormOf r.innerRaw)⟩ =
        .ok (specL, stL) :=
    stmt_unit (u := unit) (by rw [← hstL, hfs unit]; rfl)
  rw [hstm] at hstL'
  obtain ⟨-, rfl⟩ := Prod.mk.inj (Except.ok.inj hstL')
  have hxI : xIOf x = r.w.inner := by unfold xIOf; rw [hst]; rfl
  rw [hxI] at hset
  rw [hset] at hset'
  have hst1 : st'' = st := by
    have := congrArg (fun a : FlockDeploy.Ins × SessionSpec × Array Stmt => a.2.2.toList) (Option.some.inj hset')
    simp at this
    first | exact this.symm | exact this
  subst hst1
  have htabs : tabsOf #[st''] = #[Setup.ofCircuit st''] := by simp [tabsOf]
  refine ⟨(s', spec, #[st'']), by rw [hxI]; exact hset, ?_, ?_, ?_⟩
  · rw [htabs, hm, ← hmm]; rfl
  · rw [htabs, hkl, ← hkk]; rfl
  · intro k hk2 rounds hrd hpos
    obtain ⟨i'', hins'', hR⟩ := staged_rounds hr hh hsp hstk hrs hsh
    rw [hins] at hins''
    cases hins''
    rw [hR f' hf unit inn hload' k hk2 rounds hrd hpos]
    dsimp only
    rw [canon_eq (B := Bk (xIOf x)) hsched (by omega) hk2]

/-- **`InnerCoinsR` at a dense compiled statement**: `InnerAt` (`staged_innerAt`) and the extra claims (`ExtrasR`). -/
theorem innerCoinsR_of {b : Brings} {Bk : Msg → ℕ} {s : ℕ} {x : Msg} {c : ℕ → Coin} {st0 : Claims.Recursion.Stmt}
    (hst : parse x = some st0) (hd : Claims.Recursion.Form.Dense st0.form) (hB : RestRead.kI x ≤ Bk (xIOf x))
    (hE : ExtrasR recRead b Bk s x c) : InnerCoinsR recRead b Bk s x c := by
  intro xO hsh i spec sts hsu hsz f inn hf hload
  obtain ⟨v, hv, hm, hk, hrd⟩ := staged_innerAt hst hd hB xO hsh i (setupO_some hsu).1 f _ inn hf hload
  exact ⟨v, hv, hm, hk, hE xO hsh i spec sts hsu hsz f inn hf hload v hv, hrd⟩

/-- **`PosTieR` at row D's position reader**, at a dense compiled statement, from `ExtrasR`. -/
theorem posTieR_staged {b : Brings} {pieces : Msg → List OpenPiece} {Bk : Msg → ℕ} {s : ℕ} {x : Msg}
    {c : ℕ → Coin} {st0 : Claims.Recursion.Stmt} (hst : parse x = some st0)
    (hd : Claims.Recursion.Form.Dense st0.form) (hR : PiecesInRangeR pieces) (hB : RestRead.kI x ≤ Bk (xIOf x))
    (hE : ExtrasR recRead b Bk s x c) : PosTieR recRead b pieces posAt Bk s x c :=
  posTieR_of hst hd hR (innerCoinsR_of hst hd hB hE)

/-- **A rep's verifier rows are `canonRounds`'** (the coin side of `VVTie`): where check 5 loads the inner session at
a staged part and builds rep `r`'s values at a shape, they are `verifierRows` at `canonRounds` of the coins. -/
theorem rep_canon {untr : TrReader} {b : Brings} {Bk : Msg → ℕ} {s : ℕ} {x : Msg} {c : ℕ → Coin}
    (hI : InnerAt untr b Bk s x c) {xO : Msg} (hsh : showsPart untr b s x xO c) {i : RecIns}
    (hins : insO xO = some i) {f : RecCoins.Source} {unit : String} {inn : RecCoins.Inner}
    (hf : RecCoins.Source.atC3 i.src i.inner i.commits = some f) (hload : f.load unit = .ok inn)
    {sh : Shape} {extras : Array Claim} {r : ℕ} {rep : RecCoins.Rep} (hr : r < 2)
    (hrep : inn.rep sh extras r = .ok rep) :
    ∃ v, setupOf false (xIOf x) = some v ∧
      RecAlgebra.verifierRows sh (canonRounds v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c).tail.dropLast r)
        extras RecCoins.ZERO_COMB = .ok (rep.v, rep.info, rep.kappa) := by
  obtain ⟨v, hv, -, -, hrd⟩ := hI xO hsh i hins f unit inn hf hload
  obtain ⟨hrounds, hvr⟩ := rep_ok hrep
  refine ⟨v, hv, ?_⟩
  rw [← verifierRows_filter (hrd r hr _ hrounds (verifierRows_pos hvr))]
  exact hvr

end RecDeploy.OpenLower
