import Proofs.Properties.Challenger
import Proofs.Flock.Claims.Complete.Json.Record

/-!
# `RoundsByDevice`, reduced to the binding of the reader's checks to the device

`roundCoins` reads each use's coins with `drawCoins` on the use's record (`useCoins`), from the one session
`trustedSessionOf` reads off the ledger, the pins and the session's files, and refuses unless that session's log shows
exactly the uses' roots (`exactlyReason`), releases each pair of `order` in order (`releaseOrderReason`) and each
group's uses one run each (`runsWhy`). `roundCoins_ok` reads all of that off an accepted read.

What ties those checks to the device is two binding facts about the reader's code, stated as Props:

* `SessionBinding`: the session `trustedSessionOf` accepts on a withheld window is one the device ran under the key,
  which its ledger opens withheld on the window, with that commitment, those roots and that log; or the finder outputs
  a break from every record with the same ledger, pins, main and session.
* `CoinBinding`: what `drawCoins` reads from such a session is what the session released, or the finder outputs a
  break from the record.

`roundsByDevice_of` derives `RoundsByDevice`'s conclusion from them for every read with at least one use and no pair
`(a, a)` in `order`; `drawnByDevice_of` derives `DrawnByDevice` from the same two. Neither side condition can be
dropped. With no uses, the conclusion's break disjunct names no record, so a read under a key the device never used
satisfies neither disjunct. A pair `(a, a)` whose family the log receives once passes `releaseOrderReason` while
`Session.Before s a a` fails (`order_self_gap`).
-/

namespace Properties.Challenger

open Lean (Json)
open SampledProofs SampledProofs.Challenger

theorem none_of_forall {α : Type} {o : Option α} (h : ∀ x, o = some x → False) : o = none := by
  cases o with
  | none => rfl
  | some x => exact (h x rfl).elim

theorem mapM_ok_getElem? {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {ys : List β}, l.mapM f = .ok ys → ∀ {i : Nat} {x : α}, l[i]? = some x →
      ∃ y, ys[i]? = some y ∧ f x = .ok y
  | [], _, _, _, _, hx => by cases hx
  | a :: l, ys, h, i, x, hx => by
    rw [List.mapM_cons] at h
    cases ha : f a with
    | error e => rw [ha] at h; cases h
    | ok y =>
      rw [ha] at h
      cases hl : l.mapM f with
      | error e => rw [hl] at h; cases h
      | ok zs =>
        rw [hl] at h
        cases h
        cases i with
        | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at hx
          subst hx
          exact ⟨y, rfl, ha⟩
        | succ i =>
          obtain ⟨y', hy', hf⟩ := mapM_ok_getElem? hl (i := i) (by simpa using hx)
          exact ⟨y', by simpa using hy', hf⟩

/-- A list of JSON values that a reader of strings reads to `lines` is `lines`, as JSON strings. -/
theorem strs_of_mapM {F : Json → Except String String} (hF : ∀ j s, F j = .ok s → j = Json.str s) :
    ∀ {ls : List Json} {lines : List String}, ls.mapM F = .ok lines → ls = lines.map Json.str
  | [], lines, h => by
    rw [List.mapM_nil] at h
    cases h
    rfl
  | j :: ls, lines, h => by
    rw [List.mapM_cons] at h
    cases hj : F j with
    | error e => rw [hj] at h; cases h
    | ok s =>
      rw [hj] at h
      cases hl : ls.mapM F with
      | error e => rw [hl] at h; cases h
      | ok ss =>
        rw [hl] at h
        cases h
        show j :: ls = Json.str s :: ss.map Json.str
        rw [hF j s hj, strs_of_mapM hF hl]

theorem map_str_inj : ∀ {xs ys : List String}, xs.map Json.str = ys.map Json.str → xs = ys
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | x :: xs, y :: ys, h => by
    simp only [List.map_cons, List.cons.injEq] at h
    obtain ⟨h1, h2⟩ := h
    injection h1 with h1
    rw [h1, map_str_inj h2]

theorem getD_mkObj_of {l : List (String × Json)} (hd : FlockComplete.JsonRT.Distinct l) {k : String} {v : Json}
    (hm : (k, v) ∈ l) : getD (Json.mkObj l) k = v := by
  simp only [getD, field?, FlockComplete.JsonRT.getObjVal?_mkObj hd hm, Except.toOption, Option.getD_some]

/-- A use's record carries the read's ledger, pins, main and session. -/
theorem useCoins_get (coins : Json) (i : Nat) :
    getD (useCoins coins i) "ledger" = getD coins "ledger" ∧ getD (useCoins coins i) "pins" = getD coins "pins" ∧
      getD (useCoins coins i) "main" = getD coins "main" ∧
        getD (useCoins coins i) "session" = getD coins "session" := by
  unfold useCoins
  exact ⟨getD_mkObj_of (by simp) (by simp), getD_mkObj_of (by simp) (by simp), getD_mkObj_of (by simp) (by simp),
    getD_mkObj_of (by simp) (by simp)⟩

/-- Use `i`'s record is the one `roundCoins` hands `drawCoins` for it: the read's fields with its `i`th segment. -/
theorem useCoins_eq {coins : Json} {ss : Array Json} (hs : getD coins "segments" = .arr ss) {i : Nat} {s : Json}
    (hi : ss[i]? = some s) :
    useCoins coins i = Json.mkObj [("ledger", getD coins "ledger"), ("main", getD coins "main"),
      ("pins", getD coins "pins"), ("segment", s), ("session", getD coins "session")] := by
  unfold useCoins
  rw [hs]
  simp only [Array.getD_eq_getD_getElem?, hi, Option.getD_some]

theorem zip_draw {us : List Use} {ls : Array Json} {ξs : List ByteArray} {F : Use × Json → Except String ByteArray}
    {P : Use → Json → ByteArray → Prop} (hsz : (ls.size == us.length) = true) (hm : (us.zip ls.toList).mapM F = .ok ξs)
    (hF : ∀ u s r, F (u, s) = .ok r → P u s r) {i : Nat} {u : Use} (hu : us[i]? = some u) :
    ∃ ξ s, ξs[i]? = some ξ ∧ ls[i]? = some s ∧ P u s ξ := by
  have hi : i < us.length := (List.getElem?_eq_some_iff.1 hu).1
  have hl : i < ls.size := by
    have : ls.size = us.length := beq_iff_eq.1 hsz
    omega
  have hz : (us.zip ls.toList)[i]? = some (u, ls[i]) := by
    rw [List.getElem?_zip_eq_some]
    exact ⟨hu, by simp [hl]⟩
  obtain ⟨y, hy, hf⟩ := mapM_ok_getElem? hm hz
  exact ⟨y, ls[i], hy, by simp [hl], hF u ls[i] y hf⟩

/-- **What an accepted `roundCoins` read checked**: the session `trustedSessionOf` reads, whose roots are exactly the
uses' (`exactlyReason`), on a withheld window; the session's log, as strings, whose order `releaseOrderReason` and
`runsWhy` pass; and each use's coins, what `drawCoins` reads from the use's record (`useCoins`). -/
theorem roundCoins_ok {key : ChallengerKey} {w : String} {us : List Use} {order : List (String × String)}
    {groups : List (List String)} {coins : Json} {ξs : List ByteArray}
    (h : roundCoins key w us order groups coins = .ok ξs) :
    (∃ cr, trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key (getD coins "pins")
        (getD coins "main") (getD coins "session") = .ok cr ∧ exactlyReason cr.1 cr.2 us = none) ∧
      withheldOn (getD coins "ledger") w = true ∧
      (∃ (ls : Array Json) (lines : List String), getD (getD coins "session") "log" = .arr ls ∧
        ls.toList = lines.map Json.str ∧ releaseOrderReason (receivedOrder lines) order = none ∧
        runsWhy us groups (receivedOrder lines) = none) ∧
      ∀ (i : Nat) (u : Use), us[i]? = some u →
        ∃ ξ, ξs[i]? = some ξ ∧ drawCoins key w u.label u.after (useCoins coins i) = .ok ξ := by
  unfold roundCoins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases (‹(throw _ : Except String _) = Except.ok _›); done) | skip
  all_goals
    cases h
    refine ⟨⟨_, ‹_›, by first | assumption | exact none_of_forall ‹_›⟩, ‹_›,
      ⟨_, _, ‹_›, strs_of_mapM ?_ ‹_›, by first | assumption | exact none_of_forall ‹_›,
        by first | assumption | exact none_of_forall ‹_›⟩, ?_⟩
    · intro j s hj
      try dsimp only at hj
      split at hj <;>
        first | (cases hj; rfl) | (cases hj; done) | (cases (‹(throw _ : Except String _) = Except.ok _›); done)
    · intro i u hu
      obtain ⟨ξ, s, hξ, hs, hd⟩ := zip_draw (P := fun u s r => drawCoins key w u.label u.after
          (Json.mkObj [("ledger", getD coins "ledger"), ("main", getD coins "main"), ("pins", getD coins "pins"),
            ("segment", s), ("session", getD coins "session")]) = .ok r) ‹_› ‹List.mapM _ _ = Except.ok _›
        (fun u s r hf => by
          try dsimp only at hf
          repeat' split at hf
          all_goals first | (cases hf; done) | (cases (‹(throw _ : Except String _) = Except.ok _›); done) |
            (cases hf; assumption))
        hu
      exact ⟨ξ, hξ, by rw [useCoins_eq ‹_› hs]; exact hd⟩

/-- **What an accepted `drawCoins` read checked**: the session `trustedSessionOf` reads, on a withheld window. -/
theorem drawCoins_ok {key : ChallengerKey} {w label : String} {after : ByteArray} {coins : Json} {ξ : ByteArray}
    (h : drawCoins key w label after coins = .ok ξ) :
    (∃ cr, trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key (getD coins "pins")
        (getD coins "main") (getD coins "session") = .ok cr) ∧ withheldOn (getD coins "ledger") w = true := by
  unfold drawCoins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases (‹(throw _ : Except String _) = Except.ok _›); done) | skip
  all_goals exact ⟨⟨_, ‹_›⟩, ‹_›⟩

theorem pairwise_zipIdx {α : Type} : ∀ (l : List α) (k : Nat), (l.zipIdx k).Pairwise fun x y => x.2 < y.2
  | [], _ => List.Pairwise.nil
  | a :: l, k => by
    rw [List.zipIdx_cons, List.pairwise_cons]
    exact ⟨fun y hy => Nat.lt_of_lt_of_le (Nat.lt_succ_self k) (List.le_snd_of_mem_zipIdx hy), pairwise_zipIdx l (k + 1)⟩

theorem le_of_getLast? {α : Type} {l : List (α × Nat)} (hp : l.Pairwise fun x y => x.2 < y.2) {x y : α × Nat}
    (hx : x ∈ l) (hy : l.getLast? = some y) : x.2 ≤ y.2 := by
  obtain ⟨ys, rfl⟩ := List.getLast?_eq_some_iff.1 hy
  rw [List.pairwise_append] at hp
  rcases List.mem_append.1 hx with h | h
  · exact Nat.le_of_lt (hp.2.2 x h y (List.mem_singleton_self y))
  · obtain rfl := List.mem_singleton.1 h
    exact Nat.le_refl _

/-- **The order check's guarantee**: where `releaseOrderReason` passes the labels, each pair `(a, b)` of `order` with
`a ≠ b` has every `a` before every `b`. -/
theorem before_of_releaseOrderReason {labels : List String} {order : List (String × String)}
    (h : releaseOrderReason labels order = none) {a b : String} (hab : (a, b) ∈ order) (hne : a ≠ b) :
    ∀ i j : Nat, labels[i]? = some a → labels[j]? = some b → i < j := by
  intro i j hi hj
  have hx := List.findSome?_eq_none_iff.1 h (a, b) hab
  dsimp only at hx
  have hmem : (a, i) ∈ labels.zipIdx.filter (·.1 == a) :=
    List.mem_filter.2 ⟨List.mem_zipIdx_iff_getElem?.2 hi, by simp⟩
  obtain ⟨⟨a', i0⟩, hlast⟩ : ∃ y, (labels.zipIdx.filter (·.1 == a)).getLast? = some y := by
    cases hl : (labels.zipIdx.filter (·.1 == a)).getLast? with
    | none => rw [List.getLast?_eq_none_iff] at hl; rw [hl] at hmem; cases hmem
    | some y => exact ⟨y, rfl⟩
  have hii0 : i ≤ i0 := le_of_getLast? (List.Pairwise.filter _ (pairwise_zipIdx labels 0)) hmem hlast
  have hjb : labels[j]'(List.getElem?_eq_some_iff.1 hj).1 = b := (List.getElem?_eq_some_iff.1 hj).2
  obtain ⟨j0, hj0⟩ : ∃ j0, labels.findIdx? (· == b) = some j0 := by
    cases hf : labels.findIdx? (· == b) with
    | none =>
      have := List.findIdx?_eq_none_iff.1 hf _ (List.getElem_mem (List.getElem?_eq_some_iff.1 hj).1)
      simp [hjb] at this
    | some j0 => exact ⟨j0, rfl⟩
  have hj0j : j0 ≤ j := by
    obtain ⟨_, _, hpre⟩ := List.findIdx?_eq_some_iff_getElem.1 hj0
    refine Nat.le_of_not_lt fun hjj => hpre j hjj ?_
    simp [hjb]
  have hi0j0 : i0 ≤ j0 := by
    simp only [hlast, hj0] at hx
    refine Nat.le_of_not_lt fun hlt => ?_
    simp [hlt] at hx
  have hij : i ≠ j := by
    rintro rfl
    rw [hi] at hj
    exact hne (Option.some.inj hj)
  omega

local instance lawfulBEqByteArrayRounds : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

/-- **The exactly check's guarantee**: where `exactlyReason` passes, each family of the layout shows exactly the uses'
roots. -/
theorem roots_of_exactlyReason {c : LiveCommitment} {roots : Roots} {us : List Use} (h : exactlyReason c roots us = none)
    {l : String} {n : Nat} (hl : (l, n) ∈ c.families) :
    ((roots.find? (·.1 == l)).map (·.2)).getD [] = Use.roots us l := by
  have hx := List.findSome?_eq_none_iff.1 h (l, n) hl
  dsimp only at hx
  split at hx
  · exact beq_iff_eq.1 ‹_›
  · cases hx

theorem Session.families_of_commitment {s : Session} {c : LiveCommitment} (h : s.commitment = some c) :
    c.families = s.families := by
  unfold Session.commitment at h
  split at h
  · cases h
    rfl
  · cases h

theorem Session.rootsOf_of {s : Session} {c : LiveCommitment} {roots : Roots} (hc : s.commitment = some c)
    (hr : releasedRootsOf c s.log = .ok roots) (ℓ : String) :
    s.rootsOf ℓ = ((roots.find? (·.1 == ℓ)).map (·.2)).getD [] := by
  simp only [Session.rootsOf, hc, hr]

/-- **The session the reader trusts is the device's** (a fact about the reader's code, not yet proved): where
`trustedSessionOf` accepts the session's files `session` from the ledger `ledger` on window `w` under `key`, and
`withheldOn` the window, the device ran under `key` a session `s` that its own ledger opens withheld on `w`
(`OpensOn`), whose commitment is the one read, whose log reads to the roots read, and whose log the session's files
hold, line for line; or the finder outputs a break from every record that carries those ledger, pins, main and session
files. The reader checks the record's, the close's and every ledger line's LMS signature under `key`
(`Challenger.verifies`) and binds the log by its chain, so a signature over a message the device didn't sign is a
`forgery` witness and two different strings behind one of those messages' digests a `collision` witness. -/
def SessionBinding : Prop :=
  ∀ (dev : Device) (key : ChallengerKey) (w : String) (ledger pins main session : Json) (cr : LiveCommitment × Roots),
    trustedSessionOf ledger (Json.mkObj [("root", Json.str w)]) key pins main session = .ok cr →
    withheldOn ledger w = true →
      (∃ s ∈ dev.sessions key.pub, OpensOn dev key.pub w s.id ∧ s.commitment = some cr.1 ∧
          releasedRootsOf cr.1 s.log = .ok cr.2 ∧
          ∀ ls : Array Json, getD session "log" = .arr ls → ls.toList = s.log.map Json.str) ∨
        ∀ (label : String) (rec : Json), getD rec "ledger" = ledger → getD rec "pins" = pins →
          getD rec "main" = main → getD rec "session" = session → Break dev key label rec

/-- **What the reader reads from the device's session is what it released** (a fact about the reader's code, not yet
proved): where the session `trustedSessionOf` reads from the record `rec` is the device's session `s` (its commitment
`cr.1`, its log's roots `cr.2`), what `drawCoins` reads from `rec` for family `label` after `after` is what `s`
released of `label` after `after`; or the finder outputs a break from `rec`. `challengerStream` checks each opening's
coin against the commitment's root (`openingHashed`), the device's tree over its own coins (`deviceHashed`), so a coin
other than the device's is a `collision` witness. -/
def CoinBinding : Prop :=
  ∀ (dev : Device) (key : ChallengerKey) (w label : String) (after : ByteArray) (rec : Json) (ξ : ByteArray)
    (s : Session) (cr : LiveCommitment × Roots), s ∈ dev.sessions key.pub → s.commitment = some cr.1 →
    releasedRootsOf cr.1 s.log = .ok cr.2 →
    trustedSessionOf (getD rec "ledger") (Json.mkObj [("root", Json.str w)]) key (getD rec "pins") (getD rec "main")
      (getD rec "session") = .ok cr →
    drawCoins key w label after rec = .ok ξ → ξ = s.released label after ∨ Break dev key label rec

/-- **`RoundsByDevice`'s conclusion, from the two binding facts**: for every read `roundCoins` accepts with at least
one use and no pair `(a, a)` in `order`, one session of the device under the key, which its ledger opens withheld on
the window, released each use's coins, shows exactly the uses' roots, releases each pair of `order` in order and each
group's uses one run each; or the finder outputs a break from some use's record. -/
theorem roundsByDevice_of (hS : SessionBinding) (hC : CoinBinding) (dev : Device) (key : ChallengerKey) (w : String)
    (us : List Use) (order : List (String × String)) (groups : List (List String)) (coins : Json)
    (ξs : List ByteArray) (hus : us ≠ []) (hord : ∀ ab ∈ order, ab.1 ≠ ab.2)
    (h : roundCoins key w us order groups coins = .ok ξs) :
    (∃ s ∈ dev.sessions key.pub, OpensOn dev key.pub w s.id ∧
        (∀ (i : Nat) (u : Use), us[i]? = some u → ξs[i]? = some (s.released u.label u.after)) ∧
        (∀ ℓ ∈ s.families.map (·.1), s.rootsOf ℓ = Use.roots us ℓ) ∧
        (∀ ab ∈ order, s.Before ab.1 ab.2) ∧ ∀ g ∈ groups, s.Runs g us) ∨
      ∃ (i : Nat) (u : Use), us[i]? = some u ∧ Break dev key u.label (useCoins coins i) := by
  obtain ⟨⟨cr, htr, hex⟩, hwh, ⟨ls, lines, hlog, hls, hro, hrw⟩, hdraw⟩ := roundCoins_ok h
  rcases hS dev key w (getD coins "ledger") (getD coins "pins") (getD coins "main") (getD coins "session") cr htr hwh
    with ⟨s, hs, hopen, hc, hr, hslog⟩ | hB
  · by_cases hbr : ∃ (i : Nat) (u : Use), us[i]? = some u ∧ Break dev key u.label (useCoins coins i)
    · exact .inr hbr
    · left
      have hlines : lines = s.log := map_str_inj (hls.symm.trans (hslog ls hlog))
      subst hlines
      refine ⟨s, hs, hopen, ?_, ?_, ?_, Session.runs_of_runsWhy hrw⟩
      · intro i u hu
        obtain ⟨ξ, hξ, hd⟩ := hdraw i u hu
        obtain ⟨hl, hp, hm, hss⟩ := useCoins_get coins i
        rcases hC dev key w u.label u.after (useCoins coins i) ξ s cr hs hc hr (by rw [hl, hp, hm, hss]; exact htr) hd
          with he | hb
        · rw [hξ, he]
        · exact absurd ⟨i, u, hu, hb⟩ hbr
      · intro ℓ hℓ
        obtain ⟨⟨l, n⟩, hmem, rfl⟩ := List.mem_map.1 hℓ
        rw [Session.rootsOf_of hc hr]
        exact roots_of_exactlyReason hex (by rw [Session.families_of_commitment hc]; exact hmem)
      · rintro ⟨a, b⟩ hab i j hi hj
        exact before_of_releaseOrderReason hro hab (hord _ hab) i j hi hj
  · right
    obtain ⟨u, us', rfl⟩ := List.exists_cons_of_ne_nil hus
    obtain ⟨hl, hp, hm, hss⟩ := useCoins_get coins 0
    exact ⟨0, u, rfl, hB u.label (useCoins coins 0) hl hp hm hss⟩

/-- **`DrawnByDevice`, from the same two binding facts.** -/
theorem drawnByDevice_of (hS : SessionBinding) (hC : CoinBinding) : DrawnByDevice := by
  intro dev key w label after coins ξ h
  obtain ⟨⟨cr, htr⟩, hwh⟩ := drawCoins_ok h
  rcases hS dev key w _ _ _ _ cr htr hwh with ⟨s, hs, hopen, hc, hr, -⟩ | hB
  · rcases hC dev key w label after coins ξ s cr hs hc hr htr h with he | hb
    · exact .inl ⟨s, hs, hopen, he⟩
    · exact .inr hb
  · exact .inr (hB label coins rfl rfl rfl rfl)

/-- **A pair `(a, a)` of `order` is outside what the order check shows**: on a log that receives family `a` once,
`releaseOrderReason` passes `[(a, a)]`, while `Session.Before s a a` fails. -/
theorem order_self_gap (a : String) (s : Session) (hs : receivedOrder s.log = [a]) :
    releaseOrderReason (receivedOrder s.log) [(a, a)] = none ∧ ¬ s.Before a a := by
  refine ⟨?_, fun hb => Nat.lt_irrefl 0 (hb 0 0 (by simp [hs]) (by simp [hs]))⟩
  rw [hs]
  simp [releaseOrderReason, List.findIdx?_cons, List.findSome?_cons]

end Properties.Challenger
