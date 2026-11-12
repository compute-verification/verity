module

public import Proofs.Properties.Verity.Sampled.Window.Sessions

/-!
# Sampled proofs' coupling over a window: the rounds a session's trace leaves on one stream

A member's session's trace answers its rounds at releases of the one play's log (`Window/Answered.lean`), and the
window's readings say which opening is which session's round (`Window/Sessions.lean`). This module reads, on lists
alone, the rounds the trace's answered stops leave on one rep stream of the session's tree, as `Stops.lean` does at
one session:

* **The answered stops of one stream are at its openings, in order** (`session_stream_logs`): where the stops each
  release their coins at an opening a rep round of the session picks, every release a round of the session picks is
  the link's, one of theirs or the closing round's, the tree's first opening is of its stream named the session's, and
  the session has a
  round for each opening of the stream, the logs after which the stops on that stream release their coins are the
  logs before each of its openings.
* **The rounds of one stream** (`session_stream_rounds`): so the rounds the trace appends to a stream are one per
  opening of it, the `k`-th after the log that has opened it `k` times.
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness.Discharge.Tables (addRoundJ)

/-- **The answered stops of one stream of a session are at its openings, in order**, at the readings of a window a
session at a time, of distinct tables: where each answered stop `as` releases its round's coins after one of the logs
`ms` of `Lg`, each longer than the one before, at an opening a round of session `u` picks, a rep round of the stop's
table and rep; every release a round of session `u` picks in `Lg` is after `L₀` or one of `ms`, or is the session's
closing round's, and only the link's round picks after `L₀`; each opening of `Lg` is one the device serves, each tree
of `Lg` names its streams distinctly,
the first opening of the `u`-th tree `Lg` opens is of its stream named the session's, that tree's stream `s` is named
table `t`'s rep `r`, and session `u` has a round for each of `Lg`'s openings of it: the logs after which the stops
submitting on table `t`'s rep `r` release their coins are the logs before each of `Lg`'s openings of `s`, in order
(`stream_logs`, at a session of a window). -/
theorem session_stream_logs {γ : Type _} {J : ℕ} (tab : γ → Fin J) (rep : γ → Fin 2) {R : ℕ}
    {tables : Array String} (ht : tables.toList.Nodup) {per : ℕ} (hper : 0 < per)
    (reveal : ℕ → Slot → Exchange → Option Claims.Msg) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {as : List γ} {ms : List (List Release)} {Lg L₀ : List Release} {u τ : ℕ}
    (hτ : (openedTrees Lg)[u]? = some τ)
    (hF : List.Forall₂ (fun a m => ∃ (j : Fin R) (k : ℕ) (x : Release) (L' : List Release),
      j.val / per = u ∧ slotOf tables.size per (j.val % per) = .rep (tab a).val (rep a) k ∧
        (windowReadingAt tables per reveal record recordDigest).round j m x.frame ∧ Lg = m ++ x :: L') as ms)
    (hpw : ms.Pairwise (fun m m' => m.length < m'.length))
    (hcov : ∀ (j : Fin R) L₁, j.val / per = u →
      PicksIn (windowReadingAt tables per reveal record recordDigest).round Lg j L₁ →
        L₁ = L₀ ∨ L₁ ∈ ms ∨ slotOf tables.size per (j.val % per) = closeSlot tables.size)
    (hL₀ : ∀ j : Fin R, PicksIn (windowReadingAt tables per reveal record recordDigest).round Lg j L₀ →
      slotOf tables.size per (j.val % per) = .link)
    (hS : LogServed Lg) (hn : ∀ sp ∈ helloSpecs Lg, sp.streams.toList.Nodup)
    (hfirst : ∀ L₁ x L₂ s n, Lg = L₁ ++ x :: L₂ → x.frame = .opening τ s n → τ ∉ openedTrees L₁ →
      ((helloSpecs L₁)[τ]?.bind fun sp => sp.streams[s]?) = some _root_.Spot.Challenger.TREE_SESSION)
    (t : Fin J) (htl : t.val < tables.size) (r : Fin 2) {s : ℕ}
    (hs : ((helloSpecs Lg)[τ]?.bind fun sp => sp.streams[s]?) = tables[t.val]?.map (repStreamName · r.val))
    (hcovL : ∀ k < openedOf Lg τ s, ∃ j : Fin R, j.val / per = u ∧
      slotOf tables.size per (j.val % per) = .rep t.val r k) :
    ((as.zip ms).filter fun x => decide (tab x.1 = t ∧ rep x.1 = r)).map Prod.snd =
      prefixesWhere (fun x => x.frame.opensAt τ s) Lg := by
  have hlen := hF.length_eq
  refine filter_logs (xs := as.zip ms) (by rw [List.map_snd_zip hlen.ge]; exact hpw)
    (fun a => decide (tab a = t ∧ rep a = r)) fun m => ⟨?_, ?_⟩
  · rintro ⟨⟨a, m'⟩, hx, hpx, hxm⟩
    have hxm' : m' = m := hxm
    rw [← hxm']
    obtain ⟨j, k, x, L', hju, hj, hP, he⟩ := List.forall₂_zip hF hx
    have htr : tab a = t ∧ rep a = r := of_decide_eq_true hpx
    have hP' : sessionRound tables (j.val / per) (slotOf tables.size per (j.val % per)) m' x.frame := hP
    rw [hju, hj, htr.1, htr.2] at hP'
    obtain ⟨i, s', n, hfr, ⟨sp, hsp, hslt, -⟩, hu, hname, -⟩ := hP'
    rw [slotStream_lt htl r] at hname
    have hi : i = τ := by
      have h := openedTrees_getElem?_append hu (x :: L')
      rw [← he, hτ] at h
      exact (Option.some.inj h).symm
    rw [hi] at hfr hsp hname
    have hss : s' = s := stream_eq he hn ⟨sp, hsp, hslt⟩ hname hs
    rw [hss] at hfr
    exact ⟨x, L', he, by rw [hfr]; simp [Frame.opensAt]⟩
  · rintro ⟨x, L₂, he, hx⟩
    obtain ⟨n, hxn⟩ := opening_of_opensAt hx
    have hsv := hS m x L₂ τ s n he hxn
    by_cases hmem : τ ∈ openedTrees m
    · obtain ⟨v, hv⟩ := List.getElem?_of_mem hmem
      have hvu : v = u := by
        have h := openedTrees_getElem?_append hv (x :: L₂)
        rw [← he] at h
        exact nodup_getElem?_inj (openedTrees_nodup Lg) h hτ
      rw [hvu] at hv
      obtain ⟨sp, hsp, -⟩ := id hsv
      have hname : ((helloSpecs m)[τ]?.bind fun sp => sp.streams[s]?) =
          tables[t.val]?.map (repStreamName · r.val) := by
        have hsp' : (helloSpecs Lg)[τ]? = some sp := by
          rw [he]
          exact helloSpecs_getElem?_append hsp _
        rw [hsp, Option.bind_some]
        rw [hsp', Option.bind_some] at hs
        exact hs
      have hk : openedOf m τ s < openedOf Lg τ s := by
        rw [he]
        exact openedOf_lt m hxn L₂
      obtain ⟨j, hju, hj⟩ := hcovL _ hk
      have hpick : (windowReadingAt tables per reveal record recordDigest).round j m x.frame := by
        show sessionRound tables (j.val / per) (slotOf tables.size per (j.val % per)) m x.frame
        rw [hju, hj]
        exact ⟨τ, s, n, hxn, hsv, hv, hname.trans (slotStream_lt htl r).symm, rfl⟩
      have hmm : m ∈ ms := by
        rcases hcov j m hju ⟨x, L₂, he, hpick⟩ with e | h | h
        · have hl : slotOf tables.size per (j.val % per) = .link := hL₀ j (by rw [← e]; exact ⟨x, L₂, he, hpick⟩)
          rw [hj] at hl
          cases hl
        · exact h
        · rw [hj] at h
          have h' : Slot.rep t.val r (openedOf m τ s) = .rep tables.size 0 1 := h
          rw [Slot.rep.injEq] at h'
          exact absurd h'.1 (by omega)
      obtain ⟨a, hz⟩ := forall₂_zip_right hF hmm
      obtain ⟨j', k', x', L'', -, hj', hP', he'⟩ := List.forall₂_zip hF hz
      obtain ⟨hxe, -⟩ := List.cons.inj (List.append_cancel_left (he.symm.trans he'))
      rw [← hxe] at hP'
      have hjj : j' = j := windowReadingAt_unique ht (by omega) hper reveal record recordDigest hP' hpick
      rw [hjj, hj] at hj'
      obtain ⟨ht't, hr'r, -⟩ := Slot.rep.inj hj'
      refine ⟨(a, m), hz, decide_eq_true ?_, rfl⟩
      show tab a = t ∧ rep a = r
      exact ⟨Fin.ext ht't.symm, hr'r.symm⟩
    · have h1 := helloSpecs_bind_append (hfirst m x L₂ s n he hxn hmem) (x :: L₂)
      rw [← he, hs] at h1
      exact absurd h1 rep_ne_session

/-- **The rounds the answered stops leave on one stream of a session are one per opening of it**, under
`session_stream_logs`'s premises: where the round each answered stop on table `t`'s rep `r` appends is `f` at the
count of the stream's openings its log has, the stream's rounds are `f` at none, one, and so on up to the count of
them in `Lg` (`stream_rounds`, at a session of a window). -/
theorem session_stream_rounds {γ : Type _} {J : ℕ} (tab : γ → Fin J) (rep : γ → Fin 2) (rd : γ → Flock.Round)
    {R : ℕ} {tables : Array String} (ht : tables.toList.Nodup) {per : ℕ} (hper : 0 < per)
    (reveal : ℕ → Slot → Exchange → Option Claims.Msg) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {as : List γ} {ms : List (List Release)} {Lg L₀ : List Release} {u τ : ℕ}
    (hτ : (openedTrees Lg)[u]? = some τ)
    (hF : List.Forall₂ (fun a m => ∃ (j : Fin R) (k : ℕ) (x : Release) (L' : List Release),
      j.val / per = u ∧ slotOf tables.size per (j.val % per) = .rep (tab a).val (rep a) k ∧
        (windowReadingAt tables per reveal record recordDigest).round j m x.frame ∧ Lg = m ++ x :: L') as ms)
    (hpw : ms.Pairwise (fun m m' => m.length < m'.length))
    (hcov : ∀ (j : Fin R) L₁, j.val / per = u →
      PicksIn (windowReadingAt tables per reveal record recordDigest).round Lg j L₁ →
        L₁ = L₀ ∨ L₁ ∈ ms ∨ slotOf tables.size per (j.val % per) = closeSlot tables.size)
    (hL₀ : ∀ j : Fin R, PicksIn (windowReadingAt tables per reveal record recordDigest).round Lg j L₀ →
      slotOf tables.size per (j.val % per) = .link)
    (hS : LogServed Lg) (hn : ∀ sp ∈ helloSpecs Lg, sp.streams.toList.Nodup)
    (hfirst : ∀ L₁ x L₂ s n, Lg = L₁ ++ x :: L₂ → x.frame = .opening τ s n → τ ∉ openedTrees L₁ →
      ((helloSpecs L₁)[τ]?.bind fun sp => sp.streams[s]?) = some _root_.Spot.Challenger.TREE_SESSION)
    (t : Fin J) (htl : t.val < tables.size) (r : Fin 2) {s : ℕ}
    (hs : ((helloSpecs Lg)[τ]?.bind fun sp => sp.streams[s]?) = tables[t.val]?.map (repStreamName · r.val))
    (hcovL : ∀ k < openedOf Lg τ s, ∃ j : Fin R, j.val / per = u ∧
      slotOf tables.size per (j.val % per) = .rep t.val r k)
    (f : ℕ → Flock.Round)
    (hf : ∀ a m, (a, m) ∈ as.zip ms → tab a = t ∧ rep a = r → rd a = f (openedOf m τ s)) :
    as.foldl (fun rs a => addRoundJ rs (tab a) (rep a) (rd a)) (fun _ _ => []) t r =
      (List.range (openedOf Lg τ s)).map f := by
  refine (foldl_addRoundJ tab rep rd as (fun _ _ => []) t r).trans ((List.nil_append _).trans ?_)
  exact rounds_of_logs hF.length_eq (fun a => decide (tab a = t ∧ rep a = r)) rd f
    (session_stream_logs tab rep ht hper reveal record recordDigest hτ hF hpw hcov hL₀ hS hn hfirst t htl r hs
      hcovL)
    fun a m hx hp => hf a m hx (of_decide_eq_true hp)

end Verity.Sampled
