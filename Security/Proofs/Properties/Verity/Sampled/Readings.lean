module

public import Proofs.Properties.Verity.Sampled.Decoders
public import Proofs.Properties.Verity.Sampled.Walk

/-!
# Sampled proofs' coupling: the readings at a layout

The readings `MovesReach` reads, at a layout of the session's rounds (`Slot`): where each round's message is fixed on
the period's game, and what the play's output reveals for it.

* **Which opening a round's reading picks** (`slotRound`): the link's is the first opening, round `k` of table `t`'s rep
  `r` stream an opening of that stream of the first tree opened after `k` openings of it, each one the device serves
  (`Served`). The device's trees, as its log shows them, have room for a served opening (`Served.cap`) and answer it
  with a round's coins (`Served.next`), so the walk stops there at an opening's coin.
* **What the play reveals** (`slotReveal`): the link's message (`linkIn`) and round `k`'s message on the rep stream
  (`roundIn`), in the record v2 reads.
* **The readings at a layout** (`readingAt`): each picks only openings (`readingAt_picks`), and where the layout is
  injective and the tables distinct, no two rounds pick one frame (`readingAt_unique`).
-/

@[expose] public section

namespace Verity.Sampled

/-- **The tree the log's first opening opens**; `none` where the log opens none. -/
def firstOpened (log : List Release) : Option ℕ :=
  log.findSome? fun r => match r.frame with
    | .opening i _ _ => some i
    | _ => none

/-- **The device serves an opening of tree `i`'s stream `s` for `n` coins after the log `log`**: the log's `i`-th
`Hello` has a stream `s` with a round left after the log's openings of it, `n` is at most a block, and the log's
releases wrote fewer than `maxTreeLines` coin-tree lines. -/
def Served (log : List Release) (i s n : ℕ) : Prop :=
  ∃ sp, (helloSpecs log)[i]? = some sp ∧ s < sp.streams.size ∧ openedOf log i s < 2 ^ sp.roundsLog ∧ n ≤ sp.block ∧
    Leaves.logTreeLines log < maxTreeLines

/-- **Whether a frame after the log `pre` is the round at the slot**: the link's, an opening the device serves before
any other opening; round `k` of table `t`'s rep `r` stream, an opening the device serves of the first tree opened, of
its stream named `t/rep<r>`, after `k` openings of that stream. -/
def slotRound (tables : Array String) : Slot → List Release → Frame → Prop
  | .link, pre, fr => ∃ i s n, fr = .opening i s n ∧ Served pre i s n ∧ firstOpened pre = none
  | .rep t r k, pre, fr => ∃ i s n, fr = .opening i s n ∧ Served pre i s n ∧ firstOpened pre = some i ∧
      ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = tables[t]?.map (repStreamName · r.val) ∧
      openedOf pre i s = k

/-- **What the play reveals for the round at the slot**: the link's message, or round `k`'s on table `t`'s rep `r`
stream, in the record v2 reads. -/
def slotReveal (tables : Array String) : Slot → Exchange → Option Claims.Msg
  | .link => linkIn tables
  | .rep t r k => fun o => tables[t]?.bind fun a => roundIn (repStreamName a r.val) k o

/-- **The readings at the layout `L`** of the tables `tables`, with the record the play's output reveals (`record`)
and the record's digest (`recordDigest`). -/
def readingAt {R : ℕ} (tables : Array String) (L : Fin R → Slot) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) : Reading R where
  round j := slotRound tables (L j)
  reveal j := slotReveal tables (L j)
  record := record
  recordDigest := recordDigest

theorem slotRound_opening {tables : Array String} {σ : Slot} {pre : List Release} {fr : Frame}
    (h : slotRound tables σ pre fr) : ∃ i s n, fr = .opening i s n := by
  cases σ with
  | link =>
    obtain ⟨i, s, n, h, -⟩ := h
    exact ⟨i, s, n, h⟩
  | rep t r k =>
    obtain ⟨i, s, n, h, -⟩ := h
    exact ⟨i, s, n, h⟩

/-- **Each round's reading picks only openings.** -/
theorem readingAt_picks {R : ℕ} (tables : Array String) (L : Fin R → Slot) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) :
    ∀ j pre fr, (readingAt tables L record recordDigest).round j pre fr → ∃ i s n, fr = .opening i s n :=
  fun _ _ _ h => slotRound_opening h

theorem nodup_getElem?_inj {α : Type} {l : List α} (hl : l.Nodup) {i j : ℕ} {a : α} (hi : l[i]? = some a)
    (hj : l[j]? = some a) : i = j := by
  obtain ⟨hi', rfl⟩ := List.getElem?_eq_some_iff.1 hi
  obtain ⟨hj', hj''⟩ := List.getElem?_eq_some_iff.1 hj
  exact (List.Nodup.getElem_inj_iff hl).1 hj''.symm

/-- **No two slots pick one frame** after one log, where the tables are distinct. -/
theorem slotRound_unique {tables : Array String} (ht : tables.toList.Nodup) {σ σ' : Slot} {pre : List Release}
    {fr : Frame} (h : slotRound tables σ pre fr) (h' : slotRound tables σ' pre fr) : σ = σ' := by
  cases σ with
  | link =>
    cases σ' with
    | link => rfl
    | rep t' r' k' =>
      obtain ⟨_, _, _, _, _, hf⟩ := h
      obtain ⟨_, _, _, _, _, hf', _⟩ := h'
      rw [hf] at hf'
      cases hf'
  | rep t r k =>
    cases σ' with
    | link =>
      obtain ⟨_, _, _, _, _, hf⟩ := h'
      obtain ⟨_, _, _, _, _, hf', _⟩ := h
      rw [hf] at hf'
      cases hf'
    | rep t' r' k' =>
      obtain ⟨i, s, n, rfl, ⟨sp, hsp, hs, -⟩, -, hname, hk⟩ := h
      obtain ⟨i', s', n', he, -, -, hname', hk'⟩ := h'
      obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj he
      have hX : ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some sp.streams[s] := by
        rw [hsp, Option.bind_some, Array.getElem?_eq_getElem hs]
      rw [hX] at hname hname'
      obtain ⟨a, ha, hae⟩ := Option.map_eq_some_iff.1 hname.symm
      obtain ⟨a', ha', hae'⟩ := Option.map_eq_some_iff.1 hname'.symm
      obtain ⟨rfl, rfl⟩ := repStreamName_inj (hae.trans hae'.symm)
      rw [← Array.getElem?_toList] at ha ha'
      obtain rfl := nodup_getElem?_inj ht ha ha'
      subst hk hk'
      rfl

/-- **No two rounds pick one frame** after one log, where the layout is injective and the tables distinct. -/
theorem readingAt_unique {R : ℕ} {tables : Array String} {L : Fin R → Slot} (hL : Function.Injective L)
    (ht : tables.toList.Nodup) (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {j j' : Fin R}
    {pre : List Release} {fr : Frame} (h : (readingAt tables L record recordDigest).round j pre fr)
    (h' : (readingAt tables L record recordDigest).round j' pre fr) : j = j' :=
  hL (slotRound_unique ht h h')

/-- **The device's trees answer an opening it serves** with the stream's next round, the log's count of its openings,
where the trees are the ones the log shows. -/
theorem Served.next {pre : List Release} {ts : List OpenTree} (hT : TreesOf pre ts) {i s n : ℕ}
    (h : Served pre i s n) :
    ∃ T, ts[i]? = some T ∧ ∃ (hs : s < T.spec.streams.size) (hr : T.opened ⟨s, hs⟩ < 2 ^ T.spec.roundsLog),
      T.next s n = some (⟨s, hs⟩, ⟨T.opened ⟨s, hs⟩, hr⟩) ∧ T.opened ⟨s, hs⟩ = openedOf pre i s := by
  obtain ⟨sp, hsp, hs, hr, hn, -⟩ := h
  rw [← hT.specs, List.getElem?_map] at hsp
  obtain ⟨T, hTi, hTs⟩ := Option.map_eq_some_iff.1 hsp
  subst hTs
  have ho : T.opened ⟨s, hs⟩ = openedOf pre i s := (openedIn_some hTi hs).symm.trans (hT.opened i s)
  have hr' : T.opened ⟨s, hs⟩ < 2 ^ T.spec.roundsLog := by
    rw [ho]
    exact hr
  exact ⟨T, hTi, hs, hr', OpenTree.next_eq hs hr' hn, ho⟩

/-- **The device's trees have room for an opening it serves**: they hold fewer than `maxTreeLines` coin-tree lines,
where they are the ones the log shows. -/
theorem Served.cap {pre : List Release} {ts : List OpenTree} (hT : TreesOf pre ts) {i s n : ℕ}
    (h : Served pre i s n) : treeLines ts < maxTreeLines := by
  obtain ⟨-, -, -, -, -, hc⟩ := h
  rw [hT.lines]
  exact hc

end Verity.Sampled
