module

public import Proofs.Properties.Verity.SampledTwo.FormDraw
public import Proofs.Properties.Verity.SampledTwo.CoinBytes
public import Proofs.Properties.Verity.SampledTwo.Fits
public import Proofs.Properties.Verity.SampledTwo.Within
public import Proofs.Properties.Verity.SampledTwo.RoundOne
public import Proofs.Properties.Verity.SampledTwo.StageLog
public import Proofs.Properties.Verity.SampledTwo.FirstCoins
public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Properties.Verity.SampledDraw

/-!
# One pair of stage coins, from the stages' bytes

`PlayCoinsOne` (`FormDraw.lean`) asks, of two accepted plays with one parameters and one log before the statements'
round, for one round-0 coin and one second stage's coins that carry the bytes each play's samples read
(`SpotFlock.RunsAs.PlayCoins`). Here it is a theorem where, stage by stage, one of the two plays' bytes starts the
other's (`OneStarts`): the first stage's (`FirstBytesOne`) and the second's (`SecondBytesOne`).

* **`PlayCoinsOne` from the bytes** (`playCoinsOne_of_bytes`). An accepted play's first-stage bytes are within one
  coin's and its second-stage bytes within `stage` coins' (`sourcesFit`), and any bytes within a coin's are a coin's
  first bytes (`coinsOfBytes`, `stagesOfBytes`). So where one of two byte strings starts the other, one coin's bytes
  start with both (`coin_of_oneStarts`), and one second stage's coins' do (`stage_of_oneStarts`). A sample reads no
  byte past its source (`readsWithin`, at a held copy with no closure map, which an accepted play's is:
  `closure_of_held`, `closure_of_formAccepts`), so coins whose bytes start with a play's bytes carry what its
  samples read (`playCoins_of_starts`).
* **The bytes, from the log before the statements' round** (`oneStarts_of_before`). The check reads each stage's
  bytes off the play's log, as the start of the coins the log shows released in the stage's family after its
  record's digest (`FirstShown`, `SecondShown`). Two plays with one log before the statements' round share the
  releases before the first one the locator puts on a later round, which at the run's locator is an opening
  (`OpensFrom`, `opensFrom_run`). Where both plays' bytes start the family's coins those shared releases show, and
  the family's coins answer one digest on each play's log (`CoinMap.OneRoot`), the two plays read one string of
  coins, so one's bytes start the other's.
* **`SecondBytesOne`** (`secondBytesOne_of`). No release of the second stage's sample follows an opening
  (`sample_before_openings`), so the log before the statements' round shows every coin of that sample the log shows
  (`freshCoins_before`).
* **`FirstBytesOne`** (`firstBytesOne_of`), from `FirstShownEarly`: the bytes the first stage read start the coins
  that releases before every opening show. Such releases start the log before the statements' round
  (`prefix_before_of_no_opening`).

**Not established here**: `FirstShownEarly`, a hypothesis. `FirstShown` reads the first stage's bytes off the whole
log, and no lemma here holds the coins it reads to be released before an opening; `firstShownEarly`
(`FirstEarly.lean`) does, at the cut's read.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- **One of two byte strings starts the other.** -/
def OneStarts (a b : ByteArray) : Prop := (∃ u, b = a ++ u) ∨ ∃ u, a = b ++ u

/-- **One of the two plays' first-stage bytes starts the other's**: at two plays the check accepts whose logs are the
device's, with one parameters and one log before round `stage + 1`, the bytes one run of record's first stage has
start the other's. -/
def FirstBytesOne (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o o' k k' p e e' i i', FormAccepts C o → Good o → FormAccepts C o' → Good o' → C o = some k →
    C o' = some k' → Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) →
    SessionMsgs.formOf k = some (p, e) → SessionMsgs.formOf k' = some (p, e') → inputOn C o = some i →
    inputOn C o' = some i' → OneStarts i.source i'.source

/-- **One of the two plays' second-stage bytes starts the other's**: at two plays the check accepts whose logs are
the device's, with one parameters and one log before round `stage + 1`, the bytes one run of record's interior has
start the other's. -/
def SecondBytesOne (A : Live₂) (C : CaseOf) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o o' k k' p e e' i i' it it', FormAccepts C o → Good o → FormAccepts C o' → Good o' → C o = some k →
    C o' = some k' → Messages.logBefore L o (A.h.stage + 1) = Messages.logBefore L o' (A.h.stage + 1) →
    SessionMsgs.formOf k = some (p, e) → SessionMsgs.formOf k' = some (p, e') → inputOn C o = some i →
    i.interior = some it → inputOn C o' = some i' → i'.interior = some it' → OneStarts it.source it'.source

/-- **One coin's bytes start with both of two byte strings** within a coin's bytes, one of which starts the other. -/
theorem coin_of_oneStarts {a b : ByteArray} (h : OneStarts a b) (ha : a.size ≤ SpotFlock.L)
    (hb : b.size ≤ SpotFlock.L) :
    ∃ c : FlockDeploy.Coin, (∃ u, SpotFlock.src c = a ++ u) ∧ ∃ u, SpotFlock.src c = b ++ u := by
  rcases h with ⟨u, rfl⟩ | ⟨u, rfl⟩
  · obtain ⟨c, hc⟩ := coinsOfBytes (a ++ u) hb
    exact ⟨c, ⟨_, hc.trans ByteArray.append_assoc⟩, ⟨_, hc⟩⟩
  · obtain ⟨c, hc⟩ := coinsOfBytes (b ++ u) ha
    exact ⟨c, ⟨_, hc⟩, ⟨_, hc.trans ByteArray.append_assoc⟩⟩

/-- **`R` coins' bytes start with both of two byte strings** within `R` coins' bytes, one of which starts the
other. -/
theorem stage_of_oneStarts {R : ℕ} {a b : ByteArray} (h : OneStarts a b) (ha : a.size ≤ R * SpotFlock.L)
    (hb : b.size ≤ R * SpotFlock.L) :
    ∃ cs : Fin R → FlockDeploy.Coin,
      (∃ u, SpotFlock.stageSrc cs = a ++ u) ∧ ∃ u, SpotFlock.stageSrc cs = b ++ u := by
  rcases h with ⟨u, rfl⟩ | ⟨u, rfl⟩
  · obtain ⟨cs, hc⟩ := stagesOfBytes R (a ++ u) hb
    exact ⟨cs, ⟨_, hc.trans ByteArray.append_assoc⟩, ⟨_, hc⟩⟩
  · obtain ⟨cs, hc⟩ := stagesOfBytes R (b ++ u) ha
    exact ⟨cs, ⟨_, hc⟩, ⟨_, hc.trans ByteArray.append_assoc⟩⟩

/-- **`PlayCoinsOne` from the stages' bytes** (`FirstBytesOne`, `SecondBytesOne`): one coin's bytes start with both
plays' first-stage bytes and one second stage's coins' with both plays' second-stage bytes (`coin_of_oneStarts`,
`stage_of_oneStarts`, at the sizes `sourcesFit` gives), and such coins carry what each play's samples read
(`playCoins_of_starts`). -/
theorem playCoinsOne_of_bytes {A : Live₂} {C : CaseOf} {Good : Verity.Exchange → Prop} {L : CoinMap.Locate}
    (hof : HeldOfAt A C) (hparts : PartsHeldAt A C) (h₁ : FirstBytesOne A C Good L)
    (h₂ : SecondBytesOne A C Good L) : PlayCoinsOne A C Good L := by
  intro o o' k k' p e e' i i' it it' hA hG hA' hG' hk hk' hlog hf hf' hi hit hi' hit'
  have hc : A.h.reg.first.closure = none :=
    (closure_of_held (heldAt_of_parts hof hparts o i it hA hi hit)).trans (closure_of_formAccepts hA hi)
  have hw := readsWithin A.h hc
  obtain ⟨hs, hss⟩ := sourcesFit A C o i it hA hi hit
  obtain ⟨hs', hss'⟩ := sourcesFit A C o' i' it' hA' hi' hit'
  obtain ⟨c₀, h0, h0'⟩ :=
    coin_of_oneStarts (h₁ o o' k k' p e e' i i' hA hG hA' hG' hk hk' hlog hf hf' hi hi') hs hs'
  obtain ⟨c₁, h1, h1'⟩ :=
    stage_of_oneStarts (h₂ o o' k k' p e e' i i' it it' hA hG hA' hG' hk hk' hlog hf hf' hi hit hi' hit') hss hss'
  exact ⟨c₀, c₁, playCoins_of_starts hw h0 h1, playCoins_of_starts hw h0' h1'⟩

/-! ## The bytes, from the log before the statements' round -/

/-- **The locator `L` puts only openings on the rounds from `n` on.** -/
def OpensFrom (L : CoinMap.Locate) (n : ℕ) : Prop :=
  ∀ pre fr b j, CoinMap.locAt L pre fr b = some j → n ≤ j → ∃ i s c, fr = .opening i s c

/-- **The run's locator puts only openings past the second stage's last round**: it puts a fresh coin on a round at
most `stage`, and no `Hello` on any. -/
theorem opensFrom_run (A : Live₂) (Ls : CoinMap.Locate) (tab : ℕ → String) :
    OpensFrom (SessionCoins.runLocate A Ls tab) (A.h.stage + 1) := by
  intro pre fr b j h hj
  cases fr with
  | opening i s c => exact ⟨i, s, c, rfl⟩
  | hello sp => exact absurd h (by simp [CoinMap.locAt])
  | fresh l =>
    have h' : (Ls pre (.fresh l) b).filter (fun n => decide (n ≤ A.h.stage)) = some j := h
    obtain ⟨-, hle⟩ := Option.filter_eq_some_iff.1 h'
    have := of_decide_eq_true hle
    omega

/-- A release of the log before a round is one of the log's. -/
theorem mem_of_before (L : CoinMap.Locate) (n : ℕ) (log : List Verity.Release) {x : Verity.Release}
    (hx : x ∈ (CoinMap.before L n [] log).1) : x ∈ log := by
  rcases CoinMap.before_split L (i := n) [] log with ⟨-, h1⟩ | ⟨rl, rest, j, hsplit, -, -, -⟩
  · rwa [h1] at hx
  · rw [hsplit]
    exact List.mem_append_left _ hx

/-- **The log before round `n` shows every coin of a family that no opening comes before**: where the locator puts
only openings on the rounds from `n` on, and no release of a coin of family `label` after `after` follows an
opening, the coins of that family the log shows are those its log before round `n` shows. -/
theorem freshCoins_before {L : CoinMap.Locate} {n : ℕ} (hL : OpensFrom L n) {log : List Verity.Release}
    {label : String} {after : ByteArray}
    (h : ∀ a r b, log = a ++ r :: b → ∀ i s c, r.frame = .opening i s c →
      ∀ x ∈ b, ¬ (x.frame = .fresh label ∧ x.answers = after)) :
    Verity.Sampled.freshCoins label after log =
      Verity.Sampled.freshCoins label after (CoinMap.before L n [] log).1 := by
  rcases CoinMap.before_split L (i := n) [] log with ⟨-, h1⟩ | ⟨rl, rest, j, hsplit, -, hj, hnj⟩
  · rw [h1]
  · obtain ⟨i, s, c, hr⟩ := hL _ _ _ _ hj hnj
    have hrest := h _ rl rest hsplit i s c hr
    have hnil : Verity.Sampled.freshCoins label after (rl :: rest) = [] :=
      Verity.Sampled.freshCoins_nil_of fun q hq hqf => by
        rcases List.mem_cons.1 hq with hq | hq
        · rw [hq, hr] at hqf
          exact Verity.Frame.noConfusion hqf.1
        · exact hrest q hq hqf
    conv_lhs => rw [hsplit, freshCoins_append, hnil, List.append_nil]

/-- **Releases before every opening start the log before round `n`**, where the locator puts only openings on the
rounds from `n` on. -/
theorem prefix_before_of_no_opening {L : CoinMap.Locate} {n : ℕ} (hL : OpensFrom L n)
    {log a b : List Verity.Release} (hab : log = a ++ b) (ha : ∀ r ∈ a, ∀ i s c, r.frame ≠ .opening i s c) :
    a <+: (CoinMap.before L n [] log).1 := by
  rcases CoinMap.before_split L (i := n) [] log with ⟨-, h1⟩ | ⟨rl, rest, j, hsplit, -, hj, hnj⟩
  · rw [h1, hab]
    exact List.prefix_append a b
  · obtain ⟨i, s, c, hr⟩ := hL _ _ _ _ hj hnj
    rcases List.append_eq_append_iff.1 (hab.symm.trans hsplit) with ⟨t, hB, -⟩ | ⟨t, ha', ht⟩
    · exact ⟨t, hB.symm⟩
    · cases t with
      | nil =>
        rw [List.append_nil] at ha'
        rw [ha']
      | cons y t' =>
        rw [List.cons_append] at ht
        have hy : rl = y := (List.cons.inj ht).1
        exact absurd hr (ha rl (by rw [ha', hy]; simp) i s c)

/-- The coins of a family that the start of a log shows start those the log shows. -/
theorem flatten_freshCoins_prefix (label : String) (after : ByteArray) {a l : List Verity.Release} (h : a <+: l) :
    (Verity.Sampled.freshCoins label after a).flatten <+: (Verity.Sampled.freshCoins label after l).flatten := by
  obtain ⟨t, rfl⟩ := h
  rw [freshCoins_append, List.flatten_append]
  exact List.prefix_append _ _

/-- Empty bytes start any bytes. -/
theorem empty_starts {a b : ByteArray} (h : a.size = 0) : ∃ u, b = a ++ u := by
  have hl : (Properties.Challenger.bitsOf a).length = 0 := by rw [CoinMap.length_bitsOf, h]
  refine Verity.Sampled.append_of_bitsOf_prefix ?_
  rw [List.length_eq_zero_iff.1 hl]
  exact List.nil_prefix

/-- **Two plays' bytes on the releases they share**: where the bits of each play's bytes start the coins of family
`ℓ` that the releases `B` show after the play's digest, `B`'s releases are on the second play's log, and that log
shows `ℓ`'s coins after one digest, one of the two plays' bytes starts the other's. Where `B` shows a coin of `ℓ`
after the first play's digest, the second play's digest is that one, so both read one string of coins; where it
shows none, the first play's bytes are empty. -/
theorem oneStarts_of_before {ℓ : String} {B log' : List Verity.Release} {after after' : ByteArray}
    {b b' : ByteArray} (hsub : ∀ x ∈ B, x ∈ log') (hone' : CoinMap.OneRoot ℓ after' log')
    (hpre : Properties.Challenger.bitsOf b <+: (Verity.Sampled.freshCoins ℓ after B).flatten)
    (hpre' : Properties.Challenger.bitsOf b' <+: (Verity.Sampled.freshCoins ℓ after' B).flatten) :
    OneStarts b b' := by
  by_cases hnil : Verity.Sampled.freshCoins ℓ after B = []
  · rw [hnil, List.flatten_nil] at hpre
    have hl : (Properties.Challenger.bitsOf b).length = 0 := by rw [List.prefix_nil.1 hpre, List.length_nil]
    rw [CoinMap.length_bitsOf] at hl
    exact Or.inl (empty_starts (by omega))
  · obtain ⟨x, hx, hxf, hxa⟩ := exists_fresh_of_coins hnil
    have haa : after = after' := hxa.symm.trans (hone' x (hsub x hx) hxf)
    rw [← haa] at hpre'
    rcases Nat.le_total (Properties.Challenger.bitsOf b).length (Properties.Challenger.bitsOf b').length with
      hle | hle
    · exact Or.inl (Verity.Sampled.append_of_bitsOf_prefix (List.prefix_of_prefix_length_le hpre hpre' hle))
    · exact Or.inr (Verity.Sampled.append_of_bitsOf_prefix (List.prefix_of_prefix_length_le hpre' hpre hle))

/-- **`SecondBytesOne`**, at a locator that puts only openings past the second stage's last round: each play's log
before the statements' round shows every coin of the second stage's sample its log shows (`freshCoins_before`, by
`sample_before_openings`), the two plays share those releases, and every coin of the second stage's family answers
the play's one digest (`CoinMap.StageDraws`). -/
theorem secondBytesOne_of {A : Live₂} {C : CaseOf} {Good : Verity.Exchange → Prop} {L : CoinMap.Locate}
    {ℓ₂ : String} (hL : OpensFrom L (A.h.stage + 1)) (hShow : SecondShown C Good)
    (hDraw : CoinMap.StageDraws (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind secondDraw) ℓ₂)
    (hRd : SessionCoins.InteriorReads C) (hlog : SessionCoins.InteriorLogged C Good) :
    SecondBytesOne A C Good L := by
  intro o o' k k' p e e' i i' it it' hA hG hA' hG' _ _ hb _ _ hi hit hi' hit'
  have hb' : CoinMap.before L (A.h.stage + 1) [] o.log = CoinMap.before L (A.h.stage + 1) [] o'.log := hb
  have hsub : ∀ x ∈ (CoinMap.before L (A.h.stage + 1) [] o.log).1, x ∈ o'.log := by
    intro x hx
    rw [hb'] at hx
    exact mem_of_before _ _ _ hx
  obtain ⟨label, after, hd, hpre⟩ := hShow o hA hG i it hi hit
  obtain ⟨label', after', hd', hpre'⟩ := hShow o' hA' hG' i' it' hi' hit'
  obtain ⟨hl, -⟩ := hDraw o ⟨hA, hG⟩ label after hd
  obtain ⟨hl', hone'⟩ := hDraw o' ⟨hA', hG'⟩ label' after' hd'
  rw [hl] at hd hpre
  rw [hl'] at hd' hpre'
  rw [freshCoins_before hL fun a r b hab i s c hr =>
    sample_before_openings C Good hRd hlog hA hG hd hab hr] at hpre
  rw [freshCoins_before hL fun a r b hab i s c hr =>
    sample_before_openings C Good hRd hlog hA' hG' hd' hab hr, ← hb'] at hpre'
  exact oneStarts_of_before hsub hone' hpre hpre'

/-- **The releases before every opening show the bytes the first stage read**: on a play the form accepts whose log
is the device's, the bits of the bytes the run's first stage read, where it read any, start the coins that the
log's releases before some point show released in the first sample's family after its digest, and none of those
releases is an opening. It is `FirstShown` with the coins it reads released before every opening. -/
def FirstShownEarly (C : CaseOf) (Good : Verity.Exchange → Prop) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ i read, inputOn C o = some i → FirstRead i read →
    0 < (i.source.extract 0 read).size →
    ∃ label after a b, (C o).bind firstDraw = some (label, after) ∧ o.log = a ++ b ∧
      (∀ r ∈ a, ∀ j s c, r.frame ≠ .opening j s c) ∧
      Properties.Challenger.bitsOf (i.source.extract 0 read) <+: (Verity.Sampled.freshCoins label after a).flatten

/-- **`FirstBytesOne`**, at a locator that puts only openings past the second stage's last round, from
`FirstShownEarly`: an accepted play's first stage read all its bytes (`sourceRead_holds`), the releases before
every opening start the log before the statements' round (`prefix_before_of_no_opening`), which the two plays
share, and every coin of the first stage's family answers the play's one digest (`CoinMap.StageDraws`). A play
whose first stage has no byte starts any other's. -/
theorem firstBytesOne_of {A : Live₂} {C : CaseOf} {Good : Verity.Exchange → Prop} {L : CoinMap.Locate}
    {ℓ₁ : String} (hL : OpensFrom L (A.h.stage + 1)) (hShow : FirstShownEarly C Good)
    (hDraw : CoinMap.StageDraws (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind firstDraw) ℓ₁) :
    FirstBytesOne A C Good L := by
  intro o o' k k' p e e' i i' hA hG hA' hG' _ _ hb _ _ hi hi'
  have hb' : CoinMap.before L (A.h.stage + 1) [] o.log = CoinMap.before L (A.h.stage + 1) [] o'.log := hb
  have hsub : ∀ x ∈ (CoinMap.before L (A.h.stage + 1) [] o.log).1, x ∈ o'.log := by
    intro x hx
    rw [hb'] at hx
    exact mem_of_before _ _ _ hx
  obtain ⟨read, hread⟩ := firstRead_of_accepts hA hi
  obtain ⟨read', hread'⟩ := firstRead_of_accepts hA' hi'
  have hsrc := extract_of_size_le (sourceRead_holds C o hA i read hi hread)
  have hsrc' := extract_of_size_le (sourceRead_holds C o' hA' i' read' hi' hread')
  by_cases h0 : 0 < i.source.size
  · by_cases h0' : 0 < i'.source.size
    · obtain ⟨label, after, a, b, hd, hab, ha, hpre⟩ :=
        hShow o hA hG i read hi hread (by rw [hsrc]; exact h0)
      obtain ⟨label', after', a', b', hd', hab', ha', hpre'⟩ :=
        hShow o' hA' hG' i' read' hi' hread' (by rw [hsrc']; exact h0')
      obtain ⟨hl, -⟩ := hDraw o ⟨hA, hG⟩ label after hd
      obtain ⟨hl', hone'⟩ := hDraw o' ⟨hA', hG'⟩ label' after' hd'
      rw [hl, hsrc] at hpre
      rw [hl', hsrc'] at hpre'
      have hp := hpre.trans (flatten_freshCoins_prefix ℓ₁ after (prefix_before_of_no_opening hL hab ha))
      have hp' := hpre'.trans (flatten_freshCoins_prefix ℓ₁ after' (prefix_before_of_no_opening hL hab' ha'))
      rw [← hb'] at hp'
      exact oneStarts_of_before hsub hone' hp hp'
    · exact Or.inr (empty_starts (by omega))
  · exact Or.inl (empty_starts (by omega))

/-- **`PlayCoinsOne` from what the log shows of the stages' bytes**, at a locator that puts only openings past the
second stage's last round: `playCoinsOne_of_bytes` at `firstBytesOne_of` and `secondBytesOne_of`. -/
theorem playCoinsOne_of_shown {A : Live₂} {C : CaseOf} {Good : Verity.Exchange → Prop} {L : CoinMap.Locate}
    {ℓ₁ ℓ₂ : String} (hL : OpensFrom L (A.h.stage + 1)) (hof : HeldOfAt A C) (hparts : PartsHeldAt A C)
    (hFirst : FirstShownEarly C Good)
    (hDraw₁ : CoinMap.StageDraws (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind firstDraw) ℓ₁)
    (hSecond : SecondShown C Good)
    (hDraw₂ : CoinMap.StageDraws (fun o => FormAccepts C o ∧ Good o) (fun o => (C o).bind secondDraw) ℓ₂)
    (hlog : SessionCoins.InteriorLogged C Good) : PlayCoinsOne A C Good L :=
  playCoinsOne_of_bytes hof hparts (firstBytesOne_of hL hFirst hDraw₁)
    (secondBytesOne_of hL hSecond hDraw₂ (SessionCoins.interiorReads C fun _ _ _ hh => held_size hh) hlog)

end Verity.SampledTwo
