module

public import Proofs.Properties.Verity.SampledTwo.FormDraw
public import Proofs.Properties.Verity.SampledTwo.CoinBytes
public import Proofs.Properties.Verity.SampledTwo.Fits
public import Proofs.Properties.Verity.SampledTwo.Within

/-!
# One pair of stage coins, from the stages' bytes

`PlayCoinsOne` (`FormDraw.lean`) asks, of two accepted plays with one parameters and one log before the statements'
round, for one round-0 coin and one second stage's coins that carry the bytes each play's samples read
(`SpotFlock.RunsAs.PlayCoins`). Here it is a theorem where, stage by stage, one of the two plays' bytes starts the
other's (`OneStarts`): the first stage's (`FirstBytesOne`) and the second's (`SecondBytesOne`).

* An accepted play's first-stage bytes are within one coin's and its second-stage bytes within `stage` coins'
  (`sourcesFit`), and any bytes within a coin's are a coin's first bytes (`coinsOfBytes`, `stagesOfBytes`). So where
  one of two byte strings starts the other, one coin's bytes start with both (`coin_of_oneStarts`), and one second
  stage's coins' do (`stage_of_oneStarts`).
* A sample reads no byte past its source (`readsWithin`, at a held copy with no closure map, which an accepted
  play's is: `closure_of_held`, `closure_of_formAccepts`), so coins whose bytes start with a play's bytes carry what
  its samples read (`playCoins_of_starts`).

**Not established here**: `FirstBytesOne` and `SecondBytesOne`, each a hypothesis. The check reads each stage's bytes
off the play's log, as the start of the coins the log shows released in the stage's family after its record's digest
(`FirstShown`, `SecondShown`). Two plays with one log before the statements' round share the releases before the
first round a session opens, and no more, so each hypothesis needs its stage's coins to be released before that round
is opened. For the second stage's, that is what the form's check holds the log it read to
(`SessionCoins.readLinkFirst`). For the first stage's, no lemma here gives it.
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

end Verity.SampledTwo
