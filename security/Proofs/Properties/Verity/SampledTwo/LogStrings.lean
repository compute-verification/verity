module

public import Proofs.Properties.Verity.SampledTwo.RoundPrefix
public import Proofs.Properties.Verity.SampledTwo.AcceptsProof

/-!
# The two-stage audit's messages, fixed by digests the log before each round holds

`MessagesOpen` bounds the messages from three facts about the digests a play fixes for a round (`Dg`), the strings it
shows for them (`raw`) and a context (`ctx`): `ShowsStrings`, `Explains` and `FixedBefore`. Here the context is the
play's log before the round (`CoinMap.before`, `Messages.logBefore`), and the digests are a reading of that log and
of the tape. `FixedBefore` then holds at every round (`Messages.fixedBefore_log`): the log before a round is the same
on any play at coins the history before the round shows (`Messages.before_shown`). So the bound rests on
`ShowsStrings` and `Explains`, round by round (`Messages.reachedLog`, `Messages.messagesFixedLog`).

A string the device's log holds no digest of, a file the check reads under a digest the parameters name, is not among
the strings. Two plays that read two such files sent two blobs with one SHA-512, and the finder outputs those: the
round's message is a function of the strings and the context, or the two plays give a pair of their own
(`Messages.ExplainsOr`, `Messages.orPair`, `Messages.breaks_or`). `Messages.blobsAlt` is that pair for the blobs sent.

* `CoinMap.before_split`: a log is its log before round `i`, then, where there is one, the first release the locator
  puts on a round at least `i`, then the rest.
* `Messages.ShowsAt`, `Messages.showsStrings_of_rounds`: `ShowsStrings` round by round.
* `Messages.showsAt_none`, `Messages.explains_const`: a round whose message is the same on every accepted play needs
  no digest and no string.
* `Messages.blobsAlt`, `Messages.blobsAlt_collides`, `Messages.blobsAlt_of`: two blobs two plays sent, different, with
  one SHA-512.
* `Messages.endAnswers`: the reading that gives, as one digest, the bytes the release that ends the log before the
  round answers.
* `showsAt_form_zero`, `explains_form_zero`, `showsAt_form_mid`, `explains_form_mid`: at the form's reading, round 0
  and rounds 2 to `stage` read the empty message.

What this does not prove: `ShowsAt` and `ExplainsOr` at round 1, at the statements' round or at a session's round of
the form's reading.
-/

@[expose] public section

namespace Verity.SampledTwo.CoinMap

/-- **A log is its log before round `i`, then the release that ends it, then the rest**: either the log before round
`i` is the whole log and ends at no release, or the log is the log before round `i`, then a release the locator puts
on a round at least `i`, whose frame and bytes the log before round `i` ends at, then the rest. -/
theorem before_split (L : Locate) {i : ℕ} : ∀ (pre log : List Verity.Release),
    ((before L i pre log).2 = none ∧ (before L i pre log).1 = log) ∨
      ∃ rl rest j, log = (before L i pre log).1 ++ rl :: rest ∧
        (before L i pre log).2 = some (rl.frame, rl.answers) ∧
        locAt L (pre ++ (before L i pre log).1) rl.frame rl.answers = some j ∧ i ≤ j
  | pre, [] => Or.inl (by simp [before])
  | pre, rl :: log => by
    by_cases h : ∃ j, locAt L pre rl.frame rl.answers = some j ∧ i ≤ j
    · obtain ⟨j, hj, hij⟩ := h
      have e := before_cons_pos L (i := i) (pre := pre) (rl := rl) log ⟨j, hj, hij⟩
      refine Or.inr ⟨rl, log, j, ?_, ?_, ?_, hij⟩
      · rw [e]
        rfl
      · rw [e]
      · rw [e, List.append_nil]
        exact hj
    · have e := before_cons_neg L (i := i) (pre := pre) (rl := rl) log h
      rcases before_split L (i := i) (pre ++ [rl]) log with ⟨h2, h1⟩ | ⟨rl', rest, j, hlog, h2, hj, hij⟩
      · refine Or.inl ⟨?_, ?_⟩
        · rw [e]
          exact h2
        · rw [e]
          exact congrArg (fun l => rl :: l) h1
      · refine Or.inr ⟨rl', rest, j, ?_, ?_, ?_, hij⟩
        · rw [e]
          exact congrArg (fun l => rl :: l) hlog
        · rw [e]
          exact h2
        · rw [e]
          rw [List.append_assoc] at hj
          exact hj

end Verity.SampledTwo.CoinMap

namespace Verity.SampledTwo.Messages

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The log before the round as the context -/

/-- **The context of round `i`: the play's log before the round**, at the locator `L`. -/
noncomputable def logBefore (L : CoinMap.Locate) (o : Verity.Exchange) (i : ℕ) :
    List Verity.Release × Option (Verity.Frame × ByteArray) :=
  CoinMap.before L i [] o.log

/-- **`FixedBefore` at digests read off the log before the round, with that log as the context**: at a frame whose
coin map is the locator `L`'s, where the digests a play of the frame fixes for round `i` are a reading `g` of its tape
and its log before round `i`. Nothing is asked of the strings. -/
theorem fixedBefore_log (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Fr : Frame V A x θ F c σ)
    (hFr : ∀ ω r, Fr.place ω r = CoinMap.place V θ F c σ L r ω) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L)
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) (i : ℕ)
    (g : TapeOf V θ F c σ → List Verity.Release × Option (Verity.Frame × ByteArray) → Option (List Msg))
    (hDg : ∀ ω r₀, Dg (Fr.out ω r₀) i = g ω (CoinMap.before L i [] (Fr.out ω r₀).log)) :
    FixedBefore Fr Rd (stringsFx Dg (logBefore L)) (stringsOpn Acc Good raw (logBefore L)) i :=
  fixedBefore_strings_before V A x θ F c σ L Fr hFr Rd Acc Good hShow Dg raw (logBefore L) i g hDg
    fun ω r r' ha hg _ _ hs =>
      Option.some.inj (fixed_of_before V A x θ F c σ L Fr hFr Rd Acc Good hShow
        (fun o j => some (logBefore L o j)) i some ω (fun _ => rfl) r r' ha hg hs)

/-- **The frame at the locator `L`'s coin map**, whose prover is the one of the digests `Dg` and the strings `raw`,
with the log before each round as the context. -/
noncomputable abbrev frameLog (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) : Frame V A x θ F c σ :=
  frameBy V A x θ F c σ (fun ω r => CoinMap.place V θ F c σ L r ω) Rd (stringsFx Dg (logBefore L))
    (stringsOpn Acc Good raw (logBefore L))

/-! ## A pair of the two plays' own -/

/-- **The pair two plays give for round `i`**: the one `alt` reads off the two plays where there is one, else two of
the strings they show at one place (`stringsPair`). -/
noncomputable def orPair {κ : Type} (raw : Verity.Exchange → ℕ → Option (List Msg))
    (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (φ : List Msg × κ)
    (o o' : Verity.Exchange) (i : ℕ) : Option (Msg × Msg) :=
  match alt o o' i with
  | some p => some p
  | none => stringsPair raw φ o o' i

/-- **Every pair `alt` gives is a SHA-512 collision.** -/
def AltCollides (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) : Prop :=
  ∀ o o' i p, alt o o' i = some p → p.1 ≠ p.2 ∧ Verity.sha512Msg p.1 = Verity.sha512Msg p.2

/-- **The round's message is a function of the strings and the context, or the two plays give a pair**: two plays the
check accepts whose logs are the device's, with one context and the same strings for round `i`, read the same message
at round `i`, or `alt` gives a pair at them. -/
def ExplainsOr {κ : Type} {A : Live₂} (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ)
    (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (i : ℕ) : Prop :=
  ∀ o o', Acc o → Good o → Acc o' → Good o' → ctx o i = ctx o' i → raw o i = raw o' i →
    Rd.msg o i = Rd.msg o' i ∨ ∃ p, alt o o' i = some p

/-- **`ExplainsOr` from `Explains`**, at any `alt`. -/
theorem explainsOr_of {κ : Type} {A : Live₂} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop}
    {raw : Verity.Exchange → ℕ → Option (List Msg)} {ctx : Verity.Exchange → ℕ → κ}
    (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) {i : ℕ}
    (h : Explains Rd Acc Good raw ctx i) : ExplainsOr Rd Acc Good raw ctx alt i :=
  fun o o' hA hG hA' hG' hc hr => Or.inl (h o o' hA hG hA' hG' hc hr)

/-- **Two accepted plays with two messages give a SHA-512 collision**, where the message is a function of the strings
and the context or the two plays give a pair, and every pair `alt` gives is a collision: `alt`'s pair, or two of the
strings the plays show under one digest. -/
theorem breaks_or {κ : Type} {A : Live₂} (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ)
    (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (i : ℕ) (halt : AltCollides alt)
    (hE : ExplainsOr Rd Acc Good raw ctx alt i) :
    Breaks Rd (stringsOpn Acc Good raw ctx) (orPair raw alt) i := by
  classical
  intro φ o o' m m' ho ho' hm hm' hne
  cases ha : alt o o' i with
  | some p =>
    refine ⟨p, ?_, (halt o o' i p ha).1, (halt o o' i p ha).2⟩
    simp only [orPair, ha]
  | none =>
    obtain ⟨hA, hG, hc, bs, hbs, hd⟩ := ho
    obtain ⟨hA', hG', hc', bs', hbs', hd'⟩ := ho'
    have hraw : bs ≠ bs' := by
      intro e
      rcases hE o o' hA hG hA' hG' (hc.trans hc'.symm) (by rw [hbs, hbs', e]) with h | ⟨p, hp⟩
      · rw [hm, hm'] at h
        exact hne (Option.some.inj h)
      · rw [ha] at hp
        cases hp
    obtain ⟨p, hp, hp1, hp2⟩ := exists_zip_collide Verity.sha512Msg bs bs' (hd.trans hd'.symm) hraw
    have hex : ∃ p : Msg × Msg, (p.1 ≠ p.2 ∧ Verity.sha512Msg p.1 = Verity.sha512Msg p.2) ∧
        ∃ bs bs', raw o i = some bs ∧ raw o' i = some bs' ∧ p ∈ bs.zip bs' :=
      ⟨p, ⟨hp1, hp2⟩, bs, bs', hbs, hbs', hp⟩
    have hs : stringsPair raw φ o o' i = some hex.choose := by
      unfold stringsPair
      split
      next => rfl
      next h => exact absurd hex h
    refine ⟨hex.choose, ?_, hex.choose_spec.1.1, hex.choose_spec.1.2⟩
    simp only [orPair, ha, hs]

open Classical in
/-- **Two blobs two plays sent, different, with one SHA-512**, if there are any. -/
noncomputable def blobsAlt (o o' : Verity.Exchange) (_ : ℕ) : Option (Msg × Msg) :=
  if h : ∃ p : Msg × Msg, (p.1 ≠ p.2 ∧ Verity.sha512Msg p.1 = Verity.sha512Msg p.2) ∧
      ∃ b ∈ o.sent.blobs, ∃ b' ∈ o'.sent.blobs, p = (b.data.toList, b'.data.toList) then some h.choose else none

/-- **The blobs' pair is a SHA-512 collision.** -/
theorem blobsAlt_collides : AltCollides blobsAlt := by
  intro o o' i p hp
  unfold blobsAlt at hp
  split at hp
  next h =>
    obtain rfl := Option.some.inj hp
    exact h.choose_spec.1
  next => cases hp

/-- **Two plays that sent two blobs, different, with one SHA-512 give the blobs' pair.** -/
theorem blobsAlt_of {o o' : Verity.Exchange} {b b' : ByteArray} (hb : b ∈ o.sent.blobs) (hb' : b' ∈ o'.sent.blobs)
    (hne : b ≠ b') (hh : Flock.Sha512.hash b = Flock.Sha512.hash b') (i : ℕ) : ∃ p, blobsAlt o o' i = some p := by
  have hex : ∃ p : Msg × Msg, (p.1 ≠ p.2 ∧ Verity.sha512Msg p.1 = Verity.sha512Msg p.2) ∧
      ∃ b ∈ o.sent.blobs, ∃ b' ∈ o'.sent.blobs, p = (b.data.toList, b'.data.toList) := by
    refine ⟨(b.data.toList, b'.data.toList), ⟨fun e => hne (ByteArray.ext (Array.ext' e)), ?_⟩, b, hb, b', hb', rfl⟩
    show Verity.sha512Msg b.data.toList = Verity.sha512Msg b'.data.toList
    rw [Verity.Sampled.sha512Msg_bytes, Verity.Sampled.sha512Msg_bytes, hh]
  refine ⟨hex.choose, ?_⟩
  unfold blobsAlt
  split
  next => rfl
  next h => exact absurd hex h

/-- **The messages' term at the log before each round**: the sum over the rounds of the square roots of the successes
of the finders that output `alt`'s pair at two plays, or two of the strings the two plays show at one place, each
finder at `q` SHA-512 calls. -/
noncomputable def missLog (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (Dg raw : Verity.Exchange → ℕ → Option (List Msg))
    (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (q : ℕ) : ℝ≥0∞ :=
  ∑ i ∈ Finset.range (A.L.V.k x),
    ENNReal.ofReal (Real.sqrt ((finderBy V (A := A) (x := x) θ F c σ (fun ω r => CoinMap.place V θ F c σ L r ω) Rd
      (stringsFx Dg (logBefore L)) (stringsOpn Acc Good raw (logBefore L)) (orPair raw alt) i q).Pr
        Verity.sha512Msg))

/-! ## `ShowsStrings`, round by round -/

/-- **An accepted play shows, for round `i`, strings that have its digests**: `ShowsStrings` at one round. -/
def ShowsAt {A : Live₂} (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) (i : ℕ) : Prop :=
  ∀ o, Acc o → Good o → (∃ m, Rd.msg o i = some m) ∧
    ∃ ds bs, Dg o i = some ds ∧ raw o i = some bs ∧ bs.map Verity.sha512Msg = ds

/-- **`ShowsStrings` from its rounds.** -/
theorem showsStrings_of_rounds {A : Live₂} {x : Msg} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop}
    {Dg raw : Verity.Exchange → ℕ → Option (List Msg)} (h : ∀ i < A.L.V.k x, ShowsAt Rd Acc Good Dg raw i) :
    ShowsStrings A x Rd Acc Good Dg raw :=
  fun o hA hG i hi => h i hi o hA hG

/-- **A round that reads one message on every accepted play shows no string**: with no digest and no string for the
round, `ShowsAt` holds. -/
theorem showsAt_none {A : Live₂} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop}
    {Dg raw : Verity.Exchange → ℕ → Option (List Msg)} {i : ℕ} (m : Msg)
    (hm : ∀ o, Acc o → Good o → Rd.msg o i = some m) (hd : ∀ o, Acc o → Good o → Dg o i = some [])
    (hr : ∀ o, Acc o → Good o → raw o i = some []) : ShowsAt Rd Acc Good Dg raw i :=
  fun o hA hG => ⟨⟨m, hm o hA hG⟩, [], [], hd o hA hG, hr o hA hG, rfl⟩

/-- **A round that reads one message on every accepted play is explained by any strings and context.** -/
theorem explains_const {κ : Type} {A : Live₂} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop}
    {raw : Verity.Exchange → ℕ → Option (List Msg)} {ctx : Verity.Exchange → ℕ → κ} {i : ℕ} (m : Msg)
    (hm : ∀ o, Acc o → Good o → Rd.msg o i = some m) : Explains Rd Acc Good raw ctx i :=
  fun o o' hA hG hA' hG' _ _ => (hm o hA hG).trans (hm o' hA' hG').symm

/-- **The digest the log before a round ends at**: the bytes the release that ends the log before the round answers,
as one digest; none where no release ends it. -/
def endAnswers (p : List Verity.Release × Option (Verity.Frame × ByteArray)) : Option (List Msg) :=
  p.2.map fun fb => [fb.2.data.toList]

/-! ## The bound from `ShowsAt` and `Explains` -/

/-- **The messages at the rounds the run reaches, at digests read off the log before each round**: where each round a
release is put on answers with its coin (`ShowsPlaced`), the digests a play of the frame fixes for a round are a
reading of its tape and its log before the round, an accepted play shows strings that have them (`ShowsAt`), and the
round's message is a function of the strings and the log before the round, or the two plays give a pair that is a
collision (`ExplainsOr`, `AltCollides`). -/
theorem reachedLog (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L)
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg))
    (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (q : ℕ)
    (g : ℕ → TapeOf V θ F c σ → List Verity.Release × Option (Verity.Frame × ByteArray) → Option (List Msg))
    (hDg : ∀ i < A.L.V.k x, ∀ ω r₀, Dg ((frameLog V A x θ F c σ L Rd Acc Good Dg raw).out ω r₀) i =
      g i ω (CoinMap.before L i [] ((frameLog V A x θ F c σ L Rd Acc Good Dg raw).out ω r₀).log))
    (hO : ∀ i < A.L.V.k x, ShowsAt Rd Acc Good Dg raw i) (halt : AltCollides alt)
    (hE : ∀ i < A.L.V.k x, ExplainsOr Rd Acc Good raw (logBefore L) alt i) :
    MessagesReached V A x θ F c σ (frameLog V A x θ F c σ L Rd Acc Good Dg raw) Rd Acc Good
      (missLog V A x θ F c σ L Rd Acc Good Dg raw alt q) :=
  reachedBy V A x θ F c σ (fun ω r => CoinMap.place V θ F c σ L r ω) Rd Acc Good (stringsFx Dg (logBefore L))
    (stringsOpn Acc Good raw (logBefore L)) (orPair raw alt) q
    (opens_strings (logBefore L) (showsStrings_of_rounds hO))
    (fun i hi => fixedBefore_log V A x θ F c σ L (frameLog V A x θ F c σ L Rd Acc Good Dg raw) (fun _ _ => rfl) Rd
      Acc Good hShow Dg raw i (g i) (hDg i hi))
    (fun i hi => breaks_or Rd Acc Good raw (logBefore L) alt i halt (hE i hi))

/-- **The messages' bound at digests read off the log before each round**: `reachedLog`, where the run reaches every
round (`RunsThrough`). -/
theorem messagesFixedLog (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (hShow : ShowsPlaced A x Rd Acc Good L)
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg))
    (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (q : ℕ)
    (g : ℕ → TapeOf V θ F c σ → List Verity.Release × Option (Verity.Frame × ByteArray) → Option (List Msg))
    (hDg : ∀ i < A.L.V.k x, ∀ ω r₀, Dg ((frameLog V A x θ F c σ L Rd Acc Good Dg raw).out ω r₀) i =
      g i ω (CoinMap.before L i [] ((frameLog V A x θ F c σ L Rd Acc Good Dg raw).out ω r₀).log))
    (hO : ∀ i < A.L.V.k x, ShowsAt Rd Acc Good Dg raw i) (halt : AltCollides alt)
    (hE : ∀ i < A.L.V.k x, ExplainsOr Rd Acc Good raw (logBefore L) alt i)
    (hT : RunsThrough (frameLog V A x θ F c σ L Rd Acc Good Dg raw) Rd Acc Good) :
    MessagesFixed V A x θ F c σ (frameLog V A x θ F c σ L Rd Acc Good Dg raw) Rd Acc Good
      (missLog V A x θ F c σ L Rd Acc Good Dg raw alt q) :=
  messagesFixed_of_reached V A x θ F c σ _ Rd Acc Good _ hT
    (reachedLog V A x θ F c σ L Rd Acc Good hShow Dg raw alt q g hDg hO halt hE)

end Verity.SampledTwo.Messages

namespace Verity.SampledTwo

open Claims

/-! ## The form's reading at the rounds that read the empty message -/

section Form

variable (A : Live₂) (C : CaseOf) (S : SessionReading) (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin)

/-- **Round 0 of the form's reading shows no string.** -/
theorem showsAt_form_zero (Acc Good : Verity.Exchange → Prop) (Dg raw : Verity.Exchange → ℕ → Option (List Msg))
    (hd : ∀ o, Acc o → Good o → Dg o 0 = some []) (hr : ∀ o, Acc o → Good o → raw o 0 = some []) :
    Messages.ShowsAt (formReading A C S dev) Acc Good Dg raw 0 :=
  Messages.showsAt_none [] (fun o _ _ => form_msg_zero A C S dev o) hd hr

/-- **Round 0 of the form's reading is explained by any strings and context.** -/
theorem explains_form_zero {κ : Type} (Acc Good : Verity.Exchange → Prop)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) :
    Messages.Explains (formReading A C S dev) Acc Good raw ctx 0 :=
  Messages.explains_const [] fun o _ _ => form_msg_zero A C S dev o

/-- **Rounds 2 to `stage` of the form's reading show no string.** -/
theorem showsAt_form_mid (Acc Good : Verity.Exchange → Prop) (Dg raw : Verity.Exchange → ℕ → Option (List Msg))
    {n : ℕ} (h2 : 2 ≤ n) (h3 : n ≤ A.h.stage) (hd : ∀ o, Acc o → Good o → Dg o n = some [])
    (hr : ∀ o, Acc o → Good o → raw o n = some []) :
    Messages.ShowsAt (formReading A C S dev) Acc Good Dg raw n :=
  Messages.showsAt_none [] (fun o _ _ => form_msg_mid A C S dev o h2 h3) hd hr

/-- **Rounds 2 to `stage` of the form's reading are explained by any strings and context.** -/
theorem explains_form_mid {κ : Type} (Acc Good : Verity.Exchange → Prop)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) {n : ℕ} (h2 : 2 ≤ n)
    (h3 : n ≤ A.h.stage) : Messages.Explains (formReading A C S dev) Acc Good raw ctx n :=
  Messages.explains_const [] fun o _ _ => form_msg_mid A C S dev o h2 h3

end Form

end Verity.SampledTwo
