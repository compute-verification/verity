module

public import Proofs.Properties.Verity.SampledTwo.RoundPrefix
public import Proofs.Properties.Verity.SampledTwo.AcceptsProof

/-! The two-stage audit's messages, fixed by digests the log before each round holds; `ShowsAt` and
`ExplainsOr` at round 1, the statements' round and a session's round are not proved here. -/

@[expose] public section

namespace Verity.SampledTwo.CoinMap

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

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

noncomputable def logBefore (L : CoinMap.Locate) (o : Verity.Exchange) (i : ℕ) :
    List Verity.Release × Option (Verity.Frame × ByteArray) :=
  CoinMap.before L i [] o.log

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

noncomputable abbrev frameLog (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) : Frame V A x θ F c σ :=
  frameBy V A x θ F c σ (fun ω r => CoinMap.place V θ F c σ L r ω) Rd (stringsFx Dg (logBefore L))
    (stringsOpn Acc Good raw (logBefore L))

noncomputable def orPair {κ : Type} (raw : Verity.Exchange → ℕ → Option (List Msg))
    (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (φ : List Msg × κ)
    (o o' : Verity.Exchange) (i : ℕ) : Option (Msg × Msg) :=
  match alt o o' i with
  | some p => some p
  | none => stringsPair raw φ o o' i

def AltCollides (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) : Prop :=
  ∀ o o' i p, alt o o' i = some p → p.1 ≠ p.2 ∧ Verity.sha512Msg p.1 = Verity.sha512Msg p.2

def ExplainsOr {κ : Type} {A : Live₂} (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ)
    (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (i : ℕ) : Prop :=
  ∀ o o', Acc o → Good o → Acc o' → Good o' → ctx o i = ctx o' i → raw o i = raw o' i →
    Rd.msg o i = Rd.msg o' i ∨ ∃ p, alt o o' i = some p

theorem explainsOr_of {κ : Type} {A : Live₂} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop}
    {raw : Verity.Exchange → ℕ → Option (List Msg)} {ctx : Verity.Exchange → ℕ → κ}
    (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) {i : ℕ}
    (h : Explains Rd Acc Good raw ctx i) : ExplainsOr Rd Acc Good raw ctx alt i :=
  fun o o' hA hG hA' hG' hc hr => Or.inl (h o o' hA hG hA' hG' hc hr)

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
noncomputable def blobsAlt (o o' : Verity.Exchange) (_ : ℕ) : Option (Msg × Msg) :=
  if h : ∃ p : Msg × Msg, (p.1 ≠ p.2 ∧ Verity.sha512Msg p.1 = Verity.sha512Msg p.2) ∧
      ∃ b ∈ o.sent.blobs, ∃ b' ∈ o'.sent.blobs, p = (b.data.toList, b'.data.toList) then some h.choose else none

theorem blobsAlt_collides : AltCollides blobsAlt := by
  intro o o' i p hp
  unfold blobsAlt at hp
  split at hp
  next h =>
    obtain rfl := Option.some.inj hp
    exact h.choose_spec.1
  next => cases hp

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

noncomputable def missLog (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (L : CoinMap.Locate) (Rd : Reading A)
    (Acc Good : Verity.Exchange → Prop) (Dg raw : Verity.Exchange → ℕ → Option (List Msg))
    (alt : Verity.Exchange → Verity.Exchange → ℕ → Option (Msg × Msg)) (q : ℕ) : ℝ≥0∞ :=
  ∑ i ∈ Finset.range (A.L.V.k x),
    ENNReal.ofReal (Real.sqrt ((finderBy V (A := A) (x := x) θ F c σ (fun ω r => CoinMap.place V θ F c σ L r ω) Rd
      (stringsFx Dg (logBefore L)) (stringsOpn Acc Good raw (logBefore L)) (orPair raw alt) i q).Pr
        Verity.sha512Msg))

def ShowsAt {A : Live₂} (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) (i : ℕ) : Prop :=
  ∀ o, Acc o → Good o → (∃ m, Rd.msg o i = some m) ∧
    ∃ ds bs, Dg o i = some ds ∧ raw o i = some bs ∧ bs.map Verity.sha512Msg = ds

theorem showsStrings_of_rounds {A : Live₂} {x : Msg} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop}
    {Dg raw : Verity.Exchange → ℕ → Option (List Msg)} (h : ∀ i < A.L.V.k x, ShowsAt Rd Acc Good Dg raw i) :
    ShowsStrings A x Rd Acc Good Dg raw :=
  fun o hA hG i hi => h i hi o hA hG

theorem showsAt_none {A : Live₂} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop}
    {Dg raw : Verity.Exchange → ℕ → Option (List Msg)} {i : ℕ} (m : Msg)
    (hm : ∀ o, Acc o → Good o → Rd.msg o i = some m) (hd : ∀ o, Acc o → Good o → Dg o i = some [])
    (hr : ∀ o, Acc o → Good o → raw o i = some []) : ShowsAt Rd Acc Good Dg raw i :=
  fun o hA hG => ⟨⟨m, hm o hA hG⟩, [], [], hd o hA hG, hr o hA hG, rfl⟩

theorem explains_const {κ : Type} {A : Live₂} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop}
    {raw : Verity.Exchange → ℕ → Option (List Msg)} {ctx : Verity.Exchange → ℕ → κ} {i : ℕ} (m : Msg)
    (hm : ∀ o, Acc o → Good o → Rd.msg o i = some m) : Explains Rd Acc Good raw ctx i :=
  fun o o' hA hG hA' hG' _ _ => (hm o hA hG).trans (hm o' hA' hG').symm

def endAnswers (p : List Verity.Release × Option (Verity.Frame × ByteArray)) : Option (List Msg) :=
  p.2.map fun fb => [fb.2.data.toList]

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

section Form

variable (A : Live₂) (C : CaseOf) (S : SessionReading) (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin)

theorem showsAt_form_zero (Acc Good : Verity.Exchange → Prop) (Dg raw : Verity.Exchange → ℕ → Option (List Msg))
    (hd : ∀ o, Acc o → Good o → Dg o 0 = some []) (hr : ∀ o, Acc o → Good o → raw o 0 = some []) :
    Messages.ShowsAt (formReading A C S dev) Acc Good Dg raw 0 :=
  Messages.showsAt_none [] (fun o _ _ => form_msg_zero A C S dev o) hd hr

theorem explains_form_zero {κ : Type} (Acc Good : Verity.Exchange → Prop)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) :
    Messages.Explains (formReading A C S dev) Acc Good raw ctx 0 :=
  Messages.explains_const [] fun o _ _ => form_msg_zero A C S dev o

theorem showsAt_form_mid (Acc Good : Verity.Exchange → Prop) (Dg raw : Verity.Exchange → ℕ → Option (List Msg))
    {n : ℕ} (h2 : 2 ≤ n) (h3 : n ≤ A.h.stage) (hd : ∀ o, Acc o → Good o → Dg o n = some [])
    (hr : ∀ o, Acc o → Good o → raw o n = some []) :
    Messages.ShowsAt (formReading A C S dev) Acc Good Dg raw n :=
  Messages.showsAt_none [] (fun o _ _ => form_msg_mid A C S dev o h2 h3) hd hr

theorem explains_form_mid {κ : Type} (Acc Good : Verity.Exchange → Prop)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) {n : ℕ} (h2 : 2 ≤ n)
    (h3 : n ≤ A.h.stage) : Messages.Explains (formReading A C S dev) Acc Good raw ctx n :=
  Messages.explains_const [] fun o _ _ => form_msg_mid A C S dev o h2 h3

end Form

end Verity.SampledTwo
