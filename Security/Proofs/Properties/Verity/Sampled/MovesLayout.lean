module

public import Proofs.Properties.Verity.Sampled.RootB

/-!
# `MovesReach` at the session's layout

At the readings and the decoders at the session's layout (`Live.layout`, `Live.tables`) and the sample-coupled coin map
under `fieldEnc`, every clause of `MovesOf` holds at one witness of the joint law's support off the link's and the
rounds' empty classes (`ReachesOff`): one widened sample, one set of link points and one set of rep rounds, since the
three sit together inside the writer's output.

* **`MovesOf` at a witness** (`ReachesOff.movesOf`): the rep rounds (`firstStop_reps_verdict`) fix the outcome as the
  writer's; the trace's first stop fixes a cut (`firstStop_fixes`), the link points are released after a log `L₀` in
  order (`firstStop_order`), and that release is the log's first opening (`firstOpened_link`). Then the record decodes
  to the session's draw file at the sample (`recordDraw_firstStop`), the roots decoded at the first stop are the
  record's `root_B` (`firstStop_rootB`), the link points are the record's (`link_pts`), and the proof files are v2's
  (`proofs_of_sent`).
* **`MovesReach`** (`movesReach_layout`): `ReachesOff.movesOf` at every outcome, with what it takes of each accepted
  play whose log v2 read is the device's stated for every tape.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open SampledProofs.Challenger
open FlockSoundness.Discharge.Tables (LiveOutJ)

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **`MovesOf` at a witness of the joint law's support, at the session's layout**: at the readings and the decoders
at `Live.layout` and `Live.tables`, with a round and at most the trace's `N` rounds but one, and at the sample-coupled
coin map under `fieldEnc`, an outcome off the link's and the rounds' empty classes, beside a play that the sampled
verdict accepts, that misses nowhere and whose log v2's check read is the device's, is the writer's at the widened
sample, the decoded registration and an outcome whose moves are those v2's files decode to (`MovesOf`). It takes, for
the record's sample (`recordDraw_firstStop`), that v2 reads the sample's coins of the family `label` answering
`after` (`hN`) and that every release of that family precedes every opening (`hP`); and, for the rep rounds
(`firstStop_reps_verdict`), of the device's log that its first opening is the session stream's, that the layout has a
slot for each opening of a rep stream and that it opens only the record's rounds, and of v2's read and the record the
statement `hans`. That the log has an opening and that its trees name their streams distinctly come from the verdict
(`verdict_opens`, `verdict_nodup`). The roots (`firstStop_rootB`) and the link points (`link_pts`) read the same first
opening. -/
theorem ReachesOff.movesOf {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {A : Live} (hA : SetsUp V θ F A) (hDg : DigestsAt V θ) (h0r : 0 < A.rounds)
    (hRN : A.rounds ≤ A.N + 1) (ht : PopTied θ A) (hR : DrawReadAll k coinsRead) (hH : DrawHeld k)
    (hfix : ∀ S, Flock.norm (A.dj S) = A.dj S) (reg : Option Claims.Msg → A.Reg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {rd : Reading A.rounds}
    (hrd : rd = readingAt A.tables A.layout record recordDigest) {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} (hD : D = decodersAt reg A.layout) {label : String} {after : ByteArray} {m : ℕ}
    {κ : A.Ω → Fin m → Fin 512 → Bool} (hm : _root_.SampledProofs.Challenger.MAX_COINS ≤ m)
    (hsrc : ∀ ω, _root_.Properties.SampledProofs.bytesOf (A.src ω 0) =
      _root_.Properties.SampledProofs.bytesOf (coinBytes m 64 (κ ω) 0))
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} (hM : M = sampleMap fieldEnc V θ F c A rd σ label after κ)
    {o : A.Out × CoinTape (V.game θ F c) σ}
    (h : ReachesOff V θ F c A rd σ D M (linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1))
      (roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) o)
    (hv : V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2)) (hn : NoMiss V θ F c rd σ o.2)
    (hl : Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ o.2))
    (hN : SampleNamed k (CoinTape.run (V.game θ F c) σ o.2).sent label after)
    (hP : ∀ pre r post, (CoinTape.run (V.game θ F c) σ o.2).log = pre ++ r :: post → ∀ i s n,
      r.frame = .opening i s n → ∀ q ∈ post, ¬ (q.frame = .fresh label ∧ q.answers = after))
    (hfirst : ∀ pre r post, (CoinTape.run (V.game θ F c) σ o.2).log = pre ++ r :: post → ∀ i s n,
      r.frame = .opening i s n → (∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') →
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION)
    (hcovL : ∀ i s (r : Fin 2),
      ((helloSpecs (CoinTape.run (V.game θ F c) σ o.2).log)[i]?.bind fun sp => sp.streams[s]?) =
        A.tables[0]?.map (repStreamName · r.val) →
      ∀ k < openedOf (CoinTape.run (V.game θ F c) σ o.2).log i s, ∃ j, A.layout j = .rep 0 r k)
    (hO : ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some text →
      Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ o.2).log)
    (hans : ∀ {label : String} {after : ByteArray} {J : Json} {ξ : ByteArray} {lines : List String}
      {pre post : List Release} {y : Release} {trees : Array LoggedTree} {i s n : ℕ} {name : String}
      {spec : Flock.SessionSpec} {record : String} {sess : Flock.RunRecord} {st : Flock.StreamRec},
      readLog label after [J] ξ lines = some (pre ++ y :: post) → coinTreesOf lines = .ok trees →
      trees.size = 1 → ∀ {tabsB : Array (String × Option ℕ)}, sessionTreeReason trees tabsB J = none →
      y.frame = .opening i s n → ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name →
      (name == TREE_SESSION) = false →
      Flock.Record.decode spec record = .ok sess → Flock.JsonParse.parse record = .ok J →
      spec.sha512Digests = true → sess.streams.find? (·.name == name) = some st →
      openedOf pre i s < st.rounds.size → y.answers = st.rounds[openedOf pre i s]!.sha) :
    ∃ (S : Finset (Fin A.n)) (R : A.Reg) (out : LiveOutJ (A.J S R) A.mP), o.1 = (R, S, A.wr S R out) ∧
      MovesOf A (CoinTape.run (V.game θ F c) σ o.2).sent S R out := by
  subst hrd hD hM
  obtain ⟨ω, pts, hL0, hr⟩ := h
  obtain ⟨reps, ho, hreps⟩ := firstStop_reps_verdict hA hDg h0r hRN reg record recordDigest rfl rfl rfl ω pts hL0 hr
    hv hn hl hfirst hcovL hO hans
  obtain ⟨_, _, _, _, hrec₀, -⟩ := accepts_sent hA (play_ends V θ F c σ o.2).1 hv (filesOn_play hA hDg hv)
  have hop := verdict_opens hA (play_ends V θ F c σ o.2).1 hv hl hrec₀
  have hnd := verdict_nodup hA (play_ends V θ F c σ o.2).1 hv hl hrec₀
  have hjl := A.layout_link h0r
  have hK := firstStop_fixes V θ F c A (readingAt A.tables A.layout record recordDigest) σ
    (sampleMap fieldEnc V θ F c A (readingAt A.tables A.layout record recordDigest) σ label after κ)
    (readingAt_picks A.tables A.layout record recordDigest) A.tables ⟨0, h0r⟩
    (fun pre fr h => by
      show slotRound A.tables (A.layout ⟨0, h0r⟩) pre fr
      rw [hjl]
      exact h) ω _ hr hop
  obtain ⟨L₀, as, ms, ⟨hLA, hc⟩, -, -, -, hgt, -, hcov⟩ := firstStop_order V θ F c A A.layout_injective
    A.tables_nodup (hA.layout_lt _ _) record recordDigest reg rfl rfl σ
    (sampleMap fieldEnc V θ F c A (readingAt A.tables A.layout record recordDigest) σ label after κ)
    (linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) (B := roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1))
    (linkBits fieldEnc) (fun _ _ => rfl) (fun _ _ h => h) (fun _ x => roundBits fieldEnc x) (fun _ _ _ => rfl)
    (fun _ _ _ h => h) ω pts hK hL0 hr rfl (picksOnce_readingAt A.tables A.layout record recordDigest hnd) hRN
  obtain ⟨_, i, s₀, n₀, b₀, L', -, -, -, hL₀⟩ := id hLA
  have h₀ := firstOpened_link (logServed_run V θ F c σ o.2) hjl hcov hgt hL₀ rfl
  have hno := noOpening_of_firstOpened h₀
  refine ⟨_, _, _, ho, fun text proofs spec sts sess hrec hpr hs hd =>
    ⟨?_, ?_, ?_, hreps text proofs spec sts sess hrec hpr hs hd, proofs_of_sent reg A.layout hpr⟩⟩
  · rw [sample_extend]
    exact recordDraw_firstStop fieldEnc hA ht hR hH (readingAt_picks A.tables A.layout record recordDigest) hm hsrc
      hfix ω (firstStop_playReaches V θ F c A _ σ _ ω hr) hv hl hN hP hrec
  · exact firstStop_rootB hA rfl rfl hr hn hv hl hfirst hLA hc h₀ hrec hs hd
  · exact (link_pts hA (play_ends V θ F c σ o.2).1 hv hl hL₀ rfl hno (hfirst L₀ _ L' hL₀ i s₀ n₀ rfl hno) rfl hrec
      hs hd).symm

/-- **`MovesReach` at the session's layout** (`couplingTarget_assembled`'s `hM` at these readings, decoders and coin
map): `ReachesOff.movesOf` at every outcome, where what it takes of a play holds of every tape whose play the sampled
verdict accepts and whose log v2's check read is the device's. -/
theorem movesReach_layout {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {A : Live} (hA : SetsUp V θ F A) (hDg : DigestsAt V θ) (h0r : 0 < A.rounds)
    (hRN : A.rounds ≤ A.N + 1) (ht : PopTied θ A) (hR : DrawReadAll k coinsRead) (hH : DrawHeld k)
    (hfix : ∀ S, Flock.norm (A.dj S) = A.dj S) (reg : Option Claims.Msg → A.Reg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {rd : Reading A.rounds}
    (hrd : rd = readingAt A.tables A.layout record recordDigest) {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} (hD : D = decodersAt reg A.layout) {label : String} {after : ByteArray} {m : ℕ}
    {κ : A.Ω → Fin m → Fin 512 → Bool} (hm : _root_.SampledProofs.Challenger.MAX_COINS ≤ m)
    (hsrc : ∀ ω, _root_.Properties.SampledProofs.bytesOf (A.src ω 0) =
      _root_.Properties.SampledProofs.bytesOf (coinBytes m 64 (κ ω) 0))
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} (hM : M = sampleMap fieldEnc V θ F c A rd σ label after κ)
    (hN : ∀ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      SampleNamed k (CoinTape.run (V.game θ F c) σ x).sent label after)
    (hP : ∀ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      ∀ pre r post, (CoinTape.run (V.game θ F c) σ x).log = pre ++ r :: post → ∀ i s n,
        r.frame = .opening i s n → ∀ q ∈ post, ¬ (q.frame = .fresh label ∧ q.answers = after))
    (hfirst : ∀ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      ∀ pre r post, (CoinTape.run (V.game θ F c) σ x).log = pre ++ r :: post → ∀ i s n,
        r.frame = .opening i s n → (∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') →
          ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION)
    (hcovL : ∀ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) → ∀ i s (r : Fin 2),
        ((helloSpecs (CoinTape.run (V.game θ F c) σ x).log)[i]?.bind fun sp => sp.streams[s]?) =
          A.tables[0]?.map (repStreamName · r.val) →
        ∀ k < openedOf (CoinTape.run (V.game θ F c) σ x).log i s, ∃ j, A.layout j = .rep 0 r k)
    (hO : ∀ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ x).sent = some text →
        Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ x).log)
    (hans : ∀ {label : String} {after : ByteArray} {J : Json} {ξ : ByteArray} {lines : List String}
      {pre post : List Release} {y : Release} {trees : Array LoggedTree} {i s n : ℕ} {name : String}
      {spec : Flock.SessionSpec} {record : String} {sess : Flock.RunRecord} {st : Flock.StreamRec},
      readLog label after [J] ξ lines = some (pre ++ y :: post) → coinTreesOf lines = .ok trees →
      trees.size = 1 → ∀ {tabsB : Array (String × Option ℕ)}, sessionTreeReason trees tabsB J = none →
      y.frame = .opening i s n → ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name →
      (name == TREE_SESSION) = false →
      Flock.Record.decode spec record = .ok sess → Flock.JsonParse.parse record = .ok J →
      spec.sha512Digests = true → sess.streams.find? (·.name == name) = some st →
      openedOf pre i s < st.rounds.size → y.answers = st.rounds[openedOf pre i s]!.sha) :
    MovesReach V θ F c k (Binding.Logged Binding.readsV2) A rd σ D M
      (linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) (roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) :=
  fun _ h ⟨hv, hn, hl⟩ => h.movesOf hA hDg h0r hRN ht hR hH hfix reg record recordDigest hrd hD hm hsrc hM hv hn hl
    (hN _ hv hl) (hP _ hv hl) (hfirst _ hv hl) (hcovL _ hv hl) (hO _ hv hl) hans

end Verity.Sampled
