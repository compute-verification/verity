module

public import Proofs.Properties.Verity.Sampled.Window.Steps
public import Proofs.Properties.Verity.Sampled.Window.Assembled
public import Proofs.Properties.Verity.Sampled.RecordDraw

/-!
# Sampled proofs' leg over a window: the sessions of the window's one sample

`Window/Steps.lean` takes the window's sessions at the sample the coin map's coins give (`WindowSessions`, of every
play on a tape through the coin map's first stop). Here that is what the verifier's acceptance gives of a play, read
at the play alone, at the sample-coupled coin map.

* **What the verifier's acceptance gives of a play** (`WindowRead`, the premise): every release of the sample's
  family precedes every opening, and wherever the play's coins of that family agree with coins whose bytes are the
  window's at `ω`, the play has the sessions of the window's sample at `ω` (`WindowSessions`).
* **The sessions at the sample-coupled coin map** (`windowSessions_firstStop`): a play from the sample-coupled first
  stop on the widened coins `ω` that `WindowRead` holds of has the sessions of the sample at `ω`. The walk from the
  sample's fill stops at a log of the fill's releases, then nothing or an opening (`walkFill_req`); the order leaves
  the play's coins of the family those releases' (`freshCoins_of_stopTail`), which agree with the sample's coins.
* **The members' steps at the sample-coupled coin map** (`memberSteps_sampleMap`) and **the leg**
  (`windowSpotSound_sessions`): `windowSpotSound_assembled` at the sessions' readings and decoders, with the members'
  steps from `WindowRead` at each accepted play.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **What the verifier's acceptance gives of the play `x`, at a window**: every release of the family `label`
answering `after` precedes every opening (`fresh`), and wherever the play's coins of that family agree with `nc`
frames' coins `κ` whose bytes are the window's at `ω`, the window's sample at `ω` ended
(`Flock.Guarantees.sampleEnded`) and the play has its sessions (`sessions`, `WindowSessions`). -/
structure WindowRead (W : Window) (tb : String) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray) (label : String) (after : ByteArray) (nc : ℕ) (x : Exchange) :
    Prop where
  fresh : ∀ pre r post, x.log = pre ++ r :: post → ∀ i s n, r.frame = .opening i s n →
    ∀ q ∈ post, ¬ (q.frame = .fresh label ∧ q.answers = after)
  sessions : ∀ (κ : Fin nc → Fin 512 → Bool) (ω : W.Ω),
    _root_.Properties.Spot.bytesOf (W.src ω 0) = _root_.Properties.Spot.bytesOf (coinBytes nc 64 κ 0) →
    CoinsAgree (freshCoins label after x.log) (fillCoins κ) →
    Flock.Guarantees.sampleEnded W.src W.law W.nJ ω = true ∧ WindowSessions W tb ev filesDec (W.L.draw ω) x

/-- The window's sessions at an equal sample. -/
theorem WindowSessions.of_eq {W : Window} {tb : String} {ev : ℕ → Exchange → Option Claims.Msg}
    {filesDec : Claims.Msg → Array ByteArray} {S S' : Finset (Fin W.nJ)} {x : Exchange}
    (h : WindowSessions W tb ev filesDec S x) (e : S = S') : WindowSessions W tb ev filesDec S' x :=
  e ▸ h

/-- **The play on a tape through a position is a play from the position**, at a position that keeps the invariant. -/
theorem playReaches_through {g : Game Exchange} {σ : Strategy g} {R : ℕ} {Ks₀ : Fin R → DigestCut Claims.Msg g}
    {q : Pos g σ Claims.Msg (Fin R)} (hi : q.Inv Ks₀) (w : CoinTape q.g' q.s') :
    PlayReaches q.g' q.s' (CoinTape.run g σ (q.back w)) := by
  rw [hi.run w]
  exact run_reaches _ _ w

/-- The window's bytes at `ω` are the bytes of the window's coins at `nc` frames, where the window reads `nc` frames'
bytes (`bytesOf_byteCoins`). -/
theorem Window.bytesOf_srcCoinsAt {nc : ℕ} {W : Window} (hL : W.Lb 0 = nc * 64) (ω : W.Ω) :
    _root_.Properties.Spot.bytesOf (W.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes nc 64 (W.srcCoinsAt nc ω) 0) := by
  have h : W.srcCoinsAt nc = byteCoins W.src hL := by
    unfold Window.srcCoinsAt
    exact dif_pos hL
  rw [h]
  exact bytesOf_byteCoins W.src hL ω

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The sessions of the sample, from the sample-coupled first stop**: at the sample-coupled coin map at the window's
bytes as `nc` frames' coins, a play from the first stop on the widened coins `ω` that `WindowRead` holds of has the
sessions of the window's sample at `ω`. -/
theorem windowSessions_of_reach (W : Window) (rd : Reading W.rounds) (hrd : PicksOpenings rd)
    (σ : Strategy (V.game θ F c)) (label : String) (after : ByteArray) {nc : ℕ} (hL : W.Lb 0 = nc * 64)
    {tb : String} {ev : ℕ → Exchange → Option Claims.Msg} {filesDec : Claims.Msg → Array ByteArray}
    (ω : (W.extend (windowSampleMap fieldEnc V θ F c W rd σ label after (W.srcCoinsAt nc)).widening).L.Ω)
    {x : Exchange}
    (hreach : PlayReaches
      (windowFirstStop V θ F c W rd σ (windowSampleMap fieldEnc V θ F c W rd σ label after (W.srcCoinsAt nc)) ω).g'
      (windowFirstStop V θ F c W rd σ (windowSampleMap fieldEnc V θ F c W rd σ label after (W.srcCoinsAt nc)) ω).s'
      x)
    (hR : WindowRead W tb ev filesDec label after nc x) :
    Flock.Guarantees.sampleEnded W.src W.law W.nJ ω.1 = true ∧ WindowSessions W tb ev filesDec
      ((W.extend (windowSampleMap fieldEnc V θ F c W rd σ label after (W.srcCoinsAt nc)).widening).L.draw ω) x := by
  rw [windowFirstStop_sampleMap fieldEnc V θ F c W rd σ label after (W.srcCoinsAt nc) ω] at hreach
  obtain ⟨L, hc, hO⟩ := walkFill_req (label := label) (after := after) hrd (frames (V.families θ)) (V.families θ)
    (V.signer θ) θ F c _ (fun a => releases_bind _ _ _ _) σ (roundCuts V θ F c rd) (roundCuts_heq V θ F c rd) nc
    (W.srcCoinsAt nc ω.1) ω.2 id
  obtain ⟨log', hlog, htail⟩ := hO x hreach
  have hP := hR.fresh
  have hc' : CoinsAgree (freshCoins label after x.log) (fillCoins (W.srcCoinsAt nc ω.1)) := by
    rw [hlog] at hP ⊢
    rw [freshCoins_of_stopTail hP htail]
    exact hc
  obtain ⟨hE, hS⟩ := hR.sessions (W.srcCoinsAt nc ω.1) ω.1 (Window.bytesOf_srcCoinsAt hL ω.1) hc'
  exact ⟨hE, hS.of_eq (W.sample_extend _ ω).symm⟩

/-- **The sessions of the sample, on a tape through the sample-coupled first stop** (`windowSessions_of_reach`, at
the play on the tape, `playReaches_through`). -/
theorem windowSessions_firstStop (W : Window) (rd : Reading W.rounds) (hrd : PicksOpenings rd)
    (σ : Strategy (V.game θ F c)) (label : String) (after : ByteArray) {nc : ℕ} (hL : W.Lb 0 = nc * 64)
    {tb : String} {ev : ℕ → Exchange → Option Claims.Msg} {filesDec : Claims.Msg → Array ByteArray}
    (ω : (W.extend (windowSampleMap fieldEnc V θ F c W rd σ label after (W.srcCoinsAt nc)).widening).L.Ω)
    {z : CoinTape (V.game θ F c) σ}
    (hz : ∃ w, z = (windowFirstStop V θ F c W rd σ
      (windowSampleMap fieldEnc V θ F c W rd σ label after (W.srcCoinsAt nc)) ω).back w)
    (hR : WindowRead W tb ev filesDec label after nc (CoinTape.run (V.game θ F c) σ z)) :
    Flock.Guarantees.sampleEnded W.src W.law W.nJ ω.1 = true ∧ WindowSessions W tb ev filesDec
      ((W.extend (windowSampleMap fieldEnc V θ F c W rd σ label after (W.srcCoinsAt nc)).widening).L.draw ω)
      (CoinTape.run (V.game θ F c) σ z) := by
  obtain ⟨w, hw⟩ := hz
  have hreach := playReaches_through (Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c rd)
    ((windowSampleMap fieldEnc V θ F c W rd σ label after (W.srcCoinsAt nc)).start ω))).2.2 w
  have hreach' : PlayReaches
      (windowFirstStop V θ F c W rd σ (windowSampleMap fieldEnc V θ F c W rd σ label after (W.srcCoinsAt nc)) ω).g'
      (windowFirstStop V θ F c W rd σ (windowSampleMap fieldEnc V θ F c W rd σ label after (W.srcCoinsAt nc)) ω).s'
      (CoinTape.run (V.game θ F c) σ z) := by
    rw [hw]
    exact hreach
  exact windowSessions_of_reach V θ F c W rd hrd σ label after hL ω hreach' hR

variable (k : Verity.Registration.Composite)

/-- **The members' steps at the sample-coupled coin map** (`memberSteps_of_sessions`): at a window whose members each
have the one table `tb` and that reads `nc` frames' bytes, at the sessions' readings and decoders and at the
sample-coupled coin map at the family `label` answering `after` and the window's bytes as `nc` frames' coins, wherever
`WindowRead` holds of every play the sampled verdict accepts and `P` holds of. The coin map writes the link points'
and each round's coins' bits into a stop's opening by definition (`windowOpenMap`), and the plays on tapes through
its first stop have the sessions of its sample (`windowSessions_firstStop`). -/
theorem memberSteps_sampleMap (P : Exchange → Prop) (W : Window) (t : ℕ) (reg : Option Claims.Msg → ByteArray)
    (tb : String) (ev : ℕ → Exchange → Option Claims.Msg) (filesDec : Claims.Msg → Array ByteArray)
    (record₀ : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) (σ : Strategy (V.game θ F c))
    (label : String) (after : ByteArray) {nc : ℕ} (hL : W.Lb 0 = nc * 64) (hMA : MembersAt W tb)
    (hWR : ∀ z : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ z) →
      P (CoinTape.run (V.game θ F c) σ z) →
      WindowRead W tb ev filesDec label after nc (CoinTape.run (V.game θ F c) σ z)) :
    MemberSteps V θ F c k P W t (sessionReading W.rounds tb ev record₀ recordDigest) σ
      (fun _ => sessionDecoders reg tb filesDec)
      (windowSampleMap fieldEnc V θ F c W (sessionReading W.rounds tb ev record₀ recordDigest) σ label after
        (W.srcCoinsAt nc))
      (fun _ _ => linkEmptyAt fieldEnc (finish (V.signer θ) θ F c σ.1))
      (fun _ => roundEmptyAt fieldEnc (finish (V.signer θ) θ F c σ.1))
      (StepInv V θ F c W (sessionReading W.rounds tb ev record₀ recordDigest) σ
        (windowSampleMap fieldEnc V θ F c W (sessionReading W.rounds tb ev record₀ recordDigest) σ label after
          (W.srcCoinsAt nc))) :=
  memberSteps_of_sessions V θ F c k P W t reg tb ev filesDec record₀ recordDigest σ
    (windowSampleMap fieldEnc V θ F c W (sessionReading W.rounds tb ev record₀ recordDigest) σ label after
      (W.srcCoinsAt nc))
    (fun _ _ _ _ => rfl) (fun _ _ _ _ _ => rfl) hMA fun ω z hz hv hp =>
      windowSessions_firstStop V θ F c W (sessionReading W.rounds tb ev record₀ recordDigest)
        (windowReadingAt_picks #[tb] 1026 (sessionReveal #[tb] ev) record₀ recordDigest) σ label after hL ω hz
        (hWR z hv hp)

omit θ F c in
/-- **The readings of every window's sessions** (`sessionReading`), at each window's table `tb W`, record and record
digest. -/
abbrev sessionReadings (tb : Window → String) (ev : ℕ → Exchange → Option Claims.Msg)
    (record₀ : Window → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Window → ByteArray → Array ByteArray → ByteArray → Claims.Msg) : WindowReadings :=
  fun W θ F c => sessionReading W.rounds (tb W) ev (record₀ W θ F c) (recordDigest W θ F c)

omit θ F c in
/-- **SPOT's leg over a window of several members, from what the verifier's acceptance gives of a play**:
`windowSpotSound_assembled` at the sessions' readings (`sessionReadings`) and decoders (`sessionDecoders`), under
`fieldEnc`. Soundness's vocabulary at `.sampled` reading, at the deployment's windows `S` and the constructions with
the binding's finders at the device's history `dev`, the window's projection at those readings and decoders and the
sample-coupled coin maps at the family `label`, the strategy's `afterAt` and the window's bytes as `nc` frames'
coins, with `S` covering the one-stage form, meets SPOT's target at composite `k` wherever, at each window of `S`:
the device printed each play's log after the parameters (`hP`); its history at each play signed what the play's
signatures are on (`hS`); the window reads `nc` frames' bytes (`hL`); its members each have the one table `tb W`
with hm96-sha512 rows (`hMA`); and `WindowRead` holds of each play the sampled verdict accepts, whose log v2 read is
the device's, whose blobs sent hold each pinned table and each member's files at their registered files and whose
sample answers the strategy's `afterAt` (`hWR`). The members' steps are `memberSteps_sampleMap`'s at the invariant
`StepInv`, and each round's reading picks only openings (`windowReadingAt_picks`). -/
theorem windowSpotSound_sessions (K : WindowConstructions V) (dev : Exchange → Properties.Challenger.History)
    (reg : Option Claims.Msg → ByteArray) (tb : Window → String) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray)
    (record₀ : Window → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Window → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String) (nc : ℕ)
    {S : ByteArray → Array ByteArray → Option Window}
    (hD : WindowDeployed V ((K.withBinding dev k).withProject V (sessionReadings tb ev record₀ recordDigest)
      (fun W _ _ _ _ _ => sessionDecoders reg (tb W) filesDec)
      (windowSampleMaps fieldEnc V (sessionReadings tb ev record₀ recordDigest) label (afterAt V k)
        fun W => W.srcCoinsAt nc)) S)
    (hCov : WindowCovers V k S)
    (hP : ∀ θ F W, S θ F = some W → ∀ o : Exchange, o.params = θ → Lines.PrintsLog (dev o) o.log)
    (hS : ∀ θ F W, S θ F = some W → ∀ c, Binding.SignsPlay V dev θ F c)
    (hL : ∀ θ F W, S θ F = some W → W.Lb 0 = nc * 64)
    (hMA : ∀ θ F W, S θ F = some W → MembersAt W (tb W))
    (hWR : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c))
      (z : CoinTape (V.game θ F c) σ), V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ z) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ z) →
      TablesSent (CoinTape.run (V.game θ F c) σ z).sent → FilesSent V (CoinTape.run (V.game θ F c) σ z) →
      SampledDigest k (CoinTape.run (V.game θ F c) σ z).sent (afterAt V k θ F c σ) →
      WindowRead W (tb W) ev filesDec (label θ F c σ) (afterAt V k θ F c σ) nc
        (CoinTape.run (V.game θ F c) σ z)) :
    V.SpotSound k :=
  windowSpotSound_assembled V k fieldEnc (sessionReadings tb ev record₀ recordDigest) K dev
    (fun W _ _ _ _ _ => sessionDecoders reg (tb W) filesDec) label nc hD hCov
    (fun W θ F c σ => StepInv V θ F c W (sessionReadings tb ev record₀ recordDigest W θ F c) σ
      (windowSampleMaps fieldEnc V (sessionReadings tb ev record₀ recordDigest) label (afterAt V k)
        (fun W => W.srcCoinsAt nc) W θ F c σ))
    hP hS hL
    (fun θ F W _ c =>
      windowReadingAt_picks #[tb W] 1026 (sessionReveal #[tb W] ev) (record₀ W θ F c) (recordDigest W θ F c))
    (fun θ F W hW c σ =>
      memberSteps_sampleMap V θ F c k
        (fun o => (Binding.Logged Binding.readsV2 o ∧ TablesSent o.sent ∧ FilesSent V o) ∧
          SampledDigest k o.sent (afterAt V k θ F c σ))
        W (V.cost θ F c σ) reg (tb W) ev filesDec (record₀ W θ F c) (recordDigest W θ F c) σ (label θ F c σ)
        (afterAt V k θ F c σ) (hL θ F W hW) (hMA θ F W hW)
        fun z hv hp => hWR θ F W hW c σ z hv hp.1.1 hp.1.2.1 hp.1.2.2 hp.2)

end Verity.Sampled
