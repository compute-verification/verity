module

public import Proofs.Properties.Verity.Sampled.Window.Leg
public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.Verity.Sampled.Digest

/-!
# Sampled proofs' leg over a window, assembled from its pieces

`windowSpotSound_of_sample` at the cut the one-session leg uses (`couplingTarget_assembled`): the log v2's check read
is the device's (`Binding.Logged Binding.readsV2`), v2 reads each pinned table at its registered file (`TablesSent`),
and v2 samples at the strategy's `afterAt` (`SampledDigest`). The binding, the digest and the sample's coins are stated
of the period's game and not of a session, so they hold at a window as they do at one session:

* **(a), the binding** (`Binding.logged_tables_le`): at the read finder and forger at the device's history `dev`,
  wherever the device printed each play's log (`Binding.binds_of_prints`) and its history at each play signed what the
  play's signatures are on (`Binding.SignsPlay`).
* **(a'), the digest** (`prob_not_sampledDigest_le`): at the bound digests' two-continuation finder at the window's
  budget (`windowBindFinder`).
* **The sample's coins** (`Window.srcCoinsAt`, `Window.avg_srcCoinsAt`): the window's bytes as `m` fresh frames' coins,
  uniform by A3 wherever the window's bytes are `m` frames.

What is left is `windowSpotSound_assembled`'s hypotheses: the deployment's side (`WindowDeployed`, `WindowCovers`), the
device's printed log and signatures, the window's bytes as whole frames, that each round's reading picks only
openings, and the members' steps off the openings' empty classes at their counts (`MemberSteps`).
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The sample's coins at `m` frames, at a window**: the window's bytes as `m` fresh frames' coins (`byteCoins`)
where the window reads `m` frames' bytes, else none. -/
noncomputable def Window.srcCoinsAt (m : ℕ) (W : Window) : W.Ω → Fin m → Fin 512 → Bool :=
  if h : W.Lb 0 = m * 64 then byteCoins W.src h else fun _ _ _ => false

/-- **The window's sample's coins are uniform** where the window reads `m` frames' bytes (`Window.avg_byteCoins`). -/
theorem Window.avg_srcCoinsAt {m : ℕ} {W : Window} (hL : W.Lb 0 = m * 64) (H : (Fin m → Fin 512 → Bool) → ℝ≥0∞) :
    avg (fun ω => H (W.srcCoinsAt m ω)) = avg H := by
  have h : W.srcCoinsAt m = byteCoins W.src hL := by
    unfold Window.srcCoinsAt
    exact dif_pos hL
  rw [h]
  exact W.avg_byteCoins hL H

variable (V : Vocab d α) (k : Verity.Registration.Composite)

/-- **The bound digests' two-continuation finder, at a window** (`bindFinder`): the commitment's cut's, `σ`'s play on
two independent tapes from the start, at the window's budget, `2t + 2h` SHA-512 calls. -/
noncomputable def windowBindFinder (W : Window) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) : Claims.Finder :=
  (bindCut V k θ F c).finder σ (2 * V.cost θ F c σ + 2 * W.h)

theorem windowBindFinder_calls : ∀ (W : Window) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (x : (windowBindFinder V k W θ F c σ).S),
    ((windowBindFinder V k W θ F c σ).cost x : ℝ) ≤ W.qM (V.cost θ F c σ) := fun W θ F c σ _ => by
  show ((2 * V.cost θ F c σ + 2 * W.h : ℕ) : ℝ) ≤ W.qM (V.cost θ F c σ)
  simp [Window.qM]

/-- **The window's constructions with the binding's finders**: `K`'s with the read finder and forger at the device's
history `dev` (`Binding.readFinder`, `Binding.forger`) and the bound digests' finder at composite `k`
(`windowBindFinder`). -/
noncomputable def WindowConstructions.withBinding {V : Vocab d α} (K : WindowConstructions V)
    (dev : Exchange → Properties.Challenger.History) (k : Verity.Registration.Composite) : WindowConstructions V :=
  { K with
    readFinder := Binding.readFinder V dev
    forger := Binding.forger V dev
    bindFinder := windowBindFinder V k
    bind_calls := windowBindFinder_calls V k }

/-- **SPOT's leg over a window of several members, assembled from its pieces**: `windowSpotSound_of_sample` at the
binding's cut and the digest's. Soundness's vocabulary at `.sampled` reading, at the deployment's windows `S` and the
constructions with the binding's finders at the device's history `dev`, the window's projection at the readings `rd`,
the members' decoders `D` and the sample-coupled coin maps at the family `label`, the strategy's `afterAt` and the
window's bytes as `nc` frames' coins, with `S` covering the one-stage form, meets SPOT's target at composite `k`
wherever, at each window of `S`: the device printed each play's log after the parameters (`hP`); its history at each
play signed what the play's signatures are on (`hS`); the window reads `nc` frames' bytes (`hL`); each round's reading
picks only openings (`hR`); and the members' steps hold off the openings' empty classes at their counts (`hm`,
`MemberSteps`) at accepted plays whose log v2 read is the device's, whose blobs sent hold each pinned table at its
registered file and whose sample answers the strategy's `afterAt`. -/
theorem windowSpotSound_assembled (e : 𝔽 ≃ (Fin 128 → Bool)) (rd : WindowReadings) (K : WindowConstructions V)
    (dev : Exchange → Properties.Challenger.History)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowDecoders V θ F c W σ)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String) (nc : ℕ)
    {S : ByteArray → Array ByteArray → Option Window}
    (hD : WindowDeployed V ((K.withBinding dev k).withProject V rd D
      (windowSampleMaps e V rd label (afterAt V k) fun W => W.srcCoinsAt nc)) S)
    (hCov : WindowCovers V k S)
    (I : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      (W.extend (windowSampleMaps e V rd label (afterAt V k) (fun W => W.srcCoinsAt nc) W θ F c σ).widening).L.Ω →
        Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (hP : ∀ θ F W, S θ F = some W → ∀ o : Exchange, o.params = θ → Lines.PrintsLog (dev o) o.log)
    (hS : ∀ θ F W, S θ F = some W → ∀ c, Binding.SignsPlay V dev θ F c)
    (hL : ∀ θ F W, S θ F = some W → W.Lb 0 = nc * 64)
    (hR : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) j pre fr, (rd W θ F c).round j pre fr →
      ∃ i s n, fr = .opening i s n)
    (hm : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      MemberSteps V θ F c k
        (fun o => (Binding.Logged Binding.readsV2 o ∧ TablesSent o.sent) ∧
          SampledDigest k o.sent (afterAt V k θ F c σ))
        W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (windowSampleMaps e V rd label (afterAt V k) (fun W => W.srcCoinsAt nc) W θ F c σ)
        (fun _ _ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
        (fun _ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1)) (I W θ F c σ)) :
    V.SpotSound k :=
  windowSpotSound_of_sample V k e rd (K.withBinding dev k) D label (afterAt V k) (fun W => W.srcCoinsAt nc) hD hCov
    (fun o => Binding.Logged Binding.readsV2 o ∧ TablesSent o.sent)
    (fun θ F c σ o => SampledDigest k o.sent (afterAt V k θ F c σ)) I
    (fun θ F W hW H => Window.avg_srcCoinsAt (hL θ F W hW) H)
    (fun θ F W hW c σ => Binding.logged_tables_le V k Binding.readsV2 dev θ F c σ
      (Binding.binds_of_prints V k dev θ (hP θ F W hW)) (hS θ F W hW c))
    (fun θ F W _ c σ => prob_not_sampledDigest_le V k θ F c σ (2 * V.cost θ F c σ + 2 * W.h))
    hR hm

end Verity.Sampled
