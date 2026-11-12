module

public import Proofs.Properties.Verity.Sampled.CoinMap
public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.Verity.SampledRead
public import Proofs.Properties.Verity.Sampled.Digest

/-!
# Sampled proofs' coupling, assembled from its pieces

`CouplingTarget` at the leg's constructions with the binding's read finder and forger (`Binding.withBinding`), the
bound digests' two-continuation finder (`Constructions.withBind`), the projection at the sample-coupled coin maps
(`Constructions.withProject` at `sampleMaps`) and the messages' two-continuation finders (`Constructions.withMessages`),
cut at the binding's property of the play, the log v2's check read is the device's (`Binding.Logged Binding.readsV2`)
and v2 reads each pinned table at its registered file (`TablesSent`), and at the digest's: v2 samples at the strategy's
`afterAt` (`SampledDigest`).

* **(a), the binding** (`Binding.logged_tables_le`): an accepted play whose log v2 read isn't the device's, or whose
  blobs sent hold a pinned table at other bytes than its registered file, is at most as likely as the read finder's
  collision plus the forger's forgery, wherever the device printed each play's log (`Binding.binds_of_prints`, at
  `Lines.PrintsLog`).
* **(a'), the digest** (`digestBound_of`): an accepted play whose sample doesn't answer `afterAt` is at most as likely as
  the square root of the bound digests' finder's success, a SHA-512 collision on `bind`'s messages
  (`prob_not_sampledDigest_le`).
* **(b), the logged coupling** (`couplingTarget_withProject_of_sample`), off the openings' empty classes (`linkEmpty`,
  `roundEmpty`): the coin map's premise holds within the openings' gaps and empty classes over the stops wherever the
  session's sampling coins are uniform (`sampleMap_couples`), and the decoders' premise holds pointwise
  (`decodes_of_digests`) wherever v2's parameters name the vocabulary's digests, the writer keeps an accepted
  transcript's view and the outcomes on the joint law's support off the empty classes are the writer's at the moves
  v2's files decode to.
* **The sample's coins** (`srcCoinsAt`, `avg_srcCoinsAt`): the session's bytes as `m` fresh frames' coins, uniform by A3
  wherever the session's bytes are `m` frames.
* **The sample under the binding's read** (`coupled_draw_coins_v2`): the sample's piece at the property of the play that
  (a) gives, `Binding.Logged Binding.readsV2`, rather than `Vocab.Logged`. v2 reads no coin of an opening, so the
  device's log is the log it read only where that read shows coins. The read shows every coin of the sample's family
  (`DrawReadAll`, `drawReadAll_logRead`), so the device's log has that family's coins as the read has them
  (`freshCoins_ofV2`).

What the pieces leave is this module's hypotheses, each named in `couplingTarget_assembled`: the device's printed log
(`hP`), the session's bytes as whole frames (`hL`), the openings' gaps and empty classes within the hiding term
(`hε`), the session v2 sets up (`hA`), the vocabulary's digests (`hDg`), the writer's view (`hW`), and the moves on the
joint law's support off the empty classes (`hM`), with the readings `rd`, the decoders `D`, the sample's family `label`
and the field's encoding `e` as parameters, and the sample's digest the strategy's `afterAt`.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ)
open Properties.Challenger (Draws)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The sample's coins at `m` frames**: the session's bytes as `m` fresh frames' coins (`srcCoins`) where the session
reads `m` frames' bytes, else none. -/
noncomputable def srcCoinsAt (m : ℕ) (A : Live) : A.Ω → Fin m → Fin 512 → Bool :=
  if h : A.Lb 0 = m * 64 then srcCoins A h else fun _ _ _ => false

/-- **The sample's coins are uniform** where the session reads `m` frames' bytes (`avg_srcCoins`). -/
theorem avg_srcCoinsAt {m : ℕ} {A : Live} (hL : A.Lb 0 = m * 64) (H : (Fin m → Fin 512 → Bool) → ℝ≥0∞) :
    avg (fun ω => H (srcCoinsAt m A ω)) = avg H := by
  rw [srcCoinsAt, dite_eq_left hL]
  exact avg_srcCoins A hL H

/-! ## The sample under the binding's read -/

/-- **No coin of `m` whole frames is empty.** -/
theorem chunks_ne_nil_of_size : ∀ (m : ℕ) (ξ : ByteArray),
    ξ.size = _root_.Spot.Challenger.COIN_BYTES * m → ∀ cs ∈ chunks m ξ, cs ≠ []
  | 0, _, _, _, h => by simp [chunks] at h
  | m + 1, ξ, h, cs, hc => by
    have h64 : _root_.Spot.Challenger.COIN_BYTES ≤ ξ.size := by
      rw [h, Nat.mul_succ]
      exact Nat.le_add_left _ _
    rw [chunks, List.mem_cons] at hc
    rcases hc with rfl | hc
    · intro he
      have hl := congrArg List.length he
      rw [Verity.bitsOfBytes, List.length_map, List.length_range, ByteArray.size_extract, Nat.min_eq_left h64,
        List.length_nil] at hl
      have : 0 < _root_.Spot.Challenger.COIN_BYTES := by decide
      omega
    · have hs : (ξ.extract _root_.Spot.Challenger.COIN_BYTES ξ.size).size =
          _root_.Spot.Challenger.COIN_BYTES * m := by
        rw [ByteArray.size_extract, Nat.min_self, h, Nat.mul_succ, Nat.add_sub_cancel]
      exact chunks_ne_nil_of_size m _ hs cs hc

/-- **A family's coins the read shows are the device's**: where each read of `log` (`Binding.Read.ofV2`) is of the
release at its place in `olog`, and every coin of family `label` after `after` in `log` is shown, `olog` has the
family's coins `log` has. -/
theorem freshCoins_ofV2 (label : String) (after : ByteArray) :
    ∀ {log olog : List Verity.Release}, List.Forall₂ Binding.Read.Of (log.map Binding.Read.ofV2) olog →
      (∀ cs ∈ freshCoins label after log, cs ≠ []) → freshCoins label after olog = freshCoins label after log
  | [], _, h, _ => by
    rw [List.map_nil, List.forall₂_nil_left_iff] at h
    subst h
    rfl
  | ⟨xf, xa, xc⟩ :: xs, _, h, hne => by
    rw [List.map_cons, List.forall₂_cons_left_iff] at h
    obtain ⟨⟨yf, ya, yc⟩, ys, ⟨hfr, han, hco⟩, hr, rfl⟩ := h
    change xf = yf at hfr
    change xa = ya at han
    subst hfr han
    cases xf with
    | fresh l =>
      by_cases hm : l = label ∧ xa = after
      · have hx : freshCoins label after (⟨.fresh l, xa, xc⟩ :: xs) = xc :: freshCoins label after xs := by
          unfold freshCoins
          exact List.filterMap_cons_some (ite_eq_left hm)
        have hy : freshCoins label after (⟨.fresh l, xa, yc⟩ :: ys) = yc :: freshCoins label after ys := by
          unfold freshCoins
          exact List.filterMap_cons_some (ite_eq_left hm)
        rw [hx] at hne
        have hne0 : xc ≠ [] := hne xc (List.mem_cons.2 (Or.inl rfl))
        have hc : (if xc.isEmpty then none else some xc) = some xc :=
          ite_eq_right fun h => hne0 (List.isEmpty_iff.1 h)
        have hxy : xc = yc := hco xc (Option.mem_def.2 hc)
        rw [hx, hy, hxy, freshCoins_ofV2 label after hr fun cs hc => hne cs (List.mem_cons.2 (Or.inr hc))]
      · have hx : freshCoins label after (⟨.fresh l, xa, xc⟩ :: xs) = freshCoins label after xs := by
          unfold freshCoins
          exact List.filterMap_cons_none (ite_eq_right hm)
        have hy : freshCoins label after (⟨.fresh l, xa, yc⟩ :: ys) = freshCoins label after ys := by
          unfold freshCoins
          exact List.filterMap_cons_none (ite_eq_right hm)
        rw [hx] at hne
        rw [hx, hy, freshCoins_ofV2 label after hr hne]
    | hello s =>
      have hx := freshCoins_cons_step (r := ⟨.hello s, xa, xc⟩) (fun _ h => Verity.Frame.noConfusion h) label
        after xs
      have hy := freshCoins_cons_step (r := ⟨.hello s, xa, yc⟩) (fun _ h => Verity.Frame.noConfusion h) label
        after ys
      rw [hx] at hne
      rw [hx, hy, freshCoins_ofV2 label after hr hne]
    | opening i s n =>
      have hx := freshCoins_cons_step (r := ⟨.opening i s n, xa, xc⟩) (fun _ h => Verity.Frame.noConfusion h)
        label after xs
      have hy := freshCoins_cons_step (r := ⟨.opening i s n, xa, yc⟩) (fun _ h => Verity.Frame.noConfusion h)
        label after ys
      rw [hx] at hne
      rw [hx, hy, freshCoins_ofV2 label after hr hne]

/-- **What the sample asks of a read the binding's property holds at**: `DrawRead`, with every coin of the sample's
family the returned log releases after the sample's digest shown. -/
def DrawReadAll (k : Verity.Registration.Composite) (read : Verity.Sent → Option (List Verity.Release)) : Prop :=
  ∀ s a b e p c ev ξ dr, Verity.SameAccepts k s → V2Case k s a b e →
    V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr →
      ∃ log, read s = some log ∧ ShowsDraw log dr ξ ∧ ∀ cs ∈ freshCoins dr.label dr.after log, cs ≠ []

/-- **The read with v2's own tests shows every coin of the sample's family**: they are the `m` whole frames of the coins
v2 sampled from. -/
theorem fresh_logRead_ne {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hS : Verity.SameAccepts k s)
    (hcase : V2Case k s a b e) (hacc : V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr)
    {log : List Verity.Release} (h : logRead s = some log) :
    ∀ cs ∈ freshCoins dr.label dr.after log, cs ≠ [] := by
  obtain ⟨hp, hc, hev, hξ, hdr, -⟩ := hacc
  obtain ⟨key, hch, hafter⟩ := source_of hdr
  obtain ⟨raw, hraw, hrξ⟩ := Properties.Challenger.oneStageZk_reads p c dr hdr ev ξ hξ
  rw [hrξ] at hraw
  obtain ⟨cm, roots, hts, hstream⟩ := drawCoins_ok hraw
  obtain ⟨_, hsn⟩ := trustedChallengePhaseOf_ok hts
  obtain ⟨_, hso⟩ := challengePhaseNamed_ok hsn
  obtain ⟨ls, lines, hlog, hls, hrr⟩ := challengePhaseOf_ok hso
  rw [logRead_eq (caseOf_of hS hcase hp hc hev) hch hξ (linesOf_of hlog hls)] at h
  obtain ⟨rs, -, h⟩ := Option.bind_eq_some_iff.1 h
  rw [hafter] at hstream
  rw [hafter, fresh_readLog _ _ _ _ h, countFresh_eq, ← releasedRootsOf_ok hrr]
  exact chunks_ne_nil_of_size _ _ (challengerStream_ok hstream)

/-- **The read with v2's own tests meets `DrawReadAll`**, with no premise. -/
theorem drawReadAll_logRead (k : Verity.Registration.Composite) : DrawReadAll k logRead := by
  intro s a b e p c ev ξ dr hS hcase hacc
  obtain ⟨log, hlog⟩ := Option.isSome_iff_exists.1 (logRead_isSome hS hcase hacc)
  exact ⟨log, hlog, shows_logRead hS hcase hacc hlog, fresh_logRead_ne hS hcase hacc hlog⟩

/-- **The play's log shows the sample** where v2's read of it is the device's (`Binding.Logged Binding.readsV2`):
`shows_of_logged` at the binding's property. -/
theorem shows_of_readsV2 {k : Verity.Registration.Composite} (hR : DrawReadAll k coinsRead) {o : Verity.Exchange}
    {a b e : ByteArray} {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hS : Verity.SameAccepts k o.sent)
    (hcase : V2Case k o.sent a b e) (hacc : V2Accepts (Core.Blobs.ofArray o.sent.blobs) a b e p c ev ξ dr)
    (hl : Binding.Logged Binding.readsV2 o) : ShowsDraw o.log dr ξ := by
  obtain ⟨log, hlog, hs, hne⟩ := hR o.sent a b e p c ev ξ dr hS hcase hacc
  have hv : Binding.readsV2 o.sent = some (log.map Binding.Read.ofV2) :=
    congrArg (Option.map (List.map Binding.Read.ofV2)) hlog
  unfold ShowsDraw
  rw [freshCoins_ofV2 dr.label dr.after (hl _ (Option.mem_def.2 hv)) hne]
  exact hs

/-- **The sample in a play** at the binding's property: `logged_draw` with `Binding.Logged Binding.readsV2`. -/
theorem logged_draw_v2 {V : Vocab d α} {k : Verity.Registration.Composite} (hR : DrawReadAll k coinsRead)
    {o : Verity.Exchange} {p₀ : _root_.Spot.OneStageZk.Params} (hθ : ParamsAt o.params p₀)
    (hv : V.Verdict k .sampled o) (hl : Binding.Logged Binding.readsV2 o) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧ ShowsDraw o.log dr ξ := by
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hθ (verdict_case hv)
  obtain ⟨L, n, units, hd⟩ := drew_of hcase hacc
  exact ⟨dr, L, n, ξ, units, hd, shows_of_readsV2 hR hv.2.1 hcase hacc hl⟩

/-- **The sample's piece at the binding's property**: `coupled_draw_coins_tied` with `Binding.Logged Binding.readsV2`
in place of `Vocab.Logged`. At a session v2 sets up and tied to v2's verifier, every outcome whose first coins are the
play's coins of `dr`'s family released after `dr`'s digest selects exactly the proof blocks v2 sampled. -/
theorem coupled_draw_coins_v2 {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V) (A : Live)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Strategy (V.game θ F c)) (hA : SetsUp V θ F A)
    (ht : PopTied θ A) (hR : DrawReadAll k coinsRead) {o : Verity.Exchange} (hθ : o.params = θ)
    (hv : V.Verdict k .sampled o) (hl : Binding.Logged Binding.readsV2 o) {m B : ℕ}
    (κ : A.Ω → Fin m → Fin (8 * B) → Bool)
    (hsrc : ∀ ω, _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m B (κ ω) 0)) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧
      ∀ ω : A.Ω × (K.coins A θ F c σ).T,
        freshCoins dr.label dr.after o.log <+: (List.ofFn fun j => List.ofFn (κ ω.1 j)) →
        (K.live A θ F c σ).L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨p₀, hp₀, -⟩ := hA.params
  obtain ⟨dr, L, n, ξ, units, hd, hs⟩ := logged_draw_v2 hR (by rw [hθ]; exact hp₀) hv hl
  have hn := n_of_popTied ht hv.2.1 hθ hd
  exact ⟨dr, L, n, ξ, units, hd, fun ω hp => (extend_draw A _ ω).trans
    (draw_of_drew hd A ω.1 (law_of_setsUp hA hv.2.1 hθ hd hn) hn
      (List.IsPrefix.trans hs (by rw [hsrc]; exact fresh_prefix_coinBytes hp)))⟩

/-- **The leg's constructions, assembled**: `K`'s with the binding's read finder and forger at the device's history
`dev`, the bound digests' finder at composite `k`, the projection at the readings `rd`, the decoders `D` and the
sample-coupled coin maps at the sample's family `label`, its digest `after` and the coins `κ`, and the messages'
two-continuation finders at `rd`. -/
noncomputable def Constructions.assembled (e : 𝔽 ≃ (Fin 128 → Bool)) {V : Vocab d α} (K : Constructions V)
    (dev : Exchange → Properties.Challenger.History) (k : Verity.Registration.Composite) (rd : Readings)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String)
    (after : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ByteArray) {m : ℕ}
    (κ : (A : Live) → A.Ω → Fin m → Fin 512 → Bool) : Constructions V :=
  (((Binding.withBinding K dev).withBind k).withProject V rd D (sampleMaps e V rd label after κ)).withMessages rd

/-- **The stops' gaps within the hiding term**: at `2^-257` a gap, four gaps at the first stop and at each of the at
most `A.N` stops after it (`couplesErr`: twice the opened coins' gap, `openGap`, and twice the empty class's,
`openEmpty`) are at most the coin trees' hiding term wherever `2(1 + A.N) ≤ maxLeaves`. -/
theorem Live.stops_le_hidingTerm {A : Live} (h : 2 * (1 + A.N) ≤ maxLeaves) :
    4 * ((1 + A.N : ℕ) : ℝ≥0∞) / 2 ^ 257 ≤ A.hidingTerm := by
  unfold Live.hidingTerm
  refine ENNReal.div_le_div_right ?_ _
  exact_mod_cast (by omega : 4 * (1 + A.N) ≤ 2 * maxLeaves)

variable (e : 𝔽 ≃ (Fin 128 → Bool)) (V : Vocab d α) (k : Verity.Registration.Composite)

/-- **The coupling, assembled from its pieces**: at the assembled constructions, `CouplingTarget` holds at the session
`A`, the parameters `θ` and the committed files `F` wherever the device printed each play's log after `θ` (`hP`), its
history at each play signed what the play's signatures are on (`hS`, `Binding.SignsPlay`; `Device.signsPlay_dev` at
the deployed device), the sample's coins are uniform (`hκ`), the openings' gaps and empty classes over the stops are at
most the hiding term (`hε`, which four gaps a stop meet wherever `2(1 + A.N) ≤ maxLeaves`, `Live.stops_le_hidingTerm`),
`A` is the session v2 sets up (`hA`), v2's parameters name the vocabulary's digests (`hDg`), the session's writer keeps
an accepted transcript's view (`hW`), and on the joint law's support off the empty classes the outcomes beside accepted
plays whose log v2 read is the device's, whose blobs sent hold each pinned table at its registered file and whose sample
answers the strategy's `afterAt` are the writer's at the moves v2's files decode to (`hM`). -/
theorem couplingTarget_assembled (K : Constructions V) (dev : Exchange → Properties.Challenger.History)
    (rd : Readings)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String) {m : ℕ}
    (κ : (A : Live) → A.Ω → Fin m → Fin 512 → Bool) (A : Live) (θ : ByteArray) (F : Array ByteArray)
    (hP : ∀ o : Exchange, o.params = θ → Lines.PrintsLog (dev o) o.log)
    (hS : ∀ c, Binding.SignsPlay V dev θ F c)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ A ω)) = avg H)
    (hε : ∀ c σ, couplesErr V θ F c A (rd A θ F c) σ (D A θ F c σ) (sampleMaps e V rd label (afterAt V k) κ A θ F c σ)
      (fun q _ s => openGap (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) +
        openEmpty (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool))
      (fun q => openGap (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool) +
        openEmpty (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool)) ≤
        A.hidingTerm)
    (hA : SetsUp V θ F A) (hDg : DigestsAt V θ) (hW : WritesView A)
    (hM : ∀ c σ, MovesReach V θ F c k
      (fun o => (Binding.Logged Binding.readsV2 o ∧ TablesSent o.sent) ∧ SampledDigest k o.sent (afterAt V k θ F c σ))
      A (rd A θ F c) σ
      (D A θ F c σ) (sampleMaps e V rd label (afterAt V k) κ A θ F c σ) (linkEmpty e (finish (V.signer θ) θ F c σ.1))
      (roundEmpty e (finish (V.signer θ) θ F c σ.1))) :
    CouplingTarget V k (K.assembled e dev k rd D label (afterAt V k) κ) A θ F :=
  couplingTarget_withProject_of_sample e V k rd ((Binding.withBinding K dev).withBind k) D label (afterAt V k) κ A
    (fun o => Binding.Logged Binding.readsV2 o ∧ TablesSent o.sent) θ F
    (fun c σ o => SampledDigest k o.sent (afterAt V k θ F c σ)) hκ
    (fun c σ => Binding.logged_tables_le V k Binding.readsV2 dev θ F c σ (Binding.binds_of_prints V k dev θ hP) (hS c))
    (digestBound_of V k _ (by rfl) A θ F) hε (fun c σ => decodes_of_digests hA hDg (fun _ h => h.1.2) hW (hM c σ))

end Verity.Sampled
