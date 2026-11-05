module

public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Properties.Composition

/-!
# v2 reads the sample's coins of one family and one digest

The record's draw (`recordDraw_firstStop`) and the sample-coupled coin map (`sampleMap`) take a family `label` and a
digest `after` fixed by the parameters, the committed files, the commitment and the strategy, before any coin, and v2's
draw in every play the sampled verdict accepts at them (`SampleNamed`). v2 draws under the challenger its parameters
name, at the coins of its family released after the registration's digest (`source_of`):

* **The family** (`draw_label`, `labelAt`): the parameters' challenger family, a function of the parameters `θ` alone,
  since v2's case reads the parameters `θ` names for it (`params_of_case`).
* **The digest** (`draw_after`): sampled proofs' entry in the components' digests that the prover's commitment binds
  (`BindsAt`, `Core.Composition.bind`), where the composition accepted. The commitment is fixed before any coin, and
  the cases are sent at the close, so two of a strategy's plays can draw at two digests only where the two lists of
  digests they bind into the one commitment differ, which SHA-512 colliding on two different messages of `bind`
  gives (`collides_of_binds`, at two given plays, and `opensOnce_or_collides`, by
  `Properties.Composition.bindCollision`).

So `SampleNamed` holds at the family `labelAt θ` and the digest `afterAt` of the strategy's accepted plays wherever
those plays bind the commitment to one list of digests (`OpensOnce`, `sampleNamed_run`); v2's acceptance gives no more.
The logged reading isn't needed (`sampleNamed_layout`, the form `movesReach_layout`'s `hN` takes).
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Properties.Challenger (Draws)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The family -/

open Classical in
/-- **The sample's family at the parameters `θ`**: the challenger's family v2's parameters in `θ` name (`ParamsAt`),
or the empty string where they name none. -/
noncomputable def labelAt (θ : ByteArray) : String :=
  if h : ∃ l p key, ParamsAt θ p ∧ p.base.challenger = some (key, l) then h.choose else ""

theorem labelAt_eq {θ : ByteArray} {p : _root_.SampledProofs.OneStageZk.Params} {key} {l : String}
    (hp : ParamsAt θ p) (hc : p.base.challenger = some (key, l)) : labelAt θ = l := by
  have h : ∃ l p key, ParamsAt θ p ∧ p.base.challenger = some (key, l) := ⟨l, p, key, hp, hc⟩
  rw [labelAt, dite_eq_left h]
  obtain ⟨p', key', hp', hc'⟩ := h.choose_spec
  rw [ParamsAt.unique hp' hp, hc] at hc'
  exact (Prod.mk.inj (Option.some.inj hc')).2.symm

/-- **v2 draws at the parameters' family**: where composite `k` accepted what the prover sent and v2's case in it
accepts at the draw `dr`, `dr`'s family is the one v2's parameters in what was sent name. -/
theorem draw_label {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {ev : _root_.SampledProofs.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws}
    {p₀ : _root_.SampledProofs.OneStageZk.Params} (hS : Verity.SameAccepts k s) (hcase : V2Case k s a b e)
    (hacc : V2Accepts (_root_.Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr) (hθ : ParamsAt s.params p₀) :
    dr.label = labelAt s.params := by
  obtain ⟨hp, -, -, -, hdr, -⟩ := hacc
  obtain rfl : p = p₀ := Sum.inl.inj (params_of_case hS hcase hp hθ)
  obtain ⟨key, hch, -⟩ := source_of hdr
  exact (labelAt_eq hθ hch).symm

/-! ## The digest -/

/-- **The digests what the prover sent binds**: the three byte strings decode under the composition's formats, and
`ds`, each of composite `k`'s components' ids with the digest of its case's commitment, in order, is what the
commitment binds (`Core.Composition.bind`). -/
def BindsAt (k : Verity.Registration.Composite) (s : Verity.Sent) (ds : List (String × ByteArray)) : Prop :=
  ∃ ps c ev, Verity.Decoded s ps c ev ∧ Properties.Composition.digests (s.components k) ps ev = some ds ∧
    Core.Composition.bind (Verity.verifier k).id ds = c

/-- A `mapM` whose values each carry their entry's key lists the entries' keys. -/
theorem mapM_keys {β : Type _} {f : β → Option (String × ByteArray)} {key : β → String}
    (hk : ∀ x y, f x = some y → y.1 = key x) :
    ∀ {l : List β} {ys : List (String × ByteArray)}, l.mapM f = some ys → ys.map Prod.fst = l.map key
  | [], ys, h => by
    cases ys with
    | nil => rfl
    | cons _ _ => simp at h
  | a :: l, ys, h => by
    cases ha : f a with
    | none => simp [List.mapM_cons, ha] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [List.mapM_cons, ha, hl] at h
      | some zs =>
        simp only [List.mapM_cons, ha, hl, Option.pure_def, Option.bind_eq_bind, Option.bind_some,
          Option.some.injEq] at h
        subst h
        rw [List.map_cons, List.map_cons, hk a b ha, mapM_keys hk hl]

/-- **v2 draws at the digest sampled proofs' case commits**: where composite `k` accepted what the prover sent and
v2's case in it accepts at the draw `dr`, the digests the commitment binds (`BindsAt`), keyed by distinct ids, give
sampled proofs `dr`'s digest. -/
theorem draw_after {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {ev : _root_.SampledProofs.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hS : Verity.SameAccepts k s)
    (hcase : V2Case k s a b e) (hacc : V2Accepts (_root_.Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr) :
    ∃ ds, BindsAt k s ds ∧ (ds.map Prod.fst).Nodup ∧ (VStar.Recursive.Sampled.PROTOCOL, dr.after) ∈ ds := by
  obtain ⟨-, hc, -, -, hdr, -⟩ := hacc
  obtain ⟨_, _, hafter⟩ := source_of hdr
  obtain ⟨ps, c₀, evs, ⟨hps, hc0, hevs⟩, x, hx, hx1, rfl, rfl, rfl⟩ := hcase
  obtain ⟨ps', c₀', evs', ⟨hps', hc0', hevs'⟩, _, _, ⟨⟨⟨-, -, hfst, -⟩, -, hdig, -, -⟩, -⟩, -⟩ := hS
  rw [Except.ok.inj (hps'.symm.trans hps), Except.ok.inj (hevs'.symm.trans hevs)] at hfst hdig
  rw [Except.ok.inj (hc0'.symm.trans hc0)] at hdig
  obtain ⟨ds, hds, hb⟩ := Option.map_eq_some_iff.1 hdig
  have hds' : (Properties.Composition.triples (s.components k) ps evs).mapM
      (fun x => (Properties.Composition.digestOf x.1 x.2.2.commitment).map (x.1.id, ·)) = some ds := hds
  have hkeys : ds.map Prod.fst = (Properties.Composition.triples (s.components k) ps evs).map (fun x => x.1.id) :=
    mapM_keys (fun x y h => by obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 h; rfl) hds'
  refine ⟨ds, ⟨ps, c₀, evs, ⟨hps, hc0, hevs⟩, hds, hb⟩, ?_, ?_⟩
  · have h2 : (Properties.Composition.triples (s.components k) ps evs).map (fun x => x.1.id) =
        (s.components k).map (·.id) :=
      (List.map_map ..).symm.trans (congrArg (List.map (·.id)) hfst)
    rw [hkeys, h2]
    exact components_nodup k s
  · obtain ⟨y, hy, hfy⟩ := Properties.Composition.mapM_some hds' x hx
    have hdg : Properties.Composition.digestOf x.1 x.2.2.commitment = some c.digest := by
      rw [hx1]
      show Properties.Composition.digestOf _root_.SampledProofs.OneStage.protocol x.2.2.commitment = _
      unfold Properties.Composition.digestOf
      rw [show _root_.SampledProofs.OneStage.protocol.commitment x.2.2.commitment = .ok c from hc]
      rfl
    have hid : x.1.id = VStar.Recursive.Sampled.PROTOCOL := (congrArg (·.id) hx1).trans rfl
    simp only [hdg, Option.map_some, Option.some.injEq, hid] at hfy
    rw [hafter, hfy]
    exact hy

/-- **A play of the period's game ends at its commitment.** -/
theorem play_commitment (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ) :
    (CoinTape.run (V.game θ F c) σ x).commitment = c := by
  obtain ⟨log, s, h⟩ := Verity.Game.PlayReaches.of_bind (f := finish (V.signer θ) θ F c σ.1)
    (releases (frames (V.families θ)) (V.families θ) []) σ.2 (run_reaches (V.game θ F c) σ x)
  obtain ⟨ρ, ho⟩ : ∃ ρ : Fin (V.signer θ).draws → Bool, CoinTape.run (V.game θ F c) σ x =
      ⟨θ, F, c, σ.1, log, s.1, List.ofFn ρ, (V.signer θ).signatures σ.1 log s.1 (List.ofFn ρ), (s.2 ρ).1.1,
        (s.2 ρ).1.2⟩ := h
  rw [ho]

/-! ## One digest over a strategy's plays -/

/-- **`σ`'s accepted plays bind the commitment to one list of digests**: wherever the sampled verdict accepts two of
`σ`'s plays, the digests each binds into the commitment (`BindsAt`) are one list. -/
def OpensOnce (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) : Prop :=
  ∀ x x' : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x') → ∀ ds ds',
      BindsAt k (CoinTape.run (V.game θ F c) σ x).sent ds → BindsAt k (CoinTape.run (V.game θ F c) σ x').sent ds' →
        ds = ds'

/-- **Two of `σ`'s plays that bind two lists of digests give a SHA-512 collision**: both bind them into the game's
commitment, fixed before any coin, so `bind`'s messages of the two differ and have one SHA-512 digest. A finder that
holds the two plays outputs that pair. -/
theorem collides_of_binds (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) (x x' : CoinTape (V.game θ F c) σ)
    {ds ds' : List (String × ByteArray)} (hB : BindsAt k (CoinTape.run (V.game θ F c) σ x).sent ds)
    (hB' : BindsAt k (CoinTape.run (V.game θ F c) σ x').sent ds') (hne : ds ≠ ds') :
    Properties.Composition.bindMessage (Verity.verifier k).id ds ≠
        Properties.Composition.bindMessage (Verity.verifier k).id ds' ∧
      Flock.Sha512.hash (Properties.Composition.bindMessage (Verity.verifier k).id ds) =
        Flock.Sha512.hash (Properties.Composition.bindMessage (Verity.verifier k).id ds') := by
  obtain ⟨_, c₀, _, ⟨-, hc0, -⟩, -, hb⟩ := hB
  obtain ⟨_, c₀', _, ⟨-, hc0', -⟩, -, hb'⟩ := hB'
  have e1 : (CoinTape.run (V.game θ F c) σ x).sent.commitment = c := play_commitment V θ F c σ x
  have e2 : (CoinTape.run (V.game θ F c) σ x').sent.commitment = c := play_commitment V θ F c σ x'
  rw [e1] at hc0
  rw [e2] at hc0'
  have hcc : c₀' = c₀ := Except.ok.inj (hc0'.symm.trans hc0)
  rcases Properties.Composition.bindCollision (Verity.verifier k).id ds ds' with hb₁ | hb₁ | hb₁
  · exact (hb₁ (by rw [hb, hb', hcc])).elim
  · exact (hne hb₁).elim
  · exact hb₁

/-- **`σ`'s accepted plays bind one list of digests, or SHA-512 collides**: else two of them bind two different lists
into the one commitment, and `bind`'s messages of the two differ and have one SHA-512 digest (`collides_of_binds`). -/
theorem opensOnce_or_collides (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Strategy (V.game θ F c)) :
    OpensOnce V k θ F c σ ∨ ∃ ds ds', ds ≠ ds' ∧
      (∃ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧
        BindsAt k (CoinTape.run (V.game θ F c) σ x).sent ds) ∧
      (∃ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧
        BindsAt k (CoinTape.run (V.game θ F c) σ x).sent ds') ∧
      Properties.Composition.bindMessage (Verity.verifier k).id ds ≠
        Properties.Composition.bindMessage (Verity.verifier k).id ds' ∧
      Flock.Sha512.hash (Properties.Composition.bindMessage (Verity.verifier k).id ds) =
        Flock.Sha512.hash (Properties.Composition.bindMessage (Verity.verifier k).id ds') := by
  by_cases h : OpensOnce V k θ F c σ
  · exact Or.inl h
  refine Or.inr (Classical.byContradiction fun hn => h ?_)
  intro x x' hv hv' ds ds' hB hB'
  by_contra hne
  exact hn ⟨ds, ds', hne, ⟨x, hv, hB⟩, ⟨x', hv', hB'⟩, collides_of_binds V k θ F c σ x x' hB hB' hne⟩

/-- **`dg` is sampled proofs' digest in an accepted play of `σ`**: the sampled verdict accepts the play at some tape,
and the digests its commitment binds give sampled proofs `dg`. -/
def DigestAt (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) (dg : ByteArray) : Prop :=
  ∃ (x : CoinTape (V.game θ F c) σ) (ds : List (String × ByteArray)),
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ BindsAt k (CoinTape.run (V.game θ F c) σ x).sent ds ∧
      (VStar.Recursive.Sampled.PROTOCOL, dg) ∈ ds

open Classical in
/-- **The sample's digest at `σ`**: sampled proofs' digest in one of `σ`'s accepted plays (`DigestAt`), or the empty
string where the sampled verdict accepts none. -/
noncomputable def afterAt (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) : ByteArray :=
  if h : ∃ dg, DigestAt V k θ F c σ dg then h.choose else ByteArray.empty

theorem afterAt_spec {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {σ : Strategy (V.game θ F c)} (h : ∃ dg, DigestAt V k θ F c σ dg) :
    DigestAt V k θ F c σ (afterAt V k θ F c σ) := by
  rw [afterAt, dite_eq_left h]
  exact h.choose_spec

/-- **v2 reads the sample's coins of the family `labelAt θ` answering `afterAt`** in every play of `σ` the sampled
verdict accepts, where `A` is the session v2 sets up on the parameters and `σ`'s accepted plays bind the commitment to
one list of digests (`OpensOnce`). -/
theorem sampleNamed_run {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {A : Live} (hA : SetsUp V θ F A) {σ : Strategy (V.game θ F c)} (hO : OpensOnce V k θ F c σ)
    (x : CoinTape (V.game θ F c) σ) (hv : V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x)) :
    SampleNamed k (CoinTape.run (V.game θ F c) σ x).sent (labelAt θ) (afterAt V k θ F c σ) := by
  intro a b e p c' ev ξ dr hcase hacc
  have hS := hv.2.1
  obtain ⟨p₀, hp₀, -⟩ := hA.params
  have hpe : (CoinTape.run (V.game θ F c) σ x).sent.params = θ := (play_ends V θ F c σ x).1
  have hθ : ParamsAt (CoinTape.run (V.game θ F c) σ x).sent.params p₀ := by
    rw [hpe]
    exact hp₀
  refine ⟨(draw_label hS hcase hacc hθ).trans (congrArg labelAt hpe), ?_⟩
  obtain ⟨ds, hB, hnd, hmem⟩ := draw_after hS hcase hacc
  obtain ⟨x', ds', hv', hB', hmem'⟩ := afterAt_spec ⟨dr.after, x, ds, hv, hB, hmem⟩
  rw [hO x' x hv' hv ds' ds hB' hB] at hmem'
  exact (Prod.mk.inj (eq_of_ids hnd hmem hmem' rfl)).2

/-- **`movesReach_layout`'s `hN`** at the family `labelAt θ` and the digest `afterAt`, where `σ`'s accepted plays bind
the commitment to one list of digests (`OpensOnce`); it reads nothing of the logged reading. -/
theorem sampleNamed_layout {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray}
    {F : Array ByteArray} {c : ByteArray} {A : Live} (hA : SetsUp V θ F A) {σ : Strategy (V.game θ F c)}
    (hO : OpensOnce V k θ F c σ) :
    ∀ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
        SampleNamed k (CoinTape.run (V.game θ F c) σ x).sent (labelAt θ) (afterAt V k θ F c σ) :=
  fun x hv _ => sampleNamed_run hA hO x hv

end Verity.Sampled
