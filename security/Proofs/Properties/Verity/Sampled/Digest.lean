module

public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Properties.Verity.SampledRead
public import Proofs.Properties.Composition

/-!
# Sampled proofs' coupling: the digest v2's sample answers

v2 samples at the coins its challenger releases after the digest sampled proofs' case commits (`draw_after`). That digest
is the prover's: the commitment fixes it before any coin only as one entry of the list of digests `bind` hashes into it
(`BindsAt`, `Core.Composition.bind`), and the cases are sent at the close. So it is fixed where the commitment is, at
the play's start, by the SHA-512 of a message (`bindMessage`) the play reveals only at its close: a cut of the period's
game (`bindCut`, a `Verity.Game.DigestCut`) whose revealed messages have one SHA-512, since each play binds its list
into the one commitment (`bind_hash_eq`).

* **The clause** (`SampledDigest`, `afterAt`): v2 samples at `afterAt σ`, sampled proofs' entry in the list whose message
  is the one `σ`'s plays most likely reveal at the cut (`bindMode`). An accepted play that doesn't miss at the cut
  meets it (`sampledDigest_of_noMiss`).
* **The finder** (`bindFinder`, `Constructions.withBind`): the cut's two-continuation finder, `σ`'s play on two
  independent tapes from the start, outputting the two messages `bind` hashed, at `2t + 2h` SHA-512 calls on every coin
  (`Live.qM`, `bindFinder_calls`). Where the two lists differ, its output is a SHA-512 collision on them.
* **The bound** (`prob_not_sampledDigest_le`): an accepted play that misses the clause is at most as likely as the
  square root of the finder's success (`DigestCut.missP_le`); at `cr/sha-512` at most `√(q²/2^513)`
  (`prob_not_sampledDigest_le_strict`). It is the coupling's (a') at the clause (`digestBound_of`).
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Properties.Challenger (Draws)
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The digests the commitment binds -/

/-- **The digests what the prover sent binds**: the three byte strings decode under the composition's formats, and
`ds`, each of composite `k`'s components' ids with the digest of its case's commitment, in order, is what the
commitment binds (`Core.Composition.bind`). -/
def BindsAt (k : Verity.Registration.Composite) (s : Verity.Sent) (ds : List (String × ByteArray)) : Prop :=
  ∃ ps c ev, Verity.Decoded s ps c ev ∧ Properties.Composition.digests (s.components k) ps ev = some ds ∧
    Core.Composition.bind (Verity.verifier k).id ds = c

/-- What the prover sent binds at most one list: its three byte strings decode one way. -/
theorem BindsAt.unique {k : Verity.Registration.Composite} {s : Verity.Sent} {ds ds' : List (String × ByteArray)}
    (h : BindsAt k s ds) (h' : BindsAt k s ds') : ds = ds' := by
  obtain ⟨ps, _, ev, ⟨hps, -, hev⟩, hd, -⟩ := h
  obtain ⟨ps', _, ev', ⟨hps', -, hev'⟩, hd', -⟩ := h'
  rw [Except.ok.inj (hps'.symm.trans hps), Except.ok.inj (hev'.symm.trans hev)] at hd'
  exact Option.some.inj (hd.symm.trans hd')

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

/-- Each entry of a list an `Option` `mapM` succeeds on has its value among the values. -/
theorem mapM_entry {β γ : Type _} {f : β → Option γ} :
    ∀ {l : List β} {ys : List γ}, l.mapM f = some ys → ∀ x ∈ l, ∃ y ∈ ys, f x = some y
  | [], _, _, x, hx => absurd hx List.not_mem_nil
  | a :: l, ys, h, x, hx => by
    cases ha : f a with
    | none => simp [List.mapM_cons, ha] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [List.mapM_cons, ha, hl] at h
      | some zs =>
        simp only [List.mapM_cons, ha, hl, Option.pure_def, Option.bind_eq_bind, Option.bind_some,
          Option.some.injEq] at h
        subst h
        rcases List.mem_cons.1 hx with rfl | hx
        · exact ⟨b, List.mem_cons.2 (Or.inl rfl), ha⟩
        · obtain ⟨y, hy, e⟩ := mapM_entry hl x hx
          exact ⟨y, List.mem_cons.2 (Or.inr hy), e⟩

/-- **v2 samples at the digest sampled proofs' case commits**: where composite `k` accepted what the prover sent and
v2's case in it accepts at the sample `dr`, the digests the commitment binds (`BindsAt`), keyed by distinct ids, give
sampled proofs `dr`'s digest. -/
theorem draw_after {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hS : Verity.SameAccepts k s)
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
  · obtain ⟨y, hy, hfy⟩ := mapM_entry hds' x hx
    have hdg : Properties.Composition.digestOf x.1 x.2.2.commitment = some c.digest := by
      rw [hx1]
      show Properties.Composition.digestOf _root_.Spot.OneStage.protocol x.2.2.commitment = _
      unfold Properties.Composition.digestOf
      rw [show _root_.Spot.OneStage.protocol.commitment x.2.2.commitment = .ok c from hc]
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

/-- **Two lists bound into one commitment have one SHA-512 message digest**: `bind` of both is the commitment the
composition decodes, so SHA-512 of their two messages is one, or the lists are one
(`Properties.Composition.bindCollision`). Where the lists differ, the two messages are a SHA-512 collision. -/
theorem bind_hash_eq {k : Verity.Registration.Composite} {s s' : Verity.Sent} (hc : s.commitment = s'.commitment)
    {ds ds' : List (String × ByteArray)} (hB : BindsAt k s ds) (hB' : BindsAt k s' ds') :
    Flock.Sha512.hash (Properties.Composition.bindMessage (Verity.verifier k).id ds) =
      Flock.Sha512.hash (Properties.Composition.bindMessage (Verity.verifier k).id ds') := by
  obtain ⟨_, c₀, _, ⟨-, hc0, -⟩, -, hb⟩ := hB
  obtain ⟨_, c₀', _, ⟨-, hc0', -⟩, -, hb'⟩ := hB'
  rw [hc] at hc0
  have hcc : c₀' = c₀ := Except.ok.inj (hc0'.symm.trans hc0)
  rcases Properties.Composition.bindCollision (Verity.verifier k).id ds ds' with h | h | h
  · exact (h (by rw [hb, hb', hcc])).elim
  · rw [h]
  · exact h.2

/-! ## The commitment's cut -/

/-- **The message `bind` hashes over the digests `ds`**, as a message a cut reveals. -/
def bindMsg (k : Verity.Registration.Composite) (ds : List (String × ByteArray)) : Claims.Msg :=
  (Properties.Composition.bindMessage (Verity.verifier k).id ds).data.toList

/-- Two lists bound into one commitment with one message are one. -/
theorem eq_of_bindMsg {k : Verity.Registration.Composite} {s s' : Verity.Sent} (hc : s.commitment = s'.commitment)
    {ds ds' : List (String × ByteArray)} (hB : BindsAt k s ds) (hB' : BindsAt k s' ds')
    (h : bindMsg k ds = bindMsg k ds') : ds = ds' := by
  obtain ⟨_, c₀, _, ⟨-, hc0, -⟩, -, hb⟩ := hB
  obtain ⟨_, c₀', _, ⟨-, hc0', -⟩, -, hb'⟩ := hB'
  rw [hc] at hc0
  have hcc : c₀' = c₀ := Except.ok.inj (hc0'.symm.trans hc0)
  rcases Properties.Composition.bindCollision (Verity.verifier k).id ds ds' with h₁ | h₁ | h₁
  · exact (h₁ (by rw [hb, hb', hcc])).elim
  · exact h₁
  · exact (h₁.1 (ByteArray.ext (Array.ext' h))).elim

theorem sha512Msg_bytes (b : ByteArray) : Verity.sha512Msg b.data.toList = (Flock.Sha512.hash b).data.toList :=
  rfl

open Classical in
/-- **What a play reveals for the commitment `c`**: where what the prover sent has the commitment `c` and binds a list
of digests (`BindsAt`), the message `bind` hashed over it (`bindMsg`); else none. -/
noncomputable def bindReveal (k : Verity.Registration.Composite) (c : ByteArray) (o : Exchange) : Option Claims.Msg :=
  if h : o.sent.commitment = c ∧ ∃ ds, BindsAt k o.sent ds then some (bindMsg k h.2.choose) else none

theorem bindReveal_of {k : Verity.Registration.Composite} {c : ByteArray} {o : Exchange}
    {ds : List (String × ByteArray)} (hc : o.sent.commitment = c) (hB : BindsAt k o.sent ds) :
    bindReveal k c o = some (bindMsg k ds) := by
  have h : o.sent.commitment = c ∧ ∃ ds, BindsAt k o.sent ds := ⟨hc, ds, hB⟩
  rw [bindReveal, dite_eq_left h, BindsAt.unique h.2.choose_spec hB]

theorem bindReveal_some {k : Verity.Registration.Composite} {c : ByteArray} {o : Exchange} {w : Claims.Msg}
    (h : bindReveal k c o = some w) :
    o.sent.commitment = c ∧ ∃ ds, BindsAt k o.sent ds ∧ w = bindMsg k ds := by
  by_cases hc : o.sent.commitment = c ∧ ∃ ds, BindsAt k o.sent ds
  · rw [bindReveal, dite_eq_left hc] at h
    exact ⟨hc.1, hc.2.choose, hc.2.choose_spec, (Option.some.inj h).symm⟩
  · rw [bindReveal, dite_eq_right hc] at h
    cases h

variable (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The commitment's cut**: at the play's start, where the commitment `c` is fixed, the message `bind` hashed over
the digests each play binds into it (`bindReveal`). -/
noncomputable def bindCut : DigestCut Claims.Msg (V.game θ F c) := .here (bindReveal k c)

/-- **The commitment's cut agrees under SHA-512**: each play binds its digests into the one commitment
(`bind_hash_eq`). -/
theorem bindCut_agrees : (bindCut V k θ F c).Agrees Verity.sha512Msg := fun _ _ _ _ h h' => by
  obtain ⟨hc, ds, hB, rfl⟩ := bindReveal_some h
  obtain ⟨hc', ds', hB', rfl⟩ := bindReveal_some h'
  rw [bindMsg, bindMsg, sha512Msg_bytes, sha512Msg_bytes, bind_hash_eq (hc.trans hc'.symm) hB hB']

/-- **The message `σ`'s plays most likely reveal at the commitment's cut** (`revealMode`). -/
noncomputable def bindMode (σ : Strategy (V.game θ F c)) : Claims.Msg :=
  revealMode (bindReveal k c) (V.game θ F c) σ

/-! ## The clause -/

/-- **v2 samples at the digest `after`**: in every v2 case what the prover sent composite `k` accepts, the sample answers
`after`. -/
def SampledDigest (s : Verity.Sent) (after : ByteArray) : Prop :=
  ∀ a b e p c ev ξ dr, V2Case k s a b e → V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr →
    dr.after = after

/-- **`dg` is sampled proofs' digest in an accepted play of `σ` whose message is the most likely one**: the sampled
verdict accepts the play at some tape, the message of the digests its commitment binds is `bindMode`, and those digests
give sampled proofs `dg`. -/
def DigestAt (σ : Strategy (V.game θ F c)) (dg : ByteArray) : Prop :=
  ∃ (x : CoinTape (V.game θ F c) σ) (ds : List (String × ByteArray)),
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ BindsAt k (CoinTape.run (V.game θ F c) σ x).sent ds ∧
      bindMsg k ds = bindMode V k θ F c σ ∧ (VStar.Recursive.Sampled.PROTOCOL, dg) ∈ ds

open Classical in
/-- **The sample's digest at `σ`**: sampled proofs' digest in one of `σ`'s accepted plays whose message is the most
likely one (`DigestAt`), or the empty string where there is none. -/
noncomputable def afterAt (σ : Strategy (V.game θ F c)) : ByteArray :=
  if h : ∃ dg, DigestAt V k θ F c σ dg then h.choose else ByteArray.empty

theorem afterAt_spec {σ : Strategy (V.game θ F c)} (h : ∃ dg, DigestAt V k θ F c σ dg) :
    DigestAt V k θ F c σ (afterAt V k θ F c σ) := by
  rw [afterAt, dite_eq_left h]
  exact h.choose_spec

/-- **An accepted play that doesn't miss at the commitment's cut samples at `afterAt`**: its message is the most likely
one, so its digests are those of `afterAt`'s play (`eq_of_bindMsg`), which give sampled proofs one digest. -/
theorem sampledDigest_of_noMiss (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ)
    (hv : V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x)) (hn : ¬ (bindCut V k θ F c).missOn σ x) :
    SampledDigest k (CoinTape.run (V.game θ F c) σ x).sent (afterAt V k θ F c σ) := by
  intro a b e p c' ev ξ dr hcase hacc
  obtain ⟨ds, hB, hnd, hmem⟩ := draw_after hv.2.1 hcase hacc
  have hr : bindReveal k c (CoinTape.run (V.game θ F c) σ x) = some (bindMsg k ds) :=
    bindReveal_of (play_commitment V θ F c σ x) hB
  have hm : bindMsg k ds = bindMode V k θ F c σ := Classical.byContradiction fun hne =>
    hn (show ∃ w, bindReveal k c (CoinTape.run (V.game θ F c) σ x) = some w ∧
      w ≠ revealMode (bindReveal k c) (V.game θ F c) σ from ⟨_, hr, hne⟩)
  obtain ⟨x', ds', -, hB', hm', hmem'⟩ := afterAt_spec V k θ F c ⟨dr.after, x, ds, hv, hB, hm, hmem⟩
  have hcc : (CoinTape.run (V.game θ F c) σ x').sent.commitment = (CoinTape.run (V.game θ F c) σ x).sent.commitment :=
    (play_commitment V θ F c σ x').trans (play_commitment V θ F c σ x).symm
  rw [eq_of_bindMsg hcc hB' hB (hm'.trans hm.symm)] at hmem'
  exact (Prod.mk.inj (eq_of_ids hnd hmem hmem' rfl)).2

/-! ## The finder and the bound -/

/-- **The clause's bound at a budget**: a play of the period's game whose sampled verdict accepts and whose sample doesn't
answer `afterAt σ` is at most as likely as the square root of the commitment's cut's two-continuation finder's success,
at any budget `q`. -/
theorem prob_not_sampledDigest_le (σ : Strategy (V.game θ F c)) (q : ℕ) :
    prob (fun o => V.Verdict k .sampled o ∧ ¬ SampledDigest k o.sent (afterAt V k θ F c σ)) (V.game θ F c) σ ≤
      ENNReal.ofReal (Real.sqrt (((bindCut V k θ F c).finder σ q).Pr Verity.sha512Msg)) := by
  refine le_trans ?_ (DigestCut.missP_le Verity.sha512Msg _ (bindCut_agrees V k θ F c) σ q)
  rw [prob_eq_ofReal_pr, DigestCut.missP_eq_ofReal]
  exact ENNReal.ofReal_le_ofReal (Claims.Pr_mono fun x hx => Classical.byContradiction fun hn =>
    hx.2 (sampledDigest_of_noMiss V k θ F c σ x hx.1 hn))

/-- **The clause's bound at `cr/sha-512`**: where the finder at `q` calls meets the strict birthday bound, at most
`√(q²/2^513)`. -/
theorem prob_not_sampledDigest_le_strict (σ : Strategy (V.game θ F c)) (q : ℕ)
    (hCR : Claims.StrictBirthday Verity.sha512Msg ((bindCut V k θ F c).finder σ q)) :
    prob (fun o => V.Verdict k .sampled o ∧ ¬ SampledDigest k o.sent (afterAt V k θ F c σ)) (V.game θ F c) σ ≤
      ENNReal.ofReal (Real.sqrt ((q : ℝ) ^ 2 / 2 ^ (513 : ℝ))) := by
  refine le_trans ?_ (DigestCut.missP_le_strict Verity.sha512Msg _ (bindCut_agrees V k θ F c) σ q hCR)
  rw [prob_eq_ofReal_pr, DigestCut.missP_eq_ofReal]
  exact ENNReal.ofReal_le_ofReal (Claims.Pr_mono fun x hx => Classical.byContradiction fun hn =>
    hx.2 (sampledDigest_of_noMiss V k θ F c σ x hx.1 hn))

omit θ F c

/-- **The bound digests' two-continuation finder**, at the session `A`: the commitment's cut's, `σ`'s play on two
independent tapes from the start, outputting the two messages `bind` hashed, at `2t + 2h` SHA-512 calls on every coin
(`Live.qM`), `t` the strategy's calls. -/
noncomputable def bindFinder (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) : Claims.Finder :=
  (bindCut V k θ F c).finder σ (2 * V.cost θ F c σ + 2 * A.h)

theorem bindFinder_calls : ∀ (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (x : (bindFinder V k A θ F c σ).S),
    ((bindFinder V k A θ F c σ).cost x : ℝ) ≤ A.qM (V.cost θ F c σ) := fun A θ F c σ _ => by
  show ((2 * V.cost θ F c σ + 2 * A.h : ℕ) : ℝ) ≤ A.qM (V.cost θ F c σ)
  simp [Live.qM]

/-- **The leg's constructions with the bound digests' finder at `k`** (`bindFinder`): `K`'s otherwise. -/
noncomputable def Constructions.withBind {V : Vocab d α} (K : Constructions V) (k : Verity.Registration.Composite) :
    Constructions V :=
  { K with bindFinder := Verity.Sampled.bindFinder V k, bind_calls := Verity.Sampled.bindFinder_calls V k }

/-- **(a') at the clause**: at constructions whose bound digests' finder is `bindFinder`, the coupling's digest bound
holds at the session `A` and the clause that v2 samples at `afterAt` (`DigestBound`). -/
theorem digestBound_of (K : Constructions V) (hK : K.bindFinder = bindFinder V k) (A : Live) (θ : ByteArray)
    (F : Array ByteArray) :
    DigestBound V k K A θ F fun c σ o => SampledDigest k o.sent (afterAt V k θ F c σ) := fun c σ => by
  rw [hK]
  exact prob_not_sampledDigest_le V k θ F c σ _

end Verity.Sampled
