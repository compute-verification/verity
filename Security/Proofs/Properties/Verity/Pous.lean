import Properties.Verity.Game
import Properties.Pous.EndToEndAt
import Proofs.Properties.Composition
import Proofs.Core.Game.Prob

/-!
# PoUS's leg of the one theorem, at its component

What `Vocab.PousSound` reduces to at PoUS's compiled component (`PousCheck.RunProtocol.protocol`), proved from the
composition's properties and the compiled check alone, with no premise:

- `target_of_split` (`pousSound_of_split` at PoUS): a subprotocol's target follows from three bounds wherever its
  finder meets its collision premise, the verdict with no break, no forgery and a failed claim at most its published
  `δ`, the verdict with a forgery at most its forgery term, and the verdict with a break at most its loss at the
  finder's collision probability.
- `componentAccepts_pous` (`verdict_pous`): PoUS's verdict is onsite's, and its case decodes to parameters, setup
  outputs and evidence on which the verifier's whole decision at the case's point accepts (`Pous.EndToEnd.AcceptedAt`),
  the event `Pous.Period.PeriodHeldAt` bounds.
- `digestOf_fixed` and `pous_digest_fixed`: two plays of one composite on one parameter string, one commitment and one
  blob, both accepted as `same` compositions, carry one digest of each component's commitment at each index, PoUS's
  setup outputs' included, or the composite's `bind` hashed two different messages to one SHA-512
  (`Properties.Composition.bindCollision`). It is the composition's half of the fork form: the setup outputs PoUS's run
  is checked against are fixed with the commitment, before any coin.
- `error_eq_top`, `target_at_of_delta_top`: at parameters whose published `δ` is `⊤` the target holds, so a discharge
  only has to bound the parameters it publishes a finite `δ` for; and `Pous.EndToEnd.acceptedAt_deployed`: at
  `Run.DEPLOYED` the decision `AcceptedAt` is `Pous.EndToEnd.Accepted`, the one `Pous.Period.PeriodHeld` reads.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **A target from its three parts**: wherever subprotocol `a`'s finder meets its collision premise, its verdict with
no collision break `brk`, no forgery `forged` and a failed claim has probability at most its published `δ` at the
strategy's SHA-512 calls, its verdict with a forgery at most its forgery term there, and its verdict with a break at
most its loss at the finder's collision probability. Elsewhere its error is `⊤`. -/
theorem target_of_split (k : Verity.Registration.Composite) (a : Subprotocol)
    (brk forged : ∀ θ F c : ByteArray, Game.Strategy (V.game θ F c) → Exchange → Prop)
    (hgood : ∀ θ F c σ, V.FinderCR a θ F c σ →
      Game.prob (fun o => V.Verdict k a o ∧ ¬ brk θ F c σ o ∧ ¬ forged θ F c σ o ∧ ¬ V.Holds k θ F c σ a o)
        (V.game θ F c) σ ≤ V.delta θ a (V.cost θ F c σ))
    (hforged : ∀ θ F c σ, V.FinderCR a θ F c σ →
      Game.prob (fun o => V.Verdict k a o ∧ forged θ F c σ o) (V.game θ F c) σ ≤ V.forgery θ a (V.cost θ F c σ))
    (hbrk : ∀ θ F c σ, V.FinderCR a θ F c σ →
      Game.prob (fun o => V.Verdict k a o ∧ brk θ F c σ o) (V.game θ F c) σ ≤
        V.loss a θ (ENNReal.ofReal ((V.finder a θ F c σ).Pr sha512Msg))) :
    V.Target k a := by
  intro θ F c σ
  unfold Vocab.error condError
  split_ifs with hcr
  · calc Game.prob (fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ
        ≤ Game.prob (fun o => ((V.Verdict k a o ∧ ¬ brk θ F c σ o ∧ ¬ forged θ F c σ o ∧ ¬ V.Holds k θ F c σ a o) ∨
            (V.Verdict k a o ∧ forged θ F c σ o)) ∨ (V.Verdict k a o ∧ brk θ F c σ o)) (V.game θ F c) σ :=
          Game.prob_mono (fun o ⟨hv, hh⟩ => (em (brk θ F c σ o)).elim (fun hb => Or.inr ⟨hv, hb⟩)
            (fun hb => Or.inl ((em (forged θ F c σ o)).elim (fun hf => Or.inr ⟨hv, hf⟩)
              (fun hf => Or.inl ⟨hv, hb, hf, hh⟩)))) _ σ
      _ ≤ _ := Game.prob_or_le _ _ _ σ
      _ ≤ _ := add_le_add (Game.prob_or_le _ _ _ σ) le_rfl
      _ ≤ _ := add_le_add (add_le_add (hgood θ F c σ hcr) (hforged θ F c σ hcr)) (hbrk θ F c σ hcr)
  · exact le_top

/-- **PoUS's target from its three parts** (`target_of_split` at PoUS). -/
theorem pousSound_of_split (k : Verity.Registration.Composite)
    (brk forged : ∀ θ F c : ByteArray, Game.Strategy (V.game θ F c) → Exchange → Prop)
    (hgood : ∀ θ F c σ, V.FinderCR .pous θ F c σ →
      Game.prob (fun o => V.Verdict k .pous o ∧ ¬ brk θ F c σ o ∧ ¬ forged θ F c σ o ∧ ¬ V.Holds k θ F c σ .pous o)
        (V.game θ F c) σ ≤ V.delta θ .pous (V.cost θ F c σ))
    (hforged : ∀ θ F c σ, V.FinderCR .pous θ F c σ →
      Game.prob (fun o => V.Verdict k .pous o ∧ forged θ F c σ o) (V.game θ F c) σ ≤
        V.forgery θ .pous (V.cost θ F c σ))
    (hbrk : ∀ θ F c σ, V.FinderCR .pous θ F c σ →
      Game.prob (fun o => V.Verdict k .pous o ∧ brk θ F c σ o) (V.game θ F c) σ ≤
        V.loss .pous θ (ENNReal.ofReal ((V.finder .pous θ F c σ).Pr sha512Msg))) :
    V.PousSound k :=
  V.target_of_split k .pous brk forged hgood hforged hbrk

/-- A subprotocol's error is `⊤` at a strategy where its published `δ` is. -/
theorem error_eq_top {a : Subprotocol} {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (h : V.delta θ a (V.cost θ F c σ) = ⊤) : V.error a θ F c σ = ⊤ := by
  unfold Vocab.error condError
  split_ifs
  · simp only [h, top_add]
  · rfl

/-- **The target holds at parameters whose published `δ` is `⊤`**, at every committed file, commitment and
strategy. -/
theorem target_at_of_delta_top (k : Verity.Registration.Composite) (a : Subprotocol) (θ : ByteArray)
    (h : ∀ n, V.delta θ a n = ⊤) (F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) :
    Game.prob (fun o => V.Verdict k a o ∧ ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ ≤ V.error a θ F c σ := by
  rw [V.error_eq_top (h _)]
  exact le_top

end Vocab

/-- **PoUS's component accepts only at onsite, on a case the verifier's whole decision accepts**: the three byte
strings decode, PoUS's triple among onsite's components is `PousCheck.RunProtocol.protocol`'s, its parameters,
commitment and evidence decode to `p`, `pub` and `e`, and setup accepts `e`'s `C*` against its commitment and publishes
`pub` while the composed check at `p`'s point accepts the run (`Pous.EndToEnd.AcceptedAt`). -/
theorem componentAccepts_pous {k : Verity.Registration.Composite} {s : Sent} (h : ComponentAccepts k .pous s) :
    k = .onsite ∧ ∃ ps c ev, Decoded s ps c ev ∧
      ∃ x ∈ Properties.Composition.triples (s.components k) ps ev, x.1 = PousCheck.RunProtocol.protocol ∧
        ∃ (p : PousCheck.RunProtocol.Params) (pub : PousCheck.SetupProtocol.Public)
          (e : PousCheck.RunProtocol.Evidence),
          (canonicalJson x.2.1.params.toUTF8 >>= PousCheck.RunProtocol.decodeParams) = .ok p ∧
          PousCheck.RunProtocol.decodeCommitment x.2.2.commitment = .ok pub ∧
          PousCheck.RunProtocol.decodeEvidence x.2.2.evidence = .ok e ∧
          Pous.EndToEnd.AcceptedAt p.point p.trusted.toList e.setup.commitment e.setup.store pub e.digests e.session
            p.sessionId := by
  obtain ⟨P, hP, ps, c, ev, hdec, ⟨Q, n, cs⟩, hx, hx1, hacc⟩ := h
  cases k with
  | remote => simp [component] at hP
  | onsite =>
    have hP' : (PousCheck.RunProtocol.protocol : BlobProtocol) = P := Option.some.inj hP
    subst hP'
    have hQ : Q = PousCheck.RunProtocol.protocol := hx1
    subst hQ
    obtain ⟨p, pub, e, hp, hc, he, hA⟩ := hacc
    obtain ⟨h1, h2⟩ := (PousCheck.RunProtocol.accepts_iff _ _ _ _ _ _ _ _ _).1
      ((PousCheck.RunProtocol.protocol_accepts p pub e).1 hA)
    exact ⟨rfl, ps, c, ev, hdec, _, hx, rfl, p, pub, e, hp, hc, he, h1, h2⟩

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **PoUS's verdict** is onsite's, on a case the verifier's whole decision accepts (`componentAccepts_pous`). -/
theorem verdict_pous {k : Verity.Registration.Composite} {o : Exchange} (h : V.Verdict k .pous o) :
    k = .onsite ∧ ∃ ps c ev, Decoded o.sent ps c ev ∧
      ∃ x ∈ Properties.Composition.triples (o.sent.components k) ps ev, x.1 = PousCheck.RunProtocol.protocol ∧
        ∃ (p : PousCheck.RunProtocol.Params) (pub : PousCheck.SetupProtocol.Public)
          (e : PousCheck.RunProtocol.Evidence),
          (canonicalJson x.2.1.params.toUTF8 >>= PousCheck.RunProtocol.decodeParams) = .ok p ∧
          PousCheck.RunProtocol.decodeCommitment x.2.2.commitment = .ok pub ∧
          PousCheck.RunProtocol.decodeEvidence x.2.2.evidence = .ok e ∧
          Pous.EndToEnd.AcceptedAt p.point p.trusted.toList e.setup.commitment e.setup.store pub e.digests e.session
            p.sessionId :=
  componentAccepts_pous h.2.2

end Vocab

/-- The composition's formats decode one parameter string, one commitment and one evidence string one way. -/
theorem decoded_unique {s s' : Sent} (hp : s.params = s'.params) (hc : s.commitment = s'.commitment)
    {ps ps' : List Composition.Named} {c c' : ByteArray} {ev ev' : List Composition.Case}
    (h : Decoded s ps c ev) (h' : Decoded s' ps' c' ev') : ps = ps' ∧ c = c' := by
  obtain ⟨h1, h2, -⟩ := h
  obtain ⟨h1', h2', -⟩ := h'
  rw [← hp] at h1'
  rw [← hc] at h2'
  exact ⟨Except.ok.inj (h1.symm.trans h1'), Except.ok.inj (h2.symm.trans h2')⟩

/-- A list's `mapM` into `Option`, where it succeeds, is the function at each index. -/
theorem mapM_getElem? {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {ys : List β}, l.mapM f = some ys → ∀ (i : ℕ) (x : α), l[i]? = some x → f x = ys[i]?
  | [], _, _, _, _, hx => by simp at hx
  | a :: l, ys, h, i, x, hx => by
    cases ha : f a with
    | none => simp [List.mapM_cons, ha] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [List.mapM_cons, ha, hl] at h
      | some zs =>
        simp only [List.mapM_cons, ha, hl, Option.pure_def, Option.bind_eq_bind, Option.bind_some,
          Option.some.injEq] at h
        subst h
        cases i with
        | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at hx
          subst hx
          simpa using ha
        | succ i =>
          simpa using mapM_getElem? hl i x (by simpa using hx)

/-- **One commitment carries one digest at each index, or `bind` collides**: two plays of composite `k` on one
parameter string, one commitment and one blob, both accepted as `same` compositions, have at each index `i` components
whose case commitments have one digest, or the composite's `bind` hashed two different messages
(`Properties.Composition.bindMessage`) to one SHA-512. -/
theorem digestOf_fixed {k : Verity.Registration.Composite} {s s' : Sent} (hp : s.params = s'.params)
    (hc : s.commitment = s'.commitment) (h : SameAccepts k s) (h' : SameAccepts k s')
    {ps : List Composition.Named} {c : ByteArray} {ev : List Composition.Case} (hd : Decoded s ps c ev)
    {ps' : List Composition.Named} {c' : ByteArray} {ev' : List Composition.Case} (hd' : Decoded s' ps' c' ev')
    (i : ℕ) {x x' : Protocol × Composition.Named × Composition.Case}
    (hx : (Properties.Composition.triples (s.components k) ps ev)[i]? = some x)
    (hx' : (Properties.Composition.triples (s'.components k) ps' ev')[i]? = some x') :
    Properties.Composition.digestOf x.1 x.2.2.commitment = Properties.Composition.digestOf x'.1 x'.2.2.commitment ∨
      ∃ ds ds', Properties.Composition.bindMessage (verifier k).id ds ≠
          Properties.Composition.bindMessage (verifier k).id ds' ∧
        Flock.Sha512.hash (Properties.Composition.bindMessage (verifier k).id ds) =
          Flock.Sha512.hash (Properties.Composition.bindMessage (verifier k).id ds') := by
  obtain ⟨ps1, c1, ev1, hd1, pub, prof, ⟨⟨-, -, hbind, -⟩, -⟩, -⟩ := h
  obtain ⟨ps2, c2, ev2, hd2, pub', prof', ⟨⟨-, -, hbind', -⟩, -⟩, -⟩ := h'
  obtain ⟨hps1, hc1⟩ := decoded_unique rfl rfl hd1 hd
  obtain ⟨hps2, hc2⟩ := decoded_unique rfl rfl hd2 hd'
  have hev1 : ev1 = ev := Except.ok.inj (hd1.2.2.symm.trans hd.2.2)
  have hev2 : ev2 = ev' := Except.ok.inj (hd2.2.2.symm.trans hd'.2.2)
  have hcc : c = c' := (decoded_unique hp hc hd hd').2
  rw [hps1, hev1] at hbind
  rw [hps2, hev2] at hbind'
  cases hds : Properties.Composition.digests (s.components k) ps ev with
  | none => simp [hds] at hbind
  | some ds =>
    cases hds' : Properties.Composition.digests (s'.components k) ps' ev' with
    | none => simp [hds'] at hbind'
    | some ds' =>
      simp only [hds, Option.map_some, Option.some.injEq] at hbind
      simp only [hds', Option.map_some, Option.some.injEq] at hbind'
      rcases Properties.Composition.bindCollision (verifier k).id ds ds' with hne | rfl | hcol
      · exact absurd (hbind.trans (hc1.trans (hcc.trans (hc2.symm.trans hbind'.symm)))) hne
      · left
        unfold Properties.Composition.digests at hds hds'
        have e := (mapM_getElem? hds i x hx).trans (mapM_getElem? hds' i x' hx').symm
        have e' := congrArg (Option.map Prod.snd) e
        simpa [Option.map_map, Function.comp_def] using e'
      · exact Or.inr ⟨ds, ds', hcol⟩

/-- **PoUS's setup outputs are fixed with onsite's commitment, or `bind` collides**: in two plays at onsite on one
parameter string, one commitment and one blob, both accepted as `same` compositions, where PoUS's case at index `i` of
the first decodes its commitment to the setup outputs `pub`, the second has PoUS's case at the same index, decoding
its commitment to setup outputs `pub'` with PoUS's digest of `pub`, or onsite's `bind` hashed two different messages to
one SHA-512. -/
theorem pous_digest_fixed {s s' : Sent} (hp : s.params = s'.params) (hc : s.commitment = s'.commitment)
    (hb : s.blobs = s'.blobs) (h : SameAccepts .onsite s) (h' : SameAccepts .onsite s')
    {ps : List Composition.Named} {c : ByteArray} {ev : List Composition.Case} (hd : Decoded s ps c ev)
    {i : ℕ} {x : Protocol × Composition.Named × Composition.Case}
    (hx : (Properties.Composition.triples (s.components .onsite) ps ev)[i]? = some x)
    (hP : x.1 = PousCheck.RunProtocol.protocol) {pub : PousCheck.SetupProtocol.Public}
    (hpub : PousCheck.RunProtocol.decodeCommitment x.2.2.commitment = .ok pub) :
    ∃ ps' c' ev', Decoded s' ps' c' ev' ∧ ∃ x', (Properties.Composition.triples (s'.components .onsite) ps' ev')[i]? =
        some x' ∧ x'.1 = PousCheck.RunProtocol.protocol ∧ ∃ pub' : PousCheck.SetupProtocol.Public,
      PousCheck.RunProtocol.decodeCommitment x'.2.2.commitment = .ok pub' ∧
      (PousCheck.RunProtocol.digest pub = PousCheck.RunProtocol.digest pub' ∨
        ∃ ds ds', Properties.Composition.bindMessage (verifier .onsite).id ds ≠
            Properties.Composition.bindMessage (verifier .onsite).id ds' ∧
          Flock.Sha512.hash (Properties.Composition.bindMessage (verifier .onsite).id ds) =
            Flock.Sha512.hash (Properties.Composition.bindMessage (verifier .onsite).id ds')) := by
  have h'' := h'
  obtain ⟨ps', c', ev', hd', pub2, prof2, ⟨⟨⟨-, -, hfst', -⟩, hall', -⟩, -⟩, -⟩ := h''
  have hcs : s'.components .onsite = s.components .onsite := by
    unfold Sent.components
    rw [hb]
  have hx1 : (s.components .onsite)[i]? = some x.1 := (List.getElem?_zip_eq_some.1 hx).1
  have hm : ((Properties.Composition.triples (s'.components .onsite) ps' ev').map (·.1))[i]? = some x.1 := by
    rw [hfst', hcs, hx1]
  rw [List.getElem?_map] at hm
  cases hx' : (Properties.Composition.triples (s'.components .onsite) ps' ev')[i]? with
  | none => simp [hx'] at hm
  | some x' =>
    rw [hx'] at hm
    have hx'1 : x'.1 = x.1 := Option.some.inj hm
    have hmem := List.mem_of_getElem? hx'
    obtain ⟨Q', n', k'⟩ := x'
    have hQ' : Q' = PousCheck.RunProtocol.protocol := hx'1.trans hP
    subst hQ'
    obtain ⟨p', (pub' : PousCheck.SetupProtocol.Public), e', -, hc', -, -⟩ := hall' _ hmem
    refine ⟨ps', c', ev', hd', _, hx', rfl, pub', hc', ?_⟩
    rcases digestOf_fixed hp hc h h' hd hd' i hx hx' with he | hcol
    · left
      obtain ⟨Q, n, k⟩ := x
      have hQ : Q = PousCheck.RunProtocol.protocol := hP
      subst hQ
      have h1 : Properties.Composition.digestOf PousCheck.RunProtocol.protocol k.commitment =
          some (PousCheck.RunProtocol.digest pub) := by
        unfold Properties.Composition.digestOf
        have hpub' : PousCheck.RunProtocol.protocol.commitment k.commitment = .ok pub := hpub
        rw [hpub']
        rfl
      have h2 : Properties.Composition.digestOf PousCheck.RunProtocol.protocol k'.commitment =
          some (PousCheck.RunProtocol.digest pub') := by
        unfold Properties.Composition.digestOf
        have hc'' : PousCheck.RunProtocol.protocol.commitment k'.commitment = .ok pub' := hc'
        rw [hc'']
        rfl
      exact Option.some.inj (h1.symm.trans (he.trans h2))
    · exact Or.inr hcol

end Verity

/-- **At `Run.DEPLOYED` the verifier's whole decision is `Pous.EndToEnd.Accepted`**, the decision
`Pous.Period.PeriodHeld` reads: `Run.check` is `Run.checkAt` at `Run.DEPLOYED`, and the deployed point's parameters are
`q`. -/
theorem Pous.EndToEnd.acceptedAt_deployed (trusted : List ByteArray) (c : PousCheck.SetupProtocol.Commitment)
    (C : Array (Array ByteArray)) (pub : PousCheck.SetupProtocol.Public) (digests : ByteArray)
    (r : PousCheck.Run.Session) (sid : ByteArray) :
    Pous.EndToEnd.AcceptedAt PousCheck.Run.DEPLOYED trusted c C pub digests r sid ↔
      Pous.EndToEnd.Accepted trusted c C pub digests r sid :=
  Iff.rfl
