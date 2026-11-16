module

public import Proofs.Properties.Verity.SampledTwo.Majority
public import Proofs.Hash.RowBinding
public import Proofs.Accounting.ReadUnit.Sha

/-!
# Two readings of a bound wire are a SHA-512 collision

`ReadsBind` (`SampledTwo.Majority`) asks for a finder `collOf` that, given two witnesses in the relation that both bind
a unit holding the wire `g` and read `g` differently, outputs a SHA-512 collision. This module gives one and proves it
(`readsBind_of`) where each witness carries the row-leaf opening its reading of each placed wire rests on (`OpensBy`).

* **The openings** (`OpensBy`): where a witness's transcript opens the wire `g`, which is not a constant and sits at
  `p`, `openOf` returns an opening the row-leaf check accepts at `p` whose row holds the reading's bits at `p.offset`.
* **The finder** (`collOfOpen`): a collision among the messages both witnesses' openings of every wire hash
  (`hashedAll`), where one exists.
* **The proof** (`readsBind_of`): a constant wire opens only to its constant, so both readings agree on it. At a placed
  wire the two openings are accepted at one root and rank, and row binding (`Hash.SecurityProofs.RowBinding`) gives
  equal rows, which read equal bits, or a collision in what the two checks hashed.
* **The bound from what the instance owes** (`majorityBound_opens`): `majorityBound_ecr` with `HoldsOnWitness` from
  `holdsOnWitness_of` and `ReadsBind` from `readsBind_of`.

**What `OpensBy` does and does not assume.** For every `readX` some `openOf` meets `OpensBy` (`exists_opensBy`):
`Opens` says an accepted opening exists, and choice picks one. So `OpensBy` restricts no transcript. What it fixes is the `openOf` the finder
reads, and the bound means something only where that `openOf` is computed from the witness: `ecr/sha-512` at the
binding finder (`ExpectedCR`) prices the SHA-512 calls of an algorithm, and a finder that reads openings picked by
choice is none. The finder's pair is among `hashedAll` (`collOfOpen_mem`), the messages the row-leaf checks hash on
both witnesses' openings of every wire, so a `dColl` that prices it is at least the calls that compute `openOf` on
both witnesses plus `hashedAll`'s length.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- `Verity.sha512`'s bytes are `FlockDeploy.sha512` of the same message. -/
theorem sha512_bytes (m : Msg) : (Verity.sha512 m).bytes = FlockDeploy.sha512 m :=
  Accounting.ReadUnit.Gates.isSha512_sha512 m

/-- **The witness carries its openings**: where `readX w` opens the wire `g`, which is not a constant and sits at `p`,
`openOf w g` is an opening the row-leaf check accepts at `p` whose row holds the reading's bits from `p.offset`.
Some `openOf` meets it at every `readX` (by choice), so it is a condition on the `openOf` the finder reads, which the
instance takes from the witness. -/
def OpensBy (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (openOf : Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening) : Prop :=
  ∀ w (g : Fin (V.circuitOf θ F).N) (p : Place), (V.read θ F).consts g = none → V.place θ F c g = some p →
    V.Opens θ F c g (readX w g) → ∃ o, openOf w g = some o ∧
      Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 p.verifier p.root p.rank o ∧
      ∀ i : Fin ((V.circuitOf θ F).w g), o.row[p.offset + i.val]? = some ((readX w g).getLsbD i.val)

/-- **`OpensBy` restricts no transcript**: at every `readX` some `openOf` meets it, picked by choice among the openings
`Opens` says exist. Such an `openOf` is not computed from the witness, so `ExpectedCR` at its finder is no reading of
`ecr/sha-512`; the instance's `openOf` is the witness's own openings. -/
theorem exists_opensBy (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)) :
    ∃ openOf, OpensBy V θ F c readX openOf := by
  classical
  refine ⟨fun w g =>
    if h : ∃ p, V.place θ F c g = some p ∧ ∃ o : Hash.RowLeaf.Opening,
        Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 p.verifier p.root p.rank o ∧
          ∀ i : Fin ((V.circuitOf θ F).w g), o.row[p.offset + i.val]? = some ((readX w g).getLsbD i.val)
    then some h.choose_spec.2.choose else none, ?_⟩
  intro w g p hc hp ho
  have h : ∃ p, V.place θ F c g = some p ∧ ∃ o : Hash.RowLeaf.Opening,
      Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 p.verifier p.root p.rank o ∧
        ∀ i : Fin ((V.circuitOf θ F).w g), o.row[p.offset + i.val]? = some ((readX w g).getLsbD i.val) := by
    have h' := ho
    simp only [Vocab.Opens, hc] at h'
    exact h'
  have hp' : h.choose = p := Option.some.inj (h.choose_spec.1.symm.trans hp)
  refine ⟨h.choose_spec.2.choose, dite_eq_left h, ?_, ?_⟩
  · rw [← hp']; exact h.choose_spec.2.choose_spec.1
  · rw [← hp']; exact h.choose_spec.2.choose_spec.2

/-- The messages the row-leaf check hashes on `w`'s opening of the wire `g`: none where `g` has no place or no opening. -/
def hashedOf (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (openOf : Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening)
    (w : Claims.Sampling.Wit) (g : Fin (V.circuitOf θ F).N) : List Msg :=
  match V.place θ F c g with
  | none => []
  | some p =>
    match openOf w g with
    | none => []
    | some o => Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512 p.verifier p.rank o ++
        Hash.RowLeaf.innerHashed Verity.sha512 p.verifier o

/-- The messages both witnesses' openings of every wire hash. -/
noncomputable def hashedAll (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (openOf : Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening)
    (w₁ w₂ : Claims.Sampling.Wit) : List Msg :=
  (List.finRange (V.circuitOf θ F).N).flatMap fun g =>
    hashedOf V θ F c openOf w₁ g ++ hashedOf V θ F c openOf w₂ g

open Classical in
/-- **The finder**: a SHA-512 collision among `hashedAll`, where one exists. -/
noncomputable def collOfOpen (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (openOf : Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening)
    (w₁ w₂ : Claims.Sampling.Wit) : Option (Msg × Msg) :=
  if h : ∃ m ∈ hashedAll V θ F c openOf w₁ w₂, ∃ m' ∈ hashedAll V θ F c openOf w₁ w₂,
      Claims.Collision FlockDeploy.sha512 m m' then
    some (h.choose, h.choose_spec.2.choose)
  else none

/-- **The finder's pair is among the messages it hashes**: what `collOfOpen` outputs is a SHA-512 collision of two
messages of `hashedAll`. -/
theorem collOfOpen_mem (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (openOf : Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening)
    (w₁ w₂ : Claims.Sampling.Wit) (m m' : Msg) (h : collOfOpen V θ F c openOf w₁ w₂ = some (m, m')) :
    m ∈ hashedAll V θ F c openOf w₁ w₂ ∧ m' ∈ hashedAll V θ F c openOf w₁ w₂ ∧
      Claims.Collision FlockDeploy.sha512 m m' := by
  unfold collOfOpen at h
  split at h
  next hc =>
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
    exact ⟨hc.choose_spec.1, hc.choose_spec.2.choose_spec.1, hc.choose_spec.2.choose_spec.2⟩
  next => cases h

/-- **Two readings of a bound wire are a collision**: where the witnesses carry their openings (`OpensBy`), the finder
`collOfOpen` meets `ReadsBind`. -/
theorem readsBind_of (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (openOf : Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening)
    (hO : OpensBy V θ F c readX openOf) : ReadsBind V A x θ F c readX (collOfOpen V θ F c openOf) := by
  classical
  rintro w₁ w₂ g - - ⟨u₁, hu₁, hg₁⟩ ⟨u₂, hu₂, hg₂⟩ hne
  have hopen : ∀ (w : Claims.Sampling.Wit) u,
      u ∉ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w) → g ∈ (V.partsOf θ F).io u →
      V.Opens θ F c g (readX w g) := by
    intro w u hu hg
    simp only [Accounting.unbound, Finset.mem_filter, Finset.mem_univ, true_and, not_or, not_exists, not_and,
      not_not] at hu
    exact hu.2 g hg
  have ho₁ := hopen w₁ u₁ hu₁ hg₁
  have ho₂ := hopen w₂ u₂ hu₂ hg₂
  have hbits : (∀ i : Fin ((V.circuitOf θ F).w g), (readX w₁ g).getLsbD i.val = (readX w₂ g).getLsbD i.val) →
      False := fun h => hne (BitVec.eq_of_getLsbD_eq fun i hi => h ⟨i, hi⟩)
  cases hc : (V.read θ F).consts g with
  | some b =>
    have h₁ := ho₁
    have h₂ := ho₂
    simp only [Vocab.Opens, hc] at h₁ h₂
    exact (hbits fun i => (h₁ i).trans (h₂ i).symm).elim
  | none =>
    obtain ⟨p, hp⟩ : ∃ p, V.place θ F c g = some p := by
      have h := ho₁
      simp only [Vocab.Opens, hc] at h
      obtain ⟨p, hp, -⟩ := h
      exact ⟨p, hp⟩
    obtain ⟨o₁, hw₁, ha₁, hr₁⟩ := hO w₁ g p hc hp ho₁
    obtain ⟨o₂, hw₂, ha₂, hr₂⟩ := hO w₂ g p hc hp ho₂
    have hh₁ : hashedOf V θ F c openOf w₁ g =
        Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512 p.verifier p.rank o₁ ++
          Hash.RowLeaf.innerHashed Verity.sha512 p.verifier o₁ := by
      simp only [hashedOf, hp, hw₁]
    have hh₂ : hashedOf V θ F c openOf w₂ g =
        Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512 p.verifier p.rank o₂ ++
          Hash.RowLeaf.innerHashed Verity.sha512 p.verifier o₂ := by
      simp only [hashedOf, hp, hw₂]
    have hmem : ∀ m,
        m ∈ Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512 p.verifier p.rank o₁ ++
            Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512 p.verifier p.rank o₂ ∨
          m ∈ Hash.RowLeaf.innerHashed Verity.sha512 p.verifier o₁ ++
            Hash.RowLeaf.innerHashed Verity.sha512 p.verifier o₂ →
        m ∈ hashedAll V θ F c openOf w₁ w₂ := by
      intro m hm
      simp only [hashedAll, List.mem_flatMap, List.mem_finRange, true_and]
      refine ⟨g, ?_⟩
      rw [hh₁, hh₂]
      simp only [List.mem_append] at hm ⊢
      tauto
    have hcol : ∃ m ∈ hashedAll V θ F c openOf w₁ w₂, ∃ m' ∈ hashedAll V θ F c openOf w₁ w₂,
        Claims.Collision FlockDeploy.sha512 m m' := by
      rcases Hash.SecurityProofs.RowBinding Verity.sha512 Verity.sha512 p.verifier p.root p.rank o₁ o₂ ha₁ ha₂ with
        hrow | ⟨m, hm, m', hm', hne', heq⟩ | ⟨m, hm, m', hm', hne', heq⟩
      · refine (hbits fun i => ?_).elim
        have h₂ := hr₂ i
        rw [← hrow] at h₂
        exact Option.some.inj ((hr₁ i).symm.trans h₂)
      · refine ⟨m, hmem m (Or.inl hm), m', hmem m' (Or.inl hm'), ?_⟩
        have he : FlockDeploy.sha512 m = FlockDeploy.sha512 m' := by
          rw [← sha512_bytes, ← sha512_bytes]
          exact heq
        exact (And.intro hne' he : Claims.Collision FlockDeploy.sha512 m m')
      · refine ⟨m, hmem m (Or.inr hm), m', hmem m' (Or.inr hm'), ?_⟩
        have he : FlockDeploy.sha512 m = FlockDeploy.sha512 m' := by
          rw [← sha512_bytes, ← sha512_bytes]
          exact heq
        exact (And.intro hne' he : Claims.Collision FlockDeploy.sha512 m m')
    unfold collOfOpen
    rw [dite_eq_left hcol]
    exact ⟨_, _, rfl, hcol.choose_spec.2.choose_spec.2⟩

/-- **The witness fixed at registration, from what the instance owes**: where the vocabulary's transcript is the
majority (`hX`), the statement parses with the session's partitions, the units are the replay blocks (`hU`), the design
caps the claim's share (`hcap`), the witnesses carry their openings (`hO`) and the binding finder at `collOfOpen` meets
`ExpectedCR` (`hbind`), an accepted run of the frame's prover whose extracted witness is in the session's relation,
where the claim fails on the vocabulary's transcript, is at most `√(min(W, 2n) · E[cost] / 2^256)` likely. -/
theorem majorityBound_opens (V : Vocab d α) (A : Live₂) (x : Msg) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (Fr : Frame V A x θ F c σ)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (openOf : Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening) (dColl : ℕ)
    (hX : V.extract θ F c σ = majority V A x θ F c σ Fr readX)
    (hs : A.L.parse x = some s) (hP : A.L.partition s = some Pt) (hd : A.L.two s = some dd)
    (hU : UnitsAreReplay V θ F c FlockDeploy.sha512 A.L.commit s Pt dd readX)
    (hcap : CapsAt dd Pt ((V.certOf θ).ε * V.units θ F) A.δ) (hO : OpensBy V θ F c readX openOf)
    (hbind : Claims.ExpectedCR FlockDeploy.sha512
      (bindFinder V A x θ F c σ Fr (collOfOpen V θ F c openOf) dColl)) :
    MajorityBound V A x θ F c σ Fr (V.extract θ F c σ)
      (ENNReal.ofReal (Real.sqrt (min ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) *
        (Ex (fun p => ((bindFinder V A x θ F c σ Fr (collOfOpen V θ F c openOf) dColl).cost p : ℝ)) /
          2 ^ (256 : ℝ))))) :=
  majorityBound_ecr V A x θ F c σ Fr readX (collOfOpen V θ F c openOf) dColl hX
    (holdsOnWitness_of V A x θ F c s Pt dd hs hP hd readX hU hcap) (readsBind_of V A x θ F c readX openOf hO) hbind

end Verity.SampledTwo
