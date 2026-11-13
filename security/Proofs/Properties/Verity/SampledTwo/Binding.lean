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

The finder's cost covers hashing `hashedAll`, which reads every wire's opening of both witnesses; a `dColl` that
bounds its collision probability under `ecr/sha-512` prices that hashing.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- `Verity.sha512`'s bytes are `FlockDeploy.sha512` of the same message. -/
theorem sha512_bytes (m : Msg) : (Verity.sha512 m).bytes = FlockDeploy.sha512 m :=
  Accounting.ReadUnit.Gates.isSha512_sha512 m

/-- **The witness carries its openings**: where `readX w` opens the wire `g`, which is not a constant and sits at `p`,
`openOf w g` is an opening the row-leaf check accepts at `p` whose row holds the reading's bits from `p.offset`. -/
def OpensBy (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (openOf : Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening) : Prop :=
  ∀ w (g : Fin (V.circuitOf θ F).N) (p : Place), (V.read θ F).consts g = none → V.place θ F c g = some p →
    V.Opens θ F c g (readX w g) → ∃ o, openOf w g = some o ∧
      Hash.RowLeaf.accepts Verity.sha512 Verity.sha512 p.verifier p.root p.rank o ∧
      ∀ i : Fin ((V.circuitOf θ F).w g), o.row[p.offset + i.val]? = some ((readX w g).getLsbD i.val)

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

end Verity.SampledTwo
