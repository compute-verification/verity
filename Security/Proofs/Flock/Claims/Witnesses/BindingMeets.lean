import Proofs.Flock.Claims.Sampling.Deployed

/-!
# Binding's antecedent at the deployed sampling record

The sampling `Binding` reads two openings that pass. At the deployed record (`SamplingDeploy.deploy rel c`) whether one
passes depends on the code: the commitment's addressing of a position (`at_`), its registered ports (`own`) and the path
an opening's data carries (`π`) are fields of `SamplingDeploy.Code`, which the tree has no concrete instance of, and a
code that addresses nothing opens nothing (`Binding_dead`). So the antecedent is met at the code's executable check:

* `Binding_meets`: **at every code, at SHA-512, an opening passes exactly when C-Flock's own registered check
  (`Flock.Registered.opensLeaf`) accepts** its hm96 leaf at the port whose root the code addresses, at the leaf index it
  names, through the path the opening carries. The antecedent is the executable verifier's own opening check, no
  weaker or stronger.
* `Binding_meets_one`: an opening passes at every code that registers one port of one leaf at the root that opening's
  row climbs to, and addresses its leaf `0` there with an empty path (`opensLeaf_one`: the executable check accepts
  there).
* `Binding_dead`: the probe that the record alone meets nothing: at a code whose `at_` addresses no position, no opening
  passes at any `H`.
-/

namespace Claims.Sampling

open FlockSoundness.Refine FlockSoundness.Binding
open Properties.NetworkCertifier.Bytes (bs)

theorem find?_bs_root (own : Array _root_.Flock.Registered.Port) (root : Msg) :
    own.find? (fun p => decide (bs p.root = root)) = own.find? fun p => decide (p.root = bytes root) := by
  congr 1
  funext p
  exact decide_eq_decide.2 ⟨fun h => by rw [← h, bytes_bs], fun h => by rw [h, bs_bytes]⟩

/-- **Binding's antecedent at the deployed record**: at every code, at SHA-512, an opening passes exactly when C-Flock's
registered check accepts its hm96 leaf at the port whose root the code addresses, at the leaf index it names, through
the path the opening carries. -/
theorem Binding_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : SamplingDeploy.Code)
    (roots : List Msg) (o : Opening Bool) :
    (SamplingDeploy.deploy rel c).commit.Opens sha512Msg roots o = true ↔
      ∃ root i reg, c.at_ roots o.pos = some (root, i) ∧
        c.own.find? (fun p => decide (p.root = bytes root)) = some reg ∧
        _root_.Flock.Registered.opensLeaf reg
          (bytes (digBytes (H512 (hm96Input (SamplingDeploy.row c.place c.bits o) (c.salt o))))) i (bytes (c.π o)) =
          true := by
  rw [SamplingDeploy.deploy_commit]
  unfold CommitmentScheme.Opens
  simp only [pathCommit]
  cases hat : c.at_ roots o.pos with
  | none => simp [OC.eval]
  | some ri =>
    obtain ⟨root, i⟩ := ri
    simp only [ocBind_eval, hm96Leaf_sha512]
    rw [regPath_sha512, find?_bs_root]
    constructor
    · rintro ⟨reg, h1, h2⟩
      exact ⟨root, i, reg, rfl, h1, h2⟩
    · rintro ⟨root', i', reg, h0, h1, h2⟩
      cases h0
      exact ⟨reg, h1, h2⟩

/-- C-Flock's registered check accepts a row at the one leaf of a port registered at the root that row's leaf climbs to
with no siblings. -/
theorem opensLeaf_one (dom payload : ByteArray) :
    _root_.Flock.Registered.opensLeaf
      { name := "", value := "",
        root := _root_.Flock.FV3.Sha512.climb dom (_root_.Flock.Registered.rowLeaf dom 0 payload) 0 0 (bytes []),
        leaves := 1, words := none, wordBits := 16, domain := dom, readsRel := #[] } payload 0 (bytes []) = true := by
  have h1 : _root_.Flock.log2ceil 1 = 0 := rfl
  simp [_root_.Flock.Registered.opensLeaf, _root_.Flock.FV3.Sha512.leafAs, _root_.Flock.Registered.rowLeaf, h1,
    size_bytes]

/-- **An opening passes at the deployed record**: at every code that registers one port of one leaf, at the root the
opening's row climbs to with no siblings, and addresses that root's leaf `0` with an empty path. -/
theorem Binding_meets_one (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (c : SamplingDeploy.Code) (roots : List Msg) (o : Opening Bool) (root : Msg) (dom : ByteArray)
    (hat : c.at_ roots o.pos = some (root, 0)) (hπ : c.π o = [])
    (hroot : bytes root = _root_.Flock.FV3.Sha512.climb dom (_root_.Flock.Registered.rowLeaf dom 0
      (bytes (digBytes (H512 (hm96Input (SamplingDeploy.row c.place c.bits o) (c.salt o)))))) 0 0 (bytes []))
    (hown : c.own =
      #[{ name := "", value := "", root := bytes root, leaves := 1, words := none, wordBits := 16, domain := dom,
          readsRel := #[] }]) :
    (SamplingDeploy.deploy rel c).commit.Opens sha512Msg roots o = true := by
  refine (Binding_meets rel c roots o).2
    ⟨root, 0,
      { name := "", value := "", root := bytes root, leaves := 1, words := none, wordBits := 16, domain := dom,
        readsRel := #[] }, hat, by rw [hown]; simp, ?_⟩
  rw [hπ, hroot]
  exact opensLeaf_one dom _

/-- **Probe: the record alone meets nothing.** At a code whose `at_` addresses no position, no opening passes at any
`H`, so `Binding` holds there with nothing to bind. -/
theorem Binding_dead (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : SamplingDeploy.Code)
    (hat : ∀ roots pos, c.at_ roots pos = none) (H : Msg → Msg) (roots : List Msg) (o : Opening Bool) :
    (SamplingDeploy.deploy rel c).commit.Opens H roots o = false := by
  rw [SamplingDeploy.deploy_commit]
  unfold CommitmentScheme.Opens
  simp [pathCommit, hat, OC.eval]

end Claims.Sampling
