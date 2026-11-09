module

public import Definitions.NetworkCertifier.Guarantees.PoolRoots
public import Definitions.NetworkCertifier.Witness

/-!
# The nontrivial lemma of the opened rows' roots

`RootsOfChecked`. With nothing checking the roots, one run of no steps whose rows are empty meets its leak relation at
the empty rows, while the empty pool's row is `[true]`: a hash that tells the empty row's outer message from every
other gives the two rows different digests (`digest_empty_ne`), so `CarriedRoots` fails there.

Its honest lemma is open: it needs a deployed check that holds each run's opened and left rows to their registered
roots in the circuit (`CheckedRoots`).
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling

/-- **The empty row's digest is no other row's** under a hash that gives the empty row's outer message, `Q(0)`, a digest
no other message has. -/
theorem digest_empty_ne {G : Hash.Msg → Hash.Digest}
    (hG : ∀ m, m ≠ Hash.RowSeg.outerPrefix 0 → G m ≠ G (Hash.RowSeg.outerPrefix 0)) :
    Hash.RowSeg.digest G [] ≠ Hash.RowSeg.digest G [true] := by
  have h0 : Hash.RowSeg.outerMsg G [] = Hash.RowSeg.outerPrefix 0 := by
    simp [Hash.RowSeg.outerMsg, Hash.RowSeg.segmentMsgs, Hash.RowSeg.segCount, Hash.RowSeg.SEG_BITS]
  have hlen : (Hash.RowSeg.outerPrefix 1).length = (Hash.RowSeg.outerPrefix 0).length := by
    simp [Hash.RowSeg.outerPrefix, Hash.padTo, Hash.be]
    omega
  have h1 : Hash.RowSeg.outerMsg G [true] ≠ Hash.RowSeg.outerPrefix 0 := by
    intro he
    have := congrArg List.length he
    simp [Hash.RowSeg.outerMsg, Hash.RowSeg.segmentMsgs, Hash.RowSeg.segCount, Hash.RowSeg.SEG_BITS, hlen,
      Verity.Digest.bytes] at this
  unfold Hash.RowSeg.digest
  rw [h0]
  exact fun h => hG _ h1 h.symm

/-- **`RootsOfChecked`'s conclusion fails where nothing checks the roots**: one run of no steps, its rows empty, meets
its leak relation at the empty rows, while the empty pool's row is `[true]`, whose digest differs under a hash that gives
the empty row's outer message its own digest. -/
theorem RootsOfChecked_nontrivial :
    ¬ CarriedRoots (K := 1)
      (fun m => if m = Hash.RowSeg.outerPrefix 0 then (fun _ => 0) else (fun _ => 1)) (fun _ => notLay)
      (fun _ => 0) (fun _ => ∅) (fun _ _ => True) (fun _ => []) (fun _ => []) 1 (fun _ : Fin 1 => notStmt)
      (fun _ => notPart) (fun _ _ => false) (fun _ _ _ _ => []) (fun _ => false) (fun _ => [true]) := by
  intro h
  have hleak : ∀ i : Fin 1, LeakAtCarried (K := 1) (fun _ => notLay) (fun _ => 0) (fun _ => ∅) (fun _ _ => True)
      (fun _ => []) (fun _ => []) 1 notStmt notPart (fun _ => false) (fun _ _ => []) [] [] ∅ [] := fun _ =>
    ⟨⟨fun t ht => absurd ht (Nat.not_lt_zero t), fun t ht => absurd ht (Nat.not_lt_zero t),
      fun t ht => absurd ht (Nat.not_lt_zero t)⟩, Finset.disjoint_empty_left _, rfl, fun _ hy => by simp at hy,
      fun _ => false, by simp [AsReg], fun _ => none, fun _ _ hb => by simp at hb, trivial, rfl, rfl,
      fun t ht => absurd ht (Nat.not_lt_zero t)⟩
  have he := h (fun _ => []) (fun _ => ∅) (fun _ => []) (fun _ => []) hleak 0
  have hopen : poolOpen (fun _ : Fin 1 => false) (fun _ => [true]) (fun _ => []) 0 = [true] := by
    simp [poolOpen]
  rw [hopen] at he
  refine digest_empty_ne (fun m hm hG => ?_) he
  have := congrFun hG 0
  simp [hm] at this

end NetTiming.Witness
