module

public import Proofs.Properties.Flock.Nets.Sha0
public import Proofs.Properties.Flock.Nets.Sha1
public import Proofs.Properties.Flock.Nets.Sha2
public import Proofs.Properties.Flock.Nets.Hm1
public import Proofs.Properties.Flock.Nets.Hm3

/-!
# The two slot circuits' runs

`sha512x3`'s three compressions run from bit 4608 to bit 248814, each row they add reading at most 128 bits, and output
their certified words concatenated (`seg_shaOuts`), each bit reading at most 16 bits (`shaOuts_width`). `hm96`'s tail
after its inputs runs from bit 3072 to bit 244562 the same way and outputs `b ‖ c` (`seg_hmTail`, `hmOut`): a bit of `c`
reads at most 16 bits, and a bit of `b`, its digest bit masked by up to 1536 salt bits, at most 1552 (`hmOut_width`).
-/

@[expose] public section

open Flock FlockSoundness.Discharge.Hm
open Flock.HmNets hiding and

namespace FlockDeploy.Concrete.Nets

namespace Seg

variable {α β : Type} {W : Nat}

theorem mapM_nil (f : β → M α) (n : Nat) : Seg W (([] : List β).mapM f) n [] n := by
  rw [List.mapM_nil]
  exact Seg.pure' _ _

theorem mapM_cons {f : β → M α} {b : β} {l : List β} {n n₁ n₂ : Nat} {a : α} {as : List α}
    (h₁ : Seg W (f b) n a n₁) (h₂ : Seg W (l.mapM f) n₁ as n₂) : Seg W ((b :: l).mapM f) n (a :: as) n₂ := by
  rw [List.mapM_cons]
  exact Seg.bind' h₁ (Seg.bind' h₂ (Seg.pure' _ _))

end Seg

/-- **`sha512x3`'s three compressions run from bit 4608 to bit 248814**, adding rows of at most 128 bits each. -/
theorem seg_shaOuts : Seg 128 (shaOuts sha3Ins) 4608
    [(ffOut sha0Cv sha0).flatten, (ffOut sha1Cv sha1).flatten, (ffOut sha2Cv sha2).flatten] 248814 := by
  unfold shaOuts sha3Ins
  refine Seg.mapM_cons (n₁ := 86010) ?_ (Seg.mapM_cons (n₁ := 167412) ?_ (Seg.mapM_cons (n₁ := 248814) ?_
    (Seg.mapM_nil _ _)))
  -- Unifying `vars 0 512` with a pair's projection unfolds `vars` past the recursion limit; reduce the pairs first.
  all_goals dsimp only
  · exact Seg.bind' seg_sha0 (Seg.pure' _ _)
  · exact Seg.bind' seg_sha1 (Seg.pure' _ _)
  · exact Seg.bind' seg_sha2 (Seg.pure' _ _)

theorem flatten_width {ws : List (List Lin)} (h : ∀ w ∈ ws, ∀ f ∈ w, f.s.length ≤ 16) :
    ∀ f ∈ ws.flatten, f.s.length ≤ 16 := fun f hf =>
  let ⟨w, hw, hfw⟩ := List.mem_flatten.1 hf
  h w hw f hfw

theorem shaOuts_width : ∀ o ∈ [(ffOut sha0Cv sha0).flatten, (ffOut sha1Cv sha1).flatten, (ffOut sha2Cv sha2).flatten],
    ∀ f ∈ o, f.s.length ≤ 16 :=
  List.forall_mem_cons.2 ⟨flatten_width sha0_out, List.forall_mem_cons.2 ⟨flatten_width sha1_out,
    List.forall_mem_cons.2 ⟨flatten_width sha2_out, fun _ h => by cases h⟩⟩⟩

/-- `hm96`'s output `b ‖ c` from its certified compressions. -/
noncomputable def hmOut (key : ByteArray) : List Lin :=
  (List.range 512).map (fun i => maskBit key (vars 1536 1536) i ((ffOut hm1Cv hm1).flatten.getD (wordBit i) Lin.zero)) ++
    (List.range 512).map fun t => (ffOut hm3Cv hm3).flatten.getD (wordBit t) Lin.zero

/-- **`hm96`'s tail runs from bit 3072 to bit 244562**, adding rows of at most 128 bits each. -/
theorem seg_hmTail (key : ByteArray) : Seg 128 (hmTail key (vars 0 512) (vars 512 1024) (vars 1536 1536)) 3072
    (vars 0 512, vars 512 1024, vars 1536 1536, hmOut key) 244562 := by
  have h2 : Seg 128 (compress (saltMid.map constW) ((saltWords (vars 1536 1536)).take 16)) 84474 (ffOut hm2Cv hm2)
      165291 := by
    rw [saltMid_eq_lit]
    exact seg_hm2
  have h3 : Seg 128 (compress (ffOut hm2Cv hm2) ((saltWords (vars 1536 1536)).drop 16 ++ saltTail.map constW)) 165291
      (ffOut hm3Cv hm3) 244562 := by
    rw [← ffOutK_eq hm2Cv hm2]
    exact seg_hm3
  exact Seg.bind' seg_hm1 (Seg.bind' h2 (Seg.bind' h3 (Seg.pure' _ _)))

theorem length_symm : ∀ (a b : List Var), (HmNets.symm a b).length ≤ a.length + b.length
  | [], ys => by simp [HmNets.symm]
  | x :: xs, [] => by simp [HmNets.symm]
  | x :: xs, y :: ys => by
    unfold HmNets.symm
    split
    · have := length_symm xs (y :: ys)
      simp only [List.length_cons] at this ⊢
      omega
    · split
      · have := length_symm (x :: xs) ys
        simp only [List.length_cons] at this ⊢
        omega
      · have := length_symm xs ys
        simp only [List.length_cons]
        omega

theorem vars_width (k n : Nat) : ∀ f ∈ vars k n, f.s.length ≤ 1 := by
  intro f hf
  obtain ⟨i, -, rfl⟩ := List.mem_map.1 hf
  simp [Lin.var]

/-- A masked bit reads at most its digest bit's bits and 1536 more, one per salt bit. -/
theorem maskBit_width (key : ByteArray) {y : List Lin} (hy : ∀ f ∈ y, f.s.length ≤ 1) (i : Nat) (x : Lin) :
    (maskBit key y i x).s.length ≤ x.s.length + 1536 := by
  have hget : ∀ j, (y.toArray.getD j Lin.zero).s.length ≤ 1 := fun j => by
    rw [Array.getD_eq_getD_getElem?, List.getElem?_toArray]
    cases h : y[j]? with
    | none => simp [Lin.zero]
    | some f => exact hy f (List.mem_of_getElem? h)
  have key : ∀ l : List Nat, (l.foldr (fun j acc => if Hm96.bit key (i + j) then y.toArray.getD j Lin.zero ^^^ acc
      else acc) x).s.length ≤ x.s.length + l.length := by
    intro l
    induction l with
    | nil => simp
    | cons j l ih =>
      rw [List.foldr_cons]
      split
      · refine Nat.le_trans (length_symm _ _) ?_
        have := hget j
        rw [List.length_cons]
        omega
      · rw [List.length_cons]
        omega
  have := key (List.range 1536)
  rw [List.length_range] at this
  exact this

theorem getD_width {ws : List (List Lin)} (h : ∀ w ∈ ws, ∀ f ∈ w, f.s.length ≤ 16) (k : Nat) :
    (ws.flatten.getD k Lin.zero).s.length ≤ 16 := by
  rw [List.getD_eq_getElem?_getD]
  cases hk : ws.flatten[k]? with
  | none => simp [Lin.zero]
  | some f =>
    obtain ⟨w, hw, hf⟩ := List.mem_flatten.1 (List.mem_of_getElem? hk)
    exact h w hw f hf

/-- **Each bit of `hm96`'s output reads at most 1552 bits.** -/
theorem hmOut_width (key : ByteArray) : ∀ f ∈ hmOut key, f.s.length ≤ 1552 := by
  intro f hf
  rcases List.mem_append.1 hf with hf | hf
  · obtain ⟨i, -, rfl⟩ := List.mem_map.1 hf
    have := maskBit_width key (vars_width 1536 1536) i ((ffOut hm1Cv hm1).flatten.getD (wordBit i) Lin.zero)
    have := getD_width hm1_out (wordBit i)
    omega
  · obtain ⟨t, -, rfl⟩ := List.mem_map.1 hf
    have := getD_width hm3_out (wordBit t)
    omega

end FlockDeploy.Concrete.Nets
