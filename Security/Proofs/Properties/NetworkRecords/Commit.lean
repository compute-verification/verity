import Proofs.Properties.NetworkRecords.Program

/-!
# A committed record's digest binds its frames

`committedDigest` is the hex of SHA-256 of `recordInput` (`committedDigest_ok`), which writes the header and each row's
32-byte SHA-256 digest of `rowInput`, which writes the row's frames. So two grids of `B`-byte frames with one digest,
where SHA-256 has no collision on the strings the two digests hash, have the same frames row by row (`frames_eq`), and a
committed record whose digest is the Program's grid's commits to it (`commitsTo`).
-/

open NetworkCheck hiding Params
open NetworkCheck.Check (Params Grid)
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj split_bytes be32_size be64_size error_ne_ok)

set_option maxRecDepth 20000

namespace Properties.NetworkRecords.Commit

open Bytes

theorem sha256_size (x : ByteArray) : (Flock.Sha256.hash x).size = 32 := by
  unfold Flock.Sha256.hash
  exact Array.size_ofFn

theorem foldl_bs {α : Type} (F : α → ByteArray) : ∀ (l : List α) (init : ByteArray),
    bs (l.foldl (fun acc a => acc ++ F a) init) = bs init ++ (l.map fun a => bs (F a)).flatten
  | [], init => by simp
  | a :: l, init => by
    rw [List.foldl_cons, foldl_bs F l, bs_append]
    simp

theorem afoldl_bs {α : Type} (F : α → ByteArray) (arr : Array α) (init : ByteArray) :
    bs (arr.foldl (fun acc a => acc ++ F a) init) = bs init ++ (arr.toList.map fun a => bs (F a)).flatten := by
  rw [← Array.foldl_toList, foldl_bs]

theorem map_inj_of {α β γ : Type} {f : α → β} {g : α → γ} :
    ∀ {L L' : List α}, L.map f = L'.map f → (∀ a ∈ L, ∀ b ∈ L', f a = f b → g a = g b) → L.map g = L'.map g
  | [], [], _, _ => rfl
  | [], _ :: _, e, _ | _ :: _, [], e, _ => by cases e
  | a :: L, b :: L', e, h => by
    simp only [List.map_cons, List.cons.injEq] at e ⊢
    exact ⟨h a (List.mem_cons_self ..) b (List.mem_cons_self ..) e.1,
      map_inj_of e.2 fun x hx y hy => h x (List.mem_cons_of_mem _ hx) y (List.mem_cons_of_mem _ hy)⟩

/-- `taggedInput` under one tag writes its payload whole. -/
theorem taggedInput_inj {tag : String} {a b : ByteArray} (h : taggedInput tag a = taggedInput tag b) : a = b := by
  unfold taggedInput at h
  have e := congrArg bs h
  simp only [bs_append, List.append_assoc, List.append_cancel_left_eq] at e
  obtain ⟨-, e''⟩ := split_bytes e (by rw [be64_size, be64_size])
  exact bs_inj e''

theorem header_def (l : String) (w : Nat) (g : Grid) : header l w g = Flock.canon (Lean.Json.mkObj [
    ("link", .str l), ("window", Lean.toJson w), ("buckets", Lean.toJson g.buckets),
    ("allocation", Lean.toJson g.allocation), ("frame_bytes", Lean.toJson g.frameBytes),
    ("bucket_ms", Lean.toJson g.bucketMs), ("status", .str "complete")]) := rfl

theorem committedDigest_ok {l : String} {w : Nat} {grid : Records.RawGrid} {g : Grid} {dg : String}
    (h : Records.committedDigest l w grid g = .ok dg) :
    ∃ hd, header l w g = .ok hd ∧ dg = Flock.hexOf (Flock.Sha256.hash (recordInput hd grid)) := by
  unfold Records.committedDigest at h
  rw [← header_def] at h
  cases hh : header l w g with
  | error _ => rw [hh] at h; exact (error_ne_ok h).elim
  | ok hd => rw [hh] at h; exact ⟨hd, rfl, (Except.ok.inj h).symm⟩

/-- The frames of one row with one `rowInput`, where every frame is `B ≥ 1` bytes, are one. -/
theorem rowInput_inj {B : Nat} (hB : 0 < B) {a b : Array (Option ByteArray)}
    (ha : ∀ f ∈ (a.filterMap id).toList, f.size = B) (hb : ∀ f ∈ (b.filterMap id).toList, f.size = B)
    (h : rowInput a = rowInput b) : (a.filterMap id).toList = (b.filterMap id).toList := by
  have e := congrArg bs (taggedInput_inj h)
  have hid : ∀ x : Array ByteArray, ∀ i : ByteArray, x.foldl (· ++ ·) i = x.foldl (fun acc f => acc ++ id f) i :=
    fun _ _ => rfl
  rw [hid, hid, afoldl_bs, afoldl_bs] at e
  obtain ⟨-, e'⟩ := split_bytes e (by rw [be32_size, be32_size])
  have e'' := flatten_inj hB (by simpa [bs_length] using ha) (by simpa [bs_length] using hb) e'
  exact List.map_injective_iff.mpr (fun x y h => bs_inj h) e''

/-- **Two grids with one record digest have one grid's frames**, where every frame of both is `B ≥ 1` bytes and
SHA-256 has no collision on the strings the two digests hash. -/
theorem frames_eq {l : String} {w : Nat} {g : Grid} {G P : Records.RawGrid} {dg : String}
    (hB : 0 < g.frameBytes) (hG : ∀ row ∈ G.toList, ∀ f ∈ (row.filterMap id).toList, f.size = g.frameBytes)
    (hP : ∀ row ∈ P.toList, ∀ f ∈ (row.filterMap id).toList, f.size = g.frameBytes)
    (eG : Records.committedDigest l w G g = .ok dg) (eP : Records.committedDigest l w P g = .ok dg)
    (hX : ∀ x ∈ committedInputs l w G g ++ committedInputs l w P g, ∀ y ∈ committedInputs l w G g ++
      committedInputs l w P g, Flock.Sha256.hash x = Flock.Sha256.hash y → x = y) :
    frames G = frames P := by
  obtain ⟨hd, hh, e0⟩ := committedDigest_ok eG
  obtain ⟨hd', hh', e⟩ := committedDigest_ok eP
  obtain rfl : hd = hd' := Except.ok.inj (hh.symm.trans hh')
  have e := e0.symm.trans e
  have mem : ∀ grid, recordInput hd grid ∈ committedInputs l w grid g := fun grid => by
    unfold committedInputs
    rw [hh]
    exact List.mem_append_right _ (List.mem_singleton_self _)
  have hr := hX _ (List.mem_append_left _ (mem G)) _ (List.mem_append_right _ (mem P)) (hexOf_inj e)
  have e1 := congrArg bs (taggedInput_inj hr)
  rw [afoldl_bs, afoldl_bs] at e1
  have e2 := flatten_inj (k := 32) (by decide) (by simp [bs_length, Records.rowDigest, Records.taggedSha256,
    sha256_size]) (by simp [bs_length, Records.rowDigest, Records.taggedSha256, sha256_size])
    (List.append_cancel_left e1)
  unfold frames
  refine map_inj_of e2 fun a ha b hb hab => ?_
  have hrow : ∀ grid, ∀ r ∈ grid.toList, rowInput r ∈ committedInputs l w grid g := fun grid r hr => by
    simp only [committedInputs, List.mem_append, List.mem_map]
    exact Or.inl ⟨r, hr, rfl⟩
  have hi := hX _ (List.mem_append_left _ (hrow G a ha)) _ (List.mem_append_right _ (hrow P b hb)) (bs_inj hab)
  exact rowInput_inj hB (hG a ha) (hP b hb) hi

/-- A committed record whose digest is the Program's grid's commits to it, where that grid's frames are `B` bytes. -/
theorem commitsTo {p : Params} {l : String} {w : Nat} {dg : String} {P : Records.RawGrid}
    (eP : Records.committedDigest l w P p.grid = .ok dg)
    (hP : ∀ row ∈ P.toList, ∀ f ∈ (row.filterMap id).toList, f.size = p.grid.frameBytes) : CommitsTo p l w dg P := by
  refine ⟨eP, fun G hG eG => ?_⟩
  by_cases hc : ∃ x ∈ committedInputs l w G p.grid ++ committedInputs l w P p.grid,
      ∃ y ∈ committedInputs l w G p.grid ++ committedInputs l w P p.grid,
        x ≠ y ∧ Flock.Sha256.hash x = Flock.Sha256.hash y
  · exact Or.inr hc
  · refine Or.inl (frames_eq hG.1 hG.2 hP eG eP fun x hx y hy he => ?_)
    by_contra hne
    exact hc ⟨x, hx, y, hy, hne, he⟩

end Properties.NetworkRecords.Commit
