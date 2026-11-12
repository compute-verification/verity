module

public import Proofs.Properties.NetworkCertifier.Records.Program
public import Proofs.Properties.NetworkCertifier.Check.Check

/-!
# A committed record's digest binds its frames

`committedDigest` is the hex of SHA-512 of `recordInput` (`committedDigest_ok`), which writes the header and each row's
64-byte SHA-512 digest of `rowInput` (`rowDigest_ok`), which writes the row's frames; each refuses what `tagged_sha512`
refuses. So two grids of `B`-byte frames with one digest, where SHA-512 has no collision on the strings the two digests
hash, have the same frames row by row (`frames_eq`), and a committed record whose digest is the Program's grid's commits
to it (`commitsTo`).
-/

@[expose] public section

open NetworkCheck hiding Params
open NetworkCheck.Check (Params Grid)
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj split_bytes be32_size error_ne_ok hash_size
  taggedInput_inj)
open Properties.NetworkCertifier.CheckProofs (bind_ok)

set_option maxRecDepth 20000

namespace Properties.NetworkRecords.Commit

open Bytes

theorem foldl_bs {α : Type} (F : α → ByteArray) : ∀ (l : List α) (init : ByteArray),
    bs (l.foldl (fun acc a => acc ++ F a) init) = bs init ++ (l.map fun a => bs (F a)).flatten
  | [], init => by simp
  | a :: l, init => by
    rw [List.foldl_cons, foldl_bs F l, bs_append]
    simp

theorem afoldl_bs {α : Type} (F : α → ByteArray) (arr : Array α) (init : ByteArray) :
    bs (arr.foldl (fun acc a => acc ++ F a) init) = bs init ++ (arr.toList.map fun a => bs (F a)).flatten := by
  rw [← Array.foldl_toList, foldl_bs]

/-- A fold that appends each element's bytes, where each may refuse, refuses if one does; otherwise each gave bytes and
the result is `init` followed by them in order. -/
theorem foldlM_ok {α : Type} (F : α → Flock.V ByteArray) :
    ∀ (l : List α) (init rows : ByteArray),
      l.foldlM (fun acc a => do return acc ++ (← F a)) init = .ok rows →
        (∀ a ∈ l, F a = .ok ((F a).toOption.getD .empty)) ∧
          bs rows = bs init ++ (l.map fun a => bs ((F a).toOption.getD .empty)).flatten
  | [], init, rows, h => by
    have h' : init = rows := Except.ok.inj h
    subst h'
    simp
  | a :: l, init, rows, h => by
    simp only [List.foldlM_cons] at h
    cases hF : F a with
    | error e => rw [hF] at h; exact (error_ne_ok h).elim
    | ok d =>
      rw [hF] at h
      obtain ⟨h1, h2⟩ := foldlM_ok F l (init ++ d) rows h
      refine ⟨fun b hb => ?_, ?_⟩
      · rcases List.mem_cons.mp hb with rfl | hb
        · rw [hF]
          rfl
        · exact h1 b hb
      · rw [h2, bs_append]
        simp [hF, Except.toOption]

theorem afoldlM_ok {α : Type} (F : α → Flock.V ByteArray) (arr : Array α) (init rows : ByteArray)
    (h : arr.foldlM (fun acc a => do return acc ++ (← F a)) init = .ok rows) :
    (∀ a ∈ arr.toList, F a = .ok ((F a).toOption.getD .empty)) ∧
      bs rows = bs init ++ (arr.toList.map fun a => bs ((F a).toOption.getD .empty)).flatten :=
  foldlM_ok F arr.toList init rows (by rw [Array.foldlM_toList]; exact h)

theorem map_inj_of {α β γ : Type} {f : α → β} {g : α → γ} :
    ∀ {L L' : List α}, L.map f = L'.map f → (∀ a ∈ L, ∀ b ∈ L', f a = f b → g a = g b) → L.map g = L'.map g
  | [], [], _, _ => rfl
  | [], _ :: _, e, _ | _ :: _, [], e, _ => by cases e
  | a :: L, b :: L', e, h => by
    simp only [List.map_cons, List.cons.injEq] at e ⊢
    exact ⟨h a (List.mem_cons_self ..) b (List.mem_cons_self ..) e.1,
      map_inj_of e.2 fun x hx y hy => h x (List.mem_cons_of_mem _ hx) y (List.mem_cons_of_mem _ hy)⟩

theorem header_def (l : String) (w : Nat) (g : Grid) : header l w g = Flock.canonUtf8 (Lean.Json.mkObj [
    ("link", .str l), ("window", Lean.toJson w), ("buckets", Lean.toJson g.buckets),
    ("allocation", Lean.toJson g.allocation), ("frame_bytes", Lean.toJson g.frameBytes),
    ("bucket_ms", Lean.toJson g.bucketMs), ("status", .str "complete")]) := rfl

/-- A row's digest is SHA-512 of its `rowInput`. -/
theorem rowDigest_ok {row : Array (Option ByteArray)} {d : ByteArray} (h : Records.rowDigest row = .ok d) :
    ∃ x, rowInput row = .ok x ∧ d = Flock.Sha512.hash x := by
  unfold Records.rowDigest at h
  exact Properties.Identity.taggedSha512_ok.mp h

theorem committedDigest_ok {l : String} {w : Nat} {grid : Records.RawGrid} {g : Grid} {dg : String}
    (h : Records.committedDigest l w grid g = .ok dg) :
    ∃ hd x, header l w g = .ok hd ∧ recordInput hd grid = .ok x ∧ dg = Flock.hexOf (Flock.Sha512.hash x) := by
  unfold Records.committedDigest at h
  rw [← header_def] at h
  obtain ⟨hd, hh, h⟩ := bind_ok h
  obtain ⟨rows, hr, h⟩ := bind_ok h
  obtain ⟨d, hd', h⟩ := bind_ok h
  obtain ⟨x, hx, rfl⟩ := Properties.Identity.taggedSha512_ok.mp hd'
  refine ⟨hd, x, hh, ?_, (Except.ok.inj h).symm⟩
  unfold recordInput
  rw [hr]
  exact hx

/-- The frames of one row with one `rowInput`, where every frame is `B ≥ 1` bytes, are one. -/
theorem rowInput_inj {B : Nat} (hB : 0 < B) {a b : Array (Option ByteArray)} {x : ByteArray}
    (ha : ∀ f ∈ (a.filterMap id).toList, f.size = B) (hb : ∀ f ∈ (b.filterMap id).toList, f.size = B)
    (h : rowInput a = .ok x) (h' : rowInput b = .ok x) : (a.filterMap id).toList = (b.filterMap id).toList := by
  unfold rowInput at h h'
  have e := congrArg bs (taggedInput_inj h h')
  have hid : ∀ x : Array ByteArray, ∀ i : ByteArray, x.foldl (· ++ ·) i = x.foldl (fun acc f => acc ++ id f) i :=
    fun _ _ => rfl
  rw [hid, hid, afoldl_bs, afoldl_bs] at e
  obtain ⟨-, e'⟩ := split_bytes e (by rw [be32_size, be32_size])
  have e'' := flatten_inj hB (by simpa [bs_length] using ha) (by simpa [bs_length] using hb) e'
  exact List.map_injective_iff.mpr (fun x y h => bs_inj h) e''

/-- **Two grids with one record digest have one grid's frames**, where every frame of both is `B ≥ 1` bytes and
SHA-512 has no collision on the strings the two digests hash. -/
theorem frames_eq {l : String} {w : Nat} {g : Grid} {G P : Records.RawGrid} {dg : String}
    (hB : 0 < g.frameBytes) (hG : ∀ row ∈ G.toList, ∀ f ∈ (row.filterMap id).toList, f.size = g.frameBytes)
    (hP : ∀ row ∈ P.toList, ∀ f ∈ (row.filterMap id).toList, f.size = g.frameBytes)
    (eG : Records.committedDigest l w G g = .ok dg) (eP : Records.committedDigest l w P g = .ok dg)
    (hX : ∀ x ∈ committedInputs l w G g ++ committedInputs l w P g, ∀ y ∈ committedInputs l w G g ++
      committedInputs l w P g, Flock.Sha512.hash x = Flock.Sha512.hash y → x = y) :
    frames G = frames P := by
  obtain ⟨hd, xG, hh, hxG, e0⟩ := committedDigest_ok eG
  obtain ⟨hd', xP, hh', hxP, e⟩ := committedDigest_ok eP
  obtain rfl : hd = hd' := Except.ok.inj (hh.symm.trans hh')
  have mem : ∀ {grid x}, recordInput hd grid = .ok x → x ∈ committedInputs l w grid g := fun hx => by
    simp [committedInputs, hh, hx, Except.toOption]
  obtain rfl := hX _ (List.mem_append_left _ (mem hxG)) _ (List.mem_append_right _ (mem hxP))
    (hexOf_inj (e0.symm.trans e))
  unfold recordInput at hxG hxP
  obtain ⟨rG, hrG, hxG⟩ := bind_ok hxG
  obtain ⟨rP, hrP, hxP⟩ := bind_ok hxP
  obtain rfl := taggedInput_inj hxG hxP
  have ⟨okG, e1⟩ := afoldlM_ok Records.rowDigest G _ _ hrG
  have ⟨okP, e1'⟩ := afoldlM_ok Records.rowDigest P _ _ hrP
  have size : ∀ {grid : Records.RawGrid}, (∀ r ∈ grid.toList, Records.rowDigest r =
      .ok ((Records.rowDigest r).toOption.getD .empty)) →
      ∀ x ∈ grid.toList.map (fun r => bs ((Records.rowDigest r).toOption.getD .empty)), x.length = 64 :=
    fun ok x hx => by
      obtain ⟨r, hr, rfl⟩ := List.mem_map.mp hx
      obtain ⟨_, -, e⟩ := rowDigest_ok (ok r hr)
      rw [bs_length, e, hash_size]
  have e2 := flatten_inj (by decide) (size okG) (size okP) (List.append_cancel_left (e1.symm.trans e1'))
  unfold frames
  refine map_inj_of e2 fun a ha b hb hab => ?_
  obtain ⟨xa, hxa, ea⟩ := rowDigest_ok (okG a ha)
  obtain ⟨xb, hxb, eb⟩ := rowDigest_ok (okP b hb)
  have hrow : ∀ {grid r x}, r ∈ grid.toList → rowInput r = .ok x → x ∈ committedInputs l w grid g :=
    fun hr hx => List.mem_append_left _ (List.mem_filterMap.mpr ⟨_, hr, by rw [hx]; rfl⟩)
  obtain rfl := hX _ (List.mem_append_left _ (hrow ha hxa)) _ (List.mem_append_right _ (hrow hb hxb))
    (ea.symm.trans ((bs_inj hab).trans eb))
  exact rowInput_inj hB (hG a ha) (hP b hb) hxa hxb

/-- A committed record whose digest is the Program's grid's commits to it, where that grid's frames are `B` bytes. -/
theorem commitsTo {p : Params} {l : String} {w : Nat} {dg : String} {P : Records.RawGrid}
    (eP : Records.committedDigest l w P p.grid = .ok dg)
    (hP : ∀ row ∈ P.toList, ∀ f ∈ (row.filterMap id).toList, f.size = p.grid.frameBytes) : CommitsTo p l w dg P := by
  refine ⟨eP, fun G hG eG => ?_⟩
  by_cases hc : ∃ x ∈ committedInputs l w G p.grid ++ committedInputs l w P p.grid,
      ∃ y ∈ committedInputs l w G p.grid ++ committedInputs l w P p.grid,
        x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y
  · exact Or.inr hc
  · refine Or.inl (frames_eq hG.1 hG.2 hP eG eP fun x hx y hy he => ?_)
    by_contra hne
    exact hc ⟨x, hx, y, hy, hne, he⟩

end Properties.NetworkRecords.Commit
