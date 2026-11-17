module

public import Flock.Hm96

/-! Level 3: the digests' widths, and the `hm96-sha512/v1` leaf's binding up to a SHA-512 collision (`Default512.leaf_binding`). -/

@[expose] public section

namespace Flock

def Sha256.HasCollision : Prop := ∃ u v : ByteArray, u ≠ v ∧ Sha256.hash u = Sha256.hash v

def Sha512.HasCollision : Prop := ∃ u v : ByteArray, u ≠ v ∧ Sha512.hash u = Sha512.hash v

theorem Sha256.hash_size (m : ByteArray) : (Sha256.hash m).size = 32 := by
  unfold Sha256.hash
  exact Array.size_ofFn

theorem Sha512.hash_size (m : ByteArray) : (Sha512.hash m).size = 64 := by
  unfold Sha512.hash
  exact Array.size_ofFn

theorem ByteArray.eq_of_get!_64 {x x' : ByteArray} (hx : x.size = 64) (hx' : x'.size = 64)
    (h : ∀ i, i < 64 → x.get! i = x'.get! i) : x = x' := by
  apply ByteArray.ext
  apply Array.ext
  · rw [ByteArray.size_data, ByteArray.size_data, hx, hx']
  · intro i h1 h2
    have hi : i < 64 := by rw [ByteArray.size_data, hx] at h1; exact h1
    have := h i hi
    change x.data[i]! = x'.data[i]! at this
    rwa [getElem!_pos x.data i h1, getElem!_pos x'.data i h2] at this

namespace Hm96

def xor64 (x m : ByteArray) : ByteArray := ByteArray.mk ((Array.range 64).map fun i => x.get! i ^^^ m.get! i)

theorem xor64_size (x m : ByteArray) : (xor64 x m).size = 64 := by
  rw [← ByteArray.size_data]
  simp [xor64]

theorem xor64_cancel {x x' m : ByteArray} (hx : x.size = 64) (hx' : x'.size = 64) (h : xor64 x m = xor64 x' m) : x = x' := by
  apply ByteArray.eq_of_get!_64 hx hx'
  intro i hi
  have := congrArg (fun b : ByteArray => b.data[i]!) h
  simp only [xor64] at this
  rw [getElem!_pos _ i (by simpa using hi), getElem!_pos _ i (by simpa using hi)] at this
  simpa [Array.getElem_map, Array.getElem_range] using this

def leafPair (P S : ByteArray) (M : ByteArray → ByteArray) (x x' y y' : ByteArray) : ByteArray × ByteArray :=
  let u := P ++ (xor64 x (M y) ++ Sha512.hash (S ++ y))
  let u' := P ++ (xor64 x' (M y') ++ Sha512.hash (S ++ y'))
  if u = u' then (S ++ y, S ++ y') else (u, u')

theorem leafPair_collides (P S : ByteArray) (M : ByteArray → ByteArray) (x x' y y' : ByteArray) (hx : x.size = 64)
    (hx' : x'.size = 64) (hne : (x, y) ≠ (x', y'))
    (h : Sha512.hash (P ++ (xor64 x (M y) ++ Sha512.hash (S ++ y))) =
      Sha512.hash (P ++ (xor64 x' (M y') ++ Sha512.hash (S ++ y')))) :
    (leafPair P S M x x' y y').1 ≠ (leafPair P S M x x' y y').2 ∧
      Sha512.hash (leafPair P S M x x' y y').1 = Sha512.hash (leafPair P S M x x' y y').2 := by
  by_cases ho : P ++ (xor64 x (M y) ++ Sha512.hash (S ++ y)) = P ++ (xor64 x' (M y') ++ Sha512.hash (S ++ y'))
  · simp only [leafPair, ho, ↓reduceIte]
    have hcs := (ByteArray.append_right_inj P).mp ho
    have hb : xor64 x (M y) = xor64 x' (M y') := ByteArray.append_inj_left hcs (by rw [xor64_size, xor64_size])
    rw [hb] at hcs
    have hc : Sha512.hash (S ++ y) = Sha512.hash (S ++ y') := (ByteArray.append_right_inj _).mp hcs
    refine ⟨fun hs => ?_, hc⟩
    have hy : y = y' := (ByteArray.append_right_inj S).mp hs
    subst hy
    exact hne (by rw [xor64_cancel hx hx' hb])
  · simp only [leafPair, ho, ↓reduceIte]
    exact ⟨ho, h⟩

theorem leaf_binding_of (P S : ByteArray) (M : ByteArray → ByteArray) (x x' y y' : ByteArray) (hx : x.size = 64)
    (hx' : x'.size = 64) (hne : (x, y) ≠ (x', y'))
    (h : Sha512.hash (P ++ (xor64 x (M y) ++ Sha512.hash (S ++ y))) =
      Sha512.hash (P ++ (xor64 x' (M y') ++ Sha512.hash (S ++ y')))) : Sha512.HasCollision :=
  let hp := leafPair_collides P S M x x' y y' hx hx' hne h
  ⟨_, _, hp.1, hp.2⟩

theorem Default512.leaf_binding (x x' y y' : ByteArray) (hx : x.size = 64) (hx' : x'.size = 64) (hne : (x, y) ≠ (x', y'))
    (h : Default512.leaf x y = Default512.leaf x' y') : Sha512.HasCollision :=
  leaf_binding_of Default512.leafPrefix Default512.saltPrefix Default512.mask x x' y y' hx hx' hne h

theorem Default512.leaf_size (x y : ByteArray) : (Default512.leaf x y).size = 64 := Sha512.hash_size _

end Hm96

end Flock
