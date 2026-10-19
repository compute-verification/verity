import Proofs.Flock.Claims.ZK.ExeLayout

/-!
# A prover of levels ≥ 1 reading salts by level and position

A Ligerito prover of levels ≥ 1 (`Composed.LigProver`) gets rep `r`'s levels ≥ 1 salts as one array `HiOf Tz`, and the
masked prover reads them by level and position (`ligeritoPZk`'s `salt l q`, `1 ≤ l < |levels|`, `q < 2^d`).

* `HiAt`, `hiSalt`: a level `l ≥ 1` of the schedule and a position of its tree; level `l`'s salt at position `p` of an
  array of levels ≥ 1's salts, at `hiIdx`, empty off the schedule.
* `SaltProver`, `SaltsOn`, `ofSalts`: a prover that reads salts by level and position, that it reads only the schedule's,
  and the Ligerito prover reading them at `hiIdx`.
* `saltBlob`, `blobSalt`, `PcExe`: the salts as one string, 192 bytes each in `hiIdx` order, and back; the prover sending
  that string, so a rep's levels ≥ 1 value carries its salts.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Masked

/-- A level `l ≥ 1` of the schedule and a position `p` of its tree. -/
def HiAt (ls : List FlockSoundness.Accounting.Level) (l p : ℕ) : Prop :=
  1 ≤ l ∧ l < ls.length ∧ p < 2 ^ (ls.getD l default).logLen

instance (ls : List FlockSoundness.Accounting.Level) (l p : ℕ) : Decidable (HiAt ls l p) := by
  unfold HiAt; infer_instance

/-- **Level `l`'s salt at position `p`** of the array `hi` of levels ≥ 1's salts in level order (`hiIdx`), empty off the
schedule. -/
def hiSalt (ls : List FlockSoundness.Accounting.Level) {n : ℕ} (hi : Fin n → ZkView.Salt) (l p : ℕ) : ByteArray :=
  if HiAt ls l p then fAt ByteArray.empty (fun k => ZkView.saltBytes (hi k)) (hiOff ls l + p) else ByteArray.empty

/-- **A prover of Ligerito's levels ≥ 1 reading salts by level and position**: rep `r`'s bytes after `T'` from the
table, the statement, the coins, the restart `y₁'` and the salts, level `l`'s at position `p` (`ligeritoPZk`'s
`salt l p`). -/
abbrev SaltProver : Type :=
  ∀ {mPts : ℕ} (Tz : ZkSession.TabZK 𝔽 mPts) (st : _root_.Flock.Stmt), Composed.CoinsZK (Composed.RestOf Tz st) →
    Fin 2 → (L : ℕ) → (Fin L → 𝕂) → (ℕ → ℕ → ByteArray) → ByteArray

/-- **`Pc` reads only the schedule's levels ≥ 1 and their positions**: salts that agree there give the same bytes. -/
def SaltsOn (Pc : SaltProver) : Prop :=
  ∀ {mPts : ℕ} (Tz : ZkSession.TabZK 𝔽 mPts) (st : _root_.Flock.Stmt) (c : Composed.CoinsZK (Composed.RestOf Tz st))
    (r : Fin 2) (L : ℕ) (y : Fin L → 𝕂) (s s' : ℕ → ℕ → ByteArray),
    (∀ l p, HiAt Tz.sch.levels l p → s l p = s' l p) → Pc Tz st c r L y s = Pc Tz st c r L y s'

/-- **The Ligerito prover of `Pc`**: levels ≥ 1's salts read at `hiIdx` (`hiSalt`). -/
def ofSalts (Pc : SaltProver) : Composed.LigProver :=
  fun Tz st c r L y hi => Pc Tz st c r L y (hiSalt Tz.sch.levels hi)

/-- A string's first 192 bytes, zero-padded to 192. -/
def fit192 (b : ByteArray) : ByteArray := b.extract 0 192 ++ ⟨Array.replicate (192 - b.size) 0⟩

/-- **Levels ≥ 1's salts as one string**: level `l`'s salt at position `p`, fitted to 192 bytes, in level order and
position order, so at bytes `192·(hiOff l + p)` on. -/
def saltBlob (ls : List FlockSoundness.Accounting.Level) (s : ℕ → ℕ → ByteArray) : ByteArray :=
  (List.range' 1 (ls.length - 1)).foldl (fun acc l =>
    (List.range (2 ^ (ls.getD l default).logLen)).foldl (fun acc p => acc ++ fit192 (s l p)) acc) ByteArray.empty

/-- **Level `l`'s salt at position `p` read back from the string `b`**: its 192 bytes at `192·(hiOff l + p)`, empty off
the schedule. -/
def blobSalt (ls : List FlockSoundness.Accounting.Level) (b : ByteArray) (l p : ℕ) : ByteArray :=
  if HiAt ls l p then b.extract (192 * (hiOff ls l + p)) (192 * (hiOff ls l + p + 1)) else ByteArray.empty

/-- **The levels ≥ 1 prover the deployed session's encoding reads**: rep `r`'s levels ≥ 1 value is its salts as one
string (`saltBlob`), from which the encoding recomputes levels ≥ 1 as the session does. -/
def PcExe : SaltProver := fun Tz _ _ _ _ _ s => saltBlob Tz.sch.levels s

theorem foldl_congr_mem {α β : Type} {f g : α → β → α} :
    ∀ (xs : List β) (a : α), (∀ x ∈ xs, ∀ a, f a x = g a x) → xs.foldl f a = xs.foldl g a
  | [], _, _ => rfl
  | x :: xs, a, h => by
    simp only [List.foldl_cons]
    rw [h x List.mem_cons_self a]
    exact foldl_congr_mem xs _ fun y hy => h y (List.mem_cons_of_mem _ hy)

/-- **`PcExe` reads only the schedule's levels ≥ 1 and their positions.** -/
theorem saltsOn_PcExe : SaltsOn PcExe := by
  intro mPts Tz st c r L y s s' h
  show saltBlob Tz.sch.levels s = saltBlob Tz.sch.levels s'
  unfold saltBlob
  refine foldl_congr_mem _ _ fun l hl acc => foldl_congr_mem _ _ fun p hp acc' => ?_
  obtain ⟨h1, h2⟩ := List.mem_range'_1.1 hl
  rw [h l p ⟨h1, by omega, List.mem_range.1 hp⟩]

end Masked

end Claims.ZKMal
