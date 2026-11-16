module

public import Proofs.Properties.Verity.SampledTwo.Reading
public import Proofs.Flock.Soundness.Discharge.ZkView.Decode

/-!
# The releases agree with the run's coins as the form reads them

`DevAgrees` (`Projection.lean`) at the form's reading (`formReading`): on an accepted play whose log is the device's,
coins the coin map's agreement holds of agree with every release the log holds for a read round, as the form reads two
coins' agreement: on the bytes the run's bytes carry of them at the two stages' rounds (`prefixAgree`), as the sessions'
pieces say at a session's.

* `take_flatMap`: chunks of one length `k`, cut after `k u` items, are the first `u` chunks;
* `coinByte_of_elems`: two coins with the same elements below `u` have the same bytes below `16 u`, since a coin's bytes
  are its elements' 16 bytes each, in order (`FlockDeploy.Coin.bytes`);
* `prefixAgree_of_elems`: so they agree on their first `w` bytes wherever `w ≤ 16 u`;
* `devAgrees_formReading`: `DevAgrees` at `formReading` from what each group's pieces give at its rounds:
  `FirstElems` and `SecondElems` (the release's coin and the run's share their elements below some `u`, and the run's
  bytes carry at most `16 u` bytes of the round's coin: the coin map's agreement and the check's refusal of bytes past
  what the stage read), and `SessionsAgree` (the sessions' reading's agreement).
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- **Two coins have the same elements below `u`.** -/
def ElemsBelow (u : ℕ) (c c' : FlockDeploy.Coin) : Prop :=
  ∀ j : Fin 65536, j.val < u → c j = c' j

/-- **Chunks of one length, cut after `k u` items, are the first `u` chunks.** -/
theorem take_flatMap {α β : Type} (f : α → List β) (k : ℕ) (hk : ∀ a, (f a).length = k) :
    ∀ (l : List α) (u : ℕ), (l.flatMap f).take (k * u) = (l.take u).flatMap f
  | [], u => by simp
  | a :: l, 0 => by simp
  | a :: l, u + 1 => by
    rw [List.flatMap_cons, List.take_succ_cons, List.flatMap_cons, List.take_append,
      List.take_of_length_le (by rw [hk, Nat.mul_succ]; omega), hk,
      show k * (u + 1) - k = k * u by rw [Nat.mul_succ]; omega, take_flatMap f k hk l u]

theorem flatMap_congr_mem {α β : Type} {f g : α → List β} :
    ∀ {l : List α}, (∀ a ∈ l, f a = g a) → l.flatMap f = l.flatMap g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.flatMap_cons, List.flatMap_cons, h a (by simp), flatMap_congr_mem fun b hb => h b (by simp [hb])]

theorem lt_of_mem_take_finRange {n u : ℕ} {a : Fin n} (ha : a ∈ (List.finRange n).take u) : a.val < u := by
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
  simp only [List.length_take, List.length_finRange] at hi
  simp only [List.getElem_take, List.getElem_finRange, Fin.val_cast]
  omega

/-- **The same elements below `u`, the same bytes below `16 u`.** -/
theorem coinBytes_take {u : ℕ} {c c' : FlockDeploy.Coin} (h : ElemsBelow u c c') :
    c.bytes.take (16 * u) = c'.bytes.take (16 * u) := by
  have hk : ∀ (d : FlockDeploy.Coin) (i : Fin 65536), ((d i).toBytes.data.toList).length = 16 := fun d i => by
    simpa using FlockSoundness.Discharge.ZkView.toBytes_size (d i)
  unfold FlockDeploy.Coin.bytes
  rw [take_flatMap _ 16 (hk c), take_flatMap _ 16 (hk c')]
  exact flatMap_congr_mem fun a ha => by rw [h a (lt_of_mem_take_finRange ha)]

theorem coinByte_of_elems {u : ℕ} {c c' : FlockDeploy.Coin} (h : ElemsBelow u c c') (i : Fin SpotFlock.L)
    (hi : i.val < 16 * u) : SpotFlock.coinByte c i = SpotFlock.coinByte c' i := by
  have e : ∀ l : List UInt8, l.getD i.val 0 = ((l.take (16 * u))[i.val]?).getD 0 := fun l => by
    rw [List.getElem?_take, if_pos hi, List.getD_eq_getElem?_getD]
  simp only [SpotFlock.coinByte, Fin.mk.injEq]
  rw [e c.bytes, e c'.bytes, coinBytes_take h]

/-- **The same bytes below `w`, agreement on the first `w` bytes.** -/
theorem prefixAgree_of_bytes {w : ℕ} {c c' : FlockDeploy.Coin}
    (h : ∀ i : Fin SpotFlock.L, i.val < w → SpotFlock.coinByte c i = SpotFlock.coinByte c' i) :
    prefixAgree w c c' := by
  unfold prefixAgree SpotFlock.src Properties.Spot.bytesOf
  apply ByteArray.ext
  simp only [ByteArray.data_extract]
  apply Array.ext
  · simp
  · intro j hj₁ _
    simp only [Array.size_extract, Array.size_ofFn] at hj₁
    simp only [Array.getElem_extract, Array.getElem_ofFn]
    have key := h ⟨0 + j, by omega⟩ (by show 0 + j < w; omega)
    rw [key]

theorem prefixAgree_of_elems {u w : ℕ} {c c' : FlockDeploy.Coin} (hw : w ≤ 16 * u) (h : ElemsBelow u c c') :
    prefixAgree w c c' :=
  prefixAgree_of_bytes fun i hi => coinByte_of_elems h i (by omega)

variable (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
  (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Acc Good : Verity.Exchange → Prop)
  (Ag : Verity.Exchange → Coins A x → Prop)

/-- **The first stage's release and the run's coin share what the run's bytes carry** (pieces: the coin map's
agreement and the first stage's coins): on an accepted play whose log is the device's, at coins `Ag` holds of, the
release for round 0 and the run's coin there have the same elements below some `u`, and the first stage's bytes are at
most `16 u`. -/
def FirstElems : Prop :=
  ∀ o, Acc o → Good o → ∀ r : Coins A x, Ag o r → ∀ i : Fin (A.L.V.k x), i.val = 0 → ∀ c, dev o.log 0 = some c →
    ∃ inp u, inputOn C o = some inp ∧ inp.source.size ≤ 16 * u ∧ ElemsBelow u c (r i)

/-- **The second stage's releases and the run's coins share what the run's bytes carry** (pieces: the coin map's
agreement and the second stage's coins): as `FirstElems`, at each round `1` to `stage`, the second stage's bytes of
that round's coin (`stageWidth`). -/
def SecondElems : Prop :=
  ∀ o, Acc o → Good o → ∀ r : Coins A x, Ag o r → ∀ i : Fin (A.L.V.k x), 0 < i.val → i.val ≤ A.h.stage →
    ∀ c, dev o.log i.val = some c →
    ∃ inp it u, inputOn C o = some inp ∧ inp.interior = some it ∧ stageWidth it.source (i.val - 1) ≤ 16 * u ∧
      ElemsBelow u c (r i)

/-- **The sessions' releases agree with the run's coins** (piece: the sessions' coins): at each read round past the
statements', the round is a session's, and the release agrees with the run's coin as the sessions' reading says. -/
def SessionsAgree : Prop :=
  ∀ o, Acc o → Good o → ∀ r : Coins A x, Ag o r → ∀ i : Fin (A.L.V.k x), readsAt A.h i.val → A.h.stage + 1 < i.val →
    ∀ c, dev o.log i.val = some c →
    ∃ t, SpotFlock.slotOf A.h (i.val + 1) = some t ∧ S.agree o t (i.val - SpotFlock.off A.h t) c (r i)

/-- **`DevAgrees` at the form's reading**, from the three groups' agreements. -/
theorem devAgrees_formReading (hF : FirstElems A x C dev Acc Good Ag) (hS : SecondElems A x C dev Acc Good Ag)
    (hT : SessionsAgree A x S dev Acc Good Ag) : DevAgrees A x (formReading A C S dev) Acc Good Ag := by
  intro o hA hG r hAg i hr c hc
  dsimp only [formReading] at hr hc ⊢
  by_cases h0 : i.val = 0
  · obtain ⟨inp, u, hin, hw, he⟩ := hF o hA hG r hAg i h0 c (h0 ▸ hc)
    simp only [h0, ↓reduceIte]
    exact ⟨inp, hin, prefixAgree_of_elems hw he⟩
  · by_cases hs : i.val ≤ A.h.stage
    · obtain ⟨inp, it, u, hin, hit, hw, he⟩ := hS o hA hG r hAg i (Nat.pos_of_ne_zero h0) hs c hc
      simp only [h0, hs, ↓reduceIte]
      exact ⟨inp, it, hin, hit, prefixAgree_of_elems hw he⟩
    · have hlt : A.h.stage + 1 < i.val := by
        have := hr.1
        omega
      obtain ⟨t, ht, hag⟩ := hT o hA hG r hAg i hr hlt c hc
      simp only [h0, hs, ↓reduceIte, ht]
      exact hag

end Verity.SampledTwo
