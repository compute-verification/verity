module

public import Proofs.Properties.Verity.SampledTwo.AcceptsProof
public import Proofs.Flock.SpotFlock.Coins

/-!
# Every byte string is some coins' bytes

`coinsCover`: `CoinsCover` at every held copy. A coin's bytes and the coins are one for one (`SpotFlock.coinEquiv`), and
so are `R` coins' bytes and `R` coins (`SpotFlock.stageEquiv`), so every byte string of one coin's length is a coin's
`src` and every one of `stage` coins' length is a second stage's `stageSrc`. `acceptsRun_of` is `acceptsRun` with that
premise discharged.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

theorem eq_of_getElem {a b : ByteArray} (hs : a.size = b.size)
    (h : ∀ k (hk : k < a.size), a[k] = b[k]'(by omega)) : a = b := by
  have h1 : a.extract 0 (0 + a.size) = b.extract 0 (0 + a.size) := by
    rw [ByteArray.extract_eq_extract_iff_getElem (by omega) (by omega)]
    intro k hk
    simpa using h k hk
  rw [Nat.zero_add, ByteArray.extract_zero_size, hs, ByteArray.extract_zero_size] at h1
  exact h1

/-- **A byte string is the bytes of its bytes.** -/
theorem bytesOf_getElem {b : ByteArray} {n : ℕ} (hb : b.size = n) :
    Properties.Spot.bytesOf (fun i : Fin n => (b[i.val]'(by rw [hb]; exact i.isLt)).toFin) = b := by
  refine eq_of_getElem (by rw [size_bytesOf, hb]) fun k hk => ?_
  rw [getElem_bytesOf]
  exact UInt8.ofFin_toFin _

/-- **Every byte string is some coins' bytes** (`CoinsCover`), at every held copy. -/
theorem coinsCover (h : SpotFlock.Held) : CoinsCover h := by
  refine ⟨fun b hb => ?_, fun b hb => ?_⟩
  · obtain ⟨F, hF⟩ : ∃ F : Fin 1 → Fin SpotFlock.L → Fin 256,
        F = fun _ i => (b[i.val]'(by rw [hb]; exact i.isLt)).toFin := ⟨_, rfl⟩
    refine ⟨SpotFlock.coinEquiv.symm F, ?_⟩
    have e : SpotFlock.coinByte (SpotFlock.coinEquiv.symm F) = F 0 :=
      congrFun (SpotFlock.coinEquiv.apply_symm_apply F) 0
    show Properties.Spot.bytesOf (SpotFlock.coinByte _) = b
    rw [e, hF]
    exact bytesOf_getElem hb
  · obtain ⟨F, hF⟩ : ∃ F : Fin 1 → Fin (h.stage * SpotFlock.L) → Fin 256,
        F = fun _ i => (b[i.val]'(by rw [hb]; exact i.isLt)).toFin := ⟨_, rfl⟩
    refine ⟨(SpotFlock.stageEquiv h.stage).symm F, ?_⟩
    have e : SpotFlock.stageByte ((SpotFlock.stageEquiv h.stage).symm F) = F 0 :=
      congrFun ((SpotFlock.stageEquiv_apply h.stage _).symm.trans
        ((SpotFlock.stageEquiv h.stage).apply_symm_apply F)) 0
    show Properties.Spot.bytesOf (SpotFlock.stageByte _) = b
    rw [e, hF]
    exact bytesOf_getElem hb

/-- **The form's accept is the run's accept**, `acceptsRun` with `CoinsCover` discharged (`coinsCover`). -/
theorem acceptsRun_of (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hfit : SourcesFit A C) (hplay : PlayAccepts A C S P') (hsess : SessionMsgs A C S P') (hreads : ReadsOnly A.h) :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun A x C S dev P' (coinsCover A.h) hfit hplay hsess hreads

end Verity.SampledTwo
