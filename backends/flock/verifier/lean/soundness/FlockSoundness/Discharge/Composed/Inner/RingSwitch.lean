import FlockSoundness.Discharge.ZkExec.PlaysRep
import FlockSoundness.Discharge.Composed.Defs

/-!
# The ring switch is complete at the replay's claim points

`verify --zk` (`Zk.verifyRep`) checks the two masked claims of the replay (`ab`, from the lincheck, and `c`, from the
zerocheck) through their ring-switch messages: `claimCheck` of claim `k` on `ŝ_k` is a public constant, and the
claim's value is a masked message, so the check is the affine constraint on the pads `h` that `verifyRep` hands
`verifyInner` as "ring switch ab" (`rp.abValue + claimCheck ab ŝ_ab`) and "ring switch c" (`rp.cValue + claimCheck c
ŝ_c`). Then `ringSwitch` transposes each `ŝ_k`, recombines it at its coins `r''_k`, and batches the claims at `γ` into
`T`, the target that level 0's lane phase starts from.

The honest prover's messages at the claims the replay derives, for a witness `z`:

* `ŝ_k = shat (claimOf m c_k) z`: bit-slice `v` of `z` evaluated at `c_k`'s point beyond the packed seven bits
  (`OpeningPhase.lean`'s `shat`, §12.2);
* the lincheck's final message `zPartial = zPartialOf (claimOf m ab) z + h_zp`: the 64 skip positions of `z`, each
  evaluated at `ab`'s point (§10), under its pads;
* the zerocheck's `final_c = (claimOf m c).eval z + h_fc`: the `c` claim's value under its pad.

`ringSwitch_holds` says both constraints hold at every pads vector `h` (`LinHolds`: `⟨a, h⟩ = c`, the relation
`verifyInner`'s batched check combines). `ringSwitch_target` says the batched target of honest messages is
`Σ_y f(y)·b(y)`, with `f` the packed witness and `b` the batched basis (`OpenRel.b`'s form). `eqWord_eq_eqAt` and
`shat_eqWord` say the rank check's weights at a point (`eqWord`, the `eqw` of `coinedOf`) are `shat`'s weights `eqAt`
at it, so the mask-slot part of `ŝ_k` is `Φ_k(u)` at the claim's point beyond the packed bits
(`(c_k.x.extract 1).toList`, `Switched.z`).

None of the three reads a coin's value: no division or inverse is taken, so they hold at every coin.
-/

namespace FlockSoundness.Discharge.Composed.RingSwitch

open Flock Model Refine

local notation "𝔽" => FlockLevel3.GF128

/-- **An affine constraint holds at the pads `h`**: `⟨a, h⟩ = c`, the relation `Zk.verifyInner` batches
(`⟨Σ γ_k a_k, Y⟩ = Σ γ_k c_k + β·τ`); `Lin.masked v i` is the clear value `v + h_i` behind the masked `v`. -/
def LinHolds (l : Zk.Lin) (h : Array F128) : Prop := dot l.a h = l.c

/-- An executable claim as the model's, at `m` address bits (its value too, though nothing here reads it). -/
noncomputable def claimOf (m : ℕ) (c : Flock.Claim) : Model.Claim 𝔽 m := ⟨fOf c.value, vecOf 64 c.skip, vecOf (m - 6) c.x⟩

theorem two_pow_eq_mul64 {m : ℕ} (hm : 6 ≤ m) : 2 ^ m = 2 ^ (m - 6) * 64 := by
  rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel hm]

/-- The address `64u + i` of skip position `i` of the high address `u`. -/
def addr64 {m : ℕ} (hm : 6 ≤ m) (u : Fin (2 ^ (m - 6))) (i : Fin 64) : Fin (2 ^ m) :=
  ⟨64 * u.val + i.val, by rw [two_pow_eq_mul64 hm]; have := u.isLt; have := i.isLt; omega⟩

/-- **The honest lincheck's final message, unmasked** (§10): skip position `i` of the witness evaluated at the claim's
point, `Σ_u eq(x, u)·z[64u + i]`. -/
noncomputable def zPartialOf {m : ℕ} (hm : 6 ≤ m) (c : Model.Claim 𝔽 m) (z : Witness m) (i : Fin 64) : 𝔽 :=
  ∑ u : Fin (2 ^ (m - 6)), eqAt c.x u.val * ((z (addr64 hm u i)).val : 𝔽)

/-- **The ring switch's two constraints hold for the honest prover** (completeness of §12.2 under the pads, at the
claims `Zk.replay` derives). On a run of the masked replay, at any pads vector `h` and witness `z`, if the lincheck's
final message is the honest one under its pads, `final_c` is the `c` claim's value under its pad, and each of the two
ring-switch messages is the honest `ŝ` at its claim, then `verifyRep`'s constraints "ring switch ab" and
"ring switch c" hold at `h`. -/
theorem ringSwitch_holds (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) {t t' : Transcript}
    {rp : Zk.Replayed} (hrun : Zk.replay st lay p root t = .ok (rp, t')) (hm : 7 ≤ st.m) (z : Witness st.m)
    (h : Array F128)
    (hzp : ∀ i : Fin 64, fOf p.lincheck.zPartial[i.val]! =
      zPartialOf (by omega) (claimOf st.m rp.claims[0]!) z i + fOf h[lay.zp i.val]!)
    (hfc : fOf p.zerocheck.finalC = (claimOf st.m rp.claims[1]!).eval z + fOf h[lay.fc]!)
    (hs : ∀ (k : Fin 2) (v : Fin 128),
      fOf (p.ringSwitches[k.val]!)[v.val]! = shat hm (claimOf st.m rp.claims[k.val]!) z v) :
    LinHolds (rp.abValue.addConst (Zk.claimCheck rp.claims[0]! p.ringSwitches[0]!)) h ∧
      LinHolds (rp.cValue.addConst (Zk.claimCheck rp.claims[1]! p.ringSwitches[1]!)) h := by
  sorry

/-- **The batched target of honest ring switches** (completeness of §12.3–§12.5, GF128's claims to Ligerito's target).
On a run of `ringSwitch` on well-formed rounds (each round's coins as many as it issued), if every claim's message is the honest `ŝ` and `f` is the packed witness (bit `v` of `f y` is
`z[128y + v]`), the batched value `T` is `Σ_y f(y)·b(y)` for the batched basis `b(y) = Σ_k γ_k·Φ_{r''_k}(eq(z_k, y))`
of the switched claims (`OpenRel.b`'s form), at each claim's point beyond the packed bits `z_k`. -/
theorem ringSwitch_target (m : ℕ) (hm : 7 ≤ m) (claims : Array Flock.Claim) (rs : Array (Array F128))
    {t t' : Transcript} {T : F128} {sws : Array Switched} (hwf : RoundsWF t.rounds)
    (hrun : ringSwitch m claims rs t = .ok ((T, sws), t')) (z : Witness m) (f : Fin (2 ^ (m - 7)) → 𝔽) (hz : ∀ y v, z (addr128 hm y v) = execArith.unpack (f y) v)
    (hs : ∀ k < claims.size, ∀ v : Fin 128, fOf (rs[k]!)[v.val]! = shat hm (claimOf m claims[k]!) z v) :
    sws.size = claims.size ∧
      (∀ k < claims.size, (sws[k]!).z = claims[k]!.x.extract 1 ∧ (sws[k]!).z.size = m - 7 ∧
        (sws[k]!).rpp.size = 7) ∧
      fOf T = ∑ y : Fin (2 ^ (m - 7)), f y * ∑ k : Fin claims.size, fOf (sws[k.val]!).gamma *
        phi execArith (vecOf 7 (sws[k.val]!).rpp) (eqAt (vecOf (m - 7) (sws[k.val]!).z) y.val) := by
  sorry

/-- **The rank check's weights are the ring switch's** (`eqWord`, `coinedOf`'s `eqw`): at a point given as a list,
`eqWord p y` is `eqAt` of its entries at `y`, the weight `shat` gives word `y` at a claim whose point beyond the packed
bits is `p`. -/
theorem eqWord_eq_eqAt (p : List F128) (y : ℕ) :
    fOf (eqWord p y) = eqAt (fun j : Fin p.length => fOf p[j.val]) y := by
  sorry

/-- **The honest `ŝ` at the rank check's point**: at a claim of `m − 6` coordinates, `ŝ[v]` is
`Σ_y eqWord(p, y)·z[128y + v]` at the claim's point beyond the packed bits, `p = (c.x.extract 1).toList` (`Switched.z`),
so its mask-slot words carry the rank check's weights `eqWord(p, x)`. -/
theorem shat_eqWord {m : ℕ} (hm : 7 ≤ m) (c : Flock.Claim) (hx : c.x.size = m - 6) (z : Witness m) (v : Fin 128) :
    shat hm (claimOf m c) z v =
      ∑ y : Fin (2 ^ (m - 7)), fOf (eqWord (c.x.extract 1).toList y.val) * ((z (addr128 hm y v)).val : 𝔽) := by
  sorry

end FlockSoundness.Discharge.Composed.RingSwitch
