import FlockSoundness.Discharge.ZkLink.Link
import FlockSoundness.Binding.Rows
import FlockSoundness.Protocol.Discharge.ZkBind.Bind

/-!
# `--zk` value binding: the committed values are the drawn units' own

The compiled `--zk` count curve (`ZkLink.zk_flock_countC_execOS_cr`) takes the binding of wires to registered commit
strings for `--zk` tables as a named structure (`vb : ZkLink.ValueBindingZ … (decodeOfPlace … place) Hc`). This is
`Binding.Layout` and `Binding.HmRows` (the public path's `HmRows.binding`) for plans of `TabZK` tables:

* `LayoutZ`: the commit strings and their values, where a drawn unit's `--zk` table's witness holds each one's value and
  salt, and `decode_row` (in a satisfying witness the decoder's bit on each of the unit's wires is the bit of the row
  the witness holds at the wire's commit string);
* `HmRowsZ`: its `hm96-sha512/row/v1` or `/row/v2` rows, with `HmRowsZ.Computes`, `Assumptions.HmRowComputes` at the
  drawn units' rows of `--zk` tables;
* `HmRowsZ.bindingZ`: the `ValueBindingZ`, with `collide_spec` proved (explicit SHA-512 collisions, `Refine.H512`), and
  the registration fact from `HmRowComputes` (`HmRowsZ.registered`).

`bindingZ_bit` says what the binding binds: at a placed drawn unit, the value every satisfying witness opens a wire's
commit string to carries, on that wire, the bit of the unit's own column in its block, and its commitment is the
registered one.
-/

namespace FlockSoundness.Discharge.ZkBind

open Game Model Accounting Refine Audit ZkSession ZkLink Binding

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
  (A : Arith F K) {C : Circuit} {n : ℕ} (P : Audit.Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

variable {A P planZ tab} {decode : DecoderZK (C := C) planZ tab}

namespace LayoutZ

variable (lay : LayoutZ A P planZ tab decode) (scheme : lay.Pos → ExplicitLeaf lay.Val)

end LayoutZ

namespace HmRowsZ

variable {lay : LayoutZ A P planZ tab decode} (rs : HmRowsZ lay)

end HmRowsZ

/-- **What the `--zk` value binding binds: a placed drawn unit's own columns.** With the decoder of the drawn units'
places (`decodeOfPlace`), in every witness of drawn unit `u`'s `--zk` table that satisfies its statement, the opening of
the commit string of each of `u`'s wires (`inst.wire x`, the wire of the unit's column `x`) has the registered
commitment, and its value's bit on that wire is the witness's bit at the unit's column `x` in its block. -/
theorem bindingZ_bit (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    {lay : LayoutZ A P planZ tab (decodeOfPlace A P planZ tab place)} (rs : HmRowsZ lay) (hHm : rs.Computes)
    (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) (M : MsgZ planZ tab S R u) (hM : SatZ A planZ tab S R u M)
    (x : Fin (place S R u h).R.w) (hg : (place S R u h).inst.wire x ∈ P.io u) :
    (rs.bindingZ hHm).com ((rs.bindingZ hHm).opening S R u M ((rs.bindingZ hHm).pos ((place S R u h).inst.wire x))) =
        (rs.bindingZ hHm).cs R ((rs.bindingZ hHm).pos ((place S R u h).inst.wire x)) ∧
      (rs.bindingZ hHm).bit ((rs.bindingZ hHm).val ((rs.bindingZ hHm).opening S R u M
          ((rs.bindingZ hHm).pos ((place S R u h).inst.wire x)))) ((place S R u h).inst.wire x) =
        bit ((tabOfZ planZ tab S R u).S.block (witZ A planZ tab S R u M) (place S R u h).o ((place S R u h).col x)) := by
  have hs := (rs.bindingZ hHm).opening_spec S R u h M hM _ hg
  refine ⟨hs.1, hs.2.trans ?_⟩
  simp only [decodeOfPlace, h, ↓reduceDIte, UnitPlace.decode]
  exact (place S R u h).inst.decode_wire ((place S R u h).agree hM) x

end FlockSoundness.Discharge.ZkBind
