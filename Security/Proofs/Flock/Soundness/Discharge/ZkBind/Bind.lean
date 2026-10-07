import Proofs.Flock.Soundness.Discharge.ZkLink.Link
import Proofs.Flock.Soundness.Binding.Rows

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

/-- The message type of drawn unit `u`'s `--zk` table. -/
abbrev MsgZ (S : Finset (Fin n)) (R : Reg) (u : Fin n) : Type :=
  Message F (tabOfZ planZ tab S R u).sch.k0 (tabOfZ planZ tab S R u).sch.level₀.logCols

/-- The witness a message of drawn unit `u`'s `--zk` table packs. -/
abbrev witZ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M : MsgZ planZ tab S R u) :
    Witness (tabOfZ planZ tab S R u).S.m :=
  wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M

/-- A message of drawn unit `u`'s `--zk` table packs a witness that satisfies the table's statement. -/
def SatZ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M : MsgZ planZ tab S R u) : Prop :=
  (tabOfZ planZ tab S R u).S.Satisfies (witZ A planZ tab S R u M)

/-- **The layout of the drawn units' committed wires in registered leaves, for `--zk` tables** (`Binding.Layout`'s
fields over `TabZK` plans). `decode_row`: in a satisfying witness the decoder's bit on each of the unit's wires is the bit
of the value the witness holds at the wire's commit string. -/
structure LayoutZ (decode : DecoderZK (C := C) planZ tab) where
  Pos : Type
  [finPos : Fintype Pos]
  [decPos : DecidableEq Pos]
  [inhPos : Inhabited Pos]
  Val : Type
  [finVal : Fintype Val]
  [decVal : DecidableEq Val]
  [inhVal : Inhabited Val]
  pos : Fin C.N → Pos
  bit : Val → Fin C.N → Bool
  regLeaf : Reg → Pos → Dig
  rowBits : Pos → ℕ
  rowOf : (p : Pos) → (Fin (rowBits p) → Bool) → Val
  rowAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → (p : Pos) → Fin (rowBits p) →
    Fin (2 ^ (tabOfZ planZ tab S R u).S.m)
  saltAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Pos → Fin 192 → Fin 8 →
    Fin (2 ^ (tabOfZ planZ tab S R u).S.m)
  decode_row : ∀ S R u, u ∈ S → ∀ M, SatZ A planZ tab S R u M → ∀ g ∈ P.io u,
    decode S R u M g = bit (rowOf (pos g) (bitsAt (witZ A planZ tab S R u M) (rowAt S R u (pos g)))) g

attribute [instance] LayoutZ.finPos LayoutZ.decPos LayoutZ.inhPos LayoutZ.finVal LayoutZ.decVal LayoutZ.inhVal

variable {A P planZ tab} {decode : DecoderZK (C := C) planZ tab}

namespace LayoutZ

variable (lay : LayoutZ A P planZ tab decode) (scheme : lay.Pos → ExplicitLeaf lay.Val)

/-- The value a drawn unit's `--zk` table's message holds at commit string `p`. -/
noncomputable def row (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M : MsgZ planZ tab S R u) (p : lay.Pos) : lay.Val :=
  lay.rowOf p (bitsAt (witZ A planZ tab S R u M) (lay.rowAt S R u p))

/-- The salt it holds there: the witness's 192 bytes at `saltAt`. -/
noncomputable def salt (S : Finset (Fin n)) (R : Reg) (u : Fin n) (M : MsgZ planZ tab S R u) (p : lay.Pos) :
    List UInt8 :=
  bytesAt (witZ A planZ tab S R u M) (lay.saltAt S R u p)

/-- **The registration fact**: in a satisfying witness of a drawn unit's `--zk` table, the leaf of the value and salt
the witness holds at each of the unit's commit strings is the registered leaf there. -/
def Registered : Prop :=
  ∀ S R u, u ∈ S → ∀ M, SatZ A planZ tab S R u M → ∀ g ∈ P.io u,
    (scheme (lay.pos g)).leaf (lay.row S R u M (lay.pos g)) (lay.salt S R u M (lay.pos g)) = lay.regLeaf R (lay.pos g)

/-- **The value binding of a `--zk` layout** (`Layout.binding`'s): an opening is a commit string, a value and a salt,
its commitment the commit string and its leaf, and `collide` the leaf scheme's extractor. -/
noncomputable def bindingZ (hreg : lay.Registered scheme) : ValueBindingZ A P planZ tab decode H512 where
  Pos := lay.Pos
  Val := lay.Val
  pos := lay.pos
  bit := lay.bit
  Opening := lay.Pos × lay.Val × List UInt8
  inhOp := ⟨(default, default, [])⟩
  Com := lay.Pos × Dig
  val o := o.2.1
  com o := (o.1, (scheme o.1).leaf o.2.1 o.2.2)
  cs R p := (p, lay.regLeaf R p)
  opening S R u M p := (p, lay.row S R u M p, lay.salt S R u M p)
  opening_spec S R u hu M hM g hg :=
    ⟨congrArg (Prod.mk (lay.pos g)) (hreg S R u hu M hM g hg), (lay.decode_row S R u hu M hM g hg).symm⟩
  collide o o' := (scheme o.1).pair o.2.1 o.2.2 o'.2.1 o'.2.2
  collide_spec o o' hc hv := by
    obtain ⟨p, v, y⟩ := o
    obtain ⟨p', v', y'⟩ := o'
    simp only [Prod.mk.injEq] at hc
    obtain ⟨rfl, hl⟩ := hc
    exact (scheme p).pair_spec v y v' y' hv hl

end LayoutZ

/-- **A `--zk` layout's `hm96-sha512/row/v1` or `/row/v2` rows** (`Binding.HmRows`' fields over `TabZK` plans). -/
structure HmRowsZ (lay : LayoutZ A P planZ tab decode) where
  isRow : lay.Pos → Prop
  [decRow : DecidablePred isRow]
  enc : lay.Val → List UInt8
  enc_inj : Function.Injective enc
  other : lay.Pos → ExplicitLeaf lay.Val
  bcAt : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → lay.Pos → Fin 128 → Fin 8 →
    Fin (2 ^ (tabOfZ planZ tab S R u).S.m)
  bc_registered : ∀ S R u, u ∈ S → ∀ M, SatZ A planZ tab S R u M → ∀ g ∈ P.io u, isRow (lay.pos g) →
    H512 (K512.leafPrefix ++ bytesAt (witZ A planZ tab S R u M) (bcAt S R u (lay.pos g))) = lay.regLeaf R (lay.pos g)
  other_registered : ∀ S R u, u ∈ S → ∀ M, SatZ A planZ tab S R u M → ∀ g ∈ P.io u, ¬ isRow (lay.pos g) →
    (other (lay.pos g)).leaf (lay.row S R u M (lay.pos g)) (lay.salt S R u M (lay.pos g)) = lay.regLeaf R (lay.pos g)

attribute [instance] HmRowsZ.decRow

/-- A place a `--zk` circuit hashes a row: a drawn unit, a message of its table, and one of its wires. -/
structure RowSiteZ (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length) (N : ℕ) where
  S : Finset (Fin n)
  R : Reg
  u : Fin n
  M : MsgZ planZ tab S R u
  g : Fin N

namespace HmRowsZ

variable {lay : LayoutZ A P planZ tab decode} (rs : HmRowsZ lay)

/-- Each commit string's leaf scheme: `hm96-sha512/row/v1` at a row, `other` elsewhere. -/
def scheme (p : lay.Pos) : ExplicitLeaf lay.Val :=
  if rs.isRow p then ExplicitLeaf.hm96 rs.enc rs.enc_inj else rs.other p

/-- **`HmRowComputes` at the drawn units' rows of `--zk` tables**: at each wire of a drawn unit held by a row, in a
satisfying witness of the unit's table, the `b ‖ c` the row's `hm96` slot outputs is the commit string of the row's
bytes and the salt the witness holds, under SHA-512 and the executable's key (`K512`). -/
def Computes : Prop :=
  Assumptions.HmRowComputes (fun l => digBytes (H512 l)) K512.saltPrefix K512.mask
    (fun w : RowSiteZ planZ tab C.N =>
      w.u ∈ w.S ∧ SatZ A planZ tab w.S w.R w.u w.M ∧ w.g ∈ P.io w.u ∧ rs.isRow (lay.pos w.g))
    (fun w => rs.enc (lay.row w.S w.R w.u w.M (lay.pos w.g)))
    (fun w => lay.salt w.S w.R w.u w.M (lay.pos w.g))
    (fun w => bytesAt (witZ A planZ tab w.S w.R w.u w.M) (rs.bcAt w.S w.R w.u (lay.pos w.g)))

/-- **The registration fact from `HmRowComputes`**, for `--zk` tables (`HmRows.registered`'s). -/
theorem registered (hHm : rs.Computes) : lay.Registered rs.scheme := by
  intro S R u hu M hM g hg
  unfold scheme
  split_ifs with hrow
  · have hc : bytesAt (witZ A planZ tab S R u M) (rs.bcAt S R u (lay.pos g)) =
        commitString (rs.enc (lay.row S R u M (lay.pos g))) (lay.salt S R u M (lay.pos g)) :=
      hHm ⟨S, R, u, M, g⟩ ⟨hu, hM, hg, hrow⟩
    have hb := rs.bc_registered S R u hu M hM g hg hrow
    rw [hc] at hb
    exact hb
  · exact rs.other_registered S R u hu M hM g hg hrow

/-- **The value binding of a `--zk` layout's rows**: `LayoutZ.bindingZ` at the rows' scheme, from `HmRowComputes`. -/
noncomputable def bindingZ (hHm : rs.Computes) : ValueBindingZ A P planZ tab decode H512 :=
  lay.bindingZ rs.scheme (rs.registered hHm)

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
