import Proofs.Flock.Soundness.Registered.ReadsX
import Proofs.Flock.Soundness.Discharge.Layout.RowEnc
import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
import Proofs.Flock.Level3.RowV2
import Proofs.Flock.VBridge.Family.Registered

/-!
# Two registrants' records at one registered root and position

`RowsShared` (`Family/Registered.lean`) asks that two sessions' registrant records (`ReadsX`) agree where they read one
registered root at one position. Each record's opening opens that root there (`ReadsX.committed`), so they agree up to
a SHA-512 collision, in a pair fixed by the two records rather than by a run:

* `registrants_eq_or_collides`: two registered reads at one root and position carry one row's bytes (each at its own
  encoding, `enc`), or their openings collide (`TwoOpenings.collides`), or hm96's leaves do (`hm96Pair_spec`);
* `rowEncP_get_eq`, `rowEncPre_get_eq`: one row's bytes (`rowEncP` under a first block, what `sha512/row/v1` or
  `sha512/row/v2` hashes, `Hidden.rowPre`) fix its bits at every width and either label, so two sessions' values agree
  at each bit both widths hold; `encPortH_of_none`: a port without segments hashes those bytes (`HiddenExec.encPortH`);
* `ReadsUnsegmented`: no registered read is of a segmented row (`sha512/row-seg/v1`), whose bytes (`Layout.encSeg`)
  need not fix its bits;
* `SharedRows`, `rowsShared_of_sharedRows`: `RowsShared` where two sessions' registrant records at one root and
  position hold one row's bytes, as they do when the family takes them from one choice, and no read is segmented;
* `NoRegistrantCollision`, `sharedRows_of`, `rowsShared_of`: `SharedRows`, and so `RowsShared`, where no two such
  records give either collision;
* `noRegistrantCollision`, `sharedRows`, `rowsShared`: the three hold (`RowsShared` where no read is segmented), since
  both collisions are `RegColl`'s (`Recursive/Registrants.lean`), which `Accepted` excludes and the recursive audit
  prices once.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Refine FlockSoundness.Binding FlockSoundness.Registered Audit

section Two

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Layout P plan tab decode} {rs : HmRowsX lay}
  {C' : Circuit} {n' : ℕ} {P' : Audit.Partition C' n'} {Reg₁ : Type} {mPts' : ℕ}
  {plan' : Finset (Fin n') → Reg₁ → List (TabSpec mPts')}
  {tab' : (S : Finset (Fin n')) → (R : Reg₁) → Fin n' → Fin (plan' S R).length}
  {decode' : Audit.Partition.DecoderB (C := C') (Reg := Reg₁) plan' tab'}
  {lay' : Layout P' plan' tab' decode'} {rs' : HmRowsX lay'}

/-- **Two registrants' rows at one registered root and position are one row's bytes, or SHA-512 collides**: where two
records' registered reads `p`, `p'` name one root (`port`) and one position (`readAt`), each record's opening opens it
there (`ReadsX.committed`), so their leaves are one, or the two openings conflict (`TwoOpenings.collides`); one leaf
over two rows' bytes, each at its read's encoding, is hm96's collision (`hm96Pair_spec`). -/
theorem registrants_eq_or_collides (rd : ReadsX lay rs) (rd' : ReadsX lay' rs') {p : lay.Pos} {p' : lay'.Pos}
    (hp : rd.isRead p) (hp' : rd'.isRead p') (hport : rd.port p = rd'.port p') (hat : rd.readAt p = rd'.readAt p') :
    rs.enc p (rd.wv p) = rs'.enc p' (rd'.wv p') ∨
      Collides (⟨rd.port p, rd.readAt p, rd.leaf p, rd.wpath p, rd'.leaf p', rd'.wpath p'⟩ : TwoOpenings).pair ∨
      Collides (hm96Pair (rs.enc p (rd.wv p)) (rd.ws p) (rs'.enc p' (rd'.wv p')) (rd'.ws p')) := by
  by_cases hl : rd.leaf p = rd'.leaf p'
  · by_cases he : rs.enc p (rd.wv p) = rs'.enc p' (rd'.wv p')
    · exact Or.inl he
    · exact Or.inr (Or.inr (hm96Pair_spec _ _ _ _ he hl))
  · refine Or.inr (Or.inl (TwoOpenings.collides _ ⟨hl, rd.committed p hp, ?_⟩))
    show Flock.Registered.opensLeaf (rd.port p) (rd'.leaf p').1 (rd.readAt p) (rd'.wpath p') = true
    rw [hport, hat]
    exact rd'.committed p' hp'

end Two

/-! ## One row's bytes at two widths -/

section Enc

open FlockSoundness.Discharge.Layout

/-- `sha512/row/v1`'s first block begins with the label's `v`. -/
theorem rowPrefix_head (w : ℕ) : (Flock.HmRow.rowPrefix w).data.toList.head? = some 118 := by
  have hA : ("verity/sha512-row/v1".toUTF8.push 0 ++ ByteArray.mk #[1, 16]).data.toList.head? = some 118 := by
    decide
  unfold Flock.HmRow.rowPrefix Flock.Hm96.pad
  rw [ByteArray.data_append, ByteArray.data_append, Array.toList_append, Array.toList_append, List.head?_append,
    List.head?_append, hA]
  rfl

/-- `sha512/row/v2`'s first block begins with the label's `v`. -/
theorem rowPrefixV2_head (l : ℕ) : (Flock.HmRow.rowPrefixV2 l).data.toList.head? = some 118 := by
  have hA : "verity/sha512-row/v2".toUTF8.data.toList.head? = some 118 := by decide
  rw [FlockLevel3.RowV2.prefix_data, List.append_assoc, List.append_assoc, List.head?_append, hA]
  rfl

/-- A hidden row's first block (`sha512/row/v1`'s or `sha512/row/v2`'s) begins with the label's `v`. -/
theorem rowPre_head (v2 : Bool) (w : ℕ) : (FlockSoundness.Discharge.Hidden.rowPre v2 w).data.toList.head? = some 118 := by
  unfold FlockSoundness.Discharge.Hidden.rowPre
  split
  · exact rowPrefixV2_head _
  · exact rowPrefix_head _

theorem rowPre_length (v2 : Bool) (w : ℕ) : (FlockSoundness.Discharge.Hidden.rowPre v2 w).data.toList.length = 128 := by
  rw [Array.length_toList]
  exact FlockSoundness.Discharge.Hidden.size_rowPre v2 w

theorem rowEncP_head_shaped {pre : ℕ → ByteArray} (hpre : ∀ w, (pre w).data.toList.head? = some 118) {W : ℕ}
    {v : BitVal W} (h : RowShaped v) : (rowEncP pre v).head? = some 118 := by
  simp [rowEncP, h, List.head?_append, hpre]

theorem rowEncP_head_other {pre : ℕ → ByteArray} {W : ℕ} {v : BitVal W} (h : ¬ RowShaped v) :
    (rowEncP pre v).head? = some 0 := by
  simp [rowEncP, h, List.head?_append, List.head?_replicate]

theorem rowEncP_row_length {pre : ℕ → ByteArray} (hpre : ∀ w, (pre w).data.toList.length = 128) {W : ℕ}
    {v : BitVal W} (h : RowShaped v) : (rowEncP pre v).length = 128 + 2 * (v.1.val / 16) := by
  simp only [rowEncP, h, ↓reduceIte, List.length_append, hpre, rowBytes, List.length_ofFn]

theorem rowEncP_other_length {pre : ℕ → ByteArray} {W : ℕ} {v : BitVal W} (h : ¬ RowShaped v) :
    (rowEncP pre v).length = 128 + W + (2 * W + 1) := by
  simp only [rowEncP, h, ↓reduceIte, List.length_append, List.length_replicate, bitCode_length]

/-- `get_of_rowBytes` at two widths. -/
theorem get_of_rowBytes₂ {W W' : ℕ} {v : BitVal W} {w : BitVal W'} (hv : RowShaped v) (h : rowBytes v = rowBytes w)
    (i : ℕ) (hi : i < W) (hi' : i < W') (hiv : i < v.1.val) : v.2 ⟨i, hi⟩ = w.2 ⟨i, hi'⟩ := by
  have hj : i / 8 < 2 * (v.1.val / 16) := by have := hv.1; omega
  have hl : i / 8 < (rowBytes v).length := by simp only [rowBytes, List.length_ofFn]; exact hj
  have e := List.getElem_of_eq h hl
  simp only [rowBytes, List.getElem_ofFn] at e
  have := congrFun (byteOf_inj e) ⟨i % 8, Nat.mod_lt _ (by norm_num)⟩
  simp only [BitVal.get, show 8 * (i / 8) + i % 8 = i by omega, hi, hi', ↓reduceDIte] at this
  exact this

/-- **One row's bytes fix its bits at every width, under either first block** (`rowEncP_inj` at two widths and two
first blocks of 128 bytes): a row's length fixes its words and its bytes its bits, both zero past the words; a first
block's first byte (`v`) tells a row from any other value; two other values are `bitCode`s of one width. -/
theorem rowEncP_get_eq {pre pre' : ℕ → ByteArray} (hl : ∀ w, (pre w).data.toList.length = 128)
    (hl' : ∀ w, (pre' w).data.toList.length = 128) (hh : ∀ w, (pre w).data.toList.head? = some 118)
    (hh' : ∀ w, (pre' w).data.toList.head? = some 118) {W W' : ℕ} {v : BitVal W} {w : BitVal W'}
    (h : rowEncP pre v = rowEncP pre' w) (i : ℕ) (hi : i < W) (hi' : i < W') : v.2 ⟨i, hi⟩ = w.2 ⟨i, hi'⟩ := by
  by_cases hv : RowShaped v <;> by_cases hw : RowShaped w
  · have hlen := congrArg List.length h
    rw [rowEncP_row_length hl hv, rowEncP_row_length hl' hw] at hlen
    have hk : v.1.val = w.1.val := by have := hv.1; have := hw.1; omega
    simp only [rowEncP, hv, hw, ↓reduceIte] at h
    obtain ⟨-, hb⟩ := List.append_inj h (by rw [hl, hl'])
    by_cases hiv : i < v.1.val
    · exact get_of_rowBytes₂ hv hb i hi hi' hiv
    · rw [hv.2 ⟨i, hi⟩ (by show v.1.val ≤ i; omega), hw.2 ⟨i, hi'⟩ (by show w.1.val ≤ i; omega)]
  · have := congrArg List.head? h
    rw [rowEncP_head_shaped hh hv, rowEncP_head_other hw] at this
    exact absurd this (by decide)
  · have := congrArg List.head? h
    rw [rowEncP_head_other hv, rowEncP_head_shaped hh' hw] at this
    exact absurd this (by decide)
  · have hlen := congrArg List.length h
    rw [rowEncP_other_length hv, rowEncP_other_length hw] at hlen
    obtain rfl : W = W' := by omega
    simp only [rowEncP, hv, hw, ↓reduceIte] at h
    rw [bitCode_inj (List.append_inj h rfl).2]

/-- `rowEncP_get_eq` at two hidden rows' first blocks (`Hidden.rowPre`, `sha512/row/v1`'s or `sha512/row/v2`'s). -/
theorem rowEncPre_get_eq {b b' : Bool} {W W' : ℕ} {v : BitVal W} {w : BitVal W'}
    (h : rowEncP (FlockSoundness.Discharge.Hidden.rowPre b) v = rowEncP (FlockSoundness.Discharge.Hidden.rowPre b') w)
    (i : ℕ) (hi : i < W) (hi' : i < W') : v.2 ⟨i, hi⟩ = w.2 ⟨i, hi'⟩ :=
  rowEncP_get_eq (rowPre_length b) (rowPre_length b') (rowPre_head b) (rowPre_head b') h i hi hi'

/-- **A port without segments hashes a row's bytes under its first block** (`HiddenExec.encPortH`: `rowEncP` at
`Hidden.rowPre`, `sha512/row/v1`'s or `sha512/row/v2`'s). -/
theorem encPortH_of_none {p : Flock.Port} {W : ℕ} (h : p.segs = none) :
    FlockSoundness.Discharge.HiddenExec.encPortH (W := W) p =
      rowEncP (FlockSoundness.Discharge.Hidden.rowPre p.v2) := by
  simp [FlockSoundness.Discharge.HiddenExec.encPortH, h]

/-- **One row's bytes at two ports without segments fix the bits both widths hold** (`encPortH_of_none`,
`rowEncPre_get_eq`). -/
theorem bits_of_encPortH {p p' : Flock.Port} {W W' : ℕ} {v : BitVal W} {w : BitVal W'} (hs : p.segs = none)
    (hs' : p'.segs = none)
    (h : FlockSoundness.Discharge.HiddenExec.encPortH p v = FlockSoundness.Discharge.HiddenExec.encPortH p' w)
    (b : ℕ) (hb : b < W) (hb' : b < W') : v.2 ⟨b, hb⟩ = w.2 ⟨b, hb'⟩ :=
  rowEncPre_get_eq ((congrFun (encPortH_of_none hs) v).symm.trans (h.trans (congrFun (encPortH_of_none hs') w)))
    b hb hb'

end Enc

/-! ## At the family -/

section family

open FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Recursion FlockSoundness.Model
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockSoundness.Discharge.HiddenExec (encPortH)

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- Obligation: holds where the family registers reads only at rows of ports its sessions' reads files `own` name, as
`RowsRegistered` takes them, and each session's registered check runs: that check refuses such a port when it is a bit
row (`Registered.checkPort`: `sha512/row/v2` or `/row-seg/v1`), and every segmented port is one (`HmRow.parse` gives
a port segments only with `v2`).
**No registered read is of a segmented row.** At every accepted outer proof, each session's registered reads are rows
of ports without segments. Acceptance does not give it: `hown` speaks of the drawn units' reads only, and `rd` is the
family's field. A segmented row's bytes (`Layout.encSeg`, what `sha512/row-seg/v1` hashes) need not fix its bits
across two ports or two widths, so one row's bytes (`SharedRows`) would not give one row's bits (`RowsShared`). -/
def ReadsUnsegmented : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s : Fin F.stmts.size) (u : Fin (F.Vs s).nv)
    (q : Fin (F.Vs s).c.ports.size), (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.Vs s).c.ports[q.val]!.segs = none

/-- Obligation: discharged by `Accepted`'s `¬ RegColl` (`noRegistrantCollision`), which the recursive audit prices once
(`regFinder`, `prob_regColl_le`); no finder behind the transfer's `cr/sha-512` term prices it, since each sees one
session's record.
**No two sessions' registrant records collide at one registered root and position.** At every accepted outer
proof, where two sessions' registered reads name one root and one position, neither their openings
(`TwoOpenings.pair`) nor their hm96 leaves (`hm96Pair`) are a SHA-512 collision. The pair is fixed by the history (each
record is a field of the session the family builds there), not by a session's run, so no finder of the transfer's
`cr/sha-512` terms (`ZkOuter.CR`: each session's tables, its registered values and its registered paths, each against
that session's own record) outputs it; `RegColl` is the pair at the history, two of its sessions' reads. -/
def NoRegistrantCollision : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s s' : Fin F.stmts.size)
    (u : Fin (F.Vs s).nv) (u' : Fin (F.Vs s').nv) (q : Fin (F.Vs s).c.ports.size) (q' : Fin (F.Vs s').c.ports.size),
    (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.port (.inl (u, q)) = (F.zs R ω t R₂ s').rd.port (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) →
    ¬ Collides (⟨(F.zs R ω t R₂ s).rd.port (.inl (u, q)), (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)),
        (F.zs R ω t R₂ s).rd.leaf (.inl (u, q)), (F.zs R ω t R₂ s).rd.wpath (.inl (u, q)),
        (F.zs R ω t R₂ s').rd.leaf (.inl (u', q')), (F.zs R ω t R₂ s').rd.wpath (.inl (u', q'))⟩ : TwoOpenings).pair ∧
      ¬ Collides (hm96Pair
        (encPortH (F.Vs s).c.ports[q.val]! (Decode.regVal (F.zs R ω t R₂ s) (.inl (u, q))))
        ((F.zs R ω t R₂ s).rd.ws (.inl (u, q)))
        (encPortH (F.Vs s').c.ports[q'.val]! (Decode.regVal (F.zs R ω t R₂ s') (.inl (u', q'))))
        ((F.zs R ω t R₂ s').rd.ws (.inl (u', q'))))

/-- Obligation: discharged by `NoRegistrantCollision` (`sharedRows`).
**Two sessions' registrant records at one registered root and position hold one row's bytes.** At every accepted
outer proof, where two sessions' registered reads name one root and one position, the registrants' values there have
one row's bytes, each at its port's encoding (`HiddenExec.encPortH`). It holds where the family's sessions take the
registrant's opening at a root and position from one choice, the same in every session; `sharedRows_of` gives it from
`NoRegistrantCollision` otherwise. -/
def SharedRows : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s s' : Fin F.stmts.size)
    (u : Fin (F.Vs s).nv) (u' : Fin (F.Vs s').nv) (q : Fin (F.Vs s).c.ports.size) (q' : Fin (F.Vs s').c.ports.size),
    (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.port (.inl (u, q)) = (F.zs R ω t R₂ s').rd.port (.inl (u', q')) →
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) →
    encPortH (F.Vs s).c.ports[q.val]! (Decode.regVal (F.zs R ω t R₂ s) (.inl (u, q))) =
      encPortH (F.Vs s').c.ports[q'.val]! (Decode.regVal (F.zs R ω t R₂ s') (.inl (u', q')))

/-- **`SharedRows` where no two registrant records collide** (`registrants_eq_or_collides`). -/
theorem sharedRows_of (h : F.NoRegistrantCollision dg N k Rw) : F.SharedRows dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat
  obtain ⟨h1, h2⟩ := h R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat
  rcases registrants_eq_or_collides (F.zs R ω t R₂ s).rd (F.zs R ω t R₂ s').rd hr hr' hport hat with he | hc | hc
  · exact he
  · exact absurd hc h1
  · exact absurd hc h2

/-- **`RowsShared` from one row's bytes at unsegmented reads** (`SharedRows`, `ReadsUnsegmented`,
`bits_of_encPortH`). -/
theorem rowsShared_of_sharedRows (h : F.SharedRows dg N k Rw) (hu : F.ReadsUnsegmented dg N k Rw) :
    F.RowsShared dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat b hb hb'
  exact bits_of_encPortH (hu R ω t R₂ τ o ha s u q hr) (hu R ω t R₂ τ o ha s' u' q' hr')
    (h R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat) b _ _

/-- **`RowsShared` where no two registrant records collide and no read is segmented**. -/
theorem rowsShared_of (h : F.NoRegistrantCollision dg N k Rw) (hu : F.ReadsUnsegmented dg N k Rw) :
    F.RowsShared dg N k Rw :=
  F.rowsShared_of_sharedRows dg N k Rw (F.sharedRows_of dg N k Rw h) hu

/-- **No two sessions' registrant records collide** (`NoRegistrantCollision`): both collisions are `RegColl`'s, which
`Accepted` excludes. -/
theorem noRegistrantCollision : F.NoRegistrantCollision dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' _ _
  exact ⟨fun hc => ha.2.1 ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inl hc⟩,
    fun hc => ha.2.1 ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inr hc⟩⟩

/-- **`SharedRows`** (`noRegistrantCollision`). -/
theorem sharedRows : F.SharedRows dg N k Rw := F.sharedRows_of dg N k Rw (F.noRegistrantCollision dg N k Rw)

/-- **`RowsShared` where no read is segmented** (`noRegistrantCollision`). -/
theorem rowsShared (hu : F.ReadsUnsegmented dg N k Rw) : F.RowsShared dg N k Rw :=
  F.rowsShared_of dg N k Rw (F.noRegistrantCollision dg N k Rw) hu

end VStarFamily

end family

end FlockVBridge.Family
