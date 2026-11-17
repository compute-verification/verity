module

public import Proofs.Flock.Soundness.Discharge.Hidden.Defs
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-! `Wide/Defs.lean`'s six statements without their `0 < st.c.upv` hypothesis, which `HmRow.check`'s refusal
of `units_per_vu = 0` discharges. -/

@[expose] public section

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

def OutputsCommittedTailChecked : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → tags.isTmpl = false →
    Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D : Draw.UnitSample}, Draw.UnitSample.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    (∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g j : ℕ}, g < st.c.g → g < 2 ^ st.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size),
    (pop.digest D.proved[o.val * st.c.g + g] (HmOut.nIn st.c + j) st.c.ports.size).data.toList =
      Binding.commitString
        ((HmRow.rowPrefix (st.c.ports.getD (HmOut.nIn st.c + j) default).words).data.toList ++
          outBytes (Hm.chunkBit (L.model regs) z o) st.c g j)
        (Hm.bytesOf (Hm.chunkBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))

def OutputsBindingTailChecked : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins coins' : Coins} {tables : List (String × ByteArray)} {d d' : Lean.Json}
    {partition partition' : Option Lean.Json} {program program' : Option ByteArray} {st st' : Stmt},
    tags.typed = false → tags.isTmpl = false →
    Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    Stmt.setupHidden tags cf pf coins' tables (some d') partition' program' = .ok st' →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D D' : Draw.UnitSample}, Draw.UnitSample.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    Draw.UnitSample.ofJson d' = .ok D' → (∀ u ∈ D'.proved, u < D'.population) →
    (∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ (L' : Layout st') (regs' : List (Model.Region st'.c.kLog (st'.m - st'.c.kLog))),
    (∀ r ∈ st'.regions.toList, Refine.regionOf st'.c.kLog (st'.m - st'.c.kLog) r ∈ regs') →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ {z' : Model.Witness (L'.model regs').m}, (L'.model regs').Satisfies z' →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) (o' : Fin (2 ^ (L'.model regs').nbl)) {g g' j : ℕ},
    g < st.c.g → g < 2 ^ st.c.g.log2 → g' < st'.c.g → g' < 2 ^ st'.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size) (hr' : o'.val * st'.c.g + g' < D'.proved.size),
    D.proved[o.val * st.c.g + g] = D'.proved[o'.val * st'.c.g + g'] →
    outBytes (Hm.chunkBit (L.model regs) z o) st.c g j ≠ outBytes (Hm.chunkBit (L'.model regs') z' o') st'.c g' j →
    Binding.Collides (commitPair
      ((HmRow.rowPrefix (st.c.ports.getD (HmOut.nIn st.c + j) default).words).data.toList ++
        outBytes (Hm.chunkBit (L.model regs) z o) st.c g j)
      (Hm.bytesOf (Hm.chunkBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))
      ((HmRow.rowPrefix (st'.c.ports.getD (HmOut.nIn st'.c + j) default).words).data.toList ++
        outBytes (Hm.chunkBit (L'.model regs') z' o') st'.c g' j)
      (Hm.bytesOf (Hm.chunkBit (L'.model regs') z' o') 192 (Hm.saltCol st'.c g' (HmOut.nIn st'.c + j))))

def OutputsCommittedV2Checked : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → tags.isTmpl = false →
    Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D : Draw.UnitSample}, Draw.UnitSample.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    (∀ p ∈ st.c.ports.toList, p.segs = none) →
    st.c.outRowPorts = Array.replicate st.c.outPorts.size 1 →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).words % 64 = 0) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g j : ℕ}, g < st.c.g → g < 2 ^ st.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size),
    (pop.digest D.proved[o.val * st.c.g + g] (HmOut.nIn st.c + j) st.c.ports.size).data.toList =
      Binding.commitString
        ((HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default)).data.toList ++
          outBytes (Hm.chunkBit (L.model regs) z o) st.c g j)
        (Hm.bytesOf (Hm.chunkBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))

def OutputsBindingV2Checked : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins coins' : Coins} {tables : List (String × ByteArray)} {d d' : Lean.Json}
    {partition partition' : Option Lean.Json} {program program' : Option ByteArray} {st st' : Stmt},
    tags.typed = false → tags.isTmpl = false →
    Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    Stmt.setupHidden tags cf pf coins' tables (some d') partition' program' = .ok st' →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D D' : Draw.UnitSample}, Draw.UnitSample.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    Draw.UnitSample.ofJson d' = .ok D' → (∀ u ∈ D'.proved, u < D'.population) →
    (∀ p ∈ st.c.ports.toList, p.segs = none) →
    st.c.outRowPorts = Array.replicate st.c.outPorts.size 1 →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).words % 64 = 0) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ (L' : Layout st') (regs' : List (Model.Region st'.c.kLog (st'.m - st'.c.kLog))),
    (∀ r ∈ st'.regions.toList, Refine.regionOf st'.c.kLog (st'.m - st'.c.kLog) r ∈ regs') →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ {z' : Model.Witness (L'.model regs').m}, (L'.model regs').Satisfies z' →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) (o' : Fin (2 ^ (L'.model regs').nbl)) {g g' j : ℕ},
    g < st.c.g → g < 2 ^ st.c.g.log2 → g' < st'.c.g → g' < 2 ^ st'.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size) (hr' : o'.val * st'.c.g + g' < D'.proved.size),
    D.proved[o.val * st.c.g + g] = D'.proved[o'.val * st'.c.g + g'] →
    outBytes (Hm.chunkBit (L.model regs) z o) st.c g j ≠ outBytes (Hm.chunkBit (L'.model regs') z' o') st'.c g' j →
    Binding.Collides (commitPair
      ((HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default)).data.toList ++
        outBytes (Hm.chunkBit (L.model regs) z o) st.c g j)
      (Hm.bytesOf (Hm.chunkBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))
      ((HmRow.prefixOf (st'.c.ports.getD (HmOut.nIn st'.c + j) default)).data.toList ++
        outBytes (Hm.chunkBit (L'.model regs') z' o') st'.c g' j)
      (Hm.bytesOf (Hm.chunkBit (L'.model regs') z' o') 192 (Hm.saltCol st'.c g' (HmOut.nIn st'.c + j))))

def OutputsCommittedWideChecked : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → tags.isTmpl = false →
    Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop →
    ∀ {D : Draw.UnitSample}, Draw.UnitSample.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    (∀ p ∈ st.c.ports.toList, p.segs = none) →
    st.c.outRowPorts = Array.replicate st.c.outPorts.size 1 →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).words % 64 = 0) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g j : ℕ}, g < st.c.g → g < 2 ^ st.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size),
    (pop.digest D.proved[o.val * st.c.g + g] (HmOut.nIn st.c + j) st.c.ports.size).data.toList =
      Binding.commitString
        ((HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default)).data.toList ++
          outBytes (Hm.chunkBit (L.model regs) z o) st.c g j)
        (Hm.bytesOf (Hm.chunkBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))

def OutputsBindingWideChecked : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins coins' : Coins} {tables : List (String × ByteArray)} {d d' : Lean.Json}
    {partition partition' : Option Lean.Json} {program program' : Option ByteArray} {st st' : Stmt},
    tags.typed = false → tags.isTmpl = false →
    Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    Stmt.setupHidden tags cf pf coins' tables (some d') partition' program' = .ok st' →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop →
    ∀ {D D' : Draw.UnitSample}, Draw.UnitSample.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    Draw.UnitSample.ofJson d' = .ok D' → (∀ u ∈ D'.proved, u < D'.population) →
    (∀ p ∈ st.c.ports.toList, p.segs = none) →
    st.c.outRowPorts = Array.replicate st.c.outPorts.size 1 →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).words % 64 = 0) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ (L' : Layout st') (regs' : List (Model.Region st'.c.kLog (st'.m - st'.c.kLog))),
    (∀ r ∈ st'.regions.toList, Refine.regionOf st'.c.kLog (st'.m - st'.c.kLog) r ∈ regs') →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ {z' : Model.Witness (L'.model regs').m}, (L'.model regs').Satisfies z' →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) (o' : Fin (2 ^ (L'.model regs').nbl)) {g g' j : ℕ},
    g < st.c.g → g < 2 ^ st.c.g.log2 → g' < st'.c.g → g' < 2 ^ st'.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size) (hr' : o'.val * st'.c.g + g' < D'.proved.size),
    D.proved[o.val * st.c.g + g] = D'.proved[o'.val * st'.c.g + g'] →
    outBytes (Hm.chunkBit (L.model regs) z o) st.c g j ≠ outBytes (Hm.chunkBit (L'.model regs') z' o') st'.c g' j →
    Binding.Collides (commitPair
      ((HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default)).data.toList ++
        outBytes (Hm.chunkBit (L.model regs) z o) st.c g j)
      (Hm.bytesOf (Hm.chunkBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))
      ((HmRow.prefixOf (st'.c.ports.getD (HmOut.nIn st'.c + j) default)).data.toList ++
        outBytes (Hm.chunkBit (L'.model regs') z' o') st'.c g' j)
      (Hm.bytesOf (Hm.chunkBit (L'.model regs') z' o') 192 (Hm.saltCol st'.c g' (HmOut.nIn st'.c + j))))

end FlockSoundness.Discharge.Hidden
