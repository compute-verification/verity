import Definitions.Accounting.Isa
import Models.Accounting.Assumptions.TS

/-!
# The instruction table's premises (trusted)

A7 (`Accounting.Assumptions.TS.SheetSemantics`) and A6 (`SheetWords`) restated over a device's instruction table
(`Accounting.Isa`, note:20261019T0520Z-draft-mm-isa), so that what's left trusted is one premise per row family and
two about the class list and the labelling. Each is a `Prop` taken as a hypothesis, claimed for a cheater's labelled
run (`Anatomy`) and its heads by class (`Heads`) at §1.3's labelling, and each docstring opens with its claim id
(`verity.core.claims`) and names its document.

* `IsaComplete`: every owned right fixed word's head is an instance of a class of the table (the class list is
  complete).
* `IsaLabelled`: the labelling puts each head on a route its class's row allows, and the words on a route are the
  heads it puts there.
* `IsaRowsMma` … `IsaRowsWarp`: each family's rows match the PTX sections they cite: an instance writes at most its
  row's words, and a copy's instances head nothing.

What they give (`Models.Accounting.Guarantees.Isa`): with the table's discipline (`Accounting.Isa.Row.ok`, decided)
and a model that reads the table (`Accounting.Isa.ReadsTable`), A6 and A7.
-/

namespace Accounting.Assumptions.Isa

open Accounting.TS Accounting.Isa

variable {K R T L U F X : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U]

/-- `accounting-isa-complete` (PTX ISA 9.4's instruction index; the CUDA Binary Utilities' instruction-set lists, §4.3
Hopper and §4.4 Blackwell; the SASS opcode census, RT-ISA's corpus of 164 binaries on two nodes under CUDA 13.0,
art:d11bc611d0e9ed530b018e1d9b873d124b866a3545cde708022621d604d2ca3a; on sm_120, RT-MACHINE-2's triage of the
Blackwell opcodes A7's menu didn't name, art:fac984dca8c0141d83e5f147ba04787c02b7d59bd49edcb1ffb64727eeeb7e0c).
**The class list is complete**: every owned right fixed committed word has a head (§1.3, Definition 2), and that head
is an instance of a class of the table, which the labelling puts on a route (`on`) or on none (`stray`): the run's
fixed words are at most its heads, class by class. Every documented and every corpus opcode is placed or read as
writing no word (`isa_table.py` fails on one that isn't), so what falsifies it is a head of an opcode neither the
lists nor the corpus name, or a placement that is wrong. -/
def IsaComplete (A : Anatomy K T L U) (H : Heads K T L U) : Prop :=
  A.fixed ≤ ∑ k, (H.stray k + ∑ h, H.on k h)

/-- `accounting-isa-labelled` (§1.3's labelling). **The labelling reads the table**: the words on route `h` are the
heads it puts on `h`, it puts every head of a class the table doesn't trace on a route, and only on one the class's
row allows (`Fits`: its writer kind, `S_V` or a table route, or its side unit). A model that reads the table
(`ReadsTable`) has such a route for every class, so the labelling exists for every run; this premise is that the
anatomy's route words are counted by it. -/
def IsaLabelled (Tb : K → Row) (σ : U → K) (Z : Model K R T L U F X) (A : Anatomy K T L U) (H : Heads K T L U) :
    Prop :=
  (∀ h, A.words h = ∑ k, H.on k h) ∧ (∀ k, (Tb k).route ≠ .traced → H.stray k = 0) ∧
    ∀ k h, (Tb k).route ≠ .traced → H.on k h ≠ 0 → Fits Tb σ Z k h

/-- `accounting-isa-rows-mma` (PTX ISA 9.4 §9.7.16.5.14 mma, §9.7.16.3 block scaling, §9.7.16.6 mma.sp). **The mma.sync
rows hold**: an MMA instruction writes its accumulator fragment's words and nothing else. -/
def IsaRowsMma (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) : Prop := RowsOf .mma Tb A H

/-- `accounting-isa-rows-wgmma` (PTX ISA 9.4 §9.7.17). **The wgmma rows hold**: a warp's wgmma instruction writes its
quarter of the warpgroup's m64 × N accumulator, 16·N words (4096 at N = 256), and nothing else. -/
def IsaRowsWgmma (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) : Prop := RowsOf .wgmma Tb A H

/-- `accounting-isa-rows-tcgen05` (PTX ISA 9.4 §9.7.18.10). **The tcgen05.mma rows hold**: a tcgen05.mma writes its
tensor-memory accumulator's words (M × N) and nothing else. -/
def IsaRowsTcgen05 (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) : Prop := RowsOf .tcgen05 Tb A H

/-- `accounting-isa-rows-tmem` (PTX ISA 9.4 §9.7.18.8, §9.7.18.9). **The tensor-memory rows hold**: tcgen05.ld, .st,
.cp and .shift write their shape's words and compute nothing. -/
def IsaRowsTmem (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) : Prop := RowsOf .tmem Tb A H

/-- `accounting-isa-rows-lane` (PTX ISA 9.4 §9.7.1–§9.7.9, §9.7.20). **The lane rows hold**: a lane instruction writes
its destination registers, at most two words a lane, and a register move copies. -/
def IsaRowsLane (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) : Prop := RowsOf .lane Tb A H

/-- `accounting-isa-rows-uniform` (CUDA Binary Utilities, the instruction set's "Uniform Datapath Instructions"; no
PTX instruction names the uniform datapath). **The uniform rows hold**: a uniform instruction writes one uniform
register, or a pair, and LDCU.128 four. -/
def IsaRowsUniform (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) : Prop := RowsOf .uniform Tb A H

/-- `accounting-isa-rows-memory` (PTX ISA 9.4 §9.7.10.8 ld, §9.7.10.11 st, §9.7.10.28 cp.async, §9.7.16.5.15
ldmatrix, §9.7.16.5.16 stmatrix, §9.7.15.5 atom, §9.7.15.6 red, §9.7.15.7 red.async). **The memory rows hold**: a
load, store, atomic or reduction writes its operand's bytes, every lane, and an atomic its old value's too. -/
def IsaRowsMemory (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) : Prop := RowsOf .memory Tb A H

/-- `accounting-isa-rows-bulk` (PTX ISA 9.4 §9.7.10.28.4 bulk copy, §9.7.10.28.5 tensor copy). **The bulk rows hold**:
a bulk or tensor copy or reduction writes its box's or copy's bytes. -/
def IsaRowsBulk (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) : Prop := RowsOf .bulk Tb A H

/-- `accounting-isa-rows-texture` (PTX ISA 9.4 §9.7.12.3 tex, §9.7.12.4 tld4, §9.7.13.1 suld, §9.7.13.2 sust,
§9.7.13.3 sured; CUDA Binary Utilities' SUATOM). **The texture rows hold**: a texture or surface instruction writes its
texels' or elements' words, and a surface atomic its old value's too. -/
def IsaRowsTexture (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) : Prop := RowsOf .texture Tb A H

/-- `accounting-isa-rows-warp` (PTX ISA 9.4 §9.7.10.6 shfl.sync, §9.7.15.10 vote.sync, §9.7.15.11 match.sync,
§9.7.15.13 redux.sync). **The warp rows hold**: a shuffle, vote or match writes one word a lane, a warp reduction
one word. -/
def IsaRowsWarp (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) : Prop := RowsOf .warp Tb A H

end Accounting.Assumptions.Isa
