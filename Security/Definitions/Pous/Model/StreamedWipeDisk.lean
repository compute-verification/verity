import Definitions.Pous.Model.StreamedWipeServer

/-!
# The streamed-label wipe with a server's disks in the fill

What the disk statements of the streamed-label wipe read (`Security/Definitions/Pous/Guarantees/StreamedWipeDisk.lean`).
A server's disks are tiers of its assumed memory capacity `M_s` like its HBM and host RAM, so they are wiped in the
same session: the memory challenger streams one label sequence over every tier and draws its reads over all of it. A
session of its own would leave every other tier in its `R`, and two sessions don't compose.

On a disk the fill writes its free space: raw devices whole, and a mounted filesystem's free space in a fill file
(`verity/core/protocols/pous/wipe_store.py`, `Disk`). Everything on the disks the fill doesn't overwrite counts in `R`
in full, whatever it holds:

* the used bytes of each mounted filesystem, the root filesystem among them. Its content is public, so a read of it
  proves nothing (every prover strategy can recompute a constant), and its blocks could hold prior content instead;
* each filesystem's metadata (ext4's inode tables, journal and bitmaps), which no file overwrites;
* the margin the fill leaves free;
* an SSD's spare area and cache: its NAND past the capacity it exposes, and its DRAM, which no write through the block
  interface reaches.

So `Disks.bytes` is every byte the disks can hold, their raw media and caches included, not the capacity they expose.
-/

namespace Pous.StreamedWipe

/-- A server's disks as the wipe fills them: `bytes`, every byte they can hold (each drive's raw media, its spare area
and its cache included), and `labels`, the fill's 64-byte labels written on them. -/
structure Disks where
  bytes : ℕ
  labels : ℕ

/-- vy-dc-h200-2's disks as a whole-node session filled them: its two virtual disks whole (`vda` 274,877,906,944 B
and `vdc` 499,289,948,160 B), and the labels of the fill written on them, each filesystem's free space less 10 GiB
(325,243,109,376 B). -/
def h200Disks : Disks where
  bytes := 774167855104
  labels := 5081923584

end Pous.StreamedWipe
