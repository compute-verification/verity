import Definitions.Pous.Claims.StreamedWipeDisk

/-!
# The streamed-label wipe with a server's disks in the fill

`Pous.Guarantees.StreamedWipeMeetsServer` with the server's disks among the tiers the one session fills. The vocabulary
is `Security/Definitions/Pous/Model/StreamedWipeDisk.lean`, the game `Security/Definitions/Pous/Model/StreamedWipe.lean`,
the proof `Security/Proofs/Pous/StreamedWipe/Disk.lean`.

`StreamedWipeMeetsDisk` is its claim (`Security/Definitions/Pous/Claims/StreamedWipeDisk.lean`) at the wipe's own
verdict, `StreamedWipe.Accepts`, with its honest lemma (`Pous.Claims.StreamedWipeMeetsDisk_meets`: its numeric
antecedents hold at the session on the deployed H200's card and disks as measured) and its nontrivial lemma
(`Pous.Claims.StreamedWipeMeetsDisk_fails`: the claim is false at `StreamedWipe.acceptAll`), both in
`Security/Proofs/Pous/StreamedWipe/Disk.lean`. The PRF term is the advantage of the distinguisher built from the
prover, `StreamedWipe.wipeDistinguisher P`.

**What it rests on.** No named assumption, as `StreamedWipe`: the PRF (C3) enters as its advantage in the conclusion,
and every tier counted, C5 and C6 are the game's shape, which the challenger's deployment must make true. On a disk,
every tier counted means `D` is every byte the disks can hold, each SSD's spare area and cache included, and what the
fill doesn't overwrite (the root filesystem, filesystem metadata, the margin) counts in `R` in full. The root
filesystem needs no premise of its own: it is in `R`. **What it covers:** the sequential reveal (batch size 1), the
only one the verdict accepts.
-/

namespace Pous.Guarantees

/-- **A server keeps at most `B + E` bits of its prior content through a wipe that fills its disks too.** A server
whose tiers besides its disks hold `n` labels of the fill in `M` bits (`512·n ≤ M`: its HBM and host RAM, the memory
the fill doesn't reach counted), and whose disks hold `D` bytes of which the fill covers `f` labels of 64 bytes
(`64·f ≤ D`: their free space, written raw or as fill files), is wiped in one session of `n + f` labels on its assumed
memory capacity `M_s = M + 8·D`. Then the memory challenger reads `k` labels, one at a time and timed, at a ratio
`r ≥ 1` with `(r − 1)^k · 2^129 ≤ r^k`, drawn over all `n + f`. A server that keeps at least
`B = serverBound (n + f) r k ((M − 512·n) + (8·D − 512·f))` bits plus its egress `E` of uniform prior content through
the reads is accepted with probability at most `2^-128` plus the PRF's advantage against
`StreamedWipe.wipeDistinguisher P`. `B` is the fill's share over the whole session plus every bit outside the fill:
the other tiers' `M − 512·n` and the disks' `8·D − 512·f`, each SSD's spare area and cache, the root filesystem,
filesystem metadata and the margin among them. It is `Pous.Claims.StreamedWipeMeetsDisk`
at `StreamedWipe.Accepts`.

The disks add `8·D − 512·f` and their share of the fill, about `512·f/r` bits. A DGX H200's boot pair (two 1.92 TB
drives of 2 TiB raw NAND each) with both drives written raw leaves its spare area and cache, about 562 GB, at `k = 2^17`
reads (`r = 1466`) with a 2.69 GB share of the fill. Same footing as `StreamedWipe`. Proof:
`PousStreamedWipe.streamed_wipe_meets_disk`. Honest lemma `Pous.Claims.StreamedWipeMeetsDisk_meets`, nontrivial lemma
`Pous.Claims.StreamedWipeMeetsDisk_fails`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeetsDisk : Prop := Pous.Claims.StreamedWipeMeetsDisk StreamedWipe.Accepts

end Pous.Guarantees
