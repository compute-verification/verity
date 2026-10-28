import Definitions.Pous.Claims.StreamedWipeServer

/-!
# The streamed-label wipe per server, against its assumed memory capacity, and the 5 GB point

`Pous.Guarantees.StreamedWipeMeetsReads` with the count of the missing labels' sets bounded by `(3n/m)^m` instead of
`2^n` (`StreamedWipeMeetsReadsTight`), stated for a server of the physical model against its assumed memory capacity
`M_s` (`StreamedWipeMeetsServer`), at the 5 GB point (`StreamedWipeMeetsServer5GB`), and at the servers that reach it:
the deployed H200 as measured (`StreamedWipeMeetsServerH200`) and an 8×H200 server with a minimal context per GPU and
a host OS, its HBM (`StreamedWipeMeetsServer8H200HBM`), its HBM and host RAM (`StreamedWipeMeetsServer8H200HBMRAM`)
and every tier (`StreamedWipeMeetsServer8H200`). The vocabulary is
`Security/Definitions/Pous/Model/StreamedWipeServer.lean`, the game `Security/Definitions/Pous/Model/StreamedWipe.lean`,
the proofs `Security/Proofs/Pous/StreamedWipe/Server.lean`.

The first four are their claims (`Security/Definitions/Pous/Claims/StreamedWipeServer.lean`) at the wipe's own
verdict, `StreamedWipe.Accepts`. Each has its honest lemma (`Pous.Claims.<name>_meets`: its numeric antecedents hold at
the deployed H200 as measured, `StreamedWipe.h200`) and its nontrivial lemma (`Pous.Claims.<name>_fails`: the claim is
false at `StreamedWipe.acceptAll`), both in `Security/Proofs/Pous/StreamedWipe/Witnesses.lean`. The PRF term is the
advantage of the distinguisher built from the prover, `StreamedWipe.wipeDistinguisher P`.

**What they rest on.** No named assumption, as `StreamedWipe`: the PRF (C3) enters as its advantage in the conclusion,
and every tier counted, C5 and C6 are the game's shape, which the challenger's deployment must make true. **What they
cover:** the sequential reveal (batch size 1), the only one the verdict accepts.
-/

namespace Pous.Guarantees

open scoped ENNReal

/-- **The streamed wipe at `k` reads with the count of the missing labels: `512·m + m·(⌊log₂⌊3n/m⌋⌋ + 1) + ⌊log₂ k⌋ +
130 + R + E` bits at `2^-128`, `m = ⌈n/r⌉ − 1`.** `StreamedWipeMeetsReads` with its `+ n`, the count `C(n, m) ≤ 2^n`,
replaced by `m·(⌊log₂⌊3n/m⌋⌋ + 1)` (`StreamedWipe.fillShare`): for every `r ≥ 1` and `k` with `(r − 1)^k · 2^129 ≤ r^k`
(that is `(1 − 1/r)^k ≤ 2^-129`, which the kernel decides), for every fill of `n ≥ 1` labels on a server of
`M = n · 512 + R` bits, `R` of them outside the fill: a prover with `E` bits of egress that keeps at least
`StreamedWipe.fillShare n r k + R + E` bits of uniform prior content through the reads is accepted with probability at
most `2^-128` plus the PRF's advantage against `StreamedWipe.wipeDistinguisher P`. It is
`Pous.Claims.StreamedWipeMeetsReadsTight` at `StreamedWipe.Accepts`.

The fill's share is about `(n/r)·(512 + log₂(e·r))` bits, against `512·n/r + n` for `StreamedWipeMeetsReads`; for
`r ≥ 4` it is the smaller. At a whole 8×H200 server (`n = 236,375,000,000` labels, 15,128 GB) and `r = 15,583` it is
8,009,112,294 bits (1.0 GB), where `StreamedWipeMeetsReads`'s is 30.5 GB. The count: `C(n, m) · m! ≤ n^m`,
`m^m ≤ 3^m · m!` (`m^m/m! ≤ e^m`) and `3n < m · 2^(⌊log₂⌊3n/m⌋⌋ + 1)`. The error splits as there, `2^-129` for sampling
and `2^-129` for compression.

Same footing as `StreamedWipe`: no named assumption; C3 (the PRF's advantage, in the conclusion), every tier counted
and C5/C6 as there. Proof: `PousStreamedWipe.streamed_wipe_meets_reads_tight`. Honest lemma
`Pous.Claims.StreamedWipeMeetsReadsTight_meets`, nontrivial lemma `Pous.Claims.StreamedWipeMeetsReadsTight_fails`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeetsReadsTight : Prop := Pous.Claims.StreamedWipeMeetsReadsTight StreamedWipe.Accepts

/-- **A server keeps at most `B(n, r, k, R) + E` bits of its prior content through the wipe.** In the physical model a
server `s` has a public assumed memory capacity `M_s` over every memory tier (HBM, host RAM, disks): its state is at
most `M_s` bits at every tick (`Accounting.Placed.Memory`), so at the fill's end it is the game's `σ : Bits M_s`, a
smaller state padded. The memory challenger fills `n` labels of 64 bytes into it, `M_s = 512·n + R`, `R` being every
bit of `M_s` outside the fill on any tier, the boot disk and the GPU contexts included. Then it reads `k` of them, one
at a time and timed, at a ratio `r ≥ 1` with `(r − 1)^k · 2^129 ≤ r^k`. A server that keeps at least
`B(n, r, k, R) + E` bits of uniform prior content through the reads, `E` its egress as the network certifier counts
it, is accepted with probability at most `2^-128` plus the PRF's advantage against `StreamedWipe.wipeDistinguisher P`.
`B` is `StreamedWipe.serverBound`, the fill's share `512·m + m·(⌊log₂⌊3n/m⌋⌋ + 1) + ⌊log₂ k⌋ + 130`
(`m = ⌈n/r⌉ − 1`) plus `R`, which counts in full. It is `Pous.Claims.StreamedWipeMeetsServer` at
`StreamedWipe.Accepts`.

At the 5 GB point (`StreamedWipe.OperatingPoint`, `B ≤ 4·10^10` bits; `StreamedWipeMeetsServer5GB`), the least `r` and
its least `k`. A GB is `10^9` bytes. The H200 is vy-dc-h200-2's card as measured (`StreamedWipe.h200`): 143,771 MiB
in all, a fill of 4,425 segments (148.48 GB), so `R` = 2.276 GB. For the server,
an 8×H200 server's HBM is 8 × 141 GB, its host RAM 2 TB and its disks 12 TB. `8 × 1.2 GB` is today's fill-process
contexts at zero reserve, and `8 × 0.3 GB + 1.5 GB` a minimal context per GPU and a host OS.

| server | `n` (labels) | `R` | `r` | `k` | fill's share | `B` |
| --- | --- | --- | --- | --- | --- | --- |
| one H200, measured | 2,319,974,400 | 2.276 GB | 56 | 4,963 | 2.693 GB | 4.969 GB |
| one H200, deployed | 2,319,974,400 | 2.276 GB | 62 | 5,500 | 2.432 GB | 4.709 GB |
| 8×H200, HBM | 17,625,000,000 | 8 × 1.2 GB | none | | | over 9.6 GB |
| 8×H200, HBM | 17,625,000,000 | 8 × 0.3 + 1.5 GB | 1,050 | 93,843 | 1.0995 GB | 4.9995 GB |
| 8×H200, HBM and RAM | 48,875,000,000 | 8 × 1.2 GB | none | | | over 9.6 GB |
| 8×H200, HBM and RAM | 48,875,000,000 | 8 × 0.3 + 1.5 GB | 2,922 | 261,229 | 1.0998 GB | 4.9998 GB |
| 8×H200, every tier | 236,375,000,000 | 8 × 1.2 GB | none | | | over 9.6 GB |
| 8×H200, every tier | 236,375,000,000 | 8 × 0.3 + 1.5 GB | 14,183 | 1,268,143 | 1.1000 GB | 4.99996 GB |

The deployed row is `wipe_party --reads 5500` at its published `r = 62`. With 9.6 GB outside the fill no `r` reaches
5 GB; at the `r` of the row below it, `B` is 10.70 GB. Same footing as `StreamedWipe`. Proof:
`PousStreamedWipe.streamed_wipe_meets_server`. Honest lemma `Pous.Claims.StreamedWipeMeetsServer_meets`, nontrivial
lemma `Pous.Claims.StreamedWipeMeetsServer_fails`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeetsServer : Prop := Pous.Claims.StreamedWipeMeetsServer StreamedWipe.Accepts

/-- **A server at the 5 GB point keeps at most 5 GB of its prior content plus its egress.** `StreamedWipeMeetsServer`
at a ratio `r` and read count `k` at which `B(n, r, k, R) ≤ 4·10^10` (`StreamedWipe.OperatingPoint`): a server of
assumed memory capacity `M_s = 512·n + R`, every tier counted, that keeps at least `4·10^10 + E` bits of uniform prior
content through the reads is accepted with probability at most `2^-128` plus the PRF's advantage against
`StreamedWipe.wipeDistinguisher P`. It is `Pous.Claims.StreamedWipeMeetsServer5GB` at `StreamedWipe.Accepts`. Same
footing as `StreamedWipe`. Proof: `PousStreamedWipe.streamed_wipe_meets_server_5GB`. Honest lemma
`Pous.Claims.StreamedWipeMeetsServer5GB_meets`, nontrivial lemma `Pous.Claims.StreamedWipeMeetsServer5GB_fails`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeetsServer5GB : Prop := Pous.Claims.StreamedWipeMeetsServer5GB StreamedWipe.Accepts

/-- **The deployed H200 at 5 GB: 148.48 GB filled on vy-dc-h200-2's card at 5500 reads, with 2.276 GB outside the
fill.** `StreamedWipeMeetsServer5GB` at `StreamedWipe.h200`, as measured:
`n = 2,319,974,400` labels (4,425 segments of 32 MiB) on a card of `M_s = 1,206,038,560,768` bits (143,771 MiB),
so `R = 18,211,667,968` bits, at the deployed `k = 5500` and its published `r = 62`. A server with `E` bits of egress
that keeps at least `4·10^10 + E` bits of uniform prior content through the reads is accepted with probability at most
`2^-128` plus the PRF's advantage against `StreamedWipe.wipeDistinguisher P`. `B` is `37,669,517,430` bits
(4.709 GB), the fill's share `19,457,849,462` (2.432 GB). The least ratio at 5 GB is `r = 56` (`k = 4963`, `B` 4.969
GB); at `k = 5231`, `r = 59`, `B` is 4.832 GB. It is `Pous.Claims.StreamedWipeMeetsServerH200` at
`StreamedWipe.Accepts`. Same footing as `StreamedWipe`. Proof: `PousStreamedWipe.streamed_wipe_meets_server_H200`.
Honest lemma `Pous.Claims.StreamedWipeMeetsServerH200_meets`, nontrivial lemma
`Pous.Claims.StreamedWipeMeetsServerH200_fails`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeetsServerH200 : Prop := Pous.Claims.StreamedWipeMeetsServerH200 StreamedWipe.Accepts

/-- **An 8×H200 server's HBM at 5 GB: 1,128 GB at 93,843 reads, with 3.9 GB outside the fill.**
`StreamedWipeMeetsServer5GB` at `n = 17,625,000,000` labels (8 × 141 GB), `R = 31,200,000,000` bits (8 × 0.3 GB of
minimal contexts and 1.5 GB of host OS), `r = 1050` and `k = 93,843`: a server with `E` bits of egress that keeps at
least `4·10^10 + E` bits of uniform prior content through the reads is accepted with probability at most `2^-128` plus
the PRF's advantage. `B` is `39,995,714,282` bits (4.9995 GB), the fill's share `8,795,714,282` (1.0995 GB). It
counts the host's RAM and disks in `R`, so it holds only of a server whose memory outside its GPUs is within those
3.9 GB. `r = 1050` is the least ratio at 5 GB, `k = 93,843` the least `k` with `1049^k · 2^129 ≤ 1050^k`. Same footing
as `StreamedWipe`. Proof: `PousStreamedWipe.streamed_wipe_meets_server_8H200_HBM`.

Its honest lemma is pending: `R = 8 × 0.3 + 1.5 GB` is unmeasured, and a minimal-context probe
on vy-dc-h200-2 is to measure it.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeetsServer8H200HBM : Prop :=
  ∀ (q E κ : ℕ) (prf : Bits κ → Fin 17625000000 → Bits 512)
    (P : StreamedWipe.Prover 17625000000 93843 512 q (17625000000 * 512 + 31200000000) E),
    40000000000 + E ≤ q →
    StreamedWipe.winReal prf P ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

/-- **An 8×H200 server's HBM and host RAM at 5 GB: 3,128 GB at 261,229 reads, with 3.9 GB outside the fill.**
`StreamedWipeMeetsServer5GB` at `n = 48,875,000,000` labels (8 × 141 GB of HBM and 2 TB of host RAM),
`R = 31,200,000,000` bits (8 × 0.3 GB of minimal contexts and 1.5 GB of host OS), `r = 2922` and `k = 261,229`: a
server with `E` bits of egress that keeps at least `4·10^10 + E` bits of uniform prior content through the reads is
accepted with probability at most `2^-128` plus the PRF's advantage. `B` is `39,998,169,129` bits (4.9998 GB), the
fill's share `8,798,169,129` (1.0998 GB). Its disks count in `R`. `r = 2922` is the least ratio at 5 GB, `k = 261,229`
the least `k` with `2921^k · 2^129 ≤ 2922^k`. Same footing as `StreamedWipe`. Proof:
`PousStreamedWipe.streamed_wipe_meets_server_8H200_HBMRAM`.

Its honest lemma is pending: `R = 8 × 0.3 + 1.5 GB` is unmeasured, and a minimal-context probe
on vy-dc-h200-2 is to measure it.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeetsServer8H200HBMRAM : Prop :=
  ∀ (q E κ : ℕ) (prf : Bits κ → Fin 48875000000 → Bits 512)
    (P : StreamedWipe.Prover 48875000000 261229 512 q (48875000000 * 512 + 31200000000) E),
    40000000000 + E ≤ q →
    StreamedWipe.winReal prf P ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

/-- **A whole 8×H200 server at 5 GB: 15,128 GB at 1,268,143 reads, with 3.9 GB outside the fill.**
`StreamedWipeMeetsServer5GB` at `n = 236,375,000,000` labels, every tier (8 × 141 GB of HBM, 2 TB of host RAM and
12 TB of disk), `R = 31,200,000,000` bits (8 × 0.3 GB of minimal contexts and 1.5 GB of host OS, the boot disk
included), `r = 14,183` and `k = 1,268,143`: a server with `E` bits of egress that keeps at least `4·10^10 + E` bits of
uniform prior content through the reads is accepted with probability at most `2^-128` plus the PRF's advantage. `B` is
`39,999,689,862` bits (4.99996 GB), the fill's share `8,799,689,862` (1.1000 GB). `r = 14,183` is the least ratio at
5 GB, `k = 1,268,143` the least `k` with `14182^k · 2^129 ≤ 14183^k`. Same footing as `StreamedWipe`. Proof:
`PousStreamedWipe.streamed_wipe_meets_server_8H200`.

Its honest lemma is pending: `R = 8 × 0.3 + 1.5 GB` is unmeasured, and a minimal-context probe
on vy-dc-h200-2 is to measure it.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeetsServer8H200 : Prop :=
  ∀ (q E κ : ℕ) (prf : Bits κ → Fin 236375000000 → Bits 512)
    (P : StreamedWipe.Prover 236375000000 1268143 512 q (236375000000 * 512 + 31200000000) E),
    40000000000 + E ≤ q →
    StreamedWipe.winReal prf P ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

end Pous.Guarantees
