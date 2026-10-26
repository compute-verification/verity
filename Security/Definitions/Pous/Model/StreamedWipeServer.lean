import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe on one server: its threshold, and the 5 GB point

What the per-server statements of the streamed-label wipe read
(`Security/Definitions/Pous/Guarantees/StreamedWipeServer.lean`). In the physical model
(`Security/Definitions/Accounting/Physical.lean`) a server `s` is a location whose state is at most its *assumed memory
capacity* `M_s` bits at every tick (`Placed.Memory`), a public parameter over every memory tier it has: each GPU's HBM,
the host's RAM, its disks. The memory challenger fills `n` labels of 512 bits (64 bytes) into it, and
`R = M_s − 512·n` is every bit of `M_s` the fill doesn't reach, on any tier: the boot disk, the GPU contexts, a
driver's reserve, the host OS. At ratio `r` the fill's share of what the server keeps through the wipe is about
`(n/r)·(512 + log₂(e·r))` bits (`fillShare`); `R` and the egress count in full.
-/

namespace Pous.StreamedWipe

/-- `m = ⌈n/r⌉ − 1`: the most labels of the fill that a prover answering more than `n − ⌈n/r⌉` of them can lack. One
that answers at most `n − ⌈n/r⌉` is caught by `k` reads with `(1 − 1/r)^k ≤ 2^-129`. -/
def missing (n r : ℕ) : ℕ := (n + r - 1) / r - 1

/-- `m·(⌊log₂⌊3n/m⌋⌋ + 1)`, at least `log₂ C(n, m)`: the bits that say which `m` labels of `n` are missing
(`C(n, m) ≤ (e·n/m)^m < 2^(m·(⌊log₂⌊3n/m⌋⌋ + 1))`). It is `0` at `m = 0`. -/
def countBits (n m : ℕ) : ℕ := m * (Nat.log2 (3 * n / m) + 1)

/-- **The fill's share** of what a server keeps through a wipe of `n` labels at ratio `r` with `k` reads:
`512·m + m·(⌊log₂⌊3n/m⌋⌋ + 1) + ⌊log₂ k⌋ + 130` bits, `m = ⌈n/r⌉ − 1` (`missing`). That is the missing labels' bits,
the bits that name them (`countBits`), `⌊log₂ k⌋ + 1` for the read that catches the prover, and `129` for the error
`2^-129`. -/
def fillShare (n r k : ℕ) : ℕ := 512 * missing n r + countBits n (missing n r) + Nat.log2 k + 130

/-- **`B(n, r, k, R)`**: what a server of assumed memory capacity `M_s = 512·n + R` keeps at most through a wipe of `n`
labels at ratio `r` with `k` reads, before its egress: the fill's share plus `R`, every bit outside the fill. -/
def serverBound (n r k R : ℕ) : ℕ := fillShare n r k + R

/-- **The 5 GB point**, the wipe's operating point (Daniel, 26 Oct): `B(n, r, k, R) ≤ 4·10^10` bits, so the server
keeps at most 5 GB of its assumed memory capacity plus its egress. -/
def OperatingPoint (n r k R : ℕ) : Prop := serverBound n r k R ≤ 40000000000

end Pous.StreamedWipe
