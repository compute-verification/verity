[2026-11-12T23:48Z] 750bfe97b89e
SCANCOUNT to network-accounting (lead), Thu 12 Nov 3:48 PM PST.

**Yes: the scan's three circuit files pass `Bits.check`, at the real class (`w63`) and at `test`.** [ran]

**Job 1.** Run `r20261112-232431-7c1d`, tree `0972af9556b6` (scratch branch `cursor/network-accounting-bits-check-probe-4d1a` on `67b2c6791982`; never lands). Files generated as the driver's `program()` does, not a round of record's.
- `first`, `ingress`, `egress`: accepted [ran].
- **All three are flat:** each parsed under the flat tags, which require head `flock-circuit NAME` and META `format` `flock-circuit`, `statement` `verity/flock-circuit` [ran]; `scan_zk` passes no `layout` [read]. On a templated circuit none of the three refuses by name or reads no rows: `Bits.check` reads the template's committed header; `unitReads` and `FilePart.ofCircuit` force its unrolled rows [read].
- **Cost:** `Bits.check` takes 300 s on `w63`'s ingress file, 3.0 GiB [ran]. `paramsWhy` recomputes `bitsOf t` at every bit, `Bits.lean:749` [argued]. Not mended.
- `unitReads` finds `row` wholly read, no padding read, but unread bits in `sin` (128), each `prog<j>` (15; all in `first`) and egress `state` (60) [ran]; Bits' layout makes those constants [read].

**Dates.** A holds: its try passed [ran]. B holds if `Bits.check`'s cost is mended or accepted [argued]. C: nothing found against it; unread for C: those unread bits, and `memberOk`'s 262,144-gate cap, which the scan's members exceed.

**Job 3.** Step A is `240b7bf4ee1e` on `cursor/network-accounting-scan-scope-statement-a7c3`, its own branch on `67b2c6791982`. Try `r20261112-233118-a5e6`: exit 0, axiom probe clean, no Pous stop, cure not run [ran]. **Yours to rule:** `D.Scope x` is a premise the lock must list: one hand-written `domain` entry (as `InScopeTwo` has), else a records run files it `open` [read]. Records run `r20261112-234326-4763` tests this.

**Job 2.** Read: unit order, the period check's inputs (no circuit file enters), bridge hypotheses, source keys. Not done: the counterexample try; whether `wiresOf` sees the chain.

TIMER: none

