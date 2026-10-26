## 19:25Z from soundness: names (Daniel, 11:09 AM PDT), for all new Lean, docstrings, notes and PR text

The composed protocol (onsite) is made of **six subprotocols**: sampled proofs, PoUW, PoUS, the bandwidth audit, the
network certifier and the wipe. Use those names only, with no aliases ("integrity", "work", "space", "communication"),
and **"leg" is retired** (in Linear: project, milestone, issue). The theorem's subprotocol targets are named
`SampledProofsSound`, `PouwSound`, `PousSound`, `BandwidthAuditSound`, `NetworkCertifierSound` and `WipeSound`
(ONETHM, `cursor/proofs-one20-48ae`); `Verity.soundness_of_subprotocols` is the composition (#4613). Record keys,
wire strings (the verdict keys `integrity`, `communication`) and `Core.Protocol` keep their names.
