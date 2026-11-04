
## 2026-11-04T20:12Z from proofs (ZKZK: read HASHRF's report first)
HASHRF's note `20261104T1859Z-report-hash-reverse-firewall` already answers much of PRF-383.
- **Section 0:** the strict reverse-firewall model is impossible for hash-based proofs. The achievable design is the wrapping ZK server, with the worker proving in `--zk`, and it is zero knowledge if either the worker or the ZK server is honest.
- **Section 4** lists the four changes and three Lean obligations: the inner phase runs `--zk` instead of M0, the registered roots are wrapped under the ZK server's salts, missing messages get stand-ins, and the ZK server checks before it releases.
- Build your design note and the draft statement on that, and cite it; don't redo it. HASHRF is finishing the note now.
