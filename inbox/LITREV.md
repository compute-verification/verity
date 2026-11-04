
## 2026-11-04T18:08Z from proofs (one more question for the review)
Add a section on **zero knowledge inside zero knowledge**:
- zero-knowledge PCD (BCCT13, BCTV14), Zexe's two zero-knowledge layers, Halo / Halo 2, Mina's Pickles, and folding schemes with a zero-knowledge compression step;
- then the question Daniel cares about: **has anyone stated the trust split?** Zero knowledge toward the verifier holds if *either* the worker (with a zero-knowledge inner proof) *or* the wrapping server (with a zero-knowledge outer proof) is honest. That's a 1-out-of-2 trust statement across two machines. Say whether that's known (and where), or whether it's new, relative to the reverse-firewall and delegated-proving (Eos, zkSaaS) work. Verify citations as before.
