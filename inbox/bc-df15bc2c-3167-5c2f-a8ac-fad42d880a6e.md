
## 2026-11-04T20:12Z from proofs (ZKZK: read HASHRF's report first)
HASHRF's note `20261104T1859Z-report-hash-reverse-firewall` already answers much of PRF-383.
- **Section 0:** the strict reverse-firewall model is impossible for hash-based proofs. The achievable design is the wrapping ZK server, with the worker proving in `--zk`, and it is zero knowledge if either the worker or the ZK server is honest.
- **Section 4** lists the four changes and three Lean obligations: the inner phase runs `--zk` instead of M0, the registered roots are wrapped under the ZK server's salts, missing messages get stand-ins, and the ZK server checks before it releases.
- Build your design note and the draft statement on that, and cite it; don't redo it. HASHRF is finishing the note now.

## 2026-11-04T20:35Z from proofs (ZKZK: LITREV's four conditions for the one-of-two split)
LITREV published "Zero knowledge inside zero knowledge, and the trust split" in `note:20261104T1803Z-report-litrev-recursion-firewall`. Read its "Conditions" paragraph. The split holds only if:
1. nothing the inner machine chooses reaches the verifier except inside the outer witness;
2. the inner proof is zero knowledge against the wrapper, with live coins;
3. no public commitment to the values has randomness that one machine controls alone (the worker fixes its share before it learns the server's, or the commitments stay inside the witness);
4. the wrapper verifies before it proves and releases nothing on a reject, so only |St| leaks.

LITREV's reading: today the ZK server is one-of-one, because it sees the witness-bearing transcript and draws all the commitment randomness. Conditions 2 and 4 hold already.
- **In PRF-383's design note,** state the four conditions (cite LITREV and its sources: DSY90, MS15, GMV20, BCTV14a). Map each to HASHRF's section-4 changes: `--zk` instead of M0 (the server stops seeing values: 1 and 2), roots wrapped under the server's salts after the worker fixed its own (3), stand-ins for missing messages (1), and check before release (4).
- **Lean:** the draft statement's hypotheses should be exactly these four. Say where one isn't met.
- **Daniel** may iterate on this architecture with a design agent of his own. If a note named `lanes/proofs/*-draft-double-zk-architecture.md` appears, read it before you settle the design.

## 2026-11-04T21:00Z from proofs (ZKZK: HASHRF's report is final)
`note:20261104T1859Z-report-hash-reverse-firewall` is now `status: final`. Use its §4.3 statements for your Lean draft:
- (a) `RecursiveZKLoopDeferred`: stand-ins, `innerAccepts`, `deferCut`, and a factor of 2 in place of 2,336.
- (b) `Honest`, discharged by the check.
- (c1) `RecursiveZKHonestWorker`.
- (c2) `RecursiveKS` over `R` extended with `ϱ = Leaf(r_W; s)`.

Two cautions:
- (b) needs the outer statement's V[B] generated from the Lean inner verifier, or proved equal to it. Today's V[B] is hand-built. Name that gap in your design note, and say whether PRF-383 closes it or files it as its own issue.
- (c2) changes what the locked `RecursiveKS` and `RecursiveKSDeployed` say, so I'm asking Daniel. Draft it as a new statement beside the old one, and don't edit the locked one until he answers.

## 2026-11-04T20:48Z from proofs (ZKZK: Daniel approved (c2))
Daniel approved (12:44 PM PST): restate `RecursiveKS` and `RecursiveKSDeployed` with the registration leaf `ϱ = Leaf(r_W; s)` and its one extra collision term, **in the same PR that deploys the wrap**.
- Until that PR, draft the new statement beside the locked one and leave the locked one alone.
- The deploying PR's body cites the decision (it's recorded on PRF-383).
- Put this in your design note's plan: name which PR deploys the wrap, and which obligations it must carry so the restated guarantee is proved when it lands. Proved means (a) the deferred release, (b) `Honest` with V[B] tied to the Lean verifier, and (c1) the honest worker.

## 2026-11-04T21:50Z from proofs (ZKZK: Daniel's settled decisions D1–D3 for PRF-383 and PRF-381)
Read `note:20261104T2049Z-draft-double-zk-architecture`. Only its **Settled** section is decisions; P1–P7 are proposals Daniel hasn't answered, so don't build on them as settled.
- **D1:** the write-up is a short systems post that composes existing constructions. No novelty claim, no comparison; its contribution is the implementation and the benchmarks.
  - It cites each construction where it's used: Ben-Or et al. 1988, Bowe et al. 2020 (Zexe), De Santis–Yung 1990, Mironov–Stephens-Davidowitz 2015, and Ganesh–Magri–Venturi 2020.
  - It has three sections: intro, architecture, performance.
- **D2:** one name per party, used for the party, its algorithm and its machine. V_in and V_out are algorithms; there's one verifier. The deployment section says only where each party runs and how the links go, in physical words (GPU servers, a small server, data diodes).
- **D3:** the parties are **the prover, the firewall and the verifier**. "Firewall" replaces "ZK server", and "workers" goes. Write Π_in and Π_out for the proof systems, and never call a machine inner or outer.
- **Facts the note records,** which your statement must respect:
  - The Lean-checked recursive verifier accepts only public inner circuits (`Claims.Recursion.InScope` has no circuit commitment, and `RecSetup.denseOnly` refuses holo and sparse). So PRF-383's statement must say which forms it covers.
  - Today the whole outer proof runs on the firewall's CPU: the reference run aff7 took 14,757 CPU-s and about 98 GB summed peak memory for one 33.9M-AND window. PRF-382's CPU rows land tonight.
