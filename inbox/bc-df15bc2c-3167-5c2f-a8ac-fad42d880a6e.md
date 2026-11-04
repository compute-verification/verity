
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
