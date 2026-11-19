[2026-11-19T08:05Z] 56a87f55670d
FROM top, Thu 19 Nov about 12:10 AM PST. ONE MORE ITEM for your reading task, and it is now the most important one. Daniel's point: the "draw proof" is not a new thing; it is the RECURSION picture. The inner protocol runs SPOT with the verifier running the draw itself; the OUTER protocol sees only a proof that the inner verifier accepted, so the design, the partition and the sample are all witness of the outer statement.

6. Read the recursion as it is on `origin/one-circuit` and answer with file and line, READ or INFERRED: `verity/protocols/proofs/spot/verifier/lean/VStarRecursive/` (`Sampled.lean`, `SampledTwo.lean`, `SampledTwoDesc.lean` near 76 `firstSamplingDesign`), `VStarRecursion/Protocol.lean` (near 603 `checkBlobs`, `heldVerdicts`), V*'s builders in Python (search `VStar`, `vstar`, `recursive` under `verity/protocols/proofs/`), `security/Definitions/Claims/RecursionGeneric.lean`, `security/Proofs/Properties/Verity/Sampled/RecursiveRefused.lean`, `security/Definitions/Core/Rec.lean`, `Core/ZkCompiler.lean`.
   (a) What is the outer statement's PUBLIC input today: are the partition, the sampling design, the sampled indices and the number of inner proofs public to the outer verifier, or witness?
   (b) Does the inner verifier circuit (V*) compute the draw from the coins INSIDE the circuit, or does it take the sample as a public input that the outer verifier recomputes?
   (c) Where do the inner coins come from (the live challenger's ledger?) and how does the outer statement bind them?
   (d) Is the recursive form runnable today on `one-circuit` (which test or script runs it to an accept, at what size), or refused (`RecursiveRefused`)?
   (e) The smallest change that would make the design and the partition WITNESS of the outer statement with only (roots, the rates, the coins, accept) public: which files, and what is genuinely new.
Put it first in your note and in your final message. Same deadline; if you run out of time, answer 6 and drop options (A) to (C).

