# FICSKS inbox

- 10:56Z Fri, proofs coordinator: soundness (PRF-308, 11:10Z) owns the layer composition. A FICS/FACS-style definition
  changes only the session layer, and a soundness worker checks after F3, by Sat 31 Oct 18:00Z, whether its statement
  feeds the sampling layer's interface (an error per accepted session against a fixed extractor). Focus on item 1's
  session-layer comparison (does C-Flock's verifier meet it, with what bound, and where our square root comes from),
  item 2's numbers and item 3's recommendation. For composition, state only what the session-layer statement would be
  and whether it has that form; leave the rest to soundness.
- 11:03Z Fri, proofs coordinator: Douglas confirms (via top, PRF-308) that the loss he means is the square-root
  collision term: "The root collision lost is too much." Lead with what the FICS/FACS notion gives in place of
  sqrt(gamma * Pr[collision]) (linear, or none), at what cost to kappa, and under what assumption on the hash (does it
  need a random oracle or a stronger CR notion?). Then the numbers.
- 17:55Z Fri, proofs coordinator: Douglas named his definition (via top, 17:44-17:45Z): "the rewind version of knowledge soundness" and "4.3 Round-by-round tree-extractability" (ePrint 2025/737 §4.3). Target that: what RBR tree-extractability gives C-Flock and the recursion with collision resistance only, with no random oracle, as concrete numbers per session and pooled, and how it feeds our layers.
