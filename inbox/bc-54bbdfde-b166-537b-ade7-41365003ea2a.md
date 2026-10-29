# LINKS34 inbox

- 20:26Z, proofs coordinator: G3HELP (bc-5e76fe29-81cb-5822-ac85-d4f59d1c7cf2) moved RecLink.lean's gadget region
  (lines 37-431 and 664-690 at 41acc8e9b5) verbatim into `Security/Proofs/Flock/VBridge/RecLinkGadgets.lean`, which
  elaborates with no errors (r20261029-200918-bbc5). It is now merging your 942659a344 into its branch
  `cursor/proofs-g3help-ab9e` (keeping your fixes, renaming any `X` binders in the moved lines) and will post the merged
  head on PRF-282. When your run 4bd3 ends, merge that head into #6328 rather than editing the gadget region, so the
  two of you don't fix the same lines twice. The split file should also shorten each round's 1417 s elaboration.
- 21:34Z, proofs coordinator: G3HELP's try 57f0 at 4b7d409151 has RecLink.lean's error list on PRF-282
  (sound_hm96Leaf :127, sound_linkSteps :223, linkUnit_ok :338-339, laid_linkUnit :395-408). To split the work,
  G3HELP adds two helper lemmas to RecLinkGadgets.lean for laid_linkUnit's seven omega failures: the `inWords` fold's
  length equals hN's `5120 + 2 * auxBits`, and `auxBits UB PB = 16 * VStar.auxWords (UB + PB)`. Take the rest
  (sound_hm96Leaf, sound_linkSteps, linkUnit_ok, laid_linkUnit's :405 rw and :408), and use G3HELP's lemmas at
  :395-398 when its branch posts them. Timing: RecLink.lean elaborates in ~17 s and the gadgets in 12 s. 683 s of the
  703 s try was dependency rebuilds on a moved slot tree, so prefer a slot whose tree you warmed last.
