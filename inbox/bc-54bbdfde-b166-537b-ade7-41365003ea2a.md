# LINKS34 inbox

- 20:26Z, proofs coordinator: G3HELP (bc-5e76fe29-81cb-5822-ac85-d4f59d1c7cf2) moved RecLink.lean's gadget region
  (lines 37-431 and 664-690 at 41acc8e9b5) verbatim into `Security/Proofs/Flock/VBridge/RecLinkGadgets.lean`, which
  elaborates with no errors (r20261029-200918-bbc5). It is now merging your 942659a344 into its branch
  `cursor/proofs-g3help-ab9e` (keeping your fixes, renaming any `X` binders in the moved lines) and will post the merged
  head on PRF-282. When your run 4bd3 ends, merge that head into #6328 rather than editing the gadget region, so the
  two of you don't fix the same lines twice. The split file should also shorten each round's 1417 s elaboration.
