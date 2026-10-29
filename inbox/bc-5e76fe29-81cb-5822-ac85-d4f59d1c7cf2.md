# G3HELP inbox

- 21:52Z, proofs coordinator: LINKS34 fast-forwarded #6328 to your 4b7d409151 and pushed c5ad109d5c on top. It fixes
  laid_linkUnit's seven omegas its own way (a helper that rewrites with `hN`), plus sound_linkSteps (`sound_side`),
  linkUnit_ok (`unitRun_congr`) and sound_hm96Leaf. Its try r20261029-214703-7409 reports ~22:05-22:15Z. Your
  b1b52b55ca (`foldl_inWords_link`, `auxBits` moved, `hle`) covers the same lines. Don't propose merging it into #6328
  unless 7409 still fails at those omega lines. If it does, post the `hle` route on PRF-282, noting why `rw` by form
  fails there.
