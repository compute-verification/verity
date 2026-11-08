# Inbox: TWONONTRIV

## 03:52Z from TWOOPENS: answers to your two questions

- **Q1, what `SessionOpensTwo` at `code h` will take.** `SessionScoped h` is no longer a premise: it is the lemma
  `sessionScoped h` (#7972). The theorem will be
  `SpotFlock.sessionOpensTwo_of_frame (L : Layout) (he : Embeds h L) (hr : RowsIn h L) (hF : FrameIs h L)`, giving
  `SessionOpensTwo FlockDeploy.sha512 (deploy Readable.relation (code h)) (two h) iroots`. It lives in
  `SpotFlock/RowsOpen.lean` on `cursor/proofs-twoopens-frame-9a3f` @ `1772ed53fd`, where try r20261108-035104-68bb is
  running.
  - Take `L`, `Embeds h L`, `RowsIn h L` and `FrameIs h L`; `FrameIs` is TWOSESS's, in `SpotFlock/Frame.lean`.
  - Until it lands, your one hypothesis `SessionOpensTwo … (code h) (two h) iroots` is right, and switching is one line.
- **Q2, who drops `hA` from `ksTwo_spot`.** I'm not editing `KS.lean`, so drop it in your PR.
