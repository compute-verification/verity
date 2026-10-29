# COINTREE inbox

- 22:47Z, proofs coordinator: SIGNEDCOINS (bc-a42a2838) found the device signs nothing about a `--zk` session's coin
  tree; the coin server draws it from OS randomness. Decision (PRF-302): the challenger device draws each session's tree
  (key, nonce, root, spec) and logs it in its release log, which its signed close covers. The coin server takes the tree
  from the device rather than drawing it. In your tree-mode work for plain and inner sessions, have the coin server
  accept a device-drawn tree (a `--tree-from` input or the like) and keep OS drawing only behind a test flag, so
  SIGNEDCOINS' phase 2 slots in. Coordinate the interface with SIGNEDCOINS on PRF-302.
