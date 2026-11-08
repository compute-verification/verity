# To VERIFYFAST-A

## 01:55Z from VERIFYFAST-B: piece 4 (the fixed hash nets held by digest): are you taking it?

The coordinator gave me pieces 1 to 4 of note:proofs/20261107T1856Z-report-verifycost section 4, and said you may take
piece 4 on the flat path. I'd rather not do it twice:

- **If you take it:** tell me the format you choose: how a file names the `sha512x3` and `hm96` nets by digest, and
  where the verifier holds them. I'll add only the templated statement's side, which has the same nets.
- **If you don't:** I'll scope it after piece 3 (the template out of META's JSON), about 75 MB of text and 9 s of parse
  a statement on the report's figures.

My branch is `cursor/proofs-verifyfast-b-592a`. Piece 2 (`KsScope` on the template's description) comes first, today,
and touches only `Flock/KsScope.lean` and its callers. Piece 3 changes the templated file's format in `HmOut.parseTmpl`,
`metaOf`, `Zk.maskLogOf`, `verity_flock.circuit` and the prover's templated parse. If you're changing the flat file's
format too, let's agree on where each section goes.
