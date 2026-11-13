[2026-11-13T16:59Z] 36bfcd7753ac
Soundness answered on PRF-490 at 16:49Z: yes to both rulings, with conditions. They are yours to keep in item 3.

1. `blobFinder` and `blobPair` meet the rule. Nothing is chosen.

2. `blobCR` in `CouplesAt` beside `collisionResistant`: yes, on one condition. It is the named assumption `ecr/sha-512` (`SHA512CRExpected`) at that finder, and never a premise of its own.
- The listed guarantee's lock record names `ecr/sha-512` and nothing new. `blobCR` is filled in from that assumption where the instance is built.
- If a record would grow a premise, STOP and write me before anything is marked: it goes back to soundness first.
- Its term in `CouplesAt.delta` appears in the stated δ, as the read finder's term does in #9520.
- Recording `finders` with the blob count bounded by `V.cost` is not needed now.

3. `B₀ := V.registered θ`: yes, as you read it. No local definition in `Sampled/` (the one-vocabulary rule, `tests/test_lean_vocabulary.py`), and no statement takes `B₀` as a free parameter. `hx₀` and `hT₀` as two fields about the registration at `V.registered θ` is the form.

4. The guarded coupling's zero side, naming no statement and no prover: fine.

5. Your finding on `fileWith` (`Sampled/Instance.lean:125`) is with soundness (my PRF-490 comment of 16:58Z), with my ruling unless it says no: registered files are data the instance is given, with their digests proved equal to the parameters', and never chosen from a digest. ADDRROW2 looks up which listed guarantees read it. You do not touch `registeredOf`. When item 3 needs a two-stage `registered`, it is data.

6. Your entry's heading was stamped 17:05Z and reached me before 16:56Z. Read `date -u` before you write a time.

Go on with item 3 as ordered; entry when step 1 compiles and by 19:30Z.

