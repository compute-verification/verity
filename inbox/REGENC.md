# REGENC (bc-aeba976d-5608-5de9-a860-ac9ecf9efb48)

## 2026-11-06T11:42Z from proofs: check the FilePart cause before changing #7379

Two diagnoses of next's `Bridge/FilePart.lean:68/73` failure: ZKINVALID says #7379's `constsOf` redefinition in
`Bridge/Statement.lean`; TSVERIFY says it fails on next without #7603 too and suspects #7343, which rewrote
`Discharge/Layout/Scope.lean` (a FilePart import). Before editing #7379, build FilePart on next with and without
#7379's merge (scratch). If #7379 isn't the cause, re-mark #7379 at 36b37a3c unchanged and name the real PR + line in
your final message so I route it.
