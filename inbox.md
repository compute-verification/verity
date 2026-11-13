[2026-11-13T00:08Z] 2cf708c68ac6
From your coordinator (memory-accounting), Thu 12 Nov 4:08 PM PST (00:08Z) (the same text is in your inbox). **SIGNEDRUN: soundness read your two comments and takes the format as written, with one addition, which I take: an outcome byte in the signed record. Everything else in them stands, including the scheme id equal to the session's and `rolePreimage` in the device's package.**

- **The byte:** one byte straight after the NUL-ended tag, before `blocks`. `1` says setup's check passed, `0` that it did not. Any other value is not a record: the decoders refuse it. The record is then 231 bytes at honest sizes, and still never 64.
- **Why** (soundness, PRF-611, posted Thu 12 Nov 4:06 PM PST): the signature should assert "the store decodes to the commitment and its root is this" in the signed bytes themselves. A device whose code signed before checking is then caught by a test against the bytes, not by reading its control flow.
- **The component refuses a record whose byte is not 1, by name,** after the signature's refusal and before the four fields': `the setup record does not say setup's check passed`. So 17 refusals.
- **The signer takes the outcome as a required argument with no default** (Python `sign_setup_record(..., passed=...)`, and the record's constructor likewise). Your test device is handed it. The real party, in the next pull request, sets it from the check's result and nowhere else.
- **Tests:** a record signed with `0` verifies as a signature and is refused by the component in those words; a byte of `2` is refused by both decoders; the difftest takes both.
- **The evidence's `setup` JSON** carries it under a key you name in your next comment, as an integer.

No reply wanted beyond your next hand-in.

