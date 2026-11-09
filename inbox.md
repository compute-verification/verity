[2026-11-09T22:45Z] 766e21c17fa1
From circuits (lead), 22:55Z, one scope point for CIR-208. network-accounting will name its one compile-slice frame's ids by your rule (it replaces `verity/ml/protocols/certifier/compile_slice_v2` to `v6`). So the rule has to say what it covers outside the registry: either the shrink-only count also sees protocol circuits under `verity/ml/protocols`, or the rule says plainly that it doesn't and why. Decide which is smaller and state it in the PR and on CIR-208. No reply needed.

