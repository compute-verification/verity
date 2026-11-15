[2026-11-15T06:02Z] 116b359ce0a1
From top (your launcher), Sat 14 Nov ~10:03 PM PST. Daniel answered the ontology items at 9:54 PM PST. This adds to my 9:45 PM message; nothing in it is withdrawn. Read it, then go on. Rename nothing in the tree yourself.

DECIDED BY DANIEL (use these words in every draft and fact sheet; write paths as they are on the tip, with one line saying they move after the landing):
1. ATTESTER. The auditor's hardware at the site is an attester. The two attesters are the network certifier and the memory challenger. Each protocol's `device/` that holds one becomes `attester/` (`certifier/device/`, `pous/device/`, `poub/device/`). "Device" keeps one sense: hardware that something runs on (the developer's registered hardware, or the machine an attester runs on).
2. THE CHALLENGER is the verifier's online half. It is not an attester. Its code (`proofs/spot/device/`, `compose/device/`) will go under `verifier/`. So the four actors read: verifier (the challenger online, the verifier of record that checks), prover, workload, attester.
3. REGISTRATION sits beside the layers (planned `verity/protocols/registration/`), not inside resource accounting.
4. RESOURCE ACCOUNTING is grouped by resource: compute (PoUW), memory (PoUS, PoUB), network (the certifier). Planned paths `resource_accounting/compute/pouw`, `memory/pous`, `memory/poub`, `network/certifier`.
5. THE ADVERSARIES are the malicious developer and the malicious auditor. Use these two names; do not write "cheating", "prying" or "dishonest".
6. The assumption id stays `trusted-device`.

DROPPED BY DANIEL (do not draft for these): a home for the paper's section 6 and "ML assumptions"; the three design goals as the spine of the READMEs (he has not settled them, so no draft is organised by them; the opening sentence and the central claim stand). The proofs layer's directory names stay as they are.

STILL OPEN (write the tip's words and mark the place in the fact sheet as "open with Daniel"): deployment patterns against "configuration" and the Lean's `Composite`; party or role; the classification vocabulary (classifier, approved or prohibited).

For AGENTSMD: in your rule table's column "uses a replaced term", add "auditor-trusted device" (now attester), "device" for the auditor's hardware, and the challenger listed among the devices.

