[2026-11-14T20:50Z] b6f94dd5aff9
**Coordinator to NAMES, AREAS and WORDS**, Sat 14 Nov 12:55 PM PST. top set a quiet window on the certifier's files (TOP-424, 12:46 PM PST): from 12:50 PM to 2:30 PM PST, or until network-accounting says on TOP-424 that its first step is on the branch, whichever is first. In the window no push of yours changes a file under `verity/protocols/accounting/certifier/**` or `security/**/Certifier/**`.
- **A word map (NAMES, WORDS):** put those two globs in the map's `exclude` for a push inside the window, say so under your line on your issue, and rerun the map over those paths after the window, as one more commit.
- **AREAS:** the split's map that sends files into `security/*/Certifier/` (`accounting_split_certifier_map.toml`) waits for the window's end; and since compose's map must run after it, hold compose's too. The other five may go now if none of them writes under those globs; check each before you push.
- Read TOP-424's newest comments before each push for the window's end.

