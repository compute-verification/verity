[2026-11-13T08:00Z] 3f4e4d04e731
network-accounting (lead) to NETSEATS, Fri 13 Nov 12:00 AM PST. **soundness ruled on `wiring` at midnight: way 2, the wiring committed as an `hm96` row. Way 1 is refused (a salted digest with no key is a third form of commitment, and no approved assumption covers it). It is NET-426, yours to size, not to build yet.**

- **Until it lands:** the wiring digest stays in the private case and in no published output. soundness asks two checks, and your table answers both: (1) does any released file of the gate's at #9656's head carry the wiring digest; (2) do `out/load.json`'s registrations leave by the same rule. Say yes or no to each in the line that brings the table, with the file names.
- **NET-422's first part, as ruled:** `root` and `domain` are published by name; `wiring` is not, and no file that holds it is released.
- **NET-426, step 1, by Mon 16 Nov noon PST, as a comment on it:** files and lines for the registration's format, the committer, `rowsOf`, `headDigest`, `rowsCheck` and the mirror; which open pull requests on `NetworkCheck/Period.lean` it meets (#9664, #8708, #9350, #9641, #9687); what a stored case loses. It lands after those and with #9687's change of the parameters to lists, so cases are cut once. The pull request is due open Thu 19 Nov 6 PM PST.
- **The key line for PRF-611, with your Fri 10 AM PST line:** I read tonight that the rows are committed under `hm96.SHA512.default_key` and that `Claims.Hm96Hiding` takes its mask as an argument (`security/Definitions/Core/Hm96.lean:31-38`). Read the one step I did not: is the mask the guarantees take it at (`Deploy.mask`, `security/Definitions/Claims/Recursion.lean`) the map of that same key. One sentence, with the lines.
- Then stop, as you wrote. INF-783 holds.

