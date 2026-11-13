[2026-11-13T00:30Z] 08a29c751671
# WATERLINE report, NET-389: #9563 is open, unmarked; both suites pass at its head

Thu 12 Nov 4:30 PM PST.

**Head `536f2beb74f5fe8128c805c34b2125bf8eae49ef`**, `cursor/na-capture-reader-refusal-a989`.

**The README's new sentences:**

> A capture of every interface (`-i any`) is a Linux cooked one, where `tcpdump` refuses a filter that holds `vlan`, and the filter without it keeps a frame under one tag and none under two. The reader walks the one tag a cooked v1 capture holds after its header, where the header's type names it (a v2 capture holds nothing of that tag); a second tag can reach either as four bytes before the IP header that no type names, and the reader refuses a cooked frame whose type says IP where no IP header starts by that name.

No test added: the six-case cooked test already builds that frame, v1 and v2 [read].

**The suites**, each launched once, logs read by me [ran]:

- `r20261113-000518-6912`, `verity-network-traces`: exit 0, 675 passed.
- `r20261113-000518-b9b4`, `repository`: exit 0, 352 passed.

**[#9563](https://github.com/compute-verification/verity/pull/9563)**, head `536f2beb74f5`, base `main`, unmarked. NET-389 is commented (`--worker` alone: the tool refuses it with `--as`).

**Not checked:**

- A served run; another kernel.
- The four other suites the change touches.
- The repository suite's warnings on its node.
- On Ethernet by name, two tags need `vlan and vlan and (…)` [ran]. The README's clause is unchanged: yours to order.

Scratch and logs: notes, `tools/net389/`.

TIMER: none

