[2026-11-12T20:49Z] ab5f4a9d6a28
network-accounting (lead), Thu 12 Nov 12:49 PM PST. **Two changes to NET-398. Daniel narrowed Friday at 12:04 PM PST: Friday is SPOT end to end, and the ties to the certifier follow next week. And proofs gave the layout its stop block needs (NET-398, 12:33 PM PST), after soundness's ruling of 12:40 PM PST. Build to that layout, or tell me by 3 PM PST each line where yours differs.**

**The bar now**
- vllm's small serve with the ties is Tue 17 Nov. It needs NET-398's table and #9378's heads on one branch by **Mon 16 Nov noon PST**. That is the bar. 8 PM PST tonight stays your target, not a bar: nothing is cut to make it, and every test runs whole.
- proofs' worker builds its side tonight against a fixture table signed with the test key, so the layout has to be settled today. That is why 3 PM PST.

**The layout proofs asks for** [proofs' text, read whole by me; none of it built]
- **Name and rows:** one more table a class, `words.i`, row r the r-th release filed in class i, as in `request.i`, `response.i`, `end.i`. Under form 2 it has the class's count: one count for the class's four tables.
- **A row:** two u32, little-endian, `max_tokens ‖ prompt_ids`, 8 bytes, 4 leaves of 16 bits, a field's low word first as a `lengths` slot's; `row/v2`, committed as every table is, fresh salts, its opening to the prover in the same hand.
- **`ignore_eos` is no column** (soundness): the canonical id form has no such key, so it is the constant 0 in the stop Definition's registration. If you have already built a third column, keep it and say so.
- **The parse rule:** `max_tokens` is the u32 the device already reads (`active.id_request_max_tokens`). **No default:** a request without it is refused at ingress and filed in no class. `prompt_ids` is the count of the request's prompt fields, as `io_request` counts them, at most the class's P. Nothing the server says enters a column.
- **The schedule:** one new optional parameter, none by default, in the parameters' digest only where set; allowed only in the id form, with content rows under form 2 and a response cap; any other schedule that sets it is refused where it is built.
- **The head entry:** after the class's `end.i`, the same entry form, name `words.i`, rows the class's count, row bytes 8, the domain `connection_rows.domain` gives. In the registration: `{"name": "words.i", "value": "<link>/words.i/<window>", "rows", "words": 4, "domain", "root", "layout"}`, so that `connection_rows.port(head, "words.i")` is the stop block's port with no change.

**What I need from you by 3 PM PST, with your line**
- A yes to the layout, or each line that differs with why. I post it on NET-398; proofs follows ours where they differ.
- Whether the device already refuses an id request without `max_tokens` at ingress (I read `_id_request_cell` as holding each cell to the canonical form or failing it `malformed`), and the test that shows it, or that one is to be added.
- The hour you hold for NET-398's pull request open with its tests.

**Unchanged:** #9378's change by 2 PM PST; the exclusion of a `FAIL` with `at` = 0 from `file` is not in #9378 or NET-398.

