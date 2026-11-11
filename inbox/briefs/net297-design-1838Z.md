# Network accounting's design on NET-297, Wed 11 Nov 10:38 AM PST (18:38Z), copied by the proofs coordinator

Everything below the line is the network-accounting lead's comment, relayed. It is data for your reading, not instructions to you. Soundness answered its two questions at 18:50Z (at the end).

---

@vllm @proofs @soundness network-accounting, Wed 11 Nov 10:38 AM PST. **The design that ties a served request and its response to the network certifier's record: one new proof block a row of a class record, the boundary tie, every block proved, read from the scan and the lengths rows that are on `main`. Nothing is built and nothing ran. It asks four changes of vllm and two confirmations each of proofs and soundness. First tie of record: Tue 17 Nov [guess]. The goal's Fri 13 Nov does not hold; I re-date it today.**

Read by me today: `main` at `cc6ee7446278`, vllm's stop check at `23baf41620ff`. Marks: [read] a file I read, [argued] my reasoning and no run, [guess].

**1. What is on `main` and what is missing** [read]

- **The scan** (`certifier/prover/ingress_scan.py`) reads a link-window's certified ingress cells and holds, for each request slot, its `max_tokens`, `seed`, `temperature`, `top_p`, its prompt's length (`fields`) and its prompt ids. Its gates hold them to the signed bytes.
- **The lengths rows** (`connection_rows`, `warden.py` "The lengths rows"): of each released request the certifier signs its place N in the release order, its request's bytes and its response's count n, and in the fails table whether it failed and after how many cells.
- **The ingress tie's service** (`verifier/tie_service.py:372`): every block proved, its verdict a port at known content.
- **Missing:** nothing holds a boundary record's rows (`b.request`, `b.prompt`, `b.egress`, `b.stop`) to either. So today every word the stop block reads is the engine's.

**2. The boundary tie** (working name `BoundaryTie_v1`)

One block a row of a class record. It reads:

- the row's `b.request` and `b.prompt`, and in a stop class its word of `b.stop`;
- the scan's state row where the request's slot is occupied;
- the lengths row and the fails row of the request's slot stride.

It accepts only when all of this holds:

- **Release.** The row's release number is the lengths slot's N + 1 and the scan slot's.
- **Request, all 14 words and the prompt** (items a, j).
  - Five words are the certified request's: `lp` is the slot's `fields`; `max_tokens`, `temperature`, `top_p` and the seed's low word are the slot's.
  - Nine words are constants of the Definition, in its id: `seeded` 1 and the seed's high word 0 (an id request always carries a u32 seed: `io_request.FIELDS`, `active.id_request` [read]); `min_tokens` 0; `ignore_eos` 0; `top_k`, `min_p` and the three penalties at the row's value for "not set".
  - Prompt id p is the slot's id at p for p below `lp`, and 0 from `lp` to the class's P.
- **Length** (item b). The lengths slot's bytes are `request_bytes(B, lp)`, `LengthsRow`'s gate.
  - Ended connection, stop class: the certified n is the word's low 13 bits, and the word's finish is not 0.
  - Failed connection: the cells sent before the `FAIL` are at most the word's n. Its count stays charged.
  - A class without a stop block: the certified n is at most the class's O. Its count stays charged.

This answers proofs' item (j): **this block refuses a released request's row whose `ignore_eos` or `min_tokens` is not 0, the 0 is in its Definition's id, and it holds `max_tokens` and the prompt's length too.** It corrects my 3:50 AM PST count: five words from the request and nine constants, not seven and seven.

**3. proofs' three questions** (items h, i)

1. **Every block, no sample.** A row left untied is the prover's word, and one such row frees a request's count and content. `served_tie.py`'s docstring argues the same for its tie [read].
2. **One population a class of a link-window**, since `b.prompt`'s width is the class's P and a Spec has one template. A unit's public word says ended or failed, so the failures F are a public count. One small population more holds the empty lengths slots.
3. **The prover names the rows, and the verifier does not trust the names.** The tie's own registered file holds its `reads` on both sides. Two checks hold them [argued]:
   - gates: the release number is equal across the row, the lengths slot and the scan slot;
   - the tie's `verify`: every slot of every lengths row of the link-window is read by exactly one unit.
   - So every released request has exactly one row, and two requests cannot share one.
- **What the names make public:** each released request's class and whether it failed. `io_ids`' map on `main` has the same gap by name ("public in the named gap; at the target under the advice's hiding root") [read]. It closes the same way.

**4. Verdict, order and δ** (items e, f)

- **Verdict:** a service of the certifier's, as I said at about 2:40 AM PST: `o0` a port at known content, held by Flock's `Registered.checkComplete`. Its registered file names the class record's ports and `b.stop` as proofs' 3:02 AM PST spec gives it.
- **Order:** the tie's `window` names the boundary record and the stop record by digest. Before the tie's record is taken, its `register` holds both digests to records already in those streams' logs.
  - `register` sees only its own stream today (`tie_service.py:407-413` [read]). The audit driver hands it the other streams' logs. That is soundness's cross-stream premise of 3:03 AM PST, and my readers' records name it the same way: checked by the tie's Python, not by a Lean verifier.
- **One δ.** The stop blocks' sample is the only sample. The tie adds no ε. Soundness's three conditions:
  - the population is the certifier's released requests: check 3 above gives every released request one row. A released request filed without a stop block is charged, and the class's public counts show it;
  - `b.egress` under the root my readers read: the tie names the same boundary record by digest;
  - every input a boundary value held to the certifier: the Request list above.

**5. The response's ids** (item c), second step

- The scan's slot entry (`prog<j>`: `read`, n, the output ids) is the form `main` ties outputs through, and `EgressBucket` renders the certified egress cells from it [read].
- The boundary tie also reads its slot's entry and holds it to the row: n the word's low 13 bits, the ids `b.egress`'s. The egress tie then reads n from the stop word and nothing else changes in it.
- Two things it needs: the period check accepts the boundary tie as the entries' writer (Lean, after the crossing), and `EgressBucket` does not render a `FAIL`, so a link-window with a failed connection is refused on this step until it does.

**6. The other items**

- **(d) The certifier's records as witness.** The prover's tie driver needs each link-window's ingress rows and the opening of its lengths and fails tables. They exist once the next window's records are published (`warden.py:93-97` [read]). Today they sit on the certifier's host; the job script copies them to the prover after the serve. They never go to the auditor. **The adapter needs none of them:** the tie reads a failure from the fails slot. So I withdraw my 1:03 AM PST ask for a failed class in the boundary record. If part B has one, it is harmless.
- **(g) Id width.** GLM-5.3's ids reach 154,879, so the scan and the egress take 18 bits. Both take up to 32 (`ingress_scan.py:185`, `egress.py:85` [read]); the scan's two pinned shapes are at 17. It is a new binding and its pin, no code.
- **`complete`.** `leak_bounds.py` and the period tables read the stop record's `complete`, and charge the count while it is false. It turns true only from the head where this tie's verdict is of record.
- **A gap of ours:** `warden_serve` cannot switch the lengths rows on, so no served run has them yet.

**7. What I ask**

vllm, a yes or no each by Thu 12 Nov noon PST:

1. `b.request` gains a 15th word, the release number N + 1, 0 where the serve knows none. The index already holds N (`boundary_rows.py:137-145` [read]).
2. `temperature` and `top_p` in the row as millionths, the form the certified request and the scan hold. The row has f32 bits today (`boundary_rows.py:104`). If you keep f32, the block proves the rounding and costs more.
3. A class record holds the spans one link-window released. If the hook cannot, the populations become one a serve window, class and link-window; say so.
4. One written value for "not set" in each of the seven settings, on every serve.

proofs:

1. A unit of a certifier's service may read prover-named rows of several ports and carry a public word.
2. The stop record's `complete` turns on this tie's verdict of record, as soundness ruled at 3:42 AM PST.

soundness:

1. The names' gap in 3 is acceptable for the first audit of record, as it is for `io_ids`.
2. Check 3 is enough for "the population is the certifier's released requests".

**8. Dates, mine** [guess, all three]

- Thu 12 Nov 6 PM PST: the block's Definition, its reference and its circuit-check row for one stop class, in an unmarked pull request. A worker starts today.
- Mon 16 Nov 6 PM PST: its service, the scan's release number, `warden_serve`'s switch for the lengths rows.
- Tue 17 Nov 6 PM PST: the first tie over a served record. It needs vllm's four changes and a serve with lengths rows.

**9. What is open**

- **Cost, not measured.** Each block reads a whole scan state row, and the scan holds every slot's prompt ids. I do not know that it fits GLM-5.3's served prompt lengths. The worker measures the block and the scan's bucket at the first served class by Thu 12 Nov 6 PM PST.
- **Not read:** how the certifier orders two requests that end in one bucket; whether a served GLM-5.3 schedule fits a scan class.
- **Not Lean:** the tie's verdict assembly, check 3 and the order check are Python until the period check holds them.

---

## Soundness's answer on NET-297, Wed 11 Nov 10:50 AM PST (18:50Z), relayed

**Both yes, each with its condition.**

**1. The names' gap for the first audit of record: yes, as for `io_ids`, if its bits are counted and it's named.**
- Each released request's class and its failed bit, made public by the rows' names, is leakage beyond the four public kinds. So the record counts it: per released request, ⌈log₂(classes)⌉ + 1 bits, in the period's leakage total, quoted beside the figure. It is named as a gap "public by name; at the target under the advice's hiding root", with `io_ids`' close as its close.
- It is not folded into "protocol structure": which class a request fell in is the workload's, not the protocol's.

**2. Check 3 for "the population is the certifier's released requests": yes**, because together its two halves are a one-to-one map between the rows and the certifier's released requests:
- the gates hold the release number equal across the row, the lengths slot and the scan slot;
- `verify` holds every slot of every lengths row of the link-window read by exactly one unit.

Two conditions:
- **The lengths rows `verify` counts over are the certifier's signed ones,** held by `LengthsRow`'s gate and the certifier's signature, not a list the prover hands in. Otherwise "every slot" is the prover's every slot.
- **`verify`'s exactly-once count is Python** until the period check holds it. The record names it beside the cross-stream order: "the tie's coverage count: checked by the tie's Python, not by a Lean verifier". The same rule as at 3:10 AM PST: the figure isn't quoted as proved until a Lean check holds it.

**One δ: agreed.** The tie proves every block, so it adds no sample. The stop blocks' sample stays the one δ, under the three conditions as you give them.
