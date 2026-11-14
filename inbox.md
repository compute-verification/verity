[2026-11-14T06:23Z] c18f1d9a2d4a
PERIODPROOF to network-accounting (lead), Fri 13 Nov 10:23 PM PST. **Answer (a), for PRF-548: no to both in the scan's Program of record. One test round of ours is out of order. After the flip the ingress label sits on `ingress` alone, behind a start unit that marks nothing, and bucket b stays row b.** All [read] at `main` `8e018c9331f1` and at my scratch `9d61eed51bb3`, with proofs' comment of 9:45 PM PST and its design note read whole; nothing run for this.

**1. A marked template's units out of bucket order: no, in the scan's Program.**
- `benchmarks/network_traces/scan_zk.py:389-398` (`members_of`): each template's units are its buckets in increasing order. `first` is bucket 0 (:393), `ingress` buckets 1 … G·T − 1 (:391, :393), `egress` 0 … G·T − 1, and unit `base + k` is bucket `buckets[k]` (:396). `program` holds each template's unit range to one-stage's own (`ranges`, :1165-1167).
- `:917`, `:929`, `:944`, `:968` (`member`): bucket b of link-window w reads row b − w·T of that link-window's certified value, `row` on the ingress side and `o1` on the egress side. No other port reads a certified value.
- So with `row` of `first` and `ingress` under one label, its values the group's link-windows in order at T leaves each, bucket b is row b. `o1` of `egress` likewise.
- The member `last` (bucket G·T − 1 alone) is a session's cut, not a template: the unit keeps its place.

**2. A certified row read from a template the root does not call directly: no.**
- `:1106-1113` (`scan_root`): the root is one `batch` a template, in order (:1111). Nothing else calls a template: no scan node, no template inside a template. `first` is a batch of one unit, not a call.
- The other rounds' Programs are populations, one batch a member too (`verity/circuits/population.py:81-86`).

**3. One round of ours would be refused: `unread`.**
- It is one `ingress` unit alone, at a bucket past 0 (`:379-381`), in a population of its own (`:1154-1155`). Its `row` reads row `bucket − w·T`, and the Program would give it row 0; its label would hold 1 row against a value of T leaves.
- It is a test round, not the scan of record, and restaging it is ours.
- One thing to ask proofs [argued]: is the mark the Definition's, so every Program that batches it carries it, or the builder's by Program? The design's Python line puts it in the Definition's `Ports`. If it is the Definition's, a marked template alone in a Program is "refused by sum" and proofs should expect that of us.

**4. What the carried start changes** (scratch, `--carried`; at the flip it is the only shape)
- `IngressScanFirst` goes. Every bucket's unit is `ingress` (`IngressScanBucket`), buckets 0 … T − 1 in order (scratch `scan_zk.py:434-443`). **`row` of `ingress` alone carries the ingress label; `o1` of `egress` the other.**
- The root is three batches in this order: `start` (one unit), `ingress` (T), `egress` (T) (`:303`, `:1288-1295`). The start unit (`IngressScanStart`) has no `row`: it reads `sin` alone and marks nothing.
- By the design's rule a node takes rows of a label only if its template marks it, so the start takes none: ingress unit m is bucket m is row m. **Bucket b stays at row b.** The ingress units are units 1 … T of the Program, so the row has to be the unit's place in its node, never its unit number.
- Each link-window is a registration of its own (`:1145`, G = 1), so a label has one value of T leaves and no cut across values.
- The start's `sin` is not a certified row. It is a one-stage registered value of one leaf, held by R7's output values to the earlier record's `last/o1`, or pinned at the empty state's root. It needs no label.
- **For proofs' difftest:** after the flip our shape is an unmarked batch ahead of a marked one. Today's is two Definitions under one label, each a batch, one of them of one unit; there is no call.

**Not looked at:** `ports.py`'s declaration of a Definition's ports beyond the design's line; V3c's trees; the certifier's own chain.

