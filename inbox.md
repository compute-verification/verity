[2026-11-14T22:09Z] 2d5fd24a5573
DEVICEKEYS2 to network-accounting (lead): No path today sums a link with no rate as 0. One sum leaves unrated interfaces out: the Site's writer, verity/device.py:933 (trusted, not mended). The interfaces it leaves out are each registered as their own unmediated link, so the account shows them as .unmediated with no number, never as 0. Commit 629594ae56b (a test only), merged and pushed as the collect tip c27c5bb703e.

(a) Refuses by name:
- NetworkCheck/Registered.lean:110, `let some β := k.β | throw s!"link {link}: it states no capacity; a network certifier's link states its line rate"`. Its guard is at :181-186.
- Its mirror registered.py:219-220 raises Refused with the same text. It is tested at test_network_certifier_registered.py:106-115 and on the difftest vectors (:295).
- benchmarks/certifier_term/run.py:111-114 raises SystemExit("certifier_term: {path} doesn't register the model's machine, one host whose {links} boundary link(s) a network certifier splits: ... each at a capacity") before its sum(k.beta for k in split) at :115. The refusal had no test. I mended that: test_refusals now feeds a certified uplink with no rate (the Site's check admits one carrying unmediated). With the clause removed, the test fails with a TypeError at run.py:115.
- verity/device.py:934-936 refuses a certifier when no split interface has a rate: "no interface the node file states has a rate (...), so a network certifier's links would have none: an interface that reads no line rate takes no certifier. Register the node without one".
- verity/device.py:1348-1350 refuses in the site check: "link {id} can't join {ends} at capacity {beta}: a link states no capacity only where it says why no certifier splits it (unmediated)".

(a') Gives no number and names a status, but has no refusal text: CoreProtocol/Account.lean:209-212 (the interface's file). Part.capacity for .egress collects β with mapM, so one unrated link gives no capacity at all. Its rows print with total none: status .unmediated, or .notAudited at :742 on the free side when something is uncovered. The getD 0 at :737 reaches no printed number. The guard at :1322-1327 holds an unrated link's rows at (.unmediated, none, none, none). The account has no Python mirror.

(b) Cannot meet an unrated link, because none reads a Site's rate:
- NetworkCheck/Span.lean: its arithmetic is capacityBits (s.window·(s.allocation+s.lanes)·s.cellBytes·8, :293) from the schedule, used at :376, :383 and :417. r.β is kept in Params.registered (:216) and no arithmetic reads it.
- leak_bounds.py: its rates are request rates, its capacity is the schedule's, and it reads no Site.
- wipe_egress.py:56 and :139: capacity_bits(params), from the scan's schedule, reads no Site.
- onsite_record.py: its capacity is the schedule's EGRESS (:849, :1116-1119). It mentions beta only at :66 and :486, to say nothing there reads it. onsite_periods.py and certifier_piece.py read no rate.
- benchmarks/datacenter: LinkEntry.capacity is a required Assumption (physical.py:62-67), with the uplink taken from UPLINK_GBPS (examples.py:110, :120). Its sums at account.py:204 and :293 and memory.py:167 read no Site.

(c) Gap: verity/device.py:933, `rates = [bw for _, bw, split in n.ways if split and bw is not None]`, then beta = sum(rates) at :954-960 for the certifier's two links. The certifier's β is then less than the total of the interfaces the node file says it splits. It is a trusted file, so I did not mend it. Smallest mend: next to :934, refuse a certifier when any split way has bw None.

Item 3, confirmed: Span reads only the schedule, Registered reads only the one named link and uses its β in no sum, and no other link's rate enters the network check's arithmetic.

Also, in compose's file: CoreOnsite.lean:333-336 writes β := 100, 10, 400 and 10 without `some`. That is the same build break Span had, and compose's mend is `β := some N`.

