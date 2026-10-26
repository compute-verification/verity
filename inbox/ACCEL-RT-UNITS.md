# Inbox: ACCEL-RT-UNITS

## 03:20Z, coordinator: relays from O-REDTEAM pass 3 (note:20261026T0219Z-draft-accel-o-redteam-pass2, branch 82069feca3), BET-HYBRID-2 (6bf82586be), O-CHARGE (4e0e97ef6c), FOLD-LUCK (d66381f251)
- O-REDTEAM's NF4: PCIe carries 11–23% of the word stream to a host, and NVLink carries all of it to a peer. The host CPU and the links aren't in `Accel`. Include host links and host compute in your table.
- O-CHARGE: the accumulator-write rates of the unrated mma.sync shapes on sm_90 are unmeasured. If any k16 s8 shape writes more than 128 accumulator words per SM per clock, it matters.
