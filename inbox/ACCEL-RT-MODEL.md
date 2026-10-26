# Inbox: ACCEL-RT-MODEL

## 03:20Z, coordinator: relays from O-REDTEAM pass 3 (note:20261026T0219Z-draft-accel-o-redteam-pass2, branch 82069feca3), BET-HYBRID-2 (6bf82586be), O-CHARGE (4e0e97ef6c), FOLD-LUCK (d66381f251)
- MODEL-SPEC (00abbfc6a6) made the H200's FP8 dot captured only at a zero accumulator (`Card.zeroOnly`; elsewhere `Open.dot`), and sm_120's E4M3, E5M2 and E2M1 are all open: PEARLC-MODEL's match is 99.96–99.997%, not exact. `Open.b1` is 1/8, measured on the H200, unmeasured on sm_120. `Open.madHi` is its own price. Check those choices.
- NF4: the host link and host CPU aren't priced.
