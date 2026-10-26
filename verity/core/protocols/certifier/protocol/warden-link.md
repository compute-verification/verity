Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## A warden link (`service.WardenProfile`)

A warden link (`LinkWindows(warden=WardenProfile(windows, digest))`) is N2 on the active warden's own two records of
each link-window (`warden.Warden`): its responses on egress and its requests on ingress, on its one grid, each
certified under its schedule's parameters digest (`warden.WardenParams.digest`, which its deployment publishes) rather
than `certificate.parameters_digest`. The unit is the plain one, `NetworkLinkWindow` v5{EGRESS, INGRESS} with no count
check on either end (COUNTS = 0): each head check holds its end's head to the warden's grid and digest and to its
signed digest. The warden is the auditor's device, and its code puts every bucket's cells (r for each live
connection, the rest padding), so its counts are its schedule's: no occupancy profile holds them, and the period's
schedule terms charge them (log2 #S_s and Σ_c log2(cap_c + 1)). The warden runs no shaper clock and signs no clock
certificate, so a window declares no sync, the verifier reads none, and the charge is 0. Its public items are
`PUBLIC_WARDEN`: the parameters digests and the schedule, and no charge or occupancy. The in-the-clear check is the
opening's root and the head checks alone: the records' cells are the scan's (`ingress_scan`, over the warden's rows),
and the certifier's host audit doesn't read a warden's records. A warden link takes no occupancy, served or advised
profile, and its two ends are one grid (`ValueError` otherwise). Its verdict of record is `python_verdict`'s: the Lean
(`NetworkCheck.LinkWindow`) has no warden form, so `verify` doesn't run it on a warden link (`NO_LEAN_FORM`).

What this establishes: on every accepted link-window of a warden link, both heads are the warden's for that
link-window under its schedule's digest, complete, and signed by the warden under its key and session. It doesn't
check the counts in them: that the cells follow the schedule is the device's code, which the scan's units hold.
