Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## A period, as onsite's communication component (`verity/certifier/v0`)

Onsite composes the network certifier as one `Core.Protocol`, `NetworkCheck.Period` (`network/lean`, the
`network-period` executable; its formats are its module docstring). Its commitment is one-stage's registration record of
each of the period's `m` link-windows on one link and of each compute audit; its evidence is one-stage's answer on each,
and per link-window the two certificates and the clock certificate the certifier issued, that certificate alone: the
sync the window ran under and the clock's salt are witness, in neither the evidence nor the profile (the link and `m`).
The certifier, the auditor's trusted device, picks each sync from Σ_sync itself, so the check takes no sync; the
schedules term, `#Σ_sync^m`, is the same whichever syncs it picked; and N2's proof reading each sync through its opening
is a later change to its circuit (PROTOCOL.md's gap "The clock's opening"). The period's link-windows are
the session's windows `w₀ … w₀ + m − 1` (the parameter `first` is `w₀`), so one certifier session spans many periods.
It accepts when every link-window's registration names the link's Program and the provenance
`{"kind": "served", "window": w}`, both certificates and the clock certificate verify under the certifier's key in its session and name the link and `w`, the
clock certificate is egress and signs a salted digest (`clock_commitment`: neither a sync's `clock_digest` itself nor
an earlier link-window's clock digest), and one-stage accepted the window's one unit (`subset:1`),
whose verifier of record is `LinkWindows.verify` (C-Flock's Lean verifier on the `--zk` proof, then `certified()` and
the sync); and when every compute audit's answer accepted a subset draw. On acceptance its public output is `K` with its
terms, each `⌈log2⌉` of a factor of `NetTiming.Guarantees.PeriodLeakAccepted`: the schedules (`#Σ_sync^m`), the cells
(`Σ_c (cap_c + 1)`), and per audit its free inputs, its faulty sets (`Σ_{j ≤ b} C(N, j)`, with `b` one-stage's bound on
wrong units at its δ), `16 · b` and `Dj^k`. Beside `K` it publishes `eps`, the probability that `K` fails to bound the
period's leak: the union of the audits' δ, an exact `"num/den"` string as PoUS's and remote bandwidth's publics carry
theirs. The parameters name the layout terms (free inputs, horizon `k`, cell caps,
`Dj`); the composition's theorem is to tie the audits' answers to the sampled-proofs component's.
`verity/core/protocols/network/tests/test_network_period_protocol.py` holds it to its Python mirror
(`verity/core/protocols/network/tests/period_case.py`) on signed periods.
