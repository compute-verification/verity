Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## A period, as onsite's communication component (`verity/certifier/v0`)

Onsite composes the network certifier as one `Core.BlobProtocol`, `NetworkCheck.Period` (`network/lean`, the
`network-period` executable; its formats are its module docstring). Its commitment is one-stage's
registration record of each of the period's `m` link-windows on one link and of each compute audit; its evidence is
one-stage's answer on each compute audit and, per link-window, one-stage v2's evidence on its proof (its sessions'
digests and the proofs' files by their SHA-512, the files themselves the case's blobs), the salts N2's prover delivered
for the drawn unit's row commitments, and the two certificates and the clock certificate the certifier issued, that
certificate alone: the sync the window ran under and the clock's salt are witness, in neither the evidence nor the
profile (the link and `m`). A link-window that carries an answer is refused by name: the check runs one-stage v2 on its
proof itself.
The certifier, the auditor's trusted device, picks each sync from Σ_sync itself, so the check takes no sync; the
schedules term, `#Σ_sync^m`, is the same whichever syncs it picked; and N2's proof reading each sync through its opening
is a later change to its circuit (PROTOCOL.md's gap "The clock's opening"). The period's link-windows are
the session's windows `w₀ … w₀ + m − 1` (the parameter `first` is `w₀`), so one certifier session spans many periods.
It accepts when every link-window's registration names the link's Program and the provenance
`{"kind": "served", "window": w}`, both certificates and the clock certificate verify under the certifier's key in its session and name the link and `w`, the
clock certificate is egress and signs a salted digest (`clock_commitment`: neither a sync's `clock_digest` itself nor
an earlier link-window's clock digest), one-stage v2 (`SampledProofs.OneStageZk.checkBlobs`, C-Flock's Lean verifier on
each session's `--zk` proof), run by the check under the parameters' `one_stage` with the link-window's served domain as
its port and the stream's earlier registrations as its log, accepted the window's one unit (`subset:1`), and the proof's
public row opens under the delivered salt to the verifier's own on a plain link (both certificates' digests, each end's
registered root, grid and profile row) and its output to the word 3 (`LinkWindow.opened`, as N2's verdict of record
`LinkWindows.verify` makes it; a served or advised link's, whose public row reads the sync or advice, is refused there);
and when every compute audit's answer accepted a subset or stratified draw (a work law's draw is
stratified). On acceptance its public output is `K` with its
terms, each `⌈log2⌉` of a factor of `NetTiming.Guarantees.PeriodLeakAccepted`: the schedules (`#Σ_sync^m`), the cells
(`Σ_c (cap_c + 1)`), and per audit its free inputs, its faulty sets (`Σ_{j ≤ b} C(N, j)`, with `b` one-stage's bound on
wrong units at its δ), `16 · b` and `Dj^k`. Beside `K` it publishes `eps`, the probability that `K` fails to bound the
period's leak: the union of the audits' δ, an exact `"num/den"` string as PoUS's and remote bandwidth's publics carry
theirs. The parameters name the layout terms (free inputs, monitoring period `k`, cell caps,
`Dj`), the link's two ends (grid and occupancy profile), N2's run and the auditor's one-stage v2 parameters for the
link-windows; the composition's theorem is to tie the audits' answers to the sampled-proofs component's.
`verity/core/protocols/network/tests/test_network_period_protocol.py` holds it to its Python mirror
(`verity/core/protocols/network/tests/period_case.py`) on signed periods of both forms.

### The rows form: the network certifier's own run

The form above is the certifier-only path's. In the network certifier's own run (`warden_serve`, where it functions as
an active warden) there is no clock certificate: one fixed schedule (`WardenParams`) lays out every link-window's cells,
and every row registration binds its digest. The rows form's parameters name that schedule (`schedule`) in place of
Σ_sync, and the scan's layout terms (`scan`). Per link-window its commitment is the scan's one-stage registration, and
its evidence is the scan's answer, the three records' certificates (requests, ingress; responses, egress; fails, egress
on the link `<link>-fails`) and the two row registrations (`verity-network-warden/registration/v2`, ingress then
egress). A link-window passes when:

- the scan's registration names the scan's Program;
- each record's certificate verifies under the deployment key in the session and names its record's link, end and `w`;
- each registration is what `warden_rows.check_registration` reads of the link's rows at `w`: its parameters are the
 bound schedule's digest, its domain the session's, and its head is signed under the deployment key in the session;
- the scan's registration holds both rows' roots and domains in its `roots` or `registered`, so an answer for another
 window's rows is refused.

Then the scan's answer is refused by name: the check takes no answer for a link-window, and one-stage v2 can't run the
scan's proof itself (the scan's Program is a population of several templates drawn stratified, a statement each, which
v2's one circuit, one public file and one instance a session don't cover). So no rows-form period accepts until
one-stage runs the scan. Once it does, with one schedule the schedules term is 0, each link-window's scan adds its free
inputs, faulty sets, `16 · b` and `Dj^k`, as an audit does, the public lists those terms under `scans`, and `eps` is
the union of the audits' and the scans' δ.
