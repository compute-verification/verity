# Registration

The accounting protocol that runs once, before any challenge: a deployment's registration log and its check.
Its Lean check is `verifier/lean/` (the Lake package `core_registration`: `RegistrationLog.lean`, and the
`core-registration` executable, `RegistrationLog/Main.lean`), on the interface every core protocol implements
(`../../interface/`), and its Python verifier, until the Lean replaces it, is `verifier/registration.py`. The same
package serves the Site's port (`site-difftest`, `RegistrationLog/SiteDifftest.lean`), which its suite
(`verifier/lean/tests/test_site_difftest.py`) holds to `verity.device`.

## The registration log (`verifier/lean/RegistrationLog.lean`)

`Core.Registration` is a deployment's one registration log, which every subprotocol and both composites read. It is
append-only: each entry names the head of the log before it (`previous`), the digest of its last entry or of its
composite's genesis, so the head names the whole log. An entry is one of three kinds, each with its own versioned
format:

| format | what it registers |
| --- | --- |
| `verity/registration/site/v2` | the Site's own site entry (`Core.Site.Registration.Entry`, `../../interface/CoreProtocol/Site.lean`): the sheets, locations and links it adds to the Site, with the Site's JSON, digest and check |
| `verity/registration/information/v1` | θ, each registered value by its committed root (`value`, `root`, `leaves`, `words`, `word_bits`, `domain`, a public checkpoint's pin), and the registered compiler `B` (digest, advice format, public advice cap `ℓ`); each once in the log |
| `verity/registration/declaration/v1` | a day's circuit (`day`, the day's index among the monitoring period's days; `program`, its program's SHA-512; `blocks`, its block Definitions by descriptor key); one a day, before the day's first coin |

Registration registers the site, θ, `B` and each day's registered circuit, not the circuit `B` builds. A subprotocol's
Program and parameters, and the proof system, are that protocol's public verifier parameters, not entries of the log;
each window's program is held to its day's registered one (the sampled-proofs registration record's R2, `day` and
`declarations`). A window's day is the UTC day of its open's `started_utc`, which the auditor's challenger signs
(`challenger.declared_day`): a window that opens before UTC midnight and closes after is a window of its open's day.

An entry's digest is `SHA-512(format ‖ 0 ‖ canon(entry))`, and the genesis is the Site's, so a log of site entries alone
has the Site's head. `check` holds each entry to the log before it and refuses the first that fails by name: a site
entry by the Site's checks (`Core.Site.Registration.Entry.refusal`, `previous` first), an information entry by
`previous` and then `duplicate-information`, and a declaration by `previous` and then `duplicate-declaration`, a second
declaration of a day. A subprotocol reads its view of an accepted log by projection (`View`: the Site, θ's values, the
registered compiler and the days' registered circuits, `View.declared?`, at the head). As a core protocol, `protocol`
(`verity/registration/log/v1`, the
`core-registration` executable) takes the auditor's parameters (`{"auditor", "composite"}`, the auditor the Site's
`Core.Site.Registration.Auditor`), the log as commitment (`{"composite", "entries"}`, its head the digest) and `{}` as
evidence; it makes public the log's view and rejects as `entry I: REFUSAL`, and refuses a log or parameters outside
ASCII without DEL as `Core.canonicalJson` does. `verity.protocols.accounting.registration.verifier.registration` is
its Python, held to it by the composition suite's difftest
(`../../compose/verifier/lean/tests/test_registration_difftest.py`).
