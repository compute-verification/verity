# Registration

Before an audit can check anything, the developer says what is being audited: the hardware at the site, the
registered information θ (each registered value, by its committed root), the registered compiler `B`, and each day's
circuit, declared before that day's first coin. Registration is that record and its check, and every other protocol
reads what it accepted.

Think of it as a ledger that only grows. Each entry names the head of the log before it, so the head names the whole
log. The check reads the entries in order and refuses the first one that fails, by name.

## Try it

~~~sh
uv run tools/check/suites.py verity/protocols/registration/verifier/lean --quick
~~~

It builds the Lean check and holds the Python verifier to it on generated logs and their refusals, and the Site's check
to its Python mirror.

## What a log holds

A log holds three kinds of entry:

| format | what it registers |
| --- | --- |
| `verity/registration/site/v2` | the Site's own site entry (`Core.Site.Registration.Entry`, `../interface/CoreProtocol/Site.lean`): the sheets, locations and links it adds to the Site, with the Site's JSON and digest, and its check `verifier/lean/RegistrationLog/SiteCheck.lean`'s |
| `verity/registration/information/v1` | θ, each registered value by its committed root (`value`, `root`, `leaves`, `words`, `word_bits`, `domain`, a public checkpoint's pin), and the registered compiler `B` (digest, advice format, public advice cap `ℓ`); each once in the log |
| `verity/registration/declaration/v1` | a day's circuit (`day`, the day's index among the monitoring period's days; `program`, its program's SHA-512; `blocks`, its block Definitions by descriptor key); one a day, before the day's first coin |

A day's circuit is registered, not the circuit `B` builds from it. A protocol's own Program and parameters are that
protocol's public parameters, not entries of the log.

The check is Lean (`verifier/lean/`), with a Python verifier in `verifier/registration.py` until the Lean replaces it.
